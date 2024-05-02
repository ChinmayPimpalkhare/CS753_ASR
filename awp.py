from typing import List, Optional

import torch
from torch import Tensor, nn
import numpy as np

from .awp_auxiliary import TextLabels, timed_2_collapsed_text_with_map, word_levenshtein_distance


class AlignWithPurpose(nn.Module):
    def __init__(self,
                 num_of_paths: int = 5,
                 margin: float = 0,
                 device: Optional[str] = None,
                 blank_index: int = 28):
        super().__init__()
        self.num_of_paths = num_of_paths
        self.margin = margin
        self.device = device
        self.blank_index = blank_index
        self.text_labels = TextLabels()

    def shift_f_minimum_wer(self, paths: List, gt_text: str, max_char_dist: int = 2):
        """
        :param paths: sampled alignments
        :param gt text: sample gt text
        :param max_char_dist: maximum levenshtein distance allowed in word to perform shift
        :return: paths with better WER (one wrong random word is corrected in each path)
        """
        pairs = []
        for i in range(self.num_of_paths):
            labels = [item[i] for item in paths]
            labels_text = self.text_labels.labels_to_text(labels)
            collapsed_text, text_to_index_map = timed_2_collapsed_text_with_map(labels_text)
            dist, aligned_sample, aligned_gt = word_levenshtein_distance(''.join(collapsed_text), gt_text,
                                                                         align_strings=True)

            candidate_inds = []
            for j, (a, b) in enumerate(zip(aligned_sample, aligned_gt)):
                if a != b and len(a) == len(b):
                    char_dist, aligned_a, aligned_b = word_levenshtein_distance([c for c in a], [c for c in b],
                                                                                align_strings=True)
                    if char_dist <= max_char_dist and '^' not in aligned_a and '^' not in aligned_b:
                        candidate_inds.append(j)

            if len(candidate_inds) == 0:
                pairs.append(labels)
            else:
                random_ind = np.random.choice(candidate_inds)
                # a - a word with max_char_dist dist at most, b - gt word
                a, b = aligned_sample[random_ind], aligned_gt[random_ind]
                shift_path = [l for l in labels]

                # find a in the collapsed text
                a_begin_ind = 0
                if random_ind > 0:
                    aligned_text_up_to_a = [w for w in aligned_sample[:random_ind] if w != "^"]  # ignore indels
                    a_begin_ind = len(' '.join(aligned_text_up_to_a)) + 1

                # substitue each wrong char in a with the correct char from b
                for j, (aa, bb) in enumerate(zip(a, b)):
                    if aa != bb:
                        aa_map = text_to_index_map[a_begin_ind + j]
                        new_label = self.text_labels.text_to_labels(bb)[0]
                        for ind in aa_map:
                            shift_path[ind] = new_label
                pairs.append(shift_path)

        return list(map(list, zip(*pairs)))

    def shift_f_low_latency(self, paths: List[List], k: int = 1):
        """
        :param paths: sampled alignments
        :param k: number of steps to shift keft the path
        :return: paths shifted k steps to the left
        """
        pairs = []
        for i in range(self.num_of_paths):
            path_labels = [a[i] for a in paths]
            shifted_path = self._shift_k(path_labels, k)
            pairs.append(shifted_path)
        return list(map(list, zip(*pairs)))

    def _shift_k(self, path: List, k: int = 1):
        """
        :param path: sampled alignment
        :param k: number of steps to shift keft the path
        :return: path shifted k steps to the left
        """
        indexes_to_remove = [i for i in range(len(path)) if
                             (path[i] == self.blank_index and i == 0) or (path[i] == path[i - 1] and i > 0)]
        if len(indexes_to_remove) == 0:
            return path
        shift_items = []
        for i in range(k):
            random_item = np.random.randint(0, len(indexes_to_remove) - 1)
            shift_items.append(indexes_to_remove[random_item])
            indexes_to_remove.pop(random_item)
        return [j for i, j in enumerate(path) if i not in shift_items] + [self.blank_index] * len(shift_items)

    def _sample_n_paths(self, softmax_ctc: Tensor):
        """
        sample n paths with respect to the ctc probabilites
        :param softmax_ctc: model output ctc after softmax
        :return: N sampled paths
        """
        ctc_swapped = softmax_ctc.squeeze(1).swapaxes(0, 1)
        paths = [list(torch.utils.data.WeightedRandomSampler(line, self.num_of_paths, replacement=True)) for line in
                 ctc_swapped]
        return paths

    def _paths_mass_prob(self,
                         paths: Tensor,
                         softmax_ctc: Tensor,
                         model_pred_length: Tensor,
                         eps: float = 1e-7):
        """
        compute the path probability mass
        :param paths: ctc alignments
        :param softmax_ctc: model logits after softmax
        :param model_pred_length:  max length of all given paths
        :return: avg of the paths probability
        """
        log_indexes_probs = torch.log(softmax_ctc.squeeze(2).swapaxes(1, 2).gather(2, paths) + eps)
        for idx, pred_length in enumerate(model_pred_length):
            log_indexes_probs[idx, pred_length:, :] = torch.zeros(
                (log_indexes_probs.shape[1] - pred_length, self.num_of_paths))
        return torch.sum(log_indexes_probs, dim=1) / (model_pred_length.unsqueeze(-1))

    def forward(self,
                logits: Tensor,
                shift_function: str,
                model_pred_length: Tensor,
                gt_texts: Optional[List[str]] = None,
                softmax_temp: float = 1.) -> Tensor:
        """
        Calculate AWP loss, a Hinge loss between the mass probability of N sampled paths
        and their respective counterparts after applying a shift that enhence specific property.
        :param logits: logits dim is B x V x 1 x T
        :param shift_function: the shift function designed to enhence specific property
        :param model_pred_length: target alignment maximum length, determined by the input length
        :return: Tensor, the computed AWP loss
        """
        assert shift_function in [
            "f_low_latency", "f_minimum_wer"], f"You can add your own shift function to enhance a specific property"
        softmax_ctc = nn.functional.softmax(logits, dim=1, dtype=torch.float32)
        softmax_ctc_for_paths = nn.functional.softmax(logits / softmax_temp, dim=1,
                                                      dtype=torch.float32) if softmax_temp != 1 else softmax_ctc

        sampled_paths, shifted_paths = [], []
        for softmax_ctc_i, path_length in zip(softmax_ctc_for_paths, model_pred_length):
            sampled_paths.append(self._sample_n_paths(softmax_ctc_i))

        if shift_function == "f_low_latency":
            for path in sampled_paths:
                shifted_paths_tmp = self.shift_f_low_latency(paths=path)
                shifted_paths.append(shifted_paths_tmp)
        elif shift_function == "f_minimum_wer":
            for path, gt_text in zip(sampled_paths, gt_texts):
                shifted_paths_tmp = self.shift_f_minimum_wer(paths=path, gt_text=gt_text)
                shifted_paths.append(shifted_paths_tmp)
        # You can add your own implementation here for another shift function

        sampled_paths_probs = self._paths_mass_prob(torch.tensor(sampled_paths, device=self.device), softmax_ctc,
                                                    model_pred_length)
        shifted_paths_prob = self._paths_mass_prob(torch.tensor(shifted_paths, device=self.device), softmax_ctc,
                                                   model_pred_length)
        all_paths_subtract_pairs = sampled_paths_probs - shifted_paths_prob
        loss = torch.clamp((all_paths_subtract_pairs) + self.margin, min=0)
        # avoid adding margin when the paths are equal
        loss = loss * torch.where(all_paths_subtract_pairs.abs() > 0, 1, 0) if self.margin > 0 else loss
        return torch.sum(loss) / len(sampled_paths)


if __name__ == "__main__":
    pass
