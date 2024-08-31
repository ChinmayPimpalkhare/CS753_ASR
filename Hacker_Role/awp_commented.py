# Import necessary libraries and modules
from typing import List, Optional
import torch
from torch import Tensor, nn
import numpy as np

# Import auxiliary functions from a custom module named "awp_auxiliary"
from .awp_auxiliary import TextLabels, timed_2_collapsed_text_with_map, word_levenshtein_distance

# Define a class named "AlignWithPurpose" inheriting from nn.Module
class AlignWithPurpose(nn.Module):
    # Initialize the class with default parameters
    def __init__(self,
                 num_of_paths: int = 5,
                 margin: float = 0,
                 device: Optional[str] = None,
                 blank_index: int = 28):
        super().__init__()
        # Initialize class attributes
        self.num_of_paths = num_of_paths
        self.margin = margin
        self.device = device
        self.blank_index = blank_index
        # Initialize an instance of TextLabels for label manipulation
        self.text_labels = TextLabels()

    # Define a function for shifting paths to minimize Word Error Rate (WER)
    def shift_f_minimum_wer(self, paths: List, gt_text: str, max_char_dist: int = 2):
        """
        :param paths: sampled alignments
        :param gt text: sample gt text
        :param max_char_dist: maximum levenshtein distance allowed in word to perform shift
        :return: paths with better WER (one wrong random word is corrected in each path)
        """
        # Initialize an empty list to store corrected paths
        pairs = []
        # Iterate over the number of paths
        for i in range(self.num_of_paths):
            # Extract labels for the current path
            labels = [item[i] for item in paths]
            # Convert labels to text
            labels_text = self.text_labels.labels_to_text(labels)
            # Collapse the text and create a map for alignment
            collapsed_text, text_to_index_map = timed_2_collapsed_text_with_map(labels_text)
            # Compute Levenshtein distance between collapsed text and ground truth text
            dist, aligned_sample, aligned_gt = word_levenshtein_distance(''.join(collapsed_text), gt_text,
                                                                         align_strings=True)
            # Initialize a list to store candidate indices for correction
            candidate_inds = []
            # Iterate over aligned sample and ground truth
            for j, (a, b) in enumerate(zip(aligned_sample, aligned_gt)):
                # Check for mismatch and matching lengths
                if a != b and len(a) == len(b):
                    # Compute Levenshtein distance between individual characters
                    char_dist, aligned_a, aligned_b = word_levenshtein_distance([c for c in a], [c for c in b],
                                                                                align_strings=True)
                    # Check if character distance is within the maximum allowed
                    if char_dist <= max_char_dist and '^' not in aligned_a and '^' not in aligned_b:
                        candidate_inds.append(j)
            # If no candidate indices found, add the original path
            if len(candidate_inds) == 0:
                pairs.append(labels)
            # If candidate indices found, correct a random word in the path
            else:
                random_ind = np.random.choice(candidate_inds)
                # Get the word to be corrected and its corresponding ground truth word
                a, b = aligned_sample[random_ind], aligned_gt[random_ind]
                shift_path = [l for l in labels]
                # Find the beginning index of the word in the collapsed text
                a_begin_ind = 0
                if random_ind > 0:
                    aligned_text_up_to_a = [w for w in aligned_sample[:random_ind] if w != "^"]  # ignore indels
                    a_begin_ind = len(' '.join(aligned_text_up_to_a)) + 1
                # Substitute each wrong character in 'a' with the correct character from 'b'
                for j, (aa, bb) in enumerate(zip(a, b)):
                    if aa != bb:
                        aa_map = text_to_index_map[a_begin_ind + j]
                        new_label = self.text_labels.text_to_labels(bb)[0]
                        for ind in aa_map:
                            shift_path[ind] = new_label
                pairs.append(shift_path)
        return list(map(list, zip(*pairs)))

    # Define a function for shifting paths to reduce latency
    def shift_f_low_latency(self, paths: List[List], k: int = 1):
        """
        :param paths: sampled alignments
        :param k: number of steps to shift keft the path
        :return: paths shifted k steps to the left
        """
        # Initialize an empty list to store shifted paths
        pairs = []
        # Iterate over the number of paths
        for i in range(self.num_of_paths):
            # Extract labels for the current path
            path_labels = [a[i] for a in paths]
            # Shift the path to the left by k steps
            shifted_path = self._shift_k(path_labels, k)
            pairs.append(shifted_path)
        return list(map(list, zip(*pairs)))

    # Define a function to shift a path by k steps to the left
    def _shift_k(self, path: List, k: int = 1):
        """
        :param path: sampled alignment
        :param k: number of steps to shift keft the path
        :return: path shifted k steps to the left
        """
        # Find indexes of elements to be removed for shifting
        indexes_to_remove = [i for i in range(len(path)) if
                             (path[i] == self.blank_index and i == 0) or (path[i] == path[i - 1] and i > 0)]
        # If no indexes found for removal, return the original path
        if len(indexes_to_remove) == 0:
            return path
        # Initialize a list to store items to be shifted
        shift_items = []
        # Randomly select items to shift
        for i in range(k):
            random_item = np.random.randint(0, len(indexes_to_remove) - 1)
            shift_items.append(indexes_to_remove[random_item])
            indexes_to_remove.pop(random_item)
        # Shift the path by removing selected items and adding blank indices
        return [j for i, j in enumerate(path) if i not in shift_items] + [self.blank_index] * len(shift_items)

    # Define a function to sample N paths with respect to the CTC probabilities
    def _sample_n_paths(self, softmax_ctc: Tensor):
        """
        sample n paths with respect to the ctc probabilites
        :param softmax_ctc: model output ctc after softmax
        :return: N sampled paths
        """
        # Swap axes to prepare for sampling
        ctc_swapped = softmax_ctc.squeeze(1).swapaxes(0, 1)
        # Sample paths using WeightedRandomSampler
        paths = [list(torch.utils.data.WeightedRandomSampler(line, self.num_of_paths, replacement=True)) for line in
                 ctc_swapped]
        return paths

    # Define a function to compute the probability mass of paths
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
        # Compute log probabilities of paths
        log_indexes_probs = torch.log(softmax_ctc.squeeze(2).swapaxes(1, 2).gather(2, paths) + eps)
        # Zero out probabilities beyond model prediction length
        for idx, pred_length in enumerate(model_pred_length):
            log_indexes_probs[idx, pred_length:, :] = torch.zeros(
                (log_indexes_probs.shape[1] - pred_length, self.num_of_paths))
        # Compute average probability mass of paths
        return torch.sum(log_indexes_probs, dim=1) / (model_pred_length.unsqueeze(-1))

    # Define the forward method to compute AWP loss
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
        # Ensure valid shift function is provided
        assert shift_function in [
            "f_low_latency", "f_minimum_wer"], f"You can add your own shift function to enhance a specific property"
        # Compute softmax probabilities
        softmax_ctc = nn.functional.softmax(logits, dim=1, dtype=torch.float32)
        softmax_ctc_for_paths = nn.functional.softmax(logits / softmax_temp, dim=1,
                                                      dtype=torch.float32) if softmax_temp != 1 else softmax_ctc

        # Initialize lists to store sampled and shifted paths
        sampled_paths, shifted_paths = [], []
        # Iterate over softmax probabilities and model prediction lengths
        for softmax_ctc_i, path_length in zip(softmax_ctc_for_paths, model_pred_length):
            # Sample paths based on softmax probabilities
            sampled_paths.append(self._sample_n_paths(softmax_ctc_i))

        # Apply shift function to paths
        if shift_function == "f_low_latency":
            for path in sampled_paths:
                shifted_paths_tmp = self.shift_f_low_latency(paths=path)
                shifted_paths.append(shifted_paths_tmp)
        elif shift_function == "f_minimum_wer":
            for path, gt_text in zip(sampled_paths, gt_texts):
                shifted_paths_tmp = self.shift_f_minimum_wer(paths=path, gt_text=gt_text)
                shifted_paths.append(shifted_paths_tmp)
        # You can add your own implementation here for another shift function

        # Compute probability masses of sampled and shifted paths
        sampled_paths_probs = self._paths_mass_prob(torch.tensor(sampled_paths, device=self.device), softmax_ctc,
                                                    model_pred_length)
        shifted_paths_prob = self._paths_mass_prob(torch.tensor(shifted_paths, device=self.device), softmax_ctc,
                                                   model_pred_length)
        # Compute AWP loss using hinge loss with a margin
        all_paths_subtract_pairs = sampled_paths_probs - shifted_paths_prob
        loss = torch.clamp((all_paths_subtract_pairs) + self.margin, min=0)
        # Avoid adding margin when the paths are equal
        loss = loss * torch.where(all_paths_subtract_pairs.abs() > 0, 1, 0) if self.margin > 0 else loss
        # Return the mean AWP loss
        return torch.sum(loss) / len(sampled_paths)


# Entry point for execution
if __name__ == "__main__":
    pass
