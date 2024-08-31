from typing import List, Optional

import torch
from torch import Tensor, nn
import numpy as np

from awp_auxiliary import TextLabels, timed_2_collapsed_text_with_map, word_levenshtein_distance

paths = [[1, 2, 2, 2, 10, 10], [3, 2, 2, 10, 10, 10], [4, 5, 6, 10, 10, 10], [1, 1, 1, 1, 2, 10]]
gt_text = "bck"
max_char_dist = 2

num_of_paths = 6
text_labels = TextLabels()

pairs = []
for i in range(num_of_paths):
    labels = [item[i] for item in paths]
    labels_text = text_labels.labels_to_text(labels)
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
                new_label = text_labels.text_to_labels(bb)[0]
                for ind in aa_map:
                    shift_path[ind] = new_label
        pairs.append(shift_path)

print(pairs)