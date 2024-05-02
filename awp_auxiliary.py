from typing import Dict, Optional, List, Union

import edlib
import numpy as np

class TextLabels:
    def __init__(self,
                 blank_char: str = '^',
                 labels=None):
        if labels is not None:
            self._labels = labels
        else:
            self._labels = 'abcdefghijklmnopqrstuvwxyz\' '
        self.label_count = len(self._labels) - (1 if blank_char in self._labels else 0)
        self.inds = range(self.label_count)
        self.blank_char = blank_char

        self.label_mapping = dict(zip(self._labels, self.inds))
        self.label_mapping_inv = dict(zip(self.inds, self._labels))

    def text_to_labels(self,
                       txt: str
                       ) -> List[int]:
        return [self.label_mapping[c] if c in self.label_mapping else len(self.inds) for c in txt]

    def labels_to_text(self, label_list) -> str:
        return u''.join([self.label_mapping_inv[l] if 0 <= l < self.label_count else
                         self.blank_char for l in label_list])


def timed_2_collapsed_text_with_map(timed_pred_text: Union[str, np.ndarray],
                           blank_label: Union[str, int] = '^',
                           space_label: Union[str, int] = ' '):
    if isinstance(timed_pred_text, np.ndarray):
        assert isinstance(blank_label, int) and isinstance(space_label, int)
    else:
        assert isinstance(blank_label, str) and isinstance(space_label, str)

    labels_out = []
    previous_label = -1

    final_labels_to_ctc_out_map = []
    for ind, l in enumerate(timed_pred_text):
        if l != previous_label and l != blank_label:
            if len(labels_out) > 0 and l == space_label and labels_out[-1] == space_label:
                final_labels_to_ctc_out_map[-1] = np.append(final_labels_to_ctc_out_map[-1], ind)
            else:
                labels_out.append(l)
                final_labels_to_ctc_out_map.append(np.array([ind]))

        elif l == previous_label and l != blank_label:
            final_labels_to_ctc_out_map[-1] = np.append(final_labels_to_ctc_out_map[-1], ind)

        previous_label = l

    return np.array(labels_out), final_labels_to_ctc_out_map


def levenshtein_distance(str1: str,
                         str2: str,
                         indle_char: str = '^',
                         align_strings: bool = True):
    assert isinstance(str1, str)
    assert isinstance(str2, str)
    assert isinstance(indle_char, str)
    if len(str1) == 0 or len(str2) == 0:
        dist = max(len(str1), len(str2))
        if align_strings:
            str1 = str1 if len(str1) else indle_char * len(str2)
            str2 = str2 if len(str2) else indle_char * len(str1)
            return dist, str1, str2
        return dist

    assert len(set(str1 + str2)) <= 128, f"strings length <= 128 -> edlib's requirement"
    align_res = edlib.align(query=str1, target=str2, task='path')
    dist = align_res['editDistance']

    if align_strings:
        # edlib.getNiceAlignment does not work well for empty strings
        if '' in [str1, str2]:
            return dist, str1 if str1 else indle_char * len(str2), str2 if str2 else indle_char * len(str1)
        try:
            align_dict = edlib.getNiceAlignment(align_res, str1, str2, gapSymbol=indle_char)
        except Exception as e:
            print(f"Exception in edlib_distance: {e}\nstr1: {str1}, str2: {str2}, align_res: {align_res}")
            raise e
        return dist, align_dict['query_aligned'], align_dict['target_aligned']
    else:
        return dist

def word_levenshtein_distance(list1: Union[List[str], str],
                              list2: Union[List[str], str],
                              indle_char: str = '^',
                              package: str = 'edlib',
                              align_strings=False):
    """
    compute levenshtein distance on the word level. This is done by converting words to unique chars and running on
    regular char level levenshtein distance. If aligned strings are returned, the chars representation of the provided
    words lists would be returned. if string would be provided instead of list, the string would be split by spaces.
    :param list1: lists of words for comparison with the other one
    :param list2: lists of words for comparison with the other one
    :param indle_char: char representing difference in aligned strings. Relevant only if align_strings is True
    :return: word level levenshtein distance and aligned words lists as strings if align_strings is True
    """
    if isinstance(list1, str):
        list1 = list1.split()
    if isinstance(list2, str):
        list2 = list2.split()
    words_set = set(list1 + list2)
    unicode_start = 161  # unicode decimal point to start getting characters from, before that there are control points
    word_to_char = {word: chr(word_ind) for word_ind, word in enumerate(words_set, start=unicode_start)}
    str1 = ''.join(word_to_char[w] for w in list1)
    str2 = ''.join(word_to_char[w] for w in list2)
    lev_dist_kwargs = dict(indle_char=indle_char, align_strings=align_strings)
    if align_strings:
        char_to_word = {v: k for (k, v) in word_to_char.items()}
        char_to_word[indle_char] = indle_char
        dist, aligned1, aligned2 = levenshtein_distance(str1, str2, **lev_dist_kwargs)
        lw1 = [char_to_word[ch] for ch in aligned1]
        lw2 = [char_to_word[ch] for ch in aligned2]
        return dist, lw1, lw2
    else:
        return levenshtein_distance(str1, str2, **lev_dist_kwargs)


