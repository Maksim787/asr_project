from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def _correct_sentence(self, text: str) -> str:
        # Remove double spaces
        text = text.strip()
        while '  ' in text:
            text = text.replace('  ', ' ')
        # TODO: correct mistakes
        return text

    def ctc_decode_enhanced(self, inds: list[int]) -> str:
        return self._correct_sentence(self.ctc_decode(inds))

    def ctc_decode(self, inds: list[int]) -> str:
        letters = [self.ind2char[ind] for ind in inds]

        # Store parsed letters and last letter
        result = []
        last_letter = self.EMPTY_TOK

        for letter in letters:
            if letter == last_letter:
                # Skip the same letter
                continue
            # Update last_letter
            last_letter = letter
            if letter == self.EMPTY_TOK:
                # Skip EMPTY_TOK
                continue
            # This letter is not the same as previous and it is not EMPTY_TOK
            result.append(letter)
        return ''.join(result)

    @staticmethod
    def _get_best_prefixes(state: dict[tuple[str, str], float], beam_size: int) -> dict[tuple[str, str], float]:
        # Calculate the probability of each prefix
        prefix_total_prob = defaultdict(float)
        for (pref, last_char), pref_prob in state.items():
            prefix_total_prob[pref] += pref_prob
        # Take only the best prefixes
        return dict(sorted(prefix_total_prob.items(), key=lambda kv: kv[1], reverse=True)[:beam_size])

    @staticmethod
    def _truncate_state_to_best_prefixes(state: dict[tuple[str, str], float], best_prefixes: dict[str, float]) -> dict[tuple[str, str], float]:
        return {(pref, last_char): pref_prob for (pref, last_char), pref_prob in state.items() if pref in best_prefixes}

    def _extend_and_merge(self, probs_for_time_t: torch.Tensor, state: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
        new_state = defaultdict(float)
        # Iterate over next possible characters
        for next_char_ind, next_char_prob in enumerate(probs_for_time_t.tolist()):
            # Iterate over last prefixes
            for (pref, last_char), pref_prob in state.items():
                next_char = self.ind2char[next_char_ind]
                # Find new prefix
                if next_char == last_char or next_char == self.EMPTY_TOK:
                    new_pref = pref
                else:
                    new_pref = pref + next_char
                # Add probability to prefix
                new_state[(new_pref, next_char)] += pref_prob * next_char_prob
        return new_state

    def ctc_beam_search(self, log_probs: torch.Tensor, log_probs_length: int, beam_size: int) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        log_probs = log_probs[:log_probs_length]
        char_length, voc_size = log_probs.shape
        assert char_length == log_probs_length
        probs = torch.exp(log_probs)

        assert voc_size == len(self.ind2char)
        # (prefix, last_token) -> log(prob)
        state: dict[tuple[str, str], float] = {('', self.EMPTY_TOK): 1.0}
        best_prefixes: dict[str, float] = {'': 1.0}
        for probs_for_time_t in probs:
            state = self._truncate_state_to_best_prefixes(state, best_prefixes)
            state = self._extend_and_merge(probs_for_time_t, state)
            best_prefixes = self._get_best_prefixes(state, beam_size)
        hypos = [Hypothesis(self._correct_sentence(prefix), prob) for prefix, prob in best_prefixes.items()]
        return sorted(hypos, key=lambda x: x.prob, reverse=True)
