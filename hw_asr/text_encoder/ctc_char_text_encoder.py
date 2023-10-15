from typing import List, NamedTuple
from collections import defaultdict

import torch

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    log_prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def ctc_decode(self, inds: List[int]) -> str:
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
    def _truncate(state: dict[tuple[str, str], float], beam_size: int) -> dict[tuple[str, str], float]:
        return dict(sorted(state.items(), key=lambda kv: kv[1])[-beam_size:])

    def _extend_and_merge(self, frame: torch.Tensor, state: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
        new_state = defaultdict(float)
        # Iterate over next possible characters
        for next_char_ind, next_char_log_prob in frame:
            # Iterate over last prefixes
            for (pref, last_char), pref_log_prob in state.items():
                next_char = self.ind2char[next_char_ind]
                # Find new prefix
                if next_char == last_char or next_char == self.EMPTY_TOK:
                    new_pref = pref
                else:
                    new_pref = pref + last_char
                # Add probability to prefix
                new_state[(new_pref, last_char)] += pref_log_prob + next_char_log_prob
        return new_state

    def ctc_beam_search(self, log_probs: torch.Tensor, log_probs_length,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(log_probs.shape) == 2
        char_length, voc_size = log_probs.shape
        assert voc_size == len(self.ind2char)

        # (prefix, last_token) -> log(prob)
        state: dict[tuple[str, str], float] = {('', self.EMPTY_TOK): 0.0}  # log(1) = 0
        for frame in log_probs:
            state = self._extend_and_merge(frame, state)
            state = self._truncate(state, beam_size)

        hypos = [Hypothesis(prefix, log_prob) for (prefix, last_char), log_prob in state.items()]
        return sorted(hypos, key=lambda x: x.log_prob, reverse=True)
