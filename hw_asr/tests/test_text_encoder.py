import unittest
import torch

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder, Hypothesis


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d " \
               "dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        text_encoder = CTCCharTextEncoder()

        eps = 1e-25
        log_probs_length = 2

        #       t_0,    t_1
        # a     0.3     0.3
        # b     0.1     0.2
        # ^     0.6     0.5

        # beam_size=2:
        # 1. a -> 0.3
        #    b -> 0.1
        #    ^ -> 0.6
        # truncate: a -> 0.3    b -> 0.1
        # 2. aa -> 0.09
        #    a^ -> 0.15
        #    ab -> 0.06
        #    ^a -> 0.18
        #    ^b -> 0.12
        #    ^^ -> 0.3
        # Total:
        #    a -> 0.42
        #    ^ -> 0.3
        #    b -> 0.12
        #    ab -> 0.06

        # beam_size=3:
        # 1. a -> 0.3
        #    b -> 0.1
        #    ^ -> 0.6
        # 2. aa -> 0.09
        #    a^ -> 0.15
        #    ab -> 0.06
        #    ^a -> 0.18
        #    ^b -> 0.12
        #    bb -> 0.02
        #    b^ -> 0.05
        #    ba -> 0.03
        # Total:
        #    a -> 0.42
        #    ^ -> 0.3
        #    b -> 0.19
        #    ab -> 0.06
        #    ba -> 0.03

        results = {
            2: [Hypothesis('a', 0.42), Hypothesis('', 0.3)],
            3: [Hypothesis('a', 0.42), Hypothesis('', 0.3), Hypothesis('b', 0.19)]
        }
        for beam_size in [2, 3]:
            probs = torch.full((log_probs_length, len(text_encoder.char2ind)), eps)

            # Find indices of 'ab^'
            ab_ind = [text_encoder.char2ind[c] for c in f'ab{text_encoder.EMPTY_TOK}']
            a_ind, b_ind, blank_ind = ab_ind

            # Fill probabilities
            probs[0, a_ind] = 0.3
            probs[0, b_ind] = 0.1
            probs[0, blank_ind] = 0.6

            probs[1, a_ind] = 0.3
            probs[1, b_ind] = 0.2
            probs[1, blank_ind] = 0.5

            for p in probs:
                self.assertAlmostEqual(p.sum(), 1)

            # Take logarithm
            log_probs = torch.log(probs)

            # Apply beam search
            hypotheses = text_encoder.ctc_beam_search(log_probs, log_probs_length, beam_size=beam_size)
            self.assertEqual(len(hypotheses), beam_size)
            self.assertTrue(all(isinstance(h, Hypothesis) for h in hypotheses))
            self.assertTrue(all(isinstance(h.text, str) for h in hypotheses))
            self.assertTrue(all(isinstance(h.prob, float) for h in hypotheses))

            for h_true, h_pred in zip(results[beam_size], hypotheses):
                self.assertAlmostEqual(h_true.prob, h_pred.prob)
                self.assertEqual(h_true.text, h_pred.text)
