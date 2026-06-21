import math
from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC


# ---------------------------------------------------------------------------
# Provided utility — do NOT modify
# ---------------------------------------------------------------------------

def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == float('-inf'):
        return b
    if b == float('-inf'):
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


class Wav2Vec2Decoder:
    def __init__(
            self,
            model_name="facebook/wav2vec2-base-100h",
            lm_model_path="lm/3-gram.pruned.1e-7.arpa.gz",
            beam_width=3,
            alpha=1.0,
            beta=1.0,
            temperature=1.0,
        ):
        """
        Args:
            model_name (str): Pretrained Wav2Vec2 model from HuggingFace.
            lm_model_path (str): Path to a KenLM .arpa/.arpa.gz model.
                Pass None to disable LM (Tasks 1–3).
            beam_width (int): Number of hypotheses kept during beam search.
            alpha (float): LM weight used in shallow fusion and rescoring.
                score = log_p_acoustic + alpha * log_p_lm + beta * num_words
            beta (float): Word insertion bonus (see above).
            temperature (float): Scales acoustic logits before softmax.
                T < 1 sharpens the distribution (model more confident).
                T > 1 flattens it (model less confident, giving LM more
                influence). T = 1.0 leaves logits unchanged.
        """
        # Interact with processor/model ONLY here and in decode() to obtain
        # logits — no further model calls are allowed anywhere else.
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    # -----------------------------------------------------------------------
    # Provided utility — do NOT modify
    # -----------------------------------------------------------------------

    def _ids_to_text(self, token_ids: List[int]) -> str:
        """Convert a list of token IDs to a decoded string."""
        text = ''.join(self.vocab[i] for i in token_ids)
        return text.replace(self.word_delimiter, ' ').strip().lower()

    # KenLM scores are base-10 log-probs; the acoustic log_softmax scores are
    # natural logs.  Convert the LM score to natural log so that alpha weighs
    # two quantities in the same base (score = log_p_ac + alpha*log_p_lm + ...).
    _LOG10_TO_LN = math.log(10.0)

    def _lm_log_prob(self, text: str, eos: bool = False) -> float:
        """Natural-log LM probability of `text` (empty string scores as 0)."""
        if not text:
            return 0.0
        return self.lm_model.score(text, bos=True, eos=eos) * self._LOG10_TO_LN

    # -----------------------------------------------------------------------
    # Tasks 1–4: implement the methods below
    # -----------------------------------------------------------------------

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V).

        Returns:
            str: Decoded transcript.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        best_path = log_probs.argmax(dim=-1).tolist()  # (T,) best token per step

        collapsed: List[int] = []
        prev = None
        for token_id in best_path:
            if token_id != prev:               # collapse consecutive repeats
                if token_id != self.blank_token_id:  # drop blanks
                    collapsed.append(token_id)
            prev = token_id
        return self._ids_to_text(collapsed)

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM).

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.
            return_beams (bool): Return all beam hypotheses for second-pass
                LM rescoring.

        Returns:
            Union[str, List[Tuple[List[int], float]]]:
                str - best decoded transcript (if return_beams=False).
                List[Tuple[List[int], float]] - list of (token_ids, log_prob)
                    tuples sorted best-first (if return_beams=True).
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        NEG_INF = float('-inf')

        # Each prefix maps to (log_p_blank, log_p_non_blank): the probability
        # of the collapsed prefix accumulated over paths ending in a blank vs.
        # ending in a real (non-blank) symbol. Splitting the two is what makes
        # repeated-character handling correct in CTC prefix beam search.
        beams = {(): (0.0, NEG_INF)}

        for t in range(T):
            lp = log_probs[t].tolist()
            next_beams: dict = {}

            def _get(prefix):
                return next_beams.get(prefix, (NEG_INF, NEG_INF))

            for prefix, (p_b, p_nb) in beams.items():
                p_total = _log_add(p_b, p_nb)
                last = prefix[-1] if prefix else None

                for c in range(V):
                    p_c = lp[c]

                    if c == self.blank_token_id:
                        # Blank keeps the prefix unchanged, feeds p_blank.
                        nb_b, nb_nb = _get(prefix)
                        next_beams[prefix] = (_log_add(nb_b, p_total + p_c), nb_nb)
                    elif c == last:
                        # Same symbol as prefix end -> two distinct paths:
                        # (1) stay in prefix, only from the non-blank branch
                        nb_b, nb_nb = _get(prefix)
                        next_beams[prefix] = (nb_b, _log_add(nb_nb, p_nb + p_c))
                        # (2) genuine repeat -> extend, only from blank branch
                        ext = prefix + (c,)
                        eb, enb = _get(ext)
                        next_beams[ext] = (eb, _log_add(enb, p_b + p_c))
                    else:
                        # New distinct symbol -> extend from both branches.
                        ext = prefix + (c,)
                        eb, enb = _get(ext)
                        next_beams[ext] = (eb, _log_add(enb, p_total + p_c))

            # Prune to beam_width by total (blank + non-blank) probability.
            ranked = sorted(
                next_beams.items(),
                key=lambda kv: _log_add(kv[1][0], kv[1][1]),
                reverse=True,
            )[:self.beam_width]
            beams = dict(ranked)

        final = sorted(
            ((list(prefix), _log_add(p_b, p_nb)) for prefix, (p_b, p_nb) in beams.items()),
            key=lambda x: x[1],
            reverse=True,
        )
        if return_beams:
            return final
        return self._ids_to_text(final[0][0])

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion.

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size.

        Returns:
            str: Decoded transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM shallow fusion")

        log_probs = torch.log_softmax(logits, dim=-1)
        T, V = log_probs.shape
        NEG_INF = float('-inf')

        beams = {(): (0.0, NEG_INF)}

        # The LM term depends only on a prefix's text, so cache it across the
        # whole utterance (the same prefix recurs every time a blank/repeat
        # leaves it unchanged).  score = acoustic + alpha*log_p_lm + beta*words
        lm_cache: dict = {}

        def lm_term(prefix):
            if prefix not in lm_cache:
                text = self._ids_to_text(list(prefix))
                lm_cache[prefix] = (
                    self.alpha * self._lm_log_prob(text)
                    + self.beta * len(text.split())
                )
            return lm_cache[prefix]

        def combined(item):
            prefix, (p_b, p_nb) = item
            return _log_add(p_b, p_nb) + lm_term(prefix)

        for t in range(T):
            lp = log_probs[t].tolist()
            next_beams: dict = {}

            def _get(prefix):
                return next_beams.get(prefix, (NEG_INF, NEG_INF))

            for prefix, (p_b, p_nb) in beams.items():
                p_total = _log_add(p_b, p_nb)
                last = prefix[-1] if prefix else None

                for c in range(V):
                    p_c = lp[c]
                    if c == self.blank_token_id:
                        nb_b, nb_nb = _get(prefix)
                        next_beams[prefix] = (_log_add(nb_b, p_total + p_c), nb_nb)
                    elif c == last:
                        nb_b, nb_nb = _get(prefix)
                        next_beams[prefix] = (nb_b, _log_add(nb_nb, p_nb + p_c))
                        ext = prefix + (c,)
                        eb, enb = _get(ext)
                        next_beams[ext] = (eb, _log_add(enb, p_b + p_c))
                    else:
                        ext = prefix + (c,)
                        eb, enb = _get(ext)
                        next_beams[ext] = (eb, _log_add(enb, p_total + p_c))

            # Shallow fusion: prune by acoustic + LM combined score.
            ranked = sorted(next_beams.items(), key=combined, reverse=True)
            beams = dict(ranked[:self.beam_width])

        # Final pick: re-score surviving beams with end-of-sentence context.
        def final_score(item):
            prefix, (p_b, p_nb) = item
            text = self._ids_to_text(list(prefix))
            return (
                _log_add(p_b, p_nb)
                + self.alpha * self._lm_log_prob(text, eos=True)
                + self.beta * len(text.split())
            )

        best_prefix = max(beams.items(), key=final_score)[0]
        return self._ids_to_text(list(best_prefix))

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs.

        Args:
            beams (List[Tuple[List[int], float]]): List of (token_ids, log_prob)
                tuples from beam_search_decode(logits, return_beams=True).

        Returns:
            str: Best rescored transcript.
        """
        if not self.lm_model:
            raise ValueError("KenLM model required for LM rescoring")

        best_text, best_score = '', float('-inf')
        for token_ids, acoustic_log_prob in beams:
            text = self._ids_to_text(token_ids)
            score = (
                acoustic_log_prob
                + self.alpha * self._lm_log_prob(text, eos=True)
                + self.beta * len(text.split())
            )
            if score > best_score:
                best_score, best_text = score, text
        return best_text

    # -----------------------------------------------------------------------
    # Provided — do NOT modify
    # -----------------------------------------------------------------------

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Run the full decoding pipeline on a raw audio tensor.

        Args:
            audio_input (torch.Tensor): 1-D or 2-D audio waveform at 16 kHz.
            method (str): One of "greedy", "beam", "beam_lm", "beam_lm_rescore".

        Returns:
            str: Decoded transcript (lowercase).
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        # Temperature scaling (Task 3): flatten/sharpen the distribution
        # before log_softmax.  T=1.0 is a no-op.  Your decoders must call
        # torch.log_softmax on the logits they receive — do not call it here.
        logits = logits / self.temperature

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                f"Unknown method '{method}'. "
                "Choose one of: 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


# ---------------------------------------------------------------------------
# Quick debug helper — run this file directly to sanity-check your decoder
# on the provided examples/ clips before evaluating on the full test sets.
# ---------------------------------------------------------------------------

def test(decoder: Wav2Vec2Decoder, audio_path: str, reference: str) -> None:
    import jiwer

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, f"Expected 16 kHz, got {sr} Hz for {audio_path}"

    print("=" * 60)
    print(f"REF : {reference}")

    for method in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        try:
            hyp = decoder.decode(audio_input, method=method)
        except NotImplementedError:
            print(f"  [{method}] not yet implemented")
            continue
        except ValueError as e:
            print(f"  [{method}] skipped ({e})")
            continue
        cer = jiwer.cer(reference, hyp)
        wer = jiwer.wer(reference, hyp)
        print(f"  [{method}] {hyp}")
        print(f"           WER={wer:.2%}  CER={cer:.2%}")


if __name__ == "__main__":
    # Reference transcripts are lowercase to match the evaluation manifests.
    # examples/ clips are for quick debugging only — use data/librispeech_test_other/
    # and data/earnings22_test/ for all reported metrics.
    test_samples = [
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample1.wav", "if you are generous here is a fitting opportunity for the exercise of your magnanimity if you are proud here am i your rival ready to acknowledge myself your debtor for an act of the most noble forbearance"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample2.wav", "and if any of the other cops had private rackets of their own izzy was undoubtedly the man to find it out and use the information with a beat such as that even going halves and with all the graft to the upper brackets he'd still be able to make his pile in a matter of months"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample3.wav", "guess a man gets used to anything hell maybe i can hire some bums to sit around and whoop it up when the ships come in and bill this as a real old martian den of sin"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample4.wav", "it was a tune they had all heard hundreds of times so there was no difficulty in turning out a passable imitation of it to the improvised strains of i didn't want to do it the prisoner strode forth to freedom"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample5.wav", "marguerite tired out with this long confession threw herself back on the sofa and to stifle a slight cough put up her handkerchief to her lips and from that to her eyes"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample6.wav", "at this time all participants are in a listen only mode"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample7.wav", "the increase was mainly attributable to the net increase in the average size of our fleets"),
        ("/content/drive/MyDrive/Colab Notebooks/hw2/examples/sample8.wav", "operating surplus is a non cap financial measure which is defined as fully in our press release"),
    ]

    decoder = Wav2Vec2Decoder(lm_model_path=None)  # set lm_model_path for Tasks 4+

    for audio_path, reference in test_samples:
        test(decoder, audio_path, reference)
