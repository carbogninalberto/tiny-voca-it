"""Italian language profile, front end, mixed-language loading and voice adaptation.

Tests marked with eSpeak use the real eSpeak NG ``it`` voice when it is
installed; every other test is pure (parsed eSpeak strings, synthetic models).
"""

import shutil
import tempfile
import unittest
from pathlib import Path

import torch

from tiny_voca.config import Config
from tiny_voca.languages import ENGLISH, ITALIAN, profile_for
from tiny_voca.model import TinyVoca
from tiny_voca.text import (frontend_fingerprint, merge_geminates, parse_espeak, phonemize, tokenize)

ESPEAK = shutil.which("espeak-ng") is not None


def small(language="en", speakers=("F2", "M2")):
    return Config.for_language(language, speakers, channels=32, heads=2, feedforward=64, kernel=3, encoder_layers=1,
                               prosody_layers=1, acoustic_layers=1, predictor_channels=16, cep_causal=12, cep_anti=4,
                               noise_cep=8, max_frames=400)


def italian(espeak_output, punctuation=None):
    pieces = [(False, parse_espeak(espeak_output))] + ([(True, punctuation)] if punctuation else [])
    return tokenize(pieces, ITALIAN)


class ProfileTests(unittest.TestCase):
    def test_english_contract_unchanged(self):
        self.assertEqual(len(ENGLISH.symbols), 76)
        self.assertEqual(Config().to_dict().get("language"), None)          # legacy configs keep their exact dict
        self.assertEqual(Config.from_dict(Config().to_dict()), Config())
        self.assertEqual(frontend_fingerprint(), frontend_fingerprint(ENGLISH))

    def test_italian_profile_identity(self):
        self.assertEqual(len(ITALIAN.symbols), len(set(ITALIAN.symbols)))
        self.assertEqual(ITALIAN.symbols[:17], ENGLISH.symbols[:17])           # specials and punctuation share ids
        self.assertNotEqual(frontend_fingerprint(ITALIAN), frontend_fingerprint())
        self.assertEqual(ITALIAN.digest, profile_for("it").digest)
        c = small("it", ("IT1",))
        self.assertEqual(c.to_dict()["language"], "it")
        self.assertEqual(Config.from_dict(c.to_dict()), c)
        with self.assertRaises(ValueError):
            Config(language="it")                                               # English vocabulary size
        with self.assertRaises(ValueError):
            Config(language="xx")

    def test_long_consonants_map_to_short_aligner_labels(self):
        for phone in ITALIAN.consonants:
            label = ITALIAN.aligner_label(phone)
            if phone.startswith("dz"):
                self.assertEqual(label, "dzː")          # the recogniser has no plain "dz" label
                continue
            self.assertIn(label, ITALIAN.phone_set)
            self.assertEqual(label, phone.rstrip("ː"))

    def test_masks_come_from_each_models_profile(self):
        english, it = TinyVoca(small("en")), TinyVoca(small("it", ("IT1",)))
        again = TinyVoca(small("en"))
        self.assertEqual(int(english.phone_tokens.sum()), len(ENGLISH.phones) + 1)
        self.assertEqual(int(it.phone_tokens.sum()), len(ITALIAN.phones) + 1)
        self.assertEqual(int(it.vowel_tokens.sum()), 7)
        self.assertTrue(torch.equal(english.phone_tokens, again.phone_tokens))   # creation order does not leak


class ItalianTokenTests(unittest.TestCase):
    def symbols(self, espeak_output):
        return [s for s in italian(espeak_output).symbols[1:-1]]

    def test_three_geminate_spellings_become_one_long_token(self):
        self.assertEqual(self.symbols("p_ˈa_l_l_a"), ["p", "a", "lː", "a"])     # two separate phones
        self.assertEqual(self.symbols("k_ˈa_ss_a"), ["k", "a", "sː", "a"])      # doubled symbol
        self.assertEqual(self.symbols("ɲ_ˈo_kː_ɪ"), ["ɲ", "o", "kː", "i"])      # length mark
        self.assertEqual(self.symbols("p_ˈi_tsː_a"), ["p", "i", "tsː", "a"])

    def test_minimal_pairs_stay_distinct(self):
        pairs = (("p_ˈa_l_a", "p_ˈa_l_l_a"), ("p_ˈe_n_a", "p_ˈe_n_n_a"), ("n_ˈo_n_o", "n_ˈɔ_n_n_o"), ("k_ˈa_z_a", "k_ˈa_ss_a"))
        for short, long in pairs:
            self.assertNotEqual(italian(short).ids, italian(long).ids)

    def test_stress_is_kept_and_lengths_match(self):
        result = italian("tʃ_i_tː_ˈa", "?")
        self.assertEqual(len(result.symbols), len(result.stress))
        self.assertEqual(result.stress[result.symbols.index("a")], 1)
        self.assertEqual(result.symbols[-2:], ("?", "<eos>"))

    def test_vowel_length_dropped_and_no_english_fallback(self):
        self.assertEqual(self.symbols("t__r_ˈeː_n_t_a"), ["t", "r", "e", "n", "t", "a"])
        result = italian("θ_ˈi_ŋ_k")                     # English 'think': θ is not Italian
        self.assertIn("<unk>", result.symbols)
        self.assertEqual(result.unknown, ("θ",))
        self.assertNotIn("ɹ", ITALIAN.phone_set)

    def test_identical_consonants_across_words_are_not_merged(self):
        result = tokenize([(False, parse_espeak("i_n n_ˈo_m_e"))], ITALIAN)
        self.assertEqual(result.symbols[1:-1], ("i", "n", "<sp>", "n", "o", "m", "e"))

    def test_merge_geminates_keeps_strongest_stress(self):
        self.assertEqual(merge_geminates([("l", 0), ("l", 1)], ITALIAN), [("lː", 1)])


@unittest.skipUnless(ESPEAK, "requires the espeak-ng executable")
class ItalianEspeakTests(unittest.TestCase):
    def phones(self, text):
        result = phonemize(text, ITALIAN)
        self.assertEqual(result.unknown, ())
        return [s for s in result.symbols if s in ITALIAN.phone_set]

    def test_contract_words(self):
        self.assertEqual(self.phones("palla"), ["p", "a", "lː", "a"])
        self.assertEqual(self.phones("cassa"), ["k", "a", "sː", "a"])
        self.assertEqual(self.phones("anno"), self.phones("hanno"))              # intentional homophones
        self.assertIn("ɲ", self.phones("gnocchi"))
        self.assertIn("kː", self.phones("gnocchi"))
        self.assertIn("ʎ", self.phones("figlio"))
        self.assertIn("ʎ", self.phones("gli"))
        self.assertEqual(self.phones("perché")[-1], "e")
        self.assertEqual(self.phones("città")[-1], "a")
        self.assertEqual(phonemize("città", ITALIAN).stress[-2], 1)

    def test_elision_numbers_and_currency(self):
        self.assertEqual(self.phones("l'amico")[:3], ["l", "a", "m"])
        self.assertNotIn("'", phonemize("un'altra", ITALIAN).symbols)
        self.assertGreater(len(self.phones("3,14")), 15)                         # "tre virgola quattordici"
        self.assertNotIn(":", phonemize("12:30", ITALIAN).symbols)
        self.assertIn("ɛ", self.phones("€ 25,50"))                               # "euro ..."

    def test_abbreviation_expansion_keeps_sentence_end(self):
        symbols = phonemize("Ne parla il prof. Rossi, ecc.", ITALIAN).symbols
        self.assertEqual(symbols.count("."), 1)
        self.assertEqual(symbols[-2], ".")

    def test_full_sentence_with_terminal_character(self):
        # The last character of an utterance must survive (eSpeak newline safeguard).
        self.assertEqual(self.phones("Il nonno mangia la pizza")[-1], "a")


class MixedLanguageLoadingTests(unittest.TestCase):
    def test_english_and_italian_checkpoints_load_in_either_order(self):
        from tiny_voca.infer import load_model
        from tiny_voca.train import export_model
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for language, speakers in (("en", ("F2", "M2")), ("it", ("IT1",))):
                torch.manual_seed(0)
                paths[language] = Path(directory) / f"{language}.pt"
                export_model(TinyVoca(small(language, speakers)), paths[language], 1)
            for order in (("en", "it"), ("it", "en")):
                models = {language: load_model(paths[language]) for language in order}
                self.assertEqual(models["it"].config.speakers, ("IT1",))
                self.assertEqual(models["it"].phone_tokens.numel(), len(ITALIAN.symbols))
                self.assertEqual(models["en"].phone_tokens.numel(), 76)

    def test_italian_checkpoint_with_wrong_profile_is_rejected(self):
        from tiny_voca.infer import load_model
        from tiny_voca.train import export_model
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "it.pt"
            export_model(TinyVoca(small("it", ("IT1",))), path, 1)
            state = torch.load(path, weights_only=True)
            state["language"]["profile_digest"] = "0" * 64
            torch.save(state, path)
            with self.assertRaises(ValueError):
                load_model(path)


class AdaptationTests(unittest.TestCase):
    def source(self):
        torch.manual_seed(1)
        from tiny_voca.train import TrainingSystem
        system = TrainingSystem(small("it", ("IT1", "IT2")))
        return system, {"config": system.model.config.to_dict(), "system": system.state_dict(), "step": 5}

    def test_speaker_tables_migrate_by_name(self):
        from tiny_voca.train import TrainingSystem, adaptation_init
        source_system, source = self.source()
        target = TrainingSystem(small("it", ("ME", "IT2")))
        report = adaptation_init(target, source, "IT1")
        self.assertEqual(report["speaker_rows"], {"ME": "IT1", "IT2": "IT2"})
        for table in ("voices", "frame_voices"):
            new, old = getattr(target.model, table).weight, getattr(source_system.model, table).weight
            self.assertTrue(torch.equal(new[0], old[0]))      # new voice from IT1
            self.assertTrue(torch.equal(new[1], old[1]))      # IT2 by name, not by index
        self.assertTrue(torch.equal(target.model.encoder.norm.weight, source_system.model.encoder.norm.weight))

    def test_inference_checkpoint_source_and_unknown_init_voice(self):
        from tiny_voca.train import TrainingSystem, adaptation_init
        source_system, source = self.source()
        inference = {"config": source["config"], "model": source_system.model.state_dict(), "step": 5}
        report = adaptation_init(TrainingSystem(small("it", ("ME",))), inference, "IT2")
        self.assertEqual(report["fresh"], ["mel_head.weight", "mel_head.bias"])
        with self.assertRaises(ValueError):
            adaptation_init(TrainingSystem(small("it", ("ME",))), source, "NOPE")

    def test_shape_mismatch_is_rejected(self):
        from tiny_voca.train import TrainingSystem, adaptation_init
        _, source = self.source()
        wider = Config.for_language("it", ("ME",), channels=64, heads=2, feedforward=64, kernel=3, encoder_layers=1,
                                    prosody_layers=1, acoustic_layers=1, predictor_channels=16, cep_causal=12, cep_anti=4,
                                    noise_cep=8, max_frames=400)
        with self.assertRaises(ValueError):
            adaptation_init(TrainingSystem(wider), source, "IT1")

    def test_frozen_tensors_do_not_move_after_optimizer_steps(self):
        from helpers import sample
        from tiny_voca.data import collate
        from tiny_voca.languages import ITALIAN as P
        from tiny_voca.train import TrainingSystem, adaptation_init, adaptation_parameters
        _, source = self.source()
        for mode in ("embedding", "acoustic"):
            system = TrainingSystem(small("it", ("ME",)))
            adaptation_init(system, source, "IT1")
            trainable = adaptation_parameters(system, mode)
            before = {k: v.clone() for k, v in system.state_dict().items()}
            optimizer = torch.optim.AdamW(trainable, lr=1e-2, weight_decay=0.1)
            item = sample(system.model.config)
            item["tokens"] = torch.tensor([P.symbol_to_id[s] for s in
                                           ("<bos>", "tʃ", "a", "o", "<sp>", "p", "a", "lː", "a", ".", "<eos>", "<eos>")])
            item["speaker"] = 0
            for _ in range(2):
                optimizer.zero_grad()
                loss, _, _ = system.objective(collate([item]), 1.0)
                loss.backward()
                optimizer.step()
            after = system.state_dict()
            moved = {k for k in before if not torch.equal(before[k], after[k])}
            self.assertIn("model.voices.weight", moved)
            if mode == "embedding":
                self.assertEqual(moved, {"model.voices.weight", "model.frame_voices.weight"})
            else:
                self.assertFalse(any(k.startswith(("model.encoder.", "model.tokens.", "model.stress.")) for k in moved))
                self.assertTrue(any(k.startswith("model.acoustic_stack.") for k in moved))


class ValidationBatchTests(unittest.TestCase):
    def test_frame_budget_and_size(self):
        from tiny_voca.train import validation_batches
        lengths = [100, 100, 100, 900, 2000]
        batches = validation_batches(range(5), lengths, 8, frame_budget=2000)
        self.assertEqual(batches, [[0, 1, 2], [3], [4]])
        self.assertTrue(all(len(b) * max(lengths[i] for i in b) <= 2000 for b in batches))
        self.assertEqual(validation_batches(range(5), lengths, 2), [[0, 1], [2, 3], [4]])


if __name__ == "__main__":
    unittest.main()


class ClosureShiftTests(unittest.TestCase):
    def test_closure_moves_from_vowel_to_stop_and_keeps_totals(self):
        import numpy as np
        from tiny_voca.align import AlignmentSettings, closure_shift
        symbols = ["<bos>", "p", "a", "tː", "a", "<eos>"]
        durations = [2, 3, 10, 3, 6, 2]
        level = np.full(sum(durations), -20.0)
        level[10:15] = -60.0                       # last 5 vowel frames are the silent closure
        new = closure_shift(symbols, durations, level, ITALIAN.closure_consonants, ITALIAN.vowel_set, 12.0)
        self.assertEqual(list(new), [2, 3, 5, 8, 6, 2])
        self.assertEqual(int(new.sum()), sum(durations))
        # Sonorants and vowel-internal dips are untouched; the vowel keeps its minimum.
        same = closure_shift(["a", "lː", "a"], [10, 3, 5], np.full(18, -60.0), ITALIAN.closure_consonants, ITALIAN.vowel_set, 12.0)
        self.assertEqual(list(same), [10, 3, 5])
        self.assertNotIn("closure_drop_db", AlignmentSettings().to_dict())      # English identity unchanged
