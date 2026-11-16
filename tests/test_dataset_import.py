"""Import helpers: duplicate grouping, deterministic splits, audio checks (synthetic only)."""

import random
import unittest

import numpy as np

from tiny_voca.dataset_import import assign_splits, audio_hash, check_audio, connected_groups, dedup_text


def records(count=60):
    rows = []
    for i in range(count):
        rows.append({"id": f"r{i:03d}", "text": f"Frase numero {i}.", "audio_sha256": f"a{i}", "group": f"book-{i // 3}",
                     "seconds": 5.0 + i % 4})
    return rows


class DatasetImportTests(unittest.TestCase):
    def test_split_does_not_depend_on_input_order(self):
        rows = records()
        shuffled = rows[:]
        random.Random(3).shuffle(shuffled)
        first = assign_splits(rows, connected_groups(rows), seed=1)
        second = assign_splits(shuffled, connected_groups(shuffled), seed=1)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "validation", "test"})

    def test_duplicates_are_connected_across_groups(self):
        rows = records()
        rows[10]["text"] = "Ciao, mondo!"
        rows[40]["text"] = "ciao mondo"                 # same normalised transcript
        rows[20]["audio_sha256"] = rows[50]["audio_sha256"]  # identical decoded audio
        groups = connected_groups(rows)
        self.assertEqual(groups["r010"], groups["r040"])
        self.assertEqual(groups["r020"], groups["r050"])
        self.assertEqual(groups["r010"], groups["r011"])     # same book
        splits = assign_splits(rows, groups, seed=7)
        self.assertEqual(splits["r010"], splits["r040"])
        self.assertEqual(splits["r020"], splits["r050"])

    def test_too_few_groups_fails_clearly(self):
        rows = [{**r, "group": "one-book", "audio_sha256": "x"} for r in records(6)]
        with self.assertRaises(ValueError):
            assign_splits(rows, connected_groups(rows), seed=1)

    def test_dedup_text_ignores_case_accents_and_punctuation(self):
        self.assertEqual(dedup_text("Perché?  L’amico!"), dedup_text("perche l'amico"))
        self.assertNotEqual(dedup_text("pala"), dedup_text("palla"))

    def test_audio_checks(self):
        tone = 0.1 * np.sin(np.linspace(0, 2000, 24000)).astype(np.float32)
        check_audio(tone, 24000)
        for bad, rate in ((tone, 22050), (np.zeros(24000, np.float32), 24000), (tone[:1000], 24000),
                          (np.stack((tone, tone)), 24000), (np.full(24000, np.nan, np.float32), 24000), (tone * 20, 24000)):
            with self.assertRaises(ValueError):
                check_audio(bad, rate)
        self.assertEqual(audio_hash(tone), audio_hash(tone.astype(np.float64)))


if __name__ == "__main__":
    unittest.main()
