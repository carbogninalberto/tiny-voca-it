import unittest

from tiny_voca.stream import MAX_CHARACTERS, utterances


class UtteranceTests(unittest.TestCase):
    def test_sentences_and_clauses(self):
        self.assertEqual(utterances("Ciao!  Come stai? Bene; grazie."), ["Ciao!", "Come stai?", "Bene;", "grazie."])
        self.assertEqual(utterances("  ...  "), [])
        long = ", ".join(["una parte della frase abbastanza lunga"] * 12) + "."
        parts = utterances(long)
        self.assertGreater(len(parts), 1)
        self.assertTrue(all(len(p) <= MAX_CHARACTERS for p in parts))
        self.assertEqual(" ".join(parts), long)


if __name__ == "__main__":
    unittest.main()
