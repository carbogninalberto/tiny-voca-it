import json
import shutil
import subprocess
import unittest
from pathlib import Path

from tiny_voca.italian_numbers import cardinal, expand_numbers

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    "Il treno parte alle 12:30 dal binario 7 e il biglietto costa € 25,50.",
    "Un terremoto di magnitudo 5,8 ha colpito la costa.",
    "Ha vinto per 3 a 1, con il 15% di possesso in più.",
    "Nato il 19/11/2026, arrivato 1° e lei 2ª, al 23° posto e al 26° giro.",
    "Costa 1 € o 12.500 euro, alle 0:15 e alle 1:00. Alle 1:30 parto.",
    "Temperatura -5 gradi, telefono 3471234567890, codice 007.",
    "Nel 1998 erano 21 persone, 0,05 di differenza, 108 pagine, 2.500.000 abitanti.",
    "Ha 33 anni e 101 amici; la lezione dura 45 minuti, 2 ore o 1000 secondi.",
    "Prezzi: 99$, £ 3, 3,14 e 1,5%.",
    "Nessun numero qui, solo parole: perché la città è così silenziosa?",
]


class NumberTests(unittest.TestCase):
    def test_cardinals(self):
        expected = {21: "ventuno", 28: "ventotto", 108: "centotto", 180: "centottanta", 1000: "mille", 2026: "duemilaventisei",
                    21000: "ventunomila", 1_000_000: "un milione", 2_500_000: "due milioni cinquecentomila"}
        for n, words in expected.items():
            self.assertEqual(cardinal(n), words)

    def test_expansions(self):
        self.assertEqual(expand_numbers(CASES[0]), "Il treno parte alle dodici e trenta dal binario sette e il biglietto costa venticinque euro e cinquanta.")
        self.assertEqual(expand_numbers("33 e 23°"), "trentatré e ventitreesimo")
        self.assertEqual(expand_numbers("alle 1:00"), "all'una")
        self.assertEqual(expand_numbers(CASES[-1]), CASES[-1])

    def test_browser_port_agrees(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node is not installed")
        script = (f"import {{ expandNumbers }} from '{(ROOT / 'playground/web/numbers.js').as_uri()}';"
                  f"console.log(JSON.stringify({json.dumps(CASES, ensure_ascii=False)}.map(expandNumbers)));")
        output = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True, text=True, check=True).stdout
        for case, js in zip(CASES, json.loads(output)):
            self.assertEqual(js, expand_numbers(case), case)


if __name__ == "__main__":
    unittest.main()
