"""Italian numbers to words, applied to the text before phonemisation (Python and, identically, the
browser's ``playground/web/numbers.js``; ``tests/test_numbers.py`` checks they agree).

It is a pre-normalisation step outside the front-end identity, so existing checkpoints keep their
fingerprint. Handled: cardinals (with Italian elisions: ventuno, ventotto, centotto, ventitré),
thousands separators (12.500), decimals (5,8 -> cinque virgola otto), percentages, euro/dollar amounts,
times (12:30), dates (19/11/2026), ordinals (1°, 2ª), signs, and long digit strings (read digit by digit).
"""

import re

UNITS = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici",
         "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove"]
TENS = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"]
MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
          "novembre", "dicembre"]
ORDINALS = ["", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo"]


def _below_thousand(n):
    hundreds, rest = divmod(n, 100)
    words = "" if hundreds == 0 else ("cento" if hundreds == 1 else UNITS[hundreds] + "cento")
    if rest == 0:
        return words
    if rest < 20:
        tail = UNITS[rest]
    else:
        tens, unit = divmod(rest, 10)
        tail = TENS[tens]
        if unit:
            tail = (tail[:-1] if unit in (1, 8) else tail) + UNITS[unit]
    if words and tail[0] == "o":          # centootto -> centotto (cento + otto/ottanta)
        words = words[:-1]
    return words + tail


def cardinal(n):
    """Words for 0 <= n < 10**12, written as Italian writes them (one word below a million)."""
    if n == 0:
        return "zero"
    parts = []
    for size, one, many in ((10 ** 9, "un miliardo", "miliardi"), (10 ** 6, "un milione", "milioni")):
        count, n = divmod(n, size)
        if count:
            parts.append(one if count == 1 else f"{cardinal(count)} {many}")
    thousands, rest = divmod(n, 1000)
    word = ""
    if thousands:
        word = "mille" if thousands == 1 else _below_thousand(thousands) + "mila"
    if rest:
        word += _below_thousand(rest)
    if word:
        parts.append(word)
    return " ".join(parts)


def _accent(word):
    """ventitre -> ventitré (final -tre of a compound carries the accent; plain "tre" does not)."""
    return word[:-3] + "tré" if word.endswith("tre") and word != "tre" else word


def number(digits):
    """A digit string as words; leading zeros and very long strings are read digit by digit."""
    if len(digits) > 1 and digits[0] == "0" or len(digits) > 12:
        return " ".join(UNITS[int(d)] for d in digits)
    return " ".join(_accent(w) for w in cardinal(int(digits)).split(" "))


def decimal(integer, fraction):
    return f"{number(integer)} virgola {number(fraction)}"


def _amount(integer, fraction, unit, plural):
    integer = integer.replace(".", "")
    head = ("un" if integer == "1" else number(integer)) + " " + (unit if integer == "1" else plural)
    if fraction and int(fraction):
        cents = fraction.ljust(2, "0")[:2]
        head += " e " + number(str(int(cents)))
    return head


CURRENCY = {"€": ("euro", "euro"), "$": ("dollaro", "dollari"), "£": ("sterlina", "sterline")}
NUM = r"\d{1,3}(?:\.\d{3})+|\d+"


def _sub(pattern, replacement, text):
    # ASCII classes (\d, \w, \b), exactly like the JavaScript port without the u flag.
    return re.sub(pattern, replacement, text, flags=re.ASCII)


def expand_numbers(text):
    def money(match):
        symbol = match["s1"] or match["s2"]
        unit, plural = CURRENCY[symbol]
        return _amount(match["i"], match["f"], unit, plural)

    text = _sub(r"(?:(?P<s1>[€$£])\s?(?P<i>" + NUM + r")(?:,(?P<f>\d{1,2}))?|(?P<i2>" + NUM + r")(?:,(?P<f2>\d{1,2}))?\s?(?P<s2>[€$£]))",
                  lambda m: money({"s1": m["s1"], "s2": m["s2"], "i": m["i"] or m["i2"], "f": m["f"] or m["f2"]}), text)
    text = _sub(r"(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?!\d)", lambda m: _date(*m.groups()), text)
    text = _sub(r"(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])", lambda m: _time(*m.groups()), text)
    text = _sub(r"(?<!\w)(\d{1,2})\s?[°ºª](?!\w)", lambda m: _ordinal(int(m[1]), m[0][-1] == "ª"), text)
    text = _sub(r"(" + NUM + r")(?:,(\d+))?\s?%", lambda m: _plain(m[1], m[2]) + " per cento", text)
    text = _sub(r"(?:(?<=\s)|^)[-−](?=\d)", "meno ", text)
    text = _sub(r"(?<![\w,.])(" + NUM + r")(?:,(\d+))?(?![\w])", lambda m: _plain(m[1], m[2]), text)
    text = _sub(r"\b([Aa])lle una\b", r"\1ll'una", text)            # alle una -> all'una
    return _sub(r"\s+", " ", text).strip()


def _plain(integer, fraction):
    integer = integer.replace(".", "")
    return decimal(integer, fraction) if fraction else number(integer)


def _ordinal(n, feminine):
    if 0 < n <= 10:
        word = ORDINALS[n]
    else:
        stem = number(str(n))
        stem = stem.replace("tré", "tre")
        stem = stem if stem.endswith(("tre", "sei")) else stem[:-1]
        word = stem + "esimo"
    return word[:-1] + "a" if feminine else word


def _date(day, month, year):
    d, m = int(day), int(month)
    if not (1 <= d <= 31 and 1 <= m <= 12):
        return f"{number(day)} {number(month)} {number(year)}"
    return f"{'primo' if d == 1 else number(str(d))} {MONTHS[m - 1]} {number(year)}"


def _time(hours, minutes):
    h, m = int(hours), int(minutes)
    spoken = "una" if h == 1 else number(str(h))
    return spoken if m == 0 else f"{spoken} e {number(str(m))}"
