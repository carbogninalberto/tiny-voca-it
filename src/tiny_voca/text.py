"""English front end: text -> phoneme tokens with a stress feature.

The same code runs during annotation, training-set preparation and inference,
so tokens can never drift between them. Two programs can produce the phonemes:

* ``espeak`` (front-end version 3, the default): eSpeak NG, as described below.
  The current checkpoints and annotations were made with it.
* ``own`` (front-end version 4): the separate ``dizione`` package, a dictionary
  built from CMUdict, a small model for unknown words and rules. No eSpeak, so
  the whole system can be shipped under a permissive licence. Same 76 symbols.

Choose with ``set_frontend("own")`` or ``TINY_VOCA_FRONTEND=own``. Everything
below about tokens holds for both; the rest of this note is the eSpeak path.

* eSpeak NG (``en-us``) is the only pronunciation source. It is an external
  rule-based program: no neural G2P and no downloaded weights.
* A token is one eSpeak phoneme exactly as eSpeak segments it (``--sep``), for
  example ``t``, ``aɪ``, ``ɑːɹ`` or ``n̩``. This is also the label inventory of
  the phoneme recogniser used for offline forced alignment, so one aligned
  label corresponds to exactly one token.
* Lexical stress is a parallel feature (0 none, 1 primary, 2 secondary), not a
  token, so it never has to be given audio frames.
* Punctuation, word boundaries and the utterance edges are explicit tokens that
  may last zero frames. They carry pauses when the recording has them.

Versions 1 and 2 encoded IPA as UTF-8 bytes and piped text to eSpeak without a
final newline; eSpeak NG 1.51 then drops the last character of the input, so
the final phonemes of every clause were wrong. Version 3 terminates the input
and has a regression test for it.
"""

from dataclasses import dataclass
import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata

FRONTEND_VERSION = 3
ESPEAK_VOICE = "en-us"
SEPARATOR = "_"
FRONTENDS = ("espeak", "own")
FRONTEND_VARIABLE = "TINY_VOCA_FRONTEND"
_frontend = os.environ.get(FRONTEND_VARIABLE, "espeak")
if _frontend not in FRONTENDS:
    raise ValueError(f"{FRONTEND_VARIABLE} must be one of {FRONTENDS}, not {_frontend!r}")

from .languages import (ENGLISH, ITALIAN, LENGTH, PAD, BOS, EOS, SP, UNK, SPECIALS, PUNCTUATION, LanguageProfile,
                        profile_for)

# English (legacy) inventory, unchanged: module-level names keep their meaning.
CONSONANTS = ENGLISH.consonants
VOWELS = ENGLISH.vowels
PHONES = ENGLISH.phones
# Append-only: changing the order or removing a symbol invalidates checkpoints.
SYMBOLS = ENGLISH.symbols
SYMBOL_TO_ID = dict(ENGLISH.symbol_to_id)
PHONE_SET = ENGLISH.phone_set
PHONE_IDS = ENGLISH.phone_ids
VOICELESS = ENGLISH.voiceless
VOWEL_SET = ENGLISH.vowel_set
STRESS_MARKS = {"ˈ": 1, "ˌ": 2}

# eSpeak phonemes outside the English inventory (foreign words, rare allophones)
# map to the closest inventory phone(s). Anything else becomes <unk> and is reported.
FALLBACK = dict(ENGLISH.fallback)

_QUOTES = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',
                         "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")"})
_PUNCTUATION_CHARS = frozenset(PUNCTUATION)
ABBREVIATIONS = set(ENGLISH.abbreviations)
_LANGUAGE_TAG = re.compile(r"\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)")


@dataclass(frozen=True)
class Phonemized:
    symbols: tuple
    ids: tuple
    stress: tuple
    unknown: tuple  # eSpeak phonemes that had no inventory mapping

    def __len__(self):
        return len(self.ids)


def frontend():
    """Name of the program that produces the phonemes: ``"espeak"`` or ``"own"``."""
    return _frontend


def set_frontend(name):
    """Select the phonemizer for this process. Tokens of the two front ends
    must not be mixed: annotations and checkpoints record which one made them."""
    global _frontend
    if name not in FRONTENDS:
        raise ValueError(f"front end must be one of {FRONTENDS}, not {name!r}")
    _frontend = name


def _dizione():
    try:
        import dizione
    except ImportError as error:
        raise RuntimeError('the "own" front end needs the dizione package: pip install dizione') from error
    if tuple(dizione.symbols.SYMBOLS) != SYMBOLS:
        raise RuntimeError("dizione's symbol inventory differs from this model's")
    return dizione


def _italian(profile):
    return profile is not None and profile.language != "en"


def resolve(profile=None):
    """``None`` or a language code -> a profile (``None`` means legacy English)."""
    if profile is None:
        return ENGLISH
    return profile if isinstance(profile, LanguageProfile) else profile_for(profile)


def frontend_identity(profile=None):
    profile = resolve(profile)
    if profile.language != "en":
        return profile.identity()
    if _frontend == "own":
        phonemizer = _dizione()
        return {"frontend_version": phonemizer.VERSION, "phonemizer": "dizione", "symbols": list(SYMBOLS)}
    return {"frontend_version": FRONTEND_VERSION, "voice": ESPEAK_VOICE, "separator": SEPARATOR,
            "symbols": list(SYMBOLS), "fallback": {key: list(value) for key, value in sorted(FALLBACK.items())}}


def frontend_runtime(profile=None):
    """What an annotation run records about the exact program behind its
    tokens: the eSpeak version, or the fingerprint of the phonemizer build."""
    if resolve(profile).language == "en" and _frontend == "own":
        return {"phonemizer": _dizione().identity()["fingerprint"]}
    return {"espeak_version": espeak_version()}


def frontend_fingerprint(profile=None):
    return hashlib.sha256(json.dumps(frontend_identity(profile), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalize_text(text, profile=None):
    text = unicodedata.normalize("NFKC", text).translate(_QUOTES)
    text = re.sub(r"\.{3,}", "…", text)
    text = re.sub(r"-{2,}|\s-\s", "—", text)
    profile = resolve(profile)
    if profile.expansions:
        # "Sig." -> "signor" before chunking, so the period is not read as a full stop.
        pattern = r"\b(" + "|".join(sorted(map(re.escape, profile.expansions), key=len, reverse=True)) + r")\.(?=\s|$)"
        text = re.sub(pattern, lambda m: profile.expansions[m[1].lower()] + ("." if m.end() == len(text) else ""), text,
                      flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def chunks(text, profile=None):
    """Yield ``(is_punctuation, piece)``. Decimals, clock times, digit groups,
    contractions/elisions, title abbreviations and initialisms stay inside the
    text piece for eSpeak; every other punctuation mark becomes its own token."""
    abbreviations = resolve(profile).abbreviations
    start = 0
    for i, char in enumerate(text):
        if char not in _PUNCTUATION_CHARS:
            continue
        previous = text[i-1] if i else ""
        following = text[i+1] if i+1 < len(text) else ""
        if char == "'" and previous.isalnum() and following.isalnum():
            continue
        if char in ".,:" and previous.isdigit() and following.isdigit():
            continue
        if char == ".":
            word = re.search(r"([^\W\d_]+)$", text[:i])
            if word and word[1].lower() in abbreviations:
                continue
            if previous.isalpha() and following.isalpha():
                continue
        if text[start:i].strip():
            yield False, text[start:i].strip()
        yield True, char
        start = i+1
    if text[start:].strip():
        yield False, text[start:].strip()


def espeak_executable():
    executable = shutil.which("espeak-ng")
    if executable is None:
        raise RuntimeError("Install espeak-ng on the cloud machine before preprocessing/inference")
    return executable


def espeak_version():
    return subprocess.run([espeak_executable(), "--version"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()


def espeak_words(piece, voice=ESPEAK_VOICE):
    """Phonemize one text piece. Returns words as lists of ``(phone, stress)``.

    The input must end with a newline: eSpeak NG 1.51 silently drops the last
    character of unterminated standard input.
    """
    result = subprocess.run(
        [espeak_executable(), "-q", "--ipa", "--sep=" + SEPARATOR, "-v", voice, "--stdin"],
        input=piece.rstrip("\n") + "\n", text=True, capture_output=True, check=True, timeout=30)
    return parse_espeak(result.stdout)


def parse_espeak(output):
    words = []
    for word in _LANGUAGE_TAG.sub("", output).split():
        units = []
        for unit in word.split(SEPARATOR):
            stress = 0
            for mark, level in STRESS_MARKS.items():
                if mark in unit:
                    stress = stress or level
                    unit = unit.replace(mark, "")
            unit = "".join(c for c in unit if unicodedata.category(c) != "Cf" and c not in "()")
            if unit:
                units.append((unit, stress))
        if units:
            words.append(units)
    return words


def canonical(phone, profile=None):
    """Map an eSpeak phoneme to inventory phones; ``None`` when unknown."""
    profile = resolve(profile)
    if _italian(profile):
        return _canonical_italian(phone, profile)
    if phone in PHONE_SET:
        return (phone,)
    if phone in FALLBACK:
        return FALLBACK[phone]
    stripped = "".join(c for c in unicodedata.normalize("NFD", phone) if not unicodedata.combining(c))
    for candidate in (stripped, stripped.replace("ː", ""), stripped + "ː"):
        if candidate != phone and candidate in PHONE_SET:
            return (candidate,)
        if candidate != phone and candidate in FALLBACK:
            return FALLBACK[candidate]
    return None


def _canonical_italian(phone, profile):
    """Italian rules (see ``languages``): explicit same-sound spellings, long
    consonants as one token, vowel length dropped. No cross-language fallback."""
    if phone in profile.phone_set:
        return (phone,)
    if phone in profile.canonical_map:
        return profile.canonical_map[phone]
    base = phone.replace(LENGTH, "")
    base = profile.canonical_map.get(base, (base,))
    if len(base) == 1:
        base = base[0]
        if LENGTH in phone:
            if base in profile.vowel_set and profile.drop_vowel_length:
                return (base,)
            if base + LENGTH in profile.phone_set:
                return (base + LENGTH,)
        elif base in profile.phone_set:
            return (base,)
    return None


def merge_geminates(units, profile):
    """Two identical short consonants inside a word are one long consonant."""
    merged = []
    for phone, stress in units:
        if (merged and phone == merged[-1][0] and phone + LENGTH in profile.phone_set):
            merged[-1] = (phone + LENGTH, max(stress, merged[-1][1]))
        else:
            merged.append((phone, stress))
    return merged


def tokenize(pieces, profile=None):
    """Build the token sequence from ``(is_punctuation, value)`` pieces where a
    text piece's value is the parsed eSpeak word list. Pure function (testable
    without eSpeak)."""
    profile = resolve(profile)
    phone_set, vowel_set, symbol_to_id = profile.phone_set, profile.vowel_set, profile.symbol_to_id
    symbols, stress, unknown = [BOS], [0], []
    for punctuation, value in pieces:
        if punctuation:
            if symbols[-1] != value:  # collapse repeats such as "!!"
                symbols.append(value)
                stress.append(0)
            continue
        for word in value:
            if symbols[-1] in phone_set or symbols[-1] == UNK:
                symbols.append(SP)
                stress.append(0)
            units = []
            for phone, level in word:
                mapped = canonical(phone, profile)
                if mapped is None:
                    unknown.append(phone)
                    mapped = (UNK,)
                # A fallback may expand to several phones: stress goes to its first vowel.
                carrier = next((i for i, item in enumerate(mapped) if item in vowel_set), 0)
                units.extend((item, level if index == carrier else 0) for index, item in enumerate(mapped))
            if profile.geminates:
                units = merge_geminates(units, profile)
            for item, level in units:
                symbols.append(item)
                stress.append(level)
    symbols.append(EOS)
    stress.append(0)
    if not any(s in phone_set or s == UNK for s in symbols):
        raise ValueError("the text has nothing to pronounce")
    return Phonemized(tuple(symbols), tuple(symbol_to_id[s] for s in symbols), tuple(stress), tuple(unknown))


def phonemize(text, profile=None):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must not be empty")
    profile = resolve(profile)
    if profile.language == "en" and _frontend == "own":
        return tokenize(_dizione().pieces(text))
    voice = ESPEAK_VOICE if profile.language == "en" else profile.espeak_voice
    pieces = []
    for punctuation, piece in chunks(normalize_text(text, profile), profile):
        pieces.append((True, piece) if punctuation else (False, espeak_words(piece, voice)))
    return tokenize(pieces, profile)


def is_phone(token_ids, profile=None):
    """Boolean list/tensor helper: tokens that must be realised as audio."""
    ids = resolve(profile).phone_ids
    return [int(i) in ids for i in token_ids]
