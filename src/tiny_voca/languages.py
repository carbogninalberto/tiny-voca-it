"""Language profiles: everything that gives a token id its meaning.

A profile fixes the ordered symbol inventory, the phonetic classes the model
and the annotation layer derive masks from, and the rules that turn eSpeak NG
output into inventory phones. Consumers receive a profile explicitly (normally
``Config.profile``); nothing here is mutable process state, so English and
Italian models can be loaded in one process in either order without sharing
masks or token meanings.

English (``en``) is the legacy contract of format-4 artifacts. Its symbol order,
fallback table and front-end fingerprint are unchanged, and an artifact without
language metadata is English.

Italian (``it``, profile version 1) uses eSpeak NG's ``it`` voice. Its rules:

* No English fallback table. A phone outside the inventory becomes ``<unk>``
  and is reported; annotation rejects such clips instead of anglicising them.
* Consonant length is phonemic and is represented by ONE long token ``Cː``
  (``palla`` = p a lː a). eSpeak 1.51 writes a geminate in three ways — two
  separate phones (``l_l``), a doubled symbol (``ss``) or a length mark
  (``kː``, ``tsː``); all three become the same long token. The aligner
  labels a long consonant with its short label (``ALIGNER_LABELS``): the
  recogniser marks phone onsets, and the long token owns the frames up to the
  next onset, so the closure/hold lands inside it.
* Vowel length is not phonemic in Italian (eSpeak lengthens some stressed
  vowels) and is dropped; ``ɪ``/``ʊ`` (eSpeak's reduced final vowels) are
  written ``i``/``u``. Open/closed e and o (``ɛ``/``e``, ``ɔ``/``o``) stay
  distinct, as eSpeak chooses them.
* Lexical stress stays a parallel feature (0/1/2), never a token.
"""

from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
from types import MappingProxyType

PAD, BOS, EOS, SP, UNK = "<pad>", "<bos>", "<eos>", "<sp>", "<unk>"
SPECIALS = (PAD, BOS, EOS, SP, UNK)
PUNCTUATION = (",", ".", "?", "!", ";", ":", '"', "'", "(", ")", "—", "…")
LENGTH = "ː"


@dataclass(frozen=True, eq=False)
class LanguageProfile:
    language: str
    locale: str
    version: int
    espeak_voice: str
    consonants: tuple
    vowels: tuple
    voiceless: frozenset
    fricatives: frozenset          # voiceless fricatives: used by the label-free timing probe
    abbreviations: frozenset
    fallback: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    canonical_map: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    expansions: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))   # "abbreviation." -> words
    geminates: bool = False
    drop_vowel_length: bool = False
    aligner_labels: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    default_asr_language: str = "en"
    closure_consonants: frozenset = frozenset()   # stops/affricates whose silent closure precedes the release

    @cached_property
    def phones(self):
        return self.consonants + self.vowels

    @cached_property
    def symbols(self):
        return SPECIALS + PUNCTUATION + self.phones

    @cached_property
    def symbol_to_id(self):
        return MappingProxyType({symbol: index for index, symbol in enumerate(self.symbols)})

    @cached_property
    def phone_set(self):
        return frozenset(self.phones)

    @cached_property
    def phone_ids(self):
        """Tokens that must be realised as audio (phones and ``<unk>``)."""
        return frozenset(self.symbol_to_id[p] for p in self.phones) | {self.symbol_to_id[UNK]}

    @cached_property
    def vowel_set(self):
        return frozenset(self.vowels)

    @cached_property
    def vowel_ids(self):
        return frozenset(self.symbol_to_id[v] for v in self.vowels)

    def aligner_label(self, symbol):
        """Recogniser label a token is aligned with (itself unless mapped)."""
        return self.aligner_labels.get(symbol, symbol)

    def identity(self):
        """Everything that decides token meanings. Not used for English, whose
        identity is the legacy front-end identity (see ``text.frontend_identity``)."""
        return {"language": self.language, "locale": self.locale, "profile_version": self.version, "phonemizer": "espeak",
                "voice": self.espeak_voice, "separator": "_", "symbols": list(self.symbols),
                "geminates": "long-token" if self.geminates else None, "drop_vowel_length": self.drop_vowel_length,
                "canonical": {key: list(value) for key, value in sorted(self.canonical_map.items())},
                "fallback": {key: list(value) for key, value in sorted(self.fallback.items())},
                "aligner_labels": dict(sorted(self.aligner_labels.items())),
                "abbreviations": sorted(self.abbreviations), "expansions": dict(sorted(self.expansions.items()))}

    @cached_property
    def digest(self):
        return hashlib.sha256(json.dumps(self.identity(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# --- English: the legacy format-4 contract. Append-only; order is load-bearing. ---
_EN_CONSONANTS = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
                  "m", "n", "ŋ", "l", "ɹ", "j", "w", "tʃ", "dʒ", "ɾ", "ʔ")
_EN_VOWELS = ("ə", "ɐ", "ɚ", "ᵻ", "ɪ", "i", "iː", "ɛ", "æ", "ʌ", "ɑː", "ɔ", "ɔː", "ʊ", "uː", "ɜː",
              "eɪ", "aɪ", "ɔɪ", "oʊ", "aʊ", "iə", "oː",
              "ɑːɹ", "ɔːɹ", "oːɹ", "ɪɹ", "ɛɹ", "ʊɹ", "aɪɚ", "aɪə", "əl", "n̩")
_EN_FALLBACK = {
    "r": ("ɹ",), "ɻ": ("ɹ",), "ʁ": ("ɹ",), "x": ("k",), "ç": ("h",), "ɬ": ("l",), "ɫ": ("l",), "ɲ": ("n", "j"),
    "ts": ("t", "s"), "dz": ("d", "z"), "β": ("v",), "ɣ": ("ɡ",), "ʎ": ("l", "j"),
    "a": ("æ",), "aː": ("ɑː",), "ɑ": ("ɑː",), "ɒ": ("ɑː",), "ɑ̃": ("ɑː",), "ɔ̃": ("ɔː",), "ɛ̃": ("æ",),
    "e": ("eɪ",), "eː": ("eɪ",), "o": ("oʊ",), "u": ("uː",), "y": ("uː",), "yː": ("uː",),
    "ø": ("ɜː",), "øː": ("ɜː",), "œ": ("ɜː",), "ɜ": ("ɜː",), "ɛː": ("ɛ",), "ɪː": ("ɪ",),
    "əʊ": ("oʊ",), "eə": ("ɛɹ",), "ɪə": ("iə",), "ʊə": ("ʊɹ",), "aʊɚ": ("aʊ", "ɚ"), "l̩": ("əl",), "m̩": ("ə", "m"),
}

ENGLISH = LanguageProfile(
    language="en", locale="en-US", version=3, espeak_voice="en-us",
    consonants=_EN_CONSONANTS, vowels=_EN_VOWELS,
    voiceless=frozenset(("p", "t", "k", "f", "θ", "s", "ʃ", "h", "tʃ", "ʔ")),
    fricatives=frozenset(("f", "θ", "s", "ʃ")),
    abbreviations=frozenset({"mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc"}),
    fallback=MappingProxyType(_EN_FALLBACK))


# --- Italian, profile version 1 (eSpeak NG 1.51 ``it``). ---
_IT_SHORT = ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "z", "ʃ", "ts", "dz", "tʃ", "dʒ",
             "m", "n", "ɲ", "ŋ", "l", "ʎ", "r", "j", "w")
# Long consonants that occur in Italian spelling (doubling) or are inherently long between vowels (ɲ ʎ ʃ ts dz).
# eSpeak 1.51 writes intervocalic ʃ and ʎ short, so they have no long token.
_IT_LONG = tuple(c + LENGTH for c in ("p", "b", "t", "d", "k", "ɡ", "f", "v", "s", "ts", "dz", "tʃ", "dʒ",
                                      "m", "n", "ɲ", "l", "r"))
_IT_VOWELS = ("a", "e", "ɛ", "i", "o", "ɔ", "u")
# eSpeak symbols with a fixed Italian reading (not a fallback: these are the same sound).
# eSpeak writes Italian vowel+i / vowel+u sequences ("dei", "mai", "causa") with
# English-style diphthong symbols; in Italian they are two segments.
_IT_CANONICAL = {"ɪ": ("i",), "ʊ": ("u",), "ss": ("sː",), "ɾ": ("r",), "g": ("ɡ",), "ɑ": ("a",),
                 "eɪ": ("e", "i"), "ɛɪ": ("ɛ", "i"), "aɪ": ("a", "i"), "oɪ": ("o", "i"), "ɔɪ": ("ɔ", "i"), "uɪ": ("u", "i"),
                 "aʊ": ("a", "u"), "eʊ": ("e", "u"), "ɛʊ": ("ɛ", "u")}

# Token -> label of the annotation recogniser (facebook/wav2vec2-lv-60-espeak-cv-ft,
# revision ae45363b). The recogniser has no plain "dz"; it has "dzː". A long
# consonant is aligned with its short label (one onset; the token owns the hold).
_IT_ALIGNER_LABELS = {**{c: c[:-1] for c in _IT_LONG}, "dz": "dzː", "dzː": "dzː"}

ITALIAN = LanguageProfile(
    language="it", locale="it-IT", version=1, espeak_voice="it",
    consonants=_IT_SHORT + _IT_LONG, vowels=_IT_VOWELS,
    voiceless=frozenset(("p", "t", "k", "f", "s", "ʃ", "ts", "tʃ", "pː", "tː", "kː", "fː", "sː", "tsː", "tʃː")),
    fricatives=frozenset(("f", "s", "ʃ", "fː", "sː")),
    abbreviations=frozenset(("sig", "sigg", "dott", "dr", "prof", "ing", "avv", "geom", "arch", "ecc", "pag", "cap", "art",
                             "vol", "nr", "tel")),
    expansions=MappingProxyType({"sig": "signor", "sigg": "signori", "dott": "dottor", "dr": "dottor", "prof": "professor",
                                 "ing": "ingegner", "avv": "avvocato", "geom": "geometra", "arch": "architetto",
                                 "ecc": "eccetera", "pag": "pagina", "cap": "capitolo", "art": "articolo", "vol": "volume",
                                 "nr": "numero", "tel": "telefono"}),
    canonical_map=MappingProxyType(_IT_CANONICAL),
    geminates=True, drop_vowel_length=True,
    aligner_labels=MappingProxyType(_IT_ALIGNER_LABELS),
    default_asr_language="it",
    closure_consonants=frozenset(c for c in _IT_SHORT + _IT_LONG
                                 if c.rstrip(LENGTH) in ("p", "b", "t", "d", "k", "ɡ", "ts", "dz", "tʃ", "dʒ")))

PROFILES = MappingProxyType({"en": ENGLISH, "it": ITALIAN})


def profile_for(language):
    try:
        return PROFILES[language]
    except KeyError:
        raise ValueError(f"unknown language {language!r}; available: {sorted(PROFILES)}") from None
