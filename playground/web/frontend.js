// Italian front end for the browser: a line-by-line port of tiny_voca.text for
// non-English profiles. All rules (symbols, eSpeak canonicalisation, geminates,
// abbreviations) come from the model's exported metadata (`frontend_identity`),
// the same table Python uses. eSpeak NG output is supplied by the caller
// (`espeak(piece) -> Promise<string>`), so this module runs in a worker or in Node.

const PAD = "<pad>", BOS = "<bos>", EOS = "<eos>", SP = "<sp>", UNK = "<unk>";
const LENGTH = "ː";
const STRESS_MARKS = [["ˈ", 1], ["ˌ", 2]];
const QUOTES = { "‘": "'", "’": "'", "‚": "'", "‛": "'", "`": "'", "´": "'", "“": '"', "”": '"', "„": '"', "‟": '"',
                 "«": '"', "»": '"', "–": "—", "―": "—", "[": "(", "{": "(", "]": ")", "}": ")" };
const LANGUAGE_TAG = /\((?:_?[a-z0-9]_?){2,3}(?:-(?:_?[a-z0-9]_?)+)*\)/g;
const ALNUM = /[\p{L}\p{N}]/u, DIGIT = /\p{Nd}/u, ALPHA = /\p{L}/u, FORMAT = /\p{Cf}/u;
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

export class ItalianFrontend {
  constructor(identity, espeak) {
    if (identity.phonemizer !== "espeak" || identity.geminates !== "long-token") {
      throw new Error("unsupported front-end identity");
    }
    this.espeak = espeak;
    this.symbols = identity.symbols;
    this.ids = new Map(this.symbols.map((s, i) => [s, i]));
    this.punctuation = new Set(this.symbols.slice(5, 17));
    this.phones = new Set(this.symbols.slice(17));
    this.vowels = new Set(this.symbols.slice(17).filter((s) => !/[^aeiouɛɔ]/u.test(s)));
    this.canonicalMap = identity.canonical;
    this.dropVowelLength = identity.drop_vowel_length;
    this.abbreviations = new Set(identity.abbreviations);
    this.expansions = identity.expansions;
    const keys = Object.keys(this.expansions).sort((a, b) => b.length - a.length).map(escape);
    this.expansionPattern = keys.length
      ? new RegExp(`(?<![\\p{L}\\p{N}_])(${keys.join("|")})\\.(?=\\s|$)`, "giu") : null;
  }

  normalize(text) {
    text = text.normalize("NFKC").replace(/./gsu, (c) => QUOTES[c] ?? c);
    text = text.replace(/\.{3,}/g, "…").replace(/-{2,}|\s-\s/gu, "—");
    if (this.expansionPattern) {
      const whole = text;
      text = text.replace(this.expansionPattern, (match, word, offset) =>
        this.expansions[word.toLowerCase()] + (offset + match.length === whole.length ? "." : ""));
    }
    return text.replace(/\s+/gu, " ").trim();
  }

  *chunks(text) {
    const chars = Array.from(text);
    let start = 0;
    for (let i = 0; i < chars.length; i++) {
      const c = chars[i];
      if (!this.punctuation.has(c)) continue;
      const previous = i ? chars[i - 1] : "", following = i + 1 < chars.length ? chars[i + 1] : "";
      if (c === "'" && ALNUM.test(previous) && ALNUM.test(following)) continue;
      if (".,:".includes(c) && DIGIT.test(previous) && DIGIT.test(following)) continue;
      if (c === ".") {
        const word = chars.slice(0, i).join("").match(/(\p{L}+)$/u);
        if (word && this.abbreviations.has(word[1].toLowerCase())) continue;
        if (ALPHA.test(previous) && ALPHA.test(following)) continue;
      }
      const piece = chars.slice(start, i).join("").trim();
      if (piece) yield [false, piece];
      yield [true, c];
      start = i + 1;
    }
    const rest = chars.slice(start).join("").trim();
    if (rest) yield [false, rest];
  }

  static parseEspeak(output) {
    const words = [];
    for (const word of output.replace(LANGUAGE_TAG, "").split(/\s+/u).filter(Boolean)) {
      const units = [];
      for (let unit of word.split("_")) {
        let stress = 0;
        for (const [mark, level] of STRESS_MARKS) {
          if (unit.includes(mark)) { stress = stress || level; unit = unit.split(mark).join(""); }
        }
        unit = Array.from(unit).filter((c) => !FORMAT.test(c) && c !== "(" && c !== ")").join("");
        if (unit) units.push([unit, stress]);
      }
      if (units.length) words.push(units);
    }
    return words;
  }

  canonical(phone) {
    if (this.phones.has(phone)) return [phone];
    if (phone in this.canonicalMap) return this.canonicalMap[phone];
    let base = phone.split(LENGTH).join("");
    const mapped = this.canonicalMap[base] ?? [base];
    if (mapped.length !== 1) return null;
    base = mapped[0];
    if (phone.includes(LENGTH)) {
      if (this.vowels.has(base) && this.dropVowelLength) return [base];
      if (this.phones.has(base + LENGTH)) return [base + LENGTH];
    } else if (this.phones.has(base)) {
      return [base];
    }
    return null;
  }

  tokenize(pieces) {
    const symbols = [BOS], stress = [0], unknown = [];
    const isPhone = (s) => this.phones.has(s) || s === UNK;
    for (const [punctuation, value] of pieces) {
      if (punctuation) {
        if (symbols[symbols.length - 1] !== value) { symbols.push(value); stress.push(0); }
        continue;
      }
      for (const word of value) {
        if (isPhone(symbols[symbols.length - 1])) { symbols.push(SP); stress.push(0); }
        let units = [];
        for (const [phone, level] of word) {
          let mapped = this.canonical(phone);
          if (mapped === null) { unknown.push(phone); mapped = [UNK]; }
          let carrier = mapped.findIndex((item) => this.vowels.has(item));
          if (carrier < 0) carrier = 0;
          mapped.forEach((item, index) => units.push([item, index === carrier ? level : 0]));
        }
        const merged = [];
        for (const [phone, level] of units) {
          const last = merged[merged.length - 1];
          if (last && phone === last[0] && this.phones.has(phone + LENGTH)) merged[merged.length - 1] = [phone + LENGTH, Math.max(level, last[1])];
          else merged.push([phone, level]);
        }
        for (const [item, level] of merged) { symbols.push(item); stress.push(level); }
      }
    }
    symbols.push(EOS); stress.push(0);
    if (!symbols.some(isPhone)) throw new Error("the text has nothing to pronounce");
    return { symbols, ids: symbols.map((s) => this.ids.get(s)), stress, unknown };
  }

  async phonemize(text) {
    if (typeof text !== "string" || !text.trim()) throw new Error("text must not be empty");
    const pieces = [];
    for (const [punctuation, piece] of this.chunks(this.normalize(text))) {
      pieces.push(punctuation ? [true, piece] : [false, ItalianFrontend.parseEspeak(await this.espeak(piece))]);
    }
    return this.tokenize(pieces);
  }
}

export const SPECIALS = [PAD, BOS, EOS, SP, UNK];
