// Italian numbers to words before phonemisation: a line-by-line port of src/tiny_voca/italian_numbers.py
// (tests/test_numbers.py runs both on the same cases). ASCII regex classes on purpose, like the Python
// side (re.ASCII), so \d, \w and \b mean the same thing in both.

const UNITS = ["zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove", "dieci", "undici", "dodici",
  "tredici", "quattordici", "quindici", "sedici", "diciassette", "diciotto", "diciannove"];
const TENS = ["", "", "venti", "trenta", "quaranta", "cinquanta", "sessanta", "settanta", "ottanta", "novanta"];
const MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre", "ottobre",
  "novembre", "dicembre"];
const ORDINALS = ["", "primo", "secondo", "terzo", "quarto", "quinto", "sesto", "settimo", "ottavo", "nono", "decimo"];
const CURRENCY = { "€": ["euro", "euro"], "$": ["dollaro", "dollari"], "£": ["sterlina", "sterline"] };
const NUM = String.raw`\d{1,3}(?:\.\d{3})+|\d+`;

function belowThousand(n) {
  const hundreds = Math.floor(n / 100), rest = n % 100;
  let words = hundreds === 0 ? "" : (hundreds === 1 ? "cento" : UNITS[hundreds] + "cento");
  if (rest === 0) return words;
  let tail;
  if (rest < 20) tail = UNITS[rest];
  else {
    const tens = Math.floor(rest / 10), unit = rest % 10;
    tail = TENS[tens];
    if (unit) tail = (unit === 1 || unit === 8 ? tail.slice(0, -1) : tail) + UNITS[unit];
  }
  if (words && tail[0] === "o") words = words.slice(0, -1);
  return words + tail;
}

export function cardinal(n) {
  if (n === 0) return "zero";
  const parts = [];
  for (const [size, one, many] of [[1e9, "un miliardo", "miliardi"], [1e6, "un milione", "milioni"]]) {
    const count = Math.floor(n / size); n %= size;
    if (count) parts.push(count === 1 ? one : `${cardinal(count)} ${many}`);
  }
  const thousands = Math.floor(n / 1000), rest = n % 1000;
  let word = "";
  if (thousands) word = thousands === 1 ? "mille" : belowThousand(thousands) + "mila";
  if (rest) word += belowThousand(rest);
  if (word) parts.push(word);
  return parts.join(" ");
}

const accent = (word) => (word.endsWith("tre") && word !== "tre" ? word.slice(0, -3) + "tré" : word);

function number(digits) {
  if ((digits.length > 1 && digits[0] === "0") || digits.length > 12) return [...digits].map((d) => UNITS[+d]).join(" ");
  return cardinal(parseInt(digits, 10)).split(" ").map(accent).join(" ");
}

const decimal = (integer, fraction) => `${number(integer)} virgola ${number(fraction)}`;
const plain = (integer, fraction) => { integer = integer.replaceAll(".", ""); return fraction ? decimal(integer, fraction) : number(integer); };

function amount(integer, fraction, unit, plural) {
  integer = integer.replaceAll(".", "");
  let head = (integer === "1" ? "un" : number(integer)) + " " + (integer === "1" ? unit : plural);
  if (fraction && parseInt(fraction, 10)) head += " e " + number(String(parseInt(fraction.padEnd(2, "0").slice(0, 2), 10)));
  return head;
}

function ordinal(n, feminine) {
  let word;
  if (n > 0 && n <= 10) word = ORDINALS[n];
  else {
    let stem = number(String(n)).replaceAll("tré", "tre");
    stem = stem.endsWith("tre") || stem.endsWith("sei") ? stem : stem.slice(0, -1);
    word = stem + "esimo";
  }
  return feminine ? word.slice(0, -1) + "a" : word;
}

function date(day, month, year) {
  const d = +day, m = +month;
  if (!(d >= 1 && d <= 31 && m >= 1 && m <= 12)) return `${number(day)} ${number(month)} ${number(year)}`;
  return `${d === 1 ? "primo" : number(String(d))} ${MONTHS[m - 1]} ${number(year)}`;
}

function time(hours, minutes) {
  const h = +hours, m = +minutes;
  const spoken = h === 1 ? "una" : number(String(h));
  return m === 0 ? spoken : `${spoken} e ${number(String(m))}`;
}

export function expandNumbers(text) {
  text = text.replace(new RegExp(String.raw`(?:([€$£])\s?(${NUM})(?:,(\d{1,2}))?|(${NUM})(?:,(\d{1,2}))?\s?([€$£]))`, "g"),
    (_, s1, i1, f1, i2, f2, s2) => { const [unit, plural] = CURRENCY[s1 || s2]; return amount(i1 || i2, f1 || f2, unit, plural); });
  text = text.replace(/(?<!\d)(\d{1,2})\/(\d{1,2})\/(\d{4}|\d{2})(?!\d)/g, (_, d, m, y) => date(d, m, y));
  text = text.replace(/(?<![\d:])([01]?\d|2[0-3]):([0-5]\d)(?![\d:])/g, (_, h, m) => time(h, m));
  text = text.replace(/(?<!\w)(\d{1,2})\s?[°ºª](?!\w)/g, (match, n) => ordinal(+n, match.endsWith("ª")));
  text = text.replace(new RegExp(String.raw`(${NUM})(?:,(\d+))?\s?%`, "g"), (_, i, f) => plain(i, f) + " per cento");
  text = text.replace(/(?:(?<=\s)|^)[-−](?=\d)/g, "meno ");
  text = text.replace(new RegExp(String.raw`(?<![\w,.])(${NUM})(?:,(\d+))?(?!\w)`, "g"), (_, i, f) => plain(i, f));
  text = text.replace(/\b([Aa])lle una\b/g, "$1ll'una");
  return text.replace(/\s+/g, " ").trim();
}
