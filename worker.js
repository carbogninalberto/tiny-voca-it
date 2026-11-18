// Browser worker: Italian front end (eSpeak NG compiled to WebAssembly + the
// ported tiny_voca rules in frontend.js) and the model (ONNX Runtime Web).
// Messages in: {type: "load", model} and {type: "speak", id, text, speed, pitch, range, voice, emotion, intensity}.
// Messages out: status, ready, sentence, done, error.

import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs";
import ESpeakNg from "./espeak-ng/espeak-ng.js";
import { ItalianFrontend } from "./frontend.js";

const VERSION = "2";   // bump when files under models/ or espeak-ng/ change
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
ort.env.wasm.numThreads = 1;

let espeakModule = null, frontend = null;
const sessions = new Map();
let current = null;
const status = (text, progress) => postMessage({ type: "status", text, progress });

async function download(url, label) {
  const key = `${url}?v=${VERSION}`;
  let store = null;
  try {
    store = await caches.open("tiny-voca-it-" + VERSION);
    const hit = await store.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { store = null; }
  const response = await fetch(key);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  let bytes;
  if (!response.body || !total) {
    bytes = new Uint8Array(await response.arrayBuffer());
  } else {
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      status(label, received / total);
    }
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  }
  if (store) store.put(key, new Response(bytes)).catch(() => {});
  return bytes;
}

async function espeak(piece) {
  // Text goes through a file, newline-terminated (as the Python front end feeds stdin);
  // a piece starting with "-" would otherwise be read as an option.
  const instance = await ESpeakNg({
    arguments: ["--phonout", "out", "-q", "--ipa", "--sep=_", "-v", "it", "-f", "in.txt"],
    print: () => {}, printErr: () => {},
    preRun: [(m) => m.FS.writeFile("in.txt", piece.replace(/\n+$/, "") + "\n")],
    instantiateWasm: (imports, done) => { WebAssembly.instantiate(espeakModule, imports).then((i) => done(i)); return {}; },
  });
  return instance.FS.readFile("out", { encoding: "utf8" });
}

async function loadEspeak() {
  if (espeakModule) return;
  const bytes = await download("espeak-ng/espeak-ng.wasm", "Loading the Italian phonemizer (eSpeak NG)");
  status("Starting the phonemizer");
  espeakModule = await WebAssembly.compile(bytes);
}

async function loadModel(entry) {
  if (!sessions.has(entry.id)) {
    const metadata = await (await fetch(`models/${entry.file}.json?v=${VERSION}`)).json();
    const bytes = await download(`models/${entry.file}.onnx`, `Loading ${entry.label}`);
    status("Starting the model");
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] });
    sessions.set(entry.id, { session, metadata });
  }
  current = sessions.get(entry.id);
  frontend = new ItalianFrontend(current.metadata.frontend_identity, espeak);
}

function sentences(text) {
  const parts = text.replace(/\s+/g, " ").trim().split(/(?<=[.!?…;])\s+/);
  return parts.filter((part) => /[\p{L}\p{N}]/u.test(part));
}

async function fit(sentence) {
  const phones = await frontend.phonemize(sentence);
  if (phones.ids.length <= current.metadata.max_tokens - 8) return [phones];
  const pieces = sentence.split(/(?<=[,:—])\s+/);
  if (pieces.length > 1) return (await Promise.all(pieces.map(fit))).flat();
  const words = sentence.split(" ");
  if (words.length < 2) throw new Error("a single word is too long to read");
  const half = Math.ceil(words.length / 2);
  return [...await fit(words.slice(0, half).join(" ")), ...await fit(words.slice(half).join(" "))];
}

async function synthesize(phones, { speed, pitch, range, voice = 0, emotion = 0, intensity = 0 }) {
  const count = phones.ids.length;
  const integers = (values) => new ort.Tensor("int64", BigInt64Array.from(values.map(BigInt)), [1, count]);
  const scalar = (value) => new ort.Tensor("float32", Float32Array.from([value]), [1]);
  const index = (value) => new ort.Tensor("int64", BigInt64Array.from([BigInt(value)]), [1]);
  const feeds = {
    tokens: integers(phones.ids), stress: integers(phones.stress), voice: index(voice),
    speed: scalar(speed), pitch_shift: scalar(pitch * Math.LN2 / 12), pitch_range: scalar(range),
  };
  if (current.metadata.inputs.includes("emotion")) Object.assign(feeds, { emotion: index(emotion), intensity: scalar(intensity) });
  const output = await current.session.run(feeds);
  return output.audio.data;
}

async function speak(request) {
  const started = performance.now();
  const list = sentences(request.text);
  if (!list.length) throw new Error("Scrivi un testo / type some text");
  const rate = current.metadata.sample_rate;
  const pause = new Float32Array(Math.round(0.25 * rate));
  const chunks = [], phonemes = [];
  let phonemeTime = 0, modelTime = 0;
  for (let index = 0; index < list.length; index++) {
    let mark = performance.now();
    const utterances = await fit(list[index]);
    phonemeTime += performance.now() - mark;
    for (const phones of utterances) {
      mark = performance.now();
      const audio = await synthesize(phones, request);
      modelTime += performance.now() - mark;
      if (chunks.length) chunks.push(pause);
      chunks.push(audio);
      phonemes.push(phones.symbols.filter((s) => !s.startsWith("<") || s === "<sp>").join(" ").replaceAll("<sp>", "·"));
      if (phones.unknown.length) phonemes.push(`(unknown: ${phones.unknown.join(" ")})`);
    }
    postMessage({ type: "sentence", id: request.id, index, total: list.length });
  }
  const length = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const audio = new Float32Array(length);
  let offset = 0;
  for (const chunk of chunks) { audio.set(chunk, offset); offset += chunk.length; }
  postMessage({ type: "done", id: request.id, audio, sampleRate: rate, phonemes: phonemes.join(" | "), phonemeTime, modelTime,
                totalTime: performance.now() - started }, [audio.buffer]);
}

onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      await Promise.all([loadEspeak(), loadModel(data.model)]);
      postMessage({ type: "ready", model: data.model.id, step: current.metadata.step, vocoderStep: current.metadata.vocoder_step,
                    voices: current.metadata.voices, emotions: current.metadata.emotions || [] });
    } else if (data.type === "speak") {
      await speak(data);
    }
  } catch (error) {
    postMessage({ type: "error", id: data.id, message: String((error && error.message) || error) });
  }
};
