// tiny-voca-it network diagram (shared by index.html and architecture.html).
// drawNetwork(svg, onPick): draws every tensor and layer into <svg viewBox="0 0 2245 665">; onPick(id) is called on click;
// returns mark(id), which highlights a component.
window.drawNetwork = function (svg, onPick) {
  const NS = "http://www.w3.org/2000/svg";
  const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || svg).append(e); return e; };
  const text = (x, y, str, cls, parent, anchor = "middle") => { const t = el("text", { x, y, class: cls, "text-anchor": anchor }, parent); t.textContent = str; return t; };
  const MID = 300;                       // vertical centre line of the tensors
  const C = { front: ["#2a2a34", "#3a3a46", "#9b9ba6"], voice: ["#3b3470", "#4b4290", "#8b7bf0"], dsp: ["#173544", "#1f4a5e", "#7dd3fc"],
              voc: ["#173a2e", "#1f5040", "#34d399"], out: ["#173a2e", "#1f5040", "#34d399"] };
  const groups = {};
  function group(id) { const g = el("g", { class: "g", "data-id": id }); g.addEventListener("click", () => onPick(id)); (groups[id] = groups[id] || []).push(g); return g; }
  const mark = (id) => { for (const k in groups) groups[k].forEach((g) => g.classList.toggle("on", k === id)); };
  // cuboid: front face w × h centred on MID, depth d drawn up-right
  function cuboid(g, x, w, h, d, kind, faceFill) {
    const [front, top, edge] = C[kind], y = MID - h / 2, k = d * 0.6;
    el("path", { d: `M${x},${y} l${k},${-k} l${w},0 l${-k},${k} z`, fill: top, stroke: edge, "stroke-width": 1 }, g);
    el("path", { d: `M${x + w},${y} l${k},${-k} l0,${h} l${-k},${k} z`, fill: top, stroke: edge, "stroke-width": 1, opacity: .85 }, g);
    el("rect", { x, y, width: w, height: h, fill: faceFill || front, stroke: edge, "stroke-width": 1.2, class: "face" }, g);
    return { x, y, w, h, k, r: x + w + k, cx: x + (w + k) / 2 };
  }
  // n cuboids back to back (a stack of identical blocks)
  function blocks(g, x, w, h, d, n, kind, gap = 6) { let last; for (let i = 0; i < n; i++) last = cuboid(g, x + i * (w + gap), w, h, d, kind); return { ...last, x0: x }; }
  function join(a, b) {   // dashed corner lines between two tensors
    el("line", { x1: a.x + a.w, y1: a.y, x2: b.x0 ?? b.x, y2: b.y, stroke: "#6b6b76", "stroke-dasharray": "5 4" });
    el("line", { x1: a.x + a.w, y1: a.y + a.h, x2: b.x0 ?? b.x, y2: b.y + b.h, stroke: "#6b6b76", "stroke-dasharray": "5 4" });
  }
  function brace(g, x1, x2, lines, y = 120) {
    const m = (x1 + x2) / 2;
    el("path", { d: `M${x1},${y + 16} C${x1},${y + 4} ${m - 10},${y + 12} ${m},${y} C${m + 10},${y + 12} ${x2},${y + 4} ${x2},${y + 16}`, fill: "none", stroke: "#cfcfe6", "stroke-width": 2 }, g);
    lines.forEach((ln, i) => text(m, y - 10 - (lines.length - 1 - i) * 17, ln, i === 0 ? "t u" : "s", g));
  }
  function shape(g, x, a, b) { text(x, 470, a, "s", g); text(x, 489, b, "m", g); }

  // INPUT text sheet
  let g = group("text");
  el("rect", { x: 20, y: MID - 60, width: 104, height: 120, rx: 5, fill: "#0b0b0f", stroke: "#9b9ba6", class: "face" }, g);
  ["Questa è", "una prova", "del modello", "tiny-voca…"].forEach((t, i) => text(72, MID - 28 + i * 21, t, "s", g));
  text(72, 470, "INPUT", "t", g); text(72, 489, "Italian text", "m", g);
  // tokens
  g = group("phonemes");
  for (let i = 0; i < 10; i++) el("rect", { x: 175, y: MID - 85 + i * 17, width: 15, height: 14, rx: 2, fill: i % 3 ? "#2a2a34" : "#4b38d9", stroke: "#9b9ba6", class: i ? "" : "face" }, g);
  brace(g, 152, 214, ["eSpeak NG", "phonemes", "+ stress"]);
  shape(g, 183, "tokens", "[T]");
  el("path", { d: `M128,${MID} L168,${MID}`, stroke: "#9b9ba6", "stroke-width": 1.6, "marker-end": "url(#arr)" });
  // embedding
  g = group("embed"); const emb = cuboid(g, 262, 24, 170, 46, "voice");
  brace(g, 242, 320, ["Embedding", "token + stress", "+ voice + emotion"]);
  shape(g, emb.cx, "sum", "[T × 192]");
  el("line", { x1: 190, y1: MID - 85, x2: 262, y2: emb.y, stroke: "#6b6b76", "stroke-dasharray": "5 4" }); el("line", { x1: 190, y1: MID + 85, x2: 262, y2: emb.y + emb.h, stroke: "#6b6b76", "stroke-dasharray": "5 4" });
  // encoder: 4 blocks
  g = group("encoder"); const enc = blocks(g, 372, 20, 170, 46, 4, "voice");
  brace(g, 362, 500, ["Encoder", "4 × transformer block", "rotary attn · DWConv 7"]);
  shape(g, 432, "4 blocks", "[T × 192]");
  join(emb, enc);
  // durations
  g = group("duration");
  [3, 7, 4, 9, 5, 2, 6, 8, 3, 5].forEach((v, i) => el("rect", { x: 548 + i * 9, y: MID + 60 - v * 12, width: 7, height: v * 12, fill: i === 3 ? "#8b7bf0" : "#4b42a0", class: i === 3 ? "face" : "" }, g));
  brace(g, 536, 650, ["Durations", "Conv3 → Conv3 → Linear", "frames + pauses"]);
  shape(g, 592, "per phoneme", "[T × 2]");
  el("path", { d: `M${enc.r + 4},${MID} L540,${MID}`, stroke: "#9b9ba6", "stroke-width": 1.6, "marker-end": "url(#arr)" });
  // length regulator: T → N (taller)
  g = group("expand"); const reg = cuboid(g, 710, 24, 250, 46, "voice");
  for (let i = 1; i < 10; i++) el("line", { x1: 710, y1: reg.y + i * 25, x2: 734, y2: reg.y + i * 25, stroke: "#8b7bf0", "stroke-width": .8 }, g);
  brace(g, 690, 790, ["Length regulator", "repeat by duration", "+ frame position"]);
  shape(g, reg.cx, "frames", "[N × 192]");
  el("path", { d: `M650,${MID} L704,${MID}`, stroke: "#9b9ba6", "stroke-width": 1.6, "marker-end": "url(#arr)" });
  // prosody: 2 blocks + [N × 3]
  g = group("prosody"); const pro = blocks(g, 850, 20, 250, 46, 2, "voice");
  const p3 = cuboid(g, 955, 8, 250, 14, "voice", "#5a4a10");
  el("path", { d: `M${p3.x + 4},${MID + 100} C${p3.x + 4},${MID + 60} ${p3.x + 4},${MID + 20} ${p3.x + 4},${MID - 30}`, stroke: "#fbbf24", "stroke-width": 1, fill: "none" }, g);
  brace(g, 840, 985, ["Prosody stack", "2 × transformer block", "→ pitch · voicing · energy"]);
  shape(g, 915, "2 blocks → 3 tracks", "[N × 192] → [N × 3]");
  join(reg, pro);
  // acoustic: 5 blocks
  g = group("acoustic"); const ac = blocks(g, 1035, 20, 250, 46, 5, "voice");
  brace(g, 1025, 1205, ["Acoustic stack", "5 × transformer block", "conditioned on prosody"]);
  shape(g, 1112, "5 blocks", "[N × 192]");
  el("path", { d: `M${p3.r + 2},${MID} L1030,${MID}`, stroke: "#9b9ba6", "stroke-width": 1.6, "marker-end": "url(#arr)" });
  // source-filter heads
  g = group("sourcefilter");
  const h1 = cuboid(g, 1260, 14, 250, 26, "dsp"), h2 = cuboid(g, 1300, 8, 250, 16, "dsp");
  el("path", { d: `M1332,${MID - 40} l6,0 l0,-24 l3,24 l10,0 l0,-24 l3,24 l10,0 l0,-24 l3,24 l8,0`, stroke: "#7dd3fc", "stroke-width": 1.8, fill: "none" }, g);
  text(1355, MID - 72, "pulses", "s", g);
  let d = `M1332,${MID + 40}`; for (let i = 0; i < 16; i++) d += ` l2.6,${(i % 2 ? 1 : -1) * (4 + (i * 7) % 9)}`;
  el("path", { d, stroke: "#7dd3fc", "stroke-width": 1.4, fill: "none" }, g); text(1355, MID + 72, "noise", "s", g);
  brace(g, 1250, 1385, ["Source-filter", "cepstral envelopes", "× pulses + noise"]);
  shape(g, 1305, "two heads", "[N×98] · [N×36]");
  join(ac, h1);
  // waveform → Vocos mel (spectrogram face)
  g = group("vocosmel");
  d = `M1400,${MID}`; for (let i = 0; i < 26; i++) d += ` L${1400 + i * 2.4},${MID + Math.sin(i * 1.4) * (6 + (i % 6) * 4)}`;
  el("path", { d, stroke: "#9b9ba6", fill: "none", "stroke-width": 1.2 }, g);
  const mel = cuboid(g, 1478, 70, 250, 30, "dsp");
  for (let i = 0; i < 10; i++) for (let j = 0; j < 18; j++) {
    const v = Math.max(0, Math.sin(i * .9 + j * .35) * .45 + (j > 11 ? .5 : .2) - ((i * 7 + j * 3) % 5) * .05);
    el("rect", { x: 1478 + i * 7, y: mel.y + j * (250 / 18), width: 7, height: 250 / 18, fill: `rgba(125,211,252,${(.1 + v * .8).toFixed(2)})` }, g);
  }
  brace(g, 1395, 1575, ["Vocos log-mel", "STFT 1024 · hop 256", "100 mel bands"]);
  shape(g, 1520, "wave → mel", "[100 × N]");
  // prior (side input from pitch)
  g = group("prior");
  for (let i = 0; i < 4; i++) el("line", { x1: 1478, y1: 512 + i * 8, x2: 1560, y2: 512 + i * 8, stroke: "#7dd3fc", "stroke-width": 1.6, class: i ? "" : "face" }, g);
  text(1519, 554, "pulse-train prior [513 × N]", "s", g);
  // vocoder: embed + 8 ConvNeXt
  g = group("vocoder"); const vin = cuboid(g, 1640, 24, 250, 46, "voc");
  const voc = blocks(g, 1690, 16, 250, 46, 8, "voc", 5);
  brace(g, 1630, 1905, ["Neural vocoder", "Conv7 embed + 8 × ConvNeXt", "192 → 576 → 192 per block"]);
  shape(g, 1770, "8 blocks", "[192 × N]");
  join(mel, vin);
  el("path", { d: `M1562,524 C1610,524 1615,${MID + 140} 1640,${MID + 118}`, stroke: "#7dd3fc", "stroke-dasharray": "5 4", fill: "none", "marker-end": "url(#arr2)" }, g);
  // head
  g = group("vocoder"); const head = cuboid(g, 1955, 12, 300, 30, "voc");
  brace(g, 1940, 2005, ["Linear head", "|X| + Δphase"], 95);
  shape(g, 1966, "513 bins × 2", "[1026 × N]");
  join(voc, head);
  // output waveform
  g = group("audio");
  el("rect", { x: 2050, y: MID - 70, width: 170, height: 140, rx: 5, fill: "#0b0b0f", stroke: "#34d399", class: "face" }, g);
  d = `M2060,${MID}`; for (let i = 0; i < 72; i++) d += ` L${2060 + i * 2.1},${MID + Math.sin(i * .95) * (5 + 40 * Math.abs(Math.sin(i / 12)))}`;
  el("path", { d, stroke: "#34d399", fill: "none", "stroke-width": 1.5 }, g);
  brace(g, 2045, 2225, ["iSTFT", "1024 · hop 256 · phase from the prior"]);
  text(2135, 470, "OUTPUT", "t", g); text(2135, 489, "24 kHz audio", "m", g);
  el("path", { d: `M${head.r + 4},${MID} L2046,${MID}`, stroke: "#9b9ba6", "stroke-width": 1.6, "marker-end": "url(#arr)" });
  // pitch conditioning
  el("path", { d: `M960,${MID + 130} C990,520 1300,524 1474,524`, stroke: "#fbbf24", "stroke-dasharray": "4 4", fill: "none", "marker-end": "url(#arr3)" });
  text(1215, 548, "predicted pitch drives the pulses and the prior", "s");
  // arrow markers
  const defs = el("defs", {}); [["arr", "#9b9ba6"], ["arr2", "#7dd3fc"], ["arr3", "#fbbf24"]].forEach(([id, c]) => {
    const m = el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
    el("path", { d: "M0,0 L10,5 L0,10 z", fill: c }, m); });
  // groups below
  function bracket(x1, x2, label, sub, color) {
    el("path", { d: `M${x1},572 L${x1},592 L${x2},592 L${x2},572 M${(x1 + x2) / 2},592 L${(x1 + x2) / 2},609`, stroke: color, "stroke-width": 2, fill: "none" });
    el("path", { d: `M${(x1 + x2) / 2 - 5},603 L${(x1 + x2) / 2},611 L${(x1 + x2) / 2 + 5},603`, stroke: color, "stroke-width": 2, fill: "none" });
    text((x1 + x2) / 2, 633, label, "t"); text((x1 + x2) / 2, 652, sub, "s");
  }
  bracket(15, 225, "Text front end", "rules + eSpeak NG · no parameters", "#9b9ba6");
  bracket(240, 1215, "Voice model", "4.54M parameters · what to say and how", "#8b7bf0");
  bracket(1245, 1590, "Rendering", "signal processing · 26k parameters", "#7dd3fc");
  bracket(1625, 2230, "Neural vocoder", "2.22M parameters · the final sound", "#34d399");
  return mark;
};
