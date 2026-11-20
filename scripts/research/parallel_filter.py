"""Parallel voice-conversion pilot: Lorenzo says the user's sentences with the user's durations and pitch,
so the two log-mels align frame by frame; fit closed-form filters (global bias, per-phoneme bias, ridge
linear map) from Lorenzo's mel to the user's, then apply them to new Lorenzo sentences."""
import json, sys, numpy as np, torch, soundfile as sf
from tiny_voca.infer import load_model, match_frontend, speak
from tiny_voca.vocoder import load_vocoder
from tiny_voca.losses import VocosMel
from tiny_voca.text import phonemize
S = sys.argv[1]; O = f"{S}/oracle2"
CK = "runs/voice-six/inference-139000.pt"; match_frontend(CK); m = load_model(CK).eval()
voc = load_vocoder("runs/vocoder-four/vocoder-voicegain.pt").eval(); vm = VocosMel()
EM4 = torch.tensor([m.config.speakers.index("EM4")]); hop = m.config.hop
def logmel(w): return vm(w[None].float())[0]          # [frames, 100]
@torch.no_grad()
def lorenzo_forced(row, feat):
    tok = torch.tensor([row["tokens"]]); st = torch.tensor([row["stress"]]); dur = torch.tensor([row["durations"]])
    hidden, mask = m.encode(tok, st, torch.tensor([tok.shape[1]]), EM4)
    frames, fm, fl = m.expand(hidden, dur, mask, EM4)
    ph, pred = m.predict_prosody(frames, fm)
    user = feat["prosody"][: frames.shape[1]].float()[None]
    pros = torch.stack((torch.where(user[..., 1] > 0, user[..., 0], pred[..., 0]), user[..., 1], pred[..., 2]), -1)
    pros = m.hard_prosody(pros)
    ac = m.decode(ph, pros, fm)
    noise = torch.randn(int(fl[0]) * hop, generator=torch.Generator().manual_seed(0))[None]
    wave = m.vocoder(ac, m.frequency(pros[..., 0]), fl, noise=noise, voiced=(pros[..., 1] > 0).float())[0]
    owner = torch.repeat_interleave(torch.arange(tok.shape[1]), dur[0])
    return wave, pros[0], owner, tok[0], pred[0, :, 0][(pred[0, :, 1] > 0)]
rows = [json.loads(l) for l in open(f"{O}/annotations/train.jsonl")]
X, Y, P, V, groups = [], [], [], [], []
for g, row in enumerate(rows):
    feat = torch.load(f"{O}/features/{row['path']}", weights_only=False)
    wave, pros, owner, tok, own_f0 = lorenzo_forced(row, feat)
    xl, yl = logmel(wave), logmel(feat["wave"])
    n = min(len(xl), len(yl), len(owner)); speech = (yl[:n].mean(1) > yl[:n].mean(1).max() - 8).numpy()
    X.append(xl[:n].numpy()[speech]); Y.append(yl[:n].numpy()[speech]); P.append(tok[owner[:n]].numpy()[speech]); V.append(own_f0.numpy())
    groups.append(np.full(speech.sum(), g))
X, Y, P, G = map(np.concatenate, (X, Y, P, groups)); lorenzo_f0 = np.concatenate(V)
user_f0 = np.concatenate([torch.load(f"{O}/features/{r['path']}", weights_only=False)["prosody"][:, 0].numpy()[torch.load(f"{O}/features/{r['path']}", weights_only=False)["prosody"][:, 1].numpy() > 0] for r in rows])
def fit(Xa, Ya, Pa, kind, lam=30.0):
    b = (Ya - Xa).mean(0)
    if kind == "global": return {"b": b}
    if kind == "phone":
        tab = {p: (Ya[Pa == p] - Xa[Pa == p]).mean(0) * (k := (Pa == p).sum()) / (k + 20) + b * 20 / (k + 20) for p in np.unique(Pa)}
        return {"b": b, "tab": tab}
    Xm, Ym = Xa.mean(0), Ya.mean(0); Xc = Xa - Xm
    W = np.linalg.solve(Xc.T @ Xc + lam * len(Xa) / 100 * np.eye(100), Xc.T @ (Ya - Ym))
    out = {"W": W, "Xm": Xm, "Ym": Ym}
    if kind == "linear+phone":
        r = Ya - ((Xa - Xm) @ W + Ym); out["tab"] = {p: r[Pa == p].mean(0) * (k := (Pa == p).sum()) / (k + 20) for p in np.unique(Pa)}
    return out
def apply(f, Xa, Pa):
    if "W" in f: out = (Xa - f["Xm"]) @ f["W"] + f["Ym"]
    else: out = Xa + f["b"]
    if "tab" in f and "W" not in f: out = Xa + np.stack([f["tab"].get(p, f["b"]) for p in Pa])
    elif "tab" in f: out = out + np.stack([f["tab"].get(p, np.zeros(100) if "W" in f else f["b"]) for p in Pa])
    return out
kinds = ["none", "global", "phone", "linear", "linear+phone"]
report = {}
for k in kinds:     # leave-one-clip-out mel error (dB-ish log units) on the user's held-out clip
    errs = []
    for g in np.unique(G):
        tr, te = G != g, G == g
        pred = X[te] if k == "none" else apply(fit(X[tr], Y[tr], P[tr], k), X[te], P[te])
        errs.append(np.abs(pred - Y[te]).mean())
    report[k] = round(float(np.mean(errs)), 3)
print(json.dumps({"held_out_clip_mel_L1": report, "frames": int(len(X))}))
shift = float(np.log(126.1 / 113.6)); rng = 1.0   # medians measured on both voices with the same tracker (WORLD harvest)
print(json.dumps({"pitch_shift_semitones": round(shift * 12 / np.log(2), 2), "pitch_range": round(rng, 2)}))
T = ["Questa è una prova del modello tiny-voca, addestrato da zero sulla lingua italiana.", "Domani pomeriggio andremo al mare, se il tempo lo permette.", "Hai già prenotato il ristorante per domani sera?"]
filters = {k: fit(X, Y, P, k) for k in kinds if k != "none"}
outs = {k: [] for k in kinds}
with torch.no_grad():
    for i, t in enumerate(T):
        w, phones, d = speak(m, t, "EM4", pitch_shift=shift, pitch_range=float(np.clip(rng, 0.5, 2)), seed=i)
        n = min(w.shape[-1] // hop, int(d["durations"][0].sum())); xl = logmel(w[: n * hop]).numpy()        # n + 1 frames, as vocode() feeds the vocoder
        owner = torch.repeat_interleave(torch.arange(len(phones.ids)), d["durations"][0]).numpy(); owner = np.pad(owner, (0, max(0, n - len(owner))), mode="edge")[:n]; owner = np.append(owner, owner[-1])[: len(xl)]; Pt = np.array(phones.ids)[owner]
        pr = d["prosody"][0, :n]
        for k in kinds:
            mel = xl if k == "none" else apply(filters[k], xl, Pt)
            a = voc(torch.tensor(mel, dtype=torch.float32).T[None], m.frequency(pr[:, 0])[None].float(), (pr[:, 1] > 0).float()[None], voice="EM4")[0]
            outs[k] += [a.numpy(), np.zeros(12000)]
for k, parts in outs.items():
    x = np.concatenate(parts); x = x * min(0.1 / np.sqrt(np.mean(x ** 2)), 0.95 / np.abs(x).max())
    sf.write(f"{S}/myvoice/filter-{k.replace('+', '-')}.wav", x, 24000, subtype="PCM_16")
torch.save(filters, f"{S}/oracle2/filters.pt"); print("ok")
