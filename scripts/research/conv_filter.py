"""Tiny convolutional voice filter: Lorenzo log-mel -> user log-mel, residual on the closed-form
linear + per-phoneme filter. Trained on frame-aligned pairs (Lorenzo forced to the target's durations
and pitch). Variants: user clips only, user + Qwen3-TTS clips in the user's voice."""
import json, sys, time, numpy as np, torch, torch.nn as nn, soundfile as sf
from tiny_voca.infer import load_model, match_frontend, speak
from tiny_voca.vocoder import load_vocoder
from tiny_voca.losses import VocosMel
S = sys.argv[1]
torch.manual_seed(0); torch.set_num_threads(8)
CK = "runs/voice-six/inference-139000.pt"; match_frontend(CK); m = load_model(CK).eval()
voc = load_vocoder("runs/vocoder-four/vocoder-voicegain.pt").eval(); vm = VocosMel()
EM4 = torch.tensor([m.config.speakers.index("EM4")]); hop = m.config.hop
def logmel(w): return vm(w[None].float())[0]
@torch.no_grad()
def pair(row, feat):
    tok = torch.tensor([row["tokens"]]); st = torch.tensor([row["stress"]]); dur = torch.tensor([row["durations"]])
    hidden, mask = m.encode(tok, st, torch.tensor([tok.shape[1]]), EM4)
    frames, fm, fl = m.expand(hidden, dur, mask, EM4)
    ph, pred = m.predict_prosody(frames, fm)
    user = feat["prosody"][: frames.shape[1]].float()[None]
    pros = m.hard_prosody(torch.stack((torch.where(user[..., 1] > 0, user[..., 0], pred[..., 0]), user[..., 1], pred[..., 2]), -1))
    noise = torch.randn(int(fl[0]) * hop, generator=torch.Generator().manual_seed(0))[None]
    wave = m.vocoder(m.decode(ph, pros, fm), m.frequency(pros[..., 0]), fl, noise=noise, voiced=(pros[..., 1] > 0).float())[0]
    owner = torch.repeat_interleave(torch.arange(tok.shape[1]), dur[0])
    x, y = logmel(wave), logmel(feat["wave"]); n = min(len(x), len(y), len(owner))
    y = y[:n]; speech = y.mean(1) > y.mean(1).max() - 8
    return {"X": x[:n].numpy(), "Y": y.numpy(), "P": tok[0][owner[:n]].numpy(), "M": speech.numpy()}
def load(root):
    rows = [json.loads(l) for l in open(f"{root}/annotations/train.jsonl")]
    return [pair(r, torch.load(f"{root}/features/{r['path']}", weights_only=False)) for r in rows]
user, teacher = load(f"{S}/oracle2"), load(f"{S}/teacher-pairs")
held = max(range(len(user)), key=lambda i: len(user[i]["X"]))      # longest user clip = validation
print(json.dumps({"user_clips": len(user), "teacher_clips": len(teacher), "user_frames": int(sum(c["M"].sum() for c in user)),
                  "teacher_frames": int(sum(c["M"].sum() for c in teacher))}), flush=True)
def fit_base(clips, lam=30.0):
    X = np.concatenate([c["X"][c["M"]] for c in clips]); Y = np.concatenate([c["Y"][c["M"]] for c in clips]); P = np.concatenate([c["P"][c["M"]] for c in clips])
    Xm, Ym = X.mean(0), Y.mean(0); Xc = X - Xm
    W = np.linalg.solve(Xc.T @ Xc + lam * len(X) / 100 * np.eye(100), Xc.T @ (Y - Ym)); r = Y - (Xc @ W + Ym)
    tab = {p: r[P == p].mean(0) * (k := (P == p).sum()) / (k + 20) for p in np.unique(P)}
    return {"W": W, "Xm": Xm, "Ym": Ym, "tab": tab}
def base(f, X, P): return (X - f["Xm"]) @ f["W"] + f["Ym"] + np.stack([f["tab"].get(p, np.zeros(100)) for p in P])
class Net(nn.Module):
    def __init__(self, c, k=5):
        super().__init__(); self.a = nn.Conv1d(100, c, k, padding=k // 2); self.b = nn.Conv1d(c, 100, 1)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)
    def forward(self, x): return x + self.b(torch.nn.functional.gelu(self.a(x)))     # x: [1, 100, n] base-filtered mel
def tensors(clips, f):
    return [(torch.tensor(base(f, c["X"], c["P"]).T[None], dtype=torch.float32), torch.tensor(c["Y"].T[None], dtype=torch.float32),
             torch.tensor(c["M"])) for c in clips]
def train(net, data, weights, steps, val=None):
    opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-2); best = (9e9, 0); curve = []
    for s in range(1, steps + 1):
        loss = sum(w * (net(x) - y)[..., mk].abs().mean() for (x, y, mk), w in zip(data, weights)) / sum(weights)
        opt.zero_grad(); loss.backward(); opt.step()
        if val is not None and s % 100 == 0:
            with torch.no_grad(): v = float(sum((net(x) - y)[..., mk].abs().mean() for x, y, mk in val) / len(val))
            curve.append((s, round(v, 4))); best = min(best, (v, s))
    return best, curve
results, final = {}, {}
for name, use_teacher in (("user", False), ("user+qwen", True)):
    tr = [c for i, c in enumerate(user) if i != held] + (teacher if use_teacher else [])
    f = fit_base(tr); val = tensors([user[held]], f)
    with torch.no_grad(): base_l1 = float((val[0][0] - val[0][1])[..., val[0][2]].abs().mean())
    w = [2.0 if i < len(user) - 1 else 1.0 for i in range(len(tr))]
    for c in (16, 48):
        net = Net(c); t0 = time.time(); (v, s), curve = train(net, tensors(tr, f), w, 2000, val)
        params = sum(p.numel() for p in net.parameters())
        results[f"{name} conv{c}"] = {"params": params, "base_linear_phone_L1": round(base_l1, 4), "conv_best_L1": round(v, 4), "best_step": s, "train_s": round(time.time() - t0, 1)}
        print(json.dumps({f"{name} conv{c}": results[f"{name} conv{c}"]}), flush=True)
        final[(name, c)] = s
# Refit on every clip with the chosen step counts, render the test sentences.
T = ["Questa è una prova del modello tiny-voca, addestrato da zero sulla lingua italiana.", "Domani pomeriggio andremo al mare, se il tempo lo permette.", "Hai già prenotato il ristorante per domani sera?"]
shift = float(np.log(126.1 / 113.6))
tests = []
with torch.no_grad():
    for i, t in enumerate(T):
        w_, phones, d = speak(m, t, "EM4", pitch_shift=shift, seed=i)
        n = min(w_.shape[-1] // hop, int(d["durations"][0].sum())); xl = logmel(w_[: n * hop]).numpy()
        owner = torch.repeat_interleave(torch.arange(len(phones.ids)), d["durations"][0]).numpy()
        owner = np.pad(owner, (0, max(0, len(xl) - len(owner))), mode="edge")[: len(xl)]
        tests.append((xl, np.array(phones.ids)[owner], d["prosody"][0, :n]))
def render(path, mel_fn):
    parts = []
    with torch.no_grad():
        for xl, Pt, pr in tests:
            mel = mel_fn(xl, Pt)
            parts += [voc(torch.tensor(mel, dtype=torch.float32)[None] if mel.shape[0] == 100 else torch.tensor(mel.T, dtype=torch.float32)[None],
                          m.frequency(pr[:, 0])[None].float(), (pr[:, 1] > 0).float()[None], voice="EM4")[0].numpy(), np.zeros(12000)]
    x = np.concatenate(parts); sf.write(path, x * min(0.1 / np.sqrt(np.mean(x ** 2)), 0.95 / np.abs(x).max()), 24000, subtype="PCM_16")
for name, use_teacher in (("user", False), ("user+qwen", True)):
    tr = user + (teacher if use_teacher else []); f = fit_base(tr)
    tag = name.replace("+", "-")
    if use_teacher: render(f"{S}/myvoice/conv-linear-{tag}.wav", lambda xl, Pt: base(f, xl, Pt))
    for c in (16, 48):
        net = Net(c); w = [2.0 if i < len(user) else 1.0 for i in range(len(tr))]
        train(net, tensors(tr, f), w, max(final[(name, c)], 100))
        render(f"{S}/myvoice/conv{c}-{tag}.wav", lambda xl, Pt: net(torch.tensor(base(f, xl, Pt).T[None], dtype=torch.float32))[0].numpy())
        torch.save({"net": net.state_dict(), "base": f}, f"{S}/oracle2/conv{c}-{tag}.pt")
print("ok")
