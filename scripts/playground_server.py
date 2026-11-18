"""Local listening playground for work in progress. Local only.

    .venv/bin/python scripts/playground_server.py --model runs/it1-full/inference.pt --vocoder runs/vocoder/vocoder.pt

Serves a page on 0.0.0.0:8790 where text is spoken by an acoustic checkpoint and
rendered by one of: the newest checkpoint of the neural vocoder (re-read whenever
the file changes, so it follows a training run), the model's own source-filter
output, or the pretrained Vocos as a reference (only if the ``vocos`` package is
importable). Synthesis runs on the CPU with few threads so a GPU training run is
not disturbed. No authentication: run it only on a trusted network.
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tiny_voca.infer import load_model, match_frontend, speak  # noqa: E402
from tiny_voca.losses import VocosMel  # noqa: E402
from tiny_voca.vocoder import load_vocoder  # noqa: E402

PAGE = """<!doctype html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>tiny-voca playground</title>
<style>
:root{--bg:#f7f7f5;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e2e2dc;--accent:#2f6fde}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--muted:#a3a39c;--card:#20201e;--line:#34342f;--accent:#7aa7ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:720px;margin:0 auto;padding:20px 16px 48px}h1{font-size:1.3rem;margin:0 0 4px}
.muted{color:var(--muted);font-size:.9rem}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:16px}
textarea{width:100%;min-height:110px;font:inherit;padding:10px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
.row{display:flex;flex-wrap:wrap;gap:12px;margin-top:12px;align-items:end}label{display:flex;flex-direction:column;font-size:.85rem;color:var(--muted);gap:4px}
select,input[type=range]{font:inherit;min-width:140px}select{padding:6px;border-radius:6px;border:1px solid var(--line);background:var(--bg);color:var(--fg)}
button{font:inherit;padding:9px 18px;border-radius:8px;border:0;background:var(--accent);color:#fff;cursor:pointer}button:disabled{opacity:.5}
.stats{display:flex;gap:12px;margin:10px 0 4px}.stats div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:8px 14px;display:flex;flex-direction:column}
.stats b{font-size:1.4rem}.take{border-top:1px solid var(--line);padding:10px 0}.take:first-child{border-top:0}.take p{margin:0 0 6px}audio{width:100%}
</style></head><body><main>
<h1>tiny-voca playground</h1>
<div class="stats"><div><span class="muted">Voce</span><b id="voicek">–</b></div><div><span class="muted">Vocoder</span><b id="vock">–</b></div></div>
<div class="muted" id="info">loading…</div>
<div class="card">
<div class="row" style="margin:0 0 10px"><label style="flex:1">Frasi di prova<select id="sample"><option value="">— scegli —</option><option value="0">Domanda sì/no (intonazione ascendente)</option><option value="1">Domanda aperta</option><option value="2">Esclamazioni</option><option value="3">Doppie consonanti</option><option value="4">Accento che cambia significato</option><option value="5">Gli, gn, sc</option><option value="6">Elisioni e apostrofi</option><option value="7">Parole accentate</option><option value="8">Numeri e date</option><option value="9">Prezzi e misure</option><option value="10">Sigle e abbreviazioni</option><option value="11">Parole straniere</option><option value="12">Frase lunga con incisi</option><option value="13">Elenco</option><option value="14">Discorso diretto</option><option value="15">Scioglilingua</option><option value="16">Emozione: tristezza</option><option value="17">Emozione: rabbia</option><option value="18">Nomi propri</option><option value="19">Interiezioni</option></select></label>
<button id="all" type="button">Prova tutte</button></div>
<textarea id="text">Buongiorno a tutti, oggi il cielo è sereno e la temperatura è piacevole. Ti ho già detto che domani partiamo presto?</textarea>
<div class="row">
<label>Vocoder<select id="vocoder"></select></label>
<label>Voce<select id="voice"></select></label>
<label>Velocità <span id="speedv">1.00</span><input id="speed" type="range" min="0.7" max="1.4" step="0.05" value="1"></label>
<label>Intonazione <span id="rangev">1.00</span><input id="range" type="range" min="0.6" max="1.8" step="0.05" value="1"></label>
<label id="corrbox" style="display:none;flex-direction:row;align-items:center;gap:6px"><input id="corr" type="checkbox"> Correzione mel <span id="corrv"></span></label>
<label id="emobox" style="display:none">Emozione<select id="emotion"></select></label>
<label id="intbox" style="display:none">Intensità <span id="intv">0.70</span><input id="intensity" type="range" min="0" max="1" step="0.05" value="0.7"></label>
<button id="go">Parla</button></div></div>
<div class="card" id="takes"><div class="muted">Le prove appaiono qui, la più recente in alto.</div></div>
</main><script>
const $=id=>document.getElementById(id);let first=true;
const SAMPLES=["Hai già chiamato tua sorella? Viene anche lei stasera?", "Perché non me l'hai detto prima? Dove pensavi di andare?", "Che bella sorpresa! Non ci posso credere, sei proprio tu!", "Il fato e il fatto. La pala e la palla. Il caro e il carro. La sete e il sette.", "Ancora una volta gettò l'àncora. I prìncipi hanno sempre difeso i loro princìpi.", "Mio figlio mangia gli gnocchi con la sciarpa di seta e la maglia nuova.", "L'albero dell'orto è all'ombra, quell'uomo c'è sempre, dov'è l'acqua?", "Perché la città è più bella così? Però lunedì andrò là, e poi tornerò qui.", "Il 3 marzo 2025 alle 15:30 erano presenti 1.250 persone, il 12% in più dell'anno scorso.", "Costa 12,50 euro al chilo; la stanza misura 4 metri per 3,5.", "La RAI e l'ONU hanno parlato con il dott. Rossi e la sig.ra Bianchi a Roma, in via XX Settembre.", "Nel weekend ho aggiornato il computer e controllato le email del team di marketing.", "Quando arrivammo, dopo un viaggio lungo e, a dire il vero, piuttosto faticoso, la casa era ancora chiusa, e nessuno sapeva dove fossero le chiavi.", "Abbiamo comprato pane, latte, uova, formaggio, pomodori e, naturalmente, un po' di cioccolato.", "Mi guardò e disse: «Non tornerò mai più». Poi, senza aggiungere altro, se ne andò.", "Trentatré trentini entrarono a Trento, tutti e trentatré trotterellando.", "Mi dispiace tanto. Avrei voluto esserci, ma ormai è troppo tardi.", "Basta! Te l'ho ripetuto mille volte e non mi ascolti mai!", "Giuseppe Verdi nacque a Busseto, vicino a Parma, e Giacomo Leopardi a Recanati.", "Ehm, beh, insomma... mah, vedremo. Ah, dimenticavo: ciao!"];
const LABELS=["Domanda sì/no (intonazione ascendente)", "Domanda aperta", "Esclamazioni", "Doppie consonanti", "Accento che cambia significato", "Gli, gn, sc", "Elisioni e apostrofi", "Parole accentate", "Numeri e date", "Prezzi e misure", "Sigle e abbreviazioni", "Parole straniere", "Frase lunga con incisi", "Elenco", "Discorso diretto", "Scioglilingua", "Emozione: tristezza", "Emozione: rabbia", "Nomi propri", "Interiezioni"];
async function info(){const r=await fetch('info');const j=await r.json();
 const k=v=>v==null?'–':`${(v/1000).toFixed(v%1000?1:0)}k steps`;
 $('voicek').textContent=k(j.model_step);$('vock').textContent=k(j.vocoder_step);
 $('info').textContent=`voice: ${j.model_name} · vocoder: ${j.vocoder_name ?? '–'}, updated ${j.vocoder_updated ?? '–'}`;
 if(first){for(const v of j.vocoders){const o=document.createElement('option');o.value=v.id;o.textContent=v.label;$('vocoder').append(o)}
 first=false}
 const it={neutral:'neutro',joy:'gioia',sadness:'tristezza',anger:'rabbia',fear:'paura',surprise:'sorpresa',disgust:'disgusto'};
 const ce=$('emotion').value;$('emotion').innerHTML='';for(const e of (j.emotions||[])){const o=document.createElement('option');o.value=e;o.textContent=it[e]||e;$('emotion').append(o)}
 if((j.emotions||[]).includes(ce))$('emotion').value=ce;
 $('emobox').style.display=$('intbox').style.display=(j.emotions||[]).length?'':'none';
 $('corrbox').style.display=j.correction_step!=null?'flex':'none';$('corrv').textContent=j.correction_step!=null?`(pilot, ${j.correction_step} steps)`:'';
 const cur=$('voice').value;$('voice').innerHTML='';for(const v of j.voices){const o=document.createElement('option');o.value=v;o.textContent=v;$('voice').append(o)}
 if(j.voices.includes(cur))$('voice').value=cur}
$('speed').oninput=()=>$('speedv').textContent=Number($('speed').value).toFixed(2);
$('range').oninput=()=>$('rangev').textContent=Number($('range').value).toFixed(2);
$('intensity').oninput=()=>$('intv').textContent=Number($('intensity').value).toFixed(2);
$('sample').onchange=()=>{if($('sample').value!=='')$('text').value=SAMPLES[$('sample').value]};
async function say(text,tag,autoplay){const body={text,vocoder:$('vocoder').value,voice:$('voice').value,speed:Number($('speed').value),pitch_range:Number($('range').value),emotion:$('emotion').value||null,intensity:Number($('intensity').value),correction:$('corr').checked};
 const r=await fetch('synth',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 if(!r.ok)throw new Error(await r.text());const blob=await r.blob();
 const d=document.createElement('div');d.className='take';
 const label=$('vocoder').selectedOptions[0].textContent;
 d.innerHTML=`<p class="muted"></p><audio controls src="${URL.createObjectURL(blob)}"></audio>`;
 d.querySelector('p').textContent=`${tag?tag+' · ':''}${label} · ${r.headers.get('X-Detail')} · “${text.slice(0,70)}${text.length>70?'…':''}”`;
 const t=$('takes');if(t.firstElementChild&&t.firstElementChild.classList.contains('muted'))t.firstElementChild.remove();
 t.prepend(d);if(autoplay)d.querySelector('audio').play()}
$('go').onclick=async()=>{const b=$('go');b.disabled=true;b.textContent='…';
 try{const i=$('sample').value;await say($('text').value,i!==''&&$('text').value===SAMPLES[i]?LABELS[i]:'',true);info()}
 catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Parla'}};
$('all').onclick=async()=>{const b=$('all');b.disabled=true;
 try{for(let i=SAMPLES.length-1;i>=0;i--){b.textContent=`${SAMPLES.length-i}/${SAMPLES.length}`;await say(SAMPLES[i],LABELS[i],false)}info()}
 catch(e){alert(e.message)}finally{b.disabled=false;b.textContent='Prova tutte'}};
info();setInterval(info,60000);
</script></body></html>"""


class Engine:
    def __init__(self, model_paths, vocoder_path, threads):
        torch.set_num_threads(threads)
        self.model_paths = [Path(p) for p in model_paths]   # first existing one wins; re-read when it changes
        self.model, self.model_key, self.model_step = None, None, None
        self.current_model()
        self.vocoder_paths = [Path(p) for p in vocoder_path]   # first existing one wins
        self.vocoder, self.vocoder_mtime, self.vocoder_step = None, None, None
        self.features = VocosMel()
        self.lock = threading.Lock()
        try:
            from vocos import Vocos
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
        except Exception:      # optional reference only
            self.vocos = None

    def current_model(self):
        path = next((p for p in self.model_paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError("no acoustic checkpoint found")
        key = (str(path), path.stat().st_mtime)
        if key != self.model_key:
            match_frontend(str(path))
            self.model = load_model(str(path)).eval()
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.model_key, self.model_step = key, state.get("step")
        return self.model

    def current_correction(self):
        """Optional residual mel correction (scripts/pilot_mel_correction.py), re-read when it changes."""
        path = Path("runs/mel-correction-pilot/correction.pt")
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        if mtime != getattr(self, "correction_mtime", None):
            from tiny_voca.model import MelCorrection
            state = torch.load(path, map_location="cpu", weights_only=True)
            module = MelCorrection(state["channels"], low_hz=state["low_hz"])
            module.load_state_dict(state["correction"])
            self.correction, self.correction_mtime, self.correction_step = module.eval(), mtime, state.get("steps")
        return self.correction

    def current_vocoder(self):
        path = next((p for p in self.vocoder_paths if p.exists()), None)
        if path is None:
            raise FileNotFoundError("no vocoder checkpoint found")
        mtime = (str(path), path.stat().st_mtime)
        if mtime != self.vocoder_mtime:
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.vocoder = load_vocoder(path)
            self.vocoder_mtime, self.vocoder_step = mtime, state.get("step")
            self.vocoder_name = path.parent.name
        return self.vocoder

    def info(self):
        with self.lock:
            self.current_vocoder()
            model = self.current_model()
        vocoders = [{"id": "ours", "label": "Our vocoder (latest)"}, {"id": "dsp", "label": "Old (source-filter)"}]
        if self.vocos is not None:
            vocoders.append({"id": "vocos", "label": "Vocos (reference)"})
        return {"model": f"{self.model_key[0]} (step {self.model_step})", "model_step": self.model_step,
                "model_name": Path(self.model_key[0]).parent.name, "emotions": list(model.config.emotions), "voices": list(model.config.speakers), "vocoders": vocoders,
                "vocoder_step": self.vocoder_step,
                "vocoder_updated": time.strftime("%H:%M", time.localtime(self.vocoder_mtime[1])) if self.vocoder_mtime else None,
                "vocoder_name": getattr(self, "vocoder_name", None),
                "correction_step": self.correction_step if self.current_correction() is not None else None}

    def synthesize(self, text, voice, vocoder, speed, pitch_range=1.0, emotion=None, intensity=0.0, correction=False):
        with self.lock, torch.no_grad():
            start = time.time()
            model = self.current_model()
            emotion = emotion if emotion in model.config.emotions else None
            wave, _, details = speak(model, text, voice if voice in model.config.speakers else None, speed=speed, pitch_range=pitch_range,
                                     emotion=emotion, intensity=intensity)
            detail = "source-filter"
            if vocoder in ("ours", "vocos"):
                mel = self.features(wave[None])
                corrector = self.current_correction() if correction else None
                if corrector is not None and corrector.project.in_features == model.config.channels + mel.shape[-1]:
                    frames = wave.shape[-1] // model.config.hop
                    mel = corrector(details["acoustic"][:, :frames].float(), mel[:, :frames + 1])
                if vocoder == "ours":
                    ours = self.current_vocoder()
                    if ours.config.prior:
                        # The harmonic-prior vocoder takes the voice model's own pitch and voicing per frame.
                        frames = wave.shape[-1] // model.config.hop
                        prosody = details["prosody"][0, :frames]
                        f0 = model.frequency(prosody[:, 0])[None].float()
                        voiced = (prosody[:, 1] > 0).float()[None]
                        torch.manual_seed(0)
                        wave = ours(mel.transpose(1, 2), f0, voiced, voice=voice)[0]
                    else:
                        wave = ours(mel.transpose(1, 2))[0]
                    detail = f"vocoder {getattr(self, 'vocoder_name', '')} {self.vocoder_step / 1000:g}k"
                else:
                    wave = self.vocos.decode(mel.transpose(1, 2))[0]
                    detail = "Vocos"
            audio = wave.numpy()
            seconds = time.time() - start
            emotional = f", {emotion} {intensity:.2f}" if emotion else ""
            if correction and vocoder in ("ours", "vocos") and self.current_correction() is not None:
                emotional += f", mel correction {self.correction_step}"
            detail = f"voice {self.model_step / 1000:g}k, " + detail + emotional + f", intonation {pitch_range:.2f}, {len(audio) / 24000:.1f}s audio in {seconds:.2f}s"
            audio = audio / max(1e-6, float(np.abs(audio).max())) * 0.8
            buffer = io.BytesIO()
            sf.write(buffer, audio.astype(np.float32), 24000, format="WAV", subtype="PCM_16")
            return buffer.getvalue(), detail


def handler_for(engine):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, kind, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            elif self.path == "/info":
                self._send(200, json.dumps(engine.info()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain")

        def do_POST(self):
            if self.path != "/synth":
                return self._send(404, b"not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 20000:
                    raise ValueError("request too large")
                request = json.loads(self.rfile.read(length))
                text = str(request.get("text", "")).strip()
                if not text or len(text) > 600:
                    raise ValueError("text must be 1-600 characters")
                vocoder = request.get("vocoder", "ours")
                if vocoder not in ("ours", "dsp", "vocos") or (vocoder == "vocos" and engine.vocos is None):
                    raise ValueError("unknown vocoder")
                speed = min(1.5, max(0.6, float(request.get("speed", 1.0))))
                pitch_range = min(2.0, max(0.5, float(request.get("pitch_range", 1.0))))
                intensity = min(1.0, max(0.0, float(request.get("intensity") or 0.0)))
                audio, detail = engine.synthesize(text, request.get("voice"), vocoder, speed, pitch_range, request.get("emotion"), intensity,
                                                  bool(request.get("correction")))
                self._send(200, audio, "audio/wav", {"X-Detail": detail})
            except Exception as error:      # report to the page instead of dropping the connection
                self._send(400, str(error).encode(), "text/plain; charset=utf-8")

        def log_message(self, format, *args):
            print(f"{self.address_string()} {format % args}", flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, nargs="+", help="Acoustic checkpoints in order of preference (first existing is used)")
    parser.add_argument("--vocoder", required=True, nargs="+", help="Vocoder checkpoints in order of preference (first existing is used)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    engine = Engine(args.model, args.vocoder, args.threads)
    print(json.dumps({"listening": f"http://{args.host}:{args.port}", **engine.info()}), flush=True)
    ThreadingHTTPServer((args.host, args.port), handler_for(engine)).serve_forever()


if __name__ == "__main__":
    main()
