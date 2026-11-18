# Run scripts (record of the 16–18 November 2026 training rounds)

Copies of the scripts that launched, alternated and watched every run described in
`docs/analysis/quality-journey.md`. They run from the repository root and were used
from `logs/` (git-ignored, also holding the run logs).

- `EXTRA_LIBS_DIR`: a folder with packages kept outside the training environment,
  installed with `pip install --target`:
  - `pylib/`: `vocos einops encodec`, with `--no-deps`;
  - `qwenlib/`: `qwen-tts accelerate==1.12.0 sox einops`, with `--no-deps`.
- `alternate.sh` runs one GPU job at a time, in blocks; `watchdog.sh` restarts it
  and the playground and alerts on stalls, crashes and a nearly full GPU.
- Runs that failed or were superseded are kept for reference:
  - `it1-head*.sh` (spectral head);
  - `it1-vocos.sh` (direct mel, Phase 1);
  - `voice-mel-pilot.sh` (direct-route pilot);
  - `vocoder-distill.sh` (Vocos distillation);
  - `vocoder-ft.sh` (separate fine-tune, folded into `vocoder-prior.sh`).
