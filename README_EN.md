# Zundanen v2.0.0

**[Suomi](README.md) | [日本語](README_JA.md) | English**

Zundanen is a local Windows app that turns Finnish text into WAV files.

v2.0.0 replaces the speech synthesis with **its own VITS models (with FinBERT features)**. This repository includes the **Chatterbox** model (Finnish VITS, Normal only). With `Use RVC voice conversion` and an RVC model of your own it speaks in another voice (for example Zundamon). Character models with emotion styles (Normal / Ama / Tsun / Sexy) are trained with the Model Trainer from your own data; no character models are included.

The same UI includes **Batch Generation**, per-word **Accent** editing, the **Model Trainer** (RVC v2 and VITS fine-tuning) and **Voice Conversion** (WAV / folder conversion, recording, real-time conversion).

> The screenshots in `docs/` are from v1.1.0.

## Tutorial video (v1)

[https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY](https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY)

## Main changes from v1

- Speech synthesis: v1 used "Chatterbox-Finnish → RVC"; v2 uses a **Finnish VITS with emotion styles**.
- **Character**: choose a trained model.
  - **Chatterbox** (included): a Finnish VITS in the Chatterbox voice (Normal only). Turn on `Use RVC` to convert it to Zundamon or another voice.
  - **Models you train yourself** (Model Trainer → VITS): a character's voice with emotion styles. No character models are included (see “Model files”).
- **Style**: Normal / Ama / Tsun / Sexy, with adjustable Style strength.
- **Pitch / Speed / Seed**.
- **Use RVC voice conversion**: optional RVC conversion of the generated speech.
- **Batch Generation** supports Character / Style / RVC.
- **Model Trainer**: RVC (as in v1) plus VITS fine-tuning (Joint / Finnish / Style).
- New **Voice Conversion** tab: RVC conversion of a WAV or a folder, record-and-convert, in-app real-time conversion.

## Requirements

- Windows 11 x64
- WinGet (usually included in Windows 11 App Installer)
- NVIDIA GPU not required: voice generation also runs on CPU only (tested on a machine without a GPU)
- Model Trainer requires an NVIDIA CUDA GPU (developed and tested on an RTX 3060 12 GB)
- NVIDIA GPU recommended for real-time voice conversion
- Free disk space: tens of GB recommended (environments, models, outputs)
- **On the first voice generation** the app downloads the Finnish BERT model (`TurkuNLP/bert-large-finnish-cased-v1`, about 1.4 GB) from Hugging Face (internet connection needed).

## Installation

Same as v1: download with `Code → Download ZIP`, extract to a short path such as `C:\Zundanen`, then double-click `setup.bat`.

The normal setup prepares:

```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish (TTS environment; VITS and FinBERT also run here)
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
PyWORLD / WORLD vocoder
Windows-native Aalto/Kaldi runtime
Intel oneMKL runtime when required
```

Re-running setup, a full rebuild and the `doctor.ps1` check work as in v1:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

## Model files

Included in this repository:

```text
models/
└─ chatterbox.pt     Chatterbox, about 68 MB
```

- Inference-only weights (no discriminator / optimizer states), stored as float16 so that the file stays under GitHub's 100 MB limit; the app loads them in float32.
- **Not included**: character models with emotion styles (for example Zundamon or Shikoku Metan) and RVC models (`.pth` / `.index`). Their training data belongs to the respective rights holders. Train your own with the Model Trainer from data you are allowed to use; a trained model is added to `Character` automatically.
- **Character models** (not part of this repository): the Zundamon and Shikoku Metan emotion models and RVC models are distributed separately for research on [Hugging Face (Zundanen/finnish-emotion-vits)](https://huggingface.co/Zundanen/finnish-emotion-vits). Non-commercial research use only; follow the [terms of the SSS multimodal database](https://zunko.jp/multimodal_dev/login.php) when you use them. Put a character file `<name>.pt` in `models/characters` and its RVC model (`<name>.pth` and `.index`) in `runtime/rvc/exports/<name>/`, then restart the app: the character appears in `Character`, and the RVC model in the RVC model lists. Each character is one file.
- Put RVC models in `runtime/rvc/exports/<name>/` as in v1.

## Start

As in v1: `Zundanen.exe` (recommended) or `run_ui.bat` (browser, `http://127.0.0.1:8765`).

## Voice Generation

1. Choose a `Character`
2. Choose `Style` and `Style strength` (Chatterbox: Normal only)
3. Enter Finnish text
4. Adjust `Pitch` (semitones from the character's base level, ±10) and `Speed` (1.0 = normal, larger = faster) if needed
5. Check the `Seed` (with `Fix seed` OFF every generation gets a new seed)
6. Optionally turn on `Use RVC voice conversion` and choose the RVC model and `Pitch` / `Index rate` / `Protect`
7. Click `Generate voice`
8. Edit the Accent if needed and save with `Save As...`

### Recommended settings

| Purpose | Character | Style | RVC |
|---|---|---|---|
| Clearest Normal | Chatterbox | Normal | your RVC model (Zundamon: Pitch **+8**), Index rate 0.1, Protect 0.15 |
| Emotion styles | a model you trained | Ama / Tsun / Sexy | none |

- When you select Chatterbox, an RVC Pitch of 0 is set to +8 (Zundamon's level) automatically. Chatterbox has no pitch control, so the upper `Pitch` slider is disabled for it.
- A high RVC `Index rate` makes the pronunciation more Japanese-like; about 0.1 is recommended for Finnish.

### Pronunciation (development measurement)

Character error rate (CER, lower is better) of Whisper large-v2 on 60 Finnish Tatoeba sentences never used in training, 2 seeds each.

| Voice | CER |
|---|---:|
| Original Chatterbox-Finnish audio (practical ceiling) | 2.5% |
| Chatterbox | 2.3% |
| Chatterbox + Zundamon RVC (+8, Protect 0.15) | 4.7% |
| Zundamon Normal | 7.1% |
| Zundamon Ama / Tsun / Sexy | 9.2% / 7.8% / 8.1% |

The Zundamon rows are the Zundamon emotion model distributed separately for research on Hugging Face.

Words with h + consonant (e.g. *tyhjä*, *kissanhiekka*) are known to be weak in Zundamon's voice.

## Batch Generation

Generates several texts in a row with the same Character / Style / Style strength / Pitch / Speed / Seed / RVC as Voice Generation. Chatterbox can be selected as well.

CSV columns:

```text
filename,text,style,strength,speed,seed
```

Only `text` is required; empty fields use the screen settings, and each row can have its own style (`speed` 1.0 = normal; old `length_scale` columns are still read).

## Model Trainer

Switch with `VITS` / `RVC` at the top.

- **RVC**: as in v1 (Character name, Dataset folder, Epochs, Batch size, Workers, Fresh training). Creates `.pth` and `.index` in `runtime/rvc/exports/<Character>/`.
- **VITS** (advanced):
  - `Joint training` / `Finnish fine-tuning` train the base model → style branch → voice-quality deltas and add the result to Character under its Model name. **They continue a model you trained before (none is included); start with `New model (from scratch)` below.**
  - `Style fine-tuning` adds a new style to a chosen model from paired Normal and style recordings (WAV + LAB, matched by number, at least 20 pairs).
  - **New model (from scratch)**: choose `New model (from scratch)` as the `Base model` of `Joint training` to train from random weights. Inputs:
    - **Finnish folder**: Finnish WAV files with same-name TXT (at least 50 pairs; about 20 hours or more for good pronunciation)
    - **Emotion data**: one row per speaker and style (WAV folder, LAB folder – empty = the same folder –, speaker name and style Normal / Ama / Tsun / Sexy); add rows with `+`. Every speaker needs a Normal row. A LAB has the same file name as its WAV; a row without its own LABs uses the same-name LAB of another row of the same style.
    - **Bridge RVC model (.pth)** (Browse…) and **Character**: every Finnish sentence of the folder is converted to this RVC voice (each at two pitch levels, about 1 s per file; 20,000 sentences ≈ 11 h) so the character learns Finnish. Character is chosen from the entered speakers (preselected when the file name matches, e.g. Zundamon.pth → zundamon). `RVC bridge share` sets how often the converted sentences are used in training. Its .index next to it is used if present; the pitch shift is set automatically (Index 0.1 / Protect 0.15).
    - Epoch (VITS) defaults to 35. This takes much longer than continuing a model (often a day or more).
  - Training length is given in epochs (Epoch (VITS) default 4.5, Epoch (Style) default 80); Batch size and Workers can be set for your PC.

## Voice Conversion

- Left: RVC model and Pitch / Index rate / Protect
- WAV conversion: one file or every audio file in a folder (saved to `outputs/voice_conversion/`; originals are not overwritten)
- Recording: record, convert and save with `Save As...`
- Realtime voice conversion: choose input and output devices and press Start / Stop (no extra window; latency about 0.3 s)

## Accent

Same as v1.1.0 (Windows-native Aalto/Kaldi forced alignment, editing with WORLD/PyWORLD; no WSL2 / Docker Desktop).

## UI language

Help texts switch between `Suomi / 日本語 / English`. Buttons and field names stay in English.

## Not included in GitHub

`.gitignore` excludes:

```text
runtime/  config.json  outputs/  temp/  __pycache__/  *.pt (except models/chatterbox.pt)  *.pth  *.index  *.safetensors  *.wav  local_state/
```

## Rights of voices, characters and training data

The MIT license of Zundanen does not grant any rights to third-party character voices, training data, trained models or RVC models.

The included Chatterbox model (`models/chatterbox.pt`) was trained only on Finnish speech generated with Chatterbox-Finnish (MIT) from Finnish Tatoeba sentences (CC BY 2.0 FR). Its voice prompt was the reference audio bundled with Chatterbox-Finnish (`samples/reference_finnish.wav`, from Mozilla Common Voice, CC0 1.0). No Japanese character corpora were used for it.

Models you train yourself and RVC models are subject to the terms of their training data. Check the terms of each rights holder before using, publishing or redistributing them (`THIRD_PARTY_NOTICES.md`).

## Upstream projects

- Finnish-NLP/Chatterbox-Finnish: https://huggingface.co/Finnish-NLP/Chatterbox-Finnish
- ResembleAI Chatterbox: https://github.com/resemble-ai/chatterbox
- VITS: https://github.com/jaywalnut310/vits
- TurkuNLP FinBERT: https://huggingface.co/TurkuNLP/bert-large-finnish-cased-v1
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Aalto Finnish Forced Alignment: https://github.com/aalto-speech/finnish-forced-alignment
- Lingsoft aalto-kaldi-align-elg: https://github.com/lingsoft/aalto-kaldi-align-elg
- WORLD: https://github.com/mmorise/World
- PyWORLD: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
- FFmpeg: https://ffmpeg.org/
- Tatoeba: https://tatoeba.org/

See also `THIRD_PARTY_NOTICES.md`.
