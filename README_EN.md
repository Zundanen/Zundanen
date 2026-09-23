# Zundanen v1.0.0

**[Suomi](README.md) | [日本語](README_JA.md) | English**

Zundanen is a local Windows application that converts Finnish text through **Finnish-NLP/Chatterbox-Finnish → RVC** and produces WAV files with your local RVC voice models.

The same UI also includes **Batch Generation**, optional word-level **Accent** editing, and a **Model Trainer** for creating RVC v2 `.pth + .index` models from an audio folder.

## Tutorial video

[https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY](https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY)

## Requirements

- Windows 11 x64
- WinGet (normally provided by Windows 11 App Installer)
- NVIDIA GPU recommended
- Model Trainer currently requires an NVIDIA CUDA GPU
- Voice Generation can run on CPU, but may be very slow
- Several GB of free disk space recommended
- **Only Accent editing requires optional WSL2 + Docker Desktop**

Normal Voice Generation, Batch Generation, and Model Trainer do not require Docker.

## Installation

Download the repository with `Code → Download ZIP` and extract it to a short path such as:

```text
C:\Zundanen
```

Then double-click `setup.bat`.

The normal setup prepares the main environment, including:

```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
```

TTS and RVC use separate virtual environments under `runtime/`.

> WSL2, Docker Desktop, Aalto alignment, and PyWORLD are not installed by the normal `setup.bat`. Install them only when needed from `Install Accent support` in the Accent section.

### Windows path length

Some Python packages contain long internal paths. Extracting Zundanen too deeply may cause `WinError 206`. A short path such as `C:\Zundanen` is recommended.

### Re-running setup

If setup stops, fix the cause and run `setup.bat` again. Existing components are reused where possible.

To rebuild the environment from scratch:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
```

## Launch

After setup, double-click `Zundanen.exe`.

It starts the local server in the background and opens the UI in a dedicated WebView2 window. Closing the window also stops the local server.

> `Zundanen.exe` is currently unsigned, so Windows SmartScreen may show a warning. The small Windows launcher source is included in `launcher_src/`.

For the browser version, run `run_ui.bat` and open:

```text
http://127.0.0.1:8765
```

Use `install_desktop.bat` if you only need to add the desktop UI dependency to an existing installation.

### Rebuilding the EXE launcher

The repository includes the small launcher source:

```text
launcher_src/zundanen_launcher.go
```

Run `build_launcher.bat` to build:

```text
dist\Zundanen.exe
```

The Go toolchain is required only when rebuilding the launcher. Normal users do not need Go. TTS, RVC, and model files remain in the project's `runtime/` directory and are not bundled into the EXE.

## Voice Generation

1. Choose `Character`
2. Enter Finnish text
3. Adjust `Pitch` / `Index rate` / `Protect` / `Expression` if needed
4. Check `Seed`. With `Fix seed` off, each generation uses a new random seed; with it on, the entered seed is reused
5. Click `Generate voice`
6. Optionally edit Accent
7. Save with `Save As...`

The suggested filename includes date/time and the TTS seed:

```text
Zundamon_20260916_180000_seed123456.wav
```

Save options:

- `Create subtitle TXT` — creates a same-name `.txt`; enabled by default
- `Save source audio` — saves the clean pre-RVC Finnish TTS for later high-quality Accent editing; disabled by default

With source audio enabled:

```text
Zundamon_20260916_180000_seed123456.wav
Zundamon_20260916_180000_seed123456.txt             # if TXT is enabled
Zundamon_20260916_180000_seed123456_source.wav
Zundamon_20260916_180000_seed123456_source.json
```

The `_source.json` file stores metadata used to restore the original RVC settings later.

RVC models are normally detected from layouts such as:

```text
runtime/rvc/exports/
├─ CharacterA/
│  ├─ CharacterA.pth
│  └─ CharacterA.index
└─ CharacterB/
   ├─ CharacterB.pth
   └─ CharacterB.index
```

## Accent

Accent is optional. If its dependencies are missing, the Accent section shows:

```text
Install Accent support
```

The installer checks and adds only missing Accent components:

- PyWORLD / WORLD vocoder
- WSL2
- Docker Desktop
- Aalto Finnish forced-alignment Docker image

Enabling WSL2 for the first time may require a Windows restart. After restarting, open Zundanen and click `Install Accent support` again to continue.

Once installed, forced alignment runs automatically after Voice Generation. Zundanen aligns the Finnish text to the audio and displays syllable buttons for each word.

- choose the accented syllable for each word
- adjust `Strength`
- use `Reset` to clear changes
- use `Preview` to listen to the edited result

The editor uses Aalto alignment for syllable timing and WORLD/PyWORLD for the pre-RVC audio edit before running RVC.

### Import WAV

`Import WAV` can reopen earlier Voice Generation or Batch Generation results.

- a same-name `.txt` is loaded automatically when present
- Voice Generation `*_source.wav` / `*_source.json` files are detected automatically
- matching Batch sources inside `source_audio/` are detected automatically
- when source audio exists, Zundanen can use the higher-quality pre-RVC → WORLD → RVC path
- older/external WAV files without source audio fall back to WORLD editing directly on the finished WAV

## Batch Generation

Batch Generation processes many texts sequentially and prevents Windows automatic sleep while the queue is running.

Input modes:

```text
One line = one WAV
CSV
```

Each output filename includes its seed, for example:

```text
001_seed123456.wav
002_seed987654.wav
```

CSV columns:

```text
filename,text,character,pitch,index_rate,protect,expression,seed
```

Only `text` is required. With a blank `filename`, an item is saved as `001_seed123456.wav`; with `intro`, it becomes `001_intro_seed123456.wav`.

With `Fix seed` off, every queue item receives a new random seed. With it enabled, the Batch Seed value is reused. A CSV row `seed` overrides the Batch setting.

Save options:

- `Create subtitle TXT` — creates same-name `.txt` files; enabled by default
- `Save source audio` — saves the pre-RVC Finnish TTS; disabled by default

When source audio is enabled, all Batch sources are collected under `source_audio/`:

```text
outputs/batches/20260916_180000_ab12cd/
├─ 001_seed123456.wav
├─ 001_seed123456.txt
├─ 002_seed987654.wav
├─ 002_seed987654.txt
├─ source_audio/
│  ├─ 001_seed123456_source.wav
│  ├─ 001_seed123456_source.json
│  ├─ 002_seed987654_source.wav
│  └─ 002_seed987654_source.json
└─ failed_items.csv            # only when failures occurred
```

Queue state is saved automatically, and a stopped/interrupted queue can continue with `Resume Queue`.

## Model Trainer

The `Model Trainer` tab lets you configure:

- Character name
- Dataset folder
- Epochs
- Batch size
- Save every
- Workers
- GPU
- Fresh training

Training pipeline:

```text
Audio dataset
  ↓
Preprocess / 40 kHz
  ↓
RMVPE F0
  ↓
HuBERT v2 / 768-dim
  ↓
RVC v2 training
  ↓
FAISS index
  ↓
runtime/rvc/exports/<Character>/<Character>.pth
runtime/rvc/exports/<Character>/<Character>.index
```

Windows automatic sleep is prevented during training and restored after training finishes or stops.

## UI languages

Explanatory/help text can be switched between `Suomi / 日本語 / English`. Suomi is the default, and the selected language is stored locally.

Buttons, technical labels, and some status text intentionally remain in English.

## Setup diagnostics

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

The doctor checks the required TTS/RVC environment. Docker is reported only as an optional Accent component.

## GitHub Actions

The repository includes:

```text
.github/workflows/windows-smoke.yml
```

On each push / pull request, GitHub checks items including:

- PowerShell syntax for `setup.ps1`, `doctor.ps1`, and `install_accent_support.ps1`
- compilation of the small Go-based Windows launcher
- the lightweight CI setup and compilation of key Python files
- Flask UI startup
- an HTTP 200 response from `/api/state`

Standard GitHub-hosted runners do not run the full CUDA TTS/RVC pipeline or Docker-based forced alignment. GPU and Accent functionality should still be tested on a Windows PC.

## Files excluded from GitHub

`.gitignore` excludes items such as:

```text
runtime/
config.json
outputs/
temp/
__pycache__/
```

Downloaded models, virtual environments, user configuration, RVC models, and generated audio are therefore not normally committed.

## Voice and character rights

The MIT license for Zundanen itself does not grant rights to third-party character voices, training data, or RVC models.

If you use, publish, or redistribute third-party audio, datasets, `.pth`, or `.index` files, check the applicable terms from the rights holder, voice library, and dataset.

## Upstream projects

- Finnish-NLP/Chatterbox-Finnish: https://huggingface.co/Finnish-NLP/Chatterbox-Finnish
- ResembleAI Chatterbox: https://github.com/resemble-ai/chatterbox
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Aalto Finnish Forced Alignment: https://github.com/aalto-speech/finnish-forced-alignment
- Lingsoft aalto-kaldi-align-elg: https://github.com/lingsoft/aalto-kaldi-align-elg
- WORLD: https://github.com/mmorise/World
- PyWORLD: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
- FFmpeg: https://ffmpeg.org/

See `THIRD_PARTY_NOTICES.md` for licensing notes.
