# Zundanen v2.0.0

## Main change

Finnish speech is now synthesized by Zundanen's own **VITS models with FinBERT features** instead of "Chatterbox-Finnish → RVC":

- **Chatterbox** (**included** as `models/chatterbox.pt`, 68 MB, float16): Finnish VITS in the Chatterbox voice (Normal only). With `Use RVC voice conversion` and your own RVC model (for Zundamon at Pitch +8) it can speak in another voice.
- **Character models with emotion styles** (**Normal / Ama / Tsun / Sexy**, adjustable style strength, pitch and speed) are trained with the Model Trainer from your own data. No character models are included.

Measured on 60 held-out Finnish Tatoeba sentences (Whisper large-v2 CER, lower is better): original Chatterbox-Finnish audio 2.5%, Chatterbox 1.9%, Chatterbox + Zundamon RVC (Protect 0.15) 3.4%. The character models (distributed separately for research on Hugging Face), Normal / Ama / Tsun / Sexy: Zundamon 6.1% / 7.5% / 7.0% / 6.5%, Shikoku Metan 6.4% / 7.6% / 6.8% / 8.0%.

## UI

- Tabs: Voice Generation, Batch Generation, Model Trainer, Voice Conversion.
- Voice Generation: Character, Style, Style strength, Pitch (±10 semitones from the character level), Speed, Seed, optional RVC conversion (Pitch / Index rate / Protect sliders, default 0 / 0.1 / 0.15; a lower Protect keeps voiceless consonants such as /h/, /s/, /t/ closer to the source).
- Batch Generation: same settings; CSV columns `filename,text,style,strength,speed,seed` (old `length_scale` columns are still read).
- Model Trainer: RVC (as in v1.1.0) and VITS (Joint training, Finnish fine-tuning, Style fine-tuning; training length in epochs). Joint / Finnish continue a model you trained before; start with `New model (from scratch)`.
- Model Trainer → Joint training → Base model `New model (from scratch)`: a new model from random weights out of a Finnish WAV+TXT folder, emotion WAV+LAB rows (folder, LAB folder, speaker, style; more rows with +) and an RVC model (.pth) with its Character chosen from the entered speakers for the Finnish bridge (every Finnish sentence of the folder; RVC bridge share sets how often it is used; pitch shift set automatically, the character's own style tempo measured from its recordings).
- Model Trainer → VITS: `Align the start of the Finnish WAVs` (Finnish fine-tuning and New model, on by default) evens out the silence at the start of the Finnish WAVs before training, leaving `Silence before the first word (ms)` (default 100 ms, 0–500). Training uses aligned copies; the original files are not changed.
- Voice Conversion: RVC conversion of a WAV file or a whole folder, record-and-convert with Save As, in-app real-time conversion (no extra window, about 0.3 s latency).
- Accent: forced alignment as in v1.1.0. Speech from the character models is edited inside the model; imported WAVs and Chatterbox speech are edited with WORLD. Strength −50 % (weaker accent) to +50 % (strongest).

## Model files

The repository contains one model, `models/chatterbox.pt` (Chatterbox). Character models and RVC models are **not** included; the Zundamon and Shikoku Metan character models are distributed separately for non-commercial research on Hugging Face, https://huggingface.co/Zundanen/finnish-emotion-vits, under the terms of the SSS multimodal database (put a character file `<name>.pt` in `models/characters` and its RVC model in `runtime/rvc/exports/<name>/`, then restart). See README (“Model files”).

## Replaces the first v2.0.0 upload

This release replaces the first v2.0.0 upload. Changes since then:

- Retrained Chatterbox model (`models/chatterbox.pt`): CER 2.3% → 1.9%, with Zundamon RVC 4.7% → 3.4%.
- In the emotion models, s before y (e.g. *syönyt*) sounded close to "sh". Generation now corrects it automatically.
- The `Align the start of the Finnish WAVs` option in the Model Trainer (see above).
- Updated Zundamon and Shikoku Metan models on Hugging Face (Zundamon before: 7.1% / 9.2% / 7.8% / 8.1%). Shikoku Metan now uses the same style tempo as Zundamon.
- Abbreviations are read with Finnish letter names (*TV:tä* → *teeveetä*, *EU:hun* → *eeuuhun*). When the text is read differently from what you typed, a `Read as` line under the text box shows the reading.
- Setup no longer downloads Chatterbox-Finnish (about 5 GB), which v2 does not use.
- Accent editing no longer makes the whole sentence noisy: only the edited words change. Speech from the character models is rendered again by the model with the edited accent instead of being processed with WORLD. The strength slider now runs from −50 % (weaker accent) to +50 % (as strong as 50 % before). If the preview used RVC, the edited speech is converted with RVC again.

If you downloaded v2.0.0 before, get the source again (`Code → Download ZIP`, or `git pull`) and copy it over your Zundanen folder. `runtime/`, `config.json`, `outputs/` and `models/characters/` are not part of the repository, so they are kept. The Python environments are unchanged, so setup does not need to be run again. Replace `models/characters/Zundamon.pt` and `Metan.pt` with the new files from Hugging Face; the RVC models are unchanged.

## Known limitations

- The first generation downloads the FinBERT model (about 1.4 GB).
- Screenshots in `docs/` are still from v1.1.0.
