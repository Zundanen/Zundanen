# Zundanen v2.0.0

## Main change

Finnish speech is now synthesized by Zundanen's own **VITS models with FinBERT features** instead of "Chatterbox-Finnish → RVC":

- **Chatterbox** (**included** as `models/chatterbox.pt`, 68 MB, float16): Finnish VITS in the Chatterbox voice (Normal only). With `Use RVC voice conversion` and your own RVC model (for Zundamon at Pitch +8) it can speak in another voice.
- **Character models with emotion styles** (**Normal / Ama / Tsun / Sexy**, adjustable style strength, pitch and speed) are trained with the Model Trainer from your own data. No character models are included.

Measured on 60 held-out Finnish Tatoeba sentences (Whisper large-v2 CER, lower is better): original Chatterbox-Finnish audio 2.5%, Chatterbox 2.3%, Chatterbox + Zundamon RVC (Protect 0.15) 4.7%; the Zundamon emotion model (distributed separately for research on Hugging Face): Normal 7.1%, Ama / Tsun / Sexy 9.2% / 7.8% / 8.1%.

## UI

- Tabs: Voice Generation, Batch Generation, Model Trainer, Voice Conversion.
- Voice Generation: Character, Style, Style strength, Pitch (±10 semitones from the character level), Speed, Seed, optional RVC conversion (Pitch / Index rate / Protect sliders, default 0 / 0.1 / 0.15; a lower Protect keeps voiceless consonants such as /h/, /s/, /t/ closer to the source).
- Batch Generation: same settings; CSV columns `filename,text,style,strength,speed,seed` (old `length_scale` columns are still read).
- Model Trainer: RVC (as in v1.1.0) and VITS (Joint training, Finnish fine-tuning, Style fine-tuning; training length in epochs). Joint / Finnish continue a model you trained before; start with `New model (from scratch)`.
- Model Trainer → Joint training → Base model `New model (from scratch)`: a new model from random weights out of a Finnish WAV+TXT folder, emotion WAV+LAB rows (folder, LAB folder, speaker, style; more rows with +) and an RVC model (.pth) with its Character chosen from the entered speakers for the Finnish bridge (every Finnish sentence of the folder; RVC bridge share sets how often it is used; pitch shift set automatically, the character's own style tempo measured from its recordings).
- Voice Conversion: RVC conversion of a WAV file or a whole folder, record-and-convert with Save As, in-app real-time conversion (no extra window, about 0.3 s latency).
- Accent: unchanged from v1.1.0.

## Model files

The repository contains one model, `models/chatterbox.pt` (Chatterbox). Character models and RVC models are **not** included; the Zundamon and Shikoku Metan character models are distributed separately for non-commercial research on Hugging Face, https://huggingface.co/Zundanen/finnish-emotion-vits, under the terms of the SSS multimodal database (put a character file `<name>.pt` in `models/characters` and its RVC model in `runtime/rvc/exports/<name>/`, then restart). See README (“Model files”).

## Known limitations

- Words with h + consonant (e.g. *tyhjä*) were weak in the developer's Zundamon model; RVC also weakens /h/.
- The first generation downloads the FinBERT model (about 1.4 GB).
- Screenshots in `docs/` are still from v1.1.0.
