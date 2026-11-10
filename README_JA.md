# Zundanen v2.0.0

**[Suomi](README.md) | 日本語 | [English](README_EN.md)**

Zundanen は，フィンランド語のテキストから WAV を作る Windows 向けのローカルアプリです．

v2.0.0 では，フィンランド語の音声合成を **独自の VITS モデル（FinBERT 特徴量つき）** に置き換えました．このリポジトリには **Chatterbox** のモデル（フィンランド語 VITS，Normal のみ）が入っています．`Use RVC voice conversion` と自分で用意した RVC モデルで，ほかの声（例：ずんだもん）にできます．感情スタイル（Normal / Ama / Tsun / Sexy）つきのキャラクターのモデルは，Model Trainer で自分のデータから学習します．キャラクターのモデルは含まれていません．

同じ UI に，**Batch Generation**，単語ごとの **Accent** 編集，**Model Trainer**（RVC v2，VITS の追加学習），**Voice Conversion**（WAV / フォルダ変換，録音，リアルタイム変換）を含みます．

> v1 の画面写真（`docs/`）は v1.1.0 のものです．

## 使い方動画（v1）

[https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY](https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY)

## v1 からの主な変更

- 音声合成：v1 の「Chatterbox-Finnish → RVC」から，**フィンランド語 VITS（感情スタイルつき）**に変わりました．
- **Character**：学習済みのモデルを選びます．
  - **Chatterbox**（同梱）：Chatterbox の声のフィンランド語 VITS（Normal のみ）．`Use RVC` でずんだもんなどの声に変換します．
  - **自分で学習したモデル**（Model Trainer → VITS）：感情スタイルつきのキャラクターの声．キャラクターのモデルは同梱していません（「モデルファイル」を参照）．
- **Style**：Normal / Ama / Tsun / Sexy．強さ（Style strength）を変えられます．
- **Pitch / Speed / Seed**．
- **Use RVC voice conversion**：生成した音声を RVC で変換します（任意）．
- **Batch Generation**：Character / Style / RVC に対応しました．
- **Model Trainer**：RVC（v1 と同じ）に加えて，VITS の追加学習（Joint / Finnish / Style）．
- **Voice Conversion** タブ：WAV・フォルダの RVC 変換，録音して変換，アプリ内のリアルタイム変換．

## 対象環境

- Windows 11 x64
- WinGet（通常は Windows 11 の「アプリ インストーラー」に含まれます）
- NVIDIA GPU は必須ではありません．音声生成は CPU だけでも動きます（GPU のない環境で確認）
- Model Trainer（学習）は NVIDIA CUDA GPU 必須（開発・確認は RTX 3060 12 GB）
- リアルタイム変換には NVIDIA GPU を推奨します
- 空き容量：数十 GB を推奨（環境・モデル・生成物）
- **初回の音声生成時に**，フィンランド語 BERT（`TurkuNLP/bert-large-finnish-cased-v1`，約 1.4 GB）を Hugging Face から自動でダウンロードします（インターネット接続が必要です）

## インストール

v1 と同じです．GitHub から `Code → Download ZIP` で取得し，できるだけ短いパス（例：`C:\Zundanen`）へ展開して，`setup.bat` をダブルクリックします．

通常セットアップでは主に次を自動で準備します．

```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish（TTS 環境：VITS と FinBERT もこの環境で動きます）
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
PyWORLD / WORLD vocoder
Windows ネイティブ Aalto/Kaldi runtime
必要な場合は Intel oneMKL runtime
```

セットアップのやり直し・完全な作り直し・`doctor.ps1` による診断も v1 と同じです．

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

## モデルファイル

このリポジトリに入っているもの：

```text
models/
└─ chatterbox.pt     Chatterbox，約 68 MB
```

- 推論用の重みだけです（識別器・最適化の状態は除いています）．GitHub の 1 ファイル 100 MB の上限に収まるよう float16 で保存しています．アプリは float32 で読み込みます．
- **含まれないもの**：感情スタイルつきのキャラクターのモデル（ずんだもん，四国めたん など）と RVC モデル（`.pth` / `.index`）．これらの学習データの権利は，それぞれの権利者にあります．使ってよいデータから Model Trainer で自分で学習してください．学習したモデルは自動で `Character` に追加されます．
- **キャラクターのモデル**（このリポジトリには含まれません）：ずんだもん，四国めたんの感情モデルと RVC モデルは，研究用として [Hugging Face（Zundanen/finnish-emotion-vits）](https://huggingface.co/Zundanen/finnish-emotion-vits) で別に配布しています．非商用・研究目的に限ります．使うときは [SSS マルチモーダルデータベースの利用規約](https://zunko.jp/multimodal_dev/login.php) に従ってください．キャラクターのファイル `<名前>.pt` を `models/characters` に，RVC モデル（`<名前>.pth` と `.index`）を `runtime/rvc/exports/<名前>/` に置いてアプリを起動し直すと，キャラクターが `Character` に，RVC モデルが RVC モデルの一覧に出ます．キャラクター 1 人はファイル 1 つです．
- RVC モデルは v1 と同じく `runtime/rvc/exports/<名前>/` に置いてください．

## 起動

v1 と同じく `Zundanen.exe`（推奨）または `run_ui.bat`（ブラウザ版，`http://127.0.0.1:8765`）で起動します．

## Voice Generation

1. `Character` を選択
2. `Style` と `Style strength` を選択（Chatterbox は Normal のみ）
3. フィンランド語テキストを入力
4. 必要に応じて `Pitch`（キャラクターの基準の高さからの半音，±10）と `Speed`（1.0 が標準，大きいほど速い）を調整
5. `Seed` を確認（`Fix seed` が OFF なら毎回新しい Seed）
6. 必要なら `Use RVC voice conversion` をオンにして RVC モデルと `Pitch` / `Index rate` / `Protect` を選ぶ
7. `Generate voice` を押す
8. 必要なら Accent を編集し，`Save As...` で保存

### おすすめの使い分け

| 用途 | Character | Style | RVC |
|---|---|---|---|
| いちばんはっきりした Normal | Chatterbox | Normal | 自分の RVC モデル（ずんだもんなら Pitch **+8**），Index rate 0.1，Protect 0.15 |
| 感情スタイル | 自分で学習したモデル | Ama / Tsun / Sexy | なし |

- Chatterbox を選ぶと，RVC の Pitch が 0 のときは自動で +8（ずんだもんの高さ）になります．Chatterbox には高さの操作がないので，その間は上の `Pitch` スライダーは使えません．
- RVC の `Index rate` を高くすると，日本語寄りの発音になりやすくなります．フィンランド語には 0.1 前後をおすすめします．

### 発音の目安（開発時の測定）

学習に一度も使っていない Tatoeba のフィンランド語 60 文 × 2 シードを，Whisper large-v2 で書き起こした文字誤り率（CER，低いほど良い）です．

| 音声 | CER |
|---|---:|
| Chatterbox-Finnish の元の音声（上限の目安） | 2.5% |
| Chatterbox | 2.3% |
| Chatterbox ＋ ずんだもん RVC（+8，Protect 0.15） | 4.7% |
| Zundamon Normal | 7.1% |
| Zundamon Ama / Tsun / Sexy | 9.2% / 7.8% / 8.1% |

Zundamon の行は，Hugging Face で研究用に配布しているずんだもんの感情モデルの値です．

h ＋子音（例：*tyhjä*，*kissanhiekka*）を含む単語は，ずんだもんの声で崩れやすいことが分かっています．

## Batch Generation

Voice Generation と同じ Character / Style / Style strength / Pitch / Speed / Seed / RVC で，複数の文を順番に生成します．Chatterbox も選べます．

CSV の列：

```text
filename,text,style,strength,speed,seed
```

必須は `text` のみです．空欄の項目は画面の設定を使います．行ごとに別のスタイルにもできます（`speed` は 1.0 が標準．古い `length_scale` の列も読めます）．

## Model Trainer

上部の `VITS` / `RVC` で切り替えます．

- **RVC**：v1 と同じです（Character name，Dataset folder，Epochs，Batch size，Workers，Fresh training）．`runtime/rvc/exports/<Character>/` に `.pth` と `.index` を作ります．
- **VITS**（上級者向け）：
  - `Joint training` / `Finnish fine-tuning`：本体 → スタイル分岐 → 声質の差分を学習し，終わると Model name で Character に追加します．**前に学習したモデルの続きを学習します（同梱のモデルはありません）．最初は下の `New model (from scratch)` で作ってください．**
  - `Style fine-tuning`：Normal とスタイルの WAV + LAB の組（番号で対応，20 組以上）から，選んだモデル用の新しいスタイルを追加します．
  - **新しいモデル（ゼロから）**：`Joint training` の `Base model` で `New model (from scratch)` を選ぶと，ランダムな初期値から作ります．選ぶもの：
    - **Finnish folder**：フィンランド語の WAV と同名の TXT（50 組以上．良い発音には約 20 時間以上）
    - **Emotion data**：話者とスタイルごとに 1 行（WAV のフォルダ，LAB のフォルダ（空欄なら同じフォルダ），話者名，スタイル Normal / Ama / Tsun / Sexy）．`+` で行を追加します．どの話者にも Normal の行が必要です．LAB は WAV と同じファイル名にします．自分の LAB がない行は，同じスタイルのほかの行にある同じ名前の LAB を使います．
    - **Bridge RVC model (.pth)**（Browse…）と **Character**：フォルダのフィンランド語の文を全部その RVC の声に変換して（それぞれ 2 つの高さ．1 ファイル約 1 秒．20,000 文で約 11 時間），キャラクターにフィンランド語を覚えさせます．Character は入力した話者から選びます（ファイル名が合えば自動で選ばれます．例：Zundamon.pth → zundamon）．学習でそれをどれだけ使うかは `RVC bridge share` で決めます．隣に .index があれば使い，ピッチのずらし幅は自動で決めます（Index 0.1 / Protect 0.15）．
    - Epoch (VITS) の既定は 35 になります．既存のモデルの続きよりずっと長くかかります（1 日以上かかることもあります）．
  - 長さはエポック数で指定します（Epoch (VITS) 既定 4.5，Epoch (Style) 既定 80）．Batch size と Workers は PC に合わせて変えられます．

## Voice Conversion

- 左：RVC モデルと Pitch / Index rate / Protect
- WAV conversion：ファイル 1 つ，またはフォルダの中の音声をまとめて変換（`outputs/voice_conversion/` に保存，元のファイルは上書きしません）
- Recording：録音して変換し，`Save As...` で保存
- Realtime voice conversion：入力・出力デバイスを選んで Start / Stop（別ウィンドウは開きません．遅れは約 0.3 秒）

## Accent

v1.1.0 と同じです（Windows ネイティブ Aalto/Kaldi による強制アライメント，WORLD/PyWORLD による編集．WSL2 / Docker Desktop 不要）．

## UI の言語

説明文は `Suomi / 日本語 / English` で切り替えられます．ボタン名や項目名は英語固定です．

## GitHub へ含めないもの

`.gitignore` で次を除外しています．

```text
runtime/  config.json  outputs/  temp/  __pycache__/  *.pt（models/chatterbox.pt を除く）  *.pth  *.index  *.safetensors  *.wav  local_state/
```

## 音声・キャラクター・学習データの権利

Zundanen 本体の MIT ライセンスは，第三者のキャラクター音声・学習データ・学習済みモデル・RVC モデル等の利用権を与えるものではありません．

同梱の Chatterbox モデル（`models/chatterbox.pt`）は，Chatterbox-Finnish（MIT）が Tatoeba のフィンランド語の文（CC BY 2.0 FR）から作った音声だけで学習しています．声の見本には，Chatterbox-Finnish に付属の参照音声（`samples/reference_finnish.wav`．Mozilla Common Voice 由来，CC0 1.0）を使いました．日本語のキャラクター音声コーパスは使っていません．

自分で学習したモデルや RVC モデルには，その学習データの規約が適用されます．利用・公開・再配布する場合は，それぞれの権利者・利用規約を確認してください（`THIRD_PARTY_NOTICES.md`）．

## 上流プロジェクト

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

ライセンス情報は `THIRD_PARTY_NOTICES.md` も確認してください．
