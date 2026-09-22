# Zundanen v1.0.0

**[Suomi](README.md) | 日本語 | [English](README_EN.md)**

Zundanen は，フィンランド語テキストを **Finnish-NLP/Chatterbox-Finnish → RVC** の順に処理し，ローカルの RVC モデルの声で WAV を生成する Windows 向けローカルアプリです．

同じ UI に，複数文章を連続生成する **Batch Generation**，単語ごとのアクセントを編集する **Accent**，音声フォルダから RVC v2 モデル（`.pth + .index`）を作成する **Model Trainer** を含みます．

## 対象環境

- Windows 11 x64
- WinGet（通常は Windows 11 の「アプリ インストーラー」に含まれます）
- NVIDIA GPU 推奨
- Model Trainer は現在 NVIDIA CUDA GPU 必須
- Voice Generation は CPU でも起動できますが，生成は非常に遅くなる場合があります
- 数 GB 以上の空き容量を推奨
- **Accent 編集のみ任意で WSL2 + Docker Desktop が必要**

通常の Voice Generation / Batch Generation / Model Trainer に Docker は不要です．

## インストール

GitHub から `Code → Download ZIP` で取得し，できるだけ短いパスへ展開してください．

```text
C:\Zundanen
```

その後，`setup.bat` をダブルクリックします．

通常セットアップでは主に次を自動で準備します．

```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
```

TTS と RVC はプロジェクト内の `runtime/` 以下に別々の仮想環境として構築されます．

> Accent 用の WSL2 / Docker Desktop / Aalto alignment / PyWORLD は通常の `setup.bat` ではインストールしません．Accent 欄の `Install Accent support` から必要な人だけ追加できます．

### Windows のパス長

Python パッケージには長い内部パスを持つものがあります．深い場所に展開すると `WinError 206` が発生する場合があるため，`C:\Zundanen` のような短い場所を推奨します．

### セットアップをやり直す

途中で失敗した場合は原因を直して `setup.bat` を再実行できます．追加分だけ再読み込みされます．

完全に作り直す場合は，PowerShell から次を使用できます．

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
```

## 起動

セットアップ完了後は `Zundanen.exe` のダブルクリックでの起動推奨．

`Zundanen.exe` はローカルサーバーを裏で起動し，WebView2 の専用ウィンドウに UI を表示します．ウィンドウを閉じるとローカルサーバーも終了します．

> `Zundanen.exe` は現在コード署名していないため，Windows SmartScreen の警告が表示される場合があります．小型Windowsランチャーのソースは `launcher_src/` に含まれています．

ブラウザ版を使う場合は `run_ui.bat` を実行します．

```text
http://127.0.0.1:8765
```

デスクトップ UI 用依存関係だけ追加したい場合は `install_desktop.bat` を使用できます．

### EXE ランチャーを再ビルドする場合

リポジトリには小型ランチャーのソースを含めます．

```text
launcher_src/zundanen_launcher.go
```

`build_launcher.bat` を実行すると，次を生成します．

```text
dist\Zundanen.exe
```

Go ツールチェーンが必要なのはランチャーを再ビルドする場合だけです．通常のユーザーは Go をインストールする必要はありません．TTS・RVC・モデル類は `runtime/` に置かれ，EXE 本体には埋め込まれません．

## Voice Generation

1. `Character` を選択
2. フィンランド語テキストを入力
3. 必要に応じて `Pitch` / `Index rate` / `Protect` / `Expression` を調整
4. `Seed` を確認．`Fix seed` が OFF なら生成ごとに新しいランダム Seed，ON なら入力した Seed を再利用
5. `Generate voice` を押す
6. 必要なら Accent を編集
7. `Save As...` で保存

保存時の推奨ファイル名には日時と Seed が入ります．

```text
Zundamon_20260916_180000_seed123456.wav
```

保存オプション：

- `Create subtitle TXT` — 同名の `.txt` を保存．初期設定 ON
- `Save source audio` — RVC 変換前の Finnish TTS を後の Accent 再編集用に保存．初期設定 OFF

`Save source audio` を ON にした場合は，完成 WAV と同じ場所へ次のように保存します．

```text
Zundamon_20260916_180000_seed123456.wav
Zundamon_20260916_180000_seed123456.txt             # TXT を ON にした場合
Zundamon_20260916_180000_seed123456_source.wav
Zundamon_20260916_180000_seed123456_source.json
```

`_source.json` には，後で同じ RVC 設定を復元するためのメタデータが保存されます．

RVC モデルは通常，次のような構成から自動検出されます．

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

Accent は任意機能です．未導入の場合，Accent 欄に次のボタンが表示されます．

```text
Install Accent support
```

このボタンは不足している Accent 用コンポーネントだけを確認・導入します．

- PyWORLD / WORLD vocoder
- WSL2
- Docker Desktop
- Aalto Finnish forced alignment 用Docker image

WSL2 の初回有効化では Windows 再起動が必要になる場合があります．その場合は再起動後に Zundanen を開き，`Install Accent support` をもう一度押してください．

Accent support 導入後は，Voice Generation で音声を生成すると自動でフィンランド語テキストと音声を強制アライメントし，単語ごとの音節ボタンが表示されます．

- 各単語でアクセントを置く音節を選択
- `Strength` で強さを調整
- `Reset` で設定を戻す
- `Preview` で編集結果を試聴

Accent 編集では Aalto alignment で音節位置を取得し，WORLD/PyWORLD で RVC 前の音声を編集してから RVC 変換します．

### Import WAV

`Import WAV` では，Voice Generation や Batch Generation で以前保存した WAV を再編集できます．

- 同名の `.txt` があればテキストを自動読込
- Voice Generation の `*_source.wav` / `*_source.json` があればそれを自動利用
- Batch Generation の `source_audio/` に対応する source があればそれを自動利用
- source audio を見つけた場合は，RVC 前音声 → WORLD → RVC の高品質ルートで再編集
- source audio が無い古い/外部 WAV は，完成 WAV へ直接 WORLD 編集するフォールバックを使用

## Batch Generation

`Batch Generation` では複数の文章を順番に生成できます．キュー処理中は Windows の自動スリープを抑止します．

入力方法：

```text
One line = one WAV
CSV
```

通常の行モードでは各項目に Seed 値が付き，次のように保存されます．

```text
001_seed123456.wav
002_seed987654.wav
```

CSV では次の設定を指定できます．

```text
filename,text,character,pitch,index_rate,protect,expression,seed
```

必須は `text` のみです．`filename` が空なら `001_seed123456.wav` のように保存され，`intro` のように指定すると `001_intro_seed123456.wav` のようになります．

`Fix seed` が OFF なら各項目に新しいランダム Seed を使用し，ON なら Seed 欄の値を使用します．CSV 行の `seed` は Batch 画面の設定よりも優先されます．

保存オプション：

- `Create subtitle TXT` — 各 WAV と同名の `.txt` を保存．初期設定 ON
- `Save source audio` — RVC 前の Finnish TTS を保存．初期設定 OFF

source audio を ON にすると，Batch 出力フォルダ内の `source_audio/` にまとめて保存します．

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
└─ failed_items.csv            # 失敗があった場合のみ
```

キュー状態は自動保存され，停止やアプリ再起動後も `Resume Queue` で再開できます．

## Model Trainer

`Model Trainer` タブでは次を設定します．

- Character name
- Dataset folder
- Epochs
- Batch size
- Save every
- Workers
- GPU
- Fresh training

学習パイプライン：

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

学習中は Windows の自動スリープを抑止し，終了または停止時に解除します．

## UI の言語

説明・ヘルプ文は `Suomi / 日本語 / English` から切り替えられます．デフォルトは Suomi で，選択した言語はローカルに保存されます．

ボタン名・技術用語・一部の状態表示は意図的に英語固定です．

## セットアップ診断

環境を確認したい場合は次を実行します．

```powershell
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

通常機能に必要な TTS/RVC 環境を確認し，Docker は Accent 用の任意項目として表示します．

## GitHub Actions

次の workflow を同梱しています．

```text
.github/workflows/windows-smoke.yml
```

push / pull request ごとにGitHub側で，

- `setup.ps1` / `doctor.ps1` / `install_accent_support.ps1` の PowerShell 構文
- Go 製 Windows ランチャーのビルド
- 軽量 CI セットアップと主要 Python ファイルの compile
- Flask UI の起動
- `/api/state` が HTTP 200 を返すか

を自動チェックします．

通常の GitHub-hosted runner では CUDA を使った実際の TTS/RVC 生成や Docker を使う強制アライメントまではテストしません．GPU 機能と Accent 機能は Windows 実機でも確認してください．

## GitHub へ含めないもの

`.gitignore` では主に次を除外しています．

```text
runtime/
config.json
outputs/
temp/
__pycache__/
```

そのため，ダウンロード済みモデル，venv，ユーザー設定，RVC モデル，生成 WAV などは通常 repository へ入りません．

## 音声・キャラクターの権利

Zundanen 本体の MIT ライセンスは，第三者のキャラクター音声・学習データ・RVC モデル等の利用権を与えるものではありません．

第三者の学習データ，`.pth` / `.index` を利用・公開・再配布する場合は，それぞれの権利者・音声ライブラリ・データセットの利用規約を確認してください．

## 上流プロジェクト

- Finnish-NLP/Chatterbox-Finnish: https://huggingface.co/Finnish-NLP/Chatterbox-Finnish
- ResembleAI Chatterbox: https://github.com/resemble-ai/chatterbox
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Aalto Finnish Forced Alignment: https://github.com/aalto-speech/finnish-forced-alignment
- Lingsoft aalto-kaldi-align-elg: https://github.com/lingsoft/aalto-kaldi-align-elg
- WORLD: https://github.com/mmorise/World
- PyWORLD: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
- FFmpeg: https://ffmpeg.org/

ライセンス情報は `THIRD_PARTY_NOTICES.md` も確認してください．
