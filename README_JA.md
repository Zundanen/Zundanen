# Zundanen v1.0.0

**[Suomi](README.md) | 日本誁E| [English](README_EN.md)**

Zundanen は�E�フィンランド語テキストを **Finnish-NLP/Chatterbox-Finnish ↁERVC** の頁E��処琁E���E�ローカルの RVC モチE��の声で WAV を生成すめEWindows 向けローカルアプリです！E
同じ UI に�E�褁E��斁E��を連続生成すめE**Batch Generation**�E�単語ごとのアクセントを編雁E��めE**Accent**�E�音声フォルダから RVC v2 モチE���E�E.pth + .index`�E�を作�Eする **Model Trainer** を含みます！E
## 対象環墁E
- Windows 11 x64
- WinGet�E�通常は Windows 11 の「アプリ インスト�Eラー」に含まれます！E- NVIDIA GPU 推奨
- Model Trainer は現在 NVIDIA CUDA GPU 忁E��E- Voice Generation は CPU でも起動できますが�E�生成�E非常に遁E��なる場合がありまぁE- 数 GB 以上�E空き容量を推奨
- **Accent 編雁E�Eみ任意で WSL2 + Docker Desktop が忁E��E*

通常の Voice Generation / Batch Generation / Model Trainer に Docker は不要です！E
## インスト�Eル

GitHub から `Code ↁEDownload ZIP` で取得し�E�できるだけ短ぁE��スへ展開してください�E�E
```text
C:\Zundanen
```

そ�E後，`setup.bat` をダブルクリチE��します！E
通常セチE��アチE�Eでは主に次を�E動で準備します！E
```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
```

TTS と RVC はプロジェクト�Eの `runtime/` 以下に別、E�E仮想環墁E��して構築されます！E
> Accent 用の WSL2 / Docker Desktop / Aalto alignment / PyWORLD は通常の `setup.bat` ではインスト�Eルしません�E�Accent 欁E�E `Install Accent support` から忁E��な人だけ追加できます！E
### Windows のパス長

Python パッケージには長ぁE�E部パスを持つも�Eがあります．深ぁE��所に展開すると `WinError 206` が発生する場合があるため�E�`C:\Zundanen` のような短ぁE��所を推奨します！E
### セチE��アチE�Eをやり直ぁE
途中で失敗した場合�E原因を直して `setup.bat` を�E実行できます．追加刁E��け�E読み込みされます！E
完�Eに作り直す場合�E�E�PowerShell から次を使用できます！E
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
```

## 起勁E
セチE��アチE�E完亁E���E `Zundanen.exe` のダブルクリチE��での起動推奨�E�E
`Zundanen.exe` はローカルサーバ�Eを裏で起動し�E�WebView2 の専用ウィンドウに UI を表示します．ウィンドウを閉じるとローカルサーバ�Eも終亁E��ます！E
> `Zundanen.exe` は現在コード署名してぁE��ぁE��めE��Windows SmartScreen の警告が表示される場合があります．小型Windowsランチャーのソースは `launcher_src/` に含まれてぁE��す！E
ブラウザ版を使ぁE��合�E `run_ui.bat` を実行します！E
```text
http://127.0.0.1:8765
```

チE��クトッチEUI 用依存関係だけ追加したぁE��合�E `install_desktop.bat` を使用できます！E
### EXE ランチャーを�Eビルドする場吁E
リポジトリには小型ランチャーのソースを含めます！E
```text
launcher_src/zundanen_launcher.go
```

`build_launcher.bat` を実行すると�E�次を生成します！E
```text
dist\Zundanen.exe
```

Go チE�Eルチェーンが忁E��なのはランチャーを�Eビルドする場合だけです．通常のユーザーは Go をインスト�Eルする忁E���Eありません�E�TTS・RVC・モチE��類�E `runtime/` に置かれ�E�EXE 本体には埋め込まれません�E�E
## Voice Generation

1. `Character` を選抁E2. フィンランド語テキストを入劁E3. 忁E��に応じて `Pitch` / `Index rate` / `Protect` / `Expression` を調整
4. `Seed` を確認．`Fix seed` ぁEOFF なら生成ごとに新しいランダム Seed�E�ON なら�E力しぁESeed を�E利用
5. `Generate voice` を押ぁE6. 忁E��なめEAccent を編雁E7. `Save As...` で保孁E
保存時の推奨ファイル名には日時と Seed が�Eります！E
```text
Zundamon_20260916_180000_seed123456.wav
```

保存オプション�E�E
- `Create subtitle TXT`  E同名の `.txt` を保存．�E期設宁EON
- `Save source audio`  ERVC 変換前�E Finnish TTS を後�E Accent 再編雁E��に保存．�E期設宁EOFF

`Save source audio` めEON にした場合�E�E�完�E WAV と同じ場所へ次のように保存します！E
```text
Zundamon_20260916_180000_seed123456.wav
Zundamon_20260916_180000_seed123456.txt             # TXT めEON にした場吁EZundamon_20260916_180000_seed123456_source.wav
Zundamon_20260916_180000_seed123456_source.json
```

`_source.json` には�E�後で同じ RVC 設定を復允E��るため�EメタチE�Eタが保存されます！E
RVC モチE��は通常�E�次のような構�Eから自動検�Eされます！E
```text
runtime/rvc/exports/
├─ CharacterA/
━E ├─ CharacterA.pth
━E └─ CharacterA.index
└─ CharacterB/
   ├─ CharacterB.pth
   └─ CharacterB.index
```

## Accent

Accent は任意機�Eです．未導�Eの場合，Accent 欁E��次のボタンが表示されます！E
```text
Install Accent support
```

こ�Eボタンは不足してぁE�� Accent 用コンポ�Eネントだけを確認�E導�Eします！E
- PyWORLD / WORLD vocoder
- WSL2
- Docker Desktop
- Aalto Finnish forced alignment 用Docker image

WSL2 の初回有効化では Windows 再起動が忁E��になる場合があります．その場合�E再起動後に Zundanen を開き，`Install Accent support` をもぁE��度押してください�E�E
Accent support 導�E後�E�E�Voice Generation で音声を生成すると自動でフィンランド語テキストと音声を強制アライメントし�E�単語ごとの音節ボタンが表示されます！E
- 吁E��語でアクセントを置く音節を選抁E- `Strength` で強さを調整
- `Reset` で設定を戻ぁE- `Preview` で編雁E��果を試聴

Accent 編雁E��は Aalto alignment で音節位置を取得し�E�WORLD/PyWORLD で RVC 前�E音声を編雁E��てから RVC 変換します！E
### Import WAV

`Import WAV` では�E�Voice Generation めEBatch Generation で以前保存しぁEWAV を�E編雁E��きます！E
- 同名の `.txt` があれ�EチE��ストを自動読込
- Voice Generation の `*_source.wav` / `*_source.json` があれ�Eそれを�E動利用
- Batch Generation の `source_audio/` に対応すめEsource があれ�Eそれを�E動利用
- source audio を見つけた場合�E�E�RVC 前音声 ↁEWORLD ↁERVC の高品質ルートで再編雁E- source audio が無ぁE��ぁE外部 WAV は�E�完�E WAV へ直接 WORLD 編雁E��るフォールバックを使用

## Batch Generation

`Batch Generation` では褁E��の斁E��を頁E��に生�Eできます．キュー処琁E��は Windows の自動スリープを抑止します！E
入力方法！E
```text
One line = one WAV
CSV
```

通常の行モードでは吁E��E��に Seed 値が付き�E�次のように保存されます！E
```text
001_seed123456.wav
002_seed987654.wav
```

CSV では次の設定を持E��できます！E
```text
filename,text,character,pitch,index_rate,protect,expression,seed
```

忁E���E `text` のみです．`filename` が空なめE`001_seed123456.wav` のように保存され，`intro` のように持E��すると `001_intro_seed123456.wav` のようになります！E
`Fix seed` ぁEOFF なら各頁E��に新しいランダム Seed を使用し，ON なめESeed 欁E�E値を使用します．CSV 行�E `seed` は Batch 画面の設定よりも優先されます！E
保存オプション�E�E
- `Create subtitle TXT`  E吁EWAV と同名の `.txt` を保存．�E期設宁EON
- `Save source audio`  ERVC 前�E Finnish TTS を保存．�E期設宁EOFF

source audio めEON にすると�E�Batch 出力フォルダ冁E�E `source_audio/` にまとめて保存します！E
```text
outputs/batches/20260916_180000_ab12cd/
├─ 001_seed123456.wav
├─ 001_seed123456.txt
├─ 002_seed987654.wav
├─ 002_seed987654.txt
├─ source_audio/
━E ├─ 001_seed123456_source.wav
━E ├─ 001_seed123456_source.json
━E ├─ 002_seed987654_source.wav
━E └─ 002_seed987654_source.json
└─ failed_items.csv            # 失敗があった場合�Eみ
```

キュー状態�E自動保存され，停止めE��プリ再起動後も `Resume Queue` で再開できます！E
## Model Trainer

`Model Trainer` タブでは次を設定します！E
- Character name
- Dataset folder
- Epochs
- Batch size
- Save every
- Workers
- GPU
- Fresh training

学習パイプライン�E�E
```text
Audio dataset
  ↁEPreprocess / 40 kHz
  ↁERMVPE F0
  ↁEHuBERT v2 / 768-dim
  ↁERVC v2 training
  ↁEFAISS index
  ↁEruntime/rvc/exports/<Character>/<Character>.pth
runtime/rvc/exports/<Character>/<Character>.index
```

学習中は Windows の自動スリープを抑止し，終亁E��た�E停止時に解除します！E
## UI の言誁E
説明�Eヘルプ文は `Suomi / 日本誁E/ English` から刁E��替えられます．デフォルト�E Suomi で�E�選択した言語�Eローカルに保存されます！E
ボタン名�E技術用語�E一部の状態表示は意図皁E��英語固定です！E
## セチE��アチE�E診断

環墁E��確認したい場合�E次を実行します！E
```powershell
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

通常機�Eに忁E��な TTS/RVC 環墁E��確認し�E�Docker は Accent 用の任意頁E��として表示します！E
## GitHub Actions

次の workflow を同梱してぁE��す！E
```text
.github/workflows/windows-smoke.yml
```

push / pull request ごとにGitHub側で�E�E
- `setup.ps1` / `doctor.ps1` / `install_accent_support.ps1` の PowerShell 構文
- Go 製 Windows ランチャーのビルチE- 軽釁ECI セチE��アチE�Eと主要EPython ファイルの compile
- Flask UI の起勁E- `/api/state` ぁEHTTP 200 を返すぁE
を�E動チェチE��します！E
通常の GitHub-hosted runner では CUDA を使った実際の TTS/RVC 生�EめEDocker を使ぁE��制アライメントまではチE��トしません�E�GPU 機�Eと Accent 機�Eは Windows 実機でも確認してください�E�E
## GitHub へ含めなぁE��の

`.gitignore` では主に次を除外してぁE��す！E
```text
runtime/
config.json
outputs/
temp/
__pycache__/
```

そ�Eため�E�ダウンロード済みモチE���E�venv�E�ユーザー設定，RVC モチE���E�生戁EWAV などは通常 repository へ入りません�E�E
## 音声・キャラクターの権利

Zundanen 本体�E MIT ライセンスは�E�第三老E�Eキャラクター音声・学習データ・RVC モチE��等�E利用権を与えるものではありません�E�E
第三老E�E学習データ�E�`.pth` / `.index` を利用・公開�E再�E币E��る場合�E�E�それぞれ�E権利老E�E音声ライブラリ・チE�EタセチE��の利用規紁E��確認してください�E�E
## 上流�EロジェクチE
- Finnish-NLP/Chatterbox-Finnish: https://huggingface.co/Finnish-NLP/Chatterbox-Finnish
- ResembleAI Chatterbox: https://github.com/resemble-ai/chatterbox
- RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- Aalto Finnish Forced Alignment: https://github.com/aalto-speech/finnish-forced-alignment
- Lingsoft aalto-kaldi-align-elg: https://github.com/lingsoft/aalto-kaldi-align-elg
- WORLD: https://github.com/mmorise/World
- PyWORLD: https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder
- FFmpeg: https://ffmpeg.org/

ライセンス惁E��は `THIRD_PARTY_NOTICES.md` も確認してください�E�E
