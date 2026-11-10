# Zundanen v2.0.0

**Suomi | [日本語](README_JA.md) | [English](README_EN.md)**

Zundanen on paikallinen Windows-sovellus, joka tekee suomenkielisestä tekstistä WAV-tiedostoja.

Versiossa 2.0.0 puhesynteesi on korvattu **omilla VITS-malleilla (FinBERT-piirteillä)**. Tämä repositorio sisältää **Chatterbox**-mallin (suomen VITS, vain Normal). `Use RVC voice conversion` -toiminnolla ja omalla RVC-mallilla se puhuu toisella äänellä (esimerkiksi Zundamonin). Tunnetyylejä (Normal / Ama / Tsun / Sexy) käyttävät hahmomallit koulutetaan Model Trainerilla omasta datasta; hahmomalleja ei ole mukana.

Samassa käyttöliittymässä ovat **Batch Generation**, sanakohtainen **Accent**-muokkaus, **Model Trainer** (RVC v2 ja VITS-hienosäätö) sekä **Voice Conversion** (WAV- ja kansiomuunnos, nauhoitus, reaaliaikainen muunnos).

> Kansion `docs/` kuvakaappaukset ovat versiosta 1.1.0.

## Käyttöohjevideo (v1)

[https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY](https://youtu.be/e5rfWdmxM20?si=ouU6rzhPJ9XnrOmY)

## Tärkeimmät muutokset versioon 1 verrattuna

- Puhesynteesi: v1 käytti ketjua ”Chatterbox-Finnish → RVC”; v2 käyttää **suomen VITS-mallia tunnetyyleineen**.
- **Character**: valitse koulutettu malli.
  - **Chatterbox** (mukana): suomen VITS Chatterboxin äänellä (vain Normal). `Use RVC` muuntaa sen Zundamonin tai muun RVC-äänen kaltaiseksi.
  - **Itse koulutetut mallit** (Model Trainer → VITS): hahmon ääni tunnetyyleillä. Hahmomalleja ei ole mukana (katso ”Mallitiedostot”).
- **Style**: Normal / Ama / Tsun / Sexy, voimakkuutta (Style strength) voi säätää.
- **Pitch / Speed / Seed**.
- **Use RVC voice conversion**: valinnainen RVC-muunnos luodulle puheelle.
- **Batch Generation** tukee Characteria, Styleä ja RVC:tä.
- **Model Trainer**: RVC (kuten v1:ssä) sekä VITS-hienosäätö (Joint / Finnish / Style).
- Uusi **Voice Conversion** -välilehti: WAV-tiedoston tai kansion RVC-muunnos, nauhoita ja muunna, reaaliaikainen muunnos sovelluksen sisällä.

## Vaatimukset

- Windows 11 x64
- WinGet (yleensä mukana Windows 11:n App Installerissa)
- NVIDIA GPU ei ole pakollinen: puheen luonti toimii myös pelkällä CPU:lla (testattu koneella ilman GPU:ta)
- Model Trainer vaatii NVIDIA CUDA GPU:n (kehitetty ja testattu RTX 3060 12 GB:llä)
- Reaaliaikaiseen äänenmuunnokseen suositellaan NVIDIA GPU:ta
- Vapaata levytilaa suositellaan kymmeniä gigatavuja (ympäristöt, mallit, tuotokset)
- **Ensimmäisellä puheen luonnilla** sovellus lataa Hugging Facesta suomen BERT-mallin (`TurkuNLP/bert-large-finnish-cased-v1`, noin 1,4 Gt); internet-yhteys tarvitaan.

## Asennus

Kuten v1:ssä: lataa `Code → Download ZIP`, pura lyhyeen polkuun (esim. `C:\Zundanen`) ja kaksoisnapsauta `setup.bat`.

Normaali asennus valmistelee:

```text
Microsoft Visual C++ Redistributable
Microsoft Edge WebView2 Runtime
Git
FFmpeg
Python 3.11 + Finnish-NLP/Chatterbox-Finnish (TTS-ympäristö; myös VITS ja FinBERT toimivat tässä)
Python 3.12 + RVC-WebUI
HuBERT / RMVPE / RVC pretrained assets
PyWORLD / WORLD vocoder
Windows-natiivi Aalto/Kaldi runtime
Intel oneMKL runtime tarvittaessa
```

Asennuksen uudelleenajo, täysi uudelleenrakennus ja `doctor.ps1`-tarkistus toimivat kuten v1:ssä:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1 -Force
powershell -ExecutionPolicy Bypass -File .\doctor.ps1
```

## Mallitiedostot

Tässä repositoriossa on mukana:

```text
models/
└─ chatterbox.pt     Chatterbox, noin 68 Mt
```

- Pelkät päättelyyn tarvittavat painot (ilman diskriminaattoria ja optimoijan tilaa), tallennettu float16-muodossa, jotta tiedosto mahtuu GitHubin 100 Mt:n rajaan; sovellus lataa ne float32-muodossa.
- **Ei mukana**: tunnetyylejä käyttävät hahmomallit (esimerkiksi Zundamon tai Shikoku Metan) eikä RVC-malleja (`.pth` / `.index`). Niiden koulutusdatan oikeudet kuuluvat kunkin oikeudenhaltijalle. Kouluta oma mallisi Model Trainerilla datasta, jota sinulla on oikeus käyttää; koulutettu malli lisätään `Character`-valikkoon automaattisesti.
- **Hahmomallit** (eivät kuulu tähän repositorioon): Zundamonin ja Shikoku Metanin tunnemallit ja RVC-mallit jaetaan erikseen tutkimuskäyttöön [Hugging Facessa (Zundanen/finnish-emotion-vits)](https://huggingface.co/Zundanen/finnish-emotion-vits). Ne on tarkoitettu vain ei-kaupalliseen tutkimuskäyttöön; noudata niitä käyttäessäsi [SSS:n multimodaalisen tietokannan käyttöehtoja](https://zunko.jp/multimodal_dev/login.php). Laita hahmotiedosto `<nimi>.pt` kansioon `models/characters` ja sen RVC-malli (`<nimi>.pth` ja `.index`) kansioon `runtime/rvc/exports/<nimi>/` ja käynnistä sovellus uudelleen. Hahmo näkyy silloin `Character`-valikossa ja RVC-malli RVC-mallien luettelossa. Jokainen hahmo on yksi tiedosto.
- Sijoita RVC-mallit kuten v1:ssä kansioon `runtime/rvc/exports/<nimi>/`.

## Käynnistys

Kuten v1:ssä: `Zundanen.exe` (suositeltu) tai `run_ui.bat` (selain, `http://127.0.0.1:8765`).

## Voice Generation

1. Valitse `Character`
2. Valitse `Style` ja `Style strength` (Chatterbox: vain Normal)
3. Kirjoita suomenkielinen teksti
4. Säädä tarvittaessa `Pitch` (puolisävelaskelia hahmon perustasosta, ±10) ja `Speed` (1,0 = normaali, suurempi = nopeampi)
5. Tarkista `Seed` (kun `Fix seed` on pois päältä, jokainen luonti saa uuden siemenen)
6. Halutessasi ota käyttöön `Use RVC voice conversion` ja valitse RVC-malli sekä `Pitch` / `Index rate` / `Protect`
7. Napsauta `Generate voice`
8. Muokkaa tarvittaessa Accentia ja tallenna `Save As...`

### Suositellut asetukset

| Käyttö | Character | Style | RVC |
|---|---|---|---|
| Selkein Normal | Chatterbox | Normal | oma RVC-malli (Zundamon: Pitch **+8**), Index rate 0,1, Protect 0,15 |
| Tunnetyylit | itse koulutettu malli | Ama / Tsun / Sexy | ei käytössä |

- Kun valitset Chatterboxin, RVC:n Pitch 0 muuttuu automaattisesti arvoon +8 (Zundamonin korkeus). Chatterboxissa ei ole korkeuden säätöä, joten ylempi `Pitch`-liukusäädin ei ole silloin käytössä.
- Suuri RVC:n `Index rate` tekee ääntämisestä japanilaisemman; suomelle suositellaan noin arvoa 0,1.

### Ääntämisen mittaus (kehityksen aikana)

Whisper large-v2:n merkkivirhesuhde (CER, pienempi on parempi) 60 suomenkieliselle Tatoeba-lauseelle, joita ei ole käytetty koulutuksessa, kaksi siementä kullekin.

| Ääni | CER |
|---|---:|
| Alkuperäinen Chatterbox-Finnish-ääni (käytännön yläraja) | 2,5 % |
| Chatterbox | 2,3 % |
| Chatterbox + Zundamon RVC (+8, Protect 0,15) | 4,7 % |
| Zundamon Normal | 7,1 % |
| Zundamon Ama / Tsun / Sexy | 9,2 % / 7,8 % / 8,1 % |

Zundamon-rivit ovat Hugging Facessa erikseen tutkimuskäyttöön jaetun Zundamonin tunnemallin tuloksia.

Sanat, joissa on h + konsonantti (esim. *tyhjä*, *kissanhiekka*), ovat tunnetusti heikkoja Zundamonin äänellä.

## Batch Generation

Luo useita tekstejä peräkkäin samoilla Character / Style / Style strength / Pitch / Speed / Seed / RVC -asetuksilla kuin Voice Generation. Myös Chatterboxin voi valita.

CSV-sarakkeet:

```text
filename,text,style,strength,speed,seed
```

Vain `text` on pakollinen; tyhjät kentät käyttävät näytön asetuksia, ja jokaisella rivillä voi olla oma tyyli (`speed` 1,0 = normaali; vanhat `length_scale`-sarakkeet luetaan yhä).

## Model Trainer

Vaihda yläreunan valinnoilla `VITS` / `RVC`.

- **RVC**: kuten v1:ssä (Character name, Dataset folder, Epochs, Batch size, Workers, Fresh training). Luo `.pth`- ja `.index`-tiedostot kansioon `runtime/rvc/exports/<Character>/`.
- **VITS** (edistyneille):
  - `Joint training` / `Finnish fine-tuning` kouluttavat perusmallin → tyylihaaran → äänenlaadun erot ja lisäävät tuloksen Characteriin mallin nimellä. **Ne jatkavat aiemmin koulutettua mallia (mukana ei ole yhtään); aloita alla olevalla `New model (from scratch)` -valinnalla.**
  - `Style fine-tuning` lisää valitulle mallille uuden tyylin pareittaisista Normal- ja tyylinauhoituksista (WAV + LAB, paritus numeron mukaan, vähintään 20 paria).
  - **Uusi malli (tyhjästä)**: valitse `Joint training` -tilan `Base model` -kohtaan `New model (from scratch)`, niin malli koulutetaan satunnaisista painoista. Valittavat:
    - **Finnish folder**: suomenkieliset WAV-tiedostot ja samannimiset TXT-tiedostot (vähintään 50 paria; hyvään ääntämiseen noin 20 tuntia tai enemmän)
    - **Emotion data**: yksi rivi puhujaa ja tyyliä kohden (WAV-kansio, LAB-kansio – tyhjä = sama kansio –, puhujan nimi ja tyyli Normal / Ama / Tsun / Sexy); lisää rivejä `+`-painikkeella. Jokainen puhuja tarvitsee Normal-rivin. LAB-tiedostolla on sama nimi kuin WAV-tiedostolla; rivi ilman omia LAB-tiedostoja käyttää saman tyylin toisen rivin samannimistä LAB-tiedostoa.
    - **Bridge RVC model (.pth)** (Browse…) ja **Character**: kansion kaikki suomenkieliset lauseet muunnetaan tälle RVC-äänelle (kukin kahdella korkeudella, noin 1 s tiedostoa kohden; 20 000 lausetta ≈ 11 h), jotta hahmo oppii suomea. Character valitaan syötetyistä puhujista (valitaan valmiiksi, kun tiedostonimi täsmää, esim. Zundamon.pth → zundamon). `RVC bridge share` määrää, kuinka usein muunnettuja lauseita käytetään koulutuksessa. Vieressä oleva .index käytetään, jos se löytyy; sävelkorkeuden siirto asetetaan automaattisesti (Index 0,1 / Protect 0,15).
    - Epoch (VITS) -oletus on 35. Tämä kestää paljon kauemmin kuin olemassa olevan mallin jatkaminen (usein vuorokauden tai enemmän).
  - Koulutuksen pituus annetaan epookkeina (Epoch (VITS) oletus 4,5, Epoch (Style) oletus 80); Batch size ja Workers voi asettaa oman tietokoneen mukaan.

## Voice Conversion

- Vasemmalla: RVC-malli sekä Pitch / Index rate / Protect
- WAV conversion: yksi tiedosto tai kansion kaikki äänitiedostot (tallennus kansioon `outputs/voice_conversion/`; alkuperäisiä ei korvata)
- Recording: nauhoita, muunna ja tallenna `Save As...`
- Realtime voice conversion: valitse tulo- ja lähtölaite ja paina Start / Stop (ei erillistä ikkunaa; viive noin 0,3 s)

## Accent

Kuten v1.1.0:ssa (Windows-natiivi Aalto/Kaldi-kohdistus, muokkaus WORLD/PyWORLD-kirjastolla; ei WSL2:ta eikä Docker Desktopia).

## Käyttöliittymän kieli

Ohjetekstit vaihtuvat kielillä `Suomi / 日本語 / English`. Painikkeet ja kenttien nimet ovat englanniksi.

## Ei mukana GitHubissa

`.gitignore` rajaa pois:

```text
runtime/  config.json  outputs/  temp/  __pycache__/  *.pt (paitsi models/chatterbox.pt)  *.pth  *.index  *.safetensors  *.wav  local_state/
```

## Äänten, hahmojen ja koulutusdatan oikeudet

Zundanenin MIT-lisenssi ei anna oikeuksia kolmansien osapuolten hahmoääniin, koulutusdataan, koulutettuihin malleihin tai RVC-malleihin.

Mukana oleva Chatterbox-malli (`models/chatterbox.pt`) on koulutettu pelkästään puheella, jonka Chatterbox-Finnish (MIT) on luonut suomenkielisistä Tatoeba-lauseista (CC BY 2.0 FR). Äänimallina käytettiin Chatterbox-Finnishin mukana tulevaa referenssiääntä (`samples/reference_finnish.wav`, Mozilla Common Voice, CC0 1.0). Japanilaisia hahmoäänikorpuksia ei käytetty.

Itse koulutettuihin malleihin ja RVC-malleihin sovelletaan niiden koulutusdatan ehtoja. Tarkista kunkin oikeudenhaltijan ehdot ennen niiden käyttöä, julkaisua tai uudelleenjakelua (`THIRD_PARTY_NOTICES.md`).

## Lähdeprojektit

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

Katso myös `THIRD_PARTY_NOTICES.md`.
