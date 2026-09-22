# Third-party notices

Zundanen does not redistribute the following upstream projects or their model files in this repository. The normal setup and the optional Accent support installer download required components from their upstream distribution locations.

- **Finnish-NLP/Chatterbox-Finnish** — MIT license  
  https://huggingface.co/Finnish-NLP/Chatterbox-Finnish
- **ResembleAI Chatterbox** — MIT license  
  https://github.com/resemble-ai/chatterbox
- **Retrieval-based-Voice-Conversion-WebUI (RVC)** — MIT license / upstream notices apply  
  https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- **FFmpeg (Gyan Windows build)** — FFmpeg/Gyan build licensing applies  
  https://www.gyan.dev/ffmpeg/builds/  
  https://ffmpeg.org/legal.html
- **Python** — Python Software Foundation license  
  https://www.python.org/
- **PyTorch** — BSD-style license  
  https://pytorch.org/
- **Docker Desktop** — Docker Subscription Service Agreement applies  
  https://docs.docker.com/subscription-billing/desktop-license/  
  Used only by the optional Accent support. Docker Desktop is not redistributed
  in this repository and is installed separately through WinGet.
- **Aalto Speech / Finnish Forced Alignment** — MIT license  
  https://github.com/aalto-speech/finnish-forced-alignment
- **Lingsoft / aalto-kaldi-align-elg** — MIT license  
  https://github.com/lingsoft/aalto-kaldi-align-elg  
  Optional Docker API wrapper used by Accent support for Finnish word/syllable alignment. The Docker image is downloaded separately by the Accent support installer and is not redistributed in this repository.
- **WORLD Vocoder** — modified BSD license  
  https://github.com/mmorise/World  
  Used by the optional Accent editor for speech analysis, manipulation, and resynthesis.
- **PyWORLD** — MIT license  
  https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder  
  Python wrapper installed into the local RVC environment only when optional Accent support is installed; not redistributed in this repository.

Character voices, datasets, RVC `.pth` files and `.index` files may carry separate rights or usage conditions that are not granted by this application's license.


## Desktop launcher and UI dependencies

- **pywebview** — BSD 3-Clause License  
  https://github.com/r0x0r/pywebview
- **Microsoft Edge WebView2 Runtime** — Microsoft distribution/license terms apply  
  https://learn.microsoft.com/microsoft-edge/webview2/concepts/distribution
- **Go standard library/runtime** — BSD-style License  
  https://go.dev/LICENSE

`Zundanen.exe` is a small launcher compiled with Go and therefore contains portions of the Go runtime/standard library.

### Go license

Copyright 2009 The Go Authors.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Google LLC nor the names of its contributors may
  be used to endorse or promote products derived from this software
  without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
