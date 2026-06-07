# DEV.md

Project conventions and patterns for YaGGUF. (Per the repo's `CLAUDE.md`, this is the
general home for dev notes; prefer adding cross-cutting conventions here.)

## Dependency Policy

Split `requirements.txt` pins by *who owns the API contract*:

**Cap the libraries YaGGUF imports directly.** `streamlit`, `huggingface-hub`, `gguf`,
`numpy` (and `colorama`, `mistral-common`). A breaking major in these breaks our own code,
so they get an upper cap (`<MAJOR.0.0`) and we bump it deliberately — with a test — as part
of a release, typically alongside the llama.cpp version bump.

**Float the conversion tools (floor-only).** `transformers`, `tokenizers`, `sentencepiece`.
YaGGUF never imports these directly — llama.cpp's `convert_hf_to_gguf.py` does. Their
required version is dictated by *the model + that script*, not by us, and it must move
forward to support new architectures. An upper cap here silently strands newest-model
support and, worse, makes the **Update Dependencies** tab unable to fix it (it installs
`-r requirements.txt`, so it can't cross a cap). This is exactly what happened with Gemma 4:
`transformers<5.0.0` blocked the 5.x release that added `gemma4_unified`. `tokenizers` is
additionally constrained by `transformers` itself, so a cap on it is mostly redundant.

**The residual maintenance point you can't design away:** a future `transformers` will
eventually require a new `huggingface-hub` *major* (5.x already pulled hub 1.x). Because we
cap `huggingface-hub`, that collides with our cap and surfaces as a **loud pip resolver
error** during an update — not silent breakage. The fix is a deliberate hub cap bump + a
test pass on the downloader. This is the accepted cost of importing hub's API directly.

**When a model is newer than any released `transformers`:** the conversion error hint
(`converter.py::_transformers_upgrade_hint`) points users at a git-source install
(`pip install -U git+https://github.com/huggingface/transformers.git`) as a fallback. With
floor-only pins this is rarely needed, but it covers the gap before a release lands.
