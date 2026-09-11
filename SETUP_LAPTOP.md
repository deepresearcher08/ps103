# Run PS103 on another device

Everything needed (code, MoSPI data, trained models) is in the repo — just clone and run.
Requirements: Python 3.11 or 3.12 (do NOT use 3.14 unless you're ready to fight Windows Smart App Control), and 4+ GB free RAM. The AI chat needs a GPU to be fast (CPU works, just slower).

## Windows (PowerShell)

```powershell
# 1. Get code + data + models
git clone https://github.com/deepresearcher08/ps103.git
cd ps103

# 2. Dependencies
pip install -r requirements.txt

# 3. Optional: local AI assistant (skip if you want the fallback engine)
#    Install Ollama from https://ollama.com/download, then:
ollama pull llama3.1:8b

# 4. Run
streamlit run src/dashboard.py
```
Open http://localhost:8501

## macOS / Linux

```bash
git clone https://github.com/deepresearcher08/ps103.git
cd ps103
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optional AI: install Ollama (https://ollama.com/download) then: ollama pull llama3.1:8b
streamlit run src/dashboard.py
```

## Share it live (temporary public link, runs from this device)

```bash
# install once: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/
cloudflared tunnel --url http://localhost:8501 --protocol http2
```
Cloudflare quick tunnels print a `https://xxxx.trycloudflare.com` URL — that's the shareable link. It stays live only while this PC + the cloudflared process run.

## Checks if something's off

- App won't start → missing `data/` or `data/models/`: re-clone (they're tracked).
- Model errors → your pandas/sklearn versions too new/old: install `pandas>=2.0`, `scikit-learn>=1.3`, `lightgbm>=4.0`.
- Chat replies with non-AI answers only → Ollama not running (`ollama list` should show `llama3.1:8b`).
- Windows only: if pages error with `DLL load failed ... Application Control policy`, your Windows Smart App Control is blocking pip-installed binaries — turn it off (Settings → Windows Security → App & browser control → Smart App Control → Off) then reinstall: `pip install --force-reinstall scipy numpy`.