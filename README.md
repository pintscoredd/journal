# SPX 0DTE Journal

A specialized personal trading journal for SPX 0DTE options day trading. It features robust quant-based trade scoring, Implied Volatility inversion with solver fallbacks, Block Bootstrap Monte Carlo simulation, and encrypted local database for secure AI key management.

## Setup

1. Check out the repository.
2. Install the necessary dependencies into a Python virtual environment (Python 3.10+):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Application Locally

Start the Streamlit dashboard:

```bash
PYTHONPATH=. streamlit run app.py
```

## Testing

To run the unit tests in a TDD fashion (tests cover Quant analytics with time-floor checks, Monte Carlo simulations, and more):

```bash
PYTHONPATH=. pytest test_*.py
```

## Secure API Keys & Database

API keys for AI critique generation (Gemini, Groq) are AES-encrypted using the `cryptography` library and stored entirely locally in your SQLite Database `data/journal.db`.
You MUST supply a Master Password or Master Key to unlock encryption/decryption when you run the app.

1) Generate a master key locally:
   python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
2) In Streamlit Cloud, open App settings, paste keys into Secrets, for example:

[ai]
gemini_api_key = "PASTE_KEY_HERE"

[app]
master_key = "PASTE_MASTER_KEY_HERE"

## Streamlit Cloud Deployment

1. Push this repository to GitHub.
2. Log into Streamlit Community Cloud and 'Deploy an app'.
3. Select this repository and specify `app.py` as the main script.
4. Input the Master Key (and alternatively pure API keys) into the Advanced Settings -> Secrets box using the TOML format described above.
5. In your locally running GUI under "Settings & AI Keys", you can click the export button to generate the encrypted TOML snippet needed for cloud database matching, if you prefer storing encrypted blobs over raw keys.

## Features Checklist

- **Ingestion**: Supports custom raw CSV mapping or Robinhood generic formats.
- **Enrichment**: Dynamically fetches 1m & 5m charts from Yahoo Finance and calculates real-time holding PnL alongside VWAP regressions.
- **Quant Check**: Employs Newton-Raphson + Bisection solvers to derive volatility surfaces and scores trades linearly 0-100 based on Edge, Decay, and Sizing Risk.
- **Risk Analysis**: Employs block-bootstrapping to synthesize an Expected Ruin path matrix spanning 10,000 simulations.
- **AI Copilot**: Secure local LLM pipeline to analyze your execution behavior.
