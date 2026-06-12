# AI Drug Risk Analyzer

A beginner-friendly Streamlit web app that analyzes an English drug name using PubChem, RDKit, and Gemini.

The app fetches basic compound data from PubChem, calculates molecular descriptors with RDKit, assigns a simple rule-based risk score, and asks Gemini to explain the result in Korean for high school students.

## Features

- Search a drug by English name using PubChemPy
- Display PubChem CID, Canonical SMILES, molecular formula, and molecular weight
- Calculate RDKit descriptors:
  - Molecular Weight
  - LogP
  - TPSA
  - H-bond donors
  - H-bond acceptors
  - Rotatable bonds
- Calculate a simple educational risk score
- Generate a Korean explanation with Gemini
- Show results with Streamlit metrics, a table, and a bar chart

## Risk Score Rules

- `LogP > 3.5`: +1
- `TPSA < 40`: +1
- `Molecular Weight > 500`: +1
- Drug name is in the controlled/abuse-risk list: +2

Controlled/abuse-risk list:

`fentanyl, morphine, diazepam, codeine, oxycodone, hydrocodone, tramadol, alprazolam, lorazepam, methadone`

Risk levels:

- `0`: Low
- `1-2`: Moderate
- `3 or higher`: High

## Important Safety Note

This app is for education only. It is not medical advice, diagnosis, prescription guidance, or a safety guarantee. Do not use it to decide whether a drug is safe to take.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Gemini API Key

Do not hard-code your API key in the code.

Option 1: Environment variable

```powershell
$env:GOOGLE_API_KEY="your-api-key-here"
```

Option 2: Streamlit secrets

Create `.streamlit/secrets.toml`:

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

Optional: choose a Gemini model with an environment variable.

```powershell
$env:GEMINI_MODEL="gemini-2.0-flash"
```

If the Gemini API quota is exceeded, the app still shows a built-in Korean explanation so students can continue testing the project.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Deploy With Streamlit Community Cloud

GitHub Pages cannot run this app because Streamlit needs a Python server. Use Streamlit Community Cloud instead.

1. Push this project to GitHub.
2. Go to `https://share.streamlit.io`.
3. Click `Create app`.
4. Choose this repository.
5. Set the main file path to `app.py`.
6. Add this secret in the app settings:

```toml
GOOGLE_API_KEY = "your-api-key-here"
```

7. Click `Deploy`.

After deployment, Streamlit will give you a public link that students can open in their browsers.

## Example Drug Names

- `aspirin`
- `caffeine`
- `ibuprofen`
- `diazepam`

## Project Structure

```text
.
|-- app.py
|-- requirements.txt
|-- README.md
`-- .gitignore
```
