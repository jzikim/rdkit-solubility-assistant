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

> Note: The app is currently running locally at `http://localhost:8501`, but other students can only use a public URL after deployment.

## Student Contribution Guide

If you want your classmates to be able to improve the app, tell them to:

1. Fork this repository on GitHub.
2. Clone their fork and create a feature branch:

```bash
git clone https://github.com/<your-username>/rdkit-solubility-assistant.git
cd rdkit-solubility-assistant
git checkout -b feat/your-change
```

3. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

4. Run the app locally and confirm it works:

```bash
.venv\Scripts\python.exe -m streamlit run app.py
```

5. Make code changes, then commit and push the branch to their fork.
6. Create a Pull Request (PR) from their fork to `jzikim/rdkit-solubility-assistant` `main`.

This `fork + PR` workflow is the safest way for students to contribute while keeping the main repo clean.

## What to Tell Friends on Kakao

You can copy and send this message:

> 이 앱은 `PubChem`에서 SMILES를 가져와 RDKit descriptor를 만들고, scikit-learn `RandomForestRegressor`로 LogS(용해도) 예측을 해 주는 교육용 프로젝트야. 직접 수정해 보고 싶으면 이 리포지토리를 포크해서 `feat/` 브랜치 만들고 코드를 바꾼 다음 PR 보내면 돼. 먼저 로컬에서 `python -m venv .venv`, `pip install -r requirements.txt`, `.\.venv\Scripts\python.exe -m streamlit run app.py`로 확인해 봐.

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

## Property Regression Demo (educational)

This repository includes a small demo showing how to train a scikit-learn regression
model to predict a chemical property from SMILES using RDKit features.

- Dataset: `data/molecule_dataset.csv` (columns: `name,smiles,target_property`). If `target_property` is empty, the training script will fall back to RDKit-calculated LogP for demonstration purposes.
- Featurization: `src/features.py` — RDKit descriptors + Morgan fingerprint.
- Train: `src/train_property_model.py` — trains a RandomForest pipeline and saves `models/property_model.pkl`.
- Predict: `src/predict_property.py` — load model and predict for a given SMILES.

Example train run:

```bash
python -m src.train_property_model --csv data/molecule_dataset.csv --target target_property
```

Example predict run:

```bash
python -m src.predict_property --smiles "CCO"
```

Notes:
- This demo is educational. Do not use model predictions for medical or safety decisions.
- With very small datasets the model will be noisy; collect more labeled examples for better performance.

