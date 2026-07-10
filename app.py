import os

import joblib
import pandas as pd
import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors

from predict_logs import predict_logS_from_smiles

try:
    from google import genai
except ImportError:
    genai = None


CONTROLLED_ABUSE_RISK_DRUGS = {
    "fentanyl",
    "morphine",
    "diazepam",
    "codeine",
    "oxycodone",
    "hydrocodone",
    "tramadol",
    "alprazolam",
    "lorazepam",
    "methadone",
}


def get_google_api_key():
    """Read the Gemini API key from Streamlit secrets first, then environment variables."""
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        # st.secrets may not exist on a local machine yet. That is okay.
        pass

    return os.getenv("GOOGLE_API_KEY")


@st.cache_data(show_spinner=False)
def fetch_pubchem_compound(drug_name):
    """Find the first PubChem compound matching the English drug name."""
    compounds = pcp.get_compounds(drug_name, "name")

    if not compounds:
        raise ValueError("PubChem에서 해당 약물 이름을 찾지 못했습니다.")

    compound = compounds[0]

    if not compound.canonical_smiles:
        raise ValueError("PubChem 결과에 Canonical SMILES가 없습니다.")

    return {
        "name": drug_name,
        "cid": compound.cid,
        "canonical_smiles": compound.canonical_smiles,
        "molecular_formula": compound.molecular_formula,
        "pubchem_molecular_weight": compound.molecular_weight,
    }


def calculate_rdkit_descriptors(smiles):
    """Convert SMILES into an RDKit molecule and calculate basic molecular descriptors."""
    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        raise ValueError("RDKit이 SMILES 구조를 읽지 못했습니다.")

    return {
        "Molecular Weight": Descriptors.MolWt(molecule),
        "LogP": Crippen.MolLogP(molecule),
        "TPSA": rdMolDescriptors.CalcTPSA(molecule),
        "H-bond Donors": Lipinski.NumHDonors(molecule),
        "H-bond Acceptors": Lipinski.NumHAcceptors(molecule),
        "Rotatable Bonds": Lipinski.NumRotatableBonds(molecule),
    }


def calculate_risk_score(drug_name, descriptors):
    """Calculate a simple educational risk score from rule-based chemistry signals."""
    score = 0
    reasons = []

    if descriptors["LogP"] > 3.5:
        score += 1
        reasons.append("LogP가 3.5보다 큽니다.")

    if descriptors["TPSA"] < 40:
        score += 1
        reasons.append("TPSA가 40보다 작습니다.")

    if descriptors["Molecular Weight"] > 500:
        score += 1
        reasons.append("분자량이 500보다 큽니다.")

    if drug_name.strip().lower() in CONTROLLED_ABUSE_RISK_DRUGS:
        score += 2
        reasons.append("주의가 필요한 약물 목록에 포함되어 있습니다.")

    if score == 0:
        level = "Low"
    elif score <= 2:
        level = "Moderate"
    else:
        level = "High"

    return score, level, reasons


def generate_korean_explanation(drug_name, compound_info, descriptors, risk_score, risk_level, reasons):
    """Ask Gemini to explain the result in Korean for high school students."""
    api_key = get_google_api_key()

    if not api_key:
        return make_fallback_korean_explanation(
            drug_name,
            descriptors,
            risk_score,
            risk_level,
            reasons,
            "GOOGLE_API_KEY가 설정되어 있지 않아 Gemini를 호출하지 않았습니다.",
        )

    if genai is None:
        return make_fallback_korean_explanation(
            drug_name,
            descriptors,
            risk_score,
            risk_level,
            reasons,
            "google-genai 패키지가 설치되어 있지 않아 Gemini를 호출하지 않았습니다.",
        )

    reason_text = ", ".join(reasons) if reasons else "위험 점수를 올린 규칙이 없습니다."

    prompt = f"""
You are helping high school students understand a simple educational drug risk analysis app.
Write the explanation in Korean.

Drug name: {drug_name}
PubChem CID: {compound_info["cid"]}
Canonical SMILES: {compound_info["canonical_smiles"]}
Molecular formula: {compound_info["molecular_formula"]}

RDKit descriptors:
- Molecular Weight: {descriptors["Molecular Weight"]:.2f}
- LogP: {descriptors["LogP"]:.2f}
- TPSA: {descriptors["TPSA"]:.2f}
- H-bond donors: {descriptors["H-bond Donors"]}
- H-bond acceptors: {descriptors["H-bond Acceptors"]}
- Rotatable bonds: {descriptors["Rotatable Bonds"]}

Risk Score: {risk_score}
Risk Level: {risk_level}
Score reasons: {reason_text}

Requirements:
- Explain LogP, TPSA, and Risk Score in Korean.
- Make it understandable for high school students.
- Clearly say this is an educational rule-based score, not a medical diagnosis.
- Do not provide medical advice, dosage guidance, synthesis instructions, or abuse-related instructions.
- Keep the explanation concise and friendly.
"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            contents=prompt,
        )
        return response.text
    except Exception as error:
        error_text = str(error)

        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            message = (
                "Gemini API 사용량 한도에 도달해서 AI 설명 대신 기본 설명을 표시합니다. "
                "잠시 후 다시 시도하거나 Google AI Studio에서 quota/billing 상태를 확인해 주세요."
            )
        else:
            message = f"Gemini 설명 생성 중 오류가 발생해서 기본 설명을 표시합니다. 오류: {error}"

        return make_fallback_korean_explanation(
            drug_name,
            descriptors,
            risk_score,
            risk_level,
            reasons,
            message,
        )


def make_fallback_korean_explanation(drug_name, descriptors, risk_score, risk_level, reasons, note):
    """Create a Korean explanation without Gemini so the app still works during API errors."""
    reason_text = " ".join(reasons) if reasons else "이번 분석에서는 위험 점수를 올린 규칙이 없었습니다."

    return f"""
{note}

이 결과는 교육용 규칙 기반 분석입니다. 실제 의학적 진단이나 복용 판단에 사용하면 안 됩니다.

{drug_name}의 LogP는 {descriptors["LogP"]:.2f}입니다. LogP는 물보다 지방에 얼마나 잘 녹는지를 나타내는 값입니다. 값이 높으면 몸 안의 지방성 환경에 더 잘 들어갈 가능성이 있습니다.

TPSA는 {descriptors["TPSA"]:.2f}입니다. TPSA는 분자의 극성 표면적을 뜻하며, 분자가 물과 상호작용하거나 생체막을 통과하는 성질과 관련이 있습니다.

이 앱의 Risk Score는 {risk_score}점이고 Risk Level은 {risk_level}입니다. 점수 이유: {reason_text}

이 점수는 LogP, TPSA, 분자량, 그리고 미리 정한 주의 약물 목록만 사용한 단순한 예시입니다. 실제 약물의 안전성은 훨씬 많은 실험 데이터와 전문가 판단이 필요합니다.
"""


def make_descriptor_table(descriptors):
    """Create a beginner-friendly table for Streamlit."""
    rows = [
        ("Molecular Weight", descriptors["Molecular Weight"], "분자의 질량"),
        ("LogP", descriptors["LogP"], "지방에 잘 녹는 정도"),
        ("TPSA", descriptors["TPSA"], "분자의 극성 표면적"),
        ("H-bond Donors", descriptors["H-bond Donors"], "수소 결합을 줄 수 있는 개수"),
        ("H-bond Acceptors", descriptors["H-bond Acceptors"], "수소 결합을 받을 수 있는 개수"),
        ("Rotatable Bonds", descriptors["Rotatable Bonds"], "회전 가능한 결합 수"),
    ]

    return pd.DataFrame(rows, columns=["Descriptor", "Value", "Meaning"])


def make_ml_descriptor_table(descriptors):
    """Create a descriptor table for the ML logS predictor."""
    rows = [
        ("Molecular Weight", descriptors["MolWt"], "분자의 질량"),
        ("MolLogP", descriptors["MolLogP"], "지방에 잘 녹는 정도"),
        ("TPSA", descriptors["TPSA"], "분자의 극성 표면적"),
        ("NumHDonors", descriptors["NumHDonors"], "수소 결합을 줄 수 있는 개수"),
        ("NumHAcceptors", descriptors["NumHAcceptors"], "수소 결합을 받을 수 있는 개수"),
        ("NumRotatableBonds", descriptors["NumRotatableBonds"], "회전 가능한 결합 수"),
        ("RingCount", descriptors["RingCount"], "고리 수"),
        ("HeavyAtomCount", descriptors["HeavyAtomCount"], "무거운 원자 수"),
    ]

    return pd.DataFrame(rows, columns=["Descriptor", "Value", "Meaning"])


def load_local_drug_database():
    """Load the local molecule dataset into a lowercase lookup dictionary."""
    result = {}
    try:
        df = pd.read_csv("data/molecule_dataset.csv")
        if "name" in df.columns and "smiles" in df.columns:
            for _, row in df.iterrows():
                name = str(row["name"]).strip().lower()
                result[name] = {
                    "name": row["name"],
                    "smiles": row["smiles"],
                }
    except Exception:
        pass
    return result


def main():
    st.set_page_config(
        page_title="AI Drug Risk Analyzer",
        page_icon="AI",
        layout="wide",
    )

    st.title("AI Drug Risk Analyzer")
    st.caption("PubChem, RDKit, Gemini를 활용한 교육용 약물 위험도 분석 앱")

    st.warning(
        "이 앱은 교육용 예제입니다. 실제 의학적 판단, 복용, 처방, 진단에 사용하면 안 됩니다."
    )

    drug_name = st.text_input(
        "Enter an English drug name",
        placeholder="Example: aspirin, caffeine, diazepam",
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if not analyze_clicked:
        st.info("영어 약물 이름을 입력하고 Analyze 버튼을 눌러 분석을 시작하세요.")
        return

    if not drug_name.strip():
        st.error("약물 이름을 입력해 주세요.")
        return

    local_data = load_local_drug_database()
    local_entry = local_data.get(drug_name.strip().lower())
    local_found = local_entry is not None

    try:
        if local_found and local_entry.get("smiles"):
            smiles_source = local_entry["smiles"]
        else:
            with st.spinner("PubChem에서 약물 정보를 가져오는 중입니다..."):
                compound_info = fetch_pubchem_compound(drug_name.strip())
            smiles_source = compound_info["canonical_smiles"]

        with st.spinner("RDKit으로 분자 특성을 계산하는 중입니다..."):
            descriptors = calculate_rdkit_descriptors(smiles_source)

        risk_score, risk_level, reasons = calculate_risk_score(drug_name, descriptors)
    except Exception as error:
        st.error(f"분석 중 오류가 발생했습니다: {error}")
        st.stop()

    if local_found:
        st.success("This drug was found in the local dataset.")

    st.subheader("PubChem Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    if not local_found:
        info_col1.metric("PubChem CID", compound_info["cid"])
        info_col2.metric("Formula", compound_info["molecular_formula"] or "N/A")
        info_col3.metric("PubChem MW", compound_info["pubchem_molecular_weight"] or "N/A")

        st.text_area(
            "Canonical SMILES",
            value=compound_info["canonical_smiles"],
            height=80,
            disabled=True,
        )
    else:
        info_col1.metric("Local dataset SMILES", local_entry.get("smiles", "N/A"))
        info_col2.metric("Local dataset name", local_entry.get("name", "N/A"))
        info_col3.metric("Dataset source", "data/molecule_dataset.csv")
        st.text_area(
            "SMILES used for prediction",
            value=smiles_source,
            height=80,
            disabled=True,
        )

    st.subheader("RDKit Descriptors")
    descriptor_table = make_descriptor_table(descriptors)
    st.dataframe(descriptor_table, use_container_width=True, hide_index=True)

    st.subheader("Machine Learning Prediction")
    st.write(
        "이 섹션은 PubChem 또는 로컬 데이터셋에서 가져온 SMILES와 RDKit descriptor를 사용한 `RandomForestRegressor` 기반 LogS 예측 결과를 보여줍니다. "
        "이 결과는 교육용이며 실제 의약적 판단을 대신하지 않습니다."
    )
    model_path = "models/logs_model.pkl"
    if not os.path.exists(model_path):
        st.warning(
            "LogS 모델이 없습니다. 먼저 `python train_logs_model.py`를 실행하여 `models/logs_model.pkl`을 생성하세요."
        )
    else:
        try:
            with st.spinner("LogS 예측 모델을 실행하는 중입니다..."):
                ml_result = predict_logS_from_smiles(compound_info["canonical_smiles"], model_path=model_path)

            st.markdown(
                """**Model:** RandomForestRegressor  
**Target:** LogS / aqueous solubility  
**Input source:** PubChem SMILES + RDKit descriptors"""
            )
            st.metric("Predicted logS", f"{ml_result['predicted_logS']:.3f}")
            st.write("Solubility interpretation:", ml_result["interpretation"])
            st.info(
                "This drug was not found in the local database. The result below was generated using PubChem SMILES, RDKit descriptors, and a scikit-learn ML model."
            )
            ml_descriptor_table = make_ml_descriptor_table(ml_result["descriptors"])
            st.dataframe(ml_descriptor_table, use_container_width=True, hide_index=True)
        except Exception as error:
            st.error(f"Machine Learning Prediction 오류: {error}")

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Risk Score", risk_score)
    metric_col2.metric("Risk Level", risk_level)
    metric_col3.metric("LogP", f"{descriptors['LogP']:.2f}")

    st.subheader("Descriptor Bar Chart")
    chart_data = descriptor_table.set_index("Descriptor")[["Value"]]
    st.bar_chart(chart_data)

    st.subheader("Risk Score Reasons")
    if reasons:
        for reason in reasons:
            st.write(f"- {reason}")
    else:
        st.write("- 위험 점수를 올린 규칙이 없습니다.")

    st.subheader("Gemini Korean Explanation")
    with st.spinner("Gemini가 한국어 설명을 작성하는 중입니다..."):
        explanation = generate_korean_explanation(
            drug_name=drug_name.strip(),
            compound_info=compound_info,
            descriptors=descriptors,
            risk_score=risk_score,
            risk_level=risk_level,
            reasons=reasons,
        )

    st.write(explanation)


if __name__ == "__main__":
    main()
