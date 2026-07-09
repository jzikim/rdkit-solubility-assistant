import os

import pandas as pd
import pubchempy as pcp
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
import joblib
import numpy as np

from src import features
from src import train_logs_model as logs_trainer
from src import train_property_model as trainer

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
    # Retry a few times for transient network errors (e.g., WinError 10054)
    import time

    last_err = None
    for attempt in range(3):
        try:
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
        except Exception as e:
            last_err = e
            # common transient network issue: connection reset by peer
            if attempt < 2:
                time.sleep(1 + attempt * 2)
                continue
            # on final attempt raise a helpful error
            msg = str(e)
            if 'WinError 10054' in msg or 'ConnectionResetError' in msg or 'connection was forcibly closed' in msg.lower():
                raise ValueError(
                    "네트워크 연결이 끊겼습니다 (WinError 10054). 인터넷 연결을 확인하거나 SMILES를 직접 붙여넣어 주세요."
                ) from e
            raise


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
        key='drug_name',
    )

    analyze_clicked = st.button("Analyze", type="primary")

    if not analyze_clicked:
        st.info("영어 약물 이름을 입력하고 Analyze 버튼을 눌러 분석을 시작하세요.")
        return

    if not drug_name.strip():
        st.error("약물 이름을 입력해 주세요.")
        return

    try:
        with st.spinner("PubChem에서 약물 정보를 가져오는 중입니다..."):
            compound_info = fetch_pubchem_compound(drug_name.strip())

        with st.spinner("RDKit으로 분자 특성을 계산하는 중입니다..."):
            descriptors = calculate_rdkit_descriptors(compound_info["canonical_smiles"])

        risk_score, risk_level, reasons = calculate_risk_score(drug_name, descriptors)

    except Exception as error:
        st.error(f"분석 중 오류가 발생했습니다: {error}")
        st.stop()

    st.subheader("PubChem Information")
    info_col1, info_col2, info_col3 = st.columns(3)
    info_col1.metric("PubChem CID", compound_info["cid"])
    info_col2.metric("Formula", compound_info["molecular_formula"] or "N/A")
    info_col3.metric("PubChem MW", compound_info["pubchem_molecular_weight"] or "N/A")

    st.text_area(
        "Canonical SMILES",
        value=compound_info["canonical_smiles"],
        height=80,
        disabled=True,
    )

    # Simple summary for general users
    st.subheader("Quick Summary")
    sum_col1, sum_col2 = st.columns(2)
    sum_col1.metric("Estimated LogP (RDKit)", f"{descriptors['LogP']:.2f}")

    # Try to auto-load a default model and show an ML estimate in the simple view
    model_path_default = 'models/property_model.pkl'
    simple_ml_shown = False
    if os.path.exists(model_path_default):
        try:
            model_data = joblib.load(model_path_default)
            pipeline = model_data.get('pipeline')
            model_target = model_data.get('target', 'target_property')
            feat, _ = features.featurize_smiles(compound_info['canonical_smiles'])
            if feat is not None and pipeline is not None:
                feat_aligned = features.align_features_for_pipeline(pipeline, feat)
                X = feat_aligned.reshape(1, -1)
                pred = pipeline.predict(X)[0]
                # uncertainty from RandomForest
                std = None
                try:
                    estimator = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
                    if estimator is not None and hasattr(estimator, 'estimators_'):
                        tree_preds = np.array([e.predict(X) for e in estimator.estimators_])
                        std = float(tree_preds.std())
                except Exception:
                    std = None

                # applicability domain
                max_sim = None
                try:
                    train_fps = model_data.get('train_fps', None)
                    fp = features.morgan_fingerprint_array(compound_info['canonical_smiles'])
                    if fp is not None and train_fps is not None:
                        max_sim = features.compute_max_tanimoto(fp, train_fps)
                except Exception:
                    max_sim = None

                # confidence label
                conf = 'Unknown'
                if max_sim is not None and std is not None:
                    if max_sim >= 0.7 and std < 0.2:
                        conf = 'High'
                    elif max_sim >= 0.4 and (std is None or std < 0.5):
                        conf = 'Medium'
                    else:
                        conf = 'Low'
                elif max_sim is not None:
                    if max_sim >= 0.7:
                        conf = 'High'
                    elif max_sim >= 0.4:
                        conf = 'Medium'
                    else:
                        conf = 'Low'

                sum_col2.metric(f"ML estimate ({model_target})", f"{pred:.2f}")
                sum_col2.write(f"Confidence: **{conf}**")
                st.write("ML 예측은 학습 데이터 기반의 추정치입니다. (교육용)")
                simple_ml_shown = True
        except Exception:
            # ignore and fall back to info message
            pass

    if not simple_ml_shown:
        sum_col2.info("모델이 없거나 불러오지 못했습니다. '고급'에서 모델을 불러오거나 학습하세요.")

    # --- LogS prediction (educational) ---
    st.subheader('Predict aqueous solubility (logS)')
    smiles_override = st.text_input('Or paste SMILES (optional)', value='', key='smiles_input_logs')
    logs_model_path = st.text_input('LogS model path', value='models/logs_model.pkl', key='logs_model_path')
    logs_model = None
    if st.button('Load LogS model', key='load_logs_model'):
        try:
            logs_model = joblib.load(logs_model_path)
            st.success(f"Loaded LogS model (target: {logs_model.get('target','logS')})")
        except Exception as e:
            st.error(f"LogS 모델을 불러오는 중 오류가 발생했습니다: {e}")

    # try auto-load
    if logs_model is None and os.path.exists(logs_model_path):
        try:
            logs_model = joblib.load(logs_model_path)
        except Exception:
            logs_model = None

    # determine SMILES to use
    use_smiles = None
    if smiles_override and smiles_override.strip():
        use_smiles = smiles_override.strip()
    else:
        use_smiles = compound_info.get('canonical_smiles')

    if logs_model:
        pipeline = logs_model.get('pipeline')
        try:
            feat, desc = features.featurize_smiles(use_smiles)
            if feat is None:
                st.error('유효한 SMILES로부터 특징을 만들 수 없습니다.')
            else:
                feat_aligned = features.align_features_for_pipeline(pipeline, feat)
                X = feat_aligned.reshape(1, -1)
                pred = pipeline.predict(X)[0]
                # uncertainty
                std = None
                try:
                    estimator = pipeline.named_steps.get('model') if hasattr(pipeline, 'named_steps') else None
                    if estimator is not None and hasattr(estimator, 'estimators_'):
                        tree_preds = np.array([e.predict(X) for e in estimator.estimators_])
                        std = float(tree_preds.std())
                except Exception:
                    std = None

                # applicability
                max_sim = None
                try:
                    train_fps = logs_model.get('train_fps', None)
                    fp = features.morgan_fingerprint_array(use_smiles)
                    if fp is not None and train_fps is not None:
                        max_sim = features.compute_max_tanimoto(fp, train_fps)
                except Exception:
                    max_sim = None

                # interpretation
                if pred >= -1.0:
                    interp = 'Very soluble'
                elif pred >= -3.0:
                    interp = 'Moderately soluble'
                else:
                    interp = 'Poorly soluble'

                st.metric('Predicted logS', f"{pred:.3f}")
                st.write('Solubility interpretation:', interp)
                if std is not None:
                    st.write(f'Prediction std (approx.): {std:.3f}')
                if max_sim is not None:
                    st.write(f'Max Tanimoto similarity to training set: {max_sim:.3f}')

                st.subheader('Key descriptors')
                rd = features.compute_rdkit_descriptors(use_smiles)
                if rd:
                    st.write({
                        'MolWt': rd.get('MolWt'),
                        'LogP': rd.get('LogP'),
                        'TPSA': rd.get('TPSA'),
                        'NumHDonors': rd.get('NumHDonors'),
                        'NumHAcceptors': rd.get('NumHAcceptors'),
                        'NumRotatableBonds': rd.get('NumRotatableBonds'),
                        'RingCount': rd.get('RingCount'),
                        'HeavyAtomCount': rd.get('HeavyAtomCount'),
                    })

                # Gemini explanation hook for LogS
                if st.button('Explain prediction (Gemini)', key='gen_gemini_logs'):
                    if not genai:
                        st.error('google-genai 패키지가 설치되어 있지 않아 Gemini를 사용할 수 없습니다.')
                    else:
                        prompt = f"Explain in Korean (for high school students) what the predicted logS {pred:.3f} means for the molecule with SMILES {use_smiles}. Include simple references to LogP, TPSA, and solubility interpretation. Emphasize this is educational only. Descriptors: {rd}."
                        try:
                            client = genai.Client(api_key=get_google_api_key())
                            response = client.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'), contents=prompt)
                            st.write(response.text)
                        except Exception as e:
                            st.error(f'Gemini 호출 오류: {e}')
        except Exception as e:
            st.error(f'LogS 예측 중 오류: {e}')
    else:
        st.info('LogS 모델이 없습니다. 고급에서 모델을 학습하거나 업로드하세요.')

    # Advanced / technical details hidden by default
    with st.expander('고급(상세) 보기 — 기술적 정보와 모델 조작'):
        st.subheader("RDKit Descriptors")
        descriptor_table = make_descriptor_table(descriptors)
        st.dataframe(descriptor_table, use_container_width=True, hide_index=True)

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Risk Score", risk_score)
        metric_col2.metric("Risk Level", risk_level)
        metric_col3.metric("LogP", f"{descriptors['LogP']:.2f}")

        st.subheader("Descriptor Bar Chart")
        chart_data = descriptor_table.set_index("Descriptor")[['Value']]
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

        # --- ML Property Prediction integration (advanced controls) ---
        st.subheader("ML Property Prediction (고급)")

        model_path = st.text_input('Model path', value=model_path_default, key='model_path_adv')
        model_loaded = None
        if st.button('Load model', key='load_model_adv'):
            try:
                data = joblib.load(model_path)
                model_loaded = data
                st.success(f"Loaded model (target: {data.get('target','unknown')})")
            except Exception as e:
                st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")

        # Try auto-load if exists and not already loaded
        if model_loaded is None and os.path.exists(model_path):
            try:
                model_loaded = joblib.load(model_path)
            except Exception:
                model_loaded = None

        if model_loaded:
            pipeline = model_loaded.get('pipeline')
            model_target = model_loaded.get('target', 'target_property')
            st.write(f"Model target: **{model_target}**")

            # featurize current molecule and predict
            feat, desc = features.featurize_smiles(compound_info['canonical_smiles'])
            if feat is None:
                st.error('현재 분자의 SMILES로 특징을 추출할 수 없습니다.')
            else:
                feat_aligned = features.align_features_for_pipeline(pipeline, feat)
                X = feat_aligned.reshape(1, -1)
                try:
                    pred = pipeline.predict(X)[0]
                    st.metric(f"Predicted ({model_target})", f"{pred:.4f}")
                    # compare with RDKit if same target
                    if model_target in descriptors:
                        rd_val = descriptors[ 'LogP' ] if model_target == 'LogP' else descriptors.get(model_target)
                        if rd_val is not None:
                            st.write(f"RDKit {model_target}: {rd_val:.4f}")
                            st.write(f"Model - RDKit difference: {pred - rd_val:.4f}")
                except Exception as e:
                    st.error(f"모델 예측 중 오류: {e}")

        # Training UI (kept in advanced expander)
        st.subheader('Train property model')
        with st.expander('Train a new model from CSV'):
            csv_file = st.file_uploader(
                'Upload CSV (columns: name,smiles,target_property) or leave blank to use data/molecule_dataset.csv',
                type=['csv'],
                key='csv_upload_adv',
            )
            target_col = st.text_input('Target column name', value='target_property', key='target_col_adv')
            out_path = st.text_input('Output model path', value='models/property_model.pkl', key='out_path_adv')
            if st.button('Train model', key='train_model_adv'):
                # save uploaded csv to a temp path if provided
                csv_path = 'data/molecule_dataset.csv'
                if csv_file is not None:
                    csv_path = os.path.join('data', 'uploaded_dataset.csv')
                    with open(csv_path, 'wb') as f:
                        f.write(csv_file.getbuffer())

                try:
                    with st.spinner('Training model (this may take a while)...'):
                        trainer.main(csv_path, target_col, output_model=out_path)
                    st.success(f'Training finished. Model saved to {out_path}')
                except Exception as e:
                    st.error(f'Training failed: {e}')

        # Gemini hook point: provide the model prediction and RDKit descriptors to pass into Gemini if user wants
        st.subheader('Gemini hook for ML explanation')
        if st.button('Generate Gemini explanation for ML result', key='gen_gemini_adv'):
            if not genai:
                st.error('google-genai 패키지가 설치되어 있지 않아 Gemini를 사용할 수 없습니다.')
            else:
                if not model_loaded:
                    st.error('모델을 먼저 로드하거나 학습하세요.')
                else:
                    model_target = model_loaded.get('target', 'target_property')
                    pipeline = model_loaded.get('pipeline')
                    feat, desc = features.featurize_smiles(compound_info['canonical_smiles'])
                    if feat is None:
                        st.error('SMILES로부터 특징을 만들 수 없습니다.')
                    else:
                        feat_aligned = features.align_features_for_pipeline(pipeline, feat)
                        pred = pipeline.predict(feat_aligned.reshape(1, -1))[0]
                        prompt = f"Provide a short educational explanation (Korean) of the model prediction.\nProperty: {model_target}\nPredicted value: {pred:.4f}\nRDKit descriptors: {desc}\nNote: This is educational only."
                        try:
                            client = genai.Client(api_key=get_google_api_key())
                            response = client.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'), contents=prompt)
                            st.write(response.text)
                        except Exception as e:
                            st.error(f'Gemini 호출 오류: {e}')

    # --- ML Property Prediction integration ---
    st.subheader("ML Property Prediction (educational)")

    model_path = st.text_input('Model path', value='models/property_model.pkl', key='model_path_main')
    model_loaded = None
    if st.button('Load model', key='load_model_main'):
        try:
            data = joblib.load(model_path)
            model_loaded = data
            st.success(f"Loaded model (target: {data.get('target','unknown')})")
        except Exception as e:
            st.error(f"모델을 불러오는 중 오류가 발생했습니다: {e}")

    # Try auto-load if exists
    if model_loaded is None and os.path.exists(model_path):
        try:
            model_loaded = joblib.load(model_path)
        except Exception:
            model_loaded = None

    if model_loaded:
        pipeline = model_loaded.get('pipeline')
        model_target = model_loaded.get('target', 'target_property')
        st.write(f"Model target: **{model_target}**")

        # featurize current molecule and predict
        feat, desc = features.featurize_smiles(compound_info['canonical_smiles'])
        if feat is None:
            st.error('현재 분자의 SMILES로 특징을 추출할 수 없습니다.')
        else:
            feat_aligned = features.align_features_for_pipeline(pipeline, feat)
            X = feat_aligned.reshape(1, -1)
            try:
                pred = pipeline.predict(X)[0]
                st.metric(f"Predicted ({model_target})", f"{pred:.4f}")
                # compare with RDKit if same target
                if model_target in descriptors:
                    rd_val = descriptors[ 'LogP' ] if model_target == 'LogP' else descriptors.get(model_target)
                    if rd_val is not None:
                        st.write(f"RDKit {model_target}: {rd_val:.4f}")
                        st.write(f"Model - RDKit difference: {pred - rd_val:.4f}")
            except Exception as e:
                st.error(f"모델 예측 중 오류: {e}")

    # Training UI
    st.subheader('Train property model')
    with st.expander('Train a new model from CSV'):
        csv_file = st.file_uploader('Upload CSV (columns: name,smiles,target_property) or leave blank to use data/molecule_dataset.csv', type=['csv'], key='csv_upload_main')
        target_col = st.text_input('Target column name', value='target_property', key='target_col_main')
        out_path = st.text_input('Output model path', value='models/property_model.pkl', key='out_path_main')
        if st.button('Train model', key='train_model_main'):
            # save uploaded csv to a temp path if provided
            csv_path = 'data/molecule_dataset.csv'
            if csv_file is not None:
                csv_path = os.path.join('data', 'uploaded_dataset.csv')
                with open(csv_path, 'wb') as f:
                    f.write(csv_file.getbuffer())

            try:
                with st.spinner('Training model (this may take a while)...'):
                    trainer.main(csv_path, target_col, output_model=out_path)
                st.success(f'Training finished. Model saved to {out_path}')
            except Exception as e:
                st.error(f'Training failed: {e}')

    # Gemini hook point: provide the model prediction and RDKit descriptors to pass into Gemini if user wants
    st.subheader('Gemini hook for ML explanation')
    if st.button('Generate Gemini explanation for ML result', key='gen_gemini_main'):
        if not genai:
            st.error('google-genai 패키지가 설치되어 있지 않아 Gemini를 사용할 수 없습니다.')
        else:
            if not model_loaded:
                st.error('모델을 먼저 로드하거나 학습하세요.')
            else:
                model_target = model_loaded.get('target', 'target_property')
                pipeline = model_loaded.get('pipeline')
                feat, desc = features.featurize_smiles(compound_info['canonical_smiles'])
                if feat is None:
                    st.error('SMILES로부터 특징을 만들 수 없습니다.')
                else:
                    feat_aligned = features.align_features_for_pipeline(pipeline, feat)
                    pred = pipeline.predict(feat_aligned.reshape(1, -1))[0]
                    prompt = f"Provide a short educational explanation (Korean) of the model prediction.\nProperty: {model_target}\nPredicted value: {pred:.4f}\nRDKit descriptors: {desc}\nNote: This is educational only."
                    try:
                        client = genai.Client(api_key=get_google_api_key())
                        response = client.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-2.0-flash'), contents=prompt)
                        st.write(response.text)
                    except Exception as e:
                        st.error(f'Gemini 호출 오류: {e}')


if __name__ == "__main__":
    main()
