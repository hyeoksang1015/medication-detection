import base64
import gc
import os
import re
import unicodedata

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="헬스잇", page_icon="💊", layout="wide")

# 학습 시 사용한 전처리(패딩 후 리사이즈)와 동일하게 맞춰야 결과가 정확함
PAD_SIZE = 1280
OUT_SIZE = 640

# 브랜드 컬러 (초록 계열 = 안전/건강 톤)
BRAND = "#0F6E56"
BRAND_LIGHT_BG = "#E1F5EE"
BRAND_BORDER = "#1D9E75"

st.markdown(
    f"""
    <style>
    .stApp {{
        max-width: 100%;
        padding: 0.35rem 0.35rem 2rem;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }}
    [data-testid="stHeader"] {{
        background: rgba(255,255,255,0);
    }}
    [data-testid="stFileUploader"] > section {{
        padding: 0.6rem;
        border-radius: 0.8rem;
        border: 1.5px dashed #B4B2A9 !important;
    }}
    [data-testid="stBaseButton"] button {{
        min-height: 44px;
        border-radius: 0.8rem;
    }}
    [data-testid="stBaseButton"][kind="primary"] button {{
        background-color: {BRAND} !important;
        border-color: {BRAND} !important;
    }}
    [data-testid="stExpander"] {{
        border: 0.5px solid #D3D1C7 !important;
        border-radius: 0.8rem !important;
        margin-bottom: 0.5rem;
    }}
    [data-testid="stAlert"] {{
        border-radius: 0.8rem;
    }}
    /* 스플래시 화면 */
    .splash-wrap {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 60vh;
        gap: 6px;
    }}
    .splash-badge {{
        width: 88px;
        height: 88px;
        border-radius: 50%;
        background: {BRAND_LIGHT_BG};
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 44px;
    }}
    .splash-title {{
        font-size: 36px;
        font-weight: 700;
        color: {BRAND};
        margin-top: 8px;
    }}
    .splash-sub {{
        font-size: 14px;
        color: #666;
    }}
    /* 스피너 (인식 진행 화면) */
    .spinner {{
        width: 56px;
        height: 56px;
        border: 6px solid #E0E0E0;
        border-top: 6px solid {BRAND};
        border-radius: 50%;
        animation: spin 0.9s linear infinite;
        margin: 0 auto;
    }}
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}
    /* 결과 화면 뱃지 / 카드 */
    .result-badge {{
        display: inline-block;
        background: {BRAND_LIGHT_BG};
        color: {BRAND};
        font-weight: 600;
        font-size: 14px;
        padding: 6px 14px;
        border-radius: 999px;
        margin: 6px 0 4px;
    }}
    .drug-card {{
        border: 0.5px solid #D3D1C7;
        border-left: 4px solid {BRAND_BORDER};
        border-radius: 0.8rem;
        padding: 10px 14px;
        margin-top: -4px;
    }}
    .drug-card p {{ margin: 2px 0; }}
    .drug-name {{ font-size: 16px; font-weight: 600; }}
    .drug-sub {{ font-size: 13px; color: #666; }}
    @media (max-width: 768px) {{
        .stApp {{ padding: 0.2rem 0.2rem 1.2rem; }}
        .block-container {{
            padding-top: 0.6rem !important;
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }}
        h1 {{ font-size: 1.65rem !important; line-height: 1.2 !important; }}
        h2 {{ font-size: 1.25rem !important; }}
        .stImage img {{ border-radius: 0.75rem; }}
        [data-testid="stExpander"] {{ margin-bottom: 0.45rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# 세션 상태 초기화 / 화면 전환 (KeyError 방지를 위해 초기화 항목 보강)
# --------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "stage": "splash",  # splash -> upload -> processing -> result
        "raw_image": None,
        "detections": None,
        "output_rgb": None,
        "model_path": "best.pt",
        "drug_info_path": "drug_full_info.csv",
        "fallback_info_path": "drug_info.csv",
        "conf_threshold": 0.25,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def go_to(stage: str) -> None:
    st.session_state["stage"] = stage
    st.rerun()


def reset_flow() -> None:
    st.session_state["raw_image"] = None
    st.session_state["detections"] = None
    st.session_state["output_rgb"] = None
    go_to("upload")


# --------------------------------------------------------------------------
# 모델 / 매핑 데이터 로드 (CSV 로직 반영)
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path: str):
    return YOLO(path)


def _clean_name(value) -> str:
    """이름 비교용 정규화: 유니코드 정규화(NFC) + 모든 공백 제거.
    사람 눈에는 같아 보여도 한글 자모 분리형(NFD)으로 저장된 문자열은
    NFC로 통일하지 않으면 완전 일치 비교가 실패하므로 이 처리가 중요합니다."""
    text = unicodedata.normalize("NFC", str(value))
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _read_csv_robust(path: str):
    """여러 인코딩(utf-8-sig, utf-8, cp949, euc-kr)을 순서대로 시도해서 읽습니다.
    윈도우에서 엑셀로 저장한 CSV는 cp949/euc-kr인 경우가 많습니다."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError) as e:
            last_err = e
            continue
    raise last_err if last_err else RuntimeError("CSV를 읽을 수 없습니다.")


@st.cache_data
def load_drug_csv(path: str, name_col_candidates=("dl_name", "name")):
    """의약품 매핑 CSV를 로드합니다. dl_name 또는 name 컬럼을 자동 인식하고,
    정규화된 이름 비교용 clean_name 컬럼을 추가합니다."""
    if not path or not os.path.exists(path):
        return None
    try:
        df = _read_csv_robust(path).fillna("-")
    except Exception as e:
        st.error(f"⚠️ CSV 파싱 중 오류 발생 (`{path}`): {e}")
        return None

    name_col = next((c for c in name_col_candidates if c in df.columns), None)
    if name_col is None:
        st.error(
            f"⚠️ CSV 파일(`{path}`)에 이름 컬럼"
            f"({'/'.join(name_col_candidates)})이 존재하지 않습니다. "
            f"현재 컬럼: {list(df.columns)}"
        )
        return None

    df.attrs["name_col"] = name_col
    df["clean_name"] = df[name_col].apply(_clean_name)
    return df


def match_drug_row(drug_df, drug_name: str, cls_idx: int = None):
    """YOLO 예측 클래스명을 CSV의 이름 컬럼과 매칭합니다.
    1) 정규화(NFC+공백제거) 후 완전 일치
    2) 완전 일치가 없으면 정규화된 부분 포함 매칭으로 재시도
    3) 그래도 없으면 idx(클래스 인덱스) 컬럼으로 매칭 시도
    """
    if drug_df is None:
        return None

    clean_target = _clean_name(drug_name)

    exact = drug_df[drug_df["clean_name"] == clean_target]
    if not exact.empty:
        return exact.iloc[0]

    partial = drug_df[
        drug_df["clean_name"].str.contains(clean_target, na=False, regex=False)
    ]
    if not partial.empty:
        return partial.iloc[0]

    if cls_idx is not None and "idx" in drug_df.columns:
        idx_match = drug_df[drug_df["idx"].astype(str) == str(cls_idx)]
        if not idx_match.empty:
            return idx_match.iloc[0]

    return None


def get_field(row, *candidates, default="정보 없음"):
    """row(Series)에서 후보 컬럼명들을 순서대로 확인해 처음 존재하고
    비어있지 않은 값을 반환합니다."""
    if row is None:
        return default
    for col in candidates:
        if col in row.index:
            val = row.get(col)
            if val not in (None, "", "-", "nan"):
                return val
    return default


def _truthy(value) -> bool:
    """CSV 값(True/False, 'True'/'False', 1/0 등)을 불리언으로 정규화합니다."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("true", "1", "y", "yes", "해당")


# --------------------------------------------------------------------------
# 탐지 로직
# --------------------------------------------------------------------------
def run_detection(raw_image: Image.Image, model, conf_threshold: float):
    orig = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
    oh, ow = orig.shape[:2]

    pad_w = max(PAD_SIZE - ow, 0)
    pad_h = max(PAD_SIZE - oh, 0)
    left = pad_w // 2
    top = pad_h // 2

    padded = cv2.copyMakeBorder(
        orig,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    resized = cv2.resize(padded, (OUT_SIZE, OUT_SIZE))

    results = model.predict(
        resized, imgsz=OUT_SIZE, conf=conf_threshold, verbose=False
    )[0]

    scale = PAD_SIZE / OUT_SIZE
    output_img = orig.copy()
    detections = []

    for box in results.boxes:
        cls_idx = int(box.cls[0])
        conf = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 = x1 * scale - left
        y1 = y1 * scale - top
        x2 = x2 * scale - left
        y2 = y2 * scale - top

        x1 = max(0, min(x1, ow))
        y1 = max(0, min(y1, oh))
        x2 = max(0, min(x2, ow))
        y2 = max(0, min(y2, oh))

        if x2 - x1 <= 0 or y2 - y1 <= 0:
            continue

        drug_name = model.names.get(cls_idx, f"알 수 없는 클래스 ({cls_idx})")
        det_id = len(detections) + 1

        cv2.rectangle(
            output_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4
        )
        cv2.putText(
            output_img,
            str(det_id),
            (int(x1), max(int(y1) - 10, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        detections.append(
            {"id": det_id, "cls_idx": cls_idx, "drug_name": drug_name, "conf": conf}
        )

    del padded, resized, results
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
    return detections, output_rgb


# --------------------------------------------------------------------------
# 1. 스플래시 화면
# --------------------------------------------------------------------------
def render_splash() -> None:
    st.markdown(
        """
        <div class="splash-wrap">
            <div class="splash-badge">💊</div>
            <div class="splash-title">헬스잇</div>
            <div class="splash-sub">HealthEat · AI 알약 탐지 서비스</div>
            <div class="spinner"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    _, col, _ = st.columns([1, 1, 1])
    with col:
        if st.button("시작하기", use_container_width=True):
            go_to("upload")


# --------------------------------------------------------------------------
# 2. 사진 입력 화면
# --------------------------------------------------------------------------
def render_upload() -> None:
    with st.sidebar:
        st.header("⚙️ 설정")
        # st.text_input의 value와 key 값을 세션 기본값과 연동하여 안전하게 관리합니다.
        st.text_input(
            "모델 가중치 경로 (.pt)",
            value=st.session_state["model_path"],
            key="model_path",
        )
        st.text_input(
            "의약품 상세 CSV 경로 (drug_full_info.csv)",
            value=st.session_state["drug_info_path"],
            key="drug_info_path",
            help="idx/dl_name/제조사/효능군명/금기 정보 등이 담긴 상세 CSV 경로",
        )
        st.text_input(
            "보조 매핑 CSV 경로 (drug_info.csv, 선택)",
            value=st.session_state["fallback_info_path"],
            key="fallback_info_path",
            help="상세 CSV에서 매칭이 안 될 때 사용할 name/company/effect 형태의 보조 CSV 경로",
        )
        st.slider(
            "신뢰도(confidence) 임계값",
            0.05,
            0.95,
            st.session_state["conf_threshold"],
            0.05,
            key="conf_threshold",
        )

        with st.expander("🔧 디버그: 경로/캐시 정보", expanded=False):
            st.caption(f"현재 작업 디렉터리: `{os.getcwd()}`")
            for label, p in [
                ("상세 CSV", st.session_state["drug_info_path"]),
                ("보조 CSV", st.session_state["fallback_info_path"]),
                ("모델(.pt)", st.session_state["model_path"]),
            ]:
                abs_path = os.path.abspath(p) if p else "-"
                exists = os.path.exists(p) if p else False
                st.write(
                    f"- **{label}**: `{p}`\n"
                    f"  - 절대경로: `{abs_path}`\n"
                    f"  - 존재 여부: {'✅ 있음' if exists else '❌ 없음'}"
                )
            if st.button("🗑️ 캐시 초기화 (CSV/모델 다시 로드)"):
                st.cache_data.clear()
                st.cache_resource.clear()
                st.success("캐시를 비웠습니다. 다시 인식해 보세요.")

    st.markdown("### 📷 알약 사진을 등록해주세요")
    st.info("사진 찍기 또는 업로드 후, 미리보기 영역에서 이미지를 확인할 수 있습니다.")

    tab_camera, tab_upload = st.tabs(["사진 찍기", "사진 업로드"])
    image_file = None
    with tab_camera:
        camera_file = st.camera_input("카메라로 촬영", label_visibility="collapsed")
        if camera_file is not None:
            image_file = camera_file
    with tab_upload:
        uploaded_file = st.file_uploader(
            "탐지할 알약 사진을 업로드해 주세요.",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        if uploaded_file is not None:
            image_file = uploaded_file

    raw_image = None
    if image_file is not None:
        image_file.seek(0)
        raw_image = Image.open(image_file).convert("RGB")

    def pil_image_to_base64(img: Image.Image) -> str:
        buffered = cv2.imencode(".png", cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))[
            1
        ].tobytes()
        return base64.b64encode(buffered).decode("utf-8")

    if raw_image is not None:
        image_data = pil_image_to_base64(raw_image)
        st.markdown(
            f'<div class="preview-box"><img src="data:image/png;base64,{image_data}" alt="preview"></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="preview-box">
                <div class="preview-placeholder">
                    <strong>알약 사진을 등록해주세요</strong>
                    <span>사진을 선택하면 미리보기가 이 영역에 표시됩니다.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    model_path = st.session_state["model_path"]
    if not os.path.exists(model_path):
        st.error(f"🚨 모델 파일을 찾을 수 없습니다: {model_path}")

    st.write("")
    cols = st.columns([1, 1], gap="large")
    with cols[0]:
        if st.button("← 처음으로", use_container_width=True):
            go_to("splash")
    with cols[1]:
        if st.button(
            "인식 시작 →",
            type="primary",
            use_container_width=True,
            disabled=raw_image is None or not os.path.exists(model_path),
        ):
            st.session_state["raw_image"] = raw_image
            go_to("processing")


# --------------------------------------------------------------------------
# 3. 인식 진행 화면
# --------------------------------------------------------------------------
def render_processing() -> None:
    st.markdown(
        """
        <div class="splash-wrap" style="height:55vh;">
            <div class="spinner"></div>
            <div style="margin-top:18px; font-size:18px; color:#444;">
                YOLO 모델로 알약을 인식하고 있어요...
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = st.session_state.get("model_path", "best.pt")
    conf_threshold = st.session_state.get("conf_threshold", 0.25)

    try:
        model = load_model(model_path)
    except Exception as e:  # noqa: BLE001
        st.error(f"🚨 모델을 불러오지 못했습니다.\n\n{e}")
        if st.button("← 사진 입력으로 돌아가기"):
            go_to("upload")
        return

    detections, output_rgb = run_detection(
        st.session_state["raw_image"], model, conf_threshold
    )
    st.session_state["detections"] = detections
    st.session_state["output_rgb"] = output_rgb
    go_to("result")


# --------------------------------------------------------------------------
# 4. 결과 출력 화면 (KeyError 방지 .get() 안전 장치 적용 및 CSV 연동)
# --------------------------------------------------------------------------
def render_result() -> None:
    detections = st.session_state.get("detections")
    output_rgb = st.session_state.get("output_rgb")

    # 세션 상태에서 안전하게 값을 가져오고 없으면 기본값 사용
    drug_info_path = st.session_state.get("drug_info_path", "drug_full_info.csv")
    fallback_info_path = st.session_state.get("fallback_info_path", "drug_info.csv")
    drug_df = load_drug_csv(drug_info_path, name_col_candidates=("dl_name", "name"))
    fallback_df = load_drug_csv(
        fallback_info_path, name_col_candidates=("name", "dl_name")
    )

    st.markdown("### ✅ AI 탐지 결과")

    if output_rgb is not None:
        st.image(output_rgb, caption="AI 탐지 결과", width="stretch")

    if not detections:
        st.warning(
            "탐지된 알약이 없습니다. 신뢰도 임계값을 낮추거나 다른 사진으로 시도해 주세요."
        )
    else:
        st.markdown(
            f'<span class="result-badge">🎉 총 {len(detections)}개의 알약을 '
            f"탐지했습니다</span>",
            unsafe_allow_html=True,
        )

        if drug_df is None and fallback_df is None:
            st.info(
                f"ℹ️ 의약품 매핑 CSV 파일을 찾지 못했습니다 "
                f"(`{drug_info_path}`, `{fallback_info_path}`). "
                "탐지 및 클래스 표시는 정상 동작하며, 제조사 등의 상세정보는 표시되지 않습니다."
            )

        st.markdown("---")
        st.markdown("#### 📋 탐지 상세 리포트")
        st.caption(
            "AI 모델의 예측 결과입니다. 신뢰도(confidence)가 낮을수록 오탐 가능성이 높습니다."
        )

        for det in detections:
            pred_name = det["drug_name"]

            # 1차: 상세 CSV(drug_full_info.csv)에서 매칭 시도
            matched_row = match_drug_row(drug_df, pred_name, det["cls_idx"])
            used_fallback = False

            # 2차: 상세 CSV에서 실패하면 보조 CSV(drug_info.csv)로 폴백
            if matched_row is None:
                matched_row = match_drug_row(fallback_df, pred_name, det["cls_idx"])
                used_fallback = matched_row is not None

            display_name = get_field(matched_row, "dl_name", "name", default=pred_name)
            company = get_field(matched_row, "제조사", "company")
            category = get_field(matched_row, "효능군명")
            drug_type = get_field(matched_row, "전문일반")
            effect = get_field(matched_row, "effect", "효능군명")

            st.markdown(
                f"""
                <div class="drug-card">
                    <p class="drug-name">💊 {display_name}</p>
                    <p class="drug-sub">🏢 제조사: {company} | 분류: {drug_type}</p>
                    <p class="drug-sub">🎯 효능: {effect}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if matched_row is None:
                st.caption(
                    f"💡 `{drug_info_path}` / `{fallback_info_path}` CSV 파일에 "
                    f"이 알약 이름(`{pred_name}`)과 일치하는 매핑 정보가 없습니다."
                )
                continue

            if used_fallback:
                st.caption(
                    f"🔎 보조 매핑 CSV(`{fallback_info_path}`)에서 가져온 정보입니다. "
                    "금기·주의사항 등 상세 정보는 없습니다."
                )
                continue

            # 임부 / 연령 / 병용 금기 정보 (상세 CSV에만 존재하는 컬럼)
            if "임부금기_해당" in matched_row.index:
                with st.expander("🚨 임부 / 연령 / 병용 금기 정보", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(
                            label="임부 금기",
                            value="해당"
                            if _truthy(matched_row.get("임부금기_해당"))
                            else "없음",
                        )
                    with col2:
                        st.metric(
                            label="연령 금기",
                            value="해당"
                            if _truthy(matched_row.get("연령금기_해당"))
                            else "없음",
                        )
                    with col3:
                        st.metric(
                            label="병용 금기",
                            value="해당"
                            if _truthy(matched_row.get("병용금기_해당"))
                            else "없음",
                        )

                    interaction_drug = matched_row.get("병용금기_상대약", "-")
                    if interaction_drug not in ("-", "정보 없음", None, ""):
                        st.markdown(
                            f"**⚠️ 병용금기 상대약**\n\n:red[{interaction_drug}]"
                        )

                with st.expander("👵 노인 주의 정보", expanded=False):
                    if _truthy(matched_row.get("노인주의_해당")):
                        st.warning(
                            "⚠️ 이 약품은 노인 복용 시 주의가 필요합니다. 의사/약사와 상담하세요."
                        )
                    else:
                        st.success("✅ 노인 주의 특이사항이 없습니다.")

    st.markdown("---")
    st.warning(
        "⚠️ **이 결과는 AI 모델의 예측값이며 100% 정확하지 않을 수 있습니다.** "
        "실제 복용 중인 약을 확인하려면 반드시 약사·의사와 상담하거나 "
        "식품의약품안전처 '의약품안전나라(nedrug.mfds.go.kr)'에서 다시 확인해 주세요."
    )

    st.write("")
    if st.button("다시 인식하기", use_container_width=True):
        reset_flow()


# --------------------------------------------------------------------------
# 메인 라우팅
# --------------------------------------------------------------------------
def main() -> None:
    init_session_state()
    stage_router = {
        "splash": render_splash,
        "upload": render_upload,
        "processing": render_processing,
        "result": render_result,
    }
    stage_router[st.session_state["stage"]]()


if __name__ == "__main__":
    main()
