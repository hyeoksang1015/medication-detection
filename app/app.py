import gc
import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO

# ------------------------------------------------------------------
# 페이지 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="헬스잇 - AI 알약 탐지", page_icon="💊", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        max-width: 100%;
        padding: 0.35rem 0.35rem 2rem;
    }
    .block-container {
        padding-top: 1rem !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }
    [data-testid="stFileUploader"] > section {
        padding: 0.6rem;
        border-radius: 0.8rem;
    }
    [data-testid="stBaseButton"] button {
        min-height: 44px;
        border-radius: 0.8rem;
    }
    @media (max-width: 768px) {
        .stApp {
            padding: 0.2rem 0.2rem 1.2rem;
        }
        .block-container {
            padding-top: 0.6rem !important;
            padding-left: 0.25rem !important;
            padding-right: 0.25rem !important;
        }
        h1 {
            font-size: 1.65rem !important;
            line-height: 1.2 !important;
        }
        h2 {
            font-size: 1.25rem !important;
        }
        .stImage img {
            border-radius: 0.75rem;
        }
        [data-testid="stExpander"] {
            margin-bottom: 0.45rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💊 헬스잇 (Health Eat)")
st.caption("YOLOv8 기반 알약 객체 탐지 · AI-Hub 경구약제 데이터셋으로 학습된 모델 사용")
st.info("📱 모바일에서도 읽기 쉽게, 버튼과 여백을 조정해 두었습니다.")

st.warning(
    "⚠️ **이 결과는 AI 모델의 예측값이며 100% 정확하지 않을 수 있습니다.**\n\n"
    "조명, 각도, 학습 데이터에 없던 약 등에 따라 잘못 탐지/분류될 수 있습니다. "
    "실제로 복용 중인 약을 확인하려면 반드시 약사·의사와 상담하거나 "
    "식품의약품안전처 '의약품안전나라(nedrug.mfds.go.kr)'에서 다시 확인해 주세요."
)
st.markdown("---")

# ------------------------------------------------------------------
# 설정값 (사이드바에서 경로 변경 가능)
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")
    model_path = st.text_input("모델 가중치 경로 (.pt)", value="best.pt")
    drug_info_path = st.text_input(
        "의약품 정보 매핑 CSV 경로",
        value="drug_info.csv",
        help="컬럼: category_id, name, company, effect",
    )
    conf_threshold = st.slider("신뢰도(confidence) 임계값", 0.05, 0.95, 0.25, 0.05)

# 학습 시 사용한 것과 동일한 class index -> category_id(품목 코드) 매핑
# (Colab 학습 스크립트의 sorted_cat_ids 그대로)
SORTED_CAT_IDS = sorted(
    [
        1900,
        2483,
        3351,
        3483,
        3544,
        4543,
        12081,
        12247,
        12778,
        13395,
        13900,
        16232,
        16262,
        16548,
        16551,
        16688,
        18147,
        18357,
        19232,
        19552,
        19607,
        19861,
        20014,
        20238,
        20877,
        21325,
        21771,
        22074,
        22347,
        22362,
        24850,
        25367,
        25438,
        25469,
        27733,
        27777,
        27926,
        27993,
        28763,
        29345,
        29451,
        29667,
        30308,
        31863,
        31885,
        32310,
        33009,
        33208,
        33880,
        34597,
        35206,
        36637,
        38162,
        41768,
        3832,
        3743,
    ]
)
IDX_TO_CAT_ID = {idx: cat_id for idx, cat_id in enumerate(SORTED_CAT_IDS)}

# 학습 시 사용한 전처리(패딩 후 리사이즈)와 동일하게 맞춰야 결과가 정확함
PAD_SIZE = 1280
OUT_SIZE = 640


@st.cache_resource
def load_model(path: str):
    return YOLO(path)


@st.cache_data
def load_drug_info(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["category_id"] = df["category_id"].astype(int)
    return df.set_index("category_id").to_dict(orient="index")


# ------------------------------------------------------------------
# 모델 / 매핑 데이터 로드
# ------------------------------------------------------------------
model = None
model_load_error = None
if os.path.exists(model_path):
    try:
        model = load_model(model_path)
    except Exception as e:  # noqa: BLE001
        model_load_error = str(e)
else:
    model_load_error = f"파일을 찾을 수 없습니다: {model_path}"

drug_info = load_drug_info(drug_info_path)

if model_load_error:
    st.error(f"🚨 모델을 불러오지 못했습니다.\n\n{model_load_error}")

if drug_info is None:
    st.info(
        f"ℹ️ 의약품 상세정보 매핑 파일을 찾지 못했습니다 (`{drug_info_path}`). "
        "탐지는 정상 동작하지만 약 이름/제조사/효능 대신 품목 코드만 표시됩니다."
    )

# ------------------------------------------------------------------
# 업로드 & 탐지
# ------------------------------------------------------------------
col_left, col_right = st.columns([1, 1])

raw_image = None
with col_left:
    st.subheader("📸 알약 사진 업로드")
    uploaded_file = st.file_uploader(
        "탐지할 알약 사진을 업로드해 주세요.", type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        uploaded_file.seek(0)
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="원본 이미지", use_container_width=True)
    else:
        st.info("👆 사진을 업로드하면 AI 탐지가 시작됩니다.")

detections = []

with col_right:
    st.subheader("🔍 AI 탐지 결과")

    if uploaded_file is not None and model is not None:
        with st.spinner("YOLO 모델로 알약을 탐지하는 중..."):
            orig = cv2.cvtColor(np.array(raw_image), cv2.COLOR_RGB2BGR)
            oh, ow = orig.shape[:2]

            # 학습 때와 동일한 letterbox 패딩 후 리사이즈
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

            for box in results.boxes:
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # 패딩/리사이즈 전 원본 좌표계로 역변환
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

                category_id = IDX_TO_CAT_ID.get(cls_idx)
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
                    {
                        "id": det_id,
                        "cls_idx": cls_idx,
                        "category_id": category_id,
                        "conf": conf,
                    }
                )

            del padded, resized, results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if detections:
            result_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            st.image(result_rgb, caption="AI 탐지 결과", use_container_width=True)
            st.success(f"🎉 총 {len(detections)}개의 알약을 탐지했습니다.")
        else:
            st.warning(
                "탐지된 알약이 없습니다. 신뢰도 임계값을 낮추거나 다른 사진으로 시도해 주세요."
            )

    elif uploaded_file is not None and model is None:
        st.error(
            "모델이 로드되지 않아 탐지를 진행할 수 없습니다. 사이드바에서 경로를 확인해 주세요."
        )
    else:
        st.warning("⚠️ 왼쪽에서 사진을 먼저 업로드해 주세요.")

# ------------------------------------------------------------------
# 상세 리포트
# ------------------------------------------------------------------
if detections:
    st.markdown("---")
    st.markdown("## 📋 탐지 상세 리포트")
    st.caption(
        "AI 모델의 예측 결과입니다. 신뢰도(confidence)가 낮을수록 오탐 가능성이 높습니다."
    )

    for det in detections:
        info = drug_info.get(det["category_id"]) if drug_info else None
        title = f"[알약 {det['id']}] 신뢰도 {det['conf'] * 100:.1f}%"
        with st.expander(title, expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**분류 클래스 인덱스:** {det['cls_idx']}")
                st.markdown(f"**품목 코드(category_id):** {det['category_id']}")
            with col2:
                if info:
                    st.markdown(f"**💊 의약품명:** {info.get('name', '-')}")
                    st.markdown(f"**🏢 제조사:** {info.get('company', '-')}")
                    st.markdown(f"**🎯 효능:** {info.get('effect', '-')}")
                else:
                    st.markdown(
                        "**상세 정보:** 매핑 데이터 없음 "
                        f"(`{drug_info_path}`에 category_id `{det['category_id']}` 행 추가 필요)"
                    )

    st.markdown("---")
    st.info(
        "💡 AI 탐지 결과는 참고용입니다. 실제 복용 여부는 반드시 약사·의사와 상담하거나 "
        "식품의약품안전처 의약품안전나라(nedrug.mfds.go.kr)에서 다시 확인해 주세요."
    )
