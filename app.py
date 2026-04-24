from __future__ import annotations

import html
import os

import pandas as pd
import streamlit as st

from src.collector import collect_lotto_data, load_or_collect
from src.dataset import build_sequence_dataset, latest_sequence
from src.feature_engineering import build_feature_dataframe, build_supervised_tabular, latest_feature
from src.model_registry import DL_MODEL_NAMES, ML_MODEL_NAMES, MODEL_DESCRIPTIONS, MODEL_NAMES
from src.predict import collect_latest_scores
from src.preprocess import validate_raw_data
from src.recommend import build_recommendations
from src.train_dl import train_selected_dl
from src.train_ml import train_selected_ml
from src.utils import ensure_dirs, set_seed
from src.visualization import frequency_chart, gap_chart, metrics_chart, top_score_chart


st.set_page_config(page_title="Lotto 6/45 ML Recommendation Lab", layout="wide")
ensure_dirs()
set_seed(42)

st.title("한국 로또 6/45 머신러닝 추천 실험실")
st.warning(
    "이 앱은 실험용 예측 도구입니다. 로또 당첨을 보장하지 않으며, 추천 결과는 구매 판단의 근거가 될 수 없습니다."
)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def render_ad_slot(location: str = "top") -> None:
    title = _env("AD_TITLE", "광고 문의 / 후원 배너")
    body = _env("AD_TEXT", "이 영역은 제휴 광고, 후원 링크, 서비스 홍보 배너로 사용할 수 있습니다.")
    url = _env("AD_URL", "https://github.com/hyunmoan1994/lotto-ml-lab")
    image_url = _env("AD_IMAGE_URL")
    sponsor = _env("AD_SPONSOR", "Sponsored")

    compact = location == "sidebar"
    image_html = ""
    if image_url:
        image_html = (
            f'<img src="{html.escape(image_url)}" alt="{html.escape(title)}" '
            'style="width:100%;max-height:120px;object-fit:cover;border-radius:6px;margin-bottom:8px;" />'
        )

    padding = "12px" if compact else "16px 18px"
    body_size = "0.86rem" if compact else "0.94rem"
    st.markdown(
        f"""
        <a href="{html.escape(url)}" target="_blank" rel="noopener noreferrer" style="text-decoration:none;">
          <div style="
            border:1px solid #d7e3dc;
            background:#f7fbf8;
            border-radius:8px;
            padding:{padding};
            margin:10px 0 18px 0;
            color:#1f2933;
          ">
            <div style="font-size:0.72rem;color:#667085;margin-bottom:6px;">{html.escape(sponsor)}</div>
            {image_html}
            <div style="font-weight:700;margin-bottom:4px;">{html.escape(title)}</div>
            <div style="font-size:{body_size};line-height:1.45;color:#475467;">{html.escape(body)}</div>
          </div>
        </a>
        """,
        unsafe_allow_html=True,
    )


render_ad_slot("top")

with st.sidebar:
    st.header("설정")
    render_ad_slot("sidebar")
    max_draws = st.number_input("최대 조회 회차", min_value=300, max_value=3000, value=1300, step=50)
    min_draws = st.number_input("최소 수집 회차", min_value=300, max_value=1000, value=300, step=50)
    recent_draws = st.number_input("최근 학습 회차 수", min_value=300, max_value=1200, value=500, step=50)
    seq_len = st.slider("딥러닝 입력 길이", min_value=10, max_value=60, value=20, step=5)
    epochs = st.slider("딥러닝 epoch", min_value=1, max_value=20, value=4)

st.header("0. 알고리즘 선택")
default_models = ["RNN", "LSTM", "Transformer", "RandomForest", "MLP"]
selected_algorithms = st.multiselect(
    "학습하고 추천에 사용할 알고리즘",
    MODEL_NAMES,
    default=default_models,
)

if not selected_algorithms:
    st.info("최소 1개 이상의 알고리즘을 선택하세요.")
    st.stop()

with st.expander("알고리즘 설명", expanded=True):
    for name in selected_algorithms:
        st.markdown(f"**{name}**: {MODEL_DESCRIPTIONS[name]}")

st.subheader("알고리즘별 추천 세트 수")
set_cols = st.columns(min(5, len(selected_algorithms)))
sets_per_model: dict[str, int] = {}
for idx, name in enumerate(selected_algorithms):
    with set_cols[idx % len(set_cols)]:
        sets_per_model[name] = st.number_input(
            f"{name}",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            key=f"sets_{name}",
        )


def _get_raw() -> pd.DataFrame:
    raw = st.session_state.get("raw")
    if raw is None:
        raw = validate_raw_data(load_or_collect(), min_rows=int(min_draws))
        st.session_state.raw = raw
    return raw


def _train_selected(raw: pd.DataFrame, features: pd.DataFrame) -> None:
    x_tab, y_tab = build_supervised_tabular(raw, features)
    x_seq, y_seq = build_sequence_dataset(raw, seq_len=seq_len)
    dl_names = [name for name in selected_algorithms if name in DL_MODEL_NAMES]
    ml_names = [name for name in selected_algorithms if name in ML_MODEL_NAMES]
    dl_results = train_selected_dl(x_seq, y_seq, dl_names, epochs=epochs)
    ml_results = train_selected_ml(x_tab, y_tab, ml_names)
    st.session_state.dl_results = dl_results
    st.session_state.ml_results = ml_results
    st.session_state.trained_algorithms = list(selected_algorithms)
    st.session_state.metrics = pd.DataFrame(
        [{"model": r["name"], **r["valid_metrics"]} for r in dl_results + ml_results]
    )


st.header("1. 데이터 수집")
if st.button("공식 API에서 데이터 수집", type="primary"):
    with st.spinner("동행복권 공식 JSON API에서 최근 회차 데이터를 수집 중입니다."):
        try:
            st.session_state.raw = validate_raw_data(
                collect_lotto_data(
                    max_draws=int(max_draws),
                    min_draws=int(min_draws),
                    recent_draws=int(recent_draws),
                ),
                min_rows=int(min_draws),
            )
            for key in ["features", "dl_results", "ml_results", "recommendations", "scores", "metrics"]:
                st.session_state.pop(key, None)
        except Exception as exc:
            st.error(f"데이터 수집에 실패했습니다: {exc}")
            st.stop()

if "raw" in st.session_state:
    raw = st.session_state.raw
    st.success(f"{len(raw):,}개 회차 수집 완료: {int(raw.draw_no.min())}회 ~ {int(raw.draw_no.max())}회")
    st.dataframe(raw.tail(10), use_container_width=True)
    st.download_button(
        "원본 데이터 CSV 다운로드",
        raw.to_csv(index=False).encode("utf-8-sig"),
        "raw_lotto.csv",
        mime="text/csv",
    )

st.header("2. Feature 생성")
if st.button("feature 생성"):
    raw = _get_raw()
    with st.spinner("최근 빈도, gap, 홀짝비, 번호합, 연속번호, rolling 통계를 생성 중입니다."):
        st.session_state.features = build_feature_dataframe(raw)

if "features" in st.session_state:
    features = st.session_state.features
    x_tab, y_tab = build_supervised_tabular(st.session_state.raw, features)
    x_seq, y_seq = build_sequence_dataset(st.session_state.raw, seq_len=seq_len)
    st.write(f"Feature shape: {features.shape}")
    st.write(f"Tabular 학습 데이터: X={x_tab.shape}, y={y_tab.shape}")
    st.write(f"Sequence 학습 데이터: X={x_seq.shape}, y={y_seq.shape}")
    st.dataframe(features.tail(5), use_container_width=True)
    st.download_button(
        "feature CSV 다운로드",
        features.to_csv(index=False).encode("utf-8-sig"),
        "feature_lotto.csv",
        mime="text/csv",
    )

st.header("3. 모델 학습")
st.caption("선택한 알고리즘만 학습합니다. RNN/LSTM/Transformer는 PyTorch, 나머지는 scikit-learn 기반입니다.")
if st.button("선택한 모델 학습 시작"):
    raw = _get_raw()
    features = st.session_state.get("features")
    if features is None:
        features = build_feature_dataframe(raw)
        st.session_state.features = features
    with st.spinner("선택한 모델을 학습 중입니다. CPU 환경에서는 잠시 걸릴 수 있습니다."):
        _train_selected(raw, features)

if "metrics" in st.session_state:
    st.subheader("검증 지표")
    st.dataframe(st.session_state.metrics, use_container_width=True)

st.header("4. 다음 회차 추천")
if st.button("모델별 추천 생성"):
    raw = _get_raw()
    features = st.session_state.get("features")
    if features is None:
        features = build_feature_dataframe(raw)
        st.session_state.features = features
    trained = st.session_state.get("trained_algorithms", [])
    if sorted(trained) != sorted(selected_algorithms):
        st.info("현재 선택한 알고리즘으로 학습된 모델이 없어 먼저 학습합니다.")
        with st.spinner("선택한 모델을 학습 중입니다."):
            _train_selected(raw, features)
    scores = collect_latest_scores(
        st.session_state.dl_results,
        st.session_state.ml_results,
        latest_feature(features),
        latest_sequence(raw, seq_len=seq_len),
    )
    rec_df, score_df = build_recommendations(scores, sets_per_model=sets_per_model)
    st.session_state.recommendations = rec_df
    st.session_state.scores = score_df

if "recommendations" in st.session_state:
    st.dataframe(st.session_state.recommendations, use_container_width=True)
    render_ad_slot("results")
    st.subheader("모델별 번호 score")
    preview = (
        st.session_state.scores.sort_values(["model", "score"], ascending=[True, False])
        .groupby("model")
        .head(10)
    )
    st.dataframe(preview, use_container_width=True)
    st.download_button(
        "추천 결과 CSV 다운로드",
        st.session_state.recommendations.to_csv(index=False).encode("utf-8-sig"),
        "recommendations.csv",
        mime="text/csv",
    )
    st.download_button(
        "모델별 score CSV 다운로드",
        st.session_state.scores.to_csv(index=False).encode("utf-8-sig"),
        "model_scores.csv",
        mime="text/csv",
    )

st.header("5. 차트")
if "raw" in st.session_state:
    st.pyplot(frequency_chart(st.session_state.raw))
if "features" in st.session_state:
    st.pyplot(gap_chart(st.session_state.features))
if "scores" in st.session_state:
    selected = st.selectbox("Top-10 score 모델", sorted(st.session_state.scores["model"].unique()))
    st.pyplot(top_score_chart(st.session_state.scores, selected))
if "metrics" in st.session_state:
    st.pyplot(metrics_chart(st.session_state.metrics))
