from __future__ import annotations


DL_MODEL_NAMES = ["RNN", "LSTM", "Transformer"]
ML_MODEL_NAMES = [
    "RandomForest",
    "MLP",
    "ExtraTrees",
    "GradientBoosting",
    "AdaBoost",
    "LogisticRegression",
    "KNN",
]
MODEL_NAMES = DL_MODEL_NAMES + ML_MODEL_NAMES

MODEL_DESCRIPTIONS = {
    "RNN": "최근 회차들을 순서가 있는 시계열로 보고, 단순 순환 구조로 다음 회차 번호 score를 계산합니다.",
    "LSTM": "최근 회차들을 시계열로 분석하며, 장단기 패턴을 더 오래 기억하도록 설계된 모델입니다.",
    "Transformer": "최근 회차 사이의 관계를 attention 구조로 비교해 다음 회차 번호별 score를 만듭니다.",
    "RandomForest": "여러 개의 의사결정나무를 조합해 빈도, gap, rolling 통계 feature에서 번호별 score를 계산합니다.",
    "MLP": "정규화된 통계 feature를 신경망에 넣어 번호 1~45의 멀티라벨 score를 예측합니다.",
    "ExtraTrees": "무작위성이 더 큰 여러 결정나무를 사용해 과거 feature의 다양한 분기 패턴을 탐색합니다.",
    "GradientBoosting": "약한 예측기를 순차적으로 보완하면서 번호별 출현 가능 score를 계산합니다.",
    "AdaBoost": "틀린 예측에 더 큰 가중치를 주며 여러 약한 분류기를 조합하는 방식입니다.",
    "LogisticRegression": "각 번호를 독립적인 이진 분류 문제로 보고 선형 확률 score를 제공합니다.",
    "KNN": "최근 feature 패턴과 비슷했던 과거 회차들을 찾아 번호별 score를 계산합니다.",
}

