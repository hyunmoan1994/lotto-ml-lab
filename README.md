# 한국 로또 6/45 머신러닝 추천 앱

Streamlit, PyTorch, scikit-learn으로 만든 한국 로또 6/45 추천 실험 앱입니다.

> 이 프로젝트는 **실험용 예측**입니다. 로또는 무작위성이 강한 추첨이며, 이 앱은 당첨을 보장하지 않습니다. 추천 결과는 모델 점수 기반의 실험 결과로만 해석해야 합니다.

## 주요 기능

- 동행복권 공식 API 또는 보조 JSON 데이터 소스에서 최근 300회 이상 자동 수집
- 다음 회차 번호를 45개 번호에 대한 멀티라벨 분류 문제로 구성
- feature 생성:
  - 최근 5/10/30/50/100회 번호별 빈도
  - 번호별 gap
  - 홀짝비, 번호합, 평균, 표준편차
  - 낮은 번호/높은 번호 비율
  - 연속번호 개수
  - rolling count, trend, EWMA 통계
- 알고리즘 드롭다운 선택
- 알고리즘별 설명 표시
- 알고리즘별 추천 세트 수 설정
- matplotlib 차트 제공
- CSV 파일은 로컬에 생성하지 않고, Streamlit 다운로드 버튼으로만 제공

## 지원 알고리즘

- RNN: 최근 회차들을 순서가 있는 시계열로 보고 다음 회차 번호 score를 계산합니다.
- LSTM: 장단기 패턴을 더 오래 기억하도록 설계된 순환 신경망입니다.
- Transformer: attention 구조로 최근 회차 사이의 관계를 비교합니다.
- RandomForest: 여러 결정나무를 조합해 통계 feature에서 score를 계산합니다.
- MLP: 정규화된 통계 feature를 신경망에 넣어 멀티라벨 score를 예측합니다.
- ExtraTrees: 무작위성이 더 큰 결정나무 앙상블입니다.
- GradientBoosting: 약한 예측기를 순차적으로 보완하는 boosting 모델입니다.
- AdaBoost: 틀린 예측에 더 큰 가중치를 주는 boosting 모델입니다.
- LogisticRegression: 각 번호를 독립적인 이진 분류 문제로 보는 선형 모델입니다.
- KNN: 비슷한 과거 feature 패턴을 가진 회차를 참고합니다.

## 설치

```powershell
cd D:\job\lotto_ml_lab
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

가장 쉬운 방법:

```powershell
.\run_app.bat
```

직접 실행:

```powershell
streamlit run app.py
```

CLI 전체 파이프라인 실행:

```powershell
python main.py --epochs 4 --seq-len 20
```

## Railway 배포

이 저장소는 Railway 배포용 파일을 포함합니다.

- `requirements.txt`
- `Procfile`
- `railway.json`
- `nixpacks.toml`
- `.python-version`

Railway에서 GitHub 저장소를 연결하면 다음 명령으로 실행됩니다.

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port ${PORT:-8501} --server.headless true
```

## 광고/수익화 설정

앱에는 상단, 사이드바, 추천 결과 아래에 광고 배너 슬롯이 있습니다.
Railway의 Variables 메뉴에서 아래 환경변수를 설정하면 코드 수정 없이 광고 내용을 바꿀 수 있습니다.

- `AD_TITLE`: 광고 제목
- `AD_TEXT`: 광고 설명 문구
- `AD_URL`: 클릭 시 이동할 링크
- `AD_IMAGE_URL`: 배너 이미지 URL
- `AD_SPONSOR`: 작은 광고 라벨

예시:

```text
AD_TITLE=오늘의 추천 서비스
AD_TEXT=로또 분석과 함께 볼 만한 제휴 서비스입니다.
AD_URL=https://example.com
AD_IMAGE_URL=https://example.com/banner.png
AD_SPONSOR=Advertisement
```

Google AdSense 같은 스크립트형 광고는 별도 사이트 승인, 도메인 검증, 광고 정책 준수가 필요합니다. 이 앱의 기본 광고 슬롯은 제휴 링크, 후원 링크, 직접 판매 배너부터 시작하기 좋게 구성했습니다.

## 파일 구조

```text
lotto_ml_lab/
  app.py
  main.py
  run_app.bat
  requirements.txt
  README.md
  Procfile
  railway.json
  nixpacks.toml
  src/
    collector.py
    dataset.py
    evaluate.py
    feature_engineering.py
    model_registry.py
    predict.py
    preprocess.py
    recommend.py
    train_dl.py
    train_ml.py
    utils.py
    visualization.py
  tests/
    test_pipeline.py
```

## 주의

- 로컬 CSV 문서는 생성하지 않습니다.
- 학습된 모델 파일은 `models/saved/` 아래에 생성될 수 있지만 Git에는 올리지 않습니다.
- CPU 환경에서는 PyTorch와 boosting 모델 학습에 시간이 걸릴 수 있습니다.
- 성능 지표는 과거 데이터 검증용이며 미래 당첨 확률을 의미하지 않습니다.
