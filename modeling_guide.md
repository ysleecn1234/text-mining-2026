# 모델링 작업 가이드

## 현재 상태 (영민이가 완료한 부분)

영민이가 다음을 마쳤다.

### TF-IDF 트랙 전처리

- 작업 파일: `tfidf_track.ipynb`
- 저장 파일: `data/processed/work_pool_tfidf_tokens.parquet`
- 완료된 단계:
  1. parquet 3개 로딩 및 병합 (291,466건)
  2. 반복 구두점 정리 (!!!→!, ???→?, ...은 유지)
  3. Komoran 형태소 분석 (NNG/NNP/VV/VA 추출, SN→`<NUM>`)
  4. 불용어 제거 (1글자, `<NUM>`, 순수숫자, 불용어 목록)
  5. TfidfVectorizer(max_features=10000, min_df=5, max_df=0.95) 벡터화
  6. Class-level TF-IDF 기반 유형별 키워드 분석 완료
- 저장된 컬럼: newsID, binary_label, type_label, source_class, title_clean, title_morphs, title_tokens

### TF-IDF 키워드 분석 관련 참고사항

**키워드 분석 방법 변경 이력**

처음에는 **평균 TF-IDF 방식**을 사용했다:
> 각 클래스에 속한 문서들의 TF-IDF 점수를 개별 계산한 뒤 클래스별로 평균을 낸다.

이 방식으로 분석한 결과 `<NUM>` 토큰이 낚시성/정상 모든 클래스에서 점수 1위를 차지하고, `미국`, `정부`, `중국` 등 2022년 뉴스에 전반적으로 많이 나오는 단어들이 유형 구분 없이 상위를 독점하는 문제가 발생했다.

이를 개선하기 위해 두 가지 방법을 검토했다:

- **Ratio 기반**: `이 클래스 평균 TF-IDF / 나머지 클래스 평균 TF-IDF` — 직관적이나 희귀 단어에서 비율이 과도하게 튀는 문제 있음
- **Class-level TF-IDF**: 클래스별 문서를 하나로 합쳐 클래스 문서 간 TF-IDF 적용 — BERTopic 등 최신 NLP 도구들이 실제로 사용하는 방식

최종적으로 **Class-level TF-IDF**로 교체했다.

**실제 효과에 대한 솔직한 평가**

- `<NUM>` 토큰 제거: Class-level TF-IDF의 효과가 아니라, `remove_stopwords()` 함수에서 정규식(`re.match(r'^<.+>$', tok)`)과 `isdigit()` 필터를 추가해서 제거한 것임
- `미국`, `정부`, `중국` 등 공통 단어: Class-level TF-IDF로 바꿔도 여전히 상위에 등장함. `smooth_idf=True` (기본값) 설정 때문에 모든 클래스에 공통인 단어도 IDF=1이 부여되어 완전히 제거되지 않음
- Ratio 기반 TF-IDF와의 비교: 이 데이터셋에서는 실질적인 결과 차이가 없음. 어떤 방법을 써도 `미국`, `코로나`, `정부`는 상위에 올라옴 — 알고리즘 문제가 아니라 2022년 한국 뉴스 데이터의 토픽 특성(코로나 시기, 미중관계) 때문임

**유형별 키워드 분석 결과 요약**

| 유형 | 실제 변별력 있는 키워드 | 품질 |
|---|---|---|
| 선정표현(13) | 충격, 파격, 신음, 달아오르, 경악, 발가벗기, 노골적 | ★★★ 매우 명확 |
| 속어/줄임말(14) | 빠꾸, 코시, 캐리, 멘붕, 버프 | ★★★ 실제 슬랭 확인 |
| 사실과대(15) | 세계(1위), 최고, 역대, 최악 | ★★★ 과장 수식어 명확 |
| 의문유발-은닉(12) | 사람(0.424로 압도적 1위), 밝히, 배우 | ★★ 주어 은닉 패턴 |
| 의문유발-부호(11) | 어떻 (질문 유도 표현) | ★★ 적절 |
| 주어왜곡(16) | 밝히, 나서 외엔 뚜렷한 특징 없음 | ★ 가장 약함 (2,080건으로 데이터 적음) |

**추가 개선 가능 사항 (하지 않아도 발표에 지장 없음)**

- `smooth_idf=False` 설정: `class_tfidf_keywords()` 내부 TfidfVectorizer에 추가하면 모든 유형에 공통인 단어 자동으로 IDF=0으로 처리됨
- 도메인 불용어 추가: `코로나`, `미국`, `중국`, `정부`를 STOPWORDS에 추가하면 유형별 고유 단어만 남음

### BERT 트랙 전처리

- 작업 파일: `bert_track.ipynb`
- 완료된 단계:
  1. 3개 토크나이저 로드 및 테스트 완료
  2. PyTorch `ClickbaitDataset` 클래스 구현 (이진/다중 분류 공용)
  3. DataLoader 배치 테스트 완료 (batch_size=16, shape 검증)
  4. 헬퍼 함수 정의 (`get_dataset`, `get_fold_dataloaders`, `get_class_weights`)
  5. 제목 토큰 수 확인: 최대 36 → `truncation='only_second'` 안전

---

## 데이터 위치 및 로딩 방법

### parquet 파일 (data/processed/)

| 파일 | 건수 | 용도 |
|---|---|---|
| work_pool_clickbait_auto.parquet | 106,014 | 학습/검증 |
| work_pool_clickbait_direct.parquet | 40,106 | 학습/검증 |
| work_pool_nonclickbait_auto.parquet | 145,346 | 학습/검증 |
| work_pool_tfidf_tokens.parquet | 291,466 | TF-IDF 벡터 학습용 |
| test_final.parquet | 36,434 | ⚠️ **봉인 — 최종 평가 전까지 절대 열지 않는다** |

### 데이터 로딩 코드

```python
import pandas as pd

# work_pool 전체 로딩 (BERT 트랙용)
df = pd.concat([
    pd.read_parquet('data/processed/work_pool_clickbait_auto.parquet'),
    pd.read_parquet('data/processed/work_pool_clickbait_direct.parquet'),
    pd.read_parquet('data/processed/work_pool_nonclickbait_auto.parquet'),
], ignore_index=True)

# TF-IDF 트랙용
df_tfidf = pd.read_parquet('data/processed/work_pool_tfidf_tokens.parquet')
```

### 컬럼 설명

| 컬럼 | 설명 | 사용처 |
|---|---|---|
| title_clean | HTML/URL 제거된 제목 (원문 유지) | BERT 입력 |
| content_clean | HTML/URL 제거된 본문 (원문 유지) | BERT 입력 |
| title_tokens | Komoran + 불용어 제거된 토큰 리스트 | TF-IDF 입력 |
| binary_label | 0=정상, 1=낚시성 | 이진 분류 라벨 |
| type_label | 0~5 (Clickbait_Direct), -1 (나머지) | 다중 분류 라벨 |

---

## 할 일 1: TF-IDF 트랙 모델 학습

### 입력 데이터

- `work_pool_tfidf_tokens.parquet`의 TF-IDF 벡터
- `tfidf_track.ipynb`에서 이미 TfidfVectorizer를 fitting 했으므로, 해당 노트북의 벡터를 그대로 사용한다

### 학습할 모델

전통 ML 분류기를 사용한다. 최소 2~3개 비교 권장:

- **로지스틱 회귀** (LogisticRegression) — 기본 베이스라인
- **SVM** (LinearSVC) — 텍스트 분류에 강함
- **Random Forest** 또는 **LightGBM** — 선택사항

### 이진 분류

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# class_weight 불필요 (50:50 균형)
model = LogisticRegression(max_iter=1000)
```

### 다중 분류

```python
# type_label != -1 인 데이터만 필터링
df_multi = df_tfidf[df_tfidf['type_label'] != -1]

# class_weight='balanced' 필수 (7.1배 불균형)
model = LogisticRegression(max_iter=1000, class_weight='balanced')
```

### 실행 환경

- CPU로 충분하다 (로컬 또는 일반 Colab)
- 예상 시간: 5-Fold 전체 수십 분 이내

---

## 할 일 2: BERT 트랙 모델 학습

### 사전 준비

`bert_track.ipynb`의 코드를 Colab 노트북에 복사해서 사용한다. 이미 정의된 것들:

- `ClickbaitDataset` 클래스
- `get_dataset()`, `get_fold_dataloaders()`, `get_class_weights()` 함수
- `MODEL_NAMES` 딕셔너리

### 비교할 모델 3개

| 모델 | HuggingFace ID | vocab size |
|---|---|---|
| KoBERT | `skt/kobert-base-v1` | 8,002 |
| KLUE-RoBERTa | `klue/roberta-base` | 32,000 |
| KoELECTRA | `monologg/koelectra-base-v3-discriminator` | 35,000 |

### 모델 로드 방법

```python
from transformers import AutoModelForSequenceClassification

# 이진 분류 (num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(
    'klue/roberta-base', num_labels=2
)

# 다중 분류 (num_labels=6)
model = AutoModelForSequenceClassification.from_pretrained(
    'klue/roberta-base', num_labels=6
)
```

### 학습 루프 구조

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for model_key in ['kobert', 'klue', 'koelectra']:
    dataset = get_dataset(df, model_key=model_key, task='binary')
    labels = dataset.labels
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(range(len(labels)), labels)):
        train_loader, val_loader = get_fold_dataloaders(dataset, train_idx, val_idx, batch_size=16)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAMES[model_key], num_labels=2
        ).to(device)
        
        # 학습 루프 작성...
```

### 다중 분류 시 class weight 적용

```python
import torch.nn as nn

class_weights = get_class_weights(dataset.labels, device='cuda')
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

확인된 class weight 값:

| 유형 | 라벨 | weight |
|---|---|---|
| 의문유발-부호 | 0 | 0.4497 |
| 의문유발-은닉 | 1 | 0.5720 |
| 선정표현 | 2 | 1.8808 |
| 속어/줄임말 | 3 | 1.8085 |
| 사실과대 | 4 | 1.5810 |
| 주어왜곡 | 5 | 3.2136 |

### 실행 환경

- **Colab Pro** (V100/A100 GPU) 필수
- batch_size=16 (V100 기준, OOM 나면 8로 줄인다)
- 예상 시간: 모델당 5-Fold ~7.5시간, 3개 모델 합계 ~23시간

---

## 할 일 3: 성능 비교 및 최종 평가

### 평가 지표

- Accuracy, F1 (macro), Precision (macro), Recall (macro)
- 5-Fold 평균 ± 표준편차

### 비교 대상

- TF-IDF + 로지스틱 회귀 (+ SVM 등)
- KoBERT
- KLUE-RoBERTa
- KoELECTRA

### 최종 평가 절차

1. 5-Fold CV 결과로 최고 모델 확정
2. 최고 모델을 work_pool 전체로 재학습
3. `test_final.parquet` 봉인 해제 → **딱 1회** 최종 평가
4. 결과 정리 및 발표 자료 반영

---

## 절대 지켜야 할 규칙

| 규칙 | 설명 |
|---|---|
| **test_final 봉인** | 최종 모델 확정 후 딱 1번만 사용한다. 코드에서 실수로 로딩하지 않도록 주의한다 |
| **이진 분류: class_weight 불필요** | 49.9:50.1 균형이므로 가중치 설정 안 해도 된다 |
| **다중 분류: class_weight='balanced' 필수** | 7.1배 불균형 (BERT는 CrossEntropyLoss weight, sklearn은 class_weight='balanced') |
| **다중 분류: Clickbait_Direct만** | type_label != -1 인 데이터만 사용 (40,106건) |
| **5-Fold Stratified CV** | StratifiedKFold(n_splits=5, shuffle=True, random_state=42) — 이진/다중/TF-IDF/BERT 전부 동일 |
| **BERT 제목 truncation 금지** | truncation='only_second' 필수. 제목은 낚시성 판단 대상이므로 절대 자르면 안 된다 |
| **BERT max_length=512** | 이미 Dataset 클래스에 설정되어 있다 |
| **텍스트 중복 251건** | K-Fold 시 같은 텍스트가 train/val에 걸리면 leakage. 필요 시 deduplication 적용 |

---

## K-Fold 학습 흐름

```
work_pool (291,466건)
    │
    ├── Fold 1: [2,3,4,5]로 학습 → [1]로 검증
    ├── Fold 2: [1,3,4,5]로 학습 → [2]로 검증
    ├── Fold 3: [1,2,4,5]로 학습 → [3]로 검증
    ├── Fold 4: [1,2,3,5]로 학습 → [4]로 검증
    └── Fold 5: [1,2,3,4]로 학습 → [5]로 검증
    │
    ▼
  5개 검증 점수 평균 → 모델 성능 비교
    │
    ▼
  최고 성능 모델 선택 → work_pool 전체로 재학습
    │
    ▼
  test_final (36,434건)로 최종 1회 평가 🔒
```

---

## 라벨 정보

### 이진 분류 (binary_label)

| 값 | 의미 | 건수 | 비율 |
|---|---|---|---|
| 0 | 정상 | 145,346 | 49.9% |
| 1 | 낚시성 | 146,120 | 50.1% |

### 다중 분류 (type_label) — Clickbait_Direct만

| 값 | 의미 | 건수 | 비율 |
|---|---|---|---|
| 0 | 의문유발-부호 | 14,863 | 37.1% |
| 1 | 의문유발-은닉 | 11,694 | 29.2% |
| 2 | 선정표현 | 3,555 | 8.9% |
| 3 | 속어/줄임말 | 3,696 | 9.2% |
| 4 | 사실과대 | 4,228 | 10.5% |
| 5 | 주어왜곡 | 2,080 | 5.2% |
| -1 | 해당 없음 (Auto/NonCB) | 251,360 | — |

---

## Colab Pro 계정 정보

BERT 모델 학습(3모델 x 5-Fold)부터 GPU가 필요하다.
TF-IDF 트랙은 CPU로 충분하므로 로컬이나 일반 Colab에서 작업해도 된다.

- Google ID: 카카오톡 공지 참고
- Password: 카카오톡 공지 참고

---

## 참고 파일 목록

| 파일 | 내용 |
|---|---|
| preprocessing_pipeline.md | 전처리 설계 전체 (3단계 15기법, TF-IDF/BERT 분리 근거) |
| youngmin_guide.md | 전처리 담당 작업 가이드 |
| eda_work_pool.ipynb | EDA 시각화 + 해석 |
| tfidf_track.ipynb | TF-IDF 전처리 코드 + 키워드 분석 |
| bert_track.ipynb | BERT 전처리 코드 + Dataset/DataLoader + 헬퍼 함수 |
| data/processed/*.parquet | 전처리 완료된 데이터 |
| progress_log.md | 팀 전체 작업 기록 |

---

## 할 일 4: SHAP 해석 (모델 해석가능성)

> **목적**: "왜 이 제목이 낚시성으로 분류됐나?"를 단어 수준에서 설명한다.  
> 발표에서 "모델이 어떤 패턴을 학습했는가"를 보여주는 핵심 자료가 된다.

### TF-IDF + 로지스틱 회귀 (빠름, 권장)

```python
import shap

# TF-IDF 벡터 준비 (tfidf_track.ipynb의 tfidf_matrix, vectorizer 사용)
# Linear 모델에 특화된 빠른 Explainer
explainer = shap.LinearExplainer(lr_model, tfidf_matrix_train)
shap_values = explainer.shap_values(tfidf_matrix_test)  # 전체 test에 적용 가능

# 시각화
shap.summary_plot(shap_values, tfidf_matrix_test,
                  feature_names=vectorizer.get_feature_names_out())
```

- 속도: 매우 빠름 (수 초). 전체 test 세트에 적용 가능.
- 결과: 낚시성 판별에 기여한 상위 단어 (충격, 경악 등) vs 정상 판별 단어 시각화.

### BERT 모델 (느림, 샘플에만 적용)

BERT에 SHAP를 직접 적용하면 매우 느리다. 실용적인 두 가지 방법:

**방법 A — SHAP Partition Explainer (텍스트용)**
```python
import shap
from transformers import pipeline

# HuggingFace pipeline으로 래핑
pipe = pipeline("text-classification", model=bert_model, tokenizer=tokenizer)
explainer = shap.Explainer(pipe)

# ⚠️ 50~100건만 적용. 1건당 수십 초 소요.
sample_texts = ["[CLS] " + title + " [SEP] " + content[:200] for ...]
shap_values = explainer(sample_texts)
shap.plots.text(shap_values[0])  # 첫 번째 샘플 토큰별 기여도
```

**방법 B — Attention 시각화 (빠른 대안, 엄밀하진 않음)**
```python
# attention_mask 기반 시각화 (BERT가 어느 토큰에 집중했는지)
# transformers BertViz 또는 직접 attention weights 추출
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # (레이어 수, 배치, 헤드, seq, seq)
```

### 권장 실행 계획

| 단계 | 내용 | 소요 시간 |
|---|---|---|
| 1 | TF-IDF + LR: 전체 test 세트에 SHAP 적용 | 수 분 |
| 2 | TF-IDF: 낚시성/정상 상위 기여 단어 bar plot | 수 분 |
| 3 | BERT: 100건 샘플에 SHAP Partition 적용 | 1~2시간 |
| 4 | BERT: 틀린 예측 케이스의 토큰별 기여도 분석 | 수 분 |

### 설치

```bash
pip install shap
```

---

## 할 일 5: LLM 낚시성 설명 및 제목 교정

> **목적**: 탐지 모델이 낚시성으로 분류한 제목을 LLM에게 전달해  
> 낚시성 요소 설명 + 정상 제목으로의 교정안을 생성한다.  
> 발표의 마지막 파트로 활용 가능하다.

### 프롬프트 설계

```python
SYSTEM_PROMPT = """
당신은 한국어 뉴스 품질 전문가입니다.
낚시성 기사 제목을 분석하고 독자를 오도하지 않는 정확한 제목으로 교정합니다.
"""

def make_correction_prompt(title, clickbait_type):
    return f"""다음 뉴스 제목은 '{clickbait_type}' 유형의 낚시성 기사로 분류되었습니다.

제목: "{title}"

다음 두 가지를 작성해주세요:
1. 낚시성 분석: 이 제목의 어떤 표현이 낚시성인지 구체적으로 설명
2. 교정 제목: 내용을 왜곡하지 않고 사실에 충실한 제목으로 수정
"""

# 유형 이름
TYPE_NAMES = {
    0: "의문유발-부호", 1: "의문유발-은닉", 2: "선정표현",
    3: "속어/줄임말", 4: "사실과대", 5: "주어왜곡"
}
```

### OpenAI API 사용 예시

```python
from openai import OpenAI

client = OpenAI(api_key="...")  # API 키는 환경변수로 관리

def correct_clickbait(title, type_label):
    response = client.chat.completions.create(
        model="gpt-4o-mini",   # 비용 절약. 충분히 좋음.
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_correction_prompt(
                title, TYPE_NAMES[type_label]
            )}
        ],
        temperature=0.3,
        max_tokens=300,
    )
    return response.choices[0].message.content
```

### 실행 계획

- 각 유형별 대표 사례 3~5건씩 → 총 18~30건만 API 호출 (비용 최소화)
- 결과를 발표 슬라이드에 "Before → After" 형태로 삽입
- 비용 예상: gpt-4o-mini 기준 30건 ≈ $0.01 수준

### 설치

```bash
pip install openai
```

---

## 할 일 6: Streamlit 데모 (선택사항)

> `progress_log.md`에 기록된 마지막 작업.  
> 시간 여유가 있을 때 구현한다. 발표용 시연 데모로 활용 가능하다.

### 기본 구조

```python
# demo_app.py
import streamlit as st

st.title("낚시성 기사 탐지 데모")

title_input = st.text_input("뉴스 제목 입력")
content_input = st.text_area("뉴스 본문 입력 (선택)")

if st.button("분석"):
    # 1. 모델 예측
    pred_binary = binary_model.predict(...)   # 낚시/정상
    pred_type = type_model.predict(...)        # 유형 (낚시성인 경우)

    # 2. 결과 표시
    if pred_binary == 1:
        st.error(f"⚠️ 낚시성 기사: {TYPE_NAMES[pred_type]}")
    else:
        st.success("✅ 정상 기사")

    # 3. LLM 교정 (선택)
    if pred_binary == 1 and st.checkbox("LLM 교정 제목 생성"):
        corrected = correct_clickbait(title_input, pred_type)
        st.write(corrected)
```

```bash
# 실행
pip install streamlit
streamlit run demo_app.py
```

### 우선순위

발표 준비 시간에 따라 선택적으로 구현한다. 할 일 4(SHAP)와 할 일 5(LLM 교정)가 먼저다.
