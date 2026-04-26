# 전처리 파이프라인 설계

> **근거 자료**: 광운대 텍스트마이닝 수업 `2026-TM-4-1-Text-Preprocessing.pdf` (95p, Dr. Minsu Cho)
> **프로젝트**: AI Hub "낚시성 기사 탐지 데이터" 1세부 — 낚시성 자동 탐지 및 6개 유형 분류
> **작성일**: 2026-04-08

---

## 0. 설계 원칙

1. 수업에서 다룬 **3단계 파이프라인, 15가지 기법**을 기본 골격으로 사용.
2. 수업에서 강조된 "모델 종류에 따라 다르게 전처리해야 한다"는 지침에 따라 **TF-IDF 트랙**과 **BERT 트랙**을 분리.
3. 한국어 뉴스 기사에 부적절한 기법(예: 숫자 일괄 제거)은 근거를 남기고 선별 적용.
4. 수업 범위를 벗어나는 기법은 추가하지 않음. 단, 형태소 분석기 선택은 수업에서 다루지 않으므로 별도 섹션으로 정리 (아래 §4 참고).

---

## 1. Phase 1. Cleaning (정제) — 5가지

| 기법 | 내용 | 적용 |
|---|---|---|
| Remove Noise | HTML 태그, URL, 멘션(@), 해시태그(#) 제거 |  뉴스 본문에 `<br>`·URL 섞임 |
| Remove Repeated Punctuation | `!`·`?` → `!` 또는 태그화 |  낚시성 기사 특유의 감정 반복 처리 |
| Remove Punctuation | 문장부호 완전 제거 | ⚠️ **TF-IDF 트랙에만**. BERT는 문장부호 유지가 유리 |
| Remove Numbers | 숫자 제거 | ⚠️ **제거 대신 `<NUM>` 치환**. 뉴스엔 연도·통계가 의미 핵심 (강의자료 p.41 경고) |
| Remove Stopwords | 기능어(조사·어미 등) 제거 |  TF-IDF 트랙에 한해 적용 |

## 2. Phase 2. Normalization (정규화) — 6가지

| 기법 | 내용 | 적용 |
|---|---|---|
| Lowercasing | 대→소문자 |  영문 토큰에만 |
| Replace Slang / Abbreviations | 은어·줄임말 표준화 | △ 뉴스 특성상 영향 적음. 선택적 |
| Removing Elongation | `cooool` → `cool` | △ 인용문에 한해 선택적 |
| Replace Contraction | `don't` → `do not` | △ 영어용 |
| Spelling Correction | Bayesian 기반 철자 교정 | △ 기사 품질이 높아 선택적 |
| **Word Segmentation (띄어쓰기 교정)** | 확률 기반 공백 보정 |  **한국어에 매우 중요**. 강의에서 별도 섹션 할애 (p.48~49) |

## 3. Phase 3. Semantic Refinement (의미 정제) — 핵심

| 기법 | 내용 | 적용 |
|---|---|---|
| **Tokenization (토큰화)** | Word / Character / Subword(BPE·WordPiece) |  TF-IDF 트랙 → KoNLPy 형태소 토큰 / BERT 트랙 → 내부 WordPiece 자동 |
| Stemming | 규칙 기반 어간 자르기 | △ 영어 위주. 한국어에는 부적합 |
| Lemmatization | 사전 기반 표제어 복원 |  한국어는 형태소 분석기의 기본형 복원으로 대체 |
| **POS Tagging (품사 태깅)** | Rule / Statistical(HMM) / Deep Learning |  **필수**. KoNLPy로 명사·동사·형용사만 필터링해 의미 집중도↑ |

---

## 4. 한국어 형태소 분석기 선택 (KoNLPy)

수업에서는 형태소 분석기의 **원리**(룰/HMM/딥러닝 기반 POS tagging)까지만 다루고, 실제 어떤 라이브러리를 쓸지는 지정하지 않음. KoNLPy가 제공하는 주요 4종을 비교한 뒤 선택:

| 분석기 | 기반 | 특징 | 속도 | 정확도 | 비고 |
|---|---|---|---|---|---|
| **Okt** (구 Twitter) | 규칙+사전 | 간단·빠름, 신조어·구어체에 비교적 강함 | 빠름 | 중 | 어미 분석은 거침. 트위터·SNS·리뷰에 적합 |
| **Mecab** | 통계(CRF) | **가장 빠르고 일관성 높음**. 대용량에 최적 | **매우 빠름** | 높음 | 설치가 가장 까다로움(윈도우). 뉴스·대용량 처리 표준 |
| **Kkma** | 사전+규칙 | 분석이 세밀·정확. 복합명사 분리 꼼꼼 | **매우 느림** | 높음 | 36만 건 대용량엔 부적합 |
| **Komoran** | 사전+규칙 | Okt와 Mecab의 중간. 오탈자·띄어쓰기 오류에 강함 | 중간 | 중~높음 | 사용자 사전 추가 쉬움 |

### 우리 프로젝트 권장: **Mecab (1순위) / Komoran (2순위)**

**이유**
- 데이터 규모가 **36만 4천 건**이라 속도가 결정적. Kkma는 실질적으로 불가능.
- 입력이 **신문 기사(정제된 한국어)** 이므로 신조어 대응력(Okt의 장점)은 상대적으로 덜 중요.
- Mecab이 한국어 뉴스 처리의 사실상 표준이며, 일관된 형태소 분할로 TF-IDF 키워드 품질이 가장 안정적.
- Mecab 설치 문제가 생기면 **Komoran**이 대안. 사용자 사전으로 언론사·인물명 등 고유명사 보강이 쉬움.

**최종 선택은 환경 확인 후 결정** — Mecab 설치가 가능하면 Mecab, 불가능하면 Komoran.

---

## 5. 통합 파이프라인 다이어그램

```
[원문 JSON: 제목 + 본문]
        │
        ▼
 ─── 공통 Phase 1 정제 ───
   · Remove Noise (HTML/URL)
   · Remove Repeated Punctuation
        │
        ▼
 ─── 공통 Phase 2 정규화 ───
   · Word Segmentation (띄어쓰기 교정)
   · (선택) Spelling Correction
        │
        ├────────────────────┬────────────────────┐
        ▼                    ▼                    │
 [TF-IDF 트랙]          [BERT 트랙]               │
 · Remove Punctuation   · 문장부호 유지            │
 · <NUM> 치환            · 숫자 유지               │
 · Remove Stopwords     · 불용어 유지             │
 · KoNLPy 형태소 분석    · (토크나이징 없음 —       │
 · POS 필터              BERT 토크나이저가          │
   (N/V/A만 추출)         자동 처리)               │
        │                    │                    │
        ▼                    ▼                    │
  TF-IDF 벡터          [CLS] 제목 [SEP] 본문        │
  (키워드 분석용)        (모델 입력용)              │
```

---

## 6. 확정 항목

- [x] 데이터 스키마 확인 후, 제목·본문 외에 사용할 필드 결정 (단계 1-b·1-c) — 전체 데이터 사용
- [x] 형태소 분석기 최종 확정 (Mecab vs Komoran) — Komoran 사용
- [x] 불용어 사전 확정 — https://www.ranks.nl/stopwords/korean 사용
- [x] 제목·본문 최대 토큰 길이 (BERT는 512 제한) — K-Fold 적용
- [x] 이 파이프라인 코드 구현 전 최종 사용자 승인 — Colab Pro 환경에서 진행


---

## 7.  폴더 네이밍 규칙 & 분할 전략

> **팀원 전체 필독. 폴더 이름, 데이터 분할 방식, 실행 환경이 여기서 확정된다.**

### 7-1. 왜 폴더 이름을 바꾸는가?

AI Hub 원본 폴더명은 `Training` / `Validation`인데, 우리 프로젝트에서 쓰는 train / val / test 개념과 **의미가 겹쳐서 혼란**이 생긴다.

| AI Hub 원본 이름 | 우리 프로젝트 이름 | 역할 |
|---|---|---|
| `Training` | **`work_pool`** | 학습 + 검증용 (K-Fold로 내부 분할) |
| `Validation` | **`test_final`** | 최종 평가 전용 (**봉인 — 마지막에 1번만 사용**) |

### 7-2. 폴더 구조 (압축 해제 후)

```
data/
├── work_pool/                    ← AI Hub "Training" (291,466건)
│   ├── clickbait_auto/           ← _L.json 파일들 (105,838건)
│   ├── clickbait_direct/         ← _L.json 파일들 (40,281건)
│   └── nonclickbait_auto/        ← _L.json 파일들 (145,347건)
│
└── test_final/                   ← AI Hub "Validation" (36,434건) 🔒 봉인
    ├── clickbait_auto/           ← (13,429건)
    ├── clickbait_direct/         ← (4,837건)
    └── nonclickbait_auto/        ← (18,168건)
```

- `02.라벨링데이터`의 `_L.json`만 해제한다. `01.원천데이터`는 해제하지 않는다 (이유: §8 참고).
- 하위 폴더명은 클래스명 소문자 (`clickbait_auto`, `clickbait_direct`, `nonclickbait_auto`).

### 7-3. 데이터 분할 전략: 5-Fold Cross-Validation

#### 확정 사항

| 항목 | 결정 |
|---|---|
| **분할 방식** | **Stratified 5-Fold CV** (이진·다중 모두) |
| **대상 데이터** | `work_pool` 전체 (291,466건) |
| **test_final 사용 시점** | 5-Fold 완료 후 최종 1회 평가에만 사용 |
| **적용 모델** | TF-IDF 트랙 + BERT 트랙 (KoBERT, KLUE-RoBERTa, KoELECTRA) 전부 |
| **실행 환경** | **Google Colab Pro** (V100/A100 GPU) |

#### K-Fold 선택 이유

- **안정성**: 단일 80:20 split은 val 세트가 우연히 편향될 수 있음. K-Fold는 5번 돌려서 평균내므로 성능 추정의 신뢰도가 높음.
- **데이터 효율**: 모든 데이터가 한 번씩 검증에 쓰이므로 데이터 낭비 없음.
- **발표 설득력**: "5-Fold CV 결과입니다"는 교수님·청중에게 훨씬 신뢰감 있음.

#### 전체 데이터 사용 이유 (샘플링 안 하는 이유)

- 데이터가 충분히 있고 Colab Pro로 컴퓨팅 자원이 허용되므로, **쓸 수 있는 데이터는 전부 활용하는 것이 원칙**.
- 일부만 샘플링하면 "왜 있는 데이터를 안 썼냐"에 대한 답변이 약해짐.
- 시간 여유 시 "10만 건 vs 전체" 비교 실험을 보너스로 추가 가능.

#### 예상 학습 시간 (Colab Pro V100 기준)

| 모델 | 1 Fold | 5 Fold | 비고 |
|---|---|---|---|
| TF-IDF + 로지스틱 회귀 등 | ~수 분 | ~수십 분 | CPU만으로도 충분 |
| KoBERT | ~1.5시간 | ~7.5시간 | |
| KLUE-RoBERTa | ~1.5시간 | ~7.5시간 | |
| KoELECTRA | ~1.5시간 | ~7.5시간 | |
| **합계** | | **~23시간** | Colab Pro로 1~2일 |

#### K-Fold 학습 흐름

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
  5개 검증 점수 평균 → 모델 성능 비교 (KoBERT vs RoBERTa vs KoELECTRA)
    │
    ▼
  최고 성능 모델 선택 → work_pool 전체로 재학습
    │
    ▼
  test_final (36,434건)로 최종 1회 평가 🔒
```

### 7-4. test_final 봉인 규칙

1. **EDA, 전처리 개발, 하이퍼파라미터 튜닝 — 전부 `work_pool`에서만** 수행.
2. `test_final`은 **최고 모델 확정 후 딱 1번**만 열어서 최종 성능 측정.
3. test_final을 미리 보면 데이터 누출(data leakage)이 발생해 성능이 과대 추정됨.
4. 코드에서 test_final 경로를 실수로 로드하지 않도록 **경로를 별도 설정 파일로 분리** 권장.

---

## 8.  데이터 파일 구조 & 읽기 규칙 (전처리 담당자 필독)

> **이 섹션은 전처리 코드를 작성하는 영민이(및 팀원 전체)가 반드시 숙지해야 할 내용이다.**

### 8-1. 원본 폴더 구조 (다운로드 상태)

AI Hub에서 다운받은 원본 폴더는 이렇게 생겼다:

```
Training/
├── 01.원천데이터/        ← _S.json 파일들 (원천만, 라벨 없음)
│   ├── TS_Part1_Clickbait_Auto_EC.zip
│   ├── TS_Part1_Clickbait_Direct_EC.zip
│   ├── TS_Part1_NonClickbait_Auto_EC.zip
│   └── ... (카테고리별 zip)
│
└── 02.라벨링데이터/      ← _L.json 파일들 (원천 + 라벨 모두 포함)  이것만 사용
    ├── TL_Part1_Clickbait_Auto_EC.zip
    ├── TL_Part1_Clickbait_Direct_EC.zip
    ├── TL_Part1_NonClickbait_Auto_EC.zip
    └── ... (카테고리별 zip)

Validation/
├── 01.원천데이터/
└── 02.라벨링데이터/      ← 마찬가지로 이것만 사용
```

### 8-2. 핵심: 원천과 라벨이 "따로 있지만" 라벨 파일 하나로 충분하다

**폴더는 `01.원천데이터`와 `02.라벨링데이터`로 분리**되어 있다. 그래서 얼핏 보면 두 폴더를 모두 읽어서 매칭해야 할 것 같지만, **그럴 필요 없다.**

이유: `_L.json`(라벨 파일) 내부에 원천 정보가 **이미 포함**되어 있기 때문이다.

```
01.원천데이터 → _S.json 내부 구조:
{
  "sourceDataInfo": {
    "newsTitle": "...",        ← 원래 제목
    "newsContent": "...",      ← 본문
    ...
  }
  // labeledDataInfo 없음.
}

02.라벨링데이터 → _L.json 내부 구조:
{
  "sourceDataInfo": {          ← ⭐ 원천 정보가 여기도 들어있음
    "newsTitle": "...",
    "newsContent": "...",
    ...
  },
  "labeledDataInfo": {         ← 라벨 정보 추가
    "newTitle": "...",
    "clickbaitClass": 0,
    "referSentenceInfo": [...]
  }
}
```

**결론: `02.라벨링데이터/` 의 `_L.json`만 읽으면 된다. `01.원천데이터/`는 무시해도 된다.**

### 8-3. 전처리 코드에서 JSON 읽는 법 (예시)

```python
import json

with open("PO_M01_000001_L.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# ── 이진 분류용 ──
title   = data["labeledDataInfo"]["newTitle"]        # 라벨러가 만든 제목 (모델 입력)
content = data["sourceDataInfo"]["newsContent"]       # 본문 (모델 입력)
label   = 1 - data["labeledDataInfo"]["clickbaitClass"]  # 라벨 반전: 낚시=1, 정상=0

# ── 다중 분류용 (Clickbait_Direct만 해당) ──
process = str(data["sourceDataInfo"]["processPattern"])  # "11"~"16"
type_label = TYPE_MAP[process]  # 0~5 매핑
```

### 8-4. 자주 하는 실수 & 주의사항

| 실수 | 왜 위험한가 | 올바른 방법 |
|---|---|---|
| `01.원천데이터`에서 `_S.json`을 읽고 라벨을 매칭하려 함 | 불필요한 작업. `_L.json`에 다 있음 | `02.라벨링데이터`의 `_L.json`만 사용 |
| `newsTitle`(원래 제목)을 모델 입력으로 사용 | 원제는 정상 제목이라 학습이 무의미해짐 | `labeledDataInfo.newTitle`을 사용 |
| `clickbaitClass`를 그대로 라벨로 사용 | 0=낚시, 1=정상으로 관례와 반대 | `label = 1 - clickbaitClass`로 반전 |
| 다중 분류에 Auto/NonCB 데이터도 포함 | 유형 라벨(processPattern)이 없음 (99, 00) | Clickbait_Direct만 필터링 |

### 8-5. 요약 한 줄

> **`02.라벨링데이터` 폴더의 `_L.json`만 열면, 제목(`newTitle`) + 본문(`newsContent`) + 이진 라벨(`clickbaitClass`) + 유형 라벨(`processPattern`)이 전부 나온다. 원천데이터 폴더는 건드리지 않아도 된다.**

---

## 9. EDA (탐색적 데이터 분석)

EDA는 완료되었다. work_pool 데이터를 대상으로 수행했고, test_final은 사용하지 않았다.
상세 시각화와 해석은 eda_work_pool.ipynb에 있다.

### 9-1. 주요 결과 요약

| 항목 | 결과 | 시사점 |
|---|---|---|
| 이진 분류 균형 | 145,346 vs 146,120 (49.9:50.1) | class_weight 불필요 |
| 다중 분류 불균형 | 최대 14,863 / 최소 2,080 (7.1배) | class_weight='balanced' 필수 |
| 제목 길이 | 평균 32.0자, 정상 31.4 vs 낚시 32.6 | 길이 차이 미미하므로 내용이 핵심이다 |
| 본문 길이 | 50.0%가 1000자 초과, 21.7%가 1500자 초과 | head truncation 전략이 필요하다 |
| 결측값 | null, 빈문자열 0건 | 별도 처리 불필요 |
| 텍스트 중복 | 251건 (0.09%) | K-Fold 시 leakage 주의가 필요하다 |
| 물음표(?) 빈도 | 낚시성에서 3.2배 높음 | 구두점 유지 결정의 근거이다 |
| ...(3점) 빈도 | 낚시성에서 4.3배 높음 | 구두점 유지 결정의 추가 근거이다 |
| 본문/제목 비율 | 정상 30.1 vs 낚시 33.8 | 길이가 아닌 의미적 불일치가 핵심이다 |
| 구두점 처리 | 전체 유지 | 수업 p.60 근거 |
