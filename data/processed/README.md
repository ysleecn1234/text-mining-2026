# data/processed/ 파일 설명

전처리 완료된 데이터 파일 목록입니다.

---

## 📁 파일 목록

### work_pool_clickbait_auto.parquet
- **건수**: 106,014건
- **출처**: AI Hub Training → Clickbait_Auto 클래스
- **설명**: 자동 생성된 낚시성 기사 (binary_label=1, type_label=-1)
- **생성**: 지석 (JSON → parquet 변환 + 공통 전처리)

### work_pool_clickbait_direct.parquet
- **건수**: 40,106건
- **출처**: AI Hub Training → Clickbait_Direct 클래스
- **설명**: 수동 라벨링된 낚시성 기사 (binary_label=1, type_label=0~5)
- **생성**: 지석 (JSON → parquet 변환 + 공통 전처리)

### work_pool_nonclickbait_auto.parquet
- **건수**: 145,346건
- **출처**: AI Hub Training → NonClickbait_Auto 클래스
- **설명**: 정상 기사 (binary_label=0, type_label=-1)
- **생성**: 지석 (JSON → parquet 변환 + 공통 전처리)

### work_pool_tfidf_tokens.parquet
- **건수**: 291,466건 (위 3개 병합)
- **설명**: TF-IDF 트랙 전처리 완료 파일
- **생성**: 영민 (Komoran 형태소 분석 + 불용어 제거 + 구두점 신호 추가)
- **추가 컬럼**:
  - `title_tfidf`: 반복 구두점 정리 후 제목 (`?`, `...` 보존)
  - `title_morphs`: Komoran 형태소 분석 결과 (NNG/NNP/VV/VA + `<NUM>`)
  - `title_tokens`: 최종 TF-IDF 입력 토큰 (`HAS_QUESTION`, `HAS_ELLIPSIS` 포함)

### test_final.parquet
- **건수**: 36,434건
- **출처**: AI Hub Validation 전체
- **설명**: 최종 평가 전용 데이터
- **생성**: 지석

> ⚠️ **봉인** — 최고 모델 확정 후 딱 1번만 사용한다. 절대 미리 열지 않는다.

---

## 📋 공통 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `newsID` | 기사 고유 ID |
| `newTitle` | 원본 제목 (라벨러가 만든 낚시성 제목) |
| `newsContent` | 원본 본문 |
| `title_clean` | HTML/URL/이메일 제거된 제목 |
| `content_clean` | HTML/URL/이메일 제거된 본문 |
| `binary_label` | 0=정상, 1=낚시성 |
| `type_label` | 0~5 (Clickbait_Direct만), -1 (나머지) |
| `source_class` | clickbait_auto / clickbait_direct / nonclickbait_auto |

### type_label 매핑

| 값 | 유형 | 건수 |
|---|---|---|
| 0 | 의문유발-부호 | 14,863 |
| 1 | 의문유발-은닉 | 11,694 |
| 2 | 선정표현 | 3,555 |
| 3 | 속어/줄임말 | 3,696 |
| 4 | 사실과대 | 4,228 |
| 5 | 주어왜곡 | 2,080 |
| -1 | 해당없음 | 251,360 |

---

## 🔧 로딩 방법

```python
import pandas as pd

# BERT 트랙: work_pool 3개 병합
df = pd.concat([
    pd.read_parquet('data/processed/work_pool_clickbait_auto.parquet'),
    pd.read_parquet('data/processed/work_pool_clickbait_direct.parquet'),
    pd.read_parquet('data/processed/work_pool_nonclickbait_auto.parquet'),
], ignore_index=True)

# TF-IDF 트랙
df_tfidf = pd.read_parquet('data/processed/work_pool_tfidf_tokens.parquet')

# ⚠️ test_final은 최종 평가 전까지 절대 열지 않는다
```
