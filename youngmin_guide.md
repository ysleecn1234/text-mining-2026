# 영민 작업 가이드

## 현재 상태 (지석이가 완료한 부분)

지석이가 다음을 마쳤다.

- 327,900개 JSON에서 parquet 변환 완료
- 공통 전처리 적용됨: HTML 태그, URL, 이메일 제거. 구두점은 전체 유지.
- 파일 위치: data/processed/ 아래에 4개 parquet 파일
  - work_pool_clickbait_auto.parquet (106,014건)
  - work_pool_clickbait_direct.parquet (40,106건)
  - work_pool_nonclickbait_auto.parquet (145,346건)
  - test_final.parquet (36,434건) — ⚠️ 봉인. 최종 평가 전까지 열지 않는다.
- 컬럼: newsID, newTitle, newsContent, title_clean, content_clean, binary_label, type_label, source_class
- EDA 완료: eda_work_pool.ipynb 참고

## parquet 로딩 방법

```python
import pandas as pd

# work_pool 전체 로딩 (3개 합치기)
df = pd.concat([
    pd.read_parquet('data/processed/work_pool_clickbait_auto.parquet'),
    pd.read_parquet('data/processed/work_pool_clickbait_direct.parquet'),
    pd.read_parquet('data/processed/work_pool_nonclickbait_auto.parquet'),
], ignore_index=True)

# 이진 분류: binary_label (0=정상, 1=낚시)
# 다중 분류: type_label (0~5, Clickbait_Direct만. 나머지는 -1)
```

## 할 일 1: TF-IDF 트랙 전처리

순서:
1. 반복 구두점 정리 (!!!->!, ???->?). 단 ...은 그대로 유지한다.
2. 숫자를 <NUM>으로 치환한다.
3. 띄어쓰기 교정 (선택사항).
4. KoNLPy 형태소 분석 (Mecab 1순위, 안 되면 Komoran).
5. 품사 필터링: 명사(NNG, NNP), 동사(VV), 형용사(VA)만 추출한다.
6. 불용어 제거한다.
7. TfidfVectorizer로 벡터화 (max_features=10000, min_df=5, max_df=0.95).
8. 유형별 키워드 분석한다.

⚠️ 주의사항:
- ...은 절대 제거하지 않는다. EDA에서 낚시성 기사에 4.3배 높게 나왔다.
- ?도 제거하면 안 된다. 3.2배 차이가 난다.
- 구두점 유지 결정은 수업 p.60 근거다. preprocessing_pipeline.md 섹션 10에 코드가 있다.

## 할 일 2: BERT 트랙 전처리

순서:
1. title_clean, content_clean을 그대로 사용한다 (추가 정제 없음).
2. [CLS] 제목 [SEP] 본문 [SEP] 형태로 토크나이징한다.
3. truncation='only_second'로 설정한다. 제목은 보존하고 본문만 자른다.
4. max_length=512로 설정한다.
5. PyTorch Dataset, DataLoader를 구성한다.

비교할 모델 3개:
- KoBERT: skt/kobert-base-v1
- KLUE-RoBERTa: klue/roberta-base
- KoELECTRA: monologg/koelectra-base-v3-discriminator

⚠️ 주의사항:
- 제목에 절대 truncation을 걸면 안 된다. 낚시성 판단 대상이 제목이다.
- preprocessing_pipeline.md 섹션 11에 Dataset 클래스 코드가 있다.

## EDA 결과 중 반영해야 할 것

| 항목 | 내용 | 반영 방법 |
|---|---|---|
| 이진 분류 50:50 균형 | class_weight 불필요 | 이진 분류에서 가중치 설정 안 해도 된다 |
| 다중 분류 7배 불균형 | class_weight='balanced' 필수 | 다중 분류 학습 시 반드시 적용한다 |
| 본문 50% > 1000자 | BERT truncation 필수 | max_length=512, truncation='only_second' |
| 텍스트 중복 251건 | K-Fold 전 deduplication 권장 | 중복 제거 후 학습하면 leakage를 방지할 수 있다 |
| 구두점 유지 | TF-IDF에서도 완전 제거 금지 | 반복만 정리하고 개별 부호는 살려야 한다 |

## K-Fold 설정

- 5-Fold Stratified Cross-Validation
- 이진/다중/TF-IDF/BERT 전부 동일하게 5-Fold 적용한다
- Colab Pro (V100) 사용한다
- sklearn의 StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


## Colab Pro 계정 정보

BERT 모델 학습(3모델 x 5-Fold)부터 GPU가 필요하다. 아래 계정으로 Colab Pro에 접속한다.
TF-IDF 트랙은 CPU로 충분하므로 로컬이나 일반 Colab에서 작업해도 된다.

- Google ID: [EMAIL_ADDRESS]
- Password: [PASSWORD]
카카오톡으로 공지에 올리겠음.


## 참고 파일 목록

| 파일 | 내용 |
|---|---|
| preprocessing_pipeline.md | 전처리 설계 전체, TF-IDF/BERT 코드 가이드 |
| eda_work_pool.ipynb | EDA 시각화 + 해석 |
| data/processed/*.parquet | 공통 전처리 완료된 데이터 |
| data/scripts/extract_dataset.py | ZIP 압축 해제 스크립트 |
| data/scripts/common_preprocess.py | 공통 전처리 스크립트 |
