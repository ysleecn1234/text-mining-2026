# 진행 기록

## 2026-04-08

### 데이터 구조 파악
- AI Hub "낚시성 기사 탐지 데이터" 1세부 다운로드
- 84개 zip 파일 (1,247.3 MB), Training/Validation x 원천/라벨 x 3클래스 x 7카테고리 구조
- 파일명 규칙 파악: {T|V}{S|L}_Part1_{클래스}_{카테고리}.zip

### JSON 스키마 분석
- PO 카테고리 4개 zip에서 샘플 JSON 확인
- 주요 내용:
  - 모델 입력은 newTitle(라벨러가 만든 제목) + newsContent(본문)
  - clickbaitClass 라벨이 관례와 반대 (0=낚시, 1=정상)이므로 뒤집어 사용한다
  - _L.json(라벨 파일)에 원천 데이터 내용이 모두 포함되어 있다. _S.json이나 원천데이터는 불필요하다.
  - 6개 유형 라벨은 processPattern 필드(11~16)에 있고, Clickbait_Direct만 해당된다

### processPattern 분포 전수 스캔
- Clickbait_Direct의 6개 유형: 11(의문유발-부호), 12(의문유발-은닉), 13(선정표현), 14(속어/줄임말), 15(사실과대), 16(주어왜곡)
- Clickbait_Auto는 processPattern=99, NonClickbait는 00

### 스키마 및 라벨 체계 확정
- 이진 분류: Clickbait_Auto + Clickbait_Direct = 낚시(1), NonClickbait = 정상(0)
- 다중 분류: Clickbait_Direct만 사용, processPattern 11~16을 0~5로 매핑한다

---

## 2026-04-09

### 전처리 파이프라인 설계
- preprocessing_pipeline.md 작성 (수업 PDF 기반 3단계 15기법)
- TF-IDF 트랙과 BERT 트랙 분리 설계
- 형태소 분석기 비교: Mecab 1순위, Komoran 2순위

### 폴더 및 분할 전략 확정
- AI Hub Training을 work_pool (291,466건), Validation을 test_final (36,434건, 봉인)로 명명했다
- 5-Fold Stratified Cross-Validation, 전체 데이터 사용, Colab Pro 실행으로 결정했다

### extract_dataset.py 수정 및 실행
- train/val 폴더명을 work_pool/test_final로 변경했다
- 라벨 zip만 필터링 (42개 zip, _L.json만 해제)
- 327,900개 _L.json 파일 추출을 완료했다

### 공통 전처리 실행
- JSON에서 parquet으로 변환했다 (클래스별 분할 저장, OOM 방지)
- 정제 범위: HTML 태그, URL, 이메일만 제거했다
- 구두점은 전체 유지했다 (EDA에서 ?, ... 등이 낚시성 핵심 신호로 확인됨)
- 산출물:
  - work_pool_clickbait_auto.parquet (106,014건)
  - work_pool_clickbait_direct.parquet (40,106건)
  - work_pool_nonclickbait_auto.parquet (145,346건)
  - test_final.parquet (36,434건)

### EDA 수행
- eda_work_pool.ipynb 작성 및 실행 (10개 섹션, 시각화 + 해석)
- 주요 결과:
  - 이진 분류: 145,346 vs 146,120 (49.9:50.1)으로 균형 상태다
  - 다중 분류: 7.1배 불균형 (의문유발-부호 14,863건 최대 / 주어왜곡 2,080건 최소)
  - 제목 평균 32.0자, 정상 31.4 vs 낚시 32.6으로 차이가 미미하다
  - 본문 50.0%가 1000자 초과, 21.7%가 1500자를 초과한다
  - 물음표(?) 낚시성 3.2배, ...(3점) 낚시성 4.3배로 구두점 유지 근거가 된다
  - 결측값 0건, 텍스트 중복 251건 (0.09%)
  - 본문/제목 비율: 정상 30.1 vs 낚시 33.8로 가설과 반대 결과가 나왔다 (길이가 아닌 의미 불일치가 핵심)

### TF-IDF/BERT 트랙 전처리 가이드 작성
- preprocessing_pipeline.md에 섹션 10(TF-IDF), 섹션 11(BERT) 추가했다
- 영민이가 참고할 코드 가이드 및 EDA 반영 사항을 포함했다

---

## 2026-04-12

### TF-IDF 트랙 전처리 (영민) — 이전에 완료
- 작업 파일: tfidf_track.ipynb
- 저장 파일: data/processed/work_pool_tfidf_tokens.parquet
- parquet 3개 병합 (291,466건) → 반복 구두점 정리 → Komoran 형태소 분석 (NNG/NNP/VV/VA, SN→`<NUM>`) → 불용어 제거 → TfidfVectorizer 벡터화 (max_features=10000, min_df=5, max_df=0.95) → 유형별 키워드 분석 완료

### BERT 트랙 전처리 (영민)
- 작업 파일: bert_track.ipynb
- 패키지 설치: transformers 5.5.3, torch 2.11.0, sentencepiece 0.2.1, protobuf
  - kobert-tokenizer는 Python 3.14 미지원 → AutoTokenizer + sentencepiece로 대체했다
- 데이터 로딩: 3개 parquet 병합 291,466건, 결측값 0건, 빈 문자열 0건 확인
- 토크나이저 로드 결과:
  - KoBERT (skt/kobert-base-v1): vocab 8,002, SentencePiece 기반, token_type_ids 미반환
  - KLUE-RoBERTa (klue/roberta-base): vocab 32,000, BPE 기반, token_type_ids 미반환
  - KoELECTRA (monologg/koelectra-base-v3-discriminator): vocab 35,000, WordPiece 기반, token_type_ids 반환
- 토크나이징 테스트 (샘플 1건):
  - KoBERT: 제목 30토큰, 전체 512토큰 (max_length 도달)
  - KLUE-RoBERTa: 제목 21토큰, 전체 474토큰
  - KoELECTRA: 제목 21토큰, 전체 484토큰
- 제목 토큰 수 안전 확인 (5,000건 샘플, KLUE-RoBERTa 기준):
  - 평균 17.2, 최대 36, 510 초과 0건 → truncation='only_second' 안전하게 사용 가능하다
- ClickbaitDataset 클래스 구현: 이진/다중 분류 공용, on-the-fly 토크나이징, token_type_ids 미반환 시 자동 0 텐서 생성
- DataLoader 배치 테스트: batch_size=16, shape [16, 512] 정상 동작 확인
- 헬퍼 함수 정의: get_dataset(), get_fold_dataloaders(), get_class_weights()
- 다중 분류 class weights 확인:
  - 의문유발-부호 0.4497 / 의문유발-은닉 0.5720 / 선정표현 1.8808 / 속어줄임말 1.8085 / 사실과대 1.5810 / 주어왜곡 3.2136

### 모델링 가이드 작성 (영민)
- modeling_guide.md 작성: 모델링 담당 팀원을 위한 작업 가이드
- 포함 내용: 데이터 로딩 방법, TF-IDF/BERT 모델 학습 절차, K-Fold 설정, class weight, 규칙 체크리스트, Colab Pro 정보

---

## 현재 상태

| 항목 | 상태 |
|---|---|
| 데이터 구조 파악 | 완료 |
| ZIP 압축 해제 | 완료 (327,900건) |
| 공통 전처리 | 완료 (parquet 저장) |
| EDA | 완료 (eda_work_pool.ipynb) |
| TF-IDF 트랙 전처리 | 완료 (tfidf_track.ipynb) |
| BERT 트랙 전처리 | 완료 (bert_track.ipynb) |
| 모델링 가이드 | 완료 (modeling_guide.md) |
| 모델 학습 | 미시작 |
| SHAP 해석 | 미시작 |
| LLM 설명/교정 | 미시작 |
| Streamlit 데모 | 미시작 |
