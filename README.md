# 한국어 뉴스 낚시성 기사 탐지 (Clickbait Detection)
 
한국어 뉴스 기사의 제목과 본문을 분석해 **낚시성(clickbait) 기사**를 자동으로 탐지하고, 낚시성이라면 **어떤 유형**인지까지 분류하는 텍스트 마이닝 프로젝트입니다.
 
---
 
## 배경 / 문제 정의
 
낚시성 기사는 자극적인 제목으로 클릭을 유도하지만 본문이 기대에 못 미쳐 정보 신뢰도를 떨어뜨립니다. 이 프로젝트는 약 32만 건의 뉴스 데이터를 사용해 두 가지 과제를 다룹니다: (1) 기사가 낚시성인지 아닌지 가리는 **이진 분류**, (2) 낚시성 기사를 의문 유발·선정 표현·사실 과대 등 **6개 세부 유형으로 나누는 다중 분류**. 나아가 전통적 머신러닝(TF-IDF)과 딥러닝(BERT) 접근을 같은 조건에서 비교하고, 모델이 "왜 낚시성으로 판단했는지"를 설명하는 해석 단계까지 포함합니다.
 
데이터는 AI Hub의 "낚시성 기사 탐지 데이터"(약 1.2GB, 84개 압축파일)를 사용했습니다.
 
## 사용 기술
 
- **언어·환경**: Python, Jupyter Notebook, Google Colab Pro
- **데이터 처리**: pandas, Parquet(대용량 데이터 저장 포맷), 정규표현식 기반 정제(HTML 태그·URL 제거), Komoran·Mecab 형태소 분석기
- **전통적 모델(베이스라인)**: scikit-learn TF-IDF(단어 중요도 기반 벡터화) + 분류기
- **딥러닝 모델**: Hugging Face Transformers + PyTorch 기반 한국어 BERT 계열 3종 — KLUE-RoBERTa, KoBERT, KoELECTRA 파인튜닝
- **평가·검증**: 5-Fold Stratified Cross-Validation(층화 5겹 교차검증), 외부 테스트셋 평가, 데이터 누수(train/test 오염) 점검
- **해석·확장 실험**: SHAP(모델 예측 근거를 시각화하는 해석 기법), 대규모 언어모델(LLM)을 활용한 제목 교정 및 zero-shot(사전학습만으로 예측) vs 파인튜닝 성능 비교
## 프로젝트 구조
 
노트북(.ipynb) 중심으로 단계별로 구성되어 있으며, 대략 다음 흐름을 따릅니다.
 
```
data/scripts/        # 원본 데이터 추출·공통 전처리 스크립트
  ├─ extract_dataset.py      # AI Hub zip → JSON 추출
  └─ common_preprocess.py    # 정제 후 Parquet 변환
data/processed/      # 전처리된 데이터 설명
 
eda_work_pool.ipynb           # EDA(탐색적 데이터 분석): 분포·길이·구두점 패턴
 
tfidf_track.ipynb             # TF-IDF 트랙 전처리(형태소 분석·벡터화)
tfidf_modeling_shap.ipynb     # TF-IDF 모델링 + SHAP 해석
 
bert_track.ipynb              # BERT 트랙 전처리·토크나이징
bert_train_klue.ipynb         # KLUE-RoBERTa 학습
bert_train_kobert.ipynb       # KoBERT 학습
bert_train_koelectra.ipynb    # KoELECTRA 학습
bert_test_eval.ipynb          # 테스트셋 평가
bert_shap.ipynb               # BERT SHAP 해석
 
data_leakage_check.ipynb      # 데이터 누수 점검
external_test*.py / .ipynb    # 외부 데이터 일반화 성능 평가
llm_title_correction*.ipynb   # LLM 기반 제목 교정 실험
zeroshot_vs_finetuned.ipynb   # zero-shot vs 파인튜닝 비교
 
models/              # 5-Fold 교차검증 수치 결과(.json)
results/             # 모델별 성능 시각화(.png)
*.md                 # 전처리·모델링 가이드, 진행 로그 등 문서
```
 
> 데이터 흐름: **원본 추출 → 공통 전처리(Parquet) → EDA → TF-IDF / BERT 두 트랙으로 분기 → 학습 → 평가·해석**
 
학습된 모델 체크포인트(.pt)는 용량 문제로 저장소에 포함되지 않으며, 별도 링크(Google Drive)로 제공됩니다. 교차검증 수치 결과(.json)는 `models/` 폴더에 있습니다.
 
## 결과 / 산출물
 
- **이진 분류(낚시성 여부)**: 대표 모델 KLUE-RoBERTa 기준 5-Fold 평균 F1(정밀도·재현율의 조화평균) 약 **0.987**.
- **다중 분류(6개 유형)**: KLUE-RoBERTa 기준 정확도 약 **92%**, F1(macro) 약 **0.885**.
- TF-IDF 베이스라인과 BERT 계열 3종을 동일 조건에서 비교해 접근법별 성능 차이를 정리했습니다.
- SHAP 해석으로 어떤 단어·표현이 낚시성 판단에 기여했는지 시각화했고, 외부 테스트셋 평가로 실제 일반화 성능을 확인했습니다.
- 모델별 성능 그래프는 `results/`, 세부 수치는 `models/`의 JSON 파일에서 확인할 수 있습니다.
