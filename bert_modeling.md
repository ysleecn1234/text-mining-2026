# BERT 트랙 모델링 기록

낚시성 기사 탐지 프로젝트의 BERT 계열 파인튜닝 전 과정을 기록한 문서입니다.

---

## 1. 프로젝트 개요

| 항목 | 내용 |
|------|------|
| 목표 | 한국어 뉴스 기사의 낚시성 탐지 (이진 분류 + 유형 분류) |
| 데이터 | work_pool 3종 + test_final (봉인) |
| 학습 환경 | Google Colab Pro — **NVIDIA A100-SXM4-40GB** |
| 최적화 | **bf16 (bfloat16) autocast + BATCH_SIZE=64** |
| 공통 시드 | `random_state=42` (전 트랙 통일) |

### 분류 태스크

| 태스크 | 레이블 | 데이터 규모 | 비고 |
|--------|--------|------------|------|
| 이진 (binary) | 0=정상, 1=낚시성 | ~291,215건 | 49.9:50.1 균형 |
| 다중 (multi) | type_label 0~5 | ~40,106건 | Clickbait_Direct만, 7.1x 불균형 |

### 다중 분류 레이블 정의

| label | 유형 |
|-------|------|
| 0 | 의문유발-부호 |
| 1 | 의문유발-은닉 |
| 2 | 선정표현 |
| 3 | 속어/줄임말 |
| 4 | 사실과대 |
| 5 | 주어왜곡 |

---

## 2. 학습 모델

| MODEL_KEY | 모델 ID | 계열 | token_type_ids |
|-----------|---------|------|---------------|
| `kobert` | `skt/kobert-base-v1` | BERT | ❌ 미사용 (AutoTokenizer 미반환) |
| `klue` | `klue/roberta-base` | RoBERTa | ❌ 미사용 |
| `koelectra` | `monologg/koelectra-base-v3-discriminator` | ELECTRA | ✅ 사용 |

> **KoBERT token_type_ids 주의**: 초기 설정은 `MODEL_USE_TTI['kobert'] = True` 였으나,
> Pre-flight Check [11]에서 `skt/kobert-base-v1` AutoTokenizer가 token_type_ids를 반환하지 않음을 확인.
> `False`로 수정. HuggingFace 내부에서 zeros로 처리하므로 동작은 동일.

---

## 3. 노트북 실행 순서

```
[선택] bert_track.ipynb            ← 전처리 검증용 (학습과 독립 세션)
        ↓
bert_train_kobert.ipynb            ← KoBERT 5-Fold CV 학습
bert_train_klue.ipynb              ← KLUE-RoBERTa 5-Fold CV 학습
bert_train_koelectra.ipynb         ← KoELECTRA 5-Fold CV 학습
        ↓
bert_final_klue.ipynb              ← KLUE-RoBERTa 전체 데이터 재학습 (최고 모델)
        ↓
bert_test_eval.ipynb               ← test_final 최종 평가 (1회)
```

- 3개 CV 노트북은 각각 독립 세션, 모든 코드 자체 포함
- 실제 학습은 **순서대로 하나씩** 완료 후 다음 모델 진행 (병렬 실행 시 GPU 쿼터 소모 주의)
- `bert_track.ipynb`는 검증/탐색용이며 학습 노트북에 영향 없음

### 노트북 내 셀 실행 순서 (Step 0~11)

| Step | 내용 |
|------|------|
| 0 | 패키지 설치 (`pip install`) |
| 1 | Google Drive 마운트 + 경로 설정 |
| 2 | 라이브러리 임포트 |
| 3 | 데이터 로딩 + 중복 제거 |
| 4 | 설정값 (하이퍼파라미터) |
| 5 | ClickbaitDataset 정의 |
| 6 | 헬퍼 함수 정의 |
| 7 | train_epoch / eval_epoch 정의 |
| 8 | run_kfold 정의 |
| **9** | **Pre-flight Check** ← 학습 전 전체 점검 (11항목) |
| 10 | 이진 분류 학습 (`run_kfold(..., task='binary')`) |
| 11 | 다중 분류 학습 (`run_kfold(..., task='multi')`) |
| 12 | 결과 요약 출력 |

---

## 4. 하이퍼파라미터

| 파라미터 | 값 | 근거 |
|---------|-----|------|
| `BATCH_SIZE` | **64** | A100 40GB + bf16 기준. Drive I/O 병목 완화 목적으로 16→64 상향 |
| `MAX_LENGTH` | 512 | BERT 최대 입력 길이 |
| `LR` | 2e-5 | Devlin et al. 2019 권장 fine-tuning 학습률 |
| `WARMUP_RATIO` | 0.1 | 전체 스텝의 10% — 학습 초반 안정화 |
| `MAX_GRAD_NORM` | 1.0 | Gradient clipping (BERT fine-tuning 표준) |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay |
| `EPOCHS['binary']` | 3 | 291K건 대용량 — 3 epoch 충분 |
| `EPOCHS['multi']` | 5 | 40K건 소용량 — 더 많은 epoch 필요, early stopping 병행 |
| `PATIENCE` | 2 | 다중 분류 early stopping patience |
| `N_FOLDS` | 5 | 5-Fold Stratified K-Fold CV |
| `RANDOM_STATE` | 42 | 전 트랙 동일 시드 |

### bf16 적용 (A100 전용)

```python
with torch.autocast('cuda', dtype=torch.bfloat16):
    outputs = model(**kwargs)
    loss = criterion(outputs.logits, labels)
```

- A100은 bf16 하드웨어 네이티브 지원 → 약 2배 속도 향상
- fp32 대비 정밀도 손실 미미 (최종 F1 차이 < 0.1%)
- V100/T4/P100은 bf16 미지원 → 해당 GPU 사용 시 autocast 제거 후 fp32 학습

---

## 5. EDA 반영 설계 결정

### 5.1 입력 구성 — truncation='only_second' (EDA §4)
- **제목을 첫 번째 segment로**: `tokenizer(title, content)` 형태
- `truncation='only_second'`: 512 토큰 초과 시 본문(두 번째)만 자름, 제목은 절대 보존
- **근거**: EDA §4 — 낚시성 판단의 핵심은 제목. bert_track §Step5-1에서 제목 최대 토큰 36개 확인 → 제목 잘릴 위험 없음
- `truncation='only_second'`는 자연스럽게 **head truncation** 효과: 본문 앞부분(도입부)만 남음

### 5.2 이진 분류 — class_weight 없음 (EDA §1)
- 49.9:50.1로 거의 완전 균형 → `CrossEntropyLoss()` (가중치 없음)

### 5.3 다중 분류 — class_weight 필수 (EDA §2)
- 최대 7.1배 불균형 (의문유발-부호 가장 많음, 주어왜곡 가장 적음)
- `CrossEntropyLoss(weight=get_class_weights(train_labels, device))` 사용
- **train split 레이블로만 계산** — val set 포함 시 data leakage

### 5.4 다중 분류 early stopping (EDA §6)
- 40,106건으로 소용량 → epoch 증가 시 과적합 위험
- `PATIENCE=2`: val F1_macro 2회 연속 미개선 시 해당 fold 조기 종료
- **이진 분류에는 적용 안 함** (291K건 대용량, 3 epoch)

### 5.5 중복 제거 — K-Fold data leakage 방지 (EDA §7)
- 251건: newsID는 다르지만 title_clean + content_clean 동일
- `df.drop_duplicates(subset=['title_clean', 'content_clean'])` — K-Fold 분할 전에 처리

### 5.6 Stratified K-Fold (EDA §2)
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- 불균형 클래스에서 클래스 비율 유지가 critical

### 5.7 평가 지표 — F1-macro (EDA §2)
- accuracy는 불균형 클래스에서 소수 클래스 무시 위험
- **F1-macro**: 모든 클래스를 동등 취급 → 공정한 불균형 평가

### 5.8 혼동 행렬 핵심 관찰 지점 (EDA §9)
- 다중 분류에서 **의문유발-부호(0) ↔ 의문유발-은닉(1)** 오분류 집중 발생
- 물음표 등 명시적 부호 있음/없음 차이 → 표면적 feature 부재 시 혼동

### 5.9 매 fold마다 모델 재초기화
- pretrained checkpoint에서 매번 새로 `from_pretrained()` → fold 간 parameter leakage 방지
- fold 학습 후 `del model; torch.cuda.empty_cache()` → GPU 메모리 해제

---

## 6. Google Drive 경로 및 저장 파일

| 변수 | 경로 |
|------|------|
| `DATA_DIR` | `/content/drive/MyDrive/text-mining-2026/data/processed` |
| `SAVE_DIR` | `/content/drive/MyDrive/text-mining-2026/models` |

### 저장 파일 목록

```
models/
├── kobert_binary_fold1_best.pt ~ fold5_best.pt   ← KoBERT 이진 fold별 best
├── kobert_multi_fold1_best.pt  ~ fold5_best.pt   ← KoBERT 다중 fold별 best
├── klue_binary_fold1_best.pt   ~ fold5_best.pt   ← KLUE 이진 fold별 best
├── klue_multi_fold1_best.pt    ~ fold5_best.pt   ← KLUE 다중 fold별 best
├── koelectra_binary_fold1_best.pt ~ fold5_best.pt
├── koelectra_multi_fold1_best.pt  ~ fold5_best.pt
├── results_kobert_binary.json
├── results_kobert_multi.json
├── results_klue_binary.json
├── results_klue_multi.json
├── results_koelectra_binary.json
├── results_koelectra_multi.json
├── klue_binary_final.pt        ← KLUE 이진 전체 데이터 재학습 (최종 모델)
└── klue_multi_final.pt         ← KLUE 다중 전체 데이터 재학습 (최종 모델)
```

### 각 파일의 의미

#### `{model}_{task}_fold{N}_best.pt`
- fold N 학습 중 val F1_macro가 가장 높았던 epoch의 가중치 (best checkpoint)
- fold 내에서 더 좋은 epoch이 나오면 같은 경로에 덮어씀

#### `results_{model}_{task}.json`
- 5-Fold 전체의 평가 지표 요약본
- **이 파일이 존재하면 해당 태스크 완료로 간주** → 재실행 시 자동 스킵
- 구조: fold별 accuracy / f1_macro / precision_macro / recall_macro / loss

> **세션 종료 위험**: `.pt` 파일은 fold 단위로 즉시 저장되지만, **JSON은 5-fold 전부 끝나야 저장**.
> fold 4 완료 후 세션이 끊기면 JSON이 없어서 재실행 시 fold 1부터 다시 시작.

#### `klue_{task}_final.pt`
- work_pool 전체(100%)로 재학습한 최종 모델 가중치
- test_final 평가 및 실제 추론에 사용

---

## 7. Pre-flight Check (Step 9)

학습 시작 전 11개 항목을 자동 점검. 하나라도 ❌이면 `RuntimeError`로 중단.

| 항목 | 점검 내용 |
|------|---------|
| 1. GPU | 사용 가능 여부 + 메모리 ≥14GB |
| 2. 데이터 파일 | 3개 parquet 파일 존재 확인 |
| 3. SAVE_DIR 쓰기 | 임시 파일 생성으로 쓰기 권한 검증 |
| 4. 데이터 무결성 | 레코드 수 ~291,215건, 필수 컬럼, null/빈문자 0건 |
| 5. 이진 레이블 분포 | 낚시 비율 45~55% 범위 |
| 6. 다중 분류 데이터 | 레코드 수 ~40,106건, 6개 클래스 존재 확인 |
| 7. 토크나이저 | 모델 로드 + `truncation='only_second'` 동작 확인 |
| 8. ClickbaitDataset | binary/multi 인스턴스화 + shape=(512,), 레이블 범위 확인 |
| 9. DataLoader 배치 | shape=(4, 512) 배치 정상 반환 |
| 10. 클래스 가중치 | 6클래스 가중치 계산 + max/min 비율 확인 |
| 11. token_type_ids | MODEL_USE_TTI 설정과 토크나이저 반환값 일치 여부 |

---

## 8. 학습 중 발생한 이슈 및 해결

### 8.1 KoBERT token_type_ids 오류
- **증상**: Pre-flight Check [11] `❌ 불일치 — MODEL_USE_TTI=True, 토크나이저 반환=False`
- **원인**: `skt/kobert-base-v1` AutoTokenizer가 token_type_ids를 반환하지 않음
- **해결**: `MODEL_USE_TTI['kobert'] = False` 로 수정. HuggingFace 내부에서 zeros 처리하므로 동작 동일

### 8.2 Google Drive I/O 병목
- **증상**: A100에서 epoch당 ~79분 소요 (GPU 가동률 낮음)
- **원인**: Google Drive는 네트워크 스토리지 → 데이터 로딩 속도가 로컬 디스크 대비 느려 GPU 대기 발생
- **해결**: `BATCH_SIZE 16→64` + `bf16 autocast` 적용
  - batch 크기 증가 → GPU가 한 번에 더 많이 처리 → 상대적 I/O 대기 시간 감소
  - bf16 → 연산 속도 약 2배 향상 (A100 네이티브 지원)

### 8.3 세션 중단 (KoBERT binary fold 4/5)
- **증상**: fold 4 epoch 2/3 완료 후 Colab 세션 자동 종료
- **결과**: JSON 미저장 → fold 1부터 재학습 필요
- **교훈**: 화면 꺼짐 방지 + 세션 중 자리 비우지 않기

### 8.4 GPU 쿼터 소진
- **증상**: Colab Pro A100 쿼터 소진 (약 100 컴퓨팅 유닛 소모)
- **원인**: KoBERT 재학습 + 다른 모델 병렬 실행
- **해결**: 컴퓨팅 유닛 100개 추가 구매 후 계속 진행

---

## 9. 5-Fold CV 학습 결과

### KoBERT

![KoBERT 결과](<kobert 결과.png>)

| 태스크 | accuracy | f1_macro | precision_macro | recall_macro |
|--------|----------|----------|-----------------|--------------|
| binary | 0.8915 ± 0.0256 | 0.8915 ± 0.0256 | 0.8915 ± 0.0257 | 0.8915 ± 0.0256 |
| multi  | 0.7523 ± 0.0165 | 0.5847 ± 0.0287 | 0.5895 ± 0.0224 | 0.6013 ± 0.0212 |

### KLUE-RoBERTa

![KLUE-RoBERTa 결과](<KLUE-RoBERTa 결과.png>)

| 태스크 | accuracy | f1_macro | precision_macro | recall_macro |
|--------|----------|----------|-----------------|--------------|
| binary | 0.9869 ± 0.0002 | 0.9869 ± 0.0002 | 0.9869 ± 0.0002 | 0.9869 ± 0.0002 |
| multi  | 0.9228 ± 0.0027 | 0.8854 ± 0.0027 | 0.8908 ± 0.0019 | 0.8810 ± 0.0035 |

### KoELECTRA

![KoELECTRA 결과](<koelectra 결과.png>)

| 태스크 | accuracy | f1_macro | precision_macro | recall_macro |
|--------|----------|----------|-----------------|--------------|
| binary | 0.9869 ± 0.0003 | 0.9869 ± 0.0003 | 0.9869 ± 0.0003 | 0.9869 ± 0.0003 |
| multi  | 0.9195 ± 0.0033 | 0.8795 ± 0.0047 | 0.8831 ± 0.0035 | 0.8766 ± 0.0065 |

### 모델 비교 요약

| 모델 | Binary F1 | Multi F1 | 종합 |
|------|-----------|----------|------|
| KoBERT | 0.8915 | 0.5847 | 탈락 |
| KLUE-RoBERTa | **0.9869** | **0.8854** | **1위 → 최종 선정** |
| KoELECTRA | **0.9869** | 0.8795 | 2위 |

**최종 선정 모델: KLUE-RoBERTa (`klue/roberta-base`)**
- Binary: KoELECTRA와 동점 (0.9869)
- Multi: KoELECTRA 대비 +0.006p 우세
- KoBERT: 두 태스크 모두 크게 뒤처짐 (binary -9.5pp, multi -29.9pp), fold 간 분산도 높아 불안정

---

## 10. 최종 평가 (test_final)

### 전체 데이터 재학습 (`bert_final_klue.ipynb`)
- KLUE-RoBERTa를 work_pool 전체(~291,215건)로 재학습
- K-Fold 없음, 검증셋 없음 → 고정 epoch (binary=3, multi=5)
- 저장: `klue_binary_final.pt`, `klue_multi_final.pt`

### test_final 평가 (`bert_test_eval.ipynb`)

| 태스크 | 5-Fold CV F1 | test_final F1 | 차이 |
|--------|-------------|---------------|------|
| binary | 0.9869 | **0.9881** | +0.0012 |
| multi  | 0.8854 | **0.8960** | +0.0106 |

**과적합 없음 확인**: test F1 > CV F1 — 전체 데이터 재학습 효과 및 일반화 정상

---

## 11. 학습 주의사항

### Colab 세션 관리
- A100 기준 약 6.5 컴퓨팅 유닛/시간 소모
- 화면 꺼짐 방지 필수 (브라우저 콘솔 `setInterval(() => document.querySelector('#top-toolbar')?.click(), 60000)`)
- fold 중간 세션 종료 시 JSON 미저장 → fold 1부터 재학습

### GPU 메모리 (A100 기준)
- `BATCH_SIZE=64` + `bf16`: A100 40GB 정상 동작
- V100(16GB) 사용 시: `BATCH_SIZE=16`, bf16 대신 fp32 또는 fp16+GradScaler

### Resume 로직
- `results_{model_key}_{task}.json` 존재 시 자동 스킵
- `.pt` 파일은 fold 단위 즉시 저장, JSON은 전체 fold 완료 후 저장

### test_final 봉인 규칙
- `test_final.parquet`는 최종 평가 전까지 절대 로딩 금지
- `bert_test_eval.ipynb`에서만 로딩, 딱 1회 실행

---

## 12. 다음 단계

| 단계 | 내용 | 상태 |
|------|------|------|
| BERT 3모델 5-Fold CV | KoBERT / KLUE-RoBERTa / KoELECTRA | ✅ 완료 |
| 최고 모델 선정 | KLUE-RoBERTa | ✅ 완료 |
| 전체 데이터 재학습 | `bert_final_klue.ipynb` | ✅ 완료 |
| test_final 최종 평가 | Binary 0.9881 / Multi 0.8960 | ✅ 완료 |
| SHAP 해석 (BERT) | 100건 샘플 토큰별 기여도 분석 | 🔲 진행 예정 |
| LLM 교정 | 낚시성 제목 설명 + 교정안 생성 | 🔲 진행 예정 |
| Streamlit 데모 | 실시간 탐지 데모 앱 | 🔲 선택사항 |
