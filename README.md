# text-mining-2026
텍스트 마이닝 수업 팀 프로젝트(2026)

## 모델 체크포인트

학습된 모델 파일(`.pt`)은 용량 문제로 GitHub에 포함되지 않습니다.

[📁 모델 파일 다운로드 (Google Drive)](https://drive.google.com/drive/folders/1yme7jjFOVF--ejcwQXG2NOQX6lEvsBL0?usp=drive_link)

| 모델 | 태스크 | 파일 |
|---|---|---|
| KLUE-RoBERTa | 이진/다중 분류 | `klue_binary_fold1~5_best.pt`, `klue_multi_fold1~5_best.pt`, `klue_binary_final.pt`, `klue_multi_final.pt` |
| KoBERT | 이진/다중 분류 | `kobert_binary_fold1~5_best.pt`, `kobert_multi_fold1~5_best.pt` |
| KoELECTRA | 이진/다중 분류 | `koelectra_binary_fold1~5_best.pt`, `koelectra_multi_fold1~5_best.pt` |

5-Fold 교차검증 수치 결과(`.json`)는 `models/` 폴더에 포함되어 있습니다.
