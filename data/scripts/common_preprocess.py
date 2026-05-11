"""
공통 전처리: JSON → DataFrame 변환 + 공통 정제/정규화 + parquet 저장
─────────────────────────────────────────────────────────────────────
목적:
  1) work_pool/test_final의 _L.json 327,900개를 단일 DataFrame으로 변환
  2) 공통 전처리 적용 (HTML/URL 제거, 반복 구두점 정리, 띄어쓰기 교정)
  3) 공통 전처리 완료된 DataFrame을 parquet으로 저장

※ EDA는 eda_work_pool.ipynb에서 별도 수행

산출물:
  data/processed/common_preprocessed.parquet
  — 이후 TF-IDF 트랙, BERT 트랙에서 각각 이어서 전처리

실행:
  python common_preprocess.py
"""

import os
import sys
import json
import time
import re
import pandas as pd

# ─────────────────────────────────────────────────────────
# 경로 설정 (NFD 유니코드 대응)
# ─────────────────────────────────────────────────────────
MNT = '/sessions/ecstatic-keen-fermat/mnt'

def _find_project_dir():
    for e in os.listdir(MNT):
        if '_' in e and all(x not in e for x in ['cowork','claude','memory','upload']):
            return os.path.join(MNT, e)
    raise FileNotFoundError('프로젝트 폴더를 찾지 못함')

PROJ = _find_project_dir()
RAW_ROOT = os.path.join(PROJ, 'data', 'raw')
OUT_DIR = os.path.join(PROJ, 'data', 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────
# Step 1: JSON → DataFrame 변환
# ─────────────────────────────────────────────────────────
print('=' * 70)
print('[Step 1] JSON → DataFrame 변환')
print('=' * 70)

rows = []
errors = []
t0 = time.time()

for split in ['work_pool', 'test_final']:
    split_dir = os.path.join(RAW_ROOT, split)
    if not os.path.isdir(split_dir):
        print(f'  {split} 폴더 없음, 건너뜀')
        continue

    for cls in ['clickbait_auto', 'clickbait_direct', 'nonclickbait_auto']:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f'  {split}/{cls} 폴더 없음, 건너뜀')
            continue

        files = [f for f in os.listdir(cls_dir) if f.endswith('.json')]
        total = len(files)
        print(f'  {split}/{cls}: {total:,}개 로딩 중...', end='', flush=True)

        for i, fname in enumerate(files):
            filepath = os.path.join(cls_dir, fname)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                src = data.get('sourceDataInfo', {})
                lbl = data.get('labeledDataInfo', {})

                # 이진 라벨: clickbaitClass 반전 (0=낚시→1, 1=정상→0)
                raw_class = lbl.get('clickbaitClass', None)
                binary_label = 1 - int(raw_class) if raw_class is not None else None

                # 다중 라벨: processPattern (Direct만 11~16, 나머지는 -1)
                raw_pattern = str(src.get('processPattern', ''))
                if raw_pattern in ('11','12','13','14','15','16'):
                    type_label = int(raw_pattern) - 11  # 0~5로 매핑
                else:
                    type_label = -1  # Auto(99), NonCB(00) 등

                rows.append({
                    'newsID': src.get('newsID', ''),
                    'newTitle': lbl.get('newTitle', ''),
                    'newsContent': src.get('newsContent', ''),
                    'binary_label': binary_label,       # 낚시=1, 정상=0
                    'type_label': type_label,            # 0~5 (Direct만), -1 (나머지)
                    'newsCategory': src.get('newsCategory', ''),
                    'source_class': cls,
                    'split': split,
                })

            except Exception as e:
                errors.append((filepath, str(e)))

            if (i + 1) % max(1, total // 10) == 0:
                pct = (i + 1) / total * 100
                print(f' {pct:.0f}%', end='', flush=True)

        print(' done')

elapsed = time.time() - t0
print(f'\n로딩 완료: {len(rows):,}건, {elapsed:.1f}초 ({elapsed/60:.1f}분)')
if errors:
    print(f'에러 {len(errors)}건:')
    for fp, msg in errors[:5]:
        print(f'   {fp}: {msg}')

df = pd.DataFrame(rows)

# ─────────────────────────────────────────────────────────
# Step 2: 공통 정제 (Cleaning)
# ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('[Step 2] 공통 정제 (Cleaning)')
print('=' * 70)

def clean_text(text):
    """공통 정제: HTML 태그, URL, 이메일, 반복 구두점 처리"""
    if not isinstance(text, str) or text == '':
        return text

    # 1) HTML 태그 제거
    text = re.sub(r'<[^>]+>', ' ', text)

    # 2) URL 제거
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # 3) 이메일 제거
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 4) 반복 구두점 정리: ???→?, !!!→!, ...은 유지 (낚시성 신호)
    text = re.sub(r'\?{2,}', '?', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'~{2,}', '~', text)

    # 5) 연속 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# 정제 전 샘플
print('\n[정제 전 샘플]')
sample_idx = df[df['source_class'] == 'clickbait_direct'].index[:3]
for idx in sample_idx:
    print(f'  제목: {df.loc[idx, "newTitle"][:80]}')

# 정제 적용
print('\n제목 정제 중...', end='', flush=True)
t1 = time.time()
df['title_clean'] = df['newTitle'].apply(clean_text)
print(f' {time.time()-t1:.1f}초')

print('본문 정제 중...', end='', flush=True)
t2 = time.time()
df['content_clean'] = df['newsContent'].apply(clean_text)
print(f' {time.time()-t2:.1f}초')

# 정제 후 샘플
print('\n[정제 후 샘플]')
for idx in sample_idx:
    print(f'  제목: {df.loc[idx, "title_clean"][:80]}')

title_empty_after = (df['title_clean'] == '').sum()
body_empty_after = (df['content_clean'] == '').sum()
print(f'\n정제 후 빈 제목: {title_empty_after}건 / 빈 본문: {body_empty_after}건')

# ─────────────────────────────────────────────────────────
# Step 3: 띄어쓰기 교정 (선택)
# ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('[Step 3] 띄어쓰기 교정 (pykospacing)')
print('=' * 70)

try:
    from pykospacing import Spacing
    spacing = Spacing()
    HAS_SPACING = True
    print('  pykospacing 로드 성공')
except ImportError:
    HAS_SPACING = False
    print('  pykospacing 미설치 — 띄어쓰기 교정 건너뜀')
    print('  -> 설치: pip install pykospacing')

if HAS_SPACING:
    print('  본문 띄어쓰기 교정 중...')
    t3 = time.time()
    df['content_clean'] = df['content_clean'].apply(
        lambda x: spacing(x) if isinstance(x, str) and len(x) > 0 else x
    )
    print(f'  완료: {time.time()-t3:.1f}초')
else:
    print('  -> title_clean, content_clean은 정제까지만 적용된 상태로 저장')

# ─────────────────────────────────────────────────────────
# Step 4: parquet 저장
# ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('[Step 4] 저장')
print('=' * 70)

out_path = os.path.join(OUT_DIR, 'common_preprocessed.parquet')
df.to_parquet(out_path, index=False, engine='pyarrow')
print(f'  저장: {out_path}')
print(f'  크기: {os.path.getsize(out_path)/1024/1024:.1f} MB')
print(f'  행수: {len(df):,}건')
print(f'  컬럼: {list(df.columns)}')

# 최종 요약
print('\n' + '=' * 70)
print('[최종 요약]')
print('=' * 70)
print(f'  전체 건수: {len(df):,}')
print(f'  work_pool: {len(df[df["split"]=="work_pool"]):,}')
print(f'  test_final: {len(df[df["split"]=="test_final"]):,}')
print(f'  이진 라벨 — 낚시(1): {(df["binary_label"]==1).sum():,} / 정상(0): {(df["binary_label"]==0).sum():,}')
print(f'  다중 라벨 — Direct만: {(df["type_label"]>=0).sum():,}건')
print(f'  저장 컬럼:')
print(f'    - newsID, newTitle, newsContent: 원본')
print(f'    - title_clean, content_clean: 공통 전처리 적용')
print(f'    - binary_label, type_label: 라벨')
print(f'    - newsCategory, source_class, split: 메타')
print(f'\n공통 전처리 완료. 이후 트랙별 전처리는 이 parquet을 읽어서 진행.')
