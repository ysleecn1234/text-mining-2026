"""
AI Hub 146 낚시성 기사 탐지 데이터 1세부 — 압축 해제 스크립트
────────────────────────────────────────────────────────────
목적:
  84개 zip을 work_pool/test_final × 3클래스 폴더로 풀어냄.
  - work_pool = AI Hub "Training" (학습+검증용, K-Fold로 내부 분할)
  - test_final = AI Hub "Validation" (최종 평가 전용, 봉인) 라벨(TL/VL) zip만 해제하여 _L.json을 클래스별 폴더에 저장.
  원천(TS/VS) zip은 해제하지 않음 (_L.json에 원천 정보가 이미 포함됨). 원본 zip은 절대 건드리지 않음.

사용법:
  python extract_dataset.py          # dry-run (예상만 출력, 실제 해제 X)
  python extract_dataset.py --run    # 실제 해제

설계 원칙:
  - 실행 전: 각 zip의 JSON 개수, 총 용량 예측 출력
  - 실행 중: zip 단위 진행률 출력
  - 실행 후: 각 폴더 파일 수·용량, 샘플 파일 3개 이름 출력
  - 에러 발생 시 즉시 중단 후 원인 출력
"""
import os
import sys
import time
import zipfile
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────
# 경로 설정
#  NFD(macOS) 유니코드 문제 대응: os.listdir()로 동적 해석.
#  하드코딩된 경로 문자열을 피하고 실제 존재하는 이름을 사용.
# ─────────────────────────────────────────────────────────
MNT = '/sessions/ecstatic-keen-fermat/mnt'

def _find_project_dir():
    """'텍스트마이닝_프로젝트' 폴더를 listdir로 찾아 실제 NFD 이름 반환."""
    for e in os.listdir(MNT):
        if '_' in e and all(x not in e for x in ['cowork','claude','memory','upload']):
            return os.path.join(MNT, e)
    raise FileNotFoundError('프로젝트 폴더를 찾지 못함')

PROJ = _find_project_dir()
# 데이터셋 루트(146.낚시성 기사 탐지 데이터)
DS_ROOT = os.path.join(
    PROJ,
    [e for e in os.listdir(PROJ) if e.startswith('146')][0]
)
# 출력 위치: 프로젝트폴더/data/raw/
OUT_ROOT = os.path.join(PROJ, 'data', 'raw')

# ─────────────────────────────────────────────────────────
# zip 파일명 → (split, class) 매핑
#   예) TL_Part1_Clickbait_Direct_PO.zip → ('work_pool', 'clickbait_direct')
#       VS_Part1_NonClickbait_Auto_IS.zip → ('test_final', 'nonclickbait_auto')
# ─────────────────────────────────────────────────────────
def parse_zipname(name):
    # 앞 2글자: T/V(split) + S/L(source/label)
    prefix = name[:2]
    if prefix[0] == 'T':
        split = 'work_pool'    # AI Hub Training → 학습+검증용
    elif prefix[0] == 'V':
        split = 'test_final'   # AI Hub Validation → 최종 평가 전용 (봉인)
    else:
        raise ValueError(f'알 수 없는 prefix: {name}')
    # 클래스: 더 구체적인 패턴을 먼저 매칭 (NonClickbait 먼저)
    if 'NonClickbait_Auto' in name:
        cls = 'nonclickbait_auto'
    elif 'Clickbait_Direct' in name:
        cls = 'clickbait_direct'
    elif 'Clickbait_Auto' in name:
        cls = 'clickbait_auto'
    else:
        raise ValueError(f'알 수 없는 클래스: {name}')
    return split, cls


# ─────────────────────────────────────────────────────────
# 1단계: 데이터셋 루트에서 모든 zip 수집
# ─────────────────────────────────────────────────────────
all_zips = []
for root, _, files in os.walk(DS_ROOT):
    for f in files:
        if f.endswith('.zip'):
            all_zips.append(os.path.join(root, f))
all_zips.sort()

# 라벨 zip만 필터링 (TL=Training Label, VL=Validation Label)
# _L.json에 원천 정보(제목, 본문)가 이미 포함되어 있으므로 원천 zip(TS/VS)은 불필요
label_zips = [z for z in all_zips if os.path.basename(z)[:2] in ('TL', 'VL')]

print('=' * 70)
print(f'데이터셋 루트: {DS_ROOT}')
print(f'출력 루트    : {OUT_ROOT}')
print(f'전체 zip     : {len(all_zips)}개 (원천 42 + 라벨 42)')
print(f'해제 대상    : {len(label_zips)}개 (라벨 zip만)')
print('=' * 70)

# 라벨 zip이 42개가 아니면 즉시 중단 (원칙 6: 예상과 다르면 멈춘다)
assert len(label_zips) == 42, f'라벨 zip 개수 예상(42)과 다름: {len(label_zips)}'

# ─────────────────────────────────────────────────────────
# 2단계: 실행 전 점검 — 각 zip의 JSON 개수·해제 후 용량 예측
# ─────────────────────────────────────────────────────────
print('\n[사전 점검] 각 zip의 JSON 개수 및 해제 후 예상 용량 확인 중...')
plan = []                   # (zip경로, split, class, json개수, 해제후용량)
total_json = 0
total_uncompressed = 0
for zp in label_zips:
    with zipfile.ZipFile(zp) as z:
        # JSON 파일만 카운트
        infos = [i for i in z.infolist() if i.filename.lower().endswith('.json')]
        n = len(infos)
        sz = sum(i.file_size for i in infos)  # 해제 후 크기
    split, cls = parse_zipname(os.path.basename(zp))
    plan.append((zp, split, cls, n, sz))
    total_json += n
    total_uncompressed += sz

# split × class 집계 출력
agg = defaultdict(lambda: [0, 0])
for _, split, cls, n, sz in plan:
    agg[(split, cls)][0] += n
    agg[(split, cls)][1] += sz

print('\n[split × class 집계]')
print(f'{"폴더":<35} {"파일수":>10} {"용량":>12}')
print('-' * 60)
for (split, cls), (n, sz) in sorted(agg.items()):
    folder = f'{split}/{cls}'
    print(f'{folder:<35} {n:>10,}  {sz/1024/1024:>9.1f} MB')
print('-' * 60)
print(f'{"합계":<35} {total_json:>10,}  {total_uncompressed/1024/1024:>9.1f} MB')
print(f'{"":<35} {"":<10}  ({total_uncompressed/1024/1024/1024:.2f} GB)')

# ─────────────────────────────────────────────────────────
# 3단계: 실제 해제 (--run 플래그가 있을 때만)
# ─────────────────────────────────────────────────────────
if '--run' not in sys.argv:
    print('\n⚠️  DRY-RUN 모드: 실제 해제는 수행하지 않았습니다.')
    print('실제 해제하려면: python extract_dataset.py --run')
    sys.exit(0)

print('\n' + '=' * 70)
print('[실제 해제 시작]')
print('=' * 70)
Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
t0 = time.time()

for i, (zp, split, cls, n, _) in enumerate(plan, 1):
    out_dir = os.path.join(OUT_ROOT, split, cls)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = os.path.basename(zp)
    print(f'[{i:>2}/{len(plan)}] {name:<48} → {split}/{cls:<20} ({n:>6,}개)')

    # zip 내부를 순회하며 JSON만 추출
    with zipfile.ZipFile(zp) as z:
        for info in z.infolist():
            if not info.filename.lower().endswith('.json'):
                continue
            # zip 내부 경로 앞 '/' 및 중첩 폴더 방지: 기본명만 사용
            basename = os.path.basename(info.filename)
            target = os.path.join(out_dir, basename)
            # 원본 zip 불변: 읽기 전용으로만 열고 새 파일에 내용 기록
            with z.open(info) as src, open(target, 'wb') as dst:
                dst.write(src.read())

elapsed = time.time() - t0
print(f'\n해제 완료: {elapsed:.1f}초 ({elapsed/60:.1f}분)')

# ─────────────────────────────────────────────────────────
# 4단계: 검증 — 해제 결과 폴더별 파일 수·용량·샘플 3개 출력
# ─────────────────────────────────────────────────────────
print('\n' + '=' * 70)
print('[검증] 해제 결과')
print('=' * 70)
total_files = 0
total_size = 0
for split in ['work_pool', 'test_final']:
    for cls in ['clickbait_auto', 'clickbait_direct', 'nonclickbait_auto']:
        d = os.path.join(OUT_ROOT, split, cls)
        if not os.path.isdir(d):
            print(f'  {split}/{cls}: (폴더 없음)')
            continue
        files = os.listdir(d)
        sz = sum(os.path.getsize(os.path.join(d, f)) for f in files)
        total_files += len(files)
        total_size += sz
        print(f'\n  📂 {split}/{cls}')
        print(f'     파일수: {len(files):,}개   용량: {sz/1024/1024:.1f} MB')
        for f in sorted(files)[:3]:
            print(f'     샘플: {f}')

print('\n' + '=' * 70)
print(f'총 {total_files:,}개 파일, {total_size/1024/1024:.1f} MB '
      f'({total_size/1024/1024/1024:.2f} GB)')
print('=' * 70)

# 사전 예상치와 비교 (원칙 5: 변형 전후 비교)
if total_files != total_json:
    print(f'⚠️  경고: 해제 파일 수({total_files:,}) ≠ 예상({total_json:,})')
else:
    print(f'✅ 파일 수 검증: 예상과 일치 ({total_files:,}개)')
