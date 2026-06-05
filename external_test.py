"""
외부 기사 과적합 테스트 — KLUE-RoBERTa 최종 모델
- klue_binary_final.pt : 이진 분류 (0=정상, 1=낚시성)
- klue_multi_final.pt  : 다중 분류 (0~5 유형)
- 입력 : 제목(title) + 본문(content) 쌍
- 실행 환경 : 로컬 CPU
"""

import os, sys, io, torch
import torch.nn.functional as F

# Windows cp949 환경에서 한글/특수문자 출력 보장
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── 경로 ──────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "klue/roberta-base"
BIN_PT     = os.path.join(BASE_DIR, "klue_binary_final.pt")
MULTI_PT   = os.path.join(BASE_DIR, "klue_multi_final.pt")
MAX_LEN    = 128          # 외부 테스트 20건 → 128로 충분, CPU 속도 고려

TYPE_NAMES = {
    0: "의문유발-부호",
    1: "의문유발-은닉",
    2: "선정표현",
    3: "속어/줄임말",
    4: "사실과대",
    5: "주어왜곡",
}

# ── 외부 기사 샘플 20건 ───────────────────────────────────────────────
# fmt: (제목, 본문 앞부분, 실제라벨, 실제유형(-1=정상))
SAMPLES = [
    # ── 의문유발-부호 ×2 ─────────────────────────────────────────────
    (
        "정부가 감추고 있는 진실은 무엇일까?",
        "정부가 최근 발표한 정책에 대해 각계에서 의문을 제기하고 있다. 전문가들은 "
        "구체적인 근거가 공개되지 않았다며 투명한 설명을 요구하고 있다.",
        1, 0,
    ),
    (
        "그는 왜 갑자기 자취를 감췄나?",
        "방송 활동을 중단한 한 연예인이 최근 SNS 계정을 비공개로 전환해 팬들 사이에서 "
        "여러 추측이 나오고 있다. 소속사는 개인 사정이라고만 밝혔다.",
        1, 0,
    ),
    # ── 의문유발-은닉 ×2 ─────────────────────────────────────────────
    (
        "'이것' 때문에 수천 명이 피해를 입었다",
        "최근 온라인 커뮤니티에서 특정 금융 상품 관련 피해 사례가 잇따라 접수되고 있다. "
        "금융감독원은 실태 파악에 착수했다고 밝혔다.",
        1, 1,
    ),
    (
        "그 사건의 내막은 이랬다…직접 들어보니",
        "지난달 발생한 사건과 관련해 당사자가 처음으로 입을 열었다. "
        "그는 당시 상황을 상세히 설명하며 억울함을 호소했다.",
        1, 1,
    ),
    # ── 선정표현 ×2 ──────────────────────────────────────────────────
    (
        "충격! 유명 배우 충격 근황…팬들 '경악'",
        "최근 온라인에 공개된 사진 한 장이 화제를 모으고 있다. "
        "해당 배우는 과거와 달라진 외모로 누리꾼들 사이에서 다양한 반응을 얻고 있다.",
        1, 2,
    ),
    (
        "피 튀기는 충돌…그날 밤 현장의 진실",
        "지난 주말 발생한 교통사고 현장을 목격한 시민들의 증언이 잇따르고 있다. "
        "경찰은 정확한 사고 경위를 조사 중이다.",
        1, 2,
    ),
    # ── 속어/줄임말 ×2 ───────────────────────────────────────────────
    (
        "완내스 등극한 신인 배우, 팬들 '난리남'",
        "데뷔 6개월 만에 각종 화제를 모은 신인 배우가 온라인 커뮤니티에서 '최고'라는 "
        "평가를 받으며 빠르게 인지도를 높이고 있다.",
        1, 3,
    ),
    (
        "요즘 MZ 사이 대박난 이 앱, 정체가 뭐길래",
        "최근 10~30대 이용자 사이에서 빠르게 확산 중인 한 모바일 앱이 "
        "출시 두 달 만에 다운로드 100만 건을 돌파했다.",
        1, 3,
    ),
    # ── 사실과대 ×2 ──────────────────────────────────────────────────
    (
        "한국 경제 완전히 무너진다…전문가들 긴급 경고",
        "일부 경제 연구소가 내년 성장률 전망치를 하향 조정했다. "
        "다만 대부분의 전문가는 구조적 위기보다는 단기 조정 국면으로 평가하고 있다.",
        1, 4,
    ),
    (
        "이 한 가지만 해도 다이어트 100% 성공 보장",
        "인터넷에서 유행 중인 특정 식이 요법이 화제다. "
        "전문의들은 개인 차이가 크므로 무분별한 따라하기는 자제해야 한다고 조언했다.",
        1, 4,
    ),
    # ── 주어왜곡 ×2 ──────────────────────────────────────────────────
    (
        "또 터졌다…이번엔 서울 한복판에서",
        "서울 도심의 한 상업 시설에서 소규모 화재가 발생했다. "
        "소방 당국은 20분 만에 초기 진화에 성공했으며 인명 피해는 없었다.",
        1, 5,
    ),
    (
        "결국 쓰러졌다, 수백만 명이 지켜보는 가운데",
        "생방송 도중 진행자가 갑자기 쓰러지는 사고가 발생했다. "
        "제작진은 즉시 방송을 중단하고 응급 처치를 실시했다.",
        1, 5,
    ),
    # ── 정상 기사 ×8 ─────────────────────────────────────────────────
    (
        "서울시, 내년 예산 45조 원 편성 확정",
        "서울시는 2025년 본예산을 45조 2천억 원으로 확정했다고 발표했다. "
        "복지·교통·환경 분야에 집중 투자할 계획이다.",
        0, -1,
    ),
    (
        "삼성전자, 3분기 영업이익 9조 2천억 원 기록",
        "삼성전자가 3분기 연결 영업이익 9조 2천억 원을 달성했다고 공시했다. "
        "반도체 부문 회복세가 실적을 견인했다는 분석이다.",
        0, -1,
    ),
    (
        "국민건강보험공단, 건강검진 대상자 확대 발표",
        "국민건강보험공단이 내년부터 일반건강검진 대상 연령을 기존 40세에서 30세로 "
        "낮추는 방안을 발표했다.",
        0, -1,
    ),
    (
        "기상청, 이번 주말 수도권 중심 비 예보",
        "기상청은 이번 주말 수도권을 중심으로 최대 50mm의 비가 내릴 것으로 예보했다. "
        "남부 지방은 맑은 날씨가 이어질 전망이다.",
        0, -1,
    ),
    (
        "교육부, 2025학년도 대학수학능력시험 일정 공고",
        "교육부는 2025학년도 수능을 11월 14일 시행한다고 공식 발표했다. "
        "원서 접수는 8월 22일부터 9월 6일까지다.",
        0, -1,
    ),
    (
        "한국은행, 기준금리 연 3.50% 동결 결정",
        "한국은행 금융통화위원회는 이달 기준금리를 연 3.50%로 동결했다. "
        "물가 안정세를 확인하면서 추후 방향성을 결정하겠다고 밝혔다.",
        0, -1,
    ),
    (
        "코스피, 외국인 순매수에 2,650선 마감",
        "코스피 지수가 외국인 투자자들의 순매수에 힘입어 전 거래일 대비 "
        "18.3포인트 오른 2,651.4로 마감했다.",
        0, -1,
    ),
    (
        "행정안전부, 전국 재난 안전 대피 훈련 실시",
        "행정안전부는 오는 15일 전국 동시 민방위 훈련을 실시한다고 밝혔다. "
        "오후 2시 사이렌 신호와 함께 15분간 진행된다.",
        0, -1,
    ),
]

# ── 추론 ──────────────────────────────────────────────────────────────
def load_model(pt_path, num_labels):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels
    )
    state = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def predict(model, tokenizer, titles, contents):
    preds, probs = [], []
    for title, content in zip(titles, contents):
        enc = tokenizer(
            text=title,
            text_pair=content,
            truncation="only_second",
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        prob = F.softmax(out.logits, dim=-1).squeeze()
        pred = int(torch.argmax(prob))
        preds.append(pred)
        probs.append(prob.tolist())
    return preds, probs


def main():
    print("KLUE-RoBERTa 베이스 모델 로딩 중…")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("이진 분류 모델 로딩 중…")
    bin_model = load_model(BIN_PT, num_labels=2)

    print("다중 분류 모델 로딩 중…")
    multi_model = load_model(MULTI_PT, num_labels=6)

    titles   = [s[0] for s in SAMPLES]
    contents = [s[1] for s in SAMPLES]
    true_bin = [s[2] for s in SAMPLES]
    true_typ = [s[3] for s in SAMPLES]

    print("\n이진 분류 추론 중…")
    bin_preds, bin_probs = predict(bin_model, tokenizer, titles, contents)

    print("다중 분류 추론 중…")
    multi_preds, multi_probs = predict(multi_model, tokenizer, titles, contents)

    # ── 결과 출력 ─────────────────────────────────────────────────────
    LABEL_KR = {0: "정상", 1: "낚시성"}
    correct_bin = sum(p == t for p, t in zip(bin_preds, true_bin))

    correct_multi = sum(
        p == t for p, t, tb in zip(multi_preds, true_typ, true_bin)
        if tb == 1
    )
    total_clickbait = sum(1 for tb in true_bin if tb == 1)

    print("\n" + "=" * 80)
    print("  외부 기사 과적합 테스트 — KLUE-RoBERTa 최종 모델")
    print("=" * 80)
    print(f"  {'No':>3}  {'제목':42}  {'실제':6}  {'예측':6}  {'확률':6}  {'유형예측'}")
    print(f"  {'-'*76}")

    for i, (s, bp, mp, bpr, mpr) in enumerate(
        zip(SAMPLES, bin_preds, multi_preds, bin_probs, multi_probs)
    ):
        title    = s[0][:40]
        true_b   = LABEL_KR[s[2]]
        pred_b   = LABEL_KR[bp]
        conf     = f"{max(bpr):.3f}"
        match    = "✓" if bp == s[2] else "✗"
        if bp == 1:
            type_pred = TYPE_NAMES[mp]
            true_type = TYPE_NAMES[s[3]] if s[3] != -1 else "-"
            type_match = "✓" if mp == s[3] else "✗"
            type_str  = f"{type_pred} ({type_match}, 실제:{true_type})"
        else:
            type_str  = "-"
        print(f"  {i+1:>3}  {title:<42}  {true_b:<6}  {pred_b:<6}  {conf:<6}  {type_str}")

    print(f"\n  {'-'*76}")
    print(f"  이진 분류 정확도 : {correct_bin}/{len(SAMPLES)} = {correct_bin/len(SAMPLES)*100:.1f}%")
    print(f"  유형 분류 정확도 : {correct_multi}/{total_clickbait} = {correct_multi/total_clickbait*100:.1f}%  (낚시성 {total_clickbait}건 중)")
    print("=" * 80)

    # ── TSV 저장 (Notion 붙여넣기용) ──────────────────────────────────
    tsv_path = os.path.join(BASE_DIR, "external_test_result.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("No\t제목\t실제\t예측\t이진확률\t유형예측\t유형실제\t유형정오\n")
        for i, (s, bp, mp, bpr, mpr) in enumerate(
            zip(SAMPLES, bin_preds, multi_preds, bin_probs, multi_probs)
        ):
            true_b    = LABEL_KR[s[2]]
            pred_b    = LABEL_KR[bp]
            conf      = f"{max(bpr):.3f}"
            type_pred = TYPE_NAMES[mp] if bp == 1 else "-"
            true_type = TYPE_NAMES[s[3]] if s[3] != -1 else "-"
            type_ok   = ("✓" if mp == s[3] else "✗") if bp == 1 else "-"
            f.write(f"{i+1}\t{s[0]}\t{true_b}\t{pred_b}\t{conf}\t{type_pred}\t{true_type}\t{type_ok}\n")
    print(f"\n  TSV 저장 완료 → {tsv_path}")


if __name__ == "__main__":
    main()
