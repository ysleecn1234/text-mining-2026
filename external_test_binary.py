"""
외부 기사 이진 분류 — KLUE-RoBERTa 최종 모델 (max_length 128 vs 512 비교)
- klue_binary_final.pt : 이진 분류 (0=정상, 1=낚시성)
- 실제 수집 기사 11건 (2026년 5~6월)
- 동일 기사를 MAX_LEN=128 / MAX_LEN=512 로 각각 추론하여 결과 비교
- 실행 환경 : 로컬 CPU
"""

import os, sys, io, torch
import torch.nn.functional as F

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "klue/roberta-base"
BIN_PT     = os.path.join(BASE_DIR, "klue_binary_final.pt")
MAX_LENS   = [128, 512]   # 비교할 max_length 목록

# ── 실제 수집 기사 11건 ────────────────────────────────────────────────
# (제목, 본문 앞부분, 출처)  ← 정답 라벨 없음, 모델이 직접 판단
SAMPLES = [
    (
        '"하루 3차례, 비용도 싸다"…예약 폭발한 결혼식 뭐길래',
        "요즘은 남들과 똑같은 결혼식이 아니라 전통혼례를 찾는 예비부부들이 늘고 있다고 합니다. "
        "전통혼례라니 좀 특이하죠. 그런데, 뿐만 아니라 한복, 전통 소품을 활용한 'K-브라이덜 샤워'까지 등장했습니다. "
        "이들은 전통을 단순히 재현하는 데 그치지 않고, 부부와 하객 모두가 함께 즐기고 기억할 수 있는 경험으로 재해석하고 있었습니다.",
        "SBS 뉴스",
    ),
    (
        '"젠슨 황 깐부들도 파랗게 질렸는데"…상한가 찍은 종목 뭐길래',
        "젠슨 황 엔비디아 최고경영자(CEO)의 방한을 하루 앞둔 4일 LG전자·네이버 등 관련 수혜주로 주목받아온 종목들이 "
        "차익실현 매물 등장에 일제히 급락했다. 여기에 반도체 설계 업체인 브로드컴이 시장 기대치를 밑돈 매출 전망치를 발표하자 "
        "삼성전자, SK하이닉스 주가가 일제히 약세를 보였다. 이날 코스피는 전거래일 대비 1.84% 내린 8639.41로 장을 마쳤다.",
        "중앙일보",
    ),
    (
        "태안 앞바다서 체포 된 중국인 남성, 알고보니…'충격 정체'",
        "중국 반체제 인사로 알려진 인권운동가 둥광핑(68)이 고무보트를 타고 서해를 건너 한국 영해로 들어왔다가 "
        "충남 태안 앞바다에서 해경에 붙잡혔다. 둥광핑의 구속 여부는 이르면 28일 결정된다. "
        "뉴욕타임스(NYT)와 CNN 등 외신은 26일 둥광핑이 소형 고무보트를 타고 한국 영해에 진입했다가 해경에 체포됐다고 보도했다.",
        "한국경제",
    ),
    (
        "사망한 남편, 알고 보니 일본서 두 집 살림…상간녀에 위자료 청구 가능할까",
        "남편이 사망한 이후에 부정행위 사실을 알게 되었더라도 상간녀를 상대로 한 위자료 청구 소송이 가능하다는 법조계 조언이 나왔다. "
        "A씨의 남편은 무역 법인의 중역으로 일본 출장이 잦았다. 지난해 남편이 갑작스러운 심장마비로 세상을 떠나면서 "
        "유품을 정리하던 중 남편의 일본 휴대전화에서 낯선 여성, 아이들과 함께 찍은 사진 및 장기간 생활비를 송금한 내역을 발견했다.",
        "뉴시스",
    ),
    (
        "'애둘맘' 김보미, 수술 후 병원 갔다가 결국 터졌다…\"간호사 왜 이렇게 불친절\"",
        "배우 김보미가 요로결석 수술 이후 다시 찾은 병원에서 속상했던 심경을 털어놨다. "
        "김보미는 26일 자신의 SNS에 병원 검사실 앞 사진과 함께 "
        "\"수술한데 아파서 병원왔는데.. 창구에 있는 간호사 쌤들은 왜이렇게 불친절할까.. 아닌분들도 계시지만.. 하…정말…\"이라는 글을 남겼다.",
        "MK스포츠",
    ),
    (
        "'가지 부부' 아내 결국 터졌다, 남편 향해 \"입 다물어\" (이혼숙려캠프)",
        "JTBC '이혼숙려캠프' 21기 부부들의 최종 조정 과정이 공개된다. "
        "이 과정에서 '가지 남편'은 가족보다 가지를 우선으로 생각한다는 검사 결과가 공개돼 충격을 안긴다. "
        "이어진 변호사 상담에서 '가지 남편'이 부부 관계 도중 게임을 한 행동이 명백한 유책 사유라는 사실을 알게 된다.",
        "동아닷컴",
    ),
    (
        "3% 훌쩍 넘어간 예금금리…증시 활황에도 은행 예치자금 증가",
        "최근 시장금리 상승을 반영한 은행 예금금리가 우상향 곡선을 그리면서 연 3%를 웃돌고 있다. "
        "한국은행이 하반기 기준금리 인상을 예고하면서 앞으로 수신금리도 4%를 넘어설 수 있다는 전망이 나온다. "
        "19개 은행의 1년 만기 정기예금 중 절반이 넘는 상품이 우대금리 포함 최고 3.0% 이상을 제공하는 것으로 집계됐다.",
        "뉴시스",
    ),
    (
        "[팩트체크] 이번 여름 한 달 내내 비온다고?…매년 반복되는 장마 예보 정체는",
        "올해 6월 또는 7월에 한 달 내내 비가 내릴 것이란 내용의 게시물이 인터넷과 SNS에서 확산하고 있다. "
        "기상청이 지난달 공식 발표가 아니라며 한 차례 진화에 나섰지만, 여전히 비슷한 내용의 허위 정보가 횡행한다. "
        "기상청은 기후변화 등의 이유로 2009년 이후 공식 장마 전망을 내놓지 않고 있다며 가짜 뉴스에 주의할 것을 당부했다.",
        "연합뉴스",
    ),
    (
        "AI 행정시대, 열린정부 논의…행안부, OECD 국제포럼 개최",
        "행정안전부가 OECD와 해외 정부, 시민사회가 참여하는 국제행사를 열고 "
        "인공지능(AI) 시대 공공거버넌스와 열린정부 방향을 논의한다. "
        "행안부는 22일 서울에서 'OECD 열린정부 국제심포지엄'과 '세계열린정부주간 민관합동 국제포럼'을 개최한다고 밝혔다.",
        "뉴스1",
    ),
    (
        "코스피 급락에 매도 사이드카…삼성전자·SK하이닉스 약세",
        "코스피가 5일 급락하면서 프로그램매도호가 일시효력정지인 매도 사이드카가 발동됐다. "
        "한국거래소에 따르면 이날 오전 9시 8분 25초께 코스피200선물지수의 변동으로 5분간 프로그램매도호가의 효력이 정지됐다. "
        "발동 시점 당시 코스피200선물지수는 전일 종가보다 71.84포인트(5.20%) 하락한 1309.56이었다.",
        "경기일보",
    ),
    (
        "5월 수출액 877.5억달러 역대 최대…반도체 호황 덕분",
        "5월 수출액이 877억5000만달러로 집계돼 월 기준 역대 최대 기록을 경신했다. "
        "사상 처음으로 월 수출액이 3개월 연속 800억달러를 넘어서는 기록도 썼다. "
        "인공지능(AI) 투자 확대에 따라 반도체 호황이 이어진 덕분이다. "
        "산업통상부·관세청은 1일 지난달 수출액이 1년 전보다 53.2% 증가했다고 밝혔다.",
        "조선일보",
    ),
]

# ── 추론 ──────────────────────────────────────────────────────────────
def run_inference(model, tokenizer, max_len):
    """주어진 max_len으로 SAMPLES 전체 추론 후 결과 리스트 반환"""
    LABEL = {0: "정상", 1: "낚시성"}
    results = []
    for title, content, source in SAMPLES:
        enc = tokenizer(
            text=title,
            text_pair=content,
            truncation="only_second",
            max_length=max_len,
            padding="max_length",
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        prob = F.softmax(out.logits, dim=-1).squeeze()
        pred = int(torch.argmax(prob))
        results.append({
            "title":  title,
            "source": source,
            "pred":   LABEL[pred],
            "p_n":    float(prob[0]),
            "p_c":    float(prob[1]),
        })
    return results


def print_results(results, max_len):
    LABEL_CB = sum(1 for r in results if r["pred"] == "낚시성")
    print(f"\n{'='*85}")
    print(f"  [MAX_LEN = {max_len}]  외부 기사 이진 분류 예측 결과")
    print(f"{'='*85}")
    print(f"  {'No':>3}  {'제목':40}  {'출처':8}  {'판정':6}  {'정상%':7}  {'낚시%':7}")
    print(f"  {'-'*81}")
    for i, r in enumerate(results):
        print(f"  {i+1:>3}  {r['title'][:40]:<40}  {r['source']:<8}  {r['pred']:<6}  {r['p_n']*100:6.2f}%  {r['p_c']*100:6.2f}%")
    print(f"  {'-'*81}")
    print(f"  낚시성 판정: {LABEL_CB}건 / 정상 판정: {len(results)-LABEL_CB}건")


def print_diff(res128, res512):
    """두 설정 간 판정이 달라진 기사 출력"""
    diffs = [(i, r128, r512) for i, (r128, r512) in enumerate(zip(res128, res512))
             if r128["pred"] != r512["pred"]]
    print(f"\n{'='*85}")
    print(f"  [판정 변경 기사] 128 vs 512 비교  —  총 {len(diffs)}건 변경")
    print(f"{'='*85}")
    if not diffs:
        print("  판정이 변경된 기사 없음 (두 설정 결과 동일)")
    for i, r128, r512 in diffs:
        print(f"  No.{i+1}  {r128['title'][:50]}")
        print(f"        128 → {r128['pred']}  (낚시 {r128['p_c']*100:.2f}%)")
        print(f"        512 → {r512['pred']}  (낚시 {r512['p_c']*100:.2f}%)")


def main():
    print("KLUE-RoBERTa 베이스 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("이진 분류 모델 로딩 중...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model.load_state_dict(torch.load(BIN_PT, map_location="cpu"))
    model.eval()
    print("로딩 완료. 추론 시작...\n")

    all_results = {}
    for max_len in MAX_LENS:
        print(f"  MAX_LEN={max_len} 추론 중...")
        all_results[max_len] = run_inference(model, tokenizer, max_len)

    # 각 설정별 결과 출력
    for max_len in MAX_LENS:
        print_results(all_results[max_len], max_len)

    # 두 설정 비교
    print_diff(all_results[128], all_results[512])

    # TSV 저장 (비교 포함)
    tsv_path = os.path.join(BASE_DIR, "external_test_binary_result.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("No\t제목\t출처\t판정_128\t정상%_128\t낚시%_128\t판정_512\t정상%_512\t낚시%_512\t판정변경\n")
        for i, (r128, r512) in enumerate(zip(all_results[128], all_results[512])):
            changed = "변경" if r128["pred"] != r512["pred"] else "-"
            f.write(
                f"{i+1}\t{r128['title']}\t{r128['source']}\t"
                f"{r128['pred']}\t{r128['p_n']*100:.2f}%\t{r128['p_c']*100:.2f}%\t"
                f"{r512['pred']}\t{r512['p_n']*100:.2f}%\t{r512['p_c']*100:.2f}%\t"
                f"{changed}\n"
            )
    print(f"\n  TSV 저장 완료 -> {tsv_path}")

if __name__ == "__main__":
    main()
