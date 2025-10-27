import os
import time
from typing import Dict, Any, List

from flask import Flask, request, jsonify, Response
import requests
from requests_html import HTMLSession
from bs4 import BeautifulSoup

app = Flask(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
LIST_ENDPOINT = "https://portal.scourt.go.kr/pgp/pgp1011/selectJdcpctSrchRsltLst.on"

HEADERS_JSON = {
    "Content-Type": "application/json; charset=UTF-8",
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://portal.scourt.go.kr",
    "Referer": "https://portal.scourt.go.kr/pgp/pgp1011/jdcpctSrchPage.do",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"
}

UA = HEADERS_JSON["User-Agent"]

# 기본 렌더러: requests_html
RENDERER = os.getenv("RENDERER", "requests_html")  # "requests_html" | "playwright"

# ──────────────────────────────────────────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────────────────────────────────────────
def normalize_items(raw_json: Dict[str, Any]) -> List[Dict[str, str]]:
    """목록 JSON에서 Spring이 쓰기 쉬운 평평한 리스트로 변환"""
    out = []
    items = (raw_json.get("data", {}) or {}).get("dlt_jdcpctRslt", []) or []
    for n in items:
        out.append({
            "case_no": n.get("csNoLstCtt", ""),
            "court": n.get("cortNm", ""),
            "judgment_date": n.get("prnjdgYmd", ""),
            "summary": (n.get("jdcpctSumrCtt", "") or "").strip(),
            "srno": n.get("jisCntntsSrno", ""),
            "inst": n.get("jisJdcpcInstnDvsCd", ""),
        })
    return out


def extract_main_html(html: str) -> str:
    """
    상세 페이지의 본문만 최대한 뽑아내기.
    포털 구조가 바뀔 수 있으므로 후보 선택자를 여러 개 시도.
    """
    if not html:
        return ""

    soup = BeautifulSoup(html, "lxml")

    # 후보들(길이가 충분한 첫 번째 매칭 사용)
    candidates = [
        "#vfm_pgpDtlMain",
        "div.cntntsArea",
        "div#content",
        "div.mainFrame",
        "div#wrap",
        "body",
    ]
    for sel in candidates:
        node = soup.select_one(sel)
        if node and len(str(node)) > 800:
            return str(node)

    return str(soup.body) if soup.body else html


def requests_html_render(url: str, timeout: int = 35, sleep: float = 1.5) -> str:
    """
    requests_html 로 JS 렌더링하여 HTML 반환
    """
    session = HTMLSession()
    # 세션/쿠키 준비용 워밍업
    session.get(
        "https://portal.scourt.go.kr/pgp/main.on?w2xPath=PGP1011M04",
        headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"},
        timeout=20,
    )
    # 실제 상세
    r = session.get(url, headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"}, timeout=timeout)
    r.html.render(timeout=timeout, sleep=sleep)  # headless chromium 실행
    return r.html.html or ""


def playwright_render(url: str, timeout_ms: int = 35000) -> str:
    """
    선택: Playwright 로 렌더링 (pip install playwright && playwright install 필요)
    """
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=UA, locale="ko-KR")
        page = ctx.new_page()
        page.set_default_timeout(timeout_ms)
        page.goto("https://portal.scourt.go.kr/pgp/main.on?w2xPath=PGP1011M04")
        # 상세 진입
        page.goto(url)
        # 네트워크 idle 대기
        page.wait_for_load_state("networkidle")
        html = page.content()
        browser.close()
        return html


def render_detail(url: str) -> str:
    """환경변수에 따라 렌더러 선택"""
    if RENDERER == "playwright":
        return playwright_render(url)
    return requests_html_render(url)

# ──────────────────────────────────────────────────────────────────────────────
# 라우트
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return jsonify({"status": "healthy", "renderer": RENDERER})


@app.get("/crawl_list")
def crawl_list():
    """
    예: GET /crawl_list?keyword=이혼&page=1&size=10
    대법원 포털 목록 JSON을 그대로 가져와서 전처리한 리스트 반환
    """
    keyword = request.args.get("keyword", "")
    page = int(request.args.get("page", 1))
    size = int(request.args.get("size", 10))

    payload = {
        "dma_searchParam": {
            "srchwd": keyword,
            "sort": "jis_jdcpc_instn_dvs_cd_s asc, $relevance desc, prnjdg_ymd_o desc, jdcpct_gr_cd_s asc",
            "sortType": "정확도",
            "pageNo": str(page),
            "pageSize": str(size),
            "jdcpctGrCd": "111|112|130|141|180|182|232|235|201",
            "category": "jdcpct",
            "isKwdSearch": "N"
        }
    }

    try:
        res = requests.post(LIST_ENDPOINT, json=payload, headers=HEADERS_JSON, timeout=30)
        res.raise_for_status()
        raw = res.json()
    except Exception as e:
        return jsonify({"status": 500, "error": f"list fetch failed: {e}"}), 502

    items = normalize_items(raw)
    return jsonify({"status": 200, "data": items})


@app.get("/crawl_detail")
def crawl_detail():
    """
    예: GET /crawl_detail?srno=3236466&keyword=이혼
    JS 렌더링 후 본문 HTML만 반환 (Spring이 DB에 저장)
    """
    srno = request.args.get("srno", "").strip()
    keyword = request.args.get("keyword", "").strip()

    if not srno:
        return jsonify({"status": 400, "error": "srno required"}), 400

    # 메인 프레임 + 상세 진입 URL (탭: 판시사항)
    detail_url = (
        "https://portal.scourt.go.kr/pgp/main.on"
        f"?w2xPath=PGP1011M04&jsCntntsSrno={srno}&srchwd={keyword}&mnu=18&pgDvs=1"
    )

    try:
        html = render_detail(detail_url)
        if not html or len(html) < 500:
            return jsonify({"status": 502, "error": "empty or too short"}), 502

        main_html = extract_main_html(html)
        if not main_html or len(main_html) < 400:
            # 본문 추출 실패 시 원본 반환 (최소한 뼈대라도 저장)
            main_html = html

        return Response(main_html, mimetype="text/html; charset=utf-8")
    except Exception as e:
        return jsonify({"status": 500, "error": f"detail fetch failed: {e}"}), 502


if __name__ == "__main__":
    # 포트는 Spring의 application-dev.yml 에서 base-url(http://127.0.0.1:5001)과 일치해야 함
    app.run(host="0.0.0.0", port=5001)
