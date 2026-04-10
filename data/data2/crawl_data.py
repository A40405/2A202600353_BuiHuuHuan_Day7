"""
Vinmec Disease Crawler
======================
Crawl tất cả bài bệnh từ https://www.vinmec.com/vie/benh/
Mỗi bài được lưu vào 1 file .md riêng trong thư mục "vinmec_diseases/"

Cài đặt thư viện:
    pip install requests beautifulsoup4

Chạy:
    python vinmec_crawler.py
"""

import os
import re
import time
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

# ─── Cấu hình ────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.vinmec.com"
INDEX_URL       = "https://www.vinmec.com/vie/benh/"
OUTPUT_DIR      = "vinmec_diseases"
DELAY_SECONDS   = 1.5      # Nghỉ giữa mỗi request (lịch sự với server)
REQUEST_TIMEOUT = 20
LOG_FILE        = "crawler.log"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def fetch(url: str) -> BeautifulSoup | None:
    """GET một URL, trả về BeautifulSoup hoặc None nếu lỗi."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        log.warning(f"Lỗi khi tải {url}: {e}")
        return None


def slugify(text: str) -> str:
    """Chuyển tiêu đề thành tên file an toàn."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s]+", "_", text)
    text = re.sub(r"-+", "-", text)
    return text[:120]  # giới hạn độ dài


# ─── Thu thập danh sách link bài bệnh ────────────────────────────────────────

def get_all_disease_links() -> list[str]:
    """
    Trang index https://www.vinmec.com/vie/benh/ liệt kê bệnh theo chữ cái A-Z.
    Mỗi chữ cái có trang riêng: /vie/tra-cuu-benh/<letter>/
    Hàm này duyệt tất cả các trang đó và thu thập link từng bài bệnh.
    """
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    disease_links: set[str] = set()

    # 1. Trang chính /vie/benh/ (có thể đã có một số link)
    soup = fetch(INDEX_URL)
    if soup:
        _extract_disease_links(soup, disease_links)

    # 2. Các trang theo chữ cái /vie/tra-cuu-benh/<letter>/
    for letter in alphabet:
        letter_url = f"{BASE_URL}/vie/tra-cuu-benh/{letter}/"
        log.info(f"Đang lấy danh sách bệnh chữ '{letter.upper()}': {letter_url}")
        soup = fetch(letter_url)
        if soup:
            found = _extract_disease_links(soup, disease_links)
            log.info(f"  → Tìm thấy {found} link mới")
        time.sleep(DELAY_SECONDS)

    return sorted(disease_links)


def _extract_disease_links(soup: BeautifulSoup, result_set: set) -> int:
    """
    Trích xuất tất cả link dạng /vie/benh/<slug> từ trang.
    Trả về số link mới được thêm.
    """
    before = len(result_set)
    pattern = re.compile(r"^/vie/benh/[^/]+$")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Chuẩn hóa: lấy đường dẫn tương đối
        if href.startswith("http"):
            path = urlparse(href).path
        else:
            path = href

        if pattern.match(path.rstrip("/")):
            full_url = urljoin(BASE_URL, path)
            result_set.add(full_url)

    return len(result_set) - before


# ─── Parse nội dung một bài bệnh ────────────────────────────────────────────

# Các tab / section thường gặp trên trang bệnh Vinmec
SECTION_IDS = [
    ("Tổng quan",         ["tab-10728", "tong-quan"]),
    ("Nguyên nhân",       ["tab-10729", "nguyen-nhan"]),
    ("Triệu chứng",       ["tab-10730", "trieu-chung"]),
    ("Đối tượng nguy cơ", ["tab-10731", "doi-tuong-nguy-co"]),
    ("Phòng ngừa",        ["tab-10732", "phong-ngua"]),
    ("Biện pháp chẩn đoán", ["tab-10733", "chan-doan"]),
    ("Biện pháp điều trị",  ["tab-10734", "dieu-tri"]),
]


def parse_disease_page(url: str) -> dict | None:
    """
    Tải và phân tích trang bệnh Vinmec.
    Trả về dict: {title, url, sections: [{heading, content}]}
    """
    soup = fetch(url)
    if not soup:
        return None

    # ── Tiêu đề chính ──────────────────────────────────────────────────────
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else url.split("/")[-1]

    # ── Lấy nội dung từng section ──────────────────────────────────────────
    sections = []

    # Cách 1: tìm theo id tab (cấu trúc mới nhất của Vinmec)
    # Mỗi section nằm trong <div id="tab-XXXXX"> hoặc thẻ tương tự
    tab_divs = soup.find_all(
        lambda tag: tag.name in ("div", "section") and
        tag.get("id", "").startswith("tab-")
    )

    if tab_divs:
        for div in tab_divs:
            heading_tag = div.find(re.compile(r"^h[2-4]$"))
            heading = heading_tag.get_text(strip=True) if heading_tag else div.get("id", "")
            content = _extract_text_from_tag(div)
            if content.strip():
                sections.append({"heading": heading, "content": content})
    else:
        # Cách 2: fallback – lấy tất cả h2/h3 và nội dung theo sau
        sections = _extract_by_headings(soup)

    # Cách 3: nếu vẫn trống – lấy toàn bộ main content
    if not sections:
        main = (
            soup.find("main") or
            soup.find("article") or
            soup.find("div", class_=re.compile(r"content|article|detail", re.I))
        )
        if main:
            sections.append({"heading": "", "content": _extract_text_from_tag(main)})

    return {"title": title, "url": url, "sections": sections}


def _extract_text_from_tag(tag) -> str:
    """Lấy văn bản sạch từ thẻ HTML, giữ cấu trúc dòng."""
    lines = []
    for element in tag.descendants:
        if element.name in ("p", "li", "h2", "h3", "h4", "h5"):
            text = element.get_text(" ", strip=True)
            if text:
                if element.name.startswith("h"):
                    lines.append(f"\n### {text}\n")
                elif element.name == "li":
                    lines.append(f"- {text}")
                else:
                    lines.append(text)
    return "\n".join(lines)


def _extract_by_headings(soup: BeautifulSoup) -> list[dict]:
    """Fallback: tách nội dung theo h2/h3."""
    sections = []
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find("div", id=re.compile(r"content|main", re.I)) or
        soup.body
    )
    if not main:
        return sections

    current_heading = ""
    current_lines = []

    for tag in main.find_all(["h2", "h3", "p", "li", "ul", "ol"]):
        if tag.name in ("h2", "h3"):
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "content": "\n".join(current_lines),
                })
                current_lines = []
            current_heading = tag.get_text(strip=True)
        elif tag.name in ("p",):
            text = tag.get_text(" ", strip=True)
            if text:
                current_lines.append(text)
        elif tag.name == "li":
            text = tag.get_text(" ", strip=True)
            if text:
                current_lines.append(f"- {text}")

    if current_lines:
        sections.append({"heading": current_heading, "content": "\n".join(current_lines)})

    return sections


# ─── Chuyển sang Markdown ────────────────────────────────────────────────────

def to_markdown(data: dict) -> str:
    """Chuyển dict dữ liệu bài bệnh thành chuỗi Markdown."""
    lines = [
        f"# {data['title']}",
        "",
        f"> **Nguồn:** [{data['url']}]({data['url']})",
        "",
    ]
    for section in data["sections"]:
        if section["heading"]:
            lines.append(f"## {section['heading']}")
            lines.append("")
        if section["content"]:
            lines.append(section["content"].strip())
            lines.append("")
    return "\n".join(lines)


# ─── Lưu file ────────────────────────────────────────────────────────────────

def save_markdown(data: dict, output_dir: str):
    """Lưu dữ liệu bài bệnh vào file .md."""
    filename = slugify(data["title"]) + ".md"
    filepath = os.path.join(output_dir, filename)

    # Tránh ghi đè file trùng tên
    if os.path.exists(filepath):
        base = slugify(data["title"])
        slug = data["url"].rstrip("/").split("/")[-1]
        filename = f"{base}__{slug}.md"
        filepath = os.path.join(output_dir, filename)

    content = to_markdown(data)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Tạo thư mục output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info(f"Thư mục lưu bài: '{OUTPUT_DIR}/'")

    # Bước 1: Thu thập tất cả link bài bệnh
    log.info("=== Bước 1: Thu thập danh sách link bài bệnh ===")
    links = get_all_disease_links()
    log.info(f"Tổng số link tìm được: {len(links)}")

    if not links:
        log.error("Không tìm thấy link nào. Kiểm tra lại selector hoặc cấu trúc trang.")
        return

    # Lưu danh sách link để có thể resume nếu bị gián đoạn
    with open("disease_links.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(links))
    log.info("Đã lưu danh sách link vào 'disease_links.txt'")

    # Bước 2: Crawl từng bài
    log.info("=== Bước 2: Crawl từng bài bệnh ===")
    success = 0
    failed = []

    for i, url in enumerate(links, 1):
        log.info(f"[{i}/{len(links)}] Đang crawl: {url}")
        data = parse_disease_page(url)

        if data:
            filepath = save_markdown(data, OUTPUT_DIR)
            log.info(f"  ✓ Đã lưu: {filepath}")
            success += 1
        else:
            log.warning(f"  ✗ Thất bại: {url}")
            failed.append(url)

        time.sleep(DELAY_SECONDS)

    # Tổng kết
    log.info("=" * 50)
    log.info(f"Hoàn thành! Thành công: {success} | Thất bại: {len(failed)}")
    if failed:
        with open("failed_urls.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(failed))
        log.info(f"Các URL thất bại đã lưu vào 'failed_urls.txt'")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()