"""Download official course sources and preserve a reproducible manifest.

Run from the repository root. Cached upstream documents are reference material,
not part of the authored book. Requires beautifulsoup4 and PyMuPDF.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen
import json
import re
import subprocess

from bs4 import BeautifulSoup
import pymupdf

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "sources" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
PAGES = {"2024": "https://cs336.stanford.edu/spring2024/",
         "2025": "https://cs336.stanford.edu/spring2025/",
         "2026": "https://cs336.stanford.edu/"}
LECTURES = {"2024": "spring2024-lectures", "2025": "spring2025-lectures", "2026": "lectures"}


def get(url):
    with urlopen(Request(url, headers={"User-Agent": "CS336-Study-Guide/1.0"}), timeout=60) as r:
        return r.read()


def resolve(repo):
    result = subprocess.check_output(["git", "ls-remote", f"https://github.com/stanford-cs336/{repo}.git", "HEAD"], text=True, encoding="utf-8")
    return repo, result.split()[0]


def main():
    docs, schedule = [], []
    repos = set(LECTURES.values())
    for year, url in PAGES.items():
        html = get(url)
        (CACHE / f"schedule-{year}.html").write_bytes(html)
        soup = BeautifulSoup(html, "html.parser")
        docs.append(dict(id=f"S{year}", year=year, kind="schedule", url=url))
        for row in soup.select("tr"):
            cells = row.find_all("td")
            if len(cells) < 3 or not cells[0].get_text(strip=True).isdigit():
                continue
            num = int(cells[0].get_text(strip=True))
            desc = cells[2].get_text(" ", strip=True)
            links = [urljoin(url, a["href"]) for a in row.select("a[href]") if "lecture" in a.get_text().lower()]
            schedule.append(dict(year=year, lecture=num, description=desc, links=links))
            for index, link in enumerate(links):
                if "github.com" not in link:
                    name = "lecture_02_recording" if "recording" in link else f"lecture_{num:02d}"
                    link = f"https://github.com/stanford-cs336/{LECTURES[year]}/blob/main/{name}.py"
                docs.append(dict(id=f"L{year}-{num:02d}" + (f"-{index}" if index else ""), year=year, kind="lecture", title=desc, url=link))
        for a in soup.select("a[href]"):
            link = urljoin(url, a["href"])
            if "assignment" in link and link.endswith(".pdf"):
                m = re.search(r"assignment(\d)", link)
                suffix = "-safety" if "supplement" in link else ""
                sid = f"A{year}-{m[1]}{suffix}"
                if not any(d["id"] == sid for d in docs):
                    docs.append(dict(id=sid, year=year, kind="assignment", url=link))
    for d in docs:
        if "github.com/stanford-cs336/" in d["url"]:
            repos.add(d["url"].split("/")[4])
    with ThreadPoolExecutor(max_workers=6) as ex:
        commits = dict(ex.map(resolve, sorted(repos)))
    def fetch(d):
        d = d.copy()
        url = d["url"]
        if "github.com/stanford-cs336/" in url:
            bits = url.split("/")
            repo, ref, path = bits[4], bits[6], "/".join(bits[7:])
            ref = commits[repo] if ref in {"main", "master"} else ref
            d["commit"] = ref
            d["pinned_url"] = f"https://github.com/stanford-cs336/{repo}/blob/{ref}/{path}"
            url = f"https://raw.githubusercontent.com/stanford-cs336/{repo}/{ref}/{path}"
        d["download_url"] = url
        try:
            raw = get(url)
            ext = ".pdf" if url.endswith(".pdf") else ".py" if url.endswith(".py") else ".html"
            path = CACHE / f"{d['id']}{ext}"
            path.write_bytes(raw)
            if ext == ".pdf":
                pdf = pymupdf.open(stream=raw, filetype="pdf")
                text = "\n".join(f"\n--- page {i + 1} ---\n" + page.get_text() for i, page in enumerate(pdf))
                d["pages"] = len(pdf)
            elif ext == ".html":
                text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)
            else:
                text = raw.decode("utf-8")
            (CACHE / f"{d['id']}.txt").write_text(text, encoding="utf-8")
            d.update(status="downloaded", sha256=sha256(raw).hexdigest(), bytes=len(raw), text_chars=len(text))
        except Exception as error:
            d.update(status="unavailable", error=str(error))
        d["retrieved_utc"] = datetime.now(timezone.utc).isoformat()
        print(d["id"], d["status"], d.get("pages", ""), d.get("text_chars", ""), flush=True)
        return d
    with ThreadPoolExecutor(max_workers=8) as ex:
        result = list(ex.map(fetch, docs))
    (ROOT / "sources" / "manifest.json").write_text(json.dumps(dict(commits=commits, schedule=schedule, documents=result), ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
