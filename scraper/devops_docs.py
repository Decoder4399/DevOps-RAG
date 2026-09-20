import requests
from bs4 import BeautifulSoup
import time
import hashlib
import json
import os
from typing import List, Dict
from dataclasses import dataclass, asdict


@dataclass
class Document:
    content: str
    metadata: Dict[str, str]


class DevOpsDocScraper:
    def __init__(self, cache_dir: str = "./scrape_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DevOps-RAG/1.0"
        })

    def _get_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def _load_from_cache(self, url: str) -> str | None:
        cache_key = self._get_cache_key(url)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    def _save_to_cache(self, url: str, content: str):
        cache_key = self._get_cache_key(url)
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.txt")
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _fetch_page(self, url: str, retries: int = 3) -> str | None:
        cached = self._load_from_cache(url)
        if cached:
            return cached

        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                resp.raise_for_status()
                content = self._extract_content(url, resp.text)
                if content:
                    self._save_to_cache(url, content)
                return content
            except requests.RequestException as e:
                print(f"  Attempt {attempt + 1}/{retries} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None

    def _extract_content(self, url: str, html: str) -> str | None:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        selectors = [
            "article",
            "main",
            '[role="main"]',
            ".content",
            ".documentation",
            "#content",
            ".md-content",
            ".td-content",
        ]
        main_content = None
        for sel in selectors:
            main_content = soup.select_one(sel)
            if main_content:
                break

        if not main_content:
            main_content = soup.body

        if not main_content:
            return None

        text = main_content.get_text(separator="\n", strip=True)
        text = "\n".join(line for line in text.splitlines() if line.strip())

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)

        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        if title:
            text = f"# {title}\n\n{text}"

        return text if len(text) > 100 else None

    def scrape_urls(self, urls: List[str], source_name: str) -> List[Document]:
        documents = []
        print(f"\nScraping {source_name} ({len(urls)} pages)...")

        for i, url in enumerate(urls, 1):
            print(f"  [{i}/{len(urls)}] {url}")
            content = self._fetch_page(url)
            if content:
                doc = Document(
                    content=content,
                    metadata={
                        "source": source_name,
                        "url": url,
                        "title": content.split("\n")[0].replace("# ", "")[:100],
                    },
                )
                documents.append(doc)
                print(f"    OK {len(content)} chars")
            else:
                print(f"    FAIL Failed to fetch")
            time.sleep(1)

        print(f"  Scraped {len(documents)}/{len(urls)} pages for {source_name}")
        return documents

    def scrape_all(self, sources: Dict[str, List[str]]) -> List[Document]:
        all_docs = []
        for source_name, urls in sources.items():
            docs = self.scrape_urls(urls, source_name)
            all_docs.extend(docs)
        print(f"\nTotal documents scraped: {len(all_docs)}")
        return all_docs
