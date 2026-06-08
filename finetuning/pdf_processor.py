import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List

import fitz

class PDFExtractor:
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_pages(self) -> List[Dict[str, Any]]:
        pages: List[Dict[str, Any]] = []
        with fitz.open(self.pdf_path) as doc:
            for page_index, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                text = text.strip()
                if text:
                    pages.append({
                        "page": page_index,
                        "text": text,
                        "char_count": len(text),
                    })
        return pages


class TextCleaner:
    def __init__(self, min_chars_per_paragraph: int = 80):
        self.min_chars_per_paragraph = min_chars_per_paragraph

    def clean_pdf_text(self, text: str) -> str:
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\u200b", "").replace("\ufeff", "")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"(?m)^\s*\d+\s*$", "", text)

        paragraphs: List[str] = []
        for paragraph in re.split(r"\n\s*\n", text):
            paragraph = re.sub(r"\n+", " ", paragraph)
            paragraph = re.sub(r"\s+", " ", paragraph).strip()
            if paragraph:
                paragraphs.append(paragraph)

        return "\n\n".join(paragraphs)

    def clean_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned_pages: List[Dict[str, Any]] = []
        for page in pages:
            cleaned_text = self.clean_pdf_text(page["text"])
            cleaned_pages.append({
                "page": page["page"],
                "text": cleaned_text,
                "char_count": len(cleaned_text),
            })
        return cleaned_pages

    def split_into_paragraph_records(self, cleaned_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        paragraph_records: List[Dict[str, Any]] = []
        for page in cleaned_pages:
            paragraphs = page["text"].split("\n\n")
            for paragraph_index, paragraph in enumerate(paragraphs, start=1):
                paragraph = paragraph.strip()
                if len(paragraph) < self.min_chars_per_paragraph:
                    continue
                paragraph_records.append({
                    "text": paragraph,
                    "source_page": page["page"],
                    "paragraph_id": paragraph_index,
                    "char_count": len(paragraph),
                })
        return paragraph_records

    def save_jsonl(self, items: List[Dict[str, Any]], path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("w", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
