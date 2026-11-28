"""Fetch the published profile for India's current Prime Minister."""
from html.parser import HTMLParser

import requests

from src.verification.officeholders import INDIA_PM_PROFILE, india_pm_profile_holder


class _Paragraphs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paragraphs = []
        self._parts = None

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self._parts = []

    def handle_data(self, data):
        if self._parts is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag == "p" and self._parts is not None:
            self.paragraphs.append(" ".join("".join(self._parts).split()))
            self._parts = None


def retrieve_india_pm_profile() -> dict | None:
    response = requests.get(INDIA_PM_PROFILE, timeout=(4, 8), allow_redirects=False)
    response.raise_for_status()
    if response.status_code != 200:
        return None
    # The page is served as UTF-8. Decoding bytes also preserves curly apostrophes
    # when the server omits the charset in the response header.
    page = _Paragraphs()
    page.feed(response.content.decode("utf-8", errors="replace"))
    for paragraph in page.paragraphs:
        if india_pm_profile_holder(paragraph):
            return {
                "docid": "india-pmo-current-profile",
                "title": "Know the PM | Prime Minister of India",
                "text": paragraph,
                "url": INDIA_PM_PROFILE,
                "source": "official",
            }
    return None
