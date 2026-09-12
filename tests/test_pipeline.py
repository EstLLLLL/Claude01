import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import fetch_news as news
from ark_client import ArkError

ENV = {"ARK_API_KEY": "fake", "ARK_MODEL": "deepseek-v4-pro-ga-260813",
       "ARK_BASE_URL": "https://ark.cn-beijing.volces.com/api/v3", "DRY_RUN": "false"}
CONFIG = {"brands": ["OpenAI"], "locales": [{"language": "en-US", "country": "US", "max": 1}], "concurrency": 1}
ITEM = {"title": "OpenAI publishes research", "link": "https://openai.com/research", "source": "OpenAI", "pub_date": "today"}

class NewsPipelineTests(unittest.TestCase):
    def run_pipeline(self, folder, analyze):
        with patch.dict(os.environ, ENV), patch.object(news, "NEWS_DIR", str(folder / "news")), patch.object(news, "OUTPUT_DIR", str(folder / "output")), patch.object(news, "load_config", return_value=CONFIG), patch.object(news, "fetch_feed", return_value=b"unused"), patch.object(news, "parse_items", side_effect=lambda *a: [ITEM.copy()]), patch.object(news, "fetch_article_text", return_value="Public article text"), patch.object(news, "deepseek_analyze", side_effect=analyze), patch.object(news, "create_github_issue") as deliver:
            try:
                news.main()
            finally:
                deliver.assert_not_called()

    def test_api_failure_preserves_raw_and_previous_good_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); (folder / "news").mkdir()
            digest = folder / "news" / (datetime.date.today().isoformat() + ".md")
            digest.write_text("previous useful content")
            with self.assertRaises(ArkError):
                self.run_pipeline(folder, ArkError("Ark HTTP 402"))
            self.assertEqual(digest.read_text(), "previous useful content")
            raw = json.loads((folder / "output/raw-candidates.json").read_text())
            self.assertEqual(raw["companies"]["OpenAI"][0]["title"], ITEM["title"])
            self.assertEqual(raw["companies"]["OpenAI"][0]["article_text"], "Public article text")

    def test_legitimate_rejection_is_not_an_api_failure(self):
        with patch.object(news, "chat_json", return_value={"significant": False, "summary": "", "event": ""}):
            self.assertEqual(news.deepseek_analyze("Title", "Body", "Chinese"), (False, None, ""))

    def test_empty_summary_for_significant_news_fails(self):
        with patch.object(news, "chat_json", return_value={"significant": True, "summary": "", "event": "event"}):
            with self.assertRaises(ArkError):
                news.deepseek_analyze("Title", "Body", "Chinese")
