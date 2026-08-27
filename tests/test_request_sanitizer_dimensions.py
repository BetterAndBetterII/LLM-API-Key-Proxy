# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""dimensions must reach gemini_cli embeddings (review on #169)."""

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.request_sanitizer import sanitize_request_payload


class TestSanitizeDimensions(unittest.TestCase):
    def test_keeps_dimensions_for_openai_text_embedding_3(self):
        out = sanitize_request_payload(
            {"input": ["a"], "dimensions": 256},
            "openai/text-embedding-3-small",
        )
        self.assertEqual(out.get("dimensions"), 256)

    def test_keeps_dimensions_for_gemini_cli_embedding_models(self):
        out = sanitize_request_payload(
            {"input": ["a"], "dimensions": 768, "taskType": "RETRIEVAL_DOCUMENT"},
            "gemini_cli/gemini-embedding-001",
        )
        self.assertEqual(out.get("dimensions"), 768)
        self.assertEqual(out.get("taskType"), "RETRIEVAL_DOCUMENT")

    def test_strips_dimensions_for_unrelated_chat_models(self):
        out = sanitize_request_payload(
            {"messages": [], "dimensions": 256},
            "gemini_cli/gemini-2.5-flash",
        )
        self.assertNotIn("dimensions", out)


if __name__ == "__main__":
    unittest.main()
