# SPDX-License-Identifier: LGPL-3.0-only
# Copyright (c) 2026 Mirrowel

"""Embedding requests for gemini_cli must hit aembedding / :embedContent, not acompletion."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rotator_library.client.executor import RequestExecutor
from rotator_library.client.rotating_client import RotatingClient
from rotator_library.core.types import FilterResult, RequestContext
from rotator_library.providers.gemini_cli_provider import GeminiCliProvider


class _FakeCredContext:
    def __init__(self, credential: str):
        self.credential = credential
        self.stable_id = "stable-id"

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def mark_success(self, **kwargs):
        return None


class _FakeUsageManager:
    initialized = True
    states = {}

    def get_model_quota_group(self, model):
        return None

    async def get_availability_stats(self, model, quota_group):
        return {
            "available": 1,
            "total": 1,
            "rotation_mode": "sequential",
            "blocked_by": {},
        }

    async def acquire_credential(self, **kwargs):
        return _FakeCredContext("cred.json")


class _FakeFilter:
    def filter_by_tier(self, credentials, model, provider):
        return FilterResult(compatible=list(credentials))


class _FakeTransforms:
    async def apply(self, provider, model, cred, kwargs):
        return kwargs


def _embedding_response():
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
        model_dump=lambda: {},
    )


class TestExecutorEmbeddingDispatch(unittest.IsolatedAsyncioTestCase):
    async def test_custom_provider_embedding_calls_aembedding_not_acompletion(self):
        plugin = MagicMock()
        plugin.has_custom_logic.return_value = True
        plugin.skip_cost_calculation = True
        plugin.aembedding = AsyncMock(return_value=_embedding_response())
        plugin.acompletion = AsyncMock(return_value=_embedding_response())

        executor = RequestExecutor(
            usage_managers={"gemini_cli": _FakeUsageManager()},
            cooldown_manager=None,
            credential_filter=_FakeFilter(),
            provider_transforms=_FakeTransforms(),
            provider_plugins={"gemini_cli": plugin},
            http_client=MagicMock(),
        )

        context = RequestContext(
            model="gemini_cli/gemini-embedding-001",
            provider="gemini_cli",
            kwargs={
                "model": "gemini_cli/gemini-embedding-001",
                "input": ["test"],
            },
            streaming=False,
            credentials=["cred.json"],
            deadline=time.time() + 30,
            request_type="embedding",
        )

        await executor.execute(context)

        plugin.aembedding.assert_awaited_once()
        plugin.acompletion.assert_not_awaited()

    async def test_custom_provider_completion_still_calls_acompletion(self):
        plugin = MagicMock()
        plugin.has_custom_logic.return_value = True
        plugin.skip_cost_calculation = True
        plugin.aembedding = AsyncMock(return_value=_embedding_response())
        plugin.acompletion = AsyncMock(return_value=_embedding_response())

        executor = RequestExecutor(
            usage_managers={"gemini_cli": _FakeUsageManager()},
            cooldown_manager=None,
            credential_filter=_FakeFilter(),
            provider_transforms=_FakeTransforms(),
            provider_plugins={"gemini_cli": plugin},
            http_client=MagicMock(),
        )

        context = RequestContext(
            model="gemini_cli/gemini-2.5-flash",
            provider="gemini_cli",
            kwargs={
                "model": "gemini_cli/gemini-2.5-flash",
                "messages": [{"role": "user", "content": "hi"}],
            },
            streaming=False,
            credentials=["cred.json"],
            deadline=time.time() + 30,
        )

        await executor.execute(context)

        plugin.acompletion.assert_awaited_once()
        plugin.aembedding.assert_not_awaited()

    async def test_rotating_client_aembedding_sets_request_type(self):
        captured = {}

        class _FakeExecutor:
            async def execute(self, context):
                captured["request_type"] = context.request_type
                return _embedding_response()

        client = RotatingClient.__new__(RotatingClient)
        client.all_credentials = {"gemini_cli": ["cred.json"]}
        client.global_timeout = 30
        client._executor = _FakeExecutor()

        await client.aembedding(
            model="gemini_cli/gemini-embedding-001", input=["test"]
        )

        self.assertEqual(captured["request_type"], "embedding")


class TestGeminiCliProviderEmbedding(unittest.IsolatedAsyncioTestCase):
    async def test_aembedding_posts_to_embed_content_not_stream_generate(self):
        provider = GeminiCliProvider()
        provider.project_id_cache["cred.json"] = "test-project"
        provider.get_auth_header = AsyncMock(
            return_value={"Authorization": "Bearer fake-token"}
        )

        posted = {}

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"embedding": {"values": [0.1, 0.2, 0.3]}}

        async def fake_post(url, **kwargs):
            posted["url"] = url
            posted["json"] = kwargs.get("json")
            return _FakeResponse()

        client = MagicMock()
        client.post = fake_post

        response = await provider.aembedding(
            client,
            model="gemini_cli/gemini-embedding-001",
            input=["test"],
            credential_identifier="cred.json",
        )

        self.assertIn(":embedContent", posted["url"])
        self.assertNotIn("streamGenerateContent", posted["url"])
        self.assertEqual(posted["json"]["model"], "gemini-embedding-001")
        self.assertEqual(
            posted["json"]["request"]["content"]["parts"][0]["text"], "test"
        )

        data = response.data
        first = data[0]
        values = first["embedding"] if isinstance(first, dict) else first.embedding
        self.assertEqual(list(values), [0.1, 0.2, 0.3])

    async def test_aembedding_accepts_code_assist_wrapped_response(self):
        provider = GeminiCliProvider()
        provider.project_id_cache["cred.json"] = "test-project"
        provider.get_auth_header = AsyncMock(
            return_value={"Authorization": "Bearer fake-token"}
        )

        class _FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "response": {
                        "embedding": {"values": [0.4, 0.5]},
                        "usageMetadata": {"promptTokenCount": 2},
                    }
                }

        client = MagicMock()
        client.post = AsyncMock(return_value=_FakeResponse())

        response = await provider.aembedding(
            client,
            model="gemini_cli/gemini-embedding-001",
            input="hello",
            credential_identifier="cred.json",
        )
        first = response.data[0]
        values = first["embedding"] if isinstance(first, dict) else first.embedding
        self.assertEqual(list(values), [0.4, 0.5])
        self.assertEqual(response.usage.prompt_tokens, 2)


if __name__ == "__main__":
    unittest.main()
