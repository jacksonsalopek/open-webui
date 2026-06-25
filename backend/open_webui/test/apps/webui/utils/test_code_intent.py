"""Parsing/validation tests for ``open_webui.utils.code_intent``.

The LLM call itself is mocked at the ``_call_litellm`` boundary so
these tests cover ONLY the parsing, fail-open, and empty-input logic.
End-to-end validation against the real classifier model lives in the
manual smoke-test scripts, not here.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from open_webui.utils import code_intent


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch):
    # The classifier short-circuits when ``OPENAI_API_BASE_URL`` isn't
    # set; every test except the explicit empty-input one needs a base
    # URL to even reach the mocked call layer.
    monkeypatch.setenv('OPENAI_API_BASE_URL', 'http://litellm.test/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-test')


@pytest.mark.parametrize(
    'raw,expected',
    [
        ('find_symbol', 'find_symbol'),
        ('where_used', 'where_used'),
        ('explain_region', 'explain_region'),
        ('generate_code', 'generate_code'),
        ('generate_and_run', 'generate_and_run'),
        ('refactor', 'refactor'),
        # Decoration the model sometimes adds; the normalizer strips it.
        ('  refactor  ', 'refactor'),
        ('"find_symbol"', 'find_symbol'),
        ('`generate_code`', 'generate_code'),
        ('refactor.', 'refactor'),
    ],
)
def test_valid_label_passes_through(raw, expected):
    async def fake_call(**kwargs):
        return raw

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fake_call):
        result = asyncio.run(code_intent.classify_code_intent('rename foo to bar'))
    assert result == expected


def test_unknown_label_returns_unknown():
    async def fake_call(**kwargs):
        return 'totally_made_up'

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fake_call):
        result = asyncio.run(code_intent.classify_code_intent('do a thing'))
    assert result == 'unknown'


def test_timeout_returns_unknown():
    async def fake_call(**kwargs):
        await asyncio.sleep(10)
        return 'find_symbol'

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fake_call):
        result = asyncio.run(
            code_intent.classify_code_intent('find foo', timeout_seconds=0.05),
        )
    assert result == 'unknown'


def test_network_error_returns_unknown():
    async def fake_call(**kwargs):
        raise ConnectionError('litellm down')

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fake_call):
        result = asyncio.run(code_intent.classify_code_intent('find foo'))
    assert result == 'unknown'


def test_call_returning_none_yields_unknown():
    # ``_call_litellm`` swallows transport errors internally and returns
    # ``None`` rather than raising. The wrapper must still map that to
    # ``'unknown'`` without raising.
    async def fake_call(**kwargs):
        return None

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fake_call):
        result = asyncio.run(code_intent.classify_code_intent('find foo'))
    assert result == 'unknown'


def test_empty_input_short_circuits(monkeypatch):
    # Empty input must NOT hit the network. Patch ``_call_litellm`` to
    # raise so the test fails loudly if the short-circuit regresses.
    async def fail_call(**kwargs):
        raise AssertionError('classifier should not be called on empty input')

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fail_call):
        assert asyncio.run(code_intent.classify_code_intent('')) == 'unknown'
        assert asyncio.run(code_intent.classify_code_intent('   ')) == 'unknown'


def test_missing_base_url_returns_unknown(monkeypatch):
    # Drop the env var the autouse fixture set so we exercise the
    # "no endpoint configured" fail-open branch.
    monkeypatch.delenv('OPENAI_API_BASE_URL', raising=False)
    monkeypatch.delenv('OPENAI_API_BASE_URLS', raising=False)

    async def fail_call(**kwargs):
        raise AssertionError('classifier should not be called without a base URL')

    with mock.patch.object(code_intent, '_call_litellm', side_effect=fail_call):
        assert asyncio.run(code_intent.classify_code_intent('find foo')) == 'unknown'
