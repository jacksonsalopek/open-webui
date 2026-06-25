"""Tests for stopword classification (penalty, never hard-drop)."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from open_webui.retrieval.concepts.extraction.stopwords import (
    StopwordClass,
    classify,
    extend_stopwords,
    is_stopword,
)


@pytest.mark.parametrize(
    'token',
    ['the', 'and', 'with', 'for', 'from'],
)
def test_english_stopwords_classified(token: str) -> None:
    assert classify(token) == StopwordClass.ENGLISH


@pytest.mark.parametrize(
    'token',
    ['value', 'result', 'get', 'item', 'helper'],
)
def test_code_stopwords_classified(token: str) -> None:
    assert classify(token) == StopwordClass.CODE


def test_language_stopwords_classified() -> None:
    assert classify('void', language='csharp') == StopwordClass.LANGUAGE
    assert classify('void', language='python') == StopwordClass.NOT_STOPWORD


@pytest.mark.parametrize(
    'token',
    ['toolbar', 'extension', 'guardrail'],
)
def test_nonstopword_classified(token: str) -> None:
    assert classify(token) == StopwordClass.NOT_STOPWORD


def test_classify_unknown_language_falls_back_to_global() -> None:
    assert classify('the', language='haskell') == StopwordClass.ENGLISH


def test_extend_stopwords_adds_new_term() -> None:
    extend_stopwords(code=['lollipop'])
    assert classify('lollipop') == StopwordClass.CODE
    extend_stopwords(code=[])


def test_is_stopword_convenience() -> None:
    assert is_stopword('the') is True
    assert is_stopword('toolbar') is False


def test_concurrent_extension_safety() -> None:
    def add_code_word(word: str) -> None:
        extend_stopwords(code=[word])

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(add_code_word, f'projword{i}') for i in range(20)]
        for future in futures:
            future.result()

    for i in range(20):
        assert classify(f'projword{i}') == StopwordClass.CODE


@pytest.mark.parametrize(
    ('token', 'expected'),
    [
        ('if', StopwordClass.LANGUAGE),
        ('null', StopwordClass.CODE),
        ('return', StopwordClass.LANGUAGE),
        ('string', StopwordClass.LANGUAGE),
        ('var', StopwordClass.CODE),
        ('new', StopwordClass.LANGUAGE),
        ('class', StopwordClass.LANGUAGE),
        ('void', StopwordClass.LANGUAGE),
        ('record', StopwordClass.LANGUAGE),
    ],
)
def test_csharp_keywords_classified_as_language(token: str, expected: StopwordClass) -> None:
    assert classify(token, language='csharp') == expected


@pytest.mark.parametrize(
    ('token', 'expected'),
    [
        ('summary', StopwordClass.LANGUAGE),
        ('cref', StopwordClass.LANGUAGE),
        ('see', StopwordClass.LANGUAGE),
        ('param', StopwordClass.CODE),
        ('remarks', StopwordClass.LANGUAGE),
        ('returns', StopwordClass.LANGUAGE),
    ],
)
def test_xmldoc_elements_classified_as_language_csharp(
    token: str,
    expected: StopwordClass,
) -> None:
    assert classify(token, language='csharp') == expected


@pytest.mark.parametrize(
    'token',
    ['get', 'set', 'value', 'count', 'length', 'index', 'default', 'null'],
)
def test_universal_code_words_classified_as_code(token: str) -> None:
    assert classify(token) == StopwordClass.CODE


@pytest.mark.parametrize(
    'token',
    ['so', 'also', 'here', 'however', 'therefore'],
)
def test_expanded_english_filler_classified_as_english(token: str) -> None:
    assert classify(token) == StopwordClass.ENGLISH


def test_python_typescript_keywords_classified_when_language_given() -> None:
    for token in ('def', 'import', 'lambda'):
        assert classify(token, language='python') == StopwordClass.LANGUAGE
    for token in ('interface', 'extends', 'keyof'):
        assert classify(token, language='typescript') == StopwordClass.LANGUAGE


@pytest.mark.parametrize(
    'token',
    [
        'task',
        'cancel',
        'request',
        'response',
        'data',
        'state',
        'service',
        'model',
        'view',
        'controller',
        'repository',
        'widget',
        'extension',
        'toolbar',
        'clipboard',
    ],
)
def test_semantically_meaningful_words_remain_non_stopwords(token: str) -> None:
    assert classify(token) == StopwordClass.NOT_STOPWORD


def test_event_is_language_for_csharp_but_not_globally() -> None:
    assert classify('event', language='csharp') == StopwordClass.LANGUAGE
    assert classify('event') == StopwordClass.NOT_STOPWORD
