"""Golden-set tests for identifier tokenization (Lollipop-grounded fixtures)."""

from __future__ import annotations

from open_webui.retrieval.concepts.extraction.identifiers import (
    CSHARP_DEFAULT_RULES,
    PYTHON_DEFAULT_RULES,
    TYPESCRIPT_DEFAULT_RULES,
    rules_for_language,
    tokenize,
    tokenize_text,
)


def test_toolbar_view_model() -> None:
    assert tokenize('ToolbarViewModel', rules=CSHARP_DEFAULT_RULES) == [
        'toolbar',
        'view',
        'model',
    ]


def test_interface_prefix_stripped() -> None:
    # Leading I + uppercase next char is dropped entirely (C# interface convention).
    assert tokenize('IObservableObject', rules=CSHARP_DEFAULT_RULES) == [
        'observable',
        'object',
    ]


def test_interface_prefix_retained_single_i_dropped() -> None:
    # With strip off, 'i' is emitted but filtered by min_token_length=2.
    rules = CSHARP_DEFAULT_RULES._replace(strip_interface_prefix=False)
    assert tokenize('IObservableObject', rules=rules) == ['observable', 'object']


def test_underscore_field_prefix() -> None:
    assert tokenize('_selection', rules=CSHARP_DEFAULT_RULES) == ['selection']


def test_acronym_llm_service() -> None:
    # Consecutive caps before a PascalCase word form one acronym token.
    assert tokenize('LlmService', rules=CSHARP_DEFAULT_RULES) == ['llm', 'service']


def test_llama_cpp_llm_service() -> None:
    assert tokenize('LlamaCppLlmService', rules=CSHARP_DEFAULT_RULES) == [
        'llama',
        'cpp',
        'llm',
        'service',
    ]


def test_onnx_gen_ai_llm_service() -> None:
    # GenAi splits on lower→upper: Gen + Ai (not a single acronym).
    assert tokenize('OnnxGenAiLlmService', rules=CSHARP_DEFAULT_RULES) == [
        'onnx',
        'gen',
        'ai',
        'llm',
        'service',
    ]


def test_http_model_downloader() -> None:
    assert tokenize('HttpModelDownloader', rules=CSHARP_DEFAULT_RULES) == [
        'http',
        'model',
        'downloader',
    ]


def test_i_model_downloader() -> None:
    assert tokenize('IModelDownloader', rules=CSHARP_DEFAULT_RULES) == [
        'model',
        'downloader',
    ]


def test_i_notification_service() -> None:
    assert tokenize('INotificationService', rules=CSHARP_DEFAULT_RULES) == [
        'notification',
        'service',
    ]


def test_i_update_service() -> None:
    assert tokenize('IUpdateService', rules=CSHARP_DEFAULT_RULES) == ['update', 'service']


def test_i_theme_service() -> None:
    assert tokenize('IThemeService', rules=CSHARP_DEFAULT_RULES) == ['theme', 'service']


def test_dotted_namespace() -> None:
    assert tokenize('System.Threading.Tasks', rules=CSHARP_DEFAULT_RULES) == [
        'system',
        'threading',
        'tasks',
    ]


def test_observable_property_attribute() -> None:
    # Brackets stripped; inner PascalCase tokenized.
    assert tokenize('[ObservableProperty]', rules=CSHARP_DEFAULT_RULES) == [
        'observable',
        'property',
    ]


def test_relay_command_attribute_with_args() -> None:
    # Parenthetical attribute args are discarded; only the attribute name remains.
    assert tokenize(
        '[RelayCommand(CanExecute = nameof(CanSignOut))]',
        rules=CSHARP_DEFAULT_RULES,
    ) == ['relay', 'command']


def test_dictionary_generics_stripped() -> None:
    # strip_generics=True drops <...>; inner type names are not separate tokens.
    assert tokenize('Dictionary<string, int>', rules=CSHARP_DEFAULT_RULES) == [
        'dictionary',
    ]


def test_python_snake_case() -> None:
    assert tokenize('async_method_name', rules=PYTHON_DEFAULT_RULES) == [
        'async',
        'method',
        'name',
    ]


def test_python_private_helper() -> None:
    # Stopwords penalize later; tokenizer still emits 'private'.
    assert tokenize('_private_helper', rules=PYTHON_DEFAULT_RULES) == [
        'private',
        'helper',
    ]


def test_python_dunder_method() -> None:
    # All leading/trailing underscores stripped, then snake_case split.
    assert tokenize('__dunder_method__', rules=PYTHON_DEFAULT_RULES) == [
        'dunder',
        'method',
    ]


def test_typescript_camel_case() -> None:
    assert tokenize('getUserById', rules=TYPESCRIPT_DEFAULT_RULES) == [
        'get',
        'user',
        'by',
        'id',
    ]


def test_typescript_html_div_element() -> None:
    assert tokenize('HTMLDivElement', rules=TYPESCRIPT_DEFAULT_RULES) == [
        'html',
        'div',
        'element',
    ]


def test_tokenize_text_embedded_identifiers() -> None:
    result = tokenize_text(
        'The IObservableObject interface uses [ObservableProperty] markers',
        rules=CSHARP_DEFAULT_RULES,
    )
    assert 'observable' in result
    assert 'object' in result
    assert 'property' in result


def test_empty_string() -> None:
    assert tokenize('', rules=CSHARP_DEFAULT_RULES) == []


def test_whitespace_only() -> None:
    assert tokenize('   \t  ', rules=CSHARP_DEFAULT_RULES) == []


def test_single_character_filtered() -> None:
    assert tokenize('x', rules=CSHARP_DEFAULT_RULES) == []


def test_numbers_only_dropped_by_default() -> None:
    assert tokenize('42', rules=CSHARP_DEFAULT_RULES) == []


def test_numbers_only_kept_when_opt_in() -> None:
    rules = CSHARP_DEFAULT_RULES._replace(keep_pure_numeric_tokens=True)
    assert tokenize('42', rules=rules) == ['42']


def test_http2_connection_splits_number() -> None:
    assert tokenize('Http2Connection', rules=CSHARP_DEFAULT_RULES) == [
        'http',
        'connection',
    ]


def test_pure_numeric_tokens_dropped() -> None:
    result = tokenize_text(
        'value 13 port 8080 size 64KB',
        rules=CSHARP_DEFAULT_RULES,
    )
    assert '13' not in result
    assert '8080' not in result
    assert 'kb' in result  # alphabetic suffix of embedded 64KB match
    assert 'port' in result
    assert 'size' in result
    assert 'value' in result
    assert tokenize('13', rules=CSHARP_DEFAULT_RULES) == []
    assert tokenize('8080', rules=CSHARP_DEFAULT_RULES) == []


def test_alphanumeric_tokens_kept() -> None:
    assert tokenize('utf8', rules=CSHARP_DEFAULT_RULES) == ['utf']
    assert tokenize('abc123', rules=CSHARP_DEFAULT_RULES) == ['abc']
    assert tokenize('v2alpha', rules=CSHARP_DEFAULT_RULES) == ['alpha']
    assert tokenize('Size64KB', rules=CSHARP_DEFAULT_RULES) == ['size', 'kb']


def test_hex_literal_with_alpha_survives() -> None:
    assert tokenize('0xFF', rules=CSHARP_DEFAULT_RULES) == ['ff']


def test_pure_digit_sequences_dropped() -> None:
    assert tokenize('0042', rules=CSHARP_DEFAULT_RULES) == []


def test_rules_for_language_csharp() -> None:
    assert rules_for_language('csharp') == CSHARP_DEFAULT_RULES


def test_rules_for_language_python() -> None:
    assert rules_for_language('python') == PYTHON_DEFAULT_RULES


def test_rules_for_language_typescript() -> None:
    assert rules_for_language('typescript') == TYPESCRIPT_DEFAULT_RULES


# ---- Short PascalCase acronym merge (wave-7, ``NoOp`` mismatch fix) ----


def test_no_op_emits_merged_acronym() -> None:
    """``NoOp`` → ``no, op, noop`` so a query that types ``noop`` as a
    single lowercase chunk still resolves to the same atomic the
    extractor surfaces."""
    assert tokenize('NoOp', rules=CSHARP_DEFAULT_RULES) == ['no', 'op', 'noop']


def test_looks_like_no_op_merges_only_adjacent_short_parts() -> None:
    """Only adjacent parts that are both ``≤ short_pascal_merge_max_part_len``
    (default 2) get merged — ``looks`` and ``like`` are too long."""
    assert tokenize('LooksLikeNoOp', rules=CSHARP_DEFAULT_RULES) == [
        'looks',
        'like',
        'no',
        'op',
        'noop',
    ]


def test_long_compound_not_collapsed() -> None:
    """``HttpModelDownloader`` has all parts ≥ 4 chars; no merge token
    should be emitted."""
    assert 'httpmodel' not in tokenize(
        'HttpModelDownloader',
        rules=CSHARP_DEFAULT_RULES,
    )


def test_acronym_merge_blocklist_skips_common_prepositions() -> None:
    """``ByDay``, ``IsOk``, ``UpTo`` look like short-PascalCase candidates
    but their left part is a common English particle; the blocklist
    suppresses the merge so we don't pollute the atomic index with
    ``byday`` / ``isok`` / ``upto``."""
    by_day = tokenize('ByDay', rules=CSHARP_DEFAULT_RULES)
    is_ok = tokenize('IsOk', rules=CSHARP_DEFAULT_RULES)
    up_to = tokenize('UpTo', rules=CSHARP_DEFAULT_RULES)
    assert 'byday' not in by_day
    assert 'isok' not in is_ok
    assert 'upto' not in up_to


def test_merge_disabled_by_rules_flag() -> None:
    rules = CSHARP_DEFAULT_RULES._replace(emit_short_pascal_acronym_merges=False)
    assert tokenize('NoOp', rules=rules) == ['no', 'op']
