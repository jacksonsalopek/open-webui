"""AST-aware code splitter for KB ingestion.

Open WebUI's default knowledge-base chunkers (``RecursiveCharacterTextSplitter``
and ``TokenTextSplitter``) shred source files by raw character / token count,
which routinely cuts in the middle of a function body -- the embedder then
sees half-functions with no signature, and downstream retrieval can't tell
``def fetch_user`` from ``def update_user`` because the chunk it indexed
started at a random colon-newline in the middle.

This module walks the file's AST via ``tree-sitter-language-pack`` and
splits at function / class / method boundaries, so each emitted chunk is
a semantically-meaningful unit (one definition + its docstring + its
body, up to CHUNK_SIZE). Top-level imports and module-level statements
are gathered into a single "imports & module preamble" chunk that
precedes the definitions -- the embedder gets to see the modules / use
statements / globals that bind into every other chunk's identifiers.

Two grammar-level lookups govern what gets split and what gets attached:

- ``_DEFINITIONS_BY_LANG`` -- per-language ``frozenset`` of *exact*
  tree-sitter node types that should each become their own chunk.
  Exact (not substring) so e.g. JS/TS ``lexical_declaration`` doesn't
  promote every top-level ``const x = 1`` into its own chunk. Languages
  without an entry fall back to a small set of cross-grammar substring
  patterns so unmapped grammars still get *some* useful splitting.
- ``_PREFIX_NODES_BY_LANG`` -- per-language ``frozenset`` of node types
  that, when they appear as top-level siblings immediately *before* a
  definition, get glued onto that definition's chunk rather than dumped
  into the preamble. This is what keeps ``/// <summary>`` blocks,
  ``[ObservableProperty]`` attributes, Rust ``#[derive]`` items, Java
  ``@Override`` annotations, etc. with the symbol they decorate.

Python is a special case: ``decorated_definition`` already wraps
function/class + ``@decorators`` in a single node, so its prefix set
only needs ``comment`` for hash-comments above a definition.

Oversized definitions (a single AST node larger than ``chunk_size``)
trigger a third lookup:

- ``_MEMBERS_BY_LANG`` -- per-language ``frozenset`` of node types that
  count as a *member* inside a container body (methods, properties,
  fields, nested types). Used by :func:`_split_by_members` to recurse
  into an oversized class / struct / namespace and emit one chunk per
  member instead of letting the raw character splitter shred it. The
  recursion is bounded by ``_MAX_MEMBER_RECURSION_DEPTH`` (3) so a
  namespace → class → nested class → method chain still splits cleanly.
  Each member chunk carries ``ast_parent_kind`` / ``ast_parent_symbol``
  so retrieval can group "all methods of class Foo" without re-parsing.

Fall-backs:

- Unknown language (no entry in the extension map): yield the full text
  as a single chunk, then let ``RecursiveCharacterTextSplitter`` decide
  how to break it up.
- Oversized definition AND member-level recursion unavailable (no
  ``_MEMBERS_BY_LANG`` entry, no body subnode, or no member-class
  children): fall back to ``RecursiveCharacterTextSplitter`` for that
  one node so the embedder isn't handed a 30KB function as one chunk.
- Tree-sitter import / grammar load fails entirely: caller falls
  through to its existing default splitter via the ``split_code``
  return-empty contract.

The function returns ``list[Document]`` so it slots in next to
``RecursiveCharacterTextSplitter.split_documents`` at the call site.
"""

from __future__ import annotations

import logging
import os
from typing import List, NamedTuple, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


# ─── Tree-sitter binding adapters ──────────────────────────────────────────
#
# The ``tree-sitter-language-pack`` shipped in our pinned environment
# (1.8.1) bundles a Rust-pyo3 binding that diverges from the upstream
# ``tree-sitter`` Python package's API in several ways:
#
#   - ``parser.parse(source)`` requires ``str``; standard binding takes
#     ``bytes``.
#   - ``tree.root_node`` is a method, not a property.
#   - Every node accessor is a method, not a property:
#     ``node.kind()`` (vs ``node.type``),
#     ``node.start_byte()`` (vs ``node.start_byte``),
#     ``node.start_position()`` returns a ``Point`` with ``.row`` /
#     ``.column`` (vs the standard ``node.start_point`` tuple).
#   - There is no ``node.children`` list -- iterate via
#     ``node.child_count()`` + ``node.child(i)``.
#
# Before these adapters every call to the original ``split_code`` was
# silently bailing out via the ``except Exception`` in the parse path
# (``parser.parse(bytes)`` raised ``TypeError`` immediately), so the KB
# ingest pipeline has been transparently degrading to the default
# character / markdown splitter for every code file. The adapters fix
# that and are written to be tolerant of *either* binding shape so a
# future upgrade to a more standard binding doesn't re-break the
# splitter.
def _adapter(node, attr_name: str):
    """Call ``node.attr`` if it's a method, return the value otherwise.

    Lets the rest of the module write ``_adapter(node, 'start_byte')``
    once and not care whether the binding exposes ``start_byte`` as a
    method (this binding) or a property (upstream binding).
    """
    val = getattr(node, attr_name)
    return val() if callable(val) else val


def _node_kind(node) -> str:
    """Return the node's type-name string."""
    try:
        return _adapter(node, 'kind')
    except AttributeError:
        return _adapter(node, 'type')


def _node_start_byte(node) -> int:
    return _adapter(node, 'start_byte')


def _node_end_byte(node) -> int:
    return _adapter(node, 'end_byte')


def _node_start_line(node) -> int:
    """1-indexed source line of the node's first character."""
    pos = _adapter(node, 'start_position') if hasattr(node, 'start_position') else _adapter(node, 'start_point')
    if pos is None:
        return 1
    row = pos.row if hasattr(pos, 'row') else pos[0]
    return row + 1


def _node_end_line(node) -> int:
    """1-indexed source line of the node's last character."""
    pos = _adapter(node, 'end_position') if hasattr(node, 'end_position') else _adapter(node, 'end_point')
    if pos is None:
        return 1
    row = pos.row if hasattr(pos, 'row') else pos[0]
    return row + 1


def _node_child_count(node) -> int:
    cc = getattr(node, 'child_count', None)
    if cc is not None:
        return cc() if callable(cc) else cc
    children = getattr(node, 'children', None)
    return len(children) if children is not None else 0


def _node_children(node):
    """Yield direct children, working for either binding shape.

    Standard binding exposes ``node.children`` as a list-like property;
    the pinned binding requires ``node.child(i)`` lookups.
    """
    children = getattr(node, 'children', None)
    if children is not None and not callable(children):
        for c in children:
            yield c
        return
    n = _node_child_count(node)
    for i in range(n):
        yield node.child(i)


def _node_child_by_field_name(node, name: str):
    try:
        return node.child_by_field_name(name)
    except Exception:
        return None


def _tree_root_node(tree):
    """Return the tree's root node regardless of property/method shape."""
    return _adapter(tree, 'root_node')


def _parser_parse(parser, text: str):
    """Parse ``text``; tolerant of bindings that want ``str`` vs ``bytes``.

    Tries ``str`` first (the current pinned binding) and falls back to
    UTF-8 bytes on ``TypeError`` (standard upstream binding). Other
    exceptions propagate so they can be caught by the splitter's
    return-empty contract.
    """
    try:
        return parser.parse(text)
    except TypeError:
        return parser.parse(text.encode('utf-8'))


# File extension → tree-sitter language name. The pack uses canonical
# names (``python`` not ``py``, ``c_sharp`` not ``cs``); we map the
# common file extensions to those names. Extensions not in this map
# fall through to the caller's default splitter -- we deliberately
# don't auto-detect ("source code looking" content in a .txt file
# stays prose-chunked).
_EXT_TO_LANG: dict[str, str] = {
    '.py': 'python',
    '.pyi': 'python',
    '.ts': 'typescript',
    '.tsx': 'tsx',
    '.js': 'javascript',
    '.jsx': 'javascript',
    '.mjs': 'javascript',
    '.cjs': 'javascript',
    '.rs': 'rust',
    '.go': 'go',
    '.java': 'java',
    '.kt': 'kotlin',
    '.kts': 'kotlin',
    '.scala': 'scala',
    '.sc': 'scala',
    '.c': 'c',
    '.h': 'c',
    '.cpp': 'cpp',
    '.cc': 'cpp',
    '.cxx': 'cpp',
    '.hpp': 'cpp',
    '.hxx': 'cpp',
    '.cs': 'csharp',
    '.rb': 'ruby',
    '.php': 'php',
    '.swift': 'swift',
    '.lua': 'lua',
    '.gd': 'gdscript',
    '.dart': 'dart',
    '.zig': 'zig',
    '.ex': 'elixir',
    '.exs': 'elixir',
    '.erl': 'erlang',
    '.hs': 'haskell',
    '.ml': 'ocaml',
    '.fs': 'fsharp',
    '.fsx': 'fsharp',
    '.clj': 'clojure',
    '.cljs': 'clojure',
    '.r': 'r',
    '.jl': 'julia',
    '.nim': 'nim',
    '.sol': 'solidity',
    '.sh': 'bash',
    '.bash': 'bash',
    '.zsh': 'bash',
}


# ─── Per-language splittable node sets ──────────────────────────────────────
#
# Exact ``node.type`` strings that should each become their own chunk for
# the given tree-sitter grammar. Exact-match (not substring) so that
# e.g. ``lexical_declaration`` doesn't promote every top-level
# ``const x = 1`` into its own chunk in JS/TS.
#
# Languages without an entry fall through to ``_FALLBACK_DEFINITION_PATTERNS``
# (substring match) so that grammars we haven't enumerated still produce
# *some* useful splitting rather than collapsing to one giant preamble.
#
# When the grammar wraps a definition in a parent (e.g. JS
# ``export_statement`` wrapping a ``function_declaration``), include the
# *wrapper* type so the chunk text preserves the ``export`` keyword and
# the wrapper's modifiers.
#
# Method/constructor/property nodes that live *inside* a class body are
# deliberately omitted -- this splitter only walks top-level children, so
# they never appear here anyway. Per-class method recursion is tracked as
# a separate roadmap item (see ``docs/CODE_SAFETY_PIPELINE.md`` Stage 1
# improvement #4).
_DEFINITIONS_BY_LANG: dict[str, frozenset[str]] = {
    'python': frozenset({
        'function_definition',
        'class_definition',
        'decorated_definition',  # wraps function/class + @decorators
    }),
    'javascript': frozenset({
        'function_declaration',
        'generator_function_declaration',
        'class_declaration',
        'export_statement',  # `export function foo() {}` lives here
    }),
    'typescript': frozenset({
        'function_declaration',
        'generator_function_declaration',
        'class_declaration',
        'abstract_class_declaration',
        'interface_declaration',
        'type_alias_declaration',
        'enum_declaration',
        'module_declaration',  # `namespace Foo { ... }` (legacy)
        'internal_module',
        'function_signature',  # ambient `.d.ts` declarations
        'export_statement',
    }),
    'tsx': frozenset({
        'function_declaration',
        'generator_function_declaration',
        'class_declaration',
        'abstract_class_declaration',
        'interface_declaration',
        'type_alias_declaration',
        'enum_declaration',
        'module_declaration',
        'internal_module',
        'function_signature',
        'export_statement',
    }),
    'rust': frozenset({
        'function_item',
        'impl_item',
        'struct_item',
        'enum_item',
        'trait_item',
        'mod_item',
        'union_item',
        'type_item',
        'macro_definition',
    }),
    'go': frozenset({
        'function_declaration',
        'method_declaration',
        'type_declaration',  # struct/interface/alias all live here
    }),
    'java': frozenset({
        'class_declaration',
        'interface_declaration',
        'enum_declaration',
        'record_declaration',
        'annotation_type_declaration',
    }),
    'kotlin': frozenset({
        'function_declaration',
        'class_declaration',
        'object_declaration',
        'type_alias',
    }),
    'scala': frozenset({
        'function_definition',
        'class_definition',
        'object_definition',
        'trait_definition',
        'type_definition',
        'enum_definition',  # Scala 3
        'given_definition',  # Scala 3
    }),
    'c': frozenset({
        'function_definition',
        'type_definition',
        'preproc_function_def',
    }),
    'cpp': frozenset({
        'function_definition',
        'class_specifier',
        'namespace_definition',
        'template_declaration',
        'type_definition',
    }),
    # ── C# (THE focus -- WinUI / MVVM Toolkit code lands here) ──────────
    # Covers both classic block namespaces (``namespace Foo { ... }``) and
    # file-scoped namespaces (``namespace Foo;`` introduced in C# 10).
    # Includes records (C# 9+) and record structs (C# 10+).
    #
    # ``property_declaration`` / ``method_declaration`` / etc. are not in
    # this set because they live inside a containing type, not at the
    # top level. Splitting a class into its members is the v2 work
    # tracked as Stage 1 improvement #4.
    #
    # ``global_attribute`` (assembly-level attributes like
    # ``[assembly: AssemblyVersion("1.0")]``) is intentionally left as
    # preamble -- it's not a definition to navigate to and grouping with
    # ``using`` directives keeps citations cleaner.
    'csharp': frozenset({
        'class_declaration',
        'interface_declaration',
        'struct_declaration',
        'enum_declaration',
        'record_declaration',
        'record_struct_declaration',
        'delegate_declaration',
        'namespace_declaration',
        'file_scoped_namespace_declaration',
    }),
    'ruby': frozenset({
        'class',
        'module',
        'method',
        'singleton_method',
    }),
    'php': frozenset({
        'function_definition',
        'class_declaration',
        'interface_declaration',
        'trait_declaration',
        'enum_declaration',
    }),
    'swift': frozenset({
        'function_declaration',
        'class_declaration',
        'protocol_declaration',
        'struct_declaration',
        'enum_declaration',
        'extension_declaration',
    }),
    'lua': frozenset({
        'function_declaration',
        'function_definition',
        'local_function',
    }),
    'gdscript': frozenset({
        'function_definition',
        'class_definition',
    }),
    'dart': frozenset({
        'function_signature',
        'class_definition',
        'mixin_declaration',
        'extension_declaration',
        'enum_declaration',
    }),
    'zig': frozenset({
        'function_declaration',
        'variable_declaration',  # `pub const Foo = struct {...}` lives here
    }),
    'erlang': frozenset({
        'function_clause',
        'attribute',
    }),
    'haskell': frozenset({
        'function',
        'signature',
        'data_declaration',
        'newtype_declaration',
        'type_alias',
        'class_declaration',
        'instance_declaration',
    }),
    'ocaml': frozenset({
        'value_definition',
        'module_definition',
        'type_definition',
        'class_definition',
        'module_type_definition',
    }),
    'solidity': frozenset({
        'function_definition',
        'contract_declaration',
        'interface_declaration',
        'library_declaration',
        'struct_declaration',
        'enum_declaration',
        'event_definition',
        'modifier_definition',
    }),
    'bash': frozenset({
        'function_definition',
    }),
    'julia': frozenset({
        'function_definition',
        'short_function_definition',
        'macro_definition',
        'struct_definition',
        'abstract_definition',
        'primitive_definition',
        'module_definition',
    }),
    'nim': frozenset({
        'proc_declaration',
        'func_declaration',
        'method_declaration',
        'iterator_declaration',
        'template_declaration',
        'macro_declaration',
        'type_section',
    }),
}


# Substring patterns used ONLY for languages without an explicit entry in
# ``_DEFINITIONS_BY_LANG``. Deliberately small -- only patterns where the
# substring rarely false-positives across grammars. New languages should
# graduate to ``_DEFINITIONS_BY_LANG`` with exact node types rather than
# leaning on these.
_FALLBACK_DEFINITION_PATTERNS: Tuple[str, ...] = (
    'function_definition',
    'function_declaration',
    'class_definition',
    'class_declaration',
    'interface_declaration',
)


# ─── Per-language prefix node sets ──────────────────────────────────────────
#
# Node types that, when they appear as a top-level sibling immediately
# *before* a splittable definition node, should be attached to that
# definition's chunk rather than dumped into the preamble.
#
# Concretely this is what catches:
#   - leading ``//``, ``///``, ``/* */`` comments above a function
#   - Java / Kotlin / Swift / C# attributes & annotations
#   - Rust ``#[attribute_item]``
#
# Python is empty because the grammar already wraps decorators inside
# ``decorated_definition``, so they're absorbed into the chunk natively.
#
# Bare comments / attributes that have no following definition (e.g. a
# trailing license footer) fall through to the preamble -- the rule is
# "attach if a definition follows immediately, otherwise treat as
# floating content."
_PREFIX_NODES_BY_LANG: dict[str, frozenset[str]] = {
    'python': frozenset({'comment'}),
    'javascript': frozenset({'comment'}),
    'typescript': frozenset({'comment'}),
    'tsx': frozenset({'comment'}),
    'rust': frozenset({
        'line_comment',
        'block_comment',
        'attribute_item',  # `#[derive(...)]`, `#[test]`, `#[cfg(...)]`
        'inner_attribute_item',
    }),
    'go': frozenset({'comment'}),
    'java': frozenset({
        'line_comment',
        'block_comment',
        'marker_annotation',  # `@Override`
        'annotation',  # `@SuppressWarnings("foo")`
    }),
    'c': frozenset({'comment'}),
    'cpp': frozenset({'comment'}),
    # ── C# prefix gluing ────────────────────────────────────────────────
    # WinUI / MVVM Toolkit code routinely puts:
    #   /// <summary>...</summary>      ← `comment` (XML doc trivia)
    #   /// <param name="x">...</param>
    #   [ObservableProperty]            ← `attribute_list`
    #   [RelayCommand]
    #   [DllImport("user32.dll")]
    #   [Authorize(Roles = "Admin")]
    # ...on the lines immediately above a class / property / method /
    # field. The tree-sitter-c-sharp grammar exposes XML doc comments as
    # ``comment`` tokens (same node type as ``//``), and attribute groups
    # as ``attribute_list`` (one node per ``[...]`` group; multiple
    # groups stack as siblings). Both are captured here so the whole
    # decorated block lands in the chunk that owns the symbol.
    #
    # ``preprocessor_call`` (#region / #if / #pragma) is deliberately
    # NOT a prefix node -- those are typically file- or block-scoped
    # markers, not per-definition decorations, and gluing them to the
    # next definition would split #region pairs across chunks.
    'csharp': frozenset({
        'comment',
        'attribute_list',
    }),
    'kotlin': frozenset({
        'line_comment',
        'block_comment',
        'multiline_comment',
        'annotation',
    }),
    'scala': frozenset({
        'comment',
        'block_comment',
        'line_comment',
        'annotation',
    }),
    'ruby': frozenset({'comment'}),
    'php': frozenset({'comment'}),
    'swift': frozenset({
        'comment',
        'multiline_comment',
        'attribute',  # `@MainActor`, `@available(...)`
    }),
    'lua': frozenset({'comment'}),
    'gdscript': frozenset({'comment'}),
    'dart': frozenset({
        'comment',
        'documentation_comment',  # `///` dartdoc style
    }),
    'zig': frozenset({
        'line_comment',
        'doc_comment',
    }),
    'erlang': frozenset({'comment'}),
    'haskell': frozenset({'comment'}),
    'ocaml': frozenset({'comment'}),
    'solidity': frozenset({'comment'}),
    'bash': frozenset({'comment'}),
    'julia': frozenset({
        'line_comment',
        'block_comment',
    }),
    'nim': frozenset({
        'comment',
        'documentation_comment',
    }),
}


# ─── Per-language member node sets (for oversized-container recursion) ─────
#
# When a top-level definition (class / namespace / impl block / ...)
# exceeds ``chunk_size`` we recurse into its body and emit one chunk per
# member instead of letting the raw character splitter shred it. This
# map defines what counts as a "member" inside a container body.
#
# Members differ from top-level definitions: ``method_declaration``,
# ``property_declaration``, etc. only exist inside a containing type, so
# they live here and NOT in ``_DEFINITIONS_BY_LANG``.
#
# Nested type declarations (``class_declaration`` inside another class,
# etc.) are also included so the recursion works through namespace →
# class → nested-class chains. Recursion depth is bounded by
# ``_MAX_MEMBER_RECURSION_DEPTH`` to avoid pathological grammars.
#
# Languages without an entry skip member recursion entirely and fall
# back to character sharding for oversized definitions.
_MEMBERS_BY_LANG: dict[str, frozenset[str]] = {
    'python': frozenset({
        'function_definition',
        'decorated_definition',
        'class_definition',
    }),
    'javascript': frozenset({
        'method_definition',
        'field_definition',
        'class_declaration',  # nested
    }),
    'typescript': frozenset({
        'method_definition',
        'method_signature',  # interface / abstract methods
        'public_field_definition',  # TS class field syntax
        'property_signature',  # interface props
        'index_signature',
        'construct_signature',
        'call_signature',
        'abstract_method_signature',
        'class_declaration',
        'interface_declaration',
        'type_alias_declaration',
        'enum_declaration',
    }),
    'tsx': frozenset({
        'method_definition',
        'method_signature',
        'public_field_definition',
        'property_signature',
        'class_declaration',
        'interface_declaration',
    }),
    'rust': frozenset({
        'function_item',
        'function_signature_item',  # trait method signatures
        'associated_type',
        'const_item',
        'type_item',
    }),
    'java': frozenset({
        'method_declaration',
        'constructor_declaration',
        'field_declaration',
        'class_declaration',
        'interface_declaration',
        'enum_declaration',
        'record_declaration',
        'annotation_type_declaration',
    }),
    'kotlin': frozenset({
        'function_declaration',
        'property_declaration',
        'class_declaration',
        'object_declaration',
        'companion_object',
    }),
    # ── C# members (THE focus -- WinUI / MVVM Toolkit class bodies) ─────
    # Covers every member kind that should become its own chunk inside a
    # class / struct / interface / record body, plus the nested type
    # declarations so a namespace that wraps several classes splits into
    # those classes (which then split further into their own members if
    # any of them is still oversized, up to ``_MAX_MEMBER_RECURSION_DEPTH``).
    #
    # ``field_declaration`` is included because in WinUI/MVVM code a
    # field often carries the ``[ObservableProperty]`` attribute -- the
    # attribute prefix-gluing keeps the attribute attached, and emitting
    # the field as its own chunk lets that pattern be retrieved with the
    # right context.
    #
    # ``using_directive`` is NOT a member (top-level only in C#, and
    # they belong in the file preamble or the container preamble).
    'csharp': frozenset({
        'method_declaration',
        'constructor_declaration',
        'destructor_declaration',
        'operator_declaration',
        'conversion_operator_declaration',
        'property_declaration',
        'indexer_declaration',
        'event_declaration',
        'event_field_declaration',
        'field_declaration',
        # Nested types: lets namespace/class containers recurse one
        # level deeper into their type members.
        'class_declaration',
        'interface_declaration',
        'struct_declaration',
        'enum_declaration',
        'record_declaration',
        'record_struct_declaration',
        'delegate_declaration',
    }),
    'cpp': frozenset({
        'function_definition',
        'field_declaration',
        'friend_declaration',
        'class_specifier',  # nested
    }),
    'ruby': frozenset({
        'method',
        'singleton_method',
        'class',
        'module',
    }),
    'php': frozenset({
        'method_declaration',
        'property_declaration',
        'const_declaration',
        'class_declaration',
        'interface_declaration',
    }),
    'swift': frozenset({
        'function_declaration',
        'init_declaration',
        'subscript_declaration',
        'property_declaration',
        'class_declaration',
        'protocol_declaration',
        'enum_declaration',
        'struct_declaration',
    }),
    'go': frozenset({
        # Go has no class-style member syntax; the only members live
        # inside interfaces.
        'method_spec',
        'method_elem',
    }),
    'scala': frozenset({
        'function_definition',
        'value_definition',
        'variable_definition',
        'class_definition',
        'object_definition',
        'trait_definition',
        'type_definition',
    }),
    'dart': frozenset({
        'function_signature',
        'method_signature',
        'getter_signature',
        'setter_signature',
        'constructor_signature',
        'factory_constructor_signature',
    }),
}


# Tree-sitter node types that represent a container's body, used by
# ``_find_container_body`` as a fallback when ``child_by_field_name('body')``
# returns None. Different grammars name the body differently; this set
# covers the major ones. Order doesn't matter -- we scan the container's
# direct children once.
_KNOWN_BODY_TYPES: frozenset = frozenset({
    'block',                    # Python function/class body
    'class_body',               # JS/TS/Java/Kotlin/Swift
    'interface_body',           # Java
    'enum_body',                # Java/C#
    'enum_body_declaration',
    'object_body',              # Kotlin
    'declaration_list',         # C#/C++/Rust impl/mod
    'field_declaration_list',   # C/C++ struct/class, Rust struct
    'enum_variant_list',        # Rust enum
    'trait_body',               # Rust trait
    'impl_body',
    'method_spec_list',         # Go interface
    'struct_body',
    'protocol_body',            # Swift
    'enum_class_body',          # Swift
    'record_declaration_body',  # Java records
})


# Hard cap on recursion depth so a pathological grammar can't fan out
# unboundedly. 3 covers the common namespace → class → nested-class
# chain (depth 3 = method inside nested class inside class inside
# namespace) which is the deepest a realistic C# / Java / Kotlin file
# is likely to nest.
_MAX_MEMBER_RECURSION_DEPTH: int = 3


def _find_container_body(container_node):
    """Locate the body subnode of a container, or return None.

    Primary path: ``child_by_field_name('body')`` -- the canonical
    tree-sitter field for class/function/etc bodies, exposed by most
    well-maintained grammars (Python, C#, Java, JS/TS, Rust, ...).

    Fallback: scan direct children for any type in ``_KNOWN_BODY_TYPES``.
    Used for grammars that don't expose the body as a named field.
    """
    body = _node_child_by_field_name(container_node, 'body')
    if body is not None:
        return body
    try:
        for child in _node_children(container_node):
            if _node_kind(child) in _KNOWN_BODY_TYPES:
                return child
    except Exception:
        return None
    return None


def ext_to_language(filename: str) -> Optional[str]:
    """Return the tree-sitter language name for a filename, or ``None``.

    Lowercased extension match; multi-dot suffixes (e.g. ``foo.test.ts``)
    use only the trailing ``.ts``. Returns ``None`` for unknown or
    extensionless files so the caller can fall through to its default
    splitter.
    """
    _, ext = os.path.splitext(filename or '')
    if not ext:
        return None
    return _EXT_TO_LANG.get(ext.lower())


def _is_definition_node(node_type: str, language: str) -> bool:
    """Per-language exact match, with a substring fallback for unmapped grammars.

    Languages with an entry in ``_DEFINITIONS_BY_LANG`` use exact-set
    membership (no false positives from substring overlap). Languages
    without an entry fall back to ``_FALLBACK_DEFINITION_PATTERNS`` so
    that grammars we haven't enumerated still get *some* useful
    splitting rather than collapsing to one big preamble chunk.
    """
    exact = _DEFINITIONS_BY_LANG.get(language)
    if exact is not None:
        return node_type in exact
    return any(pat in node_type for pat in _FALLBACK_DEFINITION_PATTERNS)


def _is_prefix_node(node_type: str, language: str) -> bool:
    """True if ``node_type`` should glue forward onto the next definition.

    Languages without an entry in ``_PREFIX_NODES_BY_LANG`` get an empty
    set, i.e. no gluing -- comments stay in the preamble (the legacy
    behaviour before this lookup existed).
    """
    return node_type in _PREFIX_NODES_BY_LANG.get(language, frozenset())


def _byte_slice(source_bytes: bytes, node) -> str:
    """Decode a tree-sitter node's byte range back to a Python string.

    tree-sitter operates on UTF-8 bytes internally regardless of which
    binding shape exposes the offsets (str-input or bytes-input), so we
    always slice from the bytes-encoded source -- avoids surprises on
    multi-byte characters (CJK comments, emoji in identifiers).
    """
    return source_bytes[_node_start_byte(node):_node_end_byte(node)].decode(
        'utf-8', errors='replace',
    )


# A definition record returned by ``_collect_top_level``. ``node`` is the
# tree-sitter definition node itself (used for symbol name + the
# "primary" line number of the symbol). ``chunk_start_byte`` /
# ``chunk_start_line`` may extend earlier than the node when leading
# prefix siblings (comments / attributes / annotations) were glued on;
# they describe where the *emitted chunk text* actually begins in the
# source.
class _DefRecord(NamedTuple):
    node: object
    chunk_start_byte: int
    chunk_end_byte: int
    chunk_start_line: int  # 1-indexed; equals node.start_point[0]+1 when no prefix


def _walk_for_chunks(
    parent_node,
    source_bytes: bytes,
    is_splittable,  # Callable[[str], bool]
    is_prefix,      # Callable[[str], bool]
) -> Tuple[List[str], List[_DefRecord]]:
    """Generic AST walker: bucket ``parent_node.children`` into preamble + defs.

    Three buckets, in priority order per child:

    1. **Splittable definition** (per ``is_splittable``): emit as its
       own chunk, and absorb any *pending prefix* siblings into that
       chunk's byte range.
    2. **Prefix-class sibling** (per ``is_prefix``: comments, attribute
       lists, annotations): held in a pending buffer until we know
       whether a definition follows. If the next non-prefix sibling is
       a definition, the pending prefix attaches to it. If the next
       non-prefix sibling is a regular statement (or we hit EOF), the
       pending prefix flushes into the preamble.
    3. **Anything else** (imports, top-level constants, free statements,
       etc.): preamble. Any pending prefix flushes into the preamble
       first since this child has broken the run of decorations.

    This is what makes ``/// <summary>`` + ``[ObservableProperty]`` +
    ``public class Foo`` end up as one cohesive chunk in C# instead of
    the doc-comment + attribute getting orphaned in the preamble.

    Used from two places with different splittable/prefix predicates:

    - :func:`_collect_top_level` walks the module/file root.
    - :func:`_collect_members` walks a container's body when we recurse
      into an oversized class / struct / namespace.
    """
    preamble_pieces: List[str] = []
    definitions: List[_DefRecord] = []
    pending_prefix: List = []  # consecutive prefix-class siblings

    def _flush_pending_to_preamble() -> None:
        for n in pending_prefix:
            text = _byte_slice(source_bytes, n)
            if text.strip():
                preamble_pieces.append(text)
        pending_prefix.clear()

    for child in _node_children(parent_node):
        ctype = _node_kind(child)
        if is_splittable(ctype):
            if pending_prefix:
                chunk_start_byte = _node_start_byte(pending_prefix[0])
                chunk_start_line = _node_start_line(pending_prefix[0])
                pending_prefix.clear()
            else:
                chunk_start_byte = _node_start_byte(child)
                chunk_start_line = _node_start_line(child)
            definitions.append(_DefRecord(
                node=child,
                chunk_start_byte=chunk_start_byte,
                chunk_end_byte=_node_end_byte(child),
                chunk_start_line=chunk_start_line,
            ))
        elif is_prefix(ctype):
            pending_prefix.append(child)
        else:
            _flush_pending_to_preamble()
            text = _byte_slice(source_bytes, child)
            if text.strip():
                preamble_pieces.append(text)

    # EOF: any trailing prefix run with no following definition is just
    # floating content (license footer, trailing comment block) -- it
    # belongs in the preamble.
    _flush_pending_to_preamble()
    return preamble_pieces, definitions


def _collect_top_level(
    root,
    source_bytes: bytes,
    language: str,
) -> Tuple[List[str], List[_DefRecord]]:
    """Walk the module root for top-level definitions + a preamble."""
    return _walk_for_chunks(
        root, source_bytes,
        is_splittable=lambda t: _is_definition_node(t, language),
        is_prefix=lambda t: _is_prefix_node(t, language),
    )


def _collect_members(
    body,
    source_bytes: bytes,
    language: str,
) -> Tuple[List[str], List[_DefRecord]]:
    """Walk a container body (class / struct / namespace) for member defs.

    Returns ``([], [])`` when the language has no entry in
    ``_MEMBERS_BY_LANG`` -- callers should treat that as "no recursion
    possible" and fall through to character sharding.
    """
    members_set = _MEMBERS_BY_LANG.get(language)
    if not members_set:
        return [], []
    return _walk_for_chunks(
        body, source_bytes,
        is_splittable=lambda t: t in members_set,
        is_prefix=lambda t: _is_prefix_node(t, language),
    )


def _extract_symbol_name(node, source_bytes: bytes) -> str:
    """Best-effort symbol name from a tree-sitter definition node.

    Uses ``node.child_by_field_name('name')`` which is the canonical
    tree-sitter field for the identifier on most ``*_definition`` /
    ``*_declaration`` grammars. Returns ``''`` when the grammar doesn't
    expose a ``name`` field (some declarations are anonymous, and
    grammars vary) -- callers should treat empty-string as "unknown".
    """
    name_node = _node_child_by_field_name(node, 'name')
    if name_node is None:
        return ''
    try:
        return _byte_slice(source_bytes, name_node).strip()
    except Exception:
        return ''


def _split_oversized(
    text: str,
    metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """Fall back to RecursiveCharacterTextSplitter for a single oversized chunk.

    All shards inherit the parent node's metadata (including its AST
    line range) and additionally carry ``ast_kind='oversized_shard'``
    plus ``ast_oversized_shard_index`` / ``ast_oversized_shard_count``
    so the retrieval layer can stitch shards back together or display
    "part N of M" affordances in citation UI.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    pieces = splitter.split_text(text)
    count = len(pieces)
    docs: List[Document] = []
    for index, piece in enumerate(pieces):
        shard_metadata = {
            **metadata,
            'ast_kind': 'oversized_shard',
            'ast_oversized_shard_index': index,
            'ast_oversized_shard_count': count,
        }
        docs.append(Document(page_content=piece, metadata=shard_metadata))
    return docs


def _emit_chunk(
    text: str,
    base_metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    sink: List[Document],
) -> None:
    """Emit one logical chunk; cascade to character splitter if oversized.

    Note: this function does *not* attempt member-level recursion for
    oversized definitions -- that path is invoked explicitly by
    :func:`split_code` (via :func:`_split_by_members`) before falling
    here, so this remains the leaf "definitely chunk this text now"
    helper used by the preamble + fallback paths.
    """
    if not text.strip():
        return
    if len(text) <= chunk_size:
        sink.append(Document(page_content=text, metadata={**base_metadata}))
        return
    sink.extend(_split_oversized(text, base_metadata, chunk_size, chunk_overlap))


def _split_by_members(
    container_node,
    container_chunk_start_byte: int,
    container_chunk_start_line: int,
    source_bytes: bytes,
    language: str,
    parent_metadata: dict,
    chunk_size: int,
    chunk_overlap: int,
    *,
    depth: int = 0,
) -> List[Document]:
    """Recurse into a container body and emit one chunk per member.

    Returns ``[]`` when recursion is impossible or unhelpful -- caller
    (either :func:`split_code` or this function recursing) should fall
    back to character sharding via :func:`_split_oversized`. Conditions
    that return ``[]``:

    - Recursion depth has hit ``_MAX_MEMBER_RECURSION_DEPTH``.
    - The language has no entry in ``_MEMBERS_BY_LANG``.
    - The container has no recognizable body subnode
      (``_find_container_body`` returned None).
    - The body has no member-class children to split on.

    When recursion succeeds the returned list begins with a "container
    preamble" Document that carries the container's signature, any
    leading decoration (XML doc comments / attributes / annotations),
    the opening brace, and any non-member content up to the first
    member. The preamble keeps the parent's ``ast_kind`` / ``ast_symbol``
    since it *is* the parent's header -- consumers can identify it as
    the preamble vs a real member by checking
    ``ast_kind == ast_parent_kind`` (the preamble has no parent fields).

    Each member chunk carries:

    - ``ast_kind`` / ``ast_symbol`` -- the member's own type + name.
    - ``ast_start_line`` / ``ast_end_line`` / ``ast_symbol_line`` --
      computed for the member, with the same chunk-vs-symbol distinction
      as top-level chunks.
    - ``ast_parent_kind`` / ``ast_parent_symbol`` -- the immediate
      enclosing container's type + name. Lets retrieval filter "all
      methods of class Foo" without re-parsing.

    Recursion: when a member is *itself* oversized (e.g. a nested class
    that's also huge), this function recurses into it before falling
    back to character sharding. ``ast_parent_*`` fields always refer to
    the *immediate* enclosing container, not the full ancestor chain --
    tracking the full chain is left for a future enhancement.
    """
    if depth >= _MAX_MEMBER_RECURSION_DEPTH:
        return []

    members_set = _MEMBERS_BY_LANG.get(language)
    if not members_set:
        return []

    body = _find_container_body(container_node)
    if body is None:
        return []

    _, member_records = _collect_members(body, source_bytes, language)
    if not member_records:
        return []

    docs: List[Document] = []
    first = member_records[0]

    # ── Container preamble ─────────────────────────────────────────────
    # Spans from the parent chunk's start (which may itself include
    # attributes / doc-comments glued onto the container by the top-level
    # walker) up to the first member's chunk start. Captures:
    #   - the container's own decoration (`/// <summary>`, `[Authorize]`, ...)
    #   - the container signature (`public class Foo : IBar`)
    #   - the opening brace
    #   - any body content before the first member (fields, region
    #     markers, free comments that didn't glue forward)
    preamble_text = source_bytes[
        container_chunk_start_byte:first.chunk_start_byte
    ].decode('utf-8', errors='replace').rstrip()
    if preamble_text.strip():
        preamble_end_line = container_chunk_start_line + max(preamble_text.count('\n'), 0)
        # The container preamble *is* the parent itself (narrowed to its
        # header). Keep parent's ast_kind / ast_symbol / ast_symbol_line
        # -- only shrink ast_end_line to match the preamble's extent.
        preamble_md = {
            **parent_metadata,
            'ast_end_line': preamble_end_line,
        }
        _emit_chunk(preamble_text, preamble_md, chunk_size, chunk_overlap, docs)

    parent_symbol = parent_metadata.get('ast_symbol', '')
    parent_kind = parent_metadata.get('ast_kind', '')

    for rec in member_records:
        member_node = rec.node
        member_text = source_bytes[
            rec.chunk_start_byte:rec.chunk_end_byte
        ].decode('utf-8', errors='replace')
        member_md = {
            # Inherit code_split_language + caller's base_metadata + any
            # ast_parent_* keys from a deeper ancestor. The next four
            # overrides replace the parent's ast_* fields with the
            # member's, which is what we want -- this chunk *is* the
            # member, not the parent.
            **parent_metadata,
            'ast_kind': _node_kind(member_node),
            'ast_symbol': _extract_symbol_name(member_node, source_bytes),
            'ast_start_line': rec.chunk_start_line,
            'ast_end_line': _node_end_line(member_node),
            'ast_symbol_line': _node_start_line(member_node),
            'ast_parent_kind': parent_kind,
            'ast_parent_symbol': parent_symbol,
        }
        if len(member_text) <= chunk_size:
            docs.append(Document(page_content=member_text, metadata=member_md))
            continue
        # Member is itself oversized -- try one more level of recursion
        # before giving up to the character splitter. Covers e.g. a
        # nested class with many methods, or a namespace containing a
        # very large class.
        nested = _split_by_members(
            member_node,
            rec.chunk_start_byte,
            rec.chunk_start_line,
            source_bytes,
            language,
            member_md,
            chunk_size,
            chunk_overlap,
            depth=depth + 1,
        )
        if nested:
            docs.extend(nested)
        else:
            docs.extend(_split_oversized(member_text, member_md, chunk_size, chunk_overlap))

    return docs


def split_code(
    text: str,
    language: Optional[str],
    *,
    chunk_size: int = 2000,
    chunk_overlap: int = 150,
    base_metadata: Optional[dict] = None,
) -> List[Document]:
    """Split ``text`` into chunks aligned to tree-sitter definition boundaries.

    On any failure (unknown language, parser load failure, parse crash),
    returns an empty list -- the CALLER is expected to check for empty
    and fall through to ``RecursiveCharacterTextSplitter`` in that case.
    This contract keeps the chunker importable on systems that haven't
    yet installed ``tree-sitter-language-pack`` (e.g. early bootstraps).

    Per-chunk metadata
    ------------------

    Every returned :class:`~langchain_core.documents.Document` carries
    these keys on top of whatever ``base_metadata`` the caller passes
    in (caller keys are never overwritten -- the splitter only
    ``setdefault``-s for ``code_split_language``):

    - ``code_split_language`` -- the tree-sitter language name
      (``'python'``, ``'rust'``, ``'tsx'`` ...).
    - ``ast_symbol`` -- the function / class / method name, decoded
      from ``node.child_by_field_name('name')``. Empty string for the
      preamble chunk, and empty string for definition nodes whose
      grammar doesn't expose a ``name`` field.
    - ``ast_kind`` -- the tree-sitter ``node.type`` string for
      definition chunks (e.g. ``'function_definition'``,
      ``'class_declaration'``), the literal ``'preamble'`` for the
      preamble chunk, and ``'oversized_shard'`` for character-split
      fragments emitted when a single definition exceeded
      ``chunk_size``.
    - ``ast_start_line`` -- 1-indexed source line where the *chunk
      text* begins. For definition chunks this may extend earlier than
      the symbol itself when leading comments / attributes /
      annotations were glued on by the prefix-aware collector (e.g. a
      ``/// <summary>`` block and ``[ObservableProperty]`` attribute
      above a C# property end up in the same chunk and the start line
      is the doc comment's line, not the property's). For the preamble
      chunk this is always ``1``.
    - ``ast_end_line`` -- 1-indexed source line where the symbol's own
      definition ends (``node.end_point[0] + 1``). For the preamble it
      is the number of lines in the concatenated preamble text.
    - ``ast_symbol_line`` -- 1-indexed source line where the symbol
      itself starts (``node.start_point[0] + 1``); always >=
      ``ast_start_line`` and equal to it when no prefix was glued on.
      Use this for "go to definition" affordances; use
      ``ast_start_line`` for "the chunk starts here". On the preamble
      chunk and the fallback whole-file chunk this is ``1``.
    - ``ast_oversized_shard_index`` / ``ast_oversized_shard_count`` --
      only present on shards from :func:`_split_oversized`; let the
      retrieval layer rebuild or label shard groups. Shards inherit
      the parent definition's full line range, so citations on a
      shard still point at the whole definition.
    - ``ast_parent_kind`` / ``ast_parent_symbol`` -- only present on
      chunks emitted by :func:`_split_by_members` (the recursive
      "split an oversized container into one chunk per method /
      property / nested type" path). They name the *immediate*
      enclosing container's ``ast_kind`` and ``ast_symbol``, so
      retrieval can filter or group e.g. "all methods of
      ``UserService``" without re-parsing the file. The container's
      preamble chunk -- which IS the container's own header narrowed
      to its signature + opening brace + leading fields -- does *not*
      carry these fields (you can detect it as
      ``ast_parent_kind not in metadata``).
    """
    if not text or not text.strip():
        return []
    if not language:
        return []

    try:
        # Lazy import: tree-sitter wheels are ~10-30MB and we don't want
        # to pay the import cost on KB ingest paths that never touch
        # source files.
        from tree_sitter_language_pack import get_parser  # type: ignore
    except Exception as e:
        log.debug('code_splitter: tree-sitter-language-pack unavailable: %s', e)
        return []

    try:
        parser = get_parser(language)
    except Exception as e:
        log.debug(
            'code_splitter: no parser for language=%r: %s; falling back',
            language, e,
        )
        return []

    source_bytes = text.encode('utf-8')
    try:
        tree = _parser_parse(parser, text)
    except Exception as e:
        log.debug('code_splitter: parse failed for language=%r: %s', language, e)
        return []

    root = _tree_root_node(tree)
    if root is None or _node_child_count(root) == 0:
        # Parse produced nothing usable -- bail to the caller's fallback.
        # We deliberately don't check for ERROR nodes here: a partial
        # parse is still useful (tree-sitter is intentionally tolerant
        # of malformed input and will mark the bad bits as ERROR
        # nodes); as long as we have at least one child we can produce
        # chunks that the embedder can use.
        return []

    metadata = dict(base_metadata or {})
    metadata.setdefault('code_split_language', language)

    chunks: List[Document] = []
    preamble_pieces, definitions = _collect_top_level(root, source_bytes, language)

    if preamble_pieces:
        # Single preamble chunk -- the goal is to keep imports together
        # with any module-level docstrings / constants so the embedder
        # sees them as one coherent thing.
        preamble_text = '\n'.join(preamble_pieces).strip()
        # Line range is derived from the concatenated text rather than
        # the underlying nodes so it always agrees with what the
        # embedder actually saw -- the join collapses blank lines
        # between original AST siblings, so original start/end points
        # would be off-by-many.
        preamble_lines = max(preamble_text.count('\n') + 1, 1)
        preamble_metadata = {
            **metadata,
            'ast_symbol': '',
            'ast_kind': 'preamble',
            'ast_start_line': 1,
            'ast_end_line': preamble_lines,
            # Preamble has no single "symbol" -- expose the same value
            # for ``ast_symbol_line`` as ``ast_start_line`` so consumers
            # can rely on the field being present.
            'ast_symbol_line': 1,
        }
        _emit_chunk(preamble_text, preamble_metadata, chunk_size, chunk_overlap, chunks)

    for rec in definitions:
        node = rec.node
        node_text = source_bytes[rec.chunk_start_byte:rec.chunk_end_byte].decode(
            'utf-8', errors='replace',
        )
        node_metadata = {
            **metadata,
            'ast_symbol': _extract_symbol_name(node, source_bytes),
            'ast_kind': _node_kind(node),
            # ``ast_start_line`` reflects where the *chunk* starts in the
            # source, which may extend earlier than the symbol itself
            # when leading comments / attributes / annotations were
            # glued on by ``_collect_top_level``. ``ast_symbol_line``
            # points at the symbol's own first line so UIs can render
            # "go to definition" at the right location regardless.
            'ast_start_line': rec.chunk_start_line,
            'ast_end_line': _node_end_line(node),
            'ast_symbol_line': _node_start_line(node),
        }
        # Oversized definition: try member-level recursion first so the
        # embedder gets coherent method/property chunks instead of
        # character-sharded soup. Falls through to character sharding
        # below if recursion isn't possible (no body, no members,
        # unmapped language, or depth-capped).
        if len(node_text) > chunk_size:
            member_chunks = _split_by_members(
                node,
                rec.chunk_start_byte,
                rec.chunk_start_line,
                source_bytes,
                language,
                node_metadata,
                chunk_size,
                chunk_overlap,
            )
            if member_chunks:
                chunks.extend(member_chunks)
                continue
        _emit_chunk(node_text, node_metadata, chunk_size, chunk_overlap, chunks)

    if not chunks:
        # AST walk produced nothing (e.g. a file that's all top-level
        # statements in a language whose grammar we didn't classify as
        # a "definition" anywhere). Emit the full text as one chunk so
        # the caller's downstream still sees it; the character splitter
        # below will subdivide further if it exceeds chunk_size.
        fallback_lines = max(text.count('\n') + 1, 1)
        fallback_metadata = {
            **metadata,
            'ast_symbol': '',
            'ast_kind': 'preamble',
            'ast_start_line': 1,
            'ast_end_line': fallback_lines,
            'ast_symbol_line': 1,
        }
        _emit_chunk(text, fallback_metadata, chunk_size, chunk_overlap, chunks)

    return chunks


def is_code_extension(filename: str) -> bool:
    """Cheap predicate used at the call site to decide which splitter to invoke."""
    return ext_to_language(filename) is not None


__all__ = ['split_code', 'ext_to_language', 'is_code_extension']
