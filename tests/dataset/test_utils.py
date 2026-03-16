"""Tests for otter.dataset.utils module."""

from otter.dataset.utils import extract_code


class TestExtractCode:
    """Test extract_code function."""

    def test_extracts_python_code_block(self):
        response = 'Here is the code:\n```python\ndef foo():\n    return 1\n```\nDone.'
        result = extract_code(response)
        assert result == "def foo():\n    return 1"

    def test_extracts_generic_code_block(self):
        response = 'Code:\n```\ndef bar():\n    pass\n```'
        result = extract_code(response)
        assert result == "def bar():\n    pass"

    def test_no_code_block_returns_stripped_text(self):
        response = "  def baz(): pass  "
        result = extract_code(response)
        assert result == "def baz(): pass"

    def test_empty_string(self):
        result = extract_code("")
        assert result == ""

    def test_multiple_code_blocks_returns_first(self):
        response = '```python\nfirst\n```\n\n```python\nsecond\n```'
        result = extract_code(response)
        assert result == "first"

    def test_code_block_with_extra_whitespace(self):
        response = '```python\n\n  def foo():\n      return 42\n\n```'
        result = extract_code(response)
        assert result == "def foo():\n      return 42"

    def test_only_backticks_no_content(self):
        response = '```python\n```'
        result = extract_code(response)
        assert result == ""
