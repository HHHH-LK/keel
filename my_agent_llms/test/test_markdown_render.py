"""自定义极简 markdown 渲染器(返回 rich.Text,容忍残缺,永不抛)。"""
from rich.text import Text
from my_agent_llms.cli.markdown_render import render_markdown


def _plain(text: str, width: int = 80) -> str:
    r = render_markdown(text, width)
    assert isinstance(r, Text)
    return r.plain


def test_bold_strips_markers():
    r = render_markdown("这是 **重点** 哦", 80)
    assert "**" not in r.plain and "重点" in r.plain
    assert any("bold" in str(s.style) for s in r.spans)


def test_inline_code_and_italic_and_strike_and_link():
    assert "`" not in _plain("用 `pip` 装")
    assert "*" not in _plain("*斜体*")
    assert "~~" not in _plain("~~删~~") and "删" in _plain("~~删~~")
    p = _plain("见 [文档](http://x)")
    assert "文档" in p


def test_lowercase_english_link_not_broken():
    # 回归:链接文字以小写英文开头不能让整行 markup 破掉/露出字面 tag
    r = render_markdown("see [click here](http://x.com) now", 80)
    p = r.plain
    assert "click here" in p
    assert "[" not in p and "]" not in p          # 不露 \[ 或 markup tag
    assert "bright_magenta" not in p              # 不露字面 style 名


def test_link_text_default_url_dim_no_accent():
    # 像 Claude Code:链接文字用默认色(不再亮品红),URL 暗色
    r = render_markdown("见 [文档](http://x.com)", 80)
    assert "文档" in r.plain and "http://x.com" in r.plain
    assert not any("magenta" in str(s.style).lower() for s in r.spans)  # 文字去品红
    assert any("bright_black" in str(s.style).lower() for s in r.spans)  # URL 仍暗色


def test_header_drops_hashes():
    p = _plain("## 标题二")
    assert p.strip().startswith("标题二") and "#" not in p


def test_header_is_bold_without_accent_color():
    # 像 Claude Code:标题只 bold,去掉亮品红强调色
    r = render_markdown("## 标题二", 80)
    assert any("bold" in str(s.style) for s in r.spans)
    assert not any("magenta" in str(s.style).lower() for s in r.spans)


def test_bullet_list_uses_dot():
    p = _plain("- 苹果\n- 香蕉")
    assert "• 苹果" in p and "• 香蕉" in p and "- 苹果" not in p


def test_ordered_list_keeps_number():
    p = _plain("1. 第一\n2. 第二")
    assert "1. 第一" in p and "2. 第二" in p


def test_blockquote_marker():
    p = _plain("> 引用句")
    assert "引用句" in p and "▏" in p


def test_hr_is_short_dim_line():
    p = _plain("---")
    assert "─" in p and "-" not in p.replace("─", "")


def test_unterminated_bold_renders_literal():
    r = render_markdown("前缀 **未闭合", 80)
    assert "**未闭合" in r.plain


def test_never_raises_on_garbage():
    for s in ["", "*", "**", "`", "[", "###", "> ", "|"]:
        assert isinstance(render_markdown(s, 80), Text)


def test_code_block_strips_fences_and_keeps_code():
    p = _plain("```python\nprint(1)\n```", 80)
    assert "print(1)" in p and "```" not in p


def test_unclosed_code_fence_renders_in_progress():
    p = _plain("```py\nx = 1\ny = 2", 80)
    assert "x = 1" in p and "y = 2" in p and "```" not in p


def test_table_aligned_header_bold_no_box():
    md = "| 名 | 值 |\n|---|---|\n| a | 1 |\n| bb | 22 |"
    r = render_markdown(md, 80)
    p = r.plain
    assert "名" in p and "值" in p and "a" in p and "bb" in p
    assert "|" not in p and "---" not in p
    assert any("bold" in str(s.style) for s in r.spans)
    lines = [ln for ln in p.split("\n") if ln.strip()]
    assert len(lines) >= 3


def test_partial_table_renders_available_rows():
    p = render_markdown("| 名 | 值 |", 80).plain
    assert "名" in p and "值" in p and "|" not in p
