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


def test_header_drops_hashes():
    p = _plain("## 标题二")
    assert p.strip().startswith("标题二") and "#" not in p


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
