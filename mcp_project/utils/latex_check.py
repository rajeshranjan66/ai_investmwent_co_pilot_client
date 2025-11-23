# import re
# def is_latex(content):
#     # Reject if there are Markdown headings (lines starting with #)
#     if re.search(r"^#+\s", content, re.MULTILINE):
#         return False
#     # Require at least one LaTeX environment or math mode
#     return bool(re.search(r"\\begin\{.*?\}|\\\[|\\\(|\$", content))
import streamlit as st

import re
from textwrap import dedent

# def is_latex(content):
#     # Reject if there are Markdown headings (lines starting with #)
#     if re.search(r"^#+\s", content, re.MULTILINE):
#         return False
#     # Require at least one LaTeX environment or math mode with delimiters
#     return bool(re.search(r"\\begin\{.*?\}|\\\[|\\\(|\$.+\$", content, re.DOTALL))

# Only use st.latex if the content is pure LaTeX (not Markdown with $)


def is_latex(content):
    """
    More robust detection that excludes table content
    """
    # Clear indicators of table content (not LaTeX)
    table_indicators = [
        r"^#\s",  # Table headers like "# Article & Source"
        r"\t",  # Tab characters
        r"\|\s*[^|]+\s*\|",  # Pipe tables
        r"<table>|<tr>|<td>|<th>",  # HTML tables
        r"&lt;table&gt;|&lt;tr&gt;|&lt;td&gt;|&lt;th&gt;",  # HTML entities
        r"Article & Source",  # Common table headers
        r"Key Points.*Sentiment.*Potential Impact",  # Table structure
    ]

    for pattern in table_indicators:
        if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
            return False

    # Now check for actual LaTeX (require stronger evidence)
    latex_indicators = [
        r"\\begin\{.*?\}.*?\\end\{.*?\}",  # LaTeX environments
        r"\\\[.*?\\\]",  # Display math
        r"\\\(.*?\\\)",  # Inline math
        r"\$[^$]+\$",  # Dollar math with content
        r"\\frac\{.*?\}\{.*?\}",  # Fractions
        r"\\sum_\{.*?\}^\{.*?\}",  # Sums
        r"\\int_\{.*\}^\{.*\}",  # Integrals
    ]

    latex_count = sum(1 for pattern in latex_indicators if re.search(pattern, content, re.DOTALL))
    return latex_count >= 1  # At least one strong LaTeX indicator


def is_pure_latex(content):
    # No Markdown formatting, only LaTeX math delimiters
    return (
        not re.search(r"[*_#\-]", content) and
        bool(re.search(r"\\begin\{.*?\}|\\\[|\\\(|\$.+\$", content, re.DOTALL))
    )

def convert_latex_display_math(content):
    # Replace \[ ... \] with $$ ... $$
    return re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", content, flags=re.DOTALL)




def auto_wrap_latex_math(content):
    # Wrap [ ... ] containing LaTeX commands in $...$
    def bracket_replacer(match):
        expr = match.group(1)
        if re.match(r"\$.*\$", expr.strip()):
            return f"[{expr}]"
        return f"$[{expr.strip()}]$"
    content = re.sub(
        r"\[(.*?\\(?:times|div|frac|cdot|pm|sqrt|sum|prod|int|log|sin|cos|tan).*?)\]",
        bracket_replacer,
        content
    )
    # Fallback: wrap any line with LaTeX commands not already wrapped
    def line_replacer(match):
        expr = match.group(1)
        if re.match(r"\$.*\$", expr.strip()):
            return expr
        return f"${expr.strip()}$"
    content = re.sub(
        r"([^\$]*\\(?:times|div|frac|cdot|pm|sqrt|sum|prod|int|log|sin|cos|tan)[^$]*)",
        line_replacer,
        content
    )
    return content

def clean_latex(llm_text: str) -> str:
    """
    Cleans and fixes common LaTeX issues from LLM outputs.
    """
    # Remove extra dollar signs if doubled
    text = llm_text.replace("$$", "$")

    # Fix broken commands like \ext to \text, \ac to \frac, \ight to \right
    replacements = {
        r"\\ext": r"\\text",
        r"\\ac": r"\\frac",
        r"\\ight": r"\\right",
        r"\\eft": r"\\left",
        r"\\pprox": r"\\approx",
    }
    for bad, good in replacements.items():
        text = re.sub(bad, good, text)

    # Remove unwanted \n inside LaTeX blocks
    text = re.sub(r"\n+", " ", text)

    # Ensure \[...\] and \( ... \) are correctly spaced
    text = re.sub(r"\\\[", r"\\[", text)
    text = re.sub(r"\\\]", r"\\]", text)
    text = re.sub(r"\\\(", r"\\(", text)
    text = re.sub(r"\\\)", r"\\)", text)

    return text.strip()


def render_latex_blocks(text: str):
    """
    Finds LaTeX math blocks and renders them in Streamlit.
    """
    # Match display math \[...\] or inline math $...$
    display_math = re.findall(r"\\\[.*?\\\]", text)
    inline_math = re.findall(r"\$.*?\$", text)

    # Replace display math with rendered output
    for block in display_math:
        cleaned = clean_latex(block)
        st.latex(cleaned.strip("\\[").strip("\\]"))

    for block in inline_math:
        cleaned = clean_latex(block)
        st.latex(cleaned.strip("$"))







def normalize_latex(content):
    # 1. Convert \[ ... \] to $$ ... $$
    content = re.sub(r"\\\\\[(.*?)\\\\\]", r"$$\1$$", content, flags=re.DOTALL)
    # 2. Convert [ ... ] and ( ... ) with LaTeX commands to $ ... $
    def wrap_math(match):
        expr = match.group(1)
        return f"${expr.strip()}$"
    latex_cmds = r"\\(?:times|div|frac|cdot|pm|sqrt|sum|prod|int|log|sin|cos|tan|text|left|right|approx|frac|%|=|\^|_|\{|\}|begin|end|\\)"
    content = re.sub(r"\[(.*?" + latex_cmds + r"[^]]*)\]", wrap_math, content)
    content = re.sub(r"\((.*?" + latex_cmds + r"[^)]*)\)", wrap_math, content)
    # 3. Remove \$ and $ before numbers not in math context
    content = re.sub(r"\\\$([0-9,.]+)", r"\1", content)
    content = re.sub(r"(?<!\$)\$([0-9,.]+)", r"\1", content)
    # 4. Remove stray backslashes before numbers
    content = re.sub(r"\\([0-9,.]+)", r"\1", content)
    # 5. Optionally, convert $$...$$ to $...$ (if you want only inline math)
    # content = re.sub(r"\$\$(.*?)\$\$", r"$\1$", content, flags=re.DOTALL)
    # 6. Remove math in code blocks and headings (optional, advanced)
    return content


#######experimertal latex for CAGR#########
def clean_latex(llm_text: str) -> str:
    """
    Cleans and fixes common LaTeX issues from LLM outputs.
    """
    # Remove extra dollar signs if doubled
    text = llm_text.replace("$$", "$")

    # Fix broken commands like \ext to \text, \ac to \frac, \ight to \right
    replacements = {
        r"\\ext": r"\\text",
        r"\\ac": r"\\frac",
        r"\\ight": r"\\right",
        r"\\eft": r"\\left",
        r"\\pprox": r"\\approx",
    }
    for bad, good in replacements.items():
        text = re.sub(bad, good, text)

    # Remove unwanted \n inside LaTeX blocks
    text = re.sub(r"\n+", " ", text)

    # Ensure \[...\] and \( ... \) are correctly spaced
    text = re.sub(r"\\\[", r"\\[", text)
    text = re.sub(r"\\\]", r"\\]", text)
    text = re.sub(r"\\\(", r"\\(", text)
    text = re.sub(r"\\\)", r"\\)", text)

    return text.strip()


def render_latex_blocks(text: str):
    """
    Finds LaTeX math blocks and renders them in Streamlit.
    """
    # Match display math \[...\] or inline math $...$
    display_math = re.findall(r"\\\[.*?\\\]", text)
    inline_math = re.findall(r"\$.*?\$", text)

    # Replace display math with rendered output
    for block in display_math:
        cleaned = clean_latex(block)
        st.latex(cleaned.strip("\\[").strip("\\]"))

    for block in inline_math:
        cleaned = clean_latex(block)
        st.latex(cleaned.strip("$"))





def highlight_red_text(content: str) -> str:
    # Replace [red]...[/red] with HTML span for red color
    return re.sub(r"\[red\](.*?)\[/red\]", r"<span style='color: red;'>\1</span>", content, flags=re.DOTALL)

def render_full_output(text: str):
    from streamlit import markdown
    from utils.latex_check import normalize_latex, clean_latex

    # Clean and normalize
    text = normalize_latex(text)
    text = clean_latex(text)
    text = highlight_red_text(text)
    # Render with markdown (MathJax will handle LaTeX)
    markdown(text, unsafe_allow_html=True)