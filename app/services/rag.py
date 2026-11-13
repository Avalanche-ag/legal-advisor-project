"""RAG orchestration and PDF report generation."""
from pathlib import Path

from .generator import generate
from .retriever import retrieve_ipc_sections

BASE_DIR = Path(__file__).resolve().parents[2]
REPORT_PATH = str(BASE_DIR / "AI_Lawyer_Report.pdf")

IPC_QA_TEMPLATE = """
You are an AI legal assistant specializing in Indian law and the Indian Penal Code (IPC).
Use ONLY the IPC sections provided in the context to answer the user's question.
If the context does not contain a relevant section, say that you don't know — do NOT fabricate sections.

Instructions:
- Cite the section number and title for every section you rely on.
- Explain the relevant punishment or provision concisely.
- Keep the answer focused and in plain language.

Previous Conversation:
{history}

Question: {question}

Context (retrieved IPC sections):
{context}

Answer:
"""


def _format_context(sections: list[dict]) -> str:
    return "\n\n".join(
        f"Section {s['section']} ({s['chapter_title']}): {s['section_title']}\n{s['content']}"
        for s in sections
    )


def _to_text(response) -> str:
    """Extract plain text from a LangChain chat response."""
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def answer_ipc_query(query: str, history: str = "") -> tuple[str, list[dict]]:
    """Retrieve the top-3 IPC sections and generate a grounded answer.

    Returns (answer_text, retrieved_sections).
    """
    sections = retrieve_ipc_sections(query, k=3)
    response = generate(
        IPC_QA_TEMPLATE,
        question=query,
        context=_format_context(sections),
        history=history,
    )
    return _to_text(response), sections


# ---------------------------------------------------------------------------
# PDF report: Markdown -> ReportLab flowables
#
# ReportLab's Paragraph understands a small XML dialect, so the pipeline is:
#   escape XML  ->  convert inline Markdown  ->  build flowables (lists, tables)
# Everything here uses only ReportLab (already a dependency); no new packages.
# ---------------------------------------------------------------------------
import re as _re

_BOLD_RE = _re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = _re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)")
_CODE_RE = _re.compile(r"`([^`]+)`")
_LINK_RE = _re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BR_RE = _re.compile(r"&lt;br\s*/?&gt;", _re.IGNORECASE)
_BULLET_RE = _re.compile(r"^(\s*)[-*+]\s+(.*)$")
_NUMBERED_RE = _re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_SEPARATOR_RE = _re.compile(r"^:?-{2,}:?$")


def _escape_xml(text: str) -> str:
    """Escape & < > so ReportLab's mini-XML parser never chokes."""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _inline_markup(text: str) -> str:
    """Escape, then map inline Markdown onto ReportLab tags."""
    value = _escape_xml(text)
    value = _BR_RE.sub("<br/>", value)          # literal <br> / <br/> in answers
    value = _BOLD_RE.sub(r"<b>\1</b>", value)
    value = _ITALIC_RE.sub(r"<i>\1</i>", value)
    value = _CODE_RE.sub(r'<font face="Courier">\1</font>', value)
    value = _LINK_RE.sub(r"\1", value)          # keep link text, drop the URL
    return value.strip()


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _split_table_row(line: str) -> list[str]:
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    # Markdown escapes literal pipes as \|
    body = body.replace("\\|", "\x00")
    cells = [cell.replace("\x00", "|").strip() for cell in body.split("|")]
    return cells


def _build_styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import inch

    body = ParagraphStyle(
        "Body", fontName="Helvetica", fontSize=9.5, leading=13,
        spaceBefore=0, spaceAfter=4,
    )
    return {
        "title": ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=18,
                                leading=22, spaceAfter=4),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=9.5,
                                   leading=13, textColor="#555555", spaceAfter=14),
        "question": ParagraphStyle("Question", fontName="Helvetica-Bold", fontSize=11,
                                   leading=15, spaceBefore=12, spaceAfter=6),
        "body": body,
        "bullet": ParagraphStyle("Bullet", parent=body, leftIndent=14,
                                 bulletIndent=4, spaceAfter=3),
        "bullet_nested": ParagraphStyle("BulletNested", parent=body, leftIndent=30,
                                        bulletIndent=18, spaceAfter=3),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=13,
                             leading=17, spaceBefore=8, spaceAfter=4),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=11.5,
                             leading=15, spaceBefore=8, spaceAfter=4),
        "h3": ParagraphStyle("H3", fontName="Helvetica-Bold", fontSize=10.5,
                             leading=14, spaceBefore=6, spaceAfter=3),
        "table_header": ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=8.5,
                                       leading=11),
        "table_cell": ParagraphStyle("TD", fontName="Helvetica", fontSize=8.5,
                                     leading=11),
        "indent": inch,
    }


def _build_table(rows: list[list[str]], styles: dict, available_width: float):
    """Turn parsed pipe-table rows into a styled ReportLab Table."""
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, Table, TableStyle

    columns = max(len(row) for row in rows)
    rows = [row + [""] * (columns - len(row)) for row in rows]

    # Proportional widths from content length, clamped so no column collapses.
    weights = []
    for index in range(columns):
        longest = max(len(row[index]) for row in rows)
        weights.append(min(max(longest, 8), 60))
    total = float(sum(weights)) or 1.0
    widths = [max(available_width * weight / total, 0.7 * inch) for weight in weights]
    scale = available_width / sum(widths)
    widths = [width * scale for width in widths]

    data = [
        [
            Paragraph(_inline_markup(cell), styles["table_header" if row_index == 0 else "table_cell"])
            for cell in row
        ]
        for row_index, row in enumerate(rows)
    ]

    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA5B1")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E9EDF2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _markdown_to_flowables(text: str, styles: dict, available_width: float) -> list:
    """Convert one Markdown answer into a list of ReportLab flowables."""
    from reportlab.platypus import Paragraph, Spacer

    flowables: list = []
    buffer: list[str] = []

    def flush_paragraph() -> None:
        if not buffer:
            return
        joined = "<br/>".join(_inline_markup(line) for line in buffer)
        flowables.append(Paragraph(joined, styles["body"]))
        buffer.clear()

    lines = (text or "").replace("\r\n", "\n").split("\n")
    index = 0

    while index < len(lines):
        raw = lines[index]
        stripped = raw.strip()

        # --- pipe table -------------------------------------------------
        if _is_table_row(raw):
            flush_paragraph()
            block = []
            while index < len(lines) and _is_table_row(lines[index]):
                block.append(lines[index])
                index += 1

            rows = []
            for line in block:
                cells = _split_table_row(line)
                if cells and all(_SEPARATOR_RE.match(cell) for cell in cells if cell):
                    continue  # |---|---| divider
                rows.append(cells)

            if len(rows) >= 1:
                flowables.append(_build_table(rows, styles, available_width))
                flowables.append(Spacer(1, 6))
            continue

        # --- bullet / numbered list -------------------------------------
        bullet_match = _BULLET_RE.match(raw)
        number_match = _NUMBERED_RE.match(raw)
        if bullet_match or number_match:
            flush_paragraph()
            indent = len((bullet_match or number_match).group(1))
            nested = indent >= 2
            if bullet_match:
                marker, content = "\u2022", bullet_match.group(2)
            else:
                marker, content = f"{number_match.group(2)}.", number_match.group(3)
            flowables.append(
                Paragraph(
                    _inline_markup(content),
                    styles["bullet_nested" if nested else "bullet"],
                    bulletText=marker,
                )
            )
            index += 1
            continue

        # --- heading ----------------------------------------------------
        if stripped.startswith("#"):
            flush_paragraph()
            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()
            style = styles["h1" if level <= 1 else "h2" if level == 2 else "h3"]
            flowables.append(Paragraph(_inline_markup(title), style))
            index += 1
            continue

        # --- horizontal rule -------------------------------------------
        if stripped in ("---", "***", "___"):
            flush_paragraph()
            flowables.append(Spacer(1, 8))
            index += 1
            continue

        # --- blank line -------------------------------------------------
        if not stripped:
            flush_paragraph()
            index += 1
            continue

        buffer.append(stripped)
        index += 1

    flush_paragraph()
    return flowables


def generate_report(user_queries: list[str], ai_responses: list[str]) -> str:
    """Render a Q&A transcript to a PDF and return the file path.

    Markdown in the answers (**bold**, *italic*, bullet lists and pipe tables) is
    converted to real PDF formatting. ReportLab is imported lazily so a broken
    PDF dependency can never stop the Flask app from starting.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    styles = _build_styles()
    left_margin = right_margin = 54
    document = SimpleDocTemplate(
        REPORT_PATH,
        pagesize=letter,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=54,
        bottomMargin=48,
        title="AI Lawyer Report",
        author="AI Lawyer",
    )
    available_width = letter[0] - left_margin - right_margin

    story = [
        Paragraph("AI Lawyer Report", styles["title"]),
        Paragraph(
            "Below is a record of your conversation with AI Lawyer.",
            styles["subtitle"],
        ),
    ]

    for question, answer in zip(user_queries, ai_responses):
        story.append(Paragraph(f"Q: {_inline_markup(question)}", styles["question"]))
        content = _markdown_to_flowables(answer or "", styles, available_width)
        if content:
            story.extend(content)
        else:
            story.append(Paragraph("(no answer recorded)", styles["body"]))
        story.append(Spacer(1, 10))

    document.build(story)
    return REPORT_PATH
