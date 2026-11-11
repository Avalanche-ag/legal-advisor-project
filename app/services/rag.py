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


def generate_report(user_queries: list[str], ai_responses: list[str]) -> str:
    """Render a Q&A transcript to a PDF and return the file path.

    ReportLab is imported lazily so a broken image/PDF dependency can never
    prevent the Flask app itself from starting.
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import simpleSplit
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(REPORT_PATH, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "AI Lawyer Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 730, "Below is a record of your conversation with AI Lawyer.")

    y = 700
    max_width = 450
    line_height = 15

    for question, answer in zip(user_queries, ai_responses):
        c.setFont("Helvetica-Bold", 12)
        q_lines = simpleSplit(f"Q: {question}", "Helvetica-Bold", 12, max_width)
        a_lines = simpleSplit(f"A: {answer}", "Helvetica", 12, max_width)

        for line in q_lines:
            c.drawString(100, y, line)
            y -= line_height

        c.setFont("Helvetica", 12)
        for line in a_lines:
            c.drawString(100, y, line)
            y -= line_height

        y -= 20
        if y < 50:
            c.showPage()
            c.setFont("Helvetica", 12)
            y = 750

    c.save()
    return REPORT_PATH
