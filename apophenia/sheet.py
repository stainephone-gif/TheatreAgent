"""Лист архива: один лист А4 на цикл (PDF через reportlab).

Содержимое: шапка (название, номер цикла, дата, исход), призрачные цитаты, траектория дрейфа по классам,
итоговая самоэкспликация, служебная строка. Текст экспликации ужимается по кеглю, чтобы всё влезло на одну страницу.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Flowable, Frame, Paragraph, Spacer

from . import ALL_CLASSES, CLASS_NAMES_RU, OUTCOMES
from .config import resolve

_FONTS_REGISTERED = False


def register_fonts(cfg: Dict[str, Any]) -> None:
    global _FONTS_REGISTERED
    if _FONTS_REGISTERED:
        return
    s = cfg.get("sheet", {})
    pdfmetrics.registerFont(TTFont("Sans", str(resolve(cfg, s.get("font_regular", "fonts/DejaVuSans.ttf")))))
    pdfmetrics.registerFont(TTFont("SansB", str(resolve(cfg, s.get("font_bold", "fonts/DejaVuSans-Bold.ttf")))))
    pdfmetrics.registerFont(TTFont("Mono", str(resolve(cfg, s.get("font_mono", "fonts/DejaVuSansMono.ttf")))))
    _FONTS_REGISTERED = True


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Trajectory(Flowable):
    """Сетка: строки — восемь классов, столбцы — итерации; точка в классе итерации, размер по уверенности."""

    def __init__(self, classes: List[str], confidences: List[float], width: float, scale: float = 1.0):
        super().__init__()
        self.classes = classes
        self.confidences = confidences
        self.w = width
        self.k = scale
        self.row_h = 4.6 * mm * scale
        self.label_w = 54 * mm * scale
        self.h = self.row_h * len(ALL_CLASSES) + 8 * mm * scale

    def wrap(self, availWidth, availHeight):
        return self.w, self.h

    def draw(self):
        c = self.canv
        k = self.k
        n = max(1, len(self.classes))
        grid_w = self.w - self.label_w - 4 * mm
        step = min(9 * mm * k, grid_w / n)
        top = self.h - 3 * mm * k
        c.setFont("Sans", 6.5 * k)
        c.setStrokeColor(colors.Color(0.82, 0.82, 0.82))
        c.setLineWidth(0.3)
        for i, cls in enumerate(ALL_CLASSES):
            y = top - i * self.row_h - self.row_h / 2
            c.setFillColor(colors.black)
            c.drawString(0, y - 2 * k, f"{cls}")
            c.setFillColor(colors.Color(0.45, 0.45, 0.45))
            c.drawString(11 * mm * k, y - 2 * k, CLASS_NAMES_RU[cls])
            c.line(self.label_w, y, self.label_w + step * n, y)
        pts = []
        for j, cls in enumerate(self.classes):
            i = ALL_CLASSES.index(cls) if cls in ALL_CLASSES else ALL_CLASSES.index("UND")
            x = self.label_w + step * j + step / 2
            y = top - i * self.row_h - self.row_h / 2
            pts.append((x, y))
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.6 * k)
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, y1, x2, y2)
        for (x, y), conf in zip(pts, self.confidences):
            r = (0.6 * mm + 1.1 * mm * float(conf)) * k
            c.setFillColor(colors.black)
            c.circle(x, y, r, stroke=0, fill=1)
        c.setFont("Mono", 5.5 * k)
        c.setFillColor(colors.Color(0.45, 0.45, 0.45))
        yb = top - len(ALL_CLASSES) * self.row_h - 3 * mm * k
        for j, (x, _) in enumerate(pts):
            c.drawCentredString(x, yb, str(j + 1))
        c.setFillColor(colors.black)


def _flowables(rec: Dict[str, Any], cfg: Dict[str, Any], k: float, width: float) -> List[Flowable]:
    """Все элементы листа в масштабе k (k = 1 — базовые кегли: текст 10 pt, цитаты 8.5 pt, шапка 13 pt)."""
    s = cfg.get("sheet", {})
    g = colors.Color(0.45, 0.45, 0.45)
    st_h1 = ParagraphStyle("h1", fontName="SansB", fontSize=13 * k, leading=16 * k)
    st_meta = ParagraphStyle("meta", fontName="Mono", fontSize=7.5 * k, leading=10 * k, textColor=colors.Color(0.3, 0.3, 0.3))
    st_sec = ParagraphStyle("sec", fontName="SansB", fontSize=8 * k, leading=11 * k, spaceBefore=6 * k, spaceAfter=1.5 * k)
    st_quote = ParagraphStyle("q", fontName="Sans", fontSize=8.5 * k, leading=11.5 * k, leftIndent=4 * mm * k, spaceAfter=1 * k)
    st_src = ParagraphStyle("src", fontName="Sans", fontSize=6.5 * k, leading=8.5 * k, leftIndent=4 * mm * k, textColor=g, spaceAfter=3 * k)
    st_body = ParagraphStyle("body", fontName="Sans", fontSize=10 * k, leading=13.5 * k, alignment=TA_JUSTIFY, spaceAfter=6 * k)
    st_foot = ParagraphStyle("foot", fontName="Mono", fontSize=6.5 * k, leading=8.5 * k, textColor=colors.Color(0.4, 0.4, 0.4))

    out: List[Flowable] = []
    o = rec.get("outcome", {}) or {}
    oname = OUTCOMES.get(o.get("outcome", ""), o.get("outcome", ""))
    when = rec.get("finished_at", "").replace("T", " ")
    out.append(Paragraph(f"{_esc(s.get('title', 'МЕТАСОЗНАНИЕ'))} &nbsp;&nbsp;<font name='Mono' size='{9 * k:.1f}'>цикл {rec['cycle']:05d}</font>", st_h1))
    outcome_line = f"{when} &nbsp;·&nbsp; исход: {oname.upper()}"
    if o.get("outcome") == "STABILIZED":
        outcome_line += f" — {o.get('class')} ({CLASS_NAMES_RU.get(o.get('class', ''), '')})"
    elif o.get("outcome") == "OSCILLATION":
        outcome_line += f" — {' / '.join(o.get('pattern', []))}"
    elif o.get("outcome") == "UNFINISHED":
        outcome_line += " — " + ("вырождение текста" if o.get("reason") == "degenerate" else "потолок итераций")
    outcome_line += f" &nbsp;·&nbsp; итераций: {len(rec.get('iterations', []))}"
    out.append(Paragraph(outcome_line, st_meta))
    out.append(Spacer(1, 3 * mm * k))

    out.append(Paragraph("ПРИЗРАЧНЫЕ ЦИТАТЫ — прочитаны моделью в кураторских текстах, в текстах не найдены", st_sec))
    for q in rec.get("ghosts", {}).get("quotes", []):
        out.append(Paragraph(f"«{_esc(q['span'])}»", st_quote))
        src = f"прочитано в: «{_esc(q.get('title', ''))}»"
        if q.get("author"):
            src += f", {_esc(q['author'])}"
        src += f" · как {q.get('class', '')}"
        m = q.get("mapping") or {}
        if m.get("source") and m.get("target"):
            src += f": {_esc(m['source'])} → {_esc(m['target'])}"
        out.append(Paragraph(src, st_src))

    out.append(Paragraph("ТРАЕКТОРИЯ ДРЕЙФА — как классификатор прочитывал каждую версию текста", st_sec))
    out.append(Trajectory(rec.get("trajectory", []), rec.get("confidences", []), width, scale=k))

    out.append(Paragraph("САМОЭКСПЛИКАЦИЯ — последняя версия", st_sec))
    for para in [p for p in (rec.get("final_text") or "").split("\n") if p.strip()]:
        out.append(Paragraph(_esc(para.strip()), st_body))

    foot = (f"{rec.get('model', '')} · промпт классификатора {rec.get('prompts', {}).get('theory', {}).get('hash', '')} "
            f"· кодбук {rec.get('prompts', {}).get('theory', {}).get('codebook_hash', '')} · {rec.get('code_version', '')} "
            f"· {rec.get('duration_s', '')} с")
    out.append(Spacer(1, 3 * mm * k))
    out.append(Paragraph(_esc(foot), st_foot))
    out.append(Paragraph(_esc(s.get("footer", "")), st_foot))
    return out


def render_sheet(rec: Dict[str, Any], cfg: Dict[str, Any], out_path: Path) -> Path:
    """Подбирает кегль экспликации так, чтобы всё влезло на одну страницу, и пишет PDF."""
    import io

    register_fonts(cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = A4
    margin = float(cfg.get("sheet", {}).get("margin_mm", 16)) * mm
    width, height = page_w - 2 * margin, page_h - 2 * margin
    s = cfg.get("sheet", {})
    k_max = float(s.get("scale_max", 1.7))   # верхний предел: текст 17 pt
    k_min = float(s.get("scale_min", 0.6))   # нижний предел: текст 6 pt
    best: Optional[bytes] = None
    k = k_max
    while k >= k_min - 1e-9:
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        c.setTitle(f"Метасознание — цикл {rec['cycle']:05d}")
        c.setAuthor("Метасознание, генеративная инсталляция")
        frame = Frame(margin, margin, width, height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0, showBoundary=0)
        leftover = _flowables(rec, cfg, round(k, 3), width)
        frame.addFromList(leftover, c)
        c.showPage()
        c.save()
        best = buf.getvalue()
        if not leftover:
            break
        k -= 0.05
    out_path.write_bytes(best or b"")
    return out_path


def render_protocol(rec: Dict[str, Any], cfg: Dict[str, Any], out_path: Path) -> Path:
    """Протокол цикла для проверки алгоритма: все версии текста с прочтением каждой. Несколько страниц."""
    from reportlab.platypus import PageBreak, SimpleDocTemplate

    register_fonts(cfg)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    st_h1 = ParagraphStyle("ph1", fontName="SansB", fontSize=12, leading=15)
    st_meta = ParagraphStyle("pmeta", fontName="Mono", fontSize=7.5, leading=10, textColor=colors.Color(0.3, 0.3, 0.3))
    st_sec = ParagraphStyle("psec", fontName="SansB", fontSize=9, leading=12, spaceBefore=8, spaceAfter=2)
    st_body = ParagraphStyle("pbody", fontName="Sans", fontSize=9, leading=12.5, alignment=TA_JUSTIFY, spaceAfter=4)
    st_ev = ParagraphStyle("pev", fontName="Sans", fontSize=7.5, leading=10, leftIndent=4 * mm,
                           textColor=colors.Color(0.25, 0.25, 0.25), spaceAfter=1)
    st_q = ParagraphStyle("pq", fontName="Sans", fontSize=8.5, leading=11, leftIndent=4 * mm, spaceAfter=1)

    o = rec.get("outcome", {}) or {}
    flow: List[Flowable] = [
        Paragraph(f"ПРОТОКОЛ ЦИКЛА {rec['cycle']:05d} &nbsp;<font name='Mono' size='8'>{_esc(rec.get('finished_at', '').replace('T', ' '))}</font>", st_h1),
        Paragraph(f"исход: {OUTCOMES.get(o.get('outcome', ''), o.get('outcome', ''))} — {_esc(o.get('detail', ''))} · "
                  f"итераций: {len(rec.get('iterations', []))} · {rec.get('duration_s', '')} с · {_esc(rec.get('model', ''))}", st_meta),
        Paragraph("ПРИЗРАЧНЫЕ ЦИТАТЫ", st_sec),
    ]
    for q in rec.get("ghosts", {}).get("quotes", []):
        flow.append(Paragraph(f"«{_esc(q['span'])}» <font size='6.5' color='#777777'>· {q.get('class', '')} · «{_esc(q.get('title', ''))}» · "
                              f"совпадение с текстом {q.get('token_overlap', '')}</font>", st_q))
    for rd in rec.get("ghosts", {}).get("readings", []):
        flow.append(Paragraph(f"текст {rd.get('text_id')} «{_esc(rd.get('title', ''))}»: фрагментов {rd.get('n_total')}, "
                              f"верифицировано {rd.get('n_kept')}, призрачных {rd.get('n_unverified')}", st_ev))
    for it in rec.get("iterations", []):
        flow.append(Paragraph(f"ИТЕРАЦИЯ {it['n']} — прочитано как {it['primary']} ({CLASS_NAMES_RU.get(it['primary'], '')}), "
                              f"уверенность {it['confidence']:.2f}, фрагментов {it.get('n_kept', 0)}, слов {it.get('n_words', '')}", st_sec))
        for e in it.get("evidence", []):
            m = e.get("mapping") or {}
            flow.append(Paragraph(f"[{e['class']}] «{_esc(e['span'])}» — {_esc(m.get('source', ''))} → {_esc(m.get('target', ''))}", st_ev))
        if it.get("unverified"):
            flow.append(Paragraph("не найдено в тексте: " + "; ".join(f"«{_esc(u)}»" for u in it["unverified"]), st_ev))
        for para in [p for p in (it.get("text") or "").split("\n") if p.strip()]:
            flow.append(Paragraph(_esc(para.strip()), st_body))
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm,
                            title=f"Метасознание — протокол цикла {rec['cycle']:05d}")
    doc.build(flow)
    return out_path
