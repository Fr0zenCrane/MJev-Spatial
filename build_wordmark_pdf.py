"""Build the paper wordmark from its RGB artwork and alpha mask.

The source files under ``assets/wordmark`` are extracted from the arXiv PDF:
the RGB layer contains the artwork and the grayscale alpha layer contains the
original transparency mask. The generated PNG therefore uses the paper's real
wordmark instead of a background-removal approximation.

The transparent PNG is always generated. If ReportLab is installed, the
script also writes a PDF with a selectable text layer.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parent
WORDMARK = ROOT / "assets" / "wordmark"
RGB_SOURCE = WORDMARK / "source-rgb.png"
ALPHA_SOURCE = WORDMARK / "source-alpha.png"
PNG_OUTPUT = ROOT / "assets" / "mjev-spatial-title.png"
PDF_OUTPUT = ROOT / "assets" / "mjev-spatial-title.pdf"
TEXT = "MJev-Spatial: System One Spatial Intelligence via Native Attention Matching"


def build_transparent_png() -> Image.Image:
    rgb = Image.open(RGB_SOURCE).convert("RGB")
    alpha = Image.open(ALPHA_SOURCE).convert("L")
    if rgb.size != alpha.size:
        raise ValueError(f"RGB and alpha sources differ: {rgb.size} vs {alpha.size}")
    result = rgb.convert("RGBA")
    result.putalpha(alpha)
    result.save(PNG_OUTPUT, optimize=True)
    return result


def build_selectable_pdf(image: Image.Image) -> None:
    try:
        from reportlab.pdfbase.pdfmetrics import stringWidth
        from reportlab.pdfgen.canvas import Canvas
    except ImportError:
        print("ReportLab is not installed; skipped PDF output")
        return

    scale = 0.2
    width, height = image.size
    page_width, page_height = width * scale, height * scale
    canvas = Canvas(
        str(PDF_OUTPUT), pagesize=(page_width, page_height), pageCompression=1, invariant=1
    )
    canvas.setTitle(TEXT)
    canvas.setSubject("Selectable MJev-Spatial paper wordmark")
    canvas.drawImage(
        str(PNG_OUTPUT), 0, 0, width=page_width, height=page_height, mask="auto"
    )
    text = canvas.beginText()
    font_size = page_height * 0.45
    text.setFont("Helvetica", font_size)
    text.setTextRenderMode(3)
    text.setHorizScale(100 * page_width / stringWidth(TEXT, "Helvetica", font_size))
    text.setTextOrigin(0, page_height * 0.18)
    text.textOut(TEXT)
    canvas.drawText(text)
    canvas.showPage()
    canvas.save()


if __name__ == "__main__":
    artwork = build_transparent_png()
    build_selectable_pdf(artwork)
    print(PNG_OUTPUT)
