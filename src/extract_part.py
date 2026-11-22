# src/extract_part.py
# Вирізає неперервну частину повного підручника Biology 2e (1475 сторінок):
# глави 1-5 (сторінки 29-180 повного PDF) -> data/document.pdf (152 сторінки).
# Далі в роботі використовуються номери сторінок саме data/document.pdf.
import sys
from pypdf import PdfReader, PdfWriter

FULL = "data/Biology-2e_full.pdf"
FIRST, LAST = 29, 180  # 1-based, включно

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else FULL
    reader, writer = PdfReader(src), PdfWriter()
    for i in range(FIRST - 1, LAST):
        writer.add_page(reader.pages[i])
    with open("data/document.pdf", "wb") as f:
        writer.write(f)
    print(f"Збережено сторінки {FIRST}-{LAST} ({LAST - FIRST + 1} стор.) у data/document.pdf")
