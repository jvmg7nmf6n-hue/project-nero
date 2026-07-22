from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "media"
OUT_FILE = OUT_DIR / "nero_explainer.gif"
FRAME_SIZE = (1280, 720)
SLIDES = [
    ("PROJECT NERO", "From a simple dashboard to an evidence-driven trading intelligence lab.", ["Market data", "News sentiment", "Paper accountability", "Strategy repair loop"]),
    ("DAY ONE", "NERO was born as a Streamlit market-research terminal.", ["Asset selector", "Live/fallback data label", "Verdict panel", "Prediction log"]),
    ("LIVE AWARENESS", "NERO learned to read candles, news, AI sentiment, and mobile alerts.", ["BTC and crypto feeds", "Gold and macro data", "RSS news", "ntfy + email alerts"]),
    ("ACCOUNTABILITY", "Signals became auditable paper records.", ["Entry, stop, target", "Win rate and R result", "Prediction Lab", "Truth Dashboard"]),
    ("INTELLIGENCE STACK", "NERO added historical memory and quant context.", ["White House impact", "ETF flows and real yields", "GARCH volatility", "Cointegration and beta"]),
    ("STRATEGY TEST LAB", "Multiple strategies now run in parallel paper mode.", ["Old strategies", "New hypotheses", "Short strategies", "Range MR variants"]),
    ("STRATEGY DOCTOR", "Weak systems are quarantined and repaired, not blindly trusted.", ["Quarantine", "Repair candidate", "30+ trade evidence gate", "Promote or rework"]),
    ("THE REAL INVENTION", "NERO turns trading ideas into evidence.", ["No profit guarantee", "No dummy confidence", "Clean data first", "Evidence before risk"]),
]

def font(size: int, bold: bool = False):
    for name in (["arialbd.ttf", "segoeuib.ttf", "calibrib.ttf"] if bold else ["arial.ttf", "segoeui.ttf", "calibri.ttf"]):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()

def slide(title: str, subtitle: str, bullets: list[str], index: int) -> Image.Image:
    img = Image.new("RGB", FRAME_SIZE, "#071018")
    draw = ImageDraw.Draw(img)
    w, h = FRAME_SIZE
    for y in range(0, h, 36):
        shade = 28 + min(32, y // 18)
        draw.line((0, y, w, y), fill=(8, shade, 42), width=1)
    for x in range(0, w, 64):
        draw.line((x, 0, x, h), fill=(9, 22, 34), width=1)
    draw.rectangle((0, 0, w, 88), fill="#0b1522")
    draw.rectangle((0, 88, w, 92), fill="#f6c915")
    title_font, subtitle_font, bullet_font, small_font = font(62, True), font(30), font(30), font(22)
    draw.text((72, 122), title, fill="#f7fbff", font=title_font)
    draw.text((76, 206), subtitle, fill="#9ec7e8", font=subtitle_font)
    box_x, box_y, box_w, box_h = 74, 292, 720, 300
    draw.rounded_rectangle((box_x, box_y, box_x + box_w, box_y + box_h), radius=18, fill="#0e1a29", outline="#26384c", width=2)
    for i, bullet in enumerate(bullets):
        y = box_y + 42 + i * 60
        draw.ellipse((box_x + 34, y + 8, box_x + 52, y + 26), fill="#f45b69")
        draw.text((box_x + 76, y), bullet, fill="#e8f1fa", font=bullet_font)
    right_x = 860
    draw.rounded_rectangle((right_x, 162, 1210, 592), radius=20, fill="#09131f", outline="#304961", width=2)
    draw.text((right_x + 28, 194), "NERO LIVE STACK", fill="#f6c915", font=small_font)
    labels, values = ["Data", "Sentiment", "Quant", "Paper P/L", "Repair"], [0.82, 0.58, 0.66, 0.44, 0.72]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = 248 + i * 58
        draw.text((right_x + 28, y), label, fill="#d7e2ed", font=small_font)
        draw.rectangle((right_x + 140, y + 8, right_x + 318, y + 26), fill="#142234")
        draw.rectangle((right_x + 140, y + 8, right_x + 140 + int(178 * value), y + 26), fill="#2dd4bf")
    draw.text((76, 650), f"Slide {index + 1}/{len(SLIDES)}  |  Project NERO", fill="#6f8aa5", font=small_font)
    return img

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for idx, item in enumerate(SLIDES):
        frame = slide(*item, index=idx)
        frames.extend([frame] * 14)
    frames[0].save(OUT_FILE, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=False)
    print(f"Wrote {OUT_FILE}")

if __name__ == "__main__":
    main()
