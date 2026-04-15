"""Generate feature image for the video embedding benchmark blog post."""

from PIL import Image, ImageDraw, ImageFont
import os

WIDTH, HEIGHT = 1200, 630  # OG image standard
BG = (13, 13, 20)  # near-black
ACCENT = (99, 102, 241)  # indigo
ACCENT2 = (16, 185, 129)  # emerald
GRID_COLOR = (30, 30, 45)
TEXT_WHITE = (240, 240, 245)
TEXT_DIM = (140, 140, 160)
BAR_BG = (25, 25, 38)

# Model data (NDCG@10)
models = [
    ("Gemini Embedding 2", 0.769, ACCENT),
    ("Marengo 3.0", 0.760, ACCENT),
    ("Wholembed v3", 0.757, ACCENT),
    ("X-CLIP Base", 0.470, (75, 85, 120)),
    ("SigLIP 2", 0.325, (55, 60, 85)),
    ("InternVideo2 6B", 0.302, (55, 60, 85)),
]

img = Image.new("RGB", (WIDTH, HEIGHT), BG)
draw = ImageDraw.Draw(img)

# Try to load fonts
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 38)
    font_subtitle = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    font_value = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    font_metric = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
except:
    font_title = ImageFont.load_default()
    font_subtitle = font_label = font_value = font_small = font_metric = font_title

# Draw subtle grid lines
for x in range(0, WIDTH, 40):
    draw.line([(x, 0), (x, HEIGHT)], fill=(18, 18, 28), width=1)
for y in range(0, HEIGHT, 40):
    draw.line([(0, y), (WIDTH, y)], fill=(18, 18, 28), width=1)

# Title area
draw.text((60, 45), "Video Embedding Benchmark 2026", fill=TEXT_WHITE, font=font_title)
draw.text((60, 95), "Text-to-Video Retrieval  |  6 Models  |  60 Queries  |  NDCG@10", fill=TEXT_DIM, font=font_subtitle)

# Divider
draw.line([(60, 135), (WIDTH - 60, 135)], fill=GRID_COLOR, width=1)

# Bar chart area
chart_left = 240
chart_right = WIDTH - 100
chart_top = 165
bar_height = 52
bar_gap = 14
max_val = 0.85  # scale

for i, (name, score, color) in enumerate(models):
    y = chart_top + i * (bar_height + bar_gap)

    # Model name (left-aligned)
    draw.text((60, y + 14), name, fill=TEXT_DIM, font=font_label)

    # Bar background
    bar_width = chart_right - chart_left
    draw.rounded_rectangle(
        [(chart_left, y + 6), (chart_right, y + bar_height - 6)],
        radius=4, fill=BAR_BG
    )

    # Filled bar
    fill_width = int((score / max_val) * bar_width)
    if fill_width > 8:
        # Gradient effect: slightly brighter at top
        r, g, b = color
        draw.rounded_rectangle(
            [(chart_left, y + 6), (chart_left + fill_width, y + bar_height - 6)],
            radius=4, fill=color
        )
        # Lighter top half for depth
        draw.rounded_rectangle(
            [(chart_left, y + 6), (chart_left + fill_width, y + (bar_height - 6) // 2 + 3)],
            radius=4, fill=(min(r + 20, 255), min(g + 20, 255), min(b + 20, 255))
        )

    # Score value
    draw.text(
        (chart_left + fill_width + 12, y + 12),
        f"{score:.3f}",
        fill=TEXT_WHITE, font=font_value
    )

# Metric label
draw.text((chart_left, chart_top - 22), "NDCG@10 (higher is better)", fill=TEXT_DIM, font=font_metric)

# Scale ticks
for tick in [0.0, 0.2, 0.4, 0.6, 0.8]:
    x = chart_left + int((tick / max_val) * (chart_right - chart_left))
    draw.line([(x, HEIGHT - 75), (x, HEIGHT - 68)], fill=GRID_COLOR, width=1)
    draw.text((x - 10, HEIGHT - 65), f"{tick:.1f}", fill=(80, 80, 100), font=font_small)

# Bottom bar with branding
draw.rectangle([(0, HEIGHT - 42), (WIDTH, HEIGHT)], fill=(10, 10, 16))
draw.text((60, HEIGHT - 32), "mixpeek.com/research", fill=TEXT_DIM, font=font_small)
draw.text((WIDTH - 280, HEIGHT - 32), "CC0 dataset  |  MIT benchmark code", fill=(80, 80, 100), font=font_small)

# Key insight callout (top right)
draw.rounded_rectangle([(WIDTH - 340, 50), (WIDTH - 60, 120)], radius=8, fill=(20, 20, 35))
draw.text((WIDTH - 325, 60), "Top 3 API models within", fill=TEXT_DIM, font=font_small)
draw.text((WIDTH - 325, 78), "0.012 NDCG@10 of each other", fill=ACCENT2, font=font_label)
draw.text((WIDTH - 325, 98), "2x gap to best open-source", fill=TEXT_DIM, font=font_small)

out_path = os.path.join(os.path.dirname(__file__), "hero.png")
img.save(out_path, "PNG", quality=95)
print(f"Saved: {out_path}")
print(f"Size: {os.path.getsize(out_path):,} bytes")
