from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_PATH = Path("control_latency_diagram.png")
WIDTH = 1400
HEIGHT = 800
BG = "#FFFFFF"
ORANGE = "#F4B400"
DARK = "#2F2F2F"
GRAY = "#6B7280"
LIGHT = "#EEF2F7"
RED = "#D64545"
BLUE = "#67B7DC"


def get_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates += [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/seguisb.ttf",
        ]
    candidates += [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


TITLE_FONT = get_font(46, bold=True)
LABEL_FONT = get_font(30, bold=True)
TEXT_FONT = get_font(28)


def centered_text(draw, xy, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    x = xy[0] - (bbox[2] - bbox[0]) / 2
    y = xy[1] - (bbox[3] - bbox[1]) / 2
    draw.text((x, y), text, font=font, fill=fill)


def draw_arrow(draw, start, end, color=DARK, width=8):
    draw.line([start, end], fill=color, width=width)
    ex, ey = end
    sx, sy = start
    dx = ex - sx
    dy = ey - sy
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18
    wing = 12
    p1 = (ex, ey)
    p2 = (ex - ux * head + px * wing, ey - uy * head + py * wing)
    p3 = (ex - ux * head - px * wing, ey - uy * head - py * wing)
    draw.polygon([p1, p2, p3], fill=color)


def draw_robot(draw, box):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=LIGHT, outline=DARK, width=4)
    base_y = y1 - 70
    draw.rounded_rectangle((x0 + 55, base_y, x0 + 170, base_y + 34), radius=12, fill=DARK)
    draw.rounded_rectangle((x0 + 82, base_y - 115, x0 + 138, base_y), radius=18, fill="#F9FAFB", outline=DARK, width=4)
    draw.ellipse((x0 + 90, base_y - 165, x0 + 150, base_y - 105), fill="#F9FAFB", outline=DARK, width=4)
    draw.line((x0 + 120, base_y - 115, x0 + 185, base_y - 170), fill="#F9FAFB", width=24)
    draw.line((x0 + 185, base_y - 170, x0 + 240, base_y - 130), fill="#F9FAFB", width=24)
    draw.line((x0 + 240, base_y - 130, x0 + 280, base_y - 165), fill="#F9FAFB", width=20)
    draw.ellipse((x0 + 175, base_y - 180, x0 + 195, base_y - 160), fill=DARK)
    draw.ellipse((x0 + 230, base_y - 140, x0 + 250, base_y - 120), fill=DARK)
    draw.line((x0 + 278, base_y - 165, x0 + 302, base_y - 150), fill=DARK, width=7)
    draw.line((x0 + 278, base_y - 165, x0 + 298, base_y - 185), fill=DARK, width=7)


def draw_cloud(draw, box):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=LIGHT, outline=DARK, width=4)
    cy = (y0 + y1) // 2 + 10
    cloud_parts = [
        (x0 + 70, cy - 40, x0 + 170, cy + 40),
        (x0 + 130, cy - 80, x0 + 250, cy + 30),
        (x0 + 220, cy - 55, x0 + 340, cy + 45),
        (x0 + 90, cy - 5, x0 + 310, cy + 70),
    ]
    for part in cloud_parts:
        draw.ellipse(part, fill=BLUE, outline=DARK, width=4)
    centered_text(draw, ((x0 + x1) / 2, y0 + 65), "Cloud LLM API", LABEL_FONT, DARK)


def draw_clock(draw, box):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=28, fill=LIGHT, outline=DARK, width=4)
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + 20
    r = 90
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#FFF7E6", outline=RED, width=8)
    draw.line((cx, cy, cx, cy - 48), fill=RED, width=8)
    draw.line((cx, cy, cx + 38, cy), fill=RED, width=8)
    centered_text(draw, (cx, y0 + 65), "Latency", LABEL_FONT, RED)
    centered_text(draw, (cx, y1 - 45), "delay from API calls", TEXT_FONT, GRAY)


def main():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    centered_text(draw, (WIDTH / 2, 70), "Control Latency from API-Based Planning", TITLE_FONT, DARK)
    draw.line((110, 120, WIDTH - 110, 120), fill=ORANGE, width=6)

    robot_box = (70, 220, 410, 650)
    cloud_box = (520, 220, 880, 650)
    clock_box = (990, 220, 1330, 650)

    draw_robot(draw, robot_box)
    centered_text(draw, ((robot_box[0] + robot_box[2]) / 2, robot_box[1] + 36), "Robot Policy", LABEL_FONT, DARK)
    centered_text(draw, ((robot_box[0] + robot_box[2]) / 2, robot_box[3] - 22), "needs next instruction", TEXT_FONT, GRAY)

    draw_cloud(draw, cloud_box)
    draw_clock(draw, clock_box)

    draw_arrow(draw, (410, 435), (520, 435), color=DARK, width=8)
    draw_arrow(draw, (880, 435), (990, 435), color=RED, width=8)

    centered_text(draw, (465, 390), "request", TEXT_FONT, DARK)
    centered_text(draw, (935, 390), "waiting", TEXT_FONT, RED)

    centered_text(draw, (WIDTH / 2, 730), "Future: deploy a local quantized fast-reasoning LLM to reduce latency.", LABEL_FONT, DARK)

    img.save(OUT_PATH)
    print(f"Saved {OUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
