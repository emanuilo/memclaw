"""Render the README demo video (index.html) frame by frame.

  uv run --with playwright python capture.py preview 2.0 6.5 ...   -> preview_<t>.png
  uv run --with playwright python capture.py frames                -> frames/00000.png ...

Then encode (frames/ and the outputs are not kept in the repo):

  ffmpeg -framerate 30 -i frames/%05d.png -c:v libx264 -preset slow -crf 20 \
      -pix_fmt yuv420p -movflags +faststart ../memclaw-demo.mp4
  ffmpeg -framerate 30 -i frames/%05d.png -vf "fps=15,scale=800:-1:flags=lanczos,split[a][b];\
[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle" \
      ../memclaw-demo.gif

Uses the locally installed Google Chrome.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
FPS, DURATION, SCALE = 30, 30.0, 1.5


def main() -> None:
    mode = sys.argv[1]
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1280, "height": 720}, device_scale_factor=SCALE)
        page.goto((HERE / "index.html").as_uri())
        page.evaluate("window.ready")
        if mode == "preview":
            for t in sys.argv[2:]:
                page.evaluate(f"render({float(t)})")
                page.screenshot(path=str(HERE / f"preview_{t}.png"))
        else:
            out = HERE / "frames"
            out.mkdir(exist_ok=True)
            for i in range(int(FPS * DURATION)):
                page.evaluate(f"render({i / FPS})")
                page.screenshot(path=str(out / f"{i:05d}.png"))
        browser.close()


if __name__ == "__main__":
    main()
