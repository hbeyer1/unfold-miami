"""Watch the page and have Gemini 4 Argon critique it against a goal: the "watch, critique" half of the build loop
(change the code -> record it -> critique -> fix -> record again).

  python3 tools/argon_review.py --goal "the photos should condense out of the beach, not pop in" \
          [--seconds 24] [--keys "1 ArrowRight ArrowRight KeyO"] [--url http://localhost:8780] [--allow-upload]

It records the page in Chrome (reviews/<time>/page.mp4, from screenshots at 6 fps, plus stills and the console errors). Only with
--allow-upload does it send the recording to Gemini: the recording shows the personal photos as dots, so uploading it
sends them to Google. Without it, it stops after recording and prints the prompt it would send.

Needs pip install -r requirements-tools.txt, the page served (python3 serve.py), GEMINI_API_KEY (or GOOGLE_API_KEY) in
the environment, and ARGON_MODEL set to the model id AI Studio shows for Gemini 4 Argon.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

PROMPT = """You are reviewing a WebGL experience while it is being built. The video is a recording of the page in a
browser; the screenshots are stills from it; the console errors are what the browser logged.

The goal for this iteration:
{goal}

What the project is (its GEMINI.md):
{context}

Keys pressed during the recording, with the second they were pressed: {keys}
Console errors: {errors}

Judge the recording against the goal only. Be concrete: name the second, what is on screen, what is wrong, the likely
cause in the code (file and what to change), and how to tell the fix worked. Do not praise. If something cannot be
judged from the recording, say so instead of guessing.

Answer with JSON only, in this shape:
{{"score": 0-10 (how well the recording meets the goal),
  "summary": "one or two sentences",
  "issues": [{{"at_seconds": number or null, "what": "...", "severity": "low|medium|high",
              "likely_cause": "...", "fix": "...", "files": ["src/main.js", ...], "check": "how to verify"}}]}}"""


def record(url, seconds, keys, out_dir, fps=6, size=(1280, 720)):
    """Record the page as screenshots at fps (Gemini samples video at about 1 fps), pressing the keys spread over the
    time; write them as page.mp4 and keep four stills and the console errors."""
    import cv2
    import numpy as np
    from playwright.sync_api import sync_playwright
    errors, pressed, frames = [], [], []
    with sync_playwright() as p:
        args = ["--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]
        try:
            browser = p.chromium.launch(channel="chrome", args=args)   # the installed Chrome: nothing to download
        except Exception:
            browser = p.chromium.launch(args=args)
        page = browser.new_page(viewport={"width": size[0], "height": size[1]})
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        page.on("pageerror", lambda e: errors.append(str(e)))
        t0 = time.time()
        page.goto(url, wait_until="load")
        page.wait_for_function("() => document.getElementById('loading')?.classList.contains('gone')", timeout=300_000)
        loaded = time.time() - t0
        start, todo = time.time(), [((i + 1) * seconds / (len(keys) + 1), k) for i, k in enumerate(keys)]
        while (now := time.time() - start) < seconds:
            while todo and todo[0][0] <= now:
                page.keyboard.press(todo[0][1])
                pressed.append((round(now, 1), todo.pop(0)[1]))
            png = page.screenshot()
            frames.append(cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR))
            time.sleep(max(0.0, (len(frames) / fps) - (time.time() - start)))
        browser.close()
    video = out_dir / "page.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"avc1"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for f in frames:
        vw.write(f)
    vw.release()
    shots = []
    for k, i in enumerate(np.linspace(0, len(frames) - 1, 4).astype(int)):
        shots.append(out_dir / f"still-{k}.png")
        cv2.imwrite(str(shots[-1]), frames[i])
    return video, shots, errors, pressed, loaded


def critique(video, shots, prompt, model):
    from google import genai
    client = genai.Client()   # GEMINI_API_KEY or GOOGLE_API_KEY from the environment
    f = client.files.upload(file=str(video))
    while getattr(f.state, "name", "") == "PROCESSING":   # videos are processed before they can be used
        time.sleep(2)
        f = client.files.get(name=f.name)
    parts = [f] + [client.files.upload(file=str(s)) for s in shots] + [prompt]
    resp = client.models.generate_content(model=model, contents=parts,
                                          config={"response_mime_type": "application/json"})
    text = resp.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", required=True, help="what this iteration should achieve, in plain words")
    ap.add_argument("--url", default="http://localhost:8780")
    ap.add_argument("--seconds", type=float, default=24)
    ap.add_argument("--keys", default="", help='keys to press, spread over the recording, e.g. "1 ArrowRight KeyO"')
    ap.add_argument("--allow-upload", action="store_true", help="send the recording to Gemini (it shows the personal photos)")
    a = ap.parse_args()

    out = HERE / "reviews" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    video, shots, errors, pressed, loaded = record(a.url, a.seconds, a.keys.split(), out)
    print(f"recorded {video.relative_to(HERE)} (the page took {loaded:.0f} s to load), {len(errors)} console errors")
    prompt = PROMPT.format(goal=a.goal, context=(HERE / "GEMINI.md").read_text()[:12000],
                           keys=pressed or "none", errors=errors[:20] or "none")
    (out / "prompt.txt").write_text(prompt)
    if not a.allow_upload:
        print("not uploaded (add --allow-upload to send it to Gemini); the prompt is in", (out / "prompt.txt").relative_to(HERE))
        return
    model = os.environ.get("ARGON_MODEL")
    if not model:
        sys.exit("set ARGON_MODEL to the Gemini 4 Argon model id (see AI Studio)")
    review = critique(video, shots, prompt, model)
    (out / "review.json").write_text(json.dumps(review, indent=1))
    print(json.dumps(review, indent=1))


if __name__ == "__main__":
    main()
