# Unfold · Miami Beach

An evening walk on Miami Beach (7 May 2026, 19:41–20:14), told in dots. The place unfolds from the whole of Miami Beach
down to arm's length in nested pages (12 m, 3 m, 0.75 m, 0.25 m, 0.1 m), and the walk's 22 photos condense out of it
one after another along the timeline.

```sh
python3 serve.py        # http://localhost:8780
```

**scroll** or **← →** through the photos · **P** play / pause · **1–3** a moment · **drag** to look around ·
**shift-drag** to move · **↑ ↓** closer, further · **E** the photo's own eye · **F** frame the photo · **O** above it all

## Data

The page reads `data/` (`env.bin`, `photos.bin`, `meta.json`, `hd/`, `thumbs/`), which `bake.py` writes from
`source/`. Both hold personal photos and a Google-derived mesh, so they stay on this machine and are not in git.

## Building it with Gemini 4 Argon

`GEMINI.md` describes the project for Gemini (Gemini CLI reads it on its own). Each iteration has one goal: change the
code, record the page, let Argon critique the recording against the goal, fix, record again.

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-tools.txt
export GEMINI_API_KEY=...            # in your own shell; never committed
export ARGON_MODEL=...               # the Gemini 4 Argon model id, as AI Studio shows it
python3 serve.py &
.venv/bin/python tools/argon_review.py --goal "the photos condense out of the beach, not pop in" \
    --keys "Digit1 ArrowRight ArrowRight KeyO" --allow-upload
```

The recording shows the personal photos as dots: `--allow-upload` sends it to Google. Without the flag the tool only
records (`reviews/<time>/`) and writes the prompt it would send.
