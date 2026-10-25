# Unfold · Miami Beach: context for Gemini

An evening walk on Miami Beach (7 May 2026, 19:41–20:14) told in dots, in the browser. The place is a pyramid of
nested pages, each the same place resampled four times finer than the one above it (0 Miami Beach 12 m, 1 the beach
3 m, 2 the walk 0.75 m, 3 the scene 0.25 m, 4 up close 0.1 m), so every dot has a parent it condenses out of. The
walk's 22 photos are point clouds of their own (page 7) that condense out of the place one after another along the
timeline.

## Files

- `index.html`: the page, its CSS and the HUD (loading bar, intro, caption, whisper, gauge, photo strip, legend).
  three.js r180 comes from jsDelivr through the import map; there is no build step.
- `src/main.js`: the engine: loading `data/`, the pages and the photos as dots, the camera modes (drift, eye,
  overview), the timeline and playback, the keys, the labels.
- `src/shaders.js`: the dot, brush, sky, trail, finish and glow shaders. `src/bloom.js`: a soft bloom pass.
  `src/sky.js`: the sun's direction and colour.
- `bake.py`: builds `data/` from `source/`: Unfold's 28-byte dots sorted by page, then by a random key (so any prefix
  of a page is an even thinning), the photos frame after frame, and `meta.json` (pages, frames, cameras, moments, sun,
  coast, people).
- `serve.py`: serves the folder on http://localhost:8780.
- `tools/argon_review.py`: records the page and has Gemini critique it against a goal (below).

`data/`, `source/` and `work/` hold personal photos and a Google-derived mesh. They are never committed and never
uploaded; the review tool only uploads a recording when asked to (`--allow-upload`).

## How to build it

Work in small iterations, each with one goal you can see on screen:

1. Change the code toward the goal.
2. Serve the page and record it: `python3 tools/argon_review.py --goal "<the goal>" --keys "<keys to press>"`.
3. Read the critique (`reviews/<time>/review.json`): issues with the second they happen, the likely cause and a fix.
4. Fix, record again, and compare. Stop when the goal is met or the critique only finds low-severity issues.

Keys while recording: `ArrowRight` / `ArrowLeft` step through the photos, `KeyP` plays, `Digit1`–`Digit3` jump to a
moment, `KeyE` the photo's own eye, `KeyF` frame the photo, `KeyO` above it all, `ArrowUp` / `ArrowDown` closer and
further.

## Rules

- Nothing reads outside this folder; the page stays a static site served by `serve.py`.
- Keep the dot format and the page order in `bake.py` and `src/main.js` in step; the format is described in `bake.py`.
- No layer may end in a hard line or a circle: borders feather with wandering edges (see `bake.py`).
- Check the console: a review with console errors is not done.
