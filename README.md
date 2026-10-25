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
