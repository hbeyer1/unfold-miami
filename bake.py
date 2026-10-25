"""Bake the data for Unfold · Miami Beach (index.html): Unfold's engine on the evening walk of 7 May 2026.

    python3 -I bake.py [stage ...]      stages: mesh pages photos (default: all)

Reads only source/ (personal photos and a Google-derived mesh: local use only, never upload):
  frames/frames.json, frames.bin  the 22 photos as registered point clouds (world/frames/data: World Labs depth where
                                  valid, the Google mesh's ray-cast depth elsewhere; cameras in the mesh's world frame)
  upright/*.jpg                   the photos themselves, upright (selfies: the front camera's mirror image)
  land/land-K.bin, atlas-K.jpg    the Google 3D-tiles mesh in world metres (Together's bake), its texture atlases
  mesh/satellite.tif              the capture's z19 orthophoto (~0.3 m), EPSG:3857
  aerial/naip-wide.jpg, dem-*.tif USGS NAIP 4 km square and 3DEP elevation (public domain)
  dem_world.npz, meta.json        the ground heights on a world-aligned 2 m grid; moments, captions, people, coast

World frame: metres x east, y up (above sea level, USGS), z south of lat0 25.78375, lon0 -80.129.

Writes data/:
  env.bin     the pages of the place: 0 Miami Beach (12 m), 1 the beach (3 m), 2 the walk (0.75 m), 3 the scene (0.25 m),
              4 up close (0.1 m); Unfold's 28-byte dots (see unfold/bake.py), sorted by page then rnd
  photos.bin  the 22 photos as dots, the same layout, page 7, frame after frame, each frame sorted by rnd (so any prefix
              of a frame is an even thinning: a muted frame is drawn as a prefix); 'link' = the person (0 none, 1-4);
              'cover' = 255 for the photo's sky (sparse dots 600 m out, in its own colours)
  meta.json   pages, frames (camera, time, caption, dot range), moments, sun, coast, people
  thumbs/*.jpg

As in Unfold: every page is the same place resampled on a grid four times finer than the page above (nested cells), so
each dot has a parent it condenses out of; no layer ends in a line or a circle (feathers with wandering borders, the
mesh and the flat ground sharing the place with complementary weights, the satellite and NAIP colours blended across a
wide noisy band); the spacing a dot carries is measured before its feathers thin it.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
HERE = Path(__file__).resolve().parent
SRC, OUT, WORK = HERE / "source", HERE / "data", HERE / "work"
HD = OUT / "hd"                                   # the photos at full resolution, one file each (loaded when current)
RNG = np.random.default_rng(23)
SIZE_MIN, SIZE_MAX = math.log(0.001), math.log(2000.0)
PH_SIZE = (math.log(5e-5), math.log(20.0))   # the photos' own range: a selfie's pixel at 0.5 m is ~0.4 mm (below the place's)
LAT0, LON0, R_EARTH, RM, DEG = 25.78375, -80.129, 6371010.0, 6378137.0, math.pi / 180

PAGES = [
    dict(name="Miami Beach", cell=12.0),
    dict(name="the beach", cell=3.0, R=760.0, keep=330000),
    dict(name="the walk", cell=0.75, R=300.0, keep=1250000),
    dict(name="the scene", cell=0.25, R=110.0, keep=1000000),
    dict(name="up close", cell=0.045, R=18.0, keep=1000000),
    dict(name="underfoot", cell=0.015, R=6.0, keep=700000),
    None,
    dict(name="the moment"),
]
MESH_BOX = (-398.0, -636.0, 398.0, 636.0)       # where the Google mesh (and the orthophoto) reach (x0 z0 x1 z1)
NAIP_HALF = 2000.0
NAIP_BAND = 650.0                                # the USGS tile's feathered edge (the far land takes over across it)


def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def size_code(s, rng=None):
    lo, hi = rng or (SIZE_MIN, SIZE_MAX)
    return np.clip(np.round((np.log(np.maximum(s, 1e-6)) - lo) / (hi - lo) * 255), 0, 255).astype(np.uint8)


def u8(x):
    return np.clip(np.round(np.asarray(x, float) * 255), 0, 255).astype(np.uint8)


def pct_rank(v):
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), float)
    r[order] = np.linspace(0, 1, len(v)) if len(v) > 1 else 0.5
    return r


def sobel(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx * gx + gy * gy)


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ---------------------------------------------------------------------------------------------------------- noise
def vnoise(x, z, seed=0):
    xi, zi = np.floor(x).astype(np.int64), np.floor(z).astype(np.int64)
    fx, fz = x - xi, z - zi
    ux, uz = fx * fx * (3 - 2 * fx), fz * fz * (3 - 2 * fz)

    def h(i, k):
        v = (i * 374761393 + k * 668265263 + seed * 1442695041) & 0xFFFFFFFF
        v = ((v ^ (v >> 13)) * 1274126177) & 0xFFFFFFFF
        return (v ^ (v >> 16)) / 4294967295.0
    a, b, c, d = h(xi, zi), h(xi + 1, zi), h(xi, zi + 1), h(xi + 1, zi + 1)
    return (a + (b - a) * ux + (c - a) * uz + (a - b - c + d) * ux * uz) * 2 - 1


def fbm(x, z, seed=0):
    return 0.6 * vnoise(x, z, seed) + 0.3 * vnoise(2.07 * x + 5.3, 2.07 * z - 1.7, seed + 1) + 0.1 * vnoise(4.13 * x - 3.1, 4.13 * z + 7.7, seed + 2)


def feather_d(d, R, x, z, seed):
    """Kept 1 well inside a reach R (d: distance to whatever the reach is around), 0 from R on; the band in between
    wide, its border wandering (domain-warped noise at a third of R): never a circle, a capsule or a line."""
    wx = fbm(x / (0.5 * R) + 3.1, z / (0.5 * R), seed + 50)
    wz = fbm(x / (0.5 * R), z / (0.5 * R) - 2.7, seed + 60)
    n = fbm(x / (0.3 * R) + 0.6 * wx, z / (0.3 * R) + 0.6 * wz, seed)
    return smoothstep(0.95 * R, 0.35 * R, d + 0.32 * R * n) * smoothstep(R, 0.9 * R, d)


def box_feather(x, z, box, band, seed):
    """1 well inside a box, 0 at its edges, the band between wandering."""
    edge = np.minimum(np.minimum(x - box[0], box[2] - x), np.minimum(z - box[1], box[3] - z))
    wx = fbm(x / (2.0 * band) + 3.1, z / (2.0 * band), seed + 50)
    n = fbm(x / (1.2 * band) + 0.8 * wx, z / (1.2 * band) - 0.8 * wx, seed)
    return smoothstep(0.05 * band, band, edge + 0.55 * band * n) * smoothstep(0, 0.08 * band, edge)


# ---------------------------------------------------------------------------------------------------------- geo
def to_latlon(x, z):
    return LAT0 - np.asarray(z) / (R_EARTH * DEG), LON0 + np.asarray(x) / (math.cos(LAT0 * DEG) * R_EARTH * DEG)


def merc(lat, lon):
    return np.asarray(lon) * DEG * RM, np.log(np.tan(math.pi / 4 + np.asarray(lat) * DEG / 2)) * RM


def remap_points(img, x, y, interp=cv2.INTER_LINEAR, cols=1024):
    n = len(x)
    rows = max(1, -(-n // cols))
    pad = rows * cols - n
    mx = np.pad(np.asarray(x, np.float32), (0, pad)).reshape(rows, cols)
    my = np.pad(np.asarray(y, np.float32), (0, pad)).reshape(rows, cols)
    out = np.concatenate([cv2.remap(img, mx[r:r + 16000], my[r:r + 16000], interp, borderMode=cv2.BORDER_REPLICATE)
                          for r in range(0, rows, 16000)], 0)
    return out.reshape(rows * cols, -1)[:n]


class MercImage:
    def __init__(self, img, bbox):
        self.img, self.bbox = img, bbox
        self.h, self.w = img.shape[:2]

    def pix(self, x, z):
        lat, lon = to_latlon(x, z)
        mx, my = merc(lat, lon)
        x0, y0, x1, y1 = self.bbox
        return (mx - x0) / (x1 - x0) * self.w - 0.5, (y1 - my) / (y1 - y0) * self.h - 0.5

    def sample(self, x, z, interp=cv2.INTER_LINEAR):
        u, v = self.pix(np.ravel(x), np.ravel(z))
        inside = (u > 0) & (v > 0) & (u < self.w - 1) & (v < self.h - 1)
        return remap_points(self.img, u, v, interp=interp), inside

    def metres_per_px(self):
        lat, _ = to_latlon(0, 0)
        return (self.bbox[2] - self.bbox[0]) / self.w * math.cos(lat * DEG)


def naip():
    meta = json.loads((SRC / "aerial" / "meta.json").read_text())["scales"]["wide"]
    bbox = [float(v) for v in meta["bbox_3857"].split(",")]
    img = cv2.cvtColor(cv2.imread(str(SRC / "aerial" / "naip-wide.jpg")), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    return MercImage(img, bbox)


def satellite():
    im = Image.open(SRC / "mesh" / "satellite.tif")
    (_, _, _, x0, y1, _), (sx, sy, _) = im.tag_v2[33922], im.tag_v2[33550]
    a = np.asarray(im.convert("RGBA")).astype(np.float32) / 255
    h, w = a.shape[:2]
    return MercImage(a, (x0, y1 - h * sy, x0 + w * sx, y1))


class Heights:
    def __init__(self):
        d = np.load(SRC / "dem_world.npz")
        self.h, self.x0, self.z0, self.cell = d["h"].astype(np.float32), float(d["x0"]), float(d["z0"]), float(d["cell"])

    def __call__(self, x, z):
        u = (np.ravel(np.asarray(x, np.float64)) - self.x0) / self.cell
        v = (np.ravel(np.asarray(z, np.float64)) - self.z0) / self.cell
        return remap_points(self.h, u, v)[:, 0].reshape(np.shape(x))


META = json.loads((SRC / "meta.json").read_text())
FRAMES = json.loads((SRC / "frames" / "frames.json").read_text())["frames"]
MOMENTS = {m["id"]: m for m in META["moments"]}
CENTRES = np.array([MOMENTS[k]["centre"] for k in ("1941", "1944", "2012")])
CAMS = {k: np.mean([f["cam"]["pos"] for f in FRAMES if f["moment"] == k], 0) for k in ("1941", "1944", "2012")}
COAST = np.array(META["coast"])            # [[x, z]], z ascending


def walk_distance(x, z):
    """Horizontal distance to the walk (the polyline through the three moments, south to north)."""
    pts = CENTRES[:, [0, 2]]
    d = np.full(np.shape(x), np.inf)
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        t = np.clip(((x - a[0]) * ab[0] + (z - a[1]) * ab[1]) / (ab @ ab), 0, 1)
        d = np.minimum(d, np.hypot(x - (a[0] + t * ab[0]), z - (a[1] + t * ab[1])))
    return d


def sea_of(x, z, y):
    cx = np.interp(z, COAST[:, 1], COAST[:, 0])
    return (x > cx + 1.5) & (y < 1.2)


def coastness(x, z, scale):
    cx = np.interp(z, COAST[:, 1], COAST[:, 0])
    return np.exp(-np.abs(x - cx) / scale)


# ---------------------------------------------------------------------------------------------------------- grids
def vkey(P, cell):
    q = np.floor(P / cell).astype(np.int64) + (1 << 20)
    return (q[:, 0] << 42) | (q[:, 1] << 21) | q[:, 2]


def voxelize(P, C, cell, extra=()):
    key = vkey(P, cell)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    n = len(uk)
    cen = np.stack([np.bincount(inv, P[:, k], n) for k in range(3)], 1) / cnt[:, None]
    col = np.stack([np.bincount(inv, C[:, k], n) for k in range(3)], 1) / cnt[:, None]
    yy = np.bincount(inv, P[:, 1] ** 2, n) / cnt - cen[:, 1] ** 2
    out = dict(P=cen, C=col, n=cnt, ystd=np.sqrt(np.maximum(yy, 0)))
    for name, v in extra:
        out[name] = np.bincount(inv, v, n) / cnt
    return out


def lookup(keys_sorted, q):
    i = np.clip(np.searchsorted(keys_sorted, q), 0, len(keys_sorted) - 1)
    return np.where(keys_sorted[i] == q, i, -1)


def nearest_xz(Q, R, cell, K=12):
    rx = np.floor(R[:, 0] / cell).astype(np.int64)
    rz = np.floor(R[:, 1] / cell).astype(np.int64)
    rk = (rx << 32) + rz
    order = np.argsort(rk, kind="stable")
    rk_s = rk[order]
    qx = np.floor(Q[:, 0] / cell).astype(np.int64)
    qz = np.floor(Q[:, 1] / cell).astype(np.int64)
    best = np.full(len(Q), np.inf)
    arg = np.full(len(Q), -1)
    for dx in (-1, 0, 1):
        for dz in (-1, 0, 1):
            k = ((qx + dx) << 32) + (qz + dz)
            lo = np.searchsorted(rk_s, k, "left")
            hi = np.searchsorted(rk_s, k, "right")
            for j in range(K):
                idx = lo + j
                ok = idx < hi
                if not ok.any():
                    break
                ri = order[np.minimum(idx, len(order) - 1)]
                d = (R[ri, 0] - Q[:, 0]) ** 2 + (R[ri, 1] - Q[:, 1]) ** 2
                better = ok & (d < best)
                best[better] = d[better]
                arg[better] = ri[better]
    return arg


def local_spacing(P, cell):
    key = vkey(P, 4 * cell)
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return 4 * cell / np.sqrt(cnt[inv])


def raster_salience(P, lum, cell, half, cx, cz, cap=1800):
    """Edges seen from above at the page's scale: luminance at one and three cells, and breaks of slope."""
    n = int(2 * half / cell) + 3
    cr = cell if n <= cap else 2 * half / cap
    n = min(n, cap + 3)
    ix = np.clip(((P[:, 0] - cx + half) / cr).astype(int), 0, n - 1)
    iz = np.clip(((P[:, 2] - cz + half) / cr).astype(int), 0, n - 1)
    Ls = np.zeros((n, n), np.float32); Hs = np.zeros((n, n), np.float32); M = np.zeros((n, n), np.float32)
    np.add.at(Ls, (iz, ix), lum); np.add.at(Hs, (iz, ix), P[:, 1]); np.add.at(M, (iz, ix), 1)
    mask = (M > 0).astype(np.float32)
    Lm = np.where(M > 0, Ls / np.maximum(M, 1), 0).astype(np.float32)
    Hm = np.where(M > 0, Hs / np.maximum(M, 1), 0).astype(np.float32)

    def fill(img, sig):
        return cv2.GaussianBlur(img * mask, (0, 0), sig) / np.maximum(cv2.GaussianBlur(mask, (0, 0), sig), 1e-4)
    e1, e3 = sobel(fill(Lm, 0.9)), sobel(fill(Lm, 2.6))
    bend = sobel(cv2.GaussianBlur(sobel(fill(Hm, 1.2)) / (8 * cr), (0, 0), 1.0))
    on = mask > 0
    nz = lambda e: e / (np.percentile(e[on], 97) + 1e-6) if on.any() else e
    return np.maximum(np.maximum(0.8 * nz(e1), nz(e3)), 0.9 * nz(bend))[iz, ix]


# ---------------------------------------------------------------------------------------------------------- the mesh
def mesh_samples():
    """Area-weighted samples of the mesh near the walk, three densities (for the walk, the scene and up close), each
    with its texture colour, the texture's edge strength and whether it is a wall. -> work/mesh-<k>.npz"""
    land = json.loads((SRC / "land" / "land.json").read_text())
    H = Heights()
    specs = [("walk", PAGES[2]["R"] + 30, 3.2, None), ("scene", PAGES[3]["R"] + 8, 24.0, "centres"), ("close", PAGES[4]["R"] + 4, 760.0, "cams"),
             ("foot", PAGES[5]["R"] + 1.5, 5200.0, "cams")]
    acc = {s[0]: [] for s in specs}
    for m in land["meshes"]:
        t0 = time.time()
        b = (SRC / "land" / m["file"]).read_bytes()
        nv, nt = m["verts"], m["tris"]
        V = np.frombuffer(b, np.float32, nv * 3, 0).reshape(-1, 3)
        UV = np.frombuffer(b, np.uint16, nv * 2, nv * 12).reshape(-1, 2).astype(np.float32) / 65535
        F = np.frombuffer(b, np.uint32, nt * 3, nv * 16).reshape(-1, 3).astype(np.int64)
        a, b_, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
        cr = np.cross(b_ - a, c - a)
        area = 0.5 * np.linalg.norm(cr, axis=1)
        nrm = cr / np.maximum(2 * area[:, None], 1e-9)
        cen = (a + b_ + c) / 3
        dw = walk_distance(cen[:, 0], cen[:, 2])
        img = cv2.cvtColor(cv2.imread(str(SRC / "land" / m["image"])), cv2.COLOR_BGR2RGB)
        A = img.shape[0]
        imgf = cv2.GaussianBlur(img, (0, 0), 0.6).astype(np.float32) / 255
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.2)
        edge = sobel(g)
        del img, g
        for name, reach, rho, around in specs:
            if around == "centres":
                dd = np.min([np.hypot(cen[:, 0] - p[0], cen[:, 2] - p[2]) for p in CENTRES], 0)
            elif around == "cams":
                dd = np.min([np.hypot(cen[:, 0] - p[0], cen[:, 2] - p[2]) for p in CAMS.values()], 0)
            else:
                dd = dw
            sel = np.flatnonzero(dd < reach)
            if not len(sel):
                continue
            cnt = np.floor(area[sel] * rho + RNG.random(len(sel))).astype(np.int64)
            ti = np.repeat(sel, cnt)
            r1, r2 = RNG.random(len(ti)), RNG.random(len(ti))
            s1 = np.sqrt(r1)
            w0, w1, w2 = (1 - s1)[:, None], (s1 * (1 - r2))[:, None], (s1 * r2)[:, None]
            tri = F[ti]
            P = V[tri[:, 0]] * w0 + V[tri[:, 1]] * w1 + V[tri[:, 2]] * w2
            uv = UV[tri[:, 0]] * w0 + UV[tri[:, 1]] * w1 + UV[tri[:, 2]] * w2
            col = remap_points(imgf, uv[:, 0] * A - 0.5, uv[:, 1] * A - 0.5)
            e = remap_points(edge, uv[:, 0] * A - 0.5, uv[:, 1] * A - 0.5)[:, 0]
            acc[name].append(dict(P=P.astype(np.float32), C=col.astype(np.float32), e=e.astype(np.float32), ny=np.abs(nrm[ti, 1]).astype(np.float32)))
            print(f"  {m['file']} {name}: {len(ti)} samples", flush=True)
        del V, UV, F, a, b_, c, cr, imgf, edge
        print(f"  {m['file']} done in {time.time() - t0:.0f} s", flush=True)
    WORK.mkdir(exist_ok=True)
    for name, parts in acc.items():
        d = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        # the mesh's sea sits low and flat: lift it onto the sea level (no dots under the water)
        sea = sea_of(d["P"][:, 0], d["P"][:, 2], d["P"][:, 1])
        d["P"][sea, 1] = np.maximum(d["P"][sea, 1], 0.0)
        d["sea"] = sea
        np.savez(WORK / f"mesh-{name}.npz", **d)
        print(f"mesh {name}: {len(d['P'])} samples ({sea.mean():.0%} sea)", flush=True)


# ---------------------------------------------------------------------------------------------------------- pages
def ground_grid(cell, cx, cz, half, colour, height, keep_fn):
    """The flat ground on an even grid of the cell: colour and height functions of (x, z) at cell centres, kept where
    keep_fn (x, z) says (a float weight, the caller's feather)."""
    n = int(np.ceil(2 * half / cell))
    vv, uu = np.mgrid[0:n, 0:n]
    x = ((uu + 0.5) * cell - half + cx).ravel()
    z = ((vv + 0.5) * cell - half + cz).ravel()
    w = keep_fn(x, z)
    ok = w > 0.002
    x, z, w = x[ok], z[ok], w[ok]
    x = x + (RNG.random(len(x)) - 0.5) * 0.8 * cell
    z = z + (RNG.random(len(z)) - 0.5) * 0.8 * cell
    C = colour(x, z)
    y = height(x, z)
    return dict(P=np.stack([x, y, z], 1), C=C, w=w)


def build_pages():
    t0 = time.time()
    H = Heights()
    NA = naip()
    SA = satellite()
    # NAIP toned to the orthophoto where both see the place (Lab mean and spread), so their hand-over has no seam
    xs = RNG.uniform(-380, 380, 60000); zs = RNG.uniform(-620, 620, 60000)
    cs, ins = SA.sample(xs, zs)
    cn, _ = NA.sample(xs, zs)
    ok = ins & (cs[:, 3] > 0.99)
    lab = lambda c: cv2.cvtColor(np.clip(c, 0, 1).reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_RGB2LAB).reshape(-1, 3)
    ls_, ln_ = lab(cs[ok, :3]), lab(cn[ok])
    tone = (ln_.mean(0), ln_.std(0) + 1e-3, ls_.mean(0), ls_.std(0))

    def naip_toned(c):
        L = (lab(c) - tone[0]) / tone[1] * tone[3] + tone[2]
        return cv2.cvtColor(L.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_LAB2RGB).reshape(-1, 3)

    blur = {}

    def colour_at(cell):
        """Colour at the scale of a cell: the orthophoto inside the mesh box, NAIP (toned) outside, blended over a wide
        noisy band (complementary weights)."""
        if cell not in blur:
            ms = SA.metres_per_px(); mn = NA.metres_per_px()
            blur[cell] = (MercImage(cv2.GaussianBlur(SA.img, (0, 0), max(0.5, 0.45 * cell / ms)), SA.bbox),
                          MercImage(cv2.GaussianBlur(NA.img, (0, 0), max(0.5, 0.45 * cell / mn)), NA.bbox))
        sa, na = blur[cell]

        def f(x, z):
            cs_, ins_ = sa.sample(x, z)
            cn_, _ = na.sample(x, z)
            wsat = box_feather(x, z, MESH_BOX, 60.0, 11) * (cs_[:, 3] > 0.98) * ins_
            return cs_[:, :3] * wsat[:, None] + naip_toned(cn_) * (1 - wsat[:, None])
        return f

    pages = {}
    walk_c = CENTRES.mean(0)
    # 0 Miami Beach: the NAIP square, 12 m, its edges feathered
    pg = ground_grid(12.0, 90.0, 17.0, NAIP_HALF, colour_at(12.0), lambda x, z: np.maximum(H(x, z), 0.0),
                     lambda x, z: box_feather(x - 90, z - 17, (-NAIP_HALF, -NAIP_HALF, NAIP_HALF, NAIP_HALF), NAIP_BAND, 5))
    pages[0] = pg
    # mesh top heights (roofs) on a 3 m grid, for the beach page (2.5D); where the mesh ends: the bare ground
    walk = np.load(WORK / "mesh-walk.npz")
    top = {}
    k2 = np.floor(walk["P"][:, [0, 2]] / 3.0).astype(np.int64)
    kk = (k2[:, 0] << 32) + k2[:, 1]
    uk, inv = np.unique(kk, return_inverse=True)
    ymax = np.full(len(uk), -1e9); np.maximum.at(ymax, inv, walk["P"][:, 1])
    top = (uk, ymax)
    del walk

    def beach_height(x, z):
        g = np.maximum(H(x, z), 0.0)
        k = (np.floor(x / 3.0).astype(np.int64) << 32) + np.floor(z / 3.0).astype(np.int64)
        i = lookup(top[0], k)
        m = np.where(i >= 0, top[1][np.maximum(i, 0)], -1e9)
        wm = box_feather(x, z, MESH_BOX, 60.0, 11)
        return np.where(m > -1e8, g + (np.maximum(m, g) - g) * wm, g)

    # 1 the beach: 3 m, within ~760 m of the walk, 2.5D (roofs at their height)
    R1 = PAGES[1]["R"]
    pages[1] = ground_grid(3.0, walk_c[0], walk_c[2], R1 + 450, colour_at(3.0), beach_height,
                           lambda x, z: feather_d(walk_distance(x, z), R1, x, z, 101))
    # 2-4: the mesh in 3D (cells of the page), sharing the place with the flat ground where the mesh ends
    for L, src, around in ((2, "walk", "walk"), (3, "scene", "centres"), (4, "close", "cams"), (5, "foot", "cams")):
        R, cell = PAGES[L]["R"], PAGES[L]["cell"]
        d = np.load(WORK / f"mesh-{src}.npz")
        if around == "walk":
            reach = lambda x, z: feather_d(walk_distance(x, z), R, x, z, 200 + L)
        else:
            pts = CENTRES if around == "centres" else np.array(list(CAMS.values()))
            reach = lambda x, z, pts=pts: np.max([feather_d(np.hypot(x - p[0], z - p[2]), R, x - p[0], z - p[2], 200 + L + 7 * i) for i, p in enumerate(pts)], 0)
        vb = voxelize(d["P"].astype(np.float64), d["C"].astype(np.float64), cell, extra=(("e", d["e"].astype(np.float64)), ("ny", d["ny"].astype(np.float64)), ("sea", d["sea"].astype(np.float64))))
        del d
        wm_b = box_feather(vb["P"][:, 0], vb["P"][:, 2], MESH_BOX, 40.0, 11)
        parts = [dict(P=vb["P"], C=vb["C"], e=vb["e"], ny=vb["ny"], sea=vb["sea"] > 0.5, ystd=vb["ystd"], K=wm_b, w=reach(vb["P"][:, 0], vb["P"][:, 2]))]
        if L == 2:   # beyond the mesh (the north end of the walk): the flat ground, by the complementary weight
            cx, cz = walk_c[0], walk_c[2]
            half = 0.5 * np.ptp(CENTRES[:, 2]) + R + 10
            g = ground_grid(cell, cx, cz, half, colour_at(cell), lambda x, z: np.maximum(H(x, z), 0.0),
                            lambda x, z: reach(x, z) * (1 - box_feather(x, z, MESH_BOX, 40.0, 11)))
            ng = len(g["P"])
            parts.append(dict(P=g["P"], C=g["C"], e=np.zeros(ng), ny=np.ones(ng), sea=sea_of(g["P"][:, 0], g["P"][:, 2], g["P"][:, 1]),
                              ystd=np.zeros(ng), K=np.ones(ng), w=g["w"]))
        v = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        pages[L] = v
        print(f"page {L}: {len(v['P'])} cells", flush=True)
    for L in (0, 1):
        v = pages[L]
        n = len(v["P"])
        v.update(e=np.zeros(n), ny=np.ones(n), sea=sea_of(v["P"][:, 0], v["P"][:, 2], v["P"][:, 1]), ystd=np.zeros(n), K=np.ones(n))
        print(f"page {L}: {n} cells", flush=True)
    print(f"pages built in {time.time() - t0:.0f} s", flush=True)

    # parents: L1 -> L0 by column; L2 -> L1 by column; L3, L4 -> the page above by 3D cell, else column, else nearest
    for L in (1, 2, 3, 4, 5):
        child, parent = pages[L], pages[L - 1]
        pc = PAGES[L - 1]["cell"]
        k3p = vkey(parent["P"], pc); o3 = np.argsort(k3p); k3 = k3p[o3]
        k2p = vkey(parent["P"] * [1, 0, 1], pc); o2 = np.argsort(k2p, kind="stable"); k2 = k2p[o2]
        i3 = lookup(k3, vkey(child["P"], pc)) if L >= 3 else np.full(len(child["P"]), -1)
        i2 = lookup(k2, vkey(child["P"] * [1, 0, 1], pc))
        pi = np.where(i3 >= 0, o3[np.maximum(i3, 0)], np.where(i2 >= 0, o2[np.maximum(i2, 0)], -1))
        miss = pi < 0
        if miss.any():
            pi[np.flatnonzero(miss)] = nearest_xz(child["P"][miss][:, [0, 2]], parent["P"][:, [0, 2]], pc * 3)
        child["pi"] = pi
        child["par"] = np.where((pi >= 0)[:, None], parent["P"][np.maximum(pi, 0)] - child["P"], 0.0)
        print(f"  page {L}: {(pi < 0).sum()} without a parent", flush=True)
    pages[0]["par"] = np.zeros_like(pages[0]["P"])

    # salience: from above at the page's scale, the texture's edges for the mesh, vertical structure (walls, towers)
    for L in range(6):
        v = pages[L]
        lum = v["C"] @ [0.2126, 0.7152, 0.0722]
        half = NAIP_HALF if L == 0 else (PAGES[L]["R"] + 600 if L == 1 else 0.5 * np.ptp(v["P"][:, 2]) + 5)
        cxz = (90.0, 17.0) if L == 0 else (float(np.mean(v["P"][:, 0])), float(np.mean(v["P"][:, 2])))
        e = raster_salience(v["P"], lum, PAGES[L]["cell"], half, cxz[0], cxz[1])
        sal = pct_rank(e + 1e-6 * RNG.random(len(e)))
        if L >= 2:
            tex = pct_rank(v["e"] + 1e-6 * RNG.random(len(e)))
            wall = smoothstep(0.6, 0.2, v["ny"])
            sal = np.maximum(0.55 * sal + 0.45 * tex, 0.35 * wall + 0.65 * tex * wall)
            sal = pct_rank(sal + 1e-6 * RNG.random(len(sal)))
        coast = coastness(v["P"][:, 0], v["P"][:, 2], max(2.0 * PAGES[L]["cell"], 4.0)) * ~v["sea"]
        surf = coastness(v["P"][:, 0], v["P"][:, 2], max(3.0 * PAGES[L]["cell"], 8.0)) * v["sea"] * np.clip((lum - 0.3) * 3, 0, 1)
        v["coast"] = np.maximum(coast, surf)
        v["sal"] = np.maximum(np.where(v["sea"], 0.15 * sal, sal), v["coast"])

    # cover: how much of a dot its children replace (their reach times their share), and thinning to a budget
    for L in range(6):
        v = pages[L]
        if L < 5:
            c = pages[L + 1]
            ok = c["pi"] >= 0
            nchild = np.bincount(c["pi"][ok], minlength=len(v["P"]))
            esum = np.bincount(c["pi"][ok], (c["w"] * c["K"])[ok], minlength=len(v["P"]))
            v["cover"] = np.where(nchild > 0, esum / np.maximum(nchild, 1), 0.0)
        else:
            v["cover"] = np.zeros(len(v["P"]))
    # the colours of a beach at the end of the day, from the daytime orthophoto: the sea turquoise in the shallows, blue
    # further out; the beach's sand warm and pale (only light, unsaturated land between the waterline and the dunes)
    for L in range(6):
        v = pages[L]
        off = v["P"][:, 0] - np.interp(v["P"][:, 2], COAST[:, 1], COAST[:, 0])
        sea = v["sea"]
        t = np.clip(off / 260.0, 0, 1)[:, None]
        tint = np.array([0.32, 0.74, 0.72]) * (1 - t) + np.array([0.09, 0.30, 0.50]) * t
        v["C"] = np.where(sea[:, None], 0.45 * v["C"] + 0.55 * tint, v["C"])
        mx, mn = v["C"].max(1), v["C"].min(1)
        sat = (mx - mn) / np.maximum(mx, 1e-3)
        # (smooth weights only: no patches where the orthophoto's noise crosses a threshold)
        w_ = (~sea * smoothstep(-125, -80, off) * smoothstep(4, -1, off) * smoothstep(7, 4, v["P"][:, 1])
              * smoothstep(0.34, 0.12, sat) * smoothstep(0.18, 0.36, mx))[:, None]
        warm = np.clip(v["C"] * np.array([1.14, 1.04, 0.86]) * 1.06, 0, 1)
        v["C"] = v["C"] * (1 - w_) + warm * w_
    chunks = []
    for L in range(6):
        v = pages[L]
        n = len(v["P"])
        F = v["w"] * v["K"]
        wgt = 0.3 + 0.7 * v["sal"]
        keep = PAGES[L].get("keep", 260000)
        lo, hi = 0.0, 1e6
        for _ in range(60):
            k = 0.5 * (lo + hi)
            if (np.minimum(1, k * wgt) * F).sum() > keep:
                hi = k
            else:
                lo = k
        # the sea: legible but lighter than the land: denser near the shore (turquoise, the surf), thinning offshore
        offshore = v["P"][:, 0] - np.interp(v["P"][:, 2], COAST[:, 1], COAST[:, 0])
        sea_keep = np.clip(0.12 + 0.45 * np.exp(-np.maximum(offshore, 0) / 90.0) + v["coast"] * 1.5, 0, 1)
        budget = np.minimum(1, lo * wgt) * np.where(v["sea"], sea_keep, 1.0)
        if L == 0:
            budget = np.where(v["sea"], 0.0, budget)   # (the ocean field below is page 0's sea)
        base = RNG.random(n) < budget
        take = base & (RNG.random(n) < F)
        P_t = v["P"][take].copy()
        jit = v["sea"][take]
        P_t[jit, 0] += (RNG.random(jit.sum()) - 0.5) * 2.5 * PAGES[L]["cell"]
        P_t[jit, 2] += (RNG.random(jit.sum()) - 0.5) * 2.5 * PAGES[L]["cell"]
        flat = L <= 1
        spc_full = np.zeros(n)
        spc_full[base] = local_spacing(v["P"][base] * ([1, 0, 1] if flat else 1), PAGES[L]["cell"])
        spc = np.maximum(spc_full[take], PAGES[L]["cell"])
        # (focus: the coast and surf; negative on the sea, so the viewer can let the water swell)
        foc = np.where(v["sea"][take], -(0.05 + v["coast"][take]), v["coast"][take])
        chunks.append(dict(P=P_t, C=v["C"][take], sal=v["sal"][take], page=np.full(take.sum(), L), link=np.full(take.sum(), L + 1),
                           cover=v["cover"][take], S=spc, par=v["par"][take] + (v["P"][take] - P_t), focus=foc))
        print(f"  page {L}: kept {take.sum()} of {n}", flush=True)
    chunks += ocean_and_beyond(H, NA, R1)
    allp = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    OUT.mkdir(exist_ok=True)
    # tiles: each fine page cut into squares (drawn or skipped as a whole, and each drawn only as far into its
    # rnd-sorted dots as the screen can hold from where you are: level of detail without a per-dot cost)
    TILE = {0: 1e9, 1: 1e9, 2: 220.0, 3: 60.0, 4: 14.0, 5: 6.0}
    rnd = RNG.random(len(allp["P"]))
    ts = np.array([TILE[int(L)] for L in range(6)])[allp["page"]]
    tx = np.floor(allp["P"][:, 0] / ts).astype(np.int64)
    tz = np.floor(allp["P"][:, 2] / ts).astype(np.int64)
    tkey = (tx + 100000) * 200001 + (tz + 100000)
    order = np.lexsort((rnd, tkey, allp["page"]))
    n, ranges = write_bin(OUT / "env.bin", allp["P"], allp["C"], allp["sal"], allp["page"], allp["link"], rnd,
                          allp["cover"], allp["S"], allp["par"], allp["focus"], order=order)
    pg_o, tk_o, P_o, S_o = allp["page"][order], tkey[order], allp["P"][order], allp["S"][order]
    brk = np.flatnonzero(np.r_[True, (pg_o[1:] != pg_o[:-1]) | (tk_o[1:] != tk_o[:-1]), True])
    tiles = []
    for a, b in zip(brk[:-1], brk[1:]):
        Pt = P_o[a:b]
        lo_, hi_ = Pt.min(0), Pt.max(0)
        c_ = 0.5 * (lo_ + hi_)
        tiles.append([int(pg_o[a]), int(a), int(b - a)] + [round(float(v), 2) for v in c_]
                     + [round(float(np.linalg.norm(hi_ - lo_) / 2), 2), round(float(np.percentile(S_o[a:b], 99)), 4)])
    print(f"  {len(tiles)} tiles", flush=True)
    # keep the finest cells for the photos' parents
    np.savez(WORK / "parents.npz", **{f"P{L}": pages[L]["P"].astype(np.float32) for L in (2, 3, 4, 5)})
    print(f"env.bin: {n} dots, {os.path.getsize(OUT / 'env.bin') / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    return dict(file="env.bin", count=int(n), pages=ranges, tiles=tiles)



def coast_x(z):
    """The shoreline's x at z, the capture's polyline extended straight beyond its ends."""
    z = np.asarray(z, float)
    x = np.interp(z, COAST[:, 1], COAST[:, 0])
    a, b = COAST[:40], COAST[-40:]
    s0 = (a[-1, 0] - a[0, 0]) / (a[-1, 1] - a[0, 1])
    s1 = (b[-1, 0] - b[0, 0]) / (b[-1, 1] - b[0, 1])
    x = np.where(z < COAST[0, 1], COAST[0, 0] + (z - COAST[0, 1]) * s0, x)
    return np.where(z > COAST[-1, 1], COAST[-1, 0] + (z - COAST[-1, 1]) * s1, x)


def ocean_and_beyond(H, NA, R1):
    """Page 0's far field: the Atlantic out to ~9 km offshore and 9 km up and down the coast (dots ever sparser with
    the distance from the shore, continuous), and the land past the USGS tile as a faint ring whose colours are the tile's
    own edge, blurred and carried outward (an extrapolation: no imagery reaches there). Both give way smoothly: the ocean
    to the beach page near the walk (by exactly its reach), the far land to the tile (by exactly its feather)."""
    out = []
    # ---- the ocean: rows along the shore, row spacing and dot spacing growing with the distance offshore
    rows, x_off = [], 1.5
    while x_off < 9000:
        sp = min(2.0 * (1 + x_off / 90.0), 140.0)
        rows.append((x_off, sp))
        x_off += sp
    P, S, OFF = [], [], []
    for off, sp in rows:
        z = np.arange(-9000.0, 9000.0, sp) + RNG.uniform(0, sp, 1)
        z = z + RNG.uniform(-0.45, 0.45, len(z)) * sp
        o = off + RNG.uniform(-0.45, 0.45, len(z)) * sp
        P.append(np.stack([coast_x(z) + o, np.full(len(z), -0.05), z], 1)); S.append(np.full(len(z), sp)); OFF.append(o)
    P, S, OFF = np.concatenate(P), np.concatenate(S), np.concatenate(OFF)
    # thins out up and down the coast far from the place (a soft end, far beyond what the eye reaches)
    zfade = smoothstep(9000, 5000, np.abs(P[:, 2] - 17.0) + 800 * fbm(P[:, 0] / 2500, P[:, 2] / 2500, 31))
    keep = RNG.random(len(P)) < zfade
    P, S, OFF = P[keep], S[keep], OFF[keep]
    t1 = np.clip(OFF / 300.0, 0, 1)[:, None]
    t2 = np.clip((OFF - 300.0) / 4000.0, 0, 1)[:, None]
    shallow, mid, deep = np.array([0.30, 0.70, 0.68]), np.array([0.10, 0.33, 0.52]), np.array([0.05, 0.17, 0.33])
    C = (shallow * (1 - t1) + mid * t1) * (1 - t2) + deep * t2
    C = np.clip(C * (0.9 + 0.2 * RNG.random((len(P), 1))), 0, 1)
    surf = np.exp(-np.maximum(OFF, 0) / 9.0)
    C = C * (1 - 0.6 * surf[:, None]) + 0.6 * surf[:, None] * np.array([0.92, 0.95, 0.95])
    sal = np.clip(0.45 + 0.25 * fbm(P[:, 0] / 40.0, P[:, 2] / 14.0, 33), 0, 1)
    # where the beach page reaches, its own sea takes over (the same feather as its reach: complementary)
    cover = feather_d(walk_distance(P[:, 0], P[:, 2]), R1, P[:, 0], P[:, 2], 101)
    n = len(P)
    out.append(dict(P=P, C=C, sal=sal, page=np.zeros(n, int), link=np.ones(n, int), cover=cover, S=S, par=np.zeros((n, 3)),
                    focus=-(0.05 + surf)))
    print(f"  ocean: {n} dots to ~9 km offshore", flush=True)
    # ---- the land past the tile: a ring from the tile's feathered edge out to ~8 km, west of the shore
    rng_r = []
    r = NAIP_HALF - NAIP_BAND
    while r < 8000:
        sp = 30.0 * (1 + (r - NAIP_HALF + NAIP_BAND) / 1500.0)
        rng_r.append((r, min(sp, 160.0)))
        r += min(sp, 160.0)
    pts = []
    for r, sp in rng_r:   # square rings (the tile is a square), jittered: the edge never shows as a line
        nside = int(2 * r / sp)
        tt = (np.arange(nside) + RNG.random(nside)) * sp - r
        for side in range(4):
            if side == 0: q = np.stack([tt, np.full(nside, -r)], 1)
            elif side == 1: q = np.stack([tt, np.full(nside, r)], 1)
            elif side == 2: q = np.stack([np.full(nside, -r), tt], 1)
            else: q = np.stack([np.full(nside, r), tt], 1)
            q = q + RNG.uniform(-0.5, 0.5, q.shape) * sp
            pts.append(np.concatenate([q, np.full((nside, 1), sp)], 1))
    q = np.concatenate(pts)
    x, z, sp = q[:, 0] + 90.0, q[:, 1] + 17.0, q[:, 2]
    land = x < coast_x(z) - 30
    w = (1 - box_feather(x - 90, z - 17, (-NAIP_HALF, -NAIP_HALF, NAIP_HALF, NAIP_HALF), NAIP_BAND, 5)) * land
    w *= smoothstep(8000, 5500, np.maximum(np.abs(x - 90), np.abs(z - 17)) + 900 * fbm(x / 2000, z / 2000, 41))
    keep = RNG.random(len(x)) < w
    x, z, sp = x[keep], z[keep], sp[keep]
    blur = MercImage(cv2.GaussianBlur(NA.img, (0, 0), 120.0), NA.bbox)
    m = NAIP_HALF - 260.0
    cx_, cz_ = np.clip(x - 90, -m, m) + 90, np.clip(z - 17, -m, m) + 17
    col, _ = blur.sample(cx_, cz_)
    col = np.clip(col * 0.85, 0, 1)
    n = len(x)
    out.append(dict(P=np.stack([x, np.full(n, 1.0), z], 1), C=col, sal=np.full(n, 0.35), page=np.zeros(n, int), link=np.ones(n, int),
                    cover=np.zeros(n), S=sp, par=np.zeros((n, 3)), focus=np.zeros(n)))
    print(f"  far land: {n} dots", flush=True)
    return out


def write_bin(path, P, C, sal, page, link, rnd, cover, size, par, focus, order=None):
    o = np.lexsort((rnd, page)) if order is None else order
    pos = P[o].astype("<f4")
    col = np.concatenate([u8(C[o]), u8(sal[o])[:, None]], 1)
    att = np.stack([(page[o] + 16 * link[o]).astype(np.uint8), u8(rnd[o]), u8(cover[o]), size_code(size[o])], 1)
    ph = np.concatenate([par[o], focus[o][:, None]], 1).astype("<f2")
    with open(path, "wb") as f:
        for a in (pos, col, att, ph):
            f.write(np.ascontiguousarray(a).tobytes())
    ranges = {}
    ps = page[o]
    for L in np.unique(ps):
        idx = np.flatnonzero(ps == L)
        ranges[int(L)] = [int(idx[0]), int(len(idx))]
    return len(P), ranges


# ---------------------------------------------------------------------------------------------------------- the photos
REC = np.dtype([("p", "<f4", 3), ("q", "<f4", 3), ("c", "u1", 4), ("a", "u1", 4)])


def photo_crop(f, r, img_full):
    """The registered frame covers a centred part of the full photo (World Labs got a crop for some): find which, by the
    colours of the person dots (which kept the photo's own colours)."""
    H0, W0 = img_full.shape[:2]
    small = cv2.GaussianBlur(cv2.resize(img_full, (W0 // 2, H0 // 2), interpolation=cv2.INTER_AREA), (0, 0), 1.5)
    Hs, Ws = small.shape[:2]
    c = f["cam"]
    R = quat_to_mat(c["quat"]); C = np.array(c["pos"])
    L = (r["p"] - C) @ R
    z = -L[:, 2]
    xn, yn = L[:, 0] / z, L[:, 1] / z
    sel = r["c"][:, 3] > 0
    if sel.sum() < 2000:
        sel = np.ones(len(r), bool)
    idx = np.flatnonzero(sel)[::5]
    tv = math.tan(math.radians(c["fov"]) / 2)
    best = None
    for sc in np.arange(0.70, 1.031, 0.005):
        fy = (Hs * sc / 2) / tv
        fx = fy * (Ws / Hs) / c["aspect"]
        u = Ws / 2 + fx * xn[idx] - 0.5
        v = Hs / 2 - fy * yn[idx] - 0.5
        if u.min() < 0 or v.min() < 0 or u.max() > Ws - 1 or v.max() > Hs - 1:
            continue
        e = np.abs(small[np.round(v).astype(int), np.round(u).astype(int)] - r["c"][idx, :3]).mean()
        if best is None or e < best[0]:
            best = (e, sc)
    return best



def develop(img):
    """The photo brought up like a print: a gentle tone curve on luminance (shadows and mid-tones lifted, highlights
    eased toward white, never clipped: hue kept, so skin stays skin), a touch more colour, a slight unsharp mask."""
    Y = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2])[..., None]
    Y = np.clip(Y, 0, 1)
    Yt = Y + 1.1 * Y * (1 - Y) ** 2                  # shadows and mid-tones up, the top nearly untouched (faces keep their tone)
    Yt = np.where(Yt > 0.86, 0.86 + (Yt - 0.86) * 0.65, Yt)   # and a soft shoulder: nothing reaches flat white
    out = img * (Yt / np.maximum(Y, 1e-4))
    out = Yt + (out - Yt) * 1.12                      # saturation, around the new luminance
    m = out.max(-1, keepdims=True)                   # a channel over the top: pulled toward the luminance, not clipped
    lim = np.maximum(0.97, Yt)
    out = Yt + (out - Yt) * np.where(m > lim, (lim - Yt) / np.maximum(m - Yt, 1e-4), 1.0)
    out = np.clip(out, 0, 1).astype(np.float32)
    blur = cv2.GaussianBlur(out, (0, 0), 1.1)
    return np.clip(out + 0.45 * (out - blur), 0, 0.97)


def bake_photos():
    t0 = time.time()
    allr = np.fromfile(SRC / "frames" / "frames.bin", REC)
    par = np.load(WORK / "parents.npz")
    parent_keys = {}
    for L in (5, 4, 3, 2):
        k = vkey(par[f"P{L}"].astype(np.float64), PAGES[L]["cell"])
        o = np.argsort(k)
        parent_keys[L] = (k[o], o, par[f"P{L}"])
    (OUT / "thumbs").mkdir(parents=True, exist_ok=True)
    out = open(OUT / "photos.bin.tmp", "wb")
    chunks = []
    frames_meta = []
    total = 0
    for k, f in enumerate(FRAMES):
        o, n = f["dots"]
        r = allr[o:o + n]
        stem = f["stem"]
        selfie = stem[-9:].startswith("201")
        full = np.asarray(Image.open(SRC / "upright" / f"{stem}.jpg").convert("RGB"))
        if selfie:
            full = full[:, ::-1]
        full = np.ascontiguousarray(full).astype(np.float32)
        err, sc = photo_crop(f, r, full)
        c = f["cam"]
        W, Hh = (768, 1024) if c["aspect"] < 1 else (1024, 768)
        # the registered frame's picture: the full photo's centred crop at the frame's resolution
        H0, W0 = full.shape[:2]
        cw, ch = W0 * sc, H0 * sc
        crop = full[int(round((H0 - ch) / 2)):int(round((H0 + ch) / 2)), int(round((W0 - cw) / 2)):int(round((W0 + cw) / 2))]
        img = cv2.resize(crop, (W, Hh), interpolation=cv2.INTER_AREA) / 255.0
        del full, crop
        img = develop(img)
        R = quat_to_mat(c["quat"]); C = np.array(c["pos"])
        fy = (Hh / 2) / math.tan(math.radians(c["fov"]) / 2)
        fx = fy * (W / Hh) / c["aspect"]
        L = (r["p"] - C) @ R
        z = -L[:, 2]
        u = W / 2 + fx * L[:, 0] / z - 0.5
        v = Hh / 2 - fy * L[:, 1] / z - 0.5
        ok = (z > 0.05) & (u > -0.5) & (v > -0.5) & (u < W - 0.5) & (v < Hh - 0.5)
        ui, vi = np.clip(np.round(u[ok]).astype(int), 0, W - 1), np.clip(np.round(v[ok]).astype(int), 0, Hh - 1)
        # a dense depth from the registered dots: each pixel takes its nearest dot's depth (and person), up to 2.6 px away
        zmap = np.zeros((Hh, W), np.float32); pmap = np.zeros((Hh, W), np.uint8); has = np.zeros((Hh, W), np.uint8)
        zmap[vi, ui] = z[ok]; pmap[vi, ui] = r["c"][ok, 3]; has[vi, ui] = 1
        dist, lab = cv2.distanceTransformWithLabels((1 - has).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        seeds = np.zeros(lab.max() + 1, np.int64)
        sy, sx = np.nonzero(has)
        seeds[lab[sy, sx]] = sy * W + sx
        src = seeds[lab]
        zd = zmap.ravel()[src].reshape(Hh, W)
        pd = pmap.ravel()[src].reshape(Hh, W)
        valid = dist < 2.6
        # depth steps: drop the pixels smeared across a jump; the near side of a jump is an outline
        lz = np.log(np.maximum(zd, 1e-3)).astype(np.float32)
        zmax = cv2.dilate(lz, np.ones((5, 5), np.uint8)); zmin = cv2.erode(lz, np.ones((5, 5), np.uint8))
        flying = ((lz - zmin) > 0.2) & ((zmax - lz) > 0.2)
        outline = np.clip((zmax - lz - 0.12) / 0.25, 0, 1)
        valid &= ~flying
        # (how far each pixel is from one the photo has depth for: the full-resolution layer fills gaps from it)
        dh, lh = cv2.distanceTransformWithLabels((~valid).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)
        g = sobel(cv2.GaussianBlur(lum, (0, 0), 1.0))
        sal_lum = pct_rank(g.ravel()).reshape(Hh, W)
        sal = np.maximum(sal_lum ** 1.2, outline)
        person = (pd > 0) & valid
        pm = cv2.morphologyEx(person.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        sal = np.maximum(sal, pm.astype(np.float32))
        dperson = cv2.distanceTransform((~person).astype(np.uint8), cv2.DIST_L2, 5)
        focus = np.where(person, 1.0, 0.55 * np.exp(-(dperson / (0.035 * Hh)) ** 2))
        hole_d = cv2.distanceTransform(valid.astype(np.uint8), cv2.DIST_L2, 5)
        p = (0.08 + 0.55 * sal ** 1.4) * np.clip(hole_d / 6.0, 0.25, 1)
        p = np.where(person, np.maximum(p, 0.5), p)
        take = valid & (RNG.random((Hh, W)) < p)
        # the sky (no depth anywhere near, the ray above the horizon): sparse dots far out, in the photo's own colours
        # (the pastel haze, the afterglow); the border between the sky and the ground the photo has depth for frays
        vg, ug = np.mgrid[0:Hh, 0:W]
        ray_y = (np.stack([(ug + 0.5 - W / 2) / fx, -(vg + 0.5 - Hh / 2) / fy, -np.ones((Hh, W))], -1) @ R.T)[..., 1]
        ray_y = ray_y / np.sqrt(((ug + 0.5 - W / 2) / fx) ** 2 + ((vg + 0.5 - Hh / 2) / fy) ** 2 + 1)
        sky = ~valid & (dist > 6) & (ray_y > -0.02)
        SKYFILL = 56   # (the sky right along what stands against it, a head, a tower's roof, is the full-res layer's)
        near_sky = sky & (dh < SKYFILL)
        sky_d = np.clip(cv2.distanceTransform(sky.astype(np.uint8), cv2.DIST_L2, 5) / 14.0, 0, 1)
        # denser low (the glow along the horizon), thinning upward
        up = np.clip(ray_y / 0.35, 0, 1)
        take_sky = sky & ~near_sky & (RNG.random((Hh, W)) < 0.07 * sky_d * (1 - up) ** 1.5)
        zd = np.where(take_sky, 600.0, zd)
        valid = valid | take_sky
        take = take | take_sky
        p = np.where(take_sky, 0.06, p)
        vv, uu = np.nonzero(take)
        uj = uu + RNG.uniform(-0.5, 0.5, len(uu)); vj = vv + RNG.uniform(-0.5, 0.5, len(vv))
        zz = zd[vv, uu]
        Pc = np.stack([(uj + 0.5 - W / 2) / fx * zz, -(vj + 0.5 - Hh / 2) / fy * zz, -zz], 1) @ R.T + C
        dcam = np.linalg.norm(Pc - C, axis=1)
        # (a pixel registered beyond ~500 m, the sea toward the horizon, is brought in to 500 m along its ray rather than
        # dropped: dropped, it left holes in the picture, seen from its eye)
        far_ = (dcam > 500) & ~take_sky[vv, uu]
        Pc[far_] = C + (Pc[far_] - C) * (500.0 / dcam[far_])[:, None]
        dcam = np.where(far_, 500.0, dcam)
        keep = dcam < 900
        Pc, uu, vv, zz, dcam = Pc[keep], uu[keep], vv[keep], zz[keep], dcam[keep]
        col = img[vv, uu]
        pr = p[vv, uu]
        size = dcam / fx / np.sqrt(np.maximum(pr, 0.03))
        # parents: the finest page of the place that has a cell where the dot is
        parent = Pc.copy().astype(np.float64)
        found = np.zeros(len(Pc), bool)
        for Lp in (5, 4, 3, 2):
            ks, oo, PP = parent_keys[Lp]
            i = lookup(ks, vkey(Pc.astype(np.float64), PAGES[Lp]["cell"]))
            hit = (i >= 0) & ~found
            parent[hit] = PP[oo[i[hit]]]
            found |= hit
        # (a dot in the air, a person, has no cell of the place: its parent is the nearest ground dot below it)
        miss = ~found
        if miss.any():
            for Lp in (5, 4, 3, 2):
                PP = parent_keys[Lp][2]
                ni = nearest_xz(Pc[miss][:, [0, 2]].astype(np.float64), PP[:, [0, 2]].astype(np.float64), PAGES[Lp]["cell"] * 4)
                ok_ = ni >= 0
                idx_ = np.flatnonzero(miss)[ok_]
                parent[idx_] = PP[ni[ok_]]
                found[idx_] = True
                miss = ~found
                if not miss.any():
                    break
        # the full-resolution layer: every pixel the photo has depth for (drawn for the current photo only). The holes
        # in the registration (no dot within 2.6 px, or a pixel smeared across a depth step) are filled from the far
        # side of their surroundings (behind, as a background is), so from the photo's eye the picture is whole
        dvalid = valid & ~take_sky
        hseed = np.zeros(lh.max() + 1, np.int64)
        sy, sx = np.nonzero(dvalid)
        hseed[lh[sy, sx]] = sy * W + sx
        hsrc = hseed[lh]
        zn = zd.ravel()[hsrc].reshape(Hh, W)
        zfar = np.exp(cv2.dilate(np.log(np.maximum(zn, 1e-3)).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))))
        hole = ~dvalid & (~sky | near_sky) & (dh < 90)
        zh = np.where(hole, zfar, zd)
        pdh = np.where(hole & ~near_sky, np.where(zfar < zn * 1.05, pd.ravel()[hsrc].reshape(Hh, W), 0), pd)
        pdh = np.where(near_sky, 0, pdh)
        skyness = np.where(near_sky, np.clip(dh / SKYFILL, 0, 1) ** 0.7, 0.0)   # (fading to the sky's own light away from the subject)
        if os.environ.get("MASKS"):
            (WORK / "masks").mkdir(exist_ok=True)
            vis_ = (img * 0.5 * 255).astype(np.uint8)
            vis_[dvalid, 1] = np.minimum(255, vis_[dvalid, 1].astype(int) + 90)
            vis_[sky, 2] = 255; vis_[hole, 0] = 255
            Image.fromarray(vis_).save(WORK / "masks" / f"{stem[-9:]}.jpg", quality=80)
        # (every pixel: thinning the flat parts, skin, sand, to save draw time left visible holes in the skin)
        hv, hu = np.nonzero(dvalid | hole)
        hz = zh[hv, hu]
        # (each pixel jittered within itself: a regular lattice of dots beats against the screen's pixels, a moire)
        huj = hu + RNG.uniform(-0.5, 0.5, len(hu)); hvj = hv + RNG.uniform(-0.5, 0.5, len(hv))
        Ph = np.stack([(huj + 0.5 - W / 2) / fx * hz, -(hvj + 0.5 - Hh / 2) / fy * hz, -hz], 1) @ R.T + C
        hd_d = np.linalg.norm(Ph - C, axis=1)
        far_ = hd_d > 500
        Ph[far_] = C + (Ph[far_] - C) * (500.0 / hd_d[far_])[:, None]
        hd_d = np.minimum(hd_d, 500.0)
        hk = hd_d < 900
        hv, hu, Ph, hd_d = hv[hk], hu[hk], Ph[hk], hd_d[hk]
        hr = RNG.random(len(Ph)); ho = np.argsort(hr)
        hpid = pdh[hv, hu].astype(np.int64)
        hn = len(Ph)
        HD.mkdir(exist_ok=True)
        with open(HD / f"{stem[-9:]}.bin", "wb") as fh:
            fh.write(Ph[ho].astype("<f4").tobytes())
            fh.write(np.ascontiguousarray(np.concatenate([u8(img[hv, hu][ho]), u8(sal[hv, hu][ho])[:, None]], 1)).tobytes())
            fh.write(np.ascontiguousarray(np.stack([(7 + 16 * (hpid[ho] + 8)).astype(np.uint8), u8(hr[ho]), u8(skyness[hv, hu][ho]),
                                                    size_code(hd_d[ho] / fx, PH_SIZE)], 1)).tobytes())
            fh.write(np.concatenate([np.zeros((hn, 3)), focus[hv, hu][ho][:, None]], 1).astype("<f2").tobytes())
        rnd = RNG.random(len(Pc))
        o_ = np.argsort(rnd)
        pid = pd[vv, uu].astype(np.int64)
        cnt = len(Pc)
        pos = Pc[o_].astype("<f4")
        colb = np.concatenate([u8(col[o_]), u8(sal[vv, uu][o_])[:, None]], 1)
        is_sky = take_sky[vv, uu]
        att = np.stack([(7 + 16 * pid[o_]).astype(np.uint8), u8(rnd[o_]), (is_sky[o_] * 255).astype(np.uint8), size_code(size[o_], PH_SIZE)], 1)   # cover = sky
        ph = np.concatenate([(parent - Pc)[o_], focus[vv, uu][o_][:, None]], 1).astype("<f2")
        chunks.append((pos, colb, att, ph))
        im = Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)); im.thumbnail((180, 180)); im.save(OUT / "thumbs" / f"{stem[-9:]}.jpg", quality=84)
        moment = MOMENTS[f["moment"]]
        cap = next((p["caption"] for p in META["photos"] if p["stem"] == stem), "")
        frames_meta.append(dict(stem=stem[-9:], time=f["time"], t=f["t"], moment=f["moment"], cam=dict(c), focus=f["focus"], heading=f["heading"],
                                caption=cap, dots=[total, cnt], hd=dict(file=f"hd/{stem[-9:]}.bin", count=int(hn)), crop=round(float(sc), 3),
                                people=sorted({int(x) for x in np.unique(pid) if x > 0})))
        total += cnt
        print(f"  {stem[-9:]}: crop {sc:.3f} (err {err:.1f}), {cnt} dots, hd {hn} (holes filled {hole.mean():.0%}, far {far_.mean():.0%}; {person.mean():.0%} person px), parents {found.mean():.0%}", flush=True)
    # write the four arrays (each the concatenation over frames)
    for j in range(4):
        for ch in chunks:
            out.write(np.ascontiguousarray(ch[j]).tobytes())
    out.close()
    os.replace(OUT / "photos.bin.tmp", OUT / "photos.bin")
    print(f"photos.bin: {total} dots, {os.path.getsize(OUT / 'photos.bin') / 1e6:.1f} MB ({time.time() - t0:.0f} s)", flush=True)
    return dict(file="photos.bin", count=int(total), sizeLog=list(PH_SIZE)), frames_meta


# ---------------------------------------------------------------------------------------------------------- main
def main():
    stages = sys.argv[1:] or ["mesh", "pages", "photos"]
    OUT.mkdir(exist_ok=True); WORK.mkdir(exist_ok=True)
    mp = OUT / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else {}
    if "mesh" in stages:
        mesh_samples()
    if "pages" in stages:
        meta["env"] = build_pages()
    if "photos" in stages:
        meta["photos"], meta["frames"] = bake_photos()
    meta.update(
        about=__doc__.strip().split("\n")[0], sizeLog=[SIZE_MIN, SIZE_MAX], origin=[LAT0, LON0],
        pages=[dict(name=p["name"], cell=p.get("cell"), R=p.get("R")) if p else None for p in PAGES],
        moments=[dict(id=m["id"], title=m["title"], line=m["line"], start=m["start"], centre=m["centre"], cam=[round(float(v), 2) for v in CAMS[m["id"]]],
                      t0=m["t0"], t1=m["t1"], people=m["people"]) for m in META["moments"]],
        t0=META["t0"], sun=META["sun"], sunset=META["sunset"], dusk=META["dusk"],
        people=[dict(id=p["id"], label=p["label"], colour=p["colour"]) for p in META["people"]],
        coast=META["coast"][::4], towers=META["towers"], walk=[[round(float(v), 2) for v in c] for c in CENTRES],
        layout=dict(bytes=28, arrays=["position float32x3", "colour uint8x4 (rgb, salience)", "att uint8x4 (page+16*link, rnd, cover, size)",
                                      "parent float16x4 (dx, dy, dz, focus)"]))
    mp.write_text(json.dumps(meta, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
