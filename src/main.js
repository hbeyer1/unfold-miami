// Unfold · Miami Beach: Unfold's engine (unfold/main.js) on the evening walk of 7 May 2026. The place is a pyramid of
// pages (Miami Beach, the beach, the walk, the scene, up close) that open out of each other as you come closer and fill
// in where you look; the 22 photos are each their own cloud of dots in the place, all of them always there but muted,
// the current one (in the order of time) condensing in its true colours. The camera drifts to frame the current photo
// in its surroundings; drag to look around it, E for the photo's own eye. Data: unfold-miami/data (bake.py).
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { ShaderPass } from "three/addons/postprocessing/ShaderPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { Pass, FullScreenQuad } from "three/addons/postprocessing/Pass.js";
import { SoftBloomPass } from "./bloom.js";
import { DOT_VS, DOT_FS, BRUSH_VS, BRUSH_FS, SKY_VS, SKY_FS, TRAIL, FINISH, GLOW_VS, GLOW_FS } from "./shaders.js";
import { dirFromAzEl, sunColour } from "./sky.js";

const DATA = "data/";
const dbg = { only: null, noTrail: false, noAuto: false };
const $ = (id) => document.getElementById(id);
const { clamp, lerp } = THREE.MathUtils;
const smooth = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const damp = (cur, target, rate, dt) => cur + (target - cur) * (1 - Math.exp(-rate * dt));
const V = () => new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);
const LN = Math.log;
const wrapA = (a) => Math.atan2(Math.sin(a), Math.cos(a));

const PAGE_NAMES = ["Miami Beach", "the beach", "the walk", "the scene", "up close", "underfoot", "the moment"];
// the distances (camera to what it looks at) over which each page opens
// (the finer pages open early: during fly-overs the beach is drawn from its finest structure the screen can hold)
const BANDS = [null, [7000, 1800], [3500, 900], [1100, 220], [200, 40], [18, 5]];
const STEP = 6.5;   // seconds per photo when playing

// ------------------------------------------------------------------------------------------------ renderer
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance", preserveDrawingBuffer: false });
const MAX_PR = Math.min(devicePixelRatio, 2);
let pixelRatio = Math.min(devicePixelRatio, 1.5);
renderer.setPixelRatio(pixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x000000, 1);
document.body.prepend(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, innerWidth / innerHeight, 0.05, 2e5);

class TrailPass extends Pass {
  constructor() {
    super();
    this.rts = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType }));
    this.mat = new THREE.ShaderMaterial(TRAIL);
    this.copy = new THREE.ShaderMaterial({ uniforms: { tDiffuse: { value: null } }, vertexShader: TRAIL.vertexShader,
      fragmentShader: "uniform sampler2D tDiffuse; varying vec2 vUv; void main() { gl_FragColor = texture2D(tDiffuse, vUv); }" });
    this.quad = new FullScreenQuad(this.mat);
    this.cur = 0; this.keep = 0;
  }
  setSize(w, h) { this.rts.forEach((r) => r.setSize(w, h)); }
  render(r, writeBuffer, readBuffer) {
    const a = this.rts[this.cur], b = this.rts[1 - this.cur];
    this.mat.uniforms.tDiffuse.value = readBuffer.texture; this.mat.uniforms.tPrev.value = b.texture; this.mat.uniforms.uKeep.value = this.keep;
    this.quad.material = this.mat; r.setRenderTarget(a); this.quad.render(r);
    this.copy.uniforms.tDiffuse.value = a.texture; this.quad.material = this.copy;
    r.setRenderTarget(this.renderToScreen ? null : writeBuffer); this.quad.render(r);
    this.cur = 1 - this.cur;
  }
}
const composer = new EffectComposer(renderer);
composer.setPixelRatio(pixelRatio);
composer.setSize(innerWidth, innerHeight);
composer.addPass(new RenderPass(scene, camera));
composer.addPass(new ShaderPass({
  uniforms: { tDiffuse: { value: null } },
  vertexShader: "varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
  fragmentShader: "uniform sampler2D tDiffuse; varying vec2 vUv; void main() { vec3 c = texture2D(tDiffuse, vUv).rgb; float m = max(max(c.r, c.g), c.b); if (!(m < 6e4)) c = vec3(0.0); gl_FragColor = vec4(c * min(1.0, 1.25 / max(m, 1e-4)), 1.0); }",
}));
const trail = new TrailPass();
composer.addPass(trail);
const bloom = new SoftBloomPass({ strength: 1.1, threshold: 0.9, knee: 0.25, levels: 6 });   // round, only the brightest
composer.addPass(bloom);
const finish = new ShaderPass(FINISH);
finish.uniforms.uVig.value = 0.45;
composer.addPass(finish);
composer.addPass(new OutputPass());
const additive = (extra) => new THREE.ShaderMaterial({ transparent: true, depthWrite: false, depthTest: false, blending: THREE.CustomBlending,
  blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, ...extra });

const skyU = { uSunDir: { value: V().set(0, 1, 0) }, uSunCol: { value: new THREE.Color(1, 0.8, 0.6) }, uDay: { value: 0.2 }, uTime: { value: 0 }, uStars: { value: 0 }, uGlow: { value: 0 }, uHaze: { value: 1 } };
const sky = new THREE.Mesh(new THREE.SphereGeometry(1, 48, 24), new THREE.ShaderMaterial({ uniforms: skyU, vertexShader: SKY_VS, fragmentShader: SKY_FS, side: THREE.BackSide, depthTest: false, depthWrite: false }));
sky.renderOrder = -10; sky.frustumCulled = false; scene.add(sky);

const S = {
  uTime: { value: 0 }, uPx: { value: 1 }, uDpr: { value: 1 }, uSpeed: { value: 0 }, uRemember: { value: 0 }, uSigma: { value: 4.8 },
  uSizeLog: { value: new THREE.Vector2() }, uMoon: { value: new THREE.Color(0.42, 0.48, 0.95) }, uLight: { value: new THREE.Color(1, 1, 1) },
};
function resize() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight); composer.setSize(innerWidth, innerHeight);
  finish.uniforms.uAspect.value = innerWidth / innerHeight;
}
addEventListener("resize", resize);
resize();

// ------------------------------------------------------------------------------------------------ attention
const TILE = 512, COLS = 5, ROWS = 4;
class Attention {
  constructor() {
    this.rt = new THREE.WebGLRenderTarget(TILE * COLS, TILE * ROWS, { type: THREE.HalfFloatType, depthBuffer: false, minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, generateMipmaps: false });
    this.cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.decayMat = new THREE.ShaderMaterial({
      uniforms: { uK: { value: new THREE.Vector2(1, 1) } },
      vertexShader: "void main() { gl_Position = vec4(position.xy, 0.0, 1.0); }",
      fragmentShader: "uniform vec2 uK; void main() { gl_FragColor = vec4(uK, 1.0, 1.0); }",
      blending: THREE.CustomBlending, blendEquation: THREE.AddEquation, blendSrc: THREE.ZeroFactor, blendDst: THREE.SrcColorFactor,
      blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor, depthTest: false, depthWrite: false });
    this.decayQuad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.decayMat); this.decayQuad.frustumCulled = false;
    const N = 48, g = new THREE.InstancedBufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0], 3));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    this.iTile = new THREE.InstancedBufferAttribute(new Float32Array(N * 4), 4);
    this.iBrush = new THREE.InstancedBufferAttribute(new Float32Array(N * 4), 4);
    this.iAmt = new THREE.InstancedBufferAttribute(new Float32Array(N * 2), 2);
    for (const a of [this.iTile, this.iBrush, this.iAmt]) a.setUsage(THREE.DynamicDrawUsage);
    g.setAttribute("iTile", this.iTile); g.setAttribute("iBrush", this.iBrush); g.setAttribute("iAmt", this.iAmt);
    this.N = N;
    this.brushMesh = new THREE.Mesh(g, new THREE.ShaderMaterial({ vertexShader: BRUSH_VS, fragmentShader: BRUSH_FS, blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor, blendDst: THREE.OneFactor, blendSrcAlpha: THREE.ZeroFactor, blendDstAlpha: THREE.OneFactor, depthTest: false, depthWrite: false }));
    this.brushMesh.frustumCulled = false;
    this.tiles = []; this.cleared = false;
  }
  addTile(cx, cz, half) { const t = { index: this.tiles.length, cx, cz, half }; this.tiles.push(t); return t; }
  vec(t, out = new THREE.Vector4()) { return out.set(t.cx, t.cz, t.half, t.index); }
  update(dt, brushes) {
    const r = renderer, prev = r.getRenderTarget(), auto = r.autoClear;
    r.setRenderTarget(this.rt);
    if (!this.cleared) { r.setClearColor(0x000000, 0); r.clear(true, false, false); r.setClearColor(0x000000, 1); this.cleared = true; }
    r.autoClear = false;
    this.decayMat.uniforms.uK.value.set(Math.exp(-dt / 5.0), Math.exp(-dt / 240.0));
    r.render(this.decayQuad, this.cam);
    let n = 0;
    for (const b of brushes) for (const t of this.tiles) {
      if (n >= this.N) break;
      const dx = b.x - t.cx, dz = b.z - t.cz;
      if (Math.abs(dx) > t.half + b.r || Math.abs(dz) > t.half + b.r) continue;
      const ext = 2 * t.half, texel = ext / TILE;
      if (b.r > ext * 0.6) continue;
      let rr = b.r, k = 1;
      if (rr < texel * 1.2) { k = (rr / (texel * 1.2)) ** 2; rr = texel * 1.2; }
      const col = t.index % COLS, row = Math.floor(t.index / COLS);
      this.iTile.setXYZW(n, col / COLS, row / ROWS, 1 / COLS, 1 / ROWS);
      this.iBrush.setXYZW(n, dx / ext + 0.5, dz / ext + 0.5, rr / ext, 0);
      this.iAmt.setXY(n, b.now * k, b.last * k);
      n++;
    }
    if (n) { this.brushMesh.geometry.instanceCount = n; for (const a of [this.iTile, this.iBrush, this.iAmt]) a.needsUpdate = true; r.render(this.brushMesh, this.cam); }
    r.autoClear = auto; r.setRenderTarget(prev);
  }
}
const attention = new Attention();

// ------------------------------------------------------------------------------------------------ loading
const loadEl = $("loading"), barEl = $("bar");
const meta = await fetch(DATA + "meta.json").then((r) => r.json());
S.uSizeLog.value.set(meta.sizeLog[0], meta.sizeLog[1]);
const totalBytes = (meta.env.count + meta.photos.count) * 28;
let loadedBytes = 0;
async function fetchBin(file, count) {
  const res = await fetch(DATA + file);
  const buf = new Uint8Array(count * 28);
  const reader = res.body.getReader();
  for (let off = 0; ;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf.set(value, off); off += value.length; loadedBytes += value.length;
    barEl.style.width = `${(100 * loadedBytes) / totalBytes}%`;
  }
  return { pos: new Float32Array(buf.buffer, 0, count * 3), col: new Uint8Array(buf.buffer, count * 12, count * 4),
    att: new Uint8Array(buf.buffer, count * 16, count * 4), par: new Uint16Array(buf.buffer, count * 20, count * 4) };
}
function rangeGeometry(d, off, n) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(d.pos.subarray(off * 3, (off + n) * 3), 3));
  g.setAttribute("aCol", new THREE.BufferAttribute(d.col.subarray(off * 4, (off + n) * 4), 4, true));
  g.setAttribute("aAtt", new THREE.BufferAttribute(d.att.subarray(off * 4, (off + n) * 4), 4, true));
  const par = new THREE.BufferAttribute(d.par.subarray(off * 4, (off + n) * 4), 4);
  par.isFloat16BufferAttribute = true;
  g.setAttribute("aPar", par);
  g.boundingSphere = new THREE.Sphere(V(), 1e8);
  return g;
}

// attention tiles: the island, the beach along the walk, and near and eye tiles at each moment
const moments = meta.moments.map((m, i) => ({ ...m, index: i, C: V().fromArray(m.centre), camC: V().fromArray(m.cam) }));
const walkC = moments.reduce((a, m) => a.add(m.C), V()).multiplyScalar(1 / moments.length);
const tS = attention.addTile(90, 17, 2300), tF = attention.addTile(walkC.x, walkC.z, 900);
const tN = moments.map((m) => attention.addTile(m.C.x, m.C.z, 160));
const tE = moments.map((m) => attention.addTile(m.camC.x, m.camC.z, 26));
function dotUniforms() {
  return {
    ...S, uAtt: { value: attention.rt.texture },
    uTileS: { value: attention.vec(tS) }, uTileF: { value: attention.vec(tF) },
    uTileN: { value: tN.map((t) => attention.vec(t)) }, uTileE: { value: tE.map((t) => attention.vec(t)) },
    uUnfold: { value: new Array(8).fill(1) }, uPreview: { value: new Array(8).fill(0) }, uCell: { value: [12, 3, 0.75, 0.25, 0.045, 0.015, 0.015, 0.02] }, uWalk: { value: [0, 1, 2].map(() => V()) },
    uCam: { value: V() }, uRayO: { value: V() }, uRayD: { value: V().set(0, -1, 0) }, uRayAmt: { value: 0 },
    uPhotoA: { value: new THREE.Matrix4() }, uTanA: { value: new THREE.Vector2(1, 1) }, uPhotoC: { value: new THREE.Matrix4() }, uTanC: { value: new THREE.Vector2(1, 1) }, uHideA: { value: 0 }, uAtEye: { value: 0 },
    uFocusD: { value: 1e4 }, uAperture: { value: 0 }, uNear: { value: 1e5 }, uFade: { value: 1 },
    uCur: { value: 0 }, uW: { value: 0 }, uHDw: { value: 0 }, uFly: { value: 1 }, uMoment: { value: new THREE.Vector4(0, 0, 0, -1) }, uSettle: { value: 1 }, uWarm: { value: 1 }, uWarmGain: { value: 1 }, uDusk: { value: 0 },
  };
}

// ------------------------------------------------------------------------------------------------ the place
const envData = await fetchBin(meta.env.file, meta.env.count);
const envU = dotUniforms();
moments.forEach((m, i) => envU.uWalk.value[i].copy(m.C));
const envMat = additive({ uniforms: envU, vertexShader: DOT_VS, fragmentShader: DOT_FS });
// the place in tiles: each a separate draw, culled by the view, and drawn only as far into its rnd-sorted dots as the
// screen can hold from here (the rest would never show: see the level-of-detail test in the dot shader)
const PARENT_CELL = [0, 12, 3, 0.75, 0.25, 0.045];
const envPages = meta.env.tiles.map(([L, off, n, cx, cy, cz, rad, smax]) => {
  const g = rangeGeometry(envData, off, n);
  const margin = 3 * PARENT_CELL[L] + 2;
  g.boundingSphere = new THREE.Sphere(V().set(cx, cy, cz), rad + margin);
  const pts = new THREE.Points(g, envMat);
  pts.frustumCulled = L >= 2; pts.page = L; pts.n = n; pts.smax = smax; pts.C = V().set(cx, cy, cz); pts.rad = rad + margin;
  // (does the tile hold shoreline or sea dots, which the dot shader keeps drawn further out? focus != 0 in f16 bits)
  pts.coast = false;
  for (let k = off; k < off + n; k++) { if ((envData.par[4 * k + 3] & 0x7fff) !== 0) { pts.coast = true; break; } }
  scene.add(pts);
  return pts;
});
// a coarse height grid of the place (for where the cursor's ray meets it): the highest dot in each 4 m cell
const HG = (() => {
  const cell = 4, x0 = -2000, z0 = -2050, n = 1050, h = new Float32Array(n * n).fill(-1e4), lo = new Float32Array(n * n).fill(1e4);
  for (const L of ["0", "1", "2"]) {
    const [off, cnt] = meta.env.pages[L];
    for (let i = off; i < off + cnt; i++) {
      const ix = Math.floor((envData.pos[3 * i] - x0) / cell), iz = Math.floor((envData.pos[3 * i + 2] - z0) / cell);
      if (ix < 0 || iz < 0 || ix >= n || iz >= n) continue;
      const k = iz * n + ix, y = envData.pos[3 * i + 1];
      if (y > h[k]) h[k] = y;
      if (y < lo[k]) lo[k] = y;
    }
  }
  const get = (a, x, z, dflt) => { const ix = Math.floor((x - x0) / cell), iz = Math.floor((z - z0) / cell); if (ix < 0 || iz < 0 || ix >= n || iz >= n) return 0; const y = a[iz * n + ix]; return Math.abs(y) < 1e3 ? y : dflt; };
  // at: the top of what stands there (for the cursor's ray); ground: the lowest (for keeping the camera above the sand)
  return { cell, x0, z0, n, h, at: (x, z) => get(h, x, z, 0), ground: (x, z) => get(lo, x, z, 0) };
})();

// ------------------------------------------------------------------------------------------------ the photos
const photoData = await fetchBin(meta.photos.file, meta.photos.count);
const frames = meta.frames.map((f, i) => {
  const c = f.cam;
  const pos = V().fromArray(c.pos), quat = new THREE.Quaternion().fromArray(c.quat);
  const fwd = V().set(0, 0, -1).applyQuaternion(quat);
  const u = dotUniforms();
  u.uPhotoA.value.compose(pos, quat, V().set(1, 1, 1)).invert();
  const tanY = Math.tan((c.fov * Math.PI) / 360);
  u.uTanA.value.set(tanY * c.aspect, tanY);
  u.uWarmGain.value = 1.0;
  u.uSizeLog = { value: new THREE.Vector2().fromArray(meta.photos.sizeLog) };   // (the photos' own, finer size range)
  const mat = additive({ uniforms: u, vertexShader: DOT_VS, fragmentShader: DOT_FS, defines: { IS_PHOTO: 1 } });
  const [off, n] = f.dots;
  const pts = new THREE.Points(rangeGeometry(photoData, off, n), mat);
  pts.frustumCulled = false;
  scene.add(pts);
  const mom = moments.find((m) => m.id === f.moment);
  // the people's dots (for picking what the cursor rests on near the photo, and to aim the eye)
  const pick = [];
  for (let k = off; k < off + n; k += 9) pick.push(photoData.pos[3 * k], photoData.pos[3 * k + 1], photoData.pos[3 * k + 2]);
  return { ...f, i, pos, quat, fwd, u, mat, pts, n, mom, focusV: V().fromArray(f.focus), cur: 0, w: 0, settle: 1, pick: Float32Array.from(pick),
    selfie: f.stem.startsWith("201"), tanY };
});

// the photos at full resolution: one file each, fetched for the current photo and its neighbours, uploaded ahead
const hdCache = new Map();
async function loadHD(f) {
  if (f.hdPts || f.hdLoading || !f.hd) return;
  f.hdLoading = true;
  const res = await fetch(DATA + f.hd.file);
  const buf = new Uint8Array(await res.arrayBuffer());
  const n = f.hd.count;
  const d = { pos: new Float32Array(buf.buffer, 0, n * 3), col: new Uint8Array(buf.buffer, n * 12, n * 4), att: new Uint8Array(buf.buffer, n * 16, n * 4), par: new Uint16Array(buf.buffer, n * 20, n * 4) };
  const pts = new THREE.Points(rangeGeometry(d, 0, n), f.mat);
  pts.frustumCulled = false; pts.visible = true; pts.uploaded = false;
  scene.add(pts);
  f.hdPts = pts; f.hdLoading = false;
  hdCache.set(f.i, now);
}
function manageHD() {
  const want = new Set([cur, (cur + 1) % frames.length, (cur + frames.length - 1) % frames.length]);
  for (const k of want) { loadHD(frames[k]); hdCache.set(k, now); }
  for (const f of frames) {
    if (f.hdPts && !f.hdPts.uploaded) { f.hdPts.visible = true; f.hdPts.uploadFrame = (f.hdPts.uploadFrame || 0) + 1; if (f.hdPts.uploadFrame > 1) f.hdPts.uploaded = true; }
    if (f.hdPts && !want.has(f.i) && f.hdw < 0.003 && hdCache.size > 4) {   // drop the ones far behind
      scene.remove(f.hdPts); f.hdPts.geometry.dispose(); f.hdPts = null; hdCache.delete(f.i);
    }
  }
}

// the photos' glow: a few soft warm sprites per photo, over its people (or its focus), tinted by the picture
const GLOW_K = 4;
const glowAmt = new THREE.BufferAttribute(new Float32Array(frames.length * GLOW_K), 1).setUsage(THREE.DynamicDrawUsage);
const glow = (() => {
  const pos = [], col = [], rad = [];
  for (const f of frames) {
    const [off, n] = f.dots;
    const ppl = [];
    let cr = 0, cg = 0, cb = 0, cn = 0;
    for (let k = off; k < off + n; k += 7) {
      const x = photoData.pos[3 * k], y = photoData.pos[3 * k + 1], z = photoData.pos[3 * k + 2];
      if (Math.hypot(x - f.focusV.x, z - f.focusV.z) > 6) continue;
      if ((photoData.att[4 * k] >> 4) > 0) ppl.push([x, y, z]);
      cr += photoData.col[4 * k]; cg += photoData.col[4 * k + 1]; cb += photoData.col[4 * k + 2]; cn++;
    }
    const avg = new THREE.Color(cr / Math.max(cn, 1) / 255, cg / Math.max(cn, 1) / 255, cb / Math.max(cn, 1) / 255);
    const tint = new THREE.Color(1.0, 0.74, 0.46).lerp(avg, 0.25);
    for (let j = 0; j < GLOW_K; j++) {
      const p = ppl.length ? ppl[Math.floor(((j + 0.5) / GLOW_K) * ppl.length * 0.999 + (j * 7919) % Math.max(1, Math.floor(ppl.length / GLOW_K))) % ppl.length]
        : [f.focusV.x + (j - 1.5) * 0.6, f.focusV.y + 0.8, f.focusV.z];
      pos.push(p[0], p[1] + 0.2, p[2]);
      col.push(tint.r, tint.g, tint.b);
      rad.push(f.selfie ? 1.6 + 0.4 * j : 2.2 + 0.5 * j);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(pos), 3));
  g.setAttribute("aColor", new THREE.BufferAttribute(new Float32Array(col), 3));
  g.setAttribute("aR", new THREE.BufferAttribute(new Float32Array(rad), 1));
  g.setAttribute("aAmt", glowAmt);
  g.boundingSphere = new THREE.Sphere(V(), 1e8);
  const pts = new THREE.Points(g, additive({ uniforms: { uPx: S.uPx, uDpr: S.uDpr, uTime: S.uTime }, vertexShader: GLOW_VS, fragmentShader: GLOW_FS }));
  pts.frustumCulled = false;
  pts.renderOrder = 2;
  scene.add(pts);
  return pts;
})();

// ------------------------------------------------------------------------------------------------ navigation
// An orbit around what you look at (a target T, a distance r, a heading and a pitch), every part eased. Left alone it
// drifts to frame the current photo in its surroundings: a few metres behind and above its eye, swaying slowly.
const cam = { T: V(), r: 8, yaw: 0, pitch: -0.2, fov: 55 };     // current (eased)
const want = { T: V(), r: 8, yaw: 0, pitch: -0.2, fov: 55 };    // desired
let mode = "drift", lastInput = -1e9, started = false, eyeBlend = 0, flight = null, flySm = 1, flightSm = 0;
const dirOf = (yaw, pitch, out) => out.set(Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch), -Math.cos(yaw) * Math.cos(pitch));
function framePose(f, out, t) {
  // behind and above the photo's eye, looking at what the photo looks at
  const F = f.focusV, C = f.pos;
  const fh = V().set(f.fwd.x, 0, f.fwd.z).normalize();
  const back = f.selfie ? 1.25 : 3.0, lift = f.selfie ? 0.35 : 1.0;
  const sway = 0.2 * Math.sin(t * 0.045) + 0.06 * Math.sin(t * 0.11 + 1.3);
  const P = C.clone().addScaledVector(fh, -back).addScaledVector(UP, lift);
  const d = P.clone().sub(F);
  d.applyAxisAngle(UP, sway);
  const target = f.selfie ? C.clone().lerp(F, 0.9) : F.clone();
  const Pp = target.clone().add(d);
  const v = target.clone().sub(Pp);
  out.T.copy(target); out.r = v.length(); out.yaw = Math.atan2(v.x, -v.z); out.pitch = Math.asin(clamp(v.y / out.r, -1, 1)); out.fov = f.selfie ? 50 : 52;
  return out;
}
function eyePose(f, out) {
  const dist = Math.max(2.5, f.focusV.distanceTo(f.pos));
  out.T.copy(f.pos).addScaledVector(f.fwd, dist); out.r = dist;
  out.yaw = Math.atan2(f.fwd.x, -f.fwd.z); out.pitch = Math.asin(clamp(f.fwd.y, -1, 1)); out.fov = clamp(f.cam.fov, 35, 78);
  return out;
}
function overviewPose(out) {
  out.T.copy(walkC).setY(0); out.T.x -= 120; out.r = 1500; out.yaw = -0.12; out.pitch = -0.88; out.fov = 52;
  return out;
}

// ------------------------------------------------------------------------------------------------ time
let cur = 0, playing = true, lastStep = 0, changedAt = 0;
function setFrame(k, user = false) {
  k = (k + frames.length) % frames.length;
  if (k === cur && started) return;
  const prevMoment = frames[cur].moment;
  cur = k; changedAt = now; lastStep = now;
  frames[cur].settle = 0;
  if (user) lastInput = now;
  if (mode === "eye") eyePose(frames[cur], want);
  if (frames[cur].moment !== prevMoment || !started) showMoment(frames[cur].moment);
  showCaption();
  thumbs.forEach((b, i) => b.classList.toggle("on", i === cur));
}

// ------------------------------------------------------------------------------------------------ input
const pointer = { ndc: new THREE.Vector2(), over: false, down: false, moved: -1e9, lx: 0, ly: 0, speed: 0, dragged: false, button: 0, shift: false };
const el = renderer.domElement;
let wheelAcc = 0, wheelAt = 0, remember = 0, rememberHeld = false;
el.addEventListener("contextmenu", (e) => e.preventDefault());
el.addEventListener("pointerdown", (e) => {
  begin();
  pointer.down = true; pointer.dragged = false; pointer.lx = e.clientX; pointer.ly = e.clientY; pointer.button = e.button; pointer.shift = e.shiftKey;
  lastInput = now;
  try { el.setPointerCapture(e.pointerId); } catch {}
});
el.addEventListener("pointerup", () => { pointer.down = false; });
addEventListener("pointermove", (e) => {
  const nx = (e.clientX / innerWidth) * 2 - 1, ny = -(e.clientY / innerHeight) * 2 + 1;
  const dtm = Math.max(1 / 240, now - pointer.moved);
  pointer.speed = lerp(pointer.speed, Math.hypot(nx - pointer.ndc.x, ny - pointer.ndc.y) / 2 / dtm, 0.3);
  pointer.ndc.set(nx, ny); pointer.moved = now; pointer.over = e.target === el;
  if (!pointer.down) return;
  const dx = e.clientX - pointer.lx, dy = e.clientY - pointer.ly;
  if (Math.abs(dx) + Math.abs(dy) > 2) pointer.dragged = true;
  lastInput = now;
  if (mode !== "free") { want.T.copy(cam.T); want.r = cam.r; want.yaw = cam.yaw; want.pitch = cam.pitch; want.fov = cam.fov; mode = "free"; }
  if (pointer.button === 2 || pointer.shift) {   // move what you look at, over the ground
    const k = cam.r * 0.0016;
    const right = V().set(Math.cos(cam.yaw), 0, Math.sin(cam.yaw)), ahead = V().set(Math.sin(cam.yaw), 0, -Math.cos(cam.yaw));
    want.T.addScaledVector(right, -dx * k).addScaledVector(ahead, dy * k);
  } else {
    want.yaw -= dx * 0.005;
    want.pitch = clamp(want.pitch - dy * 0.004, -1.45, 0.45);
  }
  pointer.lx = e.clientX; pointer.ly = e.clientY;
});
el.addEventListener("pointerleave", () => { pointer.over = false; });
function zoomBy(dz) {
  begin();
  lastInput = now;
  if (mode !== "free") { want.T.copy(cam.T); want.r = cam.r; want.yaw = cam.yaw; want.pitch = cam.pitch; want.fov = cam.fov; mode = "free"; }
  want.r = clamp(want.r * Math.exp(dz), 0.6, 6000);
  if (want.r > 400) want.pitch = Math.min(want.pitch, lerp(-0.3, -0.95, smooth(LN(400), LN(3000), LN(want.r))));
}
el.addEventListener("wheel", (e) => {
  e.preventDefault();
  begin();
  const unit = e.deltaMode === 1 ? 33 : e.deltaMode === 2 ? 400 : 1;
  const d = e.deltaY * unit;
  if (e.ctrlKey || e.shiftKey || e.altKey) { zoomBy(clamp(d, -200, 200) * (e.ctrlKey ? 0.01 : 0.0025)); return; }   // pinch / shift: closer, further
  // the wheel moves through time: one photo for each notch (or ~100 px of a trackpad's swipe)
  if (now - wheelAt > 0.6) wheelAcc = 0;
  wheelAt = now; wheelAcc += d;
  if (Math.abs(wheelAcc) >= 90) { setFrame(cur + Math.sign(wheelAcc), true); wheelAcc = 0; playing = false; updatePlay(); }
}, { passive: false });
addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey) return;
  begin();
  lastInput = now;
  const n = Number(e.key);
  if (n >= 1 && n <= moments.length) { setFrame(frames.findIndex((f) => f.moment === moments[n - 1].id), true); return; }
  switch (e.code) {
    case "ArrowRight": setFrame(cur + 1, true); playing = false; updatePlay(); break;
    case "ArrowLeft": setFrame(cur - 1, true); playing = false; updatePlay(); break;
    case "ArrowUp": zoomBy(-0.35); break;
    case "ArrowDown": zoomBy(0.35); break;
    case "KeyE": mode = mode === "eye" ? "drift" : "eye"; if (mode === "eye") eyePose(frames[cur], want); break;
    case "KeyF": case "Escape": mode = "drift"; break;
    case "KeyO": mode = "overview"; overviewPose(want); break;
    case "KeyP": case "Space": e.preventDefault(); playing = !playing; lastStep = now; updatePlay(); break;
    case "KeyR": rememberHeld = true; break;
  }
});
addEventListener("keyup", (e) => { if (e.code === "KeyR") rememberHeld = false; });
addEventListener("blur", () => { rememberHeld = false; pointer.down = false; });

// ------------------------------------------------------------------------------------------------ overlay
const capEl = $("caption"), whisperEl = $("whisper"), clockEl = $("clock"), placeEl = $("place"), playEl = $("play");
let capTimers = [];
function showMoment(id) {
  const m = moments.find((x) => x.id === id);
  capTimers.forEach(clearTimeout); capTimers = [];
  capEl.innerHTML = `<h1>${m.title}</h1><p>${m.start} · ${m.line.split(";")[0].split(":")[0]}</p>`;
  capEl.classList.add("show");
  capTimers.push(setTimeout(() => capEl.classList.remove("show"), 6000));
}
function showCaption() {
  const f = frames[cur];
  whisperEl.textContent = f.caption;
  whisperEl.classList.add("show");
}
function updatePlay() { playEl.textContent = playing ? "❚❚" : "▶"; playEl.title = playing ? "pause (P)" : "play (P)"; }
playEl.addEventListener("click", (e) => { e.stopPropagation(); playing = !playing; lastStep = now; updatePlay(); });
const strip = $("strip");
const thumbs = frames.map((f, i) => {
  if (i > 0 && frames[i - 1].moment !== f.moment) { const gap = document.createElement("span"); gap.className = "gap"; strip.appendChild(gap); }
  const b = document.createElement("button");
  b.innerHTML = `<img src="${DATA}thumbs/${f.stem}.jpg" alt=""><small>${f.time.slice(0, 5)}</small>`;
  b.title = `${f.time} · ${f.caption}`;
  b.addEventListener("click", (e) => { e.stopPropagation(); begin(); setFrame(i, true); playing = false; updatePlay(); if (mode === "free" || mode === "overview") mode = "drift"; });
  strip.appendChild(b);
  return b;
});
const labelEls = moments.map((m) => {
  const b = document.createElement("button");
  b.className = "label";
  b.innerHTML = `${m.title}<small>${m.start}</small>`;
  b.addEventListener("click", (e) => { e.stopPropagation(); setFrame(frames.findIndex((f) => f.moment === m.id), true); mode = "drift"; });
  $("labels").appendChild(b);
  return b;
});
const gauge = $("gauge");
const gaugeEls = PAGE_NAMES.map((name, i) => { const d = document.createElement("div"); d.innerHTML = `<i></i><span>${name}</span>`; d.style.top = `${(i / 6) * 100}%`; gauge.appendChild(d); return d; });
const gaugeHead = document.createElement("b"); gauge.appendChild(gaugeHead);

function begin() {
  if (started) return;
  started = true;
  $("intro").classList.add("gone");
}

// ------------------------------------------------------------------------------------------------ the frame
let now = 0, speedSm = 0, lastMotion = 0, focusSm = 10, gazeDist = null, gazeHit = null, lastPickT = -1;
const prevCam = { pos: V(), yaw: 0, pitch: 0 };
const sunCol = new THREE.Color(), moonBase = new THREE.Color(0.42, 0.48, 0.95), tmpC = new THREE.Color(), tmpV = V(), tmpV2 = V(), buf2 = new THREE.Vector2();
const raycaster = new THREE.Raycaster();
const T0 = Date.parse(meta.t0);

function sunAt(t) {
  const s = meta.sun, i = clamp(Math.floor((t - s[0][0]) / 10), 0, s.length - 2), f = clamp((t - s[i][0]) / 10, 0, 1);
  return { az: lerp(s[i][1], s[i + 1][1], f), el: lerp(s[i][2], s[i + 1][2], f) };
}

function placeCamera(dt) {
  const f = frames[cur];
  const idle = now - lastInput;
  if (mode === "free" && idle > 12 && !dbg.noAuto) mode = "drift";
  if (mode === "drift") framePose(f, want, now);
  else if (mode === "eye") eyePose(f, want);
  // a flight: between moments (or for a big turn) the camera rises into a wider view along an arc and glides down
  // onto the next photo; the yaw turns while it is up there, slowly (no swing, no swirl at sand level)
  if (mode !== "free" && !flight) {
    const dist = cam.T.distanceTo(want.T), turn = Math.abs(wrapA(want.yaw - cam.yaw));
    if (dist > 25 || (turn > 1.2 && mode !== "eye")) {
      const lift = Math.max(0.42 * dist, turn > 1.2 ? 9 + 4 * turn : 0);
      flight = { T: cam.T.clone(), r: cam.r, yaw: cam.yaw, pitch: cam.pitch, fov: cam.fov, t0: now, dist, lift,
        dur: clamp(2.6 + 1.1 * LN(1 + dist / 30) + 0.5 * turn, 2.8, 8) };
    }
  }
  if (flight && mode !== "free") {
    const u = clamp((now - flight.t0) / flight.dur, 0, 1), e = u * u * (3 - 2 * u), arc = Math.sin(Math.PI * u);
    cam.T.copy(flight.T).lerp(want.T, e);
    cam.r = Math.exp(lerp(LN(flight.r), LN(want.r), e)) + flight.lift * Math.pow(arc, 0.8);
    cam.yaw = flight.yaw + wrapA(want.yaw - flight.yaw) * smooth(0.15, 0.85, u);
    cam.pitch = lerp(lerp(flight.pitch, want.pitch, e), -0.72, Math.pow(arc, 0.7) * smooth(6, 40, flight.lift));
    cam.fov = lerp(flight.fov, want.fov, e);
    if (u >= 1) flight = null;
  } else {
    flight = null;
    const rate = mode === "free" ? 6 : 1.4;
    cam.T.x = damp(cam.T.x, want.T.x, rate, dt); cam.T.y = damp(cam.T.y, want.T.y, rate, dt); cam.T.z = damp(cam.T.z, want.T.z, rate, dt);
    cam.r = Math.exp(damp(LN(cam.r), LN(want.r), rate, dt));
    cam.yaw += wrapA(want.yaw - cam.yaw) * (1 - Math.exp(-(mode === "free" ? 8 : 1.4) * dt));
    cam.pitch = damp(cam.pitch, want.pitch, mode === "free" ? 8 : 1.4, dt);
    cam.fov = damp(cam.fov, want.fov, 2, dt);
  }
  const d = dirOf(cam.yaw, cam.pitch, tmpV);
  camera.position.copy(cam.T).addScaledVector(d, -cam.r);
  // never under the ground
  const g = HG.ground(camera.position.x, camera.position.z);
  if (mode !== "eye" && camera.position.y < g + 0.4) camera.position.y = g + 0.4;
  camera.lookAt(cam.T);
  // at the photo's eye: its own orientation, roll and all (a tilted selfie stays tilted, its frame fills the view)
  eyeBlend = damp(eyeBlend, mode === "eye" ? 1 : 0, 2.5, dt);
  const near = eyeBlend * (1 - smooth(0.15, 2.0, camera.position.distanceTo(f.pos)));
  if (near > 0.001) camera.quaternion.slerp(f.quat, near);
  // the cursor leans the view a little
  if (!pointer.down && pointer.over) {
    const right = V().set(1, 0, 0).applyQuaternion(camera.quaternion);
    camera.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(UP, -pointer.ndc.x * 0.03)).premultiply(new THREE.Quaternion().setFromAxisAngle(right, pointer.ndc.y * 0.02));
  }
  camera.fov = cam.fov;
  camera.near = clamp(cam.r / 400, 0.02, 20);
  camera.far = 2e5;
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld();
  S.uDpr.value = pixelRatio / 1.25;
  S.uPx.value = renderer.getDrawingBufferSize(buf2).y / S.uDpr.value / (2 * Math.tan((camera.fov * Math.PI) / 360));
  return cam.r;
}

function groundAt(ndc, out) {
  raycaster.setFromCamera(ndc, camera);
  const o = raycaster.ray.origin, d = raycaster.ray.direction;
  // near the current photo: its own dots
  const f = frames[cur];
  if (o.distanceTo(f.pos) < 60) {
    let best = Infinity;
    const P = f.pick;
    for (let i = 0; i < P.length; i += 3) {
      const x = P[i] - o.x, y = P[i + 1] - o.y, z = P[i + 2] - o.z;
      const al = x * d.x + y * d.y + z * d.z;
      if (al < 0.3 || al > best) continue;
      const pp = x * x + y * y + z * z - al * al;
      if (pp < (0.015 * al) * (0.015 * al)) best = al;
    }
    if (best < Infinity) return out.copy(o).addScaledVector(d, best);
  }
  let t = 0.5, prev = 0;
  for (let i = 0; i < 200 && t < 20000; i++) {
    const x = o.x + d.x * t, y = o.y + d.y * t, z = o.z + d.z * t;
    if (y <= HG.at(x, z)) return out.set(o.x + d.x * 0.5 * (t + prev), o.y + d.y * 0.5 * (t + prev), o.z + d.z * 0.5 * (t + prev));
    prev = t; t = t * 1.07 + 0.3;
  }
  return null;
}

function update(dt, d) {
  // pages: open by the distance to what you look at
  // (the two finest, for the sand at eye level, stay folded while you fly between moments: from the arc their dense
  // patch around the cameras read as a pale slab; they open as you arrive)
  flightSm = damp(flightSm, flight ? 1 : 0, flight ? 4 : 1.5, dt);
  for (let L = 1; L <= 5; L++) {
    const [hi, lo] = BANDS[L];
    envU.uUnfold.value[L] = smooth(LN(hi), LN(lo), LN(Math.max(d, 1))) * (L >= 4 ? 1 - flightSm : 1);
    envU.uPreview.value[L] = 0;   // (no attention-driven unfolding: the place reads on its own, by distance alone)
  }
  envU.uUnfold.value[0] = 1;
  const sMin = 0.9;   // (the smallest spacing the dot shader's rules can ask for, near the current photo)
  envPages.forEach((p) => {
    p.visible = (dbg.only == null || dbg.only === p.page) && envU.uUnfold.value[p.page] + envU.uPreview.value[p.page] > 0.002;
    if (!p.visible || p.page < 2) { p.geometry.setDrawRange(0, p.n); return; }
    const dmin = Math.max(0.3, camera.position.distanceTo(p.C) - p.rad);
    const k0 = (p.smax * S.uPx.value / dmin / sMin) ** 2;
    const wantMax = Math.max(Math.min(1, 3.3 * k0 + 0.4 * Math.max(k0 - 1, 0)), p.coast ? 0.75 * Math.min(1, 30 * k0) : 0, S.uRemember.value);
    p.geometry.setDrawRange(0, Math.min(p.n, Math.ceil(p.n * Math.min(1, wantMax + 0.04))));
  });
  // motion
  const dp = camera.position.distanceTo(prevCam.pos) / Math.max(d, 1) / Math.max(dt, 1e-3);
  const dr = Math.abs(wrapA(cam.yaw - prevCam.yaw)) / Math.max(dt, 1e-3) + Math.abs(cam.pitch - prevCam.pitch) / Math.max(dt, 1e-3);
  prevCam.pos.copy(camera.position); prevCam.yaw = cam.yaw; prevCam.pitch = cam.pitch;
  const sp = Math.min(4, dp * 0.8 + dr * 0.12);
  speedSm = damp(speedSm, sp, sp > speedSm ? 6 : 2.5, dt);
  if (sp > 0.05) lastMotion = now;
  S.uSpeed.value = clamp(speedSm * 0.6, 0, 1.4);
  trail.keep = dbg.noTrail ? 0 : (0.12 + 0.6 * smooth(0.6, 1.8, speedSm)) * smooth(0.05, 0.3, speedSm);
  S.uSigma.value = lerp(3.6, 2.9, smooth(LN(600), LN(3000), LN(Math.max(d, 1))));   // a little denser: the beach legible
  remember = damp(remember, rememberHeld ? 1 : 0, rememberHeld ? 2 : 1.2, dt);
  S.uRemember.value = remember;
  raycaster.setFromCamera(pointer.ndc, camera);
  const ro = raycaster.ray.origin, rd = raycaster.ray.direction;
  const rayAmt = pointer.over && started ? clamp(1.3 - (now - pointer.moved) * 0.25, 0.25, 1) : 0;
  // the photos: the current one condenses, the rest stay muted (same moment ~20 %, other moments ~5 %)
  const f0 = frames[cur];
  // a gentle lens: the subject sharp, what is much nearer or farther soft (weaker when the focus is close: a selfie)
  const aperture = 0.008 * smooth(40, 6, d) * clamp(focusSm / 6, 0.12, 1);
  const eye = 1 - smooth(0.1, 1.2, camera.position.distanceTo(f0.pos));   // (only at its eye is the frame the photo's alone)
  frames.forEach((f, i) => {
    const isCur = i === cur ? 1 : 0;
    f.cur = damp(f.cur, isCur, isCur ? 2.2 : 1.6, dt);
    if (isCur) f.settle = Math.min(1, f.settle + dt / 1.8);
    // muted: the same moment at ~20 % of the light (its dots half drawn: the place around the photo is the moment's
    // other photos, as in Unfold the fused frames are around the picture), the other moments at ~5 % (a sixth drawn)
    const same = f.moment === f0.moment;
    const dens = lerp(same ? 0.16 : 0.08, 1, f.cur), light = lerp(same ? 0.2 : 0.05, 1, f.cur);
    f.pts.visible = dbg.only == null || dbg.only === 7;
    // (from far, a photo is a few pixels: draw only the prefix that can show, the rest would pile onto those pixels)
    const px = f.n / Math.max(1, (camera.position.distanceTo(f.focusV) / 4) ** 2);
    f.pts.geometry.setDrawRange(0, Math.max(1, Math.round(f.n * clamp(dens, 0, 1) * clamp(px / 4000, 0.02, 1))));
    const u = f.u;
    u.uCur.value = f.cur; u.uW.value = light; u.uSettle.value = isCur ? f.settle : 1;
    // the full-resolution layer: in once the photo has condensed and you are near it; the base layer gives way by as much
    // (only once arrived: not in flight, and from near the photo's own viewpoint, the drift or its eye)
    const hdT = isCur && f.hdPts && f.hdPts.uploaded ? smooth(0.55, 1, f.settle) * (1 - smooth(25, 70, camera.position.distanceTo(f.focusV)))
      * (1 - smooth(6, 14, camera.position.distanceTo(f.pos))) * (1 - smooth(0.3, 0.75, flySm)) * (flight ? 0 : 1) : 0;
    f.hdw = damp(f.hdw || 0, hdT, 3, dt);
    u.uHDw.value = f.hdw;
    if (f.hdPts) f.hdPts.visible = !f.hdPts.uploaded || (f.hdw > 0.003 && (dbg.only == null || dbg.only === 7));   // (drawn once on arrival: the upload)
    u.uCam.value.copy(camera.position); u.uRayO.value.copy(ro); u.uRayD.value.copy(rd); u.uRayAmt.value = isCur ? rayAmt : 0;
    u.uNear.value = 600; u.uAtEye.value = isCur ? eye : 0;
    u.uPhotoC.value.copy(f0.u.uPhotoA.value); u.uTanC.value.copy(f0.u.uTanA.value); u.uHideA.value = isCur ? 0 : eye;
    if (isCur) {
      const target = gazeDist ?? camera.position.distanceTo(f.focusV);
      focusSm = damp(focusSm, clamp(target, 0.4, 3000), 2.5, dt);
    }
    u.uFocusD.value = focusSm; u.uAperture.value = aperture;
  });
  // true colour while you travel (a flight, the overview, far from the photo); settled on the current photo, the rest
  // only a little greyer
  const togo = cam.T.distanceTo(want.T);
  const focused = (mode === "drift" || mode === "eye" ? 1 : mode === "free" ? 0.6 : 0) * (flight ? 0 : 1)
    * (1 - smooth(30, 150, camera.position.distanceTo(f0.focusV))) * (1 - smooth(3, 30, togo));
  flySm = damp(flySm, 1 - focused, focused > 0.5 ? 0.9 : 2.0, dt);
  envU.uFly.value = flySm;
  frames.forEach((f) => { f.u.uFly.value = flySm; });
  // the glow: the current photo strongest, its moment softly, the other moments faintly (always there)
  frames.forEach((f, i) => {
    const amt = lerp(f.moment === f0.moment ? 0.32 : 0.16, 1.0, f.cur) / Math.sqrt(GLOW_K);
    for (let j = 0; j < GLOW_K; j++) glowAmt.array[i * GLOW_K + j] = amt;
  });
  glowAmt.needsUpdate = true;
  // the place: what the current photo shows is the photo's, seen from its eye
  envU.uPhotoA.value.copy(f0.u.uPhotoA.value); envU.uTanA.value.copy(f0.u.uTanA.value);
  // inside the current photo's frame the place gives way by the photo's own presence (fully at its eye, about half from
  // nearby), so the picture and the place add up to one light instead of a brighter slab
  envU.uHideA.value = eye * f0.cur; envU.uAtEye.value = eye;
  envU.uMoment.value.set(f0.focusV.x, f0.focusV.y, f0.focusV.z, f0.selfie ? 26 : 34);
  envU.uCam.value.copy(camera.position); envU.uRayO.value.copy(ro); envU.uRayD.value.copy(rd); envU.uRayAmt.value = rayAmt;
  envU.uNear.value = Math.max(300, d * 5);
  envU.uFocusD.value = focusSm; envU.uAperture.value = aperture;
}

function paintAttention(dt, d) {
  const brushes = [];
  const calm = Math.exp(-speedSm * 1.6) * Math.exp(-pointer.speed * 0.9);
  if (started && pointer.over && !pointer.down) {
    if (now - lastPickT > 0.06) { lastPickT = now; gazeHit = groundAt(pointer.ndc, gazeHit || V()); gazeDist = gazeHit ? camera.position.distanceTo(gazeHit) : null; }
    if (gazeHit) {
      const r = Math.max(0.05, gazeDist * 0.085), idle = smooth(12, 4, now - pointer.moved);
      brushes.push({ x: gazeHit.x, z: gazeHit.z, r, now: 2.4 * dt * (0.25 + 0.75 * calm) * idle, last: 0.9 * dt * calm * idle });
    }
  } else gazeDist = null;
  // the middle of the view, and (left alone) the people of the current photo
  const still = smooth(0.6, 3.5, now - lastMotion);
  brushes.push({ x: cam.T.x, z: cam.T.z, r: Math.max(0.1, d * 0.25), now: (0.2 + 0.8 * still) * dt * calm, last: (0.06 + 0.35 * still) * dt * calm });
  const f = frames[cur];
  if (now - lastInput > 3) brushes.push({ x: f.focusV.x, z: f.focusV.z, r: Math.max(0.6, f.focusV.distanceTo(camera.position) * 0.12), now: 0.8 * dt, last: 0.2 * dt });
  attention.update(dt, brushes);
}

function updateTime(dt, d) {
  const f = frames[cur];
  const s = sunAt(f.t + (now - changedAt) * 0.5);
  dirFromAzEl(s.az, s.el, skyU.uSunDir.value);
  sunColour(s.el, sunCol);
  skyU.uSunCol.value.copy(sunCol);
  skyU.uGlow.value = damp(skyU.uGlow.value, smooth(-8, 3, s.el) * (1 - smooth(800, 6000, d)), 2, dt);
  skyU.uDay.value = damp(skyU.uDay.value, 0.45 + 0.4 * smooth(-6, 4, s.el), 2, dt);
  skyU.uHaze.value = 0.6 + 0.4 * smooth(LN(3000), LN(50), LN(Math.max(d, 1)));
  const day = smooth(-6, 4, s.el);
  S.uMoon.value.copy(moonBase).lerp(tmpC.copy(sunCol).multiplyScalar(0.6), 0.18 * day).multiplyScalar(0.7 + 0.3 * day);
  S.uLight.value.setRGB(1, 1, 1).lerp(tmpC.copy(sunCol), 0.3);
  const dusk = smooth(0.5, -4, s.el);
  envU.uDusk.value = dusk;
  const ms = T0 + (f.t + (now - changedAt) * 0.5) * 1000, dd = new Date(ms - 4 * 3600e3);
  clockEl.textContent = `Thu 7 May 2026 · ${String(dd.getUTCHours()).padStart(2, "0")}:${String(dd.getUTCMinutes()).padStart(2, "0")}:${String(dd.getUTCSeconds()).padStart(2, "0")}`;
  placeEl.textContent = `Miami Beach · ${d > 400 ? `${(camera.position.y).toFixed(0)} m up` : s.el < 0 ? `sun ${Math.abs(s.el).toFixed(0)}° below the horizon, the afterglow over the city` : `sun ${s.el.toFixed(0)}° above the city`}`;
}

function updateOverlays(d) {
  let depth = 0;
  for (let L = 1; L <= 5; L++) depth += envU.uUnfold.value[L];
  depth += smooth(3, 1.2, d);
  gaugeEls.forEach((e, i) => e.classList.toggle("on", Math.abs(depth - i) < 0.5));
  gaugeHead.style.top = `${(depth / 6) * 100}%`;
  labelEls.forEach((b, i) => {
    const m = moments[i];
    tmpV.copy(m.C).project(camera);
    const k = smooth(120, 500, d);
    const on = k > 0.02 && tmpV.z < 1 && Math.abs(tmpV.x) < 0.98 && Math.abs(tmpV.y) < 0.95;
    b.style.opacity = on ? k.toFixed(3) : "0";
    b.style.pointerEvents = on && k > 0.5 ? "auto" : "none";
    if (on) b.style.transform = `translate(${((tmpV.x * 0.5 + 0.5) * innerWidth).toFixed(1)}px, ${((-tmpV.y * 0.5 + 0.5) * innerHeight).toFixed(1)}px)`;
  });
  whisperEl.style.opacity = d < 200 ? "" : "0";
}

let ceiling = MAX_PR, perfAcc = 0, perfN = 0, perfT = 0, frameMs = 16;
function adapt(dtMs) {
  if (dtMs > 45) return;
  perfAcc += dtMs; perfN++;
  if (now - perfT < 2 || perfN < 30) return;
  const avg = perfAcc / perfN;
  perfAcc = 0; perfN = 0; perfT = now;
  let pr = pixelRatio;
  if (avg > 19.5) { ceiling = Math.max(0.75, pixelRatio - 0.05); pr = Math.max(0.75, pixelRatio - 0.15); }
  else if (avg < 17.6) pr = Math.min(ceiling, pixelRatio + 0.1);
  if (Math.abs(pr - pixelRatio) > 0.01) { pixelRatio = pr; renderer.setPixelRatio(pr); composer.setPixelRatio(pr); }
  frameMs = avg;
}

function tick(dtRaw) {
  const dt = Math.min(dtRaw, 0.1);
  now += dt;
  S.uTime.value = now; skyU.uTime.value = now; finish.uniforms.uTime.value = now;
  if (!started && now > 6 && !dbg.noAuto) begin();
  if (playing && started && now - lastStep > STEP && !pointer.down) setFrame(cur + 1);
  const d = placeCamera(dt);
  manageHD();
  update(dt, d);
  paintAttention(dt, d);
  updateTime(dt, d);
  updateOverlays(d);
  composer.render(dt);
}

// ------------------------------------------------------------------------------------------------ go
overviewPose(cam); cam.T.copy(frames[0].focusV); framePose(frames[0], want, 0);
renderer.compile(scene, camera);
for (const f of frames) { f.pts.geometry.setDrawRange(0, f.n); }
renderer.render(scene, camera);   // upload every buffer now, not when it is first needed
loadEl.classList.add("gone");
setTimeout(() => (loadEl.style.display = "none"), 1000);
$("intro").classList.add("ready");
setFrame(0);
started = false;
updatePlay();
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => { const d = clock.getDelta(); tick(d); adapt(d * 1000); });

function bench(n = 30) {
  const px = new Uint8Array(4), gl = renderer.getContext();
  tick(1 / 60); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) tick(1 / 60);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
  return (performance.now() - t0) / n;
}
function hover(x, y, speed = 0) { pointer.ndc.set(x ?? 0, y ?? 0); pointer.over = x != null; pointer.moved = now; pointer.speed = speed; }
function setMode(m) { mode = m; if (m === "eye") eyePose(frames[cur], want); if (m === "overview") overviewPose(want); }
function jump() { cam.T.copy(want.T); cam.r = want.r; cam.yaw = want.yaw; cam.pitch = want.pitch; cam.fov = want.fov; flight = null; }
window.unfold = { dbg, frames, moments, camera, renderer, composer, scene, tick, bench, begin, setFrame, setMode, jump, hover, zoomBy, cam, want, attention, envU, S,
  get cur() { return cur; }, get mode() { return mode; }, set mode(v) { mode = v; }, set playing(v) { playing = v; updatePlay(); },
  get frameMs() { return frameMs; }, get pixelRatio() { return pixelRatio; }, set pixelRatio(v) { pixelRatio = v; renderer.setPixelRatio(v); composer.setPixelRatio(v); },
  loaded: () => true };
