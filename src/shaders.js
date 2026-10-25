// GLSL for Unfold · Miami Beach (after unfold/shaders.js). Additive light on a near-black void: no sorting, no depth.
// Sizes are in 'view pixels' (CSS pixels x 1.25); uDpr turns them into device pixels.
// Dots carry: position | aCol (rgb, salience) | aAtt (page+16*link, rnd, cover, size) | aPar (to the parent, focus); see bake.py.

const COMMON = /* glsl */ `
  #define PI 3.14159265
  uint hsh(uint x) { x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u; return x; }
  float h01(uint x) { return float(hsh(x) & 0xFFFFFFu) / 16777215.0; }
  float sq(float x) { return x * x; }
  float vn(vec2 p) {
    vec2 i = floor(p), f = fract(p), u = f * f * (3.0 - 2.0 * f);
    uint x = uint(int(i.x) + 1048576), y = uint(int(i.y) + 1048576);
    float a = h01(x * 73856093u ^ y * 19349663u), b = h01((x + 1u) * 73856093u ^ y * 19349663u);
    float c = h01(x * 73856093u ^ (y + 1u) * 19349663u), d = h01((x + 1u) * 73856093u ^ (y + 1u) * 19349663u);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
  }
  float fbm01(vec2 p) { return 0.6 * vn(p) + 0.3 * vn(p * 2.07 + vec2(5.3, -1.7)) + 0.1 * vn(p * 4.13 + vec2(-3.1, 7.7)); }
`;

// ------------------------------------------------------------------------------------------------ attention
// The attention atlas (5 x 4 tiles): the whole island, the beach along the walk, and around each of the three moments a
// near and an eye tile. R: attention now (seconds), G: explored (minutes). A finer tile's say fades out toward its edge.
const ATTN = /* glsl */ `
  uniform sampler2D uAtt;
  uniform vec4 uTileS, uTileF;          // centre x z (world), half extent, tile index
  uniform vec4 uTileN[3], uTileE[3];    // per moment
  vec2 tileUV(vec4 t, vec2 xz) {
    vec2 f = (xz - t.xy) / (2.0 * t.z) + 0.5;
    float col = mod(t.w, 5.0), row = floor(t.w / 5.0);
    return (vec2(col, row) + clamp(f, 0.002, 0.998)) / vec2(5.0, 4.0);
  }
  float tileW(vec4 t, vec2 xz) {
    if (t.w < 0.0) return 0.0;
    vec2 d = (xz - t.xy) / t.z;
    float m = max(abs(d.x), abs(d.y));
    if (m > 0.97) return 0.0;
    return smoothstep(0.96, 0.5, m + 0.14 * (fbm01(d * 3.0 + t.w) * 2.0 - 1.0)) * smoothstep(0.97, 0.9, m);
  }
  vec2 tileMix(vec2 a, vec4 t, vec2 xz) { float w = tileW(t, xz); return w > 0.0 ? mix(a, max(a, textureLod(uAtt, tileUV(t, xz), 0.0).rg), w) : a; }
  vec2 attn(vec3 p) {
    vec2 xz = p.xz;
    vec2 a = textureLod(uAtt, tileUV(uTileS, xz), 0.0).rg;
    a = tileMix(a, uTileF, xz);
    for (int i = 0; i < 3; i++) {
      if (max(abs(xz.x - uTileN[i].x), abs(xz.y - uTileN[i].y)) < uTileN[i].z) {
        a = tileMix(a, uTileN[i], xz);
        a = tileMix(a, uTileE[i], xz);
      }
    }
    return a;
  }
`;

// ------------------------------------------------------------------------------------------------ the dots
// One shader for the pages of the place and for the photos (IS_PHOTO). As in Unfold: every hand-off complementary,
// every reach a wandering feather (baked), attention through world-fixed noise, the photo's frame a dissolve both ways.
export const DOT_VS = /* glsl */ `
  ${COMMON}
  ${ATTN}
  uniform float uTime, uPx, uDpr, uSpeed, uRemember, uSigma;
  uniform vec2 uSizeLog;
  uniform vec3 uMoon, uLight;
  uniform float uUnfold[8], uPreview[8], uCell[8];
  uniform vec3 uCam, uRayO, uRayD; uniform float uRayAmt;
  uniform mat4 uPhotoA; uniform vec2 uTanA;   // the photo: (place) the current one, its frame its own at its eye; (photo) its own
  uniform mat4 uPhotoC; uniform vec2 uTanC;   // (photo) the current photo, whose frame a muted one gives way inside at its eye
  uniform float uHideA, uAtEye;
  uniform float uFocusD, uAperture, uNear, uFade;
  uniform float uCur, uW, uSettle, uWarm, uWarmGain, uDusk;   // (photos) current or not, its weight, condensing, warmth
  uniform float uHDw;                      // (photos) how far the full-resolution layer has come in
  uniform float uFly;                      // 1 travelling (true colour everywhere), 0 settled on the current photo
  uniform vec3 uWalk[3];                   // the three moments: the beach around them is drawn fuller (fly-overs)
  uniform vec4 uMoment;                    // (place) where the current photo is (its focus) and how far its glow reaches
  attribute vec4 aCol, aAtt, aPar;
  varying vec3 vCol; varying float vA, vS, vFlat;

  float T(float u) { return smoothstep(0.12, 0.72, u); }
  // the photo's frame, seen from its camera: b = 0 at the frame's edge, 1 at its centre (-1 behind it), and a noise
  // that makes every border wander. Two weights from it, both shared by the photo and the place (complementary):
  //   keep      a narrow dissolve right at the edge (no line)
  //   presence  a wide taper from the centre outward: the photo thins and cools toward its edges exactly as the place
  //             comes back, so the picture fuses into the dots around it instead of standing as a slab
  vec2 frameB(mat4 M, vec2 tanv, vec3 q) {
    vec4 c = M * vec4(q, 1.0);
    if (-c.z < 0.25) return vec2(-1.0, 0.0);
    vec2 n = c.xy / -c.z / tanv;
    float bx = 1.0 - max(abs(n.x), abs(n.y));
    if (bx > 0.8) return vec2(bx, 0.0);   // (deep inside, the noise cannot change either weight: skip it)
    // the border wanders in the picture and on the ground (seen from aside, the frustum's sides are not straight lines)
    float nz = 0.55 * (fbm01(n * 2.3 + vec2(7.1, 3.3)) * 2.0 - 1.0) + 0.75 * (fbm01(q.xz * 0.45 + vec2(1.7, 9.2)) * 2.0 - 1.0);
    return vec2(1.0 - max(abs(n.x), abs(n.y)), nz);
  }
  float keepOf(vec2 bn) { return bn.x <= 0.0 ? 0.0 : smoothstep(0.0, 0.24, bn.x + 0.2 * bn.y * smoothstep(0.0, 0.12, bn.x)); }
  float presenceOf(vec2 bn) { return bn.x <= 0.0 ? 0.0 : smoothstep(0.0, 0.45, bn.x + 0.24 * bn.y * smoothstep(0.0, 0.3, bn.x)); }

  void main() {
    uint id = uint(gl_VertexID);
    float h1 = h01(id), h2 = h01(id ^ 0x9e3779b9u), h3 = h01(id ^ 0x85ebca6bu);
    float pl = floor(aAtt.x * 255.0 + 0.5);
    float page = mod(pl, 16.0), link = floor(pl / 16.0);
    int L = int(page);
    float rnd = aAtt.y, cover = aAtt.z, sal = aCol.w, focus = aPar.w;
    // (the place: focus < 0 on the sea, its magnitude the coast and surf)
    float isSea = step(focus, -0.001), coastF = abs(focus);
    #ifdef IS_PHOTO
      isSea = 0.0; coastF = focus;
    #endif
    float size = exp(mix(uSizeLog.x, uSizeLog.y, aAtt.w));
    vec3 self = position;
    #ifdef IS_PHOTO
      float isMoment = 1.0;
      float isHD = step(7.5, link);
      float person = step(0.5, mod(link, 8.0));
      // the full-resolution layer and the base layer of the current photo share it by complementary weights
      // (its sky stays in the base layer: the full-resolution one has none)
      // (in the drift its people are at full resolution, the rest of it too only at its eye: seen from aside the base
      // layer's sparser grain hides what the photo's depth gets wrong)
      float hdOn = uHDw * uCur * mix(uAtEye, 1.0, person);
      float hdMix = mix(1.0 - hdOn * (1.0 - step(0.5, cover)), hdOn, isHD);
      if (hdMix < 0.004) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      // at the current photo's eye a muted photo is hidden well inside its frame (the full hide below, without its noise)
      if (uCur < 0.5 && uHideA > 0.995) {
        vec4 cC = uPhotoC * vec4(self, 1.0);
        vec2 nC = cC.xy / max(-cC.z, 1e-4) / uTanC;
        if (-cC.z > 0.25 && 1.0 - max(abs(nC.x), abs(nC.y)) > 0.78) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      }
      float gL = 1.0, pv = 0.0, gC = 0.0, pC = 0.0;
    #else
      float isMoment = 0.0, person = 0.0;
      float gL = uUnfold[L], pv = uPreview[L];
      int lk = min(int(link), 7);
      float gC = L < 5 ? uUnfold[lk] : 0.0, pC = L < 5 ? uPreview[lk] : 0.0;
      if (clamp(gL + pv, 0.0, 1.0) * uFade < 0.002) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      // level of detail, before anything costly: the most this dot could be wanted from here (its own spacing on screen
      // against the smallest spacing the rules below can ask for); dots past it are never drawn, so skip them now
      {
        float dc0 = max(distance(self, uCam) - length(aPar.xyz), 1e-3);
        float sMin = (uMoment.w > 0.0 && distance(self.xz, uMoment.xz) < uMoment.w * 1.4) ? 0.9 : 1.8;
        float k0 = sq(size * uPx / dc0 / sMin);
        float wantMax = max(min(1.0, 3.3 * k0 + 0.4 * max(k0 - 1.0, 0.0)), 0.75 * min(1.0, 30.0 * k0) * step(0.001, abs(focus)));
        if (rnd > max(wantMax, uRemember) + 0.03) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      }
      // at the photo's eye, what lies well inside its frame is the photo's alone (the full hide below, without its noise)
      if (uHideA > 0.995) {
        vec4 cA = uPhotoA * vec4(self, 1.0);
        vec2 nA = cA.xy / max(-cA.z, 1e-4) / uTanA;
        if (-cA.z > 0.25 && 1.0 - max(abs(nA.x), abs(nA.y)) > 0.78) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      }
    #endif

    #ifdef IS_PHOTO
      vec2 at = isHD > 0.5 ? vec2(0.0) : attn(self);   // (the full-resolution layer is the picture: no attention in it)
    #else
      vec2 at = attn(self);
    #endif
    float nS = 1.0, nC = 1.0;
    if (at.x + at.y > 0.002) {
      vec2 q = self.xz / (90.0 * uCell[L]), qc = self.xz / (90.0 * uCell[min(L + 1, 7)]);
      nS = exp2(7.0 * (0.6 * vn(q + vec2(17.0 * page)) + 0.4 * vn(q * 3.7 + vec2(3.0 * page)) - 0.5));
      nC = exp2(7.0 * (0.6 * vn(qc + vec2(17.0 * (page + 1.0))) + 0.4 * vn(qc * 3.7 + vec2(3.0 * (page + 1.0))) - 0.5));
    }
    float A = 1.0 - exp(-1.6 * at.y * nS), Ar = 1.0 - exp(-1.8 * at.x * nS);
    float AC = 1.0 - exp(-1.6 * at.y * nC);

    // unfolding (the place by height, ahead where you have looked; the current photo as it condenses)
    #ifdef IS_PHOTO
      float un = mix(1.0, uSettle, uCur);
      float ue = smoothstep(0.0, 0.9, un);
    #else
      float un = clamp(gL + (1.0 - gL) * pv * smoothstep(0.1, 0.8, A), 0.0, 1.0);
      float ue = smoothstep(0.0, 0.62, un);
    #endif
    vec3 p = self + aPar.xyz * (1.0 - ue);
    float uc = clamp(gC + (1.0 - gC) * pC * smoothstep(0.1, 0.8, AC), 0.0, 1.0);
    float keepV = 1.0 - cover * T(uc);
    float frameHide = 0.0, frameKeep = 1.0, presence = 1.0, isSky = 0.0;
    #ifdef IS_PHOTO
      isSky = isHD > 0.5 ? cover : step(0.5, cover);   // (full resolution: the sky right along a subject, a degree of sky)
      if (isSky > 0.5 && uCur < 0.02) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }   // only the current photo's sky
      vec2 bn = frameB(uPhotoA, uTanA, self);
      presence = presenceOf(bn);
      frameKeep = keepOf(bn) * mix(mix(0.22, 1.0, presence), mix(0.5, 1.0, presence), person);
      frameKeep *= mix(1.0, presence * presence * presence * 0.7, isSky);   // the sky dissolves widely: no block of sky
      // the sand right at the photo's feet frays out (it is the frame's bottom edge seen from anywhere else)
      float nearOff = max(max(person, isSky), isHD * uAtEye);   // (at its eye, at full resolution: the whole picture)
      if (nearOff < 0.999) {
        vec4 qc = uPhotoA * vec4(self, 1.0);
        float nearF = smoothstep(1.2, 5.0, length(qc.xyz) + 1.6 * (fbm01(self.xz * 0.8 + vec2(4.1, 2.3)) * 2.0 - 1.0));
        frameKeep *= mix(nearF, 1.0, nearOff);
      }
      if (uCur < 0.5 && uHideA > 0.001) frameHide = presenceOf(frameB(uPhotoC, uTanC, self)) * uHideA;
      float vis = mix(1.0, smoothstep(0.0, 0.5, un), uCur) * hdMix;
      vis *= mix(smoothstep(2.5, 9.0, distance(self, uCam)), 1.0, uCur);   // a muted photo never stands in your face (no murk)
    #else
      if (uHideA > 0.001) frameHide = presenceOf(frameB(uPhotoA, uTanA, self)) * uHideA;
      float vis = T(un);
    #endif
    vis *= keepV;
    float mid = ue * (1.0 - ue) * 4.0;
    vec3 rnd3 = vec3(h1, h2, h3) - 0.5;
    p += (rnd3 + vec3(sin(uTime * 0.7 + h1 * 6.283), cos(uTime * 0.6 + h2 * 6.283), sin(uTime * 0.5 + h3 * 6.283)) * 0.4) * length(aPar.xyz) * 0.35 * mid;
    p += vec3(sin(uTime * (0.5 + 0.4 * h1) + 6.283 * h2), sin(uTime * (0.45 + 0.5 * h2) + 6.283 * h3), sin(uTime * (0.55 + 0.3 * h3) + 6.283 * h1)) * size * 0.1;

    float dc = distance(p, uCam);
    vec3 v = p - uRayO;
    float al = dot(v, uRayD);
    float ray = al > 0.0 ? exp(-sq(length(v - uRayD * al) / (0.06 * al + 0.02))) * uRayAmt : 0.0;
    if (ray > 0.002) ray *= min(nS, 2.5);
    float now = clamp(Ar + ray * 0.8, 0.0, 1.0);

    // density: the dot's own spacing on screen against the spacing wanted (wide unexplored, close where you look, wider
    // in fast motion); structure first. The current photo is dense; muted photos are drawn as a prefix (main.js)
    float delta = size * uPx / max(dc, 1e-3);
    // attention is only a subtle extra now: the place reads on its own
    float att = clamp(A * 0.45 + now * 0.3 + uRemember, 0.0, 1.0);
    // the walk: within ~300 m of the three moments the beach is drawn fuller and a little larger (its border wanders)
    float strip = 0.0;
    #ifndef IS_PHOTO
    {
      vec2 a0 = uWalk[0].xz, a1 = uWalk[1].xz, a2 = uWalk[2].xz;
      vec2 e1 = a1 - a0, e2 = a2 - a1;
      float d1 = length(self.xz - (a0 + e1 * clamp(dot(self.xz - a0, e1) / dot(e1, e1), 0.0, 1.0)));
      float d2 = length(self.xz - (a1 + e2 * clamp(dot(self.xz - a1, e2) / dot(e2, e2), 0.0, 1.0)));
      strip = smoothstep(420.0, 200.0, min(d1, d2) + 90.0 * (fbm01(self.xz / 180.0) * 2.0 - 1.0));
    }
    #endif
    float sigma = mix(mix(uSigma * mix(1.0, 0.68, strip), mix(3.2, 2.0, uCur), isMoment), mix(1.9, 1.4, isMoment), att) * (1.0 + 0.9 * uSpeed);
    // around the current photo the place gathers: closer dots, a little of its colour (a wandering reach, no disc)
    float mN = 0.0;
    if (uMoment.w > 0.0 && isMoment < 0.5) {
      float dm = distance(self.xz, uMoment.xz);
      mN = smoothstep(uMoment.w, 0.25 * uMoment.w, dm + 0.35 * uMoment.w * (fbm01(self.xz / (0.4 * uMoment.w)) * 2.0 - 1.0));
      sigma *= mix(1.0, 0.5, mN);
    }
    float k = sq(delta / sigma);
    float graze = clamp(abs(p.y - uCam.y) / max(dc, 1e-3), 0.025, 1.0);
    k *= mix(1.0, max(graze, 0.06 * isSea), (1.0 - isMoment) * smoothstep(8.0, 40.0, dc));
    // (the beach is flat and even: its sand keeps a floor of dots, so up close it reads as sand, not as a hole)
    float w = mix(mix(0.2 + 3.0 * pow(sal, 3.0), 0.25 + 2.4 * sal * sal, isMoment), 1.0, max(att * 0.85, 0.62 * strip));
    float want = k * w;
    // where the place's own dots are sparser on screen than wanted (very near: the sand at your feet), show them all
    want += (1.0 - isMoment) * max(k - 1.0, 0.0) * 0.4;
    want = max(want, isMoment * person * mix(0.3, 0.85, uCur) * min(1.0, k * 6.0));   // the people are always there (until they would crowd one pixel)
    want = max(want, (1.0 - isMoment) * coastF * 0.7 * min(1.0, k * 30.0));   // the shoreline stays drawn
    #ifdef IS_PHOTO
      want = max(want, isHD * min(1.0, k * 5.0));   // full resolution: every pixel, until they would crowd the screen
    #endif
    want *= frameKeep * (1.0 - frameHide);
    float on = smoothstep(rnd - 0.015, rnd + 0.015, want);
    vis *= on * uFade;
    if (vis < 0.003) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }

    vec4 mv = modelViewMatrix * vec4(p, 1.0);
    float z = max(-mv.z, 1e-3);
    float pxs = size * uPx / z;
    // fine, crisp dots: the place's never more than ~3 px however near (sparse up close: parts left out), the photo's ~4.5
    // the place's dots ~1.45x the photos' (legible, still crisp and round: a bright core, never a flat blob)
    float envS = mix(1.45 * mix(1.0, 1.15, strip), 1.0, isMoment);
    float s = clamp(pxs * 0.5 * envS, 1.6 * envS, mix(2.2 * envS, mix(3.3 * envS, 4.5, isMoment), smoothstep(5.0, 40.0, pxs)));
    float fill = clamp(sqrt(delta / sigma), 1.0, mix(1.25, 1.3, isMoment));
    s *= fill * (1.0 + 0.3 * sal * sal * (1.0 - isMoment));
    float coc = uAperture * uPx * abs(1.0 / uFocusD - 1.0 / z);
    coc = z < uFocusD ? coc : min(coc * 0.3, 3.0);
    s = clamp(sqrt(s * s + coc * coc), 1.0, 36.0);
    s = max(s, mix(1.0, mix(1.6, 2.2, uCur), isMoment));   // the current photo: dots just touching, a picture of fine grain
    #ifdef IS_PHOTO
      if (isHD > 0.5) s = clamp(pxs * 1.7, 1.4, 9.0);        // full resolution: each pixel a soft dot overlapping its neighbours (a continuous picture, no grain)
    #endif
    float energy = min((3.4 + 0.15 * s) / (s * s), 1.0);

    // colour. The place in true colour, exposed for a readable evening: darks lifted along a tone curve (not flattened),
    // the aerial haze taken out (a little more saturation), the sand warm. Settled on the current photo (uFly -> 0), the
    // rest only a little greyer and dimmer; where you attend it is in full colour again. The current photo keeps its own
    // colours: no lift toward white (skin stays skin), only the dark of the dusk raised a little
    const vec3 LW = vec3(0.2126, 0.7152, 0.0722);
    vec3 tc = pow(aCol.rgb, vec3(2.2));
    float lum = dot(tc, LW);
    float lumT = 1.0 - exp(-lum * mix(3.4, 2.6, isMoment));
    vec3 trueC = max(mix(vec3(lum), tc, mix(1.28, 1.08, isMoment)), 0.0) * clamp(lumT / max(lum, 0.004), 1.0, 4.0);
    trueC *= mix(vec3(1.09, 1.0, 0.86) * uLight, vec3(1.0), isMoment);   // the evening's warmth on the sand
    vec3 focusC = mix(vec3(dot(trueC, LW)), trueC, 0.74) * 0.8;
    float truth = clamp(max(uFly + now * 1.1 + A * 0.5 + uRemember + 0.35 * mN, isMoment * uCur), 0.0, 1.0);
    // (the current photo in its own colours throughout: no patches where attention happened to fall)
    float warm = isMoment * uCur * uWarm * mix(0.3, 1.0, max(presence, person));
    vec3 c = mix(focusC, trueC, truth);
    c = mix(c, tc * clamp(0.1 / max(lum, 0.02), 1.0, 1.4) * uWarmGain, warm);   // the current photo: its own colours
    // (its sand a little warmer, toward the place's, more so toward its edges where the place comes back; skin barely)
    c *= mix(vec3(1.0), mix(vec3(1.07, 1.0, 0.88), vec3(1.16, 1.0, 0.74), 1.0 - presence), warm * mix(0.8, 0.35, person));
    float coastal = (1.0 - isMoment) * coastF * mix(0.6, 1.3, isSea);
    // the sea: slow swells travelling in from the east, crests catching the light, a fine shimmer
    if (isSea > 0.5) {
      float ph = self.x * 0.05 - self.z * 0.012 + 1.4 * fbm01(self.xz / 90.0) * 6.283 + uTime * 0.85;
      float crest = pow(0.5 + 0.5 * sin(ph), 6.0);
      c *= 0.72 + 0.75 * crest + 0.25 * pow(0.5 + 0.5 * sin(uTime * (1.3 + 2.0 * h1) + 6.283 * h2), 8.0);
    }
    // the surf: white flecks running along the waterline
    c += vec3(0.85, 0.93, 1.0) * coastal * (0.3 + 0.8 * pow(0.5 + 0.5 * sin(uTime * 0.7 - (self.x - self.z) * 0.02 + h2 * 2.0), 4.0));
    float tw = pow(0.5 + 0.5 * sin(uTime * (1.1 + 2.3 * h1) + 6.283 * h2), 10.0);
    c += vec3(1.0, 0.72, 0.32) * tw * warm * focus * 0.35;            // a few gold flecks around the people
    float cityW = (1.0 - isMoment) * uDusk * step(0.82, h1) * smoothstep(1.5, 6.0, p.y) * step(h3, smoothstep(60.0, 300.0, dc));
    c = mix(c, vec3(1.0, 0.7, 0.4) * (0.7 + 0.8 * h2), cityW);     // the hotels lighting up after sunset
    float twk = pow(0.5 + 0.5 * sin(uTime * (0.6 + 1.4 * h3) + 6.283 * h1), 4.0);
    c *= mix(0.74 + 0.52 * twk, 0.92 + 0.16 * twk, isMoment * uCur);   // the current photo holds still (a picture, not a shimmer)
    c *= 1.0 + 0.3 * mid;
    c *= 1.0 + 0.2 * now;
    float far = 1.0 / (1.0 + pow(dc / (max(uNear, 1.0) * mix(1.0, 7.0, isSea)), mix(1.5, 1.15, isSea)));   // the sea reaches the horizon
    far = mix(far, mix(1.0, far, smoothstep(300.0, 900.0, dc)), isMoment);
    float crowd = clamp(want, 0.0, 1.0) / max(delta * delta * mix(1.0, graze, (1.0 - isMoment) * smoothstep(8.0, 40.0, dc)), 0.05) * s * s * 0.6;
    // how much of the screen the dots light (a dot of size s lights ~0.24 s^2 px): the place's dense patches give less
    // light per dot (they glow, they do not burn); the current photo is brought to ~85 % of its picture's brightness
    // whatever its dot density (fine dots, a luminous image), the muted ones stay faint
    // (from the density expected around the dot, not its own: every dot of a patch gets the same light, no speckle)
    float wAvg = mix(0.5, 1.0, att * 0.85);
    float wantAvg = max(k * wAvg, isMoment * person * mix(0.3, 0.85, uCur) * min(1.0, k * 6.0));
    #ifdef IS_PHOTO
      wantAvg = max(wantAvg, isHD * min(1.0, k * 5.0));
    #endif
    // the light a dot spreads (its profile integrated over its disc, view px^2) times the dots per px^2 expected there
    float kProf = mix(3.2, 1.7, isMoment * uCur);
    #ifdef IS_PHOTO
      kProf = mix(kProf, 1.0, isHD);   // (the full-resolution dots flatter still: overlapping, they add up to an even picture)
    #endif
    float foot = 0.7854 * (1.0 - exp(-kProf)) / kProf * s * s;
    float cov = clamp(wantAvg, 0.0, 1.0) * foot / max(delta * delta * mix(1.0, graze, (1.0 - isMoment) * smoothstep(8.0, 40.0, dc)), 0.05);
    float mood = (mix(1.6, 1.1, isMoment) + mix(0.5, 0.3, isMoment) * sal) * mix(1.0, 0.75, att * (1.0 - warm)) * mix(0.85, 1.0, truth)
               * mix(1.0, 1.45, (1.0 - isMoment) * step(page, 1.5));   // the island from high up: a little brighter
    if (isMoment > 0.5) {
      // a photo: exposure by coverage. The current one shows at ~70 % of its picture's own brightness whatever its dot
      // density (a picture of fine grain, never brighter than the photo: skin stays skin), the muted ones faint
      float E = mix(mix(mix(0.42, 0.58, uAtEye), 0.72, max(person, smoothstep(0.0, 0.5, focus))), 0.08, isSky);   // the people in their light, the rest of the
      E *= mix(1.0, 0.2, uFly * (1.0 - person));                            // in flight its sand stays quiet (no pale patch)
      E *= mix(mix(0.7, 1.0, presence), 1.0, person);                       // and toward its edges a little quieter (no pale slab)
      // (seen from further off the photo is not exposed up: its glow marks it, its sparse dots stay faint)
      float e = clamp(E / max(cov, 1e-3), 0.02, mix(2.2, 0.9, smoothstep(8.0, 30.0, dc)));   // (a low floor: big blurred dots near the camera stay faint)
      // (a muted photo: its people a faint presence, its sand fainter still: no pale blotches on the beach from above)
      energy = mix(clamp(mix(0.12, 0.3, person) / max(cov, 1e-3), 0.05, 1.0) * energy * mood * mix(1.0, 0.3, uFly * (1.0 - person)), e, uCur);
    } else {
      energy *= min(1.0, 0.95 / max(crowd, 1e-4)) * mood;
    }
    float mute = mix(1.0, max(uW * mix(0.4, 1.0, person), uCur), isMoment);   // a muted photo: its people, barely its sand
    vA = vis * energy * far * mute;
    vCol = c;
    vS = s;
    vFlat = isMoment * uCur;
    #ifdef IS_PHOTO
      vFlat = mix(vFlat, 1.47, isHD);   // (profile exp(-1.0 r^2), as kProf above)
    #endif
    gl_PointSize = s * uDpr;
    gl_Position = projectionMatrix * mv;
  }
`;

export const DOT_FS = /* glsl */ `
  varying vec3 vCol; varying float vA, vS, vFlat;
  void main() {
    float r = length(gl_PointCoord - 0.5) * 2.0;
    if (r > 1.0) discard;
    float core = exp(-r * r * mix(3.2, 1.7, vFlat));   // the current photo's grain a little flatter (no white pin-points)
    float disc = (1.0 - smoothstep(0.8, 1.0, r)) * (0.72 + 0.28 * smoothstep(0.45, 0.95, r));
    float a = mix(core, disc, smoothstep(5.0, 16.0, vS));
    gl_FragColor = vec4(vCol * vA * a, 1.0);
  }
`;

export const BRUSH_VS = /* glsl */ `
  attribute vec4 iTile; attribute vec4 iBrush; attribute vec2 iAmt;
  varying vec2 vQ; varying vec2 vAmt;
  void main() {
    vec2 q = position.xy;
    vec2 t = iBrush.xy + q * iBrush.z * 2.2;
    vQ = q * 2.2; vAmt = iAmt;
    vec2 a = iTile.xy + clamp(t, 0.0, 1.0) * iTile.zw;
    gl_Position = vec4(a * 2.0 - 1.0, 0.0, 1.0);
  }
`;
export const BRUSH_FS = /* glsl */ `
  varying vec2 vQ; varying vec2 vAmt;
  void main() { gl_FragColor = vec4(vAmt * exp(-dot(vQ, vQ) * 1.2), 0.0, 1.0); }
`;

export const SKY_VS = /* glsl */ `
  varying vec3 vDir;
  void main() { vDir = position; vec4 p = projectionMatrix * vec4(mat3(viewMatrix) * position, 1.0); gl_Position = vec4(p.xy, p.w * 0.99999, p.w); }
`;
export const SKY_FS = /* glsl */ `
  ${COMMON}
  uniform vec3 uSunDir, uSunCol; uniform float uDay, uTime, uStars, uGlow, uHaze;
  varying vec3 vDir;
  void main() {
    vec3 d = normalize(vDir);
    vec3 c = mix(vec3(0.002, 0.0025, 0.006), vec3(0.012, 0.016, 0.034), uDay) * (0.6 + 0.4 * smoothstep(-0.1, 0.3, d.y));
    c += mix(vec3(0.004, 0.005, 0.01), vec3(0.03, 0.034, 0.05), uDay) * exp(-abs(d.y) * 9.0) * uHaze;
    float ca = max(dot(d, uSunDir), 0.0) * smoothstep(-0.12, 0.02, d.y);
    float low = exp(-abs(uSunDir.y) * 7.0);
    vec3 sd = normalize(vec3(uSunDir.x, 0.0, uSunDir.z));
    float az = max(dot(normalize(vec3(d.x, 0.0, d.z)), sd), 0.0);
    c += uSunCol * uGlow * (pow(ca, 900.0) * 1.2 + pow(ca, 48.0) * 0.1 + pow(ca, 5.0) * 0.03);
    // the afterglow: a wide warm band along the horizon on the sun's side, after the sun has gone
    c += uSunCol * uGlow * low * (0.05 * pow(az, 1.5) * exp(-max(d.y, 0.0) * 9.0) * smoothstep(-0.06, 0.02, d.y));
    vec3 q = d * 150.0, g = floor(q);
    ivec3 gi = ivec3(g) + 4096;
    uint hs = hsh(uint(gi.x) * 73856093u ^ uint(gi.y) * 19349663u ^ uint(gi.z) * 83492791u);
    vec3 jit = vec3(h01(hs ^ 0x68e31da4u), h01(hs ^ 0xb5297a4du), h01(hs ^ 0x1b56c4e9u)) * 0.6 + 0.2;
    float st = step(0.9965, h01(hs)) * (1.0 - smoothstep(0.0, 0.16, length(fract(q) - jit)));
    c += vec3(0.6, 0.7, 1.0) * st * uStars * (0.55 + 0.45 * sin(uTime * 0.8 + float(hs & 255u))) * smoothstep(-0.05, 0.2, d.y);
    gl_FragColor = vec4(c, 1.0);
  }
`;

export const TRAIL = {
  uniforms: { tDiffuse: { value: null }, tPrev: { value: null }, uKeep: { value: 0 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse, tPrev; uniform float uKeep; varying vec2 vUv;
    void main() {
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      vec3 p = uKeep > 0.001 ? texture2D(tPrev, vUv).rgb * uKeep : vec3(0.0);
      vec3 o = max(c, p);
      if (!(max(max(o.r, o.g), o.b) < 6e4)) o = vec3(0.0);
      gl_FragColor = vec4(o, 1.0);
    }`,
};

export const FINISH = {
  uniforms: { tDiffuse: { value: null }, uTime: { value: 0 }, uAspect: { value: 1 }, uGrain: { value: 0.5 }, uVig: { value: 0.6 }, uFade: { value: 0 } },
  vertexShader: `varying vec2 vUv; void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: /* glsl */ `
    uniform sampler2D tDiffuse; uniform float uTime, uAspect, uGrain, uVig, uFade; varying vec2 vUv;
    float h(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
    void main() {
      vec3 c = texture2D(tDiffuse, vUv).rgb;
      vec2 d = vUv - 0.5; d.x *= uAspect;
      c *= 1.0 - smoothstep(0.35, 1.0, length(d)) * uVig;
      // a soft shoulder that keeps the hue: brightness compressed by the brightest channel (skin never goes white)
      float m = max(max(c.r, c.g), c.b);
      float mm = m < 0.7 ? m : 0.7 + 0.3 * (1.0 - exp(-(m - 0.7) / 0.3));
      c *= mm / max(m, 1e-5);
      c += (h(vUv * 931.0 + fract(uTime * 7.31)) - 0.5) * 0.018 * uGrain;
      c *= 1.0 - uFade;
      gl_FragColor = vec4(max(c, 0.0), 1.0);
    }`,
};

// ------------------------------------------------------------------------------------------------ the photos' glow
// Each photo is a soft warm light in the place, like a lantern or a developed print catching the light: a few round
// gaussian sprites over where its people are. From far a few pixels at least (you can spot every moment), near a wide
// soft halo that stays faint (it never washes the photo out).
export const GLOW_VS = /* glsl */ `
  uniform float uPx, uDpr, uTime;
  attribute vec3 aColor; attribute float aAmt, aR;
  varying vec3 vCol; varying float vA, vS;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    float z = max(-mv.z, 0.05);
    float px = aR * uPx / z;
    float minPx = mix(14.0, 44.0, smoothstep(40.0, 900.0, z));     // from far a lantern you can find
    float s = clamp(max(px, minPx), 4.0, 420.0);
    float near = mix(0.22, 1.0, smoothstep(4.0, 40.0, z));        // near the photo: a faint halo around it
    float breathe = 0.85 + 0.15 * sin(uTime * 0.6 + position.x * 0.37 + position.z * 0.11);
    // the light spreads over its disc: a wider sprite is fainter per pixel (the total stays about the same)
    vA = aAmt * near * breathe * clamp(2600.0 / (s * s), 0.03, 1.0) * 1.6;
    vCol = aColor;
    vS = s;
    gl_PointSize = s * uDpr;
    gl_Position = projectionMatrix * mv;
    if (aAmt < 0.002 || mv.z > -0.05) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; }
  }
`;
export const GLOW_FS = /* glsl */ `
  varying vec3 vCol; varying float vA, vS;
  void main() {
    float r2 = dot(gl_PointCoord - 0.5, gl_PointCoord - 0.5) * 4.0;
    if (r2 > 1.0) discard;
    float g = exp(-r2 * 4.2) * (1.0 - r2);
    gl_FragColor = vec4(vCol * vA * g, 1.0);
  }
`;
