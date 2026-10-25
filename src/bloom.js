// Soft bloom without squares (copied from recompose/bloom.js). Three's UnrealBloomPass sums its coarse mip levels with plain bilinear sampling, so an
// isolated bright dot turns into a soft square (or a diamond) at 1/16 and 1/32 resolution. Here every level is reached
// with a dual-filter (Kawase) downsample and climbed back up one level at a time with an 8-tap tent, adding each level
// on the way: a point source gives a round, smooth glow at every scale.
import * as THREE from "three";
import { Pass, FullScreenQuad } from "three/addons/postprocessing/Pass.js";

const VS = "varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }";

const PREFILTER = /* glsl */ `
  uniform sampler2D tSrc; uniform vec2 uTexel; uniform float uThreshold, uKnee;
  varying vec2 vUv;
  vec3 pick(vec2 uv) {   // soft-knee threshold on brightness
    vec3 c = texture2D(tSrc, uv).rgb;
    float b = max(max(c.r, c.g), c.b);
    float soft = clamp(b - uThreshold + uKnee, 0.0, 2.0 * uKnee);
    soft = soft * soft / (4.0 * uKnee + 1e-5);
    return c * max(soft, b - uThreshold) / max(b, 1e-5);
  }
  void main() {   // the first halving: the same 5-tap filter as the downsample
    vec2 h = uTexel;
    vec3 s = pick(vUv) * 4.0 + pick(vUv - h) + pick(vUv + h) + pick(vUv + vec2(h.x, -h.y)) + pick(vUv - vec2(h.x, -h.y));
    gl_FragColor = vec4(s / 8.0, 1.0);
  }`;

const DOWN = /* glsl */ `
  uniform sampler2D tSrc; uniform vec2 uTexel; varying vec2 vUv;
  void main() {
    vec2 h = uTexel;   // half a source texel diagonally: each tap is a bilinear 2x2 average
    vec3 s = texture2D(tSrc, vUv).rgb * 4.0 + texture2D(tSrc, vUv - h).rgb + texture2D(tSrc, vUv + h).rgb
           + texture2D(tSrc, vUv + vec2(h.x, -h.y)).rgb + texture2D(tSrc, vUv - vec2(h.x, -h.y)).rgb;
    gl_FragColor = vec4(s / 8.0, 1.0);
  }`;

const UP = /* glsl */ `
  uniform sampler2D tSmall, tSame; uniform vec2 uTexel; uniform float uScatter; varying vec2 vUv;
  vec3 tent(vec2 uv) {   // 8 taps on a ring: a smooth, round upsample of the smaller level
    vec2 h = uTexel;
    vec3 s = texture2D(tSmall, uv + vec2(-2.0 * h.x, 0.0)).rgb + texture2D(tSmall, uv + vec2(2.0 * h.x, 0.0)).rgb
           + texture2D(tSmall, uv + vec2(0.0, -2.0 * h.y)).rgb + texture2D(tSmall, uv + vec2(0.0, 2.0 * h.y)).rgb
           + (texture2D(tSmall, uv + vec2(-h.x, h.y)).rgb + texture2D(tSmall, uv + vec2(h.x, h.y)).rgb
           + texture2D(tSmall, uv + vec2(-h.x, -h.y)).rgb + texture2D(tSmall, uv + vec2(h.x, -h.y)).rgb) * 2.0;
    return s / 12.0;
  }
  void main() { gl_FragColor = vec4(texture2D(tSame, vUv).rgb + tent(vUv) * uScatter, 1.0); }`;

const COMPOSE = /* glsl */ `
  uniform sampler2D tSrc, tGlow; uniform vec2 uTexel; uniform float uStrength; varying vec2 vUv;
  void main() {
    vec2 h = uTexel;
    vec3 g = (texture2D(tGlow, vUv + vec2(-2.0 * h.x, 0.0)).rgb + texture2D(tGlow, vUv + vec2(2.0 * h.x, 0.0)).rgb
           + texture2D(tGlow, vUv + vec2(0.0, -2.0 * h.y)).rgb + texture2D(tGlow, vUv + vec2(0.0, 2.0 * h.y)).rgb
           + (texture2D(tGlow, vUv + vec2(-h.x, h.y)).rgb + texture2D(tGlow, vUv + vec2(h.x, h.y)).rgb
           + texture2D(tGlow, vUv + vec2(-h.x, -h.y)).rgb + texture2D(tGlow, vUv + vec2(h.x, -h.y)).rgb) * 2.0) / 12.0;
    gl_FragColor = vec4(texture2D(tSrc, vUv).rgb + g * uStrength, 1.0);
  }`;

export class SoftBloomPass extends Pass {
  constructor({ strength = 0.6, threshold = 0.5, knee = 0.25, levels = 6, scatter = 0.85 } = {}) {
    super();
    this.strength = strength; this.levels = levels;
    const mat = (fs, uniforms) => new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: fs, uniforms, depthTest: false, depthWrite: false });
    this.pre = mat(PREFILTER, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() }, uThreshold: { value: threshold }, uKnee: { value: knee } });
    this.down = mat(DOWN, { tSrc: { value: null }, uTexel: { value: new THREE.Vector2() } });
    this.up = mat(UP, { tSmall: { value: null }, tSame: { value: null }, uTexel: { value: new THREE.Vector2() }, uScatter: { value: scatter } });
    this.compose = mat(COMPOSE, { tSrc: { value: null }, tGlow: { value: null }, uTexel: { value: new THREE.Vector2() }, uStrength: { value: strength } });
    this.quad = new FullScreenQuad(null);
    const rt = () => new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false });
    this.downs = Array.from({ length: levels }, rt);
    this.ups = Array.from({ length: levels }, rt);
  }

  setSize(w, h) {
    for (let i = 0; i < this.levels; i++) {
      const s = 2 ** (i + 1);
      const W = Math.max(1, Math.round(w / s)), H = Math.max(1, Math.round(h / s));
      this.downs[i].setSize(W, H); this.ups[i].setSize(W, H);
    }
  }

  draw(renderer, material, target) { this.quad.material = material; renderer.setRenderTarget(target); this.quad.render(renderer); }

  render(renderer, writeBuffer, readBuffer) {
    const n = this.levels, v = this._v || (this._v = new THREE.Vector2()), texel = (t) => v.set(0.5 / t.width, 0.5 / t.height);
    this.pre.uniforms.tSrc.value = readBuffer.texture;
    this.pre.uniforms.uTexel.value.copy(texel(readBuffer));
    this.draw(renderer, this.pre, this.downs[0]);
    for (let i = 1; i < n; i++) {
      this.down.uniforms.tSrc.value = this.downs[i - 1].texture;
      this.down.uniforms.uTexel.value.copy(texel(this.downs[i - 1]));
      this.draw(renderer, this.down, this.downs[i]);
    }
    // climb back: each level adds its own light to the smoother light from below
    let small = this.downs[n - 1];
    for (let i = n - 2; i >= 0; i--) {
      this.up.uniforms.tSmall.value = small.texture;
      this.up.uniforms.tSame.value = this.downs[i].texture;
      this.up.uniforms.uTexel.value.copy(texel(this.downs[i]));
      this.draw(renderer, this.up, this.ups[i]);
      small = this.ups[i];
    }
    this.compose.uniforms.tSrc.value = readBuffer.texture;
    this.compose.uniforms.tGlow.value = small.texture;
    this.compose.uniforms.uTexel.value.copy(texel(readBuffer));
    this.compose.uniforms.uStrength.value = this.strength / n;
    this.draw(renderer, this.compose, this.renderToScreen ? null : writeBuffer);
  }

  dispose() { [...this.downs, ...this.ups].forEach((t) => t.dispose()); [this.pre, this.down, this.up, this.compose].forEach((m) => m.dispose()); this.quad.dispose(); }
}
