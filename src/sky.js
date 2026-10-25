// Time and light (copied from recompose/sky.js): the sun's real position (NOAA algorithm, as in october/tools/sun.py), local-time formatting
// (Pacific Daylight Time, UTC-7, all of October 2026), and conversions between GPS and world metres.

const RAD = Math.PI / 180;

/** Sun azimuth (deg from north, clockwise) and elevation (deg) at a UTC time (ms) and place. */
export function sunPosition(ms, lat, lon) {
  const jd = ms / 86400000 + 2440587.5;
  const t = (jd - 2451545) / 36525;
  const l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360;
  const m = (357.52911 + t * (35999.05029 - 0.0001537 * t)) * RAD;
  const e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t);
  const c = Math.sin(m) * (1.914602 - t * (0.004817 + 0.000014 * t)) + Math.sin(2 * m) * (0.019993 - 0.000101 * t) + Math.sin(3 * m) * 0.000289;
  const lam = (l0 + c - 0.00569 - 0.00478 * Math.sin((125.04 - 1934.136 * t) * RAD)) * RAD;
  const eps = (23 + (26 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60) / 60 + 0.00256 * Math.cos((125.04 - 1934.136 * t) * RAD)) * RAD;
  const dec = Math.asin(Math.sin(eps) * Math.sin(lam));
  const y = Math.tan(eps / 2) ** 2;
  const l0r = l0 * RAD;
  const eqt = 4 / RAD * (y * Math.sin(2 * l0r) - 2 * e * Math.sin(m) + 4 * e * y * Math.sin(m) * Math.cos(2 * l0r) - 0.5 * y * y * Math.sin(4 * l0r) - 1.25 * e * e * Math.sin(2 * m));
  const d = new Date(ms);
  const minutes = d.getUTCHours() * 60 + d.getUTCMinutes() + d.getUTCSeconds() / 60;
  const ha = ((minutes + eqt + 4 * lon) / 4 - 180) * RAD;
  const la = lat * RAD;
  const zen = Math.acos(Math.min(1, Math.max(-1, Math.sin(la) * Math.sin(dec) + Math.cos(la) * Math.cos(dec) * Math.cos(ha))));
  const az = Math.atan2(Math.sin(ha), Math.cos(ha) * Math.sin(la) - Math.tan(dec) * Math.cos(la)) / RAD + 180;
  return { az: ((az % 360) + 360) % 360, el: 90 - zen / RAD };
}

/** World direction (x east, y up, z south) toward azimuth/elevation in degrees. */
export function dirFromAzEl(az, el, out) {
  const a = az * RAD, e = el * RAD;
  return out.set(Math.sin(a) * Math.cos(e), Math.sin(e), -Math.cos(a) * Math.cos(e));
}

/** Daylight 0..1 from the sun's elevation (civil twilight in between). */
export const daylight = (el) => Math.min(1, Math.max(0, (el + 7) / 15));

/** Colour of sunlight by elevation: white at noon, gold low, red at the horizon, blue dusk below. */
export function sunColour(el, out) {
  if (el > 25) return out.setRGB(1.0, 0.96, 0.9);
  if (el > 8) { const k = (el - 8) / 17; return out.setRGB(1.0, 0.82 + 0.14 * k, 0.62 + 0.28 * k); }
  if (el > -1) { const k = (el + 1) / 9; return out.setRGB(1.0, 0.5 + 0.32 * k, 0.28 + 0.34 * k); }
  const k = Math.min(1, (-1 - el) / 6);
  return out.setRGB(1.0 - 0.45 * k, 0.5 - 0.1 * k, 0.28 + 0.4 * k);
}

const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const PDT = -7 * 3600000;
/** "Fri 3 Oct · 12:50" in Pacific Daylight Time. */
export function formatPDT(ms, withSeconds = false) {
  const d = new Date(ms + PDT);
  const hh = String(d.getUTCHours()).padStart(2, "0"), mm = String(d.getUTCMinutes()).padStart(2, "0"), ss = String(d.getUTCSeconds()).padStart(2, "0");
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} Oct · ${hh}:${mm}${withSeconds ? ":" + ss : ""}`;
}
/** "2026:10:09 16:44:13" (EXIF, local time) -> UTC ms. */
export const exifToMs = (s) => Date.parse(s.replace(/^(\d+):(\d+):(\d+) /, "$1-$2-$3T") + "-07:00");

/** Compass words for an azimuth. */
export function compass(az) {
  const names = ["north", "north-northeast", "northeast", "east-northeast", "east", "east-southeast", "southeast", "south-southeast",
    "south", "south-southwest", "southwest", "west-southwest", "west", "west-northwest", "northwest", "north-northwest"];
  return names[Math.round(az / 22.5) % 16];
}

/** Local equirectangular ENU used by the October pipeline (origin Portola Valley). */
export function makeGeo([lat0, lon0]) {
  return {
    toWorld(lat, lon) { return [(lon - lon0) * 111320 * Math.cos(((lat + lat0) / 2) * RAD), -(lat - lat0) * 110574]; },
    toGps(x, z) { const lat = lat0 - z / 110574; return [lat, lon0 + x / (111320 * Math.cos(((lat + lat0) / 2) * RAD))]; },
  };
}
