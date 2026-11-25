import { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";

/* ---------- Audio engine (Web Audio API, no dependencies) ---------- */

const curve = (fn: (x: number) => number) =>
  Float32Array.from({ length: 256 }, (_, i) => fn(i / 255));
/**
 * Fade curves where each track sits at `level` (0.5-0.9) at the midpoint.
 * 0.5 = linear, ~0.71 = equal-power, higher = more overlap in the middle.
 */
const fadeCurves = (level: number) => {
  const p = Math.log(level) / Math.log(0.5);
  return { inn: curve((x) => Math.pow(x, p)), out: curve((x) => Math.pow(1 - x, p)) };
};
type Cfg = { loop: number; level: number; repeat: boolean; trackVolume: number };

/** Per-track override of the global loop setting. No entry means "follow the global setting". */
type Mode = "once" | "loop";

/** A decoded track plus the range (in seconds) that is actually audible. */
type Loaded = { buf: AudioBuffer; lo: number; hi: number };

/** Finds where the sound really starts and ends, skipping silent lead-in and tail. */
const audibleRange = (buf: AudioBuffer, threshold: number): [number, number] => {
  const n = buf.length;
  let first = n;
  let last = -1;
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    let i = 0;
    while (i < n && Math.abs(d[i]) < threshold) i++;
    let j = n - 1;
    while (j > i && Math.abs(d[j]) < threshold) j--;
    if (i < first) first = i;
    if (j > last) last = j;
  }
  if (last <= first) return [0, buf.duration]; // entirely silent: leave it alone
  return [first / buf.sampleRate, (last + 1) / buf.sampleRate];
};

type Layer = { t: number; mid: number; offset: number };

/**
 * One track playing on loop. Each pass is its own buffer source, scheduled
 * slightly ahead so it starts `loopFade` seconds before the previous one ends
 * and the two are crossfaded. With loopFade = 0 the loop is simply gapless.
 * seek() quickly fades out whatever is playing and starts a new pass at the
 * requested position.
 */
class Voice {
  readonly duration: number;
  private master: GainNode;
  private nextStart: number;
  private nextOffset = 0;
  private nextIn = 0;
  private timer: number;
  private dead = false;
  private finished = false; // no more passes scheduled (looping off)
  private started = false; // a pass has been scheduled since the last (re)start
  private fader: GainNode; // used only for the fade-out
  private trim: GainNode; // this track's own volume multiplier
  private layers: Layer[] = [];
  private live = new Map<AudioBufferSourceNode, GainNode>();

  private ctx: AudioContext;
  private buf: AudioBuffer;
  private lo: number;
  private cfg: () => Cfg;

  constructor(ctx: AudioContext, track: Loaded, cfg: () => Cfg, out: AudioNode, fadeIn: number) {
    this.ctx = ctx;
    this.buf = track.buf;
    this.lo = track.lo;
    this.cfg = cfg;
    this.duration = track.hi - track.lo; // audible length only
    this.master = ctx.createGain();
    this.fader = ctx.createGain();
    this.trim = ctx.createGain();
    this.trim.gain.value = this.cfg().trackVolume;
    this.master.connect(this.fader);
    this.fader.connect(this.trim);
    this.trim.connect(out);
    const t = ctx.currentTime + 0.05;
    if (fadeIn > 0.01) {
      this.master.gain.value = 0;
      this.master.gain.setValueCurveAtTime(fadeCurves(this.cfg().level).inn, t, fadeIn);
    }
    this.nextStart = t;
    this.pump();
    this.timer = window.setInterval(() => this.pump(), 250);
  }

  private pump() {
    if (this.cfg().repeat) this.finished = false;
    else if (this.started) this.finished = true; // looping was switched off: no extra pass
    while (!this.finished && this.nextStart < this.ctx.currentTime + 3) this.spawn();
  }

  private spawn() {
    const { ctx, buf } = this;
    const t = this.nextStart;
    const offset = this.nextOffset;
    const inLen = this.nextIn;
    const remaining = this.duration - offset;
    const { loop, level, repeat } = this.cfg();
    const { inn, out } = fadeCurves(level);
    let L = repeat ? Math.min(loop, remaining / 2 - 0.01) : 0;
    if (L < 0.05) L = 0;

    const src = ctx.createBufferSource();
    src.buffer = buf;
    const g = ctx.createGain();
    src.connect(g);
    g.connect(this.master);
    if (inLen > 0) {
      g.gain.value = 0;
      g.gain.setValueCurveAtTime(inn, t, inLen);
    }
    if (L > 0) g.gain.setValueCurveAtTime(out, t + remaining - L, L);
    src.start(t, this.lo + offset, remaining);
    src.stop(t + remaining + 0.05);

    const layer: Layer = { t, mid: t + inLen / 2, offset };
    this.layers.push(layer);
    this.live.set(src, g);
    src.onended = () => {
      this.live.delete(src);
      this.layers = this.layers.filter((l) => l !== layer);
      g.disconnect();
    };

    this.nextStart = t + remaining - L;
    this.nextOffset = 0;
    this.nextIn = L;
    this.started = true;
    if (!repeat) this.finished = true;
  }

  /** Change this track's volume multiplier (1 = unchanged), smoothed to avoid clicks. */
  setVolume(v: number) {
    this.trim.gain.setTargetAtTime(v, this.ctx.currentTime, 0.02);
  }

  get context() {
    return this.ctx;
  }

  /** True once a non-looping track has played to its end. */
  isDone() {
    return this.started && this.finished && this.live.size === 0;
  }

  /** Current position within the track, in seconds. */
  position() {
    const now = this.ctx.currentTime;
    let cur = this.layers[0];
    for (const l of this.layers) if (l.mid <= now) cur = l;
    if (!cur) return 0;
    return Math.min(cur.offset + Math.max(0, now - cur.t), this.duration);
  }

  seek(pos: number) {
    if (this.dead) return;
    const now = this.ctx.currentTime;
    for (const [src, g] of this.live) {
      try {
        g.gain.cancelScheduledValues(now);
        g.gain.setValueAtTime(g.gain.value, now);
        g.gain.linearRampToValueAtTime(0, now + 0.04);
      } catch {
        /* seeking mid-crossfade: the stop below still applies */
      }
      try {
        src.stop(now + 0.05);
      } catch {
        /* already stopped */
      }
    }
    this.layers = [];
    this.nextStart = now + 0.02;
    this.nextOffset = Math.min(Math.max(pos, 0), this.duration - 0.1);
    this.nextIn = 0.04;
    this.finished = false;
    this.started = false;
    this.pump();
  }

  fadeOut(seconds: number) {
    if (this.dead) return;
    this.dead = true;
    window.clearInterval(this.timer); // stop scheduling new passes
    const now = this.ctx.currentTime;
    const dur = Math.max(seconds, 0.05);
    // Fade on its own node so it can never clash with a fade-in that is still running.
    this.fader.gain.setValueCurveAtTime(fadeCurves(this.cfg().level).out, now + 0.05, dur);
    window.setTimeout(() => this.dispose(), dur * 1000 + 300);
  }

  private dispose() {
    window.clearInterval(this.timer);
    for (const src of this.live.keys()) {
      try {
        src.stop();
      } catch {
        /* already stopped */
      }
    }
    this.master.disconnect();
    this.fader.disconnect();
    this.trim.disconnect();
  }
}

/* ---------- UI ---------- */

const AUDIO_EXT = /\.(mp3|wav|ogg|oga|flac|m4a|aac|opus|weba)$/i;
const byName = (a: File, b: File) =>
  a.name.localeCompare(b.name, undefined, { numeric: true });
// Stable unique id per File object, used as the list key (names/sizes can collide).
const ids = new WeakMap<File, number>();
let nextId = 0;
const idOf = (f: File) => {
  let id = ids.get(f);
  if (id === undefined) ids.set(f, (id = ++nextId));
  return id;
};
const sig = (f: File) => `${f.name}|${f.size}|${f.lastModified}`;
const dbg = (...a: unknown[]) => console.log("[player]", ...a); // diagnostics for track switching
const why = (e: unknown) =>
  e instanceof Error ? (e.name === "Error" ? e.message : `${e.name}: ${e.message}`) : String(e);
const DEFAULT_TRACK_VOL = 70; // percent; leaves headroom to raise a quiet track
const VOL_KEY = "crossfade-player:trackVolumesV2"; // V2: scale is now 0-100 with a default of 70
// Per-track volume multipliers (percent), remembered by file name + size + modified date.
const loadVolumes = (): Record<string, number> => {
  const out: Record<string, number> = {};
  try {
    const raw = localStorage.getItem(VOL_KEY);
    const o: unknown = raw ? JSON.parse(raw) : null;
    if (o && typeof o === "object") {
      for (const [k, v] of Object.entries(o)) {
        if (typeof v === "number" && Number.isFinite(v)) out[k] = Math.min(100, Math.max(0, v));
      }
    }
  } catch {
    /* storage unavailable or corrupt: start fresh */
  }
  return out;
};
const title = (f: File) => f.name.replace(/\.[^.]+$/, "");
const fmt = (s: number) => {
  const r = Math.floor(s % 60);
  return `${Math.floor(s / 60)}:${r < 10 ? "0" : ""}${r}`;
};
const volGain = (v: number) => (v / 100) ** 2; // squared so the slider feels even
const fill = (v: number, lo: number, hi: number) =>
  ({ "--p": `${((v - lo) / (hi - lo || 1)) * 100}%` }) as CSSProperties;

const Icon = ({ d, size = 20 }: { d: string; size?: number }) => (
  <svg viewBox="0 0 24 24" width={size} height={size} fill="currentColor" aria-hidden="true">
    <path d={d} />
  </svg>
);
const PREV = "M6 6h2v12H6zM20 6v12L9.5 12z";
const NEXT = "M16 6h2v12h-2zM4 6v12l10.5-6z";
const PLAY = "M8 5v14l11-7z";
const PAUSE = "M6 5h4v14H6zM14 5h4v14h-4z";
const LOOP = "M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z"; // two arrows chasing each other
const VOL_UP = "M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z";
const VOL_MUTE = "M7 9v6h4l5 5V4l-5 5H7z";
const ONCE = "M3 11h11v2H3zM14 8l4 4-4 4zM20 6h2v12h-2z"; // arrow running into an end stop
const REPEAT = "M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4zm-4-2V9h-1l-2 1v1h1.5v4H13z";

const css = `
:root{--bg:#f5f6f8;--surface:#fff;--text:#1b1d22;--muted:#6b7280;--line:#e3e5ea;--track:#d9dce3;--accent:#4756d6;--on-accent:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--surface:#1d2027;--text:#e8eaef;--muted:#8c93a1;--line:#2a2e37;--track:#394050;--accent:#8b97ff;--on-accent:#14161b}}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;display:block;min-height:0}
.app{box-sizing:border-box;width:100%;max-width:960px;margin:0 auto;padding:40px 16px;text-align:left}
.layout{display:grid;grid-template-columns:minmax(0,420px) minmax(0,1fr);gap:32px;align-items:start}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:24px 20px}
.title{font-size:20px;font-weight:600;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-height:28px}
.sub.error{color:#d6453d}
.sub{margin:4px 0 20px;color:var(--muted);font-size:14px;min-height:20px}
.seek{display:grid;grid-template-columns:auto 1fr auto;gap:12px;align-items:center;font-size:13px;color:var(--muted);font-variant-numeric:tabular-nums}
.transport{display:flex;justify-content:center;align-items:center;gap:16px;margin:20px 0 24px}
.btn{display:grid;place-items:center;width:44px;height:44px;padding:0;border-radius:50%;border:1px solid var(--line);background:var(--surface);color:var(--text);cursor:pointer}
.btn:hover:not(:disabled){border-color:var(--muted)}
.btn:disabled{opacity:.4;cursor:default}
.transport{position:relative}
.btn.toggle{position:absolute;right:0;color:var(--muted)}
.btn.toggle[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.btn.play{width:60px;height:60px;background:var(--accent);border-color:var(--accent);color:var(--on-accent)}
.btn.play svg{width:26px;height:26px}
.setting{display:block;margin-top:14px;font-size:14px}
.hint{margin:2px 0 0;font-size:12px;color:var(--muted)}
.setting span{display:flex;justify-content:space-between;color:var(--muted);margin-bottom:2px}
.drop{display:block;margin:20px 0 8px;padding:18px;border:1.5px dashed var(--track);border-radius:12px;text-align:center;color:var(--muted);font-size:14px;cursor:pointer}
.drop:hover,.drop.over{border-color:var(--accent);color:var(--text)}
.drop input{display:none}
.listhead{display:flex;justify-content:space-between;align-items:center;margin:0 0 8px 12px;color:var(--muted);font-size:13px}
.seg{display:flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{border:0;background:none;color:var(--muted);font:inherit;padding:4px 10px;cursor:pointer}
.seg button[aria-pressed=true]{background:var(--accent);color:var(--on-accent)}
.list{list-style:none;margin:0;padding:0;max-height:calc(100vh - 130px);overflow-y:auto}
@media (max-width:760px){.layout{grid-template-columns:minmax(0,1fr);gap:16px}.list{max-height:none}}
.row{display:flex;flex-wrap:wrap;align-items:center;border-radius:0 6px 6px 0}
.row.dragging{opacity:.4}
.row[data-drop=before]{box-shadow:inset 0 2px 0 var(--accent)}
.row[data-drop=after]{box-shadow:inset 0 -2px 0 var(--accent)}
.row .item{flex:1;min-width:0;width:auto}
.grip{cursor:grab;padding:0 6px;color:var(--muted);user-select:none;font-size:14px;line-height:1}
.mode{flex:none;display:grid;place-items:center;width:28px;height:28px;margin:0 6px;padding:0;border:1px solid transparent;border-radius:6px;background:none;color:var(--muted);cursor:pointer}
.mode:hover{border-color:var(--line)}
.mode[data-changed=true]{color:var(--on-accent);background:var(--accent);border-color:var(--accent)}
.volpanel{flex-basis:100%;display:flex;align-items:center;gap:10px;padding:2px 12px 8px 33px;color:var(--muted);font-size:13px}
.volpanel input{flex:1}
.volval{min-width:40px;text-align:right;font-variant-numeric:tabular-nums}
.volreset{border:1px solid var(--line);background:none;color:var(--muted);font:inherit;font-size:12px;border-radius:6px;padding:2px 8px;cursor:pointer}
.volreset:disabled{opacity:.4;cursor:default}
.mode[data-mode=once],.mode[data-mode=loop]{color:var(--on-accent);background:var(--accent);border-color:var(--accent)}
.item{display:block;width:100%;box-sizing:border-box;padding:9px 12px;border:0;border-left:3px solid transparent;border-radius:0 6px 6px 0;background:none;color:var(--muted);font:inherit;text-align:left;cursor:pointer;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.item:hover{background:var(--surface)}
.item[aria-current=true]{color:var(--text);font-weight:600;border-left-color:var(--accent);background:var(--surface)}
:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
input[type=range]{-webkit-appearance:none;appearance:none;width:100%;height:20px;margin:0;background:transparent;cursor:pointer}
input[type=range]:disabled{opacity:.4;cursor:default}
input[type=range]::-webkit-slider-runnable-track{height:4px;border-radius:2px;background:linear-gradient(to right,var(--accent) var(--p,0%),var(--track) var(--p,0%))}
input[type=range]::-moz-range-track{height:4px;border-radius:2px;background:var(--track)}
input[type=range]::-moz-range-progress{height:4px;border-radius:2px;background:var(--accent)}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:14px;height:14px;margin-top:-5px;border:0;border-radius:50%;background:var(--accent)}
input[type=range]::-moz-range-thumb{width:14px;height:14px;border:0;border-radius:50%;background:var(--accent)}
`;

/**
 * A twenty-second WAV that is practically inaudible (peaks of 24 out of 32767, about -63 dB).
 * Browsers only hook up media keys for real media elements, and Firefox only counts an element as
 * audible (and so controllable) if it is louder than about -72 dB.
 */
// Peak level of the helper tone, in 16-bit units (24 is about -63 dB). Firefox only treats it as audible above
// roughly 9 (-72 dB), and testing showed it working reliably from about 10 up. Can be overridden for testing with
// localStorage.setItem("crossfade-player:anchorLevel", "300") followed by a reload.
const anchorLevel = () => {
  try {
    const v = Number(localStorage.getItem("crossfade-player:anchorLevel"));
    if (Number.isFinite(v) && v >= 1 && v <= 3000) return v;
  } catch {
    /* storage unavailable */
  }
  return 24;
};

const silentWavUrl = () => {
  const level = anchorLevel();
  const rate = 8000;
  const n = rate * 20; // long enough not to be mistaken for a short sound effect
  const buf = new ArrayBuffer(44 + n * 2);
  const v = new DataView(buf);
  const str = (o: number, t: string) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF");
  v.setUint32(4, 36 + n * 2, true);
  str(8, "WAVEfmt ");
  v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); // PCM
  v.setUint16(22, 1, true); // mono
  v.setUint32(24, rate, true);
  v.setUint32(28, rate * 2, true);
  v.setUint16(32, 2, true);
  v.setUint16(34, 16, true);
  str(36, "data");
  v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, Math.round(level * Math.sin((2 * Math.PI * 100 * i) / rate)), true);
  // A data: URL (rather than a blob: URL) can't be revoked or expire while the browser is still loading it.
  const bytes = new Uint8Array(buf);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return "data:audio/wav;base64," + btoa(bin);
};

/**
 * Fallback anchor that needs no audio file: a quiet oscillator (about -64 dB) in its own
 * AudioContext, streamed into an <audio> element.
 */
const streamAnchor = () => {
  const ctx = new AudioContext();
  const osc = ctx.createOscillator();
  osc.frequency.value = 100;
  const gain = ctx.createGain();
  gain.gain.value = 0.0006;
  const dest = ctx.createMediaStreamDestination();
  osc.connect(gain).connect(dest);
  osc.start();
  const a = new Audio();
  a.srcObject = dest.stream;
  return { a, ctx };
};

/** Like useState, but remembered in localStorage (and range-checked when loaded). */
function useStored<T extends number | boolean>(key: string, initial: T, range?: [number, number]) {
  const name = "crossfade-player:" + key;
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(name);
      if (raw !== null) {
        const v: unknown = JSON.parse(raw);
        if (typeof v === "boolean" && typeof initial === "boolean") return v as T;
        if (typeof v === "number" && typeof initial === "number" && Number.isFinite(v)) {
          return (range ? Math.min(Math.max(v, range[0]), range[1]) : v) as T;
        }
      }
    } catch {
      /* storage unavailable or corrupt: fall back to the default */
    }
    return initial;
  });
  useEffect(() => {
    try {
      localStorage.setItem(name, JSON.stringify(value));
    } catch {
      /* storage full or blocked: settings just won't persist */
    }
  }, [name, value]);
  return [value, setValue] as const;
}

export default function App() {
  const [tracks, setTracks] = useState<File[]>([]);
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [nowPlaying, setNowPlaying] = useState<File | null>(null); // the track actually sounding
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [fade, setFade] = useStored<number>("fade", 4, [0, 12]);
  const [loopFade, setLoopFade] = useStored<number>("loopFade", 3, [0, 12]);
  const [cross, setCross] = useStored<number>("cross", 71, [50, 90]);
  const [volume, setVolume] = useStored<number>("volume", 100, [0, 100]);
  const [repeat, setRepeat] = useStored<boolean>("repeat", true);
  const [silence, setSilence] = useStored<number>("silence", -54, [-90, -20]);
  const [alpha, setAlpha] = useStored<boolean>("alphaSort", true);
  const [modes, setModes] = useState<Map<File, Mode>>(() => new Map());
  const [drag, setDrag] = useState<{ from: File; over: File | null; after: boolean } | null>(null);
  const [volumes, setVolumes] = useState<Record<string, number>>(loadVolumes);
  const [volOpen, setVolOpen] = useState<File | null>(null); // track whose volume slider is expanded
  const [pos, setPos] = useState(0);
  const [dur, setDur] = useState(0);
  const [scrub, setScrub] = useState<number | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const ctxRef = useRef<AudioContext | null>(null);
  const outRef = useRef<GainNode | null>(null);
  const voiceRef = useRef<Voice | null>(null);
  const cache = useRef(new Map<File, Promise<AudioBuffer>>());
  const token = useRef(0);

  // Mirror state into refs so callbacks always see current values.
  const tracksRef = useRef(tracks);
  const indexRef = useRef(index);
  const fadeRef = useRef(fade);
  const loopFadeRef = useRef(loopFade);
  tracksRef.current = tracks;
  indexRef.current = index;
  fadeRef.current = fade;
  loopFadeRef.current = loopFade;
  const crossRef = useRef(cross);
  crossRef.current = cross;
  const volumeRef = useRef(volume);
  volumeRef.current = volume;
  const repeatRef = useRef(repeat);
  repeatRef.current = repeat;
  const modesRef = useRef(modes);
  modesRef.current = modes;
  // Whether a given track loops: its own setting if it has one, otherwise the global toggle.
  const volumesRef = useRef(volumes);
  volumesRef.current = volumes;
  // Same curve as the master slider, so the two combine as expected (70% x 70% sounds like ~49%).
  const volFor = (f: File) => volGain(volumesRef.current[sig(f)] ?? DEFAULT_TRACK_VOL);
  const repeatFor = (f: File) => {
    const m = modesRef.current.get(f);
    return m === undefined ? repeatRef.current : m === "loop";
  };
  const silenceRef = useRef(silence);
  silenceRef.current = silence;
  const alphaRef = useRef(alpha);
  alphaRef.current = alpha;
  const addedRef = useRef<File[]>([]); // files in the order they were added
  const advancing = useRef(false);
  const loadingRef = useRef(false); // a user-requested switch is in progress
  const nowRef = useRef<File | null>(null); // the track actually playing
  const anchorCtxRef = useRef<AudioContext | null>(null); // only used by the stream-based fallback anchor
  const anchorRef = useRef<HTMLAudioElement | null>(null); // near-silent element that enables media keys
  const staleCtx = useRef<AudioContext | null>(null); // replaced context still used by the fading-out track

  const startAnchor = () => {
    void anchorCtxRef.current?.resume();
    return anchorRef.current?.play();
  };

  const getCtx = () => {
    if (!ctxRef.current) {
      const ctx = new AudioContext();
      const out = ctx.createGain(); // master volume
      out.gain.value = volGain(volumeRef.current);
      out.connect(ctx.destination);
      ctxRef.current = ctx;
      outRef.current = out;
    }
    return ctxRef.current;
  };

  // Abandon a misbehaving AudioContext; the playing track keeps using it until it fades out.
  const resetCtx = () => {
    dbg("rebuilding the audio context");
    staleCtx.current ??= ctxRef.current;
    ctxRef.current = null;
    outRef.current = null;
  };

  // Read + decode a file. Failures are retried once, rebuilding the audio context for decode errors.
  const decodeFile = async (file: File): Promise<AudioBuffer> => {
    let bytes: ArrayBuffer;
    try {
      bytes = await file.arrayBuffer();
    } catch {
      await new Promise((r) => setTimeout(r, 200));
      try {
        bytes = await file.arrayBuffer();
      } catch (e) {
        throw new Error(`couldn’t read the file (${why(e)})`);
      }
    }
    try {
      return await getCtx().decodeAudioData(bytes.slice(0)); // slice: decoding detaches its input
    } catch (e) {
      dbg("decode failed; retrying with a fresh audio context", file.name, why(e));
      for (const f of [...cache.current.keys()]) if (f !== file && f !== nowRef.current) cache.current.delete(f);
      resetCtx();
      try {
        return await getCtx().decodeAudioData(bytes.slice(0));
      } catch (e2) {
        throw new Error(`couldn’t decode the file (${why(e2)})`);
      }
    }
  };

  const load = (file: File) => {
    let p = cache.current.get(file);
    if (!p) {
      p = decodeFile(file);
      cache.current.set(file, p);
      p.catch(() => cache.current.delete(file));
    }
    return p;
  };

  const go = async (target: number, fadeOverride?: number) => {
    const list = tracksRef.current;
    if (!list.length) return;
    const i = (target + list.length) % list.length;
    const f = fadeOverride ?? fadeRef.current;
    const my = ++token.current;
    loadingRef.current = true;
    // Start the anchor element right away, while the click still counts as a user gesture (autoplay rules).
    startAnchor()?.then(
      () => dbg("media-key anchor is playing"),
      (e) => dbg("media-key anchor was blocked:", why(e))
    );
    dbg("switch requested", { to: list[i].name, attempt: my, fade: f });
    setIndex(i);
    setBusy(true);
    try {
      const buf = await load(list[i]);
      dbg("decoded", { attempt: my, seconds: Math.round(buf.duration) });
      if (my !== token.current) {
        dbg("abandoned: a newer request replaced attempt", my, "with", token.current);
        return;
      }
      const ctx = getCtx(); // fetched after loading, in case the context was rebuilt
      const [lo, hi] = audibleRange(buf, 10 ** (silenceRef.current / 20)); // dB -> amplitude
      const track: Loaded = { buf, lo, hi };
      if (ctx.state === "suspended") await ctx.resume();
      const old = voiceRef.current;
      // Build the new voice first, so a failure can't leave the old one half-faded out.
      const voice = new Voice(
        ctx,
        track,
        () => ({ loop: loopFadeRef.current, level: crossRef.current / 100, repeat: repeatFor(list[i]), trackVolume: volFor(list[i]) }),
        outRef.current!,
        old ? Math.max(f, 0.03) : 0.3
      );
      old?.fadeOut(f);
      voiceRef.current = voice;
      const stale = staleCtx.current;
      if (stale) {
        staleCtx.current = null; // close the abandoned context once the old track has faded out
        window.setTimeout(() => stale.close().catch(() => {}), f * 1000 + 1500);
      }
      dbg("now playing", { attempt: my, name: list[i].name });
      setPlaying(true);

      setNowPlaying(list[i]);
      nowRef.current = list[i];
      setError(null);

      // Keep the previous, current and next tracks decoded so Next/Previous are instant.
      const next = list[(i + 1) % list.length];
      const prev = list[(i - 1 + list.length) % list.length];
      const keep = new Set([list[i], next, prev]);
      for (const f of [...cache.current.keys()]) if (!keep.has(f)) cache.current.delete(f);
      load(next)
        .catch(() => {})
        .then(() => load(prev))
        .catch(() => {});
    } catch (e) {
      console.error("Could not play", list[i]?.name, e);
      if (!voiceRef.current) anchorRef.current?.pause();
      if (my === token.current && list[i]) {
        setError(`Couldn’t play “${title(list[i])}” – ${why(e)}.`);
      }
    } finally {
      if (my === token.current) {
        setBusy(false);
        loadingRef.current = false;
      }
    }
  };

  const nowIndex = () => (nowRef.current ? tracksRef.current.indexOf(nowRef.current) : indexRef.current);

  const setPaused = async (pause: boolean) => {
    const ctx = voiceRef.current?.context ?? ctxRef.current;
    if (!voiceRef.current || !ctx) {
      if (!pause) await go(indexRef.current);
      return;
    }
    if (pause && ctx.state === "running") {
      await ctx.suspend();
      setPlaying(false);
    } else if (!pause && ctx.state !== "running") {
      await ctx.resume();
      setPlaying(true);
    }
  };

  const toggle = () => {
    const ctx = voiceRef.current?.context ?? ctxRef.current;
    return setPaused(ctx?.state === "running");
  };

  const commitSeek = () => {
    if (scrub === null) return;
    voiceRef.current?.seek(scrub);
    setPos(scrub);
    setScrub(null);
  };

  // Rebuild the playlist in the chosen order, keeping the current track selected.
  const applyOrder = (alphabetical: boolean, current?: File) => {
    const next = alphabetical ? [...addedRef.current].sort(byName) : [...addedRef.current];
    tracksRef.current = next;
    setTracks(next);
    if (!current) return;
    const i = next.indexOf(current);
    setIndex(i);
    if (voiceRef.current && next.length > 1) load(next[(i + 1) % next.length]).catch(() => {}); // preload new "next"
  };

  const addFiles = (files: File[]) => {
    const audio = files.filter((f) => f.type.startsWith("audio/") || AUDIO_EXT.test(f.name));
    // Ignore files that are already in the playlist (or repeated within this batch).
    const known = new Set(addedRef.current.map(sig));
    const fresh = audio.filter((f) => !known.has(sig(f)) && known.add(sig(f)));
    if (!fresh.length) return;
    const current = tracksRef.current[indexRef.current];
    addedRef.current = [...addedRef.current, ...fresh];
    applyOrder(alphaRef.current, current);
  };

  const toggleSort = (alphabetical: boolean) => {
    if (alphabetical === alpha) return;
    setAlpha(alphabetical);
    applyOrder(alphabetical, tracksRef.current[indexRef.current]);
  };

  // Move `file` next to `target` in the custom order (only used in Custom mode).
  const moveTrack = (file: File, target: File, after: boolean) => {
    if (file === target) return;
    const list = addedRef.current.filter((f) => f !== file);
    const at = list.indexOf(target);
    if (at < 0) return;
    list.splice(after ? at + 1 : at, 0, file);
    addedRef.current = list;
    applyOrder(false, tracksRef.current[indexRef.current]);
  };

  const setTrackVolume = (f: File, pct: number) => {
    setVolumes((prev) => {
      const next = { ...prev };
      if (pct === DEFAULT_TRACK_VOL) delete next[sig(f)];
      else next[sig(f)] = pct;
      return next;
    });
    if (f === nowRef.current) voiceRef.current?.setVolume(volGain(pct)); // takes effect immediately
  };

  // Cycle a track's loop setting: follow global -> always once -> always loop -> follow global.
  const cycleMode = (f: File) =>
    setModes((prev) => {
      const next = new Map(prev);
      const cur = prev.get(f);
      if (cur === undefined) next.set(f, "once");
      else if (cur === "once") next.set(f, "loop");
      else next.delete(f);
      return next;
    });

  // Drop files anywhere on the page.
  useEffect(() => {
    let depth = 0;
    const isFiles = (e: DragEvent) => e.dataTransfer?.types.includes("Files");
    const enter = (e: DragEvent) => {
      if (isFiles(e)) setDragOver(++depth > 0);
    };
    const leave = (e: DragEvent) => {
      if (isFiles(e)) setDragOver(--depth > 0);
    };
    const over = (e: DragEvent) => e.preventDefault();
    const drop = (e: DragEvent) => {
      e.preventDefault();
      depth = 0;
      setDragOver(false);
      addFiles(Array.from(e.dataTransfer?.files ?? []));
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragleave", leave);
    window.addEventListener("dragover", over);
    window.addEventListener("drop", drop);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("dragover", over);
      window.removeEventListener("drop", drop);
    };
  }, []);

  // Poll playback position for the progress bar.
  useEffect(() => {
    const id = window.setInterval(() => {
      const v = voiceRef.current;
      setPos(v ? v.position() : 0);
      setDur(v ? v.duration : 0);

      // With looping off, crossfade into the next track as this one nears its end.
      const cur = nowRef.current;
      if (v && cur && !repeatFor(cur) && !advancing.current && !loadingRef.current && v.context.state === "running") {
        const done = v.isDone(); // fallback if the poll missed the end (e.g. throttled tab)
        const left = done ? 0 : v.duration - v.position();
        // Never fade for longer than half the track; keep a small lead even at 0 so the hand-off is seamless.
        const lead = Math.max(Math.min(fadeRef.current, v.duration / 2), 0.25);
        if (done || left <= lead) {
          advancing.current = true;
          // Advance from the track that is really playing, not from whatever was last clicked.
          const at = nowRef.current ? tracksRef.current.indexOf(nowRef.current) : indexRef.current;
          go(at + 1, Math.max(0.1, left - 0.1)).finally(() => {
            window.setTimeout(() => (advancing.current = false), 2000);
          });
        }
      }
    }, 100);
    return () => window.clearInterval(id);
  }, []);

  // Media keys / lock-screen controls. A looping, practically silent <audio> element anchors the session.
  useEffect(() => {
    const els: HTMLAudioElement[] = [];
    const fallback: { ctx: AudioContext | null } = { ctx: null };
    const mount = (a: HTMLAudioElement) => {
      a.hidden = true;
      document.body.appendChild(a); // attached to the page, which some browsers expect for media controls
      els.push(a);
      anchorRef.current = a;
    };
    const wav = new Audio(silentWavUrl());
    wav.loop = true;
    wav.addEventListener("error", () => {
      dbg("media-key anchor failed to load:", wav.error?.code, wav.error?.message, "- using a stream-based one instead");
      if (fallback.ctx) return;
      const s = streamAnchor();
      fallback.ctx = s.ctx;
      anchorCtxRef.current = s.ctx;
      mount(s.a);
      if (voiceRef.current?.context.state === "running") startAnchor()?.catch(() => {});
    });
    mount(wav);
    return () => {
      for (const a of els) {
        a.pause();
        a.remove();
      }
      void fallback.ctx?.close();
      anchorRef.current = null;
      anchorCtxRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!("mediaSession" in navigator)) return;
    const ms = navigator.mediaSession;
    // Drop keyboard focus first: a media key counts as keyboard input, which would otherwise
    // make the browser draw a focus ring around whichever button was last clicked.
    const fromKeys = (fn: () => unknown) => () => {
      (document.activeElement as HTMLElement | null)?.blur();
      void fn();
    };
    const handlers: [MediaSessionAction, MediaSessionActionHandler][] = [
      ["play", fromKeys(() => setPaused(false))],
      ["pause", fromKeys(() => setPaused(true))],
      ["previoustrack", fromKeys(() => go(nowIndex() - 1))],
      ["nexttrack", fromKeys(() => go(nowIndex() + 1))],
    ];
    for (const [action, handler] of handlers) {
      try {
        ms.setActionHandler(action, handler);
      } catch {
        /* action not supported by this browser */
      }
    }
    return () => {
      for (const [action] of handlers) {
        try {
          ms.setActionHandler(action, null);
        } catch {
          /* ignore */
        }
      }
    };
  }, []);

  useEffect(() => {
    const a = anchorRef.current;
    if (a) {
      if (playing) startAnchor()?.catch((e) => dbg("media-key anchor was blocked:", why(e)));
      else a.pause();
    }
    if ("mediaSession" in navigator) navigator.mediaSession.playbackState = playing ? "playing" : "paused";
  }, [playing]);

  useEffect(() => {
    if (!("mediaSession" in navigator) || !nowPlaying) return;
    navigator.mediaSession.metadata = new MediaMetadata({ title: title(nowPlaying), artist: "Crossfade Player" });
  }, [nowPlaying]);

  // Remember per-track volumes; Escape closes an open volume slider.
  useEffect(() => {
    try {
      localStorage.setItem(VOL_KEY, JSON.stringify(volumes));
    } catch {
      /* storage full or blocked */
    }
  }, [volumes]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setVolOpen(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Apply the volume slider (smoothed to avoid zipper noise).
  useEffect(() => {
    const ctx = ctxRef.current;
    const out = outRef.current;
    if (ctx && out) out.gain.setTargetAtTime(volGain(volume), ctx.currentTime, 0.02);
  }, [volume]);

  const current = tracks[index];
  const shownTrack = nowPlaying ?? current; // what's really playing, not just what was clicked
  const shown = scrub ?? pos;

  return (
    <>
      <style>{css}</style>
      <main className="app">
       <div className="layout">
        <div>
        <section className="panel">
          <h1 className="title">{shownTrack ? title(shownTrack) : "No track loaded"}</h1>
          <p className={`sub${error && !busy ? " error" : ""}`}>
            {busy
              ? `Loading “${current ? title(current) : ""}”…`
              : error
                ? error
                : shownTrack
                  ? `Track ${tracks.indexOf(shownTrack) + 1} of ${tracks.length}`
                  : "Add some audio files to start"}
          </p>

          <div className="seek">
            <span>{fmt(shown)}</span>
            <input
              type="range"
              aria-label="Seek"
              min={0}
              max={dur || 1}
              step={0.1}
              value={shown}
              disabled={!dur}
              style={fill(shown, 0, dur || 1)}
              onChange={(e) => setScrub(+e.target.value)}
              onPointerUp={commitSeek}
              onKeyUp={commitSeek}
              onBlur={commitSeek}
            />
            <span>{fmt(dur)}</span>
          </div>

          <div className="transport">
            <button className="btn" aria-label="Previous track" onClick={() => go(index - 1)} disabled={!tracks.length}>
              <Icon d={PREV} />
            </button>
            <button className="btn play" aria-label={playing ? "Pause" : "Play"} onClick={toggle} disabled={!tracks.length}>
              <Icon d={playing ? PAUSE : PLAY} />
            </button>
            <button className="btn" aria-label="Next track" onClick={() => go(index + 1)} disabled={!tracks.length}>
              <Icon d={NEXT} />
            </button>
            <button className="btn toggle" aria-label="Loop tracks by default" aria-pressed={repeat}
              title="Loop tracks until you skip (individual tracks can override this in the playlist)"
              onClick={() => setRepeat(!repeat)}>
              <Icon d={REPEAT} />
            </button>
          </div>

          <div className="seek">
            <span>Volume</span>
            <input type="range" aria-label="Volume" min={0} max={100} step={1} value={volume}
              style={fill(volume, 0, 100)} onChange={(e) => setVolume(+e.target.value)} />
            <span>{volume}%</span>
          </div>

          <label className="setting">
            <span>Crossfade to next track <b>{fade === 0 ? "off" : `${fade}s`}</b></span>
            <input type="range" min={0} max={12} step={0.5} value={fade} style={fill(fade, 0, 12)}
              onChange={(e) => setFade(+e.target.value)} />
          </label>
          <label className="setting">
            <span>Crossfade when looping <b>{loopFade === 0 ? "off" : `${loopFade}s`}</b></span>
            <input type="range" min={0} max={12} step={0.5} value={loopFade} style={fill(loopFade, 0, 12)}
              onChange={(e) => setLoopFade(+e.target.value)} />
          </label>
          <label className="setting">
            <span>Crossover volume <b>{cross}%</b></span>
            <input type="range" min={50} max={90} step={1} value={cross} style={fill(cross, 50, 90)}
              onChange={(e) => setCross(+e.target.value)} />
            <p className="hint">Each track's volume at the midpoint. 50% is linear, 71% is equal-power.</p>
          </label>
          <label className="setting">
            <span>Silence threshold <b>{silence} dB</b></span>
            <input type="range" min={-90} max={-20} step={1} value={silence} style={fill(silence, -90, -20)}
              onChange={(e) => setSilence(+e.target.value)} />
            <p className="hint">Anything quieter than this at the start or end of a track is skipped. Applies from the next track.</p>
          </label>
        </section>

        <label className={`drop${dragOver ? " over" : ""}`}>
          {dragOver ? "Drop to add" : "Choose audio files, or drop them anywhere on the page"}
          <input type="file" multiple accept="audio/*"
            onChange={(e) => { addFiles(Array.from(e.target.files ?? [])); e.target.value = ""; }} />
        </label>
        </div>

        <div>
        {tracks.length > 0 && (
          <div className="listhead">
            <span>Playlist order</span>
            <div className="seg">
              <button aria-pressed={alpha} onClick={() => toggleSort(true)}>A–Z</button>
              <button aria-pressed={!alpha} onClick={() => toggleSort(false)}>Custom</button>
            </div>
          </div>
        )}
        <ol className="list">
          {tracks.map((f, i) => {
            const mode = modes.get(f);
            const pct = volumes[sig(f)] ?? DEFAULT_TRACK_VOL;
            const volHelp = `Track volume: ${pct}%${pct === DEFAULT_TRACK_VOL ? " (default)" : ""}. Click to adjust.`;
            const dropMark =
              drag && drag.over === f && drag.from !== f ? (drag.after ? "after" : "before") : undefined;
            const modeHelp =
              mode === undefined
                ? `Follows the global loop setting (currently ${repeat ? "loop" : "play once"}). Click to force play once.`
                : mode === "once"
                  ? "Always plays once. Click to force looping."
                  : "Always loops. Click to follow the global setting.";
            return (
              <li
                key={idOf(f)}
                className={`row${drag?.from === f ? " dragging" : ""}`}
                data-drop={dropMark}
                onDragOver={(e) => {
                  if (!drag) return; // ignore file drags from outside the page
                  e.preventDefault();
                  const r = e.currentTarget.getBoundingClientRect();
                  const after = e.clientY > r.top + r.height / 2;
                  if (drag.over !== f || drag.after !== after) setDrag({ ...drag, over: f, after });
                }}
                onDrop={(e) => {
                  if (!drag) return;
                  e.preventDefault();
                  e.stopPropagation();
                  moveTrack(drag.from, f, drag.after);
                  setDrag(null);
                }}
              >
                {!alpha && (
                  <span
                    className="grip"
                    draggable
                    title="Drag to reorder"
                    aria-hidden="true"
                    onDragStart={(e) => {
                      e.dataTransfer.effectAllowed = "move";
                      e.dataTransfer.setData("text/plain", title(f)); // Firefox needs data to start a drag
                      const row = e.currentTarget.closest("li");
                      if (row) e.dataTransfer.setDragImage(row, 16, 16);
                      setDrag({ from: f, over: null, after: false });
                    }}
                    onDragEnd={() => setDrag(null)}
                  >
                    ⋮⋮
                  </span>
                )}
                <button className="item" aria-current={f === shownTrack} onClick={() => go(i)}>
                  {title(f)}
                  {busy && i === index && f !== shownTrack ? " …" : ""}
                </button>
                <button className="mode" data-changed={pct !== DEFAULT_TRACK_VOL} title={volHelp} aria-label={volHelp}
                  aria-expanded={volOpen === f} onClick={() => setVolOpen(volOpen === f ? null : f)}>
                  <Icon d={pct === 0 ? VOL_MUTE : VOL_UP} size={16} />
                </button>
                <button className="mode" data-mode={mode ?? "auto"} title={modeHelp} aria-label={modeHelp}
                  onClick={() => cycleMode(f)}>
                  <Icon d={(mode === undefined ? repeat : mode === "loop") ? LOOP : ONCE} size={16} />
                </button>
                {volOpen === f && (
                  <div className="volpanel">
                    <input type="range" aria-label={`Volume for ${title(f)}`} min={0} max={100} step={5}
                      value={pct} style={fill(pct, 0, 100)}
                      onChange={(e) => setTrackVolume(f, +e.target.value)} />
                    <span className="volval">{pct}%</span>
                    <button className="volreset" disabled={pct === DEFAULT_TRACK_VOL} onClick={() => setTrackVolume(f, DEFAULT_TRACK_VOL)}>
                      Reset
                    </button>
                  </div>
                )}
              </li>
            );
          })}
        </ol>
        </div>
       </div>
      </main>
    </>
  );
}