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
type Cfg = { loop: number; level: number; repeat: boolean };

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
    this.master.connect(this.fader);
    this.fader.connect(out);
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
const title = (f: File) => f.name.replace(/\.[^.]+$/, "");
const fmt = (s: number) => {
  const r = Math.floor(s % 60);
  return `${Math.floor(s / 60)}:${r < 10 ? "0" : ""}${r}`;
};
const volGain = (v: number) => (v / 100) ** 2; // squared so the slider feels even
const fill = (v: number, lo: number, hi: number) =>
  ({ "--p": `${((v - lo) / (hi - lo || 1)) * 100}%` }) as CSSProperties;

const Icon = ({ d }: { d: string }) => (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor" aria-hidden="true">
    <path d={d} />
  </svg>
);
const PREV = "M6 6h2v12H6zM20 6v12L9.5 12z";
const NEXT = "M16 6h2v12h-2zM4 6v12l10.5-6z";
const PLAY = "M8 5v14l11-7z";
const PAUSE = "M6 5h4v14H6zM14 5h4v14h-4z";
const REPEAT = "M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4zm-4-2V9h-1l-2 1v1h1.5v4H13z";

const css = `
:root{--bg:#f5f6f8;--surface:#fff;--text:#1b1d22;--muted:#6b7280;--line:#e3e5ea;--track:#d9dce3;--accent:#4756d6;--on-accent:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--surface:#1d2027;--text:#e8eaef;--muted:#8c93a1;--line:#2a2e37;--track:#394050;--accent:#8b97ff;--on-accent:#14161b}}
body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,sans-serif;display:block;min-height:0}
.app{box-sizing:border-box;width:100%;max-width:960px;margin:0 auto;padding:40px 16px;text-align:left}
.layout{display:grid;grid-template-columns:minmax(0,420px) minmax(0,1fr);gap:32px;align-items:start}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:24px 20px}
.title{font-size:20px;font-weight:600;margin:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-height:28px}
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
  const [busy, setBusy] = useState(false);
  const [fade, setFade] = useStored<number>("fade", 4, [0, 12]);
  const [loopFade, setLoopFade] = useStored<number>("loopFade", 3, [0, 12]);
  const [cross, setCross] = useStored<number>("cross", 71, [50, 90]);
  const [volume, setVolume] = useStored<number>("volume", 100, [0, 100]);
  const [repeat, setRepeat] = useStored<boolean>("repeat", true);
  const [silence, setSilence] = useStored<number>("silence", -54, [-90, -20]);
  const [alpha, setAlpha] = useStored<boolean>("alphaSort", true);
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
  const silenceRef = useRef(silence);
  silenceRef.current = silence;
  const alphaRef = useRef(alpha);
  alphaRef.current = alpha;
  const addedRef = useRef<File[]>([]); // files in the order they were added
  const advancing = useRef(false);

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

  const load = (file: File) => {
    let p = cache.current.get(file);
    if (!p) {
      p = file.arrayBuffer().then((ab) => getCtx().decodeAudioData(ab));
      cache.current.set(file, p);
      p.catch(() => cache.current.delete(file));
    }
    return p;
  };

  const go = async (target: number, fadeOverride?: number) => {
    const list = tracksRef.current;
    if (!list.length) return;
    const i = (target + list.length) % list.length;
    const ctx = getCtx();
    const f = fadeOverride ?? fadeRef.current;
    const my = ++token.current;
    setIndex(i);
    setBusy(true);
    try {
      const buf = await load(list[i]);
      if (my !== token.current) return; // a newer click superseded this one
      const [lo, hi] = audibleRange(buf, 10 ** (silenceRef.current / 20)); // dB -> amplitude
      const track: Loaded = { buf, lo, hi };
      if (ctx.state === "suspended") await ctx.resume();
      const old = voiceRef.current;
      old?.fadeOut(f);
      voiceRef.current = new Voice(
        ctx,
        track,
        () => ({ loop: loopFadeRef.current, level: crossRef.current / 100, repeat: repeatRef.current }),
        outRef.current!,
        old ? Math.max(f, 0.03) : 0.3
      );
      setPlaying(true);

      // Keep only the current and next track decoded; preload the next one.
      const next = list[(i + 1) % list.length];
      const keep = new Set([list[i], next]);
      for (const f of [...cache.current.keys()]) if (!keep.has(f)) cache.current.delete(f);
      load(next).catch(() => {});
    } catch (e) {
      console.error("Could not play", list[i]?.name, e);
    } finally {
      if (my === token.current) setBusy(false);
    }
  };

  const toggle = async () => {
    const ctx = ctxRef.current;
    if (!voiceRef.current || !ctx) return go(indexRef.current);
    if (ctx.state === "running") {
      await ctx.suspend();
      setPlaying(false);
    } else {
      await ctx.resume();
      setPlaying(true);
    }
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
      if (v && !repeatRef.current && !advancing.current && ctxRef.current?.state === "running") {
        const done = v.isDone(); // fallback if the poll missed the end (e.g. throttled tab)
        const left = done ? 0 : v.duration - v.position();
        // Never fade for longer than half the track; keep a small lead even at 0 so the hand-off is seamless.
        const lead = Math.max(Math.min(fadeRef.current, v.duration / 2), 0.25);
        if (done || left <= lead) {
          advancing.current = true;
          go(indexRef.current + 1, Math.max(0.1, left - 0.1)).finally(() => {
            window.setTimeout(() => (advancing.current = false), 2000);
          });
        }
      }
    }, 100);
    return () => window.clearInterval(id);
  }, []);

  // Apply the volume slider (smoothed to avoid zipper noise).
  useEffect(() => {
    const ctx = ctxRef.current;
    const out = outRef.current;
    if (ctx && out) out.gain.setTargetAtTime(volGain(volume), ctx.currentTime, 0.02);
  }, [volume]);

  const current = tracks[index];
  const shown = scrub ?? pos;

  return (
    <>
      <style>{css}</style>
      <main className="app">
       <div className="layout">
        <div>
        <section className="panel">
          <h1 className="title">{current ? title(current) : "No track loaded"}</h1>
          <p className="sub">
            {busy ? "Loading…" : tracks.length ? `Track ${index + 1} of ${tracks.length}` : "Add some audio files to start"}
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
            <button className="btn toggle" aria-label="Loop track" aria-pressed={repeat} title="Loop track"
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
              <button aria-pressed={!alpha} onClick={() => toggleSort(false)}>Added</button>
            </div>
          </div>
        )}
        <ol className="list">
          {tracks.map((f, i) => (
            <li key={idOf(f)}>
              <button className="item" aria-current={i === index} onClick={() => go(i)}>
                {title(f)}
              </button>
            </li>
          ))}
        </ol>
        </div>
       </div>
      </main>
    </>
  );
}