import { useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";
import type { CentralityMetric } from "../types";
import type { GraphBuffers } from "../graphBuffers";
import { metricArray } from "../graphBuffers";
import { COMMUNITY_COLORS } from "../constants";

/**
 * WebGL-рендерер графов на 10^4-10^5 вершин с живой силовой укладкой. Рёбра
 * рисуются индексно (drawElements по общему буферу координат): смена координат
 * стоит одну загрузку 2n float вместо 4m — только так физика укладывается в
 * кадр. Прозрачность ребра шейдер выводит из прозрачностей концов, поэтому
 * словарь на 198 050 рёбер с сервера не передаётся.
 */

export interface LiveHandle {
  onFrame: (cb: (pos: Float32Array) => void) => () => void;
  drag: (id: number, x: number, y: number, pin: boolean) => void;
  release: (id: number) => void;
}

interface Props {
  graph: GraphBuffers;
  focusAlpha: Float32Array | null;
  sizeMetric: CentralityMetric;
  selectedNode: number | null;
  onNodeClick: (nodeId: number) => void;
  live: LiveHandle | null;
  /** 0 — серые рёбра, 1 — окрашенные по сообществам */
  edgeTint: number;
  edgeOpacity: number;
}

const NODE_VS = `
attribute vec2 a_pos;
attribute float a_size;
attribute float a_alpha;
attribute vec3 a_color;
uniform vec2 u_viewport;
uniform vec2 u_fitScale;
uniform vec2 u_fitOffset;
uniform vec2 u_translate;
uniform float u_scale;
uniform float u_dpr;
uniform float u_maxSize;
varying vec4 v_color;
void main() {
  vec2 world = a_pos * u_fitScale + u_fitOffset;
  vec2 screen = world * u_scale + u_translate;
  vec2 clip = (screen / u_viewport * 2.0 - 1.0) * vec2(1.0, -1.0);
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = clamp(a_size * sqrt(u_scale) * u_dpr, 1.0 * u_dpr, u_maxSize);
  v_color = vec4(a_color, a_alpha);
}`;

const NODE_FS = `
precision mediump float;
varying vec4 v_color;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  float d2 = dot(c, c);
  if (d2 > 1.0) discard;
  float a = v_color.a * (1.0 - smoothstep(0.55, 1.0, d2));
  gl_FragColor = vec4(v_color.rgb * a, a);
}`;

const EDGE_VS = `
attribute vec2 a_pos;
attribute float a_alpha;
attribute vec3 a_color;
uniform vec2 u_viewport;
uniform vec2 u_fitScale;
uniform vec2 u_fitOffset;
uniform vec2 u_translate;
uniform float u_scale;
uniform float u_edgeAlpha;
uniform float u_tint;
varying vec4 v_color;
void main() {
  vec2 world = a_pos * u_fitScale + u_fitOffset;
  vec2 screen = world * u_scale + u_translate;
  vec2 clip = (screen / u_viewport * 2.0 - 1.0) * vec2(1.0, -1.0);
  gl_Position = vec4(clip, 0.0, 1.0);
  vec3 grey = vec3(0.40, 0.46, 0.58);
  v_color = vec4(mix(grey, a_color, u_tint), a_alpha * u_edgeAlpha);
}`;

const EDGE_FS = `
precision mediump float;
varying vec4 v_color;
void main() {
  gl_FragColor = vec4(v_color.rgb * v_color.a, v_color.a);
}`;

function compile(gl: WebGLRenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(sh) ?? "ошибка компиляции шейдера");
  }
  return sh;
}

function link(gl: WebGLRenderingContext, vs: string, fs: string): WebGLProgram {
  const p = gl.createProgram()!;
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p) ?? "ошибка линковки программы");
  }
  return p;
}

function hexToRgb(hex: string): [number, number, number] {
  const v = parseInt(hex.slice(1), 16);
  return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255];
}

const PALETTE = COMMUNITY_COLORS.map(hexToRgb);

function nodeColors(g: GraphBuffers): Float32Array {
  const color = new Float32Array(g.n * 3);
  for (let i = 0; i < g.n; i++) {
    const c = PALETTE[g.community[i] % PALETTE.length];
    color[i * 3] = c[0];
    color[i * 3 + 1] = c[1];
    color[i * 3 + 2] = c[2];
  }
  return color;
}

/** Прозрачности вершин; вне режима фокуса — единицы. */
function currentAlpha(g: GraphBuffers, focus: Float32Array | null): Float32Array {
  const a = new Float32Array(g.n).fill(1);
  if (focus && focus.length === g.n) a.set(focus);
  return a;
}

/** Радиус вершины в пикселях при единичном масштабе. */
function nodeSizes(g: GraphBuffers, metric: CentralityMetric): Float32Array {
  const values = metricArray(g, metric);
  let max = 0;
  for (let i = 0; i < g.n; i++) if (values[i] > max) max = values[i];
  const inv = max > 0 ? 1 / max : 0;

  const base = g.n > 50000 ? 0.9 : g.n > 20000 ? 1.2 : g.n > 5000 ? 1.6 : 2.4;
  const span = g.n > 50000 ? 4 : g.n > 20000 ? 5.5 : g.n > 5000 ? 7 : 10;

  const out = new Float32Array(g.n);
  for (let i = 0; i < g.n; i++) {
    if (g.isCollapsed[i]) {
      // Без потолка сообщество на 2000 вершин рисуется кругом в 160 px.
      const rel = Math.sqrt(g.memberCount[i]) / Math.sqrt(Math.max(2, maxMember(g)));
      out[i] = 5 + 22 * rel;
    } else {
      out[i] = base + span * Math.sqrt(values[i] * inv);
    }
  }
  return out;
}

let maxMemberCache = new WeakMap<GraphBuffers, number>();
function maxMember(g: GraphBuffers): number {
  const cached = maxMemberCache.get(g);
  if (cached !== undefined) return cached;
  let max = 1;
  for (let i = 0; i < g.n; i++) if (g.memberCount[i] > max) max = g.memberCount[i];
  maxMemberCache.set(g, max);
  return max;
}

export default function GraphGL({
  graph,
  focusAlpha,
  sizeMetric,
  selectedNode,
  onNodeClick,
  live,
  edgeTint,
  edgeOpacity,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const labelRef = useRef<HTMLCanvasElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);

  const glRef = useRef<{
    gl: WebGLRenderingContext;
    isGL2: boolean;
    nodeProg: WebGLProgram;
    edgeProg: WebGLProgram;
    posBuf: WebGLBuffer;
    sizeBuf: WebGLBuffer;
    alphaBuf: WebGLBuffer;
    colorBuf: WebGLBuffer;
    idxBuf: WebGLBuffer;
    maxPointSize: number;
  } | null>(null);

  // всё, что меняется каждый кадр, живёт в ref — React в отрисовке не участвует
  const stRef = useRef({
    xy: new Float32Array(0),
    transform: d3.zoomIdentity,
    width: 0,
    height: 0,
    dpr: 1,
    fit: { sx: 1, sy: 1, ox: 0, oy: 0 },
    dragId: -1,
    hoverIdx: -1,
    edgeCount: 0,
    // Копия размеров: попадание курсора считается по фактическому радиусу.
    sizes: new Float32Array(0) as Float32Array<ArrayBufferLike>,
    maxPointSize: 64,
  });

  const propsRef = useRef({ graph, focusAlpha, sizeMetric, selectedNode, edgeTint, edgeOpacity, live });
  propsRef.current = { graph, focusAlpha, sizeMetric, selectedNode, edgeTint, edgeOpacity, live };

  const rafRef = useRef(0);
  const uploadRef = useRef<(() => void) | null>(null);

  // ── подгонка масштаба под содержимое ────────────────────────────
  const updateFit = useCallback((immediate: boolean) => {
    const st = stRef.current;
    const xy = st.xy;
    const n = xy.length >> 1;
    if (n === 0 || st.width === 0) return;
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = xy[i * 2], y = xy[i * 2 + 1];
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const pad = 30;
    const ex = Math.max(maxX - minX, 1e-6);
    const ey = Math.max(maxY - minY, 1e-6);
    const s = Math.min((st.width - 2 * pad) / ex, (st.height - 2 * pad) / ey);
    const target = {
      sx: s,
      sy: s,
      ox: pad - minX * s + (st.width - 2 * pad - ex * s) / 2,
      oy: pad - minY * s + (st.height - 2 * pad - ey * s) / 2,
    };
    if (immediate) {
      st.fit = target;
    } else {
      // При живой физике границы плывут: резкая подгонка = дрожание картинки.
      const a = 0.08;
      st.fit = {
        sx: st.fit.sx + (target.sx - st.fit.sx) * a,
        sy: st.fit.sy + (target.sy - st.fit.sy) * a,
        ox: st.fit.ox + (target.ox - st.fit.ox) * a,
        oy: st.fit.oy + (target.oy - st.fit.oy) * a,
      };
    }
  }, []);

  /** Подписи на 2D-слое поверх WebGL: текста в шейдерах нет, вдали не читаются. */
  const drawLabels = useCallback(() => {
    const canvas = labelRef.current;
    const st = stRef.current;
    const p = propsRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.setTransform(st.dpr, 0, 0, st.dpr, 0, 0);
    ctx.clearRect(0, 0, st.width, st.height);

    const n = p.graph.n;
    const t = st.transform;
    const show = n < 300 || (t.k > 1.2 && n < 3000) || t.k > 4;
    if (!show) return;

    ctx.font = "11px Inter, system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "#cbd5e1";

    let drawn = 0;
    for (let i = 0; i < n && drawn < 300; i++) {
      const alpha = p.focusAlpha ? p.focusAlpha[i] : 1;
      if (alpha < 0.25) continue;
      const wx = (st.xy[i * 2] * st.fit.sx + st.fit.ox) * t.k + t.x;
      const wy = (st.xy[i * 2 + 1] * st.fit.sy + st.fit.oy) * t.k + t.y;
      if (wx < -40 || wy < -20 || wx > st.width + 40 || wy > st.height + 20) continue;
      ctx.globalAlpha = Math.min(alpha * 0.9, 0.9);
      const text = p.graph.isCollapsed[i]
        ? String(p.graph.memberCount[i])
        : String(p.graph.ids[i]);
      ctx.fillText(text, wx, wy - 8);
      drawn++;
    }
    ctx.globalAlpha = 1;
  }, []);

  // ── отрисовка кадра ─────────────────────────────────────────────
  const draw = useCallback(() => {
    const g = glRef.current;
    const st = stRef.current;
    const p = propsRef.current;
    if (!g || st.width === 0) return;
    const { gl } = g;
    const t = st.transform;

    gl.viewport(0, 0, st.width * st.dpr, st.height * st.dpr);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    const setCommon = (prog: WebGLProgram) => {
      gl.uniform2f(gl.getUniformLocation(prog, "u_viewport"), st.width, st.height);
      gl.uniform2f(gl.getUniformLocation(prog, "u_fitScale"), st.fit.sx, st.fit.sy);
      gl.uniform2f(gl.getUniformLocation(prog, "u_fitOffset"), st.fit.ox, st.fit.oy);
      gl.uniform2f(gl.getUniformLocation(prog, "u_translate"), t.x, t.y);
      gl.uniform1f(gl.getUniformLocation(prog, "u_scale"), t.k);
    };

    const bindAttr = (prog: WebGLProgram, name: string, buf: WebGLBuffer, size: number) => {
      const loc = gl.getAttribLocation(prog, name);
      if (loc < 0) return;
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
    };

    // ── рёбра: индексная отрисовка с LOD по масштабу ──────────────
    const m = p.graph.m;
    if (m > 0) {
      // Рёбра заранее перемешаны, поэтому префикс буфера — честная подвыборка.
      const budget = Math.min(m, Math.round(40000 * Math.max(1, t.k)));
      const drawn = p.focusAlpha ? m : budget;
      st.edgeCount = drawn;

      // Прозрачность компенсирует прореживание; потолок 0.55 её срезал.
      const alpha = Math.min(
        1,
        Math.max(0.02, (p.edgeOpacity * 900) / Math.sqrt(Math.max(drawn, 1)) / Math.sqrt(Math.max(1, 40 / (t.k + 1)))),
      );

      gl.useProgram(g.edgeProg);
      setCommon(g.edgeProg);
      gl.uniform1f(gl.getUniformLocation(g.edgeProg, "u_edgeAlpha"), alpha);
      gl.uniform1f(gl.getUniformLocation(g.edgeProg, "u_tint"), p.edgeTint);
      bindAttr(g.edgeProg, "a_pos", g.posBuf, 2);
      bindAttr(g.edgeProg, "a_alpha", g.alphaBuf, 1);
      bindAttr(g.edgeProg, "a_color", g.colorBuf, 3);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.idxBuf);
      gl.drawElements(gl.LINES, drawn * 2, gl.UNSIGNED_INT, 0);
    }

    // ── вершины ──────────────────────────────────────────────────
    gl.useProgram(g.nodeProg);
    setCommon(g.nodeProg);
    gl.uniform1f(gl.getUniformLocation(g.nodeProg, "u_dpr"), st.dpr);
    gl.uniform1f(gl.getUniformLocation(g.nodeProg, "u_maxSize"), g.maxPointSize);
    bindAttr(g.nodeProg, "a_pos", g.posBuf, 2);
    bindAttr(g.nodeProg, "a_size", g.sizeBuf, 1);
    bindAttr(g.nodeProg, "a_alpha", g.alphaBuf, 1);
    bindAttr(g.nodeProg, "a_color", g.colorBuf, 3);
    gl.drawArrays(gl.POINTS, 0, p.graph.n);

    drawLabels();

    if (import.meta.env.DEV) {
      (window as unknown as Record<string, unknown>).__glDebug = {
        fit: { ...st.fit },
        transform: { k: t.k, x: t.x, y: t.y },
        n: p.graph.n,
        m: p.graph.m,
        edgesDrawn: st.edgeCount,
        xyLen: st.xy.length,
        xySample: Array.from(st.xy.slice(0, 6)),
        glError: gl.getError(),
      };
    }
  }, []);

  const schedule = useCallback(() => {
    if (rafRef.current) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = 0;
      draw();
    });
  }, [draw]);

  // ── инициализация контекста и взаимодействия ────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const gl2 = canvas.getContext("webgl2", { antialias: true, alpha: true });
    const gl = (gl2 ??
      canvas.getContext("webgl", { antialias: true, alpha: true })) as WebGLRenderingContext | null;
    if (!gl) return;
    const isGL2 = !!gl2;
    if (!isGL2 && !gl.getExtension("OES_element_index_uint")) {
      console.warn("нет OES_element_index_uint: графы свыше 65535 вершин не отрисуются");
    }

    const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE) as Float32Array;
    glRef.current = {
      gl,
      isGL2,
      nodeProg: link(gl, NODE_VS, NODE_FS),
      edgeProg: link(gl, EDGE_VS, EDGE_FS),
      posBuf: gl.createBuffer()!,
      sizeBuf: gl.createBuffer()!,
      alphaBuf: gl.createBuffer()!,
      colorBuf: gl.createBuffer()!,
      idxBuf: gl.createBuffer()!,
      maxPointSize: range ? Math.min(range[1], 96) : 64,
    };

    const resize = () => {
      const rect = container.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = `${rect.width}px`;
      canvas.style.height = `${rect.height}px`;
      const labels = labelRef.current;
      if (labels) {
        labels.width = rect.width * dpr;
        labels.height = rect.height * dpr;
        labels.style.width = `${rect.width}px`;
        labels.style.height = `${rect.height}px`;
      }
      stRef.current.width = rect.width;
      stRef.current.height = rect.height;
      stRef.current.dpr = dpr;
      updateFit(true);
    };
    const obs = new ResizeObserver(() => {
      resize();
      schedule();
    });
    obs.observe(container);
    resize();

    const zoom = d3
      .zoom<HTMLCanvasElement, unknown>()
      .scaleExtent([0.05, 60])
      // d3.zoom слушает mousedown, а вершина перехватывается на pointerdown:
      // stopImmediatePropagation на одном не гасит другое, нужен этот фильтр.
      .filter((event: Event) => stRef.current.dragId < 0 && !(event as MouseEvent).ctrlKey)
      .on("zoom", (event) => {
        stRef.current.transform = event.transform;
        schedule();
      });
    d3.select(canvas).call(zoom);

    // ── попадание курсора в вершину: линейный проход ──────────────
    // На 18 772 вершинах ~0.1 мс — дешевле перестройки квадродерева каждый кадр.
    const pick = (ev: PointerEvent | MouseEvent): number => {
      const st = stRef.current;
      const p = propsRef.current;
      const rect = canvas.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      const my = ev.clientY - rect.top;
      const t = st.transform;
      const xy = st.xy;
      const sizes = st.sizes;
      // Повторяет формулу шейдера gl_PointSize = clamp(a_size * sqrt(k) * dpr,
      // dpr, maxPointSize) — это диаметр в пикселях устройства, не радиус.
      const zoom = Math.sqrt(Math.max(t.k, 1e-6));
      const capCss = st.maxPointSize / Math.max(st.dpr, 1);
      // Побеждает не ближайшая вершина, а та, в чей круг курсор попал «глубже»:
      // иначе хаб перехватывает клики по мелким соседям.
      let best = -1;
      let bestScore = 1;
      for (let i = 0; i < p.graph.n; i++) {
        const wx = (xy[i * 2] * st.fit.sx + st.fit.ox) * t.k + t.x;
        const wy = (xy[i * 2 + 1] * st.fit.sy + st.fit.oy) * t.k + t.y;
        const dx = wx - mx;
        const dy = wy - my;
        const d2 = dx * dx + dy * dy;
        if (d2 > 40 * 40) continue;
        const drawn = sizes.length > i
          ? Math.min(Math.max(sizes[i] * zoom, 1), capCss) / 2
          : 3;
        const target = Math.max(drawn + 3, 8); // 8 px — минимум, чтобы точка оставалась попадаемой
        const score = Math.sqrt(d2) / target;
        if (score <= 1 && score < bestScore) {
          bestScore = score;
          best = i;
        }
      }
      return best;
    };

    const toSim = (ev: PointerEvent): [number, number] => {
      const st = stRef.current;
      const rect = canvas.getBoundingClientRect();
      const t = st.transform;
      const wx = (ev.clientX - rect.left - t.x) / t.k;
      const wy = (ev.clientY - rect.top - t.y) / t.k;
      return [(wx - st.fit.ox) / st.fit.sx, (wy - st.fit.oy) / st.fit.sy];
    };

    let downIdx = -1;
    let moved = false;
    let downX = 0;
    let downY = 0;
    let lastPickX = -1e9;
    let lastPickY = -1e9;

    const onPointerDown = (ev: PointerEvent) => {
      if (ev.button !== 0) return;
      downIdx = pick(ev);
      moved = false;
      downX = ev.clientX;
      downY = ev.clientY;
      // Перехват у d3.zoom только когда вершину есть чем тащить: иначе запретим
      // панорамирование, начатое рядом с ней.
      if (downIdx >= 0 && propsRef.current.live) {
        ev.stopImmediatePropagation();
        canvas.setPointerCapture(ev.pointerId);
        stRef.current.dragId = downIdx;
      }
    };

    const onPointerMove = (ev: PointerEvent) => {
      const st = stRef.current;
      const p = propsRef.current;
      if (downIdx >= 0 || st.dragId >= 0) {
        const dx = ev.clientX - downX;
        const dy = ev.clientY - downY;
        if (dx * dx + dy * dy > 16) moved = true;
      }
      if (st.dragId >= 0 && p.live) {
        const [sx, sy] = toSim(ev);
        st.xy[st.dragId * 2] = sx;
        st.xy[st.dragId * 2 + 1] = sy;
        p.live.drag(st.dragId, sx, sy, true);
        schedule();
        return;
      }
      // Проход по 10^5 вершинам на каждое движение мыши заметен — только после сдвига.
      const mdx = ev.clientX - lastPickX;
      const mdy = ev.clientY - lastPickY;
      if (mdx * mdx + mdy * mdy < 9) return;
      lastPickX = ev.clientX;
      lastPickY = ev.clientY;
      const idx = pick(ev);
      st.hoverIdx = idx;
      canvas.style.cursor = idx >= 0 ? "pointer" : "grab";
      const tip = tooltipRef.current;
      if (!tip) return;
      if (idx >= 0) {
        const rect = canvas.getBoundingClientRect();
        const g = p.graph;
        tip.style.display = "block";
        tip.style.left = `${ev.clientX - rect.left + 14}px`;
        tip.style.top = `${ev.clientY - rect.top + 14}px`;
        tip.innerHTML = g.isCollapsed[idx]
          ? `<b>Сообщество #${g.community[idx]}</b><br/>${g.memberCount[idx]} вершин<br/><i>Нажмите для просмотра</i>`
          : `<b>${g.ids[idx]}</b><br/>Степень: ${g.degree[idx]}<br/>Сообщество: #${g.community[idx]}`;
      } else {
        tip.style.display = "none";
      }
    };

    const onPointerUp = (ev: PointerEvent) => {
      const st = stRef.current;
      const p = propsRef.current;
      if (st.dragId >= 0) {
        if (p.live) p.live.release(st.dragId);
        if (!moved) onNodeClickRef.current(p.graph.ids[st.dragId]);
        st.dragId = -1;
        canvas.releasePointerCapture?.(ev.pointerId);
        return;
      }
      if (downIdx >= 0 && !moved) {
        onNodeClickRef.current(propsRef.current.graph.ids[downIdx]);
      }
      downIdx = -1;
    };

    canvas.addEventListener("pointerdown", onPointerDown, true);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    // Без preventDefault потерянный контекст (сон, сброс драйвера) не вернуть.
    canvas.addEventListener("webglcontextlost", (e) => e.preventDefault());
    canvas.addEventListener("webglcontextrestored", () => {
      const g = glRef.current;
      if (!g) return;
      g.nodeProg = link(g.gl, NODE_VS, NODE_FS);
      g.edgeProg = link(g.gl, EDGE_VS, EDGE_FS);
      g.posBuf = g.gl.createBuffer()!;
      g.sizeBuf = g.gl.createBuffer()!;
      g.alphaBuf = g.gl.createBuffer()!;
      g.colorBuf = g.gl.createBuffer()!;
      g.idxBuf = g.gl.createBuffer()!;
      uploadRef.current?.();
    });

    return () => {
      obs.disconnect();
      canvas.removeEventListener("pointerdown", onPointerDown, true);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      // Флаг обязан обнуляться вместе с отменой кадра, иначе schedule() навсегда
      // считает отрисовку запланированной: в StrictMode этот cleanup срабатывает
      // сразу после первой постановки кадра, и граф не рисуется вообще.
      cancelAnimationFrame(rafRef.current);
      rafRef.current = 0;
    };
  }, [schedule, updateFit]);

  const onNodeClickRef = useRef(onNodeClick);
  onNodeClickRef.current = onNodeClick;

  // ── загрузка геометрии в GPU при смене графа ────────────────────
  useEffect(() => {
    const g = glRef.current;
    if (!g || graph.n === 0) return;
    const { gl } = g;
    const st = stRef.current;

    st.xy = graph.xy.slice();

    gl.bindBuffer(gl.ARRAY_BUFFER, g.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, st.xy, gl.DYNAMIC_DRAW);

    const color = new Float32Array(graph.n * 3);
    for (let i = 0; i < graph.n; i++) {
      const c = PALETTE[graph.community[i] % PALETTE.length];
      color[i * 3] = c[0];
      color[i * 3 + 1] = c[1];
      color[i * 3 + 2] = c[2];
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, g.colorBuf);
    gl.bufferData(gl.ARRAY_BUFFER, color, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, g.alphaBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(graph.n).fill(1), gl.DYNAMIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.idxBuf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, graph.edges, gl.STATIC_DRAW);

    updateFit(true);
    schedule();
  }, [graph, schedule, updateFit]);

  // повторная загрузка геометрии после восстановления контекста WebGL
  useEffect(() => {
    uploadRef.current = () => {
      const g = glRef.current;
      if (!g) return;
      const st = stRef.current;
      const { gl } = g;
      // Пусты ВСЕ буферы: незаполненные a_size/a_alpha/a_color дают INVALID_OPERATION.
      gl.bindBuffer(gl.ARRAY_BUFFER, g.posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, st.xy, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.sizeBuf);
      const sz0 = nodeSizes(graph, propsRef.current.sizeMetric);
      stRef.current.sizes = sz0;
      stRef.current.maxPointSize = g.maxPointSize;
      gl.bufferData(gl.ARRAY_BUFFER, sz0, gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.colorBuf);
      gl.bufferData(gl.ARRAY_BUFFER, nodeColors(graph), gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, g.alphaBuf);
      gl.bufferData(gl.ARRAY_BUFFER, currentAlpha(graph, propsRef.current.focusAlpha), gl.DYNAMIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, g.idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, graph.edges, gl.STATIC_DRAW);
      schedule();
    };
  }, [graph, schedule]);

  // ── размеры вершин: только при смене метрики или графа ──────────
  useEffect(() => {
    const g = glRef.current;
    if (!g || graph.n === 0) return;
    const { gl } = g;
    gl.bindBuffer(gl.ARRAY_BUFFER, g.sizeBuf);
    const sz = nodeSizes(graph, sizeMetric);
    stRef.current.sizes = sz;
    stRef.current.maxPointSize = g.maxPointSize;
    gl.bufferData(gl.ARRAY_BUFFER, sz, gl.DYNAMIC_DRAW);
    schedule();
  }, [graph, sizeMetric, schedule]);

  // ── прозрачности фокуса: одна перезапись буфера ─────────────────
  useEffect(() => {
    const g = glRef.current;
    if (!g || graph.n === 0) return;
    const { gl } = g;
    const alpha = focusAlpha ?? new Float32Array(graph.n).fill(1);
    const buf = focusAlpha ? alpha.slice() : alpha;
    if (selectedNode !== null) {
      const idx = graph.indexOf.get(selectedNode);
      if (idx !== undefined) buf[idx] = 1;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, g.alphaBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, buf);
    schedule();
  }, [graph, focusAlpha, selectedNode, schedule]);

  // ── отображение рёбер ───────────────────────────────────────────
  // Яркость и окраску draw() читает из propsRef, буферы не трогаем; но кадр
  // запросить надо, иначе с выключенной симуляцией изменение не видно.
  useEffect(() => {
    if (glRef.current) schedule();
  }, [edgeOpacity, edgeTint, schedule]);

  // ── приём координат от симуляции ────────────────────────────────
  useEffect(() => {
    if (!live) return;
    return live.onFrame((pos) => {
      const g = glRef.current;
      const st = stRef.current;
      if (!g || pos.length !== st.xy.length) return;
      const dragId = st.dragId;
      st.xy.set(pos);
      if (dragId >= 0) {
        // Координаты от мыши приоритетнее: воркер узнает о них следующим drag.
      }
      const { gl } = g;
      gl.bindBuffer(gl.ARRAY_BUFFER, g.posBuf);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, st.xy);
      updateFit(false);
      schedule();
    });
  }, [live, schedule, updateFit]);

  return (
    <div ref={containerRef} className="graph-canvas">
      <canvas ref={canvasRef} />
      <canvas ref={labelRef} className="label-layer" />
      <div ref={tooltipRef} className="canvas-tooltip" />
    </div>
  );
}
