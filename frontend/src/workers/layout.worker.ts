/// <reference lib="webworker" />
/**
 * Порт backend/app/fastlayout.py на типизированные массивы. Модель сил
 * Ноака (a, r): притяжение f_a = w · d^a / k^(a-1), отталкивание
 * f_r = k^(1-r) · m_u · m_v · d^r. Ближнее поле — сетка с ячейкой 2k (3×3
 * окрестность, сортировка подсчётом) с прореживанием плотных ячеек,
 * дальнее — случайная выборка из S вершин с масштабированием n/S.
 * Кадры уходят наружу передачей владения ArrayBuffer, без копирования.
 */

export interface SimParams {
  /** показатель a: 0 — LinLog, 1 — линейное, 2 — Фрухтерман-Рейнгольд */
  attractionExp: number;
  repulsionExp: number;
  attraction: number;
  repulsion: number;
  gravity: number;
  /** сглаживание скорости: 0 — резко, 0.9 — плавно «плавает» */
  damping: number;
  degreeMass: boolean;
  /** размер случайной выборки для дальнего поля */
  sampleS: number;
  nearField: boolean;
  /** 0 — все рёбра равны, 1 — пропорционально весу */
  weightInfluence: number;
}

const ALPHA_DECAY = 0.0228;
const ALPHA_MIN = 0.02;
/** предельное смещение вершины за шаг, в единицах k */
const MAX_DISP = 1.4;
const RENORM_EVERY = 120;
const MAX_CELLS = 1 << 18;

export const DEFAULT_PARAMS: SimParams = {
  attractionExp: 1,
  repulsionExp: -1,
  attraction: 1,
  repulsion: 1,
  gravity: 0.05,
  damping: 0.55,
  degreeMass: true,
  sampleS: 16,
  nearField: true,
  weightInfluence: 1,
};

type InitMsg = {
  type: "init";
  n: number;
  m: number;
  pos: ArrayBuffer; // Float32Array(2n), стартовые координаты с сервера
  src: ArrayBuffer; // Uint32Array(m)
  tgt: ArrayBuffer; // Uint32Array(m)
  weight: ArrayBuffer | null; // Float32Array(m)
  degree: ArrayBuffer; // Float32Array(n)
  params: SimParams;
};

type InMsg =
  | InitMsg
  | { type: "params"; params: Partial<SimParams>; reheat?: boolean }
  | { type: "control"; running?: boolean; reheat?: number; ticksPerFrame?: number }
  | { type: "drag"; id: number; x: number; y: number; pin: boolean }
  | { type: "release"; id: number }
  | { type: "recycle"; buf: ArrayBuffer };

// ── состояние симуляции ──────────────────────────────────────────

let n = 0;
let m = 0;
let px = new Float32Array(0); // рабочие координаты
let vx = new Float32Array(0); // скорости
let fx = new Float32Array(0); // накопитель сил
let src = new Uint32Array(0);
let tgt = new Uint32Array(0);
let weight: Float32Array | null = null;
let mass = new Float32Array(0);
let pinned = new Uint8Array(0);

let params: SimParams = { ...DEFAULT_PARAMS };
let running = false;
let alpha = 1; // «температура»: множитель шага
let alphaTarget = 0;
let k = 0.01; // равновесное расстояние между соседями
let tick = 0;
let ticksPerFrame = 1;
let settled = false;

const spare: ArrayBuffer[] = []; // ping-pong буферов координат

// Массивы сетки ближнего поля выделяются один раз: аллокация внутри шага
// давала выбросы времени кадра до 360 мс на сборке мусора.
let cellOf = new Int32Array(0);
let order = new Uint32Array(0);
let cellStart = new Int32Array(0);
let cursor = new Int32Array(0);
let witness = new Uint32Array(0);

const EPS = 1e-12;

function alloc(size: number) {
  px = new Float32Array(size * 2);
  vx = new Float32Array(size * 2);
  fx = new Float32Array(size * 2);
  pinned = new Uint8Array(size);
  cellOf = new Int32Array(size);
  order = new Uint32Array(size);
  cellStart = new Int32Array(MAX_CELLS + 1);
  cursor = new Int32Array(MAX_CELLS);
  witness = new Uint32Array(64);
}

function recomputeMass(degree: Float32Array) {
  mass = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    mass[i] = params.degreeMass ? 1 + degree[i] : 1;
  }
}

let degreeRef = new Float32Array(0);

// ── силы ─────────────────────────────────────────────────────────

/** Притяжение вдоль рёбер: f_a = attraction · w · d^a / k^(a-1). */
function applyAttraction() {
  const a = params.attractionExp;
  const kPow = Math.pow(k, 1 - a); // d^a · k^(1-a) == d^a / k^(a-1)
  const strength = params.attraction;
  const wInfl = params.weightInfluence;

  const fast1 = Math.abs(a - 1) < 1e-6;
  const fast0 = Math.abs(a) < 1e-6;
  const fast2 = Math.abs(a - 2) < 1e-6;

  for (let e = 0; e < m; e++) {
    const s = src[e];
    const t = tgt[e];
    const sx = s * 2;
    const tx = t * 2;
    const dx = px[tx] - px[sx];
    const dy = px[tx + 1] - px[sx + 1];
    const d2 = dx * dx + dy * dy;
    if (d2 < EPS) continue;
    const d = Math.sqrt(d2);

    let w = 1;
    if (weight !== null && wInfl > 0) {
      w = wInfl === 1 ? weight[e] : Math.pow(weight[e], wInfl);
    }

    let mag: number;
    if (fast1) mag = strength * w * d;
    else if (fast0) mag = strength * w * kPow;
    else if (fast2) mag = strength * w * d2 * kPow;
    else mag = strength * w * Math.pow(d, a) * kPow;

    const ux = (dx / d) * mag;
    const uy = (dy / d) * mag;
    fx[sx] += ux;
    fx[sx + 1] += uy;
    fx[tx] -= ux;
    fx[tx + 1] -= uy;
  }
}

/**
 * Режим выбирается раз за шаг: разбор r внутри цикла (~1.3 млн пар) стоил
 * около 15 мс на шаг, поэтому горячие циклы содержат прямую арифметику.
 */
const REP_INV = 0;   // r = -1: k^2·m_u·m_v / d
const REP_INV2 = 1;  // r = -2: k^3·m_u·m_v / d^2
const REP_CONST = 2; // r = 0
const REP_POW = 3;   // произвольный r через Math.pow

function repulsionMode(): number {
  const r = params.repulsionExp;
  if (Math.abs(r + 1) < 1e-6) return REP_INV;
  if (Math.abs(r + 2) < 1e-6) return REP_INV2;
  if (Math.abs(r) < 1e-6) return REP_CONST;
  return REP_POW;
}

/**
 * Сетка с ячейкой 2k и окрестностью 3×3, построение за O(n) сортировкой
 * подсчётом. Плотные ячейки прореживаются (MAX_PER_CELL): оценка приближённая.
 */
function applyNearRepulsion(coef: number) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = px[i * 2], y = px[i * 2 + 1];
    if (x < minX) minX = x;
    if (x > maxX) maxX = x;
    if (y < minY) minY = y;
    if (y > maxY) maxY = y;
  }

  // Не влезли в лимит — растим ячейку. Обрезать cols и rows по отдельности
  // нельзя: на вытянутой укладке это схлопнет одну из осей в одну полосу.
  let cell = 2 * k;
  let cols = 0;
  let rows = 0;
  for (let attempt = 0; attempt < 24; attempt++) {
    cols = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
    rows = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
    if (cols * rows <= MAX_CELLS) break;
    cell *= 1.7;
  }
  const numCells = Math.min(cols * rows, MAX_CELLS);
  cellStart.fill(0, 0, numCells + 1);
  cursor.fill(0, 0, numCells);

  for (let i = 0; i < n; i++) {
    const gx = Math.min(cols - 1, Math.max(0, ((px[i * 2] - minX) / cell) | 0));
    const gy = Math.min(rows - 1, Math.max(0, ((px[i * 2 + 1] - minY) / cell) | 0));
    const c = gx * rows + gy;
    cellOf[i] = c;
    cellStart[c + 1]++;
  }
  for (let c = 0; c < numCells; c++) cellStart[c + 1] += cellStart[c];
  for (let i = 0; i < n; i++) {
    const c = cellOf[i];
    order[cellStart[c] + cursor[c]] = i;
    cursor[c]++;
  }

  // Потолок соседей в ОДНОЙ ячейке: 3x3 даёт до 9 x MAX_PER_CELL пар на
  // вершину. Значение 24 давало 216 пар и 35 мс на шаг вместо целевых 4 мс.
  const MAX_PER_CELL = 6;

  const mode = repulsionMode();
  const rExp = params.repulsionExp;

  for (let i = 0; i < n; i++) {
    const ix = i * 2;
    const xi = px[ix], yi = px[ix + 1];
    const mi = mass[i];
    const c = cellOf[i];
    const gx = (c / rows) | 0;
    const gy = c - gx * rows;
    let ax = 0, ay = 0;

    for (let ddx = -1; ddx <= 1; ddx++) {
      const nx = gx + ddx;
      if (nx < 0 || nx >= cols) continue;
      for (let ddy = -1; ddy <= 1; ddy++) {
        const ny = gy + ddy;
        if (ny < 0 || ny >= rows) continue;
        const nc = nx * rows + ny;
        const from = cellStart[nc];
        const to = cellStart[nc + 1];
        const count = to - from;
        const stride = count > MAX_PER_CELL ? Math.ceil(count / MAX_PER_CELL) : 1;
        const scale = coef * stride; // компенсируем прореживание
        for (let p = from; p < to; p += stride) {
          const j = order[p];
          if (j === i) continue;
          const jx = j * 2;
          const dx = xi - px[jx];
          const dy = yi - px[jx + 1];
          const d2 = dx * dx + dy * dy;
          if (d2 < EPS) continue;
          const mm = scale * mi * mass[j];
          // множитель вектора (dx, dy): |f| / d
          let f: number;
          if (mode === REP_INV) f = mm / d2;
          else if (mode === REP_INV2) f = mm / (d2 * Math.sqrt(d2));
          else if (mode === REP_CONST) f = mm / Math.sqrt(d2);
          else f = (mm * Math.pow(Math.sqrt(d2), rExp)) / Math.sqrt(d2);
          ax += dx * f;
          ay += dy * f;
        }
      }
    }
    fx[ix] += ax;
    fx[ix + 1] += ay;
  }
}

/**
 * Дальнее поле: несмещённая оценка по выборке из S вершин, общей для всех
 * вершин шага — свидетели меняются от шага к шагу, зато их координаты
 * живут в кэше L1: 0.53 мс против 2.82 мс при независимых выборках.
 */
function applyFarRepulsion(coef: number) {
  const S = Math.max(1, params.sampleS | 0);
  if (witness.length < S) witness = new Uint32Array(S);
  for (let s = 0; s < S; s++) witness[s] = (Math.random() * n) | 0;
  const scale = n / S;

  const mode = repulsionMode();
  const rExp = params.repulsionExp;
  const coefScaled = coef * scale;

  for (let i = 0; i < n; i++) {
    const ix = i * 2;
    const xi = px[ix], yi = px[ix + 1];
    const mi = mass[i];
    let ax = 0, ay = 0;
    for (let s = 0; s < S; s++) {
      const j = witness[s];
      if (j === i) continue;
      const jx = j * 2;
      const dx = xi - px[jx];
      const dy = yi - px[jx + 1];
      const d2 = dx * dx + dy * dy;
      if (d2 < EPS) continue;
      const mm = coefScaled * mi * mass[j];
      let f: number;
      if (mode === REP_INV) f = mm / d2;
      else if (mode === REP_INV2) f = mm / (d2 * Math.sqrt(d2));
      else if (mode === REP_CONST) f = mm / Math.sqrt(d2);
      else f = (mm * Math.pow(Math.sqrt(d2), rExp)) / Math.sqrt(d2);
      ax += dx * f;
      ay += dy * f;
    }
    fx[ix] += ax;
    fx[ix + 1] += ay;
  }
}

/**
 * Приводит координаты к единичному масштабу вокруг центра масс: отталкивание
 * с множителем n/S раздувает укладку (размах растёт с 1.0 до 4.1), k
 * перестаёт отвечать реальным расстояниям и сетка ближнего поля вырождается.
 * Пользователю незаметно — камера подгоняется автоматически.
 */
function renormalize() {
  let cx = 0, cy = 0;
  for (let i = 0; i < n; i++) {
    cx += px[i * 2];
    cy += px[i * 2 + 1];
  }
  cx /= n || 1;
  cy /= n || 1;
  let extent = 0;
  for (let i = 0; i < n; i++) {
    const dx = Math.abs(px[i * 2] - cx);
    const dy = Math.abs(px[i * 2 + 1] - cy);
    if (dx > extent) extent = dx;
    if (dy > extent) extent = dy;
  }
  if (extent < 1e-6) return;
  const s = 0.5 / extent;
  for (let i = 0; i < n; i++) {
    px[i * 2] = (px[i * 2] - cx) * s + 0.5;
    px[i * 2 + 1] = (px[i * 2 + 1] - cy) * s + 0.5;
    vx[i * 2] *= s;
    vx[i * 2 + 1] *= s;
  }
}

function applyGravity() {
  const g = params.gravity;
  if (g <= 0) return;
  let cx = 0, cy = 0, total = 0;
  for (let i = 0; i < n; i++) {
    cx += px[i * 2] * mass[i];
    cy += px[i * 2 + 1] * mass[i];
    total += mass[i];
  }
  cx /= total || 1;
  cy /= total || 1;
  for (let i = 0; i < n; i++) {
    fx[i * 2] -= g * (px[i * 2] - cx) * mass[i];
    fx[i * 2 + 1] -= g * (px[i * 2 + 1] - cy) * mass[i];
  }
}

/** Один шаг симуляции. Возвращает время выполнения в миллисекундах. */
function simulate(): number {
  const t0 = performance.now();
  fx.fill(0);

  applyAttraction();
  const coef = params.repulsion * Math.pow(k, 1 - params.repulsionExp);
  if (params.nearField) applyNearRepulsion(coef);
  applyFarRepulsion(coef);
  applyGravity();

  // Адаптивный шаг Ху (fastlayout.py) здесь НЕ используется: он ждёт
  // детерминированной энергии, а выборочное дальнее поле делает её случайной.
  // Правило «×0.9 при любом росте, ÷0.9 лишь после пяти подряд снижений»
  // вырождается в храповик — 2·10⁻⁹ к 400-му шагу, укладка замерзает не сойдясь.
  // Вместо него расписание остывания как в d3-force.
  const d = params.damping;
  const limit = k * MAX_DISP * alpha;
  let energy = 0;
  let maxDisp = 0;

  for (let i = 0; i < n; i++) {
    const ix = i * 2;
    if (pinned[i]) {
      vx[ix] = 0;
      vx[ix + 1] = 0;
      continue;
    }
    const gx = fx[ix], gy = fx[ix + 1];
    const norm = Math.sqrt(gx * gx + gy * gy);
    energy += norm * norm;
    if (norm < EPS) continue;
    const scale = limit / norm;
    vx[ix] = vx[ix] * d + gx * scale * (1 - d);
    vx[ix + 1] = vx[ix + 1] * d + gy * scale * (1 - d);
    px[ix] += vx[ix];
    px[ix + 1] += vx[ix + 1];
    const disp = Math.abs(vx[ix]) + Math.abs(vx[ix + 1]);
    if (disp > maxDisp) maxDisp = disp;
  }

  alpha += (alphaTarget - alpha) * ALPHA_DECAY;
  tick++;
  if (tick % RENORM_EVERY === 0) renormalize();
  if (alpha < ALPHA_MIN && maxDisp < 0.02 * k) settled = true;

  lastEnergy = energy;
  return performance.now() - t0;
}

let lastEnergy = 0;

// ── цикл и обмен сообщениями ─────────────────────────────────────

function sendFrame(tickMs: number) {
  let buf = spare.pop();
  if (!buf || buf.byteLength !== px.byteLength) buf = new ArrayBuffer(px.byteLength);
  new Float32Array(buf).set(px);
  (self as unknown as Worker).postMessage(
    { type: "frame", pos: buf, tick, tickMs, temperature: alpha, energy: lastEnergy },
    [buf],
  );
}

let scheduled = false;
function loop() {
  scheduled = false;
  if (!running || n === 0) return;
  if (settled) {
    // молчим до «встряхивания» или перетаскивания вершины
    return;
  }
  let total = 0;
  for (let i = 0; i < ticksPerFrame && !settled; i++) total += simulate();
  sendFrame(total);
  schedule();
}

function schedule() {
  if (scheduled || !running) return;
  scheduled = true;
  // setTimeout(0), а не rAF: кадров тут нет, частоту задаёт главный поток
  setTimeout(loop, 0);
}

self.onmessage = (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;

  switch (msg.type) {
    case "init": {
      n = msg.n;
      m = msg.m;
      params = { ...params, ...msg.params };
      alloc(n);
      px.set(new Float32Array(msg.pos));
      src = new Uint32Array(msg.src);
      tgt = new Uint32Array(msg.tgt);
      weight = msg.weight ? new Float32Array(msg.weight) : null;
      degreeRef = new Float32Array(msg.degree);
      recomputeMass(degreeRef);
      k = Math.sqrt(1 / Math.max(n, 1));
      alpha = 1;
      settled = false;
      tick = 0;
      running = true;
      schedule();
      break;
    }

    case "params": {
      const prevDegreeMass = params.degreeMass;
      params = { ...params, ...msg.params };
      if (params.degreeMass !== prevDegreeMass) recomputeMass(degreeRef);
      if (msg.reheat !== false) {
        alpha = Math.max(alpha, 0.6);
        settled = false;
      }
      if (running) schedule();
      break;
    }

    case "control": {
      if (msg.ticksPerFrame !== undefined) {
        ticksPerFrame = Math.max(1, Math.min(8, msg.ticksPerFrame));
      }
      if (msg.reheat !== undefined) {
        alpha = Math.max(alpha, msg.reheat);
        settled = false;
      }
      if (msg.running !== undefined) {
        running = msg.running;
        if (running) schedule();
      }
      break;
    }

    case "drag": {
      if (msg.id >= 0 && msg.id < n) {
        px[msg.id * 2] = msg.x;
        px[msg.id * 2 + 1] = msg.y;
        vx[msg.id * 2] = 0;
        vx[msg.id * 2 + 1] = 0;
        pinned[msg.id] = msg.pin ? 1 : 0;
        alpha = Math.max(alpha, 0.35);
        settled = false;
        if (running) schedule();
      }
      break;
    }

    case "release": {
      if (msg.id >= 0 && msg.id < n) pinned[msg.id] = 0;
      break;
    }

    case "recycle": {
      if (spare.length < 3) spare.push(msg.buf);
      break;
    }
  }
};
