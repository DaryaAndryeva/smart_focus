import { useEffect, useRef, useState, useCallback } from "react";
import type { SimParams } from "./workers/layout.worker";
import { DEFAULT_PARAMS } from "./workers/layout.worker";

export type { SimParams };
export { DEFAULT_PARAMS };

export interface LiveStats {
  tick: number;
  tickMs: number;
  temperature: number;
  energy: number;
}

export interface LiveGraph {
  n: number;
  /** стартовые координаты (2n) в нормализованных единицах */
  pos: Float32Array;
  src: Uint32Array;
  tgt: Uint32Array;
  weight: Float32Array | null;
  /** степени вершин — используются как массы */
  degree: Float32Array;
}

type FrameCb = (pos: Float32Array) => void;

/**
 * Фоновая силовая симуляция в воркере: свежие координаты уходят подписчикам
 * onFrame мимо состояния React, иначе компонент перерисовывался бы 60 раз в секунду.
 */
export function useLiveLayout(graph: LiveGraph | null, enabled: boolean) {
  const workerRef = useRef<Worker | null>(null);
  const subscribers = useRef(new Set<FrameCb>());
  const statsRef = useRef<LiveStats>({ tick: 0, tickMs: 0, temperature: 0, energy: 0 });
  const paramsRef = useRef<SimParams>({ ...DEFAULT_PARAMS });
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const worker = new Worker(new URL("./workers/layout.worker.ts", import.meta.url), {
      type: "module",
    });
    worker.onmessage = (ev: MessageEvent) => {
      const msg = ev.data;
      if (msg.type !== "frame") return;
      const pos = new Float32Array(msg.pos);
      statsRef.current = {
        tick: msg.tick,
        tickMs: msg.tickMs,
        temperature: msg.temperature,
        energy: msg.energy,
      };
      for (const cb of subscribers.current) cb(pos);
      worker.postMessage({ type: "recycle", buf: msg.pos }, [msg.pos]);
    };
    workerRef.current = worker;
    return () => {
      worker.terminate();
      workerRef.current = null;
    };
  }, []);

  useEffect(() => {
    const worker = workerRef.current;
    if (!worker || !graph) {
      setReady(false);
      return;
    }
    // копии: буферы уходят воркеру по transfer и отсоединяются у вызывающего
    const pos = graph.pos.slice();
    const src = graph.src.slice();
    const tgt = graph.tgt.slice();
    const weight = graph.weight ? graph.weight.slice() : null;
    const degree = graph.degree.slice();

    const transfer: Transferable[] = [pos.buffer, src.buffer, tgt.buffer, degree.buffer];
    if (weight) transfer.push(weight.buffer);

    worker.postMessage(
      {
        type: "init",
        n: graph.n,
        m: graph.src.length,
        pos: pos.buffer,
        src: src.buffer,
        tgt: tgt.buffer,
        weight: weight ? weight.buffer : null,
        degree: degree.buffer,
        params: paramsRef.current,
      },
      transfer,
    );
    worker.postMessage({ type: "control", running: enabled });
    setReady(true);
    // enabled меняется отдельным эффектом, здесь только начальное значение
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [graph]);

  useEffect(() => {
    workerRef.current?.postMessage({ type: "control", running: enabled });
  }, [enabled]);

  const onFrame = useCallback((cb: FrameCb) => {
    subscribers.current.add(cb);
    return () => {
      subscribers.current.delete(cb);
    };
  }, []);

  const setParams = useCallback((patch: Partial<SimParams>, reheat = true) => {
    paramsRef.current = { ...paramsRef.current, ...patch };
    workerRef.current?.postMessage({ type: "params", params: patch, reheat });
  }, []);

  const reheat = useCallback((t = 1) => {
    workerRef.current?.postMessage({ type: "control", reheat: t });
  }, []);

  const drag = useCallback((id: number, x: number, y: number, pin: boolean) => {
    workerRef.current?.postMessage({ type: "drag", id, x, y, pin });
  }, []);

  const release = useCallback((id: number) => {
    workerRef.current?.postMessage({ type: "release", id });
  }, []);

  const setTicksPerFrame = useCallback((t: number) => {
    workerRef.current?.postMessage({ type: "control", ticksPerFrame: t });
  }, []);

  return { ready, onFrame, setParams, reheat, drag, release, setTicksPerFrame, statsRef, paramsRef };
}
