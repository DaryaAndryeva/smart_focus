import { useEffect, useState } from "react";
import NumberField from "./NumberField";
import type { SimParams, LiveStats } from "../useLiveLayout";

interface Props {
  enabled: boolean;
  onEnabledChange: (v: boolean) => void;
  params: SimParams;
  onParamsChange: (patch: Partial<SimParams>) => void;
  onReheat: () => void;
  statsRef: React.RefObject<LiveStats>;
}

function Slider({
  label, value, min, max, step, onChange, hint, disabled,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
  disabled?: boolean;
}) {
  return (
    <NumberField
      label={label}
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={onChange}
      hint={hint}
      disabled={disabled}
    />
  );
}

/** Название модели сил по показателям степени в классификации Ноака. */
function forceModelName(a: number, r: number): string {
  const d = a - r;
  if (Math.abs(a) < 0.15 && Math.abs(r + 1) < 0.15) return "LinLog — максимальное разделение кластеров";
  if (Math.abs(a - 1) < 0.15 && Math.abs(r + 1) < 0.15) return "линейная";
  if (Math.abs(a - 2) < 0.15 && Math.abs(r + 1) < 0.15) return "Фрухтерман–Рейнгольд — кластеры сливаются";
  return `разность показателей a−r = ${d.toFixed(1)}: чем меньше, тем сильнее разделяются кластеры`;
}

export default function PhysicsPanel({
  enabled,
  onEnabledChange,
  params,
  onParamsChange,
  onReheat,
  statsRef,
}: Props) {
  const [stats, setStats] = useState<LiveStats>({ tick: 0, tickMs: 0, temperature: 0, energy: 0 });
  const [open, setOpen] = useState(true);
  // ALPHA_MIN воркера равен 0.02: ниже него симуляция встаёт и перестаёт слать кадры
  const settled = stats.tick > 0 && stats.temperature <= 0.02;

  // 4 раза в секунду: обновление на каждом кадре — это 60 лишних перерисовок React в секунду
  useEffect(() => {
    const id = setInterval(() => {
      if (statsRef.current) setStats({ ...statsRef.current });
    }, 250);
    return () => clearInterval(id);
  }, [statsRef]);

  return (
    <section className="collapsible-section">
      <button className="section-toggle" onClick={() => setOpen(!open)}>
        Физика укладки <span>{open ? "⌄" : "›"}</span>
      </button>

      {open && (
        <div className="section-body">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={enabled}
              onChange={(e) => onEnabledChange(e.target.checked)}
            />
            Живая симуляция
          </label>

          {enabled && (
            <p className="hint">
              {stats.tickMs.toFixed(1)} мс/шаг · шаг {stats.tick} · температура{" "}
              {stats.temperature.toFixed(2)}
            </p>
          )}

          <div className="layout-grid" style={{ marginBottom: 8 }}>
            <button className="btn-primary" style={{ fontSize: 12 }} onClick={onReheat}>
              Встряхнуть
            </button>
            <button
              className="btn-secondary"
              style={{ fontSize: 12, marginTop: 0 }}
              onClick={() => onEnabledChange(!enabled)}
            >
              {enabled ? "Пауза" : "Продолжить"}
            </button>
          </div>

          <h4 className="sub-head">Модель сил</h4>
          <p className="hint">{forceModelName(params.attractionExp, params.repulsionExp)}</p>
          {!enabled && (
            <p className="hint hint-warn">
              Параметры действуют только при включённой живой симуляции:
              сейчас показана укладка, посчитанная на сервере.
            </p>
          )}
          {enabled && settled && (
            <p className="hint">
              Укладка сошлась. После смены параметра она пересчитается сама;
              если картинка почти не меняется — нажмите «Встряхнуть».
            </p>
          )}

          <Slider
            label="Показатель притяжения a"
            value={params.attractionExp}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => onParamsChange({ attractionExp: v })}
            hint="0 — LinLog, 1 — линейное, 2 — Фрухтерман–Рейнгольд"
            disabled={!enabled}
          />
          <Slider
            label="Показатель отталкивания r"
            value={params.repulsionExp}
            min={-2}
            max={-0.2}
            step={0.1}
            onChange={(v) => onParamsChange({ repulsionExp: v })}
            disabled={!enabled}
          />
          <Slider
            label="Сила отталкивания"
            value={params.repulsion}
            min={0.1}
            max={5}
            step={0.1}
            onChange={(v) => onParamsChange({ repulsion: v })}
            disabled={!enabled}
          />
          <Slider
            label="Сила притяжения"
            value={params.attraction}
            min={0.1}
            max={5}
            step={0.1}
            onChange={(v) => onParamsChange({ attraction: v })}
            disabled={!enabled}
          />
          <Slider
            label="Гравитация"
            value={params.gravity}
            min={0}
            max={0.5}
            step={0.01}
            onChange={(v) => onParamsChange({ gravity: v })}
            hint="удерживает компоненты связности вместе"
            disabled={!enabled}
          />
          <Slider
            label="Плавность движения"
            value={params.damping}
            min={0}
            max={0.95}
            step={0.05}
            onChange={(v) => onParamsChange({ damping: v }, )}
            hint="выше — вершины плавнее «плывут»"
            disabled={!enabled}
          />
          <Slider
            label="Влияние веса ребра"
            value={params.weightInfluence}
            min={0}
            max={2}
            step={0.1}
            onChange={(v) => onParamsChange({ weightInfluence: v })}
            disabled={!enabled}
          />

          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={params.degreeMass}
              onChange={(e) => onParamsChange({ degreeMass: e.target.checked })}
            />
            Масса по степени (хабы отталкивают сильнее)
          </label>
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={params.nearField}
              onChange={(e) => onParamsChange({ nearField: e.target.checked })}
            />
            Ближнее поле по сетке
          </label>
          <Slider
            label="Выборка дальнего поля S"
            value={params.sampleS}
            min={4}
            max={48}
            step={2}
            onChange={(v) => onParamsChange({ sampleS: v })}
            hint="больше — точнее, но дороже каждый шаг"
            disabled={!enabled}
          />

        </div>
      )}
    </section>
  );
}
