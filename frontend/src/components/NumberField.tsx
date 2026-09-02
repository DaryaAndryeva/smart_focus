import { useState } from "react";

interface Props {
  label: string;
  value: number;
  min: number;
  max: number;
  /** Шаг для стрелок; дробный шаг задаёт и число знаков при показе. */
  step?: number;
  onChange: (v: number) => void;
  hint?: string;
  suffix?: string;
  disabled?: boolean;
}

function decimalsFor(step: number): number {
  const s = String(step);
  const dot = s.indexOf(".");
  return dot === -1 ? 0 : s.length - dot - 1;
}

/**
 * Поле ввода числа с явными границами — замена ползунку. Пока поле в фокусе,
 * показывается черновик: иначе набрать «500» при минимуме 100 невозможно, первая
 * же цифра будет поднята до минимума. Границы применяются на blur и по Enter.
 */
export default function NumberField({
  label, value, min, max, step = 1, onChange, hint, suffix, disabled,
}: Props) {
  const decimals = decimalsFor(step);
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? value.toFixed(decimals);

  const commit = () => {
    if (draft === null) return;
    const parsed = Number(draft.replace(",", "."));
    const next = Number.isFinite(parsed) && draft.trim() !== ""
      ? Math.min(max, Math.max(min, parsed))
      : value;
    setDraft(null);
    if (next !== value) onChange(+next.toFixed(decimals));
  };

  const nudge = (dir: 1 | -1) => {
    const base = draft !== null ? Number(draft.replace(",", ".")) : value;
    const from = Number.isFinite(base) ? base : value;
    const next = +Math.min(max, Math.max(min, from + dir * step)).toFixed(decimals);
    setDraft(next.toFixed(decimals));
    if (next !== value) onChange(next);
  };

  // Минус типографский, а разделитель с пробелами: «-2.0–-0.2» читается как каша.
  const fmt = (x: number) => x.toFixed(decimals).replace("-", "−");

  return (
    <label className="number-field">
      <span className="number-field-head">
        {label}
        <span className="number-field-range">{fmt(min)} … {fmt(max)}</span>
      </span>
      <span className="number-field-body">
        <input
          type="number"
          inputMode="decimal"
          min={min}
          max={max}
          step={step}
          value={shown}
          disabled={disabled}
          onFocus={() => setDraft(value.toFixed(decimals))}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commit}
          onKeyDown={(e) => {
            // Enter применяет значение сам: поле может остаться активным, blur не случится.
            if (e.key === "Enter") { commit(); e.currentTarget.blur(); }
            else if (e.key === "ArrowUp") { e.preventDefault(); nudge(1); }
            else if (e.key === "ArrowDown") { e.preventDefault(); nudge(-1); }
          }}
        />
        {suffix && <span className="number-field-suffix">{suffix}</span>}
      </span>
      {hint && <span className="hint number-field-hint">{hint}</span>}
    </label>
  );
}
