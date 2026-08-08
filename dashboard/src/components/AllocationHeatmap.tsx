import { useState } from "react";
import type { DaemonState } from "../types";
import { N_LAYERS, PER_LAYER_TENSOR_KINDS } from "../types";

// spec §8 panel 2: "the single most important visual in the project." 28 layers x 7 tensor
// kinds, each cell colored by assigned bit-width, plus a standalone token_embd cell.
const BIT_MIN = 2;
const BIT_MAX = 8;

// Sequential scale, low bits (aggressive compression) -> high bits (rich). Distinct steps so
// adjacent bit-widths are still told apart at a glance, not just a smooth gradient.
const BIT_COLORS: Record<number, string> = {
  2: "#1a2f4d",
  3: "#1e4a7a",
  4: "#1f6fa8",
  5: "#2b93b8",
  6: "#4bb89e",
  7: "#8fce6a",
  8: "#e8e356",
};

function colorForBits(bits: number | undefined): string {
  if (bits == null) return "#2a2a2a";
  const clamped = Math.max(BIT_MIN, Math.min(BIT_MAX, Math.round(bits)));
  return BIT_COLORS[clamped] ?? "#2a2a2a";
}

export function AllocationHeatmap({ state }: { state: DaemonState }) {
  const [hovered, setHovered] = useState<{ name: string; bits: number } | null>(null);
  const alloc = state.allocation;

  return (
    <section className="panel">
      <h2 className="panel__title">Allocation heatmap</h2>
      <div className="heatmap">
        <div className="heatmap__grid">
          <div className="heatmap__corner" />
          {PER_LAYER_TENSOR_KINDS.map((kind) => (
            <div key={kind} className="heatmap__col-label">
              {kind}
            </div>
          ))}
          {Array.from({ length: N_LAYERS }, (_, layer) => (
            <RowCells
              key={layer}
              layer={layer}
              alloc={alloc}
              onHover={setHovered}
            />
          ))}
        </div>

        <div className="heatmap__embd">
          <div className="heatmap__col-label">token_embd</div>
          <div
            className="heatmap__cell heatmap__cell--embd"
            style={{ background: colorForBits(alloc.token_embd) }}
            onMouseEnter={() => setHovered({ name: "token_embd", bits: alloc.token_embd })}
            onMouseLeave={() => setHovered(null)}
          />
        </div>
      </div>

      <div className="heatmap__footer">
        <div className="heatmap__legend">
          {[2, 3, 4, 5, 6, 7, 8].map((b) => (
            <span key={b} className="heatmap__legend-item">
              <span className="heatmap__legend-swatch" style={{ background: BIT_COLORS[b] }} />
              {b}b
            </span>
          ))}
        </div>
        <div className="heatmap__tooltip" aria-live="polite">
          {hovered ? `${hovered.name} — ${hovered.bits}-bit` : "hover a cell"}
        </div>
      </div>
    </section>
  );
}

function RowCells({
  layer,
  alloc,
  onHover,
}: {
  layer: number;
  alloc: Record<string, number>;
  onHover: (v: { name: string; bits: number } | null) => void;
}) {
  return (
    <>
      <div className="heatmap__row-label">L{layer}</div>
      {PER_LAYER_TENSOR_KINDS.map((kind) => {
        const name = `blk.${layer}.${kind}`;
        const bits = alloc[name];
        return (
          <div
            key={kind}
            className="heatmap__cell"
            style={{ background: colorForBits(bits) }}
            onMouseEnter={() => onHover({ name, bits })}
            onMouseLeave={() => onHover(null)}
          />
        );
      })}
    </>
  );
}
