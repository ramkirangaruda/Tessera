import type { DaemonState } from "../types";

// spec §8 panel 1: device free RAM, model footprint, headroom. Updates on profile switch.
export function MemoryBudgetBar({ state }: { state: DaemonState }) {
  const ramFree = state.ram_free_bytes ?? 0;
  const footprint = state.footprint_bytes ?? 0;
  const headroom = Math.max(ramFree - footprint, 0);
  const usedPct = ramFree > 0 ? Math.min((footprint / ramFree) * 100, 100) : 0;

  const gb = (b: number) => (b / 1024 ** 3).toFixed(2);

  return (
    <section className="panel">
      <h2 className="panel__title">Memory budget</h2>
      <div className="memory-bar">
        <div className="memory-bar__track">
          <div
            className="memory-bar__fill"
            style={{ width: `${usedPct}%` }}
            data-testid="memory-bar-fill"
          />
        </div>
        <div className="memory-bar__legend">
          <span>
            <strong>{gb(footprint)} GB</strong> model
          </span>
          <span>
            <strong>{gb(headroom)} GB</strong> headroom
          </span>
          <span>
            <strong>{gb(ramFree)} GB</strong> free on device
          </span>
        </div>
      </div>
    </section>
  );
}
