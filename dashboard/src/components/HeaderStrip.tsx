import type { DaemonState } from "../types";

function formatBytes(bytes: number | null): string {
  if (bytes == null) return "—";
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${(bytes / 1024 ** 2).toFixed(0)} MB`;
}

export function HeaderStrip({ state, live }: { state: DaemonState; live: boolean }) {
  const bits = state.allocation.token_embd ?? "—";

  return (
    <header className="header-strip">
      <div className="header-strip__title">Tessera</div>
      <div className="header-strip__fields">
        <Field label="device" value={state.device_name ?? "none"} />
        <Field label="domain" value={state.domain ?? "—"} accent />
        <Field label="profile" value={`~${bits}-bit`} />
        <Field label="footprint" value={formatBytes(state.footprint_bytes)} />
        <span className={`status-dot ${state.wiped ? "status-dot--wiped" : "status-dot--live"}`} />
        <span className="header-strip__conn">{live ? "connected" : "mock data"}</span>
      </div>
    </header>
  );
}

function Field({ label, value, accent }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className="header-strip__field">
      <span className="header-strip__label">{label}</span>
      <span className={accent ? "header-strip__value header-strip__value--accent" : "header-strip__value"}>
        {value}
      </span>
    </div>
  );
}
