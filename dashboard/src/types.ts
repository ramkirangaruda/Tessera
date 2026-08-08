// Mirrors daemon/server.py::DaemonState.to_dict() — keep in sync.
export interface DaemonState {
  connected: boolean;
  device_name: string | null;
  domain: string | null;
  profile_id: string | null;
  footprint_bytes: number | null;
  ram_free_bytes: number | null;
  allocation: Record<string, number>; // tensor_name -> bits
  wiped: boolean;
}

export interface DaemonEvent {
  event: "connected" | "wiped" | "domain_switch";
  state: DaemonState;
}

// spec §4.1: the four calibration domains
export type Domain = "chat" | "code" | "math" | "summ";

// spec §3: 7 per-layer tensor kinds x 28 layers, plus token_embd
export const PER_LAYER_TENSOR_KINDS = [
  "attn_q",
  "attn_k",
  "attn_v",
  "attn_o",
  "mlp_gate",
  "mlp_up",
  "mlp_down",
] as const;
export const N_LAYERS = 28;

// One point on the quality-vs-memory Pareto plot (spec §8 panel 3).
export interface ParetoPoint {
  label: string; // e.g. "Q4" (uniform) or "code@1100MB" (a Tessera profile)
  memory_mb: number;
  quality: number; // higher is better (e.g. -perplexity, or pass@1)
  kind: "uniform" | "tessera";
}
