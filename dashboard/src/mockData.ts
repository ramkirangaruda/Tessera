import type { DaemonState, ParetoPoint } from "./types";
import { N_LAYERS, PER_LAYER_TENSOR_KINDS } from "./types";

// Used when the daemon (daemon/server.py) isn't reachable, so the dashboard is still developable
// and demoable standalone. Shapes mirror what the real daemon/eval pipeline will eventually emit.

function mockAllocation(seed: number): Record<string, number> {
  const alloc: Record<string, number> = {};
  const rand = mulberry32(seed);
  alloc["token_embd"] = 6;
  for (let layer = 0; layer < N_LAYERS; layer++) {
    for (const kind of PER_LAYER_TENSOR_KINDS) {
      // MLP tensors (spec §3: ~88% of per-layer weight) skew richer in the mock, mirroring the
      // allocator's expected real behavior of spending bytes where bytes actually live.
      const isMlp = kind.startsWith("mlp_");
      const base = isMlp ? 5 : 3;
      const jitter = Math.floor(rand() * 3);
      alloc[`blk.${layer}.${kind}`] = Math.max(2, Math.min(8, base + jitter));
    }
  }
  return alloc;
}

function mulberry32(seed: number) {
  let t = seed;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export const MOCK_DAEMON_STATE: DaemonState = {
  connected: true,
  device_name: "Laptop (dev machine)",
  domain: "code",
  profile_id: "code@1100MB",
  footprint_bytes: 1_149_203_968,
  ram_free_bytes: 2_147_483_648,
  allocation: mockAllocation(7),
  wiped: false,
};

export const MOCK_DAEMON_STATE_BY_DOMAIN: Record<string, DaemonState> = {
  code: MOCK_DAEMON_STATE,
  math: {
    ...MOCK_DAEMON_STATE,
    domain: "math",
    profile_id: "math@1100MB",
    allocation: mockAllocation(13),
  },
  chat: {
    ...MOCK_DAEMON_STATE,
    domain: "chat",
    profile_id: "chat@1100MB",
    device_name: "Raspberry Pi Zero 2W",
    footprint_bytes: 380_000_000,
    ram_free_bytes: 512_000_000,
    allocation: mockAllocation(21),
  },
};

// Placeholder for eval/figures.py output (spec §13: "Pareto: Tessera vs uniform Q2/Q3/Q4/Q6/Q8").
export const MOCK_PARETO_POINTS: ParetoPoint[] = [
  { label: "Q2", memory_mb: 420, quality: 0.31, kind: "uniform" },
  { label: "Q3", memory_mb: 610, quality: 0.48, kind: "uniform" },
  { label: "Q4", memory_mb: 810, quality: 0.62, kind: "uniform" },
  { label: "Q6", memory_mb: 1180, quality: 0.79, kind: "uniform" },
  { label: "Q8", memory_mb: 1540, quality: 0.88, kind: "uniform" },
  { label: "code@1100MB", memory_mb: 1096, quality: 0.74, kind: "tessera" },
  { label: "math@900MB", memory_mb: 900, quality: 0.63, kind: "tessera" },
  { label: "chat@380MB", memory_mb: 380, quality: 0.4, kind: "tessera" },
];
