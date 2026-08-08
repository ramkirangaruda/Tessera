import {
  CartesianGrid,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ParetoPoint } from "../types";

// spec §8 panel 3: uniform-quantization baseline as a line, current Tessera profile marked as
// a point that should sit above the line. Animates on profile switch (points re-render on
// `points` prop change — Recharts handles the transition).
export function ParetoPlot({ points, activeProfileId }: { points: ParetoPoint[]; activeProfileId: string | null }) {
  const uniform = points.filter((p) => p.kind === "uniform").sort((a, b) => a.memory_mb - b.memory_mb);
  const tessera = points.filter((p) => p.kind === "tessera");

  return (
    <section className="panel">
      <h2 className="panel__title">Pareto: quality vs memory</h2>
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid stroke="var(--grid-color, #333)" strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="memory_mb"
            name="memory"
            unit="MB"
            stroke="var(--axis-color, #999)"
            label={{ value: "memory (MB)", position: "insideBottom", offset: -10, fill: "var(--axis-color, #999)" }}
          />
          <YAxis
            type="number"
            dataKey="quality"
            name="quality"
            stroke="var(--axis-color, #999)"
            label={{ value: "quality", angle: -90, position: "insideLeft", fill: "var(--axis-color, #999)" }}
          />
          <Tooltip
            cursor={{ strokeDasharray: "3 3" }}
            content={({ active, payload }) => {
              if (!active || !payload?.length) return null;
              const p = payload[0].payload as ParetoPoint;
              return (
                <div className="pareto-tooltip">
                  <strong>{p.label}</strong>
                  <div>{p.memory_mb} MB</div>
                  <div>quality {p.quality.toFixed(2)}</div>
                </div>
              );
            }}
          />
          <Legend />
          <Line
            data={uniform}
            dataKey="quality"
            name="uniform baseline (Q2/Q3/Q4/Q6/Q8)"
            stroke="#7a7a7a"
            dot={{ r: 4 }}
            legendType="line"
          />
          <Scatter
            data={tessera}
            name="Tessera profiles"
            fill="#4bb89e"
            shape={(props: any) => {
              const isActive = props.payload?.label === activeProfileId;
              return (
                <circle
                  cx={props.cx}
                  cy={props.cy}
                  r={isActive ? 8 : 5}
                  fill={isActive ? "#e8e356" : "#4bb89e"}
                  stroke={isActive ? "#fff" : "none"}
                  strokeWidth={2}
                />
              );
            }}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </section>
  );
}
