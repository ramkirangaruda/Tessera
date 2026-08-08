import { useState } from "react";
import "./App.css";
import { HeaderStrip } from "./components/HeaderStrip";
import { MemoryBudgetBar } from "./components/MemoryBudgetBar";
import { AllocationHeatmap } from "./components/AllocationHeatmap";
import { ParetoPlot } from "./components/ParetoPlot";
import { useDaemonSocket } from "./useDaemonSocket";
import { MOCK_DAEMON_STATE_BY_DOMAIN, MOCK_PARETO_POINTS } from "./mockData";

function App() {
  const { state: liveState, live, lastEvent } = useDaemonSocket();
  const [mockDomain, setMockDomain] = useState<"code" | "math" | "chat">("code");

  // While no daemon is connected, let the demo rig cycle mock domains manually to preview the
  // heatmap redistributing (spec §12 step 5) without a live handshake.
  const state = live ? liveState : MOCK_DAEMON_STATE_BY_DOMAIN[mockDomain];

  return (
    <div className="app">
      <HeaderStrip state={state} live={live} />

      {!live && (
        <div className="mock-banner">
          No daemon connected — showing mock data.{" "}
          <div className="mock-banner__buttons">
            {(["code", "math", "chat"] as const).map((d) => (
              <button
                key={d}
                className={d === mockDomain ? "mock-btn mock-btn--active" : "mock-btn"}
                onClick={() => setMockDomain(d)}
              >
                {d}
              </button>
            ))}
          </div>
        </div>
      )}

      {lastEvent === "wiped" && <div className="wipe-banner">session wiped ✓</div>}

      <div className="grid">
        <MemoryBudgetBar state={state} />
        <AllocationHeatmap state={state} />
        <ParetoPlot points={MOCK_PARETO_POINTS} activeProfileId={state.profile_id} />
      </div>
    </div>
  );
}

export default App;
