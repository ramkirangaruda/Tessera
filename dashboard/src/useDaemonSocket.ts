import { useEffect, useRef, useState } from "react";
import type { DaemonEvent, DaemonState } from "./types";
import { MOCK_DAEMON_STATE } from "./mockData";

const WS_URL = import.meta.env.VITE_DAEMON_WS_URL ?? "ws://localhost:8420/ws";

/** Live daemon state over the websocket (daemon/server.py `/ws`), falling back to mock data
 * when no daemon is reachable so the dashboard is developable/demoable standalone. */
export function useDaemonSocket(): { state: DaemonState; live: boolean; lastEvent: DaemonEvent["event"] | null } {
  const [state, setState] = useState<DaemonState>(MOCK_DAEMON_STATE);
  const [live, setLive] = useState(false);
  const [lastEvent, setLastEvent] = useState<DaemonEvent["event"] | null>(null);
  const retryRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;

    function connect() {
      if (cancelled) return;
      ws = new WebSocket(WS_URL);

      ws.onopen = () => setLive(true);

      ws.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          if ("event" in data) {
            const evt = data as DaemonEvent;
            setState(evt.state);
            setLastEvent(evt.event);
          } else {
            setState(data as DaemonState);
          }
        } catch {
          // ignore malformed frame
        }
      };

      ws.onclose = () => {
        setLive(false);
        if (!cancelled) {
          retryRef.current = window.setTimeout(connect, 3000);
        }
      };

      ws.onerror = () => {
        ws?.close();
      };
    }

    connect();
    return () => {
      cancelled = true;
      if (retryRef.current) window.clearTimeout(retryRef.current);
      ws?.close();
    };
  }, []);

  return { state, live, lastEvent };
}
