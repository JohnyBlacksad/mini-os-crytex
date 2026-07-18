import { describe, expect, it, vi } from "vitest";
import { createCrytexApi } from "./api";
import type { BackendEvent } from "./ui-types";
import { invoke } from "@tauri-apps/api/core";

const listenMock = vi.hoisted(() => vi.fn());

vi.mock("@tauri-apps/api/event", () => ({
  listen: listenMock,
}));

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

describe("createCrytexApi", () => {
  it("should request run diagnostics through tauri ipc", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      configurable: true,
      value: {},
    });
    vi.mocked(invoke).mockResolvedValue({
      project_id: "project-1",
      run_id: "run-1",
      trace_ids: ["trace-1"],
      runtime: {},
      tasks: [],
      events: [],
      review_task_ids: [],
      critic_feedback: [],
      remediation_events: [],
      rag_context_sent_to_model: false,
      human_reward_recorded: false,
    });

    const sink = vi.fn();
    const api = createCrytexApi(sink);
    await api.exportRunDiagnostics({
      project_id: "project-1",
      run_id: "run-1",
      trace_id: null,
    });

    expect(invoke).toHaveBeenCalledWith("export_run_diagnostics", {
      request: {
        project_id: "project-1",
        run_id: "run-1",
        trace_id: null,
      },
    });
    expect(sink).toHaveBeenCalledWith(expect.objectContaining({
      name: "export_run_diagnostics",
    }));
  });

  it("should forward backend event payloads from tauri event stream", async () => {
    const unlisten = vi.fn();
    const backendEvent: BackendEvent = {
      type: "RunObserved",
      project_id: "project-1",
      payload: {
        action: "task_execution_started",
        task_id: "task-1",
        trace_id: "trace-1",
      },
    };

    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      configurable: true,
      value: {},
    });
    listenMock.mockImplementation(async (_eventName, handler) => {
      handler({ payload: backendEvent });
      return unlisten;
    });

    const handler = vi.fn();
    const api = createCrytexApi(vi.fn());
    const unsubscribe = await api.subscribeBackendEvents(handler);

    expect(listenMock).toHaveBeenCalledWith("crytex://event", expect.any(Function));
    expect(handler).toHaveBeenCalledWith(backendEvent);

    unsubscribe();
    expect(unlisten).toHaveBeenCalledOnce();
  });
});
