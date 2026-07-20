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

  it("should prove managed model runtime through tauri ipc", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      configurable: true,
      value: {},
    });
    vi.mocked(invoke).mockResolvedValue({
      trace_id: "trace-managed-proof",
      downloaded: true,
      activated: true,
      generated: true,
      failure_reasons: [],
      runtime_probe: {
        passed: true,
        generated_preview: "CRYTEX_PROBE_OK",
      },
    });

    const sink = vi.fn();
    const api = createCrytexApi(sink);
    await api.proveManagedModelRuntime({
      model_id: "local-qwen",
      trace_id: "trace-managed-proof",
      max_tokens: 16,
      timeout_seconds: 5,
    });

    expect(invoke).toHaveBeenCalledWith("prove_managed_model_runtime", {
      request: {
        model_id: "local-qwen",
        trace_id: "trace-managed-proof",
        max_tokens: 16,
        timeout_seconds: 5,
      },
    });
    expect(sink).toHaveBeenCalledWith(expect.objectContaining({
      name: "prove_managed_model_runtime",
    }));
  });

  it("should trigger lora adapter training through tauri ipc", async () => {
    Object.defineProperty(window, "__TAURI_INTERNALS__", {
      configurable: true,
      value: {},
    });
    vi.mocked(invoke).mockResolvedValue({
      adapter: {
        id: "codegen-v2",
        file_path: "adapters/codegen/codegen-v2",
        base_model: "mistral-7b",
        task_kind: "codegen",
        agent_role: null,
        active: true,
      },
      promoted: true,
      benchmark_gate: {
        accepted: true,
      },
      metrics: {},
    });

    const sink = vi.fn();
    const api = createCrytexApi(sink);
    await api.trainLoraAdapter({
      task_kind: "codegen",
      agent_role: null,
    });

    expect(invoke).toHaveBeenCalledWith("train_lora_adapter", {
      request: {
        task_kind: "codegen",
        agent_role: null,
      },
    });
    expect(sink).toHaveBeenCalledWith(expect.objectContaining({
      name: "train_lora_adapter",
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
