export type MemorySystemName = "working" | "episodic" | "semantic" | "procedural";

export const MEMORY_SYSTEMS: readonly MemorySystemName[] = [
  "working",
  "episodic",
  "semantic",
  "procedural",
];

/** Caller-facing request. Every serialized request uses the production `full` path. */
export interface CivilizationRequestInit {
  text: string;
  answerOptions: readonly string[];
  requestId?: string;
  sessionId?: string;
  memoryItems?: readonly string[];
  ruleItems?: readonly string[];
  stateValues?: readonly number[];
  taskName?: string;
  seed?: number;
  readMemory?: boolean;
  writeMemory?: boolean;
  includeGlobalMemory?: boolean;
  allowResolvedGlobalMemory?: boolean;
  readouts?: readonly string[];
}

/** Wire-format request payload as accepted by the service. */
export interface CivilizationRequestPayload {
  id: string;
  text: string;
  memory_items: string[];
  rule_items: string[];
  state_values: number[];
  answer_options: string[];
  task_name: string;
  seed: number;
  controls: string[];
  readouts: string[];
  orion_memory: {
    session_id: string;
    read: boolean;
    write: boolean;
    include_global: boolean;
    allow_resolved_global: boolean;
  };
}

export interface Prediction {
  requestId: string;
  optionId: number;
  optionText: string;
  scores: Record<string, number[]>;
  trace: Record<string, unknown>;
  memoryTrace: Record<string, unknown>;
  raw: Record<string, unknown>;
}

export interface Job {
  jobId: string;
  status: string;
  result: Record<string, unknown> | null;
  error: Record<string, unknown> | null;
  raw: Record<string, unknown>;
}

export interface MemoryCell {
  cell_id?: string;
  memory_system?: string;
  content?: string;
  summary?: string;
  confidence?: number;
  importance?: number;
  [key: string]: unknown;
}

/**
 * Declared runtime capabilities, in the wire shape the service reports.
 * `adapter_execution` and `hidden_states` are false for text-only runtimes.
 */
export interface RuntimeCapabilities {
  kind: string;
  label: string;
  provider_agnostic: boolean;
  local_weights: boolean;
  chat_completions: boolean;
  hidden_states: boolean;
  adapter_execution: boolean;
  ablation_controls: boolean;
  production_full_mode_only: boolean;
  notes: string[];
}

export interface CapabilitiesResponse {
  status: string;
  runtime_label: string | null;
  capabilities: RuntimeCapabilities | null;
}

export interface HealthResponse {
  stage?: string;
  status?: string;
  ready?: boolean;
  [key: string]: unknown;
}

export interface ClientOptions {
  token?: string;
  tokenEnv?: string;
  timeoutMs?: number;
  allowInsecureHttp?: boolean;
  fetch?: typeof fetch;
  userAgent?: string;
}

export interface DownloadResult {
  path: string;
  bytes: number;
  sha256: string;
  verified: boolean;
}

export interface WriteMemoryOptions {
  sessionId: string;
  memorySystem: MemorySystemName;
  content: string;
  summary?: string;
  confidence?: number;
  importance?: number;
  ttlSeconds?: number;
  metadata?: Record<string, unknown>;
}

export interface ReadMemoryOptions {
  sessionId: string;
  query: string;
  memorySystem?: MemorySystemName;
  includeExpired?: boolean;
  limit?: number;
}
