import { createHash } from "node:crypto";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { dirname } from "node:path";

import {
  CivilizationAPIError,
  CivilizationError,
  CivilizationInferenceError,
  CivilizationTransportError,
} from "./errors.js";
import { buildRequestPayload, jobFromPayload, predictionFromRow } from "./request.js";
import type {
  CapabilitiesResponse,
  CivilizationRequestInit,
  ClientOptions,
  DownloadResult,
  HealthResponse,
  Job,
  MemoryCell,
  MemorySystemName,
  Prediction,
  ReadMemoryOptions,
  WriteMemoryOptions,
} from "./types.js";

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);
const DEFAULT_TIMEOUT_MS = 120_000;
const USER_AGENT = "astreusn-civilization-transformer-sdk/0.00.08";

type JsonRecord = Record<string, unknown>;

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Client for the Civilization decision, memory, job, and export APIs.
 *
 * The SDK is runtime-agnostic: it talks to a service endpoint and reports the
 * runtime capabilities that endpoint declares, instead of assuming any
 * particular model family behind it. The bearer token is never included in
 * error messages or in the client's string form.
 */
export class CivilizationClient {
  readonly baseUrl: string;
  readonly timeoutMs: number;

  private readonly token?: string;
  private readonly tokenEnv?: string;
  private readonly allowInsecureHttp: boolean;
  private readonly fetchImpl: typeof fetch;
  private readonly userAgent: string;

  constructor(baseUrl: string, options: ClientOptions = {}) {
    let parsed: URL;
    try {
      parsed = new URL(baseUrl);
    } catch {
      throw new Error("baseUrl must be an absolute HTTP(S) URL without credentials, query, or fragment");
    }
    if (
      (parsed.protocol !== "http:" && parsed.protocol !== "https:") ||
      parsed.username !== "" ||
      parsed.password !== "" ||
      parsed.search !== "" ||
      parsed.hash !== ""
    ) {
      throw new Error("baseUrl must be an absolute HTTP(S) URL without credentials, query, or fragment");
    }
    this.allowInsecureHttp = options.allowInsecureHttp ?? false;
    if (parsed.protocol === "http:" && !LOOPBACK_HOSTS.has(parsed.hostname) && !this.allowInsecureHttp) {
      throw new Error("plain HTTP is allowed only for loopback addresses unless allowInsecureHttp is true");
    }
    if (options.token !== undefined && options.tokenEnv !== undefined) {
      throw new Error("provide token or tokenEnv, not both");
    }
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.token = options.token;
    this.tokenEnv = options.tokenEnv;
    this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (!(this.timeoutMs > 0)) {
      throw new Error("timeoutMs must be > 0");
    }
    this.fetchImpl = options.fetch ?? globalThis.fetch;
    if (typeof this.fetchImpl !== "function") {
      throw new Error("no fetch implementation available; pass options.fetch");
    }
    this.userAgent = options.userAgent ?? USER_AGENT;
  }

  toString(): string {
    const auth = this.token !== undefined || this.tokenEnv !== undefined ? "configured" : "none";
    return `CivilizationClient(baseUrl=${this.baseUrl}, auth=${auth}, timeoutMs=${this.timeoutMs})`;
  }

  private authorization(): string | undefined {
    let token = this.token;
    if (token === undefined && this.tokenEnv !== undefined) {
      token = process.env[this.tokenEnv];
    }
    if (token !== undefined && token.trim() === "") {
      throw new CivilizationError("configured bearer token is empty");
    }
    return token === undefined ? undefined : `Bearer ${token}`;
  }

  private async request(method: string, path: string, payload?: unknown): Promise<JsonRecord> {
    const headers: Record<string, string> = {
      Accept: "application/json",
      "User-Agent": this.userAgent,
    };
    const authorization = this.authorization();
    if (authorization !== undefined) headers["Authorization"] = authorization;
    let body: string | undefined;
    if (payload !== undefined) {
      headers["Content-Type"] = "application/json";
      body = JSON.stringify(payload);
    }
    const url = `${this.baseUrl}${path}`;
    let response: Response;
    try {
      response = await this.fetchImpl(url, {
        method,
        headers,
        body,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (error) {
      throw new CivilizationTransportError(`Civilization API ${method} ${path} failed: ${String(error)}`, {
        cause: error,
      });
    }
    const raw = await response.text();
    if (!response.ok) {
      throw new CivilizationAPIError(response.status, method, path, raw);
    }
    let decoded: unknown;
    try {
      decoded = JSON.parse(raw);
    } catch (error) {
      throw new CivilizationTransportError(`Civilization API ${method} ${path} returned invalid JSON`, {
        cause: error,
      });
    }
    if (!isRecord(decoded)) {
      throw new CivilizationTransportError(`Civilization API ${method} ${path} returned a non-object response`);
    }
    return decoded;
  }

  async health(): Promise<HealthResponse> {
    return (await this.request("GET", "/health")) as HealthResponse;
  }

  async ready(): Promise<HealthResponse> {
    return (await this.request("GET", "/ready")) as HealthResponse;
  }

  async metrics(): Promise<JsonRecord> {
    return this.request("GET", "/metrics");
  }

  /** Runtime capabilities declared by the service endpoint. */
  async capabilities(): Promise<CapabilitiesResponse> {
    return (await this.request("GET", "/v1/capabilities")) as unknown as CapabilitiesResponse;
  }

  async predictRaw(request: CivilizationRequestInit): Promise<JsonRecord> {
    return this.request("POST", "/v1/predict", buildRequestPayload(request));
  }

  async predict(request: CivilizationRequestInit): Promise<Prediction> {
    const payload = await this.predictRaw(request);
    const rows = payload["rows"];
    if (!Array.isArray(rows) || rows.length !== 1 || !isRecord(rows[0])) {
      throw new CivilizationInferenceError("prediction response must contain exactly one row");
    }
    return predictionFromRow(request, rows[0]);
  }

  async batchRaw(requests: readonly CivilizationRequestInit[]): Promise<JsonRecord> {
    if (requests.length === 0) throw new Error("requests must not be empty");
    return this.request("POST", "/v1/batch", {
      requests: requests.map((request) => buildRequestPayload(request)),
    });
  }

  async batch(requests: readonly CivilizationRequestInit[]): Promise<Prediction[]> {
    if (requests.length === 0) throw new Error("requests must not be empty");
    const identifiers = requests.map((request) => request.requestId ?? "");
    if (new Set(identifiers).size !== identifiers.length) {
      throw new Error("batch requestId values must be unique");
    }
    const payload = await this.batchRaw(requests);
    const rows = payload["rows"];
    if (!Array.isArray(rows) || rows.length !== requests.length) {
      throw new CivilizationInferenceError("batch response row count does not match request count");
    }
    const byId = new Map<string, CivilizationRequestInit>();
    for (const request of requests) {
      if (request.requestId !== undefined) byId.set(request.requestId, request);
    }
    return rows.map((row) => {
      if (!isRecord(row)) throw new CivilizationInferenceError("batch response contains a non-object row");
      const request = byId.get(String(row["id"] ?? ""));
      if (request === undefined) throw new CivilizationInferenceError("batch response contains an unknown request id");
      return predictionFromRow(request, row);
    });
  }

  async writeMemory(options: WriteMemoryOptions): Promise<MemoryCell> {
    const payload: JsonRecord = {
      session_id: options.sessionId,
      memory_system: options.memorySystem,
      content: options.content,
      summary: options.summary ?? options.content,
      confidence: options.confidence ?? 1.0,
      importance: options.importance ?? 1.0,
      metadata: options.metadata ?? {},
    };
    if (options.ttlSeconds !== undefined) payload["ttl_seconds"] = options.ttlSeconds;
    const response = await this.request("POST", "/admin/orion/memory/write", payload);
    return (isRecord(response["cell"]) ? response["cell"] : {}) as MemoryCell;
  }

  async readMemory(options: ReadMemoryOptions): Promise<MemoryCell[]> {
    const payload: JsonRecord = {
      session_id: options.sessionId,
      query: options.query,
      include_expired: options.includeExpired ?? false,
      limit: options.limit ?? 4,
    };
    if (options.memorySystem !== undefined) payload["memory_system"] = options.memorySystem;
    const response = await this.request("POST", "/admin/orion/memory/read", payload);
    const results = response["results"];
    if (!Array.isArray(results)) {
      throw new CivilizationTransportError("memory response results must be a list");
    }
    return results.filter(isRecord) as MemoryCell[];
  }

  async consolidateMemory(options: {
    sessionId: string;
    episodicCellIds: readonly string[];
    summary: string;
    content: string;
    metadata?: Record<string, unknown>;
  }): Promise<MemoryCell> {
    const response = await this.request("POST", "/admin/orion/memory/consolidate", {
      session_id: options.sessionId,
      episodic_cell_ids: [...options.episodicCellIds],
      summary: options.summary,
      content: options.content,
      metadata: options.metadata ?? {},
    });
    return (isRecord(response["cell"]) ? response["cell"] : {}) as MemoryCell;
  }

  async memoryStatus(): Promise<JsonRecord> {
    const response = await this.request("GET", "/admin/orion/memory");
    return isRecord(response["orion_memory"]) ? response["orion_memory"] : {};
  }

  async globalStoreStatus(): Promise<JsonRecord> {
    const response = await this.request("GET", "/admin/orion/global-store/status");
    return isRecord(response["global_store"]) ? response["global_store"] : {};
  }

  async saveGlobalStore(): Promise<JsonRecord> {
    return this.request("POST", "/admin/orion/global-store/save", {});
  }

  async loadGlobalStore(): Promise<JsonRecord> {
    return this.request("POST", "/admin/orion/global-store/load", {});
  }

  async submitJob(request: CivilizationRequestInit): Promise<Job> {
    return jobFromPayload(await this.request("POST", "/v1/jobs", { request: buildRequestPayload(request) }));
  }

  async getJob(jobId: string): Promise<Job> {
    return jobFromPayload(await this.request("GET", `/v1/jobs/${encodeURIComponent(jobId)}`));
  }

  async waitJob(jobId: string, options: { timeoutMs?: number; pollIntervalMs?: number } = {}): Promise<Job> {
    const timeoutMs = options.timeoutMs ?? this.timeoutMs;
    const pollIntervalMs = options.pollIntervalMs ?? 100;
    if (!(timeoutMs > 0) || !(pollIntervalMs > 0)) {
      throw new Error("timeoutMs and pollIntervalMs must be > 0");
    }
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const job = await this.getJob(jobId);
      if (["completed", "failed", "canceled"].includes(job.status)) return job;
      if (Date.now() >= deadline) {
        throw new Error(`job ${jobId} did not finish within ${timeoutMs} ms`);
      }
      await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
    }
  }

  async jobResult(jobId: string, options: { offset?: number; limit?: number } = {}): Promise<JsonRecord> {
    const query = new URLSearchParams({
      offset: String(options.offset ?? 0),
      limit: String(options.limit ?? 200),
    });
    return this.request("GET", `/v1/jobs/${encodeURIComponent(jobId)}/result?${query}`);
  }

  async submitBatchJobs(requests: readonly CivilizationRequestInit[]): Promise<JsonRecord> {
    if (requests.length === 0) throw new Error("requests must not be empty");
    return this.request("POST", "/v1/jobs/batch", {
      requests: requests.map((request) => buildRequestPayload(request)),
    });
  }

  async createExport(jobIds: readonly string[], options: { exportId?: string } = {}): Promise<JsonRecord> {
    const payload: JsonRecord = { job_ids: [...jobIds] };
    if (options.exportId !== undefined) payload["export_id"] = options.exportId;
    return this.request("POST", "/v1/jobs/export", payload);
  }

  async createPackage(exportId: string): Promise<JsonRecord> {
    return this.request("POST", `/v1/jobs/export/${encodeURIComponent(exportId)}/package`, {});
  }

  async getPackage(exportId: string): Promise<JsonRecord> {
    return this.request("GET", `/v1/jobs/export/${encodeURIComponent(exportId)}/package`);
  }

  /** Stream one export package to disk and verify its SHA-256. */
  async downloadPackage(
    exportId: string,
    outputPath: string,
    options: { expectedSha256?: string } = {},
  ): Promise<DownloadResult> {
    const path = `/v1/jobs/export/${encodeURIComponent(exportId)}/download`;
    const headers: Record<string, string> = { "User-Agent": this.userAgent };
    const authorization = this.authorization();
    if (authorization !== undefined) headers["Authorization"] = authorization;
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.baseUrl}${path}`, {
        method: "GET",
        headers,
        signal: AbortSignal.timeout(this.timeoutMs),
      });
    } catch (error) {
      throw new CivilizationTransportError(`package download failed: ${String(error)}`, { cause: error });
    }
    if (!response.ok) {
      throw new CivilizationAPIError(response.status, "GET", path, await response.text());
    }
    const buffer = Buffer.from(await response.arrayBuffer());
    const sha256 = createHash("sha256").update(buffer).digest("hex");
    if (options.expectedSha256 !== undefined && sha256 !== options.expectedSha256) {
      throw new CivilizationTransportError("downloaded package SHA-256 does not match expectedSha256");
    }
    await mkdir(dirname(outputPath), { recursive: true });
    const temporary = `${outputPath}.part`;
    try {
      await writeFile(temporary, buffer);
      await rename(temporary, outputPath);
    } catch (error) {
      await rm(temporary, { force: true });
      throw new CivilizationTransportError(`package download could not be written: ${String(error)}`, {
        cause: error,
      });
    }
    return {
      path: outputPath,
      bytes: buffer.byteLength,
      sha256,
      verified: options.expectedSha256 === undefined || sha256 === options.expectedSha256,
    };
  }
}

export type { MemorySystemName };
