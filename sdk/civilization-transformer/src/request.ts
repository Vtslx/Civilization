import { randomUUID } from "node:crypto";

import { CivilizationInferenceError } from "./errors.js";
import type {
  CivilizationRequestInit,
  CivilizationRequestPayload,
  Job,
  Prediction,
} from "./types.js";

const SESSION_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$/;

function nonEmptyStrings(values: readonly string[] | undefined, name: string): string[] {
  const result = [...(values ?? [])];
  for (const value of result) {
    if (typeof value !== "string" || value.trim() === "") {
      throw new Error(`${name} must contain non-empty strings`);
    }
  }
  return result;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Validate and serialize one request into the wire payload. */
export function buildRequestPayload(init: CivilizationRequestInit): CivilizationRequestPayload {
  if (typeof init.text !== "string" || init.text.trim() === "") {
    throw new Error("text must be a non-empty string");
  }
  const answerOptions = nonEmptyStrings(init.answerOptions, "answerOptions");
  if (answerOptions.length < 2) {
    throw new Error("answerOptions must contain at least two options");
  }
  const requestId = init.requestId ?? randomUUID().replace(/-/g, "");
  if (typeof requestId !== "string" || requestId.trim() === "") {
    throw new Error("requestId must be a non-empty string");
  }
  const sessionId = init.sessionId ?? "default";
  if (typeof sessionId !== "string" || !SESSION_ID_PATTERN.test(sessionId)) {
    throw new Error("sessionId contains unsupported characters");
  }
  const taskName = init.taskName ?? "custom";
  if (typeof taskName !== "string" || taskName.trim() === "") {
    throw new Error("taskName must be a non-empty string");
  }
  const stateValues = [...(init.stateValues ?? [0.5, 0.5, 0.5])];
  if (stateValues.length !== 3 || stateValues.some((value) => typeof value !== "number" || !Number.isFinite(value))) {
    throw new Error("stateValues must contain exactly three finite numbers");
  }
  return {
    id: requestId,
    text: init.text,
    memory_items: nonEmptyStrings(init.memoryItems, "memoryItems"),
    rule_items: nonEmptyStrings(init.ruleItems, "ruleItems"),
    state_values: stateValues.map(Number),
    answer_options: answerOptions,
    task_name: taskName,
    seed: Math.trunc(init.seed ?? 202),
    controls: ["full"],
    readouts: nonEmptyStrings(init.readouts ?? ["api_choice"], "readouts"),
    orion_memory: {
      session_id: sessionId,
      read: init.readMemory ?? true,
      write: init.writeMemory ?? true,
      include_global: init.includeGlobalMemory ?? false,
      allow_resolved_global: init.allowResolvedGlobalMemory ?? false,
    },
  };
}

/** Validate one inference row and expose it as a typed prediction. */
export function predictionFromRow(
  init: CivilizationRequestInit,
  row: Record<string, unknown>,
): Prediction {
  const response = row["response"];
  if (row["status"] !== "ok" || !isRecord(response)) {
    const error = row["error"] ?? "inference returned a non-ok row";
    throw new CivilizationInferenceError(String(error));
  }
  const optionId = response["predicted_option_id"];
  const options = [...init.answerOptions];
  if (typeof optionId !== "number" || !Number.isInteger(optionId) || optionId < 0 || optionId >= options.length) {
    throw new CivilizationInferenceError("inference response contains an invalid predicted_option_id");
  }
  const rawScores = isRecord(response["scores"]) ? response["scores"] : {};
  const scores: Record<string, number[]> = {};
  for (const [name, values] of Object.entries(rawScores)) {
    if (Array.isArray(values)) scores[name] = values.map(Number);
  }
  return {
    requestId: String(row["id"] ?? init.requestId ?? ""),
    optionId,
    optionText: options[optionId]!,
    scores,
    trace: isRecord(response["trace"]) ? response["trace"] : {},
    memoryTrace: isRecord(row["orion_memory"]) ? row["orion_memory"] : {},
    raw: row,
  };
}

/** Normalize a job payload, accepting both wrapped and unwrapped shapes. */
export function jobFromPayload(payload: Record<string, unknown>): Job {
  const job = isRecord(payload["job"]) ? payload["job"] : payload;
  if (!isRecord(job) || !job["job_id"]) {
    throw new Error("response does not contain a job");
  }
  return {
    jobId: String(job["job_id"]),
    status: String(job["status"] ?? "unknown"),
    result: isRecord(job["result"]) ? job["result"] : null,
    error: isRecord(job["error"]) ? job["error"] : null,
    raw: job,
  };
}
