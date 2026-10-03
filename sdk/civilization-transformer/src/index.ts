/**
 * Civilization Transformer SDK
 *
 * Provider-agnostic TypeScript client for the Civilization decision, memory,
 * job, and export service. The SDK assumes nothing about the model behind the
 * endpoint: runtime capabilities are declared by the service and reported
 * through {@link CivilizationClient.capabilities}.
 */

export { CivilizationClient } from "./client.js";
export {
  CivilizationAPIError,
  CivilizationError,
  CivilizationInferenceError,
  CivilizationTransportError,
} from "./errors.js";
export { buildRequestPayload, jobFromPayload, predictionFromRow } from "./request.js";
export { MEMORY_SYSTEMS } from "./types.js";
export type {
  CapabilitiesResponse,
  CivilizationRequestInit,
  CivilizationRequestPayload,
  ClientOptions,
  DownloadResult,
  HealthResponse,
  Job,
  MemoryCell,
  MemorySystemName,
  Prediction,
  ReadMemoryOptions,
  RuntimeCapabilities,
  WriteMemoryOptions,
} from "./types.js";

/** SDK release line. */
export const SDK_VERSION = "0.00.08-crab";
