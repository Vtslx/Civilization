export class CivilizationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "CivilizationError";
  }
}

/** The service could not be reached or returned an unusable payload. */
export class CivilizationTransportError extends CivilizationError {
  constructor(message: string, options?: { cause?: unknown }) {
    super(message);
    this.name = "CivilizationTransportError";
    if (options?.cause !== undefined) this.cause = options.cause;
  }
}

/** The service answered with a non-2xx status. The bearer token is never included. */
export class CivilizationAPIError extends CivilizationError {
  readonly statusCode: number;
  readonly method: string;
  readonly path: string;
  readonly body: string;

  constructor(statusCode: number, method: string, path: string, body: string) {
    super(`Civilization API ${method} ${path} returned HTTP ${statusCode}: ${body.slice(0, 500)}`);
    this.name = "CivilizationAPIError";
    this.statusCode = statusCode;
    this.method = method;
    this.path = path;
    this.body = body;
  }
}

/** The HTTP call succeeded but the inference row failed validation. */
export class CivilizationInferenceError extends CivilizationError {
  constructor(message: string) {
    super(message);
    this.name = "CivilizationInferenceError";
  }
}
