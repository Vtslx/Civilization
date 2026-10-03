import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { createServer } from "node:http";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { after, before, describe, it } from "node:test";

import {
  CivilizationAPIError,
  CivilizationClient,
  CivilizationInferenceError,
} from "../dist/index.js";

const PACKAGE_BYTES = Buffer.from('{"export":"crab","rows":2}\n', "utf8");
const PACKAGE_SHA256 = createHash("sha256").update(PACKAGE_BYTES).digest("hex");
const TOKEN = "test-token";

function readJson(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      try {
        resolve(JSON.parse(Buffer.concat(chunks).toString("utf8")));
      } catch (error) {
        reject(error);
      }
    });
  });
}

function sendJson(response, status, payload) {
  const body = Buffer.from(JSON.stringify(payload), "utf8");
  response.writeHead(status, { "Content-Type": "application/json", "Content-Length": body.byteLength });
  response.end(body);
}

function predictionRow(request, optionId) {
  return {
    id: request.id,
    status: "ok",
    response: {
      predicted_option_id: optionId,
      scores: { api_choice: request.answer_options.map((_, index) => (index === optionId ? 1 : 0)) },
      trace: { runtime: "fake", adapter_execution: false, hidden_states_available: false },
    },
    orion_memory: { retrieved: 1 },
  };
}

let server;
let baseUrl;
const jobs = new Map();

before(async () => {
  server = createServer(async (request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const authorized = request.headers.authorization === `Bearer ${TOKEN}`;

    if (url.pathname === "/health" || url.pathname === "/ready") {
      sendJson(response, 200, { status: "ok", ready: true });
      return;
    }
    if (url.pathname === "/metrics") {
      if (!authorized) {
        sendJson(response, 401, { status: "error", error: "missing or invalid bearer token" });
        return;
      }
      sendJson(response, 200, { requests_received: 3 });
      return;
    }
    if (url.pathname === "/v1/capabilities") {
      sendJson(response, 200, {
        status: "ok",
        runtime_label: "civilization-v1:provider",
        capabilities: {
          kind: "provider",
          label: "openai-compatible:test-model",
          provider_agnostic: true,
          local_weights: false,
          chat_completions: true,
          hidden_states: false,
          adapter_execution: false,
          ablation_controls: false,
          production_full_mode_only: true,
          notes: ["text-only decisions"],
        },
      });
      return;
    }
    if (url.pathname === "/v1/predict") {
      const payload = await readJson(request);
      sendJson(response, 200, { status: "ok", rows: [predictionRow(payload, 1)] });
      return;
    }
    if (url.pathname === "/v1/batch") {
      const payload = await readJson(request);
      sendJson(response, 200, {
        status: "ok",
        rows: payload.requests.map((record, index) => predictionRow(record, index % record.answer_options.length)),
      });
      return;
    }
    if (url.pathname === "/admin/orion/memory/write") {
      const payload = await readJson(request);
      sendJson(response, 200, { status: "ok", cell: { cell_id: "episodic-000001", ...payload } });
      return;
    }
    if (url.pathname === "/admin/orion/memory/read") {
      const payload = await readJson(request);
      sendJson(response, 200, { status: "ok", results: [{ cell_id: "episodic-000001", query: payload.query }] });
      return;
    }
    if (url.pathname === "/v1/jobs" && request.method === "POST") {
      const payload = await readJson(request);
      jobs.set("job-1", { job_id: "job-1", status: "completed", result: { rows: [predictionRow(payload.request, 0)] } });
      sendJson(response, 200, { status: "ok", job: jobs.get("job-1") });
      return;
    }
    if (url.pathname === "/v1/jobs/job-1") {
      sendJson(response, 200, { status: "ok", job: jobs.get("job-1") });
      return;
    }
    if (url.pathname === "/v1/jobs/job-1/result") {
      sendJson(response, 200, { status: "ok", rows: jobs.get("job-1").result.rows });
      return;
    }
    if (url.pathname === "/v1/jobs/export") {
      sendJson(response, 200, { status: "ok", export: { export_id: "export-1", job_ids: ["job-1"] } });
      return;
    }
    if (url.pathname === "/v1/jobs/export/export-1/package" || url.pathname === "/v1/jobs/export/export-1/download") {
      response.writeHead(200, { "Content-Type": "application/gzip", "Content-Length": PACKAGE_BYTES.byteLength });
      response.end(PACKAGE_BYTES);
      return;
    }
    sendJson(response, 404, { status: "error", error: "not found" });
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  await new Promise((resolve) => server.close(resolve));
});

function client(options = {}) {
  return new CivilizationClient(baseUrl, { token: TOKEN, ...options });
}

describe("CivilizationClient against a stub service", () => {
  it("reports declared runtime capabilities instead of assuming a model", async () => {
    const capabilities = await client().capabilities();
    assert.equal(capabilities.capabilities.kind, "provider");
    assert.equal(capabilities.capabilities.adapter_execution, false);
    assert.equal(capabilities.capabilities.hidden_states, false);
  });

  it("predicts and resolves option text", async () => {
    const prediction = await client().predict({
      text: "Which channel?",
      answerOptions: ["cobalt", "amber"],
      requestId: "req-1",
      sessionId: "session-a",
    });
    assert.equal(prediction.optionId, 1);
    assert.equal(prediction.optionText, "amber");
    assert.deepEqual(prediction.scores.api_choice, [0, 1]);
    assert.equal(prediction.memoryTrace.retrieved, 1);
  });

  it("batches requests and rejects duplicate request ids locally", async () => {
    const predictions = await client().batch([
      { text: "One", answerOptions: ["a", "b"], requestId: "batch-1" },
      { text: "Two", answerOptions: ["a", "b"], requestId: "batch-2" },
    ]);
    assert.equal(predictions.length, 2);
    await assert.rejects(
      () =>
        client().batch([
          { text: "One", answerOptions: ["a", "b"], requestId: "dup" },
          { text: "Two", answerOptions: ["a", "b"], requestId: "dup" },
        ]),
      /must be unique/,
    );
  });

  it("writes and reads memory", async () => {
    const api = client();
    const cell = await api.writeMemory({
      sessionId: "session-a",
      memorySystem: "episodic",
      content: "The canary passed health checks.",
    });
    assert.equal(cell.cell_id, "episodic-000001");
    const results = await api.readMemory({ sessionId: "session-a", query: "canary" });
    assert.equal(results.length, 1);
    assert.equal(results[0].query, "canary");
  });

  it("submits, waits for, and reads a job", async () => {
    const api = client();
    const job = await api.submitJob({ text: "Decide", answerOptions: ["a", "b"], requestId: "job-req" });
    assert.equal(job.jobId, "job-1");
    const waited = await api.waitJob("job-1", { timeoutMs: 2000 });
    assert.equal(waited.status, "completed");
    const result = await api.jobResult("job-1");
    assert.equal(result.rows.length, 1);
  });

  it("downloads an export package and verifies its SHA-256", async () => {
    const api = client();
    const exportRecord = await api.createExport(["job-1"], { exportId: "export-1" });
    assert.equal(exportRecord.export.export_id, "export-1");
    const directory = await mkdtemp(join(tmpdir(), "civilization-sdk-"));
    const destination = join(directory, "export.tar.gz");
    const download = await api.downloadPackage("export-1", destination, { expectedSha256: PACKAGE_SHA256 });
    assert.equal(download.verified, true);
    assert.equal(download.bytes, PACKAGE_BYTES.byteLength);
    assert.deepEqual(await readFile(destination), PACKAGE_BYTES);
    await assert.rejects(
      () => api.downloadPackage("export-1", destination, { expectedSha256: "0".repeat(64) }),
      /SHA-256 does not match/,
    );
  });

  it("surfaces HTTP failures as API errors without leaking the token", async () => {
    const unauthorized = client({ token: "wrong-token" });
    await assert.rejects(
      () => unauthorized.metrics(),
      (error) => {
        assert.ok(error instanceof CivilizationAPIError);
        assert.equal(error.statusCode, 401);
        assert.ok(!String(error.message).includes("wrong-token"));
        return true;
      },
    );
  });

  it("validates the endpoint and the client options", () => {
    assert.throws(() => new CivilizationClient("http://remote.example.test"), /loopback/);
    assert.throws(() => new CivilizationClient("ftp://127.0.0.1"), /absolute HTTP\(S\) URL/);
    assert.throws(() => new CivilizationClient("http://127.0.0.1", { token: "a", tokenEnv: "B" }), /not both/);
    assert.throws(() => client({ timeoutMs: 0 }), /timeoutMs must be > 0/);
    assert.doesNotThrow(() => new CivilizationClient("http://remote.example.test", { allowInsecureHttp: true }));
  });

  it("rejects malformed requests before any network call", async () => {
    const api = client();
    await assert.rejects(() => api.predict({ text: " ", answerOptions: ["a", "b"] }), /text must be a non-empty/);
    await assert.rejects(() => api.predict({ text: "x", answerOptions: ["only-one"] }), /at least two options/);
    await assert.rejects(
      () => api.predict({ text: "x", answerOptions: ["a", "b"], sessionId: "bad id" }),
      /sessionId contains unsupported characters/,
    );
    await assert.rejects(
      () => api.predict({ text: "x", answerOptions: ["a", "b"], stateValues: [1, 2] }),
      /exactly three finite numbers/,
    );
  });

  it("raises an inference error for a non-ok row", async () => {
    const failing = new CivilizationClient(baseUrl, {
      token: TOKEN,
      fetch: async () =>
        new Response(JSON.stringify({ status: "ok", rows: [{ id: "x", status: "error", error: "boom" }] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    });
    await assert.rejects(
      () => failing.predict({ text: "x", answerOptions: ["a", "b"] }),
      (error) => error instanceof CivilizationInferenceError && /boom/.test(error.message),
    );
  });
});
