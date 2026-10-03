/**
 * Live end-to-end test against a running Civilization service.
 *
 * Required environment:
 *   CIVILIZATION_BASE_URL   e.g. http://127.0.0.1:8765
 * Optional:
 *   CIVILIZATION_TOKEN_ENV  environment variable holding the bearer token
 *   CIVILIZATION_TOKEN      bearer token directly
 *   CIVILIZATION_EXPECT_OPTION_ID  expected option id for the live prediction
 *
 * The suite is skipped when CIVILIZATION_BASE_URL is unset, so `npm test`
 * stays offline-safe.
 */

import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import { CivilizationClient } from "../dist/index.js";

const baseUrl = process.env.CIVILIZATION_BASE_URL;
const tokenEnv = process.env.CIVILIZATION_TOKEN_ENV;
const token = process.env.CIVILIZATION_TOKEN;
const expectedOptionId = process.env.CIVILIZATION_EXPECT_OPTION_ID;

function client() {
  const options = { allowInsecureHttp: true };
  if (tokenEnv !== undefined) options.tokenEnv = tokenEnv;
  else if (token !== undefined) options.token = token;
  return new CivilizationClient(baseUrl, options);
}

describe("live Civilization service", { skip: baseUrl === undefined }, () => {
  it("reports health and declared capabilities", async () => {
    const api = client();
    const ready = await api.ready();
    assert.equal(ready.ready, true);
    const capabilities = await api.capabilities();
    assert.equal(capabilities.status, "ok");
    assert.ok(capabilities.capabilities, "capabilities must be declared by the service");
    console.log("declared capabilities:", JSON.stringify(capabilities));
  });

  it("runs a real decision, batch, memory, job, and export cycle", async () => {
    const api = client();
    const sessionId = `live-${Date.now()}`;
    const prediction = await api.predict({
      text: "The deployment signature and health checks are valid, and no critical verification is unresolved. Choose the supported operation.",
      answerOptions: ["approve", "reject"],
      sessionId,
      requestId: `live-sync-${Date.now()}`,
      taskName: "sdk_live_test",
      memoryItems: ["The deployment signature and health checks are valid."],
      ruleItems: ["Reject only when a critical verification is unresolved."],
      stateValues: [0.9, 0.1, 0.8],
    });
    console.log("prediction:", prediction.optionId, prediction.optionText, "trace.runtime =", prediction.trace.runtime);
    if (expectedOptionId !== undefined) {
      assert.equal(prediction.optionId, Number(expectedOptionId));
    }
    assert.ok(["approve", "reject"].includes(prediction.optionText));

    const batch = await api.batch([
      {
        text: "Evidence A is verified. Choose.",
        answerOptions: ["approve", "reject"],
        sessionId,
        requestId: `live-batch-1-${Date.now()}`,
      },
      {
        text: "A critical verification is unresolved. Choose.",
        answerOptions: ["approve", "reject"],
        sessionId,
        requestId: `live-batch-2-${Date.now()}`,
      },
    ]);
    assert.equal(batch.length, 2);

    const cell = await api.writeMemory({
      sessionId,
      memorySystem: "episodic",
      content: "The live SDK test wrote this episode.",
      summary: "live sdk episode",
    });
    assert.ok(cell.cell_id, "memory write must return a cell id");
    const retrieved = await api.readMemory({ sessionId, query: "live sdk episode" });
    assert.ok(retrieved.length >= 1, "memory read must retrieve the written episode");

    const job = await api.submitJob({
      text: "Choose the supported operation for the queued request.",
      answerOptions: ["approve", "reject"],
      sessionId,
      requestId: `live-job-${Date.now()}`,
    });
    const finished = await api.waitJob(job.jobId, { timeoutMs: 180_000, pollIntervalMs: 250 });
    assert.equal(finished.status, "completed");
    // A completed job must also carry a successful decision row: a job whose row
    // failed is not a usable result, whatever its status says.
    const jobRows = await api.jobResult(job.jobId);
    const body = jobRows.result ?? jobRows;
    const rows = body.rows ?? [];
    assert.ok(rows.length >= 1, "job result must contain at least one row");
    assert.equal(rows[0].status, "ok", `job row failed: ${JSON.stringify(rows[0].error ?? {})}`);
    console.log("job row option:", rows[0].response?.predicted_option_id, "attempts:", rows[0].response?.trace?.provider_attempts);

    const created = await api.createExport([job.jobId], { exportId: `live-export-${Date.now()}` });
    const exportId = created.export?.export_id ?? created.export_id;
    assert.ok(exportId, "export must return an export id");
    const packageInfo = await api.createPackage(exportId);
    const directory = await mkdtemp(join(tmpdir(), "civilization-live-"));
    const destination = join(directory, `${exportId}.tar.gz`);
    const manifest = await api.getPackage(exportId);
    const expectedSha256 = manifest.package?.sha256 ?? manifest.sha256;
    const download = await api.downloadPackage(exportId, destination, expectedSha256 ? { expectedSha256 } : {});
    console.log("package:", JSON.stringify({ bytes: download.bytes, sha256: download.sha256, verified: download.verified, packageInfo: Object.keys(packageInfo) }));
    assert.ok(download.bytes > 0);
  });
});
