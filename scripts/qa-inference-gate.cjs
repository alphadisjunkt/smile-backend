const assert = require('node:assert/strict');
const { createInferenceGate } = require('../inference-gate');
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

(async () => {
  const gate = createInferenceGate({ concurrency: 1, maxQueue: 3, waitMs: 1000 });
  const first = await gate.acquire();
  const order = [];
  const waiters = [1, 2, 3].map(i => gate.acquire().then(release => { order.push(i); return release; }));
  await assert.rejects(gate.acquire(), e => e.statusCode === 503);
  assert.equal(gate.snapshot().queued, 3);
  first(); first(); // Releasing twice must not admit an extra job.
  const second = await waiters[0];
  assert.deepEqual(order, [1]);
  second(); const third = await waiters[1]; third(); const fourth = await waiters[2]; fourth();
  assert.deepEqual(order, [1, 2, 3]);
  assert.equal(gate.snapshot().highWater, 1);
  assert.equal(gate.snapshot().active, 0);

  const bounded = createInferenceGate({ concurrency: 1, maxQueue: 2, waitMs: 15 });
  const held = await bounded.acquire();
  await assert.rejects(bounded.acquire(), e => e.code === 'inference_busy');
  assert.equal(bounded.snapshot().queued, 0);
  const controller = new AbortController();
  const cancelled = bounded.acquire(controller.signal);
  controller.abort();
  await assert.rejects(cancelled, e => e.code === 'request_cancelled');
  assert.equal(bounded.snapshot().queued, 0);
  held();
  await assert.rejects(bounded.acquire(controller.signal), e => e.code === 'request_cancelled');

  const shared = createInferenceGate({ concurrency: 2, maxQueue: 16, waitMs: 1000 });
  let running = 0, peak = 0;
  await Promise.all(Array.from({ length: 12 }, async (_, i) => {
    const release = await shared.acquire();
    try { running++; peak = Math.max(peak, running); await pause(i % 3); if (i === 3) throw Error('mock native error'); }
    catch (e) { assert.equal(e.message, 'mock native error'); }
    finally { running--; release(); }
  }));
  assert.equal(peak, 2);
  assert.equal(shared.snapshot().active, 0);
  assert.equal(shared.snapshot().queued, 0);
  assert.throws(() => createInferenceGate({ concurrency: 0 }));
  console.log('PASS inference gate: shared concurrency, FIFO, bounded queue/wait, disconnect, failure release and idempotent release');
})().catch(error => { console.error(error); process.exitCode = 1; });
