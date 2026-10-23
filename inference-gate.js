'use strict';

// One shared budget for BOTH routes, per server process/replica. Only decoded
// inference owns a slot; cache hits never queue and waiting jobs own no raster.
function createInferenceGate({ concurrency = 1, maxQueue = 8, waitMs = 3000, now = Date.now } = {}) {
  if (![concurrency, maxQueue, waitMs].every(Number.isSafeInteger) || concurrency < 1 || concurrency > 4 || maxQueue < 0 || maxQueue > 32 || waitMs < 1 || waitMs > 10000) {
    throw new RangeError('Invalid inference capacity');
  }
  let active = 0, highWater = 0, accepted = 0, rejected = 0, cancelled = 0, totalWaitMs = 0;
  const queue = [];
  const busy = () => Object.assign(new Error('Analysis is busy. Please retry shortly.'), { statusCode: 503, code: 'inference_busy' });
  function take() {
    active++; accepted++; highWater = Math.max(highWater, active);
    let released = false;
    return () => {
      if (released) return;
      released = true; active--;
      while (queue.length && active < concurrency) queue.shift().admit();
    };
  }
  function acquire(signal) {
    if (signal?.aborted) return Promise.reject(Object.assign(new Error('Request disconnected'), { code: 'request_cancelled' }));
    if (active < concurrency) return Promise.resolve(take());
    if (queue.length >= maxQueue) { rejected++; return Promise.reject(busy()); }
    return new Promise((resolve, reject) => {
      const started = now();
      const remove = () => { const index = queue.indexOf(job); if (index >= 0) queue.splice(index, 1); clearTimeout(timer); signal?.removeEventListener('abort', abort); };
      const abort = () => { remove(); cancelled++; reject(Object.assign(new Error('Request disconnected'), { code: 'request_cancelled' })); };
      const job = { admit() { remove(); totalWaitMs += now() - started; resolve(take()); } };
      const timer = setTimeout(() => { remove(); rejected++; reject(busy()); }, waitMs);
      queue.push(job);
      signal?.addEventListener('abort', abort, { once: true });
    });
  }
  return { acquire, snapshot: () => ({ concurrency, maxQueue, waitMs, active, queued: queue.length, highWater, accepted, rejected, cancelled, totalWaitMs }) };
}

module.exports = { createInferenceGate };
