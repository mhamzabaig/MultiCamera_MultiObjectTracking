/*
 * Shared browser helpers: one API client, one error path, one poller.
 *
 * Every failing response in this system is problem+json with `code`, `message`,
 * and optional `details`. Parsing that in one place means a page never has to
 * invent its own error text, and the message a user sees is the one the server
 * actually chose — including the useful ones, like how many frames were usable.
 */

const api = {
  async request(method, path, { json, form, signal } = {}) {
    const init = { method, signal, headers: {} };
    if (json !== undefined) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(json);
    } else if (form !== undefined) {
      init.body = form; // let the browser set the multipart boundary
    }

    const response = await fetch(path, init);
    if (response.status === 204) return null;

    const isJson = (response.headers.get('content-type') || '').includes('json');
    const payload = isJson ? await response.json() : await response.text();

    if (!response.ok) {
      const error = new Error(
        (payload && payload.message) || `Request failed (${response.status})`,
      );
      error.status = response.status;
      error.code = payload && payload.code;
      error.details = payload && payload.details;
      throw error;
    }
    return payload;
  },

  get: (path, opts) => api.request('GET', path, opts),
  post: (path, opts) => api.request('POST', path, opts),
  patch: (path, opts) => api.request('PATCH', path, opts),
  del: (path, opts) => api.request('DELETE', path, opts),
};

/* --------------------------------------------------------------- toasts */
function toast(message, kind = 'ok', detail = '') {
  const host = document.getElementById('toasts');
  if (!host) return;
  const node = document.createElement('div');
  node.className = `toast ${kind}`;
  node.textContent = message;
  if (detail) {
    const sub = document.createElement('div');
    sub.className = 'detail';
    sub.textContent = detail;
    node.appendChild(sub);
  }
  host.appendChild(node);
  // Errors stay longer: they usually carry an instruction worth reading.
  setTimeout(() => node.remove(), kind === 'bad' ? 8000 : 4000);
}

/** Report an API error using the server's own wording. */
function toastError(error, fallback = 'Something went wrong') {
  const details = error && error.details;
  let detail = '';
  if (details && typeof details === 'object') {
    detail = Object.entries(details)
      .filter(([, value]) => typeof value !== 'object')
      .map(([key, value]) => `${key}: ${value}`)
      .join('  ·  ');
  }
  toast((error && error.message) || fallback, 'bad', detail);
}

/* --------------------------------------------------------------- polling */
/*
 * Polls are self-throttling: the next tick is scheduled only after the current
 * one settles, so a slow response cannot pile up overlapping requests. They also
 * pause while the tab is hidden — a dashboard left open in a background tab
 * should not keep a GPU-backed status endpoint busy all day.
 */
function poll(fn, intervalMs) {
  let stopped = false;
  let timer = null;

  const tick = async () => {
    if (stopped) return;
    if (!document.hidden) {
      try {
        await fn();
      } catch (error) {
        // Polls stay quiet: a transient failure will be retried, and a toast per
        // tick would bury the page in noise.
        console.debug('poll failed', error);
      }
    }
    if (!stopped) timer = setTimeout(tick, intervalMs);
  };

  tick();
  return () => {
    stopped = true;
    if (timer) clearTimeout(timer);
  };
}

/* --------------------------------------------------------------- format */
const fmt = {
  num: (value, digits = 1) =>
    value === null || value === undefined ? '—' : Number(value).toFixed(digits),
  int: (value) => (value === null || value === undefined ? '—' : String(value)),
  /*
   * Milliseconds, but seconds once the number stops being a latency and starts
   * being an age: a stalled camera's newest frame is tens of thousands of ms old,
   * and "57187" in a column headed Latency reads as a glitch rather than as a dead
   * stream.
   */
  ms: (value) => {
    if (value === null || value === undefined) return '—';
    const n = Number(value);
    return n >= 5000 ? `${(n / 1000).toFixed(1)} s` : `${n.toFixed(0)} ms`;
  },
  when: (iso) => {
    if (!iso) return '—';
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
  },
  clock: (iso) => {
    if (!iso) return '';
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? iso.slice(11, 23) : date.toLocaleTimeString();
  },
};

/** Map a camera or pipeline state onto a badge class. */
function stateClass(state) {
  if (['streaming', 'running', 'ok'].includes(state)) return 'ok';
  if (['connecting', 'reconnecting', 'idle', 'stopped', 'degraded'].includes(state)) return 'warn';
  return 'bad';
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

/* --------------------------------------------------------- health pill */
poll(async () => {
  const pill = document.getElementById('health-pill');
  if (!pill) return;
  try {
    const health = await api.get('/api/system/health');
    const device = health.device.type === 'cuda' ? 'GPU' : 'CPU';
    pill.textContent = `${health.status} · ${device}`;
    pill.className = `health ${health.status === 'ok' ? 'ok' : 'degraded'}`;
    pill.title = `${health.device.name} · v${health.version} · ${health.env}`;
  } catch {
    pill.textContent = 'offline';
    pill.className = 'health bad';
  }
}, 5000);
