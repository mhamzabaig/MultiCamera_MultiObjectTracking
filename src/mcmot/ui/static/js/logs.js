/*
 * Log viewer over the structured JSON log.
 *
 * Structured logging is only worth the effort if the context survives to the
 * reader, so each line renders its bound fields — `camera_id`, `track_id`,
 * `request_id` — rather than a flattened message string.
 */

const NOISE = new Set(['timestamp', 'level', 'event', 'logger']);

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

function contextOf(entry) {
  return Object.entries(entry)
    .filter(([key, value]) => !NOISE.has(key) && value !== null && typeof value !== 'object')
    .map(([key, value]) => `${key}=${value}`)
    .join(' ');
}

function matches(entry, needle) {
  if (!needle) return true;
  return JSON.stringify(entry).toLowerCase().includes(needle);
}

async function refresh() {
  const host = document.getElementById('logs');
  const level = document.getElementById('level').value;
  const limit = document.getElementById('limit').value || 200;
  const needle = document.getElementById('filter').value.trim().toLowerCase();

  const params = new URLSearchParams({ limit });
  if (level) params.set('level', level);

  let payload;
  try {
    payload = await api.get(`/api/system/logs?${params}`);
  } catch (error) {
    host.innerHTML = '<div class="empty">Could not read the log.</div>';
    console.debug(error);
    return;
  }

  if (!payload.file) {
    document.getElementById('meta').textContent = 'File logging is disabled in configuration.';
    host.innerHTML = '<div class="empty">No log file.</div>';
    return;
  }

  const shown = payload.entries.filter((entry) => matches(entry, needle));
  document.getElementById('meta').textContent =
    `${shown.length} of ${payload.entries.length} returned lines` +
    `${payload.truncated ? ', older records exist beyond this window' : ''} · ${payload.file}`;

  if (!shown.length) {
    host.innerHTML = '<div class="empty">Nothing matches.</div>';
    return;
  }

  // Newest first: on a page you scroll, the top is where you look.
  const fragment = document.createDocumentFragment();
  shown
    .slice()
    .reverse()
    .forEach((entry) => {
      const line = document.createElement('div');
      line.className = 'logline';
      line.innerHTML = `
        <span class="ts">${escapeHtml(fmt.clock(entry.timestamp))}</span>
        <span class="lvl ${escapeHtml(entry.level || '')}">${escapeHtml(entry.level || '')}</span>
        <span class="msg">${escapeHtml(entry.event || '')}
          <span class="ctx">${escapeHtml(contextOf(entry))}</span>
        </span>`;
      fragment.appendChild(line);
    });

  host.innerHTML = '';
  host.appendChild(fragment);
}

document.getElementById('refresh').addEventListener('click', refresh);
['level', 'limit', 'filter'].forEach((id) =>
  document.getElementById(id).addEventListener('change', refresh),
);

let stopPolling = null;

function applyFollow() {
  if (stopPolling) {
    stopPolling();
    stopPolling = null;
  }
  if (document.getElementById('live').value === '1') {
    stopPolling = poll(refresh, 3000);
  } else {
    refresh();
  }
}

document.getElementById('live').addEventListener('change', applyFollow);
applyFollow();
