/*
 * Settings: show the configuration, and explain the handful of values that decide
 * whether the system names people correctly.
 *
 * The explanations are here rather than in the config file because this is where
 * someone reads them while wondering why a person shows as Unknown.
 */

const EXPLANATIONS = [
  [
    'identity.tau_accept',
    'Minimum similarity to accept a match. Raise it if strangers get named; lower it if enrolled people stay Unknown.',
  ],
  [
    'identity.tau_margin',
    'How far ahead of the runner-up the winner must be. This is what blocks a confident but wrong match between two similar-looking people.',
  ],
  ['identity.top_k', 'How many gallery vectors each search returns before per-person aggregation.'],
  ['identity.vote_window', 'How many recent verdicts a track keeps before deciding on a name.'],
  ['detector.conf_threshold', 'Minimum detection confidence for a person box to enter tracking.'],
  ['tracker.match_thresh', 'IoU required to associate a detection with an existing track.'],
  ['pipeline.jpeg_quality', 'Quality of the JPEG sent to the browser. Lower means less bandwidth.'],
  ['pipeline.queue_size', 'Always 1: latest-frame-wins. A deeper queue trades freshness for coverage.'],
];

function lookup(config, path) {
  return path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), config);
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

(async () => {
  try {
    const payload = await api.get('/api/system/settings');

    document.getElementById('config').textContent = JSON.stringify(payload.config, null, 2);

    const files = payload.config_files || [];
    document.getElementById('files').textContent = files.length
      ? `Loaded from: ${files.join('  →  ')} (later files override earlier ones), then .env and environment variables.`
      : 'No configuration files found; every value is a code default.';

    const body = document.getElementById('key-settings');
    const rows = EXPLANATIONS.map(([path, effect]) => [path, lookup(payload.config, path), effect]).filter(
      ([, value]) => value !== undefined,
    );

    if (!rows.length) {
      body.innerHTML = '<tr><td colspan="3" class="empty">No values reported.</td></tr>';
      return;
    }

    body.innerHTML = '';
    rows.forEach(([path, value, effect]) => {
      const row = document.createElement('tr');
      row.innerHTML = `
        <td style="font-family:var(--mono);font-size:12.5px">${escapeHtml(path)}</td>
        <td class="num">${escapeHtml(value)}</td>
        <td class="hint" style="margin:0">${escapeHtml(effect)}</td>`;
      body.appendChild(row);
    });
  } catch (error) {
    toastError(error, 'Could not load settings');
    document.getElementById('config').textContent = 'Unavailable.';
  }
})();
