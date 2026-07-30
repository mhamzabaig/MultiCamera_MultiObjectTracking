/*
 * Dashboard: one screen that answers "is this working right now".
 *
 * It surfaces database-versus-index divergence rather than hiding it. The two
 * stores have no shared transaction, and the whole recovery design rests on the
 * index being a disposable derived cache — which is only useful if someone can see
 * when it has drifted.
 */

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

function uptime(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  if (hours) return `${hours}h ${minutes}m`;
  if (minutes) return `${minutes}m ${total % 60}s`;
  return `${total}s`;
}

async function refresh() {
  const [health, persons, cameras, tracking] = await Promise.all([
    api.get('/api/system/health').catch(() => null),
    api.get('/api/persons').catch(() => null),
    api.get('/api/cameras').catch(() => ({ cameras: [] })),
    api.get('/api/tracking').catch(() => ({ pipelines: [] })),
  ]);

  if (health) {
    setText('d-status', health.status);
    setText('d-device', health.device.name);
    setText('d-uptime', uptime(health.uptime_s));
  }

  if (persons) {
    setText('d-persons', fmt.int(persons.stats.persons));
    setText('d-vectors', fmt.int(persons.stats.index_vectors));

    const box = document.getElementById('divergence');
    const diverged = persons.stats.index_vectors !== persons.stats.embeddings;
    box.hidden = !diverged;
    if (diverged) {
      setText('div-db', persons.stats.embeddings);
      setText('div-idx', persons.stats.index_vectors);
    }
  }

  setText('d-cameras', fmt.int(cameras.cameras.length));
  const running = tracking.pipelines.filter((p) => p.running);
  setText('d-tracking', fmt.int(running.length));
  setText('d-people', fmt.int(running.reduce((total, p) => total + (p.people || 0), 0)));

  renderCameras(cameras.cameras, tracking.pipelines);
}

function renderCameras(cameras, pipelines) {
  const body = document.getElementById('d-camera-rows');
  if (!cameras.length) {
    body.innerHTML = '<tr><td colspan="9" class="empty">No cameras open.</td></tr>';
    return;
  }
  body.innerHTML = '';
  cameras.forEach((camera) => {
    const pipeline = pipelines.find((p) => p.camera_id === camera.camera_id);
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${escapeHtml(camera.camera_id)}</td>
      <td><span class="badge">${escapeHtml(camera.kind)}</span></td>
      <td><span class="badge ${stateClass(camera.state)}">${escapeHtml(camera.state)}</span></td>
      <td class="num">${fmt.num(camera.fps, 1)}</td>
      <td class="num">${fmt.ms(camera.latency_ms)}</td>
      <td class="num">${fmt.int(camera.frames_delivered)}</td>
      <td class="num">${fmt.int(camera.frames_dropped)}</td>
      <td class="num">${fmt.int(camera.reconnects)}</td>
      <td>${
        pipeline && pipeline.running
          ? `<span class="badge ok">${fmt.num(pipeline.fps, 1)} fps</span>`
          : '<span class="badge">off</span>'
      }</td>`;
    body.appendChild(row);
  });
}

async function refreshLogs() {
  const host = document.getElementById('d-logs');
  try {
    const { entries } = await api.get('/api/system/logs', { });
    // Only state changes and problems: per-request lines would bury them.
    const interesting = entries.filter(
      (entry) => entry.event && entry.event !== 'http.request' && entry.event !== 'pipeline.metrics',
    );
    const recent = interesting.slice(-14);
    if (!recent.length) {
      host.innerHTML = '<div class="empty">Nothing logged yet.</div>';
      return;
    }
    host.innerHTML = '';
    recent.reverse().forEach((entry) => {
      const line = document.createElement('div');
      line.className = 'logline';
      line.innerHTML = `
        <span class="ts">${escapeHtml(fmt.clock(entry.timestamp))}</span>
        <span class="lvl ${escapeHtml(entry.level || '')}">${escapeHtml(entry.level || '')}</span>
        <span class="msg">${escapeHtml(entry.event)}</span>`;
      host.appendChild(line);
    });
  } catch (error) {
    host.innerHTML = '<div class="empty">Could not read the log.</div>';
    console.debug(error);
  }
}

refresh();
refreshLogs();
poll(refresh, 2000);
poll(refreshLogs, 6000);
