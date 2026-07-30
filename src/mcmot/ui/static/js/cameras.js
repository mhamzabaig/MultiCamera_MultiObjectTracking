/* Camera management: add, watch, remove. */

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

document.getElementById('add').addEventListener('click', async () => {
  const cameraId = document.getElementById('camera-id').value.trim();
  const source = document.getElementById('source').value.trim();
  if (!cameraId || !source) {
    toast('A camera needs both an id and a source', 'bad');
    return;
  }
  try {
    const camera = await api.post('/api/cameras', { json: { camera_id: cameraId, source } });
    toast(`Camera ${camera.camera_id} added`, 'ok', `${camera.kind} · ${camera.state}`);
    document.getElementById('camera-id').value = '';
    document.getElementById('source').value = '';
    await refresh();
  } catch (error) {
    toastError(error, 'Could not add the camera');
  }
});

async function remove(cameraId, tracking) {
  const warning = tracking
    ? `Camera "${cameraId}" is being tracked. Closing it stops tracking too. Continue?`
    : `Close camera "${cameraId}"?`;
  if (!window.confirm(warning)) return;
  try {
    await api.del(`/api/cameras/${encodeURIComponent(cameraId)}`);
    toast(`Camera ${cameraId} closed`, 'ok');
    await refresh();
  } catch (error) {
    toastError(error, 'Could not close the camera');
  }
}

async function refresh() {
  const body = document.getElementById('rows');
  let cameras = [];
  let pipelines = [];
  try {
    [{ cameras }, { pipelines }] = await Promise.all([
      api.get('/api/cameras'),
      api.get('/api/tracking'),
    ]);
  } catch (error) {
    body.innerHTML = '<tr><td colspan="10" class="empty">Could not load cameras.</td></tr>';
    console.debug(error);
    return;
  }

  if (!cameras.length) {
    body.innerHTML = '<tr><td colspan="10" class="empty">No cameras open.</td></tr>';
    return;
  }

  body.innerHTML = '';
  cameras.forEach((camera) => {
    const tracked = pipelines.some((p) => p.camera_id === camera.camera_id && p.running);
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${escapeHtml(camera.camera_id)}${tracked ? ' <span class="badge ok">tracking</span>' : ''}</td>
      <td style="max-width:280px;word-break:break-all">${escapeHtml(camera.source)}</td>
      <td><span class="badge">${escapeHtml(camera.kind)}</span></td>
      <td><span class="badge ${stateClass(camera.state)}">${escapeHtml(camera.state)}</span>${
        camera.last_error ? `<div class="hint">${escapeHtml(camera.last_error)}</div>` : ''
      }</td>
      <td class="num">${fmt.num(camera.fps, 1)}</td>
      <td class="num">${fmt.ms(camera.latency_ms)}</td>
      <td class="num">${fmt.int(camera.frames_delivered)}</td>
      <td class="num">${fmt.int(camera.frames_dropped)}</td>
      <td class="num">${fmt.int(camera.reconnects)}</td>
      <td></td>`;

    const button = document.createElement('button');
    button.className = 'small danger';
    button.textContent = 'Close';
    button.addEventListener('click', () => remove(camera.camera_id, tracked));
    row.lastElementChild.appendChild(button);
    body.appendChild(row);
  });
}

refresh();
poll(refresh, 2000);
