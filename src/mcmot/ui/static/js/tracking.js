/*
 * Tracking: pick a camera, start the pipeline, watch the annotated stream.
 *
 * **The stream is an `<img>`.** `multipart/x-mixed-replace` makes the browser
 * replace the picture in place, so live video costs one tag and no JavaScript at
 * all. The only care needed is teardown: the `src` must be cleared when leaving,
 * or the browser holds the connection open and the server keeps encoding for a
 * viewer that is gone.
 *
 * **Metrics come from the API, not from the stream.** The numbers on screen are the
 * worker's own counters, so FPS is what the pipeline achieved rather than what the
 * browser managed to render.
 */

const cameraSelect = document.getElementById('camera');
const streamImg = document.getElementById('stream');
const streamHint = document.getElementById('stream-hint');
const startButton = document.getElementById('start');
const stopButton = document.getElementById('stop');

let streaming = null; // camera id currently attached to the <img>

function selected() {
  return cameraSelect.value || null;
}

/* ------------------------------------------------------------- cameras */
async function loadCameras() {
  const previous = selected();
  let cameras = [];
  try {
    cameras = (await api.get('/api/cameras')).cameras;
  } catch (error) {
    console.debug(error);
  }

  cameraSelect.innerHTML = '';
  cameras.forEach((camera) => {
    const option = document.createElement('option');
    option.value = camera.camera_id;
    option.textContent = `${camera.camera_id} — ${camera.kind} (${camera.state})`;
    cameraSelect.appendChild(option);
  });

  if (previous && cameras.some((camera) => camera.camera_id === previous)) {
    cameraSelect.value = previous;
  }

  const hint = document.getElementById('camera-hint');
  if (!cameras.length) {
    hint.textContent = 'No cameras open. Add one above — a webcam index, an RTSP URL, or a phone MJPEG URL.';
  } else {
    hint.textContent = `${cameras.length} camera${cameras.length === 1 ? '' : 's'} open.`;
  }
  return cameras;
}

document.getElementById('add').addEventListener('click', async () => {
  const source = document.getElementById('source').value.trim();
  const cameraId = document.getElementById('new-id').value.trim();
  if (!source || !cameraId) {
    toast('A camera needs both an id and a source', 'bad');
    return;
  }
  try {
    await api.post('/api/cameras', { json: { camera_id: cameraId, source } });
    toast(`Camera ${cameraId} added`, 'ok');
    document.getElementById('source').value = '';
    document.getElementById('new-id').value = '';
    await loadCameras();
    cameraSelect.value = cameraId;
    await refresh();
  } catch (error) {
    toastError(error, 'Could not add the camera');
  }
});

/* ------------------------------------------------------------- stream */
function attachStream(cameraId) {
  if (streaming === cameraId) return;
  detachStream();
  // Cache-buster: without it a browser will happily reuse a completed stream
  // response from an earlier start, showing a frozen picture.
  streamImg.src = `/api/tracking/${encodeURIComponent(cameraId)}/stream?t=${Date.now()}`;
  streaming = cameraId;
  streamHint.textContent = `Streaming ${cameraId}.`;
}

function detachStream() {
  if (!streaming) return;
  // Assigning an empty src is what actually closes the connection.
  streamImg.removeAttribute('src');
  streaming = null;
  streamHint.textContent = 'Not streaming.';
}

streamImg.addEventListener('error', () => {
  if (streaming) streamHint.textContent = `Stream for ${streaming} ended.`;
});

/* ------------------------------------------------------------- control */
startButton.addEventListener('click', async () => {
  const cameraId = selected();
  if (!cameraId) return;
  startButton.disabled = true;
  try {
    await api.post(`/api/tracking/${encodeURIComponent(cameraId)}/start`);
    toast(`Tracking ${cameraId}`, 'ok', 'Models load on the first start, so give it a moment.');
    attachStream(cameraId);
  } catch (error) {
    toastError(error, 'Could not start tracking');
  } finally {
    await refresh();
  }
});

stopButton.addEventListener('click', async () => {
  const cameraId = selected();
  if (!cameraId) return;
  stopButton.disabled = true;
  try {
    await api.post(`/api/tracking/${encodeURIComponent(cameraId)}/stop`);
    toast(`Stopped tracking ${cameraId}`, 'ok');
    detachStream();
  } catch (error) {
    toastError(error, 'Could not stop tracking');
  } finally {
    await refresh();
  }
});

cameraSelect.addEventListener('change', async () => {
  detachStream();
  await refresh();
});

/* ------------------------------------------------------------- refresh */
async function refresh() {
  const [cameras, tracking] = await Promise.all([
    api.get('/api/cameras').catch(() => ({ cameras: [] })),
    api.get('/api/tracking').catch(() => ({ pipelines: [] })),
  ]);

  renderPipelines(tracking.pipelines);

  const cameraId = selected();
  const pipeline = tracking.pipelines.find((item) => item.camera_id === cameraId);
  const camera = cameras.cameras.find((item) => item.camera_id === cameraId);

  startButton.disabled = !cameraId || Boolean(pipeline && pipeline.running);
  stopButton.disabled = !pipeline || !pipeline.running;

  if (pipeline) {
    setText('m-fps', fmt.num(pipeline.fps, 1));
    setText('m-latency', fmt.ms(pipeline.latency_ms));
    setText('m-people', fmt.int(pipeline.people));
    setText('m-identified', fmt.int(pipeline.identified));
    if (pipeline.last_error) streamHint.textContent = `Pipeline error: ${pipeline.last_error}`;
  } else {
    ['m-fps', 'm-latency', 'm-people', 'm-identified'].forEach((id) => setText(id, '—'));
  }
  setText('m-state', camera ? camera.state : '—');

  // Reattach if a pipeline is running but the picture is not showing — covers a
  // reload with tracking already live.
  if (pipeline && pipeline.running && !streaming) attachStream(cameraId);
}

function renderPipelines(pipelines) {
  const body = document.getElementById('pipelines');
  if (!pipelines.length) {
    body.innerHTML = '<tr><td colspan="11" class="empty">No pipelines running.</td></tr>';
    return;
  }
  body.innerHTML = '';
  pipelines.forEach((p) => {
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${escapeHtml(p.camera_id)}</td>
      <td><span class="badge ${stateClass(p.state)}">${escapeHtml(p.state)}</span></td>
      <td class="num">${fmt.num(p.fps, 1)}</td>
      <td class="num">${fmt.ms(p.latency_ms)}</td>
      <td class="num">${fmt.num(p.detect_ms, 1)}</td>
      <td class="num">${fmt.num(p.reid_ms, 1)}</td>
      <td class="num">${fmt.int(p.people)}</td>
      <td class="num">${fmt.int(p.identified)}</td>
      <td class="num">${fmt.int(p.frames_processed)}</td>
      <td class="num">${fmt.int(p.frames_skipped_stale)}</td>
      <td class="num">${fmt.int(p.embeddings_computed)}</td>`;
    body.appendChild(row);
  });
}

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

/* ------------------------------------------------------------- boot */
window.addEventListener('pagehide', detachStream);

(async () => {
  await loadCameras();
  await refresh();
  poll(refresh, 1000);
})();
