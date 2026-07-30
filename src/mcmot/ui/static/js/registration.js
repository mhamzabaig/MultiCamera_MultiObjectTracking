/*
 * Registration: capture a burst from this device's camera and upload it.
 *
 * Three things about this page are load-bearing.
 *
 * **The secure-context check happens first.** `getUserMedia` is undefined on
 * `http://192.168.x.x`, and the browser's own failure is a bare `TypeError` that
 * tells a user nothing. Detecting it up front lets the page explain the actual
 * problem — you need HTTPS or localhost — instead of appearing broken.
 *
 * **Frames are captured from the video element, not the stream.** Drawing the
 * current frame to a canvas costs nothing and works identically on every browser,
 * whereas ImageCapture is still unevenly supported on Android.
 *
 * **Whole frames are uploaded, not crops.** The server runs the same detector the
 * tracking pipeline uses, so enrolment and inference agree on what a person crop
 * is. Cropping here would be a second implementation that could silently disagree.
 */

const preview = document.getElementById('preview');
const deviceSelect = document.getElementById('device');
const enableButton = document.getElementById('enable');
const switchButton = document.getElementById('switch');
const captureButton = document.getElementById('capture');
const retakeButton = document.getElementById('retake');
const saveButton = document.getElementById('save');
const nameInput = document.getElementById('name');
const statusLine = document.getElementById('status');
const bar = document.getElementById('bar');
const thumbs = document.getElementById('thumbs');

let stream = null;
let captured = []; // Blobs, in capture order
let capturing = false;

function cameraSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

function setStatus(text) {
  statusLine.textContent = text;
}

function refreshButtons() {
  const ready = Boolean(stream) && !capturing;
  captureButton.disabled = !ready;
  retakeButton.disabled = capturing || captured.length === 0;
  saveButton.disabled = capturing || captured.length === 0;
}

/* ------------------------------------------------------------- camera */
async function listDevices() {
  // Labels are only populated after permission is granted, so this runs again
  // once the stream is live.
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cameras = devices.filter((device) => device.kind === 'videoinput');
  deviceSelect.innerHTML = '';
  cameras.forEach((camera, index) => {
    const option = document.createElement('option');
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    deviceSelect.appendChild(option);
  });
  switchButton.hidden = cameras.length < 2;
}

async function startCamera(deviceId) {
  if (stream) stream.getTracks().forEach((track) => track.stop());

  const video = deviceId
    ? { deviceId: { exact: deviceId } }
    : // Rear camera by default on a phone: registration wants a full-body shot.
      { facingMode: { ideal: 'environment' } };
  video.width = { ideal: 1280 };
  video.height = { ideal: 720 };

  stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
  preview.srcObject = stream;
  await listDevices();

  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings();
  setStatus(`Camera live at ${settings.width || '?'}×${settings.height || '?'}. Ready to capture.`);
  enableButton.textContent = 'Restart camera';
  refreshButtons();
}

enableButton.addEventListener('click', async () => {
  if (!cameraSupported()) {
    toast('This browser will not grant camera access on an insecure address', 'bad');
    return;
  }
  try {
    await startCamera(deviceSelect.value || null);
  } catch (error) {
    // NotAllowedError is a denied permission prompt; NotFoundError is no camera.
    const reason =
      error.name === 'NotAllowedError'
        ? 'Permission was denied. Allow camera access and try again.'
        : error.name === 'NotFoundError'
          ? 'No camera was found on this device.'
          : error.message;
    toast('Could not open the camera', 'bad', reason);
    setStatus(reason);
  }
});

switchButton.addEventListener('click', () => {
  const options = Array.from(deviceSelect.options);
  if (options.length < 2) return;
  const next = (deviceSelect.selectedIndex + 1) % options.length;
  deviceSelect.selectedIndex = next;
  startCamera(deviceSelect.value).catch((error) => toastError(error));
});

deviceSelect.addEventListener('change', () => {
  if (stream) startCamera(deviceSelect.value).catch((error) => toastError(error));
});

/* ------------------------------------------------------------- capture */
function grabFrame() {
  const width = preview.videoWidth;
  const height = preview.videoHeight;
  if (!width || !height) return null;

  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  canvas.getContext('2d').drawImage(preview, 0, 0, width, height);
  return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.92));
}

function renderThumbs() {
  thumbs.innerHTML = '';
  // Only the last twelve: thirty thumbnails is a wall of near-identical images,
  // and every retained object URL is a live decoded bitmap.
  captured.slice(-12).forEach((blob) => {
    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);
    img.onload = () => URL.revokeObjectURL(img.src);
    thumbs.appendChild(img);
  });
}

captureButton.addEventListener('click', async () => {
  if (!stream) return;
  const total = Math.max(8, Math.min(60, Number(document.getElementById('count').value) || 30));
  const gap = Math.max(60, Math.min(1000, Number(document.getElementById('gap').value) || 150));

  captured = [];
  capturing = true;
  refreshButtons();
  thumbs.innerHTML = '';

  for (let i = 0; i < total; i += 1) {
    const blob = await grabFrame();
    if (blob) captured.push(blob);
    bar.style.width = `${((i + 1) / total) * 100}%`;
    setStatus(`Capturing ${i + 1} of ${total} — keep moving slowly.`);
    if (i % 3 === 2) renderThumbs();
    await new Promise((resolve) => setTimeout(resolve, gap));
  }

  capturing = false;
  renderThumbs();
  refreshButtons();
  setStatus(`Captured ${captured.length} frames. Review them, then save.`);
});

retakeButton.addEventListener('click', () => {
  captured = [];
  thumbs.innerHTML = '';
  bar.style.width = '0';
  setStatus('Discarded. Capture again when ready.');
  refreshButtons();
});

/* ------------------------------------------------------------- save */
saveButton.addEventListener('click', async () => {
  const name = nameInput.value.trim();
  if (!name) {
    toast('Enter a name first', 'bad');
    nameInput.focus();
    return;
  }
  if (!captured.length) return;

  const form = new FormData();
  form.append('name', name);
  captured.forEach((blob, index) => form.append('frames', blob, `frame-${index}.jpg`));

  saveButton.disabled = true;
  retakeButton.disabled = true;
  setStatus('Uploading and building the gallery…');
  saveButton.innerHTML = '<span class="spinner"></span> Saving';

  try {
    const person = await api.post('/api/persons', { form });
    toast(
      `${person.name} registered successfully`,
      'ok',
      `${person.embedding_count} gallery embeddings from ${captured.length} frames`,
    );
    setStatus(`${person.name} registered with ${person.embedding_count} embeddings.`);
    captured = [];
    thumbs.innerHTML = '';
    bar.style.width = '0';
    nameInput.value = '';
    loadHistory();
  } catch (error) {
    // The server's message is the useful one here — it says how many frames were
    // usable and why the rest were not.
    toastError(error, 'Registration failed');
    setStatus(error.message);
  } finally {
    saveButton.textContent = 'Save';
    refreshButtons();
  }
});

/* ------------------------------------------------------------- history */
async function loadHistory() {
  const body = document.getElementById('history');
  try {
    const { persons } = await api.get('/api/persons');
    if (!persons.length) {
      body.innerHTML = '<tr><td colspan="4" class="empty">Nobody registered yet.</td></tr>';
      return;
    }
    body.innerHTML = '';
    persons
      .slice()
      .sort((a, b) => b.created_at.localeCompare(a.created_at))
      .forEach((person) => {
        const row = document.createElement('tr');
        const avatar = person.representative_image
          ? `<img class="avatar" src="/api/persons/${person.id}/image" alt="">`
          : '<div class="avatar"></div>';
        row.innerHTML = `
          <td>${avatar}</td>
          <td>${escapeHtml(person.name)}</td>
          <td class="num">${person.embedding_count}</td>
          <td>${fmt.when(person.created_at)}</td>`;
        body.appendChild(row);
      });
  } catch (error) {
    body.innerHTML = '<tr><td colspan="4" class="empty">Could not load history.</td></tr>';
    console.debug(error);
  }
}

/** Escape a value before it goes anywhere near innerHTML. */
function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

/* ------------------------------------------------------------- boot */
if (!cameraSupported()) {
  document.getElementById('insecure-warning').hidden = false;
  enableButton.disabled = true;
  setStatus('Camera access needs HTTPS or localhost on this browser.');
}
loadHistory();
