/* The gallery: browse, rename, delete, rebuild. */

function escapeHtml(value) {
  const node = document.createElement('span');
  node.textContent = value == null ? '' : String(value);
  return node.innerHTML;
}

document.getElementById('rebuild').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  button.disabled = true;
  button.innerHTML = '<span class="spinner"></span> Rebuilding';
  try {
    const result = await api.post('/api/persons/index/rebuild');
    toast('Search index rebuilt', 'ok', `${result.vectors} vectors indexed`);
    await refresh();
  } catch (error) {
    toastError(error, 'Rebuild failed');
  } finally {
    button.disabled = false;
    button.textContent = 'Rebuild search index';
  }
});

async function rename(person) {
  const next = window.prompt(`New name for "${person.name}":`, person.name);
  if (next === null) return;
  const trimmed = next.trim();
  if (!trimmed || trimmed === person.name) return;
  try {
    await api.patch(`/api/persons/${encodeURIComponent(person.id)}`, { json: { name: trimmed } });
    toast(`Renamed to ${trimmed}`, 'ok');
    await refresh();
  } catch (error) {
    toastError(error, 'Could not rename');
  }
}

async function remove(person) {
  const message =
    `Delete "${person.name}" and all ${person.embedding_count} gallery embeddings?\n\n` +
    'This cannot be undone; they would need to register again.';
  if (!window.confirm(message)) return;
  try {
    await api.del(`/api/persons/${encodeURIComponent(person.id)}`);
    toast(`${person.name} deleted`, 'ok');
    await refresh();
  } catch (error) {
    toastError(error, 'Could not delete');
  }
}

async function refresh() {
  const body = document.getElementById('rows');
  let payload;
  try {
    payload = await api.get('/api/persons');
  } catch (error) {
    body.innerHTML = '<tr><td colspan="6" class="empty">Could not load the gallery.</td></tr>';
    console.debug(error);
    return;
  }

  const { persons, stats } = payload;
  setText('s-persons', fmt.int(stats.persons));
  setText('s-embeddings', fmt.int(stats.embeddings));
  setText('s-vectors', fmt.int(stats.index_vectors));
  setText('s-tracked', fmt.int(stats.tracked));
  document.getElementById('divergence').hidden = stats.index_vectors === stats.embeddings;

  if (!persons.length) {
    body.innerHTML =
      '<tr><td colspan="6" class="empty">Nobody enrolled yet — start on the Registration page.</td></tr>';
    return;
  }

  body.innerHTML = '';
  persons
    .slice()
    .sort((a, b) => a.name.localeCompare(b.name))
    .forEach((person) => {
      const row = document.createElement('tr');
      const avatar = person.representative_image
        ? `<img class="avatar" src="/api/persons/${encodeURIComponent(person.id)}/image" alt="">`
        : '<div class="avatar"></div>';
      row.innerHTML = `
        <td>${avatar}</td>
        <td>${escapeHtml(person.name)}</td>
        <td class="num">${person.embedding_count}</td>
        <td>${escapeHtml(fmt.when(person.created_at))}</td>
        <td class="num" style="font-size:12px">${escapeHtml(person.id.slice(0, 8))}</td>
        <td></td>`;

      const cell = row.lastElementChild;
      const renameButton = document.createElement('button');
      renameButton.className = 'small';
      renameButton.textContent = 'Rename';
      renameButton.addEventListener('click', () => rename(person));

      const deleteButton = document.createElement('button');
      deleteButton.className = 'small danger';
      deleteButton.textContent = 'Delete';
      deleteButton.style.marginLeft = '6px';
      deleteButton.addEventListener('click', () => remove(person));

      cell.append(renameButton, deleteButton);
      body.appendChild(row);
    });
}

refresh();
poll(refresh, 5000);
