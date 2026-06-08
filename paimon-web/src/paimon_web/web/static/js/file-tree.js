// Shared file tree functionality for both workspace and subtask pages

let _ftEnvId = null;
let _ftSubtaskName = null;

function initFileTree(envId, subtaskName) {
    _ftEnvId = envId;
    _ftSubtaskName = subtaskName;
}

function encodeFilePath(filepath) {
    return filepath.split('/').map(
        segment => encodeURIComponent(segment)
    ).join('/');
}

function toggleDirectory(btn) {
    const dirItem = btn.closest('.directory-item');
    const children = dirItem.querySelector('.file-tree-children');
    const collapsed = btn.querySelector('.collapsed');
    const expanded = btn.querySelector('.expanded');

    if (children.style.display === 'none') {
        children.style.display = 'block';
        collapsed.style.display = 'none';
        expanded.style.display = 'inline';
    } else {
        children.style.display = 'none';
        collapsed.style.display = 'inline';
        expanded.style.display = 'none';
    }
}

async function viewFileInline(filepath, btn) {
    const item = btn.closest('.file-item-inline, .file-item');
    let preview = item.querySelector('.file-preview-inline');

    if (preview) {
        preview.remove();
        return;
    }

    preview = document.createElement('div');
    preview.className = 'file-preview-inline';

    const imageExts = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'];
    const isImage = imageExts.some(ext => filepath.toLowerCase().endsWith(ext));

    if (isImage) {
        preview.innerHTML = '<div class="loading-text">Loading...</div>';
        item.appendChild(preview);

        try {
            const downloadUrl = `/api/runs/${_ftEnvId}/subtasks/${_ftSubtaskName}/files/${encodeFilePath(filepath)}/download`;
            const img = document.createElement('img');
            img.src = downloadUrl;
            img.className = 'file-preview-image';
            img.alt = filepath;
            img.onerror = () => {
                preview.innerHTML = '<pre class="error-text">Error: Failed to load image</pre>';
            };
            preview.innerHTML = '';
            preview.appendChild(img);
        } catch (err) {
            preview.innerHTML = `<pre class="error-text">Error: ${err.message}</pre>`;
        }
    } else {
        preview.innerHTML = '<pre>Loading...</pre>';
        item.appendChild(preview);

        try {
            const url = `/api/runs/${_ftEnvId}/subtasks/${_ftSubtaskName}/files/${encodeFilePath(filepath)}`;
            const resp = await fetch(url);
            if (!resp.ok) throw new Error("Failed to load");
            const data = await resp.json();
            preview.querySelector('pre').textContent = data.content;
        } catch (err) {
            preview.querySelector('pre').textContent = "Error: " + err.message;
        }
    }
}

async function viewStructureInline(filepath, btn) {
    const item = btn.closest('.file-item-inline, .file-item');
    let viewer = item.querySelector('.viewer-structure-inline');

    if (viewer) {
        viewer.remove();
        return;
    }

    viewer = document.createElement('div');
    viewer.className = 'viewer-structure-inline';
    item.appendChild(viewer);

    try {
        const url = `/api/runs/${_ftEnvId}/subtasks/${_ftSubtaskName}/files/${encodeFilePath(filepath)}/structure`;
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("Failed to load");
        const data = await resp.json();

        let v = $3Dmol.createViewer(viewer, {backgroundColor: 'white'});
        let model = v.addModel(data.content, data.format);
        v.setStyle({}, {sphere: {radius: 0.4}, stick: {radius: 0.1}});

        if (data.format === 'cif') {
            v.addUnitCell(model, {box: {color: 'black'}});
        }

        v.zoomTo();
        v.render();
    } catch (err) {
        viewer.innerHTML = `<div class="viewer-error">Error: ${err.message}</div>`;
    }
}

function downloadFile(filepath) {
    const url = `/api/runs/${_ftEnvId}/subtasks/${_ftSubtaskName}/files/${encodeFilePath(filepath)}/download`;
    window.location.href = url;
}
