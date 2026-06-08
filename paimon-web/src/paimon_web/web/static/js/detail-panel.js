// Detail panel functionality for unified workspace
// Requires: file-tree.js (loaded before this script)

// Module-level state for detail panel
let _detailEnvId = null;

function initDetailPanel(envId) {
    _detailEnvId = envId;

    let knownFileCount = 0;
    let wasInitialState = true;
    let filesLoadedAfterTransition = false;

    const statusEl = document.getElementById('detail-status');
    if (!statusEl) return;

    // File count change detection and state transition detection
    statusEl.addEventListener('htmx:afterRequest', (evt) => {
        const xhr = evt.detail.xhr;
        if (!xhr) return;

        const newCount = parseInt(xhr.getResponseHeader('X-File-Count') || '0');
        if (knownFileCount > 0 && newCount > knownFileCount) {
            triggerRefreshShine();
        }
        knownFileCount = newCount;

        // Detect state transition from initial to non-initial
        const currentIsInitial = isInitialState();
        if (wasInitialState && !currentIsInitial && !filesLoadedAfterTransition) {
            htmx.ajax('GET', `/runs/${envId}/detail-files-fragment`, {
                target: '#detail-files',
                swap: 'innerHTML'
            });
            filesLoadedAfterTransition = true;
        }
        wasInitialState = currentIsInitial;
    });
}

function isInitialState() {
    const statusBadge = document.querySelector('#detail-status .status-badge');
    if (!statusBadge) return true;
    return statusBadge.classList.contains('status-pending');
}

function triggerRefreshShine() {
    const btn = document.getElementById('refresh-files-btn');
    if (btn) {
        btn.classList.add('shine');
        setTimeout(() => btn.classList.remove('shine'), 3000);
    }
}
