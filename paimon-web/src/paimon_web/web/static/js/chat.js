// Chat functionality for unified workspace

// Polling state
let _pollInterval = null;
let _pollRunId = null;
let _chatEventOffset = 0;
let _currentContentDiv = null;
let _currentToolCallsDiv = null;
let _isStreaming = false;
let _lastEventName = null;
let _lastProcessAlive = true;

// Suppress HTMX status polling during streaming to prevent
// stale filesystem state from overwriting optimistic badge
document.addEventListener('htmx:beforeSwap', function(evt) {
    if (_isStreaming && evt.detail.target
        && evt.detail.target.id === 'detail-status') {
        evt.detail.shouldSwap = false;
    }
});

// Events that indicate a status change worth refreshing the detail panel
const _statusEvents = new Set([
    'TaskComplete', 'TaskFail',
    'InputRequiredEvent', 'InputRequiredWithStepEvent',
    'StartSubtask', 'SubtaskSuccess', 'SubtaskFail',
]);

function _notifyStatusChange() {
    document.body.dispatchEvent(new CustomEvent('statusRefresh'));
}

function _setStatusBadgeOptimistic(status) {
    const badge = document.querySelector('#detail-status .status-badge');
    if (!badge) return;
    badge.className = 'status-badge status-' + status;
    badge.textContent = status;
}

function initChat(runId) {
    const messageInput = document.getElementById('message-input');
    const chatInputArea = document.querySelector('.chat-input-area');
    const dropOverlay = document.getElementById('drop-zone-overlay');
    const fileInput = document.getElementById('chat-files');

    if (!messageInput) return;

    // Auto-grow textarea
    messageInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 160) + 'px';
    });

    // Ctrl+Enter to submit
    messageInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            e.preventDefault();
            document.getElementById('chat-form').dispatchEvent(new Event('submit'));
        }
    });

    // Drag-and-drop file upload
    if (chatInputArea && dropOverlay && fileInput) {
        let dragCounter = 0;

        chatInputArea.addEventListener('dragenter', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter++;
            if (dragCounter === 1) {
                dropOverlay.classList.add('visible');
            }
        });

        chatInputArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter--;
            if (dragCounter === 0) {
                dropOverlay.classList.remove('visible');
            }
        });

        chatInputArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
        });

        chatInputArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            dragCounter = 0;
            dropOverlay.classList.remove('visible');

            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateChatFileLabel(fileInput);
            }
        });
    }
}

function updateChatFileLabel(input) {
    const fileCount = input.files.length;
    const displayText = fileCount > 0
        ? `${fileCount} file${fileCount > 1 ? 's' : ''} attached`
        : '';
    document.getElementById('chat-file-names').textContent = displayText;
}

async function sendMessage(e, runId) {
    e.preventDefault();
    const messageInput = document.getElementById('message-input');
    const sendButton = document.getElementById('send-button');
    const chatMessages = document.getElementById('chat-messages');
    const fileInput = document.getElementById('chat-files');
    const message = messageInput.value.trim();

    if (!message) return;

    _isStreaming = true;
    messageInput.disabled = true;
    sendButton.disabled = true;
    _setStatusBadgeOptimistic('running');

    // Upload files first if any
    let filePaths = [];
    if (fileInput.files.length > 0) {
        const formData = new FormData();
        for (const file of fileInput.files) {
            formData.append('files', file);
        }
        try {
            const uploadRes = await fetch(
                `/api/runs/${runId}/files`, {method: 'POST', body: formData}
            );
            if (uploadRes.ok) {
                const uploadData = await uploadRes.json();
                filePaths = uploadData.files || [];
            }
        } catch (err) {
            console.error('File upload error:', err);
        }
        fileInput.value = '';
        document.getElementById('chat-file-names').textContent = '';
    }

    // Remove hint
    const hint = chatMessages.querySelector('.chat-hint');
    if (hint) hint.remove();

    // Add user message
    const turnDiv = document.createElement('div');
    turnDiv.className = 'chat-turn';
    let fileAttachmentsHtml = '';
    if (filePaths.length > 0) {
        const fileNames = filePaths.map(path => path.split('/').pop());
        const badges = fileNames.map(name =>
            `<span class="file-badge"><span class="file-badge-icon">@</span>${escapeHtml(name)}</span>`
        ).join('');
        fileAttachmentsHtml = `<div class="file-attachments">${badges}</div>`;
    }
    turnDiv.innerHTML = `
        <div class="chat-message-user">
            <div class="message-label">You</div>
            <div class="message-content">${escapeHtml(message)}</div>
            ${fileAttachmentsHtml}
        </div>
        <div class="chat-message-assistant streaming">
            <div class="message-label">Assistant</div>
            <div class="tool-calls"></div>
            <div class="message-content"></div>
        </div>
    `;
    chatMessages.appendChild(turnDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    const assistantDiv = turnDiv.querySelector('.chat-message-assistant');
    const contentDiv = assistantDiv.querySelector('.message-content');
    const toolCallsDiv = assistantDiv.querySelector('.tool-calls');

    _currentContentDiv = contentDiv;
    _currentToolCallsDiv = toolCallsDiv;

    // Stream response
    const payload = { message };
    if (filePaths.length > 0) {
        payload.files = filePaths;
    }

    streamResponse(runId, payload, {
        assistantDiv,
        contentDiv,
        toolCallsDiv,
        chatMessages,
        messageInput,
        sendButton
    });
}

function streamResponse(runId, payload, elements) {
    const { assistantDiv, contentDiv, toolCallsDiv, chatMessages,
            messageInput, sendButton } = elements;

    fetch(`/api/runs/${runId}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    }).then(response => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        function readChunk() {
            reader.read().then(({ done, value }) => {
                if (done) {
                    assistantDiv.classList.remove('streaming');
                    messageInput.value = '';
                    messageInput.style.height = 'auto';
                    messageInput.disabled = false;
                    sendButton.disabled = false;
                    messageInput.focus();
                    syncChatOffset().then(() => {
                        _isStreaming = false;
                        _notifyStatusChange();
                    });
                    return;
                }

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.slice(6);
                        handleStreamData(data, contentDiv, toolCallsDiv, chatMessages);
                    }
                }
                readChunk();
            });
        }
        readChunk();
    }).catch(err => {
        console.error('Stream error:', err);
        contentDiv.textContent = '[Connection error]';
        assistantDiv.classList.remove('streaming');
        messageInput.style.height = 'auto';
        messageInput.disabled = false;
        sendButton.disabled = false;
        syncChatOffset().then(() => {
            _isStreaming = false;
            _notifyStatusChange();
        });
    });
}

function renderEvent(event, contentDiv, toolCallsDiv) {
    if (_statusEvents.has(event.name)) {
        _notifyStatusChange();
    }
    if (event.name === 'ToolCall') {
        const toolDiv = document.createElement('div');
        toolDiv.className = 'tool-call-item';
        toolDiv.innerHTML = `<span class="tool-icon">></span> ${escapeHtml(event.tool)}`;
        toolCallsDiv.appendChild(toolDiv);
    } else if (event.name === 'ToolCallResult' || event.name === 'SimpleToolCallResultEvent') {
        const lastTool = toolCallsDiv.lastElementChild;
        if (lastTool) {
            const status = event.success ? 'ok' : 'err';
            lastTool.innerHTML += ` <span class="tool-status-${status}">[${status}]</span>`;
        }
    } else if (event.name === 'AgentOutput' && event.content) {
        contentDiv.textContent += event.content;
    } else if (event.name === 'InputRequiredWithStepEvent') {
        contentDiv.textContent = event.question || '';
    } else if (event.name === 'StartSubtask') {
        const div = document.createElement('div');
        div.className = 'subtask-indicator';
        div.innerHTML = `> ${escapeHtml(event.subtask_name)} (${escapeHtml(event.agent)})`;
        toolCallsDiv.appendChild(div);
    } else if (event.name === 'SubtaskSuccess') {
        const div = document.createElement('div');
        div.className = 'subtask-indicator success';
        div.innerHTML = `[ok] ${escapeHtml(event.subtask_name)}`;
        toolCallsDiv.appendChild(div);
    } else if (event.name === 'SubtaskFail') {
        const div = document.createElement('div');
        div.className = 'subtask-indicator failed';
        div.innerHTML = `[err] ${escapeHtml(event.subtask_name)}`;
        toolCallsDiv.appendChild(div);
    } else if (event.name === 'TaskComplete') {
        contentDiv.textContent = event.report || 'Workflow completed.';
    } else if (event.name === 'TaskFail') {
        contentDiv.textContent = `Workflow failed: ${event.excuse || 'Unknown'}`;
    }
}

function handleStreamData(data, contentDiv, toolCallsDiv, chatMessages) {
    try {
        const event = JSON.parse(data);
        if (event.type === 'event') {
            renderEvent(event, contentDiv, toolCallsDiv);
        }
    } catch (e) {
        if (data.startsWith('error:')) {
            contentDiv.textContent += '\n[Error: ' + data.slice(7) + ']';
        } else {
            contentDiv.textContent += data;
        }
    }
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function _createAssistantDiv() {
    const div = document.createElement('div');
    div.className = 'chat-message-assistant';
    div.innerHTML = `
        <div class="message-label">Assistant</div>
        <div class="tool-calls"></div>
        <div class="message-content"></div>
    `;
    return div;
}

function appendChatEvents(events, chatMessages) {
    const hint = chatMessages.querySelector('.chat-hint');
    if (hint && events.length > 0) hint.remove();

    for (const event of events) {
        if (event.name === 'user_msg') {
            const userDiv = document.createElement('div');
            userDiv.className = 'chat-message-user';
            let filesHtml = '';
            if (event.files && event.files.length > 0) {
                const badges = event.files.map(f =>
                    `<span class="file-badge"><span class="file-badge-icon">@</span>${escapeHtml(f)}</span>`
                ).join('');
                filesHtml = `<div class="file-attachments">${badges}</div>`;
            }
            userDiv.innerHTML = `
                <div class="message-label">You</div>
                <div class="message-content">${escapeHtml(event.content || '')}</div>
                ${filesHtml}
            `;
            chatMessages.appendChild(userDiv);

            const assistantDiv = _createAssistantDiv();
            chatMessages.appendChild(assistantDiv);
            _currentContentDiv = assistantDiv.querySelector('.message-content');
            _currentToolCallsDiv = assistantDiv.querySelector('.tool-calls');
        } else if (_currentContentDiv && _currentToolCallsDiv) {
            renderEvent(event, _currentContentDiv, _currentToolCallsDiv);
        }
    }
    if (events.length > 0) {
        _lastEventName = events[events.length - 1].name;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

function renderChatFromEvents(events, chatMessages) {
    _currentContentDiv = null;
    _currentToolCallsDiv = null;
    appendChatEvents(events, chatMessages);
    _chatEventOffset = events.length;
}

function startChatPolling(runId) {
    _pollRunId = runId;
    if (_pollInterval) return;
    _pollInterval = setInterval(pollChatEvents, 3000);
}

function stopChatPolling() {
    if (_pollInterval) {
        clearInterval(_pollInterval);
        _pollInterval = null;
    }
}

async function pollChatEvents() {
    if (_isStreaming || !_pollRunId) return;
    try {
        const res = await fetch(
            `/api/runs/${_pollRunId}/events?offset=${_chatEventOffset}`
        );
        if (!res.ok) return;
        const data = await res.json();
        if (data.events.length > 0) {
            const chatMessages = document.getElementById('chat-messages');
            appendChatEvents(data.events, chatMessages);
        }
        _chatEventOffset = data.total;
        // Notify status panel on process death
        if (_lastProcessAlive && !data.process_alive) {
            _notifyStatusChange();
        }
        _lastProcessAlive = data.process_alive;
        // Cursor: show if process alive and last event is not terminal
        if (_currentContentDiv) {
            const el = _currentContentDiv
                .closest('.chat-message-assistant');
            if (!el) return;
            const _terminalEvents = new Set([
                'InputRequiredWithStepEvent',
                'InputRequiredEvent',
                'TaskComplete',
                'TaskFail',
            ]);
            if (data.process_alive && !_terminalEvents.has(_lastEventName)) {
                el.classList.add('streaming');
            } else {
                el.classList.remove('streaming');
            }
        }
    } catch (err) {
        console.error('Poll error:', err);
    }
}

async function syncChatOffset() {
    if (!_pollRunId) return;
    try {
        const res = await fetch(
            `/api/runs/${_pollRunId}/events?offset=999999`
        );
        if (res.ok) {
            const data = await res.json();
            _chatEventOffset = data.total;
        }
    } catch (err) { /* ignore */ }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
