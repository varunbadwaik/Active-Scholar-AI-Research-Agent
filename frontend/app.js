/* ═══════════════════════════════════════════════════════════════════════════
   Active Scholar — Frontend Application Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;
const POLL_INTERVAL = 3000; // ms

// ── DOM References ──────────────────────────────────────────────────────────
const apiStatusDot = document.querySelector('.status-dot');
const apiStatusText = document.querySelector('.status-text');

const heroSection = document.getElementById('heroSection');
const progressSection = document.getElementById('progressSection');
const reportSection = document.getElementById('reportSection');
const errorSection = document.getElementById('errorSection');

const historyList = document.getElementById('historyList');

// ── State ───────────────────────────────────────────────────────────────────
let currentJobId = null;
let pollTimer = null;
let stepTimers = [];
let startTime = null;
let clockTimer = null;
let currentStep = 0;
let rawMarkdown = '';

// ── Health Check ────────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      apiStatusDot.classList.add('online');
      apiStatusDot.classList.remove('offline');
      apiStatusText.textContent = 'API Online';
    } else {
      throw new Error();
    }
  } catch {
    apiStatusDot.classList.add('offline');
    apiStatusDot.classList.remove('online');
    apiStatusText.textContent = 'API Offline';
  }
}

// ── History ─────────────────────────────────────────────────────────────────
async function fetchHistory() {
  try {
    const res = await fetch(`${API_BASE}/reports`);
    if (!res.ok) return;
    const reports = await res.json();

    if (reports.length === 0) {
      historyList.innerHTML = '<div class="history-loading">No reports yet</div>';
      return;
    }

    historyList.innerHTML = '';
    reports.forEach(r => {
      const date = new Date(r.created_at * 1000).toLocaleDateString();
      const topic = r.filename.replace(/^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_/, '').replace(/\.md$/, '').replace(/_/g, ' ');

      const item = document.createElement('div');
      item.className = 'history-item';
      item.innerHTML = `
        <div class="history-topic" title="${topic}">${topic}</div>
        <div class="history-date">${date} • ${(r.size / 1024).toFixed(1)}KB</div>
      `;
      item.addEventListener('click', () => loadReport(r.filename));
      historyList.appendChild(item);
    });
  } catch (err) {
    historyList.innerHTML = '<div class="history-loading">Failed to load</div>';
  }
}

async function loadReport(filename) {
  try {
    // Show loading state if needed, or just switch to report view
    const res = await fetch(`${API_BASE}/reports/${filename}`);
    if (!res.ok) throw new Error('Failed to load report');

    const text = await res.text();

    // Mock a "result" object structure to reuse renderReport
    const mockData = {
      report_markdown: text,
      metadata: { topic: filename.replace(/^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_/, '').replace(/\.md$/, '').replace(/_/g, ' ') },
      sources: [], // We don't have structured sources for saved reports yet, unless we parse them
      claims: [],
      contradictions: []
    };

    renderReport(mockData, true); // true = isHistory

  } catch (err) {
    showError(err.message);
  }
}

// ── Section Visibility ──────────────────────────────────────────────────────
function showSection(section) {
  [heroSection, progressSection, reportSection, errorSection].forEach(s =>
    s.classList.add('hidden')
  );
  section.classList.remove('hidden');
}

// ── Timer ───────────────────────────────────────────────────────────────────
function startClock() {
  startTime = Date.now();
  clockTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    elapsedTime.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
  }, 1000);
}

function stopClock() {
  if (clockTimer) clearInterval(clockTimer);
}

function getElapsedFormatted() {
  if (!startTime) return '—';
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  if (elapsed < 60) return `${elapsed}s`;
  const mins = Math.floor(elapsed / 60);
  const secs = elapsed % 60;
  return `${mins}m ${secs}s`;
}

// ── Progress Steps ──────────────────────────────────────────────────────────
const STEPS = ['search', 'ingest', 'analyze', 'report'];

function setStep(stepIndex) {
  currentStep = stepIndex;
  document.querySelectorAll('.step').forEach((el, i) => {
    el.classList.remove('active', 'completed');
    if (i < stepIndex) el.classList.add('completed');
    if (i === stepIndex) el.classList.add('active');
  });
}

function advanceStepAuto() {
  if (currentStep < STEPS.length - 1) {
    setStep(currentStep + 1);
  }
}

// ── Submit Research ─────────────────────────────────────────────────────────
researchForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const topic = topicInput.value.trim();
  if (!topic) return;

  searchBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/research`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        topic: topic,
        scope: scopeSelect.value,
        max_search_rounds: parseInt(roundsSelect.value, 10),
        mode: modeSelect.value,
      }),
    });

    if (!res.ok) throw new Error(`Server returned ${res.status}`);

    const data = await res.json();
    currentJobId = data.job_id;

    // Show progress
    progressTopic.textContent = `"${topic}"`;
    progressBadge.textContent = 'Running';
    progressBadge.className = 'progress-badge running';
    showSection(progressSection);
    setStep(0);
    startClock();

    // Start auto-advancing steps for visual feedback
    const isQuick = modeSelect.value === 'quick';
    if (isQuick) {
      stepTimers = [
        setTimeout(() => advanceStepAuto(), 5000),
        setTimeout(() => advanceStepAuto(), 12000),
        setTimeout(() => advanceStepAuto(), 20000),
      ];
    } else {
      stepTimers = [
        setTimeout(() => advanceStepAuto(), 8000),
        setTimeout(() => advanceStepAuto(), 20000),
        setTimeout(() => advanceStepAuto(), 40000),
      ];
    }

    // Begin polling
    pollForResult(data.job_id);

  } catch (err) {
    showError(err.message);
  } finally {
    searchBtn.disabled = false;
  }
});

// ── Poll for Result ─────────────────────────────────────────────────────────
function pollForResult(jobId) {
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API_BASE}/research/${jobId}`);
      if (!res.ok) throw new Error(`Poll failed: ${res.status}`);

      const data = await res.json();

      if (data.status === 'running') return; // Still going

      // Job finished — clear timers
      clearInterval(pollTimer);
      stepTimers.forEach(t => clearTimeout(t));
      stopClock();

      if (data.status && data.status.startsWith('error')) {
        showError(data.status);
        return;
      }

      // Success — render report and refresh history
      fetchHistory();
      renderReport(data);

    } catch (err) {
      clearInterval(pollTimer);
      stepTimers.forEach(t => clearTimeout(t));
      stopClock();
      showError(err.message);
    }
  }, POLL_INTERVAL);
}

// ── Render Report ───────────────────────────────────────────────────────────
function renderReport(data, isHistory = false) {
  setStep(STEPS.length); // All completed

  // Parse data
  const reportMd = data.report_markdown || 'No report content available.';
  rawMarkdown = reportMd;
  const sources = data.sources || [];
  const claims = data.claims || [];
  const contradictions = data.contradictions || [];

  // Stats
  statSources.textContent = sources.length || '-';
  statClaims.textContent = claims.length || '-';
  statConflicts.textContent = contradictions.length || '-';
  statTime.textContent = isHistory ? 'Saved' : getElapsedFormatted();

  // Report meta
  const meta = data.metadata || {};
  reportMeta.textContent = meta.topic ? `Topic: ${meta.topic}` : '';

  // Body — simple markdown to HTML
  reportBody.innerHTML = simpleMarkdownToHtml(reportMd);

  // Sources
  sourcesCount.textContent = sources.length;
  sourcesList.innerHTML = '';
  if (sources.length === 0) {
    sourcesList.innerHTML = '<div class="source-item" style="justify-content:center; color:var(--text-tertiary)">Sources are not available for historical reports yet.</div>';
  } else {
    sources.forEach(src => {
      const score = src.credibility_score || 0;
      const scoreClass = score >= 0.7 ? 'high' : score >= 0.4 ? 'medium' : 'low';
      const item = document.createElement('div');
      item.className = 'source-item';
      item.innerHTML = `
        <span class="source-score ${scoreClass}">${(score * 100).toFixed(0)}%</span>
        <div class="source-info">
          <div class="source-title">${escapeHtml(src.title || 'Untitled')}</div>
          <div class="source-url"><a href="${escapeHtml(src.url || '#')}" target="_blank" rel="noopener">${escapeHtml(src.domain || src.url || '')}</a></div>
        </div>
      `;
      sourcesList.appendChild(item);
    });
  }

  showSection(reportSection);
}

// ── Simple Markdown → HTML ──────────────────────────────────────────────────
function simpleMarkdownToHtml(md) {
  let html = md
    // Code blocks
    .replace(/```[\s\S]*?```/g, (match) => {
      const code = match.replace(/```\w*\n?/, '').replace(/```$/, '');
      return `<pre><code>${escapeHtml(code)}</code></pre>`;
    })
    // Headers
    .replace(/^#### (.+)$/gm, '<h4>$1</h4>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    // Bold & Italic
    .replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Blockquotes
    .replace(/^> (.+)$/gm, '<blockquote>$1</blockquote>')
    // Unordered lists
    .replace(/^[\-\*] (.+)$/gm, '<li>$1</li>')
    // Ordered lists
    .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
    // Links
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
    // Horizontal rules
    .replace(/^---+$/gm, '<hr/>')
    // Paragraphs — double newlines
    .replace(/\n\n/g, '</p><p>')
    // Single newlines inside paragraphs
    .replace(/\n/g, '<br/>');

  // Wrap consecutive <li> in <ul>
  html = html.replace(/(<li>.*?<\/li>)(?:<br\/>)?/g, '$1');
  html = html.replace(/((?:<li>.*?<\/li>)+)/g, '<ul>$1</ul>');

  return `<p>${html}</p>`;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ── Error ───────────────────────────────────────────────────────────────────
function showError(msg) {
  errorMessage.textContent = msg || 'An unexpected error occurred.';
  showSection(errorSection);
  stopClock();
}

// ── Buttons ─────────────────────────────────────────────────────────────────
copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(rawMarkdown).then(() => {
    const original = copyBtn.querySelector('span').textContent;
    copyBtn.querySelector('span').textContent = 'Copied!';
    setTimeout(() => {
      copyBtn.querySelector('span').textContent = original;
    }, 2000);
  });
});

newSearchBtn.addEventListener('click', () => {
  topicInput.value = '';
  showSection(heroSection);
  topicInput.focus();
});

retryBtn.addEventListener('click', () => {
  showSection(heroSection);
  topicInput.focus();
});

// ── Init ──
checkHealth();
fetchHistory();
setInterval(checkHealth, 30000);
topicInput.focus();
