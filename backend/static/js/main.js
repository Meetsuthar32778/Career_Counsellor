/**
 * main.js - AI Career Counsellor Frontend Logic
 * ================================================
 * Chat session, live ML analysis panel with Personality Radar Chart,
 * keyword extraction, speech output, and progress tracking.
 */

// ============================================================
// State
// ============================================================
let currentSessionId = null;
let isProcessing = false;
let speechEnabled = false;
let allDetectedKeywords = [];

const TOKEN = localStorage.getItem('token');
const USERNAME = localStorage.getItem('username');
if (!TOKEN) { window.location.href = '/'; }

const greetEl = document.getElementById('userGreeting');
if (greetEl) greetEl.textContent = `Hi, ${USERNAME || 'Student'}`;

function getHeaders() { return { 'Content-Type': 'application/json', 'Authorization': `Bearer ${TOKEN}` }; }
function logout() { localStorage.clear(); window.location.href = '/'; }

// ============================================================
// Keyword maps
// ============================================================
const FIELD_KEYWORDS = {
    CSE: ["code","program","software","algorithm","data","computer","app","website","automate","AI","artificial","machine learning","debug","hack","tech","digital","python","java","database","server","cloud","logic","internet","network"],
    MECHANICAL: ["machine","engine","gear","motor","thermal","heat","automobile","robot","manufacture","design","CAD","turbine","piston","fluid","torque","vehicle","aerospace","friction","tool","workshop"],
    CIVIL: ["building","bridge","road","structure","construct","concrete","soil","foundation","architecture","urban","plan","infrastructure","dam","survey","environment","drainage","earthquake","steel","cement","city"],
    ELECTRICAL: ["circuit","voltage","current","power","electric","wire","transformer","battery","solar","energy","signal","frequency","generator","grid","semiconductor","sensor","microcontroller","renewable","LED"],
    BIOLOGY: ["cell","DNA","gene","organism","species","evolution","body","health","medicine","bacteria","virus","ecology","plant","animal","microscope","biotechnology","anatomy","immune","blood","protein"],
    CHEMISTRY: ["chemical","reaction","molecule","atom","compound","element","acid","base","organic","bond","solution","lab","experiment","polymer","catalyst","pharmaceutical","formula","periodic","titration","crystal"],
    PHYSICS: ["force","gravity","motion","energy","quantum","relativity","wave","light","particle","nuclear","electromagnetic","velocity","acceleration","mass","newton","einstein","thermodynamics","optics","cosmos","universe"],
    MANAGEMENT: ["lead","team","manage","business","strategy","market","finance","organize","plan","entrepreneur","profit","invest","negotiate","communicate","HR","brand","customer","revenue","startup","budget","delegate","motivate"]
};

const FIELD_COLORS_HEX = {
    CSE: '#06b6d4', MECHANICAL: '#f97316', ELECTRICAL: '#3b82f6', CIVIL: '#10b981',
    BIOLOGY: '#ec4899', CHEMISTRY: '#8b5cf6', PHYSICS: '#ef4444', MANAGEMENT: '#f59e0b'
};
const FIELD_COLORS = {
    CSE: 'cse', MECHANICAL: 'mechanical', ELECTRICAL: 'electrical', CIVIL: 'civil',
    BIOLOGY: 'biology', CHEMISTRY: 'chemistry', PHYSICS: 'physics', MANAGEMENT: 'management'
};
const FIELD_NAMES = {
    CSE: 'Computer Science', MECHANICAL: 'Mechanical Engg', ELECTRICAL: 'Electrical Engg',
    CIVIL: 'Civil Engg', BIOLOGY: 'Biology', CHEMISTRY: 'Chemistry',
    PHYSICS: 'Physics', MANAGEMENT: 'Management'
};

// Current scores for radar
let fieldScores = { CSE:12.5, MECHANICAL:12.5, ELECTRICAL:12.5, CIVIL:12.5, BIOLOGY:12.5, CHEMISTRY:12.5, PHYSICS:12.5, MANAGEMENT:12.5 };

// ============================================================
// Session
// ============================================================
async function startNewSession() {
    if (isProcessing) return;
    isProcessing = true;
    const btn = document.getElementById('newSessionBtn');
    btn.textContent = 'Starting...'; btn.disabled = true;
    try {
        const res = await fetch('/api/chat/start', { method: 'POST', headers: getHeaders() });
        if (res.status === 401) { logout(); return; }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed');
        currentSessionId = data.session_id;
        allDetectedKeywords = [];
        fieldScores = { CSE:12.5, MECHANICAL:12.5, ELECTRICAL:12.5, CIVIL:12.5, BIOLOGY:12.5, CHEMISTRY:12.5, PHYSICS:12.5, MANAGEMENT:12.5 };
        document.getElementById('chatMessages').innerHTML = '';
        document.getElementById('inputArea').style.display = 'block';
        document.getElementById('progressContainer').style.display = 'block';
        document.getElementById('analysisPanel').style.display = 'flex';
        document.getElementById('sessionStatus').textContent = 'Session active';
        initAnalysisPanel();
        drawRadarChart();
        updateProgress(1, 10);
        addMessage('bot', data.welcome_message);
        await showTypingThenMessage(data.first_question);
        renderSuggestions(data.suggestions);
        showToast('Session started! Answer honestly.');
    } catch (err) { showToast('Error: ' + err.message); }
    btn.textContent = '+ New Session'; btn.disabled = false;
    isProcessing = false;
}

async function sendMessage() {
    if (isProcessing || !currentSessionId) return;
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    isProcessing = true;
    input.value = ''; input.style.height = 'auto';
    addMessage('user', text);
    showTyping();
    try {
        const res = await fetch('/api/chat/message', {
            method: 'POST', headers: getHeaders(),
            body: JSON.stringify({ message: text, session_id: currentSessionId }),
        });
        if (res.status === 401) { logout(); return; }
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed');
        removeTyping();
        updateProgress(data.question_number, data.total_questions);
        // Update analysis
        extractAndDisplayKeywords(text);
        if (data.detected_interests) {
            fieldScores = { ...fieldScores, ...data.detected_interests };
            updateRFBars(data.detected_interests);
            updateSemanticAnalysis(data.detected_interests);
            updateDecisionPath(data.detected_interests, data.is_complete);
            drawRadarChart();
        }
        await showTypingThenMessage(data.bot_message);
        renderSuggestions(data.suggestions);
        if (data.is_complete) {
            document.getElementById('sessionStatus').textContent = 'Analysis complete';
            document.getElementById('chatInput').disabled = true;
            document.getElementById('sendBtn').disabled = true;
            showToast('Analysis complete! Loading results...');
            setTimeout(() => { window.location.href = `/result/${currentSessionId}`; }, 2500);
        }
    } catch (err) { removeTyping(); showToast('Error: ' + err.message); }
    isProcessing = false;
}

// ============================================================
// UNIQUE FEATURE: Personality Radar Chart (Canvas)
// ============================================================
function drawRadarChart() {
    const canvas = document.getElementById('radarChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    const cx = W / 2, cy = H / 2; // Removed the +10 offset to keep it centered
    const R = Math.min(cx, cy) - 45; // Increased padding to ensure labels fit
    const fields = Object.keys(fieldScores);
    const n = fields.length;
    const maxVal = Math.max(...Object.values(fieldScores), 30);

    ctx.clearRect(0, 0, W, H);

    // Draw grid rings (3 levels)
    for (let level = 1; level <= 3; level++) {
        const r = (R / 3) * level;
        ctx.beginPath();
        for (let i = 0; i <= n; i++) {
            const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
            const x = cx + r * Math.cos(angle);
            const y = cy + r * Math.sin(angle);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.12)';
        ctx.lineWidth = 1;
        ctx.stroke();
    }

    // Draw axis lines + labels
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
        const x = cx + R * Math.cos(angle);
        const y = cy + R * Math.sin(angle);
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x, y);
        ctx.strokeStyle = 'rgba(59, 130, 246, 0.15)';
        ctx.stroke();

        // Labels
        const lx = cx + (R + 22) * Math.cos(angle);
        const ly = cy + (R + 22) * Math.sin(angle);
        ctx.fillStyle = '#94a3c4';
        ctx.font = '10px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const shortName = fields[i].length > 4 ? fields[i].substring(0, 4) : fields[i];
        ctx.fillText(shortName, lx, ly);
    }

    // Draw data polygon
    ctx.beginPath();
    for (let i = 0; i <= n; i++) {
        const idx = i % n;
        const angle = (Math.PI * 2 / n) * idx - Math.PI / 2;
        const val = fieldScores[fields[idx]] / maxVal;
        const x = cx + R * val * Math.cos(angle);
        const y = cy + R * val * Math.sin(angle);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.fillStyle = 'rgba(59, 130, 246, 0.2)';
    ctx.fill();
    ctx.strokeStyle = '#60a5fa';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw data points
    for (let i = 0; i < n; i++) {
        const angle = (Math.PI * 2 / n) * i - Math.PI / 2;
        const val = fieldScores[fields[i]] / maxVal;
        const x = cx + R * val * Math.cos(angle);
        const y = cy + R * val * Math.sin(angle);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = FIELD_COLORS_HEX[fields[i]] || '#7b93ff';
        ctx.fill();
        ctx.strokeStyle = '#0c1222';
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

// ============================================================
// Analysis Panel
// ============================================================
function initAnalysisPanel() {
    const container = document.getElementById('rfProbBars');
    container.innerHTML = '';
    const fields = ['CSE','MECHANICAL','ELECTRICAL','CIVIL','BIOLOGY','CHEMISTRY','PHYSICS','MANAGEMENT'];
    fields.forEach(f => {
        container.innerHTML += `
            <div class="rf-bar-item rf-${f.toLowerCase()}">
                <div class="rf-bar-label"><span>${FIELD_NAMES[f]}</span><span id="rfVal_${f}">12.5%</span></div>
                <div class="rf-bar-track"><div class="rf-bar-fill" id="rfBar_${f}" style="width:12.5%"></div></div>
            </div>`;
    });
    document.getElementById('keywordTags').innerHTML = '<span class="kw-placeholder">Waiting for responses...</span>';
    document.getElementById('semanticAnalysis').innerHTML = '<span class="kw-placeholder">Waiting for responses...</span>';
    document.getElementById('decisionPath').innerHTML = '<span class="kw-placeholder">Waiting for analysis...</span>';
}

function updateRFBars(interests) {
    for (const [field, score] of Object.entries(interests)) {
        const bar = document.getElementById(`rfBar_${field}`);
        const val = document.getElementById(`rfVal_${field}`);
        if (bar && val) { bar.style.width = Math.max(score, 1) + '%'; val.textContent = score.toFixed(1) + '%'; }
    }
}

function extractAndDisplayKeywords(text) {
    const lower = text.toLowerCase();
    const container = document.getElementById('keywordTags');
    let newTags = [];
    for (const [field, keywords] of Object.entries(FIELD_KEYWORDS)) {
        keywords.forEach(kw => {
            if (lower.includes(kw.toLowerCase())) {
                const key = kw + '_' + field;
                if (!allDetectedKeywords.includes(key)) {
                    allDetectedKeywords.push(key);
                    newTags.push({ word: kw, field });
                }
            }
        });
    }
    if (newTags.length > 0) {
        if (container.querySelector('.kw-placeholder')) container.innerHTML = '';
        newTags.forEach(t => {
            const tag = document.createElement('span');
            tag.className = `kw-tag ${FIELD_COLORS[t.field]}`;
            tag.textContent = `${t.word} (${t.field})`;
            container.appendChild(tag);
        });
    }
}

function updateSemanticAnalysis(interests) {
    const container = document.getElementById('semanticAnalysis');
    const sorted = Object.entries(interests).sort((a,b) => b[1] - a[1]).slice(0, 4);
    if (sorted.every(s => s[1] === 12.5)) return;
    container.innerHTML = '';
    sorted.forEach(([field, score]) => {
        const sim = (score / 100 * 0.95 + 0.05).toFixed(3);
        container.innerHTML += `<div class="sem-item"><div class="sem-label"><span>${FIELD_NAMES[field] || field}</span><span>${sim}</span></div><div class="sem-bar"><div class="sem-fill" style="width:${score}%"></div></div></div>`;
    });
}

function updateDecisionPath(interests, isComplete) {
    const container = document.getElementById('decisionPath');
    const sorted = Object.entries(interests).sort((a,b) => b[1] - a[1]);
    const top = sorted[0];
    if (sorted.every(s => s[1] === 12.5)) return;
    const checks = sorted.slice(0, 5);
    let html = '';
    checks.forEach(([field, score], i) => {
        const isTop = i === 0;
        const keyword = FIELD_KEYWORDS[field] ? FIELD_KEYWORDS[field][0] : field;
        html += `<div class="decision-item"><div class="decision-dot ${isTop ? 'yes' : 'no'}"></div><div class="decision-text">Contains <strong>'${keyword}'</strong>?<br><span style="font-size:0.7rem;color:var(--text-muted)">${isTop ? 'Yes' : 'No'} (${(score/100).toFixed(2)} ${isTop ? '>=' : '<'} ${(checks[0][1]/100).toFixed(2)})</span></div></div>`;
    });
    if (isComplete) {
        html += `<div class="decision-item"><div class="decision-dot leaf"></div><div class="decision-text"><span class="decision-result">✅ Leaf: ${FIELD_NAMES[top[0]] || top[0]}</span></div></div>`;
    }
    container.innerHTML = html;
}

// ============================================================
// Suggestions
// ============================================================
function renderSuggestions(suggestions) {
    const container = document.getElementById('suggestionChips');
    if (!suggestions || suggestions.length === 0) {
        container.style.display = 'none';
        container.innerHTML = '';
        return;
    }
    
    container.innerHTML = '';
    suggestions.forEach(text => {
        const btn = document.createElement('button');
        btn.className = 'chip-btn';
        btn.textContent = text;
        btn.onclick = () => {
            const input = document.getElementById('chatInput');
            // Append with space if there's already text, otherwise just set it
            if (input.value.trim().length > 0) {
                input.value = input.value.trim() + ' ' + text;
            } else {
                input.value = text;
            }
            input.focus();
            // Automatically expand textarea if needed
            input.style.height = 'auto'; 
            input.style.height = Math.min(input.scrollHeight, 120) + 'px';
            // Optional: Hide chips after clicking to encourage sending or editing
            // container.style.display = 'none'; 
        };
        container.appendChild(btn);
    });
    container.style.display = 'flex';
}

// ============================================================
// Messages
// ============================================================
function addMessage(role, text) {
    const chatEl = document.getElementById('chatMessages');
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'bot' ? '🤖' : '👤';
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.textContent = text;
    msg.appendChild(avatar); msg.appendChild(bubble);
    chatEl.appendChild(msg);
    chatEl.scrollTop = chatEl.scrollHeight;
    if (role === 'bot' && speechEnabled) speakText(text);
}
async function showTypingThenMessage(text) {
    showTyping();
    await sleep(Math.min(800 + text.length * 5, 2000));
    removeTyping();
    addMessage('bot', text);
}
function showTyping() {
    removeTyping();
    const chatEl = document.getElementById('chatMessages');
    const w = document.createElement('div');
    w.className = 'message bot'; w.id = 'typingIndicator';
    const a = document.createElement('div'); a.className = 'message-avatar'; a.textContent = '🤖';
    const t = document.createElement('div'); t.className = 'typing-indicator';
    t.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
    w.appendChild(a); w.appendChild(t); chatEl.appendChild(w); chatEl.scrollTop = chatEl.scrollHeight;
}
function removeTyping() { const el = document.getElementById('typingIndicator'); if (el) el.remove(); }
function updateProgress(c, t) {
    const p = Math.round((c / t) * 100);
    document.getElementById('progressLabel').textContent = `Question ${c} of ${t}`;
    document.getElementById('progressPercent').textContent = p + '%';
    document.getElementById('progressFill').style.width = p + '%';
}

const SPEAK_ON_SVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H3v6h3l5 4V5z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M15.54 8.46a5 5 0 010 7.07M19.07 4.93a9 9 0 010 12.73" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
const SPEAK_OFF_SVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M11 5L6 9H3v6h3l5 4V5z" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"/><path d="M23 9l-6 6M17 9l6 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';

function toggleSpeak() { 
    speechEnabled = !speechEnabled; 
    const btn = document.getElementById('speakBtn');
    if (speechEnabled) {
        btn.innerHTML = SPEAK_ON_SVG;
        showToast('Speech Output ON');
        if (window.speechSynthesis) {
            const u = new SpeechSynthesisUtterance('');
            u.volume = 0;
            window.speechSynthesis.speak(u);
        }
    } else {
        btn.innerHTML = SPEAK_OFF_SVG;
        showToast('Speech Output OFF');
        if (window.speechSynthesis) window.speechSynthesis.cancel();
    }
}

function speakText(text) { 
    if (!speechEnabled || !window.speechSynthesis) return; 
    window.speechSynthesis.cancel(); 
    const cleanText = text.replace(/🎓|💡|🎉|🤖|✅/g, ''); // Remove emojis that sound weird
    const u = new SpeechSynthesisUtterance(cleanText); 
    u.rate = 0.95; 
    window.speechSynthesis.speak(u); 
}

// ============================================================
// Utils
// ============================================================
function showToast(msg) { const t = document.getElementById('toast'); t.textContent = msg; t.classList.add('show'); setTimeout(() => t.classList.remove('show'), 3000); }
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
const chatInput = document.getElementById('chatInput');
if (chatInput) chatInput.addEventListener('input', function() { this.style.height = 'auto'; this.style.height = Math.min(this.scrollHeight, 120) + 'px'; });
