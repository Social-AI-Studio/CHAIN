(function () {
    'use strict';

    // ── State ────────────────────────────────────────────────

    let levels = [];
    let currentLevel = null;
    let currentStep = 0;
    let experimentLog = null;
    let metrics = null;
    const imageCache = new Map();

    // ── DOM helpers ──────────────────────────────────────────

    const $ = (id) => document.getElementById(id);
    const BASE = 'step-examples';

    function imagePath(level, step) {
        return `${BASE}/${level.id}/${level.run_dir}/images/step_${step}.png`;
    }

    function metricsPath(level) {
        return `${BASE}/${level.id}/level_${level.level_num}_metrics.json`;
    }

    function logPath(level) {
        return `${BASE}/${level.id}/${level.run_dir}/experiment_log.json`;
    }

    // ── Init ─────────────────────────────────────────────────

    async function init() {
        try {
            const resp = await fetch('data/levels.json');
            const data = await resp.json();
            levels = data.levels;
        } catch (err) {
            console.error('Failed to load levels.json:', err);
            return;
        }

        setupOverlay();
        setupGlobalKeyboard();
        renderShowcase();
        setupFailureModes();
        renderLeaderboard();
        setupCopyBibtex();
    }

    // ── Showcase ─────────────────────────────────────────────

    const VLM_SHOWCASE_LEVEL_NUMS = [0, 15, 16, 61];
    const DIFFUSION_SHOWCASE_MODELS = [
        {
            key: 'hunyuan-v1.5',
            label: 'Hunyuan v1.5',
            puzzle1: 'video-examples/puzzle-1/huanyuan-v1.5-image2video.mp4',
            puzzle2: 'video-examples/puzzle-2/hunyuan-video-v1.5.mp4',
        },
        {
            key: 'kling-v2.6',
            label: 'Kling v2.6',
            puzzle1: 'video-examples/puzzle-1/kling-v2.6.mp4',
            puzzle2: 'video-examples/puzzle-2/kling-v2.6.mp4',
        },
        {
            key: 'sora-2',
            label: 'Sora 2',
            puzzle1: 'video-examples/puzzle-1/sora-2.mp4',
            puzzle2: 'video-examples/puzzle-2/sora-2.mp4',
        },
        {
            key: 'veo-3.1',
            label: 'Veo 3.1',
            puzzle1: 'video-examples/puzzle-1/veo3.1.mp4',
            puzzle2: 'video-examples/puzzle-2/veo3.1.mp4',
        },
        {
            key: 'wan-v2.6',
            label: 'Wan v2.6',
            puzzle1: 'video-examples/puzzle-1/wan-v2.6-image2video.mp4',
            puzzle2: 'video-examples/puzzle-2/wan-v2.6.mp4',
        },
    ];

    const vlmPreviewState = new Map();
    let vlmPreviewObserver = null;
    let diffusionPreviewObserver = null;

    function renderShowcase() {
        renderVlmShowcase();
        renderDiffusionShowcase();
    }

    function renderVlmShowcase() {
        const grid = $('showcase-vlm-grid');
        if (!grid) return;
        grid.innerHTML = '';

        const byNum = new Map(levels.map((l) => [l.level_num, l]));
        const selected = VLM_SHOWCASE_LEVEL_NUMS
            .map((n) => byNum.get(n))
            .filter(Boolean);

        if (!vlmPreviewObserver) {
            vlmPreviewObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    const st = vlmPreviewState.get(entry.target);
                    if (!st) return;
                    if (entry.isIntersecting) {
                        startVlmPreview(st);
                    } else {
                        stopVlmPreview(st);
                    }
                });
            }, { threshold: 0.25 });
        }

        selected.forEach((level) => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'showcase-card';
            card.dataset.kind = 'vlm';
            card.dataset.levelId = level.id;

            const thumb = document.createElement('div');
            thumb.className = 'showcase-thumb';

            const img = document.createElement('img');
            img.alt = `${level.name} preview`;
            img.src = imagePath(level, 0);
            thumb.appendChild(img);

            const meta = document.createElement('div');
            meta.className = 'showcase-meta';

            const title = document.createElement('div');
            title.className = 'showcase-title';
            title.textContent = level.name;

            const sub = document.createElement('div');
            sub.className = 'showcase-sub';
            const dot = document.createElement('span');
            dot.className = `diff-dot diff-${level.difficulty}`;
            const diffLabel = document.createElement('span');
            diffLabel.textContent = level.difficulty[0].toUpperCase() + level.difficulty.slice(1);
            sub.appendChild(dot);
            sub.appendChild(diffLabel);

            meta.appendChild(title);
            meta.appendChild(sub);

            card.appendChild(thumb);
            card.appendChild(meta);

            card.addEventListener('click', () => openVlmOverlay(level));

            grid.appendChild(card);

            const st = { card, img, level, step: 0, timer: null };
            vlmPreviewState.set(card, st);
            vlmPreviewObserver.observe(card);
        });
    }

    function startVlmPreview(st) {
        if (st.timer) return;
        // Preload first few frames to avoid flicker
        const preloadCount = Math.min(st.level.total_steps, 5);
        for (let i = 0; i < preloadCount; i++) {
            const path = imagePath(st.level, i);
            if (!imageCache.has(path)) {
                const img = new Image();
                img.onload = () => imageCache.set(path, img);
                img.src = path;
            }
        }
        st.timer = window.setInterval(() => {
            if (!st.level.total_steps) return;
            st.step = (st.step + 1) % st.level.total_steps;
            const path = imagePath(st.level, st.step);
            const cached = imageCache.get(path);
            st.img.src = cached ? cached.src : path;
        }, 2000);
    }

    function stopVlmPreview(st) {
        if (!st.timer) return;
        window.clearInterval(st.timer);
        st.timer = null;
    }

    function renderDiffusionShowcase() {
        const topGrid = $('showcase-diffusion-grid-top');
        const bottomGrid = $('showcase-diffusion-grid-bottom');
        const legacyGrid = $('showcase-diffusion-grid');

        if (topGrid) topGrid.innerHTML = '';
        if (bottomGrid) bottomGrid.innerHTML = '';
        if (legacyGrid) legacyGrid.innerHTML = '';

        if (!topGrid && !bottomGrid && !legacyGrid) return;

        const videos = [];
        const split = Math.floor(DIFFUSION_SHOWCASE_MODELS.length / 2);

        DIFFUSION_SHOWCASE_MODELS.forEach((model) => {
            const card = document.createElement('button');
            card.type = 'button';
            card.className = 'showcase-card';
            card.dataset.kind = 'diffusion';
            card.dataset.modelKey = model.key;

            const thumb = document.createElement('div');
            thumb.className = 'showcase-thumb';

            const video = document.createElement('video');
            video.muted = true;
            video.loop = true;
            video.playsInline = true;
            video.preload = 'metadata';

            const source = document.createElement('source');
            source.type = 'video/mp4';
            source.src = model.puzzle1;
            video.appendChild(source);
            video.load();

            thumb.appendChild(video);

            const meta = document.createElement('div');
            meta.className = 'showcase-meta';

            const title = document.createElement('div');
            title.className = 'showcase-title';
            title.textContent = model.label;
            meta.appendChild(title);

            card.appendChild(thumb);
            card.appendChild(meta);

            card.addEventListener('click', () => openDiffusionOverlay(model));

            if (legacyGrid) {
                legacyGrid.appendChild(card);
            } else if (topGrid && bottomGrid) {
                const idx = videos.length;
                (idx < split ? topGrid : bottomGrid).appendChild(card);
            } else {
                (topGrid || bottomGrid).appendChild(card);
            }
            videos.push(video);
        });

        if (!diffusionPreviewObserver) {
            diffusionPreviewObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    const video = entry.target;
                    if (entry.isIntersecting) {
                        video.play().catch(() => {});
                    } else {
                        video.pause();
                    }
                });
            }, { threshold: 0.3 });
        }

        videos.forEach((video) => diffusionPreviewObserver.observe(video));
    }

    // ── Overlay ──────────────────────────────────────────────

    let overlayKind = null; // 'vlm' | 'diffusion' | null
    let overlayPrevOverflow = '';
    let levelLoadNonce = 0;

    function setupOverlay() {
        const overlay = $('overlay');
        const closeBtn = $('overlay-close');
        const backdrop = overlay ? overlay.querySelector('.overlay-backdrop') : null;

        if (closeBtn) {
            closeBtn.addEventListener('click', closeOverlay);
        }

        if (backdrop) {
            backdrop.addEventListener('click', closeOverlay);
        }
    }

    function setupGlobalKeyboard() {
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && overlayKind) {
                e.preventDefault();
                closeOverlay();
                return;
            }

            if (overlayKind !== 'vlm') return;
            if (!currentLevel) return;
            if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;

            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                updateStep(currentStep - 1);
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                updateStep(currentStep + 1);
            }
        });
    }

    function openOverlay(kind, title) {
        const overlay = $('overlay');
        const body = $('overlay-body');
        const titleEl = $('overlay-title');
        const closeBtn = $('overlay-close');
        if (!overlay || !body || !titleEl || !closeBtn) return;

        overlayKind = kind;
        titleEl.textContent = title || '';
        body.innerHTML = '';

        overlay.classList.remove('hidden');
        overlay.setAttribute('aria-hidden', 'false');

        overlayPrevOverflow = document.body.style.overflow;
        document.body.style.overflow = 'hidden';

        closeBtn.focus();
    }

    function closeOverlay() {
        const overlay = $('overlay');
        const body = $('overlay-body');
        const titleEl = $('overlay-title');
        if (!overlay || !body || !titleEl) return;

        overlayKind = null;
        levelLoadNonce++;
        currentLevel = null;
        currentStep = 0;
        experimentLog = null;
        metrics = null;

        body.innerHTML = '';
        titleEl.textContent = '';

        overlay.classList.add('hidden');
        overlay.setAttribute('aria-hidden', 'true');

        document.body.style.overflow = overlayPrevOverflow;
    }

    function openVlmOverlay(level) {
        openOverlay('vlm', `${level.name} \u2014 ${level.subtitle}`);

        const tpl = $('vlm-viewer-template');
        const body = $('overlay-body');
        if (!tpl || !body) return;

        body.appendChild(tpl.content.cloneNode(true));
        setupVlmViewerControls();
        selectLevel(level);
    }

    function setupVlmViewerControls() {
        const slider = $('step-slider');
        const btnPrev = $('btn-prev');
        const btnNext = $('btn-next');

        if (slider) {
            slider.addEventListener('input', (e) => {
                updateStep(parseInt(e.target.value, 10));
            });
        }

        if (btnPrev) {
            btnPrev.addEventListener('click', () => updateStep(currentStep - 1));
        }

        if (btnNext) {
            btnNext.addEventListener('click', () => updateStep(currentStep + 1));
        }
    }

    function openDiffusionOverlay(model) {
        openOverlay('diffusion', model.label);

        const body = $('overlay-body');
        if (!body) return;

        const wrap = document.createElement('div');
        wrap.className = 'overlay-video-wrap';

        const controls = document.createElement('div');
        controls.className = 'overlay-video-controls';
        controls.innerHTML = `
            <button type="button" class="level-btn active" data-puzzle="puzzle-1">Puzzle 1</button>
            <button type="button" class="level-btn" data-puzzle="puzzle-2">Puzzle 2</button>
        `;

        const frame = document.createElement('div');
        frame.className = 'overlay-video-frame';

        const video = document.createElement('video');
        video.muted = true;
        video.loop = true;
        video.playsInline = true;
        video.autoplay = true;
        video.controls = true;

        const source = document.createElement('source');
        source.type = 'video/mp4';
        video.appendChild(source);

        frame.appendChild(video);
        wrap.appendChild(controls);
        wrap.appendChild(frame);
        body.appendChild(wrap);

        const buttons = controls.querySelectorAll('.level-btn');
        const loadPuzzle = (puzzleKey) => {
            buttons.forEach((btn) => {
                btn.classList.toggle('active', btn.dataset.puzzle === puzzleKey);
            });

            const src = puzzleKey === 'puzzle-2' ? model.puzzle2 : model.puzzle1;
            source.src = src;
            video.load();
            video.play().catch(() => {});
        };

        buttons.forEach((btn) => {
            btn.addEventListener('click', () => loadPuzzle(btn.dataset.puzzle));
        });

        loadPuzzle('puzzle-1');
    }

    // ── Select Level ─────────────────────────────────────────

    async function selectLevel(level) {
        const nonce = ++levelLoadNonce;
        currentLevel = level;
        currentStep = 0;

        // Setup slider
        const slider = $('step-slider');
        const loading = $('image-loading');
        if (!slider || !loading) return;
        slider.max = level.total_steps - 1;
        slider.value = 0;

        loading.textContent = 'Loading\u2026';
        loading.classList.remove('hidden');

        // Fetch experiment log
        try {
            const logResp = await fetch(logPath(level));
            const logData = await logResp.json();
            if (nonce !== levelLoadNonce) return;
            experimentLog = logData;
        } catch (err) {
            console.error('Failed to load experiment log:', err);
            experimentLog = null;
        }

        // Fetch metrics (handle Infinity in JSON)
        try {
            const metricsResp = await fetch(metricsPath(level));
            const text = await metricsResp.text();
            if (nonce !== levelLoadNonce) return;
            metrics = JSON.parse(text.replace(/\bInfinity\b/g, 'null'));
        } catch (err) {
            console.error('Failed to load metrics:', err);
            metrics = null;
        }

        if (nonce !== levelLoadNonce) return;
        renderMetrics();
        updateStep(0);
        preloadImages(level);
    }

    // ── Step Navigation ──────────────────────────────────────

    function updateStep(step) {
        if (!currentLevel) return;
        const maxStep = currentLevel.total_steps - 1;
        step = Math.max(0, Math.min(step, maxStep));
        currentStep = step;

        // Update slider
        $('step-slider').value = step;
        $('step-label').textContent = `Step ${step} / ${maxStep}`;

        // Update image
        showImage(step);

        // Update reasoning
        renderReasoning(step);
    }

    function showImage(step) {
        const img = $('step-image');
        const loading = $('image-loading');
        const path = imagePath(currentLevel, step);

        if (imageCache.has(path)) {
            img.src = imageCache.get(path).src;
            loading.classList.add('hidden');
        } else {
            loading.classList.remove('hidden');
            const loader = new Image();
            loader.onload = function () {
                imageCache.set(path, loader);
                // Only update if still on this step
                if (currentStep === step) {
                    img.src = loader.src;
                    loading.classList.add('hidden');
                }
            };
            loader.onerror = function () {
                loading.textContent = 'Image not found';
                loading.classList.remove('hidden');
            };
            loader.src = path;
        }
    }

    function preloadImages(level) {
        for (let i = 0; i < level.total_steps; i++) {
            const path = imagePath(level, i);
            if (!imageCache.has(path)) {
                const img = new Image();
                img.onload = function () {
                    imageCache.set(path, img);
                };
                img.src = path;
            }
        }
    }

    // ── Metrics Panel ────────────────────────────────────────

    function renderMetrics() {
        const panel = $('metrics-panel');
        if (!metrics) {
            panel.innerHTML = '<dt>Status</dt><dd>No data</dd>';
            return;
        }

        const solved = metrics.solved_tasks > 0;
        const fmt = (v, suffix) => {
            if (v === null || v === undefined) return '\u2014';
            if (typeof v === 'number') {
                if (suffix === '$') return `$${v.toFixed(2)}`;
                if (Number.isInteger(v)) return String(v);
                return v.toFixed(2);
            }
            return String(v);
        };

        panel.innerHTML = `
            <dt>Solved</dt>
            <dd class="${solved ? 'metric-solved' : 'metric-failed'}">${solved ? 'Yes' : 'No'}</dd>
            <dt>Accuracy</dt>
            <dd>${fmt(metrics.accuracy)}</dd>
            <dt>Steps Taken</dt>
            <dd>${fmt(metrics.avg_steps_solved)}</dd>
            <dt>Dist. to Optimal</dt>
            <dd>${fmt(metrics.distance_to_optimal)}</dd>
            <dt>Tokens / Step</dt>
            <dd>${fmt(metrics.tokens_per_step)}</dd>
            <dt>Total Cost</dt>
            <dd>${fmt(metrics.cost_total, '$')}</dd>
        `;
    }

    // ── Reasoning Panel ──────────────────────────────────────

    function renderReasoning(step) {
        $('reasoning-step-label').textContent = `Step ${step}`;
        const content = $('reasoning-content');

        if (!experimentLog || !experimentLog[step]) {
            content.innerHTML = '<div class="reasoning-block reasoning-initial">No data available for this step.</div>';
            return;
        }

        const entry = experimentLog[step];
        let html = '';

        if (entry.step_type === 'initial') {
            html += `<div class="reasoning-block reasoning-initial">
                <div class="reasoning-label">Initial State</div>
                <div>${escapeHtml(entry.observation?.description || 'Puzzle initialized.')}</div>
            </div>`;
        } else {
            // Agent reasoning
            if (entry.agent_response) {
                html += `<div class="reasoning-block reasoning-agent">
                    <div>${formatAgentResponse(entry.agent_response)}</div>
                </div>`;
            }

            // Action + result as a compact inline block
            if (entry.tool_call) {
                const fn = entry.tool_call.function;
                let argsCompact = '';
                if (fn.arguments) {
                    try {
                        const parsed = typeof fn.arguments === 'string'
                            ? JSON.parse(fn.arguments)
                            : fn.arguments;
                        argsCompact = JSON.stringify(parsed, null, 0)
                            .replace(/^\{/, '').replace(/\}$/, '')
                            .replace(/"(\w+)":/g, '$1: ');
                    } catch {
                        argsCompact = fn.arguments;
                    }
                }

                const result = entry.tool_result;
                const isError = result && result.status !== 'success';
                const statusClass = isError ? 'action-badge-error' : 'action-badge-ok';
                const statusText = result
                    ? escapeHtml(result.message || result.status)
                    : '';

                html += `<div class="action-block ${isError ? 'action-error' : ''}">
                    <span class="action-fn">${escapeHtml(fn.name)}</span>(<span class="action-args">${escapeHtml(argsCompact)}</span>)${result ? `<span class="action-badge ${statusClass}">${statusText}</span>` : ''}
                </div>`;
            }
        }

        content.innerHTML = html;
    }

    function formatAgentResponse(text) {
        // Insert newline before **bold** headings that are jammed against preceding text
        text = text.replace(/([^\n])\*\*(.+?)\*\*/g, '$1\n\n**$2**');
        let safe = escapeHtml(text);
        safe = safe.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        safe = safe.replace(/\n/g, '<br>');
        return safe;
    }

    function escapeHtml(str) {
        if (!str) return '';
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }


    // ── Leaderboard ─────────────────────────────────────────

    // Sorting state: null = default order, { key, dir: 'asc'|'desc' }
    let leaderboardData = null;
    let leaderboardSort = null;

    async function renderLeaderboard() {
        try {
            const resp = await fetch('data/leaderboard.json');
            leaderboardData = await resp.json();
        } catch (err) {
            console.error('Failed to load leaderboard.json:', err);
            const chart = $('results-chart');
            if (chart) {
                chart.innerHTML = '<div class="chart-empty">Failed to load leaderboard data.</div>';
            }
            return;
        }

        buildLeaderboardHead();
        buildLeaderboardBody();
        renderResultsChart();
    }

    function buildLeaderboardHead() {
        const { columns, column_groups: colGroups } = leaderboardData;
        const thead = $('leaderboard-head');
        thead.innerHTML = '';

        // Column group header row
        const groupRow = document.createElement('tr');
        const modelGroupTh = document.createElement('th');
        modelGroupTh.rowSpan = 2;
        modelGroupTh.textContent = 'Model';
        modelGroupTh.className = 'sortable';
        modelGroupTh.addEventListener('click', () => cycleSort('model'));
        groupRow.appendChild(modelGroupTh);

        for (const group of colGroups) {
            const th = document.createElement('th');
            th.colSpan = group.columns.length;
            th.className = 'col-group-header';
            th.textContent = group.label;
            groupRow.appendChild(th);
        }
        thead.appendChild(groupRow);

        // Column label row
        const labelRow = document.createElement('tr');
        for (const col of columns) {
            if (col.key === 'model') continue;
            const th = document.createElement('th');
            th.className = 'sortable';

            const label = col.label;
            const defaultArrow = col.higher_better ? '\u2191' : '\u2193';
            let sortIndicator = ' ' + defaultArrow;

            if (leaderboardSort && leaderboardSort.key === col.key) {
                sortIndicator = leaderboardSort.dir === 'asc' ? ' \u25B2' : ' \u25BC';
            }

            th.textContent = label + sortIndicator;
            th.addEventListener('click', () => cycleSort(col.key));
            labelRow.appendChild(th);
        }
        thead.appendChild(labelRow);
    }

    function cycleSort(key) {
        // 3-state cycle: default → desc → asc → default
        if (!leaderboardSort || leaderboardSort.key !== key) {
            leaderboardSort = { key, dir: 'desc' };
        } else if (leaderboardSort.dir === 'desc') {
            leaderboardSort = { key, dir: 'asc' };
        } else {
            leaderboardSort = null;
        }
        buildLeaderboardHead();
        buildLeaderboardBody();
    }

    function buildLeaderboardBody() {
        const { columns, sections } = leaderboardData;
        const allRows = sections.flatMap((s) => s.rows);
        const dataKeys = columns.filter((c) => c.higher_better !== null).map((c) => c.key);

        // Compute best values
        const best = {};
        for (const key of dataKeys) {
            const col = columns.find((c) => c.key === key);
            const vals = allRows.map((r) => r[key]).filter((v) => v !== null && v !== undefined);
            if (vals.length === 0) continue;
            best[key] = col.higher_better ? Math.max(...vals) : Math.min(...vals);
        }

        const tbody = $('leaderboard-body');
        tbody.innerHTML = '';

        if (leaderboardSort) {
            // Sorted: flat list, no section headers
            const sorted = [...allRows].sort((a, b) => {
                const va = a[leaderboardSort.key];
                const vb = b[leaderboardSort.key];
                if (leaderboardSort.key === 'model') {
                    const cmp = String(va).localeCompare(String(vb));
                    return leaderboardSort.dir === 'asc' ? cmp : -cmp;
                }
                const cmp = (va ?? -Infinity) - (vb ?? -Infinity);
                return leaderboardSort.dir === 'asc' ? cmp : -cmp;
            });
            for (const row of sorted) {
                tbody.appendChild(buildRow(row, columns, best));
            }
        } else {
            // Default: grouped by section
            for (const section of sections) {
                const sectionRow = document.createElement('tr');
                sectionRow.className = 'section-label';
                const sectionTd = document.createElement('td');
                sectionTd.colSpan = columns.length;
                sectionTd.textContent = section.label;
                sectionRow.appendChild(sectionTd);
                tbody.appendChild(sectionRow);

                for (const row of section.rows) {
                    tbody.appendChild(buildRow(row, columns, best));
                }
            }
        }
    }

    function buildRow(row, columns, best) {
        const tr = document.createElement('tr');
        for (const col of columns) {
            const td = document.createElement('td');
            const val = row[col.key];
            if (col.key === 'model') {
                td.className = 'cell-model';
                td.textContent = val;
            } else {
                td.textContent = formatCell(val, col.format);
                if (val === best[col.key]) {
                    td.classList.add('cell-best');
                }
            }
            tr.appendChild(td);
        }
        return tr;
    }

    function formatCell(val, format) {
        if (val === null || val === undefined) return '\u2014';
        switch (format) {
            case 'percent': return val.toFixed(1);
            case 'int':     return String(Math.round(val));
            case 'float':   return val.toFixed(2);
            default:        return String(val);
        }
    }

    // ── Headline Results Chart ───────────────────────────────

    function renderResultsChart() {
        const chart = $('results-chart');
        if (!chart) return;

        if (!leaderboardData) {
            chart.innerHTML = '<div class="chart-empty">Loading\u2026</div>';
            return;
        }

        const hasOverallAcc = leaderboardData.columns.some((c) => c.key === 'overall_acc');
        const hasPassAt1 = leaderboardData.columns.some((c) => c.key === 'pass_at_1');
        const accKey = hasOverallAcc ? 'overall_acc' : (hasPassAt1 ? 'pass_at_1' : null);

        if (!accKey) {
            chart.innerHTML = '<div class="chart-empty">No overall accuracy field found in leaderboard.json.</div>';
            return;
        }

        const rows = leaderboardData.sections
            .flatMap((s) => s.rows)
            .filter((r) => r && r.model);

        const sorted = rows
            .map((r) => ({ model: r.model, acc: r[accKey] }))
            .filter((r) => typeof r.acc === 'number')
            .sort((a, b) => b.acc - a.acc)
            .slice(0, 8);

        if (sorted.length === 0) {
            chart.innerHTML = '<div class="chart-empty">No rows available to render.</div>';
            return;
        }

        const maxAcc = 100;
        chart.innerHTML = '';

        const fills = [];

        sorted.forEach((row, idx) => {
            const chartRow = document.createElement('div');
            chartRow.className = `chart-row${idx === 0 ? ' is-best' : ''}`;

            const model = document.createElement('div');
            model.className = 'chart-model';
            model.textContent = row.model;

            const bar = document.createElement('div');
            bar.className = 'chart-bar';

            const fill = document.createElement('div');
            fill.className = 'chart-bar-fill';
            fill.style.width = '0%';
            fill.dataset.targetWidth = String(Math.max(0, Math.min(row.acc, maxAcc)));
            bar.appendChild(fill);

            const value = document.createElement('div');
            value.className = 'chart-value';
            value.textContent = `${row.acc.toFixed(1)}%`;

            chartRow.appendChild(model);
            chartRow.appendChild(bar);
            chartRow.appendChild(value);
            chart.appendChild(chartRow);

            fills.push(fill);
        });

        window.requestAnimationFrame(() => {
            fills.forEach((fill) => {
                fill.style.width = `${fill.dataset.targetWidth}%`;
            });
        });
    }

    // ── BibTeX Copy ──────────────────────────────────────────

    function setupCopyBibtex() {
        const btn = $('copy-bibtex');
        const bib = $('bibtex');
        if (!btn || !bib) return;

        btn.addEventListener('click', () => {
            const text = bib.textContent;
            navigator.clipboard.writeText(text).then(() => {
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
            });
        });
    }

    // ── Qualitative Failure Modes ────────────────────────────

    const failurePreviewState = new Map();
    let failurePreviewObserver = null;

    function setupFailureModes() {
        const cards = Array.from(document.querySelectorAll('.failure-card'));
        if (cards.length === 0) return;

        if (!failurePreviewObserver) {
            failurePreviewObserver = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    const st = failurePreviewState.get(entry.target);
                    if (!st) return;
                    st.visible = entry.isIntersecting;
                    if (st.visible) {
                        startFailurePreview(st);
                    } else {
                        stopFailurePreview(st);
                    }
                });
            }, { threshold: 0.25 });
        }

        cards.forEach((card) => {
            const img = card.querySelector('.failure-img');
            const action = card.querySelector('.failure-action');
            const label = card.querySelector('.failure-step-label');
            const levelId = card.dataset.levelId;
            const pattern = card.dataset.pattern || 'trial';

            if (!img || !action || !label || !levelId) return;

            const level = levels.find((l) => l.id === levelId);
            if (!level) {
                label.textContent = 'Missing level';
                return;
            }

            const st = {
                card,
                level,
                pattern,
                img,
                action,
                label,
                timer: null,
                visible: false,
                steps: null,
                stepIdx: 0,
                log: null,
            };

            failurePreviewState.set(card, st);
            failurePreviewObserver.observe(card);
            loadFailureExample(st);
        });
    }

    async function loadFailureExample(st) {
        try {
            const resp = await fetch(logPath(st.level));
            st.log = await resp.json();
        } catch (err) {
            console.error('Failed to load failure-mode log:', err);
            st.label.textContent = 'Failed to load';
            return;
        }

        st.steps = chooseFailureSteps(st.log, st.pattern);
        st.stepIdx = 0;

        preloadFailureImages(st.level, st.steps);
        renderFailureStep(st, 0);

        if (st.visible) {
            startFailurePreview(st);
        }
    }

    function preloadFailureImages(level, steps) {
        const list = (steps || []).slice(0, 8);
        list.forEach((step) => {
            const path = imagePath(level, step);
            if (imageCache.has(path)) return;
            const img = new Image();
            img.onload = () => imageCache.set(path, img);
            img.src = path;
        });
    }

    function chooseFailureSteps(log, pattern) {
        const total = Array.isArray(log) ? log.length : 0;
        if (total <= 1) return [0];

        const desiredLen = Math.min(8, Math.max(4, total - 1));

        if (pattern === 'pickup') {
            const steps = pickWindowAroundPickup(log, desiredLen);
            return steps.length ? steps : pickWindowMaxFailures(log, desiredLen);
        }

        if (pattern === 'late') {
            return pickWindowNearEnd(log, desiredLen);
        }

        return pickWindowMaxFailures(log, desiredLen);
    }

    function pickWindowNearEnd(log, len) {
        const total = log.length;
        const safeLen = Math.min(len, Math.max(1, total - 1));
        const start = Math.max(1, total - safeLen);
        return range(start, start + safeLen);
    }

    function pickWindowAroundPickup(log, len) {
        const total = log.length;
        const safeLen = Math.min(len, Math.max(1, total - 1));
        const idx = log.findIndex((e, i) =>
            i > 0 &&
            e &&
            e.tool_call &&
            e.tool_call.function &&
            e.tool_call.function.name === 'pickup'
        );
        if (idx === -1) return [];

        let start = Math.max(1, idx - Math.floor(safeLen / 2));
        if (start + safeLen > total) start = total - safeLen;
        start = Math.max(1, start);
        return range(start, start + safeLen);
    }

    function pickWindowMaxFailures(log, len) {
        const total = log.length;
        const safeLen = Math.min(len, Math.max(1, total - 1));

        let bestStart = 1;
        let bestScore = -1;

        for (let start = 1; start + safeLen <= total; start++) {
            let score = 0;
            for (let i = start; i < start + safeLen; i++) {
                const st = log[i]?.tool_result?.status;
                if (st && st !== 'success') score++;
            }
            if (score > bestScore) {
                bestScore = score;
                bestStart = start;
            }
        }

        return range(bestStart, bestStart + safeLen);
    }

    function range(a, b) {
        const out = [];
        for (let i = a; i < b; i++) out.push(i);
        return out;
    }

    function startFailurePreview(st) {
        if (st.timer) return;
        if (!st.steps || st.steps.length <= 1) return;
        st.timer = window.setInterval(() => {
            st.stepIdx = (st.stepIdx + 1) % st.steps.length;
            renderFailureStep(st, st.stepIdx);
        }, 2000);
    }

    function stopFailurePreview(st) {
        if (!st.timer) return;
        window.clearInterval(st.timer);
        st.timer = null;
    }

    function renderFailureStep(st, idx) {
        if (!st.steps || !st.steps.length) return;
        const step = st.steps[idx];

        const path = imagePath(st.level, step);
        const cached = imageCache.get(path);
        st.img.src = cached ? cached.src : path;

        st.label.textContent = `Step ${idx + 1} / ${st.steps.length}`;

        const entry = st.log && st.log[step] ? st.log[step] : null;
        st.action.innerHTML = entry
            ? buildActionBlockHtml(entry.tool_call, entry.tool_result)
            : '<div class="action-block">No data available.</div>';
    }

    function formatToolArgsCompact(fnArgs) {
        if (!fnArgs) return '';
        try {
            const parsed = typeof fnArgs === 'string' ? JSON.parse(fnArgs) : fnArgs;
            return JSON.stringify(parsed, null, 0)
                .replace(/^\{/, '')
                .replace(/\}$/, '')
                .replace(/"(\w+)":/g, '$1: ');
        } catch {
            return String(fnArgs);
        }
    }

    function buildActionBlockHtml(toolCall, toolResult) {
        if (!toolCall || !toolCall.function) return '<div class="action-block">No tool call.</div>';

        const fn = toolCall.function;
        const argsCompact = formatToolArgsCompact(fn.arguments);

        const result = toolResult;
        const isError = result && result.status !== 'success';
        const statusClass = isError ? 'action-badge-error' : 'action-badge-ok';
        const statusText = result ? escapeHtml(result.message || result.status) : '';

        return `<div class="action-block ${isError ? 'action-error' : ''}">
            <span class="action-fn">${escapeHtml(fn.name)}</span>(<span class="action-args">${escapeHtml(argsCompact)}</span>)${result ? `<span class="action-badge ${statusClass}">${statusText}</span>` : ''}
        </div>`;
    }

    // ── Boot ─────────────────────────────────────────────────

    document.addEventListener('DOMContentLoaded', init);
})();
