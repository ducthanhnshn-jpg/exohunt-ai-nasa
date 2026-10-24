/* ==========================================================================
   ExoHunt AI - NASA Exoplanet Discovery Engine
   Vanilla JavaScript interactivity & classification demo
   ========================================================================== */

document.addEventListener('DOMContentLoaded', () => {
    // 1. Set up the tab system first so tab switching always works
    initTabs();
    initPresetSelector();
    initSliders();
    initModelSwitcher();
    initLightbox();

    // 2. Background animation
    try {
        initBackgroundParticles();
    } catch (e) {
        console.warn("Background animation could not start, skipping:", e);
    }

    // 3. First classification result
    try {
        updateInferenceEngine();
    } catch (e) {
        console.warn("Could not initialise the classification demo:", e);
    }

    // 4. Look for the optional local Python backend (app.py)
    detectBackend();

    // 5. Redraw the canvas when the window is resized
    window.addEventListener('resize', () => {
        const playgroundTab = document.getElementById('tab-playground');
        if (playgroundTab && playgroundTab.classList.contains('active')) {
            renderCurrentLightcurve();
        }
    });
});

/* ==========================================================================
   1. AI NEURAL CONSTELLATION / AEROSPACE MESH BACKGROUND ANIMATION
   ========================================================================== */
function initBackgroundParticles() {
    const canvas = document.createElement('canvas');
    canvas.id = 'starfield';
    document.body.prepend(canvas);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let width = canvas.width = window.innerWidth || 1200;
    let height = canvas.height = window.innerHeight || 800;

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth || 1200;
        height = canvas.height = window.innerHeight || 800;
    });

    const particles = [];
    const numParticles = Math.min(Math.floor((width * height) / 18000), 80);

    for (let i = 0; i < numParticles; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            radius: Math.random() * 2.5 + 1,
            vx: (Math.random() - 0.5) * 0.4,
            vy: (Math.random() - 0.5) * 0.4,
            alpha: Math.random() * 0.5 + 0.2
        });
    }

    function animateParticles() {
        if (!ctx) return;
        ctx.clearRect(0, 0, width, height);

        for (let i = 0; i < particles.length; i++) {
            let p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;

            // Draw particle node
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(37, 99, 235, ${p.alpha})`;
            ctx.fill();

            // Draw constellation lines between nearby nodes
            for (let j = i + 1; j < particles.length; j++) {
                let p2 = particles[j];
                let dx = p.x - p2.x;
                let dy = p.y - p2.y;
                let dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 140) {
                    ctx.beginPath();
                    ctx.moveTo(p.x, p.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.strokeStyle = `rgba(79, 70, 229, ${(1 - dist / 140) * 0.15})`;
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(animateParticles);
    }
    animateParticles();
}

/* ==========================================================================
   2. TAB NAVIGATION SYSTEM (WORKS EVEN IF OTHER PARTS FAIL)
   ========================================================================== */
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanels = document.querySelectorAll('.tab-panel');

    tabButtons.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = btn.getAttribute('data-tab');
            if (!targetId) return;

            // Remove active from all
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanels.forEach(p => p.classList.remove('active'));

            // Add active to current
            btn.classList.add('active');
            const targetPanel = document.getElementById(targetId);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }

            // Re-render canvas with current values if the playground is selected
            if (targetId === 'tab-playground') {
                setTimeout(renderCurrentLightcurve, 60);
            }
        });
    });
}

/* Helper: render the light curve for the current preset or slider values */
function renderCurrentLightcurve() {
    const v = getCurrentInputs();
    if (v) renderLightcurve(v.depth, v.duration, v.shape);
}

/* ==========================================================================
   3. PRESET TARGET DATA & SELECTOR
   Real objects: values from NASA's cumulative KOI table (data/cumulative.csv).
   ========================================================================== */
const PRESET_TARGETS = {
    'kepler-227b': {
        name: 'Kepler-227 b (K00752.01)',
        label: 'CONFIRMED',          // NASA catalogue disposition
        split: 'training',           // split used in the notebook (random_state=42)
        srad: 0.927,
        steff: 5455,
        period: 9.488,
        depth: 615.8,
        slogg: 4.467,
        duration: 2.958,
        shape: 'U',
        desc: 'Confirmed exoplanet in the Kepler-227 system, with a short orbit and a clear transit depth.'
    },
    'kepler-227c': {
        name: 'Kepler-227 c (K00752.02)',
        label: 'CONFIRMED',
        split: 'training',
        srad: 0.927,
        steff: 5455,
        period: 54.418,
        depth: 874.8,
        slogg: 4.467,
        duration: 4.507,
        shape: 'U',
        desc: 'The second planet in the Kepler-227 system, with a wider orbit and a longer transit.'
    },
    'kepler-664b': {
        name: 'Kepler-664 b (K00755.01)',
        label: 'CONFIRMED',
        split: 'training',
        srad: 1.046,
        steff: 6031,
        period: 2.526,
        depth: 603.3,
        slogg: 4.438,
        duration: 1.655,
        shape: 'U',
        desc: 'Confirmed planet with a 2.53-day period and a very stable transit signal.'
    },
    'false-binary': {
        name: 'K00753.01 (Eclipsing Binary)',
        label: 'FALSE POSITIVE',
        split: 'training',
        srad: 0.868,
        steff: 5853,
        period: 19.899,
        depth: 10829,
        slogg: 4.544,
        duration: 1.782,
        shape: 'V',
        desc: 'Eclipsing binary star (false positive). A very large, V-shaped drop in brightness (>10,000 ppm) that mimics an exoplanet.'
    },
    'super-earth': {
        name: 'Hypothetical super-Earth (not a real KOI)',
        hypothetical: true,
        srad: 0.65,
        steff: 3800,
        period: 35.50,
        depth: 450.0,
        slogg: 4.70,
        duration: 3.80,
        shape: 'U',
        desc: 'Hypothetical example (not a real KOI): a super-Earth of about 1.5 Earth radii orbiting a cool dwarf star.'
    },
    'hot-jupiter': {
        name: 'Hypothetical hot Jupiter (not a real KOI)',
        hypothetical: true,
        srad: 1.20,
        steff: 6200,
        period: 1.50,
        depth: 12500.0,
        slogg: 4.20,
        duration: 2.20,
        shape: 'U',
        desc: 'Hypothetical example (not a real KOI): a gas giant orbiting very close to its star, producing a very deep dip with a flat, U-shaped bottom.'
    }
};

/* Real outputs of the saved models in ./weight/ for the four real KOIs above.
   Computed offline from each object's 36 catalogue features (median-imputed and scaled with
   the training-set scaler, exactly as in the notebook), because the models cannot run on a
   static web page. All four objects were in the training split, so these are in-sample. */
const PRESET_MODEL_OUTPUTS = {
    'kepler-227b': {
        'Random Forest': { 'CONFIRMED': 0.98,     'CANDIDATE': 0.02,     'FALSE POSITIVE': 0.0 },
        'XGBoost':       { 'CONFIRMED': 0.92546,  'CANDIDATE': 0.072498, 'FALSE POSITIVE': 0.002042 },
        '1D-CNN':        { 'CONFIRMED': 0.886719, 'CANDIDATE': 0.110791, 'FALSE POSITIVE': 0.00249 }
    },
    'kepler-227c': {
        'Random Forest': { 'CONFIRMED': 0.99,     'CANDIDATE': 0.01,     'FALSE POSITIVE': 0.0 },
        'XGBoost':       { 'CONFIRMED': 0.978217, 'CANDIDATE': 0.020429, 'FALSE POSITIVE': 0.001354 },
        '1D-CNN':        { 'CONFIRMED': 0.951819, 'CANDIDATE': 0.046411, 'FALSE POSITIVE': 0.001769 }
    },
    'kepler-664b': {
        'Random Forest': { 'CONFIRMED': 0.83,     'CANDIDATE': 0.05,     'FALSE POSITIVE': 0.12 },
        'XGBoost':       { 'CONFIRMED': 0.888496, 'CANDIDATE': 0.055386, 'FALSE POSITIVE': 0.056118 },
        '1D-CNN':        { 'CONFIRMED': 0.72779,  'CANDIDATE': 0.165385, 'FALSE POSITIVE': 0.106825 }
    },
    'false-binary': {
        'Random Forest': { 'CONFIRMED': 0.04,     'CANDIDATE': 0.12,     'FALSE POSITIVE': 0.84 },
        'XGBoost':       { 'CONFIRMED': 0.001982, 'CANDIDATE': 0.069824, 'FALSE POSITIVE': 0.928194 },
        '1D-CNN':        { 'CONFIRMED': 0.02884,  'CANDIDATE': 0.408954, 'FALSE POSITIVE': 0.562205 }
    }
};

/* Test-set results recorded in the notebook (1,913 test KOIs). */
const MODEL_INFO = {
    'XGBoost': {
        color: '#1e40af',
        title: '⚡ XGBoost Pure Physics',
        file: 'weight/xgboost_physics_model.json',
        metrics: 'test accuracy 79.3% · macro-F1 0.76 · ROC-AUC 0.966 · 5-fold CV macro-F1 0.739 ± 0.010'
    },
    'Random Forest': {
        color: '#0e7a3a',
        title: '🌲 Random Forest Pure Physics',
        file: 'weight/random_forest_baseline.pkl',
        metrics: 'test accuracy 78.5% · precision/recall CONFIRMED 0.83/0.82, FALSE POSITIVE 0.86/0.86 · trained after SMOTE balancing (4,018 samples per class)'
    },
    '1D-CNN': {
        color: '#4338ca',
        title: '🧠 PyTorch 1D-CNN Pure Physics',
        file: 'weight/1d_cnn_weights.pth',
        metrics: 'test accuracy 74.9% · ROC-AUC 0.959 · hardest class CANDIDATE (precision 0.50, recall 0.62)'
    }
};

let currentModel = 'XGBoost';
let currentTargetKey = 'kepler-227b';
let backendAvailable = false;
let requestCounter = 0;

function initPresetSelector() {
    const presetBtns = document.querySelectorAll('.preset-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const targetKey = btn.getAttribute('data-preset');
            currentTargetKey = targetKey;
            const data = PRESET_TARGETS[targetKey];

            if (data) {
                setSliderValue('slider-srad', data.srad);
                setSliderValue('slider-steff', data.steff);
                setSliderValue('slider-period', data.period);
                setSliderValue('slider-depth', data.depth);
                setSliderValue('slider-slogg', data.slogg);
                setSliderValue('slider-duration', data.duration);

                updateInferenceEngine();
            }
        });
    });
}

function setSliderValue(id, val) {
    const slider = document.getElementById(id);
    if (slider) {
        slider.value = val;
        const display = document.getElementById(id + '-val');
        if (display) {
            display.textContent = val + (slider.getAttribute('data-unit') || '');
        }
    }
}

/* Inputs for the current view: the exact catalogue values for a preset,
   otherwise the slider positions. */
function getCurrentInputs() {
    const preset = PRESET_TARGETS[currentTargetKey];
    if (preset) {
        return { srad: preset.srad, steff: preset.steff, period: preset.period, depth: preset.depth,
                 slogg: preset.slogg, duration: preset.duration, shape: preset.shape };
    }
    const ids = ['srad', 'steff', 'period', 'depth', 'slogg', 'duration'];
    const v = {};
    for (const id of ids) {
        const el = document.getElementById('slider-' + id);
        if (!el) return null;
        v[id] = parseFloat(el.value);
    }
    v.shape = v.depth > 7000 ? 'V' : 'U';
    return v;
}

/* ==========================================================================
   4. SLIDERS & LIVE TELEMETRY UPDATES
   ========================================================================== */
function initSliders() {
    const sliders = document.querySelectorAll('.telemetry-slider');
    sliders.forEach(slider => {
        slider.addEventListener('input', (e) => {
            const val = e.target.value;
            const unit = e.target.getAttribute('data-unit') || '';
            const displayId = e.target.id + '-val';
            const display = document.getElementById(displayId);
            if (display) {
                display.textContent = val + unit;
            }

            // Deselect presets when the user moves a slider by hand
            document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
            currentTargetKey = 'custom';

            updateInferenceEngine();
        });
    });
}

/* ==========================================================================
   5. MODEL SWITCHER (A/B TESTING)
   ========================================================================== */
function initModelSwitcher() {
    const switchBtns = document.querySelectorAll('.switch-btn');
    switchBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            switchBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            currentModel = btn.getAttribute('data-model') || 'XGBoost';
            updateInferenceEngine();
        });
    });
}

/* ==========================================================================
   6. CLASSIFICATION ENGINE
   - Real Kepler objects: precomputed outputs of the saved models.
   - Anything else: the local Python backend (app.py) if it is running,
     otherwise a clearly labelled rule-based illustration.
   ========================================================================== */
function detectBackend() {
    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: 'XGBoost' })
    })
    .then(res => (res.ok ? res.json() : null))
    .then(data => {
        if (data && data.probs && data.source === 'model') {
            backendAvailable = true;
            const status = document.getElementById('status-text');
            if (status) status.textContent = 'Python inference server connected';
            updateInferenceEngine();
        }
    })
    .catch(() => { /* static hosting: no backend */ });
}

function updateInferenceEngine() {
    const v = getCurrentInputs();
    if (!v) return;
    const requestId = ++requestCounter;
    const preset = PRESET_TARGETS[currentTargetKey];
    const precomputed = PRESET_MODEL_OUTPUTS[currentTargetKey];

    updatePresetDescription(preset);
    renderLightcurve(v.depth, v.duration, v.shape);

    if (precomputed) {
        renderPredictionResults(precomputed[currentModel], 'precomputed');
        updateResultDetails('precomputed', v, preset);
        return;
    }

    const showRules = () => {
        if (requestId !== requestCounter) return;
        renderPredictionResults(ruleBasedIllustration(v), 'rules');
        updateResultDetails('rules', v, preset);
    };

    if (!backendAvailable) {
        showRules();
        return;
    }

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: currentModel,
            srad: v.srad, steff: v.steff, period: v.period, depth: v.depth, slogg: v.slogg, duration: v.duration
        })
    })
    .then(res => (res.ok ? res.json() : Promise.reject(res.status)))
    .then(data => {
        if (requestId !== requestCounter) return;
        if (data.source === 'model') {
            renderPredictionResults(data.probs, 'backend');
            updateResultDetails('backend', v, preset);
        } else {
            showRules();
        }
    })
    .catch(showRules);
}

/* Simple rules of thumb, NOT a trained model. */
function ruleBasedIllustration(v) {
    let probConfirmed, probCandidate, probFalsePos;
    if (v.depth > 8000 && v.srad < 1.0) {
        // Eclipsing-binary signature: very deep dip on a small star
        probFalsePos = 0.82; probCandidate = 0.12; probConfirmed = 0.06;
    } else if (v.depth >= 100 && v.depth <= 5000 && v.period > 1.0 && v.srad >= 0.5 && v.srad <= 2.0) {
        // Typical planetary transit signature
        probConfirmed = 0.55; probCandidate = 0.30; probFalsePos = 0.15;
    } else if (v.depth < 100 || v.period > 100) {
        // Weak or long-period signal -> stays a candidate
        probCandidate = 0.55; probConfirmed = 0.20; probFalsePos = 0.25;
    } else {
        probFalsePos = 0.45; probCandidate = 0.35; probConfirmed = 0.20;
    }
    return { 'CONFIRMED': probConfirmed, 'CANDIDATE': probCandidate, 'FALSE POSITIVE': probFalsePos };
}

function renderPredictionResults(probs, source) {
    const resultBox = document.getElementById('prediction-result-box');
    const statusEl = document.getElementById('pred-status-label');
    const titleEl = document.getElementById('pred-main-title');
    const confEl = document.getElementById('pred-conf-score');

    // Top class
    let prediction = 'CONFIRMED';
    let maxProb = probs['CONFIRMED'];
    if (probs['CANDIDATE'] > maxProb) { prediction = 'CANDIDATE'; maxProb = probs['CANDIDATE']; }
    if (probs['FALSE POSITIVE'] > maxProb) { prediction = 'FALSE POSITIVE'; maxProb = probs['FALSE POSITIVE']; }
    const confidence = (maxProb * 100).toFixed(1);

    if (resultBox && titleEl && confEl) {
        resultBox.className = 'prediction-result-box';

        if (prediction === 'CONFIRMED') {
            resultBox.classList.add('confirmed');
            titleEl.textContent = '🌟 CONFIRMED EXOPLANET';
            titleEl.style.color = 'var(--status-confirmed)';
        } else if (prediction === 'CANDIDATE') {
            resultBox.classList.add('candidate');
            titleEl.textContent = '🔭 PLANETARY CANDIDATE';
            titleEl.style.color = 'var(--status-candidate)';
        } else {
            resultBox.classList.add('false-positive');
            titleEl.textContent = '⚠️ FALSE POSITIVE';
            titleEl.style.color = 'var(--status-false-pos)';
        }

        if (source === 'rules') {
            confEl.textContent = `Rule-based estimate: ${confidence}% (no trained model)`;
        } else {
            confEl.textContent = `AI confidence (${currentModel}): ${confidence}%`;
        }
    }

    if (statusEl) {
        statusEl.textContent = {
            precomputed: 'AI CLASSIFICATION RESULT · SAVED MODEL (PRECOMPUTED)',
            backend: 'AI CLASSIFICATION RESULT · LIVE PYTHON BACKEND',
            rules: 'RULE-BASED ILLUSTRATION · NOT A TRAINED MODEL'
        }[source];
    }

    // Update probability bars
    const bars = [
        ['bar-confirmed', 'val-confirmed', 'CONFIRMED'],
        ['bar-candidate', 'val-candidate', 'CANDIDATE'],
        ['bar-false-pos', 'val-false-pos', 'FALSE POSITIVE']
    ];
    for (const [barId, valId, key] of bars) {
        const bar = document.getElementById(barId);
        const val = document.getElementById(valId);
        if (bar && val) {
            const p = (probs[key] * 100).toFixed(1);
            bar.style.width = p + '%';
            val.textContent = p + '%';
        }
    }
}

/* ==========================================================================
   7. LIVE TRANSIT LIGHTCURVE SIMULATOR (CANVAS, ILLUSTRATION ONLY)
   ========================================================================== */
function renderLightcurve(depth = 615.8, duration = 3.0, shape = 'U') {
    const canvas = document.getElementById('lightcurve-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const containerWidth = canvas.parentElement.clientWidth || 600;
    const width = canvas.width = containerWidth - 56;
    const height = canvas.height = 220;

    ctx.clearRect(0, 0, width, height);

    // Background: deep aerospace monitor theme
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 50) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
    }
    for (let y = 0; y < height; y += 35) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }

    // Normal flux line (y = 50)
    const baseFluxY = 50;

    // Dip size from depth (maximum dip down to y = 190), width from duration
    const dipMagnitude = Math.min(Math.max((depth / 10000) * 120, 20), 140);
    const dipWidth = Math.min(Math.max(duration * 26, 45), width * 0.6);
    const centerX = width / 2;

    // Draw the light curve
    ctx.beginPath();
    ctx.moveTo(0, baseFluxY);

    const isVShape = shape === 'V';

    for (let x = 0; x <= width; x += 2) {
        let y = baseFluxY;
        const distFromCenter = Math.abs(x - centerX);

        if (distFromCenter < dipWidth / 2) {
            if (isVShape) {
                const factor = 1 - (distFromCenter / (dipWidth / 2));
                y = baseFluxY + (dipMagnitude * factor);
            } else {
                const factor = Math.cos((distFromCenter / (dipWidth / 2)) * (Math.PI / 2));
                y = baseFluxY + (dipMagnitude * Math.pow(factor, 0.6));
            }
        }

        y += (Math.random() - 0.5) * 2.5;
        ctx.lineTo(x, y);
    }

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3.5;
    ctx.shadowColor = '#0284c7';
    ctx.shadowBlur = 14;
    ctx.stroke();
    ctx.shadowBlur = 0;

    const lowestY = baseFluxY + dipMagnitude;
    ctx.beginPath();
    ctx.arc(centerX, lowestY, 6, 0, Math.PI * 2);
    ctx.fillStyle = '#f59e0b';
    ctx.shadowColor = '#f59e0b';
    ctx.shadowBlur = 12;
    ctx.fill();
    ctx.shadowBlur = 0;

    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px "JetBrains Mono", monospace';
    ctx.fillText('Normalized stellar flux = 1.0', 16, 28);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 13px "JetBrains Mono", monospace';
    ctx.fillText(`Transit depth -${depth} ppm | duration ${duration} h`, Math.max(16, centerX - 120), lowestY - 14);
}

/* ==========================================================================
   8. RESULT DETAILS (WHAT PRODUCED THE NUMBERS ABOVE)
   ========================================================================== */
function updatePresetDescription(preset) {
    const el = document.getElementById('preset-desc');
    if (!el) return;
    el.textContent = preset ? preset.desc : 'Custom values set with the sliders.';
}

function updateResultDetails(source, v, preset) {
    const xaiContent = document.getElementById('xai-content');
    if (!xaiContent) return;
    const info = MODEL_INFO[currentModel];
    let html = '';

    if (source === 'precomputed') {
        html = `<div style="margin-bottom: 0.6rem; color: ${info.color}; font-weight: 800; font-size: 1.05rem;">${info.title} (${info.metrics}):</div>
        <strong>${preset.name}</strong> — NASA catalogue label: <strong>${preset.label}</strong>.<br>
        These probabilities are the real output of the saved model (<code style="overflow-wrap: anywhere;">${info.file}</code>) for this object's 36 catalogue features, precomputed offline because the models cannot run on a static web page. The object was in the <strong>${preset.split}</strong> split, so this is an in-sample prediction and will look more confident than for a new object.`;
    } else if (source === 'backend') {
        html = `<div style="margin-bottom: 0.6rem; color: ${info.color}; font-weight: 800; font-size: 1.05rem;">${info.title} — live output from the Python backend:</div>
        Live output of the saved model (<code style="overflow-wrap: anywhere;">${info.file}</code>) from <code>app.py</code>. Only the 6 slider values (depth <strong>${v.depth} ppm</strong>, period <strong>${v.period} days</strong>, star radius <strong>${v.srad} R☉</strong>, …) are sent; the other 30 features are set to their training-set averages, so treat this as a what-if illustration rather than a real classification.`;
    } else {
        html = `<div style="margin-bottom: 0.6rem; color: #b45309; font-weight: 800; font-size: 1.05rem;">📐 Rule-based illustration (no trained model):</div>
        ${preset && preset.hypothetical ? '<strong>' + preset.name + '</strong>. ' : ''}The trained models need all 36 catalogue features, so on this static page they only run for the four real Kepler objects. These bars come from simple rules of thumb instead: very deep dips around small stars look like eclipsing binaries, 100–5,000 ppm dips with periods over 1 day look planet-like, and very shallow dips or periods over 100 days stay candidates. The model buttons do not change this estimate. Run the Python backend (<code>python app.py</code>) to score custom values with the real models.`;
    }

    xaiContent.innerHTML = html;
}

/* ==========================================================================
   9. LIGHTBOX MODAL FOR RESEARCH PLOTS
   ========================================================================== */
function initLightbox() {
    const modal = document.getElementById('plot-modal');
    const modalImg = document.getElementById('modal-img');
    const closeBtn = document.getElementById('modal-close');
    const plotContainers = document.querySelectorAll('.plot-img-container');

    if (!modal || !modalImg) return;

    plotContainers.forEach(container => {
        container.addEventListener('click', () => {
            const img = container.querySelector('img');
            if (img) {
                modalImg.src = img.src;
                modal.classList.add('active');
            }
        });
    });

    if (closeBtn) {
        closeBtn.addEventListener('click', () => {
            modal.classList.remove('active');
        });
    }

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.remove('active');
        }
    });
}
