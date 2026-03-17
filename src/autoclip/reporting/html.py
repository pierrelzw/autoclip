"""Self-contained HTML report generation for AutoClip analysis."""

from __future__ import annotations

import json

from autoclip.models import AnalysisResult


def generate_report_html(
    analysis: AnalysisResult,
    video_relative_path: str,
) -> str:
    """Generate a self-contained HTML report from analysis data.

    The report includes:
    - Statistics summary (durations, reduction%, category counts)
    - Transcript diff view with removed words highlighted
    - Timeline visualization
    - Video preview with click-to-seek

    All CSS, JS, and data are inlined. No external resources.

    Args:
        analysis: The analysis result data.
        video_relative_path: Relative path from report to original video.

    Returns:
        Complete HTML string.
    """
    data_json = analysis.model_dump_json()
    # Escape </script> sequences to prevent breaking out of inline <script> tags
    safe_json = data_json.replace("</", "<\\/")
    return _build_html(safe_json, video_relative_path)


def _build_html(data_json: str, video_relative_path: str) -> str:
    """Build the full HTML document with embedded data and styles."""
    # Escape for safe embedding inside <script> tags
    escaped_video_path = json.dumps(video_relative_path).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AutoClip Analysis Report</title>
{_css()}
</head>
<body>
<div id="app"></div>
<script>
const data = {data_json};
const videoPath = {escaped_video_path};
</script>
{_js()}
</body>
</html>"""


def _css() -> str:
    """Return all CSS styles as an inline style block."""
    return """<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f0f0f0; color: #333; line-height: 1.6; }
.container { max-width: 1400px; margin: 0 auto; padding: 20px; }
.header { text-align: center; margin-bottom: 24px; }
.header h1 { font-size: 28px; color: #1a1a1a; }
.header .subtitle { font-size: 16px; color: #666; margin-top: 4px; }
.card { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.card h2 { font-size: 18px; margin-bottom: 12px; color: #1a1a1a; }

/* Stats */
.stats-toggle { cursor: pointer; color: #4a90d9; font-size: 14px; margin-top: 8px; user-select: none; }
.stats-toggle:hover { text-decoration: underline; }
.stats-detail { display: none; margin-top: 12px; }
.stats-detail.open { display: block; }
.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; }
.stat-item { padding: 8px 12px; background: #f8f8f8; border-radius: 4px; }
.stat-item .label { font-size: 12px; color: #888; text-transform: uppercase; }
.stat-item .value { font-size: 20px; font-weight: 600; color: #1a1a1a; }

/* Two-column layout */
.content-row { display: flex; gap: 20px; }
.content-left { flex: 1; min-width: 0; }
.content-right { width: 400px; flex-shrink: 0; }

/* Transcript */
.transcript-box { max-height: 500px; overflow-y: auto; padding: 12px; background: #fafafa; border-radius: 4px; border: 1px solid #eee; }
.word { display: inline; cursor: pointer; padding: 1px 2px; border-radius: 2px; }
.word:hover { background: #e8f0fe; }
.word.removed { text-decoration: line-through; color: #d32f2f; opacity: 0.7; }
.word.removed:hover { opacity: 1; }
.badge { display: inline-block; font-size: 10px; padding: 1px 5px; border-radius: 3px; margin-left: 2px; vertical-align: middle; font-weight: 600; }
.badge-filler { background: #ffcdd2; color: #b71c1c; }
.badge-repeat { background: #ffe0b2; color: #e65100; }
.badge-false-start { background: #e1bee7; color: #6a1b9a; }
.badge-pause { background: #fff9c4; color: #f57f17; }
.badge-llm_suggested { background: #c8e6c9; color: #1b5e20; }
.pause-pill { display: inline-block; background: #fff3e0; color: #e65100; font-size: 11px; padding: 2px 8px; border-radius: 12px; margin: 0 4px; vertical-align: middle; font-weight: 500; cursor: pointer; }
.pause-pill.removed { text-decoration: line-through; background: #ffcdd2; color: #b71c1c; opacity: 0.7; }
.pause-pill.removed:hover { opacity: 1; }
.legend { margin-top: 12px; font-size: 12px; color: #888; }
.legend span { margin-right: 12px; }

/* Timeline */
.timeline-container { position: relative; margin: 12px 0; }
.timeline-bar { display: flex; height: 28px; border-radius: 4px; overflow: hidden; background: #eee; }
.timeline-segment { height: 100%; position: relative; }
.timeline-segment.kept { background: #66bb6a; }
.timeline-segment.removed { background: #ef5350; }
.timeline-tooltip { display: none; position: absolute; bottom: 34px; left: 50%; transform: translateX(-50%); background: #333; color: #fff; padding: 4px 8px; border-radius: 4px; font-size: 11px; white-space: nowrap; z-index: 10; pointer-events: none; }
.timeline-segment:hover .timeline-tooltip { display: block; }
.timeline-labels { display: flex; justify-content: space-between; font-size: 11px; color: #888; margin-top: 4px; }
.timeline-legend { font-size: 12px; color: #888; margin-top: 8px; }
.timeline-legend span { margin-right: 16px; }
.timeline-legend .dot { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: middle; }
.dot-kept { background: #66bb6a; }
.dot-removed { background: #ef5350; }

/* Video */
.video-wrapper video { width: 100%; border-radius: 4px; background: #000; }
.video-fallback { padding: 40px; text-align: center; color: #999; background: #f5f5f5; border-radius: 4px; }

@media (max-width: 768px) {
    .content-row { flex-direction: column; }
    .content-right { width: 100%; }
}
</style>"""


def _js() -> str:
    """Return all JavaScript as an inline script block."""
    return """<script>
(function() {
    const app = document.getElementById('app');
    const candidates = data.candidates || [];
    const words = data.words || [];
    const params = data.applied_params || {};
    const threshold = params.threshold || 0.7;
    const categories = params.categories || [];
    const categoryMap = {
        'stutter': 'repeat', 'repeat': 'repeat', 'filler': 'filler',
        'false_start': 'false-start', 'long_pause': 'pause'
    };
    const cliToInternal = {
        'filler': ['filler'], 'repeat': ['stutter', 'repeat'],
        'false-start': ['false_start'], 'pause': ['long_pause']
    };

    // Derive which candidates are applied
    const activeReasons = new Set();
    categories.forEach(cat => {
        (cliToInternal[cat] || []).forEach(r => activeReasons.add(r));
    });

    const removalMap = {};
    candidates.forEach(c => {
        if (!activeReasons.has(c.reason)) return;
        let effThreshold = threshold;
        if (c.reason === 'false_start') effThreshold = Math.max(threshold, 0.85);
        if (c.confidence >= effThreshold) {
            removalMap[c.word_id] = c;
        }
    });

    // Candidate lookup by word_id
    const candidateMap = {};
    candidates.forEach(c => { candidateMap[c.word_id] = c; });

    // Stats — compute cleaned duration from retained real words (matching pipeline logic)
    const originalDur = data.original_duration_sec || 0;
    const appliedList = Object.values(removalMap);
    const removedIds = new Set(Object.keys(removalMap));
    const retainedReal = words.filter(w => !w.text.startsWith('[') && !removedIds.has(w.id));
    // Merge adjacent retained words into segments (gap <= 0.1s)
    let cleanedDur = 0;
    if (retainedReal.length > 0) {
        let segStart = retainedReal[0].start_sec, segEnd = retainedReal[0].end_sec;
        for (let i = 1; i < retainedReal.length; i++) {
            if (retainedReal[i].start_sec - segEnd <= 0.1) {
                segEnd = retainedReal[i].end_sec;
            } else {
                cleanedDur += segEnd - segStart;
                segStart = retainedReal[i].start_sec;
                segEnd = retainedReal[i].end_sec;
            }
        }
        cleanedDur += segEnd - segStart;
    }
    const reductionPct = originalDur > 0 ? ((originalDur - cleanedDur) / originalDur * 100) : 0;

    // Count by CLI category
    const catCounts = {};
    appliedList.forEach(c => {
        const cc = categoryMap[c.reason] || c.reason;
        catCounts[cc] = (catCounts[cc] || 0) + 1;
    });

    function fmtTime(sec) {
        const m = Math.floor(sec / 60);
        const s = (sec % 60).toFixed(1);
        return m > 0 ? m + 'm ' + s + 's' : s + 's';
    }

    function getBadgeLabel(candidate) {
        if (candidate.source === 'llm') return 'llm_suggested';
        return categoryMap[candidate.reason] || candidate.reason;
    }

    function getBadgeClass(label) {
        return 'badge badge-' + label.replace('-', '-');
    }

    // Build HTML
    let html = '<div class="container">';

    // Header
    html += '<div class="header">';
    html += '<h1>AutoClip Analysis Report</h1>';
    html += '<p class="subtitle">' + fmtTime(originalDur) + ' \\u2192 ' + fmtTime(cleanedDur) + ' (' + reductionPct.toFixed(1) + '% reduction)</p>';
    html += '</div>';

    // Stats card
    html += '<div class="card">';
    html += '<h2>Summary</h2>';
    html += '<span class="stats-toggle" onclick="document.getElementById(&quot;stats-detail&quot;).classList.toggle(&quot;open&quot;)">\\u25B6 Show details</span>';
    html += '<div id="stats-detail" class="stats-detail"><div class="stats-grid">';
    html += '<div class="stat-item"><div class="label">Original</div><div class="value">' + fmtTime(originalDur) + '</div></div>';
    html += '<div class="stat-item"><div class="label">Cleaned</div><div class="value">' + fmtTime(cleanedDur) + '</div></div>';
    html += '<div class="stat-item"><div class="label">Reduction</div><div class="value">' + reductionPct.toFixed(1) + '%</div></div>';
    html += '<div class="stat-item"><div class="label">Language</div><div class="value">' + escHtml(data.detected_language || 'unknown') + '</div></div>';
    html += '<div class="stat-item"><div class="label">Threshold</div><div class="value">' + threshold + '</div></div>';
    html += '<div class="stat-item"><div class="label">Words</div><div class="value">' + words.length + '</div></div>';
    Object.keys(catCounts).sort().forEach(cat => {
        html += '<div class="stat-item"><div class="label">' + cat + '</div><div class="value">' + catCounts[cat] + '</div></div>';
    });
    html += '</div></div></div>';

    // Two-column: transcript + video
    html += '<div class="content-row">';

    // Transcript
    html += '<div class="content-left"><div class="card">';
    html += '<h2>Transcript</h2>';
    html += '<div class="transcript-box">';
    words.forEach(w => {
        const isRemoved = !!removalMap[w.id];
        const candidate = candidateMap[w.id];
        const isPause = w.text === '[PAUSE]';
        if (isPause) {
            const dur = (w.end_sec - w.start_sec).toFixed(1);
            const pauseCls = isRemoved ? 'pause-pill removed' : 'pause-pill';
            html += '<span class="' + pauseCls + '" data-start="' + w.start_sec + '">[PAUSE ' + dur + 's]</span> ';
        } else if (isRemoved && candidate) {
            const label = getBadgeLabel(candidate);
            html += '<span class="word removed" data-start="' + w.start_sec + '">' + escHtml(w.text) + '<span class="' + getBadgeClass(label) + '">' + label + '</span></span> ';
        } else {
            html += '<span class="word" data-start="' + w.start_sec + '">' + escHtml(w.text) + '</span> ';
        }
    });
    html += '</div>';
    html += '<div class="legend">';
    html += '<span><span class="badge badge-filler">filler</span> Keyword match</span>';
    html += '<span><span class="badge badge-repeat">repeat</span> Stutter/repeat</span>';
    html += '<span><span class="badge badge-false-start">false-start</span> False start</span>';
    html += '<span><span class="badge badge-pause">pause</span> Long pause</span>';
    html += '<span><span class="badge badge-llm_suggested">llm_suggested</span> LLM detected</span>';
    html += '</div>';
    html += '</div></div>';

    // Video
    html += '<div class="content-right"><div class="card">';
    html += '<h2>Video Preview</h2>';
    html += '<div class="video-wrapper">';
    if (videoPath) {
        html += '<video id="video-player" controls preload="metadata" src="' + escAttr(videoPath) + '">';
        html += '</video>';
        html += '<div class="video-fallback" style="display:none" id="video-fallback">Video not available. Place the original video file at: ' + escHtml(videoPath) + '</div>';
    } else {
        html += '<div class="video-fallback" id="video-fallback">Video preview not available (source was a URL).</div>';
    }
    html += '</div>';
    html += '</div></div>';

    html += '</div>'; // content-row

    // Timeline
    html += '<div class="card">';
    html += '<h2>Timeline</h2>';
    html += '<div class="timeline-container">';
    html += buildTimeline();
    html += '</div></div>';

    html += '</div>'; // container
    app.innerHTML = html;

    // Video error handler
    const vid = document.getElementById('video-player');
    if (vid) {
        vid.addEventListener('error', function() {
            vid.style.display = 'none';
            document.getElementById('video-fallback').style.display = 'block';
        });
    }

    // Click-to-seek
    document.addEventListener('click', function(e) {
        const el = e.target.closest('[data-start]');
        if (!el || !vid) return;
        const t = parseFloat(el.dataset.start);
        if (!isNaN(t)) {
            vid.currentTime = t;
            vid.play().catch(function(){});
        }
    });

    function buildTimeline() {
        if (words.length === 0) return '<div class="timeline-bar"></div>';

        // Collect all time segments
        const realWords = words.filter(w => !w.text.startsWith('['));
        if (realWords.length === 0) return '<div class="timeline-bar"></div>';

        const totalStart = realWords[0].start_sec;
        const totalEnd = realWords[realWords.length - 1].end_sec;
        const totalDur = totalEnd - totalStart;
        if (totalDur <= 0) return '<div class="timeline-bar"></div>';

        let segments = '';
        realWords.forEach(w => {
            const isRemoved = !!removalMap[w.id];
            const cand = candidateMap[w.id];
            const pct = ((w.end_sec - w.start_sec) / totalDur * 100);
            if (pct <= 0) return;
            const cls = isRemoved ? 'removed' : 'kept';
            let tooltip = escHtml(w.text) + ' (' + w.start_sec.toFixed(2) + 's - ' + w.end_sec.toFixed(2) + 's)';
            if (isRemoved && cand) {
                tooltip += ' - Removed: ' + getBadgeLabel(cand);
            }
            segments += '<div class="timeline-segment ' + cls + '" style="width:' + pct.toFixed(4) + '%">';
            segments += '<div class="timeline-tooltip">' + tooltip + '</div>';
            segments += '</div>';
        });

        let labels = '<div class="timeline-labels">';
        for (let i = 0; i < 5; i++) {
            const t = totalStart + (totalDur * i / 4);
            labels += '<span>' + fmtTime(t) + '</span>';
        }
        labels += '</div>';

        let legend = '<div class="timeline-legend">';
        legend += '<span><span class="dot dot-kept"></span>Kept segments</span>';
        legend += '<span><span class="dot dot-removed"></span>Removed segments</span>';
        legend += '</div>';

        return '<div class="timeline-bar">' + segments + '</div>' + labels + legend;
    }

    function escHtml(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
    }
    function escAttr(s) {
        return escHtml(s);
    }
})();
</script>"""
