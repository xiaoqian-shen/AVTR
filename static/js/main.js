/* AVTR project page: tabs, tables, and interactive widgets (no dependencies). */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ reveal
     Set up before anything else, so that an error further down can never leave .reveal content invisible.
     (CSS hides .reveal only under html.js; index.html drops that class if this script never runs.) */
  (function reveal() {
    const items = Array.prototype.slice.call(document.querySelectorAll('.reveal'));
    const showAll = () => items.forEach(r => r.classList.add('in'));
    try {
      if ('IntersectionObserver' in window) {
        const io = new IntersectionObserver(entries => entries.forEach(e => {
          if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
        }), { rootMargin: '0px 0px -8% 0px' });
        items.forEach(r => io.observe(r));
      } else {
        showAll();
      }
    } catch (err) {
      showAll();
    }
    window.AVTR_REVEAL = true;
  })();

  const D = window.AVTR_DATA || {};
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const fmt1 = v => (v == null ? '–' : Number(v).toFixed(1));
  /* run one widget; a failure is logged and never takes the rest of the page down */
  function safe(name, fn) {
    try { fn(); } catch (err) { if (window.console) console.error('[AVTR] ' + name + ' failed:', err); }
  }

  safe('header', () => {
    const header = $('.site-header');
    if (!header) return;
    const links = $$('.section-nav a[href^="#"]');
    const sections = links.map(link => ({ link, target: $(link.getAttribute('href')) })).filter(item => item.target);
    const update = () => {
      header.classList.toggle('is-scrolled', window.scrollY > 16);
      const threshold = header.getBoundingClientRect().height + 32;
      let current = null;
      sections.forEach(item => {
        const box = item.target.getBoundingClientRect();
        if (box.top <= threshold && box.bottom > threshold) current = item.link;
      });
      links.forEach(link => {
        if (link === current) {
          if (link.getAttribute('aria-current') !== 'location') link.setAttribute('aria-current', 'location');
        } else if (link.hasAttribute('aria-current')) {
          link.removeAttribute('aria-current');
        }
      });
    };
    window.addEventListener('scroll', update, { passive: true });
    window.addEventListener('resize', update);
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(update).catch(() => {});
    update();
  });

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    if (tag === 'button') node.type = 'button';
    if (attrs) Object.entries(attrs).forEach(([k, v]) => {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;
      else node.setAttribute(k, v);
    });
    (children || []).forEach(c => node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c));
    return node;
  }

  /* ------------------------------------------------------------------- tabs
     WAI-ARIA tabs: roving tabindex, ArrowLeft/ArrowRight/Home/End move focus and select (automatic activation). */
  function initTabs(group) {
    const list = $('.tab-buttons', group);
    if (!list) return;
    const tabs = $$('[data-tab]', list);
    if (!tabs.length) return;
    const panels = tabs.map(t => document.getElementById(t.dataset.tab));
    list.setAttribute('role', 'tablist');
    tabs.forEach((t, i) => {
      const p = panels[i];
      if (!t.id) t.id = t.dataset.tab + '-tab';
      t.type = 'button';
      t.setAttribute('role', 'tab');
      if (p) {
        t.setAttribute('aria-controls', p.id);
        p.setAttribute('role', 'tabpanel');
        p.setAttribute('aria-labelledby', t.id);
      }
    });
    function select(i, focus) {
      tabs.forEach((t, j) => {
        const on = j === i;
        t.classList.toggle('active', on);
        t.setAttribute('aria-selected', on ? 'true' : 'false');
        t.tabIndex = on ? 0 : -1;
        const p = panels[j];
        if (p) { p.classList.toggle('active', on); p.hidden = !on; }
      });
      // content that was hidden measured 0 px wide: let width-dependent layout (equations, route detail) refit
      window.dispatchEvent(new Event('resize'));
      if (focus) tabs[i].focus();
    }
    tabs.forEach((t, i) => {
      t.addEventListener('click', () => select(i, false));
      t.addEventListener('keydown', e => {
        let k = -1;
        if (e.key === 'ArrowRight') k = (i + 1) % tabs.length;
        else if (e.key === 'ArrowLeft') k = (i - 1 + tabs.length) % tabs.length;
        else if (e.key === 'Home') k = 0;
        else if (e.key === 'End') k = tabs.length - 1;
        if (k < 0) return;
        e.preventDefault();
        select(k, true);
      });
    });
    select(Math.max(0, tabs.findIndex(t => t.classList.contains('active'))), false);
  }
  $$('[data-tabs]').forEach(group => safe('tabs', () => initTabs(group)));

  /* ----------------------------------------------------------------- tables */
  function buildTable(spec) {
    const table = el('table', { class: 'results-table' });
    const thead = el('thead');
    const groupRow = el('tr', {}, [el('th', { class: 'model-col', rowspan: 2, scope: 'col', text: 'Model' })]);
    spec.groups.forEach(g => groupRow.appendChild(el('th', { class: 'group-head', colspan: g.span, scope: 'colgroup', text: g.label })));
    const colRow = el('tr');
    spec.cols.forEach(c => colRow.appendChild(el('th', { scope: 'col', text: c })));
    thead.appendChild(groupRow);
    thead.appendChild(colRow);
    table.appendChild(thead);

    // best value per column among non-reference rows; ties are all bold unless the Ours row is among them,
    // then only Ours is (a baseline that merely matches AVTR is not highlighted)
    const nCols = spec.cols.length;
    const best = new Array(nCols).fill(-Infinity);
    spec.sections.forEach(s => {
      if (s.reference) return;
      s.rows.forEach(r => r.slice(1).forEach((v, j) => {
        if (typeof v === 'number' && v > best[j]) best[j] = v;
      }));
    });
    const oursBest = new Array(nCols).fill(false);
    spec.sections.forEach(s => {
      if (s.reference) return;
      s.rows.forEach(r => { if (/Ours/.test(r[0])) r.slice(1).forEach((v, j) => { if (v === best[j]) oursBest[j] = true; }); });
    });

    const tbody = el('tbody');
    spec.sections.forEach(s => {
      if (s.label) tbody.appendChild(el('tr', { class: 'section-row' }, [el('td', { colspan: nCols + 1 }, [el('span', { class: 'section-label', text: s.label })])]));
      s.rows.forEach(r => {
        const ours = /Ours/.test(r[0]);
        const tr = el('tr', { class: ours ? 'ours-row' : (s.reference ? 'ref-row' : '') });
        tr.appendChild(el('td', { class: 'model-col', text: r[0] }));
        r.slice(1).forEach((v, j) => {
          const isBest = !s.reference && typeof v === 'number' && v === best[j] && (ours || !oursBest[j]);
          tr.appendChild(el('td', { class: isBest ? 'best' : '', text: fmt1(v) }));
        });
        tbody.appendChild(tr);
      });
    });
    table.appendChild(tbody);
    return table;
  }
  [['image', '#table-image'], ['video', '#table-video'], ['small', '#table-small']].forEach(([k, sel]) => safe('table ' + k, () => {
    const host = $(sel);
    if (host && D.tables && D.tables[k]) host.appendChild(buildTable(D.tables[k]));
  }));

  /* --------------------------------------------------------- route explorer */
  function traceNode(parts) {
    const wrap = el('div', { class: 'trace' });
    parts.forEach(([kind, text]) => {
      if (kind === 'queries') {
        const q = el('span', { class: 'q-cells', title: 'continuous visual queries' });
        for (let i = 0; i < 5; i++) q.appendChild(el('i'));
        wrap.appendChild(q);
      } else if (kind === 'lines') {
        const l = el('span', { class: 't-lines', title: 'textual reasoning trace' });
        for (let i = 0; i < 3; i++) l.appendChild(el('i'));
        wrap.appendChild(l);
      } else {
        wrap.appendChild(el('code', { class: 'tok ' + kind.replace('tok-', ''), text }));
      }
    });
    return wrap;
  }
  safe('routes', () => {
    const host = $('#route-explorer');
    const detail = $('.route-detail', host || document);
    if (!host || !detail || !D.routes) return;
    const cards = $$('.route-card', host);
    let current = (cards.find(c => c.classList.contains('active')) || {}).dataset;
    current = current && D.routes[current.route] ? current.route : Object.keys(D.routes)[0];

    function fill(root, key) {
      const r = D.routes[key];
      if (!r) return;
      const name = $('.route-name', root), trace = $('.route-trace', root), desc = $('.route-desc', root);
      if (name) { name.textContent = r.name; name.className = 'route-name ' + key; }
      if (trace) { trace.textContent = ''; trace.appendChild(traceNode(r.trace)); }
      if (desc) desc.textContent = r.desc;
    }
    function select(key) {
      current = key;
      cards.forEach(c => {
        const on = c.dataset.route === key;
        c.classList.toggle('active', on);
        c.setAttribute('aria-pressed', on ? 'true' : 'false');
      });
      fill(detail, key);
    }
    // reserve the height of the tallest route (measured offscreen) so clicking never shifts the layout
    function fitHeight() {
      const w = detail.getBoundingClientRect().width;
      if (!w) return;
      const probe = detail.cloneNode(true);
      probe.removeAttribute('id');
      probe.removeAttribute('aria-live');
      $$('[id]', probe).forEach(n => n.removeAttribute('id'));
      probe.setAttribute('aria-hidden', 'true');
      probe.style.cssText = `position:absolute;left:-10000px;top:0;visibility:hidden;width:${w}px;min-height:0;`;
      host.appendChild(probe);
      let maxH = 0;
      Object.keys(D.routes).forEach(k => { fill(probe, k); maxH = Math.max(maxH, probe.getBoundingClientRect().height); });
      probe.remove();
      if (maxH) detail.style.minHeight = Math.ceil(maxH) + 'px';
    }
    cards.forEach(c => c.addEventListener('click', () => select(c.dataset.route)));
    select(current);
    fitHeight();
    let raf = 0;
    window.addEventListener('resize', () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(fitHeight); });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(fitHeight).catch(() => {});
  });

  /* ---------------------------------------------------- soft-target playground
     Each slider sets a route's accuracy (%), used as its reward R_m; the route costs C(m) are never displayed. */
  safe('playground', () => {
    const ctrl = $('#pg-controls');
    const bars = $('#pg-bars');
    if (!ctrl || !bars) return;
    const routes = [
      { key: 'direct', name: 'Direct', cost: 0, acc: 25 },
      { key: 'visual', name: 'Visual', cost: 0.01, acc: 100 },
      { key: 'textual', name: 'Textual', cost: 0.04, acc: 50 },
      { key: 'both', name: 'Visual + Textual', cost: 0.05, acc: 75 },
    ];
    routes.forEach(r => {
      r.val = el('span', { class: 'pg-val', 'aria-hidden': 'true' });
      r.input = el('input', { type: 'range', min: 0, max: 100, step: 5, value: r.acc, 'aria-label': `${r.name} accuracy` });
      r.input.addEventListener('input', update);
      ctrl.appendChild(el('div', { class: 'pg-row ' + r.key }, [el('span', { class: 'pg-name', text: r.name }), r.input, r.val]));
      r.bar = el('div', { class: 'pg-bar-fill ' + r.key });
      r.pct = el('span', { class: 'pg-pct' });
      bars.appendChild(el('div', { class: 'pg-bar-row ' + r.key }, [
        el('span', { class: 'pg-name', text: r.name }), el('div', { class: 'pg-bar-track', 'aria-hidden': 'true' }, [r.bar]), r.pct,
      ]));
    });
    function update() {
      const z = routes.map(r => Number(r.input.value) / 100 - r.cost);   // R_m - C(m)
      const mx = Math.max(...z);
      const e = z.map(v => Math.exp(v - mx));
      const sum = e.reduce((a, b) => a + b, 0);
      const p = e.map(v => v / sum);
      const pmax = Math.max(...p);
      routes.forEach((r, i) => {
        const acc = Number(r.input.value);
        r.val.textContent = acc + '%';
        r.input.setAttribute('aria-valuetext', acc + '%');
        r.bar.style.width = (p[i] * 100).toFixed(1) + '%';   // honest: the track is the full 0–100% scale
        r.bar.classList.toggle('top', p[i] === pmax);
        r.pct.textContent = (p[i] * 100).toFixed(1) + '%';
      });
    }
    update();
  });

  /* ------------------------------------------------------------- qualitative */
  function answerRow(cls, text) {
    return el('div', { class: 'answer ' + cls }, [
      el('code', { class: 'tok answer', text: '<answer>' }), el('b', { text }), el('code', { class: 'tok answer', text: '</answer>' }),
    ]);
  }
  function imagineTrace(src) {
    return el('div', { class: 'vis-trace' }, [
      el('code', { class: 'tok visual', text: '<imagine>' }),
      el('img', { src, alt: 'Decoded visual state (visualization only)', loading: 'lazy' }),
      el('code', { class: 'tok visual', text: '</imagine>' }),
    ]);
  }
  function routeCard(kind, ex) {
    const isVis = kind === 'visual';
    const r = ex[kind];
    const card = el('div', { class: 'qcard ' + kind });
    card.appendChild(el('div', { class: 'qcard-head' }, [
      el('span', { class: 'qcard-title', text: isVis ? 'Visual route' : 'Textual route' }),
      el('span', { class: 'verdict ' + (r.correct ? 'ok' : 'bad'), html: r.correct ? '<i class="fas fa-check" aria-hidden="true"></i> correct' : '<i class="fas fa-xmark" aria-hidden="true"></i> wrong' }),
    ]));
    const body = el('div', { class: 'qcard-body' });
    if (isVis) {
      body.appendChild(imagineTrace('static/images/qual/' + r.image));
      body.appendChild(el('p', { class: 'vis-note', text: 'decoded for visualization only' }));
    } else {
      body.appendChild(el('div', { class: 'txt-trace' }, [
        el('code', { class: 'tok textual', text: '<think>' }),
        el('p', { text: r.head }),
        el('span', { class: 'omitted', text: '··· ' + r.omitted + ' ···' }),
        el('p', { text: r.tail }),
        el('code', { class: 'tok textual', text: '</think>' }),
      ]));
    }
    body.appendChild(answerRow(r.correct ? 'ok' : 'bad', r.answer));
    card.appendChild(body);
    return card;
  }
  function questionBar(q, opts, gt, src) {
    return el('div', { class: 'qbar' }, [
      el('div', { class: 'qbar-main' }, [
        el('span', { class: 'qbar-label', text: 'Question' }),
        el('div', {}, [el('p', { class: 'qbar-q', text: q }), opts ? el('p', { class: 'qbar-opts', text: opts.join('     ') }) : el('span')]),
      ]),
      el('div', { class: 'qbar-side' }, [gt ? el('span', { class: 'qbar-gt', text: 'GT: ' + gt }) : el('span'), el('span', { class: 'qbar-src', text: src || '' })]),
    ]);
  }
  Object.entries(D.qual || {}).forEach(([key, ex]) => safe('qual ' + key, () => {
    const host = $('#q-' + key);
    if (!host || !ex) return;
    host.appendChild(questionBar(ex.question, ex.options, ex.gt, ex.source));
    const inputs = el('div', { class: 'qinputs' });
    (ex.inputs || []).forEach(([img, cap]) => inputs.appendChild(el('figure', {}, [el('img', { src: 'static/images/qual/' + img, alt: cap, loading: 'lazy' }), el('figcaption', { text: cap })])));
    host.appendChild(inputs);
    host.appendChild(el('div', { class: 'qcards' }, [routeCard('visual', ex), routeCard('textual', ex)]));
  }));
  Object.entries(D.videoExamples || {}).forEach(([key, v]) => safe('video example ' + key, () => {
    const host = $('#q-' + key);
    if (!host || !v) return;
    host.appendChild(questionBar(v.question, null, null, v.source));
    const strip = el('div', { class: 'timeline' });
    v.times.forEach((t, i) => {
      const salient = i === v.salient;
      const kids = [];
      if (salient) kids.push(el('span', { class: 'salient-tag', text: 'salient frame' }));
      const label = t ? 'Frame at ' + t : 'Video frame ' + (i + 1);
      kids.push(el('img', { src: `static/images/qual/${v.framePrefix}_${String(i).padStart(2, '0')}.jpg`, alt: (salient ? 'Salient ' : '') + label, loading: 'lazy' }));
      kids.push(el('figcaption', { text: t || 'Frame ' + (i + 1) }));
      strip.appendChild(el('figure', { class: salient ? 'salient' : '' }, kids));
    });
    host.appendChild(strip);
    host.appendChild(el('div', { class: 'qcard visual single' }, [
      el('div', { class: 'qcard-head' }, [el('span', { class: 'qcard-title', text: 'Visual reflection' })]),
      el('div', { class: 'qcard-body row' }, [imagineTrace('static/images/qual/' + v.image), answerRow(v.answerClass || '', v.answer)]),
    ]));
  }));

  /* --------------------------------------------------------------- chapters */
  safe('chapters', () => {
    const video = $('.video-frame video');
    const bar = $('#chapters');
    if (!video || !bar) return;
    const btns = $$('button[data-t]', bar);
    const playback = $('#video-toggle');
    if (!btns.length) return;
    const starts = btns.map(b => Number(b.dataset.t));
    function mark() {
      const t = video.currentTime || 0;
      let idx = -1;
      starts.forEach((s, i) => { if (t >= s - 0.05) idx = i; });
      btns.forEach((b, i) => {
        const on = i === idx;
        b.classList.toggle('active', on);
        if (on) b.setAttribute('aria-current', 'true'); else b.removeAttribute('aria-current');
      });
    }
    let pending = null;   // seek target requested before the metadata was available
    function seekPending() {
      if (pending == null) return;
      const t = pending;
      pending = null;
      try { video.currentTime = t; } catch (e) { /* ignore */ }
    }
    function play() {
      const p = video.play();
      if (p && p.catch) p.catch(() => {});
    }
    btns.forEach((b, i) => b.addEventListener('click', () => {
      if (video.readyState < 1) {             // HAVE_NOTHING: seeking now would be ignored
        if (pending == null) video.addEventListener('loadedmetadata', seekPending, { once: true });
        pending = starts[i];
        play();                                // also starts loading where preload is ignored (iOS)
      } else {
        try { video.currentTime = starts[i]; } catch (e) { /* ignore */ }
        play();
      }
    }));
    // no native controls (so no progress bar): the chapter buttons seek and play; clicking the video, or Space/Enter
    // while it has focus, pauses and resumes
    function toggle() { if (video.paused || video.ended) play(); else video.pause(); }
    function syncPlayback() {
      if (!playback) return;
      const action = video.ended ? 'Replay' : video.paused ? 'Play' : 'Pause';
      $('i', playback).className = video.paused || video.ended ? 'fas fa-play' : 'fas fa-pause';
      playback.setAttribute('aria-label', action + ' walkthrough video');
      playback.title = action + ' video';
      playback.classList.toggle('is-playing', !video.paused && !video.ended);
    }
    if (playback) playback.addEventListener('click', toggle);
    ['play', 'pause', 'ended'].forEach(event => video.addEventListener(event, syncPlayback));
    video.addEventListener('click', toggle);
    video.addEventListener('keydown', e => {
      if (e.key === ' ' || e.key === 'Enter') { e.preventDefault(); toggle(); }
    });
    // autoplay (muted, so browsers allow it; the video has no audio anyway), unless the visitor asked for reduced motion
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) video.pause();
    else if (video.paused) play();             // the autoplay attribute alone is ignored in some setups
    video.addEventListener('timeupdate', mark);
    video.addEventListener('seeked', mark);
    video.addEventListener('loadedmetadata', mark);
    mark();
    syncPlayback();
  });

  /* ------------------------------------------------------------------ bibtex */
  safe('copy', () => {
    const btn = $('#copy-bib');
    const pre = $('#bib-text');
    if (!btn || !pre) return;
    const isApple = /Mac|iPhone|iPad|iPod/i.test(navigator.platform || navigator.userAgent || '');
    const DEFAULT = ['far fa-copy', 'Copy'];
    let timer = 0;
    function setLabel(icon, text) {
      btn.textContent = '';
      btn.appendChild(el('i', { class: icon, 'aria-hidden': 'true' }));
      btn.appendChild(document.createTextNode(' '));
      btn.appendChild(el('span', { class: 'copy-label', text }));
    }
    function flash(icon, text, ms) {
      clearTimeout(timer);
      setLabel(icon, text);
      timer = setTimeout(() => setLabel(DEFAULT[0], DEFAULT[1]), ms);
    }
    function selectBib() {
      const sel = window.getSelection();
      if (!sel) return null;
      const range = document.createRange();
      range.selectNodeContents(pre);
      sel.removeAllRanges();
      sel.addRange(range);
      return sel;
    }
    function fallback() {
      let ok = false;
      try {
        const sel = selectBib();
        ok = !!sel && document.execCommand('copy');
        if (ok) sel.removeAllRanges();
      } catch (e) { ok = false; }
      if (ok) flash('fas fa-check', 'Copied', 1600);
      else { selectBib(); flash('far fa-keyboard', isApple ? 'Press ⌘C' : 'Press Ctrl+C', 4000); }   // text stays selected
    }
    btn.addEventListener('click', () => {
      const text = pre.textContent;
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        navigator.clipboard.writeText(text).then(() => flash('fas fa-check', 'Copied', 1600), fallback);
      } else {
        fallback();
      }
    });
  });

  /* --------------------------------------------- tab panels without focusables */
  safe('tabpanel focus', () => {
    $$('[role="tabpanel"]').forEach(p => {
      const focusable = p.querySelector('a[href], button, input, select, textarea, [tabindex]:not([tabindex="-1"])');
      if (focusable) p.removeAttribute('tabindex'); else p.tabIndex = 0;
    });
  });
})();
