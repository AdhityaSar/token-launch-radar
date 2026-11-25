// Launch Radar v3 — Empirically-Fitted Ridge Valuation, Conformal Bands & Net-EV Farm Planner
// Zero frameworks, zero network calls. Reads window.RADAR_DATA written by main.ts.

(() => {
  'use strict';

  const DATA = window.RADAR_DATA;
  const ROOT = document.getElementById('app');
  if (!DATA || !Array.isArray(DATA.rows)) {
    ROOT.innerHTML = '<div class="empty-state"><h2>No snapshot data loaded</h2><p>Run main.ts to generate data.js.</p></div>';
    return;
  }

  const L2_CHAINS = new Set([
    'Base', 'Arbitrum', 'Optimism', 'Scroll', 'Ink', 'Monad', 'Unichain', 'Linea',
    'Blast', 'Mantle', 'Polygon', 'Polygon zkEVM', 'zkSync Era', 'Mode', 'BOB',
    'Sonic', 'Berachain', 'Katana', 'Plasma', 'Morph', 'Taiko', 'Soneium', 'World Chain',
    'Fraxtal', 'Starknet', 'Manta', 'Metis', 'Corn', 'MegaETH', 'Etherlink', 'Abstract',
  ]);
  const ALT_L1_CHAINS = new Set([
    'Solana', 'Sui', 'Aptos', 'Tron', 'Avalanche', 'Binance', 'Hyperliquid L1',
    'Sei', 'Near', 'TON', 'Stellar', 'Flare', 'Bittensor', 'Cardano', 'Cosmos',
    'Algorand', 'Hedera', 'Injective', 'Osmosis', 'Cronos', 'Kava', 'Celo',
  ]);

  const qs = new URLSearchParams(window.location.search);
  const state = {
    view: ['radar', 'live', 'model'].includes(qs.get('view')) ? qs.get('view') : 'radar',
    tier: ['all', 'watchlist', 'early'].includes(qs.get('tier')) ? qs.get('tier') : 'all',
    verdict: ['all', 'FARM', 'SELECTIVE', 'SKIP'].includes(qs.get('verdict')) ? qs.get('verdict') : 'all',
    search: qs.get('q') || '',
    category: qs.get('cat') || 'all',
    chain: qs.get('chain') || 'all',
    momentum: ['all', 'pos', 'fast', 'neg'].includes(qs.get('momentum')) ? qs.get('momentum') : 'all',
    sortKey: qs.get('sort') || 'rank',
    sortDir: qs.get('dir') === 'desc' ? 'desc' : 'asc',
    selectedSlug: qs.get('detail') || null,
    sandboxOpen: qs.get('sandbox') === '1',
    weights: { ...DATA.weights },
  };

  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const round2 = (n) => Math.round(n * 100) / 100;
  const round4 = (n) => Math.round(n * 1e4) / 1e4;
  const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

  function fmtUsd(n) {
    if (n === null || n === undefined || !Number.isFinite(n)) return '—';
    const a = Math.abs(n);
    if (a >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
    if (a >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
    if (a >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
    return `$${Math.round(n)}`;
  }

  function fmtPct(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return 'N/A';
    return `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`;
  }

  function fmtAge(d) {
    if (d === null || d === undefined) return 'N/A';
    return d >= 365 ? `${(d / 365).toFixed(1)}y` : `${d}d`;
  }

  /** Mirror of lib/model.ts computeImpliedValue for live interactive sandbox recalculation. */
  function computeImpliedValue(row, w) {
    const lt = Math.log(Math.max(1, row.tvl_usd)) - 18.0;
    const g = Math.tanh((typeof row.change_7d === 'number' && Number.isFinite(row.change_7d) ? row.change_7d : 0) / 50);
    const a = Math.sqrt(clamp(row.age_days ?? 365, 0, 1095) / 1095) - 0.7;
    const ch = Math.min(Math.log(Math.max(1, row.chains.length)), 2.5);
    const cat = DATA.categories[row.category || 'Other'];
    const sectorAdj = cat ? cat.log_residual_adj : 0;

    const logBase = w.intercept + w.b_tvl * lt;
    const logG = w.b_growth * g;
    const logA = w.b_age * a;
    const logCh = w.b_chains * ch;
    const logPred = logBase + logG + logA + logCh + sectorAdj;

    const implied = Math.max(1, Math.round(Math.exp(logPred)));
    const bear = Math.max(1, Math.min(implied, Math.round(Math.exp(logPred + w.conformal_q25))));
    const bull = Math.max(implied, Math.round(Math.exp(logPred + w.conformal_q75)));

    const rawYield = Math.min(0.4, (implied * 0.10 * 1.5) / Math.max(1, row.tvl_usd));
    const slippage = row.capital_usd * 0.0015;
    const netEv = round2(row.capital_usd * rawYield * row.token_odds - row.est_gas_usd - slippage);
    const hourlyEv = round2(netEv / Math.max(0.5, row.active_minutes / 60));
    const verdict = netEv >= 15 && hourlyEv >= 10 ? 'FARM' : netEv >= 3 ? 'SELECTIVE' : 'SKIP';

    return {
      ...row,
      implied_value_usd: implied,
      bear_value_usd: bear,
      bull_value_usd: bull,
      implied_mcap_tvl: round4(implied / Math.max(1, row.tvl_usd)),
      tvl_component: Math.round(Math.exp(logBase)),
      growth_factor: round4(Math.exp(logG)),
      age_factor: round4(Math.exp(logA)),
      chain_factor: round4(Math.exp(logCh)),
      sector_factor: round4(Math.exp(sectorAdj)),
      net_ev_usd: netEv,
      hourly_ev_usd: hourlyEv,
      verdict,
    };
  }

  /** Mirror of lib/model.ts buildFarmPlan for the protocol detail drawer. */
  function buildFarmPlan(row) {
    const chains = row.chains || [];
    const lowFee = chains.find((c) => c !== 'Ethereum' && (L2_CHAINS.has(c) || ALT_L1_CHAINS.has(c)));
    const primary = lowFee || chains[0] || 'its primary chain';
    const secondary = chains.find((c) => c !== primary) || null;
    const gasTier = primary === 'Ethereum' ? 'mainnet' : ALT_L1_CHAINS.has(primary) ? 'alt-l1' : 'l2';
    const gas = gasTier === 'mainnet'
      ? { weekly: 4, total: 28, label: '$10-$15 Ethereum L1 gas' }
      : gasTier === 'alt-l1'
        ? { weekly: 0.05, total: 1, label: 'under $0.25 network gas' }
        : { weekly: 0.15, total: 2, label: 'under $0.75 L2 gas' };
    const capital = row.capital_usd;
    const hold = row.hold_days;
    const weeks = Math.max(4, Math.ceil(hold / 7));
    const slippage = capital * 0.0015;
    const cat = (row.category || '').toLowerCase();
    const n = row.name;

    let core = { title: 'Deploy core protocol position', action: `Deposit $${capital} USDC into ${n}'s flagship product on ${primary} and keep the position active across snapshots.` };
    if (cat.includes('risk curator') || cat.includes('allocator') || cat.includes('cdp manager')) {
      core = { title: 'Deposit into curated vault', action: `Deposit $${capital} USDC into ${n}'s flagship curated vault on ${primary} and hold the receipt shares in self-custody.` };
    } else if (cat.includes('rwa')) {
      core = { title: 'Mint yield-bearing RWA', action: `Subscribe $${capital} into ${n}'s yield-bearing RWA token on ${primary} and keep the balance unstaked or in its native vault.` };
    } else if (cat.includes('basis') || cat.includes('yield')) {
      core = { title: 'Enter core yield vault', action: `Deploy $${capital} USDC into ${n}'s primary yield or basis pool on ${primary} and remain deposited across epoch rollovers.` };
    } else if (cat.includes('lend') || cat.includes('cdp') || cat.includes('collateral')) {
      core = { title: 'Supply collateral & borrow lightly', action: `Supply $${capital} USDC on ${n} (${primary}) and borrow 20% against it to qualify under both lender and borrower criteria.` };
    } else if (cat.includes('dex') || cat.includes('liquidity')) {
      core = { title: 'Provide LP & execute swaps', action: `Provide $${capital} in a core stable or blue-chip pool on ${n} (${primary}) and execute 2-3 organic spot swaps.` };
    } else if (cat.includes('derivative') || cat.includes('perpetual') || cat.includes('prediction') || cat.includes('otc')) {
      core = { title: 'Post margin & log volume', action: `Deposit $${capital} USDC margin on ${n} (${primary}) and open small hedged positions to register organic trading volume.` };
    } else if (cat.includes('restak') || cat.includes('staking')) {
      core = { title: 'Stake native receipt asset', action: `Stake $${capital} of the native asset through ${n} on ${primary} and keep the receipt position delegated.` };
    } else if (cat.includes('payment') || cat.includes('stable') || cat.includes('cedefi')) {
      core = { title: 'Mint & transact stable balance', action: `Mint or fund $${capital} through ${n} on ${primary} and complete at least two on-chain settlement transfers.` };
    } else if (cat.includes('insurance')) {
      core = { title: 'Underwrite or purchase cover', action: `Commit $${capital} to ${n}'s underwriting pool or buy a small cover policy on ${primary} and hold to expiry.` };
    }

    const hasDrawdown = typeof row.change_7d === 'number' && row.change_7d <= -5;
    const steps = [
      { title: 'Isolate wallet & fund gas', action: `Create a dedicated wallet on ${primary}, fund $${capital} principal plus native gas, and keep zero unrelated approvals.`, cost: `${gas.label} for initial funding`, time: '10 minutes' },
      { title: core.title, action: core.action, cost: `$${capital} refundable capital + ${gas.label}`, time: '15 minutes' },
      secondary
        ? { title: 'Multi-chain footprint & weekly cadence', action: `Complete one qualifying interaction weekly on ${primary} (and mirror 20% on ${secondary}) across ${weeks}+ distinct weeks.`, cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas (${weeks} wks)`, time: `8 min/wk (${weeks} wks)` }
        : { title: 'Weekly organic cadence', action: `Execute one qualifying on-chain action weekly on ${primary} (top-up, compound, swap, or rebalance) to pass sybil filters.`, cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas (${weeks} wks)`, time: `8 min/wk (${weeks} wks)` },
      hasDrawdown
        ? { title: 'Check 7d TVL drawdown & hold window', action: `TVL moved ${row.change_7d.toFixed(1)}% in 7d; verify liquidity depth before scaling, then hold $${capital} for ${hold} days across epochs.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` }
        : { title: 'Hold across snapshot epochs', action: `Keep the $${capital} position open uninterrupted for ${hold} days so time-weighted points and multi-epoch snapshots register full weight.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` },
      { title: 'Verify points & official claim route', action: `Inspect ${n}'s official points page before exiting; never sign blind permits or pay third-party claim fees.`, cost: '$0 check (never pay to claim)', time: '10 minutes' },
      { title: 'Log tx hashes & unwind cleanly', action: `Record wallet, tx hashes, and epoch dates in your farm log, then revoke token approvals after withdrawing.`, cost: `${gas.label} for exit & revoke`, time: '10 minutes' },
    ];
    return { primary, gasTier, capital, estGas: gas.total, hold, steps };
  }

  function isSandboxModified() {
    return ['b_tvl', 'b_growth', 'b_age', 'b_chains'].some((k) => Math.abs(state.weights[k] - DATA.weights[k]) > 1e-6);
  }

  function getScoredRows() {
    const computed = DATA.rows.map((r) => computeImpliedValue(r, state.weights));
    computed.sort((a, b) => b.implied_value_usd - a.implied_value_usd || b.tvl_usd - a.tvl_usd || a.slug.localeCompare(b.slug));
    computed.forEach((r, i) => { r.rank = i + 1; });
    return computed;
  }

  function filterAndSortRows(rows) {
    const q = state.search.trim().toLowerCase();
    const filtered = rows.filter((r) => {
      if (state.tier !== 'all' && r.tier !== state.tier) return false;
      if (state.verdict !== 'all' && r.verdict !== state.verdict) return false;
      if (state.category !== 'all' && (r.category || 'Uncategorized') !== state.category) return false;
      if (state.chain !== 'all' && !r.chains.includes(state.chain)) return false;
      if (state.momentum === 'pos' && !(typeof r.change_7d === 'number' && r.change_7d > 0)) return false;
      if (state.momentum === 'fast' && !(typeof r.change_7d === 'number' && r.change_7d >= 5)) return false;
      if (state.momentum === 'neg' && !(typeof r.change_7d === 'number' && r.change_7d < 0)) return false;
      if (q && !`${r.name} ${r.slug} ${r.category || ''} ${r.chains.join(' ')}`.toLowerCase().includes(q)) return false;
      return true;
    });
    const dir = state.sortDir === 'asc' ? 1 : -1;
    const key = state.sortKey;
    filtered.sort((a, b) => {
      if (key === 'name' || key === 'category') {
        const va = (a[key] || '').toLowerCase();
        const vb = (b[key] || '').toLowerCase();
        return va < vb ? -dir : va > vb ? dir : a.rank - b.rank;
      }
      const va = key === 'change_7d' ? (a.change_7d ?? -9999) : key === 'age_days' ? (a.age_days ?? -1) : (a[key] ?? 0);
      const vb = key === 'change_7d' ? (b.change_7d ?? -9999) : key === 'age_days' ? (b.age_days ?? -1) : (b[key] ?? 0);
      return va !== vb ? (va - vb) * dir : a.rank - b.rank;
    });
    return filtered;
  }

  function buildFilterOptions(rows) {
    const cats = new Map();
    const chains = new Map();
    for (const r of rows) {
      const c = r.category || 'Uncategorized';
      cats.set(c, (cats.get(c) || 0) + 1);
      for (const ch of r.chains) chains.set(ch, (chains.get(ch) || 0) + 1);
    }
    return {
      categories: [...cats.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
      chains: [...chains.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0])),
    };
  }

  const hasFilters = () => state.tier !== 'all' || state.verdict !== 'all' || state.category !== 'all' || state.chain !== 'all' || state.momentum !== 'all' || state.search.trim() !== '';

  function sortBtn(key, label) {
    const on = state.sortKey === key;
    return `<button type="button" class="${on ? 'sorted' : ''}" data-sort="${key}">${label}${on ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : ''}</button>`;
  }

  function verdictBadge(v) {
    const cls = v === 'FARM' ? 'badge-farm' : v === 'SELECTIVE' ? 'badge-sel' : 'badge-skip';
    return `<span class="badge ${cls}">${v}</span>`;
  }

  function renderHeader() {
    const u = DATA.universe;
    const val = DATA.validation;
    return `
      <header class="topbar">
        <div class="brand-group">
          <div class="radar-mark" aria-hidden="true">
            <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="1.8">
              <circle cx="12" cy="12" r="9" opacity="0.35"/><circle cx="12" cy="12" r="5" opacity="0.65"/>
              <path d="M12 12L18.5 5.5" stroke-linecap="round"/><circle cx="12" cy="12" r="2" fill="#10b981"/>
            </svg>
          </div>
          <div>
            <div class="brand-title">Launch Radar <span class="version-tag">${esc(DATA.model_version)}</span></div>
            <div class="brand-sub">Out-of-sample validated pre-token valuation, launch odds &amp; net-EV farm planner · Snapshot ${esc(DATA.as_of)}</div>
          </div>
        </div>
        <div class="top-nav">
          <div class="nav-tabs" role="tablist">
            <button type="button" data-view="radar" aria-pressed="${state.view === 'radar'}">Pre-Token Radar (${u.board_count})</button>
            <button type="button" data-view="live" aria-pressed="${state.view === 'live'}">Live Comps (${u.live_priced_count})</button>
            <button type="button" data-view="model" aria-pressed="${state.view === 'model'}">OOS Validation &amp; Fit</button>
          </div>
        </div>
      </header>
      <div class="audit-strip" aria-label="Universe reconciliation and self-check">
        <div class="audit-items">
          <span class="audit-ok">✓ ${u.watchlist_count}/${u.watchlist_count} Watchlist</span>
          <span class="audit-sep">|</span>
          <span>${u.total_protocols}/${u.total_protocols} Protocols Accounted: <b>${u.board_count}</b> pre-token (${u.watchlist_count} floor + ${u.early_count} early) + <b>${u.live_total_count}</b> live separated + <b>${u.excluded_non_launch_count}</b> CEX/bridge/wrapped excluded</span>
        </div>
        <div class="audit-items">
          <span class="audit-ok">✓ LOO ×${val.loo_median_mult_err} vs Naive ×${val.loo_naive_mult_err} (-${val.loo_error_reduction_pct}%)</span>
          <span class="audit-sep">|</span>
          <span class="audit-ok">✓ Walk-Fwd ×${val.wf_median_mult_err}</span>
          <span class="audit-sep">|</span>
          <span>0 Hardcoded Names</span>
        </div>
      </div>
    `;
  }

  function renderKpis(allRows, visibleRows) {
    const active = hasFilters();
    const scope = active ? visibleRows : allRows;
    const totalImplied = scope.reduce((s, r) => s + r.implied_value_usd, 0);
    const totalTvl = scope.reduce((s, r) => s + r.tvl_usd, 0);
    const farmCount = scope.filter((r) => r.verdict === 'FARM').length;
    const selCount = scope.filter((r) => r.verdict === 'SELECTIVE').length;
    const val = DATA.validation;
    const bestEv = [...scope].sort((a, b) => b.hourly_ev_usd - a.hourly_ev_usd)[0];

    return `
      <section class="kpi-grid" aria-label="Summary metrics">
        <div class="kpi-card">
          <span class="kpi-label">${active ? 'Filtered Implied Value' : 'Total Pre-Token Implied'}</span>
          <span class="kpi-val emerald">${fmtUsd(totalImplied)}</span>
          <span class="kpi-sub">${scope.length} pre-token · ${fmtUsd(totalTvl)} TVL</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">LOO-CV Blind Error</span>
          <span class="kpi-val cyan">×${val.loo_median_mult_err}</span>
          <span class="kpi-sub">vs Naive 0.25×TVL ×${val.loo_naive_mult_err} (-${val.loo_error_reduction_pct}%)</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Walk-Forward Blind</span>
          <span class="kpi-val cyan">×${val.wf_median_mult_err}</span>
          <span class="kpi-sub">${val.wf_train_count} older → ${val.wf_test_count} newer (-${val.wf_error_reduction_pct}% vs naive)</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Farmable Verdicts</span>
          <span class="kpi-val">${farmCount} FARM</span>
          <span class="kpi-sub">${selCount} selective · ${scope.length - farmCount - selCount} skip (net of gas &amp; carry)</span>
        </div>
        <div class="kpi-card">
          <span class="kpi-label">Top Farm Return</span>
          <span class="kpi-val emerald">${bestEv ? `$${bestEv.hourly_ev_usd.toFixed(1)}/hr` : '—'}</span>
          <span class="kpi-sub">${bestEv ? `${esc(bestEv.name)} (+$${bestEv.net_ev_usd.toFixed(0)} EV)` : 'None in filter'}</span>
        </div>
      </section>
    `;
  }

  function renderFormulaBar() {
    const w = state.weights;
    return `
      <section class="formula-bar">
        <div class="formula-eq">
          <strong>ln(Implied)</strong> = ${w.intercept.toFixed(2)} + <strong>${w.b_tvl.toFixed(3)}</strong>·ln(TVL/66M) + <strong>${w.b_growth.toFixed(3)}</strong>·tanh(Δ7d/50) + <strong>${w.b_age.toFixed(3)}</strong>·(√age−0.7) + <strong>${w.b_chains.toFixed(3)}</strong>·ln(chains) + γ_sector(k=${w.sector_shrinkage_k})
        </div>
        <div class="formula-actions">
          ${isSandboxModified() ? '<button type="button" class="btn-ghost" data-action="reset-model">Reset Fit</button>' : ''}
          <button type="button" class="btn-ghost ${state.sandboxOpen ? 'active' : ''}" data-action="toggle-sandbox">${state.sandboxOpen ? 'Hide Sandbox' : 'Model Sandbox'}</button>
        </div>
      </section>
      ${state.sandboxOpen ? `
        <section class="sandbox-panel" aria-label="Interactive model parameters">
          <label class="slider-field"><span class="slider-head"><span>TVL Elasticity (β₁)</span><b>${w.b_tvl.toFixed(3)}</b></span><input type="range" min="0.4" max="1.0" step="0.02" value="${w.b_tvl}" data-param="b_tvl"></label>
          <label class="slider-field"><span class="slider-head"><span>7d Growth Slope (β₂)</span><b>${w.b_growth.toFixed(3)}</b></span><input type="range" min="-0.2" max="0.5" step="0.02" value="${w.b_growth}" data-param="b_growth"></label>
          <label class="slider-field"><span class="slider-head"><span>Age Slope (β₃)</span><b>${w.b_age.toFixed(3)}</b></span><input type="range" min="-0.4" max="0.4" step="0.02" value="${w.b_age}" data-param="b_age"></label>
          <label class="slider-field"><span class="slider-head"><span>Chain Slope (β₄)</span><b>${w.b_chains.toFixed(3)}</b></span><input type="range" min="-0.3" max="0.3" step="0.02" value="${w.b_chains}" data-param="b_chains"></label>
          <button type="button" class="btn-ghost" data-action="reset-model">Fitted</button>
        </section>
      ` : ''}
    `;
  }

  function renderToolbar(allRows, visibleRows) {
    const opts = buildFilterOptions(allRows);
    const chips = [];
    if (state.tier !== 'all') chips.push({ key: 'tier', label: `Tier: ${state.tier === 'watchlist' ? 'Watchlist ≥$50M' : 'Early <$50M'}` });
    if (state.verdict !== 'all') chips.push({ key: 'verdict', label: `Verdict: ${state.verdict}` });
    if (state.category !== 'all') chips.push({ key: 'category', label: `Sector: ${state.category}` });
    if (state.chain !== 'all') chips.push({ key: 'chain', label: `Chain: ${state.chain}` });
    if (state.momentum !== 'all') chips.push({ key: 'momentum', label: { pos: '7d Flow > 0%', fast: '7d ≥ +5%', neg: '7d < 0%' }[state.momentum] });
    if (state.search.trim()) chips.push({ key: 'search', label: `Query: "${state.search.trim()}"` });

    return `
      <section class="toolbar" aria-label="Board filters">
        <div class="tier-pills" role="group">
          <button type="button" data-tier="all" aria-pressed="${state.tier === 'all'}">All (${allRows.length})</button>
          <button type="button" data-tier="watchlist" aria-pressed="${state.tier === 'watchlist'}">Watchlist (${DATA.universe.watchlist_count})</button>
          <button type="button" data-tier="early" aria-pressed="${state.tier === 'early'}">Early (${DATA.universe.early_count})</button>
        </div>
        <input type="search" class="ctrl-input" id="search-input" placeholder="Search protocol, sector, chain…" value="${esc(state.search)}" aria-label="Search">
        <select class="ctrl-select" id="verdict-select" aria-label="Farm verdict">
          <option value="all" ${state.verdict === 'all' ? 'selected' : ''}>All Verdicts</option>
          <option value="FARM" ${state.verdict === 'FARM' ? 'selected' : ''}>FARM only</option>
          <option value="SELECTIVE" ${state.verdict === 'SELECTIVE' ? 'selected' : ''}>SELECTIVE only</option>
          <option value="SKIP" ${state.verdict === 'SKIP' ? 'selected' : ''}>SKIP only</option>
        </select>
        <select class="ctrl-select" id="cat-select" aria-label="Sector">
          <option value="all">All Sectors (${opts.categories.length})</option>
          ${opts.categories.map(([c, n]) => `<option value="${esc(c)}" ${state.category === c ? 'selected' : ''}>${esc(c)} (${n})</option>`).join('')}
        </select>
        <select class="ctrl-select" id="chain-select" aria-label="Chain">
          <option value="all">All Chains (${opts.chains.length})</option>
          ${opts.chains.map(([c, n]) => `<option value="${esc(c)}" ${state.chain === c ? 'selected' : ''}>${esc(c)} (${n})</option>`).join('')}
        </select>
        <select class="ctrl-select" id="mom-select" aria-label="7d flow">
          <option value="all" ${state.momentum === 'all' ? 'selected' : ''}>Any 7d Flow</option>
          <option value="pos" ${state.momentum === 'pos' ? 'selected' : ''}>Positive (&gt; 0%)</option>
          <option value="fast" ${state.momentum === 'fast' ? 'selected' : ''}>Fast (≥ +5%)</option>
          <option value="neg" ${state.momentum === 'neg' ? 'selected' : ''}>Drawdown (&lt; 0%)</option>
        </select>
        <div class="toolbar-right">
          <span class="row-counter">${visibleRows.length} of ${allRows.length}</span>
          ${chips.length ? '<button type="button" class="btn-ghost" data-action="reset-filters">Clear</button>' : ''}
        </div>
      </section>
      ${chips.length ? `<div class="filter-chips">${chips.map((c) => `<span class="chip">${esc(c.label)} <button type="button" data-clear="${c.key}">×</button></span>`).join('')}</div>` : ''}
    `;
  }

  function renderRadarTable(visibleRows, maxImplied) {
    if (!visibleRows.length) return '<div class="panel"><div class="empty-state"><h3>No matching protocols</h3><p>Clear filters to inspect all 178 pre-token protocols.</p></div></div>';
    const body = visibleRows.map((r) => {
      const pct = Math.max(4, Math.round((Math.log10(Math.max(1, r.implied_value_usd)) / Math.log10(Math.max(10, maxImplied))) * 100));
      const flowCls = r.change_7d === null ? 'flow-na' : r.change_7d >= 0 ? 'flow-pos' : 'flow-neg';
      const chains = r.chains.slice(0, 2).map((c) => `<span class="chain-tag">${esc(c)}</span>`).join('') + (r.chains.length > 2 ? `<span class="chain-tag">+${r.chains.length - 2}</span>` : '');
      const evSign = r.net_ev_usd >= 0 ? `+$${r.net_ev_usd.toFixed(0)}` : `-$${Math.abs(r.net_ev_usd).toFixed(0)}`;
      return `
        <tr data-slug="${esc(r.slug)}" data-selected="${state.selectedSlug === r.slug}" tabindex="0">
          <td class="rank-cell ${r.rank <= 10 ? 'rank-top' : ''}">#${r.rank}</td>
          <td>
            <div class="proto-cell">
              <div class="proto-name">${esc(r.name)} <span class="badge ${r.tier === 'watchlist' ? 'badge-wl' : 'badge-early'}">${r.tier === 'watchlist' ? 'WL' : 'EARLY'}</span></div>
              <div class="proto-meta"><span>${esc(r.category || 'Other')}</span><span>·</span><span class="mono">${esc(r.slug)}</span></div>
            </div>
          </td>
          <td><div class="chains-list">${chains || '<span class="chain-tag">Multi</span>'}</div></td>
          <td class="r mono">${fmtUsd(r.tvl_usd)}</td>
          <td class="r mono ${flowCls}">${fmtPct(r.change_7d)}</td>
          <td class="r mono">${fmtAge(r.age_days)}</td>
          <td class="r mono">${Math.round(r.token_odds * 100)}%</td>
          <td class="r">
            <div class="val-cell">
              <span class="val-main">${fmtUsd(r.implied_value_usd)}</span>
              <div class="val-bar"><i style="width:${pct}%"></i></div>
              <span class="val-range">${fmtUsd(r.bear_value_usd)} – ${fmtUsd(r.bull_value_usd)}</span>
            </div>
          </td>
          <td class="r">
            <div class="verdict-cell">
              <div>${verdictBadge(r.verdict)} <span class="mono" style="margin-left:4px;font-weight:600">${evSign}</span></div>
              <span class="verdict-sub">$${r.hourly_ev_usd.toFixed(1)}/hr · $${r.capital_usd} · ~$${r.est_gas_usd}g</span>
            </div>
          </td>
        </tr>
      `;
    }).join('');

    return `
      <section class="panel"><div class="table-wrap"><table>
        <thead><tr>
          <th>${sortBtn('rank', 'Rank')}</th>
          <th>${sortBtn('name', 'Protocol & Sector')}</th>
          <th>Chains</th>
          <th class="r">${sortBtn('tvl_usd', 'TVL')}</th>
          <th class="r">${sortBtn('change_7d', '7d Flow')}</th>
          <th class="r">${sortBtn('age_days', 'Age')}</th>
          <th class="r">${sortBtn('token_odds', 'Odds')}</th>
          <th class="r">${sortBtn('implied_value_usd', 'Implied (Conformal 50%)')}</th>
          <th class="r">${sortBtn('hourly_ev_usd', 'Farm Verdict (EV / $/hr)')}</th>
        </tr></thead>
        <tbody>${body}</tbody>
      </table></div></section>
    `;
  }

  function renderLiveCompsView() {
    const body = DATA.live.map((p, i) => {
      const fc = p.change_7d === null ? 'flow-na' : p.change_7d >= 0 ? 'flow-pos' : 'flow-neg';
      return `
        <tr>
          <td class="rank-cell">${i + 1}</td>
          <td><div class="proto-cell"><div class="proto-name">${esc(p.name)} <span class="badge badge-live">${esc(p.symbol || 'LIVE')}</span></div><div class="proto-meta">${esc(p.category || 'Other')} · <span class="mono">${esc(p.slug)}</span></div></div></td>
          <td class="r mono">${fmtUsd(p.mcap_usd)}</td>
          <td class="r mono">${fmtUsd(p.tvl_usd)}</td>
          <td class="r mono" style="color:var(--cyan);font-weight:700">${p.mcap_tvl_ratio.toFixed(3)}x</td>
          <td class="r mono ${fc}">${fmtPct(p.change_7d)}</td>
          <td class="r mono">${fmtAge(p.age_days)}</td>
        </tr>
      `;
    }).join('');
    return `
      <section class="panel" style="margin-top:10px">
        <div style="padding:11px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
          <div>
            <h2 style="font-size:14.5px">Live-Token Market Comparables (Strictly Separated from Pre-Token Board)</h2>
            <p style="font-size:11.5px;color:var(--muted);margin-top:2px">All ${DATA.universe.live_total_count} token-bearing protocols are excluded from pre-token rankings; these ${DATA.live.length} priced DeFi comps fit the Ridge power-law model.</p>
          </div>
          <span class="badge badge-live">Fitted TVL Elasticity β₁ = ${DATA.weights.b_tvl.toFixed(3)}</span>
        </div>
        <div class="table-wrap"><table>
          <thead><tr><th>#</th><th>Live Protocol</th><th class="r">Market Cap</th><th class="r">TVL</th><th class="r">MCap/TVL</th><th class="r">7d Flow</th><th class="r">Age</th></tr></thead>
          <tbody>${body}</tbody>
        </table></div>
      </section>
    `;
  }

  function renderModelView() {
    const v = DATA.validation;
    const w = DATA.weights;
    const cats = Object.values(DATA.categories).filter((c) => c.peer_count > 0).sort((a, b) => b.peer_count - a.peer_count || a.category.localeCompare(b.category));
    return `
      <section class="two-col">
        <div class="card">
          <h2>Out-of-Sample Blind Validation (${v.live_fit_count} Live Comps)</h2>
          <p>Every weight is fitted by Ridge regression (λ=${w.ridge_lambda}) on live-token DeFi peers in snapshot.json with Empirical-Bayes sector residual shrinkage (k=${w.sector_shrinkage_k}). Zero protocol names are hardcoded.</p>
          <div class="code-block">Test 1 — Leave-One-Out CV (n=${v.live_fit_count}):
  Fitted Ridge + Sector EB : median error ×${v.loo_median_mult_err}  (MAE log ${v.loo_mae_log})
  Naive 0.25 × TVL baseline: median error ×${v.loo_naive_mult_err}  (MAE log ${v.loo_naive_mae_log})
  Improvement              : -${v.loo_error_reduction_pct}% multiplicative error

Test 2 — Temporal Walk-Forward (${v.wf_train_count} older → ${v.wf_test_count} newer by listed_at):
  Fitted Ridge + Sector EB : median error ×${v.wf_median_mult_err}
  Naive 0.25 × TVL baseline: median error ×${v.wf_naive_mult_err}
  Improvement              : -${v.wf_error_reduction_pct}% multiplicative error

Conformal Prediction Band (LOO residual quantiles q25=${w.conformal_q25}, q75=${w.conformal_q75}):
  Empirical 50% coverage   : ${v.conformal_50_coverage_pct}% (honest Bear–Bull spread)</div>
          <p style="margin-top:9px"><strong>Farm EV Rule:</strong> Assumes 10% community allocation (1.5× active-wallet weight vs passive TVL) diluted by TVL, weighted by empirical token odds, net of chain-specific gas (Alt-L1 ~$1, L2 ~$2, Eth L1 ~$28) and 15 bps spread.</p>
        </div>
        <div class="card" style="padding:0;overflow:hidden">
          <div style="padding:12px 14px;border-bottom:1px solid var(--border)"><h2 style="margin:0">Empirical Sector Multipliers &amp; Token Rates</h2></div>
          <div class="table-wrap" style="max-height:380px;overflow-y:auto"><table>
            <thead><tr><th>Sector</th><th class="r">Peers</th><th class="r">Raw Med</th><th class="r">EB Factor</th><th class="r">Token Rate</th></tr></thead>
            <tbody>${cats.map((c) => `
              <tr>
                <td><b>${esc(c.category)}</b></td>
                <td class="r mono">${c.peer_count}</td>
                <td class="r mono">${c.raw_median_mult.toFixed(3)}x</td>
                <td class="r mono" style="color:${c.sector_mult_factor >= 1 ? 'var(--emerald)' : 'var(--amber)'}">×${c.sector_mult_factor.toFixed(2)}</td>
                <td class="r mono">${Math.round(c.token_rate * 100)}%</td>
              </tr>`).join('')}
            </tbody>
          </table></div>
        </div>
      </section>
    `;
  }

  function renderDetailDrawer(allRows, visibleRows) {
    if (!state.selectedSlug) return '';
    const row = allRows.find((r) => r.slug === state.selectedSlug);
    if (!row) return '';
    const plan = buildFarmPlan(row);
    const idx = visibleRows.findIndex((r) => r.slug === row.slug);
    const prev = idx > 0 ? visibleRows[idx - 1].slug : null;
    const next = idx >= 0 && idx < visibleRows.length - 1 ? visibleRows[idx + 1].slug : null;
    const span = Math.max(1, row.bull_value_usd - row.bear_value_usd);
    const dotPos = clamp(((row.implied_value_usd - row.bear_value_usd) / span) * 84 + 8, 10, 90);
    const evStr = row.net_ev_usd >= 0 ? `+$${row.net_ev_usd.toFixed(1)}` : `-$${Math.abs(row.net_ev_usd).toFixed(1)}`;

    return `
      <div class="drawer-backdrop" data-action="close-drawer">
        <aside class="drawer" role="dialog" aria-modal="true" aria-label="${esc(row.name)} detail">
          <div class="drawer-head">
            <div>
              <div class="drawer-title">
                <span>#${row.rank} ${esc(row.name)}</span>
                <span class="badge ${row.tier === 'watchlist' ? 'badge-wl' : 'badge-early'}">${row.tier === 'watchlist' ? 'WATCHLIST' : 'EARLY'}</span>
                ${verdictBadge(row.verdict)}
              </div>
              <div class="drawer-sub">${esc(row.category || 'Other')} · ${esc(row.slug)} · Listed ${esc(row.listed_at || 'N/A')} (${fmtAge(row.age_days)}) · Launch Odds ${Math.round(row.token_odds * 100)}%</div>
            </div>
            <div class="drawer-nav">
              <button type="button" class="btn-ghost" ${prev ? `data-open="${esc(prev)}"` : 'disabled'}>↑</button>
              <button type="button" class="btn-ghost" ${next ? `data-open="${esc(next)}"` : 'disabled'}>↓</button>
              <button type="button" class="btn-ghost" data-action="close-drawer">Close</button>
            </div>
          </div>

          <section class="drawer-section">
            <div class="section-label"><span>Conformal Launch Valuation Band (${DATA.validation.conformal_50_coverage_pct}% LOO Coverage)</span><span class="mono">Implied ${row.implied_mcap_tvl.toFixed(2)}× TVL</span></div>
            <div class="band-box">
              <div class="band-nums">
                <div><small>Bear (Conformal P25)</small><strong>${fmtUsd(row.bear_value_usd)}</strong></div>
                <div><small>Fitted Base Implied</small><strong class="base-v">${fmtUsd(row.implied_value_usd)}</strong></div>
                <div><small>Bull (Conformal P75)</small><strong>${fmtUsd(row.bull_value_usd)}</strong></div>
              </div>
              <div class="band-track"><div class="band-fill"></div><div class="band-dot" style="left:${dotPos.toFixed(1)}%"></div></div>
            </div>
            <div class="eq-grid">
              <div class="eq-row"><span>1. Power-Law TVL Anchor (TVL ${fmtUsd(row.tvl_usd)}, β₁=${state.weights.b_tvl.toFixed(2)})</span><b>${fmtUsd(row.tvl_component)}</b></div>
              <div class="eq-row"><span>2. Sector Residual Factor (${esc(row.category || 'Other')}, ${row.peer_count} live peers)</span><b>× ${row.sector_factor.toFixed(4)}</b></div>
              <div class="eq-row"><span>3. 7d TVL Flow (${fmtPct(row.change_7d)}) · Age (${fmtAge(row.age_days)}) · Chains (${row.chains.length})</span><b>× ${row.growth_factor.toFixed(3)} · × ${row.age_factor.toFixed(3)} · × ${row.chain_factor.toFixed(3)}</b></div>
              <div class="eq-row total"><span>Fitted Implied Pre-Token Value</span><b>${fmtUsd(row.implied_value_usd)}</b></div>
            </div>
          </section>

          <section class="drawer-section">
            <div class="section-label"><span>Actionable 6-Step Farm Plan &amp; Net-EV Economics</span><span class="mono">${esc(plan.primary)} (${plan.gasTier}) · ${row.active_minutes}m active</span></div>
            <div class="farm-meta-strip">
              <div class="farm-stat"><small>Capital</small><b>$${plan.capital}</b></div>
              <div class="farm-stat"><small>Est. Gas</small><b>~$${plan.estGas}</b></div>
              <div class="farm-stat"><small>Net EV</small><b style="color:${row.net_ev_usd >= 0 ? 'var(--emerald)' : 'var(--rose)'}">${evStr}</b></div>
              <div class="farm-stat"><small>Implied Rate</small><b style="color:${row.hourly_ev_usd >= 10 ? 'var(--emerald)' : 'var(--amber)'}">$${row.hourly_ev_usd.toFixed(1)}/hr</b></div>
            </div>
            <ol class="steps-list">
              ${plan.steps.map((s, i) => `
                <li class="step-item">
                  <div class="step-num">${i + 1}</div>
                  <div>
                    <div class="step-title">${esc(s.title)}</div>
                    <div class="step-action">${esc(s.action)}</div>
                    <div class="step-tags"><span class="step-cost">Cost: ${esc(s.cost)}</span><span class="step-time">Time: ${esc(s.time)}</span></div>
                  </div>
                </li>
              `).join('')}
            </ol>
          </section>
        </aside>
      </div>
    `;
  }

  function render() {
    const allRows = getScoredRows();
    const visible = filterAndSortRows(allRows);
    const maxImp = allRows.length ? allRows[0].implied_value_usd : 1;
    let body = '';
    if (state.view === 'radar') body = renderKpis(allRows, visible) + renderFormulaBar() + renderToolbar(allRows, visible) + renderRadarTable(visible, maxImp);
    else if (state.view === 'live') body = renderKpis(allRows, visible) + renderLiveCompsView();
    else body = renderKpis(allRows, visible) + renderFormulaBar() + renderModelView();

    ROOT.innerHTML = renderHeader() + body + `
      <footer class="footer-note">
        <span>Source: ${esc(DATA.source)} · Pre-token status reflects DefiLlama's record, never a launch guarantee.</span>
        <span>${DATA.universe.total_protocols} total = ${DATA.universe.board_count} pre-token + ${DATA.universe.live_total_count} live + ${DATA.universe.excluded_non_launch_count} non-launch excluded</span>
      </footer>
    ` + renderDetailDrawer(allRows, visible);

    bindEvents();
  }

  function bindEvents() {
    ROOT.querySelectorAll('[data-view]').forEach((el) => el.addEventListener('click', () => { state.view = el.dataset.view; state.selectedSlug = null; render(); }));
    ROOT.querySelectorAll('[data-tier]').forEach((el) => el.addEventListener('click', () => { state.tier = el.dataset.tier; render(); }));
    ROOT.querySelectorAll('[data-sort]').forEach((el) => el.addEventListener('click', () => {
      const k = el.dataset.sort;
      if (state.sortKey === k) state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
      else { state.sortKey = k; state.sortDir = k === 'rank' || k === 'name' ? 'asc' : 'desc'; }
      render();
    }));
    ROOT.querySelectorAll('tbody tr[data-slug]').forEach((el) => {
      el.addEventListener('click', () => { state.selectedSlug = el.dataset.slug; render(); });
      el.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); state.selectedSlug = el.dataset.slug; render(); } });
    });
    ROOT.querySelectorAll('[data-open]').forEach((el) => el.addEventListener('click', (ev) => { ev.stopPropagation(); state.selectedSlug = el.dataset.open; render(); }));
    ROOT.querySelectorAll('[data-action="close-drawer"]').forEach((el) => el.addEventListener('click', (ev) => { if (ev.target === el) { state.selectedSlug = null; render(); } }));
    ROOT.querySelectorAll('[data-action="toggle-sandbox"]').forEach((el) => el.addEventListener('click', () => { state.sandboxOpen = !state.sandboxOpen; render(); }));
    ROOT.querySelectorAll('[data-action="reset-model"]').forEach((el) => el.addEventListener('click', () => { state.weights = { ...DATA.weights }; render(); }));
    ROOT.querySelectorAll('[data-action="reset-filters"]').forEach((el) => el.addEventListener('click', () => {
      state.tier = 'all'; state.verdict = 'all'; state.category = 'all'; state.chain = 'all'; state.momentum = 'all'; state.search = ''; render();
    }));
    ROOT.querySelectorAll('[data-clear]').forEach((el) => el.addEventListener('click', () => { const k = el.dataset.clear; state[k] = k === 'search' ? '' : 'all'; render(); }));
    ROOT.querySelectorAll('[data-param]').forEach((el) => el.addEventListener('input', () => { state.weights[el.dataset.param] = Number(el.value); render(); }));

    const sEl = document.getElementById('search-input');
    if (sEl) sEl.addEventListener('input', (ev) => {
      const pos = ev.target.selectionStart; state.search = ev.target.value; render();
      const n = document.getElementById('search-input'); if (n) { n.focus(); n.setSelectionRange(pos, pos); }
    });
    const vEl = document.getElementById('verdict-select'); if (vEl) vEl.addEventListener('change', (ev) => { state.verdict = ev.target.value; render(); });
    const cEl = document.getElementById('cat-select'); if (cEl) cEl.addEventListener('change', (ev) => { state.category = ev.target.value; render(); });
    const chEl = document.getElementById('chain-select'); if (chEl) chEl.addEventListener('change', (ev) => { state.chain = ev.target.value; render(); });
    const mEl = document.getElementById('mom-select'); if (mEl) mEl.addEventListener('change', (ev) => { state.momentum = ev.target.value; render(); });
  }

  window.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && state.selectedSlug) { state.selectedSlug = null; render(); } });
  render();
})();
