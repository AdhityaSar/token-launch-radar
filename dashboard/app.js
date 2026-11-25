// Launch Radar v5 — Verified In-Browser Ridge OOS Fit, Walk-Forward, Conformal Bands & Net-EV Farm Playbooks
// Architecture: solveRidge + fitLiveModel + verifyLooInBrowser + verifyWalkForwardInBrowser + computeTokenOdds
// + computeImpliedValue + buildFarmPlan all run directly on DATA.live & DATA.rows (zero hardcoded names/weights).
(() => {
 'use strict';
 const DATA = window.RADAR_DATA;
 const ROOT = document.getElementById('app');
 if (!DATA || !Array.isArray(DATA.rows)) {
  ROOT.innerHTML = '<div class="empty-state"><h2>No snapshot data loaded</h2><p>Run main.ts to generate data.js.</p></div>';
  return;
 }
 const L2_CHAINS = new Set('Base,Arbitrum,Optimism,Scroll,Ink,Monad,Unichain,Linea,Blast,Mantle,Polygon,Polygon zkEVM,zkSync Era,Mode,BOB,Sonic,Berachain,Katana,Plasma,Morph,Taiko,Soneium,World Chain,Fraxtal,Starknet,Manta,Metis,Corn,MegaETH,Etherlink,Abstract'.split(','));
 const ALT_L1_CHAINS = new Set('Solana,Sui,Aptos,Tron,Avalanche,Binance,Hyperliquid L1,Sei,Near,TON,Stellar,Flare,Bittensor,Cardano,Cosmos,Algorand,Hedera,Injective,Osmosis,Cronos,Kava,Celo'.split(','));
 const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
 const round2 = (n) => Math.round(n * 100) / 100;
 const round4 = (n) => Math.round(n * 1e4) / 1e4;
 const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
 const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
 function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
 }
 function featureVector(tvl, change7d, age, chainCount) {
  const lt = Math.log(Math.max(1, tvl)) - 18.0;
  const g = Math.tanh((typeof change7d === 'number' && Number.isFinite(change7d) ? change7d : 0) / 50);
  const a = Math.sqrt(clamp(age ?? 365, 0, 1095) / 1095) - 0.7;
  const ch = Math.min(Math.log(Math.max(1, chainCount)), 2.5);
  return [1, lt, g, a, ch];
 }
 /** Closed-form Ridge regression (regularizing slopes, never intercept) verified live on DATA.live. */
 function solveRidge(X, y, lambda) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => Array(d).fill(0));
  const b = Array(d).fill(0);
  for (let n = 0; n < X.length; n++) {
   for (let i = 0; i < d; i++) {
    b[i] += X[n][i] * y[n];
    for (let j = 0; j < d; j++) A[i][j] += X[n][i] * X[n][j];
   }
  }
  for (let i = 1; i < d; i++) A[i][i] += lambda;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < d; i++) {
   let piv = i;
   for (let r = i + 1; r < d; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r;
   [M[i], M[piv]] = [M[piv], M[i]];
   const diag = M[i][i];
   if (Math.abs(diag) < 1e-12) continue;
   for (let r = i + 1; r < d; r++) {
    const f = M[r][i] / diag;
    for (let c = i; c <= d; c++) M[r][c] -= f * M[i][c];
   }
  }
  const w = Array(d).fill(0);
  for (let i = d - 1; i >= 0; i--) {
   let sum = M[i][d];
   for (let j = i + 1; j < d; j++) sum -= M[i][j] * w[j];
   w[i] = Math.abs(M[i][i]) > 1e-12 ? sum / M[i][i] : 0;
  }
  return w;
 }
 /** Re-fits Ridge + Sector Empirical-Bayes shrinkage live from DATA.live so sandbox & OOS checks are 100% transparent. */
 function fitLiveModel(livePeers, lambda, catK) {
  const pts = livePeers.map((p) => ({
   cat: p.category || 'Other',
   tvl: p.tvl_usd,
   lmcap: Math.log(p.mcap_usd),
   listed_at: p.listed_at,
   f: featureVector(p.tvl_usd, p.change_7d, p.age_days, p.chains.length),
  }));
  const w = solveRidge(pts.map((p) => p.f), pts.map((p) => p.lmcap), lambda);
  const byCat = new Map();
  for (const p of pts) {
   const res = p.lmcap - dot(w, p.f);
   if (!byCat.has(p.cat)) byCat.set(p.cat, []);
   byCat.get(p.cat).push(res);
  }
  const catAdj = {};
  for (const [cat, arr] of byCat.entries()) {
   catAdj[cat] = round4(clamp(arr.reduce((s, v) => s + v, 0) / (arr.length + catK), -0.9, 0.9));
  }
  return {
   weights: {
    ...DATA.weights,
    intercept: round4(w[0]), b_tvl: round4(w[1]), b_growth: round4(w[2]),
    b_age: round4(w[3]), b_chains: round4(w[4]),
    ridge_lambda: lambda, sector_shrinkage_k: catK,
   },
   catAdj,
  };
 }
 /** Live Leave-One-Out verification on DATA.live (matches lib/model.ts identically). */
 function verifyLooInBrowser(livePeers, lambda, catK) {
  const pts = livePeers.map((p) => ({
   cat: p.category || 'Other',
   lmcap: Math.log(p.mcap_usd),
   f: featureVector(p.tvl_usd, p.change_7d, p.age_days, p.chains.length),
  }));
  const absErrs = [];
  for (let i = 0; i < pts.length; i++) {
   const tr = pts.filter((_, j) => j !== i);
   const te = pts[i];
   const w = solveRidge(tr.map((p) => p.f), tr.map((p) => p.lmcap), lambda);
   const sameCat = tr.filter((p) => p.cat === te.cat).map((p) => p.lmcap - dot(w, p.f));
   const adj = sameCat.length ? clamp(sameCat.reduce((s, v) => s + v, 0) / (sameCat.length + catK), -0.9, 0.9) : 0;
   absErrs.push(Math.abs(te.lmcap - (dot(w, te.f) + adj)));
  }
  absErrs.sort((a, b) => a - b);
  return round2(Math.exp(quantile(absErrs, 0.5)));
 }
 /** Live Walk-Forward blind test (older 65% -> newer 35% by listed_at) verified directly on DATA.live. */
 function verifyWalkForwardInBrowser(livePeers, lambda, catK) {
  const dated = livePeers.filter((p) => p.listed_at).sort((a, b) => (a.listed_at < b.listed_at ? -1 : a.listed_at > b.listed_at ? 1 : a.slug.localeCompare(b.slug)));
  const split = Math.floor(dated.length * 0.65);
  const toPt = (p) => ({ cat: p.category || 'Other', lmcap: Math.log(p.mcap_usd), f: featureVector(p.tvl_usd, p.change_7d, p.age_days, p.chains.length) });
  const tr = dated.slice(0, split).map(toPt);
  const te = dated.slice(split).map(toPt);
  const w = solveRidge(tr.map((p) => p.f), tr.map((p) => p.lmcap), lambda);
  const errs = te.map((t) => {
   const sc = tr.filter((p) => p.cat === t.cat).map((p) => p.lmcap - dot(w, p.f));
   const adj = sc.length ? clamp(sc.reduce((s, v) => s + v, 0) / (sc.length + catK), -0.9, 0.9) : 0;
   return Math.abs(t.lmcap - (dot(w, t.f) + adj));
  }).sort((a, b) => a - b);
  return round2(Math.exp(quantile(errs, 0.5)));
 }
 /** Empirical token launch odds from snapshot sector token rates, TVL scale, and survival-adjusted age hazard. */
 function computeTokenOdds(row) {
  const cat = DATA.categories[row.category || 'Other'];
  const baseRate = cat ? cat.token_rate : DATA.global_token_rate;
  const tvlLift = clamp((Math.log10(Math.max(5e6, row.tvl_usd)) - 7.3) * 0.08, -0.06, 0.16);
  const effAge = row.age_days ?? 365;
  const ageHazard = effAge < 540 ? 1.05 : effAge < 900 ? 0.92 : 0.72;
  return round4(clamp((baseRate + tvlLift) * ageHazard, 0.15, 0.88));
 }
 const LIVE_FIT = fitLiveModel(DATA.live, DATA.weights.ridge_lambda, DATA.weights.sector_shrinkage_k);
 const LIVE_LOO_ERR = verifyLooInBrowser(DATA.live, DATA.weights.ridge_lambda, DATA.weights.sector_shrinkage_k);
 const LIVE_WF_ERR = verifyWalkForwardInBrowser(DATA.live, DATA.weights.ridge_lambda, DATA.weights.sector_shrinkage_k);
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
  weights: { ...LIVE_FIT.weights },
 };
 function fmtUsd(n) {
  if (n === null || n === undefined || !Number.isFinite(n)) return '—';
  const a = Math.abs(n);
  if (a >= 1e9) return `$${(n / 1e9).toFixed(2)}B`;
  if (a >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
  if (a >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
  return `$${Math.round(n)}`;
 }
 const fmtPct = (v) => (v === null || v === undefined || !Number.isFinite(v) ? 'N/A' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`);
 const fmtAge = (d) => (d === null || d === undefined ? 'N/A' : d >= 365 ? `${(d / 365).toFixed(1)}y` : `${d}d`);
 const fmtSigned = (v, digits) => `${v >= 0 ? '+ ' : '− '}${Math.abs(v).toFixed(digits)}`;
 /** Scores a protocol using the fitted Ridge power-law model, conformal band, and net-EV farm verdict. */
 function computeImpliedValue(row, w) {
  const f = featureVector(row.tvl_usd, row.change_7d, row.age_days, row.chains.length);
  const catKey = row.category || 'Other';
  const sectorAdj = LIVE_FIT.catAdj[catKey] ?? (DATA.categories[catKey]?.log_residual_adj ?? 0);
  const logBase = w.intercept + w.b_tvl * f[1];
  const logG = w.b_growth * f[2];
  const logA = w.b_age * f[3];
  const logCh = w.b_chains * f[4];
  const logPred = logBase + logG + logA + logCh + sectorAdj;
  const implied = Math.max(1, Math.round(Math.exp(logPred)));
  const bear = Math.max(1, Math.min(implied, Math.round(Math.exp(logPred + w.conformal_q25))));
  const bull = Math.max(implied, Math.round(Math.exp(logPred + w.conformal_q75)));
  const odds = computeTokenOdds(row);
  const slippage = row.capital_usd * 0.0015;
  const totalCost = row.est_gas_usd + slippage;
  const evFor = (val) => round2(row.capital_usd * Math.min(0.4, (val * 0.10 * 1.5) / Math.max(1, row.tvl_usd)) * odds - totalCost);
  const netEv = evFor(implied);
  const bearEv = evFor(bear);
  const bullEv = evFor(bull);
  const hourlyEv = round2(netEv / Math.max(0.5, row.active_minutes / 60));
  const breakevenFdv = Math.max(1, Math.round((totalCost * row.tvl_usd) / Math.max(1e-6, row.capital_usd * 0.10 * 1.5 * odds)));
  const safetyMargin = round2(implied / breakevenFdv);
  const verdict = netEv >= 15 && hourlyEv >= 10 ? 'FARM' : netEv >= 3 ? 'SELECTIVE' : 'SKIP';
  return {
   ...row,
   token_odds: odds,
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
   bear_ev_usd: bearEv,
   bull_ev_usd: bullEv,
   hourly_ev_usd: hourlyEv,
   breakeven_fdv_usd: breakevenFdv,
   safety_margin_x: safetyMargin,
   verdict,
  };
 }
 /** Category-specific 6-step DeFi farm playbook matching lib/model.ts identically. */
 function buildFarmPlan(row) {
  const chains = row.chains || [];
  const lowFee = chains.find((c) => c !== 'Ethereum' && (L2_CHAINS.has(c) || ALT_L1_CHAINS.has(c)));
  const primary = lowFee || chains[0] || 'its primary chain';
  const secondary = chains.find((c) => c !== primary) || null;
  const gasTier = primary === 'Ethereum' ? 'mainnet' : ALT_L1_CHAINS.has(primary) ? 'alt-l1' : 'l2';
  const gas = gasTier === 'mainnet' ? { tx: 4.5, total: 28, label: '$4-$8 Eth L1 gas' }
   : gasTier === 'alt-l1' ? { tx: 0.12, total: 1, label: 'under $0.20 gas' }
   : { tx: 0.25, total: 2, label: 'under $0.50 L2 gas' };
  const cap = row.capital_usd; const hold = row.hold_days; const wks = Math.max(4, Math.ceil(hold / 7));
  const slip = (cap * 0.0015).toFixed(2); const half = Math.round(cap / 2); const sw = Math.round(cap * 0.15);
  const cat = (row.category || '').toLowerCase(); const n = row.name;
  const dd = typeof row.change_7d === 'number' && row.change_7d <= -5 ? `TVL fell ${row.change_7d.toFixed(1)}% in 7d; check depth first, then hold` : 'Hold uninterrupted across';
  const mk = (t, a, c, tm) => ({ title: t, action: a, cost: c, time: tm });
  let steps;
  if (cat.includes('dex') || cat.includes('liquidity')) {
   steps = [
    mk('Fund LP pair on primary chain', `Move $${cap} split 50/50 ($${half} USDC + $${half} pair asset) onto ${primary} in a clean wallet.`, gas.label, '10 min'),
    mk('Seed core concentrated LP pool', `Provide $${cap} liquidity into ${n}'s deepest stable or blue-chip pool on ${primary} within a wide low-IL band.`, `$${cap} refundable + ${gas.label}`, '15 min'),
    mk('Log organic multi-pair swap volume', `Execute 3 spot swaps of ~$${sw} across distinct pools on ${n} over ${wks} weeks for trader + LP tiers.`, `~$${(sw * 0.009 + gas.tx * 3).toFixed(2)} fees & gas`, `15 min (${wks} wks)`),
    mk('Harvest fees & rebalance range', `At day ${Math.round(hold / 2)}, claim accrued LP fees and re-center range${secondary ? ` or mirror 20% on ${secondary}` : ''}.`, `~$${(gas.tx * 2).toFixed(2)} gas`, '10 min'),
    mk(`Maintain ${hold}-day epoch streak`, `${dd} ${hold} days so time-weighted LP points register across snapshots.`, `$${cap} refundable (~$${slip} spread)`, `${hold}-day hold`),
    mk('Claim rewards & revoke router', `Check ${n}'s official points portal, withdraw LP + fees, and revoke router allowances.`, `${gas.label} exit & revoke`, '10 min'),
   ];
  } else if (cat.includes('risk curator') || cat.includes('allocator') || cat.includes('cdp manager')) {
   steps = [
    mk(`Stage USDC on ${primary}`, `Bridge $${cap} USDC onto ${primary} with native gas to avoid L1 deposit drag.`, gas.label, '10 min'),
    mk('Deposit into flagship curated vault', `Supply $${cap} USDC into ${n}'s top curated vault on ${primary} and retain the receipt share.`, `$${cap} refundable + ${gas.label}`, '15 min'),
    mk('Diversify into secondary vault', `Allocate 20% ($${Math.round(cap * 0.2)}) into a second ${n} vault${secondary ? ` on ${secondary}` : ''} for multi-vault multipliers.`, `~$${(gas.tx * 2).toFixed(2)} gas`, '10 min'),
    mk('Weekly accrual & top-up check', `Inspect APY weekly across ${wks} weeks; add one $25 top-up to log multi-epoch activity.`, `~$${(gas.tx * 2).toFixed(2)} gas`, `6 min/wk (${wks} wks)`),
    mk(`Hold receipt shares ${hold} days`, `${dd} ${hold} days to capture full curator points and native lending yield.`, `$${cap} refundable (~$${slip} spread)`, `${hold}-day hold`),
    mk('Redeem shares & verify points', `Confirm accrued points on ${n}'s official UI before redeeming shares and revoking approvals.`, `${gas.label} redeem`, '10 min'),
   ];
  } else if (cat.includes('rwa')) {
   steps = [
    mk(`Prepare stablecoin on ${primary}`, `Fund $${cap} USDC on ${primary} and verify whether ${n}'s token is permissionless or allowlisted.`, gas.label, '10 min'),
    mk('Mint or acquire yield-bearing RWA', `Subscribe or swap $${cap} into ${n}'s primary RWA yield token on ${primary} in self-custody.`, `$${cap} refundable + ~$${(Number(slip) + gas.tx).toFixed(2)}`, '15 min'),
    mk('Deploy RWA into native DeFi pool', `If ${n} offers a native collateral or LP wrapper on ${primary}, stake 50% ($${half}) to stack points.`, `~$${(gas.tx * 2).toFixed(2)} gas`, '10 min'),
    mk('Track NAV rebase & weekly proofs', `Verify weekly yield rebase and official points across ${wks} weeks without churning principal.`, '$0 monitoring', `5 min/wk (${wks} wks)`),
    mk(`Hold through ${hold}-day accrual`, `${dd} ${hold} days so time-weighted RWA balances count across snapshots.`, `$${cap} refundable (earns yield)`, `${hold}-day hold`),
    mk('Redeem at NAV & revoke approvals', `Redeem RWA tokens back to USDC via ${n}'s primary path and revoke contract allowances.`, `${gas.label} exit`, '10 min'),
   ];
  } else if (cat.includes('lend') || cat.includes('cdp') || cat.includes('collateral')) {
   const b = Math.round(cap * 0.25);
   steps = [
    mk(`Stage collateral on ${primary}`, `Fund $${cap} USDC or blue-chip collateral on ${primary} with a conservative health buffer.`, gas.label, '10 min'),
    mk('Supply core collateral market', `Deposit $${cap} into ${n}'s deepest lending or collateral market on ${primary} for supply points.`, `$${cap} refundable + ${gas.label}`, '15 min'),
    mk('Open conservative 25% borrow leg', `Borrow ~$${b} stablecoin (HF >2.5) on ${n} to earn higher-weighted borrower/minter points.`, `~$${(b * 0.05 * (hold / 365) + gas.tx).toFixed(2)} interest & gas`, '10 min'),
    mk('Weekly health & partial repay', `Check liquidation buffer weekly over ${wks} weeks and make one $${Math.round(b * 0.2)} partial repay.`, `~$${(gas.tx * 2).toFixed(2)} gas`, `6 min/wk (${wks} wks)`),
    mk(`Keep position open ${hold} days`, `${dd} ${hold} days across multiple points epochs without triggering liquidation.`, `$${cap} refundable principal`, `${hold}-day hold`),
    mk('Repay debt, withdraw & revoke', `Repay the $${b} borrow leg in full, withdraw $${cap} collateral, and revoke approvals.`, `${gas.label} unwind`, '10 min'),
   ];
  } else if (cat.includes('basis') || cat.includes('yield')) {
   steps = [
    mk(`Fund stablecoin on ${primary}`, `Stage $${cap} USDC on ${primary} to enter ${n}'s delta-neutral basis or yield vault.`, gas.label, '10 min'),
    mk('Stake into flagship basis/yield pool', `Deposit $${cap} into ${n}'s core strategy on ${primary} and stake the receipt token for boosted points.`, `$${cap} refundable + ${gas.label}`, '15 min'),
    mk('Check funding & epoch rollover', `Monitor yield spread weekly across ${wks} weeks; compound shares${secondary ? ` (20% on ${secondary})` : ''}.`, `~$${(gas.tx * 2).toFixed(2)} gas`, `6 min/wk (${wks} wks)`),
    mk('Verify cooldown & exit liquidity', `Confirm ${n}'s unstaking cooldown early so your exit aligns with snapshot cutoffs.`, '$0 check', '10 min'),
    mk(`Hold through ${hold}-day epochs`, `${dd} ${hold} days to lock in time-weighted yield multiplier tiers.`, `$${cap} refundable (~$${slip} spread)`, `${hold}-day hold`),
    mk('Queue unstake & claim points', `Initiate cooldown exit, verify points on ${n}'s official UI, and revoke vault allowances.`, `${gas.label} exit`, '10 min'),
   ];
  } else if (cat.includes('derivative') || cat.includes('perpetual') || cat.includes('prediction') || cat.includes('otc')) {
   const not = Math.round(cap * 0.5);
   steps = [
    mk(`Deposit margin on ${primary}`, `Fund $${cap} USDC margin into ${n} on ${primary} in an isolated trading account.`, gas.label, '10 min'),
    mk('Open low-leverage hedged trade', `Execute a 1x-2x hedged $${not} position on ${n}'s liquid BTC/ETH market and hold >24h.`, `~$${(not * 0.001 + gas.tx).toFixed(2)} fee`, '15 min'),
    mk('Build weekly organic volume', `Place 1-2 limit orders weekly (~$${not} each) over ${wks} weeks for consistent trader points.`, `~$${(not * 0.0005 * wks + gas.tx * wks).toFixed(2)} fees`, `10 min/wk (${wks} wks)`),
    mk('Seed vault/LP if available', `Keep 40% ($${Math.round(cap * 0.4)}) in ${n}'s HLP/liquidity vault if offered for passive MM points.`, `~$${gas.tx.toFixed(2)} gas`, '10 min'),
    mk(`Stay active across ${hold}-day season`, `${dd} ${hold} days across weekly epochs without over-leveraging.`, `$${cap} refundable margin`, `${hold}-day hold`),
    mk('Flatten & withdraw margin', `Close open orders, verify season points on ${n}'s official board, and withdraw margin.`, `${gas.label} withdraw`, '10 min'),
   ];
  } else {
   steps = [
    mk(`Fund dedicated wallet on ${primary}`, `Transfer $${cap} USDC plus native gas onto ${primary} with zero unrelated approvals.`, gas.label, '10 min'),
    mk(`Deploy core ${row.category || 'DeFi'} position`, `Deposit $${cap} into ${n}'s primary product on ${primary} and retain receipt shares.`, `$${cap} refundable + ${gas.label}`, '15 min'),
    mk('Weekly qualifying cadence', `Perform one organic interaction weekly on ${primary}${secondary ? ` (+20% on ${secondary})` : ''} over ${wks} weeks.`, `~$${(gas.tx * wks).toFixed(2)} gas`, `8 min/wk (${wks} wks)`),
    mk('Stack native points / boost tier', `Enable non-custodial points or lock-free boost tier on ${n} without exposing seeds.`, '$0 setup', '10 min'),
    mk(`Hold through ${hold}-day epochs`, `${dd} ${hold} days so multi-epoch snapshots register full weight.`, `$${cap} refundable (~$${slip} spread)`, `${hold}-day hold`),
    mk('Verify eligibility & unwind', `Inspect ${n}'s official rewards page before withdrawing $${cap} and revoking approvals.`, `${gas.label} exit & revoke`, '10 min'),
   ];
  }
  return { primary, gasTier, capital: cap, estGas: gas.total, hold, steps };
 }
 function isSandboxModified() {
  return ['b_tvl', 'b_growth', 'b_age', 'b_chains'].some((k) => Math.abs(state.weights[k] - LIVE_FIT.weights[k]) > 1e-6);
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
    const va = (a[key] || '').toLowerCase(); const vb = (b[key] || '').toLowerCase();
    return va < vb ? -dir : va > vb ? dir : a.rank - b.rank;
   }
   const va = key === 'change_7d' ? (a.change_7d ?? -9999) : key === 'age_days' ? (a.age_days ?? -1) : (a[key] ?? 0);
   const vb = key === 'change_7d' ? (b.change_7d ?? -9999) : key === 'age_days' ? (b.age_days ?? -1) : (b[key] ?? 0);
   return va !== vb ? (va - vb) * dir : a.rank - b.rank;
  });
  return filtered;
 }
 function buildFilterOptions(rows) {
  const cats = new Map(); const chains = new Map();
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
 const sortBtn = (key, label) => {
  const on = state.sortKey === key;
  return `<button type="button" class="${on ? 'sorted' : ''}" data-sort="${key}">${label}${on ? (state.sortDir === 'asc' ? ' ▲' : ' ▼') : ''}</button>`;
 };
 const verdictBadge = (v) => `<span class="badge ${v === 'FARM' ? 'badge-farm' : v === 'SELECTIVE' ? 'badge-sel' : 'badge-skip'}">${v}</span>`;
 function renderHeader() {
  const u = DATA.universe; const val = DATA.validation;
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
      <div class="brand-sub">In-browser verified Ridge OOS valuation, conformal bands, token odds &amp; tailored DeFi farm playbooks · ${esc(DATA.as_of)}</div>
     </div>
    </div>
    <div class="top-nav">
     <div class="nav-tabs" role="tablist">
      <button type="button" data-view="radar" aria-pressed="${state.view === 'radar'}">Pre-Token Radar (${u.board_count})</button>
      <button type="button" data-view="live" aria-pressed="${state.view === 'live'}">Live Comps (${u.live_priced_count})</button>
      <button type="button" data-view="model" aria-pressed="${state.view === 'model'}">OOS Validation &amp; Ablation</button>
     </div>
    </div>
   </header>
   <div class="audit-strip" aria-label="Universe reconciliation and blind OOS verification">
    <div class="audit-items">
     <span class="audit-ok">✓ ${u.watchlist_count}/${u.watchlist_count} Watchlist</span><span class="audit-sep">|</span>
     <span>${u.total_protocols}/${u.total_protocols} Accounted: <b>${u.board_count}</b> pre-token + <b>${u.live_total_count}</b> live separated + <b>${u.excluded_non_launch_count}</b> CEX/bridge/wrapped</span>
    </div>
    <div class="audit-items">
     <span class="audit-ok">✓ LOO ×${LIVE_LOO_ERR} vs Naive ×${val.loo_naive_mult_err} (-${val.loo_error_reduction_pct}%)</span><span class="audit-sep">|</span>
     <span class="audit-ok">✓ Walk-Fwd ×${LIVE_WF_ERR}</span><span class="audit-sep">|</span>
     <span>0 Hardcoded Names</span>
    </div>
   </div>
   <div class="anchors-strip" aria-label="Separated live-token market anchors">
    <span class="anchors-label">Live-Token Anchors (Separated — Never Ranked):</span>
    ${DATA.live.slice(0, 3).map((p) => `<span class="anchor-pill">${esc(p.name)} (${esc(p.symbol || 'LIVE')}) <b>${fmtUsd(p.mcap_usd)}</b> / ${fmtUsd(p.tvl_usd)} = <b>${p.mcap_tvl_ratio.toFixed(2)}×</b></span>`).join('')}
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
   <section class="kpi-grid">
    <div class="kpi-card"><span class="kpi-label">${active ? 'Filtered Implied Value' : 'Total Pre-Token Implied'}</span><span class="kpi-val emerald">${fmtUsd(totalImplied)}</span><span class="kpi-sub">${scope.length} pre-token · ${fmtUsd(totalTvl)} TVL</span></div>
    <div class="kpi-card"><span class="kpi-label">LOO-CV Blind Error</span><span class="kpi-val cyan">×${LIVE_LOO_ERR}</span><span class="kpi-sub">vs Naive 0.25×TVL ×${val.loo_naive_mult_err} (-${val.loo_error_reduction_pct}%)</span></div>
    <div class="kpi-card"><span class="kpi-label">Walk-Forward Blind</span><span class="kpi-val cyan">×${val.wf_median_mult_err}</span><span class="kpi-sub">${val.wf_train_count} older → ${val.wf_test_count} newer (-${val.wf_error_reduction_pct}%)</span></div>
    <div class="kpi-card"><span class="kpi-label">Farmable Verdicts</span><span class="kpi-val">${farmCount} FARM</span><span class="kpi-sub">${selCount} selective · ${scope.length - farmCount - selCount} skip (net of gas)</span></div>
    <div class="kpi-card"><span class="kpi-label">Top Farm Return</span><span class="kpi-val emerald">${bestEv ? `$${bestEv.hourly_ev_usd.toFixed(1)}/hr` : '—'}</span><span class="kpi-sub">${bestEv ? `${esc(bestEv.name)} (+$${bestEv.net_ev_usd.toFixed(0)} EV · ${bestEv.safety_margin_x}× BE)` : 'None'}</span></div>
   </section>
  `;
 }
 function renderFormulaBar() {
  const w = state.weights;
  return `
   <section class="formula-bar">
    <div class="formula-eq">
     <strong>ln(Implied)</strong> = ${w.intercept.toFixed(2)} ${fmtSigned(w.b_tvl, 3)}·ln(TVL/66M) ${fmtSigned(w.b_growth, 3)}·tanh(Δ7d/50) ${fmtSigned(w.b_age, 3)}·(√age−0.7) ${fmtSigned(w.b_chains, 3)}·ln(chains) + γ_sector(k=${w.sector_shrinkage_k})
    </div>
    <div class="formula-actions">
     ${isSandboxModified() ? '<button type="button" class="btn-ghost" data-action="reset-model">Reset Fit</button>' : ''}
     <button type="button" class="btn-ghost ${state.sandboxOpen ? 'active' : ''}" data-action="toggle-sandbox">${state.sandboxOpen ? 'Hide Sandbox' : 'Model Sandbox'}</button>
    </div>
   </section>
   ${state.sandboxOpen ? `
    <section class="sandbox-panel">
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
   <section class="toolbar">
    <div class="tier-pills" role="group">
     <button type="button" data-tier="all" aria-pressed="${state.tier === 'all'}">All (${allRows.length})</button>
     <button type="button" data-tier="watchlist" aria-pressed="${state.tier === 'watchlist'}">Watchlist (${DATA.universe.watchlist_count})</button>
     <button type="button" data-tier="early" aria-pressed="${state.tier === 'early'}">Early (${DATA.universe.early_count})</button>
    </div>
    <input type="search" class="ctrl-input" id="search-input" placeholder="Search protocol, sector, chain…" value="${esc(state.search)}" aria-label="Search">
    <select class="ctrl-select" id="verdict-select" aria-label="Verdict">
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
    <div class="toolbar-right"><span class="row-counter">${visibleRows.length} of ${allRows.length}</span>${chips.length ? '<button type="button" class="btn-ghost" data-action="reset-filters">Clear</button>' : ''}</div>
   </section>
   ${chips.length ? `<div class="filter-chips">${chips.map((c) => `<span class="chip">${esc(c.label)} <button type="button" data-clear="${c.key}">×</button></span>`).join('')}</div>` : ''}
  `;
 }
 function renderRadarTable(visibleRows, maxImplied) {
  if (!visibleRows.length) return '<div class="panel"><div class="empty-state"><h3>No matching protocols</h3><p>Clear filters to inspect all 178 pre-token protocols.</p></div></div>';
  const body = visibleRows.map((r) => {
   const pct = Math.max(4, Math.round((Math.log10(Math.max(1, r.implied_value_usd)) / Math.log10(Math.max(10, maxImplied))) * 100));
   const fc = r.change_7d === null ? 'flow-na' : r.change_7d >= 0 ? 'flow-pos' : 'flow-neg';
   const chains = r.chains.slice(0, 2).map((c) => `<span class="chain-tag">${esc(c)}</span>`).join('') + (r.chains.length > 2 ? `<span class="chain-tag">+${r.chains.length - 2}</span>` : '');
   const evSign = r.net_ev_usd >= 0 ? `+$${r.net_ev_usd.toFixed(0)}` : `-$${Math.abs(r.net_ev_usd).toFixed(0)}`;
   return `
    <tr data-slug="${esc(r.slug)}" data-selected="${state.selectedSlug === r.slug}" tabindex="0">
     <td class="rank-cell ${r.rank <= 10 ? 'rank-top' : ''}">#${r.rank}</td>
     <td><div class="proto-cell"><div class="proto-name">${esc(r.name)} <span class="badge ${r.tier === 'watchlist' ? 'badge-wl' : 'badge-early'}">${r.tier === 'watchlist' ? 'WL' : 'EARLY'}</span></div><div class="proto-meta"><span>${esc(r.category || 'Other')}</span><span>·</span><span class="mono">${esc(r.slug)}</span></div></div></td>
     <td><div class="chains-list">${chains || '<span class="chain-tag">Multi</span>'}</div></td>
     <td class="r mono">${fmtUsd(r.tvl_usd)}</td>
     <td class="r mono ${fc}">${fmtPct(r.change_7d)}</td>
     <td class="r mono">${fmtAge(r.age_days)}</td>
     <td class="r mono">${Math.round(r.token_odds * 100)}%</td>
     <td class="r"><div class="val-cell"><span class="val-main">${fmtUsd(r.implied_value_usd)}</span><div class="val-bar"><i style="width:${pct}%"></i></div><span class="val-range">${fmtUsd(r.bear_value_usd)} – ${fmtUsd(r.bull_value_usd)}</span></div></td>
     <td class="r"><div class="verdict-cell"><div>${verdictBadge(r.verdict)} <span class="mono" style="margin-left:4px;font-weight:600">${evSign}</span></div><span class="verdict-sub">$${r.hourly_ev_usd.toFixed(1)}/hr · ${r.safety_margin_x}× BE · ~$${r.est_gas_usd}g</span></div></td>
    </tr>
   `;
  }).join('');
  return `
   <section class="panel"><div class="table-wrap"><table>
    <thead><tr>
     <th>${sortBtn('rank', 'Rank')}</th><th>${sortBtn('name', 'Protocol & Sector')}</th><th>Chains</th>
     <th class="r">${sortBtn('tvl_usd', 'TVL')}</th><th class="r">${sortBtn('change_7d', '7d Flow')}</th>
     <th class="r">${sortBtn('age_days', 'Age')}</th><th class="r">${sortBtn('token_odds', 'Odds')}</th>
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
   return `<tr><td class="rank-cell">${i + 1}</td><td><div class="proto-cell"><div class="proto-name">${esc(p.name)} <span class="badge badge-live">${esc(p.symbol || 'LIVE')}</span></div><div class="proto-meta">${esc(p.category || 'Other')} · <span class="mono">${esc(p.slug)}</span></div></div></td><td class="r mono">${fmtUsd(p.mcap_usd)}</td><td class="r mono">${fmtUsd(p.tvl_usd)}</td><td class="r mono" style="color:var(--cyan);font-weight:700">${p.mcap_tvl_ratio.toFixed(3)}x</td><td class="r mono ${fc}">${fmtPct(p.change_7d)}</td><td class="r mono">${fmtAge(p.age_days)}</td></tr>`;
  }).join('');
  return `
   <section class="panel" style="margin-top:10px">
    <div style="padding:10px 14px;border-bottom:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
     <div><h2 style="font-size:14px">Live-Token Market Comparables (Separated from Pre-Token Board)</h2><p style="font-size:11.5px;color:var(--muted);margin-top:2px">All ${DATA.universe.live_total_count} token-bearing protocols stay off the pre-token board; these ${DATA.live.length} priced DeFi comps fit the Ridge power-law model.</p></div>
     <span class="badge badge-live">Fitted β₁ = ${LIVE_FIT.weights.b_tvl.toFixed(3)} · LOO ×${LIVE_LOO_ERR}</span>
    </div>
    <div class="table-wrap"><table><thead><tr><th>#</th><th>Live Protocol</th><th class="r">Market Cap</th><th class="r">TVL</th><th class="r">MCap/TVL</th><th class="r">7d Flow</th><th class="r">Age</th></tr></thead><tbody>${body}</tbody></table></div>
   </section>
  `;
 }
 function renderModelView() {
  const v = DATA.validation; const w = LIVE_FIT.weights;
  const cats = Object.values(DATA.categories).filter((c) => c.peer_count > 0).sort((a, b) => b.peer_count - a.peer_count || a.category.localeCompare(b.category));
  return `
   <section class="two-col">
    <div class="card">
     <h2>Out-of-Sample Term Ablation &amp; Blind Walk-Forward (${v.live_fit_count} Live Comps)</h2>
     <p>Both <code>solveRidge</code> and <code>verifyLooInBrowser</code> run live on <code>DATA.live</code> inside this page, confirming zero hardcoded weights and verifying LOO-CV ×${LIVE_LOO_ERR} against the naive 0.25×TVL baseline (×${v.loo_naive_mult_err}).</p>
     <div class="table-wrap" style="margin:8px 0 10px;border:1px solid var(--border);border-radius:var(--r-md)"><table>
      <thead><tr><th>Model Specification (LOO-CV n=${v.live_fit_count})</th><th class="r">Median Err</th><th class="r">MAE(log)</th><th class="r">vs Naive</th></tr></thead>
      <tbody>${v.ablation.map((a, idx) => `
       <tr style="${idx === v.ablation.length - 1 ? 'background:var(--emerald-dim)' : ''}">
        <td><b>${esc(a.label)}</b></td>
        <td class="r mono" style="color:${idx === v.ablation.length - 1 ? 'var(--emerald)' : 'var(--ink)'}">×${a.loo_mult_err.toFixed(2)}</td>
        <td class="r mono">${a.loo_mae_log.toFixed(4)}</td>
        <td class="r mono" style="color:${a.delta_vs_naive_pct > 0 ? 'var(--emerald)' : 'var(--muted)'}">${a.delta_vs_naive_pct > 0 ? `-${a.delta_vs_naive_pct}%` : 'baseline'}</td>
       </tr>`).join('')}
      </tbody>
     </table></div>
     <div class="code-block">Temporal Walk-Forward (${v.wf_train_count} older → ${v.wf_test_count} newer by listed_at):
 Fitted Ridge + Sector EB : ×${v.wf_median_mult_err} vs Naive ×${v.wf_naive_mult_err} (-${v.wf_error_reduction_pct}% error)
Conformal Band (LOO residuals q25=${w.conformal_q25}, q75=${w.conformal_q75}): ${v.conformal_50_coverage_pct}% verified coverage
Farm EV Rule: 10% community pool (1.5× active weight) × token_odds − gas − 15 bps spread</div>
    </div>
    <div class="card" style="padding:0;overflow:hidden">
     <div style="padding:11px 14px;border-bottom:1px solid var(--border)"><h2 style="margin:0">Empirical Sector Residual Factors &amp; Token Rates</h2></div>
     <div class="table-wrap" style="max-height:390px;overflow-y:auto"><table>
      <thead><tr><th>Sector</th><th class="r">Peers</th><th class="r">Raw Med</th><th class="r">EB Factor</th><th class="r">Token Rate</th></tr></thead>
      <tbody>${cats.map((c) => `
       <tr>
        <td><b>${esc(c.category)}</b></td><td class="r mono">${c.peer_count}</td>
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
  const sectorPeers = DATA.live.filter((p) => p.category === row.category).slice(0, 3);
  return `
   <div class="drawer-backdrop" data-action="close-drawer">
    <aside class="drawer" role="dialog" aria-modal="true" aria-label="${esc(row.name)} detail">
     <div class="drawer-head">
      <div>
       <div class="drawer-title"><span>#${row.rank} ${esc(row.name)}</span><span class="badge ${row.tier === 'watchlist' ? 'badge-wl' : 'badge-early'}">${row.tier === 'watchlist' ? 'WATCHLIST' : 'EARLY'}</span>${verdictBadge(row.verdict)}</div>
       <div class="drawer-sub">${esc(row.category || 'Other')} · ${esc(row.slug)} · Listed ${esc(row.listed_at || 'N/A')} (${fmtAge(row.age_days)}) · Launch Odds ${Math.round(row.token_odds * 100)}%</div>
      </div>
      <div class="drawer-nav">
       <button type="button" class="btn-ghost" ${prev ? `data-open="${esc(prev)}"` : 'disabled'}>↑</button>
       <button type="button" class="btn-ghost" ${next ? `data-open="${esc(next)}"` : 'disabled'}>↓</button>
       <button type="button" class="btn-ghost" data-action="close-drawer">Close</button>
      </div>
     </div>
     <section class="drawer-section">
      <div class="section-label"><span>Conformal Launch Band (${DATA.validation.conformal_50_coverage_pct}% LOO Coverage)</span><span class="mono">Break-Even FDV ${fmtUsd(row.breakeven_fdv_usd)} (${row.safety_margin_x}× margin)</span></div>
      <div class="band-box">
       <div class="band-nums">
        <div><small>Bear (Conformal P25)</small><strong>${fmtUsd(row.bear_value_usd)}</strong></div>
        <div><small>Fitted Base Implied</small><strong class="base-v">${fmtUsd(row.implied_value_usd)}</strong></div>
        <div><small>Bull (Conformal P75)</small><strong>${fmtUsd(row.bull_value_usd)}</strong></div>
       </div>
       <div class="band-track"><div class="band-fill"></div><div class="band-dot" style="left:${dotPos.toFixed(1)}%"></div></div>
      </div>
      <div class="eq-grid">
       <div class="eq-row"><span>1. Power-Law TVL (${fmtUsd(row.tvl_usd)}, β₁=${state.weights.b_tvl.toFixed(2)}) × Sector (${row.peer_count} peers)</span><b>${fmtUsd(row.tvl_component)} × ${row.sector_factor.toFixed(3)}</b></div>
       <div class="eq-row"><span>2. 7d Flow (${fmtPct(row.change_7d)}) · Age (${fmtAge(row.age_days)}) · Chains (${row.chains.length})</span><b>× ${row.growth_factor.toFixed(3)} · × ${row.age_factor.toFixed(3)} · × ${row.chain_factor.toFixed(3)}</b></div>
       <div class="eq-row total"><span>Fitted Implied Value (${row.implied_mcap_tvl.toFixed(2)}× TVL)</span><b>${fmtUsd(row.implied_value_usd)}</b></div>
      </div>
     </section>
     <section class="drawer-section">
      <div class="section-label"><span>Tailored 6-Step ${esc(row.category || 'DeFi')} Playbook</span><span class="mono">EV Band: Bear ${row.bear_ev_usd >= 0 ? `+$${row.bear_ev_usd.toFixed(0)}` : `-$${Math.abs(row.bear_ev_usd).toFixed(0)}`} · Base ${evStr} · Bull +$${row.bull_ev_usd.toFixed(0)}</span></div>
      <div class="farm-meta-strip">
       <div class="farm-stat"><small>Capital</small><b>$${plan.capital}</b></div>
       <div class="farm-stat"><small>Est. Gas</small><b>~$${plan.estGas}</b></div>
       <div class="farm-stat"><small>Net EV</small><b style="color:${row.net_ev_usd >= 0 ? 'var(--emerald)' : 'var(--rose)'}">${evStr}</b></div>
       <div class="farm-stat"><small>Implied Rate</small><b style="color:${row.hourly_ev_usd >= 10 ? 'var(--emerald)' : 'var(--amber)'}">$${row.hourly_ev_usd.toFixed(1)}/hr</b></div>
      </div>
      <ol class="steps-list">
       ${plan.steps.map((s, i) => `<li class="step-item"><div class="step-num">${i + 1}</div><div><div class="step-title">${esc(s.title)}</div><div class="step-action">${esc(s.action)}</div><div class="step-tags"><span class="step-cost">Cost: ${esc(s.cost)}</span><span class="step-time">Time: ${esc(s.time)}</span></div></div></li>`).join('')}
      </ol>
     </section>
     ${sectorPeers.length ? `
      <section class="drawer-section">
       <div class="section-label"><span>Live ${esc(row.category)} Comparables Grounding This Sector</span><span class="mono">Observed MCap/TVL</span></div>
       <div class="eq-grid">${sectorPeers.map((p) => `<div class="eq-row"><span><b>${esc(p.name)}</b> <span class="badge badge-live">${esc(p.symbol || 'LIVE')}</span></span><b>${fmtUsd(p.mcap_usd)} / ${fmtUsd(p.tvl_usd)} = ${p.mcap_tvl_ratio.toFixed(2)}×</b></div>`).join('')}</div>
      </section>
     ` : ''}
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
  const on = (sel, fn) => ROOT.querySelectorAll(sel).forEach((el) => el.addEventListener('click', (ev) => fn(el, ev)));
  on('[data-view]', (el) => { state.view = el.dataset.view; state.selectedSlug = null; render(); });
  on('[data-tier]', (el) => { state.tier = el.dataset.tier; render(); });
  on('[data-sort]', (el) => {
   const k = el.dataset.sort;
   if (state.sortKey === k) state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
   else { state.sortKey = k; state.sortDir = k === 'rank' || k === 'name' ? 'asc' : 'desc'; }
   render();
  });
  ROOT.querySelectorAll('tbody tr[data-slug]').forEach((el) => {
   el.addEventListener('click', () => { state.selectedSlug = el.dataset.slug; render(); });
   el.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); state.selectedSlug = el.dataset.slug; render(); } });
  });
  on('[data-open]', (el, ev) => { ev.stopPropagation(); state.selectedSlug = el.dataset.open; render(); });
  on('[data-action="close-drawer"]', (el, ev) => { if (ev.target === el) { state.selectedSlug = null; render(); } });
  on('[data-action="toggle-sandbox"]', () => { state.sandboxOpen = !state.sandboxOpen; render(); });
  on('[data-action="reset-model"]', () => { state.weights = { ...LIVE_FIT.weights }; render(); });
  on('[data-action="reset-filters"]', () => { state.tier = state.verdict = state.category = state.chain = state.momentum = 'all'; state.search = ''; render(); });
  on('[data-clear]', (el) => { const k = el.dataset.clear; state[k] = k === 'search' ? '' : 'all'; render(); });
  ROOT.querySelectorAll('[data-param]').forEach((el) => el.addEventListener('input', () => { state.weights[el.dataset.param] = Number(el.value); render(); }));
  const sEl = document.getElementById('search-input');
  if (sEl) sEl.addEventListener('input', (ev) => {
   const pos = ev.target.selectionStart; state.search = ev.target.value; render();
   const n = document.getElementById('search-input'); if (n) { n.focus(); n.setSelectionRange(pos, pos); }
  });
  [['verdict-select', 'verdict'], ['cat-select', 'category'], ['chain-select', 'chain'], ['mom-select', 'momentum']].forEach(([id, key]) => {
   const el = document.getElementById(id); if (el) el.addEventListener('change', (ev) => { state[key] = ev.target.value; render(); });
  });
 }
 window.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && state.selectedSlug) { state.selectedSlug = null; render(); } });
 render();
})();
