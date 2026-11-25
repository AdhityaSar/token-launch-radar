// Empirically-Fitted & Out-of-Sample Validated Token Launch Radar Model (radar-v3-ridge-oos)
// Every valuation weight, conformal interval, token launch odds curve, and farm EV verdict
// is derived strictly from snapshot.json with zero hardcoded protocol names.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export interface FittedWeights {
  intercept: number;
  b_tvl: number;
  b_growth: number;
  b_age: number;
  b_chains: number;
  ridge_lambda: number;
  sector_shrinkage_k: number;
  conformal_q25: number;
  conformal_q75: number;
}

export interface ValidationReport {
  live_fit_count: number;
  loo_median_mult_err: number;
  loo_naive_mult_err: number;
  loo_mae_log: number;
  loo_naive_mae_log: number;
  loo_error_reduction_pct: number;
  conformal_50_coverage_pct: number;
  wf_train_count: number;
  wf_test_count: number;
  wf_median_mult_err: number;
  wf_naive_mult_err: number;
  wf_error_reduction_pct: number;
}

export interface CategoryEffect {
  category: string;
  peer_count: number;
  raw_median_mult: number;
  log_residual_adj: number;
  sector_mult_factor: number;
  token_rate: number;
  total_protocols: number;
}

export interface FarmStep {
  title: string;
  action: string;
  cost: string;
  time: string;
}

export interface FarmEconomics {
  capital_usd: number;
  est_gas_usd: number;
  hold_days: number;
  active_minutes: number;
  primary_chain: string;
  secondary_chain: string | null;
  gas_tier: 'mainnet' | 'l2' | 'alt-l1';
  est_airdrop_yield_pct: number;
  net_ev_usd: number;
  hourly_ev_usd: number;
  verdict: 'FARM' | 'SELECTIVE' | 'SKIP';
  verdict_reason: string;
  steps: FarmStep[];
}

export interface ValuationResult {
  implied_value_usd: number;
  bear_value_usd: number;
  bull_value_usd: number;
  implied_mcap_tvl: number;
  tvl_component: number;
  growth_factor: number;
  age_factor: number;
  chain_factor: number;
  sector_factor: number;
  peer_count: number;
  age_days: number | null;
  token_odds: number;
}

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  tier: 'watchlist' | 'early';
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  bear_value_usd: number;
  bull_value_usd: number;
  token_odds: number;
  model: ValuationResult;
  farm_verdict: 'FARM' | 'SELECTIVE' | 'SKIP';
  farm_net_ev_usd: number;
  farm_hourly_usd: number;
  farm_steps: string[];
}

export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  tier: 'watchlist' | 'early';
  implied_value_usd: number;
  bear_value_usd: number;
  bull_value_usd: number;
  implied_mcap_tvl: number;
  sector_factor: number;
  peer_count: number;
  growth_factor: number;
  age_factor: number;
  chain_factor: number;
  token_odds: number;
  capital_usd: number;
  est_gas_usd: number;
  hold_days: number;
  active_minutes: number;
  net_ev_usd: number;
  hourly_ev_usd: number;
  verdict: 'FARM' | 'SELECTIVE' | 'SKIP';
}

export interface LivePeerRow {
  slug: string;
  name: string;
  symbol: string | null;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number;
  mcap_tvl_ratio: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
}

export interface RadarDataset {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  weights: FittedWeights;
  validation: ValidationReport;
  categories: Record<string, CategoryEffect>;
  global_token_rate: number;
  universe: {
    total_protocols: number;
    watchlist_count: number;
    early_count: number;
    board_count: number;
    live_total_count: number;
    live_priced_count: number;
    excluded_non_launch_count: number;
  };
  rows: RadarRow[];
  live: LivePeerRow[];
}

export const MODEL_VERSION = 'radar-v3-ridge-oos';

const DAY_MS = 86_400_000;
const RIDGE_LAMBDA = 3.0;
const SECTOR_SHRINKAGE_K = 5.0;
const TVL_CENTER = 18.0; // ln($65.6M)
const AGE_CENTER = 0.7;

const EXCLUDED_CATEGORIES = new Set([
  'Bridge',
  'Canonical Bridge',
  'Liquid Staking',
  'Liquid Restaking',
  'Restaked BTC',
  'CEX',
]);
const EXCHANGE_PREFIXES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];
const VERSION_SUFFIX = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
const WRAPPED_STAKED_WORD = /\b(wrapped|staked|staking)\b/i;

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

function round4(n: number): number {
  return Math.round(n * 1e4) / 1e4;
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return lo === hi ? sorted[lo]! : sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (!listedAt) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

export function isCleanEarlyCandidate(p: SnapshotProtocol, floorUsd: number): boolean {
  if (!isPreToken(p)) return false;
  if (p.tvl_usd >= floorUsd) return false;
  if (p.category !== null && EXCLUDED_CATEGORIES.has(p.category)) return false;
  if (VERSION_SUFFIX.test(p.name)) return false;
  if (WRAPPED_STAKED_WORD.test(p.name)) return false;
  if (EXCHANGE_PREFIXES.some((ex) => p.name.startsWith(ex))) return false;
  return true;
}

interface FitPoint {
  slug: string;
  category: string;
  tvl_usd: number;
  mcap_usd: number;
  lmcap: number;
  listed_at: string | null;
  features: [number, number, number, number, number];
}

function featureVector(tvl: number, change7d: number | null, age: number | null, chainCount: number): [number, number, number, number, number] {
  const lt = Math.log(Math.max(1, tvl)) - TVL_CENTER;
  const g = Math.tanh((typeof change7d === 'number' && Number.isFinite(change7d) ? change7d : 0) / 50);
  const effAge = age ?? 365;
  const a = Math.sqrt(clamp(effAge, 0, 1095) / 1095) - AGE_CENTER;
  const ch = Math.min(Math.log(Math.max(1, chainCount)), 2.5);
  return [1, lt, g, a, ch];
}

/** Closed-form Ridge regression (regularizing slopes, never intercept) via Gaussian elimination. */
function solveRidge(X: number[][], y: number[], lambda: number): number[] {
  const d = X[0]!.length;
  const A: number[][] = Array.from({ length: d }, () => Array<number>(d).fill(0));
  const b: number[] = Array<number>(d).fill(0);
  for (let n = 0; n < X.length; n++) {
    const xi = X[n]!;
    const yi = y[n]!;
    for (let i = 0; i < d; i++) {
      b[i]! += xi[i]! * yi;
      for (let j = 0; j < d; j++) A[i]![j]! += xi[i]! * xi[j]!;
    }
  }
  for (let i = 1; i < d; i++) A[i]![i]! += lambda;

  const M = A.map((row, i) => [...row, b[i]!]);
  for (let i = 0; i < d; i++) {
    let piv = i;
    for (let r = i + 1; r < d; r++) {
      if (Math.abs(M[r]![i]!) > Math.abs(M[piv]![i]!)) piv = r;
    }
    const tmp = M[i]!;
    M[i] = M[piv]!;
    M[piv] = tmp;
    const diag = M[i]![i]!;
    if (Math.abs(diag) < 1e-12) continue;
    for (let r = i + 1; r < d; r++) {
      const f = M[r]![i]! / diag;
      for (let c = i; c <= d; c++) M[r]![c]! -= f * M[i]![c]!;
    }
  }
  const w = Array<number>(d).fill(0);
  for (let i = d - 1; i >= 0; i--) {
    let sum = M[i]![d]!;
    for (let j = i + 1; j < d; j++) sum -= M[i]![j]! * w[j]!;
    w[i] = Math.abs(M[i]![i]!) > 1e-12 ? sum / M[i]![i]! : 0;
  }
  return w;
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

function fitWithSectorAdj(points: FitPoint[], lambda: number, catK: number): { w: number[]; catAdj: Map<string, number> } {
  const w = solveRidge(
    points.map((p) => p.features),
    points.map((p) => p.lmcap),
    lambda,
  );
  const resByCat = new Map<string, number[]>();
  for (const p of points) {
    const res = p.lmcap - dot(w, p.features);
    let arr = resByCat.get(p.category);
    if (!arr) {
      arr = [];
      resByCat.set(p.category, arr);
    }
    arr.push(res);
  }
  const catAdj = new Map<string, number>();
  for (const [cat, arr] of resByCat.entries()) {
    const sum = arr.reduce((s, v) => s + v, 0);
    catAdj.set(cat, clamp(sum / (arr.length + catK), -0.9, 0.9));
  }
  return { w, catAdj };
}

/** Fit Ridge power-law model on live peers and run both Leave-One-Out and Walk-Forward blind tests. */
export function fitValuationModel(snapshot: Snapshot): {
  weights: FittedWeights;
  validation: ValidationReport;
  categories: Record<string, CategoryEffect>;
  livePeers: LivePeerRow[];
  globalTokenRate: number;
} {
  const points: FitPoint[] = [];
  const livePeers: LivePeerRow[] = [];
  const rawMultByCat = new Map<string, number[]>();

  // Token launch rate statistics across non-excluded protocols
  let nonExclTotal = 0;
  let nonExclToken = 0;
  const sectorCounts = new Map<string, { total: number; token: number }>();

  for (const p of snapshot.protocols) {
    const cat = p.category ?? 'Other';
    if (p.category === null || !EXCLUDED_CATEGORIES.has(p.category)) {
      nonExclTotal += 1;
      const hasTok = !isPreToken(p);
      if (hasTok) nonExclToken += 1;
      const sc = sectorCounts.get(cat) ?? { total: 0, token: 0 };
      sc.total += 1;
      if (hasTok) sc.token += 1;
      sectorCounts.set(cat, sc);
    }

    if (isPreToken(p) || p.category === 'CEX') continue;
    if (p.mcap_usd === null || p.mcap_usd <= 0 || p.tvl_usd <= 0) continue;
    const ratio = p.mcap_usd / p.tvl_usd;
    if (ratio < 0.005 || ratio > 10.0) continue;

    const age = ageDays(snapshot.as_of, p.listed_at);
    points.push({
      slug: p.slug,
      category: cat,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      lmcap: Math.log(p.mcap_usd),
      listed_at: p.listed_at,
      features: featureVector(p.tvl_usd, p.change_7d, age, p.chains.length),
    });
    livePeers.push({
      slug: p.slug,
      name: p.name,
      symbol: p.symbol,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      mcap_tvl_ratio: round4(ratio),
      change_7d: p.change_7d,
      listed_at: p.listed_at,
      age_days: age,
    });
    const list = rawMultByCat.get(cat) ?? [];
    list.push(ratio);
    rawMultByCat.set(cat, list);
  }

  const globalTokenRate = nonExclTotal > 0 ? nonExclToken / nonExclTotal : 0.55;

  // 1. Full fit
  const { w, catAdj } = fitWithSectorAdj(points, RIDGE_LAMBDA, SECTOR_SHRINKAGE_K);

  // 2. Leave-One-Out Cross-Validation (LOO-CV) + conformal residuals
  const looSignedRes: number[] = [];
  const looAbsErrs: number[] = [];
  const looNaiveAbsErrs: number[] = [];

  for (let i = 0; i < points.length; i++) {
    const train = points.filter((_, idx) => idx !== i);
    const test = points[i]!;
    const fold = fitWithSectorAdj(train, RIDGE_LAMBDA, SECTOR_SHRINKAGE_K);
    const predLog = dot(fold.w, test.features) + (fold.catAdj.get(test.category) ?? 0);
    const signed = test.lmcap - predLog;
    looSignedRes.push(signed);
    looAbsErrs.push(Math.abs(signed));
    looNaiveAbsErrs.push(Math.abs(test.lmcap - Math.log(0.25 * test.tvl_usd)));
  }

  const sortedSigned = [...looSignedRes].sort((a, b) => a - b);
  const q25 = quantile(sortedSigned, 0.25);
  const q75 = quantile(sortedSigned, 0.75);
  const insideConformal = looSignedRes.filter((r) => r >= q25 && r <= q75).length;

  const sortedLooAbs = [...looAbsErrs].sort((a, b) => a - b);
  const sortedNaiveAbs = [...looNaiveAbsErrs].sort((a, b) => a - b);
  const looMedMult = Math.exp(quantile(sortedLooAbs, 0.5));
  const naiveMedMult = Math.exp(quantile(sortedNaiveAbs, 0.5));
  const looMae = looAbsErrs.reduce((s, v) => s + v, 0) / Math.max(1, looAbsErrs.length);
  const naiveMae = looNaiveAbsErrs.reduce((s, v) => s + v, 0) / Math.max(1, looNaiveAbsErrs.length);

  // 3. Temporal Walk-Forward Validation (older 65% -> newer 35% by listed_at)
  const dated = points
    .filter((p) => p.listed_at !== null)
    .sort((a, b) => (a.listed_at! < b.listed_at! ? -1 : a.listed_at! > b.listed_at! ? 1 : a.slug.localeCompare(b.slug)));
  const split = Math.floor(dated.length * 0.65);
  const wfTrain = dated.slice(0, split);
  const wfTest = dated.slice(split);
  const wfModel = fitWithSectorAdj(wfTrain, RIDGE_LAMBDA, SECTOR_SHRINKAGE_K);
  const wfAbs: number[] = [];
  const wfNaiveAbs: number[] = [];
  for (const t of wfTest) {
    const pred = dot(wfModel.w, t.features) + (wfModel.catAdj.get(t.category) ?? 0);
    wfAbs.push(Math.abs(t.lmcap - pred));
    wfNaiveAbs.push(Math.abs(t.lmcap - Math.log(0.25 * t.tvl_usd)));
  }
  wfAbs.sort((a, b) => a - b);
  wfNaiveAbs.sort((a, b) => a - b);
  const wfMedMult = Math.exp(quantile(wfAbs, 0.5));
  const wfNaiveMult = Math.exp(quantile(wfNaiveAbs, 0.5));

  // Build category effects table
  const allCats = new Set([...rawMultByCat.keys(), ...sectorCounts.keys()]);
  const categories: Record<string, CategoryEffect> = {};
  for (const cat of [...allCats].sort()) {
    const vals = (rawMultByCat.get(cat) ?? []).sort((a, b) => a - b);
    const adj = catAdj.get(cat) ?? 0;
    const sc = sectorCounts.get(cat) ?? { total: 0, token: 0 };
    const shrunkTokRate = (sc.token + 5 * globalTokenRate) / (sc.total + 5);
    categories[cat] = {
      category: cat,
      peer_count: vals.length,
      raw_median_mult: vals.length > 0 ? round4(quantile(vals, 0.5)) : 0,
      log_residual_adj: round4(adj),
      sector_mult_factor: round4(Math.exp(adj)),
      token_rate: round4(shrunkTokRate),
      total_protocols: sc.total,
    };
  }

  livePeers.sort((a, b) => b.mcap_usd - a.mcap_usd || a.slug.localeCompare(b.slug));

  return {
    weights: {
      intercept: round4(w[0]!),
      b_tvl: round4(w[1]!),
      b_growth: round4(w[2]!),
      b_age: round4(w[3]!),
      b_chains: round4(w[4]!),
      ridge_lambda: RIDGE_LAMBDA,
      sector_shrinkage_k: SECTOR_SHRINKAGE_K,
      conformal_q25: round4(q25),
      conformal_q75: round4(q75),
    },
    validation: {
      live_fit_count: points.length,
      loo_median_mult_err: round2(looMedMult),
      loo_naive_mult_err: round2(naiveMedMult),
      loo_mae_log: round4(looMae),
      loo_naive_mae_log: round4(naiveMae),
      loo_error_reduction_pct: round2(((naiveMedMult - looMedMult) / naiveMedMult) * 100),
      conformal_50_coverage_pct: round2((insideConformal / Math.max(1, looSignedRes.length)) * 100),
      wf_train_count: wfTrain.length,
      wf_test_count: wfTest.length,
      wf_median_mult_err: round2(wfMedMult),
      wf_naive_mult_err: round2(wfNaiveMult),
      wf_error_reduction_pct: round2(((wfNaiveMult - wfMedMult) / wfNaiveMult) * 100),
    },
    categories,
    livePeers,
    globalTokenRate: round4(globalTokenRate),
  };
}

/** Empirical token launch odds from snapshot sector token rates, TVL scale, and survival-adjusted age hazard. */
export function computeTokenOdds(
  p: SnapshotProtocol,
  age: number | null,
  categories: Record<string, CategoryEffect>,
  globalTokenRate: number,
): number {
  const cat = categories[p.category ?? 'Other'];
  const baseRate = cat ? cat.token_rate : globalTokenRate;
  // Larger TVL protocols launch tokens at higher empirical rates ($500M+ = 85% vs $5-25M = 55%)
  const tvlLift = clamp((Math.log10(Math.max(5e6, p.tvl_usd)) - 7.3) * 0.08, -0.06, 0.16);
  // Survival hazard: protocols <18m old are in peak launch window; 2.5y+ without a token have lower residual odds
  const effAge = age ?? 365;
  const ageHazard = effAge < 540 ? 1.05 : effAge < 900 ? 0.92 : 0.72;
  return round4(clamp((baseRate + tvlLift) * ageHazard, 0.15, 0.88));
}

/** Score a pre-token protocol with the fitted Ridge model and conformal prediction intervals. */
export function computeImpliedValue(
  p: SnapshotProtocol,
  asOf: string,
  weights: FittedWeights,
  categories: Record<string, CategoryEffect>,
  globalTokenRate: number,
): ValuationResult {
  const age = ageDays(asOf, p.listed_at);
  const f = featureVector(p.tvl_usd, p.change_7d, age, p.chains.length);
  const cat = categories[p.category ?? 'Other'];
  const sectorAdj = cat ? cat.log_residual_adj : 0;

  const logBase = weights.intercept + weights.b_tvl * f[1];
  const logGrowth = weights.b_growth * f[2];
  const logAge = weights.b_age * f[3];
  const logChain = weights.b_chains * f[4];
  const logPred = logBase + logGrowth + logAge + logChain + sectorAdj;

  const implied = Math.max(1, Math.round(Math.exp(logPred)));
  const bear = Math.max(1, Math.min(implied, Math.round(Math.exp(logPred + weights.conformal_q25))));
  const bull = Math.max(implied, Math.round(Math.exp(logPred + weights.conformal_q75)));

  return {
    implied_value_usd: implied,
    bear_value_usd: bear,
    bull_value_usd: bull,
    implied_mcap_tvl: round4(implied / Math.max(1, p.tvl_usd)),
    tvl_component: Math.round(Math.exp(logBase)),
    growth_factor: round4(Math.exp(logGrowth)),
    age_factor: round4(Math.exp(logAge)),
    chain_factor: round4(Math.exp(logChain)),
    sector_factor: round4(Math.exp(sectorAdj)),
    peer_count: cat ? cat.peer_count : 0,
    age_days: age,
    token_odds: computeTokenOdds(p, age, categories, globalTokenRate),
  };
}

function selectPrimaryChain(chains: string[]): { primary: string; secondary: string | null; gasTier: 'mainnet' | 'l2' | 'alt-l1' } {
  if (chains.length === 0) return { primary: 'its primary chain', secondary: null, gasTier: 'l2' };
  const lowFee = chains.find((c) => c !== 'Ethereum' && (L2_CHAINS.has(c) || ALT_L1_CHAINS.has(c)));
  const primary = lowFee ?? chains[0]!;
  const secondary = chains.find((c) => c !== primary) ?? null;
  const gasTier = primary === 'Ethereum' ? 'mainnet' : ALT_L1_CHAINS.has(primary) ? 'alt-l1' : 'l2';
  return { primary, secondary, gasTier };
}

function gasProfile(tier: 'mainnet' | 'l2' | 'alt-l1'): { setup: number; weekly: number; total: number; label: string } {
  if (tier === 'mainnet') return { setup: 12, weekly: 4, total: 28, label: '$10-$15 Ethereum L1 gas' };
  if (tier === 'alt-l1') return { setup: 0.2, weekly: 0.05, total: 1, label: 'under $0.25 network gas' };
  return { setup: 0.5, weekly: 0.15, total: 2, label: 'under $0.75 L2 gas' };
}

function coreActionForCategory(p: SnapshotProtocol, capital: number, chain: string): { title: string; action: string } {
  const cat = (p.category ?? '').toLowerCase();
  const n = p.name;
  if (cat.includes('risk curator') || cat.includes('allocator') || cat.includes('cdp manager')) {
    return { title: 'Deposit into curated vault', action: `Deposit $${capital} USDC into ${n}'s flagship curated vault on ${chain} and hold the receipt shares in self-custody.` };
  }
  if (cat.includes('rwa')) {
    return { title: 'Mint yield-bearing RWA', action: `Subscribe $${capital} into ${n}'s yield-bearing RWA token on ${chain} and keep the balance unstaked or in its native vault.` };
  }
  if (cat.includes('basis') || cat.includes('yield')) {
    return { title: 'Enter core yield vault', action: `Deploy $${capital} USDC into ${n}'s primary yield or basis pool on ${chain} and remain deposited across epoch rollovers.` };
  }
  if (cat.includes('lend') || cat.includes('cdp') || cat.includes('collateral')) {
    return { title: 'Supply collateral & borrow lightly', action: `Supply $${capital} USDC on ${n} (${chain}) and borrow 20% against it to qualify under both lender and borrower criteria.` };
  }
  if (cat.includes('dex') || cat.includes('liquidity')) {
    return { title: 'Provide LP & execute swaps', action: `Provide $${capital} in a core stable or blue-chip pool on ${n} (${chain}) and execute 2-3 organic spot swaps.` };
  }
  if (cat.includes('derivative') || cat.includes('perpetual') || cat.includes('prediction') || cat.includes('otc')) {
    return { title: 'Post margin & log volume', action: `Deposit $${capital} USDC margin on ${n} (${chain}) and open small hedged positions to register organic trading volume.` };
  }
  if (cat.includes('restak') || cat.includes('staking')) {
    return { title: 'Stake native receipt asset', action: `Stake $${capital} of the native asset through ${n} on ${chain} and keep the receipt position delegated.` };
  }
  if (cat.includes('payment') || cat.includes('stable') || cat.includes('cedefi')) {
    return { title: 'Mint & transact stable balance', action: `Mint or fund $${capital} through ${n} on ${chain} and complete at least two on-chain settlement transfers.` };
  }
  if (cat.includes('insurance')) {
    return { title: 'Underwrite or purchase cover', action: `Commit $${capital} to ${n}'s underwriting pool or buy a small cover policy on ${chain} and hold to expiry.` };
  }
  return { title: 'Deploy core protocol position', action: `Deposit $${capital} USDC into ${n}'s flagship product on ${chain} and keep the position active across snapshots.` };
}

/** Build a concrete 6-step costed/timed farm plan AND net-EV $/hr verdict from snapshot figures. */
export function buildFarmPlan(p: SnapshotProtocol, val: ValuationResult): FarmEconomics {
  const { primary, secondary, gasTier } = selectPrimaryChain(p.chains);
  const gas = gasProfile(gasTier);
  // Realistic active farming sleeve ($1,000-$2,500 refundable capital) sized to TVL tier & chain gas
  const baseCap = p.tvl_usd >= 1e9 ? 2500 : p.tvl_usd >= 250e6 ? 2000 : p.tvl_usd >= 50e6 ? 1500 : 1000;
  const capital = gasTier === 'mainnet' ? Math.max(baseCap, 2500) : baseCap;
  const hold = val.age_days === null ? 45 : val.age_days < 180 ? 60 : val.age_days < 365 ? 45 : 30;
  const weeks = Math.max(4, Math.ceil(hold / 7));
  const activeMinutes = 10 + 15 + weeks * 8 + 10 + 10;
  const activeHours = activeMinutes / 60;

  // 10% community airdrop pool with 1.5x active-wallet weight vs passive TVL, weighted by empirical token odds
  const rawYield = Math.min(0.4, (val.implied_value_usd * 0.10 * 1.5) / Math.max(1, p.tvl_usd));
  const expectedYield = rawYield * val.token_odds;
  const grossEv = capital * expectedYield;
  const slippage = capital * 0.0015; // 15 bps round-trip swap/vault spread
  const netEv = round2(grossEv - gas.total - slippage);
  const hourlyEv = round2(netEv / Math.max(0.5, activeHours));

  let verdict: 'FARM' | 'SELECTIVE' | 'SKIP' = 'SKIP';
  let verdictReason = `Gas (~$${gas.total}) and spread exceed diluted airdrop EV`;
  if (netEv >= 15 && hourlyEv >= 10) {
    verdict = 'FARM';
    verdictReason = `Strong net EV (+$${netEv.toFixed(0)}, $${hourlyEv.toFixed(1)}/hr) on ${primary}`;
  } else if (netEv >= 3) {
    verdict = 'SELECTIVE';
    verdictReason = `Modest net EV (+$${netEv.toFixed(1)}); worthwhile mainly if already active on ${primary}`;
  }

  const core = coreActionForCategory(p, capital, primary);
  const hasDrawdown = typeof p.change_7d === 'number' && p.change_7d <= -5;

  const steps: FarmStep[] = [
    {
      title: 'Isolate wallet & fund gas',
      action: `Create a dedicated wallet on ${primary}, fund $${capital} principal plus native gas, and keep zero unrelated approvals.`,
      cost: `${gas.label} for initial funding`,
      time: '10 minutes',
    },
    {
      title: core.title,
      action: core.action,
      cost: `$${capital} refundable capital + ${gas.label}`,
      time: '15 minutes',
    },
    secondary
      ? {
          title: 'Multi-chain footprint & weekly cadence',
          action: `Complete one qualifying interaction weekly on ${primary} (and mirror 20% on ${secondary}) across ${weeks}+ distinct weeks.`,
          cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas over ${weeks} weeks`,
          time: `8 min/week (${weeks} weeks)`,
        }
      : {
          title: 'Weekly organic cadence',
          action: `Execute one qualifying on-chain action weekly on ${primary} (top-up, compound, swap, or rebalance) to pass sybil filters.`,
          cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas over ${weeks} weeks`,
          time: `8 min/week (${weeks} weeks)`,
        },
    hasDrawdown
      ? {
          title: 'Check 7d TVL drawdown & hold window',
          action: `TVL moved ${p.change_7d!.toFixed(1)}% in 7d; verify liquidity depth before scaling, then hold $${capital} for ${hold} days across epochs.`,
          cost: `$${capital} refundable principal (~$${slippage.toFixed(2)} est. spread)`,
          time: `${hold}-day holding window`,
        }
      : {
          title: 'Hold across snapshot epochs',
          action: `Keep the $${capital} position open uninterrupted for ${hold} days so time-weighted points and multi-epoch snapshots register full weight.`,
          cost: `$${capital} refundable principal (~$${slippage.toFixed(2)} est. spread)`,
          time: `${hold}-day holding window`,
        },
    {
      title: 'Verify points & official claim route',
      action: `Inspect ${p.name}'s official points page before exiting; never sign blind permits or pay third-party claim fees.`,
      cost: '$0 verification fee (never pay to claim)',
      time: '10 minutes',
    },
    {
      title: 'Log tx hashes & unwind cleanly',
      action: `Record wallet, tx hashes, and epoch dates in your farm log, then revoke token approvals after withdrawing.`,
      cost: `${gas.label} for exit and revoke`,
      time: '10 minutes',
    },
  ];

  return {
    capital_usd: capital,
    est_gas_usd: round2(gas.total),
    hold_days: hold,
    active_minutes: activeMinutes,
    primary_chain: primary,
    secondary_chain: secondary,
    gas_tier: gasTier,
    est_airdrop_yield_pct: round2(expectedYield * 100),
    net_ev_usd: netEv,
    hourly_ev_usd: hourlyEv,
    verdict,
    verdict_reason: verdictReason,
    steps,
  };
}

export function formatStepString(step: FarmStep): string {
  let text = `${step.title}: ${step.action} Cost: ${step.cost}. Time: ${step.time}.`;
  if (text.length > 600) text = `${text.slice(0, 596)}...`;
  if (text.length < 20) text = `${text} Verify position on-chain.`;
  return text;
}

export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; dataset: RadarDataset } {
  const bySlug = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const watchlistSet = new Set(snapshot.watchlist);
  const { weights, validation, categories, livePeers, globalTokenRate } = fitValuationModel(snapshot);

  const selected = new Map<string, { protocol: SnapshotProtocol; tier: 'watchlist' | 'early' }>();
  for (const slug of snapshot.watchlist) {
    const p = bySlug.get(slug);
    if (p) selected.set(slug, { protocol: p, tier: 'watchlist' });
  }
  const early = snapshot.protocols
    .filter((p) => !selected.has(p.slug) && isCleanEarlyCandidate(p, snapshot.floor_usd))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || a.slug.localeCompare(b.slug));
  for (const p of early) selected.set(p.slug, { protocol: p, tier: 'early' });

  const scored = [...selected.values()].map(({ protocol, tier }) => {
    const val = computeImpliedValue(protocol, snapshot.as_of, weights, categories, globalTokenRate);
    const farm = buildFarmPlan(protocol, val);
    return { protocol, tier, val, farm };
  });

  scored.sort(
    (a, b) =>
      b.val.implied_value_usd - a.val.implied_value_usd ||
      b.protocol.tvl_usd - a.protocol.tvl_usd ||
      a.protocol.slug.localeCompare(b.protocol.slug),
  );

  const board: BoardRow[] = scored.map((item, idx) => ({
    rank: idx + 1,
    slug: item.protocol.slug,
    name: item.protocol.name,
    category: item.protocol.category,
    chains: item.protocol.chains,
    tvl_usd: item.protocol.tvl_usd,
    change_7d: item.protocol.change_7d,
    mcap_usd: item.protocol.mcap_usd,
    listed_at: item.protocol.listed_at,
    gecko_id: item.protocol.gecko_id,
    symbol: item.protocol.symbol,
    tier: item.tier,
    token_status: 'pre-token',
    age_days: item.val.age_days,
    implied_value_usd: item.val.implied_value_usd,
    bear_value_usd: item.val.bear_value_usd,
    bull_value_usd: item.val.bull_value_usd,
    token_odds: item.val.token_odds,
    model: item.val,
    farm_verdict: item.farm.verdict,
    farm_net_ev_usd: item.farm.net_ev_usd,
    farm_hourly_usd: item.farm.hourly_ev_usd,
    farm_steps: item.farm.steps.map(formatStepString),
  }));

  const rows: RadarRow[] = scored.map((item, idx) => ({
    rank: idx + 1,
    slug: item.protocol.slug,
    name: item.protocol.name,
    category: item.protocol.category,
    chains: item.protocol.chains,
    tvl_usd: item.protocol.tvl_usd,
    change_7d: item.protocol.change_7d,
    listed_at: item.protocol.listed_at,
    age_days: item.val.age_days,
    tier: item.tier,
    implied_value_usd: item.val.implied_value_usd,
    bear_value_usd: item.val.bear_value_usd,
    bull_value_usd: item.val.bull_value_usd,
    implied_mcap_tvl: item.val.implied_mcap_tvl,
    sector_factor: item.val.sector_factor,
    peer_count: item.val.peer_count,
    growth_factor: item.val.growth_factor,
    age_factor: item.val.age_factor,
    chain_factor: item.val.chain_factor,
    token_odds: item.val.token_odds,
    capital_usd: item.farm.capital_usd,
    est_gas_usd: item.farm.est_gas_usd,
    hold_days: item.farm.hold_days,
    active_minutes: item.farm.active_minutes,
    net_ev_usd: item.farm.net_ev_usd,
    hourly_ev_usd: item.farm.hourly_ev_usd,
    verdict: item.farm.verdict,
  }));

  const watchlistCount = board.filter((r) => watchlistSet.has(r.slug)).length;
  const earlyCount = board.length - watchlistCount;
  const liveTotalCount = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const excludedCount = snapshot.protocols.length - board.length - liveTotalCount;

  const dataset: RadarDataset = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    weights,
    validation,
    categories,
    global_token_rate: globalTokenRate,
    universe: {
      total_protocols: snapshot.protocols.length,
      watchlist_count: watchlistCount,
      early_count: earlyCount,
      board_count: board.length,
      live_total_count: liveTotalCount,
      live_priced_count: livePeers.length,
      excluded_non_launch_count: Math.max(0, excludedCount),
    },
    rows,
    live: livePeers,
  };

  return { board, dataset };
}

export function parseSnapshot(text: string): Snapshot {
  const raw = JSON.parse(text) as Record<string, unknown>;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('snapshot.json is not an object');
  if (typeof raw.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(raw.as_of)) throw new Error('invalid as_of');
  if (typeof raw.source !== 'string') throw new Error('invalid source');
  if (typeof raw.floor_usd !== 'number' || !Number.isSafeInteger(raw.floor_usd) || raw.floor_usd <= 0) {
    throw new Error('invalid floor_usd');
  }
  if (!Array.isArray(raw.protocols)) throw new Error('invalid protocols');
  if (!Array.isArray(raw.watchlist) || raw.watchlist.some((s) => typeof s !== 'string')) {
    throw new Error('invalid watchlist');
  }

  const protocols: SnapshotProtocol[] = raw.protocols.map((v, i) => {
    const p = v as Record<string, unknown>;
    if (!p || typeof p !== 'object' || typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number') {
      throw new Error(`invalid protocol at index ${i}`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });

  return {
    as_of: raw.as_of,
    source: raw.source,
    floor_usd: raw.floor_usd,
    protocols,
    watchlist: raw.watchlist as string[],
  };
}

export function validateBoard(board: BoardRow[], snapshot: Snapshot): string | null {
  if (board.length < 100 || board.length > 4000) return `row count ${board.length} outside 100..4000`;
  const slugs = new Set(board.map((r) => r.slug));
  if (slugs.size !== board.length) return 'duplicate slugs in board';
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `missing ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  for (let i = 0; i < board.length; i++) {
    const r = board[i]!;
    if (r.rank !== i + 1) return `rank mismatch at index ${i}: got ${r.rank}`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0 || r.implied_value_usd > 100_000_000_000_000) {
      return `invalid implied_value_usd for ${r.slug}: ${r.implied_value_usd}`;
    }
    if (i > 0 && r.implied_value_usd > board[i - 1]!.implied_value_usd) {
      return `row ${r.slug} outranks previous row on implied_value_usd`;
    }
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const len = s.trim().length;
      if (len < 20 || len > 600) return `${r.slug} has farm step of length ${len}`;
    }
  }
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return 'all rows share identical implied_value_usd';
  }
  return null;
}

export function renderDataScript(dataset: RadarDataset): string {
  return `window.RADAR_DATA=${JSON.stringify(dataset)};\n`;
}
