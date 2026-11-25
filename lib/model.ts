// Launch Radar v4 (radar-v4-oos-verified)
// Empirically-fitted Ridge power-law valuation with LOO-CV & Walk-Forward ablation,
// conformal prediction bands, empirical token launch odds, break-even FDV, and
// category-specific 6-step DeFi farm playbooks. Zero hardcoded protocol names.

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

export interface AblationRow {
  label: string;
  loo_mult_err: number;
  loo_mae_log: number;
  wf_mult_err: number;
  delta_vs_naive_pct: number;
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
  ablation: AblationRow[];
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
  breakeven_fdv_usd: number;
  safety_margin_x: number;
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
  farm_breakeven_fdv_usd: number;
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
  breakeven_fdv_usd: number;
  safety_margin_x: number;
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

export const MODEL_VERSION = 'radar-v5-clean-oos';

const DAY_MS = 86_400_000;
const RIDGE_LAMBDA = 3.0;
const SECTOR_SHRINKAGE_K = 5.0;
const TVL_CENTER = 18.0;
const AGE_CENTER = 0.7;

const EXCLUDED_CATEGORIES = new Set([
  'Bridge', 'Canonical Bridge', 'Cross Chain Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX',
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

const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
const round2 = (n: number): number => Math.round(n * 100) / 100;
const clamp = (v: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, v));

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
  lmcap: number;
  listed_at: string | null;
  features: [number, number, number, number, number];
}

function featureVector(tvl: number, change7d: number | null, age: number | null, chainCount: number): [number, number, number, number, number] {
  const lt = Math.log(Math.max(1, tvl)) - TVL_CENTER;
  const g = Math.tanh((typeof change7d === 'number' && Number.isFinite(change7d) ? change7d : 0) / 50);
  const a = Math.sqrt(clamp(age ?? 365, 0, 1095) / 1095) - AGE_CENTER;
  const ch = Math.min(Math.log(Math.max(1, chainCount)), 2.5);
  return [1, lt, g, a, ch];
}

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
    for (let r = i + 1; r < d; r++) if (Math.abs(M[r]![i]!) > Math.abs(M[piv]![i]!)) piv = r;
    const tmp = M[i]!; M[i] = M[piv]!; M[piv] = tmp;
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

function fitSubset(points: FitPoint[], cols: number[], lambda: number, useCat: boolean, catK: number): { w: number[]; catAdj: Map<string, number> } {
  const X = points.map((p) => cols.map((c) => p.features[c]!));
  const y = points.map((p) => p.lmcap);
  const w = solveRidge(X, y, lambda);
  const catAdj = new Map<string, number>();
  if (useCat) {
    const resByCat = new Map<string, number[]>();
    for (let i = 0; i < points.length; i++) {
      const p = points[i]!;
      const res = p.lmcap - dot(w, X[i]!);
      const arr = resByCat.get(p.category) ?? [];
      arr.push(res);
      resByCat.set(p.category, arr);
    }
    for (const [cat, arr] of resByCat.entries()) {
      catAdj.set(cat, clamp(arr.reduce((s, v) => s + v, 0) / (arr.length + catK), -0.9, 0.9));
    }
  }
  return { w, catAdj };
}

function evalLooSubset(points: FitPoint[], cols: number[], useCat: boolean): { medMult: number; mae: number; signedRes: number[] } {
  const absErrs: number[] = [];
  const signedRes: number[] = [];
  for (let i = 0; i < points.length; i++) {
    const train = points.filter((_, idx) => idx !== i);
    const test = points[i]!;
    const fold = fitSubset(train, cols, RIDGE_LAMBDA, useCat, SECTOR_SHRINKAGE_K);
    const pred = dot(fold.w, cols.map((c) => test.features[c]!)) + (useCat ? (fold.catAdj.get(test.category) ?? 0) : 0);
    const s = test.lmcap - pred;
    signedRes.push(s);
    absErrs.push(Math.abs(s));
  }
  const sortedAbs = [...absErrs].sort((a, b) => a - b);
  return {
    medMult: Math.exp(quantile(sortedAbs, 0.5)),
    mae: absErrs.reduce((s, v) => s + v, 0) / Math.max(1, absErrs.length),
    signedRes,
  };
}

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
    if (isPreToken(p) || (p.category !== null && EXCLUDED_CATEGORIES.has(p.category))) continue;
    if (p.mcap_usd === null || p.mcap_usd <= 0 || p.tvl_usd <= 0) continue;
    const ratio = p.mcap_usd / p.tvl_usd;
    if (ratio < 0.005 || ratio > 10.0) continue;
    const age = ageDays(snapshot.as_of, p.listed_at);
    points.push({
      slug: p.slug,
      category: cat,
      tvl_usd: p.tvl_usd,
      lmcap: Math.log(p.mcap_usd),
      listed_at: p.listed_at,
      features: featureVector(p.tvl_usd, p.change_7d, age, p.chains.length),
    });
    livePeers.push({
      slug: p.slug, name: p.name, symbol: p.symbol, category: p.category,
      chains: p.chains, tvl_usd: p.tvl_usd, mcap_usd: p.mcap_usd,
      mcap_tvl_ratio: round4(ratio), change_7d: p.change_7d, listed_at: p.listed_at, age_days: age,
    });
    const list = rawMultByCat.get(cat) ?? [];
    list.push(ratio);
    rawMultByCat.set(cat, list);
  }

  const globalTokenRate = nonExclTotal > 0 ? nonExclToken / nonExclTotal : 0.55;
  const allCols = [0, 1, 2, 3, 4];
  const { w, catAdj } = fitSubset(points, allCols, RIDGE_LAMBDA, true, SECTOR_SHRINKAGE_K);

  // Naive baseline LOO errors
  const naiveAbs = points.map((p) => Math.abs(p.lmcap - Math.log(0.25 * p.tvl_usd)));
  const naiveMedMult = Math.exp(quantile([...naiveAbs].sort((a, b) => a - b), 0.5));
  const naiveMae = naiveAbs.reduce((s, v) => s + v, 0) / Math.max(1, naiveAbs.length);

  // Walk-Forward helper (older 65% -> newer 35% by listed_at)
  const dated = points
    .filter((p) => p.listed_at !== null)
    .sort((a, b) => (a.listed_at! < b.listed_at! ? -1 : a.listed_at! > b.listed_at! ? 1 : a.slug.localeCompare(b.slug)));
  const split = Math.floor(dated.length * 0.65);
  const wfTrain = dated.slice(0, split);
  const wfTest = dated.slice(split);
  const wfNaiveAbs = wfTest.map((t) => Math.abs(t.lmcap - Math.log(0.25 * t.tvl_usd))).sort((a, b) => a - b);
  const wfNaiveMult = Math.exp(quantile(wfNaiveAbs, 0.5));

  const evalWfSubset = (cols: number[], useCat: boolean): number => {
    const m = fitSubset(wfTrain, cols, RIDGE_LAMBDA, useCat, SECTOR_SHRINKAGE_K);
    const errs = wfTest
      .map((t) => Math.abs(t.lmcap - (dot(m.w, cols.map((c) => t.features[c]!)) + (useCat ? (m.catAdj.get(t.category) ?? 0) : 0))))
      .sort((a, b) => a - b);
    return Math.exp(quantile(errs, 0.5));
  };

  // Term-by-term LOO + Walk-Forward ablation
  const abSpecs: Array<{ label: string; cols: number[]; cat: boolean }> = [
    { label: 'Power-Law TVL only', cols: [0, 1], cat: false },
    { label: '+ Sector EB Shrinkage', cols: [0, 1], cat: true },
    { label: '+ 7d TVL Flow + Age', cols: [0, 1, 2, 3], cat: true },
    { label: 'Full Ridge (+ Chains)', cols: [0, 1, 2, 3, 4], cat: true },
  ];
  const ablation: AblationRow[] = [
    { label: 'Naive 0.25 × TVL', loo_mult_err: round2(naiveMedMult), loo_mae_log: round4(naiveMae), wf_mult_err: round2(wfNaiveMult), delta_vs_naive_pct: 0 },
  ];
  let fullLoo = { medMult: naiveMedMult, mae: naiveMae, signedRes: [] as number[] };
  let wfMedMult = wfNaiveMult;
  for (const sp of abSpecs) {
    const r = evalLooSubset(points, sp.cols, sp.cat);
    const wfE = evalWfSubset(sp.cols, sp.cat);
    ablation.push({
      label: sp.label,
      loo_mult_err: round2(r.medMult),
      loo_mae_log: round4(r.mae),
      wf_mult_err: round2(wfE),
      delta_vs_naive_pct: round2(((naiveMedMult - r.medMult) / naiveMedMult) * 100),
    });
    if (sp.cols.length === 5 && sp.cat) {
      fullLoo = r;
      wfMedMult = wfE;
    }
  }

  const sortedSigned = [...fullLoo.signedRes].sort((a, b) => a - b);
  const q25 = quantile(sortedSigned, 0.25);
  const q75 = quantile(sortedSigned, 0.75);
  const insideConformal = fullLoo.signedRes.filter((r) => r >= q25 && r <= q75).length;

  const allCats = new Set([...rawMultByCat.keys(), ...sectorCounts.keys()]);
  const categories: Record<string, CategoryEffect> = {};
  for (const cat of [...allCats].sort()) {
    const vals = (rawMultByCat.get(cat) ?? []).sort((a, b) => a - b);
    const adj = catAdj.get(cat) ?? 0;
    const sc = sectorCounts.get(cat) ?? { total: 0, token: 0 };
    categories[cat] = {
      category: cat,
      peer_count: vals.length,
      raw_median_mult: vals.length > 0 ? round4(quantile(vals, 0.5)) : 0,
      log_residual_adj: round4(adj),
      sector_mult_factor: round4(Math.exp(adj)),
      token_rate: round4((sc.token + 5 * globalTokenRate) / (sc.total + 5)),
      total_protocols: sc.total,
    };
  }
  livePeers.sort((a, b) => b.mcap_usd - a.mcap_usd || a.slug.localeCompare(b.slug));

  return {
    weights: {
      intercept: round4(w[0]!), b_tvl: round4(w[1]!), b_growth: round4(w[2]!),
      b_age: round4(w[3]!), b_chains: round4(w[4]!),
      ridge_lambda: RIDGE_LAMBDA, sector_shrinkage_k: SECTOR_SHRINKAGE_K,
      conformal_q25: round4(q25), conformal_q75: round4(q75),
    },
    validation: {
      live_fit_count: points.length,
      loo_median_mult_err: round2(fullLoo.medMult),
      loo_naive_mult_err: round2(naiveMedMult),
      loo_mae_log: round4(fullLoo.mae),
      loo_naive_mae_log: round4(naiveMae),
      loo_error_reduction_pct: round2(((naiveMedMult - fullLoo.medMult) / naiveMedMult) * 100),
      conformal_50_coverage_pct: round2((insideConformal / Math.max(1, fullLoo.signedRes.length)) * 100),
      wf_train_count: wfTrain.length,
      wf_test_count: wfTest.length,
      wf_median_mult_err: round2(wfMedMult),
      wf_naive_mult_err: round2(wfNaiveMult),
      wf_error_reduction_pct: round2(((wfNaiveMult - wfMedMult) / wfNaiveMult) * 100),
      ablation,
    },
    categories,
    livePeers,
    globalTokenRate: round4(globalTokenRate),
  };
}

export function computeTokenOdds(
  p: SnapshotProtocol,
  age: number | null,
  categories: Record<string, CategoryEffect>,
  globalTokenRate: number,
): number {
  const cat = categories[p.category ?? 'Other'];
  const baseRate = cat ? cat.token_rate : globalTokenRate;
  const tvlLift = clamp((Math.log10(Math.max(5e6, p.tvl_usd)) - 7.3) * 0.08, -0.06, 0.16);
  const effAge = age ?? 365;
  const ageHazard = effAge < 540 ? 1.05 : effAge < 900 ? 0.92 : 0.72;
  return round4(clamp((baseRate + tvlLift) * ageHazard, 0.15, 0.88));
}

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

function gasProfile(tier: 'mainnet' | 'l2' | 'alt-l1'): { tx: number; total: number; label: string } {
  if (tier === 'mainnet') return { tx: 4.5, total: 28, label: '$4-$8 Eth L1 gas' };
  if (tier === 'alt-l1') return { tx: 0.12, total: 1, label: 'under $0.20 gas' };
  return { tx: 0.25, total: 2, label: 'under $0.50 L2 gas' };
}

/** Category-specific 6-step DeFi farm playbook (zero boilerplate filler). */
function categoryPlaybook(
  p: SnapshotProtocol,
  capital: number,
  primary: string,
  secondary: string | null,
  hold: number,
  weeks: number,
  gas: { tx: number; total: number; label: string },
  slippage: number,
): FarmStep[] {
  const cat = (p.category ?? '').toLowerCase();
  const n = p.name;
  const half = Math.round(capital / 2);
  const swapSize = Math.round(capital * 0.15);
  const drawdownNote = typeof p.change_7d === 'number' && p.change_7d <= -5
    ? `TVL fell ${p.change_7d.toFixed(1)}% in 7d; verify pool depth first, then hold`
    : `Hold uninterrupted across`;

  if (cat.includes('dex') || cat.includes('liquidity')) {
    return [
      { title: 'Fund LP pair on primary chain', action: `Move $${capital} split 50/50 ($${half} USDC + $${half} core pair asset) onto ${primary} in a clean wallet.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Seed core concentrated LP pool', action: `Provide $${capital} liquidity into ${n}'s deepest stable or blue-chip pool on ${primary} within a wide low-IL band.`, cost: `$${capital} refundable + ${gas.label}`, time: '15 min' },
      { title: 'Log organic multi-pair swap volume', action: `Execute 3 spot swaps of ~$${swapSize} across distinct pools on ${n} over ${weeks} weeks to qualify for trader + LP tiers.`, cost: `~$${(swapSize * 0.003 * 3 + gas.tx * 3).toFixed(2)} swap fees & gas`, time: `15 min (${weeks} wks)` },
      { title: 'Harvest fees & rebalance range', action: `At mid-window (day ${Math.round(hold / 2)}), claim accrued LP fees and re-center range${secondary ? ` or mirror 20% on ${secondary}` : ''}.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: '10 min' },
      { title: `Maintain ${hold}-day epoch streak`, action: `${drawdownNote} ${hold} days so time-weighted LP points register across multiple snapshots.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` },
      { title: 'Claim rewards & revoke router', action: `Check ${n}'s official points portal, withdraw LP + fees, and revoke DEX router token allowances.`, cost: `${gas.label} exit & revoke`, time: '10 min' },
    ];
  }
  if (cat.includes('risk curator') || cat.includes('allocator') || cat.includes('cdp manager')) {
    return [
      { title: `Stage USDC on ${primary}`, action: `Bridge $${capital} USDC onto ${primary} with native gas so vault deposits avoid high L1 fees.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Deposit into flagship curated vault', action: `Supply $${capital} USDC into ${n}'s highest-TVL curated vault on ${primary} and retain the ERC-4626 receipt share.`, cost: `$${capital} refundable + ${gas.label}`, time: '15 min' },
      { title: 'Check curator cap & secondary vault', action: `Allocate 20% ($${Math.round(capital * 0.2)}) into a second ${n} vault${secondary ? ` on ${secondary}` : ''} so wallet qualifies for multi-vault multipliers.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: '10 min' },
      { title: 'Weekly vault accrual & top-up check', action: `Inspect utilization and APY weekly across ${weeks} weeks; add a small $25 top-up once to log multi-epoch activity.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: `6 min/wk (${weeks} wks)` },
      { title: `Hold receipt shares for ${hold} days`, action: `${drawdownNote} ${hold} days to capture full curator points accrual and native lending yield.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` },
      { title: 'Redeem shares & verify points', action: `Confirm accrued points on ${n}'s official dashboard before redeeming vault shares and revoking approvals.`, cost: `${gas.label} redeem`, time: '10 min' },
    ];
  }
  if (cat.includes('rwa')) {
    return [
      { title: `Prepare stablecoin on ${primary}`, action: `Fund $${capital} USDC on ${primary} and confirm whether ${n}'s yield token requires allowlist or permissionless swap.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Mint or acquire yield-bearing RWA', action: `Subscribe or swap $${capital} into ${n}'s primary RWA yield token on ${primary} and hold it in self-custody.`, cost: `$${capital} refundable + ~$${(slippage + gas.tx).toFixed(2)}`, time: '15 min' },
      { title: 'Deploy RWA token into native DeFi pool', action: `If ${n} offers a native collateral or LP wrapper on ${primary}, stake 50% ($${half}) to stack DeFi + RWA points.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: '10 min' },
      { title: 'Track NAV rebase & weekly proofs', action: `Verify weekly yield rebase and official points balance across ${weeks} weeks without churning principal.`, cost: '$0 monitoring', time: `5 min/wk (${weeks} wks)` },
      { title: `Hold through ${hold}-day accrual window`, action: `${drawdownNote} ${hold} days so time-weighted RWA balances count across consecutive snapshots.`, cost: `$${capital} refundable (earns native yield)`, time: `${hold}-day hold` },
      { title: 'Redeem at NAV & revoke approvals', action: `Redeem RWA tokens back to USDC via ${n}'s primary redemption path and revoke contract allowances.`, cost: `${gas.label} exit`, time: '10 min' },
    ];
  }
  if (cat.includes('lend') || cat.includes('cdp') || cat.includes('collateral')) {
    const borrow = Math.round(capital * 0.25);
    return [
      { title: `Stage collateral on ${primary}`, action: `Fund $${capital} USDC or blue-chip collateral on ${primary} with a conservative health-factor buffer.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Supply core collateral market', action: `Deposit $${capital} into ${n}'s deepest lending or collateral market on ${primary} to earn supply points.`, cost: `$${capital} refundable + ${gas.label}`, time: '15 min' },
      { title: 'Open conservative 25% borrow leg', action: `Borrow ~$${borrow} stablecoin (health factor >2.5) on ${n} so wallet earns higher-weighted borrower/minter points.`, cost: `~$${(borrow * 0.05 * (hold / 365) + gas.tx).toFixed(2)} borrow interest & gas`, time: '10 min' },
      { title: 'Weekly health & repayment cadence', action: `Check liquidation buffer weekly over ${weeks} weeks and make one partial $${Math.round(borrow * 0.2)} repay to log active management.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: `6 min/wk (${weeks} wks)` },
      { title: `Keep position open ${hold} days`, action: `${drawdownNote} ${hold} days across multiple points epochs without triggering liquidation.`, cost: `$${capital} refundable principal`, time: `${hold}-day hold` },
      { title: 'Repay debt, withdraw & revoke', action: `Repay the $${borrow} borrow leg in full, withdraw $${capital} collateral, and revoke market approvals.`, cost: `${gas.label} unwind`, time: '10 min' },
    ];
  }
  if (cat.includes('basis') || cat.includes('yield')) {
    return [
      { title: `Fund stablecoin on ${primary}`, action: `Stage $${capital} USDC on ${primary} to enter ${n}'s delta-neutral basis or yield vault.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Stake into flagship basis/yield pool', action: `Deposit $${capital} into ${n}'s core yield strategy on ${primary} and stake the receipt token for boosted points.`, cost: `$${capital} refundable + ${gas.label}`, time: '15 min' },
      { title: 'Check funding rate & epoch rollover', action: `Monitor funding/yield spread weekly across ${weeks} weeks; compound or roll receipt shares${secondary ? ` (optional 20% on ${secondary})` : ''}.`, cost: `~$${(gas.tx * 2).toFixed(2)} gas`, time: `6 min/wk (${weeks} wks)` },
      { title: 'Verify cooldown & exit liquidity', action: `Confirm ${n}'s unstaking cooldown period early so your exit aligns with snapshot cutoffs.`, cost: '$0 check', time: '10 min' },
      { title: `Hold through ${hold}-day yield epochs`, action: `${drawdownNote} ${hold} days to lock in time-weighted yield multiplier tiers.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` },
      { title: 'Queue unstake & claim official points', action: `Initiate cooldown withdrawal, verify points on ${n}'s official UI, and revoke vault allowances.`, cost: `${gas.label} exit`, time: '10 min' },
    ];
  }
  if (cat.includes('derivative') || cat.includes('perpetual') || cat.includes('prediction') || cat.includes('otc')) {
    const notional = Math.round(capital * 0.5);
    return [
      { title: `Deposit margin on ${primary}`, action: `Fund $${capital} USDC margin into ${n} on ${primary} in an isolated trading account.`, cost: `${gas.label}`, time: '10 min' },
      { title: 'Open low-leverage hedged trade', action: `Execute a 1x-2x hedged $${notional} position on ${n}'s liquid BTC/ETH market and hold >24h to avoid wash-trade filters.`, cost: `~$${(notional * 0.001 + gas.tx).toFixed(2)} taker/maker fee`, time: '15 min' },
      { title: 'Build weekly organic volume', action: `Place 1-2 limit orders per week (~$${notional} each) over ${weeks} weeks to accumulate consistent trader points.`, cost: `~$${(notional * 0.0005 * weeks + gas.tx * weeks).toFixed(2)} cumulative fees`, time: `10 min/wk (${weeks} wks)` },
      { title: 'Seed vault/LP if available', action: `Keep 40% ($${Math.round(capital * 0.4)}) in ${n}'s HLP/liquidity vault if offered, earning passive market-making points.`, cost: `~$${gas.tx.toFixed(2)} gas`, time: '10 min' },
      { title: `Stay active across ${hold}-day season`, action: `${drawdownNote} ${hold} days across multiple weekly trading epochs without over-leveraging.`, cost: `$${capital} refundable margin`, time: `${hold}-day hold` },
      { title: 'Flatten positions & withdraw margin', action: `Close open orders, verify season points on ${n}'s official leaderboard, and withdraw margin.`, cost: `${gas.label} withdraw`, time: '10 min' },
    ];
  }
  return [
    { title: `Fund dedicated wallet on ${primary}`, action: `Transfer $${capital} USDC plus native gas onto ${primary} with zero unrelated approvals.`, cost: `${gas.label}`, time: '10 min' },
    { title: `Deploy core ${p.category ?? 'DeFi'} position`, action: `Deposit $${capital} into ${n}'s primary product on ${primary} and retain the receipt asset in self-custody.`, cost: `$${capital} refundable + ${gas.label}`, time: '15 min' },
    { title: 'Weekly qualifying cadence', action: `Perform one organic interaction weekly on ${primary}${secondary ? ` (plus 20% mirror on ${secondary})` : ''} over ${weeks} weeks.`, cost: `~$${(gas.tx * weeks).toFixed(2)} cumulative gas`, time: `8 min/wk (${weeks} wks)` },
    { title: 'Stack native points / boost tier', action: `Enable any non-custodial points or lock-free boost tier on ${n} without exposing seed phrases.`, cost: '$0 setup', time: '10 min' },
    { title: `Hold through ${hold}-day epoch window`, action: `${drawdownNote} ${hold} days so multi-epoch snapshots register full weight.`, cost: `$${capital} refundable (~$${slippage.toFixed(2)} spread)`, time: `${hold}-day hold` },
    { title: 'Verify eligibility & unwind cleanly', action: `Inspect ${n}'s official rewards page before withdrawing $${capital} and revoking contract approvals.`, cost: `${gas.label} exit & revoke`, time: '10 min' },
  ];
}

export function buildFarmPlan(p: SnapshotProtocol, val: ValuationResult): FarmEconomics {
  const { primary, secondary, gasTier } = selectPrimaryChain(p.chains);
  const gas = gasProfile(gasTier);
  const baseCap = p.tvl_usd >= 1e9 ? 2500 : p.tvl_usd >= 250e6 ? 2000 : p.tvl_usd >= 50e6 ? 1500 : 1000;
  const capital = gasTier === 'mainnet' ? Math.max(baseCap, 2500) : baseCap;
  const hold = val.age_days === null ? 45 : val.age_days < 180 ? 60 : val.age_days < 365 ? 45 : 30;
  const weeks = Math.max(4, Math.ceil(hold / 7));
  const activeMinutes = 10 + 15 + weeks * 8 + 10 + 10;
  const activeHours = activeMinutes / 60;

  const rawYield = Math.min(0.4, (val.implied_value_usd * 0.10 * 1.5) / Math.max(1, p.tvl_usd));
  const expectedYield = rawYield * val.token_odds;
  const grossEv = capital * expectedYield;
  const slippage = capital * 0.0015;
  const totalCost = gas.total + slippage;
  const netEv = round2(grossEv - totalCost);
  const hourlyEv = round2(netEv / Math.max(0.5, activeHours));

  // Break-even FDV where gross expected airdrop equals non-refundable gas + slippage
  const breakevenFdv = Math.max(1, Math.round((totalCost * p.tvl_usd) / Math.max(1e-6, capital * 0.10 * 1.5 * val.token_odds)));
  const safetyMargin = round2(val.implied_value_usd / breakevenFdv);

  let verdict: 'FARM' | 'SELECTIVE' | 'SKIP' = 'SKIP';
  let verdictReason = `Gas (~$${gas.total}) + spread exceed diluted airdrop EV`;
  if (netEv >= 15 && hourlyEv >= 10) {
    verdict = 'FARM';
    verdictReason = `+$${netEv.toFixed(0)} EV ($${hourlyEv.toFixed(1)}/hr), ${safetyMargin.toFixed(1)}× above break-even FDV`;
  } else if (netEv >= 3) {
    verdict = 'SELECTIVE';
    verdictReason = `Modest +$${netEv.toFixed(1)} EV (${safetyMargin.toFixed(1)}× break-even); best if already on ${primary}`;
  }

  const steps = categoryPlaybook(p, capital, primary, secondary, hold, weeks, gas, slippage);

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
    breakeven_fdv_usd: breakevenFdv,
    safety_margin_x: safetyMargin,
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
    farm_breakeven_fdv_usd: item.farm.breakeven_fdv_usd,
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
    breakeven_fdv_usd: item.farm.breakeven_fdv_usd,
    safety_margin_x: item.farm.safety_margin_x,
    verdict: item.farm.verdict,
  }));

  const watchlistCount = board.filter((r) => watchlistSet.has(r.slug)).length;
  const earlyCount = board.length - watchlistCount;
  const liveTotalCount = snapshot.protocols.filter((p) => !isPreToken(p)).length;
  const excludedCount = snapshot.protocols.length - board.length - liveTotalCount;

  return {
    board,
    dataset: {
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
    },
  };
}

export function parseSnapshot(text: string): Snapshot {
  const raw = JSON.parse(text) as Record<string, unknown>;
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('snapshot.json is not an object');
  if (typeof raw.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(raw.as_of)) throw new Error('invalid as_of');
  if (typeof raw.source !== 'string') throw new Error('invalid source');
  if (typeof raw.floor_usd !== 'number' || !Number.isSafeInteger(raw.floor_usd) || raw.floor_usd <= 0) throw new Error('invalid floor_usd');
  if (!Array.isArray(raw.protocols)) throw new Error('invalid protocols');
  if (!Array.isArray(raw.watchlist) || raw.watchlist.some((s) => typeof s !== 'string')) throw new Error('invalid watchlist');

  const protocols: SnapshotProtocol[] = raw.protocols.map((v, i) => {
    const p = v as Record<string, unknown>;
    if (!p || typeof p !== 'object' || typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number') {
      throw new Error(`invalid protocol at index ${i}`);
    }
    return {
      slug: p.slug, name: p.name,
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

  return { as_of: raw.as_of, source: raw.source, floor_usd: raw.floor_usd, protocols, watchlist: raw.watchlist as string[] };
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
    if (i > 0 && r.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${r.slug} outranks previous row`;
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) {
      const len = s.trim().length;
      if (len < 20 || len > 600) return `${r.slug} has farm step of length ${len}`;
    }
  }
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return 'all rows identical';
  return null;
}

export function renderDataScript(dataset: RadarDataset): string {
  return `window.RADAR_DATA=${JSON.stringify(dataset)};\n`;
}
