// Peer-Calibrated Token Launch Valuation Model (radar-v2)
// Derives every valuation multiple, range band, and farm plan strictly from snapshot.json.
// No protocol name is ever hardcoded or special-cased.

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

export interface ModelParams {
  shrinkage_k: number;
  multiple_min: number;
  multiple_max: number;
  growth_weight: number;
  growth_scale: number;
  maturity_floor: number;
  maturity_range: number;
  maturity_days: number;
  unknown_age_days: number;
  chain_weight: number;
  chain_log_cap: number;
}

export const DEFAULT_PARAMS: ModelParams = {
  shrinkage_k: 5,
  multiple_min: 0.08,
  multiple_max: 0.85,
  growth_weight: 0.35,
  growth_scale: 50,
  maturity_floor: 0.65,
  maturity_range: 0.45,
  maturity_days: 1095,
  unknown_age_days: 365,
  chain_weight: 0.05,
  chain_log_cap: 2.5,
};

export const MODEL_VERSION = 'radar-v2-peer-eb';

export interface CategoryCalibration {
  category: string;
  peer_count: number;
  raw_median: number | null;
  base_multiple: number;
  bear_multiple: number;
  bull_multiple: number;
}

export interface CalibrationSummary {
  live_peer_count: number;
  global_median: number;
  global_p25: number;
  global_p75: number;
  categories: Record<string, CategoryCalibration>;
}

export interface FarmStep {
  title: string;
  action: string;
  cost: string;
  time: string;
}

export interface FarmPlanSummary {
  capital_usd: number;
  est_gas_usd: number;
  hold_days: number;
  primary_chain: string;
  secondary_chain: string | null;
  gas_tier: 'mainnet' | 'l2' | 'alt-l1';
  steps: FarmStep[];
}

export interface ValuationBreakdown {
  base_multiple: number;
  bear_multiple: number;
  bull_multiple: number;
  peer_count: number;
  growth_pct: number | null;
  growth_multiplier: number;
  age_days: number | null;
  maturity_multiplier: number;
  chain_count: number;
  chain_multiplier: number;
  implied_value_usd: number;
  bear_value_usd: number;
  bull_value_usd: number;
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
  model: ValuationBreakdown;
  farm_summary: {
    capital_usd: number;
    est_gas_usd: number;
    hold_days: number;
    primary_chain: string;
    gas_tier: 'mainnet' | 'l2' | 'alt-l1';
  };
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
  base_multiple: number;
  peer_count: number;
  growth_multiplier: number;
  maturity_multiplier: number;
  chain_multiplier: number;
  capital_usd: number;
  est_gas_usd: number;
  hold_days: number;
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
}

export interface RadarDataset {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  params: ModelParams;
  calibration: CalibrationSummary;
  universe: {
    total_protocols: number;
    watchlist_count: number;
    early_count: number;
    board_count: number;
    live_count: number;
    excluded_non_launch_count: number;
  };
  rows: RadarRow[];
  live: LivePeerRow[];
}

const DAY_MS = 86_400_000;
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

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

function quantile(sorted: number[], q: number): number {
  if (sorted.length === 0) return 0;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo]!;
  return sorted[lo]! + (sorted[hi]! - sorted[lo]!) * (pos - lo);
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

/** Calibrate category MCap/TVL base, bear (P25), and bull (P75) multiples from live-token peers in the snapshot. */
export function calibrateCategoryMultiples(
  protocols: SnapshotProtocol[],
  params: ModelParams = DEFAULT_PARAMS,
): { calibration: CalibrationSummary; livePeers: LivePeerRow[] } {
  const byCat = new Map<string, number[]>();
  const allRatios: number[] = [];
  const livePeers: LivePeerRow[] = [];

  for (const p of protocols) {
    if (isPreToken(p) || p.category === 'CEX') continue;
    if (p.mcap_usd === null || p.mcap_usd <= 0 || p.tvl_usd <= 0) continue;
    const ratio = p.mcap_usd / p.tvl_usd;
    if (ratio < 0.005 || ratio > 10.0) continue;

    const cat = p.category ?? 'Other';
    let list = byCat.get(cat);
    if (!list) {
      list = [];
      byCat.set(cat, list);
    }
    list.push(ratio);
    allRatios.push(ratio);
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
    });
  }

  allRatios.sort((a, b) => a - b);
  const globalMedian = allRatios.length > 0 ? quantile(allRatios, 0.5) : 0.25;
  const globalP25 = allRatios.length > 0 ? quantile(allRatios, 0.25) : 0.08;
  const globalP75 = allRatios.length > 0 ? quantile(allRatios, 0.75) : 0.75;

  const k = params.shrinkage_k;
  const categories: Record<string, CategoryCalibration> = {};

  for (const [cat, vals] of [...byCat.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    vals.sort((a, b) => a - b);
    const n = vals.length;
    const rawMed = quantile(vals, 0.5);
    const rawP25 = quantile(vals, 0.25);
    const rawP75 = quantile(vals, 0.75);

    const logShrink = (val: number, prior: number, lo: number, hi: number): number => {
      const wVal = clamp(val, 0.04, 1.5);
      const wPrior = clamp(prior, 0.04, 1.5);
      const blended = Math.exp((n * Math.log(wVal) + k * Math.log(wPrior)) / (n + k));
      return clamp(blended, lo, hi);
    };

    const shrunkBase = logShrink(rawMed, globalMedian, params.multiple_min, params.multiple_max);
    const shrunkBear = logShrink(rawP25, globalP25, params.multiple_min * 0.45, shrunkBase * 0.92);
    const shrunkBull = logShrink(rawP75, globalP75, shrunkBase * 1.15, params.multiple_max * 2.2);

    categories[cat] = {
      category: cat,
      peer_count: n,
      raw_median: round4(rawMed),
      base_multiple: round4(shrunkBase),
      bear_multiple: round4(shrunkBear),
      bull_multiple: round4(shrunkBull),
    };
  }

  livePeers.sort((a, b) => b.mcap_usd - a.mcap_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));

  return {
    calibration: {
      live_peer_count: allRatios.length,
      global_median: round4(globalMedian),
      global_p25: round4(globalP25),
      global_p75: round4(globalP75),
      categories,
    },
    livePeers,
  };
}

/** Score a single protocol strictly from its snapshot figures and the empirical peer calibration. */
export function computeImpliedValue(
  p: SnapshotProtocol,
  asOf: string,
  calibration: CalibrationSummary,
  params: ModelParams = DEFAULT_PARAMS,
): ValuationBreakdown {
  const catKey = p.category ?? 'Other';
  const catCal = calibration.categories[catKey];
  const baseMultiple = catCal
    ? catCal.base_multiple
    : clamp(calibration.global_median, params.multiple_min, params.multiple_max);
  const bearMultiple = catCal
    ? catCal.bear_multiple
    : clamp(calibration.global_p25, params.multiple_min * 0.45, baseMultiple * 0.92);
  const bullMultiple = catCal
    ? catCal.bull_multiple
    : clamp(calibration.global_p75, baseMultiple * 1.15, params.multiple_max * 2.2);
  const peerCount = catCal ? catCal.peer_count : 0;

  const rawGrowth = typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : 0;
  const growthMult = 1 + params.growth_weight * Math.tanh(rawGrowth / params.growth_scale);

  const age = ageDays(asOf, p.listed_at);
  const effectiveAge = age ?? params.unknown_age_days;
  const maturityRatio = clamp(effectiveAge, 0, params.maturity_days) / params.maturity_days;
  const maturityMult = params.maturity_floor + params.maturity_range * Math.sqrt(maturityRatio);

  const chainCount = Math.max(1, p.chains.length);
  const chainMult = 1 + params.chain_weight * Math.min(Math.log(chainCount), params.chain_log_cap);

  const composite = growthMult * maturityMult * chainMult;
  const implied = Math.max(1, Math.round(p.tvl_usd * baseMultiple * composite));
  const bear = Math.max(1, Math.min(implied, Math.round(p.tvl_usd * bearMultiple * composite)));
  const bull = Math.max(implied, Math.round(p.tvl_usd * bullMultiple * composite));

  return {
    base_multiple: round4(baseMultiple),
    bear_multiple: round4(bearMultiple),
    bull_multiple: round4(bullMultiple),
    peer_count: peerCount,
    growth_pct: p.change_7d,
    growth_multiplier: round4(growthMult),
    age_days: age,
    maturity_multiplier: round4(maturityMult),
    chain_count: chainCount,
    chain_multiplier: round4(chainMult),
    implied_value_usd: implied,
    bear_value_usd: bear,
    bull_value_usd: bull,
  };
}

function classifyGasTier(chain: string): 'mainnet' | 'l2' | 'alt-l1' {
  if (chain === 'Ethereum') return 'mainnet';
  if (ALT_L1_CHAINS.has(chain)) return 'alt-l1';
  if (L2_CHAINS.has(chain)) return 'l2';
  return 'l2';
}

function selectPrimaryChain(chains: string[]): { primary: string; secondary: string | null; gasTier: 'mainnet' | 'l2' | 'alt-l1' } {
  if (chains.length === 0) return { primary: 'its primary chain', secondary: null, gasTier: 'l2' };
  // Prefer a low-fee L2 or Alt-L1 for farming when the protocol supports multiple chains
  const lowFee = chains.find((c) => c !== 'Ethereum' && (L2_CHAINS.has(c) || ALT_L1_CHAINS.has(c)));
  const primary = lowFee ?? chains[0]!;
  const secondary = chains.find((c) => c !== primary) ?? null;
  return { primary, secondary, gasTier: classifyGasTier(primary) };
}

function gasEstimate(tier: 'mainnet' | 'l2' | 'alt-l1'): { setup: number; weekly: number; total: number; label: string } {
  if (tier === 'mainnet') return { setup: 12, weekly: 4, total: 28, label: '$10-$15 Ethereum L1 gas' };
  if (tier === 'alt-l1') return { setup: 0.2, weekly: 0.05, total: 1, label: 'under $0.25 network gas' };
  return { setup: 0.5, weekly: 0.15, total: 2, label: 'under $0.75 L2 gas' };
}

function capitalForTvl(tvl: number): number {
  if (tvl >= 1_000_000_000) return 350;
  if (tvl >= 250_000_000) return 200;
  if (tvl >= 50_000_000) return 100;
  return 50;
}

function holdDaysForAge(age: number | null): number {
  if (age === null) return 45;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

function coreActionForCategory(p: SnapshotProtocol, capital: number, chain: string): { title: string; action: string } {
  const cat = (p.category ?? '').toLowerCase();
  const n = p.name;
  if (cat.includes('risk curator') || cat.includes('allocator') || cat.includes('cdp manager')) {
    return {
      title: 'Deposit into curated vault',
      action: `Deposit $${capital} USDC into ${n}'s flagship curated vault on ${chain} and retain the vault receipt token in your wallet.`,
    };
  }
  if (cat.includes('rwa')) {
    return {
      title: 'Mint yield-bearing RWA',
      action: `Mint or subscribe $${capital} into ${n}'s yield-bearing RWA token on ${chain} and hold it in self-custody.`,
    };
  }
  if (cat.includes('basis') || cat.includes('yield')) {
    return {
      title: 'Enter core yield strategy',
      action: `Deploy $${capital} USDC into ${n}'s primary yield or basis vault on ${chain} and keep shares staked through epoch rollovers.`,
    };
  }
  if (cat.includes('lend') || cat.includes('cdp') || cat.includes('collateral')) {
    return {
      title: 'Supply collateral & borrow lightly',
      action: `Supply $${capital} USDC or blue-chip collateral on ${n} (${chain}) and borrow 20% against it to register active borrower usage.`,
    };
  }
  if (cat.includes('dex') || cat.includes('liquidity')) {
    return {
      title: 'Provide LP & execute swaps',
      action: `Provide $${capital} in a core stable or blue-chip pool on ${n} (${chain}) and execute 2-3 organic spot swaps.`,
    };
  }
  if (cat.includes('derivative') || cat.includes('perpetual') || cat.includes('prediction') || cat.includes('otc')) {
    return {
      title: 'Post margin & generate volume',
      action: `Deposit $${capital} USDC margin on ${n} (${chain}) and open small delta-neutral or hedged positions to log real volume.`,
    };
  }
  if (cat.includes('restak') || cat.includes('staking')) {
    return {
      title: 'Stake native receipt asset',
      action: `Deposit $${capital} of the native staking asset into ${n} on ${chain} and keep the receipt position delegated.`,
    };
  }
  if (cat.includes('payment') || cat.includes('stable') || cat.includes('cedefi')) {
    return {
      title: 'Mint & transact stable balance',
      action: `Mint or fund $${capital} through ${n} on ${chain} and complete at least two on-chain transfers or settlement actions.`,
    };
  }
  if (cat.includes('insurance')) {
    return {
      title: 'Underwrite or purchase cover',
      action: `Commit $${capital} to ${n}'s underwriting pool or purchase a small cover policy on ${chain} and hold it to expiry.`,
    };
  }
  return {
    title: 'Deploy core protocol position',
    action: `Deposit $${capital} USDC into ${n}'s primary product on ${chain} and keep the position active across snapshots.`,
  };
}

/** Build a concrete 6-step costed and timed farm plan strictly from snapshot fields. */
export function buildFarmPlan(p: SnapshotProtocol, age: number | null): FarmPlanSummary {
  const { primary, secondary, gasTier } = selectPrimaryChain(p.chains);
  const gas = gasEstimate(gasTier);
  const capital = capitalForTvl(p.tvl_usd);
  const hold = holdDaysForAge(age);
  const core = coreActionForCategory(p, capital, primary);
  const hasDrawdown = typeof p.change_7d === 'number' && p.change_7d <= -5;
  const weeks = Math.max(4, Math.ceil(hold / 7));

  const step3: FarmStep = secondary
    ? {
        title: 'Multi-chain footprint & weekly cadence',
        action: `Repeat one qualifying interaction weekly on ${primary} (and optionally mirror 20% of size on ${secondary}) so activity spans ${weeks}+ distinct weeks.`,
        cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas over ${weeks} weeks`,
        time: `10 min/week (${weeks} weeks)`,
      }
    : {
        title: 'Weekly organic cadence',
        action: `Execute one qualifying on-chain action weekly on ${primary} (top-up, compound, swap, or rebalance) to avoid single-tx sybil filters.`,
        cost: `~$${(gas.weekly * weeks).toFixed(2)} cumulative gas over ${weeks} weeks`,
        time: `10 min/week (${weeks} weeks)`,
      };

  const step4: FarmStep = hasDrawdown
    ? {
        title: 'Monitor 7d TVL drawdown & hold window',
        action: `TVL moved ${p.change_7d!.toFixed(1)}% over 7d; verify pool liquidity health before scaling, then hold the $${capital} position for ${hold} days across snapshot epochs.`,
        cost: `$${capital} principal remains refundable (monitor slippage)`,
        time: `${hold}-day holding window (5 min weekly check)`,
      }
    : {
        title: 'Hold across snapshot epochs',
        action: `Maintain the $${capital} position uninterrupted for at least ${hold} days so time-weighted points and multi-epoch snapshots register full weight.`,
        cost: `$${capital} principal stays refundable in position`,
        time: `${hold}-day holding window`,
      };

  const steps: FarmStep[] = [
    {
      title: 'Isolate wallet & fund gas',
      action: `Create a dedicated farming wallet on ${primary}, fund it with $${capital} principal plus native gas, and keep zero unrelated approvals on it.`,
      cost: `${gas.label} for initial funding`,
      time: '10 minutes',
    },
    {
      title: core.title,
      action: core.action,
      cost: `$${capital} refundable capital + ${gas.label}`,
      time: '15 minutes',
    },
    step3,
    step4,
    {
      title: 'Verify points & official claim route',
      action: `Check ${p.name}'s official points dashboard before unwinding; never sign blind permits or pay third-party claim fees.`,
      cost: '$0 verification fee (never pay to claim)',
      time: '10 minutes',
    },
    {
      title: 'Log tx hashes & unwind cleanly',
      action: `Record wallet address, deposit/withdraw tx hashes, and epoch dates in your farm ledger, then revoke token allowances after exiting.`,
      cost: `${gas.label} for exit and revoke`,
      time: '10 minutes',
    },
  ];

  return {
    capital_usd: capital,
    est_gas_usd: Math.round(gas.total * 100) / 100,
    hold_days: hold,
    primary_chain: primary,
    secondary_chain: secondary,
    gas_tier: gasTier,
    steps,
  };
}

export function formatStepString(step: FarmStep): string {
  let text = `${step.title}: ${step.action} Cost: ${step.cost}. Time: ${step.time}.`;
  if (text.length > 600) text = `${text.slice(0, 596)}...`;
  if (text.length < 20) text = `${text} Verify position on-chain.`;
  return text;
}

export function buildRadar(
  snapshot: Snapshot,
  params: ModelParams = DEFAULT_PARAMS,
): { board: BoardRow[]; dataset: RadarDataset } {
  const bySlug = new Map(snapshot.protocols.map((p) => [p.slug, p]));
  const watchlistSet = new Set(snapshot.watchlist);
  const { calibration, livePeers } = calibrateCategoryMultiples(snapshot.protocols, params);

  const selected = new Map<string, { protocol: SnapshotProtocol; tier: 'watchlist' | 'early' }>();
  for (const slug of snapshot.watchlist) {
    const p = bySlug.get(slug);
    if (p) selected.set(slug, { protocol: p, tier: 'watchlist' });
  }

  const earlyCandidates = snapshot.protocols
    .filter((p) => !selected.has(p.slug) && isCleanEarlyCandidate(p, snapshot.floor_usd))
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));

  for (const p of earlyCandidates) {
    selected.set(p.slug, { protocol: p, tier: 'early' });
  }

  const scored = [...selected.values()].map(({ protocol, tier }) => {
    const val = computeImpliedValue(protocol, snapshot.as_of, calibration, params);
    const plan = buildFarmPlan(protocol, val.age_days);
    return { protocol, tier, val, plan };
  });

  scored.sort(
    (a, b) =>
      b.val.implied_value_usd - a.val.implied_value_usd ||
      b.protocol.tvl_usd - a.protocol.tvl_usd ||
      (a.protocol.slug < b.protocol.slug ? -1 : a.protocol.slug > b.protocol.slug ? 1 : 0),
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
    model: item.val,
    farm_summary: {
      capital_usd: item.plan.capital_usd,
      est_gas_usd: item.plan.est_gas_usd,
      hold_days: item.plan.hold_days,
      primary_chain: item.plan.primary_chain,
      gas_tier: item.plan.gas_tier,
    },
    farm_steps: item.plan.steps.map(formatStepString),
  }));

  const rows: RadarRow[] = board.map((r) => ({
    rank: r.rank,
    slug: r.slug,
    name: r.name,
    category: r.category,
    chains: r.chains,
    tvl_usd: r.tvl_usd,
    change_7d: r.change_7d,
    listed_at: r.listed_at,
    age_days: r.age_days,
    tier: r.tier,
    implied_value_usd: r.implied_value_usd,
    bear_value_usd: r.bear_value_usd,
    bull_value_usd: r.bull_value_usd,
    base_multiple: r.model.base_multiple,
    peer_count: r.model.peer_count,
    growth_multiplier: r.model.growth_multiplier,
    maturity_multiplier: r.model.maturity_multiplier,
    chain_multiplier: r.model.chain_multiplier,
    capital_usd: r.farm_summary.capital_usd,
    est_gas_usd: r.farm_summary.est_gas_usd,
    hold_days: r.farm_summary.hold_days,
  }));

  const watchlistCount = board.filter((r) => watchlistSet.has(r.slug)).length;
  const earlyCount = board.length - watchlistCount;
  const preTokenTotal = snapshot.protocols.filter(isPreToken).length;

  const dataset: RadarDataset = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    params,
    calibration,
    universe: {
      total_protocols: snapshot.protocols.length,
      watchlist_count: watchlistCount,
      early_count: earlyCount,
      board_count: board.length,
      live_count: livePeers.length,
      excluded_non_launch_count: Math.max(0, preTokenTotal - board.length),
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
