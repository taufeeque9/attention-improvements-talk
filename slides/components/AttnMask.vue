<script setup lang="ts">
/*
  AttnMask — render an n×n causal attention pattern as a small SVG heatmap.
  Each row i shows which keys j (j <= i) a query at position i attends to.

  Visual convention (shared across NSA / CSA / HCA):
    cell  → single uncompressed token read (full opacity, per-cell gap)
    block → m uncompressed tokens read as a block (full opacity, per-cell gap)
    cmp   → ONE compressed entry summarizing m tokens (faint, no internal gap;
            rendered as a single continuous bar so it visually differs from a
            block of m discrete tokens)
*/

type Preset =
  | 'dense'
  | 'causal'
  | 'sliding-window'
  | 'nsa-cmp'
  | 'nsa-slc'
  | 'nsa-win'
  | 'nsa-all'
  | 'dsa-topk'
  | 'v4-csa'
  | 'v4-hca';

const props = withDefaults(defineProps<{
  preset: Preset;
  n?: number;
  size?: number;
  title?: string;
  window?: number;      // sliding-window / nsa-win / nsa-all / v4-*
  block?: number;       // nsa-cmp / nsa-slc / nsa-all / v4-csa / v4-hca
  picks?: number;       // nsa-slc / nsa-all / dsa-topk / v4-csa
  color?: string;
  showAxes?: boolean;
}>(), {
  n: 32,
  size: 180,
  window: 4,
  block: 4,
  picks: 3,
  color: '#f43f5e',
  showAxes: false,
});

type Cell =
  | { kind: 'none' }
  | { kind: 'cell'; opacity: number }
  | { kind: 'compressed'; opacity: number };

const NONE: Cell = { kind: 'none' };

function classify(i: number, j: number): Cell {
  if (j > i) return NONE;
  switch (props.preset) {
    case 'dense':
      return NONE;
    case 'causal':
      return { kind: 'cell', opacity: 1 };
    case 'sliding-window':
      return (i - j) < props.window ? { kind: 'cell', opacity: 1 } : NONE;
    case 'nsa-cmp': {
      // One compressed entry per source block, up to query's block.
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      return kBlock <= qBlock ? { kind: 'compressed', opacity: 0.5 } : NONE;
    }
    case 'nsa-slc': {
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      if (kBlock > qBlock) return NONE;
      const seed = (i * 9301 + 49297) % 233280;
      const pickStride = Math.max(1, Math.floor((qBlock + 1) / props.picks));
      return kBlock % pickStride === seed % pickStride
        ? { kind: 'cell', opacity: 1 }
        : NONE;
    }
    case 'nsa-win':
      return (i - j) < props.window ? { kind: 'cell', opacity: 1 } : NONE;
    case 'nsa-all': {
      // win takes priority (individual recent tokens, full opacity).
      if (i - j < props.window) return { kind: 'cell', opacity: 1 };
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      if (kBlock > qBlock) return NONE;
      // slc — top-n full blocks (per-cell rendering preserves "m tokens read").
      const seed = (i * 9301 + 49297) % 233280;
      const pickStride = Math.max(1, Math.floor((qBlock + 1) / props.picks));
      if (kBlock % pickStride === seed % pickStride) return { kind: 'cell', opacity: 1 };
      // cmp — every block contributes one compressed entry.
      return { kind: 'compressed', opacity: 0.35 };
    }
    case 'dsa-topk': {
      const x = ((i + 1) * 2654435761 ^ (j + 1) * 40503) >>> 0;
      const density = props.picks / Math.max(1, i + 1);
      return (x % 10000) / 10000 < density ? { kind: 'cell', opacity: 1 } : NONE;
    }
    case 'v4-csa': {
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      if (i - j < props.window) return { kind: 'cell', opacity: 1 };
      if (kBlock > qBlock) return NONE;
      const seed = ((i + 1) * 2654435761) >>> 0;
      const pickStride = Math.max(1, Math.floor((qBlock + 1) / props.picks));
      return kBlock % pickStride === seed % pickStride
        ? { kind: 'compressed', opacity: 0.5 }
        : NONE;
    }
    case 'v4-hca': {
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      if (i - j < props.window) return { kind: 'cell', opacity: 1 };
      if (kBlock > qBlock) return NONE;
      return { kind: 'compressed', opacity: 0.5 };
    }
  }
}

// Compressed bar for row i over compressed-source-block kBlock:
// returns { opacity, span } where span is the number of source cells the bar
// spans (clipped by causality), or null if no compressed entry there.
function compressedBar(i: number, kBlock: number): { opacity: number; span: number } | null {
  const j0 = kBlock * props.block;
  if (j0 > i) return null;
  const c = classify(i, j0);
  if (c.kind !== 'compressed') return null;
  const j1 = Math.min(j0 + props.block - 1, i);
  return { opacity: c.opacity, span: j1 - j0 + 1 };
}

const cell = props.size / props.n;
const numBlocks = Math.ceil(props.n / props.block);
</script>

<template>
  <div class="attn-mask inline-flex flex-col items-center gap-1">
    <div v-if="title" class="text-xs opacity-70">{{ title }}</div>
    <svg
      :width="size"
      :height="size"
      :viewBox="`0 0 ${size} ${size}`"
      class="attn-mask-svg rounded bg-zinc-900/30"
    >
      <!-- causal diagonal hint -->
      <line x1="0" y1="0" :x2="size" :y2="size" stroke="#888" stroke-width="0.4" stroke-dasharray="2 2" opacity="0.35" />

      <!-- Pass 1: compressed-entry bars (one continuous rect per source block;
           the inter-block gap comes from omitting the last 0.5 unit, like cells) -->
      <g v-for="i in n" :key="`cmp-r${i}`">
        <template v-for="kBlock in numBlocks" :key="`cmp-b${i}-${kBlock}`">
          <rect
            v-if="compressedBar(i - 1, kBlock - 1)"
            :x="(kBlock - 1) * props.block * cell"
            :y="(i - 1) * cell"
            :width="compressedBar(i - 1, kBlock - 1)!.span * cell - 0.5"
            :height="cell - 0.5"
            :fill="color"
            :fill-opacity="compressedBar(i - 1, kBlock - 1)!.opacity"
            rx="0.5"
          />
        </template>
      </g>

      <!-- Pass 2: individual-token cells (drawn on top of any cmp bar) -->
      <g v-for="i in n" :key="`r${i}`">
        <template v-for="j in n" :key="`c${i}-${j}`">
          <rect
            v-if="classify(i - 1, j - 1).kind === 'cell'"
            :x="(j - 1) * cell"
            :y="(i - 1) * cell"
            :width="cell - 0.5"
            :height="cell - 0.5"
            :fill="color"
            :fill-opacity="(classify(i - 1, j - 1) as { kind: 'cell'; opacity: number }).opacity"
            rx="0.5"
          />
        </template>
      </g>

      <text
        v-if="showAxes"
        :x="size / 2"
        :y="size - 2"
        text-anchor="middle"
        fill="#888"
        class="axis-hint"
      >key j →</text>
    </svg>
  </div>
</template>

<style scoped>
/* Pin SVG text size in CSS — Slidev's theme cascades a large base font-size
   to descendants and SVG attribute-based font-size can lose to it. */
svg.attn-mask-svg {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
svg.attn-mask-svg .axis-hint {
  font-size: 9px;
}
</style>
