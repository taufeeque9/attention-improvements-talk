<script setup lang="ts">
/*
  AttnMask — render an n×n causal attention pattern as a small SVG heatmap.
  Each row i shows which keys j (j <= i) a query at position i attends to.
  Filled cell = attended. Used to visually compare dense / SWA / NSA / DSA / V4.
*/

type Preset =
  | 'dense'
  | 'causal'
  | 'sliding-window'
  | 'nsa-cmp'
  | 'nsa-slc'
  | 'nsa-win'
  | 'dsa-topk'
  | 'v4-csa'
  | 'v4-hca';

const props = withDefaults(defineProps<{
  preset: Preset;
  n?: number;
  size?: number;
  title?: string;
  // Pattern-specific knobs.
  window?: number;      // sliding-window / nsa-win
  block?: number;       // nsa-cmp / nsa-slc / v4-csa / v4-hca
  picks?: number;       // nsa-slc / dsa-topk / v4-csa
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

function attended(i: number, j: number): number {
  if (j > i) return 0;
  switch (props.preset) {
    case 'dense':
      return 0; // (unused; "dense" still respects causality below)
    case 'causal':
      return 1;
    case 'sliding-window':
      return (i - j) < props.window ? 1 : 0;
    case 'nsa-cmp': {
      // Coarse: attended to compressed-block centers up to query's block.
      const qBlock = Math.floor(i / props.block);
      const kBlock = Math.floor(j / props.block);
      // Only the first cell of each compressed block counts as "attended."
      return kBlock <= qBlock && j % props.block === 0 ? 0.85 : 0;
    }
    case 'nsa-slc': {
      // Sparse: a few selected blocks (pseudo-deterministic by row).
      const kBlock = Math.floor(j / props.block);
      const qBlock = Math.floor(i / props.block);
      if (kBlock > qBlock) return 0;
      // pick `picks` blocks deterministically per row
      const seed = (i * 9301 + 49297) % 233280;
      const pickStride = Math.max(1, Math.floor((qBlock + 1) / props.picks));
      return (kBlock % pickStride === seed % pickStride) ? 1 : 0;
    }
    case 'nsa-win':
      return (i - j) < props.window ? 1 : 0;
    case 'dsa-topk': {
      // Scattered top-k selection: pseudo-random sparse sampling.
      const x = ((i + 1) * 2654435761 ^ (j + 1) * 40503) >>> 0;
      const density = props.picks / Math.max(1, i + 1);
      return (x % 10000) / 10000 < density ? 1 : 0;
    }
    case 'v4-csa': {
      // Compressed (every `block` tokens collapse) + top-k of compressed entries.
      const kBlock = Math.floor(j / props.block);
      const qBlock = Math.floor(i / props.block);
      // Sliding window component (recent uncompressed)
      if (i - j < props.window) return 1;
      // Compressed cells (first of each block) — sparsely selected
      if (kBlock > qBlock) return 0;
      if (j % props.block !== 0) return 0;
      const x = ((i + 1) * 2654435761 ^ (kBlock + 1) * 40503) >>> 0;
      const density = props.picks / Math.max(1, qBlock + 1);
      return (x % 10000) / 10000 < density ? 0.9 : 0;
    }
    case 'v4-hca': {
      // Heavily compressed (large block), dense over compressed entries.
      const kBlock = Math.floor(j / props.block);
      const qBlock = Math.floor(i / props.block);
      if (i - j < props.window) return 1;
      if (kBlock > qBlock) return 0;
      if (j % props.block !== 0) return 0;
      return 0.7;
    }
  }
}

const cell = props.size / props.n;
</script>

<template>
  <div class="attn-mask inline-flex flex-col items-center gap-1">
    <div v-if="title" class="text-xs opacity-70">{{ title }}</div>
    <svg
      :width="size"
      :height="size"
      :viewBox="`0 0 ${size} ${size}`"
      class="rounded bg-zinc-900/30"
    >
      <!-- causal diagonal hint -->
      <line x1="0" y1="0" :x2="size" :y2="size" stroke="#888" stroke-width="0.4" stroke-dasharray="2 2" opacity="0.35" />
      <g v-for="i in n" :key="`r${i}`">
        <g v-for="j in n" :key="`c${j}`">
          <rect
            v-if="attended(i - 1, j - 1) > 0"
            :x="(j - 1) * cell"
            :y="(i - 1) * cell"
            :width="cell - 0.5"
            :height="cell - 0.5"
            :fill="color"
            :fill-opacity="attended(i - 1, j - 1)"
            rx="0.5"
          />
        </g>
      </g>
      <text
        v-if="showAxes"
        :x="size / 2"
        :y="size - 2"
        text-anchor="middle"
        fill="#888"
        font-size="9"
      >key j →</text>
    </svg>
  </div>
</template>
