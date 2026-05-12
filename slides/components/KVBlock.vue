<script setup lang="ts">
/*
  KVBlock — visualize per-token KV-cache footprint for one attention variant.
  Width is proportional to total cached elements per token, so several
  KVBlocks placed side by side give a direct visual comparison.

  Element counts assume K and V are both cached (factor 2 for MHA/GQA/MQA).
  MLA caches a single shared latent c^{KV} of dim d_c (no factor 2).
*/

const props = withDefaults(defineProps<{
  kind: 'MHA' | 'GQA' | 'MQA' | 'MLA';
  H?: number;
  G?: number;
  dk?: number;
  dc?: number;
  // Pixel width corresponding to the reference (MHA) element count.
  scaleMax?: number;
  // Optional label override; otherwise derived from kind + params.
  label?: string;
  showCount?: boolean;
  color?: string;
}>(), {
  H: 32,
  G: 8,
  dk: 128,
  dc: 512,
  scaleMax: 480,
  showCount: true,
});

const elementsRef = 2 * props.H * props.dk; // MHA reference
const elements = (() => {
  switch (props.kind) {
    case 'MHA': return 2 * props.H * props.dk;
    case 'GQA': return 2 * props.G * props.dk;
    case 'MQA': return 2 * props.dk;
    case 'MLA': return props.dc;
  }
})();

const widthPx = Math.max(8, (elements / elementsRef) * props.scaleMax);
const stripeCount = (() => {
  switch (props.kind) {
    case 'MHA': return props.H;
    case 'GQA': return props.G;
    case 'MQA': return 1;
    case 'MLA': return 1;
  }
})();

const colorFor: Record<string, string> = {
  MHA: '#60a5fa',  // blue-400
  GQA: '#a78bfa',  // violet-400
  MQA: '#f472b6',  // pink-400
  MLA: '#34d399',  // emerald-400
};
const fill = props.color ?? colorFor[props.kind];

const derivedLabel = props.label ?? ({
  MHA: `MHA · H=${props.H}`,
  GQA: `GQA-${props.G}`,
  MQA: `MQA`,
  MLA: `MLA · d<sub>c</sub>=${props.dc}`,
} as const)[props.kind];

const countLabel = `${elements.toLocaleString()} el`;
</script>

<template>
  <div class="kv-block inline-flex flex-col items-start gap-1">
    <div class="text-xs opacity-70" v-html="derivedLabel"></div>
    <svg :width="widthPx" height="56" :viewBox="`0 0 ${widthPx} 56`" class="rounded">
      <rect :width="widthPx" height="56" :fill="fill" fill-opacity="0.18" rx="3" />
      <g v-for="i in stripeCount" :key="i">
        <rect
          :x="((i - 1) / stripeCount) * widthPx + 1"
          y="3"
          :width="Math.max(1, widthPx / stripeCount - 2)"
          height="50"
          :fill="fill"
          fill-opacity="0.85"
          rx="1.5"
        />
      </g>
    </svg>
    <div v-if="showCount" class="text-[10px] opacity-60 tabular-nums">{{ countLabel }} / tok</div>
  </div>
</template>
