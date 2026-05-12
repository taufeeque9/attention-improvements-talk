<script setup lang="ts">
/*
  ReceptiveField — illustrate how SWA's per-layer window expands into a
  much larger effective receptive field through depth. Stacks `layers`
  rows of tokens; the top-center token's reachable ancestors at each
  layer below are highlighted, showing the triangular fan-out of width
  `layers * window`.
*/

const props = withDefaults(defineProps<{
  tokens?: number;
  layers?: number;
  window?: number;
  size?: number;
}>(), { tokens: 25, layers: 4, window: 2, size: 360 });

const cellW = props.size / props.tokens;
const layerH = 28;
const totalH = props.layers * layerH + 12;

// Top-most layer, center token
const focus = Math.floor(props.tokens / 2);

// For each layer (top = 0), compute the reachable set width (one-sided)
function reachAt(layerFromTop: number): { lo: number; hi: number } {
  const radius = layerFromTop * props.window;
  return {
    lo: Math.max(0, focus - radius),
    hi: Math.min(props.tokens - 1, focus + radius),
  };
}
</script>

<template>
  <svg :width="size" :height="totalH" :viewBox="`0 0 ${size} ${totalH}`" class="receptive-field-svg rounded bg-zinc-900/30">
    <g v-for="L in layers" :key="`l${L}`">
      <g v-for="t in tokens" :key="`l${L}t${t}`">
        <rect
          :x="(t - 1) * cellW + 2"
          :y="(L - 1) * layerH + 6"
          :width="cellW - 4"
          :height="layerH - 8"
          :fill="(t - 1) >= reachAt(L - 1).lo && (t - 1) <= reachAt(L - 1).hi
            ? '#60a5fa'
            : '#475569'"
          :fill-opacity="(t - 1) >= reachAt(L - 1).lo && (t - 1) <= reachAt(L - 1).hi
            ? 0.8 - (L - 1) * 0.12
            : 0.15"
          rx="2"
        />
      </g>
      <!-- Layer label -->
      <text :x="size - 6" :y="(L - 1) * layerH + layerH / 2 + 10" fill="#888" text-anchor="end" class="layer-label">
        layer {{ layers - L + 1 }}
      </text>
    </g>

    <!-- Crown the focus token -->
    <rect
      :x="focus * cellW + 2"
      :y="6"
      :width="cellW - 4"
      :height="layerH - 8"
      fill="none"
      stroke="#fbbf24"
      stroke-width="2"
      rx="2"
    />
  </svg>
</template>

<style scoped>
svg.receptive-field-svg {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
svg.receptive-field-svg .layer-label {
  font-size: 10px;
}
</style>
