<script setup lang="ts">
/*
  ReceptiveField — illustrate how SWA's per-layer window expands into a
  much larger effective receptive field through depth, for a *decoder-only*
  (causal) model. Every modern frontier model that interleaves SWA with
  full-attention is decoder-only and causal, so the band extends only to
  the LEFT of the focus token — never to the right.

  Reads as: focus token at the bottom-right; layers stacked upward; band
  fans left and up, widening at deeper layers. Cells to the right of the
  focus exist (so the audience sees them) but are never attended to.
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

/*
  Place the focus near the right edge so the leftward causal wedge fits,
  while leaving a small strip of "future" cells visible on the right —
  they exist in the diagram (greyed out) but are never highlighted,
  making the causal-mask story visible.
*/
const focus = props.tokens - 3;

/*
  Causal receptive-field radius at a given layer.
  Layer L can reach L * window tokens to the left of focus (inclusive of
  focus itself). Row index `layerFromTop` (0 = top row) maps to the
  deepest layer, so the widest band sits at the top.
*/
function reachAt(layerFromTop: number): { lo: number; hi: number } {
  const layerNum = props.layers - layerFromTop; // top row → deepest
  const radius = layerNum * props.window;
  return {
    lo: Math.max(0, focus - radius),
    hi: focus, // causal: never attend to future positions
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
            ? 0.35 + (layers - L) * 0.15
            : 0.15"
          rx="2"
        />
      </g>
      <!-- Layer label: top row = deepest layer; bottom row = layer 1 -->
      <text :x="size - 6" :y="(L - 1) * layerH + layerH / 2 + 10" fill="#888" text-anchor="end" class="layer-label">
        layer {{ layers - L + 1 }}
      </text>
    </g>

    <!-- Crown the focus column across all layers -->
    <rect
      :x="focus * cellW + 2"
      :y="6"
      :width="cellW - 4"
      :height="totalH - 12"
      fill="none"
      stroke="#fbbf24"
      stroke-width="1.5"
      stroke-dasharray="3 2"
      rx="2"
    />
    <!-- Solid crown around the focus cell at the bottom row (the query position) -->
    <rect
      :x="focus * cellW + 2"
      :y="(layers - 1) * layerH + 6"
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
