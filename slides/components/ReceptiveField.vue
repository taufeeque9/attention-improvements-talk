<script setup lang="ts">
/*
  ReceptiveField — illustrate how SWA's per-layer window expands into a
  much larger effective receptive field through depth.

  Reads as: focus token at the *bottom*, layers stacked upward. Each
  row's highlighted band shows the *receptive field at that layer* — i.e.
  which token positions contributed to the focus's representation by the
  end of that layer's attention. Band widens as we go up (deeper layer →
  more accumulated context), so the shape is a downward-pointing triangle
  with the wide base on top, matching the "reach grows with depth" framing.
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

// Center column — the position whose receptive field we're tracking.
const focus = Math.floor(props.tokens / 2);

/*
  Receptive field radius at a given layer (1-indexed).
  Layer L's RF = L * window tokens on each side of focus.
  Row index in the SVG is `layerFromTop` (0 = top row); the top row
  represents the deepest layer (props.layers), so the wide band sits there.
*/
function reachAt(layerFromTop: number): { lo: number; hi: number } {
  const layerNum = props.layers - layerFromTop; // top row → deepest
  const radius = layerNum * props.window;
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
