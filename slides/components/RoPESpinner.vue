<script setup lang="ts">
/*
  RoPESpinner — animate RoPE's per-position rotation of (q, k) dim-pairs.
  Multiple 2D planes side by side, each representing a dimension pair.
  Higher frequencies rotate faster as `position` increments; lower
  frequencies barely move. The same `position` ticks across all planes
  so the audience sees "at this position, here's what's happening to
  every pair simultaneously."
*/
import { ref, onMounted, onBeforeUnmount, computed } from 'vue';

const props = withDefaults(defineProps<{
  /** How many dim pairs to show, ordered fast → slow. */
  pairs?: number;
  /** Max position the loop reaches before wrapping back to 0. */
  maxPos?: number;
  /** Milliseconds between position increments. */
  tickMs?: number;
  /** Size (px) of one plane. */
  size?: number;
  /** RoPE base — controls the frequency spread (10000 in the paper). */
  base?: number;
  /** Total model dim used to compute the per-pair frequency. */
  d?: number;
}>(), { pairs: 4, maxPos: 16, tickMs: 280, size: 110, base: 10000, d: 64 });

const position = ref(0);
let timer: number | undefined;

onMounted(() => {
  timer = window.setInterval(() => {
    position.value = (position.value + 1) % (props.maxPos + 1);
  }, props.tickMs);
});
onBeforeUnmount(() => { if (timer !== undefined) window.clearInterval(timer); });

/*
  RoPE frequency for pair i (0-indexed, paper convention):
    θ_i = 1 / base^(2i / d)
  We pick `pairs` indices spread roughly evenly across log(θ) so the
  visual shows the full fast→slow spectrum.
*/
const pairIndices = computed(() => {
  const idx: number[] = [];
  for (let k = 0; k < props.pairs; k++) {
    // Spread across i ∈ [0, d/2 - 1] on a log scale
    const frac = props.pairs === 1 ? 0 : k / (props.pairs - 1);
    idx.push(Math.floor(frac * (props.d / 2 - 1)));
  }
  return idx;
});

function thetaFor(i: number): number {
  return 1 / Math.pow(props.base, (2 * i) / props.d);
}

// Angle in radians for pair i at the current position.
function angle(i: number): number {
  return thetaFor(i) * position.value;
}

function freqLabel(i: number): string {
  const t = thetaFor(i);
  // Cycles per position-step
  if (t > 0.5) return `fast (θ≈${t.toFixed(2)}/pos)`;
  if (t > 0.05) return `mid (θ≈${t.toFixed(3)}/pos)`;
  return `slow (θ≈${t.toExponential(1)}/pos)`;
}

const half = computed(() => props.size / 2);
const r = computed(() => props.size * 0.36);
</script>

<template>
  <div class="rope-spinner flex flex-col items-center gap-2">
    <div class="flex items-center gap-3">
      <div class="text-xs opacity-70 tabular-nums">position p = <span class="font-semibold text-amber-400">{{ position }}</span></div>
    </div>

    <div class="flex gap-4 items-end">
      <div v-for="(i, k) in pairIndices" :key="k" class="flex flex-col items-center gap-1">
        <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="rope-plane rounded bg-zinc-900/30">
          <!-- Reference circle -->
          <circle :cx="half" :cy="half" :r="r" fill="none" stroke="#444" stroke-width="0.7" stroke-dasharray="2 3" />
          <!-- Axes -->
          <line :x1="half - r - 4" :y1="half" :x2="half + r + 4" :y2="half" stroke="#555" stroke-width="0.5" />
          <line :x1="half" :y1="half - r - 4" :x2="half" :y2="half + r + 4" stroke="#555" stroke-width="0.5" />
          <!-- Initial (unrotated) ghost vector -->
          <line :x1="half" :y1="half" :x2="half + r" :y2="half" stroke="#fff" stroke-width="1.2" stroke-dasharray="2 2" opacity="0.35" />
          <!-- Rotated vector q (and implicitly k by the same angle) -->
          <line
            :x1="half"
            :y1="half"
            :x2="half + r * Math.cos(-angle(i))"
            :y2="half + r * Math.sin(-angle(i))"
            stroke="#fbbf24"
            stroke-width="2.2"
            stroke-linecap="round"
          />
          <!-- Arrow head -->
          <circle :cx="half + r * Math.cos(-angle(i))" :cy="half + r * Math.sin(-angle(i))" r="3" fill="#fbbf24" />
          <!-- Pair index label -->
          <text :x="6" :y="14" class="pair-label" fill="#aaa">pair i = {{ i }}</text>
        </svg>
        <div class="text-[10px] opacity-60">{{ freqLabel(i) }}</div>
      </div>
    </div>
  </div>
</template>

<style scoped>
svg.rope-plane {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
svg.rope-plane .pair-label {
  font-size: 9px;
}
</style>
