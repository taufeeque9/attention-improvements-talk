<script setup lang="ts">
/*
  PrefillVsDecode — animated split-screen.
  Left:   prefill — the full causal triangle lights up in parallel waves
          (compute-bound: many flops over a fixed-size matrix).
  Right:  decode  — one new token streams the entire KV cache, one block at
          a time, into a single new logit row (bandwidth-bound: re-read the
          cache every step).
*/

const props = withDefaults(defineProps<{
  n?: number;
  size?: number;
}>(), { n: 16, size: 280 });

const cell = props.size / props.n;
const cells: Array<{ i: number; j: number; delay: number }> = [];
for (let i = 0; i < props.n; i++) {
  for (let j = 0; j <= i; j++) {
    // Diagonal wave: all cells with the same i+j light up together.
    cells.push({ i, j, delay: (i + j) * 0.025 });
  }
}
const decodeRow = props.n - 1;
const streamCells = Array.from({ length: props.n }, (_, j) => ({ j, delay: j * 0.12 }));
</script>

<template>
  <div class="grid grid-cols-2 gap-8">
    <!-- Prefill -->
    <div class="flex flex-col items-center gap-2">
      <div class="text-sm font-semibold">Prefill</div>
      <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="rounded bg-zinc-900/40">
        <rect
          v-for="(c, idx) in cells"
          :key="`p${idx}`"
          :x="c.j * cell + 1"
          :y="c.i * cell + 1"
          :width="cell - 2"
          :height="cell - 2"
          fill="#60a5fa"
          class="prefill-cell"
          :style="{ animationDelay: `${c.delay}s` }"
          rx="1"
        />
      </svg>
      <div class="text-xs opacity-70">all cells in parallel · <span class="text-blue-400 font-semibold">compute-bound</span></div>
      <div class="text-[11px] opacity-60">O(n<tspan>²</tspan>) work · matmuls saturate Tensor Cores</div>
    </div>

    <!-- Decode -->
    <div class="flex flex-col items-center gap-2">
      <div class="text-sm font-semibold">Decode (one step)</div>
      <svg :width="size" :height="size" :viewBox="`0 0 ${size} ${size}`" class="rounded bg-zinc-900/40">
        <!-- KV cache columns (full strip) -->
        <rect
          v-for="(c, idx) in streamCells"
          :key="`kv${idx}`"
          :x="c.j * cell + 1"
          :y="2"
          :width="cell - 2"
          :height="size - 4 - cell"
          fill="#444"
          fill-opacity="0.25"
          rx="1"
        />
        <!-- Streaming highlight -->
        <rect
          v-for="(c, idx) in streamCells"
          :key="`s${idx}`"
          :x="c.j * cell + 1"
          :y="2"
          :width="cell - 2"
          :height="size - 4 - cell"
          fill="#f43f5e"
          class="decode-stream"
          :style="{ animationDelay: `${c.delay}s` }"
          rx="1"
        />
        <!-- new-token row -->
        <rect
          v-for="(c, idx) in streamCells"
          :key="`n${idx}`"
          :x="c.j * cell + 1"
          :y="size - cell - 1"
          :width="cell - 2"
          :height="cell - 2"
          fill="#fbbf24"
          class="decode-newtok"
          :style="{ animationDelay: `${c.delay + 0.05}s` }"
          rx="1"
        />
      </svg>
      <div class="text-xs opacity-70">re-stream entire cache · <span class="text-rose-400 font-semibold">bandwidth-bound</span></div>
      <div class="text-[11px] opacity-60">O(n·d) bytes per token · HBM-limited</div>
    </div>
  </div>
</template>

<style scoped>
.prefill-cell {
  opacity: 0;
  animation: prefill-flash 2.4s ease-out infinite;
}
.decode-stream {
  opacity: 0;
  animation: decode-pulse 2.4s ease-out infinite;
}
.decode-newtok {
  opacity: 0;
  animation: decode-light 2.4s ease-out infinite;
}
@keyframes prefill-flash {
  0%, 92%, 100% { opacity: 0; }
  6% { opacity: 0.95; }
  60% { opacity: 0.55; }
}
@keyframes decode-pulse {
  0%, 90%, 100% { opacity: 0; }
  4%, 12% { opacity: 0.85; }
  20%, 88% { opacity: 0.25; }
}
@keyframes decode-light {
  0%, 90%, 100% { opacity: 0; }
  10% { opacity: 1; }
  50%, 88% { opacity: 0.6; }
}
</style>
