<script setup lang="ts">
/*
  GQAExplorer — interactive MHA ↔ GQA ↔ MQA explorer.
  Click a G chip to pick the number of KV groups; query heads stay constant
  at H, KV heads collapse into G groups. KV-cache footprint scales linearly
  with G so the audience sees the bytes literally shrink as G drops.
*/
import { ref, computed } from 'vue';

const props = withDefaults(defineProps<{
  H?: number;
  dk?: number;
  initialG?: number;
}>(), { H: 32, dk: 128, initialG: 8 });

const G = ref(props.initialG);
const options = computed(() => {
  // Show divisors of H, capped to a reasonable count of chips.
  const divisors: number[] = [];
  for (let g = 1; g <= props.H; g++) {
    if (props.H % g === 0) divisors.push(g);
  }
  return divisors;
});

const kind = computed(() => {
  if (G.value === props.H) return 'MHA';
  if (G.value === 1) return 'MQA';
  return `GQA-${G.value}`;
});

const intensity = computed(() => props.H / G.value);
const cacheBytes = computed(() => 2 * G.value * props.dk);
const cacheRef = computed(() => 2 * props.H * props.dk);
const cacheFrac = computed(() => cacheBytes.value / cacheRef.value);

// Build query heads grid + KV groups, computing colors per group.
const queryHeads = computed(() => {
  const groupSize = props.H / G.value;
  return Array.from({ length: props.H }, (_, h) => ({
    id: h,
    group: Math.floor(h / groupSize),
  }));
});

// Color palette — distinct hues, soft saturation.
function groupColor(g: number, total: number): string {
  const hue = (g * 360) / Math.max(total, 1);
  return `hsl(${hue.toFixed(0)}, 65%, 60%)`;
}
</script>

<template>
  <div class="gqa-explorer flex flex-col gap-3">
    <!-- Controls -->
    <div class="flex flex-wrap items-center gap-2 text-sm">
      <span class="opacity-70 mr-2">G =</span>
      <button
        v-for="g in options"
        :key="g"
        class="px-2 py-0.5 rounded text-xs transition-all border"
        :class="g === G
          ? 'bg-amber-500/80 text-black border-amber-300 font-semibold'
          : 'border-zinc-600 opacity-70 hover:opacity-100'"
        @click="G = g"
      >{{ g }}</button>
      <span class="ml-3 px-2 py-0.5 rounded bg-zinc-700/40 text-xs font-semibold">
        {{ kind }}
      </span>
    </div>

    <!-- Query heads row -->
    <div>
      <div class="text-[10px] opacity-60 mb-1">H = {{ H }} query heads (each asks a different question)</div>
      <div class="flex gap-[2px] flex-wrap" style="max-width: 720px;">
        <div
          v-for="qh in queryHeads"
          :key="`q${qh.id}`"
          class="w-4 h-5 rounded-sm"
          :style="{ background: groupColor(qh.group, G) }"
          :title="`query head ${qh.id} → KV group ${qh.group}`"
        />
      </div>
    </div>

    <!-- KV heads row, sized so cache footprint shrinks visually -->
    <div>
      <div class="text-[10px] opacity-60 mb-1">G = {{ G }} KV head{{ G === 1 ? '' : 's' }} ({{ H / G }} queries each)</div>
      <div class="flex gap-[2px] items-stretch">
        <div
          v-for="g in G"
          :key="`kv${g}`"
          class="h-10 rounded-sm flex items-center justify-center text-[9px] text-zinc-900 font-semibold"
          :style="{
            background: groupColor(g - 1, G),
            width: `${(720 / H)}px`,
          }"
        >KV</div>
      </div>
    </div>

    <!-- Footprint + readouts -->
    <div class="flex items-end gap-6 pt-2 text-xs">
      <div class="flex flex-col">
        <span class="opacity-60">cache bytes / token (relative to MHA)</span>
        <div class="relative w-[280px] h-4 bg-zinc-700/30 rounded mt-1">
          <div
            class="absolute inset-y-0 left-0 rounded transition-all duration-300"
            :style="{
              width: `${cacheFrac * 100}%`,
              background: 'linear-gradient(90deg, #60a5fa, #a78bfa)',
            }"
          />
          <div class="absolute inset-0 flex items-center justify-end pr-1 text-[10px] tabular-nums font-semibold">
            {{ (cacheFrac * 100).toFixed(1) }}%
          </div>
        </div>
      </div>
      <div class="flex flex-col items-start">
        <span class="opacity-60">long-context decode intensity = H/G</span>
        <span class="text-amber-400 text-xl font-bold tabular-nums">{{ intensity }} <span class="text-xs opacity-70">FLOPs/byte</span></span>
      </div>
    </div>
  </div>
</template>
