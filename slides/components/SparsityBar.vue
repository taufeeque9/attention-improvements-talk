<script setup lang="ts">
/*
  SparsityBar — show a "full context" bar overlaid with the (small) active
  K/V slice a sparse method actually attends to, broken down by component.
  Used on the NSA slide to make ~8%-active visceral.
*/

const props = withDefaults(defineProps<{
  total: number;
  parts: Array<{ label: string; value: number; color: string }>;
  width?: number;
  height?: number;
}>(), { width: 640, height: 56 });

const active = props.parts.reduce((s, p) => s + p.value, 0);
const fracActive = active / props.total;

function px(v: number): number {
  return (v / props.total) * (props.width - 2);
}
</script>

<template>
  <div class="sparsity-bar flex flex-col gap-2">
    <!-- The full-context bar -->
    <div class="relative rounded" :style="{ width: `${width}px`, height: `${height}px`, background: 'rgba(120,120,120,0.18)' }">
      <!-- Stacked active components, left-aligned -->
      <div class="absolute inset-y-1 left-1 flex gap-[2px]">
        <div
          v-for="(p, idx) in parts"
          :key="idx"
          class="rounded-sm flex items-center justify-center text-[10px] text-zinc-900 font-semibold"
          :style="{ width: `${px(p.value)}px`, height: `${height - 8}px`, background: p.color }"
          :title="`${p.label}: ${p.value} tokens`"
        >
          <span v-if="px(p.value) > 28">{{ p.value }}</span>
        </div>
      </div>
      <!-- Labels strip beneath -->
      <div class="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] opacity-80 tabular-nums">
        {{ active.toLocaleString() }} / {{ total.toLocaleString() }} tokens
        <span class="text-amber-400 font-semibold">({{ (fracActive * 100).toFixed(1) }}%)</span>
      </div>
    </div>

    <!-- Legend -->
    <div class="flex gap-4 text-[11px] flex-wrap">
      <div v-for="(p, idx) in parts" :key="`leg${idx}`" class="flex items-center gap-1">
        <span class="inline-block w-3 h-3 rounded-sm" :style="{ background: p.color }" />
        <span class="opacity-80">{{ p.label }}</span>
        <span class="opacity-50 tabular-nums">· {{ p.value }}</span>
      </div>
    </div>
  </div>
</template>
