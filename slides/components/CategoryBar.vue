<script setup lang="ts">
/*
  CategoryBar — top-of-slide strip indicating which "wall" the slide attacks.
    a = KV-cache compression (blue, attacks decode bandwidth)
    c = sparse attention (rose, attacks prefill compute)
    d = systems / kernels (emerald)
*/

const props = withDefaults(defineProps<{
  cat: 'a' | 'c' | 'd';
  text?: string;
}>(), {});

const meta: Record<string, { color: string; text: string }> = {
  a: { color: '#60a5fa', text: '(a) KV-cache compression — decode bandwidth' },
  c: { color: '#f43f5e', text: '(c) Sparse attention — prefill compute' },
  d: { color: '#10b981', text: '(d) Kernels & systems' },
};
const m = meta[props.cat];
</script>

<template>
  <div class="absolute top-0 left-0 right-0 flex items-center px-6 py-1 text-[10px] uppercase tracking-wider opacity-80"
       :style="{ borderTop: `3px solid ${m.color}`, color: m.color }">
    <span>{{ text || m.text }}</span>
  </div>
</template>
