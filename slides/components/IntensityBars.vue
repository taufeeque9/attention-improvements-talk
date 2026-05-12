<script setup lang="ts">
/*
  IntensityBars — horizontal bar chart on a log x-axis with a single
  vertical reference line (e.g. the H100 roofline ridge). Built for the
  "intensity climbed from 1 to ~240" arc slide, where only intensity
  varies and the full roofline chart is overkill.
*/

const props = withDefaults(defineProps<{
  items: Array<{ label: string; value: number; color?: string; note?: string }>;
  ridge?: { value: number; label: string };
  unit?: string;
  xMin?: number;
  xMax?: number;
  width?: number;
  rowHeight?: number;
}>(), {
  unit: 'FLOPs / byte',
  xMin: 1,
  xMax: 600,
  width: 560,
  rowHeight: 30,
});

const padL = 80, padR = 16, padT = 18, padB = 28;
const innerW = props.width - padL - padR;
const totalH = padT + props.items.length * props.rowHeight + padB;

const logXMin = Math.log10(props.xMin);
const logXMax = Math.log10(props.xMax);

function xPos(x: number): number {
  return padL + ((Math.log10(Math.max(x, props.xMin)) - logXMin) / (logXMax - logXMin)) * innerW;
}

const xTicks = (() => {
  const ticks: number[] = [];
  for (let i = Math.ceil(logXMin); i <= Math.floor(logXMax); i++) {
    ticks.push(Math.pow(10, i));
  }
  return ticks;
})();
</script>

<template>
  <svg
    :width="width"
    :height="totalH"
    :viewBox="`0 0 ${width} ${totalH}`"
    class="intensity-bars rounded bg-zinc-900/20"
  >
    <!-- Bars -->
    <g v-for="(item, idx) in items" :key="`bar${idx}`">
      <!-- Label -->
      <text
        :x="padL - 8"
        :y="padT + idx * rowHeight + rowHeight / 2 + 4"
        text-anchor="end"
        fill="#ddd"
        class="row-label"
      >{{ item.label }}</text>
      <!-- Bar -->
      <rect
        :x="xPos(xMin)"
        :y="padT + idx * rowHeight + 4"
        :width="Math.max(2, xPos(item.value) - xPos(xMin))"
        :height="rowHeight - 8"
        :fill="item.color || '#60a5fa'"
        fill-opacity="0.85"
        rx="3"
      />
      <!-- Value -->
      <text
        :x="xPos(item.value) + 6"
        :y="padT + idx * rowHeight + rowHeight / 2 + 4"
        :fill="item.color || '#60a5fa'"
        class="row-value"
      >{{ item.value }}</text>
    </g>

    <!-- Ridge / reference line -->
    <g v-if="ridge">
      <line
        :x1="xPos(ridge.value)"
        :y1="padT - 4"
        :x2="xPos(ridge.value)"
        :y2="padT + items.length * rowHeight"
        stroke="#34d399"
        stroke-width="1.5"
        stroke-dasharray="4 3"
        opacity="0.85"
      />
      <text
        :x="xPos(ridge.value)"
        :y="padT - 7"
        text-anchor="middle"
        fill="#34d399"
        class="ridge-label"
      >{{ ridge.label }}</text>
    </g>

    <!-- X axis -->
    <line
      :x1="padL"
      :y1="padT + items.length * rowHeight"
      :x2="padL + innerW"
      :y2="padT + items.length * rowHeight"
      stroke="#888"
      stroke-width="0.8"
    />
    <g v-for="t in xTicks" :key="`xt${t}`">
      <line
        :x1="xPos(t)"
        :y1="padT + items.length * rowHeight"
        :x2="xPos(t)"
        :y2="padT + items.length * rowHeight + 4"
        stroke="#888"
      />
      <text
        :x="xPos(t)"
        :y="padT + items.length * rowHeight + 16"
        text-anchor="middle"
        fill="#999"
        class="tick-label"
      >{{ t >= 1000 ? `${t/1000}k` : t }}</text>
    </g>
    <text
      :x="padL + innerW / 2"
      :y="totalH - 6"
      text-anchor="middle"
      fill="#bbb"
      class="axis-label"
    >arithmetic intensity ({{ unit }}, log scale)</text>
  </svg>
</template>

<style scoped>
/* Pin SVG text size in CSS so Slidev's theme cascade doesn't inflate it. */
svg.intensity-bars {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
svg.intensity-bars .row-label {
  font-size: 12px;
  font-weight: 500;
}
svg.intensity-bars .row-value {
  font-size: 12px;
  font-weight: 600;
}
svg.intensity-bars .ridge-label {
  font-size: 10px;
  font-weight: 600;
}
svg.intensity-bars .tick-label {
  font-size: 9px;
}
svg.intensity-bars .axis-label {
  font-size: 10px;
}
</style>
