<script setup lang="ts">
/*
  Roofline — log-x throughput chart with the standard hockey-stick ridge.
  X axis = arithmetic intensity (FLOPs/byte, log scale).
  Y axis = relative throughput (compute-bound plateau = 1).
  Below the ridge: bandwidth-bound (throughput grows with intensity).
  At/above the ridge: compute-bound (throughput saturates at peak).
*/

const props = withDefaults(defineProps<{
  peak?: number;
  xMin?: number;
  xMax?: number;
  width?: number;
  height?: number;
  markers?: Array<{ x: number; label: string; color?: string; tone?: 'dim' | 'bright' }>;
  // How many markers to reveal (use with v-click-style stepping). -1 = all.
  revealed?: number;
}>(), {
  peak: 290,
  xMin: 0.5,
  xMax: 600,
  width: 560,
  height: 280,
  markers: () => [],
  revealed: -1,
});

const padL = 44, padR = 16, padT = 16, padB = 32;
const innerW = props.width - padL - padR;
const innerH = props.height - padT - padB;

const logXMin = Math.log10(props.xMin);
const logXMax = Math.log10(props.xMax);

function xPos(x: number): number {
  return padL + ((Math.log10(x) - logXMin) / (logXMax - logXMin)) * innerW;
}

function yPos(throughput: number): number {
  // throughput in [0, 1.05]
  return padT + innerH - throughput * innerH * 0.95;
}

function throughputAt(x: number): number {
  return Math.min(x / props.peak, 1);
}

// Build ridge polyline
const ridgePoints = (() => {
  const pts: string[] = [];
  for (let i = 0; i <= 80; i++) {
    const lx = logXMin + (i / 80) * (logXMax - logXMin);
    const x = Math.pow(10, lx);
    pts.push(`${xPos(x).toFixed(1)},${yPos(throughputAt(x)).toFixed(1)}`);
  }
  return pts.join(' ');
})();

// X-axis ticks at powers of 10
const xTicks = (() => {
  const ticks: number[] = [];
  for (let i = Math.ceil(logXMin); i <= Math.floor(logXMax); i++) {
    ticks.push(Math.pow(10, i));
  }
  return ticks;
})();

function isVisible(idx: number): boolean {
  if (props.revealed < 0) return true;
  return idx < props.revealed;
}
</script>

<template>
  <svg :width="width" :height="height" :viewBox="`0 0 ${width} ${height}`" class="rounded bg-zinc-900/20">
    <!-- Bandwidth-bound shaded region (below ridge) -->
    <polygon
      :points="`${padL},${padT + innerH} ${ridgePoints} ${padL + innerW},${padT + innerH}`"
      fill="#fb7185"
      fill-opacity="0.06"
    />
    <!-- Compute-bound region (above ridge ceiling) -->
    <rect
      :x="xPos(peak)"
      :y="padT"
      :width="padL + innerW - xPos(peak)"
      :height="innerH * 0.05"
      fill="#34d399"
      fill-opacity="0.10"
    />

    <!-- Axes -->
    <line :x1="padL" :y1="padT + innerH" :x2="padL + innerW" :y2="padT + innerH" stroke="#aaa" stroke-width="0.8" />
    <line :x1="padL" :y1="padT" :x2="padL" :y2="padT + innerH" stroke="#aaa" stroke-width="0.8" />

    <!-- X ticks -->
    <g v-for="t in xTicks" :key="`xt${t}`">
      <line :x1="xPos(t)" :y1="padT + innerH" :x2="xPos(t)" :y2="padT + innerH + 4" stroke="#aaa" />
      <text :x="xPos(t)" :y="padT + innerH + 16" text-anchor="middle" fill="#999" font-size="10">
        {{ t >= 1000 ? `${t/1000}k` : t }}
      </text>
    </g>
    <text :x="padL + innerW / 2" :y="height - 6" text-anchor="middle" fill="#bbb" font-size="11">
      arithmetic intensity (FLOPs / byte, log)
    </text>
    <text :x="14" :y="padT + innerH / 2" text-anchor="middle" fill="#bbb" font-size="11"
          :transform="`rotate(-90, 14, ${padT + innerH / 2})`">
      throughput / peak
    </text>

    <!-- Ridge -->
    <polyline :points="ridgePoints" fill="none" stroke="#34d399" stroke-width="2" />

    <!-- Peak horizontal dashed reference -->
    <line :x1="xPos(peak)" :y1="yPos(1)" :x2="padL + innerW" :y2="yPos(1)"
          stroke="#34d399" stroke-width="1" stroke-dasharray="4 3" opacity="0.55" />
    <text :x="padL + innerW - 4" :y="yPos(1) - 4" text-anchor="end" fill="#34d399" font-size="10">
      peak ≈ {{ peak }}
    </text>

    <!-- Markers -->
    <g v-for="(m, idx) in markers" :key="`m${idx}`">
      <g v-if="isVisible(idx)">
        <line
          :x1="xPos(m.x)"
          :y1="yPos(throughputAt(m.x))"
          :x2="xPos(m.x)"
          :y2="padT + innerH"
          :stroke="m.color || '#fbbf24'"
          stroke-width="1"
          stroke-dasharray="2 3"
          opacity="0.6"
        />
        <circle
          :cx="xPos(m.x)"
          :cy="yPos(throughputAt(m.x))"
          r="5"
          :fill="m.color || '#fbbf24'"
          :fill-opacity="m.tone === 'dim' ? 0.5 : 0.95"
          stroke="#fff"
          stroke-width="1"
        />
        <text
          :x="xPos(m.x)"
          :y="yPos(throughputAt(m.x)) - 10"
          text-anchor="middle"
          :fill="m.color || '#fbbf24'"
          font-size="11"
          font-weight="600"
        >{{ m.label }}</text>
      </g>
    </g>
  </svg>
</template>
