<script setup lang="ts">
/*
  MLARoPE — three-stage diagram showing the MLA × RoPE conflict and the fix.
  Each stage shows the matmul chain that produces a key vector from the
  cached latent. Stage 1 = no RoPE (absorption works). Stage 2 = naive
  RoPE on K_raw (absorption breaks). Stage 3 = decoupled fix.
  Autoplay cycles through the three stages on a slow interval; clicking a
  stage chip pins it.

  All math notation is rendered via SVG <tspan> with .it / .sup / .sub
  classes (defined in scoped CSS) — no foreignObject, no KaTeX, just
  italic + baseline-shifted tspans so symbols read as proper math.
*/
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';

const props = withDefaults(defineProps<{
  autoplay?: boolean;
  intervalMs?: number;
}>(), { autoplay: true, intervalMs: 3500 });

type Stage = 0 | 1 | 2;
const stage = ref<Stage>(0);
const pinned = ref(false);
let timer: number | undefined;

const stageMeta = [
  { id: 0, label: 'no RoPE', desc: 'absorption: fold W^UK into W^Q at inference, no K decompression' },
  { id: 1, label: 'naive RoPE', desc: 'absorption breaks — R_p is per-token, sits between the two weights' },
  { id: 2, label: 'decoupled fix', desc: 'absorbable content path + small RoPE-only side channel' },
] as const;

function step() {
  stage.value = ((stage.value + 1) % 3) as Stage;
}
function startAutoplay() {
  stopAutoplay();
  if (!props.autoplay) return;
  timer = window.setInterval(step, props.intervalMs);
}
function stopAutoplay() {
  if (timer !== undefined) { window.clearInterval(timer); timer = undefined; }
}
function pick(s: Stage) { stage.value = s; pinned.value = true; stopAutoplay(); }
function resume() { pinned.value = false; startAutoplay(); }

onMounted(() => { if (!pinned.value) startAutoplay(); });
onBeforeUnmount(stopAutoplay);

const isStage = (s: number) => stage.value === s;
const description = computed(() => stageMeta[stage.value].desc);
</script>

<template>
  <div class="mla-rope flex flex-col items-center gap-3">
    <!-- Stage chips -->
    <div class="flex items-center gap-2 text-sm">
      <button
        v-for="m in stageMeta"
        :key="m.id"
        class="px-2 py-0.5 rounded text-xs border transition-all"
        :class="m.id === stage
          ? 'bg-amber-500/80 text-black border-amber-300 font-semibold'
          : 'border-zinc-600 opacity-60 hover:opacity-100'"
        @click="pick(m.id as Stage)"
      >{{ m.label }}</button>
      <button v-if="pinned" class="ml-2 text-[10px] opacity-60 hover:opacity-100 underline" @click="resume">
        resume
      </button>
    </div>

    <!-- Diagram -->
    <svg width="720" height="220" viewBox="0 0 720 220" class="mla-rope-svg rounded bg-zinc-900/30">
      <!-- =========== STAGE 0: no RoPE (absorption works) ===========
        Two stacked chains. TOP = naive decompression (c → W^UK → K_raw → ·q → logit).
        BOTTOM = absorbed equivalent (c → ·q' → logit), where q' carries
        W^UK pre-multiplied into W^Q. Same logit, no K decompression.
        Staggered fade-in via .stage0-top / .stage0-bottom CSS animations
        visually narrates: "here's the naive chain… now absorb W^UK into
        W^Q… same answer, shorter chain."
      -->
      <g v-if="isStage(0)" class="stage-group">

        <!-- TOP ROW: naive decompression -->
        <g class="stage0-top">
          <!-- c -->
          <rect x="80" y="14" width="60" height="36" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
          <text x="110" y="32" text-anchor="middle" class="box-label" fill="#34d399">
            <tspan class="it">c</tspan>
          </text>
          <text x="110" y="44" text-anchor="middle" class="dim-label" fill="#bbb">
            <tspan class="it">d</tspan><tspan class="sub">c</tspan>
          </text>

          <!-- W^UK arrow -->
          <path d="M140,32 L218,32" stroke="#a78bfa" stroke-width="1.8" marker-end="url(#arrow-stage0-uk)"/>
          <text x="180" y="24" text-anchor="middle" class="mat-label" fill="#a78bfa">
            <tspan class="it">W</tspan><tspan class="sup">UK</tspan>
          </text>

          <!-- K_raw -->
          <rect x="225" y="14" width="70" height="36" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
          <text x="260" y="32" text-anchor="middle" class="box-label" fill="#a78bfa">
            <tspan class="it">K</tspan><tspan class="sub">raw</tspan>
          </text>
          <text x="260" y="44" text-anchor="middle" class="dim-label" fill="#bbb">
            <tspan class="it">d</tspan><tspan class="sub">h</tspan>/head
          </text>

          <!-- ·q arrow -->
          <path d="M295,32 L380,32" stroke="#60a5fa" stroke-width="1.8" marker-end="url(#arrow-stage0-q)"/>
          <text x="338" y="24" text-anchor="middle" class="mat-label" fill="#60a5fa">
            · <tspan class="it">q</tspan> (= <tspan class="it">W</tspan><tspan class="sup">Q</tspan> <tspan class="it">h</tspan><tspan class="sub">t</tspan>)
          </text>

          <!-- logit -->
          <rect x="388" y="14" width="90" height="36" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
          <text x="433" y="35" text-anchor="middle" class="box-label" fill="#fbbf24">
            attn logit
          </text>

          <text x="540" y="35" class="dim-label" fill="#888">
            <tspan class="it">d</tspan><tspan class="sub">h</tspan>-dim · <tspan class="it">K</tspan><tspan class="sub">raw</tspan> materialized per token
          </text>
        </g>

        <!-- CONNECTOR -->
        <g class="stage0-connector">
          <path d="M260,55 L260,95" stroke="#fbbf24" stroke-width="1.4" stroke-dasharray="3 3" marker-end="url(#arrow-stage0-down)"/>
          <text x="280" y="80" class="absorb-label" fill="#fbbf24">
            ↓ precompute <tspan class="it">W'</tspan><tspan class="sup">Q</tspan> = (<tspan class="it">W</tspan><tspan class="sup">UK</tspan>)<tspan class="sup">⊤</tspan> <tspan class="it">W</tspan><tspan class="sup">Q</tspan>  · absorb <tspan class="it">W</tspan><tspan class="sup">UK</tspan> into <tspan class="it">W</tspan><tspan class="sup">Q</tspan>
          </text>
        </g>

        <!-- BOTTOM ROW: absorbed equivalent -->
        <g class="stage0-bottom">
          <!-- c (same x as top so eye anchors) -->
          <rect x="80" y="118" width="60" height="36" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
          <text x="110" y="136" text-anchor="middle" class="box-label" fill="#34d399">
            <tspan class="it">c</tspan>
          </text>
          <text x="110" y="148" text-anchor="middle" class="dim-label" fill="#bbb">
            <tspan class="it">d</tspan><tspan class="sub">c</tspan>
          </text>

          <!-- Fused arrow ·q' (color blends purple W^UK + blue W^Q to signal fusion) -->
          <defs>
            <linearGradient id="fused-q" x1="0" x2="1" y1="0" y2="0">
              <stop offset="0%" stop-color="#a78bfa"/>
              <stop offset="100%" stop-color="#60a5fa"/>
            </linearGradient>
          </defs>
          <path d="M140,136 L380,136" stroke="url(#fused-q)" stroke-width="2.4" marker-end="url(#arrow-stage0-qp)"/>
          <text x="260" y="128" text-anchor="middle" class="mat-label" fill="#a78bfa">
            · <tspan class="it">q′</tspan>
          </text>
          <text x="260" y="158" text-anchor="middle" class="dim-label" fill="#bbb">
            <tspan class="it">q′</tspan> = <tspan class="it">W'</tspan><tspan class="sup">Q</tspan> <tspan class="it">h</tspan><tspan class="sub">t</tspan> ∈ ℝ<tspan class="sup"><tspan class="it">d</tspan><tspan class="sub">c</tspan></tspan>
          </text>

          <!-- logit — same x and width as top row so the two logit boxes line up vertically -->
          <rect x="388" y="118" width="90" height="36" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
          <text x="433" y="139" text-anchor="middle" class="box-label" fill="#fbbf24">
            attn logit
          </text>

          <text x="490" y="135" class="dim-label" fill="#34d399">
            <tspan class="it">d</tspan><tspan class="sub">c</tspan>-dim dot product
          </text>
          <text x="490" y="150" class="dim-label" fill="#34d399">
            no <tspan class="it">K</tspan><tspan class="sub">raw</tspan> ever materialized
          </text>
        </g>

        <!-- ≡ sits dead between the two logit boxes (both centered at x=433) -->
        <text x="433" y="88" text-anchor="middle" dominant-baseline="middle" fill="#fbbf24" class="equiv-sign">≡</text>
      </g>

      <!-- Stage-0-specific arrow markers (colored to match the arrow they end) -->
      <defs>
        <marker id="arrow-stage0-uk" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#a78bfa"/>
        </marker>
        <marker id="arrow-stage0-q" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#60a5fa"/>
        </marker>
        <marker id="arrow-stage0-qp" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#60a5fa"/>
        </marker>
        <marker id="arrow-stage0-down" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#fbbf24"/>
        </marker>
      </defs>

      <!-- =========== STAGE 1: naive RoPE breaks absorption =========== -->
      <g v-if="isStage(1)" class="stage-group">
        <rect x="40" y="40" width="80" height="44" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
        <text x="80" y="60" text-anchor="middle" class="box-label" fill="#34d399">
          <tspan class="it">c</tspan> (latent)
        </text>
        <text x="80" y="76" text-anchor="middle" class="dim-label" fill="#bbb">
          <tspan class="it">d</tspan><tspan class="sub">c</tspan>
        </text>

        <path d="M120,62 L180,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage1)"/>
        <text x="150" y="55" text-anchor="middle" class="mat-label" fill="#a78bfa">
          <tspan class="it">W</tspan><tspan class="sup">UK</tspan>
        </text>

        <rect x="190" y="40" width="70" height="44" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="225" y="65" text-anchor="middle" class="box-label" fill="#a78bfa">
          <tspan class="it">K</tspan><tspan class="sub">raw</tspan>
        </text>

        <!-- R_p between W^UK·c and final K -->
        <path d="M260,62 L320,62" stroke="#f43f5e" stroke-width="2" marker-end="url(#arrow-stage1)"/>
        <text x="290" y="55" text-anchor="middle" class="mat-label" fill="#f43f5e">
          <tspan class="it">R</tspan><tspan class="sub">p</tspan> (RoPE)
        </text>

        <rect x="330" y="40" width="80" height="44" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="370" y="65" text-anchor="middle" class="box-label" fill="#a78bfa">
          <tspan class="it">K</tspan><tspan class="sub">rot</tspan>
        </text>

        <path d="M410,62 L470,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage1)"/>
        <text x="440" y="55" text-anchor="middle" class="mat-label" fill="#60a5fa">
          · <tspan class="it">q</tspan>
        </text>

        <rect x="480" y="40" width="100" height="44" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
        <text x="530" y="65" text-anchor="middle" class="box-label" fill="#fbbf24">attention logit</text>

        <!-- Red "can't fold" warning under the R block -->
        <path d="M260,110 Q295,150 330,110" fill="none" stroke="#f43f5e" stroke-width="1.5" stroke-dasharray="3 3"/>
        <text x="295" y="170" text-anchor="middle" class="absorb-label" fill="#f43f5e">
          <tspan class="it">R</tspan><tspan class="sub">p</tspan> is per-token — sits between <tspan class="it">W</tspan><tspan class="sup">UK</tspan> and <tspan class="it">W</tspan><tspan class="sup">Q</tspan> → can't fold
        </text>
        <text x="295" y="186" text-anchor="middle" class="dim-label" fill="#f43f5e">
          decompression cost returns. MLA's bandwidth win evaporates.
        </text>
      </g>

      <!-- =========== STAGE 2: decoupled fix =========== -->
      <g v-if="isStage(2)" class="stage-group">
        <!-- Latent c -->
        <rect x="20" y="20" width="70" height="40" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
        <text x="55" y="42" text-anchor="middle" class="box-label" fill="#34d399">
          <tspan class="it">c</tspan> (latent)
        </text>

        <!-- hidden h for RoPE side channel -->
        <rect x="20" y="124" width="70" height="40" rx="6" fill="#94a3b8" fill-opacity="0.20" stroke="#94a3b8"/>
        <text x="55" y="146" text-anchor="middle" class="box-label" fill="#94a3b8">
          <tspan class="it">h</tspan><tspan class="sub">t</tspan>
        </text>

        <!-- Content path -->
        <path d="M90,40 L160,40" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="125" y="33" text-anchor="middle" class="mat-label" fill="#a78bfa">
          <tspan class="it">W</tspan><tspan class="sup">UK</tspan>
        </text>

        <rect x="170" y="20" width="90" height="40" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="215" y="36" text-anchor="middle" class="box-label" fill="#a78bfa">
          <tspan class="it">K</tspan><tspan class="sup">C</tspan> (content)
        </text>
        <text x="215" y="52" text-anchor="middle" class="dim-label" fill="#bbb">absorbable</text>

        <!-- RoPE side path -->
        <path d="M90,144 L160,144" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="125" y="135" text-anchor="middle" class="mat-label" fill="#a78bfa">
          <tspan class="it">W</tspan><tspan class="sup">KR</tspan>
        </text>
        <path d="M170,144 L210,144" stroke="#f43f5e" stroke-width="2" marker-end="url(#arrow-stage2)"/>
        <text x="190" y="135" text-anchor="middle" class="mat-label" fill="#f43f5e">
          <tspan class="it">R</tspan><tspan class="sub">p</tspan>
        </text>

        <rect x="220" y="124" width="90" height="40" rx="6" fill="#fb7185" fill-opacity="0.25" stroke="#fb7185"/>
        <text x="265" y="140" text-anchor="middle" class="box-label" fill="#fb7185">
          <tspan class="it">K</tspan><tspan class="sup">R</tspan> (rotated)
        </text>
        <text x="265" y="156" text-anchor="middle" class="dim-label" fill="#bbb">
          <tspan class="it">d</tspan><tspan class="sub">h</tspan><tspan class="sup">R</tspan> = 64, shared
        </text>

        <!-- Concat join -->
        <path d="M260,40 C320,40 320,92 360,92" stroke="#aaa" stroke-width="1.4" fill="none" marker-end="url(#arrow-stage2)"/>
        <path d="M310,144 C340,144 340,92 360,92" stroke="#aaa" stroke-width="1.4" fill="none" marker-end="url(#arrow-stage2)"/>

        <rect x="370" y="72" width="80" height="40" rx="6" fill="#a78bfa" fill-opacity="0.18" stroke="#a78bfa" stroke-dasharray="3 2"/>
        <text x="410" y="88" text-anchor="middle" class="box-label" fill="#a78bfa">
          <tspan class="it">K</tspan> = [<tspan class="it">K</tspan><tspan class="sup">C</tspan> ; <tspan class="it">K</tspan><tspan class="sup">R</tspan>]
        </text>
        <text x="410" y="104" text-anchor="middle" class="dim-label" fill="#bbb">concat</text>

        <path d="M450,92 L520,92" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="485" y="84" text-anchor="middle" class="mat-label" fill="#60a5fa">
          · <tspan class="it">q</tspan>
        </text>

        <rect x="530" y="72" width="120" height="40" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
        <text x="590" y="96" text-anchor="middle" class="box-label" fill="#fbbf24">
          <tspan class="it">q</tspan><tspan class="sup">⊤</tspan><tspan class="it">K</tspan><tspan class="sup">C</tspan>
          + <tspan class="it">q</tspan><tspan class="sup">⊤</tspan><tspan class="it">K</tspan><tspan class="sup">R</tspan>
        </text>

        <!-- Annotation -->
        <text x="360" y="200" text-anchor="middle" class="absorb-label" fill="#34d399">
          <tspan class="it">C</tspan> path stays absorbable (one matmul against latent) · <tspan class="it">R</tspan> path is small, head-shared
        </text>
      </g>

      <!-- Arrow defs (per stage so colors can differ; here we use neutral) -->
      <defs>
        <marker id="arrow-stage0" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#aaa"/>
        </marker>
        <marker id="arrow-stage1" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#aaa"/>
        </marker>
        <marker id="arrow-stage2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto">
          <path d="M0,0 L10,5 L0,10 z" fill="#aaa"/>
        </marker>
      </defs>
    </svg>

    <div class="text-xs opacity-80">{{ description }}</div>
  </div>
</template>

<style scoped>
svg.mla-rope-svg {
  font-family: ui-sans-serif, system-ui, sans-serif;
}
svg.mla-rope-svg .box-label   { font-size: 12px; font-weight: 600; }
svg.mla-rope-svg .dim-label   { font-size: 10px; }
svg.mla-rope-svg .mat-label   { font-size: 11px; }
svg.mla-rope-svg .absorb-label { font-size: 11px; font-style: italic; }
svg.mla-rope-svg .equiv-sign  { font-size: 22px; font-weight: 700; }

/* Math tspans: italic for variables, baseline-shift for sub/super. */
svg.mla-rope-svg .it  { font-style: italic; }
svg.mla-rope-svg .sup { baseline-shift: super; font-size: 75%; }
svg.mla-rope-svg .sub { baseline-shift: sub;   font-size: 75%; }

.stage-group { transition: opacity 0.3s ease; }

/*
  Stage 0 visual narrative: top row appears, then the "absorb" connector,
  then the collapsed bottom row. Each part stays visible after its
  reveal so the final state shows both equivalent chains side by side.
*/
.stage0-top       { animation: fadeIn 600ms ease-out both; animation-delay: 100ms; }
.stage0-connector { animation: fadeIn 600ms ease-out both; animation-delay: 1100ms; }
.stage0-bottom    { animation: fadeIn 600ms ease-out both; animation-delay: 1900ms; }

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(4px); }
  to   { opacity: 1; transform: translateY(0); }
}
</style>
