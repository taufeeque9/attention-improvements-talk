<script setup lang="ts">
/*
  MLARoPE — three-stage diagram showing the MLA × RoPE conflict and the fix.
  Each stage shows the matmul chain that produces a key vector from the
  cached latent. Stage 1 = no RoPE (absorption works). Stage 2 = naive
  RoPE on K_raw (absorption breaks). Stage 3 = decoupled fix.
  Autoplay cycles through the three stages on a slow interval; clicking a
  stage chip pins it.
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
  { id: 0, label: 'no RoPE', desc: 'absorption works' },
  { id: 1, label: 'naive RoPE', desc: 'absorption breaks' },
  { id: 2, label: 'decoupled fix', desc: 'absorbable core + RoPE side channel' },
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
      <!--
        Three rows of pipeline, drawn at fixed coordinates and shown/hidden
        per stage. Each row is the matmul chain producing the key vector
        that participates in q·k.
      -->

      <!-- =========== STAGE 0: no RoPE (absorption works) =========== -->
      <g v-if="isStage(0)" class="stage-group">
        <!-- Latent c -->
        <rect x="40" y="40" width="80" height="44" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
        <text x="80" y="60" text-anchor="middle" class="box-label" fill="#34d399">c (latent)</text>
        <text x="80" y="76" text-anchor="middle" class="dim-label" fill="#bbb">d_c</text>

        <!-- W^{UK} -->
        <path d="M120,62 L190,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage0)"/>
        <text x="155" y="55" text-anchor="middle" class="mat-label" fill="#a78bfa">W^UK</text>

        <!-- K_raw -->
        <rect x="200" y="40" width="80" height="44" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="240" y="60" text-anchor="middle" class="box-label" fill="#a78bfa">K_raw</text>
        <text x="240" y="76" text-anchor="middle" class="dim-label" fill="#bbb">d_h per head</text>

        <!-- dot with q -->
        <path d="M280,62 L350,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage0)"/>
        <text x="315" y="55" text-anchor="middle" class="mat-label" fill="#60a5fa">· q</text>

        <!-- result -->
        <rect x="360" y="40" width="100" height="44" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
        <text x="410" y="65" text-anchor="middle" class="box-label" fill="#fbbf24">attention logit</text>

        <!-- Absorption brace beneath -->
        <path d="M120,110 Q200,140 280,110" fill="none" stroke="#34d399" stroke-width="1.5" stroke-dasharray="3 3"/>
        <text x="200" y="155" text-anchor="middle" class="absorb-label" fill="#34d399">
          fold W^UK into q at inference → q'^⊤ · c (one matmul, d_c-dim)
        </text>
      </g>

      <!-- =========== STAGE 1: naive RoPE breaks absorption =========== -->
      <g v-if="isStage(1)" class="stage-group">
        <rect x="40" y="40" width="80" height="44" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
        <text x="80" y="60" text-anchor="middle" class="box-label" fill="#34d399">c (latent)</text>
        <text x="80" y="76" text-anchor="middle" class="dim-label" fill="#bbb">d_c</text>

        <path d="M120,62 L180,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage1)"/>
        <text x="150" y="55" text-anchor="middle" class="mat-label" fill="#a78bfa">W^UK</text>

        <rect x="190" y="40" width="70" height="44" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="225" y="65" text-anchor="middle" class="box-label" fill="#a78bfa">K_raw</text>

        <!-- R_p between W^UK·c and final K -->
        <path d="M260,62 L320,62" stroke="#f43f5e" stroke-width="2" marker-end="url(#arrow-stage1)"/>
        <text x="290" y="55" text-anchor="middle" class="mat-label" fill="#f43f5e">R_p (RoPE)</text>

        <rect x="330" y="40" width="80" height="44" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="370" y="65" text-anchor="middle" class="box-label" fill="#a78bfa">K_rot</text>

        <path d="M410,62 L470,62" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage1)"/>
        <text x="440" y="55" text-anchor="middle" class="mat-label" fill="#60a5fa">· q</text>

        <rect x="480" y="40" width="100" height="44" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
        <text x="530" y="65" text-anchor="middle" class="box-label" fill="#fbbf24">attention logit</text>

        <!-- Red "can't fold" warning under the R block -->
        <path d="M260,110 Q295,150 330,110" fill="none" stroke="#f43f5e" stroke-width="1.5" stroke-dasharray="3 3"/>
        <text x="295" y="170" text-anchor="middle" class="absorb-label" fill="#f43f5e">
          R_p is per-token — sits between W^UK and W^Q → can't fold
        </text>
        <text x="295" y="186" text-anchor="middle" class="dim-label" fill="#f43f5e">
          decompression cost returns. MLA's bandwidth win evaporates.
        </text>
      </g>

      <!-- =========== STAGE 2: decoupled fix =========== -->
      <g v-if="isStage(2)" class="stage-group">
        <!-- Latent c -->
        <rect x="20" y="20" width="70" height="40" rx="6" fill="#34d399" fill-opacity="0.25" stroke="#34d399"/>
        <text x="55" y="42" text-anchor="middle" class="box-label" fill="#34d399">c (latent)</text>

        <!-- hidden h for RoPE side channel -->
        <rect x="20" y="124" width="70" height="40" rx="6" fill="#94a3b8" fill-opacity="0.20" stroke="#94a3b8"/>
        <text x="55" y="146" text-anchor="middle" class="box-label" fill="#94a3b8">h_t</text>

        <!-- Content path -->
        <path d="M90,40 L160,40" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="125" y="33" text-anchor="middle" class="mat-label" fill="#a78bfa">W^UK</text>

        <rect x="170" y="20" width="90" height="40" rx="6" fill="#a78bfa" fill-opacity="0.25" stroke="#a78bfa"/>
        <text x="215" y="36" text-anchor="middle" class="box-label" fill="#a78bfa">K^C (content)</text>
        <text x="215" y="52" text-anchor="middle" class="dim-label" fill="#bbb">absorbable</text>

        <!-- RoPE side path -->
        <path d="M90,144 L160,144" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="125" y="135" text-anchor="middle" class="mat-label" fill="#a78bfa">W^KR</text>
        <path d="M170,144 L210,144" stroke="#f43f5e" stroke-width="2" marker-end="url(#arrow-stage2)"/>
        <text x="190" y="135" text-anchor="middle" class="mat-label" fill="#f43f5e">R_p</text>

        <rect x="220" y="124" width="90" height="40" rx="6" fill="#fb7185" fill-opacity="0.25" stroke="#fb7185"/>
        <text x="265" y="140" text-anchor="middle" class="box-label" fill="#fb7185">K^R (rotated)</text>
        <text x="265" y="156" text-anchor="middle" class="dim-label" fill="#bbb">d_h^R = 64, shared</text>

        <!-- Concat join -->
        <path d="M260,40 C320,40 320,92 360,92" stroke="#aaa" stroke-width="1.4" fill="none" marker-end="url(#arrow-stage2)"/>
        <path d="M310,144 C340,144 340,92 360,92" stroke="#aaa" stroke-width="1.4" fill="none" marker-end="url(#arrow-stage2)"/>

        <rect x="370" y="72" width="80" height="40" rx="6" fill="#a78bfa" fill-opacity="0.18" stroke="#a78bfa" stroke-dasharray="3 2"/>
        <text x="410" y="88" text-anchor="middle" class="box-label" fill="#a78bfa">K = [K^C ; K^R]</text>
        <text x="410" y="104" text-anchor="middle" class="dim-label" fill="#bbb">concat</text>

        <path d="M450,92 L520,92" stroke="#aaa" stroke-width="1.5" marker-end="url(#arrow-stage2)"/>
        <text x="485" y="84" text-anchor="middle" class="mat-label" fill="#60a5fa">· q</text>

        <rect x="530" y="72" width="120" height="40" rx="6" fill="#fbbf24" fill-opacity="0.25" stroke="#fbbf24"/>
        <text x="590" y="96" text-anchor="middle" class="box-label" fill="#fbbf24">q^⊤K^C + q^⊤K^R</text>

        <!-- Annotation -->
        <text x="360" y="200" text-anchor="middle" class="absorb-label" fill="#34d399">
          C path stays absorbable (one matmul against latent) · R path is small, head-shared
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
svg.mla-rope-svg .box-label { font-size: 11px; font-weight: 600; }
svg.mla-rope-svg .dim-label  { font-size: 9px; }
svg.mla-rope-svg .mat-label  { font-size: 10px; }
svg.mla-rope-svg .absorb-label { font-size: 11px; font-style: italic; }
.stage-group { transition: opacity 0.3s ease; }
</style>
