# Attention Improvements — Slides

## Setup

```bash
cd slides
pnpm install
pnpm dev   # opens http://localhost:3030
```

Slides live in [slides.md](slides.md). Edit and the browser hot-reloads.

## Export

```bash
pnpm build      # static site -> dist/
pnpm export     # PDF (requires playwright; pnpm exec playwright install chromium)
```

## Conventions

- One slide per `---`.
- Speaker notes go after `<!--` on the slide they belong to (Slidev convention).
- Math uses `$...$` (KaTeX).
- Citations link to PDFs in `../papers/` or arXiv.
- Each section's slides live in this single `slides.md` for now; we can split with `src:` imports later if it gets too long.
