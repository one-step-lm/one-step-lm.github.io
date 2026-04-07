# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Project page for the paper "One-step Language Modeling via Continuous Denoising" (Lee, Yoo, Agarwal, Shah, Huang, Raghunathan, Hong, Boffi, Kim — arXiv:2602.16813). A static GitHub Pages site hosted at `one-step-lm.github.io`.

Two main pages:
- `index.html` — paper landing page (results, figures, BibTeX, text samples carousel)
- `blog/index.html` — explanatory blog post with KaTeX math rendering and embedded MP4 animations

## Local Preview

```bash
python3 -m http.server 8090
```

Then open `http://localhost:8090`. No build step — pure static HTML/CSS/JS.

## Architecture

**CSS framework**: Bulma (`static/css/bulma.min.css`) with custom styles in `static/css/index.css`. Blog has extensive inline `<style>` blocks for dark/light theming and article typography.

**JS**: jQuery-based. `static/js/index.js` handles navbar, interpolation slider, and video autoplay. `static/js/show_generations.js` powers the text sample carousel on the main page. Blog page has inline JS for dark mode toggle and scroll-based video playback.

**Math**: Blog uses KaTeX (CDN-loaded) with `$$`/`$` delimiters via `auto-render`.

**Inline review comments**: `index.html` defines a custom `<sheel-comment>` element for collaborator review notes (yellow callout boxes). Toggle visibility via its `display` CSS property.

## Code Directory

`code/` contains standalone PyTorch demo scripts (not part of the website build):
- `train.py` — training loop for toy flow matching models (v_pred, x_pred_mse, x_pred_ce modes)
- `models.py` — SimpleMLP and SimpleMLPWithVocab
- `datasets.py` — toy dataset generation
- `loss_function_demonstration.py`, `plotting_utils.py` — visualization utilities
- `create_overview_gif.py` — generates the overview animation

These require PyTorch. The main training code lives in a separate repo (`github.com/david3684/flm`).

## Key Conventions

- Figures go in `figures/` (mix of PNG, GIF, PDF). Blog videos go in `blog/videos/` (MP4).
- Template is forked from [nerfies.github.io](https://github.com/nerfies/nerfies.github.io) — keep the Bulma/jQuery structure consistent.
- The paper PDF is stored at `assets/paper.pdf`.
