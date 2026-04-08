---
layout: default
title: GPT-2 Activation Viewer
description: Lightweight GPT-2 prompt-level activation viewer included with the repository.
permalink: /gpt2-viewer/
---

# GPT-2 Activation Viewer

This repository also includes a lightweight GPT-2 activation viewer for prompt-level inspection. It is separate from the small-transformer retrieval experiments and is best treated as a compact inspection tool rather than part of the main paper evidence.

<figure class="paper-figure">
  <img src="figures/gpt2_activation_viewer.png" alt="GPT-2 activation viewer interface">
  <figcaption><strong>Figure 1.</strong> Prompt-level GPT-2 activation viewer bundled with the repository.</figcaption>
</figure>

## Build A Viewer Payload

```bash
.venv/bin/python scripts/build_interactive_model_viewer.py \
  --prompt "The secret code is 73914. Repeat the secret code exactly:" \
  --prompt-id demo_prompt_001 \
  --task copy \
  --model gpt2-small \
  --out outputs/viewer_payload.json
```

## Run The Viewer

```bash
cd webapp
.venv/bin/python -m http.server 8000
```

Then open `http://localhost:8000` and upload `outputs/viewer_payload.json`.

## Repository Sources

- [Viewer payload builder]({{ site.repository_url }}/blob/main/scripts/build_interactive_model_viewer.py)
- [Web UI]({{ site.repository_url }}/tree/main/webapp)
- [Repository root]({{ site.repository_url }})
