# Mechanistic Transformer Circuits

This repository studies a simple version of a very large question.

Transformers are defined by fixed equations and trained by gradient descent, yet after training they behave as if they contain reusable internal operators, circuits, and feature geometry. Those learned internal structures are part of what we experience as model capability or "intelligence." The goal here is to study that process in settings small enough to inspect directly.

The repository uses tiny transformers and controlled retrieval tasks because they make one part of the problem tractable: if a model learns a real mechanism, we can often recover it, track it across training, and separate genuine circuit formation from memorization.

## Current Findings

The current results support a clear empirical story.

On a clean symbolic key-value retrieval task, a tiny transformer learns a real in-context retrieval circuit rather than a fake shortcut. The recovered mechanism is consistent with a layered decomposition: an upstream support computation prepares the final state, and a downstream head performs the dominant retrieval-write step.

On a single-story next-token benchmark, the same measurement stack fails in the right way. Train accuracy rises, held-out behavior collapses, and no convincing circuit story locks in. That negative control matters because it shows the tooling does not automatically hallucinate interpretable mechanisms when the task mostly invites memorization.

On the larger textual retrieval benchmark, the project gets its main result: a retrieval mechanism can be watched as it emerges during training. Across a full `150`-epoch, `6`-run matrix with curriculum on and off and seeds `0`, `1`, and `2`, the stable object is not one fixed head graph. It is a staged retrieval motif:

- an upstream support or setup component
- a downstream retrieval or write component
- strong query-key and selected-value structure
- a slot-matching computation that is real but often more distributed and later-settling

So the current repo already supports a strong claim: small transformers repeatedly learn a staged retrieval motif, and that motif can be measured across checkpoints, seeds, and curricula.

## Current Research Question

The next question is narrower and deeper:

**why does gradient descent reliably discover motifs at all?**

The empirical work here already says what forms and when. The next step is to understand why optimization repeatedly builds a specific internal operator pattern instead of some other solution. That means moving from circuit description to motif-specific formation laws: how partial subfunctions appear, how they become coordinated, and how gradient descent turns fixed architecture into reusable computation.

This is also where polysemanticity enters the story. A model may not store one clean concept per neuron or one clean function per direction. Features can be packed, reused, and superposed. Understanding how circuits live inside those mixed representations is part of the same formation question, not a separate curiosity.

## Broader Questions

The broader agenda behind the repository is larger than any one benchmark:

- Where does useful computation or "intelligence" actually occur inside a trained network?
- How do learned operators become reusable circuits and feature geometry?
- Why do some motifs appear reliably under gradient descent while others do not?
- If we understand those mechanisms well enough, can we influence outputs in a principled way?

Those questions remain open. This repository does not claim a universal law of circuit formation or a complete theory of intelligence. What it does provide is a controlled experimental path into those questions.

## Start Here

- [Public report](docs/index.md)
- [Full repository report](results.md)
- [Reproducibility](docs/reproducibility.md)
