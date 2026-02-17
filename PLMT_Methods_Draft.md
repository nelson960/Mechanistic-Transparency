# Prompt-Level Mechanistic Transparency (PLMT): Methods Draft

## 1. Objective and Scope

Given a prompt \(x\) and generated response tokens \(y_{1:T}\), I estimate a **Prompt->Response Transparency Graph** that explains each target token decision \(y_t\) using internal model components and source tokens, then validate the explanation with **internal counterfactual interventions**.

Primary claim tested in this section:

1. The graph provides high-fidelity attribution of the model's logit margin for each target token.
2. Intervening on graph-selected components causes predicted directional changes in logits and output behavior.

This methods section is model-family-agnostic for autoregressive transformer LMs.

## 2. Setup and Notation

Let \(M\) be a frozen causal language model with vocabulary \(V\). For token decision step \(t\):

- \(y_t\): generated token under deterministic decoding (temperature \(=0\)).
- \(\ell_t(k)\): pre-softmax logit of token \(k \in V\).
- \(k_t^*\): fixed foil token chosen once from the clean run as
  \[
  k_t^* = \arg\max_{k \neq y_t} \ell_t(k).
  \]
- Target margin estimand:
  \[
  \Delta_t = \ell_t(y_t) - \ell_t(k_t^*).
  \]

Fixing \(k_t^*\) from the clean pass avoids post-intervention foil drift.

Let \(r_t \in \mathbb{R}^{d_{\text{model}}}\) be the final residual vector at decision step \(t\), \(W_U \in \mathbb{R}^{d_{\text{model}} \times |V|}\) the unembedding, and \(b_U \in \mathbb{R}^{|V|}\) optional output bias.

Define logit-direction vector:
\[
u_t = W_U[:, y_t] - W_U[:, k_t^*].
\]
Then
\[
\Delta_t = u_t^\top r_t + \beta_t, \quad \beta_t = b_U[y_t] - b_U[k_t^*].
\]

## 3. Residual Decomposition and Component Scores

Decompose residual at step \(t\):
\[
r_t = \sum_{c \in \mathcal{C}} r_{t,c},
\]
where \(\mathcal{C}\) includes embeddings, attention head outputs, MLP outputs, and optionally feature-basis reconstructions.

Component contribution to margin:
\[
s_{t,c} = u_t^\top r_{t,c}.
\]
Predicted margin reconstruction:
\[
\widehat{\Delta}_t = \beta_t + \sum_{c \in \mathcal{C}} s_{t,c}.
\]

Reconstruction error (token-level):
\[
\varepsilon_t = \frac{|\Delta_t - \widehat{\Delta}_t|}{\max(1, |\Delta_t|)}.
\]

## 4. Source Attribution for Attention Heads

For head \(h\), write at step \(t\):
\[
o^h_t = \sum_{i \le t} a^h_{t,i} \cdot W_O^h W_V^h z_i,
\]
where \(a^h_{t,i}\) is attention from destination \(t\) to source position \(i\), and \(z_i\) is head input at source.

Define source-to-margin contribution:
\[
s_{t,h,i} = u_t^\top \left(a^h_{t,i} W_O^h W_V^h z_i\right).
\]
Consistency:
\[
\sum_{i \le t} s_{t,h,i} = s_{t,h}.
\]

This grounds source traces in output-margin effects, not raw attention mass.

## 5. Transparency Graph Construction

For each decision token \(t\), construct graph \(G_t = (N_t, E_t)\):

- Nodes \(N_t\): source tokens \(x_i\), components \(c\), optional features \(f\), target token \(y_t\), foil \(k_t^*\).
- Edges \(E_t\):
  - component->target with weight \(s_{t,c}\),
  - source->head with weight \(s_{t,h,i}\),
  - optional feature->target with feature score.

### 5.1 Selection and Thresholding

Sort components by \(|s_{t,c}|\) descending: \(c_{(1)}, c_{(2)}, \dots\).

Absolute mass covered by top-\(K\):
\[
A_t(K) = \frac{\sum_{j=1}^{K}|s_{t,c_{(j)}}|}{\sum_{c \in \mathcal{C}}|s_{t,c}|}.
\]

Choose minimal \(K_t\) s.t. \(A_t(K_t) \ge \tau_{\text{abs}}\), with preregistered \(\tau_{\text{abs}}=0.90\).

For selected heads, retain source edges meeting either:

1. top-\(m\) by \(|s_{t,h,i}|\) (default \(m=8\)), or
2. \(|s_{t,h,i}| \ge \lambda_{\text{src}} |\Delta_t|\) (default \(\lambda_{\text{src}}=0.02\)).

### 5.2 Anti-Double-Counting Rule

Primary completeness is computed at the component level only (\(s_{t,c}\)). Source edges are explanatory decomposition of selected head terms and are not added again to total completeness.

## 6. Interaction Accounting

To test non-additivity between components \(a,b\):

\[
\delta_t(S) = \Delta_t - \Delta_t^{(-S)},
\]
where \(\Delta_t^{(-S)}\) is margin after ablating set \(S\).

Pair interaction:
\[
I_{t}(a,b) = \delta_t(\{a,b\}) - \delta_t(\{a\}) - \delta_t(\{b\}).
\]

If median \(|I_t(a,b)|/|\Delta_t| > \eta\) for top pairs (default \(\eta=0.10\)), interaction terms are reported explicitly in graph summaries.

## 7. Internal Counterfactual Validation

All interventions are internal (same original prompt), no clean/corrupt prompt pairs required.

### 7.1 Necessity Tests

For a selected set \(S\), ablate outputs at decision step \(t\) (or full forward span if preregistered) and measure:

- Margin drop: \(\delta_t(S)\).
- Token flip indicator: \(\mathbb{1}[\arg\max_k \ell_t^{(-S)}(k) \ne y_t]\).
- Distribution shift: \(D_{KL}(p_t \| p_t^{(-S)})\).

Necessity score:
\[
N_t(S) = \frac{\delta_t(S)}{\max(1,|\Delta_t|)}.
\]

### 7.2 Sufficiency Tests

Define complement ablation: preserve selected set \(S\), ablate \(\mathcal{C}\setminus S\).
Measure:

- Margin retention:
  \[
  R_t(S) = \frac{\Delta_t^{(S\ \text{only})}}{\Delta_t}.
  \]
- Decision preservation:
  \[
  P_t(S) = \mathbb{1}[\arg\max_k \ell_t^{(S\ \text{only})}(k) = y_t].
  \]

### 7.3 Prediction Calibration

Predicted drop from attribution:
\[
\widehat{\delta}_t(S) = \sum_{c \in S} s_{t,c}.
\]

Observed drop:
\[
\delta_t(S) = \Delta_t - \Delta_t^{(-S)}.
\]

Fit calibration model across all tested \(t,S\):
\[
\delta_t(S) = \alpha + \beta \widehat{\delta}_t(S) + \epsilon.
\]
Report \(R^2\), slope \(\beta\), intercept \(\alpha\), MAE.

## 8. Evaluation Tasks and Sampling

Task families:

1. Copy/retrieval.
2. Coreference.
3. Two-hop composition.
4. Simple arithmetic/constraint prompts.
5. Optional refusal/policy behavior.

Per prompt, predefine target decision token(s): first answer token and one key content token where applicable.

Inclusion criteria for token-level evaluation:

- Clean-run target token is top-1 under deterministic decoding.
- Margin floor for stability: \(|\Delta_t| \ge 0.5\) logit units (preregistered).

## 9. Metrics and Acceptance Criteria

| Category | Metric | Definition | Acceptance Criterion |
|---|---|---|---|
| Reconstruction | Residual fidelity | Median \(\varepsilon_t\) | \(\le 0.10\) |
| Completeness | Top-set completeness | Median \(A_t(K_t)\), \(\tau_{\text{abs}}=0.90\) | \(\ge 0.90\) by construction; report \(K_t\) distribution |
| Necessity | Margin removal | Median \(N_t(S_{topK})\) for preregistered \(K\) | \(\ge 0.50\) |
| Necessity | Token impact | Flip rate after ablating \(S_{topK}\) | \(\ge 0.40\) |
| Sufficiency | Margin retention | Median \(R_t(S_{topK})\) | \(\ge 0.60\) |
| Sufficiency | Decision preservation | Mean \(P_t(S_{topK})\) | \(\ge 0.70\) |
| Calibration | Attribution vs intervention | \(R^2, \beta, \alpha, MAE\) | \(R^2 \ge 0.70,\ 0.8 \le \beta \le 1.2,\ |\alpha| \le 0.1\) |
| Specificity control | Null ablations | Effect of random size-matched sets | At least 2x smaller than selected-set effect |
| Interaction | Pairwise non-additivity | Median top-pair \(|I_t|/|\Delta_t|\) | \(\le 0.20\), else report interaction-augmented graph |

`S_topK` is the preregistered selected component set size (default \(K=5\) or adaptive \(K_t\) from Section 5; choose one and fix before runs).

## 10. Statistical Procedure

- Report point estimates with 95% bootstrap CIs (1000 resamples, stratified by task family at prompt level).
- Use two-sided paired tests for selected vs random ablation effects where needed.
- Correct multiple comparisons for per-task subgroup analyses (Holm-Bonferroni).

## 11. Reproducibility and Logging

Required logging for each run:

- Model identifier, checkpoint hash, tokenizer version.
- Precision and device settings.
- Random seeds for sampling/bootstraps.
- Decoding config (temperature, top-p, max tokens).
- Full prompt text and tokenization.
- Selected \(k_t^*\), \(\Delta_t\), \(s_{t,c}\), selected edges, intervention outcomes.

Release artifacts:

1. Prompt suite and split definitions.
2. Raw attribution caches.
3. Intervention result tables.
4. Scripts that regenerate all plots and tables from raw outputs.

## 12. Recommended Default Preregistration Block

Use this fixed block unless a study-specific change is justified:

- \(\tau_{\text{abs}} = 0.90\)
- \(\lambda_{\text{src}} = 0.02\)
- source top-\(m = 8\)
- \(K=5\) for fixed-size necessity/sufficiency tests
- margin inclusion floor \(|\Delta_t| \ge 0.5\)
- 1000 bootstrap resamples
- deterministic decoding, temperature \(=0\)

## 13. Threats to Validity (Methods-Level)

- Basis dependence: feature-level analyses are representation-dependent; report basis choice and stability checks.
- Intervention realism: zero ablation can be out-of-distribution; include mean-replacement sensitivity analysis.
- Residual linearization limits: LayerNorm and nonlinear paths can induce approximation error; bounded by reported reconstruction and calibration metrics.
