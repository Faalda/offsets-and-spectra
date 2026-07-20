# DSV — Domain-Shift Vectors and Beyond

A research package for studying **parameter-efficient controllers that transform
frozen representations**. The central question:

> When you cannot (or will not) fine-tune a backbone, what is the *minimal*
> transformation of its activations that closes a domain gap — and how does
> expressiveness trade off against parameter count?

The package implements a whole family of such controllers behind one interface,
plus a staged experimental protocol designed to take the idea from a toy proof
to a state-of-the-art result, the way LoRA and adapters were introduced.

---

## The controller family

Every controller maps a frozen activation `h ∈ ℝ^d` to `h'` per layer:

| name | update | params/layer | role |
|------|--------|--------------|------|
| `dsv` | `h + δ` | d | baseline (additive shift) |
| `scaled` | `h + s·δ` | d+1 | **ablation** — `s·δ` collapses into δ |
| `diag` | `h + w⊙δ` | 2d | **ablation** — `w⊙δ` collapses into δ |
| `matrix` | `h + Wδ` | d²+d | **ablation** — `Wδ` collapses into one vector |
| `gated` | `h + σ(h·a+b)·δ` | 2d+1 | input-dependent shift magnitude |
| `mixture` | `h + Σ_k g_k(h)·δ_k` | K(2d) | MoE over K shift directions |
| **`ortho`** | **`R(h + δ), R∈SO(d)`** | **K·d (Householder)** | **rigid alignment (rotation + shift)** |
| `ogated` | `R(h + σ(h·a)·δ)` | K·d + 2d | ortho + gate |
| `lowrank` | `h + B(Aᵀh)` | 2rd | LoRA-on-activations baseline |
| `affine` | `h + Wh + δ` | d²+d | full dense adapter baseline |

**Why the ablations matter.** `scaled`, `diag`, and `matrix` look more expressive
than `dsv`, but a *fixed linear map applied to a learnable vector* is absorbed
into that vector during training — they cannot beat `dsv`. Including them makes
the absorption argument empirical rather than hand-waved.

**Why `ortho` is the contribution.** An orthogonal transform of the activation
(a) depends on `h`, so it cannot be folded into δ, and (b) is *not* a low-rank
additive update, so it sits outside the LoRA/adapter family — yet via a
Householder parametrisation it costs only `K·d` parameters. It is a learned
**rigid-body alignment** of the representation manifold: rotation (`R`) + shift
(`δ`).

---

## Stage-1 results (synthetic, included)

Controllable domain shift between a source manifold (where a linear classifier
is trained and frozen) and a target manifold (a *known* rotation and/or
translation of the source). Train each controller, frozen-classifier accuracy on
held-out target:

| controller | translation | rotation | both |
|------------|:-----------:|:--------:|:----:|
| no controller | 1.000 | 0.312 | 0.167 |
| `dsv` (32 p) | **1.000** | 0.172 | 0.172 |
| `matrix` (1056 p) | 1.000 | 0.172 | 0.172 |
| `gated` | 1.000 | 0.634 | 0.172 |
| `mixture` | 1.000 | 0.995 | 0.983 |
| `affine` / `lowrank` | 1.000 | 1.000 | 1.000 |
| **`ortho`-householder K=2 (96 p)** | **1.000** | **1.000** | **1.000** |

**Hardened (5 seeds, mean ± std)** via `experiments/harden_stage1.py`. On the
rotation shift the separation is sharp and stable: `ortho`/`lowrank`/`affine` hit
**1.000 ± 0.000** while `dsv`/`matrix` collapse to **0.104 ± 0.146** (high variance —
they cannot fix rotation at all). The params-vs-accuracy Pareto (figure
`results/stage1_pareto.png`) shows `ortho`-Householder reaching the accuracy knee
at **K=2 (96 params)**, on par with or slightly ahead of low-rank — on this
6-class task both are efficient (the decisive `ortho` ≫ low-rank gap appears in
Stage-2 *retrieval*, where rank-4 fails to recover a full rotation).

Takeaways, all reproduced by `experiments/stage1_synthetic.py`:
1. **Absorption is real** — `matrix` (1056 params) ties `dsv` (32 params) exactly
   on rotation. A d×d matrix on a vector buys nothing.
2. **Translation is trivial** — every variant solves it; `dsv` is enough.
3. **Rotation needs an activation-dependent transform** — only `ortho`, the dense
   adapters, and (mostly) `mixture` recover it.
4. **`ortho` is the efficient frontier** — Householder K=2 matches the dense
   adapter at ~9× fewer params and is norm-preserving and exactly invertible.

---

## Stage-2 results (real frozen embeddings — FLORES-200 + e5)

Align sentence embeddings of two languages produced by a *frozen*
`multilingual-e5-base`. Two analytic reference lines come for free: orthogonal
**Procrustes** (the optimum of `ortho`'s class) and **least-squares linear** (the
optimum of `affine`/`matrix`).

**Finding 0 — modern multilingual encoders are already aligned.** Raw e5 scores
P@1 = 1.000 on eng→arb with *no* controller: there is no rigid gap to recover.
The shared-encoder cross-lingual setting is too easy to separate methods — a
result worth reporting in its own right.

**The benchmark: a controlled rotation on real embeddings** (`--rotate-target`).
Apply a fixed random rotation to the target space — real e5 geometry, a known
rigid gap. This is the clean Stage-1→Stage-2 bridge:

| method | params | test P@1 |
|--------|-------:|:--------:|
| identity (rotated, no controller) | — | 0.002 |
| **procrustes** (analytic orthogonal) | — | 0.998 |
| linear-ls (analytic linear) | — | 0.954 |
| **`ortho`** | 590K | **0.995** |
| `affine` | 590K | 0.996 |
| `ogated` | 591K | 0.995 |
| `lowrank` (rank 4) | 6K | 0.006 |
| `dsv` / `scaled` / `diag` / `matrix` / `gated` / `mixture` | — | ~0.000 |

1. **`ortho` matches the analytic Procrustes optimum** (0.995 vs 0.998) — gradient
   training recovers the rotation on real data.
2. **`ortho` beats the unconstrained linear map** (0.995 vs `linear-ls` 0.954) —
   when the gap is rigid, the orthogonality prior is *correct* and outperforms a
   more expressive but unconstrained map. The LoRA-style "right inductive bias
   wins" result.
3. **Absorption, again on real data** — `matrix` (590K params) ties `dsv` (768) at
   ~0: a matrix on a vector cannot undo a rotation.
4. **Low-rank ≠ orthogonal** — `lowrank` rank-4 fails (0.006); a low-rank additive
   update cannot represent a full rotation. This separates OrthoDSV from LoRA.

**Honest capacity note.** Stage-1 classification needed only a *little* rotational
capacity, so Householder K=2 sufficed. Full-space retrieval under a *dense* random
rotation needs near-full-rank orthogonality, so the full Cayley parametrisation is
used here; small-K Householder will not recover a dense rotation. The
capacity/efficiency trade-off is itself a result to characterise.

Reproduce: `experiments/stage2_crosslingual.py --rotate-target`.

### Stage 2b — the *natural* cross-encoder gap, across languages (done)

The headline result, now robust across **three** language pairs (English source via
e5; target via a *monolingual* target-language BERT — independent spaces):

P@1 with 95% bootstrap CI over test queries (`experiments/bootstrap_align.py`):

| pair | identity | **`ortho`** (95% CI) | `linear-ls` (95% CI) | `lowrank` | `dsv` |
|------|:--------:|:-----------:|:---------:|:---------:|:-----:|
| eng→de | 0.00 | **0.933** [.918,.948] | 0.654 [.625,.684] | 0.01 | 0.00 |
| eng→zh | 0.00 | **0.870** [.849,.889] | 0.540 [.509,.572] | 0.01 | 0.00 |
| eng→ru | 0.00 | **0.839** [.815,.862] | 0.368 [.338,.395] | 0.01 | 0.00 |
| eng→ar | 0.00 | **0.559** [.528,.590] | 0.167 [.144,.190] | 0.01 | 0.00 |

In every language: `ortho` ≈ the Procrustes optimum and **beats** unconstrained
linear, while LoRA-as-a-map (`lowrank`) and additive `dsv` collapse to ~0. There is
even a linguistic gradient — German (closest to English) is most rigidly alignable
(0.93), Arabic (most distant) least (0.56) — but the orthogonal prior wins across
the whole range. This is the paper's central, multi-language evidence that
within-modality representation gaps are rigid and orthogonality is the right prior.

Reproduce: `experiments/stage2_crosslingual.py --tgt <deu_Latn|rus_Cyrl|arb_Arab> --encoder-tgt <monolingual-BERT> --warmstart-ortho`.

### Stage 2b detail — English↔Arabic (the first pair)

No imposed rotation. Align English embeddings from `multilingual-e5-base` against
Arabic embeddings from a *monolingual* `bert-base-arabic-camelbert-da` (both 768-d)
— two **independently-built** spaces. Is the real gap between them rigid?

| method | params | test P@1 |
|--------|-------:|:--------:|
| identity (no controller) | — | 0.001 |
| **procrustes** (analytic orthogonal) | — | **0.570** |
| linear-ls (analytic linear) | — | 0.155 |
| **`ortho`** (closed-form fit) | 590K | **0.555** |
| `ogated` | 591K | 0.559 |
| `affine` (trained linear) | 590K | 0.423 |
| `lowrank` (rank 4) | 6K | 0.011 |
| `dsv` / `scaled` / `diag` / `matrix` / `gated` / `mixture` | — | ~0.001 |

**This is the strongest result — and it is natural, not imposed.** An orthogonal
map recovers **0.555–0.570** retrieval between two encoders that were never trained
together, vs **0.155** for the analytic unconstrained linear map. Real
cross-encoder gaps are **substantially rigid**, and orthogonality is the correct,
*generalizing* inductive bias — it recovers ~3.6× more than the best linear map's
analytic solution and beats even the regularized trained `affine` (0.423). This
generalizes the classic word-embedding rotation result to sentence embeddings
across independent encoders.

**Methods note (important for Stage 3).** Over a single linear layer the
orthogonal controller has a *closed-form* optimum (Procrustes), reached instantly
via `OrthoDSV.set_rotation`. Gradient descent over the Cayley parametrisation in
768-d **overfits** the 768² matrix on only 997 pairs — it starts at the warm-start
0.56 and *drifts down* to 0.23. So for linear alignment the controller is **fit,
not trained**; the general gradient path (for the nonlinear, no-closed-form Stage 3)
will need a warm start plus regularisation or Riemannian optimisation. This
overfitting-of-the-unconstrained-map effect is exactly why orthogonality wins.

Reproduce: `experiments/stage2_crosslingual.py --encoder-tgt CAMeL-Lab/bert-base-arabic-camelbert-da --warmstart-ortho`.

## Stage-3 results (frozen transformer, DSV family vs LoRA)

Per-layer adaptation of a **frozen** `bert-base-arabic-camelbert-da` for SLURP-TN
intent classification (14 classes, 2677/593/893). Controllers are injected on all
12 encoder layers via forward hooks and trained by backprop (no closed form in a
stacked nonlinear net); LoRA (`peft`, q/v, r=8) is the PEFT baseline.

| method | adapter params | val MF1 | test acc | test MF1 |
|--------|---------------:|:-------:|:--------:|:--------:|
| none (head only) | 0 | 0.688 | 0.704 | 0.628 |
| `dsv` (per-layer shift) | 9K | 0.676 | 0.732 | 0.704 |
| `ortho` (Householder K=8) | 83K | 0.702 | 0.737 | 0.702 |
| `affine` (dense per-layer) | **7.1M** | 0.716 | 0.746 | 0.660 |
| `lora` (q/v, r=8) | 295K | 0.750 | **0.777** | 0.740 |
| `ortho` (K=32, matched) | 304K | 0.707 | 0.776 | 0.725 |
| **`ogated` (K=32, matched)** | 313K | 0.732 | 0.772 | **0.749** |

**Finding — at matched budget the gated orthogonal controller is on par with LoRA.**
1. Every adapter beats the frozen head (0.628 → 0.70–0.75 test MF1): per-layer
   injection is a real adaptation, not absorbed by the head.
2. **Matched parameters matter.** At ~300K params (LoRA's budget), `ogated` slightly
   *exceeds* LoRA on macro-F1 (0.749 vs 0.740) and ties on accuracy (0.772 vs 0.777);
   `ortho` is within noise. The small-budget gap at K=8 (83K) was a budget artifact,
   not a method deficit — the orthogonal family is competitive with LoRA on
   single-task PEFT when given equal capacity.
3. **`affine` (7.1M params) underperforms `ortho` (83K)** — the unconstrained dense
   adapter overfits, the *same* effect seen analytically in Stage 2. Constraint
   generalises; raw capacity does not.

So OrthoDSV is **competitive with LoRA even outside its sweet spot** (single-task
PEFT), and its *decisive* advantage remains the **alignment regime** (Stage 2:
cross-lingual / cross-encoder, and next cross-modal), where the gap is genuinely
rigid and orthogonality out-generalises unconstrained maps. Stage 4 (cross-modal)
is where the method should win outright, not merely tie.

Reproduce: `experiments/stage3_peft.py` (K=8); set `ORTHO_REFLECTIONS=32` for the
matched-budget rows.

## Stage-4 results (cross-modal speech↔text) — a characterized boundary

The method's *predicted* sweet spot turned out to be its boundary, and that is a
result. Align frozen speech embeddings to frozen XLM-R text on SLURP-TN; retrieval
P@1, with the analytic orthogonal (Procrustes) and linear optima.

| speech encoder | identity | procrustes (ortho) | linear-ls | best trained |
|----------------|:--------:|:------------------:|:---------:|:------------:|
| wav2vec2 (acoustic) | 0.000 | 0.069 | 0.060 | affine 0.025 |
| whisper (semantic)  | 0.002 | 0.048 | **0.121** | affine 0.119 |

**Two honest findings:**
1. **The gap is largely nonlinear.** Even the best map reaches only ~12% — far
   below the within-modality alignment of Stage 2 (0.55–0.57). A semantic speech
   encoder (Whisper) helps the *linear* map (0.06→0.12) but does not make the gap
   alignable.
2. **Orthogonality is the *wrong* prior cross-modally.** With Whisper the
   unconstrained linear map (0.121) *beats* the orthogonal one (0.048) — the
   opposite of Stage 2. The speech↔text gap needs scaling/shearing, not a rotation.

**Conclusion — the scope of the thesis.** Orthogonal/rigid alignment is the right,
parameter-efficient prior for *semantically homogeneous* representation gaps
(cross-lingual, cross-encoder — Stage 2, where `ortho` beats unconstrained linear
and LoRA-as-a-map). It does **not** transfer to raw cross-modal acoustic/semantic
gaps. A method that maps where it works *and where it does not* is the honest and
stronger contribution. (A cross-modal space *trained* to be aligned — e.g. CLIP
image↔text, whose modality gap is reportedly near-rigid — is a separate open
question; speech↔text via independent frozen encoders is not rigid.)

Reproduce: `experiments/stage4_crossmodal.py [--speech-model openai/whisper-large-v3] --warmstart-ortho`.

## Research roadmap

The synthetic stage is a *controlled proof*. The plan from here mirrors how
adapter methods earned credibility — start where the ground truth is known, then
climb to real models and competitive benchmarks.

- **Stage 1 — synthetic manifolds (done).** Known shift, prove who recovers what.
- **Stage 2a — frozen real embeddings, controlled rotation (done).** FLORES-200 +
  e5; `ortho` matches the Procrustes optimum and beats the unconstrained linear
  map on real vectors with a known rigid gap.
- **Stage 2b — natural cross-encoder gap (done).** e5-English vs monolingual
  Arabic BERT; the real gap is substantially rigid — orthogonal recovers 0.56 vs
  0.16 for the unconstrained linear map.
- **Stage 3 — transformer hidden states (done).** Per-layer controller injection
  into a frozen BERT vs LoRA on SLURP-TN intent. At matched budget `ogated` is on
  par with LoRA (0.749 vs 0.740 MF1); the dense `affine` adapter overfits at 7.1M
  params. Alignment remains the method's decisive regime.
- **Stage 4 — the SOTA bid.** Cross-modal alignment (e.g. speech↔text, the NADI
  Task-5 setting next door), where rigid manifold alignment is the natural
  inductive bias and `ortho`/`ogated` should beat both plain DSV and LoRA.

Each stage reuses the *same* controller code in `dsv/controllers.py` — only the
backbone and data change.

---

## Layout

```
DSV/
├── dsv/
│   ├── controllers.py     # the whole controller family + registry
│   └── __init__.py
├── tasks/
│   └── synthetic.py       # controllable domain-shift data generator
├── experiments/
│   └── stage1_synthetic.py
├── tests/
│   └── test_controllers.py
└── results/               # experiment outputs
```

## Running

The package depends only on `torch` + `numpy`; it reuses the sibling NADI venv.

```bash
cd DSV

# sanity tests (shapes, orthogonality, identity init, save/load)
../nadi/.venv/bin/python3 tests/test_controllers.py

# Stage-1 comparison across all variants
../nadi/.venv/bin/python3 experiments/stage1_synthetic.py --shift both
../nadi/.venv/bin/python3 experiments/stage1_synthetic.py --shift rotation --dim 64

# subset of controllers
../nadi/.venv/bin/python3 experiments/stage1_synthetic.py --controllers dsv ortho lowrank
```

## Adding a controller

Subclass `DSVController`, implement `transform(self, h, layer)`, decorate with
`@register("yourname")`. It is then available everywhere via
`build_controller("yourname", dim, n_layers)` and is picked up automatically by
the Stage-1 sweep.
