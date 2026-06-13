# Research Paper Plan: MaxDiff Decoding

**Working title:** *Free Energy Decoding: Adapting Maximum Diffusion Reinforcement Learning to Autoregressive Token Generation*

---

## 1. Thesis and contribution

**Anchor claim:** MaxDiff RL's free-energy exploration mechanism — designed to decorrelate an embodied agent's *state* trajectories during continuous control — can be transplanted from action selection in continuous control to **token selection in autoregressive decoding**.

**Contribution type:** Theoretical unification / methodological adaptation (an "option (b)" paper). The contribution is the *adaptation itself*: recognizing that autoregressive decoding exhibits the same temporal-correlation pathology MaxDiff RL was built to overcome, and that the minimum-free-energy path objective (Berrueta et al., Eq. 31) transfers to decoding over token embeddings. Concretely, we **implement the MLE-path law of motion** (Berrueta et al., Supp. Note 2.6, Eqs. 37–38) directly in the model's output-embedding space — a *predict-then-project* decoder — with the per-step free-energy objective recovered as its overdamped reduction.

**What this paper is NOT:** It is not primarily a horse-race against existing samplers, and not a rederivation of MaxDiff RL. It references and *uses* MaxDiff RL's results and interpretations; it does not reprove them.

---

## 2. The core mapping (paper centerpiece)

A table making the adaptation concrete and rigorous:

| MaxDiff RL (control) | MaxDiff Decoding (this work) |
|---|---|
| State $x_t$ | Output-embedding state $e_t$ along the trajectory (token context $x_{0:t}$ sets the local geometry) |
| Action $u_t$ | Next token $v$ |
| State increment / velocity $\dot{x}$ | Embedding displacement $\dot{e}_t = e(v_t) - e(v_{t-1})$ |
| Acceleration $\ddot{x}$ — MLE-path law of motion (Eq. 37) | One integration step to a target embedding $e^*_{v_{t+1}}$ |
| Local controllability tensor $C[x^*]$ / Gramian $W$ | Local diffusivity $C(t)$ over output-token embeddings, reshaping the descent |
| Reward / potential $V[x] = -r$ | Surprisal as potential $V = -\log p_\theta(v \mid x_{0:t})$ |
| Temperature $\alpha$ | Exploration/coherence dial $\alpha$ |
| Min free-energy path dist. (Eq. 31) | Per-step free-energy objective (overdamped reduction / baseline) |
| MLE path = mode of $P^V_{\max}$ (Supp. Note 2.6) | Predict-then-project decoder over the token lattice (**primary method**) |

**Per-step objective.** Treat decoding as a particle tracing a path through output-embedding space: each token is a step, the generated sequence is the trajectory. Two quantities govern a step:

- **Potential energy** $V(v) = -\log p_\theta(v \mid x_{0:t})$ — the surprisal. Low potential = high model probability; the particle is pulled "downhill" toward coherent continuations.
- **Kinetic energy** $T(v) = \tfrac{\alpha}{2} \, (e(v) - e(v_{t-1}))^\top C(t)^{-1} (e(v) - e(v_{t-1}))$ — embedding displacement measured in the local diffusion metric $C(t)$.

**Per-step free-energy objective (overdamped reduction / baseline).** Scoring each candidate token by the Hamiltonian / free energy and selecting

$$F(v) = T(v) + V(v) = -\log p_\theta(v \mid x_{0:t}) + \frac{\alpha}{2} \, (e(v) - e(v_{t-1}))^\top C(t)^{-1} (e(v) - e(v_{t-1}))$$

$$v^* = \operatorname*{argmin}_v F(v) \qquad \text{(or sample } v \sim \operatorname{softmax}(-F/T)\text{)}$$

is the **first-order / overdamped** discretization of the path distribution — the Gibbs / Onsager–Machlup per-step weight $\exp(-F)$ with $F = T + V$ (potential = surprisal, kinetic = displacement in the inverse local-diffusivity metric). It is memoryless, lattice-native, and cheap. We keep it as the **baseline** and as the degenerate case the full method must reduce to. The least-action reading ($S = \sum_t L(v_t)$ with $L = T - V$; the most-probable path extremizes the action — the same variational backbone as max caliber) remains the lead intuition. Note the two forms are Legendre-conjugate descriptions of one mechanics, *not* obtained by flipping $+V \to -V$ in a per-step $\operatorname{argmin}$: a literal per-step $\operatorname{argmin}(T - V)$ would *reward* surprisal, the opposite of the goal.

**Primary method — the law of motion in embedding space (inertial limit).** The deeper object in Berrueta et al. is not the per-step weight but the **MLE path** of the minimum-free-energy distribution: the trajectory the diffusing particle would actually trace. It satisfies the second-order law of motion (Supp. Note 2.6, Eq. 37; Eq. 38 is the LTV simplification where $\nabla C \approx 0$):

$$\ddot{x} = -C[x] \cdot \left( \nabla_x V[x] + \tfrac{1}{2} \dot{x}^\top \nabla_x C^{-1}[x] \, \dot{x} \right) \qquad \text{(Eq. 37, full)}$$

$$\ddot{x} = -C[x] \cdot \nabla_x V[x] = -W(t,t_0) \cdot \nabla_x V[x] \qquad \text{(Eq. 38, LTV)}$$

This is **inertial (undamped) gradient descent** with the descent direction reshaped by the local diffusion tensor $C$ (= controllability Gramian $W$). We implement *this* in token-embedding space, instantiating the continuous state as the output embedding — write $x \equiv e$, with $e_t$ the embedding at step $t$, so the law of motion is read in embedding coordinates ($\nabla_x \to \nabla_e$). It carries three forces, each with a decoding meaning:

- **Inertia** ($\dot{e}$ carried across steps): momentum in semantic space — *coherence as the tendency to keep moving in the current direction of meaning*. No damping ⇒ the decoder keeps drifting/exploring rather than settling; add an optional momentum-decay term as an anti-runaway / coherence knob.
- **$-C \nabla_e V$**: descend surprisal, but **only along directions the next-token distribution can actually steer**. Where $C$ is rank-deficient (collapsed distribution) descent in those directions is gated off — controllability made literal: the model descends fluency only along axes it has a foothold on.
- **$-\tfrac{1}{2} C \, \dot{e}^\top \nabla_e C^{-1} \dot{e}$**: drift toward higher local diffusivity (larger $\det C$) — the **anti-degeneration force**, steering away from repetition / mode-collapse traps toward open semantic space. Its potential is exactly the $\tfrac{\alpha}{2} \log \det C$ exploration bonus of main-text Eq. 6.

**Decoder (evolve in embedding space, then project).** The dynamics run **entirely in output-embedding space**: the state is an embedding $e_t \in \mathbb{R}^d$, and integrating the law of motion produces an embedding **trajectory** $E^* = [\, e^*_{v_1} \;\; e^*_{v_2} \;\; \cdots \;\; e^*_{v_T} \,] \in \mathbb{R}^{d \times T}$ — one target embedding per column. Only once the trajectory is formed is it mapped back to token space, applying the embedding→token projection $\Pi$ columnwise.

Because the velocity is reconstructed from a position difference, $\dot{e}_t \approx e(v_t) - e(v_{t-1}) = e_t - e_{t-1}$, the self-consistent integrator is **Störmer / position Verlet**, $e_{t+1} = 2 e_t - e_{t-1} + a_t\,\Delta t^2$, *not* velocity-Verlet. With $\Delta t = 1$, state $e_t = e(v_t)$, and the full acceleration $a_t = \ddot{e}$ from Eq. 37 (read in embedding coordinates), this carries the **full** $a_t$ (coefficient 1), so the two forces enter with coefficients 1 and $\tfrac12$ — *not* $\tfrac12$ and $\tfrac14$. (The $\tfrac12 a\,\Delta t^2$ of velocity-Verlet is correct only for a separately-maintained on-step velocity; pairing it with the displacement proxy $e_t - e_{t-1}$ would cancel the acceleration to leading order, $e^*_{v_{t+1}} \to e_t + \dot{e}_t$, collapsing the decoder to constant-velocity drift.)

$$e^*_{v_{t+1}} = \underbrace{2 e(v_t) - e(v_{t-1})}_{=\, e(v_t) + \dot{e}_t} - C(t) \nabla_e V(e_t) - \tfrac{1}{2} C(t) \, \dot{e}_t^\top \nabla_e C^{-1} \dot{e}_t$$

The projection $\Pi$, applied to each column of $E^*$, maps the embedding trajectory back to tokens:

$$v_{t+1} = \operatorname*{argmin}_v \lVert e(v) - e^*_{v_{t+1}} \rVert^2_{C(t)^{-1}} \qquad \text{(deterministic)}$$

$$v_{t+1} \sim \operatorname{softmax}\!\left[ -\tfrac{1}{2} \lVert e(v) - e^*_{v_{t+1}} \rVert^2_{C(t)^{-1}} \right] \qquad \text{(stochastic)}$$

The stochastic form is the discretized one-step diffusion kernel $p_{\max}(e_{t+1} \mid e_t)$ (Eq. 28) **recentered on the inertial target** $e^*_{v_{t+1}}$; the deterministic decoder is its zero-temperature limit. First implementation: take the LTV form (Eq. 38) — drop the $\nabla C^{-1}$ term, leaving only $\nabla_e V$, which we finite-difference over the candidate set (top-$k$ surprisals as samples of $V$ around $e_t$).

> **Closed-loop caveat.** $C(t)$ and $V$ are recomputed from the realized token context $x_{0:t}$, so the projection $\Pi$ is in practice interleaved step-by-step — decode $v_{t+1}$ from $e^*_{v_{t+1}}$, feed it back, recompute the local geometry — rather than applied to a fully pre-computed trajectory. The "evolve the whole matrix $E^*$ in embedding space, then map back" statement is the clean continuous-limit framing; it is *exact* in the open-loop limit where the local geometry is frozen along the embedding path, and otherwise names the conceptual structure the interleaved decoder realizes.

**Local diffusivity (computed at each step):**

$$\mu(t) = \sum_v p_v \, e(v)$$

$$C(t) = \sum_v p_v \, e(v) e(v)^\top - \mu(t) \mu(t)^\top \qquad \text{(full)}$$

$$C(t) \approx \sum_{v \in \text{top-}k} p_v \, e(v) e(v)^\top - \mu(t) \mu(t)^\top \qquad \text{(truncated, practical)}$$

where $e(v)$ is the **output (unembedding)** vector for token $v$ and $p_v = p_\theta(v \mid x_{0:t})$.

---

## 3. Gating experiment (MUST run before drafting)

The entire paper rests on $C(t)$ being non-trivial. A theory paper claiming "decoding is free-energy minimization" still needs to show the free-energy surface is non-degenerate.

**Checks:**
1. **Eigenspectrum of $C(t)$:** Is it meaningfully far from rank-1, or does probability mass concentrate on near-collinear embeddings?
2. **Step-to-step variation:** Does the eigenstructure of $C(t)$ actually move across generation steps, or is it ~static (which would hollow out the "non-stationary local geometry" claim)?
3. **$\det C(t)$ vs. surprisal regime:** Do high-entropy contexts produce visibly larger $\det C(t)$ than low-entropy contexts?

**Decision rule:**
- Positive → adaptation has an empirical anchor; proceed to drafting.
- Near-degenerate everywhere → method collapses to scalar-surprisal sampling in practice; fall back to a lower-dimensional embedding projection (analogous to MaxDiff's Jacobian-projected $C[y^*]$ for high-dim systems) before reassessing.

**Model:** `Qwen/Qwen3.5-2B` or `Qwen/Qwen3.5-4B` (dense, single consumer GPU, small $d$ for cheap $C(t)$). Use **Base** variants to avoid chat-template effects on the raw next-token distribution. MoE vs. dense is irrelevant to the method (only the final distribution and unembedding matrix are used), so model choice is driven by iteration speed.

---

## 4. Open questions to resolve during the work

1. **How tightly does the analogy hold?** MaxDiff RL's goal is *data acquisition* for learning; decoding's goal is *output quality*. The shared element is the *mechanism* (anti-correlation of sequential trajectories), not the downstream goal. Plan: state a clearly **bounded claim** — adopt the mechanism on the hypothesis that anti-correlation improves output diversity/coherence, explicitly distinct from MaxDiff RL's motivation. Test where it holds and where it strains; report honestly.
2. **Does $C(t)$ carry real information?** Resolved by §3.
3. **Deterministic vs. stochastic selection.** Both fall out of the law-of-motion decoder: deterministic = project to the nearest token to the inertial target $e^*_{v_{t+1}}$; stochastic = sample from $\operatorname{softmax}[-\tfrac{1}{2}\lVert e(v) - e^*_{v_{t+1}}\rVert^2_{C^{-1}}]$ (the Eq. 28 diffusion kernel recentered on $e^*_{v_{t+1}}$), with the deterministic case its zero-temperature limit. Treat as a tunable, not a fundamental distinction.
4. **Overdamped vs. inertial — decided.** We implement the inertial law of motion (Eqs. 37–38) as the primary method; the per-step $\operatorname{argmin} F$ objective is its first-order / overdamped reduction, kept as a baseline. (Previously deferred to §5; now resolved.)
5. **Which state evolves the ODE?** The law of motion assumes a smoothly-varying embedding path $e(t)$. The raw last-token embedding $e(v_t)$ jumps each step; a pooled / decaying-context embedding may behave more like a continuous state. Pin down $e_t = e(v_t)$ vs. a context-pooled $e_t$ empirically — it governs how literally the inertia term applies.

---

## 5. Proposed section structure

1. **Introduction** — Decoding suffers the same pathology MaxDiff RL addresses: sequential, correlated generation prone to degenerate trajectories (repetition, mode collapse, low-diversity ruts). Propose adapting MaxDiff RL's free-energy exploration to decoding. State the bounded claim up front.
2. **Background** — Max caliber (Jaynes → Dixit/Pressé), MaxDiff RL, and the min-free-energy path distribution (Eq. 31). **Reference, do not rederive.** Use their ergodicity/diffusive-exploration results as supporting context.
3. **Method: MaxDiff Decoding** — The mapping table (§2) is the centerpiece. Lead with the **law of motion in embedding space** (Eqs. 37–38) and the predict-then-project decoder as the primary method; present the per-step free-energy / least-action objective as its overdamped reduction and baseline; define $C(t)$ over output embeddings; give deterministic and stochastic variants. Note isotropic $C(t) \propto I$ recovers scalar-surprisal selection as a degenerate case.
4. **Related sampling approaches** — Position each by its *reference object*:
   - **Locally typical sampling** (Meister et al.): scalar reference = step entropy; geometry-blind. Special case of our objective when $C(t)$ is isotropic.
   - **Contrastive search/decoding** (Su et al.): uses embedding geometry (cosine-to-context) but penalizes similarity to past tokens, not displacement weighted by a local covariance.
   - **Nucleus / temperature sampling:** probability-mass / global-entropy references.
   - **Our distinction:** tensor-valued, step-varying $C(t)$ → direction-aware in embedding space. Two tokens with equal surprisal but different embedding directions are treated *differently*.
5. **Theory** — Which MaxDiff guarantees transfer and which don't:
   - Ergodicity is *approximate at best*. Breaks due to (i) irreversible token states (open quote, code block, committed syntax — the text analog of the ant flipping over), (ii) non-stationary $C(t)$ (grows each step; violates stationarity assumption of Theorem 2.2), (iii) discrete vocabulary vs. continuous-state proofs.
   - **Projection ≠ teleport (continuity caveat).** The law of motion lives in continuous embedding space; tokens are a coarse lattice and some displacements $e(v) - e(v_{t-1})$ are large — so the path-continuity constraint that *generates* the diffusion form (Eq. 27) is only loosely satisfied. This is the paper's own end-of-Supp-Note-2.6 caveat about discontinuous / teleporting agents, where the continuity machinery weakens. Strongest when decoding in a smoothly-varying context embedding rather than the raw last-token vector.
   - **No damping → instability.** Eq. 38 is undamped; Störmer/position Verlet on a jagged surprisal landscape can oscillate or diverge. Add the artificial damping the paper notes — it doubles as a coherence dial. Keep damping an *explicit, tunable* coefficient: note that the integrator-coefficient slip (velocity-Verlet's $\tfrac12 a$ paired with a position-difference velocity) would inject ad-hoc half-strength damping by accident — exactly the kind of hidden knob to avoid.
   - Frame honestly: the objective *promotes* ergodic-like behavior by avoiding low-C traps, without inheriting formal guarantees.
6. **Empirical anchor** — Report the §3 $C(t)$ structure results (eigenspectrum, step-to-step variation, $\det C$ vs. surprisal regime).
7. **Discussion / limitations / future work** — Falsifiable predictions; where the RL↔decoding analogy is partial; possible extensions (trajectory-level scoring, projected C for high-dim, online/agentic decoding where non-stationary C is most relevant).

---

## 6. Key references

- Berrueta, Pinosky, Murphey — *Maximum diffusion reinforcement learning*, Nature Machine Intelligence (2024). Primary source; Eq. 31 (min-free-energy path), Supplementary Notes 2.4–2.6 (diffusion, ergodicity, inertial gradient descent).
- Dixit et al. — *Maximum caliber* (variational principle for dynamical systems).
- Jaynes — *Information theory and statistical mechanics* (1957).
- Meister et al. — *Locally typical sampling*.
- Su et al. — *Contrastive search / contrastive decoding*.
- Haarnoja et al. — SAC (MaxEnt RL reference point).

---

## 7. Immediate next steps

1. Load `Qwen3.5-2B-Base`; implement $C(t)$ computation over top-$k$ output embeddings.
2. Run the three §3 checks; log eigenspectra and $\det C$ across ~hundreds of generation steps over varied prompts.
3. Decide proceed / project-to-subspace based on the decision rule.
4. If proceeding: implement the **evolve-then-project decoder** — start with the LTV law of motion (Eq. 38). Per step: compute $C(t)$, finite-difference $\nabla_e V$ over top-$k$, form the target embedding $e^*_{v_{t+1}}$, then select by $\operatorname{argmin}$ (deterministic) or $\operatorname{softmax}$ (stochastic) projection $\Pi$ onto the vocabulary. Validate that it reduces to the per-step free-energy baseline when inertia is disabled.
5. Ablate: inertia on/off, LTV (Eq. 38) vs. full (Eq. 37 with $\nabla C^{-1}$), deterministic vs. stochastic, damping on/off, $e_t = e(v_t)$ vs. context-pooled state. Then draft §2/§3 and build out.
