# Auditing Representation-Space Fraud Detection: A Bayesian IRL Framework for Calibrated Uncertainty and Its Limits

Shivam Swarup   
JAIN University (Deemed-to-be) Bengaluru, India   
shivam@jainuniversity.ac.in

Divya Prakash Shrivastava Zayed University Dubai, UAE Divya.shrivastava@zu.ac.ae

Rakesh Thakur JAIN University (Deemed-to-be) Bengaluru, India

## Abstract

Shilling detectors for recommender systems are almost always evaluated by one number, detection AUROC, with no account of when it can be trusted. We audit representation-space detection with VEIL (Variational Evaluation via Inverse-reinforcement Learning), a pairwise (Bradley–Terry) Bayesian reward-inference framework over frozen encoder embeddings, after the Bayesian auditing paradigm for LLM alignment [1]: (A) a base posterior; (B) a hierarchica ARD extension with a learned confidence; and (C) a label-routed organic/attacker mixture used as an evidence-accumulation check. We prove (Prop. 1) that no detector built on a representation invariant to some transformation of user profiles can beat chance against an attack indistinguishable from organic profiles under that invariance. Detection is near-perfect against synthetic-structure attacks (AUROC 0.936–1.000) but collapses under donor-mimicry, which clones a real user’s item set and splices in 1–3 target items: to chance on MovieLens-1M (0.48–0.51), partially on Amazon Musical Instruments, and least on CiaoDVD. Donor-mimicry is not undetectable: a non-learned baseline, each profile’s maximum raw item-set Jaccard similarity to any other, reaches AUROC 0.92–0.97 (average precision 0.50–0.57) on all four donor-mimicry configurations, including ML-1M, where every embedding-based score is at chance. The signal is in the raw profiles; LightGCN’s representation discards it, placing injected profiles closer to their true donor than its zero-propagation MF reduction does (distance 1.98 vs. 3.09). Severity varies by dataset, encoder, and detection method, with no one axis predicting the others. The limit lies in the representation, not the attack, and raw-similarity checks are a necessary complement to any learned detector.

Keywords: Shilling attack detection; recommender systems; Bayesian inverse reinforcement learning; calibrated uncertainty; representation invariance; graph neural networks

## 1 Introduction

A shilling-detection paper is, almost without exception, evaluated by a single number: detection AUROC on whatever attacks the authors chose to inject. That number says nothing about when it can be trusted: whether confidence tracks the actual dificulty of the attack, or whether a slightly more realistic attack would collapse it to chance unnoticed. Table 1 shows exactly this in our own pipeline: every synthetic and GAN-generated attack we test – classic random/average/bandwagon injection, AUSH, and AUSH with per-profile rating-style heterogenization – is detected at AUROC 0.936–1.000 regardless of which of our three inference components is used, and a single change to the attack (cloning a real user’s item-set and rating history and splicing in only 1–3 target items) drives detection to AUROC 0.48–0.51: exact chance, across all three components and an 8-seed sweep. A detector reporting only the first number would have shipped, unqualified, a system blind to the second attack.

This gap is not primarily a modeling problem to be closed with a better detector; it is structural. Proposition 1 proves that any detector, however trained or complex, operating on a representation invariant to some transformation of user profiles cannot beat chance once an attack’s profiles are indistinguishable under that invariance – a property of the representation, not a claim that the attack itself is undetectable (§6.6). This holds for every classifier in this paper, and for the widely-used label-free encoder we test (§6.5) it is not hypothetical: donor-mimicry attacks, constructed to approximately satisfy the proposition’s hypothesis, drive detection from near-ceiling to exact chance. Nor is severity predictable from any single axis of comparison: §6.8 shows it varies by dataset, encoder architecture, and detection method, with a ranking on one axis not transferring once a second also varies. Given a failure mode that is provable, empirically confirmed, and not bounded by checking any one dimension, the right response is not a detector claiming to have solved it but an auditing framework that reports calibrated confidence in its own detections and is checked against exactly these axes. VEIL, our BIRL-style auditing framework, is built for that job. Its three-stage structure – quantify ambiguity in a posterior, audit that posterior’s trustworthiness via calibrated epistemic uncertainty, validate the audit against evidence accumulation – follows the shift of Bou et al.’s Alignment Auditor for verifying LLM objectives [1], which we adapt to shilling detection: “is this user manipulating the system” becomes a Bayesian reward to infer, not a label to predict.

Concretely, we fit a static pairwise Bradley–Terry reward model by variational inference over frozen recommender embeddings (§4) in three components: (A) a base posterior with fixed confidence; (B) a hierarchical extension with per-dimension relevance priors and a learned confidence α as the calibrated-uncertainty audit signal; and (C) a two-branch organic/attacker mixture whose attackerbranch posterior we watch contract as evidence accumulates. Component (C) routes profiles by their labels, so it is an auditing experiment, not an unsupervised detector (a label-free attempt yields only a weakly attack-related partition; Appendix A). We run all three on a supervised GNN (Sup-GNN, trained on the attack labels: a circular positive control) and a label-free LightGCN (the honest test).

Running VEIL end to end surfaces three findings, plus a validation check: Component C’s attacker-branch posterior contracts monotonically as injected-profile evidence accumulates, decoupled from point-estimate accuracy (§6.4) – expected Bayesian behavior, confirming the machinery works rather than a finding in its own right. First, and unexpected: Component B’s learned α varies across the attack-dificulty ladder but not as a clean “confident when easy, unsure when hard” dial – lowest on the easiest configurations, peaking on the hardest attack still well above chance, and intermediate once detection has collapsed (§6.3). Second, the donor-mimicry collapse has a specific, traceable mechanism on our label-free encoder: LightGCN’s adjacency construction and BPR loss never read a rating value, only item identity (§6.5), so no realism added to an attack’s rating values could have mattered. This is not a new vulnerability class [2–5]; our contribution is a mechanistic, calibrated account of where and why that limit bites a BIRL-style auditing framework, not its discovery. Third, the collapse is not a property of ML-1M, of LightGCN, or of VEIL’s detection method alone: it is severe on ML-1M, partial on Musical Instruments, and largely absent on CiaoDVD (§6.7); the matrix-factorization reduction of LightGCN is more robust than LightGCN on ML-1M’s hardest point but less robust on both CiaoDVD points; and VEIL beats a classical RDMA/WDMA baseline on CiaoDVD but is comparable to it on ML-1M, where both collapse (§6.8). No one axis predicts the others, which is the empirical counterpart to Proposition 1: a detectability limit whose severity cannot be assumed to transfer without checking.

## 2 Related Work

Shilling attacks and detection. Profile-injection attacks on collaborative filtering were first demonstrated by Lam and Riedl [6]; Random, Average, and Bandwagon were formalized, together with an analysis of algorithm robustness against them, by Mobasher et al. [7]. Classical detectors classify users from hand-crafted rating-statistics features such as RDMA and WDMA [8], which we use as a baseline (§6.8). Attack generation has since moved to GAN-based profile synthesis (AUSH; [9]), GAN-plus-graph-convolution attacks [10], and difusion-based generators [11]. Recent detectors include multi-view cross-attention [12], credibility-weighted matrix factorization [13], federated [14], graph-convolutional group [15], collusive-behavior group [16], and temporal/relational [17] methods. Our concern is complementary to any one of these: not a new detector, but when a detector’s reported accuracy can be trusted.

Bayesian inverse reinforcement learning. Component B’s learned confidence and perdimension ARD structure follow the hierarchical-prior direction of Choi and Kim [18], reduced here to an empirical-Bayes/ARD treatment with no MDP, Q-function, or policy solving (§3). Imani and Ghoreishi [19] address scalability of Bayesian IRL, and Hossain et al. [20] quantify how much behavioral data a Bayesian IRL model needs, the sample-size analogue of the posterior-ambiguity question Component A asks.

Sybil defense and profile-copying attacks. The donor-mimicry result (§6) confirms a known limit class rather than introducing a new attack. SybilGuard and SybilLimit [2, 3] are the foundational social-graph Sybil defenses. CopyAttack [4] is the closest analogue to donor-mimicry in the recommender-attack literature: it copies real user profiles across domains to evade detectors that expect synthetic structure, the mechanism our donor-mimicry attack uses within a single domain (§6.5). RecUP [5] is a GAN-based poisoning attack designed so injected profiles’ summary statistics stay indistinguishable from organic ones, the same evasion goal donor-mimicry pursues by copying real profiles outright rather than approximating their statistics.

Bayesian uncertainty in recommender systems. To our knowledge, calibrated posterior inference has not been applied to shilling detection or attacker-profile identification; the closest work, Chen et al. [21], learns reward functions adversarially but does not quantify uncertainty. We flag this as a gap in our literature search, not evidence that no such work exists.

Auditing frameworks. The Alignment Auditor [1] verifies LLM objectives by inferring a Bayesian posterior over them and auditing that posterior’s uncertainty. Our three-stage structure (quantify a posterior, audit its calibrated uncertainty, validate against evidence accumulation) follows that template (§4); what we add is the setting, the invariance result of §6.6, and the cross-axis empirical checks, not the template itself.

## 3 Preliminaries / Problem Setup

We adopt a static pairwise Bradley–Terry reformulation over frozen embeddings rather than a sequential MDP-IRL formulation, because MovieLens-1M [22] interaction logs are not agent trajectories in any sense that supports a clean identifiability argument: there is no well-defined action sequence, state transition, or “demonstrator policy” to invert. Casting the problem as static preference inference over paired profile embeddings avoids that objection, at the declared cost (§7) of not modeling how ratings accrue over time.

Formally: for each profile x (organic or injected) we have a frozen embedding $\phi ( x ) \in \mathbb { R } ^ { 6 4 }$ (LightGCN) or R<sup>64</sup> <sup>or</sup> <sup>128</sup> (Sup-GNN raw HGT / post-fusion). We form pairs $( x ^ { + } , x ^ { - } )$ of matched organic/injected profiles (random or k-NN matching) and posit a Bradley–Terry preference likelihood

$$
P ( x ^ { + } \succ x ^ { - } \mid \theta ) = \sigma { \Big ( } \alpha \cdot \theta ^ { \top } { \big ( } \phi ( x ^ { + } ) - \phi ( x ^ { - } ) { \big ) } { \Big ) }\tag{1}
$$

over a linear reward $r _ { \theta } ( x ) = \theta ^ { \top } \phi ( x )$ : a reward-inference problem with no MDP, Q-function, or policy solving. Without an MDP, this reduces to Bayesian logistic regression on embedding diferences; we retain the IRL terminology because the structure follows the Bayesian reward-inference auditing template of Bou et al. [1], not because we solve a sequential IRL problem.

## 4 VEIL – Components A/B/C as Stages 1–3

Stage 0 (shared). Extract frozen $\phi$ for every profile from one of two encoders: a Sup-GNN detector (trained with focal loss on the injected labels, so ϕ already encodes organic-vs-injected, making it a circular positive-control encoder), or a LightGCN recommender, trained purely with BPR ranking loss over the poisoned bipartite graph and blind to the organic/injected label by construction (§6.5 gives the mechanism). Standardize with train-pool mean/std and cache.

Stage 1 – Component A: quantify ambiguity. Fixed-confidence pairwise BIRL. Prior $\theta \sim$ $\mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I )$ , mean-field posterior $q ( \theta ) = \mathcal { N } ( \mu , \mathrm { d i a g } ( \sigma ^ { 2 } ) )$ , fit by the reparameterization-trick ELBO

$$
\begin{array} { r } { \mathbb { E } _ { q } \Big [ \sum \log \sigma \big ( \alpha \theta ^ { \top } \Delta \phi \big ) \Big ] - \mathrm { K L } \big ( q \| \mathcal { N } ( 0 , \sigma _ { 0 } ^ { 2 } I ) \big ) } \end{array}\tag{2}
$$

with $\alpha \equiv 1$ fixed and minibatch Adam (settings in Appendix J). The posterior’s marginal covariance log-determinant and $\mu , \sigma$ spread (per config in Appendix D) quantify the ambiguity: how tightly the evidence pins down the reward direction.

Stage 2 – Component B: audit trustworthiness via calibrated epistemic uncertainty. Extends A with a learned confidence $\alpha \sim \mathrm { G a m m a } ( \tau _ { 1 } , \tau _ { 2 } )$ and per-dimension ARD precisions $\lambda _ { d } \sim \mathrm { G a m m a } ( \gamma _ { 1 } , \gamma _ { 2 } )$ shaping $\mu _ { d } \sim \mathcal { N } ( 0 , ( \beta \lambda _ { d } ) ^ { - 1 } )$ , jointly optimized on the extended ELBO and warm-started from A’s posterior (hyperparameters: Appendix J). The learned α is the audit signal: how much of the pairwise log-odds the model will stake its confidence on, a variational posterior quantity rather than a fixed hyperparameter. §6.3 reports what it tracks across the ladder: a real but non-monotonic pattern.

Stage 2/3 – Component C: posterior contraction as evidence accumulates. Ground-truth labels route profiles into two independent B-style branches, organic-anchored and attacker-anchored, each fit exactly as B. Held-out detection compares branch reward projections $r _ { \mathrm { a t t } } ( x ) - r _ { \mathrm { o r g } } ( x )$

calibrated to P(injected) by a 1-D logistic fit on the train split. Because routing uses the labels, Component C is an auditing experiment, not an unsupervised detector; an inferred responsibility q(z<sub>i</sub>) converges stably but to a partition only weakly tied to the attackers (Appendix A). §6.4 checks that the attacker branch’s posterior (looser than the organic branch’s, as the minority class) contracts monotonically with more injected-profile evidence, as correct Bayesian behavior requires. Algorithm boxes: Appendix K.

## 5 Experimental Setup

• Dataset. MovieLens-1M [22], 5-core filtered (999,611 interactions after filtering, from 1,000,209 raw), subsampled to 30% of users before injection.

• Encoders. Sup-GNN – supervised, trained on injected labels via focal loss; a positive control, expected to be circular. LightGCN – label-free, BPR ranking loss only; the honest baseline. Both were trained from scratch: Sup-GNN 20 epochs, LightGCN 30 BPR epochs, dim 64, 3 layers.

• Attack ladder (5 run configs). Five configs of increasing dificulty, sharing Sup-GNN/ LightGCN encoders: classic random + average + bandwagon injection; AUSH (GAN-generated profiles); AUSH wrapped in a heterogenization attack (organic-matched profile/filler/rating-style variation); the same plus donor-mimicry (each fake profile copies a real organic user’s item-set and ratings, splicing in 1–3 target items); and an 8-seed sweep repeating the donor-mimicry config to quantify variance at the collapse point. Results in Table 1; per-config parameters in Appendix J.

• Compute. CPU-only, Intel i3-1115G4 (2 cores / 4 threads), 8 GB RAM, no GPU. Wall-clock per stage: Appendix I.

• Metrics. Pairwise accuracy, Brier, ECE on held-out preference pairs; AUROC, AP, accuracy, F1/precision/recall on held-out organic-vs-injected detection; ELBO and posterior log-determinant as ambiguity/contraction diagnostics.

## 6 Results and Discussion

## 6.1 Full attack-dificulty ladder

All values are held-out detection metrics (Components A/B) or combined detection metrics (Component C), rounded to 3 significant figures for AUROC/AP and 2 decimals for α; the supplementary materials give full-precision values.

Structural/synthetic attacks (rows 1–4): AUROC 0.936–1.000. The floor across A/B/C on this segment is 0.936 (Component C, lightgcn\_aush\_het), driven by the more sophisticated heterogenized attack – not a uniform “regardless of sophistication” ceiling.

Donor-mimicry (row 5): collapses to chance. At seed 42, Component A AUROC is 0.500 and AP 0.051 (base rate ≈ 27/571 ≈ 4.7% positive class, so AP near the base rate is exactly the chance-level signature). Across the full 8-seed sweep, AUROC (mean ± population std) is 0.494 ± 0.061 (A), 0.488 ± 0.062 (B), and 0.481 ± 0.058 (C).

<table><tr><td>config</td><td>attack</td><td>A AUROC</td><td>A AP</td><td>B AUROC</td><td>Bα</td><td>C AUROC</td><td> $\mathrm { ~ C ~ } \alpha _ { \mathrm { o r g } } / \alpha _ { \mathrm { a t t } }$ </td></tr><tr><td>supgnn_base</td><td>random+avg+bw</td><td>0.998</td><td>0.961</td><td>0.998</td><td>3.22</td><td>0.998</td><td>3.24 / 3.00</td></tr><tr><td>lightgcn_base</td><td>random+avg+bw</td><td>0.999</td><td>0.995</td><td>1.000</td><td>0.42</td><td>1.000</td><td>0.47 / 0.30</td></tr><tr><td>lightgcn_aush</td><td>AUSH</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.40</td><td>1.000</td><td>0.69 / 0.31</td></tr><tr><td>lightgcn_aush_het</td><td>AUSH+Het.</td><td>0.942</td><td>0.423</td><td>0.944</td><td>0.70</td><td>0.936</td><td> $0 . 8 3 \mathrm { ~ / ~ } 0 . 4 9$ </td></tr><tr><td>lightgcn_aush_het_donor</td><td>+Donor-mimicry</td><td>0.500</td><td>0.051</td><td>0.513</td><td>0.54</td><td>0.507</td><td>0.54 / 0.39</td></tr></table>

Table 1: Attack-dificulty ladder, LightGCN honest baseline + Sup-GNN circular control; all columns are seed-42 runs (8-seed α means: §6.3; donor-mimicry 8-seed AUROCs: text above).
<table><tr><td>config</td><td>AUROC (95% CI)</td><td>ECE (95% CI)</td></tr><tr><td>lightgcn_base (easy)</td><td>1.000 [0.999, 1.000]</td><td>0.006 [0.004, 0.008]</td></tr><tr><td>lightgcn_aush_het</td><td>0.945 [0.932, 0.958]</td><td>0.090 [0.082, 0.099]</td></tr><tr><td>CiaoDVD (easy)</td><td>0.877 [0.866, 0.888]</td><td>0.177 [0.171, 0.184]</td></tr><tr><td>Musical Instruments (hard)</td><td>0.805 [0.797, 0.814]</td><td>0.263 [0.259, 0.268]</td></tr><tr><td>lightgcn_aush_het_donor (hard)</td><td>0.494 [0.440, 0.549]</td><td>0.374 [0.360, 0.388]</td></tr></table>

Table 2: Component A, profile-level AUROC and ECE, 8-seed mean [95% CI], five of ten swept configs (the rest: Appendix C).

## 6.2 Calibration tracks (and sharpens) the same collapse

AUROC alone cannot say whether a detector’s confidence is trustworthy where it fails. We run an 8-seed sweep on all ten configs (five ladder points, five cross-dataset points of §6.7) and report profile-level Brier, NLL, and ECE via the same 1-D logistic calibration Component C uses (§4), with Student-t 95% CIs. Table 2 shows a subset for Component A; all configs and reliability bins: Appendix C.

ECE moves in lockstep with AUROC and amplifies the story: every swept config with AUROC $\geq 0 . 9 4$ has $\mathrm { E C E } \leq 0 . 0 9$ , while every config below 0.88 has ECE 0.18–0.37 – the detector is not merely wrong at the hard end, it is confidently wrong. The reliability diagrams (Figure 1, Appendix C) show this: the easy config sits near the diagonal’s extremes, the partial-collapse config is uniformly below the diagonal (overconfident), and the full-collapse config is flat near the base rate – the calibrated probability carries no information, as Proposition 1 predicts.

## 6.3 Component B’s learned confidence α – non-monotonic, flagged

Within the LightGCN honest-baseline ladder (8-seed mean ± SE), α is lowest on the two easiest configs (lightgcn\_base 0.386±0.006, lightgcn\_aush 0.329±0.021) and highest on the hardest non-collapsed config $( 0 . 6 8 8 \pm 0 . 0 2 2$ on lightgcn\_aush\_het), then drops back to an intermediate $0 . 5 5 1 \pm 0 . 0 3 0$ where detection has fully collapsed to chance. This is an inverted-U over the four ladder points, peaking at the hard-but-still-separable attack rather than the collapsed one; the peak’s 95% CI ([0.637, 0.739]) overlaps neither the collapsed point’s ([0.480, 0.623]) nor the easy points’, though the ordering of the two easy points is not resolved (their CIs overlap slightly). A plausible, post-hoc reading: α tracks how much pairwise log-odds signal remains for the model to stake weight on, which is not monotonic in AUROC-defined dificulty. The spreads cover inference seeds over one trained encoder per config only (§7).

## 6.4 Component C – posterior contraction as a validation check

A sanity check, not a novel finding: posterior contraction under more evidence is what a correctlyimplemented Bayesian model should do. In an evidence-count ablation on the base LightGCN config (full table: Appendix F, Table 14), from $n _ { \mathrm { i n j e c t e d } } = 2 0 \ \mathrm { t o \ 1 9 1 }$ , log |Σ| falls monotonically (78.49 → 15.95), mean σ falls (1.858 → 1.139), and α rises $( 0 . 1 8 3  0 . 2 9 7 )$ , while branch-detection AUROC stays saturated (0.999 → 1.000): the contraction reflects how confident the posterior is entitled to be, decoupled from whether the point estimate already detects well.

Does this require ground-truth routing? We tested replacing label-based routing with a fully label-free stochastic variational-EM procedure (Appendix A). The runs converge to one partition with roles swapped (label switching, unresolvable by the ELBO); a label-free rule (smaller cluster = attacker) gives a stable post-hoc AUROC of about 0.61, but the partition agrees only weakly with the true labels (ARI 0.05): a partial capability, not a drop-in replacement.

## 6.5 Why LightGCN goes blind under donor-mimicry: a mechanistic account

Two functions in src/models/lightgcn.py explain why: the adjacency builder collapses every (user, item) pair to one unweighted edge regardless of rating value, and the BPR loss takes only item IDs, never a rating tensor, so rating value is structurally never read (line-level citations in Appendix J). The LightGCN embedding is therefore a function purely of which items a profile rated (and, through propagation, which other users share them), never how they rated them. Donormimicry exploits exactly this: it copies a real organic user’s full item-set verbatim and splices in only 1–3 target items, so the profile is, on the only axis LightGCN can see, nearly indistinguishable from an organic one. The rating-value levers (per-profile style, filler noise, target-rating jitter) that made lightgcn\_aush\_het harder than lightgcn\_aush do not move LightGCN separability beyond what item-set structure gives; donor-mimicry is the first lever that also perturbs the item-set axis, which is why it – and only it – collapses detection.

This is not a novel discovery. That representation-space Sybil/shilling detectors are vulnerable to attacks that copy real user behavior is well established [2–5]; our contribution is showing precisely, down to the two functions above, why this encoder is blind to it.

Direct measurement, not just code inspection. We also tested the propagation-depth account directly. For each donor-mimicry-injected profile in the ML-1M hard-ladder point we recovered its true donor (not stored by the attack; reconstructed by item-set Jaccard overlap, median 0.96, with 81 of 91 profiles confidently matched) and compared injected-to-donor embedding distance under LightGCN and under its $n _ { \mathrm { l a y e r s } } = 0$ reduction (matrix factorization; §6.8) on the identical poisoned dataset, with no retraining. LightGCN places injected profiles reliably closer to their true donor than MF does (mean Euclidean distance 1.98 vs. 3.09; paired diference 1.11, n = 81). SEMs, cosine similarities, and a random-organic-user control, which rules out a general embedding-scale diference between the two fits, are in Appendix G. This is consistent with, though on a single config not suficient alone to establish causally, the propagation-depth account: deeper message passing appears to smooth an injected profile toward its donor, erasing the residual signal Proposition 1 identifies as the only thing donor-mimicry leaves behind. The same architecture signature – LightGCN, the only propagating encoder among four tested, scoring lowest – reappears on a second dataset, Musical Instruments, once two non-propagating sequential encoders are added alongside MF (§6.8). The donor closeness visible here in an embedding is visible without one too: raw item-set Jaccard detects these profiles directly, AUROC 0.92–0.97 where LightGCN collapses $( \mathrm { A p p e n d i x \ : B } ) - \mathrm { a }$ limit of what LightGCN discards, not of the attack (§6.6).

## 6.6 Formalizing the invariance argument

§6.5 shows, at the level of two specific functions, that LightGCN’s representation is blind to rating values. We now state the general property this instantiates, so the mechanism is legible as a property of any representation with the same invariance, not an artifact of one implementation.

Let X be the space of user profiles and $\pi : X  S$ a map onto the “structural signature” a representation actually consumes – for LightGCN, $\pi ( x )$ is the deduplicated (item, co-occurrence) structure of x’s ratings, discarding rating values entirely (exactly what build\_norm\_adj/bpr\_loss do; §6.5). Say a representation $\phi : X \to \mathbb { R } ^ { d }$ is π-invariant if it factors as $\phi = \phi \circ \pi$ for some $\bar { \phi } : S \to \mathbb { R } ^ { d }$ , i.e. ϕ(x) depends on x only through $\pi ( x )$

Proposition 1 (Representation-invariance undetectability). Let $\phi$ be π-invariant, $Y \in \{ 0 , 1 \}$ the organic/attacker label, and $D : \mathbb { R } ^ { d }  \mathbb { R }$ any measurable detector score (this includes every detector in this paper – Components $A / B / C ^ { \prime } s$ reward projections – and any other function of ϕ(X) whatsoever, linear or not, Bayesian or not). If an attack produces profiles X such that $\pi ( X ) \mid Y = 1$ and $\pi ( X ) \mid Y = 0$ are identically distributed on the attack’s support, then $D ( \phi ( X ) )$ has identical distribution under $Y { = } 1$ and $Y { = } 0$ on that support, and D achieves $A U R O C = 0 . 5$ exactly, regardless of D’s complexity or how it was $\it { \Omega } \mathcal { f } t .$

Proof sketch. $\phi = \bar { \phi } \circ \pi$ is deterministic, so pushing the (by hypothesis, equal) conditional distributions of $\pi ( X )$ under $Y { = } 0$ and $Y { = } 1$ through ϕ<sup>¯</sup> gives $\phi ( X ) \mid Y = 1 \stackrel { d } { = } \phi ( X ) \mid Y = 0 , \mathrm { i . e . } \ \phi ( X ) \perp Y$ on the attack’s support, so $I ( Y ; \phi ( X ) ) = 0$ . By the data-processing inequality, for any (possibly stochastic) D, $I ( Y ; D ( \phi ( X ) ) ) \leq I ( Y ; \phi ( X ) ) = 0$ , and mutual information is non-negative, so $I ( Y ; D ( \phi ( X ) ) ) = 0 ;$ : the score $D ( \phi ( X ) )$ is statistically independent of $Y$ , hence identically distributed under both labels. $\mathrm { A U R O C } = P ( { \mathrm { s c o r e } } _ { 1 } > { \mathrm { s c o r e } } _ { 0 } ) + { \textstyle { \frac { 1 } { 2 } } } P ( \mathrm { t i e } )$ for two independent draws from the same distribution is exactly 0.5 by the symmetry of that draw. □

Corollary 1. The bound in Proposition 1 is architecture-agnostic: it holds for Components $A / B / C ,$ for the auditing machinery built around them, and equally for any other classifier trained on $\phi ( X ) \textrm { -- }$ calibration, uncertainty quantification, or ensembling cannot recover signal the representation never carried.

What this does and does not claim. The hypothesis – that an attack achieves exactly matching structural distributions – is an existence claim about attacks, not a property of every attack. Donormimicry approximately satisfies it: it copies a real user’s full item-set verbatim and splices in only 1–3 target items (§6.5), so $\pi ( X )$ matches an organic profile’s on every coordinate except those few items. The proposition therefore predicts $\mathrm { A U R O C }  0 . 5$ as the splice fraction vanishes, not that every run hits exactly 0.5: the shared target items across attacker profiles are a small residua violation, consistent with results that are close to but not always at chance (0.48–0.54 across configs and datasets, §6.1, §6.7; exactly 0.500 only at seed 42 on the original ladder). We report the proposition as an idealized limiting case that the empirical mechanism approaches, not one donor-mimicry achieves exactly.

The bound is about the representation, not the attack. Proposition 1 bounds detectors that are functions of a π-invariant $\phi ,$ not a detector on the raw profile, which need not share $\phi \mathrm { { s } }$ invariance. Raw item-set Jaccard similarity – no embedding – detects donor-mimicry at AUROC

<table><tr><td>Dataset</td><td>Point</td><td>A AUROC</td><td>B AUROC</td><td>C AUROC</td></tr><tr><td>ML-1M</td><td>easy</td><td>0.999</td><td>1.000</td><td>1.000</td></tr><tr><td>ML-1M</td><td>hard (matched, no AUSH)†</td><td> $0 . 5 3 3 \pm 0 . 0 2 8$ </td><td> $0 . 5 3 6 \pm 0 . 0 3 7$ </td><td> $0 . 5 3 0 \pm 0 . 0 3 0$ </td></tr><tr><td>Musical Instruments</td><td>easy</td><td> $0 . 9 9 1$ </td><td> $0 . 9 9 2$ </td><td> $0 . 9 9 1$ </td></tr><tr><td>Musical Instruments</td><td>hard†</td><td> $0 . 8 0 5 \pm 0 . 0 0 9$ </td><td> $0 . 8 0 8 \pm 0 . 0 0 9$ </td><td> $0 . 8 0 8 \pm 0 . 0 1 0$ </td></tr><tr><td>CiaoDVD</td><td> $\mathrm { e a s y } ^ { \dagger }$ </td><td> $0 . 8 7 7 \pm 0 . 0 1 2$ </td><td> $0 . 8 7 8 \pm 0 . 0 1 0$ </td><td> $0 . 8 7 9 \pm 0 . 0 1 2$ </td></tr><tr><td>CiaoDVD</td><td>hard†</td><td> $0 . 8 4 6 \pm 0 . 0 0 8$ </td><td> $0 . 8 4 8 \pm 0 . 0 0 7$ </td><td> $0 . 8 4 6 \pm 0 . 0 1 0$ </td></tr></table>

Table 3: Three-dataset generalization check: easy (classic attacks) vs. hard (classic+heterogenize+donor-mimicry, no AUSH) AUROC, LightGCN encoder. <sup>†</sup>8-seed mean±std; unmarked rows are single runs (§7). ML-1M’s hard point is the matched-recipe config, distinct from the full-ladder point in Table 1.

0.92–0.97 on every config where LightGCN and every embedding-based detector, VEIL included, sit at or near chance (Appendix B): the profiles are near-duplicates of a specific organic user, trivially visible in the raw data. LightGCN’s particular invariance (item-ID structure smoothed by propagation) discards a signal a coarser, embedding-free view retains. “Detectability limit” here is a property of the representations we test, not a claim that donor-mimicry defeats detection outright.

## 6.7 Does the donor-mimicry collapse generalize across datasets?

A single-dataset ladder cannot say whether the collapse reflects donor-mimicry in general or ML-1M’s embedding geometry. We run the same LightGCN + Components $\mathrm { A / B / C }$ pipeline on two further datasets – Amazon Musical Instruments [23] and CiaoDVD, a trust-aware DVD-rating dataset [24], each at its own post-5-core-filter scale (sizes in Appendix J) – holding the attack recipe fixed: classic Random+Average+Bandwagon for the easy point, classic+heterogenize+donor-mimicry for the hard point, deliberately without AUSH.

The original ML-1M donor-mimicry point (§6.1; AUSH+heterogenize+donor-mimicry, AUROC 0.48–0.51) is not comparable to the other datasets’ hard points, which never used AUSH, so we also ran ML-1M under the matched recipe (matched config: no AUSH), giving AUROC 0.530–0.536 (8-seed mean; Table 3), a small, seed-consistent margin above the “full ladder” number.

The result does not replicate uniformly. On ML-1M (matched), detection collapses from nearceiling to chance-adjacent $( 0 . 9 9 9 - 1 . 0 0 0  0 . 5 3 0 - 0 . 5 3 6 )$ . On Musical Instruments the collapse is partial (0.991–0.992 → 0.805–0.808): a real drop (≈ 0.18), well above chance. On CiaoDVD it is largely absent $( 0 . 8 7 7 - 0 . 8 7 9  0 . 8 4 6 - 0 . 8 4 8 )$ , because the classic-attack baseline never reached near-ceiling, leaving little headroom. Representation-space detectability limits under donor-mimicry are therefore dataset-dependent, not universal, and cannot be assumed to transfer without testing. What drives the divergence is open: neither catalog size nor rating density explains it (§7).

## 6.8 Does the collapse depend on architecture or detection method too?

§6.7 varies the dataset while holding architecture (LightGCN) and detection method (VEIL’s Components $\mathrm { A / B / C ) }$ fixed. We now vary those two axes independently against the same donormimicry attack, to test whether either produces a stable ranking on its own.

Architecture. We reran the full ladder with LightGCN’s propagation depth set to $n _ { \mathrm { l a y e r s } } = 0$ an exact reduction to classical BPR matrix factorization [25] (confirmed by training-loop inspection; not a separately implemented architecture). Easy and AUSH points are indistinguishable between the two (both near-ceiling). On ML-1M’s donor-mimicry point MF is substantially more robust than LightGCN (AUROC 0.632 vs. 0.481–0.494). On both CiaoDVD points the ranking reverses: LightGCN is more robust (classic: 0.877–0.879 vs. 0.814; donor-mimicry: 0.846–0.848 vs. 0.764), plausibly because MF underfits CiaoDVD’s much sparser per-user signal (≈ 15 vs. ≈ 165 ratings/user). Full 6-config table: Appendix G.

<table><tr><td>Architecture</td><td>A AUROC</td><td>B AUROC</td><td>C AUROC</td></tr><tr><td>LightGCN</td><td> $0 . 8 0 0 \pm 0 . 0 1 0$ </td><td> $0 . 8 0 2 \pm 0 . 0 1 0$ </td><td> $0 . 8 0 3 \pm 0 . 0 1 1$ </td></tr><tr><td>mean-pool</td><td> $0 . 8 7 4 \pm 0 . 0 0 6$ </td><td> $0 . 8 8 1 \pm 0 . 0 0 7$ </td><td> $0 . 8 8 5 \pm 0 . 0 0 7$ </td></tr><tr><td>SASRec</td><td> $0 . 9 0 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 1 \pm 0 . 0 0 2$ </td><td> $0 . 9 0 2 \pm 0 . 0 0 3$ </td></tr><tr><td>MF</td><td> $0 . 9 4 0 \pm 0 . 0 0 2$ </td><td> $0 . 9 4 1 \pm 0 . 0 0 3$ </td><td> $0 . 9 4 1 \pm 0 . 0 0 3$ </td></tr></table>

Table 4: Four-architecture comparison, Musical Instruments, order-preserving donor-mimicry, 8-seed mean±std. Ranking is MF > SASRec > mean-pool > LightGCN on every component; MF’s 95% CI does not overlap SASRec’s on any component (e.g. Component C: [0.938, 0.943] vs. [0.900, 0.904]).

Sequential encoders. A sequential encoder might resist donor-mimicry, since the attack clones a donor’s item set but not necessarily its temporal order. We tested SASRec (causal self-attention, embedding dim 64, matching LightGCN/MF) and an order-agnostic mean-pooled control (same architecture, causal mask and position removed) on Musical Instruments; both are label-free with no free per-user parameter vector. The donor-mimicry generator used elsewhere in this paper does not preserve temporal order (items copied in arbitrary order, targets always appended last, synthetic regular-cadence timestamps) – an artifact a sequential encoder can exploit. A 3-seed check confirms it: SASRec reaches AUROC 0.9997–0.9999 (SD ≈ 0.0001) on the original generator, while mean-pool on the identical profiles scores far lower (0.840–0.857) – positional, not a genuine signal. We built an order-preserving variant (each target item takes the position/timestamp of the donor item it displaced; verified bit-identical to the original on every other attack/config). All results below use this corrected generator.

LightGCN – the only architecture here that performs graph propagation – is the clear outlier at the bottom, reproducing the ML-1M donor-distance signature on a second dataset; the secondary MF/SASRec/mean-pool ordering is measured but not mechanistically explained (Appendix G).

Detection method. We compared VEIL against a classical baseline (A2: per-user RDMA/ WDMA/LengthVar rating-value statistics [8] + logistic regression, same poisoned dataset and split). On ML-1M both collapse under donor-mimicry (A2: 0.982 → 0.584; VEIL: 1.00 → 0.53); on CiaoDVD A2 collapses more (A2: 0.840 → 0.617; VEIL: 0.878 → 0.847). Full table: Appendix H.

One unifying observation, not three. Collapse severity has a real, measured ranking by dataset (ML-1M worst, Musical Instruments partial, CiaoDVD least; §6.7), but that ranking does not transfer once a second axis varies: MF beats LightGCN on ML-1M’s donor-mimicry point but loses to it on both CiaoDVD points, and VEIL beats the classical baseline on CiaoDVD but only matches it on ML-1M. Dataset, architecture, and detection method interact, so a result from varying one is not safe to generalize along the others without checking. See Appendix B.

## 7 Limitations

• Three datasets, partial ladder coverage, unresolved divergence. §6.7 covers ML-1M, Amazon Musical Instruments, and CiaoDVD, but only 2 of the 5 ladder points (classic-easy and donor-mimicry-hard) were run on the two new datasets, and Table 3’s ML-1M-easy and Musical-Instruments-easy AUROCs are single runs (their 8-seed sweeps, used in §6.2, agree to within 0.001). Whether the pattern holds on the missing points or on other domains is untested. What drives the divergence is unresolved: catalog size does not track collapse severity (CiaoDVD: 2,069 items, least collapse; ML-1M: 3,416, most; Musical Instruments: 27,637, between), and rating density (ML-1M ≈ 165, CiaoDVD ≈ 15.6, Musical Instruments ≈ 8.9 ratings/user) does not either. CiaoDVD’s resistance is largely an artifact of weak baseline separability.

• No live or human study. All attacks and “organic” profiles come from static, already-collected datasets.

• Static pairwise BIRL, not sequential MDP-IRL. §3: the framework cannot model how a profile accrues ratings over time.

• Encoder and baseline coverage remains narrow. Sup-GNN and LightGCN are GNN-family encoders on the same bipartite graph, and the collapse is mechanistically explained for LightGCN specifically (§6.5); §6.8 adds matrix factorization and finds a dataset-dependent, not uniform, relationship to LightGCN. We did not pursue modern learned shilling-detector baselines (our only baseline is the classical RDMA/WDMA classifier). §6.8 adds SASRec and an order-agnostic mean-pooled control on Musical Instruments’ donor-mimicry point, at one embedding dimension with early-stopped training and no hyperparameter sweep; this project’s CPU-only setup (§5) still limited us to that one dataset/config, not ML-1M/CiaoDVD, and our claims are scoped accordingly.

• Component C’s routing is ground-truth, not inferred. It is an auditing experiment, not a deployable detector; our label-free attempt (Appendix A, one configuration) found only a weakly attack-related partition (ARI 0.05).

• Component B α spread covers inference seeds only. The 8-seed spreads (§6.3) vary pair sampling, split, and VI initialization over one trained encoder per config; encoder-retraining variance is not measured, and the inverted-U reading is post-hoc.

## 8 Conclusion

We set out from a structural claim rather than a benchmark score: representation-space fraud detection built on an invariant embedding has a provable blind spot (Proposition 1) – a property of the representation, not the attack: raw item-set Jaccard, no embedding, detects the same profiles at AUROC 0.92–0.97 where LightGCN collapses (§6.6). That blind spot is not reliably bounded by checking any single dataset, architecture, or detection method; and the right response is therefore a framework that reports calibrated confidence in its own detections and is checked against exactly those axes. Building and testing VEIL, structured after Bou et al.’s Alignment Auditor [1], produced the three findings stated in §1 and reported in §6 – Component B’s confidence α is non-monotonic in attack dificulty; donor-mimicry collapses detection from AUROC 0.936–1.000 to exact chance (0.48–0.51) through a traceable, already-known mechanism; and the severity of that collapse ranks diferently by dataset, architecture, and detection method, with no axis predicting the others – plus a validation check: Component C’s posterior contracts monotonically with evidence, run as a label-routed auditing experiment, not a deployable detector.

The practical upshot is that a single AUROC number, reported without auditing whether confidence degrades correctly under harder evidence or generalizes beyond the dataset, architecture, or detection method it was measured with, cannot distinguish a robust detector from one attackdesign choice, dataset switch, or architecture swap away from chance. §7 bounds how far that claim travels.

## References

[1] Matthieu Bou, Nyal Patel, Arjun Jagota, Satyapriya Krishna, and Sonali Parbhoo. The alignment auditor: A Bayesian framework for verifying and refining LLM objectives. In International Conference on Learning Representations (ICLR), 2026.

[2] Haifeng Yu, Michael Kaminsky, Phillip B. Gibbons, and Abraham Flaxman. SybilGuard: Defending against sybil attacks via social networks. In Proceedings of the 2006 Conference on Applications, Technologies, Architectures, and Protocols for Computer Communications (SIGCOMM), 2006.

[3] Haifeng Yu, Phillip B. Gibbons, Michael Kaminsky, and Feng Xiao. SybilLimit: A near-optimal social network defense against sybil attacks. In IEEE Symposium on Security and Privacy (S&P), 2008.

[4] Wenqi Fan, Tyler Derr, Xiangyu Zhao, Yao Ma, Hui Liu, Jianping Wang, Jiliang Tang, and Qing Li. Attacking black-box recommendations via copying cross-domain user profiles. In Proceedings of the 2021 IEEE 37th International Conference on Data Engineering (ICDE), 2021.

[5] Xuxin Zhang, Jian Chen, Rui Zhang, Chen Wang, and Ling Liu. Attacking recommender systems with plausible profile. IEEE Transactions on Information Forensics and Security, 16: 4788–4802, 2021. doi: 10.1109/TIFS.2021.3117078.

[6] Shyong K. Lam and John Riedl. Shilling recommender systems for fun and profit. In Proceedings of the 13th International Conference on World Wide Web (WWW ’04), pages 393–402, 2004. doi: 10.1145/988672.988726.

[7] Bamshad Mobasher, Robin Burke, Runa Bhaumik, and Chad Williams. Toward trustworthy recommender systems: An analysis of attack models and algorithm robustness. ACM Transactions on Internet Technology, 7(4), 2007. doi: 10.1145/1278366.1278372.

[8] Paul-Alexandru Chirita, Wolfgang Nejdl, and Cristian Zamfir. Preventing shilling attacks in online recommender systems. In Proceedings of the 7th ACM International Workshop on Web Information and Data Management (WIDM), pages 67–74, 2005. doi: 10.1145/1097047.1097061.

[9] Chen Lin, Si Chen, Hui Li, Yanghua Xiao, Lianyun Li, and Qian Yang. Attacking recommender systems with augmented user profiles. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management (CIKM), 2020.

[10] Fan Wu, Min Gao, Junliang Yu, Zongwei Wang, Kecheng Liu, and Xu Wang. Ready for emerging threats to recommender systems? A graph convolution-based generative shilling attack. Information Sciences, 2021.

[11] Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, and Hongzhi Yin. Controllable and stealthy shilling attacks via dispersive latent difusion. arXiv preprint arXiv:2508.01987, 2025.

[12] Z. Zhai, C. Xu, Y. Li, and S. Su. MVCA: Multi-view cross-attention framework for robust shilling attack detection. Symmetry, 2026. doi: 10.3390/sym18030497.

[13] J. Zhao, W. Zhang, S. Wang, Q. Bai, and K. Liu. AttResRec: Learning user credibility for attack resistant matrix factorization recommendation. IEEE Transactions on Services Computing, 18:3509–3522, 2025. doi: 10.1109/tsc.2025.3604375.

[14] Y. Jiang, Y. Zhou, C. Li, and Y. Wang. On the detection of shilling attacks in federated collaborative filtering. In 2020 International Symposium on Reliable Distributed Systems (SRDS), pages 185–194, 2020. doi: 10.1109/srds51746.2020.00026.

[15] S. Wang, P. Zhang, H. Wang, H. Yu, and F. Zhang. Detecting shilling groups in online recommender systems based on graph convolutional network. Information Processing & Management, 59(5):103031, 2022.

[16] Y. Xu, P. Zhang, H. Yu, and F. Zhang. Detecting group shilling attacks in recommender systems based on user multi-dimensional features and collusive behaviour analysis. The Computer Journal, 67:604–616, 2023. doi: 10.1093/comjnl/bxad003.

[17] Shuguang Zhang, Lingjie Liu, Xuntao Zhi, Yu Cheng, Xinyu Zheng, and Yunlong Wang. Shilling attack detection method integrating item temporal information and relational networks. Concurrency and Computation: Practice and Experience, 37(15–17):e70150, 2025. doi: 10.1002/ cpe.70150.

[18] Jaedeug Choi and Kee-Eung Kim. Hierarchical Bayesian inverse reinforcement learning. IEEE Transactions on Cybernetics, 45(4):793–805, 2015.

[19] Mahdi Imani and Seyede Fatemeh Ghoreishi. Scalable inverse reinforcement learning through multi-fidelity Bayesian optimization. IEEE Transactions on Neural Networks and Learning Systems, 33:4125–4132, 2021.

[20] Tahera Hossain, Wanggang Shen, Anindya Das Antar, Snehal Prabhudesai, Sozo Inoue, Xun Huan, and Nikola Banovic. A Bayesian approach for quantifying data scarcity when modeling human behavior via inverse reinforcement learning. ACM Transactions on Computer-Human Interaction, 2022. doi: 10.1145/3551388.

[21] Xiaocong Chen, Lina Yao, Xianzhi Wang, Aixin Sun, Wenjie Zhang, and Quan Z. Sheng. Generative adversarial reward learning for generalized behavior tendency inference. arXiv preprint arXiv:2105.00822, 2021.

[22] F. Maxwell Harper and Joseph A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive Intelligent Systems, 5(4):19:1–19:19, 2015. doi: 10.1145/2827872.

[23] Yupeng Hou, Jiacheng Li, Xiangjun Fu, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. Bridging language and items for retrieval and recommendation: Benchmarking LLMs as semantic encoders. arXiv preprint arXiv:2403.03952, 2024.

[24] Guibing Guo, Jie Zhang, Daniel Thalmann, and Neil Yorke-Smith. ETAF: An extended trust antecedents framework for trust prediction. In Proceedings of the 2014 International Conference on Advances in Social Networks Analysis and Mining (ASONAM), 2014.

[25] Stefen Rendle, Christoph Freudenthaler, Zeno Gantner, and Lars Schmidt-Thieme. BPR: Bayesian personalized ranking from implicit feedback. In Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence (UAI), pages 452–461, 2009.

## A Component C, Unsupervised Routing

Unsupervised routing: a stable partition, but only weakly tied to the attackers. We tested replacing Component C’s ground-truth routing with a stochastic variational-EM procedure on CiaoDVD’s classic-attack (easy) configuration, the smallest diagnostic before any larger commitment. No true label was used anywhere in the fitting procedure; true labels were used only afterward, to evaluate how well the recovered assignment matched reality – standard practice for validating an unsupervised method.

This is an important contrast to state plainly. Proposition 1 and the mechanism in §6.6 describe information that is structurally absent from this encoder’s representation: LightGCN never sees rating values in the first place, so no downstream method, supervised or not, can recover from ϕ what was never represented in it – a limit of this representation, not of donor-mimicry attacks generally (a raw, embedding-free score does recover it; Appendix B). The CiaoDVD classic-attack configuration is diferent: the organic-versus-attacker distinction is present in the representation and is not hard to recover with supervision – labeled Component C achieves AUROC 0.827–0.869 on this exact configuration. The open question here is narrower and diferent in kind: whether an unsupervised estimator, with no label access at all, can reliably find that same distinction on its own.

The answer is a qualified yes, with a precise limitation. Four label-free initializations were tried: k-means clustering, isolation-forest anomaly scoring, a genuinely random 50/50 split, and a principal-component threshold. All numbers below are for this one configuration (2,095 profiles, 273 injected). The original runs did not save their assignments, so we regenerated the seed-42 runs with identical code and settings, reproducing every recorded mixing weight πˆ, AUROC, and final ELBO to the printed precision.

k-means and PCA-threshold converge to the same partition with the roles swapped. The two runs’ hard assignments agree on 98.4% of profiles once the organic and attacker labels are swapped (Pearson r = −0.97 between their posteriors; adjusted Rand index 0.94 between the partitions), a split of about 41%/59%. The PCA-threshold run started from a 16.7% split and abandoned it (ARI 0.015 with its own converged partition). Their apparent contrast, AUROC 0.61 against 0.38 (post-hoc, true labels), is this swap: flipping the PCA run’s roles gives 0.62. The two branches are near-exchangeable (they difer only through initialization and a weak minority prior), so the final ELBO is essentially the same for both orientations (−164.86 vs. −164.00, a gap smaller than the 6.6-nat spread across k-means restarts alone) and cannot choose between them. This is ordinary label switching in a two-component mixture, not two competing estimators of which one is right.

A label-free orientation rule fixes this. Calling the smaller cluster “attacker” (equivalently π <ˆ 0.5, consistent with the model’s generic minority prior) gives post-hoc AUROC 0.610–0.618 across all eight runs (each of the two initializations at seed 42 and three further seeds). The rule uses no labels, but we adopted it after seeing the swap, not in advance. Restarts are stable: pairwise ARI between runs is 0.94–0.98 both within each initialization and across the two, so the AUROC contrast is not run-to-run instability.

But the partition is only weakly tied to the attackers. It agrees with the true labels at ARI 0.049. Its smaller side (41% of profiles, against a true injection rate of 13%) is 20.6% injected against 7.8% on the other side, roughly a 1.6× enrichment, and that enrichment is what produces the

AUROC. In the embedding the model was fit on, the injected profiles are not a cluster (silhouette −0.06 for the true partition, against 0.41 for this one), so a cluster-quality criterion prefers the wrong partition, and neither the ELBO, silhouette, nor restarts can reveal that the stable solution is not the attack split.

The other two initializations. Random 50/50 ends at a genuinely diferent and weak solution (ARI 0.13 with the shared partition, silhouette 0.09, final ELBO −1309). Isolation-forest initialization ends at a related but distinct partition (ARI 0.60 with the shared one, same orientation, AUROC 0.529). The isolation-forest and local-outlier-factor detectors themselves flag 15.5% and 12.7% of profiles, and none of the 273 injected profiles is flagged by either: density-based anomaly detection finds real minority structure that is unrelated to the attack.

We read this as a limit on unsupervised recovery on this task, not an implementation gap. The fit is reproducible and its orientation is fixable by convention, but the partition it finds is only weakly related to the attackers, so unsupervised Component C is a partial capability, not a replacement for label-routing. This is a one-configuration diagnostic and we make no claim that it generalizes.

A residual signal that a linear reward cannot express. On ML-1M’s full-ladder donormimicry configuration, a radial statistic on the frozen embeddings (distance from the embedding centroid) recovers a small residual signal, AUROC 0.58 averaged over VEIL’s eight resampled test splits (0.59–0.60 for the closely related Isolation Forest and k-means-distance scores), against 0.49 for VEIL’s Component A on the same splits. The evidence is weak: only k-means-distance survives a Bonferroni correction over the 20 tests examined (the radial statistic alone, tested separately at $p = 0 . 0 1$ , does not), and on the easier matched (no-AUSH) variant the gap shrinks to a near-tie (0.56 vs. 0.53). VEIL’s reward is linear in the frozen embedding and cannot express a radial displacement, and a label-trained linear probe on the same embeddings also stays at chance (0.47), so the residual is consistent with a function-class limit rather than a supervision gap; it is not evidence that a generic baseline outperforms VEIL.

## B Label-Free Detectability of the Injected Profiles, Three Datasets

This diagnostic asks whether the information VEIL uses on the easy attacks is also recoverable without labels from the same frozen embeddings, and whether the answer depends on the dataset. It is not a competing detector: no score is tuned, and AUROC is reported raw (higher score = more anomalous = predicted attacker), so a value below 0.5 means the injected profiles look more normal than organic ones. We use four scores on the frozen LightGCN embeddings: mean distance to the 10 nearest neighbours (kNN), Local Outlier Factor with 20 neighbours (LOF), distance to the nearest of two k-means centres, and a radial statistic (distance from the embedding centroid). Isolation Forest, run on ML-1M only, tracked the k-means-distance and radial scores closely and is omitted. We add a fifth score that uses no embedding at all: each profile’s maximum Jaccard similarity, over its raw rated-item set, to any other profile’s raw item set (self excluded) – the obvious raw-data check, since a profile that nearly duplicates an existing user is an easy anomaly signal. Computed via sparse binary user-item matrix products, row-chunked (500 rows/chunk) so the full $N \times N$ similarity matrix is never materialized densely (Musical Instruments has $N = 1 8 { , } 3 7 0$ profiles). For each of VEIL’s 8 sweep seeds we recompute the scores on that seed’s own re-split, re-standardized embeddings and evaluate on that seed’s test split, so every label-free number is paired with VEIL’s on the same split; Jaccard does not depend on an embedding or a split, so it is computed once per config and re-evaluated on each seed’s test indices. Table 5 gives the 8-split (a) Some score separates every attack type on every dataset; what varies is how many scores do. Taking each type’s best score with the orientation supplied by the labels (a descriptive quantity, not a detector), separability is 0.93 / 0.93 / 1.00 (random / average / bandwagon) on ML-1M’s classic config, 0.90 / 0.89 / 1.00 on CiaoDVD, and 0.91 / 0.90 / 1.00 on Musical Instruments. On Musical Instruments all four scores separate every type (at least 0.78). On CiaoDVD only LOF does (0.89–0.91 on every type): k-means distance is at chance (0.47–0.54 raw) and kNN separates only bandwagon (0.998; 0.54–0.55 on the other two).

means and Table 6 full-pool bootstrap intervals. CiaoDVD and Musical Instruments have only the classic random+average+bandwagon config and its donor-mimicry variant (no AUSH config was run there), so the per-type results of Table 7 use those three attack types; ML-1M additionally has an AUSH config.
<table><tr><td>dataset</td><td>config</td><td>VEIL  $\mathrm { A } / \mathrm { B } / \mathrm { C }$ </td><td>kNN</td><td>LOF</td><td>k-means</td><td>radial</td><td>Jaccard</td></tr><tr><td>ML-1M</td><td>classic mix</td><td>1.000/1.000/1.000</td><td>0.613</td><td>0.344</td><td>0.618</td><td>0.670</td><td>0.340</td></tr><tr><td></td><td>AUSH</td><td>1.000/1.000/1.000</td><td>0.000</td><td>0.350</td><td>0.971</td><td>0.937</td><td>1.000</td></tr><tr><td></td><td>donor, full ladder</td><td>0.494/0.488/0.481</td><td>0.478</td><td>0.547</td><td>0.603</td><td>0.577</td><td>0.974</td></tr><tr><td></td><td>donor, matched (no AUSH)</td><td>0.533/0.536/0.530</td><td>0.444</td><td>0.530</td><td>0.558</td><td>0.557</td><td>0.924</td></tr><tr><td>CiaoDVD</td><td>classic mix</td><td>0.877/0.878/0.879</td><td>0.299</td><td>0.102</td><td>0.503</td><td>0.365</td><td>0.389</td></tr><tr><td></td><td>donor (no AUSH)</td><td>0.846/0.848/0.846</td><td>0.477</td><td>0.472</td><td>0.582</td><td>0.631</td><td>0.942</td></tr><tr><td>MusInst</td><td>classic mix</td><td>0.990/0.991/0.991</td><td>0.140</td><td>0.075</td><td>0.118</td><td>0.107</td><td>0.341</td></tr><tr><td></td><td>donor (no AUSH)</td><td>0.805/0.808/0.808</td><td>0.360</td><td>0.410</td><td>0.500</td><td>0.504</td><td>0.940</td></tr></table>

Table 5: Label-free anomaly scores vs. labeled VEIL, mixed-attack configs. All values are AUROC against the true labels, raw (higher score = more anomalous = predicted attacker; never flipped, so a value below 0.5 means the injected profiles look more normal than organic ones). VEIL: mean over its 8 sweep seeds (each on its own train/test re-split). kNN/LOF/k-means/radial: recomputed on each seed’s own re-split and re-standardized embeddings and averaged over the same 8 test splits. Jaccard: raw item sets, no embedding, evaluated on the identical 8 test splits (bold: the four donor-mimicry configs, where Jaccard is the only score to detect what every embedding-based score – VEIL included – misses; Table 8). No AUSH config exists for CiaoDVD or Musical Instruments (MusInst). Full-pool bootstrap intervals: Table 6.

<table><tr><td>dataset</td><td>config</td><td>kNN</td><td>LOF</td><td></td><td>k-means</td><td></td><td>radial</td></tr><tr><td>ML-1M</td><td>classic mix</td><td>0.620 [0.569, 0.672]</td><td>0.335</td><td>[0.304, 0.364]</td><td>0.617 [0.580, 0.653]</td><td>0.668</td><td>[0.626, 0.706]</td></tr><tr><td></td><td>AUSH</td><td>0.000 [0.000, 0.000]</td><td>0.355 [0.302,</td><td>0.411]</td><td>0.972 [0.966, 0.979]</td><td>0.939</td><td>[0.929, 0.949]</td></tr><tr><td></td><td>donor, full ladder</td><td>0.471 [0.412, 0.540]</td><td>0.536</td><td>[0.470, 0.604]</td><td>0.597 [0.544, 0.652]</td><td>0.580</td><td>[0.521, 0.636]</td></tr><tr><td></td><td>donor, matched (no AUSH)</td><td>0.438 [0.400, 0.476]</td><td>0.525</td><td>[0.491, 0.562]</td><td>0.538 [0.504, 0.576]</td><td>0.541</td><td>[0.503, 0.577]</td></tr><tr><td>CiaoDVD</td><td>classic mix</td><td>0.304 [0.275, 0.334]</td><td>0.105</td><td>[0.090, 0.121]</td><td>0.509 [0.487, 0.532]</td><td>0.362</td><td>[0.323, 0.400]</td></tr><tr><td></td><td>donor (no AUSH)</td><td>0.482 [0.447, 0.516]</td><td>0.473</td><td>[0.436, 0.509]</td><td>0.568 [0.534, 0.600]</td><td>0.621</td><td>[0.590, 0.649]</td></tr><tr><td>MusInst</td><td>classic mix</td><td>0.142 [0.136, 0.148]</td><td>0.073</td><td>[0.066, 0.080]</td><td>0.119 [0.114, 0.124]</td><td>0.108</td><td>[0.103, 0.114]</td></tr><tr><td></td><td>donor (no AUSH)</td><td>0.365 [0.354, 0.376]</td><td>0.415 [0.404, 0.427]</td><td></td><td>0.503 [0.492, 0.516]</td><td>0.508</td><td>[0.496, 0.519]</td></tr></table>

Table 6: Companion to Table 5: label-free AUROC over the full pool of profiles (embeddings as cached, not re-split), with 95% bootstrap intervals over profiles. Pool values difer slightly from the 8-split means because they use a single standardization and all profiles.

<table><tr><td>dataset</td><td>config</td><td>attack</td><td>VEIL A†</td><td>kNN</td><td>LOF</td><td>k-means</td><td>radial</td></tr><tr><td>ML-1M</td><td>classic mix</td><td>random</td><td>0.999</td><td>0.928</td><td>0.364</td><td>0.807</td><td>0.896</td></tr><tr><td></td><td></td><td>average</td><td>0.999</td><td>0.930</td><td>0.392</td><td>0.828</td><td>0.906</td></tr><tr><td></td><td></td><td>bandwagon</td><td>1.000</td><td>0.000</td><td>0.279</td><td>0.232</td><td>0.221</td></tr><tr><td></td><td>AUSH</td><td>aush</td><td>1.000</td><td>0.000</td><td>0.350</td><td>0.971</td><td>0.937</td></tr><tr><td>CiaoDVD</td><td>classic mix</td><td>random</td><td>0.825</td><td>0.447</td><td>0.098</td><td>0.538</td><td>0.141</td></tr><tr><td></td><td></td><td>average</td><td>0.825</td><td>0.456</td><td>0.114</td><td>0.505</td><td>0.207</td></tr><tr><td></td><td></td><td>bandwagon</td><td>0.966</td><td>0.002</td><td>0.094</td><td>0.466</td><td>0.737</td></tr><tr><td>MusInst</td><td>classic mix</td><td>random</td><td>0.987</td><td>0.212</td><td>0.087</td><td>0.146</td><td>0.147</td></tr><tr><td></td><td></td><td>average</td><td>0.984</td><td>0.211</td><td>0.103</td><td>0.144</td><td>0.149</td></tr><tr><td></td><td></td><td>bandwagon</td><td>1.000</td><td>0.000</td><td>0.037</td><td>0.066</td><td>0.027</td></tr></table>

Table 7: Per-attack-type AUROC on the classic configs (positives: that attack type only; negatives: organic profiles), raw as in Table 5. Label-free columns are 8-split means. <sup>†</sup>VEIL Component A per type is from the saved posterior on a single split (seed 42), reproducing the recorded overall AUROC; it is only approximately comparable to the 8-split label-free means. <sup>‡</sup>AUSH is a single attack type, so this VEIL entry is the 8-seed mean.

(b) Orientation ambiguity replicates and strengthens: the sign is dataset-dependent, not only attack-dependent. On ML-1M’s classic config, random and average are conventionally oriented for three of the four scores (kNN 0.93, radial 0.90–0.91, k-means 0.81–0.83), bandwagon is inverted for every score, and AUSH is inverted for kNN and LOF. On Musical Instruments every score on every type is inverted (all twelve type-by-score AUROCs are at most 0.21, and the mixed config gives 0.075–0.140), so a conventionally oriented label-free detector would perform far worse than chance there. CiaoDVD is inverted in nine of twelve cells (LOF 0.09–0.11 on every type). The sign that would be needed cannot be read of the data without labels.

(c) The mixing collapse is conditional on sign conflicts, not a general property of mixing. On ML-1M’s classic config, kNN, k-means distance, and radial have opposite signs across types (radial: 0.90, 0.91, 0.22), and their mixed AUROC is 0.61–0.67, far below the per-type values. CiaoDVD’s radial score shows the same conflict (0.14 and 0.21 against 0.74; mixed 0.365). Where the signs agree, mixing does not collapse the score: LOF on ML-1M (per-type 0.28–0.39, mixed 0.34), LOF on CiaoDVD (per-type 0.09–0.11, mixed 0.10, i.e. 0.90 once oriented), and every score on Musical Instruments (mixed, oriented: 0.86–0.93). CiaoDVD’s kNN is not a sign conflict: all three types are below 0.5, and its mixed 0.299 reflects two near-chance types (0.45, 0.46) diluting one perfectly inverted type (0.002).

The raw-profile Jaccard baseline detects donor-mimicry where every embedding-based score, including VEIL, does not. Item-set Jaccard needs no embedding: it is computed directly on each profile’s raw rated-item set. On all four donor-mimicry configs across all three datasets it reaches raw AUROC 0.924–0.974 (8-seed mean; Table 5), on the identical profiles where VEIL and every embedding-based label-free score sit at or near chance. Average precision is more moderate (0.50–0.57 raw) because Jaccard similarity is symmetric: a donor-mimicry profile’s nearest neighbour is its clone, but the true donor’s nearest neighbour is that same clone right back, so every true donor scores as “anomalous” as the profile it was copied into. Table 8 isolates this: flagging organic profiles whose best match is an injected profile at Jaccard $\geq 0 . 5$ (the same threshold Appendix G uses to identify true donors) and excluding them from the negative class (a diagnostic that uses the labels to identify donors, not a detector) raises AUROC to 0.9997–1.0000 and AP to 0.994–1.000 on every config. The number of organic profiles flagged this way is large relative to the injected count – from 90 flagged (ML-1M full ladder, 91 injected) to 1,773 (Musical Instruments, 2,397 injected) – confirming the donor-symmetry efect is real and substantial, not a rounding artifact, and that the raw (not donor-excluded) number is the fair one to headline: a deployed detector does not know in advance which organic users are donors.

<table><tr><td>config</td><td>organic flagged/total</td><td>raw AUROC /AP</td><td>donor-excl. AUROC / AP</td></tr><tr><td>ML-1M, full ladder</td><td>90  / 1,812</td><td>0.974 0.515</td><td>1.0000 /1.000</td></tr><tr><td>ML-1M, matched (no AUSH)</td><td>257 / 1,812</td><td>0.924 0.503</td><td>1.0000 1.000</td></tr><tr><td>CiaoDVD, donor</td><td>227 / 1,822</td><td>0.942 /0.569</td><td>0.9999 /0.999</td></tr><tr><td>Musical Instruments, donor</td><td>1,773 / 15,973</td><td>0.940 / 0.549</td><td>0.9997 / 0.994</td></tr></table>

Table 8: Donor-efect diagnostic: an organic profile is flagged as a likely true donor if its best raw-Jaccard match is an injected profile at similarity ≥ 0.5. “raw” includes all organic profiles in the negative class (the number we headline); “donor-excl.” drops the flagged organic profiles from the negative class, isolating how much the symmetry efect costs. Both columns are 8-seed means, same test splits as Table 5.

On the classic (non-donor) configs Jaccard is not uniformly useful: raw AUROC is near or below chance (0.340–0.389) on plain classic mixes and strongly inverted on AUSH+heterogenize (0.044), but perfect on un-heterogenized AUSH (1.000) – config-dependent, like every other label-free score in this appendix. The score has no tunable parameter.

The ML-1M donor-mimicry result does not generalize – for the other four label-free scores. Restricting to kNN/LOF/k-means/radial, the best score is at or slightly above VEIL’s own near-chance value on ML-1M (0.603 vs. 0.494 on the full ladder; 0.558 vs. 0.533 on the matched variant; significance is discussed in Appendix A). On CiaoDVD and Musical Instruments VEIL does not collapse on the donor configs (0.846 and 0.805) and among those four scores outperforms every one of them: the best is 0.631 on CiaoDVD (radial; k-means distance 0.582, kNN and LOF below 0.5) and 0.504 on Musical Instruments (the other three at or below chance). Jaccard is the exception to all of this, and the more important comparison for what a real attacker would try first.

A label-assisted score edging VEIL is not evidence that a baseline beats it. On CiaoDVD’s classic config, LOF with the orientation supplied by the labels reaches 0.895 on the full pool (95% interval [0.879, 0.910]; 0.898 averaged over the 8 splits), against VEIL’s 0.877 (Component A; 0.878 and 0.879 for B and C). The interval reaches VEIL’s value at its lower edge, so this is at most a slight edge, and it uses label knowledge (the sign) that VEIL does not need. It is consistent with the function-class theme of Appendix A: a density-based score is not expressible as a linear reward, and CiaoDVD’s classic config is where VEIL’s separability is weakest among the easy configs.

Caveats. The per-type VEIL numbers in Table 7 come from a saved posterior on a single split (seed 42) and are only approximately comparable to the 8-split label-free means. Oriented values use labels to choose the sign and describe available information only. All results use LightGCN embeddings with k = 10 neighbours (LOF: 20), fixed in advance; in a neighbourhood-size sweep (not shown) density scores change strongly with scale (for example, kNN on ML-1M’s AUSH config is 0.000 for every k up to 100 and 0.96 at k = 400), so both the strength and the sign of density scores depend on the scale chosen.

## C Full Calibration Table (A/B/C, All Seed-Swept Configs)

8-seed mean AUROC/ECE (A, B) and AUROC/ECE (C); Brier and NLL given for Component A only here for space – full Brier/NLL/reliability-bin data for every component is in the supplementary per-config sweep files (sweep\_vi\_seed.json), not retyped here. 95% CIs (Student-t, n − 1 = 7 df) are in those files, not reproduced in this table for space; Table 2 in the main text gives Component A’s CIs for the same configs.
<table><tr><td>config</td><td>A AUROC</td><td>A ECE</td><td>A Brier</td><td>A NLL</td><td>B AUROC</td><td>B ECE</td><td>C AUROC</td><td>C ECE</td></tr><tr><td>supgnn_base</td><td>0.997</td><td>0.006</td><td>0.006</td><td>0.056</td><td>0.999</td><td>0.002</td><td>0.999</td><td>0.002</td></tr><tr><td>lightgcn_base</td><td>1.000</td><td>0.006</td><td>0.005</td><td>0.047</td><td>1.000</td><td>0.006</td><td>1.000</td><td>0.005</td></tr><tr><td>lightgcn_aush</td><td>1.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>1.000</td><td>0.000</td><td>1.000</td><td>0.000</td></tr><tr><td>lightgcn_aush_het</td><td>0.945</td><td>0.090</td><td>0.071</td><td>0.287</td><td>0.946</td><td>0.087</td><td>0.945</td><td>0.077</td></tr><tr><td>lightgcn_aush_het_donor</td><td>0.494</td><td>0.374</td><td>0.232</td><td>0.679</td><td>0.488</td><td>0.375</td><td>0.481</td><td>0.366</td></tr><tr><td>Musical Instruments (easy)</td><td>0.990</td><td>0.050</td><td>0.038</td><td>0.139</td><td>0.991</td><td>0.049</td><td>0.991</td><td>0.048</td></tr><tr><td>Musical Instruments (hard)</td><td>0.805</td><td>0.263</td><td>0.183</td><td>0.544</td><td>0.808</td><td>0.262</td><td>0.808</td><td>0.261</td></tr><tr><td>CiaoDVD (easy)</td><td>0.877</td><td>0.177</td><td>0.137</td><td>0.472</td><td>0.878</td><td>0.177</td><td>0.879</td><td>0.175</td></tr><tr><td>CiaoDVD (hard)</td><td>0.846</td><td>0.187</td><td>0.151</td><td>0.494</td><td>0.848</td><td>0.187</td><td>0.846</td><td>0.186</td></tr><tr><td>ML-1M matched (no AUSH)</td><td>0.533</td><td>0.326</td><td>0.246</td><td>0.690</td><td>0.536</td><td>0.326</td><td>0.530</td><td>0.324</td></tr></table>

Table 9: Calibration across every seed-swept config (all now ≥8 seeds).

![](images/7074c621736805257fcaae50740d766c6be3f059205c38703d12fd242a78f8c4.jpg)

![](images/c2f402c59e84fcea698fb317c63f8011ea6de8f861aa8b4e9bd6dec4ae7bdb1f.jpg)

![](images/0e7342f315d510dd6ec6e4a675179ea2725b4102ea8cb53328c4e0128d829c11.jpg)  
struments, matched hard config), and full collapse (ML-1M full-ladder donor-mimicry). Point size<sup>Figure</sup> <sup>1:</sup> <sup>Pooled</sup> <sup>(8-seed)</sup> <sup>reliability</sup> <sup>diagrams,</sup> <sup>Component</sup> <sup>A:</sup> <sup>easy,</sup> <sup>partial</sup> <sup>collapse</sup> <sup>(Musical</sup> bin count. Dashed line: perfect calibration.Instruments, matched hard config), and full collapse (ML-1M full-ladder donor-mimicry). Point size ∝ bin count. Dashed line: perfect calibration.

## D Component A Full Per-Config Table

Appendix tables round to 4 decimal places (finer than the main text’s 3 significant figures); full-float precision for every cell is in the supplementary results file.

<table><tr><td></td><td colspan="5">test</td><td colspan="4">train</td></tr><tr><td>config</td><td>AUROC</td><td>AP</td><td>Acc</td><td>F1</td><td>pairs-Brier/ECE</td><td>AUROC</td><td>AP</td><td>Acc</td><td>F1</td></tr><tr><td>supgnn_base</td><td>0.9982</td><td>0.9614</td><td>0.9984</td><td>0.9939</td><td>0.0029 0.0176</td><td>0.9987</td><td>0.9872</td><td>0.9959</td><td>0.9843</td></tr><tr><td>lightgcn_base</td><td>0.9993</td><td>0.9948</td><td>0.9920</td><td>0.9697</td><td>0.0001 0.0004</td><td>1.0000</td><td>1.0000</td><td>0.9986</td><td>0.9948</td></tr><tr><td>lightgcn_aush</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>0.0000 0.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>lightgcn_aush_het</td><td>0.9424</td><td>0.4234</td><td>0.9492</td><td>0.5246</td><td>0.0428 0.0318</td><td>0.9781</td><td>0.6188</td><td>0.9655</td><td>0.6406</td></tr><tr><td>lightgcn_aush_het_donor</td><td>0.4996</td><td>0.0511</td><td>0.9142</td><td>0.0755</td><td>0.3554 0.2697</td><td>0.7433</td><td>0.1067</td><td>0.9189</td><td>0.1563</td></tr></table>

Table 10: Component A, full per-config detection metrics (test and train splits) at 4-decimal precision, plus held-out preference-pair Brier/ECE.

## E Component B Full Per-Config Table

<table><tr><td>config</td><td>α</td><td> $\lambda _ { \mathrm { m i n } }$ </td><td> $\lambda _ { \mathrm { m a x } }$ </td><td>test AUROC</td><td>test AP</td><td>test Acc</td><td>test F1</td><td>train AUROC</td></tr><tr><td>supgnn_base</td><td>3.2168</td><td>0.0268</td><td>0.4751</td><td>0.9982</td><td>0.9512</td><td>0.9984</td><td>0.9939</td><td>1.0000</td></tr><tr><td>lightgcn_base</td><td>0.4209</td><td>0.0729</td><td>0.3904</td><td>0.9996</td><td>0.9973</td><td>0.9920</td><td>0.9697</td><td>1.0000</td></tr><tr><td>lightgcn_aush</td><td>0.4025</td><td>0.1317</td><td>0.2386</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>lightgcn_aush_het</td><td>0.7044</td><td>0.0512</td><td>0.4866</td><td>0.9442</td><td>0.4391</td><td>0.9527</td><td>0.5574</td><td>0.9801</td></tr><tr><td>lightgcn_aush_het_donor</td><td>0.5412</td><td>0.1076</td><td>0.4984</td><td>0.5134</td><td>0.0523</td><td>0.9177</td><td>0.0784</td><td>0.7382</td></tr></table>

Table 11: Component B, full per-config learned $\alpha ,$ ARD precision range $( \lambda _ { \operatorname* { m i n } } , \lambda _ { \operatorname* { m a x } } )$ , and detection metrics at 4-decimal precision (seed-42 headline run).

<table><tr><td>config</td><td>α mean</td><td>std</td><td>SE</td><td>95% CI (t, df= 7)</td></tr><tr><td>lightgcn_base</td><td>0.3861</td><td>0.0174</td><td>0.0062</td><td>[0.3715, 0.4006]</td></tr><tr><td>lightgcn_aush</td><td>0.3286</td><td>0.0604</td><td>0.0213</td><td>[0.2781, 0.3791]</td></tr><tr><td>lightgcn_aush_het</td><td>0.6877</td><td>0.0610</td><td>0.0216</td><td>[0.6367, 0.7387]</td></tr><tr><td>lightgcn_aush_het_donor</td><td>0.5514</td><td>0.0850</td><td>0.0301</td><td>[0.4803, 0.6225]</td></tr></table>

Table 12: Component B learned α, 8-seed aggregation (seeds 42, 1–7; sample std, $\mathrm { S E } = \mathrm { s t d } / \sqrt { 8 } )$ Seeds vary pair sampling, split, and VI initialization over one trained Stage-0 feature cache per config; encoder-retraining variance is not captured. Source: supplementary results file.

## F Component C Full Per-Config Table + Evidence Ablation

<table><tr><td>config</td><td> $\alpha _ { \mathrm { o r g } }$ </td><td> $\alpha _ { \mathrm { a t t } }$ </td><td>test AUROC</td><td>test AP</td><td>test Brier</td><td>test ECE</td><td>train AUROC</td></tr><tr><td>supgnn_base</td><td>3.2373</td><td>3.0016</td><td>0.9982</td><td>0.9512</td><td>0.0016</td><td>0.0017</td><td>1.0000</td></tr><tr><td>lightgcn_base</td><td>0.4670</td><td>0.2973</td><td>0.9997</td><td>0.9978</td><td>0.0063</td><td>0.0062</td><td>1.0000</td></tr><tr><td>lightgcn_aush</td><td>0.6916</td><td>0.3056</td><td>1.0000</td><td>1.0000</td><td>0.0000</td><td>0.0000</td><td>1.0000</td></tr><tr><td>lightgcn_aush_het</td><td>0.8285</td><td>0.4898</td><td>0.9364</td><td>0.4642</td><td>0.0707</td><td>0.0837</td><td>0.9848</td></tr><tr><td>lightgcn_aush_het_donor</td><td>0.5398</td><td>0.3934</td><td>0.5075</td><td>0.0631</td><td>0.2384</td><td>0.3889</td><td>0.7465</td></tr></table>

Table 13: Component C, full per-config organic/attacker branch α and combined-detection metrics at 4-decimal precision.

Component C attacker-branch evidence-accumulation ablation (§6.4), full precision, base LightGCN config. The ablation script does not record which feature cache it read; we identify it as the base-LightGCN cache because the ablation’s injected-count ceiling (191) matches that config’s injected count exactly, so this identification is an inference, not a logged fact (Table 14).

<table><tr><td>Ninjected</td><td>Ntrain pairs</td><td>log |Σ|</td><td>σ</td><td>α</td><td>branch AUROC (test)</td></tr><tr><td>20</td><td>300</td><td>78.4868</td><td>1.8578</td><td>0.1831</td><td>0.9990</td></tr><tr><td>50</td><td>750</td><td>59.0742</td><td>1.5937</td><td>0.2108</td><td>0.9992</td></tr><tr><td>100</td><td>1500</td><td>37.6224</td><td>1.3496</td><td>0.2527</td><td>0.9998</td></tr><tr><td>150</td><td>2250</td><td>20.2082</td><td>1.1777</td><td>0.2697</td><td>0.9997</td></tr><tr><td>191</td><td>2865</td><td>15.9541</td><td>1.1388</td><td>0.2973</td><td>0.9997</td></tr></table>

Table 14: Component C attacker-branch posterior contraction vs. injected evidence count, full precision.

## G Matrix Factorization $( n _ { \mathrm { l a y e r s } } = 0 )$ – Full Results, and the Donor-Distance Check

Method. lightgcn\_layers=0 on the existing LightGCN/\_lightgcn\_embeddings path – no separate module. Confirmed by full training-loop inspection to be an exact reduction to classical BPR matrix factorization [25]: the propagation loop never executes, so the returned embedding is the raw layer-0 embedding table; negative sampling and the BPR loss are layer-count-independent by construction; $L _ { 2 }$ regularization was already computed on raw layer-0 embeddings, not something disabled for this comparison.

<table><tr><td>config</td><td>Stage 0 (s)</td><td>A AUROC</td><td>B AUROC</td><td>C AUROC</td><td>LightGCN comparison</td></tr><tr><td>ml1m_mf_base</td><td>191.0</td><td>0.9941</td><td>0.9944</td><td>0.9956</td><td>indistinguishable (both near-ceiling)</td></tr><tr><td>ml1m_mf_aush</td><td>221.2</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>both saturate</td></tr><tr><td>ml1m_mf_aush_het</td><td>220.4</td><td>0.9451</td><td>0.9465</td><td>0.9547</td><td>nearly identical</td></tr><tr><td>ml1m_mf_aush_het_donor‡</td><td>1100.6</td><td>0.6320</td><td>0.6424</td><td>0.6305</td><td>MF substantially more robust (LightGCN: 0.481–0.494)</td></tr><tr><td>ciaodvd_mf_classic</td><td>23.6</td><td>0.8137</td><td>0.8119</td><td>0.8148</td><td>LightGCN notably better (0.877–0.879)</td></tr><tr><td>ciaodvd_mf_het_donor</td><td>19.3</td><td>0.7644</td><td>0.7675</td><td>0.7614</td><td>LightGCN notably better (0.846–0.848)</td></tr></table>

Table 15: MF full ladder, all 6 configs. <sup>‡</sup>Raw wall-clock includes an anomalous ≈ 890 s machine-level freeze (confirmed by a simultaneously-silent independent resource monitor, so not workload-specific); freeze-adjusted compute ≈ 210 s, consistent with the other MF configs. Reported as measured, not excluded.

Architecture ranking is dataset/config-dependent, not uniform: MF is notably more robust than LightGCN specifically on ML-1M’s donor-mimicry point, but consistently worse than LightGCN on both CiaoDVD points – plausibly MF underfitting CiaoDVD’s much sparser per-profile signal (≈ 15 ratings/user vs. ML-1M’s ≈ 165).

Donor-distance check, full protocol (§6.5). On ml1m\_aush\_het\_donor, we reconstructed the poisoned dataset (deterministic given seed=42; identical for both the LightGCN and MF cached runs) and recovered each injected profile’s true donor via item-set Jaccard overlap – donor-mimicry copies the donor’s item set almost wholesale, so the true donor’s overlap with an injected profile is far higher than any other organic user’s by construction (mean Jaccard 0.8517, median 0.9600; 81 of 91 injected profiles confidently matched at Jaccard $\geq 0 . 5$ , 10 excluded as ambiguous). We then measured injected-to-donor distance in each encoder’s frozen phi, plus an injected-to-random-organic-user control.
<table><tr><td>metric  $( \mathrm { m e a n } \pm \mathrm { S E M } , n = 8 1 )$ </td><td>LightGCN</td><td>MF</td></tr><tr><td>injected→donor Euclidean dist.</td><td> $1 . 9 8 2 6 \pm 0 . 0 3 6 1$ </td><td> $3 . 0 9 3 1 \pm 0 . 0 6 2 8$ </td></tr><tr><td>injected→donor cosine similarity</td><td> $0 . 9 6 7 3 \pm 0 . 0 0 2 0$ </td><td> $0 . 9 2 0 3 \pm 0 . 0 0 3 7$ </td></tr><tr><td>injected→random-organic Euclidean dist. (control)</td><td> $1 0 . 9 1 3 4 \pm 0 . 3 7 5 1$ </td><td> $1 1 . 0 3 0 4 \pm 0 . 2 4 1 1$ </td></tr><tr><td>injected→random-organic cosine similarity (control)</td><td> $0 . 0 6 3 3 \pm 0 . 0 4 3 4$ </td><td> $0 . 0 1 3 3 \pm 0 . 0 1 9 8$ </td></tr></table>

Table 16: Injected-to-donor vs. injected-to-random-organic embedding distance, LightGCN vs. MF, ml1m\_aush\_het\_donor. Paired diference (MF − LightGCN Euclidean distance to true donor): $\mathrm { 1 . 1 1 0 4 \pm 0 . 0 4 3 3 \ S E M }$ , n = 81, approx. paired $t = 2 5 . 6 5$

Both encoders place injected profiles far closer to their true donor than to a random organic user, confirming the donor match is meaningful and not an artifact of the matching procedure; the control distances are nearly identical between encoders (10.91 vs. 11.03), which rules out a general embedding-scale diference as the explanation for the donor-gap result. The donor-specific gap (LightGCN 1.98 vs. MF 3.09) is real, tight $( \mathrm { S E M } \leq 0 . 0 7$ on both sides), and in the direction the propagation-smoothing hypothesis predicts. This is correlational and single-config, not a controlled ablation isolating propagation depth from every other diference between the two fits – reported in §6.5 as supporting evidence for the mechanistic account, not as independent proof of it.

Sequential encoders vs. MF, full analysis (§6.8). On Musical Instruments’ order-preserving donor-mimicry config, Table 4 ranks MF > SASRec > mean-pool > LightGCN on every component. LightGCN’s position is explained by the same propagation mechanism as the donor-distance result above: it is the only one of the four architectures that mixes an injected profile’s embedding with its neighbors’, including its donor’s, and the other three – which compute a representation directly from a profile’s own interactions, with no graph difusion – all score well above it. The ordering among those three is not explained by that mechanism, or by any other we tested: MF has no sequence information at all and a free per-user parameter vector, the opposite design from both sequence encoders, yet scores highest of the four, ahead of SASRec (95% CI does not overlap on any component, e.g. Component C [0.938, 0.943] vs. [0.900, 0.904]), which in turn beats the order-agnostic mean-pool control. We report this secondary ordering as measured, not explained, and flag it as an open question rather than assert a cause.

## H Classical Baseline (A2): Full Results

Method. RDMA/WDMA/LengthVar hand-crafted rating-value features [8] + logistic regression (class\_weight="balanced", $C { = } 1 . 0 )$ , fit on the same poisoned dataset and stratified train/test split Components $\mathrm { A / B / C }$ use for each config – same label access at train time, same held-out test set, same metric set – deliberately excluding DegSim-style neighbor-similarity features (an $O ( N ^ { 2 } )$ scaling risk flagged and excluded from this baseline). Operates on raw rating triples, never on the frozen embedding phi; Stage 0 (GNN training) is never invoked.
<table><tr><td>dataset</td><td>point</td><td>A2 AUROC</td><td>A2 AP</td><td>VEIL AUROC range note</td><td></td></tr><tr><td>CiaoDVD</td><td>easy (classic)</td><td>0.8397</td><td>0.3451</td><td>0.877-0.879</td><td>comparable, VEIL slightly ahead</td></tr><tr><td></td><td>CiaoDVD hard (het+donor)</td><td>0.6171</td><td>0.1630</td><td>0.846–0.848</td><td>A2 collapses far more; VEIL more robust here</td></tr><tr><td>ML-1M</td><td>easy (base)</td><td>0.9822</td><td>0.8085</td><td>0.999-1.000</td><td>comparable, both near-ceiling</td></tr><tr><td>ML-1M</td><td>hard (matched, no AUSH)</td><td>0.5838</td><td>0.1669</td><td>0.530-0.536</td><td>both collapse substantially</td></tr></table>

Table 17: A2 (RDMA/WDMA/LengthVar + logistic regression) vs. VEIL, all 4 tested points. $n _ { \mathrm { t e s t } } = 6 2 9$ (CiaoDVD) / 626 (ML-1M), $n _ { \mathrm { i n j e c t e d , t e s t } } = 8 2$ throughout – split sizes match VEIL’s own exactly, confirming the fair-comparison design.

Not a uniform result: on ML-1M the classical baseline also collapses substantially under donormimicry, supporting this paper’s structural (not VEIL-specific) framing; on CiaoDVD the classical baseline collapses more than VEIL does, the reverse asymmetry from what either simple hypothesis (baseline is robust vs. baseline also collapses) would have predicted alone.

## I Wall-Clock Summary

<table><tr><td>config</td><td>Stage 0</td><td>Comp. A</td><td>Comp. B</td><td>Comp. C</td><td>total</td></tr><tr><td>supgnn_base</td><td>2697.5s</td><td>8.4s</td><td>4.2s</td><td>8.2s</td><td>2719.6s (Stage0 log) ) / 20.9s (ABC log)</td></tr><tr><td>lightgcn_base</td><td>2661.6s</td><td>7.7s</td><td>4.7s</td><td></td><td>2679.3s (Stage0) / 12.4s (AB log)</td></tr><tr><td>lightgcn_aush</td><td>2570.9s</td><td>5.4s</td><td>4.7s</td><td>10.5s</td><td>2591.6s</td></tr><tr><td>lightgcn_aush_het</td><td>2272.8s</td><td>4.8s</td><td>4.2s</td><td>8.4s</td><td>2290.2s</td></tr><tr><td>lightgcn_aush_het_donor</td><td>5239.2s</td><td>4.6s</td><td>4.4s</td><td>9.5s</td><td>5257.7s</td></tr></table>

Table 18: Per-stage wall-clock. Hardware: i3-1115G4 (2 cores / 4 threads), CPU-only, threads=4.

Note on the lightgcn\_base row. No single log covers Stage 0 plus Components A, B, and C for this config. Component C was run for it separately and its wall-clock was not captured, hence the blank entry and the split totals.

## J Hyperparameters

The default configuration file, verbatim:

```yaml
run: { seed: 42, threads: 4, device: cpu }
data: { dataset: ml-1m, users_frac: 0.30, injection_rate: 0.05,
filler_budget_pct: 0.03, max_filler_items: 120, n_target_items: 5,
test_frac: 0.30, aush_pretrain_epochs: 12, aush_hidden_dim: 128,
aush_lambda_push: 1.0, het_pop_exponent: 0.6, het_body_quantile: 0.85,
het_target_rating_lo: 4.0, het_target_rating_hi: 5.0,
het_filler_noise: 0.6, het_pref_boost: 5.0, het_donor_min_ratings: 10 }
gnn: { embedding: user_fused, hidden_dim: 64, epochs: 20, lr: 0.001,
lightgcn_dim: 64, lightgcn_layers: 3, lightgcn_epochs: 30,
lightgcn_lr: 0.001, lightgcn_batch_size: 2048, lightgcn_reg: 0.0001 }
pairs: { pairs_per_injected: 20, pairs_per_anchor: 15, matching: random }
vi: { steps: 3000, batch_size: 256, lr: 0.01, mc_samples: 4, sigma0: 1.0,
alpha: 1.0 } # Component A
hier: { steps: 2000, lr: 0.005, mc_samples: 4, tau1: 1.0, tau2: 1.0,
beta: 1.0, gamma1: 1.0, gamma2: 1.0, alpha_init: 1.0 } # Component B
branch: { steps: 2000, lr: 0.005, mc_samples: 4,
eval_mc_samples: 32 } # Component C
```

Dataset sizes after 5-core filtering. ML-1M: 999,611 interactions (30% of users sampled before injection). Amazon Musical Instruments: 53,246 users / 27,637 items / 472,422 reviews (30% of users sampled). CiaoDVD: 1,822 users / 2,069 items / 28,374 reviews, not subsampled, since its 5-core survivor is already smaller than ML-1M’s 30% subsample.

Attack ladder, per-config detail (Table 1 has the results). Five configs of increasing dificulty:

1. supgnn\_base / lightgcn\_base – random + average + bandwagon @ 5% injection each.

2. lightgcn\_aush – AUSH (GAN-based, src/attacks/aush.py), 12 pretrain epochs (data.aush\_pretrain\_epochs: 12).

3. lightgcn\_aush\_het – AUSH + HeterogenizeAttack wrapper: per-profile size from the organic degree distribution, stochastic popularity/preference-weighted filler, jittered 4.0–5.0 target ratings, per-profile (bias, noise) rating style (het\_per\_profile\_style: true).

4. lightgcn\_aush\_het\_donor – adds –donor-mimicry: each fake profile copies a real organic user’s full (items, ratings) and splices in a random 1–3-item target subset (het\_donor\_min\_ratings: 10).

5. An 8-seed sweep (42, 1, 2, . . . , 7) repeating config 4 to quantify seed variance at the collapse point (sweep\_vi\_seed.py).

Why LightGCN is blind to rating values, line-by-line (moved here from §6.5 for space): src/models/lightgcn.py’s build\_norm\_adj (line 40) builds the bipartite user-item adjacency from raw (user\_idx, item\_idx) pairs, explicitly deduplicating (np.unique(np.stack([row, col], axis=1), axis=0), line 71) into one unweighted edge per (user, item) pair – rating value is never read. bpr\_loss (line 153) takes users, pos\_items, neg\_items – item IDs only, no rating tensor in the signature at all.

## K Algorithm Boxes

Pseudocode below mirrors the prose in §4 and the actual control flow of the Component A/B/C implementations and the shared VI module; function names are the real ones in the codebase, not renamed for presentation.

Component A (component\_a.py:run):   
input: cached features Phi (organic + injected), config cfg   
pairs\_train, pairs\_test <- make\_ab\_pairs(feats, cfg, split)   
# Delta-phi = phi(x+) - phi(x-)   
q <- ELBOModel(dim, alpha=1.0 FIXED, sigma0=cfg.vi.sigma0)   
# mu, log-sigma^2 parameters   
for step in 1..cfg.vi.steps: # 3000   
batch <- sample\_minibatch(pairs\_train, cfg.vi.batch\_size) # 256   
theta\_samples <- reparameterize(q.mu, q.sigma, mc\_samples)   
# cfg.vi.mc\_samples=4   
L <- mean\_s[ sum\_i log sigmoid(alpha \* theta\_samples\_s^T Delta-phi\_i) ]   
- KL(q || N(0, sigma0^2 I))   
Adam.step(maximize L) # lr=cfg.vi.lr=0.01   
report <- {posterior: q, pairs\_test metrics, detection\_{train,test} metrics}   
save\_report(report); save(q, "component\_a\_posterior.pt")   
Component B (component\_b.py:build\_model, :run):   
input: same features, warm-start <- load("component\_a\_posterior.pt")   
q <- ELBOModel(dim, alpha \~ Gamma(tau1, tau2), lambda\_d \~ Gamma(gamma1, gamma2),   
mu\_d \~ N(0, (beta \* lambda\_d)^-1), warm\_start=A’s (mu, sigma))   
for step in 1..cfg.hier.steps: # 2000   
batch <- sample\_minibatch(pairs\_train, ...)   
(theta\_s, alpha\_s, lambda\_s) <- reparameterize(q, mc\_samples)   
L <- mean\_s[ sum\_i log sigmoid(alpha\_s \* theta\_s^T Delta-phi\_i) ]   
- KL(q(theta,alpha,lambda) || hierarchical prior) # extended ELBO   
Adam.step(maximize L) # lr=cfg.hier.lr=0.005   
report <- {posterior (incl. learned alpha, lambda\_min/max), pairs/detection metrics}   
save\_report(report); save(q, "component\_b\_posterior.pt")   
Component C (component\_c.py:\_branch\_pools, :\_fit\_branch, :\_combined\_metrics,   
:run):

input: same features, ground-truth organic/injected labels   
organic\_pool, attacker\_pool <- \_branch\_pools(feats, split) # routed by label   
for anchor in {"organic", "attacker"}:   
branch[anchor] <- \_fit\_branch(cfg, feats, anchor)   
# same as Component B; cfg.branch.steps=2000; warm-started from A   
r\_org(x) <- branch["organic"].reward\_projection(x) # theta\_org^T phi(x)   
r\_att(x) <- branch["attacker"].reward\_projection(x) # theta\_att^T phi(x)   
score(x) <- r\_att(x) - r\_org(x)   
(w, b) <- \_logistic\_fit(score[train], y[train]) # 1-D logistic calibration   
P(injected | x) <- sigmoid(w \* score(x) + b)   
report <- \_combined\_metrics(cfg, feats, branch, split) # AUROC/AP/Brier/ECE   
save\_report(report)