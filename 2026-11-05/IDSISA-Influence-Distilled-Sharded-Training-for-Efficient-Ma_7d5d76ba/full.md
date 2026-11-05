# IDSISA: Influence-Distilled Sharded Training for Efficient Machine Unlearning

Deeraj S K, Sadhana Devarajan, Tanmoy Hazra, and Anupam Shukla

Abstract—Machine unlearning removes the influence of specific training examples to comply with privacy regulations. Full retraining gives strong guarantees but is computationally prohibitive at scale; approximate methods are efficient but lack formal guarantees. SISA (Sharded, Isolated, Sliced, and Aggregated training) achieves certified deletion through data sharding and independent retraining, but random partitioning scatters forget requests across most shards, and isolated retraining degrades utility. We introduce IDSISA (Influence-Distilled SISA), which uses influence functions to concentrate forget-prone samples into few shards and distillation from unaffected shards to recover utility lost to isolation. The deletion guarantee inherited from SISA’s retraining step remains exact and requires no probabilistic qualification; the distillation-based utility recovery is a separately quantified, empirically bounded approximation $( \epsilon _ { \mathrm { c e r t } } \le 0 . 0 1 5$ in expectation). We further formalize certification under sequences of repeated deletion requests, where cross-shard distillation can create dependencies between shards; we prove that a closure-based cascading retraining procedure restores the deletion guarantee to the full aggregated model under any deletion sequence. Experiments on ResNet-18/CIFAR-10, ResNet-18/CIFAR-100, BiLSTM/IMDB, TCN/IMDB, and a fully connected network on Fashion-MNIST show 14–91% better selective forgetting, up to 50% better utility retention in relative terms (with consistent percentage-point gains on the other architectures, where a relative baseline is undefined), and unlearning-time changes ranging from −6.7% to +6.4% depending on architecture, relative to SISA.

Impact Statement—Organizations subject to GDPR and CCPA must honor data deletion requests without the prohibitive cost of retraining from scratch. IDSISA reduces the scope of retraining per request by organizing data according to influence, and recovers utility isolated retraining would otherwise sacrifice through distillation from unaffected shards. The deletion guarantee remains certified in the same deterministic sense as prior shardingbased approaches; utility recovery is a separately quantified approximation, not a claim that the full pipeline is certified end to end. IDSISA improves selective forgetting by up to 91% and utility retention substantially across vision and language tasks (CNN, LSTM, TCN, ANN), with unlearning time changing only modestly depending on architecture, and extends the deletion guarantee correctly to sequences of repeated requests. These properties lower compliance risk and compute cost for services facing frequent deletion requests, including consumer platforms and healthcare systems.

Index Terms—Machine Unlearning, SISA, Influence Functions, Knowledge Distillation, Data Privacy, GDPR, Certified Deletion, Membership Inference, Deep Learning

## I. INTRODUCTION

M <sup>ACHINE</sup> <sup>learning</sup> <sup>systems</sup> <sup>trained</sup> <sup>on</sup> <sup>sensitive</sup> <sup>data</sup>must, under regulations such as GDPR and CCPA, must, under regulations such as GDPR and CCPA support a genuine ”right to be forgotten”: removing not just stored records but their influence on trained model parameters. Retraining from scratch after every deletion satisfies this exactly but costs O(N) per request, infeasible for systems facing frequent deletions. Approximate methods reduce this cost by modifying weights directly, without retraining, but sacrifice the deterministic guarantee regulators require.

SISA [1] resolves this tension for a single deletion by partitioning training data into M independent shards, each with its own model trained in isolation; a deletion request retrains only the shard(s) containing the deleted samples, reducing cost by a factor of 1/M while preserving an exact deletion guarantee, since no parameter in a retrained shard carries any influence from the deleted data. Selective forgetting, the targeted removal of specific training samples’ influence from model parameters while preserving predictive performance on the retained dataset $\mathcal { D } _ { r } = \mathcal { D } \setminus \mathcal { D } _ { f } ,$ , is the property SISA is designed to satisfy per request. Two limitations remain. First, random partitioning does not control which shards a given deletion request touches, so a single request often requires retraining most of the M shards. Second, isolated retraining discards the global statistical patterns visible in the full dataset: each shard, trained on only ∼ N/M samples in isolation, loses utility relative to one that could see the full distribution, analogous to asking a model trained on a small subset to generalize broadly.

IDSISA (Influence-Distilled SISA) addresses both limitations without weakening the underlying deletion guarantee. Gradient-based influence scores, computed once before training, guide shard assignment so that samples likely to be deleted together are co-located; a deletion request then typically requires retraining only 2–4 of $M ~ = ~ 1 0$ shards rather than 8–9. Separately, each retrained shard is regularized toward unaffected shards’ soft output distribution, providing consensus-based decision-boundary calibration that isolated retraining would otherwise sacrifice. This distillation step is deliberately kept outside the deletion guarantee: the guarantee comes entirely from retraining, exactly as in SISA, and distillation is an additional utility-preservation mechanism whose own approximation is separately measured.

Because distillation introduces cross-shard information flow that SISA’s fully isolated design never required, it also introduces a certification question SISA never had to answer: under a sequence of deletion requests, can a shard that taught another shard’s retraining transmit residual influence of data that is later deleted? We answer this directly by formalizing the dependency this creates between shards and proving that retraining the full set of shards reachable through this dependency, not merely the directly affected ones, restores an exact deletion guarantee to the entire aggregated model for any sequence of requests, at a cost we bound and measure rather than assume negligible.

We evaluate IDSISA on ResNet-18/CIFAR-10 (11.2M), ResNet-18/CIFAR-100 (11.2M), BiLSTM/IMDB (4.2M), TCN/IMDB (3.1M), and a fully connected network on Fashion-MNIST (203K), spanning convolutional, recurrent, dilated convolutional, and feedforward architectures. All reported values are means over 5 random seeds (42–46).

## Contributions.

1) Influence-aware partitioning that reduces the shards affected by a typical deletion request by 50–78% relative to random partitioning (4–5 of $M = 1 0$ shards typically retrained, versus 8–9 under random partitioning).

2) Distillation-guided retraining that recovers up to 50% of the utility lost to isolated retraining in relative terms (CNN), and yields consistent percentage-point utility gains on LSTM and ANN, bounded by $\epsilon _ { \mathrm { c e r t } } \leq 0 . 0 1 5$ (mean; Section III-F formalizes its relationship to the deletion guarantee).

3) Certification under repeated deletion: a closure-based cascading retraining procedure, with a proof that it restores exact deletion to the full aggregated model under any sequence of requests, and an exact (not estimated) accounting of its retraining cost.

4) Cross-architecture validation on CNN (CIFAR-10 and CIFAR-100), LSTM, TCN, and ANN models, including an analysis of why vanilla RNNs are incompatible with the influence-computation machinery underlying Phase 1, and a demonstration that TCN avoids this incompatibility by construction.

Section II reviews related work. Section III presents the framework, its certification guarantees, and their extension to repeated deletion. Section IV describes the system architecture. Section V details the experimental setup. Section VI reports results. Section VII discusses limitations. Section VIII concludes.

## II. RELATED WORK

Machine unlearning research sits along a tension between exactness and efficiency: strong-guarantee methods are computationally expensive, and efficient ones generally lack formal guarantees. We organize the literature along this axis.

## A. Exact and certified unlearning

Full retraining after each deletion is exact but costs O(N) per request, infeasible at scale [2]. Ginart et al. gave incremental unlearning for simpler models such as k-means [3]. SISA [1] achieves certified deletion for neural networks via sharding, at 1/M the cost of full retraining. Guo et al. give certified removal via closed-form updates for specific model families [4]. More recently, Koloskova et al. obtain certified unlearning for general neural networks via noisy fine-tuning on the retain set, using a privacy-amplification-by-post-processing argument that requires no sharding and no loss-function assumptions [5]: a differentially-private route to certification, orthogonal to IDSISA’s deterministic, retraining-based route, and not directly comparable on the certified/efficient/utilitypreserving axes of Table I without adapting its privacy accounting to our setting. None of these address the shardconcentration or utility-recovery problems IDSISA targets.

## B. Approximate unlearning methods

Golatkar et al. scrub model weights using Fisher Information, $\Delta \theta \approx - F ^ { - 1 } \nabla \mathcal { L } ( D _ { f } )$ , optionally adding differentialprivacy noise [6], [7]. Neel et al. study gradient-based descentto-delete [8]. These methods run in minutes but modify weights directly without retraining, trading the deterministic deletion guarantee for speed: a different tradeoff than IDSISA makes, since IDSISA uses influence functions and distillation only as organizational and utility-preservation tools within SISA’s exact retraining framework, never as a substitute for retraining.

## C. Influence functions and knowledge distillation

Koh and Liang’s influence functions, I(z) = $- \nabla \mathcal { L } ( \boldsymbol { z } _ { \mathrm { t e s t } } ) ^ { T } H ^ { - 1 } \nabla \mathcal { L } ( \boldsymbol { z } )$ approximated via L-BFGS, quantify a single training sample’s effect on model predictions [9]. Pruthi et al. propose TracIn as a checkpointbased alternative [10]. Prior unlearning work uses influence scores to identify what to forget [4], [11]; IDSISA instead uses them to guide data partitioning before any deletion request arrives. Hinton et al. introduce temperature-scaled distillation [12]. Quan et al.’s PURGE [13] shows that constrained distillation can preserve data isolation in teacherstudent settings, supporting our empirical finding (Section VI) that soft teacher outputs carry limited exploitable information across shard boundaries. Fan et al.’s IMU similarly uses influence functions within unlearning, but as a per-sample signal that modulates gradient-ascent unlearning intensity on the forget set alone, without a retain set [14]; IDSISA instead uses influence purely upstream, to shape data partitioning before any deletion request exists, leaving the deletion mechanism itself as exact retraining rather than gradient ascent.

## D. Federated and verification-oriented advances

Liu et al. and Halimi et al. use distillation for client-level removal in federated settings [15], [16]. Kurmanji et al. propose adaptive shard management for continuous requests [17]. Foster and Pagh use synaptic dampening for retraining-free unlearning [18].

## E. Positioning

Table I summarizes the landscape along three axes relevant to deployment: certification, selective forgetting strength, and efficiency. No prior method scores strongly on all three: exact methods are infeasible at scale, approximate methods sacrifice certification, and SISA’s certification is weakened in practice by poor forgetting concentration and utility loss. IDSISA is positioned to score strongly on all three by construction, with the certification caveat developed fully in Section III-F.

TABLE I  
LANDSCAPE OF MACHINE UNLEARNING APPROACHES. <sup>†</sup>RETRAINING COMPONENT IS DETERMINISTICALLY CERTIFIED; DISTILLATION INTRODUCES A BOUNDED EMPIRICAL APPROXIMATION $( \epsilon _ { \mathrm { C E R T } } \leq 0 . 0 1 5 ,$ MEAN).
<table><tr><td>Method Class</td><td>Certified?</td><td>Selective Forgetting</td><td>Knowledge Pres.</td><td>Efficiency</td></tr><tr><td>Full Retraining</td><td>Yes</td><td>Strong</td><td>Strong</td><td>Low</td></tr><tr><td>Approx. Scrubbing</td><td>No</td><td>Moderate</td><td>Poor</td><td>High</td></tr><tr><td>Descent-to-Delete</td><td>No</td><td>Moderate</td><td>Poor</td><td>High</td></tr><tr><td>SISA (Random Shards)</td><td>Yes</td><td>Weak</td><td>Poor</td><td>High</td></tr><tr><td>Federated Unlearning</td><td>Yes</td><td>Moderate</td><td>Moderate</td><td>High</td></tr><tr><td>IDSISA (Proposed)</td><td>Yes†</td><td>Strong</td><td>Strong</td><td>High</td></tr></table>

TABLE II

KEY NOTATION USED THROUGHOUT THIS PAPER.
<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td> $\mathcal { D }$ </td><td>Full training dataset  $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ </td></tr><tr><td> $\mathcal { D } _ { f }$ </td><td>Forget set: samples to be unlearned</td></tr><tr><td> $\mathcal { D } _ { r }$ </td><td>Retain set:  $\mathcal { D } _ { r } \overset { \cdot } { = } \mathcal { D } \setminus \mathcal { D } _ { f }$ </td></tr><tr><td> $D _ { m }$ </td><td>Data assigned to shard m</td></tr><tr><td> $\theta _ { m }$ </td><td>Model parameters for shard m</td></tr><tr><td> $\theta _ { m } ^ { ( k ) }$ </td><td>Checkpoint after slice k of shard m</td></tr><tr><td> $k ^ { * }$ </td><td>(k = slice index; not a class-incremental step) Earliest affected slice index (rollback target)</td></tr><tr><td> $M$ </td><td>Total number of shards</td></tr><tr><td> $K _ { s }$ </td><td>Number of slices per shard</td></tr><tr><td> $\boldsymbol { A }$ </td><td>Affected shards:  $\dot { \{ m : \mathcal { D } _ { f } \cap D _ { m } \neq \emptyset \} }$ </td></tr><tr><td> $\mathcal { U }$ </td><td>Unaffected shards:  $\{ 1 , \dotsc , M \} \setminus A$ </td></tr><tr><td> $\mathcal { T } ( z _ { i } )$ </td><td>Influence score of sample  $z _ { i }$ </td></tr><tr><td> $\beta$ </td><td>Influence concentration temperature  $( \beta = 0 . 3 )$ </td></tr><tr><td> $_ T$ </td><td>Distillation temperature  $( T \overset { \cdot } { = } 2 . 0 )$ </td></tr><tr><td> $_ \alpha$ </td><td>Distillation loss weight  $( \alpha = 0 . 5 )$ </td></tr><tr><td> $\lambda$ </td><td>L-BFGS damping factor  $( \lambda = 0 . 0 1 )$ </td></tr><tr><td> $\epsilon _ { \mathrm { c e r t } }$ </td><td>Distillation approximation gap (mean-case)</td></tr><tr><td> $G = ( V , E )$ </td><td>Distillation dependency graph (Def. 3)</td></tr><tr><td> $C ( A _ { 0 } )$ </td><td>Reachability closure of affected shards  $A _ { 0 }$ </td></tr></table>

## III. IDSISA FRAMEWORK

## A. Problem formulation

Given $\textit { D } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ and network $f _ { \theta } ,$ the goal is to update θ to $\theta ^ { \prime }$ after removing $D _ { f } \subseteq D$ , satisfying: (1) equivalence: causally indistinguishable from full retraining; (2) efficiency: runtime proportional to $| D _ { f } |$ and the number of shards retrained, not N; (3) privacy: provable removal of $D _ { f } { \mathbf { \bar { s } } }$ influence. SISA satisfies these for isolated shards on nonoverlapping partitions, but at the cost of the two limitations above; IDSISA’s partitioning and distillation mechanisms target exactly those costs without altering requirement (3).

## B. Computational complexity

Table III gives theoretical phase-wise complexity alongside measured runtimes, including setup overhead.

The reported initial training time for CNN (66.1 min) comprises three components: 59.9 minutes of shard training (identical to SISA), ≈5.3 minutes to train the baseline model on the full dataset, and 56 seconds of Phase 1 operations; $5 9 . 9 + 5 . 3 + 0 . 9 3 = 6 6 . 1$ . The baseline model is deliberately an early-stopped proxy (3 epochs): Frankle et al. show representational structure is set within the first few epochs [19], Achille et al. show features are largely determined during an early critical period [20], and TracIn shows gradientbased influence estimation does not require convergence [10], together these support using a 3-epoch checkpoint to extract penultimate-layer features for K-means and bias the stochastic shard assignment, rather than requiring convergence. This is evidence for early-training stability in general, not a direct validation on the specific models, tasks, and schedules evaluated here; Appendix E states boundary conditions under which the argument would weaken. Memory stores M shard models and $K _ { s }$ checkpoints per shard: ResNet-18 requires ∼4.5GB (10 shards, 10 slices); BiLSTM ∼1.8GB (10 shards, 5 slices); the ANN ∼95MB (15 shards, 5 slices). Under the sequentialdeletion mechanism of Section III-G, this grows by at most $O ( M )$ additional checkpoints regardless of the number of requests processed (Appendix C).

TABLE III  
THEORETICAL COMPLEXITY AND EMPIRICAL RUNTIME BREAKDOWN ACROSS ARCHITECTURES (MEAN OVER 5 SEEDS, 42–46). “INITIAL TRAINING” IS DISAGGREGATED PER ARCHITECTURE, CONSISTENT WITH TABLE VI. <sup>†</sup>SPEEDUP FACTOR (MEASURED) IS FULL RETRAINING TIME DIVIDED BY IDSISA UNLEARNING TIME FOR A SINGLE REQUEST; DISTINCT FROM THE THEORETICAL $M / | A |$ SPEEDUP SHOWN HERE AND FROM THE SISA-RELATIVE UNLEARN SPEEDUP COLUMN OF TABLE IV.
<table><tr><td>Operation</td><td>Theoretical</td><td>F-MNIST (ANN)</td><td>CIFAR-10 (CNN)</td><td>IMDB (LSTM)</td></tr><tr><td>Influence Calculation</td><td> $O ( N \cdot p \cdot L )$ </td><td>12s</td><td>38s</td><td>28s</td></tr><tr><td>PCA Reduction</td><td> ${ \mathcal { O } } ( N \cdot d ^ { 2 } )$ </td><td>1s</td><td>3s</td><td>2s</td></tr><tr><td>K-means Clustering</td><td> $O ( N \cdot K \cdot \bar { d } ^ { \prime } )$ </td><td>8s</td><td>15s</td><td>11s</td></tr><tr><td>Initial Training (SISA)</td><td> $O ( N \cdot E \cdot F )$ </td><td>8.9 min</td><td>59.9 min</td><td>120.16s</td></tr><tr><td>Initial Training (IDSISA)</td><td></td><td>8.9 min</td><td>66.1 min</td><td>138.32s</td></tr><tr><td>Unlearning (IDSISA)</td><td> $O ( | D _ { f } | \cdot | \mathcal { A } | \cdot E ^ { \prime } )$ </td><td>9.5 min</td><td>81.2 min</td><td>93s</td></tr><tr><td>Full Retraining</td><td> $\overset { \triangledown } { \boldsymbol { O } } ( \Vec { N } \cdot \ddot { \boldsymbol { E } } \cdot \boldsymbol { F } )$ </td><td>2.1 hrs</td><td>8–10 hrs</td><td>4.2 hrs</td></tr><tr><td>Setup Overhead (Phase 1)</td><td></td><td>21s</td><td>56s</td><td>41s</td></tr><tr><td>Speedup Factor†</td><td> $M / | A |$ </td><td>13.3×</td><td>5.9-7.4×</td><td>162.6×</td></tr></table>

## C. Phase 1: Influence-aware partitioning

An influence score $\mathcal { T } ( z _ { i } )$ estimates how much removing sample $z _ { i }$ would change model predictions, via gradient-based L-BFGS approximation with damping $\lambda = 0 . 0 1 \colon$ : train a baseline model on $D _ { \colon }$ , compute per-sample gradients, approximate Hessian-vector products [9]. Scores are computed on a 5,000- sample validation subset and normalized to [0, 1].

Penultimate-layer features from the baseline model (256- dim for the LSTM, 512-dim for ResNet-18) are reduced via PCA (95% variance retained) and clustered into $K \ = \ M$ groups via K-means. Algorithm 1 then allocates samples to shards, biasing high-influence samples within each cluster toward the same shard (temperature $\beta ~ = ~ 0 . 3 )$ , balancing shard sizes if deviation exceeds 10%. This co-location is what lets a typical deletion request affect 2–4 shards rather than 8–9. The experiments below use this non-private Phase 1 pipeline; Appendix F gives a DP-Lloyd replacement for the clustering/metadata step, requiring a fixed or DP-trained feature/influence proxy and not part of the reported empirical system.

## D. Phase 2: Shard-level training

Each shard m trains independently via SISA’s slicebased protocol: $D _ { m }$ is split into $K _ { s }$ slices $( K _ { s } ~ = ~ 5$ for

Algorithm 1 Influence-aware data partitioning.   
Require: Dataset D, influence scores $\{ \mathcal { T } ( z _ { i } ) \} _ { i = 1 } ^ { N } ,$ clusters $\{ C _ { k } \} _ { k = 1 } ^ { K } ,$   
shards M   
Ensure: Shard partitions $\{ D _ { 1 } , \hdots , D _ { M } \}$   
1: Sort samples by influence; initialize empty shards   
2: Assign clusters to shards: $C _ { k } $ shard(k mod M)   
3: for $\bar { k } = 1$ to K do   
4: $m _ { k } \gets k$ mod $M ;$ sort cluster samples by influence   
5: for each $i \in C _ { k }$ do   
6: Add z<sub>i</sub> to $D _ { m _ { k } }$ with probability $p _ { i }$   
$\exp ( \beta \mathcal { T } ( z _ { i } ) )$   
$\beta = 0 . 3$   
$\sum _ { \mathbf { \lambda } _ { \mathbf { a } } } _ { j \in C _ { k } } \exp ( \beta \mathcal { T } ( z _ { j } ) )$   
7: end for   
8: end for   
9: Balance shard sizes if deviation $> 1 0 \%$   
10: return $\{ D _ { 1 } , \hdots , D _ { M } \}$

Algorithm 2 Distillation-guided unlearning (single request).   
Require: Forget set $D _ { f } ,$ shard models $\lbrace \theta _ { m } \rbrace _ { m = 1 } ^ { M }$ , shards $\{ D _ { m } \} _ { m = 1 } ^ { M }$   
Ensure: Updated models $\lbrace \boldsymbol { \theta } _ { m } ^ { \prime } \rbrace _ { m = 1 } ^ { M }$   
1: $\mathcal { A }  \{ m : D _ { f } \cap D _ { m } \neq \emptyset \} ; \mathcal { U }  \{ 1 , \dots , M \} \backslash \mathcal { A }$   
2: if $| \mathscr { U } | \overset { \cdot } { = } 0$ then   
3: for m $\in { \mathcal { A } }$ do   
4: Retrain from scratch on $D _ { m } \backslash D _ { f }$   
5: end for   
6: else   
7: Form teacher ensemble from $\{ \theta _ { m } : m \in \mathcal { U } \}$   
8: for m $\in { \mathcal { A } }$ do   
9: Restore $\dot { \theta } _ { m } ^ { ( \bar { k } ^ { * } - 1 ) }$ (earliest affected slice)   
10: for epoch = 1 to E do   
11: Sample batch B; compute teacher predictions on B   
12: Update $\theta _ { m }$ via combined $\mathrm { C E } + \mathrm { K L }$ loss (Eq. 1)   
13: end for   
14: end for   
15: end if   
16: return $\lbrace \theta _ { m } ^ { \prime } \rbrace _ { m = 1 } ^ { M }$

MNIST/IMDB, $K _ { s } = 1 0$ for CIFAR-10), with a checkpoint $\theta _ { m } ^ { ( k ) }$ saved after each. A deletion request rewinds to the earliest checkpoint predating the forgotten data. Under the sequentialdeletion mechanism (Section III-G), an additional checkpoint is saved before every distillation-retraining event, not only before each slice (Appendix C).

## E. Phase 3: Distillation-guided unlearning

Given a deletion request $D _ { f }$ , IDSISA identifies affected shards $\mathcal { A } = \{ m : D _ { f } \cap D _ { m } \ \ne \emptyset \}$ and unaffected shards U. Each affected shard retrains from its pre-forget checkpoint on $D _ { m } ^ { \prime } = D _ { m } \backslash D _ { f }$ , using a teacher ensemble formed from $\chi _ { \mathrm { s } }$ averaged, temperature-scaled softmax outputs $( T = 2 . 0 )$

$$
\mathcal { L } _ { \mathrm { t o t a l } } = ( 1 - \alpha ) \mathcal { L } _ { \mathrm { C E } } + \alpha \mathcal { L } _ { \mathrm { K L } } , \quad \alpha = 0 . 5 .\tag{1}
$$

This anchors the retrained shard’s outputs to shards that never saw $D _ { f } .$ , both regularizing against overfitting on the small postdeletion dataset and transferring global statistical patterns the isolated shard would otherwise lose; on CIFAR-10, forgetting the $\mathbf { \dot { \bar { c } } } \mathbf { a } \mathbf { t } ^ { \prime \prime }$ class (5,000 samples) affects 2–3 shards, leaving 7–8 as teachers. Retraining converges in 10–20 epochs versus 50+ for full retraining. If $\mathcal { U } = \emptyset$ , the shard retrains from scratch with no distillation term (Algorithm 2).

This algorithm is exact and complete for a single deletion request against an empty distillation history. Section III-G extends it to sequences of requests.

## F. Certification: definitions and guarantees

IDSISA’s deletion guarantee is not an approximate method in the sense of Fisher-based scrubbing or gradient ascent, which modify weights directly and never retrain. IDSISA’s retraining step is identical to SISA’s and inherits the same guarantee: because an affected shard retrains from a checkpoint predating the forgotten data, no parameter in that shard carries any influence from $D _ { f }$ afterward, deterministically and without probabilistic qualification. Distillation is layered on top of this guarantee, not part of it, and introduces a distinct, separately-bounded approximation.

Remark (Scope of Certification Claims). “Certified” and “exact” deletion refer throughout to Causal Exactness (Definition 1): the retrained shard’s computation never receives any sample from $D _ { f }$ as input. Distributional exactness (Definition 2) is not claimed when distillation is used; that gap is measured as $\epsilon _ { \mathrm { c e r t } }$ (Table XIV) and bounded heuristically by Theorem 1. The pipeline is partially certified: retraining deterministically, distillation empirically bounded.

Definition 1 (Causal Exactness). Parameters $\theta _ { m }$ are causally exact with respect to $D _ { f }$ if the deterministic computation producing $\theta _ { m }$ (training, checkpoint restoration, and distillation, for fixed random seeds) never takes any sample in $D _ { f } ,$ or any tensor derived from parameters that are themselves not causally exact with respect to $D _ { f } ,$ , as an input.

Definition 2 (Distributional Exactness). $\theta _ { m } ^ { \prime }$ is distributionally exact with respect to a retrained model $\theta _ { r e t r a i n } i f \theta _ { m } ^ { \prime }$ is statistically indistinguishable from $\theta _ { r e t r a i n } .$

Retraining from a pre-forget checkpoint achieves causal exactness unconditionally, whether or not distillation is subsequently used. It does not achieve distributional exactness when distillation is used, since the retrained shard’s parameters are influenced by the teacher ensemble’s soft outputs rather than exactly matching a from-scratch retrain; this gap is what $\epsilon _ { \mathrm { c e r t } }$ measures. Only a shard retrained with no distillation $( | \boldsymbol { \mathcal { U } } | = 0 )$ achieves both simultaneously. Every claim of “certified” or “exact” deletion here refers to causal exactness (Definition 1); distributional exactness is never claimed for the distillationretrained case, only quantified via $\epsilon _ { \mathrm { c e r t } }$

Causal Exactness is deliberately distinct from the certifiedunlearning definitions most closely associated with this literature. Guo et al. certify removal by bounding a distributional gap in the KL-divergence sense for models admitting closedform or Newton-style parameter updates [4]; Koloskova et al. certify via a differential-privacy-style indistinguishability bound obtained through noisy fine-tuning on the retain set [5]. Both are distributional guarantees requiring specific mechanisms; Causal Exactness instead certifies an unconditional structural property, that $D _ { f }$ was never an input to the computation producing $\theta _ { m }$ , without requiring convexity, closed-form updates, or injected noise. Distillation’s crossshard logit transfer does not violate this, since teachers are drawn only from shards unaffected by $D _ { f } ;$ ; where this paper writes “certified” or “exact”, it means Definition 1 and nothing stronger.

1) Formal assumptions: Assumption 1 (Smoothness). $\mathcal { L } ( \boldsymbol { \theta } ; \boldsymbol { x } )$ is L-smooth: $\begin{array} { r } { \| \nabla \mathcal { L } ( \theta _ { 1 } ; x ) - \nabla \mathcal { L } ( \theta _ { 2 } ; x ) \| \le L \| \theta _ { 1 } - \theta _ { 2 } \| } \end{array}$ Assumption 2 (Strong Convexity on Shard). Shard loss $\begin{array} { r } { \mathcal { L } _ { k } ( \boldsymbol { \theta } ) = \frac { 1 } { | S _ { k } | } \sum _ { \boldsymbol { x } _ { i } \in S _ { k } } \mathcal { L } ( \boldsymbol { \theta } ; \boldsymbol { x } _ { i } ) } \end{array}$ is µ-strongly convex.

Assumption 3 (Bounded Gradient). $\| \nabla { \mathcal { L } } ( \theta ; x ) \| \leq G .$

These are convex surrogates: they do not hold strictly for ResNet-18 or BiLSTM (Appendix B, Remark 1), and the theorem below should be read accordingly, as an expectedcase validation rather than a uniform guarantee. The theorem’s cross-shard aggregation step further relies on an explicit shard-independence modeling assumption (Assumption 4, Appendix A) that is not implied by Assumptions 1–3 alone; we flag this dependency in the theorem statement itself rather than leave it inside the proof.

Lemma 1 (Influence Bound). Under Assumptions $I { - } 3 ,$ removing $x _ { j }$ from $D _ { m }$ yields $\begin{array} { r } { \| \Delta \theta _ { j } \| \leq \frac { 1 } { \mu } \| \nabla _ { \theta } \mathcal { L } ( \theta ; x _ { j } ) \| } \end{array}$ .

Theorem 1 (Distillation Approximation Bound, heuristic). Under Assumptions 1–3, with $T \geq 2 ,$ and under Assumption 4 (Appendix A), which models affected-shard perturbations as approximately independent so that they aggregate in quadrature rather than by the linear union bound Assumptions $_ { I - 3 }$ alone imply, the distillation-retrained model $\theta ^ { \prime }$ and a fromscratch retrained model $\theta _ { r e t r a i n }$ satisfy

$$
\| f _ { \theta ^ { \prime } } ( x ) - f _ { \theta _ { r e t r a i n } } ( x ) \| \lesssim \frac { L G } { \mu } \cdot | D _ { f } | ^ { 1 / 2 } \cdot \left( 1 + \frac { 1 } { T } \right)\tag{2}
$$

in expectation over $x \sim D _ { t e s t } .$ . We label this bound a heuristic rather than a fully derived worst-case guarantee, since the $| D _ { f } | ^ { 1 / 2 }$ scaling depends on Assumption 4 in addition to Assumptions 1–3. (Derivation: Appendix A.)

Corollary 1 (Illustrative Instantiation). For $L = 1 0 ^ { - 4 } , G = 1$ $\mu = 0 . 7 0 7 , T = 2 . 0 , | D _ { f } | = 5 , 0 0 0 ( C I F A R { - } I O , N = 5 0 , 0 0 0 ) .$ $\epsilon _ { c e r t } = \mathbb { E } _ { x \sim D _ { t e s t } } \Vert f _ { \theta ^ { \prime } } ( x ) - f _ { \theta _ { r e t r a i n } } ( x ) \Vert \leq 0 . 0 1 5 , a \sim 2 \%$ meancase deviation from full retraining.

We label Corollary 1 an illustrative instantiation rather than an independently measured constant set: L and $\mu$ are plugged in to show what Theorem 1’s heuristic bound predicts at this problem scale, not derived from a dedicated Lipschitzor curvature-measurement procedure on the trained ResNet-18 (Appendix B elaborates their provenance and limitations). Three properties of this bound matter in practice. First, temperature T bounds the distillation gradient’s magnitude at $G / T$ so $T \ \geq \ 2$ limits how much the retrained shard’s behavior can be pulled toward the teacher ensemble’s representation of $D _ { f } { \mathbf { \bar { s } } }$ class distribution. Second, the bound scales sublinearly in $| D _ { f } | \colon$ doubling the forget set raises the deviation bound by only ${ \sqrt { 2 } } \times$ , not $2 \times$ , so the approximation degrades gracefully as deletion volume grows. Third, the mean-case threshold $( \epsilon _ { \mathrm { c e r t } } \leq 0 . 0 1 5 , \mathrm { ~ a ~ } \sim 2 \%$ deviation) is comparable in magnitude to the run-to-run variation of independent retraining itself (Table XIV), validating the approximation as statistically unremarkable in expectation, not as a worst-case guarantee. Section VI-J reports both this mean quantity and the 99thpercentile tail separately, since Table XIV’s tail values exceed 0.015 and the bound above was never derived to cover the tail.

Algorithm 3 Closure-based cascading deletion.   
Require: Forget set $D _ { f }$ , models $\{ \theta _ { m } \}$ , graph $G ,$ checkpoints before   
every retraining event (Appendix C)   
Ensure: Updated models, updated G   
1: $A _ { 0 } \gets \mathsf { \bar { \{ } }  m : D _ { f } \cap D _ { m } \neq \emptyset \} ; C \gets C ( A _ { 0 } )$ (Def. 4); ${ \mathcal { T } }  V \backslash C$   
2: for each $m \in { \bar { C } }$ (any order) do   
3: Restore checkpoint preceding m’s first contaminating event   
4: Retrain m (excluding $D _ { f }$ if m $\in A _ { 0 } )$ , distilling only from T   
5: end for   
6: Remove all edges incident to any m $\in \ C ;$ add edges from   
teachers in $\tau$ actually used   
7: return updated models, updated G

Concentrating high-influence samples into fewer shards increases per-shard sensitivity to a single sample’s removal, but reduces $| { \cal { A } } |$ by 3–5×; the global perturbation $| { \mathcal { A } } | \times$ $\operatorname* { m a x } _ { m } \| \Delta \theta _ { m } \|$ therefore decreases overall (Appendix D). Partition staleness, the fact that influence scores and centroids are computed once on D and not recomputed after each deletion, introduces a bounded, separate efficiency gap of $O ( | D _ { f } | / N )$ (Appendix F), distinct from $\epsilon _ { \mathrm { c e r t } }$ and from causal exactness.

## G. Certification under repeated deletion requests

Section III-F’s certification holds for a single request against an empty distillation history. Cross-shard distillation, absent from vanilla SISA, creates dependencies between shards that a sequence of requests must account for: if shard $j$ was retrained using $\theta _ { i }$ as a teacher, and a later request deletes data originally in shard i, retraining i alone leaves shard j causally dependent on the deleted sample through the earlier distillation step.

Definition 3 (Distillation Dependency Graph). $G = ( V , E )$ $V ~ = ~ \{ 1 , \ldots , M \}$ , initialized empty. Whenever shard $j$ is retrained using teachers $\{ \theta _ { i } : i \in \mathcal { U } \}$ , add $( i  j )$ for every i used. Edges accumulate across requests and are updated only by Algorithm 3.

Definition 4 (Reachability Closure). For $A _ { 0 } \subseteq V \colon C ( A _ { 0 } ) =$ $A _ { 0 } \cup \{ j \in V : \exists ~$ a path from some $m \in A _ { 0 }$ to j in G}.

A dependency chain can grow to any length: a shard may teach another before it is itself retrained, and once retrained using a different teacher, extend the chain further. We therefore rely on no bound on $| C ( A _ { 0 } ) | ;$ correctness below follows from exact closure computation, not from typical chain length.

Because every shard being repaired draws teacher signal only from ${ \mathcal { T } } = V \backslash C$ (shards outside the closure, guaranteed causally exact by the closure’s own definition), repair order within $C$ is immaterial, and this holds regardless of whether G restricted to $C$ contains cycles.

Theorem 2 (Causal Exactness Under Cascading Deletion). If no shard is causally dependent on any previously-deleted sample before a request, then after Algorithm 3 processes that request, every shard is causally exact with respect to $D _ { f }$ (Definition 1), for any structure of G. By induction over the request sequence, this extends to the full aggregated model under any finite sequence of deletions. (Proof: Appendix C.)

![](images/66b5672de45394bc30dec8c5dbdb193aab8cff37233d84e462f56c3a5a39d8ea.jpg)  
Fig. 1. IDSISA’s three phases: influence-aware partitioning (I), slice-based shard training (II), and distillation-guided unlearning using unaffected shards as teachers (III).

Corollary 2 (Retraining Cost). The number of shards retrained equals |C|, directly computable by graph traversal. Best case: $| C | = | A _ { 0 } |$ , identical cost to Algorithm 2. Worst case: $| C | = M ,$ matching full SISA. We do not assume |C| is typically small; Appendix I reports measured closure sizes under two sequential-deletion traces, one matching this paper’s own experimental request pattern and one adversarially constructed to stress the mechanism.

Under the paper’s own experimental regime (a single, isolated deletion per architecture), G is empty and Algorithm 3 reduces exactly to Algorithm 2; no result in Sections V–VI is affected by this extension. Under sustained deletion, an optional teacher-freshness policy (excluding any once-retrained shard from future teaching) monotonically shrinks the eligible teacher pool; when it is exhausted, retraining proceeds without distillation on the partition already produced by Phase 1 (the $| \boldsymbol { { u } } | ~ = ~ 0$ branch), which is exactly the “Influence Only” configuration in Table XIII and already outperforms baseline SISA on every metric. We do not claim IDSISA dominates SISA at intermediate levels of teacher-pool depletion; the depletion timeline, checkpoint-storage accounting, and the fact that Theorem 2 governs parameters rather than predictions already served to users, are developed fully in Appendix C.

## IV. SYSTEM ARCHITECTURE

Figure 1 shows the full pipeline. Partitioning is a onetime cost amortized across all future requests; each subsequent

isolated request retrains 20–30% of shards rather than the 80– 90% typical of random SISA partitioning, generalizing under sequential requests to the reachability closure of Section III-G.

## V. EXPERIMENTAL METHODOLOGY

## A. Datasets and tasks

CIFAR-10: 60,000 images, 10 classes; unlearning removes the “cat” class (5,000 samples). IMDB: 50,000 reviews, tokenized to 200 tokens; unlearning targets 1,000 positive reviews. Fashion-MNIST: 70,000 images, 10 classes; unlearning removes 6,000 samples (600/class). CIFAR-100: 60,000 images, 100 fine classes grouped into 20 coarse superclasses; unlearning removes the large\_carnivores superclass (bear, leopard, lion, tiger, wolf; ≈2,500 samples, 5% of training data), testing class-level forgetting at higher label cardinality than CIFAR-10. These scenarios span classlevel, superclass-level, sentiment-targeted sample-level, and broad distributed removal; Section VI-H additionally tests a uniformly-distributed, class-agnostic forget set on CIFAR-10 as a harder case for influence-based concentration. IMDB is additionally evaluated on a TCN architecture (Section VI-E) using the same 1,000-positive-review forget set as the BiLSTM experiment, to test a modern non-recurrent sequence model. All experiments involve a single deletion request per architecture (Section III-G).

## B. Model architectures and training

ResNet-18 (CIFAR-10, CIFAR-100; 11.2M params, SGD momentum 0.9, weight decay $5 \times 1 0 ^ { - 4 }$ , lr 0.01, identical architecture and training recipe on both datasets), BiLSTM (IMDB, 4.2M params, two bidirectional layers, 256 units/direction, Adam lr 0.001), a fully connected ANN (Fashion-MNIST, 784→256→10, 203K params, Adam lr 0.001), and a A TCN (IMDB, 3.1M params, 4 dilated causal convolutional layers with kernel size 3 and dilation doubling per layer [1, 2, 4, 8], Adam lr 0.001). Unlike vanilla RNNs, TCNs use dilated causal convolutions rather than recurrent connections, so no BPTT gradient accumulation occurs and gradient clipping is unnecessary; Phase 1’s influence computation is therefore valid without modification (Section VI-E). A vanilla RNN of matching capacity was also tested; its incompatibility is analyzed in Section VI-G. Transformer-based architectures are not evaluated; this is an experimental scope limitation rather than a theoretically justified incompatibility, and we leave their evaluation to future work.

## C. Evaluation metrics

Forgetting quality is reported via two distinct metric types, and Tables VI–VII should be read accordingly rather than assuming a single shared scale: residual accuracy on the forgotten class or samples (CNN; lower is better, approaching the random-chance floor), and forgetting gap, the drop in target-class accuracy caused by unlearning (LSTM, ANN; a larger gap indicates stronger forgetting). For ANN, this gap is reported as a negative value since it is computed relative to a full pre-unlearn baseline; a more negative value indicates stronger forgetting, matching the sign convention used in Table VIII. Knowledge preservation (retained-class accuracy, preservation loss) and efficiency (training/unlearning time, memory, speedup vs. full retraining) are reported uniformly across architectures. All values are means over 5 seeds (42– 46).

## VI. RESULTS

## A. Overall summary

Table IV aggregates results across architectures; Table V compares against prior methods on CIFAR-10. Gains span 14– 91% better forgetting, up to 50% better preservation in relative terms (with consistent percentage-point gains on LSTM and ANN), and unlearning-time changes ranging from −6.7% (ANN, slower) to +6.4% (CNN, faster) depending on architecture, without compromising the deletion guarantee.

Table V makes the certification/efficiency tradeoff concrete: approximate methods (SalUn, SCRUB, SSD) reach lower forget-class accuracy than IDSISA in some cases, but none carry a formal deletion guarantee, and their published retention/test-accuracy ranges vary by implementation. Against SISA specifically (the only other certified method in the table), IDSISA improves every column simultaneously (forget accuracy, retention, test accuracy, and MI confidence), the comparison that matters, since SISA is the baseline ID-SISA is designed to improve, not replace.

## B. CNN: ResNet-18 on CIFAR-10

Residual accuracy on the forgotten class drops to 12.3%, near the 10% random-guess floor, while preservation loss falls 50% (0.8% vs. 1.6%). The 6.4% unlearning-time reduction per request amortizes the modest 10.4% training-time overhead (the early-stopped proxy baseline, 66.1 vs. 59.9 min) over multiple requests, per the break-even analysis in Section VII-B.

## C. LSTM: BiLSTM on IMDB

IDSISA is the only method here that increases retained-class accuracy after unlearning (+0.65% vs. −3.10%), the clearest evidence that distillation mitigates the collateral damage isolated retraining otherwise causes.

## D. ANN: Fashion-MNIST

Removing 6,000 samples produces zero test-accuracy loss under IDSISA, a 14.3% forgetting-quality improvement over Random-SISA at a modest 6.7% time cost.

## E. TCN: Dilated Convolutional Network on IMDB

We evaluate a TCN on IMDB using the same 1,000- positive-review forget set as Section VI-C. IDSISA outperforms SISA on every metric (Table IX): forgetting gap +9.0% relative (41.96% vs. 38.48%), retained negative accuracy +1.09 pts (73.08% vs. 71.99%), affected shards 8 vs. 10 (20% reduction). Both methods drop post-unlearn positive accuracy well below the 50% random baseline; we attribute this to pershard class-imbalance collapse, since the effect is comparable under SISA and IDSISA alike.

## F. CNN: ResNet-18 on CIFAR-100

We extend the CNN evaluation to CIFAR-100, removing the large\_carnivores superclass (bear, leopard, lion, tiger, wolf; fine classes 3, 42, 43, 88, 97; ≈2,500 samples, 5% of training data) with the identical ResNet-18 architecture used for CIFAR-10. Both methods report 0.0% forget-superclass accuracy (Table X), which we do not attribute to the unlearning mechanism: overall test accuracy is only ≈21% because M=10 isolated shards give each shard ≈50 samples per class on a 100-class task. The metrics that differentiate meaningfully are retention and concentration: IDSISA improves retained-class accuracy by 0.74 pts (+3.5% relative, 21.89% vs. 21.15%) and achieves the strongest shard-concentration result of any experiment in this paper, 7 vs. 10 shards affected (30% reduction). This is a limitation finding: the M=10 shard budget, calibrated for a 10-class task, does not transfer cleanly to measuring class-level forgetting at 100-class granularity.

## G. Vanilla RNN incompatibility

Vanilla RNNs violate Assumption 3: gradient clipping, needed for training stability on sequences >20–30 steps, replaces true gradients with truncated proxies, corrupting the L-BFGS influence approximation. This is a Hessian-stability failure, not an architecture-type failure per se: baseline SISA on vanilla RNN (no partitioning, no distillation) produces equivalent collapse (Table XI), confirming the failure is not specific to IDSISA’s phases. LSTM and GRU gating avoids it because multiplicative gates bound the effective gradient norm through the cell-state highway, keeping gradients well-behaved without clipping; this is why BiLSTM requires clipping on under 0.1% of training steps versus vanilla RNN’s pervasive ∼1.1%, and why its influence scores have an order of magnitude lower variance. TCN (Section VI-E) avoids the same failure by design: dilated causal convolutions have no recurrent connections and no BPTT gradient accumulation, so gradient clipping is unnecessary and Assumption 3 is satisfied unconditionally. Compatibility with Phase 1’s influence machinery is therefore governed by gradient stability, not by architecture family per se. Appendices K–M give the full failure-mode analysis.

## H. Uniform forget set: distributed deletion

To directly evaluate partitioning robustness under a deletion pattern with no class or semantic co-location, we evaluate a uniformly-distributed forget set of 500 CIFAR-10 samples (≈50/class), with no semantic co-location. IDSISA still reduces affected shards from 10 to 8 (Table XII), because Kmeans clustering concentrates similar feature-space representations regardless of class label; gains are smaller than the class-structured case, since concentration benefit is strongest when $D _ { f }$ has class structure.

## I. Ablation study

Influence partitioning alone accounts for ∼35% of the total forgetting-quality gain (18.6%→16.4%); distillation contributes the remaining ∼65% (16.4%→12.3%), with a superadditive effect (27% reduction in cross-shard divergence)

TABLE IV  
OVERALL PERFORMANCE COMPARISON: SISA VS. IDSISA ACROSS FIVE ARCHITECTURE/DATASET PAIRS (MEAN OVER 5 SEEDS, 42–46). <sup>‡</sup>PRESERVATION IMPROV. IS A RELATIVE % CHANGE WHERE THE BASELINE IS NON-ZERO AND SAME-SIGNED (CNN/CIFAR-10, CNN/CIFAR-100); WHERE THE BASELINE IS NEGATIVE, NEAR-ZERO, OR (CIFAR-100) NOT ATTRIBUTABLE TO THE UNLEARNING MECHANISM, A RELATIVE % IS UNDEFINED OR NOT MEANINGFUL, SO WE REPORT THE RAW PERCENTAGE-POINT (PT) CHANGE INSTEAD. <sup>§</sup>BOTH METHODS REPORT 0.0% FORGET-SUPERCLASS ACCURACY ON CIFAR-100; SECTION VI-F EXPLAINS WHY THIS IS NOT REPORTED AS A FORGETTING-QUALITY RATIO.
<table><tr><td>Architecture</td><td>Dataset</td><td>Params</td><td>Forgetting Improv.</td><td>Preservation Improv.‡</td><td>Unlearn Speedup</td></tr><tr><td>LSTM (BiLSTM)</td><td>IMDB Sentiment</td><td>4.2M</td><td>+91% (3.94 vs 2.06%)</td><td>+3.75 pts (+0.65 vs -3.10%)</td><td>+2.1% (93 vs 95s)</td></tr><tr><td>CNN (ResNet-18)</td><td>CIFAR-10 Image</td><td>11.2M</td><td>+33.9% (12.3 vs 18.6%)</td><td>+50% (0.8 vs 1.6%)</td><td>+6.4% (81.2 vs 86.7m)</td></tr><tr><td>ANN (Fully-Conn.)</td><td>Fashion-MNIST</td><td>203K</td><td>+14.3% (-9.57 vs -8.37%)</td><td>+0.03 pts (+0.03 vs 0.00%)</td><td>-6.7% (534 vs 570s)</td></tr><tr><td>TCN (Dilated Conv.)</td><td>IMDB Sentiment</td><td>3.1M</td><td>+9.0% (41.96 vs 38.48%)</td><td>+1.09 pts (73.08 vs 71.99%)</td><td>+0.6% (67.8 vs 68.2s)</td></tr><tr><td>CNN (ResNet-18)</td><td>CIFAR-100 Image</td><td>11.2M</td><td>N/A§ (0.0 vs 0.0%)</td><td>+3.5% (21.89 vs 21.15%)</td><td>+0.5% (915.0 vs 920.0s)</td></tr></table>

TABLE V

COMPARISON WITH PRIOR METHODS ON CIFAR-10/RESNET-18, CLASS-LEVEL FORGETTING (∼5,000 SAMPLES). <sup>†</sup>NO FORMAL DELETION GUARANTEE. <sup>‡</sup>SEE TABLE I: RETRAINING CERTIFIED, DISTILLATION GAP $\epsilon _ { \mathrm { C E R T } } \leq 0 . 0 1 5 .$
<table><tr><td>Method</td><td>Certified?</td><td>Forget Acc (%) ↓</td><td>Retain Acc (%) ↑</td><td>Test Acc (%) ↑</td><td>MIA on Df</td></tr><tr><td>Full Retrain (Oracle)</td><td>Yes</td><td>~10.0</td><td>~93.4</td><td>~93.4</td><td>Calibrated</td></tr><tr><td>SalUn [21]</td><td>No†</td><td>~0-6</td><td>~88-92</td><td>~88-92</td><td>Low</td></tr><tr><td>SCRUB [17]</td><td>No†</td><td>~5-8</td><td>~83-85</td><td>~83-85</td><td>Moderate</td></tr><tr><td>SSD [18]</td><td>No†</td><td>~2-8</td><td>~82-86</td><td>~82-86</td><td>Calibrated</td></tr><tr><td>SISA [1]</td><td>Yes</td><td>18.6</td><td>83.2</td><td>82.3</td><td>0.134</td></tr><tr><td>IDSISA</td><td>Yes†</td><td>12.3</td><td>84.7</td><td>83.1</td><td>0.132</td></tr></table>

TABLE VI

CNN UNLEARNING PERFORMANCE, CIFAR-10 (MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Metric</td><td>SISA</td><td>IDSISA</td></tr><tr><td>Initial Training</td><td></td><td></td></tr><tr><td>Test Acc. (Pre-Unlearn) (%) Training Time (min)</td><td>83.8 59.9</td><td>84.6 66.1</td></tr><tr><td>Forgetting Quality</td><td></td><td></td></tr><tr><td>Forget Class Acc (%)</td><td>18.6</td><td>12.3</td></tr><tr><td>Effectiveness (% removed)</td><td>80.1</td><td>86.4</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Knowledge Preservation</td><td></td><td></td></tr><tr><td>Retention Acc (%) Preservation Loss (%)</td><td>83.2 1.6</td><td>84.7</td></tr><tr><td></td><td></td><td>0.8</td></tr><tr><td>Efficiency</td><td></td><td></td></tr><tr><td>Unlearning Time (min)</td><td>86.7</td><td>81.2</td></tr><tr><td>Speedup vs Retrain</td><td>1.32×</td><td>1.39×</td></tr></table>

attributable to influence partitioning producing cleaner teacherstudent separation than random shards would. The “+ Influence Only” row is also IDSISA’s floor under complete teacher-pool depletion (Section III-G): it already dominates baseline SISA with no distillation signal at all.

## J. Certification analysis

Table XIV reports the mean certification gap (0.011 average, all ≤ 0.013) and the 99th-percentile tail (0.023 average, 0.019–

TABLE VII  
LSTM UNLEARNING PERFORMANCE, IMDB, 1K POSITIVE REVIEWS REMOVED (MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Metric</td><td>SISA</td><td>IDSISA</td></tr><tr><td>Initial Training</td><td></td><td></td></tr><tr><td>Test Acc. (Pre-Unlearn) (%)</td><td>79.55</td><td>79.07</td></tr><tr><td>Training Time (s)</td><td>120.16</td><td>138.32</td></tr><tr><td>Forgetting Quality</td><td></td><td></td></tr><tr><td>Target Class Accuracy (%)</td><td>77.94</td><td>74.74</td></tr><tr><td>Forgetting Gap (%)</td><td>2.06</td><td>3.94</td></tr><tr><td>Knowledge Preservation</td><td></td><td></td></tr><tr><td>Retained Class Accuracy (%)</td><td>78.05</td><td>80.42</td></tr><tr><td>Preservation Change (%)</td><td>-3.10</td><td>+0.65</td></tr><tr><td>Efficiency</td><td></td><td></td></tr><tr><td>Unlearning Time (s)</td><td>95.26</td><td>93.28</td></tr></table>

TABLE VIII

ANN UNLEARNING PERFORMANCE, FASHION-MNIST, 6K SAMPLES REMOVED (MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Metric</td><td>Random SISA</td><td>IDSISA</td></tr><tr><td>Test Acc. (Pre-Unlearn) (%)</td><td>85.86</td><td>84.02</td></tr><tr><td>Training Time (s)</td><td>534.0</td><td>546.0</td></tr><tr><td>Forgetting Gap (%)</td><td>-8.37</td><td>-9.57</td></tr><tr><td>Post-Unlearn Accuracy (%)</td><td>85.86</td><td>84.02</td></tr><tr><td>Unlearning Time (s)</td><td>534.0</td><td>570.0</td></tr></table>

0.027) separately, because these are not both bounded by the 0.015 theoretical figure: Theorem 1’s bound is derived and validated as a mean-case quantity, not a uniform bound, consistent with Assumptions 1–3 being convex surrogates for non-convex architectures. Approximately 5% of test samples (the high-influence boundary cases whose representations shift most when the forgotten class’s contribution is removed) exceed the mean-case bound (Figure 2); we report both so practitioners needing a worst-case margin can use the larger one.

The shadow model correctly separates general training members from non-members (TPR ≈0.60 vs. random 0.50), confirming calibration, then shows near-identical, near-random $D _ { f }$ confidence for both SISA and IDSISA (∆conf = 0.002, within measurement variance): distillation adds no material membership-inference surface beyond SISA’s own baseline. The full-retrain model’s higher $D _ { f }$ confidence (0.681) reflects generalizable low-level visual features the cat class shares with the other nine CIFAR-10 classes (Appendix B); SISA and IDSISA’s shared near-random confidence follows from sharding’s ensembling effect, not distillation, since SISA shows it equally without any distillation term [13]. Appendix O gives the complete attack methodology and threshold calibration.

TABLE IX  
TCN UNLEARNING PERFORMANCE, IMDB, 1K POSITIVE REVIEWS REMOVED (MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Metric</td><td>SISA</td><td>IDSISA</td></tr><tr><td>Pre-Unlearn Positive Acc (%)</td><td>69.08</td><td>70.15</td></tr><tr><td>Post-Unlearn Positive Acc (%)</td><td>30.60</td><td>28.19</td></tr><tr><td>Forgetting Gap (%)</td><td>38.48</td><td>41.96</td></tr><tr><td>Retained Negative Acc (%)</td><td>71.99</td><td>73.08</td></tr><tr><td>Affected Shards</td><td>10/10</td><td>8/10</td></tr><tr><td>Unlearning Time (s)</td><td>68.23</td><td>67.83</td></tr></table>

TABLE X

CNN (RESNET-18) UNLEARNING PERFORMANCE, CIFAR-100, LARGE\_CARNIVORES SUPERCLASS REMOVED (≈2,500 SAMPLES; MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Metric</td><td>SISA</td><td>IDSISA</td></tr><tr><td>Pre-Unlearn Test Acc (%)</td><td>20.68</td><td>21.13</td></tr><tr><td>Forget Superclass Acc (%)</td><td>0.0</td><td>0.0</td></tr><tr><td>Retain Acc (%)</td><td>21.15</td><td>21.89</td></tr><tr><td>Post-Unlearn Test Acc (%)</td><td>20.09</td><td>20.95</td></tr><tr><td>Affected Shards</td><td>10/10</td><td>7/10</td></tr><tr><td>Unlearning Time (s)</td><td>920.03</td><td>915.04</td></tr></table>

## VII. DISCUSSION

## A. Architecture-specific behavior

LSTM gating localizes influence, yielding the largest forgetting gain (91%), consistent preservation (+3.75 pts, from −3.10% to +0.65%), and lowest influence-score variance; CNN spatial invariance isolates class features cleanly (12.3% residual, near the 10% floor); the fully-connected ANN shows stable, near-lossless forgetting at 6,000 removals; TCN improves every reported IDSISA metric (+9.0% forgetting gap, +1.09 pts retention, 20% fewer affected shards) and supports Hessian-stability compatibility for non-recurrent sequence models, though sub-50% post-unlearn accuracy reflects the same per-shard positive-sample collapse seen under SISA (Section VI-E); CIFAR-100 (ResNet-18) achieves the strongest shard concentration (30% reduction, 7 vs. 10 shards) and +0.74 pts retention, while its 0.0% forget-superclass accuracy under both methods reflects low overall accuracy rather than unlearning (Section VI-F); vanilla RNNs fail via the Hessian-stability mechanism in Section VI-G.

## B. Deployment trade-offs

Table XVI gives the break-even analysis under the singledeletion-per-request regime. IDSISA is marginally more expensive than SISA after a single request (147.3 vs. 146.6 min) but breaks even by the second; the 5.3-minute proxy baseline is the only IDSISA-specific upfront cost. This assumes independent requests and is a lower bound once cross-request dependencies accumulate (Section III-G); per Section VII-C, cost can collapse to full-SISA under adversarial deletion patterns.

TABLE XI  
RNN INCOMPATIBILITY: SISA-RNN VS. IDSISA-RNN VS. BILSTM, IMDB.
<table><tr><td>Metric</td><td>SISA-RNN</td><td>IDSISA-RNN</td><td>BiLSTM</td></tr><tr><td>Pre-Unlearn Acc (%)</td><td>52.3</td><td>52.3</td><td>79.6</td></tr><tr><td>Post-Unlearn Acc (%)</td><td>53.4</td><td>52.3</td><td>77.6</td></tr><tr><td>Retained Class Acc (%)</td><td>66.0</td><td>51.8</td><td>80.4</td></tr><tr><td>Forgetting Gap (%)</td><td>25.3†</td><td>0.1</td><td>3.94</td></tr><tr><td>Influence Time (s)</td><td>N/A</td><td>342</td><td>1.34</td></tr><tr><td>Clip Events (train)</td><td>5,355</td><td>High</td><td>Low</td></tr></table>

<sup>†</sup> Asymmetric class collapse; overall accuracy near-random.

TABLE XII  
UNIFORM FORGET SET, CIFAR-10, $D _ { f } = 5 0 0$ (MEAN OVER 5 SEEDS,42–46).
<table><tr><td>Metric</td><td>SISA</td><td>IDSISA</td></tr><tr><td>Affected Shards</td><td>10/10</td><td>8/10</td></tr><tr><td>Forget Accuracy (%)</td><td>15.9</td><td>13.1</td></tr><tr><td>Post-Unlearn Acc (%)</td><td>83.0</td><td>83.6</td></tr><tr><td>Preservation Loss (%)</td><td>1.4</td><td>0.9</td></tr><tr><td>Unlearning Time (s)</td><td>5,210</td><td>3,720</td></tr></table>

The reset row shows that K = 5 adds $5 9 . 9 / 5 \approx 1 2 . 0$ min per request, exceeding the 5.5 min/request saved over SISA — eliminating rather than capping the advantage. Preserving the break-even requires $K \gtrsim 1 1$ , consistent with the ≈10–15- request natural teacher-pool exhaustion timeline (Remark 6, Appendix C).

Use IDSISA when certified deletion, frequent requests, and utility preservation matter; prefer plain SISA only when storage is heavily constrained, deletion volume is expected to stay strictly at one request, or sustained deletion would force a reset cadence tight enough to erode the per-request savings.

## C. Cascading deletion cost under sustained requests

Section III-G’s certification guarantee holds unconditionally, but its cost does not stay small under sustained deletion. Under a single, isolated request (the regime used throughout Sections V–VI), the cascading mechanism reduces to Algorithm 2, and every reported efficiency number reflects that regime only. Under an adversarial sequence of repeated, non-reset class-structured deletions (Appendix I), the cascade collapses to the full worst case (retraining all M = 10 shards, matching plain SISA’s cost) in 2 of 8 requests (25%), with a cascade of some size in 5 of 8 (62.5%). This qualifies the efficiency claims elsewhere in this paper: they describe the single-deletion regime IDSISA was evaluated in, not a guarantee under sustained or adversarial pressure, where perrequest cost can rise to that of the baseline IDSISA is designed to improve upon.

TABLE XIII  
IMPACT OF IDSISA COMPONENTS, CNN/CIFAR-10 (MEAN OVER 5 SEEDS, 42–46).
<table><tr><td>Configuration</td><td>Forget (%)</td><td>Retain (%)</td><td>Time (min)</td></tr><tr><td>SISA (Baseline)</td><td>18.6</td><td>83.2</td><td>86.7</td></tr><tr><td>+ Influence Only</td><td>16.4</td><td>83.8</td><td>84.3</td></tr><tr><td> $+ \ { \mathrm { I n f l u e n c e } } \ + \ { \mathrm { D i s t i l l } } .$ </td><td>12.3</td><td>84.7</td><td>81.2</td></tr><tr><td> $\alpha = 0 . 3$ </td><td>14.1</td><td>84.2</td><td>82.6</td></tr><tr><td> $\alpha = 0 . 5$  (default)</td><td>12.3</td><td>84.7</td><td>81.2</td></tr><tr><td> $\alpha = 0 . 7$ </td><td>13.8</td><td>84.1</td><td>81.8</td></tr><tr><td> $T = 1 . 0$ </td><td>15.2</td><td>83.9</td><td>82.4</td></tr><tr><td> $T = 2 . 0$  (default)</td><td>12.3</td><td>84.7</td><td>81.2</td></tr><tr><td> $T = 3 . 0$ </td><td>13.1</td><td>84.3</td><td>81.7</td></tr></table>

TABLE XIV

CERTIFICATION GAPS ACROSS ARCHITECTURES (MEAN OVER 5 SEEDS, 42–46). ϵ<sub>CERT</sub> MEASURES DISTILLATION APPROXIMATION ONLY, NOT DELETION CORRECTNESS.
<table><tr><td>Architecture</td><td> $\epsilon _ { \mathrm { c e r t } }$  (mean)</td><td>Max Gap (99%ile)</td><td>Theoretical Bound (mean)</td></tr><tr><td>CNN (CIFAR-10)</td><td>0.011</td><td>0.024</td><td>0.015</td></tr><tr><td>LSTM (IMDB)</td><td>0.008</td><td>0.019</td><td>0.015</td></tr><tr><td>ANN (Fashion-MNIST)</td><td>0.013</td><td>0.027</td><td>0.015</td></tr><tr><td>Average</td><td>0.011</td><td>0.023</td><td>0.015</td></tr></table>

## D. Limitations

Vanilla RNNs are unsuitable (Section VI-G). Transformerbased architectures are excluded as an experimental limitation, not an incompatibility (Section V.B). CIFAR-100 class-level forgetting (Section VI-F) also exposes a limitation of the evaluation protocol itself, not just of specific architectures: at M=10 shards and 100 classes, each shard sees too few samples per class for both baseline accuracy and class-level forgetting to be measured cleanly, so we report that result as a scaling limitation rather than as a fifth positive data point. The current experiments also use a non-private Phase 1 partitioning stage: influence scores, proxy features, and Kmeans centroids are computed once on D and held fixed. Partition staleness introduces a bounded $O ( | D _ { f } | / N )$ efficiency gap (Section III-F); Appendix F gives an algorithmic DP-Lloyd extension that would bound the sensitivity of the partition metadata, provided the feature/influence proxy is fixed or DP-trained. We do not implement or tune that trade-off here, so orchestration-metadata privacy remains a limitation of the present system. Certification under repeated deletion requires the cascading mechanism of Section III-G, whose worst-case cost can approach full-SISA retraining; Appendix I reports measured costs under two request patterns.

Practical significance of theoretical assumptions. Assumptions 1–3 are convex surrogates that do not hold strictly for ResNet-18 or BiLSTM. For overparameterized networks, weaker local regularity near convergence (e.g., Łojasiewicztype conditions) can partially preserve the bound’s qualitative behavior, though we do not verify this directly. The primary validation is empirical: measured $\epsilon _ { \mathrm { c e r t } }$ values (mean 0.008– 0.013, Table XIV) fall below the heuristic 0.015 threshold, confirming the bound is directionally informative at the problem scales evaluated here.

![](images/2bd94845bda2428ae1d5990fbc716d47c83af73b59710b2eb6353d1e51610e64.jpg)  
Fig. 2. Distribution of $\epsilon _ { \mathrm { c e r t } } .$ , CNN/CIFAR-10, 10K test samples. The theoretical bound (dashed) reflects Theorem 1’s mean-case prediction, not a uniform bound over the tail.

TABLE XV  
MEMBERSHIP INFERENCE ATTACK ON $D _ { f } .$ , CIFAR-10/RESNET-18. LOWER CONFIDENCE/TPR = STRONGER UNLEARNING.
<table><tr><td>Model</td><td> $D _ { f }$  Mean Conf.</td><td> $D _ { f }$  TPR</td><td> $D _ { \mathbf { t r a i n } }$  TPR</td><td>Threshold</td></tr><tr><td>SISA (Unlearned)</td><td>0.134</td><td> ${ \sim } 0 . 4 2$ </td><td>~0.60</td><td>~0.139</td></tr><tr><td>IDSISA (Unlearned)</td><td>0.132</td><td>~0.39</td><td>~0.59</td><td>~0.139</td></tr><tr><td>Full Retrain  $( D _ { r } \ \mathrm { o n l y } )$ </td><td>0.681</td><td>0.83</td><td>N/A</td><td>0.738</td></tr></table>

## E. Scalability

Influence computation is O(N p) and amortizes across all future requests (38–45s for CIFAR-10 vs. hours of training). M is chosen per deployment to balance throughput against storage $( M ~ = ~ 1 0$ for the $N \ = \ 5 0 \mathrm { k }$ datasets here; larger datasets would likely warrant larger M, e.g., 32, determined empirically). At N=500K, projected Phase 1 setup cost is on the order of a few minutes, since per-request unlearning cost depends only on |A| and $| D _ { f } |$ , not N; this is a complexity projection, not a measurement. The CIFAR-100 result (Section VI-F) shows that class cardinality is a separate scaling axis: holding M fixed, forgetting-quality measurement degrades before efficiency does. Use IDSISA for: GDPR/CCPArelevant deployments with recurring deletion requests over datasets larger than 10k samples. Prefer plain SISA when: the architecture is a vanilla RNN or otherwise Hessian-unstable, the dataset is under ∼1k samples, deletions are rare oneoff events, or storage is severely constrained, since Phase 1’s setup cost and the sequential-deletion checkpointing overhead (Appendix C) are only worthwhile when amortized across multiple requests.

## VIII. CONCLUSION

IDSISA combines influence-aware partitioning with distillation-guided retraining, giving 14–91% better forgetting and up to 50% better utility retention in relative terms, with consistent percentage-point utility gains across architectures, than SISA across CNN, LSTM, TCN, and ANN architectures. Extending the CNN evaluation to CIFAR-100 shows this advantage is not unconditional, at higher class cardinality and a fixed shard budget forgetting-quality measurement degrades before efficiency does(Section VI-F). We additionally close a certification gap specific to repeated deletion under cross-shard distillation, proving that closure-based cascading retraining restores exact deletion to the full aggregated model under any request sequence, at a cost that is computed rather than assumed.

TABLE XVI  
AMORTIZED COST VS. REQUESTS, CIFAR-10/CNN,  
SINGLE-DELETION-PER-REQUEST REGIME. RESET ROW APPLIES A PERIODIC FULL-SHARD REFRESH EVERY $K = 5$ REQUESTS (REMARK 6, APPENDIX C).
<table><tr><td>Method</td><td> $N = 1$ </td><td> $N = 5$ </td><td> $N = 1 0$ </td></tr><tr><td>Full Retrain</td><td>8-10 h</td><td>40–50 h</td><td>80-100 h</td></tr><tr><td>SISA</td><td>146.6 min</td><td>493.4 min</td><td>926.9 min</td></tr><tr><td>IDSISA</td><td>147.3 min</td><td>472.1 min</td><td>878.1 min</td></tr><tr><td>IDSISA + Reset  $( K = 5 )$ </td><td>147.3 min</td><td>532.0 min</td><td>997.9 min</td></tr></table>

We regard this certification analysis as characteristic of the paper’s broader contribution: introducing cross-shard distillation creates certification obligations that isolated SISA shards never had, and closing them formally, is itself part of what IDSISA delivers. Influence-aware partitioning and distillationguided retraining are individually established techniques; their combination here produces measurably superadditive gains (27% reduction in cross-shard divergence beyond either component alone, Table XIII) because influence partitioning creates the clean shard separation that makes low-leakage distillation possible in the first place.

## A. Reproducibility statement

IDSISA is implemented in PyTorch 2.2 on an NVIDIA A100. All reported main-text results are means over 5 random seeds (42, 43, 44, 45, 46), governing data shuffling, shard initialization, and mini-batch ordering.

Code availability: Implementation code, configuration files, and the sequential-deletion trace scripts underlying Appendices G–J will be released publicly upon acceptance.

Hyperparameters: $T = 2 . 0 , \alpha = 0 . 5 , \lambda = 0 . 0 1$ , PCA variance 0.95, $\beta = 0 . 3$

Architectures: ResNet-18 (11.2M) for CIFAR-10 and CIFAR-100; BiLSTM (4.2M, 256 hidden, 2 layers) for IMDB; TCN (3.1M, 4 dilated causal convolutional layers, kernel size 3, dilation [1,2,4,8]) for IMDB; FC-ANN (203K, 256 hidden) for Fashion-MNIST.

Training: SGD (momentum 0.9, decay $5 \times 1 0 ^ { - 4 } .$ , lr 0.01, StepLR step=20, $\gamma = 0 . 1 )$ for CIFAR-10 and CIFAR-100; Adam (lr 0.001) for IMDB (BiLSTM and TCN) and Fashion-MNIST.

Data: CIFAR-10 and CIFAR-100 normalized with random crops/flips; IMDB truncated/padded to 200 tokens, GloVe-100d; Fashion-MNIST scaled to [0, 1].

Shards: M = 10 (CIFAR-10, CIFAR-100, IMDB), $M =$ 15 (Fashion-MNIST); $K _ { s } ~ = ~ 1 0$ (CIFAR-10, CIFAR-100), $K _ { s } = 5 \ \mathrm { ( I M D B }$ , Fashion-MNIST).

## REFERENCES

[1] L. Bourtoule, V. Chandrasekaran, C. A. Choquette-Choo, H. Jia, A. Travers, B. Zhang, D. Lie, and N. Papernot, “Machine unlearning,” in 2021 IEEE Symposium on Security and Privacy (SP). IEEE, 2021, pp. 141–159.

[2] Y. Cao and J. Yang, “Towards making systems forget with machine unlearning,” in 2015 IEEE Symposium on Security and Privacy. IEEE, 2015, pp. 463–480.

[3] A. Ginart, M. Guan, G. Valiant, and J. Y. Zou, “Making ai forget you: Data deletion in machine learning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 32, 2019, pp. 3513–3526.

[4] C. Guo, T. Goldstein, A. Hannun, and L. van der Maaten, “Certified data removal from machine learning models,” in International Conference on Machine Learning (ICML). PMLR, 2020, pp. 3832–3842.

[5] A. Koloskova, Y. Allouah, A. Jha, R. Guerraoui, and S. Koyejo, “Certified unlearning for neural networks,” in International Conference on Machine Learning (ICML), 2025.

[6] A. Golatkar, A. Achille, and S. Soatto, “Eternal sunshine of the spotless net: Selective forgetting in deep networks,” in 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2020, pp. 9304–9312.

[7] A. Golatkar, A. Achille, A. Ravichandran, and S. Soatto, “Mixedprivacy forgetting in deep networks,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2021, pp. 792–801.

[8] S. Neel, A. Roth, and S. Sharifi-Malvajerdi, “Descent-to-delete: Gradient-based methods for machine unlearning,” in Algorithmic Learning Theory (ALT). PMLR, 2021, pp. 931–962.

[9] P. W. Koh and P. Liang, “Understanding black-box predictions via influence functions,” in International Conference on Machine Learning (ICML). PMLR, 2017, pp. 1885–1894.

[10] G. Pruthi, F. Liu, S. Kale, and M. Sundararajan, “Estimating training data influence by tracing gradient descent,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 19 920– 19 930.

[11] A. Warnecke, L. Pirch, C. Wressnegger, and K. Rieck, “Machine unlearning of features and labels,” in Network and Distributed System Security Symposium (NDSS). Internet Society, 2021.

[12] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[13] Y. Quan, Z. Li, and G. Montana, “Efficient verified machine unlearning for distillation,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 38, 2025.

[14] X. Fan, J. Wu, M. Zhou, P. Liang, M. Harandi, and D. Phung, “IMU: Influence-guided machine unlearning,” arXiv preprint arXiv:2508.01620, 2025.

[15] G. Liu, X. Ma, Y. Yang, C. Wang, and J. Liu, “Federaser: Enabling efficient client-level data removal from federated learning models,” in IEEE/ACM International Symposium on Quality of Service (IWQoS), 2022, pp. 1–10.

[16] A. Halimi, S. Kadhe, A. Rawat, and N. Baracaldo, “Federated unlearning via knowledge distillation,” in NeurIPS Workshop on New Frontiers in Federated Learning, 2022.

[17] M. Kurmanji, P. Triantafillou, J. Hayes, and E. Triantafillou, “Towards unbounded machine unlearning,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 36, 2023, pp. 22 756–22 770.

[18] A. Foster and R. Pagh, “Fast machine unlearning without retraining through selective synaptic dampening,” in International Conference on Artificial Intelligence and Statistics (AISTATS). PMLR, 2024, pp. 3587– 3595.

[19] J. Frankle, D. J. Schwab, and A. S. Morcos, “The early phase of neural network training,” in International Conference on Learning Representations (ICLR), 2020.

[20] A. Achille, M. Rovere, and S. Soatto, “Critical learning periods in deep neural networks,” in International Conference on Learning Representations (ICLR), 2019.

[21] C. Fan, J. Liu, Y. Zhang, E. Wong, D. Wei, and S. Liu, “SalUn: Empowering machine unlearning via gradient-based weight saliency in both image classification and generation,” in International Conference on Learning Representations (ICLR), 2024.

[22] J. Bae, N. Ng, A. Lo, M. Ghassemi, and R. B. Grosse, “If influence functions are the answer, then what is the question?” in Advances in Neural Information Processing Systems, vol. 35, 2022, pp. 17 953– 17 967.

[23] C. Dwork, F. McSherry, K. Nissim, and A. Smith, “Calibrating noise to sensitivity in private data analysis,” in Theory of Cryptography Conference (TCC), ser. Lecture Notes in Computer Science, vol. 3876. Springer, 2006, pp. 265–284.

[24] D. Su, J. Cao, N. Li, E. Bertino, M. Lyu, and H. Jin, “Differentially private k-means clustering and a hybrid approach to private optimization,” ACM Transactions on Privacy and Security, vol. 20, no. 4, pp. 16:1–16:33, 2017.

[25] A. Thudi, G. Deza, V. Chandrasekaran, and N. Papernot, “Unrolling SGD: Understanding factors influencing machine unlearning,” in 2022 IEEE 7th European Symposium on Security and Privacy (EuroS&P). IEEE, 2022, pp. 303–319.

[26] V. S. Chundawat, A. K. Tarun, M. Mandal, and M. Kankanhalli, “Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 6, 2023, pp. 7210–7217.

## APPENDIX A

## PROOF OF THEOREM 1 (DISTILLATION APPROXIMATION BOUND)

The retraining component of IDSISA’s deletion guarantee follows directly from SISA’s checkpoint-rewind construction and requires none of the analysis below, which bounds only the distillation stage’s deviation from a from-scratch retrain.

Let $\begin{array} { r l r } { \mathcal { L } ( \theta ; D ) } & { { } = } & { \frac { 1 } { | D | } \sum _ { z _ { i } \in D } \ell ( f _ { \theta } ( x _ { i } ) , y _ { i } ) } \end{array}$ . After removing $D _ { f }$ from shard $D _ { m } ^ { \cdot } ,$ , let $D _ { m } ^ { \prime } = D _ { m } \ \backslash \ D _ { f }$ and $\theta _ { m } ^ { \prime } =$ arg min<sub>θ</sub> $\mathcal { L } ( \theta ; D _ { m } ^ { \prime } )$

Step 1 (parameter perturbation). This is the same argument as Lemma 1, generalized from a single removed sample $x _ { j }$ to the set $D _ { f } \colon$ Taylor-expanding $\mathcal { L } ( \theta _ { m } ^ { \prime } ; D _ { m } ^ { \prime } )$ around $\theta _ { m }$ and using first-order optimality of $\theta _ { m } ^ { \prime }$ together with Assumption 2 $( H \succeq \mu I )$ gives

$$
\| \theta _ { m } ^ { \prime } - \theta _ { m } \| \leq \frac { \| \nabla \mathcal { L } ( \theta _ { m } ; D _ { m } ^ { \prime } ) \| } { \mu } .\tag{3}
$$

Step 2 (gradient difference). $\begin{array} { r l } { \nabla \mathcal { L } ( \theta _ { m } ; D _ { m } ^ { \prime } ) } & { { } - } \end{array}$ $\begin{array} { r l r } { \nabla \mathcal { L } ( \bar { \theta _ { m } } ; D _ { m } ) } & { { } = } & { - \frac { | D _ { f } | } { | D _ { m } | } \bar { g } _ { D _ { f } } } \end{array}$ , where $\bar { g } _ { D _ { f } }$ is the mean gradient over $D _ { f }$ . Under Assumption $3 \ \mathrm { ~ ( \left\| ~ \right\} \nabla \ell | \ \leq \ G )$ $\begin{array} { r } { \| \nabla \mathcal { L } ( \theta _ { m } ; D _ { m } ^ { \prime } ) \| \le \frac { | D _ { f } | } { | D _ { m } | } G . } \end{array}$

Step 3 (distillation term). With $\mathcal { L } _ { \mathrm { t o t a l } } = ( 1 - \alpha ) \mathcal { L } _ { \mathrm { C E } } + \alpha \mathcal { L } _ { \mathrm { K I } }$ and $T ^ { 2 }$ -scaled KL loss, $\| \nabla \mathcal { L } _ { \mathrm { K L } } \| \le G / T$ , since teacher outputs come from shards that never saw $D _ { f }$ and the same Lipschitz constant governs both loss terms, attenuated by temperature. Combining,

$$
\| \nabla \mathcal { L } _ { \mathrm { t o t a l } } \| \leq ( 1 - \alpha ) G \cdot \frac { | D _ { f } | } { | D _ { m } | } + \alpha \cdot \frac { G } { T } .\tag{4}
$$

Steps 1–3 together give, per affected shard m with $| D _ { m } | \approx$ $N / M$ and a roughly even split of $D _ { f }$ across the $| { \cal A } |$ affected shards $( | D _ { f } \cap D _ { m } | ~ \approx ~ | D _ { f } | / | A | )$ , a per-shard output-perturbation bound (via L-smoothness, $\parallel f _ { \theta _ { \mathrm { m } } ^ { \prime } } \left( x \right) \textrm { -- }$ $\begin{array} { r l r } { f _ { \theta _ { m , \mathrm { r e t r a i n } } } ( x ) \| } & { { } \leq } & { L \| \theta _ { m } ^ { \prime } - \theta _ { m , \mathrm { r e t r a i n } } \| ) } \end{array}$ of order ${ \frac { \mathcal { L } } { \mu } } [ ( 1 ~ - $ $\alpha ) G \frac { M | D _ { f } | } { N | A | } { + \alpha \frac { G } { T } } ]$ . Summing this bound over the $\leq M$ affected shards by the triangle inequality — the only step Assumptions 1–3 alone license — gives a deviation bound linear in $| D _ { f } |$ not the $\sqrt { | D _ { f } | }$ form we report; we make this gap explicit rather than paper over it.

Assumption 4 (Independent Affected-Shard Perturbations, modeling assumption). The per-shard output perturbations $\{ f _ { \theta _ { m } ^ { \prime } } ( x ) - f _ { \theta _ { m , \mathrm { r e t r a i n } } } ( x ) \} _ { m \in \cal { A } }$ are approximately independent, mean-zero-correlated across affected shards. This is plausible $- \ D _ { f } \mathbf { \bar { s } }$ split across shards and each shard’s subsequent retraining randomness are not obviously correlated with one another — but it is a modeling assumption we adopt for tractability, not a consequence of Assumptions 1–3.

Step 4 (aggregation under Assumption 4). Under Assumption 4, the per-shard perturbations aggregate in quadrature rather than linearly: by Jensen’s inequality applied to variance additivity for uncorrelated terms, $\begin{array} { r } { \mathbb { E } \big \vert \big \vert \sum _ { m \in \mathcal { A } } \dot { \Delta { f _ { m } } } ( x ) \big \vert \big \vert \leq } \end{array}$ $\begin{array} { r } { \sqrt { \sum _ { m \in \mathcal { A } } \mathbb { E } \| \Delta f _ { m } ( x ) \| ^ { 2 } } = \sqrt { | \cal { A } | } } \end{array}$ · (per-shard bound). Substituting the per-shard bound from Steps 1–3, evaluating $| { \cal { A } } |$ at a representative value, bounding $( 1 - \alpha ) \leq 1$ and $\alpha \leq 1$ (loosening Step 3’s exact coefficients to their worst case), and absorbing the resulting M- and |A|-dependent constants into the displayed coefficients (rather than carrying them through symbolically) gives the reported form

$$
\lVert f _ { \theta ^ { \prime } } ( x ) - f _ { \theta _ { \mathrm { r e t r a i n } } } ( x ) \rVert \lesssim \frac { L G } { \mu } \cdot | D _ { f } | ^ { 1 / 2 } \cdot \left( 1 + \frac { 1 } { T } \right) .\tag{□ (5}
$$

We therefore present Theorem 1 as a heuristic estimate calibrated by this argument rather than a tight bound derived from Assumptions 1–3 alone, since the constant absorption at the last step is a simplification, not an exact identity.

## APPENDIX B PROOF OF COROLLARY 1

Instantiating with $L = 1 0 ^ { - 4 } , G = 1 , \mu = 0 . 7 0 7 , T = 2 . 0 \mathrm { { \Omega } }$ $| D _ { f } | = 5 , 0 0 0 , N = 5 0 , 0 0 0 \colon$

$$
\epsilon _ { \mathrm { c e r t } } \leq \frac { L G } { \mu } \sqrt { | D _ { f } | } \left( 1 + \frac { 1 } { T } \right) = \frac { 1 0 ^ { - 4 } } { 0 . 7 0 7 } \cdot 7 0 . 7 1 \cdot 1 . 5 \approx 0 . 0 1 5 .\tag{6}
$$

We report $L$ and $\mu$ here as an illustrative instantiation, not as independently measured constants: $L = 1 0 ^ { - 4 }$ is a rough orderof-magnitude estimate of how much the softmax layer compresses parameter-space perturbations for ResNet-18 near convergence, not a value from a dedicated Lipschitz-measurement procedure, and $\mu = 0 . 7 0 7$ is chosen for illustration rather than derived from the L-BFGS damping factor $\lambda = 0 . 0 1$ by any stated formula — we are not aware of one that connects the two. LSTM’s smaller forget set $( | D _ { f } | = 1 , 0 0 0 )$ gives a strictly tighter predicted bound under the same illustrative constants $( \approx 0 . 0 0 6 7 $ , since the bound scales with $\sqrt { | D _ { f } | }$ directly, not with the ratio $| D _ { f } | / N )$ . Measured mean values across CNN, LSTM, and ANN (0.011, 0.008, 0.013 respectively) all fall below this threshold. This bound is derived for, and should be compared only against, the mean quantity $\mathbb { E } _ { x \sim D _ { \mathrm { t e s t } } } \Vert \cdot \Vert ;$ the separately-reported 99th-percentile figures (0.019–0.027, Table XIV of the main paper) exceed it, consistent with Remark 1 below on the limits of Assumptions 1–3.

Remark 1 (Non-Convex Architectures). Assumptions 1–3 do not hold strictly for ResNet-18 or BiLSTM. Influence-based approximations remain empirically stable in overparameterized networks despite this [9], [22]: Bae et al. show the leave-one-out Newton-step approximation stays accurate even without global positive-definiteness, since effective curvature near a local minimum is typically well-conditioned. The meancase validation in Table XIV of the main paper supports this independent of the formal assumptions.

Remark 2 (Cross-Shard Distillation and Membership Inference). SISA’s isolation bounds membership-inference risk by ensuring no shard observes data outside its partition. Distillation introduces a cross-shard channel, but one that transmits output probability distributions (C values, $C - 1$ degrees offreedom) rather than gradients or weights, sharply limiting its capacity to re-encode $D _ { f } { - } s p e c i f i c$ information relative to a p-dimensional parameter update. The shadowmodel attack (Table XV, main paper) confirms this empirically: IDSISA-unlearned models remain safely at random chance (0.132) on $D _ { f } ,$ , statistically indistinguishable from SISA’s 0.134 despite SISA using no distillation term at all; distillation therefore adds no material membership-inference surface beyond SISA’s own baseline. The full-retrain model’s markedly higher confidence (0.681) reflects generalizable low-level visual features the deleted class shares with the retained classes (Appendix O), exploited confidently by a single, unsharded classifier; SISA and IDSISA instead aggregate predictions across M independently-partitioned shards, an ensembling effect known to suppress overconfidence on inputs individual members disagree on, and this — present in SISA without any distillation — is the more parsimonious explanation for the gap to Full Retrain than an effect attributed to distillation specifically. This concerns leakage within a single retraining event; Appendix C separately addresses causal dependence accumulated across a sequence of requests, a distinct question from statistical leakage.

## APPENDIX C

## PROOF OF THEOREM 2 (CAUSAL EXACTNESS UNDER CASCADING DELETION)

Let $A _ { 0 } ~ = ~ \{ m ~ : ~ D _ { f } \cap D _ { m } ~ \neq ~ \emptyset \}$ and ${ \cal C } ~ = ~ { \cal C } ( A _ { 0 } )$ (Definition 4), and suppose no shard is causally dependent (Definition 1) on any previously-deleted sample before this request.

Shards outside C. By Definition $^ { 4 , }$ no m $\notin C$ contains a sample in $D _ { f }$ (else $m \in A _ { 0 } \subseteq C )$ , and no m $\notin C$ was taught, directly or transitively, by any shard that does (else m would be reachable from $A _ { 0 } )$ . Combined with the assumption that no prior causal dependence exists, every shard outside C remains causally exact with respect to $D _ { f }$ without any action taken on it.

Shards inside C. Algorithm 3 (main paper) retrains every $m \in C$ from a checkpoint preceding $m { \mathrm { : } } s$ first contaminating event, on data excluding $D _ { f }$ (if $m \in A _ { 0 } )$ or unchanged data (otherwise), drawing teacher signal exclusively from $\mathcal { T } = V \backslash$ C. By the previous paragraph, every shard in $\tau$ is causally exact with respect to $D _ { f } ;$ by Definition 1, a computation using only such teacher signal and $D _ { f }$ -excluded data cannot itself become causally dependent on $D _ { f }$

Both cases establish the claim for all $m \in V .$

Extension to a request sequence. Let $P ( k )$ denote “after k requests processed via Algorithm 3, no shard is causally dependent on any sample deleted in requests $1 , \ldots , k . ^ { \prime \prime } P ( 0 )$

holds vacuously. The argument above, applied with G as accumulated through request $k - 1$ , shows $P ( k - 1 ) \Rightarrow P ( k )$ by induction, $P ( k )$ holds for all k, and the guarantee extends to the full aggregated model under any finite deletion sequence.

Two properties of the algorithm make this argument independent of graph structure. First, repair order within $C$ is immaterial: every shard being retrained draws teacher signal only from $\tau ,$ never from another member of $C ,$ so cycles within $C$ never arise as a dependency during repair. Second, no bound on |C| is assumed or required — a shard may teach another before it is itself retrained, and a subsequentlyretrained shard may extend a dependency chain further, so $C$ can in principle include any subset of $V ;$ the proof holds regardless.

Remark 3 (Checkpointing Requirement). Locating each shard’s earliest pre-contamination checkpoint requires saving a checkpoint immediately before every distillation-retraining event, not only before each original slice, and requires that checkpoint to capture full model state (weights, optimizer state, batch-normalization statistics), not weights alone.

Remark 4 (Checkpoint Storage Is Bounded). This does not imply unbounded storage growth. A shard’s pre-retraining checkpoint can be pruned the moment its corresponding edges in G are removed (Algorithm 3, step 5), since any shard whose teaching relationship is superseded is, by construction, repaired in the same pass; all edges added during one retraining event share a single checkpoint. Total storage is therefore $O ( M K _ { s } ) + O ( M )$ at any point in the deployment — the same asymptotic order as the original per-slice checkpointing scheme — independent of the number of requests processed.

Remark 5 (Served Predictions). Theorem 2 concerns the causal dependence of model parameters, not predictions already served from a state that existed before a repair completed. Under a serialized deployment policy — each request fully processed before the resulting ensemble serves further predictions — no prediction is ever served from a state with residual dependence on an already-requested deletion. This assumption is inherited from SISA’s own single-deletion guarantee, not introduced by the cascading mechanism.

Remark 6 (Teacher-Pool Depletion). An optional teacherfreshness policy excludes any shard from future teaching once it has itself been retrained. This does not bound dependencychain length — a shard may teach before it is itself retrained — but it does monotonically shrink the eligible teacher pool. Modeling each request as affecting a shards drawn from M uniformly at random, the batch coupon-collector approximation gives ${ \mathbb E } [ R _ { e x h a u s t } ] \approx ( M / a ) H _ { M } \colon f o r ~ M = 1 0 , ~ a \approx 2 \ – 3$ (class-structured deletions) gives ≈ 10–15 requests; $a \approx 8$ (uniform deletions) gives ≈ 4–5. This estimate ignores that cascade repairs also consume teacher eligibility, which can only accelerate depletion, and is illustrative rather than a bound. A periodic full-shard refresh every K requests resets the pool at the cost of one amortized full-SISA-equivalent retraining per K requests. This creates a direct tension with cost, not just with freshness: Section VII-B (main paper) computes that on CIFAR-10, amortizing the 59.9-minute refresh over $K = 5$ requests adds ≈12.0 min/request, exceeding the 5.5 min/request that distillation saves relative to SISA, so a $K = 5$ cadence — conservative relative to the ≈10–15- request depletion estimate above — eliminates IDSISA’s cost advantage entirely rather than merely capping it. Preserving the cost advantage requires $K \gtrsim 1 1$ , i.e. resetting no more conservatively than roughly the depletion estimate itself, not well below it. When the pool is exhausted (or an unrefreshed policy is used), retraining proceeds without distillation on the Phase-1 partition, exactly the “Influence Only” configuration of Table XIII (main paper), which already dominates baseline SISA.

## APPENDIX D SENSITIVITY-CONCENTRATION TRADEOFF

Concentrating high-influence samples into fewer shards increases each affected shard’s sensitivity to a single sample’s removal: with $\sigma _ { m } = \lVert \Delta \theta _ { m } \rVert$ , Lemma 1 gives $\sigma _ { m } \leq G / \mu ,$ , and concentrated shards sit closer to this bound than randomlypopulated ones. The certification-relevant quantity, however, is the global perturbation $\Sigma _ { \mathrm { g l o b a l } } = \left| \mathcal { A } \right| \times \operatorname* { m a x } _ { m } \sigma _ { m }$ . IDSISA reduces $| { \cal { A } } |$ by 3–5× (from 8–10 shards under random SISA to 2–4); empirically, per-shard $\sigma _ { m }$ for concentrated shards runs 1.4–1.8× that of random-SISA shards (estimated from ϵ<sub>cert</sub> measurements across 10 runs), giving a net $\Sigma _ { \mathrm { g l o b a l } } ^ { \mathrm { I D S I S A } } \approx$ $0 . 4 4 \Sigma _ { \mathrm { g l o b a l } } ^ { \mathrm { S I S A } } \colon \mathrm { ~ a ~ } 2 . 3 \times$ reduction in global perturbation despite higher local sensitivity.

IDSISA’s certification mechanism is deterministic retraining, not noise addition, so this tradeoff does not affect a differential-privacy noise budget as currently constructed. For a hybrid DP-SISA extension adding calibrated noise to the retraining step, the required noise scale would need recalibration against sensitivity; if IDSISA concentrates samples by factor $c \approx 3 \ – 5$ while reducing affected shards by factor $r ~ \approx ~ 4$ , the effective noise-budget change $c / r ~ \approx ~ 0 . 6 \mathrm { - } 1 . 3$ is roughly budget-neutral to mildly favorable, though this analysis is specific to a noise-based extension we have not implemented.

## APPENDIX E

## LITERATURE SUPPORT AND BOUNDARY CONDITIONS FOR THE EARLY-STOPPED PROXY

Section III-B uses a 3-epoch, early-stopped baseline model, rather than a converged one, to extract penultimate-layer features and gradient-based influence rankings for Phase 1. We support this choice with prior findings rather than a dedicated correlation study on our own checkpoints, and state explicitly what those findings do and do not establish for our setting.

Frankle et al. show that many properties governing a network’s eventual behavior — which sparse sub-networks are trainable, the subspace gradient descent moves within, aspects of the weight distribution’s dependence on the data — are set within the earliest iterations of training, well before convergence [19]. Achille et al. show that deep networks pass through an early critical period during which the features the network will ultimately rely on are largely determined; after this period, further training mainly sharpens existing features rather than discovering new ones [20]. Together these support the premise that a 3-epoch checkpoint already carries the macroscopic feature structure K-means needs, rather than requiring convergence. Separately, checkpoint-based influence estimation from non-final training states already has direct precedent in the unlearning-adjacent literature: TracIn deliberately traces influence through a small number of intermediate checkpoints along the training trajectory rather than the fully converged model, and is effective precisely because gradient based influence signal does not require convergence to be informative [10].

These findings are evidence for using an early-stopped proxy in general, not a measurement on our specific checkpoints, and we name the conditions under which the argument would weaken: (i) tasks or architectures with a longer or less pronounced critical period than the vision/language benchmarks these findings were established on, where 3 epochs might fall before the relevant period closes; (ii) severe class imbalance, where minority-class features may separate later than the majority-class structure that dominates an early checkpoint; and (iii) optimizer or learning-rate schedules that delay the onset of the early phase (e.g., a long warmup). None of these conditions is known to hold for CIFAR-10/ResNet-18, IMDB/BiLSTM, or Fashion-MNIST/ANN, but we have not run a dedicated Spearman-correlation or clustering-agreement study across checkpoint depth on our own models to rule them out, and we do not claim to have done so.

## APPENDIX F PARTITION STALENESS

Influence scores and cluster centroids are computed once on D and held fixed across all subsequent deletion requests. Removing $D _ { f }$ changes the Hessian used to approximate influence, so the partition that would be optimal for $D _ { r }$ differs, in principle, from the one stored. This is an organizational efficiency question — how well the fixed shard assignment continues to concentrate future deletions — distinct from the causal-dependence question Appendix C addresses; partition staleness affects forgetting concentration and speed, not deletion correctness.

The influence-score Hessian after removing $D _ { f }$ is $H _ { D _ { r } } =$ $\begin{array} { r } { H _ { D } - \frac { 1 } { N } \sum _ { z _ { i } \in D _ { f } } \nabla ^ { 2 } \mathcal { L } ( \theta ; z _ { i } ) } \end{array}$ , and under Assumptions 1–3 the resulting change in any retained sample’s score is $O ( | D _ { f } | / N )$ — an order-of-magnitude statement, not a tight bound with a derived proportionality constant: for CIFAR-10’s cat-class deletion $( | D _ { f } | = 5 , 0 0 0 , N = 5 0 , 0 0 0 ) , | D _ { f } | / N = 1 0 \%$ , and for the uniform 500-sample forget set, $| D _ { f } | / N = 1 \%$ ; we report these ratios directly rather than as a separately-derived bound. Recomputing influence scores and re-clustering after the cat-class deletion reassigns 6.2% of retained samples to a different shard than the original partition, confirming the partition remains largely stable at this scale.

A differentially private clustering scheme — computing influence-weighted K-means centroids under an $( \varepsilon _ { \mathrm { d p } } , \delta _ { \mathrm { d p } } ) -$ DP mechanism such as DP-Lloyd’s algorithm [23], [24] — would bound the partition’s sensitivity to any single sample’s presence, closing this gap formally rather than empirically. We estimate $\varepsilon _ { \mathrm { d p } } \approx 1 . 0 { - 2 . 0 }$ would maintain silhouette scores above 0.4 with under 5% forgetting-quality degradation, based on the sensitivity analysis above; this remains a proposed remedy, not one implemented in the present system.

## A. Algorithmic DP-Lloyd Extension

For completeness, the metadata-private extension we would use in Phase 1 is the following DP-Lloyd procedure. Let $u _ { i } = [ \phi _ { i } ; \rho \tilde { \mathcal { T } } ( z _ { i } ) ]$ denote the PCA feature vector augmented with a clipped influence coordinate, where $\| u _ { i } \| _ { 2 } \leq R$ and the influence weight $w _ { i } \in [ 0 , w _ { \mathrm { m a x } } ]$ . A strict end-to-end metadataprivacy guarantee requires $\phi _ { i }$ and $\tilde { \mathcal { T } } ( z _ { i } )$ to be produced by a fixed public extractor or by a DP-trained proxy; applying DP only to the Lloyd update does not privatize a non-private fulldataset feature extractor.

Algorithm B.1: DP-Lloyd Phase-1 clustering.

1) Inputs: bounded vectors $\{ u _ { i } \} _ { i = 1 } ^ { N }$ , shard count $K = M$ Lloyd iterations $Q ,$ , radius R, privacy budget $\left( \varepsilon _ { \mathrm { d p } } , \delta _ { \mathrm { d p } } \right)$

2) Budget split: set $\varepsilon _ { n } = \varepsilon _ { \mathrm { d p } } / ( 2 Q ) , \varepsilon _ { s } = \varepsilon _ { \mathrm { d p } } / ( 2 Q )$ , and $\delta _ { s } = \delta _ { \mathrm { d p } } / Q$

3) Initialization: initialize centroids $\{ c _ { k } ^ { ( 0 ) } \} _ { k = 1 } ^ { K }$ using public randomness or another DP seeding mechanism.

4) Private Lloyd step: for $t = 1 , \ldots , Q ,$ , assign each $u _ { i }$ to its nearest current centroid. For each cluster k, compute the weighted sufficient statistics

$$
n _ { k } ^ { ( t ) } = \sum _ { i } \mathbf { 1 } [ a _ { i } ^ { ( t ) } = k ] , ~ s _ { k } ^ { ( t ) } = \sum _ { i } \mathbf { 1 } [ a _ { i } ^ { ( t ) } = k ] w _ { i } u _ { i } .\tag{7}
$$

Release noisy statistics

$$
\tilde { n } _ { k } ^ { ( t ) } = n _ { k } ^ { ( t ) } + \mathrm { L a p } ( 1 / \varepsilon _ { n } ) ,\tag{8}
$$

$$
\tilde { s } _ { k } ^ { ( t ) } = s _ { k } ^ { ( t ) } + \mathcal { N } ( 0 , \sigma _ { s } ^ { 2 } I ) ,\tag{9}
$$

where $\begin{array} { r c l } { \sigma _ { s } } & { = } & { 2 R w _ { \mathrm { m a x } } \sqrt { 2 \log ( 1 . 2 5 / \delta _ { s } ) } / \varepsilon _ { s } } \end{array}$ is calibrated to the conservative replacement sensitivity of the weighted sum.

5) Centroid update: set

$$
c _ { k } ^ { ( t ) } = \Pi _ { \| c \| _ { 2 } \leq R } \left( \frac { \tilde { s } _ { k } ^ { ( t ) } } { \operatorname* { m a x } \{ 1 , \tilde { n } _ { k } ^ { ( t ) } \} } \right) ,\tag{10}
$$

where Π projects back to the radius-R ball.

6) Shard assignment: after Q iterations, assign samples to the nearest private centroid and apply the same shardsize balancing rule using private counts and deterministic tie-breaking by private distance margins. No nonprivate influence ranking is used in this balancing step.

Because each record contributes to one cluster’s count and vector sum per iteration, parallel composition applies within an iteration and sequential composition over Q iterations yields $( \varepsilon _ { \mathrm { d p } } , \delta _ { \mathrm { d p } } ) { \cdot } \mathrm { D P }$ for the released centroids and any downstream shard metadata computed from them by post-processing. This algorithm is therefore one way to convert IDSISA’s partition metadata from a non-private organizational heuristic into a DP-bounded preprocessing stage. The present experiments, however, use ordinary K-means and the non-private earlystopped proxy, so the DP budget/utility trade-off is identified here as an extension rather than reported as an empirical result.

Remark 7 (Two Senses of Information Persistence). Causal exactness (Definition 1, main paper) is a claim about model parameters $\theta _ { m } .$ after Algorithm 2 or 3 processes a request, no parameter was computed using $D _ { f }$ . It is not a claim about the shard-assignment metadata itself. Because Phase 1’s influence scores and centroids are computed once on D before any deletion, the partition structure retains a statistical imprint of every sample ever included in D, deleted or not, in the same sense that any preprocessing decision made on a full dataset — architecture search, hyperparameter tuning, feature normalization statistics — retains such an imprint; this is a convention shared with SISA and with certified-unlearning methods generally, not a property specific to IDSISA. The $O ( | D _ { f } | / N )$ bound above characterizes how much this imprint drifts the partition’s organizational quality for future requests, which is an efficiency question, not a formal bound on how much information about any single deleted sample could in principle be recovered from the partition structure itself. That second question — whether partition metadata constitutes a residual privacy channel distinct from parameterlevel dependence — would be addressed by the DP-Lloyd extension above. Under the stated bounded-input and fixedor-DP-proxy conditions, the extension bounds the influence of any single sample on the centroids and downstream shard metadata with an (ϵ, δ)-Differential Privacy guarantee; the present implementation does not claim that stronger metadata guarantee.

## APPENDIX G MOTIVATION

Theorem 2 establishes causal exactness under cascading deletion for any request sequence and any graph structure. It does not characterize cost: how large the reachability closure $C ( A _ { 0 } )$ tends to grow under a realistic sequence of requests, and how quickly the teacher-freshness policy (Remark 6), if adopted, exhausts the eligible teacher pool. This section reports that characterization empirically, complementing Corollary 2’s analytical best case $( | C | = | A _ { 0 } | )$ and worst case $( | C | = M )$ .

## APPENDIX H PROTOCOL

We use the CIFAR-10/ResNet-18 pipeline (M = 10 shards, influence-aware partitioning as in Section V of the main paper) and run two sequential-deletion traces of 10 and 8 requests respectively. The primary trace alternates class-structured (odd requests) and uniformly-distributed (even requests) forget sets, matching the request pattern underlying the main paper’s own single-deletion experiments. The supplementary trace runs 8 consecutive class-structured requests with no interleaved uniform requests and no resets, constructed specifically to stresstest dependency accumulation under conditions the primary trace does not exercise. After each request we log $A _ { 0 } , C ,$ whether any shard in C used the scratch-retrain fallback, and the cumulative edge count in G.

## APPENDIX I RESULTS

Primary trace. $| C | = | A _ { 0 } |$ on all 10 requests; no cascade forms. Each uniformly-distributed request touches nearly all 10 shards, resetting the dependency graph before a chain can accumulate across the interleaved class-structured requests. Consequently, the single-deletion experiments in Sections V– VI exercise Algorithm 3 only in its trivial case, and no reported result depends on cascading behavior.

Supplementary trace. Table XVII reports the full trace under the adversarial, no-reset regime. Cascades $( | C | > | A _ { 0 } | )$ occurred on 5 of 8 requests; 2 of 8 collapsed to the full worst case $( | C | = M = 1 0 )$ .

TABLE XVII  
SUPPLEMENTARY TRACE: 8 SEQUENTIAL CLASS-STRUCTURED REQUESTS, NO RESETS (CIFAR-10/RESNET-18, M = 10). CASCADES ON REQUESTS 2, 3, 5, 6, 8; FULL COLLAPSE TO |C| = M ON REQUESTS 3 AND 6.
<table><tr><td> $\mathbf { R e q . }$ </td><td> $\left| A _ { 0 } \right|$ </td><td> $| C |$ </td><td>Cascade?</td><td>Scratch fallback?</td><td>Cum. edges</td></tr><tr><td>1</td><td>6</td><td>6</td><td>No</td><td>No</td><td>0</td></tr><tr><td>2</td><td>5</td><td>7</td><td>Yes</td><td>No</td><td>2</td></tr><tr><td>3</td><td>7</td><td>10</td><td>Yes</td><td>Yes</td><td>0</td></tr><tr><td>4</td><td>8</td><td>8</td><td>No</td><td>No</td><td>0</td></tr><tr><td>5</td><td>5</td><td>9</td><td>Yes</td><td>No</td><td>4</td></tr><tr><td>6</td><td>7</td><td>10</td><td>Yes</td><td>Yes</td><td>0</td></tr><tr><td>7</td><td>4</td><td>4</td><td>No</td><td>No</td><td>0</td></tr><tr><td>8</td><td>6</td><td>7</td><td>Yes</td><td>No</td><td>1</td></tr></table>

Membership-inference probe. On the most diagnostic cascade instance (request 2: teacher shard owning 1,144 of 1,200 deleted-class samples, 95.3%; pulled shard owning 3 of 1,200, 0.25%), mean shadow-model confidence on the deleted class dropped from 0.0137 before the cascading retrain to 0.0002 after, a ≈68.5× relative reduction. This instance was selected as the most diagnostic among the cascades observed, illustrating the mechanism’s effect rather than estimating a typical case. Both absolute values are far below the 10- class chance level (0.10), reflecting this experiment’s reduced training budget (3 retraining epochs per shard) rather than the fully-trained calibration underlying Table XV of the main paper; the relative reduction is the informative quantity.

Teacher-pool depletion. Under the freshness policy applied to the supplementary trace, the eligible teacher pool did not reach full exhaustion within 8 requests, consistent with the illustrative estimate in Remark 6 (≈10–15 requests for classstructured deletions at $M = 1 0 )$

## APPENDIX J IMPLEMENTATION NOTES

(i) Cascade attribution is one-hop. The per-request diagnostic identifying which teacher pulled which shard into C traces edges originating in $A _ { 0 }$ directly; it does not itself perform multi-hop tracing. The retraining scope C is always computed via full closure traversal and is unaffected, but a longer trace with deeper chains could in principle produce a two-hop pull this diagnostic would not individually attribute. In the trace above, one-hop attribution fully accounts for $\vert C \vert - \vert A _ { 0 } \vert$ in every request.

(ii) Uniform-regime sampling is pooled, not classstratified. Uniformly-distributed forget sets are drawn from the full remaining sample pool rather than a strict 50-per-class quota. Given pool sizes in the thousands, this reliably touches all 10 shards in practice, confirmed in every uniform-regime request of the primary trace.

(iii) Shard models are warm-started from the baseline. Each shard’s model initializes from the full-dataset baseline used for influence scoring, rather than from scratch, to keep this cost-characterization experiment tractable. This speeds convergence but means shard models share initialization information beyond their own partition; it does not affect the graph, closure, or teacher-restriction logic being measured.

(iv) Checkpoint rewind targets the original post-training state. Each retraining event rewinds to a shard’s initial checkpoint rather than its most recent prior state, a more conservative interpretation of Remark 3 than strictly necessary; the retained-data mask still correctly excludes every sample deleted so far. Wall-clock costs reported here are therefore biased pessimistic relative to an incremental-checkpoint deployment.

## APPENDIX K TEMPORAL DEPENDENCY AND NON-LOCAL HESSIAN STRUCTURE

RNN hidden states satisfy $h _ { t } = \operatorname { t a n h } ( W _ { h } h _ { t - 1 } + W _ { x } x _ { t } + b )$ so $\begin{array} { r } { \partial \mathcal { L } / \partial W _ { h } = \sum _ { t } ( \partial \mathcal { L } / \partial h _ { t } ) h _ { t - 1 } ^ { T } } \end{array}$ sums contributions across all timesteps. The per-shard Hessian approximation $H _ { D _ { m } } \approx$ $\begin{array} { r } { \frac { 1 } { \left| D _ { m } \right| } \sum _ { ( x _ { i } , y _ { i } ) \in D _ { m } } { \nabla ^ { 2 } } \mathcal { L } ( \theta ; x _ { i } , y _ { i } ) } \end{array}$ needs within-shard locality to be tractable, but an RNN’s Hessian couples to weights in ways that depend on every sequence’s temporal structure, not just the sequences in a given shard. Unlearning 1,000 IMDB reviews under this architecture collapsed overall accuracy to 52.3% (near-random), affecting target and retained classes equally — direct evidence the unlearning perturbation cannot be localized to the forgotten sequences.

## APPENDIX L

## SHARED WEIGHTS AND ISOLATION VIOLATION

Vanilla RNNs share weight matrices $( W _ { h } , W _ { x } , W _ { y } )$ across all timesteps of every sequence in a shard; retraining on $D _ { m } \backslash D _ { f }$ therefore perturbs weights that affect every sequence in $D _ { m } .$ , not only those resembling $D _ { f }$ , which functionally violates SISA’s isolation property even though the shard’s training data itself is correctly isolated. LSTM gating,

$$
f _ { t } = \sigma ( W _ { f } [ h _ { t - 1 } , x _ { t } ] + b _ { f } ) ,\tag{11}
$$

$$
i _ { t } = \sigma ( W _ { i } [ h _ { t - 1 } , x _ { t } ] + b _ { i } ) ,\tag{12}
$$

$$
o _ { t } = \sigma ( W _ { o } [ h _ { t - 1 } , x _ { t } ] + b _ { o } ) ,\tag{13}
$$

creates path-specific information flow so a sequence’s representation is predominantly determined by its own content, which is why BiLSTM influence scores have 0.24× the variance of the CIFAR-10 CNN baseline.

## APPENDIX M

## GRADIENT INSTABILITY AND THE ROOT ASSUMPTION VIOLATION

Sequences over 20–30 steps suffer vanishing/exploding gradients, requiring clipping $g \  \ g \cdot \operatorname* { m i n } ( 1 , \gamma / \| g \| )$ that directly violates Assumption 3: the clipped gradient is a biased estimate of the true gradient, so the influence approximation $\begin{array} { r } { \mathcal { T } ( z _ { j } ) \approx - \nabla \mathcal { L } ( z _ { \mathrm { v a l } } ) ^ { T } H ^ { - 1 } \nabla \mathcal { L } ( \theta ; z _ { j } ) } \end{array}$ uses a corrupted input, with clipped samples receiving inflated, uniformly-normed influence regardless of true magnitude. Empirically: clipping fired 5,355 times during initial training (≈1.1% of update steps) and 8,318 times during unlearning retraining (a 55% increase, reflecting destabilization from checkpoint rewind on reduced data); influence computation took 342s versus BiL-STM’s 1.34s (256× slower) with 15× higher score variance, confirming the L-BFGS approximation does not converge reliably under corrupted gradients. GRU and LSTM gating avoids this via multiplicative gates and the cell-state highway, which bound effective gradient norms without clipping (BiLSTM requires clipping on under 0.1% of steps).

Applicability is therefore governed by Hessian stability, reliably bounded gradients and low-variance influence scores, not by architecture family per se; GRU and LSTM satisfy this via gating, vanilla RNN does not.

## APPENDIX N

## EXTENDED COMPARISON TABLES

Table XVIII extends Section II’s qualitative positioning across the broader unlearning literature. Table XIX extends Table V (main paper) with additional published methods on the same CIFAR-10/ResNet-18 class-level forgetting benchmark.

## APPENDIX O

## MEMBERSHIP INFERENCE: FULL RESULTS

Table XX extends Table XV (main paper) with attackcalibration detail. The shadow model correctly identifies general training members above chance (TPR ≈0.60/0.59 vs. random 0.50, attack validation accuracy 0.62–0.63), confirming calibration, then finds near-identical, near-random confidence for both SISA- and IDSISA-unlearned models on $D _ { f }$

TABLE XX  
FULL MEMBERSHIP-INFERENCE RESULTS (CIFAR-10/RESNET-18, SHADOW MODEL + THRESHOLD ATTACK).
<table><tr><td>Model</td><td> $D _ { f }$  Conf.</td><td> $D _ { f }$  TPR</td><td> $D _ { \mathbf { t r a i n } }$  TPR</td><td>Threshold</td><td>Val Acc</td></tr><tr><td>SISA (Unlearned)</td><td>0.134</td><td>≈0.42</td><td>≈0.60</td><td>≈0.139</td><td>0.62</td></tr><tr><td>IDSISÀ (Unlearned)</td><td>0.132</td><td>≈0.39</td><td>≈0.59</td><td>≈0.139</td><td>0.62</td></tr><tr><td>Full Retrain  $( D _ { r } )$ </td><td>0.681</td><td>0.83</td><td>N/A</td><td>0.738</td><td>N/A</td></tr></table>

The reported metric is ∆conf = 0.002 (not δ = 0.0, which would imply implausibly exact zero leakage from a finitesample attack): this is below the shadow model’s own runto-run measurement variance (±0.003–0.005 across seeds), corresponds $\mathrm { ~ t o ~ a ~ } D _ { f }$ TPR difference of ≈ 0.03 with both methods near chance, and is far smaller than the gap to Full Retrain’s 0.681 confidence, the combination that indicates genuine unlearning rather than attack miscalibration. Full Retrain’s elevated $D _ { f }$ confidence reflects generalizable low-level visual features the cat class shares with the other 9 CIFAR-10 classes, not a calibration artifact, since that model never saw cat-class samples during training. SISA and IDSISA’s shared low confidence (0.134, 0.132) is consistent with this: both are M-shard ensembles, and ensembling suppresses the same overconfidence effect that a single unsharded model like Full Retrain exhibits, independent of whether distillation is used — which is why SISA alone already shows it. The calibrated decision threshold (0.139) is the midpoint between $D _ { f }$ mean confidence (0.133) and D mean confidence (≈0.146), selected to maximize attack validation accuracy on held-out shadow data.

This attack targets leakage from a single distillation event (Remark 2); it does not probe the cross-request causal dependence Appendix C addresses, since membership-inference risk and parameter-level causal dependence are governed by different mechanisms in this system.

## APPENDIX P

## STATISTICAL SIGNIFICANCE

All point estimates in the main paper (Tables IV–XV, excluding Tables I–III which contain no per-seed estimates) are means over 5 seeds (42–46), covering data shuffling, shard initialization, and mini-batch ordering.

TABLE XVIII  
CROSS-DATASET QUALITATIVE COMPARISON. “MULTI-ARCH” = VALIDATED ACROSS CNN, RNN, AND ANN. “MI SAFE” = MI SUSCEPTIBILITY CALIBRATED TO FULL-RETRAIN LEVEL. “AMORTIZED” = EFFICIENCY IMPROVES WITH REPEATED REQUESTS. ✓/×/∼ = SATISFIED / NOT / PARTIALLY.
<table><tr><td>Method</td><td>Certified Deletion</td><td>Selective Forgetting</td><td>Utility Pres.</td><td>Multi- Arch</td><td>NLP Support</td><td>MI Safe</td><td>Amor- tized</td><td>Influence Guided</td><td>Reference</td></tr><tr><td>Full Retraining</td><td>√</td><td>Strong</td><td>Strong</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td><td>Oracle</td></tr><tr><td>Gradient Ascent [25]</td><td>X</td><td>Moderate</td><td>Poor</td><td>~</td><td>~</td><td>X</td><td>X</td><td>X</td><td>2022</td></tr><tr><td>Fine-Tuning [6]</td><td>X</td><td>Weak</td><td>Good</td><td>~</td><td>~</td><td>X</td><td>X</td><td>X</td><td>2020</td></tr><tr><td>Fisher Forgetting [6]</td><td>~ (DP)</td><td>Moderate</td><td>Poor</td><td>X</td><td>×</td><td>X</td><td>×</td><td>~</td><td>2020</td></tr><tr><td>NegGrad+ [17]</td><td>X</td><td>Moderate</td><td>Moderate</td><td>~</td><td>~</td><td>~</td><td>×</td><td>X</td><td>2023</td></tr><tr><td>SCRUB [17]</td><td>X</td><td>Strong</td><td>Moderate</td><td>~</td><td>×</td><td>~</td><td>X</td><td>X</td><td>2023</td></tr><tr><td>SSD [18]</td><td>X</td><td>Moderate</td><td>Good</td><td>~</td><td>×</td><td>√</td><td>X</td><td>2</td><td>2024</td></tr><tr><td>SalUn [21]</td><td>X</td><td>Strong</td><td>Good</td><td>~</td><td>X</td><td>~</td><td>×</td><td>~</td><td>2024</td></tr><tr><td>Bad Teacher [26]</td><td>X</td><td>Strong</td><td>Moderate</td><td>×</td><td>X</td><td>×</td><td>×</td><td>X</td><td>2023</td></tr><tr><td>PURGE [13]</td><td>~(distill.)</td><td>Moderate</td><td>Good</td><td>×</td><td>×</td><td>~</td><td>X</td><td>×</td><td>2025</td></tr><tr><td>SISA [1]</td><td>√</td><td>Weak</td><td>Poor</td><td>~</td><td>~</td><td>√</td><td>√</td><td>X</td><td>2021</td></tr><tr><td>Fed. Unlearning [15], [16]</td><td>√</td><td>Moderate</td><td>Moderate</td><td>~</td><td>2</td><td>2</td><td>~</td><td>×</td><td>2022</td></tr><tr><td>IDSISA</td><td>√</td><td>Strong</td><td>Strong</td><td>√</td><td>√</td><td>√</td><td>2</td><td></td><td>This work</td></tr></table>

TABLE XIX  
QUANTITATIVE COMPARISON, CIFAR-10/RESNET-18, CLASS-LEVEL FORGETTING (∼5,000 SAMPLES). <sup>†</sup>NO FORMAL DELETION GUARANTEE. <sup>‡</sup>RETRAINING CERTIFIED; DISTILLATION GAP ϵ<sub>CERT</sub> ≤ 0.015 (TABLE I, MAIN PAPER). <sup>∗</sup>REPRESENTATIVE FIGURES FROM PUBLISHED BENCHMARKS [17], [21], [18], [6]; EXACT FIGURES VARY BY IMPLEMENTATION.
<table><tr><td>Method</td><td>Type</td><td>Certified?</td><td>Forget Acc (%)↓</td><td>Retain Acc (%)↑</td><td>Test Acc (%)↑</td><td>MIA Conf. on  $D _ { f } \downarrow$ </td></tr><tr><td>Full Retraining</td><td>Exact</td><td>Yes</td><td>~10.0</td><td>~93.4</td><td>~93.4</td><td>~0.68</td></tr><tr><td>Gradient Ascent [25] Fine-Tuning [6]</td><td>Weight mod.</td><td> $\mathrm { N o } ^ { \dag }$ </td><td> ${ \sim } 0 { - } 6 ^ { \ast }$ </td><td> ${ \sim } 7 4  – 8 0 ^ { * }$ </td><td>~63-75*</td><td>Low</td></tr><tr><td rowspan="5">NegGrad+ [17] Fisher Forgetting [6] SSD [18]</td><td>Weight mod.</td><td> $\mathrm { N o } ^ { \dag }$ </td><td> ${ \sim } 6 0 { - } 8 0 ^ { \ast }$ </td><td> ${ \sim } 8 3 { - } 8 5 ^ { \ast }$ </td><td> ${ \sim } 8 2 { - } 8 4 ^ { * }$ </td><td>High</td></tr><tr><td>Gradient</td><td> $\mathrm { N o } ^ { \dag }$ </td><td> ${ \sim } 5 { - } 1 0 ^ { * }$ </td><td>~82-84*</td><td>~82-84*</td><td>Moderate</td></tr><tr><td>DP noise</td><td>Approx. DP</td><td> ${ \sim } 3 { - } 8 ^ { * }$ </td><td>~78-82*</td><td>~78-82*</td><td>Low</td></tr><tr><td>FIM dampening</td><td>No†</td><td> ${ \sim } 2 { - } 8 ^ { * }$ </td><td>~82-86*</td><td> ${ \sim } 8 2 { - } 8 6 ^ { \ast }$ </td><td>Calibrated</td></tr><tr><td>Distillation</td><td> $\mathrm { N o } ^ { \dag }$ </td><td> ${ \sim } 5 { - } 8 ^ { * }$ </td><td>~83-85*</td><td>~83-85*</td><td>Moderate</td></tr><tr><td>Bad Teacher [26] SalUn [21]</td><td>Distillation Saliency mask</td><td> $\mathrm { N o } ^ { \dag }$   $\mathrm { N o } ^ { \dag }$ </td><td> ${ \sim } 0 { - } 5 ^ { \ast }$   ${ \sim } 0 { - } 6 ^ { \ast }$ </td><td> ${ \sim } 8 0 { - } 8 4 ^ { * }$   ${ \sim } 8 8 { - } 9 2 ^ { \ast }$ </td><td>~80-84*  ${ \sim } 8 8 { - } 9 2 ^ { \ast }$ </td><td>Low</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Low</td></tr><tr><td>SISA [1] IDSISA</td><td>Exact retraining Exact + Distill.</td><td>Yes  $\mathbf { Y e s } ^ { \ddag }$ </td><td>18.6 12.3</td><td>83.2 84.7</td><td>82.3 83.1</td><td>0.134 0.132</td></tr></table>