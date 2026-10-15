# Learning Graph Energies for Finite-Time Generation

Amartya Banerjee Dept. of Computer Science UNC Chapel Hill

Xingyu Xu Carnegie Mellon University

Caroline Moosmüller Dept. of Mathematics UNC Chapel Hill

Harlin Lee   
SDSS   
UNC Chapel Hill

## Abstract

Energy-based models can represent distributions over discrete structures such as molecular graphs, but an energy that is accurate at equilibrium need not yield accurate samples when the sampler stops after a fixed number of steps. We introduce finite-horizon energy learning, which adapts the acceptance energy of a Metropolis–Hastings sampler to the distribution it returns after a prescribed number of steps, with initialization, proposals, and annealing schedule held fixed. For graph gen eration, a frozen pretrained model supplies initialization and proposals, and a learned energy governs acceptance. A training-only discriminator scores completed samples, and coupled continuations of the accepted and rejected branches of a proposal give an acceptance gradient estimator that is unbiased for a fixed discriminator under stated conditions, without equilibrated negative samples or diferentiation through discrete states. Generation uses the learned energy in the original single-trajectory sampler, without discriminator evaluations or reranking. On twelve pairwise binary targets with exactly evaluated terminal laws, learned energies have lower median terminal total variation than the equilibrium-correct energy in every tested short-horizon setting, and than its numerically tuned scalar multiple in most settings. On MOSES, a learned two-coeficient combination of pretrained Graph Energy Matching (GEM) potentials lowers FCD by 3.7–10.5% relative to scale-calibrated GEM at matched sampling steps, with validity gains at the shortest budget and a sub-point validity tradeof at higher budgets.

## 1 INTRODUCTION

For a generator stopped after K transitions, sampling accuracy is determined by its terminal distribution, not by its equilibrium law. The terminal distribution depends on initialization, proposals, acceptance decisions, and the annealing schedule. Consequently, even an energy that represents the target exactly at equilibrium need not minimize error after a finite run. Prior work on short-run energy-based models establishes the importance of this distinction (Nijkamp et al., 2019; Agoritsas et al., 2023).

We study whether terminal sample feedback can improve finite-run accuracy by changing the acceptance energy alone, while retaining the initialization, proposal kernel, and schedule. A two-state chain illustrates why this can help (Appendix B.3). Let the target be uniform on {a, b}, start the chain at a, and propose a flip with probability 3/4. At inverse temperature one, the correct energy accepts every flip and returns b with probability 3/4 after one step. Raising the energy of b by log(3/2) lowers the acceptance probability to 2/3, making the one-step output exactly uniform but moving the equilibrium probability of b to 2/5.

We introduce finite-horizon energy learning,<sup>1</sup> which optimizes the acceptance energy using discriminator rewards on completed K-step samples. For pretrained graph generators such as Graph Energy Matching (GEM; Balcerak et al., 2026), which refines graphs with Metropolis–Hastings (MH) steps using gradientinformed proposals from a learned potential (Zhang et al., 2022b), this separates the frozen potential used for initialization and proposals from the energy adapted for acceptance. An acceptance decision changes the state from which subsequent proposals are drawn, so its efect must be assessed through the remaining trajectory.

Following the coupled MH derivative of Arya et al.

![](images/ca2ccc7a7e3bbb764e898077314bfefc9df85aa44f02636811f68e62be2302d1.jpg)  
Figure 1: Finite-horizon energy learning for a pretrained GEM graph sampler. Generation (top): one K-step MH trajectory whose initialization and proposals use GEM’s frozen potential $V _ { \mathrm { G } } ;$ only the acceptance energy $E _ { \theta }$ , with coeficients $\theta = \left( c , \omega \right)$ in (6), is learned. Training (dashed; absent at generation): at a selected decision t, the accepted and rejected branches are continued to step K with shared future random numbers, and the diference of their terminal rewards weights $\nabla _ { \boldsymbol { \theta } } \alpha _ { \boldsymbol { \theta } , t }$ . The displayed update is (5) with selection indicators $I _ { i t }$ and inclusion probabilities $p _ { i t } ;$ N is the number of base chains.

(2023, 2024), we continue both the accepted and rejected branches of a selected decision to the terminal horizon using shared random numbers. Their terminalreward diference weights the derivative of the local acceptance probability. The discriminator and additional continuations are used only during training; generation uses the learned acceptance energy in a single trajectory (Figure 1).

The finite-state study isolates acceptance-energy adaptation from density misspecification by comparing against a target energy that is already correct at equilibrium, and against a numerically tuned scalar multiple of that energy. The molecular study then applies the same terminal-reward objective to a two-coeficient combination of frozen GEM potentials on MOSES (Polykovskiy et al., 2020). Our contributions are:

• Finite-horizon optimization of acceptance energies. We formulate learning through the MH acceptance energy to maximize expected terminal discriminator reward, with initialization, the proposal kernel, and the annealing schedule fixed. Applying established coupled MH derivatives to subsampled accept/reject continuations gives an unbiased gradient estimator for a fixed discriminator under the conditions in Section 3.3.

• Exact finite-state evidence beyond scalar tempering. At $K \in \{ 4 , 8 , 1 6 \}$ , learned energies achieve lower median terminal total variation (TV) than the equilibrium-correct energy in all 18 reference and all 18 new-target settings. They also outperform its numerically tuned scalar multiple in 14 and 15 settings, respectively. These comparisons identify finite-horizon improvements beyond the reported scalar-temperature control in most settings (Section 4).

• Adaptation of frozen graph potentials. On MOSES, a two-coeficient acceptance energy combining pretrained GEM potentials reduces Fréchet ChemNet distance (FCD) by 3.7–10.5% relative to scale-calibrated GEM with exact MH acceptance at matched refinement steps, with budgetdependent validity trade-ofs. Generation retains a single trajectory and requires neither discriminator evaluations nor additional continuations (Section 5).

## 2 RELATED WORK

Training energies with finite-run samplers. Short-run EBMs treat a fixed number of MCMC steps as part of the generator (Nijkamp et al., 2019; Agoritsas et al., 2023), and contrastive divergence and GEM’s contrastive refinement raise energy on generated negatives held fixed during diferentiation (Hinton, 2002; Balcerak et al., 2026). These methods learn the energy, and with it the sampler’s target, from short-run negatives. We instead fix a pretrained sampler’s initialization and proposals and learn only its acceptance energy, for an objective defined on the K-step output.

Adapting proposals and transition dynamics. ALOE and EB-GFN learn discrete energies jointly with amortized samplers (Dai et al., 2020; Zhang et al., 2022a), A-NICE-MC trains MCMC proposals adversarially (Song et al., 2017), and reinforcement learning adapts MCMC kernels (Wang et al., 2025a). For graphs, DeFoG separates the trained denoiser from the choice of sampling dynamics (Qin et al., 2025). These approaches change how states are proposed or how the sampler moves; we keep the proposal kernel and schedule fixed and change only the acceptance energy.

Diferentiating through discrete randomness. Coupled stochastic derivatives diferentiate programs with discrete random choices (Arya et al., 2022), and Arya et al. (2023, 2024) specialize them to MH by coupling accepted and rejected continuations. We apply this estimator to a terminal discriminator reward, subsample decisions with inverse inclusion probabilities (Horvitz and Thompson, 1952), and compare it with a score-function estimator in Section 4.

Graph generation and reward adaptation. Discrete difusion and flow models generate graphs by learned denoising (Vignac et al., 2023; Campbell et al., 2022; Siraudin et al., 2024), whereas GEM samples from a learned potential with MH refinement. Reward fine-tuning methods such as GDPO, DRAKES, and Graph-GRPO update the denoiser or flow network (Liu et al., 2024; Wang et al., 2025b; Zhu et al., 2026), and TFG-Flow and clean-sample Markov chains apply a reward or conditioning signal during sampling (Lin et al., 2025; Phunyaphibarn and Sung, 2026). Here the pretrained networks stay fixed; only the acceptance energy changes, contrasting a model with a weaker version of itself as in autoguidance (Karras et al., 2024).

## 3 FINITE-HORIZON ENERGY LEARNING

## 3.1 Sampler and Terminal Law

Let $p$ be the data distribution on a finite discrete state space $x ;$ for graphs, a state assigns a categorical label to every node and node pair, including a no-edge category. Starting from $X _ { 0 } \sim q _ { 0 }$ , at steps $t = 0 , \ldots , K - 1$ the sampler proposes $Y _ { t } \sim Q _ { t } ( \cdot \mid X _ { t } )$ and accepts it with the MH probability

$$
\alpha _ { \theta , t } ( x , y ) = \operatorname* { m i n } \left\{ 1 , e ^ { - \beta _ { t } \left[ E _ { \theta } ( y ) - E _ { \theta } ( x ) \right] } \frac { Q _ { t } ( x \mid y ) } { Q _ { t } ( y \mid x ) } \right\} ;\tag{1}
$$

rejection leaves the state unchanged. Writing $P _ { \theta , t }$ for the resulting transition kernel, the terminal law after K steps is

$$
q _ { \theta , K } = q _ { 0 } P _ { \theta , 0 } P _ { \theta , 1 } \cdot \cdot \cdot P _ { \theta , K - 1 } .\tag{2}
$$

The initialization $q _ { 0 }$ , the proposal kernel $Q _ { t }$ , and the inverse-temperature schedule $\beta _ { t }$ do not depend on $\theta ,$ which enters only through the acceptance energy $E _ { \theta }$ . A fixed proposal kernel does not imply identical realized proposals: once two runs make diferent decisions, later proposals start from diferent states. The annealed kernels need not share a common invariant distribution; our objective is defined on the terminal law ${ q } _ { \theta , K }$ , not on the equilibrium density ∝ $\cdot e ^ { - \beta E _ { \theta } }$ at a fixed temperature.

## 3.2 Fixed-Discriminator Objective

A discriminator with logit $d _ { \psi } ( x )$ is trained to distinguish data from samples of $q _ { \theta , K }$ . With $\sigma$ the logistic function and terminal reward $\boldsymbol { r } _ { \psi } ( \boldsymbol { x } ) = \log \sigma ( d _ { \psi } ( \boldsymbol { x } ) )$ ， each energy update ascends the non-saturating generator objective (Goodfellow et al., 2014)

$$
J ( \theta ; \psi ) = \mathbb { E } _ { X _ { K } \sim q _ { \theta , K } } [ r _ { \psi } ( X _ { K } ) ]\tag{3}
$$

with $\psi$ held fixed, and the discriminator is refitted between updates. If the discriminator is the optimal unrestricted classifier for the balanced, unweighted population problem, held fixed at the current generator, and $p > 0$ wherever $q _ { \theta , K } > 0$ , then $- \nabla _ { \theta } J$ is the gradient of a divergence between ${ q } _ { \theta , K }$ and $p$ (Appendix B), which Section 4 uses as a numerical reference. The weighted, finitely trained molecular discriminator need not satisfy these conditions; the estimator below holds for any fixed discriminator.

Exact likelihood training of an energy takes its negativephase expectation under the equilibrium model distribution. GEM’s contrastive term instead raises energy on generated negatives held fixed during diferentiation (Balcerak et al., 2026, Section 3.4 and Appendix I); with short-chain negatives, such a detached update need not equal the exact likelihood gradient (Hinton, 2002; Agoritsas et al., 2023). Even exact likelihood training targets the equilibrium law: with population data and an energy family that contains the target, its optimum is the equilibrium-correct energy, which need not be the best acceptance energy for a K-step run. Equation (3) instead scores the returned samples.

## 3.3 Gradient Estimation with Paired Continuations

Continuation values. Let $V _ { t + 1 } ( s ) = \mathbb { E } [ r _ { \psi } ( X _ { K } ) \ |$ $X _ { t + 1 } = s ]$ be the expected terminal reward of a chain that is in state s after step t and continues under the current energy and the remaining schedule. Because q<sub>0</sub> and $Q _ { t }$ do not depend on $\theta ,$ the product rule applied to (2) gives (Arya et al., 2023, 2024)

$$
\nabla _ { \theta } J = \sum _ { t = 0 } ^ { K - 1 } \mathbb { E } \big [ \Delta V _ { t + 1 } \nabla _ { \theta } \alpha _ { \theta , t } ( X _ { t } , Y _ { t } ) \big ] ,\tag{4}
$$

where $\Delta V _ { t + 1 } = V _ { t + 1 } ( Y _ { t } ) - V _ { t + 1 } ( X _ { t } )$ and $( X _ { t } , Y _ { t } )$ are the state and proposal of a chain run with the current energy (Appendix A.1). Each term diferentiates only the local acceptance probability; the efect of the decision on later steps enters through $\Delta V _ { t + 1 } ,$ , whose sign indicates whether accepting or rejecting yields the higher expected terminal reward.

Paired continuations. The two possible next states are $X _ { t + 1 } ^ { + } = Y _ { t }$ and $X _ { t + 1 } ^ { - } = X _ { t }$ . Continuing each to step K gives rewards $R ^ { \pm } = r _ { \psi } ( X _ { K } ^ { \pm } )$ , and $\mathbb { E } [ R ^ { + } - R ^ { - } \ |$ $X _ { t } , Y _ { t } ] \ = \ \Delta V _ { t + 1 }$ whenever each branch follows the correct transition law. Both continuations use the same future random numbers, mapped through their own proposal distributions; this coupling changes only the joint law of the branches and can reduce the variance of $R ^ { + } - R ^ { - }$ One branch coincides with the base trajectory, so only the other is simulated. Branches that reach the same state with the same carried sampler quantities agree from then on and can be merged.

Subsampled decisions. To limit simulation, we select a subset S of decisions from N base chains, including step t of chain i with probability $p _ { i t }$ , and use

$$
\widehat { g } = \frac { 1 } { N } \sum _ { ( i , t ) \in \mathcal { S } } \frac { R _ { i t } ^ { + } - R _ { i t } ^ { - } } { p _ { i t } } \nabla _ { \theta } \alpha _ { \theta , t } ( X _ { i t } , Y _ { i t } ) .\tag{5}
$$

For a fixed discriminator, $\widehat g$ is unbiased for (4) if (i) the acceptance probabilities are diferentiable at the sampled decisions, (ii) both continuations follow the correct law, (iii) $p _ { i t } > 0$ wherever $\nabla _ { \boldsymbol { \theta } } \alpha _ { \boldsymbol { \theta } , t } \neq 0$ , and (iv) selection depends only on the chain prefix, the proposal, and quantities fixed before the update, not on continuation randomness. The sum is normalized by N, not by |S|, and states, rewards, and selection probabilities are held fixed during diferentiation. At unsaturated proposals, $\nabla _ { \theta } \alpha _ { \theta , t } = - \beta _ { t } \alpha _ { \theta , t } \nabla _ { \theta } [ E _ { \theta } ( Y _ { t } ) - E _ { \theta } ( X _ { t } ) ]$ , and strictly saturated proposals have zero derivative. At an MH tie, where the acceptance ratio equals one while its derivative is nonzero, $\alpha _ { \theta , t }$ is not diferentiable; the implementation uses zero there, and the unbiasedness statement excludes such points. The statement also does not extend to clipping, coordinate constraints, adaptive optimization, discriminator refitting, or convergence of the alternating procedure (Appendix A.1).

## 3.4 Training and Generation

Algorithm 1 starts from an initial energy and a discriminator fitted to data and initial generated samples, and alternates one ascent step Update(θ, gb) with discriminator refitting. The exact-state experiments select one uniformly random decision per chain; molecular training selects about four per chain, with inclusion probabilities increasing with the acceptance-gradient magnitude, scaled using the preceding update (Appendix A). Both studies report the final iterate, without test-based checkpoint selection; Appendix A.4 bounds the transitions simulated per update. At generation time, the learned energy replaces the original acceptance energy in the same K-step sampler, without the discriminator or continuations; the per-step cost can still depend on the energy (Section 5).

Algorithm 1 Finite-horizon energy learning   
Require: Data $\mathcal { D } ;$ initial $\theta , \psi ;$ fixed $q _ { 0 } , Q _ { t } , \beta _ { t } ;$ horizon   
$K ; N$ base chains   
1: for each energy update do   
2: Freeze $\theta , \psi ,$ and the decision-selection rule   
3: Simulate $N$ chains to $K ;$ cache random draws   
4: Select decisions $s$ with probabilities $p _ { i t }$   
5: for $( i , t ) \in S$ do   
6: $X _ { i , t + 1 } ^ { + }  Y _ { i t } , \quad X _ { i , t + 1 } ^ { - }  X _ { i t }$   
7: Continue $X ^ { + } , X ^ { - }$ to $K$ with shared draws   
8: $R _ { i t } ^ { \pm } \gets r _ { \psi } ( X _ { i K } ^ { \pm } )$   
9: end for   
10: Compute $\widehat g$ by (5); θ ← Update $i ( \theta , { \widehat { g } } )$   
11: Refit $d _ { \psi }$ on $\mathcal { D }$ and $\{ X _ { i K } \} _ { i = 1 } ^ { \tilde { N } }$   
12: Refresh the selection rule for the next update   
13: end for   
14: return $E _ { \theta }$ ▷ used in a single trajectory

## 4 EXACT FINITE-STATE STUDY

We first ask whether an energy that is exact at equilibrium can be improved for a short run, and whether a scalar temperature change explains the improvement. Small state spaces allow exact terminal laws, while learning uses sampled trajectories.

## 4.1 Targets, Sampler, and Comparisons

Targets are synthetic pairwise binary models $p \propto e ^ { - H }$ on $L = 6$ or $L = 8$ spins (64 or 256 configurations) with random fields and couplings (Appendix B). For each $L \in \{ 6 , 8 \}$ , the reference set contains three independently drawn target distributions, and the new-target set contains three further draws. Each target is specified by a fixed set of unary fields and pairwise couplings, held constant across horizons and training seeds. We use identifiers such as Ref-6-1 and New-8-3 to specify the target set, number of spins, and instance index. The index identifies a distribution, not a training seed. Training is repeated from scratch on each new target; no fitted energy is transferred between targets. The sampler starts from the product of the target’s onevariable marginals, proposes a flip of one uniformly chosen spin, and raises the inverse temperature from 0.2 to 1 over the first half of the run before holding it fixed. Every compared energy uses this sampler at a given horizon; unlike GEM’s fixed-length ramp, the schedule scales with K. We compare:

![](images/9438bb5ddb5a46beccb56c02b30e1b95ac74bdd9c492bed96cf9ea564cf72b25.jpg)

![](images/4e8eca97fc61d1b46f9ef95299ef268e5317fbed82b80ca394c38b9814b73024.jpg)

![](images/4440409ad2dd929124d422e077b1b0a87c18e3c766687d3fecd0c44c74278935.jpg)  
Figure 2: Finite-time sampling on the reference targets (original runs). (a) Exact terminal total variation (TV) against the number of steps $K ;$ each learned point is an energy trained separately for that horizon with $N = 5 1 2$ trajectories per update. Curves are medians over six targets after aggregating training seeds. (b) Recovery (%) of the numerical reference’s improvement over the restricted energy, for learning with a fitted or an oracle discriminator, against trajectories per update $N ;$ 100% is not an upper bound. (c) TV-minimizing multiplier $\beta ^ { * }$ of $E = \beta H$ under the shared annealing schedule, for individual targets (thin) and their median (thick); the dashed line marks $\beta = 1$ . Shading shows interquartile ranges, not confidence intervals. Horizontal axes are logarithmic, as is the vertical axis in (c).

• Restricted energy: unary and ring interactions, fitted by population maximum likelihood, scalarcalibrated for TV; learning starts from it.

• Correct H: the target energy, recovered by population pseudo-likelihood (Besag, 1975), exact at equilibrium.

• Scalar-tuned βH: the multiple of H that minimizes exact terminal TV, an oracle control.

• Learned, fitted discriminator (ours): all unary and pairwise coeficients (21 for $L = 6 ,$ 36 for $L = 8 )$ 2 a family that contains H, trained by Algorithm 1 with a discriminator linear in the same features.

• Learned, oracle discriminator: the same sampled training, with rewards from the exact density ratio between the target and the current generator.

• Numerical reference: exact-gradient minimization over the learned family of the divergence $D ( q \| p ) =$ 2 $\operatorname { K L } ( { \frac { 1 } { 2 } } ( p + q ) \parallel p )$ of Section 3.2; it is not optimized for TV and not certified optimal.

Learned energies use 2,000 updates, $N \ = \ 5 1 2$ trajectories per update unless stated otherwise, three training seeds, and the final iterate. The fitted discriminator starts from the exact population logistic fit and is then refitted on target and generated samples; exact marginals and population fits are idealizations of this study. We report total variation, $\begin{array} { r } { \mathrm { T V } ( q , p ) = \frac { 1 } { 2 } \sum _ { x } \vert q ( x ) - p ( x ) \vert } \end{array}$ |, computed exactly, so reported variation comes only from targets and training seeds.

## 4.2 Comparison with the Equilibrium-Correct Energy

At K = 4, 8, 16, the learned energy has lower median TV than H in all 18 target–horizon settings of the reference set (Figure 2a, Table 1). A fresh-seed replication on the same targets again improves all 18 settings, with 53 of 54 individual fits better than H. On the new targets, learning improves all 18 settings and all 54 fits; median TV falls from 0.553 to 0.518, from 0.413 to 0.307, and from 0.240 to 0.069 at K = 4, 8, 16. Because H is exact at equilibrium, these gains correct the finite-run law rather than the density. Horizons within a target are dependent, so the counts describe the experiment rather than constitute independent tests.

## 4.3 Comparison with Scalar Tempering

Scalar tempering tests whether rescaling H explains these gains. Learned energies have lower TV than scalar-tuned βH in 14 of 18 reference settings, in both the original runs and the replication, and in 15 of 18 new-target settings. The three new-target exceptions all occur on New-8-2, where the scalar-tuned control has lower TV at each tested horizon (Table 4). The control is an oracle, since it uses the target and exact terminal probabilities. Its optimal multiplier is far from one at short horizons: the median over reference targets falls from 9.04 at K = 4 to 2.00 at $K = 8$ and 1.22 at K = 16 (Figure 2c). This is a target-specific calibration under the horizon-scaled schedule, not a general law relating temperature to budget.

Table 1: Exact terminal TV at short horizons (lower is better). Entries are medians over six targets of each target’s median over three training seeds. Reference rows summarize the original runs in Figure 2a; Table 4 reports per-target results for a fresh-seed replication on the reference targets and for the new targets. Bold marks the smaller median.
<table><tr><td>Targets</td><td>K</td><td>Correct H</td><td>Learned (ours)</td></tr><tr><td rowspan="3">Reference (original runs)</td><td>4</td><td>0.329</td><td>0.197</td></tr><tr><td>8</td><td>0.201</td><td>0.063</td></tr><tr><td>16</td><td>0.089</td><td>0.030</td></tr><tr><td rowspan="3">New targets</td><td>4</td><td>0.553</td><td>0.518</td></tr><tr><td>8</td><td>0.413</td><td>0.307</td></tr><tr><td>16</td><td>0.240</td><td>0.069</td></tr></table>

## 4.4 Training Budget, Estimators, and Long Horizons

Figure 2b varies the number of training trajectories per update, N. For each target and horizon, with TV errors e<sub>base</sub>, e<sub>learned</sub>, and $e _ { \mathrm { r e f } }$ of the restricted, learned, and reference energies, the recovery is $1 0 0 ( e _ { \mathrm { b a s e } } - e _ { \mathrm { l e a r n e d } } ) / ( e _ { \mathrm { b a s e } } - e _ { \mathrm { r e f } } )$ , computed in each of the 42 settings (six targets, seven horizons) before taking medians. Zero means no gain over the restricted energy and 100 means matching the reference; because the restricted family is misspecified, the gain combines density and finite-time corrections. With the fitted discriminator, median recovery rises from 52.4% at N = 32 to 79.4% at 128 and 90.4% at 512, where the median TV gap to the reference is 0.024. With the oracle discriminator, it reaches 96.8% at $N = 5 1 2$ . At matched numbers of simulated transitions, a leave-oneout score-function estimator also improves on H in all 18 short-horizon reference settings; coupling has lower median TV in 12 of them (median diference −0.0015) but took 4.66 times the CPU time (Appendix A.3). Terminal feedback is thus efective with either estimator, without an eficiency advantage for coupling.

Learning is least efective at long horizons. Each point in Figure 2a is a separately trained energy; beyond

$K = 6 4 .$ , the learned energies are less accurate than H, while the numerical reference stays near zero TV. Because the learned family contains both H and the reference solution, the shortfall arises in sampled training, not in the energy family. Discriminator accuracy, gradient variance over long continuations, and the fixed optimization budget are candidate causes that this study does not separate.

## 5 ADAPTING PRETRAINED GEM POTENTIALS ON MOSES

We next adapt a pretrained graph sampler, changing only the acceptance energy and comparing at matched numbers of refinement steps.

## 5.1 Two-Coeficient Energy and Controlled Comparison

We use the pretrained GEM model (Balcerak et al., 2026) on MOSES (Polykovskiy et al., 2020). Each chain fixes a node count drawn from the training distribution. GEM initializes a random graph with at most 225 greedy edits under its released potential $V _ { \mathrm { G } } ;$ K refinement steps then use gradient-informed categorical proposals and the released inverse-temperature schedule, which rises from about 0.78 to 17.62 over 500 steps and then stays constant. We retain this initialization, proposal mechanism, and schedule. K counts refinement steps after initialization, not all edits, network evaluations, or elapsed computation, and runs with $K < 5 0 0$ stop during the ramp. The learned acceptance energy is

$$
E _ { \theta } ( x ) = c \big [ V _ { \mathrm { G } } ( x ) + \omega \{ V _ { \mathrm { G } } ( x ) - V _ { \mathrm { p r e } } ( x ) \} \big ] ,\tag{6}
$$

where $V _ { \mathrm { p r e } }$ is GEM’s frozen transport-only predecessor checkpoint, c sets the energy scale, and ω weights the diference between checkpoints, as in autoguidance (Karras et al., 2024). Equation (6) is the learned energy $E _ { \theta }$ of Figure 1, with $\theta = \left( c , \omega \right)$ ; both potentials stay frozen, and the combination remains invariant to node relabeling. Rewards come from a training-only discriminator, a LoRA-adapted copy of $V _ { \mathrm { G } }$ with about 9.7 million trainable parameters (Hu et al., 2022).

The baseline at each budget is GEM with exact MH acceptance and an acceptance scale calibrated on development FCD: 0.70 at K = 125, 250 and 0.865 at $K = 1 0 0 0$ , with ω = 0. Learning starts from this calibrated energy. Exact MH difers from GEM’s released acceptance rule in how proposal retry stages enter the acceptance ratio (Appendix C); the released rule is used only to reuse the trained energies without retraining (Table 5). Training uses 80 energy updates with 256 base chains and about four selected decisions per chain. The discriminator is fitted to generated graphs and node-count-matched training molecules, with invalid generated graphs up-weighted, and is refitted after each update (Appendix A). The energy-update objective uses discriminator rewards rather than FCD; development FCD informs baseline calibration and discriminator screening, and it selects the static-coeficient controls in Section 5.4.

Table 2: GEM with and without finite-horizon energy learning at matched refinement steps K, with exact MH acceptance. Mean ± sample SD over three paired seeds, each with 25,000 requested graphs; validity and VUN are % of requested graphs. Bold marks the better mean in each pair, not significance.
<table><tr><td>K</td><td>Method</td><td>FCD ↓</td><td>Validity ↑</td><td>VUN↑</td></tr><tr><td rowspan="2">125</td><td>GEM (calibrated)</td><td> $2 . 5 0 8 \pm 0 . 0 2 6$ </td><td> $8 7 . 1 3 \pm 0 . 1 4$ </td><td> $8 5 . 1 5 \pm 0 . 0 9$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 2 . 3 6 6 \pm 0 . 0 6 3 }$ </td><td> ${ \bf 9 2 . 5 5 \pm 0 . 2 4 }$ </td><td> ${ \bf 9 0 . 4 4 \pm 0 . 2 5 }$ </td></tr><tr><td rowspan="2">250</td><td>GEM (calibrated)</td><td> $2 . 0 4 4 \pm 0 . 0 3 0$ </td><td> ${ \bf 9 5 . 4 3 \pm 0 . 1 0 }$ </td><td> ${ \bf 9 2 . 7 6 \pm 0 . 0 5 }$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 1 . 8 2 9 \pm 0 . 0 2 1 }$ </td><td> $9 4 . 6 5 \pm 0 . 3 3$ </td><td> $9 2 . 4 5 \pm 0 . 2 9$ </td></tr><tr><td rowspan="2">1000</td><td>GEM (calibrated)</td><td> $1 . 5 6 1 \pm 0 . 0 1 8$ </td><td> ${ \bf 9 7 . 8 4 \pm 0 . 1 4 }$ </td><td> $9 3 . 7 7 \pm 0 . 1 2$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 1 . 5 0 4 \pm 0 . 0 1 8 }$ </td><td> $9 7 . 8 2 \pm 0 . 0 3$ </td><td> ${ \bf 9 3 . 9 2 \pm 0 . 0 9 }$ </td></tr></table>

## 5.2 Metrics and Decision Rule

We report Fréchet ChemNet distance (FCD) (Preuer et al., 2018), validity, and VUN. FCD compares the means and covariances of ChemNet features of generated molecules with those of the MOSES scafold-test reference of 176,225 molecules, using the canonical largest fragment of each valid output and retaining duplicates. Validity is the percentage of requested graphs that decode and sanitize; following GEM’s convention, a disconnected but sanitizable molecule counts as valid. VUN is the percentage of requested graphs that are valid, unique, and absent from the training set. Failed outputs are retained, so both rates use all requested graphs as denominators. Development FCD instead uses a fixed reference of 16,010 molecules selected deterministically from the MOSES training data, with 2,048 generated graphs per candidate from a generation stream separate from the final evaluation; this split does not imply that these molecules were excluded from the pretraining data of the released GEM checkpoints. Because encoding and processing choices afect the scores, they are compared within this implementation, not with published leaderboard values.

Each comparison uses three seeds with 25,000 requested graphs each; both methods share the generation seed paired with each training seed, as well as the initialization, proposals, schedule, decoder, and evaluator, and seed-level diferences are the units of replication. An internally prespecified joint rule requires lower FCD in a one-sided paired t test at level 0.05 and one-sided 95% lower confidence bounds above −1 percentage point for the validity and VUN diferences. The exact-MH comparison at K = 250 was the original primary comparison; the other budgets and the released-rule analysis followed it, and no multiplicity adjustment is applied (Appendix C.3). Matched steps are not matched compute: at K = 250, generation took 0.932 H100 GPUseconds per requested graph with the learned energy versus 0.859 for calibrated GEM.

## 5.3 Results

The learned energy lowers FCD in every paired seed at all three budgets (Table 2, Figure 3); relative changes use unrounded seed means. At K = 125, all three metrics improve: FCD decreases by 5.7% (2.508 to 2.366), validity rises by 5.42 points, and VUN by 5.29 points; these yield gains are relative to a baseline scale selected for FCD and partly reflect that choice. At K = 250, FCD decreases by 10.5% (2.044 to 1.829), while mean validity and VUN decrease by 0.78 and 0.32 points. The mean validity decrease was below one percentage point, but its one-sided lower confidence bound (−1.25 points) crossed the prespecified margin; non-inferiority was therefore not established, and this original primary comparison does not meet the joint criterion. At K = 1000, FCD decreases by 3.7% (1.561 to 1.504), validity and VUN change by −0.02 and +0.15 points. The joint criterion is met at K = 125 and 1000.

The K = 125 yield gains persist under strict scoring that requires connected molecules and passes stricter representation checks (validity +4.77, VUN +4.65 points), and equalizing the number of valid molecules entering FCD changes the paired diference only from −0.142 to −0.138. These checks reuse the generated samples and address evaluation sensitivity, not the efect of encoding mismatches in the discriminator’s training pool. Reusing the learned energies without retraining under GEM’s released acceptance rule again lowers FCD at all budgets. The joint criterion holds at $K = 2 5 0$ and 1000 but not at $K = 1 2 5$ , which used two prespecified seeds and an uncalibrated released-GEM baseline (Table 5); these reuse analyses are not independent replications.

![](images/b513e5b1d8cebe45fd22b9dcc985ba2445edd652b177a6d633eecc3591c0d855.jpg)  
Figure 3: FCD–validity changes at matched budgets. Circles: calibrated GEM; squares: GEM with the learned energy (both exact MH); arrows join matchedbudget pairs. FCD decreases at every budget, whereas validity rises at $K = 1 2 5$ , falls at $K = 2 5 0$ , and is nearly unchanged at $K = 1 0 0 0$ . Bars show one sample SD over three seeds; each budget has its own calibrated baseline and trained coeficients.

## 5.4 Same-Family Controls and Protocol Conditions

The baseline calibrates one coeficient while learning adjusts two, so Table 2 does not separate the efect of gradient-based learning from that of a second coeficient. Static two-coeficient settings selected on development FCD reach FCD 2.280 at $K = 1 2 5$ (learned: 2.366), with lower validity (91.10% versus 92.55%); this FCD diference is unresolved with three seeds. ${ \mathrm { A t ~ } } K = 2 5 0 .$ the static setting reaches 1.811 (learned: 1.829) but fails its development validity threshold. Finite-diference training under a diferent optimization schedule reaches exact-rule FCD 1.821. These controls use diferent selection and optimization budgets and compute costs, and there is no corresponding three-seed static evaluation at $K = 1 0 0 0$ (Appendix C.4). The reported comparisons do not establish an FCD advantage over coeficient selection or finite-diference training within the same energy family. Lower FCD also does not imply that every molecular property improves: molecular-weight distance worsens at $K = 1 0 0 0$

Three protocol conditions afect interpretation $( \mathrm { A p - }$ pendix C.3). At $K \ : = \ : 1 2 5$ , the allowed rate of reencoding mismatches in the discriminator’s training pool was raised from 1% to 2% before the pool was generated; the pool had 1.65% mismatches, so the original threshold would have stopped the experiment. At $K = 1 0 0 0$ , the discriminator passed its development screen only with a revised negative pool. At $K = 5 0 0$ the screen failed and no energy was trained, so that budget is missing, not an evaluated failure.

## 6 DISCUSSION AND CONCLUSION

We formulated finite-horizon energy learning to optimize the distribution returned by an MH sampler while retaining its initialization, proposal kernel, and annealing schedule. The finite-state experiments isolate a benefit of this objective: learned energies improve median terminal TV over an equilibrium-correct energy in every tested short-horizon target–horizon setting, and over its numerically tuned scalar multiple in most settings. These gains concern the finite-run distribution rather than correction of an inaccurate equilibrium density.

On MOSES, adapting two coeficients of frozen GEM potentials reduces FCD by 3.7–10.5% relative to scalecalibrated GEM with exact MH acceptance at matched refinement steps. The adapted generator uses a single trajectory without discriminator evaluations or additional continuations. The molecular results do not establish superiority over same-family coeficient selection, and the original K = 250 comparison does not establish validity non-inferiority. Thus, the observed FCD gains should not be interpreted as uniformly improved sample quality or reduced computational cost.

The present procedure trains a separate energy for each horizon, and sampled training falls behind the correct energy at long horizons in the finite-state study. Training across budgets and understanding this longhorizon shortfall remain directions for further work. The results support acceptance-energy learning as a means of adapting finite-run generation while retaining the proposal mechanism.

## AI use statement

Generative AI tools were used to assist with drafting portions of the manuscript and revising the text for clarity, organization, consistency checks and readability. The authors take full responsibility for the final manuscript, including its originality, technical claims, reported results, and references.

## References

Agoritsas, E., Catania, G., Decelle, A., and Seoane, B. (2023). Explaining the efects of non-convergent MCMC in the training of energy-based models. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research.

Arya, G., Schauer, M., Schäfer, F., and Rackauckas, C. (2022). Automatic diferentiation of programs with discrete randomness. In Advances in Neural Information Processing Systems, volume 35.

Arya, G., Schauer, M., and Seyer, R. (2024). Gradient estimation via diferentiable Metropolis-Hastings. arXiv preprint arXiv:2406.14451.

Arya, G., Seyer, R., Schäfer, F., Chandra, K., Lew, A. K., Huot, M., Mansinghka, V. K., Ragan-Kelley, J., Rackauckas, C., and Schauer, M. (2023). Diferentiating Metropolis-Hastings to optimize intractable densities. In ICML 2023 Workshop on Diferentiable Almost Everything. arXiv:2306.07961.

Balcerak, M., Shit, S., Prabhakar, C., Kaltenbach, S., Albergo, M. S., Du, Y., and Menze, B. (2026). Graph energy matching: Transport-aligned energybased modeling for graph generation. arXiv preprint arXiv:2603.23398.

Besag, J. (1975). Statistical analysis of non-lattice data. Journal of the Royal Statistical Society. Series D (The Statistician), 24(3):179–195.

Campbell, A., Benton, J., De Bortoli, V., Rainforth, T., Deligiannidis, G., and Doucet, A. (2022). A Continuous Time Framework for Discrete Denoising Models. In Advances in Neural Information Processing Systems.

Dai, H., Singh, R., Dai, B., Sutton, C., and Schuurmans, D. (2020). Learning discrete energy-based models via auxiliary-variable local exploration. In Advances in Neural Information Processing Systems, volume 33.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., and Bengio, Y. (2014). Generative adversarial nets. In Advances in Neural Information Processing Systems, volume 27.

Hinton, G. E. (2002). Training products of experts by minimizing contrastive divergence. Neural Computation, 14(8):1771–1800.

Horvitz, D. G. and Thompson, D. J. (1952). A generalization of sampling without replacement from a finite universe. Journal of the American Statistical Association, 47:663–685.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Karras, T., Aittala, M., Kynkäänniemi, T., Lehtinen, J., Aila, T., and Laine, S. (2024). Guiding a difusion model with a bad version of itself. In Advances in Neural Information Processing Systems, volume 37.

Lin, H., Li, S., Ye, H., Yang, Y., Ermon, S., Liang, Y., and Ma, J. (2025). TFG-Flow: Training-free Guidance in Multimodal Generative Flow. In International Conference on Learning Representations.

Liu, Y., Du, C., Pang, T., Li, C., Lin, M., and Chen, W. (2024). Graph Difusion Policy Optimization. In Advances in Neural Information Processing Systems.

Nijkamp, E., Hill, M., Zhu, S.-C., and Wu, Y. N. (2019). Learning non-convergent non-persistent short-run MCMC toward energy-based model. In Advances in Neural Information Processing Systems, volume 32.

Phunyaphibarn, P. and Sung, M. (2026). Reward-Guided Discrete Difusion via Clean-Sample Markov Chain for Molecule and Biological Sequence Design. arXiv preprint arXiv:2602.09424.

Polykovskiy, D., Zhebrak, A., Sanchez-Lengeling, B., Golovanov, S., Tatanov, O., Belyaev, S., Kurbanov, R., Artamonov, A., Aladinskiy, V., Veselov, M., Kadurin, A., Johansson, S., Chen, H., Nikolenko, S., Aspuru-Guzik, A., and Zhavoronkov, A. (2020). Molecular sets (MOSES): A benchmarking platform for molecular generation models. Frontiers in Pharmacology, 11:565644.

Preuer, K., Renz, P., Unterthiner, T., Hochreiter, S., and Klambauer, G. (2018). Fréchet ChemNet distance: A metric for generative models for molecules in drug discovery. Journal of Chemical Information and Modeling, 58(9):1736–1741.

Qin, Y., Madeira, M., Thanou, D., and Frossard, P. (2025). DeFoG: Discrete Flow Matching for Graph Generation. In Proceedings of the 42nd International Conference on Machine Learning.

Siraudin, A., Malliaros, F. D., and Morris, C. (2024). COMETH: A Continuous-Time Discrete-State Graph Difusion Model. arXiv preprint arXiv:2406.06449.

Song, J., Zhao, S., and Ermon, S. (2017). A-NICE-MC: Adversarial training for MCMC. In Advances in Neural Information Processing Systems, volume 30.

Vignac, C., Krawczuk, I., Siraudin, A., Wang, B., Cevher, V., and Frossard, P. (2023). DiGress: Discrete Denoising Difusion for Graph Generation. In International Conference on Learning Representations.

Wang, C., Chen, W. Y., Kanagawa, H., and Oates, C. J. (2025a). Reinforcement learning for adaptive MCMC. In Proceedings of The 28th International Conference on Artificial Intelligence and Statistics, volume 258 of Proceedings of Machine Learning Research, pages 640–648.

Wang, C., Uehara, M., He, Y., Wang, A., Biancalani, T., Lal, A., Jaakkola, T., Levine, S., Wang, H., and Regev, A. (2025b). Fine-Tuning Discrete Difusion Models via Reward Optimization with Applications to DNA and Protein Design. In International Conference on Learning Representations.

Zhang, D., Malkin, N., Liu, Z., Volokhova, A., Courville, A., and Bengio, Y. (2022a). Generative flow networks for discrete probabilistic modeling. In Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pages 26412–26428.

Zhang, R., Liu, X., and Liu, Q. (2022b). A Langevinlike Sampler for Discrete Distributions. In Proceedings of the 39th International Conference on Machine Learning.

Zhu, B., Bo, D., Zhang, D. C., and Wang, X. (2026). Graph-GRPO: Training Graph Flow Models with Reinforcement Learning. arXiv preprint arXiv:2603.10395.

# Learning Graph Energies for Finite-Time Generation: Supplementary Material

Appendix A derives the gradient estimator and describes the implementation, Appendix B describes the finite-state experiments, and Appendix C describes the molecular evaluation.

## A GRADIENT DERIVATION AND IMPLEMENTATION

## A.1 Gradient Identity and Estimator

We specialize the coupled MH derivative of Arya et al. (2023, 2024) to the finite-horizon objective. Fix the discriminator, write $r \ = \ r _ { \psi }$ as a column vector over $x ,$ , and abbreviate $P _ { t } = P _ { \theta , t }$ and $\alpha _ { t } ~ = ~ \alpha _ { \theta , t }$ , so that $J = q _ { 0 } P _ { 0 } \cdot \cdot \cdot P _ { K - 1 } r$ . Because $q _ { 0 }$ does not depend on $\theta ,$ the product rule gives

$$
\nabla _ { \theta } J = \sum _ { t = 0 } ^ { K - 1 } q _ { t } \left( \nabla _ { \theta } P _ { t } \right) V _ { t + 1 } , \qquad q _ { t } = q _ { 0 } P _ { 0 } \cdot \cdot \cdot P _ { t - 1 } , \qquad V _ { t + 1 } = P _ { t + 1 } \cdot \cdot \cdot P _ { K - 1 } r ,
$$

with $V _ { K } = r ; q _ { t }$ is the law of $X _ { t } ,$ , and $V _ { t + 1 } ( s ) = \operatorname { \mathbb { E } } [ r ( X _ { K } ) \mid X _ { t + 1 } = s ]$ . The MH kernel is $\begin{array} { r } { P _ { t } ( x , \cdot ) = \sum _ { y } Q _ { t } ( y \mid } \end{array}$ $x ) \{ \alpha _ { t } ( x , y ) \delta _ { y } + ( 1 - \alpha _ { t } ( x , y ) ) \delta _ { x } \}$ , and $Q _ { t }$ does not depend on $\theta ,$ so

$$
\left[ ( \nabla _ { \theta } P _ { t } ) V _ { t + 1 } \right] ( x ) = \sum _ { y } Q _ { t } ( y \mid x ) \nabla _ { \theta } \alpha _ { t } ( x , y ) \left\{ V _ { t + 1 } ( y ) - V _ { t + 1 } ( x ) \right\} .
$$

Taking the expectation over $X _ { t } \sim q _ { t }$ and $Y _ { t } \sim Q _ { t } ( \cdot \mid X _ { t } )$ gives (4). Self-proposals contribute nothing, because $V _ { t + 1 } ( y ) - V _ { t + 1 } ( x ) = 0$ when $y = x$ . Only the local acceptance probability is diferentiated in each term; the efect of a decision on later steps enters through the value diference.

Paired continuations. Given $( X _ { t } , Y _ { t } )$ , let $R ^ { + }$ and $R ^ { - }$ be the terminal rewards of continuations from $Y _ { t }$ and $X _ { t } ,$ , each following the chain’s transition law from step $t + 1$ onward under the current energy and schedule. Then $\mathbb { E } [ R ^ { + } - R ^ { - } \ | \ X _ { t } , Y _ { t } ] = V _ { t + 1 } ( Y _ { t } ) - V _ { t + 1 } ( X _ { t } )$ for any joint coupling of the two continuations; sharing random numbers changes only their joint law. By the Markov property, the base chain’s own continuation from its realized $X _ { t + 1 }$ has the correct conditional law, so it can serve as one branch, provided that whether step t is selected does not depend on the base chain’s future randomness. Shared uniforms pass through each branch’s categorical distribution, and branches merge only when their states and all carried sampler quantities coincide, after which they agree. Continuations keep their original step indices, so they use the remaining part of the schedule.

Subsampling. Let $I _ { i t }$ indicate that decision t of base chain i is selected in the current update, with $\mathrm { P r } ( I _ { i t } =$ $1 \mid \mathcal { F } _ { i t } ) = p _ { i t }$ , where $\mathcal { F } _ { i t }$ contains the prefix $X _ { i , 0 : t }$ , the proposal $Y _ { i t }$ , and quantities fixed before the update, and $I _ { i t }$ is independent of the continuation randomness given $\mathcal { F } _ { i t } . \mathrm { ~ H ~ } p _ { i t } > 0$ whenever $\nabla _ { \theta } \alpha _ { t } ( X _ { i t } , Y _ { i t } ) \neq 0$ , then

$$
\mathbb { E } \Big [ \frac { I _ { i t } } { p _ { i t } } ( R _ { i t } ^ { + } - R _ { i t } ^ { - } ) \nabla _ { \theta } \alpha _ { t } ( X _ { i t } , Y _ { i t } ) \Big ] = \mathbb { E } \big [ \{ V _ { t + 1 } ( Y _ { t } ) - V _ { t + 1 } ( X _ { t } ) \} \nabla _ { \theta } \alpha _ { t } ( X _ { t } , Y _ { t } ) \big ] ,
$$

with unselected terms set to zero. Summing over t and averaging over the N independent base chains shows that (5) is unbiased for $\nabla _ { \boldsymbol { \theta } } J$ . Dividing by the number of selected decisions instead of N would introduce a bias. States, rewards, and inclusion probabilities are treated as constants during diferentiation, so no gradient flows through sampled categorical states, inclusion weights, or the discriminator.

Regularity. Write $\Delta _ { t } = \beta _ { t } [ E _ { \theta } ( Y _ { t } ) - E _ { \theta } ( X _ { t } ) ] - \log \{ Q _ { t } ( X _ { t } \mid Y _ { t } ) / Q _ { t } ( Y _ { t } \mid X _ { t } ) \}$ , so that $\alpha _ { t } = \operatorname* { m i n } \{ 1 , e ^ { - \Delta _ { t } } \}$ . If $\Delta _ { t } > 0$ , then $\nabla _ { \theta } \alpha _ { t } = - \beta _ { t } \alpha _ { t } \nabla _ { \theta } [ E _ { \theta } ( Y _ { t } ) - E _ { \theta } ( X _ { t } ) ]$ , so increasing the proposed state’s energy relative to the current one lowers acceptance. If $\Delta _ { t } \ < \ 0$ , the proposal is strictly saturated and $\nabla _ { \theta } \alpha _ { t } = 0 ;$ if the reverse proposal probability is zero, $\alpha _ { t } = 0$ and its derivative vanishes. At a tie, $\Delta _ { t } = 0$ with $\nabla _ { \theta } \Delta _ { t } \neq 0 , \alpha _ { t }$ is not diferentiable; the implementation uses zero there, and no unbiasedness claim is made at ties. The statement assumes exact arithmetic and a fixed discriminator. It does not extend to floating-point efects, gradient clipping, coordinate constraints, adaptive optimization, discriminator refitting, or the alternating procedure as a whole, and it implies nothing about convergence or mixing.

Finite-state checks cover 96 configurations with asymmetric proposals, self moves, varying schedules, and the tie convention. Eleven deliberately injected implementation faults were detected where applicable. Additiona inclusion-weight and coordinate checks agree with enumerated gradients to approximately $1 0 ^ { - 1 3 }$ at regular points.

## A.2 Molecular Optimization

The coeficients in (6) are written in energy coordinates $( u , w )$ as $c = a _ { 0 } e ^ { u }$ and $\omega = w e ^ { - u }$ , so that $E _ { \theta } =$ $a _ { 0 } e ^ { u } V _ { \mathrm { G } } + a _ { 0 } w ( V _ { \mathrm { G } } - V _ { \mathrm { p r e } } )$ , starting at $( u , w ) = ( 0 , 0 )$ , which is the calibrated baseline. The initial scales are $a _ { 0 } = 0 . 7 0 , 0 . 7 0 , 0 . 8 6 4 9$ at $K = 1 2 5 , 2 5 0$ , 1000 (rounded to 0.865 in the main text). The learned energy is kept unchanged when the acceptance rule changes (Table 5); coeficients are not rescaled using another setting’s calibration.

AdamW operates on the reparameterized coordinates $( u ^ { \prime } , w )$ , where $u ^ { \prime } = \log ( e ^ { u } + 0 . 5 7 w )$ , so that $E _ { \theta } ~ =$ $a _ { 0 } [ e ^ { u ^ { \prime } } V _ { \mathrm { G } } + w ( 0 . 4 3 V _ { \mathrm { G } } - V _ { \mathrm { p r e } } ) ]$ . The feasible set is $u \in [ - 1 . 5 , 1 . 5 ] , w \in [ - 1 , 3 ]$ , and $e ^ { u } + 0 . 5 7 w \geq e ^ { - 1 . 5 }$ , so the argument of the logarithm defining $u ^ { \prime }$ is at least $e ^ { - 1 . 5 }$ . After each update, u and w are clipped to their intervals, and w is then raised where necessary to satisfy the last constraint. The discriminator logit is $- \beta _ { \mathrm { r e f } } ( V _ { \psi } - V _ { \mathrm { G } } )$ where $V _ { \psi }$ is the LoRA-adapted potential and $\beta _ { \mathrm { r e f } } = 1 7 . 6 1 9 5 5 2 6 7 9 5 9 7 8 5 2$ is fixed to the final inverse temperature of GEM’s released acceptance schedule; it is held constant across sampling budgets and discriminator updates.

Table 3: Molecular training configuration. Both pretrained potentials are frozen; only the two energy coordinates parameterize the acceptance energy. The discriminator is used only during training.
<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Energy updates</td><td>80 updates; 256 base chains per update; approximately four selected decisions per chain on average; final iterate.</td></tr><tr><td>Decision selection</td><td>Inclusion probability  $\operatorname* { m i n } ( 1 , \lambda _ { j } h _ { i t } )$  with  $h _ { i t }$  an acceptance-gradient norm in rescaled coordinates and guidance weight three.  $\lambda _ { 0 } = 0 . 0 5 0 1 ;$  later scales  $\lambda _ { j }$  use the preceding update&#x27;s sensitivity sum.</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $0 . 0 2 ,$  five-update warm-up then linear decay to zero, moments (0.9, 0.999), no weight decay.</td></tr><tr><td>Coordinates</td><td>AdamW on  $( u ^ { \prime } , w )$  with  $u ^ { \prime } = \log ( e ^ { u } + 0 . 5 7 w ) ;$  gradient components divided by (0.676, 0.152) before clipping to norm one. After each update, clip  $u \in [ - 1 . 5 , \bar { 1 . 5 } ]$  and w  $\in [ - 1 , 3 ]$  , then raise w where necessary so that  $e ^ { u } + \dot { 0 } . 5 7 w \ge \dot { e } ^ { - 1 . 5 } .$ </td></tr><tr><td>Discriminator</td><td>Rank-64 LoRA adaptation of  $V _ { \mathrm { G } }$  on all 12 layers and the scalar head (Hu et al., 2022); approximately 9.7 million trainable parameters; logit  $- \beta _ { \mathrm { r e f } } ( V _ { \psi } - V _ { \mathrm { G } } )$  with fixed  $\beta _ { \mathrm { r e f } } \approx 1 7 . 6 2$ </td></tr><tr><td>Initial discriminator fit</td><td>25,600 generated examples; 1,000 logistic updates with 128 generated and 128 node-count-matched data graphs per update. AdamW learning rate  $2 \times 1 0 ^ { - 5 } .$  200-step warm-up, no weight decay, clipping to one.</td></tr><tr><td>Refitting</td><td>One discriminator step after each energy update, reusing the first 128 of the 256 base-chain terminal graphs generated before that update, together with 128 node-count-matched training graphs.</td></tr></table>

At $K = 2 5 0$ , discriminator negatives are calibrated GEM samples. Invalid negatives receive weight three in the logistic loss, selected from $\{ 1 , 3 , 9 , 2 7 \}$ by held-out classifier loss. At $K = 1 0 0 0$ , half the negatives are calibrated samples and half come from guidance settings $( c , \omega ) = ( 0 . 5 , 1 ) , ( 0 . 8 6 4 9 , 1 ) , ( 0 . 5 , 0 . 5 )$ ; the invalid weight is fixed to three. The $K = 1 2 5$ pool also mixes calibrated and guided samples. The three $K = 1 0 0 0$ discriminator seeds share their negative pool. Gradient clipping is active in about 85% of the K = 250 updates. The exact-state optimizer does not use these molecular clipping and coordinate settings.

## A.3 Alternative Estimators

A matched-transition experiment uses all six original exact targets, $K = 4 , 8 , 1 6$ , three fresh seeds, and the same 2,000-update schedule. Coupled learning uses 512 base chains; leave-one-out score-function learning uses 704, 736, and 752 chains, respectively, so that both simulate the same number of transitions. Both improve on the correct energy in all 18 settings. Coupled learning has lower median TV in 12 settings, with median diference −0.0015, but takes 4.66 times the CPU runtime. Learning rates were not separately tuned. A fixed-gradient diagnostic likewise establishes no general eficiency advantage over leave-one-out scores or paired finite diferences. These findings support terminal learning, not superiority of coupling.

## A.4 Computational Resources and Continuation Work

Molecular energy training uses NVIDIA H100 GPUs: energy updates take approximately 4.86 GPU-hours per seed at $K = 2 5 0$ and 16.3 at $K = 1 0 0 0$ , excluding discriminator preparation. At $K = 2 5 0$ , generation takes 0.859 H100 GPU-seconds per requested graph for calibrated GEM and 0.932 for our two-potential energy, excluding packed batches. Main comparisons match sampling steps, not elapsed time or runtime at equal quality. For fixed graph size and bounded retries, both samplers require work proportional to K beyond initialization.

Let $s$ contain the selected decisions. Simulating the N base chains and one full alternative continuation per selected decision uses at most

$$
N K + \sum _ { ( i , t ) \in \mathscr { S } } ( K - t - 1 )
$$

refinement transitions, excluding initialization, derivative evaluations, and discriminator work; earlier branch merging can lower this count. Branching at every step gives $N K + N K ( K - 1 ) / 2$ transitions; selecting an expected $F$ decisions per chain gives at most $N K + N F ( K - 1 )$ in expectation. With one uniformly selected step, as in the exact study, the expected count is $N ( 3 K - 1 ) / 2$ . These are counts of simulated transitions, not measured wall-clock ratios, and they do not bound mixing time. Molecular training targets about four decisions per chain. At deployment no continuations or discriminator evaluations are needed.

## B FINITE-STATE EXPERIMENTS

## B.1 Targets and Optimization

The finite-state study uses no external dataset and is separate from the molecular experiments: each target is a synthetic pairwise binary (Ising) model $p ( s ) \propto e ^ { - H ( s ) }$ over configurations $\boldsymbol s = ( s _ { 1 } , \ldots , s _ { L } )$ with $L = 6 { \mathrm { ~ o r ~ } } L = 8$ variables. Here $\begin{array} { r } { H ( s ) = - \sum _ { i } h _ { i } s _ { i } - \sum _ { i < j } J _ { i j } s _ { i } s _ { j } } \end{array}$ , with $s _ { i } \in \{ - 1 , 1 \} , h _ { i } \sim \mathcal { N } ( 0 , 0 . 2 5 ^ { 2 } )$ , and $J _ { i j } \sim \mathcal { N } ( 0 , 1 . 2 5 ^ { 2 } / L )$ the fields $h _ { i }$ bias individual variables, and the couplings $J _ { i j }$ create dependencies between them. The reference and new-target sets each contain three targets of each size, drawn independently from this distribution, each with its own fields and couplings. Target IDs such as $\mathtt { R e f - } 6 \mathtt { - } 1$ identify these draws by set, number of spins, and instance index; they do not denote methods, dificulty levels, or training seeds. Enumeration gives target probabilities, initial marginals, and transition laws. The initial law is the product of target marginals; proposals flip one uniformly chosen spin, and rejection supplies self transitions. The inverse temperature rises from 0.2 to 1 over the first $\lceil K / 2 \rceil$ steps, including both endpoints, and is then fixed.

A restricted unary-and-ring energy is fitted by population maximum likelihood. Its scalar multiplier minimizes terminal TV over 81 logarithmically spaced values from 0.05 to 20. The learned family adds all unary and pairwise features, with 21 or 36 coeficients. Population pseudo-likelihood recovers the correct energy H up to a constant (Besag, 1975). The discriminator is linear in those features and an intercept, initialized by the exact population logistic fit and subsequently updated from generated and target samples.

The original learning grid uses $K \in \{ 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 , 2 5 6 \}$ , trajectory counts $N \in \{ 3 2 , 1 2 8 , 5 1 2 \}$ , oracle or fitted discriminators, coupled or independent continuations, and three seeds. All runs use 2,000 Adam updates, learning rate 0.02 decaying to zero, moments (0.9, 0.999), no clipping or weight decay, and final-iterate evaluation. Each chain selects one uniformly chosen decision. The discriminator receives at most 128 generated and 128 target samples per update.

The numerical reference minimizes $D ( q \| p ) = 2 \mathrm { K L } ( \frac { 1 } { 2 } ( p + q ) \| p )$ over the learned family, with $q = q _ { \theta , K }$ . Consider the balanced, unweighted population problem of classifying samples of $p$ against samples of $q ,$ and let $d ^ { * }$ be its optimal unrestricted discriminator, so that $\sigma ( d ^ { * } ) = p / ( p + q )$ . Hold $d ^ { * }$ fixed at the current generator, and assume $p > 0$ wherever $q > 0$ , which holds here because every target $p \propto e ^ { - H }$ has full support. Using $\begin{array} { r } { \sum _ { x } \nabla _ { \theta } q ( x ) = 0 } \end{array}$

$$
\nabla _ { \theta } \mathbb { E } _ { q } [ \log \sigma ( d ^ { * } ) ] = \sum _ { x } \nabla _ { \theta } q ( x ) \log \frac { p ( x ) } { p ( x ) + q ( x ) } = - \nabla _ { \theta } D ( q \| p ) ,
$$

where $d ^ { * }$ is not diferentiated. The reward gradient for this ideal, frozen discriminator is therefore the negative gradient of $D .$ The relation describes the finite-state setting; it does not imply that the weighted, finitely trained molecular discriminator optimizes the same divergence. It is also separate from the estimator of Section 3.3, which is unbiased for $\nabla _ { \boldsymbol { \theta } } J$ with any fixed discriminator under the conditions stated there. The reference uses exact gradients, a coeficient box of [−20, 20], two starts, bounded L-BFGS-B, and projected-Adam refinement. It is a numerical reference rather than a certified global solution. Eight of 42 reported reference configurations fail the projected-gradient tolerance, mainly at acceptance kinks; additional minimizations and directional checks change the objective by at most about ${ \check { 7 } } . 3 \times 1 0 ^ { - 1 1 }$ . The TV recovery fraction divides the learned improvement over the calibrated restricted energy by the reference’s improvement. Because the reference minimizes $D ,$ not TV, recovery does not measure a known maximum TV gain.

## B.2 Repeated Training and Scalar Controls

Fresh-seed replication uses seeds 1001–1003 on the reference targets; new-target training uses seeds 3001–3003. Both follow the 512-chain, 2,000-update procedure fixed before these runs, without selecting final iterates by TV. Table 4 lists each target by its ID and horizon $K ,$ and reports our method’s median and sample SD across three training seeds. Its reference block reports the fresh-seed replication, whereas Table 1 and Figure 2 report the original runs; the replication’s cross-target medians therefore difer slightly from Table 1. The medians determine win counts. Horizons within a target are dependent, so these counts are descriptive.

The scalar control minimizes exact terminal TV for $\beta H$ over nonnegative scales, including zero and the infinitescale downhill-only limit. Logarithmic grids of 513 to 4,097 points are refined around detected minima until the best TV values change by less than $\bar { 1 } 0 ^ { - 9 }$ . The omitted large-β tail is controlled to $1 0 ^ { - 1 0 }$ using the smallest positive single-flip energy increase. This controls the tail, not global optimality inside the search interval. All 18 reference and 18 new-target searches stabilize. Learned medians beat tuned $\beta H$ in 14 of 18 reference settings in both the original runs and the fresh-seed replication, and in 15 of 18 new-target settings; the three new-target exceptions are all on New-8-2. Independent probability propagation verifies the terminal laws of these follow-up runs.

## B.3 Finite-Time and Equilibrium Distributions

Let the target be uniform on $\{ a , b \}$ , start at a, and propose a flip with probability $3 / 4 .$ . The correct energy accepts every flip, giving probability $3 / 4$ to b after one step. At inverse temperature one, setting $E ( b ) - E ( a ) = \log ( 3 / 2 )$ reduces acceptance to $2 / 3 { : }$ the one-step output is uniform, although the equilibrium probability of b is $2 / 5$

![](images/03408c2ce2f3526f3fc137f3fab4d369d7588dc5f0ded8d1070936aa49b9b951.jpg)  
Figure 4: Finite-time accuracy need not imply equilibrium accuracy. One-step output and equilibrium probability for the two-state example, computed exactly.

Table 4: Exact terminal total variation by target distribution and sampling horizon; lower is better. Each target is a synthetic pairwise binary model with random fields and couplings (Appendix B.1); no external dataset is used. Target IDs specify the set (Ref or New), number of spins (6 or 8), and instance index (1–3). Each instance has fixed fields and couplings; its three rows evaluate the same target at horizons K = 4, 8, 16, the number of MH transitions. The controls are the equilibrium-correct energy H and its numerically tuned scalar multiple βH; because H is exact at equilibrium, lower TV than H reflects a finite-horizon improvement. Learned columns report the median and sample standard deviation of terminal TV across three training seeds; TV itself is evaluated exactly, so the SD reflects only variation across training seeds. The reference block repeats training with fresh seeds on the original targets, whereas the new-target block trains from scratch on independently drawn target distributions to test whether the gains recur; no trained energy is transferred. Table 1 and Figure 2 use the original reference-target runs. All learned runs use N = 512 and 2,000 updates. Bold identifies the lowest of the three TV values using unrounded results.
<table><tr><td>Target ID</td><td>K</td><td>Correct H</td><td>Tuned βH</td><td>Learned median</td><td>Seed SD</td></tr><tr><td colspan="6">Reference targets: fresh-seed replication</td></tr><tr><td>Ref-6-1</td><td>4</td><td>0.1266</td><td>0.0613</td><td>0.0529</td><td>0.005</td></tr><tr><td>Ref-6-1</td><td>8</td><td>0.0845</td><td>0.0284</td><td>0.0275</td><td>0.004</td></tr><tr><td>Ref-6-1</td><td>16</td><td>0.0593</td><td>0.0279</td><td>0.0208</td><td>0.004</td></tr><tr><td>Ref-6-2</td><td>4</td><td>0.3435</td><td>0.2319</td><td>0.2284</td><td>0.001</td></tr><tr><td>Ref-6-2</td><td>8</td><td>0.2389</td><td>0.1247</td><td>0.0805</td><td>0.009</td></tr><tr><td>Ref-6-2</td><td>16</td><td>0.1234</td><td>0.0524</td><td>0.0359</td><td>0.005</td></tr><tr><td>Ref-6-3</td><td>4</td><td>0.3999</td><td>0.3126</td><td>0.3045</td><td>&lt; 0.001</td></tr><tr><td>Ref-6-3</td><td>8</td><td>0.2326</td><td>0.1041</td><td>0.0810</td><td>0.024</td></tr><tr><td>Ref-6-3</td><td>16</td><td>0.0838</td><td>0.0242</td><td>0.0759</td><td>0.007</td></tr><tr><td>Ref-8-1</td><td>4</td><td>0.2672</td><td>0.1803</td><td>0.1534</td><td>0.002</td></tr><tr><td>Ref-8-1</td><td>8</td><td>0.1549</td><td>0.0917</td><td>0.0435</td><td>0.003</td></tr><tr><td>Ref-8-1</td><td>16</td><td>0.0935</td><td>0.0633</td><td>0.0247</td><td>0.004</td></tr><tr><td>Ref-8-2</td><td>4</td><td>0.3142</td><td>0.2235</td><td>0.2095</td><td>&lt; 0.001</td></tr><tr><td>Ref-8-2</td><td>8</td><td>0.1686</td><td>0.0914</td><td>0.0496</td><td>0.003</td></tr><tr><td>Ref-8-2</td><td>16</td><td>0.0612</td><td>0.0369</td><td>0.0318</td><td>0.011</td></tr><tr><td>Ref-8-3</td><td>4</td><td>0.3701</td><td>0.1807</td><td>0.1844</td><td>&lt; 0.001</td></tr><tr><td>Ref-8-3</td><td>8</td><td>0.3012</td><td>0.0939</td><td>0.1174</td><td>&lt; 0.001</td></tr><tr><td>Ref-8-3</td><td>16</td><td>0.2198</td><td>0.0636</td><td>0.0923</td><td>0.007</td></tr><tr><td colspan="6">New targets: separately trained</td></tr><tr><td>New-6-1</td><td>4</td><td>0.6815</td><td>0.6149</td><td>0.6106</td><td></td></tr><tr><td>New-6-1</td><td>8</td><td>0.4960</td><td>0.3693</td><td>0.3459</td><td>&lt; 0.001</td></tr><tr><td>New-6-1</td><td>16</td><td>0.2572</td><td>0.1132</td><td>0.0746</td><td>0.002</td></tr><tr><td>New-6-2</td><td>4</td><td>0.1657</td><td>0.0762</td><td>0.0434</td><td>0.008</td></tr><tr><td>New-6-2</td><td>8</td><td>0.0977</td><td>0.0469</td><td>0.0338</td><td>0.002</td></tr><tr><td>New-6-2</td><td>16</td><td>0.0658</td><td>0.0448</td><td>0.0160</td><td>0.002 0.005</td></tr><tr><td>New-6-3</td><td>4</td><td>0.4313</td><td>0.3193</td><td>0.2996</td><td></td></tr><tr><td>New-6-3</td><td>8</td><td>0.2894</td><td>0.1612</td><td>0.1241</td><td>0.002</td></tr><tr><td>New-6-3</td><td>16</td><td>0.1416</td><td>0.0432</td><td>0.0319</td><td>0.005 0.002</td></tr><tr><td>New-8-1</td><td>4</td><td>0.5173</td><td>0.4715</td><td>0.4711</td><td>&lt; 0.001</td></tr><tr><td>New-8-1</td><td>8</td><td>0.3651</td><td>0.3172</td><td>0.2677</td><td>&lt; 0.001</td></tr><tr><td>New-8-1</td><td>16</td><td>0.2224</td><td>0.1898</td><td>0.0643</td><td>0.002</td></tr><tr><td>New-8-2</td><td>4</td><td>0.5886</td><td>0.5116</td><td>0.5655</td><td>0.002</td></tr><tr><td>New-8-2</td><td>8</td><td>0.4601</td><td>0.3307</td><td>0.4097</td><td>0.002</td></tr><tr><td>New-8-2</td><td>16</td><td>0.2856</td><td>0.1180</td><td>0.2553</td><td>0.002</td></tr><tr><td>New-8-3</td><td>4</td><td>0.6175</td><td>0.5830</td><td>0.5716</td><td>&lt; 0.001</td></tr><tr><td>New-8-3</td><td>8</td><td>0.4878</td><td>0.4123</td><td>0.3664</td><td>0.003</td></tr><tr><td>New-8-3</td><td>16</td><td>0.3019</td><td>0.1852</td><td>0.0968</td><td>0.002</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## C MOLECULAR EVALUATION

## C.1 Sampler, Calibration, and Metrics

Each chain draws 8–27 nodes according to the MOSES training histogram and keeps this count fixed. Initialization performs at most 225 greedy edits under $V _ { \mathrm { G } }$ . We retain the released proposal settings, approximately $\beta _ { \mathrm { p r o p } } = 2 . 3 6$ $\lambda _ { X } = 4 . 0 0$ , and $\lambda _ { E } = 4 . 8 3$ . If no label changes, the sampler retries after multiplying these settings by 0.9, for at most 20 stages. Acceptance uses an inverse-temperature ramp from approximately 0.78 to 17.62 over 500 steps, then holds the final value. Continuations retain their original step index. Values here are rounded for readability; full-precision proposal and acceptance-temperature constants are provided in molecular.py in the supplementary code, and its YAML file specifies experiment settings. The molecular example in the supplement implements the training method but does not reconstruct the original development, selection, and final evaluation pipeline.

Exact MH marginalizes both forward and reverse proposal probabilities over the same 20 stages. GEM’s released rule instead uses the realized forward retry path and a reverse marginal with up to 256 stages and log-mass floor −50; it is not exact MH for the same target. Both rules have parameter-independent proposal terms. Network calls are padded to 256 graphs; sampler precision follows the release, while discriminator scoring and refitting use the highest float32 precision.

Development FCD, used for baseline calibration, discriminator screening, and static-coeficient selection, compares 2,048 generated graphs per candidate setting, from a generation stream separate from the final comparisons, with a fixed reference of 16,010 molecules: the MOSES training molecules whose SHA-256 hash of the canonica SMILES is divisible by 100 (about 1% of the training set). Final FCD uses the separate MOSES scafold-test reference of 176,225 molecules. The development molecules were not held out from the pretraining of the released GEM checkpoints. The following calibration scales are rounded to three decimal places. Exact GEM selects 0.7 at $K = 2 5 0$ from {0.25, 0.35, 0.5, 0.7, 0.865, 1, 1.5, 2, 3}, a grid extended after an initial boundary solution, and 0.865 at $K = 1 0 0 0$ from $\{ 0 . 5 , 0 . 8 6 5 , 1 \}$ . At K = 125, the selected 0.7 nearly ties 0.865 in development FCD. Released-rule GEM at $K = 2 5 0$ selects 0.5 over 0.7 and 1; the first two difer by only 0.0013 in FCD but substantially in validity. The released baselines at $K = 1 2 5$ , 1000 use scale one, not a budget-calibrated scale.

Every requested graph counts in the denominators of validity and VUN, and failed outputs are not replaced. Native validity permits disconnected sanitizable molecules; native FCD uses their canonical largest fragments, retaining duplicates. Strict validity requires connectedness and the stated canonical representation checks. Novelty uses the unfiltered training set of 1,584,663 canonical SMILES. Software is Python 3.10.21, PyTorch 2.5.1 with CUDA 12.1, RDKit 2026.03.6, fcd\_torch 1.0.7, NumPy 2.2.6, and SciPy 1.15.2. Scoring runs on CPUs. The graph encoding cannot represent every aromatic-hydrogen configuration, so these scores are interpreted within the same encoding and evaluator rather than against published leaderboard values.

## C.2 Released-Rule Deployment and Uncertainty

Table 5: Deployment with GEM’s released acceptance rule. “GEM + learned energy $( \mathrm { O u r s } ) ^ { \dag }$ reuses the energies from Table 2 without retraining, with released-rule acceptance replacing exact MH. Metrics and denominators match Table 2: FCD, validity (%), and VUN (%), with 25,000 requested graphs per seed. Entries are means ± sample SD over two prespecified seeds at $K = 1 2 5$ and three elsewhere. Baselines are uncalibrated released GEM at $K = 1 2 5$ , 1000 and calibrated GEM at $K = 2 5 0$ . Bold marks the better mean within each pair, not statistical significance.
<table><tr><td> $K$ </td><td>Method</td><td>FCD↓</td><td>Validity ↑</td><td>VUN↑</td></tr><tr><td rowspan="2">125</td><td>GEM (released)</td><td> $2 . 4 0 6 \pm 0 . 0 0 0 3$ </td><td> ${ \bf 9 3 . 2 5 \pm 0 . 0 5 }$ </td><td> ${ \bf 9 0 . 9 9 \pm 0 . 0 6 }$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 2 . 2 9 5 \pm 0 . 0 1 1 }$ </td><td> $9 2 . 4 5 \pm 0 . 4 4$ </td><td> $9 0 . 4 3 \pm 0 . 4 4$ </td></tr><tr><td rowspan="2">250</td><td>GEM (calibrated)</td><td> $1 . 9 2 0 \pm 0 . 0 1 2$ </td><td> $9 3 . 4 8 \pm 0 . 0 9$ </td><td> $9 1 . 2 0 \pm 0 . 1 0$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 1 . 7 8 4 \pm 0 . 0 3 3 }$ </td><td> ${ \bf 9 5 . 0 2 \pm 0 . 4 2 }$ </td><td> ${ \bf 9 2 . 9 7 \pm 0 . 4 1 }$ </td></tr><tr><td rowspan="2">1000</td><td>GEM (released)</td><td> $1 . 5 2 9 \pm 0 . 0 1 3$ </td><td> ${ \bf 9 8 . 0 2 \pm 0 . 0 3 }$ </td><td> $9 4 . 0 2 \pm 0 . 2 2$ </td></tr><tr><td>GEM + learned energy (Ours)</td><td> $\mathbf { 1 . 4 8 7 \pm 0 . 0 2 2 }$ </td><td> $9 7 . 9 5 \pm 0 . 0 8$ </td><td> ${ \bf 9 4 . 2 1 \pm 0 . 1 1 }$ </td></tr></table>

The internally prespecified decision rule combines a one-sided paired FCD test at level 0.05 with one-sided 95% lower confidence limits above −1 percentage point for the validity and VUN diferences, each computed as learned minus baseline. The exact-rule comparison meets it at $K = 1 2 5 , 1 0 0 0 ;$ at $K = 2 5 0$ , the validity lower limit of −1.25 points does not establish non-inferiority. Released-rule deployment meets it at K = 250, 1000, not at $K = 1 2 5$ , where the validity and VUN lower limits are −2.52 and −2.80 points. A later three-seed analysis also fails those conditions and does not replace the prespecified two-seed result. No multiplicity adjustment is applied across the six comparisons.

Training seed k is paired with generation seed 1911 + k. The two or three seed-level diferences, not individual graphs, are the observations for the small-sample t tests. Energies are reused across acceptance rules, and some discriminator data are shared.

Learned rows use H100 GPUs. One released-GEM K = 1000 comparator seed mixes 63 H100 and 35 L40-class batches; the two H100-only seeds give an unresolved FCD comparison $( p = 0 . 1 1 )$ . All K = 125 rows use H100s. Runtime estimates exclude packed batches.

## C.3 Protocol Changes and Sensitivity

Protocols were recorded internally, not in an external registry. The original K = 250 comparison followed static development results and exploratory pilots but preceded final learned outcomes. Its released-rule hypothesis followed development results, and later budgets followed earlier tests. The six comparisons are therefore not independent prospective replications. No final test score selected the deployed iterate.

At K = 125, a baseline-only encoding check found 22 mismatches among 1,833 native-valid graphs (1.20%). The discriminator-pool eligibility threshold was raised from 1% to 2% before any pool output or discriminator training. The completed pool contained 368 mismatches among 22,270 valid graphs (1.65%): 322 non-isomorphic re-encodings and 46 explicit-hydrogen cases. The change afected eligibility, not evaluation or sample handling The original rule would have stopped the experiment.

Strict scoring preserves the exact-rule K = 125 improvement: validity and VUN gains are 4.77 and 4.65 points. About 0.64 points of the native validity gain arise from disconnected output. Matching the valid-input counts changes the FCD diference from −0.1419 to −0.1379; all fixed subsample comparisons remain negative. These checks reuse generated samples and do not establish that encoding mismatches in the discriminator’s training pool are harmless. The released-rule K = 125 joint failure also remains under strict scoring.

At K = 1000, the initial discriminator failed a screen relating reward to development FCD. A revised negative pool containing guided samples passed at the correlation boundary of −0.50. Several screened settings also supplied negatives, and all three discriminator seeds shared the pool. At K = 500, the screen failed with correlation approximately −0.071; no energy was trained and no second attempt was made. The absence of that budget is not an evaluated negative generator result.

## C.4 Scope of the GEM Comparison

Coeficient-selection controls give FCD 2.280 at K = 125, versus 2.366 for learning, but lower validity (91.10% versus 92.55%); the FCD diference is unresolved with three seeds. At $K = 2 5 0$ , their FCD is 1.811 versus 1.829 with exact acceptance, and 1.769 versus 1.784 with released acceptance; rate diferences are small. Finite-diference training gives exact-rule FCD 1.821 under a diferent optimization schedule. These comparisons do not establish an FCD advantage for learning over the alternatives; we report only these summaries.

The K = 125 static setting was selected among four development candidates. The K = 250 setting has the lowest development FCD among tested settings but is validity-ineligible: 94.97% against a 95.44% threshold. These descriptive controls use three seeds with shared generation streams; there is no corresponding three-seed static evaluation at $K = 1 0 0 0$

The six-setting coeficient grid costs 3.2 H100-hours at $K = 2 5 0$ , versus 4.86 per seed for energy updates before discriminator preparation. Schedule compression is inconclusive, and cross-budget transfer is confounded by the change of scale. A larger residual energy did not improve development FCD at matched scales; its planned tests and a random-sign placebo test were not completed. Molecular-weight distance worsens at $K = 1 0 0 0$ despite improved FCD, precluding a claim of uniformly improved molecular properties.