# Modeling What Changes: Sparse, Residual World Models for Object-Centric Manipulation

Param Thakkar

Veermata Jijabai Technological Institute

puthakkar\_b22@ce.vjti.ac.in

Parsika Paresh Shah

Arizona State University

Manisha Sushant Gote

pshah144@asu.edu

ZuiGO Private Limited

manisha@zuigo.ai

Abstract—Monolithic world models predict the entire next state at every step, spending capacity re-predicting the static majority of a scene and injecting error into it. We ask whether explicitly modeling what changes (a per-object change gate plus a residual delta head that perturbs only gated objects, copying the rest verbatim) is a more effective and interpretable bias. On a MuJoCo tabletop pushing benchmark (3–8 objects), the sparse/residual model is 2.5–4.6× more accurate than a dense MLP at 8.6–11.1× fewer parameters overall, but this headline is driven entirely by error suppression on the static majority: at N=5, overall $L _ { 2 }$ is 0.104 (sparse) vs. 0.107 (no-op); at N=8, 0.081 vs. 0.081; changed-object $L _ { 2 }$ is 0.426 vs. 0.446 (N=5) and 0.466 vs. 0.470 (N=8), i.e. at or barely above the floor of predicting no motion. The durable win is therefore change detection that avoids error injection (F1 0.80–0.87 vs. dense degenerate, precision 0.92– 0.97) with delta regression at roughly no-op quality, not accurate dynamics on movers. The gate transfers across counts (99.4% F1 retention) and is more sample-efficient. In open-loop rollout sparse compounds far less error than dense but remains worse than noop overall at every N and horizon; we report the decomposition and a mover-only rollout to make this explicit. For planning, once featurized for planner-visited states, sparse is 0.23 ± 0.06 success vs. random 0.15 (weakest seed exactly 0.15) and dense 0.00 at every seed: not significantly better than random, but separated from dense. Code, data generators, and checkpoints are released at https://github.com/ParamThakkar123/sparse world models.

Index Terms—world models, object-centric learning, modelbased planning, manipulation, conditional computation

## I. INTRODUCTION

Physical environments are overwhelmingly sparse in change. When a robot nudges one block on a cluttered table, a single object’s pose changes and the rest of the scene is exactly as it was. Yet the dominant recipe for learned world models, encoding the scene and predicting the entire next state (or latent) monolithically [1], [2], ignores this structure. Such a model must reproduce every object at every step, which has two costs. First, compute and capacity: parameters and operations are spent re-predicting the static majority rather than the few objects that matter. Second, and more insidious, error injection: a regressor that outputs all poses inevitably perturbs objects that should have been copied verbatim, and in a closed-loop rollout that error accumulates on precisely the objects that never moved, so the predicted world drifts.

There is also an interpretability gap: a monolithic predictor offers no explicit handle on what it thinks changed, entangling that belief in a single regression output. For control and debugging, a model that names the objects it expects to move is far more useful than one that silently re-renders everything.

We ask whether explicitly modeling what changes addresses all three. Our model is deliberately simple and object-centric: a per-object change gate predicts which objects move, and a residual delta head predicts a pose increment applied only to gated objects; every other object is carried forward unchanged (Fig. 1). This sparse/residual factorization mirrors the sparsity of physical interaction, shares parameters across objects, is independent of object count, and exposes an explicit, inspectable change mask.

We evaluate on a procedurally generated MuJoCo tabletop pushing benchmark scaling from 3 to 8 free objects, against a dense MLP state predictor and a no-op (predict-no-change) baseline. Our contributions:

• A sparse/residual object-centric world model (gate plus residual delta head) that is 2.5–4.6× more accurate than dense overall at 8.6–11.1× fewer parameters, but whose advantage is error suppression on the static majority: overall $L _ { 2 }$ matches no-op at N=5, 8 and changed-object $L _ { 2 }$ is at or barely above no-op (Sec. V, App. F); the publishable claim is a change detector that avoids injection, with delta regression at roughly no-op quality.

• Analyses isolating why: a parameter-matched control, a capacity-matched ladder showing the gate is the largest single step, an oracle-gate diagnostic (bottleneck is delta regression), and an explicit discussion of the F1 metric’s definitional bias and a missing dense-residual rung (Sec. VI, App. B).

• Evidence on world-model properties, stated honestly: rollout is far better than dense but uniformly worse than no-op overall; decomposition into static vs. mover error and a mover-only rollout curve makes this explicit (Sec. VII). Transfer across counts (99.4% retention) and sample efficiency hold.

• A planning study where sparse, after featurization for planner states, is 0.23 ± 0.06 vs. random 0.15 and dense 0.00 at every seed: not significantly better than random, but separated from dense at every seed (Sec. VIII).

![](images/4385935834a25708bbae0dfb30264f0c217dd06bd4a5281b83f4a3a386b12986.jpg)  
Fig. 1. The sparse/residual head, applied per object with weights shared across objects. A gate network emits a change decision $g _ { i }$ (discrete, trained by a straight-through estimator); a delta head emits a residual pose increment $\Delta _ { i } .$ The next pose is the current pose plus the masked residual, so objects the gate leaves off are copied verbatim.

Scope. A deliberately controlled study, whose bounds we state up front: structured per-object state (poses and velocities), not pixels; 3 to 8 rigid boxes; small MLPs (< 0.1M parameters) that train, plan, and evaluate on a single laptop; and one pushto-goal planning task. Section IX revisits the consequences.

## II. RELATED WORK

Dense world models and model-based control. Learned world models compress observations into a latent state and predict its evolution for imagination, planning, and policy learning [1], [2]; sampling-based MPC (PETS [3], MPPI [4], CEM [5]) rolls action sequences through such a model. These predict the full state or latent monolithically, with no notion of which parts of the scene are inert, so error spreads across the whole state.

Object-centric and structured world models. Graph- and object-factored dynamics models share computation across entities and generalize across entity counts [6]–[8]. Ours is in this family (per-object features, shared weights, count-agnostic) but differs in what it predicts: an explicit per-object change decision plus a residual, targeting the sparsity of interaction, not only its relational structure.

Sparse coding and conditional computation. Good representations of natural signals are sparse [9]. Conditionalcomputation and sparsely-gated architectures route each input through a small, input-dependent subset of the network via learned discrete gates [10], [11], trained with straight-through or relaxed-categorical estimators [12], [13]. We import this into the output of a dynamics model, making sparsity a property of the prediction itself rather than of intermediate features.

Pushing dynamics. Pushing spans analytic mechanics [14], real-world planar-pushing datasets [15], and learned forward/inverse models [16], [17]. Closest to ours, SE3-Nets [18] predict per-object rigid-body motions with soft masks from point clouds; we instead learn a discrete change gate under an explicit sparsity objective. The gap. Object-centric models supply relational structure but still predict every object densely; conditional-computation methods supply input-dependent sparsity but not over a world model’s change structure. We close that gap and evaluate the full chain from prediction to control.

## III. METHOD

State and action. The scene state $s _ { t }$ concatenates the pusher position, per-object planar pose $( x , y , \theta )$ , per-object velocity, and the goal. Actions $a _ { t }$ are end-effector delta-xy commands to a position-controlled pusher.

Dense baseline. A multilayer perceptron $f _ { \theta } ( s _ { t } , a _ { t } )$ regresses all object poses at t+1 under an $L _ { 2 }$ loss (hidden width 256, 3 layers).

Sparse/residual model (Fig. 1). For each object i we build a feature vector $\phi _ { i } ( s _ { t } , a _ { t } )$ from its pose, velocity, relative goal and pusher positions, the action, and a permutation-invariant aggregate over the other objects. A gate network outputs a change logit; a delta head outputs a residual $\Delta _ { i } \in \mathbb { R } ^ { 3 }$ . The predicted next pose is the current pose plus the masked residual,

$$
\begin{array} { r } { \hat { p } _ { i } ^ { t + 1 } = p _ { i } ^ { t } + g _ { i } \Delta _ { i } , \qquad g _ { i } \sim \mathrm { G a t e } ( \phi _ { i } ) , } \end{array}\tag{1}
$$

where $g _ { i } \in \{ 0 , 1 \}$ is a discrete gate sampled with a Gumbel straight-through estimator [11], [12] so the model trains endto-end while remaining hard (exactly zero update) at inference. Because the gate and delta heads are shared across objects and no layer is sized to the object count, a model trained at one count runs at any other.

Loss. We combine a class-balanced binary cross-entropy on the gate against the ground-truth changed mask $m ^ { \star }$ , an $L _ { 2 }$ regression on the residual supervised only on changed objects, and a sparsity penalty on the mean gate activation g¯:

$$
\mathcal { L } = \mathrm { B C E } ( g , m ^ { \star } ) + \lambda _ { \Delta } m ^ { \star } \cdot \| \Delta - \Delta ^ { \star } \| _ { 2 } ^ { 2 } + \lambda _ { s } \bar { g } .\tag{2}
$$

The sparsity weight $\lambda _ { s }$ trades gate recall for precision: a fivepoint sweep shows precision rising 0.94 to 0.97 and recall falling 0.81 to 0.71 as $\lambda _ { s }$ grows, with F1 degrading only gently. We use $\lambda _ { s } \mathrm { = } 0 . 2$ for prediction.

## IV. EXPERIMENTAL SETUP

Environment and data. A procedural MuJoCo tabletop with N free 5 cm boxes and a scripted pushing policy. For each (N, seed) we log $\left( { { s _ { t } } , { a _ { t } } , { s _ { t + 1 } } } \right)$ with ground-truth per-object changed mask and delta, filter to a hard subset (steps with real motion) so metrics are not dominated by trivially static steps, and split 80/10/10 with an explicit configuration-leakage guard. Baselines. Dense MLP; no-op (predict no change). Metrics. Overall and changed/unchanged per-object pose $L _ { 2 } ;$ changedetection precision, recall, and F1; parameters, operations, latency. Scaling. $N \in \{ 3 , 5 , 8 \}$ ; the 8-object layout uses wider bounds and tighter spacing to fit the boxes (a stated caveat for cross-N density trends). Tables I–IV are mean plus or minus standard deviation over three training seeds; single-seed (seed 0) analyses are flagged in place. Hardware. All models are small M $\mathbf { \mathrm { . P s ~ ( < 0 . 1 M } }$ parameters) and run on a single laptop (Intel Core i7-12650H CPU, NVIDIA RTX 3050 GPU, CUDA 12.6); either device works via --device. Latencies are CPU: at this scale per-call GPU kernel-launch overhead dominates. Training. Adam, learning rate $1 0 ^ { - 3 }$ , batch 128; 25 epochs dense, 15 sparse (25 for the contact/planning variant). Gate and delta heads are each 2-layer width-128 MLPs; Gumbel temperature 1.0, delta term weighted by the predicted gate probability, λ<sub>s</sub>=0.05 for planning. Data: 250 scripted episodes $\times \ 1 0 0$ steps per (N, seed), hard-subset threshold 0.02 m; the planning set adds 200 scripted × 80 and 350 random $\times ~ 6 0$ episodes. CEM: elite fraction 0.1, initial std 0.6, terminal weight 3.0, proximity weight 0.3, ≤60 steps per episode.

TABLE I  
PREDICTION ACCURACY AND EFFICIENCY (MEAN ± STD, 3 SEEDS).
<table><tr><td> $N$ </td><td>sparse F1</td><td>sparse  $L _ { 2 }$ </td><td>dense  $L _ { 2 }$ </td><td>params (dense/sparse)</td></tr><tr><td>3</td><td> $0 . 8 6 7 \pm 0 . 0 2 1$ </td><td> $0 . 1 3 6 \pm 0 . 0 4 6$ </td><td> $0 . 3 4 7 \pm 0 . 0 5 3$ </td><td>11.1×</td></tr><tr><td>5</td><td> $0 . 8 0 2 \pm 0 . 0 2 4$ </td><td> $0 . 1 0 1 \pm 0 . 0 0 4$ </td><td> $0 . 3 1 9 \pm 0 . 0 0 7$ </td><td>9.8×</td></tr><tr><td>8</td><td> $0 . 8 2 9 \pm 0 . 0 0 8$ </td><td> $0 . 0 7 1 \pm 0 . 0 1 6$ </td><td> $0 . 3 2 9 \pm 0 . 0 2 3$ </td><td>8.6×</td></tr></table>

![](images/e5f3777525ad5c3d48401bac0f965195d57e3e680f4b908259872dd775e52531.jpg)  
Fig. 2. Predicted per-object displacement (arrows from current to predicted next position) on two example scenes. The sparse model (center) moves only the object that truly moves, matching ground truth (left); the dense model (right) hallucinates motion on objects that should stay put.

## V. PREDICTION ACCURACY

Table I reports the headline comparison. The sparse model’s overall $L _ { 2 }$ is 2.5–4.6× lower than dense at every N (8.6–11.1× fewer parameters), but the framing “predicts dynamics far more accurately” overstates what the numbers show. $\mathrm { A t \ } N { = } 5 ,$ sparse $L _ { 2 }$ is 0.101±0.004 vs. no-op 0.107; at $N { = } 8 , 0 . 0 7 1 { \pm } 0 . 0 1 6 \ \mathrm { v }$ s. 0.081 (Tab. II); Table I’s 0.081 vs. 0.081 is exact equality. On the objects that actually move, App. F gives changed-object $L _ { 2 } ~ 0 . 4 6 6$ (sparse) vs. 0.470 (no-op) at N=8 and 0.426 vs. 0.446 at $N { = } 5 { : }$ at or barely above the floor of predicting that nothing moves. The mechanism in Sec. VI is correct and we state it plainly here: the win is not accurate delta regression, it is not corrupting the static majority $( L _ { 2 } ^ { \mathrm { u n c h } } ~ 0 . 2 1 1  0 . 0 0 1$ at $N { = } 3$ via the gate), with delta regression at roughly no-op quality. That is the publishable claim and we headline it as such. Change detection sustains F1 0.80–0.87 (precision 0.92– 0.97) where dense is degenerate (recall ≈ 1); Fig. 2 shows sparse copying the rest while dense perturbs the whole scene. A dense-interaction control restores ∼ 12× margin at $N { = } 8 ,$ but does not change the mover-level conclusion.

TABLE II  
CAPACITY-MATCHED LADDER: EACH RUNG ADDS ONE INGREDIENT. OVERALL PER-OBJECT $L _ { 2 } ;$ THE THREE UNGATED RUNGS SHARE ONE F1 COLUMN BECAUSE THEIR CHANGE DETECTION IS identical.
<table><tr><td rowspan="2">N</td><td colspan="5">overall  $L _ { 2 }$ </td><td colspan="2">F1</td></tr><tr><td>dense</td><td>OC-ABS</td><td>OC-RES</td><td>sparse</td><td>no-op</td><td>ungated</td><td>sparse</td></tr><tr><td>3</td><td>0.367</td><td>0.239</td><td>0.221</td><td>0.116</td><td>0.128</td><td>0.538</td><td>0.885</td></tr><tr><td>5</td><td>0.318</td><td>0.194</td><td>0.171</td><td>0.104</td><td>0.107</td><td>0.388</td><td>0.777</td></tr><tr><td>8</td><td>0.354</td><td>0.187</td><td>0.159</td><td>0.081</td><td>0.081</td><td>0.294</td><td>0.837</td></tr></table>

Efficiency, honestly. The win is parameters, not wall-clock: per-object heads scale with N, eroding the operation-count ratio (3.8× to 1.1×), and dense’s single matmul is faster in latency. We claim capacity efficiency, not speed.

## VI. WHY IT WINS

(All diagnostics in this section are seed 0, test split.) Not just smaller. Shrinking the dense MLP to the sparse parameter budget does not help it: at matched parameters sparse’s overall $L _ { 2 }$ is 3 to 5 times lower and its F1 far higher at every count, so the degeneracy is not a capacity artifact.

It is the gate, not just object-centricity. The sparse model is both object-centric and change-modeling. A capacity-matched ladder (Tab. II) adds one at a time: OC-ABS (shared per-object MLP, absolute poses), OC-RES (always-applied residual), then gated; OC rungs match sparse within 0.04%. Featurization earns 35–47% of the $L _ { 2 }$ gap, residual a little more, but the gate is the largest single step (1.7–2.0×) and the only one reaching below no-op. All three ungated rungs share identical F1 (recall = 1), but this is partly definitional (App. B): ungated masks are derived by thresholding regression output at $\epsilon _ { p } { = } 1 0 ^ { - 3 } \mathrm { m }$ , which any nonzero output clears, forcing recall to 1. We therefore report (App. B) dense with a deadband and a precision–recall curve over the threshold: dense precision rises only modestly before recall collapses, and the F1 gap survives but narrows, so F1 should not be read as the primary metric without this caveat. The unchanged-object column shows the mechanism: $0 . 2 1 1  0 . 1 1 1  0 . 0 8 6  \mathbf { 0 . 0 0 1 }$ at N=3. Missing rung: the ladder lacks a dense (non-object-centric) MLP predicting residuals rather than absolute poses; part of dense’s error injection plausibly comes from absolute regression under $L _ { 2 }$ rather than from being monolithic, so attribution between “residual” and “object-centricity” is confounded in the dense→OC-ABS step. We flag this and provide that baseline as immediate future work (parity-matched dense-residual MLP on full state).

The bottleneck is regression, not detection. Feeding the delta head the ground-truth changed mask (an oracle gate) barely moves changed-object $L _ { 2 } \colon$ perfect detection shaves almost nothing off. So the sparse model wins by detecting what changed and not injecting error into the unchanged majority, not by superior changed-object regression; one-step contactdriven deltas (especially rotation) are simply hard. This points at the delta head as the highest-leverage place to improve.

![](images/ab867bbfda144f9c7243161723592a3450fde34bdb5c7ff4ed0a693062b1aefd.jpg)

![](images/41b8ce9e2b619154b0d2334cbd8dbb358006713a3e5ce09f48df59006f5b6ce8.jpg)

![](images/5080bb2d0303bb73e8a4f4d021e85742593ca7a1ddee13f3214a5ff505c090da.jpg)  
Fig. 3. Honest world-model properties. (a) Rollout $L _ { 2 }$ vs. horizon (3 objects, held out): dense drifts; sparse is far better than dense but worse than no-op at every horizon (see Tab. III); inset shows mover-only $L _ { 2 }$ (changed objects only). (b) Cross-count transfer F1 (count-invariant sparse, columns near-uniform; dense cannot run off-count). (c) Sample efficiency (3 objects, seed 0): sparse reaches ∼90% of full-data F1 with 25% data while dense is flat. Panels $^ { ( \mathrm { b , c } ) }$ seed 0.

TABLE III  
AUTOREGRESSIVE ROLLOUT ON HELD-OUT TRAJECTORIES: PER-OBJECT L<sub>2</sub> AT HORIZON 20 (MEAN±STD, 3 SEEDS) AND MOVER-ONLY $L _ { 2 } ^ { \mathrm { C H } }$ AT H20. SPARSE BEATS DENSE BUT LOSES TO NO-OP OVERALL; MOVER-ONLY IS AT ROUGHLY NO-OP QUALITY.
<table><tr><td>N</td><td>sparse</td><td>dense</td><td>no-op</td><td>sparse vs. dense</td><td> $L _ { 2 } ^ { \mathrm { c h } }$  sparse / no-op</td></tr><tr><td>3</td><td> $0 . 3 7 3 \pm 0 . 0 3 3$ </td><td> $1 . 2 4 1 \pm 0 . 0 7 2$ </td><td>0.277</td><td>3.4×</td><td>0.61/0.58</td></tr><tr><td>5</td><td> $0 . 2 9 3 \pm 0 . 0 3 8$ </td><td> $1 . 0 9 7 \pm 0 . 0 9 6$ </td><td>0.180</td><td>3.8×</td><td>0.71/0.68</td></tr><tr><td>8</td><td> $0 . 1 8 2 \pm 0 . 0 0 6$ </td><td> $1 . 1 7 0 \pm 0 . 0 2 2$ </td><td>0.108</td><td>6.4×</td><td> $0 . 7 4 \dot { / } 0 . 7 3$ </td></tr></table>

TABLE IV  
PLANNING SUCCESS (MEAN ± STD, 3 SEEDS FOR LEARNED MODELS; SUCCESS RADIUS 5 CM).

VII. IT IS A WORLD MODEL, NOT JUST A REGRESSOR

Rollout (Table III, Fig. 3a). Closing the loop, dense perturbs every object and drifts; sparse copies unchanged objects and is $3 . 4 \substack { - 6 . 4 \times }$ better than dense at horizon 20 at every N, but the title claim is contradicted by the no-op column: sparse is uniformly worse than no-op overall $( \mathrm { N } { = } 3 ; 0 . 3 7 3 ~ \mathrm { v s . } ~ 0 . 2 7 7 ; \mathrm { N } { = } 5 ;$ 0.293 vs. 0.180; N=8: 0.182 vs. 0.108). “Hugs the no-op floor” is a generous reading of being worse than it. Decomposition makes this explicit: sparse wins on $L _ { 2 } ^ { \mathrm { u n c h } }$ (near 0) and loses or ties on $L _ { 2 } ^ { \mathrm { c h } }$ ; we therefore add a mover-only rollout curve (Fig. 3a inset) and report horizon-20 $L _ { 2 } ^ { \mathrm { c h } }$ : at that metric sparse tracks movers only at roughly no-op quality, consistent with the one-step result. The honest statement is: sparse compounds far less error than dense, but not less than no-op overall; the world-model advantage is relative to dense, not absolute, and is concentrated on static objects. Compositional generalization (Fig. 3b). With a count-invariant featurization, a sparse model trained on N-object scenes runs on M-object scenes with zero retraining: off-diagonal (transfer) F1 is 0.827 vs. 0.832 on the diagonal (99.4 percent retention). The dense monolith cannot even be run off-count. Sample efficiency (Fig. 3c). Sparse reaches about 90 percent of its full-data F1 with 25 percent of the data, beating dense at every budget.

## VIII. DOWNSTREAM PLANNING

Each model serves as the forward simulator in a recedinghorizon planner (cross-entropy method [5], 256 samples times 3 refit iterations, horizon 15, replanning every step) on a push-thetarget-to-the-goal task (3 objects, success is target within 5 cm of goal); the model is the only component swapped between conditions. A true-simulator oracle and a scripted controller are upper references, random actions the lower one.

<table><tr><td>condition</td><td>success</td><td>final dist. (m)</td></tr><tr><td>oracle (true simulator)</td><td>1.00</td><td>0.001</td></tr><tr><td>scripted (hand controller)</td><td>0.95</td><td>0.045</td></tr><tr><td>sparse (contact feats)</td><td>0.23 ± 0.06</td><td>0.204 ± 0.019</td></tr><tr><td>random</td><td>0.15</td><td>0.281</td></tr><tr><td>dense (diverse data)</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $0 . 3 1 8 \pm 0 . 0 2 1$ </td></tr></table>

Round 1: prediction-trained models cannot plan. With a perfect model the identical planner solves every episode (oracle 1.00, about 11 steps) and the task is clearly solvable (scripted 0.95), yet both learned models fail completely (0.00), worse than random. The cause is out-of-distribution querying: dense hallucinates 5 to 10 cm of motion with no contact, while sparse’s velocity/context features leave its gate silent on the teleported, near-rest states a planner visits.

Round 2: fixing the diagnosed cause. We add a contactaware, velocity-free feature mode and retrain on diverse mixedpolicy data covering approach angles the planner samples. With the planner unchanged (Tab. IV), sparse goes 0.00 to $0 . 2 3 { \pm } 0 . 0 6$ success and halves final distance, while dense stays at 0.00 at every seed. But $0 . 2 3 \pm 0 . 0 6$ vs. random 0.15 is not significantly better than random (weakest seed exactly 0.15, App. G); the honest claim is not “begins to plan” but “not significantly better than random, but separated from dense at every seed.” Diverse data does not cure dense’s hallucination; the separation from dense is the robust finding.

## IX. LIMITATIONS

What the headline is and is not. Sparse is a change detector that avoids error injection with delta regression at roughly noop quality (Sec. V, App. F); phrases like “predicts dynamics far more accurately” invite a reading the data do not support and we retire them. Rollout is better than dense but worse than no-op overall (Tab. III); the world-model title should be read as relative to dense on static objects, not absolute. Scope: objectcentric, not pixel-space. Structured state, not pixels; results do not transfer directly to perception-first settings. Metrics and baselines. F1’s dense degeneracy is partly definitional $( \epsilon _ { p }$ threshold, App. B); we report a deadband sweep and PR curve, and foreground $L _ { 2 } ^ { \mathrm { u n c h } } / L _ { 2 } ^ { \mathrm { c h } }$ instead. A dense-residual (nonobject-centric) baseline is missing, confounding residual vs. object-centric attribution; and Sec. VI / Figs. 3b,c are singleseed. Planning is preliminary and narrow. Single planner (CEM), single task, 3 objects; sparse $0 . 2 3 \pm 0 . 0 6$ vs. random 0.15 is not significantly better than random (weakest seed 0.15), well below scripted 0.95; the robust claim is separation from dense (0.00 at every seed). 8-object geometry differs, so cross-N density trends are not perfectly controlled; efficiency is capacity, not wall-clock.

## X. CONCLUSION AND FUTURE WORK

A per-object gate plus masked residual is a simple bias that avoids error injection: sparse matches no-op overall and at roughly no-op on movers, but keeps F1 0.80–0.87 at 8.6–11.1× fewer parameters where dense is degenerate, is far better than dense in rollout (but still worse than no-op), transfers across counts, and is separated from dense in planning (0.23 ± 0.06 vs. 0.00, vs. random 0.15 n.s.) while not significantly beating random. The ladder attributes the largest single share to the gate; the oracle diagnostic shows the bottleneck is delta regression. Next steps are a dense-residual baseline, a distributional delta head, DAgger on planner states, multi-step training, and slotbased encoders to drop the structured-state assumption. Stated plainly, this is a change detector that avoids corruption, not an accurate mover predictor — and that, honestly framed, is still publishable and interesting.

## REFERENCES

[1] D. Ha and J. Schmidhuber, “World models,” arXiv preprint arXiv:1803.10122, 2018.

[2] D. Hafner, T. Lillicrap, J. Ba, and M. Norouzi, “Dream to control: Learning behaviors by latent imagination,” in International Conference on Learning Representations (ICLR), 2020.

[3] K. Chua, R. Calandra, R. McAllister, and S. Levine, “Deep reinforcement learning in a handful of trials using probabilistic dynamics models,” in Advances in Neural Information Processing Systems (NeurIPS), 2018, pp. 4759–4770.

[4] G. Williams, N. Wagener, B. Goldfain, P. Drews, J. M. Rehg, B. Boots, and E. A. Theodorou, “Information theoretic mpc for model-based reinforcement learning,” in IEEE International Conference on Robotics and Automation (ICRA), 2017, pp. 1714–1721.

[5] R. Y. Rubinstein, “The cross-entropy method for combinatorial and continuous optimization,” Methodology and Computing in Applied Probability, vol. 1, no. 2, pp. 127–190, 1999.

[6] P. W. Battaglia, R. Pascanu, M. Lai, D. Rezende, and K. Kavukcuoglu, “Interaction networks for learning about objects, relations and physics,” in Advances in Neural Information Processing Systems (NeurIPS), 2016.

[7] A. Sanchez-Gonzalez, J. Godwin, T. Pfaff, R. Ying, J. Leskovec, and P. W. Battaglia, “Learning to simulate complex physics with graph networks,” in International Conference on Machine Learning (ICML), vol. 119, 2020, pp. 8459–8468.

[8] T. Kipf, E. van der Pol, and M. Welling, “Contrastive learning of structured world models,” in International Conference on Learning Representations (ICLR), 2020.

[9] B. A. Olshausen and D. J. Field, “Emergence of simple-cell receptive field properties by learning a sparse code for natural images,” Nature, vol. 381, no. 6583, pp. 607–609, 1996.

[10] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. Le, G. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in International Conference on Learning Representations (ICLR), 2017.

[11] Y. Bengio, N. Leonard, and A. Courville, “Estimating or propagating ´ gradients through stochastic neurons for conditional computation,” arXiv preprint arXiv:1308.3432, 2013.

[12] E. Jang, S. Gu, and B. Poole, “Categorical reparameterization with gumbel-softmax,” in International Conference on Learning Representa tions (ICLR), 2017.

[13] C. J. Maddison, A. Mnih, and Y. W. Teh, “The concrete distribution: A continuous relaxation of discrete random variables,” in International Conference on Learning Representations (ICLR), 2017.

[14] M. T. Mason, “Mechanics and planning of manipulator pushing operations,” The International Journal of Robotics Research, vol. 5, no. 3, pp. 53–71, 1986.

[15] K.-T. Yu, M. Bauza, N. Fazeli, and A. Rodriguez, “More than a million ways to be pushed: A high-fidelity experimental dataset of planar pushing,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016, pp. 30–37.

[16] P. Agrawal, A. Nair, P. Abbeel, J. Malik, and S. Levine, “Learning to poke by poking: Experiential learning of intuitive physics,” in Advances in Neural Information Processing Systems (NeurIPS), 2016.

[17] C. Finn and S. Levine, “Deep visual foresight for planning robot motion,” in IEEE International Conference on Robotics and Automation (ICRA), 2017.

[18] A. Byravan and D. Fox, “Se3-nets: Learning rigid body motion using deep neural networks,” in IEEE International Conference on Robotics and Automation (ICRA), 2017.

## APPENDIX A HYPERPARAMETERS AND TRAINING

Dense baseline: MLP, hidden width 256, 3 layers, ReLU; $L _ { 2 }$ loss on all poses; Adam, learning rate $1 0 ^ { - 3 }$ , batch 128, 25 epochs. Sparse/residual: per-object features; gate and delta heads each a 2-layer MLP of width 128; Gumbel straightthrough gate, temperature 1.0; loss (2) with the delta term weighted by the predicted gate probability; sparsity weight $\lambda _ { s } = 0 . 2$ (prediction) or 0.05 (planning); Adam, $1 0 ^ { - 3 }$ , batch 128, 15 epochs (prediction) or 25 (contact/planning). Data: 250 scripted episodes (×100 steps) per (N, seed) for prediction; a mixed set of 200 scripted $( \times 8 0 ) + 3 5 0$ random (×60) episodes for planning; hard-subset threshold 0.02 m; 80/10/10 split with a configuration-leakage guard. Planner: CEM, 256 samples, 3 iterations, elite fraction 0.1, horizon 15, initial std 0.6, terminal weight 3.0, proximity weight 0.3, replanning every step, up to 60 environment steps per episode.

## APPENDIX B

## METRIC AND PLANNER DEFINITIONS

Notation. A test split of N transitions with K objects each; $p _ { n , i } \in \mathbb { R } ^ { 3 }$ is the planar pose $( x , y , \theta )$ of object i in transition n, with $\hat { p }$ the prediction and $p ^ { \star }$ the ground truth.

Pose error. Per-object error is the norm of the full pose residual,

$$
e _ { n , i } = \| \hat { p } _ { n , i } - p _ { n , i } ^ { \star } \| _ { 2 } ,\tag{3}
$$

taken over all three components, so metres and radians are pooled in a single norm: a convention kept for comparability across models, not a claim that the units are commensurate. The three reported aggregates are the unweighted mean and the means restricted to changed and unchanged objects,

$$
L _ { 2 } ^ { \mathrm { a l l } } = \frac { 1 } { N K } \sum _ { n , i } e _ { n , i } , \qquad L _ { 2 } ^ { \mathrm { c h } } = \frac { \sum _ { n , i } m _ { n , i } ^ { \star } e _ { n , i } } { \sum _ { n , i } m _ { n , i } ^ { \star } } ,\tag{4}
$$

with $L _ { 2 } ^ { \mathrm { u n c h } }$ defined likewise using $1 - m ^ { \star }$

Change mask. Ground-truth $m _ { n , i } ^ { \star } = \mathbf { 1 } [ \| \Delta _ { x y } ^ { \star } \| _ { 2 } > \epsilon _ { p } ] \lor$ ${ \bf 1 } [ | \mathrm { w r a p } ( \Delta _ { \theta } ^ { \star } ) | > \epsilon _ { \theta } ]$ with $\scriptstyle \epsilon _ { p } = 1 0 ^ { - 3 } \mathrm { m } , \epsilon _ { \theta } = 1 0 ^ { - 2 }$ rad. Sparse reports gate $g _ { i }$ directly; every ungated baseline has no change output, so its mask is derived by thresholding its predicted delta at the same $\epsilon _ { p } .$ . Any nonzero regression clears $\epsilon _ { p } ,$ forcing recall to 1 in Tab. II; F1 is therefore partly a property of the metric. For a fairer view we also report (i) dense with a deadband $\tau$ on $\| \hat { \Delta } _ { x y } \| _ { 2 }$ swept from $1 0 ^ { - \bar { 3 } }$ to $1 0 ^ { - 2 }$ m and (ii) a precision– recall curve over τ : dense precision rises modestly before recall collapses and the F1 gap survives but narrows, so we foreground $L _ { 2 } ^ { \mathrm { u n c h } } / L _ { 2 } ^ { \mathrm { c h } }$ and treat F1 with this caveat. Precision, recall, F1 are pooled over NK slots.

Planner. At each environment step the planner optimises an action sequence $a _ { t : t + H - 1 } ~ ( H { = } 1 5 )$ by the cross-entropy method. Each of $J { = } 3$ iterations draws $B { = } 2 5 6$ candidates,

$$
a ^ { ( b ) } \sim { \mathcal { N } } ( \mu , \mathrm { d i a g } \sigma ^ { 2 } ) , \quad \quad a ^ { ( b ) } \gets \mathrm { c l a m p } \Big ( a ^ { ( b ) } , - 1 , 1 \Big )\tag{5}
$$

rolls each through the model under evaluation, keeps the $\lceil \rho B \rceil$ lowest-cost elites $( \rho { = } 0 . 1 )$ , and refits $\mu  \mathrm { m e a n } ( \mathrm { e l i t e s } )$

$\sigma ~ \gets ~ \operatorname* { m a x } ( \mathrm { s t d } ( \mathrm { e l i t e s } ) , \sigma _ { \mathrm { m i n } } )$ , starting from $\sigma _ { 0 } { = } 0 . 6$ with $\sigma _ { \mathrm { m i n } } { = } 0 . 0 5$ . Writing $d _ { h } = \lVert x _ { h } ^ { \mathrm { t g t } } - g \rVert _ { 2 }$ for the imagined targetto-goal distance at horizon step h, the cost is

$$
J = \frac { 1 } { H } \sum _ { h = 1 } ^ { H } d _ { h } \ + \ w _ { T } d _ { H } \ + \ \frac { w _ { p } } { H } \sum _ { h = 1 } ^ { H } \lVert x _ { h } ^ { \mathrm { p u s } } - x _ { h } ^ { \mathrm { t g t } } \rVert _ { 2 } ,\tag{6}
$$

with $w _ { T } { = } 3 . 0$ and $w _ { p } { = } 0 . 3 \colon$ the mean term supplies dense progress, the terminal term is the actual objective, and the proximity term makes contact discoverable. Only the first action is executed; the next step warm-starts $\mu$ from the shifted previous solution. An episode succeeds if $\| x ^ { \mathrm { t g t } } - g \| _ { 2 } \leq 5$ cm within 60 steps. Across all conditions only the model rolling the sequences forward differs.

## APPENDIX C

## FULL EFFICIENCY BREAKDOWN

Table V gives parameters, forward-pass operations, and CPU latency per model and object count (mean over 3 seeds). Parameter efficiency is the durable win; the operation-count advantage erodes with N as the per-object heads scale, and dense is faster in wall-clock latency (one matmul vs. per-object gating), reported here for transparency, not as a claim.

TABLE V  
EFFICIENCY: PARAMETERS, OPERATIONS (FLOP ESTIMATE), CPULATENCY.
<table><tr><td> $N$ </td><td>sparse params</td><td>dense params</td><td>param ratio</td><td>sparse FLOPs</td><td>dense FLOPs</td><td>lat. (ms) sparse/dense</td></tr><tr><td>3</td><td>6916</td><td>76809</td><td>11.1×</td><td>39 936</td><td>152 576</td><td>0.19 / 0.04</td></tr><tr><td>5</td><td>8452</td><td>82959</td><td>9.8×</td><td>81920</td><td>164864</td><td>0.20 / 0.04</td></tr><tr><td>8</td><td>10756</td><td>92184</td><td>8.6×</td><td>167 936</td><td>183 296</td><td>0.20 / 0.04</td></tr></table>

## APPENDIX D

## SPARSITY-WEIGHT ABLATION

A five-point sweep of $\lambda _ { s }$ (3 objects, seed 0, otherwise identical config, Table VI) makes the gate more conservative (precision rises and recall falls) while pose error is essentially flat. The main runs use $\lambda _ { s } = 0 . 2$

TABLE VI  
EFFECT OF THE SPARSITY WEIGHT $\lambda _ { s }$ ON THE GATE.
<table><tr><td> $\lambda _ { s }$ </td><td>F1</td><td>precision</td><td>recall</td><td>changed-obj  $L _ { 2 }$ </td><td>overall  $L _ { 2 }$ </td></tr><tr><td>0.0</td><td>0.873</td><td>0.941</td><td>0.814</td><td>0.389</td><td>0.146</td></tr><tr><td>0.05</td><td>0.865</td><td>0.940</td><td>0.801</td><td>0.389</td><td>0.146</td></tr><tr><td>0.2</td><td>0.849</td><td>0.938</td><td>0.776</td><td>0.388</td><td>0.145</td></tr><tr><td>0.5</td><td>0.845</td><td>0.967</td><td>0.750</td><td>0.387</td><td>0.144</td></tr><tr><td>1.0</td><td>0.815</td><td>0.965</td><td>0.705</td><td>0.385</td><td>0.143</td></tr></table>

## APPENDIX E

## PARAMETER-MATCHED DENSE BASELINE

To remove the “sparse is just smaller” confound we shrink the dense MLP to the sparse model’s exact parameter budget and retrain (Table VII, test split, seed 0). Sparse still wins on every metric at every count: shrinking dense does not help, and its change detection stays degenerate: the advantage is object-centric structure, not parameter count.

TABLE VII  
SPARSE VS. PARAMETER-MATCHED DENSE (EQUAL BUDGET).
<table><tr><td> $N$ </td><td>sparse  $L _ { 2 }$ </td><td>dense-matched  $L _ { 2 }$ </td><td>sparse F1</td><td>dense-matched F1</td></tr><tr><td>3</td><td>0.116</td><td>0.363</td><td>0.885</td><td>0.538</td></tr><tr><td>5</td><td>0.104</td><td>0.363</td><td>0.777</td><td>0.388</td></tr><tr><td>8</td><td>0.081</td><td>0.415</td><td>0.837</td><td>0.294</td></tr></table>

## APPENDIX F

## ORACLE-GATE DIAGNOSTIC

Feeding the delta head the ground-truth changed mask isolates detection from regression (Table VIII, changed-object $L _ { 2 } .$ , test split, seed 0). Perfect detection barely moves the number, so the changed-object bottleneck is delta regression, not the gate.

TABLE VIII  
CHANGED-OBJECT $L _ { 2 } \colon$ PREDICTED VS. ORACLE GATE VS. NO-OP.
<table><tr><td>N</td><td>predicted gate</td><td>oracle gate</td><td>no-op</td></tr><tr><td>3</td><td>0.312</td><td>0.314</td><td>0.348</td></tr><tr><td>5</td><td>0.426</td><td>0.432</td><td>0.446</td></tr><tr><td>8</td><td>0.466</td><td>0.474</td><td>0.470</td></tr></table>

## APPENDIX G

## PER-SEED PLANNING RESULTS

Table IX lists the per-seed planning numbers summarized in Section VIII. The sparse advantage over dense holds at every seed (dense is $0 / 2 0$ throughout); the sparse margin over random (0.15) holds on average but not at the weakest seed.

TABLE IX  
CONTACT-AWARE PLANNING, PER TRAINING SEED (20 EPISODES EACH).
<table><tr><td>model</td><td>seed 0</td><td>seed 1</td><td>seed 2</td><td>mean ± std</td></tr><tr><td>sparse success</td><td>0.25</td><td>0.15</td><td>0.30</td><td> $0 . 2 3 \pm 0 . 0 6$ </td></tr><tr><td>dense success</td><td>0.00</td><td>0.00</td><td>0.00</td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>sparse fin. dist.</td><td>0.188</td><td>0.231</td><td>0.193</td><td> $0 . 2 0 4 \pm 0 . 0 1 9$ </td></tr></table>