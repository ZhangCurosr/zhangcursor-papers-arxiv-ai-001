# Can the Remaining Answers Change the Prediction? Worst-Case Stability Checks for Personalized Rating Models

Priyanka Siddappa, Muntaqim Ahmed Raju, Shifat Haider Al Amin, and Ruizhe Ma

University of Massachusetts Lowell, Lowell, MA 01854, USA

Email: {Priyanka\_Siddappa, MuntaqimAhmed\_Raju, MdShifat\_AlAmin, Ruizhe\_Ma}@uml.edu

Abstract—Missing ratings can overturn a personalized model’s current prediction. For completed models with additive response scores, exact worst case bounds check all missing assignments without changing the current predictor. We evaluate this audit retrospectively in two self-triage cohorts, 108 clinicians rating three simulated psychiatric interviews, and 126 counseling demonstrations annotated by ten experts. Under shared random half histories, Fisher adaptation based on observed ratings reduces log loss relative to response frequency by 0.1071 nats in the clinician audit (95% CI [0.0867,0.1261]) and 0.0223 in counseling (95% CI [0.0121,0.0321]). The counseling test excludes target experts and source videos from training. Certification covers 5.4% and 10.5% of targets; fixed width acquisition raises coverage to 13.7% and 28.6% at the same budgets. With weights and responses held fixed, the observed-rating update also has lower loss than an affine partial update while certifying exactly the same evaluated targets. The certificate establishes agreement with the completed model, not correctness or stability at intermediate updates. These rating tasks test the method; clinical usefulness remains untested.

Index Terms—model auditing, trustworthy AI, clinical decision support, missing responses, completion certification

## I. INTRODUCTION

Personalized rating models use an individual’s observed responses to predict how that person will rate other items. We study this setting in clinical and health-related rating tasks, including clinicians’ symptom ratings of simulated psychiatric interviews [1], [2], [3], [4]. When part of the calibration history is unavailable, the model must predict from incomplete information. Even a confident prediction may change once the remaining responses become available.

This raises a practical question: can any assignment of the missing ratings make the completed model disagree with the current class? The idea resembles calling an election when the leader wins under every allocation of the uncounted votes. Here, the uncounted inputs are missing ratings. A completion certificate establishes that the current class wins under every completed history.

Our main contribution is to separate the prediction made from partial history from the calculation of its completion certificate (Fig. 1). When the completed model adds contributions from individual responses, separability gives exact bounds over all missing assignments. The partial predictor can still use a different update, such as Observed Fisher’s update based on observed curvature. This allows us to audit its current class without changing its predictions. The guarantee concerns the fitted model’s behavior; it does not establish correctness or provide a causal explanation.

We evaluate completion agreement, error, responsiveness to personal history, and fitted-model variation (RQ1); predictive gains across people, interviews, and categories (RQ2); and acquisition and stopping policies (RQ3). Models are fitted separately within four rating datasets, used as methodological testbeds.

## II. RELATED WORK

Prior certification studies ask whether predictions agree across plausible training datasets [5], [6], including missingdata completions for Naive Bayes [7] and repairs for nearestneighbor classifiers [8], [9]. Our work applies this perspective to incomplete personal histories, with the learned encoder and adaptation rule held fixed. The methodological distinction is the separation of the partial predictor from its completion audit: the current prediction can use observed curvature, while certification uses an affine map of the completed history.

Efficient adaptation provides the computational basis for this separation. Differentiable closed-form solvers combine learned representations with ridge regression [10], [11], [12], and ALPaCA learns features and priors for Bayesian regression [13]. Attentive neural processes instead aggregate context separately for each target [14]. These methods motivate our adaptation and attention studies. For certification, the relevant restriction is additive dependence on response labels; the cue encoder can remain nonlinear.

The resulting guarantee differs from those studied in selective classification and conformal prediction. Selective classification controls the tradeoff between accepted predictions and error [15], [16], while conformal methods address outcome coverage under missingness [17]. Completion agreement concerns the behavior of the fitted model, so it can certify an incorrect prediction. Our matched-coverage confidence comparisons assess this distinction empirically.

Adaptive testing connects completion auditing to stopping. Besides estimating latent traits [18], projection-based rules examine whether hypothetical remaining answers can change a final classification [19]. We instead certify individual predicted ratings over a fixed calibration history and distinguish stopping for one target from stopping for all targets sharing that history.

![](images/b278862af67d84ecf7c1af34b426a0efcbef19c7f6fa03608d4845e3a20869c5.jpg)

Fig. 1. Completion auditing for a fixed model. Partial predictions may use observed curvature; optional width acquisition ranks responses once.  
![](images/858ee22bf09e06e680a4ad7387f5c7e3b1c6cf25e691cffc90cdf8e4fbf65228.jpg)  
Fig. 2. Minimum completion margins after 25 and 26 ratings. Retrieving item 78 certifies item 74; item 25 remains unresolved.

EDDI and DIME guide acquisition through information gain [20], [21], [22], [23]. Our acquisition experiments examine a different tradeoff: whether selecting responses for lower predictive loss also improves completion coverage. They compare these objectives without claiming to reproduce the complete EDDI or DIME systems.

## III. COMPLETION AUDITS FOR NEURAL ADAPTATION

a) Overview: Calibration ratings describe one person’s history; target ratings are what we predict. Item cues are known, but some calibration ratings are hidden. The audit checks whether the current class wins under every missing assignment. Coverage is the fraction of targets that pass.

For example, after 25 ratings, ridge predicts “absent” for item 74, but a missing-history completion can overturn it. Revealing item 78, rated “absent,” raises its worst case margin from −.0087 to .0410 and certifies the class. Item 25 remains unresolved (Fig. 2).

For one person, let $\textit { J } = \ \{ 1 , \ldots , H \}$ index calibration cases with known cues $x _ { j }$ and responses $y _ { j } \in \{ 1 , \ldots , K \}$ Responses in $O \subseteq J$ are observed; those in $U = J \setminus O$ are hidden. All T target cues $x _ { t }$ are known before acquisition. Parameters are learned from other people. A completion assigns a category to each hidden response, allowing every assignment in $\{ 1 , \bar { \dots } , K \} ^ { | U | }$ . This set imposes no statistical independence assumption. Retrieval reveals a recorded response; cues and fitted maps stay fixed.

## A. Affine neural adapters

An affine response map adds one contribution per rating to a baseline score. Its case encoder can be nonlinear; additivity is required only in the responses. A nonlinear encoder produces population logits $b ( \boldsymbol { x } ) ~ \in ~ \mathbb { R } ^ { K }$ and a basis $A ( x ) \ \in \ \mathbb { R } ^ { K \times }$ r centered over output classes. Define $\begin{array} { r l } { p _ { j } } & { { } = } \end{array}$ softmax $( b _ { j } )$ $S _ { j } ~ = ~ \mathrm { d i a g } ( p _ { j } ) - p _ { j } p _ { j } ^ { \intercal }$ , and the one-hot vector $e _ { y _ { j } }$ . All unqualified sums below run over J. With positive cue weights $\alpha _ { t j }$ and damping $\lambda > 0$ , the Fisher adapter is

$$
D _ { t } = \lambda I + \sum _ { j } \alpha _ { t j } A _ { j } ^ { \top } S _ { j } A _ { j } ,\tag{1}
$$

$$
W _ { t j } = \alpha _ { t j } A _ { t } D _ { t } ^ { - 1 } A _ { j } ^ { \top } ,\tag{2}
$$

$$
\ell _ { t } ( y ) = b _ { t } + \sum _ { j } W _ { t j } ( e _ { y _ { j } } - p _ { j } ) .\tag{3}
$$

Global adaptation sets $\alpha _ { t j } ~ = ~ 1$ . Routed adaptation uses H softma $\mathrm { x } _ { j } ( q ( \boldsymbol { x } _ { t } ) ^ { \top } k ( \boldsymbol { x } _ { j } ) / \bar { \sqrt { r } } )$ , where q and k are learned linear maps from cues to $\mathbb { R } ^ { r }$ . The maps never see responses. Equation (3) is one damped Fisher update from a zero personal vector in logits $b ( x ) + A ( x ) z ;$ matrices are evaluated with linear solves.

The ridge adapter uses r learned features plus an intercept, $\phi ( x ) \in \mathbb { R } ^ { r + 1 }$ :

$$
C _ { t } = \lambda I + \sum _ { j } \alpha _ { t j } \phi _ { j } \phi _ { j } ^ { \top } ,\tag{4}
$$

$$
W _ { t j } = s \alpha _ { t j } ( \phi _ { t } ^ { \top } C _ { t } ^ { - 1 } \phi _ { j } ) I _ { K } ,\tag{5}
$$

where $s = \exp ( \exp ( \gamma , - 3 , 3 ) )$ for learned scalar γ. Both families retain Eq. (3); the reference damping is $\lambda = 1$ . Point logits use only the observed residual terms. They equal the mean completed logits if hidden responses have marginals $p _ { j } ;$ their softmax need not equal the mean completed probability vector.

Matched frequency uses the same encoder and head architectures, separately fitted without a completion penalty. Its logits are $b _ { t } + V f _ { O } + ( n / H ) v$ for learned $V , v ,$ , observed count $n ,$ and frequency vector $f _ { O }$ (zero at $n = 0 )$ . Completed logits are $\begin{array} { r } { b _ { t } + v + \sum _ { i } ( V / H ) e _ { y _ { j } } } \end{array}$ . Equation (7) applies with $W _ { t j } = V / H$ and $\begin{array} { r } { c _ { t } \ = \ \stackrel { \_ } { b } _ { t } \ + v \ + \ \sum _ { j \in O } ( V / H ) e _ { y _ { j } } } \end{array}$ . The audit concerns the completed class; changing n can change intermediate predictions. Equal widths use random ordering.

Identical frequency maps discard item identity and can yield high coverage with weak personalization.

## B. Exact completion certificate

Write

$$
c _ { t } = b _ { t } - \sum _ { j } W _ { t j } p _ { j } + \sum _ { j \in O } W _ { t j } e _ { y _ { j } } .\tag{6}
$$

For candidate class a and competitor $b ,$ the smallest completed logit margin is

$$
L _ { t a b } = c _ { t a } - c _ { t b } + \sum _ { j \in U } \operatorname* { m i n } _ { k } \{ ( W _ { t j } ) _ { a k } - ( W _ { t j } ) _ { b k } \} .\tag{7}
$$

Proposition 1. For fixed parameters, cases, and routing based only on cues, Eq. (7) is the attained minimum over all response completions. Class a is the unique winner under every completion if and only if $L _ { t a b } > 0 f o r$ all $b \neq a .$ . Revealing a response cannot decrease any pairwise lower bound.

Proof. Each hidden response contributes a separate summand. Choosing its minimizing category attains the joint minimum because the completion set is a Cartesian product. Positivity of every pairwise minimum is equivalent to a common unique winner. A reveal restricts the completion set, so its minimum cannot decrease. □

A negative bound supplies a counterexample: assigning each hidden response its minimizing category makes the competitor beat class a.

a) Auditing Observed Fisher: This predictor recomputes its Fisher update from the observed ratings and their cues. Its partial logits are

$$
D _ { O } = \lambda I + \sum _ { j \in O } A _ { j } ^ { \top } S _ { j } A _ { j } ,\tag{8}
$$

$$
\hat { \ell } _ { t } ^ { O } = b _ { t } + A _ { t } D _ { O } ^ { - 1 } \sum _ { j \in O } A _ { j } ^ { \top } ( e _ { y _ { j } } - p _ { j } ) .\tag{9}
$$

At full history, $D _ { J }$ depends only on cues, giving Eq. (3) with uniform weights. We apply the completed-map bounds to the unchanged partial winner. Passing certifies agreement with every completed history, without retraining. Intermediate updates may disagree because their curvature changes. A stopping rule must therefore retain the first certified class rather than continue updating it.

The guarantee conditions on a fitted adapter and completion set. For a specified finite family of fits $\mathcal { M } .$ , a common class is certified by $\mathrm { m i n } _ { m \in \mathcal { M } } L _ { t a b } ^ { ( m ) } > 0$ for every competitor. This covers the listed fits, not all plausible parameters, and is stronger than certifying their mean logits. At evaluation, maps and margins use IEEE double precision with threshold $1 0 ^ { - 8 }$ in original logit units. Certification requires every competing margin to exceed this threshold; smaller positive margins remain unresolved. Point ties use the first category index. Numerical checks are not interval arithmetic verification.

## C. Supporting response acquisition

For response $j$ and class pair $( a , b )$ , define its removable interval width

$$
d _ { t j a b k } = ( W _ { t j } ) _ { a k } - ( W _ { t j } ) _ { b k } ,\tag{10}
$$

$$
\Delta _ { t j } ^ { a b } = \operatorname* { m a x } _ { k } d _ { t j a b k } - \operatorname* { m i n } _ { k } d _ { t j a b k } .\tag{11}
$$

We sum $\Delta _ { t j } ^ { a b }$ over all targets and ordered class pairs, then rank responses once. Reveals do not change this order. At any budget, the largest scores minimize remaining total width because the objective separates by position. This is width optimality, not optimality for certification count or loss.

Expected gain rescans hypothetical reveals, weighting newly certified targets by $p _ { j k } = \mathrm { s o f t m a x } ( b _ { j } ) _ { k }$ . Section IV-D tests calibrated and personal probabilities. Scores within $1 0 ^ { - 1 2 }$ of the maximum use width order to break ties. Hidden labels never score test queries. This myopic objective optimizes immediate expected count.

a) Computational cost: After encoding and routing, Fisher needs $O ( H K r ^ { 2 } )$ shared cue curvature work and $O ( H r ^ { 2 } + r ^ { 3 } + K r ^ { 2 } + H K ^ { 2 } r )$ per target to form its maps. Ridge costs $O ( H d ^ { 2 } + d ^ { 3 } + H K ^ { 2 } )$ per target, with $d = r + 1$ . Target costs multiply by $T .$ Dense margins and width scores each take $O ( T H K ^ { 3 } )$ ; ranking adds $O ( H \log H )$ . The implementation stores $O ( T H K ^ { 2 } )$ maps and materializes $O ( T H K ^ { 3 } )$ pair differences. At budget $B ,$ lookahead repeats $O ( T H K ^ { 3 } )$ work per retrieval. The dense evaluator does not exploit ridge’s scalar identity structure.

## D. Training

The encoder has 24 tanh units and rank four. We average full and masked history cross entropy, with training budgets zero, one quarter, one half, or full history. Fisher and ridge add $\begin{array} { r } { \beta \log { \sum _ { b } \exp ( - L _ { t a _ { t } b } ) } } \end{array}$ over at most 12 sampled targets, with $\beta = . 0 3$ and detached full history winner $a _ { t } .$ Adam uses learning rate .01, weight decay .001, gradient clipping at 10, and 60 epochs across cohorts. Layperson sensitivity varies rank (2, 4, 8), width (12, 24, 48), damping (.25, 1, 4), and penalty (0, .03, .06) one factor at a time, without replacing the reference model.

a) Response influence control: A separate experiment freezes an unpenalized ridge teacher and multiplies each map by a cue-only gate in $[ f , 1 . 5 ]$ , initialized at one. Floors $f = 0 , . 5$ are crossed with penalties $\beta = 0 , . 0 3$ . The .5 floor preserves half of each single-response logit contrast, not class changes or accuracy.

## IV. EXPERIMENTAL DESIGN

All experiments retrospectively hide recorded responses.

## A. Cohorts

RepVig: 198 laypeople each rate 20 of 45 vignettes as emergency, nonemergency, or self care, giving 3,960 judgments. A hash divides each history into ten calibration and ten target responses. The study reports informed consent, TU Berlin approval AWB\_KOP\_2\_230711, and registration DRKS00032895 [24], [25].

Schmieding: 91 US Mechanical Turk participants each rate 45 vignettes at the same urgency levels, giving 4,095 judgments. A hash divides histories into 22 calibration and 23 target responses. The study reports informed consent and TU Berlin approval FEU\_03\_20180615 [26]. Both cohorts use one-hot vignette identity.

Psychopathology assessment: 108 practicing, mostly early career clinicians provide 132 sheets across three simulated interviews [1], [27]. Each sheet rates 100 Association for Methodology and Documentation in Psychiatry (AMDP) items, reduced to absent, present, or not assessable. The last is an observed category, not a mask. A hash assigns 50 calibration and 50 target items. Features identify item and interview; expert ratings serve only secondary evaluation. Grouped folds test new clinicians on familiar items and interviews. The study reports written consent and Heidelberg approval 2024-567 [1].

AnnoMI: the full public release contains 133 motivational interviewing demonstrations annotated by ten experts [28]. We use 64 therapist utterances selected by hash from seven jointly annotated interviews as calibration. The other 126 interviews, from 112 source videos, supply 4,666 target ratings of therapist behavior in four categories. Five expert folds under three new seeds hold out experts and exclude their target source videos from fitting. TF IDF and 32 SVD features use training target texts and shared calibration texts only; no target interview identity is supplied. AnnoMI was collected independently and contains demonstrations rather than patient encounters. Each target interview has one annotator, so expert and interview assignments are confounded.

## B. Splits, models, and metrics

RepVig, Schmieding, and psychopathology use three partici pant folds under three seeds, grouping clinicians across sheets and excluding test participants from training. Controls are population and ordinary frequency MLPs, matched frequency, Observed Fisher, and target conditioned attention pooling encoded case/response pairs. The reference tables do not audit Observed Fisher; a separate checkpoint audit below does. Ordinary frequency receives no certificate. Population ignores history and is trivially invariant.

Budgets are 0, 25, 50, 75, and 100% of history, rounded down. Outcomes are log loss in nats, accuracy, coverage, certified error, and Brier score. Tables, figures, and contrasts average targets within sheet, sheets within person, and seeds within person, then weight people equally. Paired 95% intervals use 10,000 person bootstrap draws, keeping sheets together. These exploratory intervals condition on fitted models, exclude fitting and architecture search uncertainty, and are unadjusted unless stated.

Selections are formed within each fitted seed. A target is history-active when its predicted class differs between zero and full history. Active coverage is the certified fraction of this subset. Conditional metrics divide equally person weighted joint mass by selected mass; empty subsets are undefined. Confidence is the largest softmax probability from the partial history logits. It ranks individual targets, with fractional selection at the boundary to match certificate coverage exactly; no deployment threshold is selected. Top label calibration error uses ten equal width bins and the same weights, then averages seeds.

Internal protocols preceded computations without external preregistration; clinician settings preceded inspection of its results.

## C. Additional controls

Exploratory clinician controls select width (12, 24, 48) and epochs (40, 80, 120) using 20% of training clinicians and mean half/full validation loss, then refit on all training clinicians. A stricter test crosses clinician partitions with held psychiatric interviews, excludes test clinicians and interviews from fitting, and uses item identity alone. Half of the new interview’s ratings supply calibration. These three interviews were previously inspected.

Confidence targets are reselected within each bootstrap draw at matched weighted coverage. Category metrics use the same person weights before conditioning. Joint selection ranks confidence within certified predictions. Class weighting uses inverse training category frequencies, with unchanged architecture, epochs, and masks; it tests equal category emphasis, not clinical error costs.

## D. Learned policies, calibration, and fitted model variation

This extension reserves disjoint people for fitting, policy selection, temperature calibration, and testing. Selection and calibration each receive 15% of outer training people, rounded up; no refitting follows selection. Its smaller fitting sets require separate comparisons.

Seven policies share one frozen ridge: random, width, three expected gain variants, learned immediate loss reduction, and terminal loss. Gain variants use raw, temperature-scaled, or updated personal probabilities. Our control regresses realized loss reductions for a frozen predictor, motivated by DIME [21], without estimating mutual information or inheriting its guarantees. A second policy uses REINFORCE on final halfbudget loss [29]. Adam at .003 trains for 300 epochs, with selection at epochs 100, 200, or 300. Hidden labels enter training objectives, never test query scores.

We use temperature scaling [30], fitting temperatures in [.25, 4] on calibration people; classes and certificates are unchanged. Five ridge initializations also share random masks and a width order based on their summed scores. These fits share one original three-fold partition.

## E. Response preservation experiment

Nine clinician and 15 AnnoMI splits compare a 60-epoch teacher, four gates trained for 60 additional epochs, and an unconstrained continuation with the same additional budget. AnnoMI controls receive 60 epochs, sampling 128 targets per expert per epoch and evaluating all targets. Three common random half masks test coverage on the same teacher-active subset. AnnoMI averages targets within expert, then experts equally. Intervals resample its ten experts. Source-video exclusions preceded inspection of external performance. Earlier overlapping-source fits are diagnostic only.

## F. Completion agreement without refitting

The exploratory checkpoint audit uses nine tuned clinician and 15 AnnoMI Observed Fisher fits, saved frequency and ridge controls, and three shared random half masks. Clinician masks differ from the tuned table. Fixed width acquisition and an affine partial rule use the same Fisher parameters, isolating the partial update. No models are refitted or selected for coverage.

## G. Stopping at the first certificate

The post-review stopping replay evaluates every prefix under the checkpoint audit’s three random orders and fixed width order. It retains each target’s first certified class, falling back to the full-history prediction if unresolved. We distinguish responses needed per target from those needed for all targets sharing a history. No models are refitted; intervals use 10,000 person bootstrap draws.

## V. RESULTS

## A. Completion agreement without changing the predictor

With weights and revealed responses held fixed, replacing Observed Fisher’s partial update with the affine rule raises random-mask loss by .0578 nats in clinicians (95% CI [.0472,.0690]) and .0591 in AnnoMI ([.0459,.0720]). Both rules certify exactly the same targets (Table I), preserving the lower-loss predictions without reducing measured coverage.

Against frequency, Observed Fisher’s loss differences are −.1071 ([−.1261, −.0867]) in clinicians and −.0223 ([−.0321, −.0121]) in AnnoMI. Against ridge, its loss is lower by .0621 ([.0476,.0768]) and .0174 ([.0066,.0273]), respectively. These comparisons share random masks. With width acquisition, the clinician difference from ridge is −.0139 ([−.0295, .0022]), which includes zero.

Observed Fisher’s history-active fractions are 32.1% in clinicians and 30.9% in AnnoMI. Width certifies 8.0% and 15.8% of those active targets, versus 13.7% and 28.6% overall. Error among its width-certified predictions is 10.3% and 5.6%. These subsets are not matched for coverage.

## B. Where predictive gains generalize

For new clinicians on familiar interviews under shared random masks, ridge minus population loss is −.1240 (95% CI [−.1355, −.1133]). Differences from ordinary and matched frequency are −.0488 ([−.0681, −.0282]) and −.0982 ([−.1071, −.0889]). Observed Fisher has the lowest loss (.5788; Table II). These reference masks differ from the checkpoint audit above.

With equal validation grids, ridge’s loss differences from ordinary and matched frequency are −0.0433 (95% CI [−0.0583, −0.0271]) and −0.0665 ([−0.0750, −0.0579]). This loss advantage is not established on held interviews, where the difference from ordinary frequency is .0198 ([−0.0006, .0446]). This interval conditions on only three interviews.

Ridge improves loss over population in both layperson cohorts, but neither comparison with ordinary frequency establishes an advantage (Table III). On AnnoMI, ridge gives .8442 loss versus .8492 for frequency, a difference of −.0050 (95% CI [−.0133, .0040]). Population and matched frequency give .8490 and .8513; Observed Fisher gives .8269. Thus the counseling results support Observed Fisher, but not ridge superiority, on unseen experts and source videos. They concern behavior annotations, not diagnoses or treatment decisions.

Ridge improves absent and present recall but reduces not assessable recall (Table IV). Among recorded not assessable targets, its width-certified error is 60.7% (95% CI [52.3,68.7]). Counts refer to distinct ratings.

Class reweighting raises not assessable recall from 26.7% to 63.2%, but lowers precision from 65.9% to 38.0% and raises loss from .6863 to .7861. It also leaves the held-interview gap unresolved. For certified errors, collecting the remaining history cannot correct the completed model’s class; improving the predictor or referring the item for human review remains necessary.

## C. What certified coverage measures

Saved minimizing completions overturn both initial predictions in Fig. 2.

Matched frequency certifies 92.2%, 91.6%, and 81.3% of half-history predictions in RepVig, Schmieding, and clinicians, despite weaker clinician prediction. In the same cohort order, ridge’s zero-response coverage is 42.3%, 33.7%, and 3.0%. Its active fractions are 6.6%, 4.8%, and 23.6%; half-budget width coverage on these active subsets is 29.2%, 13.6%, and 23.4%. High aggregate coverage can therefore coexist with limited response dependence.

With width acquisition, error among ridge-certified targets is 25.6%, 22.7%, and 12.6% across these cohorts (Fig. 3). Confidence selection at matched coverage gives 24.6%, 21.9%, and 11.5%; paired intervals favor confidence in each cohort. Requiring both confidence and certification gives 24.0% clinician coverage and 7.83% error. Equal observed error to confidence alone does not establish equivalence (difference interval [−0.38, 0.39] percentage points).

In the calibration experiment, temperature scaling reduces clinician width loss to .6540, Brier score from .3734 to .3702, and calibration error from .0425 to .0268, preserving classes and certificates. Intervals for the loss changes include zero in both layperson cohorts.

Under common width masks, average individual coverage is 48.5%, mean-logit coverage is 49.6%, and one class is certified across all five fits on 40.5% of clinician targets. Random masks give 32.2%, 33.1%, and 27.0%. All five partial classes agree on 95.6% of width targets, but this alone does not imply completion invariance. The common certificate covers only the specified fits.

## D. Can the penalty preserve personalization?

Removing the routed Fisher penalty lowers width loss from .6167 to .5974 but reduces coverage from 47.5% to 13.8%.

Paired loss difference (nats); negative favors first method  
TABLE I  
COMPLETION AGREEMENT AT HALF HISTORY WITHOUT REFITTING. RANDOM MASKS ARE SHARED; WIDTH MASKS DIFFER BY MODEL. ACTIVE COVERAGE USES EACH MODEL’S OWN ACTIVE SUBSET. THE TWO OBSERVED FISHER POINT RULES SHARE WEIGHTS AND MASKS.
<table><tr><td></td><td></td><td colspan="3">Clinicians: 25/50 ratings</td><td colspan="3">AnnoMI: 32/64 ratings</td></tr><tr><td>Predictor</td><td>Retrieval</td><td>Loss</td><td>Coverage (%)</td><td>Active coverage (%)</td><td>Loss</td><td>Coverage (%)</td><td>Active coverage (%)</td></tr><tr><td>Frequency</td><td>Random</td><td>0.6916</td><td>n/a</td><td>n/a</td><td>0.8492</td><td>n/a</td><td>n/a</td></tr><tr><td>Ridge</td><td>Random</td><td>0.6466</td><td>20.8</td><td>3.9</td><td>0.8442</td><td>41.4</td><td>0.0</td></tr><tr><td>Observed Fisher</td><td>Random</td><td>0.5845</td><td>5.4</td><td>1.4</td><td>0.8269</td><td>10.5</td><td>0.2</td></tr><tr><td>Same fit, affine point</td><td>Random</td><td>0.6423</td><td>5.4</td><td>1.4</td><td>0.8860</td><td>10.5</td><td>0.2</td></tr><tr><td>Ridge</td><td>Width</td><td>0.5890</td><td>57.9</td><td>37.0</td><td>0.8402</td><td>48.2</td><td>0.5</td></tr><tr><td>Observed Fisher</td><td>Width</td><td>0.5751</td><td>13.7</td><td>8.0</td><td>0.8286</td><td>28.6</td><td>15.8</td></tr><tr><td>Same fit, affine point</td><td>Width</td><td>0.6012</td><td>13.7</td><td>8.0</td><td>0.8498</td><td>28.6</td><td>15.8</td></tr></table>

![](images/ea6cce740a1a5f6527a6e5f1dc4c67021573128b879881adc32321425d2379ef.jpg)

![](images/5596d00bf7d998ea3e0446539fd7de5105dad786d1f61c485a0924c235f909d0.jpg)

![](images/79c0e6f35f40b9dfac87dfa0ecd4936de8fd5d95b09716e4897bc2437dbcdc50.jpg)  
Fig. 3. Ridge results, equally weighted by person. (A) Paired loss contrasts (95% intervals); random masks except width versus random. (B) Coverage at zero and half history. (C) Error at half history. Half-history results in B and C use width acquisition; confidence matches certified coverage.

TABLE II  
CLINICIAN PREDICTION WITH 25 OF 50 RATINGS. REFERENCE MODELS SHARE RANDOM MASKS; TUNED USES INNER VALIDATION; UNSEEN HOLDS OUT CLINICIANS AND AN INTERVIEW. ACCURACY AND COVERAGE ARE FRACTIONS; N/A: NOT EVALUATED.
<table><tr><td></td><td colspan="3">Reference</td><td colspan="2">Loss</td></tr><tr><td>Model</td><td>Loss</td><td>Acc.</td><td>Cov.</td><td></td><td>Tuned Unseen</td></tr><tr><td>Population</td><td>.8103</td><td>.620</td><td>n/a</td><td>.7725</td><td>1.1174</td></tr><tr><td>Frequency</td><td>.7352</td><td>.660</td><td>n/a</td><td>.6875</td><td>1.0177</td></tr><tr><td>Matched frequency</td><td>.7845</td><td>.632</td><td>.813</td><td>.7108</td><td>1.0827</td></tr><tr><td>Attention</td><td>.7419</td><td>.655</td><td>n/a</td><td>.6990</td><td>n/a</td></tr><tr><td>Observed Fisher</td><td>.5788</td><td>.770</td><td>n/a</td><td>.5823</td><td>n/a</td></tr><tr><td>Global Fisher</td><td>.6653</td><td>.733</td><td>.239</td><td>n/a</td><td>n/a</td></tr><tr><td>Routed Fisher</td><td>.6666</td><td>.731</td><td>.248</td><td>n/a</td><td>n/a</td></tr><tr><td>Weighted ridge</td><td>.6863</td><td>.723</td><td>.316</td><td>.6443</td><td>1.0375</td></tr></table>

Because the active fractions also change, the gate experiment compares certification on a common subset defined by the unpenalized teacher.

At floor .5, the penalty raises aggregate coverage but worsens loss by .0066 nats in clinicians (95% CI [.0055,.0079]) and .0047 in AnnoMI ([.0022,.0071]). It does not improve certification on the common teacher-active subset under random masks. Unconstrained continuation has lower loss in both cohorts. Preserving individual response contrasts therefore does not establish a better prediction and certification tradeoff.

## E. Choosing which responses to retrieve

For ridge, fixed width acquisition improves clinician loss over random by .0345 nats (95% CI [.0288,.0400]). The loss difference is −.00481 in RepVig ([−.00996, −.00006]) and −.0035 in Schmieding $( [ - . 0 0 8 0 , . 0 0 0 6 ] )$ . The first narrowly excludes zero among exploratory comparisons; the second does not. Figure 3 also shows their coverage differences.

TABLE III  
PREDICTION AT HALF HISTORY. RANDOM MASKS ARE SHARED; WIDTH USES THE SAME RIDGE MODEL. LOSS IS IN NATS; OTHER METRICS ARE FRACTIONS. N/A: CERTIFICATE NOT EVALUATED.
<table><tr><td>Model</td><td>Policy</td><td>Loss Acc.</td><td></td><td> $\operatorname { C o v . }$ </td><td>Cert. error</td></tr><tr><td>RepVig (N = 198)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Population</td><td></td><td>random .6940 .702</td><td></td><td>n/a</td><td>n/a</td></tr><tr><td>Frequency</td><td></td><td>random .6814</td><td>.705</td><td>n/a</td><td>n/a</td></tr><tr><td>Matched frequency random .6896</td><td></td><td></td><td>.700</td><td>.922</td><td>.281</td></tr><tr><td>Attention</td><td></td><td>random .6822</td><td>.703</td><td>n/a</td><td>n/a</td></tr><tr><td>Observed Fisher</td><td></td><td>random .6939</td><td>.697</td><td>n/a</td><td>n/a</td></tr><tr><td>Global Fisher</td><td></td><td>random .6890</td><td>.705</td><td>.659</td><td>.236</td></tr><tr><td>Routed Fisher</td><td>random .6899</td><td></td><td>.705</td><td>.654</td><td>.236</td></tr><tr><td>Weighted ridge</td><td>random</td><td>.6854</td><td>.705</td><td>.748</td><td>.244</td></tr><tr><td>Weighted ridge</td><td>width</td><td>.6806</td><td>.706</td><td>.813</td><td>.256</td></tr><tr><td>Schmieding  $( N = 9 1 )$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Population</td><td></td><td>random .6464</td><td>.728</td><td>n/a</td><td>n/a</td></tr><tr><td>Frequency</td><td>random .6290</td><td></td><td>.733</td><td>n/a</td><td>n/a</td></tr><tr><td>Matched frequency</td><td>random .6415</td><td></td><td>.728</td><td>.916</td><td>.256</td></tr><tr><td>Attention</td><td>random .6284</td><td></td><td>.728</td><td>n/a</td><td>n/a</td></tr><tr><td>Observed Fisher</td><td>random .6367</td><td></td><td>.720</td><td>n/a</td><td>n/a</td></tr><tr><td>Global Fisher</td><td>random .6298</td><td></td><td>.731</td><td>.434</td><td>.190</td></tr><tr><td>Routed Fisher</td><td>random .6318</td><td></td><td>.728</td><td>.454</td><td>.190</td></tr><tr><td>Weighted ridge</td><td>random .6334</td><td></td><td>.732</td><td>.730</td><td>.212</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Weighted ridge</td><td>width</td><td>.6299</td><td>.738</td><td>.783</td><td>.227</td></tr></table>

TABLE IV

CLINICIAN RESULTS BY RECORDED CATEGORY AT 25 RESPONSES. FREQUENCY AND RIDGE RECALL USE SHARED RANDOM MASKS; WIDTH USES RIDGE WITH WIDTH ACQUISITION.
<table><tr><td rowspan="2">Rating</td><td colspan="4">Recall (%)</td><td rowspan="2">Ridge / width (%) Coverage Cert. error</td></tr><tr><td>n</td><td>Freq.</td><td>Ridge</td><td>Width</td></tr><tr><td>Absent</td><td>3748</td><td>84.5</td><td>88.8</td><td>89.1</td><td>53.2</td><td>5.1</td></tr><tr><td>Present</td><td>1962</td><td>43.6</td><td>60.8</td><td>67.3</td><td>46.1</td><td>15.5</td></tr><tr><td>Not assessable</td><td>890</td><td>37.3</td><td>26.7</td><td>28.5</td><td>30.3</td><td>60.7</td></tr></table>

Expected gain raises coverage over width in both layperson cohorts without established loss gains. In clinicians, it instead raises loss by .0207 ([.0157,.0260]) and reduces coverage by 1.47 points ([0.66,2.27]). Calibrated and personal response probabilities do not close this gap.

The learned-policy comparison uses separate selection and calibration people. Here clinician width loss is .6594. Learned loss acquisition improves it by .0050 (95% CI [.0022,.0078]), while coverage falls from 50.7% to 49.1%. Terminal loss optimization gives .6577 and 47.1% coverage, without an established loss gain. Neither learned policy establishes improvement in the layperson cohorts.

## F. Stopping at the first certificate

With Observed Fisher, width reduces the mean responses needed per target from 38.07 to 34.75 of 50 in clinicians and from 47.36 to 42.11 of 64 in AnnoMI (Table V). Paired reductions are 3.32 (95% CI [3.18,3.45]) and 5.25 ([4.69,5.93]). However, settling every target needs nearly the full shared history. All retained predictions agree with the actual completed model; their errors remain 22.1% and 32.2%. This evaluates model-level stopping, not clinical workload reduction. The stopping replay evaluated 270,384 records.

TABLE V  
MEAN RATINGS NEEDED TO RETAIN OBSERVED FISHER PREDICTIONS AT THE FIRST CERTIFICATE. THE ALL-TARGET COUNT IS THE LATEST STOPPING POINT AMONG TARGETS SHARING A HISTORY.
<table><tr><td>Cohort</td><td>Order</td><td>Per target</td><td>All targets</td></tr><tr><td>Clinicians  $( H = 5 0 )$ </td><td>Random</td><td>38.07</td><td>49.62</td></tr><tr><td rowspan="3">AnnoMI  $( H = 6 4 )$ </td><td>Width</td><td>34.75</td><td>49.23</td></tr><tr><td>Random</td><td>47.36</td><td>64.00</td></tr><tr><td>Width</td><td>42.11</td><td>64.00</td></tr></table>

The 72-checkpoint audit (506,970 records) found no certified disagreement with completed predictions. Independent NumPy checks reproduced sampled margins within $2 . 2 3 \times 1 0 ^ { - 1 5 }$ , and small-history enumeration checked the bounds and gain scores.

## VI. DISCUSSION

The main methodological finding is that completion auditing can preserve a useful partial prediction rule (RQ1). Observed Fisher and its affine partial control share fitted parameters and certify exactly the same evaluated targets, yet the observedcurvature update predicts better in both audits. This supports separating the choice of partial predictor from the calculation of completion bounds. The completed model must still retain the response structure needed by the bound. Low coverage measures how rarely completion agreement can be certified under unrestricted assignments; it does not measure clinical safety or the probability of a revision.

The predictive evidence depends on the task and comparison (RQ2). Ridge improves familiar-interview prediction but does not establish superiority over frequency on held psychiatric interviews or AnnoMI. Under shared random masks, Observed Fisher has lower loss in both checkpoint audits but certifies fewer targets than ridge. A choice based only on loss would therefore differ from one based only on coverage. The not assessable category and the unsuccessful gate experiment further show why neither aggregate loss nor aggregate coverage is sufficient.

The supporting acquisition analysis exposes the same tension (RQ3): width improves clinician prediction over random, while learned loss acquisition yields a further gain at lower coverage. Full EDDI, DIME, and other reinforcement learning systems remain untested. Confidence selection gives lower error than certification at matched coverage. The faithful counterfactual audit answers whether unavailable history can overturn the current class in the completed model. This describes the fitted predictor, not the causal or clinical reasons for a rating [31], [32]. It neither justifies stopping clinical assessment nor guarantees unchanged intermediate updates.

That auditing role still needs prospective evaluation. The present experiments use retrospective masks, known target cues, and equal query costs. Three simulated psychiatric interviews and counseling demonstrations do not establish performance in patient encounters. AnnoMI broadens source coverage, but only ten experts contribute labels and their interview assignments are confounded. New clinicians and interviews, predictions committed before outcome collection, and measured response burden are needed to assess clinical usefulness.

## VII. CONCLUSION

Completion auditing checks whether missing ratings can overturn the current class in a fixed completed model. Exact bounds for affine response maps leave the partial predictor unchanged. In the clinician and counseling audits, Observed Fisher had lower loss than frequency and its affine partial control, while certifying the same evaluated targets as the latter. Across four datasets, predictive gains and coverage varied by task. Width acquisition increased coverage and reduced the ratings needed per target, but settling all targets required nearly the full shared history. Confidence selection gave lower error at matched coverage for ridge in the three reference cohorts. Certification establishes completion agreement but can preserve incorrect predictions. Prospective studies must establish whether this audit improves a clinical task or reduces response burden before it guides assessment stopping.

## REFERENCES

[1] E. Lenz, J. Naamanka, W. Trabert, R. Bottlender, B. Malchow, A. Meyer Lindenberg, T. Gradinger, and E. Schwarz, “Benchmarking large language models against practicing clinicians on psychopathological assessment,” npj Digital Medicine, vol. 9, art. 518, 2026, doi:10.1038/s41746-026- 02852-7.

[2] M. A. Raju, P. Siddappa, M. S. H. Al Amin, and R. Ma, “Multimodal machine learning in healthcare: A tutorial and review,” Preprints, 2026, doi:10.20944/preprints202512.1445.v2.

[3] M. A. Raju, P. Siddappa, M. S. H. Al Amin, R. Ma, and M. R. Ahammed, “CheXFuse: Guided differential joint attention fusion for robust multimodal chest X-ray classification in data-scarce medical settings,” arXiv:2610.25617, 2026.

[4] M. A. Raju and R. Ma, “CirrGuide: A deep cascaded framework for liver cirrhosis segmentation and severity classification from T2-weighted MRI,” arXiv:2609.14010, 2026.

[5] A. P. Meyer, A. Albarghouthi, and L. D’Antoni, “The dataset multiplicity problem: How unreliable data impacts predictions,” in FAccT, 2023, doi:10.1145/3593013.3593988.

[6] M. Sabanayagam et al., “Exact certification of (graph) neural networks against label poisoning,” in ICLR, 2025.

[7] S. Bian, X. Ouyang, Z. Fan, and P. Koutris, “Naive Bayes classifiers over missing data: Decision and poisoning,” in ICML, PMLR 235, 2024, pp. 3913–3934.

[8] A. Z. Fan and P. Koutris, “Certifiable robustness for nearest neighbor classifiers,” in ICDT, LIPIcs 220, 2022, pp. 6:1–6:20, doi:10.4230/LIPIcs.ICDT.2022.6.

[9] P. Siddappa, M. A. Raju, S. H. Al Amin, and R. Ma, “Alert-list concordance for evaluating imputation strategies in ICU sepsis prediction,” arXiv:2610.22635, 2026.

[10] L. Bertinetto, J. F. Henriques, P. H. S. Torr, and A. Vedaldi, “Metalearning with differentiable closed-form solvers,” in ICLR, 2019.

[11] K. A. Wang, J. Shi, and E. B. Fox, “Test-time regression: A unifying framework for designing sequence models with associative memory,” J. Mach. Learn. Res., vol. 27, no. 169, pp. 1–41, 2026.

[12] G. Bao et al., “FAAST: Forward-only associative learning via closed-form fast weights for test-time supervised adaptation,” arXiv:2605.04651v2, 2026.

[13] J. Harrison, A. Sharma, and M. Pavone, “Meta-learning priors for efficient online Bayesian regression,” in Algorithmic Foundations of Robotics XIII, Springer Proc. Adv. Robot., vol. 14, 2020, pp. 318–337, doi:10.1007/978- 3-030-44051-0\_19.

[14] H. Kim et al., “Attentive neural processes,” in ICLR, 2019.

[15] R. El-Yaniv and Y. Wiener, “On the foundations of noise-free selective classification,” J. Mach. Learn. Res., vol. 11, pp. 1605–1641, 2010.

[16] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in NeurIPS, 2017.

[17] M. Zaffran, A. Dieuleveut, J. Josse, and Y. Romano, “Conformal prediction with missing values,” in ICML, PMLR 202, 2023, pp. 40578– 40604.

[18] D. J. Weiss and G. G. Kingsbury, “Application of computerized adaptive testing to educational problems,” J. Educ. Meas., vol. 21, no. 4, pp. 361–375, 1984, doi:10.1111/j.1745-3984.1984.tb01040.x.

[19] X. Luo, D. Kim, and P. Dickison, “Projection-based stopping rules for computerized adaptive testing in licensure testing,” Appl. Psychol. Meas., vol. 42, no. 4, pp. 275–290, 2018, doi:10.1177/0146621617726790.

[20] C. Ma, S. Tschiatschek, K. Palla, J. M. Hernández-Lobato, S. Nowozin, and C. Zhang, “EDDI: Efficient dynamic discovery of high-value information with partial VAE,” in ICML, PMLR 97, 2019, pp. 4234–4243.

[21] S. Gadgil, I. Covert, and S.-I. Lee, “Estimating conditional mutual information for dynamic feature selection,” in ICLR, 2024.

[22] J. Feng, Q. Li, and Y. Li, “BRiG-AFA: Bellman risk-to-go learning for non-myopic active feature acquisition,” arXiv:2608.02305, 2026.

[23] V. Schütz, H. Wu, R. Rezvan, L. Aronsson, and M. Haghir Chehreghani, “AFABench: A generic framework for benchmarking active feature acquisition,” in ACM SIGKDD, 2026, pp. 9765–9776, doi:10.1145/3770855.3817493.

[24] M. Kopka et al., “The RepVig framework for designing use-case specific representative vignettes and evaluating triage accuracy of laypeople and symptom assessment applications,” Scientific Reports, vol. 14, art. 30614, 2024, doi:10.1038/s41598-024-83844-z.

[25] M. Kopka et al., “Data set supplementing ‘A Use-Case Specific Framework for Designing Representative Vignettes (RepVig) and Evaluating Triage Accuracy of Laypeople and Symptom-Assessment Applications’,” Zenodo, 2024, doi:10.5281/zenodo.12805049.

[26] M. L. Schmieding et al., “Benchmarking triage capability of symptom checkers against that of medical laypersons: Survey study,” J. Med. Internet Res., vol. 23, no. 3, e24475, 2021, doi:10.2196/24475.

[27] E. Lenz et al., “LLMs\_psychopathologies,” public data and code, 2026. Accessed: Oct. 4, 2026. [Online]. Available: https://github.com/42elenz/ LLMs\_psychopathologies

[28] Z. Wu, S. Balloccu, V. Kumar, R. Helaoui, D. Reforgiato Recupero, and D. Riboni, “Creation, analysis and evaluation of AnnoMI, a dataset of expert-annotated counselling dialogues,” Future Internet, vol. 15, no. 3, art. 110, 2023, doi:10.3390/fi15030110.

[29] R. J. Williams, “Simple statistical gradient-following algorithms for connectionist reinforcement learning,” Machine Learning, vol. 8, pp. 229–256, 1992, doi:10.1007/BF00992696.

[30] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in ICML, PMLR 70, 2017, pp. 1321–1330.

[31] C. Rudin, “Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead,” Nature Machine Intelligence, vol. 1, pp. 206–215, 2019, doi:10.1038/s42256-019-0048-x.

[32] M. Ghassemi, L. Oakden-Rayner, and A. L. Beam, “The false hope of current approaches to explainable artificial intelligence in health care,” The Lancet Digital Health, vol. 3, no. 11, pp. e745–e750, 2021, doi:10.1016/S2589-7500(21)00208-9.