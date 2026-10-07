# CONFIDENCE-ORDERING REVERSAL UNDER CONTEXTUAL PRIORS IN NEURAL DECODING

Xinyu Zhang <sup>1,2†</sup> Sichao Liu<sup>1†∗</sup>

<sup>1</sup> KTH <sup>2</sup> Karolinska Institutet

## ABSTRACT

Contextual priors improve neural-to-language decoding by reshaping candidate scores. However, confidence is read from the same reshaped scores, so the errors a prior leaves behind can become more confident with no change in accuracy to reveal it. In this work, we ask how a prior shapes the confidence of these errors, studying speech retrieval on MEG-MASC and MOUS with local decoding scores and a contextual prior combined by additive shallow fusion, and the fused top-two margin as confidence. Among initially incorrect predictions, we find a confidence-ordering reversal: a larger margin makes a repair, an error that fusion corrects, more likely when the correct candidate starts near the top of the local ranking, but less likely when it starts lower. On MEG-MASC, pooled correctness AUROC is 0.87, yet the AUROC separating repairs from residual errors, which fusion leaves uncorrected, falls from 0.70 at initial ranks 2–3 to 0.39 at ranks 21–50. Errors starting beyond rank 20, inside the reversed region, make up 46.6% of all errors after fusion. We propose a score-level account: a repair must first close the correct candidate’s initial deficit, which limits its final margin, whereas a residual error can build a large margin between two incorrect candidates. Through a causal intervention that changes only the fusion weight, we show that the reversal moves to deeper ranks, as the account predicts, and that under a word-level LM prior it keeps moving after accuracy gain peaks, so a weight chosen for accuracy does not settle confidence. Guided by this account, we read local and prior scores separately: read before fusion, the prior’s own scores already separate repairs from residual errors where the fused margin reverses, and estimators built on local and prior scores let a selective decoder answer on 74.5% of windows instead of 56.7%, with 92% of its output sets still containing the correct candidate. Confidence after contextual fusion should retain the local and contextual evidence behind each prediction, not just the fused scores. <sup></sup> Project Page <sup>§</sup> Code

## 1 INTRODUCTION

Contextual priors can improve neural-to-language decoding by supplying linguistic structure and sequential context that local neural predictions can miss (Willett et al., 2023; d’Ascoli et al., 2025). They also change the remaining errors. Recent MEG work reports fluent sentences whose content differs from the intended text when neural evidence is weak (Zhang et al., 2026a), and invasive brain-to-text systems exhibit confident but incorrect neural predictions (Huang et al., 2026). In sustained BCI use, users must identify and correct decoded errors (Speier et al., 2016; Card et al., 2026); when confidence determines which outputs are flagged for review, a confidently scored error can pass unflagged and be left for the user to catch. Evaluation nonetheless centres on decoding accuracy or text similarity (Silva et al., 2024). When confidence guides selective output (Kim et al., 2026), the prior shapes both the prediction and the fused scores used to judge it.

We find that strong overall confidence discrimination can conceal opposite orderings of correction outcomes. We study this in offline speech retrieval (Defossez et al., 2023), retaining a fixed candidate ´ pool and separate local and prior scores (Figure 1). The fused margin, the gap between the two highest fused scores, serves as a confidence measure. Among initially incorrect predictions, repairs become correct after fusion and residual errors remain incorrect. Across MEG-MASC (Gwilliams et al., 2023) and MOUS (Schoffelen et al., 2019), larger margins favour repairs when the correct candidate starts near the top of the local ranking, but residual errors when it starts lower (Figure 2). On MEG-MASC, pooled correctness AUROC is 0.87, yet repair-separation AUROC falls from 0.70 at ranks 2–3 to 0.39 at ranks 21–50. We call this a confidence-ordering reversal. The reversed region is where errors concentrate: residual errors starting beyond rank 20 account for 46.6% of all post-fusion errors on MEG-MASC. The pre-fusion margin favours residual errors in every group, so fusion itself produces the positive ordering near the top. How does a prior make some errors more confident, and what information can still help identify them?

![](images/2c64237251863bec0f549a5249d517ebce7dcbb6454e611936f0eb8466c7ced2.jpg)  
Figure 1: Contextual fusion can make an error more confident. A local encoder scores candidates from the current MEG window, while a contextual prior from preceding windows adjusts these scores through additive fusion with weight $\alpha = 2$ . The prior operates at a coarser granularity, so several candidates share one score. Fusion advances the correct candidate (⋆) from rank 40 to rank 7 without correcting the prediction, while increasing the top-two margin from 0.3 to 0.8. Bracket mark the prior and fused margins. Values are illustrative; real cases are in Table 6.

We trace the reversal to how repairs and residual errors form their margins. A repair must first close the correct candidate’s initial score deficit, so its lead is bounded by the remaining contextual advantage, whereas a residual error can acquire a large lead between two incorrect candidates even as the correct candidate advances. Because this bound follows from additive fusion, it holds wherever a prior is added to local scores, and it predicts that increasing fusion weight extends positive repair ordering toward larger initial deficits.

We test this prediction with a causal intervention that varies only the fusion weight, keeping local and prior scores fixed. Under the history-aggregation prior, built from preceding windows, positive ordering extends from no rank group to ranks 11–20 and to larger score deficits, and the boundary never retreats in participant-bootstrap resamples (Figure 3). Under a word-level teacher-forced language-model prior, it also expands down the initial ranking and continues after net accuracy gain has peaked: accuracy and confidence ordering are two different responses to the same prior.

Grouping errors by initial rank exposes the reversal, and the margin-formation account locates its source: fusion merges local competition and prior support into one margin. Guided by this account, we read the two scores separately: the prior’s own margin separates the outcomes that the fused margin reverses (Figure 2C), and learned estimators built on both recover positive repair ordering. Feature comparisons show that local features improve selective output beyond richer fused-score features, with further gains from prior features. At matched 92% coverage, the full estimator emits on 74.5% of test windows versus 56.7% for the fused margin. Local and prior scores retain information for identifying confident errors that the final margin does not express.

This paper proceeds in three steps.

Revealing where confident errors arise. In two MEG datasets, the fused margin orders repairs and residual errors oppositely by initial rank, even though pooled correctness AUROC stays high.

Explaining and testing causally. A score-level account of margin formation explains one way priors make errors more confident; a causal intervention on the fusion weight alone confirms the predicted extension of positive ordering, which accuracy gain does not track.

Identifying confident errors from retained information. Learned estimates recover positive repair ordering, and local and prior features improve selective output beyond richer fused-score features.

## 2 RELATED WORK

Contextual priors and the errors they leave behind. Contextual priors support BCI communication (Speier et al., 2016; Herff et al., 2015), semantic reconstruction (Tang et al., 2023), and noninvasive decoding with sentence-level context or language-model assistance (d’Ascoli et al., 2025; Levy et al., 2026). When neural evidence is weak, contextual support can dominate the final choice.´ Reviews and attribution studies already separate neural contributions from contextual gains (Silva et al., 2024; Zhang et al., 2026b; Gorenshtein et al., 2026a). Correctness alone does not show how much neural evidence contributed, and the remaining errors take characteristic forms: non-invasive systems can produce fluent but incorrect text (Zhang et al., 2026a), while invasive decoders exhibit overconfident neural predictions (Huang et al., 2026). Evaluating contextual assistance therefore also requires assessing whether its remaining errors are identifiable.

Confidence formation under contextual priors. Confidence already guides selective EEG output (Kim et al., 2026), language-model assistance (Huang et al., 2026), and error detection after P300 fusion (Gorenshtein et al., 2026b). In ASR, however, shallow fusion can improve recognition while degrading confidence (Kannan et al., 2018; Li et al., 2021; Futami et al., 2021). Beam analysis connects confidence to local and sequence-level competition (Jia & Van hamme, 2026), while selective classification reveals group differences concealed by aggregate performance (Jones et al., 2021). These observations share a source: under additive fusion, the prior changes both candidate competi tion and the scores that judge the result. What remains unclear is how confidence orders correction outcomes across groups of initial errors, and how that ordering changes with prior contribution.

Judging outputs and contextual revisions. In deployment, BCI users correct decoded outputs (Card et al., 2026), and shared-control systems arbitrate between neural commands and machine assistance (Lee et al., 2025; Deng et al., 2020). For such decisions, selective prediction and prediction sets formalise accepting, expanding, or withholding outputs (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017; Romano et al., 2020), while adaptive fusion adjusts language-model influence (Variani et al., 2022; Unni et al., 2022; Gong et al., 2022). For contextual revisions, retaining the local prediction provides the alternative, so confidence must support two judgements: final correctness and improvement over that prediction. Applying these decision rules requires knowing which information in local, prior, and fused scores supports each judgement after the prior has acted.

## 3 METHOD

## 3.1 LOCAL SCORES AND CONTEXTUAL FUSION

Following the speech-retrieval framework of Defossez et al. (2023), each query is a word-aligned,´ 3-s MEG window paired with its correct audio segment in a fixed candidate pool $\mathcal { V } = \{ 1 , \ldots , \bar { O } \}$ At window index t, the local decoder maps the neural window $x _ { t }$ to scores $s _ { t } \in \mathbb { R } ^ { O }$ , while the contextual prior maps preceding history $h _ { t - 1 }$ to scores $\pi _ { t }$ over the same candidates. Let $Y _ { t }$ be the correct-candidate random variable and $y _ { t }$ its evaluation label; j and k index candidates. The distributions $q _ { t } = \mathrm { s o f t m a x } ( s _ { t } )$ and $p _ { t } = \operatorname { s o f t m a x } ( \pi _ { t } )$ define the local and prior model probabilities. Additive shallow fusion with weight α ≥ 0 (Kannan et al., 2018) gives

$$
\tilde { s } _ { t , j } = s _ { t , j } + \alpha \pi _ { t , j } ,
$$

$$
\operatorname* { P r } _ { \alpha } ( Y _ { t } = j \mid x _ { t } , h _ { t - 1 } ) = { \frac { q _ { t } ( j ) p _ { t } ( j ) ^ { \alpha } } { \sum _ { k } q _ { t } ( k ) p _ { t } ( k ) ^ { \alpha } } } .\tag{1}
$$

(2)

Equation 2 is the softmax of Eq. 1, so fusion is a tempered product of the two distributions and reduces to the local one at $\alpha = 0 ;$ for $\alpha > 0$ it gives the prior temperature $T _ { \mathrm { p r i o r } } = 1 / \alpha$ while leaving local scores unchanged, so α controls the prior’s contribution relative to neural evidence (Appendix A). The local and fused predictions are $\hat { y } _ { t } = \arg \operatorname* { m a x } _ { j } s _ { t , j }$ and $\tilde { y } _ { t } = \arg \operatorname* { m a x } _ { j } \tilde { s } _ { t , j }$

For a score vector a with largest entries $a _ { ( 1 ) } \geq a _ { ( 2 ) }$ , its top-two margin is $m ( a ) = a _ { ( 1 ) } - a _ { ( 2 ) }$ equal to the log odds between its two leading softmax candidates. The pre-fusion margin is $m ( s _ { t } )$

and the fused margin is $m ( \tilde { s } _ { t } )$ . The prior margin is computed over the prior’s own entries, distinct sentences or words, before their scores are shared by the candidates that map to them; entries with equal scores remain separate. All three margins are evaluated by how they order outcomes.

## 3.2 CORRECTION OUTCOMES, ORDERING, AND REACH

Among initially incorrect predictions $( { \hat { y } } _ { t } \ \neq \ y _ { t } )$ , a repair has $\tilde { y } _ { t } ~ = ~ y _ { t }$ and a residual error has $\tilde { y } _ { t } \ne y _ { t } ;$ among initially correct predictions, a regression has $\tilde { y } _ { t } \ne y _ { t }$ . Let $n _ { \mathrm { r e p a i r } }$ and $n _ { \mathrm { r e g r e s s i o n } }$ count repairs and regressions over N windows, so that retrieval accuracy changes by $\Delta { \mathrm { R @ 1 } } =$ $( n _ { \mathrm { r e p a i r } } - n _ { \mathrm { r e g r e s s i o n } } ) / N$

The correct candidate’s initial rank is $r _ { t } ^ { 0 } = 1 + \left| \{ j : s _ { t , j } > s _ { t , y _ { t } } \} \right|$ , the number of candidates scoring strictly above it plus one. We compare repairs with residual errors in fixed groups B of initially incorrect predictions, by the correct candidate’s initial rank: 2–3, 4–5, 6–10, 11–20, and $^ { 2 1 - 5 0 , }$ adding 51–100 for the word-level sweeps. Within $B ,$ draw indices r and e uniformly from its repairs and residual errors. Repair-separation AUROC measures whether confidence z ranks the repair above the residual error:

$$
\begin{array} { r } { \mathrm { A U R O C } _ { \mathrm { r e p a i r } , \mathcal { B } } ( z ) = \operatorname* { P r } ( z _ { r } > z _ { e } ) + \frac { 1 } { 2 } \operatorname* { P r } ( z _ { r } = z _ { e } ) . } \end{array}\tag{3}
$$

Values above 0.5 favour repairs; values below 0.5 favour residual errors. We call these positive and negative ordering. Pooled correctness AUROC applies the same expression with r and e drawn from all final correct and all final incorrect predictions. Repair rates across margin quartiles within each group give a complementary view of this ordering. Effective correction reach is the 90th percentile of the initial ranks among repairs:

$$
R _ { 9 0 } = Q _ { 0 . 9 0 } \big ( r _ { t } ^ { 0 } \mid \hat { y } _ { t } \neq y _ { t } , \tilde { y } _ { t } = y _ { t } \big ) .\tag{4}
$$

Reach summarises the repair distribution independently of confidence scores. Each AUROC curve is described by its deepest positively ordered initial-rank group. For a single downward crossing of 0.5, this locates the reversal boundary; Figure 3C also interpolates the crossing between group midpoints. Full curves describe settings with no observed crossing or multiple sign changes.

## 3.3 DATASETS AND CONTEXTUAL PRIORS

We evaluate MEG-MASC and MOUS (Gwilliams et al., 2023; Schoffelen et al., 2019) with three independently trained local decoders per dataset (Table 1). Their neural encoder maps each wordaligned MEG window to a time-resolved speech representation, and candidate scores are inner products with frozen wav2vec2-large-xlsr-53 audio representations (Conneau et al., 2021). Stimulusidentity splits evaluate zero-shot retrieval of segments unseen during local-decoder training. Both datasets support the reversal and prior-margin analyses; controlled sweeps, cross-intervention comparisons, and selective-output evaluation use MEG-MASC (Appendix B).

Table 1: MEG datasets and retrieval accuracy with the history-aggregation prior at $\alpha = 2 . 7 .$
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Language</td><td rowspan="2">Participants</td><td rowspan="2">Sensors</td><td rowspan="2">Hours</td><td rowspan="2">Candidates</td><td colspan="2">R@1(%)</td></tr><tr><td>Local</td><td>Fused</td></tr><tr><td>MEG-MASC</td><td>English</td><td>27</td><td>208</td><td>56.2</td><td>1,464</td><td>44.42</td><td>58.82</td></tr><tr><td>MOUS</td><td>Dutch</td><td>96</td><td>272</td><td>80.9</td><td>825</td><td>21.72</td><td>29.98</td></tr></table>

Contextual priors differ in three respects that our interventions vary separately: the source of history (true preceding words, the decoder’s own predictions, or accumulated local scores), the granularity at which prior scores are assigned (sentence or word, shared by all candidates that map to the same entry), and the form of the update (additive fusion or non-additive pruning). Every prior shares one property: candidates that map to the same entry receive the same prior score, so fusion cannot reorder them and the prior acts on competition between entries while local scores act within one.

Our main setting uses a history-aggregation prior that accumulates evidence only from preceding windows. Candidates from the same source sentence form a bucket ${ \mathcal { C } } _ { b } ,$ and stimulus timing identifies query sentence presentations and word onsets. Within each presentation, the prior integrates localscore support from earlier windows in word-onset order to infer which bucket matches the query.

With $b ( j )$ identifying candidate $j ^ { \dagger } \mathbf { s }$ bucket, $u _ { t }$ the vector of bucket-size-normalised support from local scores, $H _ { t }$ the accumulated bucket scores, and $\eta$ the update rate,

$$
H _ { t } = ( 1 - \eta ) H _ { t - 1 } + \eta u _ { t } , \qquad \pi _ { t , j } = \mathcal { L } ( H _ { t - 1 } , b ( j ) ) ,\tag{5}
$$

where $\mathcal { L }$ retains up to the four strongest active bucket scores and broadcasts each to that bucket’s candidates, giving the prior score vector $\pi _ { t } ^ { \mathrm { b u c k e t } }$ over distinct buckets. Prediction reads $H _ { t - 1 }$ before the current update; state resets at each presentation, and validation selects $\alpha = 2 . 7$ and $\eta = 0 . 1$

Language-model (LM) priors replace accumulated support with Qwen3-4B-Base likelihoods (Yang et al., 2025) at the sentence level (mean per-token log-likelihood per candidate sentence) or the word level (log probability of each candidate’s aligned word, shared by candidates carrying that word). Each comes in two history sources: free-running priors condition on preceding predicted words, and teacher-forced priors condition on true preceding words, probing correction when the history is right. For the word-level free-running prior, we generate history once from the frozen decoder’s top-1 predictions, yielding one prior-score vector per decoder for the whole sweep. Hard bucket pruning changes the update form: it uses the history-aggregation state to retain four buckets and discard the rest, without adding scores. Appendix C gives the constructions and validation selection.

Sweeps vary only α, holding both score streams, history, candidates, and test windows fixed: eight weights from 0.5 to 4.0 for history aggregation, sixteen from 0.05 to 16 for each word-level LM prior. Every weight determines its own repairs and residual errors, against which all three margins are evaluated; cross-intervention comparisons use validation-selected operating points.

## 3.4 CONFIDENCE ESTIMATION AND EVALUATION

The analyses above use the correct candidate to label outcomes. A system predicting without it must judge two things: whether the output is correct, and whether the prior’s revision improved on the local prediction. Output confidence targets the first, including whether an output set contains the correct candidate. Intervention confidence targets $B ( I _ { t } )$ , the expected benefit of contextual revision over the local prediction given observable information $I _ { t }$ . Under top-1 error loss,

$$
\begin{array} { r l } & { B ( I _ { t } ) = \operatorname* { P r } ( \tilde { y } _ { t } = y _ { t } \mid I _ { t } ) - \operatorname* { P r } ( \hat { y } _ { t } = y _ { t } \mid I _ { t } ) } \\ & { \qquad = \operatorname* { P r } ( \operatorname { r e p a i r } \mid I _ { t } ) - \operatorname* { P r } ( \operatorname { r e g r e s s i o n } \mid I _ { t } ) . } \end{array}\tag{6}
$$

All probabilities use the same conditioning information. A repair and a prediction that remains correct share the final-correctness label, but only the repair improves on the local prediction. Among initially incorrect windows, final correctness is repair success; across all windows, $B ( I _ { t } )$ also depends on preserving correct local predictions.

For output confidence, multilayer perceptrons predict fused top-K success for $K \in \{ 1 , 3 , 5 , 1 0 , 2 0 \}$ Decoder-score estimators combine two local and six fused score features; prior-feature estimators use seven features of prior state, current support, and sentence position; the full estimator combines both. Additional estimators use only the fused margin, the six fused features, or the two local features, with the same hidden-layer architecture and training procedure. Training and calibration use validation data, with cross-validation folds formed by presentation.

These estimates drive an output policy, which selects the smallest K meeting its validation-selected success-probability threshold, or abstains if none qualifies. Emission is the fraction of windows on which a set is output, and coverage is the fraction of emitted sets containing the correct candidate. Validation requires at least 90% coverage and mean emitted set size at most 4.8. We compare decoder-score and full estimators with a margin-binned reference that applies conformal set calibration (Romano et al., 2020) in 20 fused-margin bins, reporting emission rate, mean set size, and coverage, also grouped by initial rank (Appendix I.3).

To compare selection at a fixed output size, fixed-K policies emit their top-K set or abstain, with validation thresholds maximising emission subject to at least 90% coverage (Appendix I.3).

Throughout, estimates are means and standard deviations across the three local decoders, with reach computed within each decoder before averaging; we report a repair-separation cell only when every decoder has at least five repairs, and initial rank serves only as an evaluation label. Uncertainty is quantified at three levels: Wilson intervals describe the quartile repair rates in Figure 2A, B, which pool evaluated predictions with fixed quartile membership; sentence-cluster bootstrap supports rate contrasts and within-sentence comparisons; and participant bootstrap, sharing each resampled participant set across weights, groups, confidence measures, and decoders, gives the core AUROC intervals and boundary stability (Appendix D).

## 4 RESULTS

## 4.1 THE FUSED MARGIN REVERSES ITS ORDERING ACROSS INITIAL RANKS

Confidence ordering reverses as the correct candidate starts lower in the local ranking. Pooled correctness AUROC is 0.872 on MEG-MASC (Figure 2), yet among initially incorrect predictions, repair-separation AUROC falls from 0.704 at ranks 2–3 to 0.392 at ranks 21–50. Pooled over all initially incorrect predictions, AUROC is 0.646, so grouping by initial rank exposes opposite orderings that this pooled comparison does not reveal. For the same outcomes, the pre-fusion margin remains negatively ordered across all five groups, with AUROCs of 0.30–0.33. Fusion changes the ordering to positive near the top, while deeper groups remain negative. Repairs concentrate at low initial ranks, leaving a substantial share of errors deeper in the ranking: on MEG-MASC, residua errors starting beyond rank 20 account for 46.6% of all post-fusion errors.

A. Before fusion

![](images/7ad0dc6e63e6aa3565767e35c8725f20c5de20416043af3ce9fce916d0bf87be.jpg)

![](images/619fa5277ceaabc5255938dff5b3c07b27146284450750c672008c0c85242b41.jpg)  
C. All initial-rank groups

![](images/59e565605543da96338049cd2ae721f515b2ec021a09a0a1afb6390b03a31833.jpg)  
Figure 2: Confidence ordering reverses as the correct candidate starts lower in the local ranking. Initially incorrect windows on MEG-MASC (filled) and MOUS (hollow). Panels A and B use the same windows and outcomes, with one point per quartile at its mean margin; the upper and lower rows show initial ranks 2–3 and 21–50, respectively, with different vertical scales. (A) Before fusion. Repair rate decreases with margin in both groups. (B) After fusion. From the smallest to largest quartile, repair rate rises at ranks 2–3 and falls at ranks 21–50. (C) All initial-rank groups. Repair-separation AUROC of the fused, pre-fusion, and prior margins; above 0.5 favours repairs, below 0.5 residual errors. Bands are Wilson 95% intervals in A, B, and ±1 SD across three local decoders in C; dashed curves interpolate quartile points. MOUS ranks 21–50 average 13 repairs per decoder; Appendix E reports full rates and participant-level intervals.

Repair rates across margin quartiles show the same reversal. From the smallest to largest fused margin quartile on MEG-MASC, repair rate rises from 42.4% to 84.4% at initial ranks 2–3, but falls from 1.37% to 0.55% at ranks 21–50. Sentence-cluster bootstrap supports both endpoint contrasts (Appendix E). Rates also rise at ranks 4–5 and fall at ranks 11–20 under this contrast; at ranks 6–10, the interval for the highest-to-lowest quartile ratio includes one.

The reversal replicates on MOUS and at the participant level. There, repair-separation AUROC falls from 0.62 at ranks 2–3 to 0.41 at ranks 21–50, and participant-level intervals support positive ordering near the top and negative ordering further down on both datasets. On MOUS, the 6–10 and 11–20 intervals lie below 0.5, while ranks 21–50 average 13 repairs per decoder and have an interval crossing 0.5 (Appendix E.4).

The reversal persists within sentences, among repairable errors, and for other confidence scores. Negative fused-margin ordering persists within sentence presentations and among errors whose correct candidate already leads its sentence bucket, the subset an additive prior can in principle repair (Table 5; Appendix G.3). Entropy, sentence-score mass gap, and concentration also order repairs negatively (0.36, 0.46, 0.43 at ranks 21–50; Appendix I.2).

## 4.2 A SCORE-LEVEL ACCOUNT OF MARGIN FORMATION

A prior can make an error more confident while moving its correct candidate up. In Table $^ { 6 , }$ the prior advances the correct candidate from rank 40 to rank 7 while the margin grows from 0.07 to 3.54 around an incorrect winner. We account for this by comparing how repairs and residual errors form their final leads.

A repair’s margin shrinks as the deficit it must close grows. Let $G _ { t } ^ { 0 } = \operatorname* { m a x } _ { j \neq y _ { t } } s _ { t , j } - s _ { t , y _ { t } }$ be the correct candidate’s initial score deficit and $\begin{array} { r } { \Delta _ { t } ^ { \pi } = \operatorname* { m a x } _ { j } \pi _ { t , j } - \operatorname* { m i n } _ { j } \pi _ { t , j } } \end{array}$ the prior score range. On a repair, comparison with the strongest local competitor gives:

$$
G _ { t } ^ { 0 } \leq \alpha \Delta _ { t } ^ { \pi } ,\tag{7}
$$

$$
m ( \tilde { s } _ { t } ) \leq \alpha \Delta _ { t } ^ { \pi } - G _ { t } ^ { 0 } .\tag{8}
$$

Increasing α relaxes both constraints, allowing repairs to close larger deficits with larger margins.

Deep residual errors gain margin between two wrong candidates. With post-fusion score deficit $\begin{array} { r } { \tilde { G } _ { t } = \operatorname* { m a x } _ { j \neq y _ { t } } \tilde { s } _ { t , j } - \tilde { s } _ { t , y _ { t } } } \end{array}$ , the fused margin equals $- \tilde { G } _ { t }$ on a repair and is at most $\tilde { G } _ { t }$ on a residual error. In the latter case, equality holds when the correct candidate is runner-up, in the absence of ties. The correct candidate’s runner-up share among residual errors falls from 75.2% at initial ranks 2–3 to 2.1% at ranks 21–50 (Table 7). The margin at deep initial ranks therefore mostly compares two incorrect candidates, which can separate even while the correct candidate remains far behind; this is one way fusion leaves an error uncorrected yet more confident.

The bound certifies most reversals under history aggregation but few under the word-level LM. Negative ordering follows when a residual-error margin exceeds a repair’s upper bound in more than half of within-group pairs (Appendix A). This range-based condition certifies 16 of 21 negative cells (weight and rank-group pairs) under history aggregation and 1 of 29 under the wordlevel teacher-forced prior, whose full score range can greatly exceed the correct candidate’s prior advantage over the strongest local competitor. $\mathrm { { A t } } \alpha = 0 . 5$ under this prior, measured AUROC falls from 0.674 at ranks 2–3 to 0.419 at ranks 21–50, while the range bound does not certify this negative ordering (Appendix F.3), so the bound alone does not imply the reversal.

These constraints motivate the hypothesis that increasing the prior’s contribution extends positive repair ordering towards larger initial score deficits. We examine this response by varying the fusion weight while holding both score streams fixed.

## 4.3 CONTROLLED CHANGES IN PRIOR CONTRIBUTION MOVE CONFIDENCE ORDERING

We vary only the fusion weight, keeping local and prior scores fixed. Predictions and outcome labels are recomputed at each weight. Effective correction reach summarises where the resulting repairs begin. At the main history-aggregation setting, 84.8% of repairs start within the top five ranks and 96.3% within the top ten, giving $R _ { 9 0 } = 7 ~ ( \mathrm { F i g u r e } ~ 3 )$

Raising the history-aggregation weight pushes positive ordering to deeper ranks. Reach grows from 3.0 to 9.3, and the deepest positively ordered group advances from none through 2–3, 4–5, and 6–10 to 11–20; every increase preserves or advances it, and the sequence is non-decreasing in all 1,000 participant-bootstrap resamples (Appendix G.1). On the same outcome labels, the pre-fusion margin stays negatively ordered in all 32 reported cells (0.111–0.378).

Positive ordering also extends to larger initial score deficits. In five equal-mass deficit groups fixed across weights, the deepest positively ordered group advances from none to the fourth group. The 90th percentile of initial deficits among repairs grows from 0.61 to 3.24 score units, while the fifth group remains negatively ordered at $\alpha = 4 . 0$ (Appendix G.2). Thus positive ordering expands in both initial-rank and score-deficit coordinates under this prior.

Under a teacher-forced word-level LM prior, positive ordering continues to expand after accuracy gain peaks. Between $\alpha = 0 . 0 5$ and 1, reach grows from 3.0 to 19.3, and positive ordering extends from no group to every observed group through rank 100. Validation selects $\alpha = 0 . 5$ for accuracy; there, positive ordering reaches ranks 6–10, and the gain is 11.20 percentage points. At α = 1, positive ordering extends further, but the gain falls to 9.16 percentage points.

A. Fusion-weight sweep  
![](images/9d26ce97ebd1ae66c0cb6e2737ac7f74f705614c6b502dad0ffc1f67ff085329.jpg)

![](images/bc85e6b4af23bde7939608e070303dbf960d26b12615e42298df109a280030e9.jpg)

![](images/016dbb8fb5969614d67f9826ac5bf9cdd1e5d69ed7fda66ea150363eb893a60a.jpg)  
Figure 3: Changing prior contribution moves the confidence-ordering structure. MEG-MASC. His-agg.: history aggregation; TF/FR-LM: teacher-forced/free-running LM prior. (A) Fusionweight sweep. Six weights from the eight-setting history-aggregation sweep, with local and prior scores fixed; grey shows pre-fusion-margin AUROC on the same outcomes. (B) Contextual interventions. Six validation-selected settings: filled markers are sentence-level interventions, hollow markers and dashed curves are word-level LM priors; hard bucket pruning is non-additive. (C) Reversal boundary tracks reach. Arrows follow increasing α for history aggregation and the word-level teacher-forced prior, the latter through $\alpha = 0 . 7 5 ;$ at higher weights all observed groups through rank 100 are above chance. Open markers correspond to B; the bottom band denotes no positive ordering. Crossings interpolate between group midpoints; the dashed log–log fit summarises the displayed association. Full curves, boundaries, and event counts are in Appendix G.

With predicted history, positive ordering appears only after accuracy gain turns negative. At $\alpha = 0 . 1 5$ , the word-level free-running prior has reach 3.0 and gain 1.25 points, with no positive group; at $\alpha = 0 . 5$ , AUROC is 0.539 at ranks 2–3 and below 0.5 elsewhere, while gain is −0.69 points. The reversal therefore appears with predicted history as well, although positive ordering emerges only after gain turns negative. At larger weights, sign changes multiply and the deepest positive group retreats even as reach grows (Appendix G.4).

Priors with similar accuracy gains can differ in confidence ordering. Every prior in Section 3 shows negative repair ordering, but its extent varies. The sentence-level teacher-forced LM prior and the history-aggregation prior improve accuracy by 14.74 and 14.40 percentage points, respectively, yet retain positive ordering through ranks 11–20 and 4–5. Correction reach captures another aspect of this variation: hard bucket pruning and the sentence-level teacher-forced prior have similar reaches of 15.3 and 15.7, and both retain positive ordering through ranks 11–20. Similar reach can also accompany different ordering boundaries: history aggregation at α = 4.0 reaches 9.3 with positive ordering through ranks 11–20, whereas the word-level teacher-forced prior at $\alpha = 0 . 5$ reaches 10.3 with positive ordering through ranks 6–10. Prior construction thus shapes both the extent of correction and the ordering of its outcomes.

Nearly half of the remaining errors start beyond rank 20, inside the reversed region; since fusion mixes local and prior support into one margin, we read these two scores separately to identify these errors.

## 4.4 OBSERVABLE INFORMATION FOR CONFIDENCE ESTIMATION

Read before fusion, the prior’s own margin separates the outcomes that the fused margin reverses. The final margin records the winner’s lead without distinguishing how local competition and prior support produced it. On the same repairs and residual errors, the prior margin gives 0.937 at ranks 21–50 against 0.392 for the fused margin; under history aggregation it ranges from 0.819 to 0.937 across the five MEG-MASC groups and from 0.84 to 0.99 on MOUS (Figure 2C), stays above 0.7 in all 32 cells of the fusion-weight sweep while the fused margin reverses (Appendix G.1), and among initially correct MEG-MASC predictions it separates those that remain correct from regressions with AUROC 0.777 ± 0.022. Word-level LM priors discriminate less: 0.568–0.759 for

Table 2: Confidence estimation and selective output with the history-aggregation prior on MEG-MASC; means over three local decoders, with estimator seeds averaged. (a) Top-1 repair and top-5 inclusion AUROC (Table 20). (b) Emission at matched 92.24% test-curve coverage; K is set size.  
(a) Confidence-estimation AUROC
<table><tr><td>Initial</td><td>1 Success</td><td>Decoder-</td><td>Prior-</td></tr><tr><td>rank</td><td>target</td><td>score</td><td>feature</td></tr><tr><td rowspan="2">11-20</td><td>Top-1</td><td>0.716</td><td>0.848</td></tr><tr><td>Top-5</td><td>0.552</td><td>0.752</td></tr><tr><td rowspan="2">21-50</td><td>Top-1</td><td>0.653</td><td>0.910</td></tr><tr><td> $\mathrm { T o p } { \cdot } 5$ </td><td>0.522</td><td>0.867</td></tr></table>

(b) Selective output
<table><tr><td>Features</td><td>Dim.</td><td>Emission (%)</td><td>Mean K</td></tr><tr><td>Fused margin</td><td>1</td><td> $\overline { { 5 6 . 7 \pm 0 . 5 } }$ </td><td>6.3</td></tr><tr><td>Fused scores</td><td>6</td><td> $6 6 . 6 \pm 0 . 7$ </td><td>5.8</td></tr><tr><td>Decoder scores</td><td>8</td><td> $7 2 . 5 \pm 0 . 5$ </td><td>5.2</td></tr><tr><td>Full</td><td>15</td><td> $7 4 . 5 \pm 0 . 4$ </td><td>5.2</td></tr></table>

repairs and 0.545 for preservation with teacher-forced history (α = 0.5), and 0.532–0.701 (ranks 2–10) and 0.532 with predicted history $( \alpha = 0 . 1 5 ;$ Tables 17 and 18).

Learned estimators rank repairs above residual errors where the fused margin reverses. Under history aggregation, decoder-score and prior-feature estimators reach repair-separation AUROCs of 0.653 and 0.910 at ranks 21–50 (Table 2(a)). Among the 10% of these deep errors scored highest by a prior-state estimator (three prior features), repairs occur at 8.40%, against 1.80% for decoder scores and a base rate of 1.05% (Appendix I.2). For top-five inclusion, prior-feature discrimination remains strong while decoder-score discrimination is near chance. Across all test windows, the full estimator achieves a correctness AUROC of 0.954, compared with 0.872 for the fused margin, and remains unchanged under any strictly increasing recalibration (Appendices A and I.2).

Retained scores let selective output emit more, including where errors concentrate. Emission at matched 92.24% test-curve coverage (Appendix I.4) rises from 56.7% with the fused margin to 66.6% with six fused features, then 72.5% and 74.5% as local and prior features are added (Table 2(b)). For validation-selected policies, adding prior features at ranks 21–50 raises emission from 32.5% to 38.2% and coverage from 36.4% to 50.8%. At output size one, the full estimator emits at 58.3% with 90.5% coverage, versus 48.7% and 89.1% for margin binning (Appendix I.3).

## 5 DISCUSSION

A contextual prior sets both which prediction wins and how confident its errors look. Larger fused margins make repairs more likely near the top of the local ranking but less likely further down, where nearly half of remaining errors start, while pooled AUROC stays high. Raising the fusion weight moves this reversal to deeper ranks, and under a word-level LM prior it keeps moving after accuracy gain peaks. The prior’s own margin, read before fusion, separates repairs from residual errors.

The prior can widen the gap between wrong candidates as easily as it lifts the right one. It adds to every margin only a weighted difference of its own scores, whether or not that difference favours the correct candidate. Confidence ordering therefore follows how the prior reshapes competition, not how much it improves accuracy, and the repair-margin bound holds for any additive fusion. Seeing this requires the local scores, which links confidence analysis to source attribution (Zhang et al., 2026b): identifying what context changed precedes judging whether the change succeeded.

Retained scores restore what recalibration cannot. Recalibrating the margin preserves its ordering, so no calibration can undo the reversal, and other summaries of the fused scores, such as entropy, reverse as well (Section 4.1); keeping local and prior scores separate restores the ordering (Table 2). A weight chosen for accuracy does not determine confidence, and we can judge a prior by how well its retained scores reveal its correction outcomes, alongside its accuracy gain.

Limitations and future work. Our offline retrieval setting keeps the candidate pool fixed and score streams separate, which makes each correction traceable and the prior’s contribution controllable. Free-running histories come from local top-1 rather than fused predictions; teacher-forced histories bracket what richer histories, such as N-best predictions, would provide. Generated candidates would add a changing pool and feedback from fused outputs, and fixed-set assistive interfaces such as BCI spellers (Speier et al., 2016) offer a direct test of revision benefit. Our estimators target final correctness; estimating revision benefit, and training objectives that make successful corrections identifiable, are natural next steps. Confidence estimation after contextual fusion should therefore retain the local evidence and contextual support behind each prediction, rather than relying on a summary of the fused scores alone.

## ACKNOWLEDGEMENTS

The authors acknowledge support from the Vetenskapsradet under award 2023-00493, the NAISS˚ 2025/22-1173, 2025/23-185, and 2026/3-376.

## AI USE STATEMENT

The authors used ChatGPT and Claude to polish writing and LaTeX formatting. The authors produced and verified all analyses, results, and claims and take responsibility for the final content.

## REFERENCES

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 2020.

Nicholas S Card, Tyler Singer-Clark, Hamza Peracha, Carrina Iacobacci, Xianda Hou, Maitreyee Wairagkar, Zachery Fogg, Elena C Offenberg, Leigh R Hochberg, Sergey D Stavisky, et al. Longterm independent use of an intracortical brain–computer interface for speech and cursor control. Nature Medicine, 2026.

Alexis Conneau, Alexei Baevski, Ronan Collobert, Abdelrahman Mohamed, and Michael Auli. Unsupervised cross-lingual representation learning for speech recognition. In Proc. Interspeech, 2021.

Stephane d’Ascoli, Corentin Bel, J´ er´ emy Rapin, Hubert Banville, Yohann Benchetrit, Christophe´ Pallier, and Jean-Remi King. Towards decoding individual words from non-invasive brain record-´ ings. Nature Communications, 16(1):10521, 2025. doi: 10.1038/s41467-025-65499-0.

Alexandre Defossez, Charlotte Caucheteux, J´ er´ emy Rapin, Ori Kabeli, and Jean-R´ emi King. De-´ coding speech perception from non-invasive brain recordings. Nature Machine Intelligence, 5 (10):1097–1107, 2023.

Xiaoyan Deng, Zhu Liang Yu, Canguang Lin, Zhenghui Gu, and Yuanqing Li. A Bayesian shared control approach for wheelchair robot with brain machine interface. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2020.

Bradley Efron and Robert J Tibshirani. An introduction to the bootstrap. Chapman and Hall/CRC, 1994.

Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal of Machine Learning Research, 11(53):1605–1641, 2010.

Hayato Futami, Hirofumi Inaguma, Masato Mimura, Shinsuke Sakai, and Tatsuya Kawahara. ASR rescoring and confidence estimation with ELECTRA. In 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2021.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30, 2017.

Zhuo Gong, Daisuke Saito, and Nobuaki Minematsu. Entropy-based dynamic rescoring with language model in E2E ASR systems. Applied Sciences, 12(19):9690, 2022.

Alon Gorenshtein, Mahmud Omar, Eric L Jia, Yosef Adiniaev, Oved Daniel, Jonathan Kruskal, Muneeb Ahmed, Olga R Brook, Yiftach Barash, and Eyal Klang. Quantifying large language model influence in brain computer interface communication for amyotrophic lateral sclerosis. medRxiv, 2026a.

Alon Gorenshtein, Mahmud Omar, Eric L Jia, Yosef Adiniaev, Oved Daniel, Jonathan Kruskal, Muneeb Ahmed, Olga R Brook, Eyal Klang, and Yiftach Barash. Limits of trial-adaptive neural language fusion across large language models in P300 brain computer interfaces. medRxiv, 2026b.

Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70, 2017.

Laura Gwilliams, Graham Flick, Alec Marantz, Liina Pylkkanen, David Poeppel, and Jean-R¨ emi´ King. Introducing MEG-MASC: A high-quality magneto-encephalography dataset for evaluating natural speech processing. Scientific Data, 10(1):862, 2023.

Christian Herff, Dominic Heger, Adriana De Pesters, Dominic Telaar, Peter Brunner, Gerwin Schalk, and Tanja Schultz. Brain-to-text: decoding spoken phrases from phone representations in the brain. Frontiers in Neuroscience, 9:217, 2015.

Jingya Huang, Sowmya Manojna Narasimha, Aashish N Patel, Ram Dyuthi Sristi, Gal Mishne, and Vikash Gilja. Probabilistic co-control in brain-computer interfaces: Uncertainty as a control signal in brain-to-text decoding. bioRxiv, 2026.

Yichen Jia and Hugo Van hamme. Leveraging beam search information for confidence estimation in E2E ASR. IEEE Open Journal ofSignal Processing, 7, 2026.

Erik Jones, Shiori Sagawa, Pang Wei Koh, Ananya Kumar, and Percy Liang. Selective classification can magnify disparities across groups. In International Conference on Learning Representations, 2021.

Anjuli Kannan, Yonghui Wu, Patrick Nguyen, Tara N. Sainath, Zhifeng Chen, and Rohit Prabhavalkar. An analysis of incorporating an external language model into a sequence-to-sequence model. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.

Soowon Kim, Byung-Kwan Ko, and Seo-Hyun Lee. Confidence-aware neural decoding of overt speech from EEG: Toward robust brain–computer interfaces. In 2026 14th International Conference on Brain-Computer Interface (BCI), 2026.

Johannes Y Lee, Sangjoon Lee, Abhishek Mishra, Xu Yan, Brandon McMahan, Brent Gaisford, Charles Kobashigawa, Mike Qu, Chang Xie, and Jonathan C Kao. Brain–computer interface control with artificial intelligence copilots. Nature Machine Intelligence, 2025.

Jarod Levy, Mingfang Zhang, Svetlana Pinet, J ´ er´ emy Rapin, Hubert Banville, St ´ ephane d’Ascoli,´ and Jean-Remi King. Noninvasive decoding of typed sentences from human brain activity.´ Nature Neuroscience, 2026.

Qiujia Li, David Qiu, Yu Zhang, Bo Li, Yanzhang He, Philip C Woodland, Liangliang Cao, and Trevor Strohman. Confidence estimation for attention-based sequence-to-sequence models for speech recognition. In 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.

Yaniv Romano, Matteo Sesia, and Emmanuel J. Candes. Classification with valid and adaptive\` coverage. In Advances in Neural Information Processing Systems, volume 33, 2020.

Jan-Mathijs Schoffelen, Robert Oostenveld, Ngoc H. L. Lam, Julia Udden, Annika Hult ´ en, and Peter´ Hagoort. A 204-subject multimodal neuroimaging dataset to study language processing. Scientific Data, 6(1):17, 2019. doi: 10.1038/s41597-019-0020-y.

Alexander B Silva, Kaylo T Littlejohn, Jessie R Liu, David A Moses, and Edward F Chang. The speech neuroprosthesis. Nature Reviews Neuroscience, 2024.

William Speier, C Arnold, and Nader Pouratian. Integrating language models into classifiers for BCI communication: a review. Journal ofNeural Engineering, 13(3):031002, 2016.

Jerry Tang, Amanda LeBel, Shailee Jain, and Alexander G. Huth. Semantic reconstruction of continuous language from non-invasive brain recordings. Nature Neuroscience, 26(5):858–866, 2023.

Vinit Unni, Shreya Khare, Ashish Mittal, Preethi Jyothi, Sunita Sarawagi, and Samarth Bharadwaj. Adaptive discounting of implicit language models in RNN-transducers. In 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

Ehsan Variani, Michael Riley, David Rybach, Cyril Allauzen, Tongzhou Chen, and Bhuvana Ramabhadran. On adaptive weight interpolation of the hybrid autoregressive transducer. In Proc. Interspeech 2022, 2022.

Francis R. Willett, Erin M. Kunz, Chaofei Fan, Donald T. Avansino, Guy H. Wilson, Eun Young Choi, Foram Kamdar, Matthew F. Glasser, Leigh R. Hochberg, Shaul Druckmann, et al. A high performance speech neuroprosthesis. Nature, 620(7976):1031–1036, 2023.

Edwin B Wilson. Probable inference, the law of succession, and statistical inference. Journal ofthe American Statistical Association, 1927.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Bianca Zadrozny and Charles Elkan. Transforming classifier scores into accurate multiclass probability estimates. In Proceedings ofthe eighth ACM SIGKDD international conference on Knowledge discovery and data mining, 2002.

Mingfang Zhang, Jarod Levy, Cedric Rommel, J´ er´ emy Rapin, Corentin Bel, Julie Bonnaire, Daniel´ Nieto, Pierre Bourdillon, Svetlana Pinet, Stephane d’Ascoli, et al. Accurate decoding of natural´ sentences from non-invasive brain recordings. arXiv preprint arXiv:2608.18114, 2026a.

Xinyu Zhang, Sichao Liu, Runhao Lu, Alexandra Woolgar, and Lihui Wang. What are we actually decoding? source attribution for non-invasive brain-to-language retrieval. arXiv preprint arXiv:2605.24524, 2026b.

## A FORMAL PROPERTIES OF CONTEXTUAL FUSION

Equation 2 multiplies local candidate odds by the corresponding prior odds raised to α. For $\alpha > 0$ this gives the prior temperature $1 / \alpha$ , independently of calibration.

## A.1 BOUNDED CORRECTIVE CAPACITY

Let $j ^ { * } = \arg \operatorname* { m a x } _ { j \neq y _ { t } } s _ { t , j }$ be the strongest local competitor on an initially incorrect window. Repair requires

$$
G _ { t } ^ { 0 } = s _ { t , j ^ { * } } - s _ { t , y _ { t } } \le \alpha ( \pi _ { t , y _ { t } } - \pi _ { t , j ^ { * } } ) \le \alpha \Delta _ { t } ^ { \pi } ,\tag{9}
$$

where $\begin{array} { r } { \Delta _ { t } ^ { \pi } = \operatorname* { m a x } _ { j } \pi _ { t , j } - \operatorname* { m i n } _ { j } \pi _ { t , j } } \end{array}$ . Comparison with the same competitor bounds the repaired prediction’s final margin:

$$
\begin{array} { r l } & { m ( \tilde { s } _ { t } ) \leq \tilde { s } _ { t , y _ { t } } - \tilde { s } _ { t , j ^ { * } } } \\ & { \qquad = \alpha ( \pi _ { t , y _ { t } } - \pi _ { t , j ^ { * } } ) - G _ { t } ^ { 0 } } \\ & { \qquad \leq \alpha \Delta _ { t } ^ { \pi } - G _ { t } ^ { 0 } . } \end{array}\tag{10}
$$

For a fixed fusion weight and prior score range, closing a larger deficit leaves less room for the repaired prediction’s margin. Ordering additionally depends on the margins of residual errors. Write $z _ { t } = m ( \tilde { s } _ { t } )$ and draw window indices r and e independently and uniformly from repairs and residual errors, respectively, in the same group B. Then

$$
\mathrm { A U R O C } _ { \mathrm { r e p a i r } , \mathcal { B } } ( z ) \le 1 - \operatorname* { P r } \left[ z _ { e } > \alpha \Delta _ { r } ^ { \pi } - G _ { r } ^ { 0 } \right] .\tag{11}
$$

The event on the right makes the residual-error margin exceed the repair margin. If its probability exceeds one half, repair-separation AUROC is below one half under the tie convention in Eq. 3. The sufficient condition thus connects the individual repair bound to the distribution of residual-error margins.

## A.2 MARGIN INTERPRETATION AND RECALIBRATION

On a repair, the correct candidate wins, so

$$
m ( \tilde { s } _ { t } ) = \tilde { s } _ { t , y _ { t } } - \operatorname* { m a x } _ { j \neq y _ { t } } \tilde { s } _ { t , j } = - \tilde { G } _ { t } .\tag{12}
$$

On a residual error, let k and ℓ index the two largest fused scores. Then

$$
m ( \tilde { s } _ { t } ) = \tilde { s } _ { t , k } - \tilde { s } _ { t , \ell } \leq \tilde { s } _ { t , k } - \tilde { s } _ { t , y _ { t } } = \tilde { G } _ { t } .\tag{13}
$$

Without ties, equality holds exactly when the correct candidate is runner-up. When the correct candidate is below the top two, the margin compares two incorrect candidates.

For a strictly increasing function $f , z _ { r } > z _ { e }$ if and only if $f ( z _ { r } ) > f ( z _ { e } )$ . Recalibration by $f$ thus preserves every pairwise ordering and leaves AUROC unchanged. A non-decreasing function can introduce ties but cannot reverse an individual pair. Re-estimating confidence from additional score features can change the ordering, as evaluated in Appendix I.

## B DATASETS, RETRIEVAL TASK, AND LOCAL DECODER

MEG-MASC supplies 71,736 test windows in 6,027 sentence presentations, scored against 1,464 audio candidates from 123 sentences; MOUS uses an 825-candidate pool. Dataset sizes are in Table 1. Stimulus timing identifies query presentations, and source metadata identifies candidate sentence buckets. Presentations define prior resets and sentence-resampling units.

We split stimulus identities 70/10/20 into training, validation, and test sets, assigning every occurrence to the same split. MEG-MASC training and validation windows overlapping test windows from the same audio file are removed. We deduplicate candidates by stimulus identity and window index. Each neural window spans −0.35 to +2.65 seconds relative to word onset.

The neural encoder combines spatial sensor mixing, participant-specific adaptation, and temporal convolutions to predict frozen wav2vec2-large-xlsr-53 audio representations (Baevski et al., 2020;

Conneau et al., 2021). Scores are Frobenius inner products with normalised audio targets. Training uses matched-window positives and in-batch negatives, with early stopping on validation loss. The three independently trained local decoders provide cached full-pool scores shared across all interventions.

## C CONTEXTUAL INTERVENTIONS AND OPERATING-POINT SELECTION

## C.1 HISTORY AGGREGATION AND PRUNING

At the selected history-aggregation setting, let $\mathcal { T } _ { t }$ contain the 64 highest-scoring local candidates and $\theta _ { t } = Q _ { 0 . 9 4 } ( s _ { t } )$ . Each sentence bucket receives its largest positive score excess, normalised by the square root of its size:

$$
u _ { t } ( b ) = \frac { \operatorname* { m a x } \bigr ( \{ ( s _ { t , j } - \theta _ { t } ) _ { + } : j \in \mathcal { T } _ { t } \cap \mathcal { C } _ { b } \} \cup \{ 0 \} \bigr ) } { \sqrt { | \mathcal { C } _ { b } | } } .\tag{14}
$$

Here $( v ) _ { + } = \operatorname* { m a x } ( v , 0 )$ . Equation 5 accumulates this support with $\eta = 0 . 1$ . A bucket is active when $H _ { t - 1 } ( b ) > 0 .$ , so $\mathcal { L } ( \bar { H _ { t - 1 } } , b ) = H _ { t - 1 } ( b )$ for the four highest-scoring active buckets and 0 otherwise; since state values are non-negative, $\Delta _ { t } ^ { \pi } = \operatorname* { m a x } _ { b } H _ { t - 1 } ( b )$ . Prior margin is the top-two gap of $H _ { t - 1 }$ over all buckets, with inactive buckets entering as zero. Fewer than four buckets are active only at the first window of a presentation, where no candidate is boosted, and hard pruning retains every candidate. Prediction reads the preceding state before the current update and uses $\alpha = 2 . 7$ , selected on validation. State resets at each presentation; support comes from preceding windows in word-onset order.

Hard bucket pruning uses the same state but keeps only candidates in its four leading buckets:

$$
\begin{array} { r } { \tilde { s } _ { t , j } ^ { \mathrm { m a s k } } = \left\{ { \begin{array} { l l } { s _ { t , j } , } & { b ( j ) \in \mathrm { T o p I d x } _ { 4 } ( H _ { t - 1 } ) , } \\ { - \infty , } & { \mathrm { o t h e r w i s e } , } \end{array} } \right. } \end{array}\tag{15}
$$

where $\mathrm { T o p I d x _ { 4 } }$ returns the four highest-scoring bucket indices. The retained local scores and state update are unchanged.

## C.2 LANGUAGE-MODEL PRIORS

Qwen3-4B-Base scores either candidate sentences by mean per-token log-likelihood or aligned candidate words by

$$
\pi _ { t , j } = \log _ { \operatorname { L M } } ( w _ { j } \mid h _ { t - 1 } ) .
$$

Teacher-forced histories contain true preceding words; free-running histories contain decoded preceding words. Histories end before the current word and reset at presentation boundaries. In the word-level free-running sweep, each frozen local decoder supplies one predicted history and one prior-score vector per window for all weights. Changing α therefore changes only the combination of the fixed local and prior scores.

The sentence-level free-running and teacher-forced weights are 0.465 and 1.86. Word-level weights maximise mean validation accuracy gain across decoders, selecting 0.15 and 0.5, respectively. Both word-level priors are swept over sixteen weights from 0.05 to 16. Candidates carrying the same word share its score; we compute the prior margin over distinct words before assigning it to candidates.

## D STATISTICAL AND SELECTION PROTOCOLS

Outcome labels are recomputed at every fusion weight, and all confidence measures use those same labels. Thus, fixed pre-fusion and prior scores can have changing AUROCs. Initially incorrect presentation-start windows, where no bucket is active and $\pi _ { t } = 0$ , are retained as residual errors with prior margin 0; excluding them lowers prior-margin AUROC by at most 0.05 (ranks 2–3) and changes fused-margin AUROC by at most 0.004. We compute estimates within each local decoder and then average them; learned estimators first average their five training seeds within a decoder. A repair-separation cell is omitted if any decoder has fewer than five repairs. Reach uses the interpolated 90th percentile within each decoder. Boundary summaries report the deepest group with mean AUROC above 0.5; full curves retain non-contiguous positive groups.

Table 3: Repair rates across initial-rank groups when initially incorrect predictions are grouped by (a) fused margin or (b) pre-fusion margin. Q1 and Q4 contain the smallest and largest margins. Q4/Q1 divides their unrounded repair rates, and its intervals are sentence-cluster bootstrap 95% intervals. Panel (b) uses the same evaluated predictions and fusion-outcome labels as (a).  
(a) Fused margin
<table><tr><td rowspan="2">Initial rank</td><td rowspan="2">Predictions</td><td colspan="4">Repair rate (%)</td><td colspan="2">Q4/Q1</td></tr><tr><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td><td>Ratio</td><td>95% interval</td></tr><tr><td>2-3</td><td>32,277</td><td>42.44</td><td>59.75</td><td>72.61</td><td>84.39</td><td>1.988</td><td>[1.933,2.047]</td></tr><tr><td>4-5</td><td>13,725</td><td>30.80</td><td>37.60</td><td>43.57</td><td>48.18</td><td>1.564</td><td>[1.467, 1.668]</td></tr><tr><td>6-10</td><td>16,861</td><td>20.61</td><td>22.85</td><td>22.16</td><td>19.15</td><td>0.929</td><td>[0.842, 1.022]</td></tr><tr><td>11-20</td><td>15,244</td><td>8.27</td><td>7.43</td><td>6.32</td><td>3.73</td><td>0.451</td><td>[0.357, 0.547]</td></tr><tr><td>21-50</td><td>16,613</td><td>1.37</td><td>1.47</td><td>0.77</td><td>0.55</td><td>0.404</td><td>[0.223, 0.654]</td></tr></table>

(b) Pre-fusion margin
<table><tr><td rowspan="2">Initial rank</td><td colspan="4">Repair rate (%)</td><td colspan="2">Q4/Q1</td></tr><tr><td>Q1</td><td>Q2</td><td>Q3</td><td>Q4</td><td>Ratio</td><td>95% interval</td></tr><tr><td>2-3</td><td>78.34</td><td>75.47</td><td>66.32</td><td>39.05</td><td>0.498</td><td>[0.481, 0.514]</td></tr><tr><td>4-5</td><td>56.44</td><td>48.70</td><td>37.02</td><td>17.98</td><td>0.319</td><td>[0.293, 0.345]</td></tr><tr><td>6-10</td><td>32.12</td><td>27.71</td><td>18.27</td><td>6.67</td><td>0.208</td><td>[0.181, 0.233]</td></tr><tr><td>11-20</td><td>11.34</td><td>8.53</td><td>4.38</td><td>1.50</td><td>0.132</td><td>[0.097, 0.172]</td></tr><tr><td>21-50</td><td>2.14</td><td>1.16</td><td>0.60</td><td>0.27</td><td>0.124</td><td>[0.061, 0.209]</td></tr></table>

Sentence-cluster bootstrap resamples presentations, and participant-cluster bootstrap resamples 27 MEG-MASC or 96 MOUS participants, each with 1,000 replicates and 95% percentile intervals. Each participant sample is shared across weights, rank groups, confidence measures, and decoders; its statistic is the three-decoder mean. Quartile analyses retain their original group membership and pool repeated decoder evaluations of each window. Wilson intervals (Wilson, 1927) describe quartile proportions; sentence bootstrap (Efron & Tibshirani, 1994) quantifies Q4/Q1 contrasts. Withinsentence AUROC pairs repairs with residual errors from the same presentation. Estimator targets and output-selection protocols are given in Appendix I.

## E EVIDENCE FOR CONFIDENCE-ORDERING REVERSAL

## E.1 QUARTILE REPAIR RATES

Repair rates are higher in the largest than in the smallest fused-margin quartile at initial ranks 2–3 and 4–5, and lower at ranks 11–20 and 21–50 (Table 3). The Q4/Q1 interval contains one at ranks 6–10, placing the transition between groups 4–5 and 11–20 under this contrast. Pre-fusion-margin repair rates decrease across quartiles in all five groups on the same outcome labels.

## E.2 REPAIR DISTRIBUTIONS AND CONFIDENCE ORDERING

The two datasets have similar repair distributions: 67.2% of MEG-MASC repairs and 65.3% of MOUS repairs begin at ranks 2–3, while each assigns 0.6% to ranks 21–50. On MEG-MASC, 84.8% of repairs begin within the top five and 96.3% within the top ten, giving $R _ { 9 0 } = 7 \quad$ . Table 4 reports the confidence measures with decoder-level variation and participant-level intervals.

## E.3 WITHIN-SENTENCE COMPARISON

Pairing repairs with residual errors from the same presentation retains the reversal (Table 5). Above initial rank 10, fused-margin AUROC changes from 0.415 across presentations to 0.344 within presentations; prior-margin AUROC remains positive in all five primary groups.

Table 4: Repair-separation AUROC at the main operating point. Point estimates are means across three local decoders. For the MEG-MASC margins, ± denotes across-decoder SD. Brackets give participant-cluster bootstrap 95% intervals, sharing each resample across decoders (27 participants on MEG-MASC, 96 on MOUS). Entropy-based confidence and repair counts are means.  
(a) MEG-MASC
<table><tr><td colspan="4"></td><td rowspan="2">Fused top-16 entropy-based confidence</td></tr><tr><td>Initial rank</td><td>Fused margin</td><td>Pre-fusion margin</td><td>Prior margin</td></tr><tr><td rowspan="2">2-3</td><td> $0 . 7 0 4 \pm 0 . 0 0 6$ </td><td> $\overline { { 0 . 3 1 0 \pm 0 . 0 0 7 } }$ </td><td> $\overline { { 0 . 8 3 8 \pm 0 . 0 0 7 } }$ </td><td>0.67</td></tr><tr><td> $[ 0 . 6 8 7 , 0 . 7 1 8 ]$   $\overline { { 0 . 5 7 9 \pm 0 . 0 0 1 } }$ </td><td> $[ 0 . 3 0 1 , 0 . 3 2 1 ]$   $\overline { { 0 . 3 1 9 \pm 0 . 0 0 6 } }$ </td><td> $[ 0 . 8 2 7 , 0 . 8 4 7 ]$   $\overline { { 0 . 8 1 9 \pm 0 . 0 0 9 } }$ </td><td></td></tr><tr><td>4-5</td><td>[0.559, 0.596] 0.487 ± 0.007</td><td>[0.308, 0.329] 0.328 ± 0.002</td><td>[0.812, 0.826] 0.838 ± 0.003</td><td>0.55</td></tr><tr><td>6-10</td><td>[0.464, 0.505] 0.417 ± 0.018</td><td>[0.320, 0.335] 0.312 ± 0.016</td><td>[0.828, 0.849] 0.889 ± 0.004</td><td>0.47</td></tr><tr><td>11-20</td><td>[0.397, 0.439] 0.392 ± 0.015</td><td>[0.296, 0.329] 0.298 ± 0.031</td><td>[0.881, 0.898] 0.937 ± 0.011</td><td>0.38</td></tr><tr><td>21-50</td><td>[0.351, 0.435]</td><td>[0.271, 0.327]</td><td>[0.918, 0.956]</td><td>0.36</td></tr></table>

(b) MOUS
<table><tr><td>Initial rank</td><td>Fused margin</td><td>Pre-fusion margin</td><td>Prior margin</td><td>Repairs per local decoder</td></tr><tr><td>2-3</td><td>0.621 [0.600, 0.640]</td><td>0.307 [0.295, 0.320]</td><td>0.844 [0.831, 0.856]</td><td>1,438</td></tr><tr><td>4-5</td><td>0.540 [0.514, 0.565]</td><td>0.334 [0.317, 0.350]</td><td>0.868 [0.853, 0.882]</td><td>405</td></tr><tr><td>6-10</td><td>0.467 [0.440, 0.495]</td><td>0.342 [0.322, 0.363]</td><td>0.902 [0.888, 0.916]</td><td>267</td></tr><tr><td>11-20</td><td>0.416 [0.383, 0.450]</td><td>0.338 [0.305, 0.371]</td><td>0.952 [0.940, 0.963]</td><td>79</td></tr><tr><td>21-50</td><td>0.411 [0.304, 0.523]</td><td>0.377 [0.282, 0.457]</td><td>0.990 [0.985, 0.994]</td><td>13</td></tr></table>

Table 5: Within-sentence repair-separation AUROC. Chance is exactly 0.5; 11+ pools all initial ranks above 10.
<table><tr><td>Confidence measure</td><td>2-3 4-5</td><td>6-10</td><td>11-20</td><td>21-50</td><td>11+</td></tr><tr><td>Fused margin</td><td>0.672</td><td>0.520 0.404</td><td>0.357</td><td>0.284</td><td>0.344</td></tr><tr><td>Pre-fusion margin</td><td>0.284</td><td>0.276 0.256</td><td>0.242</td><td>0.204</td><td>0.268</td></tr><tr><td>Prior margin</td><td>0.757</td><td>0.686 0.644</td><td>0.658</td><td>0.719</td><td>0.694</td></tr><tr><td>Comparison pairs</td><td>6,457</td><td>1,279 1,308</td><td>326</td><td>58</td><td>1,097</td></tr></table>

## E.4 REMAINING ERRORS AND PARTICIPANT UNCERTAINTY

Across the three local decoders, 41,508 evaluated predictions have an initial rank above 20. Fusion repairs 179, leaving 41,329 residual errors: 46.6% of all post-fusion errors. On this fixed subset, fused-margin repair-separation AUROC is 0.385 with sentence-cluster bootstrap interval [0.347, 0.425], compared with 0.950 [0.934, 0.965] for prior margin. Pre-fusion margin and fused entropy-based confidence yield 0.305 and 0.365, respectively.

Participant-cluster intervals identify positively and negatively ordered groups on both datasets (Table 4). On MOUS, intervals for ranks 6–10 and 11–20 are below 0.5; ranks 21–50 contain 10–18 repairs per decoder and have a wider interval spanning 0.5. Pooling all initial ranks above 20, fusedmargin intervals are [0.342, 0.430] for MEG-MASC and [0.314, 0.538] for MOUS, while priormargin intervals are [0.933, 0.967] and [0.992, 0.997].

## F MARGIN FORMATION AND EMPIRICAL BOUND TIGHTNESS

## F.1 REPRESENTATIVE OUTCOMES

Figure 4 and Table 6 provide score profiles and summary values for the four outcome types.

Table 6: Representative MEG-MASC windows from one local decoder.
<table><tr><td>Outcome</td><td>Correct-candidate rank Margin before → after</td><td></td></tr><tr><td>Repair</td><td>22 → 1</td><td> $\overline { { 0 . 0 5  0 . 6 1 } }$ </td></tr><tr><td>Regression</td><td>1 → 2</td><td> $0 . 0 4  2 . 4 1$ </td></tr><tr><td>Remained correct</td><td>1 → 1</td><td> $6 . 0 9  2 0 . 3 4$ </td></tr><tr><td>Residual error</td><td> $4 0  7$ </td><td> $0 . 0 7  3 . 5 4$ </td></tr></table>

## F.2 CORRECT-CANDIDATE PARTICIPATION IN THE FUSED MARGIN

As the initial rank increases, the fused margin increasingly compares two incorrect candidates (Table 7). The difference $\tilde { G } _ { t } - m ( \tilde { s } _ { t } )$ is the correct candidate’s deficit below the runner-up and equals zero when the correct candidate is runner-up.

Table 7: Correct-candidate participation on residual errors. Values are mean $\pm \mathrm { \textbf { S D } }$ across local decoders. The deficit is measured below the fused runner-up.
<table><tr><td></td><td>Initial rank Runner-up share (%) Median</td><td> $\overline { { G _ { t } - m ( \tilde { s } _ { t } ) } }$ </td></tr><tr><td>2-3</td><td> $\overline { { 7 5 . 2 2 \pm 0 . 9 0 } }$ </td><td> $\overline { { 0 . 0 0 0 \pm 0 . 0 0 0 } }$ </td></tr><tr><td>4-5</td><td> $4 1 . 1 8 \pm 1 . 3 0$ </td><td> $0 . 2 5 9 \pm 0 . 0 3 8$ </td></tr><tr><td>6-10</td><td> $2 6 . 3 2 \pm 0 . 4 9$ </td><td> $0 . 7 7 0 \pm 0 . 0 4 0$ </td></tr><tr><td>11-20</td><td> $9 . 7 8 \pm 0 . 4 9$ </td><td> $1 . 6 1 4 \pm 0 . 1 0 4$ </td></tr><tr><td>21-50</td><td> $2 . 1 3 \pm 0 . 1 3$ </td><td> $2 . 8 7 6 \pm 0 . 1 5 0$ </td></tr></table>

## F.3 EMPIRICAL EVALUATION OF THE REPAIR-MARGIN BOUNDS

For repair r, let $z _ { r } = m ( \tilde { s } _ { r } )$ and let $j _ { r } ^ { * }$ be the strongest local incorrect competitor. Substituting the range bound α ${ \Delta } _ { r } ^ { \pi } { - } G _ { r } ^ { 0 }$ or the competitor bound $c _ { r } = \alpha ( \pi _ { r , y _ { r } } - \pi _ { r , j _ { r } ^ { * } } ) - G _ { r } ^ { 0 }$ into Eq. 11 gives AUROC upper bounds $U _ { \mathrm { r a n g e } }$ and $U _ { \mathrm { c o m p } } .$ With measured $\mathbf { A U R O C } \ A$ , we have $A \leq U _ { \mathrm { c o m p } } \leq U _ { \mathrm { r a n g e } } ;$ an upper bound below 0.5 certifies negative ordering.

Table 8 summarises all populated cells, each a weight and rank group. Statistics are averaged across decoders, then equally across cells. Under history aggregation, the range bound certifies 16 of 21 negative cells. The other five have AUROCs of 0.434–0.487 and upper bounds of 0.504–0.558, leaving their signs unresolved by this bound.

Table 8: Empirical tightness of repair-margin bounds across prior constructions. Gaps average all included cells, which index fusion weights and initial-rank groups. The final column reports cells with mean $U _ { \mathrm { r a n g e } } < 0 . 5$ among those with mean $A < 0 . 5$
<table><tr><td rowspan="2">Prior</td><td rowspan="2">Cells</td><td colspan="2">Mean upper-bound gap</td><td rowspan="2">Reversals certified by range bound</td></tr><tr><td> $\overline { { U _ { \mathrm { r a n g e } } - A } }$ </td><td> $\overline { { U _ { \mathrm { c o m p } } - A } }$ </td></tr><tr><td>History aggregation</td><td>32</td><td>0.041</td><td>0.028</td><td>16/21</td></tr><tr><td>Word-level teacher-forced LM</td><td>83</td><td>0.403</td><td>0.223</td><td>1/29</td></tr></table>

For the word-level teacher-forced prior, the score range spans the 709 distinct words represented in the candidate pool and can greatly exceed the correct candidate’s prior advantage over its strongest local competitor. The range bound certifies 1 of the 29 negative cells, while the competitor-based bound certifies 11, all at $\alpha \leq 0 . 5$ . The slack of the competitor-based bound is

![](images/9b17234eff22017a5a024c6cfd0bc91989c6bb62e12282f9d5850d73b727fde2.jpg)  
Figure 4: Candidate scores before and after contextual fusion. Rows show a repair, a regression, a prediction that remains correct, and a residual error from local decoder seed 0 on MEG-MASC. The left and right columns show local and fused scores. Candidates are grouped by sentence and retain the same horizontal positions within each pair; stars mark the correct candidate. Both score profiles in each row are shifted by the initial local winner’s score, preserving ranks and margins. Shaded blocks highlight the contextual and correct-candidate sentence buckets. In the residual example, the correct candidate rises from rank 40 to rank 7 while the prediction remains incorrect.

$$
c _ { r } - z _ { r } = \operatorname* { m a x } _ { k \neq y _ { r } } \tilde { s } _ { r , k } - \tilde { s } _ { r , j _ { r } ^ { \ast } } .
$$

The competitor-based bound is attained when $j _ { r } ^ { * }$ remains the strongest incorrect competitor after fusion. It is attained more often under history aggregation than under the word-level teacher-forced prior. The empirical tightness therefore depends on both the prior score range and the competitors that determine the final margin.

## G CORRECTION-REACH INTERVENTIONS

## G.1 HISTORY-AGGREGATION FUSION-WEIGHT SWEEP

Table 9 reports all eight weights with fixed local scores, prior scores, candidate pool, prior-state trajectories, and test windows. The 32 sufficiently populated cells retain negative pre-fusion ordering (0.111–0.378) and positive prior-margin ordering (0.724–0.937), while the fused margin becomes positive across progressively higher initial-rank groups. Reach increases from 3.0 to 9.3, accompanied by additional repairs and regressions (Table 10). Figure 5 shows the control margins, and Figure 6 the score-deficit analysis.

The history-aggregation boundary sequence is non-decreasing in all 1,000 participant resamples and matches the point-estimate sequence at every weight in 903 of the 1,000 resamples.

Table 9: History-aggregation sweep: fused-margin repair-separation AUROC (three-decoder means) and deepest positive group. All eight weights are shown. Dashes denote fewer than five repairs in at least one decoder. Control curves are in Figure 5.
<table><tr><td></td><td>6-10 11-20 21-50</td></tr><tr><td>α 2-3 4-5 0.5 0.229</td><td></td></tr><tr><td>一</td><td>0.143 一 一</td></tr><tr><td>0.303 0.200 0.180</td><td>none</td></tr><tr><td>0.7 1 0.399</td><td>一 none</td></tr><tr><td>0.275 0.210</td><td>一 none</td></tr><tr><td>1.5 0.520 0.379</td><td>0.319 0.265 2-3</td></tr><tr><td>2 0.613 0.480 2.7</td><td>0.399 0.336 0.350 2-3</td></tr><tr><td>0.704 0.579 0.487</td><td>0.417 0.392 4-5</td></tr><tr><td>0.758 0.633 0.551</td><td>0.468 0.434 6-10</td></tr><tr><td>3.3 4 0.802 0.687</td><td>0.602 0.533 0.452</td></tr><tr><td>11-20</td><td></td></tr></table>

Table 10: History-aggregation reach and intervention outcomes at six fusion weights. Counts are per-decoder means.
<table><tr><td>α Repairs</td><td> $R _ { 9 0 }$ </td><td>∆R@1 (pp)</td><td>Regressions</td></tr><tr><td>0.5 2,376</td><td>3.0</td><td>3.30</td><td>7.3</td></tr><tr><td>1.0 4,554</td><td>4.0</td><td>6.33</td><td>16.7</td></tr><tr><td>2.0 8,278</td><td>5.0</td><td>11.49</td><td>34.3</td></tr><tr><td>2.7 10,381</td><td>7.0</td><td>14.40</td><td>51.3</td></tr><tr><td>3.3 11,910</td><td>8.0</td><td>16.52</td><td>65.3</td></tr><tr><td>4.0 13,410</td><td>9.3</td><td>18.58</td><td>88.0</td></tr></table>

## G.2 SCORE-DEFICIT GROUPS

Grouping by the initial score deficit $G _ { t } ^ { 0 } = \operatorname* { m a x } _ { j \neq y _ { t } } s _ { t , j } - s _ { t , y _ { t } }$ gives the same direction of boundary movement (Table 11; Figure 6). Five equal-mass groups of initially incorrect predictions remain fixed across weights, with cutoffs 1.24, 2.51, 3.97, and 6.03. Score-deficit reach, $\mathsf { \bar { D } } _ { 9 0 } = Q _ { 0 . 9 0 } ( G _ { t } ^ { 0 } \mid$ repair), is computed within each decoder and averaged. It rises from 0.61 to 3.24 while the deepest positive group advances from none to the fourth group.

![](images/61fe956a7c6f6414878f74371889c118594e770238cc5bd351164a5fe4167156.jpg)  
Figure 5: Pre-fusion and prior margins across the fusion-weight sweep. Both panels use the history-aggregation prior on MEG-MASC and the same repairs and residual errors at each $\alpha .$ Darker curves indicate larger weights; 0.5 marks chance. (A) Pre-fusion margin remains below chance in every reported initial-rank group. (B) Prior margin remains above chance. Groups with fewer than five repairs in any decoder are omitted, as in Table 9.

![](images/c14c9993477a88950bb72bfdc8a00e20149e4d44005f93d9f976da1096d2006e.jpg)  
Figure 6: Fused-margin ordering in score-deficit groups. History-aggregation prior on MEG-MASC, with the same repairs and residual errors at each α as in Figure 5. Darker curves indicate larger weights; 0.5 marks chance. Five equal-mass groups of initial score deficit are fixed across weights, with cutoffs 1.24, 2.51, 3.97, and 6.03; the boundary moves toward larger deficits as α increases. The $\alpha = 0 . 5$ curve is omitted because only one group qualifies. Omitted cells follow the event-count rule of Table 11.

Table 11: Fused-margin repair-separation AUROC in fixed score-deficit groups on MEG-MASC. Entries are means across three local decoders; dashes denote fewer than five repairs in at least one decoder. Groups 1–5 are ordered by increasing initial score deficit.
<table><tr><td rowspan="2">α</td><td colspan="5">Score-deficit group</td><td rowspan="2">Deepest group mean  $\mathrm { A U R O C } > 0 . 5$ </td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td>0.5</td><td>0.441</td><td>1</td><td>1</td><td>一</td><td>一</td><td>none</td></tr><tr><td>0.7</td><td>0.561</td><td>0.148</td><td>一</td><td>一</td><td>一</td><td>1</td></tr><tr><td>1.0</td><td>0.678</td><td>0.257</td><td></td><td>一</td><td>一</td><td>1</td></tr><tr><td>1.5</td><td>0.777</td><td>0.447</td><td>0.306</td><td></td><td>一</td><td>1</td></tr><tr><td>2.0</td><td>0.835</td><td>0.575</td><td>0.401</td><td>0.305</td><td></td><td>2</td></tr><tr><td>2.7</td><td>0.886</td><td>0.669</td><td>0.519</td><td>0.409</td><td></td><td>3</td></tr><tr><td>3.3</td><td>0.910</td><td>0.718</td><td>0.592</td><td>0.473</td><td>0.380</td><td>3</td></tr><tr><td>4.0</td><td>0.928</td><td>0.758</td><td>0.648</td><td>0.524</td><td>0.419</td><td>4</td></tr></table>

## G.3 STRUCTURALLY REPAIRABLE ERRORS

Shared bucket scores preserve the local order within each bucket: $\tilde { s } _ { t , i } - \tilde { s } _ { t , j } = s _ { t , i } - s _ { t , j }$ for $i , j \in \mathcal { C } _ { b }$ An additive repair therefore requires the correct candidate to have the highest local score within its bucket. With $\mathbf { i } [ \cdot ]$ denoting the indicator function, the structurally repairable subset is

$$
\mathrm { r e p a i r a b l e } _ { t } = \mathbf { 1 } \left[ s _ { t , y _ { t } } = \operatorname* { m a x } _ { j \in \mathcal { C } _ { b ( y _ { t } ) } } s _ { t , j } \right] .\tag{16}
$$

This subset contains 98.6% of initial errors at ranks 2–3 and 72.5% at ranks 21–50, retaining every repair. Across the eight-weight sweep, restricting comparisons to this subset changes fused-margin AUROC by at most 0.004 and preserves every boundary in each decoder. At the main operating point, pre-fusion ordering remains negative and prior-margin ordering positive. The reversal therefore persists among errors satisfying the within-bucket condition for repair.

## G.4 WORD-LEVEL LM SWEEPS

Tables 12 and 13 report all sixteen weights for each word-level prior with both score channels fixed. The teacher-forced prior extends positive fused-margin ordering from no group at $\alpha = 0 . 0 5$ to all six observed groups at $\alpha = 1$ , as reach increases from 3.0 to 19.3. All observed groups remain positive at larger weights, so no crossing is observed through rank 100 for $\alpha \geq 1$ . Figure 3C plots the observed crossings at $\alpha \leq 0 . 7 5$

This extension continues after net accuracy gain peaks: between $\alpha = 0 . 5$ and 1, gain declines from 11.20 to 9.16 percentage points while positive ordering expands from ranks 6–10 to all observed groups. At $\alpha \geq 3 .$ , regressions outnumber repairs even though fused-margin repair ordering remains positive throughout the observed range.

The free-running prior produces a different trajectory. At its validation-selected weight, $\alpha = 0 . 1 5 ,$ reach is 3.0 and gain is 1.25 percentage points. Positive ordering first appears at ranks $^ { 2 - 3 }$ when $\alpha = 0 . 5$ , where gain is −0.69 points. $\mathrm { { A t } } \ \alpha = 2$ and 3, ranks 21–50 remain negative while ranks 51–100 are positive; at $\alpha = 4$ , the deepest positive group moves back to ranks 11–20. Fusion weight therefore changes both the shape and extent of positive ordering.

Pre-fusion and prior margins. At each weight, we evaluate the pre-fusion and prior margins on the same repair and residual-error labels as the fused margin, showing how ordering changes for fixed input scores as outcomes change. Through $\alpha = 1$ , pre-fusion ordering is negative and prior-margin ordering positive in every reported group for both word-level priors (Table 15). At larger weights, pre-fusion AUROC approaches 0.5, and some groups become positive. For the teacher-forced prior, ranks 11–20 reach 0.503 at $\alpha = 1 . 5 .$ , with a net accuracy gain of 5.05 percentage points. Teacherforced prior-margin ordering stays positive throughout the sweep; the free-running prior margin has one negative group, ranks 51–100 at $\alpha = 1 6 ( 0 . 4 7 9 )$ .

Table 12: Word-level teacher-forced LM sweep. Fused-margin AUROCs, reach, and accuracy gain are three-decoder means; validation selects $\alpha = 0 . 5$ . We show every tested weight and all six rank groups. Dashes follow the repair-count rule. Event counts are in Table 14.
<table><tr><td rowspan="2">α</td><td colspan="7"></td><td rowspan="2">Deepest  $R _ { 9 0 }$ </td><td rowspan="2">∆R@1 (pp)</td></tr><tr><td>2-3</td><td>4-5</td><td>6-10</td><td>Initial rank 11-20</td><td>21-50</td><td>51-100</td><td>positive group</td></tr><tr><td>0.05</td><td>0.173</td><td>0.112</td><td></td><td>一</td><td>一</td><td>一</td><td>none</td><td>3.0</td><td>+2.21</td></tr><tr><td>0.1</td><td>0.301</td><td>0.212</td><td>0.172</td><td></td><td></td><td></td><td>none</td><td>3.0</td><td>+4.20</td></tr><tr><td>0.15</td><td>0.399</td><td>0.305</td><td>0.248</td><td>0.205</td><td></td><td></td><td>none</td><td>4.0</td><td>+5.88</td></tr><tr><td>0.2</td><td>0.471</td><td>0.378</td><td>0.301</td><td>0.257</td><td></td><td>一</td><td>none</td><td>5.0</td><td>+7.29</td></tr><tr><td>0.25</td><td>0.526</td><td>0.432</td><td>0.356</td><td>0.291</td><td>0.260</td><td>一</td><td>2-3</td><td>5.3</td><td>+8.47</td></tr><tr><td>0.3</td><td>0.571</td><td>0.475</td><td>0.394</td><td>0.335</td><td>0.278</td><td></td><td>2-3</td><td>6.3</td><td>+9.45</td></tr><tr><td>0.4</td><td>0.633</td><td>0.547</td><td>0.480</td><td>0.417</td><td>0.356</td><td>0.267</td><td>4-5</td><td>8.3</td><td>+10.65</td></tr><tr><td>0.5</td><td>0.674</td><td>0.595</td><td>0.535</td><td>0.484</td><td>0.419</td><td>0.366</td><td>6-10</td><td>10.3</td><td>+11.20</td></tr><tr><td>0.75</td><td>0.732</td><td>0.664</td><td>0.623</td><td>0.567</td><td>0.517</td><td>0.486</td><td>21-50</td><td>15.0</td><td>+10.80</td></tr><tr><td>1</td><td>0.762</td><td>0.701</td><td>0.667</td><td>0.617</td><td>0.571</td><td>0.557</td><td>51-100</td><td>19.3</td><td>+9.16</td></tr><tr><td>1.5</td><td>0.785</td><td>0.733</td><td>0.705</td><td>0.671</td><td>0.640</td><td>0.625</td><td>51-100</td><td>25.7</td><td>+5.05</td></tr><tr><td>2</td><td>0.800</td><td>0.747</td><td>0.717</td><td>0.686</td><td>0.667</td><td>0.647</td><td>51-100</td><td>30.0</td><td>+1.31</td></tr><tr><td>3</td><td>0.806</td><td>0.765</td><td>0.733</td><td>0.703</td><td>0.687</td><td>0.682</td><td>51-100</td><td>35.7</td><td>-4.19</td></tr><tr><td>4</td><td>0.810</td><td>0.772</td><td>0.739</td><td>0.707</td><td>0.695</td><td>0.695</td><td>51-100</td><td>39.3</td><td>-7.72</td></tr><tr><td>8</td><td>0.808</td><td>0.768</td><td>0.741</td><td>0.708</td><td>0.695</td><td>0.695</td><td>51-100</td><td>47.7</td><td>-14.41</td></tr><tr><td>16</td><td>0.796</td><td>0.750</td><td>0.727</td><td>0.690</td><td>0.680</td><td>0.677</td><td>51-100</td><td>52.7</td><td>-18.55</td></tr></table>

Table 13: Word-level free-running LM sweep. Fused-margin AUROCs, reach, and accuracy gain are three-decoder means; validation selects $\alpha = 0 . 1 5$ . We show every tested weight and all six rank groups. Dashes follow the repair-count rule. Event counts are in Table 14.
<table><tr><td rowspan="2">α</td><td colspan="7">Initial rank</td><td rowspan="2">Deepest  $R _ { 9 0 }$ </td><td rowspan="2">∆R@1 (pp)</td></tr><tr><td> $2 { - } 3$ </td><td>4-5</td><td>6-10</td><td>11-20</td><td>21-50</td><td>51-100</td><td>positive group</td></tr><tr><td>0.05</td><td>0.124</td><td>0.082</td><td>一</td><td>一</td><td>一</td><td>一</td><td>none</td><td>2.0</td><td>+0.64</td></tr><tr><td>0.1</td><td>0.214</td><td>0.169</td><td></td><td></td><td>一</td><td>一</td><td>none</td><td>3.0</td><td>+1.03</td></tr><tr><td>0.15</td><td>0.288</td><td>0.206</td><td>0.170</td><td>一</td><td>一</td><td>一</td><td>none</td><td>3.0</td><td>+1.25</td></tr><tr><td>0.2</td><td>0.347</td><td>0.264</td><td>0.225</td><td>一</td><td>一</td><td>一</td><td>none</td><td>3.0</td><td>+1.28</td></tr><tr><td>0.25</td><td>0.394</td><td>0.305</td><td>0.253</td><td></td><td>一</td><td>一</td><td>none</td><td>4.0</td><td>+1.20</td></tr><tr><td>0.3</td><td>0.431</td><td>0.347</td><td>0.290</td><td>0.259</td><td>一</td><td>一</td><td>none</td><td>4.0</td><td>+1.00</td></tr><tr><td>0.4</td><td>0.492</td><td>0.395</td><td>0.348</td><td>0.330</td><td>一</td><td>一</td><td>none</td><td>4.0</td><td>+0.27</td></tr><tr><td>0.5</td><td>0.539</td><td>0.433</td><td>0.381</td><td>0.358</td><td>0.211</td><td>一</td><td>2-3</td><td>5.0</td><td>-0.69</td></tr><tr><td>0.75</td><td>0.607</td><td>0.506</td><td>0.452</td><td>0.419</td><td>0.338</td><td>一</td><td>4-5</td><td>5.3</td><td>-3.71</td></tr><tr><td>1</td><td>0.645</td><td>0.537</td><td>0.482</td><td>0.435</td><td>0.420</td><td></td><td>4-5</td><td>6.0</td><td>-6.78</td></tr><tr><td>1.5</td><td>0.691</td><td>0.584</td><td>0.534</td><td>0.474</td><td>0.427</td><td>0.486</td><td>6-10</td><td>7.0</td><td>-12.15</td></tr><tr><td>2</td><td>0.714</td><td>0.608</td><td>0.564</td><td>0.496</td><td>0.442</td><td>0.520</td><td>51-100</td><td>8.0</td><td>-16.23</td></tr><tr><td>3</td><td>0.739</td><td>0.644</td><td>0.592</td><td>0.527</td><td>0.451</td><td>0.519</td><td>51-100</td><td>9.0</td><td>-21.49</td></tr><tr><td>4</td><td>0.755</td><td>0.668</td><td>0.606</td><td>0.548</td><td>0.463</td><td>0.462</td><td>11-20</td><td>9.7</td><td>-24.66</td></tr><tr><td>8</td><td>0.782</td><td>0.691</td><td>0.632</td><td>0.547</td><td>0.491</td><td>0.454</td><td>11-20</td><td>11.3</td><td>-30.04</td></tr><tr><td>16</td><td>0.787</td><td>0.690</td><td>0.626</td><td>0.529</td><td>0.487</td><td>0.463</td><td>11-20</td><td>13.0</td><td>-33.13</td></tr></table>

Table 14: Repairs and regressions in the word-level sweeps, shown as means per local decoder.
<table><tr><td rowspan="2">α</td><td colspan="2">Teacher-forced</td><td colspan="2">Free-running</td></tr><tr><td>Repairs</td><td>Regressions</td><td>Repairs</td><td>Regressions</td></tr><tr><td>0.05</td><td>1,748.0</td><td>161.0</td><td>832.3</td><td>373.3</td></tr><tr><td>0.1</td><td>3,358.3</td><td>343.3</td><td>1,536.7</td><td>796.3</td></tr><tr><td>0.15</td><td>4,771.7</td><td>553.7</td><td>2,121.0</td><td>1,223.3</td></tr><tr><td>0.2</td><td>6,027.0</td><td>796.7</td><td>2,597.3</td><td>1,680.0</td></tr><tr><td>0.25</td><td>7,120.7</td><td>1,048.0</td><td>3,000.3</td><td>2,137.7</td></tr><tr><td>0.3</td><td>8,086.0</td><td>1,308.7</td><td>3,323.0</td><td>2,606.0</td></tr><tr><td>0.4</td><td>9,542.7</td><td>1,899.3</td><td>3,783.0</td><td>3,586.7</td></tr><tr><td>0.5</td><td>10,575.0</td><td>2,542.3</td><td>4,058.7</td><td>4,554.0</td></tr><tr><td>0.75</td><td>11,972.0</td><td>4,222.0</td><td>4,303.0</td><td>6,962.0</td></tr><tr><td>1</td><td>12,434.0</td><td>5,865.0</td><td>4,269.0</td><td>9,131.3</td></tr><tr><td>1.5</td><td>12,378.0</td><td>8,756.0</td><td>3,908.7</td><td>12,627.7</td></tr><tr><td>2</td><td>12,016.7</td><td>11,078.3</td><td>3,540.3</td><td>15,181.3</td></tr><tr><td>3</td><td>11,251.7</td><td>14,260.7</td><td>3,035.3</td><td>18,449.3</td></tr><tr><td>4</td><td>10,659.0</td><td>16,195.3</td><td>2,712.3</td><td>20,399.7</td></tr><tr><td>8</td><td>9,402.0</td><td>19,742.0</td><td>2,119.3</td><td>23,671.7</td></tr><tr><td>16</td><td>8,625.0</td><td>21,934.7</td><td>1,783.3</td><td>25,551.0</td></tr></table>

Table 15: Pre-fusion and prior-margin repair-separation AUROC in the word-level sweeps. Each range spans the reported three-decoder means over all sufficiently populated initial-rank groups and tested weights in the indicated interval. Both measures use the same cells and outcome labels.
<table><tr><td>Word-level LM prior</td><td>Weights</td><td>Cells</td><td>Pre-fusion margin</td><td>Prior margin</td></tr><tr><td rowspan="2">Teacher-forced</td><td> $\alpha \leq 1$ </td><td>47</td><td>0.102-0.494</td><td rowspan="2">0.554-0.913 0.566–0.706</td></tr><tr><td> $\alpha > 1$ </td><td>36</td><td>0.477–0.509</td></tr><tr><td rowspan="2">Free-running</td><td>α≤ 1</td><td>36</td><td>0.098-0.475</td><td>0.522-0.711</td></tr><tr><td> $\alpha > 1$ </td><td>36</td><td>0.450–0.525</td><td>0.479–0.596</td></tr></table>

## G.5 REACH AND ORDERING ACROSS INTERVENTIONS

At the selected operating points, effective correction reach spans 3.0–15.7 across six interventions (Table 16). Both free-running priors have no positively ordered group; the history-aggregation and word-level teacher-forced priors extend positive ordering through ranks 4–5 and 6–10; hard bucket pruning and the sentence-level teacher-forced prior extend it through ranks 11–20 (Table 17).

Configurations with similar reach can have different ordering boundaries. History-aggregation fusion at $\alpha = 4$ has $R _ { 9 0 } = 9 . 3 $ and positive ordering through ranks 11–20; the word-level teacherforced prior at $\alpha = 0 . 5$ has $R _ { 9 0 } = 1 0 . 3 $ and positive ordering through ranks 6–10. Thus, reach locates the repair distribution, while the AUROC curves additionally describe how repair and residualerror margins are arranged.

## H PRIOR INFORMATION AND REVISION OUTCOMES

For the history-aggregation prior, the highest-scoring sentence bucket is correct on 98% of windows in the highest prior-margin quintile and 20% in the lowest. Among initially correct predictions, the highest prior-margin region places 99.9% of correct candidates in the prior’s leading bucket. Shuffling candidate-to-sentence assignments changes net additional correct predictions from $1 0 , 3 2 9 \pm 1 9 0 \mathrm { t o } - 1 9 8 \pm 2 3$ per decoder. This contrast links the gain to aligning accumulated support with candidate identity.

Table 18 evaluates prior margin among initially correct test predictions, using those that remain correct as positives and regressions as negatives. At the tested settings, prior margin distinguishes

Table 16: Reach and intervention outcomes at the selected operating points on MEG-MASC. Reach and accuracy gain are mean ± SD across three local decoders; repair and regression counts are perdecoder means.
<table><tr><td>Intervention</td><td> $R _ { 9 0 }$ </td><td>Repairs</td><td>Regressions</td><td>nn@1 (pp)</td></tr><tr><td>Sentence LM, free-running</td><td> $3 . 0 \pm 0 . 0$ </td><td>2,529.0</td><td>1,579.7</td><td> $\overline { { 1 . 3 2 \pm 0 . 1 3 } }$ </td></tr><tr><td>Word LM, free-running</td><td> $3 . 0 \pm 0 . 0$ </td><td>2,121.0</td><td>1,223.3</td><td> $1 . 2 5 \pm 0 . 0 7$ </td></tr><tr><td>History aggregation</td><td> $7 . 0 \pm 0 . 0$ </td><td>10,380.7</td><td>51.3</td><td> $1 4 . 4 0 \pm 0 . 2 6$ </td></tr><tr><td>Word LM, teacher-forced</td><td> $1 0 . 3 \pm 0 . 6$ </td><td>10,575.0</td><td>2,542.3</td><td> $1 1 . 2 0 \pm 0 . 2 3$ </td></tr><tr><td>Hard bucket pruning</td><td> $1 5 . 3 \pm 0 . 6$ </td><td>16,452.7</td><td>992.7</td><td> $2 1 . 5 5 \pm 0 . 2 4$ </td></tr><tr><td>Sentence LM, teacher-forced</td><td> $1 5 . 7 \pm 0 . 6$ </td><td>14,148.3</td><td>3,573.7</td><td> $1 4 . 7 4 \pm 0 . 4 1$ </td></tr></table>

Table 17: Fused- and prior-margin repair-separation AUROC at the six selected settings (threedecoder means). Dashes follow the repair-count rule; blank cells were not assessed. The 51–100 group is used only for word-level priors.
<table><tr><td>Prior</td><td>Margin</td><td>2-3</td><td>4-5</td><td>6-10</td><td>11-20</td><td>21-50</td><td>51-100</td></tr><tr><td rowspan="2">Sentence LM, free-running</td><td>Fused</td><td>0.314</td><td>0.197</td><td>0.156</td><td>一</td><td>一</td><td></td></tr><tr><td>Prior</td><td>0.603</td><td>0.663</td><td>0.782</td><td></td><td></td><td></td></tr><tr><td rowspan="2">History aggregation</td><td>Fused</td><td>0.704</td><td>0.579</td><td>0.487</td><td>0.417</td><td>0.392</td><td></td></tr><tr><td>Prior</td><td>0.838</td><td>0.819</td><td>0.838</td><td>0.889</td><td>0.937</td><td></td></tr><tr><td rowspan="2">Hard bucket pruning</td><td>Fused</td><td>0.838</td><td>0.745</td><td>0.656</td><td>0.559</td><td>0.454</td><td></td></tr><tr><td>Prior</td><td>0.773</td><td>0.713</td><td>0.675</td><td>0.638</td><td>0.621</td><td></td></tr><tr><td rowspan="2">Sentence LM, teacher-forced</td><td>Fused</td><td>0.778</td><td>0.722</td><td>0.661</td><td>0.585</td><td>0.481</td><td></td></tr><tr><td>Prior</td><td>0.816</td><td>0.846</td><td>0.861</td><td>0.869</td><td>0.873</td><td></td></tr><tr><td rowspan="2">Word LM, free-running</td><td>Fused</td><td>0.288</td><td>0.206</td><td>0.170</td><td></td><td>一</td><td></td></tr><tr><td>Prior</td><td>0.532</td><td>0.560</td><td>0.701</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Word LM, teacher-forced</td><td>Fused</td><td>0.674</td><td>0.595</td><td>0.535</td><td>0.484</td><td>0.419</td><td>0.366</td></tr><tr><td>Prior</td><td>0.568</td><td>0.600</td><td>0.617</td><td>0.659</td><td>0.752</td><td>0.759</td></tr></table>

preserved correct predictions from regressions more strongly for history aggregation than for the word-level LM priors.

Table 18: Prior-margin AUROC among initially correct MEG-MASC predictions at validationselected weights. Positives remain correct; negatives are regressions. Means are across three decoders, with SD where available.
<table><tr><td>Prior</td><td>α AUROC</td></tr><tr><td>History aggregation 2.7</td><td>0.777 ± 0.022</td></tr><tr><td>Word-level teacher-forced LM 0.5</td><td>0.545</td></tr><tr><td>Word-level free-running LM 0.15</td><td>0.532</td></tr></table>

## I DIRECT CONFIDENCE MEASURES AND LEARNED ESTIMATORS

## I.1 FEATURES AND PREDICTION TARGETS

Table 19 separates local, fused, and prior information. The prior state summarises preceding windows; current support comes from the present window. Sentence-position features use the window index and the annotated presentation length. Removing position features in separately fitted top-1 estimators changes repair-separation AUROC over local errors from 0.775 to 0.783; removing support entropy as well gives 0.783.

Two-layer MLPs with 32 units per hidden layer predict fused top-K inclusion for $K \in$ {1, 3, 5, 10, 20}. Training uses five-fold, presentation-grouped validation data; per-head temperature calibration (Guo et al., 2017) uses out-of-fold logits. Repair-focused evaluations use test windows where the local top-K set excludes the correct candidate; at K = 1, success is repair. Tied candidates share a rank, so rank-based inclusion resolves ties optimistically. Pooled correctness uses all test windows and fused top-1 correctness. Results average five estimator seeds within each of three local decoders.

Table 19: Observable inputs to the confidence estimators. Decoder scores combine local and fused features (8); prior features combine prior state and additional prior features (7); the full estimator uses all 15.
<table><tr><td>Feature group</td><td>Count Inputs</td><td></td></tr><tr><td>Local</td><td>2</td><td>Pre-fusion margin; local top-16 entropy-based confidence.</td></tr><tr><td>Fused</td><td>6</td><td>Fused-score gaps at  $K \in \{ 1 , 3 , 5 , 1 0 , 2 0 \}$  ; fused top-16 entropy-based confidence.</td></tr><tr><td>Prior state</td><td>3</td><td>History indicator; prior-state entropy-based confidence; prior- state margin.</td></tr><tr><td>Additional prior</td><td>4</td><td>Current-support entropy-based confidence; leading-bucket agreement between prior state and current support; two sentence-position features.</td></tr></table>

## I.2 OUTCOME DISCRIMINATION

At initial ranks 21–50, negative repair ordering extends beyond the fused margin: fused top-16 entropy-based confidence, sentence-score mass gap, and sentence-score concentration give AU-ROCs of 0.36, 0.46, and 0.43. The last two summarise fused score mass between and across sentence buckets. Table 20 shows the positive ordering recovered by learned estimates. Over all test windows, the full estimator achieves an AUROC of 0.954, compared with 0.872 for the fused margin; among local errors, the latter achieves 0.646, falling to 0.415 above rank 10 and 0.344 when comparisons are restricted within presentations.

Table 20: AUROC for fused top-K success within initial-rank groups on MEG-MASC. Values are means across three local decoders, averaged over estimator seeds within each decoder. The K = 1 columns measure repair-separation AUROC, and $n _ { \mathrm { r e p a i r } }$ counts top-1 repairs per local decoder.  
(a) Top-1 success
<table><tr><td>Initial rank</td><td> $n _ { \mathrm { r e p a i r } }$ </td><td>Local scores</td><td>Decoder scores</td><td>Prior features</td></tr><tr><td>2-3</td><td>6,971</td><td>0.399</td><td>0.891</td><td>0.720</td></tr><tr><td>4-5</td><td>1,832</td><td>0.318</td><td>0.833</td><td>0.728</td></tr><tr><td>6-10</td><td>1,191</td><td>0.284</td><td>0.779</td><td>0.775</td></tr><tr><td>11-20</td><td>327</td><td>0.240</td><td>0.716</td><td>0.848</td></tr><tr><td>21-50</td><td>58</td><td>0.271</td><td>0.653</td><td>0.910</td></tr></table>

(b) Top-5 success
<table><tr><td>Initial rank</td><td>Decoder scores</td><td>Prior features</td><td>Full</td></tr><tr><td>6-10</td><td>0.663</td><td>0.681</td><td>0.676</td></tr><tr><td>11-20</td><td>0.552</td><td>0.752</td><td>0.676</td></tr><tr><td>21-50</td><td>0.522</td><td>0.867</td><td>0.771</td></tr></table>

Table 21 complements ranking with rare-repair precision. At ranks 21–50, the prior-state estimator’s highest-scoring 10% of predictions have an 8.40% repair rate, compared with 1.80% for decoder scores and a population rate of 1.05%.

Table 21: Rare-repair prediction with learned decoder-score and prior-state estimators on MEG-MASC. P@10% is the repair rate among the 10% highest-scoring predictions. The prior-state estimator uses the prior-state group in Table 19.
<table><tr><td rowspan="2">Initial rank</td><td rowspan="2">Repair rate (%)</td><td colspan="2">AUPRC</td><td colspan="2">P@10% (%)</td></tr><tr><td>Decoder-score</td><td>Prior-state</td><td>Decoder-score</td><td>Prior-state</td></tr><tr><td>11-20</td><td>6.44</td><td>0.129</td><td>0.396</td><td>13.70</td><td>36.40</td></tr><tr><td>21-50</td><td>1.05</td><td>0.018</td><td>0.242</td><td>1.80</td><td>8.40</td></tr></table>

## I.3 SELECTIVE-OUTPUT POLICIES

At the history-aggregation setting $\alpha = 2 . 7 ,$ , adaptive policies emit the smallest set whose calibrated success probability exceeds its threshold, or abstain. We select thresholds on validation under coverage $\ge 9 0 \%$ and mean $K \leq 4 . 8$ The margin-binned reference applies conformal calibration in 20 fused-margin bins; isotonic calibration (Zadrozny & Elkan, 2002) of that margin finds no point satisfying both constraints. Table 22 reports the selected policies on test windows; validation constraints are not guaranteed on test, where mean K slightly exceeds 4.8, and fixed-K coverage falls below 90% for $\bar { K } \ge 3$ with learned estimators and at every K for margin-based policies (Table 24). Emission uses all windows as its denominator; coverage and mean K use emitted sets.

Table 22: Test performance of validation-selected selective-output policies with the historyaggregation prior. Means and SDs are across three local decoders, after averaging estimator seeds within each decoder. The stratified analysis uses these adaptive policies; fixed-K thresholds are selected separately.
<table><tr><td>Policy</td><td>Emission (%) Coverage (%)</td><td></td><td>Mean K</td></tr><tr><td>Margin-binned</td><td> $5 5 . 9 \pm 8 . 0$ </td><td> $9 1 . 7 \pm 3 . 0$ </td><td>4.90</td></tr><tr><td>Decoder-score</td><td> $7 3 . 6 \pm 0 . 1$ </td><td> $9 1 . 8 \pm 0 . 2$ </td><td> $5 . 0 2 \pm 0 . 0 5$ </td></tr><tr><td>Full</td><td> $7 4 . 5 \pm 0 . 8$ </td><td> $9 2 . 2 \pm 0 . 3$ </td><td> $5 . 1 8 \pm 0 . 0 8$ </td></tr></table>

We apply the same policies to all test windows before grouping by initial rank (Table 23). Adding prior features raises both emission and coverage at ranks 21–50; beyond rank 50, coverage remains low for both policies.

Table 23: Selective output by initial rank on MEG-MASC with the history-aggregation prior. Test results at each policy’s selected operating point, averaged across local decoders. Emission uses all windows in the group as its denominator; coverage and mean K use emitted sets. Rates are percentages.
<table><tr><td rowspan="2">Initial rank</td><td colspan="3">Decoder-score</td><td colspan="3">Full</td></tr><tr><td>Emission</td><td>Coverage</td><td>Mean K</td><td>Emission</td><td>Coverage</td><td>Mean K</td></tr><tr><td>1</td><td>96.8</td><td>100.0</td><td>2.36</td><td>96.9</td><td>100.0</td><td>2.33</td></tr><tr><td>2-3</td><td>89.1</td><td>99.7</td><td>5.48</td><td>86.9</td><td>99.5</td><td>5.92</td></tr><tr><td>4-5</td><td>72.3</td><td>97.8</td><td>9.15</td><td>70.0</td><td>97.1</td><td>8.80</td></tr><tr><td>6-10</td><td>51.7</td><td>93.1</td><td>11.20</td><td>55.6</td><td>93.5</td><td>10.32</td></tr><tr><td>11-20</td><td>35.6</td><td>73.9</td><td>12.35</td><td>43.8</td><td>82.5</td><td>12.02</td></tr><tr><td>21-50</td><td>32.5</td><td>36.4</td><td>12.61</td><td>38.2</td><td>50.8</td><td>12.86</td></tr><tr><td>51+</td><td>30.2</td><td>3.4</td><td>12.76</td><td>30.9</td><td>7.3</td><td>13.75</td></tr></table>

For fixed-K output, validation instead chooses a separate threshold for each policy and K, maximising emission at coverage $\geq 9 0 \%$ . All searches are feasible; no additional mean-set-size constraint applies. Table 24 reports test emission and coverage at these thresholds for every assessed K.

## I.4 FEATURE COMPARISON AT MATCHED OPERATING POINTS

The six feature sets in Table 25 share cached inputs, a hidden-layer architecture, and the same training procedure. For each run, shifting its five thresholds together traces a test operating curve.

Table 24: Fixed-size selective output on MEG-MASC with the history-aggregation prior. Thresholds are selected on validation separately for each policy and K. Entries are test percentages averaged across local decoders, with estimator seeds averaged where applicable. Emission uses all test windows as its denominator; coverage uses emitted sets.
<table><tr><td>Policy</td><td>Rate</td><td> $K = 1$ </td><td> $\overline { { K = 3 } }$   $K = 5$ </td><td> $K = 1 0$ </td><td> $K = 2 0$ </td></tr><tr><td rowspan="2">Full</td><td>Emission</td><td>58.3</td><td>71.2</td><td>77.1</td><td>86.3 99.5</td></tr><tr><td>Coverage</td><td>90.5</td><td>89.2</td><td>88.5</td><td>87.4 86.0</td></tr><tr><td rowspan="2">Decoder-score</td><td>Emission</td><td>56.8</td><td>69.3</td><td>74.8</td><td>84.2 99.2</td></tr><tr><td>Coverage</td><td>90.7</td><td>89.6</td><td>88.9</td><td>87.7 86.0</td></tr><tr><td rowspan="2">Margin-binned</td><td>Emission</td><td>48.7</td><td>55.8</td><td>63.0</td><td>76.3 94.1</td></tr><tr><td>Coverage</td><td>89.1</td><td>88.8</td><td>87.7</td><td>86.5 86.6</td></tr><tr><td rowspan="2">Margin isotonic</td><td>Emission</td><td>49.7</td><td>57.9</td><td>64.7</td><td>75.9 98.4</td></tr><tr><td>Coverage</td><td>88.4</td><td>87.8</td><td>87.0</td><td>86.7 86.0</td></tr></table>

Linear interpolation matches emission at 74.52% or coverage at 92.24%, the reference full policy’s operating values. Runs without a feasible validation point start from the search’s penalised optimum. We match per run, then average within and across decoders. These are test-curve comparisons; Table 22 separately evaluates validation-selected policies.

Table 25: Feature comparison with the history-aggregation prior. Matched emission and coverage are percentages; parentheses give mean emitted set size. Validation counts report runs with a feasible point under coverage $\geq 9 0 \%$ and mean $K \leq 4 . 8 ,$ before the test-curve sweep. Values are matched per run and then averaged over seeds within decoder; SDs are across decoders.
<table><tr><td>Features</td><td>Coverage at 74.52% emission</td><td>Emission at  $9 2 . 2 4 \% \mathrm { c o v e r a g e }$ </td><td>Validation feasible</td></tr><tr><td>Fused margin</td><td> $\overline { { 8 7 . 2 \pm 0 . 1 \ : ( 7 . 1 ) } }$ </td><td> $\overline { { 5 6 . 7 \pm 0 . 5 \ : ( 6 . 3 ) } }$ </td><td>0/15</td></tr><tr><td>Fused scores</td><td> $8 9 . 3 \pm 0 . 2 ( 5 . 5 ) $ </td><td> $6 6 . 6 \pm 0 . 7 ( 5 . 8 )$ </td><td>6/15</td></tr><tr><td>Local scores</td><td> $8 6 . 4 \pm 0 . 1 ( 7 . 7 )$ </td><td> $5 1 . 9 \pm 0 . 9 ( 8 . 2 )$ </td><td>0/15</td></tr><tr><td>Prior features</td><td> $8 7 . 9 \pm 0 . 3 ( 6 . 4 )$ </td><td> $6 1 . 5 \pm 1 . 4 ( 6 . 3 )$ </td><td>0/15</td></tr><tr><td>Decoder scores</td><td> $9 1 . 4 \pm 0 . 2 ( 5 . 1 )$ </td><td> $7 2 . 5 \pm 0 . 5 ( 5 . 2 )$ </td><td>15/15</td></tr><tr><td>Full</td><td> $9 2 . 2 \pm 0 . 2 ( 5 . 2 )$ </td><td> $7 4 . 5 \pm 0 . 4 ( 5 . 2 )$ </td><td>15/15</td></tr></table>