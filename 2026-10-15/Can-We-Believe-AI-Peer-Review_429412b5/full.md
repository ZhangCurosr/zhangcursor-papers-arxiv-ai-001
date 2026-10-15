# Can We Believe AI Peer Review?

Zongrui Yang Haoyuan Li Zhongsheng Wang Gillian Dobbie Jiamou Liu University of Auckland Auckland, New Zealand

## Abstract

As large language models become widely used in scientific review and manuscript revision, AI scores are increasingly serving as feedback for assessing paper quality and guiding edits. Yet whether a single AI score reliably reflects the reviewer’s judgment, and whether revision gains based on that score are endorsed by other reviewers, remain understudied. Using 200 papers from ICLR 2025 and 2026 and approximately 14,000 AI reviews, we study score reliability, score-distribution estimation, and cross-model agreement on revision gains. Repeated reviews fluctuate even under identical conditions. Finer-grained scoring protocols reduce score variance but also systematically shift mean scores, while the gains assigned to manuscript revisions depend strongly on the reviewer.

## 1 Introduction

As large language models become integrated into research workflows, AI-assisted peer review is moving from experimental studies into practical use. Submission volumes at major AI conferences have continued to grow, increasing the pressure on peer review. Conferences including ICLR and AAAI have begun exploring AI assistance in the review process (Zou and Thakkar, 2025; AAAI, 2025). Meanwhile, researchers are using AI review tools to assess paper quality, obtain revision suggestions, and judge the effects of revisions through changes in scores. Both applications rest on a shared premise: AI judgments are reliable enough to provide meaningful evidence for research decisions.

This premise does not necessarily hold. First, AI scores are stochastic: even when the paper, model, and review protocol remain fixed, repeated reviews can yield different scores, making an individual score or an observed gain an unreliable representation of the model’s own assessment. Second, AI judgments depend on the model: reviewers may have different evaluation preferences and, even when they assess original papers similarly, may not endorse the same revisions. Ignoring these issues can lead users to mistake random fluctuations for differences in paper quality and score gains under a particular model for broadly endorsed improvements. How much, then, can we trust AI scores and the gains observed after revision?

![](images/a028147515f7f19d73ed0a56c688ce269f19f3cc8084859fc699648e817ff5ae.jpg)  
Figure 1: Number of submissions to major AI conferences over time.

Prior work has examined AI reviewers’ evaluation capabilities, scoring biases, and sensitivity to paper presentation. ReviewEval (Garg et al., 2025), TitleTrap (Du, 2025), and LLM-as-a-Reviewer (Li et al., 2026a) study different aspects of AI reviewing, while Rhetoric (Li et al., 2026b) further reveals differences in reviewers’ responses to revision gains. Practical use of AI scores nevertheless calls for a connected set of answers: How reliable is a single score? How many reviews are needed to estimate a reviewer’s expected score accurately? Can score estimation be improved under a limited reasoning budget? Does an observed gain reflect random fluctuation, or an improvement endorsed by a particular reviewer? These questions determine when AI scores can serve as research feedback.

To address these questions, we conduct approximately 14,000 AI reviews of 200 papers from ICLR 2025 and ICLR 2026, systematically studying scoring randomness, estimation accuracy, and crossmodel agreement on revision gains. Our main contributions are:

Quantifying the reliability of a single AI score. We use repeated sampling to analyze score fluctuations and their effects on relative paper rankings, then derive sufficient review budgets for specified mean-score error and success-probability targets.

Exploring methods to improve scoring reliability. We examine how score granularity affects variability and systematic shifts, and test whether auxiliary score probabilities improve distribution estimation under a limited reasoning budget.

Assessing the credibility of revision gains. Through repeated scoring and cross-model evaluation, we separate scoring fluctuations from reviewer dependence and test whether different reviewers consistently endorse the same revisions through score gains.

## 2 Related Work

AI peer review. Existing studies primarily assess LLM reviewing ability through the quality of review comments. ReviewEval (Garg et al., 2025) examines factuality, analytical depth, and the actionability of suggestions; LLM-as-a-Reviewer (Li et al., 2026a) compares model and human reviews and tests resistance to prompt injection. These studies help establish whether LLMs can generate high-quality reviews, but devote less attention to how review outcomes inform subsequent decisions. In particular, the extent of score variation when the same paper is reviewed repeatedly, and whether these scores reliably distinguish paper quality before and after revision, remain understudied.

Score stability and estimation. Prior work finds inconsistencies across repeated LLM scoring runs and systematic biases (Stureborg et al., 2024); output formats and explanation requirements also affect evaluation results (Chiang and Lee, 2023). To address scoring uncertainty, G-Eval (Liu et al., 2023) uses score-token probabilities, while Homan et al. (2026) examine how many repeated ratings statistical tests require. However, existing work rarely directly answers how much a single score differs from a mean over repeated scores when the reviewer and reviewing conditions are fixed. We target the expected score and score distribution under fixed reviewing conditions, analyze how repeated sampling affects estimation precision, and evaluate the gains from probability-assisted estimation against independently sampled target scores.

![](images/360a770a9d6f6e497185a0ec91e56e50aaf951b336ff63d51866389413d3503d.jpg)  
Figure 2: Can we believe AI Peer Review?

AI-assisted revision. Beyond evaluating papers, LLM scores can be used to select and guide revisions. TitleTrap (Du, 2025) and Rhetoric (Li et al., 2026b) show that changes to titles and rhetorical presentation can affect AI scores. A score increase may therefore reflect either an improvement in substantive paper quality or a selector’s adaptation to particular scoring preferences. Existing work has not sufficiently separated these sources. We hold the candidate edit set fixed and vary the selector and scorer separately, comparing score gains under shared revision content. This tests whether LLM-score-guided revision strategies remain stable across reviewers and how much of the observed gains can be attributed to sampling uncertainty.

## 3 Problem Formulation and Study Design

## 3.1 What Does a Review Score Estimate?

Score distributions and repeated sampling. We treat one LLM paper-review procedure as one sample. Given a paper $P _ { i }$ and review configuration r (model, prompt, reasoning settings, and execution environment), an individual score is a random variable $Y _ { r } ( P _ { i } )$ with expectation $\mu _ { r , i }$ and variance $\sigma _ { r , i } ^ { 2 }$ . For the same paper, we perform k independent reviews and use their mean $\bar { Y } _ { r , i , k }$ to estimate the expected score under the fixed configuration. Under IID scores within each paper, this mean has variance $\sigma _ { r , i } ^ { 2 } / k$

Evaluation validity. We use the mean human review score $H _ { i }$ as an external reference for paper quality. Correlation and error relative to this reference, together with AUC for acceptance status, serve as indicators of evaluation validity.

Scoring reliability. For a fixed paper and review configuration, scores should exhibit low randomness, and their expectation should be reliably estimable from a finite number of repeated reviews. We report both score stability and relative-ranking stability as indicators of scoring reliability.

Evaluation validity for revisions. For a revision $P _ { i } \to P _ { i } ^ { \prime }$ , the reviewer’s expected gain is $\Delta _ { r , i } = \mu _ { r } ( P _ { i } ^ { \prime } ) - \mu _ { r } ( P _ { i } )$ , estimated by the difference between independent mean scores assigned by the same Scorer to the original and revised manuscripts. Gains from different Scorers are compared on identical revised versions. For fixed original and revised manuscripts, independent scores on the two sides give the gain estimate a variance of $( \sigma _ { r , i } ^ { 2 } + \sigma _ { r , i } ^ { \prime 2 } ) / k$ . An observed score increase contains sampling error from both versions. In the crossover experiment, we score each version repeatedly and keep the Scorer of the original-paper baseline consistent, establishing separate controls for within-model fluctuations and between-model response differences.

## 3.2 Papers and Review Configurations

We draw experimental samples from public ICLR 2025 and ICLR 2026 submissions by stratifying jointly on official subject area and mean human score interval, selecting 100 papers per year. The paper lists and configurations are provided in the supplementary material.

For LLM paper review, PDFs are converted into ordered text and images, retaining the main text, figures, tables, and appendices; review inputs are anonymized. A common task asks models to assess novelty, technical correctness, experimental evidence, and limitations, then produce an overall score from 1 to 10 with review comments.

Original integer scoring uses GPT-5.6-Sol, GPT-6-Luna, and GPT-6-Sol, with ten reviews per paper per configuration, totaling 6,000. The granularity experiment compares integer and forcednoninteger protocols for the latter two models. Distribution estimation targets GPT-5.6-Sol reasoning scores. Revision experiments use Luna and Claude Haiku 5.5, with three scores each for original and revised manuscripts; one paper has 224 images, exceeding Claude’s input limit, so crossover comparisons use the shared 199 papers. Additional configuration details are provided in the supplementary material.

## 4 How Reliable Is a Single AI Review?

We first characterize score variation and validity for individual papers, then test whether independent reviews reproduce the same paper rankings. We further estimate ranking reliability under finite sampling and recommend review budgets.

## 4.1 Repeated Scores and Human References

Table 1 summarizes ten-score repeatability and external associations. Across configurations, 30.0%– 56.5% of papers receive ten identical scores, while 91.5%–100% have a score range of at most one point. Fluctuations mainly occur between adjacent categories. GPT-6-Sol has a within-paper SD of 0.272, compared with 0.374 and 0.386 for the other configurations, and a Spearman correlation of 0.409 with human scores. Final-acceptance AUC is 0.7134 for GPT-5.6-Sol, 0.6484 for GPT-6-Luna, and 0.7138 for GPT-6-Sol.

Within-paper SD averages unbiased paper-level variances before taking the square root. Each set of ten scores forms 45 unordered pairs; exact agreement is 72.86%, 72.36%, and 85.19%, respectively. GPT-6-Sol stays within a one-point range for all 200 papers, including 113 with ten identical scores.

## 4.2 Do Independent Reviews Reproduce Rankings?

We form disjoint equal-sized blocks from each paper’s ten scores, comparing means of one through five reviews. Enumerating all unordered block partitions gives 45, 630, 2,100, 1,575, and 126 partitions, respectively. Each configuration covers 9,900 within-year paper pairs, averaged over all partitions. Rankings are classified as the same strict order, a tie in at least one block, or reversed strict orders.

More reviews mainly improve strict ranking reproducibility by reducing ties. From one to five reviews per block, strict agreement rises from 62.14% to 84.47% for GPT-5.6-Sol and from 63.00% to 85.71% for Luna. GPT-6-Sol rises from 57.52% to 74.13%, with 25.08% of pairs still tied in at least one five-score block. Its reversal rate is 0.79%, below 2.15% and 2.44% for the other configurations. Finer means establish more strict comparisons, including reversals of pairs previously tied. We next specify a sampling target through absolute meanscore estimation error.

<table><tr><td>Model</td><td>Mean</td><td>Identical 10/10</td><td> $\mathrm { R a n g e } \le 1$ </td><td>SD</td><td>Spearman  $\rho$ </td><td>AUC</td></tr><tr><td>GPT-5.6-Sol</td><td>4.2755</td><td>30.0%</td><td>95.0%</td><td>0.374</td><td>0.378</td><td>0.713</td></tr><tr><td>GPT-6-Luna</td><td>3.4730</td><td>32.5%</td><td>91.5%</td><td>0.386</td><td>0.346</td><td>0.648</td></tr><tr><td>GPT-6-Sol</td><td>4.6635</td><td>56.5%</td><td>100.0%</td><td>0.272</td><td>0.409</td><td>0.714</td></tr></table>

Table 1: Repeated integer scores and external references. Each configuration covers 200 papers with ten reviews per paper. SD is the square root of average within-paper sample variance. Correlations, AUC, and related metrics use ten-score means.

<table><tr><td>Model</td><td>Reviews</td><td>Same</td><td>Tie</td><td>Reverse</td></tr><tr><td>GPT-5.6-Sol</td><td>1</td><td>62.14</td><td>36.71</td><td>1.15</td></tr><tr><td></td><td>2</td><td>73.52</td><td>24.74</td><td>1.74</td></tr><tr><td></td><td>3</td><td>78.92</td><td>19.08</td><td>2.00</td></tr><tr><td></td><td>4</td><td>82.18</td><td>15.71</td><td>2.10</td></tr><tr><td></td><td>5</td><td>84.47</td><td>13.38</td><td>2.15</td></tr><tr><td>GPT-6-Luna</td><td>1</td><td>63.00</td><td>35.60</td><td>1.40</td></tr><tr><td></td><td>2</td><td>74.91</td><td>23.05</td><td>2.04</td></tr><tr><td></td><td>3</td><td>80.40</td><td>17.30</td><td>2.30</td></tr><tr><td></td><td>4</td><td>83.58</td><td>14.03</td><td>2.39</td></tr><tr><td></td><td>5</td><td>85.71</td><td>11.85</td><td>2.44</td></tr><tr><td>GPT-6-Sol</td><td>1</td><td>57.52</td><td>42.14</td><td>0.34</td></tr><tr><td></td><td>2</td><td>66.43</td><td>33.03</td><td>0.54</td></tr><tr><td></td><td>3</td><td>70.24</td><td>29.11</td><td>0.65</td></tr><tr><td></td><td>4</td><td>72.56</td><td>26.71</td><td>0.73</td></tr><tr><td></td><td>5</td><td>74.13</td><td>25.08</td><td>0.79</td></tr></table>

Table 2: Agreement between disjoint review blocks (%). All ten observations participate in the partitions; Same denotes identical strict rankings, Tie a tie in at least one block, and Reverse opposite strict rankings. Values may not sum to 100 due to rounding.

## 4.3 How Many Reviews Are Enough?

Weighting the sampled papers equally, we define the overall success rate of k new reviews as

$$
A ( k , \epsilon ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { P r } \bigl ( | \bar { Y } _ { i , k } - \mu _ { i } | < \epsilon \bigr ) .
$$

We assume IID scores within each paper, independence across papers, and future scores independent of the pilot. We require $A ( k , \epsilon ) \geq 0 . 9 5$ This means that, for a randomly selected paper, the mean of repeated reviews falls within ϵ points of the model’s expected score with probability at least 95%. Let $\begin{array} { r } { V \stackrel { \textstyle - } { = } N ^ { - 1 } \sum _ { i } \sigma _ { i } ^ { 2 } } \end{array}$ . Under independent and identically distributed reviews, Chebyshev’s inequality gives

$$
A ( k , \epsilon ) \geq 1 - \frac { V } { k \epsilon ^ { 2 } } , \qquad k \geq \left\lceil \frac { V } { 0 . 0 5 \epsilon ^ { 2 } } \right\rceil .
$$

This budget depends only on the average variance and requires neither normally distributed scores nor a known score distribution. Substituting the average sample variance $\hat { V }$ from ten pilot scores gives planning budgets of 45, 48, and 24 reviews at a 0.25-point tolerance, and 12, 12, and 6 reviews at a 0.5-point tolerance.

Confidence control with finite pilot data. We also use the known score range [1, 10] to construct an upper confidence bound U for V. The sample variance of ten scores for each paper equals the average half-squared difference over all score pairs. Treating it as a U-statistic over unordered score pairs, we obtain a one-sided 95% upper bound depending only on $\hat { V }$ through random disjoint pairing and a KL-Chernoff bound (Hoeffding, 1963). Each model has an effective pairing count of $2 0 0 \lfloor 1 0 / 2 \rfloor = 1 0 0 0 ;$ the 9,000 score pairs that share observations are not treated as independent. Specifically, let $B = ( 1 0 - 1 ) ^ { 2 } / 2 = 4 0 . 5$ and $M = 1 0 0 0$ , and solve for the root above $\hat { V } ^ { \dagger }$

$$
M \operatorname { k l } ( { \hat { V } } / B , U / B ) = \log 2 0 ,
$$

where $\operatorname { k l } ( p , q ) = p \log ( p / q ) + ( 1 - p )$ log((1 − $p ) / ( 1 - q ) )$ . This inversion satisfies $\operatorname* { P r } ( V > U ) \leq$ 0.05. The key step expresses all overlapping score pairs as the conditional expectation of random disjoint matchings, then uses Jensen’s inequality to obtain the moment-generating-function bound for 1000 independent bounded kernels. Substituting U into the same budget formula gives a sufficient budget for an overall success rate of 95% on future new reviews, with 95% pilot confidence for each model.

The two budgets correspond to different information requirements: planning budgets reflect the scale of observed variance, while confidencecontrolled budgets also cover score tails that have not been sufficiently observed. Halving the tolerance approximately quadruples both budgets. In practice, the budget can be selected according to the error tolerance and confidence requirement. The bound’s construction, proof, and simultaneouscontrol results for all three models appear in the supplementary material.

<table><tr><td>Model</td><td> $\hat { V }$ </td><td> $U$ </td><td>Planning .25 / .5</td><td>Certified .25 / .5</td></tr><tr><td>GPT-5.6-Sol</td><td>0.1401</td><td>0.4114</td><td>45 / 12</td><td>132 /33</td></tr><tr><td>GPT-6-Luna</td><td>0.1489</td><td>0.4257</td><td>48 / 12</td><td>137 / 35</td></tr><tr><td>GPT-6-Sol</td><td>0.0741</td><td>0.2978</td><td>24/6</td><td>96 / 24</td></tr></table>

Table 3: Mean-score precision budgets without empirical score distributions. Planning uses $\hat { V } ;$ ; Certified uses the one-sided 95% upper bound U for $V .$ . The paired budgets correspond to error tolerances of 0.25 and 0.5 points. Budgets count new reviews and exclude the ten pilot scores; Certified confidence is computed separately for each model.

![](images/990325ea35e1e2e3d0d4fd7f9c609b5ffb7e0a50d66db24716be371d7728e0a5.jpg)  
Figure 3: Variance-based precision bounds at a halfpoint tolerance. Dashed curves plug in observed variance for planning; solid curves use the one-sided 95% variance upper bounds. The dotted horizontal line marks 95% success.

Figure 3 plots the budgets for the three configurations at a tolerance of 0.5 points as lower bounds on the success rate. Dashed curves substitute observed variances, and solid curves use their upper confidence bounds, corresponding to the planning scale and confidence-controlled sufficient budgets, respectively. The curves are calculated from the variance relationship without resampling the observed frequencies of the ten scores.

## 5 Can We Improve Scoring Reliability?

Repeated reviews improve the reliability of scores and paper rankings, but reaching reliable judgments about closely scored papers may require many independent calls. Can AI-score reliability be improved without substantially increasing review costs?

## 5.1 Finer Scores Change Both Variance and Location

If high-quality peer review is viewed as a ranking of paper quality, that ranking need not be inherently discrete. Scores, however, are discrete integers. This motivates testing whether finer output granularity improves model stability. We fix paper inputs and all other review requirements, replacing only the integer protocol with a 1–10 score required to have a nonzero fractional part. Luna and Sol each produce five repeated scores for all 200 papers.

Score variability decreases for both models, while their means rise by 0.693 and 0.266, respectively. Luna’s MAE relative to mean human scores also decreases, whereas Sol’s MAE remains largely unchanged. The decimal protocol thus changes both location and variability on the scoring scale. To estimate the original integer reviewer, we next retain its protocol and use additional information to improve estimation.

## 5.2 Estimating the Same Reviewer with Fewer Reasoning Calls

Beyond comparing integer and decimal scoring, we evaluate two simple strategies for estimating the integer score distribution $p _ { i }$ of a fixed GPT-5.6- Sol reasoning configuration with fewer reasoning calls. Quick extracts token probabilities for candidate integer scores from one non-reasoning call, normalizes them, adjusts their concentration using temperature $T ,$ and calibrates score locations through an affine mapping $s \mapsto a + b s$ . Mapped locations are clipped to [1, 10], and probability mass at non-integer positions is allocated linearly to the two adjacent score categories, yielding $q _ { Q }$ Standard combines this estimate with the empirical distribution of two target reasoning scores: $q _ { S } = w \hat { p } _ { 2 } + ( 1 - w ) q _ { Q }$

We use the 198 papers with available auxiliary probabilities and evaluate both strategies over 30 random paper-level splits, each containing 99 training and 99 test papers. Quick uses the first valid non-reasoning output for each paper. Its parameters $T , a ,$ b are selected by minimizing Brier loss against the training papers’ reasoning scores, over a grid of six temperatures, 13 offsets, and three slopes. With these parameters fixed, Standard selects w from eight candidate weights by enumerating all ten ways to divide each training paper’s five reasoning scores into two inputs and three independent targets. Evaluation on test papers follows the same input–target separation, with all parameters fixed from the training set.

<table><tr><td>Model</td><td>Integer / Decimal SD</td><td>SD change</td><td>Mean shift [95% CI]</td><td>MAE change [95% CI]</td></tr><tr><td>GPT-6-Luna</td><td>0.390 / 0.329</td><td>-15.6%</td><td>+0.693 [0.649, 0.739]</td><td>-0.307 [-0.389, -0.223]</td></tr><tr><td>GPT-6-Sol</td><td>0.286 / 0.249</td><td>-13.2%</td><td>+0.266 [0.233, 0.298]</td><td>-0.001 [-0.050, 0.044]</td></tr></table>

Table 4: Integer versus forced-decimal scoring. Shift and error changes are Decimal-Integer; brackets give paired paper-bootstrap 95% intervals. MAE uses public mean human review scores as the reference.

Risk evaluation with independent targets. For a predicted distribution q and independent target scores $Z _ { 1 } , \ldots , Z _ { m } .$ , we compute

$$
\begin{array} { l } { \displaystyle { \widehat { R } ( q ) = \| q \| _ { 2 } ^ { 2 } - \frac { 2 } { m } \sum _ { j } q ( Z _ { j } ) } } \\ { \displaystyle { + \frac { 1 } { m ( m - 1 ) } \sum _ { j \neq l } \mathbf { 1 } [ Z _ { j } = Z _ { l } ] . } } \end{array}
$$

Conditional on q, its expectation is $\| q - p _ { i } \| _ { 2 } ^ { 2 }$ The collision rate between target scores estimates $\| p _ { i } \| _ { 2 } ^ { 2 }$ , so variability from the finite target sample is not directly added to prediction risk. For n direct reasoning samples, risk is $( 1 - \| p _ { i } \| _ { 2 } ^ { 2 } ) / n ,$ also estimated using the collision rate of independent scores.

Standard reduces risk by 0.0388 relative to two direct reasoning samples, with a paired 95% interval of [-0.0524, -0.0255]; its risk difference from three reasoning samples is +0.0018 [-0.0057, 0.0095]. Auxiliary probabilities provide useful information alongside two reasoning samples, reaching a risk level close to three direct reasoning samples. Quick’s risk is close to that of a single reasoning sample, and fusion is more effective than replacing the scorer alone. The table reports the call composition separately; actual dollar costs depend on input length and caching.

The effect of fusion on risk can also be expressed directly. If the auxiliary call and the two reasoning scores are conditionally independent, and the empirical reasoning distribution is unbiased for $p _ { i } ,$ then $\mathbb { E } \Vert q _ { S } - p _ { i } \Vert _ { 2 } ^ { 2 } = w ^ { 2 } ( 1 - \Vert p _ { i } \Vert _ { 2 } ^ { 2 } ) / 2 + ( 1 -$ $w ) ^ { 2 } \mathbb { E } \lVert q _ { Q } - p _ { i } \rVert _ { 2 } ^ { 2 } .$ The reasoning term captures direct-sampling error, while the auxiliary term captures the discrepancy between the calibrated distribution and the target. Selecting the weight on training papers allocates risk between the two terms; debiased test-set evaluation assesses whether that choice remains effective on papers excluded from fitting.

## 5.3 Independent-Target Validation

A separate existing experiment collects ten additional independent reasoning scores for five papers from each year, using subsets of the original five scores for prediction. Quick and Standard are fitted on the other 188 papers. Standard obtains a Brier score of 0.353 and Wasserstein-1 distance of 0.19; three direct reasoning samples obtain 0.379 and 0.18, respectively. The independent targets give a similar estimation pattern to the main experiment. These losses include finite-target variability and are reported separately from the debiased risk in Table 5; Table 6 retains all five baselines. This ten-paper evaluation uses new target scores and separates evaluation papers from parameter training.

## 6 Can We Trust Revision Gains?

Accurately estimating one reviewer’s expected score does not establish whether other reviewers endorse its judgments. This matters particularly for AI-assisted manuscript revision: when a revised paper receives a higher score, can we trust that increase to reflect an effective improvement rather than a particular reviewer’s preferences? Through controlled manuscript revision and cross-model reviewing, we study whether different reviewers respond consistently to the same edits and how much of the observed disagreement can be explained by scoring randomness.

<table><tr><td>Method</td><td>R / Q calls</td><td>Risk</td><td>Risk 95% CI</td><td>Held-out AUC</td></tr><tr><td>Quick</td><td>0/1</td><td>0.2427</td><td>[0.2029, 0.2860]</td><td>0.7092</td></tr><tr><td>Standard</td><td>2/1</td><td>0.0829</td><td>[0.0750, 0.0911]</td><td>0.7194</td></tr><tr><td>1 Reasoning</td><td>1/0</td><td>0.2434</td><td>[0.2051, 0.2828]</td><td>0.6895</td></tr><tr><td>2 Reasoning</td><td>2/0</td><td>0.1217</td><td>[0.1025, 0.1414]</td><td>0.7063</td></tr><tr><td>3 Reasoning</td><td>3/0</td><td>0.0811</td><td>[0.0684, 0.0943]</td><td>0.7124</td></tr><tr><td>5 Reasoning</td><td>5/0</td><td>0.0487</td><td>[0.0410, 0.0566]</td><td>0.7162</td></tr></table>

Table 5: Held-out squared distribution risk. R/Q denote full reasoning reviews and non-reasoning probability calls, respectively. All risks use the shared 198 papers and the same held-out aggregation procedure; 95% intervals bootstrap papers conditional on the 30 fitted splits. Direct-sampling risk is calculated using the $1 / n$ relationship.

<table><tr><td>Method</td><td>Brier W1</td></tr><tr><td>Quick</td><td>0.497 0.36</td></tr><tr><td>Standard</td><td>0.353 0.19</td></tr><tr><td>1 Reasoning</td><td>0.552 0.28</td></tr><tr><td>3 Reasoning</td><td>0.379 0.18</td></tr><tr><td>5 Reasoning</td><td>0.344 0.14</td></tr></table>

Table 6: Existing validation with independent target reviews. Five papers per year, each with ten new target scores; all methods use the same ten papers. Quick/Standard are fitted on the other 188 papers, and reasoning baselines enumerate subsets of the original five scores. Brier is the squared loss against individual one-hot targets; W1 is distance to the target empirical CDF, in score points.

## 6.1 Holding Revisions Fixed

With more precise score estimation, we examine another use case: AI scores as revision feedback. The workflow has three roles: the Writer generates candidate text edits from the original manuscript and existing reviews; the Selector decides whether to accept each edit; and the Scorer evaluates the complete manuscript after edits are applied. The Scorer does not receive selection rationales. Revisions are required to preserve facts, experimental data, and non-text materials.

Candidate edits generated by Luna are selected by Luna or Haiku and then scored by both, forming a complete 2 × 2 crossover. Haiku-generated edits add three role combinations, for seven configurations in total. Each Scorer independently scores the original and revised manuscripts three times each, and all comparisons use the same 199 papers. Of Luna’s 9,327 candidate edits, 8,459 are accepted by both Selectors, with a mean per-paper acceptance-set Jaccard similarity of 0.919; the final versions still differ for 182 papers. This lets us compare selection effects with fixed candidate edits and scoring effects with fixed final versions.

![](images/f7cdc3f0cd29a59b241052958373d1622bec9df4acddbf0da96834ef8b30b681.jpg)  
Figure 4: Mean revision gains for identical manuscript versions under different scorers. Bars show paper-level bootstrap 95% intervals.

## 6.2 Opposite Gains for the Same Revisions

For identical revisions generated and selected by Luna, Luna assigns a mean gain of +0.474 and Haiku -0.348, a difference of 0.822 [0.717, 0.928]. Replacing the Selector with Haiku still yields +0.472 and -0.296, respectively; the cross-scorer gap changes by only -0.054 [-0.116, 0.010]. Figure 4 shows the separation in scorer responses under the same selection condition.

Luna also assigns positive average gains to Haiku-generated revisions, while Haiku assigns its own selected revisions a gain of +0.050. The result is asymmetric: Luna endorses revisions from both Writers; Haiku assigns negative gains to Luna’s revisions and near-zero gains to its own. Changing the Selector has a smaller effect on the aggregate results for Luna’s candidate edits. Across these two sets of Luna revisions, the final Scorer changes the direction of the gain.

<table><tr><td>Writer</td><td>Selector</td><td>Scorer</td><td>Mean gain</td><td>95% CI</td><td>+/0/-</td></tr><tr><td>Luna</td><td>Luna</td><td>Luna</td><td>+0.474</td><td>[0.367, 0.578]</td><td>120 / 54 /25</td></tr><tr><td>Luna</td><td>Haiku</td><td>Luna</td><td>+0.472</td><td>[0.362, 0.580]</td><td>129 / 41 / 29</td></tr><tr><td>Luna</td><td>Luna</td><td>Haiku</td><td>-0.348</td><td>[-0.432, -0.265]</td><td>32 / 63 / 104</td></tr><tr><td>Luna</td><td>Haiku</td><td>Haiku</td><td>-0.296</td><td>[-0.379, -0.214]</td><td>31 / 63 / 105</td></tr><tr><td>Haiku</td><td>Luna</td><td>Luna</td><td>+0.302</td><td>[0.238, 0.367]</td><td>108 / 70 / 21</td></tr><tr><td>Haiku</td><td>Haiku</td><td>Luna</td><td>+0.281</td><td>[0.216, 0.347]</td><td>96 / 84 / 19</td></tr><tr><td>Haiku</td><td>Haiku</td><td>Haiku</td><td>+0.050</td><td>[0.000, 0.104]</td><td>57 / 102 / 40</td></tr></table>

Table 7: Revision gains across writer, selector, and scorer. Each configuration includes 199 papers, with three scores each for the original and revised versions. The 95% intervals use 10,000 paper-level paired bootstrap replicates. The final column counts papers with positive/zero/negative differences between three-score means.

<table><tr><td>Target</td><td>Pearson</td><td>Adjusted Pearson [95% CI]</td></tr><tr><td>Original scores</td><td>0.804</td><td>0.843 [0.783, 0.890]</td></tr><tr><td>Luna / Luna</td><td>0.415</td><td>0.521 [0.336, 0.686]</td></tr><tr><td>Luna / Haiku</td><td>0.407</td><td>0.518 [0.337, 0.673]</td></tr><tr><td>Haiku / Haiku</td><td>0.085</td><td></td></tr></table>

Table 8: Cross-scorer correlation before and after revision. Adjusted subtracts the scoring-noise variance of three-score means from the observed between-paper variance; intervals bootstrap papers. This diagnostic assumes independent, zero-mean scoring errors across models and manuscript versions. Gain rows: Writer/Selector.

These controls serve different purposes. Holding the final manuscript fixed and changing the Scorer directly measures differing responses to the same textual edits; holding the Writer’s candidate edits fixed and changing the Selector tests whether selection changes this gap. The two Selectors accept many of the same edits. Even when final versions differ, Luna and Haiku assign opposite aggregate gains under both selection conditions. Cross-reviewer disagreement can thus be studied without comparing two revisions with different content.

## 6.3 Agreement on Originals Does Not Predict Agreement on Gains

The two Scorers have a Pearson correlation of 0.804 on original-paper means, but their paperlevel gain correlations for identical revisions fall to 0.415, 0.407, and 0.085. Gain directions agree exactly for 28.1%, 30.2%, and 41.2% of papers, respectively, including cases in which both assign zero gain.

We estimate the noise contribution from repeated scores: the noise variance of original-paper means is $s _ { o } ^ { 2 } / 3$ , and that of gains is $( s _ { o } ^ { 2 } + s _ { v } ^ { 2 } ) / 3$ We subtract these components from the corresponding between-paper variances while retaining the cross-model covariance. Adjusted correlations for Luna revisions are approximately 0.52, still below 0.843 for original papers. Scoring fluctuations explain part of the correlation decline, while opposite aggregate gains for fixed revised versions directly demonstrate differing reviewer responses.

## 7 Conclusion

Using AI scores as research feedback involves two inferences: from individual outputs to expected judgments, and from those judgments to revision gains. Repeated reviews improve mean-score precision, with budgets determined by average variance and error tolerance and further controlled for uncertainty in finite pilot data. Under a fixed scoring protocol, auxiliary probabilities combined with a small number of reasoning scores improve distribution estimation; changing output granularity alters both variability and the scoring center. The revision crossover further shows that even highly correlated original-paper judgments can lead to gains in opposite directions for identical revisions. Score precision clarifies one reviewer’s judgment, while cross-reviewer evaluation of shared revisions tests whether this feedback receives consistent endorsement.

## Limitations

Sampling budgets target new independent scores under a fixed configuration; changes to model versions, prompts, or service environments change the target. Confidence bounds use the full 1–10 range and are conservative when tail observations are limited. Human scores come from current public records, and PDFs may be updated versions; the two years use different rating categories, and correlations do not directly observe latent quality. Calibration is validated on one API configuration, while the revision crossover includes only two models and omits one role combination. Factual preservation in revisions is required by the editing protocol but has not been validated through blinded human assessment of compliance and quality gains; using existing reviews to generate edits may also affect the independence assumption in noise adjustment. Experiments use public anonymized manuscripts. The findings analyze scoring behavior; automated editing and scoring do not replace authors’ responsibility for paper content.

## References

AAAI. 2025. AAAI launches AI-powered peer review assessment system. Association for the Advancement of Artificial Intelligence.

Cheng-Han Chiang and Hung-yi Lee. 2023. A closer look into using large language models for automatic evaluation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 8928– 8942, Singapore. Association for Computational Linguistics.

Shurui Du. 2025. TitleTrap: Probing presentation bias in LLM-based scientific reviewing. In Proceedings of the 5th Workshop on Evaluation and Comparison of NLP Systems, pages 119–125, Mumbai, India. Association for Computational Linguistics.

Madhav Krishan Garg, Tejash Prasad, Tanmay Singhal, Chhavi Kirtani, Murari Mandal, and Dhruv Kumar. 2025. ReviewEval: An evaluation framework for AI-generated reviews. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20542–20564, Suzhou, China. Association for Computational Linguistics.

Wassily Hoeffding. 1963. Probability inequalities for sums of bounded random variables. Journal of the American Statistical Association, 58(301):13–30.

Christopher M. Homan, Flip Korn, Deepak Pandita, and Chris Welty. 2026. How many ratings per item are

necessary for reliable significance testing? In Findings of the Association for Computational Linguistics: EACL 2026, pages 4258–4273, Rabat, Morocco. Association for Computational Linguistics.

Lingyao Li, Junjie Xiong, Changjia Zhu, Runlong Yu, Chen Chen, Junyu Wang, Renkai Ma, and Zhicong Lu. 2026a. LLM-as-a-reviewer: Benchmarking their ability, divergence, and prompt injection resistance as paper reviewers. Preprint, arXiv:2605.25415.

Ming Li, Chenguang Wang, Xirui Li, Xinyue Zeng, Dianqi Li, Peng Shi, Dawei Zhou, and Tianyi Zhou. 2026b. How can rhetoric reward-hack AI reviewers? dissecting rhetorical sensitivity in AI-based peer review. Preprint, arXiv:2608.08975.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. 2023. G-eval: NLG evaluation using Gpt-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522, Singapore. Association for Computational Linguistics.

Rickard Stureborg, Dimitris Alikaniotis, and Yoshi Suhara. 2024. Large language models are inconsistent and biased evaluators. Preprint, arXiv:2405.01724.

James Zou and Nitya Thakkar. 2025. Leveraging LLM feedback to enhance review quality. ICLR Blog.

## A Data and Sampling

## A.1 Submission Populations and Sample Composition

The study includes 100 papers each from ICLR 2025 and ICLR 2026. The sampling populations contain 11,520 and 19,474 submissions with publicly available official reviews, respectively, including accepted, rejected, withdrawn, and deskrejected papers. The paper lists were fixed before downloading PDFs or running AI reviews. AI scores, PDF availability, and overlap with earlier samples played no role in selection.

The 2025 sample was stratified jointly by official primary subject area and mean human-score bin. The 2026 sample uses the second version of the sampling design, which adds submission status to these dimensions.

## A.2 Stratified Quotas and Within-Stratum Sampling

For 2025, integer quotas for three nested sample sizes of 100, 300, and 1,000 were jointly solved using SciPy milp and HiGHS. For each subject area, score bin, and nonempty area-by-score-bin stratum, the sample count was constrained to the floor or ceiling of its population share multiplied by the sample size, keeping the count discrepancy strictly below one paper. Additional constraints ensured coverage of subject areas and nonempty score bins. Within each stratum, papers were sorted by their exact mean human score, with random ordering of ties, and selected by systematic sampling with a random start. The sample of 1,000 was drawn first, followed by 300 from that sample and then 100 from the 300.

The second 2026 design uses area-by-score-binby-status strata. Status categories are Poster, Oral, rejected, withdrawn, and desk rejected. Status quotas were assigned by the largest-remainder method, with the total number of accepted papers fixed at 27. Area, score-bin, and cross-stratum quotas retained the floor/ceiling constraints, and within-stratum selection again used systematic sampling after sorting by exact mean score. Section A.3 reports the final status quotas and counts.

The human mean for each paper is the unweighted average of its valid official review scores. Bins are assigned using the exact, unrounded mean. Nonempty bins range from [1,2) to [9,10] in 2025. The 2026 population includes means below 1, with nonempty bins from [0,1) to [8,9); its [9,10] bin is empty. The complete distributions follow.

## A.3 Subject-Area and Submission-Status Distributions

All 21 official primary subject areas are represented. The table retains the original area names without merging small areas.

The 2025 sample contains 36 accepted, 41 rejected, and 23 withdrawn papers. The 2026 sample contains 27 accepted, 43 rejected, 26 withdrawn, and 4 desk-rejected papers. Together, these comprise 63 accepted, 84 rejected, 49 withdrawn, and 4 desk-rejected papers.

Quota rounding allows rare categories to enter a small sample. For example, only nine papers in the 2025 population have a mean human score of at least 9, and one is retained in the sample of 100: a sample share of 1% versus a population share of approximately 0.078%. Coverage of subject areas and score bins does not imply coverage of every small cross-stratum; some receive a rounded quota of zero. Results give equal weight to the selected papers, without reweighting to population shares.

## A.4 Human Scores, PDF Versions, and Acceptance Labels

Human scores are taken from the public records in the frozen data and may include post-rebuttal updates. Paper PDFs may also be updated versions. Human score options are 1, 3, 5, 6, 8, and 10 for 2025, and 0, 2, 4, 6, 8, and 10 for 2026. Section C therefore also reports associations and errors separately by year.

Acceptance labels are derived from final public status: accepted\_\* is labeled 1, and rejected, withdrawn, and desk-rejected papers are labeled 0. The primary analysis uses all 200 papers, with 63 positives and 137 negatives. Excluding withdrawals leaves 151 papers: 63 positives and 88 negatives. The full and withdrawal-excluded samples contain 100 and 77 papers in 2025, and 100 and 74 in 2026. Positive-negative pairs with equal predicted scores receive a contribution of 0.5 to AUC.

## A.5 Shared Samples and Paper Lists Across Experiments

The two papers missing auxiliary probabilities differ from the paper affected by Claude’s image limit. The full calibration sample contains 198 papers, with 63 accepted and 135 nonaccepted. Excluding its 48 withdrawn papers leaves 150 papers, with 63 accepted and 87 nonaccepted. The revision experiment uses a fixed shared set of 199 papers: 100 from 2025 and 99 from 2026.

<table><tr><td>Item</td><td>ICLR 2025</td><td>ICLR 2026</td></tr><tr><td>Population with public official reviews</td><td>11,520</td><td>19,474</td></tr><tr><td>Study sample</td><td>100</td><td>100</td></tr><tr><td>Official primary subject areas</td><td>21</td><td>21</td></tr><tr><td>Population mean human score</td><td>5.1496</td><td>4.2027</td></tr><tr><td>Sample mean human score</td><td>5.1983</td><td>4.2117</td></tr><tr><td>Population acceptance rate</td><td>32.14%</td><td>27.48%</td></tr><tr><td>Sample acceptance rate</td><td>36.00%</td><td>27.00%</td></tr><tr><td>Maximum area-share deviation (pp)</td><td>0.4635</td><td>0.5573</td></tr><tr><td>Maximum score-bin share deviation (pp)</td><td>0.9219</td><td>0.9131</td></tr><tr><td>Acceptance status constrained in sampling</td><td>No</td><td>Yes</td></tr></table>

Table S1: Sampling and representativeness in the two years.

<table><tr><td>Category</td><td>2025 population</td><td>2025 sample</td><td>Deviation (pp)</td><td>2026 population</td><td>2026 sample</td><td>Deviation (pp)</td></tr><tr><td>[0,1)</td><td>0</td><td>0</td><td>+0.000</td><td>54</td><td>1</td><td>+0.723</td></tr><tr><td>[1, 2)</td><td>25</td><td>1</td><td>+0.783</td><td>236</td><td>1</td><td>-0.212</td></tr><tr><td>[2, 3)</td><td>243</td><td>2</td><td>-0.109</td><td>1946</td><td>10</td><td>+0.007</td></tr><tr><td>[3, 4)</td><td>1643</td><td>14</td><td>-0.262</td><td>4458</td><td>23</td><td>+0.108</td></tr><tr><td>[4,5)</td><td>2928</td><td>25</td><td>-0.417</td><td>6927</td><td>35</td><td>-0.571</td></tr><tr><td>[5, 6)</td><td>3142</td><td>27</td><td>-0.274</td><td>4063</td><td>21</td><td>+0.136</td></tr><tr><td>[6, 7)</td><td>2585</td><td>22</td><td>-0.439</td><td>1541</td><td>7</td><td>-0.913</td></tr><tr><td>[7, 8)</td><td>817</td><td>7</td><td>-0.092</td><td>228</td><td>1</td><td>-0.171</td></tr><tr><td>[8,9)</td><td>128</td><td>1</td><td>-0.111</td><td>21</td><td>1</td><td>+0.892</td></tr><tr><td>[9,10]</td><td>9</td><td>1</td><td>+0.922</td><td>0</td><td>0</td><td>+0.000</td></tr></table>

Table S2: Population and sample distributions of mean human-score bins. Deviations are sample shares minus population shares, in percentage points; empty bins are shown as zero.

![](images/4cb18c60562308448ee1b68fbfc2f4136cc86876a7108be0f4c2d027e353e4d0.jpg)

![](images/246c0135e722ea0098b4e6fed879c7f1f5d073fb639c05a4b8a6767324cdc19b.jpg)  
Figure A1: Score-bin shares in each year’s population and sample of 100 papers. The plotted bins follow the nonempty range for each year.

<table><tr><td>Category</td><td>2025 pop.</td><td>2025 sample</td><td>Deviation (pp)</td><td>2026 pop.</td><td>2026 sample</td><td>Deviation (pp)</td></tr><tr><td>alignment, fairness, safety, privacy, and societal considerations</td><td>1039</td><td>9</td><td>-0.019</td><td>1667</td><td>9</td><td>+0.440</td></tr><tr><td>applications to computer vision, audio, language, and other modalities</td><td>1359</td><td>12</td><td>+0.203</td><td>2687</td><td>14</td><td>+0.202</td></tr><tr><td>applications to neuroscience &amp; cognitive science</td><td>214</td><td>2</td><td>+0.142</td><td>372</td><td>2</td><td>+0.090</td></tr><tr><td>applications to physical sciences (physics, chemistry, biology, etc.)</td><td>540</td><td>5</td><td>+0.313</td><td>846</td><td>4</td><td>-0.344</td></tr><tr><td>applications to robotics, autonomy, planning</td><td>242</td><td>2</td><td>-0.101</td><td>589</td><td>3</td><td>-0.025</td></tr><tr><td>causal reasoning</td><td>116</td><td>1</td><td>-0.007</td><td>189</td><td>1</td><td>+0.029</td></tr><tr><td>datasets and benchmarks</td><td>753</td><td>7</td><td>+0.464</td><td>1551</td><td>8</td><td>+0.036</td></tr><tr><td>foundation or frontier models, including LLMs</td><td>1281</td><td>11</td><td>-0.120</td><td>2623</td><td>13</td><td>-0.469</td></tr><tr><td>generative models</td><td>1045</td><td>9</td><td>-0.071</td><td>1646</td><td>9</td><td>+0.548</td></tr><tr><td>infrastructure, software libraries, hardware, systems, etc.</td><td>86</td><td>1</td><td>+0.253</td><td>176</td><td>1</td><td>+0.096</td></tr><tr><td>interpretability and explainable AI</td><td>505</td><td>4</td><td>-0.384</td><td>776</td><td>4</td><td>+0.015</td></tr></table>

Table S3: Distributions across the 21 primary subject areas, using the official labels for each year.

<table><tr><td>Category</td><td>2025 pop.</td><td>2025 sample</td><td>Deviation (pp)</td><td>2026 pop.</td><td>2026 sample</td><td>Deviation (pp)</td></tr><tr><td>learning on graphs and other geometries &amp; topologies</td><td>428</td><td>4</td><td>+0.285</td><td>528</td><td>3</td><td>+0.289</td></tr><tr><td>learning on time series and dynamical systems</td><td>265</td><td>2</td><td>-0.300</td><td>457</td><td>2</td><td>-0.347</td></tr><tr><td>learning theory</td><td>358</td><td>3</td><td>-0.108</td><td>498</td><td>2</td><td>-0.557</td></tr><tr><td>neurosymbolic &amp; hybrid AI systems (physics-informed, logic &amp; formal reasoning, etc.)</td><td>105</td><td>1</td><td>+0.089</td><td>222</td><td>1</td><td>-0.140</td></tr><tr><td>optimization</td><td>509</td><td>4</td><td>-0.418</td><td>725</td><td>4</td><td>+0.277</td></tr><tr><td>other topics in machine learning (i.e., none of the above)</td><td>551</td><td>5</td><td>+0.217</td><td>795</td><td>4</td><td>-0.082</td></tr><tr><td>probabilistic methods (Bayesian methods, variational inference, sampling, UQ, etc.)</td><td>235</td><td>2</td><td>-0.040</td><td>371</td><td>2</td><td>+0.095</td></tr><tr><td>reinforcement learning</td><td>666</td><td>6</td><td>+0.219</td><td>1050</td><td>5</td><td>-0.392</td></tr><tr><td>transfer learning, meta learning, and lifelong learning</td><td>370</td><td>3</td><td>-0.212</td><td>560</td><td>3</td><td>+0.124</td></tr><tr><td>unsupervised, self-supervised, semi-supervised, and supervised representation learning</td><td>853</td><td>7</td><td>-0.405</td><td>1146</td><td>6</td><td>+0.115</td></tr></table>

Table S3: Distributions across the 21 primary subject areas, using the official labels for each year. (continued)

<table><tr><td>Category</td><td>2025 population</td><td>2025 sample</td><td>Deviation (pp)</td><td>2026 population</td><td>2026 sample</td><td>Deviation (pp)</td></tr><tr><td>accepted_oral</td><td>213</td><td>3</td><td>+1.151</td><td>224</td><td>1</td><td>-0.150</td></tr><tr><td>accepted_poster</td><td>3110</td><td>30</td><td>+3.003</td><td>5127</td><td>26</td><td>-0.327</td></tr><tr><td>accepted_spotlight</td><td>380</td><td>3</td><td>-0.299</td><td>0</td><td>0</td><td>+0.000</td></tr><tr><td>desk_rejected</td><td>8</td><td>0</td><td>-0.069</td><td>685</td><td>4</td><td>+0.482</td></tr><tr><td>rejected</td><td>4908</td><td>41</td><td>-1.604</td><td>8352</td><td>43</td><td>+0.112</td></tr><tr><td>withdrawn</td><td>2901</td><td>23</td><td>-2.182</td><td>5086</td><td>26</td><td>-0.117</td></tr></table>

Table S4: Submission-status distributions. The 2025 categories include Spotlight, which is absent from the 2026 status classification used here.

<table><tr><td>Experiment</td><td>papers</td><td>Shared Scores or inputs per paper Sample details</td><td></td></tr><tr><td>Repeated integer scores on originals</td><td></td><td>200 10 per model, three models Historical and additional runs</td><td>combined: 6,000 scores</td></tr><tr><td>Integer/forced-decimal comparison</td><td></td><td>200 Luna and Sol: 5 per protocol Integer controls: first five under</td><td>same configuration; 1,000 decimal scores each</td></tr><tr><td>Probability-assisted distribution estimation</td><td></td><td>198 5 target reasoning scores plus auxiliary probabilities</td><td>Missing probabilities: 2025- paper_056 and 2026-paper_ 053</td></tr><tr><td>Independent-target validation</td><td></td><td>10 5 existing scores for prediction; 10 additional target scores</td><td>5 per year; parameters fitted on the other 188</td></tr><tr><td>Revision crossover</td><td></td><td>revision</td><td>199 3 each for original and each Claude excludes 2026-paper_ 067, which has 224 images</td></tr></table>

Table S5: Shared samples and scoring inputs across experiments.

Scoring models receive anonymized manuscript content. Human scores, acceptance status, and other models’ results remain in the analysis metadata.

## B Review and Revision Protocols

## B.1 Models and Review Settings

Each experiment fixes its review configuration. Sections C and D use ten original-manuscript scores per paper, whereas distribution estimation in Section E uses a separately stored set of five target reasoning scores per paper.

Auxiliary probability calls use reasoning.effort="none", top\_logprobs=20, and max\_output\_tokens=4096, and request output-token log probabilities. Target reasoning scores use the same manuscript input and scoring schema, with reasoning.effort="medium" and an output budget of 16,000 tokens. Temperature is not explicitly set in the requests.

Each review runs in an independent session. Models can access only the preloaded manuscript text and images; tools, web search, memory, and prior context are disabled. Analyses use valid results that pass format and execution checks.

## B.2 Manuscript Inputs and Image Processing

Each original manuscript is represented by text.json, index.json, and figures/. Text records contain a stable id, PDF page number, record type, and text. The sequence in index.json specifies the interleaving of text and images. The input loader follows this sequence to assemble the full text, table text, references, and appendices. Images are loaded after relative-path and SHA-256 checks. Adjacent text records may be merged without changing their order.

All scorers receive preloaded text and images. Haiku processes images in the same order, resizing the longest side to 1,568 pixels and re-encoding them before submission, with a limit of 100 images per call. Paper 2026-paper\_067 contains 224 images and exceeds this limit, so the revision crossover experiment uses the remaining 199 papers.

Across the 200 input packages, PDF page markers total 4,923 pages and image-index entries total 2,584. The median, 90th percentile, and maximum page counts are 23, 38, and 69; the corresponding image counts are 9, 23, and 224. There are 133,967 text records containing 15,324,311 characters. Page counts come from PDF page markers in the packages, and image counts from index entries rather than individual figure or table panels.

## B.3 Integer-Scoring Task

The scorer acts as an ICLR reviewer, reads the complete anonymized manuscript with interleaved images, evaluates novelty, technical correctness, experimental evidence, and limitations, and returns an integer overall score from 1 to 10. The anchors are 1 for strong reject, 3 for reject, 5 for marginally below the acceptance threshold, 6 for marginally above it, 8 for accept, and 10 for strong accept. Other intermediate integers are permitted.

<table><tr><td>Purpose</td><td>Model identifier</td><td>Reasoning Calls per paper setting</td><td></td></tr><tr><td>Repeated integer scores on originals</td><td>gpt-5.6-sol</td><td>medium</td><td>10</td></tr><tr><td>Repeated integer scores on originals</td><td>gpt-6-luna</td><td>low</td><td>10</td></tr><tr><td>Repeated integer scores on originals</td><td>gpt-6-sol</td><td>low</td><td>10</td></tr><tr><td>Calibration target reasoning scores</td><td>gpt-5.6-sol</td><td>medium</td><td>5</td></tr><tr><td>Quick auxiliary probability calls gpt-5.6-sol</td><td></td><td>none</td><td>1 for prediction</td></tr><tr><td>Forced-decimal scoring</td><td>gpt-6-luna, gpt- 6-sol</td><td>low</td><td>5</td></tr><tr><td>Luna revision scorer</td><td>gpt-6-luna</td><td>low</td><td>3 each for original and revised</td></tr><tr><td>Haiku revision scorer</td><td>claude-haiku-5-5 low</td><td></td><td>3 each for original and revised</td></tr><tr><td>Writer</td><td>Luna or Haiku</td><td>medium</td><td>1 valid candidate manuscript per paper</td></tr><tr><td>Selector</td><td>Luna or Haiku</td><td>high</td><td>1 valid decision set per paper</td></tr></table>

Table S6: Models and reasoning settings.

The output contains an overall score, confidence from 1 to 5, a brief summary, up to three strengths, up to three concerns, and a scoring rationale. Key judgments cite PDF page numbers. The scorer does not receive human scores, final decisions, or other reviews. Confidence is not used to weight scores.

## B.4 Forced-Decimal Protocol

The decimal experiment keeps the inputs, models, and other requirements unchanged, replacing the integer requirement with a mandatory nonzero fractional part. The score field changes from integer to number, while its range remains 1–10. Local validation additionally checks that the score is noninteger. All 1,000 valid scores from each model satisfy this requirement.

An earlier pilot that merely allowed decimals was discarded and is excluded from all analyses. The full forced-decimal experiment reused the 20- paper pilot run under the same protocol, then incorporated the remaining papers to obtain five scores each for 200 papers per model. The integer controls were verified paper by paper and round by round against the first five scores in the ten-review original-manuscript experiment, matching the number of repetitions.

## B.5 Inputs to the Writer, Selector, and Scorer

The Luna Writer uses the first existing GPT-5.6- Sol review as feedback; source records identify this configuration for all 200 papers. The Haiku Writer uses Haiku’s first review of the original manuscript. The two Writer conditions thus follow their respective feedback pipelines. With the Writer fixed, both Selectors read the same candidate diffs. With the final manuscript fixed, both Scorers read the same text version. Scorers receive no candidate diffs, selection rationales, other scorers’ scores, or comparative prompt identifying the manuscript as revised.

The Writer returns text records to replace. The Selector gives an acceptance decision and a reason for each candidate edit, with exactly one decision per candidate. The program validates record IDs, decision counts, and output format before generating the final manuscript.

The Writer is instructed to preserve facts, numerical values, data, evidential scope, record order, page numbers, and nontext fields. Leading instructions and page markers must remain unchanged. Returned text is merged into a candidate manuscript, from which block-level diffs are generated. Each Selector comment is limited to 32 characters; truncations of longer Claude comments are recorded. The final manuscript is copied from the original records, replacing only the text of accepted edits with candidate text. Section F.5 reports structural checks.

<table><tr><td>Role</td><td>Input</td><td>Output</td></tr><tr><td>Writer</td><td>Full original, images, one specified existing first List of replacement text records id, text review</td><td></td></tr><tr><td>Selector</td><td>Full original, images, all candidate diffs</td><td>id, accept, comment for each diff</td></tr><tr><td>Scorer</td><td>Full selected revision and images</td><td>Same scoring JSON as for originals</td></tr></table>

Table S7: Information inputs and outputs for the three revision roles.

## C Extended Score Reliability Results

## C.1 Metrics and Aggregation Units

Let the ten scores for paper i be $Y _ { i 1 } , \dots , Y _ { i , 1 0 } .$ , with mean ${ \bar { Y } } _ { i }$ . Its unbiased within-paper variance is

$$
S _ { i } ^ { 2 } = \frac { 1 } { 9 } \sum _ { t = 1 } ^ { 1 0 } ( Y _ { i t } - \bar { Y } _ { i } ) ^ { 2 } .
$$

The mean within-paper variance is $\begin{array} { r l } { \hat { V } } & { { } = } \end{array}$ $N ^ { - 1 } \sum _ { i } S _ { i } ^ { 2 }$ , and the within-paper SD reported here is $\sqrt { \hat { V } }$ . The SD of all scores additionally includes differences in mean scores between papers. The identical-10/10 rate is the fraction of papers with ten identical observations, and the range is the maximum minus the minimum score. Pairwise exact agreement is first averaged over the ${ \binom { 1 0 } { 2 } } = 4 5$ unordered score pairs within each paper, then across papers.

All GPT-6-Sol papers have a score range of at most one point, and 113 have ten identical scores. The corresponding identical-score counts are 60 for GPT-5.6-Sol and 65 for Luna. Year-specific results show the same configuration differences: Sol’s within-paper SD is 0.2614 in 2025 and 0.2824 in 2026.

## C.2 Association with Human Scores and Final Acceptance

Pearson correlation, Spearman correlation, MAE, and RMSE compare each paper’s ten-review mean with its public mean human score. AUC uses the same ten-review mean to predict the acceptance labels defined in Section A.4.

Spearman correlations range from 0.4017 to 0.5457 in 2025 and from 0.1572 to 0.2345 in 2026. Full-sample AUCs range from 0.6756 to 0.7852 in 2025 and from 0.6010 to 0.6185 in 2026. These year-specific results use the scores and labels for the corresponding year.

## C.3 Score Concentration and Paper-Level Variance Distributions

Luna’s scores are concentrated at 2–4. Sol assigns a score of 5 in 1,042 cases, or 52.1% of its 2,000 scores. Scores from 5 to 7 account for 65.20% of Sol’s outputs, most of them at 5 or 6. The range of repeated scores and numerical separation between papers therefore describe different aspects of scoring.

## C.4 First Five Versus Last Five Reviews

The ten reviews combine historical scores with later runs under the same configuration. The table separates rounds 1–5 from rounds 6–10, computes the difference between batch means within each paper, and bootstraps papers.

For all three models, the interval for the overall difference between batches includes zero. This analysis records shifts between batch positions; the precision budgets in Section D concern the expectation of future independent scores under a fixed configuration.

## C.5 Ranking Reproducibility Across All Ten Reviews

For each paper, we construct two nonoverlapping blocks of k scores from its ten reviews. Treating the pair of blocks as unordered gives

$$
{ \begin{array} { c } { L _ { k } = { \frac { 1 } { 2 } } { \binom { 1 0 } { k } } { \binom { 1 0 - k } { k } } , } \\ { k = 1 , \ldots , 5 . } \end{array} }
$$

<table><tr><td>Model</td><td colspan="2">Year</td><td>N</td><td>Mean</td><td>Identical 10/10 Range ≤1</td></tr><tr><td>GPT-5.6-Sol</td><td>All</td><td>200</td><td>4.2755</td><td>30.0%</td><td>95.0%</td></tr><tr><td>GPT-5.6-Sol</td><td>2025</td><td>100</td><td>4.4550</td><td>24.0%</td><td>96.0%</td></tr><tr><td>GPT-5.6-Sol</td><td>2026</td><td>100</td><td>4.0960</td><td>36.0%</td><td>94.0%</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>200</td><td>3.4730</td><td>32.5%</td><td>91.5%</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>100</td><td>3.8050</td><td>25.0%</td><td>93.0%</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>100</td><td>3.1410</td><td>40.0%</td><td>90.0%</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>200</td><td>4.6635</td><td>56.5%</td><td>100.0%</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>100</td><td>4.8550</td><td>61.0%</td><td>100.0%</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>100</td><td>4.4720</td><td>52.0%</td><td>100.0%</td></tr><tr><td colspan="6"></td></tr><tr><td>Model</td><td>Year</td><td>N</td><td>Pairwise agreement</td><td>Mean within-paper variance</td><td>Within-paper SD</td></tr><tr><td>GPT-5.6-Sol</td><td>All</td><td>200</td><td>72.86%</td><td>0.140056</td><td>0.3742</td></tr><tr><td>GPT-5.6-Sol</td><td>2025</td><td>100</td><td>71.04%</td><td>0.146111</td><td>0.3822</td></tr><tr><td>GPT-5.6-Sol</td><td>2026</td><td>100</td><td>74.67%</td><td>0.134000</td><td>0.3661</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>200</td><td>72.36%</td><td>0.148889</td><td>0.3859</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>100</td><td>69.60%</td><td>0.160333</td><td>0.4004</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>100</td><td>75.11%</td><td>0.137444</td><td>0.3707</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>200</td><td>85.19%</td><td>0.074056</td><td>0.2721</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>100</td><td>86.33%</td><td>0.068333</td><td>0.2614</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>100</td><td>84.04%</td><td>0.079778</td><td>0.2824</td></tr></table>

Table S8: Complete repeatability results for ten original-manuscript reviews. Each model has 200 papers and 2,000 scores overall, with 100 papers per year.

<table><tr><td>Model</td><td>Year</td><td>Pearson</td><td>Spearman</td><td>MAE</td><td>RMSE</td><td>AUC: all</td><td>AUC: excl. withdrawn</td></tr><tr><td>GPT-5.6-Sol</td><td>All</td><td>0.4498</td><td>0.3777</td><td>1.1034</td><td>1.4003</td><td>0.7134</td><td>0.7128</td></tr><tr><td>GPT-5.6-Sol</td><td>2025</td><td>0.5782</td><td>0.5206</td><td>1.0878</td><td>1.3859</td><td>0.7812</td><td>0.7558</td></tr><tr><td>GPT-5.6-Sol</td><td>2026</td><td>0.2344</td><td>0.2031</td><td>1.1190</td><td>1.4145</td><td>0.6185</td><td>0.6438</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>0.4234</td><td>0.3457</td><td>1.4597</td><td>1.8345</td><td>0.6484</td><td>0.6438</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>0.4654</td><td>0.4017</td><td>1.5527</td><td>1.8996</td><td>0.6756</td><td>0.6416</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>0.2190</td><td>0.1572</td><td>1.3667</td><td>1.7672</td><td>0.6010</td><td>0.6186</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>0.4629</td><td>0.4086</td><td>1.0150</td><td>1.2581</td><td>0.7138</td><td>0.7103</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>0.5656</td><td>0.5457</td><td>0.9450</td><td>1.1729</td><td>0.7852</td><td>0.7639</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>0.2641</td><td>0.2345</td><td>1.0850</td><td>1.3378</td><td>0.6185</td><td>0.6367</td></tr></table>

Table S9: Associations with human scores and acceptance discrimination by year. Full-sample and withdrawalexcluded AUCs use their respective denominators; other metrics use all papers in the specified year.

<table><tr><td>Model</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>GPT-5.6-Sol</td><td>20</td><td>64</td><td>493</td><td>463</td><td>751</td><td>163</td><td>36</td><td>3</td><td>7</td><td>0</td></tr><tr><td>GPT-6-Luna</td><td>33</td><td>405</td><td>644</td><td>487</td><td>383</td><td>38</td><td>0</td><td>10</td><td>0</td><td>0</td></tr><tr><td>GPT-6-Sol</td><td>0</td><td>19</td><td>252</td><td>415</td><td>1042</td><td>251</td><td>11</td><td>10</td><td>0</td><td>0</td></tr></table>

<table><tr><td>Model</td><td>Share at 5–7</td><td>Share at 5–6</td></tr><tr><td>GPT-5.6-Sol</td><td>47.50%</td><td>45.70%</td></tr><tr><td>GPT-6-Luna</td><td>21.05%</td><td>21.05%</td></tr><tr><td>GPT-6-Sol</td><td>65.20%</td><td>64.65%</td></tr></table>

Table S10: Counts at scores 1–10 and score concentration. Each row contains all 2,000 scores for one model. The intervals 5–7 and 5–6 include both endpoints.

<table><tr><td>Model</td><td>Minimum variance</td><td>Q25</td><td>Median</td><td>Q75</td><td>Q90</td><td>Maximum</td></tr><tr><td>GPT-5.6-Sol</td><td>0.0000</td><td>0.0000</td><td>0.1389</td><td>0.2333</td><td>0.2667</td><td>0.4889</td></tr><tr><td>GPT-6-Luna</td><td>0.0000</td><td>0.0000</td><td>0.1778</td><td>0.2417</td><td>0.2778</td><td>0.6667</td></tr><tr><td>GPT-6-Sol</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0.1000</td><td>0.2333</td><td>0.2778</td></tr></table>

<table><tr><td>Model</td><td>Range 0</td><td>Range 1</td><td>Range 2</td><td>Range 3</td></tr><tr><td>GPT-5.6-Sol</td><td>60</td><td>130</td><td>10</td><td>0</td></tr><tr><td>GPT-6-Luna</td><td>65</td><td>118</td><td>17</td><td>0</td></tr><tr><td>GPT-6-Sol</td><td>113</td><td>87</td><td>0</td><td>0</td></tr></table>

Table S11: Quantiles of unbiased paper-level sample variance and counts by score range. Quantiles use NumPy’s default linear interpolation; ranges use all ten scores.

![](images/42f78721ca184bd6a56878c56ae303450ed70312f0ae840ab4f81ee4b690afa1.jpg)  
Figure C1: Empirical cumulative distributions of paper-level sample SD for the three models, including all 200 papers.

<table><tr><td>Model</td><td>First-five mean</td><td>Last-five mean</td><td>Last-first and 95% CI</td><td>Mean absolute difference: ten versus first five</td></tr><tr><td>GPT-5.6-Sol</td><td>4.2790</td><td>4.2720</td><td>-0.0070 [-0.0410, +0.0270]</td><td>0.0835</td></tr><tr><td>GPT-6-Luna</td><td>3.4590</td><td>3.4870</td><td>+0.0280 [-0.0090, +0.0650]</td><td>0.0890</td></tr><tr><td>GPT-6-Sol</td><td>4.6530</td><td>4.6740</td><td>+0.0210 [-0.0020, +0.0440]</td><td>0.0465</td></tr></table>

Table S12: Mean-score comparison between the two batches of five reviews. Intervals use 10,000 paper-level bootstrap resamples. Mean absolute differences compare the ten-review mean with the first-five mean.

The resulting numbers of partitions are 45, 630, 2,100, 1,575, and 126. For each partition, block means are compared between papers from the same year, then averaged over partitions and paper pairs. The two years yield $2 { \binom { 1 0 0 } { 2 } } = 9$ , 900 distinct paper pairs.

Same denotes the same strict ordering in both blocks; Tie denotes a tie in at least one block; and Reverse denotes opposite strict orderings. These categories are mutually exclusive and exhaustive.

## C.6 Close-Pair Analysis with Independent Grouping Scores

To examine close paper pairs, we group pairs by the absolute difference between their first-five means, reserving the last five scores for ranking comparisons. Among the last five, single-review blocks enumerate all 10 partitions, and two-review blocks enumerate all 15 partitions. Grouping and ranking comparisons thus use separate scores.

The first-five mean differences are grouped as 0, (0, 0.4], (0.4, 1], (1, 2], and greater than 2 points. These differences are finite-sample estimates. Same, Tie, and Reverse below measure reproducibility between two subsequent independent review blocks, without treating the grouping estimate as the correct ordering.

For pairs whose first-five mean difference lies in (0, 0.4], subsequent single-review comparisons produce ties in 74.84%–85.01% of cases; with tworeview means, this remains 55.73%–68.45%. For the (1, 2] group, strict same-order rates with tworeview means reach 97.95%–99.58%. Improvements averaged across all pairs therefore combine contributions from different score-gap regions.

Each paper participates in multiple pairs, creating dependence across pairs. We enumerate all partitions exactly and report descriptive proportions, without treating the 9,900 paper pairs as independent observations for binomial intervals.

## C.7 Semantic Consistency in the Existing Review Corpus

The existing text-analysis corpus contains five original-manuscript reviews each from Luna and Sol and three from Haiku, for a shared set of 199 papers. It excludes the later rounds 6–10. We use every available round in this fixed corpus: five reviews yield 10 within-model pairs, three yield 3, and cross-model comparisons yield 25 or 15 pairs.

Each strength and concern is encoded with sentence-transformers/all-mpnet-base-v2. For two lists of the same polarity, we compute the pairwise cosine-similarity matrix and use the Hungarian algorithm for one-to-one matching. The sum of matched cosines is divided by the longer list’s length, with unmatched items contributing zero:

$$
S ( A , B ) = \frac { \operatorname* { m a x } _ { \mathcal { M } } \sum _ { ( a , b ) \in \mathcal { M } } \cos ( e _ { a } , e _ { b } ) } { \operatorname* { m a x } ( | A | , | B | ) } .
$$

All strength and concern lists in the corpus are nonempty. Review-pair values are first averaged within each paper and condition, then across papers. The table covers 15,522 review pairs. Confidence intervals retain the original analysis’s 5,000 paperlevel bootstrap resamples.

Repeated reviews from the same model have closer opinion lists on average, and concern lists are closer than strength lists. An existing assistant inspection of 40 matched opinion pairs found shared cores, partial overlaps, and complementary observations; it does not constitute a human blindevaluation gold standard. General-purpose NLI also misclassified some compound technical statements. We therefore retain the reproducible continuous semantic measure rather than converting automatic entailment thresholds into rates of substantive agreement.

![](images/9de0b51a7709e6950072ab17e029fc1528a99819560ee4462cc3a738407b2485.jpg)

<table><tr><td>Model</td><td>Reviews per block</td><td>Partitions</td><td>Same</td><td>Tie</td><td>Reverse</td></tr><tr><td>GPT-5.6-Sol</td><td>1</td><td>45</td><td>62.14%</td><td>36.71%</td><td>1.15%</td></tr><tr><td>GPT-5.6-Sol</td><td>2</td><td>630</td><td>73.52%</td><td>24.74%</td><td>1.74%</td></tr><tr><td>GPT-5.6-Sol</td><td>3</td><td>2100</td><td>78.92%</td><td>19.08%</td><td>2.00%</td></tr><tr><td>GPT-5.6-Sol</td><td>4</td><td>1575</td><td>82.18%</td><td>15.71%</td><td>2.10%</td></tr><tr><td>GPT-5.6-Sol</td><td>5</td><td>126</td><td>84.47%</td><td>13.38%</td><td>2.15%</td></tr><tr><td>GPT-6-Luna</td><td>1</td><td>45</td><td>63.00%</td><td>35.60%</td><td>1.40%</td></tr><tr><td>GPT-6-Luna</td><td>2</td><td>630</td><td>74.91%</td><td>23.05%</td><td>2.04%</td></tr><tr><td>GPT-6-Luna</td><td>3</td><td>2100</td><td>80.40%</td><td>17.30%</td><td>2.30%</td></tr><tr><td>GPT-6-Luna</td><td>4</td><td>1575</td><td>83.58%</td><td>14.03%</td><td>2.39%</td></tr><tr><td>GPT-6-Luna</td><td>5</td><td>126</td><td>85.71%</td><td>11.85%</td><td>2.44%</td></tr><tr><td>GPT-6-Sol</td><td>1</td><td>45</td><td>57.52%</td><td>42.14%</td><td>0.34%</td></tr><tr><td>GPT-6-Sol</td><td>2</td><td>630</td><td>66.43%</td><td>33.03%</td><td>0.54%</td></tr><tr><td>GPT-6-Sol</td><td>3</td><td>2100</td><td>70.24%</td><td>29.11%</td><td>0.65%</td></tr><tr><td>GPT-6-Sol</td><td>4</td><td>1575</td><td>72.56%</td><td>26.71%</td><td>0.73%</td></tr><tr><td>GPT-6-Sol</td><td>5</td><td>126</td><td>74.13%</td><td>25.08%</td><td>0.79%</td></tr><tr><td>Model</td><td>First-ive mean difference</td><td></td><td>Paper pairs 1-review Same</td><td>1-review Tie</td><td>1-review Reverse</td></tr><tr><td>GPT-5.6-Sol</td><td>0</td><td>824</td><td>6.57%</td><td>91.40%</td><td>2.04%</td></tr><tr><td>GPT-5.6-Sol</td><td>(0, 0.4]</td><td>1920</td><td>17.33%</td><td>79.49%</td><td>3.17%</td></tr><tr><td>GPT-5.6-Sol</td><td>(0.4, 1]</td><td>2364</td><td>52.94%</td><td>45.63%</td><td>1.43%</td></tr><tr><td>GPT-5.6-Sol</td><td>(1, 2]</td><td>3251</td><td>91.51%</td><td>8.36%</td><td>0.13%</td></tr><tr><td>GPT-5.6-Sol</td><td>&gt;2</td><td>1541</td><td>99.97%</td><td>0.03%</td><td>0.00%</td></tr><tr><td>GPT-6-Luna</td><td>0</td><td>775</td><td>7.65%</td><td>89.63%</td><td>2.72%</td></tr><tr><td>GPT-6-Luna</td><td>(0, 0.4]</td><td>1832</td><td>22.25%</td><td>74.84%</td><td>2.91%</td></tr><tr><td>GPT-6-Luna</td><td>(0.4, 1]</td><td>2828</td><td>57.86%</td><td>40.80%</td><td>1.34%</td></tr><tr><td>GPT-6-Luna</td><td>(1, 2]</td><td>2926</td><td>91.17%</td><td>8.74%</td><td>0.10%</td></tr><tr><td>GPT-6-Luna</td><td>&gt;2</td><td>1539</td><td>99.80%</td><td>0.20%</td><td>0.00%</td></tr><tr><td>GPT-6-Sol</td><td>0</td><td>2062</td><td>1.13%</td><td>98.67%</td><td>0.20%</td></tr><tr><td>GPT-6-Sol</td><td>(0, 0.4]</td><td>1425</td><td>13.92%</td><td>85.01%</td><td>1.07%</td></tr><tr><td>GPT-6-Sol</td><td>(0.4, 1]</td><td>3047</td><td>74.25%</td><td>25.46%</td><td>0.29%</td></tr><tr><td>GPT-6-Sol</td><td>(1, 2]</td><td>2688</td><td>97.34%</td><td>2.66%</td><td>0.00%</td></tr><tr><td>GPT-6-Sol</td><td>&gt;2</td><td>678</td><td>100.00%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Model</td><td>First-five mean difference</td><td>Paper pairs 2-review Same</td><td></td><td>2-review Tie</td><td>2-review Reverse</td></tr><tr><td>GPT-5.6-Sol</td><td>0</td><td>824</td><td>14.07%</td><td>82.16%</td><td>3.77%</td></tr><tr><td>GPT-5.6-Sol</td><td>(0, 0.4]</td><td>1920</td><td>33.44%</td><td>61.59%</td><td>4.98%</td></tr><tr><td>GPT-5.6-Sol</td><td>(0.4, 1]</td><td>2364</td><td>74.64%</td><td>23.58%</td><td>1.78%</td></tr><tr><td>GPT-5.6-Sol</td><td>(1, 2]</td><td>3251</td><td>98.15%</td><td>1.81%</td><td>0.04%</td></tr><tr><td>GPT-5.6-Sol</td><td>&gt;2</td><td>1541</td><td>100.00%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>GPT-6-Luna</td><td>0</td><td>775</td><td>16.70%</td><td>78.43%</td><td>4.88%</td></tr><tr><td>GPT-6-Luna</td><td>(0, 0.4]</td><td>1832</td><td>39.53%</td><td>55.73%</td><td>4.73%</td></tr><tr><td>GPT-6-Luna</td><td>(0.4, 1]</td><td>2828</td><td>77.43%</td><td>20.80%</td><td>1.77%</td></tr><tr><td>GPT-6-Luna</td><td>(1, 2]</td><td>2926</td><td>97.95%</td><td>1.98%</td><td>0.08%</td></tr><tr><td>GPT-6-Luna</td><td>&gt;2</td><td>1539</td><td>99.99%</td><td>0.01%</td><td>0.00%</td></tr><tr><td>GPT-6-Sol</td><td>0</td><td>2062</td><td>2.86%</td><td>96.81%</td><td>0.33%</td></tr><tr><td>GPT-6-Sol</td><td>(0, 0.4]</td><td>1425</td><td>29.74%</td><td>68.45%</td><td>1.82%</td></tr><tr><td>GPT-6-Sol</td><td>(0.4, 1]</td><td>3047</td><td>90.13%</td><td>9.56%</td><td>0.31%</td></tr><tr><td>GPT-6-Sol</td><td>(1, 2]</td><td>2688</td><td>99.58%</td><td>0.42%</td><td>0.00%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-6-Sol</td><td>&gt;2</td><td>678</td><td>100.00%</td><td>0.00%</td><td>0.00%</td></tr><tr><td>Comparison</td><td>Rounds</td><td>Review pairs per paper</td><td>Papers</td><td>Strengths and 95% CI</td><td>Concerns and 95% CI</td></tr><tr><td>Luna repeated reviews</td><td>5</td><td>10</td><td>199</td><td>0.544 [0.534, 0.553]</td><td>0.618 [0.611, 0.625]</td></tr><tr><td>Sol repeated reviews</td><td>5</td><td>10</td><td>199</td><td>0.545 [0.532, 0.558]</td><td>0.639 [0.630, 0.647]</td></tr><tr><td>Haiku repeated reviews</td><td>3</td><td>3</td><td>199</td><td>0.569 [0.558, 0.580]</td><td>0.607 [0.599, 0.615]</td></tr><tr><td>Luna-Sol</td><td> $5 \times 5$ </td><td>25</td><td>199</td><td>0.459 [0.451, 0.467]</td><td>0.545 [0.539, 0.552]</td></tr><tr><td>Luna-Haiku</td><td> $5 \times 3$ </td><td>15</td><td>199</td><td>0.464 [0.457, 0.472]</td><td>0.532 [0.526, 0.538]</td></tr><tr><td>Sol-Haiku</td><td> $5 \times 3$ </td><td>15</td><td>199</td><td>0.419 [0.410, 0.428]</td><td>0.502 [0.495, 0.508]</td></tr></table>

Table S13: Complete ranking reproducibility results for 1–5 reviews per block. Percentages are computed before rounding; displayed values may differ slightly from a sum of 100%.  
Figure C2: Strict same-order rates from the last five reviews. The first five define the horizontal-axis groups. Solid lines use two reviews per block, and dashed lines use one. Groups are defined separately for each model; full denominators appear in the table above.

Table S14: Ranking reproducibility after grouping by independent score differences. Rounds 1–5 define groups; rounds 6–10 provide comparisons. Each model includes all 9,900 within-year paper pairs.

Table S15: Semantic proximity in the fixed five-/three-review original-manuscript corpus. Values are mean cosines after list matching, not percentages of viewpoint agreement.

## D Review Budgets and Mean-Score Precision

## D.1 Average Success Probability Across Papers

We give equal weight to a fixed set of $N = 2 0 0$ papers. For paper i, let $\mu _ { i }$ and $\sigma _ { i } ^ { 2 }$ denote the mean and variance of scores under a fixed configuration. Let $\bar { Y } _ { i , k }$ be the mean of k future independent scores, and define

$$
\begin{array} { c } { \displaystyle { A ( k , \epsilon ) = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { P r } \big ( \vert \bar { Y } _ { i , k } - \mu _ { i } \vert < \epsilon \big ) , } } \\ { \displaystyle { V = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sigma _ { i } ^ { 2 } . } } \end{array}
$$

The target is $A ( k , \epsilon ) \geq 1 - \beta$ . This is the overall probability that the mean of k scores estimates the model’s expected score to within ϵ for a uniformly selected sample paper. The main paper sets $\beta =$ 0.05 and considers $\epsilon = 0 . 2 5$ and 0.5.

Under independent, identically distributed scoring within each paper, Var $( \bar { Y } _ { i , k } ) ~ = ~ \sigma _ { i } ^ { 2 } / k$ . Applying Chebyshev’s inequality to the failure event $| \bar { Y } _ { i , k } - \mu _ { i } | \geq \epsilon$ gives

$$
\begin{array} { c } { \displaystyle { A ( k , \epsilon ) \geq 1 - \frac { V } { k \epsilon ^ { 2 } } , } } \\ { \displaystyle { k \geq \left\lceil \frac { V } { \beta \epsilon ^ { 2 } } \right\rceil . } } \end{array}
$$

This expression gives a sufficient budget when the score scale is known and the mean and variance exist. It requires neither a between-paper score gap nor a normality assumption or an empirical histogram as ground truth.

## D.2 Planning Budgets from Observed Variance

With n pilot scores per paper, estimate within-paper variance by $\hat { V } = \bar { N ^ { - 1 } } \bar { \sum _ { i } } S _ { i } ^ { 2 }$ . Substituting $\hat { V }$ into the preceding expression gives the Planning budget. With ten pilot scores, the three configurations have $\hat { V }$ values of 0.140056, 0.148889, and 0.074056. Planning budgets are 45, 48, and 24 at a tolerance of 0.25 points, and 12, 12, and 6 at 0.5 points.

Planning expresses the budget implied by the observed variance. The next section also accounts for uncertainty in variance estimation from a finite pilot sample.

## D.3 An Upper Bound on Mean Variance from Nonoverlapping Pairs

Scores lie in $[ a , b ] = [ 1 , 1 0 ]$ . Define the symmetric kernel

$$
\begin{array} { c } { { h ( y , y ^ { \prime } ) = \displaystyle \frac { ( y - y ^ { \prime } ) ^ { 2 } } { 2 } , } } \\ { { 0 \leq h \leq B = \displaystyle \frac { ( b - a ) ^ { 2 } } { 2 } = 4 0 . 5 . } } \end{array}
$$

For two independent, identically distributed scores, $\mathbb { E } h ( Y , Y ^ { \prime } ) = \sigma _ { i } ^ { 2 }$ . The unbiased sample variance for each paper also equals the average kernel value over all score pairs:

$$
S _ { i } ^ { 2 } = { \binom { n } { 2 } } ^ { - 1 } \sum _ { r < s } h ( Y _ { i r } , Y _ { i s } ) .
$$

Let $K = \lfloor n / 2 \rfloor$ . Randomly permute each paper’s n scores and form K nonoverlapping pairs from the first 2K scores. Let $T _ { i }$ be their mean kernel value and $T = N ^ { - 1 } \textstyle \sum _ { i } T _ { i }$ its average across papers. Conditional on the observed scores, every unordered pair has the same probability of selection, so

$$
\begin{array} { c } { { \mathbb { E } _ { \pi } [ T _ { i } \mid Y _ { i 1 } , \ldots , Y _ { i n } ] = S _ { i } ^ { 2 } , } } \\ { { \mathbb { E } _ { \pi } [ T \mid Y ] = \hat { V } . } } \end{array}
$$

For fixed permutations, there are $M = N K$ nonoverlapping kernel values. Assuming calls are independent across papers, these values are independent, and their expectations average to V. For any real λ, Jensen’s inequality gives

$$
\begin{array} { l } { \displaystyle \mathbb { E } \exp \left( \lambda M \frac { \hat { V } } { B } \right) } \\ { \leq \mathbb { E } \exp \left( \lambda M \frac { T } { B } \right) . } \end{array}
$$

For $X \in [ 0 , 1 ]$ , the chord bound for the exponential function gives

$$
e ^ { \lambda X } \leq 1 + X ( e ^ { \lambda } - 1 ) .
$$

The kernel values may have different expectations. Multiplying the moment-generating functions of the independent variables and applying concavity of the logarithm yields a bound depending only on the overall mean $V / B ;$

$$
\begin{array} { l } { \displaystyle \mathbb { E } \exp \left( \lambda M \frac { \hat { V } } { B } \right) } \\ { \displaystyle \leq \left[ 1 + \frac { V } { B } ( e ^ { \lambda } - 1 ) \right] ^ { M } . } \end{array}
$$

For $v < V$ , taking $\lambda < 0$ and optimizing the Chernoff parameter gives

$$
\operatorname* { P r } ( \hat { V } \leq v ) \leq \exp \left[ - M \operatorname { k l } \left( \frac { v } { B } , \frac { V } { B } \right) \right] ,
$$

where

$$
\operatorname { k l } ( p , q ) = p \log { \frac { p } { q } } + ( 1 - p ) \log { \frac { 1 - p } { 1 - q } } .
$$

Given the observed $\hat { V }$ , let $U _ { \alpha }$ be the root above $\hat { V }$ of

$$
M \operatorname { k l } \left( { \frac { \hat { V } } { B } } , { \frac { U _ { \alpha } } { B } } \right) = \log { \frac { 1 } { \alpha } } .
$$

Inverting the lower-tail bound yields

$$
\operatorname* { P r } ( V > U _ { \alpha } ) \leq \alpha .
$$

The score range also implies $V \leq ( b - a ) ^ { 2 } / 4 =$ 20.25, so the final upper bound can be capped at 20.25. The derivation uses a Chernoff argument for bounded variables and random grouping of a U-statistic; see Hoeffding (1963).

With ten pilot reviews, all 45 pairs per paper, or 9,000 pairs overall, contribute to the unbiased estimate $\hat { V }$ , while the concentration bound uses $M = 2 0 0 \times 5 = 1 , 0 0 0$ . These quantities serve variance estimation and confidence control, respectively; the overlapping 9,000 pairs are not treated as independent kernel values.

## D.4 The Two 95% Levels

Choose $k ( U _ { \alpha } ) = \lceil U _ { \alpha } / ( \beta \epsilon ^ { 2 } ) \rceil$ from the pilot data. For separately collected, independent future scores,

$$
\operatorname* { P r } _ { \mathrm { p i l o t } } \{ A ( k ( U _ { \alpha } ) , \epsilon ) \ge 1 - \beta \} \ge 1 - \alpha .
$$

The main paper uses $\alpha = \beta = 0 . 0 5$ . The first 95% is the confidence level for the variance upper bound constructed from the pilot data; the second is the overall success probability of future mean-score estimation. The budget k counts new independent reviews and excludes the existing pilot reviews.

Averaging over both pilot-data and future-score randomness gives a joint success lower bound of $( 1 - \alpha ) ( 1 - \beta )$ . This equals 0.9025 when $\alpha = \beta =$ 0.05. To require a joint lower bound of at least 0.95 in advance, one can choose $\alpha = 0 . 0 1 , \beta = 0 . 0 4$ giving $0 . 9 9 \times 0 . 9 6 = 0 . 9 5 0 4$ . Section D.5 reports the corresponding budgets.

## D.5 Simultaneous Control Across Models and Different Tolerances

Assigning failure probability $\alpha / 3$ to each model’s variance bound gives simultaneous coverage of at least $1 - \alpha$ for all three bounds by Bonferroni’s inequality. Each model’s overall future-score success probability remains $1 - \beta$

Halving the tolerance approximately quadruples the variance-based budget, and a higher success target also increases the budget. All tables round budgets upward to integer review counts.

## D.6 Five Versus Ten Pilot Reviews

Ten pilot reviews increase the number of independent kernel values available to the bound from 400 to 1,000. Observed variances change little, but the upper bounds tighten: Certified budgets at a halfpoint tolerance decrease from 53, 55, and 43 to 33, 35, and 24.

<table><tr><td>Model</td><td>Confidence/success convention</td><td>Variance upper bound</td><td> $\epsilon { = } 1$ </td><td> $\epsilon { = } . 2 5$ </td><td> $\epsilon = . 5$ </td><td> $\epsilon { = } 1$ </td></tr><tr><td>GPT-5.6-Sol</td><td>Per-model 95% confidence / 95% success</td><td>0.411370</td><td>823</td><td>132</td><td>33</td><td>9</td></tr><tr><td>GPT-5.6-Sol</td><td>Simultaneous 95% confidence / 95% success per model</td><td>0.475736</td><td>952</td><td>153</td><td>39</td><td>10</td></tr><tr><td>GPT-5.6-Sol</td><td>Joint pilot/future success ≥95%</td><td>0.504356</td><td>1261</td><td>202</td><td>51</td><td>13</td></tr><tr><td>GPT-6-Luna</td><td>Per-model 95% confidence / 95% success</td><td>0.425662</td><td>852</td><td>137</td><td>35</td><td>9</td></tr><tr><td>GPT-6-Luna</td><td>Simultaneous 95% confidence / 95% success per model</td><td>0.490875</td><td>982</td><td>158</td><td>40</td><td>10</td></tr><tr><td>GPT-6-Luna</td><td>Joint pilot/future success  $2 9 5 \%$ </td><td>0.519842</td><td>1300</td><td>208</td><td>52</td><td>13</td></tr><tr><td>GPT-6-Sol</td><td>Per-model 95% confidence / 95% success</td><td>0.297823</td><td>596</td><td>96</td><td>24</td><td>6</td></tr><tr><td>GPT-6-Sol</td><td>Simultaneous 95% confidence / 95% success per model</td><td>0.354954</td><td>710</td><td>114</td><td>29</td><td>8</td></tr><tr><td>GPT-6-Sol</td><td>Joint pilot/future success  $2 9 5 \%$ </td><td>0.380626</td><td>952</td><td>153</td><td>39</td><td>10</td></tr></table>

Table S16: Sufficient budgets with confidence control from ten pilot reviews. Per-model control uses $\alpha { = } . 0 5$ and $\beta { = } . 0 5 ;$ simultaneous control across three models uses $\alpha { = } . 0 5 / 3$ per model and $\beta { = } . 0 5 ;$ joint control uses $\alpha { = } . 0 1$ and $\beta { = } . 0 4$

<table><tr><td>Model</td><td>Success target</td><td>Tolerance</td><td>Planning</td><td>Certified (per-model 95% confidence)</td></tr><tr><td>GPT-5.6-Sol</td><td>90%</td><td>0.25</td><td>23</td><td>66</td></tr><tr><td>GPT-5.6-Sol</td><td>90%</td><td>0.5</td><td>6</td><td>17</td></tr><tr><td>GPT-5.6-Sol</td><td>95%</td><td>0.25</td><td>45</td><td>132</td></tr><tr><td>GPT-5.6-Sol</td><td>95%</td><td>0.5</td><td>12</td><td>33</td></tr><tr><td>GPT-5.6-Sol</td><td>99%</td><td>0.25</td><td>225</td><td>659</td></tr><tr><td>GPT-5.6-Sol</td><td>99%</td><td>0.5</td><td>57</td><td>165</td></tr><tr><td>GPT-6-Luna</td><td>90%</td><td>0.25</td><td>24</td><td>69</td></tr><tr><td>GPT-6-Luna</td><td>90%</td><td>0.5</td><td>6</td><td>18</td></tr><tr><td>GPT-6-Luna</td><td>95%</td><td>0.25</td><td>48</td><td>137</td></tr><tr><td>GPT-6-Luna</td><td>95%</td><td>0.5</td><td>12</td><td>35</td></tr><tr><td>GPT-6-Luna</td><td>99%</td><td>0.25</td><td>239</td><td>682</td></tr><tr><td>GPT-6-Luna</td><td>99%</td><td>0.5</td><td>60</td><td>171</td></tr><tr><td>GPT-6-Sol</td><td>90%</td><td>0.25</td><td>12</td><td>48</td></tr><tr><td>GPT-6-Sol</td><td>90%</td><td>0.5</td><td>3</td><td>12</td></tr><tr><td>GPT-6-Sol</td><td>95%</td><td>0.25</td><td>24</td><td>96</td></tr><tr><td>GPT-6-Sol</td><td>95%</td><td>0.5</td><td>6</td><td>24</td></tr><tr><td>GPT-6-Sol</td><td>99%</td><td>0.25</td><td>119</td><td>477</td></tr><tr><td>GPT-6-Sol</td><td>99%</td><td>0.5</td><td>30</td><td>120</td></tr></table>

Table S17: Planning and Certified budgets for different future success targets. Certified uses a per-model 95% variance upper bound throughout, while the success target varies separately.

<table><tr><td>Model</td><td>Pilot n</td><td>Variance</td><td>95% variance upper bound</td><td>Planning  $\epsilon = . 5$ </td><td>Certified  $\epsilon { = } . 5$ </td></tr><tr><td>GPT-5.6-Sol</td><td>5</td><td>0.139000</td><td>0.654330</td><td>12</td><td>53</td></tr><tr><td>GPT-5.6-Sol</td><td>10</td><td>0.140056</td><td>0.411370</td><td>12</td><td>33</td></tr><tr><td>GPT-6-Luna</td><td>5</td><td>0.152000</td><td>0.679444</td><td>13</td><td>55</td></tr><tr><td>GPT-6-Luna</td><td>10</td><td>0.148889</td><td>0.425662</td><td>12</td><td>35</td></tr><tr><td>GPT-6-Sol</td><td>5</td><td>0.082000</td><td>0.536812</td><td>7</td><td>43</td></tr><tr><td>GPT-6-Sol</td><td>10</td><td>0.074056</td><td>0.297823</td><td>6</td><td>24</td></tr></table>

Table S18: Five versus ten pilot reviews under the same configuration. The five-review pilot uses the first five reviews. The future success target is 95%, with a tolerance of 0.5 points.

## D.7 A Reference Bound Using Only the Score Range

Without pilot variance estimates, the paper-specific Hoeffding bound gives

$$
\mathrm { P r } ( | \bar { Y } _ { i , k } - \mu _ { i } | \geq \epsilon ) \leq 2 \exp \left( - \frac { 2 k \epsilon ^ { 2 } } { 8 1 } \right) .
$$

Thus $k \geq \lceil 8 1 \log ( 2 / \beta ) / ( 2 \epsilon ^ { 2 } ) \rceil$ suffices to control each paper’s failure probability and hence the average failure probability. With $\beta = . 0 5$ , tolerances of 0.25 and 0.5 points require 2,391 and 598 reviews, respectively. Our primary analysis uses pilot variances and their upper bounds to exploit the observed dispersion of scores.

## E Score Granularity and Distribution Estimation

## E.1 Integer Versus Forced-Decimal Scoring

We compare the two protocols within papers, with five scores per protocol: 1,000 observations per model and protocol. Integer scores are exactly the first five from the ten-review original-manuscript experiment. The table also reports year-specific results and both acceptance-label conventions.

The forced-decimal protocol raises Luna’s mean score by 0.693 points and Sol’s by 0.266. Their within-paper SDs decrease from 0.390 to 0.329 and from 0.286 to 0.249, respectively. Luna’s MAE decreases in both years. Sol’s MAE changes in opposite directions across years, with both intervals including zero.

Output granularity affects exact agreement and score range under the integer protocol. If scores switch only between adjacent integers s and $s + 1$ with probabilities $1 - \pi$ and π, the variance is $\pi ( 1 - \pi )$ and the probability of disagreement between two reviews is $2 \pi ( 1 - \pi )$ . This two-point model shows how switching between adjacent levels can produce frequent disagreement within a small range. The decimal experiment directly measures the variance and score-location changes caused by a protocol change, without assuming that lower variance implies a more accurate estimate of an unchanged scoring target.

## E.2 Construction of Quick and Standard

The target is the score distribution $p _ { i } ( s )$ under the fixed $\mathrm { G P T } { - } 5 . 6 { - } \mathrm { S o l }$ medium integer configuration, for $s ~ \in ~ \{ 1 , \ldots , 1 0 \}$ . An auxiliary call uses a nonreasoning configuration and extracts the visible probabilities of valid candidate score tokens. Restricting candidates to 1–10 and renormalizing yields $r _ { i } ( s )$ . The analysis does not assign probability mass to candidates absent from the response.

Quick first adjusts probability concentration:

$$
\widetilde { r } _ { i } ( s ; T ) = \frac { \operatorname* { m a x } ( r _ { i } ( s ) , 1 0 ^ { - 9 } ) ^ { 1 / T } } { \sum _ { u \in S _ { i } } \operatorname* { m a x } ( r _ { i } ( u ) , 1 0 ^ { - 9 } ) ^ { 1 / T } } ,
$$

Here $s _ { i }$ is the valid candidate set returned by the call. It then maps score locations to $\scriptstyle x _ { s } =$ $\mathrm { c l i p } ( a + b s , 1 , 1 0 )$ and linearly distributes the mass at each location between adjacent integers. Integer locations receive all their mass; otherwise, the weights at $\lfloor x _ { s } \rfloor$ and $\lceil x _ { s } \rceil$ are $\lceil x _ { s } \rceil - x _ { s }$ and $x _ { s } - \lfloor x _ { s } \rfloor$ , respectively. Denote the resulting distribution by $q _ { Q , i }$

Standard combines Quick with two reasoning scores from the target configuration:

$$
q _ { S , i } = w \hat { p } _ { i , 2 } + ( 1 - w ) q _ { Q , i } .
$$

<table><tr><td>Model</td><td>Year</td><td>Protocol</td><td></td><td>Mean</td><td>Within-paper SD</td><td>Identical 5/5</td></tr><tr><td>GPT-6-Luna</td><td></td><td>All Integer</td><td></td><td>3.4590</td><td>0.3899</td><td>45.5%</td></tr><tr><td>GPT-6-Luna</td><td></td><td>All Forced decimal</td><td></td><td>4.1524</td><td>0.3290</td><td>3.5%</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>Integer</td><td></td><td>3.8080</td><td>0.3975</td><td>41.0%</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>Forced decimal</td><td></td><td>4.4850</td><td>0.3184</td><td>5.0%</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>Integer</td><td></td><td>3.1100</td><td>0.3821</td><td>50.0%</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>Forced decimal</td><td></td><td>3.8198</td><td>0.3392</td><td>2.0%</td></tr><tr><td>GPT-6-Sol</td><td></td><td>All Integer</td><td></td><td>4.6530</td><td>0.2864</td><td>66.0%</td></tr><tr><td>GPT-6-Sol</td><td></td><td>All Forced decimal</td><td></td><td>4.9190</td><td>0.2485</td><td>17.0%</td></tr><tr><td>GPT-6-Sol GPT-6-Sol</td><td></td><td>2025 Integer</td><td></td><td>4.8380</td><td>0.2793</td><td>69.0%</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>Forced decimal</td><td></td><td>5.0720</td><td>0.2587</td><td>16.0%</td></tr><tr><td></td><td>2026</td><td>Integer</td><td></td><td>4.4680</td><td>0.2933</td><td>63.0%</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>Forced decimal</td><td></td><td>4.7660</td><td>0.2380</td><td>18.0%</td></tr><tr><td>Model</td><td>Year</td><td>Protocol</td><td>Spearman</td><td>MAE</td><td>RMSE</td><td>AUC: all</td><td>AUC: excl. withdrawn</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>Integer</td><td>0.3367</td><td>1.4687</td><td>1.8500</td><td>0.6427</td><td>0.6399</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>Forced decimal</td><td>0.3687</td><td>1.1620</td><td>1.4507</td><td>0.6675</td><td>0.6668</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>Integer</td><td>0.3974</td><td>1.5417</td><td>1.9033</td><td>0.6795</td><td>0.6504</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>Forced decimal</td><td>0.4008</td><td>1.2029</td><td>1.4874</td><td>0.6940</td><td>0.6612</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>Integer</td><td>0.1368</td><td>1.3957</td><td>1.7951</td><td>0.5850</td><td>0.6060</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>Forced decimal</td><td>0.2120</td><td>1.1212</td><td>1.4131</td><td>0.6190</td><td>0.6485</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>Integer</td><td>0.4096</td><td>1.0207</td><td>1.2612</td><td>0.7069</td><td>0.7028</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>Forced decimal</td><td>0.3998</td><td>1.0195</td><td>1.2572</td><td>0.7201</td><td>0.7250</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>Integer</td><td>0.5407</td><td>0.9503</td><td>1.1774</td><td>0.7743</td><td>0.7483</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>Forced decimal</td><td>0.5221</td><td>0.8911</td><td>1.1103</td><td>0.7713</td><td>0.7449</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>Integer</td><td>0.2290</td><td>1.0910</td><td>1.3397</td><td>0.6136</td><td>0.6348</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>Forced decimal</td><td>0.2492</td><td>1.1478</td><td>1.3887</td><td>0.6525</td><td>0.7013</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table S19: Complete metrics for integer and forced-decimal protocols. Means, human-score associations, and errors use all papers in the specified year; AUCs use the full and withdrawal-excluded samples separately.

<table><tr><td>Model</td><td>Year</td><td>Mean change and 95% CI</td><td>MAE change and 95% CI</td></tr><tr><td>GPT-6-Luna</td><td>All</td><td>+0.693 [+0.649, +0.739]</td><td>-0.307 [-0.389, -0.223]</td></tr><tr><td>GPT-6-Luna</td><td>2025</td><td>+0.677 [+0.613, +0.744]</td><td>-0.339 [-0.453, -0.221]</td></tr><tr><td>GPT-6-Luna</td><td>2026</td><td>+0.710 [+0.648, +0.771]</td><td>-0.274 [-0.400, -0.145]</td></tr><tr><td>GPT-6-Sol</td><td>All</td><td>+0.266 [+0.233, +0.298]</td><td>-0.001 [-0.050, +0.044]</td></tr><tr><td>GPT-6-Sol</td><td>2025</td><td>+0.234 [+0.187, +0.279]</td><td>-0.059 [-0.119, +0.001]</td></tr><tr><td>GPT-6-Sol</td><td>2026</td><td>+0.298 [+0.250, +0.345]</td><td>+0.057 [-0.011, +0.125]</td></tr></table>

Table S20: Changes in mean score and MAE, decimal minus integer. Overall rows retain the main paper’s original intervals; year-specific rows use paired paper-level bootstrap resampling.

A Quick prediction uses one auxiliary call. Standard uses that same auxiliary call and two reasoning reviews.

## E.3 Parameter Search and Train–Test Splits

The 198 papers are sorted by anonymous ID, and 30 random 99/99 train–test splits are generated without additional stratification by year. The search grid is

$$
\begin{array} { c } { { T \in \{ 1 , 2 , 3 , 5 , 8 , 1 2 \} , } } \\ { { a \in \{ - 2 . 0 , - 1 . 8 , . . . , 0 . 4 \} , } } \\ { { b \in \{ 0 . 9 , 1 . 0 , 1 . 1 \} , } } \\ { { w \in \{ 0 . 3 , 0 . 4 , . . . , 1 . 0 \} . } } \end{array}
$$

Quick has 234 parameter combinations. First, $( T , a , b )$ is selected using Brier loss against five target scores per training paper. Next, all ten partitions of five scores into two inputs and three targets are enumerated, and the mean training loss selects one of eight candidate weights. Test papers do not participate in parameter selection, and Standard’s input and target scores do not overlap.

The parameter combinations selected across the 30 splits are as follows.

At test time, Quick consistently uses the first valid auxiliary output and is evaluated against five independent target scores. Standard enumerates ten partitions per paper, first averaging over partitions, then over that paper’s test-set appearances, and finally equally across the 198 papers. Each paper appears in the test set 8–23 times.

## E.4 Squared Distribution Risk with Independent Targets

For a predicted distribution q and independent target scores $Z _ { 1 } , \ldots , Z _ { m }$ , define

$$
\begin{array} { l } { { \displaystyle { \widehat R } ( q ) = \sum _ { s } q ( s ) ^ { 2 } - \frac { 2 } { m } \sum _ { j = 1 } ^ { m } q ( Z _ { j } ) } } \\ { { \displaystyle ~ + \frac { 1 } { m ( m - 1 ) } \sum _ { j \ne \ell } \mathbf { 1 } [ Z _ { j } = Z _ { \ell } ] } . } \end{array}
$$

Conditional on $q ,$ the expectation of the second term is $\begin{array} { r } { - 2 \sum _ { s } q ( s ) p _ { i } ( s ) } \end{array}$ and that of the final term is $\textstyle \sum _ { s } p _ { i } ( s ) ^ { 2 }$ . Therefore,

$$
\mathbb { E } [ \widehat { R } ( q ) \mid q ] = \sum _ { s } ( q ( s ) - p _ { i } ( s ) ) ^ { 2 } .
$$

The final term estimates the squared probability sum through collisions between independent target scores, removing variation due to a finite target sample. Individual paper-level risk estimates may be negative; these values are retained when averaging.

For a frequency estimator based on n direct scores, expected risk is

$$
\mathbb { E } \| \hat { p } _ { i , n } - p _ { i } \| _ { 2 } ^ { 2 } = \frac { 1 - \sum _ { s } p _ { i } ( s ) ^ { 2 } } { n } .
$$

Direct-sampling baselines estimate the numerator from the collision statistic of the five existing scores and divide by n. Risks for 1, 2, 3, and 5 reviews are estimates of expected risk under this sampling law, without treating the five-score histogram as a known target distribution.

For an auxiliary prediction conditionally independent of two reasoning scores, and a frequency estimator unbiased for $p _ { i }$ , Standard’s risk decomposes as

$$
\begin{array} { l } { \displaystyle \mathbb { E } \| q _ { S , i } - p _ { i } \| _ { 2 } ^ { 2 } } \\ { \displaystyle = \frac { w ^ { 2 } } { 2 } \big ( 1 - \| p _ { i } \| _ { 2 } ^ { 2 } \big ) + ( 1 - w ) ^ { 2 } \mathbb { E } \| q _ { Q , i } - p _ { i } \| _ { 2 } ^ { 2 } . } \end{array}
$$

The cross term vanishes because the reasoningfrequency error has mean zero and is conditionally independent of the auxiliary prediction. Weights are selected on training papers, and the combined predictor’s risk is evaluated on test papers.

## E.5 Pooled and Year-Specific Risks

Overall Standard risk is 0.0829. Year-specific risks are 0.0932 in 2025 and 0.0727 in 2026, with 99 papers per year; two direct reasoning reviews yield 0.1520 and 0.0914. All intervals hold the original parameters and splits fixed while resampling papers, preserving the same within-paper aggregation order.

## E.6 Bootstrap with Parameter Refitting

For each paper-level refit bootstrap, we draw 198 papers with replacement from the 198-paper set, record each paper’s multiplicity, and use it as a weight within the original 30 train–test splits. The training side searches all 234 Quick parameter combinations and eight Standard weights again. The test side continues to exclude training papers for that split. Refit results retain the original withinpaper averaging over test appearances and are then aggregated using the bootstrap paper weights.

<table><tr><td>T</td><td>a b</td><td>W</td><td>Splits</td></tr><tr><td>12</td><td>0.2 0.9</td><td>0.7</td><td>7</td></tr><tr><td>5</td><td>0.2</td><td>0.9 0.7</td><td>5</td></tr><tr><td>8</td><td>-0.2</td><td>1.0 0.7</td><td>3</td></tr><tr><td>8</td><td>0.2</td><td>0.9 0.6</td><td>3</td></tr><tr><td>8</td><td>0.2</td><td>0.9 0.7</td><td>2</td></tr><tr><td>12</td><td>0.2</td><td>0.9 0.6</td><td>2</td></tr><tr><td>2</td><td>-0.4</td><td>1.0 0.6</td><td>1</td></tr><tr><td>3</td><td>-0.2</td><td>1.0 0.6</td><td>1</td></tr><tr><td>12</td><td>0.4</td><td>0.9 0.6</td><td>1</td></tr><tr><td>12</td><td>-0.2</td><td>1.0 0.7</td><td>1</td></tr><tr><td>5</td><td>-0.2</td><td>1.0 0.7</td><td>1</td></tr><tr><td>5</td><td>-0.2</td><td>1.0 0.6</td><td>1</td></tr><tr><td>3</td><td>0.2</td><td>0.9 0.6</td><td>1</td></tr><tr><td>5</td><td>0.2</td><td>0.9 0.6</td><td>1</td></tr></table>

Table S21: Selection frequencies of Quick parameters and Standard weights. Each row is a combination jointly selected in a split; frequencies sum to 30.

<table><tr><td>Year</td><td>Papers Method</td><td></td><td>Risk and 95% paper-bootstrap CI</td></tr><tr><td>All</td><td>198</td><td>Quick</td><td>0.2427 [0.2029, 0.2860]</td></tr><tr><td>All</td><td></td><td>198 Standard</td><td>0.0829 [0.0750, 0.0911]</td></tr><tr><td>All</td><td></td><td>198 1 Reasoning</td><td>0.2434 [0.2051, 0.2828]</td></tr><tr><td>All</td><td></td><td>198 2 Reasoning</td><td>0.1217 [0.1025, 0.1414]</td></tr><tr><td>All</td><td></td><td>198 3 Reasoning</td><td>0.0811 [0.0684, 0.0943]</td></tr><tr><td>All</td><td></td><td>198 5 Reasoning</td><td>0.0487 [0.0410, 0.0566]</td></tr><tr><td>2025</td><td></td><td>99 Quick</td><td>0.2141 [0.1634, 0.2709]</td></tr><tr><td>2025</td><td></td><td>99 Standard</td><td>0.0932 [0.0821, 0.1041]</td></tr><tr><td>2025</td><td></td><td>99 1 Reasoning</td><td>0.3040 [0.2495, 0.3586]</td></tr><tr><td>2025</td><td></td><td>99 2 Reasoning</td><td>0.1520 [0.1247, 0.1793]</td></tr><tr><td>2025</td><td></td><td>99 3 Reasoning</td><td>0.1013 [0.0832, 0.1195]</td></tr><tr><td>2025</td><td></td><td>99 5 Reasoning</td><td>0.0608 [0.0499, 0.0717]</td></tr><tr><td>2026</td><td></td><td>99 Quick</td><td>0.2713 [0.2129, 0.3350]</td></tr><tr><td>2026</td><td></td><td>99 Standard</td><td>0.0727 [0.0620, 0.0839]</td></tr><tr><td>2026</td><td></td><td>99 1 Reasoning</td><td>0.1828 [0.1333, 0.2354]</td></tr><tr><td>2026</td><td></td><td>99 2 Reasoning</td><td>0.0914 [0.0667, 0.1177]</td></tr><tr><td>2026</td><td></td><td>99 3 Reasoning</td><td>0.0609 [0.0444, 0.0785]</td></tr><tr><td>2026</td><td></td><td>99 5 Reasoning</td><td>0.0366 [0.0267, 0.0471]</td></tr></table>

Table S22: Risks from the same held-out analysis, aggregated by year. Parameters remain those fitted on the original mixed-year training sets; test results are grouped by paper year without year-specific refitting.

This procedure preserves the original split design while propagating uncertainty from paper resampling and parameter selection. We run 1,000 resamples using frozen scores and probabilities, without new reviews. The original, unresampled fit and paper-level risks match the saved main-paper results to a tolerance of $1 0 ^ { - 1 2 }$

With refitting, the risk difference between Standard and two reasoning reviews is −0.0388, with a 95% interval of [−0.0525, −0.0259]. Relative to three reasoning reviews, it is +0.0018, with an interval of [−0.0062, +0.0091]. The finding that Standard reduces risk relative to two reasoning reviews and approaches the risk of three persists after parameter reselection.

## E.7 Acceptance AUC Under the Same Held-Out Design

All methods use the same 198 papers, parameters, and test splits as the risk analysis. Predictions are distributional expected scores; parameters are not fitted to acceptance labels. Direct-reasoning baselines enumerate all equal-sized subsets of the five scores, while Standard enumerates the ten twoinput combinations. AUC is computed on test papers for each combination, then averaged over combinations and the 30 splits.

On all 198 papers, Quick exceeds one reasoning review in AUC by +0.0197, and Standard exceeds two reasoning reviews by +0.0131. Excluding withdrawals gives differences of +0.0141 and +0.0115. This table reports point estimates; intervals for squared risk concern a different statistic and are not AUC intervals.

## E.8 Independent-Target Validation on Ten Papers

The existing independent-target validation sampled five papers per year, excluding the long-input paper 2026-paper\_067 before sampling. Each paper received ten additional GPT-5.6-Sol medium target scores. Predictions use the original five scores. Quick and Standard are fitted on the other 188 papers, yielding $T = 8 , a = 0 . 2 , b = 0 . 9 , w = 0 . 7$

For Quick, every available auxiliary output for a paper forms a separate one-call prediction, and losses are averaged without first combining the outputs into a single prediction. Standard uses the first auxiliary output and all two-score combinations from the original five scores. Direct-reasoning methods enumerate all subsets of the corresponding size. All methods first average over prediction combinations within each paper, then equally over the ten papers.

Brier loss is computed against each of the ten one-hot targets and then averaged, including randomness in the target scores themselves. W1 compares the predicted distribution with the empirical CDF of the ten target scores, in score points. Modal accuracy compares the modes of the prediction and the ten target scores, choosing the smallest score in a tie. Mean-score error and $P ( Y \geq 6 )$ error compare the respective means and tail probabilities.

In this independent-target set, Standard has Brier loss 0.353, W1 of 0.194 points, and modal accuracy of 69%; three reasoning reviews yield 0.379, 0.178 points, and 77%. These metrics measure the full distribution, cumulative distribution, and most likely score, respectively, and are reported together.

## E.9 Historical Call Costs

The existing cost audit uses the original script’s historical pretax USD rates per million tokens: 4 for ordinary input, 5 for cache writes, 0.4 for cached input, and 20 for output. Call cost is computed from usage fields as

$$
\mathrm { c o s t } = \frac { 4 ( I - C - W ) + 0 . 4 C + 5 W + 2 0 O } { 1 0 ^ { 6 } } ,
$$

Here I denotes input tokens, C cached-input tokens, W cache-write tokens, and O output tokens.

First and subsequent calls differ in cache state, so the table reports both historical averages. The main paper’s R/Q columns describe call composition; this table gives observed costs for those calls without combining different cache states into a single deployment price. The frozen summaries do not contain consistent end-to-end latency measurements for comparing methods under matched conditions, so no latency estimate is reported.

## F Revision Crossover Experiment

## F.1 Seven-Cell Design and Paired Scoring

Three letters denote Writer, Selector, and Scorer, in that order, with L for Luna and H for Haiku. Luna candidates are processed by two Selectors and evaluated by two Scorers, yielding LLL, LHL, LLH, and LHH. Haiku candidates yield HLL, HHL, and HHH; HLH was not collected. All cells use the same 199 papers.

Section B.5 describes the Writers’ feedback sources. Each Writer’s candidate set is held fixed across Selectors, and both Scorers read the same final manuscript for a given Writer/Selector combination. Each Scorer independently evaluates the original and revised manuscripts three times each. A Scorer’s original-manuscript baseline is reused across revision conditions. Paper-level gain is defined as

<table><tr><td>Method</td><td>Risk</td><td>Fixed-parameter 95% CI (10,000 resamples)</td><td>Refit 95% CI (1,000 resamples)</td></tr><tr><td>Quick</td><td>0.2427</td><td>[0.2029, 0.2860]</td><td>[0.2020, 0.2934]</td></tr><tr><td>Standard</td><td>0.0829</td><td>[0.0750, 0.0911]</td><td>[0.0746, 0.0924]</td></tr><tr><td>1 Reasoning</td><td>0.2434</td><td>[0.2051, 0.2828]</td><td>[0.2040, 0.2823]</td></tr><tr><td>2 Reasoning</td><td>0.1217</td><td>[0.1025, 0.1414]</td><td>[0.1020, 0.1412]</td></tr><tr><td>3 Reasoning</td><td>0.0811</td><td>[0.0684, 0.0943]</td><td>[0.0680, 0.0941]</td></tr><tr><td>5 Reasoning</td><td>0.0487</td><td>[0.0410, 0.0566]</td><td>[0.0408, 0.0565]</td></tr></table>

Table S23: Fixed-parameter and refit intervals. Direct-sampling baselines have no fitted parameters; their two columns use different bootstrap resampling settings.

<table><tr><td>Method</td><td>R/Q</td><td>Mean held-out AUC: all 198</td><td>Excl. withdrawn: 150 papers</td><td>Independent-subset AUC (198 papers)</td></tr><tr><td>Quick</td><td>0/1</td><td>0.7092</td><td>0.7039</td><td>0.7092</td></tr><tr><td>Standard</td><td>2/1</td><td>0.7194</td><td>0.7187</td><td>0.7194</td></tr><tr><td>1 Reasoning</td><td>1/0</td><td>0.6895</td><td>0.6898</td><td>0.6899</td></tr><tr><td>2 Reasoning</td><td>2/0</td><td>0.7063</td><td>0.7072</td><td>0.7066</td></tr><tr><td>3 Reasoning</td><td>3/0</td><td>0.7124</td><td>0.7135</td><td>0.7127</td></tr><tr><td>5 Reasoning</td><td>5/0</td><td>0.7162</td><td>0.7176</td><td>0.7162</td></tr></table>

Table S24: Acceptance discrimination under a common held-out design. The last column additionally selects score subsets independently for each paper; the main columns use the same round-index subset across papers in each comparison.
<table><tr><td>Method</td><td>Brier</td><td>W1 (points)</td><td>Modal accuracy</td><td>Absolute target-mean error (points)</td><td>Absolute error in P(score≥6)</td></tr><tr><td>Quick</td><td>0.497</td><td>0.363</td><td>60.0%</td><td>0.315</td><td>0.141</td></tr><tr><td>Standard</td><td>0.353</td><td>0.194</td><td>69.0%</td><td>0.163</td><td>0.113</td></tr><tr><td>1 Reasoning</td><td>0.552</td><td>0.276</td><td>72.0%</td><td>0.276</td><td>0.132</td></tr><tr><td>2 Reasoning</td><td>0.422</td><td>0.194</td><td>75.0%</td><td>0.194</td><td>0.108</td></tr><tr><td>3 Reasoning</td><td>0.379</td><td>0.178</td><td>77.0%</td><td>0.178</td><td>0.104</td></tr><tr><td>5 Reasoning</td><td>0.344</td><td>0.140</td><td>80.0%</td><td>0.140</td><td>0.100</td></tr></table>

Table S25: Complete metrics for the existing ten-paper independent-target validation. The two-reasoning row is computed from existing scores; all five fixed baselines are retained.
<table><tr><td>Call type</td><td>Shared papers</td><td>Recorded calls</td><td>Missing usage</td><td>Mean cost, all calls (USD)</td><td>Mean first-call cost per paper</td></tr><tr><td>Auxiliary probability calls</td><td>198</td><td>578</td><td>0</td><td>0.067784</td><td>(USD) 0.153340</td></tr><tr><td>medium reasoning reviews</td><td>198</td><td>990</td><td>0</td><td>0.063410</td><td>0.171158</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table S26: Historical per-call costs computed from recorded usage in the existing shared sample.

$$
\hat { \Delta } _ { r , i } = \frac { 1 } { 3 } \sum _ { t = 1 } ^ { 3 } Y _ { r , i t } ^ { \prime } - \frac { 1 } { 3 } \sum _ { t = 1 } ^ { 3 } Y _ { r , i t } .
$$

Positive, zero, and negative gains are determined by the difference between the two sums of three scores, avoiding floating-point changes to direction. Overall gains and cross-Scorer differences use paired paper-level resampling.

Both years retain opposite average gains for Luna revisions under the two Scorers. For example, year-specific gains for LLL are +0.280 and +0.670, whereas those for LLH are −0.417 and −0.279. The year-specific results for LHL and LHH likewise remain positive and negative, respectively.

## F.2 Candidate Edits and Selector Agreement

The two Selectors independently accept or reject each candidate edit. For accepted sets $A _ { i } , B _ { i } ,$ , Jaccard similarity is $J _ { i } = | A _ { i } \cap B _ { i } | / | A _ { i } \cup B _ { i } |$ , defined as 1 when both sets are empty. The table reports candidate counts and acceptance intersections for each Writer.

Of Luna’s 9,327 candidates, 8,459 are accepted by both Selectors and 661 receive different decisions. Decisions are identical for 17 papers; the other 182 have different final texts. Candidate acceptance and final scoring occur at different stages: a large overlap in accepted candidates coexists with opposite gains from the two Scorers on fixed final manuscripts.

Haiku generates 2,587 candidates. Luna and Haiku accept 2,442 and 2,450, respectively, with 2,354 accepted by both. Mean paper-level Jaccard similarity is 0.9198. The Writers produce different numbers of edits, while selection agreement within their respective candidate sets is similar.

## F.3 Paper-Level Gain Directions and Cross-Model Differences

Beyond overall gains, we classify each paper’s difference between three-review means as positive, zero, or negative. The table gives complete direction contingency tables for the three sharedmanuscript conditions.

For Luna-written, Luna-selected manuscripts, 46 papers have positive Luna gains and negative Haiku gains, and none show the reverse combination. Both gains are positive for 27 papers, negative for 19, and zero for 10. The two strict reversal combinations total 61 papers for Luna-written, Haikuselected manuscripts and 23 for Haiku-written, Haiku-selected manuscripts.

At least one Scorer assigns zero gain in 53.8%, 45.7%, and 70.4% of papers across the three conditions. Direction disagreement can therefore be separated into strict reversals and cases involving zero gain; the full contingency tables preserve this distinction.

## F.4 Scoring Noise and Correlation Adjustment

Under a model with fixed manuscripts and independent, zero-mean scoring errors, the noise variance of the three-review original mean is $e _ { o , i } = S _ { o , i } ^ { 2 } / 3 .$ and the noise variance of the gain estimate is

$$
e _ { \Delta , i } = \frac { S _ { o , i } ^ { 2 } + S _ { v , i } ^ { 2 } } { 3 } .
$$

Let $v _ { x } , v _ { y }$ be the observed across-paper variances for two Scorers, $e _ { x } , e _ { y }$ their mean noise variances, and c their observed covariance. Keeping the covariance unchanged and subtracting diagonal noise terms gives

$$
\rho _ { \mathrm { a d j } } = { \frac { c } { \sqrt { ( v _ { x } - e _ { x } ) ( v _ { y } - e _ { y } ) } } } .
$$

The original-manuscript correlation is 0.804 before adjustment and 0.843 after it. Observed gain correlations for the two Luna revision conditions are 0.415 and 0.407, rising to 0.521 and 0.518 after adjustment. For Haiku revisions, the observed correlation is 0.085 and the adjusted value is approximately 0.230, with a wide interval. Some bootstrap samples have nonpositive noise-adjusted variances or correlations outside the usual range. The table retains valid-resample counts and out-ofrange cases.

The adjustment assumes independent errors across models and between original and revised manuscripts. Writers use existing first reviews to generate revisions; Section B.5 specifies their actual relationship to the original-score baselines. We use noise subtraction to diagnose the contribution of scoring error. Mean cross-Scorer differences for fixed final manuscripts are computed directly from paired scores and do not depend on this adjustment.

<table><tr><td>Writer/ Selector/ Scorer</td><td>Year</td><td>N</td><td>Original mean</td><td>Revised mean</td><td>Gain and 95% CI</td><td>Positive/zero/ negative</td></tr><tr><td>LLL</td><td>All</td><td>199</td><td>3.441</td><td>3.915</td><td>+0.474 [+0.367, +0.578]</td><td>120/54/25</td></tr><tr><td>LLL</td><td>2025</td><td>100</td><td>3.797</td><td>4.077</td><td>+0.280 [+0.137, +0.417]</td><td>50/35/15</td></tr><tr><td>LLL</td><td>2026</td><td>99</td><td>3.081</td><td>3.751</td><td>+0.670 [+0.512, +0.818]</td><td>70/19/10</td></tr><tr><td>LHL</td><td>All</td><td>199</td><td>3.441</td><td>3.913</td><td>+0.472 [+0.362, +0.580]</td><td>129/41/29</td></tr><tr><td>LHL</td><td>2025</td><td>100</td><td>3.797</td><td>4.103</td><td>+0.307 [+0.153, +0.453]</td><td>60/23/17</td></tr><tr><td>LHL</td><td>2026</td><td>99</td><td>3.081</td><td>3.721</td><td>+0.640 [+0.481, +0.791]</td><td>69/18/12</td></tr><tr><td>LLH</td><td>All</td><td>199</td><td>4.241</td><td>3.893</td><td>-0.348 [-0.432, -0.265]</td><td>32/63/104</td></tr><tr><td>LLH</td><td>2025</td><td>100</td><td>4.473</td><td>4.057</td><td>-0.417 [-0.540, -0.293]</td><td>15/28/57</td></tr><tr><td>LLH</td><td>2026</td><td>99</td><td>4.007</td><td>3.727</td><td>-0.279 [-0.387, -0.175]</td><td>17/35/47</td></tr><tr><td>LHH</td><td>All</td><td>199</td><td>4.241</td><td>3.945</td><td>-0.296 [-0.379, -0.214]</td><td>31/63/105</td></tr><tr><td>LHH</td><td>2025</td><td>100</td><td>4.473</td><td>4.117</td><td>-0.357 [-0.477, -0.233]</td><td>17/23/60</td></tr><tr><td>LHH</td><td>2026</td><td>99</td><td>4.007</td><td>3.771</td><td>-0.236 [-0.340, -0.135]</td><td>14/40/45</td></tr><tr><td>HLL</td><td>All</td><td>199</td><td>3.441</td><td>3.742</td><td>+0.302 [+0.238, +0.367]</td><td>108/70/21</td></tr><tr><td>HLL</td><td>2025</td><td>100</td><td>3.797</td><td>4.003</td><td>+0.207 [+0.120, +0.293]</td><td>47/38/15</td></tr><tr><td>HLL</td><td>2026</td><td>99</td><td>3.081</td><td>3.478</td><td>+0.397 [+0.303, +0.495]</td><td>61/32/6</td></tr><tr><td>HHL</td><td>All</td><td>199</td><td>3.441</td><td>3.722</td><td>+0.281 [+0.216, +0.347]</td><td>96/84/19</td></tr><tr><td>HHL</td><td>2025</td><td>100</td><td>3.797</td><td>3.993</td><td>+0.197 [+0.117, +0.280]</td><td>40/49/11</td></tr><tr><td>HHL</td><td>2026</td><td>99</td><td>3.081</td><td>3.448</td><td>+0.367 [+0.273, +0.468]</td><td>56/35/8</td></tr><tr><td>HHH</td><td>All</td><td>199</td><td>4.241</td><td>4.291</td><td>+0.050 [+0.000, +0.104]</td><td>57/102/40</td></tr><tr><td>HHH</td><td>2025</td><td>100</td><td>4.473</td><td>4.500</td><td>+0.027 [-0.050, +0.107]</td><td>27/50/23</td></tr><tr><td>HHH</td><td>2026</td><td>99</td><td>4.007</td><td>4.081</td><td>+0.074 [+0.010, +0.138]</td><td>30/52/17</td></tr></table>

Table S27: Complete means, gains, and year-specific results for the seven-cell revision experiment. Overall intervals retain the main paper’s original 10,000-resample bootstrap outputs; year-specific intervals use 10,000 paper-level resamples.
<table><tr><td>Writer</td><td>Candidates</td><td>Luna accepts</td><td>Haiku accepts</td><td>accept</td><td>Both Luna only 1</td><td>Haiku only</td><td>Mean paper-level Jaccard</td></tr><tr><td>Luna</td><td>9327</td><td>8857</td><td>8722</td><td>8459</td><td>398</td><td>263</td><td>0.9188</td></tr><tr><td>Haiku</td><td>2587</td><td>2442</td><td>2450</td><td>2354</td><td>88</td><td>96</td><td>0.9198</td></tr></table>

Table S28: Complete comparison of the two Selectors’ accepted sets with the Writer fixed. Jaccard similarity is computed per paper and then averaged over 199 papers.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>candidates</td><td>9327</td></tr><tr><td>accepted_luna</td><td>8857</td></tr><tr><td>accepted_haiku</td><td>8722</td></tr><tr><td>both</td><td>8459</td></tr><tr><td>luna_only</td><td>398</td></tr><tr><td>haiku_only</td><td>263</td></tr><tr><td>decision_disagreements</td><td>661</td></tr><tr><td>Mean paper-level Jaccard</td><td>0.9188</td></tr><tr><td>Paper-level Jaccard Q25/median/Q75 Candidates per paper: Q25/median/Q75</td><td>0.8817 / 0.9400 / 0.9694</td></tr><tr><td>Identical decision sets</td><td>33.0 / 41.0 / 53.5</td></tr><tr><td></td><td>17</td></tr><tr><td>Identical final text</td><td>17</td></tr></table>

Table S29: Paper-level selection statistics for Luna candidates. Candidate totals, acceptance intersections, and one-sided acceptances count diffs; quantiles and identical-decision counts are paper-level quantities.

<table><tr><td>Writer/Selector</td><td>Luna direction</td><td>Haiku positive</td><td>Haiku zero</td><td>Haiku negative</td></tr><tr><td>LL</td><td>Positive</td><td>27</td><td>47</td><td>46</td></tr><tr><td>LL</td><td>Zero</td><td>5</td><td>10</td><td>39</td></tr><tr><td>LL</td><td>Negative</td><td>0</td><td>6</td><td>19</td></tr><tr><td>LH</td><td>Positive</td><td>27</td><td>44</td><td>58</td></tr><tr><td>LH</td><td>Zero</td><td>1</td><td>13</td><td>27</td></tr><tr><td>LH</td><td>Negative</td><td>3</td><td>6</td><td>20</td></tr><tr><td>HH</td><td>Positive</td><td>31</td><td>47</td><td>18</td></tr><tr><td>HH</td><td>Zero</td><td>21</td><td>46</td><td>17</td></tr><tr><td>HH</td><td>Negative</td><td>5</td><td>9</td><td>5</td></tr></table>

Table S30: Paper-level gain directions under Luna and Haiku. The nine counts for each Writer/Selector condition sum to 199.

![](images/15ead0a947f0d137392db6770641e1c70de221fbc2db20bbd4d358bbf5d0e54a.jpg)

![](images/49522f5a23b22fb33cfcb4d87e7e33233afdca71fe440db10f72f56b9608116a.jpg)

![](images/66bc86ac5857ace5683570bb2f7fedc59fa18430a7182e80a156d5319fbbd4c8.jpg)

Figure F1: Each panel contains gains for 199 papers, with Luna on the horizontal axis and Haiku on the vertical axis. Coincident coordinates are combined into bubbles whose area is proportional to the paper count. Dashed lines mark equal gains under the two Scorers. Coordinates are not jittered.
<table><tr><td>Writer/Selector</td><td>Gain Pearson</td><td>Same direction (including both zero)</td><td>Strict reversal</td><td>At least one zero</td></tr><tr><td>LL</td><td>0.415</td><td>28.1%</td><td>23.1%</td><td>53.8%</td></tr><tr><td>LH</td><td>0.407</td><td>30.2%</td><td>30.7%</td><td>45.7%</td></tr><tr><td>HH</td><td>0.085</td><td>41.2%</td><td>11.6%</td><td>70.4%</td></tr><tr><td>Writer/Selector</td><td>Luna–Haiku and 95%CI</td><td></td><td>Luna mean gain-noise SD</td><td>Haiku mean gain-noise SD</td></tr><tr><td>LL</td><td>+0.822 [+0.717, +0.928]</td><td></td><td>0.237</td><td>0.210</td></tr><tr><td>LH</td><td>+0.769 [+0.662, +0.879]</td><td></td><td>0.243</td><td>0.232</td></tr><tr><td>HH</td><td>+0.231 [+0.154, +0.310]</td><td></td><td>0.245</td><td>0.237</td></tr><tr><td>Target</td><td>correlation and 95% CI</td><td>Observed Adjusted correlation and 95% CI</td><td>Valid adjusted resamples</td><td>Out-of-range resamples</td></tr><tr><td>Original</td><td>0.804 [0.741, 0.854]</td><td>0.843 [0.783, 0.890]</td><td>10000</td><td>0</td></tr><tr><td>Luna Writer/Luna Selector gain</td><td>0.415 [0.261, 0.551]</td><td>0.521 [0.336, 0.686]</td><td>10000</td><td>0</td></tr><tr><td>Luna Writer/Haiku Selector</td><td>0.407 [0.255, 0.541]</td><td>0.518 [0.337, 0.673]</td><td>10000</td><td>0</td></tr><tr><td>gain Haiku Writer/Haiku Selector gain</td><td>0.085 [-0.071, 0.237]</td><td>0.230 [-0.200, 0.902]</td><td>9767</td><td>189</td></tr></table>

Table S31: Paper-level gain associations, direction distributions, and mean cross-Scorer differences. Gain-noise SD is estimated from the sample variances of three original and three revised scores, taking the square root within each paper before averaging.

Table S32: Correlations before and after noise adjustment. Intervals retain the original analysis’s 10,000 paper-level bootstrap resamples. Out-of-range adjusted correlations are not clipped.

## F.5 Structural Checks of Manuscript Packages

We compare original and final packages paper by paper for the four Writer/Selector conditions, checking record counts and order, fields other than text, leading instructions and page-marker text, image indices and stored hashes, and the existence of referenced image files.

All four conditions pass these checks. The Lunawritten conditions accept 8,857 and 8,722 text blocks; the Haiku-written conditions accept 2,442 and 2,450. Recomputing SHA-256 for all 9,440 image files across the four final-package conditions gives exact matches to the original image-index hashes. This audit covers input-package structure, text-replacement rules, and image files. The study did not separately collect blinded expert labels for factual preservation or quality changes in the revisions.

## F.6 Real Revision Examples Selected by a Fixed Rule

Cases are selected from five direction strata in LL: Luna positive/Haiku negative, both positive, both negative, Luna negative/Haiku positive, and at least one zero. Papers in each stratum are sorted by a fixed hash rule, and the first is selected. The Lunanegative/Haiku-positive stratum is empty. One additional case is selected by the same rule from Lunapositive/Haiku-zero papers in the Haiku-written, Haiku-selected condition.

Each case shows the first accepted block of suitable display length after sorting edits by page number and text-block ID. Both original and revised text must contain 70–750 characters after whitespace normalization; if no block qualifies, the first accepted block is used. Scores always refer to the complete revised manuscript. The displayed block illustrates the edit and is not assigned responsibility for the whole-paper gain.

F.6.1 Case 1: Luna Positive / Haiku Negative Paper 2025-paper\_092; Luna Writer/Luna Selector; 46 papers in this direction stratum. The displayed edit is on PDF page 3, text block 27, diff d0003, and was accepted by this condition’s Selector.

Original:

3. We demonstrate theoretically how a) SGFT can learn highly performant policies, despite the bias introduced by the SGFT objective and b) how this enables SGFT to overcome compounding errors which plague standard MBRL approaches.

Revised:

3. We analyze how value-transfer bias and model error affect SGFT policies. The resulting guarantee is conditional on uniform closeness of Vsim to the optimal real-world value and improvability under real-world dynamics; neither condition is empirically verified in our experiments.

## F.6.2 Case 2: Both Positive

Paper 2026-paper\_005; Luna Writer/Luna Selector; 27 papers in this direction stratum. The displayed edit is on PDF page 1, text block 10, diff d0002, and was accepted by this condition’s Selector.

Original:

<table><tr><td>Writer/Selector</td><td>Papers</td><td>Edited text blocks</td><td>Image entries Nontext fields /</td><td>protected records / image-index checks</td></tr><tr><td>LL</td><td>199</td><td>8857</td><td></td><td>2360 All passed</td></tr><tr><td>LH</td><td>199</td><td>8722</td><td></td><td>2360 All passed</td></tr><tr><td>HH</td><td>199</td><td>2450</td><td></td><td>2360 All passed</td></tr><tr><td>HL</td><td>199</td><td>2442</td><td></td><td>2360 All passed</td></tr></table>

Table S33: Structural checks for the four final-text conditions. Image counts refer to index entries; each condition uses the same 199 original papers.
<table><tr><td>Scorer</td><td>Three original scores</td><td>Three revised scores</td><td>Whole-paper mean gain</td></tr><tr><td>Luna</td><td>3,4,4</td><td>5,5,4</td><td>+1.000</td></tr><tr><td>Haiku</td><td>5,6,5</td><td>5,5,5</td><td>-0.333</td></tr></table>

Table S34: Original and complete revised-manuscript scores for Case 1.

Despite rapid progress, two gaps remain. First, no research has scaled EFMs beyond the levels nec- essary to fully exploit this potential—previous efforts have been limited by computational resources and dataset availability. Second, no systematic, quantitative analysis of EFM scaling behavior has been conducted, leaving fundamental questions unanswered about optimal resource allocation and scaling strategies for neural data—in particular, how to allocate a fixed compute budget across model size and number of training epochs.

## Revised:

Despite rapid progress, two gaps remain. First, previous EFM efforts have explored a narrower range of model and data scales than those considered here, in part because of computational resources and dataset availability. Second, systematic quantitative analysis of EFM scaling behavior remains limited, leaving open questions about resource allocation and how to divide a fixed compute budget between model size and training duration.

## F.6.3 Case 3: Both Negative

Paper 2025-paper\_020; Luna Writer/Luna Selector; 19 papers in this direction stratum. The displayed edit is on PDF page 2, text block 15, diff

d0004, and was accepted by this condition’s Selector.

## Original:

We show that each Koopman operator is represented by the exponential of a matrix constructed with Toeplitz matrices and diagonal matrices. This allows us to apply Krylov subspace methods (Gallopou- los & Saad, 1992; Güttel, 2013; Hashimoto & Nodera, 2016) to compute the estimation of Koopman operators with low computational costs. By virtue of the universal property of Toeplitz matrices (Ye & Lim, 2016), we can show the universality of the proposed model with a linear algebraic approach. We also show a generalization bound of the proposed model using a reproducing kernel Hilbert space (RKHS) associated with the Fourier functions. We can analyze both the universality and generalization error with the same framework.

## Revised:

We represent each layer by the exponential of a matrix constructed from Toeplitz and diagonal matrices. This structure allows Krylov subspace methods to compute matrix-exponential actions. Using the universal property of Toeplitz matrices (Ye & Lim, 2016), we establish a finite-dimensional functionapproximation result. We also analyze generalization using an RKHS associated with Fourier functions, with the bound depending on operator norms and requiring appropriate controls on the model class.

<table><tr><td>Scorer</td><td>Three original scores</td><td>Three revised scores</td><td>Whole-paper mean gain</td></tr><tr><td>Luna</td><td>3,4,4</td><td>5,5,5</td><td>+1.333</td></tr><tr><td>Haiku</td><td>4,4,4</td><td>5,5,5</td><td>+1.000</td></tr></table>

Table S35: Original and complete revised-manuscript scores for Case 2.
<table><tr><td>Scorer</td><td>Three original scores</td><td>Three revised scores</td><td>Whole-paper mean gain</td></tr><tr><td>Luna</td><td>3,4,4</td><td>3,3,3</td><td>-0.667</td></tr><tr><td>Haiku</td><td>4,4,4</td><td>3,3,3</td><td>-1.000</td></tr></table>

Table S36: Original and complete revised-manuscript scores for Case 3.

## F.6.4 Case 4: At Least One Zero

Paper 2026-paper\_066; Luna Writer/Luna Selector; 107 papers in this direction stratum. The displayed edit is on PDF page 1, text block 9, diff d0003, and was accepted by this condition’s Selector.

## Original:

Large Language models (LLMs) such as Transformer (Vaswani et al., 2017), BERT (Devlin et al., 2019a), PaLM (Chowdhery et al., 2023), and GPT-4o (Hurst et al., 2024) have demonstrated remarkable capabilities in natural language understanding and generation, which helps to achieve a wide range of tasks such as language translation, sentiment analysis, and question answering. Similarly, video Transformers such as TimeSformer (Bertasius et al., 2021), ViViT (Arnab et al., 2021), and Video Swin Transformer (Liu et al., 2022) have illustrated remarkable progress in capturing spatio-temporal dynamics, greatly advancing applications such as video understanding and editing.

## Revised:

Video Transformers, including TimeSformer (Bertasius et al., 2021), ViViT (Arnab et al., 2021), and Video Swin Transformer (Liu et al., 2022), model spatio-temporal dependencies. This manuscript examines the computational cost of exact block attention and a proposed dynamic approach. The review identified fundamental correctness and dimensionality errors in that approach; the results below therefore define the target computation, not a validated efficient algorithm.

## F.6.5 Case 5: Haiku Revision, Luna Positive / Haiku Zero

Paper 2025-paper\_093; Haiku Writer/Haiku Selector; 47 papers in this direction stratum. The displayed edit is on PDF page 1, text block 20, diff d0002, and was accepted by this condition’s Selector.

## Original:

To address this gap, in this paper, we curate a comprehensive ex- trapolation benchmark consisting of seven datasets sourced from Matminer (Ward et al., 2018), accompanied by well-defined train/validation/test splits based on real-

## Revised:

To address this gap, in this paper, we curate a comprehensive extrapolation benchmark consisting of seven tasks derived from four datasets sourced from Matminer (Ward et al., 2018), accompanied by well-defined train/validation/test splits based on real-

<table><tr><td>Scorer</td><td>Three original scores</td><td>Three revised scores</td><td>Whole-paper mean gain</td></tr><tr><td>Luna</td><td>1,1,1</td><td>1,1, 1</td><td>+0.000</td></tr><tr><td>Haiku</td><td>2,2,3</td><td>2,1, 1</td><td>-1.000</td></tr></table>

Table S37: Original and complete revised-manuscript scores for Case 4.
<table><tr><td>Scorer</td><td>Three original scores</td><td>Three revised scores</td><td>Whole-paper mean gain</td></tr><tr><td>Luna</td><td>3,4,4</td><td>4,4,4</td><td>+0.333</td></tr><tr><td>Haiku</td><td>4,5,5</td><td>5,5,4</td><td>+0.000</td></tr></table>

Table S38: Original and complete revised-manuscript scores for Case 5.  
requires independent verification.

## G Reproduction Resources

The code package includes all data, code, configurations, and external files needed to reproduce the analyses and figures.

## H Ethical Considerations and Responsible Use

Data Sources and Licensing. This study uses publicly available ICLR 2025 and 2026 submissions and associated review metadata obtained through OpenReview. We selected 100 papers from each year for academic analysis of AI-based peer review. The original manuscripts and their contents remain subject to the copyrights and applicable licenses of their respective authors. Our reported findings primarily consist of independently computed statistics and analyses. Any redistribution of source materials or derived artifacts is subject to the applicable licenses and terms of use.

Use of AI Models. Large language models are the primary research objects in this study. They are used to evaluate manuscripts, generate and select candidate revisions, and assess scoring consistency. These model-generated reviews and revisions constitute experimental outputs rather than independent human assessments. The experimental procedures, statistical analyses, and interpretation of the results remain the responsibility of the authors.

Ethical Risks. The study analyzes publicly available research manuscripts and review information without recruiting human participants. Manuscript inputs are anonymized for AI evaluation. Our findings should not be interpreted as evidence that AI-generated reviews can replace human peer reviewers or reliably determine publication decisions. Automated manuscript revisions may also introduce inaccuracies, and their scientific validity