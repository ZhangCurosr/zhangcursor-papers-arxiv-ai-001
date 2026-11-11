# SafeRoute-MAS: Coverage-Aware and Risk-Calibrated Message Routing for Multi-Agent Scientific Reasoning

Zhiming Xue<sup>\*</sup>   
College of Engineering   
Northeastern University Boston, USA   
xue.zh@northeastern.edu   
Jianan Liu   
Independent Researcher   
Austin, USA   
jiananliu2408@gmail.com   
Aicheng Xie   
Independent Researcher   
San Jose, USA   
xieaichen789@gmail.com   
Weiran Yan   
Independent Researcher   
Santa Clara, USA   
yanwerran@gmail.com   
Jing Yang   
Olin Business School   
Washington University in St. Louis   
St. Louis, USA   
jing.y@wustl.edu

Abstract—We introduce SafeRoute-MAS, a coverage-aware, risk-calibrated procedure that keeps all specialist messages or removes one using only pre-integration information. On a sealed cache of 5,328 committees, its source-compatible operating point gains +1.80 percentage points (95% CI [+1.13, +2.52]) while removing 12.76% of message tokens. We also run 4,580 controlled integrations over a stratified 458-committee sample on a public gpt-oss-120b endpoint. The sample estimate is −0.09 points; design weighting to the original workload gives +0.94 points (95% CI [−1.31, +3.50]) and 18.72% acted-input reduction. A weighted token-matched comparison likewise does not establish a selector advantage (+1.68 points, CI [−1.27, +5.10]). Only 13 of 88 cached non-neutral labels (14.8%) retain direction by repeated-call majority. Expected-corruption UCL is 4.60% after weighting, but only 2 of 11 prespecified marginals pass the support/risk rule. We therefore propose a numerical qualification with Keep-All fallback, applied here retrospectively rather than validated prospectively.

Index Terms—multi-agent systems, scientific reasoning, message routing, selective prediction, risk calibration, context efficiency

## I. INTRODUCTION

Large language model (LLM) systems increasingly solve difficult tasks through committees of specialized agents. Debate and round-table consensus can expose an integrator to complementary evidence [1], [2]. More communication is not uniformly beneficial, however. A plausible but misleading message may turn a correct committee answer into an error, while redundant messages consume scarce context. Communication pruning can reduce cost [3], but a deployed router cannot use the gold correctness of a proposal and should not sacrifice helpful minority evidence merely because it disagrees with the majority.

We ask whether a router can preserve correctness while deciding which message to ignore. SciAgentTrace [4] provides cached full-committee and leave-one-out (LOO) outcomes from the DHD controlled replay protocol [5], allowing exact

evaluation of KEEP-ALL and PRUNE-ONE within the cache.   
We never add LOO effects to simulate multiple deletions.

We propose SafeRoute-MAS, shown in Fig. 1: a targetcalibrated three-class message-risk model coupled to a selective gate that acts only above a validation-frozen threshold. Our contributions are:

• a leakage-safe, coverage-aware keep/prune formulation over real counterfactual actions, with explicit calibration and abstention;

• a numerical target-backend qualification procedure combining input validation, repetition, matched generic pruning, accuracy/token/risk gates, and Keep-All fallback; and

• a transparent retrospective diagnostic that finds a positive source-cache operating point but no conclusive endpoint gain or selector advantage, thereby identifying when the system should abstain.

## II. RELATED WORK

Multi-agent debate and consensus use interacting viewpoints to improve factuality and reasoning [1], [2]. MasRouter chooses collaboration modes, roles, and models at the system level [6]; SafeRoute instead selects a realized message within a fixed committee.

Communication methods prune topology [3], train efficient dialogues [7], or prune runtime messages [8]; random pruning may already preserve accuracy [9]. SafeRoute instead learns one message-level action under empirical risk control, without gold correctness and with its corresponding cached replay. AgentPrune and MessageSieve require actions outside the released cache and online system re-execution. Calibration [10] and selective classification [11] motivate abstention; our UCL is not a future-distribution guarantee.

![](images/e5081d8d96680a01f9554f9c64256b0a55c2a95aedfe07a064e01a24ecda341c.jpg)  
Fig. 1. SafeRoute-MAS and its action-level evaluation. Visible message and committee features feed a calibrated selective gate, which chooses KEEP-ALL or one PRUNE-ONE action; the corresponding replay measures accuracy, token cost, and risk. Ablations do not establish committee context or the diversity rule as independent sources of gain.

## III. PROBLEM FORMULATION

For problem q, committee $C _ { q } = \{ m _ { 1 } , . . . , m _ { K } \}$ produces a cached full outcome $y _ { q } ^ { \mathrm { a l l } } \in \mathsf { \bar { \{ 0 , 1 \} } }$ . Removing message i yields cached outcome $y _ { q , - i }$ . Following the DHD trajectoryvalue intervention [5], the released signed effect is

$$
e _ { q i } = y _ { q } ^ { \mathrm { a l l } } - y _ { q , - i } \in \{ - 1 , 0 , + 1 \} ,\tag{1}
$$

where −1 is harmful (removal recovers the answer), 0 is neutral, and +1 is helpful (removal corrupts the answer). These outcomes are labels and evaluation fields only.

The router estimates $p _ { h } ( i ) , p _ { n } ( i )$ , and $p _ { + } ( i )$ from information visible before integration. Candidate removal utility is

$$
U _ { i } = p _ { h } ( i ) - \lambda p _ { + } ( i ) + \gamma s _ { i } ,\tag{2}
$$

where $s _ { i }$ is visible-character context share, $\lambda \ : = \ : 2 ,$ and the primary accuracy operating point fixes $\gamma = 0$ . Character share requires no tokenizer at decision time; all reported savings use official tokens. A context-weighted sensitivity uses $\gamma = 0 . 2 5$ For $i ^ { * } = \arg \operatorname* { m a x } _ { i } U _ { i }$ , the policy removes $i ^ { * }$ only if its score clears threshold τ and the frozen K/diversity abstention rules; otherwise it keeps all messages. The central risk is $\mathrm { P r } ( e _ { q i } =$ +1 | action), measured directly on selected actions.

## IV. SAFEROUTE-MAS DESIGN

The full implementation retains text, committee context, and a diversity rule as frozen engineering choices. Section VI-C shows that metadata alone is the smallest empirically sufficient representation; only calibration and observation masks have favorable point-estimate ablations, and neither is established as independently necessary.

## A. Coverage-Aware Message Encoding

The encoder concatenates role name/description, reasoning lens, proposed answer, reasoning direction, key idea, critical assumption, what-to-check, and self-reported confidence. TF– IDF word 1–2 grams represent text. Six structured fields have separate observation bits, so an absent field is not treated as semantically empty text. Numeric features include parse status, confidence parsed from visible language, character/word counts, and field completeness. No reference answer, proposal correctness, judge rationale, integrator output, or effect label enters the model.

## B. Committee Context

Per-message context features include normalized-answer agreement, unique/majority status, reasoning-lens frequency, relative confidence and length, context share, and committee completeness/entropy summaries. Normalization is deterministic over raw proposal text. Benchmark identity is included in the primary model, with a no-benchmark ablation and a LOBO evaluation.

## C. Calibrated Selective Gate

A class-balanced averaged logistic SGD model predicts the three effect classes. Message rows receive weight $1 / K$ so each committee has equal training weight. A problem-disjoint validation subset fits sigmoid calibration; the remaining validation problems select $( \tau , m )$ over a frozen threshold and diversitymargin grid. Validation requires at least 5% action coverage and at most 3% observed helpful pruning. The untouched test criterion is a 95% problem-cluster bootstrap upper confidence limit (UCL) no greater than 5%, with at least 100 actions over 50 acted-on problems.

The diversity rule always abstains at $K = 2$ . If the selected message is the only member of its normalized-answer cluster or reasoning-lens cluster, it must clear $\tau + m$ . As Sec. VI-C shows, this conservative rule did not improve test accuracy and is retained only because it was frozen in the primary policy before test access.

Implementation uses scikit-learn [12]: 25,000 TF–IDF 1–2 grams (min\_df=3, max\_df=.995), one-hot categories, and averaged class-balanced logistic $\mathrm { S G D ~ ( L 2 ~ \alpha ~ \alpha ~ = ~ 3 ~ \times ~ 1 0 ^ { - 5 } ; }$ 2,000 iterations). Validation is split 1:2 for sigmoid calibration and gate development. Searching 201 utility quantiles and $m \in \{ 0 , . 0 5 , . 1 0 , . 1 5 \}$ selects $\tau = - . 0 4 5 5 8 , m = . 0 5 ; \tau$ is a ranking cutoff, not a zero-utility threshold. Seed 20261031 is fixed. For the cached-primary analysis, clustered CIs use 5,000 accuracy and 10,000 risk/token resamples; independentendpoint resampling is specified in Table III.

## V. EXPERIMENTAL SETUP

## A. Data and Leakage Controls

We freeze access-controlled SciAgentTrace v2.1.0 [4] at commit c36465d and use its gpt-oss-120b LOO slice [13]: 93,282 actions, 6,663 problems, and 26,652 committee states across JEEBench [14], LAB-Bench [15], MaScQA [16], Omni-MATH-2 [17], [18], and SciBench [19]. A role-only shell has six empty fields but is rendered as a no-score notice plus role header; 1,584 are retained. The headline matrix in [5] excludes them; the remaining 42-row difference reflects the v2.1.0 revision. Effects comprise 5,431 harmful, 84,306 neutral, and 3,545 helpful rows for $K \in \{ 2 , 3 , 4 , 5 \}$ Source-problem hashes and exact duplicates remain in one benchmark-stratified 60/20/20 split; test has 1,332 problems and 5,328 committees.

Automated tests assert constant $y ^ { \mathrm { a l l } }$ , K LOO targets, signedeffect consistency, split exclusivity, provenance for 41 inputs, and absence of result fields. Split files and executions are hashed; source-hash clustering gives CI [+1.13, +2.50].

Before computing review-driven results, we hash-froze an input-validation-first protocol. Shell committees are excluded from router fitting because remaining LOO actions are relative to the unfiltered prompt and cannot represent a second deletion. We retrain and recalibrate on shell-free train/validation data. At test, a validator removes one shell and stops; otherwise the router may remove one message. The inspected test cache makes this review-driven, not a new untouched holdout.

For endpoint confirmation we froze all 158 validator actions plus 300 learned-tail actions (60 per benchmark). Groq’s public openai/gpt-oss-120b receives five calls per full/pruned prompt (temperature 0, low reasoning, 4,096- token cap): 4,580 integrations. Calls were planned full(r) then pruned(r); 450/458 committees preserve complete adjacent order, with retry/resume delays in eight. Repeat IDs are not shared seeds, and this hosted backend is not the source endpoint. The unweighted estimate describes only this sample. For the source workload, validator actions receive weight one, learned-tail actions in benchmark b receive $N _ { b } / 6 0 _ { ; }$ , and nonactions zero. Before inspecting control outcomes, we froze a token-matched deletion control for the learned tail.

## B. Baselines and Metrics

Offline baselines are Keep-All, random prune-one, lowest self-confidence/completeness, longest, and minorityanswer pruning, thresholded to validation coverage. Proposalcorrectness and oracle LOO are diagnostics. Metrics are accuracy, recoveries/corruptions, coverage, and official HARMONY\_GPT\_OSS tokens [20]. Endpoint estimates distinguish sample from design-weighted populations; accuracy is pairing-invariant, while corruption uses repeat-ID, independence-expected, and feasible-coupling definitions. Problem-cluster bootstraps preserve all K.

Procedure 1 (proposed target-backend qualification). On label-blind target calls, prune only if all of the following hold: (i) the design-weighted full-workload accuracy lower 95% bound is at least −1 pp; (ii) acted-input reduction is at least 10%; (iii) the weighted selector-minus-matched-control lower bound is nonnegative; (iv) the weighted expected-corruption UCL is at most 5%; and (v) every prespecified benchmark, K, and completeness marginal on which the deployed pipeline acts has at least 59 acted problem clusters and UCL at most 5%. A zero-corruption group uses the larger of its clustered UCL and $1 \mathrm { ~ - ~ } . 0 5 ^ { 1 / n }$ over independent problem clusters. Marginals with zero pipeline actions by design are exempt because KEEP-ALL incurs no pruning risk. The learned gate abstains at K = 2, but the validator acts on 30 endpoint $K = 2$ committees, so that marginal is not exempt here. This is a global intersection–union decision [21]: any applicable failure or insufficient group returns KEEP-ALL. The complete rule was formulated after observing endpoint results and is applied here only retrospectively.

TABLE I  
PREVIOUSLY SEALED TEST ROUTING RESULTS. TOK. IS OFFICIAL SPECIALIST-MESSAGE PAYLOAD REDUCTION; UCL IS THE 95% PROBLEM-CLUSTER BOOTSTRAP UPPER LIMIT FOR HELPFUL PRUNES/ACTIONS.
<table><tr><td>Method</td><td>Acc.</td><td>∆pp</td><td>Rec./Corr.</td><td>Cov.</td><td>Tok.</td><td>UCL</td></tr><tr><td>Keep-All</td><td>74.89</td><td>0.00</td><td>0/0</td><td>0.0</td><td>0.0</td><td>一</td></tr><tr><td>SafeRoute-MAS</td><td>76.69</td><td>+1.80</td><td>164/68</td><td>43.5</td><td>12.76</td><td>3.61</td></tr><tr><td>Lowest self-conf.</td><td>76.63</td><td>+1.75</td><td>205/112</td><td>40.2</td><td>12.07</td><td>6.14</td></tr><tr><td>Completeness</td><td>76.13</td><td>+1.24</td><td>142/76</td><td>39.8</td><td>11.08</td><td>4.34</td></tr><tr><td>Longest</td><td>76.11</td><td>+1.22</td><td>145/80</td><td>40.1</td><td>13.59</td><td>4.50</td></tr><tr><td>Minority answer</td><td>75.51</td><td>+0.62</td><td>164/131</td><td>39.4</td><td>12.75</td><td>7.28</td></tr><tr><td>Random (100 seeds)a</td><td>75.76</td><td>+0.87</td><td>121.9/75.6</td><td>40.8</td><td>11.68</td><td>1</td></tr><tr><td>Proposal-correct. (diag.)</td><td>77.05</td><td>+2.16</td><td>205/90</td><td>40.6</td><td>12.33</td><td>4.94</td></tr><tr><td>Oracle LOO (cached upper)</td><td>84.29</td><td>+9.40</td><td>501/0</td><td>9.4</td><td>2.53</td><td>0.00</td></tr></table>

<sup>a</sup>The mean random policy has no single selected-action set, so no UCL is assigned. Rows below the rule are non-deployable diagnostics using gold correctness or realized LOO labels.

## VI. RESULTS

## A. Overall Routing Performance

Table I reports sealed results. SafeRoute gains 1.80 points, removes 12.76% of tokens (12.75–12.77% across serializations), and exceeds the random 100-seed expectation by 0.93 points (95% CI [0.46, 1.42]). After Holm adjustment [22], only differences versus random and minority-answer remain significant.

Lowest-self-confidence is statistically indistinguishable from SafeRoute (+0.06 points, CI [−0.53, +0.66]), but its 6.14% helpful-prune UCL exceeds 5%. Completeness and longest pass; SafeRoute is one of several qualifying policies, with the highest point estimate, not uniquely safe.

The cached oracle reaches +9.40 points at 9.4% coverage, quantifying recoverable headroom within the available singledeletion action space.

## B. Input Validation, Missingness, and Token Accounting

In the original frozen policy, 53 of 96 net recoveries arise in 177 schema-incomplete committees, and LAB-Bench supplies 88; 4,673 complete non-LAB committees have 44 recoveries and 44 corruptions. Across the action table, shell labels are asymmetric (673 harmful versus 25 helpful), but the independent-endpoint validator branch changes accuracy by −0.25 points, so this effect does not reproduce.

The frozen review experiment uses a one-shell validator (shortest shell; index tie-break) and a shell-free learned model. The tail retains +0.87 points on 5,170 committees; the exact cascade gains +1.91, adding +0.84 beyond validation (95% CI [+.19, +1.52]). The validator’s 8 corruptions among 158 actions yield 5.06% conditional risk and 8.78% UCL. Thus cleanup helps in-cache but fails the same conditional-risk criterion.

TABLE II  
FROZEN-PRIMARY SUBGROUP RISK AND COVERAGE. RISK AND UCL ARE CONDITIONAL HELPFUL PRUNES/ACTIONS; CORR. IS CORRUPTIONS/ALL. LOBO ∆ USES THE SHELL-FREE, NO-BENCHMARK-FEATURE HELD-OUT EXPERIMENT. PERCENTAGES THROUGHOUT.
<table><tr><td>Scope</td><td>Cov.</td><td>Risk</td><td>UCL</td><td>Corr.</td><td>LOBO ∆ pp</td></tr><tr><td>JEEBench</td><td>50.0</td><td>0.49</td><td>1.46</td><td>0.24</td><td>+0.05</td></tr><tr><td>LAB-Bench</td><td>74.2</td><td>4.78</td><td>6.79</td><td>3.55</td><td>+8.76</td></tr><tr><td>MaScQA</td><td>36.2</td><td>2.13</td><td>4.44</td><td>0.77</td><td>-0.04</td></tr><tr><td>Omni-MATH-2</td><td>41.2</td><td>2.97</td><td>3.88</td><td>1.23</td><td>-0.45</td></tr><tr><td>SciBench</td><td>22.8</td><td>0.95</td><td>2.75</td><td>0.22</td><td>-0.09</td></tr><tr><td> $K = 2$ </td><td>0.0</td><td></td><td></td><td>0.00</td><td>一</td></tr><tr><td> $K = 3$ </td><td>49.5</td><td>1.52</td><td>2.35</td><td>0.75</td><td>一</td></tr><tr><td> $K = 4$ </td><td>59.5</td><td>3.28</td><td>4.36</td><td>1.95</td><td>一</td></tr><tr><td> $K = 5$ </td><td>64.9</td><td>3.70</td><td>4.77</td><td>2.40</td><td>一</td></tr><tr><td>Complete</td><td>42.2</td><td>2.90</td><td>3.61</td><td>1.22</td><td>一</td></tr><tr><td>Incomplete</td><td>79.7</td><td>3.55</td><td>6.21</td><td>2.82</td><td>一</td></tr></table>

Conditional risk counts corruptions per action; committee risk counts harmed committees. Table II reports both together with coverage. The aggregate frozen policy does not certify every subgroup: LAB-Bench and incomplete committees exceed the 5% UCL, as do 8 acted-on cells in the complete 36-cell benchmark×K×completeness analysis. The ordinary bootstrap assigns zero to zero-event cells; our retrospective reanalysis instead uses the exact problem-cluster safeguard above. Of 36 cross-cells, 9 have no action, 17 have insufficient support, 6 pass, and 4 fail risk. A validation-only guard improves the cached aggregate but does not eliminate LAB’s violation; subgroup gating is not a distribution-free guarantee.

At the frozen policy’s validation token budget, realized cached-test reductions are 11.91–12.76% across policies; savings are therefore not unique to SafeRoute. Confidence attains +2.03 points but its 5.27% helpful-prune UCL exceeds the criterion. SafeRoute attains +1.80 points (UCL 3.61%), compared with completeness at +1.50 (4.30%) and longest at +1.24 (4.07%).

## C. Ablation and Sensitivity

Matched-coverage ablations are post-hoc. Metadata alone matches the primary model. Removing masks gives +1.60 points; removing context, benchmark, calibration, or diversity gives +2.08, +1.63, +1.41, or +2.44. No component is causally necessary; context and diversity remain frozen engineering choices.

Variant UCLs are 2.99–3.91%. On acted candidates, predicted versus observed helpful risk is 2.54% versus 2.93% (ECE 0.61%).

## D. Benchmark and Committee-Size Robustness

Figure 2 and Table II show strong heterogeneity: LAB-Bench supplies 88 of 96 original net recoveries, whereas the non-LAB gain is +0.17 (95% CI [−0.30, +0.63]). Shell-free LOBO is clearly positive only for LAB. LAB has the lowest base accuracy and highest shell-committee rate, but is neither the longest nor most answer-divergent benchmark; no observed property defines a universal portability rule.

![](images/696e4786ebd11ace82a322469774d6b8bd0dd7c4ba69fa21f9f47938ea15c358.jpg)

![](images/1c50b447280c23c0b35087e0e7658cbca39c186861101920dd03a6d453242a52.jpg)  
Fig. 2. Cached primary-split accuracy changes by benchmark and committee size (problem-cluster 95% CIs). Heterogeneity motivates target-specific qualification; $K = 2$ is a frozen no-action boundary.

TABLE III  
RETROSPECTIVE ENDPOINT DIAGNOSTIC. WTD. ROWS USE SOURCE-WORKLOAD DESIGN WEIGHTS; TOK. IS ACTED INTEGRATOR-INPUT REDUCTION. CIS/UCLS USE PROBLEM CLUSTERS.
<table><tr><td>Estimand</td><td>Units</td><td colspan="2">∆ pp (95% CI)</td><td>Tok.</td><td>Risk UCL</td></tr><tr><td>Frozen 458 sample</td><td>2,290</td><td></td><td> $- 0 . 0 9 \ [ - 1 . 8 5 , 1 . 6 1 ]$ </td><td>12.63</td><td>5.37a</td></tr><tr><td>Wtd. full workload</td><td>5,328</td><td></td><td> $+ 0 . 9 4 \ [ - 1 . 3 1 , 3 . 5 0 ]$ </td><td>18.72</td><td>4.60b</td></tr><tr><td>Validator branch</td><td>790</td><td></td><td> $- 0 . 2 5 \ [ - 2 . 8 4 , 2 . 1 4 ]$ </td><td>3.97</td><td>7.23b</td></tr><tr><td>Balanced learned tail</td><td>1,500</td><td></td><td>0.00 [−2.32, 2.30]</td><td>16.81</td><td>4.94b</td></tr><tr><td>Wtd. tail vs matched</td><td>3,599</td><td>+1.68</td><td>[−1.27, 5.10]</td><td>+1.63</td><td></td></tr></table>

<sup>a</sup>Repeat-ID pairing; <sup>b</sup>independence-expected corruption. Tok. for the fullworkload row is among routed actions.

Across K = 3, 4, 5, the primary gains are 3.30, 2.10, and 1.80 points. A model trained on $K \leq 4$ acts on every $K = 5$ committee, so its threshold does not transfer across K.

In 478 complete LAB committees, SafeRoute acts 349 times (62 recoveries, 19 corruptions), a 5.4% conditional risk. Post-hoc role/length diagnostics do not reproduce its 43 net recoveries: the strongest fixed-role rule yields only +2.93 points. Thus the aggregate criterion does not certify this stratum, and its mechanism remains unresolved.

## E. Retrospective Endpoint Qualification Diagnostic

Table III separates sample and target-population estimands. The frozen sample gives −0.087 points, but it censors nonactions, includes every validator action, and balances the learned tail across benchmarks. Design weighting over 158 validator and 3,599 learned-tail actions, with zero effect assigned to non-actions, gives +0.943 points on the 5,328- committee workload (95% CI $[ - 1 . 3 1 5 , + 3 . 5 0 1 ] )$ and 18.72% acted-input reduction. Thus neither benefit nor noninferiority is established. The weighted tail-minus-matched estimate is +1.680 points (95% $\mathrm { C I } \ [ - 1 . 2 7 0 , + 5 . 0 9 8 ] )$ ; weighted token reductions are 19.31% and 17.68%, respectively, so selector advantage is also unestablished.

The accuracy difference is pairing invariant, but corruption counts are not. Repeat-ID pairing gives 4.15% risk and 5.37% UCL; five cyclic pairings give UCLs 5.02–5.36%, independence-expected corruption gives 3.98% and 5.24%, and feasible coupling UCLs span 4.34–6.06%. After design weighting, expected risk is 3.05% (UCL 4.60%). By five-call majority, only 13 of 88 cached non-neutral labels (14.8%) retain the same direction; backend shift and repeated-call variability cannot be separated. Of 11 applicable endpoint marginals, 2 pass, five lack 59 problem clusters, and four additional, sufficiently supported marginals exceed the 5% UCL.

Procedure 1 therefore rejects Groq on accuracy and matchedcontrol; the subgroup gate also fails from both insufficient evidence and observed excess risk. The verdict is unchanged for accuracy margins 0.5–1.5 pp, risk limits 4–6%, and token floors 5–15%.

## VII. DISCUSSION AND LIMITATIONS

SafeRoute combines a deployment-visible representation with a calibrated selective gate. Each decision uses its own cached replay; no result sums single-message effects. Metadata-only performance and the context/diversity ablations make this a scoped routing method, not evidence that every advertised component independently helps.

The validator-first rerun separates cleanup from routing within the single-deletion action space. It supports a positive tail in the inspected cache, but endpoint repetition does not confirm it; unvalidated groups should abstain.

The repeated experiment uses a new hosted backend; it cannot isolate endpoint shift from sampling variation, and temperature-zero calls are not seed-controlled. The 1-pp accuracy margin, 10% token floor, and 5% risk tolerance are author-declared engineering values. We distinguish demonstrated diagnostics from deployment claims: Procedure 1 was formalized after observing the endpoint experiment, its Groq application is retrospective, and prospective validation remains future work. As a proposed future deployment rule, SafeRoute defaults to Keep-All unless the exact target passes every gate globally.

## VIII. CONCLUSION

SafeRoute identifies a positive source-compatible cached operating point, but neither the balanced endpoint sample nor its design-weighted workload estimate confirms an accuracy gain or selector advantage. Its applied contribution is a numerical target-specific qualification and abstention procedure, demonstrated here retrospectively: validate inputs, calibrate on the deployment backend, compare with matched pruning, check aggregate and subgroup risks, and otherwise retain Keep-All.

## REFERENCES

[1] Y. Du et al., “Improving factuality and reasoning in language models through multiagent debate,” in Proc. ICML, 2024, pp. 11 733–11 763.

[2] J. C.-Y. Chen, S. Saha, and M. Bansal, “ReConcile: Round-table conference improves reasoning via consensus among diverse LLMs,” in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics, 2024, pp. 7066– 7085.

[3] G. Zhang et al., “Cut the crap: An economical communication pipeline for LLM-based multi-agent systems,” in Proc. Int. Conf. Learn. Representations, 2025.

[4] C.-H. Yang, J. Jiang, C.-H. Yang, V. Vasudevan, H. Zheng, V. Vishwanath, and R. Thakur, “SciAgentTrace: A matched crossdomain dataset of scientific agent protocols,” Hugging Face dataset, version v2.1.0, 2026. [Online]. Available: https://huggingface.co/ datasets/AgentsSci/scientific-agent-protocol-traces

[5] C.-H. Yang et al., “Wrong but useful: Trajectory value beyond answer correctness in multi-agent messages,” 2026, arXiv:2608.14375.

[6] Y. Yue et al., “MasRouter: Learning to route LLMs for multi-agent systems,” in Proc. ACL, 2025, pp. 15 549–15 572.

[7] W. Chen et al., “Optima: Optimizing effectiveness and efficiency for LLM-based multi-agent system,” in Findings ofACL, 2025, pp. 11 534– 11 557.

[8] N. You et al., “MessageSieve: Runtime message pruning for accurate and efficient LLM-based multi-agent communication,” in Proc. EMNLP, 2026, pp. 49 544–49 566.

[9] J. Zhang et al., “Rethinking the evaluation of efficiency methods for multi-agent systems,” in Proc. EMNLP, 2026, pp. 18 527–18 562.

[10] C. Guo et al., “On calibration of modern neural networks,” in Proc. ICML, 2017, pp. 1321–1330.

[11] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in Advances in Neural Information Processing Systems 30, 2017.

[12] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.

[13] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” 2025, arXiv:2508.10925.

[14] D. Arora, H. Singh, and Mausam, “Have LLMs advanced enough? A challenging problem solving benchmark for large language models,” in Proc. EMNLP, 2023, pp. 7527–7543.

[15] J. M. Laurent et al., “LAB-Bench: Measuring capabilities of language models for biology research,” 2024, arXiv:2407.10362.

[16] M. Zaki et al., “MaScQA: Investigating materials science knowledge of large language models,” Digital Discovery, vol. 3, pp. 313–327, 2024.

[17] B. Gao et al., “Omni-MATH: A universal olympiad level mathematic benchmark for large language models,” in Proc. Int. Conf. Learn. Representations, 2025.

[18] M. Ballon et al., “Benchmarks saturate when the model gets smarter than the judge,” 2026, arXiv:2601.19532.

[19] X. Wang et al., “SciBench: Evaluating college-level scientific problemsolving abilities of large language models,” in Proc. ICML, 2024, pp. 50 622–50 649.

[20] D. Kundel, “How to run gpt-oss with Transformers,” OpenAI Developers Cookbook, 2025, Accessed: Nov. 2, 2026. [Online]. Available: https: //developers.openai.com/cookbook/articles/gpt-oss/run-transformers

[21] R. L. Berger, “Multiparameter hypothesis testing and acceptance sampling,” Technometrics, vol. 24, no. 4, pp. 295–300, 1982.

[22] S. Holm, “A simple sequentially rejective multiple test procedure,” Scandinavian Journal of Statistics, vol. 6, no. 2, pp. 65–70, 1979.