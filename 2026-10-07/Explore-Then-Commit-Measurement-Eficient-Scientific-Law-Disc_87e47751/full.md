# Explore, Then Commit: Measurement-Eficient Scientific Law Discovery with Language Models

Kautik Mandve<sup>1</sup> Dileepa Fernando<sup>2</sup>

<sup>1</sup>Vernon Hills High School, Vernon Hills, Illinois, USA <sup>2</sup>Information Systems Technology and Design Singapore University of Technology and Design, Singapore

2 October 2026

## Abstract

Scientific law discovery requires selecting measurements and converting evidence into a governing equation. We evaluate an explore-then-commit protocol in which a large language model proposes hypotheses, a programmatic planner gathers measurements, and a fresh prompt synthesizes the final law from fixed observations. The protocol combines structured probes, automatic numerical diagnostics, restricted measurement batches, and optional interpreter access. Across 576 NewtonBench trials, we compare eight configurations on 12 physics modules using GPT-4.1-mini and a medium-dificulty GPT-4.1 replication. On medium tasks, interpreter-enabled planners use 8.6 versus 22.5 measurements per trial for GPT-4.1-mini and 8.9 versus 43.0 for GPT-4.1. Their mean magnitude-based root-mean-squared logarithmic error falls from 2.514 to 0.202 and from 0.626 to 0.149, respectively. An additional audit retains incomplete and invalid submissions in a coverage-sensitive analysis. Observed symbolic-accuracy gains are less consistent across modules, and random acquisition is competitive with disagreement scoring. Measurement savings occur in every module, but unequal batch constraints prevent attributing them solely to acquisition quality. These results support the complete protocol as a promising measurement-eficient configuration, while leaving its causal components and generalization beyond noiseless direct-equation tasks unresolved.

## 1 Introduction

Scientific problem solving couples two decisions: which observation to obtain next and which explanation to retain afterward. A discovery agent can waste measurements by repeatedly testing similar inputs, or fit a familiar equation before establishing its dependencies. Improving these decisions requires an evaluation that distinguishes the measurements collected, the predictive quality of the resulting function, and whether that function expresses the hidden law. These objectives are related, but success on one does not guarantee success on the others.

For example, a function may closely predict the outputs observed in one input range while omitting a weak dependency that becomes important elsewhere. Conversely, an agent may identify the right functional structure with imperfect fitted constants. Reporting measurement counts, numerical fit, and symbolic recovery together makes these diferent outcomes visible and helps locate where a discovery procedure succeeds or fails.

NewtonBench makes this problem accessible through hidden physics simulators whose governing laws deliberately depart from textbook equations [1]. Altered exponents and dependencies reduce the usefulness of memorized formulas. Its experiments also associate some failures of interpreterenabled agents with premature exploitation of early hypotheses. This motivates inspecting how acquisition and equation construction are organized, rather than treating access to a numerical tool as suficient for discovery. Our study tests a protocol within NewtonBench’s existing task suite.

Equation discovery has a substantial history outside large language models (LLMs). Evolutionary search [2], sparse identification of dynamics [3], AI Feynman [4], and PySR [5] recover interpretable mathematical structure; SRBench compares symbolic-regression methods systematically [6]. Although many evaluations use fixed datasets, discovery is not inherently passive: Bongard and Lipson combined model inference with informative perturbation experiments [7]. LLM-SR incorporates language-model proposals into equation search [8], while LLM-SRBench examines scientific equation discovery on memorization-resistant tasks [9]. DiscoveryWorld and BoxingGym broaden agent evaluation to simulated discovery and experimental design [10, 11].

Our acquisition rules draw on the principle that competing hypotheses help identify informative measurements. Bayesian experimental design formalizes information gathering [12]; query by committee selects through disagreement [13]; Bayesian Active Learning by Disagreement uses a probabilistic information criterion [14]. BED-LLM connects language-model information gathering with Bayesian experimental design [15]. Here the committees are language-model proposals, not calibrated posterior samples, and the scores are heuristic disagreement measures rather than mutual-information estimates.

We evaluate an explore-then-commit configuration. Structured probes and automatic numerical diagnostics support initial inference. A language model proposes equations, a programmatic planner selects further inputs, and a fresh prompt synthesizes a final law from fixed observations and any draft. Interpreter-enabled variants additionally permit model-initiated Python analysis. This allocation of work relates to program-aided language models and Program of Thoughts [16, 17], and to separating LLM proposals from external checks in robotic planning [18]. Unlike verification against a known planning model, our numerical checks cannot certify an unknown law’s symbolic correctness.

The study asks whether the complete configuration improves measurement use and recovery relative to free-form agents, whether disagreement scoring improves on random acquisition, and whether the findings persist across models and stricter numerical accounting. Across 576 trials, the clearest pattern is fewer measurements and lower mean numerical error. Symbolic gains are concentrated in fewer modules, and random selection remains competitive. We also identify submissions whose finite reported error conceals incomplete predictions. The contribution is therefore a reproducible protocol comparison and evaluation audit: it establishes the behavior of the implemented configurations while distinguishing that evidence from causal claims about their individual components.

## 2 Materials and Methods

## 2.1 Tasks and Experimental Grid

Each task hides a scalar function $y = f ^ { \star } ( x )$ . An agent queries inputs and outputs, then submits an executable $\mathrm { P y }$ thon function named discovered\_law. The twelve modules cover gravity, Coulomb and magnetic forces, Fourier conduction, Snell refraction, radioactive decay, harmonic motion, Malus intensity, sound speed, Hooke energy, Bose–Einstein occupation, and heat transfer. We use the direct-output vanilla\_equation tier, law version $\mathtt { v 0 } .$ , and zero observation noise. The labels denote benchmark task families; the altered laws are not claims about actual physics.

GPT-4.1-mini supplies $8 \times 1 2 \times 2 \times 2 = 3 8 4$ trials across eight configurations, twelve modules, easy and medium dificulties, and two rollouts per cell. GPT-4.1 supplies another $8 \times 1 2 \times 2 = 1 9 2$ medium trials. This second grid was selected after larger primary-study diferences appeared on medium tasks; it is a targeted replication. Recorded aliases are gpt41mini and gpt41; provider routes and exact hosted-model snapshots are not fully resolved. Generation temperature is 0.4 and planner pool seeds are 0 and 1. Rollouts are not paired through common model-randomness seeds. All 576 final submissions are retained, including sixteen interpreter-enabled baseline outcomes that reached their turn limit; no terminal failure artifacts occur.

## 2.2 Configurations, Budgets, and Protocol

We compare two free-form baselines, without and with an interpreter, against variance, pairwise, and random planners in both interpreter conditions. Interpreter access means the LLM can execute its own analysis code. Every planner also receives automatic sensitivity estimates and power-law regression diagnostics, which the baselines do not receive. Thus an agent without an interpreter can still receive substantive numerical assistance.

Baselines permit ten model-action rounds plus a forced final-submission call when needed. Their experiment parser accepts variable-length measurement lists; a suggested batch limit of twenty is unenforced. Planner protocol p4 allows eight exploration turns followed by one synthesis turn without an interpreter or at most three with one. Each post-bootstrap planner batch is capped at four measurements. An exploration turn can instead perform analysis, repair an action, or submit a draft. Turn limits therefore do not specify equal measurement budgets. We count individual simulator evaluations, and interpret savings as outcomes of these complete interaction rules.

![](images/6384bc556047256f4e0aba69ca6b9d18e5b552f848828bcc2a64ab84e1ca3ca3.jpg)  
Figure 1: The planner protocol. The large language model (LLM) proposes laws; every variant receives automatic numerical diagnostics. Fresh synthesis receives fixed observations and any draft, with no new simulator access.

Figure 1 shows the division of work. Bootstrap probes vary one coordinate at a time around a transformed midpoint, using at least three and at least d + 1 measurements for d inputs. Positive observations support log-ratio sensitivity estimates and a least-squares fit of log y to the log inputs. The supplied exponents and residuals can expose multiplicative structure directly. During exploration, the model proposes up to five hypotheses. Scored planners require at least two parseable candidates; the random variant can continue with fallback hypotheses. A draft submission ends exploration.

Scored acquisition evaluates calibrated hypotheses on a pool of 2048 scrambled Sobol inputs. A fit filter retains candidates consistent with observations. Variance acquisition measures disagreement after standardizing each hypothesis across the pool; pairwise acquisition uses mean absolute disagreement after median and interquartile-range normalization. Boundary penalties and withinbatch diversity modify selection. Random acquisition samples unused pool points without replacement, bypassing filtering and score regularization while sharing bootstrap probes, diagnostics, batch constraints, and synthesis. Supplementary Sections S1–S3 give pseudocode, formulas, constants, and domains.

Calibration fits a scale and, with suficient observations, an ofset to each candidate. It therefore adjusts a proposed equation’s predictions without optimizing every nuisance parameter independently. The diagnostics and fit filter use only acquired observations, and the shared evaluation grid is never presented to the agent. Candidate-pool coordinates follow each simulator’s linear or logarithmic input distribution. These choices define the tested acquisition procedure; no pool-size or calibration ablation was conducted.

Fresh synthesis receives the task, observations, and any nontrivial draft rather than the full exploration history. It permits local analysis when an interpreter is available and refuses new experiment requests. A valid synthesis law replaces the draft; otherwise the draft is retained. The instruction favors retaining a draft unless observations disagree, so the reset does not eliminate all anchoring. Recorded traces confirm a synthesis pass and nonempty diagnostics in all 432 planner trials; three use bootstrap measurements alone.

## 2.3 Metrics, Aggregation, and Verification

Symbolic Accuracy (SA) is NewtonBench’s saved binary GPT-4.1 judgment at temperature 0.6. The evaluator can accept fitted-constant diferences while rejecting incorrect dependencies. SA is benchmark-judged recovery, not deterministic symbolic equivalence. We preserve every recorded judgment without rerunning the evaluator.

Numerical error is replayed locally on deterministic, shared grids of 5000 inputs per module, using dificulty-specific truth. The benchmark’s magnitude-based root-mean-squared logarithmic error (RMSLE) is

$$
R = \sqrt { \frac { 1 } { | V | } \sum _ { j \in V } \left[ \log ( 1 + | \hat { f } ( x _ { j } ) | ) - \log ( 1 + | f ^ { \star } ( x _ { j } ) | ) \right] ^ { 2 } } .\tag{1}
$$

The implementation removes input NaN pairs, then averages non-NaN squared log diferences indexed by V . Infinite terms can invalidate the score, and function exceptions invalidate evaluation. Absolute values discard sign. Because the retained set V can depend on the submission, a finite score can conceal undefined predictions. Reported RMSLE means exclude non-finite trial scores and disclose the retained count.

Our retrospective coverage analysis requires a submission to return finite predictions at every sampled input where the target is finite. Let $C _ { t }$ indicate this completeness for trial t. We report $Q ( \tau )$ , the fraction of $a l l$ trials with $C _ { t } = 1$ and finite RMSLE at most τ. Invalid and incomplete submissions remain in the denominator. Thresholds 0.01, 0.1, and 1 summarize the curve; they are descriptive choices, not calibrated scientific tolerances. Sign disagreements are audited separately.

Each baseline row has 24 rollouts; each planner-family row averages three equally represented variants, totaling 72. It does not select the best variant or combine answers. Balanced cells give modules equal weight for SA and measurement means, although finite-only error averages can weight them unequally. We average repeats and variants within modules for sensitivity analysis, using 10,000 module-resampling draws, leave-one-module-out ranges, and four secondary sign-flip diagnostics (Supplementary Section S4). These fixed-suite summaries do not resolve uncertainty from two rollouts per cell.

Numerical replay uses independent evaluation inputs shared across models, configurations, and repeats. The dificulty-specific truth changes while the input grid stays fixed. This removes unequal test samples as a source of between-configuration error diferences; it does not make diferently acquired training observations equivalent or establish performance outside the benchmark’s input distribution.

Supplementary Material S1, appended after the references, documents additional methods and results. The S2 code-and-data snapshot is provided with the arXiv ancillary files and contains sanitized records, code, dependencies, and hashes. Ofline replay reproduces the existing scores and leaves the original data unchanged. All 49 protocol, replay, and coverage tests pass. The reference dependency snapshot uses Python 3.10.18; this audit uses Python 3.11.7. Claude and Codex assisted software and manuscript development, including this revision’s analysis, writing, and data-rendered figures; details appear in the software-assistance disclosure. No new modelgenerated trial outcomes were created for this revision.

## 3 Results

## 3.1 Measurement Use and Numerical Fit

Table 1 summarizes every model–dificulty–family condition. On medium tasks, interpreter-enabled planners use 8.6 measurements per rollout versus 22.5 for the GPT-4.1-mini baseline, a 61.9% reduction from unrounded means. GPT-4.1 uses 8.9 versus 43.0, a 79.3% reduction. Without an interpreter, savings are 79.9% and 88.6%, respectively. Figure 2 shows positive savings in all twelve modules in each of the four model–tool comparisons. This consistency describes the implemented protocols, including their unequal batch constraints.

Table 1: Complete family-level results. N/C denote no interpreter/interpreter access; GPT prefixes are omitted. SA denotes Symbolic Accuracy and RMSLE root-mean-squared logarithmic error. Credited counts symbolic recoveries; Finite counts finite RMSLE scores; Meas. counts simulator evaluations per rollout. RMSLE averages only finite scores. Planner rows average all three acquisition variants.
<table><tr><td>Model /</td><td>difficulty</td><td>Configuration</td><td>n</td><td>Credited</td><td>SA (%)</td><td>RMSLE</td><td>Finite</td><td>Meas.</td></tr><tr><td>4.1-mini</td><td>easy</td><td>Baseline N</td><td>24</td><td>5/24</td><td>20.8</td><td>1.405</td><td>24/24</td><td>43.2</td></tr><tr><td>4.1-mini</td><td>easy</td><td>Planner N</td><td>72</td><td>34/72</td><td>47.2</td><td>1.165</td><td>71/72</td><td>9.4</td></tr><tr><td>4.1-mini</td><td>easy</td><td>Baseline C</td><td>24</td><td>13/24</td><td>54.2</td><td>1.046</td><td>24/24</td><td>20.8</td></tr><tr><td>4.1-mini</td><td>easy</td><td>Planner C</td><td>72</td><td>39/72</td><td>54.2</td><td>0.290</td><td>71/72</td><td>8.5</td></tr><tr><td>4.1-mini</td><td>medium</td><td>Baseline N</td><td>24</td><td>2/24</td><td>8.3</td><td>2.059</td><td>23/24</td><td>44.0</td></tr><tr><td>4.1-mini</td><td>medium</td><td>Planner N</td><td>72</td><td>27/72</td><td>37.5</td><td>1.167</td><td>66/72</td><td>8.8</td></tr><tr><td>4.1-mini</td><td>medium</td><td>Baseline C</td><td>24</td><td>4/24</td><td>16.7</td><td>2.514</td><td>24/24</td><td>22.5</td></tr><tr><td>4.1-mini</td><td>/medium</td><td>Planner C</td><td>72</td><td>29/72</td><td>40.3</td><td>0.202</td><td>72/72</td><td>8.6</td></tr><tr><td>4.1 / medium</td><td></td><td>Baseline N</td><td>24</td><td>4/24</td><td>16.7</td><td>3.018</td><td>22/24</td><td>95.2</td></tr><tr><td></td><td>4.1 / medium</td><td>Planner N</td><td>72</td><td>21/72</td><td>29.2</td><td>0.634</td><td>68/72</td><td>10.9</td></tr><tr><td>4.1 / medium</td><td></td><td>Baseline C</td><td>24</td><td>9/24</td><td>37.5</td><td>0.626</td><td>24/24</td><td>43.0</td></tr><tr><td>4.1 / medium</td><td></td><td>Planner C</td><td>72</td><td>31/72</td><td>43.1</td><td>0.149</td><td>72/72</td><td>8.9</td></tr></table>

Medium-task mean RMSLE for interpreter-enabled planners falls from 2.514 to 0.202 on GPT-4.1-mini and from 0.626 to 0.149 on $\mathrm { G P T - 4 . 1 }$ . Without an interpreter, the changes are 2.059 to 1.167 and 3.018 to 0.634. Thus both numerical fit and measurement use favor the planner families in all four comparisons. The second model reproduces this direction despite stronger baseline recovery, rather than merely reproducing an advantage over a uniformly unsuccessful baseline.

Easy GPT-4.1-mini tasks show less separation in symbolic recovery: interpreter-enabled planners match the baseline at 54.2%, while using 8.5 versus 20.8 measurements. The no-interpreter planner family reaches 47.2% versus 20.8% for its baseline. Supplementary Table S4 reports all eight easy-task configurations, including variants below their matching baseline. Lower measurement counts do not establish lower total compute or API cost; recorded output tokens omit prompt tokens and the protocol adds external calculations and synthesis.

The family average also conceals diferent easy-task tradeofs. With an interpreter, variance receives 11 symbolic credits, compared with 13 for baseline and 14 each for pairwise and random. Random has the lowest mean numerical error in this group, at 0.102. Thus the unchanged familylevel SA does not indicate identical behavior among its components, and the medium-task ordering of scored and random numerical performance does not extend to every condition.

![](images/6cc50b13c8c73d3bd467040332f9e141ae5be3ac65678e76294ae8b423adeb11.jpg)  
Figure 2: Medium-task measurement savings. Each point subtracts six planner rollouts’ mean from two matching baseline rollouts’ mean. All module-level diferences are positive; comparisons include unequal batch limits and automatic planner diagnostics.

## 3.2 Symbolic Recovery and the Random Control

On GPT-4.1-mini medium tasks, planner-family SA increases from 8.3% to 37.5% without an interpreter and from 16.7% to 40.3% with one. These gains are heterogeneous: the no-interpreter comparison has five positive module efects and seven ties, while the interpreter comparison has four positive efects, one negative efect, and seven ties. GPT-4.1 gains are smaller, at 12.5 and 5.6 percentage points; the latter involves two positive modules and ten ties. Mean diferences remain positive when any one module is removed, but none of the four unadjusted sign-flip diagnostics is below 0.05 (Supplementary Section S4.1 and Table S2).

For GPT-4.1 without an interpreter, three module efects are positive, one is negative, and eight are tied. The second model therefore repeats the positive aggregate direction without producing a broad shift in symbolic success across the suite. A tied module can reflect common success or common failure; the recovery heatmap distinguishes these cases instead of treating every tie as evidence that the task was solved.

Table 2: All medium-task configurations; 24 rollouts per row. Int. denotes interpreter access. Credited, Finite, and Complete are counts out of 24. Complete requires predictions throughout the sampled finitetarget domain; RMSLE means exclude non-finite scores.
<table><tr><td>Model</td><td>Configuration</td><td>Credited</td><td>SA (%)</td><td>RMSLE</td><td>Finite</td><td>Complete</td><td>Meas.</td></tr><tr><td>4.1-mini</td><td>Baseline, no int.</td><td>2</td><td>8.3</td><td>2.059</td><td>23</td><td>21</td><td>44.0</td></tr><tr><td>4.1-mini</td><td>Baseline, int.</td><td>4</td><td>16.7</td><td>2.514</td><td>24</td><td>22</td><td>22.5</td></tr><tr><td>4.1-mini</td><td>Variance, no int.</td><td>8</td><td>33.3</td><td>1.394</td><td>22</td><td>22</td><td>9.0</td></tr><tr><td>4.1-mini</td><td>Pairwise, no int.</td><td>9</td><td>37.5</td><td>0.899</td><td>21</td><td>21</td><td>7.7</td></tr><tr><td>4.1-mini</td><td>Random, no int.</td><td>10</td><td>41.7</td><td>1.195</td><td>23</td><td>21</td><td>9.8</td></tr><tr><td>4.1-mini</td><td>Variance, int.</td><td>10</td><td>41.7</td><td>0.189</td><td>24</td><td>24</td><td>8.3</td></tr><tr><td>4.1-mini</td><td>Pairwise, int.</td><td>10</td><td>41.7</td><td>0.155</td><td>24</td><td>24</td><td>8.3</td></tr><tr><td>4.1-mini</td><td>Random, int.</td><td>9</td><td>37.5</td><td>0.263</td><td>24</td><td>24</td><td>9.0</td></tr><tr><td>4.1</td><td>Baseline, no int.</td><td>4</td><td>16.7</td><td>3.018</td><td>22</td><td>22</td><td>95.2</td></tr><tr><td>4.1</td><td>Baseline, int.</td><td>9</td><td>37.5</td><td>0.626</td><td>24</td><td>23</td><td>43.0</td></tr><tr><td>4.1</td><td>Variance, no int.</td><td>7</td><td>29.2</td><td>0.504</td><td>22</td><td>22</td><td>10.7</td></tr><tr><td>4.1</td><td>Pairwise, no int.</td><td>7</td><td>29.2</td><td>0.532</td><td>24</td><td>22</td><td>11.3</td></tr><tr><td>4.1</td><td>Random, no int.</td><td>7</td><td>29.2</td><td>0.875</td><td>22</td><td>20</td><td>10.7</td></tr><tr><td>4.1</td><td>Variance, int.</td><td>10</td><td>41.7</td><td>0.115</td><td>24</td><td>24</td><td>9.2</td></tr><tr><td>4.1</td><td>Pairwise, int.</td><td>10</td><td>41.7</td><td>0.118</td><td>24</td><td>24</td><td>8.3</td></tr><tr><td>4.1</td><td>Random, int.</td><td>11</td><td>45.8</td><td>0.213</td><td>24</td><td>24</td><td>9.2</td></tr></table>

The complete acquisition comparison in Table 2 gives no consistent symbolic-recovery ordering. For GPT-4.1-mini with an interpreter, variance and pairwise each recover 10 of 24 laws versus 9 for random. Without an interpreter, random obtains 10 recoveries, pairwise 9, and variance 8. GPT-4.1 with an interpreter reverses the first ordering: random obtains 11 recoveries versus 10 for each scored variant. Diferences of one recovery cannot establish reliable separation with only two rollouts per module.

Scored interpreter-enabled variants do have lower mean RMSLE than random in both models: 0.189 and 0.155 versus 0.263 on GPT-4.1-mini, and 0.115 and 0.118 versus 0.213 on GPT-4.1. This ordering is descriptive; filtering and fallback behavior difer alongside scoring. Supplementary Figure S1 shows the module-level recovery pattern. Seven medium GPT-4.1-mini modules remain unrecovered by any configuration; harmonic motion, Malus intensity, and Bose–Einstein occupation have no credited recoveries anywhere in the collection. GPT-4.1 does obtain some Snell recoveries.

## 3.3 Prediction Completeness and Error

Fifteen of 576 replayed scores are non-finite. Another fourteen submissions have finite RMSLE despite missing predictions where the target is finite; all fourteen concern Snell refraction. One easy GPT-4.1-mini interpreter-enabled baseline predicts at only 770 of 4259 finite-target points yet receives RMSLE 0.957. The Snell target itself is finite on 4259 easy and 4264 medium points out of 5000, so completeness is assessed relative to that domain, not undefined target outputs. In total, 29 submissions fail the completeness check. Separately, two complete submissions have sign disagreements that magnitude RMSLE cannot penalize as sign errors.

The fit advantage persists under this stricter accounting (Figure 3; Supplementary Table S3). All 72 interpreter-enabled medium planner submissions are complete in each model. At τ = 0.1, the all-trial rates are 59.7% versus 25.0% for the GPT-4.1-mini baseline and 66.7% versus 54.2% for

![](images/c865b5da99c6ac8ff60be81dd67557275ec0033142317aa591c1105487106761.jpg)  
Figure 3: Coverage-sensitive medium-task error curves. Each curve counts all rollouts, requiring complete predictions on the sampled finite-target domain and RMSLE at most the threshold. Incomplete or invalid submissions prevent curves from reaching 100%.

GPT-4.1. The gaps depend on the tolerance. Requiring completeness prevents selective prediction domains from improving the reported pass rate, while retaining the distinction between a close predictive function and a recovered symbolic law.

Both sign-disagreeing submissions have RMSLE below 0.1 and consequently satisfy that coverage-sensitive threshold. Completeness repairs one weakness of finite-only averages, but does not repair the underlying metric’s sign insensitivity. The analysis therefore preserves the legacy error measure for comparability while exposing the additional information needed to interpret it. No original judge decision or submitted function is replaced by the retrospective audit.

## 4 Discussion

The repeated finding is a useful complete configuration: structured acquisition, automatic diagnostics, and fresh synthesis produce lower mean numerical error using fewer measurements under the implemented rules. The numerical advantage survives an audit that retains incomplete and invalid submissions. This matters for scientific agents whose measurements are costly, but the practica interpretation should remain specific to the interface: lower simulator use is established, whereas superior information acquisition at an equal measurement budget is not.

The largest limitation is that several interventions change together. Baselines accept uncapped measurement arrays; planners use bounded batches, bootstrap probes, automatic regression, hypothesis prompting, and fresh synthesis. In particular, log-linear diagnostics can supply much of a multiplicative law’s structure, even without model-initiated Python. A cap-matched baseline is the first decisive follow-up. Separately removing diagnostics and fresh synthesis, with matched measurement availability and model-call opportunities, would identify their contributions. Fitting a dedicated symbolic-regression solver to the same observations would test the value added by LLM synthesis. These comparisons have not been performed.

Random acquisition further limits a scoring-specific explanation. Its competitive recovery suggests that shared protocol components may account for much of the observed improvement; it does not establish equivalence of acquisition rules. More independent rollouts and prospectively fixed comparisons are needed. Module-resampling summaries cannot compensate for two repetitions per cell, and leave-one-module-out positivity cannot exclude efects concentrated in a few tasks. Medium-only replication was selected after inspecting primary results and uses the same provider’s model family. Neither design feature establishes broad external validity.

Evaluation also separates predictive usefulness from scientific explanation. A harmonic submission can have RMSLE below $1 0 ^ { - 3 }$ despite using the wrong damping dependency (Supplementary Section S4.4). Close numerical fit can miss weak dependencies; sampled completeness cannot guarantee of-grid validity. The stochastic GPT-4.1 judge shares a model family with discovery, can accept coeficient diferences, and has not been independently adjudicated. The sign-insensitive metric and limited provider metadata further constrain interpretation and exact recollection reproducibility.

The study covers noiseless direct equations, not harder laws, alternative versions, or indirect dynamics. Sensitivity to pool size, bootstrap design, regularization, and temperature remains unmeasured. Both tested baselines benefit from interpreter access, so these data do not demonstrate reversal of NewtonBench’s tool-assistance paradox. No preregistration is documented; the coverage and leave-one-module-out analyses are retrospective. Within that scope, the evidence supports making acquisition and synthesis separately inspectable, accounting for all numerical assistance, and reporting predictive coverage alongside symbolic recovery.

For subsequent evaluations, this suggests publishing the observations available at synthesis and the numerical diagnostics supplied to each configuration. Those artifacts make the division of work auditable and allow later comparisons to reuse identical evidence rather than recollecting it under diferent conditions.

## 5 Conclusions

Across 576 NewtonBench trials, the explore-then-commit configuration uses fewer measurements and achieves lower mean numerical error than the implemented baselines. The fit advantage repeats across two models and remains favorable when incomplete predictions are retained in the analysis. Symbolic gains are less uniform, and random acquisition competes with disagreement scoring. These findings support the complete protocol within the tested conditions. A measurement-cap-matched baseline, followed by diagnostic and synthesis ablations with more independent rollouts, is needed to determine which components cause the improvements and how reliably they generalize.

## Reproducibility and Software Assistance

Supplementary Material S1 follows the references. The S2 code-and-data snapshot is included in the arXiv ancillary files and provides the 576 recorded trials, ofline replay, analysis scripts, dependencies, and a SHA-256 manifest. Original NewtonBench software is distributed under its retained MIT license. Model provider calls and symbolic judging are not required for ofline reproduction.

Anthropic Claude and OpenAI Codex assisted with software implementation, manuscript development, literature checking, analysis code, and figure-generation code. Earlier tool-version identifiers were not fully preserved. The figures are rendered from the recorded data. No additional model-generated experimental outcomes were created for this revision.

## References

[1] Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Jiayang Cheng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, and Simon See. NewtonBench: Benchmarking generalizable scientific law discovery in LLM agents. In International Conference on Learning Representations, 2026. doi: 10.48550/arXiv.2510.07172. URL https://openreview.net/forum?id=Gk6umqW74m.

[2] Michael Schmidt and Hod Lipson. Distilling free-form natural laws from experimental data. Science, 324(5923):81–85, 2009. doi: 10.1126/science.1165893.

[3] Steven L. Brunton, Joshua L. Proctor, and J. Nathan Kutz. Discovering governing equations from data by sparse identification of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 113(15):3932–3937, 2016. doi: 10.1073/pnas.1517384113.

[4] Silviu-Marian Udrescu and Max Tegmark. AI Feynman: A physics-inspired method for symbolic regression. Science Advances, 6(16):eaay2631, 2020. doi: 10.1126/sciadv.aay2631.

[5] Miles Cranmer. Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv preprint arXiv:2305.01582, 2023. URL https://arxiv.org/abs/2305.01582.

[6] William La Cava, Patryk Orzechowski, Bogdan Burlacu, Fabrício Olivetti de França, Marco Virgolin, Ying Jin, Michael Kommenda, and Jason H. Moore. Contemporary symbolic regression methods and their relative performance. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2021. URL https://arxiv.org/abs/2107.14351.

[7] Josh Bongard and Hod Lipson. Automated reverse engineering of nonlinear dynamical systems. Proceedings of the National Academy of Sciences, 104(24):9943–9948, 2007. doi: 10.1073/pnas.0609476104.

[8] Parshin Shojaee, Kazem Meidani, Shashank Gupta, Amir Barati Farimani, and Chandan K. Reddy. LLM-SR: Scientific equation discovery via programming with large language models. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2404.18400.

[9] Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D. Doan, and Chandan K. Reddy. LLM-SRBench: A new benchmark for scientific equation discovery with large language models. In 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 55325–55359. PMLR, 2025. URL https://proceedings.mlr.pr ess/v267/shojaee25a.html.

[10] Peter Jansen, Marc-Alexandre Côté, Tushar Khot, Erin Bransom, Bhavana Dalvi Mishra, Bodhisattwa Prasad Majumder, Oyvind Tafjord, and Peter Clark. DiscoveryWorld: A virtual environment for developing and evaluating automated scientific discovery agents. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, 2024. URL https://arxiv.org/abs/2406.067 69.

[11] Kanishk Gandhi, Michael Y. Li, Lyle Goodyear, Agam Bhatia, Louise Li, Aditi Bhaskar, Mohammed Zaman, and Noah D. Goodman. BoxingGym: Benchmarking progress in automated experimental design and model discovery. arXiv preprint arXiv:2501.01540, 2025. URL https://arxiv.org/abs/2501.0 1540.

[12] Kathryn Chaloner and Isabella Verdinelli. Bayesian experimental design: A review. Statistical Science, 10(3):273–304, 1995. doi: 10.1214/ss/1177009939.

[13] H. S. Seung, M. Opper, and H. Sompolinsky. Query by committee. In Fifth Annual ACM Workshop on Computational Learning Theory, pages 287–294, 1992. doi: 10.1145/130385.130417.

[14] Neil Houlsby, Ferenc Huszár, Zoubin Ghahramani, and Máté Lengyel. Bayesian active learning for classification and preference learning. arXiv preprint arXiv:1112.5745, 2011. URL https://arxiv.or g/abs/1112.5745.

[15] Deepro Choudhury, Sinead Williamson, Adam Goliński, Ning Miao, Freddie Bickford Smith, Michael Kirchhof, Yizhe Zhang, and Tom Rainforth. BED-LLM: Intelligent information gathering with LLMs and bayesian experimental design. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2508.21184.

[16] Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. PAL: Program-aided language models. In International Conference on Machine Learning, 2023. URL https://arxiv.org/abs/2211.10435.

[17] Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023. URL https://arxiv.org/abs/2211.12588.

[18] Drejc Pesjak and Jure Žabkar. Robot planning via llm proposals and symbolic verification. Machine Learning and Knowledge Extraction, 8(1):22, 2026. doi: 10.3390/make8010022.

## Supplementary Material S1

This supplementary material accompanies the NewtonBench protocol study and supplies detailed methods and secondary results. LLM denotes large language model, SA Symbolic Accuracy, and RMSLE root-mean-squared logarithmic error. All values derive from the same 576 recorded trials; no additional acquisition or judging was performed. The companion code-and-data archive is designated S2.

## S1 Exploration and Synthesis Protocol

A configured bootstrap count of three is raised to at least d + 1 for a d-variable task. Structured probes begin at the midpoint in transformed coordinates, then move each variable separately to its 0.75 fraction. If more probes are needed, the 0.25 fraction is used; thus one-variable Hooke tasks receive three probes. All other inputs remain centered. The system computes log-ratio sensitivity estimates where positive observations permit them, and fits

$$
\log y \approx \beta _ { 0 } + \sum _ { k = 1 } ^ { d } \beta _ { k } \log x _ { k }\tag{S1}
$$

by least squares when suficient positive data are available. The estimates, residuals, and fitted exponents are presented as advisory diagnostics, not as ground truth. They can directly expose a power-law structure and therefore constitute substantive numerical assistance, including for the agents without an interpreter.

During exploration, a model may propose up to five symbolic hypotheses in a JSON committee. Scored planners require at least two parseable hypotheses; otherwise they request repair. The random variant can continue using fallback hypotheses because selection does not require a disagreement signal. Interpreter-enabled variants may instead perform a Python calculation. A draft submission ends exploration and is retained for the synthesis stage.

After exploration, a new message history receives the task description, serialized observations, and any nontrivial draft. The preceding conversation is not copied, although the draft carries information derived during exploration. The synthesis instruction favors retaining a draft unless the observations disagree; a context reset therefore does not eliminate all anchoring. Interpreterenabled synthesis permits local Python analysis. Experiment requests are refused. A valid synthesis submission replaces the draft; the draft is retained only when synthesis yields no valid submission. The audit of recorded traces verifies one synthesis pass in each of the 432 planner trials. Only three planner trials use bootstrap measurements alone. All 432 record nonempty automatic diagnostics.

Algorithm S1 Planner protocol evaluated in this study   
1: Construct candidate pool $x ;$ collect structured bootstrap data D.   
2: for at most eight exploration turns do   
3: Present observations and automatic diagnostics to the LLM.   
4: Receive a draft law, an allowed local calculation, or a hypothesis committee.   
5: if a draft law is submitted then   
6: Store draft; break.   
7: else if an allowed local calculation is requested then   
8: Execute and return its output; continue.   
9: else if a scored planner lacks a valid committee then   
10: Request repair; continue.   
11: end if   
12: Select up to four unused inputs: uniformly for random, otherwise by filtered disagreement   
and diversity.   
13: Query simulator and append the measurements to D.   
14: end for   
15: Start fresh synthesis with D and any draft; disable new measurements.   
16: Return a valid synthesis law, or the draft if synthesis yields none.

## S2 Acquisition Details

## S2.1 Candidate Pool, Filtering, and Scores

The candidate pool contains $2 0 4 8 = 2 ^ { 1 1 }$ scrambled Sobol points mapped into module-specific linear or logarithmic input coordinates. The power-of-two size is compatible with the Sobol construction and bounds scoring cost; it was not selected by a reported pool-size sweep. The coordinates and bounds are specified in Section S3. Previously selected pool points are excluded from further acquisition. Structured bootstrap probes are constructed separately rather than sampled uniformly from the pool.

For scored acquisition, parseable hypotheses are calibrated to observed data using $\tilde { h } _ { i } ( x ) =$ $a _ { i } h _ { i } ( x ) + b _ { i }$ . With at least three finite pairs, both coeficients are fitted by least squares; with two pairs only a scale is fitted. The filter ranks candidates by root-mean-square residual divided by median absolute observed output, using a mean-absolute-output or unit fallback if that denominator is not positive and finite. Up to four candidates with normalized error at most 0.75 survive. If fewer than two meet the threshold, the two lowest-error candidates survive; if fewer than two fits are valid, filtering falls back to the original committee. Common nuisance symbols are initially evaluated as unit constants before this afine calibration. Consequently, the filter does not optimize every free parameter in an arbitrary equation.

Let $p _ { i j } = \tilde { h } _ { i } ( x _ { j } )$ denote a retained prediction on the currently available pool. Variance acquisition standardizes each hypothesis across pool points using its finite-prediction mean $\mu _ { i }$ and population standard deviation $\sigma _ { i } { \mathrm { : } }$

$$
z _ { i j } = ( p _ { i j } - \mu _ { i } ) / \sigma _ { i } , \qquad s _ { \mathrm { v a r } } ( x _ { j } ) = \frac { 1 } { | I _ { j } | } \sum _ { i \in I _ { j } } ( z _ { i j } - \bar { z } _ { j } ) ^ { 2 } ,\tag{S2}
$$

where $I _ { j }$ indexes finite standardized predictions and $\bar { z } _ { j }$ is their mean. Constant finite hypotheses are represented by zeros. At least two usable predictions are needed. This is variance across pool-standardized hypotheses, not variance divided by the mean output at an input point.

Pairwise acquisition centers each hypothesis by its finite-prediction median $m _ { i }$ and scales it by its interquartile range $q _ { i }$ , using its standard deviation if $q _ { i } = 0$ . A nonpositive fallback scale makes that hypothesis unusable. With normalized predictions $r _ { i j }$

$$
s _ { \mathrm { p a i r } } ( x _ { j } ) = \frac { 1 } { | P _ { j } | } \sum _ { ( i , k ) \in P _ { j } } | r _ { i j } - r _ { k j } | ,\tag{S3}
$$

where $P _ { j }$ contains finite pairs with $i < k$ . Thus pairwise acquisition uses normalized mean absolute disagreement rather than raw absolute output diferences.

Both scores are rank-transformed, penalized near input-domain boundaries, weighted by the fraction of finite predictions, and combined with diversity during batch selection. Section S2.2 gives the constants and selection formula. Random acquisition samples unused pool points without replacement and bypasses fit filtering and score regularization. It shares the bootstrap, diagnostics, exploration prompt, batch constraint, and fresh synthesis. It therefore compares acquisition configurations within a common protocol but is not a clean ablation of phase separation.

## S2.2 Regularization and Batch Diversity

Let $\rho _ { j } \in [ 0 , 1 ]$ be the ordinal rank of a finite raw score, with ranks assigned in ascending score order. Ties are assigned positions by the implementation’s sorting routine rather than averaged ranks. Let $u _ { j k }$ be candidate $j ^ { \prime } \mathrm { s }$ percentile position along coordinate $k ,$ and $\begin{array} { r } { e _ { j } = \operatorname* { m i n } _ { k } \operatorname* { m i n } ( u _ { j k } , 1 - u _ { j k } ) } \end{array}$ . The regularized score is

$$
\tilde { s } _ { j } = \rho _ { j } \left[ \mathrm { c l i p } \left( \frac { e _ { j } - 0 . 0 8 } { 0 . 5 - 0 . 0 8 } , 0 , 1 \right) \right] ^ { 1 . 5 } v _ { j } ,\tag{S4}
$$

where $v _ { j }$ is the fraction of finite hypothesis predictions. Rank power is one. Candidates within the highest-scoring quarter are eligible, with at least the batch size eligible when enough finite scores exist. The first selected point has highest normalized eligible score. Subsequent points maximize $0 . 6 5 s _ { j } ^ { \prime } + 0 . 3 5 \delta _ { j }$ , where $s _ { j } ^ { \prime }$ is min–max-normalized eligible score and $\delta _ { j }$ is distance to the nearest point already selected in that batch in percentile coordinates, divided by ${ \sqrt { d } } .$ This is a heuristic batch-diversity rule, not a joint information-gain optimizer. Random acquisition bypasses these operations.

## S3 Input Domains and Evaluation Grids

Table S1 lists the input distributions used for both candidate-pool mapping and numerical evaluation. The acquisition pool uses scrambled Sobol inputs; the evaluation grid uses an independently seeded NumPy PCG64 generator. Grid seeds are obtained from the first eight bytes of the SHA 256 digest of a version identifier and module name, interpreted as a little-endian unsigned integer. The 5000-point grid is reused across models, methods, and dificulties, while ground truth remains dificulty-specific. Full seeds, bounds, source hashes, and grid hashes are included in Archive S2. These generated test points are not sent to the agent.

Table S1: Input domains. L denotes log-uniform sampling and U uniform sampling. Units and parameter interpretation follow the source simulator; these are computational benchmark domains.  
Module Variables and distributions   
Gravity $m _ { 1 } , m _ { 2 } : \mathrm { L } [ 1 , 1 0 ^ { 3 } ] ; r : \mathrm { L } [ 1 , 1 0 ]$   
Coulomb $q _ { 1 } , q _ { 2 } , r : \mathrm { L } [ 1 0 ^ { - 1 } , 1 0 ]$   
Magnetic $I _ { 1 } , I _ { 2 } , r : \mathrm { L } \bar { [ 1 0 ^ { - 3 } , 1 0 ^ { - 1 } ] }$   
Fourier $k : \mathrm { L } [ 1 0 ^ { - 1 } , 1 0 ] ; A : \mathrm { L } [ \dot { 1 } 0 ^ { - 4 } , 1 0 ^ { - 2 } ] ; \Delta T : \mathrm { L } [ 1 0 , 1 0 ^ { 3 } ] ; d : \mathrm { L } [ 1 0 ^ { - 2 } , 1 ]$   
Snell $n _ { 1 } , n _ { 2 } : \mathrm { U } [ 1 , 1 . 5 ] ; \mathrm { a n g l e ~ i n ~ d e g r e e s : \ U [ 0 , 9 0 ] }$   
Decay $N _ { 0 } : \mathrm { L } [ 1 , \bar { 1 0 ^ { 2 } } ] ; \bar { \lambda } : \mathrm { L } [ 1 0 ^ { - 3 } , 1 0 ^ { - 1 } ] ; t : \mathrm { L } [ \bar { 1 0 ^ { - 2 } } , 1 0 ]$   
Harmonic $k : \mathrm { L } [ \bar { 1 0 ^ { 2 } } , 1 0 ^ { \bar { 4 } } ] ; m : \bar { \mathrm { L } } [ 1 0 ^ { - 1 } , 1 0 ] ; b : \mathrm { L } [ \bar { 1 0 ^ { - 2 } } , 1 ]$   
Malus $I _ { 0 } : \mathrm { L } [ 1 0 ^ { 2 } , 2 \times 1 0 ^ { 3 } ] ;$ angle in radians: $\mathrm { U } [ 1 0 ^ { - 6 } , \pi / 2 ]$   
Sound $\gamma : \mathrm { { U } } [ \bar { 1 } . 3 , 1 . 7 ] ; T : \mathrm { { L } } [ 1 0 , 1 0 ^ { 3 } ] ; M : \mathrm { { L } } [ 1 0 ^ { - \bar { 3 } } , 1 0 ^ { - 1 } ]$   
Hooke $x : \mathrm { L } [ 1 0 ^ { - 3 } , 1 ]$   
Bose–Einstein $\omega : \dot { \mathrm { L } [ 1 0 ^ { 8 } , 1 0 ^ { 1 0 } ] } ; T : \mathrm { L } [ 1 0 , 1 0 ^ { 3 } ]$   
Heat transfer $m : \dot { \mathrm { L } [ 1 0 ^ { - 3 } , 1 0 ^ { \dot { 3 } } ] } ; c : \dot { \mathrm { L } [ 1 0 ^ { 2 } , 1 0 ^ { \dot { 4 } } ] } ; \Delta T : \mathrm { L } [ 1 0 , 1 0 ^ { 3 } ]$

## S4 Extended Results and Sensitivity Analyses

## S4.1 Module-Level Recovery and Uncertainty

Each baseline row contains 24 rollouts. Each planner-family row averages all three acquisition variants with equal representation, giving 72 rollouts; it is neither a best-of-three selection nor an ensemble that combines their answers. SA and measurement means give every module equal weight because their cells are balanced. Finite-only RMSLE means can implicitly give modules diferent weights when submissions are non-finite, which motivates reporting coverage and all-trial curves as well.

For module-level comparisons, average repeats and planner variants within each module, then subtract the matching baseline. We report the twelve individual efect directions, their mean, a 95% percentile interval obtained by resampling the twelve module efects 10,000 times with seed zero, and the range after removing each module in turn. These are sensitivity summaries for a fixed benchmark suite. They do not estimate uncertainty over an independently sampled task population and do not resolve rollout-level uncertainty. A two-sided sign-flip enumeration is retained for the four medium-task SA contrasts as a secondary diagnostic, with its unadjusted values disclosed. We do not use trial-index paired tests or homogeneous-binomial confidence intervals as evidence of generalization. Sign-flip enumeration assumes independent, sign-symmetric module efects under the null. The study does not randomize whole-protocol labels at module level, so these values are sensitivity diagnostics rather than design-based randomization evidence.

On GPT-4.1-mini medium tasks, the planner families raise observed SA from 8.3% to 37.5% without an interpreter and from 16.7% to 40.3% with one. The average diferences remain positive when any one module is removed (Table S2). However, seven modules have zero family–baseline SA diference in each comparison. The interpreter-enabled comparison has four positive module efects and one negative efect, while the no-interpreter comparison has five positive efects. A positive mean does not describe uniform symbolic improvement.

On GPT-4.1, the observed gains are 12.5 points without an interpreter and 5.6 with one. The latter is concentrated in two modules, with ten ties. The four unadjusted sign-flip diagnostic values are 0.0625 and 0.125 for GPT-4.1-mini without and with an interpreter, and 0.5 for both GPT-4.1 contrasts. None is below 0.05 even before multiplicity adjustment. Some module-resampling intervals exclude zero, but those intervals are not inversions of the sign-flip calculation. We report the distinction rather than selecting the more favorable summary. More rollouts are needed to determine the stability of the underlying cell means.

Table S2: Medium-task Symbolic Accuracy (SA) diferences in percentage points $( \mathrm { p p } ) . \ + / - / 0$ counts modules with positive, negative, and tied planner–baseline efects. Resampling intervals reweight the twelve fixed modules; leave-one-out ranges remove one module at a time. Neither summarizes stochastic rollout uncertainty or population generalization.
<table><tr><td>Model</td><td>Tool access</td><td>∆ SA (pp)</td><td>Resampling interval</td><td> $+ / - / 0$ </td><td>Leave-one-out range</td></tr><tr><td>4.1-mini</td><td>No interpreter</td><td>+29.2</td><td> $[ + 8 . 3 , + 5 4 . 2 ]$ </td><td>5/0/7</td><td> $[ + 2 2 . 7 , + 3 1 . 8 ]$ </td></tr><tr><td>4.1-mini</td><td>Interpreter</td><td>+23.6</td><td> $[ + 2 . 8 , + 4 7 . 2 ]$ </td><td>4/1/7</td><td> $[ + 1 6 . 7 , + 2 7 . 3 ]$ </td></tr><tr><td>4.1</td><td>No interpreter</td><td>+12.5</td><td> $[ - 8 . 3 , + 3 3 . 3 ]$ </td><td>3/1/8</td><td> $[ + 4 . 5 , + 1 8 . 2 ]$ </td></tr><tr><td>4.1</td><td>Interpreter</td><td>+5.6</td><td> $[ + 0 . 0 , + 1 5 . 3 ]$ </td><td>2/0/10</td><td> $[ + 1 . 5 , + 6 . 1 ]$ </td></tr></table>

Figure S1 shows where symbolic credit occurs. Medium-task GPT-4.1-mini planners recover several altered algebraic laws, particularly gravity with an interpreter, magnetic force, Fourier conduction, and heat transfer. Seven modules remain unrecovered by any configuration at that dificulty. Across the entire collection, underdamped harmonic motion, Malus’s law, and Bose– Einstein occupation have no credited recoveries. GPT-4.1 produces some Snell recoveries, so the latter is not a universal failure across the two models.

<table><tr><td colspan="8">GPT-4.1-mini</td></tr><tr><td>gravity</td><td>0</td><td>0</td><td>0</td><td>50</td><td>100</td><td>100</td><td>100 100</td></tr><tr><td>Coulomb</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>magnetic</td><td>50</td><td>0</td><td>100 100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Fourier</td><td>50</td><td>50</td><td>100</td><td>100 100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>Snell</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>decay</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>harmonic</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Malus</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>sound</td><td>0</td><td>100</td><td>100</td><td>100 100</td><td>100</td><td>100</td><td>50</td></tr><tr><td>Hooke</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td>0</td></tr><tr><td>BE</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>heat</td><td>0 N</td><td>50 C</td><td>100 100 Pair N</td><td>100</td><td>100 Varc</td><td>100</td><td>100 C</td></tr><tr><td colspan="10">VarN PairCc Rand Base Base Rand N</td></tr></table>

<table><tr><td colspan="8">GPT-4.1</td></tr><tr><td>0</td><td>100</td><td>0</td><td>0</td><td>0</td><td>100</td><td>100</td><td>100</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>50</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr><tr><td>50</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>50</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>50</td><td>50</td><td>50</td><td>50</td><td>100</td><td>100</td><td>100</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>0</td><td>100 C</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100 C</td></tr><tr><td>Bae N Base</td><td>VarN</td><td>PairN</td><td></td><td>Rand N</td><td>Varc</td><td>Pairc</td><td>Rand</td></tr></table>

Figure S1: Medium-task Symbolic Accuracy percentages for each module and configuration. Each cell contains two rollouts, so a single changed outcome moves it by 50 points. N and C denote no interpreter and interpreter access; Var, Pair, and Rand denote acquisition rules. Colors encode the displayed percentages, not confidence.

## S4.2 Prediction-Domain Audit

The numerical metric is the benchmark’s magnitude-based RMSLE, replayed on shared grids. For trial t, let $G _ { t }$ contain the sampled inputs with finite ground-truth outputs and let $C _ { t } = 1$ only when the submitted function evaluates successfully and predicts finite outputs everywhere in this nonempty set. The coverage-sensitive pass rate is

$$
Q ( \tau ) = \frac { 1 } { n } \sum _ { t = 1 } ^ { n } \mathbf { 1 } \{ C _ { t } = 1 , \ R _ { t } \ \mathrm { f i n i t e } , \ R _ { t } \leq \tau \} .\tag{S5}
$$

All trials remain in the denominator. Completeness is assessed on the sampled finite-target domain, not between grid points, and does not check output sign. The thresholds are retrospective descriptive choices.

The replay reproduces all previously reported shared-grid scores. Fifteen of 576 trial scores are non-finite. In addition, fourteen submissions have finite RMSLE but omit predictions at some points where the target is finite. All fourteen are Snell-law submissions. For example, one GPT-4.1- mini easy baseline with an interpreter predicts on only 770 of 4259 finite-target points, yet receives a finite RMSLE of 0.957. A low or moderate reported score can therefore refer to a restricted subset of the evaluation domain.

This is partly distinct from invalid target points: the Snell truth is finite on 4259 easy and 4264 medium inputs out of 5000, reflecting its restricted domain. Our completeness check is relative to those finite-target points, not a demand to predict a real-valued output where the target itself is undefined. Fourteen finite-score submissions fail that additional check. Together with the fifteen non-finite-score outcomes, 29 submissions fail the complete-domain criterion. Two submissions also disagree with the target’s sign at one or more finite pairs, an error the magnitude metric does not penalize as a sign error.

Table S3: Coverage-sensitive medium-task results. Complete gives submissions defined on the whole sampled finite-target domain. The remaining columns give $1 0 0 Q ( \tau )$ : the percentage of all rollouts that are complete and have finite RMSLE at most τ. Non-finite and incomplete predictions remain in the denominator. Thresholds are retrospective descriptive choices.
<table><tr><td>Model</td><td>Configuration</td><td>Complete</td><td> $\tau = 0 . 0 1$ </td><td> $\tau = 0 . 1$ </td><td> $\tau = 1$ </td></tr><tr><td>4.1-mini</td><td>Baseline N</td><td>21/24</td><td>16.7</td><td>20.8</td><td>41.7</td></tr><tr><td>4.1-mini</td><td>Planner N</td><td>64/72</td><td>36.1</td><td>38.9</td><td>54.2</td></tr><tr><td>4.1-mini</td><td>Baseline C</td><td>22/24</td><td>25.0</td><td>25.0</td><td>50.0</td></tr><tr><td>4.1-mini</td><td>Planner C</td><td>72/72</td><td>48.6</td><td>59.7</td><td>98.6</td></tr><tr><td>4.1</td><td>Baseline N</td><td>22/24</td><td>33.3</td><td>33.3</td><td>54.2</td></tr><tr><td>4.1</td><td>Planner N</td><td>64/72</td><td>36.1</td><td>44.4</td><td>72.2</td></tr><tr><td>4.1</td><td>Baseline C</td><td>23/24</td><td>45.8</td><td>54.2</td><td>91.7</td></tr><tr><td>4.1</td><td>Planner C</td><td>72/72</td><td>55.6</td><td>66.7</td><td>98.6</td></tr></table>

The numerical-fit pattern remains favorable under this stricter accounting (Table S3). All 72 medium-task interpreter-enabled planner submissions have complete finite-target coverage in each model. $\mathrm { A t } ~ \tau = 0 . 1$ , their all-trial rates are 59.7% versus 25.0% for the GPT-4.1-mini baseline and 66.7% versus 54.2% for GPT-4.1. The gaps vary with the threshold, and are descriptive rather than claims about a uniquely appropriate tolerance. Complete coverage still does not imply a correct equation.

## S4.3 Complete Easy-Task Comparison

Table S4 reports every GPT-4.1-mini easy configuration using the same shared-grid replay and completeness criterion as the main paper. Every row retains all 24 rollouts for SA and measurement means. RMSLE means include only finite scores, whose counts are shown. No GPT-4.1 easy grid was collected. This table adds a presentation of existing records rather than new experiments or a selection of favorable variants.

Table S4: All GPT-4.1-mini easy-task configurations, 24 rollouts per row. Int. denotes interpreter access. Credited, Finite, and Complete are counts out of 24. Complete requires predictions throughout the sampled finite-target domain. RMSLE means exclude non-finite scores.
<table><tr><td>Configuration</td><td>Credited</td><td>SA (%)</td><td>RMSLE</td><td>Finite</td><td>Complete</td><td>Meas.</td></tr><tr><td>Baseline, no int.</td><td>5</td><td>20.8</td><td>1.405</td><td>24</td><td>24</td><td>43.2</td></tr><tr><td>Baseline, int.</td><td>13</td><td>54.2</td><td>1.046</td><td>24</td><td>22</td><td>20.8</td></tr><tr><td>Variance, no int.</td><td>10</td><td>41.7</td><td>1.105</td><td>24</td><td>24</td><td>9.5</td></tr><tr><td>Pairwise, no int.</td><td>12</td><td>50.0</td><td>1.109</td><td>23</td><td>23</td><td>9.3</td></tr><tr><td>Random, no int.</td><td>12</td><td>50.0</td><td>1.279</td><td>24</td><td>24</td><td>9.5</td></tr><tr><td>Variance, int.</td><td>11</td><td>45.8</td><td>0.392</td><td>23</td><td>23</td><td>8.0</td></tr><tr><td>Pairwise, int.</td><td>14</td><td>58.3</td><td>0.381</td><td>24</td><td>23</td><td>8.2</td></tr><tr><td>Random, int.</td><td>14</td><td>58.3</td><td>0.102</td><td>24</td><td>24</td><td>9.3</td></tr></table>

## S4.4 Weak Dependencies and Numerical Fit

An instructive harmonic submission uses $\sqrt { k / m - ( b / ( 2 m ) ) ^ { 2 } }$ where the target is $\sqrt { k / m - b / ( 2 m ^ { 2 } ) }$ The damping dependencies difer algebraically, but their numerical efect is weak across much of the sampled domain. A recorded submission consequently has RMSLE below $1 0 ^ { - 3 }$ while receiving no symbolic credit. This is evidence of metric insensitivity to a weak dependency, not evidence that the judge rejected an equivalent rearrangement. Automatic power-law diagnostics can help expose multiplicative structure, but a good fit to that structure cannot certify additive, trigonometric, or reciprocal-exponential dependencies.

Sign disagreement is a separate failure mode. Both audited sign-disagreeing submissions have complete sampled domains: the GPT-4.1 medium Bose–Einstein variance planner with an interpreter, trial 0, and the GPT-4.1-mini easy decay random planner with an interpreter, trial 1. Their magnitude RMSLE values are approximately 0.0512 and 0.0970, respectively. Both therefore pass the coverage-sensitive threshold of 0.1 despite wrong signs at some inputs. These outcomes are distinct from the 29 incomplete submissions and remain in all reported analyses.

## S5 Artifact Verification

The S2 code-and-data snapshot includes the original 43-test ofline suite, six additional coverage tests, and scripts to regenerate the new analysis. Source-data hashes are checked before and after law execution. The coverage audit instruments the numerical metric to count finite-target and finite-prediction pairs without changing the value returned by the metric. When evaluation fails before that calculation, the submission is recorded as incomplete. The raw judge outcome is never replaced by a coverage result. The supplementary validation additionally checks all 576 row identities, every regenerated RMSLE value against the earlier replay, figure/table inputs, and preservation of the source snapshot. Consult the archive README for executable commands and the distinction between original results, retrospective analyses, and prospective experiments that have not been run.

## S5.1 Portable Files and Ofline Commands

Archive S2 includes sanitized copies of recorded trial JSON, the flat trial table, original benchmark and planner code, deterministic replay, and the retrospective analyses. Copied paths and organization identifiers are replaced; submitted laws, recorded evaluations, measurement counts, and output-token counts are verified unchanged. A SHA-256 manifest covers the shipped files. Replaying records never invokes the acquisition agents or the symbolic judge. It executes the saved law functions locally against the source simulator’s ground truth.

From the extracted archive root, the executable commands are:

python3 -m unittest tests/test\_planner\_study.py \

paper/test\_shared\_grid\_rmsle.py \

paper\_make\_20260908/analysis/test\_audit.py

The three test paths belong to one command. Then run

python3 paper\_make\_20260908/analysis/audit.py

python3 paper\_make\_20260908/analysis/assets.py

The first script regenerates coverage, shared-grid, trace, and source-integrity records; the second regenerates figures and tables, including the complete easy-task comparison. All outputs stay inside the new manuscript directory. The six added tests cover partial NaN predictions, invalid target points, infinite predictions, sign changes, empty finite-target sets, and mismatched array shapes.

The arXiv source archive supplies the main manuscript and this supplementary material in a single document, together with their bibliography and graphics. It uses the standard LaTeX article class. The top-level file is main.tex; compile it with XeLaTeX and BibTeX, or with Tectonic. The concise main paper preserves the earlier numerical results; the coverage and leave-one-module-out analyses are retrospective. Measurement-cap matching, component ablations, and further rollouts are prospective studies and are not included as completed results.