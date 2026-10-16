# CLAIMSCOPE: Label-Blind Witness Verification for Mathematical Claim Auditing

Mushir Akhtar mushirakhtar.ml@gmail.com

## Abstract

Mathematical agents are commonly evaluated by answer accuracy or proof acceptance. We introduce CLAIMSCOPE, a 72-task benchmark for auditing proposed claims through a verdict, scope, counterexample, and repair. Tasks cover optimizerstate transport, finite-horizon affine recurrences, and exact 2 × 2 linear algebra. We evaluate Qwen3-8B with direct generation and four matched two-pass conditions, each starting from the same draft. A witness checker evaluates only the model’s submitted state, without consulting reference verdicts or supplying a replacement witness. Its feedback still conveys information about validity: a verified refutation establishes that the claim is false, and rejection reasons can suggest structural fixes. In this evaluation, verifier feedback yields 23/39 accepted counterexamples on invalid claims and 25/72 safe completions, compared with 15/39 and 20/72 for checker-free generic feedback. Strict completion remains 17/72; verdict accuracy changes from 47/72 to 46/72. The claim-grouped safe-completion difference is .094, with a bootstrap 95% interval of [.000, .198]. All seven task-level gains follow horizon/input-length diagnostics. A post-hoc schema-only correction of the direct outputs reaches the same 25 safe tasks without another model call. These results separate claim classification from executable evidence, while locating the observed gains in witness-structure correction rather than demonstrating broader reasoning improvement.

## 1 Introduction

Reliable mathematical agents need more than correct final answers. A derivation may agree with several examples yet fail as a universal statement: it may use the wrong quantifier, omit a singular case, or confuse equality of an update direction with equality of a complete transition. An audit must therefore determine not only whether a claim is valid, but also its justified scope, the reason for any failure, and an appropriate repair.

For example, transferring state between optimizers can preserve the gradient direction yet change the parameter update because the learning rates or weight-decay factors differ. A useful audit identifies the missing condition, exhibits a parameter state where the proposed identity fails, and specifies a correction. Such checks can support review of derivations and implementations; the small claims here model this local task, not an entire training pipeline.

Existing benchmarks emphasize complementary endpoints. MATH [Hendrycks et al., 2021] and GSM8K [Cobbe et al., 2021] score problem solutions, while miniF2F [Zheng et al., 2021], LeanDojo [Yang et al., 2023], ProofNet [Azerbayev et al., 2023], and AlphaGeometry [Trinh et al., 2024] target formalization or proof. COUNTERMATH asks for proof or disproof through counterexamples [Li et al., 2025], and MathDebugger diagnoses errors in synthetic questions and solutions [Liang et al., 2025].

In this work, we introduce CLAIMSCOPE for structured audits of already-stated local claims. Most closely, CRAFT searches numerically for counterexamples to machine-learning and optimization claims and presents a found witness to the model for revision [Liu et al., 2026]. Our checker instead validates the witness proposed by the model. It neither searches for nor supplies a replacement witness or repair. This distinction concerns who constructs the evidence, not whether the feedback conveys information about the answer. Recent systems combine language models with proof-assistant feedback or self-correction [Ren et al., 2025, Lin et al., 2025, Zhao et al., 2025]; our evaluation unit is a local claim and a submitted refutation rather than a complete proof.

We use label-blind operationally: the checker does not consult the stored reference verdict, repair, or diagnostic label when producing feedback. It is not verdict-independent. A verified counterexample logically implies an invalid claim, whereas a rejected witness establishes neither truth nor falsity. A diagnostic rejection can also reveal how to fix the witness. This setting is related to counterexample guided synthesis [Jha and Seshia, 2014], but checks the model’s proposed state instead of returning a new counterexample. We ask whether this feedback improves audits relative to matched revision without a checker.

We contribute (i) a 72-task benchmark with computational counterexample validation; (ii) a protocol for model-proposed witness checking with an explicit information boundary; and (iii) a paired analysis of classification, executable evidence, and complete structured audits. The evaluation includes constant policies, claim-grouped uncertainty, and a schema-only analysis to distinguish structural correction from stronger mathematical reasoning. The benchmark, checking and scoring code, and recorded outputs are available in the project repository.<sup>1</sup>

## 2 Benchmark and protocol

Tasks and reference validation. The benchmark contains 24 tasks per family: optimizer-state transport, affine recurrences, and linear algebra (Table 1). There are 33 valid and 39 invalid claims. Optimizer tasks vary learning-rate scaling, bias correction, stabilizer placement, decay semantics, and the distinction between direction and complete update. Recurrence tasks test finite-horizon unrolling, indexing, and the singular case $\rho = 1$ . Matrix tasks test determinant, inverse, trace, and transpose identities.

Reference validity rests on algebraic identities under the stated assumptions, not agreement on a finite diagnostic grid. For optimizer claims, symbolic substitution of the candidate transport into the update gives either an identically zero discrepancy or a refutable expression. Standard finite-sum and matrix identities establish the other valid cases. Every invalid claim has an admissible, computationally verified counterexample in the requested schema. Appendix A.1 gives the validation rules and explicit optimizer counterexamples. Numerical optimizer checks use relative tolerance $1 0 ^ { - 1 0 }$ and absolute tolerance $1 0 ^ { - 1 2 }$ ; recurrence and matrix checks use exact rational arithmetic.

Table 1: Benchmark composition. Claim groups collect prompts with the same formal specification, including assumptions and evaluation target.
<table><tr><td>Family</td><td>Tasks</td><td>Valid / invalid</td><td>Groups</td><td>Witness check</td></tr><tr><td>Optimizer transport</td><td>24</td><td>9/ 15</td><td>24</td><td>numerical updates</td></tr><tr><td>Affine recurrence</td><td>24</td><td>12 / 12</td><td>16</td><td>exact unrolling</td></tr><tr><td>Linear algebra</td><td>24</td><td>12 / 12</td><td>8</td><td>exact matrix identities</td></tr><tr><td>Total</td><td>72</td><td>33/39</td><td>48</td><td></td></tr></table>

Response and scope. Each prompt supplies the claim, assumptions, allowed labels, and witness schema, but no answer key. The response contains a task identifier, verdict, scope, failure mode, counterexample, and repair. Scope denotes the designated evaluation target—for example, an optimizer direction versus a complete transition—not a free-form proof of the strongest possible theorem. Repairs are identifiers for corrections defined in the prompt. Table 2 details their scoring. The following task illustrates the full audit.

```latex
Claim and assumptions. For $x _ { k + 1 } = \rho x _ { k } + u _ { k }$ with constant inputs, the finite-horizon value is
$\begin{array} { r } { \widetilde { x } _ { T } = \rho ^ { T } x _ { 0 } + \sum _ { k = 0 } ^ { T - 1 } \rho ^ { T - k } u _ { k } } \end{array}$
Audit. Invalid; scope: finite-horizon value; failure: input exponent.
Counterexample. $\rho = 2 , x _ { 0 } = 1 , T = 2 .$ , inputs [1, 1]. Iteration gives $x _ { 1 } = 3 , x _ { 2 } = 7 ;$ the claim gives
4 + 4 + 2 = 10.
$\begin{array} { r } { \mathbf { R e p a i r . \ U s e \ } x _ { T } = \rho ^ { T } x _ { 0 } + \sum _ { k = 0 } ^ { T - 1 } \rho ^ { T - 1 - k } u _ { k } . } \end{array}$
```

Scoring. Safe completion is a conjunction of structured-output validity, correct identifier and verdict, reference scope and repair, and the witness condition:

$$
\operatorname { S a f e } _ { i } ( y ) = \operatorname { J S O N } _ { i } ( y ) \wedge \operatorname { I D } _ { i } ( y ) \wedge \operatorname { V e r d i c t } _ { i } ( y ) \wedge \operatorname { S c o p e } _ { i } ( y ) \wedge \operatorname { R e p a i r } _ { i } ( y ) \wedge W _ { i } ( y ) .
$$

Here $W _ { i }$ requires a verified refutation for an invalid claim and a null witness for a valid claim. Strict completion additionally requires the reference failure-mode label. Scope, repair, and failuremode scores compare normalized labels, not semantic equivalence of arbitrary text. Thus safe completion means compliance with this audit specification; it is not a proof of every field’s unrestricted mathematical meaning. To avoid rewarding an always-null witness policy, we separately report accepted counterexamples among invalid claims and inappropriate non-null witnesses among valid claims.

Table 2: Interpretation of the response fields and their checks.
<table><tr><td>Field</td><td>Check and limitation</td></tr><tr><td>Verdict</td><td>Valid or invalid, compared with the algebraically audited reference.</td></tr><tr><td>Scope</td><td>Canonical target: optimizer direction or full transition, finite-horizon value, or matrix identity. No open-ended theorem-strength comparison.</td></tr><tr><td>Repair</td><td>Canonical correction named in the prompt, or none for a valid claim. Equivalent free-form formulas and alternative restrictions receive no automatic equivalence credit.</td></tr><tr><td>Failure mode</td><td>Canonical diagnostic label. Required for strict, but not safe, completion because different descriptions can identify the same defect.</td></tr><tr><td>Counterexample</td><td>For an invalid claim, parse the state, enforce assumptions, and recompute both expressions. Reported expected/actual values are not trusted. For a valid claim, require null.</td></tr></table>

Revision and feedback. Direct generation makes one call. Each revision condition makes the same draft call and one review call. Self-reflection requests a critical review; counterexample-guided revision asks the model to seek a refutation; generic feedback warns that the witness may be wrong while explicitly stating that no checker supplies a task-specific signal. Verifier feedback instead reports the result of checking the submitted state. These conditions compare tool-assisted revision with checker-free self-feedback [Madaan et al., 2023], rather than trained outcome or process supervision [Cobbe et al., 2021, Lightman et al., 2023].

The checker returns verified for an admissible refutation, rejected for a malformed, inadmissible, or non-refuting witness, and not applicable when the draft does not request a witness check. A rejection can name a validation error, including a mismatch between the horizon and number of inputs. The model receives the claim, its draft, the status and reason, and a fixed revision instruction. It receives no reference record, replacement witness, repair, or proof. Although the verifier loads hash-checked formal task sources separately from the model context, its feedback function uses no reference labels. Reproducing all 72 saved feedback messages with label-free records confirms this separation.

## 3 Experiment

Configuration and comparison. We evaluate Qwen3-8B [Yang et al., 2025] with 4-bit NormalFloat quantization on one NVIDIA RTX A4500, temperature zero, thinking disabled, and at most 512 new tokens per call. All five conditions use the same 72 prompts. Every two-pass draft is byte-identical to its direct counterpart. Each revision condition therefore uses 144 calls with the same token ceilings, against 72 calls for direct generation. The generic-feedback condition is the closest checker-free control in budget and review wording. Appendix A.6 records the model revision, runtime, and provenance. All analyses below use these saved outputs; the additional baselines require no model generation.

Claim-grouped result. We give equal weight to each of the 48 formal claim groups before averaging safe completion, because multiple prompts can express the same claim. The grouped score is .205 for generic feedback and .299 for verifier feedback, a difference of .094. A 10,000-sample paired bootstrap over groups gives a percentile 95% interval of [.000, .198]. Six groups improve, one declines, and 41 do not change. The interval includes zero, and groups can still share formula components: these point estimates do not establish a general improvement. Task-level counts in Table 3 provide the underlying descriptive results.

Table 3: Counts on the frozen task set. CE is counterexample-guided revision. Cex counts verified witnesses on 39 invalid claims (higher is better); Non-null counts inappropriate non-null witnesses on 33 valid claims (lower is better). The last three rows are deterministic, model-free policies or post-processing; they are not additional generation conditions.
<table><tr><td>Condition</td><td>JSON 172</td><td>Verdict 172</td><td>Cex /39</td><td>Non-null /33</td><td>Safe 1/72</td><td>Strict /72</td></tr><tr><td>Direct</td><td>72</td><td>46</td><td>15</td><td>26</td><td>18</td><td>15</td></tr><tr><td>Self-reflection</td><td>70</td><td>44</td><td>12</td><td>26</td><td>15</td><td>13</td></tr><tr><td>CE-guided</td><td>72</td><td>46</td><td>14</td><td>26</td><td>17</td><td>14</td></tr><tr><td>Generic feedback</td><td>71</td><td>47</td><td>15</td><td>24</td><td>20</td><td>17</td></tr><tr><td>Verifier feedback</td><td>71</td><td>46</td><td>23</td><td>25</td><td>25</td><td>17</td></tr><tr><td>Always valid, null witness</td><td>72</td><td>33</td><td>0</td><td>0</td><td>33</td><td>33</td></tr><tr><td>Always invalid, null witness</td><td>72</td><td>39</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Direct + schema-only rule</td><td>72</td><td>46</td><td>23</td><td>26</td><td>25</td><td>17</td></tr></table>

Evidence and classification. Generic-to-verifier safe completion changes from 20/72 to 25/72, with seven gains and two losses; strict completion remains 17/72. Accepted counterexamples on false claims increase from 15/39 to 23/39, but verdict accuracy decreases from 47/72 to 46/72. The direct draft calls all 39 invalid claims invalid, but also rejects 26 of 33 valid claims. Its classification bias therefore leaves substantial room for improving true-claim recognition rather than merely finding more refutations. Among invalid claims, safe completion increases from 11/39 under generic feedback to 18/39 with verification; among valid claims, it decreases from 9/33 to 7/33.

The constant policies expose an important endpoint limitation. An always-null witness satisfies the isolated witness condition on all 33 valid claims without constructing any evidence. An always-invalid verdict scores 39/72. More strongly, an always-valid policy that copies the prompt’s scope and identifier, emits null, and selects no repair or failure obtains 33/72 safe completions, exceeding every model condition. Valid claims do not require a proof in this benchmark. Safe completion therefore must be read together with class-conditional counterexample and classification scores, not as a standalone measure of reliable mathematical reasoning.

Where revision helps. Family-level safe counts for direct/generic/verifier feedback are 4/4/4 for optimizer transport, 1/3/8 for recurrences, and 13/13/13 for linear algebra, each out of 24. All seven gains over generic feedback follow a rejection identifying a horizon/input-length mismatch; none follows only a non-refutation message. These gains span six formal claim groups and four recurrence formulas. Of 27 verifier-condition revisions, 26 change only the counterexample field; the remaining revision produces no parseable audit. No parseable revision changes the verdict.

Schema-only analysis. To assess how much the witness representation explains, we apply one post-hoc rule to direct outputs: if a recurrence witness reports a horizon different from its input-list length, set the horizon to that length and leave every other field unchanged. The task quantifies over admissible horizons, so this selects a different witness rather than changing the claim. The rule uses only the public schema, no reference label or formula evaluation, and changes 21 records. It reaches the same 25 safe tasks as verifier feedback, including the same seven gains over direct generation. This is not a prospective baseline or a binary-feedback ablation. It shows that the observed safe-task set can be recovered by deterministic structural correction; the data do not require a stronger explanation in terms of mathematical search.

## 4 Discussion and limitations

The study isolates model-proposed witness validation from checker-driven counterexample search, but not validity feedback from diagnostic hints. A verified status entails falsity, and a specific schema error can suggest a correction. In the saved drafts, all 15 verified states belong to invalid claims; the 21 horizon/input-length rejections split into 12 valid and nine invalid claims, while 29 non-refuting states split into 14 valid and 15 invalid claims. Thus rejection is not a disguised reference verdict, but the channel is informative and should not be described as verdict-blind. A binary-only condition would be needed to isolate the value of status alone.

The benchmark is small, synthetic, and limited to one quantized model with a short deterministic output budget. Repeated formulas, even across distinct assumptions, limit independence. The results neither establish behavior on unseen mathematical families nor demonstrate open-ended theorem proving or optimizer performance in training. Numerical optimizer validation is tolerance-based, unlike the exact recurrence and matrix checks. Canonical scope and repair labels also restrict the endpoint: an alternative mathematically sound repair can fail the label comparison. Algebraic validation of reference claims, explicit class-conditional metrics, and the trivial-policy results are therefore essential to interpreting this testbed. Larger independent claim sets, semantic repair checking, and proof obligations for valid claims remain important directions.

## 5 Conclusion

CLAIMSCOPE evaluates structured audits of mathematical claims and separates verdict classification from executable counterexamples. In this fixed Qwen3-8B evaluation, diagnostic verifier feedback yields more accepted counterexamples and safe completions than matched generic feedback, without better verdict or strict-completion scores. Grouped uncertainty includes no improvement, and a schema-only rule recovers the same safe-task set. The clearest result is a witness-structure correction effect, not evidence that verification alone improves general mathematical reasoning.

## References

Zhangir Azerbayev et al. Proofnet: Autoformalizing and formally proving undergraduate-level mathematics. arXiv preprint arXiv:2302.12433, 2023.

Karl Cobbe et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Dan Hendrycks et al. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Susmit Jha and Sanjit A. Seshia. Are there good mistakes? a theoretical analysis of counterexampleguided inductive synthesis. In Computer Aided Verification, pages 484–501. Springer, 2014.

Yinghui Li et al. One example shown, many concepts known! counterexample-driven conceptual reasoning in mathematical LLMs. In Proceedings ofthe 42nd International Conference on Machine Learning, 2025.

Hao Liang et al. Mathdebugger: Detecting and diagnosing errors in synthetic mathematical data. arXiv preprint arXiv:2502.19058, 2025.

Hunter Lightman et al. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Yong Lin et al. Goedel-Prover-V2: Scaling formal theorem proving with scaffolded data synthesis and self-correction. arXiv preprint arXiv:2508.03613, 2025.

Yirui Liu, Jiayu Qin, Penghang Liu, and Jian Chen. CRAFT: Counterexample-guided repair and falsification of theorems in ML and optimization. In ICML 2026 Workshop: AI as a Tool for Mathematics, Computer Science, and Machine Learning, 2026.

Aman Madaan et al. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651, 2023.

Z. Z. Ren et al. DeepSeek-Prover-V2: Advancing formal mathematical reasoning via reinforcement learning for subgoal decomposition. arXiv preprint arXiv:2504.21801, 2025.

Trieu H. Trinh, Yuhuai Wu, Quoc V. Le, He He, and Thang Luong. Solving olympiad geometry without human demonstrations. Nature, 625:476–482, 2024.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Kaiyu Yang et al. Leandojo: Theorem proving with retrieval-augmented language models. arXiv preprint arXiv:2306.15626, 2023.

Xutong Zhao et al. Boosting LLM reasoning via spontaneous self-correction. arXiv preprint arXiv:2506.06923, 2025.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: A cross-system benchmark for formal olympiad-level mathematics. arXiv preprint arXiv:2109.00110, 2021.

## A Supplementary evaluation details

## A.1 Reference validation and admissible counterexamples

For an optimizer state, write

$$
Q = \sqrt { V / ( 1 - \beta _ { 2 } ^ { t } ) } + \epsilon , A = \frac { M } { ( 1 - \beta _ { 1 } ^ { t } ) Q } , W _ { \mathrm { M } } = ( 1 - \alpha \lambda _ { \mathrm { M } } ) W - \alpha D .
$$

The AdamW transition is $W _ { \mathrm { A } } = ( 1 - \eta \lambda _ { \mathrm { A } } ) W - \eta A$ . Its discrepancy from the reference equals

$$
W _ { \mathrm { A } } - W _ { \mathrm { M } } = ( \alpha \lambda _ { \mathrm { M } } - \eta \lambda _ { \mathrm { A } } ) W + \alpha D - \eta A .
$$

Substituting $M = ( 1 - \beta _ { 1 } ^ { t } ) ( \alpha / \eta ) D Q$ gives $A = ( \alpha / \eta ) D$ , hence equal step directions, but leaves the first decay term. Matching the full transition requires either matched decay or the transport

$$
M = ( 1 - \beta _ { 1 } ^ { t } ) \left[ \frac { \alpha } { \eta } D + \left( \frac { \alpha \lambda _ { \mathrm { M } } } { \eta } - \lambda _ { \mathrm { A } } \right) W \right] Q .
$$

For a direction claim, the comparison is $A = ( \alpha / \eta ) D$ , equivalently $\eta A = \alpha D$ . For protocols that explicitly use the reference decay, that decay replaces AdamW decay in the discrepancy.

The admissible optimizer domain enforces finite values, $V \geq 0$ , positive learning rates, $0 < \beta _ { 1 } < 1$ $0 \leq \beta _ { 2 } < 1$ , integer $t \geq 1 , \epsilon \geq 0$ , and $Q > 0$ , together with the prompt’s extra assumptions. The audit substitutes each candidate formula and its stated constraints into the appropriate discrepancy. All nine valid optimizer specifications reduce identically to zero; each of the other 15 has a verified admissible state. This universal-identity check is distinct from testing a finite grid.

The following three cases show why accidental numerical agreement is insufficient. Use $W = D =$ $V = 1 , \beta _ { 1 } = . 9 , \beta _ { 2 } = . 9 9 9 , t = 1 , \epsilon = 0 , \alpha = . 1 , \mathrm { a n d } \eta = . 0 1$

• PILOT-OPT-002: with $\lambda _ { \mathrm { M } } = . 1$ and $\lambda _ { \mathrm { A } } = . 0 1$ , the direction transport gives $A = 1 0$ . The complete updates are $W _ { \mathrm { A } } = . 8 9 9 9$ and $W _ { \mathrm { M } } = . 8 9$ . The stipulated $\epsilon = 0 , V > 0$ does not match decay. The reference verdict is invalid; the canonical failure is decay mismatch and the repair is full transport.

• PI $\mathtt { L O T - O P T - } 0 1 3 \colon$ choose $\lambda _ { \mathrm { M } } = . 0 1$ and $\lambda _ { \mathrm { A } } = . 1 , \mathrm { s o } \alpha \lambda _ { \mathrm { M } } = \eta \lambda _ { \mathrm { A } } = . 0 0 1$ . The candidate omits $\alpha / \eta ,$ giving $A = 1$ rather than 10. Thus $\eta A = . 0 1 \neq . 1 = \alpha D$ despite matched decay. The reference verdict is invalid; the failure is learning-rate scaling and the repair is direction transport.

• PILOT-OPT-024: with $\lambda _ { \mathrm { M } } ~ = ~ . 1$ and $\lambda _ { \mathrm { A } } ~ = ~ . 0 1$ , reversed decay compensation gives $A =$ $1 0 + ( . 0 1 - 1 ) = 9 . 0 1 $ and $W _ { \mathrm { A } } = . 9 0 9 8 \neq . 8 9 = W _ { \mathrm { M } }$ . The reference verdict is invalid; the failure is decay compensation and the repair uses the full transport with the sign shown above.

For recurrences, induction gives $\begin{array} { r } { x _ { T } = \rho ^ { T } x _ { 0 } + \sum _ { k = 0 } ^ { T - 1 } \rho ^ { T - 1 - k } u _ { k } . \mathrm { A t } \rho = 1 } \end{array}$ this becomes $x _ { 0 } + \sum _ { k }$ u<sub>k</sub>; for constant input u and $\rho \neq 1$ , the sum is $u ( 1 - \rho ^ { T } ) / ( 1 - \rho )$ . The valid specifications use these identities only under their required conditions or handle $\rho = 1$ piecewise. For matrices, the valid specifications are the reversed-order transpose, cyclic trace, determinant product, and reversed-order inverse identities; inverse checks require the relevant matrices to be invertible. Invalid specifications in both families have exact rational witnesses. Across all families, the 39 invalid claims admit witnesses in the published schema; representability therefore does not impose a lower ceiling.

## A.2 Recorded worked example

The main-text example is PILOT-REC-023. Its shared draft labels the input-exponent claim invalid and selects the exact-unroll repair, but reports horizon 2 with only one input, [1]. The checker returns rejected with the local reason horizon must equal the number of supplied inputs. This diagnostic does not specify which field to change, but it directly identifies the inconsistent pair of fields. The model changes the input list to [1,1], retaining $\rho = 2 , x _ { 0 } = 1$ , and horizon 2. The resulting $7 \neq 1 0$ comparison verifies the refutation. The other audit fields do not change. In the schema-only analysis, the alternative change to horizon 1 also refutes this claim: iteration gives $^ { 3 , }$ while the candidate gives 4. Thus the model’s final witness is not the only admissible structural correction.

## A.3 Scoring contract

The scorer extracts the final parseable audit, compares its task identifier with the originating task, and checks the five required audit fields. Additional JSON keys are not rejected by this implementation. A misspelled reported identifier receives zero identifier credit without changing which benchmark task is scored. Scope, repair, verdict, and failure labels are stripped of surrounding whitespace and lowercased before comparison. The score does not trim spaces inside JSON key names or infer missing fields.

Consequently, reference agreement for scope and repair should not be read as unrestricted semantic assessment. A verified witness does not certify the repair, and the chosen canonical repair need not be the only mathematical correction. Optimizer witnesses use scalar states, recurrences use rational initial state and input lists, and matrix witnesses use rational 2 × 2 arrays. Reported numerical values or diagnostic prose cannot substitute for the state.

The scorer’s isolated witness flag treats a missing witness field like null on valid claims. Safe completion still requires parseable structured output, so this behavior cannot make an empty response safe. We avoid interpreting that aggregate flag as evidence quality: Table 3 instead reports verified witnesses on invalid claims and non-null witnesses on valid claims. For verifier feedback, seven valid-claim responses explicitly contain null, 25 contain non-null witnesses, and one has no parseable audit. For generic feedback these counts are nine, 24, and zero. An always-null policy satisfies the isolated witness requirement on 33/72 tasks, but verifies no invalid claim.

Table 4: Safe completion separated by reference verdict and family.
<table><tr><td>Condition</td><td>Invalid /39</td><td>Valid /33</td><td>Optimizer /24</td><td>Recurrence /24</td><td>Matrix /24</td></tr><tr><td>Direct</td><td>11</td><td>7</td><td>4</td><td>1</td><td>13</td></tr><tr><td>Self-reflection</td><td>8</td><td>7</td><td>3</td><td>1</td><td>11</td></tr><tr><td>CE-guided</td><td>10</td><td>7</td><td>4</td><td>1</td><td>12</td></tr><tr><td>Generic feedback</td><td>11</td><td>9</td><td>4</td><td>3</td><td>13</td></tr><tr><td>Verifier feedback</td><td>18</td><td>7</td><td>4</td><td>8</td><td>13</td></tr></table>

## A.4 Feedback information and revision traces

A verified status is logically sufficient for an invalid verdict. An unparseable draft receives an unavailable status; this branch does not occur in the 72 shared drafts. The checker never draws the explicit reference label, repair, failure mode, or stored witness into the review message. All saved feedback messages are reproducible after removing those reference fields and retaining only the task specification. This implementation-level separation does not imply statistical independence: Table 5 quantifies the actual relationship with validity. In particular, the not-applicable cases already contain valid draft verdicts with null witnesses; the checker does not itself certify those claims.

Table 5: Draft feedback stratified by audited reference verdict. Non-refutation reports that the submitted state does not disprove the claim under its assumptions.
<table><tr><td>Feedback</td><td>Valid claim</td><td>Invalid claim</td><td>Total</td></tr><tr><td>Verified refutation</td><td>0</td><td>15</td><td>15</td></tr><tr><td>Rejected: horizon/input-length mismatch</td><td>12</td><td>9</td><td>21</td></tr><tr><td>Rejected: non-refutation</td><td>14</td><td>15</td><td>29</td></tr><tr><td>Not applicable</td><td>7</td><td>0</td><td>7</td></tr></table>

Among optimizer, recurrence, and matrix drafts respectively, the verified/rejected/not-applicable counts are 8/16/0, 1/23/0, and 6/11/7. Final predictions change on 1/24, 21/24, and 5/24 tasks. The optimizer change is a parse failure; the other 26 changes affect only the witness. Therefore no well-formed final revision changes a draft verdict. All 39 truly invalid claims already have invalid draft verdicts. Among these, accepted witnesses increase from 15 to 23. Among the 26 valid claims incorrectly called invalid in the draft, no verifier-condition response corrects the verdict.

The seven safe gains over generic feedback are PILOT-REC-002, 005, 007, 009, 010, 012, and 023. All follow horizon/input-length rejection and cover input-exponent, initial-exponent, ρ = 1 linear, and constant-input geometric formulas. The two safe losses, PILOT-REC-004 and 017, are valid piecewise-geometric claims that generic feedback accepts but verifier feedback continues to reject. No gain over generic feedback follows only a non-refutation message. This is descriptive attribution to message types, not an experiment separating the effects of diagnosis and status.

Table 6: Verdict confusion: reference class → reported class. V and I denote valid and invalid; M denotes missing or unparseable verdict.
<table><tr><td>Condition</td><td>V→V</td><td>V→I</td><td>V→M</td><td>I→V</td><td>I→I</td><td>I→M</td></tr><tr><td>Direct</td><td>7</td><td>26</td><td>0</td><td>0</td><td>39</td><td>0</td></tr><tr><td>Self-reflection</td><td>7</td><td>26</td><td>0</td><td>0</td><td>37</td><td>2</td></tr><tr><td>CE-guided</td><td>7</td><td>26</td><td>0</td><td>0</td><td>39</td><td>0</td></tr><tr><td>Generic feedback</td><td>9</td><td>24</td><td>0</td><td>1</td><td>38</td><td>0</td></tr><tr><td>Verifier feedback</td><td>7</td><td>25</td><td>1</td><td>0</td><td>39</td><td>0</td></tr></table>

## A.5 Paired and grouped results

Table 7: Task transitions from generic feedback to verifier feedback. The witness row includes only the 39 invalid claims; other rows include all 72.
<table><tr><td>Metric</td><td>0 → 0</td><td>0 → 1</td><td>1 → 0</td><td>1 → 1</td></tr><tr><td>Verified invalid-claim witness</td><td>16</td><td>8</td><td>0</td><td>15</td></tr><tr><td>Safe completion</td><td>45</td><td>7</td><td>2</td><td>18</td></tr><tr><td>Strict completion</td><td>53</td><td>2</td><td>2</td><td>15</td></tr><tr><td>Verdict correct</td><td>24</td><td>1</td><td>2</td><td>45</td></tr></table>

A claim group is the serialized formal specification with task identifiers, surface wording, and reference outputs excluded. The 48 groups comprise 24 optimizer specifications, 16 recurrence specifications, and eight matrix identities. Each matrix identity has three wordings. Different groups can still share formulas and differ only in assumptions or evaluation target, so grouping does not establish independence. Within each group we average prompt scores; the reported overall grouped score is the unweighted mean of these group averages.

Table 8: Equal-weight means over 48 formal claim groups.
<table><tr><td>Condition</td><td>Verdict</td><td>Safe</td><td>Strict</td></tr><tr><td>Direct</td><td>.632</td><td>.184</td><td>.122</td></tr><tr><td>Self-reflection</td><td>.590</td><td>.149</td><td>.108</td></tr><tr><td>CE-guided</td><td>.632</td><td>.177</td><td>.115</td></tr><tr><td>Generic feedback</td><td>.642</td><td>.205</td><td>.142</td></tr><tr><td>Verifier feedback</td><td>.632</td><td>.299</td><td>.153</td></tr><tr><td>Always valid</td><td>.417</td><td>.417</td><td>.417</td></tr><tr><td>Always invalid</td><td>.583</td><td>.000</td><td>.000</td></tr><tr><td>Direct + schema-only rule</td><td>.632</td><td>.299</td><td>.153</td></tr></table>

For the paired safe difference, we sample 48 groups with replacement 10,000 times using seed 20260906, retaining generic and verifier outcomes together. The percentile interval is [.000, .198] around a mean difference of .09375. Six groups improve, one declines, and 41 remain unchanged. No independent-task significance claim follows from these data. The schema-only rule is chosen after inspecting the saved feedback, not prespecified. It modifies horizon to match input-list length without running a checker or consulting a label. It produces the same set of 25 safe task identifiers as verification, although the final witnesses need not be equal.

## A.6 Configuration, cost, and reproducibility

The task file is unchanged throughout these comparisons. Its SHA-256 hash is

Table 9: Inference configuration shared by the five generation conditions.
<table><tr><td>Component Setting</td><td></td></tr><tr><td>Model</td><td>Qwen/Qwen3-8B</td></tr><tr><td>Revision</td><td>b968826d9c46dd6066d109eabc6255188de91218</td></tr><tr><td>Quantization</td><td>4-bit NormalFloat; model thinking mode off</td></tr><tr><td>Decoding</td><td>Temperature 0; at most 512 new tokens per call</td></tr><tr><td>Hardware</td><td>One NVIDIA RTX A4500</td></tr><tr><td>Runtime</td><td>PyTorch 2.12.1+cu126; Transformers 5.16.1; CUDA 12.6</td></tr><tr><td>Draft control</td><td>Byte-identical direct and first-pass outputs for all 72 tasks</td></tr><tr><td>Integrity</td><td>Public task-file hash and reference-file hash checked separately</td></tr></table>

Table 10: Recorded inference cost. Seconds are summed host-measured generation wall times, not GPU-kernel time; model loading is excluded.
<table><tr><td>Condition</td><td>Model calls</td><td>Output tokens</td><td>Seconds</td></tr><tr><td>Direct</td><td>72</td><td>8,719</td><td>853.9</td></tr><tr><td>Self-reflection</td><td>144</td><td>17,457</td><td>1,757.5</td></tr><tr><td>CE-guided</td><td>144</td><td>17,418</td><td>1,744.9</td></tr><tr><td>Generic feedback</td><td>144</td><td>17,367</td><td>1,772.3</td></tr><tr><td>Verifier feedback</td><td>144</td><td>17,436</td><td>1,672.4</td></tr></table>

Hash verification establishes artifact identity, not mathematical correctness. The reference-validation analysis explicitly supplies invalid labels and witnesses for PILOT-OPT-002, 013, and 024; the public prompts and recorded responses remain fixed. The separate audited reference contains 33 valid and 39 invalid claims. Feedback reproduction with label-free records confirms that rescoring does not alter the messages the model actually receives.

The accompanying scripts/reanalyze.py script reads the saved five-condition outputs, checks shared drafts, audits reference specifications, and writes per-task scores, class-conditional counts, grouped comparisons, deterministic baselines, and a provenance file. It takes an optional –artifact-root and performs no model inference. It records the input, scorer, and analysisscript hashes. The original reference is preserved alongside the audited reference so that the label changes and every aggregate can be traced to individual tasks. The computational checks support this finite benchmark; they do not constitute a general proof system for arbitrary mathematical text.