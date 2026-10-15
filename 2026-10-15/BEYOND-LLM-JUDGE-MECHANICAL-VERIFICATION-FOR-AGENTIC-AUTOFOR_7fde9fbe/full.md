# BEYOND LLM JUDGE: MECHANICAL VERIFICATION FOR AGENTIC AUTOFORMALIZATION

Anna Mei Harvard University amei@g.harvard.edu

Michael R. Douglas CMSA, Harvard University and AletheAI, Inc mdouglas@cmsa.fas.harvard.edu

## ABSTRACT

We introduce a novel method for verifying Lean formalizations that goes beyond LLM-judged faithfulness. Unlike most verification methods, which rely on LLMs or trained models, our method combines LLM-based semantic review with mechanical numerical and vacuity checks, providing checkable feedback and detecting errors missed by LLM judges. We apply this method, along with our Extract-Verify-Proof autoformalization pipeline, to construct the Formalized Library of Mathematical Functions (FLMF), a partial formalization of the NIST Digital Library of Mathematical Functions (DLMF). On three labeled datasets, our method achieves the lowest accepted-error rate among the evaluated methods, measured as the fraction of incorrect statements accepted by the verification method. Expert review of the results also identifies translation and source defects in existing datasets. Code available at https://github.com/annamei1/ FLMF-Ch7ErrFunc-Release.git.

## 1 INTRODUCTION

With the rapid development of agentic coding assistants, the formalization of mathematics has become very accessible to researchers and has received a lot of attention. However, to this day, the most trusted and widely used library is still Mathlib (The mathlib Community, 2020), whose correctness is guaranteed by careful human review and which is therefore trusted by its downstream users. For the same reason, it also faces a slow review process and limited coverage of existing mathematics, blocking the formalization of more advanced mathematical work. Efforts have been made to resolve this problem by building agentic systems to formalize textbooks, research blueprints, and dependencyguided libraries at increasing scale (Gloeckle et al., 2026; Rammal et al., 2026; Wang et al., 2026; Cabral et al., 2026; Milikic et al., 2026; Douglas et al., 2026).

In contrast to Mathlib, autoformalizations of this kind face severe challenges to their reliability. An autoformalizer can change a domain, index, sign, or branch convention, or introduce contradictory hypotheses, and still produce a provable statement. Audits of formal benchmarks document such defects (An et al., 2026).

We therefore propose a verification method that combines numerical verification, an LLM check, and a vacuity check, along with an agentic autoformalization pipeline, to allow the creation of trustworthy reusable formal libraries. We chose the Digital Library of Mathematical Functions (DLMF) as our testing ground, and propose the Formalized Library of Mathematical Functions (FLMF) as the product of our pipeline and verification method.

This paper makes three contributions:

• Numerical and vacuity verification method (Section 3). We introduce numerical and vacuity checks on Lean statements, which are mechanical and provide checkable feedback on points of failure. As indicated by their names, a numerical check verifies numerical validity of a Lean statement, and a vacuity check looks for vacuous formalization, for example a tautological definition. So far, all verification in autoformalization relies on LLM judgements, and our method complements the ambiguity and arbitrariness of an LLM judge’s verdicts. Its feedback is tractable and informative, and passing these checks adds another layer of certainty about the correctness of a Lean statement.

• Autoformalization pipeline (Section 4). Our pipeline divides the formalization process into three stages: extract, verify, and prove. We emphasize human gating at each stage, requiring every item either to pass a gate or to be documented with a reason for failure. We then use an auditor agent to revise those documents against written guidelines. We also keep track of all statement corrections to allow future studies of them.

• Corrections to existing datasets (Section 5). Expert review of our verification flags identifies multiple defects in existing datasets. Among the statements that are labeled correct, we found 4 defects among ProofNet items labeled faithful by Ground False, 61 defects among correct-labeled Beyond Compilation items, and 224 defects in RLM25 (Table 3). We will release corrected statements for confirmed defects.

## 2 RELATED WORK

Evaluating formalization faithfulness. Some recent works use reference-based metrics to evaluate the faithfulness of formalization: BEq and its variants check bidirectional provable equivalence (Liu et al., 2025c; Poiroux et al., 2025a), GTED compares operator trees (Liu et al., 2025b), and ASSESS combines semantic and structural similarity (Liu et al., 2026). These metrics need a reference formal statement, which is a limitation when it comes to the formalization of natural language statements. On the other hand, reference-free methods score the informal–formal pair directly. FormalAlign trains a model to predict alignment from sequence certainty and representation similarity (Lu et al., 2025a). LeanScorer breaks a statement into components and has an LLM judge each one (Xuejun et al., 2025). Beyond Compilation evaluates LLM judges for faithful natural-language-to-Lean translation (Zhang et al., 2026). FormalMATH uses back-translation followed by LLM judgments to filter formalizations that do not preserve the original problem (Yu et al., 2025). As a result, Poiroux et al. show that automatic metrics disagree with humans on research-level statements and release RLM25 (Poiroux et al., 2025a). Ground False audits formal benchmarks for kernel-checked but wrong targets, classifies each ProofNet statement’s relation to its informal source, and releases both that taxonomy and a corrected ProofNet-Verified (An et al., 2026). In this paper, we compare our verification method against them using formal–informal pair datasets.

Agentic formalization. Agentic formalization extends single statement translation to the construction of formal libraries. Gloeckle et al. coordinate parallel agents on a shared code base to formalize a graduate textbook (Gloeckle et al., 2026), while AutoformBot combines dependency-aware scheduling and collaborative version control to build libraries across multiple textbooks (Rammal et al., 2026). At the workflow level, LeanFlow studies how proof workflows and toolsets affect the completion and efficiency of formalizing entire papers (Milikic et al., 2026); Requena Pozo et al. combine iterative proof refinement, library search, and context management in a minimal agent (Requena Pozo et al., 2026); and OpenGauss provides an open-source Lean autoformalization harness (Math, Inc., 2026). FormalScience incorporates domain-expert feedback into an agentic pipeline for formalizing scientific reasoning (Meadows et al., 2026). These systems motivate our emphasis on statement verification and human gates within a larger formalization workflow.

## 3 FAITHFULNESS VERIFICATION

Our verification method consists of three components: a semantic (LLM) check, a numerical check, and a vacuity check. The first two evaluate the Lean theorem statement without the proof body. The vacuity check runs after the proof is constructed and determines if the proof was vacuous. No single check establishes faithfulness, and we treat verification as a stack in which each layer has a distinct failure model and any substantive mismatch blocks automatic acceptance. The following discussion is based on our formalization of DLMF, but a large part of it is applicable to other large-scale formalizations of natural language math. For each theorem, we attached a verification tag, which is unverified by default and turns to verified after passing the checks.

![](images/296321498a70047ff0475121578cdc76f1540607eac32bf0acd19a46a7337829.jpg)  
Figure 1: The verification workflow. A theorem is verified if it receives a semantic FAITHFUL and no numerical or vacuity FAIL.

## 3.1 SEMANTIC CHECK

The semantic checks consist of a direct LLM inspection and a back-translation check. The direct inspection passes both the Lean statement and the informal statement to an LLM judge, and the judge returns a verdict about the faithfulness of the Lean translation.

The back-translation check consists of two steps. In the first step, an agent (translator) reads the Lean theorem statement, along with all definitions involved, without comments, and translates it into plain English. Masking the comments reduces the risk of hallucination on the agent’s side, forcing it to adhere to the exact Lean code. The translation is formatted to include the statement, variables, constraints, and symbols.

As an example, the translation of §DLMF5.5.3 Γ(s)Γ(1 − s) = π/ sin(πs) would be:

STATEMENT: For any complex number z that is not an integer,   
Γ(z)Γ(1 − z) = π/ sin(πz).   
VARIABLES: z: A complex number, z ∈ C.   
CONSTRAINTS: z ̸= m for every m ∈ Z.   
SYMBOLS: Gamma - the complex Gamma function. Complex.sim - the complex sine   
function. ↑ π - the mathematical constant π, cast to a complex number.

A second agent (judge) reads this English rendering and compares it with the equation extracted from the DLMF source.<sup>1</sup> In addition, we supply a STANDING\_CONSTRAINTS.md document to capture the constraints stated earlier in the source that may be missed by the equation parser.

The direct inspection may be the most intuitive thing to do, but having a back-translation check guarantees that the Lean statement is accurate enough to preserve its meaning after a back-translation. Having two separate agents working independently also avoids the failure mode in which an agent recognizes an inconsistency but chooses to ignore it. The verdicts from the judges fall into three categories: FAITHFUL, DIVERGE, or UNCERTAIN. For the latter two, the judge also documents why it considers the translation unfaithful and waits for further action. When a theorem obtains a FAITHFUL verdict from both the direct inspection and the back-translation check, its verification status can change to verified.

## 3.2 NUMERICAL CHECK

When possible<sup>2</sup>, a stated theorem undergoes a numerical check in which a mechanical extractor translates the Lean theorem into evaluations in mpmathx, an open-source Python library for arbitraryprecision numerical computation and special functions. A special function is bound to its mpmath counterpart with the appropriate convention. The checker samples points from the stated domain and compares the function values on both sides of the identity. Exact identities use residual gates;

asymptotics use ratio or rate gates. Failure of the numerical check vetoes the verified status granted by the back-translation check.

One natural question is how to guarantee the correctness of the Lean-mpmath binding. This binding is trustworthy because we require that every special function is only defined once, and the binding is attached to the definition. When performing the numerical check, the program traces the definitions used in the Lean statement and uses the corresponding bindings. We can also observe how many times each definition/binding has been used independently for numerical checks, further confirming its reliability.

There are also limitations to the numerical check. As we are working with special functions in DLMF, we can only sample points in the domain and evaluate the functions at those points. A common failure is the inability to identify an incorrect domain for a functional equality. For example, the Lean statement may have a smaller domain than the actual statement, or it may fail to exclude isolated points from the domain. If those points are not selected by the numerical checker, it will fail to flag the error. Therefore, the numerical check should only be used as a partial check for correctness, not a complete faithfulness check.

## 3.3 VACUITY CHECK

A vacuity check scans the theorem statement and proof body to detect empty statements. Simple checks include testing whether trivial proof methods such as rfl or simp close the goal. In addition, we have a checker that scans for goal-independent proofs, which works by replacing the target goal with an opaque conclusion VacQ and checking whether the proof still compiles. If it does, then the proof is goal-independent, and the most likely cause is a set of contradictory hypotheses from which one can derive False<sup>3</sup>. If the checker flags a vacuous proof, the verified status is also reverted to unverified.

We would like to emphasize that none of these checks is perfect; they all have different domains of effectiveness and limitations, and the stack is undergoing active development as the project progresses. For example, the case that motivated the vacuity check was the original formalization of DLMF 7.13.5: its back-translation was judged faithful, yet proof review found that its completeness and ordering hypotheses for the Fresnel zeros were jointly contradictory. We also made these checks strict on purpose, so we would rather have them flag a correct statement than miss an incorrect one. (Section 6 gives more detail on this particular case.)

## 4 AUTOFORMALIZATION PIPELINE

The DLMF is organized in chapters and equations. Distinct chapters in the DLMF focus on different types of functions and rarely depend on each other; therefore, parallelization is quite feasible in FLMF. There are also special “methods” chapters, which describe mathematical techniques used to establish some of the identities. In these cases, we need to formalize, or at least state, the content of the methods before we can prove the related identities. These dependency relations are captured in a DAG (Directed Acyclic Graph) and are dynamically updated as formalization progresses. When the proving agent attempts to prove a theorem, it first looks it up in the DAG and reports whether infrastructure needed to construct the proof is missing.

This strategy is not unique to DLMF. Indeed, related works have used similar approaches (Wang et al., 2026; Cabral et al., 2026), and it plays a crucial role in the autoformalization of more sophisticated math objects that have complicated dependencies on each other.

In FLMF, we proceed with formalization in three stages: Extract, Verify, and Prove.

<table><tr><td>Stage</td><td>Agent work</td><td>Gate on every theorem / equation</td></tr><tr><td>DAG</td><td>Enumerate equations and dependencies</td><td>Open-world gap report</td></tr><tr><td>Extract</td><td>Write typed definitions and theorem statements</td><td>(i) Formalized with appropriate tag or (ii) documented in DISPOSITIONS.md</td></tr><tr><td>Verify</td><td>Run numerical and back-translation fidelity checks</td><td>(i) Obtained a verified stamp or (ii) documented in HUMAN_REVIEW.md</td></tr><tr><td>Prove</td><td>Fill frozen theorem bodies in waves</td><td>(i) Proved with clean build and 0 sorry or (ii) documented in UNPROVEN . md</td></tr></table>

Table 1: Autoformalization pipeline for FLMF.

## 4.1 EXTRACT

At this stage, an agent extracts all listed equations in a chapter by their equation numbers. The agent then writes each equation in the DAG and establishes its dependencies on existing nodes. The agent identifies all definitions needed to state the extracted equations, for example, the definitions of Bessel functions or theta functions. The definitions are only written once to guarantee consistency in downstream usage. After that, parallel agents are deployed to write Lean statements using these definitions. A complete proof is not required. Each extracted theorem carries 4 tags: DLMF equation number, Fidelity check method, Verification status, and its corresponding DAG node. The Extract workflow is gated on completion: every listed equation in the DLMF has to either (i) appear in FLMF as a theorem with appropriate tags or (ii) be documented in DISPOSITIONS.md with an explanation of why it is not stated. This is an example of a formalized statement:

/- <sub>\*\*</sub>DLMF 5.5.3<sub>\*\*</sub> -- Γ(s)Γ(1 − s) = π/ sin(πs) (complex)   
(DLMF-faithful restriction of ‘complex\_reflection’:s ̸= 0, ±1, ±2, . . .).   
DAG: E-5.5.3.   
Fidelity: numeric PASS   
Verified: verified   
-/   
theorem complex\_reflection\_dlmf   
(s : C)   
(\_hs : ∀ m : Z, s ̸= (m : C)) :   
Complex.Gamma s <sub>\*</sub> Complex.Gamma (1 - s) =   
(Real.pi : C) /   
Complex.sin ((Real.pi : C) <sub>\*</sub> s) := by   
sorry

## 4.2 VERIFY

When a theorem is first stated, its verification status is unverified by default. A successful verification check, which is the combination of semantic, numerical, and vacuity checks discussed in Section 3, allows a stamper to set this tag to verified. If a theorem fails a verification check, an agent analyzes the reason for failure. For the easy fixes, e.g., missing premises, a wrong sign, or incompatible conventions, an agent fixes the theorem and reruns the verification check; otherwise it is directed to human review. This stage is gated by requiring every equation to have either (i) a verified stamp or (ii) a documented entry in HUMAN\_REVIEW.md. If a statement does get corrected, it will be logged in an ERRATA file, keeping track of the mistakes made by agents and allowing further error analysis.

## 4.3 PROVE

After the Extract and Verify stages, we have the full list of theorems that are faithful but await to be proved. An agent surveys all open sorrys and proposes a proof plan. It contains a methods component, consisting of general proof strategies that apply to many identities, and waves of proofs that resolve the remaining theorems in dependency order. This step also observes and modifies the DAG dependency structure. The proving stage is gated on either (i) a sorry-free proof body or (ii) documentation in UNPROVEN.md.

## 4.4 AUDIT

Beyond the pipeline, we have auditor agents to inspect the human review documents at each stage (dispositions, human\_review, unproven, etc.). Very often, these documents contain a mix of simple mistakes, such as a wrong convention, and serious issues, such as a hidden assumption that is not stated in the informal text. The auditors are prompted with the human evaluation guidelines and make simple fixes while providing analysis and suggestions for the harder issues that really need a human decision.

We would like to emphasize the need for gating and human interaction at each stage. Asking the agent to work alone risks two failure modes: the agent spends many tokens working on a single problem and gets stuck, or it secretly skips the work and does not report its true status. Human review is also crucial for design choices, such as definition conventions, or special cases in which implicit constraints or specifications are conveyed but not listed.

Having the auditor significantly reduces the work of a human expert. The human expert can focus on the hard and subtle cases in which we do not readily trust the model’s judgement, while the rest are taken care of autonomously. Having an auditor does not reduce the reliability of the output, since the edited statement will have to go through the verification check again before it can be stamped verified. On very special occasions, for example, one that involves a source misprint, a human can stamp a theorem verified despite it not passing the verification check, but that is only possible after careful review by the expert, and no autonomous agents can do that.

Combined with our verification method, the workflow allows autoformalization of large-scale libraries without losing credibility. As part of this work in progress, we were able to complete the formalization of about 10% of DLMF, comprising around 1,000 entries, in 2 man-months. Further details are available in Appendix A.

## 5 EXPERIMENTS

In the following experiments, we compare our verification method with other baseline methods across different datasets. We demonstrate that ours has the lowest rate of accepting an incorrect formalization, which is the goal of building a reliable autoformalization verification. We also report runs of our method on external datasets, presenting potential defects in them.

## 5.1 SETUP

Datasets. The experiment contains 1,810 informal–formal pairs from four datasets, and the datasets provide a correctness evaluation for each pair. FLMF is a selected set of 471 statements from the ongoing DLMF formalization, combining library statements, mutations<sup>4</sup>, historical errors, and source-misprint pairs. This dataset focuses on special function equalities. ProofNet uses the original release (Azerbayev et al., 2023), with Ground False’s faithfulness verdicts as labels (An et al., 2026). It contains 371 undergraduate math problems from analysis, algebra, and topology. Beyond Compilation is a set of 400 graduate-level math problems with semantic evaluations (Zhang et al., 2026). RLM25 extracts 617 informal–formal statement pairs from six research-level formalization projects across number theory, harmonic analysis, combinatorics, etc<sup>5</sup>(Poiroux et al., 2025a). Appendix D gives a more detailed breakdown of dataset construction process.

Methods and models. Ours is the union of the three checks as discussed in Section 3, where direct inspection uses GPT-5.5, and back-translation uses Gemini 3.1 Pro as the translator and GPT-5.5 as the judge. The baseline methods are GPT-5.5 and Opus 4.6 as direct judges, each called once per pair and once in joint batches of 16 (“joint”); LeanScorer (Xuejun et al., 2025), which uses LLMs (GPT-5.5) to assess how faithfully each component of a statement is formalized; and FormalAlign (Lu et al., 2025a), a trained model that evaluates alignment between informal and formal statements, at a certainty threshold of 0.3 (Appendix E gives more detail). Every method sees the same informal statement and formal context.

<table><tr><td></td><td colspan="2">FLMF (182 C / 289 I)</td><td colspan="2">Beyond Compilation (245 C / 106 I)</td><td colspan="2">ProofNet (163 C / 204 I)</td></tr><tr><td>Method</td><td>Incorrect / accepted (rate)</td><td>Flagged / correct</td><td>Incorrect / accepted (rate)</td><td>Flagged / correct</td><td>Incorrect / accepted (rate)</td><td>Flagged / correct</td></tr><tr><td>Ours</td><td>5/105 (4.8%)</td><td>45.1%</td><td>3/154 (1.9%)</td><td>38.4%</td><td>14/153 (9.2%)</td><td>14.7%</td></tr><tr><td>GPT-5.5</td><td>28/178 (15.7%)</td><td>17.6%</td><td>10/227 (4.4%)</td><td>11.4%</td><td>26/176 (14.8%)</td><td>8.0%</td></tr><tr><td>GPT-5.5 (joint)</td><td>45/203 (22.2%)</td><td>13.2%</td><td>13/232 (5.6%)</td><td>10.6%</td><td>38/194 (19.6%)</td><td>4.3%</td></tr><tr><td>Opus 4.6</td><td>72/232 (31.0%)</td><td>12.1%</td><td>24/261 (9.2%)</td><td>3.3%</td><td>44/206 (21.4%)</td><td>0.6%</td></tr><tr><td>Opus 4.6 (joint)</td><td>88/250 (35.2%)</td><td>11.0%</td><td>42/267 (15.7%)</td><td>8.2%</td><td>56/215 (26.0%)</td><td>2.5%</td></tr><tr><td>LeanScorer</td><td>10/93 (10.8%)</td><td>54.4%</td><td>7/175 (4.0%)</td><td>31.4%</td><td>17/135 (12.6%)</td><td>27.6%</td></tr><tr><td>FormalAlign (≤ 0.3)</td><td>32/76 (42.1%)</td><td>75.8%</td><td>49/139 (35.3%)</td><td>63.3%</td><td>16/20 (80.0%)</td><td>97.5%</td></tr></table>

Table 2: Accepted-error rate by method on the three labeled datasets. Incorrect / accepted counts the incorrect items among those a method did not flag. A lower accepted-error rate means fewer incorrect Lean formalizations after the check. Flagged / correct is the percentage of all items the method flags among those labeled “correct”. The counts under each dataset are its correct (C) and incorrect (I) items.

Method coverage. The semantic checks, direct LLM judges, LeanScorer, and FormalAlign apply to every informal–formal pair. However, the goal-independent part of the vacuity check requires a completed proof and has no coverage on the original ProofNet or Beyond Compilation. In these cases, the vacuity check only looks for trivial proof methods. Similarly, the numerical check requires a statement to be numerically evaluable, which is not the case for, for example, infinite sums statements.

Metric. We evaluate acceptance by a verification method against each dataset’s labels, which are categorized into “correct” and “incorrect”. Our primary metric is the accepted-error rate: among the items a verification method accepts (does not flag), the fraction that are labeled incorrect. This metric aligns with our goal of ensuring that a Lean statement passing the gate is correct. We also report the percentage of total statements flagged.

## 5.2 MAIN RESULTS

Table 2 reports the results of running all methods on the FLMF, Beyond Compilation, and ProofNet datasets. Our method’s accepted-error rates are the lowest reported rates on all three datasets. Admittedly, our flagging rate is towards the higher end among all methods, but false alarms are generally acceptable given the help of auditing agents. We note that the “incorrect” labels from the datasets are not always accurate. The 3 items labeled incorrect that our method accepted in Beyond Compilation turn out to all have source ambiguity or incorrect labeling, and we would actually label 2 of the 3 “correct” after expert review.

In Table 3, we present an audit using our verification method on the Beyond Compilation, ProofNet, and RLM25 datasets. The flags raised by the verification pipeline then go through expert review to identify translation defects, source defects, or false alarms. In each of them, we were able to identify a number of defective statements that were labeled “correct”. We will release the corrected dataset with this paper.

Notice that the flags raised by a numerical check have a very high rate of identifying real defects, and many of those defects were not identified by a semantic LLM check. In Beyond Compilation, all 41 statements flagged by the numerical check yet labeled “correct” turn out to contain real defects. In RLM25, 8 out of the 12 flagged statements contain real defects. The four RLM25 numerical false alarms are attributed to binding errors in adapting the checker to the project’s data structures.

## 6 CASE STUDIES

We present two case studies on formalization defects identified by our method. One is identified by the vacuity checker alone and not by any other method, which is a subtle formalization defect involving contradictory hypotheses. The second is a false Taylor bound from the Beyond Compilation dataset that was faithfully translated to Lean and flagged only by our numerical check and none others. We therefore argue that even if we can make sure an LLM translates an informal statement seemingly well, the statement itself can contain subtleties or defects that are not captured by the model, resulting in an incorrect formalization.

<table><tr><td>Dataset</td><td>Check</td><td>Flagged</td><td>Labeled incorrect</td><td>Translation defects</td><td>Source defects</td><td>False alarm</td></tr><tr><td>Beyond Compilation</td><td>Numerical</td><td>76</td><td>35</td><td>8</td><td>37</td><td>0</td></tr><tr><td></td><td>Semantic</td><td>151</td><td>101</td><td>15</td><td>12</td><td>25</td></tr><tr><td></td><td>Vacuity</td><td>23</td><td>12</td><td>1</td><td>0</td><td>10</td></tr><tr><td></td><td>Combined</td><td>195</td><td>102</td><td>18</td><td>43</td><td>35</td></tr><tr><td>ProofNet</td><td>Numerical</td><td>85</td><td>85</td><td>0</td><td>0</td><td>0</td></tr><tr><td></td><td>Semantic</td><td>203</td><td>183</td><td>1</td><td>1</td><td>17</td></tr><tr><td></td><td>Vacuity</td><td>4</td><td>1</td><td>0</td><td>0</td><td>3</td></tr><tr><td></td><td>Combined</td><td>208</td><td>186</td><td>1</td><td>1</td><td>19</td></tr><tr><td>RLM25</td><td>Numerical</td><td>12</td><td>n/a</td><td>7</td><td>0</td><td>4</td></tr><tr><td></td><td>Semantic</td><td>367</td><td>n/a</td><td>184</td><td>21</td><td>143</td></tr><tr><td></td><td>Vacuity</td><td>12</td><td>n/a</td><td>3</td><td>1</td><td>8</td></tr><tr><td></td><td>Combined</td><td>373</td><td>n/a</td><td>184</td><td>21</td><td>149</td></tr></table>

Table 3: Flags across different datasets and reviewed defect types. Flagged counts all alerts by each method. Labeled incorrect counts alerts on items the dataset labels incorrect. Translation defects and Source defects count review findings among flagged items that the dataset labels correct. RLM25 has no correctness labels; therefore, all statements are considered “correct” by the dataset.

## 6.1 CASE 1: VACUOUS STATEMENT IN FLMF.

DLMF §7.13(iii) In the first quadrant of $\mathbb { C } ,$ the Fresnel integral $C ( z )$ has an infinite set of zeros $z _ { n } = x _ { n } + i y _ { n } , n = 1 , 2 , 3 , .$ ... arranged in order of increasing absolute value. ${ \bf A s } n  \infty ,$ the $x _ { n }$ and $y _ { n }$ corresponding to the zeros of $\mathbf { \bar { \mathit { C } } } ( z )$ satisfy

$$
x _ { n } \sim \lambda + \frac { \alpha ( \alpha \pi - 4 ) } { 8 \pi \lambda ^ { 3 } } + \cdot \cdot \cdot , \qquad y _ { n } \sim \frac { \alpha } { 2 \lambda } + \cdot \cdot \cdot ,\tag{DLMF §7.13.5}
$$

with

$$
\lambda = \sqrt { 4 n - 1 } , \qquad \alpha = ( 2 / \pi ) \ln ( \pi \lambda ) .\tag{DLMF §7.13.6}
$$

The extracted Lean statement was (definitions stripped)

theorem erf\_7\_13\_5 (z : N → C)   
(hzero : ∀ n, cfresnelC (z n) = 0)   
(hquad : ∀ n, 0 < (z n).re ∧ 0 < (z n).im)   
(hmono : StrictMono (fun n 7→ (z n).re))   
(hall : ∀ w : C, cfresnelC w = 0 →   
0 < w.re → 0 < w.im → ∃ n, z n = w) :   
(fun n 7→ (z n).re - fresnelCZeroReApprox n)   
=o[atTop] (fun n 7→ fresnelCZeroLambda n ˆ (−3 : Z)) ∧   
(fun n 7→ (z n).im - fresnelCZeroImApprox n)   
=o[atTop] (fun n 7→ fresnelCZeroLambda n ˆ (−1 : Z)) := by

This Lean theorem assumes that a sequence $z _ { n }$ enumerates all first-quadrant zeros of $C ( z )$ while their real parts are strictly increasing (hmono). However, the informal source explicitly orders the zeros by increasing absolute value. The assumptions in the Lean statement are contradictory because the large-index zeros will approach the real axis, and a mirror symmetry transforms them into firstquadrant zeros with arbitrarily small positive real parts. Therefore, there must be a first-quadrant zero whose real part is smaller than that of $z _ { \mathrm { 0 } }$ , the first entry of the sequence, violating the monotonicity imposed by hmono. The resulting contradiction among the hypotheses proves False, from which the stated asymptotic conclusion can be obtained vacuously by False.elim.

In verification, this Lean statement obtained a FAITHFUL verdict from the semantic checks because the checker was not able to catch this subtle mistake. Indeed, all other baseline methods accepted it as a correct translation, and the proof compiled without error. Only the vacuity check was able to spot the problem.

## 6.2 CASE 2: A DEFECTIVE INFORMAL SOURCE IN BEYOND COMPILATION.

The informal statement says that if $f$ is holomorphic on a neighborhood of the closed unit disk and $f ( 0 ) \neq 0$ , then there exists a constant $C \in \mathbb { R }$ such that

$$
\left| 1 - \sum _ { k = 0 } ^ { m } { \frac { f ^ { ( k ) } ( 0 ) } { k ! f ( 0 ) } } z ^ { k } \right| \leq C | z | ^ { m + 1 } \qquad ( m \in \mathbb { N } , \ | z | \leq 1 ) .
$$

The corresponding formalization, with its theorem name abbreviated to $\mathrm { X , }$ is

theorem X {f : C → C}   
(hf : ∃ U : Set C, IsOpen U ∧ Metric.closedBall (0 : C) 1 ⊆ U ∧   
DifferentiableOn C f U)   
(h0 : f 0 ̸= 0) :   
∃ C : R, ∀ m : N, ∀ z : C, ∥z∥ ≤ 1 →   
∥(1 : C) - (P k ∈ Finset.range (m + 1),   
((derivˆ[k] f) 0) / ((k.factorial : N) <sub>\*</sub> f 0) <sub>\*</sub> z ˆ k)∥   
≤ C <sub>\*</sub> ∥z∥ ˆ (m + 1)

This is a faithful translation of the informal statement. However, the numerical check finds the admissible counter example $f ( z ) = e ^ { z }$ . With $m = 1$ , for real $0 < z \le 1$ the claimed bound becomes

$$
| 1 - ( 1 + z ) | = | z | \leq C | z | ^ { 2 } .
$$

Dividing both sides by $| z | ^ { 2 }$ , we get $C \geq 1 / | z |$ , which no constant C can satisfy as $z  0 .$

In the experiment, this correct-labeled entry is flagged only by the numerical check among all methods. Back-translation, GPT-5.5 and Opus 4.6 in both single-pair and joint modes return FAITHFUL; LeanScorer accepts it with a score of 0.67 at a threshold of $0 . 6 ;$ FormalAlign accepts it with a certainty score of 0.394 at a threshold of 0.3. None of the stored semantic explanations identifies the impossible uniform bound.

## 7 CONCLUSION

In this paper, we introduced our verification method, conducted experiments across different benchmarks, and demonstrated that our method yields the lowest accepted-error rate, therefore yielding the most trustworthy outcome. We integrate these methods into an Extract–Verify–Prove pipeline with human gates and audits, ensuring completeness and faithfulness of the formalization results. FLMF provides the development setting and a preliminary evaluation panel, and its broader dataset remains under construction and will be presented in a separate paper.

We highlight the numerical and vacuity checks in our verification, which are complementary to the verdicts of an LLM or model-based judge, the most common approach used across all other formalization work. According to our experiments, these checks are able to spot mistakes ignored by all LLM judges, including subtle translation defects or informal source errors. The checks are also informative in pointing out the exact numerical misalignment or proof inconsistency. A theorem passing these checks can be trusted to be at least numerically consistent and not trivially vacuous. Combined with our autoformalization pipeline, we can then obtain a promising Lean formalization of the informal source.

There are certain limitations to our current methods. Numerical and vacuity checks are only applicable to numerically evaluable theorems or complete proofs. There could be defects in these checks as well, for example, a wrong binding in the numerical check or a subtly vacuous proof that does not fall within the detection range of our vacuity detector. Passing these checks doesn’t guarantee correctness either, for example a narrower domain in the Lean statement compared to the informal source is a common failure mode.

Future work will extend the numerical and vacuity checks to handle additional corner cases encountered during formalization, while guaranteeing faithfulness of the checks. We aim to reduce vacuity false alarms by distinguishing contradictory assumptions and missing mathematical content from legitimate definitional identities and routine proofs. We will also explore additional mechanical checks that complement the existing methods and provide independently checkable evidence.

## ACKNOWLEDGEMENTS

We thank Shi Chen, Yongxi Lin, Leyan Pan, Fred Rajasekaren, Christian Szegedy, Zixiao Wang, and Ruofan Zeng for valuable discussions. Most of this work was completed during AM’s internship at AletheAI in summer 2026, and we are grateful to AletheAI for the resources and support that made this work possible. We also thank the DLMF committee at NIST for their valuable feedback.

## REFERENCES

NIST Digital Library of Mathematical Functions, Release 1.2.7, 2026. https://dlmf.nist.gov/.

L. de Moura and S. Ullrich. The Lean 4 theorem prover and programming language. In CADE-28, 2021.

The mathlib Community. The Lean mathematical library. In CPP, 2020.

M. R. Douglas, S. Hoback, A. Mei, and R. Nissim. Formalization of QFT. arXiv:2603.15770, 2026.

H. Wang, R. Xie, Y. Wang, G. Gao, X. Yu, and B. Dong. Aria: an agent for retrieval and iterative auto-formalization via dependency graph. In ICLR, 2026.

R. Cabral, T. Manh, X. Yu, W. M. Tai, Z. Feng, and X. Shen. ProofFlow: a dependency graph approach to faithful proof autoformalization. In ICLR, 2026.

F. Gloeckle, A. Rammal, C. Arnal, R. Munos, V. Cabannes, G. Synnaeve, and A. Hayat. Automatic textbook formalization. arXiv:2604.03071, 2026.

A. Rammal, N. Patel, F. Gloeckle, A. Hayat, J. Kempe, R. Munos, C. Arnal, and V. Cabannes. Formalizing mathematics at scale. arXiv:2605.29955, 2026. https://github.com/facebookresearch/atlas-lean.

L. Milikic, S. Guilloud, K. Nguyen, and V. Kuncak. LeanFlow: a case study in workflow-driven Lean autoformalization. In ICML AIfor Math Workshop, 2026.

B. Requena Pozo, A. Letson, K. Nowakowski, I. Beltran Ferreiro, and L. Sarra. A minimal agent for automated theorem proving. arXiv:2602.24273, 2026.

Math, Inc. OpenGauss: an open-source autoformalization harness for Lean. Software, 2026. https://github.com/math-inc/OpenGauss.

O. An, M. J. Min, X. Si, and O. Bastani. Ground False: dissecting errors in formal mathematics benchmarks. In 3rd AIfor Math Workshop: Toward Self-Evolving Scientific Agents, 2026. https://openreview.net/forum?id=5c0RSYyIWW. ProofNet-Verified release: https://github.com/marcusm117/ProofNet-Verified.

M. Abramowitz and I. A. Stegun, editors. Handbook ofMathematical Functions. National Bureau of Standards, 1964.

E. T. Whittaker and G. N. Watson. A Course of Modern Analysis. Cambridge University Press, 4th edition, 1927.

A. Erdelyi, W. Magnus, F. Oberhettinger, and F. G. Tricomi. Higher Transcendental Functions, volumes 1–3. McGraw–Hill, 1953–1955.

A. Cuyt, V. B. Petersen, B. Verdonk, H. Waadeland, and W. B. Jones. Handbook of Continued Fractions for Special Functions. Springer, 2008.

H. S. Wall. Analytic Theory ofContinued Fractions. Van Nostrand, 1948.

W. B. Jones and W. J. Thron. Continued Fractions: Analytic Theory and Applications. Addison–Wesley, 1980.

F. Oberhettinger and L. Badii. Tables of Laplace Transforms. Springer, 1973.

A. C. Dixon. On a certain double integral. Proceedings of the London Mathematical Society, s2-2:8–15, 1905.

G. W. Anderson. A short proof of Selberg’s generalized beta formula. Forum Mathematicum, 3:415–417, 1991.

P. J. Forrester and S. O. Warnaar. The importance of the Selberg integral. Bulletin of the American Mathematical Society, 45:489–534, 2008. arXiv:0710.3981.

A. Selberg. Bemerkninger om et multipelt integral. Norsk Matematisk Tidsskrift, 26:71–78, 1944.

M. L. Mehta. Random Matrices. Elsevier, 3rd edition, 2004.

F. J. Dyson. Statistical theory of the energy levels of complex systems. I. Journal ofMathematical Physics, 3:140–156, 1962.

A.-L. Cauchy. Exercices d’analyse et de physique mathematique. Bachelier, 1841.

J. A. Wilson. Some hypergeometric orthogonal polynomials. SIAM Journal on Mathematical Analysis, 11:690–701, 1980.

G. E. Andrews, R. Askey, and R. Roy. Special Functions. Cambridge University Press, 1999.

K. G. C. von Staudt. De numeris Bernoullianis, 1845; T. Clausen, related denominator theorem, Astronomische Nachrichten, 1840.

N. E. Norlund. Vorlesungen uber Differenzenrechnung. Springer, 1924.

E. E. Kummer. Beitrag zur Theorie der Function Γ(x). Journalfur die reine und angewandte Mathematik, 35, 1847.

L. Euler. Early correspondence on the interpolation of factorials, 1729.

A.-M. Legendre. Recherches sur diverses sortes d’integrales definies. Memoires de la classe des sciences mathematiques et physiques, 1809.

J. L. Raabe. Zur Theorie der Eulerschen Integrale. Journal fur die reine und angewandte Mathematik, 20:146–159, 1840.

R. Bellman. A BriefIntroduction to Theta Functions. Holt, Rinehart and Winston, 1961.

G. Polya. Remarks on computing the probability integral in one and two dimensions. In Proceedings of the First Berkeley Symposium on Mathematical Statistics and Probability, 1949.

J. M. Blair, C. A. Edwards, and J. H. Johnson. Rational Chebyshev approximations for the inverse of the error function. Mathematics ofComputation, 30, 1976.

B. C. Carlson. Special Functions ofApplied Mathematics. Academic Press, 1977.

B. C. Carlson. Numerical computation of real or complex elliptic integrals. Numerical Algorithms, 10:13–26, 1995.

J. V. Armitage and W. F. Eberlein. Elliptic Functions. Cambridge University Press, 2006.

A. Tauber. Ein Satz aus der Theorie der unendlichen Reihen. Monatsheftefur Mathematik und Physik, 8:273–277, 1897.

H.-J. Runckel. On the zeros of the hypergeometric function. Mathematische Annalen, 191:53–58, 1971.

M. Hairer. Φ<sup>4</sup><sub>3</sub> is orthogonal to GFF. Note, 2022. https://hairer.org/Phi4.pdf.

M. Hairer, S. Kusuoka, and H. Nagoji. Singularity of solutions to singular SPDEs. arXiv:2409.10037, 2024.

N. Ebel, T. Kennedy, and S. Rychkov. Tensor renormalization group meets computer assistance. arXiv:2506.03247, 2025.

A. Ospanov, F. Farnia, and R. Yousefzadeh. APOLLO: automated LLM and Lean collaboration for advanced formal reasoning. arXiv:2505.05758, 2025.

J. Lu, K. Emond, K. Yang, S. Chaudhuri, W. Sun, and W. Chen. Lean Finder: semantic search for Mathlib that understands user intents. arXiv:2510.15940, 2025.

F. Johansson et al. mpmath: a Python libraryfor arbitrary-precisionfloating-point arithmetic, version 1.3.0, 2023.

Z. Azerbayev, B. Piotrowski, H. Schoelkopf, E. W. Ayers, D. Radev, and J. Avigad. ProofNet: Autoformalizing and Formally Proving Undergraduate-Level Mathematics. arXiv:2302.12433, 2023. https://arxiv.org/abs/2302.12433.

A. Poiroux, G. Weiss, V. Kuncak, and A. Bosselut. Reliable Evaluation and Benchmarks for Statementˇ Autoformalization. In EMNLP, pp. 17947–17969, 2025. https://aclanthology.org/2025.emnlp-main.907/.

A. Poiroux, A. Bosselut, and V. Kuncak. RLMEval: Evaluating Research-Level Neural Theorem Proving. Inˇ Findings ofEMNLP, 2025. https://aclanthology.org/2025.findings-emnlp.581/.

M. J. Min, Y. Gao, W. Sy, Z. Li, X. Si, and O. Bastani. Divide and Abstract: Autoformalization via Decomposition and Abstraction Learning. In ICLR, 2026. https://openreview.net/forum?id=NjgaeXNit3.

K. Zhang, P. Gallardo Candela, S. Murthy, Y. Xie, Z. Wang, and M. Raissi. Beyond Compilation: Evaluating Faithful Natural-Language-to-Lean Statement Formalization. arXiv:2606.31002, version 2, 2026. https://arxiv.org/abs/2606.31002v2.

Yu Xuejun, J. Zhong, Z. Feng, P. Zhai, R. Yousefzadeh, W. C. Ng, H. Liu, Z. Shou, J. Xiong, Y. Zhou, C. B. Ong, A. J. Sugiarto, Y. Zhang, W. M. Tai, H. Cao, D. Lu, J. Sun, Q. Xu, Shen Xin, and Z. Li. Mathesis: Towards Formal Theorem Proving from Natural Languages. arXiv:2506.07047, 2025. https://arxiv.org/abs/2506.07047. LeanScorer implementation: https://github.com/Huawei-AI4Math/Mathesis.

J. Lu, Y. Wan, Y. Huang, J. Xiong, Z. Liu, and Z. Guo. FormalAlign: Automated Alignment Evaluation for Autoformalization. In ICLR, 2025. https://arxiv.org/abs/2410.10135. Data and code: https://github.com/rookie-joe/FormalAlign.

Q. Liu, X. Zheng, X. Lu, Q. Cao, and J. Yan. Rethinking and Improving Autoformalization: Towards a Faithful Metric and a Dependency Retrieval-Based Approach. In ICLR, 2025. https://openreview.net/forum?id=hUb2At2DsQ.

Y. Liu, T. Zhu, X. Liu, Y. Chen, Z. Liu, Q. Guo, J. Zhang, K. Bao, and T. Luo. Generalized Tree Edit Distance (GTED): A Faithful Evaluation Metric for Statement Autoformalization. In ICML AIfor Math Workshop, 2025. https://arxiv.org/abs/2507.07399v2.

X. Liu, T. Zhu, Z. Dong, Y. Liu, Q. Guo, Z. Liu, Y. Chen, and T. Luo. ASSESS: A Semantic and Structural Evaluation Framework for Statement Similarity. In ICLR, 2026. https://arxiv.org/abs/2509.22246v2.

J. Kim, H. Han, and S.-W. Hwang. Benchmarking Testing in Automated Theorem Proving. In ACL Industry Track, pp. 2241–2260, 2026. https://aclanthology.org/2026.acl-industry.150/.

J. Meadows, L. Zhang, and A. Freitas. FormalScience: Scalable Human-in-the-Loop Autoformalisation of Science with Agentic Code Generation in Lean. In ACL, 2026. https://arxiv.org/abs/2604.23002. FormalPhysics dataset: https://huggingface.co/datasets/jmeadows17/FormalPhysics.

J. Liu, X. Lin, J. Bayer, Y. Dillies, W. Jiang, X. Liang, R. Soletskyi, H. Wang, Y. Xie, B. Xiong, Z. Yang, J. Zhang, L. Zhi, J. Li, and Z. Liu. CombiBench: Benchmarking LLM Capability for Combinatorial Mathematics. arXiv:2505.03171, 2025. https://arxiv.org/abs/2505.03171.

J. Jiang, W. He, Y. Wang, G. Gao, Y. Hu, J. Wang, N. Guan, P. Wu, C. Dai, L. Xiao, and B. Dong. FATE: A Formal Benchmark Series for Frontier Algebra of Multiple Difficulty Levels. arXiv:2511.02872, 2025. https://arxiv.org/abs/2511.02872.

Z. Yu, R. Peng, K. Ding, Y. Li, Z. Peng, M. Liu, Y. Zhang, Z. Yuan, H. Xin, W. Huang, Y. Wen, G. Zhang, and W. Liu. FormalMATH: Benchmarking Formal Mathematical Reasoning of Large Language Models. arXiv:2505.02735, 2025. https://arxiv.org/abs/2505.02735.

K. Zheng, J. M. Han, and S. Polu. MiniF2F: A Cross-System Benchmark for Formal Olympiad-Level Mathematics. In ICLR, 2022. https://arxiv.org/abs/2109.00110.

G. Tsoukalas, J. Lee, J. Jennings, J. Xin, M. Ding, M. Jennings, A. Thakur, and S. Chaudhuri. PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition. In NeurIPS Datasets and Benchmarks Track, 2024. https://arxiv.org/abs/2407.11214.

K. Yang, A. Swope, A. Gu, R. Chalamala, P. Song, S. Yu, S. Godil, R. Prenger, and A. Anandkumar. LeanDojo: Theorem Proving with Retrieval-Augmented Language Models. In NeurIPS, 2023. https://github.com/lean-dojo/LeanDojo.

L. Zhang, M. Valentino, and A. Freitas. Autoformalization in the Wild: Assessing LLMs on Real-World Mathematical Definitions. In EMNLP, 2025. https://arxiv.org/abs/2502.12065.

## A CHAPTER-LEVEL PIPELINE PROGRESS

Table 4 reports progress of FLMF on all 36 chapters. Of the 8,833 numbered DLMF equations, 1,326 (15%) are stated in FLMF, 641 (7%) are verified, and 1,142 have a sorry-free proof. Chapter 5, 7, and 20 are essentially complete. Chapters 6 and 15 are fully extracted but less fully verified (56 of 97 and 52 of 198).

Table 4: Complete DLMF chapter progress, reported as counts.
<table><tr><td>Ch.</td><td>Chapter</td><td>Targets</td><td>Extract</td><td>Verified Proved</td><td></td></tr><tr><td>1</td><td>Algebraic and Analytic Methods</td><td>705</td><td>88</td><td>38</td><td>75</td></tr><tr><td>2</td><td>Asymptotic Approximations</td><td>349</td><td>2</td><td>1</td><td>2</td></tr><tr><td>3</td><td>Numerical Methods</td><td>297</td><td>2</td><td>0</td><td>2</td></tr><tr><td>4</td><td>Elementary Functions</td><td>518</td><td>26</td><td>14</td><td>24</td></tr><tr><td>5</td><td>Gamma Function</td><td>164</td><td>162</td><td>123</td><td>137</td></tr><tr><td>6</td><td>Exponential, Logarithmic, Sine and Cosine Integrals</td><td>100</td><td>97</td><td>56</td><td>77</td></tr><tr><td>7</td><td>Error Functions, Dawson&#x27;s and Fresnel Integrals</td><td>137</td><td>137</td><td>136</td><td>137</td></tr><tr><td>8</td><td>Incomplete Gamma and Related Functions</td><td>241</td><td>15</td><td>1</td><td>15</td></tr><tr><td>9</td><td>Airy and Related Functions</td><td>230</td><td>13</td><td>4</td><td>11</td></tr><tr><td>10</td><td>Bessel Functions</td><td>698</td><td>52</td><td>10</td><td>40</td></tr><tr><td>11</td><td>Struve and Related Functions</td><td>149</td><td>11</td><td>6</td><td>11</td></tr><tr><td>12</td><td>Parabolic Cylinder Functions</td><td>177</td><td>4</td><td>1</td><td>4</td></tr><tr><td>13</td><td>Confluent Hypergeometric Functions</td><td>381</td><td>17</td><td>3</td><td>17</td></tr><tr><td>14</td><td>Legendre and Related Functions</td><td>284</td><td>14</td><td>4</td><td>11</td></tr><tr><td>15</td><td>Hypergeometric Function 2F1</td><td>202</td><td>198</td><td>52</td><td>157</td></tr><tr><td>16</td><td>Generalized Hypergeometric and Meijer G</td><td>99</td><td>5</td><td>2</td><td>5</td></tr><tr><td>17</td><td>q-Hypergeometric and Related Functions</td><td>182</td><td>7</td><td>1</td><td>7</td></tr><tr><td>18</td><td>Orthogonal Polynomials</td><td>677</td><td>56</td><td>10</td><td>43</td></tr><tr><td>19</td><td>Elliptic Integrals</td><td>464</td><td>17</td><td>3</td><td>17</td></tr><tr><td>20</td><td>Theta Functions</td><td>113</td><td>113</td><td>88</td><td>112</td></tr><tr><td>21</td><td>Multidimensional Theta Functions</td><td>59</td><td>0</td><td>n/a</td><td>n/a</td></tr><tr><td>22</td><td>Jacobian Elliptic Functions</td><td>258</td><td>22</td><td>10</td><td>18</td></tr><tr><td>23</td><td>Weierstrass Elliptic and Modular Functions</td><td>167</td><td>21</td><td>6</td><td>21</td></tr><tr><td>24</td><td>Bernoulli and Euler Polynomials</td><td>188</td><td>22</td><td>7</td><td>20</td></tr><tr><td>25</td><td>Zeta and Related Functions</td><td>174</td><td>24</td><td>10</td><td>23</td></tr><tr><td>26</td><td>Combinatorial Analysis</td><td>201</td><td>16</td><td>9</td><td>16</td></tr><tr><td>27</td><td>Functions of Number Theory</td><td>132</td><td>33</td><td>20</td><td>29</td></tr><tr><td>28</td><td>Mathieu Functions and Hill&#x27;s Equation</td><td>353</td><td>15</td><td>1</td><td>15</td></tr><tr><td>29</td><td>Lamé Functions</td><td>205</td><td>1</td><td>0</td><td>1</td></tr><tr><td>30</td><td>Spheroidal Wave Functions</td><td>114</td><td>3</td><td>0</td><td>2</td></tr><tr><td>31</td><td>Heun Functions</td><td>120</td><td>103</td><td>25</td><td>64</td></tr><tr><td>32</td><td>Painlevé Transcendents</td><td>286</td><td>6</td><td>0</td><td>6</td></tr><tr><td>33</td><td>Coulomb Functions</td><td>146</td><td>0</td><td>n/a</td><td>n/a</td></tr><tr><td>34</td><td>3j, 6j, 9j Symbols</td><td>64</td><td>18</td><td>0</td><td>17</td></tr><tr><td>35</td><td>Functions of Matrix Argument</td><td>62</td><td>0</td><td>n/a</td><td>n/a</td></tr><tr><td>36</td><td>Integrals with Coalescing Saddles</td><td>137</td><td>6</td><td>0</td><td>6</td></tr><tr><td>Total</td><td></td><td>8,833</td><td>1,326</td><td>641</td><td>1,142</td></tr></table>

## B DLMF-SIDE DISCREPANCIES IDENTIFIED TO DATE

The following potential misprints in the DLMF had been identified while formalizing and proving the corresponding FLMF statements. They have been confirmed by the DLMF committee and corrections are updated.
<table><tr><td>DLMF</td><td>Printed defect and correction</td><td>Independent evidence</td></tr><tr><td>20.10.4</td><td>The display uses  $( 1 + \beta )$  where scale covariance and the theta shift require  $( \ell + \beta )$  . The  $\theta _ { 1 }$  transform has the opposite printed sign, and the  $\theta _ { 1 }$  and shifted  $\theta _ { 2 }$  integrals are negatives of each other, not equal.</td><td>Theta quasi-periodicity already makes the printed chain inconsistent at  $\ell = 1 ;$  termwise Laplace transforms and independent 25-digit quadrature recover the corrected form.</td></tr><tr><td>20.10.5</td><td>The display uses (1 + β) where scale covariance and the theta shift require  $( \ell + \beta )$ </td><td>Termwise Laplace transforms and independent 25-digit quadrature recover the corrected form.</td></tr><tr><td>20.13.5</td><td>The alternating heat-kernel sum is printed as  $\theta _ { 4 } ;$  Poisson summation gives  $\theta _ { 2 } .$ </td><td>At  $( \alpha , t , z ) = ( 0 . 7 , 0 . 3 , 0 . 4 )$  , the sum is 1.5982 .. . and matches 5  $\theta _ { 2 }$  to  $1 . 6 \times 1 0 ^ { - 2 6 } .$  whereas the printed  $\theta _ { 4 }$  side is  $0 . 3 9 7 2 \ldots 3$  complex samples agree.</td></tr></table>

Table 5: DLMF misprint findings.

## C DLMF EXTRACTION EXAMPLE

The following is the presentation MathML (pMML) source for DLMF §5.5.3.

<m:math xmlns:m="http://www.w3.org/1998/Math/MathML"   
altimg="E3.png" altimg-height="99px" altimg-valign="-7px"   
altimg-width="149px"   
alttext="\Gamma\left(z\right)\Gamma\left(1-z\right)=\pi/\sin\left(\pi z\right),"   
display="block">   
<m:mrow>   
<m:mrow>   
<m:mrow>   
<m:mrow>   
<m:mi href="DLMF:/5.2#E1" mathvariant="normal"   
title="gamma function">&#x393;</m:mi>   
<m:mo>&#x2061;</m:mo>   
<m:mrow><m:mo>(</m:mo>   
<m:mi href="DLMF:/5.1#t1.r4"   
title="complex variable">z</m:mi>   
<m:mo>)</m:mo></m:mrow>   
</m:mrow>   
<m:mo>&#x2062;</m:mo>   
<m:mrow>   
<m:mi href="DLMF:/5.2#E1" mathvariant="normal"   
title="gamma function">&#x393;</m:mi>   
<m:mo>&#x2061;</m:mo>   
<m:mrow><m:mo>(</m:mo><m:mrow><m:mn>1</m:mn>   
<m:mo>&#x2212;</m:mo>   
<m:mi href="DLMF:/5.1#t1.r4"   
title="complex variable">z</m:mi>   
</m:mrow><m:mo>)</m:mo></m:mrow>   
</m:mrow>   
</m:mrow>   
<m:mo>=</m:mo>   
<m:mrow>   
<m:mi href="DLMF:/3.12#E1"   
title="the ratio of the circumference of a circle to its diameter">   
&#x3C0;</m:mi>   
<m:mo>/</m:mo>   
<m:mrow>   
<m:mi href="DLMF:/4.14#E1" title="sine function">sin</m:mi>   
<m:mo>&#x2061;</m:mo>   
<m:mrow><m:mo>(</m:mo><m:mrow>   
<m:mi href="DLMF:/3.12#E1"   
title="the ratio of the circumference of a circle to its diameter">   
&#x3C0;</m:mi>   
<m:mo>&#x2062;</m:mo>   
<m:mi href="DLMF:/5.1#t1.r4"   
title="complex variable">z</m:mi>   
</m:mrow><m:mo>)</m:mo></m:mrow>   
</m:mrow>   
</m:mrow>   
</m:mrow>   
<m:mo>,</m:mo>   
</m:mrow>   
</m:math>

Because the standalone pMML omits the side condition, the extractor reads the equation’s container in the full DLMF section page. Its normalized JSON output is:

```json
{
"ta ": "5.5.3",
"url": "https://dlmf.nist.gov/5.5#E3",
"kind": "equation",
"latex": [
"\\Gamma\\left(z\\right)\\Gamma\\left(1-z\\right)=\\pi/\\sin\\left(\\pi z\\right),",
"\\Gamma\\left(\\NVar{z}\\right)",
"z\\neq 0,\\pm 1,\\dots",
"\\sin\\NVar{z}",
"\\pi",
"z"
],
"eqs": [],
"section_eqs": [
"5.5.1",
"\\Gamma\\left(z+1\\right)=z\\Gamma\\left(z\\right),"
],
[
"5.5.2",
"\\psi\\left(z+1\\right)=\\psi\\left(z\\right)+\\frac{1}{z}."
],
[
"5.5.3",
"\\Gamma\\left(z\\right)\\Gamma\\left(1-z\\right)=\\pi/\\sin\\left(\\pi z\\right),"
],
[
```

"5.5.4",   
"\\psi\\left(z\\right)-\\psi\\left(1-z\\right)=-\\pi/\\tan\\left(\\pi z\\right),"   
],   
"5.5.5",   
"\\Gamma\\left(2z\\right)=\\pi^{-1/2}2^{2z-1}\\Gamma\\left(z\\right)\\Gamma\\left(z+\\tfra <sub>⌋</sub>   
,→ c{1}{2}\\right)."   
],   
"5.5.6",   
"\\Gamma\\left(nz\\right)=(2\\pi)^{(1-n)/2}n^{nz-(1/2)}\\prod\_{k=0}^{n-1}\\Gamma\\left(z+\ <sub>⌋</sub>   
,→ \frac{k}{n}\\right)."   
"5.5.7",   
"\\prod\_{k=1}^{n-1}\\Gamma\\left(\\frac{k}{n}\\right)=(2\\pi)^{(n-1)/2}n^{-1/2}."   
"5.5.8",   
"\\psi\\left(2z\\right)=\\tfrac{1}{2}\\left(\\psi\\left(z\\right)+\\psi\\left(z+\\tfrac{1} <sub>⌋</sub>   
,→ {2}\\right)\\right)+\\ln 2,"   
],   
"5.5.9",   
"\\psi\\left(nz\\right)=\\frac{1}{n}\\sum\_{k=0}^{n-1}\\psi\\left(z+\\frac{k}{n}\\right)+\\   
→ ln n."   
]   
],   
"context": ", pMML , png See also: Annotations for \u00a75.5(i) , \u00a75.5 and Ch.5   
,→ \u00a75.5(ii) Reflection \u24d8 Keywords: gamma function , psi function , reflection   
,→ formula Notes: See Olver ( 1997b , pp. 35 and 39) . Permalink:   
,→ http://dlmf.nist.gov/5.5.ii See also: Annotations for \u00a75.5 and Ch.5 5.5.3 \u0393   
,→ \u2061 ( z ) \u2062 \u0393 \u2061 ( 1 \u2212 z ) = \u03c0 / sin \u2061 ( \u03c0 \u2062 z   
,→ ) , z \u2260 0 , \u00b1 1 , \u2026 , \u24d8 Symbols: \u0393 \u2061 ( z ) : gamma function   
,→ , \u03c0 : the ratio of the circumference of a circle to its diameter , sin \u2061 z :   
,→ sine function and z : complex variable A&S Ref: 6.1.17 (without the condition on z .)   
,→ Referenced by: \u00a710.19(i) , \u00a715.4(ii) , \u00a715.8(ii) , (25.9.2) , \u00a75.21 ,   
,→ (9.10.18) , (9.12.15) Permalink: http://dlmf.nist.gov/5.5.E3 Encodings: TeX , pMML , png   
See also: Annotations for \u00a75.5(ii) , \u00a75.5 and Ch.5 <table",,→   
"xref\_context": [],   
"constraints": [   
"z\\neq 0,\\pm 1,\\dots"   
],   
"symbols": [   
"token": "\u0393",   
"href": "5.2#E1",   
"gloss": "gamma function"   
},   
"token": "z",   
"href": "5.1#t1.r4",   
"gloss": "complex variable"   
},   
"token": "\u03c0",   
"href": "3.12#E1",   
"gloss": "the ratio of the circumference of a circle to its diameter"   
},   
"token": "sin",   
"href": "4.14#E1",   
"gloss": "sine function"   
"info": {   
"referenced\_by": "\u00a710.19(i) , \u00a715.4(ii) , \u00a715.8(ii) , (25.9.2) , \u00a75.21 ,   
,→ (9.10.18) , (9.12.15)",   
"as\_ref": "6.1.17 (without the condition on z .)",   
"proof\_sketch": "",   
"sources": ""   
"keywords": "",   
"notes": "",   
"defines": "",   
"symbols\_text": "\\Gamma\\left(\\NVar{z}\\right) : gamma function , \\pi : the ratio of the   
,→ circumference of a circle to its diameter , \\sin\\NVar{z} : sine function and z :   
complex variable",,→   
"changes": []   
},   
"parser\_version": 2

## D DATASET CONSTRUCTION, PANEL COMPOSITION, AND DETECTION

## D.1 DATASET CONSTRUCTION AND LABELS

The evaluation contains 1,810 informal–formal pairs from four datasets. Table 6 gives their composi tion. Each dataset supplies its own criteria for labeling a pair correct or incorrect; these labels do not represent a common judgment of faithfulness or mathematical validity.

FLMF (182 correct / 289 incorrect). The FLMF dataset contains 471 statements in eight components: 40 library seeds, 40 copies with bound variables renamed, 7 human-approved corrections to DLMF misprints, 160 intentional mutations of the seeds, 17 historical FLMF errors, 7 printed forms of DLMF misprints, 100 sampled verified theorems, and 100 statements from the project’s errata log. The seeds and renamings each contain 38 correct and 2 incorrect items, and the sampled verified theorems contain 99 correct and 1 incorrect item. Corrected misprints are labeled correct. Mutations, historical errors, printed misprints, and errata items are labeled incorrect.

The mutations target eight mechanisms (M1–M8): impossible hypotheses, wrong objects, branches, totalization at poles, quantifier strength, indices and coercions, normalization, and lost source content. There are 12–20 mutations per mechanism and 24 simpler sanity mutations (Table 7). We exclude 20 independent control nominations. We also exclude two Fresnel coefficient pairs, comprising four items, because their stronger asymptotic claims are unresolved. Historical Fresnel errors remain included.

The 100 sampled verified theorems satisfy the same eligibility criteria as the seeds: a semantic verified stamp at the pinned commit, a single numbered DLMF equation, and a completed proof. Selection is deterministic and stratified by library area, with a cap of 20 items per stratum. This component is disjoint from the seeds, renamings, mutations, historical artifacts, and misprint pairs by both declaration name and DLMF equation.

Each of the 100 errata items is a declaration recovered from git history in its form immediately before correction. Of these, 76 correspond to errors found by the numerical or vacuity check and 24 to errors found by other methods. They form 65 erratum clusters, and 99 of the 100 statements elaborate. Together with the mutations, historical errors, printed misprints, and five library entries rejected in human review, they account for the 289 incorrect items. The 129 incorrect items other than mutations provide examples of errors arising in formalization or in the source.

ProofNet (163 correct / 204 incorrect) (Azerbayev et al., 2023). We use 367 original ProofNet statements from the Lean 4.28 port distributed with ProofNet-Verified. The formal statements are those of ProofNet, and the labels are the Ground False verdicts (An et al., 2026). The 163 items labeled faithful are correct. The incorrect items comprise 75 labeled stronger, 55 weaker, 64 incomparable, and 10 whose informal statement is ambiguous (9) or wrong (1).

Beyond Compilation (245 correct / 106 incorrect) (Zhang et al., 2026). We use 351 of the 400 released sources. The exclusions are 42 reported compilation failures, 6 entries from which no single statement can be separated from the proof, and 1 informal source that states no claim. A pair is labeled correct if the release’s GPT-5.2 judge assigns a score of at least 9. Human annotations are available for 156 items whose recorded judge score matches the candidate’s released score, and we use them as secondary labels. The annotations are predominantly acceptances and concern translations selected from model acceptances, so they do not independently assess the items labeled incorrect by the judge.

RLM25 (621 reference pairs) (Poiroux et al., 2025a). The reconstructed dataset contains reference pairs from six projects: Carleson (111), FLT (56), FLT3 (84), PFR (146), PrimeNumberTheoremAnd (99), and TestingLowerBounds (125). The published release contains 619 pairs. Reconstruction from public sources yields 621, including one extra eligible node in PFR and one in TestingLowerBounds, so membership does not exactly match the release manifest. RLM25 provides no correctness labels. We treat all reference pairs as correct and report their flag rate.

The formal context includes project definitions referenced by each statement, including definitions declared in other files and those introduced with irreducible\_def or noncomputable abbrev. For 310 of the 621 items, this supplies definitions absent from the source prefix in the statement’s own file. Statement extraction accounts for local let expressions, which is necessary to preserve two statements in full. FormalAlign’s released adapter does not receive formal context.

Proof availability. The goal-independent part of the vacuity check requires a completed proof. The 100 sampled verified FLMF theorems have completed proofs. For errata items, coverage depends on the availability of a replayable proof of the statement as written before correction. Beyond Compilation and the original ProofNet statements provide no proofs, so the vacuity check on these datasets is limited to trivial proof methods, as described in Section 5.1.

## D.2 PANEL COMPOSITION AND DETECTION

Table 6 gives the composition of the four datasets. Table 7 reports flags on the 271 FLMF seeds, renamings, mutations, historical artifacts, and misprint items.

<table><tr><td>Dataset</td><td>Component</td><td>Correct</td><td>Incorrect</td></tr><tr><td rowspan="8">FLMF</td><td>Seeds</td><td>38</td><td>2</td></tr><tr><td>Bound-variable renamings of the seeds</td><td>38</td><td>2</td></tr><tr><td>Human-approved corrected forms of DLMF</td><td>7</td><td></td></tr><tr><td>misprints Mutations of the seeds (M1–M8 + sanity)</td><td></td><td>160</td></tr><tr><td>Historical FLMF errors</td><td></td><td>17</td></tr><tr><td>Misprinted DLMF equations</td><td></td><td>7</td></tr><tr><td>Sampled verified theorems after human review</td><td>99</td><td>1</td></tr><tr><td>Errata statements</td><td></td><td>100</td></tr><tr><td rowspan="5">ProofNet (verdict by</td><td>FLMF total (471 items)</td><td>182</td><td>289</td></tr><tr><td>faithful</td><td>163</td><td></td></tr><tr><td>stronger</td><td></td><td>75</td></tr><tr><td>weaker</td><td></td><td>55</td></tr><tr><td>incomparable informal source ambiguous (9) or wrong (1)</td><td></td><td>64</td></tr><tr><td>Beyond</td><td>Full release (label: released judge score ≥ 9)</td><td>245</td><td>10</td></tr><tr><td>Compilation RLM25</td><td>Reconstructed reference pairs from all six projects,</td><td></td><td>106</td></tr><tr><td>Total</td><td>with project definitions</td><td>621</td><td></td></tr></table>

Table 6: Dataset composition (1,810 items).

<table><tr><td>FLMF subset</td><td>n Ours</td><td></td><td>GPT-5.5</td><td>GPT-5.5 (joint)</td><td>Opus 4.6</td><td>LeanScorer</td><td>FormalAlign</td><td>N</td><td>B</td><td>V</td></tr><tr><td colspan="10">Seeds, renamings, and corrected misprints</td></tr><tr><td>Seeds</td><td>40</td><td>12</td><td>8</td><td>4</td><td>3</td><td>22</td><td>37</td><td>0</td><td>9</td><td>3</td></tr><tr><td>Renamed seeds</td><td>40</td><td>15</td><td>7</td><td>4</td><td>4</td><td>21</td><td>36</td><td>0</td><td>12</td><td>3</td></tr><tr><td>Corrected misprints</td><td>7</td><td>7</td><td>7</td><td>7</td><td>6</td><td>3</td><td>1</td><td>0</td><td>7</td><td>0</td></tr><tr><td colspan="10">Mutations, historical artifacts, and printed misprints</td><td></td></tr><tr><td>M1 impossible hypotheses</td><td>20</td><td>20</td><td>20</td><td>20</td><td>19</td><td>20</td><td>18</td><td>0</td><td>20</td><td>0</td></tr><tr><td>M2 wrong object</td><td>16</td><td>16</td><td>16</td><td>16</td><td>15</td><td>16</td><td>16</td><td>15</td><td>16</td><td>0</td></tr><tr><td>M3 branches/phases</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td><td>19</td><td>18</td><td>20</td><td>0</td></tr><tr><td>M4 totalization/poles</td><td>20</td><td>18</td><td>18</td><td>18</td><td>13</td><td>19</td><td>20</td><td>13</td><td>18</td><td>1</td></tr><tr><td>M5 quantifiers/asymptotics</td><td>16</td><td>16</td><td>14</td><td>14</td><td>14</td><td>16</td><td>14</td><td>9</td><td>15</td><td>0</td></tr><tr><td>M6 indices/coercions</td><td>16</td><td>16</td><td>16</td><td>16</td><td>16</td><td>16</td><td>16</td><td>14</td><td>16</td><td>0</td></tr><tr><td>M7 normalization/scaling</td><td>16</td><td>16</td><td>15</td><td>16</td><td>15</td><td>16</td><td>16</td><td>11</td><td>16</td><td>0</td></tr><tr><td>M8 lost source content</td><td>12</td><td>11</td><td>12</td><td>12</td><td>12</td><td>12</td><td>12</td><td>10</td><td>7</td><td>0</td></tr><tr><td>Sanity mutations</td><td>24</td><td>24</td><td>24</td><td>24</td><td>24</td><td>24</td><td>23</td><td>21</td><td>24</td><td>0</td></tr><tr><td>Historical artifacts</td><td>17</td><td>15</td><td>15</td><td>14</td><td>10</td><td>14</td><td>12</td><td>6</td><td>13</td><td>1</td></tr><tr><td>Printed misprints</td><td>7</td><td>7</td><td>6</td><td>2</td><td>4</td><td>4</td><td>1</td><td>5</td><td>6</td><td>0</td></tr></table>

Table 7: Flags on the 271 FLMF seeds, renamings, mutations, historical artifacts, and misprint items.

## E FORMALALIGN THRESHOLD

FormalAlign returns a certainty score, and we flag a statement when its score is at most the chosen threshold. Table 8 reports the threshold sweep on the three labeled datasets. At the comparison threshold of 0.3, the accepted-error rates are 42.1% on FLMF, 35.3% on Beyond Compilation, and 80.0% on ProofNet. Raising the threshold reduces the number of accepted statements but does not consistently reduce the fraction labeled incorrect.

<table><tr><td></td><td colspan="2">FLMF (182 C / 289 I)</td><td colspan="2">Beyond Compilation (245 C / 106 I)</td><td colspan="2">ProofNet (163 C / 204 I)</td></tr><tr><td>Threshold</td><td>Incorrect / accepted</td><td>Flagged / correct</td><td>Incorrect / accepted</td><td>Flagged / correct</td><td>Incorrect / accepted</td><td>Flagged / correct</td></tr><tr><td>0.1</td><td>243 / 409</td><td>8.8%</td><td>100 / 338</td><td>2.9%</td><td>185 / 312</td><td>22.1%</td></tr><tr><td>0.2</td><td>111 /199</td><td>51.6%</td><td>76 /248</td><td>29.8%</td><td>81 / 120</td><td>76.1%</td></tr><tr><td>0.3</td><td>32/ 76</td><td>75.8%</td><td>49 /139</td><td>63.3%</td><td>16 / 20</td><td>97.5%</td></tr><tr><td>0.4</td><td>12 / 22</td><td>94.5%</td><td>21 / 45</td><td>90.2%</td><td>2/2</td><td>100.0%</td></tr><tr><td>0.5</td><td>1/1</td><td>100.0%</td><td>3/8</td><td>98.0%</td><td>0/0</td><td>100.0%</td></tr></table>

Table 8: FormalAlign certainty-threshold sweep on 1,189 labeled pairs. A statement is flagged when its certainty score is at most the threshold. C and I denote the dataset’s correct and incorrect labels. Incorrect / accepted counts incorrect-labeled items among accepted statements. Flagged / correct is the percentage of correct-labeled items that are flagged. The threshold of 0.3 used in the main comparison is shown in bold. The 0 / 0 entry indicates that no statements are accepted. RLM25 reference pairs are not included in this sweep.

## F COST AND RUNTIME

<table><tr><td>Channel</td><td>Items</td><td>API calls</td><td>Cost (USD)</td><td>Cost / item (USD)</td><td>Time / item (s)</td></tr><tr><td>Numerical check</td><td>1,539</td><td>3,961</td><td>314.18</td><td>0.204</td><td>77.0</td></tr><tr><td>Back-translation</td><td>1,539</td><td>3,080</td><td>99.57</td><td>0.065</td><td>29.0</td></tr><tr><td>Vacuity check</td><td>1,539</td><td>0</td><td>0.00</td><td>0.000</td><td></td></tr><tr><td>GPT-5.5 direct</td><td>918</td><td>920</td><td>33.01</td><td>0.036</td><td>13.5</td></tr><tr><td>Opus 4.6 direct</td><td>918</td><td>922</td><td>20.96</td><td>0.023</td><td>14.6</td></tr><tr><td>LeanScorer</td><td>918</td><td>1,840</td><td>66.65</td><td>0.073</td><td>20.5</td></tr><tr><td>GPT-5.5 joint</td><td>918</td><td>61</td><td>12.02</td><td>0.013</td><td>4.2</td></tr><tr><td>Opus 4.6 joint</td><td>918</td><td>61</td><td>5.43</td><td>0.006</td><td>3.1</td></tr></table>

Table 9: API cost and recorded runtime. The 918-item set comprises 200 FLMF, 351 Beyond Compilation, and 367 ProofNet statements. The 1,539-item set also includes 621 RLM25 reference pairs. Costs include retained reservations totaling \$9.64 for 25 requests whose charges remain unresolved. Time per item is the sum of final recorded job durations divided by the item count, with joint-batch time amortized across items; it excludes superseded attempts and is not elapsed wall-clock time. Vacuity runtime is omitted because 1,512 records report result-reuse overhead rather than the original checking time. FormalAlign scored 918 items locally on CPU, with 32.1 minutes of recorded inference time (2.1 seconds per item), excluding model loading.