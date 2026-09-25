# SWE-Prometheus: Measurin<sub>g</sub> En<sub>g</sub>ineerin<sub>g</sub> Governance I<sub>mprovemen</sub>t<sub>s</sub> i<sub>n</sub> R<sub>ea</sub>l<sub>-</sub>W<sub>or</sub>ld R<sub>epos</sub>it<sub>or</sub>i<sub>es</sub>

Jiajun Wu<sup>1,∗</sup> Leixin Sun<sup>1,∗</sup> Zihan Tan<sup>1,∗</sup> Yitao Liu<sup>1</sup> Shuo Li<sup>5</sup> Jiaru Qian<sup>2</sup> Shanghaoran Quan<sup>2</sup> Chuangxin Zhao<sup>4</sup> Yangxu Liao<sup>3</sup> Yang Liu<sup>2</sup> Bin Chong<sup>2,‡</sup> Guancheng Wan<sup>1,‡</sup> <sup>∗</sup> Equal contribution <sup>‡</sup> Corresponding author

Ab<sub>s</sub>t<sub>rac</sub>t<sub>:</sub> L<sub>arge</sub> l<sub>anguage mo</sub>d<sub>e</sub>l b<sub>ase</sub>d <sub>co</sub>di<sub>ng agen</sub>t<sub>s</sub> h<sub>ave ma</sub>d<sub>e su</sub>b<sub>s</sub>t<sub>an</sub>ti<sub>a</sub>l <sub>progress on repos</sub>it<sub>ory-</sub>l<sub>eve</sub>l <sub>so</sub>ft<sub>ware</sub> engineering tas<sup>k</sup>s. Existing repository benchmarks, however, usually start from a human-identified issue and evaluate whether a patch satisfies a functional signal. We present SWE-Prometheus, a benchmark for the broader task of improving repository engineering governance. Each task provides a fixed snapshot and an open-ended objective, requiring the agent to identify risks, prioritize interventions, and verify the resulting changes. SWE-Prometheus evaluates six governance dimensions through paired evidence, clean-environment probes, behavior gates, and two independent teacher ratings of the same evidence. The benchmark contains 60 repositories; ten models are evaluated on a shared 22-repository public subset, where mean Normalized Governance Improvement ranges from 0.0568 to 0.5760 and observed behavior-breakage rates range from 0% to 23%. On a frozen ten-repository batch, a repository-blind template obtains mean NGI 0.272, but its gains concentrate in Tests & CI, Quality Gates, and Documentation; it improves Reproducible Environment and Dependency & Security on none of the repositories. This baseline makes the distinction between adding governance artifacts and producing execution-backed improvements measurable. The no-op condition has median NGI zero and standard deviation 0.073; two teachers agree exactly on 57 of 60 dimension scores for the same no-op evidence. For the two highest conditional-mean systems, common-valid NGI is similar, while full-pool comparisons that include behavior failures favor Kimi-K3. These results show why repository-governance evaluation should report improvement, behavior preservation, evidence quality, and coverage together.

Ñ Home<sub>p</sub>a<sub>g</sub>e õ Hu<sub>gg</sub>in<sub>g</sub> Face § GitHub

## 1<sub>.</sub> I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Large language model based coding agents have made substantial progress on repository-level software engineering. They can search unfamiliar codebases, execute development commands, edit multiple files, and produce end-toend changes. Repository-level benchmarks such as SWE-bench made this progress measurable by pairing a real repository snapshot with a concrete issue and executable validation [1]. Agent scafolds further showed that the interface through which a model searches, edits, and tests a repository can materially afect the outcome [2].

As illustrated in Figure 1, most existing evaluations begin after a human has already identified the problem. The issue statement specifies what the agent should investigate, while tests or a reference patch provide a relatively clear success signal. This is appropriate for targeted functional repair, but it leaves a central part of software engineering unmeasured: making a repository easier to test, maintain, reproduce, and safely evolve. In practice, these properties are distributed across tests, CI, tooling, documentation, project structure, environments, and dependencies. They rarely appear as one localized defect, and their improvement cannot be established from a dif alone.

The resulting gap is between engineering artifacts and engineering outcomes. An agent may add a workflow, a lint configuration, or a test file without making the repository more useful; it may also improve one dimension while introducing dependency drift or behavioral regressions elsewhere. A suitable evaluation must therefore give the agent room to diagnose the repository, compare evidence before and after intervention, and verify that improvements survive a clean execution environment.

To address this gap, we introduce SWE-Promet<sup>h</sup>eus, a benchmark for autonomous repository-level engineering governance. Each task starts from a fixed repository snapshot and a general retrofit objective. The agent is not given a defect list, base score, hidden-test description, or target patch. It must inspect the repository, identify consequential gaps, prioritize interventions, and implement useful changes under a limited budget. The evaluator then reconstructs the base and treated states, executes the same probes in a clean environment, and applies behavior gates to identify unsafe interventions. SWE-Prometheus consequently evaluates whether an agent can leave a repository in a more trustworthy engineering state.

![](images/5f33edf9219a5e249025280a4e7f0ba9be9e7b9611529e6152174bc6eb89158c.jpg)

![](images/65890d20f1a757f7c4f6604394bc57cfb644010f3069aafd30de896af211212a.jpg)  
Figure 1: Motivation for SWE-Prometheus. Prior issue-level SWE benchmarks hand the agent a localized defect and score a patch against target tests, leaving test and CI readiness, clean-environment behavior verification, quality and maintainability outside the patch target, and documentation, reproducibility, dependency, and security health unevaluated. SWE-Prometheus replaces the issue with a governance brief, requires the agent to diagnose and retrofit the repository itself, and scores a paired base-to-treated report produced by a clean-environment verifier.

The benchmark covers six complementary governance dimensions: Tests & CI, Code Quality Gates, Documentation & Collaboration, Structure & Maintainability, Reproducible Environment, and Dependency & Security Health. Each dimension is scored from executable evidence and structured judgments, while applicability, unavailable evidence, gate strength, and behavior status remain explicit. The current release contains 60 real repositories, including 22 public instances shared across ten model rollouts and 38 private instances reserved for internal evaluation. On this shared public subset, mean Normalized Governance Improvement ranges from 0.0568 to 0.5760, while observed behavior-breakage rates range from 0% to 23%.

The auxiliary study shows how to interpret these diferences. A repository-blind template reaches mean NGI 0.272, but its gains are concentrated in D1–D3; it improves D5 and D6 on none of the ten repositories. Patch-level evidence makes this boundary concrete: on Eficient-WAM, the template’s workflow is configured to run Ruf and pytest, but its only test is assert True; the test passes while the separate Ruf probe still reports 117 errors. The baseline therefore calibrates how much score can come from standard governance artifacts and highlights why execution and failure exposure matter. The no-op condition has median NGI zero and standard deviation 0.073, while two teacher ratings agree exactly on 57 of 60 scores for the same no-op evidence. A three-pass maintainer review of the same sample of behavior-preserved patches found the changes useful and no new functional problems. Finally, the comparison between Kimi-K3 and GLM-5.3-Flash shows that improvement conditional on a successful patch and reliability across the full repository pool are distinct outcomes. Together, these controls make the benchmark’s evidence requirements and interpretation explicit.

In summary, our contributions are threefold:

• We identify repository-level engineering governance as a distinct evaluation setting that complements issuelevel functional repair, and formulate it as an open-ended Repository Retrofit task without a defect list or oracle patch.

• We introduce SWE-Prometheus, a repository-centered benchmark with six governance dimensions, paired base/treated evidence, clean-environment probes, behavior gates, public/private splits, and machine-readable release records.

• We benchmark ten coding agents and establish diagnostic controls—including no-op, mechanical, rule-based, matched, and gate-strength analyses—that separate governance improvement from template efects, behavior risk, and verifier limitations.

## 2<sub>.</sub> R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

## 2<sub>.</sub>1<sub>.</sub> C<sub>o</sub>d<sub>e an</sub>d S<sub>o</sub>ft<sub>ware</sub> E<sub>ng</sub>i<sub>neer</sub>i<sub>ng</sub> B<sub>enc</sub>h<sub>mar</sub>k<sub>s</sub>

The evaluation of code-generation capabilities has evolved from function-level correctness to increasingly realistic repository-level challenges. SWE-bench established the now-standard setting in which an agent receives a real issue, a repository snapshot, and executable tests [1]. Its focus on real GitHub repositories exposed the dificulty of understanding and modifying an unfamiliar codebase rather than completing an isolated function. SWE-bench Verified subsequently added human validation of task solvability and test quality [3], making the benchmark’s success signal more reliable.

Earlier code-generation work established complementary function-level settings: HumanEval evaluates synthesized programs with unit tests, while MBPP covers short, crowd-sourced programming problems; surveys of machine learning for code situate these tasks within a broader history of code representation and generation [4–6]. These settings are useful for measuring local synthesis, but do not test repository-wide governance or maintenance. Other repository benchmarks broaden the repair target to visual issues and multilingual or multimodal issue resolution, including CodeV, OmniGIRL, and MM-IssueLoc [7–9]. Hardware repair and architectural code-smell repair further extend issue resolution to specialized targets [10, 11]; they remain issue-driven settings rather than open-ended assessments of repository readiness.

The repository-level setting has since expanded in several directions. SWE-bench-java, Multi-SWE-bench, and SWE-PolyBench extend issue resolution to Java, multiple programming languages, and broader repository distributions [12–14]. SWE-bench Multimodal and related visual benchmarks incorporate screenshots and other non-textual evidence [15], while SWE-bench-Live and SWE-rebench study task freshness, automated collection, and contamination-aware evaluation [16, 17]. SWE-bench Science and Rust-SWE-bench further suggest that domainspecific repositories and ecosystems expose failure modes hidden by a narrow benchmark distribution [18, 19].

These benchmarks share several design principles with SWE-Prometheus: fixed revisions, isolated execution, submitted patches, and executable checks. Their primary endpoint, however, remains the successful resolution of a known defect. SWE-Prometheus changes the object of evaluation from repairing an identified issue to diagnosing and improving the engineering state of a repository.

## 2.2. Software En<sub>g</sub>ineerin<sub>g</sub> A<sub>g</sub>ents

The development of autonomous agents for repository-level software engineering has progressed alongside the benchmark ecosystem. SWE-agent shows that a purpose-built agent-computer interface can support efective search, editing, and testing [2]. OpenHands provides a general platform for software-development agents [20], while Agentless demonstrates that a deliberately simple localization, repair, and validation pipeline can remain competitive [21]. These studies highlight that agent performance depends not only on the underlying language model but also on how repository interaction is structured.

More general agent research studies reasoning interleaved with actions, learned tool use, and verbal feedback across interaction steps [22–24]. WebArena evaluates agents in realistic browser-based environments, while OmniBench measures broad virtual-agent capabilities in a diferent task and environment scope [25, 26]. Self debugging work similarly examines how execution feedback can guide code revision [27]. These methods inform the design of interactive coding agents, but are not themselves repository-governance benchmarks.

Other systems isolate complementary capabilities. RepoBench and RepoCoder focus on repository-level retrieval and code completion rather than issue resolution [28, 29]. SWE-Explore treats repository exploration as a measurable capability [30], and RepairAgent and RepoAgent investigate autonomous program repair and repository-level development workflows [31, 32]. Training-oriented work such as SWE-Gym uses real-world software engineering tasks to improve agent behavior [33].

Visual-language research also studies multi-image perception and reasoning, with benchmarks and instructiontuning resources such as MANTIS, MIRB, MuirBench, and OMIBench [34–37]. These are adjacent to multimodal repository tasks because they examine visual evidence, but their general multi-image questions are not interchangeable with software issue localization or repository-level improvement.

Together, these systems motivate our separation of reconnaissance, intervention, and verification. They generally assume that the target issue, repair objective, or evaluation signal is already specified. SWE-Prometheus instead asks the agent to discover the engineering target itself, prioritize among repository weaknesses, and justify the intervention with evidence.

## 2<sub>.</sub>3<sub>.</sub> Evaluation Be<sub>y</sub>ond Patch Correctness

Passing the original test suite is necessary for a useful patch, but it is not suficient evidence of broader software engineering quality. RepoExec studies repository-level executable evaluation [38], while SWT-Bench emphasizes validating generated fixes against real-world behavior [39]. EvalPlus shows that strengthening test suites can substantially change conclusions about functional correctness [40]. These results point to the importance of evaluation signals that are dificult to satisfy through narrow test overfitting.

Testing research provides additional tools for assessing evidence quality. Mutation testing estimates whether a test suite detects meaningful behavioral changes [41]. Empirical studies of flaky tests show that instability in the test infrastructure can itself contaminate engineering conclusions [42]. Behavioral testing frameworks such as CheckList likewise argue for testing a range of observable behaviors rather than relying on a single aggregate score [43].

Classical diagnosis provides a further conceptual basis for separating observed failures from candidate explanations: Reiter formalizes diagnosis from first principles, and logical-abduction methods have been applied to diagnosing and correcting source-code design inconsistencies [44, 45]. Our setting difers in that the agent must proactively identify governance gaps, rather than explain a supplied failure or inconsistency.

Our behavior gates and gate-strength labels build on this line of work by treating preservation evidence as a first-class outcome. A treated repository is not credited merely because a new check passes: the evaluation also asks whether existing behavior remains valid, whether the evidence is strong enough, and whether the run is reproducible.

## 2.4. Software Quality and Engineering Governance

The broader software quality literature ofers useful concepts for describing repository health. ISO/IEC 25010 organizes quality around maintainability, reliability, security, and related properties [46]. Complexity metrics [47, 48], technical-debt research [49, 50], and empirical studies of continuous integration [51, 52] provide practica measurement primitives for these dimensions.

Modern repository governance also depends on dependency scope, supply-chain provenance, and reproducible environments. OpenSSF Scorecard, SLSA, and NIST SSDF provide practical security and supply-chain guidance [53– 55]. Empirical work further shows that dependency alerts do not all have the same production relevance, making naive alert counts an inadequate proxy for risk [56].

These approaches typically audit a property, metric, or development practice in isolation. They do not evaluate whether an autonomous agent can identify the most consequential repository gaps, coordinate changes across several dimensions, and preserve existing behavior. In SWE-Prometheus, quality and governance tools are evidence probes inside a paired base-to-treated evaluation, not substitutes for the benchmark objective. The benchmark therefore evaluates whether an agent can turn engineering practices into durable repository improvements rather than merely adding configuration files.

## 2<sub>.</sub>5<sub>.</sub> R<sub>e</sub>li<sub>a</sub>bl<sub>e</sub> B<sub>enc</sub>h<sub>mar</sub>k C<sub>ons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d J<sub>u</sub>d<sub>ge</sub> C<sub>a</sub>lib<sub>ra</sub>ti<sub>on</sub>

Benchmark design determines which capabilities can be measured reliably. Weak tests, leakage, incomplete curation, and unstable environments can all change model conclusions. Benchmark construction studies show that automated setup and broader repository coverage can produce task distributions that difer materially from popular curated suites [57]. Contamination-limited evaluation and benchmark governance therefore require explicit provenance, release controls, and reproducible execution [58–60].

![](images/8b3b143392048e19acec499a16d80b90f08f741f069ba837ba8dc59d1edc38d5.jpg)  
Figure 2: Dataset and evaluation scope. The benchmark covers six governance dimensions and reports the public/private split, behavior-gate composition, and completed auxiliary-batch statistics.

Subjective dimensions introduce a second reliability challenge. LLM-as-a-judge methods can be useful, but they require calibration and agreement checks rather than unqualified trust [61, 62]. SWE-Prometheus follows these principles by retaining failed runs, measuring characterization-test strength through mutation testing, separating public and private data, and preserving per-run evidence. The remaining gap is an integrated evaluation in which the agent must discover the engineering target and improve it under realistic constraints while the evaluator checks both the intervention and the repository’s continued usefulness.

## 3. Dataset Overview

## 3<sub>.</sub>1<sub>.</sub> T<sub>as</sub>k f<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub> <sub>an</sub>d <sub>scope</sub>

Each instance is a fixed snapshot of a real GitHub repository identified by repo and base\_commit. The agent receives a general instruction:

Inspect this repository and improve its engineering readiness across the applicable governance dimensions. Preserve existing behavior and public interfaces. Make useful changes for future users and maintainers, and verify important claims with executable checks.

The task provides no issue list, base score, hidden test, or reference patch. Multiple governance strategies may be valid. The agent can inspect source code and public documentation, execute development tools, edit project files, and submit a unified patch, but cannot access evaluator evidence or other model results.

Governance <sup>d</sup>imensions. The six dimensions are scored from 1 to 5; a score of 4 denotes acceptable governance for the repository type and task scope. Figure 2 visualizes their scope: Tests & CI, Code Quality Gates, Documentation & Collaboration, Structure & Maintainability, Reproducible Environment, and Dependency & Security. Configuration presence alone is never suficient evidence of improvement; each dimension must be supported by executable or structured evidence.

Evi<sup>d</sup>ence <sup>l</sup>oop. The evaluator reconstructs the base repository, applies the agent patch to an independent treated snapshot, and runs the same installation, test, quality, complexity, dependency, and security probes on both states. Commands, exit codes, logs, environment versions, timeouts, and unavailable evidence are retained.

Each fixed base/treated evidence record is scored twice by independent teacher models using the same rubric and evidence. The released record retains both judgments, their aggregate, and their score spread. This provides a direct same-evidence inter-rater check; separately, the no-op condition measures end-to-end variation when an unchanged repository is processed through evidence collection and scoring.

Be<sup>h</sup>avior gate. Characterization tests are validated on the base snapshot and then executed on the treated snapshot. A failed treated verification is labeled behavior\_broken; it is excluded from valid NGI aggregation bu retained in behavior-risk analysis. Mutation testing labels a gate as detected, blind, vacuous, or none.

Metrics. For instance i and dimension d, raw improvement is $\Delta _ { i , d } = s _ { i , d } ^ { t r e a t e d } - s _ { i , d } ^ { b a s e }$ . Normalized Governance Improvement is

$$
N G I _ { i } = \frac { 1 } { | D _ { i } | } \sum _ { d \in D _ { i } } \frac { s _ { i , d } ^ { t r e a t e d } - s _ { i , d } ^ { b a s e } } { 5 - s _ { i , d } ^ { b a s e } } ,
$$

where $D _ { i }$ contains defined and scorable dimensions. A dimension with base score 5 receives no improvement reward but is checked for regression. We additionally report treated scores, valid and broken rates, no-regression rate, strict success, verified claim rate, and resource usage.

## 4. Dataset Creation

## 4.1. Re<sub>p</sub>ositor<sub>y</sub> selection and screenin<sub>g</sub>

SWE-Prometheus is intentionally repository-centered rather than issue-centered. We first collect candidate repositories with evidence of active development and at least one plausible governance gap. A candidate is retained only when (i) the base commit can be reconstructed, (ii) the project has meaningful executable behavior, (iii) a clean evaluation environment can be built without private credentials, (iv) at least two governance dimensions are applicable, and (v) the task can be described without prescribing a particular patch. Screening records language, build system, test entry points, dependency manager, license, commit age, and external-service requirements.

Discovery signals such as absent workflows, stale metadata, weak tests, and unpinned dependencies are used for high-recall candidate collection, not as final scores. The final score is based on executable evidence. Negative evidence is retained explicitly: an unavailable command, unsupported platform, or inapplicable dimension is recorded rather than silently treated as a zero-quality score.

## 4<sub>.</sub>2<sub>.</sub> E<sub>v</sub>id<sub>ence</sub> <sub>co</sub>ll<sub>ec</sub>ti<sub>on</sub> <sub>an</sub>d t<sub>as</sub>k <sub>pac</sub>k<sub>ag</sub>i<sub>ng</sub>

For every retained repository, the construction process records a fixed repository identifier and base commit, a governance brief, applicable dimensions, baseline evidence, characterization commands, and environment metadata. The governance brief states the engineering objective without naming a defect or prescribing a patch. This preserves room for multiple valid retrofit strategies while keeping the task reproducible.

The evaluator reconstructs the base repository, runs installation and development probes, and stores commands, exit codes, logs, tool versions, timeouts, and unavailable evidence. The same evidence schema is used after treatment. This paired design prevents a configuration file from being counted as an improvement unless the corresponding command or structured check provides useful evidence.

## 4.3. Execution <sub>p</sub>rotocol

Every rollout starts from a fresh copy of the pinned base commit. The agent receives the same task instruction and repository snapshot for a given comparison set. The harness records tool calls, changed files, patch size, wall-clock time, token usage when exposed by the scafold, and termination reason. After exit, the harness validates the patch and applies it to a second clean copy. Evaluation runs installation, characterization behavior, governance probes, and post-hoc artifact validation in that order.

Each command has a timeout, an exit-code policy, and an evidence schema. Logs are retained with tool versions and environment identifiers. Monetary cost is reported only when provider accounting is available; it is never imputed from incomplete accounting. This protocol separates infrastructure failures from engineering failures and makes each reported score auditable.

## 4<sub>.</sub>4<sub>.</sub> B<sub>e</sub>h<sub>av</sub>i<sub>or ga</sub>t<sub>es an</sub>d f<sub>a</sub>il<sub>ure</sub> t<sub>axonomy</sub>

Characterization tests are validated on the base snapshot and then executed on the treated snapshot. A failed treated verification is labeled behavior\_broken; it is excluded from valid NGI aggregation but retained in behaviorrisk analysis. Mutation testing labels a gate as detected, blind, vacuous, or none. The gate-strength label is reported alongside every behavior outcome because a zero-breakage result from a weak gate should not be interpreted as strong safety evidence.

We distinguish infrastructure failure, environment/dependency failure, invalid or empty patch, behavior regression, preserved behavior without governance improvement, and preserved behavior with positive improvement. A rollout may receive a primary failure label and secondary evidence flags. For example, a patch may preserve characterization behavior while introducing a vacuous CI gate; it should not be counted as a clean success merely because tests pass.

## 4.5. Scoring and judge calibration

Executable probes provide primary evidence for tests, CI, environment, and dependency dimensions. Documentation and maintainability use dimension-specific rubrics rather than a holistic impression. Each candidate score is accompanied by positive evidence, negative evidence, applicability, and a confidence flag. Two independent judgments are preferred; disagreements are adjudicated only after initial labels are preserved.

The calibration set is scored by independent evaluators who do not know model identity. We report agreement, adjudication rate, and sensitivity to rubric anchors, together with judge-family and prompt-order checks. These safeguards are part of the release protocol and are distinguished from completed benchmark results.

## 5. Results

## 5<sub>.</sub>1<sub>.</sub> E<sub>xper</sub>i<sub>men</sub>t<sub>a</sub>l <sub>se</sub>tti<sub>ng an</sub>d d<sub>a</sub>t<sub>ase</sub>t

The current SWE-Prometheus release contains 60 real repository instances: 22 public instances shared by all current models and 38 private instances reserved for internal evaluation. Ten models have completed one rollout on the public subset, yielding 220 public model-instance pairs. The public subset is a shared comparison set rather than a random sample of all repositories.

Candidate repositories were discovered using signals such as missing or failing CI, thin tests, absent quality gates, incomplete collaboration documentation, unpinned dependencies, and maintenance gaps. These signals support high-recall discovery and stratification, while screening confirms meaningful functionality, a stable base commit, plausible governance headroom, and evaluation without private credentials or paid services.

The evaluation follows a paired base-to-treated design. For every model-instance pair, the harness first records baseline evidence, then applies the submitted patch to a clean copy and reruns the same probes. Characterization behavior is checked before governance scores are aggregated. We report valid and behavior-broken outcomes separately, retain unavailable evidence instead of converting it into a zero, and use the shared public subset for the primary cross-model comparison. Appendix D reports the detailed aggregate tables; the release package contains the machine-readable per-instance records used to reproduce them.

## 5<sub>.</sub>2<sub>.</sub> A<sub>ux</sub>ili<sub>ary con</sub>diti<sub>ons</sub>

Model averages alone cannot separate capability from measurement artifact, so we additionally run a compact auxiliary study on ten repositories sampled from the public set with fixed strata for repository size, baseline quality, and gate availability. Every condition uses the same base snapshots and the same scoring pipeline, and al conditions reuse stored rollouts without issuing new model calls.

<table><tr><td>Agent system</td><td>Scaffold</td><td>Valid</td><td>Broken</td><td>Breakage (%)</td><td>NGI mean</td></tr><tr><td>GPT-5.6-Sol</td><td>pi</td><td>21</td><td>1</td><td>4.5</td><td>0.2956</td></tr><tr><td>Claude-Opus-5</td><td>Claude Code</td><td>18</td><td>4</td><td>18.2</td><td>0.5293</td></tr><tr><td>Kimi-K3</td><td>pi</td><td>21</td><td>1</td><td>4.5</td><td>0.5760</td></tr><tr><td>GLM-5.3-Flash</td><td>pi</td><td>17</td><td>5</td><td>22.7</td><td>0.5760</td></tr><tr><td>Qwen3.8-Max</td><td>pi</td><td>18</td><td>4</td><td>18.2</td><td>0.4630</td></tr><tr><td>GLM-5.2</td><td>pi</td><td>20</td><td>2</td><td>9.1</td><td>0.4438</td></tr><tr><td>DeepSeek-V4-Pro</td><td>pi</td><td>18</td><td>4</td><td>18.2</td><td>0.3803</td></tr><tr><td>GLM-5.3</td><td>pi</td><td>19</td><td>3</td><td>13.6</td><td>0.3198</td></tr><tr><td>DeepSeek-V4-Flash</td><td>pi</td><td>22</td><td>0</td><td>0.0</td><td>0.2083</td></tr><tr><td>MiniMax-M3</td><td>pi</td><td>22</td><td>0</td><td>0.0</td><td>0.0568</td></tr></table>

Ta<sup>bl</sup>e 1: Public shared-subset results. Valid and broken counts are out of 22 instances. NGI is aggregated only over valid runs, which preserve characterization behavior and contain scorable evidence. Claude used Claude Code; the other models used pi.

The batch includes a no-op condition that pushes an empty patch through the full pipeline to measure end-toend variation; a mechanical retrofit that adds a fixed governance template without inspecting the repository; a rule-based retrofit that first detects which artifacts are missing and adds only those, generating an import smoke test rather than a vacuous assertion; and the existing model rollouts evaluated on the same instances. A separate behavior-gate ablation uses all 60 instances because the public subset contains no none gate cases. Appendix C.5 documents the strata, the condition matrix, and the patch contents of each baseline.

## 5.3. Governance im<sub>p</sub>rovement varies widel<sub>y</sub> across a<sub>g</sub>ents

Figure 3 reports the headline outcome on the public shared subset. Mean NGI spans roughly an order of magnitude, from 0.0568 to 0.5760, while measured behavior breakage spans 0% to 22.7%. Kimi-K3 and GLM-5.3-Flash obtain the same highest valid-run mean NGI, but they sit at opposite ends of the safety axis: Kimi-K3 breaks behavior on one of 22 instances while GLM-5.3-Flash breaks it on five. The equal conditional means therefore do not imply equal overall reliability. MiniMax-M3 produces little normalized improvement despite zero observed behavior breaks. The contrast is the reason improvement, safety, and absolute treated quality are reported as separate axes rather than folded into one leaderboard number. The companion coverage view separates valid runs from behavior-broken runs before any NGI aggregation, making clear that a high score supported by fewer valid runs should be interpreted cautiously.

## 5<sub>.</sub>4<sub>.</sub> R<sub>e</sub>li<sub>a</sub>bilit<sub>y-aware compar</sub>i<sub>son o</sub>f th<sub>e</sub> t<sub>wo</sub> l<sub>ea</sub>di<sub>ng sys</sub>t<sub>ems</sub>

The valid-run mean is useful for measuring the quality of successful interventions, but it is conditional on behavior preservation. We therefore report two complementary comparisons for the two systems with the highest conditional mean. On the 17 repositories where both Kimi-K3 and GLM-5.3-Flash preserve characterization behavior, thei mean NGI values are 0.547 and 0.576, respectively. The paired diference is −0.029 for Kimi-K3, with Kimi winning on 6 repositories, GLM-5.3-Flash on 10, and one tie. This diference is smaller than the empirical no-op standard deviation of 0.073 and does not support a reliable improvement advantage for either system.

To include delivery reliability, we also report a coverage-weighted NGI, defined as the mean over all 22 public repositories after assigning a behavior-broken run zero verified improvement. This is not a replacement for the conditional NGI; it asks a diferent practical question: how much verified improvement does a system deliver over the complete evaluation pool? Under this definition, Kimi-K3 obtains 0.550 and GLM-5.3-Flash obtains 0.445. We also examine a stronger penalty by assigning each behavior-broken run an NGI of −1; under this policy the means are 0.504 and 0.218, respectively. Finally, assigning the score-theoretic lower bound of −3 gives 0.413 for Kimi-K3 and −0.237 for GLM-5.3-Flash. The lower bound follows from the rubric’s 1–5 scale and NGI normalization for an applicable dimension (base score 4, treated score 1). Across these full-pool policies Kimi-K3 remains first, although the ordering of the other systems changes substantially (Appendix D). Conditional means alone do not establish a reliable improvement diference between Kimi-K3 and GLM-5.3-Flash; their clearer distinction is delivery reliability, with one versus five behavior-broken repositories.

![](images/39a78345d62219132de4cd185d2f01aa427f387dfb97f592dceb8bb8321e1d2e.jpg)  
Mean NG (a) Trade-off between mean NGI and behavior risk

![](images/af24cfc638eed8cd36d3dcf45729d9521b88dd01072d8bf15201740d82bf36e5.jpg)  
(b) Mean NGI ranking.

![](images/b0bea132cf213a020cd727a648d926a2aaca9957918ce878466c094e8e8898bf.jpg)  
(c) Valid and behavior-breaking runs.

Figure 3: Public model outcomes and behavioral validity. (a) Trade-of between mean NGI and behavior risk. (b) Conditional valid-run NGI comparison, not a reliability-aware ranking. (c) Per-model decomposition of valid and behavior-breaking runs across the same 22-instance pool. Claude-Opus-5 ran under Claude Code while the other nine models used pi.
<table><tr><td>Comparison</td><td>Repos.</td><td>Kimi</td><td>GLM-F</td><td>Difference</td><td>Interpretation</td></tr><tr><td>Common valid runs</td><td>17</td><td>0.547</td><td>0.576</td><td>-0.029</td><td>Below no-op SD</td></tr><tr><td>Full pool, broken = 0</td><td>22</td><td>0.550</td><td>0.445</td><td>+0.105</td><td>Kimi more reliable</td></tr></table>

Ta<sup>bl</sup>e 2: Reliability-aware comparison of Kimi-K3 and GLM-5.3-Flash. “GLM-F” denotes GLM-5.3-Flash. The first row compares only repositories valid for both systems; the second retains all 22 repositories and assigns zero verified improvement to a behavior-broken treatment.

## 5<sub>.</sub>5<sub>.</sub> I<sub>ns</sub>t<sub>ance-</sub>l<sub>eve</sub>l di<sub>spers</sub>i<sub>on</sub> i<sub>s</sub> l<sub>arge re</sub>l<sub>a</sub>ti<sub>ve</sub> t<sub>o</sub> th<sub>e mo</sub>d<sub>e</sub>l <sub>sprea</sub>d

Aggregates hide how unevenly governance improvement is distributed. Figure 6 shows the per-instance NGI distribution behind each mean. Every model spans a wide interquartile range, several have valid instances at or below zero, and mean and median diverge substantially for models whose gains are concentrated in a few repositories—Claude-Opus-5 in particular has a median of 0.722 against a mean of 0.457 on the auxiliary batch, indicating a right-skewed profile driven by a subset of instances. Because each model-instance pair has been run once, this dispersion is a joint efect of repository heterogeneity, rollout variability, and scorer noise, and it is what limits the resolution of the current comparison (Section 5.10).

## 5<sub>.</sub>6<sub>.</sub> Di<sub>ag</sub>n<sub>os</sub>ti<sub>c</sub> r<sub>esu</sub>lt <sub>a</sub>n<sub>a</sub>l<sub>ys</sub>i<sub>s</sub>

This section reports the auxiliary conditions. Each subsection asks whether a specific part of the headline result survives a control.

## 5.7. Non-a<sub>g</sub>ent controls and no-o<sub>p</sub> measurement variation

The no-op condition has mean NGI −0.009 and median 0.000, with five of ten instances receiving a non-zero score change and a standard deviation of 0.073. To measure judge disagreement separately, we compare the two independent teacher ratings on the same no-op evidence: they agree exactly on 57 of 60 repository-dimension scores, with non-zero disagreement in only three cells and a mean absolute score spread of 0.067 across the 60 cells. The median no-op NGI of zero indicates no systematic reward for an unchanged repository; the 0.073 standard deviation captures end-to-end variation, including evidence collection and scoring, rather than isolating scorer-only variance. These results show high agreement on fixed evidence alongside measurable pipeline variation. Diferences of this size should be interpreted cautiously. Drift occurs in Structure & Maintainability (three instances), Documentation, and Reproducible Environment (two each). The latter dimension is sensitive to installation evidence, so runtime variability is one plausible contributor; the structural drift on an unchanged dif is more directly consistent with judge variability. The no-op results do not identify the exact source of every changed score.

The two template conditions sit inside the model range rather than below it. The mechanical baseline obtains mean NGI 0.272 and the rule-based baseline 0.254, both above DeepSeek-V4-Flash and MiniMax-M3. The perdimension decomposition clarifies what these scores represent: both baselines improve D1, D2, and D3 on nearly every repository, but improve D5 and D6 on none and D4 on only 20% of repositories. In this batch, they capture the low-cost benefit of adding governance artifacts, while the measured evidence in D5 and D6 does not improve under these repository-blind or rule-based interventions. The ordering between the two baselines is itself informative: the deterministic rule baseline, which conditions its patch on detected repository state, does not outperform a fixed template that ignores the repository entirely. Their separation, 0.018, is small relative to the observed no-op variation.

The patch-level evidence shows both why the template earns credit and where that credit stops. On Eficient-WAM, the mechanical template adds a GitHub Actions workflow that runs Ruf and pytest, plus a smoke test whose only assertion is assert True. The evaluator collects and passes that one test, but the same treated snapshot still reports 117 Ruf errors, no recognized install/build entry point, and no dependency manifest. This is a concrete example of artifact presence improving the test/CI-facing score without demonstrating meaningful behavior coverage or reproducible installation; it is consistent with the template’s zero improvement on D5 and D6. We therefore interpret its NGI of 0.272 as a useful stress test of the rubric’s artifact-facing dimensions, not as evidence that a repository-blind patch delivers the same engineering value as a targeted intervention.

## 5<sub>.</sub>8<sub>.</sub> M<sub>a</sub>t<sub>c</sub>h<sub>e</sub>d <sub>compar</sub>i<sub>son aga</sub>i<sub>ns</sub>t th<sub>e</sub> t<sub>emp</sub>l<sub>a</sub>t<sub>e</sub> b<sub>ase</sub>li<sub>nes</sub>

Comparing means across conditions conflates repository dificulty with agent capability, because an agent system and a baseline may succeed on diferent repositories. We therefore compare each agent system to the stronger of the two templates per instance, counting an instance as a win only when the gap exceeds the empirical no-op noise floor. Figure 4 reports the resulting outcome switches.

The matched view is considerably less flattering than the mean ranking. Only Kimi-K3 (9 wins, 1 loss) and GLM-5.3-Flash (8–1) beat the template on a clear majority of repositories. Claude-Opus-5 and Qwen3.8-Max are close to even at 5–4 and 4–4. GPT-5.6-Sol is indistinguishable from the template on half of the batch, and the two weakest models are behind it: DeepSeek-V4-Flash loses on six of ten repositories and MiniMax-M3 on seven. In other words, most of the middle of the agent-system comparison is not reliably doing better, repository by repository, than a patch that never reads the repository.

![](images/29d40b567c28490f341a97d48da1efabe4acbcc78dadd0770b88370112606c99.jpg)  
Figure 4: Matched per-repository wins against the stronger template baseline. Each bar shows the number of repositories on which a model outperforms the stronger of the two template baselines on the frozen 10-repository batch. A majority requires more than 5 wins. Only Kimi-K3 and GLM-5.3-Flash clearly outperform the template on a majority of repositories.

## 5<sub>.</sub>9<sub>.</sub> V<sub>er</sub>ifi<sub>er qua</sub>lit<sub>y con</sub>t<sub>ro</sub>l

A behavior gate is only as informative as it is discriminative, so we stratify outcomes by the mutation-testing label. Figure 5a shows the result over all 60 instances and ten models. In this mutation-tested sample, measured breakage falls as gate strength degrades—13% under detected gates, 8% under blind gates, and 0% under vacuous gates—while median NGI moves by only 0.06 across the same strata. This pattern is consistent with weaker gates missing regressions rather than proving that agents are safer. In particular, zero observed breakage under a gate known to miss the injected mutations is not evidence of preserved behavior. Headline breakage rates should therefore be reported with gate-strength breakdowns.

As a complementary validity check, we reviewed the same fixed sample of representative patches that had passed their behavior checks in three review passes, each time examining the patch together with its base and treated evidence. Across all three passes, the changes were judged useful for repository maintenance, and no new functional problems were identified. Repeating the review on the same sample checks whether the qualitative finding is stable rather than dependent on a single inspection.

The gate ablation itself is more nuanced than a simple inflation correction. Removing the gate and scoring every instance changes the median by at most 0.021 in either direction: it is positive for Kimi-K3 (+0.014), Qwen3.8- Max (+0.021), and GPT-5.6-Sol (+0.021), negative for GLM-5.3-Flash (−0.017), GLM-5.2 (−0.008), and GLM-5.3 (−0.007), and unchanged for the rest; the model ordering is identical with the gate on and of. Instances caught by the gate therefore tend to score near or below their model’s average rather than above it, which suggests that behavior breakage is more often a symptom of a botched intervention than the price of an aggressive high-scoring one. The gate remains necessary—it identifies which scores are untrustworthy—but its efect should be reported as a sensitivity analysis, not assumed to be a monotone bias.

## 5<sub>.</sub>10<sub>.</sub> S<sub>ens</sub>iti<sub>v</sub>it<sub>y o</sub>f th<sub>e curren</sub>t d<sub>es</sub>i<sub>gn</sub>

Given the dispersion in Figure 6, we ask what diference the current design can actually resolve. Across all 45 model pairs on their common-valid public instances, the paired instance-level NGI diference has median standard deviation 0.270 over a median of 18 shared instances. Figure 5b converts this into detection power. At 80% power and α = 0.05, the public subset resolves a true mean diference of about 0.19, and the ten-repository auxiliary batch about 0.27.

![](images/5e57cc9132795b74f4cba652630cf724dc252716951ab63b55bd48d3833c3ca5.jpg)

![](images/b3c02db3ef0e7b846ef187c86671adc017bf0247d5000f684e3c15e09a1122cc.jpg)

![](images/16f29f2b2e24f844accca71f211c8c2b829fc96ff8e418bfe7b7434a5413eb13.jpg)  
Breakage falls monotonically as gates weaken while governance scores barely move: weak gates fail to observe unsafe patches rather than observing safer agents.

(a) Verifier quality control by mutation-testing gate strength.  
![](images/5b6b74d02da41de318f91ad323f0009d9c52ec13b1c76db286b42fa790c7db50.jpg)

![](images/83b9bb359b552e7bb8f83610bc5ede271da1012748a5d4d9a068314a56c2705a.jpg)  
Paired instance-level SD is estimated from all 45 model pairs on their common-valid public instances (median n = 18).  
(<sup>b</sup>) Resolution of the current evaluation design.  
Figure 5: Verifier quality and statistical resolution. Weak gates observe fewer unsafe patches, while the current public subset has a finite resolution for detecting NGI diferences.

This has a direct consequence for how Figure 3 should be read. Adjacent models in the ranking—Kimi-K3 and GLM-5.3-Flash, or GLM-5.2 and DeepSeek-V4-Pro—are separated by less than the resolution limit and are not distinguished by this instrument. The result supports a coarse grouping into a high band, a middle band, and a low band rather than a ten-way ordering. Reaching a resolution of 0.10 requires about 110 paired observations, or roughly five repeated rollouts per model-instance pair on the current public subset; this is the concrete cost of the repeated-seed experiment we defer to future work.

## 5<sub>.</sub>11<sub>.</sub> P<sub>er-</sub>di<sub>mens</sub>i<sub>on</sub> d<sub>ecompos</sub>iti<sub>on</sub>

Aggregate NGI can hide a condition that improves one dimension while neglecting another. Figure 6 decomposes improvement across the six governance dimensions. Two patterns are consistent across models. Tests & CI improves on the large majority of valid runs for every model except MiniMax-M3, whereas Structure & Maintainability is the weakest or joint-weakest dimension for eight of ten models, falling to 19% for GPT-5.6-Sol and 14% for MiniMax-M3. Only Kimi-K3 (76%) and Claude-Opus-5 (67%) reach it on a clear majority of their valid runs, and Claude-Opus-5 is notable for reaching it while scoring lower than several competitors on the configuration-facing dimensions—the profile of a model that edits structure rather than adding files.

![](images/fb0638dfc6e46a4763070f00d09b93adb6abc8fecba7b5e37fe58b80bc4f868e.jpg)  
(a) Instance-level NGI distributions.

![](images/c74d46516eeceed3bd32beb3ce8cfa5b87e07b71a62497d42952ec4ae8b2c006.jpg)  
(<sup>b</sup>) Per-dimension improvement coverage.  
Figure 6: Model heterogeneity and dimension coverage. The paired views show both the spread of repository-level gains and the governance dimensions improved by each model.

The same decomposition applied to the template baselines helps explain their standing in Section 5.8. On this batch, both templates improve Tests & CI, Code Quality Gates, and Documentation & Collaboration on nearly every instance, and improve Reproducible Environment and Dependency & Security Health on none, with Structure & Maintainability at 20%. This pattern is consistent with a distinction between dimensions that can receive credit from configuration or content and dimensions whose evidence depends on repository execution or dependency analysis. It does not imply that configuration-facing dimensions are inherently superficial; rather, the Eficient-WAM example shows why file presence alone is weaker evidence than successful, meaningful checks.

## 5.12. Ke<sub>y</sub> insi<sub>g</sub>ht: <sub>g</sub>overnance im<sub>p</sub>rovement is not re<sub>p</sub>ositor<sub>y</sub> dia<sub>g</sub>nosis

Taken together, the controls show that some measured governance gains can come from adding visible artifacts without repository-specific diagnosis. A fixed template that never reads the project reaches mean NGI 0.272 and exceeds several models on the frozen ten-repository batch; a rule-based patch that does inspect repository state scores 0.018 lower than that template, a diference small relative to the observed no-op variation. Meanwhile, the model/template gap is not uniform across dimensions: repository-specific execution evidence provides an important distinction, but these results do not establish that it is the only capability separating stronger systems.

The result is a measurement warning rather than evidence that all artifact-facing scores are uninformative: the Eficient-WAM example shows that file presence and functional evidence can diverge, while the dimension-level results identify where this template did and did not earn credit. Future versions should strengthen executionbacked evidence requirements for the template-reachable dimensions and report template baselines alongside model scores.

## 6. Anal<sub>y</sub>sis

## 6<sub>.</sub>1<sub>.</sub> Wh<sub>a</sub>t th<sub>e</sub> b<sub>enc</sub>h<sub>mar</sub>k <sub>ma</sub>k<sub>es v</sub>i<sub>s</sub>ibl<sub>e</sub>

The benchmark occupies a middle ground between issue resolution and static repository inspection. An issue benchmark asks whether a particular defect is fixed; a repository-health audit asks whether a project exhibits a property at one point in time. SWE-Prometheus asks an agent to diagnose a project-wide gap, intervene under an open-ended objective, and provide evidence that the intervention is useful and behavior-preserving. The auxiliary controls make the two parts of this task visible: governance artifacts can be added mechanically, while repository-specific engineering outcomes require evidence from execution and inspection.

The central distinction the evidence loop enforces is between engineering presence and engineering efectiveness. A repository can contain a test suite, CI workflow, lint configuration, or dependency file without obtaining the intended benefit. The loop tests whether the associated command runs, whether its scope is meaningful, and whether the treated repository remains behaviorally compatible. Section 5.11 shows that the mechanical template raises D1–D3 but does not improve D5 or D6 and rarely changes D4. The template is therefore a calibration point: it identifies where artifact-oriented scoring is most accessible and where executable evidence demands more than adding conventional files.

The six-dimensional view also exposes asymmetric progress. Tests and environment files are often easy to add, whereas dependency security, maintainability, and collaboration practices require understanding project intent. Reporting per-dimension intervention rates rather than aggregate NGI alone is therefore not optional presentation detail; it is what prevents a template-shaped result from being read as a diagnosis-shaped one.

## 6.2. Im<sub>p</sub>rovement<sub>,</sub> safet<sub>y,</sub> and absolute <sub>q</sub>ualit<sub>y</sub> are se<sub>p</sub>arate axes

Higher NGI does not automatically imply lower risk. An agent that makes many changes may improve more dimensions and also create more opportunities for regressions. Conversely, a zero-breakage model may simply make few changes—MiniMax-M3 has no observed breaks and the lowest improvement in the release. The Kimi-K3/GLM-5.3-Flash comparison makes the distinction concrete: their valid-run means are indistinguishable at the current resolution, but Kimi-K3 has the stronger full-pool verified outcome because it breaks fewer repositories. The combination of NGI, behavior-breakage rate, coverage-weighted improvement, no-regression rate, and verified claim rate is therefore more informative than any single leaderboard number.

The gate analysis in Section 5.9 refines this. Removing the behavior gate raises the median for some models and lowers it for others rather than producing a universal inflation term, and the model ordering is unchanged. The gate’s value is not that it corrects scores but that it marks which scores cannot be trusted, and gate strength determines how much confidence a zero-breakage observation can carry at all.

## 6<sub>.</sub>3<sub>.</sub> I<sub>mp</sub>li<sub>ca</sub>ti<sub>ons</sub> f<sub>or agen</sub>t d<sub>es</sub>i<sub>gn</sub>

The task suggests three capabilities that are not captured by patch pass rates alone: repository reconnaissance, risk-aware intervention, and evidence-oriented verification. The matched comparison in Section 5.8 shows that only two of ten systems outperform the stronger template on a clear majority of the ten matched repositories. This makes repository-specific diagnosis a discriminating capability in the benchmark: aggregate gains alone do not show that an agent selected a better intervention than a repository-independent template.

Agents should therefore construct a compact repository map first, prioritize the gaps whose repair requires the project to actually build and run, and avoid changing stable source behavior when configuration or documentation is suficient. They should then verify both positive claims and negative side efects. This motivates future baselines with explicit diagnosis, planning, patching, and audit phases, and it suggests that diagnosis quality—not patch volume—is the capability the next iteration of this benchmark should isolate.

## 6<sub>.</sub>4<sub>.</sub> V<sub>a</sub>lidit<sub>y</sub> th<sub>rea</sub>t<sub>s</sub> <sub>an</sub>d <sub>m</sub>iti<sub>ga</sub>ti<sub>ons</sub>

Construct validity is threatened if the six dimensions become a checklist. We mitigate this with executable probes, mutation-based gate checks, and evidence requirements, and quantify residual exposure with template baselines rather than assuming it away. Internal validity is threatened by scafold diferences, one-shot sampling, and judge variability; Section 5.10 estimates the resolution of the current design, while paired base-to-treated measurements, fixed manifests, anonymized scoring, and calibration address other sources of variation. These safeguards do not remove the one-rollout limitation: each model-instance pair has only one agent rollout. External validity is limited by the current 60-repository sample and its language and ecosystem distribution. Releasing a larger, temporally refreshed, contamination-audited split is a direct next step.

The public table is a descriptive shared-subset result. Claude-Opus-5 was evaluated with Claude Code while the other models used pi, so the cross-scafold ordering should not be read as a causal model comparison. Even among the pi runs, each model-instance pair has one rollout. The common-valid comparison and failure-handling sensitivity analysis in Sections 5.4 and D therefore support conclusions about relative improvement and reliability more strongly than a precise ranking based only on conditional means.

## 7<sub>.</sub> Li<sub>m</sub>it<sub>a</sub>ti<sub>ons an</sub>d F<sub>u</sub>t<sub>ure</sub> W<sub>or</sub>k

Only 24 of 60 instances currently have behavior gates whose discriminative power is demonstrated through mutation testing. Blind, vacuous, and none instances remain useful for governance-state analysis but provide weaker behavioral evidence, and Section 5.9 quantifies how much weaker: measured breakage falls from 13% to 8% to 0% as gate strength degrades with almost no change in governance scores.

The current scorer uses multiple teacher models and retains their score spread. A three-pass maintainer review of the same representative preserved patches found the changes useful and identified no additional functiona problems across passes. Broader independent expert calibration is still desirable: subjective dimensions such as documentation and maintainability may contain shared judge errors. In the no-op condition, Structure & Maintainability scores drifted on three of ten unchanged repositories, which is consistent with judge variability but does not by itself isolate its cause. Future evaluation should add blinded expert annotations, adjudication, cross-family judges, and adversarial calibration.

Three of the six dimensions are also demonstrably reachable by a repository-blind template (Section 5.11). This is a property of the current rubric rather than only of the models: Tests & CI, Code Quality Gates, and Documentation & Collaboration should be tightened so that credit requires demonstrated scope and execution, bringing them closer to how Reproducible Environment and Dependency & Security Health are already substantiated.

Finally, the single-rollout design bounds what any ranking claim can mean. At the observed paired dispersion the public subset resolves diferences of about 0.19 NGI (Section 5.10), so adjacent positions in the current ordering are not separated by this instrument.

Diferent scafolds expose diferent tools, prompts, context handling, and termination behavior. Current budget artifacts also include cases where monetary cost is unavailable; eficiency conclusions should therefore use tokens, turns, wall-clock time, and tool calls until cost provenance is complete.

The public subset is defined by common model coverage rather than random sampling. Public GitHub repositories may also have appeared in model training data or later repository versions. Stronger releases require repository-family separation, near-duplicate scanning, contamination audits, and an access-controlled private evaluator.

Finally, a public distribution must not include private tasks, private verification patches, private results, or other hidden evaluation artifacts. Every model-instance pair should have an artifact manifest so that missing traces or evidence are explicit rather than represented by empty directories. D6 measures available dependency and security evidence, not comprehensive vulnerability absence.

Future work will add repeated rollouts, expert baselines, stronger characterization tests, same-scafold reruns, expert judge calibration, and a genuinely isolated private holdout. The template and deterministic baselines are now included as auxiliary measurements, but should be expanded across repository families before being treated as general lower bounds.

## 8. Conclusion

Repository-level agents are commonly evaluated as issue resolvers, while real teams also need them to improve the conditions under which a repository can be trusted. SWE-Prometheus makes this broader capability measurable by asking an agent to inspect a real repository, identify engineering risks, choose a retrofit strategy, and demonstrate that its changes work without breaking existing behavior.

The current SWE-Prometheus release shows that repository governance produces measurable diferences across agent systems. The benchmark contains 60 real repositories, while ten agent systems are evaluated on the shared 22-repository public subset; it exposes variation in governance improvement, behavior-regression risk, and dimension-specific engineering preferences. It also shows why executable evidence is necessary: a change that looks correct in a dif may fail when executed or may damage existing behavior.

The auxiliary study clarifies what these diferences mean. A repository-blind template reaches mean NGI 0.272 through changes concentrated in D1–D3, yet makes no gains in D5 or D6; the Eficient-WAM case shows why the distinction matters, since its added placeholder test passes while Ruf still reports 117 errors. At the same time, the no-op median is zero, the end-to-end standard deviation is 0.073, and two teachers agree exactly on 57 of 60 dimension scores for the same no-op evidence. A three-pass maintainer review of a fixed sample of behavior-preserved patches found useful maintenance changes and no new functional problems. The matched analysis shows only two of ten systems clearly beat the stronger template on a majority of repositories. For Kimi-K3 and GLM-5.3-Flash, common-valid improvement is similar, while Kimi-K3 ranks first across the three full-pool failure-handling policies examined (broken runs assigned 0, −1, or −3). Gate-strength stratification further shows why breakage rates must be interpreted alongside the ability of the gate to detect mutations.

Together these findings show that repository stewardship has several measurable parts: establishing useful governance practices, making them work in the target project, and preserving existing behavior. SWE-Prometheus reports these outcomes separately and includes baselines that expose how much each kind of evidence contributes. This gives researchers a way to distinguish conventional artifact completion from interventions supported by repository-specific execution and review.

SWE-Prometheus is therefore a resource for studying repository stewardship, complementing issue-level repair benchmarks with a setting that emphasizes diagnosis, prioritization, intervention, and verification. Its central question is not only whether an agent can change code, but whether it can leave a repository in a more trustworthy and maintainable state.

## Referen<sub>c</sub>e<sub>s</sub>

[1] Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=SVjPFnLBig.

[2] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. SWEagent: Agent-computer interfaces enable automated software engineering. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/079017-1601. URL https://proceedings.neurips.cc/paper\_files/paper /2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html.

[3] OpenAI. Introducing SWE-bench verified. OpenAI research publication, 2024. URL https://openai.com/index/intro ducing-swe-bench-verified/.

[4] Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL https://arxiv.org/abs/2107.03374.

[5] Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc V. Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. URL https://arxiv.org/abs/2108.07732.

[6] Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, and Charles Sutton. A survey of machine learning for big code and naturalness. ACM Computing Surveys, 51(4):81, 2018. doi: 10.1145/3212695.

[7] Linhao Zhang, Daoguang Zan, Quanshun Yang, Zhirong Huang, Dong Chen, Bo Shen, Tianyu Liu, Yongshun Gong, Pengjie Huang, Xudong Lu, Guangtai Liang, Lizhen Cui, and Qianxiang Wang. CodeV: Issue resolving with visual data. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7350–7361, 2025. doi: 10.18653/v1/2025.f indings-acl.384. URL https://aclanthology.org/2025.findings-acl.384/.

[8] Lianghong Guo, Wei Tao, Runhan Jiang, Yanlin Wang, Jiachi Chen, Xilin Liu, Yuchi Ma, Mingzhi Mao, Hongyu Zhang, and Zibin Zheng. OmniGIRL: A multilingual and multimodal benchmark for GitHub issue resolution. Proceedings of the ACM on Software Engineering, 2(ISSTA):24–46, 2025. doi: 10.1145/3728871. URL https://doi.org/10.1145/3728871. Article ISSTA002.

[9] Shaoxiong Zhan, Shi Hu, Boyu Feng, Hai Lin, Andrew Gong, Zhengda Zhou, Jiaying Zhou, Yunyun Hou, Hao Su, and Hai-Tao Zheng. MM-IssueLoc: A controlled benchmark for evaluating visual evidence in multimodal repository-level issue localization. arXiv preprint arXiv:2607.15205, 2026. URL https://arxiv.org/abs/2607.15205.

[10] Fan Cui, Hongyuan Hou, Zizhang Luo, Chenyun Yin, and Yun Liang. Hwe-bench: Benchmarking llm agents on real-world hardware bug repair tasks. arXiv preprint arXiv:2604.14709, 2026. URL https://arxiv.org/abs/2604.14709.

[11] Ion George Dinu, Marian Cristian Mihaescu, and Traian Rebedea. Smellbench: Evaluating llm agents on architectura code smell repair. arXiv preprint arXiv:2605.07001, 2026. URL https://arxiv.org/abs/2605.07001.

[12] Daoguang Zan, Zhirong Huang, Ailun Yu, Shaoxin Lin, Yifan Shi, Wei Liu, Dong Chen, Zongshuai Qi, Hao Yu, Lei Yu, Dezhi Ran, Muhan Zeng, Bo Shen, Pan Bian, Guangtai Liang, Bei Guan, Pengjie Huang, Tao Xie, Yongji Wang, and Qianxiang Wang. SWE-bench-java: A GitHub issue resolving benchmark for Java. arXiv preprint arXiv:2408.14354, 2024. URL https://arxiv.org/abs/2408.14354.

[13] Daoguang Zan, Zhirong Huang, Wei Liu, Hanwu Chen, Shulin Xin, Linhao Zhang, Qi Liu, Li Aoyan, Lu Chen, Xiaojian Zhong, Siyao Liu, Yongsheng Xiao, Liangqiang Chen, Yuyu Zhang, Jing Su, Tianyu Liu, Rui Long, Ming Ding, and Liang Xiang. Multi-SWE-bench: A multilingual benchmark for issue resolving. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-2111. URL https://proceedings.neurips.cc/paper\_files/paper /2025/hash/5afa9cb1e917b898ad418216dc726fbd-Abstract-Datasets\_and\_Benchmarks\_Track.html.

[14] Muhammad Shihab Rashid, Christian Bock, Yuan Zhuang, Alexander Buchholz, Tim Esler, Simon Valentin, Luca Franceschi, Martin Wistuba, Prabhu Teja Sivaprasad, Woo Jung Kim, Anoop Deoras, Giovanni Zappella, and Laurent Callot. SWEpolybench: A multi-language benchmark for repository level evaluation of coding agents. arXiv preprint arXiv:2504.08703, 2025. URL https://arxiv.org/abs/2504.08703.

[15] John Yang, Carlos E. Jimenez, Alex L. Zhang, Kilian Lieret, Joyce Yang, Xindi Wu, Ori Press, Niklas Muennighof, Gabriel Synnaeve, Karthik R. Narasimhan, Diyi Yang, Sida I. Wang, and Ofir Press. SWE-bench multimodal: Do AI systems generalize to visual software domains? In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/07d6332ae36730707fddddba736d7b6c-Abstrac t-Conference.html.

[16] Linghao Zhang, Shilin He, Chaoyun Zhang, et al. SWE-bench goes live! arXiv preprint arXiv:2505.23419, 2025. URL https://arxiv.org/abs/2505.23419.

[17] Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, and Boris Yangel. SWE-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. In Advances in Neural Information Processing Systems, 2025. doi: 10.52202/085713-0788. URL https://papers.neurips.cc/paper\_files/paper/2025/hash/21bec6ace947b1b5 8967b945c8ac0f10-Abstract-Datasets\_and\_Benchmarks\_Track.html.

[18] Zhipeng Xu, Jiahao Lu, Yining Zheng, Yuxin Wang, and Xipeng Qiu. SWE-bench science: Can coding agents resolve engineering tasks in science? arXiv preprint arXiv:2608.19799, 2026. URL https://arxiv.org/abs/2608.19799.

[19] Jiahong Xiang, Wenxiao He, Xihua Wang, Hongliang Tian, and Yuqun Zhang. Evaluating and improving automated repository-level rust issue resolution with llm-based agents. arXiv preprint arXiv:2602.22764, 2026. URL https: //arxiv.org/abs/2602.22764.

[20] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighof, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. OpenHands: An open platform for AI software developers as generalist agents. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=OJd3ayDDoF.

[21] Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE):801–824, 2025. doi: 10.1145/3715754. URL https://doi.org/10.1145/3715754.

[22] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE\_vluYUL-X.

[23] Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, 2023. doi: 10.52202/075280-2997. URL https://proceedings.neurips. cc/paper\_files/paper/2023/hash/d842425e4bf79ba039352da0f658a906-Abstract-Conference.html.

[24] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652, 2023. doi: 10.52202/075280-0377. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/1b44b87 8bb782e6954cd888628510e90-Abstract-Conference.html.

[25] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_fi les/paper/2024/hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html.

[26] Wendong Bu, Yang Wu, Qifan Yu, Minghe Gao, Bingchen Miao, Zhenkui Zhang, Kaihang Pan, Yunfei Li, Mengze Li, Wei Ji, Juncheng Li, Siliang Tang, and Yueting Zhuang. What limits virtual agent application? omnibench: A scalable multi-dimensional benchmark for essential virtual agent capabilities. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 5725–5748, 2025. URL https://proceedings.mlr.press/v267/bu25b.html.

[27] Xinyun Chen, Maxwell Lin, Nathanael Scharli, and Denny Zhou. Teaching large language models to self-debug. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/ 2024/hash/2460396f2d0d421885997dd1612ac56b-Abstract-Conference.html.

[28] Tianyang Liu, Canwen Xu, and Julian McAuley. RepoBench: Benchmarking repository-level code auto-completion systems. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/pape r/2024/hash/d191ba4c8923ed8fd8935b7c98658b5f-Abstract-Conference.html.

[29] Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2471–2484, 2023. doi: 10.18653/v1/2023.emnlp -main.151. URL https://aclanthology.org/2023.emnlp-main.151/.

[30] Shaoqiu Zhang, Yuhang Wang, Jialiang Liang, Yuling Shi, Wenhao Zeng, Maoquan Wang, Shilin He, et al. SWEexplore: Benchmarking how coding agents explore repositories. arXiv preprint arXiv:2606.07297, 2026. URL https: //arxiv.org/abs/2606.07297.

[31] Islem Bouzenia, Premkumar Devanbu, and Michael Pradel. Repairagent: An autonomous, llm-based agent for program repair. In International Conference on Software Engineering, pages 2188–2200, 2025. doi: 10.1109/ICSE55347.2025.00157. URL https://ieeexplore.ieee.org/abstract/document/11029914/.

[32] Qinyu Luo, Yining Ye, Shihao Liang, Zhong Zhang, Yujia Qin, Yaxi Lu, Yesai Wu, Xin Cong, Yankai Lin, Yingli Zhang, Xiaoyin Che, Zhiyuan Liu, and Maosong Sun. RepoAgent: An LLM-powered open-source framework for repositorylevel code documentation generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 436–464. Association for Computational Linguistics, 2024. doi: 10.18653/v1/ 2024.emnlp-demo.46. URL https://aclanthology.org/2024.emnlp-demo.46/.

[33] Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with SWE-Gym. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 47717–47737, 2025. URL https://proceedings.mlr.pr ess/v267/pan25g.html.

[34] Dongfu Jiang, Xuan He, Huaye Zeng, Cong Wei, Max W. F. Ku, Qian Liu, and Wenhu Chen. MANTIS: Interleaved multi-image instruction tuning. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum ?id=skLtdUVaJa.

[35] Bingchen Zhao, Yongshuo Zong, Letian Zhang, and Timothy Hospedales. Benchmarking multi-image understanding in vision and language models: Perception, knowledge, reasoning, and multi-hop reasoning. arXiv preprint arXiv:2406.12742, 2024. URL https://arxiv.org/abs/2406.12742.

[36] Fei Wang, Xingyu Fu, James Y. Huang, Zekun Li, Qin Liu, Xiaogeng Liu, Mingyu Derek Ma, Nan Xu, Wenxuan Zhou, Kai Zhang, Tianyi Yan, Wenjie Mo, Hsiang-Hui Liu, Pan Lu, Chunyuan Li, Chaowei Xiao, Kai-Wei Chang, Dan Roth, Sheng Zhang, Hoifung Poon, and Muhao Chen. MuirBench: A comprehensive benchmark for robust multi-image understanding. In The Thirteenth International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/pape r\_files/paper/2025/hash/9cf6139382f98623d08cc595622f3fb1-Abstract-Conference.html.

[37] Qiguang Chen, Chengyu Luan, Jiajun Wu, Qiming Yu, Yi Yang, Yizhuo Li, Jingqi Tong, Xiachong Feng, Libo Qin, and Wanxiang Che. OMIBench: Benchmarking olympiad-level multi-image reasoning in large vision-language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 45100–45135. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.2090. URL https://aclanthology.org/2026.acl-long.2090/.

[38] Nam Le Hai, Dung Manh Nguyen, and Nghi D. Q. Bui. On the impacts of contexts on repository-level code generation. In Findings of the Associationfor Computational Linguistics: NAACL 2025, pages 1496–1524, 2025. doi: 10.18653/v1/2025.f indings-naacl.82.

[39] Niels Mündler, Mark Niklas Müller, Jingxuan He, and Martin Vechev. SWT-Bench: Testing and validating real-world bug-fixes with code agents. In Advances in Neural Information Processing Systems, volume 37, pages 81857–81887, 2024. doi: 10.52202/079017-2601. URL https://papers.neurips.cc/paper\_files/paper/2024/hash/94f093b41fc26663 76fb1f667fe282f3-Abstract-Conference.html.

[40] Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. In Advances in Neural Information Processing Systems, volume 36, pages 21558–21572, 2023. doi: 10.52202/075280-0943. URL https://proceedings.neurips.cc/paper\_f iles/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract.html.

[41] Yue Jia and Mark Harman. An analysis and survey of the development of mutation testing. IEEE Transactions on Software Engineering, 37(5):649–678, 2011. doi: 10.1109/TSE.2010.62.

[42] Qingzhou Luo, Farah Hariri, Lamyaa Eloussi, and Darko Marinov. An empirical analysis of flaky tests. In Proceedings of the 22nd ACM SIGSOFT International Symposium on Foundations of Software Engineering, pages 643–653, 2014. doi: 10.1145/2635868.2635920. URL https://doi.org/10.1145/2635868.2635920.

[43] Marco Tulio Ribeiro, Tong Wu, Carlos Guestrin, and Sameer Singh. Beyond accuracy: Behavioral testing of nlp models with checklist. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 4902–4912, 2020. doi: 10.18653/v1/2020.acl-main.442.

[44] Raymond Reiter. A theory of diagnosis from first principles. Artificial Intelligence, 32(1):57–95, 1987. doi: 10.1016/0004 -3702(87)90062-2.

[45] Sergio Castro, Coen De Roover, Andy Kellens, Angela Lozano, Kim Mens, and Theo D’Hondt. Diagnosing and correcting design inconsistencies in source code with logical abduction. Science of Computer Programming, 76(12):1113–1129, 2011. doi: 10.1016/j.scico.2010.09.001. Special issue on Software Evolution and Adaptability.

[46] International Organization for Standardization. Systems and Software Engineering—Systems and Software Quality Requirements and Evaluation (SQuaRE)—Product Quality Model. ISO, 2023. URL https://www.iso.org/standard/781 . ISO/IEC 25010:2023.

[47] Thomas J. McCabe. A complexity measure. IEEE Transactions on Software Engineering, 2(4):308–320, 1976. doi: 10.1109/TSE.1976.233837.

[48] Shyam R. Chidamber and Chris F. Kemerer. A metrics suite for object oriented design. IEEE Transactions on Software Engineering, 20(6):476–493, 1994. doi: 10.1109/32.295895.

[49] Zengyang Li, Paris Avgeriou, and Peng Liang. A systematic mapping study on technical debt and its management. Journal of Systems and Software, 101:193–220, 2015. doi: 10.1016/j.jss.2014.12.027. URL https://doi.org/10.1016/j.jss. 2014.12.027.

[50] Jean-Louis Letouzey. The SQALE method for evaluating technical debt. In 2012 Third International Workshop on Managing Technical Debt, pages 31–36, 2012. doi: 10.1109/MTD.2012.6225997.

[51] Bogdan Vasilescu, Yue Yu, Huaimin Wang, Premkumar Devanbu, and Vladimir Filkov. Quality and productivity outcomes relating to continuous integration in github. In Proceedings of the 2015 10th Joint Meeting on Foundations of Software Engineering, pages 805–816, 2015. doi: 10.1145/2786805.2786850. URL https://doi.org/10.1145/2786805.2786850.

[52] Moritz Beller, Georgios Gousios, and Andy Zaidman. Oops, my tests broke the build: An explorative analysis of Travis CI with GitHub. In 2017 IEEE/ACM 14th International Conference on Mining Software Repositories, pages 356–367, 2017. doi: 10.1109/MSR.2017.62. URL https://doi.org/10.1109/MSR.2017.62.

[53] OpenSSF. Openssf scorecard, 2026. URL https://github.com/ossf/scorecard. Accessed 2026-09-18.

[54] SLSA. Supply-chain levels for software artifacts (SLSA) version 1.0, 2023. URL https://slsa.dev/spec/v1.0/.

[55] Murugiah Souppaya, Karen Scarfone, and Donna Dodson. Secure software development framework (SSDF) version 1.1: Recommendations for mitigating the risk of software vulnerabilities, 2022. URL https://csrc.nist.gov/pubs/sp/800/ 218/final.

[56] Jasmine Latendresse, Suhaib Mujahid, Diego Elias Costa, and Emad Shihab. Not all dependencies are equal: An empirical study on production dependencies in NPM. In IEEE/ACM International Conference on Automated Software Engineering, pages 73:1–73:12, 2022. doi: 10.1145/3551349.3556896. URL https://doi.org/10.1145/3551349.3556896.

[57] Konstantinos Vergopoulos, Mark Niklas Müller, and Martin Vechev. Automated benchmark generation for repository-leve coding tasks. arXiv preprint arXiv:2503.07701, 2025. URL https://arxiv.org/abs/2503.07701.

[58] Sayash Kapoor and Arvind Narayanan. Leakage and the reproducibility crisis in machine-learning-based science. Patterns, 4(9):100804, 2023. doi: 10.1016/j.patter.2023.100804.

[59] Colin White, Samuel Dooley, Manley Roberts, Arka Pal, Benjamin Feuer, Siddhartha Jain, Ravid Shwartz-Ziv, Neel Jain, Khalid Saifullah, Sreemanti Dey, Shubh Agrawal, Sandeep Singh Sandha, Siddhartha Naidu, Chinmay Hegde, Yann LeCun, Tom Goldstein, Willie Neiswanger, and Micah Goldblum. LiveBench: A challenging, contamination-limited LLM benchmark. In International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=sKYHBTAxVa.

[60] Fernando Martínez-Plumed, Pablo Barredo, Seán Ó hÉigeartaigh, and José Hernández-Orallo. Research community dynamics behind popular AI benchmarks. Nature Machine Intelligence, 3(7):581–589, 2021. doi: 10.1038/s42256-021-0 0339-6. URL https://www.nature.com/articles/s42256-021-00339-6.

[61] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623, 2023. doi: 10.52202/075280-2020. URL https://proceedings.neurips.cc/paper\_files/paper/2023/hash/91f18a1287b398d 378ef22505bf41832-Abstract-Datasets\_and\_Benchmarks.html.

[62] Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. G-Eval: NLG evaluation using GPT-4 with better human alignment. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 2511–2522. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main.153. URL https://aclanthology.org/2023.emnlp-main.153/.

## A. Dataset Construction Details

The appendix is divided into a small number of top-level modules following the organization of benchmark papers such as OMIBench. Appendix A covers dataset construction and instance records; Appendix B gives the detailed main experiment and agent lifecycle; Appendix C defines evaluation and scoring; Appendix D reports detailed analysis and case studies; and Appendix E documents reproducibility, release, and the data dictionary. Auxiliary experiments and statistical analyses are kept as subsections of the relevant modules rather than promoted to independent appendices.

## A<sub>.</sub>1<sub>.</sub> Artifacts and accountin<sub>g</sub>

Each instance is identified by repository, base commit, split, license, baseline scores, verification status, and gate-strength label. A complete model-instance record contains the unified patch, agent output, base evidence, treated evidence, verification result, budget record, and score record. Missing artifacts should be represented in a manifest with an explicit reason.

The evaluator distinguishes infrastructure failure, environment or dependency failure, missing or invalid patch, behavior regression, preserved behavior with no governance improvement, and preserved behavior with positive improvement. Behavior-broken runs are excluded from valid NGI aggregation but remain in failure statistics. Monetary cost is reported only when provider accounting is available; otherwise it is marked unavailable.

## A<sub>.</sub>2<sub>.</sub> R<sub>u</sub>bri<sub>c a</sub>n<sub>c</sub>h<sub>o</sub>r<sub>s</sub>

Scores of 1–2 indicate absent or non-functional governance, 3 indicates partial coverage with material gaps, 4 indicates an executable and reviewable standard, and 5 indicates broad coverage with evidence of maintainability and low regression risk.

Ta<sup>bl</sup>e 3: Compact rubric anchors used for each governance dimension.
<table><tr><td>Score</td><td>Interpretation</td></tr><tr><td>1</td><td>Absent, misleading, or actively broken practice</td></tr><tr><td>2</td><td>Minimal artifact with major gaps or unreliable execution</td></tr><tr><td>3</td><td>Partial coverage; useful in common cases but material gaps remain</td></tr><tr><td>4</td><td>Executable, reviewable, and appropriate for the repository</td></tr><tr><td>5</td><td>Broad, durable coverage with evidence of maintainability</td></tr></table>

## A<sub>.</sub>3<sub>.</sub> Instance Records and Behavior Gates

## A.3.1. Instance schema

Table 4 lists the minimum fields required for a reproducible instance. Fields are separated into task, evidence, and result records so that a missing trace cannot be mistaken for a zero score.

Ta<sup>bl</sup>e 4: Minimum instance schema for the release records.
<table><tr><td>Record</td><td>Required fields</td></tr><tr><td>Task</td><td>repository, base commit, split, language, license, applicability mask, task text</td></tr><tr><td>Environment</td><td>image or lockfile identifier, installation command, tool versions, timeout policy</td></tr><tr><td>Evidence</td><td>probe name, command, exit code, stdout/stderr digest, timestamp, applicability</td></tr><tr><td>Behavior</td><td>characterization tests, mutation sample, gate-strength label, pre/post status</td></tr><tr><td>Result</td><td>patch digest, changed files, score vector, NGI, breakage label, budget record</td></tr><tr><td>Provenance</td><td>evaluator version, prompt version, scaffold, model identifier, artifact completeness</td></tr></table>

## A.3.2. Detailed <sub>g</sub>ate-stren<sub>g</sub>th accountin<sub>g</sub>

The current 60-instance release contains 24 instances with mutation-demonstrated gates, 27 with blind gates, 3 with vacuous gates, and 6 without a usable behavior gate. This distribution motivates reporting gate strength as a first-class variable. The reported results distinguish the full set, the mutation-demonstrated subset, and the no-gate subset separately.

Ta<sup>bl</sup>e 5: Behavior-gate categories and their interpretation.
<table><tr><td>Category</td><td>Count</td><td>Interpretation</td></tr><tr><td>Detected</td><td>24</td><td>Mutation is caught by the characterization gate</td></tr><tr><td>Blind</td><td>27</td><td>Mutation is not caught; risk evidence is weak</td></tr><tr><td>Vacuous</td><td>3</td><td>Gate executes but has insufficient discriminative power</td></tr><tr><td>None</td><td>6</td><td>No usable behavior gate is available</td></tr></table>

## A<sub>.</sub>3<sub>.</sub>3<sub>.</sub> R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y c</sub>h<sub>ec</sub>kli<sub>s</sub>t

The artifact release includes: (1) a manifest of repository commits and licenses; (2) a container or environment lock for every public instance; (3) deterministic scorer configuration; (4) command-level logs and exit codes; (5) patch and trace completeness checks; (6) a separation policy for private tasks and verification evidence; (7) evaluator and judge version identifiers; (8) a script that reconstructs every main-paper table; and (9) a changelog for repository, rubric, or scoring updates.

## A.3.4. Auxiliar<sub>y</sub> ex<sub>p</sub>eriment worksheet

For every completed auxiliary experiment, this appendix records the sampling frame, model and scafold, paired comparison unit, and machine-readable result source. The ten-repository batch is frozen. Repeated-seed, samescafold, and expert conditions were not run and are not included in the reported measurements.

## A.3.5. Ex<sub>p</sub>anded rubric <sub>p</sub>rom<sub>p</sub>ts

For D1, evaluators should ask whether tests exercise important behavior and whether CI runs the relevant tests on a clean checkout. For D2, they should ask whether the quality command is meaningful, scoped, and enforced rather than merely configured. For D3, they should ask whether a new contributor can install, use, modify, and report security issues. For D4, they should ask whether changes reduce unnecessary coupling and preserve understandable boundaries. For D5, they should ask whether another user can reproduce installation and execution from the declared environment. For D6, they should ask whether dependencies are controlled and whether security checks produce inspectable evidence. In every dimension, a file’s presence without successful execution is insuficient for a score of 4 or 5.

## A<sub>.</sub>3<sub>.</sub>6<sub>.</sub> A<sub>ux</sub>ili<sub>ary-resu</sub>lt<sub>s man</sub>if<sub>es</sub>t

The following manifest records which auxiliary values are already incorporated and which analyses remain open.

The measured values are summarized in Table 16; rows outside the primary comparison are clearly identified and are not cited as empirical evidence. The archived result package contains the exact ten-instance selection and run metadata.

## A.4. Candidate Dataset and Quality Control

This appendix expands the construction description by separating the benchmark object, the construction funnel, the evaluation protocol, and the quality-control evidence. It is explicit about what is fixed, what is hidden, and how quality control is applied.

## A.4.1. Dataset construction overview

The current SWE-Prometheus release was constructed as a repository-governance benchmark rather than sampled from a synthetic checklist. The construction unit is a repository at a fixed commit. The final package contains

Ta<sup>bl</sup>e 6: Status of auxiliary-result fields.
<table><tr><td>Artifact</td><td>Available content</td><td>Reported measure</td></tr><tr><td>10-repository manifest Frozen stratified batch</td><td></td><td>Recorded in machine-readable manifest</td></tr><tr><td>Main rerun</td><td>Existing model rollouts on the batch</td><td>Observed per-instance scores</td></tr><tr><td>No-op baseline</td><td>Measured end-to-end stability</td><td>NGI mean, median, SD, and teacher agreement</td></tr><tr><td>Mechanical baseline</td><td>Measured template intervention</td><td>Patch and evidence results</td></tr><tr><td>Rule baseline</td><td>Measured deterministic intervention</td><td>Rule-based results</td></tr><tr><td>Gate ablation</td><td>Measured on all 60 instances</td><td>Gated/ungated paired statistics</td></tr><tr><td>Repeated seeds</td><td>Not included in the primary comparison</td><td>Reserved for follow-up evaluation</td></tr><tr><td>Expert treatment</td><td>Not included in the primary comparison</td><td>Reserved for follow-up evaluation</td></tr></table>

60 instances, divided into 22 public instances with shared model coverage and 38 private instances reserved for internal evaluation. The public/private distinction is applied after task validation and does not change the scoring definition.

The construction workflow is:

1. discover repositories with plausible engineering-health gaps;

2. freeze a reconstructable base commit and collect project metadata;

3. run baseline probes and assign the six-dimension applicability mask;

4. construct or validate a behavior characterization gate;

5. execute a dry-run of installation, probes, and scoring;

6. record artifacts, provenance, and exclusion reasons;

7. assign the instance to the public or private split and freeze the manifest.

The workflow intentionally separates discovery from inclusion. A repository can be useful for finding candidates because it lacks a workflow or has sparse tests, but it enters the final dataset only after its behavior, environment, and governance headroom are independently checked.

Ta<sup>bl</sup>e 7: Dataset-construction stages and concrete outputs.
<table><tr><td>Stage</td><td>Operation</td><td>Output</td></tr><tr><td>Discovery</td><td>Search active open-source repositories for governance signals</td><td>Candidate repository list</td></tr><tr><td>Pinning</td><td>Select a reachable pre-treatment commit</td><td>Repository and commit manifest</td></tr><tr><td>Baseline</td><td>Run installation, tests, quality, dependency, and secu- Base evidence and score vector rity probes</td><td></td></tr><tr><td>Applicability</td><td>Decide which of D1-D6 are meaningful</td><td>Applicability mask and rationale</td></tr><tr><td>Behavior</td><td>Validate characterization tests and mutation sensitivity</td><td>Verification patch and gate-strength label</td></tr><tr><td>Dry run</td><td>Rebuild the environment and execute the full evaluator</td><td>Environment and evaluator log</td></tr><tr><td>Packaging</td><td>Hash artifacts and remove protected material from public outputs</td><td>Instance package and release manifest</td></tr></table>

## A.4.2. Construction <sub>p</sub>rinci<sub>p</sub>les

SWE-Prometheus treats a repository as a maintained artifact rather than a collection of isolated code snippets.   
The benchmark therefore follows four principles. First, each task starts from a reconstructable repository snapshot.

Second, the objective is open-ended enough to admit multiple valid engineering interventions. Third, each claimed improvement is connected to a probe or a structured judgment. Fourth, behavior preservation is tested independently from governance improvement.

These principles distinguish three artifacts that can otherwise be conflated: the task defines the engineering objective, the agent patch represents the intervention, and the evaluator evidence determines whether the intervention is useful and safe. The developer or construction patch, when present, is a construction asset and is not exposed as an oracle to the agent.

## A<sub>.</sub>4<sub>.</sub>3<sub>.</sub> C<sub>an</sub>did<sub>a</sub>t<sub>e</sub> di<sub>scovery</sub> f<sub>unne</sub>l

Candidate discovery proceeds in four stages: broad repository retrieval, automated mechanical screening, manual validity inspection, and evaluator dry runs. Broad retrieval favors recall and may use repository metadata, visible workflow gaps, test sparsity, packaging inconsistencies, documentation gaps, and dependency hygiene signals. Mechanical screening removes repositories that cannot be cloned or pinned. Manual inspection verifies that the project has a meaningful functional core and that its governance gaps are not artifacts of a broken checkout.

The final dry run reconstructs the base snapshot, executes all applicable probes, and records unavailable evidence. A candidate is rejected when the environment cannot be made reproducible, when the task requires private services, when no dimension has plausible headroom, or when behavior cannot be characterized at all. This funnel prevents the benchmark from equating “many missing files” with “high-value engineering work.”

## A<sub>.</sub>4<sub>.</sub>4<sub>.</sub> B<sub>ase</sub>li<sub>ne assessmen</sub>t <sub>an</sub>d h<sub>ea</sub>d<sub>room</sub>

For every retained candidate, the baseline evaluator runs before any model treatment. The baseline record contains the raw probe outputs, the normalized score vector, the set of applicable dimensions, and the reason for every score. Headroom is computed relative to the dimension ceiling, but high headroom alone is not suficient for inclusion: a repository with a completely non-functional environment is not a meaningful governance task.

The baseline pass also identifies which commands are safe to run without network access and which dependencies are merely preinstalled in the evaluator image. This distinction is particularly important for D5 and D6. Declared project dependencies are audited as project artifacts; evaluator convenience tools are not counted as evidence that the repository itself is reproducible or secure.

## A.4.5. Behavior-<sub>g</sub>ate construction

Behavior gates are constructed from repository-native tests whenever possible. If the native suite is too broad, a small characterization patch targets stable public behavior and records the command used to execute it. The patch is validated on the base snapshot before any model treatment. We then assess gate strength by injecting behavior-preserving or behavior-changing mutations where the repository and language permit it. A gate is labeled detected only when the mutation is caught; a passing but mutation-blind test remains blind rather than being promoted to strong evidence.

This construction step is kept separate from governance scoring. The gate protects against destructive interventions, while the six-dimensional scorer measures engineering readiness. A strong gate therefore increases confidence in a treatment outcome but does not directly increase the treatment score.

## A<sub>.</sub>4<sub>.</sub>6<sub>.</sub> S<sub>p</sub>lit <sub>cons</sub>t<sub>ruc</sub>ti<sub>on an</sub>d l<sub>ea</sub>k<sub>age con</sub>t<sub>ro</sub>l

The public split contains the instances used for shared comparison and trace inspection. The private split retains tasks, verification patches, and outcomes that should not be available for targeted optimization. Split assignment is recorded in the manifest, and private artifacts are excluded from public result aggregation. A broader public split should additionally use repository-family separation, near-duplicate scanning, and temporal contamination checks.

## A.4.7. Instance <sub>p</sub>acka<sub>g</sub>in<sub>g</sub>

An instance package contains task metadata, a repository identity, a base commit, a dimension applicability mask, base evidence, a verification asset, and evaluator configuration. Public packages may include task descriptions and public traces, but private verification patches, hidden test details, and private model outcomes remain excluded. Every package receives a stable identifier and a manifest digest.

Ta<sup>bl</sup>e 8: Candidate screening checklist used before an instance enters the benchmark.
<table><tr><td>Gate</td><td>Pass condition</td><td>Recorded evidence</td></tr><tr><td>Identity</td><td>Repository and base commit are uniquely identified</td><td>URL, commit hash, license</td></tr><tr><td>Buildability</td><td>Snapshot can be installed or inspected with declared limits</td><td>Environment log, failure reason</td></tr><tr><td>Functionality</td><td>Project contains meaningful runnable or testable be- havior</td><td>Entry points and test sample</td></tr><tr><td>Applicability</td><td>At least two governance dimensions are relevant</td><td>Dimension mask</td></tr><tr><td>Headroom</td><td>Baseline is below the dimension ceiling without being pathological</td><td>Baseline score vector</td></tr><tr><td>Behavior</td><td>A characterization test or gate can be constructed</td><td>Patch, command, gate strength</td></tr><tr><td>Isolation</td><td>No private credential or construction artifact is re- quired</td><td>Network and secret policy</td></tr></table>

Packaging is performed before model assignment. The task text does not disclose a target score, a preferred file, or a reference implementation. If a repository contains generated files or large assets, the package records whether they are retained, omitted, or reconstructed. Any omission that can afect execution is treated as a validity concern rather than an undocumented convenience.

## A.4.8. Quality-control passes

We define three independent quality-control passes. The first checks package integrity: identifiers, commit reachability, file presence, manifest hashes, and license metadata. The second checks evaluator integrity: base evidence is reproducible, the characterization test runs on the base state, and the treated-state harness accepts a known valid patch. The third checks interpretation: the score record includes evidence, the behavior result is consistent with the logs, and missing data are labeled explicitly.

The passes are deliberately separated because a package can be technically executable but conceptually invalid, or conceptually sound but impossible to reproduce. The release checklist records pass, fail, or not-applicable for every pass and preserves the reason for every exception.

## B<sub>.</sub> D<sub>e</sub>t<sub>a</sub>il<sub>e</sub>d M<sub>a</sub>i<sub>n</sub> E<sub>xper</sub>i<sub>men</sub>t

## B.1. Task tu<sub>p</sub>le

We represent an instance as

$$
T _ { i } = ( R _ { i } , c _ { i } , M _ { i } , A _ { i } , E _ { i } , V _ { i } ) ,
$$

where $R _ { i }$ is the repository, $c _ { i }$ the pinned base commit, $M _ { i }$ the task metadata and applicability mask, $A _ { i }$ the agent-facing instruction, $E _ { i }$ the executable environment, and $V _ { i }$ the verification package. The agent receives $\left( { { R } _ { i } } , { { c } _ { i } } , { { M } _ { i } } , { { A } _ { i } } , E _ { i } \right)$ but not protected parts of $V _ { i }$ . A rollout produces a patch $P _ { i , m }$ and an execution trace $\tau _ { i , m }$ for model m.

The evaluator computes base evidence $B _ { i }$ and treated evidence $B _ { i } ( P _ { i , m } )$ . For each dimension $d ,$ the scorer returns a score in {1, 2, 3, 4, 5} or an explicit inapplicable value. The result record is therefore a tuple of evidence, scores, behavior status, and budget rather than only a scalar.

## B<sub>.</sub>2<sub>.</sub> E<sub>v</sub>id<sub>ence recor</sub>d

Every probe should be represented by the following fields: probe identifier, command, working directory, environment identifier, start and end time, exit code, timeout status, output digest, normalized result, applicability, and human-readable interpretation. The raw log may be stored separately, but the normalized record must be suficient to reconstruct why a score was assigned.

Ta<sup>bl</sup>e 9: Evidence fields and their role in auditability.
<table><tr><td>Field</td><td>Purpose</td></tr><tr><td>Probe ID</td><td>Stable name for the check across evaluator versions</td></tr><tr><td>Command</td><td>Exact executable command, including relevant flags</td></tr><tr><td>Environment</td><td>Runtime, image, dependency, and tool version identifiers</td></tr><tr><td>Exit status</td><td>Distinguishes pass, failure, timeout, and unavailable execution</td></tr><tr><td>Output digest</td><td>Links the normalized record to the retained log</td></tr><tr><td>Applicability</td><td>Indicates whether the check is meaningful for this repository</td></tr><tr><td>Interpretation</td><td>Explains the mapping from evidence to rubric score</td></tr></table>

## B<sub>.</sub>3<sub>.</sub> A<sub>pp</sub>li<sub>ca</sub>bilit<sub>y</sub> <sub>an</sub>d <sub>m</sub>i<sub>ss</sub>i<sub>ngness</sub>

An inapplicable dimension is not equivalent to a failed dimension. For example, a repository without a releasable package may not receive a packaging score, while a repository that declares a package but cannot install it receives negative evidence. The scorer therefore records applicability separately from outcome and prevents missing evidence from being converted into an unexplained zero.

Missingness is categorized as not applicable, unavailable due to environment, unavailable due to timeout, or malformed artifact. This coding supports sensitivity analyses. Complete-case summaries retain unavailable evidence as a separate outcome rather than converting it to a zero.

## B<sub>.</sub>4<sub>.</sub> N<sub>orma</sub>li<sub>za</sub>ti<sub>on</sub> <sub>an</sub>d <sub>aggrega</sub>ti<sub>on</sub>

Raw dimension scores are retained alongside normalized improvement. For an instance with applicable dimensions D , the headroom-normalized score is computed only over defined dimensions. We report the number of dimensions contributing to each aggregate, the raw score sum, the treated score sum, and the normalized value. This avoids over-interpreting an identical NGI computed from very diferent numbers of applicable dimensions.

## B<sub>.</sub>5<sub>.</sub> A<sub>g</sub>ent Protocol and Run Lifec<sub>y</sub>cle

## B.5.1. A<sub>g</sub>ent-facin<sub>g</sub> instruction tem<sub>p</sub>late

The following template describes the stable task-level contract. The actual task-specific metadata and repository path are filled by the runner. It does not prescribe which files to edit or which governance intervention to choose.

Inspect the provided repository and improve its engineering readiness across the applicable governance dimensions. Preserve existing behavior and public interfaces. Make useful changes for future users and maintainers. Use repository evidence to prioritize changes, verify important claims with executable checks, and submit the resulting patch. Do not use hidden evaluator assets or any reference patch.

The instruction is intentionally open-ended. The agent must discover the repository’s build system, tests, documentation conventions, and dependency structure rather than receiving a checklist of missing files. The runner separately provides tool descriptions, time limits, and submission semantics.

## B.5.2. Run lifec<sub>y</sub>cle

Each run follows the lifecycle below:

1. reconstruct the pinned base snapshot;

2. verify base environment and characterization behavior;

3. launch the model with the fixed task instruction;

4. archive the model output and candidate patch;

5. apply the patch to an independent clean snapshot;

6. run treated-state probes and behavior verification;

7. score the evidence and write the result manifest.

The agent-side container is discarded after submission. The evaluator does not reuse the agent’s mutable working tree because doing so could allow unrecorded caches or generated files to influence the result.

## B.5.3. Canonical-run <sub>p</sub>olic<sub>y</sub>

Retries are allowed for infrastructure failures such as a failed container launch, a transient model service error, or a verifier crash. A patch that is valid but incorrect is not retried merely to obtain a better result. When a retry occurs, the original record remains archived and the manifest identifies the canonical run with an explicit reason.

## B.5.4. Bud<sub>g</sub>et accountin<sub>g</sub>

Budget records contain model identifier, scafold, maximum wall-clock time, actual elapsed time, input and output token counts when available, number of turns, tool calls, termination reason, and provider cost when available. A missing cost field is represented as unavailable. We do not compare providers using estimated prices from incomplete ledgers.

Ta<sup>bl</sup>e 10: Run-level budget fields for later auxiliary experiments.
<table><tr><td>Field</td><td>Definition</td></tr><tr><td>Model/scaffold</td><td>Exact model and execution framework</td></tr><tr><td>Wall time</td><td>Start-to-stop elapsed time and configured timeout</td></tr><tr><td>Turns/tools</td><td>Number of agent turns and external tool calls</td></tr><tr><td>Tokens</td><td>Input, output, and total tokens when exposed</td></tr><tr><td>Termination</td><td>Completed, timeout, service failure, invalid patch, or other reason</td></tr><tr><td>Cost</td><td>Provider-reported cost; unavailable otherwise</td></tr></table>

## C<sub>.</sub> E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> <sub>an</sub>d S<sub>cor</sub>i<sub>ng</sub>

## C<sub>.</sub>1<sub>.</sub> Di<sub>mens</sub>i<sub>on-spec</sub>ifi<sub>c score anc</sub>h<sub>ors</sub>

The general 1–5 anchors are expanded below to reduce holistic scoring. A score of 4 requires executable or inspectable evidence appropriate to the repository; a configuration file by itself is not suficient.

## C<sub>.</sub>2<sub>.</sub> E<sub>v</sub>id<sub>ence su</sub>fi<sub>c</sub>i<sub>ency ru</sub>l<sub>es</sub>

For D1, a passing test command must be paired with evidence that the tests exercise meaningful behavior. For D2, the existence of a formatter or linter configuration does not establish that it runs on the project’s relevant files. For D3, prose is evaluated for actionability rather than length. For D4, a large refactor is not rewarded if it increases churn without improving boundaries. For D5, a lockfile is useful only when it agrees with the declared project metadata. For D6, a dependency report must distinguish project dependencies from tools preinstalled in the evaluator image.

## C.3. No-re<sub>g</sub>ression rules

A treated repository receives a regression flag when characterization behavior fails, an important public interface changes unexpectedly, or the patch causes the evaluator to lose the ability to run applicable probes. A governance improvement cannot compensate for a confirmed behavior break in the primary valid-run aggregate. This asymmetric rule reflects the practical cost of damaging a working project while adding maintenance metadata.

Ta<sup>bl</sup>e 11: Dimension-specific interpretation of an acceptable score.
<table><tr><td>Dim.</td><td>Score 3</td><td>Score 4–5</td></tr><tr><td>D1</td><td>Tests cover selected behavior but important paths or CI execution are missing</td><td>Important behavior is exercised in a clean, repeatable test/CI path</td></tr><tr><td>D2</td><td>Tool configuration exists but scope or enforcement is in- complete</td><td>Quality checks run, target meaningful files, and expose failures</td></tr><tr><td>D3</td><td>Basic usage or contribution information exists with mate- rial omissions</td><td>A new maintainer can use, change, and report issues re- sponsibly</td></tr><tr><td>D4</td><td>Organization is understandable locally but coupling or duplication remains</td><td>Boundaries and dependency direction are maintainable and justified</td></tr><tr><td>D5</td><td>Installation works only with undocumented assumptions</td><td>Clean setup and execution are reproducible from declared metadata</td></tr><tr><td>D6</td><td>Dependencies are declared but audit or update evidence is incomplete</td><td>Dependency and security posture are controlled and in- spectable</td></tr></table>

## C<sub>.</sub>4<sub>.</sub> J<sub>u</sub>d<sub>ge ca</sub>lib<sub>ra</sub>ti<sub>on wor</sub>k<sub>s</sub>h<sub>ee</sub>t

Before the final scoring pass, evaluators should independently score a calibration set spanning low, medium, and high baseline quality. The worksheet records score, evidence cited, uncertainty, and whether the evaluator would request a maintainer review. Agreement is computed before adjudication; adjudication changes are retained as a separate field.

## C.5. Auxiliar<sub>y</sub> Ex<sub>p</sub>eriment Desi<sub>g</sub>n

## C.5.1. Fixed 10-re<sub>p</sub>ositor<sub>y</sub> batch

The completed auxiliary conditions use one frozen batch of ten repositories. The manifest contains repository, commit, language, size band, base mean, applicable dimensions, gate strength, and public/private status. No experiment may silently substitute an easier repository after execution begins.

Ta<sup>bl</sup>e 12: Required fields for the frozen auxiliary batch manifest.
<table><tr><td>Field</td><td>Requirement</td></tr><tr><td>Instance ID</td><td>Stable repository-level identifier</td></tr><tr><td>Commit</td><td>Exact base revision used by every condition</td></tr><tr><td>Strata</td><td>Size, baseline-quality, and gate-strength bins</td></tr><tr><td>Applicability</td><td>D1-D6 mask and reason for exclusions</td></tr><tr><td>Environment</td><td>Image or lockfile and evaluator version</td></tr><tr><td>Selection rationale</td><td>Why the instance represents the intended stratum</td></tr></table>

## C<sub>.</sub>5<sub>.</sub>2<sub>.</sub> Condition matrix

The completed condition matrix contains no-op, rollouts from the evaluated agent systems, mechanical retrofit, and rule-based baseline on the same ten repositories. Behavior-gate ablation uses all 60 instances because the public subset has no “none” gate cases. Repeated-seed, same-scafold, and expert conditions were not run.

## C.5.3. Ex<sub>p</sub>lorator<sub>y q</sub>uestions

The auxiliary analyses examine whether model rollouts outperform non-agent baselines, how behavior filtering changes observed outcomes, whether mutation-detected gates provide stronger regression evidence than blind or vacuous gates, which governance dimensions respond to repository-blind templates, and how intervention scope relates to behavior risk. These are exploratory questions, not preregistered hypotheses; the analyses are descriptive and use paired instance-level records where available.

Ta<sup>bl</sup>e 13: Completed condition matrix for the auxiliary study.
<table><tr><td>Condition</td><td>Model</td><td>Repositories</td><td>Main comparison</td></tr><tr><td>No-op</td><td>None</td><td>10</td><td>Scoring and environment stability</td></tr><tr><td>Agent rollouts</td><td>Ten evaluated systems</td><td>10</td><td>Open-ended repository governance</td></tr><tr><td>Mechanical</td><td>None</td><td>10</td><td>Template-only intervention</td></tr><tr><td>Rule</td><td>Deterministic</td><td>10</td><td>Checklist-oriented automation</td></tr><tr><td>Gate ablation</td><td>Re-score</td><td>60</td><td>Effect of behavior filtering</td></tr></table>

## C.6. Statistical Anal<sub>y</sub>sis and Re<sub>p</sub>ortin<sub>g</sub>

## C.6.1. Primar<sub>y</sub> estimands

For the auxiliary ten-repository batch, the principal comparisons are paired, within-repository diferences between each available agent rollout and each non-agent condition. The Kimi-K3-versus-template contrast is one such comparison, not the sole primary estimand. Secondary outcomes include raw dimension scores, behavior-breakage rate, no-regression rate, verified-claim rate, and resource usage. We report per-instance values before mean or median aggregation.

## C<sub>.</sub>6<sub>.</sub>2<sub>.</sub> M<sub>ea</sub>n<sub>–</sub>m<sub>e</sub>di<sub>a</sub>n r<sub>epo</sub>rtin<sub>g</sub> r<sub>a</sub>ti<sub>o</sub>n<sub>a</sub>l<sub>e</sub>

The main result table reports mean NGI as its single aggregate improvement measure. This choice keeps the leaderboard compact and makes the quantity directly interpretable as average normalized governance improvement across valid runs. Because behavior-broken runs are excluded from this conditional mean, the body also reports a common-valid paired comparison and a coverage-weighted full-pool NGI for the leading systems. Because the current public comparison contains 22 instances and one rollout per model-instance pair, the mean is still descriptive rather than an estimate of a stable population-level capability.

Median NGI is retained as a secondary diagnostic in the appendix rather than the main table. It is useful for identifying whether a model’s mean is driven by a small number of unusually large improvements: a mean substantially above the median indicates a right-skewed outcome distribution, while a close mean–median pair suggests more uniform instance-level behavior. The median is not used to replace the mean because it does not reflect the magnitude of improvements above or below the typical instance and can hide the contribution of a few high-value but valid interventions.

We therefore interpret the two statistics jointly when performing detailed analysis, but use only mean NGI in the compact main table. Any future claim about ranking stability will additionally report per-instance values, bootstrap intervals, valid coverage, and breakage rate. In particular, a high median cannot compensate for a high behavior-breakage rate, and a high mean cannot be interpreted as robust if it is supported by only a small valid denominator.

Model-level outcome profiles are reported in the body as Figure 3.

Instance-level NGI distributions are reported in the body as Figure 6.

## C.6.3. Uncertaint<sub>y</sub> re<sub>p</sub>ortin<sub>g</sub>

For ten repositories, asymptotic significance tests are not persuasive. We therefore emphasize exact paired diferences and bootstrap intervals over repository units, while clearly identifying the exploratory nature of the small batch. Repeated seeds, when available, add a second source of uncertainty and should be summarized with a hierarchical table separating between-repository and within-model variation.

## C.6.4. Per-dimension re<sub>p</sub>ortin<sub>g</sub>

For each condition, the appendix reports a six-column D1–D6 table with base mean, treated mean, mean raw change, fraction improved, fraction unchanged, and fraction regressed. The table also reports the number of applicable instances per dimension. This prevents aggregate NGI from hiding a condition that improves one dimension while damaging another.

The per-dimension improvement heatmap is reported in the body as Figure 6.

Ta<sup>bl</sup>e 14: Compact auxiliary-result summary.
<table><tr><td>Condition</td><td>Valid</td><td>Mean</td><td>Median</td><td>SD</td><td>Broken</td><td>Notes</td></tr><tr><td>No-op</td><td>10</td><td>-0.009</td><td>0.000</td><td>0.073</td><td>0</td><td>end-to-end variation</td></tr><tr><td>Kimi-K3</td><td>10</td><td>0.558</td><td>0.615</td><td>0.190</td><td>0</td><td>primary treatment</td></tr><tr><td>Mechanical</td><td>10</td><td>0.272</td><td>0.275</td><td>0.047</td><td>0</td><td>template baseline</td></tr><tr><td>Rule</td><td>10</td><td>0.254</td><td>0.283</td><td>0.064</td><td>0</td><td>deterministic baseline</td></tr><tr><td>Gate-off</td><td>all 60</td><td>mixed</td><td>mixed</td><td>n/a</td><td>n/a</td><td>paired ablation</td></tr></table>

## C<sub>.</sub>6<sub>.</sub>5<sub>.</sub> Mi<sub>ss</sub>i<sub>ng an</sub>d <sub>exc</sub>l<sub>u</sub>d<sub>e</sub>d <sub>runs</sub>

The analysis records excluded runs with an exclusion code before generating aggregate tables. Codes include infrastructure, invalid patch, behavior broken, missing evidence, and inapplicable dimension. Reported tables distinguish the denominator before exclusion from the denominator used for valid NGI.

## D<sub>.</sub> F<sub>u</sub>ll R<sub>esu</sub>lt T<sub>a</sub>bl<sub>es</sub>

The body reports every result as a figure. This appendix gives the underlying numbers.

## D<sub>.</sub>1<sub>.</sub> P<sub>u</sub>bli<sub>c s</sub>h<sub>are</sub>d<sub>-su</sub>b<sub>se</sub>t <sub>resu</sub>lt<sub>s</sub>

Ta<sup>bl</sup>e 15: Pu<sup>bl</sup>ic s<sup>h</sup>are<sup>d</sup>-su<sup>b</sup>set resu<sup>l</sup>ts (underlying Figure 3). Valid and broken counts are out of 22 instances. Valid runs preserve characterization behavior and contain scorable evidence; NGI is aggregated only over valid runs. Breakage rate is broken/(valid+broken). Claude used Claude Code; the other models used pi.
<table><tr><td>Model</td><td>Scaffold</td><td>Valid</td><td>Broken</td><td>Breakage</td><td>NGI mean</td><td>NGI median</td></tr><tr><td>GPT-5.6-Sol</td><td>pi</td><td>21</td><td>1</td><td>4.5%</td><td>0.2956</td><td>0.3333</td></tr><tr><td>Claude-Opus-5</td><td>Claude Code</td><td>18</td><td>4</td><td>18.2%</td><td>0.5293</td><td>0.6771</td></tr><tr><td>Kimi-K3</td><td>pi</td><td>21</td><td>1</td><td>4.5%</td><td>0.5760</td><td>0.6042</td></tr><tr><td>GLM-5.3-Flash</td><td>pi</td><td>17</td><td>5</td><td>22.7%</td><td>0.5760</td><td>0.5694</td></tr><tr><td>Qwen3.8-Max</td><td>pi</td><td>18</td><td>4</td><td>18.2%</td><td>0.4630</td><td>0.4792</td></tr><tr><td>GLM-5.2</td><td>pi</td><td>20</td><td>2</td><td>9.1%</td><td>0.4438</td><td>0.4167</td></tr><tr><td>DeepSeek-V4-Pro</td><td>pi</td><td>18</td><td>4</td><td>18.2%</td><td>0.3803</td><td>0.3917</td></tr><tr><td>GLM-5.3</td><td>pi</td><td>19</td><td>3</td><td>13.6%</td><td>0.3198</td><td>0.2500</td></tr><tr><td>DeepSeek-V4-Flash</td><td>pi</td><td>22</td><td>0</td><td>0.0%</td><td>0.2083</td><td>0.1667</td></tr><tr><td>MiniMax-M3</td><td>pi</td><td>22</td><td>0</td><td>0.0%</td><td>0.0568</td><td>0.0000</td></tr></table>

D<sub>.</sub>2<sub>.</sub> Frozen ten-re<sub>p</sub>ositor<sub>y</sub> auxiliar<sub>y</sub> batch

D<sub>.</sub>3<sub>.</sub> B<sub>e</sub>h<sub>av</sub>i<sub>or-ga</sub>t<sub>e a</sub>bl<sub>a</sub>ti<sub>on an</sub>d <sub>ga</sub>t<sub>e-s</sub>t<sub>reng</sub>th <sub>s</sub>t<sub>ra</sub>tifi<sub>ca</sub>ti<sub>on</sub>

D<sub>.</sub>4<sub>.</sub> N<sub>o-op</sub> d<sub>r</sub>ift <sub>an</sub>d <sub>per-</sub>i<sub>ns</sub>t<sub>ance</sub> b<sub>ase</sub>li<sub>ne compar</sub>i<sub>son</sub>

## D.5. Baseline <sub>p</sub>atch contents

The mechanical condition writes seven fixed artifacts regardless of repository content: a GitHub Actions CI workflow, a .pre-commit-config.yaml, a mypy.ini, CONTRIBUTING.md, SECURITY.md, an issue-template directory, and a smoke test containing assert True. The rule-based condition first probes which of these are absent, adds only those, and replaces the vacuous smoke test with importlib.import\_module against the package name inferred from the repository layout. Neither condition invokes a language model or modifies existing source files. Both had zero behavior-broken runs among the ten tested repositories; this is an observed result for this batch, not a guarantee that such edits cannot break behavior elsewhere.

Ta<sup>bl</sup>e 16: O<sup>b</sup>serve<sup>d</sup> resu<sup>l</sup>ts on t<sup>h</sup>e <sup>f</sup>rozen ten-repository auxi<sup>l</sup>iary <sup>b</sup>atc<sup>h</sup>. Models and non-agent conditions are ranked jointly by mean NGI. NGI statistics exclude behavior-broken runs; “strict” counts valid runs passing the strict-success criterion. Rows separated by less than the no-op standard deviation are not distinguished by this instrument.
<table><tr><td>Condition</td><td>Valid</td><td>Mean</td><td>Median</td><td>SD</td><td>Broken</td><td>Strict</td></tr><tr><td>GLM-5.3-Flash</td><td>9</td><td>0.585</td><td>0.583</td><td>0.220</td><td>1</td><td>0</td></tr><tr><td>Kimi-K3</td><td>10</td><td>0.558</td><td>0.615</td><td>0.190</td><td>0</td><td>2</td></tr><tr><td>Claude-Opus-5</td><td>9</td><td>0.457</td><td>0.722</td><td>0.354</td><td>1</td><td>0</td></tr><tr><td>GLM-5.2</td><td>9</td><td>0.378</td><td>0.390</td><td>0.165</td><td>1</td><td>0</td></tr><tr><td>Qwen3.8-Max</td><td>9</td><td>0.375</td><td>0.375</td><td>0.278</td><td>1</td><td>0</td></tr><tr><td>DeepSeek-V4-Pro</td><td>8</td><td>0.346</td><td>0.281</td><td>0.269</td><td>2</td><td>1</td></tr><tr><td>GLM-5.3</td><td>10</td><td>0.343</td><td>0.271</td><td>0.219</td><td>0</td><td>0</td></tr><tr><td>GPT-5.6-Sol</td><td>9</td><td>0.292</td><td>0.333</td><td>0.147</td><td>1</td><td>0</td></tr><tr><td>Mechanical (template)</td><td>10</td><td>0.272</td><td>0.275</td><td>0.047</td><td>0</td><td>0</td></tr><tr><td>Rule-based (deterministic)</td><td>10</td><td>0.254</td><td>0.283</td><td>0.064</td><td>0</td><td>0</td></tr><tr><td>DeepSeek-V4-Flash</td><td>10</td><td>0.247</td><td>0.174</td><td>0.193</td><td>0</td><td>0</td></tr><tr><td>MiniMax-M3</td><td>10</td><td>0.104</td><td>0.042</td><td>0.216</td><td>0</td><td>0</td></tr><tr><td>No-op (observed variation)</td><td>10</td><td>-0.009</td><td>0.000</td><td>0.073</td><td>0</td><td>0</td></tr></table>

<table><tr><td>Comparison</td><td>Repositories</td><td>Kimi-K3</td><td>GLM-5.3-Flash</td><td>Difference</td><td>Kimi wins</td></tr><tr><td>Common valid runs</td><td>17</td><td>0.547</td><td>0.576</td><td>-0.029</td><td>6</td></tr><tr><td>Full pool, broken = 0</td><td>22</td><td>0.550</td><td>0.445</td><td>+0.105</td><td></td></tr></table>

Ta<sup>bl</sup>e 17: Reliability-aware comparison of the two highest conditional-mean systems. The common-valid row includes only repositories preserved by both systems; the full-pool row assigns zero verified improvement to a behavior-broken treatment. The common-valid comparison has 6 Kimi wins, 10 GLM-5.3-Flash wins, and 1 tie.

<table><tr><td>Model</td><td>Valid runs only</td><td>Broken = 0 Broken = −1</td><td>Broken = −3</td></tr><tr><td>Kimi-K3</td><td>0.576</td><td>0.550 0.504</td><td>0.413</td></tr><tr><td>GLM-5.3-Flash</td><td>0.576</td><td>0.445 0.218</td><td>-0.237</td></tr><tr><td>Claude-Opus-5</td><td>0.529</td><td>0.433 0.251</td><td>-0.112</td></tr><tr><td>Qwen3.8-Max</td><td>0.463</td><td>0.379 0.197</td><td>-0.167</td></tr><tr><td>GLM-5.2</td><td>0.444</td><td>0.403 0.313</td><td>0.131</td></tr><tr><td>DeepSeek-V4-Pro</td><td>0.380</td><td>0.311 0.129</td><td>-0.234</td></tr><tr><td>GLM-5.3</td><td>0.320</td><td>0.276 0.140</td><td>-0.133</td></tr><tr><td>GPT-5.6-Sol</td><td>0.296</td><td>0.282 0.237</td><td>0.146</td></tr><tr><td>DeepSeek-V4-Flash</td><td>0.208</td><td>0.208 0.208</td><td>0.208</td></tr><tr><td>MiniMax-M3</td><td>0.057</td><td>0.057 0.057</td><td>0.057</td></tr></table>

Ta<sup>bl</sup>e 18: Failure-handling sensitivity over all 22 public repositories. “Valid runs only” is the reported conditional NGI. The full-pool columns assign behavior-broken runs NGI values of 0, −1, or −3 and average over all repositories. The −3 policy is the score-theoretic lower bound under the 1–5 rubric normalization (base $^ { 4 , }$ treated 1) for an applicable dimension. Kimi-K3 ranks first under all three failure-aware policies; conditional NGI ties it with GLM-5.3-Flash at the reported precision.

Ta<sup>bl</sup>e 19: Matc<sup>h</sup>e<sup>d</sup> per-instance comparison against t<sup>h</sup>e stronger temp<sup>l</sup>ate <sup>b</sup>ase<sup>l</sup>ine (underlying Figure 4). Each agent system is compared against max(mechanical, rule-based) on the same repository; ties are defined heuristically as diferences within the empirical no-op noise floor of 0.073, not as formal equivalence tests. Behavior-broken instances are excluded from the win/tie/loss counts.
<table><tr><td>Model</td><td>Wins</td><td>Ties</td><td>Losses</td><td>Broken</td><td>Mean Δ</td><td>SD Δ</td></tr><tr><td>Kimi-K3</td><td>9</td><td>0</td><td>1</td><td>0</td><td>+0.265</td><td>0.195</td></tr><tr><td>GLM-5.3-Flash</td><td>8</td><td>0</td><td>1</td><td>1</td><td>+0.298</td><td>0.221</td></tr><tr><td>Claude-Opus-5</td><td>5</td><td>0</td><td>4</td><td>1</td><td>+0.170</td><td>0.360</td></tr><tr><td>GLM-5.2</td><td>5</td><td>3</td><td>1</td><td>1</td><td>+0.092</td><td>0.181</td></tr><tr><td>Qwen3.8-Max</td><td>4</td><td>1</td><td>4</td><td>1</td><td>+0.089</td><td>0.268</td></tr><tr><td>GLM-5.3</td><td>4</td><td>4</td><td>2</td><td>0</td><td>+0.051</td><td>0.221</td></tr><tr><td>DeepSeek-V4-Pro</td><td>3</td><td>3</td><td>2</td><td>2</td><td>+0.066</td><td>0.283</td></tr><tr><td>GPT-5.6-Sol</td><td>2</td><td>5</td><td>2</td><td>1</td><td>+0.004</td><td>0.137</td></tr><tr><td>DeepSeek-V4-Flash</td><td>1</td><td>3</td><td>6</td><td>0</td><td>-0.045</td><td>0.213</td></tr><tr><td>MiniMax-M3</td><td>2</td><td>1</td><td>7</td><td>0</td><td>-0.188</td><td>0.231</td></tr></table>

Ta<sup>bl</sup>e 20: Per-<sup>d</sup>imension improvement rate on t<sup>h</sup>e <sup>f</sup>rozen ten-repository <sup>b</sup>atc<sup>h</sup>. Fraction of scored instances on which each dimension improved. In this batch, template baselines improve the three dimensions primarily evidenced through configuration and content, but not the two dimensions substantially informed by container execution and dependency analysis.
<table><tr><td>Condition</td><td>D1 Tests/CI</td><td>D2 Quality</td><td>D3 Docs</td><td>D4 Structure</td><td>D5 Repro.</td><td>D6 Dep./Sec.</td></tr><tr><td>Kimi-K3</td><td>100%</td><td>100%</td><td>80%</td><td>60%</td><td>90%</td><td>100%</td></tr><tr><td>GLM-5.3-Flash</td><td>100%</td><td>77%</td><td>88%</td><td>55%</td><td>88%</td><td>88%</td></tr><tr><td>Claude-Opus-5</td><td>88%</td><td>55%</td><td>55%</td><td>77%</td><td>55%</td><td>55%</td></tr><tr><td>GLM-5.2</td><td>88%</td><td>77%</td><td>44%</td><td>33%</td><td>88%</td><td>77%</td></tr><tr><td>GLM-5.3</td><td>90%</td><td>70%</td><td>20%</td><td>40%</td><td>70%</td><td>60%</td></tr><tr><td>DeepSeek-V4-Pro</td><td>87%</td><td>62%</td><td>50%</td><td>25%</td><td>62%</td><td>62%</td></tr><tr><td>DeepSeek-V4-Flash</td><td>90%</td><td>60%</td><td>30%</td><td>30%</td><td>60%</td><td>40%</td></tr><tr><td>Qwen3.8-Max</td><td>100%</td><td>33%</td><td>55%</td><td>55%</td><td>55%</td><td>33%</td></tr><tr><td>GPT-5.6-Sol</td><td>77%</td><td>33%</td><td>77%</td><td>11%</td><td>33%</td><td>44%</td></tr><tr><td>MiniMax-M3</td><td>40%</td><td>10%</td><td>10%</td><td>10%</td><td>40%</td><td>30%</td></tr><tr><td>Mechanical (template)</td><td>100%</td><td>100%</td><td>90%</td><td>20%</td><td>0%</td><td>0%</td></tr><tr><td>Rule-based (deterministic)</td><td>100%</td><td>100%</td><td>100%</td><td>20%</td><td>0%</td><td>0%</td></tr><tr><td>No-op (observed variation)</td><td>0%</td><td>0%</td><td>10%</td><td>20%</td><td>10%</td><td>0%</td></tr></table>

## D<sub>.</sub>6<sub>.</sub> Additi<sub>ona</sub>l <sub>ana</sub>l<sub>yses</sub>

Beyond model averages, useful stratifications include repository language, project size, baseline score, number of applicable dimensions, test-suite maturity, and dependency-manager type. These analyses distinguish absolute change from headroom-normalized change and relate both to patch size and verification efort. Intervention locality across tests, workflows, documentation, packaging, dependency manifests, and source modules distinguishes targeted governance edits from broad source rewrites.

The public subset provides one rollout per model-instance pair, so inferential claims are limited accordingly. Section 5.10 quantifies the cost of reducing this limitation: reaching a resolution of 0.10 NGI requires roughly 110 paired observations, about five rollouts per model-instance pair on the current subset. Private instances and unbalanced model coverage are not treated as interchangeable with the shared public subset, and exclusions are reported before aggregation.

Ta<sup>bl</sup>e 21: Be<sup>h</sup>avior-gate a<sup>bl</sup>ation over all instances available per model. “Gate of” counts every scored instance; “gate on” is the current protocol. The model ordering is identical under both settings, and the inflation term is small and signed in both directions.
<table><tr><td>Model</td><td>Scored</td><td>Valid</td><td>NGI gate off</td><td>NGI gate on</td><td>Inflation</td><td>Breakage</td></tr><tr><td>Claude-Opus-5</td><td>22</td><td>18</td><td>+0.677</td><td>+0.677</td><td>+0.000</td><td>18%</td></tr><tr><td>Kimi-K3</td><td>60</td><td>56</td><td>+0.517</td><td>+0.531</td><td>-0.014</td><td>7%</td></tr><tr><td>GLM-5.3-Flash</td><td>58</td><td>52</td><td>+0.476</td><td>+0.458</td><td>+0.017</td><td>12%</td></tr><tr><td>Qwen3.8-Max</td><td>59</td><td>52</td><td>+0.375</td><td>+0.396</td><td>-0.021</td><td>13%</td></tr><tr><td>GLM-5.2</td><td>59</td><td>53</td><td>+0.375</td><td>+0.367</td><td>+0.008</td><td>11%</td></tr><tr><td>DeepSeek-V4-Pro</td><td>59</td><td>51</td><td>+0.367</td><td>+0.367</td><td>+0.000</td><td>15%</td></tr><tr><td>GPT-5.6-Sol</td><td>60</td><td>57</td><td>+0.229</td><td>+0.250</td><td>-0.021</td><td>6%</td></tr><tr><td>GLM-5.3</td><td>60</td><td>54</td><td>+0.215</td><td>+0.208</td><td>+0.007</td><td>11%</td></tr><tr><td>DeepSeek-V4-Flash</td><td>60</td><td>57</td><td>+0.125</td><td>+0.125</td><td>+0.000</td><td>6%</td></tr><tr><td>MiniMax-M3</td><td>60</td><td>57</td><td>+0.000</td><td>+0.000</td><td>+0.000</td><td>6%</td></tr></table>

Ta<sup>bl</sup>e 22: Gate-strengt<sup>h</sup> strati<sup>fi</sup>cation over all 60 instances and ten models (underlying Figure 5a). Vacuous and absent gates cannot observe a regression, so their breakage rate is a measurement artifact rather than evidence of safer behavior.
<table><tr><td>Gate strength</td><td>Instances</td><td>Scored runs</td><td>NGI median</td><td>NGI mean</td><td>Breakage</td><td>No-regression</td></tr><tr><td>detected</td><td>24</td><td>228</td><td>+0.333</td><td>+0.321</td><td>13%</td><td>86%</td></tr><tr><td>blind</td><td>27</td><td>247</td><td>+0.292</td><td>+0.315</td><td>8%</td><td>88%</td></tr><tr><td>vacuous</td><td>3</td><td>28</td><td>+0.271</td><td>+0.289</td><td>0%</td><td>86%</td></tr><tr><td>none</td><td>6</td><td>54</td><td>+0.250</td><td>+0.257</td><td>n/a</td><td>85%</td></tr></table>

Ta<sup>bl</sup>e 23: Instances t<sup>h</sup>at <sup>d</sup>ri<sup>f</sup>t un<sup>d</sup>er t<sup>h</sup>e no-op con<sup>d</sup>ition. Dimension changes are base → treated on an identical repository. Five of ten instances drift; the median is 0.000 and the standard deviation is 0.073.
<table><tr><td>Instance</td><td>NGI</td><td>Dimension drift</td></tr><tr><td>DreamWall-Animation/dwpicker</td><td>-0.1667</td><td>D44→ 3</td></tr><tr><td>jiajun613/Efficient-WAM</td><td>-0.1111</td><td>D5 2 → 1</td></tr><tr><td>TsingZ0/HtFLlib</td><td>+0.0972</td><td>D4 3 → 4, D5 1 → 2</td></tr><tr><td>fblissjr/ComfyUI-QwenImageWanBridge</td><td>+0.0556</td><td>D4 3 → 4</td></tr><tr><td>rivitna/Malware</td><td>+0.0333</td><td>D3 2 → 3</td></tr></table>

Repeated seeds, same-scafold reruns, and expert treatments are outside the primary comparison. Repeated seeds should cover small and large repositories, strong and weak base governance, and each gate-strength category. Same-scafold reruns are important because one model used Claude Code while the other nine used pi.

## D.7. A corrected accountin<sub>g</sub> bu<sub>g</sub>

An earlier version of the pipeline recorded all six gate=none instances as behavior-broken, because applying a nonexistent verification patch exits with a nonzero status. The gate had in fact never run on those instances. Correcting this lowered per-model breakage rates over all 60 instances from 15–17% to 6–7% for the afected models; the rates in Table 21 are post-correction. The public 22-instance subset is unafected, since every public instance carries a verification patch, so Table 1 required no revision.

Ta<sup>bl</sup>e 24: Per-instance NGI on t<sup>h</sup>e <sup>f</sup>rozen <sup>b</sup>atc<sup>h</sup>, one representative model against the three non-agent conditions. Bold marks the best of the four on each repository. Qwen3.8-Max loses to at least one template on four of ten repositories.
<table><tr><td>Instance</td><td>Qwen3.8-Max</td><td>Mechanical</td><td>Rule-based</td><td>No-op</td></tr><tr><td>sleeepeer/PoisonedRAG</td><td>+0.750</td><td>+0.264</td><td>+0.292</td><td>+0.000</td></tr><tr><td>rivitna/Malware</td><td>+0.708</td><td>+0.236</td><td>+0.275</td><td>+0.033</td></tr><tr><td>THUNLP-MT/StreamingBench</td><td>+0.708</td><td>+0.347</td><td>+0.208</td><td>+0.000</td></tr><tr><td>SkyworkAI/Skywork-Skills</td><td>+0.667</td><td>+0.333</td><td>+0.333</td><td>+0.000</td></tr><tr><td>DreamWall-Animation/dwpicker</td><td>+0.500</td><td>+0.286</td><td>+0.250</td><td>-0.167</td></tr><tr><td>avbor/HomeAssistantConfig</td><td>+0.375</td><td>+0.308</td><td>+0.292</td><td>+0.000</td></tr><tr><td>fblissjr/ComfyUI-QwenImageWanBridge</td><td>+0.167</td><td>+0.292</td><td>+0.208</td><td>+0.056</td></tr><tr><td>jiajun613/Efficient-WAM</td><td>+0.167</td><td>+0.236</td><td>+0.292</td><td>-0.111</td></tr><tr><td>dama-cyber/magic-distillation</td><td>+0.083</td><td>+0.208</td><td>+0.097</td><td>+0.000</td></tr><tr><td>TsingZ0/HtFLlib</td><td>-0.042</td><td>+0.208</td><td>+0.292</td><td>+0.097</td></tr></table>

## E<sub>.</sub> R<sub>epro</sub>d<sub>uc</sub>ibilit<sub>y an</sub>d R<sub>e</sub>l<sub>ease</sub>

## E<sub>.</sub>1<sub>.</sub> Director<sub>y</sub>-level release la<sub>y</sub>out

The release mirrors the separation between public tasks, protected verification assets, results, and traces. Public artifacts are suficient to reconstruct the announced experiments without revealing private outcomes or hidden answers. A manifest at the root maps every paper table to a script and every script output to a versioned input.

Ta<sup>bl</sup>e 25: Recommended release package layout.
<table><tr><td>Path</td><td>Content</td></tr><tr><td>tasks/</td><td>Public task metadata, commits, licenses, and applicability masks</td></tr><tr><td>environments/</td><td>Container definitions or environment locks</td></tr><tr><td>probes/</td><td>Versioned governance and behavior probe definitions</td></tr><tr><td>runs/</td><td>Canonical traces, patches, budgets, and evidence records</td></tr><tr><td>results/</td><td>Machine-readable per-instance and aggregate scores</td></tr><tr><td>analysis/</td><td>Scripts that regenerate paper tables and figures</td></tr><tr><td>docs/</td><td>Construction notes, scoring rubric, changelog, and limitations</td></tr></table>

## E<sub>.</sub>2<sub>.</sub> Artifact com<sub>p</sub>leteness rules

An artifact is complete only when its manifest exists, its digest matches the stored file, and the downstream table can identify whether it was used. An empty directory is not treated as evidence of absence. If a run is interrupted, the manifest records the last completed stage and the reason for interruption.

## E.3. Securit<sub>y</sub> and <sub>p</sub>rivac<sub>y</sub>

Repository licenses, credentials, private issue data, evaluator patches, and model-provider traces require separate review for release. Secret scanning is performed on artifacts and logs, but a positive scanner result is manually adjudicated because repository content may contain intentional examples. Private tasks remain excluded from public traces, score files, and aggregate tables.

## E.4. Result re<sub>g</sub>eneration <sub>p</sub>rocedure

The auxiliary settings and results are incorporated in the main text and Figure 4. Regeneration scripts produce summaries from the machine-readable records, preserve the frozen manifest, distinguish valid from broken runs, and support the full PDF visual check.

## E<sub>.</sub>5<sub>.</sub> D<sub>a</sub>t<sub>a</sub> Di<sub>c</sub>ti<sub>onary an</sub>d R<sub>esearc</sub>h<sub>er</sub> Ch<sub>ec</sub>kli<sub>s</sub>t

## E<sub>.</sub>5<sub>.</sub>1<sub>.</sub> C<sub>ore</sub> id<sub>en</sub>tifi<sub>ers</sub>

The following identifiers remain stable across releases. The instance identifier names a repository snapshot, the run identifier names one model execution, and the evidence identifier names one probe invocation. A score record must reference all three identifiers.

Ta<sup>bl</sup>e 26: Data dictionary for the benchmark records.
<table><tr><td>Name</td><td>Type</td><td>Definition</td></tr><tr><td>instance_id</td><td>string</td><td>Repository and pinned-snapshot identifier</td></tr><tr><td>run_id</td><td>string</td><td>Unique model/scaffold execution identifier</td></tr><tr><td>model</td><td>string</td><td>Exact model name or provider label</td></tr><tr><td>scaffold</td><td>string</td><td>Agent framework and runner version</td></tr><tr><td>base_commit</td><td>string</td><td>Commit used to reconstruct the untreated state</td></tr><tr><td>patch_digest</td><td>string</td><td>Digest of the submitted unified patch</td></tr><tr><td>behavior</td><td>enum</td><td>preserved, broken, invalid, or unavailable</td></tr><tr><td>gate_strength</td><td>enum</td><td>detected, blind, vacuous, or none</td></tr><tr><td>ngi</td><td>float</td><td>Headroom-normalized governance improvement</td></tr><tr><td>evidence_status</td><td>enum</td><td>pass, fail, timeout, unavailable, or not applicable</td></tr></table>

## E<sub>.</sub>5<sub>.</sub>2<sub>.</sub> R<sub>esearc</sub>h<sub>er c</sub>h<sub>ec</sub>kli<sub>s</sub>t

Before claiming an auxiliary result, confirm that the ten repositories were frozen in advance, every condition used the same base commit, no hidden evaluator asset was exposed, no-op results are stable, broken behavior runs are not silently dropped, score denominators are printed, and every aggregate number can be regenerated from the archived per-instance records. The ten-repository baselines and gate-ablation values are measured; repeated seeds, same-scafold reruns, and expert treatments are outside the primary comparison.

## E.5.3. Inter<sub>p</sub>retation checklist

When comparing conditions, first check paired coverage, then behavior validity, then per-dimension change, and only then aggregate NGI. A higher aggregate score is not interpreted as an improvement if it is caused by a smaller denominator, a weaker gate, or a higher breakage rate. This ordering is especially important for the small ten-repository pilot, where one instance can materially afect the mean.