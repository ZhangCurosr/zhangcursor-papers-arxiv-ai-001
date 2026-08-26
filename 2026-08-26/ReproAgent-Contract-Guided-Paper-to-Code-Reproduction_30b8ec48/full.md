# ReproAgent: Contract-Guided Paper-to-Code Reproduction

Xue Hu<sup>∗1</sup>, Zewei Pan<sup>∗2</sup>, Zhongyuan Wang<sup>1</sup>, Zhou Liu<sup>3</sup>, Zeli Su<sup>4</sup>, Wentao Zhang<sup>†3,5,6</sup>

<sup>1</sup>Beihang University, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>Peking University

<sup>4</sup>Minzu University of China, <sup>5</sup>Zhongguancun Academy

<sup>6</sup>Beijing Key Laboratory of Data Intelligence and Security (Peking University)

kernel@buaa.edu.cn, pzp0057@sjtu.edu.cn, zywang111@buaa.edu.cn

zhouliu25@stu.pku.edu.cn, rickamorty@muc.edu.cn, wentao.zhang@pku.edu.cn

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.

## Abstract

Paper-to-code reproduction asks scientific AI agents to turn research papers into executable repositories that preserve the paper’s method, protocol and artifacts. This is difficult because the specification is split: explicit paper content such as algorithms, metrics and artifacts is often lost across long agent trajectories, while implicit details such as framework defaults and conventions inherited from related work are absent from the paper. We introduce REPROA-GENT, a four-stage Prepare–Plan–Generate– Repair pipeline built around a persistent implementation contract with two channels: an implementation-requirement channel that turns paper snippets into code obligations, and a reference-evidence channel that retrieves content and structure evidence from related repositories. Both are bound to work packages, projected into file-level contracts, and consumed across generation and repair. On PaperBench Code-Dev (Starace et al., 2025), ReproAgent reaches the highest mean score among samebackbone scaffolds under both Claude-Sonnet-4.5 and Gemini-3-Flash. End-to-end channel ablations and per-paper cases support the contribution of both channels. Code and experimental artifacts are publicly available.<sup>1</sup>

## 1 Introduction

Large language models are increasingly used to assist scientific research, but a long-standing reproducibility crisis persists (Baker, 2016; Pineau et al., 2021). Systematic reviews of NLP reproduction studies report that only a small fraction of attempted reproductions recover the original scores (Belz et al., 2021), and even strong LLM judges detect only 46.7% of real-world discrepancies between papers and their accompanying code (Baumgärtner and Gurevych, 2026). Together, faster method production and persistent reproducibility gaps make it harder to validate new methods, compare baselines fairly, and build on prior work. An automated framework that turns each new paper into a faithful, runnable repository, rather than merely executable code, would make paper-specific methods, protocols, and artifacts easier to inspect and reuse.

Recent paper-to-code agents take a step in this direction (Seo et al., 2026; Xiang et al., 2025; Zhao et al., 2025; Zhou et al., 2025), but a substantial fidelity gap remains. A recurring failure mode is runnable but unfaithful code: a generated repository may import correctly, pass smoke tests, and even produce plausible experimental results, while still dropping or misimplementing key paperspecific details, such as substituting a generic training loop for the paper’s algorithm, omitting the decisive metric or artifact, or altering a loss term. Runtime success alone is therefore an insufficient proxy for faithful paper reproduction.

We identify a split-specification problem behind these failures. A paper states some implementation requirements explicitly, including algorithms, metrics, artifacts and evaluation protocols. However, many decisions needed to build a faithful repository are implicit in framework defaults, cited implementations and community conventions. Existing paper-to-code agents mainly condition generation on the paper text and transient planning context, without a persistent object that carries paper-derived obligations or binds implicit details to external evidence. As a result, explicit requirements drift out of context across long trajectories, while implicit details fall back on model priors rather than grounded evidence. Faithful reproduction therefore needs a persistent implementation contract that preserves paper-derived obligations and attaches external repository evidence to the implementation decisions it supports.

REPROAGENT instantiates this idea with a global implementation contract containing two channels. The implementation-requirement channel extracts explicit paper knowledge as code obligations, while the reference-evidence channel grounds implicit repository knowledge in related implementations. Both channels are bound to work packages and projected into file-level contracts, so the same contract guides generation and serves as the audit target for detecting missing obligations, unsupported implementation choices, broken interfaces, and runtime failures.

![](images/0de17145f0f8f082ece291b45dbd86f8f65ee830573ec4f12a5a65a45787dbed.jpg)  
Figure 1: Overview of REPROAGENT. A four-stage pipeline, Prepare, Plan, Generate, and Repair, carries an implementation-requirement channel and a reference-evidence channel into a shared file-level implementation contract. The implementation-requirement channel preserves paper-derived code obligations across generation and repair, while the reference-evidence channel supplies package-local content and structure evidence from related repositories.

REPROAGENT carries this contract through a four-stage pipeline. Prepare segments the paper, extracts implementation requirements, and builds a reference-repository index. Plan organizes requirements into work packages, binds package-local reference evidence, derives the target architecture, and freezes file-level contracts over obligations, evidence, validation checks, and file responsibilities. Generate produces the repository file by file under these local contracts. Repair audits the result against the same contract together with runtime checks, so failures are reported as violated paperderived obligations, missing evidence use, broken interfaces, or execution failures rather than only as crashing commands.

We evaluate REPROAGENT on PaperBench Code-Dev (Starace et al., 2025), using repositorylevel rubric grading to test whether generated code implements the target paper rather than merely executing. We run REPROAGENT with two backbones, Claude-Sonnet-4.5 and Gemini-3-Flash, and under each backbone REPROAGENT reaches the highest mean score among same-backbone scaffolds. Ablations of the two contract channels together with a per-paper case analysis support that both the implementation-requirement and reference-evidence channels contribute to faithful reproduction.

Our main contributions are:

• We formulate paper-to-code reproduction as contract-guided repository construction under partial paper specifications, where faithful generation requires preserving explicit paper obligations and grounding implicit implementation choices.

• We introduce a two-channel implementation contract over paper-derived requirements and relatedrepository evidence, with work-package bindings, file-level obligations, provenance records, and typed requirement-review/runtime reports that preserve both traces across generation and repair.

• We instantiate this contract in REPROA-GENT, a four-stage Prepare–Plan–Generate– Repair pipeline, and evaluate it on the full 20- paper PaperBench Code-Dev suite with samebackbone comparisons and ablations that remove each contract channel.

## 2 Related Work

LLM agents for research and code. LLM agents have been applied to research workflows (Baek et al., 2024; Lu et al., 2024; Schmidgall et al., 2025), ML engineering (Chan et al., 2024; Tang et al., 2024), and code-generation or repository-level coding settings (Qian et al., 2024; Islam et al., 2024; Hong et al., 2024; Yang et al., 2024; Xia et al., 2025; Zhang et al., 2024). Paper reproduction differs because the target is a scientific procedure described only partially in natural language: the repository must be faithful to paper-specific methods, protocols, and artifacts, not merely pass a generic execution check.

Paper-to-code reproduction. Following DLPaper2Code (Sethi et al., 2018), recent systems extend reproduction to whole repositories: Paper-Coder, the system introduced by Seo et al. (Seo et al., 2026), stages planning, analysis, and coding; Sci-Reproducer (Xiang et al., 2025) highlights missing-algorithm bottlenecks; AutoP2C (Lin et al., 2025) and HiRAS (Hong et al., 2026) push multimodal parsing and hierarchical supervision; AutoReproduce (Zhao et al., 2025) retrieves along citation graphs; DeepCode (Li et al., 2025b) treats reproduction as agentic coding. To our knowledge, prior systems do not make both paper-derived obligations and retrieved repository evidence a persistent, file-level audit object shared across planning, generation, and repair, which is the traceability layer REPROAGENT targets.

Reference retrieval and contracts. Retrievalaugmented code generation depends on retrieval granularity (Chen et al., 2024) and pulls evidence from documentation (Zhou et al., 2023; Pimparkhede et al., 2024), repositories (Zhang et al., 2023; Cheng et al., 2024), or code graphs (Li et al., 2025a), but does not tie each snippet to the scientific requirement it supports. Our contract view connects to requirements traceability and design by contract (Gotel and Finkelstein, 1994; Meyer, 1992; Cleland-Huang et al., 2014), adapted to settings where requirements are extracted from paper snippets rather than written formally. REPROA-GENT therefore splits reference evidence into content evidence for reuse or adaptation and structure evidence for architecture planning, and binds both to work-package and file-level contracts that persist across the pipeline.

Evaluation and fidelity. PaperBench (Starace et al., 2025) grades reproductions against rubrics, with related benchmarks studying reproducibility and paper-code mismatch (Yan et al., 2025; Siegel et al., 2024; Baumgärtner and Gurevych, 2026).

RePro (Zhou et al., 2025) applies paper-derived criteria as post-hoc refinement, after a candidate repository exists. REPROAGENT differs by placing the same paper-derived requirements into the forward generation contract, so the audit target is fixed before generation rather than reconstructed afterwards.

## 3 Method

REPROAGENT turns a paper into a repository by maintaining two persistent contract channels, shown in Figure 2. The implementationrequirement channel preserves what the target paper explicitly says must be implemented. The reference-evidence channel supplies evidence for implicit repository knowledge: implementation conventions, file structure, entry points, and protocol details from cited or related repositories. Both channels are bound to work packages and projected into file-level contracts, so information extracted early remains visible during planning, generation, validation, and repair. A work package is a small planning unit that bundles related requirements with the evidence supporting them; we give its full schema in §3.3.

## 3.1 Implementation Requirements

An implementation requirement is a paper snippet that must be realized in code, and we represent it as a unit

$$
u = ( { \mathrm { i d } } , { \mathrm { s r c } } , { \mathrm { s t m t } } , { \mathrm { c i t e } } , { \mathrm { s u r f } } , { \mathrm { a r t } } , { \mathrm { s t a t u s } } ) .
$$

Here id is a stable identifier, src records the paper source span the unit is drawn from, and stmt is a normalized statement of the obligation. The remaining four fields specify how the unit travels through the rest of the pipeline: cite stores explicit reference cues when they are present in the paper, and the reference-evidence channel attaches priorwork snippets through unit IDs, source anchors, and match records (§3.2); surf names the expected code surfaces, such as a model class, loss function, optimizer, metric, CLI option, config field, training script, or artifact writer; art enumerates the artifacts the unit should produce; and status is a lifecycle flag that stays active until the unit is realized. A concrete example of such a unit is given in Appendix A.1.

Extraction. We first segment the paper into section-level chunks, falling back to paragraphlevel splits when a section is too long for a single LLM pass. From each chunk, we extract only snippets that imply concrete implementation work, such as algorithmic steps, loss formulations, optimizer and training settings, dataset and preprocessing rules, and reported metrics or artifacts; pure motivation, related work narration, and discussion are skipped.

![](images/f420c686bcbf23634c3431dca73c56d601e16065e5d705504804c3f25812c624.jpg)  
Figure 2: REPROAGENT overview. The implementation-requirement channel extracts code-facing obligations from the target paper, while the reference-evidence channel retrieves content and structure evidence from related repositories. The shared contract binds both channels into work packages, global and file-level contracts, generation, and validation, yielding a repository that is checked for both requirement fidelity and runtime correctness. Representative unit and evidence records appear in Appendix A.

Preservation invariant. To make this preservation operational, we attach a coverage invariant to every active unit so that any silent loss surfaces as a checkable failure inside the pipeline. For every active unit $u \in \mathcal { U } _ { \mathrm { a c t i v e } }$ , the contract requires

$$
\begin{array} { l } { \displaystyle \mathrm { o w n } ( u ) , \mathrm { p r o j } ( u ) , \mathrm { p r o v } ( u ) \neq \emptyset , } \\ { \displaystyle \mathrm { v e r d i c t } ( u ) \in \mathcal { V } , } \end{array}
$$

where Y is the fixed verdict set returned by requirement review, with four possible values: pass, missing, wrong-surface and inconsistent-with-source. The four predicates correspond to four pipeline stages:

• Ownership. Plan assigns the unit to at least one work package.

• File projection. File planning assigns the unit to at least one file contract whose declared surfaces cover the unit’s expected surfaces.

• Provenance. Generation emits a record linking the produced file region back to the unit.

• Verdict. Requirement review returns a value in Y.

A failed predicate creates a contract violation report and a repair obligation; if the repair budget is exhausted, the unresolved verdict is retained in the diagnostics rather than silently absorbed.

## 3.2 Reference Evidence

Papers often leave implementation details implicit: a paper may cite a prior method without restating its update rule, or refer to a benchmark’s “standard” data augmentation whose canonical form lives in a reference repository. REPROAGENT therefore collects reference evidence from related repositories listed in the paper’s bibliography, and represents each evidence item as

$$
e = ( { \mathrm { r e p o } } , { \mathrm { p a t h } } , { \mathrm { r e g i o n } } , { \mathrm { u n i t s } } , { \mathrm { r o l e } } , { \mathrm { b a s i s } } ) .
$$

Here repo identifies the source repository, path the file inside it, and region the matched line range; units lists the requirement IDs the snippet supports, role records whether the snippet is meant for local reuse or for adaptation as an interface pattern, and basis is a structured match record explaining why the snippet is relevant. Each item is bound to the work package that owns its supported units, so reference code enters generation through a specific package contract rather than as global context. A concrete evidence item is given in Appendix A.2.

Content evidence. For each candidate reference repository, REPROAGENT indexes file paths, imports, exported symbols, scripts, configurations, and entry points described in the README. A snippet is retained when one of three signals connects it to an owned unit, an expected artifact, or a declared implementation surface: a symbol or interface name match, a path or configuration cue, or a structured LLM judgment. The judgment is stored in basis as the supported unit IDs, the matching surface or artifact, and whether the snippet should be reused locally or adapted as an interface pattern, rather than as a free-form note. The role field then acts as a use constraint: local reuse covers reusable formulas or transforms, while interface adaptation covers structural patterns that should be re-implemented under the package’s own interface, not copied wholesale.

Structure evidence. Reference repositories also expose where code should live: file trees, entry scripts, module boundaries, configuration conventions, dependency files, and technology-stack hints. Architecture planning combines these structural priors with the task and requirements to decide the target file tree, stable interfaces, entry points, and dependency graph. Content evidence therefore informs how to implement an obligation, while structure evidence informs where the implementation should live.

## 3.3 Work Packages and the Global Contract

A work package is the local planning unit that binds requirements to evidence. Each package records its goal, the requirement IDs it owns, its dependencies on other packages, the public interface it exposes, the artifacts it should produce, and the evidence items attached to it.

After evidence binding, the planner freezes a global contract

$$
\mathcal { G } = ( \mathcal { U } , \mathcal { W } , \mathcal { E } , \mathcal { A } , \mathcal { V } ) ,
$$

where U is the active requirement set, W the work packages, E the package-bound evidence, $\mathcal { A }$ the target architecture, and V the validation plan. Once $\mathcal { G }$ is frozen, later stages can refine local implementation choices and add derived helper files, but the active requirements, package ownerships, and evidence-package bindings carried by $\mathcal { G }$ stay fixed. File planning then projects $\mathcal { G }$ into file contracts $\mathcal { C } _ { F } = \{ c _ { f } \}$ , where each $c _ { f }$ specifies its target path, owner package, owned units, imported and exported symbols, attached evidence, produced artifacts, and review checks. Every generated file therefore has a per-file specification grounded in the global contract.

## 3.4 Generation, Validation, and Repair

Generation proceeds file by file in dependency order. Each call is conditioned on the file contract $c _ { f } .$ , the evidence attached to its owner package, and the already-generated dependency files needed for the local interface; reference code from sibling packages is not injected as global context. This package-local scoping keeps each produced file auditable: a training loop is generated against training requirements and training evidence, while an evaluation script is generated against metric and artifact requirements. Each generated file returns a provenance record linking file regions back to the requirement and evidence IDs they realize, and lists implementation choices that are inferred rather than directly specified by the contract.

Validation produces two typed reports. Requirement review evaluates each active unit against its file contract and provenance record, and returns a verdict in $\mathcal { V }$ . Runtime validation runs dependency installation, import checks, declared entry points, smoke commands, and benchmark commands when available. The two reports surface different failures: runtime validation catches broken code and missing artifacts, while requirement review catches runnable but paper-inconsistent implementations.

When either report fails, REPROAGENT issues a repair ticket rather than regenerating the repository. The ticket carries the failed units, target files, attached evidence, the edits required, and the units that already passed and must remain protected. Repair is file-local unless the ticket identifies an interface-level failure that requires coordinating two or more files, and repository regions whose contracts are unchanged skip redundant validation. The repair loop is bounded by a fixed budget; we use the same budget within each backbone setting and report the setting-specific value in §4.3.

## 4 Experimental Setup

## 4.1 Benchmark and Backbones

We evaluate on PaperBench Code-Dev (Starace et al., 2025), which contains 20 ICML 2024 papers and hierarchical rubrics for repository-level code reproduction; the rubrics grade whether a generated repository implements paper-specific methods, protocols, and artifacts rather than only running. On this benchmark, the evaluation has three layers. The full-suite layer compares the primary REPROA-GENT run with reported PaperBench Code-Dev systems, using Claude-Sonnet-4.5<sup>2</sup> as the generation backbone. The same-backbone layer compares RE-PROAGENT with simpler scaffolds under Gemini-3- Flash<sup>3</sup>, the backbone shared with BasicAgent, Iter-Agent and AiScientist. The ablation layer removes each contract channel under the same Gemini-3- Flash backbone.

## 4.2 Comparisons and Metrics

External rows reuse the model, scaffold, and runtime budget reported by their source. The AiScientist engineering report provides the BasicAgent, IterAgent, and AiScientist scaffold rows used in the same-backbone comparison. The aggregate full-suite figure includes only Code-Dev rows, and the per-paper table expands the main comparison to BasicAgent, IterAgent, AiScientist, DeepCode, and both REPROAGENT backbones; Appendix B lists the additional external rows.

We follow the Code-Dev judging protocol and report macro-averaged PaperBench score as the primary metric: weighted leaf scores aggregate to a per-paper score, and the 20 paper scores are averaged. Token, time, cost, and judge-provenance diagnostics are reported in the appendix.

## 4.3 Implementation and Ablations

REPROAGENT is implemented as a four-stage pipeline: the paper is chunked by section, reference repositories come from PaperBench metadata and related-code search subject to the benchmark blacklist on official target repositories, and planning freezes a global contract before file-level generation. Repair combines requirement review with sandboxed runtime validation, so failures become targeted tickets that patch only affected files.

![](images/77cfb3c5bbcb5bfd34143d1f33044f0f63577a43a7bc83b635acae2cd77e71b9.jpg)  
Figure 3: Full-suite PaperBench Code-Dev comparison across reported paper-to-code systems.

We cap stage-review repair at three attempts and repository-level repair at five rounds, uniformly across the Claude-Sonnet-4.5 main run and all Gemini-3-Flash full and ablation runs; holding this budget fixed isolates the channel effect from repairbudget effects. In total we report 20 papers × 4 REPROAGENT settings = 80 reproduction runs, one run per (paper, setting) pair.

Channel ablations. On the same Gemini-3-Flash backbone, No Implementation Requirements skips requirement extraction and requirement review, so architecture planning falls back on the paper’s section outline and reference-derived module structure; No Reference Evidence skips referencerepository surveying, so generation relies on requirement-derived contracts without bound content or structure evidence. All other components and repair budgets are held fixed.

## 5 Results and Analysis

## 5.1 Full-Suite Comparison

Figure 3 reports mean PaperBench Code-Dev scores across published paper-to-code systems. REPROAGENT with Claude-Sonnet-4.5 reaches 73.7, above PaperCoder (Seo et al., 2026) (45.1), AutoP2C (Lin et al., 2025) (49.2), AutoReproduce (Zhao et al., 2025) (49.6), and Deep-Reproducer (Chen et al., 2025) (63.2), and edges past the concurrent state-of-the-art DeepCode at 73.5.

Tables 1 and 2 expand this view into the 20 targets for each backbone; additional external rows appear in Appendix B. We hypothesize that the gain comes from the contract itself: requirement and evidence bindings that survive across files keep

We next isolate the contribution of each contract channel under Gemini-3-Flash. Table 3 summarizes the aggregates: removing reference evidence drops the mean by 18.1 and removing implementation requirements drops it by 14.1. The full setting beats both ablations on every one of the 20 targets, so each channel contributes capability that the other cannot recover.

<table><tr><td>Paper</td><td>BasicAgent</td><td>IterAgent</td><td>DeepCode</td><td>REPROAGENT (ours)</td></tr><tr><td>Adaptive Pruning</td><td>16.6</td><td> $2 2 . 4 _ { \uparrow 5 . 8 }$ </td><td> $5 4 . 4 _ { \uparrow 3 7 . 8 }$ </td><td> ${ \bf 6 7 . 8 } _ { \uparrow 5 1 . 2 }$ </td></tr><tr><td>All in One</td><td>24.8</td><td> $1 1 . 5 _ { \downarrow 1 3 . 3 }$ </td><td> $\overline { { 7 5 . 9 _ { \uparrow 5 1 . 1 } } }$ </td><td> $7 6 . 9 _ { \uparrow 5 2 . 1 }$ </td></tr><tr><td>BAM</td><td>25.2</td><td> $1 8 . 7 _ { \downarrow 6 . 5 }$ </td><td> $\overline { { 7 4 . 8 _ { \uparrow 4 9 . 6 } } }$ </td><td> $6 1 . 8 _ { \uparrow 3 6 . 6 }$ </td></tr><tr><td>BBOX</td><td>10.7</td><td> $1 5 . 2 _ { \uparrow 4 . 5 }$ </td><td> $6 4 . 4 _ { \uparrow 5 3 . 7 }$ </td><td> $\overline { { { \bf 8 6 . 4 } _ { \uparrow 7 5 . 7 } } }$ </td></tr><tr><td>Bridging Data Gaps</td><td>15.8</td><td> $8 . 1 _ { \downarrow 7 . 7 }$ </td><td> $\overline { { 5 8 . 1 _ { \uparrow 4 2 . 3 } } }$ </td><td> ${ \bf 6 6 . 0 _ { \mathrm { \uparrow 5 0 . 2 } } }$ </td></tr><tr><td>FRE</td><td>22.7</td><td> $1 7 . 5 _ { \downarrow 5 . 2 }$ </td><td> $\overline { { 8 1 . 4 _ { \uparrow 5 8 . 7 } } }$ </td><td> $7 6 . 8 _ { \uparrow 5 4 . 1 }$ </td></tr><tr><td>FTRL</td><td>9.3</td><td> $4 . 3 _ { \downarrow 5 . 0 }$ </td><td> $5 9 . 8 _ { \uparrow 5 0 . 5 }$ </td><td> $\overline { { 6 3 . 9 _ { \uparrow 5 4 . 6 } } }$ </td></tr><tr><td>LBCS</td><td>19.8</td><td> $2 0 . 7 _ { \uparrow 0 . 9 }$ </td><td> $\overline { { 7 4 . 7 _ { \uparrow 5 4 . 9 } } }$ </td><td> $6 5 . 8 _ { \uparrow 4 6 . 0 }$ </td></tr><tr><td>LCA on the Line</td><td>14.0</td><td> $1 3 . 3 _ { \downarrow 0 . 7 }$ </td><td> $7 4 . 9 _ { \uparrow 6 0 . 9 }$ </td><td> $6 4 . 9 _ { \uparrow 5 0 . 9 }$ </td></tr><tr><td>Mechanistic Understanding</td><td>18.1</td><td> $3 . 7 _ { \downarrow 1 4 . 4 }$ </td><td> $9 2 . 5 _ { \uparrow 7 4 . 4 }$ </td><td> $\overline { { 7 5 . 7 _ { \uparrow 5 7 . 6 } } }$ </td></tr><tr><td>PINN</td><td>24.8</td><td> $2 2 . 5 _ { \downarrow 2 . 3 }$ </td><td> $\mathbf { 9 1 . 0 } _ { \uparrow 6 6 . 2 }$ </td><td> $\overline { { 8 1 . 5 _ { \uparrow 5 6 . 7 } } }$ </td></tr><tr><td>RICE</td><td>19.8</td><td> $7 . 8 _ { \downarrow 1 2 . 0 }$ </td><td> $7 0 . 2 \substack { \mathrm { \uparrow } 5 0 . 4 }$ </td><td> $\overline { { 7 5 . 1 _ { \uparrow 5 5 . 3 } } }$ </td></tr><tr><td>Robust CLIP</td><td>29.6</td><td> $2 4 . 7 _ { \downarrow 4 . 9 }$ </td><td> $\overline { { 7 3 . 3 _ { \uparrow 4 3 . 7 } } }$ </td><td> $6 3 . 0 _ { \uparrow 3 3 . 4 }$ </td></tr><tr><td>Sample-specific Masks</td><td>30.9</td><td> $2 5 . 7 _ { \downarrow 5 . 2 }$ </td><td> $6 7 . 1 _ { \uparrow 3 6 . 2 }$ </td><td> $\overline { { 8 2 . 0 _ { \uparrow 5 1 . 1 } } }$ </td></tr><tr><td>SAPG</td><td>5.4</td><td> $6 . 3 _ { \uparrow 0 . 9 }$ </td><td> $\overline { { 7 3 . 8 _ { \uparrow 6 8 . 4 } } }$ </td><td> $\mathbf { 8 6 . 5 } _ { \mathrm { \uparrow 8 1 . 1 } }$ </td></tr><tr><td>Sequential Neural Score Estimation</td><td>41.4</td><td> $3 2 . 7 _ { \downarrow 8 . 7 }$ </td><td> $\overline { { { \bf 8 7 . 0 } _ { \uparrow 4 5 . 6 } } }$ </td><td> $7 8 . 0 _ { \cdot 1 3 6 . 6 }$ </td></tr><tr><td>Stay on Topic with Classifier-Free Guidance</td><td>10.8</td><td> $1 6 . 2 \substack { \uparrow 5 . 4 }$ </td><td> $7 0 . 5 _ { \uparrow 5 9 . 7 }$ </td><td> $\overline { { 5 8 . 5 _ { \uparrow 4 7 . 7 } } }$ </td></tr><tr><td>Stochastic Interpolants</td><td>12.4</td><td> $2 6 . 3 _ { \uparrow 1 3 . 9 }$ </td><td> ${ \bf 8 1 . 5 } _ { \uparrow 6 9 . 1 }$ </td><td> $\overline { { 8 0 . 8 _ { \uparrow 6 8 . 4 } } }$ </td></tr><tr><td>Test-Time Model Adaptation</td><td>14.3</td><td> $1 4 . 2 _ { \downarrow 0 . 1 }$ </td><td> $6 4 . 9 \substack { \cdot 5 0 . 6 }$ </td><td> $\overline { { 7 5 . 6 _ { \uparrow 6 1 . 3 } } }$ </td></tr><tr><td>What Will My Model Forget Mean score</td><td>29.7</td><td> $1 1 . 4 _ { \downarrow 1 8 . 3 }$ </td><td> $\overline { { 8 0 . 8 _ { \uparrow 5 1 . 1 } } }$ </td><td> ${ \bf 8 7 . 6 } _ { \uparrow 5 7 . 9 }$ </td></tr></table>

Table 1: Per-paper PaperBench Code-Dev comparison with Claude backbones. Claude columns for BasicAgent and IterAgent follow the PaperBench report (Starace et al., 2025); DeepCode follows its concurrent report (Li et al., 2025b). BasicAgent and IterAgent use Claude-3.5-Sonnet, while DeepCode and REPROAGENT (ours) use Claude-4.5-Sonnet. Subscripts show absolute-point differences from BasicAgent on the same paper; red upward and green downward arrows indicate the sign of the difference. best and second-best mark the row-best and row-second-best scores among the shown Claude columns.

Requirement channel: Sample-specific Masks. Sample-specific Masks scores 53.8 with the full

Holding the backbone fixed at Gemini-3-Flash, RE-PROAGENT reaches 39.7 against 19.3 for BasicAgent, 20.6 for IterAgent, and 30.5 for AiScientist. The advantage over the strongest external Gemini row is on the order of 9 absolute points, consistent with the contract effect we hypothesize in the full-suite comparison.

dependent artifacts, such as method, training, and evaluation code, consistent during generation and repair instead of being re-derived per stage.

## 5.2 Same-Backbone Comparison

## 5.3 Ablation Study

![](images/a7f29b29ae16abdca677ec9f86e712564e7b6b7402a2e045320b0f85d53d0011.jpg)  
Figure 4: Gemini channel ablations on all 20 papers. Removing reference evidence (top) or implementation requirements (bottom) lowers the mean; judge provenance and token diagnostics are in the appendix.

contract, 21.3 without implementation requirements, and 28.0 without reference evidence: the requirement drop is the dominant one. The paper’s contribution is a small set of named, implementation-critical obligations, including a per-sample mask generator, interpolation from a low-resolution mask up to the input image size, joint optimization of a shared pattern and the maskgenerator parameters, the PAD/NARROW/MEDI-UM/FULL baselines, and ablations over sharedpattern and mask channels. The full contract pins these obligations to specific work packages and files, so the generated repository keeps the baseline and ablation method registries that the rubric checks. Without the requirement channel, the planner falls back on the section outline and referencederived modules, and the generated repository keeps runnable scaffolding but drops the named baselines, ablations, and mask-channel switches, since none of them are pinned to a file. The caselevel code excerpt and a per-paper score table for additional cases appear in Appendix C.

<table><tr><td>Paper</td><td>BasicAgent</td><td>IterAgent</td><td>AiScientist</td><td>REPROAGENT (ours)</td></tr><tr><td>Adaptive Pruning</td><td>24.5</td><td> $3 . 0 _ { \downarrow 2 1 . 5 }$ </td><td> $2 7 . 2 _ { \uparrow 2 . 7 }$ </td><td> ${ \bf 4 1 . 9 _ { \uparrow 1 7 . 4 } }$ </td></tr><tr><td>All in One</td><td>20.9</td><td> $4 5 . 1 _ { \uparrow 2 4 . 2 }$ </td><td> $\overline { { 4 6 . 3 _ { \uparrow 2 5 . 4 } } }$ </td><td> ${ \bf 5 1 . 6 } _ { \uparrow 3 0 . 7 }$ </td></tr><tr><td>BAM</td><td>48.5</td><td> $4 5 . 0 _ { \downarrow 3 . 5 }$ </td><td> $\overline { { 5 6 . 6 _ { \mathrm { \uparrow 8 . 1 } } } }$ </td><td> $4 8 . 5$ </td></tr><tr><td>BBOX</td><td>15.4</td><td> $8 . 3 _ { \downarrow 7 . 1 }$ </td><td> $3 3 . 8 _ { \uparrow 1 8 . 4 }$ </td><td> $1 5 . 6 _ { \uparrow 0 . 2 }$ </td></tr><tr><td>Bridging Data Gaps</td><td>12.6</td><td> $1 2 . 4 _ { \downarrow 0 . 2 }$ </td><td> $2 3 . 1 _ { \uparrow 1 0 . 5 }$ </td><td> $\overline { { 5 0 . 6 _ { \uparrow 3 8 . 0 } } }$ </td></tr><tr><td>FRE</td><td>21.7</td><td> $2 3 . 9 _ { \uparrow 2 . 2 }$ </td><td> $\overline { { 3 5 . 2 _ { \uparrow 1 3 . 5 } } }$ </td><td> $2 6 . 9 _ { \uparrow 5 . 2 }$ </td></tr><tr><td>FTRL</td><td>5.9</td><td> $4 . 2 _ { \downarrow 1 . 7 }$ </td><td> $1 0 . 1 \substack { \uparrow 4 . 2 }$ </td><td> $\overline { { 4 0 . 1 _ { \uparrow 3 4 . 2 } } }$ </td></tr><tr><td>LBCS</td><td>17.8</td><td> $1 5 . 3 _ { \downarrow 2 . 5 }$ </td><td> $\overline { { 2 7 . 9 _ { \uparrow 1 0 . 1 } } }$ </td><td> $4 2 . 5 _ { \uparrow 2 4 . 7 }$ </td></tr><tr><td>LCA on the Line</td><td>13.0</td><td> $1 8 . 3 _ { \uparrow 5 . 3 }$ </td><td> $\overline { { 3 0 . 2 _ { \uparrow 1 7 . 2 } } }$ </td><td> ${ \bf 4 1 . 8 } _ { \uparrow 2 8 . 8 }$ </td></tr><tr><td>Mechanistic Understanding</td><td>14.9</td><td> $2 1 . 9 _ { \uparrow 7 . 0 }$ </td><td> $2 9 . 9 _ { \uparrow 1 5 . 0 }$ </td><td> ${ \bf 4 8 . 1 } _ { \uparrow 3 3 . 2 }$ </td></tr><tr><td>PINN</td><td>26.6</td><td> $3 0 . 8 \substack { \uparrow 4 . 2 }$ </td><td> $4 9 . 9 _ { \uparrow 2 3 . 3 }$ </td><td> ${ \bf 6 1 . 3 } _ { \uparrow 3 4 . 7 }$ </td></tr><tr><td>RICE</td><td>10.4</td><td> $8 . 9 _ { \downarrow 1 . 5 }$ </td><td> $\overline { { 1 0 . 9 _ { \uparrow 0 . 5 } } }$ </td><td> ${ \bf 3 1 . 0 _ { \uparrow 2 0 . 6 } }$ </td></tr><tr><td>Robust CLIP</td><td>15.4</td><td> $1 0 . 4 _ { \downarrow 5 . 0 }$ </td><td> $\overline { { 1 8 . 3 _ { \uparrow 2 . 9 } } }$ </td><td> $2 9 . 5 _ { \uparrow 1 4 . 1 }$ </td></tr><tr><td>Sample-specific Masks</td><td>25.4</td><td> $3 3 . 3 _ { \uparrow 7 . 9 }$ </td><td> $\overline { { 3 6 . 8 _ { \uparrow 1 1 . 4 } } }$ </td><td> ${ \ 5 3 . 8 } _ { \uparrow 2 8 . 4 }$ </td></tr><tr><td>SAPG</td><td>11.4</td><td> $1 2 . 7 _ { \uparrow 1 . 3 }$ </td><td> $\overline { { 1 9 . 9 _ { \uparrow 8 . 5 } } }$ </td><td> ${ \bf 2 9 . 0 _ { \mathrm { \uparrow 1 7 . 6 } } }$ </td></tr><tr><td>Sequential Neural Score Estimation</td><td>53.5</td><td> $6 0 . 2 \substack { \mathrm { \uparrow 6 . 7 } }$ </td><td> $\overline { { 6 4 . 9 _ { \uparrow 1 1 . 4 } } }$ </td><td> $4 5 . 7 _ { \downarrow 7 . 8 }$ </td></tr><tr><td>Stay on Topic with Classifier-Free Guidance</td><td>8.4</td><td> $1 3 . 7 _ { \uparrow 5 . 3 }$ </td><td> $2 0 . 1 \ L _ { \uparrow 1 1 . 7 }$ </td><td> $2 5 . 3 _ { \uparrow 1 6 . 9 }$ </td></tr><tr><td>Stochastic Interpolants</td><td>17.0</td><td> $\overline { { 1 7 . 4 _ { \uparrow 0 . 4 } } }$ </td><td> $1 8 . 8 \substack { \uparrow 1 . 8 }$ </td><td> $3 9 . 1 _ { \uparrow 2 2 . 1 }$ </td></tr><tr><td>Test-Time Model Adaptation</td><td>15.3</td><td> $1 8 . 1 \substack { \uparrow 2 . 8 }$ </td><td> $\overline { { 3 2 . 5 _ { \uparrow 1 7 . 2 } } }$ </td><td> ${ \bf 5 4 . 9 } _ { \uparrow 3 9 . 6 }$ </td></tr><tr><td>What Will My Model Forget Mean score</td><td>6.6 19.3</td><td> $9 . 0 _ { \uparrow 2 . 4 }$ </td><td> $\overline { { { \bf 1 7 . 9 } _ { \uparrow 1 1 . 3 } } }$ </td><td> $\underline { { 1 6 . 8 _ { \uparrow 1 0 . 2 } } }$ </td></tr></table>

Table 3: Gemini-3-Flash ablation results over 20 papers; $\Delta$ is the mean drop from the full contract.
<table><tr><td>Setting</td><td></td><td>Mean Median</td><td> $\Delta$ </td></tr><tr><td>Full contract</td><td>39.7</td><td>41.8</td><td></td></tr><tr><td>— Reference evidence</td><td>21.6</td><td>21.4</td><td>-18.1</td></tr><tr><td>— Implementation requirements</td><td>25.6</td><td>24.6</td><td>-14.1</td></tr></table>

Table 2: Per-paper PaperBench Code-Dev comparison with Gemini. Gemini columns for BasicAgent, IterAgent, and AiScientist follow the AiScientist engineering report (Chen et al., 2026); all Gemini columns use Gemini-3-Flash. Subscripts show absolute-point differences from BasicAgent on the same paper; red upward and green downward arrows indicate the sign of the difference. best and second-best mark the row-best and row-second-best scores among the shown Gemini columns.

Evidence channel: Bridging Data Gaps. Bridging Data Gaps scores 50.6 with the full contract, 38.4 without implementation requirements, and 13.9 without reference evidence: the evidence drop is the dominant one. The paper names the highlevel task and adaptor objectives, but its concrete protocol surface, including file layout, trainingloop wiring, checkpoint and trace artifacts, and configuration handling, is supplied by reference repositories rather than the paper text. With the full contract, the generated repository emits the expected checkpoint, training-trace, and resolvedconfiguration artifacts under conventional output paths, matching the pattern of checkpointed training and reproducible configuration logging. Without the evidence channel, paper obligations remain visible in the plan, but the generated repository under-implements file layout, data-handling conventions, experiment wiring, and artifact naming, so it fails to produce these expected artifacts. Appendix C gives the corresponding code-level trace and Appendix Table 6 lists the per-paper scores behind these aggregates.

## 5.4 Discussion

Two observations stand out. First, the samebackbone advantage and the full-suite gain move together, so the lift attaches to the contract rather than to any single backbone. Second, the two channels are complementary: removing either drops the mean by more than 14 points, and the failures fall into two classes, paper-specific obligations dropped without the requirement channel and repositorylevel scaffolding lost without the evidence channel. Together, these support a mechanistic claim: RE-PROAGENT benefits from a global contract whose requirement channel preserves explicit paper obligations and whose evidence channel grounds implicit repository knowledge.

## 6 Conclusion

We presented REPROAGENT, a contract-guided pipeline for paper-to-code reproduction. Its global contract binds explicit paper obligations and implicit reference-repository evidence to work packages, file contracts, and repair decisions. On PaperBench Code-Dev, REPROAGENT scores 73.7 across all 20 papers; same-backbone rows support scaffold-level gains, while channel ablations and case traces support the role of both contract channels. The takeaway is that paper obligations and reference-repository evidence are most useful when persisted as a contract: once both are bound, the same artifact anchors planning, exposes coverage failures, and scopes repair.

## 7 Limitations

REPROAGENT targets repository-level reproduction in which the paper and benchmark-allowed reference material jointly specify the implementation target. Extending the approach to other reproduction settings, modalities, and evaluation protocols is a natural direction for future work.

We report token, time, and call diagnostics in Appendix G to make the process auditable rather than to compare deployment cost; further engineering optimizations are compatible with the contract design and are also left to future work.

## References

Jinheon Baek, Sukhdeep Singh Jauhar, Silviu Cucerzan, and Sung Ju Hwang. 2024. ResearchAgent: Iterative research idea generation over scientific literature with large language models. arXiv preprint arXiv:2404.07738.

Monya Baker. 2016. 1,500 scientists lift the lid on reproducibility. Nature, 533(7604):452–454.

Tim Baumgärtner and Iryna Gurevych. 2026. SciCoQA: Quality assurance for scientific paper–code alignment. arXiv preprint arXiv:2601.12910.

Anya Belz, Shubham Agarwal, Anastasia Shimorina, and Ehud Reiter. 2021. A systematic review of reproducibility research in natural language processing. In Proceedings ofthe 16th Conference ofthe European Chapter ofthe Associationfor Computational Linguistics (EACL), pages 381–393.

Jun Shern Chan, Neil Chowdhury, Oliver Jaffe, James Aung, Dane Sherburn, Evan Mays, Giulio Starace, Kevin Liu, Leon Maksin, Tejal Patwardhan, Lilian Weng, and Aleksander M ˛adry. 2024. MLEbench: Evaluating machine learning agents on machine learning engineering. arXiv preprint arXiv:2410.07095.

Guoxin Chen, Jie Chen, Lei Chen, Jiale Zhao, Fanzhe Meng, Wayne Xin Zhao, Ruihua Song, Cheng Chen, Ji-Rong Wen, and Kai Jia. 2026. Toward autonomous long-horizon engineering for ML research. arXiv preprint arXiv:2604.13018.

Pengcheng Chen, Ning Yan, Zihan Zhao, Yixiao Lin, Huaibo Chen, Yue Hu, Qinbo Bai, Xiang Li, and Masood S. Mortazavi. 2025. Deep-Reproducer: From paper understanding to code generation. In NeurIPS 2025 Fourth Workshop on Deep Learningfor Code.

Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing.

Wei Cheng, Yuhan Wu, and Wei Hu. 2024. Dataflow-Guided Retrieval Augmentation for Repository-Level Code Completion. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7957– 7977, Bangkok, Thailand. Association for Computational Linguistics.

Jane Cleland-Huang, Orlena C. Z. Gotel, Jane Huffman Hayes, Patrick Mäder, and Andrea Zisman. 2014. Software traceability: Trends and future directions. In Proceedings ofthe Future ofSoftware Engineering (FOSE), pages 55–69.

Orlena C. Z. Gotel and Anthony C. W. Finkelstein. 1994. An analysis of the requirements traceability problem. In Proceedings ofthe IEEE International Conference on Requirements Engineering (ICRE), pages 94–101.

Hanhua Hong, Yizhi Li, Jiaoyan Chen, Sophia Ananiadou, Xiaoli Li, Jung-jae Kim, and Chenghua Lin. 2026. HiRAS: A hierarchical multi-agent framework for paper-to-code generation and execution. arXiv preprint arXiv:2604.17745.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In The Twelfth International Conference on Learning Representations.

Md. Ashraful Islam, Mohammed Eunus Ali, and Md Rizwan Parvez. 2024. MapCoder: Multi-agent code generation mirroring human cognitive processes. In Proceedings ofthe Annual Meeting ofthe Association for Computational Linguistics (ACL).

Jia Li, Xianjie Shi, Kechi Zhang, Ge Li, Zhi Jin, Lei Li, Huangzhao Zhang, Fang Liu, Yuwei Zhang, Zhengwei Tao, Yihong Dong, Yuqi Zhu, and Chongyang Tao. 2025a. GraphCodeAgent: Dual graph-guided LLM agent for retrieval-augmented repo-level code generation. arXiv preprint arXiv:2504.10046.

Zongwei Li, Zhonghang Li, Zirui Guo, Xubin Ren, and Chao Huang. 2025b. DeepCode: Open agentic coding. arXiv preprint arXiv:2512.07921.

Zijie Lin, Yiqing Shen, Qilin Cai, He Sun, Jinrui Zhou, and Mingjun Xiao. 2025. AutoP2C: An LLMbased agent framework for code repository generation from multimodal content in academic papers. arXiv preprint arXiv:2504.20115.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. 2024. The AI scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

Bertrand Meyer. 1992. Applying “design by contract”. Computer, 25(10):40–51.

Sameer Pimparkhede, Sameer Kammakomati, Srikanth Tamilselvam, Prince Kumar, Ashok Bhatt, and Pushpak Kumar. 2024. DocCGen: Document-based controlled code generation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing.

Joelle Pineau, Philippe Vincent-Lamarre, Koustuv Sinha, Vincent Larivière, Alina Beygelzimer, Florence d’Alché Buc, Emily Fox, and Hugo Larochelle. 2021. Improving reproducibility in machine learning research (a report from the NeurIPS 2019 reproducibility program). Journal ofMachine Learning Research, 22(164):1–20.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ChatDev: Communicative agents for software development. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15174–15186, Bangkok, Thailand. Association for Computational Linguistics.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. 2025. Agent laboratory: Using LLM agents as research assistants. arXiv preprint arXiv:2501.04227.

Minju Seo, Jinheon Baek, Seongyun Lee, and Sung Ju Hwang. 2026. Paper2Code: Automating code generation from scientific papers in machine learning. In The Fourteenth International Conference on Learning Representations.

Akshay Sethi, Anush Sankaran, Naveen Panwar, Shreya Khare, and Senthil Mani. 2018. DLPaper2Code: Auto-generation of code from deep learning research papers. In Proceedings of the Thirty-Second AAAI Conference on Artificial Intelligence and Thirtieth Innovative Applications ofArtificial Intelligence Conference and Eighth AAAI Symposium on Educational Advances in Artificial Intelligence, page 899. AAAI Press.

Zachary S. Siegel, Sayash Kapoor, Nitya Nagdir, Benedikt Stroebl, and Arvind Narayanan. 2024. CORE-Bench: Fostering the credibility of published research through a computational reproducibility agent benchmark. Transactions on Machine Learning Research (TMLR).

Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. 2025. PaperBench: Evaluating AI’s ability to replicate AI research. In Proceedings ofthe International Conference on Machine Learning (ICML).

Xiangru Tang, Yuliang Liu, Zefan Cai, Yanjun Shao, Junjie Lu, Yichi Zhang, Zexuan Deng, Helan Hu, Kaikai An, Ruijun Huang, Shuzheng Si, Sheng Chen, Haozhe Zhao, Liang Chen, Yan Wang, Tianyu Liu, Zhiwei Jiang, Baobao Chang, Yin Fang, and 5 others. 2024. ML-Bench: Evaluating large language models and agents for machine learning tasks on repositorylevel code. Preprint, arXiv:2311.09835.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. 2025. Demystifying LLM-based software engineering agents. Proceedings of the ACM on Software Engineering, 2(FSE).

Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, and Yulan He. 2025. SciReplicate-Bench: Benchmarking LLMs in agent-driven algorithmic reproduction from research papers. In Proceedings of the Conference on Language Modeling (COLM). ArXiv:2504.00255.

Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, Juntong Ni, George Michalopoulos, Yue Zhang, Ziyang Zhang, Mian Zhang, Zhiyu Chen, and Xinya Du. 2025. LMR-BENCH: Evaluating LLM agent’s ability on reproducing language modeling research. In Proceedings of the 2025 Conference on Empirical

Methods in Natural Language Processing, pages 6164–6186, Suzhou, China. Association for Computational Linguistics.

John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-computer interfaces enable automated software engineering. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems. Curran Associates Inc.

Fengji Zhang, Bei Chen, Yue Zhang, Jacky Keung, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Chen, and Wenqiang Chen. 2023. RepoCoder: Repository-level code completion through iterative retrieval and generation. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP).

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. 2024. AutoCodeRover: Autonomous program improvement. In Proceedings of the International Symposium on Software Testing and Analysis (ISSTA).

Xuanle Zhao, Zilin Sang, Yuxuan Li, Qi Shi, Weilun Zhao, Shuo Wang, Duzhen Zhang, Xu Han, Zhiyuan Liu, and Maosong Sun. 2025. AutoReproduce: Automatic AI experiment reproduction with paper lineage. arXiv preprint arXiv:2505.20662.

Mingyang Zhou, Quanming Yao, Lun Du, Lanning Wei, and Da Zheng. 2025. Reflective paper-to-code reproduction enabled by fine-grained verification. arXiv preprint arXiv:2508.16671.

Shuyan Zhou, Uri Alon, Frank F. Xu, Zhengbao Wang, Zhiruo Jiang, and Graham Neubig. 2023. DocPrompting: Generating code by retrieving the docs. In The Eleventh International Conference on Learning Representations.

## A Contract Case Study: Unit and Evidence Records

This appendix gives two paired schema records for a continual-RL paper that introduces a forgettingaware fine-tuning protocol. The first illustrates the implementation-requirement schema in §3.1; the second illustrates the reference-evidence schema in §3.2 and is anchored to the unit from the first. The records are compact schema views that preserve source anchors and evidence bindings used by review and repair.

## A.1 An Implementation-Requirement Unit

The unit below illustrates how Prepare extracts a requirement, Plan assigns it to a method work package, Generate projects it into a model-method file contract, and Repair verifies it with requirement review.

```jsonl
Schema-aligned units.json view for the
EWC running example
{
"unit_id": "unit_004",
"type": "method",
"statement": "implement Elastic Weight
Consolidation (EWC)",
"paper_evidence": [
"L_EWC = L_RL + (lambda/2) * sum_i F_i (
theta_i - theta_*_i)^2",
"where theta_* is the pre-trained model
and F is the diagonal of the Fisher
matrix."
],
"source_paragraph_ids": ["chunk_004_02", "
chunk_024"],
"verification_targets": [
{
"kind": "artifact",
"description": "Fisher matrix
calculation and EWC loss implementation"
}
],
"implementation_surfaces": ["
model_or_method"],
"code_obligations": [
"Implement diagonal Fisher Information
Matrix from the pre-trained policy on its
training distribution.",
"Implement the EWC loss term (lambda/2) *
sum_i F_i (theta_i - theta_*_i)^2.",
"Use lambda=10000 as the default
coefficient."
],
"status": "active"
}
```

What each field stores.

• unit\_id is a stable identifier; repair tickets and provenance records both index by this value, so the same unit remains addressable across stages.

• type categorizes the unit (here: method) and drives how Plan groups it with related units into a work package.

• statement is the normalized one-duty implementation requirement, derived from the paper rather than copied verbatim.

• paper\_evidence keeps the originating paper quotes, so review can cross-check the statement against the source.

• source\_paragraph\_ids are anchors into the segmented paper (chunk IDs), used to locate the unit in the addendum or main body and to guide review and repair.

• verification\_targets declares what reviewers should check; each target has a kind (artifact, surface, or metric) and a description.

• implementation\_surfaces declares where the obligation should appear in code; file planning must project the unit onto at least one file whose declared surfaces cover this set.

• code\_obligations expand the statement into concrete sub-duties that generation and review can verify pointwise.

• status is the lifecycle flag; review writes verdicts (pass, missing, wrong-surface, inconsistent-withsource) back into this field.

How the preservation invariant fires on this unit. Plan cannot proceed unless the EWC unit is bound to at least one owner work package; in this run it is owned by a method package together with the related PPO-fine-tuning unit. File planning cannot proceed unless the unit is projected into at least one file contract whose declared surface includes model\_or\_method; in this run the unit lands in a single dedicated method file. Generate cannot finalize that file unless the file emits a provenance record citing unit\_004, so the unit id surfaces in the per-file provenance map alongside its source paragraphs. Repair cannot mark the unit pass until requirement review verifies all three code obligations against the file contract; otherwise it issues a repair ticket carrying the unit id, the matched evidence ids attached to the work package, and the protected units that already passed.

The unit is therefore not a free-form paragraph or a planning-time prompt fragment. It is an addressable record with a fixed schema that planning, file projection, generation, validation, and repair are all conditioned on, which is what allows the same paper requirement to remain visible across the four-stage pipeline rather than being silently dropped.

## A.2 A Reference-Evidence Item

The evidence item below illustrates how a selected reference snippet is scoped to the method work package that owns the EWC unit unit\_004.

Schema-aligned evidence\_bundles.json   
view tied to unit\_004   
{   
"work\_package\_id": "wp\_methods",   
"unit\_id": "unit\_004",   
"ref\_id": "ref\_001",   
"file\_path": "model.py",   
"snippet\_preview": "model.py:13: def   
\_\_init\_\_(self, in\_features, out\_features,   
sigma0=0.5):\nmodel.py:14: super   
().\_\_init\_\_()\nmodel.py:15: self.   
in\_features = in\_features\nmodel.py:16:   
self.out\_features = out\_features\   
nmodel.py:17: self.weight = nn.   
Parameter(torch.Tensor(out\_features,   
in\_features))\nmodel.py:18: self.   
bias = nn.Parameter(torch.Tensor(   
out\_features))\nmodel.py:19:   
11   
"why\_relevant": "function \`\_\_init\_\_\` in \`   
model.py\` aligns with units   
paper\_method\_core, unit\_004 via parameter   
construction patterns reusable for the   
EWC reference network.",   
"confidence": 1.0,   
"matched\_keywords": ["nn.Parameter", "   
in\_features", "out\_features"]   
}

What each field stores.

• work\_package\_id names the package this evidence is attached to; in §3.2 the same identifier serves as the binding point between requirements and evidence.

• unit\_id names the supported requirement; here it is the EWC unit unit\_004 from Appendix A.1, which is what makes the evidence consumable by the same package contract.

• ref\_id, file\_path, and snippet\_preview together pin down (repo, path, region): the source repository, the file inside it, and the matched line range, recorded so review and repair can locate the snippet again.

• why\_relevant is the structured match basis: it explains, in one sentence, which symbol or interface aligns the snippet with the supported unit, rather than leaving the LLM judgment as a freeform note.

• confidence and matched\_keywords record the auxiliary signals used by the retain rule in §3.2, namely a symbol-level match strengthened by an LLM judgment.

• The role field, omitted from this stripped record but tracked elsewhere in the run state, marks this snippet as reusable structural evidence for parameter construction.

How the evidence enters generation. The item is bound to wp\_methods during Plan, so when Generate writes the file contract that owns unit\_004, the prompt for that file sees this snippet but no unrelated reference code from sibling work packages. If requirement review later returns a non-pass verdict on unit\_004, the repair ticket carries the same ref\_id and file\_path, allowing repair to re-anchor the edit on this specific reference snippet rather than searching the reference repository again.

## B External Baselines and Cost Notes

Table 4 records aggregate external rows that are useful context but not part of the primary samebackbone comparison. Rows include official Paper-Bench results, recent paper-to-code systems, and separately collected external reference rows. They differ in scaffold, model, runtime budget, and reporting protocol, and provide contextual reported scores rather than a primary controlled comparison.

## C Gemini Ablation Case Studies

We keep qualitative ablation cases in the appendix so that the main results focus on aggregate behavior. The examples below illustrate the channel roles behind the aggregate ablation pattern in Section 5.3. Scores in this section provide mechanism-oriented evidence for the aggregate analysis.

## C.1 Tracing One Unit in Test-Time Model Adaptation

We trace Test-Time Model Adaptation because its FOA/CMA unit is a strong trace for the implementation-requirement channel and a useful contrast for how reference evidence affects repository organization. In our consolidated 2026-05- 24 PaperBench judging snapshot, the full Gemini run scores 54.9, while removing implementation requirements scores 47.1 and removing reference evidence scores 23.8. The trace is taken from the corresponding Gemini full-run artifacts for test-time-model-adaptation.

<table><tr><td>Method</td><td>Backbone / setting</td><td>Reported score</td></tr><tr><td rowspan="5">BasicAgent (Starace et al., 2025)ª</td><td>Gemini-2.0-Flash</td><td>5.0</td></tr><tr><td>o3-mini</td><td>5.1</td></tr><tr><td>GPT-40</td><td>7.7</td></tr><tr><td>o1</td><td>19.5</td></tr><tr><td>Claude-3.5-Sonnet</td><td>35.4</td></tr><tr><td rowspan="4">IterativeAgent (Starace et al., 2025)ª</td><td>o3-mini</td><td>16.4</td></tr><tr><td>Claude-3.5-Sonnet</td><td>27.5</td></tr><tr><td>01</td><td>43.3</td></tr><tr><td></td><td></td></tr><tr><td>PaperCoder (Seo et al., 2026)</td><td>o3-mini-high</td><td>45.1</td></tr><tr><td>AutoP2C (Lin et al., 2025)</td><td>source-reported setting</td><td>49.2</td></tr><tr><td>AutoReproduce (Zhao et al., 2025)</td><td>source-reported setting</td><td>49.6 63.2</td></tr><tr><td>Deep-Reproducer (Chen et al., 2025) RePro (Zhou et al., 2025)b</td><td>GPT-5 o3-mini</td><td>62.6</td></tr><tr><td>DeepCode (Li et al., 2025b)c</td><td>Claude-Sonnet-4.5</td><td>73.5</td></tr><tr><td></td><td></td><td></td></tr><tr><td>BasicAgent (Chen et al., 2026)d</td><td>Gemini-3-Flash Gemini-3-Flash</td><td>19.3 20.6</td></tr><tr><td>IterAgent (Chen et al., 2026)d AiScientist (Chen et al., 2026)d</td><td>Gemini-3-Flash</td><td>30.5</td></tr><tr><td></td><td></td><td></td></tr><tr><td>REPROAGENT (ours)</td><td>Claude-Sonnet-4.5 Gemini-3-Flash</td><td>73.7 39.7</td></tr></table>

Table 4: Supplemental aggregate PaperBench baselines. Reported scores are percentages. Rows differ in scaffold, backbone, runtime budget, and reporting protocol, and provide context for the full-suite comparison rather than a primary controlled comparison.  
<sup>a</sup> Official 20-paper PaperBench Code-Dev scaffold rows; standard errors are reported in the cited source, and this table reports means only for compact contextual comparison. The o1 IterativeAgent row uses two seeds. <sup>b</sup> RePro reports PRroot@5, so we list it as external PRroot context separate from the Code-Dev rows. <sup>c</sup> Collected 20-paper mean over three runs per paper; the reported mean standard error across papers is 2.8 and the mean cost is \$8.88. <sup>d</sup> Collected AiScientist engineering aggregates; mean costs are \$6.25, \$27.44, and \$15.67 for BasicAgent, IterAgent, and AiScientist.

Extraction. unit\_002 is extracted as “Implement Forward-Only Prompt Adaptation with CMA”. Its obligations require prompt insertion into the ViT input sequence, a CMA-ES optimizer, a population size K ∈ [2, 28], and a forward-only prompt update loop. unit\_003 is extracted as the activation-distribution discrepancy fitness function, requiring mean/variance statistics for test CLS tokens and a fitness value passed to CMA.

Work package and contract. Planning binds both units to wp\_foa\_core, whose goal is to implement the FOA method and CMA optimizer. The work package produces methods/foa.py, optimizers/cma\_es.py, and results/source\_stats.json. Its global contract preserves the FOA loop, CMA-ES interface, source-statistics collector, fitness function, method registry, and parameter inventory for K, prompt length, alignment weights, batch size 64, and momentum 0.9. This is the implementationrequirement channel in concrete form: the paper obligations become file-owned duties rather than prose context.

Evidence binding. The evidence bundle for wp\_foa\_core records paper chunks for FOA, source-statistics calculation, and the evaluation protocols. In this case, the package is marked self\_contained and has no repository evidence\_links, because the decisive FOA/CMA obligations are already present in the paper and addendum evidence. This clarifies the ablation: the no-reference setting still recovers much of the requirement surface, while the full setting adds stronger repository-level organization, protocol wiring, and comparison surfaces.

Generated files and provenance. Generation materializes the contract in three files. methods/foa.py contains the hyperparameter sweeps, method/baseline registries, a CMA-ES implementation, source-statistics loading, and FOA routines. optimizers/cma\_es.py exposes a standalone CMA-ES ask/tell optimizer and method selectors. src/foa/utils/cma\_optimizer.py contains a forward-only FOA.adapt route under torch.no\_grad(). The generation fileprovenance record assigns all three files to wp\_foa\_core, lists unit\_002 and unit\_003 as owned units, and attaches validation checks for implementation requirements, formula/algorithm contracts, evidence-matrix coverage, and route closure. The key point is that the requirement IDs, work-package owner, generated files, and validation checks remain connected throughout the trajectory; the ablation removes one of these channels before the file-level contract is formed.

## C.2 Tracing Reference Evidence in Bridging Data Gaps

Bridging Data Gaps is a complementary case because the same snapshot shows an asymmetric ablation: the full Gemini run scores 50.6, the run without implementation requirements scores 38.4, and the run without reference evidence scores 13.9. The case therefore stresses the second channel in Figure 2: the paper requirement channel keeps the high-level adaptation and artifact-writing obligations visible, while reference evidence helps convert those obligations into a concrete repository protocol.

Requirement channel. The requirement units preserve the paper-derived duties that must survive into code: train the adaptation components, expose a runnable experiment path, save model/adaptor checkpoints, and write result artifacts that make the experiment inspectable. These duties are projected into file contracts that expect a training route and artifact writer rather than a single monolithic script. This explains the asymmetric score pattern: the remaining reference and generic task context can still support some repository structure, but the paper-specific obligations are less explicit without the requirement channel.

Reference-evidence channel. The reference channel supplies the implementation texture that the paper does not spell out line by line. In the full run, the generated repository writes checkpoints/adaptor.pth, checkpoints/trained\_model.pth,

results/ant\_training\_trace.json, and results/config\_resolved.json, matching the expected pattern of checkpointed training and reproducible configuration logging. When reference evidence is removed, the paper obligations remain visible, but the run has less support for file layout, data-handling conventions, experiment wiring, and artifact naming. This case provides a concrete trace of why the reference-evidence channel can matter even when the target paper already names the high-level task.

## C.3 Additional Ablation Patterns

The ablation pattern is not driven by a single paper. The excerpts below are illustrative traces from the full Gemini repositories, not additional scoring criteria. They show how the contract is materialized as file-owned implementation choices rather than remaining only in planning prose.

```python
Sample-specific Masks: method and ablation
factory excerpt
BASELINES = ["PAD", "NARROW", "MEDIUM", "FULL
"]
ABLATIONS = ["ONLY_delta", "ONLY_f_mask",
"SINGLE_CHANNEL_f_mask_s", "OURS
"]
def get_method_config(method_name: str) ->
MethodConfig:
if method_name == "ONLY_delta":
return MethodConfig(name="ONLY_delta",
use_delta=True,
use_f_mask=False)
elif method_name == "ONLY_f_mask":
return MethodConfig(name="ONLY_f_mask
", use_delta=False,
use_f_mask=True)
elif method_name in ["ours", "OURS", "SMM
"]:
return MethodConfig(name="OURS",
use_delta=True,
use_f_mask=True,
mask_channels=3)
elif method_name in ["PAD", "NARROW", 11
MEDIUM", "FULL"]:
return MethodConfig(name=method_name,
use_delta=True,
use_f_mask=False)
```

Bridging Data Gaps: training artifact excerpt

```python
for step in range(num_iterations):
loss_val = train_ant_step(batch, config,
model, adaptor,
classifier,
optimizer)
trace.append({"step": step, "loss":
loss_val})
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)
torch.save(adaptor.state_dict(), "checkpoints/
adaptor.pth")
torch.save(model.state_dict(), "checkpoints/
trained_model.pth")
with open("results/ant_training_trace.json",
"w") as f:
json.dump(trace, f, indent=2)
with open("results/config_resolved.json", "w")
as f:
json.dump(config, f, indent=2)
```

<table><tr><td>Paper</td><td></td><td></td><td></td><td>Full w/o impl. w/o ref. Case role</td></tr><tr><td>Sample-specific Masks</td><td>53.8</td><td>21.3</td><td>28.0</td><td>Clear two-channel benefit: the full run preserves SMM- specific optimization, mask generation, interpolation, base- lines, and ablations.</td></tr><tr><td>BAM</td><td>48.5</td><td>30.8</td><td></td><td>20.7 Large positive case: both removed-channel settings score substantially below the full contract, with the largest gap for reference evidence.</td></tr><tr><td>Bridging Data Gaps</td><td>50.6</td><td>38.4</td><td>13.9</td><td>Reference-heavy case: paper obligations remain useful, but removing reference evidence sharply reduces concrete protocol and repository-structure recovery.</td></tr><tr><td>PINN</td><td>61.3</td><td>53.3</td><td>52.2</td><td>Standardized-prior case: all variants remain strong because PINN objectives and PDE-loss structure are common, but</td></tr><tr><td>What Will My Model Forget 16.8</td><td></td><td>10.7</td><td>8.4</td><td>the full contract is still best. Scale and coverage stress case: the task has many long-tail details, and the full contract still leads both ablations.</td></tr><tr><td>LCA-on-the-Line</td><td>41.8</td><td>16.1</td><td></td><td>33.8 Boundary case: no-reference keeps a visible checklist for ObjectNet/LCA details, while the full run remains the strongest setting.</td></tr></table>

Table 5: Representative Gemini ablation cases. The case role column summarizes inspection observations alongside the score columns. “w/o impl.” removes the implementation-requirement channel; “w/o ref.” removes the referenceevidence channel.

Large two-channel gains. Sample-specific Masks and BAM are representative qualitative traces covering large two-channel gains. In Sample-specific Masks, the full run keeps the paper’s required surfaces connected: the mask generator, interpolation from a low-resolution mask to the input image size, joint optimization of the shared pattern and mask-generator parameters, PAD/NARROW/MEDIUM/FULL baselines, and ablations over shared pattern and mask channels. Removing implementation requirements disrupts this preservation path, so the repository retains runnable scaffolding but covers fewer paper-specific obligations. Removing reference evidence keeps the obligations but reduces the file and experiment organization needed to expose baselines and ablation routes in a concrete implementation. BAM shows a similar pattern at a coarser level: the full score is 48.5, while no-implementation and no-reference fall to 30.8 and 20.7, respectively. This indicates that the reference-evidence channel recovers practical implementation and evaluation structure, while the implementation-requirement channel keeps paper-specific modeling and protocol obligations explicit.

Reference-sensitive cases. Bridging Data Gaps is useful because the two ablations degrade asymmetrically. The full run reaches 50.6, the noimplementation run remains at 38.4, and the noreference run drops to 13.9. The artifact trace shows that the implementation-requirement channel preserves high-level tasks and expected outputs, while reference evidence supplies much of the concrete protocol surface: file layout, scripts, data handling conventions, and experiment wiring. This is the kind of paper where related repositories supply implementation texture that is not explicit enough in the paper text alone.

Standardized tasks. PINN is a high-scoring but smaller-margin case. The full run obtains 61.3, while the two ablations remain at 53.3 and 52.2. This pattern is consistent with the channel hypothesis. PINN-style repositories have strong public priors: collocation points, PDE residual losses, boundary losses, optimizer loops, and diagnostic plots are common implementation patterns. As a result, Gemini can recover a reasonable repository even with one channel removed. The full contract still gives the best score because it binds these common priors to the paper-specific objective, PDE setup, loss-landscape diagnostics, and expected artifacts.

Scale and coverage stress. What Will My Model Forget is a scale-sensitive case for the Gemini backbone. The full setting scores 16.8 and both ablations are lower. The target has broad long-tail coverage requirements and many benchmark-specific details, so this case separates channel utility from backbone capacity while preserving the same ordering pattern.

## C.4 Why Ablation Gaps Vary

The per-paper ablation gaps are heterogeneous, and this variation is itself informative. For this discussion, we define the reference-evidence gain as the difference between the full Gemini score and the score without reference evidence, and the implementation-requirement gain as the difference between the full Gemini score and the score without implementation requirements. The largest reference-evidence gains occur on Bridging Data Gaps (+36.7), Adaptive Pruning (+35.7), Test-Time Adaptation (+31.1), FTRL (+28.4), Allin-One (+27.9), and BAM (+27.8). The largest implementation-requirement gains occur on Mechanistic Understanding (+43.5), Sample-Specific Masks (+32.5), Adaptive Pruning (+27.6), FTRL (+26.7), LCA-on-the-Line (+25.7), and All-in-One (+23.6). These rankings suggest that the two channels help for related but distinguishable reasons.

![](images/dde055b52345aa54f31bdb40b63a86ea3aa3302dae318113aaa4ca9e1c0d9d6d.jpg)  
Figure 5: Code-level evidence for two representative ablation cases. The full generated repositories expose different concrete surfaces: Sample-specific Masks relies on implementation-requirement preservation for method-specific masks, interpolation, baselines, and ablations, while Bridging Data Gaps relies more heavily on reference evidence for protocol and training-artifact wiring.

Reference transferability. Reference evidence helps most when external artifacts provide transferable implementation structure: dataset protocols, baseline organization, metric wiring, experiment scripts, configuration conventions, or repository layout. This is broader than simply counting cloned repositories. For example, BBOX has multiple reference or dataset repositories in the preparation artifacts, yet its reference-evidence gain is small (+1.0), showing that repository count alone is not the key factor. Conversely, Bridging Data Gaps shows a large reference-evidence gain even with a compact prepared reference-repository list, because the useful signal is whether reference evidence can be fused into concrete protocol and repositorystructure decisions. In reference-sensitive cases, removing this channel leaves the paper obligations visible but makes the generated repository less able to recover executable data handling, experiment wiring, and comparison surfaces.

Requirement hardness. The implementationrequirement channel helps most when a paper’s contribution is compact but implementationcritical: a small set of named mechanisms, constraints, losses, ablations, metrics, or artifacts determines whether the repository is faithful. Sample-Specific Masks is a representative case: the decisive obligations include mask generation, interpolation from a low-resolution mask to image size, joint optimization of shared patterns and maskgenerator parameters, PAD/NARROW/MEDIUM/- FULL baselines, and ablations over shared-pattern and mask channels. Mechanistic Understanding, Adaptive Pruning, FTRL, and LCA-on-the-Line exhibit the same pattern at different scales: without explicit implementation requirements, the model can still produce plausible scaffolding, but it tends to omit or dilute the paper-specific mechanisms and protocol obligations that decide the score.

Small-gap cases. Small ablation gaps do not necessarily imply that the removed channels are irrelevant. They usually fall into one of three categories. First, some papers have strong public implementation priors; PINN is the clearest example, where collocation points, PDE residual losses, boundary losses, optimizer loops, and diagnostic plots are common enough that both ablations remain strong. Second, some papers have compressed gaps because all variants face the same central implementation bottleneck; BBOX is a case where the three Gemini variants remain close. Third, some tasks are scale or coverage stress cases, such as What Will My Model Forget, where the full contract still leads but the target contains many long-tail requirements.

Overall, the ablation gaps are best interpreted as indicators of where each channel provides leverage. Reference evidence is most valuable when external artifacts are transferable into implementation structure, while implementation requirements are most valuable when fidelity depends on non-obvious paper-specific obligations. Small gaps are therefore ambiguous: they may reflect strong public priors, limited reference transferability, or shared bottlenecks across variants, rather than evidence that the channels are unnecessary.

## D Prompting and Structured Outputs

The production prompts use a shared systemuser pattern. The current pipeline uses schemaconstrained JSON for planning stages, deterministic evidence grounding for package-level evidence bundles, and repository-edit prompts for generation and repair. We therefore report compact schema excerpts rather than the full prompt text. The released repository contains the full prompt templates and structured-output schemas.

## Prepare: ExtractedUnit schema excerpt

```python
"unit_id": "unit_002",
"type": "method | protocol | claim |
artifact | task",
"statement": "paper-specific implementation
obligation",
"hypothesis": "what claim or mechanism this
unit tests",
"decision_value": "what implementation
decision depends on it",
"paper_evidence": ["source paper/addendum
excerpt"],
"source_paragraph_ids": ["chunk_006_01"],
"reference_query_terms": ["optional
explicit method or repository cue"],
"implementation_surfaces": ["
model_or_method", "metric_formula"],
```

```python
"code_obligations": ["concrete executable
obligation"],
"runtime_interfaces": ["CLI/function/config
surface"],
"expected_artifacts": ["results/metrics.
json"],
"suggested_module_kinds": ["training_loop",
"artifact_writer"],
"implementation_notes": ["inventory or
scope notes"],
"tags": ["method", "evaluation"]
```

## Plan: contract and file-planning schema excerpt

"work\_package": {   
"work\_package\_id": "wp\_foa\_core",   
"owned\_unit\_ids": ["unit\_002", "unit\_003   
"],   
"inventories": {   
"obligation\_matrix": ["Method -> file   
path"],   
"method\_inventory": ["named method or   
baseline"],   
"parameter\_inventory": ["paper-visible   
values"]   
},   
"scope\_boundary": {   
"preserve": ["unit-level obligations   
that must survive"],   
"implementation\_focus": ["active routes   
and artifacts"]   
},   
"method\_obligations": ["what code must   
expose"]   
},   
"file\_plan": {   
"target\_file": "methods/foa.py",   
"work\_package\_id": "wp\_foa\_core",   
"owned\_units": ["unit\_002", "unit\_003"],   
"reference\_ids": ["ref or paper chunk ids   
"],   
"implementation\_surfaces": ["   
model\_or\_method"],   
"defines\_symbols": ["FOA", "CMAES"],   
"calls\_symbols": ["fitness\_function"],   
"writes\_artifacts": ["results/   
adaptation\_trace.json"],   
"scope\_boundary": {"preserve": ["file  
owned obligations"]},   
"generation\_prompt": "file-local   
implementation instruction",   
"review\_points": ["semantic and route  
closure checks"]   
}   
}

## Evidence grounding and repair schema excerpt

```jsonl
"evidence_bundle": {
"work_package_id": "wp_foa_core",
"focus": "implementation duty",
"owned_unit_ids": ["unit_002", "unit_003
"],
"evidence_links": [
{"unit_id": "unit_002", "ref_id": "
ref_001", "file_path": "repo/file.py",
"snippet_preview": "...", n
why_relevant": "...", "confidence": 0.0}
],
"context_summary": ["paper or reference
evidence summary"],
"grounding_status": "grounded |
self_contained | weak | ungrounded"
},
"repair_ticket": {
"failure_type": "semantic | runtime |
integration",
"required_fix_targets": ["methods/foa.py
"],
"allowed_changes": ["targeted repair
scope"],
"forbidden_changes": ["requirements
already preserved"],
"next_fix_scope": ["files to patch"]
}
```

## E Additional Algorithmic Details

## E.1 Algorithmic Summary

Algorithm 1 REPROAGENT pipeline   
Require: Paper P, request r, references R,   
rounds T   
Ensure: Repository C<sup>ˆ</sup>   
1: S ← ChunkPaper(P)   
2: Q ← SurveyRepos(R)   
3: U ← ExtractRequirements(S)   
4: U ← Deduplicate(U)   
5: W ← PlanWorkPackages(U)   
6: CheckPreserved(U, W)   
7: E ← CollectContentEvidence(W, Q)   
8: Z ← CollectStructureEvidence(Q)   
9: G ← BuildContract(U, W, E, Z)   
10: F ← PlanFiles(G)   
11: for f ∈ TopologicalOrder(F) do   
12: <sup>ˆ</sup>f ← GenerateFile(f, G, E)   
13: end for   
14: for t = 1 to T do   
15: v<sub>q</sub> ← RequirementReview(C<sup>ˆ</sup>, U)   
16: v ← RuntimeValidation(C<sup>ˆ</sup>)   
17: if v<sub>q</sub>.passed ∧ v<sub>r</sub>.passed then   
18: break   
19: end if   
20: τ ← RepairTicket(v<sub>q</sub>, v<sub>r</sub>, E)   
21: C ←<sup>ˆ</sup> ApplyRepair(C<sup>ˆ</sup>, τ )   
22: end for   
23: return C<sup>ˆ</sup>

## E.2 Repair Loop

Algorithm 2 Contract-preserving repair loop   
Require: Repository C<sup>ˆ</sup>, requirements U, contracts   
G, evidence E, budget T   
Ensure: Patched repository C<sup>ˆ</sup>   
1: for t = 1 to T do   
2: s ← ReviewRequirements(C<sup>ˆ</sup>, U, G)   
3: r ← RunValidation(C<sup>ˆ</sup>)   
4: if s.passed ∧ r.passed then   
5: break   
6: end if   
7: τ ← BuildRepairTicket(s, r, G, E)   
8: τ<sub>ban</sub> ← PreservedReqs(s)   
9: C ←<sup>ˆ</sup> PatchAfectedFiles(C<sup>ˆ</sup>, τ)   
10: end for   
11: return C<sup>ˆ</sup>

## F Per-Paper Scores and Generated Repository Sizes

This section gives the complete per-paper scores and generated repository sizes behind the full and ablated runs reported in the main results.

## G Token, Time, and Cost Diagnostics

Tables 7 and 8 report usage metadata for interpreting the reproduction process. Scoring-token counts come from PaperBench judge summaries; when a judge summary stores token usage at leaf level, we sum the recoverable per-leaf metadata from the same graded task tree. Pipeline-token counts, wall-clock time, and agent-call counts use exported run logger summaries when available. Runtime metadata is exported at different granularity across settings; the Claude pipeline-token cell is estimated from matched run logs and is reported only as process metadata, while hours and calls use the recoverable run metadata directly. For the Gemini family, the full run uses more pipeline tokens than either ablation, which is expected because the full setting carries both implementation requirements and reference evidence through planning and repair. Wall-clock time varies with target-paper difficulty and available runtime metadata.

## H Full Pipeline Stage Sequence

REPROAGENT comprises 15 sub-stages organized into four phases. Table 9 lists every stage with its phase assignment and a brief description.

The Repair phase operates as an iterative loop with a fixed budget within each backbone setting. In each iteration, the system executes validation, diagnosis, and regeneration in sequence. The loop terminates when all validation checks pass or the budget is exhausted.

## H.1 Process Trace and Readiness

Each run records intermediate artifacts for auditing each phase, not only the final repository and score. The stable audit records include prepared units, reference-repository records, work-package plans, contract assertions, file-planning artifacts, validation reports, stage status, quality status, and usage summaries. Table 10 summarizes coarse audit statistics from the 20 Gemini full runs.

These logged counts make the generation process more concrete. Prepare extracts implementation units from the paper itself rather than from the PaperBench rubric, then attempts to ground those units with related repositories when actionable evidence is available. Prepared reference repositories are concentrated in a subset of papers, reflecting the benchmark’s mix of papers with different amounts of reusable prior-work structure. Plan then turns prepared units into work packages and contract assertions, which are contract-shaping objects rather than file-count targets. Generate materializes the file plans into repositories, and Repair audits requirement, trace, artifact, integration, and runtime checks.

These logged counts are not inputs to Paper-Bench judging. All reported Gemini full runs completed repository generation and were submitted to the PaperBench judge. We report them alongside PaperBench scores to make the pipeline trace auditable.

<table><tr><td>Paper</td><td colspan="2">Claude</td><td colspan="2">Gemini</td><td colspan="2">w/o ref.</td><td colspan="2">w/o req.</td></tr><tr><td></td><td>Score</td><td>Files</td><td>Score</td><td>Files</td><td>Score</td><td>Files</td><td>Score</td><td>Files</td></tr><tr><td>Adaptive Pruning</td><td>67.8</td><td>46</td><td>41.9</td><td>92</td><td>6.2</td><td>23</td><td>14.3</td><td>24</td></tr><tr><td>All in One</td><td>76.9</td><td>77</td><td>51.6</td><td>48</td><td>23.7</td><td>16</td><td>28.0</td><td>36</td></tr><tr><td>BAM</td><td>61.8</td><td>37</td><td>48.5</td><td>47</td><td>20.7</td><td>18</td><td>30.8</td><td>38</td></tr><tr><td>BBOX</td><td>86.4</td><td>280</td><td>15.6</td><td>70</td><td>14.6</td><td>15</td><td>14.2</td><td>75</td></tr><tr><td>Bridging Data Gaps</td><td>66.0</td><td>151</td><td>50.6</td><td>23</td><td>13.9</td><td>17</td><td>38.4</td><td>19</td></tr><tr><td>FRE</td><td>76.8</td><td>23</td><td>26.9</td><td>32</td><td>12.7</td><td>14</td><td>25.1</td><td>20</td></tr><tr><td>FTRL</td><td>63.9</td><td>28</td><td>40.1</td><td>71</td><td>11.7</td><td>23</td><td>13.4</td><td>51</td></tr><tr><td>LBCS</td><td>65.8</td><td>40</td><td>42.5</td><td>51</td><td>34.6</td><td>24</td><td>23.2</td><td>22</td></tr><tr><td>LCA on the Line</td><td>64.9</td><td>30</td><td>41.8</td><td>22</td><td>33.8</td><td>18</td><td>16.1</td><td>15</td></tr><tr><td>Mech. Understanding</td><td>75.7</td><td>201</td><td>48.1</td><td>26</td><td>25.9</td><td>14</td><td>4.6</td><td>20</td></tr><tr><td>PINN</td><td>81.5</td><td>163</td><td>61.3</td><td>35</td><td>52.2</td><td>14</td><td>53.3</td><td>18</td></tr><tr><td>RICE</td><td>75.1</td><td>34</td><td>31.0</td><td>81</td><td>19.9</td><td>16</td><td>27.8</td><td>25</td></tr><tr><td>Robust CLIP</td><td>63.0</td><td>51</td><td>29.5</td><td>19</td><td>15.7</td><td>14</td><td>25.6</td><td>20</td></tr><tr><td>Sample-specific Masks</td><td>82.0</td><td>80</td><td>53.8</td><td>50</td><td>28.0</td><td>17</td><td>21.3</td><td>33</td></tr><tr><td>SAPG</td><td>86.5</td><td>141</td><td>29.0</td><td>25</td><td>21.5</td><td>16</td><td>24.2</td><td>24</td></tr><tr><td>Sequential Neural Score Estimation</td><td>78.0</td><td>30</td><td>45.7</td><td>14</td><td>21.5</td><td>13</td><td>42.6</td><td>26</td></tr><tr><td>Stay on Topic with Classifier-Free Guidance</td><td>58.5</td><td>31</td><td>25.3</td><td>21</td><td>21.2</td><td>20</td><td>22.2</td><td>27</td></tr><tr><td>Stochastic Interpolants</td><td>80.8</td><td>73</td><td>39.1</td><td>25</td><td>21.5</td><td>19</td><td>29.9</td><td>24</td></tr><tr><td>Test-Time Model Adaptation</td><td>75.6</td><td>49</td><td>54.9</td><td>20</td><td>23.8</td><td>12</td><td>47.1</td><td>26</td></tr><tr><td>What Will My Model Forget</td><td>87.6</td><td>23</td><td>16.8</td><td>22</td><td>8.4</td><td>12</td><td>10.7</td><td>19</td></tr><tr><td>Mean</td><td>73.7</td><td>79.4</td><td>39.7</td><td>39.7</td><td>21.6</td><td>16.8</td><td>25.6</td><td>28.1</td></tr></table>

Table 6: Per-paper scores and generated repository sizes for our full and ablated runs. File counts come from the corresponding run metadata and are reported as process metadata rather than scoring inputs.

<table><tr><td>Setting</td><td>Papers</td><td>Score</td><td>Judge tok.</td><td>Pipe tok.</td><td>Hours</td><td>Calls</td></tr><tr><td>ReproAgent / Claude</td><td>20</td><td>73.7</td><td>12,868,729</td><td>17,833,795</td><td>9.3</td><td>35.4</td></tr><tr><td>ReproAgent / Gemini full</td><td>20</td><td>39.7</td><td>6,243,588</td><td>17,459,007</td><td>5.6</td><td>11.7</td></tr><tr><td>Gemini w/o reference evidence</td><td>20</td><td>21.6</td><td>6,582,787</td><td>4,034,007</td><td>1.5</td><td>13.8</td></tr><tr><td>Gemini w/o implementation requirements</td><td>20</td><td>25.6</td><td>4,668,761</td><td>4,429,321</td><td>1.3</td><td>12.6</td></tr></table>

Table 7: Usage metadata for our runs. Judge-token cells use root-level or summed leaf metadata depending on how the PaperBench summary stores usage; pipeline-token cells use exported run metadata or the estimate described above.

<table><tr><td>Paper</td><td>Claude judge Gemini judge </td><td>Gemini pipe No-ref judge No-ref pipe No-req judge No-req pipe</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Adaptive pruning</td><td>5,423,412</td><td>2,868,933</td><td>23,047,479</td><td>1,980,736</td><td>2,236,269</td><td>3,341,264</td><td>5,919,631</td></tr><tr><td>All-in-one</td><td>5,538,759</td><td>3,780,967</td><td>9,264,024</td><td>3,429,304</td><td>3,936,893</td><td>2,250,662</td><td>2,630,553</td></tr><tr><td>BAM</td><td>30,766,863</td><td>7,768,083</td><td>16,097,296</td><td>8,827,504</td><td>4,969,452</td><td>1,892,690</td><td>15,027,467</td></tr><tr><td>BBOX</td><td>8,567,577</td><td>774,669</td><td>58,268,936</td><td>4,903,571</td><td>4,131,573</td><td>4,373,037</td><td>4,509,759</td></tr><tr><td>Bridging data gaps</td><td>4,746,750</td><td>2,300,086</td><td>8,820,069</td><td>1,864,915</td><td>4,264,829</td><td>2,412,905</td><td>4,406,997</td></tr><tr><td>FRE</td><td>37,387,691</td><td>13,780,748</td><td>8,425,041</td><td>10,006,741</td><td>2,975,272</td><td>12,737,673</td><td>3,925,572</td></tr><tr><td>FTRL</td><td>11,530,535</td><td>12,388,835</td><td>23,625,213</td><td>3,976,499</td><td>5,255,585</td><td>291,099</td><td>4,136,571</td></tr><tr><td>LBCS</td><td>35,890,914</td><td>17,312,480</td><td>9,390,925</td><td>20,577,036</td><td>5,455,156</td><td>9,045,550</td><td>6,665,934</td></tr><tr><td>LCA on the line</td><td>25,328,353</td><td>613,700</td><td>8,697,850</td><td>18,609,830</td><td>4,585,214</td><td>4,880,558</td><td>3,644,005</td></tr><tr><td>Mech. understanding</td><td>2,176,841</td><td>1,589,700</td><td>7,923,157</td><td>1,340,680</td><td>3,134,806</td><td>1,383,850</td><td>4,118,063</td></tr><tr><td>PINN</td><td>7,360,909</td><td>5,691,891</td><td>13,620,939</td><td>7,053,399</td><td>4,120,951</td><td>6,536,328</td><td>4,348,687</td></tr><tr><td>RICE</td><td>16,389,279</td><td>1,550,040</td><td>74,534,103</td><td>6,929,485</td><td>3,739,967</td><td>5,148,614</td><td>3,261,044</td></tr><tr><td>Robust CLIP</td><td>5,588,405</td><td>2,147,322</td><td>16,285,677</td><td>3,131,589</td><td>3,445,426</td><td>2,689,292</td><td>2,283,182</td></tr><tr><td>Sample-specific masks</td><td>9,405,505</td><td>2,230,104</td><td>18,789,684</td><td>2,932,425</td><td>4,813,964</td><td>604,261</td><td>2,426,577</td></tr><tr><td>SAPG</td><td>4,553,396</td><td>3,507,987</td><td>8,202,509</td><td>3,339,036</td><td>3,793,693</td><td>2,585,659</td><td>4,160,601</td></tr><tr><td>Sequential NSE</td><td>3,293,860</td><td>2,661,812</td><td>6,486,552</td><td>2,075,601</td><td>4,250,141</td><td>2,723,479</td><td>5,711,514</td></tr><tr><td>Stay on topic</td><td>3,206,265</td><td>1,684,783</td><td>13,880,313</td><td>2,100,836</td><td>4,674,944</td><td>159,171</td><td>2,992,827</td></tr><tr><td>Stochastic interpolants</td><td>3,641,790</td><td>2,171,919</td><td>8,435,217</td><td>1,364,707</td><td>4,452,320</td><td>1,731,008</td><td>1,227,333</td></tr><tr><td>Test-time adaptation</td><td>7,102,606</td><td>3,326,170</td><td>7,495,412</td><td>3,511,711</td><td>3,539,717</td><td>336,977</td><td>5,935,940</td></tr><tr><td>What will my model forget</td><td>29,474,870</td><td>36,721,521</td><td>7,889,747</td><td>23,700,128</td><td>2,903,969</td><td>28,251,151</td><td>1,254,158</td></tr></table>

Table 8: Per-paper token metadata. “Judge” is PaperBench grading-token usage, with leaf metadata summed for summaries that store usage at leaf level; “pipe” is generation-token usage from exported run metadata when available.

<table><tr><td>Phase</td><td>Stage</td><td>Engineering role</td></tr><tr><td>Prepare</td><td>Input normalization</td><td>Normalize the target, task type, entities, and explicit constraints.</td></tr><tr><td>Prepare</td><td>Paper chunking</td><td>Segment paper text into section-level chunks with overflow handling.</td></tr><tr><td>Prepare</td><td>Requirement extraction</td><td>Extract implementation requirements that need concrete code, artifacts, interfaces, or validation targets.</td></tr><tr><td>Prepare</td><td>Reference acquisition</td><td>Collect reference repositories and survey file trees, README files, and likely reusable modules.</td></tr><tr><td>Plan</td><td>Work-package planning</td><td>Group prepared requirements into implementation-oriented packages such as data, method, training, evaluation, and artifact writing.</td></tr><tr><td>Plan</td><td>Package reference evidence</td><td>Attach package-local content and structure evidence; this supports package decisions but does not decide the final file tree.</td></tr><tr><td>Plan</td><td>Reference selection</td><td>Select actionable reference repositories and evidence items for the current package needs.</td></tr><tr><td>Plan</td><td>Pipeline plan</td><td>Specify package dependencies, runnable routes, and ordering constraints.</td></tr><tr><td>Plan</td><td>Global contract synthesis</td><td>Freeze requirement ownership, evidence bindings, result targets, artifact paths, interfaces, validation checks, and cross-package dependencies.</td></tr><tr><td>Plan</td><td>Architecture planning</td><td>Decide file tree, stack, entrypoints, config surfaces, dependency rules, and</td></tr><tr><td>Plan</td><td>Package-file planning</td><td>package-to-file layout. Refine architecture into per-file contracts with owned requirements, evi- dence, interfaces, artifacts, and validation hooks.</td></tr><tr><td>Generate</td><td>Canonical representation syn- thesis</td><td>Consolidate requirements, packages, contracts, architecture, file plans, evidence links, and validator expectations.</td></tr><tr><td>Generate</td><td>Local file generation</td><td>Generate source code file by file following file contracts and dependency ordering.</td></tr><tr><td>Repair</td><td>Repair validation</td><td>Validate through requirement review, contract checks, preflight checks, and sandboxed runtime execution.</td></tr><tr><td>Repair</td><td>Repair regeneration</td><td>Build a targeted repair plan and patch affected files while preserving requirements that already pass.</td></tr></table>

Table 9: Complete stage sequence of the REPROAGENT pipeline. Stages are executed in order; within-phase stages may share intermediate artifacts.

<table><tr><td>Stage</td><td>Audit statistic</td><td>Mean</td><td>Median</td><td>Range</td></tr><tr><td>Prepare</td><td>Extracted implementation units per paper</td><td>33.8</td><td>34.0</td><td>28-38</td></tr><tr><td>Prepare</td><td>Candidate reference repositories requested per paper</td><td>24.6</td><td>25.0</td><td>10-30</td></tr><tr><td>Prepare</td><td>Prepared reference repositories per paper</td><td>1.1</td><td>0.0</td><td>0-9</td></tr><tr><td>Prepare</td><td>Implementation units grounded by references per paper</td><td>6.2</td><td>0.0</td><td>0-29</td></tr><tr><td>Plan</td><td>Work packages per paper</td><td>19.6</td><td>21.5</td><td>3-38</td></tr><tr><td>Plan</td><td>Contract assertions per paper</td><td>6.9</td><td>7.0</td><td>5-10</td></tr><tr><td>Generate</td><td>Generated files per paper</td><td>35.5</td><td>26.5</td><td>14-80</td></tr><tr><td>Runtime metadata</td><td>Pipeline agent calls per paper</td><td>11.7</td><td>6.0</td><td>3-29</td></tr></table>

Table 10: Coarse audit statistics from the 20 Gemini full runs. These logged counts summarize the scale of preparation, planning, generation, and runtime interaction recorded during execution; they are not inputs to PaperBench judging.