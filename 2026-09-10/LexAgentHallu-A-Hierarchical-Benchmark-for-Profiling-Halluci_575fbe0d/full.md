# LexAgentHallu: A Hierarchical Benchmark for Profiling Hallucinations in Legal Agents

Yujin Zhou<sup>1∗</sup>, Mingxuan Zheng<sup>1∗</sup>, Chuxue Cao<sup>1∗</sup>, Yidan Huang<sup>1</sup>

Jiale Chen<sup>1</sup>, Yike Guo<sup>1</sup> , Sirui Han<sup>1†</sup>

<sup>1</sup>Hong Kong University of Science and Technology yzhouha@connect.ust.hk

## Abstract

As large language models are increasingly deployed as tool-augmented legal agents, they introduce agentic hallucinations where toolcall and reasoning errors cascade into fabricated holdings and miscited authority. However, existing legal benchmarks evaluate only single-turn QA with outcome-level metrics, while agentic hallucination benchmarks lack legal-specific diagnostic capability. Neither can determine to what extent and how a le gal agent hallucinates along its trajectory. To address these limitations, we introduce LexAgentHallu, a legal agentic hallucination benchmark designed to evaluate to what extent and how legal agents fail along multi-step trajectories. Built through a four-stage expert-in-theloop pipeline, LexAgentHallu contains 3,414 instances across 17 legal categories and 6 task types. Each instance is annotated under a dual-layer hallucination taxonomy of 7 midlevel categories and 27 fine-grained subclasses, covering both substantive errors and agentprocedural failures. We further design finegrained metrics that quantify to what extent each failure occurs and identify how it occurs along an agent’s execution path. Our evaluation across 18 proprietary and open-source agents uncovers a Right-Answer-Wrong-Reason effect and reveals that hallucination subclasses cluster rather than scatter, forming distinct agentic framework, legal task, and category profiles. These findings, invisible to outcome level evaluation, validate the diagnostic power of LexAgentHallu for evaluating agentic hallucination in law. <sup>1</sup>

## 1 Introduction

The rapid progress of large language models (LLMs) has catalyzed a paradigm shift from static, single-turn generation to agentic reasoning, where models plan, invoke tools, and iteratively interact with their environment to solve complex tasks (Yao et al., 2023; Anthropic, 2024; Xu et al., 2026; Zheng et al., 2026; Zhu et al., 2026). This shift is particularly attractive for high-stakes domains such as law, where reliable problem-solving demands statute retrieval, precedent grounding, and multihop reasoning that exceed the capacity of parametric memory alone. Reflecting this trend, recent efforts have actively pushed legal LLMs toward tool-augmented, multi-step legal agents, demonstrating clear gains on tasks such as legal question answering and judgement prediction (Han et al., 2026a,b; Zhang et al., 2025b; Yang et al., 2025b; Zhou et al., 2026a; Yang et al., 2026).

However, while the agentic paradigm alleviates certain single-turn hallucinations, it exposes new hallucination surfaces at the agentic level. In legal agent systems, hallucinations may arise not only from legal reasoning such as fabricating case citations or misquoting statutory provisions, but also from the agentic process itself: the agent may invoke a tool with incorrect arguments, misinterpret a retrieved passage, or lose track of the user’s original request across turns. Understanding how often these failures occur and whether they stem from legal knowledge or agentic processes is essential for improving legal agent systems. (Han et al., 2025; Lin et al., 2025; Zhou et al., 2026b).

Several recent efforts have started to evaluate hallucinations in legal LLMs (Dahl et al., 2024; Hu et al., 2025; Han et al., 2025) or in general-domain agents (Liu et al., 2026; Zhu et al., 2025). Despite these advances, no existing benchmark jointly examines to what extent legal agents hallucinate and at which level legal or agentic these hallucinations originate. We identify three critical limitations:

(1) Lack of Agent-Level Hallucination Evaluation in the Legal Domain. Existing legal hallucination benchmarks (Hu et al., 2025) are confined to single-turn QA over a narrow slice of legal tasks, treating the model as a closed-book oracle and leaving tool use, planning, and multi-turn interaction essentially unmeasured.

(2) Fragmented and Coarse-Grained Hallucination Taxonomies. General-domain agent hallucination studies adopt taxonomies (e.g., tool-call vs. output errors) that are blind to legally salient distinctions such as fabricated provisions and misapplied precedents. Conversely, legal hallucination taxonomies overlook agent-specific failures such as plan deviation and tool-grounding errors. Neither line of work offers a taxonomy that is simultaneously comprehensive and fine-grained enough to diagnose legal agent behavior.

(3) Limited Error Attribution Across Knowledge and Process Layers. Current evaluations score only the final answer with exact match or LLM-as-a-judge, without attributing errors to specific steps in the agent trajectory. When a legal agent produces a hallucinated conclusion, users cannot determine whether it stems from flawed legal knowledge or faulty agentic processes, limiting actionable diagnosis.

To bridge these gaps, we introduce LexAgentHallu, the first benchmark for profiling hallucinations in legal LLM agents. LexAgentHallu is constructed through an expert-in-the-loop pipeline, in which licensed legal professionals author and verify queries, gold trajectories, and rubric-based annotations across diverse legal tasks. We propose a dual-layer hallucination taxonomy that decomposes agentic legal errors into a Substantive Layer covering legal knowledge errors and an Agentic Layer covering process-level failures. To quantify behavior under this taxonomy, we design finegrained metrics with step-level attribution, measuring to what extent and how a legal agent hallucinates along its trajectory.

In summary, our contributions are:

• We release LexAgentHallu, the first benchmark dedicated to hallucinations in legal LLM agents, featuring expert-curated queries, gold references, and taxonomy-driven rubric checklist across diverse legal tasks and categories.

• We propose a dual-layer taxonomy with finegrained sub-classes that jointly capture what the agent gets wrong about the legal and agentic level during reasoning.

• We introduce taxonomy-aligned evaluation metrics including hallucination frequency, density, substantive/procedural cleanliness, and a Right-Answer-Wrong-Reason rate that go beyond outcome-only scoring to quantify how often and how broadly each rollout hallucinates, enabling diagnosis of failure patterns at both the layer and subclass level.

• We benchmark a wide range of proprietary and open-source legal and general agents, uncovering systematic analysis invisible to outcome-level evaluation (details in §4).

## 2 Related Work

## 2.1 LLM Agents and Legal Agents

A first wave of legal LLMs, including LawGPT (Zhou et al., 2024), ChatLaw (Cui et al., 2024), Lawyer-LLaMA (Huang et al., 2023), DISC-LawLLM (Yue et al., 2023a), and Lexi-Law (Li et al., 2024a), adapted general models to legal corpora via continued pre-training and instruction tuning, but remain closed-book and single-turn. Agentic frameworks such as ReAct (Yao et al., 2023) and Plan-and-Execute (Topsakal and Akinci, 2023) extend LLMs into autonomous problem solvers that decompose tasks, call external tools, and refine intermediate results. Building on this paradigm, LawThinker (Yang et al., 2026) and LRAS (Zhou et al., 2026a) equip LLMs with search tools. While these systems expand the practical utility of LLMs in law, their reliability under multi-step, tool-augmented execution remains largely uncharacterized.

## 2.2 Legal Benchmarks

A wide range of benchmarks evaluate LLMs on legal tasks. LawBench (Fei et al., 2024) and LexEval (Li et al., 2024b) cover broad task suites in Chinese law; LegalBench (Guha et al., 2023) targets common-law reasoning; PLawBench (Shi et al., 2026) and J1-Eval (Jia et al., 2025) extend to procedural law and judicial reasoning; and LEXam (Fan et al., 2025) focuses on bar-exam-style multi-step questions. Despite their breadth, these benchmarks target task accuracy in single-turn, closed-book settings, leaving hallucination in multi-step, toolaugmented agentic workflows unexplored.

## 2.3 Hallucination Benchmarks

General hallucination benchmarks such as HaluEval (Li et al., 2023), TruthfulQA (Lin et al., 2022), FActScore (Min et al., 2023), and FELM (Zhao et al., 2023) pioneered evaluation of factual errors in LLM outputs. In the legal domain, LegalHal-Bench (Hu et al., 2025) and CitaLaw (Zhang et al., 2025a) extend hallucination evaluation to statutes and citations. However, all of these are confined to single-turn outputs, treating hallucination as a property of an isolated answer rather than a multi-step trajectory. More recently, AgentHalluBench (Liu et al., 2026), HaluAgent (Cheng et al., 2024), and ToolBH (Zhang et al., 2024) evaluate hallucinations in tool-using agents, but target generic domains with task-agnostic taxonomies and lack legal grounding. Consequently, no existing benchmark evaluates hallucinations in multi-step, toolaugmented legal workflows spanning both substantive and procedural errors.

<table><tr><td>Top-layer</td><td>Mid-layer</td><td>Subclass</td><td>Description</td></tr><tr><td rowspan="10">L1 Substantive</td><td rowspan="2">L1.1 Authority</td><td>Citation-Content Misapplication</td><td>Real source, wrong application</td></tr><tr><td>Hierarchy Error Granularity Error</td><td>Confuses source rank</td></tr><tr><td></td><td></td><td>Wrong paragraph or item</td></tr><tr><td></td><td>Conceptual Confusion</td><td>Conflates legal concepts</td></tr><tr><td rowspan="6">L1.2 Doctrine</td><td>Element Misstatement</td><td>Distorts rule elements</td></tr><tr><td>Exception Omission</td><td>Ignores provisos</td></tr><tr><td>Consequence Error</td><td>Wrong legal effect</td></tr><tr><td>Doctrinal Position Confusion</td><td>Mixes doctrinal stances</td></tr><tr><td>Discretionary-Judgment Error</td><td>Misapplies discretion</td></tr><tr><td>Jurisdiction Error</td><td>Wrong court or venue</td></tr><tr><td rowspan="5">L1.3 Procedural Law</td><td>Period Error</td><td>Wrong deadline or limit</td></tr><tr><td>Procedural-Step Error</td><td>Skips or misorders steps</td></tr><tr><td>Procedural-Outcome Error</td><td>Wrong disposition</td></tr><tr><td>Appeal Error</td><td>Wrong remedy path</td></tr><tr><td>Fact Fabrication</td><td>Invents case facts</td></tr><tr><td rowspan="4">L1.4 Application &amp; Subsumption</td><td>Fact Omission</td><td>Drops material facts</td></tr><tr><td>Element-Fact Mismatch</td><td>Maps facts to wrong elements</td></tr><tr><td>Party Confusion</td><td>Mixes up party roles</td></tr><tr><td></td><td></td></tr><tr><td rowspan="5">L2 Agent-Proc.</td><td rowspan="5">L2.1 Planning &amp; Reasoning</td><td>Premature Closure</td><td>Concludes too early</td></tr><tr><td>Syllogism Error</td><td>Invalid inference</td></tr><tr><td>Self-Contradiction</td><td>Inconsistent claims</td></tr><tr><td>Step Skip / Conflation</td><td>Merges distinct steps</td></tr><tr><td>Out-of-Context Quoting</td><td>Cite used out of context</td></tr><tr><td>L2.2 Memory</td><td>Memory Hallucination</td><td>Multi-turn forgetting and misremembering</td></tr><tr><td>L2.3 Tool-Call &amp; Observation</td><td>Tool-Call Error</td><td>Wrong tool or arguments</td></tr></table>

Table 1: Hierarchical taxonomy of hallucinations (2 layers, 7 mid-level layers, 27 fine-grained classes).

## 3 Benchmark

In this section, we introduce LEXAGENTHALLU, an expert-curated benchmark for diagnosing hallucinations in legal LLM-based agents. Existing legal benchmarks predominantly score end-task accuracy, conflating what an agent answers with how it arrives there. For autonomous agents, however, a correct answer can be reached through an unsound procedure, and a well-formed trajectory can still yield a wrong conclusion. LEXAGEN-THALLU is built around this distinction via three design choices: (i) a two-layer hallucination taxonomy jointly characterizing substantive legal errors and agent-procedural errors (§3.1); (ii) a four-stage human-in-the-loop curation pipeline concentrating evaluation on hard cases with verified ground truth (§3.2); and (iii) a taxonomy-anchored rubric checklist that converts ground-truth answers into finegrained evaluation signals for an LLM-as-a-judge system (§3.4). We describe each component below, followed by dataset statistics (§3.3).

## 3.1 Dual-Layer Hallucination Taxonomy

LEXAGENTHALLU evaluates hallucinations along two orthogonal dimensions: Layer 1 captures errors in what the agent says about the law, and Layer 2 captures errors in how it arrives at that statement. Prior legal hallucination studies address only the former at coarse granularity (Dahl et al., 2024; Hu et al., 2025; Han et al., 2025), while general agentic-evaluation work (Lin et al., 2025; Liu et al., 2026) addresses only the latter in a domainagnostic manner; neither alone suffices for legal agents. Treating the two layers as orthogonal allows a single trajectory step to incur hallucinations from one layer, the other, both, or neither, letting LEXAGENTHALLU attribute each error to both a legal-content failure mode and the agentic mechanism responsible. Developed with licensed legal practitioners over iterative consolidation rounds, the taxonomy comprises 7 mid-level categories and 27 fine-grained subclasses (Table 1). Full definitions and worked examples are in Appendix D.

## 3.2 Curation of Benchmark

To make LEXAGENTHALLU realistic in difficulty, comprehensive in legal coverage, and faithfully aligned with the taxonomy, we curate it through a four-stage pipeline as shown in the Figure 1 Each stage is paired with a dedicated human-in-the-loop step described below.

![](images/86a9fcc20bf15afc300c38a076e13f208362c0f754a2f9746b5c3367c3530cdb.jpg)  
Figure 1: The four-stage pipeline for constructing LEXAGENTHALLU.

Stage 1: Multi-source data collection. Legalagent evaluation requires both broad coverage of standardized academic tasks and exposure to upto-date professional practice, neither of which is sufficient alone. We therefore aggregate raw questions from two complementary streams. The opensource stream draws from four widely used Chinese legal benchmarks, LexEval (Li et al., 2024b), Law-Bench (Fei et al., 2024), UniLaw (Cai et al., 2025), DISC-Law-Eval (Yue et al., 2023b), and PLaw-Bench (Shi et al., 2026). The real-world-practice stream consists of items manually collected from China’s National Judicial Examination (Fakao) released in the most recent three years. Detailed descriptions of each constituent dataset and licensing, are provided in Appendix B.

Stage 2: Data Filtering. Given the heterogeneity of the collected sources, a single filtering rule would either discard too much open-ended supervision or retain too many easy closed-ended items. We therefore apply task-type-specific strategies that distinguish closed-ended questions from openended questions.

For closed-ended questions, we use a multimodel rollout filter to remove items that current LLMs can already solve reliably. Each candidate is rolled out four times by each of six diverse models: LegalDelta-4B, LegalDelta-

14B (Team et al., 2025b), Qwen3-30B-A3B (Yang et al., 2025a), Gemma3-4B (Team et al., 2025b), Llama3.3-70B (Grattafiori et al., 2024), and GLM-4.5-Flash (Team et al., 2025a). We retain only questions on which all rollouts from every model fail, yielding a hard-case subset where hallucination is empirically likely. For open-ended questions, filtering depends on available supervision. PLawBench items are retained in full as they already carry expert-authored rubric annotations.

For open-ended subsets of LexEval and Law-Bench, we apply an LLM-as-a-judge filter that scores candidates along multiple legal-quality dimensions, retaining only items receiving the highest score across all dimensions. Judge prompts and scoring rubrics are reported in Appendix H.

Stage 3: Expert Annotation and Verification. After Stage 2, the filtered pool contains both items with pre-existing expert annotations and items lacking ground-truth supervision. We therefore apply a two-track annotation protocol carried out by annotators with formal legal training. (a) Verification track: For items already carrying sourcebenchmark annotations, annotators verify each ground-truth answer against authoritative statutory and doctrinal sources, correcting inaccuracies and discarding items whose supervision cannot be reliably reconstructed. (b) De novo track: For items without annotations (primarily from the Judicial Examination set), annotators produce gold answers, supporting statutory citations, and expected reasoning trajectories. Each item is independently labeled

![](images/8713b0663465e6a1c78a9ff24b49c544d23d4271e81976849a818e70373b57d9.jpg)  
Figure 2: Distribution of LEXAGENTHALLU instances across (a) 17 legal categories and (b) 6 task types.

by two annotators.

Stage 4: Taxonomy-driven rubric checklist. A verified ground-truth answer alone is a coarse reference signal: it tells an LLM-as-a-judge whether the agent is right, but not which taxonomy subclass it violates when wrong, nor at which step the violation occurs. To bridge this gap, we introduce a taxonomy-driven rubric checklist as a distinguishing feature of LEXAGENTHALLU. Concretely, taking the question and its verified ground-truth answer as input, we prompt Claude-4.6-Sonnet to generate a fine-grained checklist in which each item is explicitly anchored to a hallucination subclass defined in Layer 1 of the taxonomy (§3.1). Each generated checklist is then human-verified: legal experts inspect every item for factual correctness, taxonomy alignment, and non-redundancy, revising or removing items where necessary. A representative example is shown in Appendix D and the full generation prompt in Appendix H. The resulting rubric checklists form the backbone of our judge system (§3.4), enabling interpretable, taxonomyaligned hallucination diagnosis.

## 3.3 Dataset Statistics

LEXAGENTHALLU comprises 3,414 expertcurated instances spanning 17 legal categories and 6 task types. As shown in Figure 2(a), the datasets anchored in criminal and civil law, with broad coverage spanning intellectual property, commercial, administrative, constitutional and international law, as well as legal history. Figure 2(b) breaks down instances by task type: 1,542 (45.2%) are subjective and 1,872 (54.8%) are objective.

![](images/858bf062278c3f55f6f93b88786b8c36f22f0d24d2081564f0f9e79088639917.jpg)  
Figure 3: Applicability of rubric checklist items across Layer-1 hallucination subclasses, colored by top-level category.

Each instance is paired with a taxonomyanchored rubric checklist (§3.2, Stage4). Figure3 summarizes the applicability of rubric items across all 19 Layer-1 subclasses. Application-to-Facts subclasses apply to over 89% of samples, reflecting the universal need for fact-based reasoning. Most Authority and Substantive-Doctrine subcalsses display comparably high applicability, with a few narrowly scoped exceptions. Procedural-Law subclasses are more context-dependent, ranging from broadly relevant (procedural-outcome, period errors) to narrowly invoked (jurisdiction errors).

## 3.4 Our Judge System

## 3.4.1 How to Judge.

Since every rubric item is anchored to a specific taxonomy subclass (§3.2, Stage 4), evaluating an agent’s output reduces to a per-item decision: whether the corresponding hallucination subclass is present in the trajectory. We implement this as a rubric-anchored LLM-as-a-judge procedure with seven dedicated prompts—one per top-level taxonomy category (L1.1–L1.4 and L2.1–L2.3)—so that each call focuses the judge on a narrow, coherent error family rather than the full 27-subclass space. Given an agent trajectory r (comprising the final answer together with intermediate reasoning, memory states, and tool-call observations) and the instance’s rubric checklist, the judge returns a binary verdict per rubric item along with the trajectory span responsible for any flagged violation. Crucially, the judge operates on the full trajectory rather than the final answer alone; this is essential for detecting Layer-2 errors. We then aggregate peritem verdicts into the rollout-level metrics defined below. Prompt templates and robustness analysis are provided in Appendix H and Appendix E.2.

## 3.4.2 Metric design.

Hallucination Frequency. Hallucination Frequency (HF) measures whether at least one hallucination occurs within a target rollout, aggregated across rollouts. The overall and layer-level hallucination frequency are defined as follows:

$$
\begin{array} { c } { { \displaystyle H F = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } \mathbb { I } \Bigg ( \sum _ { s } h _ { r , s } \geq 1 \Bigg ) , } } \\ { { \displaystyle H F _ { L \ell } = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } \mathbb { I } \Bigg ( \sum _ { s \in L \ell } h _ { r , s } \geq 1 \Bigg ) , } } \end{array}\tag{1}
$$

where $\ell \in \{ 1 , 2 \}$ denotes the taxonomy level, I(·) is the indicator function, N is the number of rollouts, and $h _ { r , s } \in \{ 0 , 1 \}$ indicates whether rollout r triggers hallucination subclass s.

Hallucination Density. Hallucination Density normalizes the number of triggered subclasses by the number of applicable ones, yielding a per-rollout severity score in [0, 1] averaged over the corpus. This design directly addresses a key limitation of HF, which is insensitive to multiplicity: under HF, rollouts violating one or ten subclasses contribute identically. Thus, the layer-level Hallucination Density is formulated by

$$
H D _ { L \ell } = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } \frac { \sum _ { s \in L _ { \ell } } } { N _ { L \ell ( r ) } } h _ { r , s } ,\tag{2}
$$

where $N _ { L \ell ( r ) }$ is the number of categories in the ℓ layer. Additional metrics—including Answer Correctness, Substantive/Procedural Cleanliness, the Right-Answer-Wrong-Reason (RAWR) rate, and the co-occurrence Lift matrix used in our analysis— are formally defined in Appendix E.

## 4 Experiments

## 4.1 Experimental Setup

To comprehensively assess legal agentic behavior, we evaluate two categories of systems on LEXA-GENTHALLU. The first comprises the LRAS family (4B, 8B, 14B), standalone legal agent models running their native workflows. The second isolates the effect of orchestration: we pair three agentic frameworks (LAWTHINKER, Plan-and-Execute, and ReAct) with a shared legal tool suite and a range of backbones, including open-source models, including Qwen3.5-9B, Qwen3.5-27B (Yang et al., 2025a), and Qwen3.6-27B (Qwen Team, 2026)) and closed-source models, such as Gemini-3.1-Pro, and GPT-5.4 (Singh et al., 2026). This design enables direct comparison between specialized legal agents and general backbones under controlled tool access. Detailed setups are in Appendix F.

<table><tr><td>Model</td><td>HF↓  $\mathbf { H F } _ { L 1 }$ </td><td> $\mathbf { H F } _ { L 2 }$ </td><td> $\mathbf { H D } _ { L 1 }$ </td><td> $ { \mathbf { H D } } _ { L 2 }$ </td></tr><tr><td colspan="5">LawThinker</td></tr><tr><td>Gemini-3.1-Pro</td><td>0.911 0.881</td><td>0.627</td><td>0.258</td><td>0.233</td></tr><tr><td>GPT-5.4</td><td>0.909 0.900</td><td>0.462</td><td>0.279</td><td>0.172</td></tr><tr><td>Qwen3.5-9B</td><td>0.987 0.944</td><td>0.931</td><td>0.355</td><td>0.470</td></tr><tr><td>Qwen3.5-27B</td><td>0.956 0.891</td><td>0.860</td><td>0.304</td><td>0.375</td></tr><tr><td>Qwen3.6-27B</td><td>0.940 0.906</td><td>0.754</td><td>0.301</td><td>0.327</td></tr><tr><td colspan="5">Plan-and-Execute</td></tr><tr><td>Gemini-3.1-Pro</td><td>0.899 0.882</td><td>0.643</td><td>0.279</td><td>0.254</td></tr><tr><td>GPT-5.4</td><td>0.920 0.905</td><td>0.666</td><td>0.293</td><td>0.256</td></tr><tr><td>Qwen3.5-9B</td><td>0.953 0.921</td><td>0.855</td><td>0.330</td><td>0.356</td></tr><tr><td>Qwen3.5-27B</td><td>0.945 0.906</td><td>0.844</td><td>0.314</td><td>0.335</td></tr><tr><td>Qwen3.6-27B</td><td>0.933 0.892</td><td>0.829</td><td>0.303</td><td>0.315</td></tr><tr><td colspan="5"></td></tr><tr><td>Gemini-3.1-Pro</td><td>ReAct 0.890 0.879</td><td>0.617</td><td>0.279</td><td>0.247</td></tr><tr><td>GPT-5.4</td><td>0.909 0.903</td><td>0.601</td><td>0.302</td><td>0.242</td></tr><tr><td>Qwen3.5-9B</td><td>0.957 0.944</td><td>0.823</td><td>0.345</td><td>0.345</td></tr><tr><td>Qwen3.5-27B</td><td>0.913 0.877</td><td>0.739</td><td>0.291</td><td>0.251</td></tr><tr><td>Qwen3.6-27B</td><td>0.916 0.875</td><td>0.742</td><td>0.286</td><td>0.251</td></tr><tr><td colspan="5">LRAS</td></tr><tr><td>LRAS-Qwen3-4B</td><td>0.924</td><td>0.922</td><td>0.614 0.324</td><td>0.249</td></tr><tr><td>LRAS-Qwen3-8B</td><td>0.914 0.912</td><td>0.600</td><td>0.319</td><td>0.238</td></tr><tr><td>LRAS-Qwen3-14B</td><td>0.911</td><td>0.909 0.580</td><td>0.311</td><td>0.231</td></tr></table>

Table 2: Overall hallucination performance. All metrics are lower-is-better; best per column in bold.

## 4.2 Main Results

Table 2 summarizes the overall hallucination performance. We find that:

1) Pervasive Hallucination Across All Systems. Whether built as a general-purpose agentic workflow (Plan-and-Execute, ReAct), a legalspecialized workflow (LawThinker), or a dedicated legal agent model (LRAS), each configuration exhibits substantial hallucination rates. Even the bestperforming entry, Gemini-3.1-Pro under ReAct, still triggers at least one hallucination in 89.0% of rollouts (HF=0.890), with $\mathrm { H F } _ { L 1 }$ never dropping below 0.875 for any model. This confirms that even frontier legal agents remain far from faithful legal reasoning.

2) Framework-Dependent Hallucination Profiles. Model rankings shift markedly across frameworks, indicating that no single orchestration strategy universally dominates. ReAct delivers the strongest suppression on overall and substantive hallucinations, with Gemini-3.1-Pro achieving the lowest HF (0.890) and Qwen3.6-27B the lowest $\mathrm { H F } _ { L 1 }$ (0.875). LawThinker excels at suppressing procedural (L2) hallucinations, where GPT-5.4 attains the best $\mathrm { H F } _ { L 2 } ( 0 . 4 6 2 )$ and $\mathrm { H D } _ { L 2 } \left( 0 . 1 7 2 \right)$ , but at the cost of elevated L1 rates. Plan-and-Execute yields uniformly mediocre results without winning any column. Finer-grained category and task profiles are provided in Section 4.3.

3) Scaling Behavior and Evaluation Stability. Performance correlates positively with model scale: within the Qwen family under LawThinker, $\mathrm { 3 . 5 \mathrm { - } 9 B  3 . 5 \mathrm { - } 2 7 B  3 . 6 \mathrm { - } 2 7 B }$ monotonically reduces $\mathrm { H F } _ { L 2 }$ from 0.931 to 0.754. Frontier closedsource models still outperform the largest opensource variants on most metrics, exposing a persistent gap in open-source legal faithfulness. LRAS narrows this gap with substantially smaller backbones: LRAS-Qwen3-14B reaches $\mathrm { H F } _ { L 2 } { = } 0 . 5 8 0$ and $\mathrm { H D } _ { L 2 } { = } 0 . 2 3 1$ , surpassing all ReAct and Planand-Execute configurations of frontier models on $\mathrm { H D } _ { L 2 } .$ , while even LRAS-Qwen3-4B outperforms the 27B Qwen baselines on $\mathrm { H F } _ { L 2 }$ . Notably, all five metrics decrease monotonically from 4B to 8B to 14B, confirming that hallucination suppression scales favorably with model size and validating the discriminative power of our evaluation framework.

## 4.3 Detailed Analysis

Right-Answer-Wrong-Reason: a correct answer rarely implies a clean reasoning process. As shown in Figure 4, even when a legal agent produces a correct final answer, its reasoning trajectory still contains at least one substantive-legal hallucination (RAWR-S= 68%) and at least one agent-procedural hallucination (RAWR-P= 37%) on average. RAWR-S is persistently high and varies only within a narrow band (0.61–0.78), from GPT-5.4 RE to Qwen3.5-9B LT, and even the constrained LRAS remain at 0.66–0.67, confirming that substantive-legal errors are largely insensitive to model scale, backbone, or agentic framework. In contrast, RAWR-P varies much more (0.07–0.85) and shows two patterns: (i) a framework effect, where LRAS keeps agent-procedural hallucinations at ≤ 0.10 for all backbones, while multi-tool frameworks leave open-source models in the 0.42–0.85 range; (ii) a backbone effect, where GPT-5.4 reaches LRAS-level cleanliness even with multi-tool frameworks (LT: 0.11; RE: 0.08), while Qwen3.5-9B LT peaks at 0.85. ReAct achieves the lowest RAWR-P among multi-tool frameworks across all backbones, suggesting that action-observation grounding helps limit reasoning drift. These results demonstrate that objectivequestion accuracy is an unreliable proxy for legal faithfulness: even the best-performing configurations exhibit over 60% substantive contamination in their reasoning trajectories.

![](images/0b1dcb505b474ee45522173cec633a0d0bd9b433da811621a758c2aed5839ea9.jpg)  
Figure 4: RAWR rates on the objective-question subset. Substantive (RAWR-S, red) and procedural (RAWR-P, blue) RAWR rates, conditional on a correct answer (AC=1). LT, PE, RE denote LAWTHINKER, PLAN-AND-EXECUTE, and REACT frameworks. Metric definitions are in Appendix E.

Task-Type and Legal-Category Profiles. As shown in Figure 5, open-ended generation tasks such as Adjudication Analysis and Case Analysis saturate $\mathrm { H F } _ { L 1 }$ near 1.00, while constrained tasks such as Legal Knowledge QA (0.82) and Judgement Prediction (0.81) sit lower, likely because shorter outputs limit the surface area for hallucination. $\mathrm { H F } _ { L 2 }$ follows a different ordering: Adjudication Analysis still leads (0.83), yet Judgement Prediction (0.71) surpasses Legal Reasoning (0.68) and Legal Consultation (0.60), suggesting that multi-step evidence aggregation stresses planning and memory even when the final output is short. At the legal-category level (Figure 8), procedural and core doctrinal branches such as Criminal Procedure (0.95)andCivilLaw(0.93) form a highhallucination cluster, while theory-oriented categories such as Constitutional Law (0.79) and Jurisprudence (0.65) rank lowest on both layers. This likely reflects a structural difference in legal reasoning. Procedural and doctrinal fields require precise rule selection, source hierarchy control, deadline computation, and fact–rule application, whereas theory-oriented categories more often involve abstract principles and conceptual explanation, leaving fewer points at which a model can misstate a concrete provision, procedural step, or factual predicate (Linna, 2026; Fan et al., 2025).

![](images/060750017dee44059fbff9cbe26c721c9ccead84e0a081e3c842c15e46564787.jpg)  
Figure 5: Hallucination frequency by task type, sorted by $\mathrm { H F } _ { L 1 }$ descending.

![](images/0757cee2ba1bda1fbec1bf561b7fb404ecb0f2343a190c0b4dee332694cd692f.jpg)  
Figure 6: $2 7 \times 2 7$ subclass co-occurrence matrix. Cell colour is $\log _ { 1 0 }$ Lift as defined in §3.4; red indicates positive co-occurrence, blue negative, white near independence. Black lines mark group boundaries.

Hallucination subclasses cluster, not scatter. The lift matrix in Figure 6 reveals structure invisible to per-rollout aggregates. First, within-layer coupling dominates: dark-red blocks concentrate along the L1.2 and L1.3 diagonals, showing that once one doctrine or procedural error appears it reliably triggers further errors of the same family. Second, specific mechanistic pairs rise sharply above chance co-occurrence. Within L1.3, Temporal/Limitation errors co-occur with Remedy-Path errors (Lift = 2.82) and Procedural-Step with Procedural-Consequence errors (Lift = 2.59), suggesting a shared procedural clock whose miscalibration simultaneously corrupts deadlines and the remedies gated by them. The strongest cross-group signal is $\mathrm { L 1 . 1 . 3  L 1 . 2 . 5 ( L i f t = 7 . 0 7 ) }$ : legal-hierarchy errors and doctrinal-stance conflation systematically co-occur, pointing to a common upstream failure in mapping source hierarchy to doctrinal position. These findings indicate that legal hallucinations are not independent slips but structured failure modes rooted in shared mechanisms, a property that aggregate accuracy or per-error rates cannot expose. Practically, this co-occurrence structure suggests that detecting high-frequency hallucination types can serve as early signals to proactively flag and mitigate their correlated low-frequency counterparts before they propagate.

## 5 Conclusion

We presented LEXAGENTHALLU, the first benchmark for evaluating to what extent and how legal LLM agents hallucinate along multi-step trajectories. Built through a four-stage expert-in-the-loop pipeline, LEXAGENTHALLU provides 3,414 instances across 17 legal categories and 6 task types, each annotated under a dual-layer taxonomy of 7 mid-level categories and 27 fine-grained subclasses covering both substantive and agentic failures. Our fine-grained metrics enable step-level attribution that localizes where each failure occurs along an agent’s execution path. Evaluation of 18 proprietary and open-source agents uncovers a Right-Answer-Wrong-Reason effect and reveals that hallucination subclasses cluster rather than scatter, forming distinct agentic framework, legal task, and category profiles. These findings, invisible to outcome-level evaluation, validate the diagnostic power of LEXAGENTHALLU and argue for a shift toward trajectory-aware evaluation in legal agent design. We anticipate that LEXAGEN-THALLU will serve as a foundation for developing more trustworthy legal agents.

## Limitations

We acknowledge several limitations in the present work. First, LexAgentHallu is grounded in the Chinese legal system: all queries, gold trajectories, and rubric checklists are written in Chinese and anchored to PRC statutes, judicial interpretations, and procedural rules. Because legal practice varies significantly across jurisdictions, and agentic legal tasks in common-law or mixed systems may surface hallucination patterns absent from our taxonomy—for example, errors in analogical reasoning from precedent, distinguishing binding from persuasive authority under stare decisis, or drafting jury instructions. This reflects a deliberate trade-off: fine-grained step-level attribution requires close alignment with jurisdiction-specific authority structures, which limits cross-jurisdictional generalizability. Future work will extend the taxonomy and benchmark to other legal systems through collaboration with legal-AI researchers across jurisdictions.

Second, as LLM capabilities continue to evolve, newer models may exhibit different performance and hallucination patterns on LexAgentHallu from those reported in this paper. To support continuous evaluation, we plan to launch an online platform that tracks the latest state-of-the-art legal agents and regularly updates per-subclass frequencies, cooccurrence patterns, and RAWR profiles, providing the community with up-to-date diagnostic results.

Finally, our benchmark experiments are limited to single-model legal agents. We make this choice to keep the evaluation setting controlled and to attribute hallucinations more clearly to individual reasoning trajectories, rather than to multi-agent interaction, role specialization, or response aggregation. Complex multi-agent systems, such as roleplayed debate or judge–advocate pipelines, may exhibit different or lower hallucination rates, but they also introduce additional variables and substantially higher deployment costs. Extending Lex-AgentHallu to such systems is an important direction for future work.

## Acknowledgments

This work is funded in part by the HKUST Start-up Fund (R9911), Theme-based Research Scheme grant (T45-205/21-N), the InnoHK initiative of the Innovation and Technology Commission of the Hong Kong Special Administrative Region Government, and the research funding under HKUST-DXM AI for Finance Joint Laboratory (DXM25EG01).

## References

Rakesh Agrawal, Tomasz Imielinski, and Arun Swami.´ 1993. Mining association rules between sets of items in large databases. SIGMOD Rec., 22(2):207–216.

Anthropic. 2024. Building effective ai agents. https://www.anthropic.com/research/ building-effective-agents.

Hua Cai, Shuang Zhao, Liang Zhang, Xuli Shen, Qing Xu, Weilin Shen, Zihao Wen, and Tianke Ban. 2025. Unilaw-r1: A large language model for legal reasoning with reinforcement learning and iterative inference. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 18117–18131, Suzhou, China. Association for Computational Linguistics.

Xiaoxue Cheng, Junyi Li, Wayne Xin Zhao, Hongzhi Zhang, Fuzheng Zhang, Di Zhang, Kun Gai, and Ji-Rong Wen. 2024. Small agent can also rock! empowering small language models as hallucination detector. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 14600–14615.

Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, and Li Yuan. 2024. Chatlaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixture-of-experts large language model. Preprint, arXiv:2306.16092.

Matthew Dahl, Varun Magesh, Mirac Suzgun, and Daniel E Ho. 2024. Large legal fictions: Profiling legal hallucinations in large language models. Journal ofLegal Analysis, 16(1):64–93.

Yu Fan, Jingwei Ni, Jakob Merane, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Etienne Salimbeni, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, and Joel Niklaus. 2025. Lexam: Benchmarking legal reasoning on 340 law exams. arXiv preprint arXiv:2505.12864.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Alan Huang, Songyang Zhang, Kai Chen, Zhixin Yin, Zongwen Shen, and 1 others. 2024. Lawbench: Benchmarking legal knowledge of large language models. In Proceedings ofthe 2024 conference on empirical methods in natural language processing, pages 7933–7962.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh

Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, and 1 others. 2023. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems, 36:44123–44279.

Sirui Han, Yidan Huang, Guoying Lu, Shuchao Wu, Zefeng Chen, Yujin Zhou, Chuxue Cao, Yuyao Zhang, Mingxuan Zheng, Bubu Hou, and 1 others. 2026a. Trustworthy legal reasoning in 2026: A mid-year review of research, products, events, and governance.

Sirui Han, Zhizhuo Kou, Ruoxi Li, Yuyao Zhang, Yujin Zhou, Chuxue Cao, Han Zhu, Kunhao Pan, Haoran Li, Conghui He, and 1 others. 2026b. Trustworthy legal reasoning: A comprehensive survey.

Sophia Simeng Han, Yoshiki Takashima, Shannon Zejiang Shen, Chen Liu, Yixin Liu, Roque K. Thuo, Sonia Knowlton, Ruzica Piskac, Scott J Shapiro, and Arman Cohan. 2025. CourtReasoner: Can LLM agents reason like judges? In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 35291–35306, Suzhou, China. Association for Computational Linguistics.

Yinghao Hu, Leilei Gan, Wenyi Xiao, Kun Kuang, and Fei Wu. 2025. Fine-tuning large language models for improving factuality in legal question answering. In Proceedings of the 31st International Conference on Computational Linguistics, pages 4410–4427, Abu Dhabi, UAE. Association for Computational Linguistics.

Quzhe Huang, Mingxu Tao, Chen Zhang, Zhenwei An, Cong Jiang, Zhibin Chen, Zirui Wu, and Yansong Feng. 2023. Lawyer llama technical report. Preprint, arXiv:2305.15062.

Zheng Jia, Shengbin Yue, Wei Chen, Siyuan Wang, Yidong Liu, Zejun Li, Yun Song, and Zhongyu Wei. 2025. Ready jurist one: Benchmarking language agents for legal intelligence in dynamic environments. arXiv preprint arXiv:2507.04037.

Haitao Li, Qingyao Ai, Qian Dong, and Yiqun Liu. 2024a. Lexilaw: A scalable legal language model for comprehensive legal understanding.

Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, and Yiqun Liu. 2024b. Lexeval: A comprehensive chinese legal benchmark for evaluating large language models. Advances in Neural Information Processing Systems, 37:25061–25094.

Junyi Li, Xiaoxue Cheng, Xin Zhao, Jian-Yun Nie, and Ji-Rong Wen. 2023. HaluEval: A large-scale hallucination evaluation benchmark for large language models. In Proceedings of the 2023 Conference on

Empirical Methods in Natural Language Processing, pages 6449–6464, Singapore. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. Preprint, arXiv:2109.07958.

Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Kun Wang, Pengfei Cao, Qingyue Wang, Lixin Zou, Xu Chen, Chuan Zhou, Jia Wu, Peng Zhang, Qingsong Wen, Shirui Pan, and 5 others. 2025. Llm-based agents suffer from hallucinations: A survey of taxonomy, methods, and directions. Preprint, arXiv:2509.18970.

Eero Linna. 2026. Challenges for generative AI in legal reasoning. Discover Artificial Intelligence.

Xuannan Liu, Xiao Yang, Zekun Li, Peipei Li, and Ran He. 2026. Agenthallu: Benchmarking automated hallucination attribution of llm-based agents. Preprint, arXiv:2601.06818.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Qwen Team. 2026. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Yuzhen Shi, Huanghai Liu, Yiran Hu, Gaojie Song, Xinran Xu, Yubo Ma, Tianyi Tang, Li Zhang, Qingjing Chen, Di Feng, and 1 others. 2026. Plawbench: A rubric-based benchmark for evaluating llms in real-world legal practice. arXiv preprint arXiv:2601.16669.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 467 others. 2026. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

5 Team, Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, Kedong Wang, Lucen Zhong, Mingdao Liu, Rui Lu, Shulin Cao, Xiaohan Zhang, Xuancheng Huang, Yao Wei, and 152 others. 2025a. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. Preprint, arXiv:2508.06471.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas Mesnard, Geoffrey

Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon, Etienne Pot, Ivo Penchev, and 197 others. 2025b. Gemma 3 technical report. Preprint, arXiv:2503.19786.

Oguzhan Topsakal and Tahir Cetin Akinci. 2023. Creating large language model applications utilizing langchain: A primer on developing llm apps fast. In International conference on applied engineering and natural sciences, volume 1, pages 1050–1056.

Frank F. Xu, Yufan Song, Boxuan Li, Yuxuan Tang, Kritanjali Jain, Mengxue Bao, Zora Zhiruo Wang, Xuhui Zhou, Zhitong Guo, Murong Cao, Mingyang Yang, Hao Yang Lu, Amaad Martin, Zhe Su, Leander Melroy Maben, Raj Mehta, Wayne Chi, Lawrence Keunho Jang, Yiqing Xie, and 2 others. 2026. Theagentcompany: Benchmarking LLM agents on consequential real world tasks. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Xinyu Yang, Chenlong Deng, and Zhicheng Dou. 2025b. Glare: Agentic reasoning for legal judgment prediction. Preprint, arXiv:2508.16383.

Xinyu Yang, Chenlong Deng, Tongyu Wen, Binyu Xie, and Zhicheng Dou. 2026. Lawthinker: A deep research legal agent in dynamic environments. Preprint, arXiv:2602.12056.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations (ICLR).

Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, and Zhongyu Wei. 2023a. Disc-lawllm: Fine-tuning large language models for intelligent legal services. Preprint, arXiv:2309.11325.

Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, and Zhongyu Wei. 2023b. Disc-lawllm: Fine-tuning large language models for intelligent legal services. Preprint, arXiv:2309.11325.

Kepu Zhang, Weijie Yu, Sunhao Dai, and Jun Xu. 2025a. Citalaw: Enhancing llm with citations in legal domain. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11183–11196.

Kepu Zhang, Weijie Yu, Zhongxiang Sun, and Jun Xu. 2025b. An explicit syllogistic legal reasoning framework for large language models. Preprint, arXiv:2504.04042.

Yuxiang Zhang, Jing Chen, Junjie Wang, Yaxin Liu, Cheng Yang, Chufan Shi, Xinyu Zhu, Zihao Lin, Hanwen Wan, Yujiu Yang, Tetsuya Sakai, Tian Feng, and Hayato Yamana. 2024. ToolBeHonest: A multilevel hallucination diagnostic benchmark for toolaugmented large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11388–11422, Miami, Florida, USA. Association for Computational Linguistics.

Yiran Zhao, Jinghan Zhang, I Chern, Siyang Gao, Pengfei Liu, Junxian He, and 1 others. 2023. Felm: Benchmarking factuality evaluation of large language models. Advances in Neural Information Processing Systems, 36:44502–44523.

Mingxuan Zheng, Yujin Zhou, Chuxue Cao, Boqin Yin, Yuyao Zhang, Jiapeng Sun, Shuaishuai Gong, Sirui Han, and Yike Guo. 2026. Skillprox: Self-evolving agent skills via proximal textual gradient descent. arXiv preprint arXiv:2608.07449.

Yujin Zhou, Chuxue Cao, Jinluan Yang, Lijun Wu, Conghui He, Sirui Han, and Yike Guo. 2026a. Lras: Advanced legal reasoning with agentic search. Preprint, arXiv:2601.07296.

Yujin Zhou, Yidan Huang, Sirui Han, and Yike Guo. 2026b. How well can large language model agents simulate complex legal dispute resolution? Artificial Intelligence and Law, pages 1–78.

Zhi Zhou, Jiang-Xin Shi, Peng-Xiao Song, Xiao-Wen Yang, Yi-Xuan Jin, Lan-Zhe Guo, and Yu-Feng Li. 2024. Lawgpt: A chinese legal knowledge-enhanced large language model. Preprint, arXiv:2406.04614.

Han Zhu, Chengkun Cai, Yuanfeng Song, Xing Chen, Sirui Han, and Yike Guo. 2026. Self-evolving deep research via joint generation and evaluation. Preprint, arXiv:2606.04507.

Zhenghao Zhu, Chuxue Cao, Sirui Han, Yuanfeng Song, Xing Chen, Caleb Chen Cao, and Yike Guo. 2025. Medinsightbench: Evaluating medical analytics agents through multi-step insight discovery in multimodal medical data. Preprint, arXiv:2512.13297.

## A Ethics Statement

All data used in this work originates from publicly available or properly licensed sources, and we have verified compliance with the respective licensing terms. The dataset does not contain any personally identifiable or sensitive information. Human participation was limited to annotation and quality verification; all annotators are domain experts holding qualifications in law or and were compensated at a fair hourly rate commensurate with their expertise.

## B Datasets

This appendix provides detailed descriptions of the five legal benchmarks used in curation of LexAgentHallu. Together, they span a broad spectrum of legal cognitive demands—from factual recall and concept recognition to multi-step reasoning and practical document drafting—enabling a comprehensive assessment of hallucination patterns across diverse task types.

## B.1 LexEval

LexEval (Li et al., 2024b) is a large-scale Chinese legal evaluation suite that organizes 23 tasks (approximately 14,150 questions) under the Legal Cognitive Ability Taxonomy (LexCog). Tasks are sourced from established legal corpora, real barexamination items, and expert-curated annotations. Beyond standard legal knowledge assessment, Lex-Eval uniquely incorporates ethical reasoning scenarios, testing whether models can navigate value conflicts that arise in legal practice. We select a representative subset covering knowledge recall, statute interpretation, and case-based reasoning tasks.

## B.2 LawBench

LawBench (Fei et al., 2024) structures its evaluation around three cognitive levels that mirror progressive stages of legal expertise: Memorization (retrieval of statutory provisions and doctrinal facts), Understanding (entity recognition, relation extraction, and semantic comprehension), and Application (multi-step reasoning over realistic legal scenarios). Its 20 tasks adopt five output formats—single-label classification, multi-label classification, regression, extraction, and generation—providing diversity in both cognitive demand and answer granularity.

## B.3 UniLaw-Eval

UniLaw-Eval (Cai et al., 2025) targets logical inference within legal contexts through 800 rigorously constructed items (426 single-choice and 374 multi-choice questions). Each item is designed to require multi-step deductive or analogical reasoning rather than surface-level pattern matching, making it particularly suitable for probing doctrinal and subsumption hallucinations (L1.2 and L1.4 in our taxonomy).

<table><tr><td>Benchmark</td><td>#Items</td><td>#Tasks</td><td>Format</td></tr><tr><td>LexEval</td><td>~14,150</td><td>23</td><td>Mixed</td></tr><tr><td>LawBench</td><td></td><td>20</td><td>Mixed</td></tr><tr><td>UniLaw-Eval</td><td>800</td><td>一</td><td>MCQ</td></tr><tr><td>DISC-LawEval</td><td></td><td>一</td><td>MCQ+Open</td></tr><tr><td>PLawBench</td><td>850</td><td>13</td><td>Open</td></tr></table>

Table 3: Overview of evaluation benchmarks.

## B.4 DISC-LawEval

DISC-LawEval (Yue et al., 2023a) adopts a dualtrack evaluation paradigm. Its objective track draws multiple-choice questions from standardized professional examinations (e.g., the National Unified Legal Profession Qualification Examination) and stratifies them into three difficulty tiers to differentiate knowledge retrieval from deep deduction. Its subjective track provides 300 expert-constructed open-ended scenarios—including legal consultation and judgment prediction—assessed along accuracy, completeness, and clarity dimensions, offering a natural testbed for detecting reasoning and application hallucinations.

## B.5 PLawBench

PLawBench (Shi et al., 2026) is a practice-oriented benchmark designed to bridge the gap between academic evaluation and real-world legal workflows. It models three core practitioner activities: public legal consultation, practical case analysis, and legal document generation. The benchmark comprises 850 questions spanning 13 practice scenarios, each accompanied by expert-designed multi-dimensional rubrics (approximately 12,500 rubric items in total). These fine-grained rubrics assess issue identification, fact extraction, structured reasoning, and document coherence, making PLawBench particularly well-suited for evaluating whether models produce legally sound and internally consistent outputs under realistic task complexity.

Summary statistics. Table 3 provides an overview of the scale, task format, and primary cognitive focus of each benchmark.

## C Annotation Details

Two experts carried out all annotation work over a period of 21 days, each contributing an average of 4–6 hours per day. For instance, without preexisting ground truth, the annotators produced goldstandard labels from scratch. In instances where ground truth was already available, the annotators verified it, confirming correctness or correcting minor discrepancies. The total annotation effort thus amounts to approximately 170–250 person-hours.

![](images/2402999811916cfad2a0ac4d97aaa17431444785577649f59a8783755d58ff57.jpg)  
Figure 7: Per-group hallucination profile across the four agent frameworks. Bars are $\operatorname { H F } _ { g }$ (any-hit rate within group).

![](images/9db10d55b33e2d807958b59650a1bd3b571abe644dd467fc0e379f3da44b9ef4.jpg)  
Figure 8: Hallucination frequency for the top-12 legal categories in LEXAGENTHALLU, sorted by $\mathrm { H F } _ { L 1 }$ descending. Procedural and core doctrinal branches (Criminal Proc., Civil Proc., Criminal Law, Civil Law) cluster above 0.93, while theory-oriented categories (Constitutional, Jurisprudence) sit notably lower.

## D Details of the Hallucination Taxonomy

This section provides the full specification of our two-layer hallucination taxonomy.

## D.1 Layer 1: Substantive Legal Hallucination

Layer 1 captures errors in the content of legal reasoning, organized along the canonical structure of legal analysis: source → rule → procedure → application. It comprises 4 mid-level categories and 19 fine-grained subcategories.

## D.1.1 L1.1 Authority Hallucination

Scope. All tasks requiring citation of authoritative legal sources (statutes, judicial interpretations, case numbers, official documents).

• L1.1.1 Source Fabrication. The model fabricates a non-existent statute, judicial interpretation, case number, or official document. Example: “Article 1300 of the Civil Code provides. . . ” (the Civil Code has only 1260 articles).

• L1.1.2 Citation–Content Misapplication. The cited source genuinely exists but is either inapplicable to the case at hand or its content is described inaccurately. Example: “Article 1062 of the Civil Code governs joint marital debts” (it actually governs joint marital property).

• L1.1.3 Hierarchy Error. The model confuses the hierarchical rank, territorial scope, or departmental authority of legal sources. Example: Invoking a local regulation to override a superior statute; treating an administrative regulation as a “law.”

• L1.1.4 Granularity Error. The statute name and article number are both correct, but the specific paragraph, item, or sub-item is wrong.

Example: Article 142, Paragraph 1 → incorrectly cited as Paragraph 2.

## D.1.2 L1.2 Doctrine Hallucination

Scope. Tasks examining substantive-law concepts, constituent elements, exceptions, legal consequences, and doctrinal positions.

• L1.2.1 Conceptual Confusion. The model misidentifies concept A as concept B at the categorical level, causing the entire reasoning chain to follow the wrong doctrinal branch. Example: Apparent agency → unauthorized agency; guarantee → debt assumption; material misunderstanding → fraud.

• L1.2.2 Element Misstatement. The correct legal concept is identified, but one or more of its constituent elements, formation conditions, or applicability thresholds are stated incorrectly. Example: Correctly identifying apparent agency but claiming it requires “fault” (the actual standard is “good faith + reasonable reliance”).

• L1.2.3 Exception Omission. A legally decisive exception, defense, justification, or limiting condition is omitted. Example: Analyzing contract validity without mentioning that “unconscionability renders the contract voidable”; discussing breach liability without noting force majeure exemption. Note: This subcategory is triggered only when the omitted exception is outcome-determinative (e.g., it distinguishes MCQ distractors or constitutes the core of an open-ended answer).

• L1.2.4 Consequence Error. The legal consequence, form of liability, remedy, or penalty is stated incorrectly. Example: Stating “damages” when the correct remedy is “restitution”; concluding “voidable” when the contract is in fact “void.”

• L1.2.5 Doctrinal Position Confusion. The model conflates the prevailing view with minority positions, or inconsistently switches between doctrinal stances within the same analysis. Example: Mixing subjective and objective theories of causation in criminal law; alternating between “not yet formed” and “void” to describe the same contract within one response.

• L1.2.6 Discretionary-Judgment Error. Indeterminate legal concepts (e.g., “relatively serious circumstances,” “unconscionability,” “sufficient to overturn the original judgment”) are applied outside their accepted boundaries. Example: Misjudging the threshold for “serious circumstances” in sentencing; incorrectly applying the boundary of unconscionability in civil law.

Distinguishing L1.2.1 from L1.2.2. The key diagnostic question is: “Did the model take the wrong conceptual branch?” If yes → L1.2.1. If the concept is correctly identified but individual elements are misstated → L1.2.2.

## D.1.3 L1.3 Procedural-Law Hallucination

Scope. Tasks involving civil procedure, criminal procedure, administrative litigation, or arbitration.

• L1.3.1 Jurisdiction Error. Errors in hierarchical, territorial, exclusive, or agreed jurisdiction. Example: Assigning a case to a basiclevel court when the amount in controversy exceeds its jurisdictional threshold.

• L1.3.2 Period Error. Errors in statutes of limitation, peremptory periods, appeal deadlines, or filing periods. Example: Stating the general civil limitation period as 2 years (correct: 3 years); stating the civil appeal period as 10 days (correct: 15 days).

• L1.3.3 Procedural-Step Error. Errors in the sequence or requirements of procedural steps such as filing, acceptance, defense, evidence submission, preservation, or enforcement. Example: Claiming the defense period is 30 days (correct: 15 days); asserting that property preservation requires prior filing of suit (pre-suit preservation is available).

• L1.3.4 Procedural-Outcome Error. Errors in procedural dispositions—confusing dismissal ofthe action (procedural defect) with dismissal of the claim (substantive ruling), or misidentifying withdrawal or default judgment. Example: Concluding “dismiss the claim” for what is actually a jurisdictional defect warranting “dismiss the action.”

• L1.3.5 Appeal Error. Errors in remedy paths including second-instance appeal, retrial, prosecutorial protest (kangsu), administrative reconsideration, or enforcement objections. Example: Advising direct litigation when administrative reconsideration is a mandatory prerequisite.

## D.1.4 L1.4 Application & Subsumption Hallucination

Scope. All tasks requiring extraction of facts from a case narrative and mapping them onto legal rules.

• L1.4.1 Fact Fabrication. The model invents facts, amounts, dates, parties, or acts not present in the problem statement. Example: The prompt contains no mention of “divorce by agreement,” yet the model states “the parties divorced by agreement.”

• L1.4.2 Fact Omission. A material fact is overlooked or a non-material fact is treated as decisive. Example: Failing to note that “the defendant is a state functionary,” leading to omission of a bribery charge.

• L1.4.3 Element–Fact Mismatch. The legal rule and its elements are stated correctly, but the facts are mapped to the wrong elements. Example: The elements of apparent agency are correctly recited, but the model claims “the counterparty was unaware” satisfies “good faith” when the prompt explicitly states the counterparty had knowledge.

• L1.4.4 Party Confusion. The identities, procedural positions, or rights/obligations of parties are mixed up. Example: Reversing “Party A” and “Party B”; treating an agent as the principal.

Distinguishing L1.4.3 from L1.2.2. L1.2.2 addresses errors in stating the elements themselves; L1.4.3 addresses correct elements incorrectly matched to the case facts.

## D.2 Layer 2: Agent-Procedural Hallucination

Layer 2 targets failures along the agent’s decisionmaking trajectory, independent of substantive legal content. It comprises 3 mid-level categories and 9 fine-grained subcategories.

## D.2.1 L2.1 Planning & Reasoning Hallucination

Scope. Reasoning subcategories can be detected within a single rollout; planning subcategories typically require an agent trajectory.

Distinction from L1.4.3. L1.4.3 captures element-to-fact correspondence errors; L2.1 captures breakdowns in the logical chain itself (deduction, analogy, causation).

• L2.1.1 Premature Closure. The model locks onto a conclusion before sufficient information has been processed, and subsequent reasoning serves only to rationalize the premature conclusion. Example: Determining the criminal charge before fully reading the case facts, then selectively citing statutes that support only that charge.

• L2.1.2 Syllogism Error. The deductive chain (major premise [statute] → minor premise [facts] → conclusion) is broken or inverted. Example: The statute requires “intent,” the facts establish only “negligence,” yet the model concludes intent is satisfied.

• L2.1.3 Self-Contradiction. Conclusions, element selections, or legal characterizations within a single output or across steps are mutually inconsistent. Example: First stating “the contract is valid,” then later stating “the contract is voidable due to unconscionability”; step 5 contradicts the statutory basis cited in step 2.

• L2.1.4 Step Skip / Conflation. Steps that require separate analysis are collapsed into one, or the model jumps from facts directly to a judgment without doctrinal grounding. Example: A case analysis that directly states “Judgment: the defendant bears full liability” without identifying elements or citing statutes.

• L2.1.5 Out-of-Context Quoting. A statutory provision is partially quoted as authority while contextual qualifications are suppressed. Example: Quoting “a loan contract is formed when the lender provides the loan” while omitting the qualifying phrase “between natural persons.”

Note on L2.1.1. A writing style that states the conclusion first and then provides complete supporting analysis does not constitute Premature Closure. L2.1.1 requires both: (a) the model fails to genuinely consider alternative options or counterexamples after committing to an answer, and (b) the reasoning trace shows no balanced analysis of competing candidates.

## D.2.2 L2.2 Memory Hallucination

Scope. Multi-turn dialogues or long agent trajectories only.

This category captures failures in maintaining consistency with information established earlier in the interaction. Specifically, it covers three manifestations: (1) forgetting premises from prior turns, including the system prompt, the user’s original request, or previously confirmed facts; (2) misremembering key information such as party names, monetary amounts, dates, or the focal issues in dispute; and (3) drifting away from the system prompt or the user’s original question—e.g., the user asks about topic A but the model responds about topic B.

## D.2.3 L2.3 Tool-Call & Observation Hallucination

Scope. Trajectories involving tool invocations only. We distinguish two failure modes: errors in calling tools versus errors in interpreting tool outputs.

• L2.3.1 Tool-Call Error. The wrong tool is selected, parameters are incorrect, invocation order is wrong, a necessary call is omitted, or redundant calls are made. Example: Using a statute-retrieval tool to search for case law; passing an incorrect article number to the retrieval API; using a generic search engine when a judicial-interpretation API is available.

• L2.3.2 Observation Misuse. The tool returns correct content, but the model misreads or misinterprets the output. Example: The retrieval tool returns five relevant statutes but the model only reads the first; a case date of 2018 is misread as 2008; the model relies on a summary snippet rather than the full text.

## E Judge

## E.1 Metrics

For each rollout r, let $\hat { Y } ( r )$ denote the predicted answer extracted from the model’s free-form output, and let $Y ( r )$ denote the gold answer. Answer correctness is defined as:

$$
\operatorname { A C } ( r ) = \mathbb { I } [ { \hat { Y } } ( r ) = Y ( r ) ] \in \{ 0 , 1 \} ,\tag{3}
$$

the Substantive Cleanliness and the Procedural Cleanliness are defined as follows:

$$
\mathrm { S C } ( r ) = \mathbb { I } ( \sum _ { s \in L 1 } h _ { r , s } = 0 ) \in \{ 0 , 1 \} ,\tag{4}
$$

$$
\mathrm { P C } ( r ) = \mathbb { I } ( \sum _ { s \in L 2 } h _ { r , s } = 0 ) \in \{ 0 , 1 \} .\tag{5}
$$

We then define the Right Answer Wrong Reasoning (RAWR) rate. The RAWR rate for the above metric can be calculated as

$$
R A W R - S = P ( S C = 0 | A C = 1 ) ,\tag{6}
$$

$$
R A W R - P = P ( P C = 0 | A C = 1 ) .\tag{7}
$$

Co-occurrence Matrix: Sections II–V We also define the hallucination frequency for each subclass s as follows:

$$
P ( s ) = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } h _ { r , s } ,\tag{8}
$$

represents the hit rate of subclass s across all rollouts. Thus, the Joint Halluciantion Frequency of a Pair of Subclasses can be derived by

$$
P ( s , t ) = \frac { 1 } { N } \sum _ { r = 1 } ^ { N } h _ { r , s } h _ { r , t } ,\tag{9}
$$

which represents the fraction of rollouts hit by subclass s and t simultaneously. Note that $h _ { r , s } h _ { r , t } = 1$ only when both indicators are 1.

Then, we derive the Lift metric:

$$
\operatorname { L i f t } ( s , t ) = { \frac { P ( s , t ) } { P ( s ) P ( t ) } } ,\tag{10}
$$

by applying Bayes’ Theorem, which can be further simplified as

$$
{ \mathrm { L i f t } } ( s , t ) = { \frac { P ( s , t ) / P ( s ) } { P ( t ) } } = { \frac { P ( t | s ) } { P ( t ) } } .\tag{11}
$$

Interpretation Lift is a standard measure from association rule mining (Agrawal et al., 1993): Lift $( s , t ) = 1$ indicates that s and t co-occur exactly as often as expected under independence; Lift $( s , t ) > 1$ indicates positive association, meaning a rollout exhibiting s is more likely than chance to also exhibit t (and vice versa); and Lift $( s , t ) < 1$ indicates negative association, meaning the two error types co-occur less often than independence would predict.

## E.2 Judge Robustness

To validate the viability of our evaluation, we engaged three legal experts, all of whom hold the Certificate of Legal Professional Qualification of the People’s Republic of China, to label 2,391 rubric items and 1,395 valid judge decisions independently. For each item, we measured both the consistency between the LLM judge and human annotations and the inter-annotator agreement among the three experts. The results are summarized in Table ??.

Taken together, these results demonstrate the robustness of our judge: the high inter-expert agreement confirms that the judgments are well-defined and stable across independent experts, while the judge–human consistency shows that our automated pipeline aligns closely with expert assessment.

## F Detailed Experimental Setup

## F.1 Hardware and Compute Environment

All experiments run on a single Slurm cluster of DGX nodes, each equipped with 8×NVIDIA H800 80 GB GPUs (224 vCPU, ∼2 TB host RAM). Opensource backbones and the judge LLM are served by vLLM 0.6.x in a Python 3.12 / CUDA 12.8 environment; closed-source models are queried through an OpenAI-compatible reseller endpoint and therefore consume no local GPUs.

## F.2 Generation Parameters

All systems are evaluated under an identical inference configuration with deterministic decoding: temperature = 0.0 (greedy), top\_p = 0.8, top\_k = 20 , and max\_tokens = 4,096 per LLM call.

## F.3 Agentic Workflow Parameters

The three orchestrations are configured as follows.

• ReAct: max\_steps = 8. Each step emits either <action> → Observation or <final\_answer>; budget exhaustion triggers a force-finalise step that composes the answer from the existing trace.

• Plan-and-Execute: max\_steps = 10. A planner first emits a typed JSON plan; each plan step is executed through the same tool catalogue as ReAct; on budget exhaustion the agent likewise produces a force-finalised answer.

• LawThinker: a two-turn scenario between a rule-based Questioner and a LawThinker trainee (Agent.Trainee.LC\_Generic), with an optional deep\_analysis sub-stage that runs a second pass over the answer;

LAWTHINKER\_DISABLE\_THINKING=1 suppresses backbone-native chain-of-thought tags so the only reasoning trace observable to the judge is the explicit dialog history.

## F.4 Shared Tool Suite

ReAct, Plan-and-Execute, and LawThinker share an identical legal tool suite. All tools are exposed verbatim in the agent system prompts; the exact same string is replayed to the judge LLM (tool\_specs.py) to enable the L2.3 tool-misuse sub-class to be evaluated byte-for-byte against the spec the model actually saw. The catalogue is partitioned into three families:

(a) Knowledge-exploration tools (7). Used to surface candidate authorities and analogous artefacts when the agent has only a natural-language question:

• law\_retrieval(query, topk) – top-k statute retrieval over a Chinese statute index (BGE-M3 dense embeddings); returns the full text of each candidate article.

• law\_recommendation(law) – given an explicit article reference (e.g. “ Article 201 of the Criminal Law”), return statutes that frequently co-cite with it.

• charge\_expansion(charges) – expand a list of criminal charges into closely related charges from the same chapter or Supreme-Court interpretation.

• case\_retrieval(type, query) – analogous-case retrieval in the civil or criminal sub-corpus.

• template\_retrieval(template\_type) – fetch the official template for a procedural document (e.g. statement of complaint, defence).

• plan\_generation(document\_type) – emit a section skeleton for the requested document.

• procedure\_retrieval(court\_type, stage) – retrieve the civil (5-stage) or criminal (3-stage) court procedure; only relevant in the moot-court setting and rarely fired on LEXAGENTHALLU.

(b) Verification tools (6). Used after a candidate authority has been picked, to confirm applicability and consistency:

<table><tr><td>Reliability</td><td>LLM-human consistency</td><td>Inter-annotator (three-expert) agreement</td></tr><tr><td>Rubric</td><td>94.22%</td><td>91.00%</td></tr><tr><td>Judge</td><td>86.90%</td><td>95.07%</td></tr></table>

Table 4: Reliability of the rubric construction and judge scoring, measured by LLM–human consistency and inter-annotator (three-expert) agreement.

• law\_check(law\_name) – exact-text lookup for a given article reference; the canonical way to verify a citation.

• fact\_law\_relevance\_check(fact, law) – decide whether a given article applies to a given fact pattern.

• crime\_law\_consistency\_check(crime, law) – verify that a charge and a cited article actually correspond under criminal law.

• document\_format\_check(document\_type, document) – format-validate a drafted procedural document.

• law\_query\_rewrite(query, context) – rewrite a vague query into a statute-grounded one before re-issuing law\_retrieval.

• procedure\_check(court\_type) – check completeness of a proposed civil/criminal procedure trajectory.

## (c) Web fallback (1).

• web\_search(query,

summarizer\_version) public-web retrieval via a Bright Data SERP API, followed by Jina-Reader page fetch and a question-aware summariser (default summarizer\_version=original, max 5 hits per query). Reserved for queries that local retrieval cannot satisfy (latest policies, local regulations, foreign law, procedural “how-to” questions).

A short tool-use discipline is appended to every system prompt (“rules of engagement”): identical (name, arguments) tuples may not be re-issued; law\_retrieval/law\_check are allowed at most two misses per dispute before the agent must escalate to web\_search or close; one good web\_search hit must be followed by <final\_answer>; common-sense or elementary-doctrine questions should close immediately without invoking any tool. Memory tools (memory\_store/memory\_fetch) are deliberately removed from the catalogue for the singlerollout benchmark setting; their implementations remain in the runtime for multi-turn scenarios.

ReAct and Plan-and-Execute expose all 14 tools in (a)+(b)+(c) at once. LawThinker exposes the (a) family during the initial response phase and the (b) family during the deep-analysis verification phase, mirroring its two-stage protocol.

## G Supplementary Detailed Analysis

## G.1 Framework Specialization (Per-Group Profile)

Figure 7 reports the within-group any-hit rate $\operatorname { H F } _ { g }$ for each of the four agent frameworks across the seven hallucination groups. The view complements main paper finding 1 (Framework-Dependent Hallucination Profiles) by exposing the specific group on which each framework concentrates errors: LawThinker / Plan-and-Execute / ReAct sit within ±2 pp of each other on every L1 group, while LRAS uniquely lifts L1.2 / L1.4 (Doctrine, Application) by 7–8 pp—an expected side-effect of its self-RL training distribution—but lowers L2.3 (Tool/Obs) by the largest margin. This directly motivates the simplicity-vs.-diversity discussion in the main paper’s tool-tradeoff analysis.

## G.2 Per-Task and Per-Domain Difficulty

Table 5 reports the full HF/HD numbers for the six task types of LEXAGENTHALLU (Figure 5 visualises only HF and annotates HD). Figure 8 reports the same metrics for the top 12 legal categories, sorted by $\mathrm { H F } _ { L 1 }$ descending.

## H Prompt Templates

## H.1 Rollout Prompt

In this section, we provide the rollout-stage prompts for LawThinker, LRAS, Plan-and-Execute, and ReAct.

<table><tr><td>Task type</td><td>HDL1</td><td>HDL2</td></tr><tr><td>Adjudication Analysis</td><td>0.492</td><td>0.403</td></tr><tr><td>Case Analysis</td><td>0.293</td><td>0.304</td></tr><tr><td>Legal Reasoning</td><td>0.240</td><td>0.259</td></tr><tr><td>Legal Consultation</td><td>0.213</td><td>0.182</td></tr><tr><td>Legal Knowledge QA</td><td>0.197</td><td>0.208</td></tr><tr><td>Judgement Prediction</td><td>0.215</td><td>0.261</td></tr></table>

## Table 5: Per-task-type hallucination profile on LEXA-GENTHALLU, sorted by HD<sub>L1</sub> descending.

You are a legal-reasoning assistant that may call domain  
specific legal tools whenever necessary.   
Available tools   
memory\_fetch - retrieve stored knowledge or context.   
law\_retrieval - retrieve the top-k most relevant statutes given   
a natural-language query.   
law\_recommendation - return statutes similar to the one provided.   
charge\_expansion - expand a list of charges with related ones.   
case\_retrieval - retrieve similar civil/criminal cases.   
template\_retrieval - fetch a document template (e.g. complaint   
document, defence document).   
writing\_plan\_generation - generate a writing plan for the given   
document type.   
procedure\_retrieval - retrieve the civil/criminal court   
procedure (stage=0 for the full procedure).   
Tool-calling format   
Whenever a tool is needed, output   
<tool\_call>{"name": ..., "arguments": ...}</tool\_call>.   
The system will respond with   
<tool\_call\_result> ... </tool\_call\_result>.   
Example: examples

## DeepVerifier Prompt

You are a deep-analysis legal assistant. Given (i) the user's   
last query or response, (ii) the current reasoning trace,   
and (iii) the result of the exploration step, decide   
whether the retrieved information is correct and relevant;   
decide whether more exploration is needed; and store any   
key facts or statutes.   
Tool principles   
1. Choose the appropriate checking tool to verify the accuracy   
and relevance of retrieved knowledge.   
2. Summarize key information and store it with memory\_store.   
Available tools   
memory\_fetch - retrieve stored knowledge / context   
memory\_store - store key knowledge / context   
law\_article\_check - verify law article content   
fact\_law\_relevance\_check - check law applicability to facts   
charge\_law\_consistency\_check - check charge-law consistency   
search\_query\_rewrite - rewrite law retrieval queries   
document\_format\_check - check document format   
procedure\_check - Check procedural compliance   
Tool-calling format   
<tool\_call>{"name": ..., "arguments": ...}</tool\_call>   
<tool\_call\_result> ... </tool\_call\_result>.   
Example: examples

## LRAS Prompt

You are a professional legal assistant skilled in answering   
legal questions accurately through in-depth research.   
Workflow:   
1. First conduct reasoning and analysis within the tags. If you   
determine that existing knowledge is insufficient to

2. To find legal authorities, use <search>query</search> to call the search tool; results will be returned within < information></information>.

## ReAct Prompt

## ### Tool List

\- fact\_law\_relevance\_check: Fact-provision relevance verification. Determine whether a given legal article (e.g. Criminal Law Article 201) applies to the case facts. Parameters: {"fact": "str", "law": "str"}

\- crime\_law\_consistency\_check: Charge-provision matching verification. Verify whether the given charge matches the specified criminal law article (e.g. Criminal Law Article 201). Parameters: {"crime": "str", "law": "str"}

document\_format\_check: Document format inspection. Check the format of legal documents. Submit complete document content without omission. Parameters: {"document\_type": " str(Complaint|Defense Statement)", "document": "str"}

\- law\_query\_rewrite: Query rewriting. Revise the original query combined with case background to make it clearer for retrieval. Parameters: {"query": "str", "context": "str"}

\- procedure\_check: Proceeding inspection. Verify the completeness of civil or criminal court procedures. Parameters: {"court\_type": "str(Civil Court|Criminal Court

4. Tool Selection: Use law\_check if the article number is   
confirmed; use law\_retrieval for natural language   
descriptions; use web\_search for newly released policies,   
local regulations, service procedures and foreign laws.

2. Upgrade Path: If law\_retrieval / law\_check fails to return valid results for over 2 times regarding one disputed point, immediately switch to web\_search or draw a conclusion. If web\_search returns relevant summaries, stop searching and output the final answer directly.

3. Output only one <action> or one <final\_answer> per round. Do not include both.

4. Do not add irrelevant labels such as Observation or Step numbers.

## Plan-and-Execute Prompt–Planner system

You are a legal task planner. Generate executable plans in JSON format for legal questions raised by users.

Output Rules (Must be strictly followed):   
1. Output a single JSON object only, with no explanations or   
Markdown code blocks.   
2. The top-level JSON must contain the field "plan" whose value   
is a string array.   
3. The plan shall include at least 2 steps. There is no upper   
limit, while 3 to 6 actionable and progressive steps are   
recommended.   
4. Each step must be specific and executable (including which   
tool to call and which dispute points to focus on). Vague   
expressions such as "think about it" or "make an analysis"   
are prohibited.   
Example (Multiple Choice Question):   
"plan": [   
"Sort out key facts of the question and identify disputed   
points (e.g. conduct nature, subjective state, role   
identity)",   
"Verify relevant legal provisions and constitutive elements   
for each option respectively",   
"Determine the only correct answer and briefly explain why   
other options are excluded"   
]   
}   
Example (Consultation Question):   
{   
"plan": [   
"Extract the user's facts and demands, and identify the   
relevant legal field",   
"Retrieve applicable laws and judicial interpretations;   
switch to web\_search if no results are obtained locally",   
"Draw conclusions combined with facts, deliver risk   
reminders and feasible suggestions"   
]   
}

## Plan-and-Execute Prompt–Excute System

You are a Plan-and-Execute legal assistant. Carry out tasks in accordance with the plan, call tools when necessary, and finally deliver explainable conclusions.

(1) Priority: Use law\_check to obtain full text if the exact article number is known (e.g. Criminal Law Article 397); use law\_retrieval for natural language descriptions only.

(2) Query Writing: Focus on "crime/legal system name + key constitutive elements" or "specific article number", avoid using complete questions; topk is set to 3-5 by default.

(3) Poor Retrieval Results: Retry once with revised keywords. If no valid results are obtained, switch to web\_search or make a conclusion based on existing information. Do not repeatedly retry the same query.

(4) Coverage: Current effective statutes, judicial interpretations and departmental rules of Chinese mainland. Low hit rate for local regulations, newly released policies, international conventions and foreign laws.

\- law\_recommendation: Similar provision recommendation. Return similar provisions given a specified legal article (e.g. Criminal Law Article 201). Parameters: {"law": "str"}

\- charge\_expansion: Related charge expansion. Return similar charges based on the given charge list. Parameters: {" charges": "List[str]"}

case\_retrieval: Similar case retrieval. Return analogous cases given case type (civil or criminal) and case information. Parameters: {"type": "str(Civil Case|Criminal Case)", " query": "str"}

template\_retrieval: Document template retrieval. Obtain templates for specified legal documents (e.g. complaint, defense statement). Parameters: {"template\_type": "str( Complaint|Defense Statement)"}

\- plan\_generation: Writing plan generation. Create a writing plan according to the document type. Parameters: {" document\_type": "str(Complaint|Defense Statement)"}

\- procedure\_retrieval: Court procedure retrieval. Only applicable to moot court scenarios. Stage stands for proceeding phase; set stage=0 for the full procedure. Civil proceedings contain 5 phases and criminal proceedings contain 3 phases. Parameters: {"court\_type": "

<<QUESTION>>   
### Plan Status ([done] Completed / [doing] In Progress / [todo]   
Pending)   
<<PLAN\_STATUS>>   
### Execution Progress (Including tool calls and original   
results)   
<<PROGRESS>>   
Continue with the next step. Maximum 10 rounds allowed.

str(Civil Court|Criminal Court)", "stage": "int(0-4|0-2)"}

\- law\_check: Legal provision verification. Return the full text of a specified legal article (e.g. Criminal Law Article 201). Parameters: {"law\_name": "str"}

Usage Guideline: Prioritize law\_check for higher accuracy when the exact article number is available.

fact\_law\_relevance\_check: Fact-provision relevance verification. Determine whether a given legal article (e.g. Criminal Law Article 201) applies to the case facts. Parameters: {"fact": "str", "law": "str"}

crime\_law\_consistency\_check: Charge-provision matching verification. Verify whether the given charge matches the specified criminal law article (e.g. Criminal Law Article 201). Parameters: {"crime": "str", "law": "str"}

\- document\_format\_check: Document format inspection. Check the format of legal documents. Submit complete document content without omission. Parameters: {"document\_type": str(Complaint|Defense Statement)", "document": "str"}

\- law\_query\_rewrite: Query rewriting. Revise the original query combined with case background to make it clearer for retrieval. Parameters: {"query": "str", "context": "str"}

\- procedure\_check: Proceeding inspection. Verify the completeness of civil or criminal court procedures. Parameters: {"court\_type": "str(Civil Court|Criminal Court )"}

\- web\_search: Web search. Retrieve legal information from public internet and generate key summaries related to the user's question.

Application Scenarios:

(1) No valid results after one keyword revision via law\_retrieval / law\_check;

(2) Inquiries involving newly released policies, local regulations, details of typical cases, international conventions or foreign laws;

(3) Consultation on practical procedures, required materials and service channels.

Rules:

(1) Compile summaries and deliver the final answer once relevant results are returned. Do not resubmit the same query.

(2) Explicit rules such as "The insured shall..." or "In accordance with Article XX of XX Law" in summaries are deemed valid results. You may refine the query at most once before concluding the task.

(3) Run web\_search no more than 2 times for a single disputed point.

Parameters: {"query": "str (Chinese keywords, within 25 characters recommended)", "summarizer\_version": "str( original|v1) (optional, original by default)"}

## ### Tool Usage Rules (Must be strictly followed)

1. No Duplicate Calls: Do not call the same tool with identical parameters twice. For retries, adjust keywords, topk value or switch to another tool.

2. Upgrade Path: If law\_retrieval / law\_check fails to return valid results for over 2 times regarding one disputed point, immediately switch to web\_search or draw a conclusion. If web\_search returns relevant summaries, stop searching and output the final answer directly.

3. Prioritize Conclusion: Give the final answer directly if the question can be judged based on general knowledge or basic legal principles. For multiple-choice questions with clear legal rationales for all options, stop tool calls immediately.

4. Tool Selection: Use law\_check if the article number is confirmed; use law\_retrieval for natural language descriptions; use web\_search for newly released policies, local regulations, service procedures and foreign laws.

\### Output Rules (Must be strictly followed)

1. Choose one format for each round: - Tool Call: Thought: <You may state "Now proceed with Step K"> <action>{"name":"Tool Name","arguments":{...}}</action> - Final Answer: Thought: ... <final\_answer>Final answer for the user</final\_answer>

2. The tool name in <action> must be selected from the tool list, and arguments must be valid JSON.

3. Output only one <action> or one <final\_answer> per round. Do not include both.

4. Focus on steps marked [doing] in the plan status. Explain the completion of current steps in Thought and proceed to the next step implicitly.

5. Do not add irrelevant labels such as Observation or Step numbers.

[Executor user turn template]

\### Current Question

## H.2 Data Filter Prompt

In this section, we provide the prompts used to filter open-ended questions of different types during data curation.

## Data Filter: LawBench 3-8 (Consultation)

You are a strict data quality reviewer examining a sample of legal consultation Q\&A.

\textbf{Task Background}

\- Question: Colloquial legal inquiries from general users, usually with redundant content, typos, missing subjects, mixed regional information and other issues.

Answer: A standard answer consists of two parts: (a) A direct response to the user's question (marked with " Answer: ..." or equivalent expressions);

(b) Citation of specific laws and regulations (marked with " Legal Basis: ..." or equivalent expressions).

\textbf{Key Principles (Must be strictly followed)}

1. You are neither required nor allowed to judge the accuracy of legal provisions cited, correctness of legal conclusions or rationality of judgments.

2. You shall score solely based on observable textual features, including structure, relevance, completeness and citation presence. Only judge existence rather than correctness.

\textbf{Scoring Criteria} (Score each item with an integer from 1 to 5)

1. Question Clarity

\- 5: The user's core demand is clearly identifiable.

\- 3: The general meaning is understandable, with redundant content or awkward expressions.

\- 1: The question is confusing with missing subjects or key information, and the inquiry cannot be figured out.

## 2. Question-Answer Relevance

\- 5: The answer clearly addresses the core demand and stays on topic.

\- 3: The answer is partially relevant but deviates or barely relates to the question.

\- 1: The answer is almost irrelevant or completely off-topic.

3. Answer Structural Compliance (Legal Structure Evaluation) - 5: The answer clearly contains a two-part structure: direct response plus legal basis. Wording may vary, but the two sections are distinguishable.

\- 1: Neither part exists; the content is empty remarks or rhetorical questions.

4. Legal Citation Format (Only judge existence, \textbf{not} accuracy)

5: At least one standard citation of specific laws or regulations appears (e.g., \textit{Article X of XX Law}, textit{According to YY Rules}, \textit{Contract Part of the Civil Code}).

\- 3: Only general legal names are mentioned without specific provisions (e.g., merely stating "in accordance with civil law").

\- 1: No legal citation of any form is present.

\- Note: Do not verify the authenticity or correctness of citations; only check whether citation formats appear in the text.

## 5. Answer Completeness

\- 5: The answer is complete with no obvious truncation.

\- 3: Slightly incomplete but the main idea is intelligible.

1: The text is confusing and the questions are   
incomprehensible.   
. Completeness of Case Elements (Legal structure assessment)   
- 5: The case contains all key elements for answering,   
including parties, conducts, time and disputes.   
- 3: Basic elements are provided with minor omissions.   
1: Severe lack of key elements, leaving no factual basis   
for answering.   
3 Question Coverage   
5: All sub-questions are responded to correspondingly.   
- 3: Most sub-questions are covered with one omission.   
skipped entirely.   
- Note: Only check if responses exist, not whether the   
answers are correct.   
4. Argumentation Quality (Legal structure assessment, no fixed   
format required)   
5: Each question is supported by identifiable legal   
analysis, containing clear positions/conclusions and   
corresponding arguments. Any valid structure is acceptable   
- 3: Merely a position with weak arguments, or plain   
reasoning without clear conclusions.   
1: No recognizable legal analysis, such as repeating the   
question, listing provisions without analysis, or giving a   
conclusion with no reasons.   
- Note: Do not evaluate the validity of analysis; only check   
for the coexistence of positions and arguments.   
5. Legal Citation Format (Only judge existence, not accuracy)   
- 5: At least one specific citation of laws or regulations is   
included.   
- 3: Only general legal names are mentioned without specific   
clauses.   
1: No legal citations of any kind are present.   
6. Answer Completeness   
- 5: The answer is fully structured with no truncation   
- 3: Slightly truncated but the main idea remains intact.   
- 1: Ends abruptly or uses long ellipses to replace key   
arguments.   
Internal Consistency   
- 5: No contradictions between answers to different sub  
questions, or between conclusions and supporting reasons.   
- 3: Minor conflicts that do not affect the main point.   
8. Substantiality of Answer   
5: Each sub-question is elaborated with detailed legal   
arguments.   
- 3: Brief arguments with key points included.   
1: Vague arguments or a single-sentence conclusion without   
analysis.   
Output Format   
Strictly follow the JSON format below. Do not add any   
explanations, prefixes, suffixes or reasoning content.   
\`\`\`json   
"Question Clarity": <1-5>,   
"Completeness of Case Elements": <1-5>,   
"Question Coverage": <1-5>,   
"Argumentation Quality": <1-5>,   
"Legal Citation Format": <1-5>,   
"Answer Completeness": <1-5>,   
"Internal Consistency": <1-5>,   
"Substantiality of Answer": <1-5>,   
"Suggestion": "<Retain | Reject>",   
"Reason": "<Brief explanation within 50 words>"   
}   
Judgment Rules   
Retain: All scores \(\ge 4\) and no score equals 1.   
Reject: Average score ranges from 3.0 to 3.9, or any item scores   
2.   
Reject: Average score \(< 3.0\), or any item scores 1.   
Question (Case & Questions)   
\_QUESTION\_\_Answer (Standard Answer)   
\_\_ANSWER\_\_Please output the JSON scores strictly in the above   
format without any extra text outside the markdown code   
block:

- 1: The answer ends abruptly with unfinished sentences or   
ellipses replacing key discussions.   
6. Internal Consistency of Answer   
- 5: Statements in different parts are consistent and free of   
contradictions.   
- 3: Minor wording conflicts exist without affecting the   
ll l i   
- 1: Obvious contradictions on the same conclusion appear   
across sections.   
7. Substantiality of Answer   
- 5: The answer contains detailed reasoning and substantial   
content.   
- 3: The content is brief but delivers basic information.   
- 1: The answer is overly short, consists of empty remarks,   
repeats the question largely, or only gives a conclusion   
without any analysis.   
\textbf{Output Format}   
Output strictly in the following JSON format. Do not add any   
explanations, prefixes, suffixes or reasoning content.   
\`\`\`json   
{   
"Question Clarity": <1-5>,   
"Question-Answer Relevance": <1-5>,   
"Answer Structural Compliance": <1-5>,   
"Legal Citation Format": <1-5>,   
"Answer Completeness": <1-5>,   
"Internal Consistency of Answer": <1-5>,   
"Substantiality of Answer": <1-5>,   
"Suggestion": "<Retain | Reject>",   
"Reason": "<Brief explanation within 50 words>"   
}   
\textbf{Judgment Reference (Comprehensive judgment based on   
scores above)}   
Retain: All scores   
>=4   
and no score equals 1.   
Reject: Average score ranges from 3.0 to 3.9, or any single item   
scores 2.   
Reject: Average score   
<3.0   
, or any single item scores 1.   
\textbf{Question (User Inquiry)}   
\_\_QUESTION\_\_   
\textbf{Answer (Lawyer / AI Response)}   
\_ANSWER\_\_   
Please output the JSON scores strictly in the above format   
without any extra text outside the markdown code block:

## Data Filter: LexEval 5-4

You are a rigorous data quality reviewer assessing samples of   
subjective questions for the National Judicial Examination.   
Task Background   
- Question: A case description followed by one or more specific   
questions (marked as "Q: 1) ... 2) ..."), which usually   
contains multiple sub-questions.   
- Answer: Legal analysis addressing the raised questions. \   
textbf{No restrictions on answer formats}, including "   
Conclusion + Reasons", comparison of multiple viewpoints,   
step-by-step analysis combined with case facts, or   
citation of relevant precedents and legal theories.   
Provide responses separately for each sub-question if   
there are several.   
Key Principles (Must be strictly followed)   
1. You are neither required nor allowed to judge the accuracy of   
cited legal provisions, correctness of legal conclusions   
or reasonableness of judgments.   
2. Score exclusively based on observable textual features such   
as structure, relevance, completeness and citation forms.   
Judge only the presence of elements rather than their   
correctness.   
Scoring Criteria (Assign an integer score from 1 to 5 for each   
item)   
1. Question Clarity   
- 5: The case facts and questions are clearly stated, with   
clear examination focuses.   
- 3: Readable with slightly awkward expressions in parts.

You are a rigorous data quality reviewer examining samples of judgment analysis generation.

## \textbf{Task Background}

\- Question: Factual findings stated by the court. Ideally, it shall include parties, cause of action/legal relationship, contract or event process, and focus of dispute.

Answer: Legal analysis and judgment document. The standard structure consists of two parts:

(a) Argumentation section starting with "The court holds..." for legal analysis on disputed facts;

(b) Clear judgment rulings (e.g., "Judgment as follows: ..." plus specific verdicts).

## \textbf{Key Principles (Must be strictly followed)}

1. You are neither required nor allowed to judge the accuracy of cited legal provisions, correctness of legal conclusions or reasonableness of judgments.

2. Score solely based on observable textual features including structure, relevance, completeness and citation forms. Only judge the presence of elements rather than their correctness.

\textbf{Scoring Criteria} (Assign an integer score from 1 to 5 for each item)

## 1. Case Clarity

\- 5: Facts are stated coherently and clearly with good readability.

\- 3: Readable but slightly disorganized.

\- 1: Confusing and fragmented text, barely intelligible.

2. Completeness of Case Elements (Legal structure assessment) - 5: The question contains all three key elements: parties, cause of action/legal relationship, and disputed facts.

\- 3: One key element is missing.

\- 1: Multiple key elements are missing, and the trial focus cannot be identified.

\- Note: Do not assess rationality or authenticity of elements; only check their presence.

## 3. Question-Answer Relevance

\- 5: The judgment analysis clearly centers on the disputed facts in the given case.

\- 3: Partially relevant with off-topic content.

\- 1: The answer is almost irrelevant or completely off-topic.

## 4. Judgment Structural Compliance (Legal structure assessment)

\- 5: The answer contains both a recognizable argumentation section (marked by "The court holds" or equivalent expressions) and explicit judgment rulings.

\- 3: Only the argumentation section or only the judgment rulings are provided.

\- 1: Neither identifiable argumentation nor judgment rulings are included.

## 5. Legal Citation Format (Only judge existence, \textbf{not} accuracy)

\- 5: At least one specific citation of laws or regulations appears (e.g., Article X of the Civil Procedure Law).

\- 3: Only general legal names are mentioned without specific clauses.

\- 1: No legal citations of any form are present.

## 6. Answer Completeness

\- 5: The content is fully organized with complete and definite judgment rulings.

\- 3: Judgment rulings exist but are not fully stated.

\- 1: The text ends abruptly, with missing rulings or unfinished sentences.

## 7. Internal Consistency

5: The viewpoints in the argumentation section are

consistent with the final judgment rulings.

\- 3: Minor discrepancies without actual contradictions.

\- 1: Obvious contradictions between argumentation and rulings.

\- Note: Do not judge legal validity; only check textual consistency.

## 8. Substantiality of Answer

\- 5: The argumentation includes detailed factual analysis and legal reasoning.

\- 3: Concise argumentation with key points covered.

\- 1: The argumentation consists merely of formal remarks with no substantial analysis.

## \textbf{Output Format}

Strictly follow the JSON format below. Do not add any

explanations, prefixes, suffixes or reasoning content.

## \`\`\`json

"Legal Citation Format": <1-5>,

## \textbf{Judgment Rules}

Retain: All scores \$\ge 4\$ and no score equals 1. - Reject: Average score ranges from 3.0 to 3.9, or any item scores 2.

\- Reject: Average score \$< 3.0\$, or any item scores 1.

\textbf{Question (Factual Findings)} \_\_QUESTION\_\_

\textbf{Answer (Court's Opinions + Judgment Rulings)} \_\_ANSWER\_\_

Please output the JSON scores strictly in the above format without any extra text outside the markdown code block:

## H.3 Rubric checklist Annotation Prompt

## Rubric Checklist Annotation (Stage 1)

You are a senior legal expert tasked with creating an \*\* evaluation checklist\*\* for a legal LLM hallucination benchmark.

## [Your Task]

Given a legal question (question + manually labeled ground truth + metadata), extract \*\*evaluation anchors\*\* (per-subclass checklist) for each Layer-1 hallucination subclass based on the ground truth.

## [Key Constraints]

1. The ground truth is authoritative. \*\*Do not question, revise or add new conclusions\*\*. You only organize existing content in the ground truth into structured check items.

2. Your output will serve as the rubric for downstream LLM-asjudge evaluation. Every item must be \*\*verifiable by another judge\*\*.

3. \*\*Do not omit any Layer-1 subclass\*\*. All 19 subclasses must be included (see schema below).

Note: In version v0.3.2, L1.1 is reorganized into four consecutive subclasses (L1.1.1 \textasciitilde{} L1.1.4). Strictly follow the numbers and field names specified here; do not use old numbering.

4. If a subclass is irrelevant to the question, set \`"applicable ": false\` and add a one-sentence \`reason\`. Do not remove the entire field.

5. Output a single JSON object only. No Markdown code blocks, no extra explanations, no \`\`\`json\`\`\` wrappers.

6. Use double quotes for all strings in JSON. No trailing commas. Keep all Chinese text unchanged.

## [Overview of Layer-1 Subclass Schema]

All subclasses must be present, and each contains the field \`" applicable"\` (true/false).

\- When \`"applicable": true\`, fill in all exclusive fields of the subclass as specified below; no omissions allowed.

\- When \`"applicable": false\`, only fill in the \`"reason"\` field with one sentence explaining irrelevance.

\- Fill in [] or null for any field that cannot be inferred from the question. Do not fabricate content.

## [L1.1.1 Source Fabrication]

```jsonl
"core": "One-sentence summary of the key points of this "level": "Primary People's Court | Intermediate People's
provision or interpretation", Court | Higher People's Court | Supreme People's Court |
"source_type": "statute|interpretation|guiding_case| null",
regulation|meeting_minutes|textbook|answer_key"} "region": "e.g. 'Court of the defendant's domicile' | '
] Court of contract performance place'",
(List all authoritative sources explicitly stated in the "special_jurisdiction": "Explanation of exclusive
ground truth; use [] if none. jurisdiction | null"
**Do not add statutes, cases or interpretations absent from }
the ground truth**.)
[L1.3.2 Period Error]
[L1.1.2 Citation-Content Misapplication] schema:
schema: "required_periods": [
" d l i " [ {"name": "Limitation of action", "value": "3 years"},
{"ref": "Civil Code Article 169", "summary": "Core rule of {"name": "Appeal period", "value": "15 days (civil cases)"}
this article (<=40 characters)"}
] ]
"common_misapplications_to_avoid": ["..."] # Similar
provisions likely to be misquoted by the model (can be [L1.3.3 Procedural-Step Error]
empty) schema:
"required_steps": ["Mediation -> Case filing -> Evidence
[L1.1.3 Hierarchy Error] presentation -> Court hearing -> Judgment"]
schema: "common_step_mistakes_to_avoid": ["Preservation can only be
"hierarchy_constraints": ["Shall not apply Rule X against applied after litigation (Incorrect, pre-litigation
Rule Y", "This case applies departmental law / superior preservation is allowed)"]
law / ..."]
(Fill in only if the case involves conflicts of legal [L1.3.4 Procedural-Outcome Error]
hierarchy; set applicable=false otherwise) schema:
"expected_outcome": "Dismiss the case filing | Dismiss the
[L1.1.4 Granularity Error] claims | Voluntary withdrawal of lawsuit | Default
schema: judgment | null"
"exact_articles_required": [ "forbidden_outcomes": ["Dismiss the claims while the correct
{"ref": "Civil Code Article 169 Paragraph 1", " ruling is dismissing the case filing"]
exact_section": "Paragraph 1" | "Item (2)" | "Whole
Article"} [L1.3.5 Appeal Error]
] schema:
(List provisions specified down to paragraphs, items or sub- "correct_appeal_path": "e.g. 'File an appeal to the
items in the ground truth; use [] for provisions only Intermediate People's Court for second instance' | 'Apply
marked at article level) for administrative reconsideration first before filing a
lawsuit'"
[L1.2.1 Conceptual Confusion]
schema: [L1.4.1 Fact Fabrication]
"core_concepts": ["Apparent Agency"] # schema:
Core legal concepts to be identified in this question "must_use_facts_from_question": ["Key Fact 1 from the
"common_confusions_to_avoid": ["Unauthorized Agency", question", "..."]
Entrusted Agency"] # Similar concepts easily confused by "facts_NOT_in_question_must_avoid": ["Any amount, time,
the model subject or conduct not stated in the question"]
[L1.2.2 Element Misstatement] [L1.4.2 Fact Omission]
schema: schema:
"must_have_elements": [ "critical facts must address": ["Key Fact 1 that determines
{"element": "Counterparty acts in good faith", "meaning": the conclusion"."..."]
"Explanation within 25 characters"} (These are factual anchors relied on by the ground truth;
] omission will definitely lead to wrong answers)
"elements_NOT_required": ["Counterparty is at fault"] #
Elements often wrongly added by the model [L1.4.3 Element-Fact Mismatch]
schema:
[L1.2.3 Exception Omission] "element_fact_pairs": [
schema: {"element": "Counterparty acts in good faith", "
"must_mention_exceptions": ["Apparent agency is not evidence_in_question": "The third party had no knowledge
established if the counterparty acts in bad faith", "..."] of Xiao Zhang's ultra vires act"},
{"element": "Appearance of authorization exists", "
[L1.2.4 Consequence Error] evidence_in_question": "The company's official seal was
schema: affixed"}
"expected_consequences": ["The principal shall bear the ]
legal effect of the agency act"]
"common_wrong_consequences": ["Order to return property [L1.4.4 Party Confusion]
instead of awarding compensation"] schema:
"parties_and_roles": [
[L1.2.5 Doctrinal Position Confusion] {"name": "Xiao Zhang", "role": "Unauthorized agent"},
schema: {"name": "Company", "role": "Principal"},
"required_position": "Prevailing Doctrine" | "Position {"name": "Third party", "role": "Counterparty"}
stated in the Supreme People's Court Interpretation XX" | ]
null
"competing_positions_to_avoid": ["Minority Doctrine X"] [Top-level Output Structure]
[L1.2.6 Discretionary Judgement Error] "difficulty": "low | medium | high",
schema: "summary_anchors": "Core checkpoints summary (<=80 characters,
"discretionary_anchors": [ remind judges of key evaluation points)",
{"factor": "Circumstances are serious", " "L1_1_authority": { L1_1_1_*, L1_1_2_*, L1_1_3_*, L1_1_4_*
reasonable_range_or_anchor": "Description within 30 }, /* 4 subclasses in sequence */
characters"} "L1_2_doctrine": { L1_2_1_*, L1_2_2_*, L1_2_3_*, L1_2_4_
] *, L1_2_5_*, L1_2_6_* },
(Fill in only if the question obviously involves judicial "L1_3_procedural_law": { L1_3_1_*, L1_3_2_*, L1_3_3_*, L1_3_4_
discretion) *, L1_3_5_* },
"L1_4_application": { L1_4_1_*, L1_4_2_*, L1_4_3_*, L1_4_4_*
[L1.3.1 Jurisdiction Error]
schema: "external_refs": [ {"ref": "...", "type": "statute|
"correct_jurisdiction": { interpretation|..."} ]
}
```

OUTPUT:   
{   
"difficulty": "medium",   
"summary\_anchors": "Core point: Determination of apparent   
agency. Analyze three elements including apparent   
authorization, bona fide counterparty and reasonable   
reliance per Civil Code Article 172; conclude the contract   
is valid.",   
"L1\_1\_authority": {   
"L1\_1\_1\_source\_fabrication": {   
"applicable": true,   
"must\_cite\_sources": [   
{"ref": "Civil Code Article 172", "core": "Rules for   
determination and legal effect of apparent agency",   
source\_type": "statute"}   
]   
},   
"L1\_1\_2\_citation\_content\_misapplication": {   
"applicable": true,   
"expected\_law\_summaries": [   
{"ref": "Civil Code Article 172", "summary": "   
Unauthorized agency is valid if the counterparty   
reasonably believes agency authority exists"}   
],   
"common\_misapplications\_to\_avoid": ["Civil Code Article   
171 (ordinary unauthorized agency)"]   
},   
"L1\_1\_3\_hierarchy\_error": {   
"applicable": false,   
"reason": "This case only applies the Civil Code with no   
conflicts of legal hierarchy."   
},   
"L1\_1\_4\_granularity\_error": {   
"applicable": true,   
"exact\_articles\_required": [   
{"ref": "Civil Code Article 172", "exact\_section": "   
Whole Article"}   
]   
}   
},   
"L1\_2\_doctrine": {   
"L1\_2\_1\_conceptual\_confusion": {   
"applicable": true,   
"core\_concepts": ["Apparent Agency"],   
"common\_confusions\_to\_avoid": ["Ordinary Unauthorized   
Agency (Civil Code Article 171)", "Entrusted Agency",

Functional Agency"]   
"L1\_2\_2\_element\_misstatement": {   
"applicable": true,   
"must\_have\_elements": [   
{"element": "Existence of apparent authorization", "   
meaning": "Objective appearance indicating valid agency   
authority"},   
{"element": "Counterparty acts in good faith", "meaning":   
"Counterparty is unaware of lack of agency authority"},   
{"element": "Reasonable reliance", "meaning": "Reliance   
formed after the counterparty exercises due care"}   
],   
"elements\_NOT\_required": ["Counterparty is at fault", "   
Principal is at fault"]   
},   
"L1\_2\_3\_exception\_omission": {   
"applicable": true,   
"must\_mention\_exceptions": ["Apparent agency is not   
established if the counterparty acts in bad faith"]   
},   
"L1\_2\_4\_consequence\_error": {   
"applicable": true,   
"expected\_consequences": ["The agency act binds the   
principal", "The principal may recover losses from the   
unauthorized agent after performance"],   
"common\_wrong\_consequences": ["The contract is invalid", "   
The contract is pending validity", "The unauthorized agent   
bears liabilities alone"]   
"L1\_2\_5\_doctrinal\_position\_confusion": {   
"applicable": false,   
"reason": "Apparent agency follows the prevailing doctrine   
with no competing theories to distinguish."   
},   
"L1\_2\_6\_discretionary\_judgement\_error": {   
"applicable": true,   
"discretionary\_anchors": [   
{"factor": "Reasonable reliance",   
reasonable\_range\_or\_anchor": "The client verified identity   
and fulfilled the duty of reasonable care"}   
}   
"L1\_3\_procedural\_law": {   
"L1\_3\_1\_jurisdiction\_error": {"applicable": false, "reason":   
"This is a substantive law consultation involving no   
jurisdiction issues."},   
"L1\_3\_2\_period\_error": {"applicable": false, "reason":   
"This case involves no limitation periods or time limits   
."},   
"L1\_3\_3\_procedural\_step\_error": {"applicable": false, "   
reason": "This case involves no litigation procedures."},   
"L1\_3\_4\_procedural\_outcome\_error": {"applicable": false,   
reason": "This case involves no procedural rulings."},   
"L1\_3\_5\_appeal\_error": " li bl " f l " "   
"This case involves no remedy or appeal procedures."}   
},   
"L1\_4\_application": {   
"L1\_4\_1\_fact\_fabrication": {   
"applicable": true,   
"must\_use\_facts\_from\_question": [   
"The company forbids Xiao Zhang from signing contracts   
exceeding 500,000 yuan",   
"Xiao Zhang signed a 1 million-yuan contract in the   
company's name",   
"The company's official seal was affixed on the contract   
"The client verified Xiao Zhang's identity with the   
company's finance department",   
"The finance department confirmed Xiao Zhang's position   
but not the authorization scope"   
],   
"facts\_NOT\_in\_question\_must\_avoid": ["Any amount, time,   
subject or conduct not stated in the question"]   
},   
"L1\_4\_2\_fact\_omission": {   
"applicable": true,   
"critical\_facts\_must\_address": [   
"The company's official seal was affixed (key evidence   
for apparent authorization)",   
"The client verified identity with the finance   
department (key evidence for reasonable reliance)"   
]   
},   
"L1\_4\_3\_element\_fact\_mismatch": {   
"applicable": true,   
"element\_fact\_pairs": [

```jsonl
{"element": "Existence of apparent authorization",
evidence_in_question": "The company's official seal was
affixed on the contract"},
{"element": "Counterparty acts in good faith", "
evidence_in_question": "The client verified identity and
received no denial"},
{"element": "Reasonable reliance", "evidence_in_question
": "The client fulfilled the duty of inquiry and due care"}
]
},
"L1_4_4_party_confusion": {
"applicable": true,
"parties_and_roles": [
{"name": "Xiao Zhang", "role": "Unauthorized agent /
Company salesperson"},
{"name": "Company", "role": "Principal"},
{"name": "Client", "role": "Counterparty"}
]
}
},
"external_refs": [
{"ref": "Civil Code Article 172", "type": "statute"}
]
}
--- Per-item template tail ---
[Question]
<<QUESTION>>
[ground_truth]
<<GROUND_TRUTH>>
[Metadata]
- Legal Field: <<LEGAL_CATEGORY>>
- Task Form: <<TASK_FORM>>
- Task Type: <<LEGAL_TASK_TYPE>>
Output a pure JSON object covering all 4 top-level categories
and 19 subclasses as required:
```

## H.4 Judge Prompt

This section presents the seven judge prompts used in our evaluation pipeline, each targeting one Layer-1 or Layer-2 hallucination group: L1.1 Authority, L1.2 Doctrine, L1.3 Procedural, L1.4 Application, L2.1 Planning & Reasoning, L2.2 Memory, and L2.3 Tool & Observation.

## Judge Prompt: L1.1 Authority

You are an expert in evaluating legal hallucinations in China.   
- You only judge whether the evaluated rollout makes mistakes in   
this subclass.   
- Judgments shall be based strictly on the ground truth (gt) and   
rubric specified in the checklist; do not add content   
beyond the rubric.   
- If the rubric marks \`applicable=false\`, this subclass is not   
applicable. Directly set \`hit=false\` and leave the   
evidence blank.   
- Do not add extra explanatory content; your only task is to   
identify errors.   
- Output a single standard JSON object only, with no extra text   
outside markdown code blocks.   
[Group] L1.1 Authority (Source/Authority Hallucination) (   
L1\_1\_authority)   
[Subclass Guidance]   
Key Judgment Points:   
- Prioritize verification against must\_cite\_sources and   
expected\_law\_summaries listed in the rubric.   
- L1.1.1 Source Fabrication: The model cites non-existent laws,   
judicial interpretations or case numbers outside the   
rubric list. Fill cited\_source with the original text from   
the model.   
- L1.1.2 Citation-Content Misapplication: The cited source is   
authentic but (1) inapplicable to the case (2) described   
content inconsistent with the original text (3) wrong   
article number.

- L1.1.3 Hierarchy Error: Applying lower-level laws against   
higher-level laws, or classifying administrative   
regulations as laws.   
- L1.1.4 Granularity Error: Correct law name and article number,   
yet incorrect paragraph, item or sub-item.   
Mutually Exclusive Classification (One error corresponds to only   
one category, follow priority below):   
Non-existent source -> L1.1.1   
Authentic source used incorrectly (inapplicable / inconsistent   
content / wrong article number) -> L1.1.2   
Correct law name and article number, only wrong paragraph/item/   
sub-item -> L1.1.4   
Note: Wrong article number (e.g. citing Article 35 instead of   
Article 36) belongs to L1.1.2 rather than L1.1.4.   
Counterexamples (all marked hit=false):   
Incomplete coverage: The rollout cites part of   
must\_cite\_sources with fully correct content and   
conclusions consistent with ground truth.   
Wrong final answer but correct cited text: Do not mark L1.1 as   
hit merely because the final answer is incorrect.   
[User prompt template]   
## Group: <<GROUP>>   
## Framework: <<FRAMEWORK>> Dataset: <<DATASET>>   
## Trajectory summary   
<<TRAJECTORY SUMMARY>>   
## Question   
<<QUESTION>>   
## Ground truth (gt)   
<<GROUND\_TRUTH>>   
## Checklist (rubric -- sliced to this group)   
<<RUBRIC\_JSON>>   
## Rollout trajectory (rendered, step-marked)   
<<TRAJECTORY\_MD>>   
## Rollout final answer   
<<FINAL ANSWER>>   
## Subclass definitions   
<<GUIDANCE>>   
\*\*Required output JSON schema\*\*   
"L1\_1\_1\_source\_fabrication": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original content and state error in one   
sentence>",   
"cited\_source": "<Original text of non-existent source;   
empty string if hit=false>"   
},   
"L1\_1\_2\_citation\_content\_misapplication": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original content and state error in one   
sentence>",   
"misapplied\_ref": "<Original text of misapplied source>"   
"L1\_1\_3\_hierarchy\_error": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original content and state error in one   
sentence>".   
"hierarchy\_issue": "<Specific description of hierarchy or   
validity error>"   
},   
"L1\_1\_4\_granularity\_error": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original content and state error in one   
sentence>",   
"granularity\_detail": "<Specific description of wrong   
paragraph/item/sub-item>"   
}   
Strictly return a single JSON object complying with the above   
schema. All subclasses must be retained.

## Judge Prompt: L1.2 Doctrine

\- L1.2.6 Discretionary Judgement Error: Misjudgment on the boundary of indeterminate legal concepts (e.g., "seriously improper", "obviously unfair").

[User prompt template]   
## Group: <<GROUP>>   
## Framework: <<FRAMEWORK>> Dataset: <<DATASET>>   
## Trajectory summary   
<<TRAJECTORY\_SUMMARY>>   
## Question   
<<QUESTION>>   
## Ground truth (gt)   
<<GROUND\_TRUTH>>   
## Checklist (rubric -- sliced to this group)   
<<RUBRIC\_JSON>>   
## Rollout trajectory (rendered, step-marked)   
<<TRAJECTORY\_MD>>   
## Rollout final answer   
<<FINAL\_ANSWER>>   
## Subclass definitions   
<<GUIDANCE>>   
Required output JSON schema   
{   
"L1\_2\_1\_conceptual\_confusion": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"confused\_pair": "<Specify the mistaken substitution between   
Concept A and Concept B>"   
},   
"L1\_2\_2\_element\_misstatement": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"wrong\_element": "<Specify the misstated constituent element   
>"

},   
"L1\_2\_3\_exception\_omission": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"missing\_exception": "<Specify the omitted exception or   
defense>"   
},   
"L1\_2\_4\_consequence\_error": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"wrong\_consequence": "<Specify the incorrect legal   
consequence or liability form>"   
},   
"L1\_2\_5\_doctrinal\_position\_confusion": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"doctrinal\_issue": "<Specify where positions shift or   
conflicting theories are mixed>"   
},   
"L1\_2\_6\_discretionary\_judgement\_error": {   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"discretion\_issue": "<Specify the misjudgment on   
discretionary boundaries>"   
}   
}   
Strictly return a single JSON object complying with the above   
schema. All subcategories must be retained.

## Judge Prompt: L1.3 Procedural-Law

\- If the rubric marks \`applicable=false\`, the subcategory is not applicable. Directly set \`hit=false\` and leave the evidence blank.

\- Do not add redundant explanations; only identify existing errors.

\- L1.3.4 Procedural-Outcome: Common civil procedure errors in China. Distinguish "dismissal of suit" (procedural issue) and "dismissal of claims" (substantive issue) and mark separately.

\- L1.3.5 Appeal: Errors in remedy approaches, such as administrative reconsideration prior to litigation and special remedies against arbitration awards.

```markdown
<<TRAJECTORY_SUMMARY>>
## Question
<<QUESTION>>
## Ground truth (gt)
<<GROUND TRUTH>>
## Checklist (rubric -- sliced to this group)
<<RUBRIC_JSON>>
## Rollout trajectory (rendered, step-marked)
<<TRAJECTORY_MD>>
## Rollout final answer
<<FINAL_ANSWER>>
## Subclass definitions
<<GUIDANCE>>
\textbf{Required output JSON schema}
\{
"L1\_3\_1\_jurisdiction\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"jurisdiction\_issue": "<Specify the specific jurisdiction
error>"
\},
"L1\_3\_2\_period\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"period\_issue": "<Specify the wrong time limit or
limitation period>"
\},
"L1\_3\_3\_procedural\_step\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"step\_issue": "<Specify the specific procedural step error
"
\},
"L1\_3\_4\_procedural\_outcome\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"outcome\_issue": "<Specify the procedural conclusion error,
e.g. dismissing claims instead of dismissing suit>"
\},
"L1\_3\_5\_appeal\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"appeal\_issue": "<Specify the specific remedy approach
error>"
\}
\}
Strictly return a single JSON object complying with the above
schema. All subcategories must be retained.
```

## Judge Prompt: L1.4 Application & Subsumption

You are an expert evaluator for legal hallucinations of Chinese   
laws (LLM-as-a-Judge).   
- You only assess whether the evaluated rollout makes errors in   
corresponding subcategories.   
- Judgments shall be strictly based on the ground truth (gt) and   
rubric in the checklist; do not add extra content beyond   
the rubric.   
- If the rubric marks \`applicable=false\`, the subcategory is not   
applicable. Directly set \`hit=false\` and leave the   
evidence blank.   
- Do not add redundant explanations; only identify existing

errors.   
Output solely a standard JSON object, with no extra text   
outside the JSON.   
[Group] L1.4 Application \& Subsumption (Fact Subsumption   
Hallucination) (L1\_4\_application)   
[Subclass guidance]   
Key Judgment Points:   
- L1.4.1 Fact Fabrication: The model states facts, amounts, time,   
parties or conducts that do not appear in the original   
question.   
- L1.4.2 Fact Omission: Key facts from the original question are   
omitted.   
- L1.4.3 Element-Fact Mismatch: Legal elements are correctly   
stated, but wrongly matched with case facts. Distinguish   
from L1.2.2, which refers to incorrect legal elements   
themselves.   
- L1.4.4 Party Confusion: Mix up Party A and Party B, or confuse   
agents with the principal parties.   
[User prompt template]   
## Group: <<GROUP>>   
## Framework: <<FRAMEWORK>> Dataset: ≤≤DATASET>>   
## Trajectory summary   
<<TRAJECTORY\_SUMMARY>>   
## Question   
<<QUESTION>>   
## Ground truth (gt)   
≤<<GROUND TRUTH>>   
## Checklist (rubric -- sliced to this group)   
<<RUBRIC\_JSON>>   
## Rollout trajectory (rendered, step-marked)   
<<TRAJECTORY\_MD>>   
## Rollout final answer   
<<FINAL\_ANSWER>>   
## Subclass definitions   
<<GUIDANCE>>   
\textbf{Required output JSON schema}   
\{   
"L1\_4\_1\_fact\_fabrication": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"fabricated\_fact": "<Specify the fabricated fact, amount,   
time, subject or conduct>"   
\},   
"L1\_4\_2\_fact\_omission": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"omitted\_fact": "<Specify the omitted or misplaced key fact   
"   
\},   
"L1\_4\_3\_element\_fact\_mismatch": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"mismatch\_detail": "<Specify the details of mismatch   
between legal elements and facts>"   
\},   
"L1\_4\_4\_party\_confusion": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"party\_issue": "<Specify the details of confusion among   
parties>"   
\}   
\}   
Strictly return a single JSON object complying with the above   
schema. All subcategories must be retained.

## Judge Prompt: L2.1 Planning & Reasoning

You are an expert evaluator for legal hallucinations of Chinese laws (LLM-as-a-Judge).

\- You only assess whether the evaluated rollout makes errors in corresponding subcategories.

\- Judgments shall be strictly based on the ground truth (gt) and rubric in the checklist; do not add extra content beyond the rubric.

\- If the rubric marks \`applicable=false\`, the subcategory is not applicable. Directly set \`hit=false\` and leave the evidence blank.

(a) No genuine examination of other options, counterexamples or opposing arguments after the step where the final answer is given;

(b) The thinking in that step fails to analyze all candidate options equally and merely rationalizes the predetermined answer.

\- Hypothesize-then-verify: A tentative answer is put forward and verified via tool calls or subsequent steps.

\- Intermediate hypothesis in multi-step reasoning with continuous analysis in later steps.

\- L2.1.2 Syllogism: Broken or reversed logical chain of major premise (legal provision) -> minor premise (case facts) -> conclusion.

\- L2.1.3 Self-Contradiction: Conflicting conclusions or element selections within a single output or across steps; inconsistent legal citations in different steps.

\- L2.1.4 Step Skip / Conflation: Drawing a direct conclusion without step-by-step reasoning; jumping from case facts to judgment without corresponding legal provisions.

\- L2.1.5 Out-of-Context Quoting: Citing partial content of legal provisions while ignoring preconditions and restrictive clauses.

[User prompt template]

\#\# Framework: <<FRAMEWORK>> Dataset: <<DATASET>>

<<TRAJECTORY\_SUMMARY>>

\#\# Question

\#\# Checklist (rubric - sliced to this group) <<RUBRIC\_JSON>>

\#\# Rollout trajectory (rendered, step-marked)

```latex
\textbf{Required output JSON schema}
\{
"L2\_1\_1\_premature\_closure": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
"origin\_step": <int|null>,
"evidence": "<Quote original text and state the error in one
sentence>",
"closure\_step": "<Step number where the conclusion is
finalized>"
\},
"L2\_1\_2\_syllogism\_error": \{
"hit": <true|false>,
"step\_indices": [<int>, ...],
```

"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"syllogism\_issue": "<Specify the faulty part among major   
premise, minor premise and conclusion>"   
\},   
"L2\_1\_3\_self\_contradiction": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"contradiction\_pair": "<Two conflicting excerpts from the   
text>"   
\},   
"L2\_1\_4\_step\_skip\_or\_conflation": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"skip\_issue": "<Specify the skipped or merged key reasoning   
steps>"   
\},   
"L2\_1\_5\_out\_of\_context\_quoting": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"oo\_context\_quote": "<The quoted excerpt taken out of   
context>"   
\}   
\}   
Strictly return a single JSON object complying with the above   
schema. All subcategories must be retained.

## Judge Prompt: L2.2 Memory

You are an expert evaluator for legal hallucinations of Chinese laws (LLM-as-a-Judge).

\- You only assess whether the evaluated rollout makes errors in corresponding subcategories.

\- Judgments shall be strictly based on the ground truth (gt) and rubric in the checklist; do not add extra content beyond the rubric.

\- If the rubric marks \`applicable=false\`, the subcategory is not applicable. Directly set \`hit=false\` and leave the evidence blank.

\- Do not add redundant explanations; only identify existing

\- Output solely a standard JSON object, with no extra text outside the JSON.

[Group] L2.2 Memory (Memory Hallucination) (L2\_2\_memory)

Key Judgment Points: L2.2 only covers errors related to memory loss or context forgetting, not all wrong answers.

Trigger conditions (satisfy any one):

\- Restating case facts in later steps inconsistently with the original question, including wrong amounts, names, time or actors.

\- Answering a different question than the user raised in the final answer (task drift).

\- Losing key qualifiers when paraphrasing the question in tool call queries.

\- Contradictory restatement of previously confirmed facts in multi-step reasoning.

\- Wrong judgment on legal compliance of options -> Classified as L1.4 or L1.2.

\- Misstating constituent elements of legal provisions -> Classified as L1.2.2.

\- Incorrect final answer with accurate restatement of facts -> Reasoning or judgment error rather than memory error.

\- Single thinking plus final answer structure (max\_step <= 2 and n\_events <= 3) generally does not trigger L2.2, unless obvious factual data conflicts with the original question.

\## Framework: <<FRAMEWORK>> Dataset: <<DATASET>>

<<TRAJECTORY\_SUMMARY>>   
## Question   
<<QUESTION>>   
## Ground truth (gt)   
<<GROUND\_TRUTH>>   
## Checklist (rubric -- sliced to this group)   
<<RUBRIC\_JSON>>   
## Rollout trajectory (rendered, step-marked)   
<<TRAJECTORY\_MD>>   
## Rollout final answer   
<<FINAL\_ANSWER>>   
## Subclass definitions   
<<GUIDANCE>>   
\textbf{Required output JSON schema}   
\{   
"L2\_2\_memory": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"forgotten\_premise": "<Forgotten or misrecorded premise,   
name, amount or time>"   
\}   
\}   
Strictly return a single JSON object complying with the above   
schema. All subcategories must be retained.

## Judge Prompt: L2.3 Tool & Observation

You are an expert evaluator for legal hallucinations of Chinese laws (LLM-as-a-Judge).

\- If the rubric marks \`applicable=false\`, the subcategory is not applicable. Directly set \`hit=false\` and leave the evidence blank.

\- Do not add redundant explanations; only identify existing errors.

\- Output solely a standard JSON object, with no extra text outside the JSON.

## [Group] L2.3 Tool \& Observation (Tool & Observation Hallucination) (L2\_3\_tool\_observation)

Judgments must comply with the following framework tool specifications without arbitrary interpretation.

\- Wrong tool selection: Violating tool selection rules, e.g. using law\_retrieval instead of law\_check with known article numbers, or retrying law\_retrieval when web\ \_search should be adopted.

\- Invalid parameters: Irrelevant query, incorrect parameter type or value, wrong law name.

\- Wrong calling order: Invoking tools exclusive to deepanalysis phase in earlier stages (only for lawthinker).

Excessive or missing calls: Repeated calls with identical tool and parameters, or failing to switch tools/terminate after >=3 unsuccessful attempts with the same query. - \`MALFORMED\` tagged tool call indicates JSON parsing failure, which shall be marked as hit=true.

\- Only adopting the first retrieved legal provision out of multiple returned items.

\- Misreading figures such as confusing the year 2018 with 2008 in case records.

\- Drawing conclusions based on full text while only reading abstracts.

\- Irrelevant provisions returned by law\_retrieval are inherent noise. Do not mark error unless irrelevant provisions are used for reasoning.

\- This group is only applicable when there is at least one tool (2) Query writing: Focus on crime/legal system names and key elements instead of full questions. topk is recommended 3-5.

call event.   
[User prompt template]   
\#\# Group: <<GROUP>>   
\#\# Framework: <<FRAMEWORK>> Dataset: <<DATASET>>   
\#\# Trajectory summary   
<<TRAJECTORY\_SUMMARY>>   
\#\# Question   
<<QUESTION>>   
\#\# Ground truth (gt)   
<<GROUND\_TRUTH>>   
\#\# Checklist (rubric - sliced to this group)   
<<RUBRIC\_JSON>>   
\#\# Rollout trajectory (rendered, step-marked)   
<<TRAJECTORY\_MD>>   
\#\# Rollout final answer   
<<FINAL\_ANSWER>>   
\#\# Subclass definitions   
<<GUIDANCE>>

(3) Retry rule: Rewrite keywords once for poor results. Switch to web\_search or make conclusion directly if still failed. Repeated retries with identical query are forbidden.

(4) Coverage: Current effective statutes, judicial interpretations and departmental rules of Chinese mainland. Low coverage for local regulations, newly released policies and foreign laws.

\- law\_recommendation: Similar provision recommendation. Returns related provisions given a legal citation. Parameters: \{"law": "str"\}

\- charge\_expansion: Related charge expansion. Returns similar charges given a list of charges. Parameters: \{"charges": "List[str]"\}

\- case\_retrieval: Similar case retrieval. Returns analogous cases given case type and information. Parameters: \{"type ": "str(Civil Case|Criminal Case)", "query": "str"\}

\- template\_retrieval: Document template retrieval. Gets templates for specified legal documents. Parameters: \{" template\_type": "str"\}

\- plan\_generation: Writing plan generation. Creates outlines for designated documents. Parameters: \{"document\_type": "str"\}

\- procedure\_retrieval: Court procedure retrieval. Only for moot court scenarios. Parameters: \{"court\_type": "str(Civil Court|Criminal Court)", "stage": "int(0-4|0-2)"\}

\- law\_check: Legal provision verification. Returns full text given exact legal citation. Parameters: \{"law\_name": str"\}

Usage rule: Prioritize law\_check when accurate article numbers are available for higher precision.

\- fact\_law\_relevance\_check: Fact-provision relevance verification. Checks applicability of a given provision to case facts. Parameters: \{"fact": "str", "law": "str"\}

crime\_law\_consistency\_check: Charge-provision matching verification. Validates correspondence between charges and criminal law articles. Parameters: \{"crime": "str", "law ": "str"\}

\- document\_format\_check: Document format inspection. Checks format of complete legal documents. Parameters: \{" document\_type": "str", "document": "str"\}

\- law\_query\_rewrite: Query rewriting. Optimizes retrieval queries combining case background. Parameters: \{"query": "str", "context": "str"\}

\- procedure\_check: Procedure inspection. Verifies completeness of court proceedings. Parameters: \{"court\_type": "str"\}

\- web\_search: Web search. Retrieves legal information from public network.

(1) No valid results after one keyword rewrite with law\ \_retrieval / law\_check;

(2) Inquiring about latest policies, local regulations, case

details or foreign laws;   
(3) Consulting practical procedures, required materials and   
service channels.   
Discipline: Consolidate results and give final answer once   
relevant summaries are obtained. Repeated calls with   
identical query are prohibited.   
\textbf{Required output JSON schema}   
\{   
"L2\_3\_1\_tool\_call\_error": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"tool\_issue": "<Specify error type: wrong tool / invalid   
parameter / excessive calls etc.>"   
\},   
"L2\_3\_2\_observation\_misuse": \{   
"hit": <true|false>,   
"step\_indices": [<int>, ...],   
"origin\_step": <int|null>,   
"evidence": "<Quote original text and state the error in one   
sentence>",   
"misuse\_step": "<Step number where misinterpretation occurs   
>″   
\}   
\}   
Strictly return a single JSON object complying with the above   
schema. All subcategories must be retained.