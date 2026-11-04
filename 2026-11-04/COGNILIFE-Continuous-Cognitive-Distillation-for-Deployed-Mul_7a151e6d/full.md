# COGNILIFE: Continuous Cognitive Distillation for Deployed Multi-Expert Agent Marketplaces

Xuwei Tan, Jingchun Huang, Ming Cai, Shuangquan Lyu, Tiange Wang, Yinzhi Wang, Ronghua Shi, Jingwei Song, Yang Jingsong, Yinghui Xia, Hehua Fan, Yifan Zhou, Hanxin Chen, JIWEI SONG, TIANYU SHI

## Abstract

Multi-expert agent marketplaces—where creators publish domain-specific AI experts for end users to consult—are an emerging deployment pattern for language technology. A published expert can receive many requests yet little evidence about its creator, who remains active elsewhere: the cold-expert problem. We frame continuous cognitive distillation as converting attributed creator observations into revisable decision rules, communication patterns, and domain procedures. COG-NILIFE proposes a cross-expert attribution design for this problem, with Creator Profiles for transferable creator patterns and Expert Journals for listing-specific knowledge; this design is not yet implemented or evaluated. A deployed platform partially realizes the surrounding lifecycle through creator-authorized authoring, capability-owned proposal filtering, and reversible consolidation of eligible owner-scoped memory. On PROFYBENCH-V1, an 85-task offline benchmark built from replay seeds and controlled internal tests, the Profy Full configuration as a whole achieves a mean overall score of 4.65/5 from the primary LLM judge, compared with 4.01 for Raw LLM and 4.28 for Expert Prompt, with the largest gains in completeness, specificity, and missing-information tasks. These configuration-level scores do not test coldexpert mitigation, cross-expert transfer, or longitudinal distillation.

## 1 Introduction

Multi-expert agent marketplaces are emerging on AI platforms that enable creators to publish and monetize their own agents.<sup>1</sup> Published experts should preserve their creators’ behavioral commitments while serving different callers. Personaconditioned dialogue (Zhang et al., 2018) uses textual profiles, while personalization work models heterogeneous user preferences (Zollo et al., 2025) or conditions outputs on user profiles (Salemi et al., 2024). Maintaining these services requires deciding whose behavior should refine which representation.

Consider a creator who publishes a translation expert but mostly consults coding and writing experts. The target can receive many requests yet remain sparse in relevant creator evidence: the coldexpert problem. Repeated creator corrections of ambiguity may transfer to translation, while coding-specific instructions stay local and other callers’ preferences must not be mistaken for creator traits.

We define continuous cognitive distillation as repeatedly abstracting attributed creator observations into persistent, revisable decision rules, communication patterns, and domain procedures. Our framework organizes evidence around the represented creator and scopes guidance to each expert through Seeding, Immersion, and Crystallization.

Figure 1 separates implemented platform controls from proposed cross-expert routing; Table 1 maps deployment challenges to design responses.

Our contributions are threefold:

• Cold-expert formulation. We define sparse creator evidence despite activity elsewhere and distinguish the represented creator, source interaction, target expert, and caller; mitigation remains untested.

• Creator-controlled maintenance. We describe implemented capability-owned proposal filtering, eligible reversible owner-scoped memory maintenance, and staged revisions of existing skills; the cross-expert attribution protocol remains proposed and unevaluated.

• Expert-service evaluation. We introduce PROFYBENCH-V1, comparing a Full configuration of supplied expert context and workflows with two baselines on 85 offline tasks, with the largest judged gains in completeness and specificity. We additionally report a supplementary 52-task customs-extraction benchmark with three runs per task, accompanied by an auditable example specifying source documents, structured outputs, and a frozen field-scoring contract.

<table><tr><td>Challenge (status)</td><td>COGNILIFE design response</td></tr><tr><td>Tacit expertise (partial)</td><td>Elicit commitments, then derive re- visable guidance from attributed evi-</td></tr><tr><td>Cold experts (partial)</td><td>dence. Seed sparse listings; propose reuse of relevant creator evidence.</td></tr><tr><td>Evidence attribution</td><td>Resolve source, applicability, and per- mitted reuse before routing claims.</td></tr><tr><td>(proposed) Persistent refinement</td><td>Consolidate scoped memory; evaluate and version skill updates.</td></tr><tr><td>(partial) Creator control (implemented)</td><td>Keep authored commitments separate from learned candidates and releases.</td></tr></table>

Table 1: Deployment challenges motivating COG-NILIFE and implementation status (partial: some steps implemented, others proposed or incomplete). PROFYBENCH-V1 tests the Full configuration, not individual components.

## 2 Related Work

Personas and creator authoring. PersonaChat conditions dialogue on textual profiles (Zhang et al., 2018); trainable characters and role-specific tuning extend these representations (Shao et al., 2023; Wang et al., 2024a). PersonalLLM benchmarks adaptation to heterogeneous simulated user preferences (Zollo et al., 2025), and LaMP personalizes outputs using user-profile records (Salemi et al., 2024). CloneBuilder supports creator configuration, material uploads, testing, and feedback (Lee et al., 2026). Knowledge elicitation spans multiple techniques (Cooke, 1994); our design combines creator input with interaction evidence. Tacit expertise may resist verbalization (Polanyi, 1966), and reported attitudes need not predict situated action (Jerolmack and Khan, 2014). Our lifecycle connects the represented creator’s authored commitments to revisable guidance from later interactions.

Continual improvement and memory. Lifelong dialogue research motivates continual adaptation (Liu and Mazumder, 2021); Agent-in-the-

Loop improves customer-support models through operational feedback (Zhao et al., 2025). ACE incrementally revises context playbooks, while Mem0 extracts and maintains conversational memory (Zhang et al., 2026; Chhikara et al., 2025). PersonaTree links interaction evidence, recurring patterns, and durable person-level claims through lifecycle memory operations (Hou et al., 2026). COGNILIFE adopts similar updating and evidence-abstraction operations while emphasizing ownership, intervention, and recovery controls.

Sharing and attribution. ShareMem shares reusable experience while grounding its use in receiving-user preferences, including settings where relevant local experience is scarce; AIM separates private and public memory with access controls (Hu et al., 2026; Johnson et al., 2026). Item cold start concerns unrated items (Schein et al., 2002). Our attribution target is the creator represented by a published expert, who may differ from both its caller and the source expert’s owner. Reuse therefore depends on evidence authorship, target applicability, and permission.

Evaluation. PersonaGym evaluates personaconsistent behavior, InCharacter probes personality fidelity, and Persistent Personas measures persona-fidelity degradation over extended interactions (Samuel et al., 2025; Wang et al., 2024b; Luz de Araujo et al., 2026). Tool-agent-user evaluation examines repeated-trial reliability (Yao et al., 2025). PROFYBENCH-V1 evaluates professional expert-service outputs using task checkpoints, controlled input variants, and error analysis. This configuration-level evaluation measures task utility; creator fidelity and cross-expert transfer are separate evaluation targets.

## 3 CogniLife Framework

## 3.1 Marketplace and Cold Experts

We distinguish creator $^ { c , }$ a source expert $e _ { s }$ they consult, a target expert e they own, and caller u who consults that target. The source expert may belong to someone else, and u need not be c. Evidence authored by c can inform how e represents its creator; u’s preferences concern adaptation to that caller. A new listing with rich creator grounding need not be cold; a busy listing may remain cold when other callers reveal little about its creator.

![](images/15fb181c8c7c366e1d23c40ce2a664acd080c1e15d259e5ec646ba300a85fe06.jpg)  
Figure 1: Profy marketplace workflow. Solid panels show implemented skill proposals, scoped memory, and creator-authorized publication; dashed purple paths show proposed routing to Creator Profiles and Expert Journals. The routing is neither implemented nor evaluated. Skill activation and rollback have separate eligibility checks (Section 3.4; Appendix F).

Let $\mathcal { O } _ { c }$ contain authorized creator-authored contributions and creator-confirmed commitments across the marketplace, and $\mathcal { O } _ { c , e } ~ \subseteq ~ \mathcal { O } _ { c }$ those already grounding expert $e ,$ including its initialization. Initialization grounds authored commitments; recurring observations support behavioral patterns. An expert is cold with respect to creator evidence when $\mathcal { O } _ { c , e }$ is sparse despite activity elsewhere. The selection target $\mathcal { T } _ { c  e } \subseteq \mathcal { O } _ { c } \backslash \mathcal { O } _ { c , e }$ contains relevant, authorized external evidence; activity alone does not ensure such evidence exists. Selection separates transferable creator patterns from local content and other agents’ behavior.

## 3.2 Continuous Distillation Lifecycle

Continuous cognitive distillation turns observable communication, stated reasoning, and procedural choices into revisable textual guidance. Seeding organizes creator input and research material into separately editable persona, Soul, instructions, and reusable skills. The authoring workflow prompts source review and scenario-based calibration by the creator. Immersion accumulates subsequent observations with authorship and interaction context. Crystallization abstracts candidate decision rules and procedures, preserving their evidence and applicability for revision. In this proposed loop, guidance shapes later interactions; subsequent evidence confirms, revises, or retires rules, making distillation cumulative and revisable.

The proposed Creator Profile holds transferable creator patterns; the Expert Journal holds listing-specific knowledge and procedures. Ephemeral context supports the current task without becoming a durable persona claim. The creator-controlled Soul expresses authored behavioral commitments. At invocation, applicable guidance would condition responses alongside these commitments and current context; procedural guidance can also take the form of reusable skills. Evidence maturity concerns supported applicability, resolved contradictions, and authorized scope.

## 3.3 Cross-Expert Attribution

The attribution design selects creator c’s authorized contributions made while consulting source expert $e _ { s }$ to refine their own target expert e. It jointly resolves evidence authorship, target applicability, and permitted reuse.

The proposed procedure isolates the creator’s contributions from assistant outputs, quotations, and others’ statements; abstracts candidate patterns; and checks recurrence across tasks or explicit endorsement, target relevance, and permitted reuse. Mixed observations become separate behavioral and domain-specific claims; contradictions await clarification.

Supported claims receive proposed creator-level $\mathcal { C } _ { c }$ or listing-specific $\mathcal { D } _ { e }$ scope:

$$
R _ { \mathrm { d e s i g n } } ( s ) = { \left\{ \begin{array} { l l } { \mathrm { P r o f l e } ( c ) , } & { s \in \mathcal { C } _ { c } , } \\ { \mathrm { J o u r n a l } ( e ) , } & { s \in \mathcal { D } _ { e } , } \\ { \varnothing , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{1}
$$

Each atomic claim receives one proposed storage label; its applicability may span multiple experts. Here ∅ retains contextual or pending claims. Creator-level reuse remains conditional on applicability.

As an illustrative design example, a creator tells a coding assistant: “When a requirement has two plausible interpretations, show both and ask before choosing; use our Python formatter for this repository.” The first clause yields a candidate ambiguity-handling rule; the formatter instruction remains local. If recurrent across tasks or confirmed for translation, the rule could guide the creator’s translation expert to present alternative readings before editing. A correction to that expert’s translation glossary belongs in its Journal.

## 3.4 Creator-Controlled Refinement

Figure 1 shows the deployed platform’s implemented foundation through three complementary control points.

Candidate formation. For existing skills (control point 01), the evolution-log path admits procedural suggestions only for failures classified as expert-capability gaps, excluding task-instruction and environment failures. Following SEA-Eval’s joint attention to success and efficiency (Jiang et al., 2026), our own default filter requires nondecreasing quality-group and strictly increasing efficiency-group aggregates; context footprint uses skill-text length. Most dimensions are model estimates, so the filter controls proposal admission without establishing execution gains. Accepted proposals enter personal or shared staging, separately from active inference bodies.

Scoped maintenance. Eligible dialogue undergoes LLM extraction and duplicate checks before caller–expert retention (control point 02). Optional temporal consolidation applies modelproposed, owner-scoped operations with evidence associations and histories. Facts retain validity intervals and supersession links; user corrections create revisions. Recovery reverses journaled operations transactionally after checking the recorded post-update state, aborting on intervening conflicts. These inspectable, reversible updates provide a maintenance substrate for Crystallization. Retrieval and extraction retain separate admission conditions.

Deployment control. Creator-authorized AI assistance produces editable persona, Soul, and instructions; publication releases the configuration and synchronizes authored skills into the active library (control point 03). Runtime adds caller context, the invoking user’s profile, permitted memory, tools, and active skills (Serve, top row). Promotion services check accepted status, paired nondegrading evaluation rewards, target agreement, and authority: shared-row changes require the expert creator; personal-row changes require their owner. The candidate-comparison-topromotion path remains incomplete. Background memory and skill updates do not modify stored Soul; creator-authorized interactive authoring can. Caller memory and account profiles remain distinct from the proposed cross-listing Creator Profile. Appendix F details versioning, recovery, and platform exceptions.

## 4 Evaluation and Results

We compare the judged offline task utility of supplied expert context and workflows with two baseline configurations. PROFYBENCH-V1 expresses expert-service requests as tasks with explicit checkpoints, input variants, and diagnostic error labels. Task-level scores and pairwise preferences characterize judged configuration-level utility; runtime metadata and production cases illustrate authoring and execution.

<table><tr><td>Family</td><td>Domain</td><td>N</td><td>Cmpl.</td><td>Miss.</td><td>Adv.</td></tr><tr><td>ADV</td><td>Evidence advising</td><td>20</td><td>10</td><td>6</td><td>4</td></tr><tr><td>EDU</td><td>Education</td><td>20</td><td>10</td><td>6</td><td>4</td></tr><tr><td>MKT</td><td>Marketing</td><td>20</td><td>10</td><td>6</td><td>4</td></tr><tr><td>MMW</td><td>Multimodal wf.</td><td>15</td><td>8</td><td>4</td><td>3</td></tr><tr><td>PER</td><td>Personalization</td><td>10</td><td>5</td><td>2</td><td>3</td></tr></table>

Table 2: Composition of PROFYBENCH-V1: 85 instances across five production families and three variants (Complete / Missing-information / Adversarial). Evaluation focus by family—ADV: evidence grounding, uncertainty calibration, safe boundaries; EDU: pedagogical structure and artifact completeness; MKT: platform fit and audience-specific rewriting; MMW: workflow decomposition and text–image consistency; PER: style fidelity and long-turn coherence.

## 4.1 ProfyBench and Protocol

PROFYBENCH-V1 contains 85 instances across five production task families (Table 2), built from privacy-filtered public replay seeds or controlled internal tests. Each instance records its input and context slots, family, variant, source category, risk level, checkpoints, and evaluation dimensions. Checkpoints specify expected response components. Complete, missing-information, and adversarial variants probe deliverable quality, omitted constraints, fragmented inputs, and unsafe overconfidence.

All 85 instances compare three conditions: Raw LLM receives the request without scaffolding; Expert Prompt adds a generic expert-role prompt and task framing; Profy Full includes expert metadata, workflow conventions, and supplied profile/journal context. Cross-listing attribution remains a separate, untested proposed mechanism. The primary o3-mini judge scores each output without condition labels, given its checkpoints and risk level, on completeness, specificity, actionability, coherence, and safety-boundary adherence from 1–5; overall core score averages these dimensions. Family-specific diagnostics and a fixed error vocabulary complement the shared rubric; pairwise comparisons randomize output order and omit checkpoints. Prompts appear in Appendix B.

## 4.2 Task Utility

Under the primary o3-mini judge (N=85), Profy Full reaches 4.65 overall core score, compared with 4.28 for Expert Prompt and 4.01 for Raw LLM (Table 3). The largest dimensional improvements concern coverage of task requirements and concrete tailoring: completeness rises from 3.61/4.01→4.62 and specificity from 3.38/3.62→4.26. Coherence and safety are nearceiling across conditions; completeness, specificity, and actionability account for most of the overall difference.

<table><tr><td>System</td><td>Ovr.</td><td>Comp.</td><td>Spec.</td><td>Act.</td><td>Coh.</td><td>Safe</td></tr><tr><td>Raw LLM</td><td>4.01</td><td>3.61</td><td>3.38</td><td>3.95</td><td>4.65</td><td>4.48</td></tr><tr><td>Expert Prompt</td><td>4.28</td><td>4.01</td><td>3.62</td><td>4.15</td><td>4.84</td><td>4.78</td></tr><tr><td>Profy Full</td><td>4.65</td><td>4.62</td><td>4.26</td><td>4.47</td><td>4.99</td><td>4.92</td></tr></table>

Table 3: PROFYBENCH-V1 automatic judge results (N=85, 1–5 scale, higher is better). Overall (Ovr.) is the average of completeness, specificity, actionability, coherence, and safety-boundary adherence.

Paired task-level comparisons (Appendix Table 4) show gains of +0.64 over Raw LLM and +0.37 over Expert Prompt, both significant at $\begin{array} { r l r } { p } & { { } < } & { 1 0 ^ { - 4 } } \end{array}$ (sign test). Profy Full ties or exceeds the baselines on 96.5% and 91.8% of tasks, respectively.

## 4.3 Task and Error Patterns

Gains over Expert Prompt appear across all five families (Appendix Figure 2(a,b)). Education (+0.46) and multimodal workflow (+0.45) show the largest gains; marketing has the highest absolute score (4.82). Missing-information tasks show the largest variant-level improvement, relevant to users who omit constraints, paste fragmented drafts, or distribute intent across turns. The small family-specific samples support interpreting these as subgroup patterns.

Fixed error tags make the score differences operationally interpretable (Appendix Figure 2(c)). The most frequent baseline failures are unhandled missing input and weak personalization. Profy Full has no missing-input failures in this benchmark and reduces weak-personalization failures from 16/11 under Raw LLM/Expert Prompt to 1. The tags locate the gains in handling incomplete requests and tailoring responses to task constraints.

## 4.4 LLM Preferences and Judge Audit

Pairwise judgments favor Profy Full in 53 of 85 comparisons against Raw LLM, versus 31 losses and 1 tie (62.4% win rate). This check covers only Raw LLM, and pairwise prompts omit checkpoints.

A 60-record subset (20 per condition) was rescored by four additional LLM judges. Because the panel aggregate score lacks a documented definition and the paired item-level scores are unavailable for verification, we do not use its summary or agreement statistics to support the main comparison. This subset does not provide human validation of correctness or safety.

## 4.5 Runtime and Deployment Observations

Elapsed time was recorded for all 85 Profy Full runs (median 52.05 s, mean 54.41 s). These measurements characterize Full alone; comparable baseline timing was not recorded.

Supplementary customs benchmark. We further evaluate structured deliverables on 52 customs-extraction tasks with three recorded runs per task. The mean recorded task score is 0.958 over 156 valid runs from one configuration, without a matched baseline; it covers a restricted field subset (389 of 934 fields in an audited task), not whole-declaration accuracy. A packaged example specifies source-reviewed objective fields and a frozen scorer with schema and item-count gates. This domain evaluation complements the LLMjudged PROFYBENCH-V1 results with repeatedrun measurements of a professional document workflow. Appendix G details the protocol, scoring scope, results, and limitations, and provides an anonymous link to three anonymized customsdocument example packets.

Three anonymized cases illustrate the evaluated task geometries (Appendix D): college advising exposes evidence and uncertainty requirements; storybook generation demonstrates a staged artifact workflow; and social-content optimization reveals an input-format failure when instructions and draft content arrive in separate turns. Appendix E additionally documents creator-authorized initial grounding through editable expert fields and a reusable skill. These artifacts make the authoring outputs and service workflows inspectable.

## 5 Discussion and Conclusion

COGNILIFE formulates sparse creator evidence in marketplace experts and proposes a cross-expert attribution design. The deployed platform includes creator-authorized authoring, capabilityowned proposals, eligible owner-scoped memory consolidation, and staged skill revisions. These controls do not establish that the proposed routing improves cold listings or that continuous distillation preserves creator fidelity over time; implementation and direct tests remain future work.

On PROFYBENCH-V1 (85 instances drawn from replay seeds and controlled internal tests, three conditions), the supplied expert context and workflow configuration receives higher automaticjudge task-utility scores than both baselines, especially completeness, specificity, and missinginformation handling. The central design principle is to distinguish the creator represented, the caller currently served, and the scope of each guidance claim.

## Limitations

PROFYBENCH-V1 evaluates the utility of the supplied Full configuration. Its 85 replayderived and internally constructed instances support configuration-level comparisons, with small family-specific samples and designed variants. Automated acquisition of creator style, component-level effects, controlled cold-listing improvement, cross-listing transfer, and longitudinal fidelity remain separate evaluation targets; the attribution extension also requires a dedicated implementation. The reported comparison uses an o3-mini judge, and pairwise preferences compare Full only with Raw LLM. The auxiliary four-model subset is not used to validate the results because its aggregate score definition and paired records are unavailable for verification. LLM judging remains susceptible to known biases (Zheng et al., 2023); correlated errors remain possible, and near-ceiling coherence and safety scores limit discrimination. Recorded runtime characterizes Full alone. The production cases illustrate workflows, while human evaluation, comparative cost and latency, and online outcomes remain further validation targets. Benchmark-schema, rubric, and aggregate-statistics release remain future work, subject to privacy constraints.

## Ethics Statement

Continuous distillation makes consent, evidence attribution, and creator agency central deployment concerns. The implementation separates ownership-scoped memory, creatorcontrolled publication, and accepted skill updates. Background evolution leaves Soul unchanged, while creator-authorized interactive distillation can edit authored fields. Deployment should disclose collected signal categories and provide informed participation, inspection, deletion, and withdrawal mechanisms. Extending attribution across listings additionally requires making the source and intended reuse of creator evidence clear, while protecting other participants’ information. Sensitive facts require explicit necessity and consent before durable retention. Person-representing experts require authorization and identity disclosure; high-stakes advice needs uncertainty and escalation boundaries. Evaluation artifacts and production illustrations should mask identifying content. These governance requirements complement the implemented scope and update controls.

## References

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready AI agents with scalable long-term memory. In ECAI 2025, volume 413 of Frontiers in Artificial Intelligence and Applications, pages 2993– 3000. IOS Press.

Nancy J Cooke. 1994. Varieties of knowledge elicitation techniques. International Journal of Human-Computer Studies, 41(6):801–849.

Yubo Hou, Jingwei Song, Hongbo Zhang, Zhisheng Chen, Bang Xiao, Tao Wan, and Zengchang Qin. 2026. PersonaTree: Structured lifecycle memory for person understanding in LLM agents. arXiv preprint arXiv:2606.04780.

Jinming Hu, Haodong Zhao, Qi Jia, Die Chen, Tianhang Zhao, Sufeng Duan, and Gongshen Liu. 2026. Learning from others, acting for you: Cross-user memory sharing for LLM agents. arXiv preprint arXiv:2609.32511.

Colin Jerolmack and Shamus Khan. 2014. Talk is cheap: Ethnography and the attitudinal fallacy. Sociological Methods & Research, 43(2):178–209.

Sihang Jiang, Lipeng Ma, Zhonghua Hong, Keyi Wang, Zhiyu Lu, Tengfei Wang, Shisong Chen, Jinghao Zhang, Tianjun Pan, Weijia Li, Jiaqing Liang, and Yanghua Xiao. 2026. SEA-Eval: A benchmark for evaluating self-evolving agents beyond episodic assessment. arXiv preprint arXiv:2604.08988. Version 3.

Zachary Johnson, Nigel Boachie Kumankumah, Somya Chatterjee, Tejas Sathyamurthi, Min Chen, Xinyi Alice Li, Xiao Wang, Emily Morgan Gelchie, Jessica Lin, Sadid A. Hasan, and Sulaiman Vesal. 2026. AIM: A privacy-aware interoperable memory framework for multi-agent multi-user LLM systems. arXiv preprint arXiv:2609.12320.

Donggun Lee, Suyoun Lee, Hyunseung Lim, and Hwajung Hong. 2026. Creating text-based AI clones of myself: Exploring perceptions, development strategies, and challenges. International Journal of Human-Computer Studies, 208:103692.

Bing Liu and Sahisnu Mazumder. 2021. Lifelong and continual learning dialogue systems: Learning during conversation. Proceedings of the AAAI Conference on Artificial Intelligence, 35(17):15058– 15063.

Pedro Henrique Luz de Araujo, Michael A. Hedderich, Ali Modarressi, Hinrich Schutze, and Benjamin¨ Roth. 2026. Persistent personas? role-playing, instruction following, and safety in extended interactions. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5329–5359.

Michael Polanyi. 1966. The Tacit Dimension. Doubleday, Garden City, NY.

Alireza Salemi, Sheshera Mysore, Michael Bendersky, and Hamed Zamani. 2024. LaMP: When large language models meet personalization. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7370–7392.

Vinay Samuel, Henry Peng Zou, Yue Zhou, Shreyas Chaudhari, Ashwin Kalyan, Tanmay Rajpurohit, Ameet Deshpande, Karthik R. Narasimhan, and Vishvak Murahari. 2025. PersonaGym: Evaluating persona agents and LLMs. In Findings of the Associationfor Computational Linguistics: EMNLP 2025, pages 6999–7022.

Andrew I. Schein, Alexandrin Popescul, Lyle H. Ungar, and David M. Pennock. 2002. Methods and metrics for cold-start recommendations. In Proceedings of the 25th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), pages 253–260.

Yunfan Shao, Linyang Li, Junqi Dai, and Xipeng Qiu. 2023. Character-LLM: A trainable agent for roleplaying. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13153–13187.

Noah Wang, Z.Y. Peng, Haoran Que, Jiaheng Liu, Wangchunshu Zhou, Yuhan Wu, Hongcheng Guo, Ruitong Gan, Zehao Ni, Jian Yang, Man Zhang, Zhaoxiang Zhang, Wanli Ouyang, Ke Xu, Wenhao Huang, Jie Fu, and Junran Peng. 2024a. RoleLLM: Benchmarking, eliciting, and enhancing role-playing abilities of large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 14743–14777.

Xintao Wang, Yunze Xiao, Jen-tse Huang, Siyu Yuan, Rui Xu, Haoran Guo, Quan Tu, Yaying Fei, Ziang Leng, Wei Wang, Jiangjie Chen, Cheng Li, and

Yanghua Xiao. 2024b. InCharacter: Evaluating personality fidelity in role-playing agents through psychological interviews. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1840–1873.

Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. 2025. τ-bench: A benchmark for toolagent-user interaction in real-world domains. In The Thirteenth International Conference on Learning Representations.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. 2026. Agentic context engineering: Evolving contexts for self-improving language models. In The Fourteenth International Conference on Learning Representations.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204– 2213.

Cen Zhao, Tiantian Zhang, Hanchen Su, Yufeng Zhang, Shaowei Su, Mingzhi Xu, Yu Liu, Wei Han, Jeremy Werner, Claire Na Cheng, and Yashar Mehdad. 2025. Agent-in-the-loop: A data flywheel for continuous improvement in LLM-based customer support. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 1919–1930.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36.

Thomas P. Zollo, Andrew Wei Tung Siah, Naimeng Ye, Ang Li, and Hongseok Namkoong. 2025. Personal-LLM: Tailoring LLMs to individual preferences. In The Thirteenth International Conference on Learning Representations.

## A Additional Robustness Panels

This section expands the aggregate PROFYBENCH-V1 results (Table 3) by task family, input variant, and failure mode. Figure 2 presents these three views of the completeconfiguration gains. Table 4 gives paired task-level gains with bootstrap 95% confidence intervals and win rates. Table 5 complements the performance evidence with implementation ownership and write boundaries: background evolution leaves Soul unchanged, while creatorauthorized interactive authoring can edit it.

<table><tr><td>Base</td><td>Metric</td><td>Gain</td><td>95% CI</td><td>Win</td></tr><tr><td>Raw</td><td>Overall</td><td>0.64</td><td>[0.48, 0.82]</td><td>76%</td></tr><tr><td>Raw</td><td>Completeness</td><td>1.01</td><td>[0.78, 1.26]</td><td>62%</td></tr><tr><td>Raw</td><td>Specificity</td><td>0.88</td><td>[0.65, 1.13]</td><td>56%</td></tr><tr><td>Prompt</td><td>Overall</td><td>0.37</td><td>[0.28, 0.48]</td><td>64%</td></tr><tr><td>Prompt</td><td>Completeness</td><td>0.61</td><td>[0.44, 0.79]</td><td>47%</td></tr><tr><td>Prompt</td><td>Specificity</td><td>0.64</td><td>[0.44, 0.85]</td><td>47%</td></tr></table>

Table 4: Paired Profy gains over two baselines on the same 85 tasks. Win denotes the fraction of task-level paired comparisons where Profy scores strictly higher on the specified metric.
<table><tr><td>Layer</td><td>Owner / writer</td><td>Purpose</td></tr><tr><td>Soul</td><td>Creator-authorized authoring, including AI assistance</td><td>Behavioral constraints; background evolution does not</td></tr><tr><td>Published configuration</td><td>Creator publication</td><td>Stable serving snapshot of expert configuration.</td></tr><tr><td>Caller profile</td><td>Invoking user</td><td>Personal information and communication preferences used</td></tr><tr><td>Engagement memory</td><td>Memory tools and background review</td><td>for that caller. User-expert scope; temporal memory also isolates external</td></tr><tr><td>Expert skills</td><td>Scoped candidate acceptance and evaluated promotion</td><td>end users. Personal candidates, shared skill versions, and rollback history.</td></tr><tr><td>Conversation context</td><td>Runtime only</td><td>Temporary history and task state for the current</td></tr></table>

Table 5: Ownership and write boundaries in COG-NILIFE. The key invariant is that automated evolution never writes to the Soul.

## B Evaluation Prompts

Listings 1 and 2 report the evaluation prompt templates. Task-specific checkpoints and familyspecific dimensions are expanded for each instance; the displayed templates abbreviate those variable lists. The single judge (o3-mini) and the four-model audit panel use rubric scoring, with task metadata and a strict JSON response covering scores, error tags, and rationale. Rubric scoring sees one output at a time, without explicit condition labels, together with the task’s checkpoints and risk level. Pairwise comparisons present the same task with two outputs in randomized order, without checkpoints or risk level, and request an A/B/tie preference. Each fully instantiated scoring request expands the variable fields shown in these templates.

![](images/fd8e3cfdaae95f94ce07dd5a8b60177f54a788200af115f6baf748b22a55b3b5.jpg)  
(a) Family-level gains over Expert Prompt.

![](images/97c6bec29bd3573d26cfadebc50fd8090e1f09d15982bcf7a74724baa83b2185.jpg)

(b) Variant-level robustness.
<table><tr><td>Protocol-defined error tag</td><td>Raw</td><td>Prompt</td><td>Full</td><td>Total</td></tr><tr><td>missing_input_not_handled</td><td>17</td><td>12</td><td>0</td><td>29</td></tr><tr><td>weak_personalization</td><td>16</td><td>11</td><td>1</td><td>28</td></tr><tr><td>artifact_incomplete</td><td>3</td><td>1</td><td>2</td><td>6</td></tr><tr><td>overclaiming</td><td>5</td><td>0</td><td>0</td><td>5</td></tr><tr><td>unsafe_boundary</td><td>5</td><td>0</td><td>0</td><td>5</td></tr><tr><td>hallucinated_evidence</td><td>3</td><td>0</td><td>0</td><td>3</td></tr></table>

(c) Counts for error tags listed in the evaluation prompt.  
Figure 2: Robustness and error analysis on PROFYBENCH-V1. The source tally also has two entries labeled evidence grounding (Raw 1; Prompt 1), a label absent from the fixed error-tag list in Listing 1; these entries are excluded here pending raw-output verification. The remaining counts show no missing-input failures and one weak-personalization failure for Full.

Listing 1: Rubric scoring prompt. Placeholders in braces are filled per task; family-specific dimensions are appended according to the task family.

You are an expert evaluator for AI   
,→ agent outputs. Score the   
,→ following system output on a 1-5   
,→ Likert scale for each dimension   
,→ .   
## Task Information   
- Task ID: {task\_id}   
- Task Family: {family}   
- Variant: {variant\_type}   
- Risk Level: {risk\_level}   
- User Input: {user\_prompt}   
- Language: {language}   
## Expected Checkpoints:   
- [must\_include] checkpoint   
,→ description 1   
- [should\_include] checkpoint   
,→ description 2   
## System Output:   
{output\_text}   
## Scoring Dimensions (1=very poor, 3=   
,→ adequate, 5=excellent):   
### General Dimensions:   
1. completeness: Does the output cover   
,→ all required checkpoints?

2. specificity: Does it use concrete   
,→ details tailored to user   
,→ constraints?   
3. actionability: Can the user   
,→ directly act on the answer?   
4. coherence: Is the output well-  
,→ structured and internally   
,→ consistent?   
5. safety\_boundary: Does it avoid   
,→ overclaiming and handle risks   
,→ appropriately?   
### Family-Specific Dimensions:   
- evidence\_grounding (ADV)   
- uncertainty\_calibration (ADV)   
## Error Tags (select ALL that apply   
,→ from this list):   
hallucinated\_evidence, overclaiming,   
,→ missing\_input\_not\_handled,   
,→ weak\_personalization,   
,→ poor\_structure,   
,→ artifact\_incomplete,   
,→ multimodal\_misalignment,   
,→ input\_format\_failure,   
,→ tool\_failure, unsafe\_boundary   
## Instructions:   
- Score each dimension 1-5   
- Select applicable error tags   
- Provide a brief rationale (2-3   
,→ sentences)   
Respond in this exact JSON format:   
{   
"scores": {   
"completeness": <int>,   
"specificity": <int>,   
"actionability": <int>,   
"coherence": <int>,

"safety\_boundary": <int>,   
"<family\_dim\_1>": <int>,   
"<family\_dim\_2>": <int>   
},   
"error\_tags": ["tag1", "tag2"],   
"rationale": "<brief explanation>"   
}

Listing 2: Pairwise preference prompt. Output order is randomized per comparison.

You are an expert evaluator. Compare   
,→ two system outputs for the same   
,→ task and choose which is better.   
## Task Information   
- User Input: {user\_prompt}   
- Task Family: {task\_family}   
- Variant: {variant\_type}   
## Output A:   
{output\_a}   
## Output B:   
{output\_b}   
## Instructions:   
Consider completeness, specificity,   
,→ actionability, safety, and   
,→ overall usefulness.   
Choose: "A", "B", or "tie"   
Respond in this exact JSON format:   
{   
"preference": "<A|B|tie>",   
"rationale": "<brief explanation>"   
}

## C Proposed Product A/B Tests

PROFYBENCH-V1 measures offline task quality. Table 6 proposes low-risk A/B tests linking benchmark observations to measurable production outcomes. Each treatment addresses a deployment question such as fragmented inputs, replay reuse, or creator onboarding, with a primary metric observable in ordinary traffic. These prospective tests, not yet run, would extend evaluation from output quality to live product behavior.

<table><tr><td>A/B test</td><td>Treatment</td><td>Primary metric</td></tr><tr><td>Input</td><td>Structured parser for</td><td>First-turn valid</td></tr><tr><td>robustness Replay reuse</td><td>pasted task content Prefilled “run similar</td><td>input rate Successful</td></tr><tr><td></td><td>case&quot; template</td><td>replay-to-run conversion</td></tr><tr><td>Expert</td><td>Example-rich</td><td>Continuation</td></tr><tr><td>onboarding</td><td>opening prompt</td><td>rate</td></tr><tr><td>Creator</td><td>Guided expert</td><td>Publish rate and</td></tr><tr><td>wizard</td><td>creation checklist</td><td>revision count</td></tr></table>

Table 6: Proposed, not yet conducted, low-risk A/B tests that connect benchmark observations to product iteration.

## D Production Case Screenshots

The screenshots document deployed interactions and artifacts that complement the offline benchmark. Figure 3 covers advising, multimodal artifact production, and marketing-copy revision. Figure 4 follows an interactive multimodal workflow through staged generation and user choices. Figure 5 shows an expert opening with diagnostic questions, consistent with its configured advising role. Together, the examples illustrate how authored guidance and execution workflows become visible service behavior. Account identifiers and user-specific details are masked.

## E Expert Distillation Lifecycle

This section documents creator-directed initial grounding within the continuous distillation lifecycle. Figure 6 follows an interactive expertauthoring run through scenario calibration, phase status, stored content, and a generated reusable skill. The output includes separate Soul (2,287 characters), persona (4,879 characters), and agent-instructions (2,584 characters) fields, written through creator-authorized distillation. Their separation makes the authored identity and operating instructions explicit write targets. Visible phase statuses expose progress through this authoring workflow, connecting elicitation and synthesis to persistent expert artifacts. Section 3 describes how background memory and evaluated skill updates extend maintenance beyond the authoring run.

## F Implementation Correspondence and Research Criteria

Implementation correspondence. Figure 1 summarizes ordinary use of released marketplace experts. Platform templates provide base instructions; nonempty released persona and Soul content replaces the corresponding layers, and expert agent instructions are appended. Creator/admin previews and per-request API rule overrides lie outside this path. The account-profile lookup describes the invoking user, rather than the expert creator’s cross-listing behavioral representation. Publication synchronizes authored shared skill bodies, while runtime reads active skill content directly. Applicable caller/workspace personal skills take precedence over same-name shared skills.

![](images/933a30af46c211763090d793d3f694cf2dbb1f5c0ef4b2827a355f3a1e17686a.jpg)  
(a) Advising replay. Exact user facts are masked while preserving tool-use telemetry.  
Figure 3: Anonymized production screenshots illustrating three COGNILIFE marketplace task geometries. The examples illustrate deployed usage related to PROFYBENCH-V1 task families: advising with source/tool traces, multimodal artifact production through multi-step workflow execution, and marketing copy optimization with concrete revision rationales. Account-side identifiers and exact user-specific admissions details have been removed or masked.

Memory scope and admission. Retrieval and extraction have distinct controls. Temporary chats and explicit memory-disable requests suppress the ordinary memory paths; account-profile loading remains separate. Expert extraction additionally requires memory enablement and activity checks, then saves caller–expert memory items. When enabled, temporal retrieval takes precedence; empty or failed retrieval permits legacy fallback only in allowed scopes. Isolated API end-users can retain permitted temporal retrieval while ordinary expert-background extraction and legacy memory tools remain disabled; nonempty end-user identifiers prohibit legacy fallback. Optional temporal consolidation can synchronize saved expert items into episodes and derive facts. Caller scoping describes storage and retrieval ownership, without guaranteeing exclusive caller visibility of derived evolution summaries.

Candidate acceptance and version promotion. The illustrated existing-skill path stages proposed content on an applicable personal or shared row, retaining the preceding active body. A separate promotion API checks accepted suggestion status, a completed evaluation record with paired numeric rewards and no degradation, target agreement, and caller authority before updating skill content and version history. The current Harbor worker records a single post-evaluation reward; it neither supplies the required before/after pair nor automatically calls promotion. The figure consequently includes no automatic stage-toactive arrow. Version and rollback services do not constitute an implemented cross-listing creator-

attribution classifier.

Platform new-skill exception. Outside the illustrated marketplace path, the platform subjects profy and profy-distill, when no marketplace-expert record exists, can accept an authorized no-parent proposal by creating an enabled personal skill in the acceptor’s active workspace. This branch requires no same-name shared or caller-owned parent; caller-owned parent lookup is not restricted to the active workspace or enabled rows. The proposal becomes both active and candidate content. This implemented acceptance branch does not establish a complete automatic dialogue-to-new-skill generation pipeline.

Different acceptance rules and Soul boundary. Memory extraction, temporal consolidation, authoring, and skill promotion are separate paths with their own checks. Confidence fields do not imply a universal evidence-maturity gate. The illustrated background paths write memory, suggestions, and skill candidates without rewriting stored Soul; accepting non-skill evolution suggestions changes their status and the evolution count. Creator-authorized distillation and editing can update Soul. The proposed attribution criteria therefore remain research requirements beyond these implemented controls.

Proposed maturity assessment. The research design can assess crystallized guidance through knowledge-boundary respect, style stability under pressure, decision-reasoning transparency, out-ofscope handling, sycophancy resistance, and longcontext coherence. These proposed axes give

![](images/503df048ac86970b6eab27c372aaf50cc2b6bdfb6e151e2ad92ea50a99cf7492.jpg)  
(b) Multimodal storybook workflow with staged image-generation actions and batch progress. Figure 3 (continued), panel (b).

改法指导（原句→改法→原因）
<table><tr><td>通勤防晒测评</td><td>&quot;早八人通勤防晒红榜！这3支清爽到像没涂!&quot;</td><td>❤赞1.2w+藏5k+</td></tr><tr><td>油皮防晒合集</td><td>&quot;混油皮夏天防晒怎么选?不黏腻不闷痘闭眼冲!&quot;</td><td>❤赞8k+藏3k+</td></tr><tr><td>近期爱用分享</td><td>&quot;最近通勤都在用这支防晒！服帖到像开了磨皮&quot;</td><td>❤赞6k+藏2k+</td></tr></table>

<table><tr><td colspan="3">田 Table 石山</td></tr><tr><td>原句 改法</td><td></td><td>原因</td></tr><tr><td>我终于找到一个适合通勤党的 清爽防晒</td><td>我终于挖到了这支自然堂轻纱防晒(SPF50+PA++++)，通 勤党的清爽防晒天花板！</td><td>补全产品名与防晒指数，增加 搜索曝光</td></tr><tr><td>它质地不油，早上涂完不会搓 泥</td><td>乳液质地上脸一抹化水，成膜巨快，早上随便抹两下就搞 定，完全不搓泥不假白</td><td>加入&quot;成膜快&quot;&quot;不假白&quot;等高搜索 量关键词</td></tr><tr><td>出门一整天也比较安心</td><td>持妆8小时下来T区微微出油但没晒红没暗沉，下午补喷一次 防晒喷雾就OK</td><td>用具体时间和效果替代模糊 的&quot;安心&quot;</td></tr><tr><td>适合和我一样怕黏腻的姐妹</td><td>同款怕黏腻、追求清爽感的姐妹评论区告诉我你夏天最怕 防晒出什么问题？</td><td>增加互动引导，提升评论数</td></tr></table>

![](images/5440e77ae508f3f1c567e8572c03392a2eb70b98832e25e6368d50b9477dd741.jpg)  
(c) Marketing-note optimization output with a revision table and final platform-style draft. Figure 3 (continued), panel (c).

![](images/7af37d723c675adb1f5f3fb8a1a07d68edf20cb15f94e2176616a46614517331.jpg)

![](images/2385fcde12ee8fa8a146c721e44730b540ed14bead826a1f1e6d81f65ae9d8fb.jpg)

(a) Opening scene: generated background + in-character “Senior” narrator.  
![](images/6d0d67f64e2cfd29df21c9822ce724df3237503c6bc31767c28bcb4f4504bf93.jpg)  
(b) Branching decision point: user picks a family-resource background.  
Figure 4: A deployed COGNILIFE expert (“Video Novel Expert”) executing a multimodal, multi-step workflow that turns one user goal—simulate Chinese college life and a major-choice decision—into an interactive HTML artifact (college life simulator.html). The expert (a) opens with a generated anime background and an in-character narrator, (b) presents a branching decision point where the chosen family-resource background (working-class / middle-class / well-connected) conditions later advice, (c) delivers major-choice advice grounded in school brand, city network, and fallback risk, and (d) closes with an ending screen that summarizes the run as stat bars (Versatility, Financial Status, and a tongue-in-cheek “Hair Remaining”) plus a structured Preference Review.

![](images/6c15752495290a531f3d7e1728e95116c1abbac15180e2cac61f2715cbce854f.jpg)  
(c) Advice dialogue grounding major choice in brand, city, and fallback risk.

![](images/2d817870032c56f4d4a801c7222e0ed8c5eb6c69ec3e2a9066402897fb541b55.jpg)  
(d) Ending screen: playful stat bars and a structured Preference Review.

Figure 4 (continued). This is a concrete instance of the multimodal-workflow (MMW) family in PROFYBENCH-V1: artifact-oriented execution, staged image generation (left pane: live action: generate log, status Done), text–image consistency, and advice-like content delivered inside an interactive simulation rather than a single chat turn. The narrative is a simulation for illustration; it is not evidence of the quality or fairness of real admissions advice, and its family-background branch has not been tested for stereotyped recommendations. Userspecific inputs are masked; the simulator content is procedurally assembled from the user’s in-session choices.

![](images/f285fdf0b14f0f763e85384cc70d9a11a6b3a799722a77abdbf905dfce168256.jpg)  
Figure 5: Opening message of a deployed COGNILIFE expert (Yu Jun PM Mentor). The message introduces a product-management methodology and asks three diagnostic questions before advising the user. This is an example of visible service-opening text; Figure 6 separately illustrates an authoring workflow. No record-level or release-version link between the two examples is established. Journal-based maintenance remains proposed.

![](images/467efd61d82f867cb23a5032efff0cd4c94daa2a5fd4797bdefeb326b756be08.jpg)

(a) Scenario calibration prompts the creator to compare reasoning styles for the draft expert.  
![](images/82a559dabb01487994e0c0aae3c0a577b5cc9318ab00813136cced0d48acfb18.jpg)  
(b) Creator-authoring stages with reported skipped and pending steps.

Figure 6: Creator-directed initial grounding in the Expert Distillation Studio. (a) Product-decision scenarios invite the creator to compare alternative reasoning styles. (b) Reported authoring stages include a research review skipped at the creator’s request, calibration awaiting confirmation, and pending verification. The later “Distillation Complete” status indicates that content was written; it does not indicate creator confirmation, quality verification, or publication. (c) Seven content fields are stored for expert record distilled-9c3a9274, including separate Soul (2,287 chars), persona (4,879 chars), and agent-instructions (2,584 chars) fields. (d) The reusable skill yujun-product-value-exchange, scoped to the same record, encodes the User Value formula, principles, and decision framework. These panels document inspectable authoring outputs and opportunities for review. Creator-authorized AI authoring can write Soul; background memory and skill maintenance follow separate update paths.

Let me quickly check the current state of things.

Explored 2 files, 3 executions, 3 thoughts

Here's a clear status of the Yu Jun (俞军) PM Expert Distillation:

Distillation Complete

Content Fields Written (into distilled-9c3a9274)

<table><tr><td colspan="3">田 Table 石山</td></tr><tr><td>Field</td><td>Status</td><td>Size</td></tr><tr><td>persona</td><td>0</td><td>4,879 chars</td></tr><tr><td>soul</td><td>☑</td><td>2,287 chars</td></tr><tr><td>agent-instructions</td><td>V</td><td>2,584 chars</td></tr><tr><td>opening-message</td><td>∇</td><td>376 chars</td></tr><tr><td>overview</td><td>∇</td><td>399 chars</td></tr><tr><td>name</td><td>V</td><td>俞军产品经</td></tr><tr><td>description</td><td>V</td><td>70 chars</td></tr><tr><td>Skill Created</td><td>V</td><td></td></tr></table>

(c) Seven authored content fields stored for the expert, including a separate Soul field.

<table><tr><td colspan="2">Expert Distillation Studio</td></tr><tr><td>soul V</td><td>2,28/ chars</td></tr><tr><td>agent-instructions ∇</td><td>2,584 chars</td></tr><tr><td>opening-message 0</td><td>376 chars</td></tr><tr><td>overview V</td><td>399 chars</td></tr><tr><td>name ∇</td><td>俞军产品经</td></tr><tr><td>description 0</td><td>70 chars</td></tr></table>

<table><tr><td colspan="3">田 Table 石山</td></tr><tr><td>Skill</td><td>Scope</td><td>Purpose</td></tr><tr><td>yujun-product-value-</td><td>distilled-</td><td>Reusable methodology encoding his User Value formula, 12</td></tr><tr><td>exchange</td><td>9c3a9274</td><td>principles, decision framework</td></tr></table>

(d) A reusable methodology skill scoped to the same authoring record. Figure 6 (continued).

content to the maturity idea. Product progression counters instead summarize accepted evolution events; they are separate from certification against these axes.

Collection and synthesis. Activity triggers determine when the implemented LLM review runs. A zero-LLM pre-filter could further separate inexpensive candidate collection from on-demand synthesis; it remains a design option. Current extraction itself uses an LLM, while task latency is reported for the evaluated Full configuration.

## G Customs Extraction Benchmark and Scoring Scope

Purpose and tasks. The supplementary customs benchmark examines structured deliverables from professional source documents, extending the empirical setting beyond the broad expertservice tasks in PROFYBENCH-V1. The inspected package contains invoices, a packing list, a contract, transport documents, and supporting material. Its instructions require canonical JSON with a header array and item-level groups under table body, using available originals and public rules. Reference annotations are stored separately in verifier-side files. One task, comprising a document packet, is the evaluation unit. The aggregate records contain 52 task identifiers, each with three valid runs. Figure 7 illustrates the inspected task.

Example materials. Three anonymized customs-document example packets are available as anonymous supplementary materials. Each packet contains a contract, invoice, packing list, waybill, and a gold.json file with partial structured reference fields; the accompanying README explains their structure and use. These illustrative materials are not the complete 52-task dataset and do not reproduce the aggregate results.

Configuration and aggregation. The recorded configuration is profy/gpt-6.1-sol/ medium, yielding 156 recorded valid runs. Let q<sub>ir</sub> be the recorded score for task i and run r. We compute

$$
\bar { q } _ { i } = \frac { 1 } { 3 } \sum _ { r = 1 } ^ { 3 } q _ { i r } , \qquad \bar { Q } = \frac { 1 } { 5 2 } \sum _ { i = 1 } ^ { 5 2 } \bar { q } _ { i } .
$$

Thus, tasks have equal weight; fields are not pooled across tasks. These statistics describe the supplied valid runs and do not estimate attemptedrun success or unreported retries. We use recorded task score for the aggregate because the detailed scoring audit covers one packaged task, without establishing contract-to-run consistency across all 52 tasks.

Audited scoring example. Task customs-1026010054001 uses the frozen scoring contract 2026-10-12-v112-af700-importdate. Of 934 registered fields, 389 sourcereviewed objective fields enter scoring: 29 header fields and 360 item-level fields across 44 items (Table 7). Four admitted fields have verified empty reference values. The remaining 545 fields are excluded because they are unresolved, insufficiently supported, or judgment-dependent. An excluded field is outside the denominator, rather than rewarded for an empty prediction.

For this example, failure of canonical-structure or schema checks, or a basic-item count other than 44, gives zero reward. For structurally valid candidates, let $\boldsymbol { a } _ { e , r }$ denote the proportion of equally weighted reference matches. The gated reward is

$$
\begin{array} { l } { { a _ { e , r } = \displaystyle \frac { 1 } { 3 8 9 } \sum _ { f \in G _ { e } } m _ { f } ( y _ { e , r , f } , g _ { e , f } ) , } } \\ { { \ } } \\ { { q _ { e , r } = \displaystyle \left\{ \begin{array} { l l } { { \mathrm { r o u n d } _ { 4 } ( a _ { e , r } ) , } } & { { V _ { e , r } = 1 , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right. } } \end{array}\tag{2}
$$

Here $G _ { e }$ is the admitted field set; $m _ { f } ~ \in ~ \{ 0 , 1 \}$ applies frozen field-specific normalization and equivalence rules. The indicator $V _ { e , r }$ is one only when canonical-structure checks (including unique header fields), the JSON schema, and the 44-row count all pass. Numeric matching requires equality after permitted formatting normalization. Rows are compared by array position, without optimal reassignment. Contract hashes check the gold, scorer, and schema files.

Coverage is concentrated: 352 scored itemlevel fields comprise eight recurring fields over 44 items—item number, transaction quantity, transaction unit, unit price, total price, currency, origin country, and final destination country. Eight further fields are scored for one item each. Product name and HS code therefore enter scoring for only one of 44 items each; specification/model fields are unscored. These counts and Equation 2 describe this example only.

Scope and limitations. The contribution is a professional extraction setting with repeated-run measurements and an inspectable scoring example. The current single configuration lacks matched comparisons and cannot isolate expert guidance, continuous distillation, or cross-expert transfer. Historical native candidates in the package use different model, prompt, skill, and execution conditions; they are not matched baselines. Author metadata records reference drafting and review by the author, without independent inter-annotator agreement or established held-out generalization. Excluded fields provide no measurement of correctness, appropriate abstention, or hallucination. Equal field weights do not represent differing business consequences. A scoreable candidate, including one produced during a business pause, does not establish wholedeclaration correctness or completed business processing. Dataset-wide scoring consistency and reproducible resource availability remain unestablished. These scores are reported separately from PROFYBENCH-V1’s 1–5 LLM-judge scores.

![](images/d0eb5ad2e348e8287029d320a07d4918197652dc034b40b347c397db5231b5db.jpg)  
Figure 7: Task formulation and scoring scope of the inspected customs-extraction example. Source documents are converted into canonical JSON. Canonical-structure, schema, and item-count checks gate comparison over 389 reviewed objective fields; 545 registered fields are excluded. Counts describe one task. A high subset score does not establish business completion.

Results and execution requirements. The macro-average recorded score is 0.958 and the median task mean is 0.966 (Table 8). Thirty-five task means are at least 0.95 and 48 are at least 0.90. Thirteen tasks have a three-run maximumminus-minimum range exceeding 0.05. Figure 8 shows individual runs, means, and observed ranges. Mean runtime is 9.71 minutes and mean estimated cost is USD 1.02 per run. Costs rest on assumed Azure pricing, caching, and long-context rates; they are estimates rather than independently

audited bills.

The inspected task records 0.9923, 0.9897, and 0.9974, averaging 0.9931. These values are consistent with its 389-field denominator. The aggregate records lack corresponding field-level difference reports, so these runs do not support attributing particular error types.

<table><tr><td>Scope item</td><td>Inspected task</td></tr><tr><td>Item rows</td><td>44</td></tr><tr><td>Registered fields</td><td>934</td></tr><tr><td>Scored objective fields</td><td>389</td></tr><tr><td>Scored header / item fields</td><td>29 / 360</td></tr><tr><td>Verified empty scored fields</td><td>4</td></tr><tr><td>Excluded fields</td><td>545</td></tr><tr><td>Items with product name scored</td><td>1/44</td></tr><tr><td>Items with HS code scored</td><td>1/44</td></tr><tr><td>Items with specification/model scored</td><td>0/44</td></tr><tr><td>Structural gate</td><td>Canonical JSON;</td></tr><tr><td></td><td>valid schema; 44</td></tr><tr><td></td><td>basic rows</td></tr></table>

Table 7: Scoring coverage of the inspected example only. Excluded fields are outside the denominator. The metric does not assess their correctness, abstention behavior, or business consequences. Admitted fields have equal weight.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Tasks</td><td>52</td></tr><tr><td>Recorded valid runs per task</td><td>3</td></tr><tr><td>Total recorded valid runs</td><td>156</td></tr><tr><td>Mean recorded task score</td><td>0.958</td></tr><tr><td>Median task mean</td><td>0.966</td></tr><tr><td>Task mean  $\geq 0 . 9 5$ </td><td>35/52</td></tr><tr><td>Task mean  $\geq 0 . 9 0$ </td><td>48/52</td></tr><tr><td>Three-run  $\mathrm { r a n g e } > 0 . 0 5$ </td><td>13/52</td></tr><tr><td>Mean runtime per run</td><td>9.71 min</td></tr><tr><td>Mean estimated cost per run</td><td>USD 1.02</td></tr></table>

Table 8: Supplementary customs results. Scores are averaged within each task and then equally across tasks. They are recorded task scores, not pooled field accuracy or whole-declaration success rates. Costs are estimates under the pricing assumptions stated in the text.

52 tasks × 3 runs per task  
![](images/f8a638ca32da2fa2c867f56d5dd4ee72054827fb08fd967c99bef8cfbfd38a0a.jpg)  
Figure 8: Recorded scores for 52 customs-extraction tasks, sorted by the mean of three runs. Dark dots show task means, light dots individual runs, and vertical bars observed minimum–maximum ranges, not confidence intervals. The dashed line marks the macro-average of 0.9576. The aggregate records and the single-task scoring audit have distinct evidential scopes.