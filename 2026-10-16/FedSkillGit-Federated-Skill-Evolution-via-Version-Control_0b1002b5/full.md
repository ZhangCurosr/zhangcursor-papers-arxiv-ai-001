# FedSkillGit: Federated Skill Evolution via Version Control

Boxun Lan<sup>1</sup>, Jiamei Li<sup>1</sup>, Weijian Li<sup>1</sup>, Chengyu Yang<sup>1</sup>, Zhuang Qi<sup>2</sup>,

Chengcheng Wang<sup>1</sup>, Xiaohong Chen<sup>1</sup>, Ming Hu<sup>1\*</sup>

<sup>1</sup>East China Normal University, Shanghai, China

<sup>2</sup>Shandong University, Jinan, China

Corresponding author: Ming Hu, mhu@sei.ecnu.edu.cn

Abstract—Large language model agents can improve their capabilities by evolving reusable natural-language skills from execution experience. In real-world deployments this experience is distributed across data owners who cannot pool their raw data, which motivates federated skill evolution. The natural first design lets each client evolve the whole skill document and upload the result for serverside aggregation. That design is poorly matched to skill evolution for three reasons. An evolved document can still carry client-specific content distilled from private traces, repeated transmission of a nearly unchanged document spends communication on knowledge that the population already holds, and the contribution of a single client is buried in a document that the server cannot meaningfully attribute. We propose FedSkillGit, a federated skill evolution framework that replaces full-document updates with commits. Each participating client submits a bounded set of patches over named rules together with its local experience, which summarizes what the round taught it, and the pair passes a two-stage privacy filter before it leaves the client. The server deduplicates the submitted patches and the local experience of all participants, updates a global experience document that only the optimizers read, and merges the surviving patches into a candidate skill. The candidate is distributed to the clients, which score it on their own validation sets and report the results, and it is published only when the reported scores improve on the current version. Experiments on three datasets and three language model families show that FedSkillGit outperforms state-of-the-art baselines by up to 22% and approaches centralized evolution without sharing raw client data.

Index Terms—federated learning, skill evolution, large language model agents

## I. INTRODUCTION

Large language model (LLM) agents can improve their task performance by reusing knowledge acquired from previous executions rather than solving similar tasks repeatedly. To support such knowledge reuse, recent approaches distill successful trajectories, failure feedback, and task-specific experience into natural-language skills that guide subsequent reasoning and execution [1]–[3]. As agents encounter new tasks and environments, these skills need to be continuously refined to incorporate the new acquired knowledge. Therefore, effective skill evolution is important for agents that are expected to continuously improve through experience.

Existing skill evolution methods primarily rely on the execution experience accumulated by individual agents, limiting skill improvement to the scope of local experience. In realworld deployments, agents operating across diverse tasks and execution environments can collectively accumulate a broader range of experiences, providing richer knowledge to support skill evolution. However, directly sharing the execution trajectories that generated these experiences risks exposing sensitive input data, private environmental state, or proprietary workflows. Therefore, how to achieve collaborative skill evolution using distributed experiences without sharing raw execution trajectories has become an important research issue.

Federated learning [4]–[10] provides a natural paradigm for such collaboration by enabling multiple clients to jointly improve a shared model without exchanging their raw data. Following this paradigm, a straightforward approach to federated skill evolution is to allow each client to refine a shared skill using its local execution experience and upload the evolved skill document to the server for aggregation. However, unlike conventional FL, which aggregates numerical model updates, skill evolution involves structured natural-language knowledge. Directly adopting the traditional FL paradigm faces three key challenges as follows. ❶ First, keeping raw traces local does not necessarily preserve privacy because an evolved skill can still encode client-specific operations, entities, or environmental information distilled from those traces. ❷ Second, local evolution often modifies only a small fraction of the shared skill, while repeatedly transmitting the entire document causes communication to grow with the accumulated skill rather than with the newly acquired knowledge. ❸ Third, the actual knowledge contributed by a client is buried in a largely unchanged document, making it difficult for the server to identify, evaluate, and combine useful revisions across clients. These limitations reveal a mismatch between the update abstraction of conventional federated learning and the structured natural-language artifacts optimized by skill evolution. Therefore, how to represent and aggregate client knowledge in a privacy-preserving, communication-efficient, and fine-grained manner is an important challenge for federated skill evolution.

To address these challenges, inspired by the incremental revision mechanism of Git in collaborative software development, we propose FedSkillGit, a novel federated skill evolution paradigm. Instead of aggregating full skill documents, clients exchange semantic commits containing fine-grained skill revisions and supporting experience, enabling efficient knowledge exchange and integration. To support this process, FedSkillGit incorporates two-stage privacy filtering to mitigate potential information leakage, a global experience document to guide local skill evolution and server-side aggregation, and version validation to control the publication of updated skills. The main contributions of this paper are summarized as follows.

• We propose FedSkillGit, a novel federated skill evolution framework that integrates fine-grained commit aggregation and version validation to enable reliable collaborative evolution of a shared skill.

• We design a cross-round experience reuse mechanism that leverages accumulated global experience to guide local skill optimization and server-side aggregation, reducing redundant knowledge exploration and integration.

• We develop a two-stage privacy filtering mechanism that combines sanitization and private-value auditing to mitigate the risk of sensitive information leakage.

• Extensive experiments on three datasets and three language model families demonstrate that FedSkillGit outperforms the strongest baseline by up to 22% while approaching centralized performance.

## II. RELATED WORK

## A. Federated learning

Federated learning (FL) [4] enables multiple clients to collaboratively train global model while keeping their raw data local. Existing FL studies have extensively investigated data and system heterogeneity in distributed scenarios. To address data heterogeneity, existing studies have explored various approaches for learning a shared global model, including proximal regularization [11], control variates [12], knowledge distillation [13], prototype-based aggregation [14], [15], and adaptive aggregation [16], [17]. Personalized FL methods instead optimize client-specific models to accommodate heterogeneous local data distributions [18], [19]. To address system heterogeneity, other studies investigate heterogeneous model architectures and resource-constrained client participation [20]–[23]. Despite these advances, FL methods primarily exchange model parameters or other model-related representations, rather than structured natural-language skills.

## B. Skill evolution

LLM agents increasingly employ skill libraries or documents to preserve and reuse knowledge acquired from prior experience. Accordingly, recent studies have investigated various approaches to automated skill acquisition, organization, and evolution. For skill acquisition, existing methods extract reusable knowledge from execution trajectories [1], [24]–[26]. To improve knowledge organization and reuse, other studies explore hierarchical skill organization [2], [27], redundancy reduction through compression and abstraction [28], [29], and large-scale skill curation and composition [30]–[34]. Beyond skill acquisition and management, recent research further investigates skill refinement and evolution through co-evolution and verification [35], parallel generation and consolidation [36], reinforcement learning [37]–[39], and agentic evolution processes [40]. Complementary research also examines the reusable capabilities that language models acquire from incontext experience [41].

However, most existing skill evolution methods assume that the experience used for evolution is available to a single owner or can be centrally accessed. As a result, they do not address how knowledge distributed across privacy-constrained data owners can be collaboratively incorporated into a shared skill. Moreover, unlike conventional federated learning where client updates are typically represented as numerical parameters or gradients, skill evolution directly modifies structured naturallanguage artefacts such as rules, procedures, and local patches. This distinction motivates aggregation mechanisms that operate over discrete revisions rather than numerical updates.

## C. Federated skill evolution

Recent work has begun to extend federated collaboration from model parameters to agent evolution. In particular, FederatedSkill [42] enables agents to collaboratively evolve skill libraries without sharing raw execution trajectories. It represents local experience as skill diffs containing structured modifications to local skill libraries and aggregates these diffs through a server-side evolution agent. Its primary objective is personalized skill evolution under client heterogeneity, where different clients may use different backbone models, agent harnesses, and task distributions. Accordingly, the server infers client-specific capability boundaries from uploaded patches and constructs a personalized skill library for each client.

Unlike FederatedSkill, which maintains personalized skill libraries for heterogeneous clients with heterogeneous tasks, FedSkillGit aims to continuously evolve a generalized skill from distributed private experience for similar tasks. This setting requires determining which locally discovered revisions should become part of the shared knowledge and whether their composition improves the resulting skill. FedSkillGit therefore treats collaborative skill evolution as a versioncontrolled knowledge integration process, where commits are individually evaluated and selectively merged into a shared version, and each new version is validated and rolled back when aggregation causes regression. Our focus is thus not client-specific personalization, but reliable accumulation and validation of distributed knowledge within a common evolving skill.

## III. METHOD

In this section, we introduce the proposed FedSkillGit framework. We first provide an overview of the framework, followed by the local skill evolution, which includes commit generation and privacy filtering. Finally, we describe the server-side skill updating process, consisting of experience updating, experience-aware aggregation, and version validation.

## A. Federated framework

We consider a federated skill evolution system consisting of a central server and N clients, denoted by $\mathcal { C } = \{ 1 , 2 , \ldots , N \}$

The server maintains a shared natural-language skill document d and a global experience document e, which jointly support collaborative skill evolution. Each client $i \in \mathcal { C }$ maintains a private training dataset $\mathcal { D } _ { i }$ and a local validation dataset V , with heterogeneous task distributions across clients. The training dataset is used for local task execution and skill refinement, while the validation dataset is reserved for evaluating candidate skills. Each client is equipped with four LLM-based agents, including an executor E, a skill optimizer O, a sanitizer S, and an auditor A, where E and O are responsible for local task execution and skill refinement, while S and A perform privacy filtering on the generated commits. On the server, a merge agent M is employed to update the global experience document and aggregate the uploaded skill patches, while version validation determines whether the resulting candidate skill is accepted as the new global skill version.

The federated evolution proceeds over multiple communication rounds. Specifically, in each communication round r, a set of $m$ participating clients denoted by $\mathcal { P } _ { r } \subseteq \mathcal { C }$ is selected for skill evolution. Participating clients generate skill updates from their private execution experience and transmit privacy-filtered commits to the server without sharing raw task data or execution trajectories. The server integrates these skill updates to produce a candidate skill, which is evaluated using the clients’ local validation datasets for version control. The objective is to collaboratively improve the task performance of a single shared global skill across heterogeneous clients while keeping their private execution data local.

## B. Overview

Fig. 1 illustrates the overall framework of our proposed Fed-SkillGit, which consists of local skill evolution on the clients and skill updating on the server. In each communication round, the selected clients refine the shared global skill using their private execution experience and submit semantic commits to the server. The server then integrates these commits to update the global experience and generate candidate skill, which undergoes version validation and determines whether to update the global skill. Through this iterative process, FedSkillGit enables collaborative skill evolution across distributed clients without sharing raw execution data.

Local Skill Evolution. In each communication round, the server distributes the current global skill to the selected clients. Each client employs an executor to perform local tasks under the guidance of the current global skill and generate execution trajectories. Based on these trajectories and the global experience, the optimizer produces a rule-level skill patch and a local experience, which are organized into a semantic commit. Since the generated commit may still contain sensitive information derived from private execution data, it proceeds through a twostage privacy protection process consisting of sanitization and auditing. The sanitizer removes potentially sensitive clientspecific information, and the audit agent examines the sanitized commit. Only commits that pass the privacy audit are transmitted to the server.

![](images/741a097e0f85f6db3b316142d41769a638723fff634bfb9cb95ce79b53f35904.jpg)  
Fig. 1. Overview of our proposed FedSkillGit. The framework consists of local skill evolution and server-side skill updating. Each selected client generates a commit containing skill patches and local experience from execution trajectories, followed by two-stage privacy filtering. The server deduplicates uploaded commits, updates global experience, and aggregates patches into a candidate skill. Finally, version validation determines whether to publish the candidate based on its performance.

Skill Updating. Upon receiving the client commits, the server performs three stages of skill updating, including experience updating, experience-aware aggregation, and version validation. First, the server deduplicates the uploaded patches and experiences and incorporates the newly acquired knowledge into the global experience document. The merge agent then integrates the remaining patches into the current global skill, guided by the accumulated experience of previous rounds, to generate a candidate skill. The global experience document also informs subsequent local optimization by recording previously observed failure patterns and attempted revisions, reducing redundant exploration and aggregation. Notably, this document is accessed only by the optimizer and merge agents, while the executor relies exclusively on the shared skill for task execution. Finally, the candidate skill is evaluated by clients on their private validation datasets, and only the resulting success rates are returned to the server. The candidate is published as the new global skill only when its aggregate validation performance is no lower than that of the current version; otherwise, the current skill is retained. The evolution process terminates after two consecutive rounds without an accepted update.

## C. Local Skill Evolution

The local skill evolution process consists of commit generation and privacy filtering two stages. In the commit generation stage, each participating client extracts skill patch and experience summaries from its private execution experience to construct a commit. In the privacy filtering stage, the generated commit undergoes privacy filtering before transmission to mitigate potential information leakage.

1) Commit generation: In communication round r, each participating client $i \in \mathcal { P } _ { r }$ <sub>r</sub> receives the current global skill $d _ { r - 1 }$ and executes its local training set $\mathcal { D } _ { i }$ using its own executor. Based on the current global skill $d _ { r - 1 }$ , the executor generates task responses and their corresponding execution trajectories. Subsequently, the skill optimizer O analyzes the trajectories with the global experience document $e _ { r - 1 }$ to extract its local patch and experience. Specifically, the local skill patch can be represented as a sequence of rule-level edits

$$
p _ { i } ^ { r } = ( \delta _ { i , 1 } ^ { r } , \ldots , \delta _ { i , j } ^ { r } , \ldots , \delta _ { i , K _ { i } ^ { r } } ^ { r } ) ,\tag{1}
$$

where $K _ { i } ^ { r }$ denotes the number of edits proposed by client i in round r. Each edit specifies a modification to a particular rule in the current skill document, formally defined as

$$
\delta _ { i , j } ^ { r } = ( o _ { i , j } ^ { r } , t _ { i , j } ^ { r } , u _ { i , j } ^ { r } ) ,\tag{2}
$$

where $o _ { i , j } ^ { r } , \ t _ { i , j } ^ { r }$ , and $u _ { i , j } ^ { r }$ denote the editing operation, target rule, and revision content, respectively. To constrain the scope

of local modifications, the editing operations include append, insert-after, and replace, which can be formulated as

$$
o _ { i , j } ^ { r } \in \{ \mathsf { a p p e n d } , \mathsf { i n s e r t - a f t e r } , \mathsf { r e p l a c e } \} .\tag{3}
$$

Meanwhile, the local experience summary $e _ { i } ^ { r }$ captures observed failure patterns and the execution experience motivating the proposed revisions. Finally, the generated skill patch and local experience are organized into a semantic commit, which is represented as

$$
c _ { i } ^ { r } = ( p _ { i } ^ { r } , e _ { i } ^ { r } ) .\tag{4}
$$

2) Privacy filtering: Although raw execution trajectories remain local, the generated commits may contain sensitive information derived from private training data. To mitigate privacy risk, FedSkillGit designs a two-stage privacy filtering mechanism consisting of sanitization and auditing.

In the first stage, an LLM-based sanitizer S transforms each generated commit according to predefined sanitization instructions, formally expressed as

$$
\tilde { c } _ { i } ^ { r } = S ( c _ { i } ^ { r } ; \mathcal { T } _ { \mathrm { s a n } } ) = ( \tilde { p } _ { i } ^ { r } , \tilde { e } _ { i } ^ { r } ) ,\tag{5}
$$

where $s$ denotes the sanitization operation guided by instructions $\mathcal { T } _ { \mathrm { s a n } }$ , and $\tilde { p } _ { i } ^ { r }$ and $\tilde { e } _ { i } ^ { r }$ represent the sanitized skill patch and local experience, respectively. Specifically, the sanitizer identifies potentially sensitive information in the free-text fields, including client-specific entities, identifiers, and environment details, and replaces them with typed placeholders. To preserve the applicability of skill revisions, each sanitized edit retains its original operation and target rule, expressed as

$$
\tilde { \delta } _ { i , j } ^ { r } = ( o _ { i , j } ^ { r } , t _ { i , j } ^ { r } , \tilde { u } _ { i , j } ^ { r } ) ,\tag{6}
$$

where $\tilde { u } _ { i , j } ^ { r }$ denotes the sanitized revision content.

In the second stage, the sanitized commit undergoes a privacy audit against a client-specific private-value set $\mathcal { Q } _ { i } .$ which is extracted from local training data and remains fixed throughout the evolution process. The auditor A examines the sanitized commit to detect potential disclosures, formally expressed as

$$
\mathbf { A } ( \widetilde { c } _ { i } ^ { r } , \mathcal { Q } _ { i } ) = \left\{ 1 , \quad \mathrm { i f } \ \forall v \in \mathcal { Q } _ { i } , \neg \mathrm { M a t c h } ( v , \widetilde { c } _ { i } ^ { r } ) , \right.\tag{7}
$$

where Match $. ( v , \tilde { c } _ { i } ^ { r } )$ indicates whether a private value v appears in the free-text fields of the sanitized commit. If the audit fails, the commit is returned to the sanitization stage for further revision. Only commits that pass the privacy audit are transmitted to the server.

The two stages serve complementary purposes. Sanitization removes potentially sensitive content while preserving the structure of skill revisions, whereas auditing provides an additional verification step based on a fixed set of clientspecific private values. This mechanism reduces the risk of disclosing sensitive information through transmitted commits.

## D. Skill Updating

After the server receives sensitivity-protected commits uploaded by clients, the next global skill is determined through experience updates, experience-based aggregation, and version verification three steps.

1) Experience Updating: The server first extracts the local experience summaries from the uploaded commits $\{ c _ { i } ^ { r } \mid i \in$ $\mathcal { P } _ { r } \}$ and performs deduplication to eliminate redundant or conflicting information, which can be expressed as

$$
\bar { c } _ { r } = \mathrm { D e d u p } \left( \{ c _ { i } ^ { r } \mid i \in \mathcal { P } _ { r } \} \right) ,\tag{8}
$$

where $\bar { c } _ { r } = ( \bar { p } _ { r } , \bar { e } _ { r } )$ represents the deduplicated local commits. Subsequently, the merge agent M integrates the experiences in these commits with the current global experience document $e _ { r - 1 }$ to obtain the updated global experience, that is

$$
e _ { r } = \mathbf { M } _ { \mathrm { e x p } } \left( e _ { r - 1 } , \bar { e } _ { r } \right) .\tag{9}
$$

The accumulated experience is organized into typed entries to facilitate structured knowledge management and subsequent retrieval.

2) Experience-aware aggregation: After updating the global experience document, the server performs experienceaware aggregation to construct a candidate skill. Specifically, the merge agent M then integrates the deduplicated patches based on the current global skill $d _ { r - 1 }$ under the guidance of the updated global experience document $e _ { r }$ . Here, the candidate skill $\tilde { d } _ { r }$ can be formulated as

$$
\tilde { d } _ { r } = \mathbf { M } ( d _ { r - 1 } , e _ { r } ) .\tag{10}
$$

During aggregation, the merge agent examines the submitted edits and classifies them as kept, deduplicated, promoted to shared rules, or rejected as invalid. The global experience document provides accumulated knowledge about previously observed failure patterns and attempted revisions, enabling the merge agent to identify redundant or previously rejected proposals. Consequently, redundant or invalid edits are excluded, while the remaining revisions are incorporated into the current global skill. Unlike conventional federated learning that aggregates numerical model updates, this process integrates natural-language revisions while preserving their intended modifications and allowing incorporated changes to be traced to their original proposals.

3) Version validation: After experience-aware aggregation, the candidate skill $\tilde { d } _ { r }$ undergoes version validation before publication. Since clients operate on heterogeneous task distributions, the skill revisions that improve performance on individual clients may not necessarily benefit the entire clients. Therefore, a population-level validation mechanism is employed to determine whether the candidate skill $\tilde { d } _ { r }$ can be the next global skill version. Specifically, the server distributes the candidate skill $\tilde { d } _ { r }$ to all clients, which independently evaluate its task success rate on their own local validation set and then return the corresponding evaluation results. The server aggregates these results to obtain the global evaluation score $s ( d _ { r } )$ , which is compared with the previously recorded score $s ( d _ { r - 1 } )$ . The candidate skill $\tilde { d } _ { r }$ is published only when its evaluation score is no lower than that of the current global skill $d _ { r - 1 }$ . Otherwise, the candidate skill is rejected, and the previous global skill is retained. In this case, the next global skill $d _ { r }$ can be formulated as

$$
d _ { r } = \left\{ \begin{array} { l l } { \tilde { d } _ { r } , } & { s ( \tilde { d } _ { r } ) \geq s ( d _ { r - 1 } ) , } \\ { d _ { r - 1 } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{11}
$$

Through this validation mechanism, the published global skill maintains a non-decreasing evaluation score across communication rounds, while clients retain their private validation data. In addition, the evolution process terminates when candidate skills are rejected in two consecutive communication rounds. To characterize the progression of skill evolution, the convergence round $r ^ { c }$ is defined as the last round in which a candidate skill is accepted.

## IV. EXPERIMENTS

In this section, we first introduce the experimental setup and then evaluate the performance of the proposed method against baselines. Subsequently, ablation studies are conducted to investigate the effectiveness of individual components. Finally, we examine the scalability of the proposed method under different numbers of clients and participation ratios.

## A. Experimental setup

We consider a federated skill evolution system consisting of 10 clients with a participation ratio of 30%, where three clients are selected in each communication round. Experiments are conducted using GPT-5.4-mini, Qwen3.5-flash, and Gemini3- flash large language models on three datasets, including SearchQA for open-domain question answering, LiveMath for research-level multiple-choice mathematical problems, and DocVQA for document-based visual question answering. Each dataset provides an initial seed skill document $d _ { 0 }$

Task success rate (TSR) is adopted as the primary evaluation metric and measured on held-out test sets using the evaluator provided by each dataset. We compare FedSkillGit with four baselines, including Centralized, No Skill, Independent, and DocMerge. For a fair comparison, consistent per-round budgets and publication criteria are applied across applicable methods, while the same executor and merging mechanisms are employed where relevant.

• Centralized. One process is trained on the pooled perround exposure, which is the same per-round data that the federated method receives, and its candidate versions are evaluated on the union of validation sets.

• No skill. The frozen executor is queried with no skill document, which separates what the model already knows from what the evolved document adds.

• Independent. Each client evolves its own copy of the seed on its own data with no collaboration, and the reported value is the mean over clients.

• DocMerge. Each client uploads exactly one mature document and a strong optimizer merges the set in a single call, which is then evaluated on the clients’ validation union. This baseline isolates iterative scheduling against a single-shot merge.

## B. Performance Comparison

TABLE I  
THE TSR RESULTS COMPARISON FOR FIVE METHODS APPLIED TO THREE DATASETS AND THREE MODELS WITH 10 CLIENTS AT 30% PARTICIPATION RATE.(%)
<table><tr><td>Model</td><td>Method</td><td>SearchQA</td><td>LiveMath</td><td>DocVQA</td></tr><tr><td rowspan="5">GPT-5.4-mini</td><td>Centralized</td><td>79.4</td><td>28.8</td><td>86.4</td></tr><tr><td>No skill</td><td>75.9</td><td>14.7</td><td>70.9</td></tr><tr><td>Independent</td><td>76.1</td><td>27.2</td><td>82.8</td></tr><tr><td>DocMerge</td><td>77.5</td><td>28.0</td><td>81.3</td></tr><tr><td>FedSkillGit</td><td>78.6</td><td>28.8</td><td>88.5</td></tr><tr><td rowspan="5">Qwen3.5-flash</td><td>Centralized</td><td>83.1</td><td>32.8</td><td>90.4</td></tr><tr><td>No skill</td><td>77.7</td><td>23.2</td><td>77.6</td></tr><tr><td>Independent</td><td>80.1</td><td>25.6</td><td>83.7</td></tr><tr><td>DocMerge</td><td>81.2</td><td>27.2</td><td>80.2</td></tr><tr><td>FedSkillGit</td><td>82.4</td><td>29.6</td><td>85.3</td></tr><tr><td rowspan="5">Gemini3-flash</td><td>Centralized</td><td>85.9</td><td>41.6</td><td>91.4</td></tr><tr><td>No skill</td><td>79.2</td><td>31.2</td><td>86.6</td></tr><tr><td>Independent</td><td>77.2</td><td>31.2</td><td>88.4</td></tr><tr><td>DocMerge</td><td>83.1</td><td>32.8</td><td>88.4</td></tr><tr><td>FedSkillGit</td><td>87.3</td><td>40.0</td><td>90.1</td></tr></table>

Table I presents the TSR results of different methods across three datasets and three models.

The no-skill row gives the reference for the gain that an evolved document provides. Federating the shared document outperforms independent evolution, since FedSkillGit exceeds Independent in all nine cells by between +1.6, on GPT-5.4-mini and LiveMath, and +10.1, on Gemini3-flash and SearchQA, with a mean gain of +4.3. FedSkillGit also recovers most of the benefit of centralizing, with a median gap of −0.8 against Centralized and a lead in two of the nine cells. By family average it leads on GPT-5.4-mini by +0.4 and trails on Gemini3-flash by −0.5 and on Qwen3.5-flash by −3.0. Single-shot merging is not sufficient, since DocMerge trails in all nine cells by between +0.8 and +7.2. One aggregation call has to reconcile every submitted edit at once and no later round can correct a poor reconciliation.

The margins vary with how well a single client’s slice represents the task. On LiveMath, where the instances are narrow in scope, FedSkillGit leads Independent by 1.6 and 4.0 points for GPT-5.4-mini and Qwen3.5-flash. On DocVQA, where instances are the most heterogeneous across clients, it leads all three model families. The gap to the Centralized reference remains largest on Gemini3-flash.

## C. Ablation Studies

TABLE II  
ABLATION OF THE GLOBAL EXPERIENCE DOCUMENT ON GPT-5.4-MINI AND SEARCHQA. TASK SUCCESS RATE (%).
<table><tr><td>Variant</td><td>TSR (%)</td></tr><tr><td>Full FedSkillGit</td><td>78.6</td></tr><tr><td>w/o experience in next-round training</td><td>76.9</td></tr><tr><td>w/o experience in the federated merge</td><td>77.2</td></tr></table>

1) Effectiveness of the global experience document: Table II isolates the global experience document on GPT-5.4- mini and SearchQA. Both ablated variants keep the same commits, merge operator and publication gate, and they differ only in whether the accumulated experience reaches the optimizer in the next round and reaches the merge agent. Removing either use lowers task success rate relative to the full protocol, from 78.6 to 76.9 and 77.2, so the global experience document contributes in both roles.

At merge time the global experience document tells the merge agent which submitted edits the population has already tried and rejected, so a proposal that looks attractive in isolation can be recognized as a repetition. At training time they tell the optimizer which failures the previous document failed to address on that client’s slice, so the next proposal starts from the residual error instead of rediscovering it.

TABLE III  
PRIVACY ABLATION (GPT-5.4-MINI + SEARCHQA). SHARE OF PAYLOADS THAT PASS THE AUDIT ON THE FIRST ATTEMPT, WITH AND WITHOUT THE SANITIZER.
<table><tr><td>Setting</td><td>Audit first-pass rate (%)</td></tr><tr><td>With sanitizer</td><td>81</td></tr><tr><td>Without sanitizer</td><td>33</td></tr></table>

2) Effectiveness of the sanitizer: Table III evaluates the two-stage privacy filter of Section III on the payloads produced by real runs, again on GPT-5.4-mini and SearchQA. Both rows report first-pass rates, so the table isolates the effect of sanitization. With sanitization in place, 81 % of payloads clear the audit on the first attempt, against 33 % when it is removed, so the share of rejected payloads falls from 67 % to 19 %.

![](images/9e2833f75eb4e2a7c97e29d7d176373d4b6b7d7cd5496f21dc872834d011f554.jpg)  
Fig. 2. Convergence round count against the number of clients, with the participation ratio fixed at 30 % on GPT-5.4-mini and SearchQA. The count rises monotonically and the increase accelerates with population size.

![](images/e34dfab70ecec989a907c1aaced1490eedef4033d03009d86b62709b04e4ac9d.jpg)  
Fig. 3. Convergence round count against the participation ratio, with ten clients on GPT-5.4-mini and SearchQA. Denser participation shortens the path, with strongly diminishing returns.

## D. Scalability

1) Impact of the number of clients: Fig. 2 plots the convergence round count against the number of clients. The count rises monotonically from five rounds at ten clients to nine rounds at one hundred clients, and the increase accelerates with population size. Raising the population from ten to thirty clients adds one round, from 5 to 6, whereas raising it from fifty to one hundred adds two, from 7 to 9. The pattern follows from how much of the document remains unresolved. In the early rounds, most submitted patches are accepted, so each round yields a large improvement. As the document matures, the remaining failures are those on which clients disagree, so submissions increasingly overlap, fewer patches survive the merge, and more rounds are needed before the two consecutive rejections that terminate training are observed.

2) Impact of the participation ratio: Fig. 3 reports the complementary measurement. Raising the participation ratio from 10 % to 50 % shortens the path from 13 rounds to 3, because a larger participating subset contributes more evidence in each round. The returns diminish sharply. Raising the ratio from 10 % to 20 % removes six rounds, from 13 to 7, whereas raising it from 40 % to 50 % removes one, from 4 to 3.

## V. CONCLUSION

We presented FedSkillGit, which evolves a single shared skill document of a population of clients with experience transference and privacy protection. Clients exchange bounded commits rather than complete documents, each commit is filtered before transmission, and the server aggregates the surviving patches into one evolving skill, guided by a global experience document that records the failure patterns and rejected revisions of earlier rounds.

Across three language model families and three datasets, federating the shared document improves task success over independent evolution in every cell by between 1.6 and 10.1 points, with a mean gain of 4.3 points, and it lands within about one point of centralized evolution, with a median gap of −0.8. No raw execution data leaves a client at any point in the protocol. The convergence cost is predictable enough to inform deployment choices, since it rises with the number of clients, from five rounds at ten clients to nine at one hundred, and falls with the participation ratio, from thirteen rounds at 10 % to three at 50 %.

Outlook. FedSkillGit evolves one shared document under a population-level acceptance test, and two extensions follow from that design. A single document admits no per-client personalisation, so a layered artefact in which a validated common core is complemented by per-client overlays is a natural next step. The publication gate consumes validation behaviour, so a reward-free or preference-driven gate would extend the protocol to tasks without an automatic evaluator. Treating the revision as the unit of collaboration may also allow the wider toolkit of version control, including branching, review and provenance, to be applied to the parts of the agent stack that are still edited by hand.

## REFERENCES

[1] Y. Yang, J. Li, Q. Pan, B. Zhan, Y. Cai, L. Du, J. Zhou, K. Chen, Q. Chen, X. Li, B. Zhang, and L. He, “Autoskill: Experience-driven lifelong learning via skill self-evolution,” arXiv preprint arXiv:2603.01145, 2026.

[2] Y. Xiong, Z. Miao, Q. Chen, L. Li, Y. Wang, S. He, J. Zhao, and K. Liu, “Skillpyramid: A hierarchical skill consolidation framework for selfevolving agents,” arXiv preprint arXiv:2606.03692, 2026.

[3] Y. Yang, Z. Gong, W. Huang, Q. Yang, Z. Zhou, Z. Huang, Y. Li, X. Gao, Q. Dai, B. Liu, K. Qiu, Y. Yang, D. Chen, X. Yang, and C. Luo, “Skillopt: Executive strategy for self-evolving agent skills,” arXiv preprint arXiv:2605.23904, 2026.

[4] H. B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Aguera y¨ Arcas, “Communication-efficient learning of deep networks from decentralized data,” in AISTATS 2017, 2017.

[5] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji, K. Bonawitz, Z. Charles, G. Cormode, R. Cummings, R. G. L. D’Oliveira, H. Eichner, S. E. Rouayheb, D. Evans, J. Gardner, Z. Garrett, A. Gascon, B. Ghazi, P. B. Gibbons, M. Gruteser, Z. Harchaoui, C. He,´ L. He, Z. Huo, B. Hutchinson, J. Hsu, M. Jaggi, T. Javidi, G. Joshi, M. Khodak, J. Konecnˇ y, A. Korolova, F. Koushanfar, S. Koyejo,´ T. Lepoint, Y. Liu, P. Mittal, M. Mohri, R. Nock, A. Ozg<sup>¨</sup> ur, R. Pagh,¨ M. Raykova, H. Qi, D. Ramage, R. Raskar, D. Song, W. Song, S. U. Stich, Z. Sun, A. T. Suresh, F. Tramer, P. Vepakomma, J. Wang,\` L. Xiong, Z. Xu, Q. Yang, F. X. Yu, H. Yu, and S. Zhao, “Advances and open problems in federated learning,” Foundations and Trends in Machine Learning 14(1–2), 2021.

[6] M. Hu, Y. Cao, A. Li, Z. Li, C. Liu, T. Li, M. Chen, and Y. Liu, “FedMut: Generalized federated learning via stochastic mutation,” in AAAI Conference on Artificial Intelligence, 2024, pp. 12 528–12 537.

[7] M. Hu, Z. Xia, D. Yan, Z. Yue, J. Xia, Y. Huang, Y. Liu, and M. Chen, “Gitfl: Uncertainty-aware real-time asynchronous federated learning using version control,” in 2023 IEEE Real-Time Systems Symposium (RTSS). IEEE, 2023, pp. 145–157.

[8] Z. Qi, L. Meng, and et al., “Cross-silo feature space alignment for federated learning on clients with imbalanced data,” in AAAI, 2025, pp. 19 986–19 994.

[9] Z. Lu, C. Jia, M. Hu, X. Xie, and M. Chen, “Gradients as an action: Towards communication-efficient federated recommender systems via adaptive action sharing,” in Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, 2025, pp. 1999–2009.

[10] W. Huang, M. Ye, Z. Shi, G. Wan, H. Li, B. Du, and Q. Yang, “Federated learning for generalization, robustness, fairness: A survey and benchmark,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 12, pp. 9387–9406, 2024.

[11] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” in MLSys 2020, 2020.

[12] S. P. Karimireddy, S. Kale, M. Mohri, S. J. Reddi, S. U. Stich, and A. T. Suresh, “Scaffold: Stochastic controlled averaging for federated learning,” in ICML 2020, 2020.

[13] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble distillation for robust model fusion in federated learning,” in NeurIPS 2020, 2020.

[14] M. Cao, T. Chen, M. Hu, Z. Qi, Y. Cui, J. Zhou, and X. Xie, “Two heads are better than one: Generalized cross-domain federated learning via dual-prototype,” in Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, 2026, pp. 59–70.

[15] L. Fu, S. Huang, Y. Lai, T. Liao, C. Zhang, and C. Chen, “Beyond federated prototype learning: Learnable semantic anchors with hyperspherical contrast for domain-skewed data,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2025, pp. 16 648–16 656.

[16] M. Hu, Z. Yue, X. Xie, C. Chen, Y. Huang, X. Wei, X. Lian, Y. Liu, and M. Chen, “Is aggregation the only choice? federated learning via layer-wise model recombination,” in Proceedings of ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD), 2024, pp. 1096–1107.

[17] M. Hu, P. Zhou, Z. Yue, Z. Ling, Y. Huang, A. Li, Y. Liu, X. Lian, and M. Chen, “Fedcross: Towards accurate federated learning via multi-model cross-aggregation,” in Proceedings of IEEE International Conference on Data Engineering (ICDE). IEEE, 2024, pp. 2137–2150.

[18] A. Fallah, A. Mokhtari, and A. Ozdaglar, “Personalized federated learning with theoretical guarantees: A model-agnostic meta-learning approach,” in NeurIPS 2020, 2020.

[19] C. T. Dinh, N. H. Tran, and T. D. Nguyen, “Personalized federated learning with moreau envelopes,” in NeurIPS 2020, 2020.

[20] T. Nishio and R. Yonetani, “Client selection for federated learning with heterogeneous resources in mobile edge,” in IEEE ICC 2019, 2019.

[21] C. Jia, M. Hu, Z. Chen, Y. Yang, X. Xie, Y. Liu, and M. Chen, “Adaptivefl: Adaptive heterogeneous federated learning for resourceconstrained aiot systems,” in Proceedings ofACM/IEEE Design Automation Conference, 2024, pp. 1–6.

[22] R. Liu, M. Hu, Z. Xia, X. Xie, J. Xia, P. Zhang, Y. Huang, and M. Chen, “Fedgraft: Memory-aware heterogeneous federated learning via model grafting,” IEEE Transactions on Mobile Computing, vol. 24, no. 12, pp. 13 506–13 519, 2025.

[23] Z. Xia, M. Hu, D. Yan, R. Liu, A. Li, X. Xie, and M. Chen, “Multisfl: Towards accurate split federated learning via multi-model aggregation and knowledge replay,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 1, 2025, pp. 914–922.

[24] R. Ju, X. Wang, X. Ding, Y. Yang, H. Wu, S. Jiang, Q. Zhang, H. Wen, X. Li, W. Wang, K. Li, Y. Liu, H. Dai, W. Wang, and T. Cao, “Embodiskill: Skill-aware reflection for self-evolving embodied agents,” arXiv preprint arXiv:2605.10332, 2026.

[25] H. Zhou, S. Guo, A. Liu, Z. Yu, Z. Gong, B. Zhao, Z. Chen, M. Zhang, Y. Chen, J. Li, R. Yang, Q. Liu, X. Yu, J. Zhou, N. Wang, C. Sun, and J. Wang, “Memento-skills: Let agents design agents,” arXiv preprint arXiv:2603.18743, 2026.

[26] G. Jiang, Z. Su, X. Qu, and Y. R. Fung, “Xskill: Continual learning from experience and skills in multimodal agents,” arXiv preprint arXiv:2603.12056, 2026.

[27] Y. Liang, R. Zhong, H. Xu, C. Jiang, Y. Zhong, R. Fang, J.-C. Gu, S. Deng, Y. Yao, M. Wang, S. Qiao, Y. Xue, X. Xu, T. Wu, K. Wang, Y. Liu, Z. Bi, J. Lou, Y. E. Jiang, H. Zhu, G. Yu, H. Hong, L. Huang, H. Xue, C. Wang, Y. Wang, Z. Shan, X. Chen, Z. Tu, F. Xiong, X. Xie, P. Zhang, Z. Gui, L. Liang, J. Zhou, C. Wu, J. Shang, Y. Gong, J. Lin, C. Xu, H. Deng, W. Zhang, K. Ding, Q. Zhang, F. Huang, N. Zhang, J. Z. Pan, G. Qi, H. Wang, and H. Chen, “Skillnet: Create, evaluate, and connect ai skills,” arXiv preprint arXiv:2603.04448, 2026.

[28] Y. Gao, Z. Li, Y. Yuan, Z. Ji, P. Ma, and S. Wang, “Skillreducer: Optimizing llm agent skills for token efficiency,” arXiv preprint arXiv:2603.29919, 2026.

[29] S. Tu, C. Xu, Q. Zhang, Y. Zhang, X. Lan, L. Li, D. Li, and D. Zhao, “Dynamic dual-granularity skill bank for agentic rl,” arXiv preprint arXiv:2603.28716, 2026.

[30] C. Wang, Z. Yu, X. Xie, W. Yao, R. Fang, S. Qiao, K. Cao, G. Zheng, X. Qi, P. Zhang, and S. Deng, “Skillx: Automatically constructing skill knowledge bases for agents,” arXiv preprint arXiv:2604.04804, 2026.

[31] S. Ouyang, J. Yan, Y. Chen, R. Han, Z. Wang, B. D. Mishra, R. Meng, C.-L. Li, Y. Jiao, K. Zha, M. Shen, V. Tirumalashetty, G. Lee, J. Han, T. Pfister, and C.-Y. Lee, “Skillos: Learning skill curation for selfevolving agents,” arXiv preprint arXiv:2605.06614, 2026.

[32] S. Shen, W. Cheng, M. Ma, A. Turcan, M. J. Zhang, and J. Ma, “Skillfoundry: Building self-evolving agent skill libraries from heterogeneous scientific resources,” arXiv preprint arXiv:2604.03964, 2026.

[33] X. Liu, X. Luo, L. Li, G. Huang, J. Liu, and H. Qiao, “Skillforge: Forging domain-specific, self-evolving agent skills in cloud technical support,” in Proceedings of the 49th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2026.

[34] Y. Tian, J. Chen, L. Zheng, M. Tao, X. Zeng, Z. Yin, H. Su, and X. Sun, “Skills-coach: A self-evolving skill optimizer via training-free grpo,” arXiv preprint arXiv:2604.27488, 2026.

[35] H. Zhang, S. Fan, H. P. Zou, Y. Chen, Z. Wang, J. Zhou, C. Li, W.-C. Huang, Y. Yao, K. Zheng, X. Liu, X. Li, and P. S. Yu, “Coevoskills: Selfevolving agent skills via co-evolutionary verification,” arXiv preprint arXiv:2604.01687, 2026.

[36] J. Ni, Y. Liu, X. Liu, Y. Sun, M. Zhou, P. Cheng, D. Wang, E. Zhao, X. Jiang, and G. Jiang, “Trace2skill: Distill trajectory-local lessons into transferable agent skills,” arXiv preprint arXiv:2603.25158, 2026.

[37] Y. Shi, Y. Chen, Z. Lu, Y. Miao, S. Liu, Q. GU, X. Cai, X. Wang, and A. Zhang, “Skill1: Unified evolution of skill-augmented agents via reinforcement learning,” arXiv preprint arXiv:2605.06130, 2026.

[38] P. Xia, J. Chen, H. Wang, J. Liu, K. Zeng, Y. Wang, S. Han, Y. Zhou, X. Zhao, H. Chen, Z. Zheng, C. Xie, and H. Yao, “Skillrl: Evolving agents via recursive skill-augmented reinforcement learning,” arXiv preprint arXiv:2602.08234, 2026.

[39] J. Wu, S. Yang, C. Yang, Y. Shen, S. Zhang, Z. Wen, and J. Tao, “Spark: Strategic policy-aware exploration via dynamic branching for long-horizon agentic learning,” arXiv preprint arXiv:2601.20209, 2026.

[40] Z. Ma, S. Yang, Y. Ji, X. Wang, Y. Wang, Y. Hu, T. Huang, and X. Chu, “Skillclaw: Let skills evolve collectively with agentic evolver,” arXiv preprint arXiv:2604.08377, 2026.

[41] S. Si, H. Zhao, Y. Lei, Q. Wang, D. Chen, Z. Wang, Z. Wang, K. Luo, Z. Wang, G. Chen, F. Qi, M. Zhang, and M. Sun, “From context to skills: Can language models learn from context skillfully?” arXiv preprint arXiv:2604.27660, 2026.

[42] J. Yang, G. Yao, Y. Zhang, R. R. Kompella, G. Liu, and S. Chang, “FederatedSkill: Federated learning for agentic skill evolution,” arXiv preprint arXiv:2606.03143, 2026.

## APPENDIX A PROMPT TEMPLATES OF THE PROTOCOL

This appendix gives the prompt templates of FedSkillGit. The executor prompt drives task execution, the optimizer prompt produces the patch and the local experience of a round, the merge prompt updates the global experience document and aggregates the patches, and the two privacy prompts implement the sanitizer and the audit. Commit packaging is derived by fixed rules rather than by a prompt.

## A. Executor prompt

## Executor: task execution under the shared skill

You are solving one task instance from {dataset}.

The skill document below is your only guidance. Nothing else in this conversation overrides it. Read it before you begin, and consult it whenever you are unsure how to proceed.

\### Skill document {skill\_document}

\### Task instance {task\_instance}

### Output requirements   
- Reason before answering, then emit exactly one answer block.

\- Use the answer format defined by the dataset: {answer\_format}

\- Do not refer to the skill document in your answer.

\- You have at most {max\_turns} turns and {max\_tokens} output tokens.

## B. Optimizer prompt

## Optimizer: patch and local experience for one round

You are improving a shared skill document using the evidence of one client. The client holds a private training set; you see its execution trajectories, not its raw data or its gold answers.

\### Current skill document {current\_document}

\### Execution trajectories on this client’s slice {trajectories}

\### Global experience document {global\_experience}

\### What to produce

1. Diagnosis. Identify why the executor failed on this slice. Ground every claim in the trajectories above.

2. Patch. Propose at most {edit\_budget} edits over named rules. Each edit is one of:

\- append <rule>: add a new rule after the named rule - insert <rule>: insert a rule immediately after it - replace <rule>: replace the body of the named rule Every edit must cite the trajectory evidence that justifies it. Do not add a rule that the global experience document already records as rejected.

3. Local experience. Summarise this round as the failure patterns observed on this slice and the expected effect of the proposed edits.

Return one patch and one local experience. Do not edit any rule that the patch does not name. Keep the total document size under {size\_ceiling} tokens.

## C. Merge prompt

## Merge: experience update and patch aggregation

You are the merge step of a federated skill evolution. You receive the current skill document and the set of commits submitted in this round. Each commit packages a patch over named rules together with the local experience of the client that produced it.

\### Current skill document {current\_document}

\### Commits from this round {commits}

\### Step 1: update the global experience document Deduplicate the reported experience. Record what this round established about the task, including any revision that was proposed and found ineffective. The experience document is context for the optimizer and for this step. It is never written into the skill document.

### Step 2: aggregate the patches   
Apply every patch to the current document and classify each edit under one of four dispositions:   
- keep: the edit is new and applies cleanly   
- deduplicate: the edit repeats one already kept   
- promote: the edit belongs in a shared rule   
- reject: the edit is structurally invalid   
An edit is dropped only when it duplicates a kept edit or is structurally invalid. Do not rewrite client text and do not average edits; the returned document must be one that a participant proposed.

Return the updated experience document, the candidate   
skill document, and one disposition record per submitted   
edit.

## D. Sanitizer prompt

## Sanitizer: rewrite a commit before transmission

\### Payload {commit\_payload}

\### How to rewrite Replace each removal with a typed placeholder, for example {ENTITY}, {PATH} or {IDENTIFIER}. Do not delete the surrounding sentence.

Keep the structure of the payload: rule identifiers, edit operations and section headers must survive verbatim, so that the patch still applies.

Do not summarise, explain or comment. Re-emit the payload field by field.

## E. Private-value extraction prompt

## Audit reference: private-value set for one client

You are given a sample of one client’s private training   
data. Extract the literal values whose appearance in text   
sent to another party would constitute a leak.   
### Data sample   
{data\_sample}   
### Output   
Return a flat list of literal strings, grouped by kind:   
- path: file or directory paths   
- identifier: table, column, or record identifiers

Copy the values exactly. Do not paraphrase, generalise or   
complete them. Prefer a short and precise list over a   
long one, because the list is used as a rejection test:   
a transmitted payload that contains any listed value is   
returned to the sanitizer.

## APPENDIX B

## LIMITATIONS

FedSkillGit studies federated skill evolution as a lightweight alternative to pooling execution traces, and its scope is bounded in five ways.

a) The shared object is a single document.: The protocol evolves one skill document that every client is scored against. This is what makes population-level validation possible, since the gate consumes one artefact that the whole population can certify. It also means that a client whose distribution differs sharply from the rest of the population can only be served by improving the common document rather than by receiving an artefact of its own. FederatedSkill takes the opposite trade, keeping a personalised library per client at the cost of an acceptance test that no population can jointly apply.

b) Validation behaviour.: Publication is earned on the union of the clients’ validation sets, so the protocol presumes that each client can score a candidate document on a heldout split. This is a mild requirement where the task has an automatic evaluator and a costly one where success is subjective or expensive to judge. In the latter case the gate would require a stronger scoring procedure, likely with a human in the loop.

c) Privacy is enforced on the payload, not against inference.: The sanitizer and the audit constrain what a transmitted commit carries, and the audit consumes a frozen privatevalue set drawn from the client’s own slice, so a verdict is reproducible. Neither stage bounds what an adversary can infer from the sequence of accepted revisions themselves. Where a formal privacy guarantee is required, the two stages should complement a mechanism such as differential privacy rather than replace it.

d) Commits are visible to the server.: Each commit is sanitized before it leaves a client and the server merges sanitized commits. The protocol does not add secure aggregation or any other cryptographic hiding of the transmitted payload, so the server sees every participant’s sanitized proposal. The privacy filter is therefore essential rather than optional.

e) Convergence cost grows with the population.: The convergence round count rises with the number of clients and falls with the participation ratio, so a deployment with many clients faces a real cost in wall-clock rounds unless the participation ratio is raised. The protocol shares a single seed document across clients, so it also assumes the clients are working on the same problem rather than on unrelated ones.

## APPENDIX C

## EXPERIMENTAL PROTOCOL DETAILS

This appendix records the settings behind the results in Section IV so that the runs can be reproduced from the paper.

a) Models.: Three model families are evaluated, one per row group of Table I, namely GPT-5.4-mini, Qwen3.5-flash and Gemini3-flash. Within each family the same model plays both roles, the executor and the optimizer, so a difference between two methods cannot be attributed to an asymmetric split between the two roles.

b) Datasets and splits.: We use SearchQA for opendomain question answering, LiveMath for research-level mathematics multiple choice and DocVQA for document visual question answering, all as shipped and including each corpus’s own seed skill document. Each dataset is partitioned so that no two clients hold the same examples, and each client’s slice is drawn from a different part of the task distribution, producing a deliberately non-IID split across question type, difficulty band and answer format. The held-out evaluation set is disjoint from every client’s slice.

c) Client and participation settings.: The main comparison uses 10 clients with a 30 % participation ratio, so three clients participate per round. The scheduler is fixed in advance rather than adapting to the observed scores, and a client that is not scheduled in a round contributes nothing to that round. All methods use the same per-round budget envelope and therefore expose the same amount of private data per round. The seed document is held fixed across every method.

d) Methods.: All methods share the executor, the perround exposure, the merge machinery wherever an aggregation step exists and the publication gate. They differ only in how the document is produced. No skill queries the frozen executor with no document, which separates what the model already knows from what the evolved document adds. Independent evolves one copy of the seed per client on that client’s own data with no collaboration and reports the mean over clients. Centralized trains one process on the pooled per-round exposure and measures candidates on $V _ { \cup }$ . DocMerge has each client upload exactly one mature document, which a strong optimizer merges in a single call, and the result is measured on $V _ { \cup }$ . This baseline isolates iterative scheduling against a single-shot merge.

e) Metric.: The reported quantity is the task success rate (TSR) on a sealed held-out split, graded per dataset with that dataset’s own evaluator. Every paired comparison uses the same items within the same run, and the reported artefact is the document published at termination, which is a rule fixed before any run. The convergence round count is reported only in Figures 2 and 3, and is measured as the number of rounds up to and including the last accepted round before the terminating pair of rollbacks in Section III-D3.