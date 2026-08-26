# Contrastive Branch Policy Optimization

Ying Wang<sup>1,2</sup> Changlin Qiu<sup>1,\*</sup> Bang Lin<sup>1</sup> Linbo Jin<sup>1</sup> Wen Jiang<sup>1</sup> Zhe Sun<sup>1</sup> Jingli Yang<sup>2</sup>

<sup>1</sup>Alibaba Group, Hangzhou, China <sup>2</sup>Harbin Institute of Technology, Harbin, China

## Abstract

Reinforcement learning with verifiable rewards (RLVR) enables language models to learn multi-turn interaction with external tools, yet its sparse outcome rewards provide no signal for identifying which intermediate decisions are responsible for success. Branch sampling induces local comparisons among alternative continuations, but existing methods tend to conflate two distinct problems: allocating a fixed rollout budget and translating branch outcomes into token-level credit. We introduce Contrastive Branch Policy Optimization (CBPO), which disentangles these two problems and assigns a dedicated mechanism to each. Generation entropy screens candidate branch positions across the entire response, while pathlevel and node-level decay distribute a fixed budget across trajectories and positions to prevent exploration from collapsing onto a few paths or adjacent tokens. A parent trajectory together with the branches that share an identical token prefix forms an exactprefix group, and the reward variation within this controlled group defines the Contrastive Branch Value (CBV), an outcome-based esti mate of local decision sensitivity that rescales continuation advantages without altering their sign. When multiple nodes are selected along the same trajectory, CBPO partitions it into non-overlapping credit segments, thereby avoiding duplicated gradients on shared tokens. Requiring only outcome rewards and no process-level annotation, CBPO provides a practical solution for fine-grained credit assignment in tool-integrated agent training. Extensive experiments on ten benchmarks—five for mathematical reasoning and five for knowledge-intensive search—show that CBPO consistently outperforms state-of-the-art policy-optimization and branch-based methods, attaining the highest macro-average accuracy in both domains and across two model scales.

## Keywords

reinforcement learning, large language models, tool-integrated reasoning, credit assignment

## 1 Introduction

Chain-of-thought prompting [33] and reinforcement learning with verifiable rewards (RLVR) [28, 29] have substantially improved the reasoning capabilities of large language models (LLMs). Many tasks, however, require current information, exact computation, or feed back from an external environment. Addressing such tasks entails multi-turn interaction with search engines, code interpreters, or both [9, 37]. Recent systems apply reinforcement learning to these interactions [7, 20], including settings that require coordinated use of multiple tools [5].

The resulting training signal nevertheless remains coarse. Most RLVR and agentic reinforcement-learning methods assign a single terminal reward to every generated token in a trajectory. Such uniform credit cannot distinguish a decisive tool request or correction from text that merely precedes a successful answer. Process supervision [22] and step-wise preference optimization [19] provide finer signals but require intermediate labels or preferences. Tree-based sampling instead compares continuations from a shared history using only outcome rewards [14]. This approach, however, still requires a principled policy for locating branches under a limited budget and mapping their outcomes to local policy updates.

Existing methods address these requirements only partially. GIGPO and Tree-GRPO construct local advantages from shared or tree-structured histories [8, 16], whereas ARPO uses generation entropy to select branch points after tool feedback [4]. Entropy provides an inexpensive proxy for token-level uncertainty [2, 24, 32], but does not measure outcome sensitivity. Allocation based solely on entropy can concentrate branches at adjacent positions, overlook consequential decisions outside tool boundaries, or emphasize variation that leaves the final answer unchanged.

Contrastive Branch Policy Optimization (CBPO) separates candidate discovery from credit assignment. CBPO scans local entropy throughout the response and applies path-level and node-level decay to distribute a fixed branch budget. A parent trajectory and branches sharing an identical token prefix form an exact-prefix group. Reward variation within this controlled group defines Contrastive Branch Value (CBV), an outcome-based estimate of local decision sensitivity. A bounded form of CBV changes the magnitude, but not the sign, of continuation advantages. Copied prefixes receive no branch gradient, and multiple branch nodes partition a parent trajectory into non-overlapping credit segments. The two signals therefore serve distinct functions: entropy identifies alternative continuations, whereas observed outcome variation estimates their task relevance.

The evaluation spans two model scales and includes five tool-augmented mathematical benchmarks and five knowledgeintensive search benchmarks. CBPO consistently outperforms state-of-the-art policy-optimization and branch-based methods, attaining the highest macro average in both domains.

The main contributions are as follows:

• Branch selection is formulated as budgeted exploration over the entire response. Fixed-interval entropy windows expose candidates beyond tool boundaries, while path-level and node-level decay prevent allocation from collapsing onto a few trajectories or adjacent positions.

• CBV estimates local decision sensitivity from reward variation within exact-prefix groups. Standardized and bounded CBV modulates continuation advantages without changing their signs. Prefix masking and non-overlapping segmentation prevent repeated credit on shared tokens.

• Experiments across ten benchmarks and two model scales show that CBPO consistently outperforms strong policyoptimization and branch-based baselines. Cross-scale ablations further attribute these gains to balanced branch allocation and outcome-contrastive credit assignment.

## 2 Related Work

## 2.1 Fine-Grained Credit Assignment

PPO optimizes a policy from trajectory-level returns [28]. GRPObased reasoning systems likewise derive one advantage from each completed response [10, 29]. Subsequent variants improve optimization stability or scaling [15, 38], while GSPO moves the optimization unit from tokens to sequences [41]. These methods train directly from outcome rewards but do not distinguish decisive intermediate choices from weakly related transitional text. Problem decomposition, computational correction, tool selection, and answer verification can contribute diferently to a multi-step solution. Assigning a single advantage may therefore reinforce efective decisions and incidental behavior together.

Process rewards [22] and step-wise preference optimization [19] supervise intermediate reasoning more directly. Preference objectives such as DPO [26] and token-entropy methods [11] ofer alternative local signals. Process labels and learned value estimates can provide dense feedback but entail annotation costs or estimation error. Entropy requires neither annotation nor learned value estimation, yet measures predictive uncertainty rather than a position’s empirical association with the final reward.

Tree-based methods reuse prefixes and sample several continuations from intermediate states, enabling local comparisons at controlled cost. Online variants derive advantages from currentpolicy descendants, within-tree contrasts, or recurring histories [14, 16]. Entropy can identify uncertain candidates at low computational cost [2, 24]. Moreover, a small fraction of high-entropy tokens can dominate efective updates [32]. EAPO combines reward polarity with token entropy. AEPO balances entropy during rollout and optimization while limiting repeated branches at consecutive high-entropy positions [6, 11]. These approaches use uncertainty to guide optimization, but divergent token distributions need not yield divergent outcomes. CBPO instead restricts entropy to candidate screening and estimates local credit from outcome contrasts among continuations sharing an identical prefix.

## 2.2 Tool-Integrated Reasoning

External tools allow language models to access current information, perform exact computation, and act on an environment [23, 37]. Tool-integrated reasoning extends beyond the binary decision to invoke a tool. A model must select a tool, construct the request, determine when to invoke it, verify the response, integrate the result, and terminate appropriately. Search provides open-world evidence, whereas code interpreters support calculation and programmatic verification. Coordinating both requires repeated transitions between internal reasoning and external actions as new observations arrive.

Work in this area has progressed from prompted reasoning and acting [9, 37] to reinforcement learning for search [18, 30], code interpreters [7, 20], and multi-tool coordination [5]. Recent WSDM studies train autonomous programmatic agents and optimize tool selection with reinforcement learning [17, 39]. Unlike prompting or supervised imitation, reinforcement learning can optimize the complete interaction from environmental feedback and terminal outcomes. In long trajectories, however, success may depend on pre-call reasoning, request construction, feedback interpretation, or post-call synthesis. Tool-return boundaries therefore capture only a subset of potentially decisive positions.

Knowledge-intensive tasks also depend on retrieval quality and timing. Adaptive retrieval can extend beyond an initial ranking when the first-stage candidate pool has insuficient recall [27]. In LLM-based fact-checking, adaptive evidence retrieval allows a model to determine when internal knowledge is inadequate or conflicts with external evidence [40]. These WSDM studies concern retrieval and verification rather than token-level credit assignment, but reveal the same structural challenge. An agent must determine where additional computation or evidence can alter the final outcome.

Long-horizon tool learning often improves planning or attribution through reward shaping, process supervision, or step-level optimization. Intermediate supervision is costly, while predefined states and step boundaries constrain the granularity ofcredit assignment. GIGPO derives step-level advantages from shared histories. ARPO branches after tool feedback and assigns the resulting advantage to the sufix [4, 8]. The former depends on repeated history states, and the latter treats tool returns as fixed branch boundaries. CBPO searches the full response for branch candidates, then attributes local credit by comparing outcomes under an identical prefix.

## 3 Preliminaries

Tool-integrated rollout. Given a problem � drawn from dataset D and a tool set T, policy $\pi _ { \theta }$ generates an interaction trajectory � that interleaves model tokens, tool requests, and environmental observations. Let $z = ( z _ { 1 } , \ldots , z _ { | z | } ) $ collect the model-generated tokens of�, and let $O _ { < t }$ denote the tool observations available before position �. Trajectory generation then factorizes autoregressively over model tokens only:

$$
P _ { \theta } ( \tau \mid x ; \mathcal { T } ) = \prod _ { t = 1 } ^ { | z | } \pi _ { \theta } ( z _ { t } \mid x , z _ { < t } , o _ { < t } ; \mathcal { T } ) .\tag{1}
$$

Observations are inserted by the environment upon each tool call; they condition subsequent generation but contribute neither likelihood terms nor policy gradients. Upon termination, a verifier assigns a bounded outcome reward �(�). Following agentic reinforcement learning formulations [4, 6], training maximizes the KL-regularized expected reward:

$$
\operatorname* { m a x } _ { \pi _ { \theta } } \mathbb { E } _ { x \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( \cdot | x ; \mathcal { T } ) } \big [ R ( \tau ) \big ] - \beta D _ { \mathrm { K L } } ( \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ) ,\tag{2}
$$

where $\pi _ { \mathrm { r e f } }$ is a frozen reference policy and $\beta$ controls the regularization strength. Shared-history, sufix-based, and tree-search agent RL methods adopt this trajectory-level reward setting [4, 8, 16].

Group-relative optimization. For a given problem, Group Relative Policy Optimization (GRPO) samples � trajectories and estimates the advantage of trajectory � as [29]:

$$
A _ { k } = \frac { R ( \tau _ { k } ) - \mu _ { R } } { \sigma _ { R } + \epsilon } , \qquad \mu _ { R } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R ( \tau _ { j } ) ,\tag{3}
$$

Here, $\mu _ { R }$ and $\sigma _ { R }$ are the within-group reward mean and standard deviation, and � provides numerical stability. Standard GRPO assigns $A _ { k }$ to every model-generated token in trajectory �, so the update magnitude is uniform along the trajectory regardless of which intermediate decisions determine the outcome.

Generation uncertainty. At generation position �, let the context be $h _ { t } = ( x , z _ { < t } )$ and the next-token distribution be $p _ { t } = \pi _ { \theta } ( \cdot |$ $h _ { t } )$ . Its Shannon entropy is

$$
H _ { t } = - \sum _ { v \in \mathcal { V } } p _ { t , v } \log { \bar { p } } _ { t , v } ,\tag{4}
$$

where $_ \mathrm { c } { } _ { V }$ denotes the vocabulary. Larger $H _ { t }$ indicates greater uncertainty over the next token. This quantity reflects the dispersion of the generation distribution at position � rather than the confidence of any particular sampled token. Prior work uses token entropy to identify critical decisions [24] and characterize exploration in language-model reasoning [2, 32].

## 4 Contrastive Branch Policy Optimization

## 4.1 Overview

CBPO addresses two coupled limitations of branch-based RL: concentrated exploration and imprecise local credit. The method uses generation uncertainty for candidate selection and observed outcome variation to modulate update magnitude. Figure 1 summarizes the resulting training framework. Starting from complete interaction trajectories, CBPO scans each response for uncertain positions and allocates a fixed branch budget through path-level and node-level decay. Rewards are then compared among continuations sharing an exact prefix to construct CBV-aware local credit. Sharedprefix deduplication and non-overlapping segmentation convert this credit into token-level advantages. Section 4.2 constructs balanced exact-prefix groups, and Section 4.3 converts their outcomes into policy updates.

The design maintains three invariants. First, every problem receives the same total rollout budget because unused branch slots are reallocated to independent complete trajectories. Second, each local comparison conditions on an identical token history, so reward variation arises only from resampled continuations. Third, copied prefixes are excluded from branch losses, and parent trajectories are partitioned into non-overlapping intervals when several nodes are selected. In tool-integrated trajectories, environmental observations condition later actions but are neither scanned nor assigned policy gradients, since they are not model-generated. These constraints separate allocation from attribution and prevent additional branches from duplicating shared-token gradients.

## 4.2 Entropy-Guided Balanced Branch Exploration

Exhaustive branching at every token is computationally infeasible, whereas branching only at tool-return boundaries assumes that consequential decisions coincide with predefined interaction events. Entropy provides an inexpensive candidate signal, but an unregularized ranking can allocate most of a fixed budget to a few trajectories or neighboring positions. This stage therefore combines full-response candidate discovery with explicit coverage control. The procedure first samples $N _ { 0 }$ complete trajectories from the current policy. Figure 2 illustrates how fixed-interval scanning identifies high-entropy decisions beyond tool-call boundaries.

Full-trajectory candidate identification. For trajectory �, CBPO places candidate boundaries throughout the response at intervals of $d _ { \mathrm { m i n } } ;$

$$
\mathcal { B } _ { i } = \{ b _ { i , q } = q d _ { \operatorname* { m i n } } \} _ { q = 1 } ^ { Q _ { i } } ,\tag{5}
$$

where $Q _ { i }$ is the number of valid candidates. Let $\mathcal { W } _ { i , b }$ denote the window of� model-generated tokens beginning at boundary $b \in \mathcal { B } _ { i }$ . Its normalized entropy is

$$
H _ { i , b } = - \frac { 1 } { | \mathcal { W } _ { i , b } | \log | \mathcal { V } | } \sum _ { t \in \mathcal { W } _ { i , b } } \sum _ { v \in \mathcal { V } _ { K } ^ { i , t } } \mathcal { P } _ { i , t , v } \log \mathcal { P } _ { i , t , v } ,\tag{6}
$$

where $\mathcal { V } _ { K } ^ { i , t }$ is the top- $\mathbf { \nabla } \cdot K$ candidate set at token �, and $\phi _ { i , t , v }$ is the probability of candidate �. The retained probability mass is not renormalized. Thus, $H _ { i , b }$ is a truncated entropy proxy used only to rank candidate locations, not the entropy of a top-� distribution.

The first window of each trajectory provides a path-specific reference:

$$
H _ { i } ^ { \mathrm { r o o t } } = H _ { i , 0 } .\tag{7}
$$

Local uncertainty at boundary $^ { b }$ is measured by the entropy increase relative to this reference:

$$
\Delta H _ { i , b } = H _ { i , b } - H _ { i } ^ { \mathrm { r o o t } } .
$$

The corresponding raw priority is

(8)

$$
P _ { i , b } ^ { \mathrm { r a w } } = \mathrm { c l i p } \left( \alpha + \gamma \Delta H _ { i , b } , 0 , 1 \right) ,\tag{9}
$$

where � sets the base priority, � scales the entropy increase, and clip restricts the score to [0, 1].

Path-node balanced budgeting. An entropy ranking alone can allocate most branches to a few trajectories or neighboring positions. Let $L _ { i }$ count branches assigned to path �, and let $l _ { i , b }$ count branches at node (�, �). CBPO updates the priority as

$$
P _ { i , b } ^ { \mathrm { b a l } } = \frac { P _ { i , b } ^ { \mathrm { r a w } } } { ( 1 + L _ { i } ) ^ { \rho _ { \mathrm { p a t h } } } ( 1 + l _ { i , b } ) ^ { \rho _ { \mathrm { n o d e } } } } ,\tag{10}
$$

where $\rho _ { \mathrm { p a t h } } , \rho _ { \mathrm { n o d e } } \ge 0$ control path-level and node-level decay. Before allocation, CBPO retains at most � candidates from each parent. A candidate remains eligible while $l _ { i , b } < B _ { \mathrm { n o d e } }$ and $L _ { i } ~ <$ $B _ { \mathrm { p a t h } }$ . The hard caps are necessary because power decay never reduces a priority exactly to zero. Updating both counts after each sampled branch progressively shifts allocation toward less-explored paths and nodes.

At each allocation step, CBPO selects the eligible candidate with the largest $P ^ { \mathrm { b a l } }$ . A branch is sampled only if this score exceeds �. Otherwise, independent complete trajectories fill the unused rollout slots, preserving the fixed budget.

At a selected node, the trajectory is decomposed at token boundary � as

![](images/2f365becf5334b29cab8167ff7162e1d7e7e7ed6b0ee79a5cb4f97620e6f6455.jpg)

Figure 1: CBPO training framework. The model first performs interactive reasoning with Python and Search tools and samples complete initial trajectories. Candidate nodes are identified across the full response, and path-level and node-level decay allocate the branch budget. Outcome rewards estimate CBV within exact-prefix groups. Shared-prefix deduplication and non-overlapping hierarchical segmentation then construct token-level advantages for the policy update.  
![](images/fd0c5943c8ff4c1625ee75e879677a275cccd07c1d1811a4054c69641da1ab8b.jpg)  
Figure 2: Full-response candidate discovery. Fixed-interval entropy scans identify high-entropy decisions beyond predefined tool-call boundaries.

$$
\tau = ( p _ { b } , c _ { b } ) , \qquad p _ { b } = \tau _ { < b } , \qquad c _ { b } = \tau _ { \geq b } ,\tag{11}
$$

Branch generation holds prefix $\scriptstyle { \mathcal { P } } b$ fixed and resamples only continuation $c _ { b } .$ . The parent and all branches sharing ${ \mathit { p } } _ { b }$ form an exactprefix group $\mathcal { G } ( \boldsymbol { p _ { b } } )$ . Comparing their outcomes controls for the preceding token history and isolates variation among the sampled continuations. The resulting balanced exact-prefix groups serve as input to the outcome-contrastive credit-assignment stage.

## 4.3 Hierarchical Policy Optimization with CBV

The first stage identifies positions at which the policy admits alternatives, but entropy alone cannot establish whether those alternatives afect the reward. The second stage therefore measures reward variation within each exact-prefix group and translates it into local credit. This formulation emphasizes continuations sampled at outcome-sensitive nodes while suppressing redundant gradients on their shared histories. Figure 3 illustrates the distinction between shared-prefix credit and CBV-aware continuation credit.

![](images/572b5344e133a70e1f002cf04473ad33640e0a333b82d9ab5654e25890d73954.jpg)  
Figure 3: Outcome-contrastive credit assignment. Exactprefix continuations are compared by outcome to construct CBV-aware credit while deduplicating shared-prefix credit.

CBPO requires only bounded outcome rewards. Following Re-Tool and ToRL [7, 20], mathematical tasks use a binary correctness reward that compares trajectory answer �ˆ(�) with reference answer $a ^ { \star }$

$$
R ( \tau ) = \left\{ \begin{array} { l l } { { 1 , } } & { { \hat { a } ( \tau ) \mathrm { ~ i s ~ s y m b o l i c a l l y ~ e q u i v a l e n t ~ t o ~ } a ^ { \star } , } } \\ { { 0 , } } & { { \mathrm { o t h e r w i s e . } } } \end{array} \right.\tag{12}
$$

Search tasks use normalized token-level F1 or an LLM-as-a-Judge score. Every reward satisfies $R ( \tau ) \ \in \ [ 0 , 1 ]$ ; Section 5.1 gives the benchmark-specific metrics.

Contrastive Branch Value. Consider exact-prefix group $\mathcal { G } _ { i } =$ $\{ \tau _ { i , 0 } , . . . , \tau _ { i , K _ { i } } \}$ , where $\tau _ { i , 0 }$ is the parent and the remaining trajectories are branches. Contrastive Branch Value (CBV) is defined as the population standard deviation of group rewards $R _ { i , 0 } , \ldots , R _ { i , K _ { i } } ;$

$$
\mathrm { C B V } _ { i } = \sqrt { \frac { 1 } { K _ { i } + 1 } \sum _ { k = 0 } ^ { K _ { i } } \left( R _ { i , k } - \bar { R } _ { i } \right) ^ { 2 } } , \qquad \bar { R } _ { i } = \frac { 1 } { K _ { i } + 1 } \sum _ { k = 0 } ^ { K _ { i } } R _ { i , k } .\tag{13}
$$

Larger CBV indicates greater reward variation among continuations sampled after the same prefix. Because raw CBV is nonnegative, direct addition would only increase advantage values. Standardization across valid nodes in the batch produces a compa rable signed signal:

$$
Z _ { i } ^ { \mathrm { C B V } } = \frac { \mathrm { C B V } _ { i } - \mu _ { \mathrm { C B V } } } { \sigma _ { \mathrm { C B V } } + \epsilon } .\tag{14}
$$

Here, $Z _ { i } ^ { \mathrm { C B V } } > 0$ denotes reward variation above the batch mean. $\operatorname { I f } \sigma _ { \operatorname { C B V } }$ falls below a numerical threshold, every $Z _ { i } ^ { \mathrm { C B V } }$ is set to zero and the base advantages are retained.

Credit assignment. CBV identifies relative outcome sensitivity at the group level, but the optimizer still requires token-level advantages. Problem-level GRPO normalization first yields base advantage $A _ { i , k }$ for each trajectory. The common prefix at node � receives the group-mean advantage:

$$
A _ { i } ^ { \mathrm { s h a r e d } } = \frac { 1 } { K _ { i } + 1 } \sum _ { k = 0 } ^ { K _ { i } } A _ { i , k } .\tag{15}
$$

Following bounded advantage modulation [11], CBV changes only the magnitude of continuation advantage �. The standardized credit is bounded as

$$
C _ { i , k } ^ { \mathrm { C B V } } = \mathrm { c l i p } \left( Z _ { i } ^ { \mathrm { C B V } } , - \frac { | A _ { i , k } | } { \phi } , \frac { | A _ { i , k } | } { \phi } \right) .\tag{16}
$$

The CBV-aware advantage is

$$
A _ { i , k } ^ { \mathrm { f i n a l } } = A _ { i , k } + \eta \ \mathrm { s i g n } ( A _ { i , k } ) \mathrm { s t o p g r a d } \left( C _ { i , k } ^ { \mathrm { C B V } } \right) ,\tag{17}
$$

where � controls modulation strength and $\phi$ limits its relative magnitude. For $0 \leq \eta < \phi _ { : }$ , the modulated advantage retains the original sign and therefore preserves the local optimization direction.

Hierarchical segmentation for multiple nodes. Let parent trajectory � contain $J _ { i }$ selected boundaries

$$
b _ { i , 1 } < b _ { i , 2 } < \cdots < b _ { i , J _ { i } } .\tag{18}
$$

Multiple selected ancestors can otherwise assign credit repeat edly to the same sufix. CBPO therefore partitions each parent trajectory into non-overlapping intervals, with token advantage

$$
\begin{array} { r } { \widetilde { A } _ { t } = \left\{ \begin{array} { l l } { A _ { i , 1 } ^ { \mathrm { s h a r e d } } , } & { 0 \leq t < b _ { i , 1 } , } \\ { \frac { 1 } { 2 } \left( A _ { i , j , 0 } ^ { \mathrm { f i n a l } } + A _ { i , j + 1 } ^ { \mathrm { s h a r e d } } \right) , } & { b _ { i , j } \leq t < b _ { i , j + 1 } , } \\ { A _ { i , J _ { i } , 0 } ^ { \mathrm { f i n a l } } , } & { t \geq b _ { i , J _ { i } } . } \end{array} \right. } \end{array}\tag{19}
$$

where $A _ { i , j , 0 } ^ { \mathrm { f i n a l } }$ is the parent’s continuation advantage at its �-th selected node. Each intermediate interval is simultaneously a sufix of the preceding node and a shared prefix of the following node. The two associated terms therefore receive equal weight.

CBPO retains the GRPO objective, replacing only trajectory-level advantage $A _ { k }$ with segmented token advantage $\widetilde { A } _ { k , t }$ . For modelgenerated token $y _ { k , t }$ , the importance ratio is

$$
r _ { k , t } ( \theta ) = { \frac { \pi _ { \theta } ( y _ { k , t } \mid x , y _ { k , < t } ) } { \pi _ { \mathrm { o l d } } ( y _ { k , t } \mid x , y _ { k , < t } ) } } ,\tag{20}
$$

and its clipped form

$$
\begin{array} { r } { \bar { r } _ { k , t } ( \theta ) = \mathrm { c l i p } \big ( r _ { k , t } ( \theta ) , 1 - \varepsilon , 1 + \varepsilon \big ) , } \end{array}\tag{21}
$$

$$
\ell _ { k , t } ( \theta ) = \mathrm { m i n } \Big ( r _ { k , t } ( \theta ) \widetilde { A } _ { k , t } , \bar { r } _ { k , t } ( \theta ) \widetilde { A } _ { k , t } \Big ) .\tag{22}
$$

The CBPO objective is

$$
\mathcal { T } _ { \mathrm { C B P O } } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ \frac { 1 } { M } \sum _ { k = 1 } ^ { M } \frac { 1 } { \left| y _ { k } \right| } \sum _ { t = 1 } ^ { | y _ { k } | } \left( \ell _ { k , t } ( \boldsymbol { \theta } ) - \beta D _ { \mathrm { K L } } \left( \pi _ { \boldsymbol { \theta } } \left| \right| \boldsymbol { \pi } _ { \mathrm { r e f } } \right) \right) \right] .\tag{23}
$$

The objective connects the two stages: balanced exploration defines the controlled comparisons, and CBV determines their contribution to non-overlapping continuation updates. Algorithm 1 summarizes the end-to-end training procedure.

Algorithm 1 Contrastive Branch Policy Optimization   
Input initial policy $\pi \theta _ { \mathrm { i n i t } }$ , reference policy $\pi _ { \mathrm { r e f } } ,$ data D, tools T   
Input training steps �, rollout budget �, initial count $N _ { 0 } ,$ caps �<sub>max</sub>, $B _ { \mathrm { n o d e } } , B _ { \mathrm { p a t h } }$   
Input �, �, �, �<sub>path</sub>, �<sub>node</sub>, �<sub>min</sub>, �, $K , \eta , \phi$   
Output trained policy ��   
1: ${ \pi } _ { \theta } \gets { \pi } _ { \theta _ { \mathrm { i n i t } } }$   
2: for training step $s = 1 , \ldots , S$ do   
3: $\pi _ { \mathrm { o l d } }  \pi _ { \theta } ;$ ; sample minibatch $\mathcal { D } _ { b } \subset \mathcal { D }$   
4: for all problems � ∈ D<sub>�</sub> do   
5: sample $N _ { 0 }$ parent trajectories; initialize rollout set $\mathcal { R } _ { x }$   
6: for all parent trajectories $\tau _ { i } \in \mathcal { R } _ { x }$ do   
7: scan $d _ { \mathrm { m i n } } .$ -spaced boundaries and compute $H _ { i , b } , \Delta H _ { i , b } , P _ { i , b } ^ { \mathrm { r a w } }$   
8: retain the top $J _ { \mathrm { m a x } }$ candidates; set $L _ { i } \gets 0$ and $l _ { i , b } \gets 0$   
9: while $| { \mathcal { R } } _ { x } | <$ < � do   
10: form eligible set $C _ { x } = \{ ( i , b ) : L _ { i } < B _ { \mathrm { p a t h } } , \ l _ { i , b } < B _ { \mathrm { n o d e } } \}$   
11: if $C _ { x } = \varnothing$ then   
12: sample $M - | \mathcal { R } _ { x } |$ independent complete trajectories; break   
13: compute $P _ { i , b } ^ { \mathrm { b a l } }$ and choose $( i ^ { * } , b ^ { * } ) = \arg \operatorname* { m a x } _ { ( i , b ) \in C _ { X } } P _ { i , b } ^ { \mathrm { b a l } }$   
14: if $P _ { i ^ { * } , b ^ { * } } ^ { \mathrm { b a l } } >$ � then   
15: fix prefix $\tau _ { i ^ { * } , < b ^ { * } }$ , resample one continuation, and add it to ${ \mathcal { R } } _ { x }$   
16: $L _ { i ^ { * } } { \ ' } \gets L _ { i ^ { * } } + 1 ; l _ { i ^ { * } , b ^ { * } } \gets l _ { i ^ { * } , b ^ { * } } + 1$   
17: else   
18: sample $M - | \mathcal { R } _ { x } |$ independent complete trajectories; break   
19: evaluate �(� ) and compute prompt-level base advantages $A _ { i , k }$ over ${ \mathcal { R } } _ { x }$   
20: build exact-prefix groups; compute $\mathrm { C B V } _ { i } , Z _ { i } ^ { \mathrm { C B V } } , A _ { i } ^ { \mathrm { s h a r e d } } ;$ , and $A _ { i , k } ^ { \mathrm { f i n a l } }$   
21: mask each copied branch prefix; assign $A _ { i , k } ^ { \mathrm { f i n a l } }$ only to its sampled sufix   
22: segment each parent by Eq. (19); keep base advantages for fallback rollouts   
23: maximize Eq. (23) and update �

## 4.4 Theoretical Analysis

Two properties clarify the roles of the two signals. Conditional entropy bounds the outcome information available in sampled continuations, which supports its use for candidate screening. CBV estimates conditional outcome variance under a fixed prefix, which supports its use for local credit. The bounded modulation additionally preserves the advantage sign and local PPO gradient direction.

Property 1: generation entropy bounds attainable outcome information. Given exact prefix ${ \boldsymbol { p } } ,$ let continuation

$C = ( Y _ { 1 } , . . . , Y _ { L } ) \sim \pi _ { \theta } ( \cdot \mid p )$ produce final outcome reward �. The chain rule for conditional entropy and the upper bound on conditional mutual information give

$$
\begin{array} { r l } & { H ( C \mid \boldsymbol { \mathfrak { p } } ) = \displaystyle \sum _ { t = 1 } ^ { L } \mathbb { E } _ { Y _ { < t } \mid \boldsymbol { \mathfrak { p } } } \big [ H ( Y _ { t } \mid \boldsymbol { \mathfrak { p } } , Y _ { < t } ) \big ] , } \\ & { I ( C ; R \mid \boldsymbol { \mathfrak { p } } ) \leq H ( C \mid \boldsymbol { \mathfrak { p } } ) . } \end{array}\tag{24}
$$

For deterministic outcome verification, a nearly deterministic continuation can carry little information about alternative out comes. High conditional entropy is not suficient, however, because variation in wording, formatting, or inconsequential reasoning may leave the answer unchanged. Thus, entropy provides an upper bound rather than a direct estimate of $I ( C ; R \mid p )$ . CBPO uses it to screen for positions where meaningful alternatives may exist, then relies on observed branch rewards for credit assignment.

Property 2: CBV estimates conditional outcome variance under a shared prefix. Suppose � continuations are sampled independently from prefix ${ \boldsymbol { p } } ,$ , yielding rewards $R _ { 1 } , \ldots , R _ { n }$ , and let $\begin{array} { r } { \bar { R } = \bar { n } ^ { - 1 } \sum _ { k } { R _ { k } } } \end{array}$ . Then

$$
\mathrm { C B V } ^ { 2 } ( p ) = \frac { 1 } { n } \sum _ { k = 1 } ^ { n } ( R _ { k } - \bar { R } ) ^ { 2 } = \frac { 1 } { 2 n ^ { 2 } } \sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } ( R _ { i } - R _ { j } ) ^ { 2 } ,\tag{25}
$$

and

$$
\mathbb { E } \big [ \mathrm { C B V } ^ { 2 } ( \boldsymbol { p } ) \mid \boldsymbol { p } \big ] = \frac { n - 1 } { n } \operatorname { V a r } ( \boldsymbol { R } \mid \boldsymbol { p } ) .\tag{26}
$$

The pairwise identity follows by expanding the squared diferences:

$$
\sum _ { i = 1 } ^ { n } \sum _ { j = 1 } ^ { n } ( R _ { i } - R _ { j } ) ^ { 2 } = 2 n \sum _ { i = 1 } ^ { n } R _ { i } ^ { 2 } - 2 \left( \sum _ { i = 1 } ^ { n } R _ { i } \right) ^ { 2 } .\tag{27}
$$

For $i \neq j ,$ conditional independence gives $\mathbb { E } [ ( R _ { i } - R _ { j } ) ^ { 2 } \mid { \boldsymbol { p } } ] =$ $2 \operatorname { V a r } ( R \mid p )$ . The � diagonal terms are zero, leaving �(�−1) nonzero ordered pairs. Substitution into $\operatorname { E q . }$ (25) yields Eq. (26).

Equations (25) and (26) show that $\textstyle { \frac { n } { n - 1 } } \operatorname { C B V } ^ { 2 } ( p )$ is an unbiased estimator of conditional outcome variance. Its pairwise form measures reward divergence among continuations sharing the same token history. For binary correctness with $q _ { \mathcal { P } } = \operatorname* { P r } ( R = 1 \mid \mathcal { p } )$ this variance is $q _ { p } ( 1 - q _ { p } )$ and is maximal at $q _ { p } = 1 / 2$ . Consequently, an uncertain node with consistent outcomes receives little CBV, whereas a node separating success from failure receives more. Bounded modulation changes update magnitude within a controlled interval while preserving the advantage sign and local PPO gradient direction.

Property 3: bounded CBV modulation preserves the update direction. Let $C = C ^ { \mathrm { C B V } }$ satisfy the clipping constraint $| C | \leq | A | / \phi$ For any nonzero base advantage �, Eq. (17) can be written as

$$
A ^ { \mathrm { f i n a l } } = w A , \qquad w = 1 + \eta \frac { C } { \vert A \vert } \in \left[ 1 - \frac { \eta } { \phi } , 1 + \frac { \eta } { \phi } \right] .\tag{28}
$$

When $0 \leq \eta < \phi _ { ; }$ , the lower endpoint is positive. Therefore, �<sup>final</sup> has the same sign as �. The PPO surrogate is positively homogeneous in its advantage argument, so $\ell ( r , w A ) = w \ell ( r , A )$ for � $> 0 .$ Because the modulation is stop-gradient, its local policy gradient is also scaled by the same positive �. CBV consequently reallocates update magnitude without reversing the preference induced by the outcome reward. When $A = 0 ,$ , the clipping constraint forces $C = 0 ;$ and the statement holds trivially.

## 5 Experiments

## 5.1 Experimental Setup

Datasets and evaluation. The evaluation covers mathematical reasoning and knowledge-intensive search, two forms of tool-integrated reasoning. Each task provides access to Web Search and Python, allowing the policy to select tools without dataset-specific routing. Mathematical evaluation uses AIME 2024, AIME 2025, MATH-500, MATH [12], and the complete GSM8K test set [3]. Reported metrics include Pass@1, the five-task macro average, and Pass@3/5 computed from five samples per problem. Knowledge-intensive search evaluation uses WebWalker [34], HotpotQA [36], 2WikiMultiHopQA [13], MuSiQue [31], and Bamboogle [25]. WebWalker is evaluated with LLM-as-a-Judge Pass@1, whereas the other four benchmarks use normalized token-level F1. Domain-specific averages are reported because these metrics are not directly comparable.

Evaluation controls. All methods use the same system prompt, tool interfaces, decoding settings, and answer-extraction procedure. Search calls return the top ten snippets and share an evaluation cache, preventing method-specific retrieval variation from confounding policy comparisons. Mathematical answers are checked with the same symbolic-equivalence verifier. Pass@3/5 uses five independent samples for every method and problem. AIME 2024 and AIME 2025 each contain 30 problems, so one additional correct answer changes accuracy by 3.3 percentage points. Accordingly, the analysis emphasizes cross-task averages, both model scales, and component ablations rather than small AIME diferences in isolation.

Training setup. The experiments use Qwen3-1.7B and Qwen3-4B backbones [35]. Training data are drawn from Tool-Star [5]. Mathematical experiments use 5,000 mathematical samples, whereas search experiments use search and dual-tool trajectories from 10,000 open-domain RL samples. All reward-based methods share the SFT initialization, training data, update count, outcome rewards, tool environment, and 16 rollout slots. OPD checkpoints follow the same benchmark and decoding protocol during evaluation. CBPO allocates six slots to initial trajectories and ten to a second branching stage. Independent complete trajectories fill any unused branch slots. Candidate scanning uses � = 20, $K = 1 0$ $d _ { \operatorname* { m i n } } = 6 4$ , and $J _ { \mathrm { m a x } } = 3$ . Allocation uses $\alpha = 0 . 2 , \gamma = 2 . 0 , \kappa = 0 . 2 5$ $\rho _ { \mathrm { p a t h } } = \rho _ { \mathrm { n o d e } } = 0 . 2 , B _ { \mathrm { n o d e } } = 3 ,$ , and $B _ { \mathrm { p a t h } } = 4 .$ . Credit modulation uses $\eta = 0 . 2 , \phi = 2 . 0$ , and a CBV standard-deviation threshold of $1 0 ^ { - 6 } .$

Baselines. At both model scales, the comparison includes Base, TIR prompting [23], SFT, and on-policy distillation (OPD) [21]. Policy-optimization baselines are GRPO [29], REINFORCE++ [15], DAPO [38], GSPO [41], EAPO [11], and OC-GRPO [1]. Agentic and branch-based baselines are GIGPO [8], Tree-GRPO [16], and ARPO [4].

Table 1: Measured results (%) for Qwen3-1.7B and Qwen3-4B under a unified protocol. Mathematical tasks report Pass@1; WebWalker reports LLM-as-a-Judge Pass@1; the other search tasks report normalized token-level F1. Math Avg. and Search Avg. are unweighted five-task means. Green subscripts show improvement over size-matched GIGPO.
<table><tr><td rowspan="2">Method</td><td colspan="6">Mathematical Reasoning</td><td colspan="6">Knowledge-Intensive Search</td></tr><tr><td>AIME24</td><td>AIME25</td><td>MATH500</td><td>GSM8K MATH</td><td>Math Avg.</td><td>WebWalker</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td><td></td><td>Search Avg.</td></tr><tr><td colspan="10">Backbone: Qwen3-1.7B</td><td></td><td></td><td></td></tr><tr><td colspan="10">Training-Free Method</td><td></td><td></td><td></td><td></td></tr><tr><td>Base</td><td>13.3</td><td>16.7</td><td>82.2</td><td>90.8</td><td>86.7</td><td>57.9</td><td>5.0</td><td>22.0</td><td>25.0</td><td>9.5</td><td>34.0</td><td>19.1</td></tr><tr><td colspan="10">TIR Prompting 16.7</td><td>16.0</td><td>48.0</td><td>31.1</td></tr><tr><td>Supervised and Distillation Methods</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">SFT 16.7</td><td></td><td>58.0</td></tr><tr><td>OPD</td><td>20.0</td><td>23.3 26.7</td><td>82.8 84.4</td><td>91.0 92.0</td><td>87.6 88.7</td><td>60.3 62.4</td><td>16.5 18.5</td><td>48.0 52.0 66.5</td><td>58.0</td><td>22.5 25.0</td><td>60.5</td><td>40.6 44.5</td></tr><tr><td colspan="10">Classic RL Method</td><td></td><td></td></tr><tr><td>GRPO</td><td>16.7</td><td>30.0</td><td>83.6</td><td>91.3</td><td>89.0</td><td></td><td>18.0</td><td>53.5 68.5</td><td></td><td>26.5</td><td>62.0</td><td>45.7</td></tr><tr><td>REINFORCE++</td><td>20.0</td><td>26.7</td><td>83.2</td><td>91.9</td><td>88.3</td><td>62.0</td><td>18.5</td><td>51.0</td><td>65.5</td><td>24.0</td><td>61.0</td><td>44.0</td></tr><tr><td>DAPO</td><td>13.3</td><td>26.7</td><td>84.8</td><td>91.5</td><td>88.8</td><td>61.0</td><td>17.5</td><td>52.0</td><td>65.0</td><td>25.0</td><td>60.5</td><td>44.0</td></tr><tr><td>GSPO</td><td>20.0</td><td>26.7</td><td>84.6</td><td>91.0</td><td>88.7</td><td>62.2</td><td>18.5</td><td>54.0</td><td>68.0</td><td>26.0</td><td>63.0</td><td>45.9</td></tr><tr><td>EAPO</td><td>23.3</td><td>30.0 33.3</td><td>85.6 85.2</td><td>91.8 92.4</td><td>90.0 89.5</td><td>64.1 64.1</td><td>20.5 19.5</td><td>55.0 54.5</td><td>69.5 69.0</td><td>32.0 27.0</td><td>65.0 64.0</td><td>48.4 46.8</td></tr><tr><td colspan="10">OC-GRPO 20.0 Agentic RL Method</td><td colspan="3"></td></tr><tr><td colspan="10">GIGPO</td><td></td><td></td><td></td></tr><tr><td>Tree-GRPO</td><td>23.3</td><td>26.7</td><td>84.2</td><td>92.4 92.0</td><td>89.1 88.8</td><td> $6 3 . 1 _ { \Delta _ { \mathrm { b a s e } } }$ </td><td>23.5</td><td>58.0</td><td>73.0</td><td>29.5</td><td>67.0</td><td>50.2∆base</td></tr><tr><td>ARPO</td><td>23.3</td><td>30.0 30.0</td><td>85.0 85.0</td><td>91.7</td><td>88.3</td><td>63.8+1.1%</td><td>24.0</td><td>58.5</td><td>73.5</td><td>30.0</td><td>68.0</td><td>50.8+1.2%</td></tr><tr><td>CBPO</td><td>26.7</td><td>33.3</td><td>86.4</td><td>93.2</td><td>64.3 90.4</td><td>+1.9% 66.0+4.6%</td><td>24.5 26.0</td><td>59.0 61.0</td><td>74.0 75.5</td><td>30.5</td><td>69.0 72.0</td><td>51.4+2.4% 53.2+6.0%</td></tr><tr><td colspan="10">26.7</td><td colspan="3">31.5</td></tr><tr><td colspan="10"></td></tr><tr><td>Training-Free Method</td><td></td><td>20.0</td><td>84.6</td><td>90.0</td><td>89.8</td><td>59.5</td><td>7.0</td><td>27.0</td><td>31.0</td><td>12.0</td><td>40.0</td><td>23.4</td></tr><tr><td colspan="10">Base 13.3 TIR Prompting 16.7</td></tr><tr><td>Supervised and Distillation Methods</td><td></td><td>26.7</td><td>85.1</td><td>92.7</td><td>89.3</td><td>62.1</td><td>17.5</td><td>41.5</td><td>47.0</td><td>19.5</td><td>53.5</td><td>35.8</td></tr><tr><td colspan="10">SFT 16.7</td></tr><tr><td>OPD</td><td>23.3</td><td>26.7 33.3</td><td>85.7 86.5</td><td>93.2 94.1</td><td>90.5 89.6</td><td>62.6 65.4</td><td>20.0 21.5</td><td>52.0 56.0</td><td>63.0 71.5</td><td>25.5 28.0</td><td>63.0 65.5</td><td>44.7 48.5</td></tr><tr><td colspan="10">Classic RL Method</td></tr><tr><td>GRPO</td><td>16.7</td><td>40.0</td><td>86.0</td><td>93.8</td><td>91.3</td><td>65.6</td><td>21.5</td><td>57.5</td><td>73.0</td><td>29.5</td><td>66.5</td><td>49.6</td></tr><tr><td>REINFORCE++</td><td>26.7</td><td>33.3</td><td>85.8</td><td>94.2</td><td>91.4</td><td>66.3</td><td>22.0</td><td>55.0</td><td>70.0</td><td>27.0</td><td>65.5</td><td>47.9</td></tr><tr><td>DAPO</td><td>13.3</td><td>30.0</td><td>87.8</td><td>93.5</td><td>91.5</td><td>63.2</td><td>21.0</td><td>56.0</td><td>69.0</td><td>28.0</td><td>65.0</td><td>47.8</td></tr><tr><td>GSPO</td><td>20.0</td><td>33.3</td><td>87.1</td><td>93.4</td><td>91.2</td><td>65.0</td><td>21.5</td><td>57.0</td><td>72.0</td><td>29.0</td><td>66.0</td><td>49.1</td></tr><tr><td>EAPO</td><td>23.3</td><td>36.7</td><td>87.2</td><td>94.4</td><td>91.8</td><td>66.7</td><td>24.0</td><td>59.5</td><td>74.5</td><td>35.0</td><td>69.0</td><td>52.4</td></tr><tr><td>OC-GRPO</td><td>23.3</td><td>33.3</td><td>87.0</td><td>94.2</td><td>90.6</td><td>65.7</td><td>22.5</td><td>58.5</td><td>73.5</td><td>29.5</td><td>67.5</td><td>50.3</td></tr><tr><td colspan="10">Agentic RL Method</td></tr><tr><td>GIGPO</td><td>26.7</td><td>33.3</td><td>86.7</td><td>94.6</td><td>91.0</td><td> $6 6 . 5 _ { \Delta _ { \mathrm { b a s e } } }$ </td><td>27.0</td><td>62.0</td><td>77.0</td><td>32.5</td><td>73.0</td><td>54.3∆base</td></tr><tr><td>Tree-GRPO</td><td>26.7</td><td>33.3</td><td>87.5</td><td>94.0</td><td>91.9</td><td>66.7+0.3%</td><td>28.0</td><td>62.5</td><td>77.5</td><td>33.0</td><td>74.0</td><td>55.0+1.3%</td></tr><tr><td>ARPO</td><td>30.0</td><td>33.3</td><td>87.4</td><td>94.1</td><td>90.7</td><td>67.1 +0.9%</td><td>29.0</td><td>63.0</td><td>78.0</td><td>33.5</td><td>75.0</td><td>55.7 +2.6% 57.5+5.9%</td></tr><tr><td>CBPO</td><td>30.0</td><td>36.7</td><td>89.5</td><td>96.3</td><td>94.0</td><td>69.3+4.2%</td><td>31.0</td><td>65.5</td><td>80.0</td><td>34.5</td><td>76.5</td><td></td></tr></table>

## 5.2 Main Results

The primary comparison evaluates whether the complete CBPO pipeline improves closed-form reasoning and open-world retrieval under matched rollout budgets. Table 1 reports mathematical macroaverage Pass@1 scores of 66.0% and 69.3% with the 1.7B and 4B backbones. These scores exceed those of ARPO, the strongest sizematched baseline by macro average, by 1.7 and 2.2 percentage points. The corresponding search averages are 53.2 and 57.5, exceeding ARPO by 1.8 points at both scales. Relative to OPD, CBPO gains 3.6 and 3.9 points on the mathematical average and 8.7 and 9.0 points on the search average. CBPO ranks first or ties for first on all five mathematical benchmarks at 1.7B and on four of five at 4B. At 4B,

GRPO leads AIME 2025 by one problem. CBPO also leads four of five search benchmarks at each scale, trailing EAPO on MuSiQue by 0.5 points. This consistency across domains and model scales supports the overall pipeline. The following analyses examine the component-specific hypotheses.

Pass@K extension. Figure 4 evaluates whether the gain persists when several candidate answers are available. From five independent samples per problem, Qwen3-1.7B with CBPO reaches macro-average Pass@1, Pass@3, and Pass@5 scores of 66.0%, 74.6%, and 77.9%. The corresponding Qwen3-4B scores are 69.3%, 76.9%, and 80.2%. CBPO exceeds ARPO at each reported value, indicating that its advantage is not confined to single-sample decoding.

(j) MATH  
![](images/3caff7e0f4de3dfc2e1229a0896fe59fc6b7e90d2e919773a5ebf9844ab45daf.jpg)

![](images/0e4eff0a2559ba19869d4002dc6df65b4d90f9fd3bc0524db8c17639e868bc0b.jpg)

![](images/e76beb2ae89a08d115e350aa3121563eddd239f8ee13c28519a05dab266eba2f.jpg)

![](images/7e2445c6819ff4a17674355e67fc22a5c3ecf81661df8eccfbc0402aaa44ec08.jpg)  
(e) MATH

![](images/5638d59212095027de504d072c4df8662d20088c9af063bba9519381c6ae6432.jpg)

![](images/9617ff4a24e0bc75d8658992ac79e8544ab9a8c0abb452a792bd61f6506e30da.jpg)

![](images/0dd2d320fe50cbe118ba37dc18f376cf0338b7ba29ad2f3ad90b7c03bddbd989.jpg)

![](images/604337f15fd6528f18c8416a12ad0bfe844a9dff5a4e664f6fd88f96f0dbf298.jpg)

![](images/99895c572d43c391f0393cdad643922abffa55adb34c0c84c18a5f55c3b1ebfa.jpg)

(i) GSM8K  
![](images/571cf18f4a5547ce1f4d6b279db822396c937780f88fda5d890c0fbaa64cef46.jpg)

![](images/64697cbc2a1ce0c7be70faa2e81f8b9c342e3d8ca3a996a411603dc17d448cd0.jpg)  
Figure 4: Pass@K comparison between ARPO and CBPO. The top and bottom rows show Qwen3-1.7B and Qwen3-4B. Pass@3/5 is computed from five samples per problem. Each benchmark uses an independent vertical axis; consequently, bar height should not be compared across benchmarks.

Training dynamics and branch selection. Figure 5 compares reward, tool use, and selected branch positions. CBPO attains a higher final mean reward than GIGPO and ARPO while averaging fewer tool calls. The performance gain therefore cannot be attributed to more frequent tool invocation. The qualitative examples show that decay can reorder high-entropy candidates before allocation, consistent with separating candidate discovery from budget control.

## 5.3 Full-Response Candidate Analysis

Three Qwen3-1.7B trajectories from the MATH test set were inspected to locate decisions associated with their outcomes. In one successful trajectory, the model enumerated all 16 one- and twodigit candidates before calling Python. The interpreter then cor rectly identified the ten primes. In a failed trajectory, the submitted code also executed correctly, but the preceding reasoning listed only six parenthesizations of an arithmetic expression. Python evaluated this incomplete list and returned two values instead of the correct four. The decisive error therefore occurred in the reasoning before tool invocation, not at the tool boundary.

The third trajectory exhibited the converse pattern. Python raised the same execution exception twice, yet the model recovered the correct answer from an algebraic derivation completed before either call. A tool-return position was therefore salient but not decisive for the final reward. Together, these cases illustrate why candidate discovery should cover the full response rather than treat tool observations as universal branch boundaries. These examples provide qualitative mechanism checks, not population-level or causal evidence. A controlled tool-boundary-only ablation remains necessary to quantify this design choice independently.

## 5.4 Ablation Study

The method motivates two quantitative predictions. Path-level and node-level decay should reduce complementary forms of allocation concentration, whereas CBV should improve learning beyond

Figure 5: Training dynamics and branch selection. (a) Mean reward and (b) mean tool calls per trajectory; thick lines denote smoothed means, with light traces and shading showing variation. In the qualitative word clouds, (c) token size increases monotonically with candidate entropy, whereas (d) token size increases monotonically with post-decay branching priority among retained candidates. Entropy therefore favors selection, but path- and node-level decay can change the ranking and retain some positions below the entropyonly cutof; sizes do not encode token frequency.

branch exploration alone. Table 2 tests these predictions under matched training, rollout, and evaluation protocols.

Removing either decay level reduces the mathematical macro average by 1.5 points at both model scales. Removing both reduces it by 2.2 points, more than either individual ablation. The two decay terms therefore regulate distinct sources of allocation concentration. Removing CBV produces the largest losses, 3.0 points for Qwen3-1.7B and 2.9 points for Qwen3-4B. Branch exploration alone is therefore insuficient; branch outcomes must also inform local credit. Most of this reduction occurs on AIME 2024 and AIME 2025, the two benchmarks with the lowest absolute accuracy.

Table 2: Component ablations for CBPO. Values are Pass@1 (%) and the five-task macro average. Shading marks the full method, and bold indicates the best value in each column.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>MATH500</td><td>GSM8K</td><td>MATH</td><td>Avg.</td></tr><tr><td colspan="7">Backbone: Qwen3-1.7B</td></tr><tr><td>CBPO</td><td>26.7</td><td>33.3</td><td>86.4</td><td>93.2</td><td>90.4</td><td>66.0</td></tr><tr><td>w/o Path-level</td><td>23.3</td><td>30.0</td><td>86.2</td><td>92.8</td><td>90.2</td><td>64.5</td></tr><tr><td>w/o Node-level</td><td>23.3</td><td>30.0</td><td>86.1</td><td>93.0</td><td>90.3</td><td>64.5</td></tr><tr><td>w/o Dual decay</td><td>20.0</td><td>30.0</td><td>86.3</td><td>92.5</td><td>90.0</td><td>63.8</td></tr><tr><td>w/o CBV</td><td>20.0</td><td>26.7</td><td>85.5</td><td>92.9</td><td>90.1</td><td>63.0</td></tr><tr><td colspan="7">Backbone: Qwen3-4B</td></tr><tr><td>CBPO</td><td>30.0</td><td>36.7</td><td>89.5</td><td>96.3</td><td>94.0</td><td>69.3</td></tr><tr><td>w/o Path-level</td><td>26.7</td><td>33.3</td><td>89.3</td><td>95.9</td><td>93.8</td><td>67.8</td></tr><tr><td>w/o Node-level</td><td>26.7</td><td>33.3</td><td>89.0</td><td>96.0</td><td>93.9</td><td>67.8</td></tr><tr><td>w/o Dual decay</td><td>23.3</td><td>33.3</td><td>89.4</td><td>95.7</td><td>93.9</td><td>67.1</td></tr><tr><td>w/o CBV</td><td>23.3</td><td>30.0</td><td>89.0</td><td>96.0</td><td>93.7</td><td>66.4</td></tr></table>

Table 3: CBPO Pass@1 (%) under diferent rollout configurations. � = � + �. Shading marks the default configuration, and bold indicates the best value in each column.
<table><tr><td colspan="3">Rollout allocation</td><td colspan="6">Pass@1 accuracy (%)</td></tr><tr><td>Total M</td><td>Initial N</td><td>Branch B</td><td>AIME24</td><td>AIME25</td><td>MATH500</td><td>GSM8K</td><td>MATH</td><td>Avg.</td></tr><tr><td colspan="9">Backbone: Qwen3-1.7B</td></tr><tr><td>4</td><td>1</td><td>3</td><td>13.3</td><td>23.3</td><td>82.2</td><td>89.9</td><td>86.0</td><td>58.9</td></tr><tr><td>4</td><td>2</td><td>2</td><td>16.7</td><td>23.3</td><td>82.8</td><td>90.5</td><td>86.8</td><td>60.0</td></tr><tr><td>8</td><td>2</td><td>6</td><td>16.7</td><td>26.7</td><td>83.7</td><td>91.0</td><td>87.8</td><td>61.2</td></tr><tr><td>8</td><td>4</td><td>4</td><td>20.0</td><td>26.7</td><td>84.0</td><td>91.5</td><td>88.3</td><td>62.1</td></tr><tr><td>16</td><td>4</td><td>12</td><td>23.3</td><td>36.7</td><td>85.8</td><td>92.9</td><td>89.8</td><td>65.7</td></tr><tr><td>16</td><td>6</td><td>10</td><td>26.7</td><td>33.3</td><td>86.4</td><td>93.2</td><td>90.4</td><td>66.0</td></tr><tr><td>16</td><td>8</td><td>8</td><td>26.7</td><td>30.0</td><td>86.1</td><td>93.0</td><td>90.2</td><td>65.2</td></tr><tr><td>16</td><td>12</td><td>4</td><td>23.3</td><td>30.0</td><td>85.8</td><td>92.6</td><td>89.8</td><td>64.3</td></tr><tr><td colspan="9">Backbone: Qwen3-4B</td></tr><tr><td>4</td><td>1</td><td>3</td><td>16.7</td><td>26.7</td><td>84.4</td><td>91.8</td><td>89.0</td><td>61.7</td></tr><tr><td>4</td><td>2</td><td>2</td><td>20.0</td><td>26.7</td><td>85.0</td><td>92.3</td><td>89.7</td><td>62.7</td></tr><tr><td>8</td><td>2</td><td>6</td><td>20.0</td><td>30.0</td><td>86.6</td><td>93.8</td><td>91.0</td><td>64.3</td></tr><tr><td>8</td><td>4</td><td>4</td><td>23.3</td><td>30.0</td><td>87.0</td><td>94.2</td><td>91.5</td><td>65.2</td></tr><tr><td>16</td><td>4</td><td>12</td><td>26.7</td><td>40.0</td><td>88.9</td><td>95.8</td><td>93.3</td><td>68.9</td></tr><tr><td>16</td><td>6</td><td>10</td><td>30.0</td><td>36.7</td><td>89.5</td><td>96.3</td><td>94.0</td><td>69.3</td></tr><tr><td>16</td><td>8</td><td>8</td><td>30.0</td><td>33.3</td><td>89.2</td><td>96.0</td><td>94.4</td><td>68.6</td></tr><tr><td>16</td><td>12</td><td>4</td><td>26.7</td><td>33.3</td><td>88.8</td><td>95.6</td><td>93.2</td><td>67.5</td></tr></table>

## 5.5 Branching Configuration

Balanced branching requires suficient independent paths for global coverage and suficient shared-prefix continuations for local comparison. This trade-of is evaluated by varying the allocation between initial trajectories � and dynamic branches �. Total budgets are $M = N + B \in \{ 4 , 8 , 1 6 \}$ , with all other settings held fixed.

Increasing � from 4 to 16 improves the macro average at both model scales. Within a fixed budget, too few initial trajectories restrict path coverage, whereas too few branches weaken exactprefix comparisons. The highest macro average occurs at � = 6, � = 10 for both backbones. This shared optimum favors branch sampling while retaining suficient independent trajectories for global coverage.

## 6 Conclusion

CBPO addresses fine-grained credit assignment by separating two operations that branch-based RL often conflates. Full-response entropy scanning with path-level and node-level decay allocates a fixed rollout budget. Reward variation within exact-prefix groups then modulates the contribution of each sampled continuation. Prefix masking and hierarchical segmentation prevent shared tokens from receiving duplicated credit. With Qwen3-1.7B and Qwen3-4B, CBPO attains mathematical macro averages of 66.0% and 69.3% and search averages of 53.2 and 57.5. Cross-scale ablations attribute distinct gains to balanced allocation and CBV. However, the independent contribution of full-response scanning still requires a controlled boundary-only ablation. Within the evaluated models, tasks, and budgets, the evidence supports a bounded design principle: uncertainty identifies alternatives, whereas observed outcome variation provides a stronger signal for local credit assignment.

## Ethical Considerations

The study uses public benchmarks and collects no new personal or humansubject data. Open-web retrieval may nevertheless expose a model to inaccurate, biased, ofensive, or privacy-sensitive material. Benchmark accuracy does not establish the reliability or social neutrality of retrieved sources. Generated Python code runs in a sandbox, and tool calls are capped to limit security and resource risks. Like other branch-based methods, CBPO samples several continuations and therefore incurs additional generation cost. A fixed rollout budget bounds this cost, and all methods are compared using matched rollout slots. The experiments evaluate benchmark accuracy rather than deployment safety. High-stakes applications would re quire domain-specific testing, content filtering, access controls, and human oversight.

## References

[1] P. Agrawal, A. Samanta, S. Ghasemlou, B. Vidolov, J. Bhandari, K. Asadi, D. Jiang, and A. Modi. 2026. Of-Context GRPO: Learning to Reason on Hard Problems using Privileged Information. arXiv preprint arXiv:2607.19313. doi:10.48550/arXiv.2607.19313

[2] D. Cheng, S. Huang, X. Zhu, B. Dai, W. X. Zhao, Z. Zhang, and F. Wei. 2025. Reasoning with Exploration: An Entropy Perspective. arXiv preprint arXiv:2506.14758. doi:10.48550/arXiv.2506.14758

[3] K. Cobbe et al. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168. doi:10.48550/arXiv.2110.14168

[4] G. Dong et al. 2025. Agentic Reinforced Policy Optimization. arXiv preprint arXiv:2507.19849. doi:10.4 8550/arXiv.2507.19849

[5] G. Dong et al. 2025. Tool-Star: Empowering LLM-Brained Multi-Tool Reasoner via Reinforcement Learning. arXiv preprint arXiv:2505.16410. doi:10.48550/arXiv.2505.16410

[6] G. Dong, L. Bao, Z. Wang, K. Zhao, X. Li, J. Jin, J. Yang, H. Mao, F. Zhang, K. Gai, G. Zhou, Y. Zhu, J.-R. Wen, and Z. Dou. 2025. Agentic Entropy-Balanced Policy Optimization. arXiv preprint arXiv:2510.14545. doi:10.48550/arXiv.2510.14545

[7] J. Feng et al. 2026. ReTool: Reinforcement Learning for Strategic Tool Use in LLMs. In International Conference on Learning Representations.

[8] L. Feng, Z. Xue, T. Liu, and B. An. 2025. Group-in-Group Policy Optimization for LLM Agent Training. arXiv preprint arXiv:2505.10978. doi:10.48550/arXiv.2505.10978

[9] Z. Gou et al. 2024. ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving. In International Conference on Learning Representations.

[10] D. Guo et al. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv preprint arXiv:2501.12948. doi:10.48550/arXiv.2501.12948

[11] Y. He, H. Wu, S. Liu, H. Ge, H. Zhou, K. Wu, Z. Zheng, Q. Lin, Z. Zhong, and Y. Zhang. 2026. Rethinking Token-Level Credit Assignment in RLVR: A Polarity-Entropy Analysis. arXiv preprint arXiv:2604.11056. doi:10.48550/arXiv.2604.11056

[12] D. Hendrycks et al. 2021. Measuring Mathematical Problem Solving with the MATH Dataset. arXiv preprint arXiv:2103.03874. doi:10.48550/arXiv.2103.03874

[13] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings ofthe 28th International Conference on Computational Linguistics. International Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625. doi:10.18653/v1/2020.coling-main.580

[14] Z. Hou, Z. Hu, Y. Li, R. Lu, J. Tang, and Y. Dong. 2025. TreeRL: LLM Reinforcement Learning with On-Policy Tree Search. In Proceedings of the Annual Meeting of the Association for Computational Linguistics.

[15] J. Hu et al. 2025. REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization. arXiv preprint arXiv:2501.03262. doi:10.48550/arXiv.2501.03262

[16] Y. Ji, Z. Ma, Y. Wang, G. Chen, X. Chu, and L. Wu. 2026. Tree Search for LLM Agent Reinforcement Learning. In International Conference on Learning Representations.

[17] Chuang Jiang, Mingyue Cheng, Xiaoyu Tao, Qingyang Mao, Jie Ouyang, and Qi Liu. 2026. TableMind: An Autonomous Programmatic Agent for Tool-Augmented Table Reasoning. In Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining. ACM, Boise, ID, USA, 260–270. doi:10.1145/3773966.3777932

[18] B. Jin, H. Zeng, Z. Yue, J. Yoon, S. O. Arik, D. Wang, H. Zamani, and J. Han. 2025. Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning. In Conference on Language Modeling.

[19] X. Lai et al. 2024. Step-DPO: Step-Wise Preference Optimization for Long-Chain Reasoning of LLMs. arXiv preprint arXiv:2406.18629. doi:10.48550/arXiv.2406.18629

[20] X. Li, H. Zou, and P. Liu. 2025. ToRL: Scaling Tool-Integrated Reinforcement Learning. arXiv preprint arXiv:2503.23383. doi:10.48550/arXiv.2503.23383

[21] Y. Li et al. 2026. Rethinking On-Policy Distillation of Large Language Models: Phenomenology, Mecha nism, and Recipe. arXiv preprint arXiv:2604.13016. doi:10.48550/arXiv.2604.13016

[22] H. Lightman et al. 2024. Let’s Verify Step by Step. In International Conference on Learning Representations

[23] H. Lin and Z. Xu. 2025. Understanding Tool-Integrated Reasoning. arXiv preprint arXiv:2508.19201. doi:10.48550/arXiv.2508.19201

[24] Z. Lin, T. Liang, J. Xu, X. Wang, R. Luo, C. Shi, S. Li, Y. Yang, and Z. Tu. 2024. Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM’s Reasoning Capability. arXiv preprint arXiv:2411.19943. doi:10.48550/arXiv.2411.19943

[25] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2023. Association for Computational Linguistics, Singapore, 5687– 5711. doi:10.18653/v1/2023.findings-emnlp.378

[26] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. In Advances in Neural Information Processing Systems.

[27] Mandeep Rathee, Sean MacAvaney, and Avishek Anand. 2025. Quam: Adaptive Retrieval through Query Afinity Modelling. In Proceedings ofthe Eighteenth ACM International Conference on Web Search and Data Mining. ACM, Hannover, Germany, 954–962. doi:10.1145/3701551.3703584

[28] J. Schulman et al. 2017. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347. doi:10.48550/arXiv.1707.06347

[29] Z. Shao et al. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv preprint arXiv:2402.03300. doi:10.48550/arXiv.2402.03300

[30] H. Sun, Z. Qiao, J. Guo, X. Fan, Y. Hou, Y. Jiang, P. Xie, Y. Zhang, F. Huang, and J. Zhou. 2025. ZeroSearch: Incentivize the Search Capability of LLMs without Searching. arXiv preprint arXiv:2505.04588. doi:10.4 8550/arXiv.2505.04588

[31] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multi hop Questions via Single-hop Question Composition. Transactions ofthe Association for Computational Linguistics 10 (2022), 539–554. doi:10.1162/tacl\_a\_00475

[32] S. Wang et al. 2025. Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Efective Reinforcement Learning for LLM Reasoning. arXiv preprint arXiv:2506.01939. doi:10.48550/arXiv.2506.01939

[33] J. Wei et al. 2022. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. In Advances in Neural Information Processing Systems.

[34] Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, and Fei Huang. 2025. WebWalker: Benchmarking LLMs in Web Traversal. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Vienna, Austria, 10290–10305. doi:10.18653/v 1/2025.acl-long.508

[35] A. Yang et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388. doi:10.48550/arXiv.2505. 09388

[36] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-1259

[37] S. Yao et al. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

[38] Q. Yu et al. 2025. DAPO: An Open-Source LLM Reinforcement Learning System at Scale. arXiv preprint arXiv:2503.14476. doi:10.48550/arXiv.2503.14476

[39] Jie Zhang, Dongsheng Bi, Tao Sun, Minghui Yang, Jian Wang, and Yiwei Wang. 2026. TOOL-CURE: Tool Selection via Curriculum-Enhanced Reinforcement Learning with Sample Screening for LLMs. In Proceedings ofthe Nineteenth ACM International Conference on Web Search and Data Mining. ACM, Boise, ID, USA, 946–954. doi:10.1145/3773966.3777952

[40] Yue Zhang, Shicheng Zhou, Xiaopeng Li, Zhiliang Tian, Yifu Gao, Shiyi Zhang, Wenqing Hou, Yuying Liu, and Bin Zhou. 2026. KnowFC: Navigating Knowledge Conflicts in Large Language Model-based Fact-Checking. In Proceedings ofthe Nineteenth ACM International Conference on Web Search and Data Mining. ACM, Boise, ID, USA, 996–1006. doi:10.1145/3773966.3777935

[41] C. Zheng, S. Liu, M. Li, X.-H. Chen, B. Yu, C. Gao, K. Dang, Y. Liu, R. Men, A. Yang, J. Zhou, and J. Lin. 2025. Group Sequence Policy Optimization. arXiv preprint arXiv:2507.18071. doi:10.48550/arXiv.2507.18071