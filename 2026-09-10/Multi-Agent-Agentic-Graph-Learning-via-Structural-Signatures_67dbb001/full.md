# Multi-Agent Agentic Graph Learning via Structural Signatures

Liang Qu<sup>1</sup> Member, IEEE, Jianxin Li<sup>1∗</sup> Senior Member, IEEE, Hongzhi Yin<sup>2</sup> Senior Member, IEEE, Hua Wang<sup>3</sup> Fellow, IEEE

Abstract—Agentic graph learning (AGL) has recently achieved promising results on graph reasoning tasks, where an agent powered by a large language model (LLM) sequentially samples the graph as evidence to support its final prediction. Existing methods either employ a single agent or orchestrate multiple role-based agents to reason and learn over the entire graph, but both essentially rely on a shared reasoning policy across different graph regions, which can be suboptimal for graphs with heterogeneous structural and semantic patterns. Inspired by the progress of multi-agent collaboration on complex reasoning tasks, a natural remedy is to let multiple agents own different memory and collaborate; however, applying this paradigm to graphs directly faces two challenges. First, existing AGL methods typically verbalize graph structures into natural-language descriptions for LLM agents, making the reasoning process sensitive to the ordering of structural information and thereby breaking the permutation-invariant nature of graphs. Second, incorporating increasingly large sampled neighborhoods leads to rapidly growing contexts, which not only increases inference cost but also makes important structural evidence vulnerable to the lost-in-the-middle problem. To address these challenges, this paper introduces a multi-agent agentic graph learning (i.e., MAAGL) framework. MAAGL partitions the graph into communities and assigns an independent agent to each community for region-specific specialization. Unlike existing methods that verbalize all sampled evidence into text, MAAGL represents structural and semantic evidence separately. Structural evidence is summarized by a dynamically updated structural signature that is permutation-invariant and fixed in size, while semantic evidence is filtered to the top-k nodes ranked by relevance. Based on historical trajectories with similar signatures, agents estimate their confidence and trigger debate-style collaboration when needed. MAAGL further learns reusable experience from historical trajectories to guide subsequent reasoning. Extensive experiments on four benchmark datasets show that MAAGL outperforms state-of-the-art AGL methods under both in-domain and zero-shot transfer settings while reducing token overhead.

Index Terms—Graph learning, Large Language Models, AI Agents

## I. INTRODUCTION

Text-attributed graphs (TAGs) are widely used to model relational data in many real-world applications [1]–[3]. A TAG consists of structural information and semantic information. The structural information is represented by nodes and edges, where nodes denote entities and edges describe the relations between them. The semantic information is the text attached to each node. For example, in social networks [4], nodes represent users and edges represent their social connections, while user profiles provide the text of the nodes. In citation networks [5], nodes represent papers and edges represent citation relations between papers, while titles, abstracts, and keywords provide the text of the papers. As a result, the core objective of learning on TAGs is to learn from the structural information and the semantic information jointly.

The traditional method to learn from both kinds of information is graph neural networks (GNNs) [6]. A GNN starts from the semantic information of each node, encoded as a feature vector, and passes it along the edges. At each layer, a node aggregates the representations of its neighbors and updates its own. Thus, after several layers, a node representation carries the semantics of its multi-hop neighborhood. Early models such as GCN [7], GraphSAGE [8], and GAT [9] mainly differ in how the neighbors are selected and how their information are aggregated. Recently, graph transformers such as Graphormer [10] and GraphGPS [11] are proposed to add attention over the whole graph for capturing long-range dependencies. Despite their success, these methods share a fundamental limitation, i.e., the same neighborhood selection and aggregation rule is applied uniformly across all nodes. But, in fact, different nodes may require different neighborhood contexts for downstream tasks.

Recently, agentic graph learning (AGL) [14] has emerged as a new paradigm that leverages the planning and reasoning capabilities of LLMs together with graph-specific tools to adaptively identify and collect relevant structural and semantic evidence for downstream graph tasks. Existing AGL methods can be roughly divided into two categories. Single-agent methods [3], [15] employ a single LLM agent to iteratively retrieve graph evidence and reason toward the final prediction. Orchestration-based methods [2], [16] instead decompose this reasoning process across multiple role-specific agents and coordinate them through a predefined workflow, with different agents responsible for different stages of graph reasoning. Despite this architectural difference, both paradigms essentially rely on a shared reasoning policy over the entire graph. However, as the graph becomes larger and more heterogeneous, mixing diverse structural and semantic patterns into a shared policy can be suboptimal.

To address the above issue, a natural solution is to introduce multi-agent collaborative reasoning (e.g., multi-agent debate [17]), which has shown strong effectiveness on complex reasoning tasks by leveraging diverse perspectives from agents with independent memories and reasoning experiences [17], [18]. However, directly applying such methods to graph reasoning tasks is non-trivial and faces two key challenges. (1) Existing multi-agent collaboration is mainly developed for language-based reasoning tasks, where both the problem context and reasoning process are naturally expressed in text, making textual exchange between agents straightforward. In graph reasoning, however, part of the evidence lies in the graph structure, which must first be verbalized into a textual sequence for agent communication, as illustrated in Figure 1 (a). This verbalization imposes an arbitrary order on inherently unordered graph neighborhoods, breaking permutation invariance and making the reasoning sensitive to neighbor ordering. As evidenced by our preliminary results in Figure 1 (c), different orderings of the same neighbors can lead to different predictions. (2) Verbalizing graph structures also causes the context to grow rapidly as the neighborhood expands. In TAGs, each retrieved node brings not only structural information but also its associated textual attributes, all of which need to be included in the textual context. Due to the combinatorial expansion of multi-hop neighborhoods, the resulting context can therefore grow rapidly with each additional hop, leading to substantial token costs, as evidenced by our preliminary results in Figure 1(d).

![](images/8f958869ffc213c2d2950c732f76ff0fe1c0213f385a8c0f48ca7c7551e2ec0e.jpg)

![](images/ebf9ba8aa84d1826421ed9c48f38d079e4d8adb0a405fd781bf8ba93e90032fb.jpg)  
(c) Effect of neighbor order

![](images/4ea53aa63710b4655b44aeff391cc7fc89fe0942240031da2450f3953120dbe6.jpg)  
(d) Token cost  
Fig. 1. (a) Agents reasoning with verbalized graph evidence. (b) Agents reasoning with the proposed structural evidence. (c) Neighbor-order sensitivity of a ReAct-based agent [12] on OGB-Arxiv [13] across five independent runs, where the retrieved neighbors are presented in different random orders. (d) Prompt tokens per query on the same nodes.

In light of the above challenges, this work proposes a multiagent agentic graph learning MAAGL framework. Specifically, to enable agent specialization, MAAGL first partitions the graph into communities and assigns one independent agent to each community. Each agent then performs independent reasoning within its assigned community to build region-specific reasoning experience. Unlike existing methods that verbalize all sampled graph evidence into text, MAAGL represents the evidence returned by each reasoning action in two parts: structural evidence and semantic evidence. For structural evidence, we summarize the observed neighborhood using a structural signature composed of a small set of graph statistics (e.g., node degree and neighborhood label entropy). The signature is dynamically updated over the sampled node set as new evidence is collected, while remaining permutation-invariant and fixed in size. For semantic evidence, MAAGL retains the textual attributes of only the top-k retrieved nodes ranked by their semantic relevance to the target node. In this way, MAAGL preserves useful structural and semantic evidence while avoiding arbitrary neighbor ordering and excessive token costs. During subsequent reasoning, each agent estimates its confidence using the success rate of past trajectories with similar structural signatures. If the confidence is below a threshold, the same score is computed for the other agents, and the top-K agents are selected for debate-style collaborative reasoning. Finally, MAAGL learns from historical reasoning trajectories and stores the learned experiences in agent memory to guide subsequent reasoning. The contributions of this work can be summarized as follows:

• We identify a fundamental mismatch in applying existing multi-agent collaborative reasoning to graph reasoning tasks. Directly verbalizing sampled graph evidence for agent reasoning can break permutation invariance and incur high token costs.

• We propose MAAGL, a multi-agent agentic graph learning framework. MAAGL separates sampled graph evidence into structural and semantic information, representing the former with a dynamically updated structural signature and filtering the latter by semantic relevance.

• We conduct extensive experiments on four benchmark datasets, where MAAGL outperforms AGL methods under both in-domain and zero-shot transfer settings while reducing token overhead.

The rest of this paper is organized as follows. Section 2 reviews related work. We formally define the problem of MAAGL in Section 3 and present its framework with technical details in Section 4. Section 5 demonstrates the experimental evaluation and Section 6 concludes the paper.

## II. RELATED WORK

## A. GNN-based Graph Learning

Graph neural networks (GNNs) [6] are the dominant approach for learning from both structural and semantic information on graphs. A GNN starts from the semantic information of each node, encoded as a feature vector, and propagates it along graph edges through message passing, so that each node gradually aggregates information from its local structure. Different GNN architectures mainly differ in how neighbors are selected and how their information is aggregated. GCN [7] performs normalized neighborhood aggregation, GraphSAGE [8] samples neighbors and applies a learnable aggregator, GAT [9] assigns attention weights to different neighbors, and JK-Net [19] combines representations from multiple layers to capture neighborhoods at different ranges. Later work extends message passing to heterogeneous graphs with different node and edge types [20], [21], while graph transformers [11], [22] introduce global attention to capture long-range dependencies. Another line of work improves their training with graph augmentation. DropEdge [23] randomly removes edges during training, which reduces over-fitting and over-smoothing in deep GNNs. GraphCL [24] perturbs the graph into several contrastive views and learns representations that agree across the views. NodeAug [25] and local augmentation [26] enrich the surroundings of each node, the former by changing its nearby attributes and edges under consistency training, the latter by generating extra neighbor features conditioned on the node.

However, GNN-based graph learning methods generally rely on neighborhood selection and aggregation rules determined by the model architecture and applied uniformly across nodes, which can be suboptimal when different nodes require different graph contexts for downstream tasks.

## B. Agent-based Graph Learning

Large language model (LLM)-powered agents have recently emerged as a promising paradigm for solving complex tasks by integrating planning, reasoning, and tool use within an autonomous loop [12], [27]. This paradigm has also been extended to graph learning, giving rise to agentic graph learning (AGL) [14]. Existing AGL methods mainly follow two lines. Single-agent methods use one LLM agent to interact with the graph and iteratively collect evidence for reasoning. ReaGAN [15] equips an agent with graph sampling tools for retrieving structural and semantic evidence such as neighbor labels, while Graph-CoT [28] lets an agent iteratively invoke graph functions and reason over the returned information. AgentGL [3] further organizes graph reasoning into a thoughtaction-observation loop [12] and optimizes the reasoning policy from collected trajectories, and GraphReAct [29] performs multi-step reasoning and acting for graph inference. Orchestration-based methods instead coordinate multiple rolespecific agents through a predefined workflow. GraphAgent [2] assigns different agents to planning, graph retrieval, and prediction, GraphTeam [30] coordinates role-specialized agents to collaboratively solve graph analysis tasks, GraphCogent [16] decomposes complex graph understanding across multiple collaborating agents, while GraphMaster [31] adopts a similar division of labor for graph data synthesis.

Despite their architectural differences, both single-agent and orchestration-based methods essentially rely on a shared reasoning policy over the entire graph. Although the policy can adaptively collect evidence for individual instances, different graph regions may exhibit distinct structural and semantic patterns. As the graph becomes larger and more heterogeneous, mixing these diverse patterns into a shared policy can therefore be suboptimal.

## C. Multi-agent Collaborative Reasoning

Multi-agent collaborative reasoning has shown promising performance on complex reasoning tasks by combining diverse perspectives from agents with independent memories and reasoning experiences [17], [18]. Existing methods mainly differ in how agents collaborate. Debate-style methods [17] involve multiple agents in the same task and iteratively refine their decisions by exchanging reasoning results: each agent reads the answers and rationales of the others, revises its own over several rounds, and the final decision is reached by consensus or voting. Routing-based methods select only a subset of agents according to their expertise for each task; AgentRouter [32] embeds the incoming task and retrieves the agents whose recorded expertise is semantically closest, and follow-up work studies how to select the best set of collaborators under a cost budget [33]. Other methods enable agents to share past experience through a common memory without direct interaction [34]. An agent writes what it has learned into a shared pool, from which other agents retrieve when they meet similar tasks. These methods are mainly developed for language-based reasoning tasks, where both the task information and reasoning process can be naturally represented and exchanged in text.

Directly applying such collaboration to graph reasoning, however, is non-trivial. Graph reasoning additionally relies on structural evidence, which must be verbalized into text before it can be exchanged between agents. This verbalization imposes an arbitrary order on inherently unordered graph neighborhoods, breaking permutation invariance and making the reasoning sensitive to neighbor ordering. Moreover, as multi-hop neighborhoods expand, verbalizing the sampled nodes together with their textual attributes can quickly increase the context length, leading to high token costs.

## III. PROBLEM FORMULATION

Text-attributed Graph (TAG): We define a TAG [2], [3] as $G = ( V , E , \mathcal { X } , \mathcal { Y } )$ , where V denotes the set of nodes and $E \subseteq V \times V$ denotes the set of edges. Each node $v \in V$ is associated with a textual attribute (e.g., the title and abstract of a paper node in academic citation graphs) $\mathbf { x } _ { v } \in \mathcal { X }$ , where $\mathcal { X }$ denotes the textual attribute space. A subset of nodes $V _ { L } \subseteq V$ is labeled, and each labeled node $v \in V _ { L }$ is associated with a class label $y _ { v } \in \mathcal { V }$ , where Y denotes the label space.

Multi-Agent Agentic Graph Learning: Given a TAG G and a target node v, multi-agent agentic graph learning employs a set of M LLM-powered agents $\mathcal { A } = \{ A _ { 1 } , . . . , A _ { M } \}$ where each agent $A _ { i }$ maintains its own reasoning policy $\pi _ { i }$ and private memory $\mathcal { M } _ { i }$ . We formulate the reasoning of each agent as a sequential decision process. At step $t ,$ agent $A _ { i }$ observes the evidence collected so far, denoted by $s _ { v } ^ { t } ,$ and selects an action $a _ { v } ^ { t } = \pi _ { i } ( s _ { v } ^ { t } , \mathcal { M } _ { i } )$ from $\mathcal { A } _ { \mathrm { { e v i } } } \cup \{ a _ { \mathrm { { p r e d } } } \}$ , where $\mathcal { A } _ { \mathrm { e v i } }$ is a set of evidence-collection actions that sample structural and semantic evidence from $G ,$ and $a _ { \mathrm { p r e d } }$ terminates the process with a prediction $\hat { y } _ { v }$ . Agents may further collaborate on the same target node, and every reasoning process is recorded as a trajectory in the memory of the corresponding agent. Given the labeled node set $V _ { L }$ , the goal is to enrich the agent memories $\{ \mathcal { M } _ { i } \} _ { i = 1 } ^ { M }$ by reasoning on $V _ { L } ,$ , such that the agents, individually or collaboratively, correctly predict the labels of unlabeled nodes without updating any parameter of the underlying LLM.

![](images/e72efbc75e6e94f598a73c381dce1d7677abe2b5bf226e169857f8af016999e0.jpg)  
Fig. 2. The overall architecture of MAAGL.

## IV. THE MAAGL FRAMEWORK

As shown in Figure 2, MAAGL consists of four stages: (a) Agent Assignment (Section IV-A), where the graph is partitioned and one agent is assigned to each region; (b) Agent Specialization (Section IV-B), where each agent independently builds region-specific reasoning experience; (c) Collaborative Reasoning (Section IV-C), where low-confidence cases trigger collaboration among selected agents; and (d) Experience Learning (Section IV-D), where historical trajectories are used to learn reusable reasoning experience.

## A. Agent Assignment

Existing AGL methods, including single-agent methods [3], [15] and orchestration-based methods [2], [16], typically use a shared reasoning policy and memory across the entire graph. As a result, reasoning experience collected from different graph regions is mixed together, even though these regions may exhibit different structural and semantic patterns. To enable agent specialization, we instead maintain multiple independent agents, each with its own policy and memory, and assign each agent to a specific graph region.

A straightforward approach is to randomly split the nodes among agents. However, such a split ignores graph connectivity and may place closely connected nodes into different regions. We therefore use community detection to group densely connected nodes and assign each community to one agent. Formally,

$$
\{ c _ { 1 } , c _ { 2 } , \ldots , c _ { M } \} = \operatorname { P a r t i t i o n } ( G ) ,\tag{1}
$$

where Partition(·) partitions the node set into M disjoint communities, and agent $A _ { i }$ is assigned to community $c _ { i } .$ . In this work, we adopt the Leiden algorithm [35] and further study the effect of different partitioning methods in Section V-F .

## B. Agent Specialization

1) Observation: After agent assignment, each agent first reasons independently within its assigned community to develop region-specific expertise. We formulate this reasoning process as a sequential decision process, where the agent repeatedly takes graph sampling actions and uses the returned evidence to determine its subsequent actions.

A key issue is how the sampled graph evidence is represented to the LLM agent. Existing AGL methods typically verbalize all returned evidence into natural language. For example, after retrieving the 1-hop neighbors of a target node, the node identities, labels, and textual attributes of the sampled neighbors are listed one by one in the prompt. As more graph evidence is sampled, this verbalized context grows accordingly. Example 1 illustrates such a representation.

Example 1 (verbalized evidence).   
“The 1-hop neighbors information is:   
[1] Node $u _ { 1 } ,$ , label $y _ { u _ { 1 } }$ , title and abstract $\cdots$   
[2] Node $u _ { 2 } ,$ label $y _ { u _ { 2 } }$ , title and abstract ...   
[200] Node $u _ { 2 0 0 } ,$ label $y _ { u _ { 2 0 0 } } .$ title and abstract $\cdots$

This representation is problematic for graph reasoning. First, graph neighborhoods are inherently unordered, whereas verbalization necessarily places the sampled neighbors into a sequence. The resulting arbitrary order can therefore affect the LLM’s reasoning. Second, multi-hop neighborhoods can expand rapidly, and verbalizing every sampled node together with its textual attribute introduces substantial token overhead.

To address this issue, unlike existing methods that uniformly verbalize all sampled graph evidence, MAAGL separates the returned evidence into structural evidence and semantic evidence and represents them differently. Structural evidence is summarized by a compact structural signature, while semantic evidence is filtered according to its relevance to the target node. We first introduce the structural signature.

Definition 1 (Structural Signature): The structural signature of a node v at reasoning step t is a vector of d structural statistics,

$$
\boldsymbol { z } _ { v } ^ { t } = [ g _ { 1 } ( v , O _ { v } ^ { t } ) , g _ { 2 } ( v , O _ { v } ^ { t } ) , \dots , g _ { d } ( v , O _ { v } ^ { t } ) ] ,\tag{2}
$$

where $O _ { v } ^ { t }$ denotes the set of sampled nodes observed up to step $t ,$ and each $g _ { j } ( \cdot )$ is a deterministic graph statistic.

In this work, we use six statistics and divide them into static and dynamic dimensions. The four static dimensions describe the structural position of $v$ and remain unchanged during reasoning. Degree measures its local connectivity. PageRank measures its global importance under random walks. The clustering coefficient measures the connectivity among its 1-hop neighbors. The cross-communityfraction measures the fraction of 1-hop neighbors that belong to communities different from that of v in Eq. (1). The remaining two dimensions are updated as the agent samples new evidence. Label entropy measures the diversity of labels among the sampled labeled nodes, while the majority-label fraction measures the proportion of the most frequent label. Both are computed over the labeled nodes in the sampled node set $O _ { v } ^ { t }$ of Definition 1. Whenever a graph action returns new labeled nodes, the dynamic dimensions are recomputed while the static dimensions keep their initial values,

$$
z _ { v } ^ { t } [ j ] = \left\{ \begin{array} { l l } { g _ { j } ( v , O _ { v } ^ { t } ) } & { \mathrm { i f ~ } g _ { j } \mathrm { ~ i s ~ d y n a m i c , } } \\ { z _ { v } ^ { 0 } [ j ] } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{3}
$$

Therefore, the signature evolves with the sampled evidence while remaining fixed in dimensionality.

For interaction with the LLM, the numeric signature is rendered into a short textual form that reports the name and value of each statistic,

$$
\mathrm { t e x t } ( z _ { v } ^ { t } ) = \bigl ( \mathrm { n a m e } ( g _ { j } ) , g _ { j } ( v , O _ { v } ^ { t } ) \bigr ) _ { j = 1 } ^ { d } .\tag{4}
$$

In addition to the signature, the observed label counts over the labeled nodes in $O _ { v } ^ { t }$ , from which the two dynamic dimensions are computed, are also reported to the agent.

For semantic evidence, MAAGL does not retain the textual attributes of all sampled nodes. Instead, the sampled nodes are ranked by their semantic relevance to the target node, measured by the cosine similarity between their text embeddings. Only the top-k textual attributes are retained. Unlike the arbitrary ordering of graph neighbors, this order is meaningful because it directly reflects semantic relevance. Accordingly, the observation of agent $A _ { i }$ for target node $v \in c _ { i }$ at step t is represented as

$$
{ s } _ { v } ^ { t } = \big ( v , \mathbf { x } _ { v } , \mathrm { t e x t } ( z _ { v } ^ { t } ) , \mathcal { X } _ { v } ^ { t , k } \big ) ,\tag{5}
$$

where $\mathbf { x } _ { v }$ is the textual attribute of the target node and $\boldsymbol { \mathcal { X } } _ { v } ^ { t , k }$ denotes the textual attributes of the top-k semantically relevant nodes among the evidence sampled up to step t. Example 2 illustrates the resulting representation.

Example 2 (MAAGL evidence representation).   
Structural evidence:   
Degree = 3   
Neighbor-label entropy = 0.92   
Observed label counts: Label 5: 1, Label 9: 1, Label 23:   
1   
Cross-community fraction = 0.33   
Semantic evidence (top-k):   
1. Node $u _ { 1 } \colon$ title and abstract ...   
2. Node u : title and abstract ...

The structural signature avoids imposing an arbitrary order on the sampled neighborhood. Since every dimension is computed from the sampled node set rather than its enumeration, it satisfies permutation invariance.

Property 1 (Permutation Invariance): For any node v and sampled node set $O _ { v } ^ { t } ,$ , the structural signature $z _ { v } ^ { t }$ is invariant to any permutation of the nodes in $O _ { v } ^ { t }$

Proof: Let $O _ { v } ^ { t } \ = \ \{ u _ { 1 } , . . . , u _ { m } \}$ be the sampled node set at step t, and let $\sigma$ be any permutation of its elements. For the dynamic dimensions, define the label proportion of class $y \in \mathcal { D }$ as $p _ { y } ( O _ { v } ^ { t } ) \ = \ | \{ u \ \in \ O _ { v } ^ { t } \ : \ y _ { u } \ = \ y \} | / | O _ { v } ^ { t } |$ Since permutation does not change the label counts, $p _ { y } ( O _ { v } ^ { t } ) =$ $p _ { y } ( \sigma ( O _ { v } ^ { t } ) )$ for every class $y .$ Hence, both the label entropy $\begin{array} { r } { \begin{array} { r } { \bar { H } ( O _ { v } ^ { t } ) = - \sum _ { y \in \mathcal { V } } p _ { y } ( O _ { v } ^ { t } ) \log p _ { y } ( O _ { v } ^ { t } ) } \end{array} } \end{array}$ and the majority-label fraction $F ( O _ { v } ^ { t } ) \stackrel { \sim } { = } \operatorname* { m a x } _ { y \in \mathcal { Y } } p _ { y } ( O _ { v } ^ { t } )$ are invariant to node ordering. For the static dimensions, degree depends only on $\left| \mathcal { N } _ { 1 } ( v ) \right|$ , clustering coefficient on the edges among $\mathcal { N } _ { 1 } ( v )$ cross-community fraction on the number of neighbors outside the community of $v ,$ and PageRank on the graph topology. None of these quantities depends on how the neighbors are enumerated. Therefore, $g _ { j } ( v , { \cal O } _ { v } ^ { t } ) = g _ { j } ( v , \sigma ( { \cal O } _ { v } ^ { t } ) )$ for every $j = 1 , \ldots , d ,$ so that $z _ { v } ^ { t }$ takes the same value for $O _ { v } ^ { t }$ and $\sigma ( O _ { v } ^ { t } )$ . Thus, the structural signature is permutation-invariant.

At the same time, the dimensionality of $z _ { v } ^ { t }$ remains fixed regardless of how many nodes have been sampled. Together with top-k filtering of semantic evidence, this representation avoids arbitrary neighbor ordering while preventing the observation from growing directly with the sampled neighborhood size.

2) Action: Given the current observation, the agent selects an evidence-collection action to acquire additional information from the graph. Consistent with the two types of evidence defined above, we consider structural evidence obtained from local graph neighborhoods and semantic evidence obtained from text similarity. Accordingly, the evidence-collection action space is

$$
\begin{array} { r } { \mathcal { A } _ { \mathrm { e v i } } = \underbrace { \left\{ a _ { \mathrm { 1 - h o p } } , a _ { \mathrm { 2 - h o p } } \right\} } _ { \mathrm { s t r u c t u r a l ~ s e a r c h } } \cup \underbrace { \left\{ a _ { \mathrm { s e m } } \right\} } _ { \mathrm { s e m a n t i c ~ s e a r c h } } . } \end{array}\tag{6}
$$

Each action returns a set of sampled nodes together with their available labels and textual attributes. As described in the previous subsection, the returned evidence is not directly verbalized. Instead, its structural information is incorporated into the structural signature, while its semantic information is filtered by relevance before being presented to the agent.

Definition 2 (Local r-hop Structural Neighborhood Search): Given a target node v and a radius $r \in \{ 1 , 2 \}$ , the action a<sub>r-hop</sub> samples the r-hop neighborhood

$$
{ \mathcal { N } } _ { r } ( v ) = \{ u \in V \mid d _ { G } ( u , v ) = r \} ,\tag{7}
$$

where $d _ { G } ( \cdot , \cdot )$ denotes the shortest-path distance on G. The sampled nodes provide structural evidence for updating the signature $z _ { v } ^ { t }$ . Their textual attributes are further ranked by semantic relevance to the target node, and only the top-k are retained as semantic evidence in the observation.

Definition 3 (Global Top-k Semantic Neighborhood Search): Given a target node $v ,$ the action $a _ { \mathrm { s e m } }$ retrieves the k nodes whose textual attributes are most semantically similar to $\mathbf { x } _ { v } \colon$

$$
\begin{array} { r } { \textstyle \mathcal { N } _ { \mathrm { s e m } } ^ { k } ( v ) = \mathrm { T o p K } _ { u \in V \backslash \{ v \} } \cos \left( \mathrm { E n c } ( \mathbf { x } _ { v } ) , \mathrm { E n c } ( \mathbf { x } _ { u } ) \right) , } \end{array}\tag{8}
$$

where Enc(·) denotes a text encoder. The textual attributes of these nodes form the semantic evidence, while their available labels are also incorporated into the sampled node set used to update the dynamic dimensions of the structural signature.

3) Prediction: With the observation and evidencecollection actions defined above, each agent independently decides whether to collect more evidence or terminate the reasoning process with a prediction. At step $t ,$ agent $A _ { i }$ selects an action according to its policy,

$$
a _ { v } ^ { t } = \pi _ { i } ( s _ { v } ^ { t } , \mathcal { M } _ { i } ) ,\tag{9}
$$

where $\boldsymbol { s } _ { v } ^ { t }$ is the current observation and $\mathcal { M } _ { i }$ is the private memory of agent $A _ { i }$ . The action is selected from $\mathcal { A } _ { \mathrm { e v i } }$ ∪ $\{ a _ { \mathrm { p r e d } } \}$ , where $\mathcal { A } _ { \mathrm { e v i } }$ contains the evidence-collection actions defined above.

If an evidence-collection action is selected, the returned nodes are added to the sampled node set. The structural signature is then updated by Eq. (3), while the semantic evidence is re-ranked and filtered to retain the top-k relevant textual attributes. The next observation becomes

$$
\begin{array} { r } { s _ { v } ^ { t + 1 } = \big ( v , \mathbf { x } _ { v } , \mathrm { t e x t } ( z _ { v } ^ { t + 1 } ) , \mathcal { X } _ { v } ^ { t + 1 , k } \big ) . } \end{array}\tag{10}
$$

The agent then continues reasoning based on the updated evidence. When $a _ { \mathrm { p r e d } }$ is selected, the reasoning process terminates and agent $A _ { i }$ produces

$$
\begin{array} { r } { \hat { y } _ { v } = A _ { i } \left( s _ { v } ^ { t } , \mathcal { M } _ { i } \right) . } \end{array}\tag{11}
$$

We allow at most $T$ reasoning steps. If the budget is exhausted before $a _ { \mathrm { p r e d } }$ is selected, the final step is used for prediction. We set $T = 3$ in our experiments. Fig. 3 shows the prompt template used at each reasoning step, which presents the structural signature, the filtered semantic evidence, the search history, and the available experiences to the agent.

Adaptive Search Termination. Different instances may require different amounts of graph evidence. Some nodes can be resolved from their own attributes or a small local neighborhood, whereas others require additional structural or semantic evidence. Requiring every instance to perform the same number of searches can therefore introduce redundant evidence collection. Besides increasing token cost, unnecessary searches may bring irrelevant nodes into the reasoning context and interfere with the final prediction.

Inspired by the Search-Constrained Thinking strategy of AgentGL [3], we encourage each agent to stop searching once the collected evidence is sufficient. Rather than learning this behavior through reinforcement learning, we implement it directly through prompting. After every evidence-collection action, the agent is instructed to review the updated evidence and determine whether another search is necessary before selecting its next action.

4) Memory: After reasoning on a labeled training node terminates, the agent stores the resulting trajectory in its private memory. Since the ground-truth label is available during this stage, we assign a binary reward according to whether the prediction is correct,

$$
r _ { v } = \mathcal { H } [ \hat { y } _ { v } = y _ { v } ] .\tag{12}
$$

You are an agent responsible for one community of   
a {graph kind}. Your task is to classify the   
target {item} into one of the candidate   
categories.   
Target {item}: "{node text}"   
Structural signature of the target node, computed   
by the system and updated as evidence is sampled:   
{for each dimension j: name $( g _ { j } ) \ = \ g _ { j } \big ( \boldsymbol { v } , O _ { v } ^ { t } \big )$   
(description); an updated dimension also shows its   
previous value}   
- observed label counts: {label: count, ...}   
Semantic evidence, the sampled nodes most relevant   
to the target (best match first):   
{top-k sampled texts, each with its label}   
Search history:   
{one line per earlier action, with the newly   
observed label counts}   
Experiences available to you:   
{for each experience: IF {metric} > {threshold}   
THEN take the {action} action}   
Candidate categories: [{label list}]   
Choose exactly one action:   
1-hop: sample the 1-hop neighbors of the target   
node.   
2-hop: sample the 2-hop neighborhood; newly   
observed labels update the signature.   
semantic: retrieve the most semantically similar   
labeled nodes in the whole graph.   
predict: output the final label based on the   
current context.   
Respond strictly in JSON: {"thought": "...",   
"action": "1-hop|2-hop|semantic|predict", "label":   
"one candidate category, required only when action   
is predict"}  
Fig. 3. Prompt template for one reasoning step of an agent. Curly braces denote placeholders filled by the system at runtime.

The complete reasoning process is recorded as

$$
\tau _ { v } = \big ( z _ { v } ^ { 0 } , s _ { v } ^ { 0 } , a _ { v } ^ { 0 } , \ldots , s _ { v } ^ { T _ { v } } , a _ { v } ^ { T _ { v } } , \hat { y } _ { v } , y _ { v } , r _ { v } \big ) ,\tag{13}
$$

where $T _ { v } \le T - 1$ denotes the step at which $a _ { \mathrm { p r e d } }$ is selected. We explicitly store the initial structural signature $z _ { v } ^ { 0 }$ with each trajectory, which will later be used to retrieve past cases with similar structural patterns during collaborative reasoning.

After all labeled nodes in community $c _ { i }$ have been processed, agent $A _ { i }$ obtains its trajectory memory

$$
\mathcal { M } _ { i } ^ { \mathrm { t r a j } } = \left\{ \tau _ { v } ~ | ~ v \in V _ { L } \cap c _ { i } \right\} ,\tag{14}
$$

where $V _ { L }$ denotes the set of labeled training nodes. The private memory of agent $A _ { i }$ is written as $\mathcal { M } _ { i } = ( \bar { \mathcal { M } } _ { i } ^ { \mathrm { t r a j } } , \mathcal { M } _ { i } ^ { \mathrm { e x p } } )$ . This stage only populates $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }$ , while $\mathcal { M } _ { i } ^ { \mathrm { e x p } }$ stores the learned experiences produced by the experience learning stage in Section IV-D. Since agents reason only over their assigned communities and do not communicate in this stage, their specialization processes can be performed in parallel. Algorithm 1 summarizes the complete agent specialization process.

## C. Collaborative Reasoning

After agent specialization, each agent has developed its own reasoning experience within the assigned community and stored historical trajectories in its trajectory memory. We next describe how these specialized agents collaborate on cases that are difficult for a single agent.

Algorithm 1 Agent Specialization   
Require: communities $\{ c _ { i } \} _ { i = 1 } ^ { M } ;$ agents $\{ A _ { i } \} _ { i = 1 } ^ { M }$ ; labeled node   
set $V _ { L } ;$ step budget T   
Ensure: trajectory memories $\{ \mathcal { M } _ { i } ^ { \mathrm { t r a j } } \} _ { i = 1 } ^ { M }$   
1: for each agent $A _ { i } , i = 1 , \dotsc , M ,$ in parallel do   
2: $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }  \emptyset$   
3: $\mathcal { M } _ { i } ^ { \mathrm { { e x p } } } \gets \emptyset$   
4: for each node $v \in V _ { L } \cap c _ { i }$ do   
5: initialize sampled node set $O _ { v } ^ { 0 }$ and compute struc  
tural signature $z _ { v } ^ { 0 }$   
6: $\bar { \mathcal { X } } _ { v } ^ { 0 , k } \gets \check { \emptyset }$   
7: $s _ { v } ^ { 0 ^ { \prime } }  ( v , \mathbf { x } _ { v } , \mathrm { t e x t } ( z _ { v } ^ { 0 } ) , \mathcal { X } _ { v } ^ { 0 , k } )$   
8: for $t = 0 , 1 , \dots , T - 1$ do   
9: select $a _ { v } ^ { t } = \pi _ { i } ( s _ { v } ^ { t } , \mathcal { M } _ { i } )$ by Eq. (9) ▷ force   
a<sub>pred</sub> when $t = T - 1$   
10: if $a _ { v } ^ { t } = a _ { \mathrm { p r e d } }$ then   
11: output $\hat { y } _ { v }$ by Eq. (11)   
12: $T _ { v } \gets t$   
13: break   
14: end if   
15: execute $a _ { v } ^ { t }$ to sample additional nodes by Def  
initions 2 and 3   
16: update $O _ { v } ^ { t + 1 }$ with the newly sampled nodes   
17: update structural signature $z _ { v } ^ { t + 1 }$ by Eq. (3)   
18: rank sampled textual evidence and retain   
${ \boldsymbol { \mathcal { X } } _ { v } ^ { t + 1 , k } }$   
19: update observation $s _ { v } ^ { t + 1 }$ by Eq. (10)   
20: review the current evidence before deciding   
whether another search is needed   
21: end for   
22: $r _ { v } \gets \mathbb { H } [ \hat { y } _ { v } = y _ { v } ]$   
23: construct $\tau _ { v }$ by Eq. (13)   
24: $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }  \mathcal { M } _ { i } ^ { \mathrm { t r a j } } \cup \{ \tau _ { v } \}$   
25: end for   
26: end for   
27: return $\{ \mathcal { M } _ { i } ^ { \mathrm { t r a j } } \} _ { i = 1 } ^ { M }$

Collaboration Trigger. After agent $A _ { i }$ produces a prediction for node v, we estimate how reliable the prediction is from its performance on structurally similar cases. Specifically, we retrieve the m trajectories in $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }$ whose stored structural signatures are most similar to the initial signature $z _ { v } ^ { 0 } .$ . The confidence of agent $A _ { i }$ is defined as their similarity-weighted success rate,

$$
\mathrm { c o n f } _ { i } ( v ) = \frac { \sum _ { \tau \in \mathcal { Q } _ { i } ^ { m } ( v ) } \mathrm { s i m } ( z _ { v } ^ { 0 } , z _ { \tau } ) r _ { \tau } } { \sum _ { \tau \in \mathcal { Q } _ { i } ^ { m } ( v ) } \mathrm { s i m } ( z _ { v } ^ { 0 } , z _ { \tau } ) } ,\tag{15}
$$

where $\mathcal { Q } _ { i } ^ { m } ( v ) \subseteq \mathcal { M } _ { i } ^ { \mathrm { t r a j } }$ denotes the retrieved trajectories, $z _ { \tau }$ is the initial structural signature stored with trajectory τ , and $r _ { \tau }$ is its reward. We use cosine similarity for sim $( \cdot , \cdot )$ . A high value indicates that agent $A _ { i }$ has frequently succeeded on nodes with similar structural signatures. Collaboration is triggered when con $\begin{array} { r } { \mathrm { ~  ~ f ~ } _ { i } ( v ) ~ < ~ \delta , } \end{array}$ , where $\delta$ is a predefined threshold.

You are agent {id}, one of several agents   
classifying the same {item} in a {graph kind}. In   
the previous round every agent reasoned   
independently. First review the other agents’   
opinions below, then give your own updated   
prediction independently. You may keep or revise   
your previous answer; do not follow the majority   
blindly, follow the evidence.   
Target {item}: "{node text}"   
Structural signature of the target node:   
{signature dimensions and observed label counts,   
as in the reasoning prompt}   
Previous-round opinions:   
{for each participant: agent id; predicted label;   
brief rationale}   
Candidate categories: [{label list}]   
Respond strictly in JSON: {"thought": "...",   
"label": "one candidate category"}  
Fig. 4. Prompt template for debate rounds after the first round. Each participant reads the previous-round opinions and independently updates its prediction.

Collaborator Selection. When collaboration is triggered, we compute the same confidence score for every other agent using its own trajectory memory. Agents that have performed well on structurally similar cases are preferred as collaborators. Specifically, we select

$$
\mathcal { K } ( v ) = \underset { j \neq i } { \arg \tan } \mathrm { { K } } \mathrm { { c o n f } } _ { j } ( v ) ,\tag{16}
$$

where con $\mathrm { f } _ { j } ( v )$ is computed from $\mathcal { M } _ { j } ^ { \mathrm { t r a j } }$ using Eq. (15). In this way, collaborator selection directly reuses the regionspecific experience accumulated during agent specialization and requires no additional training.

Collaborative Prediction. The owning agent $A _ { i }$ and the selected agents in $\kappa ( v )$ then perform debate-style collaborative reasoning [17]. In the first round, each collaborator receives the target node $\left( v , \mathbf { x } _ { v } \right)$ together with its initial structural signature $z _ { v } ^ { 0 }$ and independently runs the reasoning process described in Section IV-B. Each agent instead collects and represents its own structural and semantic evidence using its private policy and memory, thereby preserving the specialization learned from its assigned community.

After the first round, each participant outputs a prediction together with a short rationale. In the following rounds, the participants read the previous-round predictions and rationales and independently reconsider their own decisions without collecting additional graph evidence. Let $\hat { y } _ { v } ^ { ( j , \ell ) }$ denote the prediction of agent $A _ { j }$ at debate round ℓ. After $D$ rounds, the final prediction is obtained by majority vote,

$$
\hat { y } _ { v } = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } \left| \left\{ j \in \{ i \} \cup K ( v ) : \hat { y } _ { v } ^ { ( j , D ) } = y \right\} \right| .\tag{17}
$$

If multiple labels receive the same number of votes, we choose the prediction from the tied participant with the highest confidence score. Algorithm 2 summarizes the complete collaborative reasoning process. Fig. 4 shows the prompt template used in the debate rounds.

Communication Cost. A direct transfer of conventional multiagent collaboration to graphs would require agents to exchange the sampled graph evidence in textual form. For an h-hop neighborhood, this context grows with the number of sampled nodes and their textual attributes, which can lead to substantial token costs. In MAAGL, structural evidence is communicated through the fixed-dimensional signature $z _ { v } ^ { 0 } ,$ while sampled semantic evidence remains local to each agent. During debate, agents exchange only their predictions and length-bounded rationales. Therefore, the communication cost does not grow with the size of the sampled graph neighborhood.

Algorithm 2 Collaborative Reasoning   
Require: target node v; owning agent $A _ { i }$ ; agents $\{ A _ { j } \} _ { j = 1 } ^ { M } ;$   
trajectory memories $\{ \mathcal { M } _ { j } ^ { \mathrm { t r a j } } \} _ { j = 1 } ^ { M } ;$ threshold $\delta ;$ collabora  
tor number $K ;$ reasoning budget T; debate rounds $D$   
Ensure: final prediction $\hat { y } _ { v }$   
1: initialize $z _ { v } ^ { \mathrm { 0 } }$ and observation $s _ { v } ^ { 0 }$   
2: $A _ { i }$ independently reasons on v using Algorithm 1 and   
outputs $\hat { y } _ { v } ^ { ( i , 1 ) }$   
3: compute con $ { \mathrm { I } _ { i } } ( v )$ from $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }$ by Eq. (15)   
4: if conf $_ i ( v ) \geq \delta$ then   
5: return $\hat { y } _ { v } ^ { ( i , 1 ) }$   
6: end if   
7: for each agent $A _ { j } , j \neq i$ do   
8: compute con $\mathrm { f } _ { j } ( v )$ from $\mathcal { M } _ { j } ^ { \mathrm { t r a j } }$ by Eq. (15)   
9: end for   
10: select collaborators $\kappa ( v )$ by Eq. (16)   
11: for each $A _ { j } \in \mathcal { K } ( v )$ in parallel do   
12: initialize $s _ { v , j } ^ { 0 }$ using $( v , \mathbf { x } _ { v } , z _ { v } ^ { 0 } )$   
13: $A _ { j }$ independently collects and represents its own graph   
evidence   
14: output $\hat { y } _ { v } ^ { ( j , 1 ) }$ with a short rationale   
15: end for   
16: for $\ell = 2 , \ldots , D$ do   
17: for each participant $A _ { j } , j \in \{ i \} \cup \mathcal { K } ( v )$ , in parallel   
do   
18: read the round-(ℓ − 1) predictions and rationales   
19: output revised prediction $\hat { y } _ { v } ^ { ( j , \ell ) }$ with a short ratio  
nale   
20: end for   
21: end for   
22: obtain $\hat { y } _ { v }$ by majority vote using Eq. (17)   
23: return $\hat { y } _ { v }$

## D. Experience Learning

After agent specialization and collaborative reasoning, each agent has accumulated trajectories that record the sampled evidence, reasoning actions, predictions, and corresponding rewards. These trajectories provide more than individual successful or failed cases: they can reveal recurring relations between graph characteristics and effective reasoning strategies. We therefore further extract reusable experience from the trajectory memories and use it to guide subsequent reasoning.

Existing experience-learning methods for LLM agents typically store experience as free-form natural language. For graph reasoning, however, such descriptions can be ambiguous. For example, an experience such as “search for semantic evidence

when nearby labels disagree” does not specify how disagreement is measured or when the recommendation should be applied. Since MAAGL already represents structural evidence through explicitly defined signature metrics, we instead express each experience as a quantitative condition together with a reasoning recommendation.   
Experience Generation. Each agent first learns experience independently from its own trajectory memory. Following the contrastive learning principle of ExpeL [36], agent $A _ { i }$ is provided with successful and failed trajectories from $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }$ together with the names and descriptions of the structural signature metrics. The agent compares these trajectories and identifies structural conditions that are associated with different reasoning outcomes. Each generated experience is represented as   
$\boldsymbol { \epsilon } = ( g , \theta , \rho )$ (18) where $g$ is one of the signature metrics, θ specifies a threshold on that metric, and $\rho$ is a recommended reasoning strategy when the condition $g > \theta$ holds. The generated experiences are stored in the private experience memory $\mathcal { M } _ { i } ^ { \mathrm { e x p } }$ of agent $A _ { i } .$ Together with the trajectory memory, the complete private memory is therefore $\mathcal { N } _ { i } = \mathrm { \bar { \rho } } ( \mathcal { M } _ { i } ^ { \mathrm { t r a j } } , \mathrm { \bar { \mathcal { M } } } _ { i } ^ { \mathrm { e x p } } )$ . The trajectory memory provides concrete past cases, whereas the experience memory provides reusable reasoning guidance distilled from these cases.   
Cross-Agent Experience Validation. An experience learned from one community may reflect only its local graph patterns and may not generalize to other regions. We therefore validate each generated experience using the trajectories of the other agents before making it globally available. Importantly, this validation operates directly on stored trajectories and does not require additional LLM calls.   
Consider an experience $\epsilon = ( g , \theta , \rho )$ generated by agent $A _ { i } .$ For another agent $A _ { j } ,$ , let $T _ { j } ^ { + } ( \epsilon )$ denote the trajectories whose initial structural signatures satisfy $g ( z _ { \tau } ) > \theta$ and which take the recommended action $\rho$ at some step, and let ${ \mathcal { T } } _ { j } ^ { - } ( \epsilon )$ contain trajectories satisfying the same structural condition but never taking $\rho .$ We measure the effectiveness of the recommendation on agent $A _ { j }$ as   
∆<sub>j</sub>(ϵ) = ˆp<sub>j</sub>(r = 1 | T <sup>+</sup>(ϵ)) − pˆ<sub>j</sub>(r = 1 | T <sup>−</sup>(ϵ)) (19) where $\hat { p } _ { j } ( \cdot )$ denotes the empirical success rate over the corresponding trajectories. $\mathbf { A }$ positive $\Delta _ { j } ( \epsilon )$ indicates that, for nodes satisfying the structural condition, trajectories following $\rho$ achieve a higher success rate.   
We admit an experience to the shared experience set $s$ when it is supported by sufficiently many other agents,   
|{j ̸= i : ∆<sub>j</sub>(ϵ) ≥ γ}| ≥ κ (20) where $\gamma$ controls the minimum improvement required from the recommended strategy and κ specifies the minimum number of supporting agents. Experiences that pass this validation are added to $s$ and can be used by all agents in subsequent reasoning, while agent-specific experiences remain in their corresponding private experience memories. Example 3 illustrates the validation process, and Algorithm 3 summarizes the complete experience learning process.

Example 3 (cross-agent validation). Suppose agent $A _ { 1 }$ generates   
the experience ϵ = (label entropy, 0.8, semantic search). For the   
trajectories of agent $A _ { 2 }$ with label entropy above 0.8, those that   
performed semantic search achieve a higher success rate than those   
that did not, giving $\Delta _ { 2 } ( \epsilon ) = 0 . 2 7$ . Similarly, $\Delta _ { 3 } ( \epsilon ) = 0 . 2 3 .$ . If   
$\gamma = 0 . 2$ and $\kappa = 2 ,$ the experience is supported by two other   
agents and is therefore added to the shared experience set $s .$

Algorithm 3 Experience Learning   
Require: trajectory memories $\{ \mathcal { M } _ { i } ^ { \mathrm { t r a j } } \} _ { i = 1 } ^ { M } ;$ experience mem  
ories $\{ \bar { \mathcal { M } } _ { i } ^ { \mathrm { e x p } } \} _ { i = 1 } ^ { \bar { M } }$ ; signature metrics $\{ g _ { 1 } , \ldots , g _ { d } \}$ ; shared   
experience set $s ;$ thresholds $\gamma$ and $\kappa$   
Ensure: updated $\{ \mathcal { M } _ { i } ^ { \mathrm { e x p } } \} _ { i = 1 } ^ { M }$ and $s$   
1: for each agent $A _ { i } , i = 1 , \dotsc , M ,$ , in parallel do   
2: sample successful and failed trajectories from $\mathcal { M } _ { i } ^ { \mathrm { t r a j } }$   
3: generate candidate experiences $I _ { i } = \{ \epsilon = ( g , \bar { \theta , \rho } ) \}$   
from the sampled trajectories   
4: $\mathcal { M } _ { i } ^ { \mathrm { e x p } }  \mathrm { \bar { \mathcal { M } } } _ { i } ^ { \mathrm { e x p } } \cup \mathrm { \bar { \mathcal { I } } } _ { i }$   
5: end for   
6: for each experience $\epsilon \in I _ { i }$ generated by agent $A _ { i }$ do   
7: for each agent $A _ { j } , j \neq i$ do   
8: compute $\Delta _ { j } ( \epsilon )$ from $\mathcal { M } _ { j } ^ { \mathrm { t r a j } }$ by Eq. (19)   
9: end for   
10: if Eq. (20) holds then   
11: ${ \mathcal { S } } \gets { \mathcal { S } } \cup \{ \epsilon \}$   
12: end if   
13: end for   
14: return $\{ \mathcal { M } _ { i } ^ { \mathrm { e x p } } \} _ { i = 1 } ^ { M }$ and $s$

Computation Cost. The structural signature introduces only limited additional computation. The four static dimensions are computed once before reasoning and cached for subsequent episodes. During reasoning, only the two dynamic dimensions need to be updated as new labeled nodes are sampled. Since these dimensions depend on label counts, they can be updated incrementally from the newly collected evidence without recomputing the entire sampled neighborhood. Confidence estimation in Eq. (15) requires comparing the current initial signature with stored trajectory signatures, with a direct cost of $\bar { \boldsymbol { O } } ( \vert \mathcal { M } _ { i } ^ { \mathrm { t r a j } } \vert d )$ for agent $A _ { i }$ . Cross-agent collaborator selection and experience validation reuse the same cached signatures and stored trajectory outcomes and therefore require no additional graph sampling or LLM calls.

## V. EXPERIMENTS

We conduct extensive experiments to answer the following research questions (RQs):

• RQ1: How does MAAGL compare with existing stateof-the-art AGL methods?

• RQ2: How does each core component contribute to the overall performance?

• RQ3: How sensitive is MAAGL to key hyperparameters?

• RQ4: How efficient is MAAGL in terms of reasoning cost?

## A. Datasets

We evaluate MAAGL on four TAGs spanning two domains. For each domain, we use one dataset for in-domain training and evaluation, and a second, disjoint dataset from the same domain to assess zero-shot transfer. In the academic citation domain, nodes represent papers, edges represent citation relations, and the node text is the title and abstract of the paper; we train on OGB-Arxiv [13] and transfer to Corafull [37], [38]. In the e-commerce domain, nodes represent products, edges connect products that are frequently bought together, and the node text is the title and description of the product in OGB-Products and a user review of the product in Amazon-Computers; we train on the OGB-Products subset of TAPE [13], [39] and transfer to Amazon-Computers [40]. The statistics of all datasets are summarized in Table I.

TABLE I  
STATISTICS OF THE DATASETS.
<table><tr><td>Domain</td><td>Dataset</td><td>#Nodes</td><td>#Edges</td><td>#Classes</td><td>Setting</td></tr><tr><td rowspan="2">Academic</td><td>OGB-Arxiv</td><td>169,343</td><td>1,166,243</td><td>40</td><td>In-Domain</td></tr><tr><td>Cora-full</td><td>19,793</td><td>126,842</td><td>70</td><td>Zero-shot</td></tr><tr><td rowspan="2">E-commerce</td><td>OGB-Products</td><td>54,025</td><td>74,420</td><td>47</td><td>In-Domain</td></tr><tr><td>Amazon-Computers</td><td>87,229</td><td>1,256,548</td><td>10</td><td>Zero-shot</td></tr></table>

## B. Baselines

We compare MAAGL against three categories of baselines:

• GNN methods learn node representations by message passing or graph transformers and are trained on each dataset separately.

– GCN [7] aggregates the normalized features of the 1-hop neighbors in every layer.

– GraphSAGE [8] aggregates sampled neighbors with a learnable function.

– GraphGPS [11] combines local message passing with global attention in a modular graph transformer.

• Single-agent methods use a single LLM-powered agent to adaptively collect graph evidence and reason.

– Graph-CoT [28] lets the LLM iteratively invoke graph functions to collect graph information.

– ReaGAN [15] equips an agent with neighborexpansion and semantic retrieval tools, and predicts using the retrieved evidence.

– AgentGL [3] structures graph reasoning into thought-action-observation trajectories and optimizes the agent policy through reinforcement learning.

– GraphReAct [29] performs multi-step reasoning and acting for graph inference.

• Orchestration-based methods coordinate multiple rolespecific agents through a predefined workflow for graph reasoning.

– GraphTeam [30] coordinates role-specialized agents to collaboratively solve graph reasoning tasks.

– GraphAgent [2] organizes multiple role-specific agents into a predefined workflow for graph reasoning.

## C. Experimental Setup

Following AgentGL [3], on the two in-domain training datasets (OGB-Arxiv and OGB-Products) we sample 3,000 training nodes each for optimization, and for each dataset we sample 1,000 nodes from the original test split for evaluation. The supervised baselines (GCN, GraphSAGE, and GraphGPS) are trained on the same 3,000 training nodes, using the node features shipped with each dataset, with model selection on the official validation split; since their label space is tied to the training graph, they have no zero-shot transfer results. For the zero-shot transfer datasets (Cora-full and Amazon-Computers), no training is performed on the target graph and we directly evaluate on 1,000 sampled test nodes: MAAGL reuses the shared experiences and the trajectory memories learned on the in-domain dataset of the same domain, which transfer unchanged because they contain structural signatures, reasoning actions, and correctness outcomes but no label names. MAAGL adopts GPT-4o-mini as the backbone of every agent and is fully training-free: it relies solely on the accumulated trajectories and the learned experiences, without updating any model parameter. Unless stated otherwise, MAAGL partitions the graph with Leiden into M = 10 communities and uses K = 2 collaborators, D = 2 debate rounds, R = 2 learning rounds, and a confidence threshold δ set to the 25th percentile of the training confidences.

TABLE II  
OVERALL NODE CLASSIFICATION PERFORMANCE (ACCURACY AND MACRO-F1) UNDER THE IN-DOMAIN AND ZERO-SHOT TRANSFER SETTINGS.
<table><tr><td rowspan="2" colspan="2">Settings Datasets</td><td colspan="4">In-Domain</td><td colspan="4">Zero-shot Transfer</td></tr><tr><td colspan="2">OGB-Arxiv</td><td colspan="2">OGB-Products</td><td colspan="2">Cora-full</td><td colspan="2">Amazon-Comp.</td></tr><tr><td>Category</td><td>Methods</td><td>Acc</td><td>Ma-F1</td><td>Acc</td><td>Ma-F1</td><td>Acc</td><td>Ma-F1</td><td>Acc</td><td>Ma-F1</td></tr><tr><td rowspan="3">GNN Methods</td><td>GCN [7]</td><td>0.6460</td><td>0.3521</td><td>0.6710</td><td>0.4014</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td>GraphSAGE [8]</td><td>0.6210</td><td>0.3766</td><td>0.6610</td><td>0.4151</td><td></td><td></td><td>一</td><td></td></tr><tr><td>GraphGPS [11]</td><td>0.5530</td><td>0.2044</td><td>0.5170</td><td>0.2635</td><td></td><td>一</td><td>一</td><td></td></tr><tr><td rowspan="4">Single-Agent Methods</td><td>Graph-CoT [28]</td><td>0.6240</td><td>0.4165</td><td>0.3690</td><td>0.2902</td><td>0.5780</td><td>0.5399</td><td>0.8340</td><td>0.8657</td></tr><tr><td>ReaGAN [15]</td><td>0.6390</td><td>0.4764</td><td>0.7210</td><td>0.5074</td><td>0.6447</td><td>0.5974</td><td>0.7830</td><td>0.8306</td></tr><tr><td>AgentGL [3]</td><td>0.6550</td><td>0.4851</td><td>0.6860</td><td>0.5011</td><td>0.6320</td><td>0.5850</td><td>0.7100</td><td>0.6495</td></tr><tr><td>GraphReAct [29]</td><td>0.6320</td><td>0.4389</td><td>0.6650</td><td>0.4821</td><td>0.6260</td><td>0.5624</td><td>0.8110</td><td>0.8179</td></tr><tr><td rowspan="2">Orchestration-based Methods</td><td>GraphTeam [30]</td><td>0.6200</td><td>0.4487</td><td>0.5990</td><td>0.3909</td><td>0.5950</td><td>0.5457</td><td>0.6180</td><td>0.5500</td></tr><tr><td>GraphAgent [2]</td><td>0.6531</td><td>0.4652</td><td>0.6730</td><td>0.4768</td><td>0.6170</td><td>0.5762</td><td>0.7110</td><td>0.6856</td></tr><tr><td>Ours</td><td>MAAGL</td><td>0.6690</td><td>0.4832</td><td>0.7650</td><td>0.5588</td><td>0.6590</td><td>0.6035</td><td>0.8670</td><td>0.8902</td></tr></table>

## D. Overall Performance (RQ1)

We first evaluate the overall performance of MAAGL against all baselines on node classification, under both the in-domain and the zero-shot transfer settings. The results are reported in Table II, where baselines are grouped by category and our method is listed at the bottom. We summarize the key observations below.

• MAAGL achieves the best Accuracy on all four datasets. For Macro-F1, it achieves the best results on OGB-Products and Amazon-Comp., while remaining competitive with the best baselines on OGB-Arxiv and Corafull. These results demonstrate the overall effectiveness of MAAGL. A possible reason is that region-specific specialization allows each agent to learn from structurally and semantically similar cases, while selective collaboration further introduces complementary reasoning when the owning agent is less reliable.

• Agent-based methods generally outperform conventional GNN methods, particularly in terms of Macro-F1. This advantage is also reflected in the zero-shot transfer setting, where trained GNN models cannot be directly transferred to unseen graphs, whereas agent-based methods remain applicable without retraining. This observation supports the motivation of agentic graph learning, where LLM agents can adaptively collect graph evidence and exploit textual semantics for different instances, providing stronger generalization across graphs.

Existing orchestration-based methods do not consistently outperform single-agent methods. For example, GraphAgent performs competitively on OGB-Arxiv but falls behind strong single-agent methods on the other datasets. This observation is consistent with our motivation that simply organizing multiple agents does not fundamentally address the limitation of applying a shared reasoning policy across different graph regions. In contrast, MAAGL consistently improves upon both categories by allowing agents to specialize in different regions and collaborate selectively when complementary experience is needed.

## E. Ablation Study (RQ2)

To understand the contribution of each core component, we compare MAAGL with four ablated variants on OGB-Arxiv, each removing or replacing exactly one design choice: (i) w/o structural signature, which verbalizes the retrieved neighbors as raw text in the ReaGAN style and removes the structural signature from the observation, while the numeric signature is still used for the collaboration trigger and collaborator selection so that only the representation effect is isolated; (ii) w/ free-text experience, which replaces the structured experiences with free-text experiences in natural language kept in each agent’s private experience memory, without cross-agent validation; (iii) w/o collaboration, where the owning agent always answers alone; and (iv) w/ max-confidence merge, which keeps collaboration but replaces the debate and the vote with the single prediction of the most confident participant. Figure 5 reports Accuracy on node classification.

![](images/30c143afaa1b0dd62f2e1625ca99dc8c5b57168d3f826c5ee44ae03f8a4dd838.jpg)  
Fig. 5. Ablation study on node classification on OGB-Arxiv.

• Removing the structural signature causes the largest performance degradation. This confirms the importance of separating structural and semantic evidence instead of directly verbalizing sampled neighborhoods, supporting our motivation that raw textual serialization is unsuitable for graph reasoning.

• Replacing the structured experience with free-text experience consistently weakens performance. This suggests that expressing experience through explicit structural conditions and recommendations provides more reliable guidance than unconstrained natural-language lessons.

• Removing collaboration also degrades performance, showing that independently specialized agents are not sufficient for all instances. Selective collaboration allows agents to incorporate complementary experience from other regions when their own reasoning is less reliable.

• Replacing debate-based collaboration with maxconfidence merging leads to a smaller performance drop. This indicates that simply selecting the most confident prediction cannot fully exploit the complementary reasoning of multiple agents, while iterative exchange and refinement provides additional benefit.

## F. Hyperparameter Sensitivity (RQ3)

We study the sensitivity of MAAGL to four key hyperparameters: (a) the choice of community detection algorithm including Leiden, Louvain, and random partition. (b) the number of communities $M \in \{ 5 , 1 0 , 2 0 \} )$ ). (c) the number of debate rounds $( D \in \{ 1 , 2 , 3 \} )$ . (d) the confidence threshold that triggers collaboration (δ set to the 10th, 25th, or 40th percentile of the training confidences). Figure 6 reports node classification Accuracy as each hyperparameter varies while the others are fixed at their default values. We also study the effect of the number of learning rounds (from the warm-up only up to $R \ = \ 4 )$ , in which collaborative reasoning and experience learning alternate. Figure 7 reports the accuracy on the training set of OGB-Arxiv as more learning rounds are performed.

• MAAGL is relatively insensitive to the choice of community detection algorithm. Leiden, Louvain, and random partition all achieve comparable performance, while community-based partitioning provides a small advantage. This suggests that the benefit mainly comes from assigning different graph regions to independent agents rather than relying on a particular partitioning algorithm.

![](images/fb8fe2fb4f4c1efb3e167a04dfdbf3eac0a7b96e06c3bcfb4f96526accb3a607.jpg)

![](images/ee13cea66730ac24ab399c3e990e0194f217c187d0def732e7d3084bcd26caaa.jpg)  
(a) Community detection al- (b) Number of communities gorithm

![](images/057e24d6f0e8a646fa5510ad23f48f9887ddfcfa969a9fa9ded1b88482f99b45.jpg)

![](images/d3f4b106d9d164e01fa69f1054ce9ad5669bae171ef1257393fbfcfd25ee27fc.jpg)  
(c) Number of debate (d) Confidence threshold rounds

Fig. 6. Hyperparameter sensitivity of MAAGL on OGB-Arxiv.  
![](images/80703e96e62751022124b3b4b4b3a23d54ca7a2b5d65a34aab9c04835670722f.jpg)  
Fig. 7. Accuracy on the learning set of OGB-Arxiv across learning rounds.

• The number of communities has a clear effect on performance. Too few communities limit agent specialization, while too many communities divide the graph into overly small regions and reduce the experience available to each agent. A moderate number of communities provides a better balance between specialization and sufficient local experience.

• A small number of debate rounds is sufficient for effective collaboration. Increasing the number of rounds initially improves performance by allowing agents to reconsider their predictions using complementary opinions, whereas further debate provides little additional benefit and may introduce unnecessary reconsideration.

• A moderate confidence threshold achieves the best performance. A low threshold triggers collaboration for too few cases, while a high threshold invokes collaboration even when the owning agent is already reliable. This supports selectively introducing collaboration for uncertain instances.

![](images/f0ff9c11d21f759437b18655032cb190038eb07a6d9ee1066c49b365f5a9ff5e.jpg)  
Fig. 8. Tokens per cross-agent message on the four datasets.

![](images/f5e26c87e17e6829fa55f513de7542ff1deaf15f03684fe13d55dfadf7a7b64f.jpg)  
Fig. 9. Average number of search actions per episode on each dataset, with and without the Adaptive Search Termination.

• Performance improves consistently over successive learning rounds and gradually approaches saturation. This indicates that agents can progressively accumulate useful experience from historical reasoning trajectories, while the diminishing improvement in later rounds suggests convergence of the learning process.

## G. Efficiency Analysis (RQ4)

We analyze the two efficiency designs of MAAGL on all four datasets. Figure 8 compares the number of tokens one cross-agent message carries, when the message serializes the 1-hop or 2-hop neighborhood and when it carries only the structural signature, measured on the 1,000 test nodes of each dataset with 200 tokens per node text. Figure 9 reports the average number of search actions per episode on each dataset, with and without adaptive search termination.

• The structural signature substantially reduces the token cost of cross-agent communication across all datasets. Unlike serialized neighborhoods, whose size grows rapidly with the number of retrieved nodes and neighborhood depth, the signature remains compact and fixed in size. This confirms that separating structural evidence from raw textual content provides a more efficient representation for multi-agent communication.

• Adaptive search termination consistently reduces the number of search actions across all datasets. This indicates that many instances can be resolved without exhausting the full search budget, and allowing the agent to stop once sufficient evidence has been collected avoids unnecessary graph exploration and improves reasoning efficiency.

## VI. CONCLUSION

In this work, we revisited multi-agent collaboration for agentic graph learning and identified two key limitations in directly transferring existing collaborative reasoning methods to graphs. First, existing AGL methods generally rely on a shared reasoning policy across different graph regions, which can be suboptimal when these regions exhibit different structural and semantic patterns. Second, communicating graph evidence through natural-language serialization introduces arbitrary neighbor ordering and rapidly increasing token costs. To address these issues, we proposed MAAGL, which assigns independent agents to different graph communities for regionspecific specialization and represents sampled structural and semantic evidence separately. Structural evidence is summarized by a permutation-invariant structural signature, while semantic evidence is filtered according to relevance. Historical trajectories are further used to estimate agent confidence, select complementary collaborators, and learn reusable reasoning experience. Extensive experiments on four benchmark datasets demonstrate that MAAGL consistently improves over existing AGL methods.

One limitation of the current framework is that the dimensions of the structural signature are manually specified based on commonly used graph statistics. An important direction for future work is therefore to automatically discover or select task-specific structural signature dimensions from graph data and reasoning trajectories.

## ACKNOWLEDGMENT

This work is supported by the Australian Research Council under the Discovery Project scheme (No.DP240101591).

## REFERENCES

[1] F. Xia, C. Peng, J. Ren, F. G. Febrinanto, R. Luo, V. Saikrishna, S. Yu, and X. Kong, “Graph learning,” Foundations and Trends® in Signal Processing, vol. 19, no. 4, pp. 362–519, 2026.

[2] Y. Yang, J. Tang, L. Xia, X. Zou, Y. Liang, and C. Huang, “Graphagent: Agentic graph language assistant,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 26 360–26 379.

[3] Y. Sun, K. Li, D. Fan, J. Liu, and Q. Tan, “Agentgl: Towards agentic graph learning with llms via reinforcement learning,” arXiv preprint arXiv:2604.05846, 2026.

[4] Q. Tan, N. Liu, and X. Hu, “Deep representation learning for social network analysis,” Frontiers in big Data, vol. 2, p. 2, 2019.

[5] J. Gao, J. Wu, and J. Ding, “Heterogeneous graph condensation,” IEEE Transactions on Knowledge and Data Engineering, vol. 36, no. 7, pp. 3126–3138, 2024.

[6] Z. Wu, S. Pan, F. Chen, G. Long, C. Zhang, and P. S. Yu, “A comprehensive survey on graph neural networks,” IEEE transactions on neural networks and learning systems, vol. 32, no. 1, pp. 4–24, 2020.

[7] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” arXiv preprint arXiv:1609.02907, 2016.

[8] W. Hamilton, Z. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” Advances in neural information processing systems, vol. 30, 2017.

[9] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Lio, and Y. Ben-´ gio, “Graph attention networks,” arXiv preprint arXiv:1710.10903, 2017.

[10] C. Ying, T. Cai, S. Luo, S. Zheng, G. Ke, D. He, Y. Shen, and T.-Y. Liu, “Do transformers really perform badly for graph representation?” Advances in neural information processing systems, vol. 34, pp. 28 877– 28 888, 2021.

[11] L. Rampa´sek, M. Galkin, V. P. Dwivedi, A. T. Luu, G. Wolf, andˇ D. Beaini, “Recipe for a general, powerful, scalable graph transformer,” Advances in Neural Information Processing Systems, vol. 35, pp. 14 501–14 515, 2022.

[12] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao, “React: Synergizing reasoning and acting in language models,” arXiv preprint arXiv:2210.03629, 2022.

[13] W. Hu, M. Fey, M. Zitnik, Y. Dong, H. Ren, B. Liu, M. Catasta, and J. Leskovec, “Open graph benchmark: Datasets for machine learning on graphs,” Advances in neural information processing systems, vol. 33, pp. 22 118–22 133, 2020.

[14] Y. Bei, W. Zhang, S. Wang, W. Chen, S. Zhou, H. Chen, Y. Li, J. Bu, S. Pan, Y. Yu et al., “Graphs meet ai agents: Taxonomy, progress, and future opportunities,” arXiv preprint arXiv:2506.18019, 2025.

[15] M. Guo, X. Zhu, H. Xue, C. Zhang, S. Lin, J. Huang, Z. Ye, and Y. Zhang, “Reagan: Node-as-agent-reasoning graph agentic network,” arXiv preprint arXiv:2508.00429, 2025.

[16] R. Wang, S. Liang, Q. Chen, Y. Huang, M. Li, Y. Ma, D. Zhang, K. Qin, and M.-F. Leung, “Graphcogent: Mitigating llms’ working memory constraints via multi-agent collaboration in complex graph understanding,” in Proceedings ofthe ACM Web Conference 2026, 2026, pp. 3811–3822.

[17] Y. Du, S. Li, A. Torralba, J. B. Tenenbaum, and I. Mordatch, “Improving factuality and reasoning in language models through multiagent debate, 2023,” URL https://arxiv. org/abs/2305.14325, vol. 3, 2023.

[18] G. Hao, Y. Long, and Z. Zhao, “Self-evolving multi-agent systems via decentralized memory,” arXiv preprint arXiv:2605.22721, 2026.

[19] K. Xu, C. Li, Y. Tian, T. Sonobe, K.-i. Kawarabayashi, and S. Jegelka, “Representation learning on graphs with jumping knowledge networks,” in International conference on machine learning. pmlr, 2018, pp. 5453– 5462.

[20] X. Wang, H. Ji, C. Shi, B. Wang, Y. Ye, P. Cui, and P. S. Yu, “Heterogeneous graph attention network,” in The world wide web conference, 2019, pp. 2022–2032.

[21] X. Fu, J. Zhang, Z. Meng, and I. King, “Magnn: Metapath aggregated graph neural network for heterogeneous graph embedding,” in Proceedings of the web conference 2020, 2020, pp. 2331–2341.

[22] S. Yun, M. Jeong, R. Kim, J. Kang, and H. J. Kim, “Graph transformer networks,” Advances in neural information processing systems, vol. 32, 2019.

[23] Y. Rong, W. Huang, T. Xu, and J. Huang, “Dropedge: Towards deep graph convolutional networks on node classification,” arXiv preprint arXiv:1907.10903, 2019.

[24] Y. You, T. Chen, Y. Sui, T. Chen, Z. Wang, and Y. Shen, “Graph contrastive learning with augmentations,” Advances in neural information processing systems, vol. 33, pp. 5812–5823, 2020.

[25] Y. Wang, W. Wang, Y. Liang, Y. Cai, J. Liu, and B. Hooi, “Nodeaug: Semi-supervised node classification with data augmentation,” in Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 207–217.

[26] S. Liu, R. Ying, H. Dong, L. Li, T. Xu, Y. Rong, P. Zhao, J. Huang, and D. Wu, “Local augmentation for graph neural networks,” in International conference on machine learning. PMLR, 2022, pp. 14 054–14 072.

[27] A. Yehudai, L. Eden, A. Li, G. Uziel, Y. Zhao, R. Bar-Haim, A. Cohan, and M. Shmueli-Scheuer, “Survey on evaluation of llm-based agents,” arXiv preprint arXiv:2503.16416, 2025.

[28] B. Jin, C. Xie, J. Zhang, K. K. Roy, Y. Zhang, Z. Li, R. Li, X. Tang, S. Wang, Y. Meng et al., “Graph chain-of-thought: Augmenting large language models by reasoning on graphs,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 163–184.

[29] X. Yu, Z. Kuai, C. Zhou, X. Xie, R. Jiang, X. Zhang, H. Cheng, X. Zhang, and Y. Fang, “Graphreact: Reasoning and acting for multi-step graph inference,” arXiv preprint arXiv:2605.07357, 2026.

[30] X. Li, Q. Chu, Y. Chen, Y. Liu, Y. Liu, Z. Yu, W. Chen, C. Qian, C. Shi, and C. Yang, “Graphteam: Facilitating large language modelbased graph analysis via multi-agent collaboration,” arXiv preprint arXiv:2410.18032, 2024.

[31] E. Du, X. Li, T. Jin, Z. Zhang, R.-H. Li, and G. Wang, “Graphmaster: Automated graph synthesis via llm agents in data-limited environments,” arXiv preprint arXiv:2504.00711, 2025.

[32] Z. Zhang, K. Shi, Z. Yuan, Z. Wang, T. Ma, K. Murugesan, V. Galassi, C. Zhang, and Y. Ye, “Agentrouter: A knowledge-graph-guided llm

router for collaborative multi-agent question answering,” arXiv preprint arXiv:2510.05445, 2025.

[33] J. Wang, S. Zhao, H. Wang, Y. Fan, L. Zhang, Y. Liu, and T. Liu, “Optimal-agent-selection: State-aware routing framework for efficient multi-agent collaboration,” arXiv preprint arXiv:2511.02200, 2025.

[34] H. Gao and Y. Zhang, “Memory sharing for large language model based agents,” arXiv preprint arXiv:2404.09982, 2024.

[35] V. A. Traag, L. Waltman, and N. J. Van Eck, “From louvain to leiden: guaranteeing well-connected communities,” Scientific reports, vol. 9, no. 1, p. 5233, 2019.

[36] A. Zhao, D. Huang, Q. Xu, M. Lin, Y.-J. Liu, and G. Huang, “Expel: Llm agents are experiential learners,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 17, 2024, pp. 19 632–19 642.

[37] A. Bojchevski and S. Gunnemann, “Deep gaussian embedding of ¨ graphs: Unsupervised inductive learning via ranking,” arXiv preprint arXiv:1707.03815, 2017.

[38] O. Shchur, M. Mumme, A. Bojchevski, and S. Gunnemann, “Pit-¨ falls of graph neural network evaluation, 2018,” arXiv preprint arXiv:1811.05868, 1811.

[39] X. He, X. Bresson, T. Laurent, A. Perold, Y. LeCun, and B. Hooi, “Harnessing explanations: Llm-to-lm interpreter for enhanced text-attributed graph representation learning,” in International conference on learning representations, vol. 2024, 2024, pp. 5711–5732.

[40] H. Yan, C. Li, R. Long, C. Yan, J. Zhao, W. Zhuang, J. Yin, P. Zhang, W. Han, H. Sun et al., “A comprehensive study on text-attributed graphs: Benchmarking and rethinking,” Advances in Neural Information Processing Systems, vol. 36, pp. 17 238–17 264, 2023.