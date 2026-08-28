# SIGMA: Structured Noise-Efect-Aware Grouped Multi-Agent Aggregation

Mingqian Li

School of Computer Science and technology, Tongji University

Shanghai, China

mingqianli071@gmail.com

## Abstract

Cooperative multi-agent reinforcement learning (MARL) faces significant challenges in maintaining robust coordination under noisy observations. Although observation disturbances are often introduced independently across agents, their downstream efects on cooperative decision-making can become structured through underlying cooperation structures. We characterize this phenomenon as structured noise effects, where noise-induced decision efects exhibit local correlation among agents with stronger task-related dependencies while remaining globally heterogeneous across diferent agents and local structures. Existing robust MARL methods, however, rarely explicitly characterize or exploit such structure-dependent noise efects. To address this limitation, we propose SIGMA, a hierarchical collaboration framework that exploits cooperation structures to learn robust representations under noisy observations. SIGMA first organizes agents into adaptive local structures through density-based grouping and performs intra-group consensus aggregation to preserve shared task-relevant information while smoothing agentspecific representation deviations. Inter-group attention then adaptively integrates information across diferent groups to preserve global coordination while accommodating their heterogeneous contributions. Experiments on noisy-observation tasks in StarCraft II empirically validate the structured noise efects and demonstrate that SIGMA consistently improves robustness under observation noise while maintaining competitive performance in noise-free environments.

## Introduction

Cooperative multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex tasks that require coordination among multiple agents, with successful applications in multi-agent games (Samvelyan et al. 2019), multi-robot coordination (Sadhu and Konar 2018; Hu et al. 2023), and autonomous driving (Li et al. 2023). However, most existing MARL methods assume that agents can obtain reliable observations during decision-making (Rashid et al. 2018; Yu et al. 2022; Wen et al. 2022), an assumption that may not hold in real-world deployments where sensing errors, communication limitations, and environmental disturbances are inevitable (Muratore, Gienger, and Peters 2019). Noisy observations can distort agents’ local perceptions and subsequently afect their decisions, potentially disrupting coordination and degrading overall team performance (Li et al. 2022; Liu et al. 2022; Zhang et al. 2020b; Yang et al. 2023).

Noisy local observations can prevent agents from obtaining reliable information about the environment, thereby hindering cooperative policy learning (Muratore, Gienger, and Peters 2019). Beyond individual perception errors, observation uncertainty can further interfere with cooperative interactions, as unreliable local information may afect how agents coordinate with their teammates (Kilinc and Montana 2018). More importantly, the resulting influence is not necessarily confined to the agents whose observations are directly perturbed; perturbing only a single agent can substantially degrade the performance of the entire cooperative team (Lin et al. 2020). Such team-level efects are also non-uniform across agents, as comparable observation impairments can lead to diferent cooperative outcomes depending on which agents are afected (Barta, Nagy, and Gulyás 2025).

We characterize these non-independent impacts as structured noise efects, which exhibit two complementary properties: local correlation and global heterogeneity. At the local level, agents with stronger task-related dependencies tend to exhibit more correlated noise-induced decision impacts, reflecting the coupling of disturbance efects through local cooperative interactions. At the global level, the magnitude and pattern of these impacts can vary substantially across agents and local cooperation structures, resulting in heterogeneous disturbance responses across the team. Together, these two properties indicate that the downstream efects of observation noise are shaped by cooperative dependencies among agents: although disturbances are introduced independently, their impacts become structured through the underlying cooperation structures.

Since cooperation structures influence how information is exchanged and decisions are coupled among agents, a line of research has focused on modeling agent relationships in cooperative MARL.Early MARL approaches mainly relied on the centralized training and decentralized execution (CTDE) paradigm(Amato 2024). Although value decomposition like QMIX(Rashid et al. 2018; Son et al. 2019)and centralized critic approaches such as MADDPG(Lowe et al. 2017) have achieved remarkable success, the dependencies among agents are usually captured implicitly through learned value functions or centralized representations.

To overcome this limitation, subsequent studies began to explicitly model relationships among agents through learned coordination structures. Role discovery(Lhaksmana, Murakami, and Ishida 2018; Wang et al. 2020) and grouping-based(Russell and Zimdars 2003; Phan et al. 2021) approaches identify functional specialization or local cooperation units among agents. More recently, graphbased(Malysheva et al. 2018; Jiang et al. 2018; Zang et al. 2023) like MAGNet relational models have been widely adopted to capture dynamic interactions among agents. Graph neural networks model pairwise dependencies(Wang et al. 2021), while attention mechanisms enable adaptive interaction weighting. Hypergraph-based(Feng et al. 2019; Liu and Li 2025) approaches extend these ideas to higherorder interactions among multiple agents. These methods substantially improve the modeling of complex cooperation patterns in MARL. Despite these advances, they mainly focus on learning efective coordination structures and rarely investigate how cooperation structures shape the efects of observation noise.

Other studies focus on investigates robustness learning under noisy observations. Recent studies have explored several strategies to alleviate the adverse efects of observation noise. MADDPG-M(Kilinc and Montana 2018) enhances robustness by introducing a communication mechanism that enables agents to access additional information from teammates. However, this approach relies on an additional communication medium, which may not always be available in practical scenarios. (Lin et al. 2020) investigate the vulnerability of cooperative MARL algorithms under noisy observations, but they do not provide an explicit robustness learning mechanism. More recent approaches improve robustness through single agent perturbations defense (Zhang et al. 2020a), learn from adversarial attack(Zhang et al. 2021), and observation discretizations(Fu et al. 2024). However, they typically model noise as independent uncertainty associated with individual agents and overlook how cooperation relationships among agents influence the propagation and accumulation of noise efects.

To bridge this gap, we propose SIGMA, a hierarchical collaboration framework that explicitly exploits cooperation structures to improve robustness under noisy observations. Motivated by the locally correlated nature of structured noise efects, SIGMA first organizes agents into adaptive local structures through density-based grouping, providing meaningful structural units for subsequent aggregation. Within each group, high-order interactions among agents are modeled and their representations are aggregated through intragroup consensus, which preserve shared task-relevant information while smoothing agent-specific representation deviations. Given the global heterogeneity of structured noise effects, SIGMA further employs inter-group attention to adaptively model dependencies among diferent groups and integrate group-level information, thereby preserving crossgroup coordination. Through this local-to-global collaboration process, SIGMA learns robust cooperative representations while preserving both local structural information and global task coordination.

The main contributions of this work are summarized as

follows:

• We characterize structured noise efects in cooperative MARL, revealing that independently introduced observation disturbances can induce locally correlated and globally heterogeneous efects on cooperative decisionmaking through underlying cooperation structures.

• We propose SIGMA, a hierarchical collaboration framework that exploits structured noise efects through adaptive grouping and local-to-global representation aggregation. SIGMA consolidates cooperative information within adaptive local structures and further models dependencies across groups, enabling robust representation learning while preserving global coordination.

• We conduct extensive experiments on noisy-observation tasks in SMAC to empirically validate the proposed structured noise efects and evaluate the robustness of SIGMA. The results demonstrate consistent robustness improvements under observation noise while maintaining competitive performance in noise-free environments.

## Methodology

## Problem Formulation

We consider a cooperative multi-agent reinforcement learning (MARL) problem with noisy observations, which can be formulated as a decentralized partially observable Markov decision process (Dec-POMDP). The environment is defined as:

$$
\mathcal { M } = \langle \mathcal { N } , \mathcal { S } , \mathcal { A } , P , R , \Omega , \gamma \rangle ,\tag{1}
$$

where $\mathcal { N } = \{ 1 , \dots , N \}$ denotes the set of agents, $s$ represents the global state space, $\mathcal { A } = \{ A _ { 1 } , \dotsc , A _ { N } \}$ denotes the joint action space, $P$ represents the state transition function, R denotes the shared reward function, $\Omega = \{ \Omega _ { 1 } , \ldots , \Omega _ { N } \}$ represents the observation spaces, and $\gamma$ is the discount factor.

At timestep t, each agent i receives a local observation generated from the underlying environment state:

$$
o _ { i } ^ { t } = O _ { i } ( s ^ { t } ) ,\tag{2}
$$

where $s ^ { t } \in S$ denotes the global state. However, real-world environments inevitably introduce observation uncertainties due to sensing errors, communication limitations, and environmental disturbances. Therefore, the actual observation received by agent i is formulated as:

$$
\widetilde { o } _ { i } ^ { t } = o _ { i } ^ { t } + \epsilon _ { i } ^ { t } ,\tag{3}
$$

where $\boldsymbol { \epsilon } _ { i } ^ { t }$ denotes the observation disturbance. Following common robust MARL settings, observation noise is assumed to be independently sampled across agents, i.e.,

$$
\epsilon _ { i } ^ { t } \perp \epsilon _ { j } ^ { t } , \quad i \ne j .\tag{4}
$$

The objective is to learn decentralized policies

$$
\pi = \{ \pi _ { 1 } , . . . , \pi _ { N } \} ,\tag{5}
$$

that maximize the expected cumulative team reward:

$$
J ( \pi ) = \mathbb { E } _ { \pi } \left[ \sum _ { t = 0 } ^ { T } \gamma ^ { t } r ^ { t } \right] ,\tag{6}
$$

while maintaining robust cooperative behaviors under noisy observations.

Although observation disturbances are independently introduced to individual agents, independence at the observation level does not necessarily imply independent impacts on cooperative decision-making. In fully cooperative MARL, agents share the same team objective, while their individual decisions jointly contribute to the cumulative team reward. Meanwhile, the interaction dependencies underlying such cooperation are generally non-uniform: agents may difer in their task roles, local interactions, and contributions to diferent components of the shared objective. Consequently, independently introduced observation disturbances may induce cooperation-dependent efects on agent decision-making.

This distinction motivates us to investigate how observation disturbances interact with the latent cooperation structure among agents. In the following, we formalize these cooperation-dependent disturbance impacts as structured noise efects, which provide the motivation for the subsequent structure-aware representation learning framework.

## Structured Noise Efects Analysis

Although observation disturbances are independently introduced to individual agents, their induced impacts on cooperative decision-making are not necessarily independent. In fully cooperative MARL, agents share a common team objective, while their individual decisions jointly contribute to the team return. However, such global cooperation does not imply uniform interaction dependencies among all agents. Depending on their task roles, local interactions, and behavioral coordination, diferent agents may exhibit diferent degrees of dependency during task execution. Consequently, independently introduced observation disturbances can produce structure-dependent impacts through these underlying cooperative interactions.

To characterize such impacts without restricting the analysis to a specific MARL architecture, let $\Phi _ { i } ^ { t }$ denote the decision-related output of agent i at timestep t. Given the same underlying environment trajectory, the corresponding outputs under clean and noisy observations are defined as

$$
\Phi _ { i } ^ { \mathrm { c l e a n } , t } = \Phi _ { i } ( o _ { i } ^ { 0 : t } ) , \qquad \Phi _ { i } ^ { \mathrm { n o i s y } , t } = \Phi _ { i } ( \tilde { o } _ { i } ^ { 0 : t } ) .\tag{7}
$$

where $\Phi _ { i } ^ { t }$ may correspond to local action-value estimates in value-based methods or policy outputs in policy-based methods. The noise-induced decision impact of agent i is then characterized by

$$
E _ { i } ^ { t } = { \mathcal { D } } \left( \Phi _ { i } ^ { n o i s y , t } , \Phi _ { i } ^ { c l e a n , t } \right) ,\tag{8}
$$

where $\mathcal { D } ( \cdot , \cdot )$ denotes a general discrepancy measure between the clean and noisy decision outputs. This formulation distinguishes the independently introduced observation disturbance $\boldsymbol { \epsilon } _ { i } ^ { t }$ from its downstream impact $E _ { i } ^ { t }$ on cooperative decision-making. Therefore, independence among observation disturbances does not necessarily imply independence or uniformity among their induced decision impacts.

Based on this distinction, we characterize structured noise efects from two complementary perspectives: local correlation and global heterogeneity.

Local correlation. Although all agents contribute to the same global objective, their task-related interaction dependencies are generally non-uniform. Some agents may be more strongly coupled through their local interactions and coordinated behaviors, whereas others exhibit relatively weak dependencies. Such diferences can further shape how independently introduced observation disturbances afect their cooperative decisions.

Let $\mathcal { P } _ { \mathrm { s t r o n g } }$ and $\mathcal { P } _ { \mathrm { w e a k } }$ denote agent pairs with relatively strong and weak task-related interaction dependencies, respectively. We hypothesize that noise-induced decision impacts exhibit stronger temporal correlations among strongly interacting agents:

$$
\mathbb { E } _ { ( i , j ) \in \mathcal { P } _ { \mathrm { s t r o n g } } } \left[ \rho ( E _ { i } , E _ { j } ) \right] > \mathbb { E } _ { ( i , j ) \in \mathcal { P } _ { \mathrm { w e a k } } } \left[ \rho ( E _ { i } , E _ { j } ) \right] .\tag{9}
$$

Importantly, this correlation does not imply that the original observation disturbances become statistically correlated. Instead, it reflects the cooperation-dependent responses of agents to independently introduced disturbances. We refer to this property as the local correlation of structured noise efects.

Global heterogeneity. In addition to local correlation, noise-induced decision impacts may be distributed nonuniformly across the multi-agent system. Diferent agents can have diferent task roles, interaction contexts, and contributions to the shared objective, and may therefore exhibit diferent sensitivities to observation disturbances. We characterize the overall disturbance sensitivity of agent i over a trajectory as

$$
H _ { i } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } E _ { i } ^ { t } .\tag{10}
$$

The resulting sensitivities are not necessarily uniform across agents. Moreover, when agents exhibit diferent taskrelated interaction structures, such heterogeneity can also emerge at the level of cooperative groups. Therefore, structured noise efects may simultaneously exhibit locally correlated variations among strongly interacting agents and heterogeneous impact intensities across the broader cooperation structure.

Together, these two properties characterize the structured nature of observation-noise efects in cooperative MARL: noise-induced decision impacts tend to be locally correlated according to task-related interaction dependencies, while remaining globally heterogeneous across agents and cooperation structures. This observation suggests that robust cooperative decision-making should not only address observation uncertainty at the individual-agent level, but also explicitly account for the latent cooperation structures through which disturbance impacts are manifested.

## Framework Overview

The central motivation of SIGMA is that independently introduced observation disturbances can induce structured impacts on cooperative decision-making. As characterized above, these structured noise efects exhibit two complementary properties: noise-induced decision impacts tend to be locally correlated among agents with stronger task-related dependencies, while their influence remains globally heterogeneous across diferent agents and local structures. Therefore, robust cooperative learning under noisy observations requires not only handling individual observation uncertainty, but also exploiting the underlying local structures through which disturbance efects are organized.

To address this challenge, SIGMA introduces a hierarchical collaboration framework that organizes agents into adaptive local structures and progressively integrates cooperative information from local to global levels. Rather than directly aggregating information across the entire agent population, SIGMA first identifies local structural units according to the similarity of agent observations. It then performs consensus aggregation within each group to consolidate locally shared information and alleviate agent-specific disturbances. Finally, interactions among diferent groups are modeled to recover global cooperative dependencies while accounting for their heterogeneous roles. In this way, SIGMA combines local structural aggregation with global coordination to improve cooperative representation learning under noisy observations.

As illustrated in Fig. 1, SIGMA consists of three main components:

Adaptive Grouping. Motivated by the locally correlated nature ofstructured noise efects, SIGMA first identifies local structural units before hierarchical aggregation. Since taskrelated dependencies are latent and may evolve with task states, the module dynamically groups agents according to the similarity of their local observations. The resulting adaptive groups define the structural boundaries for subsequent intra-group consensus aggregation.

Intra-group Consensus Aggregation. Within each adaptive group, SIGMA models high-order interactions among agents and performs consensus aggregation over their interaction-enhanced representations. This process consolidates information shared within the local structure while smoothing agent-specific representation disturbances. By restricting consensus aggregation to adaptive groups, SIGMA avoids indiscriminately mixing information across structurally diferent agents and constructs robust group-level cooperative representations.

Inter-group Attention. The locally aggregated groups still need to exchange information to support global cooperation, while the global heterogeneity of structured noise efects indicates that diferent groups should not necessarily contribute equally. SIGMA therefore employs inter-group attention to adaptively model dependencies among group-level representations. This mechanism restores cross-group information exchange while preserving heterogeneous grouplevel contributions, producing a global task representation for decentralized decision-making.

Overall, SIGMA follows a hierarchical collaboration paradigm: agents are first organized into adaptive local structures, cooperative information is then consolidated within each group, and the resulting group-level representations are finally integrated across groups to recover global coordination. Through this local-to-global architecture, SIGMA exploits the structural characteristics of noise-induced decision impacts while preserving task-relevant cooperative information, thereby supporting robust decentralized decisionmaking under noisy observations.

## Adaptive Grouping

The local correlation of structured noise efects suggests that noise-induced impacts are associated with local task-related dependencies rather than being uniformly distributed across all agents. This motivates SIGMA to identify local structural units before performing hierarchical aggregation. However, such task-related dependencies are latent and may dynamically evolve with task states. Therefore, instead of relying on predefined team assignments, SIGMA constructs adaptive groups according to the similarity of agents’ local observations.

At timestep t, agents operating under related local task contexts tend to exhibit similar observation patterns. SIGMA therefore uses observation similarity as a practical criterion for identifying local structural relationships. Specifically, the pairwise distance between agents i and $j$ is defined as

$$
d _ { i j } ^ { t } = 1 - \frac { ( o _ { i } ^ { t } ) ^ { \top } o _ { j } ^ { t } } { \| o _ { i } ^ { t } \| \| o _ { j } ^ { t } \| } ,\tag{11}
$$

where a smaller $d _ { i j } ^ { t }$ indicates greater similarity between the local task contexts perceived by the two agents. Based on the resulting pairwise distance matrix, SIGMA employs a dynamically parameterized density-based clustering procedure to construct cooperative groups.

A key challenge of density-based clustering is that its grouping results depend on the neighborhood radius ε and the minimum number of neighboring samples $n _ { \mathrm { m i n } } .$ . Fixed clustering parameters may be unsuitable for MARL because the distribution of agent observations continuously changes during task execution. To accommodate such variations, SIGMA dynamically searches for appropriate clustering parameters according to the current observation distribution.

Specifically, SIGMA first generates several candidate neighborhood radii from the current distribution of pairwise observation distances. Instead of using fixed radius values, diferent quantiles of the distance distribution are selected so that the candidate radii can adapt to the changing observation patterns of agents:

$$
{ \mathcal { E } } ^ { t } = \left\{ Q _ { q } \left( D ^ { t } \right) \mid q \in { \mathcal { Q } } \right\} ,\tag{12}
$$

where $D ^ { t }$ contains all pairwise observation distances at timestep $t , Q _ { q } ( \cdot )$ ) denotes the q-th quantile of these distances, and Q specifies the quantiles used to generate candidate radii. Each candidate radius $\varepsilon \in \mathcal { E } ^ { t }$ is then combined with candidate values of the minimum neighborhood size $n _ { \mathrm { m i n } }$ , and DBSCAN systematically evaluates all candidate combinations of ε and $n _ { \mathrm { m i n } }$ , thereby obtaining a set of candidate grouping results.

![](images/b33a94856de07fd6361da9599f6730f9f94f3155515dc1d8810abbe8fa1c2e8a.jpg)  
Figure 1: Overall framework of SIGMA for robust cooperative representation learning under noisy observations.

To select an appropriate grouping from the candidate clustering results, SIGMA evaluates each candidate from three complementary aspects: structural quality, agent coverage, and grouping granularity. A desirable grouping should form compact and well-separated groups, retain as many agents as possible in meaningful cooperative groups, and avoid both excessive merging and fragmentation. Accordingly, the clustering score is defined as

$$
\mathcal { T } _ { \mathrm { c l u } } = S _ { \mathrm { s i l } } - \lambda _ { n } R _ { \mathrm { n o i s e } } - \lambda _ { g } R _ { \mathrm { g r o u p } } ,\tag{13}
$$

where $S _ { \mathrm { s i l } }$ denotes the silhouette score and favors groupings with high intra-group similarity and clear inter-group separation. However, relying solely on the silhouette score may favor overly restrictive partitions that leave many agents unassigned. Therefore, $R _ { \mathrm { n o i s e } }$ penalizes candidate groupings containing excessive noise points, encouraging suficient agent coverage. Meanwhile, $R _ { \mathrm { g r o u p } }$ regulates the grouping granularity by penalizing undesirable deviations in the number of discovered groups. This prevents the grouping structure from collapsing most agents into a few overly broad groups or fragmenting them into excessively small groups. The coeficients $\lambda _ { n }$ and $\lambda _ { g }$ control the strengths of the two penalties.

By jointly considering these three criteria, the evaluation favors groupings that are structurally distinguishable, sufficiently inclusive, and appropriately partitioned for subsequent intra-group aggregation.

The clustering configuration with the highest score is selected:

$$
( \varepsilon ^ { * } , n _ { \mathrm { m i n } } ^ { * } ) = \arg \operatorname* { m a x } _ { \varepsilon , n _ { \mathrm { m i n } } } \mathcal { I } _ { \mathrm { c l u } } ,\tag{14}
$$

and the corresponding grouping result is denoted as

$$
\mathcal { G } ^ { t } = \{ G _ { 1 } ^ { t } , G _ { 2 } ^ { t } , . . . , G _ { M } ^ { t } \} .\tag{15}
$$

Since DBSCAN may identify isolated agents as noise points, SIGMA assigns each such agent to an individual singleton group rather than discarding it from subsequent cooperative representation learning. Therefore, every agent remains explicitly represented in the hierarchical aggregation process.

To avoid excessive fluctuations in the grouping structure, group assignments are updated periodically rather than at every timestep. A newly obtained grouping is adopted only when it satisfies the predefined stability criterion; otherwise, the previous grouping is retained. This temporal stabilization prevents transient observation variations from causing frequent changes in cooperative group assignments.

Overall, the adaptive grouping procedure determines both the grouping structure and its density parameters according to the evolving distribution of agent observations. This enables SIGMA to construct flexible local structural units without requiring a predefined number of cooperative groups, providing the basis for subsequent intra-group consensus aggregation.

## Intra-group Consensus Aggregation

After identifying adaptive groups, SIGMA further performs intra-group consensus aggregation to construct robust grouplevel cooperative representations. The adaptive grouping provides local structural units composed of agents with related task contexts, within which agents are more likely to share task-relevant information while exhibiting locally associated responses to observation disturbances. However, individual agent representations may still contain deviations arising from agent-specific information, partial observability, and observation uncertainties. Therefore, rather than treating agents independently or directly mixing information across the entire agent population, SIGMA consolidates representations within each adaptive group to extract locally shared cooperative information while reducing the influence of individual representation deviations.

For each adaptive group $G _ { m } ^ { t } .$ , SIGMA first employs a hypergraph neural network (HGCN) to encode high-order interactions among agents within the group:

$$
z _ { i } ^ { t } = \mathrm { H G C N } ( o _ { i } ^ { t } , G _ { m } ^ { t } ) , \quad i \in G _ { m } ^ { t } ,\tag{16}
$$

where $z _ { i } ^ { t }$ denotes the interaction-enhanced representation of agent i. The hypergraph formulation captures collective dependencies among multiple agents within the same local structure, providing interaction-aware representations for subsequent consensus aggregation.

To illustrate the information shared within an adaptive group, the representation of each agent can be conceptually decomposed into a group-shared component and an individual deviation:

$$
z _ { i } ^ { t } = s _ { m } ^ { t } + \delta _ { i } ^ { t } , \quad i \in G _ { m } ^ { t } ,\tag{17}
$$

where $s _ { m } ^ { t }$ represents the semantic information shared among agents in group $G _ { m } ^ { t }$ , and $\delta _ { i } ^ { t }$ denotes the agent-specific deviation from the shared component. Such deviations may arise from diferences in local information, partial observability, and observation disturbances.

Based on these interaction-aware representations, SIGMA performs consensus aggregation within each adaptive group:

$$
g _ { m } ^ { t } = \frac { 1 } { \left| G _ { m } ^ { t } \right| } \sum _ { i \in G _ { m } ^ { t } } z _ { i } ^ { t } = s _ { m } ^ { t } + \frac { 1 } { \left| G _ { m } ^ { t } \right| } \sum _ { i \in G _ { m } ^ { t } } \delta _ { i } ^ { t } ,\tag{18}
$$

where $g _ { m } ^ { t }$ denotes the resulting group-level cooperative representation. As shown in the above formulation, intragroup aggregation preserves the shared component $s _ { m } ^ { t }$ while averaging agent-specific deviations within the local structure. This consensus process smooths individual representation fluctuations and reduces the influence of any single agent on the group-level representation.

Under structured noise efects, however, agent deviations may contain both agent-specific and locally correlated components. While agent-specific disturbances can be attenuated through aggregation, locally correlated disturbances may persist within the group, limiting the noise-reduction capability of simple averaging. Therefore, intra-group consensus provides a structured way to suppress individual disturbances while retaining information shared within each local cooperative structure.

Consequently, SIGMA obtains a set of group-level representations $\{ g _ { 1 } ^ { t } , \hdots , g _ { M } ^ { t } \}$ , each characterizing a distinct local cooperative structure. Rather than prematurely merging these representations, SIGMA retains their structural diferences for subsequent inter-group coordination.

## Inter-group Attention

After intra-group consensus aggregation, SIGMA obtains multiple group-level representations that characterize different local cooperative structures. While such local aggregation facilitates robust representation learning within each group, it may also limit information exchange across diferent groups. Since cooperative tasks generally require coordination beyond individual local structures, SIGMA further models inter-group dependencies to restore global cooperative interactions.

Meanwhile, the global heterogeneity of structured noise efects suggests that diferent groups may exhibit distinct task characteristics and disturbance sensitivities. Therefore, simply treating all groups equally may overlook their heterogeneous roles in global cooperation. To address both aspects, SIGMA employs an inter-group attention mechanism to adaptively model dependencies among diferent groups and integrate their information into a global task representation.

Given the group-level representations $\{ g _ { m } ^ { t } \} _ { m = 1 } ^ { M }$ , SIGMA employs an inter-group attention mechanism to adaptively model dependencies among diferent local cooperative structures.

Specifically, the query, key, and value representations of each group are computed as

$$
\begin{array} { r } { q _ { m } ^ { t } = W _ { q } g _ { m } ^ { t } , \qquad k _ { m } ^ { t } = W _ { k } g _ { m } ^ { t } , \qquad v _ { m } ^ { t } = W _ { v } g _ { m } ^ { t } , } \end{array}\tag{19}
$$

where $W _ { q } , W _ { k }$ , and $W _ { v }$ are learnable projection matrices. The attention coeficient between groups m and n is calculated as

$$
\alpha _ { m n } ^ { t } = \frac { \exp ( ( q _ { m } ^ { t } ) ^ { \top } k _ { n } ^ { t } / \sqrt { d } ) } { \sum _ { j = 1 } ^ { M } \exp ( ( q _ { m } ^ { t } ) ^ { \top } k _ { j } ^ { t } / \sqrt { d } ) } ,\tag{20}
$$

where d denotes the dimension of the key representation. The attention coeficient $\alpha _ { m n } ^ { t }$ characterizes the contextdependent relevance of group n to group $m$ , allowing SIGMA to distinguish heterogeneous inter-group dependencies rather than treating all groups equally.

Based on these attention weights, each group representation is refined by integrating information from other cooperative groups:

$$
\hat { g } _ { m } ^ { t } = \sum _ { n = 1 } ^ { M } \alpha _ { m n } ^ { t } v _ { n } ^ { t } .\tag{21}
$$

Through this interaction, locally aggregated group representations can exchange information across structural boundaries, recovering cross-group cooperative dependencies that may be weakened by intra-group aggregation. Meanwhile, the adaptive attention weights allow each group to selectively incorporate information from other groups according to the current task context, while preserving the distinctions among diferent cooperative structures.

As a result, each group obtains a refined representation $\hat { g } _ { m } ^ { t }$ that retains its group-specific cooperative information while incorporating relevant context from other groups. Rather than collapsing all groups into a single global representation, SIGMA maintains these refined representations separately and assigns $\hat { g } _ { m } ^ { t }$ to the agents belonging to group $\hat { G } _ { m } ^ { t }$ for subsequent decentralized decision-making.

Together with intra-group consensus aggregation, intergroup attention therefore enables SIGMA to preserve locally shared cooperative information while restoring coordination across diferent cooperative structures. This forms a hierarchical representation process from individual agents to local cooperative groups and finally to group-specific representations enriched with global cooperative context.

## Integration with Cooperative MARL Frameworks

SIGMA is designed as a representation learning module that can be integrated into existing cooperative MARL frameworks. Instead of modifying the policy optimization procedure, SIGMA focuses on learning robust cooperative representations from noisy observations and provides additional cooperative information for decentralized decision-making.

Specifically, after inter-group attention, each cooperative group obtains a refined representation $\hat { g } _ { m } ^ { t }$ that incorporates cross-group contextual information while retaining its groupspecific cooperative information. For each agent $i \in G _ { m } ^ { t } ,$ the corresponding group representation $\hat { g } _ { m } ^ { t }$ is assigned to the agent and incorporated into its decision function together with its local information:

$$
\pi _ { i } \left( a _ { i } ^ { t } \mid \tau _ { i } ^ { t } , \hat { g } _ { m } ^ { t } \right) , \quad i \in G _ { m } ^ { t } ,\tag{22}
$$

where $\boldsymbol { \tau } _ { i } ^ { t }$ denotes the local action-observation history of agent i, and $\hat { g } _ { m } ^ { t }$ denotes the refined cooperative representation of the adaptive group to which agent i belongs.

For value-based MARL algorithms, the corresponding group representation can similarly be incorporated into the individual value estimation:

$$
Q _ { i } \left( \tau _ { i } ^ { t } , a _ { i } ^ { t } , \hat { g } _ { m } ^ { t } \right) , \quad i \in G _ { m } ^ { t } .\tag{23}
$$

In this way, agents within the same adaptive group share the corresponding refined cooperative representation, while agents in diferent groups receive group-specific representations enriched with cross-group cooperative context. Therefore, SIGMA can be integrated into existing cooperative MARL algorithms as an additional representation learning module without modifying their underlying policy optimization procedures.

## Experiments

## Experimental Setup

Environment. We evaluate SIGMA on the StarCraft II Multi-Agent Challenge (SMAC), a widely used benchmark for cooperative multi-agent reinforcement learning. Experiments are conducted on the 5m\_vs\_6m and 8m\_vs\_9m scenarios, where multiple allied agents must coordinate their movements, target selection, and combat behaviors against an opposing team. These scenarios provide representative cooperative settings for evaluating both policy performance and robustness under imperfect observations.

Noise Settings. To evaluate robustness against observation uncertainty, Gaussian noise is independently introduced into the local observation of each agent:

$$
\widetilde { o } _ { i } ^ { t } = o _ { i } ^ { t } + \epsilon _ { i } ^ { t } , \qquad \epsilon _ { i } ^ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } ) ,\tag{24}
$$

where σ controls the observation-noise intensity. We consider both the noise-free setting and multiple noisy settings with $\sigma \in \{ 0 , 1 . 5 , 2 . 5 , 5 . 0 \}$ . Importantly, observation disturbances are independently sampled across agents, allowing us to investigate whether independent input disturbances can induce structured downstream efects through cooperative interactions.

Baselines. We compare SIGMA with representative cooperative MARL methods, including QMIX and HYGMA. QMIX serves as a widely adopted value-decomposition baseline, while HYGMA models structured interactions among multiple agents through dynamic grouping and hypergraphbased coordination. All methods are evaluated under the same environment and observation-noise settings to ensure a consistent comparison.

Table 1: Performance comparison under diferent observation-noise levels on SMAC.
<table><tr><td>Scenario</td><td>Method</td><td> $\sigma = 0$ </td><td> $\sigma = 1 . 5$ </td><td> $\sigma = 2 . 5$ </td><td> $\sigma = 5 . 0$ </td></tr><tr><td rowspan="3"> $5 \mathrm { m \_ v s \_ 6 m }$ </td><td>QMIX</td><td>79.87%</td><td>35.64%</td><td>40.00%</td><td>11.87%</td></tr><tr><td>HYGMA</td><td>96.88%</td><td>88.75%</td><td>83.13%</td><td>0.25%</td></tr><tr><td>SIGMA (Ours)</td><td>95.63%</td><td>94.37%</td><td>90.00%</td><td>66.25%</td></tr><tr><td rowspan="3"> $8 \mathrm { m } \_ { \mathrm { v } } \mathrm { s } \_ { \mathrm { 9 m } }$ </td><td>QMIX</td><td>83.13%</td><td>82.26%</td><td>65.89%</td><td>16.25%</td></tr><tr><td>HYGMA</td><td>98.45%</td><td>88.12%</td><td>77.50%</td><td>23.75%</td></tr><tr><td> $\mathbf { S I G M A } \left( \mathbf { O u r s } \right)$ </td><td>97.24%</td><td>95.00%</td><td>91.87%</td><td>94.37%</td></tr></table>

σ denotes the standard deviation of observation noise.

Implementation Details. For each scenario, all methods are trained using the same environment configurations and evaluation protocol. Unless otherwise specified, the converged checkpoint obtained at the end of training is used for robustness evaluation and structured noise analysis, without further policy updates during evaluation. Other training hyperparameters and implementation details are kept consistent across compared settings whenever applicable.

Evaluation Metrics. We primarily use episode win rate to evaluate cooperative policy performance under diferent observation-noise levels, while training curves are used to examine the learning dynamics of diferent methods. For the analysis of structured noise efects, we further evaluate two complementary properties at the Q-value level. Local correlation is characterized by the temporal correlation of relative noise-induced Q-efects among diferent agent pairs, whereas global heterogeneity is quantified using the normalized disparity of centered Q-efects across adaptive cooperative groups. The detailed construction of these analysis metrics and their corresponding structural controls is introduced in the structured noise analysis below.

## Performance Comparison under Noisy Observations

We first evaluate the overall robustness of SIGMA under diferent levels of observation noise. Table 1 reports the final win rates of SIGMA and the compared methods under both noise-free and noisy conditions.

As shown in Table 1, the performance diferences among the compared methods become increasingly pronounced as the observation-noise intensity increases. While all methods achieve competitive performance under noise-free or mild-noise conditions, QMIX sufers substantial degradation under stronger disturbances. The structure-aware baseline HYGMA generally provides improved robustness by explicitly modeling multi-agent interactions, but its performance also decreases as observation noise becomes stronger. In contrast, SIGMA maintains consistently high win rates across the evaluated noise levels and scenarios. These results suggest that explicitly modeling cooperation structures alone does not necessarily provide suficient robustness against noisy observations. The advantage of SIGMA lies in further exploiting these structures according to the characteristics of structured noise efects. Specifically, intra-group consensus integrates information among structurally related agents to reduce the influence of agent-specific deviations, while intergroup attention preserves and adaptively coordinates information across groups with diferent disturbance responses. Consequently, the performance advantage of SIGMA becomes more evident as local observations become increasingly unreliable.

![](images/007f50e8fb92958026b6c0aa7e7fd64d0362c52a4e4442dcb07f7da2436b7917.jpg)

![](images/28f596cc545b8b3a3c5e0a54699b99f4be7d55955e6da6f4ac35c11fd777c7b5.jpg)  
(a) $5 \mathrm { m \_ v s \_ 6 m }$  
(b) 8m\_vs\_9m  
Figure 2: Training performance of diferent methods under strong observation noise $( \sigma = 5 . 0 )$ on two SMAC scenarios.

Figure 2 further compares the learning dynamics of diferent methods under strong observation noise. Rather than considering only the final policy performance, the training curves illustrate how efectively each method learns and maintains cooperative behaviors throughout training.

As shown in Fig. 2, SIGMA achieves a faster improvement in win rate and maintains more stable performance during training compared with the baseline methods. The advantage becomes increasingly evident as training progresses, suggesting that exploiting adaptive cooperation structures facilitates more efective policy learning when local observations are disturbed. Together with the final performance results in Table 1, these results demonstrate that SIGMA improves both the robustness and learning eficiency of cooperative policies under noisy observations.

## Analysis of Structured Noise Efects

The design of SIGMA is motivated by the structured noise efects, where independently introduced observation disturbances may induce locally correlated yet globally heterogeneous efects on cooperative decision-making. To empirically examine this phenomenon, we analyze noise-induced changes in agent Q-values from two complementary perspectives: local correlation among agents and global heterogeneity across cooperative groups.

To isolate the efects induced by observation disturbances, we perform paired clean/noisy forward passes from the same decision state. At each timestep, the clean and noisy branches share the same environment state, observation history, hidden state, and available-action set, while observation noise is introduced only into the noisy branch. The resulting Q-value diferences therefore characterize the downstream decision efects associated with observation disturbances. For structural comparison, we further construct random groupings by randomly reassigning agents while preserving the number and sizes of the adaptive groups. This provides a control for determining whether the observed noise efects are specifically associated with the discovered cooperation structures rather than arbitrary partitions.

Local Correlation. We first examine whether agents within the same adaptive group exhibit more correlated noise-induced decision efects than agents belonging to different groups. For agent i, we characterize the relative $\mathrm { Q } \mathrm { - }$ efect at timestep t as the normalized change between its clean and noisy Q-values:

Table 2: Local correlation of relative Q-efects across SMAC scenarios.
<table><tr><td>Scenario</td><td>Within</td><td>Between</td><td> $\Delta \rho$ </td><td>Random  $\Delta \rho$ </td></tr><tr><td> $5 \mathrm { m \_ v s \_ 6 m }$ </td><td>0.308 ± 0.012</td><td> $0 . 0 7 6 \pm 0 . 0 0 4$ </td><td>0.232 ± 0.009</td><td>−0.006 ± 0.001</td></tr><tr><td>8m_vs_9m</td><td>0.346 ± 0.001</td><td> $0 . 2 6 5 \pm 0 . 0 0 3$ </td><td>0.082 ± 0.001</td><td>0.002 ± 0.000</td></tr></table>

Note: Results are reported as mean ± standard deviation over five repeated analyses.

$$
E _ { i , \mathrm { r e l } } ^ { t } = \frac { \left\| Q _ { i , \mathrm { n o i s y } } ^ { t } - Q _ { i , \mathrm { c l e a n } } ^ { t } \right\| _ { 2 } } { \left\| Q _ { i , \mathrm { c l e a n } } ^ { t } \right\| _ { 2 } + \epsilon } ,\tag{25}
$$

where ϵ is a small constant for numerical stability. This metric measures the magnitude of the noise-induced Q-value change relative to the agent’s original Q-value scale.

We then compute the temporal correlation of relative $\mathrm { Q } \mathrm { - }$ efects for diferent agent pairs. Let $\rho _ { \mathrm { w i t h i n } }$ and $\rho _ { \mathrm { b e t w e e n } }$ denote the average correlations for agent pairs within the same adaptive group and across diferent groups, respectively. Their diference is defined as

$$
\Delta \rho = \rho _ { \mathrm { w i t h i n } } - \rho _ { \mathrm { b e t w e e n } } .\tag{26}
$$

A positive $\Delta \rho$ indicates that agents within the same local cooperation structure exhibit more strongly correlated noise efects.

Table 2 reports the local-correlation results over five repeated runs on $5 \mathrm { m \_ v s \_ 6 m }$ . For comparison, we also report the correlation gap obtained from the matched random groupings.

As shown in Table 2, within-group agent pairs exhibit stronger correlations in their relative Q-efects than betweengroup pairs in both senarios. On $5 \mathrm { m \_ v s \_ 6 m }$ , the average correlation gap $\Delta \rho$ reaches 0.232, while the corresponding gap under random grouping remains close to zero (−0.006). A consistent pattern is observed on 8m\_vs\_9m, with an average $\Delta \rho$ of 0.082, compared with only 0.002 under random grouping. Moreover, the within-group correlation remains higher than the between-group correlation across all five repeated analyses in both scenarios. These results indicate that the local correlation of noise-induced decision efects is consistently associated with the discovered cooperation structures rather than arbitrary agent partitions, providing empirical evidence for the local-correlation property of structured noise efects.

Global Heterogeneity. Local correlation characterizes whether noise efects vary coherently among structurally related agents, but does not indicate whether diferent cooperation structures are afected to the same extent. We therefore further examine the global heterogeneity of noise efects across adaptive groups.

To focus on diferential disturbance responses across agents, we remove the common Q-efect component and obtain the centered Q-efect $E _ { i , \mathrm { c e n } } ^ { t }$ . For each adaptive group $G _ { m } ^ { t }$ , the corresponding group-level efect is computed as

Table 3: Global heterogeneity of centered Q-efects across SMAC scenarios.
<table><tr><td>Scenario</td><td>Adaptive H</td><td>Random H</td><td>∆H</td><td>Positive ∆H (%)</td></tr><tr><td> $5 \mathrm { m \_ v s \_ 6 m }$ </td><td> $1 . 0 2 3 \pm 0 . 0 0 1$ </td><td> $0 . 7 7 1 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 2 5 1 \pm 0 . 0 0 1 }$ </td><td> $\overline { { 9 2 \% \pm 0 . 0 0 3 } }$ </td></tr><tr><td>8m_vs_9m</td><td> $1 . 0 9 4 \pm 0 . 0 0 2$ </td><td> $0 . 8 0 4 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 2 9 0 \pm 0 . 0 0 2 }$ </td><td> $9 3 \% \pm 0 . 0 0 2$ </td></tr></table>

Note: H denotes the normalized group-level heterogeneity of centered Q-efects. Results are reported as mean ± standard deviation over five repeated analyses. Positive ∆H denotes the percentage of analyzed episodes satisfying $H _ { \mathrm { a d a p t i v e } } > H _ { \mathrm { r a n d o m } }$

$$
E _ { m } ^ { t } = \frac { 1 } { | G _ { m } ^ { t } | } \sum _ { i \in G _ { m } ^ { t } } E _ { i , \mathrm { c e n } } ^ { t } .\tag{27}
$$

We quantify global heterogeneity using the normalized average pairwise diference between group-level efects:

$$
H _ { \mathrm { g r o u p } } ^ { t } = \frac { \displaystyle \frac { 2 } { M ( M - 1 ) } \sum _ { m < n } \left| E _ { m } ^ { t } - E _ { n } ^ { t } \right| } { \displaystyle \frac { 1 } { M } \sum _ { m = 1 } ^ { M } E _ { m } ^ { t } + \epsilon } ,\tag{28}
$$

where M denotes the number of adaptive groups. A larger $H _ { \mathrm { g r o u p } }$ indicates stronger diferences in noise efects across local cooperation structures.

Table 3 compares the heterogeneity measured under adaptive and matched random groupings.

As shown in Table 3, adaptive groups exhibit consistently greater heterogeneity in noise-induced Q-efects than matched random groups across both scenarios. On $5 \mathrm { m \_ v s \_ 6 m }$ , the average heterogeneity increases from 0.771 under random grouping to 1.023 under adaptive grouping, yielding an average gap of $\Delta H = 0 . 2 5 1$ . A similar pattern is observed on 8m\_vs\_9m, where the average heterogeneity gap reaches 0.290.

Moreover, positive heterogeneity gaps are observed in approximately 92% and 93% of the analyzed episodes on 5m\_vs\_6m and $8 \mathrm { m } \_ \mathrm { v } \mathrm { s } \_ 9 \mathrm { m }$ , respectively. The consistency across scenarios and repeated analyses indicates that noiseinduced decision efects are not uniformly distributed across agent groups, but exhibit systematic diferences associated with the discovered cooperation structures.

## Ablation Study

To investigate the contribution of each component in SIGMA, we conduct ablation studies by removing or replacing individual modules.

We consider the following variants:

• SIGMA w/o Grouping: removes adaptive grouping and directly aggregates information among all agents.

• SIGMA-Random Group: replaces adaptive grouping with randomly generated groups while maintaining the same group sizes.

• SIGMA w/o Intra: removes intra-group consensus aggregation and uses individual agent representations directly.

![](images/a13a627ac91cd6a56f084b4acdc228328f6c980f242c53689a6981f58dd56a3e.jpg)  
Figure 3: Performance comparison of SIGMA and its ablated variants under noisy observations.

• SIGMA w/o Inter: removes inter-group attention and directly combines group-level representations.

Removing adaptive grouping and assigning all agents into a single group results in the largest performance degradation, demonstrating that preserving meaningful cooperation structures is critical for robust representation learning. The random grouping variant achieves better performance than the single-group setting, indicating that group-wise aggregation itself is beneficial; however, its inferior performance compared with SIGMA suggests that adaptively discovering cooperation structures is important for obtaining more efective representations. Removing inter-group attention causes a substantial performance drop, which highlights the importance of modeling dependencies across diferent cooperative groups for recovering global coordination. In contrast, removing intra-group aggregation leads to a relatively smaller degradation, indicating that intra-group consensus further improves representation robustness by integrating information among related agents.

## Limitations

The current experimental evaluation is subject to several limitations. In particular, the main performance results are currently reported based on individual training runs for each setting. Since MARL training may exhibit variability across random initialization, environment stochasticity, and exploration trajectories, these results should be interpreted as preliminary evidence of the robustness advantage of SIGMA rather than a complete statistical comparison. A more comprehensive evaluation with multiple random seeds, reporting mean performance and standard deviation, will be included in future experiments. In addition, the current evaluation is conducted on a limited set of SMAC scenarios. Future work will extend the evaluation to more diverse cooperative scenarios with diferent team sizes and coordination complexities to further examine the generalizability of SIGMA.

## Conclusion

In this paper, we investigated the robustness of cooperative MARL under noisy observations from the perspective of cooperation structures. We characterized structured noise efects, showing that independently introduced observation disturbances can induce decision efects that are locally correlated among structurally related agents while remaining globally heterogeneous across diferent cooperation structures. Our empirical analysis further supports these two properties across diferent cooperative scenarios, highlighting the role of underlying cooperation structures in shaping the downstream efects of observation noise.

Motivated by this observation, we proposed SIGMA, a hierarchical collaboration framework for robust cooperative representation learning. SIGMA adaptively identifies latent cooperation structures, performs intra-group consensus aggregation to integrate information among structurally related agents, and employs inter-group attention to coordinate information across diferent groups. This hierarchical design enables SIGMA to exploit local structural information while preserving dependencies and diferences across cooperation structures.

Experiments on the StarCraft II Multi-Agent Challenge show that SIGMA maintains strong cooperative performance under increasing levels of observation noise and exhibits smaller performance degradation than the evaluated baselines. Together with the structured noise analysis, these results suggest that explicitly exploiting cooperation structures is a promising approach to improving the robustness of cooperative MARL under noisy observations.

Future work will evaluate SIGMA across a broader range of cooperative tasks and disturbance settings, conduct more comprehensive multi-seed comparisons, and investigate more general mechanisms for modeling and exploiting structured noise efects in large-scale multi-agent systems.

## References

Amato, C. 2024. An introduction to centralized training for decentralized execution in cooperative multi-agent reinforcement learning. arXiv preprint arXiv:2409.03052.

Barta, Z.; Nagy, B.; and Gulyás, L. 2025. Measuring the Robustness of Multi-Agent Reinforcement Learning Systems under Partial Agent Failure. In Proceedings ofthe Intelligent Robotics FAIR 2025, 58–63.

Feng, Y.; You, H.; Zhang, Z.; Ji, R.; and Gao, Y. 2019. Hypergraph neural networks. In Proceedings of the AAAI conference on artificial intelligence, volume 33, 3558–3565.

Fu, Y.; Zhu, Y.; Chai, J.; and Zhao, D. 2024. LDR: Learning discrete representation to improve noise robustness in multiagent tasks. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 55(1): 513–525.

Hu, G.; Li, H.; Liu, S.; Zhu, Y.; and Zhao, D. 2023. NeuronsMAE: A novel multi-agent reinforcement learning environment for cooperative and competitive multi-robot tasks. In 2023 International joint conference on neural networks (IJCNN), 1–8. IEEE.

Jiang, J.; Dun, C.; Huang, T.; and Lu, Z. 2018. Graph convolutional reinforcement learning. arXiv preprint arXiv:1810.09202.

Kilinc, O.; and Montana, G. 2018. Multi-agent deep reinforcement learning with extremely noisy observations. arXiv preprint arXiv:1812.00922.

Lhaksmana, K. M.; Murakami, Y.; and Ishida, T. 2018. Role-based modeling for designing agent behavior in selforganizing multi-agent systems. International Journal of Software Engineering and Knowledge Engineering, 28(01): 79–96.

Li, C.; Liu, Q.; Zhou, Z.; Buss, M.; and Liu, F. 2022. Of-policy risk-sensitive reinforcement learning-based constrained robust optimal control. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 53(4): 2478–2491.

Li, D.; Zhang, Q.; Lu, S.; Pan, Y.; and Zhao, D. 2023. Conditional goal-oriented trajectory prediction for interacting vehicles. IEEE Transactions on Neural Networks and Learning Systems, 35(12): 18758–18770.

Lin, J.; Dzeparoska, K.; Zhang, S. Q.; Leon-Garcia, A.; and Papernot, N. 2020. On the robustness of cooperative multiagent reinforcement learning. In 2020 IEEE Security and Privacy Workshops (SPW), 62–68. IEEE.

Liu, C.; and Li, D. 2025. HYGMA: Hypergraph Coordination Networks with Dynamic Grouping for Multi-Agent Reinforcement Learning. arXiv preprint arXiv:2505.07207.

Liu, Z.; Guo, Z.; Cen, Z.; Zhang, H.; Tan, J.; Li, B.; and Zhao, D. 2022. On the robustness of safe reinforcement learning under observational perturbations. arXiv preprint arXiv:2205.14691.

Lowe, R.; Wu, Y. I.; Tamar, A.; Harb, J.; Pieter Abbeel, O.; and Mordatch, I. 2017. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 30.

Malysheva, A.; Sung, T. T.; Sohn, C.-B.; Kudenko, D.; and Shpilman, A. 2018. Deep multi-agent reinforcement learning with relevance graphs. arXiv preprint arXiv:1811.12557.

Muratore, F.; Gienger, M.; and Peters, J. 2019. Assessing transferability from simulation to reality for reinforcement learning. IEEE transactions onpattern analysis and machine intelligence, 43(4): 1172–1183.

Phan, T.; Ritz, F.; Belzner, L.; Altmann, P.; Gabor, T.; and Linnhof-Popien, C. 2021. Vast: Value function factorization with variable agent sub-teams. Advances in neural information processing systems, 34: 24018–24032.

Rashid, T.; Samvelyan, M.; Schroeder, C.; Farquhar, G.; Foerster, J.; and Whiteson, S. 2018. Qmix: Monotonic value function factorisation for deep multi-agent reinforcement learning. In International conference on machine learning, 4295–4304. Pmlr.

Russell, S. J.; and Zimdars, A. 2003. Q-decomposition for reinforcement learning agents. In Proceedings of the 20th international conference on machine learning (ICML-03), 656–663.

Sadhu, A. K.; and Konar, A. 2018. An eficient computing of correlated equilibrium for cooperative Q-learning-based multi-robot planning. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 50(8): 2779–2794.

Samvelyan, M.; Rashid, T.; De Witt, C. S.; Farquhar, G.; Nardelli, N.; Rudner, T. G.; Hung, C.-M.; Torr, P. H.; Foerster, J.; and Whiteson, S. 2019. The starcraft multi-agent challenge. arXiv preprint arXiv:1902.04043.

Son, K.; Kim, D.; Kang, W. J.; Hostallero, D. E.; and Yi, Y. 2019. Qtran: Learning to factorize with transformation for cooperative multi-agent reinforcement learning. In International conference on machine learning, 5887–5896. PMLR.

Wang, T.; Dong, H.; Lesser, V.; and Zhang, C. 2020. Roma: Multi-agent reinforcement learning with emergent roles. arXiv preprint arXiv:2003.08039.

Wang, T.; Zeng, L.; Dong, W.; Yang, Q.; Yu, Y.; and Zhang, C. 2021. Context-aware sparse deep coordination graphs. arXiv preprint arXiv:2106.02886.

Wen, M.; Kuba, J.; Lin, R.; Zhang, W.; Wen, Y.; Wang, J.; and Yang, Y. 2022. Multi-agent reinforcement learning is a sequence modeling problem. Advances in Neural Information Processing Systems, 35: 16509–16521.

Yang, Y.; Modares, H.; Vamvoudakis, K. G.; and Lewis, F. L. 2023. Cooperative finitely excited learning for dynamical games. IEEE Transactions on Cybernetics, 54(2): 797–810.

Yu, C.; Velu, A.; Vinitsky, E.; Gao, J.; Wang, Y.; Bayen, A.; and Wu, Y. 2022. The surprising efectiveness of ppo in cooperative multi-agent games. Advances in neural information processing systems, 35: 24611–24624.

Zang, Y.; He, J.; Li, K.; Fu, H.; Fu, Q.; Xing, J.; and Cheng, J. 2023. Automatic grouping for eficient cooperative multiagent reinforcement learning. Advances in neural information processing systems, 36: 46105–46121.

Zhang, H.; Chen, H.; Boning, D.; and Hsieh, C.-J. 2021. Robust reinforcement learning on state observations with learned optimal adversary. arXivpreprint arXiv:2101.08452.

Zhang, H.; Chen, H.; Xiao, C.; Li, B.; Liu, M.; Boning, D.; and Hsieh, C.-J. 2020a. Robust deep reinforcement learning against adversarial perturbations on state observations. Advances in neural informationprocessing systems, 33: 21024– 21037.

Zhang, K.; Sun, T.; Tao, Y.; Genc, S.; Mallya, S.; and Basar, T. 2020b. Robust multi-agent reinforcement learning with model uncertainty. Advances in neural information processing systems, 33: 10571–10583.