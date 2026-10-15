# GENERALIZABLE SPATIO-TEMPORAL FORECASTING BEYOND THE CLOSED-WORLD SETTING

Shengnan Guo<sup>1,2</sup>, Jinjun Cheng<sup>1</sup>, Xiaowei Mao<sup>1</sup>, Jianhao Zhang<sup>1</sup>, Haochen Lv<sup>1</sup>, Yang Jiang<sup>1</sup>, YiHeng Huang<sup>1</sup>, Yan Lin<sup>4</sup>, Youfang Lin<sup>1,3</sup>, Huaiyu Wan<sup>1,3∗</sup>

<sup>1</sup>School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China <sup>2</sup>Key Laboratory of Big Data & Artificial Intelligence in Transportation,

Ministry of Education, Beijing, China

<sup>3</sup>Beijing Key Laboratory of Traffic Data Mining and Embodied Intelligence, Beijing, China <sup>4</sup>Department of Computer Science, Aalborg University, Aalborg, Denmark {guoshn, chengjinjun, maoxiaowei, jianhaozhang}@bjtu.edu.cn {jiangyang, lvhaochen, huangyiheng, yflin, hywan}@bjtu.edu.cn lyan@cs.aau.dk

## ABSTRACT

Spatio-temporal forecasting is fundamental to many real-world systems, yet deployment environments are inherently varying: temporal distributions may evolve, spatial domains may change, and observation availability may shift from training conditions. Existing models are often tightly coupled to their training environments through node-specific representations, predefined spatial support sets, and reliance on consistently available observations, limiting generalization to changing environments. To this end, we propose VaryST, a compact spatio-temporal forecasting model trained from scratch that achieves strong generalization by dynamically constructing predictive context from available long-term observations. To accommodate temporal changes, VaryST derives node representations from long-term histories, capturing current behavior in the context of longer-term patterns. To adapt to spatial changes, it dynamically selects functionally relevant support nodes beyond predefined neighborhoods and integrates their information with recent and long-term target states and global context. To handle varying observation availability, VaryST selectively masks historical patches on which the model relies most during training, encouraging it to exploit alternative temporal and spatial dependencies for accurate forecasting when key observations become unavailable. Extensive experiments on six real-world datasets against seventeen baselines show that VaryST achieves strong in-domain forecasting accuracy, substantially improves zero- and few-shot generalization under spatial and temporal changes, and remains robust to varying observation availability.

## 1 INTRODUCTION

Spatio-temporal (ST) forecasting predicts the future states of spatially distributed nodes by modeling temporal dynamics and spatial dependencies, supporting applications such as traffic management, energy systems, and environmental monitoring. Despite substantial progress, many existing methods operate within a largely closed-world setting, assuming that temporal distributions, spatial domains, and observation availability remain stable between training and deployment.

Real-world deployment, however, is inherently varying, as shown in Figure 1. Temporal distributions evolve with seasonal variations and long-term trends. Spatial domains change as models transfer to new cities or sensor networks expand. Observation availability varies with sensor failures and communication interruptions, producing missing rates and patterns that differ from those encountered during training. These changes raise a central question: Can a spatio-temporal forecasting model remain effective as where it is deployed, when it is deployed, and what observations are available change?

![](images/fdc29b0f18155595c3ad2533a2c1035313f977253cd1a17e52ee62f1e573403c.jpg)  
Figure 1: ST models in varying deployment environments

Existing methods face several limitations in such vayring deployment environments. Specifically, node-specific embeddings and training-graph-dependent parameters restrict direct transfer to unseen nodes or networks. Besides, many models draw information only from a predefined set of nodes. Even when they dynamically adjust the importance of these nodes, they cannot seek useful information from nodes outside that set. Models may also rely heavily on observations that are consistently available during training, making accurate forecasting difficult when these observations are missing at deployment. These limitations reflect a common challenge: the forecasting mechanism is tightly coupled to the training environment.

This coupling is problematic because the information useful for forecasting is itself environment dependent. Temporal shifts require reassessing the target’s current behavior; spatial changes may alter which nodes provide useful support; and changes in observation availability may make previously useful information unavailable. We refer to the information selected to support a forecast as predictive evidence, and its organization as predictive context. We therefore argue that a generalizable model should dynamically construct its predictive context from currently available observations, rather than inherit it from the training environment. This requires characterizing the target node’s current behavior, identifying where relevant evidence can be found, and organizing that evidence for the current prediction.

Long-term observations provide the basis for dynamically constructing predictive context. A short recent window captures immediate dynamics but offers limited evidence for distinguishing persistent behavior from transient fluctuations. Longer histories reveal recurring patterns and operating regimes, while providing alternative cues when recent observations are incomplete. They therefore help characterize the target node’s current behavior and identify useful predictive support in the current deployment environment. Their value thus extends beyond a longer input horizon to enabling predictive context to be reconstructed as deployment conditions change.

Building on this perspective, we propose VaryST, a compact framework trained from scratch for generalizable ST forecasting in Varying deployment environments. VaryST dynamically constructs target-specific predictive contexts from available long-term observations. Specifically, Long-Term Observation-Derived Representations characterize node behavior without fixed identities, while Dynamic Functional Support Node Retrieval identifies relevant support nodes beyond predefined neighborhoods. Target-Centric Hierarchical Contextual Decoding organizes recent and long-term target states, retrieved supports, and global context across multiple temporal scales and spatial scopes, providing a unified interface for inductive inference. To improve robustness to varying observation availability, Importance-Guided Observation Perturbation (IGOP) masks highly relied-upon observations during training, encouraging the model to reorganize predictive evidence. Together, these designs enable VaryST to reconstruct predictive context as deployment conditions change. Our main contributions are summarized as follows:

• We identify training-environment coupling as a common challenge across temporal shifts, spatial changes, and varying observation availability, and propose VaryST to address it through dynamic predictive context construction.

• We propose Long-Term Observation-Derived Representations and Dynamic Functional Support Node Retrieval to characterize nodes without fixed identities and identify relevant support beyond predefined neighborhoods, enabling adaptation to temporal shifts and spatial changes.

• We develop Target-Centric Hierarchical Contextual Decoding to organize multi-scale targetspecific evidence through a unified inductive interface, together with Importance-Guided Observation Perturbation for robustness to varying observation availability.

• Extensive experiments on six real-world datasets against seventeen baselines demonstrate strong in-domain forecasting, zero-/few-shot generalization across temporal and spatial changes, and robustness to unseen observation conditions.

## 2 RELATED WORK

## 2.1 SPATIO-TEMPORAL FORECASTING

A central challenge in spatio-temporal forecasting is modeling temporal dependencies within each node and spatial dependencies across nodes. Recurrent models such as LSTM (Hochreiter & Schmidhuber, 1997) capture temporal dynamics. DCRNN (Li et al., 2018), STGCN (Yu et al., 2018), Graph WaveNet (Wu et al., 2019), and AGCRN (Bai et al., 2020) combine recurrent or temporal convolutional modeling with graph-based spatial modeling, while ASTGCN (Guo et al., 2019), ASTGNN (Guo et al., 2021) and PDFormer (Jiang et al., 2023) incorporate attention to capture spatio-temporal dependencies. These models achieve strong in-domain performance. However, some rely on node-specific embeddings or adjacency parameters tied to the training network, limiting their direct application to unseen nodes or networks. Recently, pretrained language model (PLM)-based approaches and spatio-temporal foundation models have demonstrated strong forecasting and generalization capabilities. OFA (Zhou et al., 2023), ST-LLM (Liu et al., 2024a), STG-LLM, (Liu et al., 2024b) and STD-PLM (Huang et al., 2025) adapt pretrained language models to spatio-temporal modeling, while UniST (Yuan et al., 2024) learns transferable knowledge from diverse spatio-temporal data. These advances highlight the value of pretrained representations and transferable knowledge. However, such capabilities alone do not ensure that a model can identify alternative predictive evidence when node behavior changes or previously available observations become missing. Some studies investigate out-of-distribution (OOD) spatio-temporal forecasting to improve robustness across environments, including spatial and temporal shifts in STONE (Wang et al., 2024) and STOP (Ma et al., 2025b). While these studies focus on particular deployment challenges, we consider a broader perspective involving changes in spatial domains, temporal distributions, and observation availability.

## 2.2 LONG-CONTEXT SPATIO-TEMPORAL MODELING

PatchTST (Nie et al., 2023), iTransformer (Liu et al., 2024c), and SparseTSF (Lin et al., 2026) demonstrate the value of extended temporal dependencies but do not explicitly incorporate spatial graph structure. SSTBAN (Guo et al., 2023), STID (Shao et al., 2022), BiST (Ma et al., 2025a), and FaST (Zhao et al., 2026) explore efficient spatio-temporal modeling, while RAST (Ruan et al., 2026) retrieves historical patterns to augment prediction. STMAE (Sun et al., 2024) and BigST (Han et al., 2024) further exploit historical representations through separate pretraining stages. These studies motivate efficient use of historical information for forecasting. These studies motivate efficient historical modeling. VaryST further exploits long-term observations directly to guide spatial support selection and predictive evidence organization.

## 3 PRELIMINARIES

Spatio-Temporal Data. We consider a spatio-temporal system defined over a spatial graph $\bar { \mathcal { G } } = ( \nu , \mathcal { A } )$ , where $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { N } \}$ is the set of N spatial nodes and A represents their spatial relations. At time t, the observations are denoted by $\mathbf { \dot { X } } _ { t } \in \mathbb { R } ^ { N \times C }$ , where $C$ is the number of features. Their availability is indicated by a binary mask $\mathbf { M } _ { t } \mathbf { \bar { \Pi } } \in \{ 0 , 1 \} ^ { N \times C }$ , where $M _ { t , i , c } = 1$ if the corresponding observation is available.

![](images/6308b716f7a3eab740354281cb60c7e5e596cf2a93398ae11049265c5ce1ff1c.jpg)  
Figure 2: The architecture of VaryST. It contains Long-term Observation-derived Representations, Dynamic Functional Support Node Retrieval, and Target-centric Hierarchical Contextual Decoding.

Varying Deployment Environments. We characterize deployment environments along three dimensions: temporal distributions, spatial domains, and observation availability. These dimensions may differ between training and deployment or change during deployment. Evolving temporal distributions refer to changes in observation distributions over time, $P _ { t } ( \mathbf { X } ) \neq P _ { t ^ { \prime } } ( \mathbf { X } )$ , due to seasonality, long-term trends, or changing operating regimes. Changing spatial domains refer to changes in the underlying graph, $\mathcal { G } \neq \mathcal { G } ^ { \intercal } .$ , including the addition or removal of nodes and spatial relations, or transfer to a different network. Varying observation availability refers to changes in the distribution of observation masks, $P ( \mathbf { M } ) \neq P ^ { j } ( \mathbf { M } )$ , reflecting different missing rates or patterns.

Generalizable Spatio-Temporal Forecasting. Given observations $\mathbf { X } _ { t - T + 1 : t }$ of $\mathcal { G }$ over the past $T$ time steps and their availability masks $\mathbf { M } _ { t - T + 1 : t }$ , we aims to predict the future Q steps:

$$
\begin{array} { r } { \hat { \mathbf { X } } _ { t + 1 : t + Q } = f _ { \theta } \left( \mathbf { X } _ { t - T + 1 : t } , \mathbf { M } _ { t - T + 1 : t } ; \mathcal { G } \right) . } \end{array}\tag{1}
$$

Our goal is to learn a forecasting model that generalizes beyond its training environment to deployment environments with different temporal distributions, spatial graphs, or observation availability. For spatial and temporal generalization, we consider both zero-shot forecasting without parameter updates and few-shot adaptation using limited target-environment training data. For changes in observation availability, we evaluate generalization to unseen missing rates and patterns without retraining.

## 4 METHODOLOGY

We propose VaryST, a compact forecasting framework trained from scratch that dynamically selects and organizes predictive evidence for varying deployment environments. Long-Term Observation-Derived Representations characterize each node’s current state without fixed node-specific embeddings and guide Dynamic Functional Support Node Retrieval, allowing predictive evidence to adapt to changing spatial domains and temporal regimes. Target-Centric Hierarchical Contextual Decoding organizes recent and long-term target states, retrieved functional supports, and global infor mation into a compact context spanning complementary temporal scales and spatial scopes, and further incorporates target-relative structural information through hop-aware encoding. Finally, Importance-Guided Observation Perturbation encourages the model to exploit alternative temporal and spatial evidence, improving robustness to varying observation availability.

## 4.1 LONG-TERM OBSERVATION-DERIVED REPRESENTATIONS

VaryST represents each node through its available long-term observations rather than a fixed nodespecific embedding, enabling node representations to adapt to the current deployment environment.

Linear interpolation and temporal differences. First-order differences help characterize temporal fluctuations, but computing them directly from incomplete observations may introduce artificial changes around missing positions. Therefore, given observations X and availability mask M, we first apply linear interpolation $\widetilde { \mathbf { X } } = \mathrm { I n t e r p } ( \mathbf { X } , \mathbf { M } )$ , and then compute $\Delta \widetilde { \mathbf X } _ { t } = \widetilde { \mathbf X } _ { t } - \widetilde { \mathbf X } _ { t - 1 }$ , initializing the first difference to zero.

Joint patch encoding. We partition $\widetilde { \mathbf { X } } , \Delta \widetilde { \mathbf { X } }$ , and M into $T _ { p } = \lceil T / L _ { p } \rceil$ temporally aligned patches. For node $v _ { n }$ , the i-th patch representation is $\mathbf { z } _ { n , i } = [ \mathbf { P } _ { n , i } \| \Delta \mathbf { P } _ { n , i } \| \mathbf { M } _ { n , i } ^ { p } ]$ , where ∥ denotes featurewise concatenation. The patch sequence is encoded via a lightweight causal Transformer with learnable temporal order embedding $\mathbf { E } ^ { j o s } \in \mathbb { R } ^ { T _ { p } \times d }$ of patches:

$$
\mathbf { H } _ { n } = \operatorname { E n c } \left( \operatorname { P r o j } ( \mathbf { Z } _ { n } ) + \mathbf { E } ^ { p o s } \right) \in \mathbb { R } ^ { T _ { p } \times d } .\tag{2}
$$

Its final hidden state $\mathbf s _ { n } = \mathbf H _ { n } [ T _ { p } ] \in \mathbb R ^ { d }$ summarizes all preceding historical patches and serves as the dynamic node representation.

Temporal and structural priors. We further incorporate lightweight node-ID-free temporal and structural priors, including time-of-day, day-of-week, and Laplacian positional embeddings (Huang et al., 2025). The resulting dynamic node representation is

$$
\mathbf { a } _ { n } = \mathbf { s } _ { n } + \mathbf { e } _ { t } ^ { t o d } + \mathbf { e } _ { t } ^ { d o w } + \mathbf { e } _ { n } ^ { l a p } .\tag{3}
$$

Because each $a _ { n }$ is derived from observations rather than a learned node identity, the same encoder can construct representations for unseen nodes and update those of existing nodes as their operating regimes change. These dynamic node representations are then used for functional support retrieval and target-centric contextual decoding.

## 4.2 DYNAMIC FUNCTIONAL SUPPORT NODE RETRIEVAL

Predefined graph neighborhoods or spatial dependencies learned over fixed training nodes may become unreliable when the spatial domain changes, informative nodes lie outside the predefined neighborhood, or previously useful observations become unavailable. We therefore retrieve support nodes dynamically from their current representations rather than inheriting a fixed support set from training.

For each target node $v _ { n }$ , VaryST measures its affinity with every candidate node $v _ { j }$ using their dynamic representations ${ \bf a } _ { n }$ and $\mathbf { a } _ { j }$ . Specifically, we employ an RBF similarity:

$$
\kappa _ { n , j } = \exp \left( - \left\| \mathbf { a } _ { n } - \mathbf { a } _ { j } \right\| _ { 2 } ^ { 2 } \right) , \qquad j \neq n .\tag{4}
$$

Because ${ \bf a } _ { n }$ captures the current observation-derived state together with temporal and structural priors, $\kappa _ { n , j }$ reflects functional compatibility under the current context rather than predefined spatial proximity. We omit the RBF bandwidth because the similarity is used only for ranking, for which any fixed positive bandwidth preserves the same ordering.

The K nodes with the highest similarities form the functional supporting set of target node $v _ { n } .$

$$
\begin{array} { r } { S _ { n } = \mathrm { T o p K } _ { j \neq n } \left( \kappa _ { n , j } \right) = \big \{ v _ { n , 1 } , \dotsc , v _ { n , K } \big \} . } \end{array}\tag{5}
$$

The retrieved nodes are ordered by decreasing similarity, and their representations are subsequently used as supporting tokens in the target-centric decoder.

Unlike predefined graph neighborhoods, the retrieved support set adapts to current node representations and can change with temporal regimes, spatial domains, and observation availability, allowing VaryST to identify alternative predictive evidence when previously relied-upon nodes become less informative or unavailable. Since retrieval requires neither node-specific identifiers nor traininggraph-specific support sets, it is naturally applicable to unseen graphs. Dynamic retrieval thus determines which nodes enter the predictive context, while the subsequent decoder determines how they influence the target.

## 4.3 TARGET-CENTRIC HIERARCHICAL CONTEXTUAL DECODING

After retrieving functionally relevant support nodes, VaryST organizes predictive evidence into a target-centric hierarchical context spanning complementary temporal scales and spatial scopes.

Multi-scale spatio-temporal context organization. For a target node $v _ { n }$ , we first construct a global token by pooling the dynamic representations of all nodes:

$$
\mathbf { g } = \mathrm { M e a n P o o l } ( \mathbf { a } _ { 1 } , \dots , \mathbf { a } _ { N } ) \in \mathbb { R } ^ { d } .\tag{6}
$$

To complement the long-term target representation ${ \bf a } _ { n }$ , we explicitly preserve its recent dynamics by projecting the latest R observations and first-order differences:

$$
\begin{array} { r } { \mathbf { r } _ { n } = \mathrm { M L P } _ { r } \left( [ \widetilde { \mathbf { X } } _ { t - R + 1 : t , n } | | \Delta \widetilde { \mathbf { X } } _ { t - R + 1 : t , n } ] \right) . } \end{array}\tag{7}
$$

Given the retrieved supporting set $S _ { n }$ , the target-specific context is organized as:

$$
\mathbf { C } _ { n } ^ { 0 } = \big [ \underbrace { \textbf { g } } _ { \mathrm { g l o b a l ~ t o k e n } } \ \big \| \underbrace { \mathbf { a } _ { n , 1 } , \dots , \mathbf { a } _ { n , K } } _ { \mathrm { f u n c t i o n a l ~ s u p p o r t i n g ~ t o k e n s } } \ \big \| \underbrace { \mathbf { r } _ { n } } _ { \mathrm { r e c e n t ~ t a r g e t ~ t o k e n } } \ \big \| \underbrace { \mathbf { a } _ { n } } _ { \mathrm { l o n g - t e r m ~ t a r g e t ~ t o k e n } } \big ] .\tag{8}
$$

This hierarchy combines the target’s immediate and long-term dynamics with functional spatial support and global system context, providing complementary evidence across temporal scales and spatial scopes.

Hop-aware relative structural encoding. Functional retrieval identifies support nodes according to their current functional similarity, but does not distinguish their structural relations to the target. To preserve such information, we encode the target-relative graph distance of each supporting node. For $v _ { n , j } \in S _ { n } ,$ , we compute its shortest-path hop distance to the target,

$$
h _ { n , j } = \mathrm { H o p D i s t } ( v _ { n } , v _ { n , j } ) ,\tag{9}
$$

and map the discretized distance to a learnable phase vector $\phi _ { h _ { n , j } } \in \mathbb { R } ^ { d / 2 }$ . We then apply a rotary transformation to the corresponding supporting token:

$$
\bar { \bf a } _ { n , j } = { \bf R } ( \phi _ { h _ { n , j } } ) { \bf a } _ { n , j } , \qquad j = 1 , \ldots , K ,\tag{10}
$$

where $\mathbf { R } ( \cdot )$ applies a block-diagonal rotation to each pair of feature dimensions. Importantly, only the supporting-node tokens are structurally encoded; the global, recent-target, and long-term-target tokens remain unchanged. Since hop distance describes a target-relative relation rather than node identity or absolute graph position, the same encoding can be directly reused on unseen graphs, providing an inductive structural prior for cross-graph decoding.

Target-centric contextual decoding. The resulting context is processed by an L-layer lightweight Transformer decoder:

$$
{ \bf C } _ { n } ^ { L } = { \mathrm { D e c o d e r } } \left( { \bf C } _ { n } ^ { 0 } \right) .\tag{11}
$$

We disable positional encoding in the decoder so that no sequential ordering is imposed on the heterogeneous context tokens; spatial structure is introduced exclusively through the target-relative hop encoding of support nodes. The long-term target token is placed at the final position, and its output state $\mathbf { \bar { o } } _ { n } = \hat { \mathbf { C } } _ { n } ^ { \hat { L } } [ - 1 ]$ aggregates information from the global, functional-support, and recenttarget contexts. Forecasts are produced as:

$$
\widehat { \mathbf { Y } } _ { n } = \mathrm { M L P } \left( \mathrm { L N } ( \mathbf { o } _ { n } ) \right) , \qquad \widehat { \mathbf { Y } } _ { n } \in \mathbb { R } ^ { Q \times C } .\tag{12}
$$

## 4.4 TRAINING WITH IMPORTANCE-GUIDED OBSERVATION PERTURBATION

To improve robustness to varying observation availability, we introduce Importance-Guided Observation Perturbation (IGOP) during training. Rather than masking observations randomly, IGOP deliberately removes historical patches on which the model relies most, creating harder, importanceguided perturbations that encourage the model to exploit alternative temporal and spatial evidence.

Patch Importance Estimation. Given the long-term patch representations of node $v _ { n } , \mathbf { H } _ { n } \ =$ $[ \mathbf { h } _ { n , 1 } , \ldots , \mathbf { \bar { h } } _ { n , T _ { p } } ]$ , we use the attention scores produced by the joint patch encoder to estimate patch importance. Let $\pmb { \alpha } _ { n } = [ \alpha _ { n , 1 } , . . . , \alpha _ { n , T _ { p } } ] ,$ , where a larger $\alpha _ { n , p }$ indicates stronger dependence on patch p. These scores are used only to guide perturbation construction.

Importance-Guided Observation Perturbation. For each node, IGOP selects the $K _ { m }$ patches with the highest attention-derived importance scores,

$$
\mathcal { P } _ { n } ^ { \mathrm { i m p } } = \mathrm { T o p K } _ { p \in \{ 1 , . . . , T _ { p } \} } \left( \alpha _ { n , p } , K _ { m } \right) ,\tag{13}
$$

All observations within the selected patches are masked to obtain a perturbed observation mask M<sup>′</sup>. Since each patch corresponds to a contiguous temporal interval, the perturbation naturally

Table 1: Dataset statistics.
<table><tr><td>Dataset</td><td>Nodes</td><td>Time Span</td><td>Interval</td><td>Usage</td></tr><tr><td>PEMS03</td><td>358</td><td>09/01/2018-11/30/2018</td><td>5 min</td><td>RQ1/3/5/6</td></tr><tr><td>PEMS04</td><td>307</td><td>01/01/2018–02/28/2018</td><td>5 min</td><td>RQ1/3/4</td></tr><tr><td>PEMS07</td><td>883</td><td>05/01/2017-08/31/2017</td><td>5 min</td><td>RQ1/3/7</td></tr><tr><td>PEMS08</td><td>170</td><td>07/01/2016- 08/31/2016</td><td>5 min</td><td>RQ1/3/4/5/6</td></tr><tr><td>PM2.5</td><td>12</td><td>03/01/2013-02/28/2017</td><td>60 min</td><td>RQ1</td></tr><tr><td>PEMSD3-Stream</td><td>655</td><td>2011-2017</td><td>5 min</td><td>RQ2</td></tr></table>

produces block-wise observation loss. The interpolated observations and temporal differences are then recomputed under M<sup>′</sup> to prevent masked values from leaking into subsequent representations.

Two-Stage Perturbation Training. For each mini-batch, a gradient-free probe first estimates patch importance and constructs M<sup>′</sup>. VaryST is then optimized on the perturbed history with the original forecasting target:

$$
{ \widehat { \mathbf { Y } } } = f _ { \theta } ( \mathbf { X } , \mathbf { M } ^ { \prime } ) , \qquad { \mathcal { L } } _ { \mathrm { p r e d } } = { \frac { 1 } { N Q C } } \left\| { \widehat { \mathbf { Y } } } - \mathbf { Y } \right\| _ { 1 } .\tag{14}
$$

The probe participates only in perturbation construction. No reconstruction objective is imposed on the masked patches; instead, forecasting supervision directly encourages the model to reorganize its reliance on the remaining temporal and spatial evidence.

## 5 EXPERIMENTS

We evaluate the proposed VaryST on real-world traffic datasets to address the following research questions: RQ1: Does VaryST remain competitive under standard in-domain forecasting across different spatio-temporal systems? RQ2: Can VaryST remain effective under evolving temporal distributions? RQ3: Can VaryST generalize to unseen spatial environments? RQ4: Is VaryST robust to varying observation availability? RQ5: How do the key designs of VaryST contribute to its effectibeness? RQ6: How do model hyperparameters affect model performance? RQ7: What is the computational efficiency of VaryST?

## 5.1 EXPERIMENTAL SETUP

Dataset. Table 1 summarizes the benchmarks. PEMS03/04/07/08 cover in-domain and crossnetwork forecasting, with PEMS04/08 additionally evaluated under random (RM) and block (CM) missingness. PM2.5 evaluates non-traffic generalization, while the 7-year PEMSD3-Stream with an expanding network assesses long-term temporal shift. All benchmarks use a 6:2:2 chronological split to forecast 12 future steps from 2016 historical steps (168 for PM2.5), measured by MAE, RMSE, and MAPE.

Baselines. We compare VaryST with three groups of models for standard forecasting: classical spatio-temporal and time-series models, including ASTGCN (Guo et al., 2019), AGCRN (Bai et al., 2020), SSTBAN (Guo et al., 2023), PDFormer (Jiang et al., 2023), iTransformer (Liu et al., 2024c), and RAST (Ruan et al., 2026); pre-trained LLM-based models, including OFA (Zhou et al., 2023), STLLM (Liu et al., 2024a), STGLLM (Liu et al., 2024b), and STD-PLM (Huang et al., 2025); and extended-history models, where STID (Shao et al., 2022), iTransformer (Liu et al., 2024c), Patch-STG (Fang et al., 2025), BiST (Ma et al., 2025a), FaST (Zhao et al., 2026), and SparseTSF (Lin et al., 2026) are adapted to the same 2016-step history as VaryST. For temporal generalization, we compare with STOP (Ma et al., 2025b), which is specifically designed for temporal distribution shifts, and STD-PLM, which reports strong generalization forecasting ability. For spatial generalization, we compare with STD-PLM, the strongest baseline that supports cross-network transfer. For missing-observation forecasting, we select FaST, STID, and BiST for their strong standardforecasting performance and efficiency, and use linear interpolation to pre-fill their missing inputs.

Settings. All experiments are conducted on a single NVIDIA RTX A4000 GPU under PyTorch 2.11. VaryST is trained from scratch with the MAE loss, Muon and AdamW optimizers with learning rate 0.001, batch size 128. Unless otherwise specified, we set the history length $T = 2 0 1 6$ , patch length $L _ { p } = 3 6$ , representation dimension $d = 9 6 .$ , number of retrieved support nodes $K = 6 4$ , recent window length $R = 1 2$ , and decoder depth $L = 3$ . For PM2.5, $T = { \bar { 1 6 } } 8$ and $L _ { p } = 1 2$ , while the other settings remain unchanged. Detailed settings are provided in Table 5.

Table 2: Overall prediction performance comparison on four PEMS datasets. Models are categorized into short-term, LLM based and long-term groups. Bold: best, underline: second best.
<table><tr><td rowspan="2">Model</td><td colspan="3">PEMS03</td><td colspan="3">PEMS04</td><td colspan="3">PEMS07</td><td colspan="3">PEMS08</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td></tr><tr><td colspan="10">Models with Short-term History (Input Horizon = 12)</td><td></td><td></td><td></td></tr><tr><td>ASTGCN AGCRN</td><td>17.85 15.98</td><td>29.88</td><td>17.65% 15.23%</td><td>22.42</td><td>34.75</td><td>15.87% 12.97%</td><td>25.98 22.37</td><td>39.65</td><td>11.84%</td><td>18.86 15.95</td><td>28.55 25.22</td><td>12.50%</td></tr><tr><td>SSTBAN</td><td>15.90</td><td>28.25 26.11</td><td>17.27%</td><td>19.83 18.88</td><td>32.26 31.10</td><td>12.56%</td><td>20.17</td><td>36.55 33.45</td><td>9.12% 8.93%</td><td>14.39</td><td>24.19</td><td>10.09% 10.06%</td></tr><tr><td>PDFormer</td><td>14.74</td><td>25.59</td><td>15.35%</td><td>18.31</td><td>29.97</td><td>12.10%</td><td>19.83</td><td>32.87</td><td>8.53%</td><td>13.58</td><td>23.51</td><td>9.05%</td></tr><tr><td>iTransformer</td><td>19.48</td><td>31.20</td><td>17.84%</td><td>22.56</td><td>35.21</td><td>16.29%</td><td>24.70</td><td>37.91</td><td>11.40%</td><td>20.05</td><td>31.90</td><td>11.99%</td></tr><tr><td>RAST</td><td>15.20</td><td>25.81</td><td>16.12%</td><td>18.39</td><td>29.93</td><td>12.43%</td><td>19.52</td><td>32.73</td><td>8.23%</td><td>14.16</td><td>23.33</td><td>9.27%</td></tr><tr><td colspan="9">Pre-trained LLM-based Models</td><td></td><td></td><td></td></tr><tr><td>OFA</td><td>20.96</td><td>33.43</td><td>19.11%</td><td>27.37</td><td>42.99</td><td>17.97%</td><td>30.53</td><td>47.51</td><td>12.98%</td><td>21.89</td><td>34.63</td><td>13.30%</td></tr><tr><td>STLLM</td><td>17.25</td><td>27.25</td><td>22.96%</td><td>19.00</td><td>30.35</td><td>13.55%</td><td>21.48</td><td>34.07</td><td>10.20%</td><td>14.67</td><td>23.50</td><td>10.63%</td></tr><tr><td>STGLLM</td><td>15.26</td><td>24.11</td><td>15.73%</td><td>20.00</td><td>32.11</td><td>13.69%</td><td>21.98</td><td>35.02</td><td>9.72%</td><td>15.53</td><td>24.74</td><td>10.15%</td></tr><tr><td>STD-PLM</td><td>14.59</td><td>25.36</td><td>14.92%</td><td>18.16</td><td>30.21</td><td>11.89%</td><td>19.25</td><td>32.84</td><td>8.06%</td><td>13.31</td><td>23.19</td><td>8.84%</td></tr><tr><td colspan="9">Models with Long-term History</td><td></td><td></td><td></td></tr><tr><td>STID†</td><td>15.56</td><td>25.37</td><td>15.25%</td><td>18.40</td><td>30.10</td><td>13.19%</td><td>19.11</td><td>31.71</td><td>8.45%</td><td>13.42</td><td></td><td></td></tr><tr><td>iTransformer†</td><td>17.25</td><td>27.70</td><td>18.62%</td><td>20.82</td><td>32.79</td><td>14.42%</td><td>22.19</td><td>34.77</td><td>10.00%</td><td>15.31</td><td>22.48 23.94</td><td>9.37% 10.61%</td></tr><tr><td>PatchSTG†</td><td>16.92</td><td>28.30</td><td>17.13%</td><td>18.67</td><td>30.44</td><td>13.10%</td><td>20.00</td><td>32.29</td><td>9.51%</td><td>13.73</td><td>23.03</td><td>9.58%</td></tr><tr><td>BiST†</td><td>15.01</td><td>24.64</td><td>14.87%</td><td>18.46</td><td>29.86</td><td>13.22%</td><td>19.55</td><td>32.53</td><td>8.87%</td><td>13.71</td><td>22.79</td><td>9.57%</td></tr><tr><td>FaST†</td><td>16.68</td><td>26.98</td><td>16.12%</td><td>18.57</td><td>30.10</td><td>12.85%</td><td>18.92</td><td>31.92</td><td>8.05%</td><td>14.22</td><td>23.21</td><td>10.31%</td></tr><tr><td>SparseTSF†</td><td>19.87</td><td>33.24</td><td>22.40%</td><td>19.91</td><td>31.94</td><td>14.51%</td><td>21.43</td><td>34.66</td><td>9.56%</td><td>16.12</td><td>25.74</td><td>11.52%</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VaryST</td><td>13.95</td><td>22.73</td><td>13.59%</td><td>17.56</td><td>29.35</td><td>11.81%</td><td>18.21</td><td>31.05</td><td>7.64%</td><td>12.89</td><td>21.99</td><td>8.50%</td></tr></table>

<sup>†</sup> Models adapted to long-term inputs by us for fair comparison (2016 steps).

## 5.2 STANDARD FORECASTING PERFORMANCE (RQ1)

Table 2 reports 12-step forecasting results on four PEMS benchmarks. Overall, long-history models outperform short-history baselines (e.g., PDFormer, RAST) and LLM-based methods (STD-PLM). VaryST achieves the best performance across all four datasets, reducing MAE by 3.16%–4.39%, RMSE by 1.71%–5.72%, and MAPE by 0.67%–8.61% over the strongest baselines. Its advantage over competitive long-history models (STID, FaST,BiST) demonstrates that simply feeding longer sequences is insufficient; VaryST crucially leverages long-term observations to derive dynamic node states and retrieve functional supports.

Table 3: Performance comparison on PM2.5.
<table><tr><td>Model</td><td>MAE</td><td>RMSE</td><td>MAPE</td></tr><tr><td>ASTGCN</td><td>27.42</td><td>48.70</td><td>52.4%</td></tr><tr><td>AGCRN</td><td>28.20</td><td>49.57</td><td>50.1%</td></tr><tr><td>SSTBAN</td><td>30.22</td><td>53.13</td><td>53.1%</td></tr><tr><td>PDFormer</td><td>26.65</td><td>47.08</td><td>51.5%</td></tr><tr><td>iTransformer</td><td>29.56</td><td>51.80</td><td>55.9%</td></tr><tr><td>STD-PLM</td><td>27.55</td><td>48.88</td><td>53.3%</td></tr><tr><td> ${ \operatorname { S T I D } } ^ { \dag }$ </td><td>31.22</td><td>54.88</td><td>64.1%</td></tr><tr><td>iTransformer†</td><td>29.97</td><td>50.52</td><td>58.5%</td></tr><tr><td> $\mathrm { B i S T ^ { \dagger } }$ </td><td>28.11</td><td>47.28</td><td>54.5%</td></tr><tr><td> $\mathrm { F a S T ^ { \dagger } }$ </td><td>30.66</td><td>51.99</td><td>59.2%</td></tr><tr><td>SparseTSF†</td><td>39.64</td><td>62.34</td><td>85.2%</td></tr><tr><td>VaryST</td><td>26.44</td><td>47.18</td><td>49.91%</td></tr></table>

<sup>†</sup>Models adapted to long-term inputs (168 steps).

Table 3 evaluates generalization beyond traffic on PM2.5, which features hourly observations across only 12 stations without an explicit graph. VaryST achieves the best MAE and MAPE alongside a competitive RMSE, demonstrating that dynamic target-centric context construction generalizes effectively to non-traffic domains.

## 5.3 TEMPORAL GENERALIZATION (RQ2)

To evaluate temporal generalization, models are trained only on 2011 data of PEMSD3-Stream and tested across 2012–2017 without updates (zero-shot transfer). As shown in Figure 3a, VaryST consistently outperforms STOP—the strongest temporal-robust baseline—across all years, reducing MAE by 16.22%–21.97%, RMSE by 15.34%–21.27%, and MAPE by 15.43%–18.08%. While STOP seeks invariant representations via distributionally robust optimization, VaryST adapts directly to evolving dynamics by deriving node states and dynamically retrieving functional supports from current long-term observations.

![](images/a237505fbc8868c85991b1f872eae931fab47eb92f38b28201fca433aa2fbfed.jpg)  
(a) Per-year zero-shot metrics on PEMSD3-Stream (2012–2017).

![](images/97225b4b23f1ec36d89b96829f8dba559de0052ceb2d9816d0e9c12e56017d7c.jpg)

![](images/e35ab5a5dd8d9417e2032073f0e258cbacd6d6428408d7c83ee11e5349226f6c.jpg)

![](images/9129ae14d28e135f9af9bd5659cffabe0de118b58c37864d308ebc02378d5d80.jpg)  
(c) MAE under RM and CM at 10– 70% rates.

Figure 3: Distribution-shift robustness of VaryST.  
![](images/b5e453acee45a7f24e0288a22ec852d0cf3d9decc2aa719f20234b05e9073890.jpg)  
(a) Ablation study on PEMS03.

![](images/e960021f546adf3574191c75f707692a9ba4e5a2c3336d65b8a1b8f140fd11f5.jpg)  
(b) IGOP ablation under missing-rate shift.

![](images/24ab347b53ec69ccf06155bd986c1533485e55ccd236def10f66d990327e0925.jpg)  
(c) Hyperparameter sensitivity on PEMS03.

![](images/ce25ce4c12f8d8ed3801cacc5b9fd372a51942fdaa15d959658bba73d27e40ab.jpg)  
(d) Efficiency on PEMS07.  
Figure 4: Ablation, hyperparameter and efficiency studies.

## 5.4 SPATIAL GENERALIZATION (RQ3)

We evaluate spatial generalization via cross-network transfer across four PEMS benchmarks under zero-shot and few-shot (1%, 5%, 10% target data) settings (full 12 directions in Appendix C.2). Figure 3b highlights results on PEMS04. Under zero-shot transfer, VaryST degrades by only 3.68%/5.30%/8.89% (RMSE/MAE/MAPE) relative to its in-domain reference, whereas STD-PLM degrades by 52.36%/63.86%/91.33%. This gap stems from VaryST constructing representations and functional supports dynamically from current observations rather than static graph topology. With few-shot adaptation, VaryST with 5%–10% target data already surpasses competitive full-data in-domain models and consistently outperforms STD-PLM across all budgets.

## 5.5 ROBUSTNESS TO OBSERVATIONS AVAILABILITY(RQ4)

We evaluate cross-network transfer across four PEMS benchmarks under zero-shot and few-shot (1%, 5%, 10%) settings (Appendix C.2). On PEMS04 (Figure 3b), zero-shot VaryST degrades by only 3.68%/5.30%/8.89% (RMSE/MAE/MAPE) relative to in-domain, whereas STD-PLM degrades by 52.36%/63.86%/91.33%, as VaryST dynamically constructs spatial supports from observations rather than static topology. With 5%–10% target data, VaryST already surpasses full-data in-domain baselines and outperforms STD-PLM across all budgets.

## 5.6 ABLATION STUDIES (RQ5)

Ablation Study On PEMS03 (Figure 4a), removing long-term history increases MAE from 13.95 to 15.95, validating the role of long observations in dynamic node representations. For spatial context construction, replacing RBF similarity with cosine raises MAE to 14.33, while removing hop-aware encoding raises it to 14.42, justifying our affinity metric and target-relative structural design. For perturbation training, IGOP trained at a 30% missing rate reduces MAE by 0.39 at a 70% test missing rate on PEMS08 without retraining (Figure 4b). Full results are in Appendix C.4.

Hyperparameter Sensitivity We evaluate context length $T \in \{ 3 6 , 2 8 8 , 8 6 4 , 2 0 1 6 , 4 0 3 2 \}$ and support count K ∈ {16, 32, 64, 128} (Figure 4c). Extending history from 36 to the default 2016 steps reduces MAE by 8.8% on PEMS03 and 12.8% on PEMS08. For support retrieval, increasing K from 16 to the default 64 lowers MAE by 2.9% on PEMS03, confirming $K = 6 4$ as an optimal trade-off. Full results are in Appendix C.5.

## 5.7 EFFICIENCY ANALYSIS (RQ7)

Figure 4d evaluates inference efficiency on PEMS07 (883 nodes). VaryST achieves the lowest MAE (18.21), reducing error by 3.8% with faster inference than FaST<sup>†</sup>, while outperforming STD-PLM with substantially lower memory footprint. This efficiency stems from our compact predictive context: long-context encoding compresses each node’s history into a single representation, limiting the decoder to process only $\bar { K + 3 }$ tokens per target.

## 6 CONCLUSION

This paper presents VaryST, which addresses generalizable spatio-temporal forecasting by dynamically constructing predictive contexts. Trained from scratch, VaryST demonstrates superior indomain accuracy, robust zero- and few-shot generalization across spatial and temporal shifts, and resilience to varying observation availability.

## REPRODUCIBILITY STATEMENT

All datasets are public benchmarks (Table 1). The training scripts, together with the complete command-line configuration of every experiment, will be released upon publication.

## REFERENCES

Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting. In Advances in Neural Information Processing Systems, volume 33, pp. 17804–17815, 2020.

Yuchen Fang, Yuxuan Liang, Bo Hui, Zezhi Shao, Liwei Deng, Xu Liu, Xinke Jiang, and Kai Zheng. Efficient large-scale traffic forecasting with transformers: A spatial data management perspective. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1, pp. 307–317, 2025.

Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, and Huaiyu Wan. Attention based spatialtemporal graph convolutional networks for traffic flow forecasting. In Proceedings of the AAAI conference on artificial intelligence, volume 33, pp. 922–929, 2019.

Shengnan Guo, Youfang Lin, Huaiyu Wan, Xiucheng Li, and Gao Cong. Learning dynamics and heterogeneity of spatial-temporal graph data for traffic forecasting. IEEE Transactions on Knowledge and Data Engineering, 34(11):5415–5428, 2021.

Shengnan Guo, Youfang Lin, Letian Gong, Chenyu Wang, Zeyu Zhou, Zekai Shen, Yiheng Huang, and Huaiyu Wan. Self-supervised spatial-temporal bottleneck attentive network for efficient longterm traffic forecasting. In IEEE International Conference on Data Engineering, pp. 1585–1596. IEEE, 2023.

Jindong Han, Weijia Zhang, Hao Liu, Tao Tao, Naiqiang Tan, and Hui Xiong. BigST: Linear complexity spatio-temporal graph neural network for traffic forecasting on large-scale road networks. Proceedings ofthe VLDB Endowment, 17(5):1081–1090, 2024. doi: 10.14778/3641204.3641217.

Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. ¨ Neural Computation, 9(8): 1735–1780, 1997.

Yiheng Huang, Xiaowei Mao, Shengnan Guo, Yubin Chen, Junfeng Shen, Tiankuo Li, Youfang Lin, and Huaiyu Wan. STD-PLM: Understanding Both Spatial and Temporal Properties of Spatial Temporal Data with PLM. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 11817–11825, 2025.

Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, and Jingyuan Wang. PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction. In Proceedings of the AAAI conference on artificial intelligence, volume 37, pp. 4365–4373, 2023.

Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neural network: Data-driven traffic forecasting. In International Conference on Learning Representations, 2018.

Shengsheng Lin, Weiwei Lin, Wentai Wu, Haojun Chen, and C. L. Philip Chen. SparseTSF: Lightweight and Robust Time Series Forecasting via Sparse Modeling. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(1):170–183, 2026. doi: 10.1109/TPAMI.2025. 3602445.

Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, and Rui Zhao. Spatialtemporal large language model for traffic prediction. arXiv preprint arXiv:2401.10134, 2024a.

Lei Liu, Shuo Yu, Runze Wang, Zhenxun Ma, and Yanming Shen. How can large language models understand spatial-temporal data? arXiv preprint arXiv:2401.14192, 2024b.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting. In International Conference on Learning Representations, 2024c.

Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, and Yang Wang. BiST: A Lightweight and Efficient Bi-directional Model for Spatiotemporal Prediction. Proceedings of the VLDB Endowment, 18(6):1663–1676, 2025a.

Jiaming Ma, Binwu Wang, Pengkun Wang, Zhengyang Zhou, Xu Wang, and Yang Wang. Robust spatio-temporal centralized interaction for ood learning. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 42165–42192, 2025b.

Yuqi Nie, Nam H Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations, 2023.

Weilin Ruan, Xilin Dang, Ziyu Zhou, Sisuo Lyu, and Yuxuan Liang. A retrieval augmented spatiotemporal framework for traffic prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pp. 39163–39172, 2026.

Zezhi Shao, Zhao Zhang, Fei Wang, Wei Wei, and Yongjun Xu. Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting. In Proceedings of the 31st ACM international conference on information & knowledge management, pp. 4454–4458, 2022.

Jiarui Sun, Yujie Fan, Chin-Chia Michael Yeh, Wei Zhang, and Girish Chowdhary. Revealing the power of masked autoencoders in traffic forecasting. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pp. 4071–4075, 2024. doi: 10.1145/3627673.3679989.

Binwu Wang, Jiaming Ma, Pengkun Wang, Xu Wang, Yudong Zhang, Zhengyang Zhou, and Yang Wang. Stone: A spatio-temporal ood learning framework kills both spatial and temporal shifts. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 2948–2959, 2024.

Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, and Chengqi Zhang. Graph wavenet for deep spatial-temporal graph modeling. In Proceedings of the International Joint Conference on Artificial Intelligence, pp. 1907–1913, 2019. doi: 10.24963/ijcai.2019/264.

Bing Yu, Haoteng Yin, and Zhanxing Zhu. Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. In Proceedings of the International Joint Conference on Artificial Intelligence, pp. 3634–3640, 2018. doi: 10.24963/ijcai.2018/505.

Yuan Yuan, Jingtao Ding, Jie Feng, Depeng Jin, and Yong Li. UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction. In Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining, pp. 4095–4106, 2024.

Yiji Zhao, Zihao Zhong, Ao Wang, Haomin Wen, Ming Jin, Yuxuan Liang, Huaiyu Wan, and Hao Wu. FaST: Efficient and Effective Long-Horizon Forecasting for Large-Scale Spatial-Temporal Graphs via Mixture-of-Experts. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowl edge Discovery and Data Mining V. 1, pp. 1975–1986, 2026.

Tian Zhou, Peisong Niu, Xue Wang, Liang Sun, and Rong Jin. One Fits All: Power General Time Series Analysis by Pretrained LM. In Advances in Neural Information Processing Systems, volume 36, 2023.

## APPENDIX

Roadmap. This appendix provides supplementary methodology, comprehensive experimental setups, and complete empirical results. Specifically, Appendix A.1 presents the pseudocode for the training and inference pipelines of VaryST, along with a complexity analysis of each component. Appendix B elaborates on dataset statistics, hyperparameter configurations, evaluation metrics, and implementation details. Due to space limitations in the main paper, Appendix C details the full experimental results, covering temporal and spatial generalization, missing-observation forecasting under rate and pattern shifts, ablation studies, and hyperparameter sensitivity analyses.

## A ALGORITHMS

## A.1 PSEUDO CODE

Algorithm 1 VaryST training and inference. Lines 3–6 run only during training; at inference the   
forward pass executes once with $\mathbf { M } ^ { \prime } = \mathbf { M } .$   
Require: history ${ \mathbf { X } } _ { t - T + 1 : t }$ , mask M, graph G, horizon $Q ;$ patch length $L _ { p } .$ , number of supports $K$   
recent steps $R ,$ masked patches $K _ { m }$   
Ensure: prediction $\widehat { \mathbf { Y } } _ { t + 1 : t + Q }$   
// Stage 1: importance-guided probe (training only)   
1: M<sup>′</sup> ← M   
2: if training then   
3: $\alpha _ { n } \gets$ patch-attention scores of the long-term encoder ▷ gradient-free probe   
4: $\mathcal { P } _ { n } ^ { \mathrm { i m p } } \dot {  } \mathrm { T o p K } ( \pmb { \alpha } _ { n } , K _ { m } ) ; \mathbf { M } ^ { \prime }  \mathrm { M } \mathrm { \bar { a s k } } ( \mathbf { M } , \mathcal { P } ^ { \mathrm { i m p } } )$   
5: end if   
// Stage 2: long-term dynamic representations   
6: Xe ← Interp(X, M<sup>′</sup>); ∆Xe ← first-order differences of $\widetilde { \mathbf { X } }$   
7: for $n = 1 , \ldots , N$ do   
8: $\mathbf { Z } _ { n }  [ \mathrm { P a t c h } ( \widetilde { \mathbf { X } } _ { n } )$ ∥ Patch $( \Delta \widetilde { \mathbf X } _ { n } )$ ∥ Patch $\left( \mathbf { M } _ { n } ^ { \prime } \right) ]$   
9: $\mathbf { a } _ { n } \gets \dot { \mathrm { E n c } } ( \mathrm { P r o j } ( \mathbf { \dot { Z } } _ { n } ) + \mathbf { E } ^ { p o s } ) [ T _ { p } ] ^ { ' } + \mathbf { e } _ { t } ^ { t o d } + { \bf \dot { e } } _ { t } ^ { d o w } + \mathbf { e } _ { n } ^ { l a p }$   
10: end for   
// Stage 3: functional support retrieval and target-centric decoding   
11: $\mathbf { g } \gets \mathrm { M e a n P o o l } ( \mathbf { a } _ { 1 } , \ldots , \mathbf { a } _ { N } )$   
12: for each target node $v _ { n }$ do   
13: $\begin{array} { r } { \mathcal { S } _ { n } \gets \bar { \mathrm { T o p K } } _ { j \neq n } ( \exp ( - \| \mathbf { a } _ { n } - \mathbf { a } _ { j } \| _ { 2 } ^ { 2 } ) ) ; \quad \mathbf { r } _ { n } \gets \mathrm { M L P } _ { r } \left( [ \widetilde { \mathbf { X } } _ { t - R + 1 : t , n } \| \Delta \widetilde { \mathbf { X } } _ { t - R + 1 : t , n } ] \right) } \end{array}$   
14: $\mathbf { C } _ { n } ^ { 0 } \gets [ \mathbf { g } \parallel \mathbf { a } _ { n , 1 } , \hdots , \mathbf { a } _ { n , K } \parallel \mathbf { r } _ { n } \parallel \mathbf { a } _ { n } ]$   
15: $\mathbf { C } _ { n } ^ { L }  \mathrm { D e c o d e r } ( \mathbf { C } _ { n } ^ { 0 } )$ with rotated keys $\widetilde { \mathbf { k } } _ { n , j } = \mathbf { R } ( \varphi _ { h _ { n , j } } ) \mathbf { k } _ { j } , v _ { j } \in S _ { n }$   
16: $\mathbf { o } _ { n }  \mathbf { C } _ { n } ^ { L } [ - 1 ] ; \quad \widehat { \mathbf { Y } } _ { n }  \mathrm { M L P } ( \mathrm { L N } ( \mathbf { o } _ { n } ) )$   
17: end for   
// Stage 4: forecasting loss   
18: if training then   
19: $\begin{array} { r l } {  { \theta  \overline { { \theta } } - \eta \nabla _ { \theta } \mathcal { L } _ { \mathrm { p r e d } } , } } & { { } \mathcal { L } _ { \mathrm { p r e d } } = \frac { 1 } { N Q C } \| \widehat { \mathbf { Y } } - \mathbf { Y } \| _ { 1 } } \end{array}$   
20: end if

## A.2 COMPLEXITY ANALYSIS

Notation. Let N be the number of nodes, B the number of sliding windows per training iteration, and $P = B N$ the number of target instances (one per window–node pair). Let T and $C$ denote the input length and the number of input features, u the unit-window length and $U = T / u$ the number of unit windows, d the token width of the temporal encoder, D the backbone width, H the number of backbone layers, and K the number of retrieved support nodes. With the global token and the recent prompt enabled, each target instance is decoded from a sequence of length $\ell = K + 3$

Temporal encoding. Each window-to-node pair is encoded independently: the unit projection (two-layer MLP over each unit window) costs $O ( T C d )$ per pair, the self-attention over the U unit tokens costs $O ( U ^ { 2 } d + U d ^ { 2 } )$ , the single-query readout attention costs $O ( U d ^ { 2 } )$ , and the output projection costs $O ( d D )$ . With U fixed $\left( U \mathrm { = } 5 6 \right.$ in our experiments), the total per iteration is $O \big ( B N \ : ( T C d + U d ^ { 2 } ) \big )$ , linear in B, N, and T.

Support Node retrieval. For every target instance, similarity to all N candidate nodes of its window is one D-dimensional inner product, followed by a top-K selection, giving $O ( B N ^ { 2 } D )$ per iteration. This is the only step super-linear in N. It is parameter-free and lies outside the gradient graph: candidate keys are detached, and the top-K indices serve only as gather indices, so the step incurs no backward cost. Gradients reach the retrieved nodes through their values, gathered from the gradient-carrying node states.

Context assembly and decoding. Gathering the K support tokens, the additive structural and temporal encodings, and the recent prompt token cost $O ( B N K D )$ in total. The backbone then processes each target instance as a sequence of length $\ell = \dot { K } + O ( 1 )$ , independent ofN: a Transformer layer costs $O ( \overline { { \ell } } ^ { 2 } D + \ell D ^ { 2 } )$ per instance, and the hop-bucket key rotation multiplies the attentionscore computation only by the constant number of buckets (10). Over all $P = B { \dot { N } }$ instances and H layers, decoding costs $O \big ( \mathrm { \bar { \it B N } } \cdot H ( \ell ^ { 2 } D + \ell D ^ { 2 } ) \big )$ . Graph-dependent quantities (Laplacian eigenvectors, hop buckets, and the hop matrix) are computed once per graph, outside the training loop.

Where the quadratic term lives. A graph Transformer attending over all N nodes pays $O ( N ^ { 2 } D )$ inside every backbone layer, in both the forward and the backward pass, and materialises $\dot { N } \times \dot { N }$ attention scores in memory. In Methodology, the quadratic term appears exactly once, in the retrieval matmul above: a single parameter-free, gradient-free matrix product per iteration. All gradientcarrying computation—temporal encoding and the backbone—scales linearly with N at fixed ℓ, and the attention footprint per instance is $\ell ^ { 2 }$ rather than $N ^ { 2 }$

## B EXPERIMENT DETAILS

## B.1 DATASET STATISTICS

All six datasets are public benchmarks whose statistics are summarized in Table 1 of the main text. The traffic datasets span 170 to 883 nodes at 5-minute sampling. The air quality dataset PM2.5 has 12 stations at hourly sampling.

PEMSD3-Stream protocol. Per-year statistics are listed in Table 4. We follow the PEMSD3- Stream setting of STOP (Ma et al., 2025b): PeMS District 3 flow over 2011–2017, one slice per year, and each year an independent graph whose station count grows from 655 to 871 as new sensors join. Training and validation use 2011 alone with the window indices released by STOP, the validation split being disjoint from training and used for early stopping, so no window of a later year is seen before testing. Testing uses the held-out windows of a year: 2011 gives the in-domain reference, while 2012–2017 are evaluated zero-shot with the 2011 checkpoint, without retraining. As in STOP, raw flow of zero marks a missing station and is excluded from the loss and the metrics, so all reported numbers are computed on observed positions.

Table 4: Per-year statistics of PEMSD3-Stream.
<table><tr><td>Year</td><td>Nodes</td><td>Time Span</td><td>Interval</td></tr><tr><td>2011</td><td>655</td><td>07/10/2011-08/31/2011</td><td>5 min</td></tr><tr><td>2012</td><td>715</td><td>07/10/2012–08/31/2012</td><td>5 min</td></tr><tr><td>2013</td><td>786</td><td>07/10/2013–08/31/2013</td><td>5 min</td></tr><tr><td>2014</td><td>822</td><td>07/10/2014-08/31/2014</td><td>5 min</td></tr><tr><td>2015</td><td>834</td><td>07/10/2015 – 08/31/2015</td><td>5 min</td></tr><tr><td>2016</td><td>850</td><td>07/10/2016–08/31/2016</td><td>5 min</td></tr><tr><td>2017</td><td>871</td><td>07/10/2017–08/31/2017</td><td>5 min</td></tr></table>

## B.2 HYPERPARAMETERS

Table 5: Default hyperparameters for the VaryST framework.
<table><tr><td>Hyperparameter</td><td>Description</td><td>Value or Choices</td></tr><tr><td>sample_len</td><td>Input history length T</td><td>2016 (168 on PM2.5)</td></tr><tr><td>output_len</td><td>Prediction horizon Q</td><td>12</td></tr><tr><td>unit_len</td><td>Patch length  $L _ { p }$ </td><td>36 (12 on PM2.5)</td></tr><tr><td>train_ratio / val_ratio</td><td>Chronological train/validation split</td><td>0.6 / 0.2</td></tr><tr><td>token_dim</td><td>Temporal Patch Token dimension d</td><td>96</td></tr><tr><td>emb_dim</td><td>backbone dim D</td><td>96</td></tr><tr><td>n_head</td><td>backbone attention heads</td><td>4</td></tr><tr><td>backbone_layers</td><td>Number of backbone layers</td><td>3</td></tr><tr><td>spatial_pe</td><td>Structural encodings</td><td>lap, hop-rope</td></tr><tr><td>node_pe_k</td><td>Laplacian PE eigenvectors</td><td>16</td></tr><tr><td>n_aux</td><td>Number of support nodes retrieved per target K</td><td>64</td></tr><tr><td>use_time_emb</td><td>Time-of-day / day-of-week embeddings</td><td>True</td></tr><tr><td>use_global_token</td><td>Global mean-pooled token</td><td>True</td></tr><tr><td>recent_len</td><td>Recent steps in the prompt token</td><td>12</td></tr><tr><td>norm_mode</td><td>Input normalization</td><td>RevIN</td></tr><tr><td>loss_type</td><td>Training loss</td><td>MAE</td></tr><tr><td>optimizer</td><td>Optimizer</td><td>Muon + AdamW</td></tr><tr><td>lr</td><td>Learning rate</td><td>0.001</td></tr><tr><td>weight_decay</td><td>Weight decay</td><td>0.0</td></tr><tr><td>dropout</td><td>Dropout</td><td>0.05</td></tr><tr><td>batch_size</td><td>The batch size for training</td><td>128</td></tr><tr><td>lr_patience</td><td>Learning-rate decay patience</td><td>5</td></tr><tr><td>patience</td><td></td><td>11</td></tr><tr><td></td><td>Early-stopping patience</td><td></td></tr><tr><td>epoch</td><td>Maximum training epochs</td><td>500</td></tr></table>

Table 5 lists the default configuration shared by all PEMS experiments. On PM2.5, the history length and the patch length, are changed to 168 and 12 respectively, as marked in the table. The sensitivity of context length, patch length, and the number of retrieved support nodes is studied in Section C.5.

## B.3 METRICS

Throughout this paper, we use the following metrics to evaluate performance:

MAE (Mean Absolute Error): Represents the average absolute difference between the predicted and actual values, providing an interpretable measure of average error magnitude.

$$
\mathrm { M A E } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } | y _ { i } - \hat { y } _ { i } | ,\tag{15}
$$

where $y _ { i }$ and $\hat { y } _ { i }$ denote the ground truth and predicted values, respectively, and n is the number of evaluated data points.

RMSE (Root Mean Squared Error): The square root of the mean squared difference between the predicted and actual values, expressed in the same units as the original data, and more sensitive to large deviations than MAE.

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( y _ { i } - \hat { y } _ { i } ) ^ { 2 } } .\tag{16}
$$

MAPE (Mean Absolute Percentage Error): Expresses errors as a percentage of the actual values, offering a scale-independent metric that facilitates comparisons across datasets.

$$
\mathrm { M A P E } = \frac { 1 0 0 } { n } \sum _ { i = 1 } ^ { n } \left| \frac { y _ { i } - \hat { y } _ { i } } { y _ { i } } \right| .\tag{17}
$$

All three metrics are computed only at observed positions, and reported values are averaged over all 12 prediction horizons; MAPE is expressed in percent. For all metrics, lower values indicate better performance.

## B.4 IMPLEMENTATION DETAILS

## B.4.1 HARDWARE AND ENVIRONMENT

All experiments run on a single NVIDIA RTX A4000 GPU under PyTorch 2.11. With early stopping, training completes in 3.5 hours on PEMS03, 4.2 hours on PEMS04, 4.4 hours on PEMS07, 3.8 hours on PEMS08, 2.7 hours on the PEMSD3-Stream train year, and 44 minutes on PM2.5, where the wall-clock covers the full training loop including validation and test passes. Peak training memory stays within 0.9–2.9 GB on the traffic benchmarks and below 0.1 GB on PM2.5.

## B.4.2 DATA SPLITTING AND LEAKAGE PREVENTION

Every dataset is split chronologically into training, validation, and test portions in a 6:2:2 ratio, with no shuffling across time. Zero-shot evaluation applies a model trained fully on the source dataset directly to the target dataset, without updating target parameters. Few-shot transfer fine-tunes on 1%, 5%, or 10% of the target training data, matching Table 9. The fraction is taken over training windows, each retaining the full 2016-step input that ends exactly where its 12-step target begins, rather than over the raw time span, which for our splits is shorter than one window and would leave nothing to train on; the retained windows are evenly spaced across the training portion, so a 1% run performs roughly 1% of the parameter updates of a full run. The same 2016-step input length also fixes the evaluation span: because a window’s target can only start where its input ends, the validation and test boundaries are shifted back by 2016 steps, so that the targets of the test windows tile the entire test portion from its first step to its last and every test step is scored. A model that under- or over-fits the windows it was fine-tuned on is therefore judged on the whole test span, never on a span as narrow as its few-shot subset. VaryST carries no node identity: representations are derived online from current observations, and its structural priors are relative relations only, namely hop distances and Laplacian eigenvectors. No entity information from a training graph therefore survives transfer. In the missing-observation experiments the input mask coincides with the loss mask, so the loss is computed only at observed positions; see the next subsubsection.

## B.4.3 BASELINE ADAPTATION AND MISSING-INPUT PROTOCOL

Dagger†-marked baselines were adapted by us to the same long-term input as VaryST, namely 2016 steps on the traffic benchmarks, while the PM2.5 comparison uses the 168-step input as stated in its table footnote. These adaptations change only the input interface, with no architectural mod ification. PatchSTG additionally requires sensor latitude and longitude metadata to construct its geospatial partitioning, which the PEMS benchmarks do not provide; we therefore synthesize virtual coordinates for each dataset so that PatchSTG operates with its original geo-partitioning intact. Under missing observations the baselines do not consume observation masks, so their inputs are pre-filled by linear interpolation. VaryST instead consumes the mask natively in its encoder, so its representations are conditioned on which observations are available.

## C FULL EXPERIMENT RESULTS

## C.1 FULL TEMPORAL GENERALIZATION RESULTS

Models are trained and validated on 2011 alone and evaluated zero-shot on each year from 2012 to 2017 as new sensors join the network; metrics are computed only at observed positions and averaged over all 12 prediction horizons. The six-year averages are summarized in Table 6 and the per-year breakdown in Table 7. VaryST attains the best MAE, RMSE, and MAPE in every year, reducing the six-year average MAE from 13.01 to 10.49. VaryST improves over STOP, the strongest baseline, by 19.4%. The gap follows from observation-derived node representations recomputed from current observations, and from retrieval that reselects supporting evidence as temporal patterns drift. The per-year curves appear in Figure 3a of the main text.

Table 6: Average Zero-shot performance on PEMSD3-Stream.
<table><tr><td>Model</td><td>MAE</td><td>RMSE</td><td>MAPE</td></tr><tr><td>STOP</td><td>13.01</td><td>22.44</td><td>16.91%</td></tr><tr><td>STD-PLM</td><td>14.40</td><td>25.06</td><td>18.55%</td></tr><tr><td>VaryST</td><td>10.49</td><td>18.22</td><td>14.05%</td></tr></table>

Table 7: Year-by-year performance on PEMSD3-Stream. 2011: In-Domain Reference
<table><tr><td rowspan="2">Year</td><td colspan="3">VaryST</td><td colspan="3">STOP</td><td colspan="3">STD-PLM</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE MAE</td><td></td><td></td><td></td><td></td><td></td><td>RMSE MAPE MAE RMSE MAPE</td></tr><tr><td>2011</td><td>10.64</td><td>16.47</td><td>13.84</td><td>12.34</td><td>18.96</td><td>16.12</td><td>11.96</td><td>18.28</td><td>16.21</td></tr><tr><td>2012</td><td>9.97</td><td>16.00</td><td>13.65</td><td>11.86</td><td>18.87</td><td>16.16</td><td>13.54</td><td>22.57</td><td>17.77</td></tr><tr><td>2013</td><td>10.01</td><td>16.68</td><td>13.60</td><td>12.27</td><td>20.05</td><td>16.26</td><td>14.24</td><td>24.16</td><td>18.23</td></tr><tr><td>2014</td><td>10.61</td><td>17.39</td><td>14.80</td><td>13.24</td><td>22.07</td><td>17.96</td><td>14.62</td><td>24.73</td><td>19.78</td></tr><tr><td>2015</td><td>10.55</td><td>17.80</td><td>14.53</td><td>12.99</td><td>22.40</td><td>17.22</td><td>14.32</td><td>24.97</td><td>18.87</td></tr><tr><td>2016</td><td>10.30</td><td>21.52</td><td>13.23</td><td>13.19</td><td>26.20</td><td>16.14</td><td>13.89</td><td>26.77</td><td>17.46</td></tr><tr><td></td><td>2017 11.51</td><td>19.93</td><td>14.50</td><td>14.51</td><td>25.05</td><td>17.70</td><td>15.81</td><td>27.15</td><td>19.18</td></tr></table>

## C.2 FULL SPATIAL GENERALIZATION RESULTS

Table 8 presents 12-step zero-shot transfer performance across all twelve pairwise source–target combinations among PEMS03, PEMS04, PEMS07, and PEMS08, evaluated in MAE, RMSE, and percentage MAPE, with the diagonal serving as the in-domain baseline. In parallel, Table 9 reports few-shot fine-tuning results using 1%, 5%, and 10% of the target training data. VaryST achieves the lowest MAE, RMSE, and MAPE across all twelve zero-shot transfer directions. Notably, its maximum MAE increase over the corresponding in-domain baseline is confined to just 9.9% in the PEMS03→PEMS07 transfer. In fact, transferring from PEMS07 to PEMS03 slightly surpasses the in-domain benchmark, achieving an MAE of 13.78 compared to 13.95. This counter-intuitive advantage aligns with our design: node representations and supports are directly constructed from observations available on the target network rather than rigid source graph structures. Under fewshot adaptation, fine-tuning VaryST on merely 1% of target data already outperforms STD-PLM adapted with 10% data across all twelve directions, while scaling the target budget to 10% yields a modest additional MAE reduction of at most 3.7%. Figure 3b compares zero-shot and in-domain MAE on PEMS04, with comprehensive analyses provided in Section 5.4.

Table 8: Zero-shot cross-dataset transfer (12-step prediction). Rows: source dataset used for full training; columns: target dataset evaluated on. Diagonal cells are the in-domain (IID) baselines. Each cell: MAE / RMSE / MAPE (%).
<table><tr><td rowspan="2">Model</td><td rowspan="2">Source</td><td colspan="4">Target</td></tr><tr><td>PEMS03</td><td>PEMS04</td><td>PEMS07</td><td>PEMS08</td></tr><tr><td rowspan="4">VaryST</td><td>PEMS03</td><td>13.95 / 22.73 / 13.59</td><td>18.51 / 30.40 / 12.84</td><td>20.02 / 32.72 / 8.88</td><td>13.81 / 22.70 / 9.31</td></tr><tr><td>PEMS04</td><td>14.03 / 22.77 / 13.44</td><td>17.56 / 29.35 / 11.81</td><td>19.91 / 32.64 / 8.48</td><td>13.97 / 22.92 / 9.07</td></tr><tr><td>PEMS07</td><td>13.78 / 22.32 / 13.24</td><td>18.15 / 29.98 / 12.54</td><td>18.21 / 31.05 / 7.64</td><td>13.46 / 22.44 / 8.86</td></tr><tr><td>PEMS08</td><td>14.88 / 24.45 / 14.61</td><td>18.80 / 30.90 / 13.19</td><td>19.53 / 32.65 / 8.54</td><td>12.89 / 21.99 / 8.50</td></tr><tr><td rowspan="4">STD-PLM</td><td>PEMS03</td><td>14.52 / 24.69 / 14.87</td><td>30.90 / 49.01 / 20.88</td><td>34.79 / 52.85 / 20.27</td><td>25.83 / 39.86 / 25.03</td></tr><tr><td>PEMS04</td><td>23.90 / 38.93 / 27.38</td><td>18.15 / 30.31 / 11.77</td><td>35.19 / 53.08 / 23.32</td><td>26.64 / 40.38 / 28.48</td></tr><tr><td>PEMS07</td><td>24.98 / 38.37 / 42.77</td><td>29.44 / 44.95 / 24.05</td><td>19.26 / 32.72 / 8.09</td><td>22.86 / 35.28 / 15.09</td></tr><tr><td>PEMS08</td><td>22.48 / 36.65 / 36.09</td><td>28.89 / 44.59 / 22.63</td><td>31.25 / 48.41 / 14.18</td><td>13.30 / 23.41 / 8.82</td></tr></table>

Table 9: Few-shot cross-dataset transfer (12-step prediction) with 1%, 5% and 10% of the target training data. Each cell: MAE / RMSE / MAPE (%).
<table><tr><td rowspan="2">Model</td><td rowspan="2">Target</td><td colspan="3">1%</td><td colspan="3">5%</td><td colspan="3">10%</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td colspan="10">Source: PEMSO3</td></tr><tr><td rowspan="2">VaryST</td><td>PEMS04</td><td>18.39</td><td>30.39</td><td>12.60</td><td>18.04</td><td>29.93</td><td>12.33</td><td>17.91</td><td>29.78</td><td>12.19</td></tr><tr><td>PEMS07 PEMS08</td><td>19.16</td><td>32.03 22.54</td><td>8.16 8.91</td><td>18.60 13.26</td><td>31.48</td><td>7.85 8.75</td><td>18.63</td><td>31.50</td><td>7.85</td></tr><tr><td rowspan="3">STD-PLM</td><td></td><td>13.50</td><td></td><td></td><td></td><td>22.31</td><td></td><td>13.17</td><td>22.21</td><td>8.69</td></tr><tr><td>PEMS04</td><td>46.75</td><td>74.69</td><td>22.00</td><td>24.45</td><td>38.70</td><td>16.19</td><td>23.38</td><td>37.08</td><td>15.54</td></tr><tr><td>PEMS07 PEMS08</td><td>28.35 24.00</td><td>43.93 36.80</td><td>12.53 16.20</td><td>25.53 18.68</td><td>40.06 29.55</td><td>11.58 14.00</td><td>24.34 18.16</td><td>38.77 29.02</td><td>10.55 12.17</td></tr><tr><td colspan="10">Source: PEMS04</td></tr><tr><td colspan="2"></td><td>14.08</td><td>22.91</td><td>13.49</td><td>14.00</td><td>22.77</td><td>13.35</td><td>13.98</td><td>22.73</td><td>13.45</td></tr><tr><td rowspan="3">VaryST</td><td>PEMS03 PEMS07</td><td>19.27</td><td>32.14</td><td>8.11</td><td>18.74</td><td>31.62</td><td>7.86</td><td>18.65</td><td>31.49</td><td>7.82</td></tr><tr><td>PEMS08</td><td>13.64</td><td>22.69</td><td>8.77</td><td>13.34</td><td>22.36</td><td>8.65</td><td>13.20</td><td>22.21</td><td>8.59</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">STD-PLM</td><td>PEMS03</td><td>20.30</td><td>32.79</td><td>21.97</td><td>18.35</td><td>30.37</td><td>18.00</td><td>17.01</td><td>27.84</td><td>16.67</td></tr><tr><td>PEMS07</td><td>27.60</td><td>42.84</td><td>12.58</td><td>25.17</td><td>39.70</td><td>11.35</td><td>24.55</td><td>39.07</td><td>10.72</td></tr><tr><td>PEMS08</td><td>21.99</td><td>33.25</td><td>15.84</td><td>18.47</td><td>29.40</td><td>12.70</td><td>18.05</td><td>28.96</td><td>11.80</td></tr><tr><td colspan="10">Source: PEMS07</td></tr><tr><td rowspan="3">VaryST</td><td>PEMS03</td><td>13.91</td><td>22.56</td><td>13.31</td><td>13.89</td><td>22.54</td><td>13.24</td><td>13.88</td><td>22.55</td><td>13.18</td></tr><tr><td>PEMS04</td><td>18.20</td><td>30.12</td><td>12.50</td><td>17.88</td><td>29.75</td><td>12.15</td><td>17.87</td><td>29.72</td><td>12.18</td></tr><tr><td>PEMS08</td><td>13.27</td><td>22.31</td><td>8.77</td><td>13.09</td><td>22.18</td><td>8.60</td><td>13.02</td><td>22.09</td><td>8.62</td></tr><tr><td rowspan="3">STD-PLM</td><td>PEMS03</td><td>20.41</td><td>33.44</td><td>26.46</td><td>18.35</td><td>30.69</td><td>18.23</td><td>16.86</td><td>27.78</td><td>16.38</td></tr><tr><td>PEMS04</td><td>32.67</td><td>53.73</td><td>20.74</td><td>24.07</td><td>38.02</td><td>16.76</td><td>23.19</td><td>36.79</td><td>15.88</td></tr><tr><td>PEMS08</td><td>21.78</td><td>33.36</td><td>18.84</td><td>18.18</td><td>29.20</td><td>11.53</td><td>18.08</td><td>29.17</td><td>11.37</td></tr><tr><td colspan="10">Source: PEMS08</td></tr><tr><td rowspan="3">VaryST</td><td>PEMS03</td><td>14.78</td><td>24.57</td><td>14.41</td><td>14.56</td><td>24.10</td><td>14.21</td><td>14.57</td><td>23.95</td><td>14.16</td></tr><tr><td>PEMS04</td><td>18.66</td><td>30.98</td><td>12.84</td><td>18.12</td><td>30.07</td><td>12.46</td><td>17.97</td><td>29.87</td><td>12.27</td></tr><tr><td>PEMS07</td><td>19.01</td><td>32.15</td><td>8.08</td><td>18.70</td><td>31.83</td><td>7.89</td><td>18.51</td><td>31.53</td><td>7.81</td></tr><tr><td rowspan="3">STD-PLM</td><td>PEMS03</td><td></td><td></td><td>23.70</td><td>18.38</td><td></td><td></td><td>16.91</td><td>28.28</td><td>16.31</td></tr><tr><td>PEMS04</td><td>19.69 47.90</td><td>32.93 84.18</td><td>23.60</td><td>24.44</td><td>31.25 38.50</td><td>17.96 16.66</td><td>23.57</td><td>37.27</td><td>15.98</td></tr><tr><td>PEMS07</td><td>27.63</td><td>42.92</td><td>12.31</td><td>25.38</td><td>39.94</td><td>11.14</td><td>24.38</td><td>38.94</td><td>10.59</td></tr></table>

## C.3 FULL MISSING-OBSERVATION RESULTS

The evaluation covers forecasting on PEMS04 and PEMS08 under point missing (RM) and spatiotemporal block missing (CM) at missing rates of 10% to 70%, with all metrics averaged over the 12 horizons at observed positions. VaryST trains one model per rate with importance-guided observation perturbation (IGOP), which removes the history patches that draw the most attention during training, whereas baseline inputs are pre-filled by linear interpolation, following Appendix B.4.3. Table 10 reports same-rate training and testing.

Table 10: Overall prediction performance with missing observations on PEMS04 and PEMS08 under missing rates from 10% to 70%. The best and second-best results are in bold and underline, respectively.
<table><tr><td rowspan="2">Model</td><td colspan="3">10%</td><td colspan="3">30%</td><td colspan="3">50%</td><td colspan="3">70%</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE RMSE</td><td></td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td colspan="10">PEMS04 Point Missing (RM)</td><td></td><td></td><td></td></tr><tr><td>FaST†</td><td>18.67</td><td>30.29</td><td>12.89</td><td>18.74</td><td>30.37</td><td>13.01</td><td>18.87 30.50</td><td></td><td>13.05</td><td>19.17</td><td>30.78</td><td>13.14</td></tr><tr><td>STID†</td><td>18.34</td><td>29.92</td><td>13.09</td><td>18.38</td><td>29.98</td><td>12.84</td><td>18.53</td><td>30.20</td><td>12.99</td><td>18.75</td><td>30.40</td><td>13.41</td></tr><tr><td>BiST†</td><td>18.47</td><td>29.94</td><td>12.88</td><td>18.53</td><td>29.94</td><td>13.25</td><td>18.53</td><td>30.10</td><td>13.22</td><td>19.00</td><td>30.42</td><td>13.69</td></tr><tr><td>VaryST</td><td>17.60</td><td>29.34</td><td>11.90</td><td>17.40</td><td>29.20</td><td>11.74</td><td>17.47</td><td>29.28</td><td>11.80</td><td>17.70</td><td>29.39</td><td>11.93</td></tr><tr><td colspan="10">PEMS04 Spatio-Temporal Block Missing (CM)</td><td colspan="3"></td></tr><tr><td>FaST†</td><td>18.79</td><td>30.32</td><td>13.07</td><td>20.39</td><td>32.74</td><td>14.16</td><td>22.02</td><td>34.70</td><td>15.99</td><td>27.17</td><td>41.55</td><td>20.67</td></tr><tr><td>STID†</td><td>18.59</td><td>30.22</td><td>13.48</td><td>19.66</td><td>32.20</td><td>13.84</td><td>21.08</td><td>33.92</td><td>14.88</td><td>23.54</td><td>38.39</td><td>16.08</td></tr><tr><td>BiST†</td><td>18.58</td><td>30.13</td><td>13.51</td><td>19.40</td><td>31.39</td><td>13.80</td><td>20.70</td><td>33.06</td><td>14.87</td><td>23.09</td><td>36.81</td><td>17.41</td></tr><tr><td>VaryST</td><td>17.80</td><td>29.55</td><td>12.05</td><td>18.44</td><td>30.42</td><td>12.48</td><td>19.50</td><td>31.76</td><td>13.39</td><td>22.41</td><td>36.08</td><td>15.29</td></tr><tr><td colspan="10">PEMS08 Point Missing (RM)</td><td colspan="3"></td></tr><tr><td>FaST†</td><td>14.15</td><td>23.26</td><td>10.08</td><td>14.18</td><td>23.36</td><td>10.14</td><td>14.29</td><td>23.65</td><td>10.18</td><td>14.77</td><td>23.97</td><td>10.47</td></tr><tr><td>STID†</td><td>13.50</td><td>22.50</td><td>9.34</td><td>13.61</td><td>22.53</td><td>9.73</td><td>13.73</td><td>22.62</td><td>9.60</td><td>14.08</td><td>23.12</td><td>10.12</td></tr><tr><td>BiST†</td><td>13.99</td><td>22.74</td><td>10.75</td><td>13.92</td><td>22.77</td><td>10.59</td><td>14.02</td><td>22.94</td><td>10.33</td><td>14.27</td><td>23.24</td><td>10.53</td></tr><tr><td>VaryST</td><td>12.98</td><td>22.13</td><td>8.61</td><td>12.91</td><td>21.91</td><td>8.56</td><td>12.96</td><td>21.96</td><td>8.55</td><td>13.30</td><td>22.27</td><td>8.73</td></tr><tr><td colspan="11">PEMS08 Spatio-Temporal Block Missing (CM)</td></tr><tr><td>FaST†</td><td>14.42</td><td>23.75</td><td>10.40</td><td>15.40</td><td>25.00</td><td>11.00</td><td>17.48</td><td>27.69</td><td>12.93</td><td>22.08</td><td>33.78</td><td>17.09</td></tr><tr><td>STID† BiST†</td><td>13.74</td><td>22.80</td><td>9.60</td><td>14.73</td><td>23.83</td><td>10.35</td><td>16.12</td><td>25.91</td><td>12.12</td><td>17.96</td><td>28.94</td><td>12.71 14.22</td></tr><tr><td>VaryST</td><td>14.06</td><td>23.12 22.32</td><td>10.54 8.63</td><td>15.03</td><td>24.14</td><td>10.83 9.63</td><td>16.31 14.89</td><td>26.19 24.67</td><td>11.80 9.72</td><td>18.51 17.16</td><td>29.11 28.08</td><td>11.56</td></tr><tr><td></td><td>13.14</td><td></td><td></td><td>14.30</td><td>23.38</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup> Models adapted to long-context inputs (2016 steps) by us for fair comparison.

Under this protocol VaryST attains the lowest metric in all 48 reported cells. Scattered point missing barely affects it: between 10% and 70% RM the same-rate MAE stays within 0.4 on both datasets, because the encoder consumes the availability mask directly and re-derives node representations from the observations that remain. Block missing is the harder regime: as entire segments vanish, the error of every model grows with the rate, and VaryST remains first throughout.

Table 11: Generalization to unseen missing rates on PEMS04 and PEMS08. Models are trained at a 30% missing rate.
<table><tr><td rowspan="2">Model</td><td colspan="3">10%</td><td colspan="3">30%</td><td colspan="3">50%</td><td colspan="3">70%</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE MAPE (%)</td><td></td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td colspan="10">PEMS04 Point Missing (RM), source = 30%</td><td colspan="3"></td></tr><tr><td>FaST†</td><td>18.69</td><td>30.31</td><td>12.97</td><td>18.74</td><td>30.37</td><td>13.01</td><td>18.88 30.59</td><td>13.09</td><td></td><td>19.22</td><td>30.97</td><td>13.39</td></tr><tr><td>STID†</td><td>18.32</td><td>29.91</td><td>12.80</td><td>18.38</td><td>29.98</td><td>12.84</td><td>18.53</td><td>30.19</td><td>12.91</td><td>18.91</td><td>30.62</td><td>13.19</td></tr><tr><td>BiST†</td><td>18.49</td><td>29.90</td><td>13.22</td><td>18.53</td><td>29.94</td><td>13.25</td><td>18.66</td><td>30.12</td><td>13.32</td><td>18.98</td><td>30.12</td><td>13.57</td></tr><tr><td>VaryST</td><td>17.60</td><td>29.35</td><td>12.07</td><td>17.40</td><td>29.20</td><td>11.74</td><td>17.55</td><td>29.38</td><td>11.73</td><td>18.28</td><td>30.12</td><td>12.34</td></tr><tr><td colspan="10">PEMS04 Spatio-Temporal Block Missing (CM), source = 30%</td><td colspan="3"></td></tr><tr><td>FaST†</td><td>19.88</td><td>32.16</td><td>13.63</td><td>20.39</td><td>32.74</td><td>14.16</td><td>23.37</td><td>36.43</td><td>18.95</td><td>38.20</td><td>57.56</td><td>39.79</td></tr><tr><td>STID†</td><td>19.07</td><td>31.42</td><td>13.21</td><td>19.66</td><td>32.20</td><td>13.84</td><td>22.27</td><td>35.67</td><td>17.73</td><td>34.73</td><td>52.86</td><td>35.44</td></tr><tr><td>BiST†</td><td>18.98</td><td>30.86</td><td>13.42</td><td>19.40</td><td>31.39</td><td>13.80</td><td>21.73</td><td>34.31</td><td>17.31</td><td>34.83</td><td>52.23</td><td>36.97</td></tr><tr><td>VaryST</td><td>18.02</td><td>29.91</td><td>12.35</td><td>18.44</td><td>30.42</td><td>12.48</td><td>20.61</td><td>34.15</td><td>14.31</td><td>34.71</td><td>58.39</td><td>32.20</td></tr><tr><td colspan="10">PEMS08 Point Missing (RM), source = 30%</td><td colspan="3"></td></tr><tr><td>FaST†</td><td>14.11</td><td>23.28</td><td>10.09</td><td>14.18</td><td>23.36</td><td>10.14</td><td>14.29</td><td>23.50</td><td>10.22</td><td>14.68</td><td>24.00</td><td>10.43</td></tr><tr><td>STID†</td><td>13.55</td><td>22.47</td><td>9.70</td><td>13.61</td><td>22.53</td><td>9.73</td><td>13.76</td><td>22.71</td><td>9.82</td><td>14.18</td><td>23.26</td><td>10.02</td></tr><tr><td>BiST†</td><td>13.88</td><td>22.74</td><td>10.58</td><td>13.92</td><td>22.77</td><td>10.59</td><td>14.03</td><td>22.90</td><td>10.68</td><td>14.42</td><td>23.44</td><td>10.88</td></tr><tr><td>VaryST</td><td>13.05</td><td>22.05</td><td>8.73</td><td>12.91</td><td>21.91</td><td>8.56</td><td>13.07</td><td>22.05</td><td>8.62</td><td>13.83</td><td>22.87</td><td>9.02</td></tr><tr><td colspan="11">PEMS08 Spatio-Temporal Block Missing (CM), source = 30%</td></tr><tr><td>FaST†</td><td>14.73</td><td>24.16</td><td>10.47</td><td>15.40</td><td>25.00</td><td>11.00</td><td>18.57</td><td>29.78</td><td>13.30</td><td>32.06</td><td>48.39</td><td>26.39</td></tr><tr><td>STID† BiST†</td><td>14.05</td><td>23.00</td><td>9.82</td><td>14.73</td><td>23.83</td><td>10.35</td><td>17.31</td><td>28.23</td><td>12.18 12.53</td><td>28.72</td><td>46.28</td><td>21.98 21.04</td></tr><tr><td>VaryST</td><td>14.44 13.84</td><td>23.40 22.83</td><td>10.34 9.21</td><td>15.03 14.30</td><td>24.14 23.38</td><td>10.83 9.63</td><td>17.30 16.42</td><td>27.90 27.82</td><td>11.18</td><td>27.43 27.76</td><td>43.88 48.80</td><td>21.84</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

<sup>†</sup> Models adapted to long-context inputs (2016 steps) by us for fair comparison.

Table 11 reports cross-rate transfer performance, where models trained at a 30% missing rate are directly evaluated across rates from 10% to 70% without retraining or altering the missing pattern. Under RM, VaryST ranks first across all 24 evaluation cells. Even at the extreme 70% missing rate, its MAE remains 0.63 and 0.35 lower than the runner-up on PEMS04 and PEMS08, respectively. Although its degradation relative to same-rate training reaches +0.58 on PEMS04 and +0.53 on PEMS08, surpassing the baselines’ maximum gap of 0.16, scattered observations still suffice to reconstruct its representations. Under CM, VaryST maintains its top ranking up to a 50% missing rate; at 70%, where contiguous evidence largely vanishes, it still achieves the best MAE and MAPE on PEMS04, outperforming the runner-up by 3.2 points in MAPE. On PEMS08 it is second to BiST on MAE, 27.76 against 27.43, and it is last on RMSE on both datasets.

Table 12: Cross-pattern generalization at 30% missing rate on PEMS04 and PEMS08.
<table><tr><td rowspan="3">Model</td><td colspan="6">PEMS04</td><td colspan="6">PEMS08</td></tr><tr><td colspan="3">RM → CM</td><td colspan="3">CM → RM</td><td colspan="3">RM → CM</td><td colspan="3">CM → RM</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td><td>MAE</td><td>RMSE</td><td>MAPE (%)</td></tr><tr><td>FaST†</td><td>20.32</td><td>32.40</td><td>15.19</td><td>19.89</td><td>32.16</td><td>13.62</td><td>16.00</td><td>26.00</td><td>11.87</td><td>14.71</td><td>24.13</td><td>10.46</td></tr><tr><td>STID†</td><td>19.96</td><td>32.14</td><td>14.71</td><td>19.06</td><td>31.41</td><td>13.13</td><td>15.33</td><td>24.85</td><td>11.28</td><td>14.02</td><td>22.95</td><td>9.80</td></tr><tr><td>BiST†</td><td>20.02</td><td>31.87</td><td>15.06</td><td>18.98</td><td>30.86</td><td>13.38</td><td>15.63</td><td>25.06</td><td>12.33</td><td>14.42</td><td>23.37</td><td>10.31</td></tr><tr><td>VaryST 19.51</td><td></td><td>31.94</td><td>13.96</td><td>18.40</td><td>30.23</td><td>12.75</td><td>15.00</td><td>24.93</td><td>10.43</td><td>14.14</td><td>23.15</td><td>9.36</td></tr><tr><td colspan="9">In-domain reference at 30% (train and test on the target pattern)</td><td></td><td></td><td></td><td></td></tr><tr><td>STID†</td><td>19.66</td><td>32.20</td><td>13.84</td><td>18.38</td><td>29.98</td><td>12.84</td><td>14.73</td><td>23.83</td><td>10.35</td><td>13.61</td><td>22.53</td><td>9.73</td></tr><tr><td>BiST†</td><td>19.40</td><td>31.39</td><td>13.80</td><td>18.53</td><td>29.94</td><td>13.25</td><td>15.03</td><td>24.14</td><td>10.83</td><td>13.92</td><td>22.77</td><td>10.59</td></tr></table>

<sup>†</sup> Models adapted to long-context inputs (2016 steps) by us for fair comparison.

Table 12 reports cross-pattern transfer at a 30% missing rate: a model trained under one pattern is evaluated under the other without further training, and the reference rows repeat the in-domain numbers at 30% from Table 10. VaryST attains the lowest MAE in three of the four directions and the lowest MAPE in all four, as its representations are re-derived from the observations that survive the switch. Its RMSE is second in both RM→CM directions by at most 0.08, and STID leads MAE and RMSE in the CM→RM direction on PEMS08 (14.14 vs. 14.02 and 23.15 vs. 22.95), consistent with STID modeling each node independently and being least sensitive to the pattern switch.

The rate and pattern sweeps are summarized in Figure 3c of the main text.

## C.4 FULL ABLATION RESULTS

The complete ablation in Table 13 covers 12-step prediction on PEMS03 and PEMS08, reporting MAE, RMSE, MAPE, and $\mathrm { M A P E _ { 1 0 } }$ , where $\mathbf { M A P E } _ { 1 0 }$ excludes ground-truth values below 10. Removing any component raises the MAE and RMSE on PEMS03. The raw MAPE deserves care. Readings below 10 carry little operational value in traffic applications, but their small denominators make relative errors fluctuate, so a few variants look marginally better under this metric. Once these low-value readings are excluded, the full model attains the best $\mathbf { M A P E } _ { 1 0 }$ on both datasets, and their apparent advantages prove to be small-value artifacts rather than genuine improvements. The hopaware structural attention causes the largest PEMS03 degradation at 0.47 MAE, and removing the global token costs 0.33 MAE on PEMS08, identifying structural modulation and global-state completion as the main sources of in-domain accuracy. The PEMS03 breakdown appears in Figure 4a of the main text.

Table 13: Ablation study of each component on PEMS03 and PEMS08 (12-step prediction).
<table><tr><td rowspan="2">Variant</td><td colspan="4">PEMS03</td><td colspan="4">PEMS08</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAPE10</td><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAPE10</td></tr><tr><td>VaryST (full)</td><td>13.95</td><td>22.73</td><td>13.59</td><td>11.00</td><td>12.89</td><td>21.99</td><td>8.50</td><td>7.29</td></tr><tr><td>w/o Laplacian PE</td><td>14.13</td><td>23.02</td><td>13.47</td><td>11.04</td><td>12.91</td><td>22.00</td><td>8.57</td><td>7.31</td></tr><tr><td>w/o Hop-RoPE</td><td>14.42</td><td>23.75</td><td>13.93</td><td>11.40</td><td>12.93</td><td>22.07</td><td>8.56</td><td>7.32</td></tr><tr><td>w/o structural encoding</td><td>14.10</td><td>23.05</td><td>13.55</td><td>11.11</td><td>12.92</td><td>22.06</td><td>8.54</td><td>7.30</td></tr><tr><td>w/o recent-12 token</td><td>14.10</td><td>23.00</td><td>13.35</td><td>10.99</td><td>12.86</td><td>21.97</td><td>8.47</td><td>7.29</td></tr><tr><td>w/o long context</td><td>15.95</td><td>25.42</td><td>15.73</td><td>13.04</td><td>15.02</td><td>25.51</td><td>9.81</td><td>8.45</td></tr><tr><td>w/o global token</td><td>14.06</td><td>22.97</td><td>13.47</td><td>11.04</td><td>13.22</td><td>22.23</td><td>8.67</td><td>7.47</td></tr><tr><td>Cosine retrieval</td><td>14.33</td><td>23.31</td><td>13.84</td><td>11.32</td><td>12.88</td><td>22.02</td><td>8.49</td><td>7.30</td></tr><tr><td>w/o tod/dow embedding</td><td>14.20</td><td>23.04</td><td>13.74</td><td>11.22</td><td>12.98</td><td>22.07</td><td>8.62</td><td>7.35</td></tr></table>

## C.5 FULL HYPERPARAMETER STUDY RESULTS

Table 14 presents the complete hyperparameter study. We evaluate 12-step prediction on PEMS03 and PEMS08 by sweeping each hyperparameter against the default configuration in Table 5: context length T across 36, 288, 864, 2016, and 4032 steps; patch length across 12, 36, and 96 steps; and the number of retrieved support nodes K across 16, 32, 64, and 128. Figure 4c illustrates the corresponding curves for PEMS03.

Context length exerts the strongest influence, characterized by a sharp initial error reduction that rapidly plateaus. Increasing T from 36 to 288 steps substantially lowers MAE by 1.04 on PEMS03 and 1.18 on PEMS08. Beyond 288 steps, the error declines gradually toward the one-week default $( T = 2 0 1 6 )$ , reaching MAEs of 13.95 and 12.89, respectively. Extending the context to two weeks (T = 4032) yields no further gain on PEMS08 and degrades MAE by 0.45 on PEMS03, confirming the default as the optimal operating point. This behavior directly reflects our representation design: compressing a 2016-step history into a single observation-derived token per node enriches the evidence underlying each token without expanding the (K+3)-token decoder sequence, up to the point of diminishing informational returns.

Spatial retrieval exhibits a similar economy. Even with only 16 support nodes, the MAE remains within 0.42 of the optimum on both benchmarks, whereas scaling K to 128 brings no consistent benefit. The RBF retrieval effectively concentrates relevant spatial signals into a small subset of nodes, while formatting them as prompt tokens preserves linear decoding complexity in K rather than quadratic scaling in total node count. Finally, patch length is the least sensitive hyperparameter; varying it across 12, 36, and 96 steps shifts MAE by at most 0.35 on both datasets, demonstrating that the representation is largely agnostic to the exact historical segmentation.

Table 14: Hyperparameter study on PEMS03 and PEMS08. The best and second-best results are in bold and underline, respectively.
<table><tr><td rowspan="2">Setting</td><td colspan="3">PEMS03</td><td colspan="3">PEMS08</td></tr><tr><td>MAE</td><td>RMSE</td><td>MAPE</td><td>MAE</td><td>RMSE</td><td>MAPE</td></tr><tr><td colspan="7">Input context length</td></tr><tr><td>36</td><td>15.30</td><td>24.37</td><td>14.88</td><td>14.78</td><td>24.91</td><td>9.67</td></tr><tr><td>288</td><td>14.26</td><td>23.08</td><td>13.87</td><td>13.60</td><td>22.79</td><td>9.02</td></tr><tr><td>864</td><td>14.31</td><td>23.37</td><td>13.65</td><td>13.40</td><td>22.53</td><td>8.90</td></tr><tr><td>2016 (default)</td><td>13.95</td><td>22.73</td><td>13.59</td><td>12.89</td><td>21.99</td><td>8.50</td></tr><tr><td>4032</td><td>14.40</td><td>23.74</td><td>13.87</td><td>12.89</td><td>21.97</td><td>8.51</td></tr><tr><td colspan="7">Patch length</td></tr><tr><td>12</td><td>14.03</td><td>22.90</td><td>13.47</td><td>13.23</td><td>22.25</td><td>8.80</td></tr><tr><td>36 (default)</td><td>13.95</td><td>22.73</td><td>13.59</td><td>12.89</td><td>21.99</td><td>8.50</td></tr><tr><td>96</td><td>14.26</td><td>23.25</td><td>13.95</td><td>13.13</td><td>22.24</td><td>8.75</td></tr><tr><td colspan="7">Number of auxiliary nodes</td></tr><tr><td>16</td><td>14.37</td><td>23.33</td><td>13.94</td><td>12.89</td><td>22.06</td><td>8.49</td></tr><tr><td>32</td><td>14.11</td><td>23.02</td><td>13.64</td><td>12.93</td><td>22.02</td><td>8.54</td></tr><tr><td>64 (default)</td><td>13.95</td><td>22.73</td><td>13.59</td><td>12.89</td><td>21.99</td><td>8.50</td></tr><tr><td>128</td><td>14.17</td><td>23.14</td><td>13.72</td><td>12.88</td><td>21.92</td><td>8.48</td></tr></table>