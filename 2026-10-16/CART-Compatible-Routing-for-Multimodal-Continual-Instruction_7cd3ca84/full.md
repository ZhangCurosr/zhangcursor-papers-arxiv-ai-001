# CART: Compatible Routing for Multimodal Continual Instruction Tuning

Ling-Xiang Zhang, Tao Hu, Da-Wei Zhou<sup>†</sup>

School of Artificial Intelligence, Nanjing University State Key Laboratory of Novel Software Technology, Nanjing University <sup>†</sup> Corresponding author {zhanglx,hut,zhoudw}@lamda.nju.edu.cn

## Abstract

Multimodal Large Language Models (MLLMs) achieve strong performance through instruction tuning, yet real-world applications often require them to continually acquire new capabilities from sequential tasks, i.e., Multimodal Continual Instruction Tuning (MCIT). In typical MLLMs, a projector is employed to align visual representations with the language model. However, existing methods primarily protect the language model from forgetting while continuously updating the projector across tasks. Such asymmetric adaptation can cause substantial representation drift in the projector, increasing the mismatch between the projector and language model and consequently exacerbating catastrophic forgetting. To address this issue, we propose Compatible Routing (CART), which explicitly coordinates the adaptation of the projector and language model. Specifically, CART maintains task-specific experts for both modules and combines them through weighted mixtures. CART jointly infers their weights by considering both expert relevance and crossmodule compatibility, thereby promoting coordinated adaptation between the two modules. To further control model growth, we introduce relevance-guided expert consolidation, which merges experts exhibiting similar relevance profiles across samples within each module. Extensive experiments demonstrate that CART achieves state-of-the-art performance.

## 1 Introduction

Multimodal Large Language Models (MLLMs) (Liu et al., 2023, Dai et al., 2023, Wang et al., 2025, Bai et al., 2025, Team, 2026) integrate visual information with the reasoning capabilities of LLMs through cross-modal alignment and multimodal instruction tuning (Zhang et al., 2026), establishing strong vision-language understanding. However, real-world tasks and their data often arrive sequentially rather than all at once (Guo et al., 2025c). MLLMs must continually acquire new capabilities without forgetting previously learned ones, giving rise to Multimodal Continual Instruction Tuning (MCIT) (Chen et al., 2024), where catastrophic forgetting caused by cross-task parameter interference remains a central challenge (French, 1999).

![](images/db771c31219bd2d451462dbab1d9f327f91f3446172f484296cf50b1b2a1b130.jpg)  
Figure 1: Performance retention during sequential learning. We plot each task’s performance, normalized to its initial instruction-tuning peak. Even when the corresponding task-specific LoRA is retrieved, performance on previously learned tasks degrades substantially as the projector is continually updated on new tasks.

Most MLLMs use a projector to connect a vision encoder to a language model, mapping visual features into the language model’s input space (Liu et al., 2023). However, existing MCIT methods mainly mitigate forgetting by protecting languageside adaptations through parameter isolation (Chen et al., 2024), selective expansion (Guo et al., 2025a), or stabilized experts (Xie et al., 2026), while sharing and continually updating the projector. Such asymmetric adaptation may alter the visual representations passed to the language model, causing preserved language-side adaptations to process unfamiliar inputs and potentially undermining their ability to retain prior capabilities.

To examine whether language-side protection is sufficient in practice, we study LLaVA-1.5-7B (Liu et al., 2024) on UCIT (Guo et al., 2025a), with the vision encoder and pre-trained language-model backbone frozen. Figure 1 compares two strategies along the same task sequence, both with a shared projector continually updated across tasks: a single LoRA sequentially trained throughout the task stream ( ), and a task-specific LoRA trained for each task as it arrives and retrieved at inference using oracle task identities ( ). Despite retrieving the corresponding task-specific LoRA, performance on earlier tasks still declines as the shared projector evolves, showing that languageside protection alone is insufficient to retain previously acquired multimodal capabilities.

![](images/16a51d2c653c2d7cc6456ece58f5e06eb47b3da9cc427f77843ec2e8284be6f1.jpg)  
Figure 2: Updating the projector shows a trade-off between learning new tasks and retaining old ones. Although updating the projector improves performance ( ), learning later tasks can reduce these gains and lower performance on earlier tasks below the frozenprojector baseline ( ). Retraining the language model with the final projector fixed largely restores performance ( ), suggesting that the mismatch between the projector and language model is indeed an important cause of forgetting.

Figure 1 shows that updating the shared projector can still cause forgetting even when task-specific language-side adaptations are preserved, making projector freezing a natural alternative. To investigate whether freezing the projector can mitigate forgetting while adapting to new tasks, Figure 2 compares two settings that either continually adapt the shared projector or keep it frozen. Both settings follow the same task sequence and train a separate language-side LoRA adapter for each task as it arrives. Each adapter is preserved after training and retrieved at inference using oracle task identities. In Figure 2, solid bars ( ) show performance differences from the frozen-projector baseline immediately after each task is learned, while hatched bars ( ) show the corresponding differences after the full task sequence, evaluated using the final shared projector and each task’s saved LoRA adapter. As shown in Figure 2, projector adaptation improves initial performance across the evaluated tasks, but as learning progresses, performance on several previously learned tasks eventually falls below the frozen-projector baseline. To further examine the source of this degradation, we fix the shared projector at the state reached after completing the entire task sequence and retrain a languageside LoRA adapter for each task using its original training data. As indicated by the arrows ( ), adapting the language-side parameters to this fixed projector recovers a substantial portion of the lost performance on several tasks. These results support two conclusions. 1) Adapting the projector improves downstream task acquisition compared with freezing it, but can also introduce forgetting. 2) Mismatch between the updated projector and historical language-side adaptations is a major source of this forgetting.

In this paper, we propose CompAtible RouTing (CART) for multimodal continual instruction tuning. Our key idea is to preserve task-specific adaptations for both the projector and language model and jointly determine how these adaptations are reused for each input. Specifically, CART instantiates task-specific experts for the projector and language model, and dynamically composes historical adaptations through weighted expert mixtures. The mixture weights are jointly estimated via coordinate ascent variational inference, considering both distribution-aware expert relevance and projector-language pairing preferences. To prevent unbounded model growth, we further introduce relevance-guided expert consolidation, which merges experts with similar relevance profiles within each module. Together, these mechanisms enable CART to mitigate forgetting while continually adapting to new multimodal tasks.

## 2 Related Work

Multimodal Large Language Models. Building on pre-trained visual representations (Radford et al., 2021), multimodal large language models (MLLMs) connect visual encoders with language models through cross-modal alignment modules (Liu et al., 2023; Dai et al., 2023). Following advances in language-model instruction tuning (Wei et al., 2021; Longpre et al., 2023; Zhang et al., 2026), multimodal instruction tuning enables MLLMs to follow diverse user requests (Liu et al., 2024; Dai et al., 2023). Their applications cover visual question answering (Goyal et al., 2017), compositional visual reasoning (Johnson et al.,

2017; Zerroug et al., 2022), and multimodal generation (Tong et al., 2025). Despite this progress, prevailing training pipelines largely rely on predefined datasets and task mixtures (Zhang et al., 2024; Chen et al., 2024). Adapting to new domains, instruction styles, and capabilities through further fine-tuning can interfere with previously learned knowledge (Zhai et al., 2023; Chen et al., 2024), motivating continual adaptation beyond the initial training stage.

Multimodal Continual Instruction Tuning aims to acquire new capabilities while retaining previously learned knowledge (Chen et al., 2024). Existing approaches mitigate forgetting through regularization, parameter-efficient adaptation, and prompting (Kirkpatrick et al., 2017; Wang et al., 2023; Yu et al., 2025; Zeng et al., 2025). Among expert-based methods, MoELoRA (Chen et al., 2024) routes inputs among low-rank experts, HiDe-LLaVA (Guo et al., 2025a) combines taskspecific expansion with task-general fusion, and SAME (Xie et al., 2026) stabilizes both routing decisions and expert updates. These studies highlight the importance of preserving and reusing task knowledge under sequential adaptation. Our CART further models distribution-aware relevance and cross-module compatibility to jointly coordinate projector and LLM experts, while bounding model growth through relevance-guided consolidation.

## 3 Preliminaries

Multimodal continual instruction tuning (MCIT) aims to extend an MLLM’s capabilities over sequential tasks while retaining previously acquired knowledge (Chen et al., 2024). We consider a task stream $\{ \mathcal { D } _ { t } \} _ { t = 1 } ^ { T }$ , where $\mathcal { D } _ { t } ~ = ~ \{ ( { \bf v } _ { i } , { \bf q } _ { i } , { \bf y } _ { i } ) \} _ { i = 1 } ^ { N _ { t } }$ contains triplets of images, instructions, and answers. Following the rehearsal-free setting, only $\mathcal { D } _ { t }$ is accessible during stage $t ,$ and historical training examples are not retained.

A typical MLLM comprises a vision encoder ϕ, a multimodal projector $\pi _ { \mathfrak { c } }$ and a language model. Given an image v and instruction q, the input embeddings are $\mathbf { z } ~ = ~ [ \pi ( \phi ( \mathbf { v } ) ) ; \psi ( \mathbf { q } ) ]$ , where $\psi$ denotes text embedding and $[ \cdot ; \cdot ]$ denotes sequence concatenation. The answer $\mathbf { y } = ( y _ { 1 } , \dots , y _ { L } )$ is generated autoregressively:

$$
p _ { \Theta } ( \mathbf { y } \mid \mathbf { v } , \mathbf { q } ) = \prod _ { j = 1 } ^ { L } p _ { \Theta } ( y _ { j } \mid \mathbf { z } , \mathbf { y } _ { < j } ) ,\tag{1}
$$

where Θ denotes the tunable parameters, with fixed

pre-trained parameters omitted for brevity. The current-task training loss is:

$$
\mathcal { L } _ { t } ( \boldsymbol { \Theta } ) = - \mathbb { E } _ { ( \mathbf { v } , \mathbf { q } , \mathbf { y } ) \sim \mathcal { D } _ { t } } \left[ \sum _ { j = 1 } ^ { L } \log p _ { \boldsymbol { \Theta } } ( y _ { j } \mid \mathbf { z } , \mathbf { y } _ { < j } ) \right] .\tag{2}
$$

Under continual learning, the model trains only on current-task data but must retain prior knowledge and perform task-agnostic inference.

Mixture of LoRA Experts LoRA (Hu et al., 2022) represents an update to a frozen weight matrix $\mathbf { W } _ { 0 } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ as ∆W = sBA, where $\textbf { A } \in$ $\mathbb { R } ^ { r \times d _ { \mathrm { i n } } }$ $\mathbf { B } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r } , r \ll \operatorname* { m i n } ( d _ { \mathrm { i n } } , d _ { \mathrm { o u t } } )$ , and $s = \alpha / r$ is a scaling factor. MoE-LoRA extends this formulation to a collection of low-rank experts with input-dependent routing. For a layer input $\mathbf { x } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ , the adapted output is:

$$
\mathbf { h } = \mathbf { W } _ { 0 } \mathbf { x } + s \sum _ { k = 1 } ^ { K } \omega _ { k } ( \mathbf { x } ) \mathbf { B } _ { k } \mathbf { A } _ { k } \mathbf { x } ,\tag{3}
$$

where $K$ is the number of available experts and the routing weights satisfy $\omega _ { k } ( { \bf x } ) \ge 0$ and $\begin{array} { r } { \sum _ { k = 1 } ^ { K } \omega _ { k } ( { \bf \bar { x } } ) = { \bf \bar { \Phi } } ^ { } 1 } \end{array}$ . A standard learned router computes $\omega ( \mathbf { x } ) = \mathrm { s o f t m a x } ( \mathbf { W } _ { g } \mathbf { x } )$ , with ${ \bf W } _ { g } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { K \times d _ { \mathrm { i n } } }$ . This formulation allows conditional combinations of lightweight adaptations while preserving the pre-trained base weights.

Discussions. Eq. (3) enables input-dependent reuse of historical adaptations through weighted expert mixtures. However, an evolving projector π can change the visual representations $\pi ( \phi ( \mathbf { v } ) )$ received by historical language-side experts. Preserving these experts alone therefore does not ensure their continued effectiveness. Effective reuse should account for both expert relevance and cross-module compatibility. Hence, an ideal model should maintain experts for both the projector and the language model while jointly determining their mixture weights.

## 4 Method

To mitigate forgetting caused by projector– language model mismatch, we introduce Compatible Routing (CART) for MCIT. Figure 3 illustrates the overall framework. Specifically, we maintain projector and LLM expert pools and jointly assign input-specific routing weights to their experts.

## 4.1 Paired Projector–LLM Experts

To prevent forgetting caused by mismatched visual and language adaptations, for each task $t ,$ we preserve the jointly trained projector and languageside adaptations as a Projector Expert $\mathbf { E } _ { \mathrm { p } } ^ { ( t ) }$ with full projector parameters and an LLM Expert $\mathbf { E } _ { \mid } ^ { ( t ) }$ parameterized by LoRA. After T tasks, this yields two expert pools $\pmb { \mathcal { E } } _ { \mathrm { p } } ~ = ~ \{ \mathbf { E } _ { \mathrm { p } } ^ { ( 1 ) } , \cdots , \mathbf { E } _ { \mathrm { p } } ^ { ( T ) } \}$ and $\pmb { \mathcal { E } } _ { 1 } = \{ \mathbf { E } _ { 1 } ^ { ( 1 ) } , \cdots , \mathbf { E } _ { 1 } ^ { ( T ) } \}$ . At inference time, to enable task-agnostic inference while avoiding crossmodule mismatch, we aim to identify relevant and compatible experts for each input. For an input x, we define the target joint routing distribution $\Pi _ { i j } ( \mathbf { x } ) = \Pi ( \mathbf { E } _ { \mathrm { p } } ^ { ( i ) } , \mathbf { E } _ { \mathrm { l } } ^ { ( j ) } | \textbf { x } )$ as:

![](images/b8cdfc52978a02d966f6990aa2b5486c01301891ee635a82ba0e0205cefe252a.jpg)  
Figure 3: Overview of CART. We jointly train and preserve expert pairs $( \mathbf { E } _ { \mathrm { p } } ^ { ( t ) } , \mathbf { E } _ { \mathrm { l } } ^ { ( t ) } )$ , retain their distribution statistics, and consolidate similar pairs. At inference, Gaussian likelihoods yield relevance scores $R _ { \mathfrak { p } } ( \mathbf { x } )$ and $R _ { \mathrm { l } } ( \mathbf { x } )$ Alternating compatibility-aware updates produce routing weights $q _ { \mathrm { p } } ( \mathbf { x } )$ and $q _ { \mathrm { l } } ( \mathbf { x } )$ , which combine projector outputs into a shared visual representation and mix LLM LoRA branches for inference without task identities.

$$
\Pi _ { i j } ( { \bf x } ) = \frac { R _ { \mathrm { p } } ^ { ( i ) } ( { \bf x } ) R _ { \mathrm { l } } ^ { ( j ) } ( { \bf x } ) C _ { i j } } { \sum _ { u , v } R _ { \mathrm { p } } ^ { ( u ) } ( { \bf x } ) R _ { \mathrm { l } } ^ { ( v ) } ( { \bf x } ) C _ { u v } } ,\tag{4}
$$

where $R _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } )$ and $R _ { 1 } ^ { ( j ) } ( { \bf x } )$ denote the relevance of projector expert $\mathbf { E } _ { \mathrm { p } } ^ { ( i ) }$ and LLM expert $\mathbf { E } _ { 1 } ^ { ( j ) }$ input x, respectively, and $C _ { i j }$ characterizes their cross-module compatibility.

## 4.2 Distribution-Aware Expert Relevance

To estimate input to expert relevance without task identities, we characterize each expert by the distribution of its training inputs in a frozen feature space. An input is considered more relevant to an expert when its representation better matches the expert’s input distribution.

Routing representations. For an input $\begin{array} { r l } { \mathbf { x } } & { { } = } \end{array}$ $( \mathbf { v } , \mathbf { q } )$ , we construct the projector and LLM routing representations as ${ \bf h } _ { \mathrm { p } } ( { \bf x } ) = \phi ( { \bf v } ) , { \bf h } _ { \mathrm { l } } ( { \bf x } )$ = $[ \phi ( \mathbf { v } ) , \psi ( \mathbf { q } ) ]$ Both feature extractors remain frozen throughout continual learning, ensuring that expert statistics collected at different stages are represented in a consistent feature space.

Expert input distributions. To convert the match between an input and an expert into a probabilistic relevance score, we model the expert-conditional routing representations as Gaussian distributions:

$$
\begin{array} { r l } & { \mathbf { h } _ { \mathrm { p } } ( \mathbf { x } ) \mid \mathbf { E } _ { \mathrm { p } } ^ { ( i ) } \sim \mathcal { N } \left( \pmb { \mu } _ { \mathrm { p } } ^ { ( i ) } , \pmb { \Sigma } _ { \mathrm { p } } \right) , } \\ & { \mathbf { h } _ { \mathrm { l } } ( \mathbf { x } ) \mid \mathbf { E } _ { \mathrm { l } } ^ { ( j ) } \sim \mathcal { N } \left( \pmb { \mu } _ { \mathrm { l } } ^ { ( j ) } , \pmb { \Sigma } _ { \mathrm { l } } \right) , } \end{array}\tag{5}
$$

where $\mu _ { \mathrm { p } } ^ { ( i ) }$ and $\mu _ { \mathrm { l } } ^ { ( j ) }$ denote expert-specific means that locate the centers of the expert input distributions, while $\Sigma _ { \mathfrak { p } }$ and $\Sigma _ { \mathrm { l } }$ are shared covariance matrices within the respective expert pools that capture feature variability and correlations.

For each side $m \in \{ { \mathfrak { p } } , 1 \}$ , we first compute the feature mean for task t as:

$$
\pmb { \mu } _ { m } ^ { ( t ) } = \frac { 1 } { N _ { t } } \sum _ { \mathbf { x } \in \mathcal { D } _ { t } } \mathbf { h } _ { m } ( \mathbf { x } ) .\tag{6}
$$

We retain $\mu _ { m } ^ { ( t ) }$ as the mean of the corresponding expert, and characterize the within-task feature variation by the scatter matrix:

$$
\mathbf { S } _ { m } ^ { ( t ) } = \sum _ { \mathbf { x } \in \mathcal { D } _ { t } } \Big ( \mathbf { h } _ { m } ( \mathbf { x } ) - \pmb { \mu } _ { m } ^ { ( t ) } \Big ) \left( \mathbf { h } _ { m } ( \mathbf { x } ) - \pmb { \mu } _ { m } ^ { ( t ) } \right) ^ { \top } .\tag{7}
$$

The within-task scatter is then aggregated across learned tasks to estimate the shared covariance: $\begin{array} { r } { \widehat { \pmb { \Sigma } } _ { m } = \sum _ { s = 1 } ^ { t } \mathbf { S } _ { m } ^ { ( s ) } / \sum _ { s = 1 } ^ { t } N _ { s } } \end{array}$ . We then regularize the covariance estimate for numerical stability $\pmb { \Sigma } _ { m } = \widehat { \pmb { \Sigma } } _ { m } + \epsilon \pmb { \mathrm { I } } .$ . As each new task arrives, we update the accumulated scatter and sample count using only its training inputs, without revisiting historical raw data.

Input–expert relevance. At inference time, we evaluate the likelihood of the input’s routing representation under each expert distribution and normalize these likelihoods using Bayes’ rule. For the projector experts, the relevance scores are:

$$
R _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) = \frac { 1 } { Z _ { \mathrm { p } } } P ( \mathbf { E } _ { \mathrm { p } } ^ { ( i ) } ) \mathcal { N } \left( \mathbf { h } _ { \mathrm { p } } ( \mathbf { x } ) ; \pmb { \mu } _ { \mathrm { p } } ^ { ( i ) } , \pmb { \Sigma } _ { \mathrm { p } } \right) .\tag{8}
$$

Here, $Z _ { \mathrm { p } }$ is the normalization constant given by:

$$
Z _ { \mathrm { p } } = \sum _ { k = 1 } ^ { T } P ( \mathbf { E } _ { \mathrm { p } } ^ { ( k ) } ) \mathcal { N } \left( \mathbf { h } _ { \mathrm { p } } ( \mathbf { x } ) ; \pmb { \mu } _ { \mathrm { p } } ^ { ( k ) } , \pmb { \Sigma } _ { \mathrm { p } } \right) .
$$

We compute $R _ { 1 } ^ { ( j ) } ( { \bf x } )$ analogously using the LLMside routing representation and distribution statistics. We adopt uniform expert priors because task frequencies at inference time are unknown.

Discussions. By modeling each expert’s traininginput distribution, we estimate its relevance to the current input without requiring task identities. The frozen feature space keeps statistics comparable across learning stages, while incremental updates enable rehearsal-free relevance estimation. These statistics also provide a shared basis for coordinated expert routing and subsequent consolidation.

## 4.3 Compatibility-Aware Routing

The target joint routing distribution Π accounts for both input–expert relevance and cross-module compatibility. To use this distribution for efficient inference, we require mixture weights for the two expert pools. Specifically, we first combine the projector outputs into a single visual representation, which is then processed by the LLM with weighted LoRA expert branches in a single forward pass. Since all LLM expert branches share this visual representation, the projector mixture weights do not depend on which LLM expert is considered. We therefore approximate the target joint distribution using a factorized routing distribution:

$$
q _ { i j } ( \mathbf { x } ) = q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) q _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ) .\tag{9}
$$

Although this approximation is factorized, we optimize its two factors jointly to account for crossmodule compatibility.

Compatibility-aware approximation. Jointly trained projector and LLM experts provide a natural compatibility prior, as their adaptations have

been learned together. We encode this preference as:

$$
C _ { i j } = \exp ( \lambda \mathbb { I } [ i = j ] ) ,\tag{10}
$$

where $\lambda \geq 0$ controls the preference for jointly trained expert pairs. This prior favors established pairings while allowing other combinations when supported by input–expert relevance.

We obtain the routing weights by minimizing the KL divergence from the target joint distribution:

$$
\left( q _ { \mathrm { p } } ^ { \star } , q _ { \mathrm { l } } ^ { \star } \right) = \underset { q _ { \mathrm { p } } \in \Delta _ { \mathrm { p } } , q _ { \mathrm { l } } \in \Delta _ { \mathrm { l } } } { \arg \operatorname* { m i n } } D _ { \mathrm { K L } } \left( q _ { \mathrm { p } } q _ { \mathrm { l } } \left. \right. \Pi \right) ,\tag{11}
$$

where $\Delta _ { \mathrm { p } }$ and $\Delta _ { \mathrm { l } }$ denote the probability simplices over the projector and LLM expert pools, respectively, and we omit the dependence on x throughout this section for readability.

Coordinated routing updates. We solve the optimization problem in Eq. (11) using coordinate ascent variational inference (CAVI) (Blei et al., 2017), alternately updating one routing distribution while holding the other fixed. We initialize the routing weights with the relevance scores: $q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ; 0 ) = \bar { R } _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) , q _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ; 0 ) = R _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } )$ . At iteration $s ,$ the coordinate updates are:

$$
\begin{array} { r l } & { q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ; s + 1 ) \propto R _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { l } } ^ { ( i ) } ( \mathbf { x } ; s ) } , } \\ & { q _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ; s + 1 ) \propto R _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { p } } ^ { ( j ) } ( \mathbf { x } ; s + 1 ) } . } \end{array}\tag{12}
$$

Each update is normalized over the corresponding expert pool. The detailed derivation of these coordinate updates is provided in Appendix B.

Each update balances an expert’s relevance to the input with the current routing weight of its paired expert in the other module. We perform a fixed number of iterations and use the resulting distributions as the final routing weights.

Expert mixture inference. We use ${ q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) }$ to weight and sum the projector expert outputs, forming a shared visual representation that is concatenated with the instruction embeddings. The LLM then processes these inputs using the MoE-LoRA formulation in Eq. (3), with $q _ { 1 } ^ { ( j ) } ( \mathbf { x } )$ serving as the expert routing weights at each adapted layer. This applies the coordinated routing weights within a single LLM forward pass.

Discussions. This approximation transforms the compatibility-aware joint distribution over expert pairs into tractable mixture weights for inference. The coordinated updates allow each module’s routing decisions to inform the other, while retaining input–expert relevance as an anchor. This incorporates cross-module compatibility into expert mixing without separately executing every projector– LLM expert pair.

## 4.4 Relevance-Guided Expert Consolidation

Introducing a new expert pair for every task causes storage costs to grow with the task sequence. To keep the expert pool bounded, we impose a capacity of $K _ { \mathrm { m a x } }$ expert pairs. Whenever this capacity is exceeded, we identify and consolidate the most similar expert pairs using the distribution statistics maintained by Section 4.2. Projector and LLM experts are consolidated together to maintain the pairing structure used by Section 4.3.

To identify potentially redundant expert pairs, we measure their distributional proximity in both expert pools. Under the shared-covariance Gaussian model in Eq. (5), we compute:

$$
d _ { \mathrm { m } } ( i , j ) = \left( { \pmb { \mu } } _ { \mathrm { m } } ^ { ( i ) } - { \pmb \mu } _ { \mathrm { m } } ^ { ( j ) } \right) ^ { \top } \ d \Sigma _ { \mathrm { m } } ^ { - 1 } \left( { \pmb \mu } _ { \mathrm { m } } ^ { ( i ) } - { \pmb \mu } _ { \mathrm { m } } ^ { ( j ) } \right) ,\tag{13}
$$

where $\mathrm { ~ m ~ } \in \{ \mathrm { p } , 1 \}$ denotes the projector or LLM expert pool.

We select the pair of indices with the smallest weighted combination of projector-side and language-side distribution distances. Similar input distributions indicate potential expert redundancy and thus guide consolidation. For the selected indices u and v, we define the sample-count weights:

$$
w _ { u } = \frac { N _ { u } } { N _ { u } + N _ { v } } , \qquad w _ { v } = \frac { N _ { v } } { N _ { u } + N _ { v } } ,\tag{14}
$$

where $N _ { u }$ and $N _ { v }$ denote the corresponding sample counts. We merge the projector experts $\mathbf { E } _ { \mathrm { p } } ^ { ( u ) }$ and $\mathbf { E } _ { \mathfrak { p } } ^ { ( v ) }$ into an single expert by averaging their parameters using these sample-count weights:

$$
\Theta _ { \mathrm { p } } ^ { \mathrm { ( n e w ) } } = w _ { u } \Theta _ { \mathrm { p } } ^ { ( u ) } + w _ { v } \Theta _ { \mathrm { p } } ^ { ( v ) } ,\tag{15}
$$

where $\Theta _ { \mathfrak { p } } ^ { ( u ) }$ denotes the parameters of $\mathbf { E } _ { \mathrm { p } } ^ { ( u ) }$ . For the LLM experts ${ \bf E } _ { 1 } ^ { ( u ) }$ and $\mathbf { E } _ { 1 } ^ { ( v ) }$ , we combine their weight updates using the same weights $w _ { u }$ and $w _ { v }$ , and apply truncated SVD (Eckart and Young, 1936) to the combined update at each adapted layer to retain the original LoRA rank r.

We also update the expert-specific means used for routing:

$$
\pmb { \mu } _ { m } ^ { \mathrm { ( n e w ) } } = w _ { u } \pmb { \mu } _ { m } ^ { \mathrm { ( } u \mathrm { ) } } + w _ { v } \pmb { \mu } _ { m } ^ { \mathrm { ( } v \mathrm { ) } } , \quad m \in \{ \mathrm { p } , 1 \} .\tag{16}
$$

The shared covariance $\Sigma _ { m }$ remains unchanged during consolidation, as it is estimated from the accumulated within-task scatter of the original training tasks. The merged experts $\mathbf { E } _ { \mathrm { p } } ^ { \mathrm { ( n e w ) } }$ and $\mathbf { E } _ { \mathrm { l } } ^ { \mathrm { ( n e w ) } }$ , together with their updated means, replace the original entries under a common index for coordinated routing. The corresponding sample count is updated to $N _ { \mathrm { n e w } } = N _ { u } + N _ { v }$ . Consolidation is repeated until the number of expert pairs does not exceed $K _ { \mathrm { m a x } }$

## 5 Experiment

## 5.1 Experiment Setup

Datasets. We evaluate on CoIN (Chen et al., 2024) and UCIT (Guo et al., 2025a), two benchmarks for multimodal continual instruction tuning. CoIN covers eight tasks, namely ScienceQA (Lu et al., 2022), TextVQA (Singh et al., 2019), ImageNet (Deng et al., 2009), GQA (Hudson and Manning, 2019), VizWiz (Gurari et al., 2018), Grounding (Kazemzadeh et al., 2014; Mao et al., 2016), VQAv2 (Goyal et al., 2017), and OCR-VQA (Mishra et al., 2019). These tasks span question answering, image classification, and referring expression grounding. UCIT comprises ArxivQA (Li et al., 2024), CLEVR-Math (Lindström and Abraham, 2022), IconQA (Lu et al., 2021), ImageNet-R (Hendrycks et al., 2021), VizWizcaption (Gurari et al., 2020), and Flickr30k (Plummer et al., 2015). It supplements CoIN with tasks selected to reduce overlap with LLaVA’s original training data, including image captioning. Detailed dataset statistics for both benchmarks are provided in Appendix A.

Comparison Methods. We compare sequential fine-tuning, EWC (Kirkpatrick et al., 2017), O-LoRA (Wang et al., 2023), MoELoRA (Chen et al., 2024), HiDe-LLaVA (Guo et al., 2025a), ProgLoRA (Yu et al., 2025), SAME (Xie et al., 2026) and DISCO (Guo et al., 2025b).

Training Setup. Following Prism (Tang et al., 2026), we use LLaVA-v1.5-7B (Liu et al., 2024) on CoIN (Chen et al., 2024) and UCIT (Guo et al., 2025a), with a CLIP ViT-L/14 encoder (336 × 336 inputs) (Radford et al., 2021) and a two-layer GELU projector. We train projector experts and LLM LoRA experts $( r = 1 6 , \alpha = 3 2 )$ , keeping the vision encoder and base LLM frozen. Each task uses one epoch of AdamW with projector/LoRA learning rates of $5 \times 1 0 ^ { - 5 } / 2 \times 1 0 ^ { - 4 }$ and a 0.03 warmup ratio. Training is conducted on four RTX

<table><tr><td>Methods</td><td>ScienceQA TextVQA ImageNet GQA VizWiz Grounding VQAv2 OCR-VQA</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>Last</td></tr><tr><td>Finetune</td><td>26.00</td><td>25.38</td><td>28.51</td><td>33.07</td><td>26.52</td><td>0.10</td><td>40.00</td><td>52.92</td><td>29.06</td></tr><tr><td>EWC</td><td>59.75</td><td>51.06</td><td>31.88</td><td>51.8442.26</td><td></td><td>34.96</td><td>55.30</td><td>64.55</td><td>48.95</td></tr><tr><td>O-LoRA</td><td>72.56</td><td>57.66</td><td>62.84</td><td>59.14 48.43</td><td></td><td>58.97</td><td>63.21</td><td>63.31</td><td>60.77</td></tr><tr><td>MoELoRA</td><td>47.34</td><td>32.91</td><td>38.73</td><td>37.15</td><td>42.48</td><td>0.97</td><td>42.77</td><td>57.50</td><td>37.48</td></tr><tr><td>HiDe-LLaVA</td><td>73.20</td><td>56.92</td><td>69.28</td><td>61.33</td><td>50.76</td><td>59.18</td><td>67.12</td><td>64.76</td><td>62.82</td></tr><tr><td>ProgLoRA</td><td>74.84</td><td>51.83</td><td>83.90</td><td>49.93</td><td>53.87</td><td>31.19</td><td>62.71</td><td>64.44</td><td>59.09</td></tr><tr><td>SAME</td><td>78.35</td><td>60.69</td><td>90.21</td><td>61.70</td><td>54.13</td><td>59.87</td><td>66.04</td><td>63.59</td><td>66.82</td></tr><tr><td>CART</td><td>84.13</td><td>63.85</td><td>97.39</td><td>61.04</td><td>61.94</td><td>81.08</td><td>67.88</td><td>66.04</td><td>72.92</td></tr></table>

Table 1: Main results on the CoIN benchmark with LLaVA-v1.5-7B as the backbone (higher is better). The best and second-best values are marked in bold and underline, respectively.
<table><tr><td>Methods</td><td>ImgNet-R</td><td>ArxivQA</td><td>VizWiz</td><td>IconQA</td><td>CLEVR</td><td>Flickr30k</td><td>Last</td></tr><tr><td>Finetune</td><td>58.03</td><td>77.63</td><td>44.39</td><td>67.40</td><td>61.77</td><td>58.22</td><td>61.24</td></tr><tr><td>EWC</td><td>39.05</td><td>77.88</td><td>43.24</td><td>45.33</td><td>39.72</td><td>55.94</td><td>50.19</td></tr><tr><td>O-LoRA</td><td>77.50</td><td>78.07</td><td>44.50</td><td>63.13</td><td>64.73</td><td>58.16</td><td>64.35</td></tr><tr><td>MoELoRA</td><td>70.07</td><td>77.70</td><td>44.69</td><td>50.03</td><td>54.03</td><td>57.34</td><td>58.98</td></tr><tr><td>HiDe-LLaVA</td><td>84.03</td><td>90.73</td><td>44.43</td><td>58.93</td><td>41.37</td><td>54.25</td><td>62.29</td></tr><tr><td>DISCO</td><td>88.88</td><td>94.25</td><td>47.52</td><td>69.50</td><td>60.75</td><td>56.32</td><td>69.54</td></tr><tr><td>SAME</td><td>89.91</td><td>91.40</td><td>55.33</td><td>77.51</td><td>68.85</td><td>55.43</td><td>73.07</td></tr><tr><td>CART</td><td>90.93</td><td>91.07</td><td>47.73</td><td>88.00</td><td>81.20</td><td>55.12</td><td>75.67</td></tr></table>

Table 2: Main results on the UCIT benchmark with LLaVA-v1.5-7B as the backbone (higher is better). The best and second-best values are marked in bold and underline, respectively.

4090 GPUs with a per-device batch size of 4, 2 gradient accumulation steps, DeepSpeed ZeRO-2, and bf16 precision. Unless otherwise specified, we use λ = 2, S = 10 CAVI iterations, capacity $K _ { \operatorname* { m a x } } = 5$ expert pairs, and covariance regularization $\epsilon = 0 . 1$ . We use PyTorch 2.8.0 (Paszke et al., 2019) and pycocotools 2.0.11 (Lin et al., 2014).

Evaluation Metrics. Following CoIN (Chen et al., 2024) and Prism (Tang et al., 2026), we report Last Accuracy (Last) and Average Accuracy $\begin{array} { r } { ( \mathrm { A v g } ) \colon \mathcal { A } _ { \mathrm { L a s t } } = \frac { 1 } { T } \sum _ { s = 1 } ^ { T } \mathcal { A } _ { s , T } } \end{array}$ and $\mathcal { A } _ { \mathrm { A v g } } =$ $\begin{array} { r } { \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \frac { 1 } { t } \sum _ { s = 1 } ^ { t } \mathcal { A } _ { s , t } , } \end{array}$ where $\mathcal { A } _ { s , t }$ is the accuracy on task s after learning t tasks, and T is the total number of tasks. Last evaluates final-stage performance, while Avg averages seen-task performance across stages.

## 5.2 Main Results

Tables 1 and 2 summarize the main results on CoIN and UCIT. On CoIN, CART obtains the highest Last Accuracy of 72.92 and exceeds the strongest baseline SAME by 6.10 points. CART also ranks first on UCIT, reaching 75.67 and improving over SAME by 2.60 points. Its leading performance across both the standard CoIN benchmark and the cleaner UCIT setting shows that the gains remain consistent under different task compositions and data characteristics.

## 5.3 Ablation Study

Table 3 evaluates distribution-aware relevance and routing strategies on UCIT. Removing distributionaware relevance lowers average accuracy from 75.67 to 71.68, with the largest drops on ImageNet-R and IconQA, supporting input–expert distribution matching. To assess the effectiveness of compatibility-aware routing, we compare three routing alternatives: Hard Routing independently selects each module’s most relevant expert; Independent Routing directly uses the relevance distributions as mixture weights, i.e., $q _ { m } ( \mathbf { x } ) = R _ { m } ( \mathbf { x } )$ for $m \in \{ { \mathfrak { p } } , 1 \}$ , without cross-module coordination.; Naive Joint Routing assigns identical routing weights to both modules by normalizing the element-wise product of their relevance distributions, i.e., $q _ { \mathrm { p } } ( \mathbf { x } ) = q _ { \mathrm { l } } ( \mathbf { x } ) \propto R _ { \mathrm { p } } ( \mathbf { x } ) \odot R _ { \mathrm { l } } ( \mathbf { x } )$ . Their average accuracies are 68.29, 69.87, and 71.80, respectively, versus 75.67 for CART, demonstrating the benefit of cross-module compatibility in coordinating projector and LLM routing.

## 5.4 Further Analysis

Effectiveness of Relevance-Guided Expert Consolidation. On six-task UCIT, Figure 4 compares consolidation strategies under different expert-pool limits $K _ { \operatorname* { m a x } } . \mathrm { A t } K _ { \operatorname* { m a x } } = 4$ , our strategy achieves 70.9 final-stage and 82.3 average accuracy, outperforming random consolidation by 6.2 and 1.0 points, respectively. This advantage persists at $K _ { \operatorname* { m a x } } = 5$ , while all strategies perform similarly at $K _ { \operatorname* { m a x } } = 3$ due to limited consolidation flexibility. These results show that merging experts with similar projector- and language-side distributions better preserves task knowledge within a bounded expert pool.

<table><tr><td>Variants</td><td>ImgNet-R</td><td>ArxivQA</td><td>VizWiz</td><td>IconQA</td><td>CLEVR</td><td>Flickr30k</td><td>Last</td><td>∆</td></tr><tr><td>Component Ablation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o Distribution-Aware Relevance</td><td>80.27</td><td>91.60</td><td>46.38</td><td>79.07</td><td>78.67</td><td>54.06</td><td>71.68</td><td>-3.99</td></tr><tr><td>Routing Strategy Comparison</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hard Routing</td><td>73.87</td><td>91.73</td><td>45.88</td><td>71.73</td><td>73.73</td><td>52.80</td><td>68.29</td><td>-7.38</td></tr><tr><td>Independent Routing</td><td>85.07</td><td>92.40</td><td>46.41</td><td>71.20</td><td>69.60</td><td>54.53</td><td>69.87</td><td>-5.80</td></tr><tr><td>Naive Joint Routing</td><td>86.33</td><td>91.33</td><td>45.99</td><td>76.33</td><td>75.67</td><td>55.13</td><td>71.80</td><td>-3.87</td></tr><tr><td>CART (Full)</td><td>90.93</td><td>91.07</td><td>47.73</td><td>88.00</td><td>81.20</td><td>55.12</td><td>75.67</td><td>一</td></tr></table>

Table 3: Ablation study on the UCIT benchmark. ∆ denotes the average accuracy change relative to full CART. Best results are highlighted in bold.

![](images/6a5677942848a0d8a09b4d4c5b74d5d0bec7c1331b8a80adc995a9de7c5af144.jpg)

![](images/458fa5cf6e81ab4ef28ca3b58c9175d211a938bd3b12145ddfe435a8a41a900f.jpg)

![](images/5812a75e22967754da64a7d27166b2102f80312690ead0444ca38a702c49991a.jpg)

![](images/fa2fe31bf5d99420c3175d13789b3d03279ad9121106b0d91e9ab442f1b662ea.jpg)

Figure 4: Analysis of expert-pool capacity and CAVI hyperparameters on the six-task UCIT benchmark. The left panels compare consolidation strategies under different expert-pool limits, while the right panels examine compatibility strength and the number of CAVI iterations. Overall, stronger compatibility coupling improves retention, and performance stabilizes after only a few CAVI iterations.  
![](images/527676e3c9529bb3a0617e5f47f376bfd96cc9504a727f69254882620e221279.jpg)  
Figure 5: Qualitative comparison with NCM-based routing. CART better attends to instruction-relevant visual evidence and produces correct answers.

Qualitative Visualization. Figure 5 compares CART with an NCM-based baseline that retains the LoRA-MoE architecture but replaces Distribution-Aware Expert Relevance and Compatibility-Aware Routing with nearest-class-mean routing. On CLEVR, CART identifies the blocks relevant to subtraction and answers one instead of three. On IconQA, it attends near the correct ruler endpoint and predicts three centimeters rather than ten. These examples show that relevance estimation and expert coordination improve routing and preserve instruction-relevant cross-modal alignment.

## 6 Conclusion

This work investigates projector–language mismatch as a source of forgetting in multimodal continual instruction tuning. We introduce CART, which preserves paired projector and LLM experts and jointly routes them using input–expert relevance and cross-module compatibility for taskagnostic inference. Relevance-guided consolidation bounds model growth by merging redundant expert pairs. Experiments on CoIN and UCIT demonstrate superior performance, with ablations supporting the effectiveness of relevance estimation and coordinated routing. These results highlight the importance of cross-module compatibility for continual multimodal adaptation.

## Limitations

While CART demonstrates strong performance on CoIN and UCIT, it has not yet been evaluated on substantially longer task sequences. Extending evaluation to longer continual streams would provide a more comprehensive assessment of longterm knowledge retention and remains an important direction for future work.

## Ethical Considerations

CART may inherit biases, harmful behaviors, and hallucinations from pretrained models and training data. Distribution-based routing may be less reliable for underrepresented inputs, potentially affecting users of applications such as visual assistance. Downstream deployment therefore requires fairness and safety evaluation after model updates, along with appropriate human oversight.

More adaptable multimodal models could be misused for privacy-invasive image analysis or misleading content generation. Although our routing mechanism does not revisit historical raw data, retaining aggregate statistics and learned experts does not provide a formal privacy guarantee. Sensitive applications require appropriate data governance and privacy assessment.

Training and evaluation consume energy. Lowrank adaptation and expert consolidation control trainable parameters and expert-pool growth, but do not necessarily reduce overall energy consumption or carbon emissions.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

David M Blei, Alp Kucukelbir, and Jon D McAuliffe. 2017. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518):859–877.

Cheng Chen, Junchen Zhu, Xu Luo, Heng Tao Shen, Jingkuan Song, and Lianli Gao. 2024. CoIN: A benchmark of continual instruction tuning for multimodel large language models. In Advances in Neural Information Processing Systems, volume 37, pages 57817–57840.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards

general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36:49250–49267.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. ImageNet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 248–255. Ieee.

Carl Eckart and Gale Young. 1936. The approximation of one matrix by another of lower rank. Psychometrika, 1(3):211–218.

Robert M French. 1999. Catastrophic forgetting in connectionist networks. Trends in cognitive sciences, 3(4):128–135.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6904–6913.

Haiyang Guo, Fanhu Zeng, Ziwei Xiang, Fei Zhu, Da-Han Wang, Xu-Yao Zhang, and Cheng-Lin Liu. 2025a. HiDe-LLaVA: Hierarchical decoupling for continual instruction tuning of multimodal large language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13572– 13586.

Haiyang Guo, Fanhu Zeng, Fei Zhu, Wenzhuo Liu, Da-Han Wang, Jian Xu, Xu-Yao Zhang, and Cheng-Lin Liu. 2025b. Federated continual instruction tuning. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1325–1335. IEEE.

Haiyang Guo, Fanhu Zeng, Fei Zhu, Jiayi Wang, Xukai Wang, Jingang Zhou, Hongbo Zhao, Wenzhuo Liu, Shijie Ma, Xu-Yao Zhang, et al. 2025c. A comprehensive survey on continual learning in generative models. arXiv e-prints, pages arXiv–2506.

Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018. VizWiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3608–3617.

Danna Gurari, Yinan Zhao, Meng Zhang, and Nilavra Bhattacharya. 2020. Captioning images taken by people who are blind. In European Conference on Computer Vision.

Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. 2021. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8349.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Drew A Hudson and Christopher D Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick. 2017. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2901–2910.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pages 787– 798.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, et al. 2017. Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13):3521–3526.

Lei Li, Yuqi Wang, Runxin Xu, Peiyi Wang, Xiachong Feng, Lingpeng Kong, and Qi Liu. 2024. Multimodal ArXiv: A dataset for improving scientific comprehension of large vision-language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14369–14387.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. 2014. Microsoft coco: Common objects in context. In European conference on computer vision, pages 740–755. Springer.

Adam Dahlgren Lindström and Savitha Sam Abraham. 2022. CLEVR-Math: A dataset for compositional language, visual and mathematical reasoning. arXiv preprint arXiv:2208.05358.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26296–26306.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892– 34916.

Shayne Longpre, Le Hou, Tu Vu, Albert Webson, Hyung Won Chung, Yi Tay, Denny Zhou, Quoc V Le, Barret Zoph, Jason Wei, et al. 2023. The flan collection: Designing data and methods for effective instruction tuning. In Proceedings of the International Conference on Machine Learning, pages 22631–22648. PMLR.

Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, volume 35, pages 2507– 2521.

Pan Lu, Liang Qiu, Jiaqi Chen, Tony Xia, Yizhou Zhao, Wei Zhang, Zhou Yu, Xiaodan Liang, and Song-Chun Zhu. 2021. IconQA: A new benchmark for abstract diagram understanding and visual language reasoning. arXiv preprint arXiv:2110.13214.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan L Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11–20.

Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. OCR-VQA: Visual question answering by reading text in images. In 2019 international conference on document analysis and recognition (ICDAR), pages 947–952. IEEE.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. 2019. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32.

Bryan A Plummer, Liwei Wang, Chris M Cervantes, Juan C Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2015. Flickr30k Entities: Collecting region-to-phrase correspondences for richer imageto-sentence models. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2641–2649.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the International Conference on Machine Learning, pages 8748–8763. PMLR.

Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards VQA models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.

Jun-Tao Tang, Yu-Cheng Shi, Zhen-Hao Xie, and Da-Wei Zhou. 2026. Prism: A plug-in reproducible infrastructure for scalable multimodal continual instruction tuning. arXiv preprint arXiv:2605.26110.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Shengbang Tong, David Fan, Jiachen Zhu, Yunyang Xiong, Xinlei Chen, Koustuv Sinha, Michael Rabbat, Yann LeCun, Saining Xie, and Zhuang Liu. 2025. Metamorph: Multimodal understanding and generation via instruction tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17001–17012.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuan-Jing Huang. 2023. Orthogonal subspace learning for language model continual learning. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10658–10671.

Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2021. Finetuned language models are zero-shot learners. arXiv preprint arXiv:2109.01652.

Zhen-Hao Xie, Jun-Tao Tang, Yu-Cheng Shi, Han-Jia Ye, De-Chuan Zhan, and Da-Wei Zhou. 2026. Same: Stabilized mixture-of-experts for multimodal continual instruction tuning. arXiv preprint arXiv:2602.01990.

Yahan Yu, Duzhen Zhang, Yong Ren, Xuanle Zhao, Xiuyi Chen, and Chenhui Chu. 2025. Progressive LoRA for multimodal continual instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 2779–2796.

Fanhu Zeng, Fei Zhu, Haiyang Guo, Xu-Yao Zhang, and Cheng-Lin Liu. 2025. ModalPrompt: Towards efficient multimodal continual instruction tuning with dual-modality guided prompt. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 12137–12152.

Aimen Zerroug, Mohit Vaishnav, Julien Colin, Sebastian Musslick, and Thomas Serre. 2022. A benchmark for compositional visual reasoning. Advances in Neural Information Processing Systems, 35:29776– 29788.

Yuexiang Zhai, Shengbang Tong, Xiao Li, Mu Cai, Qing Qu, Yong Jae Lee, and Yi Ma. 2023. Investigating the catastrophic forgetting in multimodal large language models. In NeurIPS 2023 Workshop on Instruction Tuning and Instruction Following.

Duzhen Zhang, Yahan Yu, Jiahua Dong, Chenxing Li, Dan Su, Chenhui Chu, and Dong Yu. 2024. Mm-llms: Recent advances in multimodal large language models. Findings of the association for computational linguistics: ACL 2024, pages 12401–12430.

Shengyu Zhang, Linfeng Dong, Xiaoya Li, Sen Zhang, Xiaofei Sun, Shuhe Wang, Jiwei Li, Runyi Hu, Tianwei Zhang, Guoyin Wang, et al. 2026. Instruction tuning for large language models: A survey. ACM Computing Surveys, 58(7):1–36.

## A Dataset Details

Tables 4 and 5 summarize the domains and split sizes of the datasets in CoIN and UCIT, respectively. Train and Test denote the numbers of samples in the corresponding splits.

CoIN. CoIN contains eight datasets spanning image classification, visual question answering, referring expression comprehension, visual assistance, and scientific reasoning. Its training sets contain between 20,000 and 34,602 samples, while each test set contains 3,000 samples.

<table><tr><td>Dataset</td><td>Domain</td><td>Train</td><td>Test</td></tr><tr><td>GQA</td><td>Compositional VQA</td><td>20,000</td><td>3,000</td></tr><tr><td>ImageNet</td><td>Image classification</td><td>20,000</td><td>3,000</td></tr><tr><td>TextVQA</td><td>Scene-text VQA</td><td>34,602</td><td>3,000</td></tr><tr><td>OCRVQA</td><td>OCR-based VQA</td><td>20,000</td><td>3,000</td></tr><tr><td>REC-COCO</td><td>Referring expressions</td><td>20,000</td><td>3,000</td></tr><tr><td>VizWiz</td><td>Visual assistance</td><td>20,523</td><td>3,000</td></tr><tr><td>VQAv2</td><td>General VQA</td><td>20,000</td><td>3,000</td></tr><tr><td>ScienceQA</td><td>Science reasoning</td><td>20,000</td><td>3,000</td></tr></table>

Table 4: Dataset statistics for the eight-task CoIN benchmark.

UCIT. UCIT contains six datasets spanning artistic image recognition, academic figure understanding, visual assistance, icon-based question answering, synthetic reasoning, and image captioning. Its training sets contain between 24,000 and 40,000 samples, while each test set contains 3,000 samples.

<table><tr><td>Dataset</td><td>Domain</td><td>Train</td><td>Test</td></tr><tr><td>ImageNet-R</td><td>Artistic</td><td>24,000</td><td>3,000</td></tr><tr><td>ArxivQA</td><td>Academic figures</td><td>40,000</td><td>3,000</td></tr><tr><td>VizWiz</td><td>Visual assistance</td><td>40,000</td><td>3,000</td></tr><tr><td>IconQA</td><td>Icons</td><td>30,000</td><td>3,000</td></tr><tr><td>CLEVR-Math</td><td>Synthetic reasoning</td><td>40,000</td><td>3,000</td></tr><tr><td>Flickr30k</td><td>Captioning</td><td>40,000</td><td>3,000</td></tr></table>

Table 5: Dataset statistics for the six-task UCIT benchmark.

## B Derivation of Compatibility-Aware Routing

We derive the coordinate updates used to optimize Eq. (11). For a fixed input x, we omit its dependence in the intermediate derivation. We assume that the relevance scores are strictly positive, as provided by the Gaussian model with positive expert priors.

Expanding the KL objective. With the compatibility prior $C _ { i j } = \exp ( \lambda \mathbb { I } [ i = j ] )$ , the target joint distribution is:

$$
\Pi _ { i j } = \frac { R _ { \mathrm { p } } ^ { ( i ) } R _ { 1 } ^ { ( j ) } \ \mathrm { e x p } ( \lambda \mathbb { I } [ i = j ] ) } { Z _ { \Pi } } ,\tag{17}
$$

where $\begin{array} { r } { Z _ { \Pi } = \sum _ { u , v } R _ { \mathrm { p } } ^ { ( u ) } R _ { \mathrm { l } } ^ { ( v ) } \exp ( \lambda \mathbb { I } [ u = v ] ) } \end{array}$ is independent of the routing weights. Substituting this expression into the KL divergence gives:

$$
\begin{array} { l } { { \displaystyle p _ { \mathrm { S L } } ( q _ { 0 } , \mu _ { 1 } ) \| ~ } } \\ { { \displaystyle = \sum _ { i , j } q _ { i } ^ { ( i ) } q _ { j } ^ { ( j ) } \log \frac { q _ { i } ^ { ( i ) } } { T _ { 1 , i } } } } \\ { { \displaystyle = \sum _ { i , j } q _ { i } ^ { ( i ) } q _ { j } ^ { ( j ) } \log \frac { q _ { i } ^ { ( i ) } } { T _ { 1 , i } } + \log \frac { q _ { i } ^ { ( j ) } } { T _ { 1 , i } } } } \\ { { \displaystyle \qquad - \lambda [ i - j ] + \log Z _ { 1 , i } } } \\ { { \displaystyle = \sum _ { i } q _ { i } ^ { ( i ) } \log \frac { q _ { i } ^ { ( j ) } } { T _ { 1 , i } ^ { ( i ) } } + \sum _ { j } q _ { i } ^ { ( j ) } \log \frac { q _ { i } ^ { ( j ) } } { T _ { 1 , i } ^ { ( j ) } } } } \\ { { \displaystyle \qquad - \lambda \sum _ { j } q _ { i } ^ { ( j ) } q _ { j } ^ { ( i ) } + \log Z _ { 1 } } . } \end{array}\tag{18}
$$

The last equality follows from $\begin{array} { r l } { \sum _ { i } q _ { \mathfrak { p } } ^ { ( i ) } } & { { } = } \end{array}$ $\begin{array} { r } { \sum _ { j } q _ { \mathrm { l } } ^ { ( j ) } = 1 } \end{array}$ . Dropping the constant log $Z _ { \Pi }$ recovers the objective $\mathcal { L }$

Deriving the coordinate updates. We minimize L with respect to one routing distribution while holding the other fixed. This is equivalent to coordinate ascent on the corresponding variational objective, yielding the standard coordinate ascent variational inference procedure (Blei et al., 2017).

First, fix $q _ { \mathrm { l } }$ and optimize $q _ { \mathrm { p } } .$ . Introducing a Lagrange multiplier α for its normalization constraint gives:

$$
\begin{array} { l } { { \displaystyle { \mathcal { I } } _ { \mathfrak { p } } ( q _ { \mathfrak { p } } , \alpha ) = \sum _ { i } q _ { \mathfrak { p } } ^ { ( i ) } \log \frac { q _ { \mathfrak { p } } ^ { ( i ) } } { R _ { \mathfrak { p } } ^ { ( i ) } } - \lambda \sum _ { i } q _ { \mathfrak { p } } ^ { ( i ) } q _ { \mathfrak { l } } ^ { ( i ) } } } \\ { { \displaystyle \qquad + \alpha \left( \sum _ { i } q _ { \mathfrak { p } } ^ { ( i ) } - 1 \right) , } } \end{array}\tag{19}
$$

where terms independent of $q _ { \mathrm { p } }$ have been omitted. Setting the derivative with respect to $q _ { \mathfrak { p } } ^ { ( i ) }$ to zero yields:

$$
\frac { \partial \mathcal { I } _ { \mathsf { p } } } { \partial q _ { \mathsf { p } } ^ { ( i ) } } = \log \frac { q _ { \mathsf { p } } ^ { ( i ) } } { R _ { \mathsf { p } } ^ { ( i ) } } + 1 - \lambda q _ { 1 } ^ { ( i ) } + \alpha = 0 .\tag{20}
$$

Rearranging this condition gives:

$$
q _ { \mathrm { p } } ^ { ( i ) } = R _ { \mathrm { p } } ^ { ( i ) } \exp \left( \lambda q _ { \mathrm { l } } ^ { ( i ) } - 1 - \alpha \right) .\tag{21}
$$

The factor $\exp ( - 1 - \alpha )$ is shared across projector experts and is determined by normalization. Therefore:

$$
q _ { \mathrm { p } } ^ { ( i ) } = \frac { R _ { \mathrm { p } } ^ { ( i ) } \exp ( \lambda q _ { 1 } ^ { ( i ) } ) } { \sum _ { k } R _ { \mathrm { p } } ^ { ( k ) } \exp ( \lambda q _ { 1 } ^ { ( k ) } ) } .\tag{22}
$$

For fixed $q _ { \mathrm { l } }$ , the objective is strictly convex in $q _ { \mathrm { p } } ,$ so this normalized solution is the unique minimizer of the projector coordinate subproblem.

Similarly, fixing $q _ { \mathrm { p } }$ and introducing a multiplier $\beta$ for $\begin{array} { r } { \sum _ { j } q _ { \mathrm { l } } ^ { ( j ) } = 1 } \end{array}$ gives the stationarity condition:

$$
\log \frac { q _ { 1 } ^ { ( j ) } } { R _ { 1 } ^ { ( j ) } } + 1 - \lambda q _ { \mathrm { p } } ^ { ( j ) } + \beta = 0 .\tag{23}
$$

Solving and normalizing yields the unique minimizer of the LLM coordinate subproblem:

$$
q _ { 1 } ^ { ( j ) } = \frac { R _ { 1 } ^ { ( j ) } \exp ( \lambda q _ { \mathrm { p } } ^ { ( j ) } ) } { \sum _ { k } R _ { 1 } ^ { ( k ) } \exp ( \lambda q _ { \mathrm { p } } ^ { ( k ) } ) } .\tag{24}
$$

Sequential routing updates. Restoring the input dependence, we initialize both routing distributions with their corresponding relevance scores:

$$
\begin{array} { r } { q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ; 0 ) = R _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) , } \\ { q _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ; 0 ) = R _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ) . } \end{array}\tag{25}
$$

At iteration s, we first update the projector routing distribution and then use its updated values to optimize the LLM routing distribution:

$$
\begin{array} { r l } & { q _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ; s + 1 ) = \frac { R _ { \mathrm { p } } ^ { ( i ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { l } } ^ { ( i ) } ( \mathbf { x } ; s ) } } { \sum _ { k } R _ { \mathrm { p } } ^ { ( k ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { l } } ^ { ( k ) } ( \mathbf { x } ; s ) } } , } \\ & { q _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ; s + 1 ) = \frac { R _ { \mathrm { l } } ^ { ( j ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { p } } ^ { ( j ) } ( \mathbf { x } ; s + 1 ) } } { \sum _ { k } R _ { \mathrm { l } } ^ { ( k ) } ( \mathbf { x } ) e ^ { \lambda q _ { \mathrm { p } } ^ { ( k ) } ( \mathbf { x } ; s + 1 ) } } . } \end{array}\tag{26}
$$

The relevance scores remain fixed throughout these iterations. Each update combines an expert’s input relevance with the current routing weight of its paired expert in the other module.

<table><tr><td>Seed</td><td>SAME</td><td>DISCO</td><td>CART</td></tr><tr><td>1</td><td>72.93</td><td>68.76</td><td>75.18</td></tr><tr><td>10</td><td>72.38</td><td>69.13</td><td>75.57</td></tr><tr><td>100</td><td>72.56</td><td>68.81</td><td>75.75</td></tr><tr><td>1000</td><td>71.86</td><td>68.50</td><td>75.70</td></tr><tr><td>42 (original)</td><td>73.07</td><td>69.54</td><td>75.67</td></tr><tr><td>Average</td><td>72.56</td><td>68.95</td><td>75.58</td></tr><tr><td>Std.</td><td>0.48</td><td>0.40</td><td>0.23</td></tr></table>

Table 6: Multi-seed results on UCIT. We report final average accuracy (%) over five random seeds; mean and standard deviation are computed across runs.

Each coordinate update exactly minimizes its subproblem and therefore cannot increase the KL divergence. Since the KL divergence is bounded below by zero, its values converge if the updates are continued. However, the objective is not generally jointly convex, so these updates do not guarantee a global minimizer of Eq. (11). In practice, we perform a fixed number of iterations and use the resulting distributions as the routing weights.

## C Randomness and Task-Order Robustness

## C.1 Multi-Seed Stability

To assess run-to-run stability, we rerun the UCIT experiments with five random seeds under otherwise identical settings. We use the original task order: ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR-Math → Flickr30k. For every run, we fix the random states of Python, NumPy, PyTorch, CUDA, and the data loader workers, and enable deterministic CUDA operations where available. We compare CART with SAME and DISCO, the two strongest baselines in Table 2.

Table 6 shows that CART consistently outperforms the strongest baselines across random seeds while exhibiting the smallest run-to-run variation. This confirms that its performance gains are stable and do not depend on a favorable initialization.

## C.2 Task-Order Robustness

We further evaluate CART’s sensitivity to task order on UCIT. We use seed 42 and keep every other training and evaluation setting identical to the main experiment. We consider three task sequences:

• Original: ImageNet-R → ArxivQA → VizWiz → IconQA → CLEVR-Math → Flickr30k.

<table><tr><td>Metric</td><td>Original</td><td>Reverse</td><td>Fixed random</td></tr><tr><td>Final Avg. Accuracy (%)</td><td>75.68</td><td>74.36</td><td>75.43</td></tr></table>

Table 7: Task-order robustness of CART on UCIT. Final average accuracy (%) is reported under the original, reverse, and fixed random task orders.

![](images/d16780b0492e5ef106f3ecc9bd9d372a364d0281bf3d226aacc289c3e586b2e8.jpg)  
Figure 6: Accuracy–parameter efficiency of MCIT methods on UCIT. CART achieves a Pareto-optimal trade-off between Last Accuracy and adaptation parameters.

• Reverse: Flickr30k → CLEVR-Math → IconQA → VizWiz → ArxivQA → ImageNet-R.

• Fixed random: Flickr30k → CLEVR-Math → ArxivQA → ImageNet-R → IconQA → VizWiz.

As shown in Table 7, CART maintains strong performance under all three task sequences. The limited variation across orders indicates that the method is robust to changes in the chronological arrival of tasks.

## D Parameter Efficiency

We further compare the accuracy–parameter tradeoff of MCIT methods on UCIT. In Figure 6, the horizontal axis shows the number of adaptation parameters used at inference, excluding router, mask, and pretrained backbone parameters, while the vertical axis reports Last Accuracy (Last), as defined in the Evaluation Metrics paragraph. The radius of each bubble represents the number of trainable parameters introduced for each task. Overall, CART achieves a Pareto-optimal trade-off: methods using more adaptation parameters do not attain higher Last Accuracy, whereas methods with higher Last Accuracy require at least as many adaptation parameters.

Algorithm 1 CART Training (per task t)   
Require: $\mathcal { D } _ { t } ;$ frozen $\phi , \psi ;$ current expert pools $\pmb { \mathcal { E } } _ { \mathrm { p } } , \pmb { \mathcal { E } } _ { \mathrm { l } }$ and   
their statistics; capacity $K _ { \mathrm { m a x } } ;$ covariance regularizer $\epsilon ;$   
LoRA rank r   
Ensure: Updated paired expert pools $\pmb { \mathcal { E } } _ { \mathrm { p } } , \pmb { \mathcal { E } } _ { \mathrm { l } }$ and their distri  
bution statistics   
Training task-specific experts $\mathbf { E } _ { \mathrm { p } } ^ { ( t ) } , \mathbf { E } _ { \mathrm { l } } ^ { ( t ) }$ on $\mathcal { D } _ { t }$ by Eq. (2)   
Extract routing representations $\mathbf { \dot { h } } _ { \mathrm { p } } ( \mathbf { x } )$ , h<sub>l</sub>(x) for $\mathbf { x } \in \mathcal { D } _ { t }$   
using $\phi , \psi$   
Model task distribution   
for $m \in \{ { \mathfrak { p } } , 1 \}$ do   
Compute $\mu _ { m } ^ { ( t ) }$ by Eq. $( 6 ) , \mathbf { S } _ { m } ^ { ( t ) }$ by $\mathrm { E q . } ( 7 )$   
Update $\mathbf { S } _ { m } \gets \mathbf { S } _ { m } + \mathbf { S } _ { m } ^ { ( t ) }$ and $\Sigma _ { m } \gets \mathbf { S } _ { m } / N + \epsilon \mathbf { I }$   
end for   
Append the current experts to the pools $\pmb { \mathcal { E } } _ { \mathrm { p } }  \pmb { \mathcal { E } } _ { \mathrm { p } }$ ∪   
$\{ \mathbf { E } _ { \mathrm { p } } ^ { ( t ) } \} , \pmb { \mathcal { E } } _ { \mathrm { l } }  \pmb { \mathcal { E } } _ { \mathrm { l } } \cup \{ \mathbf { E } _ { \mathrm { l } } ^ { ( t ) } \}$   
Expert consolidation   
if $| \dot { \pmb { \mathscr { E } } } _ { \mathrm { p } } | > K _ { \mathrm { m a x } }$ then   
Compute distribution distances $d _ { \mathrm { p } } ( i , j )$ and $d _ { \mathrm { l } } ( i , j )$ for   
all $i < j$ via Eq. (13)   
Select $( i , j )  \arg$ min $_ { i < j } \left[ d _ { \mathsf { p } } ( i , j ) + d _ { \mathsf { l } } ( i , j ) \right]$   
Average the parameters of $\mathbf { E } _ { \mathrm { p } } ^ { ( i ) }$ and $\mathbf { E } _ { \mathrm { p } } ^ { ( j ) }$ by Eq. (15)   
Combine the weight updates of $\mathbf { E } _ { 1 } ^ { ( i ) }$ and $\mathbf { E } _ { 1 } ^ { ( j ) }$ by   
Eq. (15), retaining LoRA rank r via truncated SVD   
Update expert-specific means $\mu _ { \mathrm { p } } ^ { \mathrm { ( n e w ) } } , \mu _ { \mathrm { l } } ^ { \mathrm { ( n e w ) } }$ by Eq. (16)   
end if

## E Pseudocode of CART

Algorithms 1 and 2 outline the training and inference procedures of CART. For each task, we jointly train a projector expert and an LLM LoRA expert, then freeze the pair and retain its input distribution statistics. At inference time, we use these statistics to estimate input–expert relevance and apply compatibility-aware routing to coordinate the two expert mixtures without requiring task identities.

Algorithm 2 CART Inference   
Require: Test input $\mathbf { x } = ( \mathbf { v } , \mathbf { q } ) ;$ trained paired expert pools   
$\bar { \pmb { \mathscr { E } } } _ { \mathrm { p } } , \pmb { \mathscr { E } } _ { \mathrm { l } }$ and their distribution statistics; frozen ϕ, ψ; compat  
ibility strength λ; CAVI iterations S   
Ensure: Response yˆ   
Extract routing representations $\mathbf { h } _ { \mathrm { p } } ( \mathbf { x } )$ , h<sub>l</sub>(x) using $\phi , \psi$   
Evaluate expert relevance scores $\mathrm { \dot { \cal R } } _ { \mathrm { p } } ( { \bf x } ) , \mathrm { \dot { \cal R } } _ { \mathrm { l } } ( { \bf x } )$ via Eq. (8)   
Compatibility-Aware Routing   
Initialize $q _ { \mathfrak { p } } ( \mathbf { x } ; 0 ) \gets R _ { \mathfrak { p } } ( \mathbf { x } )$ and $q _ { 1 } ( { \bf x } ; 0 ) \gets R _ { 1 } ( { \bf x } )$   
for $s = 0$ to $S - 1$ do   
Update $q _ { \mathrm { p } } , q _ { \mathrm { l } }$ via Eq. (12)   
end for   
Set $q _ { \mathrm { p } } ( \mathbf { x } ) \gets q _ { \mathrm { p } } ( \mathbf { x } ; S )$ and $q _ { 1 } ( { \bf x } ) \gets q _ { 1 } ( { \bf x } ; S )$   
Expert mixture inference   
Generate yˆ according to Eq. (3), using $q _ { \mathrm { p } }$ and $q _ { \mathrm { l } }$ as the   
routing weights

## F Details of Comparison Methods

EWC (Kirkpatrick et al., 2017) assigns greater protection to parameters that are important for earlier tasks. Importance is estimated from Fisher information, and a weighted quadratic penalty discourages these parameters from moving away from their previously learned values during subsequent training. O-LoRA (Wang et al., 2023) reduces interference by regularizing newly learned low-rank subspaces to remain orthogonal to those associated with earlier tasks. Historical adapters are retained, and their low-rank factors are concatenated for inference, combining task-specific updates within a unified model.

MoELoRA (Chen et al., 2024) replaces a conventional LoRA adapter with several low-rank experts, whose number is configured according to the task sequence. A learned gating network assigns inputdependent weights to these experts, allowing specialized adaptations to contribute differently across tasks.

HiDe-LLaVA (Guo et al., 2025a) applies different adaptation strategies at different depths, motivated by layerwise CKA measurements. It expands taskspecific LoRA modules in the uppermost transformer layer and retrieves them by matching visual and textual anchors. In lower layers, adapters learned on successive tasks are fused to retain shared knowledge, avoiding the need to optimize a separate routing network.

ProgLoRA (Yu et al., 2025) grows an adapter pool by allocating a fresh LoRA block to every incoming task and keeping earlier blocks fixed. Its task-aware allocation mechanism combines relevant historical blocks according to inter-task similarity, supporting knowledge transfer during subsequent adaptation. A task recall mechanism further restricts updates to reduce forgetting. The method includes a static variant that assumes access to task identities at test time and a dynamic variant that infers the appropriate allocation when these identities are unavailable. SAME (Xie et al., 2026) addresses forgetting caused by changes in both expert assignments and expert parameters. Its spectral-aware routing decomposes router updates into orthogonal subspaces to balance adaptation with historical consistency. Curvature-aware scaling uses accumulated input covariance to limit changes in expert behavior on earlier tasks. An adaptive activation strategy additionally suspends selected experts during task training, reducing computation and interference.

DISCO (Guo et al., 2025b) maintains a dynamic cache of LoRA subspaces for federated continual instruction tuning. Instruction feature prototypes, termed identity tokens, determine how client updates are matched and aggregated within this cache. During inference, subspace selective activation weights each subspace according to the similarity between the input instruction and its identity token, suppressing contributions from unrelated knowledge.