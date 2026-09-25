# Learning Beter Reasoning for Generative Recommendation with Semantic IDs

Mengdan Zhu   
Emory University   
Atlanta, GA, USA   
mengdan.zhu@emory.edu   
Yao Zhao   
Microsoft   
Redmond, WA, USA   
yaozhao2@microsoft.com   
Yufan Zhao   
Microsoft   
Redmond, WA, USA   
yufzhao@microsoft.com   
Sridhar Iyer   
Microsoft   
Redmond, WA, USA   
sridhariyer@microsoft.com   
Tao Di   
Microsoft   
Redmond, WA, USA   
taodi@microsoft.com

Sophie Di Cornell University Ithaca, NY, USA szd5@cornell.edu

Yulan Yan   
Microsoft   
Redmond, WA, USA   
yulanyan@microsoft.com   
Liang Zhao   
Emory University   
Atlanta, GA, USA   
liang.zhao@emory.edu

## Abstract

Generative recommendation reformulates item retrieval as sequence generation, allowing a unified model to directly generate the next item from a user’s interaction history. Semantic IDs further make this paradigm efective and scalable by representing each item as discrete codes, enabling knowledge sharing among semantically related items. Recent studies introduce explicit reasoning before Semantic-ID generation, helping models summarize user interests and infer possible preference transitions. However, reasoning is not inherently beneficial: Inaccurate or uninformative reasoning may mislead subsequent item generation and ultimately degrade recommendation performance. This raises a central challenge: how can a recommender select and learn efective reasoning traces and progressively evolve toward better reasoning from its own generations? In this work, we propose Evo-Rec, a three-stage framework for learning better reasoning and further enhancing it through rein forcement learning. First, we align Semantic IDs with their textual and behavioral contexts, enabling the model to understand and generate item identifiers. Second, we sample multiple candidate reasoning traces and retain those that improve the prediction of the ground-truth item, providing a stronger reasoning initialization through supervised fine-tuning. Third, we further optimize the reasoning policy through reinforcement learning with catalog constrained item generation and ranking-aware recommendation feedback. Experiments on three Amazon Review benchmarks show that Evo-Rec consistently outperforms discriminative, generative, and reasoning-enhanced recommenders across all evaluation metrics. On the challenging Video Games dataset, Evo-Rec improves Recall@5 from 0.0710 to 0.0847 and NDCG@10 from 0.0563 to 0.0746 over the strongest baseline, corresponding to relative gains of 19.3% and 32.5%, respectively. These results demonstrate the efectiveness of our framework in learning better reasoning for SID-based generative recommendation.<sup>1</sup>

CCS Concepts • Information systems → Recommender systems.

Keywords   
Generative Recommendation; Semantic IDs; Reasoning; Reinforce  
ment Learning; Large Language Models

## 1 Introduction

Sequential recommendation aims to predict the next item given a user’s chronologically ordered interaction history. For a decade the dominant formulation has been discriminative: encode the history into a user vector and score it against a table of item embeddings [4, 7, 15]. Because every item is mutually independent index in this formulation, the embedding table grows with the item catalog, nothing is shared between semantically related items, and cold items are hard to place.

Generative recommendation provides an alternative formulation by casting recommendation as autoregressive sequence generation [2, 13, 23]. In particular, Semantic-ID (SID) based methods quantize item semantics into short sequences of discrete codes, allowing the recommender to generate the identifier of the next item token by token [5, 13] so that related items share code prefixes and generalization no longer depends on interaction counts. Subsequent work has sharpened both ends of this pipeline, learning tokenizers that inject collaborative signal into the codes [16, 18, 20], adapting large language models to the code vocabulary through SID–language alignment tasks [6, 25], and scaling end-to-end gen erative recommenders in industrial and open settings [1, 9]. Once retrieval is posed as sequence generation over a language model, recommendation inherits the machinery of language models, and most notably their capacity to reason before they answer [19].

This observation has produced a fast-growing line of reasoningenhanced recommenders. One branch reasons in a latent space, deepening the computation applied to a history without emitting any text [14, 24]; another reasons explicitly in natural language, jointly learning to deliberate and to recommend [8, 21], optimizing a reasoning model against a frozen downstream retriever [12], or verbalizing user interests directly over Semantic IDs [3]. Despite this progress, existing reasoning-enhanced recommenders largely focus on enabling reasoning, while paying less attention to the quality of the reasoning trajectories. This distinction is important because reasoning is not inherently beneficial for recommendation. For the same interaction history, diferent reasoning traces may summarize diferent aspects of user preference, infer diferent transitions, or even lead to conflicting recommendation decisions. A reasoning trace can therefore be semantically plausible while being misleading for next-item prediction. Training indiscriminately on such traces may not only introduce noisy supervision, but also reinforce spurious reasoning patterns that misguide subsequent reinforce ment learning. Similar observations in general LLM reasoning have motivated selection, verification, and self-refinement of reasoning trajectories [11, 17, 22], yet how to identify and improve useful reasoning remains underexplored in generative recommendation.

This limitation persists during reinforcement learning. Existing approaches typically optimize reasoning through feedback derived from the final outcome, such as exact SID matching. Such outcome signals verify whether a prediction is correct, but provide only limited information for improving recommendation rank. In a full-catalog ranking problem, for example, reasoning that moves the ground-truth item from rank 100 to rank 2 is substantially more useful than reasoning that leaves it outside the retrieved set, even though a top-1 exact-match reward assigns both the same reward. Moreover, when reasoning and item generation are optimized as a single trajectory, the quality of the reasoning itself becomes entangled with stochasticity in SID generation. Consequently, the central problem remains underexplored: how can a generative recommender identify better reasoning trajectories and continuously improve its reasoning policy from its own generation?

We approach this problem from the perspective that reasoning can progressively self-evolve through feedback from the recommendation distribution it induces. Given a history h and a candidate reasoning trace z, useful reasoning should increase the likelihood of the ground-truth item relative to predicting from the history alone, and ultimately move that item toward the top ofthe catalog ranking. This perspective provides two complementary forms of supervision. Before reinforcement learning, multiple candidate reasoning traces can be compared according to their predictive utility, allowing the model to learn from stronger reasoning examples rather than arbitrary teacher generations. During reinforcement learning, the model can generate its own reasoning trajectories and receive ranking-aware feedback according to how each trajectory reorganizes the downstream item ranking. Together, these two mechanisms allow recommendation reasoning to progressively evolve from selected supervision toward task-aligned self-improvement.

Based on this insight, we propose Evo-Rec, a three-stage framework for learning and evolving reasoning in Semantic-ID based generative recommendation. In the first stage, we align newly introduced SID tokens with item semantics and user behavior through multi-task SID–language alignment, providing the representation foundation required for reasoning over itemic tokens. In the sec ond stage, we introduce Best-of-� Rejection Sampling SFT. For each interaction history, we sample multiple candidate reasoning traces and measure the incremental predictive utility of each trace by the change in the ground-truth SID likelihood relative to historyonly prediction. We retain the highest-utility trace only when it provides a positive improvement and use the selected trajectories to warm up the reasoning policy. Unlike standard reasoning SFT that treats generated rationales as equally useful, this stage directly optimizes which reasoning trajectories the model learns from. The third stage enables the reasoning policy to further evolve from its own generations. For each history, Evo-Rec samples multiple reasoning trajectories from the current policy and, conditioned on each trajectory, performs trie-constrained beam search over the item catalog. We use the rank of the ground-truth SID in the resulting beam to construct an NDCG-based reward and optimize the reasoning policy with GRPO.

Extensive experiments on three Amazon Review benchmarks demonstrate the efectiveness of Evo-Rec. It consistently outperforms discriminative, generative, and reasoning-enhanced recommendation baselines across all datasets and evaluation metrics. The gains are particularly pronounced on the more challenging Games dataset, where Evo-Rec improves Recall@5 from 0.0710 to 0.0847 and NDCG@10 from 0.0563 to 0.0746 over the strongest baseline, corresponding to relative improvements of 19.3% and 32.5%, respectively. Ablation studies further show that the gains arise primarily from ranking reasoning candidates by predictive utility rather than merely rejecting negative candidates, while ranking-aware NDCG feedback substantially outperforms coarse outcome rewards during reinforcement learning.

Our main contributions are summarized as follows:

• We identify reasoning quality as a central bottleneck in reasoningenhanced generative recommendation and propose Best-of-� Rejection Sampling SFT in terms of its predictive utility for the ground-truth item. This allows the model to learn from better reasoning supervision rather than treating all generated rationales as equally useful.

• We propose Evo-Rec, a progressive reasoning optimization framework that combines predictive-utility selection with rankingaware reinforcement learning. The former provides a stronger reasoning initialization by selecting useful reasoning supervision, while the latter allows the model to further evolve its reasoning policy from self-generated trajectories.

• We introduce a catalog-constrained, ranking-aware RL objective that evaluates each reasoning trajectory through the full recommendation ranking it induces and directly optimizes the reasoning tokens. Extensive experiments and analyses across three domains demonstrate consistent improvements over existing methods and reveal how reasoning selection and rankingaware feedback contribute to better recommendation.

## 2 Related Work

## 2.1 From Discriminative to Generative Recommendation

Sequential recommendation predicts the next item from a chronologically ordered interaction history. Early approaches are discrim inative: they compress the history into a single user representation and score it against an item embedding table. GRU4Rec models session sequences with recurrent networks and a ranking loss tailored to top-� gains [4]; Caser applies horizontal and vertical convolutional filters over the embedding matrix of recent interactions to capture point-level and union-level patterns [15]; and SASRec replaces recurrence with self-attention so that the relevance of every past interaction is weighted adaptively [7]. These models are strong and eficient, but they treat items as mutually independent identifiers. Consequently, the parameter count of the embedding table grows linearly with the catalog, no representation is shared between semantically related items, and items with few or no interactions are dificult to place in the embedding space, which limits generalization to the long tail.

Generative recommendation reformulates the task as sequence generation and removes the dependence on an explicit item embedding table. HSTU treats user behavior as a sequential transduction problem and demonstrates that generative recommenders follow favorable scaling behavior at trillion-parameter scale [23]. TIGER makes the item vocabulary itself semantic: it quantizes content embeddings with RQ-VAE into a SID, a short sequence of discrete codes, and autoregressively generates the SID of the next item, so that semantically similar items share code prefixes and generalization to unseen items becomes possible [13]. Later work targets the two weak points of this pipeline. On the tokenizer side, LETTER regularizes item tokenization with collaborative signals and a diversity term, addressing the mismatch between a tokenizer trained for content reconstruction and a recommender trained for pref erence prediction [16]. On the model side, LC-Rec adapts a large language model to the SID vocabulary through a series ofalignment tasks that integrate collaborative semantics into the code space [25], and complementary studies benchmark how well LLMs serve as semantic encoders for item representation [6]. Beyond individual components, OneRec [1] and MiniOneRec [9] study end-to-end generative recommenders and the scaling behavior of SID-based generation in industrial and open settings.

## 2.2 Reasoning-Enhanced Recommendation

Chain-of-thought prompting shows that generating intermediate steps before an answer substantially improves reasoning in large language models [19], and recommendation research has adopted this idea along a clear trajectory [27]. A group reasons in a latent space: ReaRec performs multi-step latent reasoning inside a sequential recommender at inference time, deepening the computation applied to a history without emitting any text [14]. Latent reasoning is eficient [10, 26], but its intermediate states are neither inspectable nor verifiable, so the quality of the reasoning itself cannot be measured or supervised. A second group therefore reasons explicitly in natural language. R<sup>2</sup>ec learns a unified architecture in which reasoning and recommendation are optimized together [21], Rec-R1 optimizes an LLM against feedback from a fixed downstream recommender [12], and SIDReasoner reasons explicitly over Semantic IDs and shows that reasoning transfers to generative recommendation [3]. However, these approaches fail to identify which reasoning trajectories are truly beneficial for downstream recommendation. To address this limitation, our Evo-Rec selects high-utility reasoning trajectories for supervised learning and further refines the policy with ranking-aware reinforcement learning, enabling progressively better recommendation reasoning.

## 3 Methodology

In this section, we present our Evo-Rec framework for progressively learning better reasoning for Semantic-ID based generative recommendation, as illustrated in Figure 1. We first formulate reasoningenhanced next-item generation and describe how Semantic IDs are grounded in language semantics and user behavior. We then introduce Best-of-� Rejection Sampling SFT, which selects reasoning trajectories according to their predictive utility for the target item. Finally, we present ranking-aware reinforcement learning, where self-generated reasoning trajectories are evaluated by the catalog rankings they induce and optimized with NDCG-based feedback, enabling the reasoning policy to progressively evolve toward more efective recommendation reasoning.

## 3.1 Problem Formulation

Let I denote an item catalog. A user interaction trajectory is a temporally ordered sequence $S = ( i _ { 1 } , \dots , i _ { T } , i _ { T + 1 } ) , i _ { t } \in \mathcal { I }$ . Given the history $H \ = \ \left( i _ { 1 } , \ldots , i _ { T } \right)$ , the goal of sequential recommendation is to predict the next item $i _ { T + 1 }$ . We cast this task as autoregressive generation by representing every item with a fixedlength Semantic ID. Formally, an item tokenizer defines a mapping $\phi : \mathcal { T }  C ^ { L } , \phi ( i ) = ( c _ { i } ^ { 1 } , \cdot \cdot \cdot , c _ { i } ^ { L } )$ , where C is a discrete code vocabulary and � is the number of code levels. Accordingly, we have the interaction history $\mathbf { h } = [ \phi ( i _ { 1 } ) ; \dots ; \phi ( i _ { T } ) ]$ , and the target SID $\mathbf { y } = \phi ( i _ { T + 1 } )$ . Instead of directly modeling $p _ { \theta } ( \mathbf { y } \mid \mathbf { h } )$ , we introduce an intermediate reasoning sequence $\mathbf { z } = \left( z _ { 1 } , \ldots , z _ { M } \right)$ . We model the joint conditional distribution of the reasoning sequence and the target SID as $p _ { \theta } ( \mathbf { z } , \mathbf { y } \mid \mathbf { h } ) = p _ { \theta } ( \mathbf { z } \mid \mathbf { h } ) p _ { \theta } ( \mathbf { y }$ | h, z). Thus, we aim to optimize the conditional marginal likelihood of the target SID: $p _ { \theta } ( \mathbf { y } \mid \mathbf { h } ) = \sum _ { \mathbf { z } } \mathcal { p } _ { \theta } ( \mathbf { z } \mid \mathbf { h } ) p _ { \theta } ( \mathbf { y } \mid \mathbf { h } , \mathbf { z } )$

## 3.2 Aligning Language and Semantic IDs in LLMs

Semantic IDs are newly introduced item tokens and therefore have no inherent meaning to a pretrained language model. Following prior SID–language alignment approaches [3, 25], we first align these itemic tokens with natural-language semantics and recom mendation behavior before introducing explicit reasoning. For each item �, we encode its textual metadata into a continuous representation and apply a residual-quantization VAE (RQ-VAE) [13] with � codebooks to obtain $\phi ( i ) = ( c _ { i } ^ { 1 } , . . . , c _ { i } ^ { L } )$ ). The resulting SID tokens are appended to the LLM vocabulary with randomly initialized embeddings and fine-tune the LLM backbone on a mixture of SID– language tasks. This stage provides the semantic and behavioral grounding required by the subsequent reasoning stages.

The first component consists of next-item prediction tasks under diferent representations of the interaction history and target. Specifically, we include SID history → SID, SID history → title, title history → title, and title history → SID. These complementary views expose the model to the same behavioral transition through both natural-language and itemic representations, encouraging knowledge learned in the language space to transfer to the SID space. We further include bidirectional title–SID translation. Beyond these structured prediction tasks, we enrich SID grounding with teacher-generated SID–text interleaved data. At the item level, SID tokens are embedded in natural-language descriptions. At the sequence level, SID interaction histories are interleaved with descriptions of the corresponding user interests and behavioral transitions. These examples expose SID tokens to substantially richer linguistic contexts to help the model associate itemic tokens with semantic concepts useful for subsequent reasoning. In addition, we mix general-domain reasoning examples into the alignment corpus to preserve the pretrained model’s general language and reasoning capabilities during recommendation adaptation. All data are optimized under a unified autoregressive objective:

![](images/1cab9c1a922558966cc91f8e37cdbe88aa2c198e36ff42f287db4f56eb456eff.jpg)  
Figure 1: The overview of our proposed Evo-Rec framework.

$$
\mathcal { L } _ { \mathrm { a l i g n } } = - \mathbb { E } _ { \mathbf { x } \sim \mathcal { D } _ { \mathrm { a l i g n } } } \left[ \sum _ { t = 1 } ^ { | \mathbf { x } | } m _ { t } ( \mathbf { x } ) \log p _ { \theta } ( x _ { t } \mid \mathbf { x } _ { < t } ) \right] ,\tag{1}
$$

where $m _ { t } ( \mathbf { x } ) \in \{ 0 , 1 \}$ specifies whether token $x _ { t }$ contributes to the loss. For recommendation prediction, translation, and generalreasoning examples, we mask the prompt and optimize only the response tokens. For SID–text interleaved examples, we instead apply full-sequence language modeling, i.e., $m _ { t } ( { \bf x } ) = 1$ for all nonpadding tokens, allowing SID embeddings to be learned from every surrounding linguistic context. After this stage, the model is able to interpret and generate SIDs in both semantic and behavioral contexts, providing the representation foundation required for recommendation reasoning. Importantly, Stage 1 does not optimize a particular chain-of-thought trajectory. Rather, it establishes the SID–language grounding on top of which Stage 2 can activate reasoning and learn from selected better reasoning trajectories.

## 3.3 Best-of-� Rejection Sampling Supervised Fine-Tuning

The quality of the CoT reasoning trajectories learned during supervised fine-tuning is critical for subsequent reinforcement learning. The quality of sampled CoTs can vary substantially: a rationale may be plausible in language yet provide little evidence for the correct recommendation. Directly fine-tuning on such rationales introduces noisy supervision. We therefore propose Best-of-� Rejection Sampling SFT, which selects CoTs according to their predictive utility for the target SID and rejects candidates that do not improve upon direct history-based prediction.

Predictive-utility scoring. For each training instance $( \mathbf { h } _ { n } , \mathbf { y } _ { n } ^ { * } )$ , we sample � explicit CoTs $\mathcal { Z } _ { n } = \{ \mathbf { z } _ { n , k } \} _ { k = 1 } ^ { K }$ from a teacher LLM. We then use a frozen phase-2 model $\hbar \hat { \theta }$ finetuned on a single-sample CoT corpus as a scorer to measure the contribution of each candidate to target prediction:

$$
\Delta _ { n , k } = \log p _ { \bar { \theta } } ( \mathbf { y } _ { n } ^ { * } \mid \mathbf { h } _ { n } , \mathbf { z } _ { n , k } ) - \log p _ { \bar { \theta } } ( \mathbf { y } _ { n } ^ { * } \mid \mathbf { h } _ { n } ) .\tag{2}
$$

Both terms score only the target SID tokens under the same interaction context and difer solely by the presence of ${ \bf z } _ { n , k }$ . Consequently, $\Delta _ { n , k }$ isolates the incremental predictive utility of the candidate CoT.

Best-of-� selection and rejection. We retain the highest-utility candidate only if the predictive utility is positive:

$$
k _ { n } ^ { * } = \arg \operatorname* { m a x } _ { 1 \leq k \leq K } \Delta _ { n , k } , \qquad { \widehat { \mathbf { z } } } _ { n } = \mathbf { z } _ { n , k _ { n } ^ { * } } { \mathrm { ~ i f ~ } } \Delta _ { n , k _ { n } ^ { * } } > 0 .\tag{3}
$$

Since the history-only term in Eq. (2) is shared by all candidates, this selection is equivalent to maximizing the rationale-conditioned likelihood of the target SID.

Supervised fine-tuning. Let $\mathcal { A } = \{ n : \Delta _ { n , k _ { n } ^ { * } } > 0 \}$ denote the accepted instances. We fine-tune the model on the selected reasoning– target pairs using the joint autoregressive objective:

$$
\mathcal { L } _ { \mathrm { B o N } } ( \theta ) = - \frac { 1 } { | \mathcal { R } | } \sum _ { n \in \mathcal { R } } [ \log p _ { \theta } ( \widehat { \mathbf { z } } _ { n } | \mathbf { h } _ { n } ) + \log p _ { \theta } ( \mathbf { y } _ { n } ^ { * } | \mathbf { h } _ { n } , \widehat { \mathbf { z } } _ { n } ) ]\tag{4}
$$

This objective trains the model to generate an outcome-supporting CoT from the interaction history and subsequently predict the target SID conditioned on that reasoning.

## 3.4 Reasoning Optimization with Constrained SID Ranking

For each history $\mathbf { h } _ { n } .$ , the current policy samples a group of � reasoning trajectories $\{ \mathbf { z } _ { n , g } \} _ { g = : } ^ { G }$ . Conditioned on each trajectory, we construct a ranked list of valid items with trie-constrained beam search and use the position of the target SID to score the trajectory.

Constrained beam search. Since a valid recommendation must correspond to an item in the catalog, unconstrained decoding wastes probability mass on invalid SID sequences. We build a prefix trie from all catalog SIDs and restrict each decoding step to valid continuations of the current prefix. Conditioned on a reasoning trajectory $\scriptstyle \mathbf { z } _ { n , g } ,$ the score of a complete SID $\mathbf { y } = ( y ^ { 1 } , \ldots , y ^ { L } )$ is

$$
s _ { \theta } ( \mathbf { y } \mid \mathbf { h } _ { n } , \mathbf { z } _ { n , g } ) = \sum _ { \ell = 1 } ^ { L } \log \widetilde { p } _ { \theta } ( y ^ { \ell } \mid \mathbf { h } _ { n } , \mathbf { z } _ { n , g } , \mathbf { y } _ { < \ell } ) ,\tag{5}
$$

where ${ \widetilde { p } } _ { \theta }$ is the model distribution after masking and renormalizing over legal trie children. At every level, beam search keeps the � highest-scoring prefixes. The resulting beam $\mathcal { R } _ { n , g } = ( \widehat { \mathbf { y } } _ { n , g } ^ { ( 1 ) } , \ldots , \widehat { \mathbf { y } } _ { n , g } ^ { ( B ) } )$ is an ordered list of distinct and catalog-valid items.

Exact-match and NDCG reward. Let $r _ { n , g }$ denote the rank of the ground-truth SID ${ \bf y } _ { n } ^ { * }$ in $\mathcal { R } _ { n , g } ,$ , with $r _ { n , g } = \infty$ if it is absent. The top-1 exact-match reward and the ranking-aware reward are

$$
R _ { n , g } ^ { \mathrm { E M } } = \mathbb { I } [ r _ { n , g } = 1 ] , \qquad R _ { \mathrm { N D C G } } = \underbrace { \mathbb { I } [ r _ { n , g } \leq 1 0 ] } _ { \mathrm { r e t r i e v a l } } \cdot \underbrace { \frac { 1 } { \log _ { 2 } ( r _ { n , g } + 1 ) } } _ { \mathrm { r a n k i n g } } .\tag{6}
$$

Exact match provides a verifiable but sparse signal: it treats a target ranked second the same as a target absent from the beam. NDCG preserves the exact-match reward of one at rank 1 while assigning progressively smaller credit to lower ranks. We therefore use NDCG@10 as the scalar training reward.

For each history, rewards are normalized across the � trajectories and optimized with GRPO. Consequently, the model learns to favor reasoning paths that rank the ground-truth item more highly.

## 4 Experiments

## 4.1 Experimental Setup

4.1.1 Datasets and Evaluation. We evaluate our framework on three public benchmark datasets: Video Games (Games), Ofice Products (Ofice), and Industrial and Scientific (Industrial) from the Amazon Review [6]. We apply 5-core filtering and sort each user’s interactions chronologically. We truncate each user’s historical interaction sequence with a sliding window whose maximum history length is 10. For each user, we adopt the timestamp split: the earliest 80% of interactions are used for training, the subsequent 10% for validation, and the latest 10% for testing. This temporal protocol prevents future interactions from entering the training history. Dataset statistics are summarized in Table 1.

For evaluation, we utilize two widely used metrics: Recall@K (R@K) and NDCG@K (N@K) with cutof K set to 5 and 10. We follow the full-item ranking setting, where the ground-truth next item is ranked against the entire item catalog rather than a sampled subset of negative items.

Table 1: Dataset statistics after preprocessing.
<table><tr><td>Dataset</td><td>#Items</td><td>#Train</td><td>#Val.</td><td>#Test</td></tr><tr><td>Games</td><td>3,858</td><td>49,133</td><td>6,142</td><td>6,142</td></tr><tr><td>Office</td><td>3,459</td><td>38,924</td><td>4,866</td><td>4,866</td></tr><tr><td>Industrial</td><td>3,686</td><td>36,259</td><td>4,533</td><td>4,533</td></tr></table>

4.1.2 Baselines. We compare against three families of methods: (1) Traditional discriminative sequential recommenders include GRU4Rec [4], Caser [15], and SASRec [7]. (2) Classic generative recommenders include TIGER [13], HSTU [23], LETTER [16], and LC-Rec [25]. (3) Reasoning-enhanced recommenders include ReaRec [14], R<sup>2</sup>ec [21], and SIDReasoner [3]. Baseline results are quoted from [3], while our method is evaluated using the same data split and evaluation protocol.

4.1.3 Implementation Details. We use Qwen3-1.7B as our LLM backbone and perform full fine-tuning throughout all stages. Every item is encoded by � = 3 semantic tokens, and the corresponding SID tokens are appended to the tokenizer vocabulary with randomly initialized embeddings. Since the three datasets have independent SID codebooks, one model is trained per dataset and no parameter is shared across domains. All stages are optimized with AdamW on 80GB A100 GPUs. In Stage-1, the backbone is aligned to the SID vocabulary by fine-tuning on a mixture of SID–language tasks. We use a learning rate of $2 \times 1 0 ^ { - 5 }$ , for up to 5 epochs. For Best-of-� rejection sampling SFT, we synthesize $N = 5$ candidate CoTs per training instance by querying the teacher LLM (GPT-5.4) five times independently with the same prompt. The selected instances are then trained for a single epoch with a learning rate of $1 \times 1 0 ^ { - 5 }$ with linear decay and 10 warm-up steps, a batch size of 72. The reinforcement learning stage implements GRPO on verl with a vLLM rollout engine over 8\*A100 GPUs with batch size of 256. Each history samples � = 16 reasoning trajectories. Conditioned on each trajectory, the SID prefix trie built from the catalog restricts every decoding step, and beam search with � = 10 returns an ordered list of ten valid items whose target rank defines the NDCG@10 reward of Eq. (6). We set the learning rate to $7 \times 1 0 ^ { - 7 }$ and training runs for at most 15 epochs. At inference time, the model first decodes its reasoning, and the SID is then produced by the same trie-constrained beam search with beam size 10 over the full item catalog.

## 4.2 Main Results

Table 2 reports the overall performance on the three Amazon Review datasets. Our method consistently outperforms all baselines across every evaluation metric and dataset. These results demonstrate that improving the quality of reasoning trajectories can consistently benefit semantic-ID based generative recommendation across domains. Notably, the improvements are generally more pronounced on NDCG than on Recall. For example, on Games, compared with the strongest baseline, N@5 and N@10 improve by 41.1% and 32.5%, while R@5 and R@10 improve by 19.3% and 11.4%. This indicates that our method does not merely increase the probability that the target item appears in the retrieved set, but more efectively promotes it toward higher positions in the ranking.

Table 2: The overall performance of diferent methods on the three datasets. The best results are highlighted in Bold.
<table><tr><td rowspan="2">Models</td><td colspan="4">Games</td><td colspan="4">Office</td><td colspan="4">Industrial</td></tr><tr><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td><td>R@5</td><td>N@5</td><td>R@10</td><td>N@10</td></tr><tr><td colspan="10">Traditional discriminative sequential recommenders</td><td></td><td></td><td></td></tr><tr><td>Caser</td><td>0.0376</td><td>0.0241</td><td>0.0659</td><td>0.0332</td><td>0.0880</td><td>0.0663</td><td>0.1114</td><td>0.0738</td><td>0.0664</td><td>0.0528</td><td>0.0852</td><td>0.0588</td></tr><tr><td>GRU4Rec</td><td>0.0329</td><td>0.0219</td><td>0.0599</td><td>0.0305</td><td>0.0682</td><td>0.0480</td><td>0.0974</td><td>0.0574</td><td>0.0788</td><td>0.0578</td><td>0.1030</td><td>0.0649</td></tr><tr><td>SASRec</td><td>0.0501</td><td>0.0345</td><td>0.0723</td><td>0.0416</td><td>0.1019</td><td>0.0824</td><td>0.1167</td><td>0.0871</td><td>0.0807</td><td>0.0647</td><td>0.0964</td><td>0.0697</td></tr><tr><td colspan="10">Classic generative recommenders</td><td colspan="3"></td></tr><tr><td>TIGER</td><td>0.0489</td><td>0.0300</td><td>0.0763</td><td>0.0402</td><td>0.1270</td><td>0.1037</td><td>0.1429</td><td>0.1121</td><td>0.1003</td><td>0.0823</td><td>0.1325</td><td>0.0924</td></tr><tr><td>HSTU</td><td>0.0539</td><td>0.0396</td><td>0.0746</td><td>0.0462</td><td>0.1204</td><td>0.1069</td><td>0.1323</td><td>0.1107</td><td>0.1008</td><td>0.0898</td><td>0.1138</td><td>0.0940</td></tr><tr><td>LETTER</td><td>0.0445</td><td>0.0294</td><td>0.0709</td><td>0.0378</td><td>0.1315</td><td>0.1074</td><td>0.1520</td><td>0.1139</td><td>0.1080</td><td>0.0850</td><td>0.1389</td><td>0.0950</td></tr><tr><td>LC-Rec</td><td>0.0441</td><td>0.0274</td><td>0.0876</td><td>0.0412</td><td>0.0964</td><td>0.0699</td><td>0.1487</td><td>0.0867</td><td>0.0805</td><td>0.0520</td><td>0.1330</td><td>0.0687</td></tr><tr><td colspan="10">Reasoning-enhanced recommenders</td><td colspan="3"></td></tr><tr><td>ReaRec</td><td>0.0568</td><td>0.0381</td><td>0.0843</td><td>0.0470</td><td>0.1173</td><td>0.0988</td><td>0.1385</td><td>0.1057</td><td>0.0973</td><td>0.0796</td><td>0.1205</td><td>0.0870</td></tr><tr><td>R²ec</td><td>0.0655</td><td>0.0399</td><td>0.0931</td><td>0.0525</td><td>0.1147</td><td>0.0894</td><td>0.1486</td><td>0.1004</td><td>0.0880</td><td>0.0774</td><td>0.1253</td><td>0.0774</td></tr><tr><td>SIDReasoner</td><td>0.0710</td><td>0.0460</td><td>0.1031</td><td>0.0563</td><td>0.1373</td><td>0.1119</td><td>0.1648</td><td>0.1208</td><td>0.1109</td><td>0.0905</td><td>0.1438</td><td>0.1010</td></tr><tr><td>Evo-Rec</td><td>0.0847</td><td>0.0649</td><td>0.1149</td><td>0.0746</td><td>0.1476</td><td>0.1223</td><td>0.1710</td><td>0.1299</td><td>0.1167</td><td>0.0976</td><td>0.1476</td><td>0.1076</td></tr></table>

This behavior is consistent with our Stage-3 reinforcement learning objective, which directly rewards reasoning trajectories according to the rank of the ground-truth SID under constrained beam search.

The largest gains are observed on Games, where our method raises R@5 from 0.0710 to 0.0847 and N@10 from 0.0563 to 0.0746. We attribute this improvement to the complementary roles of our two reasoning optimization stages: Stage 2 uses best-of-� selection to provide a stronger reasoning initialization, while Stage 3 further aligns these reasoning trajectories with the downstream ranking objective. Together, the two stages progressively transform higher-quality reasoning supervision into improved recommendation performance.

Games also exhibits substantially lower thinking baseline performance than Ofice and Industrial in Table 7, suggesting that it represents the more challenging recommendation setting and therefore provides greater RL optimization headroom. The particularly large gains on Games suggest that ranking-aware RL can be espe cially efective when the initial recommendation policy leaves more room for refinement. Moreover, the consistent improvements on Ofice and Industrial further suggest that the proposed framework is not specific to a single domain, but provides a general mechanism of learning better reasoning for recommendation.

## 4.3 Ablation Study

## 4.3.1 Best-of-� Rejection Sampling.

CoT Selection Strategy. We first isolate the contribution of the CoT selection strategy, which determines which reasoning trace is retained. At a fixed sampling budget of � = 5, all variants draw the same five candidate CoTs per training instance from the same teacher and difer only in the selection rule: random selection keeps a uniformly drawn from � candidates, rejection sampling keeps the first candidate whose predictive utility is positive $\left( \Delta _ { n , k } \ > \ 0 \right)$ , and best-of-� rejection sampling keeps the highest positive utility candidate arg max� $\Delta _ { n , k }$ as in Eq. (3). Each variant is then trained with the identical Stage-3 recipe, so any diference is attributable to the selection rule alone.

Table 3: Efect of the CoT selection rule on Games at a fixed sampling budget � = 5. All variants share the same candidate CoTs and the same training pipeline, and difer only in which candidate is retained.
<table><tr><td>Selection rule</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td>Random selection</td><td>0.0241</td><td>0.0694</td><td>0.1043</td><td>0.0466</td><td>0.0587</td></tr><tr><td>Rejection sampling</td><td>0.0248</td><td>0.0696</td><td>0.1051</td><td>0.0489</td><td>0.0594</td></tr><tr><td>Best-of-N</td><td>0.0443</td><td>0.0847</td><td>0.1149</td><td>0.0649</td><td>0.0746</td></tr></table>

Table 4: Efect of the sampling budget � on Games under the best-of-� rejection sampling rule. � = 1 corresponds to training on a single sampled CoT without selection.
<table><tr><td>Budget</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td>N=1</td><td>0.0247</td><td>0.0690</td><td>0.1013</td><td>0.0467</td><td>0.0571</td></tr><tr><td>N=3</td><td>0.0277</td><td>0.0755</td><td>0.1055</td><td>0.0518</td><td>0.0614</td></tr><tr><td>N=5</td><td>0.0443</td><td>0.0847</td><td>0.1149</td><td>0.0649</td><td>0.0746</td></tr></table>

As shown in Table 3 reports the results, Best-of-� rejection sampling improves N@10 by 27.1% over random selection and by 37.1% over rejection sampling, and the gain is most pronounced at the top of the ranking R@1, a relative gain of 83.8%. Two observations follow. First, random selection pays the full � sampling cost yet performs on par with training on a single trace (Table 4, � = 1), which shows that the benefit comes from scoring the candidates rather than from sampling more of them. Second, rejection sampling is a weak variant: accepting the first positive-utility candidate uses $\Delta _ { n , k }$ only as a binary filter and discards the ranking information it carries, and since 94.0% of all candidates are already positive, this filter is close to a random draw. Ranking candidates by predictive utility is therefore the component that matters.

Table 5: Efect of the sampling strategy in RL on Games. The number of samples and the beam size are both 10.
<table><tr><td>Sampling strategy</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td>Constrained sampling</td><td>0.0283</td><td>0.0669</td><td>0.0943</td><td>0.0483</td><td>0.0570</td></tr><tr><td>Constrained beam search</td><td>0.0443</td><td>0.0847</td><td>0.1149</td><td>0.0649</td><td>0.0746</td></tr></table>

Table 6: Efect of diferent reward designs on Games.
<table><tr><td>Reward</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td>Prefix Match</td><td>0.0425</td><td>0.0791</td><td>0.1073</td><td>0.0608</td><td>0.0699</td></tr><tr><td>Exact Match</td><td>0.0264</td><td>0.0650</td><td>0.0958</td><td>0.0461</td><td>0.0561</td></tr><tr><td>NDCG + Recall</td><td>0.0264</td><td>0.0633</td><td>0.0912</td><td>0.0449</td><td>0.0538</td></tr><tr><td>NDCG Reward</td><td>0.0443</td><td>0.0847</td><td>0.1149</td><td>0.0649</td><td>0.0746</td></tr></table>

Sampling Budget. We next vary the number of sampled CoTs $N \in \{ 1 , 3 , 5 \}$ while keeping the best-of-� rejection sampling rule fixed; $N \ = \ 1$ degenerates to training on a single sampled CoT without selection. As shown in Table 4, performance increases monotonically in � on every metric, with N@10 improving from 0.0571 at � = 1 to 0.0614 at � = 3 and 0.0746 at � = 5. These consistent gains demonstrate that our predictive-utility scoring efectively identifies increasingly useful reasoning traces from a larger candidate pool. Moreover, the continued improvement up to � = 5 suggests that the gains may not have saturated, leaving room for further improvements with larger sampling budgets.

4.3.2 Sampling Strategy. We compare the trie-constrained beam search of Section 3.4, which returns the � highest-scoring items as an ordered list, against constrained sampling, which pairs each trajectory with � candidates drawn independently from the renormalized distribution $\widetilde { p } _ { \theta }$ of Eq. (5); we set � = 10 and keep the Stage-2 model the same.

In this setting, Table 5 shows that beam search wins on every metric, by 30.9% on N@10 and 56.5% on R@1. Both decoders share the same trie, so all candidates are catalog-valid either way and the gap is not output validity but of what the reward measures. Independent draws form an unordered multiset, leaving the rank $r _ { n , g }$ ofEq. (6) undefined and reducing the reward to a hit indicator, so the policy learns to raise the marginal probability of the target rather than to rank it above its competitors. Ten unguided draws also cover little of the efective prefixes of the large Games trie, leaving most groups with zero reward. We therefore adopt constrained beam search as our sampling strategy.

4.3.3 Reward Design. We investigate how diferent rewards af fect RL optimization. For each training instance, constrained beam search produces a ranked list of $K = 1 0$ valid SID candidates. We compare four reward designs:

• Exact Match. We use the exact-match reward defined in Eq. 6, which gives a positive reward only when the complete target SID appears in the top-10 beam.

• Prefix Match. To provide denser supervision, we additionally consider a hierarchical prefix reward. Let � denote the number of consecutive SIDs matching the ground truth and � denote the total number of SIDs. We assign $\begin{array} { r } { R _ { \mathrm { p r e f i x } } ( m ) = \frac { 1 } { 2 L - m } } \end{array}$

• NDCG Reward. We use the rank-sensitive NDCG reward in Eq. 6, which assigns a larger reward when the ground-truth SID is ranked higher in the top-10 beam.

• NDCG + Recall. We also evaluate a hybrid reward that averages the binary Recall@10 signal and the NDCG@10 signal: $\begin{array} { r } { R _ { \mathrm { N D C G + R e c a l l } } = \left( 1 + \frac { 1 } { \log _ { 2 } \left( r _ { n , g } + 1 \right) } \right) / 2 , } \end{array}$ , when the ground-truth SID appears in the top-10 beam, and 0 otherwise.

As shown in Table 6, the pure NDCG reward performs best across all evaluation metrics. Exact match performs substantially worse because it treats all top-10 hits identically and therefore provides no signal for improving their relative ordering. Adding partial prefix match produces a denser learning signal, but partial SID agreement does not necessarily correspond to ranking the correct item higher. In contrast, the NDCG reward naturally captures both retrieval and ranking: it assigns zero reward when the ground-truth SID is not retrieved in the top-10 beam, while discounting successful retrievals according to their rank. Adding an additional Recall@10 term therefore overemphasizes retrieval and weakens the relative ranking signal, which may explain the degraded performance of NDCG+Recall.

## 4.4 Discussion

4.4.1 Reasoning length. Figure 2 shows how reasoning behavior evolves during Stage-3 reinforcement learning. Across all three domains, the average reasoning length decreases substantially during training and eventually converges to a shorter level, while recommendation performance continues to improve. Notably, our objective contains no length penalty; the reward is determined solely by how well a reasoning trajectory ranks the ground-truth SID under constrained beam search. The shortening of reasoning therefore emerges naturally from ranking-aware optimization.

Stage-2 provides the model with better reasoning trajectories as a useful initialization, and during Stage-3 RL, the model progressively learns to disgard those redundant or less relevant reasoning that do not contribute to ranking the target item. This behavior suggests that improved recommendation reasoning does not require increasingly long chains of thought. Instead, efective reasoning appears to depend on how eficiently a trajectory captures information that is useful for ranking. In this sense, Stage-3 RL help the model achieve stronger recommendation performance with shorter and more targeted reasoning trajectories.

4.4.2 Evolution of Thinking Performance. Table 7 compares recommendation performance under thinking mode before and after Stage-3 reinforcement learning. Although the Stage-2 reasoning policy initially exhibits relatively weak recommendation performance, Stage-3 RL consistently recovers and substantially improves all ranking metrics across the three domains. The improvement is especially pronounced at the top of the ranking. On Games, R@1 increases from 0.0103 to 0.0446, corresponding to a 333.0% relative improvement, substantially larger than the gain in R@10. This suggests that Stage-3 optimization does more than recover overall retrieval performance: it sharpens the ranking distribution by pushing the target SID toward the very top of the constrained beam. These results also clarify the diferent roles of the two training stages. Stage 2 primarily serves as a thinking mode warm-up, teaching the model to produce reasoning in the desired thinking format while using best-of-� rejection sampling to expose it to higherquality reasoning trajectories. Its objective is therefore not to directly optimize the final ranking performance. Therefore, relatively low thinking performance at this stage does not necessarily imply that the learned reasoning is inefective. More importantly, learning from better reasoning trajectories provides a stronger potential for subsequent reinforcement learning. Stage-3 RL then provides a taskaligned ranking signal, reinforcing reasoning trajectories that place the ground-truth SID higher in the beam. Consequently, the reasoning policy becomes increasingly aligned with recommendation, yielding large improvements from Stage 2 to Stage 3, particularly at the top of the ranking.

![](images/b5c2c588bfb8a6de58d12d474b377a2b43b09e6754412abfcd447feb11dac2be.jpg)  
Figure 2: Mean reasoning length over Stage-3 training.

Table 7: Thinking performance before and after Stage-3 RL.
<table><tr><td>Dataset</td><td>Stage</td><td>R@1</td><td>R@5</td><td>R@10</td><td>N@5</td><td>N@10</td></tr><tr><td></td><td>Stage 2</td><td>0.0103</td><td>0.0256</td><td>0.0405</td><td>0.0177</td><td>0.0225</td></tr><tr><td>Games</td><td>Stage 3</td><td>0.0443</td><td>0.0847</td><td>0.1149</td><td>0.0649</td><td>0.0746</td></tr><tr><td></td><td>Stage 2</td><td>0.0495</td><td>0.0789</td><td>0.0908</td><td>0.0652</td><td>0.0691</td></tr><tr><td>Office</td><td>Stage 3</td><td>0.0933</td><td>0.1476</td><td>0.1710</td><td>0.1223</td><td>0.1299</td></tr><tr><td></td><td>Stage 2</td><td>0.0435</td><td>0.0679</td><td>0.0841</td><td>0.0562</td><td>0.0614</td></tr><tr><td>Industrial</td><td>Stage 3</td><td>0.0757</td><td>0.1167</td><td>0.1476</td><td>0.0976</td><td>0.1076</td></tr></table>

Table 8: Statistics of the best-of-� corpus at $N = 5 .$ “pool” averages all � candidates of an instance, i.e. the expected utility of a random pick.
<table><tr><td></td><td>Games</td><td>Office</td><td>Industrial</td></tr><tr><td>Instances</td><td>49,133</td><td>38,924</td><td>36,259</td></tr><tr><td>Candidates (N=5)</td><td>245,665</td><td>194,620</td><td>181,295</td></tr><tr><td>Saturated candidates</td><td>1.9%</td><td>11.5%</td><td>6.7%</td></tr><tr><td>Accepted instances</td><td>47,976</td><td>37,662</td><td>35,372</td></tr><tr><td>Rejected instances</td><td>1,157</td><td>1,262</td><td>887</td></tr><tr><td>Rejection rate</td><td>2.4%</td><td>3.2%</td><td>2.4%</td></tr><tr><td>Δretained</td><td>3.72</td><td>3.03</td><td>4.10</td></tr><tr><td>Δ pool</td><td>3.16</td><td>2.60</td><td>3.61</td></tr><tr><td>Within-instance ∆ range</td><td>1.22</td><td>0.95</td><td>1.07</td></tr></table>

## 5 Analysis on Best-of-� Rejection Sampling

We further analyze the resulting Stage-2 corpus at � = 5 across all three domains. As shown in Table 8, the vast majority of sampled reasoning traces have positive predictive utility, and requiring at least one positive candidate rejects only 2.4%, 3.2%, and 2.4% of training instances. Moreover, rejected instances are predominantly near the decision boundary in Figure 3. These observations explain why rejection sampling alone provides little improvement in Table 3: the binary Δ > 0 criterion is too weak to substantially reshape the training distribution.

Although rejection is rare, the candidate reasoning traces for the same history can difer substantially in predictive utility. The mean within-instance Δ range, max<sub>�</sub> $\Delta _ { n , k } - \operatorname* { m i n } _ { k } \Delta _ { n , k } $ is 1.22, 0.94, and 1.06 on Games, Ofice, and Industrial, respectively. Best-of-� greatly exploits this within-instance variation. Compared with uniformly drawing one candidate from the same pool, the retained arg max� $\Delta _ { n , k }$ trace increases predictive utility by 0.56, 0.43, and 0.49 on Games, Ofice, and Industrial. The main gain comes from ranking candidates within each instance. We found that the dataset with smaller within-instance Δ range tends to less benefit from Best-of-� rejection sampling.

Another interesting observation is that a target that is already highly predictable from the history leaves little room for any reasoning trace to further increase its likelihood. Using log $p _ { \bar { \theta } } ( y _ { n } ^ { * } \mid h _ { n } ) >$ −0.2 as an indicator of such saturation, we find saturated-candidate rates of 1.9%, 11.5%, and 6.7% on Games, Ofice, and Industrial. This efect is particularly pronounced on Ofice and produces the sharp mass near $\Delta ^ { * } \approx 0$ in Figure 3. Thus, a small Δ does not necessarily indicate poor reasoning; it can also reflect an instance for which the history-only predictor is already close to its likelihood ceiling. Games exhibits the largest within-instance utility range (1.22) and the lowest saturation rate (1.9%), providing the largest opportunity for best-of-� to identify a substantially better reasoning trace. In contrast, Ofice has both the highest saturation rate (11.5%) and the smallest within-instance range (0.94). Its candidate utilities are therefore more tightly compressed, reducing the benefit obtainable from best-of-� selection.

![](images/d160b381b7cbb77010cb7d24a2f2a6c05f56e2b1ab10e2c55d54f2cfd825f3ae.jpg)  
Figure 3: Distribution of the best predictive utility per instance, $\Delta _ { n } ^ { * } = \operatorname* { m a x } _ { k } \Delta _ { n , k } ,$ at $N = 5 .$ Instances left of the accept bar are rejected; the dashed curve is the pooled candidate distribution a random pick draws from.

Taken together, these results reveal that best-of-� rejection sampling operates primarily as a within-instance ranking mechanism, with rejection playing a secondary role. Figure 4 further shows that this gain is not driven by a sampling-slot bias. The five candidate slots are selected by arg max� $\Delta _ { n , k }$ at approximately uniform rates across all three datasets, and their negative-utility rates are also largely stable across slots. Thus, no particular sampling position systematically produces better reasoning traces; the benefit arises from comparing multiple candidates and selecting the one with the highest predictive utility. Increasing � therefore provides more opportunities to expose high-utility traces, particularly for instances with large candidate variation. This mechanism is consistent with the monotonic improvement from � = 1 to � = 5 in Table 4.

![](images/4cf6cd3243478af34673315a9368bb83c4aa5aa0adc8ebb8ce0f57cd9c52d59d.jpg)  
Figure 4: Outcome of the candidates drawn in each sampling slot $k ,$ as a share of all instances of the domain.

## 6 Conclusion

In this work, we investigate how to learn and progressively improve efective reasoning for Semantic-ID based generative recommendation. We propose Evo-Rec, a three-stage framework that first grounds Semantic IDs in language and user behavior, then selects higher-utility reasoning trajectories through Best-of-� Rejection Sampling SFT, and finally refines the reasoning policy with catalog-constrained, ranking-aware reinforcement learning. Experiments on three Amazon Review benchmarks demonstrate consistent improvements over discriminative, generative, and reasoningenhanced recommenders, with particularly strong gains on rankingsensitive metrics. Our analyses further reveal that the efectiveness of Best-of-� rejection sampling primarily comes from distinguishing reasoning trajectories with diferent predictive utility within the same interaction history, rather than from rejection alone. Moreover, NDCG-based reinforcement learning provides a more informative signal than coarse outcome rewards by directly reflecting the ranking induced by each reasoning trajectory. It leads to particularly pronounced gains at the top of the ranking, especially on Recall@1. Interestingly, the optimized policy also tends to produce shorter reasoning while achieving stronger recommendation performance, suggesting that efective recommendation reasoning depends more on useful information than on reasoning length. Overall, these findings support a simple principle: recommendation reasoning should not merely be generated, but should be better selected and optimized according to its downstream recommendation utility.

## Ethical Considerations

Our experiments are conducted on public Amazon Review bench marks, which are distributed for research purposes and contain no directly identifying information: users appear as opaque pseudonymous identifiers, and our models take in only interaction order and item-side metadata such as titles and descriptions. Evo-Rec is evaluated in an ofline recommendation setting and does not involve deployment on private user data. If applied to real-world systems in the future, appropriate privacy protection, bias auditing, and safeguards against over-personalization should be considered when handling private interaction histories.

## References

[1] Jiaxin Deng, Shiyao Wang, Kuo Cai, Lejian Ren, Qigen Hu, Weifeng Ding, Qiang Luo, and Guorui Zhou. 2025. Onerec: Unifying retrieve and rank with generative recommender and iterative preference alignment. arXiv preprint arXiv:2502.18965 (2025).

[2] Shijie Geng, Shuchang Liu, Zuohui Fu, Yingqiang Ge, and Yongfeng Zhang. 2022. Recommendation as language processing (rlp): A unified pretrain, personalized prompt & predict paradigm (p5). In Proceedings of the 16th ACM conference on recommender systems. 299–315.

[3] Yingzhi He, Yan Sun, Junfei Tan, Yuxin Chen, Xiaoyu Kong, Chunxu Shen, Xiang Wang, An Zhang, and Tat-Seng Chua. 2026. Reasoning over semantic ids enhances generative recommendation. In Proceedings ofthe 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2. 1638–1649.

[4] Balázs Hidasi and Alexandros Karatzoglou. 2018. Recurrent neural networks with top-k gains for session-based recommendations. In Proceedings ofthe 27th ACM international conference on information and knowledge management. 843–852.

[5] Yupeng Hou, Zhankui He, Julian McAuley, and Wayne Xin Zhao. 2023. Learning vector-quantized item representation for transferable sequential recommenders. In Proceedings ofthe ACM Web Conference 2023. 1162–1171.

[6] Yupeng Hou, Jiacheng Li, Xiangjun Fu, Zhankui He, An Yan, Xiusi Chen, and Julian McAuley. 2026. Bridging language and items for retrieval and recommen dation: Benchmarking LLMs as semantic encoders. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 3251–3265.

[7] Wang-Cheng Kang and Julian McAuley. 2018. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM). IEEE, 197–206.

[8] Xiaoyu Kong, Junguang Jiang, Bin Liu, Ziru Xu, Han Zhu, Jian Xu, Bo Zheng, Jiancan Wu, and Xiang Wang. 2026. Think before Recommendation: Autonomous Reasoning-enhanced Recommender. Advances in Neural Information Processing Systems 38 (2026), 141209–141232.

[9] Xiaoyu Kong, Leheng Sheng, Junfei Tan, Yuxin Chen, Jiancan Wu, An Zhang, Xiang Wang, and Xiangnan He. 2025. Minionerec: An open-source framework for scaling generative recommendation. arXiv preprint arXiv:2510.24431 (2025).

[10] Kelvin Li, Chuyi Shang, Leonid Karlinsky, Rogerio Feris, Trevor Darrell, and Roei Herzig. 2026. Latent implicit visual reasoning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition. 33457–33466.

[11] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In International Conference on Learning Representations, Vol. 2024. 39578–39601.

[12] Jiacheng Lin, Tian Wang, and Kun Qian. 2025. Rec-r1: Bridging generative large language models and user-centric recommendation systems via reinforcement learning. arXiv preprint arXiv:2503.24289 (2025).

[13] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[14] Jiakai Tang, Sunhao Dai, Teng Shi, Jun Xu, Xu Chen, Wen Chen, Jian Wu, and Yuning Jiang. 2026. Think before recommend: Unleashing the latent reasoning power for sequential recommendation. IEEE Transactions on Knowledge and Data Engineering (2026).

[15] Jiaxi Tang and Ke Wang. 2018. Personalized top-n sequential recommendation via convolutional sequence embedding. In Proceedings ofthe eleventh ACM international conference on web search and data mining. 565–573.

[16] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024. Learnable item tokenization for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2400–2409.

[17] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. arXiv preprint arXiv:2203.11171 (2022).

[18] Ye Wang, Jiahao Xun, Minjie Hong, Jieming Zhu, Tao Jin, Wang Lin, Haoyuan Li, Linjun Li, Yan Xia, Zhou Zhao, et al. 2024. Eager: Two-stream generative recommender with behavior-semantic collaboration. In Proceedings ofthe 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 3245–3254.

[19] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. 2022. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems 35 (2022), 24824–24837.

[20] Longtao Xiao, Haozhao Wang, Cheng Wang, Linfei Ji, Yifan Wang, Jieming Zhu, Zhenhua Dong, Rui Zhang, and Ruixuan Li. 2025. Unger: Generative recommendation with a unified code via semantic and collaborative integration. ACM Transactions on Information Systems 44, 2 (2025), 1–31.

[21] Runyang You, Yongqi Li, Xinyu Lin, Xin Zhang, Wenjie Wang, Wenjie Li, and Liqiang Nie. 2026. R<sup>2</sup>ec: Towards Large Recommender Models with Reasoning. Advances in Neural Information Processing Systems 38 (2026), 62376–62405.

[22] Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. 2022. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems 35 (2022), 15476–15488

[23] Jiaqi Zhai, Lucy Liao, Xing Liu, Yueming Wang, Rui Li, Xuan Cao, Leon Gao, Zhaojie Gong, Fangda Gu, Jiayuan He, Yinghai Lu, and Yu Shi. 2024. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 235), Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (Eds.). PMLR, 58484–58509. https://proceedings.mlr.press/v235/zhai24a.html

[24] Yang Zhang, Wenxin Xu, Xiaoyan Zhao, Wenjie Wang, Fuli Feng, Xiangnan He, and Tat-Seng Chua. 2026. Reinforced latent reasoning for llm-based recommendation. In International Conference on Learning Representations, Vol. 2026. 128449–128470.

[25] Bowen Zheng, Yupeng Hou, Hongyu Lu, Yu Chen, Wayne Xin Zhao, Ming Chen, and Ji-Rong Wen. 2024. Adapting large language models by integrating collaborative semantics for recommendation. In 2024 IEEE 40th International Conference on Data Engineering (ICDE). IEEE, 1435–1448.

[26] Mengdan Zhu, Senhao Cheng, and Liang Zhao. 2026. Decompose, Look, and Reason: Reinforced Latent Reasoning for VLMs. arXiv preprint arXiv:2604.07518 (2026).

[27] Mengdan Zhu, Yufan Zhao, Tao Di, Yulan Yan, and Liang Zhao. 2026. Learning User Interests via Reasoning and Distillation for Cross-Domain News Recom mendation. arXiv preprint arXiv:2602.15005 (2026).