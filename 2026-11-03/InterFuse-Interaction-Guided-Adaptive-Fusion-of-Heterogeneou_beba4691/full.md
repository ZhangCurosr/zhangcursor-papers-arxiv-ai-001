# InterFuse: Interaction-Guided Adaptive Fusion of Heterogeneous Models for Payment Risk Ranking

Haoyu Jiang Ant International wuxian.jhy@ant-intl.com

Ruobing Zuo Ant International East China Normal University

Weiyu Han Ant International

Xiao Tan Ant International

Ze Zhao   
Ant International   
Shanghai Jiao Tong   
University

Shuai Chen Ant International

Jie Sheng Ant Internationa

Tianyi Zhang Ant International

## Abstract

Payment platforms use view-specific models trained on transaction attributes, behavioral histories, and relational structures to rank transactions under limited review budgets. However, existing fusion methods often focus on prediction scores or interactions among model representations, with less attention to how these interactions should guide aggregation using each model’s original prediction and input coverage. To address this limitation, we propose InterFuse, a fusion framework that uses a shared fusion state from cross-model interaction to guide transaction-adaptive aggregation. InterFuse encodes each model’s representation, prediction score, and input coverage as a model-specific token, using learned states for unavailable outputs. Availability-aware attention updates these tokens and produces a shared fusion state. This state then guides aggregation over available models using their updated representations, original predictions, and input coverage. We evaluate InterFuse on an industrial payment replay and three public financial benchmarks using the same upstream models across fusion methods. InterFuse achieves the highest top-ranked recall and AUC on all four datasets, improving Recall@1% by 1.11–8.09 percentage points over the strongest baselines. Ablations show the benefits of cross-model interaction and adaptive aggregation, while industrial analyses find larger interaction gains when model predictions disagree. Additional analyses examine robustness to missing outputs and the trade-of between ranking gains and fusion latency.

## CCS Concepts

• Computing methodologies → Neural networks; • Information systems → Learning to rank.

## Keywords

Fraud detection, model fusion, representation learning, attention mechanisms

## 1 Introduction

Web payment platforms handle massive volumes oftransactions but can manually review only a small fraction of them. Identifying high risk transactions therefore requires integrating information from multiple sources. Transaction attributes characterize the current payment, behavioral histories capture patterns in user, card, and device activity [22], and graph models reveal connections among accounts and transactions [17, 33]. Recent financial benchmarks also incorporate both transaction behavior and relational information [3]. These complementary sources reflect diferent aspects of the same transaction and may yield divergent risk assessments. When modeled separately using tabular, sequential, and graphbased approaches, their outputs must be fused to generate a unified risk ranking.

![](images/4dd5c48d53c26fc355b823357e24cf44fbd1f27218ff6b8b5e6998868fbb16f8.jpg)  
Figure 1: Risk views and model fusion. InterFuse uses crossmodel interaction to guide adaptive aggregation.

Existing approaches combine these model outputs at diferent levels. Score-level fusion integrates individual predictions through fixed weights or learned functions [12, 29]. Representation-level fusion additionally leverages the features learned by each model, which may encode information not captured in its prediction score. Beyond simple concatenation, the Collaborative Experts model captures relationships among experts to gate their features [16], while MMT uses attention to exchange information across model representations [6]. Other approaches model dependencies within and across modalities [18], learn to combine diferent types of interactions [31], or handle missing inputs [21, 35]. These methods provide diferent ways to model relationships among information sources and adjust their contributions. For independently trained risk models, an important question is how information learned through such interactions should guide the final aggregation when each model also provides its own prediction and input coverage.

Consider a large payment that appears risky based on its transaction attributes but is consistent with the customer’s spending history. A connection to a suspicious account may further alter this assessment. The contribution of each model therefore depends not only on its own output but also on what other models reveal about the transaction. The inputs available to each model also matter. For example, a history model may yield a similar prediction whether it is given only a few past transactions or extensive recent activity, even though the information available to the model difers. Some models may produce no output at all. Figure 1 illustrates these diferent sources of information and the task of combining their outputs. These considerations raise two related challenges:

• Cross-model interaction. How can model representations be updated using information from other models while accounting for diferences in input coverage and availability?

• Interaction-guided adaptive aggregation. How can the shared context from cross-model interaction guide each model’s contribution while retaining its original prediction and input coverage?

To address these challenges, we propose InterFuse, an interaction-guided adaptive fusion framework for heterogeneous risk models. InterFuse represents each model’s learned features, prediction score, and input coverage as a modelspecific token. Learned identities distinguish the models, while learned missing states encode unavailable outputs. Availabilityaware attention enables models to exchange information, yielding updated representations and a shared fusion state that captures cross-model information. This shared state then guides adaptive aggregation across the available models. Each updated representation is combined with its original prediction and coverage, so that aggregation can leverage both cross-model information and each model’s own assessment. The pooled representation and the shared fusion state jointly determine the final risk score. InterFuse trains only the fusion network, enabling low-cost reuse of existing business model capabilities.

In summary, the main contributions of our work are as follows:

• We propose InterFuse, a framework for combining independently trained heterogeneous risk models. It updates model representations through cross-model interaction and uses the resulting shared context to guide transactionadaptive aggregation.

• We develop availability-aware attention to handle missing outputs and aggregation guided by a shared fusion state. Original predictions and input coverage are used at both stages, allowing the fusion network to combine updated representations with each model’s original information.

• Experiments on an industrial payment replay and three public benchmarks show improved ranking performance over the strongest baselines. Controlled comparisons show that model-level updates add to global fusion, and that explicit shared-state conditioning further improves adaptive aggregation. Industrial analyses show larger interaction gains when model predictions disagree.

## 2 Related Work

View-specific models and model reuse. Tabular, sequence, and graph models learn from diferent data structures [7, 17, 22, 33], with financial benchmarks covering transaction- and client-level tasks [3, 19]. Existing model-reuse methods connect frozen models through trainable querying or cross-attention [1, 14], or align heterogeneous encoders for merging and fusion [10]. InterFuse fuses representations and predictions from fixed view-specific models, accounting for output availability and input coverage.

Cross-model interaction and aggregation. Fusion methods aggregate model predictions or integrate learned representations. MarkerFusion [32] aggregates detector predictions through a domain-conditioned probabilistic model for unsupervised and semi-supervised threat detection. Representation-level methods use set aggregation [13, 34], expert gating [16], and cross-modal attention [4, 6, 26]. Shared bottlenecks and latent spaces mediate information exchange [9, 20], while other designs use joint and individual predictors, interaction experts, or sequential modules [18, 25, 31]. Complementary encoders and adaptive routing ofer further fusion strategies [15, 24, 30]. InterFuse uses the shared fusion state from model interaction to determine each model’s weight for the current transaction. Both interaction and aggregation use the models’ original predictions and input coverage.

Fusion with incomplete inputs. Prior work addresses unseen modality combinations [35], unequal availability and learning imbalance [21, 28], and representation collapse caused by noisy fea tures [2]. Other methods reconstruct missing modalities [5, 8, 11] or enrich observed ones using other samples [36]. These settings concern both absent inputs and unequal contributions from observed sources, with diferent assumptions about the available representations. InterFuse represents missing outputs with learned states and uses availability to control attention and pooling, while coverage describes available inputs.

## 3 Problem Setup

We study the fusion of independently trained view-specific models for risk ranking. Given an event � with binary label $y \in \{ 0 , 1 \}$ , our goal is to learn a fusion function �(�) that combines the outputs of these models to produce a single risk score. In payment risk assessment, events are transactions, and higher scores indicate greater predicted risk.

We consider � models, whose architectures and representation dimensions may difer. For each model �, a binary indicator $a _ { r } ( x ) \in$ {0, 1} indicates whether its output is available. An available model provides a learned representation $h _ { r } ( x ) \in \mathbb { R } ^ { d _ { r } }$ and a prediction score �<sub>�</sub> (�) expressed as a logit. Its input coverage vector $c _ { r } ( x ) \in$ R<sup>��</sup> captures properties of the observed input, such as history length, recency, or graph degree. Availability and coverage capture diferent conditions: a model with a short history can still provide a valid output. We denote the set of available models by $\mathcal { A } ( \boldsymbol { x } ) = \{ r : a _ { r } ( \boldsymbol { x } ) = 1 \}$ and assume that at least one model is available for each event.

All model outputs pertain to the same event. Historical, relational, and coverage inputs draw on information available at the prediction time of that event. We train the fusion function using event labels while keeping the upstream models and their original prediction heads frozen. The fusion network learns from the available representations and predictions, together with input coverage and the pattern of model availability. For readability, we omit � from individual model outputs below.

The fusion function produces a scalar logit for ranking. Given a review budget of $q \% ,$ , the platform selects the highest-scoring $q \%$ of events. Recall@�% measures the fraction of all positive events recovered within this selection. We describe the task for a single binary target; for multi-target tasks, the fusion model produces one logit per target and ranks events separately for each target.

## 4 InterFuse

InterFuse updates model representations through cross-model at tention, producing a shared fusion state that guides subsequent aggregation (Figure 2). Original predictions and input coverage inform both interaction and aggregation, linking cross-model context to each model’s contribution to the final risk score.

## 4.1 Representing Model Outputs

For an available model, its model token encodes three kinds of information for joint risk assessment. The representation $h _ { r } \in \mathbb { R } ^ { d _ { r } }$ supplies learned features, the score $s _ { r }$ records the model’s risk prediction, and $c _ { r }$ describes its input coverage. The fusion network can therefore use both the existing prediction and features that become useful alongside other models.

For an available model, InterFuse concatenates these fields, projects them to a shared width, and adds a learned model identity. Availability selects this projected representation or a learned missing state:

$$
z _ { r } ^ { 0 } = \rho _ { r } + \{ { { P _ { r } } ( [ h _ { r } \mathrm { ~ } \middle |  \kern - delimiterspace } s _ { r } \mathrm { ~ } \middle | ] c _ { r } ] )  , \quad a _ { r } = 1 ,\tag{1}
$$

Here $P _ { r }$ maps the available fields to width $d ,$ and $\rho _ { r } , m _ { r } \in \mathbb { R } ^ { d }$ are learned embeddings. The projections are model-specific because the input dimensions and feature meanings can difer across sources. Their outputs share the width required by the subsequent attention layers.

Availability distinguishes an absent output from an output based on sparse observations. A sequence model with a short history contributes its representation and coverage through the available branch of Equation 1. An unavailable model instead contributes its learned missing state. Adding the model identity in both cases lets the interaction layers distinguish which source supplied an output and which source is absent.

We prepend a learned fusion token and stack the states into $Z ^ { 0 } = [ \bar { z } _ { f } ^ { 0 } ; \bar { z } _ { 1 } ^ { 0 } ; \ldots ; z _ { R } ^ { 0 } ] \in \mathbb { R } ^ { ( R + 1 ) \times d }$ . The fusion token starts from the same vector for every event and becomes event-specific through attention. Each model has one token regardless of the size of its raw input. Model identity is attached to each model token; the sequence order is used to organize the computation, without an additional positional embedding.

## 4.2 Cross-Model Interaction

Cross-model interaction updates each model’s representation using information from other models before aggregation assigns its weight. A sequence-model token can incorporate tabular information about the current payment and graph information about related entities. The same sequence output can therefore produce diferent updated tokens for diferent transaction contexts. These updates occur within the fusion network; the view-specific models and their original outputs remain fixed.

The fusion and model tokens pass through � pre-normalized Transformer blocks. Each attention head projects the normalized states into queries, keys, and values. For one head, scaled dotproduct attention [27] takes the form

$$
{ \mathrm { A t t n } } ( Z ) = { \mathrm { s o f t m a x } } _ { \mathrm { r o w } } \left( { \frac { Q K ^ { \top } } { \sqrt { d _ { k } } } } + M ( x ) \right) V ,\tag{2}
$$

where $d _ { k }$ is the head width and softmax is applied over key positions in each row. All heads use the same event-dependent mask �(�), but learn separate projections. Their outputs are concatenated and projected back to token width �.

Communication under missing inputs. The fusion state records both available model outputs and the pattern of missing sources. Other model tokens exchange observed information without directly reading another model’s missing state. Index the fusion token by 0 and model tokens by $1 , \ldots , R .$ The additive mask is

$$
M _ { i j } ( x ) = \left\{ { \begin{array} { l l } { 0 , } & { i = 0 { \mathrm { ~ o r ~ } } j = 0 { \mathrm { ~ o r ~ } } i = j , } \\ { 0 , } & { j \geq 1 { \mathrm { ~ a n d ~ } } a _ { j } ( x ) = 1 , } \\ { - \infty , } & { { \mathrm { o t h e r w i s e } } . } \end{array} } \right.\tag{3}
$$

The fusion row admits every model state. Model-token rows admit available keys, the fusion key, and self-connections. Blocked keys receive zero attention weight after softmax.

In Figure 2, the available tabular, user, and device tokens read one another, while the missing card and graph keys are blocked for these queries. The fusion token reads all five model states, including the two missing states. Queries for unavailable models retain self-connections and access to observed states, but are excluded from final pooling. In subsequent layers, the fusion token can pass information about the available source combination back to the model tokens.

Updating each token. Writing the complete multi-head operation as MHA, one block computes

$$
\begin{array} { c } { { { \widetilde { Z } ^ { \ell } = Z ^ { \ell } + \mathrm { M H A } ( \mathrm { L N } ( Z ^ { \ell } ) , M ) , } } } \\ { { { \cal Z } ^ { \ell + 1 } = { \widetilde { Z } ^ { \ell } + \mathrm { F F N } ( \mathrm { L N } ( { \widetilde { Z } ^ { \ell } } ) ) } . } } \end{array}\tag{4}
$$

The feed-forward network acts separately on each token, using shared weights across positions within the block. Residual connections retain each state alongside its learned update. We denote the final model states by $z _ { r }$ and the final fusion state by $z _ { f }$

The updated model representation �<sub>�</sub> contains information from other models for the current event. The shared fusion state $z _ { f }$ guides the subsequent aggregation.

## 4.3 Interaction-Guided Adaptive Aggregation

Interaction-guided adaptive aggregation determines how the updated model tokens contribute to the current transaction. Each model’s local network processes its updated representation, original prediction, and input coverage. The shared fusion state then guides their relative weights.

![](images/63add87ca8ef90c41c7404c24c02571bc5c80122a697c5ff18422c97b03215f8.jpg)  
Figure 2: InterFuse architecture. Frozen model outputs are encoded as model tokens, updated through availability-aware attention, and adaptively aggregated for risk prediction. Card and graph outputs are unavailable in this example.

Model-specific representations and aggregation weights. For each available model, a small network $\phi _ { r }$ maps the updated token, original logit, and coverage to $u _ { r } \in \mathbb { R } ^ { d _ { u } }$ . These networks are model-specific, and their outputs share width $d _ { u } .$ . The shared scoring network $g$ is a two-layer MLP with a GELU activation and hidden width $d _ { u } .$ It jointly transforms the concatenated fusion state and model representation before producing a scalar score. The scores are normalized over available models:

$$
\begin{array} { r l } & { u _ { r } = \phi _ { r } ( [ z _ { r } \parallel s _ { r } \parallel c _ { r } ] ) , } \\ & { \alpha _ { r } = \cfrac { \exp { g ( [ z _ { f } \parallel u _ { r } ] ) } } { \sum _ { j \in \mathcal { A } ( x ) } \exp { g ( [ z _ { f } \parallel u _ { j } ] ) } } . } \end{array}\tag{5}
$$

Normalization includes only available models. The pooled representation is $\begin{array} { r } { u = \sum _ { r \in \mathcal { A } ( x ) } \alpha _ { r } u _ { r } } \end{array}$ . The weights sum to one over this set; unavailable models receive zero weight. If only one model is available, its pooling weight is one, while the fusion state still incorporates the pattern of missing sources.

The nonlinear scoring network allows the shared fusion state to influence the relative model weights. A model’s contribution can change with its own output, its input coverage, and the information supplied by other models.

The two entry points for scores and coverage serve distinct purposes. In Equation 1, they help shape the messages exchanged during interaction. In Equation 5, they provide direct access to each model’s original assessment and input conditions after its token has been updated. Aggregation can use this original information alongside the learned cross-model context.

Prediction. The final representation concatenates the shared fusion state $z _ { f }$ and the weighted model information �. A direct prediction head maps their concatenation to a risk logit:

$$
f ( x ) = \psi ( [ z _ { f } \parallel u ] ) .\tag{6}
$$

Here $\psi$ is the trainable prediction network. The model scores $s _ { r }$ enter through the model tokens and $u _ { r } ,$ , so their role in the final ranking is learned jointly with the fusion layers.

## 4.4 Training and Inference

Fusion training. Fusion training learns to integrate existing model outputs. Each event supplies aligned representations, prediction scores, availability indicators, and input coverage. The trainable parameters comprise the model-specific projections, identity and missing embeddings, fusion token, interaction layers, aggregation networks, and prediction head. The upstream encoders and their original prediction heads remain fixed, allowing their outputs to be cached and reused across fusion-training iterations.

Let � collect the fusion parameters and � denote the number of fusion-training events. Training minimizes binary cross-entropy evaluated on the output logits:

$$
\mathcal { L } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \ell _ { \mathrm { B C E } } ( f _ { \theta } ( x _ { i } ) , y _ { i } ) .\tag{7}
$$

The loss $\ell _ { \mathrm { B C E } }$ uses the class weights of the corresponding training configuration; for multiple targets, it averages the target-wise losses.

Table 1: Overall ranking performance (mean ± standard deviation over five fusion-training seeds). Industrial reports gains over fixed B1 (pp); public datasets report percentages. B1 is fixed across seeds. R@1 denotes Recall@1%; MBD averages four targets. Bold and underline mark the best and second-best means.
<table><tr><td></td><td colspan="3">Industrial (∆pp)</td><td colspan="3">TransXion (%)</td><td colspan="3">IBM CCT (%)</td><td colspan="3">MBD (%)</td></tr><tr><td>Method</td><td>AUC</td><td>AP</td><td>R@1</td><td>AUC</td><td>AP</td><td>R@1</td><td>AUC</td><td>AP</td><td>R@1</td><td>AUC</td><td>AP</td><td>R@1</td></tr><tr><td>B1</td><td>+0.00</td><td>+0.00</td><td>+0.00</td><td>82.47</td><td>13.21</td><td>39.47</td><td>94.06</td><td>3.06</td><td>34.01</td><td>68.20</td><td>0.396</td><td>3.18</td></tr><tr><td>B5</td><td></td><td>+0.85±0.11 +0.37±0.35 +1.77±0.21</td><td></td><td>93.69±0.07</td><td>18.85±0.41</td><td>60.31±0.52</td><td>95.33±0.07</td><td>4.09±0.25</td><td>36.51±0.46</td><td>57.62±0.42</td><td>0.308±0.009</td><td>2.63±0.17</td></tr><tr><td>Concat</td><td>+1.78±0.17</td><td>+2.67±0.22 +1.99±0.44</td><td></td><td>94.30±0.07</td><td>18.76±0.37</td><td>62.64±0.62</td><td>94.69±0.40</td><td>4.69±1.07</td><td>36.25±1.75</td><td>72.84±0.81</td><td>0.647±0.024</td><td>6.20±0.14</td></tr><tr><td>Sum-pooling</td><td>-0.67±0.13 +1.37±0.31 +1.81±0.21</td><td></td><td></td><td>94.60±0.08</td><td>19.26±0.65</td><td>62.97±0.96</td><td>94.62±0.13</td><td>4.54±0.28</td><td>36.07±0.35</td><td>71.55±0.73</td><td>0.605±0.032</td><td>5.97±0.24</td></tr><tr><td>CE</td><td>-0.53±0.14 +1.19±0.48 +1.71±0.71</td><td></td><td></td><td>94.83±0.06</td><td>18.89±0.82</td><td>62.75±1.25</td><td>94.85±0.30</td><td>4.74±0.61</td><td>36.84±1.39</td><td>72.38±0.390.530±0.013</td><td></td><td>4.48±0.43</td></tr><tr><td>MMT</td><td>-0.94±0.25 +1.67±0.87 +1.96±1.30</td><td></td><td></td><td>94.63±0.28</td><td>17.90±1.00</td><td>61.64±1.51</td><td>95.29±0.66</td><td>5.11±1.06</td><td></td><td>39.34±1.82 72.25±0.360.533±0.033</td><td></td><td>5.32±0.52</td></tr><tr><td>UMI</td><td></td><td>+0.43±0.11 -6.35±0.23 +3.72±0.27</td><td></td><td>98.59±0.09</td><td>28.87±0.63</td><td>88.58±0.98</td><td>92.93±0.21</td><td>12.16±0.27</td><td></td><td>53.37±0.92 72.35±0.32 0.562±0.062</td><td></td><td>5.28±0.39</td></tr><tr><td>I²M²</td><td></td><td>-0.13±0.21 -3.07±0.47 +2.87±0.65</td><td></td><td>99.10±0.09</td><td>29.47±0.34</td><td>87.14±0.67</td><td>94.92±0.30</td><td></td><td>10.88±0.39 51.27±1.36</td><td>73.24±0.75 0.596±0.016</td><td></td><td>4.76±0.26</td></tr><tr><td>MMPareto</td><td></td><td>+1.67±0.17 +0.37±0.15 +0.50±0.34</td><td></td><td>93.91±0.13</td><td>18.96±0.47</td><td>62.20±0.74</td><td>94.58±0.18</td><td></td><td>4.78±0.71 36.69±0.79</td><td>72.76±0.470.596±0.036</td><td></td><td>6.36±0.54</td></tr><tr><td>I²MoE</td><td>+1.29±0.26 +2.59±0.68 +1.48±0.69</td><td></td><td></td><td></td><td>94.50±0.09 18.67±0.98</td><td>62.20±0.9494.93±0.49</td><td></td><td></td><td></td><td>5.02±0.78 37.43±2.4959.68±0.53 0.418±0.020</td><td></td><td>2.60±0.51</td></tr><tr><td>BALM</td><td>+1.46±0.38 +0.64±0.54 +1.27±0.38</td><td></td><td></td><td>94.75±0.08</td><td>18.97±0.70</td><td>62.08±0.64</td><td>94.87±0.30</td><td>5.48±0.82</td><td>38.31±1.32</td><td>72.94±0.74 0.590±0.040</td><td></td><td>6.87±0.67</td></tr><tr><td>InterFuse</td><td>+1.95±0.16</td><td>+3.10±0.28</td><td>+5.44±0.37</td><td>99.15±0.18</td><td>29.51±0.67</td><td>89.69±0.92</td><td>96.82±0.23</td><td>22.36±0.29</td><td>61.46±1.46</td><td>77.80±0.90</td><td>0.677±0.033</td><td>10.20±0.62</td></tr></table>

Optimizer weight decay is specified separately from the prediction loss. Supervision reaches the model-specific projection networks and their pooling weights through the final prediction.

Inference and cost. For a new transaction, InterFuse constructs observed or missing tokens from the model outputs, builds the availability mask, and applies the interaction and aggregation layers. The output logit determines the review ranking. The same fusion network accepts diferent available subsets through the mask and pooling normalization. Algorithm 1 in Appendix A.6 summarizes training and inference.

Each interaction layer processes � model slots and one fusion token, with (� + 1)<sup>2</sup> query–key scores per head. Missing models retain their slots, so the sequence length is fixed for a given model inventory. For the five-model industrial setting, this gives six positions regardless of history length or graph size. Model projections and feed-forward layers act on these tokens, while the upstream encoders process the raw histories and graphs. We measure fusion cost separately from the cost of producing its inputs.

## 5 Experiments

Our evaluation examines heterogeneous model fusion for risk ranking. We organize the experiments around four questions: RQ1, does InterFuse improve ranking across tasks and review budgets? RQ2, which inputs and fusion components account for the gains? RQ3, how does performance depend on transaction context and model availability? RQ4, what is the additional cost of fusion?

## 5.1 Experimental Setup

Datasets and view-specific models. We use an industrial payment-risk dataset and three public financial benchmarks. Industrial uses tabular, user-history, card-history, device-history, and graph models. TransXion [3] uses current-event, sender-history, receiver-history, and prior-event graph models for anti-moneylaundering prediction. IBM CCT [22] uses tabular, user-history, card-history, and prior-event graph models for card-fraud prediction. These tasks test the same fusion design with diferent sets of view-specific models.

MBD [19] evaluates next-month product purchase prediction at the client-month level. Its view-specific models describe client profiles, transaction activity, and geolocation activity, with historical inputs restricted by cutofs based on data generation time. Metrics are averaged equally over four binary targets. This task extends the evaluation to another financial prediction setting and a diferent model inventory. Table 4 reports dataset sizes and the model sets used for each task; Appendix A details the chronological splits.

Compared methods. We compare B1 (single-model reference), B5 (score-only fusion), Concat, Sum-pooling, Collaborative Experts (CE) [16], Multi-Modal Transformer (MMT) [6], UMI [35], I<sup>2</sup>M<sup>2</sup> [18], MMPareto [28], I<sup>2</sup>MoE [31], and BALM [21]. Concat, Sum-pooling, CE, MMT, and InterFuse receive the same representations ℎ, prediction scores �, availability indicators �, and input coverage �. We preserve each baseline’s core fusion structure and learning mechanisms while adapting it to the shared model outputs and risk-ranking task. In particular, I<sup>2</sup>MoE retains its interaction experts, weakly supervised objectives, and sample-dependent weighting. Appendix A.2 gives the adaptation details.

![](images/2d1d62ff370381ef6fe01177e99e32c41e39a904c2238d2c3e9fdc18118d3702.jpg)  
Figure 3: Ranking across review budgets. Shading shows the gap to the best plotted baseline at each budget; labels give 1% gains. Industrial: pp over B1; others: recall (%).

Training protocol. Within each dataset, all fusion methods use event-aligned outputs from the same view-specific model checkpoints. The upstream encoders and their original prediction heads remain frozen throughout fusion training. Only each method’s fu sion components and final prediction head are optimized. The key structural comparisons use direct prediction heads, with original model scores supplied as input features rather than additive reference logits. We select fusion checkpoints by validation Recall@1%, breaking ties by AP, AUC, and the earliest epoch. For the main comparison, we train each fusion method with seeds 17, 42, 202, 512, and 1024, selecting a checkpoint separately for each seed. View-specific model outputs, including B1 predictions, remain fixed across these runs. The component, budget, context, removal, and cost analyses retain the selected seed-42 models. All budgets in a sweep reuse the same checkpoint. Each analysis specifies the evaluation cohort used to compute the reported metrics.

Metrics. Recall@1% is the primary metric: the fraction of positive examples recovered in the top-scoring 1% of the evaluation population. AP and AUC summarize the full ranking. Budget analysis additionally uses 0.5%, 5%, 10%, and 25%. For a review fraction �, Industrial, TransXion, and IBM CCT select ⌊��⌋ events, ordering by descending score and then ascending event ID to break ties. MBD selects ⌈��⌉ client-months per target, preserves row order for ties, and averages recall equally over four targets. Public results are percentages. Industrial results are percentage-point diferences from the same fixed B1 reference at each budget. Table 1 reports means and standard deviations over the five fusion-training seeds; B1 has no fusion-training variation. Other tables and figures report the selected-model results described above.

## 5.2 Overall Ranking Performance (RQ1)

InterFuse achieves the highest mean Recall@1% on all four datasets (Table 1). It improves over the strongest baseline mean by 1.72 points on Industrial, 1.11 on TransXion, 8.09 on IBM CCT, and 3.33 on MBD. These comparisons use UMI on the three transaction tasks and BALM on MBD. The gains span diferent financial targets and view-specific model configurations under the same fusion design.

The gains also hold over methods that receive the same model information. On Industrial, InterFuse improves mean Recall@1% over Concat, CE, and MMT by 3.45–3.73 points. On IBM CCT, it reaches 61.46%, compared with 36.25–39.34% for these methods. The improvements over both feature concatenation and interactionbased baselines motivate the component comparisons in Section 5.3, which examine how model interaction and subsequent aggregation contribute to ranking.

InterFuse also achieves the highest mean AUC and AP on all four tasks. The changes in top-ranked recall and full-ranking metrics differ in magnitude: on MBD, the mean Recall@1% gain over Concat is 4.00 points, accompanied by a 0.030-point mean AP gain. Recall@1% directly captures the benefit at the review cutof, complementing the improvements in full-ranking quality.

The budget curves show where the advantage extends beyond the primary operating point (Figure 3). InterFuse leads the plotted methods at the 1% budget on all four tasks, at all five Industrial budgets, and at four of five budgets on IBM CCT and MBD. On the public tasks, some curves cross at wider budgets. The broad lead on Industrial therefore contrasts with budget-dependent method ordering on the public benchmarks.

Table 2: Component analysis, Recall@1%. Industrial: gains over B1 (pp); public datasets: percentages. Full / Both is the complete configuration.
<table><tr><td>Variant</td><td>Ind.</td><td>TX</td><td>CCT</td><td>MBD</td></tr><tr><td>InterFuse (Full / Both)</td><td>+5.44 89.69 61.46 10.20</td><td></td><td></td><td></td></tr><tr><td>Input information</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o representation h</td><td>+0.15</td><td>88.14</td><td>56.94</td><td>9.33</td></tr><tr><td>w/o score s</td><td>+2.59</td><td>89.02</td><td>59.18</td><td>9.76</td></tr><tr><td>w/o coverage c</td><td>+2.71</td><td>89.14</td><td>59.48</td><td>9.88</td></tr><tr><td>Cross-model interaction</td><td></td><td></td><td></td><td></td></tr><tr><td>Self-only routes (C1)</td><td>+2.46</td><td>61.64</td><td>38.20</td><td>9.21</td></tr><tr><td>Bridge-only</td><td>+2.69</td><td>89.14</td><td>59.88</td><td>9.78</td></tr><tr><td>Standard key mask</td><td>+4.94</td><td>89.14</td><td>60.40</td><td>10.07</td></tr><tr><td>Aggregation weights</td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform</td><td>+0.60</td><td>88.14</td><td>57.31</td><td>9.40</td></tr><tr><td>Static (global)</td><td>+1.93</td><td>88.91</td><td>58.15</td><td>9.66</td></tr><tr><td>w/o zf in weights</td><td>+2.60</td><td>89.47</td><td>59.07</td><td>10.13</td></tr><tr><td>Score/coverage entry points</td><td></td><td></td><td></td><td></td></tr><tr><td>Neither</td><td>+0.32</td><td>88.25</td><td>56.42</td><td>9.34</td></tr><tr><td>Token-only</td><td>+1.79</td><td>88.80</td><td>58.34</td><td>9.67</td></tr><tr><td>Aggregation-only</td><td>+1.85</td><td>89.02</td><td>59.74</td><td>9.71</td></tr></table>

(a) Coverage  
![](images/085ddf66be1297971a65ffc705762e5535198d3f4709f003a8e94f7e057e550d.jpg)

(b) Model disagreement  
![](images/e167e6a8b9576adc949b113ecf81cf5f0e02220dd298c8df6ab8af2ba31f099c.jpg)

(c) Review outcomes  
High-disagreement group
<table><tr><td></td><td>C1</td><td>Full</td><td>Δ</td></tr><tr><td>Positives</td><td>1,111</td><td>1,203</td><td>+92</td></tr><tr><td>Negatives</td><td>3,469</td><td>3,379</td><td>-90</td></tr><tr><td>Reviews</td><td>4,580</td><td>4,582</td><td>+2</td></tr></table>

Same global 1% review budget  
Figure 4: Full’s recall gains over (a) w/o coverage and Static across coverage groups and (b) C1 and Static across disagreement groups. Boxes show medians and interquartile ranges; whiskers follow the 1.5-IQR rule. (c) Full versus C1 review outcomes in the high-disagreement group.

## 5.3 Input and Component Analysis (RQ2)

The component comparisons examine model interaction, sharedstate-guided aggregation, and the information supplied to both stages (Table 2). Each variant is trained separately under the protocol in Section 5.1. Full / Both denotes the complete model. Table 5 reports the corresponding AP and AUC.

Cross-model updates improve on global fusion alone. C1 retains the fusion token and adaptive aggregation but restricts each model-token query to its own state. Its fusion query still reads all model tokens, so both Full and C1 use information from all available models at prediction time. Full additionally updates individual model states through cross-model interaction before aggregation. This improves Recall@1% by 2.97 points on Industrial with the same 565.25K fusion parameters, and by 23.26 points on IBM CCT. Updating individual model representations improves ranking even when a shared fusion state already summarizes all model outputs.

Bridge-only adds access to the shared fusion state, creating an indirect communication path across layers. This path recovers most of the gain over C1 on the two public transaction tasks: Bridge-only reaches 89.14% on TransXion and 59.88% on IBM CCT, compared with Full’s 89.69% and 61.46%. On Industrial, Full adds a further 2.74 points over Bridge-only. The benefit of direct model-to-model communication over shared-state bridging varies across tasks.

The shared fusion state improves transaction-adaptive weighting. Uniform averages available model states, whereas Static learns one global weight per model and renormalizes over available outputs. Static improves on Uniform on all four tasks. Full further improves over Static by 3.51 points on Industrial and 3.31 points on IBM CCT, showing the value of transaction-dependent weights beyond a global model preference.

The $\mathbf { w } / \mathbf { o } \ z _ { f }$ in weights variant is a dynamic-weight control: it uses the same scoring MLP with input [0 ∥ � ], so its weights depend on $u _ { r }$ alone. The interaction layers, $u _ { r } = \phi _ { r } ( [ z _ { r } \parallel s _ { r } \parallel c _ { r } ] )$ and $z _ { f }$ in the final prediction head remain unchanged. The control therefore retains cross-model information in $u _ { r }$ while removing only the weighting network’s explicit shared-state input. Full adds 2.83 points on Industrial and 2.39 points on IBM CCT, with positive gains on TransXion and MBD as well. Explicit shared-state conditioning improves weighting even when the model-specific inputs already contain cross-model information.

Original predictions and coverage help at both stages. Token-only and Aggregation-only both expose predictions and input coverage to the fusion network, but at diferent stages. Token-only supplies $( s , c )$ before interaction, Aggregation-only supplies them directly to aggregation, and Neither removes both entries. Full outperforms both single-entry variants on all four tasks; the Industrial gains are 3.65 and 3.58 points, respectively. Direct access to original predictions and input coverage further improves ranking after these fields have already participated in interaction.

Input fields and missing-state communication support fusion. Removing model representations ℎ lowers Recall@1% by 5.28 points on Industrial and 4.53 points on IBM CCT, showing the value of learned features beyond original predictions. Scores � and coverage � also improve recall on every task. Full gains 2.73 points over w/o coverage on Industrial, with availability handling retained in both configurations.

Standard key mask blocks unavailable keys for every query, disabling both exceptions in Equation 3: fusion access to missing states and self-connections for missing states. Full exceeds this control by 0.50 points on Industrial and 1.07 points on IBM CCT. This comparison supports the joint missing-state communication rule. Section 5.5 examines its contribution as more outputs become unavailable.

## 5.4 Analysis across Transaction Contexts (RQ3)

Interaction gains are larger when model predictions disagree on Industrial (Figure 4). Disagreement is the standard deviation of available models’ training-referenced percentile scores. All methods share group assignments and independently construct a global top-1% review queue. We report pointwise 95% percentile confidence intervals from 2,000 paired user-level bootstrap replicates. Appendix A.3 details the grouping and resampling procedures.

Full’s gain over C1 increases from 1.46 points in the lowdisagreement group to 4.87 in the high-disagreement group. The high-minus-low contrast is 3.41 points, with confidence interval [1.92, 4.43]. Full–Static gains also rise, from 2.75 to 4.02 points, with a high-minus-low contrast of 1.27 points and interval [0.09, 2.47]. Both comparisons link the value of interaction and adaptive aggregation to diferences among model predictions.

Under the same global review budget, Full allocates 4,582 reviews to the high-disagreement group and recovers 1,203 positives; C1 allocates 4,580 and recovers 1,111 (Figure 4c). Within this subgroup, Full therefore finds 92 more positives with two additional reviews. The result connects the interaction gain to improved recovery at nearly identical subgroup review allocations.

Input coverage is most useful when histories are sparse. We measure history coverage by the total valid sequence length across the user-, card-, and device-history models. Full’s gain over w/o coverage is 4.61 points in the low-coverage group and 1.58 in the highcoverage group. Their high-minus-low contrast is −3.03 points, with interval [−3.96, −2.03]. Adaptive aggregation also improves recall across coverage groups, with Full–Static gains of 3.45–3.68 points; their high-minus-low contrast is −0.23 with interval [−1.37, 0.51].

## 5.5 Robustness to Missing Model Outputs (RQ3)

We remove user-history, card-history, device-history, and graph outputs separately and jointly, keeping the current-transaction model available. Checkpoints remain fixed. Removal masks ℎ, �, �, and direct aggregation inputs and updates availability. Rates apply to originally available outputs; joint removal independently masks each optional output with probability �.

Ranking gains persist across single-model removals. Inter-Fuse exceeds clean B1 by 2.86–3.80 points across all four singlemodel removals and UMI by at least 1.45 points (Figure 5a). These gains are obtained without retraining, showing that the learned fusion function remains useful when the available model set changes. Removing user history or graph has the largest efect, indicating diferent ranking sensitivities to the missing views. Table 6 reports results for all compared methods at intermediate removal rates.

Table 3: Industrial fusion cost and Recall@1% gains over B1. Batch-1 latency excludes upstream encoders.
<table><tr><td>Method</td><td>Params. (K)</td><td>p50 (ms)</td><td>p95 (ms)</td><td>∆R@1 (pp)</td></tr><tr><td>Concat</td><td>454.66</td><td>1.147</td><td>1.323</td><td>+1.99</td></tr><tr><td>Sum-pooling</td><td>442.26</td><td>1.087</td><td>1.167</td><td>+1.81</td></tr><tr><td>CE</td><td>410.11</td><td>11.696</td><td>11.989</td><td>+1.71</td></tr><tr><td>MMT</td><td>508.42</td><td>4.454</td><td>4.622</td><td>+1.96</td></tr><tr><td>UMI</td><td>294.43</td><td>1.314</td><td>1.334</td><td>+3.72</td></tr><tr><td>I²M2</td><td>452.49</td><td>2.614</td><td>2.812</td><td>+2.87</td></tr><tr><td>MMPareto</td><td>343.43</td><td>1.336</td><td>2.123</td><td>+0.50</td></tr><tr><td>I²MoE</td><td>1,077.79</td><td>5.074</td><td>6.497</td><td>+1.48</td></tr><tr><td>BALM</td><td>508.04</td><td>2.560</td><td>2.607</td><td>+1.27</td></tr><tr><td>C1</td><td>565.25</td><td>2.808</td><td>3.277</td><td>+2.46</td></tr><tr><td>InterFuse</td><td>565.25</td><td>3.044</td><td>3.570</td><td>+5.44</td></tr></table>

(a) Complete single-model removal
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>User</td><td rowspan=1 colspan=3>Card DeviceGraph</td><td rowspan=6 colspan=1>pp4-0</td></tr><tr><td rowspan=1 colspan=1>• InterFuse</td><td rowspan=1 colspan=1>+3.09</td><td rowspan=1 colspan=1>+3.76</td><td rowspan=1 colspan=1>+3.80</td><td rowspan=1 colspan=1>+2.86</td></tr><tr><td rowspan=1 colspan=1>UMI</td><td rowspan=1 colspan=1>+1.19</td><td rowspan=1 colspan=1>+2.31</td><td rowspan=1 colspan=1>+1.87</td><td rowspan=1 colspan=1>+0.81</td></tr><tr><td rowspan=1 colspan=1>一C1</td><td rowspan=1 colspan=1>-0.42</td><td rowspan=1 colspan=1>+0.28</td><td rowspan=1 colspan=1>+0.61</td><td rowspan=1 colspan=1>-0.66</td></tr><tr><td rowspan=1 colspan=1>Concat</td><td rowspan=1 colspan=1>+0.43</td><td rowspan=1 colspan=1>+0.79</td><td rowspan=1 colspan=1>+0.86</td><td rowspan=1 colspan=1>+0.06</td></tr><tr><td rowspan=1 colspan=1>MMT</td><td rowspan=1 colspan=1>-2.24</td><td rowspan=1 colspan=1>-1.12</td><td rowspan=1 colspan=1>-0.97</td><td rowspan=1 colspan=1>-2.29</td></tr></table>

Recall@1% gain over clean B1 (pp)

## (b) Joint removal

![](images/b8091a75d9203b8c44bab13afd91e7086e38b00e7a5beb77db6f565095b04a9c.jpg)  
Figure 5: Industrial output removal. (a) Complete removal of each optional model. (b) Increasing joint removal. Values are Recall@1% gains over clean B1 (pp).

The relative advantage persists as joint removal increases. InterFuse remains ahead throughout joint removal (Figure 5b), retaining gains over clean B1 of 2.91 points at 50% and 0.72 at 75%. The remaining optional outputs still improve ranking over clean B1 at these removal rates. With all optional outputs removed, gains are −1.26 points for InterFuse, −2.18 for Standard key mask, and −3.72 for UMI. The results distinguish two efects: complementary outputs improve absolute recall, while the fusion design matters under the same reduced inputs.

Standard key mask disables fusion-token access to missing states and missing-state self-connections while preserving the rest of the fusion design. Full exceeds it at every reported condition, by 0.63 points at 50% joint removal. This supports missing-state communication during interaction, beyond excluding unavailable outputs from the final weighting and pooling steps.

## 5.6 Computational Cost (RQ4)

All methods use the same execution settings; latency is measured at batch size 1 over the fusion network only. InterFuse achieves its Industrial ranking gains with 565.25K fusion parameters and 3.044 ms median fusion latency (Table 3). Relative to C1, it improves Recall@1% by 2.97 points at the same parameter count, with an 8.4% latency increase over 2.808 ms. These measurements isolate the extra computation required for fusion of existing model outputs. Attention operates on one token per model and a fusion token, so its sequence length depends on the model inventory rather than raw history length or graph size. The Industrial configuration therefore processes six tokens per transaction.

InterFuse also has higher recall and lower median latency than CE, MMT, and I<sup>2</sup>MoE. Compared with UMI, it gains 1.71 recall points at a higher median latency; UMI uses 294.43K parameters and 1.314 ms. The table makes this quality–cost choice explicit across all eleven methods, with upstream encoder computation excluded from the timing boundary. Figure 7 in Appendix A.9 plots ranking gains against fusion latency.

## 6 Conclusion

In this work, we studied how to fuse models trained on diferent transaction views. We proposed InterFuse, which updates model representations through availability-aware attention and uses a shared fusion state to guide adaptive aggregation. Aggregation retains original predictions and input coverage to determine model contributions. Experiments demonstrate consistent improvements in risk ranking across industrial and public benchmarks. Our anal yses show that updating individual model representations adds value beyond global summarization. Aggregation further benefits from explicit access to the shared fusion state, original predictions, and input coverage after interaction. Industrial analyses find larger interaction gains when model predictions disagree. Future work could explore how this approach can help integrate existing model capabilities into payment-risk foundation models.

## References

[1] Rachit Bansal, Bidisha Samanta, Siddharth Dalmia, Nitish Gupta, Sriram Ganapathy, Abhishek Bapna, Prateek Jain, and Partha Talukdar. 2024. LLM Augmented LLMs: Expanding Capabilities through Composition. In ICLR.

[2] Abhra Chaudhuri, Anjan Dutta, Tu Bui, and Serban Georgescu. 2025. A Closer Look at Multimodal Representation Collapse. In ICML, PMLR 267, 7555–7577.

[3] Keyang Chen, Mingxuan Jiang, Yongsheng Zhao, Zeping Li, Zaiyuan Chen, Weiqi Luo, Zhixin Li, Sen Liu, Yinan Jing, Guangnan Ye, Xihong Wu, and Hongfeng Chai. 2026. TransXion: A High-Fidelity Graph Benchmark for Realistic Anti-Money Laundering. In KDD, V.2, 8707–8718.

[4] Xinyan Chen and Jianfei Yang. 2025. X-Fi: A Modality-Invariant Foundation Model for Multimodal Human Sensing. In ICLR.

[5] Ruiting Dai, Chenxi Li, Yandong Yan, Lisi Mo, Ke Qin, and Tao He. 2025. Unbiased Missing-modality Multimodal Learning. In ICCV, 24507–24517.

[6] Valentin Gabeur, Chen Sun, Karteek Alahari, and Cordelia Schmid. 2020. Multi modal Transformer for Video Retrieval. In ECCV, 214–229.

[7] Yury Gorishniy, Akim Kotelnikov, and Artem Babenko. 2025. TabM: Advancing Tabular Deep Learning with Parameter-Eficient Ensembling. In ICLR.

[8] Zhi-Hao Guan, Longfei Huang, and Yang Yang. 2026. Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning. In CVPR, 37896–37905.

[9] Konstantin Hemker, Nikola Simidjievski, and Mateja Jamnik. 2024. HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data. In NeurIPS 37.

[10] Konstantin Hemker, Nikola Simidjievski, and Mateja Jamnik. 2025. Multimodal Lego: Model Merging and Fine-Tuning Across Topologies and Modalities in Biomedicine. In ICLR

[11] Guanzhou Ke, Shengfeng He, Xiaoli Wang, Bo Wang, Guoqing Chao, Yuanyang Zhang, Yi Xie, and Hexing Su. 2025. Knowledge Bridger: Towards Training-Free Missing Modality Completion. In CVPR, 25864–25873.

[12] Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell. 2017. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. In NeurIPS 30.

[13] Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, and Yee Whye Teh. 2019. Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. In ICML, PMLR 97, 3744–3753.

[14] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023. BLIP-2: Bootstrap ping Language-Image Pre-training with Frozen Image Encoders and Large Lan guage Models. In ICML, PMLR 202, 19730–19742.

[15] Zhongyang Li, Ziyue Li, and Tianyi Zhou. 2025. R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts. In ICML, PMLR 267, 35292–35316.

[16] Yang Liu, Samuel Albanie, Arsha Nagrani, and Andrew Zisserman. 2019. Use What You Have: Video Retrieval Using Representations From Collaborative Experts. In BMVC. Corrected version, February 2020.

[17] Mingxuan Lu, Zhichao Han, Susie Xi Rao, Zitao Zhang, Yang Zhao, Yinan Shan, Ramesh Raghunathan, Ce Zhang, and Jiawei Jiang. 2022. BRIGHT – Graph Neural Networks in Real-Time Fraud Detection. In CIKM, 3342–3351.

[18] Divyam Madaan, Taro Makino, Sumit Chopra, and Kyunghyun Cho. 2024. Jointly Modeling Inter- & Intra-Modality Dependencies for Multi-modal Learning. In NeurIPS 37.

[19] Dzhambulat Mollaev, Ivan Kireev, Mikhail Orlov, Alexander Kostin, Ivan Karpukhin, Maria Postnova, Gleb Gusev, and Andrey Savchenko. 2025. Multimodal Banking Dataset: Understanding Client Needs through Event Sequences. In CIKM, 6476–6480.

[20] Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, and Chen Sun. 2021. Attention Bottlenecks for Multimodal Fusion. In NeurIPS 34.

[21] Phuong-Anh Nguyen, Tien Anh Pham, Duc-Trong Le, and Cam-Van Thi Nguyen. 2026. BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates. In CVPR, 30246–30256.

[22] Inkit Padhi, Yair Schif, Igor Melnyk, Mattia Rigotti, Youssef Mroueh, Pierre Dognin, Jerret Ross, Ravi Nair, and Erik Altman. 2021. Tabular Transformers for Modeling Multivariate Time Series. In ICASSP, 3565–3569.

[23] Ivan Rubachev, Nikolay Kartashev, Yury Gorishniy, and Artem Babenko. 2025. TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Bench marks. In ICLR.

[24] Min Shi, Fuxiao Liu, Shihao Wang, Shijia Liao, Subhashree Radhakrishnan, Yilin Zhao, De-An Huang, Hongxu Yin, Karan Sapra, Yaser Yacoob, Humphrey Shi, Bryan Catanzaro, Andrew Tao, Jan Kautz, Zhiding Yu, and Guilin Liu. 2025. Eagle: Exploring the Design Space for Multimodal LLMs with Mixture of Encoders. In ICLR.

[25] Vinitra Swamy, Malika Satayeva, Jibril Frej, Thierry Bossy, Thijs Vogels, Martin Jaggi, Tanja Käser, and Mary-Anne Hartley. 2023. MultiMoDN: Multimodal, Multi-Task, Interpretable Modular Networks. In NeurIPS 36

[26] Yao-Hung Hubert Tsai, Shaojie Bai, Paul Pu Liang, J. Zico Kolter, Louis-Philippe Morency, and Ruslan Salakhutdinov. 2019. Multimodal Transformer for Un aligned Multimodal Language Sequences. In ACL, 6558–6569.

[27] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In NeurIPS 30.

[28] Yake Wei and Di Hu. 2024. MMPareto: Boosting Multimodal Learning with Innocent Unimodal Assistance. In ICML, PMLR 235, 52559–52572.

[29] David H. Wolpert. 1992. Stacked Generalization. Neural Networks 5, 2, 241–259.

[30] Chenwei Wu, Zitao Shuai, Zhengxu Tang, Luning Wang, and Liyue Shen. 2025. Dynamic Modeling of Patients, Modalities and Tasks via Multi-modal Multi-task Mixture of Experts. In ICLR

[31] Jiayi Xin, Sukwon Yun, Jie Peng, Inyoung Choi, Jenna L. Ballard, Tianlong Chen, and Qi Long. 2025. I<sup>2</sup>MoE: Interpretable Multimodal Interaction-aware Mixture of-Experts. In ICML, PMLR 267, 68870–68888.

[32] Junshen Xu, Jiayun Zhang, and Yi Fan. 2026. Multi-Domain Marker Aggregation for Threat Detection in Cloud Environments. In WWW (Industry Track).

[33] Jie Yang, Rui Zhang, Ziyang Cheng, Dawei Cheng, Guang Yang, and Bo Wang. 2025. Grad: Guided Relation Difusion Generation for Graph Augmentation in Graph Fraud Detection. In WWW, 5308–5319.

[34] Manzil Zaheer, Satwik Kottur, Siamak Ravanbakhsh, Barnabas Póczos, Ruslan Salakhutdinov, and Alexander J. Smola. 2017. Deep Sets. In NeurIPS 30.

[35] Yunhua Zhang, Hazel Doughty, and Cees G. M. Snoek. 2023. Learning Unseen Modality Interaction. In NeurIPS 36.

[36] Yan Zhuang, Minhao Liu, Wei Bai, Yanru Zhang, Wei Li, Jiawen Deng, and Fuji Ren. 2025. Hyper-Modality Enhancement for Multimodal Sentiment Analysis with Missing Modalities. In NeurIPS 38

## A Additional Experimental Details

## A.1 Datasets and Chronological Splits

Transaction tasks. Industrial, TransXion, and IBM CCT evaluate event-level risk ranking. Their Test sets contain 7,872, 902, and 2,717 positive events, respectively; the corresponding Validation counts are 7,811, 779, and 2,116. TransXion and IBM CCT contain 2,960 and 24,924 training positives. Table 4 summarizes the tasks, sample counts, positive rates, and model sets.

Temporal inputs and model fitting. The transaction benchmarks use chronological splits, reflecting the deployment conditions emphasized by TabReD [23]. Sequence and graph inputs precede the scored event. Public datasets use separate partitions for model and fusion fitting; Validation selects checkpoints for Test evaluation. All fusion methods use the same event-level outputs from encoders and prediction heads that remain fixed throughout fusion training.

MBD prediction task. The trx/geo version predicts next-month purchases for four client-month targets using cutofs based on data generation time, with equally averaged metrics. Positive counts are 4,267, 386, 3,494, and 2,095 for Train; 644, 110, 576, and 473 for Validation; and 451, 81, 460, and 492 for Test. These targets describe product purchases, extending the evaluation beyond fraud detection.

MLPs encode profile features and fixed summaries of strictly prior transaction and geolocation events into 64-dimensional representations. Model fitting uses the first four training months; fusion fitting uses the next four, with three epochs per stage. Validation covers October–November 2022; Test covers December 2022–January 2023. Each target uses a separate head and a fixed checkpoint across all review budgets.

## A.2 Baseline Adaptations

Common prediction setting. B1 is the single-model reference; B5 performs score fusion. Structural comparisons receive aligned representations, scores, availability, and coverage, using model-specific projections for diferent input dimensions and direct prediction heads. Checkpoint selection follows Section 5.1.

Simple feature fusion. Concat joins projected model outputs; Sum-pooling sums available projections. Both are simple alternatives to cross-model interaction and adaptive aggregation.

Table 4: Tasks, data splits, and view-specific models. Public Train counts include model and fusion fitting. Positive rates refer to Test; the range for MBD covers its four product-purchase targets.
<table><tr><td>Dataset</td><td>Task / unit</td><td>Train</td><td>Validation</td><td>Test</td><td>Positive rate</td><td>View-specific models</td></tr><tr><td>Industrial</td><td>Payment risk / event</td><td>5,832,600</td><td>1,392,465</td><td>1,366,237</td><td>0.576%</td><td>Tabular; user, card, device sequences; graph</td></tr><tr><td>TransXion</td><td>AML / event</td><td>1,817,503</td><td>605,834</td><td>605,833</td><td>0.149%</td><td>Tabular; sender, receiver sequences; graph</td></tr><tr><td>IBM CCT</td><td>Card fraud /</td><td>19,512,454</td><td>2,436,496</td><td>2,437,950</td><td>0.111%</td><td>Tabular; user, card sequences; graph</td></tr><tr><td>MBD (trx/geo)</td><td>event Product purchase /</td><td>801,792</td><td>200,448</td><td>200,448</td><td>0.040- 0.245%</td><td>Profile; transaction and geolocation summaries</td></tr></table>

Collaborative Experts and MMT. CE [16] uses available-pair averaging, shared relation networks, local-plus-relation gates, and a fixed expert-update order. MMT [6] uses aggregate and feature tokens for each model summary, learned model identities and posi tion embeddings, and two post-normalization Transformer layers with four heads. Unavailable keys are masked and their output embeddings zeroed. Prediction retains expert aggregate embeddings, without InterFuse’s adaptive pooling.

Both methods classify concatenated normalized expert outputs and availability indicators. Binary heads and cross-entropy replace the retrieval components, without video or language pretraining. MBD uses four heads. MMT initializes each aggregate token from the exported model summary vector.

UMI and interaction-based alternatives. UMI [35] sums available outputs projected into a common space; a second branch learns from reliable unimodal pseudo-labels. Model-specific projections replace token reorganization. I<sup>2</sup>M<sup>2</sup> [18] integrates joint and available individual predictors, extending its two-modality factorization to the model inventory. I<sup>2</sup>MoE [31] uses individual-model, model-pair, and complete-set experts for diferent interaction types.

Training-oriented alternatives. MMPareto [28] balances joint and model-specific gradients. BALM [21] uses feature calibration and gradient rebalancing for unequal availability. Both act on the fusion network, keeping upstream encoders fixed.

## A.3 Transaction Context Groups

History coverage. The user-, card-, and device-history models each use events from the 90 days preceding the current transaction, retaining the most recent 300 events when this limit is exceeded. History coverage is the sum of the three valid sequence lengths, excluding padding. Empty histories contribute zero. An event appearing in multiple histories is counted separately because it contributes to each model’s input. The aggregate ranges from 0 to 900. Low, Medium, and High coverage correspond to fewer than 50, 50–300, and more than 300 history entries, respectively.

Prediction disagreement. We use the original prediction scores from the tabular, user-history, card-history, device-history, and graph models. For each model, we construct an empirical score distribution from its available outputs on the training reference set.

Each score is mapped to its percentile in this distribution, placing predictions on a common scale of relative risk. For a transaction �, disagreement is

$$
\begin{array} { l } { \displaystyle { D ( x ) = \sqrt { \frac { 1 } { m _ { x } } \sum _ { r \in \mathcal { A } ( x ) } \left( q _ { r } ( x ) - \bar { q } ( x ) \right) ^ { 2 } } , } } \\ { \displaystyle { \bar { q } ( x ) = \frac { 1 } { m _ { x } } \sum _ { r \in \mathcal { A } ( x ) } q _ { r } ( x ) , } } \end{array}\tag{8}
$$

where ${ \mathcal { A } } ( x )$ is the set of available models, $m _ { x } = \vert \mathcal { A } ( x ) \vert$ , and �<sub>�</sub> (�) is model $r ' s$ percentile score. Unavailable outputs are excluded.

We determine cutofs $d _ { 1 }$ and $d _ { 2 }$ from the one-third and twothirds quantiles of training-reference disagreement values, using transactions with at least two available models. The groups are Low $\left( D \leq d _ { 1 } \right)$ , Medium $( d _ { 1 } < D \le d _ { 2 } )$ , and High $\left( D > d _ { 2 } \right)$ . The reference distributions and cutofs remain fixed during evaluation and bootstrap resampling. Equal disagreement values remain in the same group. With fewer than two available models, a transaction stays in the global ranking but has no disagreement group.

Shared grouping and review allocation. All fusion methods use the same groups, defined from the view-specific models’ original inputs and predictions. The Industrial cohort contains 1,366,237 events and 7,872 positives. Each method independently selects its global top-1% queue of 13,662 events; no separate budget is assigned to individual groups. Group recall is the fraction of a group’s positive events recovered in the global review queue.

Bootstrap confidence intervals. We use 2,000 paired bootstrap replicates, sampling users with replacement and retaining all transactions associated with each sampled user. Each replicate uses the same sampled records and multiplicities across methods and groups. Model checkpoints, reference score distributions, and group cutofs remain fixed. We reconstruct each method’s global top-1% review queue within each replicate and compute group-specific recall gains. For each comparison, the high-minus-low contrast is the gain in the High group minus that in the Low group. We report twosided, pointwise 95% percentile confidence intervals using the 2.5th and 97.5th percentiles of the bootstrap contrasts. These intervals describe evaluation-sample uncertainty for the fixed models.

## A.4 Additional Metrics for the Component Analysis

Table 5: Additional AP and AUC for the component variants in Table 2. Industrial reports gains over B1 (pp); public datasets report percentages. Each value is a point estimate from the selected model for that variant.
<table><tr><td></td><td colspan="2">Industrial (∆pp)</td><td colspan="2">TransXion (%)</td><td colspan="2">IBM CCT (%)</td><td colspan="2">MBD (%)</td></tr><tr><td>Variant</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td><td>AUC</td><td>AP</td></tr><tr><td>InterFuse (Full / Both)</td><td>+1.95</td><td>+3.10</td><td>99.15</td><td>28.71</td><td>96.82</td><td>22.36</td><td>77.80</td><td>0.677</td></tr><tr><td>Input information</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o representation h</td><td>+1.75</td><td>-1.96</td><td>99.06</td><td>25.99</td><td>96.63</td><td>20.92</td><td>77.03</td><td>0.621</td></tr><tr><td>w/o score s</td><td>+1.86</td><td>+0.52</td><td>99.11</td><td>27.33</td><td>96.74</td><td>21.60</td><td>77.44</td><td>0.649</td></tr><tr><td>w/o coverage c</td><td>+1.87</td><td>+1.06</td><td>99.11</td><td>27.65</td><td>96.75</td><td>21.74</td><td>77.45</td><td>0.654</td></tr><tr><td colspan="2">Cross-model interaction</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Self-only routes (C1)</td><td>+1.72</td><td>+2.67</td><td>94.02</td><td>18.32</td><td>95.04</td><td>5.23</td><td>76.72</td><td>0.609</td></tr><tr><td>Bridge-only</td><td>+1.86</td><td>+0.80</td><td>99.11</td><td>27.48</td><td>96.74</td><td>21.69</td><td>77.47</td><td>0.652</td></tr><tr><td>Standard key mask</td><td>+1.86</td><td>+2.85</td><td>99.14</td><td>28.66</td><td>96.67</td><td>21.26</td><td>77.35</td><td>0.612</td></tr><tr><td colspan="2">Aggregation weights</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Uniform</td><td>+1.76</td><td>-1.62</td><td>99.07</td><td>26.18</td><td>96.64</td><td>21.01</td><td>77.08</td><td>0.624</td></tr><tr><td>Static (global)</td><td>+1.83</td><td>+0.03</td><td>99.10</td><td>27.12</td><td>96.71</td><td>21.49</td><td>77.28</td><td>0.643</td></tr><tr><td>w/o zf in weights</td><td>+1.91</td><td>+2.44</td><td>99.12</td><td>28.31</td><td>96.78</td><td>22.27</td><td>77.60</td><td>0.666</td></tr><tr><td colspan="2">Score/coverage entry points</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Neither</td><td>+1.74</td><td>-2.08</td><td>99.06</td><td>25.96</td><td>96.62</td><td>20.88</td><td>77.01</td><td>0.620</td></tr><tr><td>Token-only</td><td>+1.83</td><td>+0.01</td><td>99.10</td><td>27.12</td><td>96.72</td><td>21.49</td><td>77.37</td><td>0.644</td></tr><tr><td>Aggregation-only</td><td>+1.84</td><td>+0.18</td><td>99.10</td><td>27.18</td><td>96.72</td><td>21.52</td><td>77.39</td><td>0.645</td></tr></table>

## A.5 Fusion Performance across Model Subsets

![](images/d74dda8631924718b1d48cccd685d43fbfecc794108c4a426bcb289825221627.jpg)

Figure 6: Fusion performance across model subsets. Training and evaluation use the same subset; shading marks all models. Tab.: tabular; Seq.: sequences; +Trx/+Geo: profile plus transaction/geolocation summaries.

Figure 6 uses a separately trained fusion model for each subset, with upstream encoders fixed. Unlike Figure 5, the subset is unchanged between training and evaluation.

The complete model set gives InterFuse its highest recall on each dataset and a larger advantage over C1 than either two-source subset. Performance varies with the model subset: UMI achieves higher recall with tabular and graph models on all three transaction datasets.

## A.6 Training and Inference Algorithm

Algorithm 1: Training and Inference of InterFuse   
Input: Aligned model outputs $\mathcal { E } ( x ) = \{ ( h _ { r } , s _ { r } , a _ { r } , c _ { r } ) \} _ { r = 1 } ^ { R } ;$   
fusion-training labels; validation data; depth �.   
Output: Selected fusion parameters �<sup>★</sup> and ranking logits.   
// Shared forward computation (Eqs. 1–6)   
1 function Forward(E (� ); �):   
2 A ← {� : �<sub>�</sub> = 1} (assumed nonempty)   
3 $z _ { r } ^ { 0 }  \rho _ { r } + P _ { r } ( [ h _ { r } \parallel s _ { r } \parallel c _ { r } ] ) , r \in \mathcal { A }$   
4 �<sup>0</sup> ← <sub>��</sub> + �<sub>�</sub>, � ∉ A   
5 $Z ^ { 0 } \gets [ z _ { f } ^ { 0 } ; z _ { 1 } ^ { 0 } ; \dots ; z _ { R } ^ { 0 } ]$   
6 Build �(�) by Eq. 3, retaining fusion and self access   
7 for $\ell = 0 , \ldots , L - 1$ do   
8 �<sup>ℓ+1</sup> ← pre-normalized interaction block applied to   
(�<sup>ℓ</sup>, � (� ) ) (Eq. 4)   
9 � , � , . . . , �<sub>�</sub> ← token states in �<sup>�</sup>   
10 �<sub>�</sub> ← �<sub>�</sub> ( [�<sub>�</sub> ∥ �<sub>�</sub> ∥ �<sub>�</sub> ] ), � ∈ A   
11 �<sub>A</sub> ← softmax<sub>�∈A</sub>  �( [�<sub>�</sub> ∥ �<sub>�</sub> ] )   
12 � ← Í<sub>�∈A</sub> �<sub>�</sub>�<sub>�</sub>   
13 return � ( [�<sub>�</sub> ∥ � ] )   
// Fusion training; view-specific models stay fixed   
14 Initialize fusion parameters �   
15 foreach training epoch do   
16 foreach minibatch B do   
17 $f _ { \theta } ( x ) \gets \mathsf { F o r w a r d } ( \mathcal { E } ( x ) ; \theta ) , x \in \mathcal { B }$   
18 $\begin{array} { r } { \mathcal { L } _ { \mathcal { B } }  \frac { 1 } { \vert \mathcal { B } \vert } \sum _ { x \in \mathcal { B } } \ell _ { \mathrm { B C E } } ( f _ { \theta } ( x ) , y _ { x } ) } \end{array}$   
19 Update only � using L<sub>B</sub> (Eq. 7)   
20 Save the epoch checkpoint and validation metrics   
21 Select �<sup>★</sup> using the validation rule in Sec. 5.1   
// Inference; one fusion forward pass   
22 return Forward( E (�); �<sup>★</sup>) as the ranking logit

## A.7 Complete Industrial Output-Removal Results

Table 6: Complete Industrial output-removal results. Entries are Recall@1% gains over the same clean B1 reference (pp). Removal rates apply to originally available outputs. Mask-level dispersion is not reported.
<table><tr><td>View</td><td>Method</td><td>0%</td><td>25%</td><td>50%</td><td>75%</td><td>100%</td></tr><tr><td>User-history</td><td>InterFuse</td><td>+5.44</td><td>+5.16</td><td>+4.47</td><td>+3.55</td><td>+3.09</td></tr><tr><td></td><td>Standard key mask</td><td>+4.94</td><td>+4.63</td><td>+3.71</td><td>+2.88</td><td>+2.05</td></tr><tr><td></td><td>C1</td><td>+2.46</td><td>+2.08</td><td>+1.24</td><td>+0.31</td><td>-0.42</td></tr><tr><td></td><td>Concat</td><td>+1.99</td><td>+1.43</td><td>+0.89</td><td>+0.57</td><td>+0.43</td></tr><tr><td></td><td>MMT</td><td>+1.96</td><td>+1.61</td><td>+0.53</td><td>-1.12</td><td>-2.24</td></tr><tr><td></td><td>UMI</td><td>+3.72</td><td>+3.39</td><td>+2.82</td><td>+1.96</td><td>+1.19</td></tr><tr><td>Card-history</td><td>InterFuse</td><td>+5.44</td><td>+4.73</td><td>+4.20</td><td>+3.91</td><td>+3.76</td></tr><tr><td></td><td>Standard key mask</td><td>+4.94</td><td>+4.26</td><td>+3.58</td><td>+3.12</td><td>+2.53</td></tr><tr><td></td><td>C1</td><td>+2.46</td><td>+1.63</td><td>+1.07</td><td>+0.52</td><td>+0.28</td></tr><tr><td></td><td>Concat</td><td>+1.99</td><td>+1.88</td><td>+1.56</td><td>+1.03</td><td>+0.79</td></tr><tr><td></td><td>MMT</td><td>+1.96</td><td>+0.91</td><td>+0.14</td><td>-0.64</td><td>-1.12</td></tr><tr><td></td><td>UMI</td><td>+3.72</td><td>+3.11</td><td>+2.67</td><td>+2.45</td><td>+2.31</td></tr><tr><td>Device-history</td><td>InterFuse</td><td>+5.44</td><td>+5.32</td><td>+4.98</td><td>+4.29</td><td>+3.80</td></tr><tr><td></td><td>Standard key mask</td><td>+4.94</td><td>+4.72</td><td>+4.31</td><td>+3.61</td><td>+2.85</td></tr><tr><td></td><td>C1</td><td>+2.46</td><td>+2.31</td><td>+1.82</td><td>+1.04</td><td>+0.61</td></tr><tr><td></td><td>Concat</td><td>+1.99</td><td>+1.60</td><td>+1.26</td><td>+1.03</td><td>+0.86</td></tr><tr><td></td><td>MMT</td><td>+1.96</td><td>+1.69</td><td>+1.12</td><td>-0.18</td><td>-0.97</td></tr><tr><td></td><td>UMI</td><td>+3.72</td><td>+3.66</td><td>+3.18</td><td>+2.41</td><td>+1.87</td></tr><tr><td>Graph</td><td>InterFuse</td><td>+5.44</td><td>+4.39</td><td>+3.69</td><td>+3.17</td><td>+2.86</td></tr><tr><td></td><td>Standard key mask</td><td>+4.94</td><td>+3.91</td><td>+3.06</td><td>+2.34</td><td>+1.69</td></tr><tr><td></td><td>C1</td><td>+2.46</td><td>+1.52</td><td>+0.86</td><td>+0.15</td><td>-0.66</td></tr><tr><td></td><td>Concat</td><td>+1.99</td><td>+1.77</td><td>+1.16</td><td>+0.41</td><td>+0.06</td></tr><tr><td></td><td>MMT</td><td>+1.96</td><td>+0.64</td><td>-0.83</td><td>-1.76</td><td>-2.29</td></tr><tr><td></td><td>UMI</td><td>+3.72</td><td>+2.66</td><td>+1.77</td><td>+1.18</td><td>+0.81</td></tr><tr><td>Joint</td><td>InterFuse</td><td>+5.44</td><td>+4.58</td><td>+2.91</td><td>+0.72</td><td>-1.26</td></tr><tr><td></td><td>Standard key mask</td><td>+4.94</td><td>+3.81</td><td>+2.28</td><td>-0.41</td><td>-2.18</td></tr><tr><td></td><td>C1</td><td>+2.46</td><td>+0.71</td><td>-1.63</td><td>-4.17</td><td>-5.58</td></tr><tr><td></td><td>Concat</td><td>+1.99</td><td>+0.82</td><td>-0.91</td><td>-3.06</td><td>-4.62</td></tr><tr><td></td><td>MMT</td><td>+1.96</td><td>-0.76</td><td>-3.48</td><td>-5.12</td><td>-6.44</td></tr><tr><td></td><td>UMI</td><td>+3.72</td><td>+2.89</td><td>+1.12</td><td>-1.54</td><td>-3.72</td></tr></table>

Table 6 adds intermediate removal rates to Figure 5. Initial avail ability is 68.72% for user history, 58.40% for card history, 62.03% for device history, and 44.69% for graph outputs. A 100% rate removes all originally available outputs of the named model.

The fusion checkpoint remains fixed. Removal updates availability and suppresses the corresponding representations, scores, coverage, and direct aggregation inputs. The current-transaction model remains present. Single-model scenarios afect only the named model; in Joint, each originally available optional output is removed independently with probability �.

InterFuse has the highest recall at every reported condition. At complete joint removal, its gain over clean B1 is −1.26 points, versus −2.18 for Standard key mask and −3.72 for UMI.

## A.8 Figure Display Details

Figure 4(a–b) displays bootstrap recall-gain distributions (pp). These distributions reflect evaluation-sample variability with model check points held fixed.

## A.9 Ranking Quality and Fusion Latency

![](images/4102df669e6536f6b84dd4cfba404da0050fa6cddab99b11446b4b7717f49186.jpg)  
Figure 7: Industrial recall gain versus batch-1 median fusion latency. All eleven methods from Table 3; encoders excluded.