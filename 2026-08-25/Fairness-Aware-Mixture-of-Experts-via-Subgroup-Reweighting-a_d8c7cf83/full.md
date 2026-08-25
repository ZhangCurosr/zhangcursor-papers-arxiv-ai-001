# Fairness-Aware Mixture-of-Experts via Subgroup Reweighting and Gate Regularization

Sunhee Hwang

Dongyang Mirae University

Seoul, Republic of Korea

sunheehwang@dongyang.ac.kr

Abstract—Deep learning models often produce performance disparities across demographic groups, due to the training data imbalance with respect to sensitive attributes such as gender or age. To address this problem, existing work has explored fair representation learning, data re-sampling, and adversarial training, which can be broadly categorized into two main approaches. Single-stage methods typically learn a shared representation for fairness, but often struggle to handle heterogeneous subgroup distributions. Two-stage methods learn representations separately from the final prediction task, which can lead to misalignment between fairness objectives and downstream predictions. We identify routing-induced bias, a failure mode in which subgroup imbalance drives the gating network to route subgroups onto a few experts, and propose an end-to-end Mixture-of-Experts (MoE) framework that corrects it. Specifically, we apply subgroup reweighting to correct data imbalance, and introduce gate entropy regularization to prevent routing from collapsing onto subgroup attributes, keeping expert utilization both balanced and interpretable. Beyond improving fairness, the routing distribution offers an interpretable view of how subgroups are allocated across experts. Experimental results demonstrate that the proposed approach improves fairness while maintaining competitive predictive performance.

Index Terms—fairness, mixture-of-experts, equalized odds, routing-induced bias, bias mitigation

## I. INTRODUCTION

Deep neural networks have achieved remarkable success across a wide range of visual recognition tasks. In high-stakes applications such as face analysis, however, their predictions are increasingly required to be not only accurate but also fair across demographic groups defined by sensitive attributes such as gender or age [1]–[3]. In practice, this requirement is often violated: models tend to perform substantially better on some groups than others. A key reason is that when a target label is statistically correlated with a sensitive attribute in the training data, the model can minimize its loss by exploiting this shortcut rather than learning the true decision rule. As a result, the learned representation entangles the target with the sensitive attribute, and the resulting accuracy gap across groups manifests as unfairness at inference time.

A main challenge is that such bias originates not from the model architecture alone, but from the structure of the training data itself. Real-world datasets are rarely balanced: certain combinations of target label and sensitive attribute—for example, a positive label paired with an underrepresented group—appear far less frequently than others. During training, the model is exposed to the majority subgroups far more frequently and optimizes predominantly for them, while underrepresented subgroups receive little gradient signal and are effectively underfit. This imbalance therefore produces disparities not at the level of individual attributes, but across their intersections (y, s), where the scarcest subgroups suffer the largest performance drops [4], [5].

Existing approaches mitigate bias through representation learning, disentanglement, or data-level interventions [6]–[11]. However, many rely on decoupled or multi-stage pipelines, in which fair representation learning is performed separately from the downstream prediction task. Such formulations do not fully optimize fairness in an end-to-end manner. This can lead to suboptimal alignment between representation learning and the final prediction objective.

Several methods adopt single-stage training that jointly optimizes task performance and fairness [12]–[14]. However, these approaches typically rely on a shared representation space. This makes it difficult to capture heterogeneous subgroup characteristics while maintaining both fairness and accuracy. As a result, a trade-off between bias mitigation and predictive performance is often observed.

These limitations point to a deeper need: a model that can process heterogeneous subgroups through separate computational pathways rather than a single shared representation. Mixture-of-Experts (MoE) provides exactly this capability through its expert routing mechanism. However, we find that MoE introduces a new and previously underexplored failure mode: under subgroup imbalance, the gating network—optimized predominantly on majority subgroups—correlates its routing decisions with sensitive attributes, concentrating each subgroup onto a few experts and leaving others underutilized. We term this routing-induced bias. Crucially, this failure is specific to conditional routing and is therefore invisible to existing fairness methods built on a single shared representation. To address it, we propose FAMoE, a fairness-aware MoE framework that diagnoses and corrects routing-induced bias end-to-end. The source code is available at https://github.com/sunhee-hwang/FAMoE. Our main contributions are summarized as follows:

• We identify routing-induced bias as a previously underexplored failure mode in Mixture-of-Experts models, in which subgroup imbalance drives the gating network to correlate routing decisions with sensitive attributes, yielding skewed expert utilization.

• We propose FAMoE, which corrects routing-induced bias end-to-end by combining subgroup reweighting with a gate entropy regularization that, unlike its conventional use for load balancing, here keeps routing interpretable by preventing experts from being monopolized by specific subgroups.

• We show that the routing distribution serves as an interpretable diagnostic of subgroup-level behavior, revealing how fairness is achieved an inspectability absent in monolithic representation-based methods.

• We demonstrate consistent improvements in the fairness -accuracy trade-off across multiple target and sensitive attribute settings on CelebA, alongside more balanced expert utilization.

## II. RELATED WORK

## A. MoE and Routing Dynamics

MoE is a conditional computation framework in which a gating network dynamically assigns inputs to a subset of experts, enabling input-dependent processing and improved model capacity [15]. This routing mechanism enables experts to specialize in different regions of the input space. It also facilitates modeling of heterogeneous data. Prior work on MoE has primarily focused on improving efficiency and scalability. Sparse models, exemplified by the Switch Transformer, mitigate computational overhead by selectively activating a subset of experts for each incoming token [16]. However, imbalanced expert utilization remains a well-known challenge in routing design [17]. Furthermore, empirical evidence consistently demonstrates expert specialization, wherein individual experts encode distinct functional or semantic patterns [18].

Despite these advances, existing MoE research has largely emphasized routing as an optimization and efficiency problem. While prior work has considered data heterogeneity and fairness in isolation, how subgroup imbalance shapes routing decisions remains largely unexamined. Recent work has begun to explore fairness in MoE models. For example, FairMOE introduces fairness-aware mechanisms into the expert selection process using counterfactual fairness criteria to promote consistency across protected attributes [19]. However, it does not analyze the interaction between data imbalance and routing dynamics. It also does not control this interaction during training.

## B. Learning Fairness Representation

Bias mitigation in visual classification mainly relies on representation learning. Existing methods adopt either end-toend joint optimization or multi-stage frameworks to enforce fairness constraints. Under joint optimization settings, adversarial techniques utilize gradient reversal [20] to penalize the encoding of sensitive attributes. Mutual information-based objectives [12] operate on a similar principle by minimizing the statistical dependence between learned features and protected variables.

Other strategies address fairness by directly intervening in the feature space. MFD [13] and Fair-VPT [14] suppress sensitive cues within a shared representation to prevent the classifier from utilizing demographic information. To formally model the structural interactions between target and sensitive attributes, disentanglement techniques decouple the representations. FD-VAE [6] partitions the feature space into target-specific, sensitive, and shared components to capture the overlap between features.

Recent literature extends these representation learning principles to account for inherent data imbalances and spurious correlations. These strategies encompass submodular hard sample mining (SHaSAM [10]), balanced training pair construction via data augmentation (FairCL [8]), and group-aware correlation suppression (Fair-GDMS [9]).

Although effective at reducing bias, these methodologies operate exclusively on a globally shared representation space. Processing highly heterogeneous data distributions across varying subgroups within a monolithic feature space restricts the model’s representational flexibility. This shared-space bottleneck indicates the necessity of input-dependent computation pathways to accommodate subgroup-specific characteristics. While conditional architectures like Mixture-of-Experts (MoE) provide this structural flexibility, they remain largely underexplored in the context of fairness. Furthermore, as our analysis reveals, standard conditional routing mechanisms are inherently susceptible to subgroup imbalance, necessitating a new framework that jointly addresses data heterogeneity and expert utilization.

## III. FAIRNESS-AWARE MIXTURE-OF-EXPERTS FRAMEWORK

## A. Overall Framework

We propose a fairness-aware visual classification framework based on a MoE architecture, as illustrated in Fig. 1. Given an input image x, a shared backbone encoder first extracts a feature representation $z \in \mathbb { R } ^ { d }$ . This representation is then used as input to both the expert networks and the gating network.

The backbone encoder is shared across all model variants to ensure that differences in performance and fairness can be attributed to the MoE components rather than feature extraction. This design enables a controlled comparison of modeling choices within a unified feature space. The MoE head consists of N experts $\{ E _ { i } \} _ { i = 1 } ^ { N }$ and a gating network $G ( \cdot )$ . Each expert independently processes the shared feature z and produces an expert-specific output, while the gating network computes a routing weight for each expert based on the same input. The final prediction is obtained by aggregating the expert outputs using the routing weights produced by the gating network. This structure allows the model to adaptively combine multiple expert predictions in a data-dependent manner, enabling flexible modeling of diverse input patterns within a single framework.

## B. Expert Prediction and Gated Aggregation

Given the shared feature representation z, each expert $E _ { i }$ produces an intermediate representation, which is passed

![](images/740bcb11ddffa560a28f1d46acff8890d26196bf127ffe614868dcc4a64467c1.jpg)  
Fig. 1. Overview of the proposed FAMoE (Fairness-Aware Mixture-of-Experts) framework.

through a classifier $C _ { i }$ to obtain an expert-specific logit:

$$
o _ { i } = C _ { i } ( E _ { i } ( z ) ) .\tag{1}
$$

The gating network $G ( \cdot )$ computes a routing vector $g =$ $[ g _ { 1 } , \dotsc , g _ { N } ]$ , where each element represents the importance of the corresponding expert. The routing weights are normalized using a softmax function:

$$
g _ { i } = \frac { \exp ( a _ { i } ) } { \sum _ { j = 1 } ^ { N } \exp ( a _ { j } ) } ,\tag{2}
$$

where $a _ { i }$ denotes the routing score for expert $i .$

The final prediction is obtained by aggregating the expert outputs using the routing weights:

$$
\hat { o } = \sum _ { i = 1 } ^ { N } g _ { i } o _ { i } .\tag{3}
$$

For binary classification, the final prediction probability is computed as $\hat { y } ~ = ~ \sigma ( \hat { o } )$ , where $\sigma ( \cdot )$ denotes the sigmoid function.

## C. Subgroup Reweighting for Joint Attribute Imbalance

To address imbalance across target and sensitive attributes, we incorporate subgroup reweighting into the training process. Each sample is associated with a joint subgroup defined by its target label $y$ and sensitive attribute s. Let $n _ { y , \varepsilon }$ <sub>s</sub> denote the number of samples in subgroup $( y , s )$ . We assign each subgroup $( y , s )$ a weight inversely proportional to its size:

$$
w _ { y , s } \propto \frac { 1 } { n _ { y , s } } .\tag{4}
$$

This weighting scheme increases the contribution of underrepresented subgroups during training. We employ a weighted binary cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { c l s } } = \sum _ { i = 1 } ^ { B } w _ { y _ { i } , s _ { i } } \cdot \mathrm { B C E } ( \hat { o } _ { i } , y _ { i } ) ,\tag{5}
$$

where B denotes the batch size.

## D. Gate Entropy Regularization

To promote balanced utilization of experts, we introduce a gate entropy regularization term. Given the routing distribution $g$ for each input, the entropy is defined as:

$$
H ( g ) = - \sum _ { i = 1 } ^ { N } g _ { i } \log g _ { i } .\tag{6}
$$

We add this term to encourage higher entropy in the routing distribution. The overall loss function is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { c l s } } - \lambda _ { \mathrm { e n t } } H ( g ) , } \end{array}\tag{7}
$$

where $\lambda _ { \mathrm { e n t } }$ controls the strength of the regularization. This regularization reduces overly concentrated routing decisions. It also encourages the utilization of multiple experts during training. As a result, the model maintains diverse expert contributions while performing data-dependent aggregation. Importantly, this regularization is not merely a fairness penalty but a mechanism that keeps the routing distribution interpretable: by preventing an expert from being monopolized by a specific subgroup, the resulting gate weights remain a faithful diagnostic of how the model distributes subgroups across experts.

## E. Training Objective

The overall training objective combines subgroup reweighting and gate entropy regularization. Different model variants are instantiated by selectively enabling each component. Specifically, the standard MoE model is obtained by setting $w _ { y , s } = 1$ and $\lambda _ { \mathrm { e n t } } = 0$ , while the reweighted model incorporates subgroup weights with $\lambda _ { \mathrm { e n t } } = 0$ . The full model includes both subgroup reweighting and entropy regularization. All models are trained end-to-end using stochastic gradient descent, where the encoder, experts, and gating network are jointly optimized.

## IV. EXPERIMENTS

## A. Experimental Setup

We evaluate the proposed method on the CelebA dataset, which contains facial images annotated with multiple binary attributes. Following standard practice, we consider binary classification tasks with a designated target attribute and a sensitive attribute. Each sample is associated with a pair $( y , s )$ where y denotes the target label and s denotes the sensitive attribute. The dataset is split into training, validation, and test sets according to the official partition. Images are resized to 128×128 and normalized. We use standard data augmentation, including random resized cropping and horizontal flipping during training. All models share the same backbone encoder based on ResNet-18. The MoE head consists of four experts and a gating network that produces a routing distribution over experts. Each expert is a two-layer MLP (512 → 128 → 64) with ReLU activation, followed by an expert-specific linear classifier (64 → 1). All experts share the same architecture but maintain independent parameters. The gating network is a single linear layer (512 → 4) followed by a softmax. Models are trained using the Adam optimizer with a learning rate of $1 0 ^ { - 4 }$ and weight decay of $1 0 ^ { - 4 }$ for 30 epochs.

TABLE I  
COMPARISON OF FAIRNESS–ACCURACY TRADE-OFFS ON CELEBA ACROSS MULTIPLE TARGET ATTRIBUTES. WE REPORT EQUALIZED ODDS (EO), ACCURACY, AND THE FAIRNESS–ACCURACY TRADE-OFF SCORE (FATS), UNDER SENSITIVE ATTRIBUTES (MALE AND YOUNG). A LOWER FATS INDICATES A MORE FAVORABLE TRADE-OFF. \* DENOTES THE METHOD RE-IMPLEMENTED BY THE AUTHORS. MALE AND YOUNG DENOTE THE SENSITIVE ATTRIBUTE s USED TO COMPUTE EO; ALL SAMPLES (BOTH $s = 0 \ A N D \ s = 1 )$ ARE INCLUDED IN TRAINING AND EVALUATION. BOLD INDICATES THE BEST RESULT WITHIN EACH STAGE CATEGORY (1-STAGE AND 2-STAGE), REPORTED SEPARATELY FOR EACH COLUMN.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Year</td><td rowspan="3">Stage</td><td colspan="6">Target: Attractiveness</td><td colspan="6">Target: Big Nose</td></tr><tr><td colspan="3">Male</td><td colspan="3">Young</td><td colspan="3">Male</td><td colspan="3">Young</td></tr><tr><td>EO</td><td>ACC</td><td>FATS</td><td>EO</td><td>ACC</td><td>FATS</td><td>EO</td><td>ACC</td><td>FATS</td><td>EO</td><td>ACC</td><td>FATS</td></tr><tr><td>ResNet-18 [21]</td><td>2016</td><td rowspan="2">Baseline</td><td>27.8</td><td>79.6</td><td>27.9</td><td>16.8</td><td>79.8</td><td>16.9</td><td>17.6</td><td>84.0</td><td>17.7</td><td>14.7</td><td>84.5</td><td>14.8</td></tr><tr><td>*ResNet-18 [21] + MoE [15]</td><td>2016 / 1991</td><td>13.3</td><td>81.0</td><td>13.4</td><td>22.4</td><td>81.1</td><td>22.5</td><td>16.0</td><td>84.1</td><td>16.1</td><td>14.1</td><td>84.3</td><td>14.2</td></tr><tr><td>FD-VAE [6]</td><td>2021</td><td rowspan="5">2-stage</td><td>15.1</td><td>76.9</td><td>15.2</td><td>14.8</td><td>77.5</td><td>14.9</td><td>11.2</td><td>81.6</td><td>11.3</td><td>6.7</td><td>81.7</td><td>6.8</td></tr><tr><td>FSCL [7]</td><td>2022</td><td>6.5</td><td>79.1</td><td>6.6</td><td>12.4</td><td>79.1</td><td>12.5</td><td>4.7</td><td>82.9</td><td>4.8</td><td>4.8</td><td>84.1</td><td>4.9</td></tr><tr><td>FairCL [8]</td><td>2023</td><td>16.8</td><td>75.3</td><td>16.9</td><td>13.1</td><td>76.9</td><td>13.2</td><td>8.4</td><td>80.1</td><td>8.5</td><td>9.2</td><td>80.3</td><td>9.3</td></tr><tr><td>Fair-GDMS [9]</td><td>2026</td><td>7.0</td><td>83.2</td><td>7.1</td><td></td><td></td><td></td><td>13.2</td><td>84.8</td><td>13.3</td><td></td><td></td><td></td></tr><tr><td>SHaSAM [10]</td><td>2026</td><td>5.5</td><td>81.3</td><td>5.6</td><td>9.9</td><td>79.6</td><td>10.0</td><td>3.3</td><td>84.7</td><td>3.4</td><td>3.9</td><td>87.0</td><td>4.0</td></tr><tr><td>GRL [20]</td><td>2018</td><td></td><td>24.9</td><td>77.2</td><td>25.0</td><td>14.7</td><td>74.6</td><td>14.8</td><td>14.0</td><td>82.5</td><td>14.1</td><td>10.0</td><td>83.3</td><td>10.1</td></tr><tr><td>LNL [12]</td><td>2019</td><td rowspan="3">1-stage</td><td>21.8</td><td>79.9</td><td>21.9</td><td>13.7</td><td>74.3</td><td>13.8</td><td>10.7</td><td>82.3</td><td>10.8</td><td>6.8</td><td>82.3</td><td>6.9</td></tr><tr><td>MFD [13]</td><td>2021</td><td>7.4</td><td>78.0</td><td>7.5</td><td>14.9</td><td>80.0</td><td>15.0</td><td>7.3</td><td>78.0</td><td>7.4</td><td>5.4</td><td>78.0</td><td>5.5</td></tr><tr><td>Fair-VPT [14]</td><td>2024</td><td>12.0</td><td>78.6</td><td>12.1</td><td></td><td></td><td></td><td>15.9</td><td>79.9</td><td>16.0</td><td>-</td><td></td><td></td></tr><tr><td>FAMoE (Ours)</td><td>2026</td><td></td><td>4.1</td><td>80.0</td><td>4.2</td><td>8.3</td><td>79.8</td><td>8.4</td><td>4.5</td><td>80.1</td><td>4.6</td><td>3.3</td><td>80.8</td><td>3.4</td></tr></table>

We evaluate performance using classification accuracy, Equalized Odds (EO) [22], and the Fairness–Accuracy Tradeoff Score (FATS). EO measures the difference in prediction performance across subgroups defined by $( y , s )$ . Following [7], we compute EO as the average accuracy gap across sensitive groups within each target class. FATS is defined as:

$$
\mathrm { F A T S } = \mathrm { E O } + \alpha ( 1 - \mathrm { A c c } ) ,\tag{8}
$$

where α controls the trade-off between fairness and accuracy. In our experiments, we set $\alpha = 0 . 5$ to weight fairness and accuracy degradation equally, and a lower FATS indicates a more favorable balance between fairness and predictive performance. The best model is selected based on the validation FATS.

TABLE II  
ABLATION STUDY OF PROPOSED COMPONENTS ON CELEBA (TARGET: ATTRACTIVENESS).
<table><tr><td colspan="2">Components</td><td colspan="3">Sensitive: Male</td></tr><tr><td>Subgroup Reweighting</td><td>Gate Entropy Regularization</td><td>EO</td><td>ACC</td><td>FATS</td></tr><tr><td>x</td><td>x</td><td>13.3</td><td>81.0</td><td>13.4</td></tr><tr><td>√</td><td>x</td><td>4.2</td><td>79.6</td><td>4.3</td></tr><tr><td>√</td><td>√</td><td>4.1</td><td>80.0</td><td>4.2</td></tr></table>

## B. Main Results

Table I presents the comparison with existing fairness-aware methods across different target and sensitive attribute settings. Compared to one-stage methods such as GRL, LNL, and MFD, the proposed method consistently improves fairness while avoiding large drops in accuracy. Compared to two-stage approaches, including FD-VAE, FairCL, and Fair-GDMS, the proposed method achieves a more favorable balance between fairness and accuracy, as reflected in consistently lower FATS values. These results indicate that jointly addressing subgroup imbalance and routing decisions leads to improved fairness compared to existing approaches.

## C. Ablation Study

Table II shows the effect of each component in the proposed framework. The standard MoE model achieves an EO of 13.3, indicating that MoE alone does not improve fairness. Adding subgroup reweighting reduces EO from 13.3 to 4.2, confirming that data imbalance is a primary driver of routing bias. Reweighting alone, however, leaves routing concentration uncontrolled; adding gate entropy regularization further improves EO to 4.1, recovers accuracy, and yields the balanced, interpretable routing analyzed in Fig. 2. Entropy regularization thus complements reweighting by promoting balanced expert utilization.

![](images/a90966465c854fb205144de7a2a7a32a986391debf83f7c306c8fa464cee717e.jpg)  
Fig. 2. Average gate weights across subgroups for the standard MoE and the proposed method.

## D. Analysis of Expert Utilization

To investigate expert utilization patterns, we analyze the average gate weights across demographic subgroups (Fig. 2). In the standard MoE baseline, routing decisions exhibit pronounced skewness. Specific experts are disproportionately activated by certain subgroups, while others remain underutilized. This observation indicates that the baseline routing mechanism inadvertently captures spurious correlations with sensitive attributes. In contrast, our method effectively mitigates this dependency, yielding a more balanced distribution of routing weights. These empirical findings demonstrate that our approach prevents a single subgroup from dominating expert capacity, thereby promoting more equitable expert utilization across subgroups. This analysis is only possible because routing exposes expert allocation explicitly. Monolithic models that share a single representation space may reach comparable fairness scores, yet they offer no comparable window into how fairness is achieved. The gate-weight distribution in Fig. 2 thus serves as a built-in diagnostic of subgroup-level model behavior, a form of routing-level interpretability not available in monolithic representation-based methods.

## V. CONCLUSION

This study addresses the degradation of fairness in MoE models caused by imbalanced data distributions. We show that, under subgroup imbalance, conditional routing tends to induce systematic bias, skewing expert utilization based on subgroup attributes. Recognizing that existing representationfocused methods fail to capture this routing-induced bias, we propose a framework combining subgroup reweighting with gate entropy regularization. Our empirical results demonstrate that regularizing routing behaviors while correcting data imbalance promotes more balanced expert assignment and substantially improves the fairness–accuracy trade-off. Overall, this work highlights the critical necessity of explicitly accounting for expert utilization, which not only improves fairness but also renders the model’s subgroup-level behavior interpretable through its routing distribution—a form of transparency not offered by monolithic fairness methods.

## VI. ACKNOWLEDGMENT

This work was supported by the Korea Foundation for Women in Science, Engineering and Technology (WISET) with funding from the Science and Technology Promotion Fund and Lottery Fund (WISET 2026-236).

## REFERENCES

[1] N. Mehrabi, F. Morstatter, N. Saxena, K. Lerman, and A. Galstyan, “A survey on bias and fairness in machine learning,” ACM computing surveys (CSUR), vol. 54, no. 6, pp. 1–35, 2021.

[2] Y. Wang, W. Ma, M. Zhang, Y. Liu, and S. Ma, “A survey on the fairness of recommender systems,” ACM Transactions on Information Systems, vol. 41, no. 3, pp. 1–43, 2023.

[3] D. Pessach and E. Shmueli, “A review on fairness in machine learning,” ACM computing surveys (CSUR), vol. 55, no. 3, pp. 1–44, 2022.

[4] D. Plecko and E. Bareinboim, “Fairness-accuracy trade-offs: A causal perspective,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 25, 2025, pp. 26 344–26 353.

[5] D. Kim, S. Park, S. Hwang, and H. Byun, “Fair classification by loss balancing via fairness-aware batch sampling,” Neurocomputing, vol. 518, pp. 231–241, 2023.

[6] S. Park, S. Hwang, D. Kim, and H. Byun, “Learning disentangled representation for fair facial attribute classification via fairness-aware information alignment,” in Proceedings of the AAAI conference on artificial intelligence, vol. 35, no. 3, 2021, pp. 2403–2411.

[7] S. Park, J. Lee, P. Lee, S. Hwang, D. Kim, and H. Byun, “Fair contrastive learning for facial attribute classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 10 389–10 398.

[8] F. Zhang, K. Kuang, L. Chen, Y. Liu, C. Wu, and J. Xiao, “Fairnessaware contrastive learning with partially annotated sensitive attributes,” in The Eleventh International Conference on Learning Representations, 2022.

[9] H. Huang, K. Li, S. Chen, and D.-H. Wang, “Fair facial attribute recognition via group-decoupled vision transformer with mask-guided correlation suppression,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 7, 2026, pp. 5022–5030.

[10] A. Majee and R. Iyer, “Shasam: Submodular hard sample mining for fair facial attribute recognition,” in Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, 2026, pp. 7461–7471.

[11] S. Hwang, S. Park, P. Lee, S. Jeon, D. Kim, and H. Byun, “Exploiting transferable knowledge for fairness-aware image classification,” in Proceedings of the Asian Conference on Computer Vision, 2020.

[12] B. Kim, H. Kim, K. Kim, S. Kim, and J. Kim, “Learning not to learn: Training deep neural networks with biased data,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2019, pp. 9012–9020.

[13] S. Jung, D. Lee, T. Park, and T. Moon, “Fair feature distillation for visual recognition,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2021, pp. 12 115–12 124.

[14] S. Park and H. Byun, “Fair-vpt: Fair visual prompt tuning for image classification,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 12 268–12 278.

[15] R. A. Jacobs, M. I. Jordan, S. J. Nowlan, and G. E. Hinton, “Adaptive mixtures of local experts,” Neural computation, vol. 3, no. 1, pp. 79–87, 1991.

[16] W. Fedus, B. Zoph, and N. Shazeer, “Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity,” Journal of Machine Learning Research, vol. 23, no. 120, pp. 1–39, 2022.

[17] L. Wang, H. Gao, C. Zhao, X. Sun, and D. Dai, “Auxiliary-lossfree load balancing strategy for mixture-of-experts,” arXiv preprint arXiv:2408.15664, 2024.

[18] J. Oldfield, M. Georgopoulos, G. G. Chrysos, C. Tzelepis, Y. Panagakis, M. A. Nicolaou, J. Deng, and I. Patras, “Multilinear mixture of experts: Scalable expert specialization through factorization,” Advances in Neural Information Processing Systems, vol. 37, pp. 53 022–53 063, 2024.

[19] J. Germino, N. Moniz, and N. V. Chawla, “Fairmoe: counterfactuallyfair mixture of experts with levels of interpretability,” Machine Learning, vol. 113, no. 9, pp. 6539–6559, 2024.

[20] E. Raff and J. Sylvester, “Gradient reversal against discrimination: A fair neural network learning approach,” in 2018 IEEE 5th International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2018, pp. 189–198.

[21] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778.

[22] M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” Advances in neural information processing systems, vol. 29, 2016.