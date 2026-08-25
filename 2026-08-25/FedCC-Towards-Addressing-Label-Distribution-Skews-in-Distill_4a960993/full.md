# FedCC: Towards Addressing Label Distribution Skews in Distillation-Based Federated Learning

Wenxuan Ye <sup>∗†</sup>, Onur Ayan<sup>∗</sup>, Xueli An<sup>∗</sup>, Georg Carle<sup>†</sup>

∗ Huawei Heisenberg Research Center, Huawei Technologies Duesseldorf GmbH

<sup>†</sup> TUM School of Computation, Information and Technology, Technical University of Munich wenxuan.ye@tum.de, {onur.ayan, xueli.an}@huawei.com, carle@net.in.tum.de

Abstract—Federated Learning (FL) enables distributed clients to collaboratively train models without sharing raw data, making it promising for leveraging massive devices in communication networks. In distillation-based FL, each client applies its local model on an unlabeled public dataset, and shares only prediction results with the server. While heterogeneous local data introduces label distribution skew, thus biasing client models toward majority classes and leading to potentially inaccurate predictions. The lack of ground-truth labels in the public dataset hampers the server’s ability to calibrate predictions, which ultimately degrades overall performance. To address this, we propose FedCC, a simple and effective algorithm for mitigating client misclassification. Instead of being forced to classify and risking error propagation, clients are allowed to tag ambiguous samples as ‘unknown’. This additional class, together with calibrated pseudo-labels on the public data, balances confidence in majority classes against uncertainty in under-represented ones. Extensive experiments demonstrate that FedCC significantly outperforms existing methods, especially under severe label skew. In the extreme scenario where each client holds samples from only one of ten classes, FedCC achieves 67.3% accuracy, while baselines collapse to near-random results.

Index Terms—Federated learning, Distillation-based FL, Label distribution skew, Pseudo-labeling

## I. INTRODUCTION

Federated Learning (FL) [1] enables collaborative model training across decentralized clients without exposing their raw data. In the classical parameter-based approach, each client refines the global model with its private data, uploads the updated parameters to the server, and the server aggregates these updates to produce the next-round global model. While effective, this approach incurs heavy communication overhead on parameter transmission, and assumes a homogeneous model architecture, overlooking the diverse bandwidth, computation, and data distributions across real-world devices [2]–[4].

Drawing on Knowledge Distillation (KD) [5], Distillationbased FL offers an alternative for client collaboration [6]. Each client runs its local model on a public dataset and uploads only the resulting prediction vectors (i.e., class-score logits); the server then aggregates these vectors into a global prediction vector and redistributes it, reducing communication overhead by orders of magnitude. In addition, because logits are architecture-agnostic, the approach natively supports heterogeneous models, enabling flexible deployment across varied networked devices. Regarding the public dataset, various strategies have been employed, including the use of a labeled dataset [7], or the generation of synthetic data derived from local datasets or local model parameters [8]. Recent studies [9] have relaxed the requirement by adopting unlabeled ones, avoiding costly labeling processes and alleviating privacy concerns. Accordingly, this paper adopts an unlabeled public dataset as an integral component of the scenario setting.

![](images/632c0dc1359ac35c656406f16cfe7cca058bdc4a1264aed5339e74753dcf0fbb.jpg)  
Fig. 1: Comparison of distillation-based FL under label distribution skews. a) FedDF [9] averages biased local predictions, leading to suboptimal outcomes. b) Our FedCC introduces an ‘unknown’ class (highlighted in blue shadow), allowing clients to acknowledge their unknownness and thereby yielding more informative aggregation.

Despite its promise, distillation-based FL is vulnerable to label distribution skew: heterogeneous client data cause local models to overfit the majority classes, and the resulting biases are reflected in the prediction vectors [2]. The challenge is further compounded by the absence of ground-truth labels in the public dataset, which prevents the server from calibrating logits and results in distorted supervision that degrades global performance. Existing approaches in parameter-based FL (e.g., weighted model aggregation [10], model regularization [11]) presume access to client model weights. This assumption breaks down in distillation-based FL, where only logit vectors are exchanged, providing far sparser signals. Recent efforts focus on weighting client updates by estimated reliability during server-side aggregation [12], moving beyond naive averaging as in FedDF [9]. Others aim at amplifying minority-class information and aligning local models with the global objective [13]. Nevertheless, label-skew bias still creeps in, and existing schemes cannot consistently filter out misleading updates during aggregation. Empirical evidence [14] demonstrates that, under label-distribution skew, local models overwhelmingly predict their majority classes, thereby injecting errors into the aggregated prediction (as in Fig. 1a).

In response, we propose FedCC, a simple and effective algorithm designed to mitigate Client misClassification. Treating the mismatch between a client’s skewed data and the ideal balanced distribution as an open-set problem [15], we introduce an ‘unknown’ class to absorb minority and missing classes.

Instead of being forced to issue a class prediction and risking error propagation, clients are allowed to acknowledge their limitations and mark ambiguous inputs as ‘unknown’. Illustrated in Fig. 1b, this ‘unknown’ class (shown in blue shadow) enables clients to express uncertainty, yielding more informative aggregated predictions. To implement this, we incorporate the unlabeled dataset into the local objective function and employ a calibrated pseudo-label generation method, to weigh confidence in majority classes with the uncertainty hidden in unknown classes. This weight is adaptively adjusted per client and per sample, optimizing the model’s generalization under diverse data conditions.

To evaluate performance, we conduct experiments on three widely used datasets: CIFAR-10, CIFAR-100, and TinyImageNet, across various label skew scenarios. FedCC consistently outperforms state-of-the-art FL methods by at least 3% points in most scenarios, and the accuracy gap grows as label skew becomes more severe. In the extreme scenario where each client holds samples from only one of ten classes, FedCC achieves an accuracy of 67.3%, while other baselines fail to generate meaningful predictions. Beyond delivering strong global generalization, FedCC enhances local model performance on its minority classes by leveraging uncertaintyaware predictions.

To summarize, our key contributions are as follows:

• We introduce FedCC, a simple and effective method for mitigating client misclassification. We frame label skew as an open-set recognition problem, and extend the existing class framework with an ‘unknown’ class to absorb minority and missing classes.

• FedCC incorporates the unlabeled dataset into the local objective function, and proposes a novel pseudo-label generation method, enabling the model to recognize and account for unknownness. This method can be interpreted as adding a regularizer, mitigating overfitting.

• Extensive experiments demonstrate that FedCC significantly outperforms current FL methods, and the performance gap increases as label skew becomes more severe.

## II. RELATED WORKS

Parameter-based FL with label distribution skews: While FL safeguards privacy by keeping data local, the skewed label distributions across clients undermine model generalization. To stabilize heterogeneous training, methods like FedProx [11] and FedSAM [16] employ regularization, variance reduction, or gradient filtering. Other approaches target class imbalance, e.g., via rescaling logits (FedRS [17]), or calibrating predictions (FedLC [18], FedMR [13], FedVLS [19]). Alternatively, FedConcat [20] departs from model averaging by concatenating local encoders to collaboratively train a global classifier. While FedOV [14] uses open-set recognition to alleviate label skew, its reliance on synthesizing outlier samples with implicit labels limits its applicability to unlabeled data.

Distillation-based FL with label distribution skews: To curb label skew under distillation-based FL, FedNed [21] emphasizes distilling minority-class knowledge, LightTS [22] re-weights clients by strength, and Co-Boosting [8] alternates synthetic data generation with ensemble updates. However, their reliance on labeled public datasets or large-scale synthesis raises annotation and privacy concerns. When public data is unlabeled, the lack of ground-truth labels complicates bias correction. Existing solutions address this by retaining high-confidence predictions [7], clustering clients [23], or communicating only hard labels [24]. While emphasizing reliable local knowledge, these methods struggle to filter misinformation stemming from inherent local biases (As shown in Fig. 1a). Thus, effective solutions for fair client-side classification remain an open challenge.

## III. SCENARIO SETTING

Consider an FL scenario with K clients and C classes (label set $\mathbb { C } = \{ 1 , \dots , C \} )$ ). Each client k holds a private local dataset $\mathbb { D } _ { l } ^ { ( k ) } = \dot { \{ ( x , y ) \} }$ . Additionally, a global unlabeled public dataset $\mathbb { D } _ { u } = \{ { x } _ { u } \}$ is accessible to all clients and the server. Due to label skew, client label priors $P ^ { ( k ) } ( y )$ vary, though we assume the conditional distribution $P ( \boldsymbol { y } | \boldsymbol { x } )$ is consistent across clients [2]. We partition C into three mutually exclusive subsets $( \mathbb { C } _ { J } ^ { ( k ) } \cup \mathbb { C } _ { N } ^ { ( k ) } \cup \mathbb { C } _ { S } ^ { ( k ) } = \mathbb { C } )$ for each client k: majority classes $\mathbb { C } _ { J } ^ { ( \dot { k } ) }$ (sufficiently represented), minority classes $\mathbb { C } _ { N } ^ { ( k ) }$ (sparse, noisy estimates), and missing classes ${ \dot { \mathbb { C } } } _ { S } ^ { ( k ) }$ (absent). Collaborative distillation protocol: Client k deploys a heterogeneous classifier $f ^ { ( k ) } { \bar { ( } } \cdot ; \theta ^ { ( k ) } )$ . In each communication round, client k computes logits $f ^ { ( k ) } ( x _ { u } ) \in \mathbb { R } ^ { C }$ for public samples $x _ { u } ~ \in ~ \mathbb { D } _ { u }$ . These are converted to probabilities $\hat { \mathbf { y } } ^ { ( k ) } ( x _ { u } ) : = \sigma ^ { ( k ) } ( x _ { u } )$ via a temperature-scaled softmax (with temperature $\tau = 1$ omitted for brevity).

The server aggregates these probabilities into an ensemble teacher:

$$
\hat { \mathbf { y } } ^ { ( s ) } ( x _ { u } ) : = \sum _ { k } \mu _ { k } \hat { \mathbf { y } } ^ { ( k ) } ( x _ { u } ) , \quad \mu _ { k } \geq 0 , ~ \sum _ { k } \mu _ { k } = 1 .\tag{1}
$$

Clients then synchronize their models by minimizing the Kullback-Leibler divergence with the teacher distribution:

$$
\mathbb { E } _ { \boldsymbol { x } _ { u } \sim \mathbb { D } _ { u } } \left[ \operatorname { D } _ { \mathrm { K L } } \left( \hat { \mathbf { y } } ^ { ( s ) } ( \boldsymbol { x } _ { u } ) \parallel \sigma ^ { ( k ) } ( \boldsymbol { x } _ { u } ) \right) \right] .\tag{2}
$$

Post-synchronization, clients refine their models on local private data before sharing predictions in the next round.

## IV. APPROACH

FedCC targets client-side misclassification, and by lowering local errors, it ultimately improves global generalization. Sections IV-A-IV-B detail the proposed client-level algorithm, and Section IV-C outlines the overall federated training pipeline. In Sections IV-A–IV-B we focus on a single client, and therefore omit the client index for notational brevity (e.g., f, θ and σ instead of $f ^ { ( k ) }$ , θ<sup>(k)</sup> and $\sigma ^ { ( k ) } ,$ ).

## A. Local learning objective

Semi-supervised learning: We begin by reviewing Semi-Supervised Learning (SSL), a paradigm that leverages both labeled and unlabeled data to build effective models. A classical SSL strategy is self-training [25], which bootstraps a model with its own high-confidence predictions. Firstly, a classifier $f ( \cdot ; \theta _ { 0 } )$ is fitted on labeled data ${ \mathbb D } _ { l } = \{ ( x , y ) \}$ . The trained model $\theta _ { 0 }$ then assigns pseudo-labels to the unlabeled data $x _ { u } ~ \in ~ \mathbb { D } _ { u }$ by selecting the class of maximum posterior probability: $\hat { y } ( x _ { u } ) ~ = ~ \operatorname { a r g m a x } _ { c \in \mathbb { C } } f _ { c } ( x _ { u } ; \theta _ { 0 } )$ . Merging the original and pseudo-labeled examples yields an enlarged dataset on which the model is re-optimized. The overall training objective is to minimize the following loss function:

$$
\begin{array} { r l } & { \mathcal { L } ( \boldsymbol { \theta } ) : = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim \mathbb { D } _ { l } } \left[ \ell _ { \mathrm { C E } } ( f ( \boldsymbol { x } ; \boldsymbol { \theta } ) , \boldsymbol { y } ) \right] } \\ & { \quad \quad \quad + \lambda \mathbb { E } _ { \boldsymbol { x } _ { u } \sim \mathbb { D } _ { u } } \left[ \ell _ { \mathrm { C E } } ( f ( \boldsymbol { x } _ { u } ; \boldsymbol { \theta } ) , \hat { \boldsymbol { y } } ( \boldsymbol { x } _ { u } ) ) \right] } \end{array}\tag{3}
$$

where the cross-entropy loss $\ell _ { \mathrm { { C E } } } ( f ( x ) , y ) ~ = ~ - \log \sigma _ { y } ( x )$ quantifies the discrepancy between the label y and prediction logits $f ( x )$ , and $\sigma _ { y } ( x )$ is given as the probability after the softmax operation; $\lambda > 0$ balances the labeled and unlabeled terms.

Our method: As described in the scenario setting, client k maintains a labeled dataset $\mathbb { D } _ { l } ^ { ( k ) }$ and unlabeled dataset $\mathbb { D } _ { u }$ mirroring the typical SSL structure. To optimally utilize both datasets, we design the objective function based on Eq. 3 as:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) : = \underbrace { \mathbb { E } _ { ( x , y ) \sim \mathbb { D } _ { l } ^ { ( k ) } } \left[ \ell _ { \mathrm { C E } } ( f ( x ; \theta ) , y ) \right] } _ { : \mathcal { L } _ { l } ( \theta ) } } \\ & { ~ + \lambda \underbrace { \mathbb { E } _ { x _ { u } \sim \mathbb { D } _ { u } } \left[ \ell _ { \mathrm { C E } } ( f ( x _ { u } ; \theta ) , \hat { y } ( x _ { u } ) ) \right] } _ { : \mathcal { L } _ { u } ( \theta ) } } \end{array}\tag{4}
$$

where $\mathcal { L } _ { l } ( \boldsymbol { \theta } )$ and $\mathcal { L } _ { u } ( \boldsymbol { \theta } )$ denote the losses on the labeled and unlabeled datasets, respectively.

As introduced before, client k first fits a model $\theta _ { 0 }$ on its labeled data, detailed as $\theta _ { 0 } : = \arg$ min<sub>θ</sub> $\mathcal { L } _ { l } ( \boldsymbol { \theta } )$ . Ideally, $\theta _ { 0 }$ would master classes it has encountered, i.e., $\mathbb { C } _ { J } ^ { ( k ) } \bigcup \mathbb { C } _ { N } ^ { ( k ) }$ . However, in practice, the severe under-representation of minority classes $\mathbb { C } _ { N } ^ { ( \hat { k } ) }$ leads to poorly calibrated posterior probabilities, which often drift toward majority classes [18]. Furthermore, $\theta _ { 0 }$ lacks the ability to recognize samples from missing classes as it has never learned them during training.

To mitigate misclassification of minority and missing classes, we enlarge the label space with an auxiliary ‘unknown’ class $u ,$ which acts as a catch-all for samples with unreliable predictions. With the label space augmented to $\mathbb { C } \cup \{ u \}$ , the model then satisfies $\begin{array} { r } { \sum _ { y \in \mathbb { C } } \sigma _ { y } ( x _ { u } ) + \sigma _ { u } ( x _ { u } ) = 1 } \end{array}$ Building on this, we ideally partition the unlabeled dataset $\mathbb { D } _ { u }$ into samples from majority classes $\mathbb { C } _ { J } ^ { ( k ) }$ , and all others. If the sample’s true class belongs to client $k ' s$ majority set $y ( x _ { u } ) \in \mathring { \mathbb { C } } _ { J } ^ { ( k ) }$ , the pseudo-label is assigned to: $m ( x _ { u } ) : =$ $\operatorname { a r g m a x } _ { { c } \in \mathbb { C } _ { J } ^ { ( k ) } } \left( f _ { c } ( x _ { u } ; \theta _ { 0 } ) \right)$ , as abundant local examples enable the client to identify these classes reliably $[ 2 6 ] ;  { \mathrm { i f } } y ( x _ { u } ) \not \in \mathbb { C } _ { J } ^ { ( k ) }$ the sample is intended to be classified as u. Then, the ideal pseudo-label assignment is

$$
\hat { y } ( x _ { u } ) = m ( x _ { u } ) \mathrm { i f } y ( x _ { u } ) \in \mathbb { C } _ { J } ^ { ( k ) } \mathrm { e l s e } u\tag{5}
$$

The unlabeled loss $\mathcal { L } _ { u } ( \boldsymbol { \theta } )$ is therefore reformulated as:

$$
\begin{array} { r l } & { \mathbb { E } _ { x _ { u } \sim \mathbb { D } _ { u } } \big [ \mathbf { 1 } [ y ( x _ { u } ) \in \mathbb { C } _ { J } ^ { ( k ) } ] \ell _ { \mathrm { C E } } ( f ( x _ { u } ; \theta ) , m ( x _ { u } ) ) } \\ & { \qquad + \left( 1 - \mathbf { 1 } [ y ( x _ { u } ) \in \mathbb { C } _ { J } ^ { ( k ) } ] \right) \ell _ { \mathrm { C E } } ( f ( x _ { u } ; \theta ) , u ) \big ] } \end{array}\tag{6}
$$

where the indicator $\mathbf { 1 } [ P ]$ equals 1 if the event $P$ is true and 0 otherwise. Since the truth labels $y ( x _ { u } )$ are unavailable, we approximate this indicator with a soft weight $\alpha ( x _ { u } ) \in [ 0 , 1 ]$

$$
\begin{array} { r l } & { \mathcal { L } _ { u } ( \boldsymbol { \theta } ) \approx \mathbb { E } _ { \boldsymbol { x } _ { u } \sim \mathbb { D } _ { u } } \left[ \alpha ( \boldsymbol { x } _ { u } ) \ell _ { \mathrm { C E } } ( f ( \boldsymbol { x } _ { u } ; \boldsymbol { \theta } ) , m ( \boldsymbol { x } _ { u } ) ) \right. } \\ & { \qquad \left. + \left( 1 - \alpha ( \boldsymbol { x } _ { u } ) \right) \ell _ { \mathrm { C E } } ( f ( \boldsymbol { x } _ { u } ; \boldsymbol { \theta } ) , u ) \right] } \end{array}\tag{7}
$$

The comparison between Eq. 4 and Eq. 7 yields the pseudolabel in probabilistic-vector form:

$$
\hat { \mathbf { y } } ( x _ { u } ) : = \alpha ( x _ { u } ) \cdot \mathbf { e } _ { m ( x _ { u } ) } + ( 1 - \alpha ( x _ { u } ) ) \cdot \mathbf { e } _ { u }\tag{8}
$$

where ${ \bf e } _ { m \left( x _ { u } \right) }$ is the one-hot vector for the predicted class $m ( x _ { u } )$ , and $\mathbf { e } _ { u }$ represents that of class $u .$ This pseudo-label distributes probability mass over only two classes: the predicted class $m ( x _ { u } )$ and the unknown class u. We detail the concrete design of α in the following subsection.

## B. Weight design

The weight $\alpha ( x _ { u } )$ re-weighs the local loss on unlabeled data so that learning is guided by global rather than local class prevalence. Instead of relying on class probabilities, which risk privacy leakage, we derive the weights solely from entropies, since they are inexpensive to compute and reflect uncertainty.

Algorithm 1 Algorithm of FedCC pipeline   
Require: K clients, global rounds T, local labeled dataset   
$\{ \mathbb { D } _ { l } ^ { ( k ) } \} _ { k = 1 } ^ { K } ,$ public unlabeled dataset $\mathbb { D } _ { u } .$   
Training:   
1: for $t = 1$ to $T$ do   
Client k (in parallel):   
2: Obtain model $\theta _ { s y n c } ^ { ( k ) , t }$ by solving the optimization prob  
lem specified in Eq. 2. (Omit when $t = 1 )$   
3: $\theta _ { 0 } ^ { ( \lambda ) , t }$ ← arg min <sub>(k)</sub> $\mathcal { L } _ { l } ( \theta ^ { k } )$ ▷ Initialize at $\theta _ { s y n c } ^ { ( k ) , t }$   
4: $\check { \theta ^ { ( k ) , t } } \gets \check { \mathrm { F E D C C } } _ { - } \mathrm { L O C A L } \big ( \mathring { \theta } _ { 0 } ^ { ( k ) , t } , \mathbb { D } _ { l } ^ { ( k ) } , \mathbb { D } _ { u } \big )$   
5: Compute pseudo-predictions:   
$\hat { \mathbf { y } } ^ { ( k ) , t } \bar { ( x _ { u } ) } \overset { \cdot } {  } \sigma _ { \tau } ^ { ( k ) } \bar { ( x _ { u } ; \theta ^ { ( k ) , t } ) } \forall x _ { u } \in \mathbb { D } _ { u }$   
6: Upload $\{ \hat { \mathbf { y } } ^ { ( k ) , t } ( x _ { u } ) \}$ to server   
Server:   
7: $\mu ^ { ( k ) , t } ( x _ { u } ) \gets \mathrm { n o r m a l i z e } \ \{ 1 - \hat { y } _ { u } ^ { ( k ) , t } ( x _ { u } ) \} _ { k }$   
8: $\begin{array} { r } { \tilde { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } ) \gets \sum _ { k } \mu ^ { ( k ) , t } ( \tilde { x } _ { u } ) \hat { \mathbf { y } } ^ { ( k ) , t } ( x _ { u } ) } \end{array}$   
9: Set the unknown-class entry of $\tilde { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } )$ to zero, and   
renormalize the remaining entries to get $\hat { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } ) \in \mathbb { R } ^ { C }$   
10: Broadcast $\{ \hat { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } ) \}$ back to all clients   
11: end for   
12: return $\{ \theta ^ { ( k ) , T } \} _ { k = 1 } ^ { K }$   
13: procedure FEDCC\_LOCAL $( \boldsymbol { \theta } _ { 0 } ^ { ( k ) , t } , \mathbb { D } _ { l } ^ { ( k ) } , \mathbb { D } _ { u } )$   
14: Generate pseudo-labels via Eq. 8 with $\alpha ^ { ( k ) , t } ( x _ { u } )$ from   
Eq. 10   
15: Obtain model $\theta ^ { ( k ) , t }$ by optimizing Eq. 4   
16: return $\theta ^ { ( k ) , t }$   
17: end procedure   
Inference: Given a new input x   
18: Generate $\tilde { \mathbf { y } } ^ { ( s ) , T } ( x _ { u } )$ via Eq. 11   
$1 9 \colon \hat { y } = \arg \operatorname* { m a x } _ { c \in \mathbb { C } } \tilde { \mathbf { y } } _ { c } ^ { ( s ) , T } ( x _ { u } )$

Weight towards majority classes (w<sub>l</sub>): Following [27], we regard a solid-color image $x _ { 0 }$ as featureless, and probe the classifier’s intrinsic bias by feeding it into the pre-trained model, recording $\mathbf { p } ^ { ( k ) } : = \sigma ( x _ { 0 } ; \theta _ { 0 } )$ . To keep our entropy-based weighting scheme responsive even for near-certain predictions, we apply label smoothing [28]:

$$
\tilde { \mathbf { p } } ^ { ( k ) } = \left( 1 - \delta \right) \mathbf { p } ^ { ( k ) } + \delta \frac { \mathbf { 1 } } { | \mathbb { C } | }\tag{9}
$$

The weight is then $w _ { l } : = H \big ( \tilde { \mathbf { p } } ^ { ( k ) } \big )$ , where $H ( \cdot )$ denotes entropy. The smoothed entropy quantifies how strongly the model already favors its local label set and stays strictly positive even under a one-class bias.

Weight over unknown classes $( w _ { u } ) \mathbf { : }$ Classes under-represented by client k are collected in $\mathbb { C } _ { N } ^ { ( k ) } \bigcup \mathbb { C } _ { S } ^ { ( k ) }$ with number being $| \mathbb { C } | - | \mathbb { C } _ { J } ^ { ( k ) } |$ . Since the client lacks prior knowledge for those classes, we assign them the maximum uncertainty, defined as the entropy of a uniform distribution. This yields $w _ { u } : =$ $\log ( | \mathbb { C } | - | \bar { \mathbb { C } } _ { J } ^ { ( k ) } | + 1 )$ . Here, we add one virtual category to account for residual ambiguity introduced by the majority classes, reflecting that the client remains uncertain even with only one under-represented class.

Since $w _ { l }$ quantifies the client’s concentration of probability over majority classes and $w _ { u }$ captures its uncertainty about unknown classes, a natural way to trade off these two effects is through their ratio: $\begin{array} { r } { \alpha _ { b a s e } = \frac { w _ { l } } { w _ { l } + w _ { u } } \in [ 0 , 1 ] } \end{array}$ . Note that $w _ { l }$ and $w _ { u }$ are client-specific and vary across clients.

To further account for sample-specific prediction uncertainty, we quantify the pretrained model’s confidence for each unlabeled instance $x _ { u }$ by $w _ { x _ { u } } : = \operatorname* { m i n } \left( w _ { l } , H ( \sigma ( x _ { u } ; \theta _ { 0 } ) ) \right)$ . As high entropy indicates low confidence in assigning the sample to a class in $\mathbb { C } .$ , we penalize $\alpha _ { b a s e }$ with $w _ { x _ { u } }$ , leading to the final $\alpha ( x _ { u } )$ expression:

$$
\alpha ( x _ { u } ) : = \frac { w _ { l } - w _ { x _ { u } } } { w _ { l } + w _ { u } }\tag{10}
$$

This weighting factor α limits the extent to which a local model’s bias affects pseudo-labeling. α reserves a non-zero probability for the ‘unknown’ class, allowing the model to consider an alternative label rather than forcing a majority-class assignment. By deferring these uncertain cases, α curbs early mislabelling of minority samples and prevents such errors from cascading through subsequent training rounds.

After generating pseudo-labels, each client updates its local model via Eq. 4 to obtain θ, which is then used to generate predictions shared with the server.

## C. Federated training pipeline

The training process runs over $T$ communication rounds. For notational clarity, we use the superscript (k) and (s) to denote the client index and the server-aggregated information, respectively, and t to indicate the communication round.

At the beginning of round $t ,$ each client k downloads the aggregated soft targets from the previous round, denoted as $\hat { y } ^ { ( s ) , t - 1 } ( x _ { u } ) , \forall x _ { u } \in \mathbb { D } _ { u }$ , and then use Eq. 2 to synchronize its local model. Then each client adapts its model to its private labeled dataset $\mathbb { D } _ { l } ^ { ( k ) }$ , resulting in the intermediate model $\bar { \theta } _ { 0 } ^ { ( k ) , t }$ . Unlike prior distillation-based FL schemes that share predictions immediately, FedCC inserts an extra refinement step. Each client generates the pseudo-label via Eq. 8 and obtains the final round-specific parameters $\theta ^ { ( k ) , t }$ by optimizing the objective in Eq. 4. Each client then evaluates all samples in $\mathbb { D } _ { u }$ and produces temperature-scaled prediction vectors, $\hat { \mathbf { y } } ^ { ( k ) , t } ( x _ { u } ) \stackrel {  } { = } \sigma _ { \tau } ^ { ( k ) } ( x _ { u } ; \theta ^ { ( k ) , t } ) \in \mathbb { R } ^ { C + 1 }$ , where the last entry corresponds to the unknown class. These prediction vectors are uploaded to the server. Clients’ contributions are weighted by known-class confidence: we take $1 - \hat { y } _ { u } ^ { ( k ) , t } ( x _ { u } )$ as a preliminary weight, then normalize it across clients to produce $\bar { \mathbf { \rho } } _ { \mu } ( k ) , t  ( x _ { u } )$ The server then aggregates the received predictions as

$$
\tilde { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } ) = \sum _ { k } \mu ^ { ( k ) , t } ( x _ { u } ) \hat { \mathbf { y } } ^ { ( k ) , t } ( x _ { u } )\tag{11}
$$

It then zeroes out the unknown-class component and renormalizes the remaining $C$ entries to yield a valid probability vector $\hat { \mathbf { y } } ^ { ( s ) , t } ( x _ { u } ) \in \bar { \mathbb { R } } ^ { C }$ . The resulting soft targets are then redistributed to all clients to initiate round t + 1.

During inference, we adopt the ensemble strategy used in FedMD [29] and FedOV [14]. After completing all $T$ rounds, clients transmit their model $f ^ { ( k ) }$ with parameters $\theta ^ { ( k ) , T }$ to the server. The server then aggregates the predictions from all client models using Eq. 11 and selecting the most probable label among the original classes $\mathbb { C } .$ Formally, the predicted label is given by $\hat { y } = \arg \operatorname* { m a x } _ { c \in \mathbb { C } } \tilde { \mathbf { y } } _ { c } ^ { ( s ) , T } ( x _ { u } )$

Algorithm 1 details the complete FedCC procedure.

## V. EXPERIMENTS

## A. Experimental details

Dataset settings: Following the previous work [18], [19], we conduct classification tasks using CIFAR-10, CIFAR-100 and TinyImageNet datasets. We randomly divide each dataset into two parts: a public dataset and a private dataset. The public dataset, which comprises 10% of the total training samples, has all labels removed and is shared among all participants. To simulate different label skews in the private dataset, we use two partition methods: (1) Dirichlet distribution $p \sim \operatorname { D i r } ( \eta )$ for each class $c ,$ we sample $p _ { c } \sim \operatorname * { D i r } _ { K } ( \eta )$ , then assign client k a fraction $p _ { c , k }$ of the class-c samples to its private set. A lower η value indicates greater heterogeneity. (2) Pathological partition $| \mathbb { C } _ { J } ^ { ( k ) } | = N$ : each client has data from a fixed number of classes $N$ , with an equal number of samples for each class distributed among the assigned clients.

Baseline: We benchmark FedCC against nine parameter-based methods (FedAvg [1], FedProx [11], FedNova [10], FedRS [17], FedSAM [16], FedLC [18], FedMR [13], FedConcat [20], FedVLS [19]) and six distillation-based methods (FedMD [29], FedDF [9], LSR [28], FedET [30], FedHKT [23], FedCT [24]). For fairness, we exclude methods requiring a labeled public dataset or those based on generating labeled data. For details, please refer to the corresponding paper.

Experiment settings: Inspired by [23], parameter-based FL uses ResNet20 on all nodes; in distillation-based FL, clients split evenly between ResNet14 and ResNet20. The setup includes 10 clients, 15 global epochs, and 20 local epochs per step in client model training. We adopt Adam optimizer with batch size 64, learning rate 0.001. Temperature τ is set to 1, tradeoff λ in Eq. 4 to 0.2, δ in $\tilde { \mathbf { p } } ^ { ( k ) }$ (Eq. 9) to $1 0 ^ { - 3 }$ We categorize classes by client k’s bias $\mathbf { p } ^ { ( k ) }$ (Eq. 9): a class c is a majority if $p _ { c } ^ { ( k ) } > 2 \%$ , minority if $0 < p _ { c } ^ { ( k ) } \le 2 \% ,$ and missing if $p _ { c } ^ { ( k ) } = 0 .$ . We report the results based on three experiments conducted with different random seeds, presenting model accuracy with the percent sign omitted from the reported values.

TABLE I: Performance comparison under various label skews. FedCC demonstrates superior performance across almost all scenarios, and an increasing performance gap as label skew intensifies.
<table><tr><td rowspan=2 colspan=1>Method</td><td rowspan=1 colspan=3>TinyImageNet</td><td rowspan=1 colspan=7>CIFAR-100</td><td rowspan=1 colspan=4>CIFAR-10</td></tr><tr><td rowspan=1 colspan=2>p ∼ Dir(η) $0 . 5$      0.05</td><td rowspan=1 colspan=1> $\mathbf { \frac { 1 } { 4 0 } } ^ { ( \mathbb { C } _ { J } ^ { ( k ) } | = N } \mathbf { \frac { \sigma } { 2 0 } }$ </td><td rowspan=1 colspan=3> $p \sim \operatorname { D i r } ( \eta )$ 0.5     0.05</td><td rowspan=1 colspan=4> $\overline { { { _ { 2 0 } ^ { ( \mathbb { C } _ { J } ^ { ( k ) } | = N } } } }$ </td><td rowspan=1 colspan=3> $p \sim \mathrm { D i r } ( \eta )$ 0.5     0.05</td><td rowspan=1 colspan=1> $\begin{array} { r } { \frac { | \mathbb { C } _ { J } ^ { ( k ) } | = N } { 2 } } \end{array}$ </td></tr><tr><td rowspan=3 colspan=1>FedAvgFedProxFedNova</td><td rowspan=11 colspan=2> $3 6 . 2 { \scriptstyle \pm 1 . 6 }$   $2 4 . 1 _ { \pm 1 . 3 }$  $3 5 . 0 { \scriptstyle \pm 3 . 0 }$   $2 6 . 4 { \pm } 1 . 0 $  $3 4 . 9 { \scriptstyle \pm 1 . 1 }$   $2 5 . 7 _ { \pm 0 . 6 }$  $2 2 . 1 { \pm } 1 . 0 $  $3 0 . 3 { \scriptstyle \pm 1 . 1 }$  $2 8 . 0 { \overline { { \pm } } } 1 . 0 \ $  $3 4 . 8 { \overline { { \pm } } } 0 . 9$   $2 4 . \underline { { 7 } } \pm 0 . 5$  $2 8 . 3 { \scriptstyle \pm 2 . 5 }$   $1 8 . \underline { { 7 } } \pm 3 . 9$  $2 6 . 4 { \scriptstyle \pm 1 . 4 }$   $2 0 . 7 _ { \pm 0 . 4 }$ </td><td rowspan=2 colspan=1> $3 1 . 1 { \pm } 1 . 9$   $1 4 . 3 { \scriptstyle \pm 0 . 8 }$ </td><td rowspan=1 colspan=3> $4 9 . 3 { \scriptstyle \pm 1 . 2 }$   $3 4 . 2 { \scriptstyle \pm 2 . 1 }$ </td><td rowspan=1 colspan=4> $3 3 . 5 { \scriptstyle \pm 1 . 3 }$   $1 8 . 2 { \scriptstyle \pm 1 . 3 }$ </td><td rowspan=1 colspan=3> $7 2 . 6 { \scriptstyle \pm 2 . 6 }$   $4 4 . 3 { \scriptstyle \pm 4 . 2 }$ </td><td rowspan=1 colspan=1> $3 7 . 5 { \scriptstyle \pm 1 . 0 }$   $1 1 . 0 { \scriptstyle \pm 1 . 1 }$ </td></tr><tr><td rowspan=10 colspan=1> $2 7 . 2 { \scriptstyle \pm 1 . 3 }$   $1 4 . 1 { \pm } 1 . 0 $  $2 7 . 7 { \scriptstyle \pm 1 . 4 }$   $1 2 . 9 { \scriptstyle \pm 1 . 5 }$  $2 9 . 4 { \scriptstyle \pm 1 . 2 }$   $1 4 . 1 { \pm } 0 . 9$  $2 8 . 8 { \scriptstyle \pm 1 . 0 }$   $1 2 . 5 { \pm } 1 . 3 $  $2 9 . 2 { \scriptstyle \pm 0 . 8 }$   $1 5 . 1 { \pm } 1 . 3$  $2 2 . 3 { \scriptstyle \pm 2 . 3 }$   ${ \underline { { 9 . 9 } } } { \scriptstyle \pm 1 . 2 }$  $2 2 . 4 { \scriptstyle \pm 0 . 7 }$   $1 5 . 3 { \scriptstyle \pm 2 . 6 }$ </td><td rowspan=1 colspan=2> $4 8 . 9 2 0 . 9$   $2 9 . 3 { \scriptstyle \pm 0</td><td rowspan=1 colspan=1>. 7 }$ </td><td rowspan=1 colspan=4> $3 0 . 6 { \overline { { \pm } } } 1 . 8 \ $   $1 6 . 1 { \scriptstyle \pm 0 . 6 }$ </td><td rowspan=1 colspan=3> $7 0 . 1 { \scriptstyle \pm 2 . 3 }$   $4 7 . 2 { \scriptstyle \pm 2 . 0 }$ </td><td rowspan=1 colspan=1> $4 1 . 0 { \pm } 1 . 3$   $1 0 . 6 { \scriptstyle \pm 0 . 4 }$ </td></tr><tr><td rowspan=9 colspan=3> $4 7 . 8 { \scriptstyle \pm 1 . 7 }$   $3 3 . 7 { \scriptstyle \pm 1 . 7 }$  $4 6 . 3 { \scriptstyle \pm 0 . 8 }$   $2 7 . 8 { \scriptstyle \pm 0 . 9 }$  $4 8 . \underline { { 7 } } \pm 1 . 3$   $3 3 . 7 { \scriptstyle \pm 1 . 3 }$  $4 8 . 5 { \scriptstyle \pm 1 . 1 }$   $3 4 . 6 { \scriptstyle \pm 0 . 7 }$  $5 0 . 1 { \pm } 1 . 2 $   $3 4 . 6 { \pm } 0 . 9$  $4 0 . 3 { \scriptstyle \pm 2 . 4 }$   $2 1 . 2 { \pm } 1 . 7 $  $4 3 . 5 { \scriptstyle \pm 1 . 6 }$   $2 7 . 3 { \scriptstyle \pm 1 . 2 }$ </td><td rowspan=1 colspan=2>+1.7</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=2> $3 7 . 9 { \scriptstyle \pm 1 . 5 }$   $2 0 . 9 { \scriptstyle \pm 1 . 1 }$ </td><td></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3> $7 5 . 3 { \scriptstyle \pm 1 . 1 }$   $4 2 . 5 { \scriptstyle \pm 0 . 7 }$ </td><td></td></tr><tr><td rowspan=6 colspan=1>FedRS $\mathrm { F e d S A M }$ FedLCFedMR</td><td rowspan=6 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td rowspan=2 colspan=1> $3 5 . 2 { \scriptstyle \pm 0 . 8 }$   $1 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td rowspan=2 colspan=3> $3 4 . 2 _ { \pm 1 . 4 } ^ { - }$   $1 7 . 7 { \scriptstyle \pm 1 . 2 }$  $3 5 . 3 { \scriptstyle \pm 1 . 7 }$   $1 6 . 5 { \scriptstyle \pm 0 . 8 }$  $3 9 . 4 \overline { { \pm } } 1 . 1$   $2 1 . 4 { \scriptstyle \pm 0 . 4 }$ </td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=3> $7 6 . 6 { \scriptstyle \pm 1 . 9 }$   $5 3 . 0 \overline { { \pm } } 1 . 0$  $7 4 . 6 { \scriptstyle \pm 0 . 6 }$   $4 3 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td rowspan=2 colspan=1> $3 6 . 2 { \scriptstyle \pm 1 . 6 }$   $1 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td rowspan=1 colspan=2>510.8</td><td rowspan=1 colspan=1> $7 6 . 0 \overline { { \pm } } 1 . \overset { . } { 2 }$ </td><td rowspan=1 colspan=2> $4 1 . 5 { \scriptstyle \pm 0 . 6 }$ </td></tr><tr><td rowspan=4 colspan=4> $3 8 . 6 { \pm } 1 . 5$   $2 1 . 8 { \pm } 1 . 0$  $2 0 . 6 { \scriptstyle \pm 0 . 4 }$   $1 4 . 7 { \scriptstyle \pm 1 . 7 }$  $2 7 . 1 _ { \pm 3 . 5 }$   $1 8 . 3 { \scriptstyle \pm 0 . 3 }$ </td><td rowspan=2 colspan=2>+1.0</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2>76.2±0.2</td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>FedConcatFedVLS</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3> $5 7 . 8 { \scriptstyle \pm 1 . 0 }$  $6 1 . 3 { \scriptstyle \pm 1 . 0 }$   $3 4 . 0 _ { + 2 \ 1 } ^ { - }$  $6 6 . 0 { \scriptstyle \pm 2 . 8 }$   $3 9 . 5 _ { \pm 0 . 8 } ^ { - }$ </td><td rowspan=1 colspan=1> $3 8 . 4 { \scriptstyle \pm 0 . 9 }$   $1 1 . 7 { \scriptstyle \pm 0 . 2 }$  $2 8 . 5 { \pm } 1 . 2 $   $1 2 . 7 \pm 0 . 7$  $3 2 . 9 { \scriptstyle \pm 3 . 6 }$   $1 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td rowspan=5 colspan=1>FedMDFedDFLSRFedETFedHKTFedCT</td><td rowspan=5 colspan=2> $3 0 . 1 { \scriptstyle \pm 1 . 7 }$   $2 4 . 8 { \scriptstyle \pm 1 . 5 }$  $2 6 . 8 { \scriptstyle \pm 2 . 0 }$   $1 6 . 9 \overline { { \pm } } 2 . 7$  $2 6 . 2 { \scriptstyle \pm 0 . 9 }$   $1 6 . 7 \pm 1 . 1$  $2 3 . 8 { \scriptstyle \pm 3 . 8 }$   $1 8 . 5 { \scriptstyle \pm 2 . 0 }$  $2 7 . 7 { \scriptstyle \pm 1 . 0 }$   $2 1 . 3 \overline { { \pm } } 0 . 9$  $2 5 . 9 { \scriptstyle \pm 0 . 5 }$   $1 7 . 7 _ { \pm 1 . 3 }$ </td><td rowspan=5 colspan=1> $3 0 . 2 { \scriptstyle \pm 0 . 8 }$   $2 3 . 1 { \pm } 1 . 6 $  $2 0 . 7 { \scriptstyle \pm 0 . 6 }$   $1 0 . 3 { \pm } 2 . 8 $  $2 3 . 8 { \scriptstyle \pm 1 . 2 }$   $1 0 . 2 \overline { { \pm } } 0 . \bar { 6 }$  $2 1 . 3 { \scriptstyle \pm 0 . 8 }$   $1 1 . 6 { \scriptstyle \pm 1 . 0 }$  $2 6 . 9 \overline { { \pm 1 . 7 } }$   $1 4 . 6 { \overline { { \pm } } } 0 . { \overset { . } { 4 } }$  $2 1 . 2 _ { \pm 0 . 7 }$   $9 . 9 { \scriptstyle \pm 2 . 6 }$ </td><td rowspan=3 colspan=3> $3 6 . 4 { \scriptstyle \pm 0 . 7 }$   $2 8 . 1 { \scriptstyle \pm 0 . 7 }$  $3 3 . 6 { \scriptstyle \pm 1 . 2 }$   $2 2 . 1 { \pm } 0 . 8$  $3 1 . 8 { \pm } 1 . 6 $   $2 1 . 3 { \pm } 0 . 8 $ </td><td rowspan=1 colspan=4> $3 4 . 9 { \scriptstyle \pm 1 . 2 }$   $2 6 . 4 { \scriptstyle \pm 1 . 6 }$ </td><td rowspan=1 colspan=3> $6 9 . 0 { \scriptstyle \pm 0 . 7 }$   $3 4 . 2 { \scriptstyle \pm 2 . 6 }$ </td><td rowspan=1 colspan=1>54.5±1.0  $1 0 . 7 { \scriptstyle \pm 0 . 9 }$ </td></tr><tr><td rowspan=2 colspan=4> $2 7 . 6 { \scriptstyle \pm 3 . 1 }$   $1 4 . 1 { \pm } 1 . 1$  $2 7 . 8 \overline { { \pm 1 } } . 4$   $1 6 . 5 { \pm } 0 . 3 $ </td><td rowspan=2 colspan=3> $6 3 . 9 { \pm } 1 . 9$   $3 9 . 9 \overline { { \pm } } 3 . \overline { { \tau } }$  $5 8 . 6 { \scriptstyle \pm 1 . 0 }$   $3 1 . 4 \overline { { \pm } } 0 . 7$ </td><td rowspan=2 colspan=1> $5 0 . 8 { \scriptstyle \pm 1 . 4 }$   $1 1 . 7 { \scriptstyle \pm 1 . 0 }$  $2 7 . 2 { \scriptstyle \pm 1 . 4 }$   $1 0 . 0 \overline { { \pm } } 0 . \dot { 0 }$ </td></tr><tr><td rowspan=1 colspan=1>7.8+1.4 16.5-</td></tr><tr><td rowspan=1 colspan=3> $2 7 . 2 { \scriptstyle \pm 0 . 9 }$   $2 2 . 3 { \scriptstyle \pm 1 . 7 }$ </td><td rowspan=1 colspan=1>1.7</td><td rowspan=1 colspan=3> $2 8 . 0 _ { \pm 1 . 1 } ^ { - }$   $2 2 . 1 { \pm } 0 . 8$ </td><td rowspan=1 colspan=1>0.8</td><td rowspan=1 colspan=3> $6 2 . 1 { \pm } 2 . 0 $   $3 8 . 4 \overline { { \pm } } 2 . 7$ </td><td rowspan=1 colspan=1> $5 1 . 2 { \scriptstyle \pm 0 . 6 }$   $1 0 . 0 { \scriptstyle \pm 0 . 0 }$ </td></tr><tr><td rowspan=1 colspan=3> $3 4 . 4 { \scriptstyle \pm 1 . 9 }$   $2 5 . 6 { \scriptstyle \pm 0 . 5 }$  $3 3 . 2 _ { \pm 0 . 4 }$   $2 1 . 8 { \scriptstyle \pm 1 . 6 }$ </td><td rowspan=1 colspan=4> $2 6 . 7 _ { \pm 0 . 8 }$   $1 2 . 2 { \scriptstyle \pm 2 . 3 }$ </td><td rowspan=1 colspan=3> $6 7 . 6 { \overset { - } { \pm 1 } } . 4$   $4 4 . 9 { \overline { { \pm } } } 0 . 9$  $6 5 . 0 \overline { { \pm } } 1 . 2$  $2 9 . 7 _ { \pm 1 . 7 } ^ { - }$ </td><td rowspan=1 colspan=1> $4 9 . 1 \overset { - } { \pm } 0 . 4$   $1 1 . 1 { \pm } 1 . 2 $  $2 4 . 1 _ { \pm 0 . 2 }$   $1 0 . 0 \overline { { \pm } } 0 . 0$ </td></tr><tr><td rowspan=1 colspan=3>FedCC   $\mid 4 0 . 6 { \scriptstyle \pm 0 . 8 }$   $3 4 . 8 { \scriptstyle \pm 1 . 0 }$ </td><td rowspan=1 colspan=1> $3 6 . 9 { \scriptstyle \pm 2 . 0 }$   $3 4 . 3 { \pm } 1 . 3 $ _</td><td rowspan=1 colspan=3> $4 8 . 9 { \pm } 1 . 3$   $4 0 . 5 { \pm } 1 . 8 $ </td><td rowspan=1 colspan=4> $4 5 . 3 { \pm } 1 . 4 $   $4 2 . 6 { \pm } 1 . 3$ 一</td><td rowspan=1 colspan=3> $7 4 . 7 { \pm } 1 . 6$  $6 0 . 9 { \scriptstyle \pm 2 . 2 }$ </td><td rowspan=1 colspan=1> $6 8 . 2 { \scriptstyle \pm 1 . 1 }$   $6 7 . 3 { \pm } 1 . 8 $ </td></tr></table>

TABLE II: Local model performance over local and global test data, where FedCC leads in almost every case. (CIFAR-100 subset)
<table><tr><td>Test data</td><td>Local model</td><td colspan="2"> $p \sim \operatorname { D i r } ( \eta )$  0.5 0.05</td><td colspan="2"> $| \mathbb { C } _ { J } ^ { ( k ) } | = N _ { 1 0 }$ </td></tr><tr><td>Local</td><td>FedAvg FedLC FedMR SSL FedCC</td><td> $3 7 . 9 { \scriptstyle \pm 2 . 2 }$   $3 6 . 1 \pm 1 . 4$   $3 6 . 7 { \scriptstyle \pm 2 . 2 }$   $3 4 . 6 { \scriptstyle \pm 2 . 0 }$ </td><td> $6 0 . 0 { \scriptstyle \pm 2 . 7 }$   $6 0 . 5 { \overline { { \pm } } } 4 . 4 \ $   $6 0 . 2 { \scriptstyle \pm 5 . 1 }$   $5 8 . 0 { \scriptstyle \pm 2 . 9 }$ </td><td> $5 7 . 4 { \scriptstyle \pm 3 . 1 }$   $5 6 . 9 \pm 5 . 1$   $5 6 . 7 { \scriptstyle \pm 4 . 7 }$   $5 4 . 6 { \scriptstyle \pm 2 . 0 }$ </td><td> $7 2 . 3 { \scriptstyle \pm 4 . 9 }$   $7 2 . 2 { \overset { - } { \pm } } 6 . 0$   $7 0 . 6 { \scriptstyle \pm 5 . 3 }$   $\underline { { 7 0 . 5 } } \pm 5 . 2$ </td></tr><tr><td>Global</td><td>FedAvg FedLC FedMR SSL FedCC</td><td> $3 8 . 2 \overline { { \pm } } 2 . 3$   $1 7 . 6 { \scriptstyle \pm 1 . 4 }$   $1 9 . 0 { \scriptstyle \pm 1 . 0 }$   $1 9 . 3 { \pm } 1 . 5$   $1 5 . 4 { \scriptstyle \pm 1 . 1 }$   $7 1 . 1 { \pm } 0 . 3$ </td><td> $6 4 . 4 { \scriptstyle \pm 2 . 6 }$   $1 0 . 1 { \pm } 0 . 9$   $1 1 . 0 { \scriptstyle \pm 1 . 6 }$   $1 0 . 7 _ { \pm 1 . 7 } ^ { - }$   $\underline { { 1 0 . 2 } } \underline { { \pm } } 1 . 3$   $7 4 . 7 { \scriptstyle \pm 1 . 2 }$ </td><td> $6 0 . 0 { \scriptstyle \pm 4 . 6 }$   $1 1 . 3 { \scriptstyle \pm 0 . 8 }$   $1 1 . 8 { \scriptstyle \pm 1 . 0 }$   $1 1 . 6 _ { \pm 1 . 1 }$   $\underline { { 1 } } 1 . 4 \pm 0 . 9$   $7 1 . 1 { \pm 2 . 4 }$ </td><td> $7 6 . 6 { \scriptstyle \pm 3 . 3 }$   $\underline { { 7 . 2 } } \underline { { \pm } } 0 . 6$   $7 . 0 { \scriptstyle \pm 0 . 8 }$   $7 . 0 { \scriptstyle \pm 0 . 8 }$   $7 . 3 { \scriptstyle \pm 0 . 5 }$   $7 7 . 9 \pm 3 . 8$ </td></tr></table>

![](images/4c339f9707644dde99ac282230d19baca4e724cf8636540b5b6d3c16678bc9bc.jpg)

![](images/a11ae8d82d3e56d6275eca1ec5e22d7f613fbe9f7509c9c13334895833543ac7.jpg)

Fig. 2: Distribution of decision margins for local model on CIFAR-10.  
(a) p ∼ Dir(0.05)  
![](images/1344fb8ee96da8518cdfefc824cfac47beeb8fe232e71e20f504703fd0ea9d26.jpg)  
(b) $| \mathbb { C } _ { J } ^ { ( k ) } | = 2$

(a) FedAvg local (b) FedAvg global (c) FedCC local (d) FedCC global (a) FedAvg local (b) FedAvg global (c) FedCC local (d) FedCC global

Fig. 3: T-SNE visualizations of learned representations on CIFAR-10 where the local distribution follows ${ | \mathbb { C } _ { J } ^ { ( k ) } } ^ { \binom { s } { k } } = 1 |$

## B. An overall comparison

Comparison of local model performance: We evaluate the performance of local models on both local and global test data, as shown in Table II. The models are generated using FedAvg [1], FedLC [18], FedMR [13], standard SSL [25], and our FedCC. For the global test data, if the true label does not belong to the majority classes, classifying it as the ‘unknown’ class is considered valid. As expected, FedCC shows significant improvement on the global test data. On the local test set, FedCC leverages uncertainty to supply richer signals for minority classes and increases their likelihood of being correctly learned, consistently outperforming all baselines.

Experiment results, as summarized in Table I, show that FedCC outperforms current FL methods across almost all tested scenarios, with its advantage widening as label skew intensifies. On heavy skew $( p \sim \operatorname { D i r } ( 0 . 0 5 )$ or pathological partition), it is ahead by at least 3% points and often by more than 10%. In the harshest scenario where each client has access to only one class of CIFAR-10, baselines collapse to near-random guessing $( \leq 1 2 . 7 \% )$ , yet FedCC still delivers a robust 67.3%. This robustness stems from permitting clients to only predict familiar data and to label uncertain samples as ‘unknown’, thereby abstaining from unreliable predictions and reducing misleading information.

## C. Performance visualization

Decision margin: The decision margin measures the gap between the model’s score for the correct class and its highest incorrect guess. Evaluating under two challenging data partitions, Figure 2 shows margin histograms with a logarithmic y-axis. FedCC markedly shifts the distribution rightward to achieve a positive mean, whereas the best competing method remains negative. This rightward shift confirms that FedCC effectively ensures clear class separation and suppresses client side misclassification, even under severe label skew.

T-SNE analysis of learned representations: Figure 3 visualizes the learned feature representations where each client holds data from a single class $( | \mathbb { C } _ { J } ^ { ( k ) } | = 1 )$ on CIFAR-10, where each color denotes a distinct class. The embeddings are obtained by applying PCA whitening followed by t-SNE in a unified pipeline. Baseline: each client trains a local model using standard cross-entropy on its data, with the embedding of a representative client shown in Fig.3a, and that of the FedAvg global model in Fig.3b. Due to the lack of inter-class variation, local representations collapse along class-specific axes, and averaging these misaligned feature spaces yields weak global performance. FedCC: we next train local models with FedCC’s objective and aggregate them by ensembling their soft predictions. The corresponding client-side embedding (Fig.3c) cleanly isolates the client’s own class from the “unknown” region, and the aggregated embedding (Fig.3d) recovers distinct clusters for all classes. FedCC thus curbs error propagation by explicitly separating learned versus unseen classes at the client level, allowing the server to combine information without blurring minority categories.

## VI. CONCLUSION

We present FedCC, a novel distillation-based FL algorithm that mitigates client misclassification by incorporating an auxiliary ‘unknown’ class. By enabling clients to defer ambiguous predictions to this class, FedCC acts as an implicit regularizer, preventing overconfidence on majority classes and suppressing noisy updates. Extensive empirical evaluation confirms that it consistently outperforms state-of-the-art baselines in both local and global performance. Crucially, FedCC’s lightweight reliance on logit exchanges and robust performance make it well suited for deployment in real-world communication networks, which involves large numbers of devices with heterogeneous compute power, bandwidth, and data distributions.

## REFERENCES

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, “Communication-efficient learning of deep networks from decentralized data,” in Artificial intelligence and statistics. PMLR, 2017.

[2] P. Kairouz, H. B. McMahan, B. Avent, A. Bellet, M. Bennis, A. N. Bhagoji et al., “Advances and open problems in federated learning,” Foundations and trends® in machine learning, 2021.

[3] W. Ye, C. Qian, X. An, X. Yan, and G. Carle, “Advancing federated learning in 6g: A trusted architecture with graph-based analysis,” in GLOBECOM 2023-2023 IEEE Global Communications Conference. IEEE, 2023, pp. 56–61.

[4] B. Liu, W. Y. Poe, R. Trivisonno, and G. Caire, “Foundation models for generalizable semantic and goal-oriented communication,” in ICC 2026 - IEEE International Conference on Communications, 2026, pp. 1–6.

[5] G. Hinton, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[6] W. Ye, Y. Zhang, X. An, G. Carle, and Y. Ma, “Select to think: Unlocking SLM potential with local sufficiency,” in Forty-third International Conference on Machine Learning, 2026.

[7] Z. Li, X. Wang, D. Hu, N. M. Robertson, D. A. Clifton, C. Meinel, and H. Yang, “Not all knowledge is created equal: Mutual distillation of confident knowledge,” in Workshop on trustworthy and socially responsible machine learning, NeurIPS 2022, 2022.

[8] R. Dai, Y. Zhang, A. Li, T. Liu, X. Yang, and B. Han, “Enhancing oneshot federated learning through data and ensemble co-boosting,” arXiv preprint arXiv:2402.15070, 2024.

[9] T. Lin, L. Kong, S. U. Stich, and M. Jaggi, “Ensemble distillation for robust model fusion in federated learning,” Advances in neural information processing systems, 2020.

[10] J. Wang, Q. Liu, H. Liang, G. Joshi, and H. V. Poor, “Tackling the objective inconsistency problem in heterogeneous federated optimization,” Advances in neural information processing systems, 2020.

[11] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” Proceedings of machine learning and systems, 2020.

[12] X. Gong, A. Sharma, S. Karanam, Z. Wu, T. Chen, D. Doermann, and A. Innanje, “Ensemble attention distillation for privacy-preserving federated learning,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021.

[13] Z. Fan, J. Yao, R. Zhang, L. Lyu, Y. Zhang, and Y. Wang, “Federated learning under partially class-disjoint data via manifold reshaping,” arXiv preprint arXiv:2405.18983, 2024.

[14] Y. Diao, Q. Li, and B. He, “Towards addressing label skews in one-shot federated learning,” in International conference on learning representations, 2023.

[15] W. J. Scheirer, A. de Rezende Rocha, A. Sapkota, and T. E. Boult, “Toward open set recognition,” IEEE transactions on pattern analysis and machine intelligence, 2012.

[16] Z. Qu, X. Li, R. Duan, Y. Liu, B. Tang, and Z. Lu, “Generalized federated learning via sharpness aware minimization,” in International conference on machine learning, 2022.

[17] X. Li and D. Zhan, “Fedrs: Federated learning with restricted softmax for label distribution non-iid data,” in Proceedings of the ACM SIGKDD conference on knowledge discovery & data mining, 2021.

[18] J. Zhang, Z. Li, B. Li, J. Xu, S. Wu, S. Ding, and C. Wu, “Federated learning with label distribution skew via logits calibration,” in International Conference on Machine Learning, 2022.

[19] K. Guo, Y. Ding, J. Liang, Z. Wang, R. He, and T. Tan, “Exploring vacant classes in label-skewed federated learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

[20] Y. Diao, Q. Li, and B. He, “Exploiting label skews in federated learning with model concatenation,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024.

[21] J. Lu, S. Li, K. Bao, P. Wang, Z. Qian, and S. Ge, “Federated learning with label-masking distillation,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023.

[22] D. Campos, M. Zhang, B. Yang, T. Kieu, C. Guo, and C. S. Jensen, “Lightts: Lightweight time series classification with adaptive ensemble distillation,” Proceedings of the ACM on management of data, 2023.

[23] Y. Deng, J. Ren, C. Tang, F. Lyu, Y. Liu, and Y. Zhang, “A hierarchical knowledge transfer framework for heterogeneous federated learning,” in IEEE conference on computer communications. IEEE, 2023.

[24] A. Abourayya, J. Kleesiek, K. Rao, E. Ayday, B. Rao, G. I. Webb, and M. Kamp, “Little is enough: Boosting privacy by sharing only hard labels in federated semi-supervised learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2025.

[25] D. Yarowsky, “Unsupervised word sense disambiguation rivaling supervised methods,” in Annual meeting of the association for computational linguistics, 1995.

[26] Y. Zou, Z. Yu, B. Kumar, and J. Wang, “Unsupervised domain adaptation for semantic segmentation via class-balanced self-training,” in Proceedings of the European conference on computer vision, 2018.

[27] H. Lee and H. Kim, “Cdmad: Class-distribution-mismatch-aware debiasing for class-imbalanced semi-supervised learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2024.

[28] L. Yuan, F. E. Tay, G. Li, T. Wang, and J. Feng, “Revisiting knowledge distillation via label smoothing regularization,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2020.

[29] D. Li and J. Wang, “Fedmd: Heterogenous federated learning via model distillation,” arXiv preprint arXiv:1910.03581, 2019.

[30] Y. J. Cho, A. Manoel, G. Joshi, R. Sim, and D. Dimitriadis, “Heterogeneous ensemble knowledge transfer for training large models in federated learning,” International joint conferences on artificial intelligence, 2022.