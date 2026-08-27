# CRAMER: Control via Request-Aware Masking for Editing Recommenders

Zhiyuan Julian Su <sup>1</sup> <sup>2</sup> <sup>†</sup> Naihe Feng <sup>2</sup> Zhen Luther Qin <sup>2</sup> Ga Wu <sup>2</sup>

## Abstract

Sequential recommendation models, while powerful, have limited flexibility in responding to immediate user requests, making it difficult to adapt their recommendations to the user’s timely interests. Unfortunately, existing user request adaptation methods often incur high computational overhead due to either 1) retraining the entire backbone network or 2) leveraging the inference ability of large language models (a.k.a. prompt engineering), limiting their applicability in large-scale recommendation services. This paper presents Control via Request-Aware Masking for Editing Recommenders (CRAMER), a framework that takes users’ natural-language requests to immediately change sequential recommendation models’ behavior. Specifically, inspired by the model control theory, CRAMER treats user requests as control signals to modulate frozen sequential recommender backbone parameters through masking, achieving instant adaptation to diverse requests while avoiding costly retraining. Experiments on multiple large-scale benchmark datasets show that CRAMER outperforms four state-of-the-art request-aware baselines across multiple recommendation metrics while achieving minimal overhead. Moreover, the proposed framework exhibits enhanced controllability and cross-domain adaptability, establishing a new paradigm for requestaware sequential recommendation.

## 1. Introduction

Sequential recommendation models have advanced the stateof-the-art in predicting users’ next interactions by modeling temporal patterns in behavior, but they remain inflexible when users issue immediate requests (Li et al., 2023;

Ye et al., 2025; Shen et al., 2026). Real-world users frequently express on-the-fly intents in response to an initial recommendation list (e.g., “I want more exciting games”), rather than issuing explicit search queries. Handling such reactive, post-recommendation feedback introduces challenges beyond conventional sequential recommendation. First, natural-language requests may emphasize or even contradict historical preferences, requiring the model to dynamically balance immediate intent with long-term behavior patterns (Gao et al., 2021; Radlinski et al., 2022; Jin et al., 2022; Chen et al., 2023). Second, requests often contain rich semantics such as negations, constraints, or fine-grained attribute preferences, which require accurate interpretation and controllable adjustment of recommendations (Jannach et al., 2021; Moradizeyveh, 2022). Finally, real-time adaptation must be achieved efficiently: sequential recommendation backbones are inherently complex, trained and deployed models, so request-aware extensions must rely on lightweight, parameter-efficient mechanisms that preserve responsiveness without full fine-tuning (Houlsby et al., 2019; Prottasha et al., 2024; Shao et al., 2025).

To address these gaps, existing research has primarily viewed request adaptation as either an input-level or outputlevel problem. Input-level approaches augment historical sequences with control tokens or vectors (He et al., 2022; Li et al., 2023), while output-level strategies employ reranking or filtering logic after the model has generated candidates (Liu et al., 2024; Liao et al., 2026; Zhang et al., 2025). Consequently, these methods face a dilemma: they either rely on shallow representations that cannot capture nuanced user intent, or necessitate domain-specific pretraining and heavyweight inference that disrupt service efficiency. More specifically, the domain-specific pretraining or fine-tuning concern mainly refers to language-to-item representation methods, where textual requests and items are aligned in a shared semantic space or interaction sequences are modeled in language space. In contrast, the heavyweight inference concern mainly refers to LLM-based reranking or reasoning methods; in this case, the additional cost comes from the request-aware inference procedure on top of the sequential backbone, rather than from the backbone itself. Our goal is not to replace periodic or continual retraining in production systems, but to provide short-horizon, request-conditioned behavioral adaptation between model updates, especially when a user issues an immediate natural-language request or post-hoc feedback. This trade-off stems from a fundamental limitation: they treat the recommender backbone as a static black box. Crucially, they lack mechanisms for posterior control—the ability to intervene directly on the internal mechanics of a deployed model to enforce constraints not present during training. Unlike traditional user-controllable recommendation, which relies on ante-hoc conditioning, our setting demands modifying the behavior of a frozen sequential backbone on the fly. This requires moving beyond simple conditioning to actively steering the model’s parameters in response to arbitrary and complex natural-language signals, a capability largely absent in current architectures.

We propose Control via Request-Aware Masking for Editing Recommenders (CRAMER), a lightweight framework that treats a user’s natural-language request as a control input and instantaneously modulates a frozen sequential recommender backbone via parameter masking. Drawing inspiration from model control theory (Li & Rush, 2020; Li et al., 2022), CRAMER applies learned masks to the backbone’s parameters so the model’s behavior is steered toward the requested intent with minimum computational overhead (Wen et al., 2016; Frankle & Carbin, 2019). CRAMER begins by mean-pooling across all token embeddings from the request to derive a faithful representation of the user’s immediate request (Mosbach et al., 2020). A Gumbel–Top-k step (Kool et al., 2019) then produces a sparse row–column gate vector, which is decomposed into per-matrix row and column gates and converted into entrywise masks applied to the selected matrices of the frozen Transformer-based sequential recommender. For robustness, CRAMER introduces three masking strategies for attention output matrices and feed-forward networks (FFNs) in the Transformer-based backbone. The training objective for the request-to-mask module combines a prediction loss with a KL regularizer that encourages the learned gate distribution to follow a sparsity prior (details in Appendix A). Empirically, this masking-based control achieves adaptation with minimal computational overhead, outperforms four state-of-the-art request-aware baselines on multiple large-scale benchmarks, offering a practical, scalable paradigm for request-aware sequential recommendation.

## 2. Background and Related Work

Sequential recommendation aims to predict the next interaction from historical sequences of consumed items, capturing the temporal dynamics beyond static profiles (Pan et al., 2024). In the past few years, Transformer-based models have become the predominant approach (Fang et al., 2020), among which SASRec (Kang & McAuley, 2018) and BERT4Rec (Sun et al., 2019) are the most representative, consistently serving as strong baselines across diverse

scenarios (Zivic et al., 2024).

Yet, current approaches struggle to adapt when users express immediate intent through natural-language requests (Li et al., 2023). In practice, an immediate request can emphasize aspects of prior preferences, or explicitly negate them (Wu et al., 2019; Luo et al., 2020), thereby calling for models that can adapt dynamically rather than relying solely on static long-term signals. Prior request-aware approaches fall into three strands: (i) request augmentation (He et al., 2022), which conditions sequential models on usergenerated tags or requests to capture short-term intent but relies on shallow representations; (ii) language-to-item representations, which pretrain encoders to bridge natural language and items (Hou et al., 2024) or model sequences directly in language space (Li et al., 2023), improving coverage and transfer but requiring domain-specific pretraining or fine-tuning and potentially adding inference cost; and (iii) LLM-based methods, which leverage large language models for recommendation via semantic enhancement (Liu et al., 2024), constrained generation (Liao et al., 2026), listwise reasoning re-rankers (Zhang et al., 2025), or dynamic userintent prompting (Xu et al., 2025), though effective, they are often overly complex and demand high computation and latency. These limitations motivate lightweight mechanisms that condition strong sequential backbones on immediate requests.

In the request-aware sequential recommendation mentioned above, retraining or fully fine-tuning complex Transformerbased backbones is computationally prohibitive for realtime adaptation. Since these models already encode longterm preference signals, it is common to keep the backbone frozen and introduce lightweight modules for parameterefficient adaptation (Su et al., 2025; Shen et al., 2025), as in natural language processing (NLP) (Son et al., 2025) and computer vision (CV) (Qin et al., 2024). Such approaches include prefix/prompt tuning (Li & Liang, 2021; Lester et al., 2021), which prepends small vectors for only coarse control; adapter modules (Houlsby et al., 2019), which insert trainable layers but increase latency; and latent token insertion (Sun et al., 2025), which offers flexible conditioning at the cost of additional parameters. Masking methods instead stand out by learning task-dependent masks over weights or activations, enabling reversible and fine-grained control without retraining (Zhao et al., 2020; Ansell et al., 2022; Litschko et al., 2022; Tao et al., 2023; Svirsky et al., 2024), though their potential for conditioning on naturallanguage requests remains underexplored. Overall, prior work on parameter-efficient adaptation primarily targets task- or domain-level transfer, whereas CRAMER enables instance-level, request-conditioned posterior control of a frozen sequential recommender at inference time.

## 3. Methodology

## 3.1. Task Definition

Let U and I denote the sets of users and items, respectively. For a user $u \in \mathcal { U } ,$ we represent the historical interaction sequence as $\pmb { s } _ { u } = ( i _ { 1 } , \dots , i _ { T } )$ with $i _ { t } \in \mathcal { T } , t \in \{ 1 , 2 , . . . , T \}$ , and denote the ground-truth next item by $i _ { T + 1 } ^ { \star } \in \mathcal { T }$ . In addition to these common notations of sequential recommendation, in the request-aware scenario, at time step $T { + 1 }$ the user u provides a natural-language request $\mathbf { q } _ { u } ,$ , which specifies the user’s immediate intent.

We consider a trained sequential recommender $f _ { \theta }$ with parameters θ. Given the interaction sequence $\mathbf { } _ { s \mu }$ and request $\mathbf { q } _ { u }$ of user u, the model scores each $i \in \mathcal { T }$ and predicts

$$
\hat { i } _ { T + 1 } = \arg \operatorname* { m a x } _ { i \in \mathcal { T } } f _ { \theta } ( i \ \vert \ s _ { u } , \mathbf { q } _ { u } ) .\tag{1}
$$

Ideally, we want this prediction to coincide with the groundtruth next item, i.e., $\hat { i } _ { T + 1 } = i _ { T + 1 } ^ { \star }$ . The overall training objective is to maximize the total conditional log-likelihood of ground-truth next items over all users, i.e.,

$$
\operatorname* { m a x } \sum _ { u \in \mathcal { U } } \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \mathbf { q } _ { u } ; \theta \big ) ,\tag{2}
$$

which amounts to encouraging the model to assign the highest probability to the ground-truth next item $i _ { T + 1 } ^ { \star }$ given both the historical sequence and the accompanying request, consistent with the ideal case of Equation (1). In contrast to conventional sequential recommendation that relies solely on $\mathbf { } _ { s _ { u } }$ and the backbone $f _ { \theta }$ , this formulation explicitly incorporates $\mathbf { q } _ { u }$ , allowing the model to reconcile long-term preferences with immediate intent.

To optimize Objective (2), existing methods take three main routes. Some manipulate $s _ { u } , \mathrm { e . g . }$ ., by augmenting or transforming it with $\mathbf { q } _ { u }$ (He et al., 2022; Li et al., 2023; Hou et al., 2024; Liu et al., 2024), but such strategies often yield shallow control. Others introduce auxiliary request-aware modules that fuse with the backbone (Liao et al., 2026; Zhang et al., 2025), at the cost of added latency and complexity. A more direct option is to fine-tune or retrain the backbone parameters θ based on $\mathbf { q } _ { u }$ , but this is computationally expensive and impractical—since θ is often trained, deployed, and frozen in practice. Thus, the key challenge is to control the frozen sequential recommender backbone model given $\mathbf { q } _ { u }$

This motivates a mapping $\mathcal { F } _ { \phi }$ with trainable parameters $\phi ,$ which transforms $\mathbf { q } _ { u }$ into the control signal vector $\pmb { m } \in \mathbb { R } ^ { d }$ Then, we apply m to θ through a series of operations $C _ { m } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right)$ to obtain the edited parameters $\theta ^ { \prime }$ . Therefore, starting from Objective (2), we can rewrite our goal as finding

$$
\phi ^ { \star } = \arg \operatorname* { m a x } _ { \phi } \ \sum _ { u \in \mathcal { U } } \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big ) ,\tag{3}
$$

where $\begin{array} { r } { \theta ^ { \prime } = C _ { m } ( \theta ) , m = \mathcal { F } _ { \phi } ( \mathbf { q } _ { u } ) } \end{array}$

## 3.2. Variational Motivation for Model Control

Equation (3) defines the target optimization goal for requestaware sequential recommendation. Exact marginalization over all control signals is intractable; therefore, we adopt a variational perspective (Blei et al., 2017) primarily as a conceptual guide for designing a sparse, request-conditioned controller, rather than as a strict inference procedure that CRAMER aims to optimize exactly. In particular, our practical algorithm does not perform full variational inference over control signals, but instead leverages this perspective to motivate the use of structured gating and sparsity, inducing regularization conditioned on natural-language requests.

Variational Lower Bound. Because the control signals in m are actually designed to be binary (in the form of gates, details in later sections), we adopt a factorized Bernoulli prior $p ( m )$ and approximate the posterior with a meanfield Bernoulli distribution $Q _ { \phi } ( \pmb { m } | \mathbf { q } _ { u } )$ parameterized by ϕ (Equation (A.4)); see Appendix A for details. This Bernoulli parameterization is natural because the control signal is a binary row–column gate vector. We use a factorized Bernoulli prior because it matches the hard sparsity budget, admits a closed-form KL term, and keeps the regularization cost linear in the gate dimension. We use a mean-field Bernoulli posterior for the same tractability reason: the request embedding is mapped to gate logits, which define lightweight request-conditioned gate probabilities. Although the KL term assumes independent Bernoulli gates, the practical controller is not fully independent: all gate logits are generated jointly from the same request representation, and the final hard mask is further coupled by the exact k-hot Gumbel–Top-k selection. Consider a single user $u \in \mathcal { U }$ in Equation (3), for such $Q _ { \phi } ( \pmb { m } | \mathbf { q } _ { u } )$ and $p ( m )$ , the marginal likelihood admits the variational lower bound:

$$
\begin{array} { l } { \log p \big ( i _ { T + 1 } ^ { \star } \mid \pmb { s } _ { u } , \theta ^ { \prime } \big ) \ \geq \ } \\ { \displaystyle \int Q _ { \phi } ( \pmb { m } \mid \mathbf { q } _ { u } ) \log p \big ( i _ { T + 1 } ^ { \star } \mid \pmb { s } _ { u } , \theta ^ { \prime } \big ) \ \mathrm { d } \pmb { m } } \\ { \displaystyle - \operatorname { K L } [ Q _ { \phi } ( \pmb { m } \mid \mathbf { q } _ { u } ) \| p ( \pmb { m } ) ] , } \end{array}\tag{4}
$$

with the evidence lower bound (ELBO)

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E L B O } } ( u ) = \mathbb { E } _ { m \sim Q _ { \phi } } \Big [ \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big ) \Big ] } \\ { - \mathrm { K L } \Big ( Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) \mid \mid p ( m ) \Big ) . } \end{array}\tag{5}
$$

For the detailed derivation of Equation (5), please refer to Appendix A. In practice, we approximate it using Gumbel–Top-k sampling with a straight-through estimator (see Sections 3.3 and 3.4), which is a more straightforward approach and better suited for recommender systems.

Training Objective. Motivated by Equation (5), we construct a tractable surrogate training objective with a KL term that admits a closed-form expression (Equation (A.5)) under the factorized Bernoulli assumption. We denote by $\ell ( \hat { i } _ { T + 1 } , i _ { T + 1 } ^ { \star } )$ the predictive loss, i.e., the original training loss used by the backbone recommender. Our training objective ${ \mathcal { L } } ( \phi )$ is designed as

$$
\frac { 1 } { | \mathcal { U } | } \sum _ { u \in \mathcal { U } } \left[ \underbrace { \ell ( \hat { i } _ { T + 1 } ^ { ( u ) } , \dot { i } _ { T + 1 } ^ { \star ( u ) } ) } _ { \mathrm { p r e d i c t i v e ~ l o s s } } + \lambda _ { \mathrm { K L } } \cdot \underbrace { \frac { 1 } { d } \mathrm { K L } [ Q \parallel p ] } _ { \mathrm { K L \ r e g u l a r i z e r } } \right] .\tag{6}
$$

where d is the dimension of $^ { m . }$ The Objective (6) consists of two complementary terms. The first is the predictive loss, which directly drives the model to rank the ground-truth item highest given the historical sequence and request, ensuring recommendation accuracy. The second is the KL regularizer, which encourages the posterior distribution $Q _ { \phi }$ to stay close to the sparsity prior $p ( m )$ , thereby enforcing compact and stable control. Note that we divide the KL term by the dimension d of m to normalize its scale, preventing it from dominating as d increases and ensuring a balanced trade-off between predictive accuracy and sparsity control. We emphasize that Objective (6) is a variationally inspired surrogate rather than a strict ELBO. The variational view motivates sparse request-conditioned control and the Bernoulli KL regularizer, while the practical forward controller uses an exact hard k-hot mask sampled by Gumbel–Top-k with a straight-through estimator. Therefore, the KL term is computed on the Bernoulli relaxation of the gates and serves as an auxiliary sparsity-inducing regularizer aligned with the hard budget.

Based on Objective (6), we propose Control via Request-Aware Masking for Editing Recommenders (CRAMER). At a high level, CRAMER adapts a frozen sequential recommender to natural-language requests by learning lightweight binary gate vectors that map each request to masks over a pretrained backbone. The edited model fuses long-term preferences with immediate intent, enabling rapid, no-retraining adaptation while preserving fine-grained control. Figure 1 overviews the framework and the following sections detail its components.

## 3.3. Request-to-Mask Adaptation

As discussed in Section 2, masking parameters of a deep model provides a lightweight but expressive mechanism for model control. In CRAMER, masking should be viewed as a budgeted behavior-editing mechanism rather than merely as feature suppression. A sparse request-conditioned mask selectively activates or deactivates existing computation paths in the frozen sequential recommender backbone, thereby steering pre-learned preference representations toward the current request without relearning dense query-aware representations from scratch. Hard row–column masking is particularly aligned with our setting because it provides an exact k-hot sparsity budget and keeps the training-time controller consistent with the low-overhead inference mechanism used at deployment. In this section, we introduce how CRAMER converts natural-language requests into masks that are used to control the backbone.

Request Embedding. We begin by describing how CRAMER encodes a natural-language request into an embedding that conditions the recommendation model (frozen sequential recommender backbone) $f _ { \theta }$ . Unlike historical interaction sequences, requests are diverse and may contain negations, constraints, or attribute-specific preferences. To extract their semantics, we introduce a lightweight request encoder $\mathrm { E } _ { \phi _ { \mathrm { e n c } } }$ based on a pretrained language model (PLM). Given a request $\mathbf { q } _ { u } ,$ , we tokenize it and obtain contextualized token embeddings, which are mean-pooled across all tokens to form a stable representation (Mosbach et al., 2020). We adopt mean pooling because it is simple, stable, and architecture-agnostic for variable-length requests. Formally,

$$
\begin{array} { r } { \pmb { e } _ { q } = \operatorname { E } _ { \phi _ { \mathrm { e n c } } } ( \mathbf { q } _ { u } ) \in \mathbb { R } ^ { h } , } \end{array}
$$

where $h$ is the hidden dimension. The resulting semantic embedding $e _ { q }$ summarizes the user’s immediate intent, allowing CRAMER to capture request semantics in a modular form while remaining compatible with the frozen sequential recommender backbone.

Defining the Controllable Subset. For a Transformerbased sequential recommender system $f _ { \theta } ,$ , we identify a subset of its parameters as controllable subset $\theta _ { M }$ that is crucial and suitable for being masked. In Transformer architectures, FFNs constitute the majority of parameters and act as key–value memories (Geva et al., 2020; Gerber, 2025); selectively masking them directly modulates what the model “remembers.” Moreover, attention heads often exhibit redundancy (Michel et al., 2019), and the multi-head attention (MHA) output projection matrices $W _ { O }$ aggregate head outputs into the residual stream (Hu et al., 2022), so masking $W _ { O }$ provides a compact, high-leverage control knob. Guided by these observations, CRAMER supports three scopes for $\theta _ { M } \mathrm { : }$ : (i) FFNs-only, (ii) $W _ { O } – \mathrm { o n l y }$ , and (iii) FFNs $+ W _ { O }$ . This choice is not meant to rule out other Transformer components, but to select natural intervention points that transform intermediate features and recombine attentionhead outputs while still supporting structured row–column masking with low overhead. Formally, we write the maskable set of L matrices as

$$
\begin{array} { r } { \theta _ { M } = \left\{ \boldsymbol { W } ^ { ( l ) } \in \mathbb { R } ^ { \alpha _ { l } \times \beta _ { l } } \right\} _ { l = 1 } ^ { L } . } \end{array}
$$

Projection to Gate Logits. Instead of assigning a mask to every parameter in $\theta _ { M } -$ which would incur prohibitive overhead given the scale of Transformer backbones— our scheme performs gating at the row and column levels (Svirsky et al., 2024). This structured design drastically reduces the number of trainable parameters while providing fine-grained, lightweight control over the backbone. Given the semantic embedding $e _ { q } .$ , we first map it to gate logits via a linear projection layer:

![](images/8f0b4bbd0fec17d975c8e46e6be313da9194c3306e07ce83aa49dcf5d6b83803.jpg)  
Figure 1. The overview of the proposed CRAMER framework. The figure shows the whole process of CRAMER converting a natural language request into masks and controlling the sequential recommender (frozen sequential recommender backbone). The gray area with a pink dashed border represents the “Row–Column Gating Masks” paragraph in Section 3.3.

$$
z = W _ { \mathrm { p r o j } } e _ { q } + b _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d } ,
$$

where $\begin{array} { r } { d \mathrm { ~ = ~ } \sum _ { l = 1 } ^ { L } \left( \alpha _ { l } + \beta _ { l } \right) } \end{array}$ is the total number of row and column dimensions under the chosen scope, and $( W _ { \mathrm { p r o j } } , b _ { \mathrm { p r o j } } )$ are trainable parameters.

Constructing Sparse Binary Vector. To achieve lightweight yet effective control, we constrain the binary vector to be k-hot. Let $\rho \in \left( 0 , 1 \right)$ be the drop ratio and retain exactly $k = \lceil ( 1 - \rho ) d \rceil$ active entries. To obtain them, we employ the Gumbel–Top-k trick (Kool et al., 2019): for each coordinate i, we sample $g _ { i } \sim$ Gumbel(0) and form

$$
\tilde { z } _ { i } = z _ { i } + g _ { i } , \qquad i = 1 , \ldots , d .
$$

The indices of the k largest $\tilde { z } _ { i }$ form $S _ { k } .$ , and the activated entries are

$$
m _ { i } \ = \ \mathbb { I } \{ i \in S _ { k } \} , m \in \{ 0 , 1 \} ^ { d } .\tag{7}
$$

This binary vector m is precisely the instantiation of the control signal vector mentioned in Equation (3), and the first four paragraphs of this section together constitute a concrete realization of the mapping $\mathcal { F } _ { \phi }$ described in Section 3.1.

Row–Column Gating Masks. In our CRAMER framework, m acts as a row-column gate vector that compactly specifies the activations of all maskable matrices in $\theta _ { M }$ . We decompose m into per-matrix segments to obtain, for each l, a row gate vector $\dot { \pmb { r } ^ { ( l ) } } \in \{ 0 , 1 \} ^ { \alpha _ { l } }$ and a column gate vector $\pmb { c } ^ { ( l ) } \in \overline { { \{ 0 , 1 \} ^ { \beta _ { l } } } }$ , and define the entrywise mask

$$
M _ { i j } ^ { ( l ) } \ : = \ : r _ { i } ^ { ( l ) } \cdot c _ { j } ^ { ( l ) } , \quad 1 \le i \le \alpha _ { l } , \ : 1 \le j \le \beta _ { l } .
$$

Collecting all $M ^ { ( l ) }$ and applying them entrywise to the corresponding ${ \cal W } ^ { ( l ) } \in \theta _ { M }$ yields the edited backbone $f _ { \theta ^ { \prime } }$ with parameters

$$
\theta ^ { \prime } = \big ( \theta _ { / M } , \{ { \pmb W } ^ { ( l ) } \odot { \pmb M } ^ { ( l ) } : { \pmb W } ^ { ( l ) } \in \theta _ { M } \} \big ) .\tag{8}
$$

where $\theta _ { / M }$ represents the parameters in θ except $\theta _ { M }$ . The operations in this paragraph are the specific form of $C _ { m } \mathopen { } \mathclose \bgroup \left( \theta \aftergroup \egroup \right)$ mentioned in Equation (3). Thus, the semantic embedding $e _ { q }$ is converted into a structured set of row-column gating masks that modulate the frozen sequential recommender backbone with minimal overhead while retaining fine-grained control.

## 3.4. Learnable Components and Discrete Optimization

Trainable Parameters. Since the backbone $f _ { \theta }$ is frozen, training updates only the request-to-mask (Section 3.3) module. Two components are learnable: (i) the projection layer $( W _ { \mathrm { p r o j } } , b _ { \mathrm { p r o j } } )$ that maps the semantic embedding $e _ { q }$ to gate logits z, and (ii) a subset $\phi _ { t }$ of the request encoder parameters $\phi _ { \mathrm { e n c } }$ (initialized from a PLM). We consider three regimes for $\phi _ { t } \subseteq \phi _ { \mathrm { e n c } } .$ none (encoder fully frozen), last (fine-tune the last layer only), and all (end-to-end tuning). This offers a flexible way to balance adaptation capacity and efficiency across backbones and datasets.

Straight-Through Training for Gating. As we obtain m by discretely sampling z (see Equation (7)), this process blocks gradients from m to logits z. To address this issue, we adopt a straight-through estimator (STE) (Bengio et al., 2013; Jang et al., 2016) with a temperature-controlled soft surrogate in the backward pass. Concretely, the forward pass uses hard k-hot gates from Gumbel–Top-k, while the backward pass propagates gradients through

$$
v _ { i } ~ = ~ \frac { \exp ( z _ { i } / \tau ) } { \sum _ { j = 1 } ^ { d } \exp ( z _ { j } / \tau ) } , \qquad i = 1 , \ldots , d , \tau > 0 ,
$$

serving as a continuous relaxation of the binary $m _ { i }$ . This STE scheme enables end-to-end optimization of the trainable parameters $( W _ { \mathrm { p r o j } } , b _ { \mathrm { p r o j } } , \phi _ { t } )$ despite the discrete sampling of m.

## 4. Experiments and Evaluation

The implementation of CRAMER is currently available at https://github.com/zhiyuansu0326/ CRAMER-ICML2026.

## 4.1. Experimental Setup

Datasets. We consider four representative datasets: (i) Re-Dial (Li et al., 2018), a conversational recommendation dataset with about 11.3K movie-recommendation dialogues, where users explicitly mention movies and annotate whether they have seen or liked them; (ii) KuaiSAR (Sun et al., 2023), a large-scale short-video interaction dataset from Kuaishou that captures both search and recommendation behaviors, containing about 19.6M actions and 6.9M items; (iii) Beauty, a subset of the Amazon Reviews (Hou et al., 2024) data, with approximately 701.5K reviews and 112.6K items, enriched in metadata, review text and timestamps; (iv) CDs&Vinyl, another subset from Amazon Reviews (Hou et al., 2024), containing about 4.8M reviews and 701.7K items, also possessing metadata, review text and timestamps. We preprocess the four datasets in a unified manner. Limited by the computing budget, we downsample the three larger datasets (KuaiSAR, Beauty, CDs&Vinyl). For more information on data preprocessing and statistics, see the Appendix B.1.

Backbones. We adopt two widely used Transformer-based sequential recommenders as frozen sequential recommender backbones. (i) SASRec (Kang & McAuley, 2018) employs unidirectional self-attention to model sequential dependencies in user interaction histories with high efficiency, and has become a standard baseline in sequential recommendation. (ii) BERT4Rec (Sun et al., 2019) adopts a bidirectional Transformer trained with a masked item prediction objective, enabling the model to capture both left and right contexts of a target position and to produce context-rich sequence representations. These two models represent the most established architectures for sequential recommendation and provide strong non–request-aware references for our study.

Baselines. On top of the frozen sequential recommender backbones, we further compare CRAMER with several stateof-the-art request-aware methods that incorporate naturallanguage requests. (i) Query-SeqRec (He et al., 2022) introduces a request encoder to represent the query and injects it into the backbone’s sequential representation through concatenation or attention, enabling request-conditioned relevance scoring. (ii) BLaIR (Hou et al., 2024) encodes both request text and item metadata into a shared semantic space to compute similarity signals, which are then fused with the backbone’s outputs to enhance ranking with requestaware semantics. (iii) LLM-ESR (Liu et al., 2024) leverages cached LLM-derived semantic embeddings and combines them with collaborative backbone embeddings via a lightweight adapter, providing notable benefits for longtail users and items while keeping the backbone frozen. (iv) REARANK (Zhang et al., 2025) first generates an initial ranking using the backbone and then applies an LLM reranker that reasons over user history, the request, and candidate metadata to refine the list, combining sequential modeling with listwise reasoning. The integration details of these baselines with the frozen sequential recommender backbones are given in Appendix B.9.

Optional PLMs. As mentioned in Section 3.3, the request encoder $\mathrm { E } _ { \phi _ { \mathrm { e n c } } }$ is initialized from a PLM based on Transformer. We consider four PLMs to cover the efficiency–accuracy spectrum: (i) BERT style (Devlin et al., 2019), a classic encoder. (ii) RoBERTa style (Liu et al., 2019), a robust medium encoder. (iii) MiniLM style (Wang et al., 2020), a very lightweight encoder. (iv) ModernBERT style (Warner et al., 2024), a modern high-capacity base encoder. All encoders use default text preprocessing, and we apply mean-pooling over the final hidden states (Section 3.3) to produce the semantic embedding $\textstyle e _ { q } ,$ , which we find more stable than single-token pooling when handling diverse request phrasing (Mosbach et al., 2020).

Evaluation Metrics. We adopt widely used ranking metrics in recommender system evaluation. Specifically, we report HR@k (Hit Ratio), NDCG@k (Normalized Discounted Cumulative Gain) and MRR (Mean Reciprocal Rank) at cutoff values $k \in \{ 1 0 , 2 0 \}$ . These are very commonly used metrics in recommender system evaluation.

## 4.2. Overall Performance

Following previous papers (Kang & McAuley, 2018; Liu et al., 2024), we randomly sample 100 items that the user has not interacted with as the negatives paired with the only ground-truth positive for calculation of the metrics. Table 1 reports the overall performance of four baselines and our proposed CRAMER on four benchmark datasets under two frozen Transformer backbones.

Aggregate Comparison. From the results, CRAMER consistently outperforms all baselines across datasets and metrics under both SASRec and BERT4Rec backbones. Intuitively, we observe that CRAMER achieves the best results on all metrics across all experiments. After applying Benjamini–Hochberg (BH) procedure (FDR = 0.05) across all 48 paired t-tests, CRAMER remains significantly better than the strongest baseline in 41 settings (85.42%), indicating consistent and robust improvements. This demonstrates that the request-aware masking mechanism effectively augments sequential recommenders, delivering more accurate predictions without requiring full fine-tuning.

Table 1. Overall results of four baselines and our CRAMER. H@k, N@k, and M@k denote HR@k, NDCG@k, and MRR@k, respectively (averaged over five runs). For each setting, the boldface refers to the highest result, and the underline indicates the second best result. “\*” marks statistically significant improvements after BH procedure (FDR = 0.05) across all 48 t-tests.
<table><tr><td rowspan="2">Method</td><td colspan="6">ReDial</td><td colspan="6">KuaiSAR</td></tr><tr><td>H@10</td><td>H@20</td><td>N@10</td><td>N@20</td><td>M@10</td><td>M@20</td><td>H@10</td><td>H@20</td><td>N@10</td><td>N@20</td><td>M@10</td><td>M@20</td></tr><tr><td>SASRec</td><td>0.426</td><td>0.573</td><td>0.373</td><td>0.410</td><td>0.344</td><td>0.354</td><td>0.430</td><td>0.601</td><td>0.346</td><td>0.389</td><td>0.306</td><td>0.318</td></tr><tr><td>+Query-SeqRec</td><td>0.450</td><td>0.596</td><td>0.391</td><td>0.429</td><td>0.350</td><td>0.361</td><td>0.451</td><td>0.567</td><td>0.348</td><td>0.378</td><td>0.313</td><td>0.322</td></tr><tr><td>+BLaIR</td><td>0.447</td><td>0.582</td><td>0.392</td><td>0.426</td><td>0.287</td><td>0.296</td><td>0.479</td><td>0.612</td><td>0.408</td><td>0.443</td><td>0.293</td><td>0.303</td></tr><tr><td>+LLM-ESR</td><td>0.516</td><td>0.666</td><td>0.385</td><td>0.423</td><td>0.323</td><td>0.334</td><td>0.496</td><td>0.628</td><td>0.392</td><td>0.427</td><td>0.343</td><td>0.351</td></tr><tr><td>+REARANK</td><td>0.549</td><td>0.684</td><td>0.414</td><td>0.449</td><td>0.408</td><td>0.417</td><td>0.538</td><td>0.645</td><td>0.409</td><td>0.437</td><td>0.366</td><td>0.374</td></tr><tr><td>+CRAMER (Ours)</td><td>0.578*</td><td>0.694*</td><td>0.428*</td><td>0.456*</td><td>0.413</td><td>0.421</td><td>0.556*</td><td>0.748*</td><td>0.436*</td><td>0.484*</td><td>0.391*</td><td>0.405*</td></tr><tr><td>BERT4Rec</td><td>0.421</td><td>0.542</td><td>0.355</td><td>0.387</td><td>0.272</td><td>0.281</td><td>0.436</td><td>0.591</td><td>0.366</td><td>0.407</td><td>0.311</td><td>0.322</td></tr><tr><td>+Query-SeqRec</td><td>0.462</td><td>0.563</td><td>0.347</td><td>0.373</td><td>0.307</td><td>0.315</td><td>0.464</td><td>0.577</td><td>0.364</td><td>0.393</td><td>0.349</td><td>0.358</td></tr><tr><td>+BLaIR</td><td>0.466</td><td>0.654</td><td>0.395</td><td>0.442</td><td>0.333</td><td>0.348</td><td>0.480</td><td>0.652</td><td>0.401</td><td>0.445</td><td>0.339</td><td>0.352</td></tr><tr><td>+LLM-ESR</td><td>0.515</td><td>0.668</td><td>0.427</td><td>0.465</td><td>0.358</td><td>0.369</td><td>0.530</td><td>0.715</td><td>0.389</td><td>0.436</td><td>0.324</td><td>0.338</td></tr><tr><td>+REARANK</td><td>0.536</td><td>0.680</td><td>0.388</td><td>0.424</td><td>0.355</td><td>0.366</td><td>0.566</td><td>0.691</td><td>0.416</td><td>0.448</td><td>0.355</td><td>0.364</td></tr><tr><td>+CRAMER (Ours)</td><td>0.580*</td><td>0.753*</td><td>0.451*</td><td>0.497*</td><td>0.376*</td><td>0.389*</td><td>0.598*</td><td>0.717</td><td>0.434*</td><td>0.467*</td><td>0.382*</td><td>0.390*</td></tr><tr><td rowspan="2">Method</td><td colspan="6">Beauty</td><td colspan="6">CDs&amp;Vinyl</td></tr><tr><td>H@10</td><td>H@20</td><td>N@10</td><td>N@20</td><td>M@10</td><td>M@20</td><td>H@10</td><td>H@20</td><td>N@10</td><td>N@20</td><td>M@10</td><td>M@20</td></tr><tr><td>SASRec</td><td>0.442</td><td>0.574</td><td>0.385</td><td>0.419</td><td>0.338</td><td>0.348</td><td>0.480</td><td>0.658</td><td>0.398</td><td>0.444</td><td>0.360</td><td>0.374</td></tr><tr><td>+Query-SeqRec</td><td>0.479</td><td>0.606</td><td>0.352</td><td>0.384</td><td>0.302</td><td>0.313</td><td>0.511</td><td>0.698</td><td>0.406</td><td>0.454</td><td>0.355</td><td>0.370</td></tr><tr><td>+BLaIR</td><td>0.495</td><td>0.622</td><td>0.421</td><td>0.453</td><td>0.348</td><td>0.357</td><td>0.525</td><td>0.627</td><td>0.450</td><td>0.477</td><td>0.349</td><td>0.357</td></tr><tr><td>+LLM-ESR</td><td>0.503</td><td>0.701</td><td>0.445</td><td>0.495</td><td>0.379</td><td>0.395</td><td>0.560</td><td>0.699</td><td>0.434</td><td>0.470</td><td>0.346</td><td>0.355</td></tr><tr><td>+REARANK</td><td>0.548</td><td>0.681</td><td>0.474</td><td>0.509</td><td>0.335</td><td>0.344</td><td>0.612</td><td>0.719</td><td>0.454</td><td>0.481</td><td>0.378</td><td>0.385</td></tr><tr><td>+CRAMER (Ours)</td><td>0.574*</td><td>0.735*</td><td>0.489*</td><td>0.531*</td><td>0.385</td><td>0.397</td><td>0.619</td><td>0.726*</td><td>0.472*</td><td>0.498*</td><td>0.397*</td><td>0.404*</td></tr><tr><td>BERT4Rec</td><td>0.409</td><td>0.551</td><td>0.331</td><td>0.368</td><td>0.323</td><td>0.334</td><td>0.416</td><td>0.547</td><td>0.300</td><td>0.334</td><td>0.291</td><td>0.301</td></tr><tr><td>+Query-SeqRec</td><td>0.434</td><td>0.534</td><td>0.357</td><td>0.382</td><td>0.334</td><td>0.341</td><td>0.459</td><td>0.614</td><td>0.330</td><td>0.371</td><td>0.283</td><td>0.292</td></tr><tr><td>+BLaIR</td><td>0.459</td><td>0.627</td><td>0.364</td><td>0.407</td><td>0.315</td><td>0.327</td><td>0.462</td><td>0.617</td><td>0.412</td><td>0.452</td><td>0.333</td><td>0.345</td></tr><tr><td>+LLM-ESR</td><td>0.493</td><td>0.594</td><td>0.346</td><td>0.373</td><td>0.319</td><td>0.327</td><td>0.503</td><td>0.692</td><td>0.438</td><td>0.487</td><td>0.311</td><td>0.325</td></tr><tr><td>+REARANK</td><td>0.509</td><td>0.693</td><td>0.379</td><td>0.426</td><td>0.331</td><td>0.346</td><td>0.571</td><td>0.684</td><td>0.423</td><td>0.452</td><td>0.367</td><td>0.376</td></tr><tr><td>+CRAMER (Ours)</td><td>0.539*</td><td>0.734*</td><td>0.396*</td><td>0.447*</td><td>0.345*</td><td>0.359*</td><td>0.583*</td><td>0.695</td><td>0.487*</td><td>0.516*</td><td>0.433*</td><td>0.441*</td></tr></table>

Results by Dataset and Backbone. Across datasets, CRAMER shows clear advantages, particularly on largescale datasets like KuaiSAR and CDs&Vinyl, where it substantially outperforms embedding-based and generative baselines, showing its ability to integrate long-term preferences with immediate requests. On smaller datasets (e.g., ReDial), it still yields consistent gains, underscoring robustness. Across backbones, CRAMER delivers stable improvements on both SASRec and BERT4Rec by incorporating request semantics, thereby addressing the limitation that both models rely solely on users’ historical interactions for prediction. Overall, CRAMER is a general and effective framework for request-aware sequential recommendation across datasets and architectures.

Intuitive User Case Study. To provide a more intuitive understanding of how our approach improves recommendation quality, we further conduct a case study on five specific users from the CDs&Vinyl dataset. For more details, please refer to Appendix B.8.

Summary. In summary, CRAMER consistently outperforms embedding-based, generative, and reasoning-driven request-aware baselines. It achieves significant improvements across nearly all datasets, metrics, and backbones. The results in Table 1 establish CRAMER as a general, flexible and robust framework that effectively integrates longterm preferences with immediate intent while remaining efficient under frozen sequential backbones.

## 4.3. Sensitivity Analysis of Hyperparameters

While the overall results in Section 4.2 demonstrate the effectiveness of CRAMER, it is important to understand how different hyperparameters influence performance. We therefore conduct several sensitivity studies to disentangle the contribution of each design choice. In each backbone × dataset experiment, we vary only the hyperparameter of interest while keeping all other settings fixed at their optimal values (listed in Appendix B.3). Evaluation is reported in terms of NDCG@10. Among the full set of hyperparameters we examined, two are particularly crucial: (i) drop ratio $\rho ,$ which determines how many units remain active under the request-to-mask mechanism; and (ii) selection of PLM used to initialize the request encoder. In this section we focus on these two factors, while additional experiments (e.g., experiments on $\theta _ { M } , \lambda _ { \mathrm { K L } } , \phi _ { t } )$ are deferred to Appendix B.4.

![](images/c907c31a96fc5c31b7c004b898744ad681dcb9145fa0fabe7f6e979891de7d44.jpg)  
Figure 2. Sensitivity of CRAMER to drop ratio $\rho ,$ evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

![](images/2e6f06dbc14f6cb900b20ccb8016690a6e7c1719af93fedfbdc240a6c201ceec.jpg)  
Figure 3. Impact of different PLMs on request encoding, evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

Sensitivity to Drop Ratio $\rho _ { \bullet }$ Figure 2 reports results for $\rho ~ \in ~ \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 \}$ We find that performance is generally robust within a moderate range but deteriorates at extreme values. In most cases, a $\rho$ of around 0.10 achieves the best balance: too small ${ \mathrm { ~ a ~ } } \rho { \mathrm { ~ } } ( { \mathrm { i . e . } }$ , almost no parameters are masked) weakens the influence of request conditioning, while too large ${ \mathrm { ~ a ~ } } \rho \left( { \mathrm { i . e . } } \right.$ ., masking too many parameters) harms the backbone’s capacity. We also find that small-scale datasets tend to achieve their best performance at larger values of $\rho ,$ while large-scale, information-dense datasets show optimal performance at smaller $\rho .$ From a theoretical analysis, we believe this is because the risk of overfitting is greater on small-scale datasets; higher sparsity imposes stronger regularization, helping to avoid overfitting and making the model focus only on the most salient request signals. On large-scale datasets, there is enough data to support more active parameters, so denser masks allow more backbone capacity to be utilized, improving expressiveness (Hoefler et al., 2021).

Selection of PLM. Figure 3 compares MiniLM, BERT, RoBERTa, and ModernBERT as optional PLMs, which are used to initialize the request encoders. Overall, RoBERTa and ModernBERT consistently yield the best performance:

Table 2. Inference efficiency comparison for SASRec backbone. Runtime and GPU memory usage are measured as average perrequest cost under identical settings. “Vanilla Backbone” reports the backbone-only cost, and the remaining rows show the additional overhead introduced by each request-aware method.
<table><tr><td>Method</td><td>Runtime (s)</td><td>GPU Memory (MiB)</td></tr><tr><td>SASRec</td><td>0.033</td><td>2024.1</td></tr><tr><td>+Query-SeqRec</td><td>+0.021</td><td>+1587.5</td></tr><tr><td>+BLaIR</td><td>+0.029</td><td>+1125.4</td></tr><tr><td>+LLM-ESR</td><td>+0.016</td><td>+1236.2</td></tr><tr><td>+REARANK</td><td>+9.256</td><td>+9824.7</td></tr><tr><td>+CRAMER (Ours)</td><td>+0.018</td><td>+1355.6</td></tr></table>

RoBERTa excels on small-scale or linguistically diverse datasets such as ReDial and Beauty, while ModernBERT dominates on large-scale or information-dense datasets such as CDs&Vinyl and KuaiSAR. In contrast, MiniLM, though computationally efficient, underperforms due to its limited capacity, and vanilla BERT trails behind its stronger successors in most cases. This demonstrates the ability of the CRAMER framework to fully leverage better and more robust language models, marking its excellence in capturing the request information.

## 4.4. Efficiency and Overhead

Beyond effectiveness, CRAMER also exhibits lightweight inference behavior. Once the request-to-mask module is trained, processing a new request requires only a single forward pass with a lightweight projection and masking operation, resulting in minimal per-request overhead and making the method well suited for real-time recommendation. Table 2 reports the average inference cost measured by wall-clock runtime and peak GPU memory usage under identical settings. Compared to vanilla SASRec, CRAMER introduces only a 0.018s runtime overhead, ranking among the most efficient request-aware methods. Its inference cost is comparable to LLM-ESR and lower than Query-SeqRec and BLaIR, while maintaining one of the smallest GPU memory footprints. In contrast, REARANK incurs substantial overhead (over two orders of magnitude higher runtime) due to its listwise reasoning across multiple candidates. Overall, CRAMER achieves a favorable trade-off between accuracy and efficiency: once trained, it consistently improves recommendation quality while keeping inference time and memory consumption low, enabling practical and scalable real-time deployment. Inference efficiency results for BERT4Rec are provided in Appendix B.5.

## 4.5. Mask Interpretability

To evaluate whether CRAMER’s request-conditioned masks reflect meaningful semantics, we conduct an interpretability study on the ReDial dataset. Using genre labels obtained via the Open Movie Database<sup>1</sup> (one of ReDial’s original data sources), we determine whether a recommended movie belongs to the romance-related category. For a sampled group of 100 users, we issue six types of requests around the “romance” concept: (1) clear positive, (2) ambiguous positive, (3) rare-term positive, (4) clear opposite, (5) ambiguous opposite, and (6) rare-term opposite, to influence their recommendation results. For each request, we measure the proportion of romance-related items in top-10 list and compute the mean and variance across users.

Table 3. The proportion of romance-related movies in the top-10 recommendations under different request types (100 users). The first row reports the baseline results without any request.
<table><tr><td>Type</td><td>Request Text</td><td>Avg</td><td>Var</td></tr><tr><td></td><td>No request</td><td>0.286</td><td>0.0218</td></tr><tr><td>(1)</td><td>&quot;I&#x27;d like a romantic comedy.&quot;</td><td>0.432↑</td><td>0.0244</td></tr><tr><td>(2)</td><td>&quot;Something sweet and heartwarming.&#x27; 3</td><td>0.345↑</td><td>0.0253</td></tr><tr><td>(3)</td><td>&quot;I want an offbeat, slow-burn emotional drama.&quot;</td><td>0.312↑</td><td>0.0371</td></tr><tr><td>(4)</td><td>&quot;Please avoid romantic movies.&quot;</td><td>0.135 ↓</td><td>0.0187</td></tr><tr><td>(5)</td><td>&quot;Maybe something less focused on love.&quot;</td><td>0.204↓</td><td>0.0198</td></tr><tr><td>(6)</td><td>&quot;Skip movies with amour-driven plots.&quot;</td><td>0.253 ↓</td><td>0.0389</td></tr></table>

Across all six request types, the results in Table 3 demonstrate that CRAMER produces consistent and semantically aligned shifts in the recommendation distribution. Clear positive requests substantially increase the proportion of romance-related movies in the top-10 list, while clear opposite requests consistently decrease it. This directional behavior provides evidence that CRAMER’s request-conditioned masks can induce semantic shifts aligned with the intended request. Ambiguous requests also induce smaller but still noticeable shifts in the expected direction, indicating that CRAMER can interpret indirect user intent. Moreover, rareterm requests can still lead to appropriate adjustments. Although we can observe an increase in variance, it remains within acceptable limits, suggesting robustness to infrequent phrasing. These findings provide evidence that CRAMER can induce interpretable and stable request-conditioned shifts over the backbone model.

## 5. Conclusion

In this paper, we introduced CRAMER, a lightweight framework for request-aware sequential recommendation that treats natural-language requests as control inputs. We first formalized the problem and proposed a variationally motivated training objective with KL regularization and STE, enabling stable and efficient optimization. CRAMER encodes each request into a semantic embedding, which is projected into structured row–column masks that modulate frozen Transformer backbones, providing fine-grained and efficient control without retraining. Through extensive experiments on four benchmark datasets and two backbones, we demonstrated that CRAMER consistently outperforms strong request-aware baselines across multiple metrics while incurring minimal runtime and memory overhead. Overall, CRAMER establishes a new paradigm for controllable, efficient, and scalable integration of immediate user intent into sequential recommendation.

## Impact Statement

This paper presents work whose goal is to advance the field of machine learning, with a focus on enabling efficient and flexible control of sequential recommender systems through natural-language requests. The proposed approach allows deployed recommender models to adapt their behavior to users’ immediate intents without retraining, which may improve user experience and interaction quality in real-world recommendation services. In addition, by avoiding costly backbone fine-tuning or large language model inference at deployment time, this work has the potential to reduce computational overhead and energy consumption in large-scale systems.

At the same time, increased controllability of recommender systems may raise practical concerns related to unintended or inappropriate steering of recommendations, such as conflicts between short-term requests and users’ long-term interests. In this work, the proposed method is designed to provide moderate, request-conditioned adjustments rather than complete overrides of learned preferences. We believe that responsible deployment of such techniques should be accompanied by appropriate system-level safeguards and human-centered design choices. Overall, the broader impacts of this work are consistent with those commonly encountered when advancing controllable and efficient machine learning systems.

## References

Ansell, A., Ponti, E., Korhonen, A., and Vulic, I. Com-´ posable sparse fine-tuning for cross-lingual transfer. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Proceedings ofthe 60th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pp. 1778–1796, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. acl-long.125. URL https://aclanthology.org/ 2022.acl-long.125/.

Bejani, M. M. and Ghatee, M. A systematic review on overfitting control in shallow and deep neural networks. Artificial Intelligence Review, 54(8):6391–6438, 2021.

Bengio, Y., Leonard, N., and Courville, A. Estimating or ´ propagating gradients through stochastic neurons for con-

ditional computation. arXiv preprint arXiv:1308.3432, 2013.

Blei, D. M., Kucukelbir, A., and McAuliffe, J. D. Variational inference: A review for statisticians. Journal of the American statistical Association, 112(518):859–877, 2017.

Chen, X., Li, Z., Pan, W., and Ming, Z. A survey on multibehavior sequential recommendation. arXiv preprint arXiv:2308.15701, 2023.

Devlin, J., Chang, M.-W., Lee, K., and Toutanova, K. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter ofthe associationfor computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

Fang, H., Zhang, D., Shu, Y., and Guo, G. Deep learning for sequential recommendation: Algorithms, influential factors, and evaluations. ACM Transactions on Information Systems (TOIS), 39(1):1–42, 2020.

Frankle, J. and Carbin, M. The lottery ticket hypothesis: Finding sparse, trainable neural networks, 2019. URL https://arxiv.org/abs/1803.03635.

Gao, C., Lei, W., He, X., De Rijke, M., and Chua, T.-S. Advances and challenges in conversational recommender systems: A survey. AI open, 2:100–126, 2021.

Gerber, I. Attention is not all you need: The importance of feedforward networks in transformer models. arXiv preprint arXiv:2505.06633, 2025.

Geva, M., Schuster, R., Berant, J., and Levy, O. Transformer feed-forward layers are key-value memories. arXiv preprint arXiv:2012.14913, 2020.

He, Z., Zhao, H., Wang, Z., Lin, Z., Kale, A., and Mcauley, J. Query-aware sequential recommendation. In Proceedings ofthe 31st ACM International Conference on Information & Knowledge Management, pp. 4019–4023, 2022.

Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., and Peste, A. Sparsity in deep learning: Pruning and growth for efficient inference and training in neural networks. Journal ofMachine Learning Research, 22(241):1–124, 2021.

Hou, Y., Li, J., He, Z., Yan, A., Chen, X., and McAuley, J. Bridging language and items for retrieval and recommendation, 2024. URL https://arxiv.org/abs/ 2403.03952.

Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., and

Gelly, S. Parameter-efficient transfer learning for nlp. In International conference on machine learning, pp. 2790– 2799. PMLR, 2019.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Jang, E., Gu, S., and Poole, B. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

Jannach, D., Manzoor, A., Cai, W., and Chen, L. A survey on conversational recommender systems. ACM Computing Surveys (CSUR), 54(5):1–36, 2021.

Jin, J., Chen, X., Zhang, W., Huang, J., Feng, Z., and Yu, Y. Learn over past, evolve for future: Search-based timeaware recommendation with sequential behavior data. In Proceedings ofthe ACM web conference 2022, pp. 2451– 2461, 2022.

Kang, W.-C. and McAuley, J. Self-attentive sequential recommendation. In 2018 IEEE international conference on data mining (ICDM), pp. 197–206. IEEE, 2018.

Kool, W., Van Hoof, H., and Welling, M. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In International conference on machine learning, pp. 3499–3508. PMLR, 2019.

Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. In Moens, M.-F., Huang, X., Specia, L., and Yih, S. W.-t. (eds.), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 3045–3059, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main. 243. URL https://aclanthology.org/2021. emnlp-main.243/.

Li, J., Wang, M., Li, J., Fu, J., Shen, X., Shang, J., and McAuley, J. Text is all you need: Learning language representations for sequential recommendation. In Proceedings ofthe 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 1258–1267, 2023.

Li, R., Kahou, S. E., Schulz, H., Michalski, V., Charlin, L., and Pal, C. Towards deep conversational recommendations. In Advances in Neural Information Processing Systems 31 (NIPS 2018), 2018.

Li, X., Thickstun, J., Gulrajani, I., Liang, P. S., and Hashimoto, T. B. Diffusion-lm improves controllable text generation. Advances in neural information processing systems, 35:4328–4343, 2022.

Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. In Zong, C., Xia, F., Li, W., and Navigli, R. (eds.), Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 4582–4597, Online, August 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.acl-long. 353. URL https://aclanthology.org/2021. acl-long.353/.

Li, X. L. and Rush, A. M. Posterior control of blackbox generation. arXiv preprint arXiv:2005.04560, 2020.

Liao, H., Zhang, J., Lian, J., Lu, W., Wu, M., Wang, S., Zhang, Y., Huang, Y., Zhou, M., and Mao, R. Eliminating out-of-domain recommendations in llm-based recommender systems: A unified view, 2026. URL https://arxiv.org/abs/2505.03336.

Litschko, R., Vulic, I., and Glava´ s, G. Parameter-efficientˇ neural reranking for cross-lingual and multilingual retrieval, 2022. URL https://arxiv.org/abs/ 2204.02292.

Liu, Q., Wu, X., Wang, Y., Zhang, Z., Tian, F., Zheng, Y., and Zhao, X. Llm-esr: Large language models enhancement for long-tailed sequential recommendation. Advances in Neural Information Processing Systems, 37: 26701–26727, 2024.

Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O., Lewis, M., Zettlemoyer, L., and Stoyanov, V. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692, 2019.

Louizos, C., Welling, M., and Kingma, D. P. Learning sparse neural networks through l 0 regularization. arXiv preprint arXiv:1712.01312, 2017.

Luo, K., Yang, H., Wu, G., and Sanner, S. Deep critiquing for vae-based recommender systems. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pp. 1269–1278, 2020.

Maddison, C. J., Mnih, A., and Teh, Y. W. The concrete distribution: A continuous relaxation of discrete random variables. arXiv preprint arXiv:1611.00712, 2016.

Michel, P., Levy, O., and Neubig, G. Are sixteen heads really better than one? Advances in neural information processing systems, 32, 2019.

Moradizeyveh, S. Intent recognition in conversational recommender systems. arXiv preprint arXiv:2212.03721, 2022.

Mosbach, M., Khokhlova, A., Hedderich, M. A., and Klakow, D. On the interplay between fine-tuning and sentence-level probing for linguistic knowledge in pretrained transformers. arXiv preprint arXiv:2010.02616, 2020.

Pan, L., Pan, W., Wei, M., Yin, H., and Ming, Z. A survey on sequential recommendation. arXiv preprint arXiv:2412.12770, 2024.

Prottasha, N. J., Mahmud, A., Sobuj, M. S. I., Bhat, P., Kowsher, M., Yousefi, N., and Garibay, O. O. Parameterefficient fine-tuning of large language models using semantic knowledge tuning. Scientific Reports, 14(1): 30667, 2024.

Qin, J., Liu, C., Cheng, S., Guo, Y., and Arcucci, R. Freeze the backbones: a parameter-efficient contrastive approach to robust medical vision-language pre-training. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1686–1690. IEEE, 2024.

Radlinski, F., Boutilier, C., Ramachandran, D., and Vendrov, I. Subjective attributes in conversational recommendation systems: challenges and opportunities. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pp. 12287–12293, 2022.

Shao, J., Dong, J., Wang, D., Shih, K., Li, D., and Zhou, C. Deep learning model acceleration and optimization strategies for real-time recommendation systems, 2025. URL https://arxiv.org/abs/2506.11421.

Shen, C., Zhao, J., Zhang, X., Yu, W., He, M., and Fan, J. Paragon: Parameter generation for controllable multitask recommendation. In Proceedings ofthe Nineteenth ACM Conference on Recommender Systems, pp. 370–380, 2025.

Shen, C., Zhang, X., Shi, T., Zhang, C., Xie, G., Xu, J., He, M., and Fan, J. A survey of controllable learning: Methods and applications in information retrieval. Frontiers of Computer Science, 20(10):2010619, 2026.

Son, H., Son, Y., Kim, C., and Kim, Y. G. Not all adapters matter: Selective adapter freezing for memory-efficient fine-tuning of language models. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 9479–9496, 2025.

Su, Z., Dai, S., and Zhang, X. Revisiting clustering of neural bandits: Selective reinitialization for mitigating loss of plasticity. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pp. 2690–2701, 2025.

Sun, F., Liu, J., Wu, J., Pei, C., Lin, X., Ou, W., and Jiang, P. Bert4rec: Sequential recommendation with bidirectional encoder representations from transformer. In Proceedings ofthe 28th ACM international conference on information and knowledge management, pp. 1441–1450, 2019.

Sun, Y., Chen, Y., Li, Y., and Ding, B. Enhancing latent computation in transformers with latent tokens, 2025. URL https://arxiv.org/abs/2505.12629.

Sun, Z., Si, Z., Zang, X., Leng, D., Niu, Y., Song, Y., Zhang, X., and Xu, J. Kuaisar: A unified search and recommendation dataset. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management, 2023. doi: 10.1145/ 3583780.3615123. URL https://doi.org/10. 1145/3583780.3615123.

Svirsky, J., Refael, Y., and Lindenbaum, O. Finegates: Llms finetuning with compression using stochastic gates, 2024. URL https://arxiv.org/abs/2412.12951.

Tao, C., Hou, L., Bai, H., Wei, J., Jiang, X., Liu, Q., Luo, P., and Wong, N. Structured pruning for efficient generative pre-trained language models. In Findings ofthe Association for Computational Linguistics: ACL 2023, pp. 10880–10895, 2023.

Wang, W., Wei, F., Dong, L., Bao, H., Yang, N., and Zhou, M. Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. Advances in neural information processing systems, 33: 5776–5788, 2020.

Warner, B., Chaffin, A., Clavie, B., Weller, O., Hallstr´ om,¨ O., Taghadouini, S., Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., et al. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. arXiv preprint arXiv:2412.13663, 2024.

Wen, W., Wu, C., Wang, Y., Chen, Y., and Li, H. Learning structured sparsity in deep neural networks. In Proceedings of the 30th International Conference on Neural Information Processing Systems, NIPS’16, pp. 2082–2090, Red Hook, NY, USA, 2016. Curran Associates Inc. ISBN 9781510838819.

Wu, G., Luo, K., Sanner, S., and Soh, H. Deep languagebased critiquing for recommender systems. In Proceedings of the 13th ACM Conference on Recommender Systems, pp. 137–145, 2019.

Xu, X., Xu, Z., Yu, P., and Wang, J. Enhancing user intent for recommendation systems via large language models. In International Conference on Artificial Intelligence and Machine Learning Research (CAIMLR 2024), volume 13635, pp. 46–54. SPIE, 2025.

Ye, Z., Yang, J., Meng, F., Li, M., and Zhan, Y. Integrating temporal interest dynamics and virality factors for high-precision ranking in big data recommendation. Electronics, 14(18):3687, 2025. ISSN 2079- 9292. doi: 10.3390/electronics14183687. URL https: //www.mdpi.com/2079-9292/14/18/3687.

Zhang, L., Wang, B., Qiu, X., Reddy, S., and Agrawal, A. Rearank: Reasoning re-ranking agent via reinforcement learning, 2025. URL https://arxiv.org/abs/ 2505.20046.

Zhao, M., Lin, T., Mi, F., Jaggi, M., and Schutze,¨ H. Masking as an efficient alternative to finetuning for pretrained language models. In Webber, B., Cohn, T., He, Y., and Liu, Y. (eds.), Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 2226–2241, Online, November 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.emnlp-main. 174. URL https://aclanthology.org/2020. emnlp-main.174/.

Zhao, W. X., Mu, S., Hou, Y., Lin, Z., Chen, Y., Pan, X., Li, K., Lu, Y., Wang, H., Tian, C., et al. Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms. In proceedings ofthe 30th acm international conference on information & knowledge management, pp. 4653–4664, 2021.

Zivic, P., Vazquez, H., and Sanchez, J. Scaling sequential´ recommendation models with transformers. In Proceed ings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 1567–1577, 2024.

## A. Derivation of the Training Objective

We start from the request-aware maximum-likelihood formulation in Objective (2):

$$
\operatorname* { m a x } \ \sum _ { u \in \mathcal { U } } \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big ) , \quad \mathrm { w h e r e ~ } \theta ^ { \prime } = C _ { m } ( \theta ) , \ m = \mathcal { F } _ { \phi } ( \mathbf { q } _ { u } ) .
$$

According to our definition in Section 3.1, $f _ { \theta }$ is a frozen sequential backbone with parameters $\theta ,$ and only the subset $\theta _ { M }$ is subject to request-conditioned masking. Let m $\in \{ 0 , 1 \} ^ { d }$ denote the row-column gate vector (Section 3.3) that selects row/column activations for all matrices in $\theta _ { M }$ . In Section 3.2 we adopt variational inference as a motivation for a tractable surrogate. Below we present the ELBO derivation, and then clarify how our practical training aligns with it under hard k-hot control. In the original Inequality (4), we use integral to characterize the abstract “control signals”, but $m \in \{ 0 , 1 \} ^ { d }$ is actually a binary vector, so in the derivation in this section, we use summation instead of integral.

Marginal Likelihood as a Sum over Masks. For a single user $u \in \mathcal { U } ,$ the conditional likelihood is marginalized over al gates in m:

$$
\begin{array} { r l } & { \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big ) = \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \mathbf { q } _ { u } , m ; \theta \big ) } \\ & { \qquad = \log \displaystyle \sum _ { m \in \{ 0 , 1 \} ^ { d } } p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \mathbf { q } _ { u } , m ; \theta \big ) \ p ( m ) . } \end{array}\tag{A.1}
$$

Prior Distribution. We use a request-agnostic sparsity prior. Aligned with Section 3.3, we consider a factorized Bernoulli prior with activation rate equal to the actual keep ratio:

$$
p ( { \pmb m } ) = \prod _ { i = 1 } ^ { d } \pi _ { 0 } ^ { m _ { i } } ( 1 - \pi _ { 0 } ) ^ { 1 - m _ { i } } , \qquad \pi _ { 0 } : = \frac { k } { d } ,\tag{A.2}
$$

so that the prior mean exactly matches the realized budget of $k = \lceil ( 1 - \rho ) d \rceil$ active gates (note that $\pi _ { 0 }$ may differ slightly from $1 - \rho$ due to the ceiling operation).

Variational Lower Bound. Introduce a request-conditioned variational distribution $Q _ { \phi } ( \pmb { m } \mid \mathbf { q } _ { u } )$ , parameterized by ϕ. Multiply and divide inside the sum by $Q _ { \phi }$ , and apply Jensen’s inequality:

$$
\begin{array} { r l } & { \log p \left( i _ { T + 1 } ^ { * } \mid s _ { u } , \theta ^ { \prime } \right) = \log \displaystyle \sum _ { m \in \{ 0 , 1 \} } Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) \frac { p \left( i _ { T + 1 } ^ { * } \mid s _ { u } , \mathbf { q } _ { u } , m ; \theta \right) p ( m ) } { Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) } } \\ & { \qquad = \log \mathbb { E } _ { m \sim Q _ { \phi } ( \cdot \mid \mathbf { q } _ { u } ) } \left[ \frac { p \left( i _ { T + 1 } ^ { * } \mid s _ { u } , \theta ^ { \prime } \right) p ( m ) } { Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) } \right] } \\ & { \qquad \geq \mathbb { E } _ { m \sim Q _ { \phi } ( \cdot \mid \mathbf { q } _ { u } ) } \left[ \log p \left( i _ { T + 1 } ^ { * } \mid s _ { u } , \theta ^ { \prime } \right) + \log p ( m ) - \log Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) \right] } \\ & { \qquad = \underbrace { \mathbb { E } _ { m \sim Q _ { \phi } } \left[ \log p \left( i _ { T + 1 } ^ { * } \mid s _ { u } , \theta ^ { \prime } \right) \right] } _ { \mathrm { p r e d i c t i o n ~ t e m } } - \underbrace { \mathbb { E } _ { m \sim Q _ { \phi } } \left[ \log \frac { Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) } { p ( m ) } \right] } _ { \mathrm { K I } / Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) } . } \end{array}
$$

The above is the complete derivation of Equation (5). Note that $p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \mathbf { q } _ { u } , m ; \theta \big ) = p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big )$ . Hence, the per-user evidence lower bound (ELBO) is

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { E L B O } } ( u ) = \mathbb { E } _ { m \sim Q _ { \phi } } \Big [ \log p \big ( i _ { T + 1 } ^ { \star } \mid s _ { u } , \theta ^ { \prime } \big ) \Big ] - \mathrm { K L } [ Q _ { \phi } ( m \mid \mathbf { q } _ { u } ) \mid | p ( m ) ] . } \end{array}\tag{A.3}
$$

which is consistent with Equation (5). In our actual algorithm the forward controller is hard k-hot (Gumbel–Top-k) with STE for gradients (Section 3.4); therefore we treat the ELBO as motivation and use a variationally inspired surrogate objective (detailed below) rather than maximizing Equation (A.3) strictly.

Parametrization and Analytic KL. The request-to-mask module (Section 3.3) produces logits $z = W _ { \mathrm { p r o j } } e _ { q } + b _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d }$ from the semantic embedding $e _ { q }$ . We adopt a mean-field Bernoulli parameterization for regularization and monitoring:

$$
Q _ { \phi } ( m \mid { \bf q } _ { u } ) = \prod _ { i = 1 } ^ { d } \pi _ { i } ^ { m _ { i } } ( 1 - \pi _ { i } ) ^ { 1 - m _ { i } } , \qquad \pi _ { i } = \sigma ( z _ { i } ) , \sigma ( x ) = \frac { 1 } { 1 + e ^ { - x } } .\tag{A.4}
$$

With the Bernoulli prior in Equation (A.2), the KL admits a closed form:

$$
\operatorname { K L } [ Q _ { \phi } \parallel p ] = \sum _ { i = 1 } ^ { d } \Big [ \pi _ { i } \log \frac { \pi _ { i } } { \pi _ { 0 } } + ( 1 - \pi _ { i } ) \log \frac { 1 - \pi _ { i } } { 1 - \pi _ { 0 } } \Big ] ,\tag{A.5}
$$

and we use the normalized version $\overline { { \mathrm { K L } } } ~ = ~ { \textstyle \frac { 1 } { d } } \mathrm { K L } [ Q _ { \phi } \| p ]$ so that the penalty does not scale with d.

At inference, m is instantiated as an exactly k-hot vector by (deterministic) Top-k on z (we drop Gumbel noise). At training time, the forward path also uses hard k-hot masks via Gumbel–Top-k, while the backward path propagates gradients through a softmax surrogate with temperature (STE; Section 3.4). In parallel, we compute the Bernoulli parameters $\pi _ { i } = \sigma ( z _ { i } )$ and apply the analytic Bernoulli–Bernoulli KL in Equation (A.5) as an auxiliary sparsity prior. We set the prior mean to the realized keep ratio, $\pi _ { 0 } = k / d ,$ aligning the prior with the hard budget used in the forward path. Because the forward sampler is k-hot while the KL assumes independent Bernoulli gates, the overall objective is not a strict $\operatorname { E L B O ; }$ it is a variationally inspired surrogate consistent with our hard-sparsity design.

From Likelihood to Supervised Loss. We train with a supervised next-item loss $\ell ( \cdot )$ in place of − log p(·) (consistent with Section 3.2). Let $\widehat { \theta ^ { \prime } } ^ { - } = \left( \theta _ { / M } , \{ W ^ { ( l ) } \odot M ^ { ( l ) } : W ^ { ( l ) } \stackrel { \cdot } { \in } \theta _ { M } \} \right)$ denote the edited backbone obtained by applying the row–column masks (Section 3.3). A per-user surrogate objective is

$$
\begin{array} { r } { \mathcal { L } ( u ; \phi ) = \underbrace { \ell \big ( f _ { \theta ^ { \prime } } ( s _ { u } , \mathbf { q } _ { u } ) , i _ { T + 1 } ^ { \star ( u ) } \big ) } _ { \mathrm { p r e d i c t i v e ~ l o s s ~ u n d e r ~ h a r d ~ } k \mathrm { - h o t ~ f o r w a r d } } + \lambda _ { \mathrm { K L } } \cdot \overline { { \mathrm { K L } } } ( z ; \pi _ { 0 } ) , } \end{array}\tag{A.6}
$$

where $\overline { { \mathrm { K L } } } ( z ; \pi _ { 0 } )$ is computed from $\pi = \sigma ( z )$ via Equation (A.5), and gradients through the discrete selection are enabled by STE with a temperature-controlled soft surrogate (Section 3.4). Equivalently, we can view Equation (A.6) as a single-sample Monte Carlo estimator of the predictive term (with the sample drawn by Gumbel–Top-k) plus an analytic KL regularizer computed on the logits.

Final Training Objective. Aggregating and averaging Equation (A.6) over $u \in \mathcal { U }$ yields the training objective reported in Section 3.2:

$$
\mathcal { L } ( \phi ) = \frac { 1 } { | \mathcal { U } | } \sum _ { u \in \mathcal { U } } \underbrace { \Bigg [ \ell \big ( \hat { i } _ { T + 1 } ^ { ( u ) } , i _ { T + 1 } ^ { \star ( u ) } \big ) } _ { \mathrm { p r e d i c t i v e ~ l o s s } } + \lambda _ { \mathrm { K L } } \cdot \underbrace { \overline { { \mathrm { K L } } } ( z ; \pi _ { 0 } ) } _ { \mathrm { W i r ~ r e g u l a r i z e r } } \Bigg ] .
$$

The first term provides supervision for request-aware prediction under the edited backbone $f _ { \theta ^ { \prime } }$ , while the second acts as an auxiliary sparsity prior that stabilizes optimization and prevents degenerate dense masks. In summary, the variational view motivates a tractable, analytically regularized surrogate that preserves strict k-sparsity in the forward path (via Gumbel–Top-k) and affords fine-grained, lightweight control over a frozen sequential recommender backbone.

Discussion on Surrogate Objective. As mentioned before, our training objective is variationally inspired rather than a strict ELBO, because the forward pass uses hard k-hot Gumbel–Top-k masking while the KL term assumes independent Bernoulli gates. This type of approximation is standard in sparse gating and masking methods, where discrete control variables are optimized using continuous relaxations or STE (Bengio et al., 2013; Maddison et al., 2016; Jang et al., 2016; Louizos et al., 2017), and the KL term in such frameworks primarily functions as a sparsity-inducing regularizer rather than an exact posterior-matching term. In our setting, the mask acts as a control signal rather than a latent probabilistic variable, and empirical behavior is far more critical than variational tightness. We observe stable optimization, smooth mask activations, and reliable request-conditioned effects across datasets and backbones. Thus, although our objective is not a strict ELBO, it follows well-established practices in sparse neural control and maintains the desired inductive bias while remaining computationally tractable.

## B. Details of Experiments

## B.1. Data Preprocessing

We preprocess all four datasets (ReDial<sup>2</sup>, KuaiSAR<sup>3</sup>, Beauty and CDs&Vinyl<sup>4</sup>) in a unified manner to construct RecBole-style atomic files. For each user, we sort interactions chronologically and build the request text by leveraging and concatenating information from the three prior interactions (title, content, category, search keyword, etc.) before the current timestamp; when no prior history exists, we optionally fall back to the current interaction. To obtain positive instances, we filter interactions according to dataset characteristics: for ReDial and KuaiSAR we keep only positive interactions, while for Beauty and CDs&Vinyl we retain ratings greater than or equal to 4.0. Limited by the computing budget, we perform random downsampling on KuaiSAR, Beauty and CDs&Vinyl to use only a portion of the data in these datasets. Finally, only items and interactions that pass these filters are retained to form the atomic .inter and .item files used in training and evaluation. Table 4 shows the statistics of the preprocessed datasets.

Table 4. Statistics of the processed datasets. “#Items” represents the total number of items, “#Inters” represents the total number o interactions (each one contains a request), and “Average Chars” represents the average number of characters of requests.
<table><tr><td>Dataset</td><td>#Items</td><td>#Inters</td><td>Average Chars</td></tr><tr><td>ReDial</td><td>5207</td><td>36460</td><td>112.84</td></tr><tr><td>KuaiSAR</td><td>174895</td><td>260243</td><td>124.63</td></tr><tr><td>Beauty</td><td>44977</td><td>122485</td><td>226.85</td></tr><tr><td>CDs&amp;Vinyl</td><td>76368</td><td>141213</td><td>973.21</td></tr></table>

## B.2. Training of Backbones

To control the variance, in our experiments, the parameters of both backbones (SASRec and BERT4Rec) under each dataset are trained using the default settings of the RecBole library. For specific parameter settings, please refer to RecBole v1.2.1 <sup>5</sup> (Zhao et al., 2021) .

## B.3. Optimal Settings

We summarize in Table 5 the optimal hyperparameter settings used in our experiments for each backbone × dataset configuration. All hyperparameters were tuned within predefined search ranges, and the best configuration was selected based on validation NDCG@10. Below we briefly describe each hyperparameter and its search space:

• $\rho \colon$ the drop ratio of m; searched over {0.05, 0.10, 0.15, 0.20, 0.25}.

• PLM: the pretrained language model used as request encoder, selected from {BERT, RoBERTa, MiniLM, Modern-BERT}.

• θ<sub>M</sub>: the part of the Transformer backbone subject to masking; chosen from {FFNs, W<sub>O</sub>, FFNs+W<sub>O</sub>}.

• ϕ : the fine-tuning regime for the PLM; one of {none (fully frozen), last (fine-tune the last layer), all (end-to-end tuning)}.

• λ : the weight of the KL regularization term; tuned in {0.1, 0.2, 0.3, 0.4, 0.5}.

• shared: whether gates are sampled once and shared across the entire batch (1) or sampled independently per instance (0) in the training phase.

• τ (0.7 → 0.3): the temperature annealing schedule for Gumbel–Top-k sampling; choose from {linear, exponential, cosine} with a uniform start value of 0.7 and an end value of 0.3.

Table 5. Optimal hyperparameter settings for each backbone × dataset configuration.
<table><tr><td>Backbone</td><td>Dataset</td><td>ρ</td><td>PLM</td><td> $\theta _ { M }$ </td><td> $\phi _ { t }$ </td><td> $\lambda _ { \mathrm { K L } }$ </td><td>shared</td><td> $\tau ( 0 . 7  0 . 3 )$ </td></tr><tr><td>SASRec</td><td>ReDial</td><td>0.20</td><td>RoBERTa</td><td> $W _ { O }$ </td><td>last</td><td>0.4</td><td>0</td><td>cosine</td></tr><tr><td></td><td>KuaiSAR</td><td>0.05</td><td>ModernBERT</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>all</td><td>0.1</td><td>0</td><td>cosine</td></tr><tr><td></td><td>Beauty</td><td>0.10</td><td>RoBERTa</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>last</td><td>0.2</td><td>0</td><td>cosine</td></tr><tr><td></td><td>CDs&amp;Vinyl</td><td>0.10</td><td>ModernBERT</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>all</td><td>0.1</td><td>0</td><td>cosine</td></tr><tr><td>BERT4Rec</td><td>ReDial</td><td>0.15</td><td>RoBERTa</td><td> $\mathrm { F P N s }$ </td><td>last</td><td>0.3</td><td>0</td><td>cosine</td></tr><tr><td></td><td>KuaiSAR</td><td>0.05</td><td>ModernBERT</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>all</td><td>0.1</td><td>0</td><td>cosine</td></tr><tr><td></td><td>Beauty</td><td>0.10</td><td>RoBERTa</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>last</td><td>0.2</td><td>0</td><td>cosine</td></tr><tr><td></td><td>CDs&amp;Vinyl</td><td>0.05</td><td>ModernBERT</td><td> $\mathrm { F F N s } { + } W _ { O }$ </td><td>all</td><td>0.2</td><td>0</td><td>cosine</td></tr></table>

![](images/ce32badde5b86a8978707d6d51db17ec4dd6e5a354fac28640a376851d706134.jpg)

![](images/b6f49c1f1c4fd8da90ec94a85a594a878d2988fbf5433b7773a159b0b2b3ec6a.jpg)  
Figure 4. Impact of different scopes of $\theta _ { M }$ , evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

## B.4. Detailed Experiments

In this section, we present more detailed experiments. In Section 4.3, we present experiments on sensitivity to ρ and PLM $\rho$ selection, and here we present the other experiments.

Scopes of $\theta _ { M }$ . We further examine the impact of different masking scopes $\theta _ { M }$ on model performance. As described in Section 3.3, we consider three options: masking only the feed-forward networks (FFNs), masking only the output projection of multi-head attention $( W _ { O } )$ , and masking both jointly $( \mathrm { F F N s } { + } W _ { O } )$ . The results in Figure 4 show several consisten patterns. First, the overall performance across different scopes remains relatively stable, suggesting that CRAMER is robust to the precise choice of $\theta _ { M }$ . Second, the joint scope $( \mathrm { F F N s } { + } W _ { O } )$ is most often optimal, particularly on larger datasets, where combining the two sources of control enables more expressive adaptation. Third, on smaller datasets, restricting the scope to either FFNs or $W _ { O }$ alone can be advantageous, likely because a more constrained control space reduces the risk of overfitting when training data are limited (Bejani & Ghatee, 2021). This observation aligns with the intuition that FFNs mainly act as memory slots while $W _ { O }$ governs attention aggregation–in low-data regimes, focusing on a single component may provide more stable and interpretable modulation, while in large-scale scenarios, the joint scope is more beneficial as abundant data supports richer request-conditioned adaptations and fully exploits the complementary roles of FFNs and $W _ { O }$ . In practice, a simple principle emerges: for large-scale, information-dense datasets, applying joint masking to both FFNs and $W _ { O }$ is generally the best choice, whereas for smaller datasets, selecting either FFNs or $W _ { O }$ individually may be preferable. These results confirm our design motivation in Section 3.3, where both FFNs and $W _ { O }$ were identified as high-leverage control targets for request-aware adaptation.

Regimes of $\phi _ { t }$ . We further investigate the effect of different fine-tuning regimes for the request encoder parameters $\phi _ { t }$ comparing three settings: (none) fully frozen PLM, (last) tuning only the last layer, and (all) end-to-end tuning. The results in Figure 5 show that CRAMER is generally robust across regimes, but some clear patterns emerge. Tuning only the last layer tends to yield stable gains over the frozen setting, particularly in smaller datasets or scenarios with limited reques information, where modest adaptation is sufficient and helps avoid overfitting. In contrast, end-to-end fine-tuning becomes more beneficial in large-scale or information-rich datasets, where abundant data and longer request texts can support deeper adaptation of the PLM. However, full fine-tuning may occasionally harm performance in low-data regimes, reflecting optimization instability and overfitting risks. In practice, last-layer tuning offers a comparatively robust and reliable default, while full fine-tuning is best reserved for settings with sufficient scale and linguistic richness to fully exploit the request

![](images/83243ec681220a28313907f2048d8d2336ec80a66998e354404d6324f93cbc5b.jpg)  
Figure 5. Impact of different regimes of ϕ<sub>t</sub>, evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

![](images/68c94b88e8e3ebdff596a570c7f6204841e93bcbe255e8ff45d7cab83292995c.jpg)  
Figure 6. Sensitivity of CRAMER to weight λ , evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

## encoder’s capacity.

Sensitivity to the Weight of KL Regularizer $\lambda _ { \mathbf { K I } }$ . We further study the influence of the KL regularization weight $\lambda _ { \mathrm { K L } }$ on model performance. Figure 6 reports NDCG@10 across different values of $\lambda _ { \mathrm { K L } } \in \{ 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 5 \}$ . Overall, the results indicate that CRAMER is relatively robust to the precise choice of $\lambda _ { \mathrm { K I } }$ , with only moderate fluctuations across datasets and backbones. On smaller datasets such as ReDial, slightly larger values (around 0.3–0.4) tend to be more effective, likely because stronger regularization prevents overfitting under limited training signals. On the contrary, on relatively larger datasets such as KuaiSAR and CDs&Vinyl, weaker regularization (around 0.1-0.2) performs best, while overly large $\lambda _ { \mathrm { K L } }$ values consistently degrade performance by constraining the masks too strongly. Beauty shows an intermediate trend, where moderate values (around 0.2–0.3) strike a reasonable balance. These observations suggest a simple guideline: a small $\lambda _ { \mathrm { K L } }$ generally sufficient for large-scale datasets, while moderate values are preferable for low-data regimes. Extreme settings should be avoided, as they either under-regularize or over-constrain the request-to-mask distribution.

Masks Shared or Not. We study whether the gating masks are sampled once and shared across the whole mini-batch (shared=1) or sampled independently for each instance (shared=0) during training. As shown in Figure 7, the non-shared regime dominates across all datasets and both backbones: its NDCG@10 is consistently and substantially higher. Notably, the best shared result never exceeds—and often trails well behind—the non-shared results. A plausible explanation is that sharing masks across an entire batch reduces the diversity of request-conditioned adaptation signals seen during training. This compromises the model’s ability to align masks closely with individual requests, leading to systematically weaker representations. By contrast, sampling masks independently per instance maintains alignment between each request and its induced control signal, enabling more faithful request-to-mask adaptation and stronger predictive performance. From a training perspective, while the shared strategy can slightly reduce runtime overhead by avoiding per-instance sampling, this computational saving is outweighed by the clear and consistent performance degradation. Therefore, non-shared sampling should be regarded as the default choice, as it yields both more accurate and more reliable models in practice.

![](images/776a3c7be243f0afc389a4382bc82b45a6c2017785f4f54439b99c4902cf8910.jpg)  
Figure 7. The effect of whether gates are sampled once and shared across the entire batch (1) or sampled independently per instance (0) in the training phase, evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

![](images/c58cb356be56a2fc55d377cb4d66ade9ffaf143c9531431fc74ec1f4b22b3e76.jpg)  
Figure 8. The effect of different temperature annealing schedule, evaluated using NDCG@10. For each setting, five evaluations were performed, the column height shows its average result.

Temperature Annealing Schedule. We further compare different schedules for annealing the Gumbel–Softmax temperature τ from 0.7 to 0.3 during training, including linear, exponential, and cosine decays. Figure 8 presents the results. Overall, cosine annealing achieves the best performance in most datasets and backbones, consistently outperforming linear decay and often surpassing exponential decay by a clear margin. Linear decay provides competitive results and is generally more stable than exponential, which tends to underperform due to overly rapid decreases in temperature at the early stages of training. The superiority of cosine annealing is likely because it offers a smoother and more gradual reduction, balancing exploration and exploitation more effectively while preserving sufficient stochasticity in the mask sampling process. In practice, cosine decay can be recommended as the default schedule, while linear decay remains a reasonable alternative when simplicity is preferred. Exponential decay is less favorable, as its aggressive early cooling can lead to suboptima convergence and weaker final accuracy.

## B.5. Efficiency and Overhead (BERT4Rec)

Table 6. Inference efficiency comparison for BERT4Rec backbone. Runtime and GPU memory usage are measured as average per-request cost under identical settings. “Vanilla Backbone” reports the backbone-only cost, and the remaining rows show the additional overhead introduced by each request-aware method.
<table><tr><td>Method</td><td>Runtime (s)</td><td>GPU Memory (MiB)</td></tr><tr><td>BERT4Rec</td><td>0.038</td><td>2119.6</td></tr><tr><td>+Query-SeqRec</td><td>+0.023</td><td>+1620.4</td></tr><tr><td>+BLaIR</td><td>+0.027</td><td>+1102.6</td></tr><tr><td>+LLM-ESR</td><td>+0.018</td><td>+1375.3</td></tr><tr><td>+REARANK</td><td>+9.184</td><td>+9412.5</td></tr><tr><td>+CRAMER (Ours)</td><td>+0.021</td><td>+1408.1</td></tr></table>

Table 7. Controlled scaling study within the BERT4Rec family on KuaiSAR. Runtime and GPU memory report the vanilla backbone cost and the additional overhead introduced by CRAMER.
<table><tr><td>Layers</td><td>Runtime (s)</td><td>+CRAMER (s)</td><td>Runtime Ratio</td><td>GPU Mem. (MiB)</td><td>+CRAMER (MiB)</td><td>Memory Ratio</td><td>NDCG@10</td><td>+CRAMER</td></tr><tr><td>4</td><td>0.038</td><td>+0.021</td><td>55.3%</td><td>2119</td><td>+1408</td><td>66.4%</td><td>0.366</td><td>0.434</td></tr><tr><td>6</td><td>0.056</td><td>+0.026</td><td>46.4%</td><td>2938</td><td>+1543</td><td>52.5%</td><td>0.379</td><td>0.447</td></tr><tr><td>8</td><td>0.075</td><td>+0.032</td><td>42.7%</td><td>3751</td><td>+1712</td><td>45.6%</td><td>0.376</td><td>0.451</td></tr><tr><td>10</td><td>0.094</td><td>+0.039</td><td>41.5%</td><td>4603</td><td>+2011</td><td>43.7%</td><td>0.385</td><td>0.462</td></tr></table>

## B.6. Controlled Scaling Study within BERT4Rec

To further examine how CRAMER scales with larger sequential recommender backbones, we conduct a controlled scaling study within the BERT4Rec family on KuaiSAR. Starting from the default 4-layer setting, we increase the backbone depth to 6, 8, and 10 layers while keeping the other settings unchanged. Table 7 reports NDCG@10 and the additional inference overhead introduced by CRAMER, together with runtime and memory ratios relative to the vanilla backbone.

The results show that CRAMER consistently improves recommendation performance across all tested backbone depths. Although the absolute runtime and memory overhead increase with deeper backbones, the relative overhead remains controlled and even decreases as the backbone becomes larger. This matches the row–column design: for each matrix $W ^ { ( l ) } \in \mathbb { R } ^ { \alpha _ { l } \times \beta _ { l } }$ , entrywise modulation would require $\alpha _ { l } \beta _ { l }$ control variables, whereas CRAMER uses only $\alpha _ { l } + \beta _ { l }$ row– column gates. Thus, the control dimensionality grows linearly with layer width rather than quadratically with the parameter size.

## B.7. Further Discussion on PLM

To further examine whether CRAMER is inherently unstable with respect to the choice of request encoder, we conduct an additional study using four BERT-family PLMs with increasing capacity (Tiny, Mini, Medium, and Base) as the initialization of the request encoder (the “Base” version corresponds to the model used in the main experiments). As shown in Table 8, we evaluate CRAMER on SASRec across four datasets using NDCG@10.

All four datasets exhibit the exact same strictly monotonic ranking among the PLM versions (Tiny < Mini < Medium < Base), which aligns precisely with their expected capacity ordering. For a single dataset, the probability of observing such a perfect ordering purely by chance is $1 / 2 4 \approx 0 . 0 4 1 7 < 0 . 0 5$ . Observing this pattern independently and consistently across all four datasets makes it highly unlikely that the variation originates from instability in CRAMER. Instead, these results indicate that CRAMER reliably leverages the semantic information provided by the request encoder: as PLM capacity improves, the encoder yields correspondingly richer representations of user requests, leading to more accurate and fine-grained control signals.

Importantly, this behavior does not represent a limitation of our framework, but rather reflects its inherently forwardcompatible design. Even with classic and widely deployed PLMs such as BERT-base, CRAMER already achieves strong performance, and ongoing trends toward more capable yet efficient PLMs suggest that future lightweight encoders will only further enhance the model’s ability to interpret user requests. Overall, the observed sensitivity arises primarily from intrinsic differences in PLM semantic expressiveness, and CRAMER remains both robust in practice and naturally aligned with continued advancements in text encoder architectures.

Table 8. Average NDCG@10 performance (five evaluations) of CRAMER under SASRec when initialized with four BERT versions of different capacity. Across all datasets, performance exhibits a strictly monotonic improvement that aligns with PLM capacity, illustrating that CRAMER faithfully leverages the semantic quality of the request encoder.
<table><tr><td>BERT Version</td><td>Tiny</td><td>Mini</td><td>Medium</td><td>Base</td></tr><tr><td>ReDial</td><td>0.403</td><td>0.411</td><td>0.419</td><td>0.428</td></tr><tr><td>KuaiSAR</td><td>0.420</td><td>0.425</td><td>0.432</td><td>0.436</td></tr><tr><td>Beauty</td><td>0.465</td><td>0.474</td><td>0.481</td><td>0.489</td></tr><tr><td>CDs&amp;Vinyl</td><td>0.449</td><td>0.454</td><td>0.460</td><td>0.472</td></tr></table>

## B.8. Intuitive User Case Study

To complement the aggregate metrics, we further present an intuitive case study on five specific users from the CDs&Viny dataset. Table 9 lists the detailed information of these users. For every user, we compute the rank position of the ground-truth positive item among all candidates under different backbones and request-aware methods. The results are summarized in Table 10.

Table 9. Detailed interaction information for the five users selected from the CDs&Vinyl dataset. For each user with a “User ID”, “#Inters” represents the total number of this user’s interactions, and “Last Item $\mathrm { I D } ^ { \prime \prime }$ represents the ID of the item in this user’s last interaction. Because we use leave-one-out (LOO) evaluation, the item in the last interaction is the ground-truth item in evaluation.
<table><tr><td>Index</td><td>User ID</td><td>#Inters</td><td>Last Item ID</td></tr><tr><td>#1</td><td>AE25K5V5RESPJ4WKCALB3ZVYYQPQ</td><td>11</td><td>B000008KJ8</td></tr><tr><td>#2</td><td>AFE66HHU55NJMALT34HEODVGEPQA</td><td>6</td><td>B00KNTDO3I</td></tr><tr><td>#3</td><td>AG2CJZJORAG7SG32SYNTNHICMGOQ</td><td>8</td><td>B07RF4JVGJ</td></tr><tr><td>#4</td><td>AGUPFBZ756HTU4YIF4QKQEX3NS2Q</td><td>13</td><td>BO0SFXFCWA</td></tr><tr><td>#5</td><td>AHQF2VXWQPUBKYR3RMJ6VDFDYUSQ</td><td>9</td><td>B08L47S144</td></tr></table>

Table 10. Average ranking of true positive items of users selected from CDs&Vinyl (smaller is better). For each setting, five evaluations were performed, and boldface indicates the best, i.e., lowest, average rank under the same backbone.
<table><tr><td>Method</td><td>#1</td><td>#2</td><td>#3</td><td>#4</td><td>#5</td></tr><tr><td>SASRec</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>18.2</td><td>28.4</td><td>27.0</td><td>19.0</td><td>17.8</td></tr><tr><td>Query-SeqRec</td><td>16.4</td><td>23.2</td><td>18.6</td><td>23.2</td><td>15.0</td></tr><tr><td>BLaIR</td><td>12.2</td><td>20.4</td><td>10.4</td><td>13.4</td><td>13.2</td></tr><tr><td>LLM-ESR</td><td>14.0</td><td>17.0</td><td>15.8</td><td>15.2</td><td>9.4</td></tr><tr><td>REARANK</td><td>13.6</td><td>23.2</td><td>11.2</td><td>10.0</td><td>11.4</td></tr><tr><td>CRAMER (Ours)</td><td>6.6</td><td>14.2</td><td>4.2</td><td>8.8</td><td>7.4</td></tr><tr><td>BERT4Rec</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>27.6</td><td>18.6</td><td>32.4</td><td>18.8</td><td>25.2</td></tr><tr><td>Query-SeqRec</td><td>21.2</td><td>17.2</td><td>14.0</td><td>17.0</td><td>21.2</td></tr><tr><td>BLaIR</td><td>19.4</td><td>11.4</td><td>16.2</td><td>13.0</td><td>14.2</td></tr><tr><td>LLM-ESR</td><td>11.4</td><td>13.8</td><td>18.2</td><td>7.6</td><td>13.8</td></tr><tr><td>REARANK</td><td>15.0</td><td>14.2</td><td>25.4</td><td>8.8</td><td>17.0</td></tr><tr><td>CRAMER (Ours)</td><td>7.0</td><td>8.2</td><td>11.4</td><td>5.2</td><td>12.0</td></tr></table>

As shown in Table 10, vanilla SASRec and BERT4Rec often place the ground-truth item at relatively low positions, indicating their limited ability to capture the user’s immediate intent. Adding request-aware baselines consistently improves the ranking quality but still exhibiting instability and fluctuations across different users. In contrast, CRAMER achieves the highest ranks for all selected users under both backbones, demonstrating more reliable alignment with the user’s natural-language request. This case study provides an intuitive, per-user confirmation that CRAMER delivers consistent improvements at the individual level beyond aggregate metrics.

## B.9. Details of Combination of Baselines and Backbones

In this section, we describe how each baseline is combined with the sequential recommendation backbones (SASRec and BERT4Rec) in our experiments. The combination strategies are detailed as follows:

Query-SeqRec. Query-SeqRec (He et al., 2022) integrates the request encoder with the sequential backbone. The backbone models the user’s historical sequence, while the request encoder provides a semantic representation of the request. These two signals are fused through concatenation, and the fused representation is used to score candidate items.

BLaIR. BLaIR (Hou et al., 2024) encodes item metadata and natural-language requests into a unified embedding space, such that their representations can be directly compared. The cosine similarity between the semantic embedding and the item embedding is first computed in this shared space. Specifically, the semantic similarity score between the semantic embedding $v _ { q }$ and the item embedding ${ \mathbf { } } v _ { i }$ is first computed by the BLaIR encoder. This score is then integrated with the collaborative score from the backbone:

$$
\begin{array} { r } { \operatorname { s c o r e } ( i ) = \gamma \cdot \cos ( \pmb { v } _ { q } , \pmb { v } _ { i } ) + ( 1 - \gamma ) \cdot f _ { \theta } ( \pmb { s } _ { u } , i ) , } \end{array}
$$

where $\gamma$ is a tunable fusion weight. In actual experiments, we set γ to the optimal value on each backbonee × dataset.

LLM-ESR. LLM-ESR (Liu et al., 2024) augments the backbone with semantic embeddings derived from large language models. Specifically, each item i is associated with both a semantic embedding $e _ { i } ^ { \mathrm { s e m } }$ (pre-computed by an LLM and projected via an adapter) and a collaborative embedding $e _ { i } ^ { \mathrm { c o l } }$ (from the backbone). The user representation is similarly decomposed

into $( { \pmb u } ^ { \mathrm { s e m } } , { \pmb u } ^ { \mathrm { c o l } } )$ . The final score is given by:

$$
\mathrm { s c o r e } ( i ) = [ e _ { i } ^ { \mathrm { s e m } } : e _ { i } ^ { \mathrm { c o l } } ] ^ { \top } [ \pmb { u } ^ { \mathrm { s e m } } : \pmb { u } ^ { \mathrm { c o l } } ] .
$$

REARANK. REARANK (Zhang et al., 2025) is used as a reranking stage on top of the backbone. The backbone first generates an initial ranking of candidate items based on the user’s history. These candidates, together with the request, are then passed to the LLM reranker, which performs reasoning over the top candidates and outputs a refined ranking.

## C. Additional Clarifications and Analyses

This appendix provides additional clarifications, analyses, and implementation details that complement the main text. These materials are intended to improve clarity and reproducibility, and to make explicit certain design choices that are implicit in the main presentation.

## C.1. Clarification on the Variational Motivation

While Section 3.2 presents a variational lower bound formulation, we emphasize that CRAMER does not aim to perform exact variational inference over control signals. Instead, the variational perspective serves as a conceptual motivation for introducing structured sparsity and a KL regularizer that stabilizes request-conditioned gating.

In practice, the optimization objective in Equation (6) should be understood as a principled surrogate that balances predictive accuracy and control sparsity, rather than as a strict ELBO maximization. Similar uses of variational formulations as design guides have appeared in prior work on controllable generation and structured regularization.

## C.2. Non-Destructive and Reversible Masking

It is important to clarify that CRAMER’s masking mechanism does not permanently modify or prune backbone parameters. All masks are applied multiplicatively at inference time and are conditioned on the current request. The underlying backbone parameters remain fixed and intact, and different requests induce different masks without interfering with each other.

This design enables reversible and non-destructive control: removing the request immediately restores the original backbone behavior, and no retraining or parameter update is required.

## C.3. Gradient Flow Through Discrete Gates

Although the gating vector m is discrete, CRAMER employs a straight-through estimator (STE) with a continuous relaxation in the backward pass. This approach is a standard technique for training models with discrete latent variables and has been widely adopted in prior work.

Importantly, gradients never flow into the frozen sequential recommender backbone parameters; they are restricted to the request-to-mask module only. This ensures stable optimization while preserving the efficiency and modularity of the framework.