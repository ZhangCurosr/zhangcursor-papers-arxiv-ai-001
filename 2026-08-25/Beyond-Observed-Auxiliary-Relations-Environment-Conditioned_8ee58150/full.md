# Beyond Observed Auxiliary Relations: Environment-Conditioned Modeling for Multi-Behavior Recommendation

Seunghan Lee   
Korea University   
Seoul, Republic of Korea   
seunghanlee@korea.ac.kr

Hyunsik Yoo University of Illinois Urbana-Champaign Champaign, IL, USA hy40@illinois.edu

Susik Yoon   
Korea University   
Seoul, Republic of Korea   
susik@korea.ac.kr

Jian Kang MBZUAI Abu Dhabi, United Arab Emirates jian.kang@mbzuai.ac.ae

SeongKu Kang<sup>∗</sup>   
Korea University   
Seoul, Republic of Korea   
seongkukang@korea.ac.kr

## Ab<sub>s</sub>t<sub>rac</sub>t

Multi-behavior recommendation (MBR) leverages auxiliary behavioral signals, such as clicks and add-to-cart, to enhance target behavior prediction like purchases. While recent graph neural network–based approaches have achieved strong performance by systematically propagating auxiliary behavior signals, they still sufer from two fundamental challenges inherent to auxiliary behaviors: (1) missing auxiliary signals, which hinder generalization to items without auxiliary observations, and (2) unreliable auxiliary signals, which amplify noise misaligned with the target behavior. To address these challenges in a unified manner, we propose BOAR, an environment-conditioned MBR framework that addresses missing and unreliable auxiliary signals through two complementary modules conditioned on auxiliary observability. Extensive experiments demonstrate that BOAR consistently outperforms state-of-the-art baselines, achieving up to 7.82% gains in HR@10 overall and up to 44.2% gains for target items without auxiliary observations, highlighting its ability to capture hidden preferences beyond observed auxiliary relations. Our code is available at: https://github.com/LSH0411/BOAR.

## CCS Conce<sub>p</sub>ts

• Information systems → Recommender systems.

## Ke<sub>y</sub>words

Multi-behavior recommendation, Graph Rewiring, Observation bias

## ACM Reference Format:

Seunghan Lee, Hyunsik Yoo,Jian Kang, Susik Yoon, and SeongKu Kang. 2026. Beyond Observed Auxiliary Relations: Environment-Conditioned Model ing for Multi-Behavior Recommendation. In Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 11 pages. https://doi.org/10.1145/3799682.3841033

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Users in modern recommender systems rarely express their preferences through a single type of behavior. Before making a purchase, users typically engage in multiple auxiliary behaviors, such as clicking items, adding them to carts, or saving them to wishlists. These auxiliary behaviors provide indirect yet complementary signals that reflect users’ latent preferences with respect to the final target behavior, such as purchasing. Accordingly, multi-behavior recommendation (MBR) aims to jointly model auxiliary behaviors together with the target behavior to better capture user preferences and improve recommendation accuracy [4, 21].

In multi-behavior recommendation, a wide range of approaches has been explored, from traditional matrix factorization [29, 53] to deep neural network-based models [11]. Among these approaches, graph neural networks (GNNs) have recently emerged as a core paradigm by explicitly modeling the relational structure between users and items, rather than treating interactions independently [37]. By representing diferent behavior types as distinct relations on graphs, GNNs systematically propagate auxiliary behavior signals to enhance target behavior prediction [21]. This relational modeling is particularly efective in real-world settings, where target behaviors are extremely sparse. Furthermore, combined with advanced learning paradigms such as self-supervised and multi-task learning, recent methods learn richer and more generalizable representations and achieve state-of-the-art performance [14, 44, 49].

Although efective, existing GNN-based methods still face two fundamental challenges arising from the nature of auxiliary behaviors. Specifically, auxiliary signals are often missing or unreliable: their absence does not necessarily indicate the absence of user interest, while their presence does not necessarily indicate target intent.

• C1: Missingness of Auxiliary Signals. In practice, users do not necessarily exhibit auxiliary behaviors for all items they may purchase. Importantly, such missingness is not at random [27, 35]: auxiliary behaviors are observed for a skewed subset of items, driven by factors such as platform exposure policies and item popularity, rather than solely reflecting the absence of user interest.<sup>1</sup> GNNs propagating over such biased relations may overrepresent frequently exposed items with many interactions [54].

![](images/d8bda87fe101cd1747040a2d92c21af0094d10bc06b4fc277aa481d05174ee9a.jpg)  
Fi<sub>g</sub>ure 1: Overview of MBR setu<sub>p</sub>. The left <sub>p</sub>anel illustrates trainin<sub>g</sub> and inference s<sub>p</sub>aces. The ri<sub>g</sub>ht <sub>p</sub>anel <sub>p</sub>resents an em<sub>p</sub>irical evaluation with two settings, reporting HR@10: (a) Hidden preference test, removing auxiliary interactions from observed target-positive items; and (b) Noise injection test, adding auxiliary interactions to target-negative items. Target-positive and tar<sub>g</sub>et-ne<sub>g</sub>ative items are defined with res<sub>p</sub>ect to the tar<sub>g</sub>et behavior. Detailed setu<sub>p</sub>s are <sub>p</sub>rovided in Section 5.2.2.

However, MBR must generalize beyond observed auxiliary relations and accurately recommend target behavior items without any auxiliary behaviors. This is illustrated by Figure 1 (left). The training space defined by observed interactions covers only a limited subset of the inference space, while hidden target items, i.e., target-positive items without auxiliary interactions, may appear outside the training space. Failure to capture such hidden preferences confines recommendations to items that users have already encountered, limiting their practical efectiveness.

• C2: Unreliability of Auxiliary Signals. Observed auxiliary behaviors do not necessarily imply purchase intent. While aux iliary behaviors may indicate potential interest, they can also be triggered by accidental clicks or casual browsing. As a result, auxiliary behaviors contain noisy signals that are misaligned with the target behavior. GNNs are structurally vulnerable to such noise because message passing can propagate and amplify incorrect information [8, 38]. In MBR, auxiliary relation-driven propagation may amplify signals loosely related to the target behavior, thereby hindering target behavior prediction. Figure 1 (left) illustrates this issue: not all items with auxiliary behaviors are target-positive items that eventually lead to purchase, as auxiliary interactions can also involve target-negative items.

While prior methods partially address these challenges, they remain insuficient to fully resolve them, and to our knowledge, no existing method addresses both simultaneously. Regarding C1, MEMBER [23] introduces self-supervised learning to reduce overreliance on observed auxiliary interactions; however, it still relies solely on the observed auxiliary signals, without explicitly uncovering hidden preferences. Regarding C2, several studies [26, 51] regulate message passing to suppress noise propagation; however, the regulation is only weakly guided by the target behavior, which limits its efectiveness in filtering unreliable auxiliary signals. Indeed, in Figure 1 (right), we evaluate recent GNN-based methods relevant to each challenge (MEMBER [23] for C1, and MuLe [26], HGIB [51] for C2) under settings that intensify each challenge by increasing the ratio of hidden target items and auxiliary noise, respectively. Existing methods show substantial performance degradation, suggesting limited capability to handle challenges arising from auxiliary behaviors.

As a solution, we propose BOAR (Beyond Observed Auxiliary Relations), a framework that rewires given auxiliary relations into target-informative signals instead of using them as-is. This rewiring requires diferent strategies depending on auxiliary observability: when auxiliary observability is lacking, C1 calls for debiased densification to surface hidden preferences, whereas with strong auxiliary observability, C2 calls for selective pruning to filter unreliable signals. Although both challenges could in principle be addressed within a single model, doing so may yield suboptimal signals under training data imbalance: target interactions without auxiliary interactions form only a small fraction (e.g., 16.1% in Tmall), causing the model to focus overly on auxiliary-observed cases and limiting hidden preference discovery (see Section 3.2 for detailed analysis). This naturally motivates an environment-conditioned modular design: BOAR employs two lightweight modules that focus on comple mentary rewiring strategies, i.e., recovering hidden preferences and pruning unreliable signals. To coordinate these specialized modules, BOAR uses auxiliary observability as an environment condition and probabilistically adjusts their contributions to each target behavior prediction, addressing the two challenges in a complementary manner. Our main contributions are:

• We highlight two fundamental challenges of auxiliary behaviors that have not yet been fully addressed and, to our knowledge, present the first attempt to jointly address both.

• We propose BOAR, an environment-conditioned modular MBR framework that overcomes the limitations of observed auxiliary relations through conditional modular modeling.

• Extensive experiments show that BOAR delivers consistent gains, especially for target behavior without auxiliary observations, with eficiency comparable to state-of-the-art baselines.

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k<sub>s</sub>

Multi-Behavior Recommendation (MBR). MBR leverages diverse auxiliary behaviors, such as clicks and add-to-cart, to alleviate the data sparsity of target behaviors like purchases. Early studies explored various techniques, including matrix factorization [29, 53], deep neural networks [11], and attention-based approaches [16, 41, 42]. Recently, GNN-based methods have emerged as the dominant approach [21, 39, 42–44]. Existing methods adopt diverse graph encoding strategies, including integrating multiple behavior types into a unified graph [4, 21, 42, 43, 52], explicitly modeling natural behavioral sequences (e.g., click→cart→purchase) [7, 13, 30, 45], and encoding each behavior graph in parallel [23, 26, 31, 46, 51]. Further more, advanced training techniques such as self-supervised learning [14, 23, 44, 49, 51] and multi-task learning [13, 30, 31, 44, 45, 49] have been incorporated to enhance target behavior prediction.

However, reliance on auxiliary behaviors introduces the two aforementioned challenges, and recent GNN-based methods have partially addressed them. For C1, MEMBER [23] adopts a mixtureof-experts framework to distinguish purchases with auxiliary behaviors from those without. It employs self-supervised learning to reduce the dominance of purchases preceded by auxiliary behaviors, encouraging the model to better capture purchase patterns that occur independently of auxiliary signals. For C2, MuLe [26] introduces attention-based message passing to adjust the contributions of auxiliary behaviors and mitigate uncertainty during propagation, while HGIB [51] applies the information bottleneck principle to filter out information weakly related to the final prediction.

Despite these advancements, substantial room for improvement remains in addressing each challenge, and to our knowledge, no prior method has addressed both challenges simultaneously. Specifically, no dedicated efort has been made to explicitly handle bias or uncover hidden target items, limiting the ability to address C1. Meanwhile, existing methods control propagation primarily based on similarity induced by auxiliary behaviors, with limited guidance from target behaviors, which remains insuficient to resolve C2.

Graph Rewiring for Imperfect Graphs. In MBR, the two aforementioned challenges are fundamentally manifested as structural imperfections in auxiliary behavior graphs. Graph rewiring (GR) is a core methodology in graph structure learning that reconstructs edge connections to address such imperfections [9, 12, 15, 28]. GR methods typically jointly optimize graph structure and node representations to recover missing edges and mitigate unreliable con nections. A common approach infers potential connections based on learned node embedding similarities, followed by pruning unreliable edges or adding new ones [3, 20, 36, 48]. These techniques have proven efective across domains, including single-behavior recommendation [36, 48], graph classification [3], and social recommendation [20]. Such improvements are attributed to pruning unreliable connections and strengthening homophilic relationships, enhancing graph quality.

However, applying existing GR strategies to MBR may be suboptimal, as auxiliary behavior graphs contain imperfections tied to auxiliary observability. In particular, conventional similarity-based rewiring may amplify these imperfections: densification may favor items with abundant auxiliary interactions, failing to surface hidden preferences without auxiliary evidence (C1), while pruning may mistakenly remove informative auxiliary connections as noise (C2). This calls for an auxiliary observability-conditioned graph rewiring strategy tailored to the unique challenges of MBR.

## 3 P<sub>re</sub>li<sub>m</sub>i<sub>nar</sub>i<sub>es</sub>

## 3<sub>.</sub>1 P<sub>ro</sub>bl<sub>em</sub> F<sub>ormu</sub>l<sub>a</sub>ti<sub>on</sub>

Let U and I denote the sets of users and items. We consider a set of behaviors $\mathcal { B } = \{ b _ { 1 } , . . . , b _ { | \mathcal { B } | } \} ( \mathrm { e . g . }$ , click, cart, ... , buy), where $b _ { \mathrm { t a r } } =$ $b _ { | \mathcal { B } | }$ is the target behavior and $\mathcal { B } \backslash \{ b _ { \mathrm { t a r } } \}$ are auxiliary behaviors. For each behavior � ∈ B, we define a bipartite graph $G _ { b } = ( \mathcal { U } \cup \mathcal { I } , \mathcal { E } _ { b } )$ where $\mathcal { E } _ { b } \subseteq \mathcal { U } \times \mathcal { I }$ denotes the observed interactions. Let $\mathcal { E } _ { \mathrm { a u x } } =$ $\begin{array} { r } { \bigcup _ { b \in \mathcal { B } \backslash \{ b _ { \mathrm { t a r } } \} } \mathcal { E } _ { b } } \end{array}$ denote the set of all observed auxiliary interactions.

![](images/e87ddf36c5b13aa9cbbbad9fa13954923fc3de558d023fdf3557c6451af3b7ab.jpg)

![](images/0a5a355ab1eb89787f7d1a14d6275206e7c11e13bf5569b287218586ec218631.jpg)  
Fi<sub>gure</sub> 2<sub>:</sub> T<sub>ra</sub>i<sub>n</sub>i<sub>ng-se</sub>t i<sub>m</sub>b<sub>a</sub>l<sub>ance</sub> b<sub>e</sub>t<sub>ween</sub> t<sub>arge</sub>t i<sub>n</sub>t<sub>erac</sub>ti<sub>ons</sub> with (Obs) and without (Unobs) auxiliary observations (left), and the resultin<sub>g p</sub>erformance <sub>g</sub>a<sub>p</sub> between the two <sub>g</sub>rou<sub>p</sub>s (right). Results are reported on the Tmall dataset.

For each pair (�, �), we define a binary random variable $R _ { u i } \in$ {0, 1} indicating whether user � engages in the target behavior on item �. Our goal is to learn a function � (�, �) that estimates the probability that a user � exhibits the target behavior with item �:

$$
f ( u , i ) \ \approx \ \mathbb { E } [ R _ { u i } \ | \ u , i ] \ = \ P ( R _ { u i } = 1 \ | \ u , i ) .\tag{1}
$$

We focus on addressing the aforementioned challenges, C1: missing auxiliary signals and C2: unreliable auxiliary signals, arising from the nature of auxiliary behaviors.

## 3.2 Environment-Conditioned Modelin<sub>g</sub>

As discussed above, the two challenges require distinct graph rewiring strategies: densifying the target behavior graph to recover hidden target items when auxiliary evidence is lacking, and pruning the auxiliary behavior graphs to suppress noise misaligned with target behavior when auxiliary evidence is abundant.

Empirical Evidence. Figure. 2 (left) shows that among target interactions, those with auxiliary interactions substantially outnumber those without. This imbalance can make GNN-based models overly focus on items with auxiliary observations. Indeed, Figure. 2 (right) shows that even state-of-the-art methods consistently sufer a significant performance drop on hidden target items. This suggests that a single model may struggle to derive rewiring signals suited to both cases, particularly for hidden preference discovery. This motivates a design that incorporates auxiliary observability, so that rewiring signals can be efectively derived for both auxiliary-observed and -unobserved cases.

Environment-Conditioned Modeling. The above observations motivate a modular formulation in which target preference estimation is conditioned on auxiliary observability. By the law of total expectation, the target preference can be decomposed as:

$$
{ \mathbb E } [ R _ { u i } \ | \ u , i ] = \sum _ { e \in \{ 0 , 1 \} } P ( E _ { u i } { = } e \ | \ u , i ) \ { \mathbb E } [ R _ { u i } \ | \ u , i , E _ { u i } { = } e ] ,\tag{2}
$$

where $E _ { u i }$ denotes the auxiliary-observability environment of the pair (�, �). In the interaction log, the presence of auxiliary interactions provides an observable basis for this condition: $E _ { u i } = 0$ for $( u , i ) \not \in \mathcal { E } _ { \mathrm { a u x } }$ and $E _ { u i } = 1$ for $( u , i ) \in \mathcal { E } _ { \mathrm { a u x } }$ . Note that $E _ { u i } = 0$ only means that no auxiliary behavior is observed in the current log; such pairs may become observable as user activity continues. Thus, the final prediction uses $P ( E _ { u i } = e \mid u , i )$ as a soft weight, rather than relying solely on the observed environment information.

Accordingly, we model the conditional expectations with two environment-conditioned sub-modules:

$$
\begin{array} { r } { f _ { 0 } ( u , i ) \approx \mathbb { E } [ R _ { u i } \mid u , i , E _ { u i } { = } 0 ] , } \\ { f _ { 1 } ( u , i ) \approx \mathbb { E } [ R _ { u i } \mid u , i , E _ { u i } { = } 1 ] . } \end{array}\tag{3}
$$

Here, $f _ { 0 }$ focuses on rewiring under the auxiliary-unobserved condi tion, where hidden preferences should be recovered. In contrast, $f _ { 1 }$ focuses on rewiring under the auxiliary-observed condition, where unreliable auxiliary relations should be filtered.

The final prediction is obtained by softly aggregating the conditional predictions as $\begin{array} { r } { f ( u , i ) ~ = ~ \sum _ { e \in \{ 0 , 1 \} } P ( E _ { u i } = e \mid u , i ) ~ f _ { e } ( u , i ) } \end{array}$ Here, $P ( E _ { u i } = e \mid u , i )$ controls the contribution of each sub-module. This modular formulation enables coordination of complementary rewiring strategies across auxiliary-observability conditions, while behavior-specific importance is further taken into account within the modules as appropriate. The concrete instantiation is introduced in the subsequent section.

## 4 P<sub>ropose</sub>d M<sub>e</sub>th<sub>o</sub>d

We present BOAR, an environment-conditioned MBR framework (Figure 3). BOAR first constructs disentangled input representations (Sec. 4.1), and then applies two modules, with distinct graph rewiring and learning strategies conditioned on the environment:

• (Sec.4.2) Debiased densification module $f _ { 0 }$ identifies hidden target items by mining hidden preferences while debiasing auxiliary relations, providing densification signals.

• (Sec.4.3) Target-guided refinement module $f _ { 1 }$ prunes auxiliary relations misaligned with the target behavior, refining auxiliary graphs toward target-consistent relations.

Section 4.4 describes modular integration for learning and inference.

## 4.1 In<sub>p</sub>ut Re<sub>p</sub>resentation Construction

We construct input user/item representations using both the global graph and the target-behavior graph via a LightGCN [18] encoder. Let $\mathbf { \bar { E } } \in \mathbb { R } ^ { ( | \mathcal { U } | + | \mathcal { \bar { I } } | ) \times d }$ denote the initial embedding matrix that stacks user and item embeddings. We obtain:

$$
\begin{array} { r } { { \mathbf { E } } ^ { \mathrm { g l o } } = \mathrm { L G C N } \big ( \mathbf { A } _ { \mathrm { g l o } } , \mathbf { E } \big ) , \qquad { \mathbf { E } } ^ { \mathrm { t a r } } = \mathrm { L G C N } ( \mathbf { A } _ { \mathrm { t a r } } , \mathbf { E } ) , } \end{array}\tag{4}
$$

where $\mathbf { A } _ { \mathrm { g l o } }$ denotes the adjacency matrix of the global graph $\mathcal { G } _ { \mathrm { g l o } }$ constructed from all behaviors, i.e., $\textstyle { \mathcal { E } } _ { \mathrm { g l o } } = \bigcup _ { b \in { \mathcal { B } } } { \mathcal { E } } _ { b }$ , and $\mathbf { A } _ { \mathrm { t a r } }$ denotes the adjacency matrix of the target behavior graph. The global embeddings capture comprehensive multi-behavior signals and provide warm-start features for subsequent behavior-specific encoding [22, 26, 51]. In each module, the global embedding $\bar { \bf E } ^ { \mathrm { g l o } }$ serves as a source of densification $\left( f _ { 0 } \right)$ or refinement $( f _ { 1 } )$ signals, while the target embedding $\mathbf { E } ^ { \mathrm { t a r } }$ is used for final prediction. Note that, to enable environment-conditioned modeling without increasing model capacity, we split the initial embedding into two halves and encode each half with both the global and target graph encoders.

## 4<sub>.</sub>2 D<sub>e</sub>bi<sub>ase</sub>d D<sub>ens</sub>ifi<sub>ca</sub>ti<sub>on</sub> M<sub>o</sub>d<sub>u</sub>l<sub>e</sub>

The densification module $f _ { 0 }$ mines hidden preferences to provide densification signals for the target behavior graph, and consists of three components: (1) auxiliary popularity adversarial learning to adversarially suppress popularity-driven observation bias from item representations, (2) target hidden preference miner to eficiently discover hidden preferences, and (3) debiased densification learning to safely inject the mined knowledge into the final prediction.

4.2.1 Auxiliary Popularity Adversarial Learning. To reduce popularity-driven observation bias before hidden preference mining, we adapt the adversarial debiasing [40, 50, 55]. The key idea is to prevent item representations from overly encoding auxiliary behavior popularity, so that they can better reflect user preference.

This is achieved through adversarial learning with a gradient reversal layer (GRL) [10]. Following [55], we construct a proxy signal for popularity-driven bias from the total number of auxiliary interactions each item receives: $y _ { i } = 1$ if the count exceeds the median interaction count across all items, and $y _ { i } = 0$ otherwise. This signal construction strategy provides practical and balanced adversarial supervision.<sup>2</sup> We then define the adversarial supervision set as $S _ { \mathrm { a d v } } = \{ ( i , y _ { i } ) \}$

A discriminator $f _ { \phi } ,$ , implemented as a linear layer with a sigmoid function, is trained to predict the proxy signal $y _ { i }$ ofeach item �. Since popularity is an item-level property, the discriminator takes the item embedding as input and outputs $\hat { y } _ { i } = \sigma ( f _ { \phi } ( \mathrm { G R L } ( \mathbf { e } _ { i } ^ { \mathrm { g l o } } ) ) )$ , where � is the sigmoid function. The discriminator is trained adversarially via the GRL, such that the item representations are updated to hinder accurate popularity signal prediction:

$$
\mathcal { L } _ { a d v } = \mathbb { E } _ { ( i , y _ { i } ) \sim S _ { \mathrm { a d v } } } \left[ - y _ { i } \log \hat { y } _ { i } - \left( 1 - y _ { i } \right) \log \left( 1 - \hat { y } _ { i } \right) \right]\tag{5}
$$

This adversarial objective reduces the influence of popularity information in item representations, providing a basis for hidden preference mining.

4.2.2 Target Hidden Preference Miner. Based on the popularitysuppressed representations, we uncover potential target relations. Existing graph rewiring methods [15, 20] typically rely on exhaustive pairwise similarity computation to identify highly similar node pairs, incurring prohibitive costs. To address this, we propose an eficient three-stage mining strategy: (i) locality-sensitive hashing (LSH)-based candidate retrieval, (ii) learnable edge selection, and (iii) weighted graph augmentation.

Angular LSH-based Candidate Retrieval. To eficiently retrieve candidate items, we adopt angular LSH [1, 25], which enables fast approximate nearest-neighbor search based on angular similarity. Unlike traditional LSH, angular LSH constructs hash buckets using a single random projection followed by an argmax operation, which can be eficiently computed on GPUs via matrix multiplication. Moreover, its collision probability is monotonically related to angular similarity rather than L2 distance, making it better aligned with the inner product–based prediction.

We first draw a random projection matrix R $\mathbf { \tau } _ { \mathbf { \epsilon } } \in \mathbb { R } ^ { d \times d ^ { \prime } }$ with i.i.d. entries $R _ { p q } \sim N ( 0 , 1 )$ , where $d ^ { \prime }$ is the number of hash buckets and each column defines a random angular direction in the embedding space. For each node $x ,$ we normalize its representation and project it onto these directions, obtaining projection scores $\mathbf { s } _ { x }$ that reflect its angular alignment.<sup>3</sup> Each node is then assigned to a hash bucket corresponding to the dominant projection direction as: $h ( x ) = \arg \operatorname* { m a x } _ { k } ( s _ { x } ) _ { k }$ . This procedure ensures that nodes with similar angular directions are likely to be grouped into the same bucket, without explicitly computing pairwise similarities.

Debiased Densification: Target Hidden Preference Miner  
![](images/7ea461c7b7988cbd1eee5609776272ec75e4fe54749f213727aa5967d5be9cee.jpg)  
Figure 3: (Left) Overview of the BOAR framework. (Right) Construction of densification and refinement signals in each module, which are subse<sub>q</sub>uentl<sub>y</sub> incor<sub>p</sub>orated via self-su<sub>p</sub>ervised contrastive learnin<sub>g</sub>. Best viewed in color.

For each user �, we find unobserved items from the same bucket, i.e., $\{ i \mid h ( i ) = h ( u ) \}$ , and retrieve top-� items by cosine similarity to form the candidate set $\mathcal { E } _ { \mathrm { c a n d } } ( u )$ ). The total candidates across all users are aggregated as: $\begin{array} { r } { \mathcal { E } _ { \mathrm { c a n d } } = \bigcup _ { u \in \mathcal { U } } \{ ( u , i ) \ | \ i \in \mathcal { E } _ { \mathrm { c a n d } } ( u ) \} } \end{array}$

Learnable Edge Selection. Not all candidates are equally reliable for target behavior prediction. To selectively retain edges truly aligned with this objective, we adopt a learnable edge selection strategy jointly optimized with the prediction task.

For each candidate pair $( u , i ) \in \mathcal { E } _ { \mathrm { c a n d } } ,$ , we predict a binary add-orskip decision using � , implemented as a linear layer. Specifically, we compute logits $\mathbf { z } _ { u i } ^ { \mathrm { a d d } } \ = \ f _ { \mathrm { a d d } } ( [ \mathbf { e } _ { u } ^ { \mathrm { g l o } } \ \lVert \ \mathbf { e } _ { i } ^ { \mathrm { g l o } } ] ) \ \in \ \mathbb { R } ^ { 2 }$ , where each output dimension corresponds to add and skip decisions, respectively. We apply the Gumbel-Softmax [19] to obtain diferentiable selection mask:<sup>4</sup>

$$
m _ { u i } ^ { \mathrm { a d d } } = \mathrm { G u m b e l S o f t m a x } ( \mathbf { z } _ { u i } ^ { \mathrm { a d d } } ; \tau ) [ 0 ] \in \{ 0 , 1 \} ,\tag{6}
$$

The selection mask is learned via backpropagation using the Gumbel-Softmax relaxation, favoring edges beneficial for target behavior prediction while suppressing misaligned ones. We guide readers unfamiliar with the Gumbel-Softmax to [19].

Weighted Graph Augmentation. We construct the densified target graph using the selected edges. To reflect their varying importance, each edge is assigned a weight $\boldsymbol { w } _ { u i }$ that combines angular similarity and magnitude consistency [6]:

$$
w _ { u i } = \frac { 1 } { 2 } \left( 1 + \cos ( \mathbf { e } _ { u } ^ { \mathrm { g l o } } , \mathbf { e } _ { i } ^ { \mathrm { g l o } } ) \right) \cdot \exp \left( - \frac { 1 } { 2 \sigma ^ { 2 } } \| \mathbf { e } _ { u } ^ { \mathrm { g l o } } - \mathbf { e } _ { i } ^ { \mathrm { g l o } } \| _ { 2 } ^ { 2 } \right) .\tag{7}
$$

Original target edges are retained with unit weight, while selected candidates are added with weight $w _ { u i } .$ where � controls the sensitivity to the Euclidean distance between embeddings, and is set to 20 following [6]. We define the weighted adjacency matrix $\tilde { \mathbf { A } } _ { \mathrm { t a r } }$ as:

$$
\tilde { \mathbf { A } } _ { \mathrm { t a r } } ( u , i ) = \left\{ \begin{array} { l l } { 1 , \quad } & { ( u , i ) \in \mathcal { E } _ { \mathrm { t a r } } , } \\ { w _ { u i } , } & { ( u , i ) \in \mathcal { E } _ { \mathrm { c a n d } } \mathrm { a n d } m _ { u i } ^ { \mathrm { a d d } } = 1 , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

The matrix $\tilde { \mathbf { A } } _ { \mathrm { t a r } }$ is symmetric, as $w _ { u i } = w _ { i u }$ by construction.

4.2.3 Debiased Densification Learning. While the augmented graph reveals hidden preferences, it inevitably contains noise from the mining process, making direct use for final prediction suboptimal. We propose a strategy to robustly exploit densification signals.

Preference Densification Learning. We adopt self-supervised contrastive learning with an InfoNCE loss [32] to encourage consistency between embeddings from the original and augmented graphs. This alignment emphasizes signals consistently supported by both views, enabling a more stable learning than directly treating augmented edges as ground-truth relations. Specifically, we align the original target embeddings E<sup>tar</sup> (input of this module) with the augmented embeddings $\tilde { \mathbf { E } } ^ { \mathrm { t a r } } = \mathrm { L G C N } ( \bar { \tilde { \mathbf { A } } } _ { \mathrm { t a r } } , \mathbf { E } ^ { \mathrm { g l o } } )$ ). The user-side loss for densification is defined as:

$$
\mathcal { L } _ { d e n s e } ^ { u s e r } = - \sum _ { u \in \mathcal { U } } \log \frac { \exp \left( { \sin ( \mathbf { e } _ { u } ^ { \mathrm { t a r } } , \tilde { \mathbf { e } } _ { u } ^ { \mathrm { t a r } } ) / \tau } \right) } { \sum _ { u ^ { \prime } \in \mathcal { U } } \exp \left( { \sin ( \mathbf { e } _ { u } ^ { \mathrm { t a r } } , \tilde { \mathbf { e } } _ { u ^ { \prime } } ^ { \mathrm { t a r } } ) / \tau } \right) } .\tag{9}
$$

where sim(·, ·) denotes a similarity function and � is the temperature hyperparameter. The item-side loss $\mathcal { L } _ { d e n s e } ^ { i t e m }$ is defined analogously, and the overall objective is $\mathcal { L } _ { d e n s e } = \mathcal { L } _ { d e n s e } ^ { u s e r } + \mathcal { L } _ { d e n s e } ^ { i t e m }$

Debiased BPR Loss. While auxiliary popularity adversarial learning mitigates popularity-driven bias at the representation level, it does not fully eliminate observation bias in preference learning. For a more explicit debiasing, we adopt the self-normalized inverse propensity score (SNIPS) [35] to reweight positive interactions in the ranking loss. SNIPS reduces the variance of the IPS estimator by normalizing importance weights, thereby stabilizing optimization even under imperfect propensity estimates.

For each (�, �), we first estimate the auxiliary-observation propensity $\hat { p } _ { u i } = \hat { P } ( E _ { u i } = 1 \mid u , i )$ , which measures how likely the pair is to be observed in auxiliary behaviors.<sup>5</sup> The debiasing weight is defined by the normalized inverse propensity, $\begin{array} { r } { \omega _ { u i } \propto 1 / \hat { p } _ { u i } , } \end{array}$ which reduces estimator variance. Intuitively, $\omega _ { u i }$ assigns smaller weights to pairs with high auxiliary observation propensity and larger weights to those with low propensity, correcting the skew induced by observation bias. Let $s _ { u i } = \mathbf { e } _ { u } ^ { \mathrm { t a r } \top } \mathbf { e } _ { i } ^ { \mathrm { t a r } }$ denote the prediction score for target behavior. The debiased BPR loss is:

$$
\mathcal { L } _ { r a n k } = - \sum _ { ( u , i ) \in \mathcal { E } _ { \mathrm { t a r } } } \sum _ { j : ( u , j ) \notin \mathcal { E } _ { \mathrm { t a r } } } \omega _ { u i } \cdot \log \sigma ( s _ { u i } - s _ { u j } )\tag{10}
$$

Overall Learning Objective. The total loss of $f _ { 0 }$ is as follows:

$$
\mathcal { L } _ { f _ { 0 } } = \mathcal { L } _ { r a n k } + \lambda _ { a d v } \mathcal { L } _ { a d v } + \lambda _ { d e n s e } \mathcal { L } _ { d e n s e } ,\tag{11}
$$

where $\lambda _ { a d v }$ and $\lambda _ { d e n s e }$ are loss-balancing hyperparameters.

Theoretical grounding for $f _ { 0 }$ is provided in Appendix A.1.1: (i) Theorem A.1 establishes unbiasedness of the IPS-weighted loss; (ii) Lemma A.2 shows that the GRL objective suppresses popularitydriven bias from item representations; and (iii) Corollary A.3 shows that both are jointly necessary for hidden preference recovery.

## 4<sub>.</sub>3 T<sub>arge</sub>t<sub>-</sub>G<sub>u</sub>id<sub>e</sub>d R<sub>e</sub>fi<sub>nemen</sub>t M<sub>o</sub>d<sub>u</sub>l<sub>e</sub>

The refinement module $f _ { 1 }$ leverages target behavior signals to prune the auxiliary behavior graphs, and consists of two components: (1) a target-guided auxiliary graph refiner, which selectively prunes and reweights auxiliary relations, and (2) target-guided preference learning that provides target-aligned signals for auxiliary refinement and optimizes target behavior prediction.

4.3.1 Target-guided Auxiliary Graph Refiner. We perform target-guided refinement on each auxiliary behavior graph $\mathcal { G } _ { b }$ using a behavior-specific LGCN encoder, with input auxiliary embeddings $\mathbf { e } ^ { b } = \mathbf { e } ^ { \mathrm { g l o } }$ . The target embeddings $\mathbf { E } ^ { \mathrm { t a r } }$ serve as fixed anchors, and for notational simplicity, we denote $\mathbf { t } _ { x } = \mathrm { d e t a c h } ( \mathbf { e } _ { x } ^ { \mathrm { t a r } } )$

Target-Guided Feature Refinement. First, to handle feature-level misalignment, we introduce a feature-wise gate that adaptively balances auxiliary representations with target anchors. For each node �, the gate $\pmb { \alpha } ^ { b } \in ( 0 , 1 ) ^ { d }$ determines, at each feature dimension, how much to rely on the auxiliary signals versus the target anchor.

The gate is computed by conditioning on both representations:

$$
\begin{array} { r l } & { \pmb { \alpha } _ { x } ^ { b } = \sigma \Big ( \mathbf { W } _ { 1 } ^ { b } \big [ \mathbf { e } _ { x } ^ { b } \big \| \mathbf { t } _ { x } \big ] + \mathbf { b } _ { 1 } ^ { b } \Big ) \in ( 0 , 1 ) ^ { d } } \\ & { ~ \mathbf { g } _ { x } ^ { b } = \pmb { \alpha } _ { x } ^ { b } \odot \mathbf { e } _ { x } ^ { b } + \left( 1 - \alpha _ { x } ^ { b } \right) \odot \mathbf { t } _ { x } , } \end{array}\tag{12}
$$

where $x \in \mathcal { U } \cup \mathcal { I }$ . This adaptive interpolation selectively filters auxiliary features inconsistent with the target preference space.

Selective Edge Pruning. Beyond feature-level refinement, we further refine the auxiliary graph structure by selectively pruning uninformative edges. The goal is to prevent auxiliary edges inconsistent with the target preference from propagating noise. For each edge $( u , i ) \in \mathcal { E } _ { b }$ , we predict a binary keep-or-drop decision using $f _ { \mathrm { d r o p } }$ , implemented as a linear layer. Specifically, we compute logits $\mathbf { z } _ { u i } ^ { b } \overset { \cdot } { = } f _ { \mathrm { d r o p } } ( [ \mathbf { g } _ { u } ^ { b } \parallel \mathbf { g } _ { i } ^ { b } ] ) \in \mathbb { R } ^ { 2 }$ , where each output dimension corresponds to keep and drop decisions, respectively. We apply Gumbel-Softmax to obtain diferentiable pruning:

$$
m _ { u i } ^ { b } = \mathrm { G u m b e l S o f t m a x } ( \mathbf { z } _ { u i } ^ { b } ; \tau ) [ 0 ] \in \{ 0 , 1 \} ,\tag{13}
$$

Optimized by the target-guided learning introduced in the subsequent subsection, edges misaligned with the target behavior are suppressed. An edge is retained if $m _ { u i } ^ { b } = 1 ;$ ; otherwise it is pruned. Weighted Graph Refinement. We construct a refined adjacency matrix $\tilde { \mathbf { A } } _ { b }$ by selecting a subset of edges with the weighting scheme in $\operatorname { E q . } ( 7 ) ;$ , substituting refined representations $\mathbf { g } _ { u } ^ { b } , \mathbf { g } _ { i } ^ { b }$ for ${ \bf e } _ { u } ^ { \mathrm { g l o } } , { \bf e } _ { i } ^ { \mathrm { g l o } } ;$

$$
\tilde { \mathbf { A } } _ { b } ( u , i ) = \left\{ \begin{array} { l l } { w _ { u i } ^ { b } , } & { ( u , i ) \in \mathcal { E } _ { b } ^ { \prime } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{14}
$$

where $\mathcal { E } _ { b } ^ { \prime } = \{ ( u , i ) \in \mathcal { E } _ { b } \mid m _ { u i } ^ { b } = 1 \}$ } denotes the set ofretained edges after pruning. The resulting matrix $\tilde { \mathbf { A } } _ { b }$ is symmetric, as $w _ { u i } ^ { b } = w _ { i u } ^ { b }$ by construction. Given the refined adjacency matrix $\tilde { \mathbf { A } } _ { b } ,$ we obtain propagated representations as: $\tilde { \mathbf { E } } ^ { b } = \mathrm { L G C N } ( \tilde { \mathbf { A } } _ { b } , \mathbf { E } ^ { \mathrm { g l o } } )$ . The same refinement procedure is applied to each auxiliary behavior �.

4.3.2 Target-Guided Preference Learning. We introduce a target-guided preference learning that leverages contrastive alignment to provide supervisory signals for refining auxiliary graphs and strengthening target behavior prediction.

Auxiliary Refinement Learning. Specifically, we treat the target embedding t as a fixed anchor to prevent auxiliary noise from con taminating the target space, and align each auxiliary representation with it via contrastive learning. The user-side refinement loss for auxiliary behavior � is:

$$
\mathcal { L } _ { r e f i n e } ^ { u s e r , b } = - \sum _ { u \in \mathcal { U } } \log \frac { \exp \left( \sin ( \mathbf { t } _ { u } , \mathbf { e } _ { u } ^ { b } ) / \tau \right) } { \sum _ { u ^ { \prime } \in \mathcal { U } } \exp \left( \sin ( \mathbf { t } _ { u } , \mathbf { e } _ { u ^ { \prime } } ^ { b } ) / \tau \right) } .\tag{15}
$$

The item-side loss is defined analogously, and we sum both losses over all auxiliary behaviors: $\begin{array} { r } { \int _ { r e f i n e } = \sum _ { b \in \mathcal { B } \backslash \{ b _ { \mathrm { t a r } } \} } \mathcal { L } _ { r e f i n e } ^ { u s e r , b } + \mathcal { L } _ { r e f i n e } ^ { i t e m , b } . } \end{array}$ Target Preference Learning. We aggregate behavior-specific embeddings according to their relevance to the target behavior. Let $\boldsymbol { a } ( \cdot ) : \bar { \mathbb { R } } ^ { 2 d }  \mathbb { R }$ denote an attention function, implemented as a linear layer, over the concatenation of two embeddings.

$$
\alpha _ { u } ^ { b } = \frac { \exp \big ( a ( { \mathbf e _ { u } ^ { \mathrm { t a r } } } , { \mathbf e _ { u } ^ { b } } ) \big ) } { \sum _ { b ^ { \prime } \in \mathcal { B } } \exp \big ( a ( { \mathbf e _ { u } ^ { \mathrm { t a r } } } , { \mathbf e _ { u } ^ { b ^ { \prime } } } ) \big ) } , \quad { \mathbf e _ { u } ^ { \mathrm { a g g } } } = \sum _ { b \in \mathcal { B } } \alpha _ { u } ^ { b } \mathbf { e } _ { u } ^ { b } ,\tag{16}
$$

Table 1: Dataset statistics. Hidden (%) denotes the proportion <sub>o</sub>f hidd<sub>en</sub> t<sub>arge</sub>t it<sub>ems</sub> <sub>w</sub>ith<sub>ou</sub>t <sub>aux</sub>ili<sub>ary</sub> b<sub>e</sub>h<sub>av</sub>i<sub>ors.</sub>
<table><tr><td>Dataset</td><td>#Users</td><td>#Items</td><td>#Clicks</td><td>#Collects</td><td>#Carts</td><td>#Buys</td><td>Hidden (%)*</td></tr><tr><td>Tmall</td><td>41,738</td><td>11,953</td><td>1,813,498</td><td>221,514</td><td>1,996</td><td>255,586</td><td>16.38</td></tr><tr><td>Taobao</td><td>48,749</td><td>39,493</td><td>1,548,162</td><td></td><td>193,747</td><td>211,022</td><td>35.78</td></tr><tr><td>JData</td><td>93,334</td><td>24,624</td><td>1,681,430</td><td>45,613</td><td>49,891</td><td>321,883</td><td>18.55</td></tr></table>

Computed over test interactions, distinct from the training-set ratio in Fig 2.

Based on the aggregated representations, we compute the prefer ence score as $s _ { u i } = { \bf e } _ { u } ^ { \mathrm { a g g } \top } { \bf e } _ { i } ^ { \mathrm { a g g } }$ , and optimize the BPR loss on the target behavior: $\begin{array} { r } { \mathcal { L } _ { r a n k } = - \sum ( u , i ) \in \mathcal { E } _ { \mathrm { t a r } } \sum _ { j : ( u , j ) \notin \mathcal { E } _ { \mathrm { t a r } } } \log \sigma \big ( s _ { u i } - s _ { u j } \big ) } \end{array}$ Overall Learning Objective. The total loss of $\dot { f } _ { 1 }$ is as follows:

$$
\mathcal { L } _ { f _ { 1 } } = \mathcal { L } _ { r a n k } + \lambda _ { r e f i n e } \mathcal { L } _ { r e f i n e } ,\tag{17}
$$

where $\lambda _ { r e f i n e }$ is a loss-balancing hyperparameter.

## 4<sub>.</sub>4 U<sub>n</sub>ifi<sub>e</sub>d M<sub>o</sub>d<sub>u</sub>l<sub>ar</sub> L<sub>earn</sub>i<sub>ng an</sub>d I<sub>n</sub>f<sub>erence</sub>

According to our environment-conditioned design (Sec. 3.2), the final objective is expressed as a conditional expectation over modulewise objectives. Under this formulation, the auxiliary-observation propensity $\hat { p } _ { u i } = \hat { P } ( E _ { u i } = 1 \mid u , i )$ and its complement $1 - \hat { p } _ { u i }$ naturally serve as the assignment probabilities for the two modules:<sup>6</sup>

$$
\mathcal { L } _ { B O A R } = \mathbb { E } _ { ( u , i ) \in \mathcal { E } _ { \mathrm { t a r } } } \left[ \left( 1 - \hat { p } _ { u i } \right) \mathcal { L } _ { f _ { 0 } } ( u , i ) + \hat { p } _ { u i } \mathcal { L } _ { f _ { 1 } } ( u , i ) \right]\tag{18}
$$

Thus, each target interaction is softly assigned to the two modules according to its auxiliary-observation likelihood.

An alternative design is to use a gating network, jointly optimized with the model, to assign module weights from user–item embeddings. However, we found that propensity-based assignment is more efective, as it is grounded in the environment signal of auxiliary observation rather than learned as an unconstrained gate. We highlight that propensity estimation is well established, and its stability and robustness to noise have been extensively stud ied [27, 34, 35, 55].

Inference. Consistent with the training objective, the final prediction is obtained as:

$$
f ( u , i ) = \left( 1 - \hat { p } _ { u i } \right) f _ { 0 } ( u , i ) + \hat { p } _ { u i } f _ { 1 } ( u , i ) .\tag{19}
$$

This soft assignment allows each instance to leverage complementary signals from both modules. Alternatively, replacing $\hat { p } _ { u i }$ with the hard indicator $E _ { u i }$ yields a deterministic 0-1 assignment, where each instance is routed exclusively to one module and cannot benefit from the other. We validate this design choice in Sec. 5.2.3.

## 5 Ex<sub>p</sub>eriments

## 5.1 Ex<sub>p</sub>erimental Settin<sub>g</sub>s

Datasets. We use three widely used benchmark MBR datasets: (1) Tmall, from Alibaba platform and involving four behaviors (click, collect, cart, and buy); (2) Taobao, from Taobao and consisting of three behaviors (click, cart, and buy); and (3) JData, released by JD.com and containing four behaviors (click, collect, cart, and buy). For all datasets, we treat buy as the target behavior. Dataset statistics are summarized in Table 1.

Baselines. We compare BOAR with 13 baselines, covering both single- and multi-behavior recommendation methods. Specifically, we include two single-behavior models—MF-BPR [33] and Light-GCN (LGCN) [18]—and eleven multi-behavior methods: LGCN-G [18], MB-GMN [43], CIGF [17], CRGCN [45], BCIPM [47], MB-HGCN [46], COPF [49], MuLe [26], HGIB [51], MEMBER [23], and SHaRe [20]. Among these, HGIB and SHaRe are graph rewiringbased methods; HGIB focuses on pruning-based rewiring to filter noisy auxiliary interactions, while SHaRe applies similarity-based rewiring to MBR, performing pruning on auxiliary behavior graphs and densification on the target behavior graph, serving as a direct rewiring baseline.

Evaluation Protocol. We closely follow the evaluation protocols of prior work [17, 26, 43, 47, 49, 51]. We follow the widely adopted leave-one-out protocol, which holds out the most recent interaction of each user, together with all uninteracted items, as the test set. The second most recent interaction is reserved as the validation set. We evaluate performance using two ranking metrics: Hit Ratio (HR@�) and Normalized Discounted Cumulative Gain (NDCG@�). Following prior studies [26, 47, 49, 51], we set � = 10 in all experiments. We report the average performance of five independent runs, each of which uses diferent random seeds.

Performance Breakdown by Auxiliary Observability. For a more comprehensive evaluation, we report performance under three settings based on the observability of training auxiliary behaviors: general, observed, and unobserved. The general setting averages performance over all test instances. The observed setting reports the average over test instances with at least one auxiliary interaction (i.e., observed target items), whereas the unobserved setting does so over test instances without auxiliary interactions (i.e., hidden target items). This breakdown is intended to clearly reveal performance on hidden target items, which is essential for assessing generalization in real-world scenarios; strong performance in the unobserved setting indicates the model’s ability to discover novel target items beyond those that users have already interacted with. Hyperparameter Settings. Following [26, 45–47, 49, 51], we set the embedding dimension d to 64 and the batch size to 1024 for all compared methods to ensure a fair comparison; in BOAR, each module uses 32-dimensional embeddings, keeping the total dimension at 64. We use the Adam optimizer [24], where the learning rate was tuned in $\{ 5 \cdot 1 0 ^ { - 4 } , 1 0 ^ { - 4 } \}$ and the weight decay was tuned in $\{ 0 , 1 0 ^ { - 6 } \}$ . All hyperparameters are tuned via grid search on the validation set. For BOAR, the adversarial loss weight $\lambda _ { \mathrm { a d v } }$ was fixed to 0.01, while the remaining loss weights $\lambda _ { \mathrm { d e n s e } }$ and $\lambda _ { \mathrm { { r e f i n e } } }$ were tuned in the range of [0.1, 1.0]. The contrastive temperatures were tuned in the range of [0.1, 5.0]. The number of GNN layers was fixed to 2 for all graph encoders, and the LSH projection dimension $d ^ { \prime }$ was set to 32, and the candidate retrieval size � was tuned in {10, 20, 30, 40, 50}. The Gumbel-Softmax temperature was fixed to 0.2. For baseline methods, we closely follow the hyperparameter search ranges reported in the original papers.

## 5.2 Results and Anal<sub>y</sub>sis

5.2.1 Overall Performance Comparison. Table 2 reports the performance of all methods under three evaluation settings: General, Unobserved, and Observed. BOAR achieves the best performance in the general and unobserved settings across all datasets and metrics, with particularly large gains in the auxiliary-unobserved setting. In the general setting, BOAR consistently outperforms all baselines, achieving improvements of up to 7.82% in HR@10 and 15.00% in NDCG@10 over the best baseline. Notably, in the unobserved setting, BOAR achieves substantially larger margins, with maximum improvements of 44.2% in HR@10 and 36.7% in NDCG@10.

Table 2: Performance comparison. <sup>∗</sup> indicates statistical significance for � < 0.05 under t-test compared to the best baseline.
<table><tr><td rowspan="2">Dataset Types</td><td rowspan="2"></td><td rowspan="2">Metric</td><td colspan="2">Single-Behavior</td><td colspan="10">Multi-Behavior</td></tr><tr><td>MF-BPR LGCN</td><td></td><td>LGCN-G MB-GMN</td><td></td><td>CIGF</td><td>CRCGN BCIPM</td><td></td><td>MB-HGCN COPF</td><td>MuLe</td><td>HGIB</td><td></td><td>MEMBER SHaRe</td><td>BOAR</td></tr><tr><td rowspan="4">Tmall</td><td>General</td><td>HR@10 NDCG@10</td><td>0.0427 0.0210</td><td>0.0391 0.0201</td><td>0.1339 0.0681</td><td>0.0452 0.0621 0.0228 0.0320</td><td>0.0813 0.0428</td><td>0.1407 0.0765</td><td>0.1462 0.0778</td><td>0.1640 0.0883</td><td>0.2088 0.1158</td><td>0.2415 0.1274</td><td>0.3507 0.1730</td><td>0.2158 0.1133</td><td>0.3739* 0.1821*</td></tr><tr><td rowspan="3">Unobserved</td><td>HR@10</td><td>0.0662</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>0.0335</td><td>0.0692 0.0382</td><td>0.0835 0.0425</td><td>0.0298 0.0154</td><td>0.0474 0.0250</td><td>0.0327</td><td>0.0548</td><td>0.0422</td><td>0.0426 0.0549</td><td>0.0420</td><td>0.0948</td><td>0.0727</td><td>0.1367*</td></tr><tr><td>NDCG@10 HR@10</td><td>0.3996</td><td>0.3773</td><td></td><td></td><td>0.0174</td><td>0.0282</td><td>0.0223</td><td>0.0224</td><td>0.0289</td><td>0.0213</td><td>0.0594</td><td>0.0398</td><td>0.0812*</td></tr><tr><td rowspan="5">Taobao</td><td>Observed</td><td>NDCG@10</td><td>0.1891</td><td>0.4256 0.1765 0.2044</td><td>0.3740 0.1834</td><td>0.3617 0.1705</td><td>0.3841 0.1832</td><td>0.4075 0.1953</td><td>0.3762 0.1796</td><td>0.4048 0.1935</td><td>0.4557 0.2224</td><td>0.4527 0.2202</td><td>0.4554 0.2226</td><td>0.4438 0.2149</td><td>0.4587* 0.2232*</td></tr><tr><td rowspan="3">General</td><td>HR@10</td><td>0.0239</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.0178</td><td>0.1049</td><td>0.0498</td><td>0.0649</td><td>0.1174</td><td>0.1270</td><td>0.1639</td><td>0.1543 0.2121</td><td>0.2480</td><td>0.3183</td><td>0.1842</td><td>0.3432*</td></tr><tr><td>NDCG@10</td><td>0.0130 0.0095</td><td>0.0591</td><td>0.0198</td><td>0.0327</td><td>0.0655</td><td>0.0711</td><td>0.0950</td><td>0.0895</td><td>0.1346</td><td>0.1572</td><td>0.1640</td><td>0.1113</td><td>0.1886*</td></tr><tr><td>Unobserved</td><td>HR@10 NDCG@10</td><td>0.0167 0.0093</td><td>0.0124</td><td>0.0212</td><td>0.0182</td><td>0.0206</td><td>0.0177 0.0148</td><td></td><td>0.0153 0.0182</td><td>0.0248</td><td>0.0252</td><td>0.0221</td><td>0.0246</td><td>0.0352*</td></tr><tr><td rowspan="3">Observed</td><td>HR@10</td><td>0.4587</td><td>0.0068 0.4619</td><td>0.0105</td><td>0.0094</td><td>0.0102</td><td>0.0092</td><td>0.0077</td><td>0.0088</td><td>0.0075</td><td>0.0137</td><td>0.0147</td><td>0.0118</td><td>0.0140</td><td>0.0196*</td></tr><tr><td>NDCG@10</td><td>0.2252</td><td>0.2290</td><td>0.4938</td><td>0.4736</td><td>0.4849</td><td>0.4941</td><td>0.5024</td><td>0.4840</td><td>0.5343</td><td>0.5661</td><td>0.5797</td><td>0.5279</td><td>0.5435</td><td>0.5697</td></tr><tr><td>HR@10</td><td>0.3674</td><td>0.2779</td><td>0.2458</td><td>0.2318</td><td>0.2327</td><td>0.2530</td><td>0.2503</td><td>0.2446</td><td>0.2695</td><td>0.3050</td><td>0.3238</td><td>0.2738</td><td>0.2883</td><td>0.3061</td></tr><tr><td rowspan="4">Jdata</td><td>General</td><td>NDCG@10</td><td>0.2234</td><td>0.4163 0.1704 0.2411</td><td>0.1652</td><td>0.2815 0.3650 0.2272</td><td>0.4872</td><td>0.5252</td><td>0.5234</td><td>0.4497</td><td>0.5837</td><td>0.6502</td><td></td><td>0.6589 0.6378</td><td>0.6971* 0.4937*</td></tr><tr><td rowspan="3">Unobserved</td><td>HR@10</td><td>0.3610</td><td></td><td></td><td></td><td>0.2894</td><td>0.3158</td><td>0.3402</td><td>0.2723</td><td>0.4209</td><td>0.4667</td><td>0.4332</td><td>0.4584</td><td></td></tr><tr><td>NDCG@10</td><td></td><td>0.2679</td><td>0.3671</td><td>0.2295</td><td>0.2971</td><td>0.3671</td><td>0.3740</td><td>0.4164</td><td>0.2610 0.4749</td><td>0.4566</td><td></td><td>0.3947 0.4641</td><td>0.5438*</td></tr><tr><td>0.2002</td><td>0.1432</td><td>0.2103</td><td>0.1415</td><td>0.1854</td><td>0.2029</td><td>0.2272</td><td>0.2588</td><td>0.1743</td><td>0.2975</td><td>0.2769</td><td>0.2403</td><td>0.3018</td><td>0.3665*</td></tr><tr><td></td><td>Observed</td><td>HR@10 NDCG@10</td><td>0.7324</td><td>0.7189 0.7100</td><td>0.6937</td><td>0.7150</td><td>0.7452</td><td>0.7363</td><td>0.7234</td><td>0.7279</td><td>0.7693</td><td>0.7826</td><td>0.7383</td><td>0.7663</td><td>0.7757</td></tr><tr><td rowspan="2"></td><td></td><td>0.3751</td><td>0.4461</td><td>0.4372</td><td>0.4396</td><td>0.4427</td><td>0.4586</td><td>0.4458</td><td>0.4419</td><td>0.4648</td><td>0.4749</td><td>0.5340</td><td>0.4920</td><td>0.5235</td><td>0.5273</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/7dddb536d22659be44349a8b5289613a9e0fa904cf81e396dc42510c361ed636.jpg)

![](images/ca6507c40b091df742b1349b307829ca0edb5ad6c62b431b8f79214c07c34e17.jpg)  
Fi<sub>gure</sub> 4<sub>:</sub> P<sub>er</sub>f<sub>ormance</sub> d<sub>rop</sub> i<sub>n</sub> th<sub>e</sub> hidd<sub>en-pre</sub>f<sub>erence</sub> t<sub>es</sub>t<sub>.</sub>

In the observed setting, BOAR achieves the best or second-best performance across datasets; on datasets where HGIB leads, HGIB’s exclusive focus on pruning-based rewiring allows it to concentrate full capacity on the observed environment. This reflects a design trade-of: BOAR distributes model capacity across both environments, gaining substantially on unobserved items at a modest cost in the observed setting, and ultimately achieving the best overall performance in the general setting. Furthermore, SHaRe consistently underperforms BOAR, confirming that conventional similarity-based rewiring may amplify MBR imperfections rather than resolving them.

5.2.2 Stress-Testing of Graph Rewiring Modules. To verify that each rewiring module operates as intended, we design two controlled settings that directly examine the accuracy of densification and pruning: (i) hidden preference test, in which hidden target items are increased by removing a fraction � of auxiliary interactions overlapping with target-positive pairs, placing greater demand on the densification of $f _ { 0 } ;$ and (ii) noise injection test, in which auxiliary noise is increased by injecting false auxiliary interactions into a fraction � of target-negative pairs, placing greater demand on the pruning of $f _ { 1 } .$ . A method with accurate rewiring should exhibit minimal performance degradation as � increases, since correct densification recovers the removed signal and correct pruning suppresses the injected noise.<sup>7</sup>

![](images/ca7873c111483d4d43cc956fba571bc995a17ec8a8f77d82d0768dabddc79d70.jpg)

![](images/4786090011e9e5c851a34958351ace4c49d7fb4f6d3cfaa9d403c20661ca4ce6.jpg)  
Figure 5: Performance drop in the noise-injection test.

In the hidden-preference test (Figure 4), as � increases, all methods degrade, indicating poor generalization when target-positive items lack auxiliary evidence. BOAR shows the smallest performance drop across �, reflecting its ability to recover hidden preferences beyond observed auxiliary relations via debiased densification. In the noise-injection test (Figure 5), injecting noisy auxiliary interactions degrades all methods, highlighting the vulnerability of auxiliary-driven message passing. BOAR exhibits the least degradation as � increases, owing to its target-guided refinement that suppresses noise propagation in auxiliary signals. These observations collectively show the efectiveness of BOAR in robust target prediction beyond auxiliary behavior imperfections, consistent with Taobao results in Figure 1 (right).

T<sub>a</sub>bl<sub>e</sub> 3<sub>:</sub> Abl<sub>a</sub>ti<sub>on</sub> <sub>s</sub>t<sub>u</sub>d<sub>y</sub> <sub>o</sub>f BOAR <sub>on</sub> HR<sub>@</sub>10<sub>.</sub>
<table><tr><td>Setting</td><td colspan="3">1 Unobserved</td><td colspan="3">Observed</td></tr><tr><td>Datasets</td><td>Tmall</td><td>Taobao</td><td>JData</td><td>Tmall</td><td>Taobao</td><td>JData</td></tr><tr><td>BOAR</td><td>0.1367</td><td>0.0352</td><td>0.5438</td><td>0.4587</td><td>0.5697</td><td>0.7757</td></tr><tr><td>Ablation on densification</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { d e n s e } }$ </td><td>0.1081</td><td>0.0307</td><td>0.4768</td><td>0.4566</td><td>0.5607</td><td>0.7740</td></tr><tr><td>w/o  ${ \mathcal { L } } _ { \mathrm { a d v } }$ </td><td>0.1328</td><td>0.0335</td><td>0.5320</td><td>0.4549</td><td>0.5592</td><td>0.7753</td></tr><tr><td>w/o SNIPS</td><td>0.1214</td><td>0.0338</td><td>0.5352</td><td>0.4565</td><td>0.5645</td><td>0.7730</td></tr><tr><td>Ablation on refinement</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o TAGR</td><td>0.1321</td><td>0.0342</td><td>0.5391</td><td>0.4497</td><td>0.5623</td><td>0.7691</td></tr><tr><td>w/o  $\scriptstyle { \mathcal { L } } _ { \mathrm { r e f i n e } }$ </td><td>0.1329</td><td>0.0344</td><td>0.5256</td><td>0.4445</td><td>0.5445</td><td>0.7636</td></tr><tr><td>w/o Stop-gradient</td><td>0.1290</td><td>0.0334</td><td>0.5345</td><td>0.4395</td><td>0.5426</td><td>0.7587</td></tr><tr><td>Ablation on assignment</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Hard assignment</td><td>0.1315</td><td>0.0295</td><td>0.5408</td><td>0.4559</td><td>0.5607</td><td>0.7738</td></tr><tr><td>Learnable assignment</td><td>0.0609</td><td>0.0202</td><td>0.4698</td><td>0.4360</td><td>0.5411</td><td>0.7689</td></tr></table>

5.2.3 Ablation Study. We present a detailed ablation study on the densification $( f _ { 0 } )$ and refinement module $( f _ { 1 } )$ . Table 3 reports the ablation results under both settings, with $f _ { 0 }$ focusing on the unobserved setting and $f _ { 1 }$ on the observed setting. For $f _ { 0 } ,$ removing ${ \mathcal { L } } _ { \mathrm { d e n s e } }$ leads to a large performance drop, confirming that hiddenpreference mining is the primary source of gains in the unobserved environment. In addition, incorporating both ${ \mathcal { L } } _ { \mathrm { a d v } }$ and SNIPS yields the best performance, indicating a strong synergy between debiasing at the representation and prediction levels. For $f _ { 1 }$ , replacing the target-guided auxiliary graph refiner (TAGR) with the standard LGCN encoder without refinement consistently degrades performance. Moreover, $\mathcal { L } _ { \mathrm { r e f i n e } } ,$ , together with the stop-gradient operation that treats target embeddings as a fixed anchor, proves efective, supporting the validity of our design.

Lastly, we validate the modular integration design: replacing soft assignment with hard assignment degrades performance, as exclusive assignment prevents each instance from leveraging com plementary signals across both modules; substituting a learnable assignment<sup>8</sup> causes a far more severe drop, as it lacks a principled basis for module assignment, and thus jointly optimizing the assignment with the prediction objective leads to unstable training.

5.2.4 Hyperparameter Study. We provide an analysis to guide hyperparameter selection for BOAR. Here, we investigate three key hyperparameters, while the remaining ones $( \mathrm { e . g . , \lambda _ { a d v } ) }$ are fixed. Loss-balancing weights. Figure 6 presents the performance of $f _ { 0 }$ and $f _ { 1 }$ under varying $\lambda _ { \mathrm { d e n s e } }$ and $\lambda _ { \mathrm { r e f i n e } }$ , respectively. Overall, BOAR shows stable performance with respect to $\lambda _ { \mathrm { d e n s e } } ,$ while small values o $\mathrm { f } \lambda _ { \mathrm { r e f i n e } }$ are more efective across datasets. $\mathrm { A }$ suficiently large $\lambda _ { \mathrm { d e n s e } }$ is necessary to efectively exploit hidden preference supervision, whereas excessively increasing $\lambda _ { \mathrm { r e f i n e } }$ can degrade observed-item ranking due to over-regularization. In practice, moderate values consistently yield strong and stable performance.

Candidate Retrieval Size. In Figure 7, we analyze the sensitivity of BOAR to the candidate size � in angular LSH-based retrieval. Overall, BOAR shows limited sensitivity to �, with only minor performance variations across datasets, indicating that learnable edge selection efectively filters unreliable candidates even as � increases. When � is too small, the efect is limited, as potential hidden target-positive candidates may be missed.

![](images/a62ac5009a026c097624054524a2b9bd69a76ac4a23d67b7ac83deb113b9804b.jpg)

![](images/3d92870adbc4f2337c172f5ce44a9d0b6ff456c1477b4c02281009aaa2af2af5.jpg)

![](images/f6ce51bcb616685a936c74270e55c4828d4b8eb178997af960ee9c16ab1b4e17.jpg)

(a) Efect of $\lambda _ { \mathrm { d e n s e } }$ on $f _ { 0 }$ in the unobserved setting.  
![](images/321f5009a8de8c89f03f0b3c1909bccd82e9b444415d2455c4877ed8f06d28d0.jpg)

![](images/6fdbcbe59f09ad8f56ef766146d3e737f3b9b66effe1d28f1e5e685b4d4edcb5.jpg)

![](images/44c1204fb208b254949ab772218229f3ce931c8aaeaafea70a0e56d454f34fe4.jpg)  
(b) Efect of $\lambda _ { \mathrm { { r e f i n e } } }$ on $f _ { 1 }$ in the observed setting.

Fi<sub>g</sub>ure 6: Efects of loss-balancin<sub>g</sub> h<sub>yp</sub>er<sub>p</sub>arameters.  
![](images/7ebb5ee1de7c33070c01a7d998068dc6b5ce0e0fb736e393af9f823ecb3b3cee.jpg)

![](images/68e48c5b21fc848eb68edf8d5bd574acf340b4bde5c625da2db4a0be516379c3.jpg)

![](images/77a2eb715f8e0c42048da4f54220eebb46290c44b2e997ecd0124dd6da3e2a13.jpg)  
Fi<sub>gure</sub> 7<sub>:</sub> HR<sub>@</sub>10 <sub>un</sub>d<sub>er</sub> dif<sub>eren</sub>t <sub>can</sub>did<sub>a</sub>t<sub>e</sub> <sub>re</sub>t<sub>r</sub>i<sub>eva</sub>l <sub>s</sub>i<sub>zes</sub> <sub>�.</sub>

Table 4: Eficienc<sub>y</sub> anal<sub>y</sub>sis: com<sub>p</sub>arison of <sub>p</sub>er-e<sub>p</sub>och trainin<sub>g</sub> time (sec), inference time (sec), and model size (in millions). Trainin<sub>g</sub> time denotes the avera<sub>g</sub>e time <sub>p</sub>er e<sub>p</sub>och<sub>,</sub> and infer-<sub>ence</sub> ti<sub>me</sub> d<sub>eno</sub>t<sub>es</sub> th<sub>e</sub> ti<sub>me</sub> t<sub>o genera</sub>t<sub>e recommen</sub>d<sub>a</sub>ti<sub>ons</sub> f<sub>or</sub> <sub>a</sub>ll <sub>users.</sub>
<table><tr><td>Dataset</td><td>Metric</td><td>MuLe</td><td>HGIB</td><td>MEMBER</td><td>BOAR</td></tr><tr><td rowspan="3">Tmall</td><td>Train time / epoch</td><td>81.47s</td><td>86.46s</td><td>164.85s</td><td>57.93s</td></tr><tr><td>Inference time</td><td>12.52s</td><td>11.24s</td><td>12.82s</td><td>8.05s</td></tr><tr><td>#Parameters</td><td>3.44M</td><td>3.44M</td><td>3.44M</td><td>3.44M</td></tr><tr><td rowspan="3">Taobao</td><td>Train time / epoch</td><td>70.22s</td><td>64.43s</td><td>116.97s</td><td>49.86s</td></tr><tr><td>Inference time</td><td>21.69s</td><td>19.66s</td><td>22.84s</td><td>16.99s</td></tr><tr><td>#Parameters</td><td>5.65M</td><td>5.66M</td><td>5.65M</td><td>5.65M</td></tr><tr><td rowspan="3">JData</td><td>Train time / epoch</td><td>100.12s</td><td>90.83s</td><td>178.49s</td><td>95.86s</td></tr><tr><td>Inference time</td><td>5.92s</td><td>5.69s</td><td>6.22s</td><td>4.68s</td></tr><tr><td>#Parameters</td><td>7.55M</td><td>7.56M</td><td>7.55M</td><td>7.56M</td></tr></table>

## 5.3 Com<sub>p</sub>lexit<sub>y</sub> Anal<sub>y</sub>sis

Table 4 compares the per-epoch training time and inference time, as well as the model size, of BOAR with state-of-the-art MBR methods [23, 26, 51]. All experiments are conducted using PyTorch with CUDA on an RTX 5000 Ada GPU and an Intel Xeon Gold 6338 CPU.

BOAR introduces only a small number of additional parameters, mainly a few lightweight linear layers, and since the densification process is applied only during training, it achieves the fastest inference latency among all compared methods. BOAR also achieves the fastest per-epoch training time on Tmall and Taobao, and remains comparable to HGIB on JData. Overall, BOAR achieves strong performance with eficiency comparable to state-of-the-art methods, supporting its practical applicability.

## 6 C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

In this work, we first highlight two fundamental challenges of auxiliary behaviors: missing auxiliary signals and unreliable auxiliary signals. As a solution, we propose BOAR, an environmentconditioned framework that addresses missing and unreliable aux iliary signals through two complementary modules conditioned on auxiliary observability. Extensive experiments show that BOAR achieves particularly strong gains on hidden target items without auxiliary behaviors, validating its efectiveness in capturing hidden preferences beyond observed auxiliary relations. Future work may explore streaming settings with continuously evolving user behaviors and interactions.

## A A<sub>pp</sub>endix

## A.1 Su<sub>pp</sub>lementar<sub>y</sub> Proofs and Method Details

A.1.1 On the Validity of Debiasing in BOAR. We provide a theoretical interpretation of two debiasing techniques underlying BOAR: Inverse Propensity Scoring (IPS) [35] and adversarial learning [2].

IPS weighted BPR Loss (Eq. (10)). Let $O _ { u i } \in \{ 0 , 1 \}$ denote whether the target-behavior pair (�, �) is observable in the training data (e.g., due to platform exposure), with propensity $p _ { \mathrm { o b s } } ( u , i ) = \operatorname* { P r } ( O _ { u i } =$ $1 \mid u , i )$ . We assume target-positive signals are observed only when $O _ { u i } \ = \ 1$ . For an observed positive $( u , i ) \ \in \ E _ { \mathrm { t a r } }$ , we sample $j \sim$ $q ( j \mid u )$ (independent of $O _ { u i } )$ and use $\ell _ { \theta } ( u , i , j ) = - \log \sigma ( s _ { u i } - s _ { u j } )$ Although $O _ { u i } = 1$ for all logged pairs in practice, $O _ { u i }$ is treated as a Bernoulli random variable in the data-generating process, enabling importance weighting.

Define the ideal risk $\mathcal { L } _ { \mathrm { i d e a l } } = \mathbb { E } [ \ell _ { \theta } ( u , i , j ) ]$ and the IPS-weighted risk $\begin{array} { r } { \mathcal { L } _ { \mathrm { I P S } } = \mathbb { E } \bigg [ \frac { O _ { u i } } { \hat { p } _ { \mathrm { o b s } } ( u , i ) } \ell _ { \theta } ( u , i , j ) \bigg ] } \end{array}$

Theorem A.1 (Unbiasedness under correct propensities [35]). Assume $\dot { p } _ { \mathrm { o b s } } ( u , i ) > 0 . \ I f \hat { p } _ { \mathrm { o b s } } ( u , i ) = \dot { p } _ { \mathrm { o b s } } ( u , i )$ almost surely, then Bias $\left[ \mathcal { L } _ { \mathrm { I P S } } \right] = \left| \mathcal { L } _ { \mathrm { I P S } } - \mathcal { L } _ { \mathrm { i d e a l } } \right| = 0$

Proof. Since $\ell _ { \theta } ( u , i , j )$ is deterministic given $( u , i , j )$

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { I P S } } = \mathbb { E } _ { u , i , j } \left[ \ell _ { \theta } ( u , i , j ) \cdot \mathbb { E } \left[ \left. \frac { O _ { u i } } { \hat { p } _ { \mathrm { o b s } } ( u , i ) } \right| u , i \right] \right] . } \end{array}
$$

Since $\mathbb { E } [ O _ { u i } \mid u , i ] = p _ { \mathrm { o b s } } ( u , i ) = \hat { p } _ { \mathrm { o b s } } ( u , i )$ , the inner expectation is 1, yielding $\mathcal { L } _ { \mathrm { I P S } } = \mathcal { L } _ { \mathrm { i d e a l } }$ □

Connection to $E q .$ (10). Eq. (10) approximates $\mathcal { L } _ { \mathrm { I P S } }$ via Monte-Carlo over logged positives. Following [27, 35, 55], we estimate the pair-level propensity $ { p _ { \mathrm { o b s } } } ( u , i )$ via a lightweight binary classifier $\hat { p } _ { u i } ( \mathrm { A p p e n d i x } \ : \mathrm { A } . 1 . 2 ) ,$ , trained to predict whether a user–item pair is auxiliary-observed. We set $\omega _ { u i } \propto 1 / \hat { p } _ { u i } \ : ( \mathrm { S N I P S } )$ , up-weighting pairs with low auxiliary-observation propensity and down-weighting those with high propensity to correct observation bias.

Auxiliary Popularity Adversarial Learning (Eq. (5)). Let $\mathcal { D } _ { \mathrm { a b o v e } }$ and $\mathcal { D } _ { \mathrm { b e l o w } }$ denote the item distributions over items whose auxiliary interaction count exceeds $( y _ { i } = 1 )$ or falls below $( y _ { i } = 0 )$ the median (Sec. 4.2.1). Let $z = g _ { \psi } ( i ) \in Z$ be the learned item representation, with induced distributions $P _ { \psi } : = g _ { \psi } \# \mathcal { D } _ { \mathrm { a b o v e } }$ and $\begin{array} { r } { Q _ { \psi } : = g _ { \psi } \# \mathcal { D } _ { \mathrm { b e l o w } } } \end{array}$ Let $\mathcal { H } _ { d }$ be binary classifiers $\dot { h } : \mathcal { Z } \dot {  } \{ 0 , 1 \}$ where $h ( z ) { = } 1$ predicts above-median and $h ( z ) { = } 0$ predicts below-median.

Lemma A.2 (H-divergence [2]).

$$
d _ { { \mathcal { H } } _ { d } } ( P _ { \psi } , Q _ { \psi } ) : = 2 \operatorname* { s u p } _ { h \in { \mathcal { H } } _ { d } } \left| \operatorname* { P r } _ { z \sim P _ { \psi } } \left[ h ( z ) = 1 \right] - \operatorname* { P r } _ { z \sim Q _ { \psi } } \left[ h ( z ) = 1 \right] \right| .
$$

With $\varepsilon _ { d } ^ { * } : = \operatorname* { m i n } _ { h } \varepsilon _ { d } ( h )$ where $\begin{array} { r } { \varepsilon _ { d } ( h ) : = \frac { 1 } { 2 } \operatorname* { P r } _ { P _ { \psi } } [ h { = } 0 ] + \frac { 1 } { 2 } \operatorname* { P r } _ { Q _ { \psi } } [ h { = } 1 ] . } \end{array}$ and $\mathcal { H } _ { d }$ complement-closed, $d _ { \mathcal { H } _ { d } } ( P _ { \psi } , Q _ { \psi } ) = 2 ( 1 { - } 2 \varepsilon _ { d } ^ { * } )$ . Hence $\textstyle \varepsilon _ { d } ^ { * } \to \frac { 1 } { 2 }$ implies $d _ { \mathcal { H } _ { d } } ( P _ { \psi } , Q _ { \psi } )  0$

Proof. For any $h \in \mathcal { H } _ { d } , \mathrm { P r } _ { P _ { \iota / } } [ h { = } 1 ] - \mathrm { P r } _ { Q _ { \iota / } } [ h { = } 1 ] = 1 - 2 \varepsilon _ { d } ( h )$ Since $\mathcal { H } _ { d }$ is complement-closed, $\begin{array} { r } { \operatorname* { s u p } _ { h } | \operatorname* { P r } _ { P _ { \psi } } [ \dot { h } { = } 1 ] - \operatorname* { P r } _ { Q _ { \psi } } [ h { = } 1 ] | = } \end{array}$ $1 - 2 \varepsilon _ { d } ^ { * }$ . Multiplying by 2 yields the claim. □

Corollary A.3 (Why both GRL and IPS are reqired). Ifthe GRL-based adversarial objective drives $\begin{array} { r } { \varepsilon _ { d } ^ { * } \approx \frac { 1 } { 2 } } \end{array}$ , then by Lemma A.2, $d _ { \mathcal { H } _ { d } } ( P _ { \psi } , Q _ { \psi } ) \approx 0 ,$ meaning $g _ { \psi }$ suppresses popularity-driven observation bias from item representations in $\mathbf { E } ^ { \mathrm { g l o } }$ . This encourages LSH-based candidate retrieval (Sec. 4.2.2) to surface preference-aligned hidden items rather than popularity-similar ones; these candidates are incorporated into $\ddot { \mathbf { A } } _ { \mathrm { t a r } }$ and propagated to $\mathbf { E } ^ { \mathrm { t a r } }$ via LGCN, supporting reliable recovery ofhidden preferences at the representation level.

However, even with suppressed popularity-driven observation bias in representations, optimizing the ranking objective on loggedpositives can still be dominated by frequently auxiliary-observed items, since they appear more often in $\mathcal { E } _ { \mathrm { t a r } }$ regardless of their representation. The propensity-weighted objective in Theorem A.1 corrects this predictionlevel bias, while SNIPS provides variance-stabilized approximations. Therefore, removing either component leaves (i) popularity-driven observation bias in $\mathbf { E } ^ { \mathrm { g l o } }$ , degrading candidate quality for hidden items (without GRL), or (ii) observation bias in the ranking objective towardfrequently auxiliary-observed items (without IPS), both ofwhich hinder reliable recovery ofhidden preferences.

A.1.2 Propensity Computation. Following [27, 35, 55], we estimate $\hat { p } _ { u i } = \sigma ( \mathbf { e } _ { u } ^ { \mathrm { { a u x } } ^ { \top } } \mathbf { e } _ { i } ^ { \mathrm { { a u x } } } )$ via a lightweight binary classifier, where $\mathbf { E } ^ { \mathrm { a u x } } = \mathrm { L G C N } ( \mathbf { A } _ { \mathrm { a u x } } , \dot { \mathbf { E } } )$ and $\mathbf { A } _ { \mathrm { a u x } }$ is constructed from ${ { \mathcal { E } } _ { \mathrm { a u x } } } .$ The classifier is trained via binary cross-entropy with $E _ { u i }$ (Sec. 3.2) as supervision labels, where D is constructed by sampling positives from $\mathcal { E } _ { \mathrm { a u x } }$ and, for each user �, randomly sampling auxiliary-unobserved items as negatives at a 1:1 ratio:

$$
\mathcal { L } _ { \mathrm { B C E } } = - \sum _ { ( u , i ) \in \mathcal { D } } \left[ E _ { u i } \log \hat { p } _ { u i } + ( 1 - E _ { u i } ) \log ( 1 - \hat { p } _ { u i } ) \right] .\tag{20}
$$

We then apply self-normalized inverse propensity scoring:

$$
\omega _ { u i } = \frac { \hat { \rho } _ { u i } ^ { - 1 } } { \sum _ { ( u ^ { \prime } , i ^ { \prime } ) \in { \mathcal E } _ { \mathrm { t a r } } } \hat { p } _ { u ^ { \prime } i ^ { \prime } } ^ { - 1 } } ,\tag{21}
$$

where $\hat { p } _ { u i }$ is clipped to max $( \hat { p } _ { u i } , 1 0 ^ { - 5 } )$ for numerical stability.

## A<sub>c</sub>k<sub>now</sub>l<sub>e</sub>d<sub>gmen</sub>t<sub>s</sub>

This work was supported by a Korea University Grant, IITP grants funded by the Korea government (MSIT): the ICT Creative Consilience Program (IITP-2026-RS-2020-II201819) and the Artificial Intelligence Star Fellowship (IITP-2026-RS-2025-02304828). This work was also supported by NRF grants funded by the MSIT (RS-2026-25486220) and by the Basic Science Research Program funded by the Ministry of Education (NRF-2021R1A6A1A03045425).

## G<sub>e</sub>nAI U<sub>sage</sub> Di<sub>sc</sub>l<sub>osu</sub>r<sub>e</sub>

The authors employed LLM tools in a limited capacity, namely polishing grammar in the written manuscript and assisting with code debugging. All such outputs were verified and revised by the authors. The core research process, including problem formulation and model design, was conducted entirely without LLM assistance.

## R<sub>e</sub>f<sub>erences</sub>

[1] Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn, and Ludwig Schmidt. 2015. Practical and Optimal LSH for Angular Distance. In NeurIPS.

[2] Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira, and Jennifer Wortman Vaughan. 2010. A theory of learning from diferent domains. Machine Learning 79 (2010), 151–175.

[3] Wendong Bi, Lun Du, Qiang Fu, Yanlin Wang, Shi Han, and Dongmei Zhang. 2024. Make Heterophily Graphs Better Fit GNN: A Graph Rewiring Approach. IEEE Transactions on Knowledge and Data Engineering (2024).

[4] Chong Chen, Weizhi Ma, Min Zhang, Zhaowei Wang, Xiuqiang He, Chenyang Wang, Yiqun Liu, and Shaoping Ma. 2021. Graph Heterogeneous Multi-Relationa Recommendation. In AAAI.

[5] Jiawei Chen, Hande Dong, Xiang Wang, Fuli Feng, Meng Wang, and Xiangnan He. 2023. Bias and Debias in Recommender System: A Survey and Future Directions. ACM Trans. Inf. Syst. (2023).

[6] Wenjie Chen, Yi Zhang, Honghao Li, Lei Sang, and Yiwen Zhang. 2025. Dual-Domain Collaborative Denoising for Social Recommendation. IEEE Trans. Comput. Soc. Syst. 12 (2025), 2736–2751.

[7] Zhiyong Cheng, Sai Han, Fan Liu, Lei Zhu, Zan Gao, and Yuxin Peng. 2023. Multi-Behavior Recommendation with Cascading Graph Convolution Networks. In WWW.

[8] Enyan Dai, Wei Jin, Hui Liu, and Suhang Wang. 2022. Towards Robust Graph Neural Networks for Noisy Graphs with Sparse Labels. In WSDM.

[9] Kaize Ding, Zhe Xu, Hanghang Tong, and Huan Liu. 2022. Data augmentation for deep graph learning: A survey. ACM SIGKDD Explorations Newsletter 24 (2022), 61–77.

[10] Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario March, and Victor Lempitsky. 2016. Domain-adversarial training of neural networks. Journal of Machine Learning Research 17, 59 (2016), 1–35.

[11] Chen Gao, Xiangnan He, Dahua Gan, Xiangning Chen, Fuli Feng, Yong Li, Tat-Seng Chua, and Depeng Jin. 2019. Neural Multi-task Recommendation from Multi-behavior Data. In ICDE.

[12] Xinyi Gao, Tong Chen, Yilong Zang, Wentao Zhang, Quoc Viet Hung Nguyen, Kai Zheng, and Hongzhi Yin. 2024. Graph condensation for inductive node representation learning. In ICDE.

[13] Shuwei Gong, Yuting Liu, Yizhou Dang, Guibing Guo, Jianzhe Zhao, and Xingwei Wang. 2025. Multiple Purchase Chains with Negative Transfer Elimination for Multi-Behavior Recommendation. In AAAI

[14] Shuyun Gu, Xiao Wang, Chuan Shi, and Ding Xiao. 2022. Self-supervised Graph Neural Networks for Multi-behavior Recommendation. In IJCAI.

[15] Jiayan Guo, Lun Du, Wendong Bi, Qiang Fu, Xiaojun Ma, Xu Chen, Shi Han, Dongmei Zhang, and Yan Zhang. 2023. Homophily-oriented Heterogeneous Graph Rewiring. In WWW.

[16] Long Guo, Lifeng Hua, Rongfei Jia, Binqiang Zhao, Xiaobo Wang, and Bin Cui. 2019. Buying or Browsing?: Predicting Real-time Purchasing Intent us ing Attention-based Deep Network with Multiple Behavior. In SIGKDD.

[17] Wei Guo, Chang Meng, Enming Yuan, Zhicheng He, Huifeng Guo, Yingxue Zhang, Bo Chen, Yaochen Hu, Ruiming Tang, Xiu Li, and Rui Zhang. 2023. Compressed Interaction Graph based Framework for Multi-behavior Recommendation. In WWW.

[18] Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang. 2020. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. In SIGIR.

[19] Eric Jang, Shixiang Gu, and Ben Poole. 2017. Categorical Reparameterization with Gumbel-Softmax. In ICLR.

[20] Wei Jiang, Xinyi Gao, Guandong Xu, Tong Chen, and Hongzhi Yin. 2024. Chal lenging Low Homophily in Social Recommendation. In WWW.

[21] Bowen Jin, Chen Gao, Xiangnan He, Depeng Jin, and Yong Li. 2020. Multi behavior Recommendation with Graph Convolutional Networks. In SIGIR.

[22] Kyungho Kim, Sunwoo Kim, Geon Lee, Jinhong Jung, and Kijung Shin. 2025. Multi-behavior Recommender Systems: A Survey. In PAKDD.

[23] Kyungho Kim, Sunwoo Kim, Geon Lee, and Kijung Shin. 2025. A Self-Supervised Mixture-of-Experts Framework for Multi-behavior Recommendation. In CIKM.

[24] Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In ICLR.

[25] Nikita Kitaev, Łukasz Kaiser, and Anselm Levskaya. 2020. Reformer: The Eficient Transformer. In ICLR

[26] Seunghan Lee, Geonwoo Ko, Hyun-Je Song, and Jinhong Jung. 2024. MuLe: Multi-Grained Graph Learning for Multi-Behavior Recommendation. In CIKM.

[27] Haoxuan Li, Chunyuan Zheng, and Peng Wu. 2023. StableDR: Stabilized Doubly Robust Learning for Recommendation on Data Missing Not at Random. In ICLR.

[28] Jonas Linkerhägner, Cheng Shi, and Ivan Dokmanić. 2025. Joint graph rewiring and feature denoising via spectral resonance. In ICLR.

[29] Babak Loni, Roberto Pagano, Martha Larson, and Alan Hanjalic. 2016. Bayesian Personalized Ranking with Multi-Channel User Feedback. In RecSys.

[30] Chang Meng, Chenhao Zhai, Yu Yang, Hengyu Zhang, and Xiu Li. 2023. Parallel Knowledge Enhancement based Framework for Multi-behavior Recommendation. In CIKM.

[31] Chang Meng, Hengyu Zhang, Wei Guo, Huifeng Guo, Haotian Liu, Yingxue Zhang, Hongkun Zheng, Ruiming Tang, Xiu Li, and Rui Zhang. 2023. Hierarchical Projection Enhanced Multi-behavior Recommendation. In SIGKDD.

[32] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).

[34] Paul R Rosenbaum and Donald B Rubin. 1983. The central role of the propensity score in observational studies for causal efects. Biometrika 70 (1983), 41–55.

[35] Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, and Thorsten Joachims. 2016. Recommendations as treatments: Debiasing learning and evaluation. In ICML.

[36] Changxin Tian, Yuexiang Xie, Yaliang Li, Nan Yang, and Wayne Xin Zhao. 2022. Learning to Denoise Unreliable Interactions for Graph Collaborative Filtering. In SIGIR.

[37] Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua. 2019. Neural Graph Collaborative Filtering. In SIGIR.

[38] Zhonghao Wang, Danyu Sun, Sheng Zhou, Haobo Wang, Jiapei Fan, Longtao Huang, and Jiajun Bu. 2024. NoisyGL: A Comprehensive Benchmark for Graph Neural Networks under Label Noise. In NeurIPS.

[39] Wei Wei, Chao Huang, Lianghao Xia, Yong Xu, Jiashu Zhao, and Dawei Yin. 2022. Contrastive Meta Learning with Behavior Multiplicity for Recommendation. In WSDM.

[40] Chuhan Wu, Fangzhao Wu, Xiting Wang, Yongfeng Huang, and Xing Xie. 2021. Fairness-aware News Recommendation with Decomposed Adversarial Learning. In AAAI.

[41] Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Bo Zhang, and Liefeng Bo. 2020. Multiplex Behavioral Relation Learning for Recommendation via Memory Augmented Transformer Network. In SIGIR.

[42] Lianghao Xia, Chao Huang, Yong Xu, Peng Dai, Xiyue Zhang, Hongsheng Yang, Jian Pei, and Liefeng Bo. 2021. Knowledge-Enhanced Hierarchical Graph Transformer Network for Multi-Behavior Recommendation. In AAAI

[43] Lianghao Xia, Yong Xu, Chao Huang, Peng Dai, and Liefeng Bo. 2021. Graph Meta Network for Multi-Behavior Recommendation. In SIGIR.

[44] Jingcao Xu, Chaokun Wang, Cheng Wu, Yang Song, Kai Zheng, Xiaowei Wang, Changping Wang, Guorui Zhou, and Kun Gai. 2023. Multi-behavior Selfsupervised Learning for Recommendation. In SIGIR.

[45] Mingshi Yan, Zhiyong Cheng, Chen Gao, Jing Sun, Fan Liu, Fuming Sun, and Haojie Li. 2023. Cascading Residual Graph Convolutional Network for Multi Behavior Recommendation. ACM Trans. Inf. Syst. (2023).

[46] Mingshi Yan, Zhiyong Cheng, Jing Sun, Fuming Sun, and Yuxin Peng. 2023. MB-HGCN: A hierarchical graph convolutional network for multi-behavior rec ommendation. arXiv preprint arXiv:2306.10679 (2023).

[47] Mingshi Yan, Fan Liu, Jing Sun, Fuming Sun, Zhiyong Cheng, and Yahong Han. 2024. Behavior-contextualized item preference modeling for multi-behavior recommendation. In SIGIR

[48] Yonghui Yang, Le Wu, Richang Hong, Kun Zhang, and Meng Wang. 2021. Enhanced Graph Learning for Collaborative Filtering via Mutual Information Maxi mization. In SIGIR.

[49] Chenhao Zhai, Chang Meng, Yu Yang, Kexin Zhang, Xuhao Zhao, and Xiu Li. 2025. Combinatorial Optimization Perspective based Framework for Multi-behavior Recommendation. In SIGKDD.

[50] An Zhang, Wenchang Ma, Pengbo Wei, Leheng Sheng, and Xiang Wang. 2024. General Debiasing for Graph-based Collaborative Filtering via Adversarial Graph Dropout. In WWW.

[51] Hengyu Zhang, Chunxu Shen, Xiangguo Sun, Jie Tan, Yanchao Tan, Yu Rong, Hong Cheng, and Lingling Yi. 2025. Hierarchical Graph Information Bottleneck for Multi-Behavior Recommendation. In RecSys.

[52] Weifeng Zhang, Jingwen Mao, Yi Cao, and Congfu Xu. 2020. Multiplex Graph Neural Networks for Multi-behavior Recommendation. In CIKM.

[53] Zhe Zhao, Zhiyuan Cheng, Lichan Hong, and Ed H. Chi. 2015. Improving User Topic Interest Profiles by Behavior Factorization. In WWW.

[54] Huachi Zhou, Hao Chen, Junnan Dong, Daochen Zha, Chuang Zhou, and Xiao Huang. 2023. Adaptive Popularity Debiasing Aggregator for Graph Collaborative Filtering. In SIGIR.

[55] Kuiyu Zhu, Tao Qin, Pinghui Wang, and Xin Wang. 2025. Adversarial Propensity Weighting for Debiasing in Collaborative Filtering. In IJCAI.