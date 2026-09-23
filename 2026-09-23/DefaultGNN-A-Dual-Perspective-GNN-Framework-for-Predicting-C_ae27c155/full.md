# DefaultGNN: A Dual-Perspective GNN Framework for Predicting Corporate Default from Buyer-Seller Transaction Networks

Junghoon Kim   
KAIST   
Daejeon, Republic of Korea   
jhkim611@kaist.ac.kr   
Hyunsung Kim   
KAIST   
Daejeon, Republic of Korea   
hyunsung.kim@kaist.ac.kr   
Jihun Lee   
Techfin Ratings   
Seoul, Republic of Korea   
jihuny@techfinratings.com   
Seungyoon Choi   
KAIST   
Daejeon, Republic of Korea   
csyoon08@kaist.ac.kr   
YongGu Ji   
Douzone   
Seoul, Republic of Korea   
todcode@douzone.com

KyoungYong Park Techfin Ratings Seoul, Republic of Korea hess\_kpark@techfinratings.com

Chanyoung Park   
KAIST   
Daejeon, Republic of Korea   
cy.park@kaist.ac.kr

## Abstract

Corporate default prediction is a core problem in financial risk management, yet traditional credit models rely heavily on financial statements that are often sparse or unavailable for many firms. Corporate transaction networks ofer a complementary view of real economic activity, but how risk propagates through buyer–seller relationships remains underexplored. We conduct a large-scale empirical study using real-world electronic tax-invoice data spanning six years that links transaction histories with default events, revealing that transaction-driven risk is both role-dependent (buyer or seller) and scale-dependent. Based on these findings, we construct multiplex buyer-view and seller-view transaction networks and propose DefaultGNN, a dual-perspective graph neural network-based framework for corporate default prediction. DefaultGNN integrates both views to model how risk flows through transactional relationships, achieving strong improvements over both attribute-based and graph-based baselines, especially for firms with limited intrinsic risk signals. We further provide interpretable network-based explanations by visualizing how distressed trading partners contribute to default risk. In collaboration with a licensed credit rating agency, we validate that DefaultGNN’s predictions complement existing credit scoring models, improving approval rates by 7–11%p without increasing default risk among approved firms. The source code can be found at https://github.com/jhkim611/DefaultGNN.

## CCS Concepts

• Applied computing → Economics;

• Computing methodologies → Artificial intelligence.

## Keywords

Corporate Default Prediction, Graph Neural Networks, Transaction Networks, Financial Risk Propagation

ACM Reference Format:   
Junghoon Kim, Hyunsung Kim, Seungyoon Choi, KyoungYong Park, Jihun Lee, YongGu Ji, and Chanyoung Park. 2026. DefaultGNN: A Dual-Perspective GNN Framework for Predicting Corporate Default from Buyer-Seller Transaction Networks. In Proceedings ofthe 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 07–11, 2026, Rome, Italy. ACM, New York, NY, USA, 8 pages. https: //doi.org/10.1145/3799682.3840153

## 1 Introduction

Corporate credit evaluation (or default prediction), the task of determining whether a firm will default on its financial obligations in the future, plays a crucial role in financial decision-making, including lending decisions and inter-corporate trade activities. Traditional corporate credit scoring models primarily rely on financial statements and historical default data [4, 24], typically using statistical methods like logistic regression [1, 18, 27, 31, 35]. While these ap proaches are interpretable have shown stable performance, they are inherently constrained by the availability and timeliness of financial data [12, 14].

This limitation is particularly pronounced for small and medium enterprises (SMEs) and sole proprietors, who typically face infrequent financial reporting cycles and limited disclosure requirements. As a result, financial statements for these firms are often missing, outdated or incomplete, leading to information asymmetry between financial institutions and the firms being evaluated. In such settings, traditional financial statement-driven credit scoring models tend to overestimate risk for SMEs<sup>1</sup> [15, 21, 32], motivating growing interest in machine learning-based models that incorporate non-financial data in credit assessment systems.

Among such non-financial data, transactional relationships between businesses have garnered attention as they directly reflect the actual economic activity of firms. Businesses do not operate in isolation; they are interconnected through buyer-seller relationships. The default of one partner can cascade through a transaction network, impacting the financial stability of adjacent firms. Prior research has shown that credit risk contagion, i.e., the transfer of financial risk through transactions, can significantly influence default probability, with a meaningful correlation between a firm’s credit risk and that of its trading partners [5, 7, 9, 16, 20].

Despite this, large-scale studies that leverage corporate transaction data for default prediction remain limited, as traditional statistical or linear models cannot efectively incorporate the relational and multi-layered structure ofinterfirm transaction networks. Recent advances in graph neural networks (GNNs) [23, 37] have enabled relational modeling of financial systems, and have been applied to settings such as guarantee networks and supply chains for credit risk prediction [11, 25, 38]. However, existing GNN-based studies have largely underutilized transactional information. In particular, most models either focus on non-transactional relations (e.g., loans or guarantees) or simplify transactions and their accompanying risks to unidirectional and binary links, ignoring their bidirectional risk exposure and heterogeneous scale.

Specifically, in real-world transactions, risk would propagate in both directions: sellers face exposure to late or failed payments, while buyers are vulnerable to delivery failures that can disrupt downstream operations and other business relationships. Moreover, larger transactions would likely induce higher risk than smaller ones. Capturing these efects requires a more detailed analysis of how buyer–seller relationships and transaction magnitude shape default risk.

To address this gap, we conduct a large-scale empirical analysis of real-world electronic tax-invoice data, linking transaction histories with default events of trading partners (see Section 3). We examine how default risk depends on who trades with whom, in which role (buyer or seller), and at what scale, and use these findings to construct multiplex buyer- and seller-view transaction networks. These insights are then incorporated into DefaultGNN, a dual perspective GNN framework for predicting corporate default from buyer-seller transaction networks. DefaultGNN integrates both views, i.e., transactional patterns learned from separate GNN layers, to predict corporate default, enabling more accurate risk assessment especially for firms with limited intrinsic financial risk signals.

Overall, our contributions can be summarized as follows:

• Large-scale empirical study of transaction-driven default risk. We analyze real-world electronic tax-invoice data spanning six years that link buyer-seller transactions with default events, providing direct evidence for how risk propagates through corporate transaction networks.

• Role- and scale-aware modeling of transaction-based risk. We show that default risk depends on buyer/seller roles and transaction magnitude, and construct multiplex transaction networks that explicitly capture these nuances.

• Efectiveness. Our proposed DefaultGNN integrates buyer-view and seller-view embeddings to model how risk is transmitted through transactional relationships, enabling more accurate default prediction, especially for firms with limited financial data.

• Interpretable network-based explanations of default risk. DefaultGNN enables visualization of a firm’s transaction neighborhood, revealing how defaulted or distressed trading partners contribute to the predicted risk of the target firm, providing actionable insights for financial decision-making.

• Practical validation with a credit rating agency. In collaboration with Techfin Ratings, we validate that DefaultGNN’s predictions complement an existing credit scoring model, improving approval rates without increasing default risk.

## 2 Related Works

## 2.1 Modeling for Corporate Default Prediction

Corporate default prediction has traditionally relied on financial statement-based models, where bankruptcy risk is estimated using accounting ratios and firm-level attributes. Classical approaches such as the Altman Z-score [1] and the Ohlson O-score [27] remain widely used benchmarks due to their interpretability and strong theoretical grounding. However, their applicability depends critically on the availability of detailed financial statements, which are often missing, delayed or unavailable for small and medium enterprises (SMEs) and private firms [15, 21, 32].

To address these limitations, subsequent studies have explored machine learning-based credit models, including logistic regression and tree-based methods [8, 18, 35], which capture nonlinear relationships among firm attributes. While these often improve predictive performance, they typically treat firms as independent entities and do not explicitly model inter-firm relationships, overlooking the fact that default risk can propagate through economic connections.

## 2.2 Network-Based Financial Risk and GNNs

Recognizing that firms operate within interconnected economic systems, prior research has studied default risk contagion through inter-firm networks, including trade relationships, supply chains and loan-guarantee structures. These studies provide empirical evidence that a firm’s default risk is correlated with that of its business partners, highlighting the importance of relational information [5, 9, 10, 16, 25].

More recently, graph neural networks (GNNs) [23, 37, 38] have enabled scalable learning on relational financial data and have been applied to settings such as loan-guarantee networks and supplychain risk assessment, often outperforming traditional attributebased models [6, 33, 39, 40]. Notably, DGANN [11] leverages guarantee relationships to model default risk in lending systems.

Despite these advances, existing network analyses and GNNbased approaches often focus on non-transactional relations or represent transactions as single-view, unweighted and unidirectional graphs, overlooking key characteristics of real-world corporate transactions. In particular, buyer-seller perspectives and transaction magnitude, which are critical to understanding how risk propagates through transaction networks, remain largely underexplored. Our work addresses this gap by combining large-scale empirical analysis of transaction data with role- and scale-aware network modeling.

## 3 Data Description and Analysis

## 3.1 Data Collection and Preprocessing

The datasets used in this study consist of firm-level attributes, interfirm transaction records, and corporate default information spanning six years (2018–2023). Firm attributes and transaction data were obtained from a large-scale electronic tax-invoice system operated by Douzone Bizon, which is widely used by businesses in Korea for statutory reporting. Corporate default labels were integrated based on credit event records compiled by a licensed credit rating agency in accordance with Basel II default definitions. Preprocessing steps include removing firms with invalid identifiers or missing business type, discarding non-positive transaction amounts, and aggregating multiple transactions between the same pair of firms within each month by transaction role, retaining at most two directed interactions per firm pair (each firm acting as the seller) with transaction amounts summed accordingly. Data collection and usage were conducted within a secure, access-controlled environment provided by the data owner.

![](images/c6adeb453f4ebe0b0b8b304b60dd7dd5901472c974ad73b0901b6a28e1c3de71.jpg)

Figure 1: Transactions involving partners with default history are more likely to cause future default of the target firm.  
![](images/dc75e3e50f4a1244dc1640d1d7a54fac2462fbe0bc6812f759499e8696785b8f.jpg)  
Figure 2: Firms with defaulted trading partners exhibit higher future default rates than those whose partners have no prior default history.

![](images/ee170e2d8c2e84e3eb4014a755ef4f4eee3e3ae3cf0bce2681cbbf900b9acc42.jpg)

![](images/22c3ace2044bfe21fdb40b577b2c19acdd2aa9832f6b51062aec8205960d2f67.jpg)  
Figure 3: Firms that later default tend to have more defaulted trading partners.

## 3.2 Default-Transaction Correlation

3.2.1 Efect ofPartner Default. We first examine how the default history of trading partners relates to a firm’s future default risk. At the transaction level, transactions involving partners with prior (i.e., in the previous 12 months) default history are consistently associated with a higher likelihood of future (i.e., in the next 12 months) default of the target firm across all years (see Fig. 1).

This pattern persists when aggregating transactions at the firm level. As shown in Fig. 2, firms that transact with at least one defaulted partner exhibit substantially higher future default rates than those whose partners have no default history. Moreover, firms that eventually default tend to have both a higher number and a higher proportion of defaulted trading partners among their relationships (see Fig. 3). These trends remain consistent across time.

Overall, these results indicate a strong correlation between partner default history and future corporate default, underscoring the importance of modeling inter-firm dependencies.

3.2.2 Efect ofTransaction Role. A transaction can be viewed from two complementary perspectives: as a sale for the seller and as a purchase for the buyer. While prior work often models default propagation in a single direction — typically emphasizing payment risk faced by sellers, analogous to loaners exposed to guarantee risk [11] — such a unidirectional view overlooks risk exposure on the buyer side, where delivery failures or operational disruptions can also have cascading efects.

![](images/54049d0a5c34c1be59107159ca2c27808584748eaa16a9fc82834f00391d2833.jpg)

Figure 4: Both sales to and purchases from defaulted partners are associated with increased future default risk, indicating bidirectional risk propagation.  
![](images/7c89f72689380130193b2f6689735db104d57881f4a807aae92797151c827b5f.jpg)  
Figure 5: Large-scale (proportion ≥ 0.5) transactions with defaulted partners contribute substantially more to future default risk than small-scale (proportion < 0.5) transactions.

Empirical results support the relevance of both perspectives. As shown in Fig. 4, both sales to and purchases from defaulted partners are associated with elevated future default risk. This suggests that default risk propagates through transactions in a bidirectional manner, motivating the need to explicitly distinguish buyer-view and seller-view transaction networks.

3.2.3 Efect of Transaction Scale. We further investigate how transaction magnitude influences default propagation. Transaction scale is defined as the ratio of a transaction’s amount to the firm’s total transaction volume within the same month.

As shown in Fig. 5, firms that engage in large-scale transactions with defaulted partners are significantly more likely to experience future default than firms without such transactions. In contrast, small-scale transactions with defaulted partners exhibit a much weaker efect. This demonstrates that transaction magnitude plays a critical role in shaping default risk and should be explicitly incorporated into relational modeling.

3.2.4 Summary ofAnalysis. In summary, our empirical analysis identifies three key factors underlying default propagation in transaction networks: 1) the default history of trading partners, 2) the transactional role of firms as buyers or sellers, and 3) the relative scale of transactions. These results indicate that default risk propagates through transactions in a bidirectional and heterogeneous manner, with economically significant transactions contributing disproportionately to future default risk. All observed trends remain consistent across the years considered.

These findings highlight the limitations of unidirectional or binary relational modeling and motivate a dual-perspective, edgeweighted graph-based approach that explicitly accounts for heterogeneous risk exposure.

## 3.3 Multiplex Transaction Networks

Based on the transactional data, we construct multiplex inter-firm transaction networks that explicitly distinguish buyer–seller roles and transaction scale. For each year, firms are represented as nodes, and transactions between firms are represented as directed edges. To reflect asymmetric risk propagation mechanisms, we define two complementary transaction views over the same set of firms.

In the seller-view transaction network, a directed edge from firm � to firm � represents a transaction where � acts as the seller and � acts as the buyer. This view captures risk exposure arising from delayed or failed payments, where financial stress at the buyer can propagate upstream to the seller. Conversely, in the buyerview transaction network, a directed edge from � to � represents the same transaction, reflecting the buyer’s exposure to delivery failures or operational disruptions at the seller.

To account for heterogeneous economic impact, each edge is assigned a weight reflecting the relative transaction scale. Specif ically, for a transaction between firms � and �, the edge weight is defined as the ratio of the transaction amount to the total sales (seller view) or total purchases (buyer view) of the corresponding firm within the same period, i.e., month. To mitigate the efect of extreme values, the weights can be further scaled by the following function: $w _ { u v } ^ { n e w } = l o g ( 1 + \alpha \cdot w _ { u v } ^ { o r i g } ) / l o g ( 1 + \alpha )$ , where $w _ { u v } ^ { o r i g }$ denotes the initial relative transaction ratio and � is a scaling factor.

The resulting multiplex representation consists of two directed, edge-weighted graphs sharing the same node set but encoding distinct transactional semantics (see Table 1 for summary statistics). By jointly modeling these buyer- and seller-view transaction networks, this representation captures role- and scale-dependent risk propagation patterns inherent in real-world corporate transactions.

## 3.4 Firm Attributes

Each firm is represented by a set of non-financial firm attributes (108-dimensional) that capture basic operational characteristics under realistic data constraints where detailed financial statements are unavailable. These include a recent (within that year) default indicator, taxation category, business type encodings at multiple levels of granularity, and log-scaled aggregate sales and purchase statistics computed from historical transaction records. Importantly, these features do not encode information about specific trading partners or network structure, serving as a baseline that is substantially enhanced by the relational information from transaction networks.

## 4 Proposed Framework: DefaultGNN

## 4.1 Problem Setting and Overview

Let $\mathcal { D } = ( \mathcal { V } , \mathcal { E } )$ be a yearly corporate transaction dataset. V is a set of nodes, each node corresponding to an individual firm. $\boldsymbol { x } _ { \boldsymbol { v } } \in \mathbb { R } ^ { F }$ denotes the node features of each node $v \in \mathcal { V } ,$ , consisting of � firm attributes. Each node is further labeled as $y _ { v } \in \{ 0 , 1 \}$ based on whether the corresponding firm defaults within the following year, i.e., $y _ { v } = 1$ for defaulted firms/nodes and $y _ { v } = 0$ for non-defaulted firms. $\mathcal { E } = [ E ^ { s e l l } , E ^ { b u y } ]$ contains two edge sets, corresponding to the seller-view and buyer-view transaction networks defined in Section 3.3, respectively. Specifically, each directed edge (�, �) in $E ^ { v i e w } \left( v i e w \in \{ s e l l , b u y \} \right)$ is paired with edge weight $w _ { u v } ,$ , where the directions and weights in each set are defined to represent their respective view. In other words, D contains year-wise multiplex transaction networks for a set offirms. For clarity, we additionally denote $\mathcal { G } ^ { s e l l } = ( \mathcal { V } , E ^ { s e l l } )$ and $\mathcal { G } ^ { b u y } = ( \mathcal { V } , E ^ { b u y } )$ as the seller- and buyerview graphs, respectively, where V is shared among the two graphs.

![](images/606ca4f576dca444a7bfe875ffb94763cfacc9ed19d3489ed40e073710989f45.jpg)  
Figure 6: Overall framework of DefaultGNN. View-specific embeddings learned from multiplex transaction networks are fused via gating, regularized through view consistency. The fused embeddings are then used for default prediction.

Given dataset $\mathcal { D } = ( \mathcal { V } , \mathcal { E } )$ , our objective is to learn a binary classifier for corporate default prediction. In detail, the classifier is first trained on a training subset $\boldsymbol { \mathcal { V } } _ { t r a i n }$ and validated on a validation subset $\mathcal { N } _ { v a l }$ , finally evaluated by predicting the node labels in a test subset $\boldsymbol { \mathcal { N } } _ { t e s t } ,$ , whose labels were not available during training.

Our proposed DefaultGNN is a dual-perspective GNN-based framework (see Fig. 6) that (1) learns view-specific representations from each transaction network, (2) integrates them via a role-adaptive gating mechanism further stabilized through a viewconsistency regularizer, and finally (3) utilizes the fused embeddings to identify future defaults of nodes.

## 4.2 View-Specific Graph Encoders

DefaultGNN employs two parallel graph encoders, one for each transaction view. Both encoders share the same architectural design but operate on diferent graph structures, and map firms and their relationships into latent representations that capture default risk propagation patterns specific to the corresponding transaction role.

Specifically, each encoder consists of multiple GNN layers that perform structured message passing over the transaction graph, aggregating information from trading partners while accounting for transaction scale represented through edge weights. Formally, node representation for a firm node � at layer � is updated as:

$$
h _ { v } ^ { ( l + 1 ) } = \phi \left( \sum _ { u \in N ( v ) \cup ( v ) } \alpha _ { u v } ( w _ { u v } ) \mathbf { W } ^ { ( l ) } h _ { u } ^ { ( l ) } \right) ,\tag{1}
$$

where $h _ { u } ^ { ( l ) }$ denotes the representation of node � $( h _ { u } ^ { ( 0 ) } = x _ { u }$ initially), $N ( v )$ is the set of 1-hop neighbor nodes of node �, $\mathbf { W } ^ { ( l ) } \in \mathbb { R } ^ { d \times d }$ is the learnable weight matrix (� is the latent embedding dimension size and $\mathbf { W } ^ { ( 0 ) } \in \bar { \mathbb { R } } ^ { d \times F } ) , w _ { u v }$ is the edge weight corresponding to directed edge $( u , v ) , \alpha _ { u v } ( w _ { u v } )$ is a normalized aggregation coeficient that depends on both graph structure and transaction scale, and $\phi ( \cdot )$ is a non-linear activation. By incorporating transaction magnitude directly into message passing, the encoders capture heterogeneous risk exposure arising from economically significant trading relationships, consistent with our empirical analysis. After the final GNN layer, the resulting representations are taken as the view-specific embeddings: $h _ { v } ^ { s e l l }$ and $h _ { v } ^ { b u y }$

## 4.3 Role-Adaptive Gated Fusion

Given the seller- and buyer-view embeddings $h _ { v } ^ { s e l l }$ and $h _ { v } ^ { b u y }$ , DefaultGNN integrates them using a gating mechanism that adaptively determines their relative importance on a per-firm basis. The fusion gate is computed as (�(·) is the sigmoid function):

$$
g _ { v } = \sigma ( w _ { g a t e _ { v } } ^ { \top } [ h _ { v } ^ { s e l l } | | h _ { v } ^ { b u y } ] + b ) .\tag{2}
$$

The gated embeddings are computed as $\tilde { h } _ { v } ^ { s e l l } = g _ { v } \cdot h _ { v } ^ { s e l l }$ and $\tilde { h } _ { v } ^ { b u y } =$ $\left( 1 - g _ { v } \right) \cdot h _ { v } ^ { b u y }$ , respectively, and concatenated to form the fused representation $z _ { v } = [ \tilde { h } _ { v } ^ { s e l l } | | \tilde { h } _ { v } ^ { b u y } ] . z _ { v }$ is then passed to a linear classifier to predict default probability. This gating mechanism allows Default-GNN to account for heterogeneity in transactional roles, enabling the model to emphasize buyer-side or seller-side risk exposure depending on the firm’s position in the transaction network.

View-Consistency Regularization. To promote stable integration of complementary transaction views, we introduce a viewconsistency regularizer that encourages compatible risk estimates when both views are confident, guiding the fusion process toward coherent representations.

Specifically, DefaultGNN additionally produces view-specific default probabilities: $\mathcal { P } _ { v } ^ { s e l l } = P ( y = 1 | h _ { v } ^ { s e l l } ) , \boldsymbol { \phi } _ { v } ^ { b u y } = P ( y = 1 | h _ { v } ^ { b u y } )$ from which we can define view-specific confidences in range [0, 1]: $c _ { v } ^ { s e l l } = 2 | p _ { v } ^ { s e l l } - 0 . 5 | , c _ { v } ^ { b u y } = 2 | p _ { v } ^ { b u y } - 0 . 5 | .$ Disagreement between the obtained probabilities is penalized through a confidence-weighted mean squared error:

$$
\mathcal { L } _ { c o n s } = \frac { \sum _ { v } w _ { v } ^ { c } ( \phi _ { v } ^ { s e l l } - \phi _ { v } ^ { b u y } ) ^ { 2 } } { \sum _ { v } w _ { v } ^ { c } } , w _ { v } ^ { c } = c _ { v } ^ { s e l l } c _ { v } ^ { b u y } .\tag{3}
$$

This regularization encourages agreement primarily when both views are confident, while allowing divergence when either view is uncertain. As a result, it stabilizes gated fusion without suppressing view-specific representations or forcing collapse.

## 4.4 Training Objective

DefaultGNN is trained using weighted binary cross entropy loss to address the severe class imbalance inherent in our task, with far less firms experiencing future default (see Table 1). Formally, given prediction outputs $\hat { p } _ { v } = \sigma ( z _ { v } )$ the classification loss is defined as:

$$
\mathcal { L } _ { c l a s s } = - \sum _ { v \in \mathcal { V } _ { t r a i n } } ( w _ { b c e } \cdot y _ { v } \cdot l o g ( \hat { \rho } _ { v } ) + ( 1 - w _ { b c e } ) \cdot ( 1 - y _ { v } ) \cdot l o g ( 1 - \hat { p } _ { v } ) ,\tag{4}
$$

where $w _ { b c e } \in [ 0 , 1 ]$ is assigned to give larger emphasis to correctly identifying defaulted firms.

The overall training objective is $\mathcal { L } = \mathcal { L } _ { c l a s s } + \lambda \mathcal { L } _ { c o n s } ,$ , where � is a parameter controlling the strength of regularization. Through this procedure, DefaultGNN efectively integrates both buyer- and seller-view transaction networks to predict future corporate default.

## 5 Experiments

In this section, we conduct comprehensive experiments to answer the following research questions:

Table 1: Dataset statistics.
<table><tr><td>Year</td><td># Nodes (Firms)</td><td># Defaulted Nodes (proportion)</td><td># Edges (Transactions)</td></tr><tr><td>2018</td><td>811,866</td><td>8,480 (1.04%)</td><td>42,088,093</td></tr><tr><td>2019</td><td>911,485</td><td>8,794 (0.96%)</td><td>52,693,636</td></tr><tr><td>2020</td><td>950,377</td><td>7,646 (0.80%)</td><td>53,444,395</td></tr><tr><td>2021</td><td>985,320</td><td>7,788 (0.79%)</td><td>54,823,028</td></tr><tr><td>2022</td><td>930,853</td><td>13,718 (1.47%)</td><td>49,250,355</td></tr><tr><td>2023</td><td>420,687</td><td>4,763 (1.13%)</td><td>14,175,876</td></tr></table>

• RQ1. How well does our proposed DefaultGNN perform in predicting corporate default compared with baselines?

• RQ2. How efective are our multiplex networks and additional modules in enhancing DefaultGNN’s performance?

• RQ3. Is DefaultGNN robust under out-of-time settings?

• RQ4. Can DefaultGNN identify and visualize distressed trading partners that critically contribute to a target firm’s default?

## 5.1 Experimental Setup

Datasets. We evaluate DefaultGNN on our constructed multiplex transaction networks (see Section 3.3 for details) spanning six years, i.e., six distinct datasets. The statistics can be found in Table 1. Note that the statistics are equivalent for both buyer- and seller-view networks. 70%, 10% and 20% of the firms are assigned to the training, validation and test sets, respectively, where the ratio of defaulted firms is kept consistent among them, e.g., for year 2021 they each contain 5,451, 778 and 1,559 defaulted nodes (all 0.79% of their corresponding sets), respectively.

Baselines. We compare against baselines spanning four categories. Attribute-based methods (G1) (Logistic Regression [18] and XG-Boost [8]) treat firms as independent instances, only utilizing firm attributes. We also include variants that incorporate additional transaction-related information as features ("+ Trans. Rel.": transaction counts with partners grouped by business type, and "+ Def. Agg.": monthly and annual counts of defaulted partners). Standard GNNs (G2) (GCN [23], GAT [37] and DGANN [11]: a directed graph attention network predicting defaults on loan-guarantee networks) model firms as nodes and transactions as edges, and are applied on the seller-view graph. Multi-relational GNNs (G3) (R-GCN [30] and CompGCN [36]) handle multiple edge types within a single model, operating on a combined graph where buyer and seller transactions are treated as distinct relation types. Multiplex graph models (G4) (DMGI [28], DMG [26] and MGHC [19], adapted to the supervised setting) maintain separate encoders for each viewspecific graph, mirroring DefaultGNN’s dual-encoder design.

Implementation Details. DefaultGNN is implemented in PyTorch 2.2.1 with CUDA 12.1 and trained on an NVIDIA RTX A6000 GPU. The view-specific graph encoders use a 2-layer GCN [23] with em bedding dimension � = 128. The edge scaling factor �, consistency regularization weight �, and BCE weight $w _ { b c e }$ are set to 50, 0.05, and 0.9, respectively. All parameters are optimized with Adam [22] (learning rate 0.001) for up to 500 epochs with early stopping (patience 10). For all baselines, we follow the original implementations and suggested hyperparameters.

Evaluation Details. We evaluate model performance using the Accuracy Ratio (AR) metric derived from the Area Under the Curve (AUC) score, formally defined as $A R = 2 \times A U C - 1$ (range: [0, 1]). AR is a standard performance metric in credit risk modeling, widely used in industry [3, 13, 35] to assess the discriminatory power of default prediction models, especially under severe class imbalance where defaults are rare.

Table 2: Overall model performance. The best AR score for each dataset is highlighted in bold. Standard deviations for DefaultGNN: ±0.001–0.003 (All), ±0.001–0.005 (NoHist), with comparable ranges for other methods. DefaultGNN significantly outperforms the best baseline on all datasets (Wilcoxon signed-rank test, � = 0.016).
<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">2018</td><td colspan="2">2019</td><td colspan="2">2020</td><td colspan="2">2021</td><td colspan="2">2022</td><td colspan="2">2023</td></tr><tr><td>All</td><td>NoHist</td><td>All</td><td>NoHist</td><td>All</td><td>NoHist</td><td>All</td><td>NoHist</td><td>All</td><td>NoHist</td><td>All</td><td>NoHist</td></tr><tr><td rowspan="6">G1</td><td>Log. Reg. [18]</td><td>0.378 0.369</td><td>0.324</td><td>0.426</td><td>0.329</td><td>0.431</td><td>0.315</td><td>0.409 0.414</td><td>0.290</td><td>0.358</td><td>0.257</td><td>0.393 0.394</td><td>0.232</td></tr><tr><td>+ Trans. Rel.</td><td></td><td>0.313</td><td>0.406</td><td>0.313</td><td>0.410</td><td>0.297</td><td></td><td>0.297</td><td>0.362</td><td>0.260</td><td></td><td>0.244</td></tr><tr><td>+ Def. Agg.</td><td>0.381</td><td>0.326</td><td>0.420</td><td>0.324</td><td>0.429</td><td>0.320</td><td>0.412</td><td>0.295</td><td>0.363</td><td>0.258</td><td>0.396</td><td>0.246</td></tr><tr><td>XGBoost [8]</td><td>0.382</td><td>0.294</td><td>0.418</td><td>0.283</td><td>0.419</td><td>0.253</td><td>0.379</td><td>0.203</td><td>0.428</td><td>0.294</td><td>0.447</td><td>0.222</td></tr><tr><td>+ Trans. Rel.</td><td>0.378</td><td>0.289</td><td>0.419</td><td>0.284</td><td>0.424</td><td>0.260</td><td>0.373</td><td>0.220</td><td>0.434</td><td>0.302</td><td>0.448</td><td>0.223</td></tr><tr><td>+ Def. Agg.</td><td>0.399</td><td>0.414</td><td>0.419</td><td>0.284</td><td>0.423</td><td>0.258</td><td>0.389</td><td>0.210</td><td>0.430</td><td>0.297</td><td>0.456</td><td>0.234</td></tr><tr><td rowspan="3">G2</td><td>GCN [23]</td><td>0.502</td><td>0.434</td><td>0.531</td><td>0.430</td><td>0.558</td><td>0.436</td><td>0.540</td><td>0.417</td><td>0.522</td><td>0.412</td><td>0.575</td><td>0.410</td></tr><tr><td>GAT [37]</td><td>0.499</td><td>0.431</td><td>0.534</td><td>0.433</td><td>0.552</td><td>0.439</td><td>0.542</td><td>0.415</td><td>0.518</td><td>0.408</td><td>0.580</td><td>0.417</td></tr><tr><td>DGANN [11]</td><td>0.494</td><td>0.431</td><td>0.513</td><td>0.411</td><td>0.536</td><td>0.413</td><td>0.533</td><td>0.402</td><td>0.502</td><td>0.399</td><td>0.565</td><td>0.404</td></tr><tr><td rowspan="3">G3</td><td>R-GCN [30]</td><td>0.509</td><td>0.432</td><td>0.521</td><td>0.423</td><td>0.542</td><td>0.419</td><td>0.524</td><td>0.400</td><td>0.515</td><td>0.404</td><td>0.560</td><td>0.392</td></tr><tr><td>CompGCN [36]</td><td>0.508</td><td>0.425</td><td>0.528</td><td>0.434</td><td>0.554</td><td>0.429</td><td>0.523</td><td>0.398</td><td>0.519</td><td>0.413</td><td>0.561</td><td>0.395</td></tr><tr><td>DMGI [28]</td><td>0.521</td><td>0.454</td><td>0.549</td><td>0.457</td><td>0.568</td><td>0.451</td><td>0.550</td><td>0.436</td><td>0.530</td><td>0.446</td><td>0.589</td><td>0.430</td></tr><tr><td rowspan="3">G4 Ours</td><td>DMG [26]</td><td>0.515</td><td>0.440</td><td>0.536</td><td>0.445</td><td>0.563</td><td>0.429</td><td>0.551</td><td>0.440</td><td>0.523</td><td>0.442</td><td>0.583</td><td>0.421</td></tr><tr><td>MGHC [19]</td><td>0.518</td><td>0.457</td><td>0.550</td><td>0.448</td><td>0.557</td><td>0.443</td><td>0.546</td><td>0.416</td><td>0.538</td><td>0.435</td><td>0.579</td><td>0.432</td></tr><tr><td>DefaultGNN</td><td>0.542</td><td>0.482</td><td>0.574</td><td>0.480</td><td>0.586</td><td>0.473</td><td>0.583</td><td>0.472</td><td>0.558</td><td>0.463</td><td>0.605</td><td>0.455</td></tr></table>

In practical credit assessment, many firms — especially SMEs and private firms — have no prior default history, limiting the efec tiveness of models that rely on historical risk signals. To evaluate performance in such data-poor settings, we report the AR score not only on all test firms (All), but also on the subset of firms with no prior default history (NoHist). Performance on the latter reflects a model’s ability to infer emerging risk from relational transaction patterns rather than intrinsic historical indicators. For all experiments, we report the average performance of 3 independent runs.

## 5.2 Performance Comparison (RQ1)

Our main results can be found in Table 2. Attribute-based methods (G1) consistently underperform graph-based approaches by a large margin, even when augmented with transaction-related features, highlighting that encoding transactional data as simple features cannot capture the complex relational structure between firms.

Among graph-based methods, standard GNNs (G2) on the sellerview graph achieve strong results, demonstrating the value of relational modeling. However, multi-relational GNNs (G3) perform comparably to or even below standard GNNs, suggesting that merging buyer–seller signals within each GNN layer loses the view-specific information needed to capture asymmetric risk propagation.

Multiplex graph models (G4) consistently outperform both G2 and G3, confirming the importance of preserving view-specific representations. Nevertheless, DefaultGNN outperforms all G4 baselines across every year and both evaluation settings, despite sharing the same dual-encoder architecture. We attribute this to the domain-aligned gated fusion: DefaultGNN’s learned gate values exhibit a strong positive correlation (Pearson $r = 0 . 7 3 , p < 1 0 ^ { - 6 } )$ with firms’ transaction role balance, measured by annual sales / (annual sales + purchases). This indicates that the model adaptively emphasizes the seller- or buyer-view embedding depending on each firm’s economic role, assigning higher weights to the seller view for seller-dominant firms and vice versa. In contrast, more generic fusion strategies such as averaging (DMGI) or disentanglement (DMG) do not capture this role-dependent structure.

DefaultGNN’s advantage is particularly pronounced for firms without prior default history (NoHist), the most practically important setting where traditional credit models fail. This reinforces that DefaultGNN efectively captures emerging risk from transactional relationships even in the absence of intrinsic historical risk signals.

Table 3: Ablation studies on DefaultGNN (AR score).
<table><tr><td colspan="2"></td><td>All</td><td>NoHist</td></tr><tr><td>Ours</td><td>DefaultGNN</td><td>0.583</td><td>0.472</td></tr><tr><td rowspan="4">Network Variants</td><td>DefaultGNN+SellerView</td><td>0.544</td><td>0.420</td></tr><tr><td>DefaultGNN+BuyerView</td><td>0.545</td><td>0.420</td></tr><tr><td>DefaultGNN+CollapsedView1</td><td>0.531</td><td>0.406</td></tr><tr><td>DefaultGNN+CollapsedView2</td><td>0.516</td><td>0.391</td></tr><tr><td rowspan="3">Fusion Variants</td><td>DefaultGNN-EdgeWeights DefaultGNN-Gating</td><td>0.535 0.565</td><td>0.408 0.444</td></tr><tr><td>DefaultGNN-ConsReg</td><td>0.560</td><td>0.437</td></tr><tr><td>DefaultGNN+CrossAttn</td><td>0.537</td><td>0.427</td></tr></table>

## 5.3 Ablation Study (RQ2)

To assess the contribution of each component in DefaultGNN, we conduct ablation experiments on the 2021 dataset (see Table 3), with consistent trends observed across other years. Network variants isolate individual graph-related design choices: both Default-GNN+SellerView and DefaultGNN+BuyerView use only a single transaction view; DefaultGNN+CollapsedView1 combines edges from both views into a single directed graph while for Default-GNN+CollapsedView2 each transaction is reduced to a single undirected edge with both view weights as edge attributes; Default-GNN−EdgeWeights removes edge weights entirely. Fusion variants modify the integration mechanism: DefaultGNN−Gating replaces gated fusion with concatenation, DefaultGNN−ConsReg removes the consistency regularizer, and DefaultGNN+CrossAttn replaces gating with bidirectional cross-attention between views.

The full DefaultGNN achieves the highest performance over all variants. Both single-view models deteriorate similarly, indicating that both perspectives are equally important. Both collapsed views result in a larger drop, highlighting the necessity of separately modeling buyer and seller views. Removing edge weights degrades performance below even single-view models, confirming that transaction scale plays a critical role in default prediction.

Among the fusion variants, replacing gating with concatenation or removing the regularizer both degrade performance while still outperforming single-view models. Replacing gating with crossattention causes an even larger drop, confirming that preserving view-specific representations and adjusting only their relative importance is more efective than allowing views to modify each other.

Further, DefaultGNN is robust across hyperparameter choices: varying embedding dimension � ∈ {32, 64, 128, 256}, regularization weight $\lambda \in \{ 0 . 0 1 , 0 . 0 5 , 0 . 1 , 0 . 2 \}$ , and edge scaling factor � ∈ {1, 10, 50, 100, 200}, the NoHist AR ranges from 0.452 to 0.472 on the 2021 dataset, consistently outperforming the best baselines.

![](images/ff8508bcac98b11d996510b577f3c8b399324e1f8c63c3e1858144a71a042e06.jpg)  
Figure 7: Comparison of model performances under OOT settings. Bar heights represent each model’s AR score on the NoHist subset, while values in the bars show how much the OOT performance of each model diverges from the original performance, i.e., training and testing on the same year.

## 5.4 Out-of-time Analysis (RQ3)

To evaluate generalization in a realistic credit risk assessment setting where models are trained on historical data and tested on future years [2, 13, 17, 31], we compare DefaultGNN against GCN and XGBoost in two out-of-time (OOT) configurations: (1) training on a single year (2018) and testing on 2019–2023, and (2) training sequentially on three years (2018–2020) and testing on 2021–2023.

As shown in Fig. 7, DefaultGNN consistently outperforms baselines across both configurations. In the single-year setting, Default-GNN maintains strong performance even as the training-test gap increases to five years, whereas GCN and XGBoost exhibit substan tially larger degradation. In the three-year setting, sequential retraining allows DefaultGNN to closely match or even improve upon its in-year performance, demonstrating efective integration of new data over time. These results confirm that DefaultGNN generalizes well to future unseen data, a critical requirement for real-world credit risk models that must adapt to evolving economic conditions.

## 5.5 Case Studies (RQ4)

We present two case studies in Fig. 8 to illustrate how Default-GNN identifies meaningful transaction-based risk signals beyond raw prediction performance. Specifically, for each target firm, we extract a two-hop local transaction subgraph from both buyer and seller views and attribute the prediction to individual transaction edges using integrated gradients [29, 34, 41], which identifies the trading relationships and partners that most strongly drive the model’s decision.

In Fig. 8(a), DefaultGNN highlights a transaction with a previously defaulted firm located two hops away from the target firm, demonstrating its ability to capture indirect risk propagation. In contrast, XGBoost fails to incorporate this dependency and incorrectly predicts the firm as safe. In Fig. 8(b), an influential relationship corresponds to a buyer-view transaction with a defaulted trading partner, which is then propagated to the target firm through a seller view transaction. While DefaultGNN correctly captures this signal through its dual-view design, a single-view GCN misses the risk and produces an incorrect prediction, underscoring the importance of modeling asymmetric transaction roles.

Beyond improving predictive accuracy, these visualizations of fer clear and actionable insights by identifying influential trading partners and transaction links. Such interpretability can support financial institutions in understanding risk exposure, monitoring vulnerable relationships, and making informed credit decisions.

![](images/6a988754bced3b3fe7d977a0b680e967af6ab95de787770cbfb8a9a7000f7827.jpg)  
Figure 8: Case studies, where edges (transactions) and corresponding neighbors (trading partners) influential to the prediction of the target node (star-shaped) are highlighted.

## 6 Practical Deployment

To evaluate practical applicability, DefaultGNN’s predictions were validated in collaboration with Techfin Ratings, a licensed credit rating agency that operates a logistic regression-based credit scoring model (MIS) using real-time financial and operational data. Default-GNN’s predicted default scores and MIS scores were used as input features to train a separate logistic regression model, assessing whether transaction network information provides complementary value to existing credit models. The evaluation covered 1,166,927 corporate firms and 197,397 sole proprietors using unseen data from 2018 onward, with the sequentially trained DefaultGNN model from Section 5.4 (trained on 2018–2020) applied without retraining. In practice, default prediction scores are used by financial institutions to inform credit approval decisions, where firms below a given risk threshold are deemed eligible for lending. To simulate this scenario, a risk threshold was applied to each model’s predicted scores to determine firm eligibility for credit approval. Integrating Default-GNN increased the approval rate by 6.95 percentage points for corporate firms (51.57% to 58.52%) and 11.11 percentage points for sole proprietors (36.13% to 47.24%), while maintaining or reducing the default rate among approved firms (unchanged at 0.41% for corporate firms; 0.14% to 0.11% for sole proprietors). These results suggest that transaction network-based risk signals complement existing credit models by expanding the pool of approvable firms without increasing portfolio risk, particularly benefiting SMEs and sole proprietors for whom traditional financial indicators are limited.

## 7 Conclusion

In this work, we propose DefaultGNN, a GNN-based framework for corporate default prediction that leverages buyer–seller transaction networks under realistic data constraints where financial statements are limited or unavailable. Through extensive empirical analysis of large-scale transactional data, we show that inter-firm transactions encode critical risk signals related to transaction role and scale. Guided by these findings, DefaultGNN models transactions from dual buyer and seller perspectives while incorporating transaction magnitude, consistently outperforming existing baselines — particularly for firms without prior default history. Beyond predictive performance, the model provides interpretable insights that help identify influential trading partners, ofering practical value for real-world credit risk assessment and monitoring. Further, Default-GNN’s predictions complement an existing credit scoring model, greatly improving approval rates without increasing portfolio risk. Acknowledgements. This work was supported by the National Research Foundation of Korea (NRF) grants funded by the Korea government (MSIT) (RS-2024-00335098) and the Ministry of Science and ICT (RS-2022-NR068758), and by Douzone/Techfin Ratings.

## GenAI Disclosure

We acknowledge the use of LLMs (e.g., GPT-5, Claude) for limited assistance with (1) editing this paper for grammar, clarity, expression variation, and length reduction to meet page limits, and (2) minor refactoring/debugging of code used for plotting and visual ization. All AI-assisted edits and code changes were reviewed and validated by the authors, and all core ideas, methods, experiments, and interpretations are original contributions of the authors.

## References

[1] Edward I Altman. 1968. Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. The journal of finance 23, 4 (1968), 589–609.

[2] Bart Baesens, Tony Van Gestel, Stijn Viaene, Maria Stepanova,Johan Suykens, and Jan Vanthienen. 2003. Benchmarking state-of-the-art classification algorithms for credit scoring. Journal ofthe operational research society 54, 6 (2003), 627–635.

[3] Basel Committee on Banking Supervision. 2005. Studies on the Validation of Internal Rating Systems. Basel Committee Working Paper 14. Bank for International Settlements. https://www.bis.org/publ/bcbs\_wp14.pdf

[4] William H Beaver. 1966. Financial ratios as predictors of failure. Journal of accounting research (1966), 71–111.

[5] Claudia Berloco, Gianmarco De Francisci Morales, Daniele Frassineti, Greta Greco, Hashani Kumarasinghe, Marco Lamieri, Emanuele Massaro, Arianna Miola, and Shuyi Yang. 2021. Predicting corporate credit risk: Network contagion via trade credit. PLoS One 16, 4 (2021), e0250115.

[6] Wendong Bi, Bingbing Xu, Xiaoqian Sun, Zidong Wang, Huawei Shen, and Xueqi Cheng. 2022. Company-as-tribe: Company financial risk assessment on tribe style graph with hierarchical graph neural networks. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining. 2712–2720.

[7] Frederic Boissay and Reint E Gropp. 2007. Trade credit defaults and liquidity provision by firms. Technical Report. ECB working paper.

[8] Tianqi Chen. 2016. XGBoost: A Scalable Tree Boosting System. Cornell University (2016).

[9] Ting-Qiang Chen and Jian-Min He. 2012. A network model of credit risk conta gion. Discrete Dynamics in Nature and Society 2012, 1 (2012), 513982.

[10] Wentao Chen, Zhenlin Li, and Zhuoxin Xiao. 2021. On credit risk contagion of supply chain finance under COVID-19. Journal ofMathematics 2021, 1 (2021), 1281825.

[11] Dawei Cheng, Xiaoyang Wang, Ying Zhang, and Liqing Zhang. 2020. Risk guarantee prediction in networked-loans. In IJCAI International Joint Conference on Artificial Intelligence.

[12] Jonathan Crook, Viani Djeundje, Rafaella Calabrese, and Mona Hmid. 2019. Credit scoring with alternative data. (2019).

[13] Bernd Engelmann, Evelyn Hayden, and Dirk Tasche. 2003. Testing rating accuracy. Risk 16, 1 (2003), 82–86.

[14] Andreas Fuster, Paul Goldsmith-Pinkham, Tarun Ramadorai, and Ansgar Walther. 2022. Predictably unequal? The efects of machine learning on credit markets. The Journal ofFinance 77, 1 (2022), 5–47.

[15] Parikshit Ghosh, Dilip Mookherjee, Debraj Ray, et al. 2000. Credit rationing in developing countries: an overview of the theory. Readings in the theory of economic development 7 (2000), 383–401.

[16] Kay Giesecke and Stefan Weber. 2004. Cyclical correlations, credit contagion, and portfolio losses. Journal of Banking & Finance 28, 12 (2004), 3009–3036.

[17] David J Hand and William E Henley. 1997. Statistical classification methods in consumer credit scoring: a review. Journal of the royal statistical society: series a (statistics in society) 160, 3 (1997), 523–541.

[18] David W Hosmer Jr, Stanley Lemeshow, and Rodney X Sturdivant. 2013. Applied logistic regression. John Wiley & Sons.

[19] Yudi Huang, Ci Nie, Hongqing He, Yujie Mo, Yonghua Zhu, Guoqiu Wen, and Xiaofeng Zhu. 2025. Multiplex graph representation learning with homophily

and consistency. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 11835–11842.

[20] Tor Jacobson and Erik Von Schedvin. 2015. Trade credit and the propagation of corporate failure: An empirical analysis. Econometrica 83, 4 (2015), 1315–1371.

[21] Dwight Jafee and Joseph Stiglitz. 1990. Credit rationing. Handbook of monetary economics 2 (1990), 837–888.

[22] Diederik P Kingma. 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 (2014).

[23] Thomas N Kipf and Max Welling. 2016. Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907 (2016).

[24] Joginder Kumar, Vinod Kumar, Deepika Verma, and Somya Sharma. 2021. Prediction of corporate bankruptcy based on financial ratios using binary logistic regression. International Journal of Statistics and Reliability Engineering 7, 3 (2021), 376–381.

[25] Ying Liu, Shuang Liu, and Yu Lu. 2025. Supply chain financial risk assessment: A modified graph attention neural network. Finance Research Letters (2025), 108285.

[26] Yujie Mo, Yajie Lei, Jialie Shen, Xiaoshuang Shi, Heng Tao Shen, and Xiaofeng Zhu. 2023. Disentangled multiplex graph representation learning. In International conference on machine learning. PMLR, 24983–25005.

[27] James A Ohlson. 1980. Financial ratios and the probabilistic prediction of bank ruptcy. Journal ofaccounting research (1980), 109–131.

[28] Chanyoung Park, Donghyun Kim, Jiawei Han, and Hwanjo Yu. 2020. Unsupervised attributed multiplex network embedding. In Proceedings ofthe AAAI conference on artificial intelligence, Vol. 34. 5371–5378.

[29] Phillip E Pope, Soheil Kolouri, Mohammad Rostami, Charles E Martin, and Heiko Hofmann. 2019. Explainability methods for graph convolutional neural networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 10772–10781.

[30] Michael Schlichtkrull, Thomas N Kipf, Peter Bloem, Rianne Van Den Berg, Ivan Titov, and Max Welling. 2018. Modeling relational data with graph convolutional networks. In European semantic web conference. Springer, 593–607.

[31] Tyler Shumway. 2001. Forecasting bankruptcy more accurately: A simple hazard model. The journal ofbusiness 74, 1 (2001), 101–124.

[32] Joseph E Stiglitz and Andrew Weiss. 1981. Credit rationing in markets with imperfect information. The American economic review 71, 3 (1981), 393–410.

[33] Haotian Sun. 2024. Research on financial risk assessment algorithm based on graph neural network. In Proceedings ofthe 2024 4th International Conference on Big Data, Artificial Intelligence and Risk Management. 932–937.

[34] Mukund Sundararajan, Ankur Taly, and Qiqi Yan. 2017. Axiomatic attribution for deep networks. In International conference on machine learning. PMLR, 3319– 3328.

[35] Lyn Thomas, Jonathan Crook, and David Edelman. 2017. Credit scoring and its applications. SIAM.

[36] Shikhar Vashishth, Soumya Sanyal, Vikram Nitin, and Partha Talukdar. 2019. Composition-based multi-relational graph convolutional networks. arXiv preprint arXiv:1911.03082 (2019).

[37] Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Lio, and Yoshua Bengio. 2017. Graph attention networks. arXiv preprint arXiv:1710.10903 (2017).

[38] Daixin Wang, Zhiqiang Zhang, Jun Zhou, Peng Cui, Jingli Fang, Quanhui Jia, Yanming Fang, and Yuan Qi. 2021. Temporal-aware graph neural network for credit risk prediction. In Proceedings of the 2021 SIAM International Conference on Data Mining (SDM). SIAM, 702–710.

[39] Ke Xu, You Wu, Haohao Xia, Ningjing Sang, and Bingxing Wang. 2022. Graph Neural Networks in Financial Markets: Modeling Volatility and Assessing Valueat-Risk. Journal of Computer Technology and Software 1, 2 (2022).

[40] Shuo Yang, Zhiqiang Zhang, Jun Zhou, Yang Wang, Wang Sun, Xingyu Zhong, Yanming Fang, Quan Yu, and Yuan Qi. 2021. Financial risk analysis for SMEs with graph-based supply chain mining. In Proceedings of the twenty-ninth international conference on international joint conferences on artificial intelligence. 4661–4667.

[41] Zhitao Ying, Dylan Bourgeois, Jiaxuan You, Marinka Zitnik, and Jure Leskovec. 2019. Gnnexplainer: Generating explanations for graph neural networks. Advances in neural information processing systems 32 (2019).