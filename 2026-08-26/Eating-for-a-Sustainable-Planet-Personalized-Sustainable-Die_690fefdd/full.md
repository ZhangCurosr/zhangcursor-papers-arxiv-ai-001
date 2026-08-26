# Eating for a Sustainable Planet: Personalized Sustainable Diet Recommendation via Constraint-Aware Decision-Making Modeling

Ying Jin <sup>1</sup> <sup>2</sup> <sup>3</sup> Weiqing Min <sup>2</sup> <sup>3</sup> Mingyu Huang <sup>2</sup> <sup>3</sup> Shuqiang Jiang <sup>2</sup> <sup>3</sup>

## Abstract

A sustainable diet represents a multi-dimensional synergy among four essential pillars: nutrition adequacy, economic affordability, cultural acceptability, and environmental respect. Despite the prevalence of population-level sustainability modeling, practical implementation relies on effective individual-level adoption. This transition is often hindered by inter-individual heterogeneity, posing a formidable challenge in aligning sustainable diet requirements with individual preferences. To address this issue, we propose a personalized sustainable diet recommendation model based on a constraint-aware decision-making mechanism, where sustainability is incorporated through learnable constraints rather than modeled as user preferences. To systematically evaluate the proposed approach, we construct a sustainable diet dataset named SusDiet with about 150k recipes, characterized by broad coverage of sustainability indicators. Experimental results on this dataset show that our method promotes more sustainable choices without compromising individual preference. This work establishes a framework for aligning individual dietary choices with planetary health, offering quantitative evidence to guide future sustainable diet interventions and policymaking for sustainable development.

## 1. Introduction

Sustainable development stands as a defining global challenge of the 21st century (Sachs, 2012). In response to

<sup>1</sup>School of Advanced Interdisciplinary Sciences, University of Chinese Academy of Sciences, Beijing, China <sup>2</sup>State Key Laboratory of AI Safety, Institute of Computing Technology, Beijing, China <sup>3</sup>University of Chinese Academy of Sciences, Beijing, China. Correspondence to: Weiqing Min <minweiqing@ict.ac.cn>.

accelerating climate change, biodiversity loss, and deepening social inequality, the United Nations’ Sustainable Development Goals (SDGs) provide a shared framework for global progress (Lee et al., 2016; Colglazier, 2015; Hak´ et al., 2016). Among the SDGs, climate action, responsible consumption and production, zero hunger, good health and well-being, clean water use, and life on land are all closely related to dietary patterns (Rockstrom et al.¨ , 2009b;a; Steffen et al., 2015). Consequently, dietary change has been recognized as a key leverage point to address global sustainability challenges and advance multiple SDGs simultaneously (Springmann et al., 2018; Willett et al., 2019).

Within this sustainability agenda, sustainable diets are defined as dietary patterns that promote individual health and well-being while simultaneously ensuring low environmental impact, economic affordability, safety, and cultural acceptability (Organization et al., 2021; Gibney, 2025). However, translating this consensus into feasible, widely adopted practices remains an open challenge (Biesbroek et al., 2023). To address this, extensive research has sought to define the multifaceted requirements of sustainable dietary patterns (Springmann et al., 2018; Bajzelj et al.ˇ , 2021; Ye et al., 2024; Merrigan et al., 2015). Current studies often utilize mathematical optimization or scenario analysis to meet nutritional, environmental, economic, and cultural objectives within idealized constraints (Springmann et al., 2018; Gazan et al., 2018; Wilson et al., 2019). While such studies provide foundational guidance for dietary transitions at national or global scales (Heerschop et al., 2024), they primarily treat sustainability as an explicit objective or hard constraint at the population level, while neglecting inter-individual heterogeneity. This focus limits the application of global recommendations to real-world sustainable practices, highlighting the need to model sustainable dietary patterns at the individual level (de Costa et al., 2025).

Existing dietary patterns are shaped by long-standing cultural norms, economic constraints, and deeply ingrained eating habits, rendering the transition to sustainable diets a formidable challenge for the individual. Even when minimizing deviations from current dietary patterns, achieving nutritionally adequate and environmentally sustainable diets often still requires major structural changes in consumption, particularly regarding staple foods, meat, and dairy products (Chaudhary & Krishna, 2019; Wilson et al., 2019). These changes often exceed the threshold of individual acceptability. As a result, consumers often exhibit low willingness to change their dietary behavior, even when aware of the necessity (Elliott et al., 2024; Gibney, 2025; Biesbroek et al., 2023), making it difficult to map conceptual findings onto the everyday decision-making of real individuals.

To bridge this gap, we need modeling approaches focused on individual decision-making. However, current personalized diet recommendation systems, largely adapted from other domains, focus primarily on learning and optimizing for explicit user preferences or satisfaction (Qiao et al., 2025; Zhang et al., 2023; Yang et al., 2017). This paradigm implicitly assumes that food choices are driven by consciously articulated tastes and interests, and consequently treats sustainability also as an explicit preference dimension. In reality, individuals rarely make dietary decisions with explicit preference or awareness regarding sustainability (Fernqvist et al., 2024; Gonzalez et al., 2025; Nguyen et al., 2025). Therefore, treating sustainability merely as a preference to learn is conceptually misaligned with human behavior. Consequently, developing a decision-making framework that acknowledges individual differences while adhering to sustainability constraints rather than simply optimizing for user preferences remains a major challenge.

To address this challenge, we propose a novel framework that reformulates personalized sustainable diet recommendation as a constraint-aware decision-making process. Unlike approaches that model sustainability as explicit user preference, our method formulates sustainability in terms of learnable constraint thresholds and jointly integrates them with preference learning within a unified decision-making framework. In our formulation, cultural acceptability is implicitly reflected through user preference representations learned from historical dietary interactions, while other sustainability dimensions, including nutrition, environmental impact, economic cost, and animal welfare, are modeled through explicit measurable indicators. The framework consists of three technically distinct components: (1) A Mixture-of-Experts (MoE) Transformer to encode multidimensional sustainability signals from structured ingredient compositions and quantitative attributes. It can produce expert-specific embeddings that facilitate the learning of constraint encodings across heterogeneous sustainability dimensions. These representations are supervised through an uncertainty-weighted multi-task learning objective to ensure balanced modeling of diverse sustainability indicators. (2) A multi-interest attention mechanism to capture heterogeneous taste preferences from historical interactions, enabling the model to represent multiple latent preference profiles for each user. (3) A personalized constraint mechanism that learns user-specific sustainability constraint thresholds and trade-off weights. These constraints interact with sustainability signals to dynamically penalize recipes that exceed individual sustainability boundaries in the final ranking score. The overall framework is optimized with a unified objective that jointly balances preference learning and sustainability constraint enforcement, enabling effective trade-offs between personalization accuracy and multi-dimensional sustainability considerations.

In addition, we construct SusDiet, a specialized sustainability dataset designed to support large-scale modeling of sustainability-aware dietary decision-making. SusDiet contains 149,036 recipes annotated with a rich collection of sustainability indicators across multiple dimensions, including nutritional adequacy, environmental impact, economic cost, and animal welfare, providing a holistic characterization of recipe level sustainability. These recipe annotations are further integrated with real-world user recipe interaction data, resulting in a dataset comprising 179,869 users and 744,138 interaction records. Extensive experiments conducted on SusDiet demonstrate the effectiveness of the proposed approach.

## Our specific contributions are as follows:

• We propose a personalized sustainable diet recommendation framework, formulated as a constraint-aware decision-making task. By integrating a MoE-based sustainability model, a multi-interest attention mechanism for preferences learning and a personalized constraint mechanism for user-specific sustainability trade-offs, the framework effectively balances preference learning with sustainability requirements to deliver tailored, eco-friendly recommendations.

• We construct a large-scale real-world dataset with approximately 150k recipes annotated by comprehensive sustainability indicators across nutritional adequacy, environmental impact, economic cost, and animal welfare, aligned with user-recipe interactions.

• Experiments demonstrate that learning user-specific sustainability constraints yields substantially more sustainable recommendations while maintaining competitive recommendation accuracy, offering quantitative evidence for both modeling effectiveness and policy relevance.

## 2. Related Work

Research on sustainable diets has emerged as a critical domain addressing the nexus of human health and planetary boundaries. Existing research on sustainable diets can be broadly categorized into macro-level optimization aimed at policy formulation and micro-level modeling focused on individual behavioral interventions (Biesbroek et al., 2023).

Macro-level Modeling and Optimization A dominant line of work operates at the population or national level, utilizing large-scale modeling to develop dietary patterns that meet nutritional needs while minimizing environmental impacts and considering cost or cultural constraints. For example, the health and environmental benefits of shifting toward plant-based diets have been quantified across 150 countries (Springmann et al., 2018), while a nonlinear optimization framework was proposed to design country-specific diets that minimize deviations from current consumption under planetary-boundary constraints (Chaudhary & Krishna, 2019). Similarly, region-specific reference diets have been derived through extensive data analysis (Ye et al., 2024) and machine learning has been widely used to navigate trade-offs among nutrition, environmental impact, cost, and acceptability (Wilson et al., 2019; Gazan et al., 2018). While these policy-centric and optimization-based approaches provide crucial evidence for the feasibility of sustainable food systems and inform global strategies, they are inherently normative. The output of these approaches is a macrolevel optimal pattern rather than a model of how sustainable choices emerge from heterogeneous individuals’ decision processes.

Individual-level Recommendations Complementing macro-level prescriptions, individual-level modeling offers a path toward scalable, personalized interventions. By learning preferences at scale, recommendation systems (RS) can deliver tailored dietary guidance aligned with sustainability goals (He et al., 2024). Recent efforts include datasets and methods aimed at sustainability-oriented recommendations (Zhang et al., 2024a), but these studies focus primarily on nutritional and environmental dimensions. Many food RS still prioritize recommendation accuracy; for instance, an RS model was proposed to improve performance without explicitly promoting sustainability (Zhang et al., 2024b). More behavior-aware approaches attempt to integrate greener choices into preference learning. For example, Jing et al. (2025) proposed a recommendation framework to capture user preferences and attitudes toward greener options. Yet, such methods often presume that users hold explicit sustainability attitudes, which is not always true in reality. In contrast, our constraint-aware decision-making mechanism incorporates sustainability dimensions without requiring them to be explicitly represented in user preferences, thus addressing this limitation.

## 3. Task Formulation

We study the problem of personalized sustainable diet recommendation. The goal is to recommend recipes that satisfy a user’s taste preferences while adhering to their unique boundaries across various sustainability metrics. Beyond the conventional recommendation objective of learning user preferences, this task formulation requires the modeling of user-specific constraints for each sustainability dimension.

Formally, we assume there are N sustainability dimensions. Each recipe r is associated with an N-dimensional sustainability indicator vector $\pmb { s } _ { r } = [ s _ { r } ^ { 1 } , \ldots , s _ { r } ^ { N } ]$ , where each dimension corresponds to a specific sustainability criterion.

For each user u, we define a user-specific constraint vector $\lambda _ { u } = [ \lambda _ { u } ^ { 1 } , \dots , \lambda _ { u } ^ { N } ]$ , where $\lambda _ { u } ^ { n }$ denotes the bound on the nth sustainability indicator that the user is willing to tolerate. Without loss of generality, all sustainability indicators are normalized and formulated in a unified minimization form, such that lower values indicate more sustainable outcomes.

Given an estimated preference score $\hat { y } _ { u , r }$ for each user– recipe pair $( u , r )$ , where larger values indicate stronger preference, the personalized recommendation problem can be formulated as the following constrained optimization:

$$
\begin{array} { r l } { \underset { r } { \operatorname* { m a x } } } & { \hat { y } _ { u , r } } \\ { \mathrm { s . t . } } & { s _ { r } ^ { n } \leq \lambda _ { u } ^ { n } , \quad n = 1 , \ldots , N . } \end{array}\tag{1}
$$

To handle these constraints, we introduce a user-specific Lagrange multiplier vector $\pmb { \alpha } _ { u } = [ \alpha _ { u } ^ { 1 } , \dots , \alpha _ { u } ^ { N } ]$ , and define the following function:

$$
F ( u , r ) = \hat { y } _ { u , r } - \pmb { \alpha } _ { u } ^ { \top } ( \pmb { s } _ { r } - \pmb { \lambda } _ { u } ) .\tag{2}
$$

Our objective is to maximize $F ( u , r )$ . Accordingly, the task is defined as learning user-specific sustainability constraints $\lambda _ { u }$ and trade-off weights $\alpha _ { u } ,$ , and recommending the top-K recipes that maximize $F ( u , r )$ for each user.

## 4. Methodology

Based on the proposed formulation, we design a unified framework for personalized sustainable diet recommendation, as illustrated in Figure 1.

The framework consists of three synergistic components: (i) a recipe sustainability representation that encodes each recipe into a latent embedding $\mathbf { z } _ { r } \in \mathbb { R } ^ { D }$ using a Mixture-of-Experts (MoE) architecture to capture heterogeneous and potentially competing sustainability indicators such as nutritional adequacy, environmental impact, economic cost, and animal welfare; (ii) a preference learning that estimates a preference score $\hat { y } _ { u , r }$ for each user–recipe pair $( u , r )$ solely from historical interactions, thereby modeling intrinsic user interests independently of sustainability considerations; and (iii) a personalized constraint-aware trade-off that integrates the learned recipe representations and preference scores to infer user-specific decision parameters, namely a constraint encoding $\lambda _ { u }$ and a trade-off encoding $\pmb { \alpha } _ { u }$ , thereby enabling balance between individual preferences and personalized sustainability boundaries.

![](images/63e8f1c3ae88c39cd89e9ac7c8f0e1f2c7677509937ec23345b514df3bc5fac8.jpg)  
Figure 1. The proposed framework for personalized sustainable diet recommendation. The architecture including three components: (1) The Recipe Sustainability Representation (left) utilizes a MoE encoder to transform ingredient tokens and quantities into a multidimensional sustainability embedding z . (2) The Preference Learning (right) leverages self-attention and interest attention to output the preference score $\hat { y } _ { u , \cdot }$ <sub>i</sub> from user-recipe interactions. (3) The Personalized Constraint-aware Trade-off (bottom) synthesizes these outputs to instantiate the decision function via user-specific parameters $\lambda _ { u }$ and $\mathbf { { \alpha } } _ { \mathbf { { \alpha } } } \mathbf { { \alpha } } _ { \mathbf { { \alpha } } } .$ , as shown in the term ${ \pmb { \alpha } } _ { u } ^ { \top } ( { \bf z } _ { r } - \lambda _ { u } )$ . The entire system is optimized using a joint loss including sustainability, rating, and ranking objectives.

## 4.1. Recipe Sustainability Representation

This module learns sustainability-aware recipe representations by jointly modeling ingredient compositions and multiple sustainability attributes. It produces a structured embedding z<sub>r</sub> that captures both culinary semantics and sustainability profiles.

Ingredient-Quantity Fusion. For a recipe r, we represent its composition as a sequence of L ingredient tokens $\mathbf { t } _ { r } \in \mathbb { N } ^ { L }$ and corresponding quantities $\mathbf { q } _ { r } \in \mathbb { R } ^ { L }$ . Each ingredient $t _ { r , j }$ is mapped to an embedding ${ \bf e } _ { r , j } \in \mathbb { R } ^ { D }$ while its quantity $q _ { r , j }$ is projected into a continuous space $\mathbf { u } _ { r , j } = W _ { q } q _ { r , j } + b _ { q } \in \mathbb { R } ^ { d _ { q } }$ . We fuse these to form the input sequence $\mathbf { X } _ { r } = [ \mathbf { x } _ { r , 1 } , \ldots , \mathbf { x } _ { r , L } ]$ , where:

$$
\mathbf { x } _ { r , j } = W _ { \mathrm { i n } } [ \mathbf { e } _ { r , j } ; \mathbf { u } _ { r , j } ] + b _ { \mathrm { i n } } \in \mathbb { R } ^ { D } .\tag{3}
$$

Mixture-of-Experts Recipe Encoder. To capture diverse ingredient patterns, we employ a Mixture-of-Experts (MoE) encoder with $E = 4$ Transformer-based experts. Each expert e independently processes ${ \bf X } _ { r }$ and produces an expert-specific representation ${ \bf p } _ { r } ^ { \left( e \right) }$ via masked mean pooling. A gating network computes selection weights $\mathbf { g } _ { r } =$ softma $\mathfrak { c } ( \mathrm { M L P } ( \bar { \mathbf { x } } _ { r } ) )$ based on the sequence average $\bar { \mathbf { x } } _ { r }$ The final sustainability-aware embedding z is:

$$
\mathbf { z } _ { r } = \mathrm { L a y e r N o r m } \left( W _ { g } \sum _ { e = 1 } ^ { E } g _ { r } [ e ] \mathbf { p } _ { r } ^ { ( e ) } + b _ { g } \right) .\tag{4}
$$

Multi-task Sustainability Prediction. Given $\mathbf { z } _ { r }$ , we jointly predict K sustainability indicators using task-specific heads: $\hat { s } _ { r } ^ { ( k ) } = \mathrm { M L P } _ { k } ( \mathbf { z } _ { r } )$ . To stabilize training across heterogeneous target scales, we apply a signed logarithmic transformation and z-score normalization to the raw targets $s _ { r } ^ { ( k ) }$ . We optimize the module using an uncertainty-weighted multi-task loss $\mathcal { L } _ { \mathrm { t a s k } }$

$$
\mathcal { L } _ { \mathrm { t a s k } } = \sum _ { k = 1 } ^ { K } \left( \exp ( - v _ { k } ) \cdot \mathrm { M S E } ( \hat { s } _ { r } ^ { ( k ) } , s _ { r } ^ { ( k ) } ) + v _ { k } \right) ,\tag{5}
$$

where $v _ { k }$ is a learnable parameter that automatically balances the gradient contribution of each task k based on its respective noise level.

## 4.2. Preference Learning Module via Multi-interest

To capture heterogeneous and context-dependent user tastes, we design a multi-interest mechanism to learn preferenceoriented representations from the historical interaction set $\mathcal { H } _ { u } = \{ ( r _ { i } , y _ { u , r _ { i } } ) \} _ { i = 1 } ^ { N _ { u } }$ , where $y _ { u , r _ { i } }$ denotes the observed rating for recipe $r _ { i }$ and $N _ { u }$ is the interaction count.

User and Recipe Encoding. Each historical recipe $r _ { i }$ is first mapped into a latent semantic space through a learnable embedding table $\mathbf { E } _ { u } [ r _ { i } ]$ . To preserve chronological dependencies, positional embeddings are incorporated: $\mathbf { h } _ { i } = \mathbf { E } _ { u } [ r _ { i } ] + \mathbf { p } _ { i }$ , where $\mathbf { p } _ { i }$ denotes the embedding of the i-th position. We then apply MLP and self-attention over the historical sequence to capture dependencies among interacted recipes and obtain refined representations $\{ \tilde { \mathbf { h } } _ { i } \} _ { i = 1 } ^ { - N _ { u } }$

For a target recipe r, we construct representations from recipe identity, ingredient composition, and region embeddings: $\mathbf { v } _ { r , i d } \ = \ \mathbf { E } _ { \mathrm { i t e m } } [ r ] , \ \mathbf { v } _ { r , \mathrm { i n g } } \ = \ \mathbf { E } _ { \mathrm { i n g } } [ r ]$ and $\mathbf { v } _ { r , \mathrm { r e g } } =$ $\mathbf { E } _ { \mathrm { r e g } } [ r ]$ . Self-attention and pooling are further applied to obtain refined recipe representations, which are concatenated and projected into the final recipe embedding $\mathbf { e } _ { r }$

Multi-interest Attention. To capture diverse user interests from different recipe views, we apply multi-interest attention separately on the identity, ingredient, and regional representations. Taking $\tilde { \mathbf { v } } _ { r , i d }$ as the query, the attention weight for each historical interaction is computed by:

$$
\beta _ { i , r } ^ { i d } = \frac { \exp \Big ( \phi _ { Q } ^ { i d } ( \tilde { \mathbf { v } } _ { r , i d } ) ^ { \top } \phi _ { K } ^ { i d } ( \tilde { \mathbf { h } } _ { i } ) / \sqrt { d } \Big ) } { \sum _ { j = 1 } ^ { N _ { u } } \exp \Big ( \phi _ { Q } ^ { i d } ( \tilde { \mathbf { v } } _ { r , i d } ) ^ { \top } \phi _ { K } ^ { i d } ( \tilde { \mathbf { h } } _ { j } ) / \sqrt { d } \Big ) } ,\tag{6}
$$

where $\phi ( \cdot )$ represents linear projections. The interestaware user representation is then aggregated as $\mathbf { e } _ { u , i d } =$ $\begin{array} { r } { \sum _ { i = 1 } ^ { N _ { u } } \beta _ { i , r } ^ { i d } \phi _ { V } ^ { i d } ( \tilde { \mathbf { h } } _ { i } ) } \end{array}$ . Similarly, $\tilde { \mathbf { v } } _ { r , \mathrm { i n g } }$ and $\tilde { \mathbf { v } } _ { r , \mathrm { r e g } }$ are processed through the same attention mechanism to obtain the representation $\mathbf { e } _ { u , \mathrm { i n g } }$ and $\mathbf { e } _ { u , \mathrm { r e g } }$ , respectively. The final user representation is computed by ${ { \bf { e } } _ { u } } = \mathrm { M L P } \left( { \left[ { { { \bf { e } } _ { u , i d } } ; { { \bf { e } } _ { u , \mathrm { { i n g } } } } ; { { \bf { e } } _ { u , \mathrm { { r e g } } } } } \right] } \right)$

Finally, the predicted preference score is defined by the inner product:

$$
\hat { y } _ { u , r } = \langle \mathbf { e } _ { u } , \mathbf { e } _ { r } \rangle .\tag{7}
$$

## 4.3. Constraint-aware Trade-off

To enable constraint-aware trade-off during optimization, we first construct a user-specific sustainability penalty based on the learned sustainability representation:

$$
\begin{array} { r } { p _ { u , r } = \pmb { \alpha } _ { u } ^ { \top } ( \mathbf { z } _ { r } - \lambda _ { u } ) , } \end{array}\tag{8}
$$

where α and λ is learnable parameters. This term corresponds to the Lagrangian penalty in the task formulation, we incorporate it explicitly into the training objective to enforce sustainability constraints.

Sustainability-aware Penalty Loss. The sustainability loss is defined from the above penalty term:

$$
\mathcal { L } _ { \mathrm { s u s } } = \mathbb { E } _ { ( u , r ) } p _ { u , r } .\tag{9}
$$

Minimizing this term encourages the model to reduce violations of user-specific sustainability constraints.

Preference Learning Objective. Preference learning is optimized by jointly modeling rating accuracy and ranking consistency. Specifically, we minimize a rating objective

that fits normalized explicit feedback,

$$
\mathcal { L } _ { \mathrm { r a t i n g } } = \mathbb { E } _ { ( u , r ) } \left( \hat { y } _ { u , r } - y _ { u , r } ^ { \prime } \right) ^ { 2 } ,\tag{10}
$$

together with a ranking objective that enforces correct relative ordering of candidate recipes,

$$
\mathcal { L } _ { \mathrm { r a n k } } = \mathbb { E } _ { ( u , r ^ { + } , r ^ { - } ) } \left[ - \log \sigma \big ( \hat { y } _ { u , r ^ { + } } - \hat { y } _ { u , r ^ { - } } \big ) \right] ,\tag{11}
$$

where $( r ^ { + } , r ^ { - } )$ denotes a positive–negative recipe pair sampled from user interaction history.

Overall Objective. In addition to the above objectives, we apply $\ell _ { 2 }$ regularization to constraint-related parameters to ensure stable optimization. The final training objective is defined as:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { r a t i n g } } + \beta _ { 1 } \mathcal { L } _ { \mathrm { r a n k } } + \beta _ { 2 } \mathcal { L } _ { \mathrm { s u s } } + \beta _ { 3 } \| \Theta \| _ { 2 } ^ { 2 } , } \end{array}\tag{12}
$$

where $\beta _ { 1 } , \beta _ { 2 } , \beta _ { 3 } \in [ 0 , 1 ]$ are hyperparameters, and $\lVert \Theta \rVert _ { 2 } ^ { 2 }$ denotes $\ell _ { 2 }$ regularization.

## 5. Construction of Dataset

To support the development and evaluation of sustainable diet recommendation, we constructed a comprehensive dataset SusDiet that aligns recipes with multi-dimensional sustainability indicators. These indicators are derived from authoritative sources covering nutritional adequacy, environmental impact, economic cost, and animal welfare. By further aligning these recipes with large-scale user interaction data, this dataset enables the study of personalized decision-making under sustainability considerations.

## 5.1. Data Collection and Processing

We construct a recipe dataset by integrating recipes from multiple public sources, including recipeDB (Batra et al., 2020), Yummly-28k (Min et al., 2016), Yummly-66k (Min et al., 2017), WorldCuisines (Winata et al., 2025) and World Wide Recipe (Magomere et al., 2025). Each recipe is characterized by a standardized ingredient list and a country label. Specifically, we parse the recipe text by segmenting it into individual ingredient phrases. Each phrase is structured into a standardized triplet consisting of quantity, unit, and ingredient name. Ingredient information is unified through LLM-assisted cleaning and processing. Country labels are standardized according to the United Nations M49 classification<sup>1</sup>. Based on this, these recipes are associated with authoritative sustainability indicators obtained from external databases covering multiple dimensions, including nutritional adequacy, environmental impacts (Clark et al.,

Table 1. Summary statistics of the SusDiet
<table><tr><td>Category</td><td>Count</td></tr><tr><td>Recipe</td><td>149,036</td></tr><tr><td>Ingredients</td><td>7,340</td></tr><tr><td>Units</td><td>64</td></tr><tr><td>Ingredient Phrases</td><td>1,527,618</td></tr><tr><td>Countries</td><td>131</td></tr><tr><td>User</td><td>179,869</td></tr><tr><td>Interaction</td><td>744,138</td></tr></table>

2022), economic cost <sup>2</sup>, and animal welfare<sup>3</sup>.

## 5.2. LLM-Based Ingredient Structuring and Indicator Computation

To transform unstructured ingredient descriptions into a unified structured representation and to enable alignment with food product categories defined in sustainability indicator sources, we adopt an LLM-based processing method inspired by (Zhang et al., 2024a). Specifically, we employ the large language model GPT to perform component phrase analysis, quantity estimation, and sustainability indicator mapping. Ingredient phrases are decomposed into structured components of quantity, unit, and ingredient name. Ingredient weights are estimated based on the parsed quantity–unit information. In addition, each ingredient name is semantically mapped to the corresponding product category defined in the sustainability indicator databases, enabling the retrieval and computation of sustainability metrics.

We conduct human evaluation to assess the reliability of GPT outputs for these three tasks. Results show that 97%, 91%, and 94% of the samples are rated as acceptable for component phrase analysis, weight estimation, and sustainability indicator mapping, respectively. Additional details are provided in Appendix B.

## 5.3. User–Recipe Interaction Integration and Dataset Statistics

To facilitate personalized modeling, we align the constructed recipe dataset with user–recipe interaction data containing explicit ratings. These interactions are sourced from HUMMUS (Bolz et al.¨ , 2023) and the dataset introduced by (Majumder et al., 2019). We link user interactions to the specific recipes in our recipe dateset through name matching and metadata verification, resulting in a set of clean user–recipe–rating tuples.

The final dataset integrates recipes, standardized ingredients, multi-dimensional sustainability indicators, country labels, and user interactions into a unified dataset. Table 1 summarizes the key statistics of the dataset.

## 6. Empirical Evaluations

## 6.1. Experiment Settings

We evaluate the proposed framework from two aspects: how well the recommended recipes match user preferences, and how sustainable the recommended diets are across multiple dimensions, including nutritional adequacy, environmental impact, economic cost, and animal welfare. We consider a personalized top-K recipe recommendation task. For each user, the model ranks a set of candidate recipes and returns the top-K results. We report results for K = 5, 10, 20.

Baselines. Our task is closely related to general recommendation and food domain recommendation, requiring a diverse set of representative baseline methods. Accordingly, we use several widely adopted general recommendation models, including KNN(Wang et al., 2006), ICLRec(Chen et al., 2022), and MSSR(Lin et al., 2024), as general-purpose baselines. In addition, we use food recommendation methods, HAFR(Gao et al., 2020) and FGCN(Gao et al., 2022), that exploit food-related structures and relations as foodspecific baselines. Furthermore, we employ GRAPE(Jing et al., 2025) as a sustainability-oriented recommendation baseline, which accounts for sustainability indicators in the recommendation process. All baseline methods are implemented using their original settings to ensure a fair comparison. Further details are provided in Appendix C.

Evaluation metrics. Cultural acceptability is captured through learned user preferences, which we evaluate using ranking-based metrics including NDCG@K and Recall@K. Higher values indicate better cultural acceptability. For the other sustainability dimensions (nutritional adequacy, environmental impact, economic cost, and animal welfare), we compute the average indicator values of the top-K recommended recipes for each user. For these metrics, higher values indicate better outcomes for nutrition, while lower values indicate better outcomes for environment, price, and animal welfare.

## 6.2. Overall Results

Table 2 reports the overall performance of different methods under various Top-K settings, evaluated in terms of recommendation accuracy and multiple sustainability indicators.

Recommendation accuracy. Our method achieves competitive NDCG and Recall compared with state-of-the-art baselines, obtaining the best or near-best results across several cutoff values. This demonstrates that our approach remains effective in recommending recipes that align well

Table 2. Performances of different methods for Top-K recommendation. The best results are bold, and the second-best are underlined. ↑ indicates that higher is better, ↓ indicates that lower is better. Origin refers to the indicator values computed from users’ historically interacted recipes.
<table><tr><td>Top-K</td><td>Metrics</td><td>Origin</td><td>KNN</td><td>ICLRec</td><td>MSSR</td><td>HAFR</td><td>FGCN</td><td>GRAPE</td><td>Ours</td></tr><tr><td rowspan="6">N=5</td><td>NDCG</td><td></td><td>0.0051</td><td>0.0119</td><td>0.0125</td><td>0.0058</td><td>0.0111</td><td>0.0105</td><td>0.0120</td></tr><tr><td>Recall</td><td></td><td>0.0055</td><td>0.0162</td><td>0.0169</td><td>0.0075</td><td>0.0129</td><td>0.0152</td><td>0.0171</td></tr><tr><td>Nutrition ↑</td><td>38.7363</td><td>34.9750</td><td>34.5969</td><td>35.6965</td><td>34.5918</td><td>38.7869</td><td>39.5836</td><td>40.7221</td></tr><tr><td>Environment ↓</td><td>70.3760</td><td>67.6003</td><td>60.5654</td><td>63.4417</td><td>58.4069</td><td>62.7795</td><td>70.9837</td><td>69.9764</td></tr><tr><td>Economy ↓</td><td>7.3932</td><td>8.3118</td><td>4.9420</td><td>6.8017</td><td>6.2700</td><td>5.6742</td><td>5.2464</td><td>4.9258</td></tr><tr><td>Animal Welfare ↓</td><td>12.1631</td><td>16.1255</td><td>3.5907</td><td>9.7362</td><td>12.1456</td><td>6.3145</td><td>3.5327</td><td>2.7954</td></tr><tr><td rowspan="6">N=10</td><td>NDCG</td><td></td><td>0.0056</td><td>0.0125</td><td>0.0133</td><td>0.0075</td><td>0.0129</td><td>0.0115</td><td>0.0136</td></tr><tr><td>Recall</td><td></td><td>0.0075</td><td>0.0242</td><td>0.0211</td><td>0.0127</td><td>0.0186</td><td>0.0199</td><td>0.0223</td></tr><tr><td>Nutrition↑</td><td>38.7363</td><td>34.9288</td><td>35.1929</td><td>35.9286</td><td>35.6104</td><td>37.8545</td><td>39.5434</td><td>39.1429</td></tr><tr><td>Environment↓</td><td>70.3760</td><td>67.1324</td><td>59.8825</td><td>64.0748</td><td>59.8473</td><td>60.7506</td><td>69.5837</td><td>67.8675</td></tr><tr><td>Economy ↓</td><td>7.3932</td><td>8.3767</td><td>5.2925</td><td>6.8401</td><td>6.3471</td><td>5.7333</td><td>5.1543</td><td>4.8684</td></tr><tr><td>Animal Weifare ↓</td><td>12.1631</td><td>16.5685</td><td>4.4564</td><td>9.9498</td><td>11.9069</td><td>5.1571</td><td>5.2465</td><td>2.9186</td></tr><tr><td rowspan="6">N=20</td><td>NDCG</td><td></td><td>0.0062</td><td>0.0170</td><td>0.0181</td><td>0.0098</td><td>0.0147</td><td>0.0131</td><td>0.0159</td></tr><tr><td>Recall</td><td></td><td>0.0100</td><td>0.0340</td><td>0.0312</td><td>0.0212</td><td>0.0273</td><td>0.0265</td><td>0.0316</td></tr><tr><td>Nutrition↑</td><td>38.7363</td><td>34.9248</td><td>35.3663</td><td>35.9742</td><td>35.9525</td><td>37.1848</td><td>37.2563</td><td>37.5927</td></tr><tr><td>Environment↓</td><td>70.3760</td><td>66.1098</td><td>59.5418</td><td>64.1614</td><td>61.5087</td><td>64.2924</td><td>68.7837</td><td>66.2894</td></tr><tr><td>Economy ↓</td><td>7.3932</td><td>8.4511</td><td>5.6958</td><td>6.8710</td><td>6.4426</td><td>6.4037</td><td>6.8643</td><td>4.9445</td></tr><tr><td>Animal Welfare ↓</td><td>12.1631</td><td>17.0370</td><td>5.6385</td><td>10.3917</td><td>11.2196</td><td>7.8085</td><td>4.6847</td><td>3.0255</td></tr></table>

with user preferences.

Sustainability performance. Our method exhibits favorable results across multiple indicators. It consistently obtains high Nutrition scores while maintaining low Economy and Animal Welfare values, suggesting nutritionally improved, affordable, and welfare-aware recommendations. Although it does not achieve the lowest Environment score among all methods, we observe a common trade-off: recommendations with better nutritional quality often incur higher environmental impact. In contrast, our model produces a more balanced set of recommendations, avoiding extreme degradations in any single dimension. Moreover, it consistently outperforms historical user interactions across all sustainability dimensions.

Additional LLM-based baselines. We also compared against two additional LLM-based methods, Phi-3(Abdin et al., 2024) and KERL(Mohbat & Zaki, 2025), which are discussed in Appendix D.1. In brief, Phi-3 underperforms on all metrics, and while KERL achieves slightly higher recommendation accuracy, it falls significantly behind in environmental, economic, and animal welfare dimensions due to the lack of explicit constraint modeling.

Sustainability prediction performance. Table 3 presents the predictive performance of the multitask sustainability learning model across four sustainability dimensions. The model achieves low MAE and RMSE values together with consistently high $R ^ { 2 }$ scores, indicating accurate and stable prediction of sustainability indicators. These results demonstrate that the multitask learning formulation can reliably encode diverse sustainability attributes, providing high-quality sustainability estimates that support the overall recommendation performance.

Table 3. Predictive performance of the multitask sustainability learning model across different dimensions.
<table><tr><td>Task</td><td>MAE</td><td>RMSE</td><td> $R ^ { 2 }$ </td></tr><tr><td>Nutrition</td><td>0.055</td><td>0.087</td><td>0.992</td></tr><tr><td>Environment</td><td>0.052</td><td>0.083</td><td>0.993</td></tr><tr><td>Economy</td><td>0.063</td><td>0.103</td><td>0.989</td></tr><tr><td>Animal Welfare</td><td>0.026</td><td>0.112</td><td>0.987</td></tr></table>

## 6.3. Ablation Study

We conduct ablation experiments to examine the contribution of the three core modules in the proposed framework. As shown in Table 4, removing the MoE-based Recipe Sustainability Representation leads to consistent degradation in sustainability indicators, while ranking accuracy is only slightly affected, highlighting the importance of explicitly encoding sustainability information at the recipe level. Removing the interest attention mechanism in the Preference Learning results in clear drops in both NDCG and Recall, accompanied by worse sustainability performance, indicating that modeling heterogeneous user interests is essential for both personalization quality and stable sustainability outcomes. Finally, when the Personalized Constraint-aware Trade-off is removed and recommendations are generated solely based on user preferences, sustainability indicators deteriorate substantially, demonstrating that preference-only recommendation tends to amplify unsustainable historical patterns, and that the proposed trade-off module is crucial for balancing user preferences with multi-dimensional sustainability constraints.

Table 4. Ablation of MoE, interest attention, and constraint-aware trade-off modeling on Top-20 recommendation performance. MoE denotes the mixture-of-experts sustainability encoder in recipe representation; IntAttn denotes the interest attention mechanism in the preference learning module; Constraint denotes the personalized constraint-aware decision mechanism. Best results are highlighted in bold. ↑ indicates that higher is better, ↓ indicates that lower is better.
<table><tr><td>MoE</td><td>IntAttn</td><td>Constraint</td><td>NDCG</td><td>Recall</td><td>Nutrition ↑</td><td>Environment ↓</td><td>Economy ↓</td><td>Animal Welfare ↓</td></tr><tr><td>X</td><td>√</td><td>√</td><td>0.0155</td><td>0.0321</td><td>35.3628</td><td>68.6534</td><td>6.2455</td><td>4.9456</td></tr><tr><td>√</td><td>X</td><td>V</td><td>0.0110</td><td>0.0259</td><td>36.6532</td><td>68.2435</td><td>5.3234</td><td>4.3553</td></tr><tr><td>√</td><td>√</td><td>X</td><td>0.0173</td><td>0.0367</td><td>34.2454</td><td>70.8643</td><td>7.2345</td><td>5.9553</td></tr><tr><td>V</td><td>√</td><td>V</td><td>0.0159</td><td>0.0316</td><td>37.5927</td><td>66.2894</td><td>4.9445</td><td>3.0255</td></tr></table>

Table 5. Item Coverage at different top-k settings.
<table><tr><td>Metric</td><td>History</td><td>Preference-only</td><td>Ours</td></tr><tr><td>IC@5</td><td>0.61</td><td>0.42</td><td>0.36</td></tr><tr><td>IC@10</td><td>0.61</td><td>0.55</td><td>0.45</td></tr><tr><td>IC@20</td><td>0.61</td><td>0.72</td><td>0.55</td></tr></table>

Table 6. Distribution of learned user-specific constraint thresholds $\lambda _ { u } .$
<table><tr><td></td><td>mean</td><td>std</td><td>P10</td><td>P50</td><td>P90</td></tr><tr><td> $\lambda ^ { \mathrm { { N u t r . } } }$ </td><td>20.93</td><td>4.10</td><td>15.19</td><td>22.36</td><td>24.83</td></tr><tr><td> $\lambda ^ { \mathrm { E n v i . } }$ </td><td>67.25</td><td>47.40</td><td>46.11</td><td>63.63</td><td>92.83</td></tr><tr><td> $\lambda ^ { \mathrm { E c o n . } }$ </td><td>9.14</td><td>4.10</td><td>6.94</td><td>7.84</td><td>12.09</td></tr><tr><td> $\lambda ^ { \mathrm { A n i m . } }$ </td><td>9.37</td><td>20.80</td><td>3.80</td><td>5.10</td><td>13.65</td></tr></table>

## 6.4. Diversity of recommendations.

We further evaluate recommendation diversity via Item Coverage (IC)@k, defined as the proportion of unique recipes recommended to all users among the top-k results relative to the entire recipe set. Table 5 compares the coverage of the user interaction history, the preference-only model (i.e., our model without the constraint term), and our full method.

Incorporating sustainability constraints leads to a moderate reduction in coverage. However, our method maintains substantial coverage that remains comparable to the scale of observed user interactions. While the constraints prioritize more sustainable recipes, they do not strictly exclude less sustainable ones. Such items are typically ranked lower but may still be recommended.

## 6.5. Personalized Constraint Thresholds

To verify that the learned sustainability constraints are personalized, we analyze the distribution of user-specific constraint thresholds $\dot { \lambda } _ { u } = [ \lambda _ { u } ^ { \mathrm { N u t r . } } , \lambda _ { u } ^ { \mathrm { E n v i . } } , \lambda _ { u } ^ { \mathrm { E c o n . } } , \lambda _ { u } ^ { \mathrm { A i n i m . } } ]$ across all users. Table 6 reports the mean, standard deviation, and selected percentiles (P10, P50, P90) for each dimension.

Table 7. User case study comparing average sustainability indicators computed over historical interactions and Top-5 recommendations. ↑ indicates that higher is better, ↓ indicates that lower is better. Hist., Pref., and Ours denote the average indicators of the user’s historical interactions, preference-only recommendations, and our recommendations, respectively. Nutr., Envi., Econ., and Anim. denote nutrition, environment, economy, and animal welfare, respectively.
<table><tr><td>List</td><td>Nutr. ↑</td><td>Envi. ↓</td><td>Econ. ↓</td><td>Anim. ↓</td></tr><tr><td>Hist.</td><td>37.51</td><td>86.78</td><td>9.97</td><td>18.70</td></tr><tr><td>Pref.</td><td>38.06</td><td>87.22</td><td>10.34</td><td>14.78</td></tr><tr><td>Ours</td><td>39.69</td><td>71.59</td><td>6.99</td><td>12.10</td></tr></table>

The spread between percentiles (e.g., from P10 to P90) indicates that $\lambda _ { u }$ varies significantly across users for each sustainability dimension. For instance, the environmental constraint threshold ranges from 46.11 to 92.83, reflecting heterogeneous tolerance levels for environmental impact. These results demonstrate that the learned constraints are personalized and capture diverse user-specific sustainability boundaries.

## 6.6. User Case Study

To qualitatively examine the recommendation behavior of the proposed method, we present a case study on a representative user U1, which is shown in Table 7. Compared with both historical behavior and the preference-only baseline, our method substantially improves sustainability outcomes across all dimensions, while maintaining alignment with the user’s original preferences.

## 7. Discussion

In this work, we reformulate personalized sustainable diet recommendation as a constraint-aware decision-making problem, addressing the gap between population-level sustainability guidance and individual dietary behavior. By modeling sustainability as learnable, user-specific constraints, our framework better reflects how individuals make real-world food choices. The proposed approach integrates

MoE-based recipe sustainability representation learning, multi-interest preference modeling, and personalized constraint learning within a unified optimization objective. Different from the works on AI-driven climate sustainability (Zhang et al., 2025; Bulian et al., 2024), this work addresses the domain of sustainable diets. It is also distinguished from sustainable food (Thomas et al., 2025) by its scope: whereas sustainable food focuses on the environmental integrity of production, a sustainable diet represents a consumer-centric paradigm that encompasses broader eating patterns. Experiments on the large-scale SusDiet dataset constructed in this work demonstrate that our method consistently promotes more sustainable recommendations while maintaining competitive personalization accuracy. These results highlight the potential of constraintaware decision modeling as a principled foundation for sustainability-oriented recommendation systems and individualized dietary interventions.

Beyond the algorithmic performance, the broader significance of this framework lies in its potential to support more personalized and adaptive sustainable diet recommendation strategies. By modeling sustainability as a learnable userspecific constraint rather than a fixed global objective, the proposed approach enables recommendation systems to better accommodate heterogeneous dietary preferences while incorporating sustainability-oriented objectives. This formulation provides a possible pathway toward integrating sustainable considerations into everyday food recommendation scenarios. From a broader societal perspective, this work also provides a computational framework that may support future data-driven sustainable diet initiatives and public health strategies. Compared with traditional one-sizefits-all dietary guidelines, personalized recommendation frameworks offer the opportunity to account for differences in cultural background, dietary habits, and individual preference patterns when promoting sustainability-oriented food choices. In addition, the incorporation of sustainable indicators into recommendation modeling creates new possibilities for analyzing the interaction between dietary behavior and sustainability objectives at scale. Such capabilities may contribute to future research and applications related to sustainable consumption, precision nutrition, and environmentally informed dietary policy design.

Limitations. Despite the promising results, several limitations remain. First, the learned sustainability constraints are inferred from historical interaction data and may therefore only partially reflect users’ true sustainability tolerance boundaries. Although the interaction data are derived from online recipe rating platforms rather than real-world purchase records and are thus less directly constrained by factors such as price or accessibility, exposure bias and unobserved socio-economic factors may still influence observed interactions. Future work could address these issues through exposure-aware recommendation strategies and richer contextual modeling. Second, the current framework also assumes static preferences and constraints, whereas both may evolve over time as users gain experience or awareness. In addition, while SusDiet provides a comprehensive set of sustainability indicators, these metrics are subject to uncertainty and simplification, which may propagate into the learned representations. Finally, our evaluation is limited to offline experiments, and the real-world behavioral impact of constraint-aware recommendations, such as long-term adherence, trust, and potential unintended effects, remains to be validated through online deployment or user-centered studies. Addressing these limitations will be crucial for further advancing constraint-aware decision modeling and realizing its full potential in supporting sustainable and responsible individual decision-making.

## Acknowledgements

This work was supported in part by the Beijing Natural Science Foundation (JQ24021) and the National Natural Science Foundation of China (62472411 and 62125207).

## Impact Statement

Our work is motivated by the broader challenges of sustainability, public health, and responsible consumption, with the goal of supporting more informed and personalized dietary decision-making. By promoting preference-aware food choices, this research explores pathways to facilitate the broader adoption of sustainable eating habits, which can collectively contribute to mitigating greenhouse gas emissions and land-use pressures while supporting human health. At the consumer level, the emphasis on gradual, adaptive adjustments respects individual autonomy and cultural practices, potentially reducing the resistance typically associated with rigid, one-size-fits-all interventions. Beyond individual choices, the insights derived from this multi-dimensional analysis can help policy-makers better understand how sustainability targets interact with heterogeneous population preferences, thereby supporting more adaptive, evidencebased policy design. At the same time, while data-driven consumption management can drive sustainability, it introduces vital concerns regarding fairness, transparency, and data governance. Managing these unintended societal effects requires careful oversight to ensure equitable transitions.

## References

Abdin, M. I., Ade Jacobs, S., Awan, A. A., Aneja, J., Awadallah, A., Hassan Awadalla, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., Benhaim, A., Bilenko, M.,

Bjorck, J., Bubeck, S., Cai, M., Mendes, C. C. T., Chen, W., Chaudhary, V., Chopra, P., Giorno, A. D., de Rosa, G., Dixon, M., Eldan, R., Iter, D., Goswami, A., Gunasekar, S., Haider, E., Hao, J., Hewett, R. J., Huynh, J., Javaheripi, M., Jin, X., Kauffmann, P., Karampatziakis, N., Kim, D., Khademi, M., Kurilenko, L., Lee, J. R., Lee, Y. T., Li, Y., Liang, C., Liu, W., Lin, X. E., Lin, Z., Madan, P., Mitra, A., Modi, H., Nguyen, A., Norick, B., Patra, B., Perez-Becker, D., Portet, T., Pryzant, R., Qin, H., Radmilac, M., Rosset, C., Roy, S., Saarikivi, O., Saied, A., Salim, A., Santacroce, M., Shah, S., Shang, N., Sharma, H., Song, X., Ruwase, O., Wang, X., Ward, R., Wang, G., Witte, P., Wyatt, M., Xu, C., Xu, J., Xu, W., Yadav, S., Yang, F., Yang, Z., Yu, D., Zhang, C., Zhang, C., Zhang, J., Zhang, L. L., Zhang, Y., Zhang, Y., and Zhou, X. Phi-3 technical report: A highly capable language model locally on your phone. Technical Report MSR-TR-2024-12, Microsoft, August 2024. URL https://www.microsoft.co m/en-us/research/publication/phi-3-t echnical-report-a-highly-capable-lan guage-model-locally-on-your-phone/.

Bajzelj, B., Laguzzi, F., and Rˇ o¨os, E. The role of fats in¨ the transition to sustainable diets. The Lancet Planetary Health, 5(9):e644–e653, 2021.

Batra, D., Diwan, N., Upadhyay, U., Kalra, J. S., Sharma, T., Sharma, A. K., Khanna, D., Marwah, J. S., Kalathil, S., Singh, N., et al. Recipedb: a resource for exploring recipes. Database, 2020:baaa077, 2020.

Biesbroek, S., Kok, F. J., Tufford, A. R., Bloem, M. W., Darmon, N., Drewnowski, A., Fan, S., Fanzo, J., Gordon, L. J., Hu, F. B., et al. Toward healthy and sustainable diets for the 21st century: Importance of sociocultural and economic considerations. Proceedings of the National Academy ofSciences, 120(26):e2219272120, 2023.

Bolz, F., Nurbakova, D., Calabretto, S., Gerl, A., Brunie,¨ L., and Kosch, H. Hummus: a linked, healthiness-aware, user-centered and argument-enabling recipe data set for recommendation. In Proceedings of the 17th ACM Conference on Recommender Systems, pp. 1–11, 2023.

Bulian, J., Schafer, M. S., Amini, A., Lam, H., Ciaramita,¨ M., Gaiarin, B., Hubscher, M. C., Buck, C., Mede, N. G.,¨ Leippold, M., and Strauß, N. Assessing large language models on climate information. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 5580–5608. PMLR, 2024.

Chaudhary, A. and Krishna, V. Country-specific sustainable diets using optimization algorithm. Environmental Science & Technology, 53(13):7694–7703, 2019.

Chen, Y., Liu, Z., Li, J., McAuley, J., and Xiong, C. Intent contrastive learning for sequential recommendation. In Proceedings ofthe ACM Web Conference 2022, pp. 2172– 2182, 2022.

Clark, M., Springmann, M., Rayner, M., Scarborough, P., Hill, J., Tilman, D., Macdiarmid, J. I., Fanzo, J., Bandy, L., and Harrington, R. A. Estimating the environmental impacts of 57,000 food products. Proceedings of the National Academy of Sciences, 119(33):e2120584119, 2022.

Colglazier, W. Sustainable development agenda: 2030. Science, 349(6252):1048–1050, 2015.

de Costa, R., Ferrara, I., Toplak, M., Alam, A., Bowie, R., and Burnett, A. Behavioural insights and environmental sustainability: Key findings and policy implications from a systematic review. Journal ofEnvironmental Management, 390:126118, 2025.

Elliott, P. S., Devine, L. D., Gibney, E. R., and O’Sullivan, A. M. What factors influence sustainable and healthy diet consumption? a review and synthesis of literature within the university setting and beyond. Nutrition Research, 126:23–45, 2024.

Fernqvist, F., Spendrup, S., and Tellstrom, R. Understanding¨ food choice: A systematic review of reviews. Heliyon, 10(12), 2024.

Gao, X., Feng, F., He, X., Huang, H., Guan, X., Feng, C., Ming, Z., and Chua, T.-S. Hierarchical attention network for visually-aware food recommendation. IEEE Transactions on Multimedia, 22(6):1647–1659, 2020. doi: 10.1109/TMM.2019.2945180.

Gao, X., Feng, F., Huang, H., Mao, X.-L., Lan, T., and Chi, Z. Food recommendation with graph convolutional network. Information Sciences, 584:170–183, 2022.

Gazan, R., Brouzes, C. M., Vieux, F., Maillot, M., Lluch, A., and Darmon, N. Mathematical optimization to explore tomorrow’s sustainable diets: a narrative review. Advances in Nutrition, 9(5):602–616, 2018.

Gibney, E. R. Healthy and sustainable diets defined. Nature Food, 6(5):426–427, May 2025. ISSN 2662-1355. doi: 10.1038/s43016-025-01162-7. URL https://doi. org/10.1038/s43016-025-01162-7.

Gonzalez, W., Monterrosa, E., Sutiyo, W., Noor, S., Khondker, R., Bipul, M., Wanjiru, L., Wekesa, L., and Deo, A. Do consumers consider environmental factors when making food choices? insights from indonesia, bangladesh, and kenya. Working Paper 53, Global Alliance for Improved Nutrition (GAIN), Geneva, Switzerland, 2025. URL https://doi.org/10.36072/wp.53.

Hak, T., Janou´ skovˇ a, S., and Moldan, B. Sustainable devel-´ opment goals: A need for relevant indicators. Ecological Indicators, 60:565–573, 2016.

He, X., Liu, Q., and Jung, S. The impact of recommendation system on user satisfaction: A moderated mediation approach. Journal of Theoretical and Applied Electronic Commerce Research, 19(1):448–466, 2024.

Heerschop, S., Cardinaals, R., Biesbroek, S., Kanellopoulos, A., Geleijnse, J., Van’t Veer, P., and Van Zanten, H. Designing sustainable healthy diets: Analysis of two modelling approaches. Journal of Cleaner Production, 475:143619, 2024.

Jing, J., Zhang, Y., and Miao, C. Bites of tomorrow: Personalized recommendations for a healthier and greener plate. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 11897–11905, 2025.

Lee, B. X., Kjaerulf, F., Turner, S., Cohen, L., Donnelly, P. D., Muggah, R., Davis, R., Realini, A., Kieselbach, B., MacGregor, L. S., et al. Transforming our world: implementing the 2030 agenda through sustainable development goal indicators. Journal of Public Health Policy, 37(Suppl 1):13–31, 2016.

Lin, X., Luo, J., Pan, J., Pan, W., Ming, Z., Liu, X., Huang, S., and Jiang, J. Multi-sequence attentive user representation learning for side-information integrated sequential recommendation. In Proceedings ofthe 17th ACM International Conference on Web Search and Data Mining, pp. 414–423, 2024.

Magomere, J., Ishida, S., Afonja, T., Salama, A., Kochin, D., Foutse, Y., Hamzaoui, I., Sefala, R., Alaagib, A., Dalal, S., et al. The world wide recipe: A communitycentred framework for fine-grained data collection and regional bias operationalisation. In Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency, pp. 246–282, 2025.

Majumder, B. P., Li, S., Ni, J., and McAuley, J. Generating personalized recipes from historical user preferences. arXiv preprint arXiv:1909.00105, 2019.

Merrigan, K., Griffin, T., Wilde, P., Robien, K., Goldberg, J., and Dietz, W. Designing a sustainable diet. Science, 350(6257):165–166, 2015.

Min, W., Jiang, S., Sang, J., Wang, H., Liu, X., and Herranz, L. Being a supercook: Joint food attributes and multimodal content modeling for recipe retrieval and exploration. IEEE Transactions on Multimedia, 19(5):1100– 1113, 2016.

Min, W., Bao, B.-K., Mei, S., Zhu, Y., Rui, Y., and Jiang, S. You are what you eat: Exploring rich recipe information for cross-region food analysis. IEEE Transactions on Multimedia, 20(4):950–964, 2017.

Mohbat, F. and Zaki, M. J. KERL: Knowledge-enhanced personalized recipe recommendation using large language models. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 19125–19141, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl- long.938. URL https: //aclanthology.org/2025.acl-long.938/.

Nguyen, T. T. T., Hetherington, J. B., O’Connor, P. J., and Malek, L. Sustainable food consumption: Sustainabilityconscious consumers do not reduce food waste but nutrition-conscious consumers do. Resources, Conservation and Recycling, 219:108296, 2025.

Organization, W. H. et al. Health and well-being and the 2030 agenda for sustainable development in the who european region: an analysis of policy development and implementation. report of the first survey to assess member states’ activities in relation to the who european region roadmap to implement the 2030 agenda for sustainable development. Technical report, World Health Organization. Regional Office for Europe, 2021.

Petersson, T., Secondi, L., Magnani, A., Antonelli, M., Dembska, K., Valentini, R., Varotto, A., and Castaldi, S. A multilevel carbon and water footprint dataset of food commodities. Scientific Data, 8(1):127, 2021.

Poore, J. and Nemecek, T. Reducing food’s environmental impacts through producers and consumers. Science, 360 (6392):987–992, 2018.

Qiao, G., Zhang, D., Zhang, N., Shen, X., Jiao, X., Lu, W., Fan, D., Zhao, J., Zhang, H., Chen, W., et al. Food recommendation towards personalized wellbeing. Trends in Food Science & Technology, pp. 104877, 2025.

Rockstrom, J., Steffen, W., Noone, K., Persson,¨ A., Chapin,<sup>˚</sup> F. S., Lambin, E. F., Lenton, T. M., Scheffer, M., Folke, C., Schellnhuber, H. J., et al. A safe operating space for humanity. Nature, 461(7263):472–475, 2009a.

Rockstrom, J., Steffen, W., Noone, K., Persson,¨ A.,<sup>˚</sup> Chapin III, F. S., Lambin, E., Lenton, T. M., Scheffer, M., Folke, C., Schellnhuber, H. J., et al. Planetary boundaries: exploring the safe operating space for humanity. Ecology and society, 14(2), 2009b.

Sachs, J. D. From millennium development goals to sustainable development goals. The Lancet, 379(9832):2206– 2211, 2012.

Springmann, M., Wiebe, K., Mason-D’Croz, D., Sulser, T. B., Rayner, M., and Scarborough, P. Health and nutritional aspects of sustainable diet strategies and their association with environmental impacts: a global modelling analysis with country-level detail. The Lancet Planetary Health, 2(10):e451–e461, 2018.

Steffen, W., Richardson, K., Rockstrom, J., Cornell, S. E.,¨ Fetzer, I., Bennett, E. M., Biggs, R., Carpenter, S. R., De Vries, W., De Wit, C. A., et al. Planetary boundaries: Guiding human development on a changing planet. Science, 347(6223):1259855, 2015.

Thomas, A., Yee, A., Mayne, A., Mathur, M., Jurafsky, D., and Gligoric, K. What can large language models do for sustainable food? In Proceedings of the 42nd International Conference on Machine Learning. PMLR, 2025.

Wang, J., De Vries, A. P., and Reinders, M. J. Unifying userbased and item-based collaborative filtering approaches by similarity fusion. In Proceedings of the 29th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 501–508, 2006.

Willett, W., Rockstrom, J., Loken, B., Springmann, M.,¨ Lang, T., Vermeulen, S., Garnett, T., Tilman, D., De-Clerck, F., Wood, A., et al. Food in the anthropocene: the eat–lancet commission on healthy diets from sustainable food systems. The Lancet, 393(10170):447–492, 2019.

Wilson, N., Cleghorn, C. L., Cobiac, L. J., Mizdrak, A., and Nghiem, N. Achieving healthy and sustainable diets: a review of the results of recent mathematical optimization studies. Advances in Nutrition, 10:S389–S403, 2019.

Winata, G. I., Hudi, F., Irawan, P. A., Anugraha, D., Putri, R. A., Yutong, W., Nohejl, A., Prathama, U. A., Ousidhoum, N., Amriani, A., et al. Worldcuisines: A massivescale benchmark for multilingual and multicultural visual question answering on global cuisines. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 3242–3264, 2025.

Yang, L., Hsieh, C.-K., Yang, H., Pollak, J. P., Dell, N., Belongie, S., Cole, C., and Estrin, D. Yum-me: a personalized nutrient-based meal recommender system. ACM Transactions on Information Systems (TOIS), 36(1):1–31, 2017.

Ye, B., Xiong, Q., Yang, J., Huang, Z., Huang, J., He, J., Liu, L., Xia, M., and Liu, Y. Adoption of regionspecific diets in china can help achieve gains in health and environmental sustainability. Nature Food, 5(9):764– 774, 2024.

Zhang, J., Wang, Z., Liu, W., Liu, X., and Zheng, Q. A unified approach to designing sequence-based personalized food recommendation systems: tackling dynamic user behaviors. International Journal of Machine Learning and Cybernetics, 14(9):2903–2912, 2023.

Zhang, L., Zhang, Y., Zhou, X., and Shen, Z. Greenrec: A large-scale dataset for green food recommendation. In Companion Proceedings of the ACM Web Conference 2024, pp. 625–628, 2024a.

Zhang, T., Williams, A. R., Wozny, P., Cohrs, K.-H., Ponse, K., Jiralerspong, M., Bengio, Y., and Zheng, S. Ai for global climate cooperation: Modeling global climate negotiations, agreements, and long-term cooperation in ricen. In Proceedings ofthe 42nd International Conference on Machine Learning. PMLR, 2025.

Zhang, Y., Zhou, X., Meng, Q., Zhu, F., Xu, Y., Shen, Z., and Cui, L. Multi-modal food recommendation using clustering and self-supervised learning. In Pacific Rim International Conference on Artificial Intelligence, pp. 269–281. Springer, 2024b.

## A. Extended Methodology

## A.1. Recipe Sustainability Representation Module

In this module, we learn a structured and sustainability-aware representation of recipes by jointly modeling ingredient composition and multiple recipe-level sustainability attributes. This stage produces a recipe embedding that captures compositional semantics and serves as a reusable representation for downstream constraint-aware decision-making mechanism.

## A.1.1. INGREDIENT TOKENIZATION AND INPUT REPRESENTATION

Each recipe r is described by a set of standardized ingredients and their corresponding quantities in grams, $\mathcal { G } _ { r } ~ =$ $\{ ( i _ { j } , q _ { j } ) \} _ { j = 1 } ^ { \tilde { N } _ { r } }$ , where $i _ { j }$ denotes a canonical ingredient token and $q _ { j } \in \mathbb { R } _ { + }$ denotes its quantity.

To enable neural sequence modeling, we construct a vocabulary over ingredient tokens by lowercasing all ingredient strings and filtering infrequent tokens below a minimum frequency threshold. Two special symbols are reserved to represent padding and unknown ingredients.

Since recipes contain a variable number of ingredients, each ingredient list is converted into a fixed-length sequence of length L via truncation or padding. Specifically, ingredient tokens are mapped to integer indices, quantities are aligned accordingly, and a binary mask is generated to indicate valid ingredient positions. As a result, each recipe is represented by

$$
\mathbf { t } _ { r } \in \mathbb { N } ^ { L } , \quad \mathbf { q } _ { r } \in \mathbb { R } ^ { L } , \quad \mathbf { m } _ { r } \in \{ 0 , 1 \} ^ { L } ,
$$

where $\mathbf { m } _ { r } [ j ] = 1$ if the j-th position corresponds to a real ingredient and 0 otherwise. This formulation preserves both ingredient identity and quantitative contribution while allowing efficient mini-batch training.

## A.1.2. INGREDIENT–QUANTITY FUSION

To jointly encode ingredient identity and quantity information, we embed each ingredient token and project its corresponding quantity into a continuous space. Formally, for position j,

$$
\begin{array} { r l } & { \mathbf { e } _ { r , j } = \operatorname { E m b e d } ( \mathbf { t } _ { r } [ j ] ) \in \mathbb { R } ^ { D } , } \\ & { } \\ & { \mathbf { u } _ { r , j } = W _ { q } \mathbf { q } _ { r } [ j ] + b _ { q } \in \mathbb { R } ^ { d _ { q } } . } \end{array}
$$

The two representations are concatenated and linearly projected to the model dimension:

$$
\mathbf { x } _ { r , j } = W _ { \mathrm { i n } } [ \mathbf { e } _ { r , j } ; \mathbf { u } _ { r , j } ] + b _ { \mathrm { i n } } \in \mathbb { R } ^ { D } .
$$

The resulting sequence $\mathbf { X } _ { r } = [ \mathbf { x } _ { r , 1 } , \ldots , \mathbf { x } _ { r , L } ]$ serves as the input to the recipe encoder.

## A.1.3. MIXTURE-OF-EXPERTS RECIPE ENCODER

To capture heterogeneous compositional patterns across recipes, we adopt a Mixture-of-Experts (MoE) architecture consisting of $E = 4$ Transformer-based experts. Each expert independently encodes the ingredient sequence:

$$
\mathbf { H } _ { r } ^ { ( e ) } = \mathrm { T r a n s f o r m e r } ^ { ( e ) } ( \mathbf { X } _ { r } , \mathrm { p a d d i n g . m a s k } = ( \mathbf { m } _ { r } = 0 ) ) ,
$$

where padding positions are excluded from self-attention.

An expert-specific recipe representation is then obtained via masked mean pooling:

$$
\mathbf { p } _ { r } ^ { \left( e \right) } = \frac { \sum _ { j = 1 } ^ { L } \mathbf { m } _ { r } [ j ] \mathbf { H } _ { r } ^ { \left( e \right) } [ j ] } { \sum _ { j = 1 } ^ { L } \mathbf { m } _ { r } [ j ] + \varepsilon } \in \mathbb { R } ^ { D } .
$$

A gating network computes mixture weights based on a summary of the input sequence:

$$
\bar { \mathbf { x } } _ { r } = \frac { \sum _ { j = 1 } ^ { L } \mathbf { m } _ { r } [ j ] \mathbf { x } _ { r , j } } { \sum _ { j = 1 } ^ { L } \mathbf { m } _ { r } [ j ] + \varepsilon } ,
$$

$$
\mathbf { g } _ { r } = \mathrm { s o f t m a x } ( W _ { 2 } \mathrm { R e L U } ( W _ { 1 } \bar { \mathbf { x } } _ { r } ) ) .
$$

The final recipe embedding is computed as a gated combination of expert outputs:

$$
\mathbf { z } _ { r } = \mathrm { L a y e r N o r m } \left( W _ { g } \left( \sum _ { e = 1 } ^ { 4 } \mathbf { g } _ { r } [ e ] \mathbf { p } _ { r } ^ { \left( e \right) } \right) + b _ { g } \right) \in \mathbb { R } ^ { D } .
$$

This MoE formulation enables different experts to specialize in distinct ingredient compositions while maintaining a unified representation space.

## A.1.4. MULTI-TASK SUSTAINABILITY PREDICTION HEADS

Given the recipe embedding ${ \bf z } _ { r } ,$ , we jointly predict multiple sustainability-related attributes using task-specific regression heads. For each task $k \in \{ 1 , \ldots , K \}$ ,

$$
\hat { s } _ { r } ^ { ( k ) } = W _ { 2 k } \phi ( W _ { 1 k } \mathbf { z } _ { r } + b _ { 1 k } ) + b _ { 2 k } ,
$$

where $\phi ( \cdot )$ denotes a non-linear activation function. All tasks share the same recipe encoder but employ independent prediction heads, enabling shared representation learning with task-specific specialization. In our default setting, $K = 4$

## A.1.5. TARGET TRANSFORMATION AND NORMALIZATION

The raw sustainability targets exhibit heavy-tailed distributions and heterogeneous scales. To stabilize optimization, we apply a sequence of transformations to each target value $s _ { r } ^ { ( k ) }$ . For sustainability dimensions where higher values are preferred, we multiply the corresponding scores by −1.

First, a signed logarithmic transformation compresses extreme values while preserving directionality:

$$
s _ { r } ^ { ( k ) } = \mathrm { s i g n } ( s _ { r } ^ { ( k ) } ) \log ( 1 + | s _ { r } ^ { ( k ) } | ) .
$$

Second, transformed values are clipped using training-set quantiles to mitigate the influence of outliers. Finally, z-score normalization is applied:

$$
s _ { r } ^ { ( k ) } = \frac { s _ { r } ^ { ( k ) } - \mu } { \sigma } ,
$$

where $\mu$ and σ are computed on training recipes only.

## A.1.6. LEARNED TASK-WEIGHTED MULTI-TASK OBJECTIVE

We train the recipe encoder and sustainability heads using a multi-task regression objective with learned task weighting. For a mini-batch of size B, the per-task mean squared error is computed as:

$$
\ell _ { k } = \frac { 1 } { B } \sum _ { b = 1 } ^ { B } \Big ( \hat { s } _ { r _ { b } } ^ { ( k ) } - s _ { r _ { b } } ^ { ( k ) } \Big ) ^ { 2 } .
$$

Following uncertainty-based weighting, each task k is associated with a learnable parameter $v _ { k }$ . The overall objective is:

$$
\mathcal { L } _ { \mathrm { t a s k } } = \sum _ { k = 1 } ^ { K } \left( \exp ( - \boldsymbol { v } _ { k } ) \boldsymbol { \ell } _ { k } + \boldsymbol { v } _ { k } \right) .
$$

This formulation allows the model to automatically balance gradients across tasks with different noise levels and scales, without manual tuning of task weights.

## A.2. Preference Learning Module via Multi-interest Mechanism

This module learns preference-oriented user and recipe representations from sequential interaction data for personalized recommendation. User dietary preferences are inherently evolutionary and dynamic. To capture such chronological patterns and multi-view culinary tastes, we adopt a multi-interest self-attention mechanism.

## A.2.1. USER-SIDE PREFERENCE ENCODING

For a user u, let $\mathcal { H } _ { u } = \{ ( r _ { i } , y _ { u , r _ { i } } ) \} _ { i = 1 } ^ { N _ { u } }$ denote the historical interaction set, where $r _ { i }$ is an interacted recipe, $y _ { u , r _ { i } }$ is the observed rating, and $N _ { u }$ is the maximum number of interactions.

Pre-trained Structural Embedding Injection. Each historical recipe $r _ { i }$ in the sequence is mapped into a continuous semantic space via the pre-trained embedding table E : $\mathbf { E } _ { u }$

$$
\begin{array} { r } { { \bf e } _ { r _ { i } } ^ { ( 0 ) } = { \bf E } _ { u } [ r _ { i } ] , } \end{array}
$$

where ${ \bf e } _ { r _ { i } } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$

Sequential Token Construction. To capture the chronological progression of the user’s dietary journey, we inject sequential order directly into the item state. We represent each step by combining the structural recipe vector with a learnable position bias token:

$$
\begin{array} { r } { \mathbf { h } _ { i } = \mathbf { e } _ { r _ { i } } ^ { ( 0 ) } + \mathbf { p } _ { i } , } \end{array}
$$

where $\mathbf { p } _ { i } \in \mathbb { R } ^ { d }$ denotes the bias embedding for the i-th sequence position.

Self-attention over user history. To model dependencies among interactions and extract multiple latent preference patterns, we apply self-attention on $\{ \mathbf { h } _ { i } \} _ { i = 1 } ^ { N _ { u } }$ :

$$
\mathbf { Q } _ { i } ^ { u } = \mathbf { W } _ { Q } ^ { u } \mathbf { h } _ { i } , \quad \mathbf { K } _ { i } ^ { u } = \mathbf { W } _ { K } ^ { u } \mathbf { h } _ { i } , \quad \mathbf { V } _ { i } ^ { u } = \mathbf { W } _ { V } ^ { u } \mathbf { h } _ { i } ,
$$

$$
a _ { i j } ^ { u } = \frac { \exp \biggl ( \frac { ( \mathbf { Q } _ { i } ^ { u } ) ^ { \top } \mathbf { K } _ { j } ^ { u } } { \sqrt { d } } \biggr ) } { \sum _ { k = 1 } ^ { N _ { u } } \exp \biggl ( \frac { ( \mathbf { Q } _ { i } ^ { u } ) ^ { \top } \mathbf { K } _ { k } ^ { u } } { \sqrt { d } } \biggr ) } , \qquad \tilde { \mathbf { h } } _ { i } = \sum _ { j = 1 } ^ { N _ { u } } a _ { i j } ^ { u } \mathbf { V } _ { j } ^ { u } .
$$

## A.2.2. RECIPE-SIDE PREFERENCE REPRESENTATION

For a target candidate recipe r, we construct a multi-view preference profile by incorporating its main identity along with core cultural and composition dimensions.

We project the target recipe into its main ID space, standardized ingredient token space, and cultural region space via independent global lookup tables:

$$
\begin{array} { r } { \mathbf { v } _ { r , i d } = \mathbf { E } _ { \mathrm { i t e m } } [ r ] , \qquad \mathbf { v } _ { r , \mathrm { i n g } } = \mathbf { E } _ { \mathrm { i n g } } [ r ] , \qquad \mathbf { v } _ { r , \mathrm { r e g } } = \mathbf { E } _ { \mathrm { r e g } } [ r ] , } \end{array}
$$

where $\mathbf { E } _ { \mathrm { i t e m } } , \mathbf { E } _ { \mathrm { i n g } }$ , and $\mathbf { E } _ { \mathrm { r e g } }$ denote the learnable main identity, ingredient feature, and region feature embedding matrices, respectively, all mapping into $\mathbb { R } ^ { d }$ . We also apply self-attention and pooling to model interactions among these embeddings to obtain $\tilde { \mathbf { v } } _ { r , i d } , \tilde { \mathbf { v } } _ { r , \mathrm { i n g } }$ and $\tilde { \mathbf { v } } _ { r , \mathrm { r e g } } .$ . Finally, the three representations are concatenated and projected through a linear transformation to obtain the final recipe representation $\mathbf { e } _ { r }$

## A.2.3. MULTI-INTEREST ATTENTION MECHANISM

Given user history tokens $\tilde { \mathbf { h } } _ { i }$ , we extract multi-faceted and context-aware latent preference patterns through a multi-layer Transformer block driven by multi-head self-attention.

We project the target recipe embedding as a query and the history tokens as keys and values:

$$
\mathbf { Q } _ { r } ^ { c } = \mathbf { W } _ { Q } ^ { c } \tilde { \mathbf { v } } _ { r , i d } , \quad \mathbf { K } _ { i } ^ { c } = \mathbf { W } _ { K } ^ { c } \tilde { \mathbf { h } } _ { i } , \quad \mathbf { V } _ { i } ^ { c } = \mathbf { W } _ { V } ^ { c } \tilde { \mathbf { h } } _ { i } ,
$$

$$
\beta _ { i , r } = \frac { \exp \left( \frac { ( \mathbf { Q } _ { r } ^ { c } ) ^ { \top } \mathbf { K } _ { i } ^ { c } } { \sqrt { d } } \right) } { \sum _ { j = 1 } ^ { N _ { u } } \exp \left( \frac { ( \mathbf { Q } _ { r } ^ { c } ) ^ { \top } \mathbf { K } _ { j } ^ { c } } { \sqrt { d } } \right) } .
$$

The user representation is:

$$
\mathbf { e } _ { u , i d } = \sum _ { i = 1 } ^ { N _ { u } } \beta _ { i , r } \mathbf { V } _ { i } ^ { c } .
$$

Similarly, $\tilde { \mathbf { v } } _ { r , \mathrm { i n g } }$ and $\tilde { \mathbf { v } } _ { r , \mathrm { r e g } }$ are processed through the same mechanism to obtain representation $\mathbf { e } _ { u , \mathrm { i n g } }$ and representation $\mathbf { e } _ { u , \mathrm { r e g } } ,$ respectively. Finally, the three representations are concatenated and projected through a linear transformation to obtain the final recipe representation $\mathbf { e } _ { u }$

## A.2.4. PREFERENCE SCORING

Finally, the preference score $\hat { y } _ { u , r }$ for user u and recipe r is computed by:

$$
\hat { y } _ { u , r } = \langle \mathbf { e } _ { u } , \mathbf { e } _ { r } \rangle .
$$

## B. Supplementary Information on Datasets

## B.1. Recipe Collection and Integration

We collected recipe data from multiple public sources to ensure diversity and broad cultural representativeness. Primary sources include:

• recipeDB(Batra et al., 2020): A large public recipe database scraped directly from its official website, containing 118,171 recipes, serving as the backbone of our corpus.

• Yummly-28k(Min et al., 2016) & Yummly-66k(Min et al., 2017): These popular platform datasets provide 27,638 and 66,615 recipes respectively, enriching recipe variety.

• WorldCuisines(Winata et al., 2025) & World Wide Recipe(Magomere et al., 2025): These smaller, focused datasets contain 2,414 and 765 recipes, with explicit country/region labels that enhance the cultural dimension of our data.

All selected datasets provide explicit country or region level labels, which allows cultural and geographic information to be retained as a specific attribute. Each recipe entry contains a recipe name, a country or regional tag, and an ingredient list expressed in form of natural language with associated quantity expressions. Unfortunately, these sources exhibit substantial heterogeneity in formatting conventions, ingredient naming practices, and unit systems. After merging all sources, we perform a systematic cleaning and deduplication process based on recipe titles, ingredient compositions, and metadata consistency. The resulting unified recipe corpus assigns a unique identifier to each recipe, which serves as the primary key for linking ingredient-level indicators and user interaction data in later stages.

## B.2. Ingredient Parsing and Standardization

The ingredient descriptions in the raw recipe data are expressed as unstructured text, often containing linguistic variations, modifiers, and ambiguous references, and the numerical representation is diverse. Direct string matching is therefore insufficient for reliably aligning such descriptions with structured ingredient databases required for sustainability analysis. To address this, we leverage GPT to assist in the accurate standardization required for sustainability analysis.

Ingredient Phrase Parsing. Firstly, we divide the description text of the ingredients into phrases. Following the strategy of (Zhang et al., 2024a), we employ GPT to parse each phrase into a set of structured words. Each phrase is decomposed into three components: a numerical quantity, a unit name, and an ingredient name (e.g., “1; cup; milk”, “0.5; pound; pork”). Detailed prompts and parsing rules used for this step are provided in Figure 2.

Ingredient Name Standardization. The parsed ingredient names still contain numerous synonyms and variants (e.g., ”tomato”, ”tomatoes”, ”cherry tomato”). We manually map them to a unified, canonical ingredient vocabulary. Through iterative cleaning and merging, we derived a final vocabulary of 7,340 distinct standardized ingredients.

Sustainability indicator Mapping. To enable the lookup and computation of sustainability-related indicators, each ingredient phrase is mapped to standardized ingredient categories defined by external authoritative databases. These databases cover multiple sustainability dimensions, including nutritional adequacy, environmental impact, economic cost, and animal welfare. Detailed descriptions of the databases for each sustainability dimension are provided as follows:

![](images/197ccef878bdb64e8534546e911c2f6e20c240bb6d2409071f6e0f0bf899a97c.jpg)  
Figure 2. Prompt of component phrase analysis

• nutritional adequacy. We adopt standardized nutritional adequacy scores from (Clark et al., 2022), which provide nutrition indicators for 63 food product categories.

• Environmental Impact. Environmental indicators are derived from multiple life-cycle assessment studies. Specifically, (Poore & Nemecek, 2018) reports greenhouse gas emissions, land use, acidification, and eutrophication metrics for 43 food categories; (Clark et al., 2022) extends environmental impact estimates to 63 standardized food categories; and (Petersson et al., 2021) provides fine-grained carbon and water footprint data covering 324 and 320 food product categories, respectively. In our recommendation model, we use the environmental part of (Clark et al., 2022) as an environmental indicator.

• Economic Cost. Food price information is derived from the USDA Economic Research Service Purchase to Plate National Average Prices (PP-NAP) dataset<sup>4</sup>, which reports retail-level cost estimates across 14,744 food product categories.

• Animal Welfare. Animal welfare indicators are obtained from Faunalytics<sup>5</sup>, which provides welfare impact estimates covering 97 food product categories.

This semantic alignment is performed using GPT as a matching and disambiguation tool. Given an ingredient phrase and a candidate set of standardized categories, GPT identifies the most semantically appropriate match. In particular, due to the large number of categories in the economic cost database, we adopt a two-stage mapping strategy. Ingredients are first mapped to one of 197 coarse-grained categories, and subsequently assigned to a fine-grained category among the full set o 14,744 economic cost categories. Compared to rule-based or string-matching approaches, this strategy robustly handles lexical variability and ensures consistent alignment across heterogeneous data sources. Detailed prompts and parsing rules used in this step are provided in Figure 3.

## B.3. Unit Normalization and Quantity Conversion

A major challenge in aggregating ingredient-level sustainability indicators arises from the use of diverse and often informal quantity and unit expressions in recipes, ranging from volumetric units (e.g., cups, tablespoons) to qualitative descriptors (e.g., “a pinch”, “a handful”). Moreover, many units appear in multiple lexical forms (e.g., lb vs. pound), further complicating quantitative analysis.

Unit Unification. We first normalized all parsed unit terms into 64 distinct standard units. For example, ”tbsp”, ”T”, and ”tablespoons” are all mapped to ”tablespoon”.

Quantity Estimation. Subsequently, we convert all ingredient quantities into a unified measurement unit, ”grams”, to enable quantitative aggregation. Specifically, we construct unit–ingredient pairs from the previous stage, resulting in 20,105 distinct combinations. For each unit–ingredient pair, we apply GPT to estimate the corresponding mass in grams based on common culinary conventions and ingredient-specific density assumptions. The exact prompting strategy are detailed in Figure 4.

![](images/7751a98f2b3631f6e4c2de4a83977fec329ea147f281f555e67421a2114d402b.jpg)  
Figure 3. Prompt of sustainability indicator mapping

![](images/0be7d40aa5f199be193a26f2827b809bcfa4a80e3768a68e6f301198f5bee432.jpg)  
Figure 4. Prompt of quantity estimation

After this step, each recipe is represented as a set of ingredients with estimated masses in grams, providing a consistent and quantitative foundation for computing recipe-level sustainability indicators.

## B.4. Sustainability Metric Computation

With normalized ingredient representations, we compute recipe-level sustainability indicators by aggregating ingredientlevel indicators obtained from the datasets described in Appendix B.2. Recipe-level indicators are computed as linear combinations of ingredient-level indicators weighted by ingredient mass. As a result, each recipe is associated with a comprehensive vector of sustainability indicators spanning all dimensions.

## B.5. Geographic and cultural lable standardization

Country and region tags associated with recipes were standardized by mapping them to a unified taxonomy aligned with the United Nations M49 standard<sup>6</sup>.

Recipes were ultimately tagged with one of 131 distinct country labels.

Table 8. Statistics of the constructed user–recipe interaction dataset
<table><tr><td>Dataset</td><td>Recipe</td><td>User</td><td>Interaction</td></tr><tr><td>(Bölz et al., 2023)</td><td>507,335</td><td>302,412</td><td>1.916,424</td></tr><tr><td>(Majumder et al., 2019)</td><td>231,637</td><td>226,570</td><td>1,132,367</td></tr><tr><td>SusDiet(Ours)</td><td>149,036</td><td>179,869</td><td>744,138</td></tr></table>

## B.6. User–Recipe Interaction Alignment

To support personalized recommendation and preference modeling, we align the constructed recipe dataset with user–recipe interaction data. We utilize interaction from the HUMMUS(Bolz et al.¨ , 2023) dataset and an additional dataset reported in (Majumder et al., 2019), which contain 1,916,424 and 1,132,367 raw interaction records, respectively. Each record includes a username, a recipe name, and an explicit rating

We link these interactions to our constructed recipe corpus through recipe name matching and metadata verification. After alignment and filtering, we retain 744,138 valid interaction records. Each interaction is represented as a tuple indicating a user-assigned rating for a specific recipe, linked via the shared recipe identifier. The statistical results are shown in Table 8.

## B.7. Human Evaluation of GPT-Based Processing

All GPT-assisted steps, including component phrase analysis, sustainability indicator mapping, quantity estimation, are subject to human evaluation. For each step, we randomly sampled 500 instances and asked human annotators to score the GPT output based on correctness and semantic sufficiency, with a score range of 1-4 for each task. The resulting scores are used to assess the reliability of GPT-based preprocessing.

![](images/b7fd1578531d8ffb4e985ef8831a442947f2cb0d551faf56d49b0158c0d9a002.jpg)

![](images/4829c092d3eaf1174562def5dff6cd198e3f03a373de1a30c8d8709e33ce8a23.jpg)  
Figure 5. Human evaluation results of GPT performance across (A) Component Phrase Analysis, (B) Sustainability Indicator Mapping, and (C) Quantity Estimation. Left: Score distributions for each task, with the horizontal line denoting the mean human score. Right: Percentage composition of scores for each task.

The experimental results demonstrate that the GPT achieves consistently strong performance across the three tasks, with average human evaluation scores of 3.776, 3.664, and 3.352, respectively. The score distributions aggregated from all evaluators are shown in Figure 5. Across the three tasks, 97%, 94%, and 91% of the samples were rated as acceptable (scores between 3 and 4), indicating that the majority of GPT generated outputs meet practical quality requirements. These results suggest that the GPT can reliably produce usable outputs for component phrase analysis, sustainability indicator mapping, and quantity estimation.

Furthermore, we compared two alternative approaches for ingredient weight estimation: (1) providing the raw ingredient phrase directly to GPT, and (2) a two-stage approach in which the phrase is first parsed into quantity, unit, and ingredient components before weight estimation. We conducted a manual evaluation using a 4-point scale on 200 samples for each approach, resulting in average scores of 2.85 and 3.36, respectively. The results indicate that the raw ingredient phrases often contain extraneous or conflated information, which negatively impacts the GPT’s weight estimation performance. Therefore, we adopted the two-stage parsing-and-estimation approach for the final method.

## B.7.1. HUMAN EVALUATION CRITERIA

For the component phrase analysis task, a score of 4 indicates that all three elements (quantity, unit, and ingredient phrase) are correctly identified with accurate boundaries. A score of 3 denotes that the core information of all three elements is correct, though minor presentational imperfections may exist. A score of 2 reflects that at least one element is correctly identified, while the others contain notable errors. A score of 1 signifies that most or all information is incorrect, resulting in a failure to properly segment any of the elements or in incoherent or irrelevant output.

For the sustainability indicator mapping task, a score of 4 represents an exact match to an item in the reference list or a match to a standardized synonym. A score of 3 corresponds to a match to the correct broader category, with minor differences that do not affect practical usability. A score of 2 indicates a match to a related but not entirely correct item, where clear errors are present yet remain within the same general category. A score of 1 is assigned when the match is to a completely unrelated category or when the matching attempt fails entirely.

For the quantity estimation task, a score of 4 is given when the numerical estimate is either exact or falls within a reasonable standard range, consistent with common sense and culinary norms. A score of 3 is assigned for estimates with minor deviation that remain within an acceptable margin. A score of 2 reflects estimates with substantial deviation from the expected value, while a score of 1 is reserved for estimates with severe deviation that clearly contradict common sense.

## B.8. Dataset Analysis

We further conduct exploratory analyses to characterize the dataset in terms of ingredient distributions, sustainability indicator ranges, and geographic coverage. As shown in Figure 7, the ingredient frequency distribution exhibits a clear long-tail structure: a limited set of widely used ingredients appears with high frequency, while a large number of ingredients occur sparsely. This pattern reflects realistic dietary practices and ensures strong coverage of common food items.

As illustrated in Figure 6, the dataset demonstrates broad geographic coverage, with recipes spanning a diverse set of countries and regions, albeit with uneven representation across cuisines. Beyond structural statistics, the dataset integrates multiple dimensions of sustainability indicators, including environmental impact, nutritional adequacy, economic cost, and animal welfare. These indicators exhibit substantial variation across recipes and regions, highlighting diverse sustainability trade-offs present in real-world dietary choices. Collectively, these characteristics indicate that the dataset captures rich diversity across ingredients, cultures, and sustainability dimensions, while also revealing potential imbalances that should be considered in the design and evaluation of downstream modeling tasks.

## C. Experimental Setup and Implementation Details

## C.1. Implementation Details

In our experiment, we select the learning rate from the set {0.0005, 0.001, 0.005}, the embedding size d from {64, 128, 256}, and the batch size from {64, 128, 256}. We tune the number of attention heads from {2, 4, 8}. For the loss function tradeoffs, we select hyperparameter α = 0.5 the ranking weight $\beta _ { 1 }$ from [0.1, 1.0], the sustainability penalty weight $\beta _ { 2 }$ from [0.1, 2.0], and the L2 regularization coefficient $\beta _ { 3 }$ from $[ 1 0 ^ { - 6 } , 1 0 ^ { - 4 } ]$

## C.2. Baselines

We now describe the baseline methods used to evaluate the proposed model.

• KNN(Wang et al., 2006): This method reformulates memory-based collaborative filtering within a probabilistic framework that fuses user-based and item-based similarities, treating each observed rating as a weighted predictor of missing ratings. By jointly leveraging similarities across users and items with background smoothing, it improves robustness to data sparsity compared to traditional neighborhood-based approaches.

• ICLRec(Chen et al., 2022): This approach introduces a latent intent variable for sequential recommendation and learns user intent distributions from unlabeled interaction sequences via clustering, which are integrated into recommendation through intent-aware contrastive self-supervised learning.

• MSSR(Lin et al., 2024): This method models user behavior by jointly learning representations from item sequences and multiple side-information sequences using a multi-sequence integrated attention mechanism that captures both intra- and inter-sequence dependencies. It further aligns heterogeneous user representations via contrastive learning and incorporates side information of candidate items for more comprehensive preference modeling.

![](images/5b219ca18af077f30fbddbda49df0f9035c3dac0411493a933713b462ba5ee7d.jpg)  
Figure 6. Statistical map of the distribution of the number of recipes in different countries

• HAFR(Gao et al., 2020): This approach employs a hierarchical attention network to jointly model user–recipe interactions and multimodal recipe content. Ingredient-level and component-level attentions adaptively fuse different modalities conditioned on user preferences.

• FGCN(Gao et al., 2022): This method models food recommendation by constructing a heterogeneous graph over ingredient–ingredient, ingredient–recipe, and recipe–user relations, and applies multi-layer graph convolution with information propagation to capture high-order connectivity and enhance user and recipe representations.

• GRAPE(Jing et al., 2025): This approach models green food recommendation by jointly capturing users’ sequential preferences over items and multiple sustainability indicators through integrated self- and cross-attention, and introduces dedicated green loss functions to balance recommendation accuracy with the promotion of more sustainable food choices.

## D. Additional Experimental Results

## D.1. Additional LLM-based Baselines

To further evaluate LLM-based methods on our dataset, we constructed the dataset following a pipeline inspired by prior work: (i) recipes, ingredients, and user interactions were processed to form structured knowledge graphs and user behavior sequences; (ii) each recipe graph contains title, ingredients, nutrition information, and tags derived from country; (iii) from user sequences, we generated QA-style samples with constraints on sustainability attributes.

We evaluated both Phi-3-mini-128k (Abdin et al., 2024) and the KERL model (Mohbat & Zaki, 2025) on this dataset.   
Preliminary results at Top-10 recommendations are shown in Table 9.

![](images/37e193ad3cdbda681a925729702001744a1f14719d485aca67f217eca22595f5.jpg)  
Figure 7. Frequency Distribution of the Top 20 Most Common Ingredients in the Recipe Dataset

Table 9. Performance of additional LLM-based baselines (Top-10).
<table><tr><td>Metric</td><td>Phi-3</td><td>KERL</td><td>Ours</td></tr><tr><td>NDCG@10</td><td>0.0035</td><td>0.0160</td><td>0.0136</td></tr><tr><td>Recall@10</td><td>0.0087</td><td>0.0243</td><td>0.0223</td></tr><tr><td>Nutrition@10↑</td><td>48.5782</td><td>43.5833</td><td>39.1429</td></tr><tr><td>Environment@10 ↓</td><td>125.0952</td><td>76.8827</td><td>67.8675</td></tr><tr><td>Economy@ 10↓</td><td>7.5879</td><td>9.5769</td><td>4.8684</td></tr><tr><td>Animal Welfare@10↓</td><td>4.3617</td><td>20.1999</td><td>2.9186</td></tr></table>

As shown, Phi-3 performs poorly on recommendation metrics and struggles to balance user preferences with multiple sustainability dimensions, likely due to its limited understanding of these aspects. KERL slightly outperforms our method on NDCG@10 and Recall@10, but falls significantly behind on Environment, Economy, and Animal Welfare metrics. This may be because KERL does not enforce strict constraints, instead allowing greater trade-offs in favor of user preference dimensions. KERL performs relatively well on Nutrition, likely because it was originally trained on datasets containing nutritional information. Note that due to time and computational constraints, these models were used without fine-tuning.

## D.2. Comparing with LLM

We conducted experiments using Llama 3-8B to generate sustainability representations for all recipes. The generated representations were used as item-side features in most baselines (ICLRec, MSSR, HAFR, FGCN, GRAPE) and also in our framework (replacing the sustainability representation stage while keeping the constraint mechanism). Results on Top-20 recommendations are shown in Table 10.

The results show that our method consistently outperforms all LLM-based variants in recommendation accuracy and sustainability performance, indicating that LLMs alone are not sufficient for this task. When directly using LLMs for sustainability representation (Ours(LLM)), direct LLM outputs lack structured and reliable modeling, and cannot explicitly capture fine-grained sustainability dimensions in a single step, leading to worse performance than our method. This observation is consistent with Appendix B.7: in the data construction stage, we compared direct LLM generation with a structured two-stage approach, where the latter significantly outperformed the former (3.36 vs. 2.85 in human evaluation). This demonstrates that LLM outputs are inherently less reliable without additional learning. Besides, replacing the sustainability representations in conventional baselines with LLM-generated features (baseline(LLM)) does not consistently improve performance and even degrades some metrics, further indicating that LLMs are not well-suited for learning reliable sustainability representations alone.

Table 10. Top-20 recommendation performance of different methods using LLM for sustainability representations. The best results are in bold. ↑ indicates that higher is better, ↓ indicates that lower is better.
<table><tr><td>Metrics</td><td>ICLRec(LLM)</td><td>MSSR(LLM)</td><td>HAFR(LLM)</td><td>FGCN(LLM)</td><td>GRAPE(LLM)</td><td>Ours(LLM)</td><td>Ours</td></tr><tr><td>NDCG</td><td>0.0124</td><td>0.0154</td><td>0.0092</td><td>0.0105</td><td>0.0137</td><td>0.0148</td><td>0.0159</td></tr><tr><td>Recall</td><td>0.0248</td><td>0.0201</td><td>0.0159</td><td>0.0224</td><td>0.0264</td><td>0.0293</td><td>0.0316</td></tr><tr><td>Nutrition ↑</td><td>33.6432</td><td>36.0231</td><td>36.6532</td><td>32.6542</td><td>36.4352</td><td>37.1464</td><td>37.5927</td></tr><tr><td>Environment ↓</td><td>63.3256</td><td>66.7433</td><td>64.6534</td><td>71.5634</td><td>68.6534</td><td>67.6534</td><td>66.2894</td></tr><tr><td>Economy ↓</td><td>6.6743</td><td>7.0234</td><td>7.7643</td><td>8.4534</td><td>6.4764</td><td>6.3462</td><td>4.9445</td></tr><tr><td>Animal Welfare ↓</td><td>10.4234</td><td>9.3454</td><td>12.3244</td><td>12.2345</td><td>8.9753</td><td>5.7434</td><td>3.0255</td></tr></table>

## D.3. Ablation Study on the Sustainability Dimensions

To further demonstrate the necessity of modeling each sustainability dimension as an explicit constraint, we conduct ablation studies by removing the constraint for each dimension individually while keeping the rest of the framework unchanged. The results on Top-20 recommendations are summarized below. The results show that removing any single constraint consistently degrades performance on the corresponding sustainability dimension, and in some cases also negatively affects other dimensions. In contrast, the full model achieves the best overall balance across all dimensions.

Table 11. Ablation study on the sustainability dimensions. ↑/↓ indicate higher/lower is better.
<table><tr><td></td><td>Nutr.↑</td><td>Envi.↓</td><td>Econ.↓</td><td>Anim.↓</td></tr><tr><td>w/o Nutr.</td><td>35.3454</td><td>66.1432</td><td>4.9464</td><td>3.1235</td></tr><tr><td>w/o Envi.</td><td>37.6464</td><td>70.0443</td><td>6.5433</td><td>4.7532</td></tr><tr><td>w/o Econ.</td><td>36.1467</td><td>67.4213</td><td>7.0745</td><td>4.9742</td></tr><tr><td>w/o Anim.</td><td>37.4642</td><td>68.6434</td><td>6.3134</td><td>8.4523</td></tr><tr><td>Full Model</td><td>37.5927</td><td>66.2894</td><td>4.9445</td><td>3.0255</td></tr></table>

## D.4. Analysis on High-Conflict Scenarios

To address concerns regarding high-conflict scenarios, we performed a quantitative boundary test by stratifying users into Top 10% (least sustainable / high-conflict) and Bottom 10% (most sustainable) groups based on their historical environmental scores. Table 12 reports the recommendation accuracy and sustainability indicators for each group under the preference-only baseline and our full model.

Table 12. Performance on high-conflict (Top 10%) and low-conflict (Bottom 10%) user groups. History indicators are computed from users’ historically interacted recipes.
<table><tr><td rowspan="2">Metric</td><td colspan="2">History</td><td colspan="2">Preference-only</td><td colspan="2">Ours</td></tr><tr><td>Top 10%</td><td>Bottom 10%</td><td>Top 10%</td><td>Bottom 10%</td><td>Top 10%</td><td>Bottom 10%</td></tr><tr><td>NDCG@10</td><td>一</td><td></td><td>0.0123</td><td>0.0142</td><td>0.0122</td><td>0.0135</td></tr><tr><td>Recall@10</td><td></td><td></td><td>0.0256</td><td>0.0256</td><td>0.0230</td><td>0.0244</td></tr><tr><td>Nutrition@10</td><td>57.2219</td><td>22.4635</td><td>37.0862</td><td>34.5305</td><td>40.7569</td><td>35.3851</td></tr><tr><td>Environment@10</td><td>164.5703</td><td>19.6472</td><td>71.4178</td><td>54.7939</td><td>69.5293</td><td>46.2580</td></tr><tr><td>Economy@10</td><td>9.9947</td><td>4.3291</td><td>7.0616</td><td>6.1799</td><td>5.1901</td><td>4.0410</td></tr><tr><td>Animal Welfare@10</td><td>13.8657</td><td>5.1481</td><td>9.5334</td><td>7.3426</td><td>3.6893</td><td>2.3606</td></tr></table>

For the Top 10% group, whose preferences are most opposed to sustainability requirements, our model achieved improve-

ments across all four sustainability dimensions compared to the preference-only baseline, with only a marginal loss in recommendation accuracy. Furthermore, the Bottom 10% group exhibited even greater gains in sustainability indicators. This demonstrates that the sustainability constraints remain effective even at extreme boundaries, allowing the framework to nudge the most unsustainable users toward better alternatives without significantly sacrificing accuracy. These findings confirm the robustness and effectiveness of our approach in resolving high-conflict dietary trade-offs.

## D.5. User Case Study

Table 13. Detailed recipe-level results for User U1 with pork- and beef-oriented preferences. ↑/↓ indicate higher/lower is better.
<table><tr><td>Method</td><td>Recipe</td><td>Nutrition ↑</td><td>Environment↓</td><td>Economy ↓</td><td>Animal Welfare ↓</td></tr><tr><td colspan="8">Historical interactions</td></tr><tr><td>History</td><td>Spicy Beef Noodle Soup</td><td>39.23</td><td>102.88</td><td>13.83</td><td></td><td>30.90</td></tr><tr><td>History</td><td>Mapo Tofu With Crispy Chinese Sausage</td><td>45.46</td><td></td><td>64.26</td><td>14.65</td><td>21.13</td></tr><tr><td>History</td><td>Kung Pao Chicken</td><td>35.14</td><td>89.33</td><td></td><td>7.63</td><td>12.43</td></tr><tr><td>History</td><td>Braised Pork Rice</td><td>38.91</td><td>106.96</td><td></td><td>9.32</td><td>18.88</td></tr><tr><td>History</td><td>Fried Rice w/ Egg &amp; Ham</td><td>28.80</td><td>70.47</td><td></td><td>4.41</td><td>10.17</td></tr><tr><td colspan="7">Preference-only recommendations (Top-5)</td></tr><tr><td>Preference-only</td><td>Double-Cooked Pork</td><td>43.65</td><td>72.42</td><td></td><td>5.93</td><td>10.45</td></tr><tr><td>Preference-only</td><td>Beef Pepper Stir-fry</td><td>50.21</td><td></td><td>108.13</td><td>8.42</td><td>15.34</td></tr><tr><td>Preference-only</td><td>Hotpot Lamb Slices</td><td>26.11</td><td></td><td>124.70</td><td>16.61</td><td>20.42</td></tr><tr><td>Preference-only</td><td>Crispy Fried Chicken</td><td>32.48</td><td></td><td>62.21</td><td>10.80</td><td>14.91</td></tr><tr><td>Preference-only</td><td>Pork Ribs Soup</td><td>37.83</td><td>68.68</td><td></td><td>9.95</td><td>12.77</td></tr><tr><td colspan="7">Our recommendations (Top-5)</td></tr><tr><td>Ours</td><td>Spicy Beef Noodle Soup (Lean Cut)</td><td>28.82</td><td>61.22</td><td></td><td>5.65</td><td>16.25</td></tr><tr><td>Ours</td><td>Chili Garlic Pork with Eggplant (Reduced Oil)</td><td>45.84</td><td>69.37</td><td></td><td>4.26</td><td>7.10</td></tr><tr><td>Ours</td><td>Dan Dan Noodles with Lean Pork</td><td>33.92</td><td></td><td>107.84</td><td>5.18</td><td>6.43</td></tr><tr><td>Ours</td><td>Stir-fry Chicken &amp; Broccoli</td><td>39.61</td><td></td><td>73.90</td><td>15.82</td><td>18.24</td></tr><tr><td>Ours</td><td>Tomato Beef &amp; Tofu Stew (Light)</td><td>50.24</td><td></td><td>45.60</td><td>4.01</td><td>12.49</td></tr></table>