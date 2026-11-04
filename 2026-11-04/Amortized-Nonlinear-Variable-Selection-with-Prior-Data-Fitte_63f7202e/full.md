# Amortized Nonlinear Variable Selection with Prior-Data Fitted Networks

Jin Zhu and Junxian Zhu<sup>∗</sup> University of Birmingham, University of Science and Technology of China

## Abstract

Variable selection is an important task for interpreting potentially complex systems in fields such as biology, business, and sociology. The importance of variable selection has attracted considerable attention in statistics and machine learning. In this paper, we are mainly interested in model-agnostic variable selection for identifying variables that have general nonlinear dependence on the response. We propose MambaPFN, short for model-agnostic Markov blanket selection with prior-data fitted networks, which is built upon the Transformer, a building block of many popular large language models. Specifically, we consider a functional-mapping perspective on variable-selection learning, in which the Transformer is treated as a learnable mapping from a dataset to its relevant variables. To learn this mapping, MambaPFN introduces three new components: (i) a faithful, rich, and scalable scheme for generating pretraining data, (ii) an objective that targets each variable’s posterior inclusion probability, and (iii) a modified Transformer architecture tailored to variable selection. Once pretrained, MambaPFN is applied to new datasets without parameter updates. Empirically, MambaPFN achieves desirable variable-selection performance in simulation studies and exhibits a scaling law—more pretraining datasets and larger models can continuously improve its performance. Applied to a Facebook comments dataset, MambaPFN selects the fewest variables and achieves the highest computational eficiency and predictive performance, improving upon the best baselines by about 20%.

Keywords: Variable selection, Markov blanket, Transformer, Prior-data fitted networks

## 1 Introduction

Variable selection aims to identify the most relevant and informative variables in a dataset while removing irrelevant, noisy, or redundant ones. It is central to many scientific and societal applications because it can provide insight into the underlying system. For example, in biomedicine, selecting informative variables from high-dimensional omics and singlecell data supports reliable biomarker discovery (Hédou et al. 2024). In materials science, selecting compact physicochemical descriptors can reveal structure–property relationships and guide catalyst design (Han et al. 2024). Furthermore, variable selection can improve robustness and generalization while reducing efective model complexity (Li et al. 2024, Liao et al. 2024). Therefore, variable selection remains a fundamental and active problem in machine learning and data-driven science.

Literature review. Because of the importance of variable selection, dozens of methods have been developed over the last few decades; we review representative ones here. Existing variable selection methods can be divided mainly into two categories: model-based and model-agnostic methods.

Model-based variable selection imposes explicit structure on the regression function. For linear models, sparsity is induced through convex or non-concave penalties (Tibshirani 1996, Fan & Li 2001, Zhang 2010, Shen et al. 2012) or iterative algorithms (Blumensath & Davies 2009, Huang et al. 2018, Zhu et al. 2020). These methods have been extended to more complicated models, such as models in which predictors have a known group structure (Yuan & Lin 2006), models in which hierarchical penalties enforce heredity among main and interaction efects (Bien et al. 2013, Lim & Hastie 2015), single-index models (Kong & Xia 2007), and sparse additive models (Ravikumar et al. 2009, Meier et al. 2009, Huang et al. 2010). These methods yield interpretable estimators and, under suitable conditions, strong selection guarantees. Their targets and guarantees, however, are tied to the assumed

structure.

Model-agnostic variable selection is typically implemented via statistics-based methods and machine-learning (ML)-based methods.

• Statistics-based methods. Notable examples are knockof procedures, which construct negative-control variables and use them to control false discoveries in variable selection (Candès et al. 2018, Romano et al. 2020, Bates et al. 2021, Zhu et al. 2021, Spector & Janson 2022, Ren et al. 2023). Kernel-based methods instead identify variables through gradients of conditional dependence measures in reproducing kernel Hilbert spaces or through conditional covariance operators (Yamada et al. 2014, Yang et al. 2016, Chen et al. 2017, Climente-González et al. 2019, Poignard & Yamada 2020).

• ML-based methods select relevant variables using ML algorithms such as neural networks. For example, because neural networks have a strong capacity to approximate arbitrary nonlinear relationships, various methods have been developed to select variables that have complicated relationships with the response (Yoon et al. 2019, Yamada et al. 2020, Lemhadri et al. 2021, Chen et al. 2021, Luo & Halabi 2025, Jiang et al. 2024, Yang et al. 2024).

## 1.1 Motivation: The Bitter Lesson

Sutton (2019) presents the view that, in the long run, general approaches that scale with available computational power tend to outperform those based on domain-specific understanding because they are better able to take advantage of the falling cost of computation over time. This principle was proposed in 2019 and has gained increasing acceptance as the capacity of large language models has grown, e.g., ChatGPT (OpenAI 2026), Claude (Anthropic 2026), and DeepSeek (DeepSeek-AI 2026), enabling them to solve increasingly challenging problems.

However, none of the aforementioned methods implements this viewpoint because they are not amortized across datasets; in other words, given a new dataset, they need to refit a model, whether it is a neural network or a linear model, to perform variable selection. Furthermore, non-amortized methods are static and cannot exhibit a scaling-law efect across datasets, under which the performance of an amortized method would reasonably improve if more diverse datasets were used for training. At the same time, neural-network-based variable-selection methods, particularly those without distributional assumptions on the data, can be time-consuming (e.g., Chen et al. 2021). From the perspective of model deployment, because the variable-selection step aims to reduce the variable set before fitting a model, a time-consuming selection procedure is less desirable in applications.

On the other hand, methods that adopt this viewpoint are achieving increasingly strong performance through amortized learning. For example, prior-data fitted networks (PFNs) apply this amortized-learning paradigm: a model is pretrained once and then makes predictions on new datasets without retraining (Müller et al. 2022). Subsequent methods have improved their predictive accuracy, scalability, and coverage of real-data tasks (Hollmann et al. 2023, Grinsztajn et al. 2025, Zhang et al. 2025, Ma et al. 2026, Qu et al. 2026), together with a complementary theoretical understanding of these methods (Ma et al. 2025). Amortization has also moved beyond point prediction to tasks such as causal-structure identification (Lorch et al. 2022, Peng et al. 2026, Thompson et al. 2026), (conditional) average treatment efect estimation (Balazadeh Meresht et al. 2025, Robertson et al. 2025, Sauter et al. 2025), interval estimation (Ye et al. 2026), and Bayesian optimization (Müller et al. 2023, Rakotoarison et al. 2026).

These advances raise a natural question: can variable selection itself be amortized, so that the relevant variables of a new dataset are obtained without any task-specific optimization? Existing amortized models do not answer this question directly: PFN-based predictors do not output selected variables, and amortized causal-discovery methods must first recover the whole causal graph, a harder task than identifying the Markov blanket of the response.

## 1.2 Proposal and contributions

In this paper, we introduce a variable-selection method for identifying the minimal Markov blanket of the response without prespecifying the form of the regression function. The method is amortized: a Transformer is pretrained on a large collection of variable-selection tasks and can then be applied to a new dataset without parameter updates, yielding substantial computational savings.

The paper makes two main contributions.

• Methodologically, we introduce three key components for learning a suitable Transformer for amortized variable selection: (i) a faithful, rich, and scalable scheme for generating pretraining data; (ii) an objective that guides the Transformer toward estimating each variable’s posterior inclusion probability; and (iii) a modified Transformer architecture tailored to variable selection.

• Empirically, the proposal achieves higher nonlinear variable-selection accuracy than representative model-agnostic methods in our simulation studies. We also observe empirical scaling behavior: the learned Transformer performs better as either the diversity of the pretraining datasets or the Transformer model size increases. The analysis of the Facebook comments data provides further empirical evidence of the proposal’s practical utility: it selects the fewest variables while achieving the highest predictive performance. It also performs variable selection eficiently, providing more than a 10-fold speedup over the neural-network-based baseline while remaining computationally competitive with linear-model-based methods such as Lasso.

## 1.3 Organization

The remainder of this paper is organized as follows. In Section 2, we introduce some basic concepts and techniques used in this paper. Section 3 introduces the proposed method. Sections 4 and 5 present numerical analyses of synthetic datasets and a real-world dataset, respectively. We conclude the paper and provide a discussion in Section 6. Technical proofs and experimental details are deferred to the Appendix.

## 2 Preliminaries

## 2.1 Problem formulation

This paper concerns variable selection in a general regression setting. Let $Y \in \mathbb { R }$ be the response, i.e., the random variable of interest, and let $X = ( X _ { 1 } , \ldots , X _ { p } ) ^ { \intercal } \in \mathbb { R } ^ { p }$ be a p-dimensional vector of covariates used to model the response. Throughout the paper, we impose the following assumption.

Assumption 1. The probability density function of $( X _ { 1 } , \ldots , X _ { p } , Y )$ is strictly positive on $\mathbb { R } ^ { p + 1 }$

This assumption can be interpreted as an identifiability condition; for example, it excludes cases in which two variables follow an exact deterministic relationship (e.g., a linear relationship). Similar identifiability conditions are imposed in the literature to avoid such ill-posed cases (Zhang 2010). Moreover, this assumption is weak overall because it does not rely on any model structure, such as the linear model or the generalized additive model. This makes the assumption well suited to our goal of selecting variables with nonlinear efects.

Under Assumption 1, there exists a unique minimal set $S ^ { \star } \subseteq [ p ]$ (Dawid 1979, Pearl 2014)

such that the sub-vector $X _ { S ^ { \star } }$ of X indexed by $S ^ { \star }$ satisfies

$$
Y \perp X _ { [ p ] \backslash S ^ { \star } } \mid X _ { S ^ { \star } } ,
$$

This means that, given $X _ { S ^ { \star } }$ , the remaining variables $X _ { [ p ] \backslash S ^ { \prime } }$ ⋆ are independent of $Y ;$ in other words, $X _ { S ^ { \star } }$ is the smallest set of variables suficient to explain $Y$ . Thus, $X _ { S ^ { \star } }$ , which we call the Markov blanket of $Y$ , is the target of this paper.<sup>1</sup>

In practic $\mathrm { ; e } , S ^ { \star }$ is unknown, and we typically only have a dataset $\mathcal { D } _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ where $( X _ { i } , Y _ { i } ) \in \mathbb { R } ^ { p + 1 }$ are independently and identically drawn from the joint distribution of $( X , Y )$ Our goal is to identify $S ^ { \star }$ from $\mathcal { D } _ { n }$

## 2.2 Transformer

The core module of our amortized learner is the Transformer, which was first introduced by Vaswani et al. (2017) and is widely used in modern large language models. A notable property of the Transformer is its ability to make predictions from variable-length input sequences by leveraging interactions among the sequence elements. It achieves this property through several key components: embeddings, self-attention, and multilayer perceptrons, which are described below.

Embedding. A Transformer first maps its input tokens to a common representation space. Given n scalars $x _ { 1 } , \ldots , x _ { n }$ , a learnable embedding matrix E produces

$$
\pmb { H } = ( \pmb { E } x _ { 1 } , \ldots , \pmb { E } x _ { n } ) ^ { \top } \in \mathbb { R } ^ { n \times m } .\tag{1}
$$

Self-attention module. Given a token matrix $\pmb { H } \in \mathbb { R } ^ { n \times m }$ , self-attention forms query, key, and value matrices $H W _ { Q }$ $\pmb { H W } _ { K }$ , and $\mathbf { \Omega } _ { H W _ { V } }$ , where $\ b { W _ { Q } } , \ b { W _ { K } } , \ b { W _ { V } } \in \mathbb { R } ^ { m \times d }$ are learnable parameters, and aggregates the value representations using normalized query–key similarities.

With an additional learnable $W _ { O } \in \mathbb { R } ^ { d \times m }$ and an identity mapping, the self-attention map is

$$
\mathsf { S A } ( H ) = H + \mathsf { S o f t m a x } \left( \frac { ( H W _ { Q } ) ( H W _ { K } ) ^ { \top } } { \sqrt { d } } \right) ( H W _ { V } ) W _ { O } .\tag{2}
$$

For simplicity, we omit the positional embedding in Equation (1), multi-head attention and layer normalization from Equation (2), as in Kim et al. (2023), Kajitsuka & Sato (2024).

Multilayer perceptron. The multilayer perceptron (MLP) applies the same two-layer mapping independently to each row of an input $\ b { Z } \in \mathbb { R } ^ { n \times m }$ . Let h be the hidden width. The MLP introduces the learnable weight matrices $\pmb { W } _ { 1 } \in \mathbb { R } ^ { m \times h }$ and $W _ { 2 } \in \mathbb { R } ^ { h \times m }$ and the learnable bias vectors $\pmb { b } _ { 1 } \in \mathbb { R } ^ { h }$ and $\pmb { b } _ { 2 } \in \mathbb { R } ^ { m }$ to process Z via the following rule:

$$
\mathtt { M L P } ( Z ) = Z + \sigma \big ( Z W _ { 1 } + \mathbf { 1 } _ { n } b _ { 1 } ^ { \top } \big ) W _ { 2 } + \mathbf { 1 } _ { n } b _ { 2 } ^ { \top } \in \mathbb { R } ^ { n \times m } ,
$$

where ${ \bf 1 } _ { n }$ is the n-dimensional vector of ones and σ is an element-wise activation function. The complete Transformer alternately stacks multiple SA and MLP modules and takes the form

$$
\begin{array} { r } { n ^ { - 1 } \mathbf { 1 } _ { n } ^ { \top } \big ( \mathtt { M L P } _ { K } \circ \mathtt { S A } _ { K } \circ \cdot \cdot \cdot \circ \mathtt { M L P } _ { 1 } \circ \mathtt { S A } _ { 1 } ( H ) \big ) w _ { U } , } \end{array}
$$

where $n ^ { - 1 } \mathbf { 1 } _ { n } ^ { \top }$ averages the representations over all samples, and $\pmb { w } _ { U } \in \mathbb { R } ^ { m }$ is a learnable parameter that produces a scalar output.

## 3 Methodology

## 3.1 Variable selection: a mapping perspective

Variable selection can be interpreted from a mapping perspective. Specifically, $S ^ { \star }$ can be equivalently represented as a p-dimensional indicator vector $I ^ { \ast } = ( I _ { 1 } , \ldots , I _ { p } ) \in \{ 0 , 1 \} ^ { p }$ where $I _ { j } = \mathbb { I } ( j \in S ^ { \star } )$ . Variable selection seeks to obtain a function T that maps a dataset $\mathcal { D } _ { n }$ to an indicator vector $\widehat { I }$ that indicates which variables should be selected. This perspective can also be used to interpret existing variable selection methods, such as the Lasso. Given a dataset, the Lasso learns a sparse coeficient vector under the linear model assumption $\mathbb { E } [ Y | X ] = X ^ { \top } \beta$ and then derives a variable selection result $\widehat { I }$ from the estimated coeficient ${ \widehat { \beta } } .$

In this section, we introduce a learning-based method for obtaining a function $T$ that can perform variable selection. Once $T$ is learned, it can be applied directly to new datasets to select variables. Learning such a function raises three questions:

• The first question is how to collect pairs of datasets and their true variable indicators for learning an efective function $T .$ . This question arises because the true variable indicators are typically unknown in practice.

• The second question is how to design an objective function so that the learned function $T$ outputs an $\widehat { I }$ that is close to $I ^ { * }$

• The last question is how to design the function T itself. For example, it should accommodate various input shapes, as the input datasets can have diferent n and $p .$

We answer these three questions in Sections 3.2, 3.3, and 3.4, respectively.

## 3.2 Pretraining data generation

In this section, we describe our approach to generating the pretraining data $( { \mathcal { D } } _ { n } , I ^ { * } )$ . Ideally, the generation procedure should satisfy three principles:

• Computational scalability: generating $( { \mathcal { D } } _ { n } , I ^ { * } )$ should be computationally convenient so that we can generate a large amount of pretraining data and increase the likelihood of successfully training the mapping.

• Distributional richness: the generated probability density functions should represent a wide range of relationships between the covariates and the response, thereby supporting the goal of nonlinear variable selection.

• Faithfulness: the probability density function used to generate the dataset $\mathcal { D } _ { n }$ should indeed correspond to $I ^ { * }$ and should also satisfy Assumption 1.

Algorithm 1 Generation of a pretraining pair $( { \mathcal { D } } _ { n } , I ^ { \psi } )$   
Input: Graph class $\mathcal { G } ;$ function class ${ \mathcal { F } } ;$ family ${ \mathcal { P } } ^ { ( 0 ) }$ of strictly positive densities.   
Output: dataset $\mathcal { D } _ { n } = \{ ( X _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { n }$ and indicator vector $I ^ { \psi } = ( I _ { 1 } ^ { \psi } , \ldots , I _ { p } ^ { \psi } ) \in \{ 0 , 1 \} ^ { p }$   
1: Draw a causal graph ψ from $\mathcal { G }$ on the nodes $V _ { 1 } = X _ { 1 } , \ldots , V _ { p } = X _ { p }$ and $V _ { p + 1 } = Y$   
2: For $j = 1 , \ldots , p , I _ { j } ^ { \psi } = 1$ if $X _ { j }$ is $\mathrm { a }$ parent, child, or spouse of $Y ;$ otherwise $I _ { j } ^ { \psi } = 0 .$   
3: Partition the nodes of $\psi$ into topological layers $\mathcal { L } _ { 0 } , \ldots , \mathcal { L } _ { K }$   
4: for $j \in \mathcal { L } _ { 0 }$ do ▷ root nodes   
5: Draw $p _ { j } ^ { ( 0 ) }$ from ${ \mathcal { P } } ^ { ( 0 ) }$ and sample $V _ { 1 j } , \ldots , V _ { n j } \stackrel { \mathrm { i . i . d . } } { \sim } p _ { j } ^ { ( 0 ) }$   
6: end for   
7: for $k = 1 , \ldots , K$ do ▷ layer-wise sampling   
8: for $j \in \mathcal { L } _ { k }$ do   
9: Draw $f _ { j }$ from $\mathcal { F }$ and a noise distribution $q _ { j }$ from ${ \mathcal { P } } ^ { ( 0 ) }$   
10: For $i = 1 , \ldots , n ,$ set $V _ { i j } = f _ { j } \big ( V _ { i , \mathrm { p a } _ { \psi } ( j ) } \big ) + \varepsilon _ { i j }$ where $\varepsilon _ { i j } \sim q _ { j }$   
11: end for   
12: end for   
13: Set $X _ { i } = ( V _ { i 1 } , \ldots , V _ { i p } ) ^ { \top }$ and $Y _ { i } = V _ { i , p + 1 }$ for $i = 1 , \ldots , n .$

To this end, we introduce a graph-based method, whose complete procedure is summarized in Algorithm 1. Specifically, we start with a directed acyclic graph $\psi$ randomly drawn from $\mathcal { G }$ , a class of directed acyclic graphs on $p + 1$ nodes. The graph $\psi$ characterizes the causal relationships among $( X _ { 1 } , \ldots , X _ { p } , Y )$ , where each node in the graph corresponds to one variable. Since $\psi$ is known, we can directly extract the Markov blanket of $Y$ and thereby find $I ^ { \psi }$ . Specifical ${ \mathrm { l y } } ,$ the Markov blanket of $Y$ is the union of (i) the parents of $Y _ { i }$ , (ii) the children of $Y$ , and (iii) the other parents of those children. Given the graph $\psi ,$ we then generate a probability distribution. The density associated with $\psi$ admits a factorization. Specifically, let $V _ { i } = X _ { i }$ denote the ith node in ψ for $i = 1 , \ldots , p ,$ and let $V _ { p + 1 } = Y$ denote the last node. The nodes of $\psi$ can then be partitioned into topological layers $\mathcal { L } _ { 0 } , \ldots , \mathcal { L } _ { K }$ , where $\mathcal { L } _ { 0 } = \{ j : \mathrm { p a } _ { \psi } ( j ) = \emptyset \}$ contains the root nodes, and, for $k \geq 1$ , every parent of a node in $\mathcal { L } _ { k }$ belongs to an earlier layer. The resulting joint density therefore factorizes as

$$
\prod _ { j \in \mathcal { L } _ { 0 } } p _ { j } ^ { ( 0 ) } ( v _ { j } ) \prod _ { j \in \mathcal { L } _ { 1 } } p _ { j } ^ { ( 1 ) } \Bigl ( v _ { j } \mid v _ { \mathsf { p a } _ { \psi } ( j ) } \Bigr ) \times \cdots \times \prod _ { j \in \mathcal { L } _ { K } } p _ { j } ^ { ( K ) } \Bigl ( v _ { j } \mid v _ { \mathsf { p a } _ { \psi } ( j ) } \Bigr ) .\tag{3}
$$

Next, we specify the setup for the distribution $p ^ { ( 0 ) }$ and the conditional distributions $p ^ { ( 1 ) } , \ldots , p ^ { ( K ) }$ . For the distribution $p ^ { ( 0 ) }$ , to satisfy Assumption 1, we consider a set of common distributions ${ \mathcal { P } } ^ { ( 0 ) }$ , each of which has a strictly positive density. For the conditional distributions $p _ { j } ^ { ( k ) } ( v _ { j } | v _ { \mathrm { p a } _ { \psi } ( j ) } )$ , we adopt the following structural model:

$$
V _ { j } = f \big ( V _ { \mathrm { p a } _ { \psi } ( j ) } \big ) + \varepsilon _ { j } ,\tag{4}
$$

where $\varepsilon _ { j }$ is an independent random variable associated with a distribution in ${ \mathcal { P } } ^ { ( 0 ) }$ , and $f$ is a function randomly drawn from a predefined function class $\mathcal { F }$ comprising various functions. The function f should take all parents of $V _ { j }$ as arguments. We denote the resulting joint distribution by $P _ { \psi , f }$ and the collection of all possible joint distributions by P.

Next, we examine whether the above procedure satisfies computational scalability, distributional richness, and faithfulness. First, the dataset $\mathcal { D } _ { n }$ is computationally convenient to construct. Equation (3) naturally induces a layer-wise sampling algorithm. We first sample all root variables in $\mathcal { L } _ { 0 }$ . Then, for $k = 1 , \ldots , K$ , once the variables in the preceding layers have been generated, we sample all variables in $\mathcal { L } _ { k }$ according to Equation (4). A complete observation is obtained after a pass through all topological layers. Second, the family $\mathcal { P }$ is distributionally rich because the structural model allows various functional combinations and $\mathcal { G }$ can generate various graph structures. The following result illustrates this richness by showing that $\mathcal { P }$ includes common statistical models.

Proposition 1. The structural model includes generalized additive models and interaction models for the response Y as special cases.

Finally, we examine faithfulness, $I ^ { \psi } = I ^ { * }$ . Let ${ \mathrm { M B } } _ { \psi } ( Y )$ denote the Markov blanket of Y in $\psi$ . Since the densities in ${ \mathcal { P } } ^ { ( 0 ) }$ are strictly positive, the joint density (3) is also strictly positive; hence, Assumption 1 holds, which implies $S ^ { \star } \subseteq \mathrm { M B } _ { \psi } ( Y )$ (Koller & Friedman 2009, Pearl 2014). The reverse inclusion may fail because the dependence of Y on a covariate can cancel out. For example, if $Y = X _ { 1 } + \varepsilon$ and $X _ { 2 } = Y + X _ { 1 } + \varepsilon ^ { \prime }$ with $\varepsilon , \varepsilon ^ { \prime } \sim N ( 0 , 1 )$ then $X _ { 1 } \in \mathrm { M B } _ { \psi } ( Y )$ , yet $Y \perp X _ { 1 } \mid X _ { 2 }$ and thus $X _ { 1 } \notin S ^ { \star }$ . Another example concerns the specification of ${ \mathcal { F } } \colon$ if a child Z of Y is generated by $Z = Y \mathbb { I } ( X _ { 1 } > 0 ) { + } X _ { 2 } \mathbb { I } ( X _ { 1 } \leq 0 ) { + } \varepsilon$ , then $Y , X _ { 1 } , X _ { 2 }$ all afect Z. Because Y and $X _ { 2 }$ never jointly afect $Z _ { i }$ , we have $Y \perp X _ { 2 } \mid X _ { 1 } , Z$ although $X _ { 2 } \in \mathrm { M B } _ { \psi } ( Y )$

Lemma 2 shows how such cancellations can be avoided under the following restrictions.

Assumption 2. Let $\psi$ be a graph drawn from G. No $X _ { j } \in \mathrm { M B } _ { \psi } ( Y )$ satisfies any of the following conditions: $( A ) \ X _ { j }$ is a parent of both Y and a child of $Y ; ( B ) ~ X _ { j }$ is a child of Y and a parent of another child of $Y ; ( C ) X _ { j }$ is a parent of at least two children of Y .

Assumption 3. Let $f \in { \mathcal { F } }$ be a mapping from $\mathbb { R } ^ { m }$ to R. For each $i \in \{ 1 , \ldots , m \}$ and almost every u<sup>∗</sup> for $j \neq i _ { \cdot }$ , the function $u _ { i } \mapsto f ( u _ { 1 } ^ { * } , \ldots , u _ { i - 1 } ^ { * } , u _ { i } , u _ { i + 1 } ^ { * } , \ldots , u _ { m } ^ { * } )$ is not almost everywhere constant.

Assumption 4. Every distribution in ${ \mathcal { P } } ^ { ( 0 ) }$ has a log-concave density g, $i . e . , g ( \lambda s + ( 1 - \lambda ) t ) \ge$ $g ( s ) ^ { \lambda } g ( t ) ^ { 1 - \lambda }$ , for all $s , t \in \mathbb { R }$ and $\lambda \in \left[ 0 , 1 \right]$

Lemma 2. For the joint density given by Equations (3)–(4), we have $\mathrm { M B } _ { \psi } ( Y ) = S ^ { \star }$ if Assumptions 2–4 hold.

Lemma 2 imposes assumptions on $\mathcal { G } , \mathcal { F }$ , and ${ \mathcal { P } } ^ { ( 0 ) }$ to rule out the examples above. Assumption 2 is not restrictive because it concerns only the local structure of $\psi$ around Y. For example, it holds whenever Y has no children, which already allows Proposition 1 to hold, as shown in its proof. Assumption 3 excludes scenarios in which two parents never act together. It can be further relaxed to a local condition on the functions around $Y .$ analogous to Assumption 2. Assumption 4 is satisfied by many common distributions, such as the Gaussian, Laplace, and logistic distributions; in particular, it covers the Gaussian noise frequently assumed in the variable-selection literature (Zhang 2010, Huang et al. 2018, Wainwright 2019). Assumption 4 can be further relaxed because the proof requires it only for the noise terms associated with the children of $Y$ . More importantly, these assumptions are easy to implement in practice and ensure faithfulness.

## 3.3 Optimization objective

We seek a dataset-level function $T _ { \theta }$ mapping $\mathcal { D } _ { n }$ to $\{ 0 , 1 \} ^ { p }$ that minimizes the misclassification loss $\begin{array} { r } { \frac 1 p \sum _ { j = 1 } ^ { p } \mathbb { I } ( [ T _ { \theta } ( \mathscr { D } _ { n } ) ] _ { j } \neq I _ { j } ^ { \psi } ) } \end{array}$ , where $\theta$ denotes the learnable parameter vector. An objective value of zero corresponds to exact variable selection. However, directly minimizing this objective is challenging as the indicator function is non-diferentiable. We therefore consider the following diferentiable elementwise Bernoulli log-loss

$$
\mathcal { L } ( \theta ; \mathcal { D } _ { n } , I ^ { \psi } ) = - \frac { 1 } { p } \sum _ { j = 1 } ^ { p } \Big \{ I _ { j } ^ { \psi } \log [ T _ { \theta } ( \mathcal { D } _ { n } ) ] _ { j } + ( 1 - I _ { j } ^ { \psi } ) \log \big ( 1 - [ T _ { \theta } ( \mathcal { D } _ { n } ) ] _ { j } \big ) \Big \} .\tag{5}
$$

This objective function is easy to implement using standard deep-learning frameworks, which also provide various optimizers.

Proposition 3 reveals the population-level target of the optimization problem in (5). Specifically, let $\begin{array} { r } { \mathcal { R } ( \theta ) = \mathbb { E } _ { \mathcal { D } _ { n } \sim \mathcal { P } } \left[ \mathcal { L } ( \theta ; \mathcal { D } _ { n } , I ^ { \psi } ) \right] : = \mathbb { E } _ { P _ { \psi , f } \sim \mathcal { P } } \mathbb { E } _ { \mathcal { D } _ { n } \sim P _ { \psi , f } } \left[ \mathcal { L } ( \theta ; \mathcal { D } _ { n } , I ^ { \psi } ) \right] } \end{array}$ be the population loss. Here, we make two notational simplifications: (i) we use $P _ { \psi , f } \sim \mathcal { P }$ to represent a distribution drawn from the family $\mathcal { P } _ { : }$ , and (ii) we omit the subscripts of expectations whenever this causes no confusion. For each $j \in \{ 1 , \ldots , p \}$ , let $\mathbb { P } _ { \psi \sim \mathcal { P } } \left( I _ { j } ^ { \psi } = 1 \mid \mathcal { D } _ { n } \right)$ be the posterior inclusion probability of the jth covariate given a finite-sample dataset $\mathcal { D } _ { n }$ . Note that this posterior depends only on the family $\mathcal { P }$ used in the pretraining data-generation process. Proposition 3 shows that if $T _ { \theta }$ belongs to a suficiently large function class, the global minimizer of the population risk identifies the posterior inclusion probabilities.

Proposition 3 (Posterior inclusion probabilities). Under Assumptions ${ \mathcal { Q } } - { \mathcal { 4 } } ;$ , if there exists $\bar { \theta }$ such that $T _ { \bar { \theta } , j } ( \mathcal { D } _ { n } ) = \mathbb { P } _ { \psi \sim \mathcal { P } } \left( I _ { j } ^ { \psi } = 1 \mid \mathcal { D } _ { n } \right)$ for every $j$ , then $\mathcal { R } ( \theta ) \geq \mathcal { R } ( \bar { \theta } )$ for any θ.

Furthermore, when $\mathcal { D } _ { n }$ contains suficiently many observations, a stronger result holds.

Proposition 4 (Large-sample behavior). Under the same notation and assumptions as in Proposition 3, $T _ { \bar { \theta } , j } ( \mathcal { D } _ { n } ) \to I _ { j } ^ { \psi }$ as $n \to \infty$

This proposition shows that the solution to the global optimization problem converges to the deterministic true label $I _ { j } ^ { \psi }$

In practice, we often encounter finite-sample datasets; thus, for $\widehat { \theta }$ obtained by optimizing Equation (5), $T _ { \widehat { \theta } } ( { \mathcal { D } } _ { n } )$ lies in $[ 0 , 1 ] ^ { p }$ . To obtain the variable-selection results, we apply a hard threshold to the estimated inclusion probabilities $T _ { \widehat { \theta } } ( { \mathcal { D } } _ { n } )$ . Specifically, we use $\alpha \in ( 0 , 1 )$ to obtain $\widehat { I } = ( \widehat { I } _ { 1 } , \ldots , \widehat { I } _ { p } )$ , where $\widehat { I } _ { j } = \mathbb { I } \Big \{ [ T _ { \widehat { \theta } } ( \mathscr { D } _ { n } ) ] _ { j } > \alpha \Big \}$ . Given a grid of candidate $\alpha _ { 1 } , \ldots , \alpha _ { M }$ the threshold $\alpha$ is selected by the following three steps:

1. We use the same procedure described in Section 3.2 to generate pairs of $( { \mathcal { D } } _ { n } , I ^ { * } )$

2. Apply $T _ { \widehat { \theta } }$ and each candidate $\alpha _ { m }$ to $\mathcal { D } _ { n }$ to obtain $\widehat { I } ^ { ( \alpha _ { m } ) }$ , and compute the mis-selection rate $\textstyle \frac { 1 } { p } \sum _ { j = 1 } ^ { p } \mathbb { I } \{ { \widehat { I } } _ { j } ^ { ( \alpha _ { m } ) } \neq I _ { j } ^ { \psi } \}$

3. Repeat Steps 1–2 R times to compute the average mis-selection rate for each candidate value, and select the one with the smallest average mis-selection rate.

Depending on practical preferences, the mis-selection rate can be replaced by another criterion, such as the Matthews correlation coeficient. Thus, the learned posterior inclusion probabilities can be reused across diferent decision criteria, separating probability estimation from metric-specific variable selection without retraining the model.

## 3.4 Transformer for variable selection

Motivated by Proposition 3, we consider a large function class represented by a Transformer; however, the standard Transformer architecture described in Section 2.2 cannot be applied directly for three reasons: (i) its input and output shapes are incompatible with our problem; (ii) the response should not be treated as equivalent to a covariate; and (iii) it does not explicitly enforce the two permutation properties required for variable selection: invariance to permutations of the observations and equivariance to permutations of the covariates.

To address these issues, we redesign the Transformer architecture to include embedding, self-attention, and output modules.

Entry-wise embedding. Using the embedding rule in Equation (1), we construct $\pmb { H } _ { i j } ^ { ( 0 ) } =$ $E X _ { i j }$ for $i = 1 , \ldots , n$ and $j = 1 , \dots , p _ { \mathrm { { } } }$ , and construct $\pmb { H } _ { i , p + 1 } ^ { ( 0 ) } = \pmb { E } Y _ { i } + e _ { Y } \mathrm { ~ f o r ~ } i = 1 , . . . , n$ where $e _ { Y }$ is a learnable parameter used to distinguish the response from the covariates.

Bidirectional self-attention. We design a bi-attention module that first performs rowaxis attention and then column-axis attention. Intuitively, row-axis attention utilizes information from the entire observed column of each variable, while column-axis attention utilizes information across the covariates and the response. Specifically, under row-axis attention, for each variable $j \in \{ 1 , \dots , p + 1 \}$ , we take the n embeddings associated with that variable and obtain

$$
( \widetilde { \pmb { H } } _ { 1 j } , \dots , \widetilde { \pmb { H } } _ { n j } ) = \mathtt { S A } _ { r } ( \pmb { H } _ { 1 j } ^ { ( 0 ) } , \dots , \pmb { H } _ { n j } ^ { ( 0 ) } ) ,
$$

where SA<sub>r</sub> is a self-attention module as described in Section 2.2 and is applied to each variable. We then apply self-attention along the column axis; i.e., for $i \in \{ 1 , \ldots , n \}$ , we obtain

$$
( { \cal H } _ { i 1 } ^ { ( 1 ) } , \ldots , { \cal H } _ { i , p + 1 } ^ { ( 1 ) } ) = \mathtt { S A } _ { c } ( \widetilde { \cal H } _ { i 1 } , \ldots , \widetilde { \cal H } _ { i , p + 1 } ) ,
$$

where the self-attention module $\mathtt { S A } _ { c }$ is applied to each observation. Then, $H ^ { ( 1 ) }$ goes through another bidirectional self-attention module to obtain $H ^ { ( 2 ) }$ . In total, we apply the bidirectional self-attention module K times to obtain $H ^ { ( K ) }$

Output module. This module produces a p-dimensional vector of inclusion probabilities. For each observation $i$ and covariate $j ,$ , we first combine the covariate representation $\pmb { H } _ { i j } ^ { ( K ) }$ with the response representation $\pmb { H } _ { i , p + 1 } ^ { ( K ) }$ to form $\pmb { h } _ { i j } = \sigma \big ( W _ { R } \pmb { H } _ { i , p + 1 } ^ { ( K ) } + W _ { C } \pmb { H } _ { i j } ^ { ( K ) } \big ) \in \mathbb { R } ^ { m }$ where $\pmb { W } _ { R } , \pmb { W } _ { C } \in \mathbb { R } ^ { m \times m }$ are learnable matrices. We then aggregate the information across observations by

$$
z _ { j } = \sum _ { i = 1 } ^ { n } \frac { \exp \left( { \pmb q } ^ { \top } { \pmb h } _ { i j } / \sqrt { m } \right) } { \sum _ { k = 1 } ^ { n } \exp ( { \pmb q } ^ { \top } { \pmb h } _ { k j } / \sqrt { m } ) } { \pmb h } _ { i j } \in \mathbb { R } ^ { m } ,
$$

where $\textbf { \textit { q } } \in \mathbb { R } ^ { m }$ is a learnable query vector. Stacking these vectors gives the matrix $\pmb { Z } = ( z _ { 1 } , \ldots , z _ { p } ) ^ { \top }$ , which is fed into an MLP to output the p-dimensional vector of inclusion probabilities sigmoid $( \mathtt { M L P } ( Z ) \pmb { w } _ { U } + b _ { U } \mathbf { 1 } _ { p } ) \in ( 0 , 1 ) ^ { p }$ , where $\pmb { w } _ { U } \in \mathbb { R } ^ { m }$ and $b _ { U } \in \mathbb { R }$ are learnable parameters.

## 4 Simulation

In this section, we conduct simulation studies to investigate the empirical performance of the proposed method, which we call MambaPFN, short for model-agnostic Markov blanket selection with PFN. In Section 4.1, we evaluate the quality of variable selection; in Section 4.2, we examine how the performance scales with the amount of pretraining data and the number of parameters; and in Section 4.3, we analyze the computational cost.

## 4.1 Variable selection evaluation

We compare MambaPFN with several well-known variable selection methods, including Lasso (Tibshirani 1996), SCAD (Fan & Li 2001), ABESS (Zhu et al. 2020), DFS (Chen et al. 2021), STG (Yamada et al. 2020), HSIC-Lasso (Yamada et al. 2014), and Knockofs

![](images/de60ac4e79faf3614e4ec1b3a91a2f82b5cd0bd103ebc2c92e0c9b5266158b60.jpg)  
Figure 1: Comparison of baselines and MambaPFN on variable selection. Each row contains three panels showing the F1-score, AUC, and absolute model-size gap, respectively. The upper row varies $n \in \{ 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 , 1 6 0 0 \}$ with $p = 2 0$ , whereas the lower row varies $p \in \{ 1 5 , 2 0 , 2 5 , 3 0 , 3 5 \}$ with $n = 8 0 0$  
(Spector & Janson 2022). These baselines cover both linear-model-based and nonlinear variable selection methods. We also compare with an amortized method for causal graph identification, from whose learned graph we extract the Markov blanket of the response. Specifically, we use CauScale (Peng et al. 2026), a state-of-the-art method for this task.  
We report the F1-score of the selected subset and the area under the ROC curve (AUC), which evaluates the complete ranking of active versus inactive covariates; details of the evaluation of each method are described in Appendix B. We evaluate selection accuracy on held-out datasets generated by Algorithm 1 (Section 3.2) and consider two cases: in the first, the number of observations varies over 100, 200, 400, 800, and 1600 with $p = 2 0 ;$ in the second, the number of observations is fixed at 800 while p grows from 15 to 35.

The numerical results are summarized in Figure 1. As the sample size increases, the AUC and F1-score of the methods generally increase, as expected. MambaPFN typically attains the highest F1-score and AUC, reflecting its efectiveness. Moreover, when the sample size is large, the AUC of MambaPFN is high, demonstrating that it separates active variables from irrelevant ones well. STG and DFS also perform reasonably well because both capture nonlinearity when selecting variables, which allows them to surpass linear-model-based methods such as SCAD and Lasso. It is also worth comparing with CauScale, which MambaPFN outperforms, probably because learning the full causal graph is more dificult. Finally, the rightmost panels of Figure 1 show that MambaPFN maintains a substantially smaller absolute model-size gap than the baselines, and this gap approaches zero as the sample size increases.

## 4.2 Scaling law

MambaPFN is pretrained once and then applied without parameter updates. Its selection accuracy is therefore determined at the pretraining stage, and in particular by two resources: the number of pretraining datasets M and the number of network parameters N. In this section, we examine whether the performance of MambaPFN improves predictably as M and N increase, in the spirit of the scaling laws reported for language models (Kaplan et al. 2020). To separate the two resources, we follow Kaplan et al. (2020) and vary one of them while holding the other fixed.

We generate the pretraining data in advance. Specifically, we draw a pool of $3 . 2 \times 1 0 ^ { 5 }$ datasets. From this pool, we form six nested training sets consisting of the first $9 . 6 \times 1 0 ^ { 3 }$ $1 . 9 2 \times 1 0 ^ { 4 } , 4 \times 1 0 ^ { 4 } , 8 \times 1 0 ^ { 4 } , 1 . 6 \times 1 0 ^ { 5 }$ , and $3 . 2 \times 1 0 ^ { 5 }$ datasets. As for model size, following Kaplan et al. (2020), we increase the width and the depth jointly and otherwise keep the architecture. The numbers of parameters range from $5 . 9 \times 1 0 ^ { 4 }$ to $1 . 1 \times 1 0 ^ { 7 }$ . When one factor varies, the other is fixed at its largest value. We evaluate the validation loss at 25 checkpoints equally spaced on a logarithmic scale during training, and report the average of its three smallest values over these checkpoints<sup>2</sup>. The validation set is the fixed set of 2,048 held-out DGPs, generated from a seed stream disjoint from the pretraining pool, following Li et al. (2025).

![](images/a4cee9f4d7cb76980448639d2c58715e711e74cb3a24f0008d4a48018d63d047.jpg)

![](images/9c40f7416f2447ec53054573443077ad04d3c260a1fb97eff43dec78f9e03657.jpg)  
Figure 2: Validation log-loss of MambaPFN on logarithmic axes. Left: the largest model trained on nested training sets of M datasets. Right: models with N parameters trained on the full pool $( M = 3 . 2 \times 1 0 ^ { 5 } )$ . Each point is the average of the three smallest validation losses of one run over 25 checkpoints, and the lines are the fitted power laws.

The results are summarized in Figure 2. The validation loss decreases with both N and M over the ranges considered. These trends therefore indicate that the estimated inclusion probabilities move closer to the posterior inclusion probabilities as N and M grow, and suggest that the accuracy of MambaPFN can be further improved by scaling up the pretraining.

![](images/bec72912f145b45c99a91dc79fd86a799e318aa69ace9ddb5d51406a4b9ea7bf.jpg)  
Figure 3: Runtime comparison of MambaPFN with existing methods. The left panel varies n with p = 20, whereas the right panel varies p with $n = 8 0 0$

## 4.3 Computational analysis

In this section, we compare the computational cost of MambaPFN with that of the baselines. We consider two cases: in the first, we fix the dimension and increase the sample size; in the second, we fix the sample size and increase the dimension. We measure runtime in seconds, and the results are summarized in Figure 3.

MambaPFN and ABESS have the shortest runtimes, and MambaPFN is faster than Lasso, even though it tackles a more challenging, model-agnostic setting. It is nearly 10 times faster than model-agnostic methods such as Knockofs and HSIC-Lasso, and it is also faster than SCAD, which relies on a linear model. It is also about 10 times faster than CauScale, which spends more time recovering the full graph. These results confirm the computational merits of MambaPFN for selecting variables with nonlinear efects.

## 5 Facebook Comments Data Analysis

In this section, we evaluate MambaPFN on a real-world dataset for predicting the volume of comments on Facebook posts. Social media platforms such as Facebook seek to maximize the value of promoting a post. If a post shows a strong growth trend, promoting it may increase its trafic and generate additional revenue. At the same time, rapidly growing posts may carry a higher risk of spreading misinformation or provoking heated discussion; identifying such posts allows platforms to allocate human resources for timely monitoring and moderation.

To gain further insight into such growth trends, Singh et al. (2015) collected a Facebook comment-volume dataset that is publicly available from the UCI repository<sup>3</sup>. The response of interest is the number of new comments received by a post. The dataset contains 53 covariates: 4 post-related variables, such as the number of likes; 25 variables comprising five summary statistics of historical comment counts (minimum, maximum, mean, median, and standard deviation); and 24 variables related to the timing and characteristics of the post (e.g., day-of-week indicators and the number of times the post was shared). Because this real-world dataset was not used to train the Transformer, it represents a challenging out-of-distribution setting. The dataset contains 40,949 observations.

We apply MambaPFN to the Facebook dataset to investigate which variables are most important for predicting the response. For comparison, we also consider the baselines introduced in Section 4<sup>4</sup>. Because the relevant variables are unknown in this real-world dataset, we use predictive performance on the test set as a surrogate criterion. Specifically, we randomly allocate 30% of the observations to variable selection, 30% to fitting an MLP using only the selected variables, and 40% to evaluating its predictive performance. We repeat the complete split-and-evaluate procedure 10 times to reduce split randomness. We also include a setting in which all variables are selected.

The results are reported in Figure 4. MambaPFN achieves the highest $R ^ { 2 }$ , suggesting that its selected variables have a strong association with the response. In contrast, naively fitting an MLP using all variables yields a substantially lower test $R ^ { 2 }$ than performing variable selection, reflecting the advantage of variable selection. MambaPFN also significantly outperforms all other methods and the improvement upon the best baseline approaches 20%. Meanwhile, it selects the fewest variables, suggesting that it efectively captures nonlinear dependence with a more parsimonious subset. Finally, it has the shortest runtime, taking less than one second to analyze the dataset, which is about 5 and 9 times faster than HSIC-Lasso and CauScale, the two model-agnostic methods with the most competitive predictive performance.

![](images/0b284aa3a76c96ea6d02550535edda99e7150516be2313d4a5257bc81357531a.jpg)

![](images/4aec4cf85b0c8e46765d735a5e48b2c84da1a99e0932fed04d6b175fad5274bc.jpg)

![](images/4b6ec71be3c4b7217c1d8e9d03b42558583443d7fff02e1c967092b702a227f9.jpg)  
Figure 4: Comparison of methods on the Facebook comment dataset. The left panel shows the test-set $R ^ { 2 }$ , the middle panel shows the number of selected variables, and the right panel shows the runtime of each method. Error bars show one standard error across the 10 repeated data splits.

Finally, it is worth noting that MambaPFN selects three variables: (i) the time elapsed between publication of the post and the current prediction time (denoted as Base time); (ii) the number of comments in the last 24 hours; and (iii) the temporal diference in comment counts, measured over 24-hour intervals. Their associations with the response are visualized in Figure 5, and all three exhibit some degree of nonlinearity. The last variable shows the strongest nonlinearity, with a roughly V-shaped association: a large positive value indicates accelerating comment activity, whereas a large negative value can correspond to a previously active post that is now cooling. Values near zero largely represent inactive posts, so this variable captures both the direction and the scale of comment activity; this may explain why linear-model-based methods require additional variables to represent the same pattern. The number of comments in the last 24 hours has a strong positive but nonlinear association with the response, making it a direct proxy for the post’s current popularity. In contrast, Base time is negatively associated with future comments, consistent with an attention-decay process that is steep shortly after publication and then gradually levels of.

![](images/16709cf72108b2be9e4576494a19b150325c7536cacaa1127ef65bfd5d3b013f.jpg)  
Figure 5: Scatterplots of the three selected variables against the log-transformed response. Each panel shows one selected variable. The red curve denotes the nonlinear regression fit, while the black dashed line denotes the linear regression fit. The estimated posterior probabilities of the three variables are 0.94, 0.89, and 0.71, respectively.

## 6 Conclusion and Discussion

This paper brings the viewpoint of the bitter lesson to variable selection, a long-standing problem in statistics, and develops an amortized variable selection method, MambaPFN. We provide a principled implementation of this method through the design of the pretraining data, the loss, and the architecture. Both synthetic and real-world data analyses confirm the empirical merits of this method.

Several directions deserve further investigation. First, MambaPFN currently returns only the variable selection results, and a separate MLP is trained for prediction, forming a two-step procedure. It would be interesting to extend the current architecture to make sample-wise predictions based on the posterior inclusion probabilities, yielding an interpretable machine learning method that outputs both the selected variables and predictions based on them. Second, although MambaPFN saves considerable time at the selection stage, pretraining the Transformer is computationally costly. It would be interesting to consider other architectures, such as selective state space models (Gu & Dao 2023) or grouped-query attention (Ainslie et al. 2023), to speed up training and inference.

## References

Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebrón, F. & Sanghai, S. (2023), Gqa: Training generalized multi-query transformer models from multi-head checkpoints, in ‘Proceedings of the 2023 conference on empirical methods in natural language processing’, pp. 4895–4901.

Anthropic (2026), ‘System card: Claude fable 5 & claude mythos 5’.

Balazadeh Meresht, V., Kamkari, H., Thomas, V., Ma, J., Li, B., Cresswell, J. & Krishnan, R. (2025), CausalPFN: Amortized causal efect estimation via in-context learning, in ‘Advances in Neural Information Processing Systems’, Vol. 38, pp. 154945–154984.

Bates, S., Candès, E., Janson, L. & Wang, W. (2021), ‘Metropolized knockof sampling’, Journal of the American Statistical Association 116(535), 1413–1427.

Bien, J., Taylor, J. & Tibshirani, R. (2013), ‘A lasso for hierarchical interactions’, The Annals of Statistics 41(3), 1111–1141.

Blumensath, T. & Davies, M. E. (2009), ‘Iterative hard thresholding for compressed sensing’, Applied and Computational Harmonic Analysis 27(3), 265–274.

Candès, E., Fan, Y., Janson, L. & Lv, J. (2018), ‘Panning for gold: Model-X knockofs for high-dimensional controlled variable selection’, Journal of the Royal Statistical Society Series B: Statistical Methodology 80(3), 551–577.

Chen, J., Stern, M., Wainwright, M. J. & Jordan, M. I. (2017), Kernel feature selection via conditional covariance minimization, in ‘Advances in Neural Information Processing Systems’, Vol. 30, pp. 6946–6955.

Chen, Y., Gao, Q., Liang, F. & Wang, X. (2021), ‘Nonlinear variable selection via deep neural networks’, Journal of Computational and Graphical Statistics 30(2), 484–492.

Climente-González, H., Azencott, C.-A., Kaski, S. & Yamada, M. (2019), ‘Block HSIC lasso: Model-free biomarker detection for ultra-high dimensional data’, Bioinformatics 35(14), i427–i435.

Dawid, A. P. (1979), ‘Conditional independence in statistical theory’, Journal of the Royal Statistical Society Series B: Statistical Methodology 41(1), 1–15.

DeepSeek-AI (2026), ‘Deepseek-v4: Towards highly eficient million-token context intelligence’.

Fan, J. & Li, R. (2001), ‘Variable selection via nonconcave penalized likelihood and its oracle properties’, Journal of the American Statistical Association 96(456), 1348–1360.

Ge, J., Li, X., Jiang, H., Liu, H., Zhang, T., Wang, M. & Zhao, T. (2019), ‘Picasso: A sparse learning library for high dimensional data analysis in r and python’, Journal of Machine Learning Research 20(44), 1–5.

Grinsztajn, L., Flöge, K., Key, O., Birkel, F., Jund, P., Roof, B., Jäger, B., Safaric, D.,

Alessi, S., Hayler, A. et al. (2025), ‘Tabpfn-2.5: Advancing the state of the art in tabular foundation models’, arXiv preprint arXiv:2511.08667 .

Gu, A. & Dao, T. (2023), ‘Mamba: Linear-time sequence modeling with selective state spaces’, arXiv preprint arXiv:2312.00752 .

Han, Z., Chen, A., Li, Z. et al. (2024), ‘Machine learning-based design of electrocatalytic materials towards high-energy lithium–sulfur batteries development’, Nature Communications 15, 8433.

Hédou, J., Marić, I., Bellan, G. et al. (2024), ‘Discovery of sparse, reliable omic biomarkers with Stabl’, Nature Biotechnology 42, 1581–1593.

Hollmann, N., Müller, S., Eggensperger, K. & Hutter, F. (2023), TabPFN: A transformer that solves small tabular classification problems in a second, in ‘International Conference on Learning Representations’.

Huang, J., Horowitz, J. L. & Wei, F. (2010), ‘Variable selection in nonparametric additive models’, The Annals of Statistics 38(4), 2282–2313.

Huang, J., Jiao, Y., Liu, Y. & Lu, X. (2018), ‘A constructive approach to l<sub>0</sub> penalized regression’, Journal of Machine Learning Research 19(10), 1–37.

Jiang, X., Margeloiu, A., Simidjievski, N. & Jamnik, M. (2024), ProtoGate: Prototype-based neural networks with global-to-local feature selection for tabular biomedical data, in ‘International Conference on Machine Learning’, Vol. 235 of Proceedings of Machine Learning Research, PMLR, pp. 21844–21878.

Kajitsuka, T. & Sato, I. (2024), Are transformers with one layer self-attention using lowrank weight matrices universal approximators?, in ‘International Conference on Learning Representations’, Vol. 2024, pp. 35177–35205.

Kaplan, J., McCandlish, S., Henighan, T., Brown, T. B., Chess, B., Child, R., Gray, S., Radford, A., Wu, J. & Amodei, D. (2020), ‘Scaling laws for neural language models’, arXiv preprint arXiv:2001.08361 .

Kim, J., Kim, M. & Mozafari, B. (2023), Provable memorization capacity of transformers, in ‘The eleventh international conference on learning representations’.

Koller, D. & Friedman, N. (2009), Probabilistic graphical models, Vol. 165, MIT press Cambridge.

Kong, E. & Xia, Y. (2007), ‘Variable selection for the single-index model’, Biometrika 94(1), 217–229.

Lemhadri, I., Ruan, F., Abraham, L. & Tibshirani, R. (2021), ‘Lassonet: A neural network with feature sparsity’, Journal of Machine Learning Research 22(127), 1–29.

Li, C.-N., Huang, L.-W., Shao, Y.-H., Guo, T. & Mao, Y. (2024), ‘Feature selection by Universum embedding’, Pattern Recognition 153, 110514.

Li, J., Liu, Y., Liu, W., Fang, S., Wang, L., Xu, C. & Bian, J. (2025), Mars: a financial market simulation engine powered by generative foundation model, in ‘The Thirteenth International Conference on Learning Representations’.

Liao, H., Chen, H., Mi, Y., Luo, C., Horng, S.-J. & Li, T. (2024), ‘Sparse orthogonal supervised feature selection with global redundancy minimization, label scaling, and robustness’, Information Sciences 666, 120454.

Lim, M. & Hastie, T. (2015), ‘Learning interactions via hierarchical group-lasso regularization’, Journal of Computational and Graphical Statistics 24(3), 627–654.

Lorch, L., Sussex, S., Rothfuss, J., Krause, A. & Schölkopf, B. (2022), ‘Amortized inference for causal structure learning’, Advances in Neural Information Processing Systems 35, 13104–13118.

Luo, B. & Halabi, S. (2025), ‘Sparse-input neural network using group concave regularization’, Transactions on Machine Learning Research .

Ma, J., Thomas, V., Hosseinzadeh, R., Labach, A., Cresswell, J., Golestan, K., Yu, G., Caterini, A. L. & Volkovs, M. (2026), ‘Tabdpt: Scaling tabular foundation models on real data’, Advances in Neural Information Processing Systems 38, 172692–172722.

Ma, T., Wang, T. & Samworth, R. J. (2025), ‘Optimal in-context adaptivity and distributional robustness of transformers’, arXiv preprint arXiv:2510.23254 .

Meier, L., van de Geer, S. & Bühlmann, P. (2009), ‘High-dimensional additive modeling’, The Annals of Statistics 37(6B), 3779–3821.

Muennighof, N., Rush, A. M., Barak, B., Le Scao, T., Piktus, A., Tazi, N., Pyysalo, S., Wolf, T. & Rafel, C. (2023), ‘Scaling data-constrained language models’, Advances in Neural Information Processing Systems 36.

Müller, S., Feurer, M., Hollmann, N. & Hutter, F. (2023), PFNs4BO: In-context learning for bayesian optimization, in ‘Proceedings of the 40th International Conference on Machine Learning’, Vol. 202 of Proceedings of Machine Learning Research, PMLR, pp. 25444–25470.

Müller, S., Hollmann, N., Arango, S. P., Grabocka, J. & Hutter, F. (2022), Transformers can do bayesian inference, in ‘International Conference on Learning Representations’.

OpenAI (2026), ‘Gpt-6 astra: System card’.

Pearl, J. (2014), Probabilistic reasoning in intelligent systems, Vol. 88, Elsevier.

Peng, B., Chen, S., Tian, J., Qiao, Y. & Lu, C. (2026), CauScale: Neural causal discovery

at scale, in ‘Proceedings of the 43rd International Conference on Machine Learning’, Vol.   
306 of Proceedings of Machine Learning Research, PMLR, pp. 97940–97957.

Poignard, B. & Yamada, M. (2020), Sparse Hilbert–Schmidt independence criterion regression, in ‘Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics’, Vol. 108, pp. 538–548.

Prékopa, A. (1973), ‘On logarithmic concave measures and functions’, Acta Scientiarum Mathematicarum 34, 335–343.

Qu, J., HolzmÃžller, D., Varoquaux, G. & Morvan, M. L. (2026), ‘Tabiclv2: A better, faster, scalable, and open tabular foundation model’, arXiv preprint arXiv:2602.11139 .

Rakotoarison, H., Adriaensen, S., Viering, T. J., Hvarfner, C., Müller, S., Hutter, F. & Bakshy, E. (2026), α-PFN: Fast entropy search via in-context learning, in ‘Proceedings of the 43rd International Conference on Machine Learning’, Vol. 306, pp. 103387–103406.

Ravikumar, P., Laferty, J., Liu, H. & Wasserman, L. (2009), ‘Sparse additive models’, Journal of the Royal Statistical Society Series B: Statistical Methodology 71(5), 1009–1030.

Ren, Z., Wei, Y. & Candès, E. (2023), ‘Derandomizing knockofs’, Journal of the American Statistical Association 118(542), 948–958.

Robertson, J., Reuter, A., Guo, S., Hollmann, N., Hutter, F. & Schölkopf, B. (2025), Do-PFN: In-context learning for causal efect estimation, in ‘Advances in Neural Information Processing Systems’, Vol. 38, pp. 174811–174848.

Romano, Y., Sesia, M. & Candès, E. (2020), ‘Deep knockofs’, Journal of the American Statistical Association 115(532), 1861–1872.

Sauter, A., Salehkaleybar, S., van Harmelen, F., Plaat, A. & Acar, E. (2025), ‘ACTIVA:

Amortized causal efect estimation via transformer-based variational autoencoder’, arXiv preprint arXiv:2503.01290 .

Shen, X., Pan, W. & Zhu, Y. (2012), ‘Likelihood-based selection and sharp parameter estimation’, Journal of the American Statistical Association 107(497), 223–232.

Singh, K., Sandhu, R. K. & Kumar, D. (2015), Comment volume prediction using neural networks and decision trees, in ‘IEEE UKSim-AMSS 17th International Conference on Computer Modelling and Simulation, UKSim2015 (UKSim2015)’.

Spector, A. & Janson, L. (2022), ‘Powerful knockofs via minimizing reconstructability’, The Annals of Statistics 50(1), 252–276.

Sutton, R. (2019), ‘The bitter lesson’, Incomplete Ideas (blog) 13(1), 38.

Thompson, R., Zhao, H., Steinberg, D. M. & Bonilla, E. V. (2026), ‘Arrow: A foundation model for causal discovery’, arXiv preprint arXiv:2605.07204 .

Tibshirani, R. (1996), ‘Regression shrinkage and selection via the lasso’, Journal of the Royal Statistical Society Series B: Statistical Methodology 58(1), 267–288.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł. & Polosukhin, I. (2017), ‘Attention is all you need’, Advances in neural information processing systems 30.

Wainwright, M. J. (2019), High-dimensional statistics: A non-asymptotic viewpoint, Vol. 48, Cambridge university press.

Williams, D. (1991), Probability with Martingales, Cambridge University Press, Cambridge.

Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P. & Sugiyama, M. (2014), ‘High-dimensional feature selection by feature-wise kernelized lasso’, Neural computation 26(1), 185–207.

Yamada, Y., Lindenbaum, O., Negahban, S. & Kluger, Y. (2020), Feature selection using stochastic gates, in ‘International conference on machine learning’, pp. 10648–10659.

Yang, K., Ganguli, A. & Maiti, T. (2024), ‘ENNS: Variable selection, regression, classification, and deep neural network for high-dimensional data’, Journal of Machine Learning Research 25(335), 1–45.

Yang, L., Lv, S. & Wang, J. (2016), ‘Model free variable selection via kernel method’, Journal of Machine Learning Research 17(82), 1–24.

Ye, K., Gong, S., Zhou, H., Zangirolami, V. & Shi, C. (2026), ‘Can tabular foundation models amortize statistical inference?’, arXiv preprint arXiv:2609.33114 .

Yoon, J., Jordon, J. & van der Schaar, M. (2019), INVASE: Instance-wise variable selection using neural networks, in ‘International Conference on Learning Representations’.

Yuan, M. & Lin, Y. (2006), ‘Model selection and estimation in regression with grouped variables’, Journal of the Royal Statistical Society Series B: Statistical Methodology 68(1), 49–67.

Zhang, C.-H. (2010), ‘Nearly unbiased variable selection under minimax concave penalty’, The Annals of Statistics 38(2), 894–942.

Zhang, X., Ren, G., Yu, H., Yuan, H., Wang, H., Li, J., Wu, J., Mo, L., Mao, L., Hao, M. et al. (2025), ‘Limix: Unleashing structured-data modeling capability for generalist intelligence’, arXiv preprint arXiv:2509.03505 .

Zhu, J., Wang, X., Hu, L., Huang, J., Jiang, K., Zhang, Y., Lin, S. & Zhu, J. (2022), ‘abess: a fast best-subset selection library in python and r’, Journal of Machine Learning Research 23(202), 1–7.

Zhu, J., Wen, C., Zhu, J., Zhang, H. & Wang, X. (2020), ‘A polynomial algorithm for best-

subset selection problem’, Proceedings of the National Academy of Sciences 117(52), 33117– 33123.

Zhu, Z., Fan, Y., Kong, Y., Lv, J. & Sun, F. (2021), ‘DeepLINK: Deep learning inference using knockofs with applications to genomics’, Proceedings of the National Academy of Sciences 118(36), e2104683118.

## A Proofs

## A.1 Proof of Proposition 1

Proof. Choose $\psi \in \mathcal G$ such that $Y = V _ { p + 1 }$ has no children and its parents are the active covariates. Equation (4) then gives

$$
\begin{array} { r } { Y = f \Big ( X _ { \mathrm { p a } _ { \psi } ( p + 1 ) } \Big ) + \varepsilon _ { p + 1 } , } \end{array}
$$

where $\varepsilon _ { p + 1 }$ is independent of X because no covariate is generated from $Y$

For a generalized additive model, take

$$
f ( u ) = \beta _ { 0 } + \sum _ { j \in \mathrm { p a } _ { \psi } ( p + 1 ) } g _ { j } ( u _ { j } )
$$

in $\mathcal { F }$ . Substitution yields $\begin{array} { r } { Y = \beta _ { 0 } + \sum _ { j } g _ { j } ( X _ { j } ) + \varepsilon _ { p + 1 } } \end{array}$ , with $g _ { j } = 0$ outside $\mathrm { p a } _ { \psi } ( p + 1 )$ choosing $g _ { j } ( u ) = \beta _ { j } u$ gives the linear model as a special case. For an interaction model, take

$$
f ( u ) = \beta _ { 0 } + \sum _ { j } \beta _ { j } u _ { j } + \sum _ { j < k } \gamma _ { j k } u _ { j } u _ { k } ,
$$

and choose $\mathrm { p a } _ { \psi } ( p + 1 )$ to contain exactly the covariates having a nonzero main or interaction term. Equation (4) then gives the corresponding interaction model. Thus both model classes are included in the structural model. □

## A.2 Proof of Lemma 2

Proof. Strict positivity of the factors in Equation (3) implies $S ^ { \star } \subseteq \mathrm { M B } _ { \psi } ( Y )$ (Pearl 2014, Koller & Friedman 2009). For the reverse inclusion, fix $X _ { j } \in \mathrm { M B } _ { \psi } ( Y )$ . If $Y \not \sqsubset X _ { j } \mid X _ { - j } ,$ then $j \in S ^ { \star }$ : otherwise, weak union applied to the defining conditional independence for $S ^ { \star }$ would give $Y \perp X _ { j } \mid X _ { - j }$

Only the factor of Y and those of its children remain in the conditional density of $Y$ given X. Assumption 2 ensures that $X _ { j }$ occurs in exactly one of these factors: it is either a parent of $Y _ { i }$ a child of $Y ,$ or another parent of one child of Y . We use two elementary properties of the noise density g. First, $g ( t - a ) = \kappa g ( t - b )$ for almost every t implies $a = b \colon$ : integration $\mathrm { g i v e s } \ \kappa = 1$ , and a nonzero translation would make an integrable density periodic. Second, log-concavity implies

$$
g ( t - a ) g ( t - d ) = g ( t - b ) g ( t - c ) \mathrm { ~ f o r ~ a l m o s t ~ e v e r y ~ } t \quad \Longrightarrow \quad \{ a , d \} = \{ b , c \} .
$$

Indeed, the Hellinger autocorrelation $\begin{array} { r } { u \mapsto \int \sqrt { g ( t ) g ( t + u ) } } \end{array}$ dt is even and log-concave, and is strictly decreasing in |u| by the first property and the Cauchy–Schwarz inequality (Prékopa 1973). Integrating the square root of the displayed identity gives equal separations between the two pairs of shifts; after aligning their signs, unequal midpoints would make the corresponding positive, integrable centered product periodic. Hence the two pairs coincide.

Suppose, contrary to the claim, that $Y \perp X _ { j } \mid X _ { - j }$ . If $X _ { j }$ is a parent of $Y$ , the factor associated with $Y$ shows that two translates of its noise density must be proportional as $x _ { j }$ varies. The first property forces the corresponding values of $f$ to be equal, contradicting Assumption 3. The same argument, with y and $x _ { j }$ interchanged, applies when $X _ { j }$ is a child of Y .

Finally, suppose that $X _ { j }$ is another parent of a child of Y . Conditional independence and the factor associated with that child give, for almost every $( y , y ^ { \prime } , x _ { j } , x _ { j } ^ { \prime } )$ , the second displayed identity with the four shifts

$$
f ( y , x _ { j } , \dots ) , \quad f ( y ^ { \prime } , x _ { j } , \dots ) , \quad f ( y , x _ { j } ^ { \prime } , \dots ) , \quad f ( y ^ { \prime } , x _ { j } ^ { \prime } , \dots ) .
$$

The resulting equality of the two multisets requires either both y-sections or both $x _ { j } .$ -sections to agree. Fixing a nonconstant $x _ { j }$ -section forces the y-sections to be constant on a set of positive measure, while fixing a nonconstant y-section forces the $x _ { j }$ -sections to be constant. Either conclusion contradicts Assumption 3. Thus Y $\chi \ X _ { j } \ | \ X _ { - j }$ in all three cases, so $\mathrm { M B } _ { \psi } ( Y ) \subseteq S ^ { \star }$ . Combining the two inclusions proves $\mathrm { M B } _ { \psi } ( Y ) = S ^ { \star }$ □

## A.3 Proof of Proposition 3

Proof. Conditioning on $\mathcal { D } _ { n }$ in Equation (5) gives

$$
\begin{array} { r l } & { \mathcal { R } ( \theta ) = - \displaystyle \frac { 1 } { p } \sum _ { j = 1 } ^ { p } \mathbb { E } _ { \mathcal { D } _ { n } } \bigg [ \mathbb { P } _ { \psi \sim \mathcal { P } } \Big ( I _ { j } ^ { \psi } = 1 \mid \mathcal { D } _ { n } \Big ) \log T _ { \theta , j } ( \mathcal { D } _ { n } ) } \\ & { \qquad + \mathbb { P } _ { \psi \sim \mathcal { P } } \Big ( I _ { j } ^ { \psi } = 0 \mid \mathcal { D } _ { n } \Big ) \log \{ 1 - T _ { \theta , j } ( \mathcal { D } _ { n } ) \} \bigg ] . } \end{array}
$$

Substituting the assumed expression for $T _ { \bar { \theta } , j } ( \mathcal { D } _ { n } )$ and subtracting yields

$$
\mathcal { R } ( \theta ) - \mathcal { R } ( \bar { \theta } ) = \frac { 1 } { p } \sum _ { j = 1 } ^ { p } \mathbb { E } _ { \mathcal { D } _ { n } } \mathrm { K L } \Big ( \mathrm { B e r n } \Big \{ \mathbb { P } _ { \psi \sim \mathcal { P } } \big ( { I } _ { j } ^ { \psi } = 1 \ | \ \mathcal { D } _ { n } \big ) \Big \} \Big \| \mathrm { B e r n } \{ T _ { \theta , j } ( \mathcal { D } _ { n } ) \} \Big ) \ge 0 .
$$

Hence $\mathcal { R } ( \theta ) \geq \mathcal { R } ( \bar { \theta } )$ for every θ.

## A.4 Proof of Proposition 4

Proof. To compare diferent context lengths on the same data, we take the contexts to be nested: $Z _ { 1 } , Z _ { 2 } , . . .$ . is an infinite sequence of observations with $Z _ { i } ~ = ~ ( X _ { i } , Y _ { i } )$ , and ${ \mathcal { D } } _ { n } = \{ Z _ { 1 } , . . . , Z _ { n } \}$ consists of its first n terms, so that $\mathcal { D } _ { n } \subseteq \mathcal { D } _ { n + 1 }$

Let $Z _ { 1 : \infty } = ( Z _ { 1 } , Z _ { 2 } , . . . )$ denote the whole sequence. Throughout the proof, $\mathbb { P } _ { \psi \sim \mathcal { P } }$ and $\mathbb { E } _ { \psi \sim \mathcal { P } }$ refer to the joint law under which $P _ { \psi , f }$ is drawn from $\mathcal { P }$ and, given $P _ { \psi , f } , \ Z _ { 1 } , Z _ { 2 } , . . .$ . are independent with law $P _ { \psi , f }$ . Write

$$
\eta _ { j } ( \mathcal { D } _ { n } ) = \mathbb { P } _ { \psi \sim \mathcal { P } } \left( I _ { j } ^ { \psi } = 1 \mid \mathcal { D } _ { n } \right) = \mathbb { E } _ { \psi \sim \mathcal { P } } \left( I _ { j } ^ { \psi } \mid Z _ { 1 } , \ldots , Z _ { n } \right) ,
$$

so that $T _ { \bar { \theta } , j } ( { \mathcal { D } } _ { n } ) = \eta _ { j } ( { \mathcal { D } } _ { n } )$ by the definition of $\bar { \theta }$ in Proposition 3. Since $I _ { j } ^ { \psi }$ is bounded, $\{ \eta _ { j } ( \mathcal { D } _ { n } ) \} _ { n \ge 1 }$ is a martingale, and Lévy’s upward theorem (Williams 1991, Section 14.2) gives

$$
\eta _ { j } ( \mathcal D _ { n } ) \to \mathbb { E } _ { \psi \sim \mathcal P } \left( I _ { j } ^ { \psi } \mid Z _ { 1 : \infty } \right) \qquad \mathrm { a l m o s t ~ s u r e l y ~ a n d ~ i n ~ } L _ { 1 } .
$$

It remains to show that $I _ { j } ^ { \psi }$ is almost surely a function of $Z _ { 1 : \infty } .$ , in which case $\mathbb { E } _ { \psi \sim \mathcal { P } } ( I _ { j } ^ { \psi } \mid$ $Z _ { \mathrm { 1 : \infty } } ) = I _ { j } ^ { \psi }$

Let $F _ { \psi , f }$ be the distribution function of $P _ { \psi , f }$ on $\mathbb { R } ^ { p + 1 }$ and ${ \widehat { F } } _ { n }$ the empirical distribution function of $\mathcal { D } _ { n }$ . Conditionally on $P _ { \psi , f } .$ , the observations are independent with law $P _ { \psi , f }$ and the strong law of large numbers gives $\widehat { F } _ { n } ( t ) \to F _ { \psi , f } ( t )$ almost surely for each $t \in \mathbb { Q } ^ { p + 1 }$ since $\mathbb { Q } ^ { p + 1 }$ is countable, the convergence holds simultaneously for all such $t ,$ and integrating over $P _ { \psi , f } \sim \mathcal { P }$ , it holds with probability one. A distribution function is determined by its values on $\mathbb { Q } ^ { p + 1 }$ by right-continuity, and it determines the distribution, so $P _ { \psi , f }$ is almost surely a function of $Z _ { 1 : \infty }$ . By Lemma 2, ${ \cal I } _ { j } ^ { \psi } = { \cal I } _ { j } = { \mathbb I } ( j \in S ^ { \star } )$ , where $S ^ { \star }$ is the minimal Markov blanket of Y under $P _ { \psi , f }$ (Section 2); hence $I _ { j } ^ { \psi }$ is determined by $P _ { \psi , f }$ and is therefore almost surely a function of $Z _ { 1 : \infty }$ . Consequently, $\mathbb { E } _ { \psi \sim \mathcal { P } } ( I _ { j } ^ { \psi } \mid Z _ { 1 : \infty } ) = I _ { j } ^ { \psi }$ almost surely, and $T _ { \bar { \theta } , j } ( \mathcal { D } _ { n } ) \to I _ { j } ^ { \psi }$ almost surely. □

## B Experimental details

## B.1 Details of Section 4.1

Experimental design. We evaluate all methods on held-out data-generating processes (DGPs) independently generated by Algorithm 1 (Section 3.2). In the sample-size experiment, we fix $p = 2 0$ and use $n \in \{ 1 0 0 , 2 0 0 , 4 0 0 , 8 0 0 , 1 6 0 0 \}$ ; each point is averaged over 200 DGP replicates. Within a replicate, the structural parameters, response variable, and active set are shared across context lengths, and all methods receive the same dataset. Covariates are randomly permuted before evaluation. For the classical baselines, both the covariates and response are standardized within each dataset; MambaPFN applies its own internal standardization.

Implementation and selection rules. Each method determines its own model size, and the true support size is used only for evaluation. In particular,

• Lasso uses five-fold cross-validation and selects the nonzero coeficients. ABESS uses the support selected by its oficial implementation under the extended special information criterion, while SCAD selects the minimum-BIC point along its regularization path. The functions/libraries for Lasso, ABESS and SCAD are LassoCV, abess (Zhu et al. 2022), and picasso (Ge et al. 2019), respectively.

• DFS fits candidate support sizes from 0 to p and selects the minimum-BIC candidate. Our PyTorch implementation follows the authors’ reference code<sup>5</sup> for Chen et al. (2021). STG selects variables whose learned gate probabilities exceed 0.5.

• HSIC-Lasso applies three-fold positive sklearn.linear\_model.LassoCV to the block-HSIC design; for n > 800, it uses a random subsample of 800 observations. A reference implementation is provided by the pyHSICLasso library<sup>6</sup>. Knockofs is implemented with the knockpy library<sup>7</sup> and uses Gaussian model-X knockofs with the minimumvariance-reconstructability construction, a Lasso feature statistic, Ledoit–Wolf covariance shrinkage, and target FDR 0.1.

• CauScale uses the authors’ oficial implementation<sup>8</sup> and the released synthetic-data checkpoint<sup>9</sup> to estimate a directed adjacency matrix for (Y, X). After thresholding edge probabilities at 0.5, we select the estimated Markov blanket of Y , consisting of its parents, children, and other parents of its children.

Thus, no method is given |S<sup>⋆</sup>| when constructing its selected set. Continuous scores used for AUC are also method-specific: these are inclusion probabilities for MambaPFN, path-entry or coeficient scores for the penalized methods, learned gate scores for DFS and STG, the knockof statistic for Knockofs, and Markov-blanket edge scores for CauScale.

Evaluation criteria. Let $\widehat S$ and $S ^ { \star }$ denote the selected and true active sets. Precision and recall are defined as

$$
\mathrm { P r e c i s i o n } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F P } } } , \qquad \mathrm { R e c a l l } = { \frac { \mathrm { T P } } { \mathrm { T P } + \mathrm { F N } } } .
$$

Precision is the proportion of selected variables that are truly active, whereas recall is the proportion of truly active variables recovered by the method. Their harmonic mean gives

$$
\mathrm { F 1 } = \frac { \mathrm { 2 ~ P r e c i s i o n ~ R e c a l l } } { \mathrm { P r e c i s i o n + R e c a l l } } .
$$

When no variable is selected, precision and the F1-score are set to zero. The ROC-AUC is computed from the continuous variable scores within each dataset and equals the proportion of active–inactive pairs for which the active variable receives the larger score, with ties assigned weight $1 / 2$ . We also report the absolute model-size gap $| | \widehat { S } | - | S ^ { \star } | |$ to distinguish ranking errors from systematic over- or under-selection. Reported curves are replicate means, and their error bars are pointwise 95% intervals obtained from 2,000 bootstrap resamples of complete DGP replicates; for the sample-size experiment, this preserves the pairing across context lengths. Runtime, when reported, is wall-clock time per dataset for method-specific fitting and scoring and excludes DGP generation, metric calculation, and plotting. MambaPFN and CauScale inference was performed on a single NVIDIA A100 GPU with 80 GB of memory, while the remaining baselines ran on the allocated CPU cores.

## B.2 Details of Section 4.2

All runs share the same number of gradient steps and the same optimization settings, so a run cycles over its training set several times, $\mathrm { e . g . }$ , four times for the full $\mathrm { p o o l } ^ { 1 0 }$ . Fixing the number of steps ensures that a larger M brings more distinct data rather than more optimization.

When studying the efect of model size, we fix the training set at the full pool, $M = 3 . 2 \times 1 0 ^ { 5 }$ and vary the model size. As for studying the efect of pretraining data scale, we fix the largest model with $N = 1 . 1 \times 1 0 ^ { 7 }$ , so that model capacity is unlikely to be the bottleneck, and vary M over the six nested training sets. The two experiments share one run, namely the largest model trained on the full pool.

## B.3 Details of Section 5

Dataset preparation. We use Variant 1 of the Facebook Comment Volume training data from the UCI repository, containing 40,949 observations, 53 predictors, and the number of comments received in the following H hours as the response. The fourth predictor, page category, is treated as categorical. The remaining predictors are treated as numeric, and the two downstream figures retain the original feature definitions supplied with the dataset.

Leakage-separated evaluation. For each of 10 seeded repetitions, we randomly divide the data into 12,284 observations for variable selection, 12,284 for downstream MLP fitting, and 16,379 for final testing; the remaining two observations produced by integer rounding are unused. The three subsets are mutually disjoint, and every selector receives the same selection subset within a repetition. Median imputation and standardization for feature selection are estimated using only the selection subset. The MambaPFN checkpoint was pretrained with $p \leq 3 5$ , so its application to the 53 Facebook predictors also evaluates dimension extrapolation.

Downstream preprocessing and prediction. After variable selection, numeric predictors are median-imputed and standardized using only the downstream training subset, while the page-category predictor is one-hot encoded using levels observed in that subset. The response is standardized using the downstream training subset but is not log-transformed; all test metrics are computed on the original response scale. The downstream predictor is an MLP with hidden widths 256 and 128, GELU activations, and dropout 0.1. It is optimized by AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , and batch size 2,048 for at most 100 epochs. Within the downstream training subset, 15% of the observations are used for early stopping with patience $1 0 ;$ the selected number of epochs is then used to refit the MLP on the complete downstream training subset. Test R<sup>2</sup> is evaluated once per $R ^ { 2 }$ repetition, and Figure 4 reports means and standard errors across the 10 repetitions. The all-variable MLP follows the same preprocessing and fitting protocol.

Full-data descriptive analysis. To interpret the variables selected by MambaPFN, we separately apply it to all 40,949 observations after full-data median imputation and standardization. The observations are partitioned into balanced, nonoverlapping blocks of at most 1,000 rows, the raw logits are averaged across blocks, and variables with inclusion probability above 0.5 are selected. Figure 5 plots every observation using log(1 + Y ) and overlays a linear fit and a cubic-spline ridge fit with eight quantile-spaced knots; the fitted curves are drawn over the central 99% of each predictor.