# In-Context Guidance: Learning Inter-Task Synergies via Numerical Foundational Models for Few-Shot Multitask Optimization

Tingyang Wei, Haofeng Wu, Member, IEEE, Jiao Liu, Member, IEEE, Zhao Wei, Puay Siew Tan, and Yew-Soon Ong, Fellow, IEEE

Abstract—Multi-task optimization (MTO) addresses a set of optimization tasks simultaneously, often suffering from inaccurate inter-task relationship estimation under limited evaluation budgets, leading to negative transfer. This paper introduces In-Context Guidance Multitask Optimization (ICG-MTO), a novel framework that leverages numerical foundational models to improve inter-task coupling estimation in few-shot scenarios. Unlike conventional methods that rely solely on scarce observed data, ICG-MTO employs a frozen foundational model to infer auxiliary guidance through in-context learning. The framework operates through three stages: constructing an algorithm-specific in-context query from evaluated solutions, using the foundational model to infer a guidance signal characterizing predictive relationships among tasks, and translating this signal into algorithmspecific guidance for maximum-a-posteriori coupling estimation. This approach provides regularization during the early, datascarce stages of optimization and gradually relinquishes control as task-specific observations accumulate. We instantiate the framework in multitask Bayesian optimization as ICG-MTBO, using directional fitness-class queries to guide inter-task coupling estimation, and further instantiate it in MFEA-II using decisionspace-overlap queries to guide random mating probability estimation. Experiments across synthetic benchmarks and a realworld robot arm control problem, together with evaluations under different acquisition functions and evolutionary multitasking, demonstrate the effectiveness and generality of ICG-MTO for few-shot multitask optimization.

Index Terms—Multi-task optimization, evolutionary algorithms, Gaussian process, foundational models.

## I. INTRODUCTION

ULTITASK optimization aims to solve multiple optitask synergy [1], [2]. Given a predefined set of optimization tasks, the key challenge is not merely whether knowledge should be transferred [3], [4], but how strongly the search processes across tasks should be coupled so that beneficial transfer can be promoted while negative transfer is mitigated under limited evaluation budgets. For instance, the pioneering multifactorial evolutionary algorithm (MFEA) [3] employs a random mating probability, rmp, to control the intensity of inter-task transfer. The follow-up work, MFEA-II [4], introduces an adaptive mixture-model based estimator to control this transfer according to the evolving task populations. More generally, multitask optimizers contain an explicit or implicit inter-task coupling mechanism that determines how observations [4]–[8] obtained from one task influence the search on another.

Estimation of such inter-task coupling is particularly difficult under few-shot settings. Let $\mathcal { D } _ { k }$ denote the evaluated solutions collected from task $\mathcal { T } _ { k }$ , and let ϑ denote a set of generic inter-task coupling parameters. A conventional datadriven estimator can be written as

$$
\widehat { \pmb { \vartheta } } = \underset { \pmb { \vartheta } } { \operatorname { a r g m a x } } \mathcal { L } \left( \pmb { \vartheta } ; \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { K } \right) ,\tag{1}
$$

where $\mathcal { L }$ denotes an estimator-specific objective, such as a marginal likelihood [9], a mixture likelihood [4], or a fitnessbased measure of transfer utility [10]. During the early stage of optimization, when knowledge transfer is potentially most valuable, the limited datasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$ may provide insufficient evidence for reliably estimating ϑ, leading to inaccurate inter-task coupling and potentially negative transfer [4], [11]. This challenge arises across different multitask optimizers: multitask Bayesian optimization estimates the inter-task cou pling matrix of a multitask Gaussian process from limited observations [9], while evolutionary multitasking estimates random mating probabilities [3], [4] or other transfer-related parameters [5] from small evolving populations.

This motivates the use of an auxiliary source of information to guide inter-task coupling estimation during the datascarce stage of optimization. Numerical foundational models provide one such source: pretrained on broad distributions of synthetic numerical or tabular datasets, they can perform in-context prediction on small context datasets through a forward pass, without task-specific fine-tuning or additional function evaluations [12]–[14]. Building on this capability, we introduce In-Context Guidance Multitask Optimization (ICG MTO), which uses a frozen numerical foundational model to infer auxiliary guidance for inter-task coupling estimation. Rather than directly using its predictions as the coupling estimate, ICG-MTO incorporates the inferred guidance as a prior over the algorithm-specific coupling parameters:

$$
\widehat { \pmb { \vartheta } } _ { \mathrm { M A P } } = \underset { \pmb { \vartheta } } { \arg \operatorname* { m a x } } \left[ \mathcal { L } \left( \pmb { \vartheta } ; \mathcal { D } _ { 1 } , \ldots , \mathcal { D } _ { K } \right) + \log p \left( \pmb { \vartheta } \mid \pmb { \vartheta } ^ { \ast } \right) \right] ,\tag{2}
$$

where $\vartheta ^ { \ast }$ denotes the algorithm-specific reference coupling parameters derived from the inferred in-context guidance.

The ICG-MTO framework follows three stages. First, an in-context query is constructed by reorganizing the evaluated solutions into a supervised prediction problem aligned with the inter-task coupling mechanism of the base multitask optimizer. Second, the frozen numerical foundational model performs incontext prediction on the constructed queries and produces an in-context guidance signal characterizing the predictive relationships among tasks. Third, this signal is translated into algorithm-specific guidance and incorporated into the inter-task coupling estimator through a maximum-a-posteriori formulation, as shown in (2). The corresponding guidance weight is largest when evaluated solutions are scarce and is gradually annealed as task-specific observations accumulate. The framework therefore supports the inter-task coupling estimator during its least reliable stage while allowing the base multitask optimizer to progressively fall back to its conventional data-driven behavior. We instantiate this framework in two representative multitask optimizers: ICG-MTBO, which uses directional fitness-class queries to guide the inter-task coupling estimation of a multitask Bayesian optimizer, and ICG-MFEA-II, which uses decision-space-overlap queries to guide the estimation of the random mating probability matrix in MFEA-II [4].

The main contributions of this paper are summarized as follows:

• An In-Context Guidance Multitask Optimization framework is proposed to improve inter-task coupling estimation under limited evaluation budgets. It uses a frozen numerical foundational model as an auxiliary source of guidance without replacing the underlying multitask optimizer.

• An in-context query construction strategy is developed for multitask Bayesian optimization. It transforms evaluated solutions into directional fitness-class prediction queries, where task-specific inter-task guidance can be inferred.

• A guidance-informed maximum-a-posteriori estimator is developed for the multitask Gaussian process. The inferred in-context guidance is incorporated as an annealed regularization term over the inter-task coupling matrix, whose influence gradually diminishes as optimization observations accumulate.

• The framework is evaluated with two different acquisition functions and is further instantiated in MFEA-II through a decision-space-overlap query, demonstrating that the proposed in-context guidance principle is not restricted to a particular acquisition strategy or multitask optimization paradigm.

## II. PRELIMINARIES

## A. Multitask Bayesian Optimization

Multitask Bayesian optimization [9] extends conventional Bayesian optimization by jointly optimizing multiple related black-box optimization tasks. Rather than constructing an independent surrogate model for each task, multitask Bayesian optimization employs a multitask Gaussian process (MTGP) [15] to model the objective functions jointly over the decision space and the task space. By explicitly parameterizing inter-task coupling, the MTGP allows observations collected from one task to improve the predictive model of another related task, thereby facilitating knowledge transfer under limited evaluation budgets.

The MTGP instantiated in this paper follows the intrinsic coregionalization model (ICM) [9], which assumes a separable covariance structure:

$$
\kappa _ { \mathrm { m t } } \left( ( \mathbf { x } , k ) , ( \mathbf { x } ^ { \prime } , k ^ { \prime } ) \right) = \kappa _ { \mathbf { x } } \left( \mathbf { x } , \mathbf { x } ^ { \prime } \right) \cdot \mathbf { B } [ k , k ^ { \prime } ] ,\tag{3}
$$

where $\kappa _ { \mathbf { x } } ( \mathbf { x } , \mathbf { x } ^ { \prime } )$ measures the similarity between two solutions in the decision space, while $\mathbf { B } \in \mathbb { S } _ { + } ^ { K }$ denotes the positive semidefinite inter-task coupling matrix for the K optimization tasks. Under this formulation, the covariance between two observations is jointly determined by their similarity in the decision space and the coupling between their corresponding tasks. The matrix B therefore governs how information is shared across tasks and directly affects the intensity and quality of knowledge transfer within the MTGP.

In this paper, the kernel function in the decision space is defined using an automatic relevance determination (ARD) radial basis function (RBF) kernel [16]:

$$
\kappa _ { \bf x } \left( { \bf x } , { \bf x } ^ { \prime } \right) = \exp \left( - \frac { 1 } { 2 } \sum _ { i = 1 } ^ { V } \frac { ( x _ { i } - x _ { i } ^ { \prime } ) ^ { 2 } } { \ell _ { i } ^ { 2 } } \right) ,\tag{4}
$$

where $\ell _ { i }$ denotes the length-scale parameter associated with the i-th decision variable. The ARD formulation assigns a distinct length scale to each dimension, allowing the surrogate model to characterize the relative importance of individual decision variables [16].

Under the Positive Index Kernel in Gpytorch [17], the inter-task coupling matrix is parameterized as

$$
\mathbf { B } = \mathbf { L } \mathbf { L } ^ { \top } + \mathrm { d i a g } ( \mathbf { v } ) ,\tag{5}
$$

where L is a non-negative lower-triangular matrix and v is a non-negative vector. This parameterization ensures that B is positive semidefinite and therefore defines a valid task covariance matrix. Because non-negativity is imposed on both L and v, the resulting task covariances are restricted to be non-negative.

Let $\vartheta$ collect the inter-task coupling parameters, the decision-space kernel parameters, and the observation-noise parameters. Given the observations $\mathbf { y } ,$ the corresponding inputs X, and the covariance matrix $\mathbf { K } _ { \vartheta }$ , conventional MTGP training estimates ϑ through maximum likelihood estimation:

$$
\begin{array} { r l } & { \widehat { \pmb \vartheta } _ { \mathrm { M L E } } = \underset { \pmb \vartheta } { \mathrm { a r g m a x } } \log p \left( \mathbf { y } \mid \mathbf { X } , \pmb \vartheta \right) } \\ & { \quad \quad = \underset { \pmb \vartheta } { \mathrm { a r g m a x } } \left[ - \frac { 1 } { 2 } \mathbf { y } ^ { \top } \mathbf { K } _ { \pmb \vartheta } ^ { - 1 } \mathbf { y } - \frac { 1 } { 2 } \log \left| \mathbf { K } _ { \pmb \vartheta } \right| - \frac { n } { 2 } \log ( 2 \pi ) \right] . } \end{array}\tag{6}
$$

Maximum likelihood estimation relies solely on the collected observations to infer the task coupling matrix and the remaining kernel hyperparameters [9]. Its quality therefore depends on whether the available data provide sufficient evidence. Under few-shot settings, the marginal likelihood may be weakly informative such that multiple inter-task coupling structures explain the limited observations similarly well [18]. As a result, maximum likelihood estimators with limited samples can be susceptible to instability or overfitting, while GP kernel hyperparameter estimation can become ill-posed under particular model and observation conditions [18]–[21].

A Bayesian estimation perspective provides a natural means of regularizing this estimation process. Instead of estimating the parameters solely from the marginal likelihood, maximum a posteriori (MAP) estimation combines the evidence provided by the observations with additional prior information [19], [22]:

$$
\widehat { \pmb { \vartheta } } _ { \mathrm { M A P } } = \underset { \pmb { \vartheta } } { \operatorname { a r g m a x } } \left[ \log p \left( \mathbf { y } \mid \mathbf { X } , \pmb { \vartheta } \right) + \log p \left( \pmb { \vartheta } \right) \right] .\tag{7}
$$

The prior term regularizes the estimator by favoring plausible parameter configurations when the likelihood provides insufficient evidence. Such regularization can improve parameter estimation under limited observations by discouraging configurations that are weakly supported by the collected data [19]– [21].

Since negative transfer in the MTGP is governed primarily by the inter-task coupling matrix, the central question under few-shot settings is how to construct an informative guidance term for estimating B in (3). A generic ad hoc prior may not adequately represent the heterogeneous inter-task relationships arising across different optimization problems. This motivates the construction of an auxiliary, data-dependent guidance signal that complements the limited likelihood information and regularizes the estimation of the inter-task coupling matrix. In this paper, such a guidance signal is inferred through incontext learning using a frozen numerical foundational model. It is subsequently incorporated into the MTGP through a guidance-informed MAP estimator.

## B. Numerical Foundational Models and In-Context Learning

Recent advances in foundational models have demonstrated that large-scale pretraining can produce general-purpose learners that adapt to new tasks through in-context learning [23]. While this paradigm was initially established in language and vision using large text and image corpora, an emerging direction considers numericalfoundational models, whose pretraining data consist of large collections of synthetic numerical datasets [12]. These models acquire inductive biases over numerical relationships, functional structures, and predictive procedures that can be transferred to previously unseen tabular learning problems [12].

A popular numerical foundational model is Prior-Data Fitted Network (PFN) [24], which reformulates Bayesian posterior prediction as a supervised learning problem over datasets. Given a prior distribution over supervised learning tasks, PFNs are pretrained on a large collection of synthetic datasets sampled from that prior and learn to approximate Bayesian inference. Formally, given a context dataset

$$
\mathcal { C } = \left\{ ( \mathbf { x } ^ { ( i ) } , y ^ { ( i ) } ) \right\} _ { i = 1 } ^ { N } ,\tag{8}
$$

a PFN approximates the posterior predictive distribution

$$
\operatorname { P F N } \left( y \mid \mathbf { x } , { \mathcal { C } } \right) \approx p \left( y \mid \mathbf { x } , { \mathcal { C } } \right) = \int p \left( y \mid \mathbf { x } , f \right) p \left( f \mid { \mathcal { C } } \right) \mathrm { d } f ,\tag{9}
$$

where $f$ denotes the latent predictive function. Unlike conventional Bayesian inference, which performs posterior computation separately for every new dataset, the transformer amortizes this inference during pretraining and enables posterior prediction through a forward pass [12]–[14], [24].

TabPFN [12] is a practical instantiation of PFNs designed for tabular learning under small-data cases. This pretrained model can function as a general-purpose predictor that is competitive across diverse tabular datasets without task-specific fine-tuning [12]. The adaptation behavior of TabPFN can be attributed to in-context learning (ICL) [25]. Instead of updating model parameters, ICL conditions the frozen transformer on a context dataset and performs prediction directly from the provided examples. Let C in (8) denote the context set, and let $\mathbf { x } _ { q }$ denote a query sample. The prediction from ICL can be obtained as

$$
\widehat { y } _ { q } = \mathcal { F } \left( \mathbf { x } _ { q } , \mathcal { C } \right) ,\tag{10}
$$

where $\mathcal { F }$ denotes the pretrained transformer with fixed parameters. The model therefore adapts to a new prediction query entirely by conditioning on the context examples rather than through gradient-based fine-tuning.

In this paper, the numerical foundational model is not used to approximate the objective functions directly or replace the multitask Gaussian process. Instead, it serves as an auxiliary guidance module for improving inter-task coupling estimation from limited observations. To align its prediction task with the inter-task coupling mechanism of the base multitask optimizer, the evaluated solutions are reorganized into algorithm-specific in-context queries. The frozen numerical foundational model then performs in-context prediction on these queries to infer guidance signals characterizing inter-task relationships. These signals are subsequently translated into algorithm-specific reference coupling parameters and incorporated into the coupling estimator through a guidance-informed prior.

## III. IN-CONTEXT GUIDANCE MULTITASK BAYESIAN OPTIMIZATION

## A. Overview

The proposed In-Context Guidance Multitask Bayesian Optimization (ICG-MTBO) augments conventional multitask Bayesian optimization with a numerical foundational model that provides auxiliary guidance for inter-task coupling estimation under limited evaluation budgets. Unlike approaches that replace the surrogate model [26] or modify the acquisition strategy [27], ICG-MTBO preserves the conventional multitask Bayesian optimization pipeline and intervenes only in the estimation of the inter-task coupling matrix of the multitask Gaussian process (MTGP). The MTGP surrogate, uncertainty quantification, acquisition optimization, and objective evaluation therefore remain unchanged.

Figure 1 illustrates the overall workflow. Evaluated solutions are used both by the conventional MTGP estimator and by the proposed guidance mechanism, which consists of three stages:

![](images/a24db412eae22568b03d7592a3652404ff5dbadefa05cc506a02e011256d4eda.jpg)  
Fig. 1. Overall workflow of the proposed In-Context Guidance Multitask Bayesian Optimization (ICG-MTBO) framework. The upper pipeline corresponds to a conventional multitask Bayesian optimizer, where evaluated solutions are used to construct an MTGP surrogate for acquisition optimization and subsequent function evaluation. The lower pipeline augments inter-task coupling estimation through three sequential stages. (1) In-context query reorganizes the evaluated solutions into algorithm-specific prediction queries by partitioning the observations of each task into representative fitness classes (the better and worse halves in ICG-MTBO). (2) In-context guidance learning employs a frozen numerical foundational model (TabPFN) to perform cross-task prediction on the constructed queries, producing guidance signals that characterize predictive relationships among tasks. (3) Maximum-a-posteriori estimation translates the inferred guidance into reference coupling parameters and incorporates them as a prior for estimating the MTGP task coupling matrix. The resulting task kernel is used by the MTGP surrogate, while the remaining components of the Bayesian optimization framework remain unchanged.

• First, in-context query reorganizes the evaluated solutions into algorithm-specific prediction queries aligned with the inter-task coupling mechanism of MTBO.

• Second, in-context guidance learning utilizes a frozen numerical foundational model to perform cross-task prediction on the constructed queries and infer guidance signals characterizing predictive relationships among tasks.

• Finally, MAP estimation translates the inferred guidance into reference coupling parameters and incorporates them as a prior for estimating the MTGP inter-task coupling matrix. The resulting task kernel is subsequently used by the conventional MTGP surrogate for acquisition optimization and function evaluation.

ICG-MTBO therefore maintains a separation between optimization and guidance: the multitask Bayesian optimizer remains responsible for objective modeling, uncertainty quantification, and candidate selection, while the numerical foundational model provides auxiliary information for inter-task coupling estimation. This modularity also enables the same ICG principle to be instantiated with other multitask optimizers, as demonstrated later with MFEA-II [4].

## B. In-Context Query Construction

To align the prediction task of the numerical foundational model with inter-task coupling estimation, the evaluated solutions are reorganized into directional fitness-class prediction queries. Consider task $\mathcal { T } _ { k }$ with $N _ { k }$ evaluated solutions,

$$
\begin{array} { r } { \mathcal { D } _ { k } = \Big \{ \Big ( \mathbf { x } _ { k } ^ { ( i ) } , y _ { k } ^ { ( i ) } \Big ) \Big \} _ { i = 1 } ^ { N _ { k } } , } \end{array}\tag{11}
$$

where $\mathbf { x } _ { k } ^ { ( i ) } \in \mathcal { X }$ and $y _ { k } ^ { ( i ) }$ denote the evaluated solution and its objective value, respectively. Since objective values may have different scales and distributions across tasks, each observation is first assigned a task-wise rank,

$$
r _ { k } ^ { ( i ) } = \mathrm { r a n k } \left( y _ { k } ^ { ( i ) } ; \{ y _ { k } ^ { ( j ) } \} _ { j = 1 } ^ { N _ { k } } \right) ,\tag{12}
$$

and subsequently mapped into $C$ ordinal fitness classes:

$$
\ell _ { k } ^ { ( i ) } = \mathrm { c l i p } \left( \left\lfloor \frac { r _ { k } ^ { ( i ) } C } { N _ { k } } \right\rfloor , 0 , C - 1 \right) .\tag{13}
$$

In this paper, $C = 2$ is adopted, partitioning the observations of each task into the better and worse halves according to their objective values. The resulting classified dataset is

$$
\begin{array} { r } { \widetilde { D } _ { k } = \Big \{ \Big ( \mathbf { x } _ { k } ^ { ( i ) } , \ell _ { k } ^ { ( i ) } \Big ) \Big \} _ { i = 1 } ^ { N _ { k } } . } \end{array}\tag{14}
$$

For each ordered task pair $( k _ { 1 } , k _ { 2 } )$ , an in-context query is then constructed as

$$
\begin{array} { r } { \mathcal { Q } _ { k _ { 1 } \to k _ { 2 } } = \left( \widetilde { \mathcal { D } } _ { k _ { 1 } } , \{ \mathbf { x } _ { k _ { 2 } } ^ { ( i ) } \} _ { i = 1 } ^ { N _ { k _ { 2 } } } \right) , } \end{array}\tag{15}
$$

where the classified observations from $\tau _ { k } .$ provide the incontext examples and the evaluated solutions from $\mathcal { T } _ { k _ { 2 } }$ serve as query inputs. The query therefore examines whether the fitness structure observed on $\mathcal { T } _ { k _ { 1 } }$ can predict the better and worse regions of $\mathcal { T } _ { k _ { 2 } }$ . Since predictive utility can be asymmetric across tasks [28], [29], ${ \mathcal { Q } } _ { k _ { 1 } \to k _ { 2 } }$ and $\mathcal { Q } _ { k _ { 2 } \to k _ { 1 } }$ are treated as distinct directional queries. These queries are subsequently passed to the frozen numerical foundational model to infer inter-task guidance.

## C. Learning In-Context Guidance

Given the directional in-context queries constructed in Section III-B, the frozen TabPFN is used to infer predictive relationships among tasks without task-specific fine-tuning or parameter updates. For each ordered task pair $( k _ { 1 } , k _ { 2 } )$ , the classified observations of $\mathcal { T } _ { k _ { 1 } }$ provide the in-context examples, while the evaluated solutions of $\mathcal { T } _ { k _ { 2 } }$ serve as prediction inputs. For each target solution $\mathbf { x } _ { k _ { 2 } } ^ { ( i ) }$ , TabPFN produces a categorical predictive distribution:

$$
\mathbf { p } _ { k _ { 1 }  k _ { 2 } } ^ { ( i ) } = \mathcal { F } ( \ell \mid \mathbf { x } _ { k _ { 2 } } ^ { ( i ) } ; \mathcal { \widetilde { D } } _ { k _ { 1 } } ) , i = 1 , \dotsc , N _ { k _ { 2 } } ,\tag{16}
$$

where

$$
\mathbf { p } _ { k _ { 1 }  k _ { 2 } } ^ { ( i ) } \in [ 0 , 1 ] ^ { C } , \sum _ { c = 0 } ^ { C - 1 } p _ { k _ { 1 }  k _ { 2 } , c } ^ { ( i ) } = 1 .\tag{17}
$$

The predictive utility from source task $\mathcal { T } _ { k _ { 1 } }$ to target task $\mathcal { T } _ { k _ { 2 } }$ is quantified by the mean cross-entropy with respect to the true fitness-class labels:

$$
\mathrm { C E } [ k _ { 1 } , k _ { 2 } ] = - \frac { 1 } { N _ { k _ { 2 } } } \sum _ { i = 1 } ^ { N _ { k _ { 2 } } } \log \mathcal { F } \left( \ell _ { k _ { 2 } } ^ { ( i ) } \mid \mathbf { x } _ { k _ { 2 } } ^ { ( i ) } ; \boldsymbol { \widetilde { \mathcal { D } } } _ { k _ { 1 } } \right) .\tag{18}
$$

A smaller $\mathrm { C E } [ k _ { 1 } , k _ { 2 } ]$ indicates that the fitness structure observed on $\mathcal { T } _ { k _ { 1 } }$ is more informative for distinguishing the better and worse regions of $\mathcal { T } _ { k _ { 2 } }$ . Since the source and target tasks play different roles in the prediction, this utility is directional.

To obtain a bounded guidance signal, the cross-entropy is calibrated against the random-classification baseline log C:

$$
\Delta [ k _ { 1 } , k _ { 2 } ] = \operatorname { c l i p } \left( \log C - \mathrm { C E } [ k _ { 1 } , k _ { 2 } ] , 0 , \log C \right) .\tag{19}
$$

The directed guidance is then defined as

$$
S [ k _ { 1 } , k _ { 2 } ] = \left( \frac { \Delta [ k _ { 1 } , k _ { 2 } ] } { \log C } \right) ^ { 1 / \tau } , S [ k , k ] = 1 ,\tag{20}
$$

where $\tau > 0$ controls the calibration sharpness and is set to $\tau ~ = ~ 1$ in the experiments. Thus, $S [ k _ { 1 } , k _ { 2 } ] ~ \in ~ [ 0 , 1 ]$ with predictions no better than random mapped to zero and perfect cross-task classification mapped to one. Applying this procedure to all ordered task pairs yields

$$
\mathbf { S } = [ S [ k _ { 1 } , k _ { 2 } ] ] _ { k _ { 1 } , k _ { 2 } = 1 } ^ { K } \in [ 0 , 1 ] ^ { K \times K } .\tag{21}
$$

Here, $S [ k _ { 1 } , k _ { 2 } ]$ measures the predictive utility of $\mathcal { T } _ { k _ { 1 } }$ for $\mathcal { T } _ { k _ { 2 } }$ and S is therefore not necessarily symmetric.

To translate this directional guidance into a form suitable for MTGP coupling estimation, a target-specific symmetric correlation guidance matrix

$$
\mathbf { R } _ { k } \in \mathbb { R } ^ { K \times K }\tag{22}
$$

is constructed for each target task $\mathcal { T } _ { k }$ . Its correlations involving the target task retain the corresponding directed guidance:

$$
R _ { k } [ j , k ] = R _ { k } [ k , j ] = S [ j , k ] , \qquad j \neq k .\tag{23}
$$

For two non-target tasks, both transfer directions are combined using their geometric mean:

$$
R _ { k } [ j , q ] = R _ { k } [ q , j ] = { \sqrt { S [ j , q ] S [ q , j ] } } , \qquad j , q \neq k .\tag{24}
$$

The diagonal entries are fixed to one:

$$
R _ { k } [ j , j ] = 1 , \qquad j = 1 , \ldots , K .\tag{25}
$$

The complete construction is therefore

$$
R _ { k } [ j , q ] = \left\{ \begin{array} { l l } { S [ j , k ] , } & { q = k , j \neq k , } \\ { S [ q , k ] , } & { j = k , q \neq k , } \\ { \sqrt { S [ j , q ] S [ q , j ] } , } & { j , q \neq k , j \neq q , } \\ { 1 , } & { j = q . } \end{array} \right.\tag{26}
$$

Since ${ \bf R } _ { k }$ is not necessarily positive semidefinite, it is projected onto the positive semidefinite cone before being used with the MTGP:

$$
\mathbf { R } _ { k } \gets \mathrm { m a k e \_ p s d } \left( \mathbf { R } _ { k } ; \varepsilon \right) , \qquad \varepsilon = 1 0 ^ { - 4 } ,\tag{27}
$$

where eigenvalues smaller than ε are clipped before reconstructing the matrix [17]. The resulting R preserves targetspecific directional guidance while satisfying the requirements of a valid task kernel. Importantly, ${ \bf R } _ { k }$ is not used directly as the MTGP coupling matrix; it provides reference guidance for estimating the inter-task coupling matrix in (5).

## D. Guidance Adaptation through MAP Estimation

The target-specific correlation guidance matrix ${ \bf R } _ { k }$ obtained in Section III-C is not directly used as the final task coupling matrix of the MTGP. Instead, it defines a reference coupling structure

$$
\mathbf { B } _ { \mathrm { g u i d a n c e } , k } = \mathbf { R } _ { k } .\tag{28}
$$

For each target task $\mathcal { T } _ { k } .$ , a separate MTGP is fitted using observations across all tasks, and the guidance is incorporated through a prior over its inter-task coupling matrix. Specifically, we assume

$$
\mathrm { v e c } \left( \mathbf { B } \right) \sim \mathcal { N } \left( \mathrm { v e c } \left( \mathbf { B } _ { \mathrm { g u i d a n c e } , k } \right) , \tau _ { B } ^ { 2 } \mathbf { I } \right) ,\tag{29}
$$

where $\tau _ { B } ^ { 2 }$ represents the uncertainty associated with the inferred guidance. The corresponding negative log-prior, up to a constant independent of B, is

$$
- \log p ( \mathbf { B } ) = \frac { 1 } { 2 \tau _ { B } ^ { 2 } } \left. \mathbf { B } - \mathbf { B } _ { \mathrm { g u i d a n c e } , k } \right. ^ { 2 } + \mathrm { c o n s t } .\tag{30}
$$

Let ϑ collect the coupling parameters, decision-space kernel parameters, and observation-noise parameters of the MTGP. Combining the conventional marginal likelihood with the guidance-informed prior gives the MAP objective

$$
\mathcal { T } _ { k } \left( \pmb { \vartheta } \right) = - \log p \left( \mathbf { y } \mid \mathbf { X } , \pmb { \vartheta } \right) + \lambda ( t ) \left. \mathbf { B } \left( \pmb { \vartheta } \right) - \mathbf { B } _ { \mathrm { g u i d a n c e } , k } \right. ^ { 2 } ,\tag{31}
$$

where B(ϑ) denotes the coupling matrix induced by the current MTGP parameters and

$$
\lambda = \frac { 1 } { 2 \tau _ { B } ^ { 2 } } .\tag{32}
$$

Thus, the guidance complements rather than replaces conventional marginal-likelihood estimation. A larger λ places greater emphasis on the reference coupling structure, whereas a smaller λ allows the observed optimization data to dominate the estimation [19]. The remaining MTGP parameters continue to be estimated conventionally from the observations.

1) Theoretical Inspiration: The strength of the guidance can be interpreted heuristically through a pseudo-observation analogy. Let $\sigma ^ { 2 }$ denote an effective observation variance associated with the coupling estimator [30]. The relative prior precision then corresponds to approximately

$$
n _ { 0 } = \frac { \sigma ^ { 2 } } { \tau _ { B } ^ { 2 } } = 2 \sigma ^ { 2 } \lambda\tag{33}
$$

virtual observations [30]. This quantity characterizes the relative information contributed by the guidance rather than an exact effective sample size of the complete MTGP. Under few-shot observations, such supplementary information can stabilize coupling estimation; as observations accumulate, the marginal likelihood provides increasingly informative statistical evidence [20]–[22].

2) Annealed Guidance Weight: Accordingly, the guidance weight is annealed throughout optimization:

$$
\lambda ( t ) = \lambda _ { 0 } \exp \left( - \delta t \right) , \qquad \lambda _ { 0 } = 1 . 0 , \qquad \delta = 0 . 0 5 ,\tag{34}
$$

where t denotes the optimization iteration and δ controls the decay rate. Under the pseudo-observation interpretation, the corresponding guidance strength becomes

$$
n _ { 0 } ( t ) = 2 \sigma ^ { 2 } \lambda ( t ) = 2 \sigma ^ { 2 } \lambda _ { 0 } \exp { ( - \delta t ) } \longrightarrow 0 \quad \mathrm { a s } \quad t \longrightarrow \infty .\tag{35}
$$

Hence, the guidance has its strongest influence during the early data-scarce stage and progressively diminishes as observations accumulate. In the limit, $\lambda ( t ) \ \to \ 0$ and $n _ { 0 } ( t ) \ \to \ 0$ , such that the estimation of the inter-task coupling matrix falls back to the conventional marginal-likelihood-based MTGP estimator. This annealing mechanism allows the in-context guidance to regularize coupling estimation when observations are scarce without permanently constraining the task relationships learned from the optimization data.

## E. Guided Multitask Bayesian Optimization

Once the task coupling matrix has been estimated through the MAP formulation, the resultant MTGP is applied in the conventional Bayesian optimization pipeline. Let $\mu _ { k } ( { \bf x } )$ and $\sigma _ { k } ( { \bf x } )$ denote the posterior mean and posterior standard deviation of the MTGP for task $\mathcal { T } _ { k }$ at solution x. These posterior quantities are then passed to a standard acquisition function to determine the next candidate solution.

1) Lower Confidence Bound: The main experiments employ a lower confidence bound (LCB) acquisition function [31]. On the normalized objective scale used by the implementation, the acquisition value for task $\mathcal { T } _ { m }$ is defined as

$$
\alpha _ { \mathrm { L C B } } \left( \mathbf { x } ; \mathcal { T } _ { k } \right) = \mu _ { k } \left( \mathbf { x } \right) - \sqrt { \beta } \sigma _ { k } \left( \mathbf { x } \right)\tag{36}
$$

where $\beta$ controls the trade-off between exploitation and exploration. The acquisition function therefore favors solutions with either a promising posterior mean or a high predictive uncertainty.

2) Overall ICG-MTBO procedure: Algorithm 1 summarizes the complete ICG-MTBO procedure. At each optimization iteration, the collected observations are first transformed into the in-context queries. The frozen TabPFN then performs cross-task in-context learning to generate in-context guidance signals. These values are translated into target-specific reference coupling matrices, which define guidance-informed priors for the annealed MAP estimation of the MTGP inter-task coupling matrices. Finally, a conventional acquisition function selects the next candidate solution for each active target task.

3) Computational Complexity: The dominant computational cost of each optimization iteration stems from fitting a single MTGP for each active target task. The in-context guidance module additionally requires $K ( K - 1 )$ TabPFN forward queries to evaluate all ordered task pairs. These computations do not consume the evaluation budget and are assumed to be substantially less costly than evaluating the underlying blackbox objective functions in the few-shot optimization settings.

## IV. RESULTS

To evaluate the effectiveness of the proposed algorithm, we compare it with representative baseline methods on both synthetic benchmark problems and a real-world application. The baselines include the single-task Bayesian optimizer (BO) [32], [33], the multitask Bayesian optimizer (MTBO) [9], and several state-of-the-art few-shot multitask optimizers, namely BO-LCB-CKT [34], BO-LCB-BCKT [35], and the evolutionary multitask Bayesian optimizer SELF [36]. By default, the LCB acquisition function is adopted for all methods to ensure consistency with the publicly available implementations of BO-LCB-CKT, BO-LCB-BCKT, and SELF [34]–[36].

To further assess the generality of the conclusions, we also compare the proposed method with STBO and MTBO using the acquisition function, LogEI [37], a numerically stable reformulation of Expected Improvement [38]. It is worth noting that the proposed framework is not restricted to a specific multitask optimizer. In the methodology, we instantiate ICG-MTBO by introducing in-context guidance into the estimation of the MTBO task coupling matrix. To further demonstrate the generality of the framework, we additionally develop a variant based on the well-established evolutionary multitask optimizer MFEA-II [4], in which in-context guidance is applied to estimate the random mating probability (rmp) matrix and thereby improve its multitask search performance. Finally, the stability of the proposed method is examined through a sensitivity analysis.

## A. Benchmark Definitions

1) Synthetic Benchmarks: We first conduct comparative studies on the single-objective multitask benchmark suite [39]. It comprises nine multitask optimization problems, each containing two tasks whose relationship is characterized along two features, that is, the intersection of their global optima and the similarity of their search spaces. We evaluate two different versions of synthetic problems. We evaluate the

Algorithm 1: In-Context Guidance Multitask Bayesian   
Optimization   
Input: Optimization tasks $\{ \mathcal { T } _ { k } \} _ { k = 1 } ^ { K } ;$ evaluation budget   
N per task; number of fitness classes $C ;$ initial   
guidance strength $\lambda _ { 0 } ;$ decay rate $\delta ;$ calibration   
parameter $\tau ;$ trade-off parameter $\beta .$   
Output: Best solution for each optimization task.   
1 Initialize each dataset $\mathcal { D } _ { k }$ using evaluated   
Latin-hypercube samples;   
2 Set optimization iteration $t \gets 0 ;$   
3 while $\textstyle \sum _ { k = 1 } ^ { K } N _ { k } < K N$ do   
4 Identify the active task set $\mathcal { A }  \{ \mathcal { T } _ { k } : N _ { k } < N \}$   
5 Compute the annealed guidance weigh   
$\lambda ( t ) \gets \lambda _ { 0 } \exp ( - \delta t ) ;$   
6 Normalize the collected objective values;   
// Step 1: In-Context Query   
Construction   
7 foreach task $\mathcal { T } _ { k }$ do   
8 Construct fitness-class labels using (13);   
9 Form the classified dataset $\widetilde { \mathcal { D } } _ { k } ;$   
10 end   
// Step 2: Learning In-Context   
Guidance   
11 foreach ordered task pair $( k _ { 1 } , k _ { 2 } )$ with $\boldsymbol { k } _ { 1 } \neq \boldsymbol { k } _ { 2 }$ do   
12 Present $\widetilde { \mathcal { D } } _ { k _ { 1 } }$ as context to the frozen TabPFN;   
13 Predict the fitness classes of task $\mathcal { T } _ { k _ { 2 } } \mathrm { ~ : ~ }$   
14 Compute $\mathrm { C E } [ k _ { 1 } , k _ { 2 } ]$ using (18);   
15 Calibrate $S [ k _ { 1 } , k _ { 2 } ]$ using (20);   
16 end   
17 foreach active target task $\mathcal T _ { k } \in \mathcal A$ do   
18 Construct the target-specific correlation   
guidance matrix ${ \bf R } _ { k }$ using (26);   
19 Project ${ \bf R } _ { k }$ onto the positive-semidefinite cone;   
$/ /$ Step 3: Guidance adaptation   
through MAP Estimation   
20 Initialize a fresh MTGP using the observations   
shared across all tasks;   
21 Initialize the task kernel from ${ \bf R } _ { k }$ and obtain   
$\mathbf { B } _ { \mathrm { g u i d a n c e } , k } ;$   
22 Fit the MTGP by minimizing (31);   
$/ /$ Step 4: Guided Multitask   
Bayesian optimization   
23 Optimize $\alpha _ { \mathrm { L C B } }$ in (36) for task $\mathcal { T } _ { k } ;$   
24 Evaluate the selected solution and update $\mathcal { D } _ { k }$ ;   
25 end   
26 Update $t \gets t + 1 ;$   
27 end   
28 return The best solution from each $\mathcal { D } _ { k } ;$

50-dimensional problems as originally defined, and a 30- dimensional counterpart in which each task is restricted to the first 30 dimensional variables of the full 50 dimensional landscape that passes through the global optimum. Detailed settings can be found in Section S-I in supplementary materials.

![](images/b56520baa9763dd1d6d9589afb876bbaae65e5449623e1f87e3f0ff32ff15a24.jpg)  
Fig. 2. Illustration of the separable robot-arm reaching problem (SepArm). The decision variables determine the normalized angular commands of the joints, which are mapped to the corresponding physical joint angles. The objective is to minimize the distance between the end-effector position and a fixed target. Across the multitask problem bundle, the tasks share the same target and total arm length but differ in the allowable angular range of the joints, resulting in different levels of inter-task similarity.

2) Real Problems: For the real-world study, we adopt the separable robot-arm reaching problem (SepArm) [40], in which the angular positions of the joints are adjusted so that the end effector reaches a fixed target as closely as possible<sup>1</sup>. As shown in Fig. 2, the solution is the vector of normalized joint commands $\mathbf { x } \in [ 0 , 1 ] ^ { V }$ , each entry mapped to a physical angle in $[ - \pi a _ { \mathrm { m a x } } , \pi a _ { \mathrm { m a x } } ] ,$ and the objective quantifies the distance between the end effector position and the target. We construct three sets of problems where each set represents high similarity (HS), medium similarity (MS), and low similarity (LS) multitask problems across 5-D, 10- D, and 15-D search spaces, yielding nine problems P1-P9: 5-D HS/MS/LS (P1-P3), 10-D HS/MS/LS (P4-P6), and 15-D HS/MS/LS (P7-P9). More details can be found in Section S-I in our supplementary materials.

## B. Experimental Setup

All experiments follow the few-shot multitask settings, where each task is generally allocated a total budget of 100 function evaluations. For the synthetic benchmarks, every optimizer is initialized with 20 samples per task generated by Latin hypercube sampling (LHS) [41], leaving 80 iterative evaluations for the optimization phase. However, for the real SepArm problems, a smaller initial design of 5 samples per task is adopted for the lower-dimensional problems, and the total budget for each task is set to 60 function evaluations. All methods are assigned an identical evaluation budget, and each experiment is independently repeated 20 times. Performance is reported as the best objective value found so far, and the convergence curves show the mean across all runs, with a shaded band representing a standard deviation.

For the compared GP-based methods (BO, MTBO, BO-LCB-CKT, BO-LCB-BCKT, and SELF), the surrogate is a Gaussian process with an RBF ARD kernel, the inputs are normalized to $[ 0 , 1 ] ^ { V }$ , the objectives are min-max normalized, and the hyperparameters are fitted by maximizing the exact marginal likelihood. The acquisition function is optimized by multi-start gradient ascent with 5 restarts and 200 steps per restart. The LCB exploration coefficient is set to $\beta \ =$ 2.5 for the GP baselines. The remaining hyperparameters of the baselines follow their original publications. For the proposed ICG-MTBO, the in-context guidance is produced by TabPFN v2.5 [42], a tabular foundation model pretrained on synthetic datasets, accessed through the official implementation (tabpfn package, version 7.0.1) No fine-tuning is performed, and a single forward pass is utilized per estimate. The MAP estimate of the task coupling matrix is obtained with an initial regularization weight $\lambda _ { 0 } = 1$ that decays exponentially at rate 0.05 per iteration, so that the in-context guidance dominates in the data-scarce early stage and gradually falls back to the likelihood as observations accumulate.

TABLE I  
STATISTICAL SUMMARY OF THE COMPARATIVE RESULTS ON THE 30-DIMENSIONAL SYNTHETIC BENCHMARK PROBLEMS. THE +/ − / ≈ ENTRIES REPORT THE NUMBERS OF TASKS ON WHICH EACH METHOD PERFORMS SIGNIFICANTLY BETTER THAN, SIGNIFICANTLY WORSE THAN, OR STATISTICALLY SIMILARLY TO ICG-MTBO, RESPECTIVELY, ACCORDING TO THE WILCOXON TEST AT A SIGNIFICANCE LEVEL OF 0.05. THE BEST AVERAGE RANK AT EACH EVALUATION BUDGET IS HIGHLIGHTED IN BOLD WITH A SHADED BACKGROUND.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Evaluations</td><td rowspan=1 colspan=1>BO-LCB</td><td rowspan=1 colspan=1>MTBO-LCB</td><td rowspan=1 colspan=1>BO-LCB-CKT</td><td rowspan=1 colspan=1>BO-LCB-BCKT</td><td rowspan=1 colspan=1>SELF</td><td rowspan=1 colspan=1>ICG-MTBO</td></tr><tr><td rowspan=4 colspan=1> $+ / - / \approx$ </td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0/16/2</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>4/12/2</td><td rowspan=1 colspan=1>0/17/1</td><td rowspan=1 colspan=1>4/6/8</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>1/13/4</td><td rowspan=1 colspan=1>3/13/2</td><td rowspan=1 colspan=1>0/17/1</td><td rowspan=1 colspan=1>1/7/10</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>0/16/2</td><td rowspan=1 colspan=1>2/9/7</td><td rowspan=1 colspan=1>1/15/2</td><td rowspan=1 colspan=1>0/18/0</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>2/8/8</td><td rowspan=1 colspan=1>1/15/2</td><td rowspan=1 colspan=1>0/18/0</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=1>Average rank</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>4.94</td><td rowspan=1 colspan=1>3.11</td><td rowspan=1 colspan=1>3.44</td><td rowspan=1 colspan=1>5.11</td><td rowspan=1 colspan=1>2.50</td><td rowspan=1 colspan=1>1.89</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>5.06</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>3.94</td><td rowspan=1 colspan=1>5.00</td><td rowspan=1 colspan=1>2.61</td><td rowspan=1 colspan=1>1.61</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>4.83</td><td rowspan=1 colspan=1>2.39</td><td rowspan=1 colspan=1>4.28</td><td rowspan=1 colspan=1>5.06</td><td rowspan=1 colspan=1>2.89</td><td rowspan=1 colspan=1>1.56</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>4.78</td><td rowspan=1 colspan=1>2.17</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>5.06</td><td rowspan=1 colspan=1>3.28</td><td rowspan=1 colspan=1>1.56</td></tr></table>

TABLE II

STATISTICAL SUMMARY OF THE COMPARATIVE RESULTS ON THE 50-DIMENSIONAL SYNTHETIC BENCHMARK PROBLEMS. THE +/ − / ≈ ENTRIES REPORT THE NUMBERS OF TASKS ON WHICH EACH METHOD PERFORMS SIGNIFICANTLY BETTER THAN, SIGNIFICANTLY WORSE THAN, OR STATISTICALLY SIMILARLY TO ICG-MTBO, RESPECTIVELY, ACCORDING TO THE WILCOXON TEST AT A SIGNIFICANCE LEVEL OF 0.05. THE BEST AVERAGE RANK AT EACH EVALUATION BUDGET IS HIGHLIGHTED IN BOLD WITH A SHADED BACKGROUND.
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Evaluations</td><td rowspan=1 colspan=1>BO-LCB</td><td rowspan=1 colspan=1>MTBO-LCB</td><td rowspan=1 colspan=1>BO-LCB-CKT</td><td rowspan=1 colspan=1>BO-LCB-BCKT</td><td rowspan=1 colspan=1>SELF</td><td rowspan=1 colspan=1>ICG-MTBO</td></tr><tr><td rowspan=4 colspan=1> $+ / - / \approx$ </td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>0/13/5</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/7/11</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>1/15/2</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>3/6/9</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/17/1</td><td rowspan=1 colspan=1>1/10/7</td><td rowspan=1 colspan=1>1</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0/17/1</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>0/17/1</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=1>Average rank</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>4.22</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>4.56</td><td rowspan=1 colspan=1>4.83</td><td rowspan=1 colspan=1>2.72</td><td rowspan=1 colspan=1>1.67</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>3.11</td><td rowspan=1 colspan=1>4.78</td><td rowspan=1 colspan=1>5.17</td><td rowspan=1 colspan=1>1.94</td><td rowspan=1 colspan=1>1.83</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>4.44</td><td rowspan=1 colspan=1>3.00</td><td rowspan=1 colspan=1>4.61</td><td rowspan=1 colspan=1>5.28</td><td rowspan=1 colspan=1>2.17</td><td rowspan=1 colspan=1>1.50</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>4.39</td><td rowspan=1 colspan=1>2.56</td><td rowspan=1 colspan=1>4.78</td><td rowspan=1 colspan=1>5.28</td><td rowspan=1 colspan=1>2.61</td><td rowspan=1 colspan=1>1.39</td></tr></table>

One can refer to Section S-I in supplementary materials for the full details of generality study on acquisition function and MFEA-II.

## C. Discussions

1) Competitive Optimization Performance: Tables S-I–S-IV show that ICG-MTBO achieves consistently competitive performance across both the 30- and 50-dimensional benchmark suites. Summarized results can be found in Table I–II. In particular, it obtains the best average rank at every evaluation budget, with ranks improving from 1.89 to 1.56 in the 30- dimensional case and from 1.67 to 1.39 in the 50-dimensional case, showcasing that in higher-dimensional cases where the evaluation budgets are more limited, ours can achieve a larger performance margin. Although several competing methods outperform ICG-MTBO on a small number of individual tasks, the proposed method is significantly better than or statistically comparable to them in the majority of comparisons, demonstrating stable performance across different problem dimensions and stages of the optimization process.

2) The Effectiveness ofIn-Context Guidance: A direct comparison between ICG-MTBO and its unguided counterpart, MTBO-LCB, provides an ablation study of the proposed in-context guidance mechanism. Across the 30-dimensional benchmarks, ICG-MTBO performs better than, statistically similarly to, and worse than MTBO-LCB in 44, 23, and 5 out of 72 comparisons, respectively. The advantage becomes more pronounced in the 50-dimensional setting, where the corresponding counts are 57, 14, and 1. Overall, across both dimensions and all evaluation budgets, ICG-MTBO achieves better, comparable, and worse performance in 101, 37, and 6 out of 144 comparisons, respectively. Notably, at the earliest budget of 40 evaluations, ICG-MTBO is better in 27 comparisons and comparable in the remaining 9, without being significantly worse in any case. This confirms that the ICG component is particularly effective during the early few-shot stage, when the conventional likelihood-based estimate of inter-task coupling remains unreliable. As the optimization proceeds, especially in the lower-dimensional setting, the performance gap gradually narrows, and the proposed method increasingly becomes statistically comparable to MTBO-LCB. This trend is consistent with the annealing design, where the influence of the in-context guidance is progressively reduced, and the optimizer gradually falls back toward the behavior of standard MTBO-LCB once sufficient observations have been accumulated.

Nevertheless, the occasional inferior results also reveal the limitation of the guidance mechanism. The fitness-class query

## TABLE III

STATISTICAL SUMMARY OF THE COMPARATIVE RESULTS USING THE LOGEI ACQUISITION FUNCTION ON THE 50-DIMENSIONAL SYNTHETIC   
BENCHMARK PROBLEMS. THE +/ − / ≈ ENTRIES REPORT THE NUMBERS   
OF TASKS ON WHICH EACH METHOD PERFORMS SIGNIFICANTLY BETTER   
THAN, SIGNIFICANTLY WORSE THAN, OR STATISTICALLY SIMILARLY TO   
ICG-MTBO, RESPECTIVELY, ACCORDING TO THE WILCOXON TEST AT A SIGNIFICANCE LEVEL OF 0.05. THE BEST AVERAGE RANK AT EACH EVALUATION BUDGET IS HIGHLIGHTED IN BOLD WITH A SHADED BACKGROUND.

<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Evaluations</td><td rowspan=1 colspan=1>BO-LogEI</td><td rowspan=1 colspan=1>MTBO-LogEI</td><td rowspan=1 colspan=1>ICG-MTBO-LogEI</td></tr><tr><td rowspan=4 colspan=1>+/-/≈</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>2/13/3</td><td rowspan=1 colspan=1>1/11/6</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>1/13/4</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>0/14/4</td><td rowspan=1 colspan=1>2/12/4</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=1>Average rank</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>2.50</td><td rowspan=1 colspan=1>2.11</td><td rowspan=4 colspan=1>1.391.281.281.39</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>2.61</td><td rowspan=1 colspan=1>2.11</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>2.56</td><td rowspan=1 colspan=1>2.17</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>2.67</td><td rowspan=1 colspan=1>1.94</td></tr></table>

## TABLE IV

STATISTICAL SUMMARY OF THE COMPARATIVE RESULTS USING THE LOGEI ACQUISITION FUNCTION ON THE 30-DIMENSIONAL SYNTHETIC   
BENCHMARK PROBLEMS. THE +/ − / ≈ ENTRIES REPORT THE NUMBERS   
OF TASKS ON WHICH EACH METHOD PERFORMS SIGNIFICANTLY BETTER   
THAN, SIGNIFICANTLY WORSE THAN, OR STATISTICALLY SIMILARLY TO   
ICG-MTBO, RESPECTIVELY, ACCORDING TO THE WILCOXON TEST AT A SIGNIFICANCE LEVEL OF 0.05. THE BEST AVERAGE RANK AT EACH EVALUATION BUDGET IS HIGHLIGHTED IN BOLD WITH A SHADED BACKGROUND.

<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>Evaluations</td><td rowspan=1 colspan=1>BO-LogEI</td><td rowspan=1 colspan=1>MTBO-LogEI</td><td rowspan=1 colspan=1>ICG-MTBO-LogEI</td></tr><tr><td rowspan=4 colspan=1>+/-/≈</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>0/15/3</td><td rowspan=1 colspan=1>1/10/7</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>2/11/5</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>1/14/3</td><td rowspan=1 colspan=1>3/9/6</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>2/10/6</td><td rowspan=1 colspan=1>3/7/8</td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=4 colspan=1>Average rank</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>2.89</td><td rowspan=1 colspan=1>1.94</td><td rowspan=4 colspan=1>1.171.221.441.67</td></tr><tr><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>2.00</td></tr><tr><td rowspan=1 colspan=1>80</td><td rowspan=1 colspan=1>2.72</td><td rowspan=1 colspan=1>1.83</td></tr><tr><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>2.56</td><td rowspan=1 colspan=1>1.78</td></tr></table>

in ICG-MTBO provides only a coarse representation of the task landscapes, and the resultant in-context guidance may not accurately characterize complex or weakly transferable inter-task relationships. Consequently, inaccurate in-context guidance can temporarily bias the estimated task coupling matrix. The overall results therefore suggest that the main benefit of ICG lies in stabilizing and improving inter-task relationship estimation during the early few-shot stage, while its influence appropriately diminishes as the data-driven estimator of MTBO-LCB becomes more reliable.

3) Results on Real Problems: As shown in Table S-IX in supplementary materials, ICG-MTBO achieves the best overall average rank of 1.44, compared with 2.37 for BO-LCB and 2.19 for MTBO-LCB. Compared to BO-LCB, ICG-MTBO can achieve significantly better, statistically comparable, and significantly worse results on 8, 18, and 1 out of the 27 tasks, respectively. Against the unguided MTBO-LCB, the corresponding results are 7, 19, and 1, indicating that the proposed in-context guidance generally improves or preserves the performance of the underlying multitask optimizer.

The comparison of ICG-MTBO to MTBO-LCB further reveals how the benefit varies with problem dimensionality and inter-task similarity. On the 5-dimensional problems from P1 to P3, ICG-MTBO is significantly better on 3 tasks and statistically comparable on the remaining 6. On the

10-dimensional problems from P4 to P6, it records 2/6/1 better/comparable/worse results, while on the 15-dimensional problems from P7 to P9, it achieves 2/7/0. Thus, the proposed method remains competitive across all three problem scales, although its clearest advantage appears in the lowerdimensional setting. More importantly, the improvements mainly revolve around the task group with stronger intertask relatedness. For the HS problems P1, P4, and P7, ICG-MTBO achieves 4/5/0 better/comparable/worse results against MTBO-LCB, while the corresponding result is 3/6/0 for the MS problems P2, P5, and P8. In contrast, on the LS problems P3, P6, and P9, the result becomes 0/8/1. This trend suggests that the ICG component is most effective when the tasks possess sufficiently overlapping solution structures that can be identified from limited observations. When the tasks are weakly related, the guidance provides less additional benefit, but the annealed formulation generally allows ICG-MTBO to remain statistically comparable to standard MTBO-LCB rather than causing persistent negative transfer.

4) Generality Study on a Different Acquisition Function: To examine whether the proposed framework depends on a particular acquisition function, we replace LCB with LogEI while retaining the same surrogate configuration and in-context guidance mechanism. As shown in Tables S-IV–S-VIII and Tables III–IV, ICG-MTBO-LogEI achieves the best average rank at all four evaluation budgets, with ranks of 1.17, 1.22, 1.44, and 1.67 after 40, 60, 80, and 100 evaluations, respectively. Compared with its unguided counterpart, MTBO-LogEI, the proposed method performs significantly better, statistically similarly, and significantly worse in 37, 26, and 9 out of the 72 comparisons, respectively. At the earliest budget of 40 evaluations, ICG-MTBO-LogEI is better in 10 cases, comparable in 7, and worse in only 1, again demonstrating the value of in-context guidance when the task-coupling estimator is supported by only limited observations.

The advantage gradually decreases as the optimization proceeds: the better/comparable/worse counts change from 10/7/1 at 40 evaluations to 7/8/3 at 100 evaluations. This trend is consistent with the annealed MAP formulation, where the influence of the in-context guidance is strongest during the early few-shot stage and progressively diminishes as the accumulated observations allow the conventional MTBO estimator to become more reliable. Despite this gradual fallback toward standard MTBO-LogEI, the proposed method remains the best-ranked approach throughout the optimization process. These results indicate that the benefit of ICG-MTBO is not tied to LCB. Instead, the proposed framework improves the estimation of inter-task coupling independently of the subsequent acquisition strategy and can therefore be integrated with different Bayesian optimization criteria [32], [33], [38], [43] without modifying their original candidate selection mechanisms.

5) Generality Study on an Evolutionary Multitask Optimizer: To demonstrate that the proposed in-context guidance multitask optimization is not tied to the specific multitask Bayesian optimization, we further instantiate the framework on the well-known evolutionary multitask optimizer MFEA-II [4]. In this instantiation, the inter-task relationship takes the form $\vartheta \equiv \mathbf { R }$ , where $\mathbf { R } \ = \ [ r _ { i j } ]$ is the random mating probability (RMP) matrix that governs the intensity of intertask genetic transfer. Conventional MFEA-II estimates R online per generation from the current parent subpopulations $\mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { M }$ by maximizing the likelihood of a mixture model constructed in the unified decision space [4]:

![](images/97aa6bf995aa98adefedeac8ea56db97510eeb39eddd204e8f8066db4f338248.jpg)

![](images/8f57d4f73cc5046244fb9848c3adf5918d95b71910e08ad21cd23f4d250bc88e.jpg)

![](images/f1171def7305a0f2f8137c6af1d8d947fc38a5949825288e3005fec1f55c9f02.jpg)

![](images/972f16b2fc31fbe2499cbc2f22877b519504f45b8964f4a77d8c7d520bce062f.jpg)

(a) P2 (CIMS)  
![](images/2183dad0e0587114bbd86162ae6227cc1705863e61a490992a77390dfba03948.jpg)  
(c) P6 (PILS)

(b) P4 (PIHS)  
![](images/af72c139c921d1244d1b94757d3eb05e6b60ebc43387488dca312fb004ddd1e0.jpg)  
(d) P8 (NIMS)  
Fig. 3. Convergence trends of BO-LCB, MTBO-LCB, BO-LCB-CKT, BO-LCB-BCKT, SELF, and ICG-MTBO on the 30-dimensional synthetic benchmark problems. (a) P2 (CIMS). (b) P4 (PIHS). (c) P6 (PILS). (d) P8 (NIMS).

$$
\widehat { \mathbf { R } } _ { \mathrm { M L E } } = \arg \operatorname* { m a x } _ { \mathbf { R } } \mathcal { L } _ { \mathrm { M F E A - I I } } \left( \mathbf { R } ; \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { K } \right) .\tag{37}
$$

Although this data-driven estimator can enable an adaptive knowledge transfer behavior, it relies on the current populations and can therefore be unreliable in the early generations, when the subpopulations are small and have not yet concentrated to promising regions. This corresponds to the same fewshot limitation that motivates the Bayesian instantiation in III.

Following the generic ICG-MTO formulation, an algorithmspecific in-context query can be constructed from the evaluated solutions. Since the RMP matrix determines whether genetic transfer between the solutions of two tasks is likely to produce valuable offspring, the relevant signal can be modeled as the overlap between their promising regions in the decision space. At generation t, the elite subset of task $\mathcal { T } _ { k }$ is defined as:

$$
\mathcal { E } _ { k } ^ { ( t ) } = \mathrm { E l i t e } _ { \alpha } \left( \mathcal { P } _ { k } ^ { ( t ) } \right) ,\tag{38}
$$

where $\gamma$ denotes the elite ratio and the top-γ fraction of solutions with the best objective values is retained. For each task pair $( \mathcal { T } _ { i } , \mathcal { T } _ { j } )$ , the two elite sets are set to the same size and pooled to form a task-membership classification query. Each decision vector is labeled according to its original task, while the objective values are used only for elite selection and are not included as input features or predictive labels.

The frozen TabPFN performs in-context classification on a held-out split of this query, thereby serving as a classifier to distinguish samples from the two elite distributions. Let $\mathrm { C E } _ { i j } ^ { ( t ) }$ denote the resulting mean predictive cross-entropy. The in-context guidance signal for the task pair is defined as

$$
\rho _ { i j } ^ { ( t ) } = \operatorname* { m i n } \left( \frac { \mathrm { C E } _ { i j } ^ { ( t ) } } { \log 2 } , 1 \right) \in [ 0 , 1 ] .\tag{39}
$$

A cross-entropy close to log 2 indicates that the two elite sets cannot be reliably distinguished and therefore exhibit substantial decision-space overlap, yielding $\rho _ { i j } ^ { ( t ) } \approx 1$ . Conversely, a cross-entropy close to zero indicates well-separated promising regions and yields $\rho _ { i j } ^ { ( t ) } \approx 0$ . Collecting all pairwise guidance values gives the in-context guidance matrix:

$$
\mathbf { R } _ { \mathrm { F M } } ^ { \left( t \right) } = \left[ \rho _ { i j } ^ { \left( t \right) } \right] _ { i , j = 1 } ^ { K } .\tag{40}
$$

The guided estimator follows the same MAP structure as the generic formulation in Equation (2):

$$
\widehat { \mathbf { R } } _ { \mathrm { M A P } } = \arg \operatorname* { m a x } _ { \mathbf { R } } \left[ \mathcal { L } \left( \mathbf { R } ; \mathcal { P } _ { 1 } , \ldots , \mathcal { P } _ { K } \right) - \lambda ( t ) \left. \left. \mathbf { R } - \mathbf { R } _ { \mathrm { F M } } ^ { ( t ) } \right. \right. ^ { 2 } \right] .\tag{41}
$$

Since MFEA-II estimates the RMP independently for each task pair, the implementation solves the scalar problem

$$
\boldsymbol { \widehat { r } } _ { i j } ^ { ( t ) } = \arg \operatorname* { m a x } _ { \boldsymbol { r } _ { i j } \in [ 0 , 1 ] } \left[ \mathcal { L } _ { i j } \left( \boldsymbol { r } _ { i j } ; \boldsymbol { \mathcal { P } } _ { i } ^ { ( t ) } , \boldsymbol { \mathcal { P } } _ { j } ^ { ( t ) } \right) - \lambda ( t ) \left( \boldsymbol { r } _ { i j } - \boldsymbol { \rho } _ { i j } ^ { ( t ) } \right) ^ { 2 } \right] .\tag{42}
$$

The pairwise likelihood is normalized by the number of parents so that the guidance weight remains on a scale comparable to that used in the Bayesian instantiation.

![](images/6d99856bfc1aea453e44358c5d390815c7b44cd31ee3467ea1da88362bdb91c0.jpg)

![](images/2d45b448031a38d988df7cd7632b24bd7e239fbd7befdb63a4fcbf826e147876.jpg)

![](images/a0eff93e4e20cb931a0de6344ee41597b8f526b29abf474f2448618b3fca92cd.jpg)

![](images/58d281dc1bb1281501d9e9df2f96ee644ff3d984ad2e02c7beea23b8728db6a0.jpg)

![](images/35ac492ce2300f768bddc71fd7846b7cf0e51880e6d790afffa5abba9b56afa0.jpg)

(a) P1 (CIHS)  
![](images/8f1f00c031c6a4df33ac2c1e3ee48b76e9191bd18501a43cc0e22b16c3cffdb8.jpg)

![](images/8a26baeb769a1cd4ee86b69dff4f3ddd8ea9abe72da93e91a3e56d8bb386dbdd.jpg)  
(b) P6 (PILS)  
(c) P7 (NIHS)  
(d) P9 (NILS)  
Fig. 4. Convergence trends of BO-LCB, MTBO-LCB, BO-LCB-CKT, BO-LCB-BCKT, SELF, and ICG-MTBO on the 50-dimensional synthetic benchmark problems. (a) P1 (CIHS). (b) P6 (PILS). (c) P7 (NIHS). (d) P9 (NILS).

The guidance strength follows the same annealing schedule:

$$
\lambda ( t ) = \lambda _ { 0 } \exp ( - \delta t ) .\tag{43}
$$

The in-context guidance therefore has its greatest influence during the data-scarce early generations and gradually yields to the population-based likelihood as the subpopulations mature. As $\lambda ( t ) \to 0$ , the estimator recovers conventional MFEA-II exactly. The resulting algorithm is denoted ICG-MFEA-II.

The two variants are compared on the 30-dimensional synthetic benchmark suite with a population size of 20 per task and a budget of 1200 evaluations per task. To ensure a fair comparison, both methods follow the same initialization protocol. A shared space-filling LHS [41] design of 200 points per task is evaluated once, the best 20 solutions form the initial population, and the same evaluated design provides the initial in-context queries for ICG-MFEA-II. Since both methods incur the same initialization cost, they execute the same number of generations, namely 50, and differ only in the estimator of the RMP matrix.

The MAP configuration $( \lambda _ { 0 } ~ = ~ 1 , \delta ~ = ~ 0 . 0 5 )$ is inherited directly from ICG-MTBO without retuning. The elite ratio is examined at $\gamma \in \{ 0 . 5 , 0 . 3 , 0 . 1 \}$ , with the corresponding variants denoted ICG-MFEA-II-50, ICG-MFEA-II-30, and ICG-MFEA-II-10, respectively. As shown in Figure 5 and Table V, the guided variants generally improve the convergence of MFEA-II, with the most evident gains appearing in the early generations, when the population-based mixture-model estimator is least reliable.

6) Sensitivity Analysis: The sensitivity of ICG-MTBO to the MAP schedule $\lambda ( t ) = \lambda _ { 0 } e ^ { - \delta t }$ , shown in (35), is examined over six configurations as shown in Table S-XI, in the supplementary materials. In Table S-XI, the setting $( \lambda _ { 0 } { = } 1 , \delta { = } 0 . 0 5 )$ attains the best average rank at iteration 60. The detailed analysis and discussion can be found in Section S-II in supplementary materials.

![](images/c5958f6884633a3a88178d5b1da91a994f8e472a2039f5117a216af9838a49bb.jpg)  
Fig. 5. Rank-based comparison of MFEA-II and ICG-MFEA-II with elite ratios of 0.5, 0.3, and 0.1 on the synthetic benchmark problems P1-P9. For each task, the four methods are ranked by their mean objective values over 20 independent trials, with lower values receiving better ranks and ties receiving average ranks. The radial axis is reversed so rank 1 is outermost; points farther from the center therefore indicate better relative performance.

TABLE V  
COMPARATIVE RESULTS OF MFEA-II AND ICG-MFEA-II WITH DIFFERENT ELITE RATIOS ON THE SYNTHETIC BENCHMARK PROBLEMS. THE RESULTS ARE REPORTED AS THE MEAN AND STANDARD DEVIATION OVER 20 INDEPENDENT TRIALS. THE SYMBOLS +, −, AND ≈ INDICATE THAT THE CORRESPONDING METHOD PERFORMS SIGNIFICANTLY BETTER THAN, SIGNIFICANTLY WORSE THAN, OR STATISTICALLY SIMILARLY TO ICG-MFEA-II WITH AN ELITE RATIO OF 0.1, RESPECTIVELY, ACCORDING TO THE WILCOXON TEST AT A SIGNIFICANCE LEVEL OF 0.05. THE BEST MEAN VALUE FOR EACH TASK IS HIGHLIGHTED IN BOLD WITH A SHADED BACKGROUND.
<table><tr><td rowspan=1 colspan=1>Problem</td><td rowspan=1 colspan=1>Task</td><td rowspan=1 colspan=1>MFEA-II</td><td rowspan=1 colspan=1>ICG-MFEA-II-(0.5)</td><td rowspan=1 colspan=1>ICG-MFEA-II-(0.3)</td><td rowspan=1 colspan=1>ICG-MFEA-II-(0.1)</td></tr><tr><td rowspan=2 colspan=1>P1</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.773e+00 (2.2e-01)</td><td rowspan=1 colspan=1>1.395e+00 (5.0e-02) ≈</td><td rowspan=1 colspan=1>1.493e+00 (3.1e-01) ≈</td><td rowspan=2 colspan=1>1.332e+00 (9.0e-02)7.595e+02 (6.7e+01)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.168e+03 (3.1e+02)</td><td rowspan=1 colspan=1>8.741e+02 (5.4e+01)</td><td rowspan=1 colspan=1>8.925e+02 (2.9e+02) ≈</td></tr><tr><td rowspan=2 colspan=1>P2</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.358e+01 (1.4e+00)</td><td rowspan=1 colspan=1>9.976e+00 (9.2e-01) ≈</td><td rowspan=1 colspan=1>1.109e+01 (9.2e-01) ≈</td><td rowspan=1 colspan=1>1.115e+01 (6.9e-01)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.223e+03 (2.4e+02)</td><td rowspan=1 colspan=1>8.047e+02 (7.7e+01) ≈</td><td rowspan=1 colspan=1>9.163e+02 (1.6e+02) ≈</td><td rowspan=1 colspan=1>9.464e+02 (1.4e+02)</td></tr><tr><td rowspan=2 colspan=1>P3</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>2.132e+01 (5.2e-02) ≈</td><td rowspan=1 colspan=1>2.131e+01 (4.3e-02) ≈</td><td rowspan=1 colspan=1>2.135e+01 (4.6e-02) ≈</td><td rowspan=1 colspan=1>2.135e+01 (3.4e-02)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3.968e+03 (3.2e+02) +</td><td rowspan=1 colspan=1>4.274e+03 (4.1e+02) ≈</td><td rowspan=1 colspan=1>4.495e+03 (4.4e+02) ≈</td><td rowspan=1 colspan=1>4.482e+03 (1.6e+02)</td></tr><tr><td rowspan=2 colspan=1>P4</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.245e+03 (2.3e+02) ≈</td><td rowspan=1 colspan=1>9.289e+02 (2.1e+02) ≈</td><td rowspan=1 colspan=1>1.016e+03 (1.5e+02) ≈</td><td rowspan=1 colspan=1>9.243e+02 (7.2e+01)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2.809e+03 (7.5e+02) ≈</td><td rowspan=1 colspan=1>2.018e+03 (7.9e+02) ≈</td><td rowspan=1 colspan=1>2.493e+03 (1.2e+03) ≈</td><td rowspan=1 colspan=1>1.979e+03 (6.6e+02)</td></tr><tr><td rowspan=2 colspan=1>P5</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.328e+01 (1.1e+00)</td><td rowspan=1 colspan=1>1.184e+01 (8.4e-01) ≈</td><td rowspan=1 colspan=1>1.249e+01 (7.7e-01)</td><td rowspan=1 colspan=1>1.141e+01 (6.7e-01)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2.138e+07 (1.7e+07)</td><td rowspan=1 colspan=1>3.039e+06 (1.8e+06) ≈</td><td rowspan=1 colspan=1>5.019e+06 (1.6e+06) ≈</td><td rowspan=1 colspan=1>5.355e+06 (3.8e+06)</td></tr><tr><td rowspan=2 colspan=1>P6</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.308e+01 (1.4e+00) ≈</td><td rowspan=1 colspan=1>1.298e+01 (1.1e+00) ≈</td><td rowspan=1 colspan=1>1.267e+01 (7.2e-01) ≈</td><td rowspan=1 colspan=1>1.298e+01 (1.1e+00)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2.479e+01 (2.9e+00) ≈</td><td rowspan=1 colspan=1>2.452e+01 (1.8e+00) ≈</td><td rowspan=1 colspan=1>2.499e+01 (2.1e+00) ≈</td><td rowspan=1 colspan=1>2.452e+01 (1.8e+00)</td></tr><tr><td rowspan=2 colspan=1>P7</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.947e+07 (1.0e+07)</td><td rowspan=1 colspan=1>4.415e+06 (3.1e+06) ≈</td><td rowspan=1 colspan=1>3.451e+06 (2.5e+06) ≈</td><td rowspan=1 colspan=1>4.830e+06 (3.6e+06)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>1.087e+03 (1.7e+02) ≈</td><td rowspan=1 colspan=1>1.089e+03 (2.0e+02) ≈</td><td rowspan=1 colspan=1>1.020e+03 (6.5e+01) ≈</td><td rowspan=1 colspan=1>1.123e+03 (1.5e+02)</td></tr><tr><td rowspan=2 colspan=1>P8</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.751e+00 (1.6e-01) ≈</td><td rowspan=1 colspan=1>1.675e+00 (2.2e-01) ≈</td><td rowspan=1 colspan=1>1.721e+00 (3.4e-01) ≈</td><td rowspan=1 colspan=1>1.673e+00 (2.5e-01)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>3.174e+01 (3.1e+00) +</td><td rowspan=1 colspan=1>3.170e+01 (4.3e+00) ≈</td><td rowspan=1 colspan=1>3.444e+01 (4.8e+00) ≈</td><td rowspan=1 colspan=1>3.612e+01 (2.8e+00)</td></tr><tr><td rowspan=2 colspan=1>P9</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>1.063e+03 (7.2e+01) ≈</td><td rowspan=1 colspan=1>1.330e+03 (1.8e+02) ≈</td><td rowspan=1 colspan=1>1.268e+03 (2.0e+02) ≈</td><td rowspan=1 colspan=1>1.137e+03 (1.9e+02)</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4.597e+03 (7.8e+02) ≈</td><td rowspan=1 colspan=1>4.263e+03 (4.7e+02) ≈</td><td rowspan=1 colspan=1>4.524e+03 (6.1e+02) ≈</td><td rowspan=1 colspan=1>4.158e+03 (7.0e+02)</td></tr><tr><td rowspan=1 colspan=1>+/−/≈</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2/7/9</td><td rowspan=1 colspan=1>0/1/17</td><td rowspan=1 colspan=1>0/1/17</td><td rowspan=1 colspan=1>_</td></tr><tr><td rowspan=1 colspan=1>Average rank</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1>3.28</td><td rowspan=1 colspan=1>1.83</td><td rowspan=1 colspan=1>2.61</td><td rowspan=1 colspan=1>2.28</td></tr></table>

## V. CONCLUSION

This paper introduced In-Context Guidance Multitask Optimization (ICG-MTO) for improving inter-task coupling estimation under few-shot evaluation budgets. Through the ICG-MTBO instantiation, evaluated solutions were reorganized into directional in-context queries, a frozen numerical foundational model was used to infer predictive inter-task guidance, and the resulting guidance was translated into target-specific reference coupling structures for annealed MAP estimation of the MTGP inter-task coupling matrices. Experimental results on synthetic benchmarks and a real-world problem demonstrated that ICG-MTBO generally achieved better optimization performance than single-task and conventional multitask baselines. Its effectiveness across LCB and LogEI, together with the MFEA-II instantiation and sensitivity analysis, further demonstrated the generality and stability of the proposed in-context guidance framework.

Along this line of inquiry, extending in-context guidance to more complex multitask settings, including multiobjective multitask optimization [44], [45] and parametric multitask optimization [2], [45], represents a promising direction for investigating how numerical foundational models can guide inter-task coupling estimation beyond finite single-objective task collections.

## REFERENCES

[1] T. Wei, S. Wang, J. Zhong, D. Liu, and J. Zhang, “A review on evolutionary multitask optimization: Trends and challenges,” IEEE Trans. on Evol. Comput., vol. 26, no. 5, pp. 941–960, 2022.

[2] T. Wei, J. Liu, A. Gupta, P. Siew Tan, and Y.-S. Ong, “(θ ,θ<sub>u</sub>)-parametric multitask optimization: Joint search in solution and infinite task spaces,” IEEE Transactions on Evolutionary Computation, vol. 30, no. 3, pp. 1270–1283, 2026.

[3] A. Gupta, Y.-S. Ong, and L. Feng, “Multifactorial evolution: toward evolutionary multitasking,” IEEE Trans. on Evol. Comput., vol. 20, no. 3, pp. 343–357, 2016.

[4] K. K. Bali, Y.-S. Ong, A. Gupta, and P. S. Tan, “Multifactorial evolutionary algorithm with online transfer parameter estimation: MFEA-II,” IEEE Trans. on Evol. Comput., vol. 24, no. 1, pp. 69–83, 2019.

[5] J. Zhang, W. Zhou, X. Chen, W. Yao, and L. Cao, “Multisource selective transfer framework in multiobjective optimization problems,” IEEE Trans. on Evol. Comput., vol. 24, no. 3, pp. 424–438, 2020.

[6] Q. Shang, L. Zhang, L. Feng, Y. Hou, J. Zhong, A. Gupta, K. C. Tan, and H.-L. Liu, “A preliminary study of adaptive task selection in explicit evolutionary many-tasking,” in 2019 IEEE Congr. on Evol. Comput. (CEC), 2019, pp. 2153–2159.

[7] L. Zhou, L. Feng, K. C. Tan, J. Zhong, Z. Zhu, K. Liu, and C. Chen, “Toward adaptive knowledge transfer in multifactorial evolutionary computation,” IEEE Trans. on Cybern., vol. 51, no. 5, pp. 2563–2576, 2021.

[8] B. Da, A. Gupta, and Y.-S. Ong, “Curbing negative influences online for seamless transfer evolutionary optimization,” IEEE Trans. on Cybern., vol. 49, no. 12, pp. 4365–4378, 2018.

[9] K. Swersky, J. Snoek, and R. P. Adams, “Multi-task Bayesian optimization,” in Advances in Neural Information Processing Systems, C. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K. Weinberger, Eds., vol. 26. Curran Associates, Inc., 2013.

[10] M. Shakeri, E. Miahi, A. Gupta, and Y.-S. Ong, “Scalable transfer evolutionary optimization: Coping with big task instances,” IEEE Trans. on Cybern., vol. 53, no. 10, pp. 6160–6172, 2023.

[11] Y.-W. Wen and C.-K. Ting, “Parting ways and reallocating resources in evolutionary multitasking,” in 2017 IEEE Congr. on Evol. Comput. (CEC), 2017, pp. 2404–2411.

[12] N. Hollmann, S. Muller, L. Purucker, A. Krishnakumar, M. K¨ orfer,¨ S. B. Hoo, R. T. Schirrmeister, and F. Hutter, “Accurate predictions on small data with a tabular foundation model,” Nature, vol. 637, no. 8045, pp. 319–326, 2025. [Online]. Available: https: //doi.org/10.1038/s41586-024-08328-6

[13] J. Ma, V. Thomas, R. Hosseinzadeh, A. Labach, J. Cresswell, K. Golestan, G. Yu, A. L. Caterini, and M. Volkovs, “Tabdpt: Scaling tabular foundation models on real data,” in Advances in Neural Information Processing Systems, D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, Eds., vol. 38. Curran Associates, Inc., 2025, pp. 172 692–172 722. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2025/file/fc0e3f908a2116ba529ad0a1530a3675-Paper-Conference.pdf

[14] J. Qu, D. Holzmuller, G. Varoquaux, and M. Le Morvan, “TabICL:¨ A tabular foundation model for in-context learning on large data,” in Proceedings of the 42nd International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Singh, M. Fazel, D. Hsu, S. Lacoste-Julien, F. Berkenkamp, T. Maharaj, K. Wagstaff,

and J. Zhu, Eds., vol. 267. PMLR, 13–19 Jul 2025, pp. 50 817–50 847. [Online]. Available: https://proceedings.mlr.press/v267/qu25d.html

[15] E. V. Bonilla, K. Chai, and C. Williams, “Multi-task Gaussian process prediction,” in Advances in Neural Information Processing Systems, J. Platt, D. Koller, Y. Singer, and S. Roweis, Eds., vol. 20. Curran Associates, Inc., 2007.

[16] C. K. Williams and C. E. Rasmussen, Gaussian processes for machine learning. MIT press Cambridge, MA, 2006, vol. 2, no. 3.

[17] J. Gardner, G. Pleiss, K. Q. Weinberger, D. Bindel, and A. G. Wilson, “Gpytorch: Blackbox matrix-matrix gaussian process inference with gpu acceleration,” in Advances in Neural Information Processing Systems, S. Bengio, H. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, Eds., vol. 31. Curran Associates, Inc., 2018.

[18] T. Karvonen and C. J. Oates, “Maximum likelihood estimation in gaussian process regression is ill-posed,” Journal of Machine Learning Research, vol. 24, no. 120, pp. 1–47, 2023. [Online]. Available: http://jmlr.org/papers/v24/22-1153.html

[19] C. M. Bishop and N. M. Nasrabadi, Pattern recognition and machine learning. Springer, 2006, vol. 4, no. 4.

[20] T. Hastie, “The elements of statistical learning: data mining, inference, and prediction,” 2009.

[21] C. Rainey and K. McCaskey, “Estimating logit models with small samples,” Political Science Research and Methods, vol. 9, no. 3, p. 549–564, 2021.

[22] H. L. Van Trees and K. L. Bell, Detection estimation and modulation theory, part I: detection, estimation, and filtering theory. John Wiley & Sons, 2013.

[23] R. Bommasani, D. A. Hudson, E. Adeli, R. Altman, S. Arora, S. von Arx, M. S. Bernstein, J. Bohg, A. Bosselut, E. Brunskill et al., “On the opportunities and risks of foundation models,” arXiv preprint arXiv:2108.07258, 2021.

[24] S. Muller, N. Hollmann, S. P. Arango, J. Grabocka, and F. Hutter,¨ “Transformers can do bayesian inference,” in International Conference on Learning Representations, 2022. [Online]. Available: https: //openreview.net/forum?id=KSugKcbNf9

[25] Q. Dong, L. Li, D. Dai, C. Zheng, J. Ma, R. Li, H. Xia, J. Xu, Z. Wu, B. Chang et al., “A survey on in-context learning,” in Proceedings of the 2024 conference on empirical methods in natural language processing, 2024, pp. 1107–1128.

[26] X.-R. Zhang, Y.-J. Gong, Y.-T. Zhong, T. Huang, and J. Zhang, “Large language model as meta-surrogate for offline data-driven many-task optimization: A proof-of-principle study,” Information Sciences, vol. 726, p. 122762, 2026. [Online]. Available: https: //www.sciencedirect.com/science/article/pii/S0020025525008989

[27] D. Austin, A. Korikov, A. Toroghi, and S. Sanner, “Bayesian optimization with llm-based acquisition functions for natural language preference elicitation,” in Proceedings of the 18th ACM Conference on Recommender Systems, 2024, pp. 74–83.

[28] M. Gong, Z. Tang, H. Li, and J. Zhang, “Evolutionary multitasking with dynamic resource allocating strategy,” IEEE Trans. on Evol. Comput., vol. 23, no. 5, pp. 858–869, 2019.

[29] T. Wei and J. Zhong, “Towards generalized resource allocation on evolutionary multitasking for multi-objective optimization,” IEEE Computational Intelligence Magazine, vol. 16, no. 4, pp. 20–37, 2021.

[30] B. Neuenschwander, S. Weber, H. Schmidli, and A. O’Hagan, “Predictively consistent prior effective sample sizes,” Biometrics, vol. 76, no. 2, pp. 578–587, 2020.

[31] N. Srinivas, A. Krause, S. M. Kakade, and M. W. Seeger, “Informationtheoretic regret bounds for Gaussian process optimization in the bandit setting,” IEEE Trans. on Information Theory, vol. 58, no. 5, pp. 3250– 3265, 2012.

[32] P. I. Frazier, “A tutorial on Bayesian optimization,” 2018. [Online]. Available: https://arxiv.org/abs/1807.02811

[33] B. Shahriari, K. Swersky, Z. Wang, R. P. Adams, and N. de Freitas, “Taking the human out of the loop: A review of Bayesian optimization,” Proceedings of the IEEE, vol. 104, no. 1, pp. 148–175, 2016.

[34] X. Xue, Y. Hu, L. Feng, K. Zhang, L. Song, and K. C. Tan, “Surrogateassisted search with competitive knowledge transfer for expensive optimization,” IEEE Transactions on Evolutionary Computation, vol. 29, no. 6, pp. 2416–2430, 2025.

[35] Y. Lu, K. Zhang, X. Xue, L. Zhang, G. Chen, C. Cao, P. Liu, and K. C. Tan, “Multi-task surrogate-assisted search with bayesian competitive knowledge transfer for expensive optimization,” IEEE Transactions on Evolutionary Computation, pp. 1–1, 2025.

[36] S. Tan, Y. Wang, G. Sun, T. Pang, and K. Tang, “A surrogate-assisted evolutionary framework for expensive multitask optimization problems,” IEEE Trans. on Evol. Comput., 2024.

[37] S. Ament, S. Daulton, D. Eriksson, M. Balandat, and E. Bakshy, “Unexpected improvements to expected improvement for bayesian optimization,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 20 577–20 612. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/419f72cbd568ad62183f8132a3605a2a-Paper-Conference.pdf

[38] D. Zhan and H. Xing, “Expected improvement for expensive optimization: a review,” Journal of Global Optimization, vol. 78, no. 3, pp. 507–544, Nov 2020. [Online]. Available: https://doi.org/10.1007/ s10898-020-00923-x

[39] B. Da, Y.-S. Ong, L. Feng, A. K. Qin, A. Gupta, Z. Zhu, C.-K. Ting, K. Tang, and X. Yao, “Evolutionary multitasking for single-objective continuous optimization: Benchmark problems, performance metric, and baseline results,” 2017. [Online]. Available: https://arxiv.org/abs/1706.03470

[40] J.-B. Mouret and G. Maguire, “Quality diversity for multi-task optimization,” in Proceedings ofthe 2020 Genetic and Evol. Comput. Conference, ser. GECCO ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 121–129.

[41] W.-L. Loh, “On Latin hypercube sampling,” The Annals of Statistics, vol. 24, no. 5, pp. 2058 – 2080, 1996.

[42] L. Grinsztajn, K. Floge, O. Key, F. Birkel, P. Jund, B. Roof, B. J¨ ager,¨ D. Safaric, S. Alessi, A. Hayler et al., “Tabpfn-2.5: Advancing the state of the art in tabular foundation models,” arXiv preprint arXiv:2511.08667, 2025.

[43] M. R. Mes, W. B. Powell, and P. I. Frazier, “Hierarchical knowledge gradient for sequential sampling,” Journal of Machine Learning Research, vol. 12, pp. 2931–2974, Oct. 2011.

[44] T. Wei, J. Liu, A. Gupta, P. S. Tan, and Y.-S. Ong, “Bayesian forwardinverse transfer for multiobjective optimization,” in Parallel Problem Solving from Nature – PPSN XVIII. Cham: Springer Nature Switzerland, 2024, pp. 135–152.

[45] T. Wei, J. Liu, A. Gupta, C. C. Ooi, P. S. Tan, and Y.-S. Ong, “Amortized multi-objective optimization across tasks with generative solution modeling,” in Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26, D. Calvanese, Ed. International Joint Conferences on Artificial Intelligence Organization, 8 2026, pp. 6361–6369, main Track. [Online]. Available: https://doi.org/10.24963/ijcai.2026/708