# Gradient Boosting for Eficient Rank Discovery in Tensor Network Kernel Machines

Albert Saiapin

Delft Center for Systems and Control

a.saiapin@tudelft.nl

Delft University of Technology

Kim Batselier

Delft Center for Systems and Control Delft University of Technology

k.batselier@tudelft.nl

## Abstract

Tensor network kernel machines (TNKMs) combine nonlinear features with low-rank tensor representations to model complex relationships eficiently. However, selecting an appropriate tensor rank remains challenging, as conventional rank-search methods require training multiple candidate models, increasing computational costs. In this work, we focus on a canonical polyadic (CP) tensor network parameterization and propose GB-CPKM, a gradi ent boosting framework that builds a higher-capacity predictor by sequentially adding fixed rank learners. This approach provides a new perspective on rank selection by incrementally increasing model capacity, with ensemble size selected based on validation performance rather than through explicit rank search. Experiments on real-world regression benchmarks show that GB-CPKM trains up to 6ˆ faster and uses up to 2ˆ less peak memory than a conventional rank-selection method based on cross-validation, while achieving comparable predictive performance. These results highlight the potential of gradient boosting to improve the computational eficiency and scalability of tensor-network learning.

## 1 Introduction

Machine learning (ML) has become an essential tool in a wide range of scientific and industrial applications, from image classification (He et al., 2016) and natural language processing (Vaswani et al., 2017) to industrial control systems (Duan et al., 2016). However, achieving strong predictive performance requires more than fitting a model to training data. In addition to learning model parameters, practitioners must select appropriate hyperparameters (Yu & Zhu, 2020), which govern aspects of model architecture, complexity, training, and can substantially afect generalization performance and computational cost. Yet, hyperparameter selection often relies on grid search or other trial-and-error procedures (Franceschi et al., 2025) that require training and evaluating multiple candidate models.

Among the hyperparameters that influence machine learning performance, model complexity is particularly important in methods based on low-rank tensor representations. Tensor networks (TNs) represent highdimensional parameters using interconnected lower-dimensional tensors, potentially reducing storage and computational costs (Kolda & Bader, 2009; Cichocki, 2014). They have consequently been applied to neural network compression and other machine learning tasks (Cichocki et al., 2016; 2017). However, the benefits of these representations depend on the choice of tensor rank, which governs the trade-of between mode expressiveness and computational cost. Selecting an appropriate rank is therefore a key challenge in tensornetwork learning.

Tensor network kernel machines (TNKMs) (Wesel & Batselier, 2021; Batselier, 2022) combine nonlinear feature mappings with low-rank tensor representations to learn in high-dimensional feature spaces. In canonical polyadic tensor network kernel machines (CPKMs) (Saiapin & Batselier, 2026), the CP rank controls model capacity and influences computational and memory requirements. Conventional rank selection often involves training and evaluating models at multiple candidate ranks, which can be computationally expensive. To address this challenge, we propose GB-CPKM, a gradient boosting framework that incrementally builds an ensemble of fixed-rank CPKM base learners. By adding learners sequentially, GB-CPKM increases model capacity and allows the ensemble size to be selected using validation performance, providing an alternative to explicit rank search. The main contributions of this work are summarized as follows:

• We introduce GB-CPKM, a gradient boosting framework that combines CPKM base learners to construct an expressive ensemble, enabling incremental model growth and efective rank discovery.

• We systematically analyze the computational and memory requirements of the proposed approach and investigate how base-learner rank and the number of boosting stages afect trade-ofs between model complexity, predictive performance, and computational eficiency.

• We evaluate GB-CPKM on regression benchmarks of varying sizes, comparing its performance, training time, and peak memory consumption against a cross-validation-based rank-selection baseline. The results show that GB-CPKM trains up to 6ˆ faster and uses up to 2ˆ less peak memory on average, while achieving comparable predictive performance.

• We introduce GB-CPKM(H), a hybrid training strategy that combines the eficient rank discovery of gradient boosting with an additional training step over the full ensemble model. GB-CPKM(H) improves predictive accuracy while preserving the eficiency of boosting, making it a promising default choice for CP-based kernel models when both accuracy and computational cost matter.

## 2 Tensor Network Kernel Machines

## 2.1 Notation

We first establish the notation used throughout the paper. Calligraphic uppercase letters, such as $\mathcal { Z } .$ , denote sets. Scalars are denoted by italic letters (e.g., w and $W )$ , vectors by lowercase bold letters (e.g., w), matrices by uppercase bold letters (e.g., W), and tensors by uppercase bold italic letters (e.g., W). Indexed vectors are denoted by ${ \pmb w } _ { r } ^ { ( d ) }$ , where the indices and superscripts distinguish the corresponding vectors.

For a vector $w \in \mathbb { R } ^ { I }$ , its ith entry is denoted by $w _ { i } .$ . Similarly, the $( i _ { 1 } , i _ { 2 } , \dots , i _ { D } ) \dag$ th entry of a Dth-order tensor $\pmb { \mathcal { W } } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times \dots \times I _ { D } }$ is denoted by $w _ { i _ { 1 } i _ { 2 } \cdots i _ { D } }$ . Unless stated otherwise, we use zero-based indexing for tensor entries. We use $\otimes$ to denote the Kronecker product (Cichocki et al., 2016). The transpose of a vector w is denoted by $\pmb { w } ^ { \top }$ . For convenience, we represent tensors as vectors using the following vectorization convention.

Definition 2.1 (Vectorization). The vectorization operator

$$
\mathrm { v e c } \left( \cdot \right) : \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times \dots \times I _ { D } } \longrightarrow \mathbb { R } ^ { I _ { 1 } I _ { 2 } \dots I _ { D } }
$$

maps a Dth-order tensor W to a vector whose entries are the tensor entries arranged in a fixed order. Specifically,

$$
\mathrm { v e c } \left( \pmb { \mathcal { W } } \right) _ { i } : = w _ { i _ { 1 } i _ { 2 } \cdots i _ { D } } ,
$$

where

$$
i : = ( i _ { 1 } i _ { 2 } \cdot \cdot \cdot i _ { D } ) = i _ { 1 } + \sum _ { d = 2 } ^ { D } i _ { d } \prod _ { j = 1 } ^ { d - 1 } I _ { j } .
$$

Thus, the first tensor index varies fastest in the vectorized representation.

## 2.2 Tensor Networks

Tensor networks (TNs) provide a framework for eficiently representing and manipulating high-dimensional tensors by expressing them as networks of smaller tensors connected through shared indices (Cichocki, 2014). This representation can substantially reduce storage requirements when the underlying tensor admits a lowrank decomposition. In this work, we focus on the canonical polyadic decomposition.

Definition 2.2 (Canonical polyadic decomposition (CPD) (Kolda & Bader, 2009)). A Dth-order tensor $\pmb { \mathcal { W } } \in \mathbb { R } ^ { I _ { 1 } \times I _ { 2 } \times \cdots \times I _ { D } }$ admits a rank-R CP decomposition if it can be expressed as

$$
\operatorname { v e c } \left( { \pmb { \mathscr W } } \right) : = \sum _ { r = 1 } ^ { R } { \pmb v } _ { r } ^ { ( D ) } \otimes \cdots \otimes { \pmb v } _ { r } ^ { ( 1 ) } ,\tag{1}
$$

where ${ \pmb v } _ { r } ^ { ( d ) } \in \mathbb { R } ^ { I _ { d } }$ and

$$
V ^ { ( d ) } : = \left[ \pmb { v } _ { 1 } ^ { ( d ) } , \ldots , \pmb { v } _ { R } ^ { ( d ) } \right] \in \mathbb { R } ^ { I _ { d } \times R }
$$

is the factor matrix (core) associated with mode d.

For a tensor with equal mode sizes, $I _ { d } = I ,$ the original dense representation requires $\mathcal { O } ( I ^ { D } )$ storage, whereas its CP representation requires only $\mathcal { O } ( D I R )$ storage. Thus, when R is suficiently small relative to the mode sizes and tensor order, CPD can substantially reduce memory requirements. Another practical advantage is that CPD is characterized by a single rank parameter, $R ,$ which simplifies model selection. By contrast, tensor-train (TT) (Oseledets, 2011) and hierarchical Tucker (HT) (Grasedyck, 2010) decompositions involve multiple rank parameters, making their rank selection more involved. CPD can also ofer more favorable storage scaling than the Tucker decomposition (Kolda & Bader, 2009). For equal mode sizes I and a common Tucker rank R, the Tucker representation requires $\mathcal { O } ( D I R + R ^ { D } )$ storage, owing to its factor matrices and core tensor.

Despite its favorable storage properties, the efectiveness of CPD depends critically on the choice of rank R. A rank that is too low may limit model expressiveness, whereas an excessively high rank increases computational and memory costs and may lead to overfitting. Selecting an appropriate rank is therefore essential for balancing predictive accuracy and model complexity. However, conventional approaches typically evaluate multiple candidate ranks (Kolda & Bader, 2009), requiring repeated model training and increasing computational cost. This motivates the development of more eficient rank-selection strategies.

## 2.3 Model Formulation and Optimization

Consider the following linear model:

$$
f ( \pmb { x } ) = \phi ( \pmb { x } ) ^ { \top } \pmb { w } ,
$$

where $\pmb { \phi } ( \pmb { x } ) \in \mathbb { R } ^ { I _ { 1 } I _ { 2 } \cdots I _ { D } }$ is a nonlinear feature map and $\pmb { w } \in \mathbb { R } ^ { I _ { 1 } I _ { 2 } \cdots I _ { D } }$ is a weight vector learned from the dataset $\overset { \cdot } { \mathcal { D } } = \left\{ \pmb { X } \in \mathbb { R } ^ { N \times D } , \pmb { y } \in \mathbb { R } ^ { N } \right\}$

Tensor network kernel machines (TNKMs) extend this model by representing the weight vector as a tensor network and assuming that the feature map factorizes across the input dimensions:

$$
\phi ( \pmb { x } ) : = \phi ^ { ( D ) } ( x _ { D } ) \otimes \cdot \cdot \cdot \otimes \phi ^ { ( 1 ) } ( x _ { 1 } ) ,
$$

where $\phi ^ { ( d ) } : \mathbb { R }  \mathbb { R } ^ { I _ { d } }$ maps the dth component of $\pmb { x } \in \mathbb { R } ^ { D }$ to a feature vector. This tensor-product construction is related to product kernels (Hensman et al., 2018; Solin & Särkkä, 2019). Although the resulting feature space has dimension $\Pi _ { d = 1 } ^ { D } I _ { d } ,$ which grows exponentially with the input dimension when the mode sizes are comparable, the tensor network representation of w enables model responses and gradients to be computed without explicitly constructing or storing the full weight vector. For the CP decomposition (1), this computational advantage is formalized by the following result.

Theorem 2.1 (CP kernel machine (Wesel & Batselier, 2024)). Suppose that w admits a rank-R CP decomposition. Then, for the model

$$
f ( \pmb { x } ) = \left[ \phi ^ { ( D ) } ( x _ { D } ) \otimes \cdots \otimes \phi ^ { ( 1 ) } ( x _ { 1 } ) \right] ^ { \top } \pmb { w } ,
$$

its response and gradients can be computed in OpDIRq time, rather than $\textstyle { \mathcal { O } } \left( \prod _ { d = 1 } ^ { D } I _ { d } \right)$ , where $I = \operatorname* { m a x } _ { d } I _ { d }$

Training a CPKM involves solving the following nonlinear, nonconvex optimization problem:

$$
\begin{array} { r l } { \underset { \pmb { v } } { \mathrm { m i n } } } & { \frac { 1 } { 2 } \left\| \pmb { y } - \pmb { \Phi } \pmb { w } ( \pmb { v } ) \right\| _ { 2 } ^ { 2 } + \frac { \gamma } { 2 } \left\| \pmb { v } \right\| _ { 2 } ^ { 2 } , } \\ { \mathrm { s u b j e c t ~ t o } } & { \pmb { w } ( \pmb { v } ) = \displaystyle \sum _ { r = 1 } ^ { R } \pmb { v } _ { r } ^ { ( D ) } \otimes \cdots \cdot \pmb { \cdot } \otimes \pmb { v } _ { r } ^ { ( 1 ) } , } \end{array}\tag{2}
$$

where y contains the target values, Φ is the feature matrix with entries $\Phi _ { n i } : = \phi _ { i } ( { \pmb x } _ { n } )$ , and ${ \pmb x } _ { n }$ is the nth input sample from the data matrix $\pmb { X } \in \mathbb { R } ^ { N \times D }$ . The vector $\pmb { v } \in \mathbb { R } ^ { R \sum _ { d } ^ { D } I _ { d } }$ collects all entries of the cores $V ^ { ( d ) }$ whose columns are denoted by ${ \pmb v } _ { r } ^ { ( d ) }$ . The parameter γ controls the strength of the $\ell _ { 2 }$ regularization.

In this work, we focus on alternating least squares (ALS) (Uschmajew, 2012), which optimizes the tensor factors $V ^ { ( d ) }$ iteratively, for $d = 1 , \dots , D$ , while keeping the remaining factors fixed, providing a natural approach for exploiting the structure of CP decompositions.

## 3 Gradient Boosting Tensor Network Kernel Machines

Boosting is an ensemble learning technique that constructs a predictive model by combining multiple base learners into a stronger model. We construct the ensemble as an additive model (Hastie et al., 2009) of the form

$$
f _ { M } ( \pmb { x } ) = \sum _ { m = 1 } ^ { M } \beta _ { m } h _ { m } ( \pmb { x } ) ,\tag{3}
$$

where $h _ { m }$ is the mth base learner and $\beta _ { m }$ is its additive weight. Rather than training all learners independently, boosting adds them sequentially, with each new learner aiming to correct the errors of the current ensemble (Schapire & Freund, 2012). This sequential construction provides a flexible way to increase mode capacity while controlling the complexity of individual learners. In the following, we introduce the main principles of gradient boosting and describe how they can be applied to tensor network kernel machines (see Theorem 2.1).

## 3.1 Gradient Boosting

Gradient boosting (GB) provides a general framework for constructing additive models (3), using arbitrary diferentiable loss functions (Friedman, 2001). Given training data $\mathcal { D } \ = \ \{ ( \boldsymbol { x } _ { i } , y _ { i } ) | \boldsymbol { x } _ { i } \in \mathbb { R } ^ { D } , y _ { i } \in \mathbb { R } , i \ =$ $1 , \ldots , N \}$ , the objective is to minimize the empirical risk $\begin{array} { r } { \mathcal { L } ( f ) = \sum _ { i = 1 } ^ { N } l \big ( y _ { i } , f ( \pmb { x } _ { i } ) \big ) } \end{array}$ , where $l ( y _ { i } , f ( { \pmb x } _ { i } ) )$ measures the prediction error for sample i.

Starting from an initial model $f _ { 0 } ,$ , gradient boosting builds an additive model by sequentially introducing new learners. At iteration $m ,$ it computes the gradient of the loss with respect to the current predictions:

$$
g _ { i m } = \left. \frac { \partial l ( y _ { i } , \hat { y } ) } { \partial \hat { y } } \right| _ { \hat { y } = f _ { m - 1 } ( \pmb { x } _ { i } ) } , \qquad i = 1 , \ldots , N .
$$

The negative gradient, $- g _ { i m }$ , indicates the direction in which the prediction for sample i should move to decrease the loss locally. In function space, a gradient descent step would update the predictions on the training data as

$$
f _ { m } ( { \pmb x } _ { i } ) = f _ { m - 1 } ( { \pmb x } _ { i } ) - \beta _ { m } g _ { i m } ,
$$

where $\beta _ { m }$ is the step size.

However, this update specifies values only at the training samples and does not directly define a predictive function for unseen inputs. Gradient boosting addresses this by fitting a new learner $h _ { m }$ to approximate the negative gradient from the input features:

$$
h _ { m } = \arg \operatorname* { m i n } _ { h \in \mathcal { H } } \sum _ { i = 1 } ^ { N } \left[ - g _ { i m } - h ( \mathbf { x } _ { i } ) \right] ^ { 2 } ,
$$

![](images/54148e641f16ccc0ed8789527cf52fdf9753186235f824399a6abbd8f940f7c5.jpg)  
Figure 1: Comparison of the training procedures for CP tensor network kernel machines (CPKMs) using gradient boosting (GB) and rank selection via validation (CV). (a) GB-CPKM incrementally builds an ensemble by training fixed-rank R<sup>ˆ</sup> base learners and concatenating their factor matrices along the rank dimension to represent the combined model. The ensemble size M is selected based on validation performance. (b) In contrast, CV-CPKM independently trains candidate models with diferent CP ranks and selects the rank that achieves the lowest validation error. In the diagrams, each base learner model is represented by D factor matrices of size $I \times { \hat { R } }$ , where I is the number of rows and R<sup>ˆ</sup> is the CP rank. Diferent colors distinguish the trainable factor matrices. Factors in gray are fixed and are not updated.

where H denotes the class of candidate base learners. The resulting learner approximates the desired descent direction and can therefore be evaluated at both training and unseen inputs. The ensemble is then updated according to

$$
f _ { m } ( { \pmb x } ) = f _ { m - 1 } ( { \pmb x } ) + \beta _ { m } h _ { m } ( { \pmb x } ) .\tag{4}
$$

The step size $\beta _ { m }$ may be fixed or selected through a line search that minimizes the loss along the direction given by $h _ { m }$ (Bühlmann & Hothorn, 2007). This procedure is repeated for $m = 1 , \ldots , M ,$ where M is the number of boosting stages. The number of stages and other hyperparameters can be selected using validation data or early stopping.

## 3.2 CP Tensor Network Kernel Machine as a Base Learner

In general, base learners can take many forms, including decision trees (Friedman & Meulman, 2003) and neural networks (Badirli et al., 2020). In our approach, each base learner is represented as a CP kerne machine (CPKM) from Theorem 2.1:

$$
h _ { m } ( \pmb { x } ) = \phi ( \pmb { x } ) ^ { \top } \pmb { w } ( \pmb { v } _ { m } ) ,\tag{5}
$$

where ${ \pmb w } ( { \pmb v } _ { m } )$ is represented by a low-rank CPD parameterized by ${ \pmb v } _ { m }$ . For a rank- $\cdot \hat { R }$ CP decomposition, the parameter vector ${ \pmb v } _ { m }$ has $\textstyle { \hat { R } } \sum _ { d = 1 } ^ { D } I _ { d }$ entries.

Substituting (5) into the update rule (4) and expanding the recursion yields the final ensemble model:

$$
f _ { M } ( \pmb { x } ) = \phi ( \pmb { x } ) ^ { \top } \pmb { w } ( \hat { \pmb { v } } ) = \phi ( \pmb { x } ) ^ { \top } \left( \sum _ { m = 1 } ^ { M } \beta _ { m } \pmb { w } ( \pmb { v } _ { m } ) \right) .\tag{6}
$$

A key property of CP representations is that sums of CP tensors can be represented by concatenating their factor matrices $V ^ { ( d ) }$ , with the corresponding weights absorbed into the factors. Consequently, when each base learner has rank $\hat { R } _ { { \bf i } }$ , the ensemble in (6) admits a representation with rank at most MR<sup>ˆ</sup>:

$$
{ \pmb w } ( \hat { \pmb v } ) = \sum _ { m = 1 } ^ { M } \beta _ { m } { \pmb w } ( { \pmb v } _ { m } ) = \sum _ { ( r m ) = 1 } ^ { M \hat { R } } ( \beta _ { m } { \pmb v } _ { m } ) _ { r } ^ { ( D ) } \otimes \cdot \cdot \cdot \otimes ( { \pmb v } _ { m } ) _ { r } ^ { ( 1 ) } .
$$

Algorithm 1: Gradient Boosting CPKM with Squared-Error Loss   
Parameters: Number of boosting stages M, base-learner CP rank ${ \hat { R } } ,$ regularization parameter $\gamma _ { : }$   
feature map $\phi ( \cdot )$   
Data: Training inputs $\hat { \pmb X _ { \lambda } } \in \overset { \cdot } { \mathbb { R } ^ { N \times D } }$ and targets $\boldsymbol { y } \in \mathbb { R } ^ { N }$   
1 $f _ { 0 } \gets \mathbf { 0 }$   
2 for $m = 1 , \ldots , M$ do   
$^ { \prime * } \ 1$ Compute residuals of the current ensemble \*/   
3 $r _ { m } \gets y - f _ { m - 1 }$   
$^ { / * 2 }$ . Fit a regularized CPKM base learner to the residuals, using (2) \*/   
4 ${ \pmb v } _ { m } \gets \arg \operatorname* { m i n } _ { { \pmb v } } \left[ \frac { 1 } { 2 } \left\| { \pmb r } _ { m } - \Phi { \pmb w } ( { \pmb v } ) \right\| _ { 2 } ^ { 2 } + \frac { \gamma } { 2 } \left\| { \pmb v } \right\| _ { 2 } ^ { 2 } \right]$   
$^ { \prime * }$ Compute the fitted base learner’s predictions \*/   
5 $h _ { m } \gets \Phi w ( v _ { m } )$   
$\times \ 3$ . Compute the least-squares optimal step size \*/   
6 $\beta _ { m } \gets \frac { r _ { m } ^ { \top } h _ { m } } { \cdot \top \cdot }$   
$\pmb { h } _ { m } ^ { \top } \pmb { h } _ { m }$   
$/ * 4 .$ Update the ensemble predictions \*/   
7 $\pmb { f } _ { m } \gets \pmb { f } _ { m - 1 } + \beta _ { m } \pmb { h } _ { m }$   
Output: Base learner parameters $\{ \pmb { v } _ { m } \} _ { m = 1 } ^ { M }$ and step sizes $\lbrace \beta _ { m } \rbrace _ { m = 1 } ^ { M }$

For rank-one base learners, each boosting stage adds at most one CP component. Therefore, a rank-R CPKM can be interpreted as a committee of R rank-one learners. This perspective provides a way to adapt model complexity during training: each boosting stage adds new components, and the number of stages determines an upper bound on the ensemble’s CP rank. By monitoring validation performance and stopping when further boosting no longer improves generalization, the ensemble size - and hence its efective rank - can be selected automatically alongside the base-learner parameters. This provides a practical alternative to explicitly searching over candidate ranks. The main diferences between the two training strategies are illustrated in Figure 1.

The gradient boosting procedure for the squared-error loss is summarized in Algorithm 1. At each stage, a CPKM is fitted to the residuals of the current ensemble, and its contribution is determined by a least-squares optimal step size $\beta _ { m }$ . The ensemble predictions are then updated by adding the new base learner.

Computational considerations. Let E denote the number of ALS sweeps per base learner, N is the number of training samples, D is the input dimensionality, and I is the number of basis functions per dimension. We assume that each base learner has a fixed CP rank R<sup>ˆ</sup>.

Memory complexity. When $N \ \gg \ D _ { ; }$ , the peak memory required to train a CPKM of CP rank R using ALS is OpNIRq. In gradient boosting, each base learner has a fixed CP rank R<sup>ˆ</sup> and is trained sequentially. Therefore, the peak memory required for base-learner training remains $\mathcal { O } ( N I \hat { R } )$ , independently of the number of boosting stages M, assuming that intermediate training memory can be released between stages. In contrast, training a single CPKM with CP rank MR<sup>ˆ</sup> requires $\mathcal { O } ( \dot { N } I M \hat { R } )$ peak memory under the same complexity model. Although the memory required to train each base learner remains bounded by its fixed rank, the memory required to store the accumulated ensemble grows with M.

Training complexity. Under the assumed ALS implementation, the computational cost of training a rank-R CPKM is $\mathcal { O } ( E N D I ^ { 2 } R ^ { 2 } )$ . Thus, training M rank-one base learners requires a cumulative cost of $\mathcal { O } ( E N D I ^ { 2 } M )$ , assuming a comparable number of ALS sweeps per learner. By comparison, training a single rank-M CPKM has complexity $\mathcal { O } ( E N D I ^ { 2 } M ^ { 2 } )$ under the same complexity model. This suggests a potential computational advantage for gradient boosting, although the realized speedup depends on implementation details and optimization behavior.

Overall, GB-CPKM provides a flexible way to increase model capacity incrementally using fixed-rank learners, potentially reducing training costs while enabling validation-based selection of the ensemble size. However, its sequential training procedure limits parallelism across boosting stages, and validation data are needed to select the stopping point and tune hyperparameters, such as the regularization strength.

## 4 Experiments

All experiments were conducted on a Dell Latitude 7440 laptop equipped with a 13th-generation Intel Core i7-1365U processor and 16 GB of RAM. We implement the proposed GB-CPKM Algorithm 1 from scratch, using the existing TNKM library (Saiapin & Batselier, 2026) for the underlying tensor network operations and model optimization. The Python source code, datasets, and experimental configurations required to reproduce the experiments are available on GitHub.<sup>1</sup>

## 4.1 Experimental Setup

Datasets and Preprocessing. We evaluate the proposed GB-CPKM method on seven real-world regression datasets with varying characteristics and scales. Four datasets—Energy, Concrete, Naval, and HouseElectric—are obtained from the UCI Machine Learning Repository (Dua & Graf, 2017). We addi tionally consider Kin8nm (Ghahramani, 1996), Precipitation (Dong et al., 2017), and Airline (Hensman et al., 2013).

For each dataset, the input features are scaled to the unit hypercube, and the target values are standardized. Each dataset is divided into training, validation, and test sets using a 70%/10%/20% split, respectively. We use ten fixed random seeds to generate ten data splits. For each split, all the compared methods use identical training, validation, and test sets, enabling a paired comparison of predictive performance.

Model Configuration. The primary objective is to evaluate the proposed GB-CPKM method against a cross-validation-based baseline (CV-CPKM), with particular emphasis on a CP rank selection and computational eficiency. Both methods use polynomial features (Wesel & Batselier, 2024). For simplicity, the number of basis functions is held constant across input dimensions, such that $I _ { d } = I$ for all $d = 1 , \ldots , D _ { }$ The basis size I is selected separately for each dataset to provide suficient model capacity while keeping memory requirements within the available hardware constraints. For a given dataset, the same basis size is used for both methods to ensure a consistent representation. We use the alternating least squares (ALS) algorithm to train the tensor network models and individual gradient-boosting base learners for five epochs. Regularization is applied to the CP cores. The regularization coeficient γ is selected from the candidate set

$$
\gamma \in \{ 1 0 ^ { - 6 } , 1 0 ^ { - 5 } , 1 0 ^ { - 4 } , 1 0 ^ { - 3 } , 1 0 ^ { - 2 } \} .
$$

Both GB and CV tune the regularization coeficient using the validation set.

Evaluation Metrics and Computational Cost. We evaluate both methods on the test set for each of the ten data splits and report the mean and standard deviation of the resulting mean squared error (MSE). We additionally measure training time in seconds and peak memory consumption in mebibytes (MiB).

## 4.2 Efect of Base-Learner Rank

The first set of experiments investigates how the rank of individual base learners and the number of boosting stages afect the predictive performance and computational eficiency of GB-CPKM. Our aim is to understand how a fixed model capacity should be distributed between the complexity of individual learners and the number of learners in the ensemble. To this end, we consider base-learner ranks $R \in \{ 1 , 2 , 4 , 8 , 1 2 \}$ and fix the total nominal rank budget at $R _ { \mathrm { t o t a l } } ~ = ~ 2 4$ Accordingly, the number of boosting stages is set to $N _ { e s t } \ = \ 2 4 / R ,$ yielding $N _ { e s t } \in \{ 2 4 , 1 2 , 6 , 3 , 2 \}$ , respectively. This setup allows us to compare ensembles comprising many low-rank learners with ensembles comprising fewer higher-rank learners, while keeping the total nominal rank budget constant.

![](images/1a1753895bf0ecb599bb15e70c9422956fb6ec619e85265b9e60ffdddb4f4288.jpg)  
Figure 2: Ablation study of base-learner rank in GB-CPKM on four real-world datasets, with each column corresponding to a dataset. The top row shows cumulative training time (in seconds), while the bottom row reports validation MSE on a logarithmic scale. Across the evaluated datasets, higher-rank base learners achieve better predictive performance while also reducing cumulative training time. However, rank selection should also account for memory requirements, which may constrain the use of higher-rank models in practice.

Figure 2 presents the cumulative training time and validation MSE for the four datasets. The results indicate that increasing the base-learner rank generally improves predictive performance. Higher-rank configurations also tend to reduce cumulative training time, particularly when comparing base ranks 4, 8, and 12, although their relative training times vary across datasets. These findings suggest that allocating a fixed rank budget to fewer, higher-rank learners can be more efective than distributing it across a larger number of lower-rank learners, improving predictive accuracy while potentially reducing computational cost.

## 4.3 Gradient Boosting for Eficient Rank Search

In this section, we compare GB-CPKM with a cross-validation (CV) baseline for selecting CP ranks in a sequential learning setting. The goal is to investigate whether gradient boosting with fixed-rank base learners can provide an efective alternative to explicitly searching over candidate ranks.

GB-CPKM uses base learners with a fixed CP rank of 2. We use ten boosting stages for all datasets except Airline, for which we use five. In contrast, the CV-CPKM baseline evaluates a predefined set of candidate ranks. For all datasets except Airline, the candidate set is $R \in \{ 2 , 4 , 6 , \ldots , 2 0 \}$ , whereas for Airline, we use $R \in \{ 2 , 4 , 6 , 8 , 1 0 \}$ . For each candidate rank, the model is trained on the training set and evaluated on the validation set. The rank achieving the lowest validation MSE is selected. The test set is reserved exclusively for final evaluation and is not used for hyperparameter selection.

## 4.3.1 Dynamical Behavior on Small-Scale Problems

In this experiment, we investigate how increasing the number of CP rank options afect the test dynamics of CPKM under the two learning strategies, GB and CV, on small-scale problems. The goal is to characterize the trade-ofs between sequentially adding fixed-rank base learners and explicitly evaluating candidate ranks during hyperparameter selection.

Figure 3 shows how test MSE, cumulative training time, and peak memory consumption change as the number of candidate rank options increases. Across all datasets, GB-CPKM exhibits a slower increase in cumulative training time than cross-validation baseline. This advantage stems from its sequential training strategy, which reuses previously trained base learners, whereas CV-CPKM trains a separate model from scratch for each candidate rank. The same diference in training strategies also helps explain the lower peak memory consumption of GB-CPKM. Since it uses fixed-rank base learners, the memory required to train each learner and the whole model is primarily determined by the base rank. In contrast, CV-CPKM evaluates candidate models of increasing rank, requiring progressively more memory as model complexity grows. Consequently, its peak memory consumption is influenced by the largest candidate rank considered, which can lead to substantially higher memory requirements.

![](images/ec23b1f321fb6aebb78b7e2ddc9f2b3104b322dc5ac68e19faf05a79bcf05585.jpg)  
Figure 3: Comparison of GB-CPKM (red) and CV-CPKM (blue) on four real-world datasets, with each column corresponding to a dataset. From top to bottom, the rows show cumulative training time (in seconds), test MSE, and peak memory consumption (in MiB), plotted against the number of considered candidate CP ranks. Solid lines indicate the mean across 10 runs, and shaded regions represent ˘ 1 standard deviation. GB-CPKM consistently requires less training time and peak memory than CV-CPKM, while achieving comparable predictive performance. The test MSE of GB-CPKM is similar to that of CV-CPKM, with overlapping variability bands in some cases, and approaches comparable values across the evaluated datasets. These results highlight the potential of GB-CPKM to reduce computational costs without substantially compromising predictive accuracy.

Speaking of predictive performance, GB-CPKM achieves test MSE values comparable to those of CV-CPKM on most datasets. The mean test errors are similar, and their standard-deviation bands overlap in several cases, suggesting that GB-CPKM can achieve competitive predictive accuracy at a lower computational cost. However, these observations alone do not establish statistical equivalence between the methods.

## 4.3.2 Large-Scale Problems

In this experiment, we further evaluate the gradient boosting model on three large-scale datasets to investigate its scalability, while also including results for smaller datasets to provide a broader view of its performance across diferent problem sizes.

In Table 1 we present the comparison of GB-CPKM with the cross-validation baseline (CV-CPKM) with a focus on the trade-ofs between predictive performance, training time (rank search), and peak memory consumption. The results show that GB model ofers clear computational advantages over cross-validation, training up to 6ˆ faster and requiring up to 2ˆ less peak memory. In terms of predictive performance, the two methods achieve broadly comparable test MSE, although GB-CPKM yields somewhat higher errors on some datasets, for example Naval.

We also introduce a hybrid variant, GB-CPKM(H), which follows the same training procedure as GB-CPKM but includes one additional full ALS sweep over the combined model before test evaluation. This additional optimization considerably improves predictive performance on most datasets. However, optimizing the full model increases peak memory consumption to levels comparable to those of CV-CPKM, reducing the memory-eficiency advantage of the original model.

Table 1: Performance comparison of GB-CPKM and CV-CPKM across regression benchmarks. We report test mean squared error (MSE), training time (seconds), and peak memory consumption (MiB). Each experiment is repeated over 10 runs, with results reported as mean ˘ 1 standard deviation. N denotes the number of samples, and D denotes the input dimensionality. Overall, GB-CPKM trains up to 6ˆ faster and uses up to 2ˆ less peak memory, on average, than CV-CPKM, while achieving comparable test MSE. The hybrid variant, GB-CPKM(H), further improves predictive performance on most datasets by performing an additional full ALS sweep on the combined model.
<table><tr><td></td><td>I</td><td></td><td colspan="3">MSE (test) ↓</td><td colspan="2">Time (sec.) ↓</td><td colspan="2">Memory (MiB)↓</td></tr><tr><td>Data</td><td>N</td><td>D</td><td>CV-CPKM</td><td>GB-CPKM</td><td>GB-CPKM(H)</td><td>CV-CPKM</td><td>GB-CPKM</td><td>CV-CPKM</td><td>GB-CPKM</td></tr><tr><td>Energy</td><td>768</td><td>8</td><td>0.0026±0.00</td><td>0.0043±0.00</td><td>0.0038±0.00</td><td>0.14±0.02</td><td>0.02±0.01</td><td>534</td><td>444</td></tr><tr><td>Concrete</td><td>1030</td><td>8</td><td>0.1618±0.05</td><td>0.1889±0.04</td><td>0.1801±0.05</td><td>0.18±0.00</td><td>0.06±0.01</td><td>572</td><td>461</td></tr><tr><td>Kin8nm</td><td>8192</td><td>8</td><td>0.1077±0.01</td><td>0.1721±0.03</td><td>0.1209±0.01</td><td>1.28±0.03</td><td>0.33±0.02</td><td>865</td><td>551</td></tr><tr><td>Naval</td><td>11934</td><td>16</td><td>0.0000±0.00</td><td>0.0074±0.00</td><td>0.0010±0.00</td><td>1.78±0.06</td><td>0.36±0.02</td><td>1090</td><td>630</td></tr><tr><td>Precipitation</td><td>628474</td><td></td><td>0.9419±0.04</td><td>0.9454±0.04</td><td>0.9448±0.04</td><td>88.59±11.84</td><td>8.35±1.05</td><td>2885</td><td>1156</td></tr><tr><td>HouseElectric</td><td>2049280</td><td>36</td><td>0.0010±0.00</td><td>0.0018±0.00</td><td>0.0010±0.00</td><td>269.60±12.67</td><td>25.21±0.58</td><td>6058</td><td>2214</td></tr><tr><td>Airline</td><td>5929413</td><td>8</td><td>0.8272±0.06</td><td>0.8607±0.04</td><td>0.8395±0.03</td><td>196.86±7.65</td><td>45.01±3.07</td><td>9802</td><td>4023</td></tr></table>

In summary, these results suggest that gradient boosting provides a more computationally eficient alternative to explicit rank search, substantially reducing training time and memory requirements while maintaining competitive predictive performance on the evaluated large-scale problems.

## 5 Related Work

Selecting an appropriate tensor rank remains a challenging problem in tensor decomposition and tensornetwork learning. A common practice is to treat rank as a hyperparameter and select it using grid search or cross-validation over a predefined set of candidate values (Sidiropoulos et al., 2017; Novikov et al., 2017; Wesel & Batselier, 2021; Haliassos et al., 2021; Menzen et al., 2026). Although conceptually simple, these approaches require training and evaluating multiple models, potentially increasing computational and memory costs.

Bayesian approaches. Several studies address rank determination through probabilistic modeling and sparsity-inducing priors. Zhao et al. (2015) introduce probabilistic Tucker models that infer multilinear rank using structured sparsity and model evidence. Camarrone & Van Hulle (2016) propose Bayesian Nway partial least squares with automatic relevance determination (ARD) priors on CP factors to infer the number of relevant components. More recently, Kilic & Batselier (2025) develop a probabilistic framework for tensor network kernel machines that uses hierarchical sparsity-inducing priors to infer CP rank and feature dimensions. These methods ofer principled mechanisms for automatic complexity determination, but require probabilistic formulations and associated inference procedures.

Adaptive tensor-network optimization. Rank adaptation has also been studied in tensor-network optimization, particularly in the tensor-train (TT), or matrix product state (MPS), format. Holtz et al. (2012) develop ALS- and MALS-based optimization methods that allow TT ranks to adapt during optimization. Related strategies appear in density matrix renormalization group (DMRG) algorithms (Schollwöck, 2011). Similarly, Dolgov & Savostyanov (2014) propose methods for solving high-dimensional symmetric positive-definite linear systems using low-rank tensor representations and sequential optimization of tensor components. While these methods demonstrate the potential of adaptive rank optimization, they primarily focus on the tensor-train (TT) format, whereas our work investigates rank allocation in canonical polyadic (CP) tensor-network models.

Reinforcement learning. Zhang et al. (2024) formulate CP rank selection as a decision-making problem using the Deep Deterministic Policy Gradient (DDPG) algorithm. Their method combines a pretrained model of tensor information with an SVD-based exploration strategy to select ranks for convolutional-layer compression. This provides a learned alternative to conventional rank search, but is designed for a specific application and requires a reinforcement-learning framework.

Positioning of our work. Rather than introducing a separate probabilistic inference procedure, a reinforcement-learning agent, or an adaptive-rank optimization algorithm, we investigate gradient boosting with fixed-rank CP tensor-network base learners. GB-CPKM incrementally increases model capacity by adding learners, providing a straightforward alternative to explicit rank search. A comprehensive comparison with Bayesian and reinforcement-learning methods is beyond the scope of this study, as these approaches involve diferent modeling assumptions and computational procedures.

## 6 Conclusion

We introduced a new gradient boosting model, GB-CPKM, that incrementally increases model capacity using fixed-rank CPKM base learners, providing an alternative to resource-intensive CP rank hyperparameter search. Experiments on real-world regression datasets of varying sizes and dimensionalities show that GB-CPKM trains up to 6ˆ faster and uses up to 2ˆ less peak memory than a cross-validation-based rank-selection baseline, while achieving comparable predictive performance. The hybrid variant GB-CPKM(H) further improves predictive performance on most datasets through an additional ALS sweep over the combined model, though at the cost of increased peak memory consumption.

Several directions remain for future work. We plan to extend the framework to tensor train kernel machines (TTKMs) and investigate rank adaptation strategies that allow ranks to increase or decrease during training. Another promising direction is to explore broader applications of GB-CPKM beyond regression, including classification and time-series forecasting.

## Acknowledgments

This publication is part of the project Sustainable learning for Artificial Intelligence from noisy large-scale data (with project number VI.Vidi.213.017) which is financed by the Dutch Research Council (NWO).

## References

Sarkhan Badirli, Xuanqing Liu, Zhengming Xing, Avradeep Bhowmik, Khoa Doan, and Sathiya S. Keerthi. Gradient boosting neural networks: GrowNet, 2020. URL https://arxiv.org/abs/2002.07971.

Kim Batselier. Low-rank tensor decompositions for nonlinear system identification: A tutorial with examples. IEEE Control Systems Magazine, 42(1):54–74, 2022. doi: 10.1109/MCS.2021.3122268.

Peter Bühlmann and Torsten Hothorn. Boosting algorithms: Regularization, prediction and model fitting. Statistical Science, 22(4):477–505, November 2007. ISSN 2168-8745. doi: 10.1214/07-sts242.

Flavio Camarrone and Marc M. Van Hulle. Fully Bayesian tensor-based regression. In 2016 IEEE 26th International Workshop on Machine Learning for Signal Processing (MLSP), pp. 1–6, 2016.

Andrzej Cichocki. Era of big data processing: A new approach via tensor networks and tensor decompositions. ArXiv, abs/1403.2048, 2014.

Andrzej Cichocki, Namgil Lee, Ivan Oseledets, Anh-Huy Phan, Qibin Zhao, and Danilo P. Mandic. Tensor networks for dimensionality reduction and large-scale optimization: Part 1 low-rank tensor decompositions. Foundations and Trends® in Machine Learning, 9(4–5):249–429, 2016. ISSN 1935-8245. doi: 10.1561/ 2200000059.

Andrzej Cichocki, Namgil Lee, Qibin Zhao, Anh-Huy Phan, Ivan Oseledets, and Danilo P. Mandic. Tensor networks for dimensionality reduction and large-scale optimization: Part 2 applications and future perspectives. Found. Trends Mach. Learn., 9(6):431–673, May 2017. ISSN 1935-8237. doi: 10.1561/2200000067.

Sergey V. Dolgov and Dmitry V. Savostyanov. Alternating minimal energy methods for linear systems in higher dimensions. SIAM Journal on Scientific Computing, 36(5):A2248–A2271, 2014.

Kun Dong, David Eriksson, Hannes Nickisch, David Bindel, and Andrew G Wilson. Scalable log determinants for Gaussian process kernel learning. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.

Dheeru Dua and Casey Graf. UCI machine learning repository, 2017. URL http://archive.ics.uci.edu/ ml.

Yan Duan, Xi Chen, Rein Houthooft, John Schulman, and Pieter Abbeel. Benchmarking deep reinforcement learning for continuous control. In International conference on machine learning, pp. 1329–1338. PMLR, 2016.

Luca Franceschi, Michele Donini, Valerio Perrone, Aaron Klein, Cédric Archambeau, Matthias Seeger, Massimiliano Pontil, and Paolo Frasconi. Hyperparameter optimization in machine learning. Found. Trends Mach. Learn., 18(6):1054–1201, October 2025. ISSN 1935-8237. doi: 10.1561/2200000088.

Jerome Friedman and Jacqueline Meulman. Multiple additive regression trees with application in epidemiology. Statistics in Medicine, 22:1365–1381, 04 2003. doi: 10.1002/sim.1501.

Jerome H. Friedman. Greedy function approximation: A gradient boosting machine. The Annals ofStatistics, 29(5):1189 – 1232, 2001. doi: 10.1214/aos/1013203451.

Zoubin Ghahramani. Kin8nm, 1996. URL https://doi.org/10.5281/zenodo.14645866.

Lars Grasedyck. Hierarchical singular value decomposition of tensors. SIAM Journal on Matrix Analysis and Applications, 31(4):2029–2054, 2010. doi: 10.1137/090764189.

Alexandros Haliassos, Kriton Konstantinidis, and Danilo P. Mandic. Supervised learning for non-sequential data: A canonical polyadic decomposition approach, 2021. URL https://arxiv.org/abs/2001.10109.

Trevor Hastie, Robert Tibshirani, and Jerome Friedman. The Elements of Statistical Learning: Data Mining, Inference, and Prediction. Springer, New York, NY, 2nd edition, 2009. ISBN 978-0-387-84857-0.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770–778, 2016.

James Hensman, Nicolò Fusi, and Neil D. Lawrence. Gaussian processes for big data. In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence, UAI’13, pp. 282–290, Arlington, Virginia, USA, 2013. AUAI Press.

James Hensman, Nicolas Durrande, and Arno Solin. Variational Fourier features for Gaussian processes. Journal of Machine Learning Research, 18(151):1–52, 2018.

Sebastian Holtz, Thorsten Rohwedder, and Reinhold Schneider. The alternating linear scheme for tensor optimization in the tensor train format. SIAM Journal on Scientific Computing, 34(2):A683–A713, 2012.

Afra Kilic and Kim Batselier. Interpretable Bayesian tensor network kernel machines with automatic rank and feature selection, 2025. URL https://arxiv.org/abs/2507.11136.

Tamara G. Kolda and Brett W. Bader. Tensor decompositions and applications. SIAM Review, 51(3): 455–500, 2009. doi: 10.1137/07070111X.

Clara Menzen, Manon Kok, and Kim Batselier. Tensor network square root Kalman filter for online Gaussian process regression. Automatica, 183:112694, 2026. ISSN 0005-1098. doi: https://doi.org/10.1016/j. automatica.2025.112694.

Alexander Novikov, Mikhail Trofimov, and Ivan Oseledets. Exponential machines, 2017. URL https: //arxiv.org/abs/1605.03795.

I. V. Oseledets. Tensor-train decomposition. SIAM Journal on Scientific Computing, 33(5):2295–2317, 2011. doi: 10.1137/090752286.

Albert Saiapin and Kim Batselier. Tensor network kernel machines: A jax framework for machine learning and nonlinear system identification, 2026. URL https://arxiv.org/abs/2608.07043.

Robert E. Schapire and Yoav Freund. Boosting: Foundations and Algorithms. The MIT Press, 05 2012. ISBN 9780262301183. doi: 10.7551/mitpress/8291.001.0001.

Ulrich Schollwöck. The density-matrix renormalization group in the age of matrix product states. Annals of Physics, 326(1):96–192, 2011. ISSN 0003-4916. January 2011 Special Issue.

Nicholas D. Sidiropoulos, Lieven De Lathauwer, Xiao Fu, Kejun Huang, Evangelos E. Papalexakis, and Christos Faloutsos. Tensor decomposition for signal processing and machine learning. Trans. Sig. Proc., 65(13):3551–3582, July 2017. ISSN 1053-587X.

Arno Solin and Simo Särkkä. Hilbert space methods for reduced-rank Gaussian process regression. Statistics and Computing, 30(2):419–446, August 2019. ISSN 1573-1375. doi: 10.1007/s11222-019-09886-w.

André Uschmajew. Local convergence of the alternating least squares algorithm for canonical tensor approximation. SIAM Journal on Matrix Analysis and Applications, 33(2):639–652, 2012. doi: 10.1137/110843587.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Frederiek Wesel and Kim Batselier. Large-scale learning with Fourier features and tensor decompositions. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 17543–17554. Curran Associates, Inc., 2021.

Frederiek Wesel and Kim Batselier. Quantized Fourier and polynomial features for more expressive tensor network models. In Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, 02–04 May 2024.

Tong Yu and Hong Zhu. Hyper-parameter optimization: A review of algorithms and applications, 2020. URL https://arxiv.org/abs/2003.05689.

Shaoshuang Zhang, Zhao Li, Wenlong Liu, Jiaqi Zhao, and Ting Qin. Rank selection method of cp decomposition based on deep deterministic policy gradient algorithm. IEEE Access, 12:97374–97385, 2024. doi: 10.1109/ACCESS.2024.3428370.

Qibin Zhao, Liqing Zhang, and Andrzej Cichocki. Bayesian sparse Tucker models for dimension reduction and tensor completion, 2015. URL https://arxiv.org/abs/1505.02343.