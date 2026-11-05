# BAM: Bivariate Additive Models for Interpretable Tabular Prediction

Atsushi Koike<sup>[0000−0002−6262−4308]</sup>

Tohoku University, Sendai, Japan

koike@tohoku.ac.jp

Abstract. Generalized Additive Models with pairwise interactions (GA<sup>2</sup>Ms) are regarded as glass-box models, since each pairwise shape function can be visualized as a contour plot. In practice, the shape functions learned by methods such as EBM or NODE-GAM are so complex—irregular contours and sharp local fluctuations—that a practitioner cannot meaningfully interpret them or check them against domain knowledge. Auditability in this stronger sense requires substantially simpler shape functions. We argue that such simplification can match the accuracy of standard GAMs if the right feature pairs are captured. We propose Bivariate Additive Models (BAMs), a class of GA<sup>2</sup>Ms in which every pairwise shape function is constrained to a simple parametric form, instantiated as an asymmetric bivariate Gaussian. Each component is visualizable as a unimodal contour map, and its feature pair is selected automatically during training. Experiments on six tabular benchmarks show that BAM attains predictive accuracy comparable to standard GAMs and approaches that of GA<sup>2</sup>Ms. We also show the trade-of between accuracy and the number of components to inspect.

Keywords: Glass-box models · Generalized additive models · GA2M · Interpretability · Tabular prediction · Bivariate Gaussian.

## 1 Introduction

Machine learning is widely used for important decisions in society, such as medical diagnosis [4], financial lending, and judicial judgment [19]. A closely related need arises in industrial settings (e.g., a production line monitored by many sensors), where practitioners must identify which small subset of variables—or pairs of variables—drives an outcome from among hundreds of recorded signals, in a form they can quickly share with domain experts. In such settings, high predictive performance alone is insuficient: the criteria behind specific predictions must be transparent and intelligible enough to be examined, questioned, and accepted as reasonable.

Unfortunately, understanding why a particular prediction was made is typically challenging. Techniques such as LIME [18] and SHAP [13] have been proposed to provide post-hoc explanations for predictions. However, these explanations can difer substantially from actual model behavior [21]. Models whose prediction processes are inherently visible are known as glass-box models. Against this backdrop, the importance of glass-box models is increasingly recognized.

Generalized Additive Models (GAMs) are a representative class of glass-box models that achieve high predictive accuracy. GAMs compute a score for each feature and sum them up along with a bias term to obtain the final prediction. For example, when predicting disease risk from health checkup data, the model uses a per-feature shape function to compute an individual risk score for each measurement $( \mathrm { e . g . }$ ., blood pressure, BMI), and these individual scores are summed to obtain the total risk score. Visualizing these shape functions provides interpretability of the prediction process. $\mathrm { G A } ^ { 2 } \mathrm { M }$ extends this concept by incorporating pairwise interactions. In $\mathbf { G A } ^ { 2 } \mathbf { M s }$ , shape functions can depend on two features (e.g., a function of both “age” and “blood pressure”), which allows the model to capture complex relationships between features.

Existing research on GAMs and GA<sup>2</sup>Ms has primarily focused on approximating the true underlying function as accurately as possible. Consequently, the resulting shape functions tend to become excessively complex (as shown in Fig. 3), exhibiting irregular contours and sharp local fluctuations. Such complexity makes it dificult for a practitioner to interpret each shape function on its own. Checking whether the criterion the model has learned makes sense given established knowledge in the application domain is even more dificult. For applications where the model itself must be auditable—not merely accurate—these complex shape functions are unsuitable, regardless of how well they approximate the underlying function.

To address this, we argue that the shape functions of a $\mathrm { G A } ^ { 2 } \mathrm { M }$ should be substantially simpler than those produced by unconstrained fitting. A natural concern is that such radical simplification will sacrifice predictive accuracy. The central claim of this paper is that this is not necessarily true: if the right pairs of features are captured, strongly constrained pairwise functions can match the accuracy of standard GAMs and approach that of $\mathrm { { G A ^ { 2 } M s } }$

We propose Bivariate Additive Models (BAMs), a class of $\mathbf { G A } ^ { 2 } \mathbf { M s }$ in which every pairwise shape function is constrained to a simple parametric form. BAM learns the feature pair for each component via a diferentiable selection mechanism. Each component is visualizable as a single contour map. In this work, we instantiate the shape function as an asymmetric bivariate Gaussian (Bigaussian).

BAM computes the prediction as a sum of bivariate functions. This simplicity allows users to intuitively understand how input values afect the output. Figure 1 shows an example of BAM applied to the Titanic passenger survival prediction task. In this example, the model calculates a logit by summing the outputs of three components (bivariate functions) and a bias term $( - 0 . 0 1 0 )$ . The survival probability is obtained by passing this logit through a sigmoid function, where a positive logit indicates a prediction of survival. Crucially, important features for prediction are aggregated into these three components. For instance, the first component indicates that the survival probability increases for females (sex\_male=0) in the upper class (pclass=1). In this way, features important for prediction are successfully extracted into a few components.

Each component in BAM represents the deviation of the output from a baseline value (bias term). The concept of attributing deviations from a baseline to features is similar to SHAP [13]. However, while SHAP estimates feature contributions in a posthoc manner, BAM directly adds the calculated values to the final output. Consequently, BAM is inherently free from the reliability issues associated with post-hoc contribution estimation in SHAP. Furthermore, BAM can assign values to feature pairs, which is particularly efective for datasets where interactions are critical.

![](images/6459a033cc787df607c6721ac5ac08c79a05286388b4e9337381da8f1345e463.jpg)

![](images/0f0d5e8c2780d23bdac0758cd97997e7e6eea4fed51e1aa30c474551b93148c9.jpg)

![](images/49bea4ad0242a25d3092dbc5607b4eb4e7f9d1850b7d96dc2179b9bff629247a.jpg)  
logit = bias + comp.1 + comp.2 + comp.3 = (−0.010) + (0.005) + (−2.012) + (0.097) = −1.919 < 0 ⇒ prediction = 0 (true label = 0)  
Fig. 1. Visualization of passenger survival prediction on the Titanic dataset using the proposed model (BAM). The model calculates a single prediction by summing the scores from three bivariate graphs and a bias term (−0.010). The visualized values represent logits, where a positive total score indicates a prediction of survival. The blue points indicate the scores for a single sample, and the red crosses represent the centers of the Gaussian components.

We empirically demonstrate that BAMs achieve predictive accuracy comparable to standard GAMs and approach that of $\mathrm { G A } ^ { 2 } \mathrm { M s }$ . We also compare BAM visualizations with those of GA<sup>2</sup>Ms and SHAP, and show that BAM makes the overall model behavior easier to grasp. Furthermore, we show how accuracy improves as the number of components increases.

## 2 Preliminaries

## 2.1 GAM

Among glass-box models, the Generalized Additive Model (GAM) is widely known for achieving high predictive performance while maintaining interpretability. GAM calculates a score for each input feature and obtains the final output by summing these scores together with a bias term.

Given � features $\pmb { x } = [ x _ { 1 } , x _ { 2 } , \dots , x _ { m } ]$ , the GAM prediction for the target value, denoted by �, is represented as:

$$
g ( y ) = \beta _ { 0 } + \sum _ { i = 1 } ^ { m } f _ { i } ( x _ { i } ) ,\tag{1}
$$

where $g$ is a link function (e.g., logit for classification), and $\beta _ { 0 }$ represents a bias term. For instance, in disease risk prediction, each feature $x _ { i } \ ( \mathrm { e . g } $ ., BMI) yields a risk score $f _ { i } ( x _ { i } )$ , and the total risk is determined by summing these scores and the bias term $\beta _ { 0 }$

A key challenge in GAMs is the estimation of the shape functions $f _ { i } .$ . Explainable Boosting Machine (EBM) [4,11] utilizes boosted decision trees to estimate these functions, thereby achieving predictive performance comparable to black-box models. More recently, deep learning-based approaches have been proposed to learn each shape function more flexibly: Neural Additive Models (NAM) [1] use one sub-network per feature; NODE-GAM [6] adopts oblivious decision trees with diferentiable feature selection;

Neural Basis Models (NBM) [17] share basis functions across features for scalability; LA-NAM [3] introduces Bayesian principles for uncertainty estimation; and GP-NAM [22] uses a single-layer neural-network construction of Gaussian processes via random Fourier features, yielding a convex objective. These methods improve expressivity in various ways; however, the learned shape functions remain free-form rather than constrained to a compact parametric form, which can make them dificult to interpret on their own. BAM difers in design philosophy by constraining each (pairwise) shape function to a compact parametric form rather than maximizing flexibility.

## 2.2 GA<sup>2</sup>M

GA<sup>2</sup>M (Generalized Additive Models with pairwise interactions) [12] is an extension of GAMs that incorporates pairwise feature interactions. The $\mathrm { G A } ^ { 2 } \mathrm { M }$ prediction is formulated as:

$$
g ( y ) = \beta _ { 0 } + \sum _ { i = 1 } ^ { m } f _ { i } ( x _ { i } ) + \sum _ { i = 1 } ^ { m - 1 } \sum _ { j = i + 1 } ^ { m } f _ { i , j } ( x _ { i } , x _ { j } ) .\tag{2}
$$

By explicitly modeling pairwise feature interactions through $f _ { i , j } , \mathrm { G A } ^ { 2 } \mathrm { M s }$ can achieve higher predictive performance than standard GAMs. However, incorporating these pairwise functions, each of which has a potentially complex shape, makes the resulting model hard to interpret intuitively. Note that it is not necessary to include interactions for every feature pair; unimportant interactions can be omitted by setting $f _ { i , j } = 0$

Modern implementations, such as EBM and NODE-GAM, support this $\mathrm { G A } ^ { 2 } \mathrm { M }$ framework to enhance predictive performance. More recently, GRAND-SLAMIN’ [10] addresses the combinatorial growth of pairwise terms in GA<sup>2</sup>M by imposing structural constraints on which feature pairs are included, learning sparse selections of pairs through diferentiable optimization on tree-based shape functions. Like GRAND-SLAMIN’, BAM learns feature pairs end-to-end, but additionally constrains the form of each pairwise function to a parametric ellipse, which yields visualizations whose geometric semantics can be read directly from learned parameters.

## 3 Proposed Model

## 3.1 Bivariate Additive Model

BAMs are a class of $\mathbf { G A } ^ { 2 } \mathbf { M s }$ . They constrain every pairwise shape function to a simple parametric form. We first introduce the general form of BAMs and then describe the specific shape functions used in this study.

Let $\pmb { x } = [ x _ { 1 } , x _ { 2 } , \ldots , x _ { m } ] \in \mathbb { R } ^ { m }$ be the input features and $y$ be the target. We define a tuple of feature index pairs I to specify the input features for $n _ { c }$ bivariate shape functions:

$$
\bar { \cal I } = [ \ ( i [ 1 ] , j [ 1 ] ) , \ldots , ( i [ n _ { c } ] , j [ n _ { c } ] ) \ ] \ .\tag{3}
$$

The BAM prediction is defined as:

$$
g ( y ) = \beta _ { 0 } + \sum _ { k = 1 } ^ { n _ { c } } f _ { k } ( x _ { i [ k ] } , x _ { j [ k ] } ) ,\tag{4}
$$

where $g ( \cdot )$ is a link function and $f _ { k }$ is the �-th bivariate shape function. For regression tasks, we use the identity link $g ( y ) = y$ together with the mean squared error loss. For binary classification, we use the logit link $\begin{array} { r } { g ( y ) = \log \frac { y } { 1 - y } } \end{array}$ together with the binary cross-entropy loss.

## 3.2 Symmetric Bigaussian Shape Function

In this study, we employ Bigaussian (bivariate Gaussian) functions as shape functions. We first introduce the symmetric Bigaussian shape function, and then describe the asymmetric version in the next section.

Letting $z = [ z _ { 1 } , z _ { 2 } ] ^ { T }$ denote the input vector of the symmetric Bigaussian shape function $f _ { k }$ , we have:

$$
f _ { k } ( z ) = a _ { k } \exp \left( - \frac { 1 } { 2 } ( z - b _ { k } ) ^ { T } C _ { k } ^ { - 1 } ( z - b _ { k } ) \right) ,\tag{5}
$$

where $a _ { k }$ denotes the height of the Bigaussian, $\pmb { b } _ { k }$ represents its center position, and $C _ { k }$ is its covariance matrix. To clarify the shape of $f _ { k }$ expressed by Eq. (5), we rewrite it using the eigenvalues of $C _ { k }$

For the two-variable Gaussian case, let $( \lambda _ { k , 1 } , \pmb { v } _ { k , 1 } )$ and $( \lambda _ { k , 2 } , \pmb { v } _ { { k , 2 } } )$ denote the eigenvalue-eigenvector pairs of $C _ { k }$ . When each eigenvector is normalized to unit length, the eigenvectors $v _ { k , 1 } , v _ { k , 2 }$ can be represented as:

$$
\left[ \boldsymbol { v } _ { k , 1 } \boldsymbol { v } _ { k , 2 } \right] = \left[ \begin{array} { l l } { \cos \theta _ { k } - \sin \theta _ { k } } \\ { \sin \theta _ { k } \cos \theta _ { k } } \end{array} \right] .\tag{6}
$$

$C _ { k }$ can be expressed through eigen-decomposition as follows:

$$
\boldsymbol { C } _ { k } = \left[ \boldsymbol { v } _ { k , 1 } \boldsymbol { v } _ { k , 2 } \right] \left[ \begin{array} { c c } { \lambda _ { k , 1 } } & { 0 } \\ { 0 } & { \lambda _ { k , 2 } } \end{array} \right] \left[ \boldsymbol { v } _ { k , 1 } \boldsymbol { v } _ { k , 2 } \right] ^ { T } .\tag{7}
$$

Therefore, the function $f _ { k }$ can be rewritten as:

$$
f _ { k } ( z ) = a _ { k } \exp \left( - \frac { 1 } { 2 } \left\| \left[ \begin{array} { c c } { \frac { 1 } { \sqrt { \lambda _ { k , 1 } } } } & { 0 } \\ { 0 } & { \frac { 1 } { \sqrt { \lambda _ { k , 2 } } } } \end{array} \right] \left[ \begin{array} { c c } { \cos \theta _ { k } } & { \sin \theta _ { k } } \\ { - \sin \theta _ { k } \cos \theta _ { k } } \end{array} \right] ( z - b _ { k } ) \right\| ^ { 2 } \right) ,\tag{8}
$$

where all parameters indexed by $k$ are estimated through training. $\theta _ { k }$ represents the rotation angle of the ellipse (contour lines of the Bigaussian function) with respect to the original coordinate axes $( z _ { 1 } , z _ { 2 } ) . \sqrt { \lambda _ { k , 1 } }$ and $\sqrt { \lambda _ { k , 2 } }$ correspond to the scaling factors along the principal axes of the ellipse.

## 3.3 Asymmetric Bigaussian Shape Function

To enable more diverse shapes while maintaining interpretability, we allow the variance of the Gaussian function to difer along the positive and negative directions of each principal axis. To this end, we introduce a function $\phi$ that applies diferent scaling

![](images/0f174f0360d30f15f6332bdd4245ab3bd3479af9c3df93dcdaee8304a784974c.jpg)  
Fig. 2. Relationship between learned parameters and the shape of the Bigaussian function. The vector $\pmb { b } _ { k }$ represents the center coordinates of the Bigaussian. The angle $\theta _ { k }$ denotes the rotation of the first eigenvector relative to the $z _ { 1 }$ -axis. The parameters $\sqrt { \lambda _ { k , 1 } ^ { + } }$ and $\sqrt { \lambda _ { k , 1 } ^ { - } }$ represent the scaling factors along the first eigenvector in the positive and negative directions, respectively, relative to a standard Gaussian (i.e., one with identity covariance matrix). Similarly, $\sqrt { \lambda _ { k , 2 } ^ { + } }$ and $\sqrt { \lambda _ { k , 2 } ^ { - } }$ represent scaling factors along the second eigenvector in the positive and negative directions, respectively.

factors depending on whether the input is positive or negative along each axis. The function $\phi$ is defined as follows:

$$
\phi ( h ; \lambda ^ { + } , \lambda ^ { - } ) = \left\{ \begin{array} { l l } { { \lambda ^ { + } h } } & { { ( h \ge 0 ) } } \\ { { \lambda ^ { - } h } } & { { ( h < 0 ) . } } \end{array} \right.\tag{9}
$$

Next, we modify the eigenvalues $\lambda _ { k , 1 } , \lambda _ { k , 2 }$ of the covariance matrix used in Eq. (8). We assign diferent values depending on whether the input coordinate is positive or negative along each eigenvector direction. Specifically, we use $\lambda _ { k , 1 } ^ { + }$ if the input along the first eigenvector ${ \pmb v } _ { k , 1 }$ is non-negative, and $\lambda _ { k , 1 } ^ { - }$ if it is negative. We define $\lambda _ { k , 2 } ^ { + }$ and $\lambda _ { k , 2 } ^ { - }$ similarly. With these adjustments, the resulting asymmetric Bigaussian shape function is given by:

$$
f _ { k } ( z ) = a _ { k } \exp \left( - \frac { 1 } { 2 } \left\| \left[ \frac { \phi ( h _ { 1 } ; \frac { 1 } { \sqrt { \lambda _ { k , 1 } ^ { + } } } , \frac { 1 } { \sqrt { \lambda _ { k , 1 } ^ { - } } } ) } { \phi ( h _ { 2 } ; \frac { 1 } { \sqrt { \lambda _ { k , 2 } ^ { + } } } , \frac { 1 } { \sqrt { \lambda _ { k , 2 } ^ { - } } } ) } \right] \right\| ^ { 2 } \right) ,\tag{10}
$$

where

$$
{ \binom { h _ { 1 } } { h _ { 2 } } } = { \left[ \begin{array} { l l } { \cos \theta _ { k } } & { \sin \theta _ { k } } \\ { - \sin \theta _ { k } \cos \theta _ { k } } \end{array} \right] } ( z - b _ { k } ) .\tag{11}
$$

Figure 2 illustrates the geometric interpretation of each parameter. The four � parameters represent scaling factors of the ellipse in the positive and negative directions along each eigenvector, relative to a standard Gaussian function (i.e., a Gaussian with an identity covariance matrix).

## 3.4 Model Training

We implement the BAM given by Eq. (4) as a neural network, and estimate its parameters via backpropagation. The shape functions within the model are defined by Eq. (10). To

enforce the constraint that eigenvalues of the covariance matrices must be positive, we introduce an additional parameter �<sup>′</sup> and define each eigenvalue � as follows:

$$
\lambda = \exp ( s \lambda ^ { \prime } ) ,\tag{12}
$$

where � controls the magnitude of changes in eigenvalues, and we set $s = 1 . 5$ throughout this paper. In our experiments, the initial value of $\lambda ^ { \prime }$ is set ${ \mathrm { t o } } - 1 . 0$ . The amplitude $a _ { k }$ is free to take negative values, allowing each component to represent either a “peak” or a “valley” in the contour map; we initialize $a _ { k }$ and $\pmb { b } _ { k }$ from a standard normal distribution, and set the rotation angle $\theta _ { k }$ to zero.

While the input features to each shape function are specified by the tuple I in Eq. (3), we introduce a diferentiable selection mechanism to enable the neural network to automatically learn these feature selections. We express each element of the input vector $z = [ z _ { 1 } , z _ { 2 } ] ^ { T }$ for a bivariate shape function as a weighted average of all available features. Through training, the weighting coeficients are optimized to converge to a one-hot vector, thereby selecting a single specific feature. Following NODE-GAM [6], we adopt a temperature annealing parameter $T _ { e } ,$ which gradually enforces these weight coeficients to become a one-hot vector as training progresses. The input vector � is computed as:

$$
z = \mathrm { e n t m a x } _ { \alpha } ( W / T _ { e } ) { \pmb x } ,\tag{13}
$$

where $W \in \mathbb { R } ^ { 2 \times m }$ is a weight matrix specific to each component, and the entmax activation function [16] with $\alpha = 1 . 5$ is applied separately to each row of $W / T _ { e }$ . The entmax function induces sparsity in each weight vector. The temperature parameter $T _ { e }$ is annealed gradually from an initial value $T _ { \mathrm { m a x } } ~ = ~ 1 . 0$ to a final value $T _ { \mathrm { m i n } } = 0 . 0 0 1$ Given the maximum number of epochs �, the temperature at epoch �, denoted by $T _ { e }$ , is computed as:

$$
T _ { e } = \exp \left( \log T _ { \operatorname* { m a x } } - \frac { e } { E } \left( \log T _ { \operatorname* { m a x } } - \log T _ { \operatorname* { m i n } } \right) \right) ,\tag{14}
$$

where epochs are indexed from $e \ = \ 0$ to $\textit { e } = E$ . After this annealing stage, we fix the selection of each row of $W$ to the feature with the largest weight (i.e., replace the softened selection with its argmax) and continue training the remaining parameters for a second stage, so that the Gaussian parameters are refined under the fixed feature selection. Because entmax is applied to each row of $W / T _ { e }$ independently, the two rows may converge to the same feature, in which case the component reduces to a univariate shape function.

## 3.5 Multiple Restart Strategy

The optimization process of our proposed method is sensitive to initialization. To mitigate this sensitivity, we adopted a multiple restart strategy, similar to the standard practice in K-Means clustering [15]. Specifically, we executed the training process ten times with diferent random initializations for each fold and selected the model with the lowest training loss. Selecting the best model based on training loss (rather than validation loss) is unlikely to cause overfitting, due to the model’s simplicity.

## 4 Evaluation Results

This section evaluates the performance of the proposed BAM. First, we compare BAM with other predictive models. Next, we compare the visualization of BAM with those of existing methods. We then investigate how the number of Gaussian components influences model performance.

## 4.1 Datasets

We evaluated our method using three regression and three binary classification datasets. For regression, we used: California Housing [14] (CA Housing, 20,640 samples, 8 numerical features); Bike Sharing [9] (Bikeshare, 17,379 samples, 13 numerical features); and Wine Quality [7] (Wine, white wine subset, 4,898 samples, 11 numerical features). For binary classification, we used: Titanic [8] (712 samples after dropping missing values, ‘sex’ and ‘embarked’ as categorical, others numerical); Adult [2] (13,379 samples, 8 categorical features); and Diabetes Health Indicators [5] (Diabetes, balanced version, 70,692 samples, 21 numerical features). As preprocessing, categorical variables were one-hot encoded, and numerical variables (including targets for regression) were standardized.

## 4.2 Baseline Evaluation

We evaluated predictive performance (RMSE for regression, AUC for classification) using 5-fold cross-validation. For BAM, we evaluated a model with 15 asymmetric components, trained with the Adam optimizer (learning rate 0.001, batch size 128) for 150 epochs in the annealing stage, followed by 150 additional epochs with the feature selection fixed (the second stage described in Sect. 3.4).

The comparison models used in this evaluation were as follows. Linear Regression/Logistic Regression (Linear), Decision Tree (CART), and XGBoost were used with their default settings. For Explainable Boosting Machine (EBM), we used InterpretML. We set interactions=0 for the GAM configuration, while using the default setting (interactions="5x") for the $\mathrm { G A } ^ { 2 } \mathrm { M }$ configuration. For NODE-GAM, we used the authors’ implementation for both GAM and $\mathbf { G A } ^ { 2 } \mathbf { M } .$ , removing only the default preprocessing. For GRAND-SLAMIN’, we used the oficial implementation with its default architecture. As with the other baselines, we did not tune hyperparameters: we replaced the Optuna-based hyperparameter search with a standard mini-batch training loop (Adam, learning rate $5 \times \bar { 1 } 0 ^ { - \bar { 3 } }$ , batch size 128, 300 epochs). To stabilize training on high-dimensional inputs such as the one-hot encoded Adult dataset, we initialized the selection gates so that all gates start near zero and scaled the regularization coeficients inversely with the number of candidate terms. Finally, as a Deep Neural Network (DNN) baseline, we trained a 5-layer fully connected network (64-64-64-32-1 units) with 0.2 dropout using Adam (learning rate 0.001, batch size 128) for 150 epochs.

Table 1 presents the predictive performance. BAM achieves accuracy comparable to standard GAMs and approaches that of $\mathbf { G A } ^ { 2 } \mathbf { M s }$ on some datasets. On Bikeshare, BAM even outperforms the GAMs by a large margin and comes close to the $\mathbf { G A } ^ { 2 } \mathbf { M s }$ . This is likely because Bikeshare contains strong interactions between features, which univariate shape functions cannot capture.

Table 1. Comparison of predictive performance on multiple datasets (Mean ± Std). BAM denotes the proposed model with $n _ { c } = 1 5$ asymmetric Gaussian components.
<table><tr><td>Model</td><td>CA Housing (RMSE↓)</td><td>Bikeshare (RMSE↓)</td><td>Wine (RMSE↓)</td><td>Titanic (AUC↑)</td><td>Adult (AUC↑)</td><td>Diabetes (AUC↑)</td></tr><tr><td>Linear CART</td><td>0.728 ± 0.017 0.720 ± 0.010</td><td>141.907 ±2.022 58.550 ±1.849</td><td>0.755 ± 0.030 0.821 ± 0.026</td><td>0.855 ± 0.039 0.731 ± 0.024</td><td>0.907 ± 0.003 0.749 ± 0.001</td><td> $0 . 8 2 5 { \scriptstyle \pm 0 . 0 0 3 }$   $0 . 6 5 6 \pm 0 . 0 0 6$ </td></tr><tr><td>EBM (GAM) NODE-GAM (GAM)</td><td>0.550 ± 0.013 0.597 ± 0.015</td><td>99.835 ±1.465 100.234±1.486</td><td>0.698 ± 0.020 0.704 ± 0.021</td><td>0.857 ± 0.041 0.846 ± 0.045</td><td>0.928 ± 0.003 0.918 ± 0.002</td><td> $0 . 8 2 9 { \scriptstyle \pm 0 . 0 0 2 }$   $0 . 8 2 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td>BAM (ours)</td><td>0.589 ± 0.020</td><td>65.990±1.849</td><td>0.721 ± 0.021</td><td>0.869 ± 0.040</td><td>0.912 ±0.002</td><td>0.824±0.004</td></tr><tr><td>EBM (GA2M)</td><td>0.462 ± 0.012</td><td>49.428±0.852</td><td>0.667 ± 0.014</td><td>0.865 ± 0.036</td><td>0.929 ± 0.003</td><td></td></tr><tr><td>NODE-GAM (GA2M)</td><td>0.538 ± 0.024</td><td>57.637 ±3.380</td><td>0.685 ± 0.018</td><td>0.844± 0.034</td><td>0.919 ± 0.002</td><td> $0 . 8 3 1 \pm 0 . 0 0 2$   $0 . 8 3 2 \pm 0 . 0 0 2$ </td></tr><tr><td>GRAND-SLAMIN&#x27;</td><td>0.564±0.019</td><td>57.975 ±2.105</td><td>0.704 ± 0.018</td><td>0.868 ± 0.030</td><td>0.914± 0.003</td><td> $0 . 8 2 9 { \scriptstyle \pm 0 . 0 0 2 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>DNN</td><td> $0 . 5 1 2 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $5 6 . 0 8 8 \pm 1 . 0 1 1$ </td><td>0.670±0.013</td><td>0.852 ± 0.010</td><td>0.897 ± 0.002</td><td> $0 . 8 2 4 \pm 0 . 0 0 1$ </td></tr><tr><td>XGBoost</td><td> $0 . 4 6 9 \pm 0 . 0 1 2$ </td><td>40.633 ± 1.298</td><td>0.633 ± 0.016</td><td>0.838 ± 0.035</td><td>0.928 ± 0.002</td><td> $0 . 8 2 3 \pm 0 . 0 0 2$ </td></tr></table>

## 4.3 Model Visualization

Figure 1 visualizes the BAM model trained on the Titanic dataset, which consists of three bivariate components and a bias term. The visualized values represent logits, and their sum (plus the bias term) determines the final prediction; a positive total score indicates survival. The decomposition reveals how the model captures key survival factors. We discuss each component in turn and compare it with what is historically known about Titanic survival outcomes, illustrating how BAM’s visualization supports model auditing. For the first component (class and gender), the Gaussian peak lies at (sex\_male=0, pclass=1), so female passengers in first class receive a strongly positive contribution. This is consistent with the view that women in first class were more likely to have been prioritized for evacuation. The second component (age and class) identifies a strong negative contribution for adult passengers in the lower class, with the Gaussian centered in the region (age large, pclass=3). This is consistent with reported low survival rates for this group. Such a conditional structure cannot be captured by a univariate function on age or pclass alone. The third component (single-female adjustment) captures a more subtle adjustment: the Gaussian peak lies at (sex\_male=0, sibsp=0), so single female passengers receive an additional positive contribution beyond what Component 1 already provides for women.

For comparison, we visualize the results of EBM $( \mathrm { G A } ^ { 2 } \mathbf { M } )$ and SHAP in Figs. 3 and 4, respectively. Figure 3 shows the visualization of EBM. Note that for this visualization, we limited the number of interactions to half the number of features to avoid an excessive number of plots (interactions="0.5x"). Even with this reduction, the large number of plots makes it dificult to grasp the overall picture. Furthermore, the learned functions for ‘age’ and ‘fare’ exhibit irregular shapes, making it dificult to grasp underlying trends.

Figure 4 shows the SHAP dependence plots of the XGBoost model. In each plot, the horizontal axis represents the value of a feature, and the vertical axis represents its SHAP value, i.e., the contribution of that feature to the model’s output. A notable observation is that, even at the same feature value, the SHAP value varies considerably across samples, producing a vertical spread of points rather than a single curve. This spread indicates that the contribution of a feature is not determined by its own value alone, but also depends on the values of other features. However, the dependence plot does not reveal which other features are responsible for this variation, nor how they jointly shape the contribution. As a result, the underlying logic that links feature values to the model’s output remains opaque.

![](images/142d9da4449a8c14ce218f99275d01077a29ad0e71991ed0ec67f11596af2976.jpg)

![](images/fe2d625303565b4f44b14617882b5de4987de6b3695dc7db1c7719465839525e.jpg)

![](images/9c7c8cb0a5c004abeca9cc1b9da267be84c349b4898d5b9affaa4b70544c02fa.jpg)

![](images/1bf26dc09e440fe0a535b0cad1a8841caeb751dffb3f07817da0ea5da304b1a9.jpg)

![](images/c0e421aee921c0102ca7d51a87691e43dd1973d172725b6ad8dbd017caa5a92f.jpg)

![](images/816c1a76a63dd9eb8b8253052829ec2d2aeab93c921882ebdcc382189817c4b7.jpg)

![](images/21a3e012227da973106bd31ec3af16c47362b5e50e74fa5efb3fc9ed8ed71b8a.jpg)

![](images/0fa33622659a321e9cb369f0e0e5a6a42f35e4ea7298dc0ca4874c6bd8c11856.jpg)

![](images/ec664df385d86fc0621c2571c4330423540e5d0b24c7616e58abda5fba703fb6.jpg)

![](images/b27ef3154af0f836936520922cf8a7d1e740bd6947d48d9653756342aa50e4ee.jpg)

![](images/85077873cf001132f57596f69984fba3f8bf4ce7a87f10f603c6609c19f77a79.jpg)

![](images/8bbe450936766500a1ab227976118b32b883f305619faa680e7cfa88ff5f1398.jpg)

![](images/34a4aaa75c12cad1436e947eb08565f7cc17ac8eb3d64ac21e287e41de5cb5a0.jpg)  
Fig. 3. Visualization of EBM $( \mathrm { G A } ^ { 2 } \mathbf { M } )$ shape functions on the Titanic dataset. The first two rows show univariate shape functions; the last row shows pairwise interaction shape functions. The num ber of pairwise interactions was reduced from the InterpretML default (interactions="5x") to "0.5x" (half the number of features) for this figure only.

![](images/7cf3e37113b711551d87aa7085f163eaecdeb42701695cd44afcf32eee40dfbc.jpg)

![](images/c35555472469701e665c29abc5fc5151634f779140beb232789ad385cb1d41a7.jpg)

![](images/d3c4acdd2341b333d068c6a226ad2480f3ef3da8fbfeff571b92d2c4ad1a3e59.jpg)

![](images/7231478f40381f003a8fefcb01b12039bdb9efc79f03c3bba77a76c44763664d.jpg)

![](images/1639d95774b687d7b8c7bb51bba9430f7741a5ebabb62baf689946c402d020f5.jpg)

![](images/a7deec1e30164d66a419b9249d276394644f97b87f38d20577491c3a0e383504.jpg)

![](images/d32e15ad2a4b80cf120924be51ccbeb51055425eaca3c6bd63ca8ab8a90292eb.jpg)

![](images/cf73a65f4c64609113bfdd71bcef401160c2200d059e16b32933bbac2930ff24.jpg)

![](images/cfb1fc740ab6d073171a50e3ff97c0cf65faf84bd0cfceb811d0685e97834198.jpg)  
Fig. 4. SHAP dependence plots for the XGBoost model on the Titanic dataset. Each plot shows the relationship between a feature value and its SHAP value.

![](images/e326cca90eb94a4e8200e8e65f5f4aa15daf42e51b1ed950933884e2b6ecdecc.jpg)  
n<sub>c</sub>

![](images/474b23030983c6392d80055edfb61aac3d999748d909e5df50e66c55c22b9baa.jpg)  
n<sub>c</sub>

![](images/aa43047d2899c4161bf32d70c756b8dc66379963695fd37a087cd8e327b23a3f.jpg)  
n<sub>c</sub>

![](images/18aebb0d53cb8e7a23fb1a79e4270552bce2a6c9465c535308362140ac7702b5.jpg)  
n<sub>c</sub>

![](images/6a91e74260cfefa9e45426ee8639bf3808e506f91e3b45538de5b20df4317eb3.jpg)  
n<sub>c</sub>

![](images/828415368a3aa604d471dc8e99590313f800ef588430a35bd6bccb6575afe78e.jpg)  
n<sub>c</sub>  
Fig. 5. Relationship between the number of components and predictive performance (mean ± std over 5 folds).

## 4.4 Efect of the Number of Gaussian Components

Figure 5 illustrates how predictive performance changes as the number of components varies. All other experimental settings remain consistent with the baseline evaluation. The results show that increasing the number of components generally improves predictive performance. This finding suggests that the number of components serves as an efective tunable parameter, enabling practitioners to balance accuracy against the efort of inspecting the components according to their specific needs.

## 5 Conclusion

We addressed the gap between the nominal interpretability of $\mathbf { G A } ^ { 2 } \mathbf { M s }$ and what their shape functions actually allow a practitioner to inspect. We proposed Bivariate Additive Models (BAMs), a class of GA<sup>2</sup>Ms in which every pairwise shape function is constrained to a simple parametric form, instantiated as an asymmetric bivariate Gaussian. BAMs select the feature pair of each component via a diferentiable selection mechanism, and each component is visualizable as a unimodal contour map. Experiments on six tabular benchmarks showed that BAM attains predictive accuracy comparable to standard GAMs and approaches that of $\mathbf { G A } ^ { 2 } \mathbf { M s }$ . We also showed the trade-of between accuracy and the number of components to inspect. BAM is positioned not as the highest-accuracy model but as an auditable additive model: it is well suited to tabular prediction tasks where each feature pair’s contribution must be transparently inspectable, such as industrial sensing applications in which engineers need to identify and share the dominant variable pairs driving outcomes. Future work includes exploring alternative simple forms for the shape function (e.g., monotonic functions), investigating identifiability and concurvity [20] among components, and clarifying the relationship between dataset characteristics and optimal model parameters such as the number of components.

Acknowledgments. This work was supported by JSPS KAKENHI Grant Number JP24K15081.

## References

1. Agarwal, R., Melnick, L., Frosst, N., Zhang, X., Lengerich, B., Caruana, R., Hinton, G.E.: Neural additive models: interpretable machine learning with neural nets. In: Advances in Neural Information Processing Systems. vol. 34 (2021)

2. Becker, B., Kohavi, R.: Adult. UCI Machine Learning Repository (1996), DOI: https://doi.org/10.24432/C5XW20

3. Bouchiat, K., Immer, A., Yèche, H., Rätsch, G., Fortuin, V.: Improving neural additive models with bayesian principles. In: Proceedings of the 41st International Conference on Machine Learning (ICML) (2024)

4. Caruana, R., Lou, Y., Gehrke, J., Koch, P., Sturm, M., Elhadad, N.: Intelligible models for healthcare: Predicting pneumonia risk and hospital 30-day readmission. In: Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 1721–1730. ACM (2015). https://doi.org/10.1145/2783258.2788613

5. Centers for Disease Control and Prevention (CDC): Diabetes Health Indicators Dataset. https: //www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset (2015), accessed: 2025-07-27

6. Chang, C.H., Caruana, R., Goldenberg, A.: NODE-GAM: Neural generalized additive model for interpretable deep learning. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id=g8NJR6fCCl8

7. Cortez, P., Cerdeira, A., Almeida, F., Matos, T., Reis, J.: Wine Quality. UCI Machine Learning Repository (2009), DOI: https://doi.org/10.24432/C56S3T

8. Cukierski, W.: Titanic - machine learning from disaster. https://kaggle.com/competitions/ titanic (2012), kaggle

9. Fanaee-T, H.: Bike Sharing. UCI Machine Learning Repository (2013), DOI: https://doi.org/10.24432/C5W894

10. Ibrahim, S., Afriat, G.I., Behdin, K., Mazumder, R.: Grand-slamin’ interpretable additive modeling with structural constraints. In: Advances in Neural Information Processing Systems. vol. 36 (2023)

11. Lou, Y., Caruana, R., Gehrke, J.: Intelligible models for classification and regression. In: Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 150–158. ACM (2012). https://doi.org/10.1145/2339530.2339556

12. Lou, Y., Caruana, R., Gehrke, J., Hooker, G.: Accurate intelligible models with pairwise interactions. In: Proceedings of the 19th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 623–631. ACM (2013). https://doi.org/10.1145/2487575. 2487579

13. Lundberg, S.M., Lee, S.I.: A unified approach to interpreting model predictions. In: Advances in Neural Information Processing Systems. vol. 30, pp. 4768–4777 (2017)

14. Pace, R.K., Barry, R.: Sparse spatial autoregressions. Statistics and Probability Letters 33, 291–297 (1997)

15. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., Duchesnay, E.: Scikit-learn: Machine learning in Python. Journal of Machine Learning Research 12, 2825–2830 (2011)

16. Peters, B., Niculae, V., Martins, A.F.T.: Sparse sequence-to-sequence models. In: Korhonen, A., Traum, D., Màrquez, L. (eds.) Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. pp. 1504–1519. Association for Computational Linguistics (2019). https://doi.org/10.18653/v1/P19-1146

17. Radenovic, F., Dubey, A., Mahajan, D.: Neural basis models for interpretability. In: Advances in Neural Information Processing Systems. vol. 35 (2022)

18. Ribeiro, M.T., Singh, S., Guestrin, C.: "why should i trust you?": Explaining the predictions of any classifier. In: Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. pp. 1135–1144. ACM (2016). https://doi.org/10. 1145/2939672.2939778

19. Rudin, C.: Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature Machine Intelligence 1(5), 206–215 (2019). https://doi.org/10.1038/s42256-019-0048-x

20. Siems, J., Ditschuneit, K., Ripken, W., Lindborg, A., Schambach, M., Otterbach, J.S., Genzel, M.: Curve your enthusiasm: concurvity regularization in diferentiable generalized additive models. In: Advances in Neural Information Processing Systems. vol. 36 (2023)

21. Slack, D., Hilgard, S., Jia, E., Singh, S., Lakkaraju, H.: Fooling LIME and SHAP: Adversarial attacks on post hoc explanation methods. In: Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society. pp. 180–186. AIES ’20, Association for Computing Machinery, New York, NY, USA (2020). https://doi.org/10.1145/3375627.3375830

22. Zhang, W., Barr, B., Paisley, J.: Gaussian process neural additive models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38. AAAI Press (2024). https://doi.org/ 10.1609/aaai.v38i15.29628