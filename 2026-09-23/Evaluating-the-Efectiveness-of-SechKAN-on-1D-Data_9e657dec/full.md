# Evaluating the Efectiveness of SechKAN on 1D Data

Hoang-Thang Ta<sup>1,2[0000−0003−0321−5106]</sup>

<sup>1</sup> University of Information Technology, Ho Chi Minh City, Vietnam <sup>2</sup> Vietnam National University Ho Chi Minh City, Ho Chi Minh City, Vietnam thangth@uit.edu.vn

Abstract. The connection between the Kolmogorov-Arnold representation theorem (KART) and neural network design has led to the development of Kolmogorov-Arnold Networks (KANs), with applications ranging from STEM problems to AI tasks. In this paper, we investigate the efectiveness of a KAN variant, SechKAN, which relies on hyperbolic secant (sech) functions as basis functions, with a 1D projection to reduce the number of parameters to a level comparable to MLPs. We evaluate SechKAN on three 1D classification datasets: UCI Human Activity Recognition (UCI HAR), ElectricDevices, and Crop, and compare it with several efective networks, including EficientKAN, MLP, CNN1D, ResNet1D, and DSCNN1D, using approximately comparable parameter budgets. The results indicate that SechKAN achieves competitive performance across the three datasets, with particularly strong performance on Crop. Ablation studies further show that grid size and normalization afect performance, suggesting that SechKAN’s efectiveness depends on the dataset and architectural choices. Our source code and experimental implementation are publicly available at: https://github.com/hoangthang

Keywords: Kolmogorov-Arnold Networks · SechKAN · Hyperbolic Secant Functions · One-Dimensional Data

## 1 Introduction

Neural networks have achieved significant success in developing models that learn from data and mimic aspects of human intelligence. Neural network architectures have evolved from early architectures such as MLPs to RNNs, CNNs, LSTMs, TCNs, and many new architectures, especially LLMs, and have been applied to a wide range of tasks. However, while MLPs are simple and widely used due to their efectiveness, their linear transformations cannot always adapt well to diferent types of data. The introduction of KANs has provided an alternative approach for many tasks involving functions and relationships in mathematics, physics, and other fields [20].

Continuing research on novel KAN architectures, we investigate SechKAN, a KAN that uses hyperbolic secant functions as basis functions. Unlike other KANs, SechKAN applies a 1D projection after feature extraction using the sech functions to reduce the number of parameters in its layers, making them comparable to those of MLP layers. In previous work, SechKAN showed its efectiveness on several tasks, including function approximation, PDE surrogate modeling, and image classification [24]. However, its performance on 1D data remains unexplored, despite its design for one-dimensional inputs.

An evaluation of SechKAN on 1D classification data is still lacking, particularly across diverse datasets and architectural settings. To address this gap, we evaluate SechKAN on three benchmark datasets: UCI HAR, ElectricDevices, and Crop. We compare it with EficientKAN, MLP, CNN1D, ResNet1D, and DSCNN1D, covering KAN-based, fully connected, and convolutional architectures. EficientKAN is included as a representative KAN baseline based on its competitive performance in our previous comparison of KAN variants [24], while the CNN-based models are selected for their ability to extract local patterns from one-dimensional data. We also conduct ablation studies on grid size (number of basis functions) and data normalization. The main contribution of this work is a systematic empirical evaluation of SechKAN across three 1D classification datasets, including comparisons with KAN-, MLP-, and CNN-based models, as well as ablation studies of key settings.

The remainder of this paper is organized as follows. Section 2 introduces related work to our SechKAN, including research directions related to KAN and KART. Section 3 introduces the SechKAN architecture and other KAN variants, as well as CNN-based models. Section 4 contains experimental information, including datasets, training configurations, evaluation metrics, experimental results, and ablation studies. Section 5 presents the limitations of the study. Finally, Section 6 concludes the paper and discusses future work.

## 2 Related Work

KANs were recently introduced by Liu et al. [17] as a new approach to neural network design. Unlike MLPs, KANs use learnable univariate functions instead of fixed activation functions. KANs are based on the KART, which states that a continuous multivariate function can be represented using a finite combination of one-dimensional functions [13, 5]. Before KANs, KART had already inspired several neural network architectures, including spline-based models [14, 6].

KANs have been applied to various tasks, including diferential equation solving [26], time series forecasting [8], and computer vision [15]. Many studies have also explored alternative basis functions to replace the original B-spline basis. These include modified B-splines [23], polynomial functions [22], radial basis functions [16], Fourier bases [27], wavelets [4], rational and fractional Jacobi functions [2, 1], and customized activation functions [19].

Another research direction focuses on reducing the large number of parameters in KANs. Yang and Wang [28] introduced Group KAN in Kolmogorov-Arnold Transformers (KATs), which reduces the number of parameters and computational costs through weight sharing. Ta et al. [25] proposed PRKAN, which uses several parameter-reduction techniques to achieve a model size comparable to MLPs. Inspired by PRKAN, SechKAN uses hyperbolic secant functions as basis functions and a 1D projection to reduce the number of parameters [24]. Other parameter-eficient KANs include GS-KAN, which generates edge functions from a shared learnable parent function [7], LeanKAN, which provides a compact alternative to AddKAN and MultKAN [12].

CNN-based architectures are widely used for data with local and sequential structures. For 1D data, CNN1D applies one-dimensional convolution to extract local features from input sequences [11]. ResNet1D extends this approach with residual connections, enabling deeper networks to learn efectively [9]. DSCNN1D uses depthwise separable convolutions to reduce computational cost and the number of parameters while preserving local feature extraction capabilities [21, 18]. These architectures provide established convolutional baselines for evaluating SechKAN on 1D classification tasks.

## 3 Methodology

## 3.1 KART and KAN

Kolmogorov–Arnold Networks (KANs) are based on the Kolmogorov–Arnold Representation Theorem (KART), which states that any continuous multivariate function on a bounded domain can be represented using a finite sum of onedimensional functions. Let $\mathbf { x } = ( x _ { 1 } , \ldots , x _ { n } ) \in [ 0 , 1 ] ^ { n }$ . Any continuous function $f : [ 0 , 1 ] ^ { n } \to { \mathbb R }$ can be represented as [10]:

$$
f (  { \mathbf { x } } ) = f ( x _ { 1 } , \dots , x _ { n } ) = \sum _ { q = 1 } ^ { 2 n + 1 } \varPhi _ { q } \left( \sum _ { p = 1 } ^ { n } \phi _ { q , p } ( x _ { p } ) \right)\tag{1}
$$

Here, $\phi _ { q , p }$ are one-dimensional functions applied to individual input variables, while $\varPhi _ { q }$ combines their outputs. This decomposition provides the theoretical basis for representing multivariate functions using one-dimensional functions.

Liu et al. [17] introduced KANs as a neural network architecture based on this theorem. Instead of using fixed activation functions and linear weights as in MLPs, KANs use learnable one-dimensional functions on the edges. A KAN with L layers applies a sequence of function matrices $\varPhi _ { 0 } , \varPhi _ { 1 } , \ldots , \varPhi _ { L - 1 } ;$

$$
\operatorname { K A N } ( \mathbf { x } ) = ( \varPhi _ { L - 1 } \circ \varPhi _ { L - 2 } \circ \cdot \cdot \cdot \circ \varPhi _ { 1 } \circ \varPhi _ { 0 } ) \mathbf { x }\tag{2}
$$

For a layer l, the function matrix $\varPhi _ { l } \in \mathbb { R } ^ { n _ { l + 1 } \times n _ { l } }$ contains learnable functions $\phi _ { l , j , i }$ connecting the i-th node of layer l to the j-th node of layer l + 1:

$$
\phi _ { l , j , i } , \quad l = 0 , \cdots , L - 1 , \quad i = 1 , \cdots , n _ { l } , \quad j = 1 , \cdots , n _ { l + 1 }\tag{3}
$$

With n<sub>l</sub> nodes in the $l ^ { t h }$ layer, the transformation from $\mathbf { x } _ { l }$ to $\mathbf x l + 1$ is computed by the function matrix $\varPhi _ { l } \in \mathbb { R } ^ { n _ { l + 1 } \times n _ { l } }$

$$
\begin{array} { r } { \mathbf { x } _ { l + 1 } = \underbrace { \left( \begin{array} { l l l l } { \phi _ { l , 1 , 1 } ( \cdot ) } & { \phi _ { l , 1 , 2 } ( \cdot ) } & { \cdot \cdot \cdot } & { \phi _ { l , 1 , n _ { l } } ( \cdot ) } \\ { \phi _ { l , 2 , 1 } ( \cdot ) } & { \phi _ { l , 2 , 2 } ( \cdot ) } & { \cdot \cdot } & { \phi _ { l , 2 , n _ { l } } ( \cdot ) } \\ { \qquad \vdots } & { \vdots } & { \ddots } & { \vdots } \\ { \phi _ { l , n _ { l + 1 } , 1 } ( \cdot ) \phi _ { l , n _ { l + 1 } , 2 } ( \cdot ) \cdot \cdot \cdot \phi _ { l , n _ { l + 1 } , n _ { l } } ( \cdot ) } \end{array} \right) } _ { \Phi _ { l } } \mathbf { x } _ { l } } \end{array}\tag{4}
$$

In the original KAN, each learnable function is constructed using a base function and a spline function [17]:

$$
\phi ( x ) = w _ { b } b ( x ) + w _ { s } s p l i n e ( x )\tag{5}
$$

where the base function is the SiLU function:

$$
b ( x ) = s i l u ( x ) = \frac { x } { 1 + e ^ { - x } }\tag{6}
$$

and the spline function is defined as:

$$
s p l i n e ( x ) = \sum _ { i } c _ { i } B _ { i } ( x )\tag{7}
$$

where $c _ { i }$ are learnable coeficients and $B _ { i } ( x )$ are B-spline basis functions.

## 3.2 SechKAN

SechKAN consists of multiple layers, where each layer processes the output of the previous layer, with sech functions serving as the main feature extractor and a grid projection acting as a 1D projection to reduce the number of parameters [24]. As illustrated in Figure 1, an input first passes through an optional Norm1, followed by sech basis functions that transform each input feature into G basis responses (the number of basis functions), introducing an additional grid dimension. The grid projection then aggregates the G basis responses into a single representation for each input feature. The resulting representation can pass through an optional second normalization position (Norm2), followed by a base activation and feature projection to produce the main output. In parallel, a skip projection branch processes the normalized input using the same base activation followed by a linear projection. Its output is then added element-wise to the main output to produce the final output. If the skip projection branch is disabled, the layer directly returns the main output.

The sech function is smooth, bell-shaped, exponentially decaying, and infinitely diferentiable, supporting smooth gradient propagation and localized representations. Its bounded output can help maintain stable activations, although saturation may lead to small gradients for inputs far from the center. These properties make sech functions suitable as basis functions for grid-based feature representations in SechKAN, while sharing characteristics with other localized basis functions used in KANs.

![](images/f5922fc5fd3333ddc52ccf8968470cf04ee2de90b541bd541ebe316ace95d1aa.jpg)  
Fig. 1. Architecture of a SechKAN layer, consisting of sech basis functions, grid projection, normalization, feature projection, and a parallel base-projection branch. The figure is adapted from the SechKAN work [24]. Note that the illustrated sech basis functions are simulated for visualization purposes; they asymptotically approach zero as the input moves away from the center but remain nonzero for any finite input.

For a SechKAN layer, the input X has shape $( B , D )$ , where B is the batch size and $D$ is the data dimension. For image inputs, $D = C \times W \times H$ , where $C ,$ $W$ , and H denote the number of channels, width, and height, respectively. We flatten X to obtain a tensor of shape $( B , d _ { \mathrm { i n } } )$ , where $d _ { \mathrm { i n } }$ is the input dimension. The output $Y$ has shape $( B , d _ { \mathrm { o u t } } )$ , where $d _ { \mathrm { o u t } }$ is the output dimension.

Suppose that $X ^ { \prime }$ denotes the input after applying Norm1. We then map $X ^ { \prime }$ to a set of sech basis functions. Let G denote the number of basis functions (grid size), and let $\mathcal { C } = \{ c _ { g } \} _ { g = 1 } ^ { G }$ denote their learnable center locations. The centers are initialized uniformly over the interval $[ g _ { \mathrm { m i n } } , g _ { \mathrm { m a x } } ]$ as

$$
c _ { g } = g _ { \operatorname* { m i n } } + \frac { g - 1 } { G - 1 } ( g _ { \operatorname* { m a x } } - g _ { \operatorname* { m i n } } ) , \quad g = 1 , \ldots , G .\tag{8}
$$

During training, the center locations are jointly optimized with the other model parameters, while the number of basis functions $G$ remains fixed.

The sech basis responses are defined as

$$
\phi ( X ^ { \prime } ) = s \cdot \frac { 1 } { \cosh \left( \frac { X ^ { \prime } - \mathscr { C } } { w } \right) } + \delta , \quad \varPhi ( X ^ { \prime } ) \in \mathbb { R } ^ { B \times d _ { \mathrm { i n } } \times G } ,\tag{9}
$$

where s and δ are learnable scale and bias parameters, respectively, and w is a learnable width parameter shared by all sech basis functions within a layer. The width is parameterized through an unconstrained variable $\theta _ { w }$

$$
w = \exp ( \theta _ { w } ) + 1 0 ^ { - 8 } ,\tag{10}
$$

which ensures a positive width and improves numerical stability during optimization.

## 4 Experiments

## 4.1 Datasets and Training Configurations

Three benchmark one-dimensional classification datasets were used: UCI Human Activity Recognition (UCI HAR), ElectricDevices, and Crop.

UCI HAR. The dataset contains 10,299 samples represented by 561-dimensional feature vectors across six human activities, with 7,352 training and 2,947 test samples in the original subject-independent split [3]. The test set was kept unchanged, while 20% of the training subjects were held out for validation using a fixed random seed of 42, ensuring subject-disjoint training and validation sets.

ElectricDevices. The dataset contains 16,637 univariate sequences of length 96 from seven classes, with 8,926 training and 7,711 test samples.<sup>3</sup> The test set was kept unchanged, while the training set was stratified by class into 80% training and 20% validation samples using a fixed random seed of 42. Crop. The dataset contains 24,000 sequences of length 46 from 24 classes, with 7,200 training and 16,800 test samples.<sup>4</sup> The test set was kept unchanged, while the training set was stratified by class into 80% training and 20% validation samples using a fixed random seed of 42.

All models were trained on an NVIDIA GeForce RTX 3060 Ti GPU using AdamW with a learning rate of $1 0 ^ { - 3 }$ , weight decay of $1 0 ^ { - 4 }$ , batch size of 64, and a OneCycleLR scheduler. The data splits were fixed using random seed 42, while model training was independently repeated five times using random seeds 0–4. The reported results are presented as mean ± standard deviation across five runs (using five seeds) in the main experiments and three runs (using three seeds) in the ablation studies.

The architectures were designed with approximately comparable numbers of trainable parameters for fair comparison (Table 1). The MLP used a single hidden layer of 256 units with LayerNorm and SiLU. CNN1D, ResNet1D, and DSCNN1D served as 1D convolutional baselines, using three convolutional layers, an initial convolution followed by three residual blocks, and six depthwiseseparable convolutional blocks, respectively. All three models used BatchNorm and SiLU, followed by global average pooling and a fully connected output layer. EficientKAN used 256 hidden units, a grid size of 5, and spline order 3.

SechKAN used a 256-unit hidden layer, with 4 basis functions (or grid size=4) and LayerNorm for UCI HAR and ElectricDevices, and 16 basis functions (or grid size = 16) with BatchNorm for Crop. The second normalization stage was disabled, while SiLU was used as the base activation and the base-function update was disabled. The learnable width was disabled for UCI HAR and Crop but enabled for ElectricDevices. Models were trained for 20 epochs on UCI HAR and Crop and 30 epochs on ElectricDevices. The exact network structures and trainable parameter counts are reported in Table 1.

## 4.2 Results

Table 1. Network structures and numbers of trainable parameters for the models on the three datasets.
<table><tr><td>Dataset</td><td>Model</td><td>Params</td><td>Network structure</td></tr><tr><td rowspan="5">UCI HAR</td><td>CNN1D</td><td>147,048</td><td>3-layer 1D CNN.</td></tr><tr><td>DSCNN1D</td><td>147,038</td><td>6 depthwise-separable convolutional blocks.</td></tr><tr><td>EfficientKAN</td><td>147,420</td><td>561–256–6; grid size = 5, spline order = 3.</td></tr><tr><td>MLP</td><td>147,048</td><td>561–256–6; LayerNorm and SiLU.</td></tr><tr><td>ResNet1D</td><td>147,048</td><td>Initial convolution + 3 residual blocks.</td></tr><tr><td rowspan="6">Electric Devices</td><td>SechKAN CNN1D</td><td>147,070 27,335</td><td>561–256–6; grid size = 4, LayerNorm, SiLU. 3-layer 1D CNN.</td></tr><tr><td>DSCNN1D</td><td>27,335</td><td>6 depthwise-separable convolutional blocks.</td></tr><tr><td>EfficientKAN</td><td>26,780</td><td>96–256–7; grid size = 5, spline order = 3.</td></tr><tr><td>MLP</td><td>27,335</td><td>96–256–7; LayerNorm and SiLU.</td></tr><tr><td>ResNet1D</td><td>27,335</td><td>Initial convolution + 3 residual blocks.</td></tr><tr><td>SechKAN</td><td>27,359</td><td>96–256–7; grid size = 4, LayerNorm, SiLU; learn- able width enabled.</td></tr><tr><td rowspan="6">Crop</td><td>CNN1D</td><td>18,804</td><td>3-layer 1D CNN.</td></tr><tr><td>DSCNN1D</td><td>18,804</td><td>6 depthwise-separable convolutional blocks.</td></tr><tr><td>EfficientKAN</td><td>18,200</td><td>46–256–24; grid size = 5, spline order = 3.</td></tr><tr><td>MLP</td><td>18,804</td><td>46–256–24; LayerNorm and SiLU.</td></tr><tr><td>ResNet1D</td><td>18,804</td><td></td></tr><tr><td>SechKAN</td><td>18,874</td><td>Initial convolution + 3 residual blocks. 46–256–24; grid size = 16, BatchNorm, SiLU.</td></tr></table>

Params = number of trainable parameters.

Table 2 shows that SechKAN achieves competitive performance on the three datasets. On UCI HAR, SechKAN performs close to the best KAN-based model and is also comparable to CNN1D, while outperforming ResNet1D and DSCNN1D. On ElectricDevices, SechKAN achieves competitive performance, although ResNet1D obtains the highest test accuracy. SechKAN nevertheless outperforms CNN1D and MLP and remains comparable to DSCNN1D. On Crop, SechKAN achieves the highest test accuracy among the evaluated models, outperforming both DSCNN1D and ResNet1D. These results indicate that SechKAN provides competitive classification performance across datasets with diferent input dimensions and numbers of classes, with particularly strong performance on Crop.

Table 2. Performance comparison of diferent models on UCI HAR, ElectricDevices, and Crop. Values are reported as mean ± standard deviation over five independent training runs using seeds 0–4.
<table><tr><td>Dataset</td><td>Model</td><td>Val. Acc.</td><td> $\mathbf { \sigma } _ { \mathbf { T e s t } \mathbf { A c c } . }$ </td><td>Time (s)</td></tr><tr><td rowspan="6">UCI HAR</td><td>CNN1D</td><td> $\overline { { 9 3 . 4 9 \pm 1 . 7 7 } }$ </td><td> $\overline { { 9 4 . 9 4 \pm 0 . 2 2 } }$ </td><td> $\overline { { 9 . 6 8 \pm 0 . 4 8 } }$ </td></tr><tr><td>DSCNN1D</td><td> $9 2 . 3 5 \pm 2 . 3 5$ </td><td> $9 4 . 0 3 \pm 1 . 0 8$ </td><td> $1 9 . 8 5 \pm 3 . 8 3$ </td></tr><tr><td>EfficientKAN</td><td> $9 2 . 9 8 \pm 2 . 3 6$ </td><td> $\mathbf { 9 4 . 9 7 \ : \pm { \ : 0 . 5 4 } }$ </td><td> $1 0 . 8 0 \pm 0 . 2 1$ </td></tr><tr><td>MLP</td><td> $9 2 . 9 8 \pm 2 . 4 6$ </td><td> $9 4 . 2 0 \pm 1 . 1 5$ </td><td> ${ \bf 6 . 9 3 \pm 0 . 8 0 }$ </td></tr><tr><td>ResNet1D</td><td> $9 2 . 7 3 \pm 2 . 8 1$ </td><td> $9 4 . 0 5 \pm 0 . 7 8$ </td><td> $1 4 . 7 2 \pm 1 . 2 5$ </td></tr><tr><td>SechKAN</td><td> $9 3 . 2 2 \pm 2 . 5 2$ </td><td> $9 4 . 8 4 \pm 0 . 8 6$ </td><td> $9 . 7 4 \pm 0 . 7 9$ </td></tr><tr><td rowspan="6">Electric Devices</td><td>CNN1D</td><td> $\overline { { 7 6 . 8 2 \pm 0 . 6 4 } }$ </td><td> $\overline { { 5 8 . 6 6 \pm 1 . 3 2 } }$ </td><td> $\overline { { 2 1 . 5 3 \pm 2 . 0 8 } }$ </td></tr><tr><td>DSCNN1D</td><td> $8 1 . 0 4 \pm 1 . 1 1$ </td><td> $6 2 . 4 3 \pm 1 . 1 0$ </td><td> $3 6 . 5 1 \pm 1 . 1 8$ </td></tr><tr><td>EfficientKAN</td><td> $7 5 . 0 0 \pm 0 . 4 1$ </td><td> $6 5 . 6 9 \pm 0 . 4 2$ </td><td> $\mathbf { 1 6 . 7 5 \ : \pm { \ : 1 . 6 4 } }$ </td></tr><tr><td>MLP</td><td> $7 3 . 7 1 \pm 0 . 2 7$ </td><td> $5 6 . 3 0 \pm 0 . 7 3$ </td><td> $1 5 . 0 2 \pm 0 . 2 7$ </td></tr><tr><td>ResNet1D</td><td> ${ \bf 8 7 . 6 0 \pm 0 . 2 5 }$ </td><td> ${ \bf 6 9 . 8 6 \pm 0 . 6 1 }$ </td><td> $2 5 . 8 9 \pm 0 . 9 3$ </td></tr><tr><td>SechKAN</td><td> $7 5 . 5 2 \pm 1 . 0 9$ </td><td> $6 2 . 7 5 \pm 0 . 8 5$ </td><td> $2 0 . 2 4 \pm 0 . 4 1$ </td></tr><tr><td rowspan="6">Crop</td><td>CNN1D</td><td> $\overline { { 7 1 . 3 9 \pm 0 . 5 8 } }$ </td><td> $\overline { { 7 2 . 3 0 \pm 0 . 2 3 } }$ </td><td> $\overline { { 1 1 . 2 3 \pm 1 . 8 8 } }$ </td></tr><tr><td>DSCNN1D</td><td> $7 2 . 3 1 \pm 1 . 0 0$ </td><td> $7 3 . 0 3 \pm 0 . 5 1$ </td><td> $2 0 . 6 1 \pm 4 . 1 0$ </td></tr><tr><td>EfficientKAN</td><td> $6 2 . 4 2 \pm 0 . 3 2$ </td><td> $6 2 . 8 2 \pm 0 . 0 9$ </td><td> $1 2 . 1 5 \pm 0 . 8 0$ </td></tr><tr><td>MLP</td><td> $6 8 . 7 8 \pm 0 . 3 1$ </td><td> $6 8 . 7 6 \pm 0 . 2 4$ </td><td> $\mathbf { 7 . 6 1 \pm 0 . 4 3 }$ </td></tr><tr><td> $\mathrm { R e s N e t 1 D }$ </td><td> $7 1 . 4 2 \pm 0 . 5 0$ </td><td> $7 2 . 4 8 \pm 0 . 5 2$ </td><td> $1 3 . 5 0 \pm 1 . 1 4$ </td></tr><tr><td> $\operatorname { S e c h K A N }$ </td><td> $\mathbf { 7 2 . 9 7 \ : \pm { \ : 0 . 6 2 } }$ </td><td> $\mathbf { 7 3 . 4 8 \ : \pm { \ : 0 . 4 1 } }$ </td><td> $9 . 8 5 \pm 0 . 2 4$ </td></tr></table>

![](images/9635e61b2a91ab2a3aaf250c84654ddc85da0b5b0ed26565536ba7e895ee7920.jpg)  
Fig. 2. Normalized test accuracy versus training time for the evaluated models on (a) UCI HAR, (b) ElectricDevices, and (c) Crop. Both test accuracy and training time are min–max normalized to [0, 1] independently for each dataset to illustrate the trade-of between predictive performance and training eficiency.

Figure 2 complements these results by illustrating the relationship between test accuracy and training time. On UCI HAR, SechKAN combines high test accuracy with relatively low training time, whereas on ElectricDevices it provides an intermediate accuracy–time trade-of. On Crop, SechKAN achieves both the highest test accuracy and relatively low training time, placing it in a favorable region of the accuracy–time plot. In general, SechKAN achieves competitive accuracy without incurring the substantially higher training times observed for some deeper convolutional architectures, although its relative advantage depends on the dataset.

## 4.3 Ablation studies

![](images/3cb7701d651aa913375b5d49b0ad51d3a5811c3bd32b701ef05079bbec8a3fda.jpg)  
Fig. 3. Ablation study of the number of basis functions in SechKAN using 2, 4, 8, 16, and 32 basis functions. All other configurations are kept identical to those used in the main experiments, with only the number of basis functions varied. The reported test accuracy values are averaged over three runs using seeds 0–2 and normalized to the range [0, 1] within each dataset, where 0 and 1 denote the worst and best results, respectively.

In this section, we perform ablation studies on the grid size and normalization types (LayerNorm and BatchNorm) for SechKAN on three datasets. The training configuration of SechKAN is kept the same as in the main experiments, with only the grid size and normalization configuration changed. The ablations were conducted after the main comparison and were not used to select the configurations reported in Table 1. The results in Figure 3 and Figure 4 show that SechKAN’s performance is afected by both the grid size and the normalization configuration.

![](images/7a85b56e40c422ea37e3a28a42072cd126abe98a41ff939814d3f33194b986b9.jpg)  
Fig. 4. Ablation study of normalization in SechKAN, comparing BatchNorm and LayerNorm applied at Norm1 and Norm2, as well as the configuration without normalization. All other configurations are kept identical to those used in the main experiments, with only the normalization type and position varied. The reported test accuracy values are averaged over three runs using seeds 0–2 and normalized to the range [0, 1] within each dataset, where 0 and 1 denote the worst and best results, respectively.

In Figure 3, the efect of grid size varies across datasets. The highest normalized test accuracy is obtained with grid sizes of 16, 4, and 4 for Crop, ElectricDevices, and UCI HAR, respectively. This shows that increasing the grid size does not always improve performance. In Figure 4, applying BatchNorm at Norm1 achieves the highest normalized test accuracy on all three datasets, while the configuration without normalization generally gives the lowest performance. LayerNorm at Norm1 also gives competitive results, whereas normalization at Norm2 gives less consistent results. These results show that grid size and normalization placement can afect SechKAN performance across diferent datasets.

However, these experiments cover only a limited set of configurations. More extensive experiments on SechKAN’s hyperparameters and architectural choices are needed to better understand their efects.

## 5 Limitations

This study has several limitations. First, SechKAN was evaluated on only three 1D classification datasets. Although these datasets difer in input dimensions, sample sizes, and numbers of classes, they do not cover a wide range of domains. Therefore, the results may not generalize to other types of data, such as 2D and 3D data. Second, we focused only on classification and did not evaluate SechKAN on other tasks, such as regression and forecasting. Third, we did not perform an extensive search of SechKAN hyperparameters, including the activation function, learnable width, and skip connection. Therefore, the selected grid sizes and normalization settings may not be optimal for all datasets. Finally, the comparison included only one MLP, one KAN baseline, and three CNN-based models. More experiments with diverse datasets, tasks, hyperparameter settings, and baseline models are needed to better understand the generalizability and effectiveness of SechKAN.

## 6 Conclusion

In this paper, we investigated the efectiveness of SechKAN for one-dimensional classification by combining hyperbolic secant functions with a 1D projection. SechKAN was evaluated against EficientKAN, MLP, CNN1D, ResNet1D, and DSCNN1D on three benchmark datasets: UCI HAR, ElectricDevices, and Crop, under approximately comparable parameter budgets. We also conducted ablation studies to examine the efects of grid size and normalization configuration. The results show that SechKAN achieves competitive performance across the three datasets, obtaining the highest test accuracy on Crop while maintaining relatively low training time. Its performance varies across datasets, indicating that the efectiveness of evaluated architectures depends on the characteristics of the one-dimensional classification task.

The ablation studies show that both grid size and normalization configuration afect SechKAN performance, with the best configurations varying across datasets. However, the evaluation is limited to three 1D classification datasets and a limited set of baseline models. Future work will extend the evaluation to a broader range of datasets and tasks, including regression and forecasting, and investigate a wider range of architectural designs and hyperparameter configurations.

## Acknowledgements

This research was funded by the University of Information Technology, Vietnam National University Ho Chi Minh City, under Grant No. S4-2027-80616.

## References

[1] Aghaei, A.A.: FKAN: Fractional Kolmogorov–Arnold Networks with Trainable Jacobi Basis Functions. Neurocomputing 623, 129414 (2025)

[2] Aghaei, A.A., Hosseinzadeh, M., Parand, K.: RKAN: Rational Kolmogorov-Arnold Networks. Neural Networks p. 108888 (2026)

[3] Anguita, D., Ghio, A., Oneto, L., Parra, X., Reyes-Ortiz, J.L., et al.: A Public Domain Dataset for Human Activity Recognition Using Smartphones. In: Esann, vol. 3, pp. 3–4 (2013)

[4] Bozorgasl, Z., Chen, H.: Wav-KAN: Wavelet Kolmogorov-Arnold Networks. arXiv preprint arXiv:2405.12832 (2024)

[5] Braun, J., Griebel, M.: On a Constructive Proof of Kolmogorov’s Superposition Theorem. Constructive Approximation 30, 653–675 (2009)

[6] van Deventer, H., van Rensburg, P.J., Bosman, A.: KASAM: Spline Additive Models for Function Approximation. arXiv preprint arXiv:2205.06376 (2022)

[7] Eliasson, O.: GS-KAN: Parameter-Eficient Kolmogorov-Arnold Networks via Sprecher-Type Shared Basis Functions. arXiv preprint arXiv:2512.09084 (2025)

[8] Genet, R., Inzirillo, H.: A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting. arXiv preprint arXiv:2406.02486 (2024)

[9] He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778 (2016)

[10] Ismailov, V.E.: Addressing Common Misinterpretations of KART and UAT in Neural Network Literature. Neural Networks p. 108361 (2025)

[11] Kiranyaz, S., Avci, O., Abdeljaber, O., Ince, T., Gabbouj, M., Inman, D.J.: 1D convolutional neural networks and applications: A survey. Mechanical systems and signal processing 151, 107398 (2021)

[12] Koenig, B.C., Kim, S., Deng, S.: LeanKAN: A Parameter-Lean Kolmogorov-Arnold Network Layer with Improved Memory Eficiency and Convergence Behavior. Neural Networks p. 107883 (2025)

[13] Kolmogorov, A.N.: On the Representation of Continuous Functions of Many Variables by Superposition of Continuous Functions of One Variable and Addition. In: Doklady Akademii Nauk, vol. 114, pp. 953–956, Russian Academy of Sciences (1957)

[14] Leni, P.E., Fougerolle, Y.D., Truchetet, F.: The Kolmogorov Spline Network for Image Processing. In: Image Processing: Concepts, Methodologies, Tools, and Applications, pp. 54–78, IGI Global (2013)

[15] Li, C., Liu, X., Li, W., Wang, C., Liu, H., Liu, Y., Chen, Z., Yuan, Y.: U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation. In: Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, pp. 4652–4660 (2025)

[16] Li, Z.: Kolmogorov-Arnold Networks are Radial Basis Function Networks. arXiv preprint arXiv:2405.06721 (2024)

[17] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljacic, M., Hou, T., Tegmark, M.: KAN: Kolmogorov–Arnold Networks. In: International Conference on Learning Representations, pp. 70367–70413 (2025)

[18] Majumdar, S., Ginsburg, B.: MatchboxNet: 1D Time-Channel Separable Convolutional Neural Network Architecture for Speech Commands Recognition. arXiv preprint arXiv:2004.08531 (2020)

[19] Qiu, Q., Zhu, T., Gong, H., Chen, L., Ning, H.: ReLU-KAN: New Kolmogorov-Arnold Networks That Only Need Matrix Addition, Dot Multiplication, and ReLU. In: 2025 IEEE Smart World Congress (SWC), pp. 1686–1694, IEEE (2025)

[20] Somvanshi, S., Javed, S.A., Islam, M.M., Pandit, D., Das, S.: A Survey on Kolmogorov-Arnold Networks. ACM Computing Surveys 58(2), 1–35 (2025)

[21] Sørensen, P.M., Epp, B., May, T.: A Depthwise Separable Convolutional Neural Network for Keyword Spotting on an Embedded System. EURASIP Journal on Audio, Speech, and Music Processing 2020(1), 10 (2020)

[22] SS, S.: Chebyshev Polynomial-Based Kolmogorov-Arnold Networks: An Efficient Architecture for Nonlinear Function Approximation. arXiv preprint arXiv:2405.07200 (2024)

[23] Ta, H.T.: BSRBF-KAN: A Combination of B-Splines and Radial Basis Functions in Kolmogorov-Arnold Networks. In: International Symposium on Information and Communication Technology, pp. 3–15, Springer (2024)

[24] Ta, H.T.: SechKAN: Kolmogorov–Arnold Networks with Hyperbolic Secant Functions. Neurocomputing 706, 135068 (2026), ISSN 0925-2312, https://doi.org/10.1016/j.neucom.2026.135068

[25] Ta, H.T., Thai, D.Q., Tran, A., Sidorov, G., Gelbukh, A.: PRKAN: Parameter-Reduced Kolmogorov-Arnold Networks. arXiv preprint arXiv:2501.07032 (2025)

[26] Wang, Y., Sun, J., Bai, J., Anitescu, C., Eshaghi, M.S., Zhuang, X., Rabczuk, T., Liu, Y.: Kolmogorov–Arnold-Informed Neural Network: A Physics-Informed Deep Learning Framework for Solving Forward and Inverse Problems Based on Kolmogorov–Arnold Networks. Computer Methods in Applied Mechanics and Engineering 433, 117518 (2025)

[27] Xu, J., Chen, Z., Li, J., Yang, S., Wang, W., Hu, X., Ngai, E.C.H.: FourierKAN-GCF: Fourier Kolmogorov-Arnold Network–An Efective and Eficient Feature Transformation for Graph Collaborative Filtering. arXiv preprint arXiv:2406.01034 (2024)

[28] Yang, X., Wang, X.: Kolmogorov-Arnold Transformer. In: International Conference on Learning Representations, pp. 76063–76086 (2025)