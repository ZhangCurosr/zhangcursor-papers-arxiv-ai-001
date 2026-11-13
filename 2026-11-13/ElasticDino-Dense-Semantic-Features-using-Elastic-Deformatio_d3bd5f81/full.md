# ElasticDino: Dense Semantic Features using Elastic Deformation

Ulysse Mizrahi<sup>∗</sup> Sean Zaretzky<sup>∗</sup> Dov Danon Amit H. Bermano

Tel Aviv University

https://odusseys.github.io/ElasticDino/

![](images/9051b58781ce6c59446f2a4b636136466c424e27f2c212eb843eba9cff713a78.jpg)  
Figure 1. ElasticDino upscales ViT features and corrects their ar-“table”tifacts, allowing to efficiently train a variety of dense vision tasks.

## Abstract

Vision transformers (ViTs) serve as the backbone for countless vision tasks, including dense prediction tasks, such as segmentation and depth estimation, even though the quadratic complexity ofself-attention severely limitsfeature resolution and induces spatial artifacts. In this work, we introduce ElasticDino, a lightweight adapter which produces high-resolutionfeature maps, adhering to thefine details of the input image. As a key insight, we observe the consistency of artifacts across channels, and simultaneously resolve them to match the input image structure. Using a slim architecture based on spatial deformations, ElasticDino produces pixel-aligned, high-resolution feature maps efficiently. Leveraging deformations allows strictly preserving sharp semantics, and thereby avoid the smoothness biases inherent to alternative approaches. Due to the absence of

ground truth, we employ pixel-level linear heads to predict a set ofdownstream dense tasks, guiding the deformation process without leaking task specific semantics into the model. We demonstrate superior accuracy over current methods on a range ofpopular dense task benchmarks, with better computational performance.

## 1. Introduction

Foundational Visual features have become the backbone for “glass”<sup>most vision tasks in recent literature. Under this paradigm,</sup> “chair”<sub>a large foundational model is typically contrastively trained</sub> to embed images into a latent space. This embedding is rich and semantic by nature, and serves as a great starting point for various downstream tasks with significantly less training requirements. Specifically, self-supervised features trained using a Vision Transformer (ViT) architecture (i.e., DINO [3, 19, 25]) are employed in state-of-the-Art semantic, panoptic and instance segmentation [5, 14], object detection [16], monocular depth estimation [2, 22, 34, 35], optical flow estimation [27], and more.

Despite their stellar success, these models have an inherent flaw: built on self-attention, they require quadratic memory, imposing patchification of the input image, effectively restricting the resolution of the final feature maps and introducing artifacts in their semantics. Indeed, for most hardware, a patched resolution of $3 2 \times 3 2$ or 64 × 64 is typically used, even when training dense tasks such as segmentation or depth estimation. Typically, a densification head is added to recover full resolution, (e.g. DPT [23]), which requires large amounts of data to recover fine details, as attested by the large and complex datasets used to train monocular depth estimation tasks [34, 35]. This approach implicitly also requires training a specific head for each downstream task due to poor transferability.

A different approach that has recently emerged in the literature [8, 11, 13, 28, 31] seeks to enhance the resolution of the produced features, indifferent the downstream task. As we demonstrate, such approaches still suffer from performance issues due to attention-based layers, and base their upscaling process on reconciling high and low resolution features through cropping-like mechanisms, leading to inevitable bleeding of feature semantics at boundaries.

In this paper, we introduce ElasticDino, a lightweight upscaling method to enhance DINO features, yielding efficient and sharp result through a deformation-based architecture and simple training through a collection of minimalistic heads.

To achieve this we first observe that artifacts are consistent across feature-channels, and can thus be resolved for all channels simultaneously (see Figure 2, subsection 3.1). As such, matching of feature and image structure our primary goal, regardless of specific per-chanel values. Second, we observe that most feature artifacts stem from the added positional encoding or under-sampling of semantic edges, resulting in thickening of objects and region bleeding. Hence, we learn deformation fields that align all feature channels to the image structure, alternating between upsampling and deformation. We observe that deformations offer efficient structure preservation, inherently and consistently adhering to sharp semantics. This alleviates otherwise inherent smoothness biases that are typical to the mechanisms used by previous work, and hence minimizes bleeding and fuzzy transition artifacts. Critically, since no upscaled ground truth exists in our setting, instead of cropping or further under-sampling approaches, we train ElasticDino using a set of five lightweight downstream heads, each learning to predict a different dense map, while limited to a 1x1 receptive field, forcing pixel-level matching of feature semantics.

Through careful evaluation, we demonstrate that ElasticDino produces high-quality alignment of semantics and structure, while removing artifacts such as those induced by positional encoding. Using lightweight heads, we report superior results compared to other upscaling methods on semantic segmentation, open-vocabulary segmentation, and depth estimation, with minimal computational effort.

## 2. Related work

## 2.1. Vision transformers and semantic features

Vision transformers have shown exceptional performance on vision tasks, being able to model complex long-range dependencies, and setting state-of-the-art results in a variety of vision tasks such as classification [37], segmentation [5, 14], object detection [16], monocular depth estimation, [2, 34, 35], joint text-image encoding [21], Vision Large Language Models [30], and image generation [20].

Many of the above tasks relied on pre-trained ViT encoders, much like vision models previously relied on convolutional backbones such as ResNet [12]. Indeed, ViTs have shown the ability to generate spatial semantic features, which transfer to a variety of tasks. Chief among these is the DINO family [3, 19, 25], which through unsupervised training provides exceptionally strong semantic features, and has since become the go-to backbone for various state-of-the-art models [16, 34].

## 2.2. Correction of ViT artifacts

In order to correct for artifacts in ViT (in particular DINO) activations, several methods have been introduced - though without considering higher feature resolutions.

Soon after the introduction of DINO, it was realized that a large part of the artifacts encountered were due to a leakage of global semantics in patch features [9]. ”Register” tokens were introduced and added to the tokenized image sequence, meant to capture global semantics, while letting patch features absorb relatively more local semantics.

Self-Patch [36] improves upon DINO’s self-supervision approach by enforcing local invariances among neighboring patches, improving spatial consistency and downstream performance although without upsampling features to higher resolutions. This however requires retraining of the model.

More recently, Denoising Vision Transformers [33] was introduced to mainly resolve the artifacts introduced by positional encodings. They achieve this through decomposing feature maps into a ”ground truth” component and several shared positional noise components. They impose consistency between the features produced by image crops, thus removing positional information from the image features. It is important to note that positional information in feature maps is nonetheless important; removing it entirely severely reduces the out-of-the-box ability to model positional dependencies between objects or points of the image.

## 2.3. ViTs and dense vision tasks

The first ViT densification approaches were task specific, with Dense Prediction Transformer (DPT) [23] still being a popular architecture. It combines a ViT backbone with a convolutional decoder to generate high-resolution depth maps, leading to success in tasks such as monocular depth estimation [22, 34, 35]. In addition, ViT-Adapter [4] adds a trainable module to a ViT, injecting vision-specific priors. DySample [17] dynamic sampling, SAPA [18] and ReSFU [40] further refine spatial semantics. While displaying impressive performance, these approaches struggle with high-frequency details, and more importantly demand dataintensive per-task training. Thus inducing high costs and restricting applicability to specialized domains.

More recently, a series of task agnostic literature has emerged, surpassing results of the task specific approaches while offering better generalization. FeatUp [11] introduces two separate upscaling methods, both based on reconciling features for various augmentations of an image. The first is trained per-image and demonstrates honorable performance, but requires over a minute of training for every image, precluding most real-world applications. The second method results in a generic upscaler, which displays gross artifacts - unavoidable with the joint-bilataralupsampling kernels employed.

Similarly, Lift [26] uses a simple CNN architecture which combines image and features and then upsamples them. The training procedure consists in matching upscaled features produced at a lower image resolution to features produced at higher image resolution, and not upscaled. As the chosen ”ground truth” is not free of artifacts, the incentive to fully resolve them is not complete. Further, the editing of feature values, which are mixed with image encodings, does not guarantee strict preservation of semantics.

LoftUp [13] passes images, features and positional encodings to cross-attention blocks. The training procedure is two-staged. First, they compute various segmentation masks on the image, use a form of bilateral upsampling guided by the masks, and use these as pseudo-labels. In a second stage, they perform teacher-student distillation, now using as objective the reconciliation of feature values along crops. This suffers the same issues, with direct feature value editing, and reduced incentive to correct artifacts, especially bleeding.

JAFAR [8] uses cross-attention using the encoded image as the query, features as value, and both as key, to compute pixel-to-semantics attention scores and query them within the feature values. This ensures that feature semantics are preserved. The training procedure is identical to LIFT and as such lacks the incentive to fully resolve artifacts.

AnyUp [31] proposes a backbone-agnostic upsampling module capable of transferring across models without retraining the upscaling head. To address the computational inefficiencies and the tendency of global attention to reference distant, unrelated image areas, AnyUp restricts its cross-attention computation to local spatial windows, supplemented by a feature-agnostic layer to enable transferability. However, the training procedure relies on matching features from randomly sampled image crops. Similar to FeatUp, this formulation does not provide a strong enough incentive to fully resolve the inherent spatial artifacts of the backbone.

UPLiFT [28] addresses the heavy memory bottlenecks of single-step cross-attention approaches by proposing an iterative upsampling framework. It relies on an encoderdecoder architecture equipped with a local attention pooling operator to maintain linear complexity with respect to the number of visual tokens. Similar to LIFT, the training procedure optimizes a multi-step reconstruction loss that matches iteratively upscaled features from lowresolution inputs to the standard features extracted from high-resolution inputs. By explicitly treating the raw backbone features as the ground truth target, this formulation inherits the backbone’s spatial artifacts and provides no incentive to fully resolve them.

## 3. Method

ElasticDino is an upscaling network for DINO features, learned as iterative upsampling followed by elastic deformations of the features. Given an image and the corresponding feature maps generated by the ViT, our goal is to upscale the feature maps while matching the image structure. Figure 3 depicts an overview of the architecture: The input features are repeatedly upscaled and deformed, with the deformation field being predicted using convolutional layers. We note that the advantages of this approach are lightweight computation that allow for higher-resolution maps at negligible cost with respect to backbone evaluation, and better handling of high-frequency edges. Another major advantage of this approach is that since we sample from the original feature map, we diminish hallucination hazards, ensuring strong structural and semantic preservation.

As there are no ground truth upscaled features available, we train 1 × 1 convolutional heads on a collection of dense downstream tasks, ensuring no spatial context and pixelwise feature and structure matching.

We emphasize that deformations strongly restrict the output, forbidding feature editing and preventing task leakage, or neglect of specific information correlating to the tasks used or not used during training. Indeed, when training without the depth and the segmentation tasks, performance remains strong on these held-out tasks. See supplementary materials for more detail. See Supplementary material for visualizations of deformations through upscaling process.

## 3.1. Cross-Channel Consistency

Due to their patch-wise decomposition of the image and combined use of positional encodings, ViTs typically introduce artifacts into the produced feature maps. We identify two main types of artifacts (Figure 2, bottom left). First, in regions of near-constant color, positional artifacts are prominent and values may vary widely, without semantic difference. It has been observed [9] that this happens in great part from global semantics being locally encoded, but positional encodings are also a significant contributor, as they are the only varying input to the ViT in a constant patch. Second is feature bleeding (Figure 2, bottom right): objects and boundaries are dilated due to discretization.

We make the crucial observation that ViTs introduce artifacts in a way which is consistent across channels (Figure 2, top). To substantiate this, we look at edge correlation between channels, and the original input image: we analyze 1,000 random ImageNet images. In each image, we isolate near-constant regions with an unsupervised localvariance filter, followed by erosion at the patch-grid scale so that measured gradients are not contaminated by neighboring edges; images without a sufficiently large flat region are discarded, leaving about 10% of images. This selection is purely geometric and independent of the correlations being measured. On the remaining patches, we compute Sobel gradients of the backbone feature map and input image, and report the mean cross-channel Spearman correlation of gradient magnitudes, alongside two baselines: the same crosschannel correlation over the full token grid, and the channelto-image correlation within the same masked region. In flat regions, cross-channel agreement is substantially higher than channel-to-image structure for both DINOv2 (0.29 vs. 0.11) and DINOv3 (0.30 vs. 0.12), and exceeds the fullimage cross-channel baseline (0.18 and 0.22), showing that channel-aligned artifacts are concentrated where image gradients are weak rather than being a generic property of the grid. As a control, at image edges the channel-to-image correlation rises and the gap closes (0.17 vs. 0.15 for DI-NOv2; 0.19 vs. 0.23 for DINOv3), as expected when feature gradients follow real boundaries. Gradient directions are likewise shared across channels in flat regions (mean $\cos ^ { 2 } \approx 0 . 7 3$ for both backbones). These findings indicate that one can learn a correction to all channels simultaneously, leading to very efficient resolution recovery.

## 3.2. Architecture

The architecture, depicted in Fig. 3 consists of alternating Feature Deformation layers, which correct artifacts and 2x bilinear upsamples. A first deformation layer is applied, then followed by an alternation of upsampling and deformations. The deformation process is guided by the input image, allowing to introduce the image structure at that resolution to guide the deformation of the features.

Each upscaling stage is composed of a series of Feature Deformation Blocks, which take as inputs the ”dirty” features at the current resolution and the image at the same resolution. These are projected to a common hidden dimension and concatenated; and then fed to a depthwise-separable [6] residual convolution network, which outputs a 2D deformation field. The field is applied to the input features to recover the denoised features.

In other terms, given features $f ,$ an image I at a current identical resolution, a single deformation block consists of two projections $p _ { f } , p _ { I }$ , a resnet encoder R which outputs a 2D deformation field

$$
\vec { F } = R ( p _ { f } ( f ) \oplus p _ { I } ( I ) )
$$

Which is applied coordinate-wise:

$$
\hat { f } _ { u , v } = f _ { \vec { F } _ { u , v } }
$$

The motivation for using depthwise separable convolutions is the desire to achieve an extremely lightweight architecture, given the fact that feature dimensions are already high, which at higher resolution leads to heavy memory and compute utilization. We observed no performance degradation as well as much faster training and inference.

Pixel-wise downstream tasks for semantic guidance Unfortunately, we have no ground truth maps in the upscaled resolution, and thus need to define an alternate training objective.

We make the observation that a well-upscaled and denoised feature map should be able, by design, to predict any dense task pixel-wise.

We hence introduce a multi-task learning approach, where we simultaneously train linear heads for downstream dense vision tasks (see Section 3.3). Crucially, the 1x1 receptive field prevents the heads from combining feature values from different spatial positions, forcing the predictions to only use the semantics at that exact pixel. Should a larger receptive field be employed, the incentive to match the semantics at the current pixel would be removed, and structure would be learnt both in the upscaler and the downstream head.

It is important to note that this approach can only work because we are employing deformations and sampling from the features themselves. Indeed, if the upsampling process were to directly edit feature values, the semantics of the downstream tasks could easily just become ”baked $\mathrm { i n } ^ { \prime \prime }$ to the feature maps, without any incentive to jointly edit all channels simultaneously. We show the effect of such an approach in an ablation study- Fig. 5.

For each downstream task (see below), we employ a simple linear head to predict the target. Furthermore, to encourage faster learning at every resolution, the same head is applied to every intermediate feature map, and the corresponding multi-resolution predictions are all used to compute the training objective.

## 3.3. Losses and Tasks

All the tasks we use predict a dense map on the full resolution. For every task except segmentation (for which we use the standard cross-entropy loss), the base loss is defined as $\mathcal { L } _ { \mathrm { b a s e } } ( x , \hat { x } ) = | | x - \hat { x } | | ^ { 2 } + \mathrm { S S I M } ( x , \hat { x } )$ , where SSIM denotes the structural similarity loss [29]. For all tasks except segmentation, and in order to encourage reconstruction of high frequency details, we additionally penalize discrepancies between the channel-wise spatial gradients and laplacians, following [2]: $\mathcal { L } _ { \mathrm { g r a d } } ( x , \hat { x } ) = \mathcal { L } _ { \mathrm { b a s e } } ( \nabla x , \nabla \hat { x } ) , \mathcal { L } _ { \mathrm { l a p } } ( x , \hat { x } ) =$ $\mathcal { L } _ { \mathrm { b a s e } } ( \Delta x , \Delta \hat { x } )$

To encourage faster reconstruction at every stage, we compute the loss for all intermediary feature maps, as well as at every resolution. Therefore, denoting $\hat { x } _ { i , j }$ the output of the i-th stage for the j-th prediction head, ${ \hat { x } } _ { j }$ the ground truth for task $j , r _ { k } = 3 2 \times 2 ^ { k }$ where $k \in [ [ 0 , 3 ] ]$ the resolutions used, and ${ \mathcal { L } } _ { j }$ the loss for task $j$ (see the task list below), we apply the following multi-scale loss:

$$
\mathcal { L } _ { \mathrm { m s } } ( x , \hat { x } ) = \sum _ { i , j , k } \mathcal { L } _ { j } ( \mathrm { i n t e r p o l a t e } ( x _ { i } , r _ { k } ) , \mathrm { i n t e r p o l a t e } ( \hat { x } y _ { i , j } , r _ { k } ) )
$$

![](images/7f8e463c030574a4cf2d35e5a24cbf5cd964c06bd97c6b62661d2e86fdc62e0f.jpg)  
Figure 2. Visualizations of features values across different channels. We observe artifacts are consistent across channels (top row). In addition, low frequency regions of the image emphasize positional artifacts (bottom left), and the low resolution nature of the features typically induce thickening of object edges (bottom right).

This allows gradients to flow at each stage, rather than through the entire upscaling network, greatly stabilizing and speeding up the learning.

The downstream tasks we use represent a diverse sets of maps to be predicted from the image. We find each to help improve overall embedding quality (see Section 4), and believe more can be added. We use the following downstream tasks: RGB reconstruction, monocular depth, semantic segmentation, surface normals, intrinsic decomposition. Training is performed on two interspersed datasets: Hypersim [24] and Imagenet-22k [10]. See Supplementary material for details on losses and the training procedure.

## 4. Experiments

We perform quantitative and qualitative evaluation on a prototype designed to upscale DINOv2 features, and report the results below. We report semantic segmentation and depth estimation comparisons, and open vocabulary segmentation in the supplementary material. Following recent publications [31], we also perform some of the same experiments for DINOv3 [25] features. For implementation details, including results from our training head tasks, see supplementary material.

Qualitative Results are presented in Figure 4. Along side the input image, we visualize the top-3 PCA components of DINOv2 features, and resulting upscaling maps. As can be seen, artifacts are corrected and crisp semantic edges are formed, corresponding to the original image structure. ElasticDino alleviates artifacts in low variance regions of the image, and demonstrates semantic results that are decoupled from RGB variations in the image. For example, The similarly colored faucet is represented with distinctly different features compared to the background, while the flag is represented similarly throughout even though it consists of differently colored texture. For a more comprehensive demonstration of qualitative results, please refer to the supplementary material. We also note that while positional encoding related artifacts are alleviated, the positional information itself persists, and is actually extracted from the features with more ease compared to competing upscaling methods. See supplementary material for details on this experiment.

## 4.1. Quantitative Comparisons

## 4.1.1. Task Specific (Semantic segmentation)

We train simple convolution heads for semantic segmentation on top of ElasticDino and compare performance with two leading dedicated segmentation methods, Mask2Former [5] and OneFormer [14]. The data sets examined are MS-COCO [15], ADE20k [38, 39] and Cityscapes [7] (Cscapes), all standard data sets in semantic segmentation benchmarks.

ElasticDino displays higher performance as measured by Mean-IoU (Intersection over Union) on all three validation datasets, with a very large margin on ADE20k and Cityscapes. Qualitatively (In Supplementary material) we observe not only better classification ability, but much finer segmentation boundaries. This implies obvious advantages to downstream applications where sensitivity is key, such as autonomous driving.

## 4.1.2. Task Agnostic (Segmentation and Depth)

As our primary comparison, we further compare to taskagnostic methods that employ DINO feature upsampling (Table 3 for DINOv2, and Table 4 for available methods with DINOv3). These include training free Nearest neighbor and bilinear upsampling, and recent state-of-theart methods [8, 11, 13, 28, 31]. We follow established evaluation protocols, as described in by JAFAR [8], measuring mean Intersection over Union (mIoU) and pixel-level accuracy (Acc) for the semantic segmentation task over four datasets, and Root Mean Square Error (RMSE) and $\delta _ { 1 }$ (measuring the percentage of pixels where that are less than 25% of the ground-truth) for the depth estimation task over one dataset. In the supplementary material, we also compare performance for zero-shot open vocabulary segmentation, and the CLIP [21] backbone. As can be seen, ElasticDino demonstrates leading performance consistently for all datasets except Cityscapes. We interpret this surprising discrepancy to the lower ability for our method to capture small objects, possibly due to the low-frequency bias of the convolutional architecture. To support this class size bias, we provide in the supplementary material a comparison of model performance by object instance size, demonstrating that while ElasticDino vastly outperforms other methods on larger classes, it struggles with smaller ones, indicating both better semantic preservation but also imperfect reconstruction of small objects. Notably, these improved results are produced with less computational efforts and memory footprint, as reported in Table 1.

![](images/96601d9a2fb9ee13151be48fff39367eb3873289226caa7f6d967b1bae068f9e.jpg)  
Figure 3. Method overview. The input image is passed to the (DINO) backbone for the low resolution, artifact-full features. These are successively passed to a sequence feature-fixing layers and then bilinearly upsampled by a factor 2. Feature Deformation Layers project and concatenate both the image and the semantic features, which are fed to a series of depthwise residual blocks, that end up generating a deformation field that warps the feature map. Task heads induce the sole losses of the system. The are simple MLPs that operate on the individual pixel level. Each head is compared to the ground truth using a task-specific loss.

Table 1. Forward pass performance measures of ElasticDino and competing feature upscaling works over DINOv2. Note that even though ElasticDino trains more parameters, it is faster and more memory efficient, due to the linear (as opposed to attention-based) nature of our operations
<table><tr><td></td><td>ED (ours)</td><td>AnyUp</td><td>LoftUp</td><td>JAFAR</td><td>UPLift</td></tr><tr><td>Avg Time (ms)</td><td>108.41</td><td>545.42</td><td>418.89</td><td>177.79</td><td>147.27</td></tr><tr><td>Peak mem (GB)</td><td>0.71</td><td>1.94</td><td>8.31</td><td>7.33</td><td>9.22</td></tr><tr><td>#Params (K)</td><td>27,095</td><td>878</td><td>4,303</td><td>628</td><td>789</td></tr></table>

Table 2. Validation Mean IoU for semantic segmentation benchmarks. Note that results differ from the comparison benchmarks, as we use a convolutional head (rather than linear) and train for longer than specified in the JAFAR benchmark.
<table><tr><td>Model</td><td>Cscapes</td><td>COCO</td><td>ADE20k</td></tr><tr><td>Oneformer-sw-L</td><td>0.6412</td><td>0.6264</td><td>0.5109</td></tr><tr><td>Mask2Former-sw-L</td><td>0.5986</td><td>0.6239</td><td>0.4928</td></tr><tr><td>ElasticDino-L</td><td>0.6875</td><td>0.6270</td><td>0.5299</td></tr></table>

## 4.2. Ablation studies

We perform an ablation study on the architecture and design choices. In order to ensure a fair benchmark, we train ElasticDino and its ablations on the Hypersim dataset and omit the depth task, reserving it for evaluation. The training tasks are RGB, illumination, reflectance, residuals and normals [24].

First, we ablate the deformations, employing convolutions only. Second, we ablate tasks in a leave-one-out fashion. Finally, we ablate by progressively omitting tasks.

For each trained upscaler, we train a UNet head to predict the unseen depth, reserving a fraction as a test set (which is also held out from upscaler training), with SiLogLoss as the depth metric. Quantitative results are summarized in table 5, and some qualitative results are shown in figure 5.

Table 3. Comparison of various upsampling methods. All benchmarks are run using the method and code used for the JAFAR paper. Bold = best, underline = second best.
<table><tr><td></td><td colspan="8">Semantic Segmentation</td><td colspan="2">Depth Est.</td></tr><tr><td></td><td colspan="2">COCO</td><td colspan="2">VOC</td><td colspan="2">ADE20K</td><td colspan="2">Cityscapes</td><td colspan="2">COCO</td></tr><tr><td>Method</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td> $\delta _ { 1 }$  ↑</td><td>RMSE↓</td></tr><tr><td>Nearest</td><td>56.17</td><td>76.97</td><td>76.41</td><td>93.80</td><td>37.27</td><td>71.91</td><td>54.05</td><td>90.36</td><td>58.08</td><td>0.70</td></tr><tr><td>Bilinear</td><td>59.03</td><td>79.07</td><td>80.70</td><td>95.17</td><td>39.23</td><td>73.69</td><td>59.37</td><td>92.47</td><td>59.92</td><td>0.66</td></tr><tr><td>FeatUp [11]</td><td>60.10</td><td>79.95</td><td>81.08</td><td>95.32</td><td>38.82</td><td>73.74</td><td>56.06</td><td>91.86</td><td>61.69</td><td>0.64</td></tr><tr><td>LoftUp [13]</td><td>61.62</td><td>81.11</td><td>84.18</td><td>96.22</td><td>41.51</td><td>75.41</td><td>62.13</td><td>93.53</td><td>58.01</td><td>0.68</td></tr><tr><td>JAFAR [8]</td><td>60.32</td><td>80.30</td><td>84.19</td><td>96.18</td><td>40.16</td><td>74.39</td><td>61.58</td><td>93.45</td><td>60.05</td><td>0.65</td></tr><tr><td>AnyUp [31]</td><td>60.68</td><td>80.59</td><td>83.81</td><td>96.13</td><td>40.39</td><td>74.76</td><td>60.99</td><td>93.37</td><td>60.65</td><td>0.64</td></tr><tr><td>UPLift [28]</td><td>61.10</td><td>80.85</td><td>84.82</td><td>96.41</td><td>41.29</td><td>75.05</td><td>65.02</td><td>94.34</td><td>60.55</td><td>0.64</td></tr><tr><td>ED (ours)</td><td>62.72</td><td>81.74</td><td>86.00</td><td>96.78</td><td>42.10</td><td>76.00</td><td>61.33</td><td>93.44</td><td>64.14</td><td>0.57</td></tr></table>

Table 4. Comparison of models over DINOv3-small backbone. Benchmarks are run using the method and code used for the JAFAR [8] paper. Bold = best, underline = second best.
<table><tr><td rowspan="3">DinoV3</td><td colspan="8">Semantic Segmentation</td><td colspan="3">Depth Est.</td></tr><tr><td colspan="2">COCO</td><td colspan="2">VOC</td><td colspan="2">ADE20K</td><td colspan="2">Cityscapes</td><td colspan="2">COCO</td></tr><tr><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td> $\delta _ { 1 } ( \uparrow )$ </td><td>RMSE(↓)</td></tr><tr><td>JAFAR [8]</td><td>62.2</td><td>81.38</td><td>82.9</td><td>95.96</td><td>42.31</td><td>75.7</td><td>57.25</td><td>92.3</td><td>58.2</td><td>0.68</td></tr><tr><td>AnyUp [31]</td><td>62.70</td><td>81.69</td><td>83.88</td><td>96.21</td><td>42.69</td><td>76.02</td><td>58.32</td><td>92.81</td><td>59.43</td><td>0.66</td></tr><tr><td>UPLift [28]</td><td>63.52</td><td>82.14</td><td>84.86</td><td>96.58</td><td>44.12</td><td>76.62</td><td>62.31</td><td>93.82</td><td>59.29</td><td>0.66</td></tr><tr><td>ED (ours)</td><td>64.04</td><td>82.43</td><td>86.26</td><td>96.89</td><td>44.57</td><td>77.03</td><td>62.91</td><td>94.03</td><td>61.10</td><td>0.62</td></tr></table>

## 5. Discussion

The ablation of deformations results in almost imperceptible differences in feature values: because of the lack of incentive for the network to jointly modify features, task semantics are leaked and baked into the feature values. Deformation-based upsampling preserves feature semantics, allowing only local feature resampling and alleviating catastrophic forgetting. When deformations are ablated, the upsampler can directly alter feature values and may encode only the supervised task signal, e.g. depth, rather than preserving semantic structure. However, because some tasks exhibit correlation with depth prediction (e.g. surface normals), and because residual connections let the features flow through, some ability to transfer is conserved. It can be seen that the model gets to good performance with low amount of tasks but still improves with additional tasks. When ablating each task individually, we see that every task leads to some amount of improvement in performance, showing that adding more tasks improves the performance.

## 5.1. Limitations

While demonstrating promising performance, ElasticDino struggles with distinguishing small holes in the source image that are smaller than patch size

![](images/b45a586ce5854f50cfe16803c48d611e3262f38ae53991b446630a9ff551032c.jpg)

(see inset figure). Our intuition is that the convolutional layers still display low frequency biases especially at large receptive fields, even though leakage is mitigated due to the deformation. Further experimenting with alternative architectures with lower frequency bias may be rewarding.

## 5.2. Future directions

Beyond solving the above-mentioned issue, a clear future direction is the extension of ElasticDino to other modalities, such as video or 3D. Video could be handled through videospecific feature generators such as VJEPA-2 [1]. 3D could be handled through multi-view projection of features from renderings, as done in TRELLIS [32].

![](images/3d2fa4e5c35e11584d80555ecc714deb930ad9bc6c6cff23ed7e726c00189ab9.jpg)

Figure 4. Qualitative evaluation. Left: The top three PCA components of the feature maps of input images are depicted, as generated originally by DINOv2 (Low-Res), previous approaches, and us. Top Right: Semantic segmentation maps of input images, as generated by humans (GT), three previous methods, and us are also depicted. Bottom right: depth maps for the same approaches and input setting are depicted as well. Throughout all results, ElasticDino depicts sharper edges that respect the input image resolution and mitigates semantic bleeding between regions, even when RGB colors are similar.  
![](images/ed71d647cb278f0906a6b4116a67b3588591e1982b3d52a887476e1b6abb3d4a.jpg)

Table 5. Depth-estimation ablation test loss  
Figure 5. Ablating deformations weakens the preservation of original features by allowing the encoding of the task specific values in the output, instead of only blending them. From left to right: input image, low-resolution DINOv2 features, the result without deformations.
<table><tr><td>Ablation</td><td>δ1(%) ↑</td><td>∆δ1</td><td>δ3(%) ↑</td><td>AbsRel↓</td><td>SqRel↓</td><td>RMSE↓</td><td>△RMSE</td><td>log10 ↓</td><td>SILog↓</td></tr><tr><td>w/o DEFormations</td><td>26.80</td><td>-31.55</td><td>70.11</td><td>1.024</td><td>1.19</td><td>1.32</td><td>+0.718</td><td>0.238</td><td>66.30</td></tr><tr><td>w/o ILLumination</td><td>57.21</td><td>-1.14</td><td>92.83</td><td>0.428</td><td>0.278</td><td>0.620</td><td>+0.018</td><td>0.117</td><td>33.06</td></tr><tr><td>w/o REFlectance</td><td>58.43</td><td>+0.08</td><td>93.10</td><td>0.419</td><td>0.268</td><td>0.603</td><td>+0.001</td><td>0.115</td><td>32.45</td></tr><tr><td>w/o RESiduals</td><td>57.74</td><td>-0.61</td><td>93.02</td><td>0.423</td><td>0.273</td><td>0.611</td><td>+0.009</td><td>0.116</td><td>32.73</td></tr><tr><td>w/o Normals</td><td>57.81</td><td>-0.54</td><td>92.95</td><td>0.424</td><td>0.274</td><td>0.611</td><td>+0.009</td><td>0.116</td><td>32.89</td></tr><tr><td>w/o RGB</td><td>58.22</td><td>-0.13</td><td>93.13</td><td>0.418</td><td>0.266</td><td>0.603</td><td>+0.001</td><td>0.115</td><td>32.47</td></tr><tr><td>w/o RGB,ILL</td><td>57.20</td><td>-1.15</td><td>92.81</td><td>0.425</td><td>0.273</td><td>0.617</td><td>+0.015</td><td>0.117</td><td>33.24</td></tr><tr><td>w/o RGB,ILL,REF</td><td>57.17</td><td>-1.18</td><td>92.73</td><td>0.432</td><td>0.285</td><td>0.622</td><td>+0.020</td><td>0.118</td><td>33.54</td></tr><tr><td>w/o RGB,ILL,REF,RES</td><td>56.29</td><td>-2.06</td><td>92.50</td><td>0.437</td><td>0.292</td><td>0.634</td><td>+0.032</td><td>0.120</td><td>33.99</td></tr><tr><td>full model</td><td>58.35</td><td></td><td>93.17</td><td>0.417</td><td>0.265</td><td>0.602</td><td></td><td>0.115</td><td>32.46</td></tr></table>

Bold denotes the best result and Underline the second-best result. Rankings are computed from the unrounded values; consequently, identically displayed rounded values may have different ranking markers.

## 5.3. Conclusion

We have introduced ElasticDino, a solution for producing high resolution semantic features on top of vision transformers. We demonstrated ElasticDino achieves superior performance compared to existing methods, with lower memory usage and faster runtimes.

ElasticDino, like the vision transformers it enhances, does not produce content that can be directly misused. As a foundational model, any broader impact, positive or negative, is likely to stem from improved performance in downstream tasks or the emergence of new applications.

We refer to the supplementary material for additional implementation details and qualitative results.

## References

[1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba, Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. 7

[2] Aleksei Bochkovskii, Amael Delaunoy, Hugo Germain,¨ Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second, 2025. 1, 2, 4

[3] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J´ egou,´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 1, 2

[4] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions, 2023. 2

[5] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. arXiv, 2021. 1, 2, 5

[6] Franc¸ois Chollet. Xception: Deep learning with depthwise separable convolutions, 2017. 4

[7] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding, 2016. 5

[8] Paul Couairon, Loick Chambon, Louis Serrano, Jean-Emmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Jafar: Jack up any feature at any resolution, 2025. 1, 3, 5, 7, 12, 13

[9] Timothee Darcet, Maxime Oquab, Julien Mairal, and Piotr´ Bojanowski. Vision transformers need registers, 2024. 2, 3

[10] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5

[11] Stephanie Fu, Mark Hamilton, Laura E. Brandt, Axel Feldmann, Zhoutong Zhang, and William T. Freeman. Featup: A model-agnostic framework for features at any resolution. In The Twelfth International Conference on Learning Representations, 2024. 1, 2, 5, 7

[12] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 2

[13] Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, and Dan Zhang. Loftup: Learning a coordinatebased feature upsampler for vision foundation models. arXiv preprint arXiv:2504.14032, 2025. 1, 3, 5, 7

[14] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation, 2022. 1, 2, 5

[15] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft´ coco: Common objects in context, 2015. 5

[16] Shilong Liu, Zhaoyang Zeng, Tianhe Ren, Feng Li, Hao Zhang, Jie Yang, Qing Jiang, Chunyuan Li, Jianwei Yang, Hang Su, Jun Zhu, and Lei Zhang. Grounding dino: Marry ing dino with grounded pre-training for open-set object de tection, 2024. 1, 2

[17] Wenze Liu, Hao Lu, Hongtao Fu, and Zhiguo Cao. Learning to upsample by learning to sample. In Proceedings of the IEEE/CVF international conference on computer vision, pages 6027–6037, 2023. 2

[18] Hao Lu, Wenze Liu, Zixuan Ye, Hongtao Fu, Yuliang Liu, and Zhiguo Cao. Sapa: Similarity-aware point affiliation for feature upsampling. Advances in Neural Information Processing Systems, 35:20889–20901, 2022. 2

[19] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1, 2

[20] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. 2

[21] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision, 2021. 2, 6

[22] Rene Ranftl, Katrin Lasinger, David Hafner, Konrad´ Schindler, and Vladlen Koltun. Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer, 2020. 1, 2

[23] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vi-´ sion transformers for dense prediction, 2021. 1, 2

[24] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic syn thetic dataset for holistic indoor scene understanding, 2021. 5, 6

[25] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3, 2025. 1, 2, 5

[26] Saksham Suri, Matthew Walmer, Kamal Gupta, and Abhinav Shrivastava. Lift: A surprisingly simple lightweight feature transform for dense vit descriptors, 2024. 3

[27] Narek Tumanyan, Assaf Singer, Shai Bagon, and Tali Dekel. Dino-tracker: Taming dino for self-supervised point tracking in a single video, 2024. 1

[28] Matthew Walmer, Saksham Suri, Anirud Aggarwal, and Abhinav Shrivastava. Uplift: Efficient pixel-dense feature upsampling with local attenders. arXiv preprint arXiv:2601.17950, 2026. 1, 3, 5, 7

[29] Zhou Wang, Alan Conrad Bovik, Hamid Rahim Sheikh, and Eero P. Simoncelli. Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004. 4

[30] Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao. Simvlm: Simple visual language model pretraining with weak supervision, 2022. 2

[31] Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric Lenssen. Anyup: Universal feature upsampling. arXiv preprint arXiv:2510.12764, 2025. 1, 3, 5, 7, 12

[32] Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3d latents for scalable and versatile 3d generation, 2025. 8

[33] Jiawei Yang, Katie Z Luo, Jiefeng Li, Congyue Deng, Leonidas Guibas, Dilip Krishnan, Kilian Q Weinberger, Yonglong Tian, and Yue Wang. Denoising vision transformers, 2024. 2

[34] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data, 2024. 1, 2

[35] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2, 2024. 1, 2

[36] Sukmin Yun, Hankook Lee, Jaehyung Kim, and Jinwoo Shin. Patch-level representation learning for self-supervised vision transformers, 2022. 2

[37] Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers, 2022. 2

[38] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. 5

[39] Bolei Zhou, Hang Zhao, Xavier Puig, Tete Xiao, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Semantic understanding of scenes through ade20k dataset. International Journal of Computer Vision (IJCV), 2018. 5

[40] Minghao Zhou, Hong Wang, Yefeng Zheng, and Deyu Meng. A refreshed similarity-based upsampler for direct high-ratio feature upsampling. IEEE Transactions on Neural Networks and Learning Systems, 2025. 2

# ElasticDino: Dense Semantic Features using Elastic Deformation - Supplementary Material

## A. Implementation Details

We train our model using batches drawn alternately from two datasets: Hypersim [53], which provides ground truth for intrinsic decomposition and normal estimation, and ImageNet-22k [46] for increased image diversity.

To avoid introducing additional datasets for semantic segmentation and depth prediction, we utilize pseudolabels generated by pre-trained models, specifically, Mask2Former (COCO,ViT-Large) [42] and DepthAnything [57] respectively. These labels are computed for images across both of the training datasets.

To minimize parameter count and overhead on top of the base ViT prediction, we use depth-wise separable convolutions [43]. We observe no performance hit compared to regular convolutions, with significantly faster training and inference, and vastly reduced memory usage.

We train in mixed-precision using an 8-bit version of the AdamW optimizer [51], with parameters $\epsilon \ : = \ : 1 0 ^ { - 5 }$ for stability, an effective batch size of 16 and an exponential learning rate decay starting at $1 0 ^ { - 3 }$ . The final model was trained on a single A100 GPU for 24 hours, amounting to 3.42kg of CO2-equivalent emissions, fully offset by the cloud provider (Google Cloud Services), and computed using the MachineLearning Impact calculator presented in [49].

## B. Further Quantitative Results

## B.1. Open Vocabulary Evaluation

We evaluate our model on zero-Shot open-vocabulary segmentation following the method presented in MaskCLIP [58]. This training-free approach computes pixel-wise predictions by taking the argmax over similarity scores between visual features and text embeddings generated from dataset labels in textual templates. Following JAFAR [45], we evaluate on CLIP-ViT-B/16 and observe competitive results, demonstrating that ElasticDino preserves the original backbone features through its upsampling. Results against competitors that are compatible with CLIP-ViT-B/16 (JA-FAR [45], AnyUp [56], LoftUp [47]) are reported in Table 1.

Table 1. Zero-Shot Open-Vocabulary segmentation on VOC and Ade20k datasets following the methodology used in JAFAR [45] and introduced in MaskCLIP [58] paper. Competitive results indicating good alignment with original internal CLIP features.
<table><tr><td rowspan="2">Method</td><td>VOC</td><td>ADE20K</td></tr><tr><td>mIoU(↑) Acc(↑)</td><td>mIoU(↑) Acc(↑)</td></tr><tr><td>Bilinear</td><td>51.25 68.18</td><td>9.05 25.15</td></tr><tr><td>JAFAR</td><td>63.99 77.91</td><td>10.69 28.06</td></tr><tr><td>AnyUp</td><td>57.59 73.40</td><td>10.71 28.95</td></tr><tr><td>LoftUp</td><td>38.73 55.29</td><td>7.64 32.16</td></tr><tr><td>ElasticDino (ours)</td><td>58.43 73.89</td><td>10.75 28.28</td></tr></table>

## B.2. Positional Information Preservation

While positional encoding related artifacts are alleviated, positional information still persists and can be extracted more easily from the features compared to competing upscaling methods. We train a simple pixel-wise head to predict normalized $( x , y )$ coordinates from upsampled features. Bilinear upsampling performs best, as expected for a linear target, but ElasticDino indeed preserves positional information. See accuracy in $[ 1 0 ^ { - 3 } \mathrm { M S E } ]$ results below.

<table><tr><td colspan="4">JAFAR FeatUp AnyUp LoftUp Bilinear Ours</td></tr><tr><td>10−³MSE</td><td>5.3 11.5</td><td>4.2 64.0 3.2</td><td>4.5</td></tr></table>

## B.3. Semantic Segmentation and Depth Estimation over CLIP

Following the comparison against other task agnostic upsampling methods (JAFAR [45], AnyUp [56], LoftUp [47], UPLift [55]) over DINOv2 [52] and DINOv3 [54], we also compare recent works over CLIP [58] and get competitive results (Table 2). We follow established evaluation protocols, as described by JAFAR [45], measuring mean Intersection over Union (mIoU) and pixel-level accuracy (Acc) for the semantic segmentation task over four datasets, and Root Mean Square Error (RMSE) and $\delta _ { 1 }$ (measuring the percentage of pixels with a relative error under 25% of the ground-truth) for the depth estimation task over one dataset.

Table 2. Comparison of models over CLIP backbone. Benchmarks are run using the method and code used for the JAFAR [8] paper. Bold = best, underline = second best.
<table><tr><td rowspan="3">CLIP</td><td colspan="8">Semantic Segmentation</td><td colspan="2">Depth Est.</td></tr><tr><td colspan="2">COCO</td><td colspan="2">VOC</td><td colspan="2">ADE20K</td><td colspan="2">Cityscapes</td><td colspan="2">COCO</td></tr><tr><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td>mIoU(↑)</td><td>Acc(↑)</td><td> $\delta _ { 1 } ( \uparrow )$ </td><td>RMSE(↓)</td></tr><tr><td>JAFAR [8]</td><td>58.16</td><td>78.52</td><td>79.07</td><td>94.69</td><td>36.65</td><td>71.39</td><td>50.98</td><td>91.23</td><td>45.75</td><td>0.92</td></tr><tr><td>AnyUp [31]</td><td>53.05</td><td>74.54</td><td>69.85</td><td>91.46</td><td>31.16</td><td>67.35</td><td>39.63</td><td>86.62</td><td>44.81</td><td>0.96</td></tr><tr><td>ED (ours)</td><td>57.86</td><td>78.23</td><td>79.26</td><td>94.67</td><td>35.5</td><td>71.22</td><td>46.21</td><td>90.67</td><td>48.20</td><td>0.86</td></tr></table>

## B.4. Semantic Segmentation and Depth Estimation for no-seg and no-depth ElasticDino Variants

We evaluated ElasticDino variants trained without the depth and the segmentation tasks in the same framework used to evaluate the full variant against task agnostic competitors over DINOv2 [52]. This is to strengthen the claim that ElasticDino only uses these tasks for structural guidance and does not rely on task specific information. Performance remains strong on these held-out tasks (Table 3).

## C. Further Qualitative Results

In this section we offer additional qualitative results.

Dedicated Segmentation models. We evaluate ElasticDino against two leading dedicated segmentation specific approaches Mask2Former [42] and OneFormer [48] over the Cityscapes[44] dataset. We showcase that even though our approach is task agnostic we surpass leading dedicated segmentation models using only our lightweight operations over the DINOv2[52] backbone, see Figure 2.

Training heads. ElasticDino employs different tasks to train the upscaler, by co-training a set of local and minimalistic 1 × 1 convolution heads, each performing a different dense task, considering pixels individually. In Figure 8, we demonstrate the results of these seven heads, both in their final step, where the features are of 256 × 256 resolution, and in an intermediate point in the network, where the resolution is upscaled from 32 × 32 to only 64 × 64. We further demonstrate qualitative results for intrinsic decomposition using the probe heads over the validation set of HyperSim [53] in Figure 1.

PCA feature visualization. In the feature maps comparison figures (Figures 6 and 7), we visualize the upscaled feature maps of ElasticDino using the first three components of a PCA decomposition and those of JAFAR [45], AnyUp [56], LoftUp [47], UPLift [55]. As can be seen, our method achieves crisp features and cleaner maps from different artifacts as opposed to the competitors. It can be noticed that ElasticDino is better at removing artifacts originating from spatial-encodings and separates far and near objects (the bars of the forklift) as well as appearance from geometry (the shield).

Visualization of deformations along upscaling process. Figures 4, 5 visualize the cumulative deformations through the process of feature upscaling for different images. Deformations are visualized using color wheel optical flow [41] and features are visualized through the first three PCA components. It can be seen that deformations help to make the edges more crisp and reduce artifacts in the feature maps.

Linear segmentation and depth probe heads. Finally, we also compare ElasticDino against JAFAR [45], AnyUp [56], LoftUp [47] in the framework suggested by JAFAR where linear probe heads are trained over the upsamplers for segmentation and depth tasks displaying the information encoded inside the upsampled features produced by the model allows achieving top results for dense tasks such as semantic segmentation (Figures 10, 12, 9, 11) and depth estimation (Figure 13). It can be seen that ElasticDino achieves superior results with cleaner edges and often detects details other models are not able to.

## D. Experiments to understand degraded performance over Cityscapes

We investigated our change in relative performance on Cityscapes [44], in contrast to all other datasets. We hypothesize it is because Cityscapes consists mainly of smaller objects and segments, connecting to our known limitation of filling ”holes” in feature maps. To further investigate this we evaluated AP (Average Precision) metric for the different models over segments of different size categories Small $\mathrm { ( a r e a < 3 2 ^ { 2 } ) }$ , Medium $( 3 2 ^ { 2 } < \mathsf { a r e a } < 9 6 ^ { 2 } )$ , Large (area $> 9 6 ^ { 2 } )$ as taken from COCO [50] instance segmentation evaluation methodology (see Figure 3). We also measured distribution of different segment size categories over the COCO[50], Cityscapes[44], Ade20k datasets. From this can see that the lower performance of ElasticDino can over Cityscapes can be explained by its lower performance on small thin objects present in this dataset in particular.

Table 3. Comparison of no-seg, no-depth ElasticDino variants against fully trained model. Benchmarks are run using the method and code used for the JAFAR[8] paper. Bold = best, underline = second best.
<table><tr><td></td><td colspan="8">Semantic Segmentation</td><td colspan="2">Depth Est.</td></tr><tr><td>DINOv2</td><td colspan="2">COCO</td><td colspan="2">VOC</td><td colspan="2">ADE20K</td><td colspan="2">Cityscapes</td><td colspan="2">COCO</td></tr><tr><td>Method</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td>mIoU↑</td><td>Acc↑</td><td> $\delta _ { 1 }$  ↑</td><td>RMSE↓</td></tr><tr><td>ED (no-depth variant)</td><td>61.7</td><td>81.1</td><td>84.1</td><td>96.3</td><td>41.3</td><td>75.5</td><td>60.1</td><td>93.1</td><td>60.5</td><td>0.64</td></tr><tr><td>ED (no-seg variant)</td><td>62.5</td><td>81.6</td><td>85.9</td><td>96.8</td><td>41.5</td><td>75.8</td><td>59.8</td><td>93.3</td><td>64.8</td><td>0.56</td></tr><tr><td>ED (full variant)</td><td>62.72</td><td>81.74</td><td>86.00</td><td>96.78</td><td>42.10</td><td>76.00</td><td>61.33</td><td>93.44</td><td>64.14</td><td>0.57</td></tr></table>

![](images/fcc1da5ee1e27ce0b32e18d1269f9db185df392d960da3f01f8d2ff54fa701db.jpg)

![](images/f78ffd8508465c4c76188da483eb729bec9d6c2664e62a4697bae144012ac745.jpg)

![](images/183e810f166ac0f6e8640b2ab81c25bae87d822c85b85de8448fc89e1e1d5f26.jpg)  
Figure 1. Qualitative comparison of ElasticDino to dedicated segmentation approaches (Mask2Former [42],OneFormer [48]) over the cityScapes dataset.

![](images/fab52ad9f159711a92f93278026c2d3de4bef9adcf561c8cd7c41bd9e547efe1.jpg)

![](images/8dbdf02e7fa608172603efe5c022a22535345f90e121c21b47c4fb0fbfc14a4a.jpg)

![](images/5980df91922a30ace43e937fe3a288cd5227b4bf5ce205dbc80423528f4776e2.jpg)

![](images/887e5db1b50c3730b1a36d22483d7a430c2b72c2831ef85a06fb03cfcf04adae.jpg)

![](images/715b57f2f3fb6a87f62cafce78ef76982289ee04fbcc91e5871d7cb3b295a123.jpg)

![](images/d16974a910f5d804902cfad63e8e49501e0a64874e5642b328f48d31a8cc94d0.jpg)

![](images/f669cf6597b66dbdb557a87e9d7fcd6047766f49588b9cfe282c4da8ee4c24a1.jpg)

![](images/f215f153c5be1d35637524432c938dc92981e12a73089c213424e79ded22e257.jpg)  
Figure 2. Intrinsic decomposition estimation trained on Hypersim [53] with a convolutional head at resolution 128x128. For each image, the decomposition is shown on the right: top row is the ground truth and bottom row is the prediction from ElasticDino. Left to right: albedo, diffuse shading, surface normals. Even in poor light conditions, we obtain strong qualitative results.

![](images/6f380651d8c499ecaba4cd09044c82fca8ff96dda8700af7094bc7152121c346.jpg)

![](images/635a8bc5a6aa159407b1bf80ecb98fbd2d3b608ec282092010b8726571e95233.jpg)  
Figure 3. Left: Accuracy vs size. We perform better for larger segments. Right: sizes across datasets. Cityscapes consists of smaller objects.

Flow field color coding  
![](images/fa797ed825af898559cf74332542a65426e6a5c90832f969d46502b28b0be8c0.jpg)  
Figure 4. Visualization of deformations along the different resolution stages of the upscaling process and their effect on the features while being upscaled. Deformations are visualized using color wheel optical flow [41], showing where each pixel was sampled from. Features are visualized through the first three PCA components. For each image, top row shows the feature maps and bottom row shows the cumulative deformations so far. Zoomed-in viewing recommended.

![](images/75b079be702cd3b1acc6c271e4b6284ac7017236bfe4e25ea3df64c85cab0336.jpg)

![](images/93eef52085413aa363e51596f073edc94d7674867961ae857f4b8ee042083fc5.jpg)  
Figure 5. Visualization of deformations along the different resolution stages of the upscaling process and their effect on the features while being upscaled. Deformations are visualized using color wheel optical flow [41], showing where each pixel was sampled from. Features are visualized through the first three PCA components. For each image, top row shows the feature maps and bottom row shows the cumulative deformations so far. Zoomed-in viewing recommended.

![](images/63b282444210862a38dfd9026d288bd75223a4e2e0f117b2e6b9a30b6280cb46.jpg)  
Figure 6. Additional qualitative comparisons through visualizations of the first three PCA components of the resulting features. From left to right: Input image, Original DINOv2 low-res features, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours. Zoomed-in viewing recommended.

![](images/9a76ac28ea5da44b4d07dd7e1cffe67be9c09292d2484004a25496f290631fe0.jpg)  
Figure 7. Additional qualitative comparisons through visualizations of the first three PCA components of the resulting features. From left to right: Input image, Original DINOv2 low-res features, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours. Zoomed-in viewing recommended.

![](images/79170f7334f408af127db232488a464c9a879be7eed60236c2daf0f90fcb3eb8.jpg)  
Figure 8. Qualitative evaluation of our auxiliary training tasks. During training, our encoder is trained various tasks, each with an individua head and the minimal receptive field of one pixel. Their results are demonstrated here. Top to bottom: PCA of the upscaled features, RGB reconstruction, depth estimation, semantic segmentation, and four tasks of intrinsic decomposition (shading, albeo, residuals, normals) For each image, we show (left) its GT labels (or the features before upscaling for the PCA), intermediate results of our heads after upscaling to size 64 × 64 (middle), and the final results of size 256 × 256 (right). Zoomed-in viewing recommended.

![](images/11417db2dafb6c558892d7b9c6cc08e2f264e2630e46fc43471c1fa70257bfd1.jpg)  
Figure 9. Additional qualitative comparisons for the segmentation task over the COCO dataset. From left to right: Input image, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours, Ground Truth. Zoomed-in viewing recommended.

![](images/4e477ab8fe61516d15d5290c896c624157b2294e560fa30000c64bc47654cc89.jpg)  
Figure 10. Additional qualitative comparisons for the segmentation task over the ADE20K dataset. From left to right: Input image, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours, Ground Truth. Zoomed-in viewing recommended.

![](images/484f3761f90187cdb17517437214bf1593cfa5d6c2e3ce451ed892118638c274.jpg)  
Figure 11. Additional qualitative comparisons for the segmentation task over the VOC dataset. From left to right: Input image, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours, Ground Truth. Zoomed-in viewing recommended.

<table><tr><td>Image</td><td>JAFAR</td><td>AnyUp</td><td></td><td></td><td></td><td></td><td>Image</td><td>JAFAR</td><td>AnyUp</td><td>LoftUp</td><td>UPLift</td><td>ED(ours)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Figure 12. Additional qualitative comparisons for the segmentation task over the CityScapes dataset. From left to right: Input image, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours, Ground Truth. Zoomed-in viewing recommended.

![](images/0a222d08b28e77c2c8e86bc1640753682a5b6c9a05842738391cc1a8cb2715e2.jpg)  
Figure 13. Additional qualitative comparisons for the depth estimation task over the COCO dataset. From left to right: Input image, JAFAR [45], Anyup [56], LoftUp [47], UPLift [55], ours, Ground Truth. Zoomed-in viewing recommended.

## References

[41] Simon Baker, Daniel Scharstein, James P Lewis, Stefan Roth, Michael J Black, and Richard Szeliski. A database and evaluation methodology for optical flow. International journal of computer vision, 92(1):1–31, 2011.

[42] Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. arXiv, 2021.

[43] Franc¸ois Chollet. Xception: Deep learning with depthwise separable convolutions, 2017.

[44] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding, 2016.

[45] Paul Couairon, Loick Chambon, Louis Serrano, Jean-Emmanuel Haugeard, Matthieu Cord, and Nicolas Thome. Jafar: Jack up any feature at any resolution, 2025.

[46] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

[47] Haiwen Huang, Anpei Chen, Volodymyr Havrylov, Andreas Geiger, and Dan Zhang. Loftup: Learning a coordinatebased feature upsampler for vision foundation models. arXiv preprint arXiv:2504.14032, 2025.

[48] Jitesh Jain, Jiachen Li, MangTik Chiu, Ali Hassani, Nikita Orlov, and Humphrey Shi. Oneformer: One transformer to rule universal image segmentation, 2022.

[49] Alexandre Lacoste, Alexandra Luccioni, Victor Schmidt, and Thomas Dandres. Quantifying the carbon emissions of machine learning. arXiv preprint arXiv:1910.09700, 2019.

[50] Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollar. Microsoft´ coco: Common objects in context, 2015.

[51] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.

[52] Maxime Oquab, Timothee Darcet, Th´ eo Moutakanni, Huy´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.

[53] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding, 2021.

[54] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa,¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J´ egou, Patrick La-´ batut, and Piotr Bojanowski. DINOv3, 2025.

[55] Matthew Walmer, Saksham Suri, Anirud Aggarwal, and Abhinav Shrivastava. Uplift: Efficient pixel-dense fea-

ture upsampling with local attenders. arXiv preprint arXiv:2601.17950, 2026.

[56] Thomas Wimmer, Prune Truong, Marie-Julie Rakotosaona, Michael Oechsle, Federico Tombari, Bernt Schiele, and Jan Eric Lenssen. Anyup: Universal feature upsampling. arXiv preprint arXiv:2510.12764, 2025.

[57] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data, 2024.

[58] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In European Conference on Computer Vision (ECCV), 2022.