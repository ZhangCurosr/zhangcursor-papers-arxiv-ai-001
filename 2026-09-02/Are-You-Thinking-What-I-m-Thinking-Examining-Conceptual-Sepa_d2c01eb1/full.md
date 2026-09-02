# Are You Thinking What I’m Thinking? : Examining Conceptual Separation in Neural Architectures

Jaee Ponde<sup>∗</sup> Truth Audit Labs jaee@truthauditlabs.ai

Roshni Agarwal<sup>∗</sup> Karya AI roshni.agarwal@karya.in

Subhashis Banerjee<sup>†</sup> Ashoka University suban@ashoka.edu.in

## Abstract

Neural networks are increasingly employed to identify both well-defined and ambiguous concepts, yet output-level metrics reveal little about how those concepts are represented internally. Our study asks if these networks exhibit conceptual separation: if examples of the same concept form coherent representations, and whether related concepts lie closer together in the representation space. We examine this conceptual organisation in Convolutional Neural Networks (CNNs) and Large Language Models (LLMs) through geometric and distributional analysis of their internal activations. In CNNs, familiar ImageNet concepts form coherent and semantically ordered representations, while this coherence weakens for unseen concepts and suffers within-class domain shift. In LLMs, clearly distinct domains remain well separated, related subdomains move closer together, and the distinction between ambiguous topics collapses at both the mean and covariance level. These results suggest that conceptual separation can reveal structure that output accuracy alone cannot, and may serve as a useful diagnostic of how robustly a model represents the concepts it is asked to identify. Code and data available on GitHub.

## 1 Introduction

Humans have an innate understanding of conceptual separation. We recognise that a cat and a dog are different classes, while also understanding that they are more similar to each other than either is to a car. This notion of proximity becomes complicated as concepts become less well defined. Subtopics within the same domain may be difficult to separate cleanly, and subjective concepts such as hate speech often have no universally agreed boundary; even human annotators can disagree substantially on the same examples [1, 2]. Conceptual separation, therefore, is not simply about whether two labels are different, but also about how clearly and consistently that distinction can be made.

Neural networks are increasingly deployed to classify both well-defined and ambiguous concepts, despite us knowing relatively little about whether they preserve such conceptual structure. They are evaluated largely through output-level metrics that can coexist with shortcut learning, underspecification, and poor behaviour under distribution shift [3, 4]. This motivates us to look beyond benchmark performance and test a model’s understanding of the underlying task. A natural place to do so is the activation space, which has been widely used to study what neural networks encode, how representations evolve across layers, and how similar representations are across models [5, 6]. We study whether the geometry of this activation space reveals the relative proximity and separation of concepts themselves.

We present a set of empirical experiments examining conceptual separation in CNNs and LLMs. CNNs are a natural starting point because visual classification benchmarks such as ImageNet provide explicit class labels and a controlled setting for studying representational separation [7]. This gives us a controlled setting in which to first ask whether familiar classes are separated as expected. From there, we extend the same question to language, where concept boundaries are often less clear. We study distinct domains, related technical subdomains, and finally hate versus non-hate speech, which are more ambiguous. Across both modalities, we use multiple geometric and distributional views of the activation space.

Our results suggest that internal activations provide a view beyond output labels, revealing how well a concept is represented relative to others inside the model. This structure can degrade across unseen, shifted, or ambiguous concepts, motivating the need for closer inspection of internal representations before real-world deployment. Our contributions are:

• We introduce an architecture-aware empirical framework for studying conceptual separation directly in neural activation spaces.

• In CNNs, we show that familiar classes exhibit coherent internal structure, which weakens not only for unseen concepts but also domain shifts within a class.

• In LLMs, we study a progression from clearly distinct domains to ambiguous domains, finding a corresponding degradation in representational separation.

• We extend the LLM analysis beyond mean-level geometry to covariance structure, testing whether distinctions that disappear in the first moment remain visible in the second.

## 2 Related Work

Our work is related to three strands of literature. First, concept-based interpretability seeks to explain neural networks in terms of human-meaningful concepts rather than individual features. TCAV introduced Concept Activation Vectors to quantify the sensitivity of model predictions to user-defined concepts [8], while Concept Bottleneck Models explicitly route predictions through human-interpretable concept variables [9]. Our goal is complementary: rather than asking whether a particular concept influences a prediction, we study how multiple concepts are organised relative to one another in the model’s activation space.

Second, a large body of work studies the structure and similarity of learned representations. Representational Similarity Analysis compares pairwise dissimilarity structure across representational spaces [10], while methods such as SVCCA and CKA quantify similarity between neural representations across layers or models [5, 6]. In NLP, probing classifiers are widely used to test whether linguistic or semantic properties can be decoded from internal representations [11, 12]. We instead focus on conceptual separation: whether concepts form coherent internal representations and how close different concepts are to one another.

Finally, our diagnostic framing connects to work on out-of-distribution detection and domain generalisation, which studies how model behaviour changes under inputs that differ from the training distribution [13, 14, 15]. Rather than relying only on output confidence or prediction accuracy, we examine shifts in activation distributions as evidence of whether conceptual structure remains stable under unseen or shifted inputs.

## 3 Experiments on Convolutional Neural Networks

We first study conceptual separation in image classifiers, where class boundaries are relatively clear. We ask whether images from the same class are represented coherently and whether some concepts are “closer” to one another than others.

## 3.1 Setup

We study two ImageNet-trained CNN architectures: ResNet-50 [16] and MobileNetV2 [17]. For each input image x, we extract the activation vector from the layer immediately preceding the final classifier, $z ( x ) \mathbf { \bar { \Psi } } \in \mathbb { R } ^ { d }$ , with $d = 2 0 4 8$ for ResNet-50 and $d = 1 2 8 0$ for MobileNetV2 as deeper CNN layers tend to encode richer features [18]. We treat this vector as the model’s representation of the image and study whether images belonging to the same concept occupy more similar regions of this representation space than images belonging to different concepts.

![](images/20b886fb226f513a52747d979d2ec959c93ce266fef7e20e2e98fc81a1717224.jpg)

![](images/573349432e54a35681e82605b2018b81ad4be73f8a66c59fc386aaf6656a1442.jpg)  
Figure 1: KL divergence between test-image activation distributions and class activation blueprints for ResNet-50 (left) and MobileNetV2 (right). Lower diagonal values show that images are closest to their own class blueprint.

Our initial experiments use three familiar ImageNet concepts–cats, dogs, and cars. We subsequently consider unseen concepts such as rangoli and microscopy, geographic variation in road images from India and Turkey, and pose variation within the cat class. Images are drawn from publicly available Kaggle datasets [19, 20, 21, 22, 23, 24]. We use 100 randomly sampled images of each class across experiments, examples available in Appendix A.

We explore several metrics to check conceptual separation. PCA is applied to the raw activation vectors after mean-centering, cosine distance normalises by vector magnitude, and Mahalanobis distance accounts for the within-class covariance structure. Finally, we use KL divergence [25] to compare the distribution of activation across neurons. This is inspired by prior work on KL-based measures of test-sample fit [26]. We adapt this idea to study conceptual separation: both the coherence of examples within a concept and the relative proximity between different concepts. For each concept c, we construct an activation blueprint by averaging the representations of N = 100 reference images, $\begin{array} { r } { \bar { z } _ { c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } z ( x _ { i } ) } \end{array}$ , and normalising it to sum to one, $\begin{array} { r } { Q _ { c } ( k ) = \bar { z } _ { c } ( k ) / \sum _ { j } \bar { z } _ { c } ( j ) } \end{array}$ . A test image is normalised in the same way, $\begin{array} { r } { P _ { x } ( k ) = z ( x ) _ { k } / \sum _ { j } z ( x ) _ { j } } \end{array}$ , and compared with the concept blueprint using KL divergence:

$$
D _ { \mathrm { K L } } ( P _ { x } \| Q _ { c } ) = \sum _ { k } P _ { x } ( k ) \log \left( \frac { P _ { x } ( k ) } { Q _ { c } ( k ) + \epsilon } \right) , \qquad \epsilon = 1 0 ^ { - 8 }\tag{1}
$$

Here, ϵ is added for numerical stability. Intuitively, $Q _ { c }$ represents the typical relative pattern of neuron activation for concept c. A low KL divergence indicates that a test image produces a similar activation pattern, while a high value indicates that its representation departs from the concept blueprint. For inter-class comparisons, we compute KL divergence in both directions and report their average, i.e., $\begin{array} { r l } { \frac { 1 } { 2 } \left[ D _ { \mathrm { K L } } ( Q _ { c } | | \dot { Q _ { d } } ) + D _ { \mathrm { K L } } ( Q _ { d } | | \dot { Q _ { c } } ) \right] } \end{array}$ , to obtain a symmetric measure of separation between concepts. This allows us to study both separation between concepts and coherence within a concept under unseen or shifted inputs.

## 3.2 In-Distribution Concept Separation

Our first experiment is concerned with the preservation of simple semantic structure between familiar ImageNet classes: cats, dogs, and cars.

As a first order analysis, we use these activations directly for Euclidean distance and PCA, cosine distance and Mahalanobis distance. Across both ResNet-50 and MobileNetV2, intra-class distances are smaller than inter-class distances. Cats and dogs are also closer to each other than either is to cars, indicating that the activation space preserves the semantic structure we would expect. The corresponding numerical results and methodology are reported in Appendix B.

For both architectures, KL divergence is lowest when an image is compared with its own class blueprint (Figure 1). The inter-class comparison also preserves the expected semantic ordering: cats and dogs have more similar activation distributions than cats and cars (Figure 2). Unlike point-wise distances, KL divergence captures the full pattern of relative neuron activation, giving us a more holistic measure of representational similarity. This provides evidence that the models are not only identifying concepts as distinct classes, but are also organising them relationally, with some concepts represented as more alike than others.

![](images/6df579301eaa1f22a0df0c4992055d9cb0f800c894783173bdedf04083e079f7.jpg)

![](images/2e627ddd615222ab514d1dc90e663684a461100f32a495b4d1cf5f7e817b406a.jpg)  
Figure 2: Mean inter-class KL divergence for ResNet-50 (left) and MobileNetV2 (right). Both models recover the same semantic ordering: cats and dogs have the most similar activation patterns, while cats and cars are the most distinct.

![](images/e320007b8c6c388daed80d463d4c89d837f4e31d9e453d2b601d96e3ea6c42ca.jpg)

![](images/13fc76d59515eb494571d2cdbed9c3274ae047460c2950531f09bd5190e5f8e8.jpg)  
Figure 3: Intra-class KL divergence for ImageNet and out-of-distribution concepts. OOD images produce substantially less consistent activation patterns than Imagenet Classes. MobileNetV2 (left) and ResNet-50 (right).

## 3.3 Out-of-Distribution Experiments

Out-of-distribution settings admit several definitions and evaluation protocols [27]; here, we use the term in a strictly semantic sense, referring to visually coherent concepts that are not represented as explicit ImageNet classes. Detecting such unfamiliar inputs is important for reliable deployment, since models may encounter inputs that differ from those seen during training [13, 27].

We test two human-distinguishable concepts that are not explicit ImageNet classes: Rangoli, a South Asian decorative art form made from geometric and floral patterns, and microscopy images. Rangoli and microscopy therefore serve as semantically OOD concepts: while the models may have encountered visually similar features during training, they were not trained to recognise these concepts as such. We repeat the same activation-blueprint experiment to ask whether unfamiliar concepts still produce coherent internal representations.

Both concepts show substantially higher intra-class KL divergence than the familiar ImageNet classes (Figure 3). This suggests that different images from the same human-defined concept do not produce a single, stable activation pattern in the way cats, dogs, and cars do. This does not imply that the model fails to extract useful visual features; rather, those features appear less consistently organised at the concept level. We therefore treat intra-class KL divergence here as a lightweight diagnostic of representational coherence under unfamiliar inputs.

We also examine the models’ ImageNet predictions for these images. Unlike the in-distribution classes, Rangoli and microscopy images are mapped across many unrelated ImageNet labels, providing a behavioural counterpart to the activation-space result. Results and further discussion are provided in Appendix C.

## 3.4 Within-Class Distribution Shifts

We next ask whether KL divergence can detect differences within a single concept. This is important because models deployed in the real world often encounter domain shifts, where the underlying class remains unchanged but visual conditions differ, and such shifts can substantially affect mode robustness [14, 15]. In these experiments, we use 200 images for the base class to ensure distinct blueprint and held-out test sets.

We compare street images from India and Turkey. Although both sets depict roads, they differ in visual context, including vegetation, vehicles, road markings, and surrounding infrastructure. Using an Indian-road activation blueprint, Turkish road images show consistently higher KL divergence than held-out Indian images (Figure 4) suggesting that the metric is sensitive to within-class distribution shift. In this sense, the blueprint comparison can serve as a simple pre-deployment check: given a model trained on a particular class distribution, we can test whether new examples of that same class induce substantially different internal representations before relying on the model in a new environment.

![](images/a1c45b9da2263af98bb5315e957f8736daa1388d460299acef544d10193724d4.jpg)

![](images/687cdaa045835d419e2d02f77e268944966cacbaadfb9538c317496658f9e7bc.jpg)  
Figure 4: KL divergence of Indian and Turkish road images from an Indian-road activation blueprint. Turkish road images show a clear drift from the Indian baseline.

We then test a different attribute within the cat class: sleeping versus standing cats. Using a sleepingcat activation blueprint, the KL divergence distributions for sleeping and standing cats are nearly identical (Figure 5). In contrast to the road experiment, changing pose does not produce a clear shift in the activation distribution. The models therefore appear to treat sleeping and standing cats as similarly coherent instances of the same concept, at least under this metric.

![](images/fbf20015cf301c5174fe63475d3a4fc2e6c449c7329dec5ebf9f5aa1b08a98be.jpg)  
Figure 5: KL divergence of sleeping and standing cat images from a sleeping-cat activation blueprint. The two distributions largely overlap in both models.

The two experiments suggest that not all within-class variation is reflected equally in representation space: geographic and contextual changes in road scenes are detectable, while pose differences within cats are largely absorbed into the broader class representation. One possible explanation is that pose is not a distinction the ImageNet objective explicitly encourages the models to separate, whereas road scenes differ along several visual dimensions that may already influence features learned during ImageNet training.

## 4 Experiments on Large Language Models

The CNN experiments established that familiar concepts produce coherent internal representations while unfamiliar or shifted concepts do not. We now ask whether the same principle holds in language, where concept boundaries are often less perceptual and more socially constructed. We study three concept pairs that form a deliberate progression of human-separability:

1. Computer Science vs. Shakespeare–maximally distinct domains that no literate reader could confuse, serving as a positive control.

2. Information Security vs. Theory of Computation–sub-domains within computer science that a specialist can distinguish but that share technical vocabulary and register.

3. Hate speech vs. no-hate speech–a socially contested boundary where even trained annotators systematically disagree on class membership [1, 2].

This progression tests whether the degradation of representational coherence observed for visual concepts extends to textual concepts, and whether it tracks the degree of human-perceived conceptual ambiguity.

## 4.1 Setup

We extract sentence embeddings from two architecturally contrasting models: GPT-OSS-20B (N=2880) [28], an open-weight autoregressive transformer, and BERT-large-uncased [29], a 340Mparameter bidirectional masked-language encoder (N=1024). GPT’s causal attention produces representations conditioned only on prior context, while BERT’s bidirectional attention integrates the full sentence simultaneously.

For each model, we mean-pool the final-layer hidden states (weighted by the attention mask) to produce a single sentence embedding $\mathbf { f } ( x ) \in \mathbf { \overline { { \mathbb { R } } } } ^ { N }$ . For the academic pairs, 100 sentences per class are drawn from Wikipedia and academic corpora. For hate speech, 100 sentences per class come from a single corpus of tweets from the 2020 US presidential election [30], ensuring that any observed difference reflects the concept itself rather than platform or stylistic confounds. Dataset details are in Appendix F.

We report pairwise Euclidean and cosine separability ratios (inter-class distance divided by mean intra-class distance; a value of 1.0 indicates no separation) and 3D PCA projections (Figure 6). Full distance matrices and auxiliary analyses are provided in Appendix E.

## 4.2 First-Moment Analysis

Table 1: Summary of representational separability across three concept pairs and two models. Cosine separability ratio measures first-moment (centroid-level) separation; normalised Frobenius norm $\tilde { F }$ and its permutation-test p-value measure second-moment (covariance-level) separation. Values are reported as GPT-OSS-20B / BERT-large-uncased.
<table><tr><td>Concept Pair</td><td>Cosine Ratio</td><td> $\tilde { F }$ </td><td>p-value</td></tr><tr><td>CS vs. Shakespeare</td><td>1.406 / 1.487</td><td>1.238 / 1.267</td><td> $< 0 . 0 0 1 / < 0 . 0 0 1$ </td></tr><tr><td>InfoSec vs. ToC</td><td>1.358 / 1.505</td><td>1.268 / 1.195</td><td> $< 0 . 0 0 1 / < 0 . 0 0 1$ </td></tr><tr><td>Hate vs. No-Hate</td><td>1.012 / 1.012</td><td>0.416 / 0.492</td><td> $0 . 0 2 5 / 0 . 1 3 0$ </td></tr></table>

Table 1 summarises the first-moment and second-moment results across all three pairs and both models. As the conceptual distinction becomes more ambiguous from a human perspective, the geometric metrics reflect the same loss of separability.

![](images/e21799458d83c9c62e08a93e51d1b8ea30796d4533a92e6e0b3c5c9fc5aeb1e0.jpg)  
Figure 6: 3D PCA projections of sentence embeddings. Top row: GPT-OSS-20B; Bottom row: BERT-large-uncased. Columns (left to right): CS vs. Shakespeare (well-separated), InfoSec vs. ToC (moderate overlap), and Hate vs. No-Hate Speech (complete geometric entanglement).

Academic pairs. For CS vs. Shakespeare, both models show a clear separation, 3D PCA projections show two distinct spatial clusters in the leading components (Figure 6, top row) and the cosine separability ratio exceeds 1.4 (Table 1). For InfoSec vs. Theory of Computation, separation remains significant. Cosine ratios drop slightly (Table 1) but still evident of distinct clusters, consistent with the hypothesis that conceptually closer domains produce less geometrically separated representations.

Hate vs. no-hate speech. For hate speech vs. no-hate speech, first-moment separation collapses entirely. The cosine separability ratio falls to effectively unity in both models. 3D PCA projections (Figure 6, bottom row) display complete spatial entanglement.

To quantify distributional differences beyond centroids, we compute a per-neuron KL divergence between classes. Because LLM hidden states can take negative values, we cannot treat an embedding vector as a probability distribution. Instead, for each neuron j, we bin its activation values across all M sentences in each class into a 50-bin histogram with shared global bin edges, yielding a probability mass function $\tilde { H } _ { c , j }$ . The per-neuron KL divergence is:

$$
\mathrm { K L } ( \tilde { H } _ { c , j } \| \tilde { H } _ { d , j } ) = \sum _ { b } \tilde { H } _ { c , j } ( b ) \cdot \log \frac { \tilde { H } _ { c , j } ( b ) } { \tilde { H } _ { d , j } ( b ) }\tag{2}
$$

Averaging across all N neurons yields a single scalar per class pair. This captures each neuron’s full distributional shape without parametric assumptions, though it discards inter-neuron correlations (which the covariance analysis in Section 4.3 addresses).

A six-category KL divergence analysis reveals that the per-neuron KL divergence between hate speech and no-hate speech is the smallest off-diagonal value in the entire $6 \times 6$ matrix–smaller even than between Information Security and Theory of Computation. From the models’ representational perspective, hate and no-hate speech are more similar than any two academic categories, including those from the same technical discipline (Appendix E).

## 4.3 Second-Moment Analysis

An absence of first-moment separability does not preclude all representational structure: the class centroids may coincide while the shapes of their activation distributions differ. A neuron whose activations are consistently high for hate speech but highly variable for no-hate speech would produce identical centroids but different covariance matrices. We therefore test whether the classes differ at the second moment.

![](images/cedc8dfbf818ce4edc473da30706b2f358716010c4fcf9faea3fa1affb2ffbcb.jpg)  
Figure 7: Permutation test null distributions (GPT-OSS-20B, 1000 random relabellings) of the normalised Frobenius norm ${ \tilde { F } } .$ Top row: CS vs. Shakespeare; bottom row: Hate vs. No-Hate Speech. Left: unprompted; right: prompted. The red vertical line marks the observed ${ \tilde { F } } .$

We model each class’s distribution in PCA-50 space as a multivariate Gaussian $\scriptstyle { \mathcal { N } } ( \mu , \Sigma )$ , estimated from 100 sentences per class. The normalised Frobenius norm,

$$
\tilde { F } = \frac { \left. \pmb { \Sigma } _ { 1 } - \pmb { \Sigma } _ { 2 } \right. _ { F } } { \frac { 1 } { 2 } \left( \left. \pmb { \Sigma } _ { 1 } \right. _ { F } + \left. \pmb { \Sigma } _ { 2 } \right. _ { F } \right) } ,\tag{3}
$$

measures how different the two classes’ covariance matrices are, scaled by their average magnitude. A value of $\tilde { F } = 0$ means the two classes spread their activations in exactly the same way; A value of $\tilde { F } = 0$ indicates identical covariance matrices, while larger values indicate greater covariance dissimilarity relative to their overall magnitude. We assess whether the observed difference is meaningful using a permutation test. We randomly shuffle the class labels 1000 times, recompute $\tilde { F }$ each time, and build a null distribution of values expected by chance. If the observed $\tilde { F }$ (red line in Figure 7) falls far to the right of this null histogram, the covariance difference is real; if it falls inside or near the histogram, the difference is no greater than what random grouping would produce.

Academic pairs. For CS vs. Shakespeare, $\tilde { F } = 1 . 2 3 8 ( \mathrm { G P T } , p { < } 0 . 0 0 1 )$ and 1.267 (BERT, $\scriptstyle p < 0 . 0 0 1 )$ the observed covariance difference lies far beyond the null distribution in all conditions (Figure 7, top). For InfoSec vs. ToC, $\tilde { F } = 1 . 2 6 8 \ : ( \mathrm { G P T } )$ and 1.195 (BERT), both $p { < } 0 . 0 0 1$ . The symmetric Gaussian KL decomposition reveals an informative structural shift: as the pair moves from Experiment 1 to Experiment 2, the covariance term accounts for a growing fraction of the total divergence (43% → 44% in GPT; 40% → 47% in BERT). As centroids converge, whatever distinction the model maintains is increasingly carried by the shape of the activation distribution rather than its location.

Hate vs. no-hate speech. For GPT-OSS-20B, F<sup>˜</sup> = 0.416 (p=0.025)–marginal, with the red line sitting at the very right edge of the null histogram. For BERT-large-uncased, $\tilde { F } = 0 . 4 9 2 ( p { = } 0 . 1 3 0 ) \cdot$ not significant, with the red line inside the null distribution (Figure 7, bottom). The covariance difference is $3 \times$ smaller in magnitude than for the academic pairs, and critically, it is not robustly cross-model. The Gaussian KL decomposition for this pair is dominated by the covariance term (93–94%), confirming that the first moment contributes essentially nothing.

Prompting as intervention. A natural question is whether the absence of separation is a fixed property of the representations or is sensitive to framing. We prepend a pair-specific classification prompt to each sentence (e.g., “Classify this text as hate speech or not hate speech:”) and exclude prompt tokens from the mean pool, so the embedding reflects the sentence content processed through a model whose context has been set by the classification frame. Prompting does not recover separation.

For the hate pair, GPT’s F<sup>˜</sup> drops from 0.416 to 0.389 (p=0.058); BERT’s from 0.492 to 0.483 (p=0.165). The mechanism is a homogenising attractor: transformer attention is context-sensitive, and prepending an identical prompt to all 200 sentences pulls all embeddings toward a shared “classification-task” region, compressing inter-class variance. For CS vs. Shakespeare, enough signal survives this compression $( \tilde { F }$ drops from 1.238 to 1.157, still p<0.001); for hate speech the prompt-induced compression erases what little remained.

The second-moment analysis confirms that the representational collapse observed for hate speech is not limited to class centroids–it extends to the full shape of the activation distributions. For well-separated domains, covariance structure differs significantly across classes. For hate speech, neither the location nor the shape of the distributions carries a reliable class-level signal.

## 5 Discussion

This work examines whether neural networks organise their internal representations in ways that reflect conceptual separation. Across both CNNs and LLMs, representational separability broadly tracks how distinct concepts are to humans, and weakens as those distinctions become less clear.

For CNNs, familiar ImageNet classes form coherent and well-separated representations, whereas out-of-distribution concepts such as Rangoli and microscopy show substantially lower intra-class coherence. We also observe sensitivity to within-class domain shift: geographically different road images produce a measurable change in the activation blueprint, while cat-pose variation does not. Together, these results suggest that representation-space diagnostics can reveal both when an unseen concept is encoded inconsistently and when a familiar class shifts under a new domain.

For LLMs, a similar progression emerges. Clearly distinct domains remain well separated, related technical subdomains move closer together, and hate versus no-hate speech becomes nearly indistinguishable at the first moment. Second-moment analysis does not recover a reliable boundary across both models, and prompting fails to restore the lost structure. This suggests that, for ambiguous tasks, a downstream classifier may need to learn a decision boundary on top of representations that do not already encode a clear conceptual distinction.

The practical value of conceptual separation is therefore diagnostic. Before deploying a model on a new domain or an ambiguous task, one can ask whether target concepts are represented coherently and distinctly in its internal space. A lack of separation does not imply that classification is impossible, but it is evidence that the underlying representation deserves further investigation. We therefore view activation-space analysis as a useful complement to output-level evaluation, particularly for shifted, unfamiliar, or context-dependent concepts.

## 6 Limitations and Future Work

Our findings should be interpreted as indicative rather than conclusive. The empirical scope is limited to 100 samples per class and four representative architectures, and the hate-speech analysis relies on a single dataset from one platform and time, so its generalizability across taxonomies and linguistic contexts remains uncertain. Our analysis is limited primarily to the first two moments of the activation distributions and, for LLMs, only the final transformer layer, potentially missing structure present at intermediate layers. Our prompting intervention is also limited to a single classification-style template per concept pair; richer prompting strategies may produce different insights.

Several directions follow naturally from this work. First, these measures could be developed into practical diagnostic checks for conceptual separation, assessing whether a model preserves coherent representations under new domains or ambiguous inputs before deployment. Developing a unified diagnostic metric will require broader validation across architectures, datasets, and tasks. A layer-wise extension across transformer layers could reveal where conceptual separation emerges or collapses. We also plan to study adversarial perturbations, testing whether representational coherence degrades further under targeted attack. More broadly, metric discovery remains an open problem: identifying which geometric spaces best capture relational structure between concepts. Ultimately, this could enable metric imposition, where training objectives or post-hoc regularisation encourage models to preserve desired conceptual relationships.

## References

[1] Aida Mostafazadeh Davani, Mark Díaz, and Vinodkumar Prabhakaran. Dealing with disagreements: Looking beyond the majority vote in subjective annotations. Transactions of the Associationfor Computational Linguistics, 10:92–110, 2022.

[2] Eve Fleisig, Rediet Abebe, and Dan Klein. When the majority is wrong: Modeling annotator disagreement for subjective tasks. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 6715–6726. Association for Computational Linguistics, 2023.

[3] Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020.

[4] Alexander D’Amour, Katherine Heller, Dan Moldovan, et al. Underspecification presents challenges for credibility in modern machine learning. Journal of Machine Learning Research, 23:1–61, 2022.

[5] Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. SVCCA: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. In Advances in Neural Information Processing Systems, volume 30, 2017.

[6] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In Proceedings of the 36th International Conference on Machine Learning, volume 97, pages 3519–3529. PMLR, 2019.

[7] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, et al. Imagenet large scale visual recognition challenge. International Journal ofComputer Vision, 115(3):211–252, 2015.

[8] Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and Rory Sayres. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings ofMachine Learning Research, pages 2668–2677, 2018.

[9] Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings ofMachine Learning Research, pages 5338–5348, 2020.

[10] Nikolaus Kriegeskorte, Marieke Mur, and Peter Bandettini. Representational similarity analysis— connecting the branches of systems neuroscience. Frontiers in Systems Neuroscience, 2:4, 2008.

[11] Yonatan Belinkov and James Glass. Analysis methods in neural language processing: A survey. Transactions of the Association for Computational Linguistics, 7:49–72, 2019.

[12] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.

[13] Dan Hendrycks and Kevin Gimpel. A baseline for detecting misclassified and out-of-distribution examples in neural networks. In International Conference on Learning Representations, 2017.

[14] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 8340–8349, 2021.

[15] Kaiyang Zhou, Ziwei Liu, Yu Qiao, Tao Xiang, and Chen Change Loy. Domain generalization: A survey. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(4):4396–4415, 2023.

[16] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[17] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 4510–4520, 2018.

[18] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In Advances in Neural Information Processing Systems, volume 27, pages 3320–3328, 2014.

[19] Siddharth Magesh. Rangoli detection dataset. Kaggle. Accessed: 2026-08-28.

[20] Melvin Paul Jacob. Cat dog dataset. Kaggle. Accessed: 2026-08-28.

[21] Mitanshu Chakrawarty. Indian driving dataset (idd dataset). Kaggle. Accessed: 2026-08-28.

[22] Yusuf Berk Saridogan. Traffic detection project. Kaggle. Accessed: 2026-08-28.

[23] Batuhan Yildirim. Electron microscopy particle segmentation dataset. Kaggle. Accessed: 2026-08-30.

[24] prondeau. The car connection picture dataset. Kaggle. Accessed: 2026-08-30.

[25] Solomon Kullback and Richard A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 22(1):79–86, 1951.

[26] Lakmal Meegahapola, Vengateswaran Subramaniam, Lance M. Kaplan, and Archan Misra. Prior activation distribution (pad): A versatile representation to utilize dnn hidden units. arXiv preprint arXiv:1907.02711, 2019.

[27] Jingkang Yang, Kaiyang Zhou, Yixuan Li, and Ziwei Liu. Generalized out-of-distribution detection: A survey. International Journal ofComputer Vision, 132(12):5635–5662, 2024.

[28] OpenAI. gpt-oss-120b & gpt-oss-20b model card. OpenAI, August 2025.

[29] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofNAACL-HLT, pages 4171–4186, 2019.

[30] Lara Grimminger and Roman Klinger. Hate towards the political opponent: A twitter corpus study of the 2020 US elections on the basis of offensive speech and stance detection. In Proceedings of the Eleventh Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis, 2021.

## A Dataset Examples

To provide visual context for the datasets used in the CNN experiments, we show representative examples from each concept and domain considered in the study.

## A.1 In-Distribution Concepts

Figure 8 shows representative cat, dog and car images used in the ImageNet-class experiments.

Cat images — 4 randomly sampled examples out of 100

![](images/bf5989fc43890fdd1a281affdf5d49d6791da4a02d58a041ae66c02a868e33b7.jpg)

![](images/d3245b24465688f114dc97b28fdb0543feea5e9c265907d3114f4e5da5e412f7.jpg)

![](images/836ffd66c150949ca7e199e112299d2fd2251d94241d68a811b26a2de97e42d4.jpg)

![](images/ce2e014decc046d6f5d7fa0514bb3b8f94080de104a76f6a01810015b7a3b614.jpg)  
Dog images — 4 randomly sampled examples out of 100

![](images/cd6403db95102edb03797215fd12b6da0923d33e28c92a1007d79e18143817b0.jpg)

![](images/7b8542daa97186cfabba4cbfddc6fefac80fe2be64b1a5b03978a5ca5df15247.jpg)

![](images/e94b8e55694ec3f48e818f76767cb3749cc176f0f14d1920c88a6aee4a921735.jpg)

![](images/3e0cea30c37f5e8476b5b262414427ee84af9098cad410c660fdeadffb664dc4.jpg)  
Car images — 4 randomly sampled examples out of 100

![](images/7e374766a3694f399a933978994bbe139ebfb93e11de7d2dcf8f9405d68f1bdb.jpg)

![](images/114553d779f1a71a2b47cb9175f028b04238d4071e0ca4b1356875cb9a1500e1.jpg)  
Figure 8: Representative cat, dog and car images used in the in-distribution experiments.

![](images/7663d12a4fd42d2f9edbea559c72a548c9b9924dbc53c0136ad85daed0b34390.jpg)

![](images/851726e8f54d45f22cf7ff1742c9189e456553ecb6199e3e2e411e66e15926b3.jpg)

## A.2 OOD Concepts

Figure 9 shows representative Rangoli and microscopy images used in the unfamiliar-concept experiments. Although both form visually coherent human-defined categories, neither corresponds to an explicit ImageNet class.

Rangoli images — 4 randomly sampled examples out of 100

![](images/19f45588f9b2a42686130a1fa817042745b363c15beb251f2041356d32ff0eb8.jpg)

![](images/696d2968c14ff3af5c63fb4757748a5a0eff9c7389a13c2a2da5cbfc3cd82744.jpg)

![](images/6279a19f72cffbf512bd84b9a2af7539eff75652445e061054cdd66a906f9e93.jpg)  
Microscopy images — 4 randomly sampled examples out of 100

![](images/04bd409956d0a8b88d4957238ff51c41b1b289fa642b41976e49ae321016d895.jpg)

![](images/a21ab24683af68333163e083ec83c8fef5809ec4bf63d96d71f8f33dadb102a1.jpg)

![](images/3fdb58ae85ea738766357ebc34e26f3629820ba659023a7817a302ca99d2aca2.jpg)

![](images/31086a196d5c68348c90c2792dcd34f52ea063c4327f665c74186dd73fc71d57.jpg)

![](images/4596debe0f6316b2e0e41a81fe00d3e182e0638f414bb1c653a58b6ba4372b97.jpg)  
Figure 9: Representative Rangoli and microscopy images used in the unfamiliar-concept experiments.

## A.3 Within-Class Distribution Shift

Figure 10 shows representative road images from India and Turkey held out test sets used to study within-class geographic distribution shift.

India road images — 4 randomly sampled examples out of 100  
![](images/a90c51fbd48f1a4baabd2608db503e41de4ef3314e38ceb1b41b955f28ff0ed2.jpg)

![](images/80e83e673cb9328a20a3a7be297ada244f7b20d5580deea74583d9051909a695.jpg)

![](images/4e445a3ed4737485d40278808f2792d575a5c3935307c3d53aaf9def7dfdc8b7.jpg)  
Turkey road — 4 randomly sampled examples out of 100

![](images/5455bbe719d85216daa857dc4fd370cb31c7be3cb28a677a19bcacfa5d032af4.jpg)

![](images/3a6f2038d531da5ac640bcfbd2f2ea3cccf83971cf929ee0be9fb2d48bd092d2.jpg)

![](images/6459f893f7ba675eeb12648f2fd6ba7e7f9c2ac95ffc2d1de96cab891af01057.jpg)

![](images/ac86922f0b9750802abe4985835e968041ec4f16579b052bf845b4813dab93b0.jpg)

![](images/4e391ef38527133644d72ebf9e63247347a8dd06a443775ad0c84ed4275b567c.jpg)  
Figure 10: Representative road images from India and Turkey used in the within-class distributionshift experiment.

## A.4 Within-Class Pose Variation

Figure 11 shows representative sleeping and standing cat images from the held out test sets. These images were manually picked from the cat dataset [20].

Standing cat images — 4 randomly sampled examples out of 100  
![](images/94a4a3f43e32575ff08580085542d3b41feb85712c2e4b2c500763cdcb695936.jpg)

![](images/c80bfe201043cc1161d82b40d548b325302c17b8906845ba811f04095e4b3426.jpg)  
Figure 11: Representative sleeping and standing cat images used in the pose-variation experiment.

## B PCA and Distance-Based Analysis

Across both architectures, cats, dogs, and cars form distinct regions in PCA space (Figure 12). Cosine and Mahalanobis distances are also smaller within classes than across classes, with cats and dogs closer to each other than either is to cars (Figure 13). This is consistent with the KL analysis in the main paper and indicates that the expected semantic structure is present across several measures of representation geometry.

![](images/8004290b9c30d40ddd89fe0ad5eff8d4aa4de6a644e1b051158a9402cb730de1.jpg)

![](images/9d50a4b10de8e63faa02145f8a953ed1c8f1a614fc1e30300d53470d88dbe00b.jpg)  
Figure 12: PCA projections of cat, dog, and car representations for ResNet-50 and MobileNetV2.

## C Prediction Consistency for Out-of-Distribution Concepts

We further examine the models’ ImageNet predictions as a behavioural view of representational consistency. The goal is not to expect correct predictions for rangoli or microscopy, since neither is an ImageNet class. Instead, we ask whether different images from the same concept are mapped to similar known classes.

![](images/dc805b765aca53e9ba4dc4f95bdc768fbca39fc6d5be05b42fc335ae3785d023.jpg)

![](images/cc7380c5d822615d7580768bf34d0ad9dccaa68a1bf2fdf53619ac7f600613da.jpg)

![](images/4c22feba1b2054da8548521a3d109715b263b8e526d15aae52391dbfd251d022.jpg)

![](images/3abac2c502311c498a0202a57ee86559043a59b8b45db958bf1111908da7a4e1.jpg)

![](images/6a7979aaa2b7ce7c14e8ca5314ec517b11e92e3f9d51050ce441a49ec9a616a3.jpg)

![](images/8b3d0ce413ee96dcc534688b6c6540ae4a64bc0599168861e29e9067fe95ce2e.jpg)  
Figure 13: Euclidean, cosine and Mahalanobis distances between cat, dog, and car representations. Intra-class distances are consistently smaller than inter-class distances.

For cats, dogs, and cars, the predictions are strongly concentrated in the expected semantic category. For example, over 75% of cat images are predicted as cats, while the corresponding numbers rise to around 90% for dogs and cars. Even when the exact ImageNet label differs, most remaining predictions are closely related classes. For readability, we therefore collapse fine-grained labels such as individual cat breeds, dog breeds, and car types into broader semantic groups. The resulting prediction distributions are highly concentrated (Figure 14), which is consistent with the low intraclass KL divergence observed for these concepts.

Rangoli and microscopy show the opposite pattern. For microscopy, the most common prediction accounts for only 15% of ResNet-50 images and 12% of MobileNetV2 images; for Rangoli, the corresponding values are 22% and 14%. The top five predicted classes together capture only 37–63% of the images, compared with 89–99% for cats, dogs, and cars. Predictions are therefore spread across many unrelated ImageNet classes rather than concentrated around a single semantic group (Figure 15). This is not a classification error in the usual sense, since neither Rangoli nor microscopy has a correct ImageNet label. Instead, the sharp drop in prediction consistency provides complementary evidence that different examples from the same human-defined concept elicit less coherent model responses, in line with the higher intra-class KL divergence reported in the main paper.

Top 5 Predicted Classes — Cats (out of 100 images)  
![](images/75342719eaeb689d7c6b0129a5a6ac85474d7afbe48bdd807b2cf3a57af831e8.jpg)

![](images/a6d230dfac29f1f40689d8f9f00d9a367e47c30dd45261287b8e143c3c6a0706.jpg)

Top 5 Predicted Classes — Dogs (out of 100 images)  
![](images/54fd718bdef4e5b06d1af39dcedc1bc23aabcbffd1d4813acd3b51b44acbc8ea.jpg)

![](images/2bc1911036eca9edce426bdc736a8caaafaa948e46be7997488a8808db631b40.jpg)

Top 5 Predicted Classes — Cars (out of 100 images)  
![](images/4599c22e024610c446d2113990b8f9c874e04b80a1f1b6e11498d4c8437dcf30.jpg)

![](images/df613ae48b8259bb4b11192adb9086f7e02f4205446ce1424264812484b14358.jpg)  
Figure 14: Top ImageNet predictions for cats, dogs, and cars. Predictions show relatively low variation within each concept.

Top 5 Predicted Classes — Microscopy (out of 100 images)  
![](images/7e1df03e3725f470be3208211d7e00806f348498a5d495cc3adc460b9ed3cd60.jpg)

![](images/ea9e62a383fee6dda167c2c18a68075eb3c00d276350dabd4711be603d78075d.jpg)

Top 5 Predicted Classes — Rangoli (out of 100 images)  
![](images/b1f217c13ccb5e7f59a6151a52056373a832d1df268ef28ffeef8807b6524cb9.jpg)

![](images/8fff2ec2363fddf339a8d9f2c1eaa3ca4aa85bcebe55b7b6a53e437f756e61e7.jpg)  
Figure 15: Top ImageNet predictions for rangoli and microscopy images. Images from the same concept are mapped across a diverse set of ImageNet classes.

## D Representative examples from the LLM datasets

We list three sentences from each concept class used in the language-model experiments. The full text collections are included in the public code repository under data/llm\_data/. The academic concept pairs (Computer Science vs. Shakespeare and Information Security vs. Theory of Computation) consist of short definitional or canonical sentences compiled from Wikipedia and other publicly available sources. Hate and no-hate examples are social-media posts drawn from the 2020 U.S. election Twitter corpus of Grimminger and Klinger [30], included in the public code repository under data/llm\_data/, and are reproduced as in the source (including hashtags and usernames). They may contain profanity and partisan language.

Table 2: Three examples from each LLM concept class
<table><tr><td>Class</td><td>Examples</td></tr><tr><td rowspan="4">Shakespeare</td><td></td></tr><tr><td>1. To be, or not to be: that is the question 2. Brevity is the soul of wit.</td></tr><tr><td>3. All the world&#x27;s a stage, and all the men and women merely players.</td></tr><tr><td></td></tr><tr><td rowspan="4">Computer Science</td><td>1. An algorithm is a finite sequence of well-defined instructions to solve a problem.</td></tr><tr><td>2. Time complexity measures how an algorithm&#x27;s runtime grows with input size.</td></tr><tr><td>3. Big-O notation describes the upper bound of an algorithm&#x27;s growth rate.</td></tr><tr><td></td></tr><tr><td rowspan="3">Information Security</td><td>1. Information security ensures confidentiality, integrity, and availability of data.</td></tr><tr><td>2. Encryption transforms plaintext into unreadable ciphertext.</td></tr><tr><td>3. Asymmetric encryption uses a public-private key pair.</td></tr><tr><td rowspan="3">Theory of Computation</td><td>1. Theory of Computation studies abstract models of computation.</td></tr><tr><td>2. Deterministic finite automata (DFA) have exactly one transition per input symbol.</td></tr><tr><td>3. Regular languages are recognized by finite automata.</td></tr><tr><td rowspan="3">Hate Speech</td><td>1. Worship of dictator trump.</td></tr><tr><td>2. Give me 10 GOOD fucking reasons to vote trump. #BIDEN2020</td></tr><tr><td>3. Don&#x27;t let your daddy issues influence you to vote for Biden #magaszn #maga2020</td></tr><tr><td colspan="2">No-Hate Speech</td></tr><tr><td rowspan="3"></td><td>1. Yay! Voted today! #BidenHarris2020</td></tr><tr><td>2. Voted today! Please find the time and vote. #vote #BidenHarris2020</td></tr><tr><td>3. Got my #Trump2020 swag in today!!!!!! So excited!</td></tr></table>

## E LLM Distance Matrices and PCA Projections

This appendix provides the full set of distance matrices, six-category KL matrices, Frobenius null distributions, and moment decomposition analyses across both GPT-OSS-20B and BERT-largeuncased.

First vs Second Moment Separation — BERT-large-uncased Unprompted vs Prompted  
![](images/a16c6dad48f2eaca08c52c81e076dabf836b2288c12ff258054c9b63ce3d474f.jpg)

![](images/1ebc42eb9611f55789205ef4cde3e5ccf716eff5298e66905fb820d782be658d.jpg)

![](images/4df26b9c571c69c25095d1152233030c12f29382e5bc30f897d10a7f8873ca20.jpg)

![](images/7e768ce25de3d3e9695659e9696495ea4167a03c6786ca4a991d5d5d4c4ead54.jpg)

![](images/1da2deb10a719c4934142617eed9b3ef5809789719ce9a5c11c028d017381d48.jpg)

![](images/88a7f0cee5bb15e1d81e48eec276f428425222a01d7ca7733f8451def7b989a8.jpg)

![](images/4e45d5c996d65160fcd426d977010f7c8410ddf25d425d5c2de7d384ff22e610.jpg)  
Figure 16: First vs. second moment separation and Gaussian KL decomposition under unprompted and prompted conditions. Top: GPT-OSS-20B; Bottom: BERT-large-uncased. Separability ratios collapse to unity and Gaussian KL is overwhelmingly dominated by the covariance term (> 90%) for the hate speech distinction across both architectures.

## F Hate Speech Data and Labelling Framework

For hate speech, 100 sentences per class are drawn from the 2020 U.S. presidential-election Twitter corpus of Grimminger and Klinger [30]. We derive the binary Hate Speech / No-Hate Speech partition from the corpus’s hate/offensive-speech annotations.

The choice of a single, domain-controlled corpus is deliberate. Cross-domain hate speech detection is a fundamentally different problem from within-domain detection, and a dataset drawn from multiple platforms, time periods, or political contexts would conflate domain shift with the conceptual distinction we aim to isolate. By restricting to tweets from the same event window and platform, any observed difference in embedding geometry reflects a difference in the underlying concept–the linguistic intent and framing of the speech–rather than writing style, topic distribution, or platform conventions.

![](images/02e47e586f027ef1c47bf385fe88efdad9aee271f77ea0f0fbb7a9755390e960.jpg)  
Figure 17: Distance matrices for CS vs. Shakespeare. Left: Euclidean; Right: Cosine. Within each block: GPT (top) and BERT (bottom).

![](images/84cd6933e54fd64cdc2533d39d660c33e3118e9c3df82c85f2499e9ef6873d2b.jpg)  
Figure 18: Distance matrices for InfoSec vs. Theory of Computation. Layout as in Figure 17.

![](images/9e7df5be27cd61d8f7c03c78f75e98bc17f7adf06133752c941b0fb172688a9c.jpg)  
Figure 19: Distance matrices for Hate Speech vs. No-Hate Speech. Layout as in Figure 17.

This design connects to the broader challenge of hate speech annotation. Unlike academic domain labels, which can be assigned with high inter-annotator agreement, hate speech labels are socially contested: annotators disagree on boundary cases depending on cultural context, personal sensitivity, and the operational definition used [1, 2]. The geometric analysis in the main paper sidesteps the labelling problem by asking whether pre-trained representations encode any concept-level distinction between these classes, regardless of where the decision boundary is drawn. The finding that they do not–at either the first or second moment–suggests that the difficulty of hate speech annotation may have a structural counterpart in the representational geometry of general-purpose language models.

![](images/ed329817838f09c6b8588c28afa71cae6ae4a74955aa1a8e8b90333f04366d82.jpg)

![](images/b521c5238ac53faaa7cdd5791d1717d3d6f7e3affe5e30a8cca698d799149922.jpg)  
Figure 20: Per-neuron KL divergence matrix across all six categories (Left: GPT; Right: BERT). Hate and No-Hate Speech have the smallest off-diagonal KL of any pair.

![](images/f63ffa3716a6a0f1c69ea198f4df52405d30a623e2c5bab5f5f29b8eb6166e37.jpg)

![](images/b98fa9b7c246fadf9962e900b015829362afa638a2ddc9a66c297b8f95959c90.jpg)

![](images/9c813c04b7844e6abe94e570f9f445438fd440ff131a29d482f32f288ce1ddbb.jpg)

![](images/e254c9bd2cc08e13b070ac1b2a5ce164c7f129507e61d875e32e4b258edc2ada.jpg)  
Figure 21: Permutation test null distributions (BERT-large-uncased) of the normalised Frobenius norm $\tilde { F }$ for CS vs. Shakespeare (top) and Hate Speech (bottom).

![](images/e5cc68ffea10b216335bf59f96b49880e9a1def646c96dd07be1e51ed352ab04.jpg)

![](images/5ec7d7bc83bf4b50f6cd436fbdb379789de949c1667236a5179c844e22eda3e8.jpg)

![](images/4160a19391a09c7be97e0900ea4e0897abdfd1324411ccefc928d924c9e49073.jpg)

![](images/05106e0a7535fb8b736042281d38c7eb2db0537df6aef1af4e8292099434f4cd.jpg)  
Figure 22: Permutation test null distributions for InfoSec vs. Theory of Computation. Top: GPT-OSS-20B; Bottom: BERT-large-uncased. Left: Unprompted; Right: Prompted.