# Contrastive Learning for Aspect Representation towards Explainable Recommendation

Emrul Hasan

Department of Computer Science

Toronto Metropolitan University

Toronto, Canada

e1hasan@torontomu.ca

Chen Ding

Department of Computer Science

Toronto Metropolitan University

Toronto, Canada

cding@torontomu.ca

Abstract—In this work, we propose a novel recommendation model, CLARER (Contrastive Learning for Aspect Representation towards Explainable Recommendation) that integrates aspect features learned from textual reviews with rating information to improve the accuracy and explainability of recommendations. Our proposed framework learns user and item representations by combining rating-based features and aspectbased features from reviews. Specifically, rating-based features are learned through a multi-layer perceptron (MLP) model, while aspect-specific review representations are learned using a transformer encoder to capture the semantic information and contrastive learning to better distinguish user preferences. To provide explanations, we train a transformer decoder, using the final representations of users and items from both rating and aspect-based features as context. Experimental results in three benchmark data sets demonstrate that our model achieves superior performance compared to baseline methods in both recommendation (accuracy) and explanation generation.

Index Terms—Aspect-aware, Feature representation, Explainability, Contrastive Learning, Review-based Recommendation

## I. INTRODUCTION

While recommendation systems are essential for delivering personalized suggestions in today’s digital landscape, traditional approaches relying on latent features from past interactions often lack interpretability [1]–[5]. To enhance the credibility and user acceptance of the recommendation results, it is essential to provide clear justifications or explanations for the recommendations [6], [7]. Justifications can be expressed in several different forms, e.g., using a predefined template [8], [9], highlighting the important parts [10], generating short descriptions [11], [12]. Given the significance of explainable recommendations, this crucial area of research has received considerable attention in recent years [13]–[15].

One of the main challenges in explainable recommendation systems is to identify and incorporate more contextual information and fine-grained user preferences and opinions into personalization and explanation generation. Many existing approaches rely on user and item ID-based feature representation and use them as contextual information in the explanation generation process [12], [16], [17]. However, ID-based features exist in a semantic space different from that of natural language, leading to poor alignment between two different semantic spaces and limited personalization in explanation generation. Furthermore, due to the sparsity of historical rating data, ID-based features often fail to capture accurate representations of users and items [18]. As a result, incorporating only ID-based features into the recommendation model performs suboptimally both in rating prediction and in explanation generation.

To address these issues, previous research has attempted to map user and item IDs onto textual representations and integrate them into the decoding process for explanation generation [16]. Although this approach introduces a degree of personalization, it also increases the model complexity and suffers from inconsistencies in mapping ID-based features to natural language explanations. Consequently, there remains a need for a more effective approach that could directly integrate user preferences into the recommendation generation process while ensuring interpretability and coherence.

Another major challenge is the feature extraction from user reviews. Capturing fine-grained user preferences from historical interactions is crucial to improving recommendation performance. User-generated reviews provide the valuable information, but extracting meaningful features from textual data remains difficult due to its unstructured and noisy nature. Some recent approaches have employed external aspect extraction tools to pre-process textual data before incorporating them into the explanation generation model [19]. Although these methods have shown promise, they rely heavily on external tools, making them less adaptable across different domains. Thus, developing a self-contained feature extraction mechanism that effectively captures aspect-based user preferences remains an open challenge.

To overcome limitations of existing explainable recommendation methods, we propose CLARER (Contrastive Learning for Aspect Representation towards Explainable Recommendation), a novel framework that jointly integrates ID-based features with aspect-aware representations learned from user reviews. The key contribution of our work can be summarized as follows.

1) We propose CLARER, a novel explainable recommendation framework that seamlessly integrates aspect-based representations with user-item latent factors, improving both rating prediction and explanation generation. To the best of our knowledge, CLARER is the first to combine ID-based features with aspect representations for both rating prediction and explanation generation (i.e., the decoder is initialized using the combined feature).

2) We introduce a novel aspect representation learning technique that uses predefined domain-specific aspects and contrastive learning. We are the first to apply contrastive learning for aspect representation in explainable recommendation systems.

3) We run experiments on three real-world datasets and the results show that our approach outperforms state-of-theart baselines in both accuracy and explainability, with aspectaware representations driving the performance gains.

## II. RELATED WORK

This section presents two key areas relevant to our work: aspect representation and explainable recommendation systems.

## A. Aspect Representation

Deep learning, particularly representation learning, is widely used to automatically capture meaningful features from reviews. For example, CNN-based models [4], [20] extract features from the review text, while attention mechanisms prioritize important reviews and integrate rating-based and review-based features. Notable models include NARRE [21], which evaluates the usefulness of the review, and mutual attention methods [22] that capture dynamic interactions. Hierarchical attention networks [23] distinguish user and item structures, while gating mechanisms [24] adaptively balance ratings and reviews. Further advances, such as MSAR [25] and TARMF [26], improve feature fusion using gated layers and recurrent neural networks.

Although various topic modeling and attention-based models are applied to general review-based or aspect-based recommendation systems, aspect-specific representations for explainable recommendations have yet to be explored fully. [27] utilizes non-negative matrix factorization [28] to extract latent topics from review texts while applying matrix factorization to model rating data. Topic-based methods, such as Latent Dirichlet Allocation (LDA), are used to model aspects from reviews for recommendation tasks [29], [30] and extended with aspect-aware models by [31]. These topic-based models integrate learned topics into rating-based frameworks, such as matrix factorization, enhancing the interpretability of recommendation systems. However, these methods primarily rely on bag-of-words representations, overlooking comprehensive semantic information from reviews, which limits their ability to fully capture meaningful insights.

Very recently, [13] incorporates aspect representation as the context for explanation generation. They extract relevant aspects from the review, which are then encoded using the NLP technique. We argue that the dependency on an external tool might restrict the ability of the model to capture all the aspects that users discuss in the review, leading to suboptimal representation learning. To this end, we propose using contrastive learning for predefined aspect representation learning. This technique contrasts the individual aspect with the entire representation of the review. Therefore, the aspect representations are rich, leading to performance improvement. To the best of our knowledge, we are the first group to use this technique in building explainable recommendation systems.

## B. Explainable Recommendation Systems

One of the main challenges in generating personalized explanations is incorporating contextual information into explanation generation [10], [11], [21], [32]. The traditional approach relies on the user and item ID-based feature representation. For instance, [16] leverages historical rating data to learn user and item representations for personalization. In addition, this model is trained to generate reviews by integrating both user-item representations and review features as contextual information for explanation generation. Similarly, [12] utilizes historical user-item rating data to model user-item interactions, using them to initialize the decoder network for template-controlled explanation generation. ID-based feature representation models face challenges in handling sparse useritem interactions.

Meanwhile, [17] employs historical interaction data and an attention based sequence-to-sequence model for attributeconditioned review generation. Expanding on the role of IDs in explanation generation, [33] incorporates IDs as prompts to enhance explanation generation using large language models (LLMs). Although LLM-based models demonstrate promising potential, they struggle to effectively capture user-item collaborative preferences, particularly complex and nonlinear relationships between users and items. Recently, some works have focused on using textual reviews as contextual information to initialize the decoder model for explanation generation. For instance, [14] integrates the opinionated review text to enhance the recommendation and explanation. [13] integrates multimodal information, e.g., ID-based features, aspect features, retrieval sentences, and interpretation text, improving the quality of the explanation. On the other hand, [34] uses a variational autoencoder with GMM for rating reconstruction and a multigating mixture of experts for explanation generation, increasing personalized recommendation and effectively handling the data sparsity issue. Despite their effectiveness, LLM-based approaches convey out-of-domain information inherently since they are trained on the general internet data. In addition, they ignore integrating the strength of both rating and aspect-based features for rating prediction and explanation generation. To this end, we propose to integrate aspect-specific features for both rating prediction and explanation generation.

## III. METHODS

In this section, we present our proposed framework, CLARER, as illustrated in Fig. 1, and provide a detailed description of its three main components.

## A. Priliminaries

Let us consider a user $u ,$ an item $v ,$ a review $x ,$ and an explanation e (during training only), the objective is to predict the overall rating $r _ { u v } .$ , which reflects the user’s preference for the item, along with an explanation $e \prime$ that justifies why the item is recommended.

## B. User and Item Representation

We employ a simple yet effective method for user and item ID-based representation using a multilayer perceptron (MLP). While MLP-based ID representations have certain limitations, they are favored in our approach due to their simplicity, computational efficiency, and ability to model complex nonlinear interactions [1]. We begin by randomly initializing the embeddings for users and items. These embeddings are concatenated and propagated through a series of hidden layers. The final output of the hidden layer is combined with the aspect-level representation derived from user-generated reviews, as discussed in section III (C). The combined feature is fed into a linear layer to generate a scaler value, e.g., a rating score, which is optimized by minimizing the difference from the ground-truth rating.

Formally, consider $p _ { u }$ and $q _ { v }$ to be the latent representations of the user u and the item v respectively. To learn the representation, the latent vectors are concatenated as follows.

$$
\boldsymbol { z } = \mathrm { C o n c a t e n a t e } ( p _ { u } , q _ { v } )\tag{1}
$$

This concatenated vector is the input of the MLP model, $\begin{array} { r } { \mathbf { e } . \mathbf { g } . , h _ { 0 } = z . } \end{array}$ The hidden layer $h _ { l }$ and the final layer are defined as follows.

$$
h _ { l } = \mathrm { R e L U } ( W _ { l } h _ { l - 1 } + b _ { l } )\tag{2}
$$

$$
h _ { L } = \mathrm { R e L U } ( W _ { L } h _ { L - 1 } + b _ { L } )\tag{3}
$$

Here, $W _ { l }$ and $W _ { L }$ denote the weight matrices for the hidden layer and the final layer respectively, while $b _ { l }$ and $b _ { L }$ represent the bias terms for the l-th and L-th layer, respectively. $R e L U$ is the activation function.

![](images/14b1ad0dffeb7f3e05b3b8c975fe9288a8cf9e966d42b52349bc1bf57730a1ef.jpg)  
Fig. 1. The detailed architecture of the proposed model.

## C. Aspect Representation

Aspect representation learning involves learning the rich representation of various aspects discussed in the review. To this end, we utilize the contrastive learning method [35] to learn representations of reviews at the aspect level and combine them with ID-based features to predict ratings and generate explanations. In our approach, we consider domaindependent aspects. The details of the predefined aspects for each domain are shown in Table I. The aspects in the TripAdvisor dataset are predefined by the TripAdvisor platform, whereas for Amazon and Yelp reviews, we select and define aspects through data analysis, e.g., topic analysis.

TABLE I  
ASPECT CATEGORIES ACROSS DOMAINS
<table><tr><td rowspan=1 colspan=1>Aspects</td><td rowspan=1 colspan=1>Domain</td></tr><tr><td rowspan=1 colspan=1>Plot, Storyline, Cultural Relevance, Emotion, Humor,Acting, Direction, Visuals, Sound, Genre, Charac-ters, Entertainment</td><td rowspan=1 colspan=1>Amazon(MT)</td></tr><tr><td rowspan=1 colspan=1>Service, Food, Ambience, Cleanliness, Price, Variety,Location, Wait Time, Noise</td><td rowspan=1 colspan=1>Yelp</td></tr><tr><td rowspan=1 colspan=1>Service, Location, Cleanliness, Sleep, Room Size,Check-in, Dining, Value, Amenities, Safety</td><td rowspan=1 colspan=1>TripAdvisor</td></tr></table>

The process starts with obtaining embeddings for all predefined aspects using a pre-trained LLM embedding model; here we use a sentence transformer [36]. Next, we train a transformer encoder to generate the review representation. Finally, the review features are contrasted with the aspect embeddings to learn the final review-based aspect-level representation.

For a given set of aspects $a _ { 1 } , a _ { 2 } , . . . , a _ { K }$ where K is the number of aspects, the representation of an aspect $a _ { k }$ can be expressed as

$$
a _ { k } ^ { e } = E m b e d d i n g ( a _ { k } )\tag{4}
$$

To effectively extract meaningful features from textual reviews, we train a transformer encoder model [37]. The encoder processes user-generated reviews to learn rich, context-aware representations. The model uses self-attention mechanisms to capture long-range dependencies and semantic relationships within the text, enabling a fine-grained understanding of the review content. Once the review features are extracted, they are contrasted with aspect embeddings and trained using contrastive loss. For each aspect, the review feature is treated as an augmented version that better aligns with the aspect, while negative samples are generated from the aspect’s embedding using a random generator. We consider the review as the positive counterpart of the aspect, assuming that the aspect is inherently reflected in the review, allowing the model to learn this relationship effectively.

Given an input review sequence of tokens $\begin{array} { r l } { \mathbf { X } } & { { } = } \end{array}$ $( x _ { 1 } , x _ { 2 } , \dots , x _ { T } )$ , the encoder processes the input through multiple layers consisting of self-attention and feed-forward networks. Each input token is first mapped to an embedding space and then combined with positional encodings. The combined feature E is represented as follows:

$$
\mathbf { E } = \mathrm { E m b e d d i n g } ( \mathbf { X } ) + \mathrm { P o s i t i o n a l E n c o d i n g } ( \mathbf { X } )
$$

$$
\begin{array} { c } { { \bf Q } = { \bf E } W ^ { Q } , \quad { \bf K } = { \bf E } W ^ { K } , \quad { \bf V } = { \bf E } W ^ { V } } \\ { \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \mathrm { A t t e n t i o n } ( { \bf Q } , { \bf K } , { \bf V } ) = \mathrm { s o f t m a x } \left( \frac { { \bf Q } { \bf K } ^ { T } } { \sqrt { d _ { k } } } \right) { \bf V } } \\ { \quad \quad \quad \quad } \\ { \quad \quad \quad \quad \mathrm { M u l t i H e a d } ( { \bf Q } , { \bf K } , { \bf V } ) = \mathrm { C o n c a t } ( \mathrm { h e a d } _ { 1 } , \dots , \mathrm { h e a d } _ { h } ) W ^ { O } } \\ { \quad \quad \quad { \bf Z } = \mathrm { L a y e r N o r m } ( E + \mathrm { M u l t i H e a d } ( { \bf Q } , { \bf K } , { \bf V } ) ) } \\ { \quad \quad \quad \quad \quad \quad \quad \mathrm { F F N } ( { \bf Z } ) = \mathrm { R e L U } ( { \bf Z } W _ { 1 } + b _ { 1 } ) W _ { 2 } + b _ { 2 } } \\ { \quad \quad { \bf Z } _ { o } = \mathrm { L a y e r N o r m } ( { \bf Z } + \mathrm { F F N } ( { \bf Z } ) ) } \end{array}
$$

$\mathbf { Z } _ { o }$ is the contextual representation of the review. We train this encoder module with contrastive loss to obtain the aspectlevel representation.

Let $\mathbf { Z } _ { o } \in \mathbb { R } ^ { d }$ be the review embedding, $a _ { k } ^ { e } \in \mathbb { R } ^ { d }$ be the positive samples (e.g. equation $4 ) , a _ { j } ^ { e }$ be a randomly sampled negative aspect embedding from the current batch and $\tau$ be the temperature parameter. The contrastive loss $\mathcal { L } _ { i , j }$ can be defined as

$$
\mathcal { L } _ { i , j } = - \log \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { \exp ( \sin ( a _ { k } ^ { e } , Z _ { 0 } ) / \tau ) } { \sum _ { j = 1 } ^ { 2 N } \mathcal { Y } _ { [ j \neq i ] } \exp ( \sin ( a _ { k } ^ { e } , a _ { j } ^ { e } ) / \tau ) }\tag{5}
$$

where K and N are the total number of aspects and the number of negative samples respectively. Total loss is obtained by averaging over all the aspects.

D. Integration of ID-based Features with Aspect Representation

First, ID-based user and item representations are learned using MLP, as discussed in Section III (B). Then, aspect-based features are incorporated using a gating function, defined as:

$$
\chi = \sigma ( W \cdot Z _ { o } + b ) * Z _ { o } + ( 1 - \sigma ( W \cdot Z _ { o } + b ) ) * h _ { L } )\tag{6}
$$

where $Z _ { o }$ represents the aspect-based features and $h _ { L }$ denotes the ID-based representation of users and items. The sigmoid activation function $\sigma ( \cdot )$ acts as the gate function that controls the contribution of two different representations. The final user-item representation, χ, effectively encodes both the interaction dynamics and the aspect-specific information.

## E. Rating Prediction

The recommendation can be defined as a rating prediction problem where the goal is to predict a rating score $r _ { u v }$ based on the representation of the user u and the item v. Once we obtain the combined feature representation from the ratings and reviews, we apply a simple regression layer to predict the ratings as follows.

$$
\hat { r } _ { u , i } = \sigma ( \mathbf { W } ^ { r } \boldsymbol { \chi } + \mathbf { b } ^ { r } ) + b ^ { r }\tag{7}
$$

where $\mathbf { W } ^ { r } \in \mathbb { R } ^ { d \times d } , ~ \mathbf { b } ^ { r } \in \mathbb { R } ^ { d } .$ , and $b ^ { r } \in \mathbb { R }$ are weight parameters, and $\sigma ( \cdot )$ is the sigmoid function. The task is trained with Means Squared Error loss as:

$$
\mathcal { L } _ { r } = \frac { 1 } { | T | } \sum _ { ( u , i ) \in \mathcal { T } } ( r _ { u , i } - \hat { r } _ { u , i } ) ^ { 2 }\tag{8}
$$

where $r _ { u , i }$ is the ground truth rating and $\tau$ denotes the total number of samples.

## F. Explanation Generation

Inspired by the performance of the autoregressive language model [37], we use a transformer-based decoder model for explanation generation. Given an input sequence of tokens $\textbf { Y } = \ ( y _ { 1 } , y _ { 2 } , . . . , y _ { T } )$ , the decoder generates the output sequence autoregressively by attending to both the previous user-item representation generated from review and ID and the output of the encoder. The decoder first applies masked self-attention to prevent attending to future tokens. Given the user-item representation $\chi ,$ the key and value are updated as:

$$
P ( y _ { t } | y _ { < t } , \chi ) = \mathrm { s o f t m a x } ( \mathbf { Z } _ { \mathbf { o } } ^ { \prime } W ^ { O } + b )\tag{9}
$$

where $W ^ { O }$ and b are learned parameters. $\mathbf { Z _ { o } ^ { \prime } }$ is the decoder output which follows a similar operation as discussed for the encoder module in Section III(C). This process is repeated for each token until the last token is generated. The model is trained using the cross-entropy loss and the total explanation loss $\mathcal { L } _ { e }$ is defined as follows:

$$
\mathcal { L } _ { e } = \frac { 1 } { | \tau | } \sum _ { ( u , i ) \in \tau } \frac { 1 } { | S _ { u , i } | } \sum _ { t = 1 } ^ { | S _ { u , i } | } - \log p ( y _ { t } )\tag{10}
$$

where τ, and $S _ { u , i }$ are the training set and the ground-truth explanation for user-item pair $( u , i )$

## G. Multi-task Learning

The core tasks of our framework are rating prediction and explanation generation. The individual training task ignores the influence of personalization in the explanation generation [15]. Thus, we propose a multi-task learning objective function that trains all the tasks simultaneously.

$$
O = \lambda _ { 1 } \mathcal { L } _ { r } + \lambda _ { 2 } \mathcal { L } _ { i , j } + \lambda _ { 3 } \mathcal { L } _ { e } + \lambda _ { n } \| \Theta \| ^ { 2 }\tag{11}
$$

where Θ is the set of model parameters, and $\lambda _ { 1 } , \lambda _ { 2 }$ , and $\lambda _ { 3 }$ are regularization weights for user-item representation, aspect learning, and explanation generation, respectively. The regularization strength is controlled by the hyperparameter $\lambda _ { n } .$

## IV. EXPERIMENTS

## A. Dataset

Three real-world datasets from different domains are used to evaluate the proposed model. Hotel ratings and reviews are collected from TripAdvisor<sup>1</sup>. In this work, we use one from the Hong Kong (HK) region [12]. For the restaurant domain, the Yelp Challenge 2019 dataset<sup>2</sup> is used. Movie review dataset is sourced from Amazon’s 5-core Movies & TV dataset (MT)<sup>3</sup>. We filter each dataset by removing users and items with fewer than 20 interactions to ensure sufficient information. Each record includes a user ID, item ID, rating (1–5), explanation, item feature, and textual review. For explanation generation, one sentence per review containing a feature is selected as the ground-truth. Key statistics are shown in Table II.

TABLE II STATISTICS OF DATASETS.
<table><tr><td></td><td>TA-HK</td><td>YELP19</td><td>AZ-MT</td></tr><tr><td># of users</td><td>9,765</td><td>27,147</td><td>7,506</td></tr><tr><td># of items</td><td>6,280</td><td>20,266</td><td>7,360</td></tr><tr><td># of reviews</td><td>320,023</td><td>1,293,247</td><td>441,783</td></tr><tr><td># of features</td><td>5,069</td><td>7,340</td><td>5,399</td></tr><tr><td>Avg. # of reviews / user</td><td>32.77</td><td>47.64</td><td>58.86</td></tr><tr><td>Avg. # of reviews / item</td><td>50.96</td><td>63.81</td><td>60.02</td></tr><tr><td> $\operatorname { A v g } .$  # of words / explanation</td><td>13.01</td><td>12.32</td><td>14.14</td></tr></table>

TA and AZ denote TripAdvisor and Amazon, respectively.

TABLE III  
PERFORMANCE ON RATING PREDICTION USING MAE AND RMSE
<table><tr><td rowspan="2"></td><td colspan="2">Yelp</td><td colspan="2">Amazon</td><td rowspan="2"></td><td colspan="2">TripAdvisor</td></tr><tr><td>RMSE ↓</td><td>MAE ↓</td><td>RMSE ↓</td><td>MAE ↓</td><td>RMSE ↓</td><td>MAE ↓</td></tr><tr><td>PMF</td><td>1.097</td><td>0.883</td><td>1.235</td><td>0.913</td><td></td><td>0.870</td><td>0.704</td></tr><tr><td>NARRE</td><td>1.028</td><td>0.791</td><td>1.176</td><td>0.865</td><td></td><td>0.796</td><td>0.612</td></tr><tr><td>DAML</td><td>1.014</td><td>0.784</td><td>1.173</td><td>0.858</td><td></td><td>0.793</td><td>0.617</td></tr><tr><td>RGCL</td><td>1.008</td><td>0.784</td><td>1.160</td><td>0.872</td><td></td><td>0.791</td><td>0.611</td></tr><tr><td>NRT</td><td>1.016</td><td>0.796</td><td>1.188</td><td>0.853</td><td></td><td>0.797</td><td>0.611</td></tr><tr><td>CAML</td><td>1.036</td><td>0.798</td><td>1.191</td><td>0.888</td><td></td><td>0.818</td><td>0.622</td></tr><tr><td>PETER</td><td>1.017</td><td>0.793</td><td>1.181</td><td>0.863</td><td></td><td>0.814</td><td>0.635</td></tr><tr><td>SERMON</td><td>1.002</td><td>0.784</td><td>1.159</td><td>0.837</td><td></td><td>0.791</td><td>0.599</td></tr><tr><td>CLARER</td><td>0.716</td><td>0.509</td><td>0.999</td><td>0.700</td><td></td><td>0.642</td><td>0.493</td></tr></table>

Bold values indicate the best performance.

## B. Evaluation Metrics

We evaluate rating prediction using RMSE and MAE, which measure the difference between the predicted and actual ratings. For explanation generation, we use BLEU-1, BLEU-4, ROUGE-2, and ROUGE-L (F1 scores) to assess similarity with ground truth text. Higher scores indicate better alignment.

## C. Baseline Methods

Performances of each of the tasks are compared with the state-of-the-art models discussed below.

1) Rating Prediction: PMF [38] is a probabilistic matrix factorization model that learns the latent factors of users and items. Both NARRE [21] and DAML [22] leverage textual information to learn the user and item representations by modeling the user and item documents. RGCL [39] advances the user-item representation by exploiting the advantage of graph contrastive learning, demonstrating superior performance over several state-of-the-art models. NRT [16], CAML [40], PETER [41], and SERMON [13] focus on both rating prediction and explanation generation; therefore, we discuss them in the following sub-section to avoid repetition.

2) Explanation Generation: NRT [16] uses deep learning to learn the user and item representation from IDs, leveraging them for rating prediction and tip generation. CAML [40] uses historical reviews of users to represent their preferences and employs co-attention mechanisms to identify the most relevant reviews and concepts, integrating these concepts to generate explanation texts. ReXPlug [42] takes advantage of the large language model, e.g., GPT-2 to generate explanation text while predicting the ratings. Similarly, PETER [41] is a transformerbased explanation generation model that incorporates user and item IDs as tokens to the input sequence, demonstrating enhanced performance. Following that, PEPLER [33] builds on a pre-trained transformer model to produce explainable recommendations by using prompts that combine user and item ID vectors. To bridge the gap between these prompts and the pre-trained model, the method introduces sequential tuning and recommendation-based regularization techniques. SERMON [13] is a powerful explanation generation model that considers aspects as context when generating explanations.

## D. Experimental details and Reproducibility

We choose the all-MiniLM-L6-v2 Sentence Transformer due to its lightweight architecture and competitive performance, and its 384-dimensional embeddings offers low inference latency. The multitask weights (1.0 for explanation and contrastive tasks, 0.5 for rating prediction) are empirically set to prioritize tasks with more complex objectives while maintaining contribution from all learning signals. The feedforward layer configuration of [256 → 128 → 64], inspired by [43], provides a compact yet expressive projection space for joint representations. We adopt 6 Transformer layers and 8 attention heads to ensure sufficient model capacity without incurring excessive computational cost. The hidden dimension of 1024 is selected based on standard practice for mini versions of the Transformer model, ensuring compatibility with pretrained components.

Adam optimizer with a learning rate of 0.001 and L2 regularization coefficient of 0.1 is chosen based on its robust convergence properties in similar multi-objective learning. The batch size is selected as 64 to balance memory usage and convergence stability, and early stopping (patience of 3 epochs) is used to prevent overfitting and reduce unnecessary computation.

## V. RESULTS

## A. Rating Prediction Accuracy

Table III presents the performance of our proposed CLARER model in rating prediction tasks, compared to models that focus exclusively on rating prediction, as well as those that jointly address rating prediction and explanation generation problems. Our model consistently outperforms both types of models in terms of RMSE and MAE across all three datasets, achieving the best performance in all evaluations. The results show that our model achieves the highest performance improvement over the latest state-of-the-art model, SERMON [13], in the largest data set (Yelp), reducing RMSE by 29% and MAE by 35%, demonstrating substantial gains in the accuracy of rating prediction. The smallest improvement is observed in the TripAdvisor dataset, with 14% lower RMSE and 16% lower MAE compared to SERMON, while the Amazon dataset shows 19% and 18% reductions in RMSE and MAE, respectively. These results highlight our model’s effectiveness across different domains, with particularly strong performance on larger datasets.

The results also indicate that the prediction errors from review-based models (NARRE, DAML, RGCL) are much lower than rating-based models (PMF), demonstrating the effectiveness of review features. Furthermore, the recent explainable recommendation model beats both the rating- and review-based baselines in the rating prediction tasks, indicating the effectiveness of the advanced methodologies they adopted.

TABLE IV  
PERFORMANCE COMPARISON ON EXPLANATION GENERATION TASKS USING BLEU AND ROUGE SCORES
<table><tr><td rowspan="2">Models</td><td colspan="4">Yelp</td><td colspan="4">TripAdvisor</td><td colspan="4">Amazon</td></tr><tr><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td></tr><tr><td>NRT</td><td>10.5</td><td>0.67</td><td>1.35</td><td>12.25</td><td>15.78</td><td>0.85</td><td>1.90</td><td>12.25</td><td>13.37</td><td>1.44</td><td>1.97</td><td>10.77</td></tr><tr><td>CAML</td><td>9.91</td><td>0.56</td><td>1.25</td><td>12.39</td><td>14.43</td><td>0.86</td><td>1.92</td><td>12.39</td><td>11.19</td><td>1.12</td><td>1.24</td><td>8.11</td></tr><tr><td>ReXPlug</td><td>8.59</td><td>0.57</td><td>1.11</td><td>9.97</td><td>12.64</td><td>0.71</td><td>1.61</td><td>9.97</td><td>10.80</td><td>1.29</td><td>1.22</td><td>8.73</td></tr><tr><td>PETER</td><td>10.29</td><td>0.69</td><td>1.43</td><td>12.61</td><td>15.33</td><td>0.89</td><td>1.94</td><td>12.61</td><td>13.78</td><td>1.68</td><td>1.97</td><td>11.07</td></tr><tr><td>PEPLER</td><td>10.42</td><td>0.73</td><td>1.46</td><td>11.64</td><td>15.62</td><td>1.09</td><td>2.22</td><td>11.64</td><td>13.78</td><td>1.05</td><td>2.22</td><td>13.19</td></tr><tr><td>SERMON</td><td>10.66</td><td>0.76</td><td>1.60</td><td>10.87</td><td>16.69</td><td>1.18</td><td>2.49</td><td>13.63</td><td>14.13</td><td>1.92</td><td>2.74</td><td>12.29</td></tr><tr><td>CLARER</td><td>12.60</td><td>7.16</td><td>7.90</td><td>14.31</td><td>11.52</td><td>4.79</td><td>4.53</td><td>11.70</td><td>14.46</td><td>1.12</td><td>5.12</td><td>12.30</td></tr></table>

![](images/5f0cea161a97890efca0a521cf48deaa90f7c818bd446dfeaf92a116fe3759a6.jpg)  
Fig. 2. MAE comparison across CLARER variants

![](images/02ed446fe088465270682bb6fe35e879a91ef5641f82ebb4db5d84f8371b68dd.jpg)  
Fig. 3. ROUGE score comparison across CLARER variants

## B. Explanation Generation

The performance of CLARER in explanation generation tasks is presented in Table IV. Our CLARER model demonstrates strong and consistent performance across three datasets when evaluated using BLEU1, BLEU4, ROUGE-2 (R2), and ROUGE-L (RL) metrics. In general, CLARER outperforms the selected baselines, achieving the highest scores in most metrics and datasets, especially excelling in the ROUGEbased metric (e.g., R2), which emphasizes the fluency and structural similarity of generated explanations. For example, CLARER achieves the best R2 scores in all three datasets, the best RL score on Yelp, and the second-best RL score on Amazon, indicating its strength in generating contextually rich and coherent explanations. CLARER outperforms all baselines in all evaluation metrics in the Yelp datasets. It also achieves the highest BLEU1 scores in the Amazon datasets and BLEU4 in the TripAdvisor datasets.

## C. Ablation Studies

We perform ablation studies on both rating prediction and explanation generation tasks by evaluating two variants of our proposed framework. The first variant, CLARER-O, serves as the base model, which utilizes overall rating-based and reviewbased features without considering criteria or aspects for user and item representations. The second variant, CLARER-C, extends this by incorporating criteria-level rating features alongside the overall ratings to improve the quality of the representation. The original CLARER model is the most comprehensive, which further integrates aspect-based features with existing rating-based representations, aiming to capture more fine-grained user preferences and item characteristics for improved performance. To implement CLARER-O, we keep the MLP-based rating prediction component and the review feature component, removing the upper part (anything above the review feature) shown in Figure 1. To implement CLARER-C, we combine criteria ratings with overall ratings for rating prediction (no aspect representation with contrastive learning, as shown in Figure 1). Finally, for CLARER, we replace the criteria ratings with aspects as input to the contrastive learning component.

For the Amazon and Yelp datasets, criteria-level ratings are not available; therefore, we evaluated only the overall rating-based and aspect-based feature representations for these datasets. As shown in Figure 2, across the three datasets, incorporating aspect-based representations consistently leads to lower MAE compared to the other two variants, highlighting the effectiveness of our proposed aspect representation learning method. Specifically, CLARER achieves the lowest MAE values of 0.49, 0.70, and 0.51 on the TripAdvisor, Amazon, and Yelp datasets, respectively. In contrast, using only overall rating features results in the highest prediction errors. When criteria-level ratings are available (on TripAdvisor), integrating them also leads to a lower MAE, confirming their usefulness, although they remain slightly less effective than aspect-based features. Similarly, as shown in Figure 3 for explanation generation, ROUGE scores (RL) indicate that our proposed CLARER outperforms the other two variants. The highest ROUGE score is achieved on the Yelp dataset, while the lowest is reported on the TripAdvisor dataset from CLARER-O.

TABLE V  
ABLATION STUDIES OF DIFFERENT FEATURE-BASED PERFORMANCE ON EXPLANATION GENERATION
<table><tr><td rowspan="2">Models</td><td colspan="4">Yelp</td><td colspan="4">TripAdvisor</td><td colspan="4">Amazon</td></tr><tr><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td><td>BLEU-1</td><td>BLEU-4</td><td>R-2</td><td>R-L</td></tr><tr><td>CLARER-O</td><td>12.94</td><td>7.12</td><td>7.68</td><td>14.29</td><td>11.33</td><td>4.69</td><td>4.52</td><td>11.64</td><td>11.32</td><td>4.05</td><td>4.56</td><td>12.03</td></tr><tr><td>CLARER-C</td><td></td><td></td><td></td><td></td><td>9.77</td><td>1.83</td><td>1.35</td><td>9.42</td><td></td><td></td><td></td><td></td></tr><tr><td>CLARER</td><td>12.60</td><td>7.16</td><td>7.90</td><td>14.31</td><td>11.52</td><td>4.79</td><td>4.53</td><td>11.70</td><td>14.46</td><td>4.47</td><td>5.12</td><td>12.30</td></tr></table>

Table V presents more detailed comparative results of the ablation study on explanation generation in three data sets. Performance is measured using BLEU scores, which evaluate n-gram overlaps, as well as ROUGE-2 and ROUGE-L scores. We can see that the original CLARER model outperforms the two variants in most cases across three datasets (only slightly worse on BLEU1 for Yelp), suggesting that the incorporation of aspect-level information plays a crucial role in enhancing the fluency, relevance, and overall quality of the explanations.

## D. Discussion

Using contrastive learning for aspect-based representation, CLARER effectively captures nuanced user and item features, resulting in more accurate and personalized predictions. The contrastive objective encourages the model to learn discriminative and aspect-aware review embeddings, which, in turn, facilitates a more precise alignment between user preferences and item features. This aspect-centric alignment is critical for improving recommendation quality in multi-criteria settings.

Despite its strong performance in rating prediction, CLARER does not consistently outperform all baselines in the explanation generation task across the three datasets. In the Yelp dataset, CLARER achieves the best performance across all evaluation metrics, demonstrating the strength of its aspect-aware representations. These results suggest that the effectiveness of aspect-based modeling is domain-dependent and that careful selection of aspect terms is critical for accurate user and item representation learning.

BLEU-1 scores are lower on the TripAdvisor dataset, likely because the reviews in this dataset are longer and more detailed, using a wider variety of words (e.g., talking about service, location, cleanliness, etc.). BLEU-4 scores are lower in the Amazon dataset, possibly because reviews in this dataset cover many different product types, making it harder for models to generate exact 4-word sequences that match the reference. ROUGE-L scores are moderate in the TripAdvisor dataset, possibly because the models find it challenging to follow the structured style of the reviews in this dataset. For Amazon, the big differences in review length and writing style between categories make it even harder for models to match the structure of the reference explanations, leading to lower ROUGE-L scores. Future work could explore alternative sets of aspects tailored to specific domains/datasets to enhance generalization. In addition, selecting a more advanced embedding model for aspect representation - potentially a larger transformer-based encoder - could further improve the quality of generated explanations.

## VI. CONCLUSION

In this paper, we introduce CLARER, a novel recommendation framework that combines user- and item-ID-based features with aspect-based review representation to improve both recommendation accuracy and explanation generation. Experimental results demonstrate that CLARER outperforms existing state-of-the-art recommendation models across multiple benchmark datasets, highlighting its ability to generate precise and interpretable recommendations. Furthermore, the model’s contrastive learning framework significantly improves the robustness of representations, reducing noise, and enhancing generalization.

Our work contributes to the growing body of research on explainable AI in recommendation systems by providing users with meaningful and interpretable justifications for recommendations, enhancing the trust and transparency of the system. A potential future direction could be the integration of multimodal data sources, such as images and user interaction logs, which may further refine the quality of the recommendation and enrich the interpretability.

## VII. ACKNOWLEDGMENT

The authors sincerely appreciate the reviewers for their invaluable efforts and constructive feedback.

[1] X. He, L. Liao, H. Zhang, L. Nie, X. Hu, and T.-S. Chua, “Neural collaborative filtering,” in Proceedings of the 26th international conference on world wide web, 2017, pp. 173–182.

[2] B. Sarwar, G. Karypis, J. Konstan, and J. Riedl, “Item-based collaborative filtering recommendation algorithms,” in Proceedings of the 10th international conference on World Wide Web, 2001, pp. 285–295.

[3] S. Zhang, L. Yao, A. Sun, and Y. Tay, “Deep learning based recommender system: A survey and new perspectives,” ACM computing surveys (CSUR), vol. 52, no. 1, pp. 1–38, 2019.

[4] L. Zheng, V. Noroozi, and P. S. Yu, “Joint deep modeling of users and items using reviews for recommendation,” in Proceedings of the tenth ACM Int. Conf. on Web Search and Data Mining, 2017, pp. 425–434.

[5] Y. Xu, L. Zhu, Z. Cheng, J. Li, Z. Zhang, and H. Zhang, “Multi-modal discrete collaborative filtering for efficient cold-start recommendation,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 1, pp. 741–755, 2021.

[6] Y. Zhang, X. Chen et al., “Explainable recommendation: A survey and new perspectives,” Foundations and Trends® in Information Retrieval, vol. 14, no. 1, pp. 1–101, 2020.

[7] N. Tintarev and J. Masthoff, “Explaining recommendations: Design and evaluation,” in Recommender systems handbook. Springer, 2015, pp. 353–382.

[8] Y. Zhang, G. Lai, M. Zhang, Y. Zhang, Y. Liu, and S. Ma, “Explicit factor models for explainable recommendation based on phrase-level sentiment analysis,” in Proceedings ofthe 37th international ACM SIGIR conference on Research & development in IR, 2014, pp. 83–92.

[9] L. Li, L. Chen, and R. Dong, “Caesar: context-aware explanation based on supervised attention for service recommendations,” Journal of Intelligent Info. Sys., vol. 57, no. 1, pp. 147–170, 2021.

[10] X. Chen, H. Chen, H. Xu, Y. Zhang, Y. Cao, Z. Qin, and H. Zha, “Personalized fashion recommendation with visual explanations based on multimodal attention network: Towards visually explainable recommendation,” in Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2019, pp. 765–774.

[11] H. Chen, X. Chen, S. Shi, and Y. Zhang, “Generate natural language explanations for recommendation,” arXiv preprint arXiv:2101.03392, 2021.

[12] L. Li, Y. Zhang, and L. Chen, “Generate neural template explanations for recommendation,” in Proceedings of the 29th ACM International Conf. on Information & Knowledge Management, 2020, pp. 755–764.

[13] H. Liao, S. Wang, H. Cheng, W. Zhang, J. Zhang, M. Zhou, K. Lu, R. Mao, and X. Xie, “Aspect-enhanced explainable recommendation with multi-modal contrastive learning,” ACM Transactions on Intelligent Systems and Technology, vol. 16, no. 1, pp. 1–24, 2025.

[14] N. Wang, H. Wang, Y. Jia, and Y. Yin, “Explainable recommendation via multi-task learning in opinionated text data,” in The 41st international ACM SIGIR conference on research & development in information retrieval, 2018, pp. 165–174.

[15] W. Ma, M. Zhang, Y. Cao, W. Jin, C. Wang, Y. Liu, S. Ma, and X. Ren, “Jointly learning explainable rules for recommendation with knowledge graph,” in The world wide web conference, 2019, pp. 1210–1221.

[16] P. Li, Z. Wang, Z. Ren, L. Bing, and W. Lam, “Neural rating regression with abstractive tips generation for recommendation,” in Proceedings of the 40th International ACM SIGIR conference on Research and Development in Information Retrieval, 2017, pp. 345–354.

[17] L. Dong, S. Huang, F. Wei, M. Lapata, M. Zhou, and K. Xu, “Learning to generate product reviews from attributes,” in 15th EACL 2017 Software Demonstrations. ACL, 2017, pp. 623–632.

[18] S. Natarajan, S. Vairavasundaram, S. Natarajan, and A. H. Gandomi, “Resolving data sparsity and cold start problem in collaborative filtering recommender system using linked open data,” Expert Systems with Applications, vol. 149, p. 113248, 2020.

[19] T.-H. Le and H. W. Lauw, “Synthesizing aspect-driven recommendation explanations from reviews.” IJCAI, 2020.

[20] S. Seo, J. Huang, H. Yang, and Y. Liu, “Interpretable convolutional neural networks with dual local and global attention for review rating prediction,” in Proceedings of the eleventh ACM conference on recommender systems, 2017, pp. 297–305.

[21] C. Chen, M. Zhang, Y. Liu, and S. Ma, “Neural attentional rating regression with review-level explanations,” in Proceedings of the 2018 world wide web conference, 2018, pp. 1583–1592.

[22] D. Liu, J. Li, B. Du, J. Chang, and R. Gao, “Daml: Dual attention mutual learning between ratings and reviews for item recommendation,” in Proceedings of the 25th ACM SIGKDD international conference on knowledge discovery & data mining, 2019, pp. 344–352.

[23] X. Dong, J. Ni, W. Cheng, Z. Chen, B. Zong, D. Song, Y. Liu, H. Chen, and G. De Melo, “Asymmetrical hierarchical networks with attentive interactions for interpretable review-based recommendation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 34, no. 05, 2020, pp. 7667–7674.

[24] H. Xia, Z. Wang, B. Du, L. Zhang, S. Chen, and G. Chun, “Leveraging ratings and reviews with gating mechanism for recommendation,” in Proceedings of the 28th ACM international conference on information and knowledge management, 2019, pp. 1573–1582.

[25] Q. Peng, H. Liu, Y. Yu, H. Xu, W. Dai, and P. Jiao, “Mutual self attention recommendation with gated fusion between ratings and reviews,” in Database Systems for Advanced Applications: 25th International Conference, DASFAA 2020, Jeju, South Korea, September 24–27, 2020, Proceedings, Part III 25. Springer, 2020, pp. 540–556.

[26] Y. Lu, R. Dong, and B. Smyth, “Coevolutionary recommendation model: Mutual learning between ratings and reviews,” in Proceedings of the 2018 World Wide Web Conference, 2018, pp. 773–782.

[27] Y. Bao, H. Fang, and J. Zhang, “Topicmf: Simultaneously exploiting ratings and reviews for recommendation,” in Proceedings of the AAAI conference on artificial intelligence, vol. 28, no. 1, 2014.

[28] D. D. Lee and H. S. Seung, “Learning the parts of objects by nonnegative matrix factorization,” nature, vol. 401, no. 6755, pp. 788–791, 1999.

[29] J. McAuley and J. Leskovec, “Hidden factors and hidden topics: understanding rating dimensions with review text,” in Proceedings of the 7th ACM conference on Recommender systems, 2013, pp. 165–172.

[30] Q. Diao, M. Qiu, C.-Y. Wu, A. J. Smola, J. Jiang, and C. Wang, “Jointly modeling aspects, ratings and sentiments for movie recommendation (jmars),” in Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining, 2014, pp. 193–202.

[31] Z. Cheng, Y. Ding, L. Zhu, and M. Kankanhalli, “Aspect-aware latent factor model: Rating prediction with ratings and reviews,” in Proceedings of the 2018 world wide web conference, 2018, pp. 639–648.

[32] L. Chen and F. Wang, “Explaining recommendations based on feature sentiments in product reviews,” in Proceedings of the 22nd international conference on intelligent user interfaces, 2017, pp. 17–28.

[33] L. Li, Y. Zhang, and L. Chen, “Personalized prompt learning for explainable recommendation,” ACM Transactions on Information Systems, vol. 41, no. 4, pp. 1–26, 2023.

[34] F. Tang, Y. Shen, H. Zhang, Z. Tan, W. Zhang, G. Hou, K. Song, W. Lu, and Y. Zhuang, “Gavamoe: Gaussian-variational gated mixture of experts for explainable recommendation,” arXiv preprint arXiv:2410.11841, 2024.

[35] P. H. Le-Khac, G. Healy, and A. F. Smeaton, “Contrastive representation learning: A framework and review,” Ieee Access, vol. 8, pp. 193 907– 193 934, 2020.

[36] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” arXiv preprint arXiv:1908.10084, 2019.

[37] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[38] A. Mnih and R. R. Salakhutdinov, “Probabilistic matrix factorization,” Advances in neural information processing systems, vol. 20, 2007.

[39] J. Shuai, K. Zhang, L. Wu, P. Sun, R. Hong, M. Wang, and Y. Li, “A review-aware graph contrastive learning framework for recommendation,” in Proceedings of the 45th international ACM SIGIR conference on research and development in IR, 2022, pp. 1283–1293.

[40] S. Kang, W. Kweon, D. Lee, J. Lian, X. Xie, and H. Yu, “Distillation from heterogeneous models for top-k recommendation,” in Proceedings of the ACM Web Conference 2023, 2023, pp. 801–811.

[41] L. Li, Y. Zhang, and L. Chen, “Personalized transformer for explainable recommendation,” arXiv preprint arXiv:2105.11601, 2021.

[42] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever et al., “Language models are unsupervised multitask learners,” OpenAI blog, vol. 1, no. 8, p. 9, 2019.

[43] E. Hasan, C. Ding, and A. Cuzzocrea, “Multi-criteria rating and review based recommendation model,” in 2022 IEEE International Conference on Big Data (Big Data). IEEE, 2022, pp. 5494–5503.