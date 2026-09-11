# EXPLORING SECOND-ORDER PATTERN RECOGNITION IN SPEAKER RECOGNITION

Yanze Xu<sup>1</sup>, Wenwu Wang<sup>1</sup>, Mark D. Plumbley<sup>2</sup>

<sup>1</sup>Centre for Vision, Speech and Signal Processing, University of Surrey, Guildford, UK <sup>2</sup>Department of Informatics, King’s College London, London, UK

## ABSTRACT

In classical pattern recognition tasks, neural networks are trained to recognise human-defined patterns for model inputs. Some Explainable AI (XAI) methods can explain other latent patterns that underlie the network’s recognition of inputs as human-defined patterns; in this work, we call these latent patterns second-order patterns, and we propose to discover them. To this end, we apply a hierarchical clustering algorithm to analyse whether representations learned by a speaker recognition network from utterances naturally form hierarchical clusters. Each resulting cluster represents a second-order pattern that characterises how the network recognises some known utterances as speaker identities. All the resulting secondorder patterns are then semantically interpreted using the existing Hierarchical Cluster-Class Matching (HCCM) method.

Furthermore, we propose a new task, second-order pattern recognition, to identify which discovered second-order patterns characterising known utterances are exhibited by an unseen utterance. To achieve this, we design the Hierarchical Cluster Navigation and Assignment (HCNA) method. HCNA recognises a known second-order pattern as applying to an unseen utterance when the unseen utterance’s network representation lies within the extrapolation space of the cluster regarded as that second-order pattern. Our experiments show that the extrapolation mechanism introduced by HCNA substantially improves performance on the second-order pattern recognition task.

Index Terms— Explainable AI, Second-order Pattern Recognition, Hierarchical Clustering, Extrapolation Space, Gradient Cost

## 1. INTRODUCTION

Classical pattern recognition [1] aims to discriminate and recognise human-defined patterns in data. Pattern recognition tasks (e.g. image classification and speaker recognition) are usually implemented by training neural networks to learn a non-linear mapping from observed data directly to humandefined patterns [2, 3]. However, most neural networks operate as black boxes, posing risks when deployed in real-world applications. The field of explainable artificial intelligence (XAI) [4, 5] aims to explain the decision-making processes of trained models, especially neural networks. Some XAI techniques [6, 7] are designed to explain some latent patterns that underlie, or contribute to, how the network recognises inputs as human-defined patterns.

In cognitive psychology, a person’s belief about an object is regarded as a first-order belief, while another person’s belief about that person’s belief is regarded as a second-order belief [8]. Inspired by this, we regard the first-order belief of a network to be that the network “believes” that inputs can be recognised as human-defined patterns, while we regard the second-order beliefofan XAI technique as this technique “believes” that some latent patterns characterise the network’s first-order belief (i.e. recognising data as human-defined patterns). On this basis, we next define those human-defined patterns in the network’s first-order belief asfirst-order patterns, and those latent patterns in the XAI technique’s second-order belief as second-order patterns.

Two research questions are then asked. First, does the real world contain a large number of second-order patterns that remain to be jointly discovered and understood by humans, neural networks, and XAI techniques? Secondly, once a set of second-order patterns characterising the network’s recognition of known data as first-order patterns has been successfully discovered, can we automatically recognise which second-order patterns associated with the known data are exhibited by unseen data? We define this new recognition task as second-order pattern recognition.

This work aims to realise such a second-order pattern recognition task based on a speaker recognition network [9, 3]. Specifically, we first extract latent representations (i.e. speaker embeddings) learned by a trained speaker recognition network from a group of known utterances. We next apply a hierarchical clustering algorithm [10], Single-Linkage Clustering (SLINK) [11, 12], which has been introduced into the XAI domain [4], to analyse whether the speaker embeddings naturally form clusters with hierarchical relationships. Each resulting cluster is a second-order pattern since it is what SLINK “believes” at the second-order level characterises how the network discriminates and recognises speaker identities of known utterances.

The above second-order patterns (i.e. hierarchical representation clusters) discovered can be made understandable to humans through semantic interpretation. In particular, we apply the existing Hierarchical Cluster-Class Matching (HCCM) method [4] to match the resulting hierarchical representation clusters with predefined semantic classes in a one-to-one manner. Each cluster is semantically interpreted according to the semantic class with which it is best matched, thereby simultaneously providing a semantic interpretation of the second-order pattern represented by that cluster.

Moreover, the discovered second-order patterns characterising our network’s recognition of the examined known utterances are treated as recognition targets in the second-order pattern recognition task to determine which of them apply to an unseen utterance. To achieve this, we propose the Hierarchical Cluster Navigation and Assignment (HCNA) method, which identifies applicable second-order patterns by determining which representation clusters of known utterances the unseen utterance’s network representation belongs to.

Following McInnes et al. [13], an unseen representation belongs to a cluster of known representations when it lies within the empirical space defined by the representations in that cluster. Instead of using the empirical space, HCNA estimates the extrapolation space [14] for each cluster, defined as the space to which the known representations in that cluster can generalise. To estimate this extrapolation space, HCNA introduces the Gradient Cost, which quantifies the gradient expenditure required to optimise a unit difference in loss function. The Gradient Cost is next used to adjust the cluster’s empirical space to obtain the extrapolation space, for checking whether an unseen representation lies within it.

Overall, the second-order pattern recognition task is operationalised primarily through the proposed HCNA, while its realisation also relies on SLINK to first discover second-order patterns and HCCM to subsequently interpret them. Hence, these three methods form a complete second-order pattern recognition system. Several evaluation metrics are introduced to assess this system. Experimental results show that introducing HCNA’s extrapolation mechanism substantially improves the system’s performance on the second-order pattern recognition task.

## 2. RELATED WORK

Although speaker embeddings are not understandable to humans, existing XAI studies infer the semantic information encoded in speaker embeddings by training downstream classifiers to map speaker embeddings to semantic classes related to gender, nationality, or phonetic [15, 16, 17]. Successful downstream classification is then taken to indicate that the corresponding semantic information is encoded in the speaker embeddings. In comparison, our work directly decodes the semantic information inside speaker embeddings by first analysing second-order patterns of embeddings (i.e. using SLINK) and then associating these patterns with semantic classes (i.e. using HCCM).

There are similar works applying clustering algorithms to network representations, treating the resulting clusters as pseudo-labels rather than second-order patterns, and then predicting whether an unseen input belongs to these pseudolabels [18, 19, 20]. In comparison, second-order patterns in this work are defined with respect to the second-order belief of an XAI technique, practically characterising the network’s decision, whereas the pseudo-labels in these related works are neither explicitly defined nor assigned a clear role. Moreover, second-order patterns are interpreted using HCCM in our work, whereas the pseudo-labels in these related works are often not understandable to humans.

## 3. METHODOLOGY

## 3.1. Discovering and Interpreting Second-Order Patterns

Let $f ( \cdot )$ denote a trained speaker recognition network, which maps an utterance x to a speaker embedding $a = f ( x )$ . This representation a is then used to determine the speaker identity. We define a set of known utterances $X \ = \ \{ x _ { i } \}$ , which is independent of the data used to train $f .$ . The network representations of X are denoted by $A = \left\{ a _ { i } \right\}$ , where $a _ { i } = f ( x _ { i } )$

We apply the hierarchical clustering algorithm [21], SLINK [11, 12], to representations A. SLINK analyses a natural hierarchical organisation underlying representations $A ,$ denoted by $\mathcal { H } = h _ { 1 } , h _ { 2 } , . . . ,$ where each $h _ { i }$ is an index set containing all representations belonging to the i-th hierarchical representation cluster. Moreover, every cluster analysed by SLINK has a sibling cluster, such that $h _ { i }$ characterises a context in which its representations are discriminated from the representations of $h _ { i } { \ ' } { \leqslant }$ s sibling cluster. In this sense, under the speaker recognition network’s first-order belief, the network “believes” that the input utterances should generate corresponding representations to be recognised as speaker identities, while under SLINK’s second-order belief, SLINK “believes” that $h _ { i }$ represents a latent pattern characterising how some representations are organised in order to realise the network’s first-order belief; $h _ { i }$ in SLINK’s second-order belief is regarded as a second-order pattern.

To provide semantic interpretations of the above hierarchical representation clusters regarded as second-order patterns, the Hierarchical Cluster-Class Matching (HCCM) method [4] is used to determine which predefined semantic classes (i.e. classes pre-labelled for the representations and their utterances) best match the hierarchical representation clusters under a one-to-one correspondence. Each hierarchical representation cluster is then interpreted by the predefined semantic class that achieves the highest matching degree with that cluster. As more predefined semantic classes become available, more clusters regarded as second-order patterns can be semantically interpreted. The matching degree between a cluster h and a semantic class $c ,$ where c is an index set containing all representations whose speaker identities belong to that semantic class, is quantified below using the

L-score [4]:

$$
L ( h , c ) = \frac { | h \cap c | } { \operatorname* { m a x } ( | h | , | c | ) } .\tag{1}
$$

Overall, SLINK discovers second-order patterns that contribute to recognising the speaker identities of some known utterances, while HCCM provides semantic interpretations for these second-order patterns. In the next section, we propose the Hierarchical Cluster Navigation and Assignment (HCNA) method to recognise whether the second-order patterns discovered from known utterances apply to an unseen input.

## 3.2. HCNA - Candidate Path

The HCNA method first identifies candidate second-order patterns that has potential to apply to an unseen utterance. For an unseen utterance $x _ { \mathrm { u n s e e n } }$ with representation $a _ { \mathrm { u n s e e n } } = f ( x _ { \mathrm { u n s e e n } } )$ , HCNA finds the index of the nearest known representation in A as follows:

$$
i ^ { * } = \arg \operatorname* { m i n } _ { i } \mathrm { d i s t } \big ( a _ { \mathrm { u n s e e n } } , a _ { i } \big ) ,\tag{2}
$$

where dist $( \cdot , \cdot )$ denotes the Euclidean distance metric. The corresponding representation $a _ { i ^ { * } }$ is therefore the known representation in A nearest to $a _ { \mathrm { u n s e e n } }$ . Since $a _ { i ^ { * } }$ has already participated in constructing the hierarchical representation organisation using SLINK, all clusters containing it form a nested path $h ^ { ( 1 ) } \breve { \to } h ^ { ( 2 ) } \to \cdots \to h ^ { ( m ) }$ , where m is the number of clusters containing $a _ { i ^ { * } } . \mathrm { ~ A l l }$ clusters (i.e. second-order patterns) on this path are treated as candidate clusters (i.e. candidate second-order patterns), which HCNA subsequently navigates one by one to recognise which of them actually apply to the unseen utterance x<sub>unseen</sub>.

Determining whether a candidate second-order pattern $h ^ { ( t ) }$ applies to $x _ { \mathrm { u n s e e n } }$ is equivalent to determining whether $a _ { \mathrm { u n s e e n } }$ belongs to the corresponding cluster, i.e. is a member of that cluster. Recall that, under SLINK, the distance between two sibling clusters is determined by the minimum distance between their representations. We define the birth distance of $h ^ { ( t ) }$ , denoted by $d _ { \mathrm { b i r t h } } ^ { ( t ) }$ , as the distance at which its parent cluster splits into $h ^ { ( t ) }$ and its sibling cluster. Following McInnes et al.’s method [13] for determining cluster membership without reclustering the data, the empirical space of $h ^ { ( t ) }$ is the space covered by its known representations within the range defined by $d _ { \mathrm { b i r t h } } ^ { ( t ) }$ . Accordingly, $x _ { \mathrm { u n s e e n } }$ is recognised as belonging to $h ^ { ( t ) }$ only if $a _ { \mathrm { u n s e e n } }$ lies within this empirical space, equivalently when dist $( a _ { i ^ { * } } , a _ { \mathrm { u n s e e n } } ) < d _ { \mathrm { b i r t h } } ^ { ( t ) }$   
1

## 3.3. HCNA - Gradient Cost and Extrapolation

HCNA offers a unique way to determine cluster membership for an unseen utterance’s representation, thereby confirming which candidate second-order patterns actually apply to the unseen utterance. HCNA examines whether the unseen representation lies within the extrapolation spaces of the candidate clusters, rather than within their empirical spaces.

We now compute the extrapolation space of a candidate cluster $h ^ { ( t ) }$ step by step. Specifically, we first introduce a new quantity termed Gradient Cost, defined as the amount of gradient adjustment required per unit difference in the loss function. As our speaker recognition network f is trained using the angular prototypical loss function [22] that minimising angular differences, we therefore formulate Gradient Cost as the amount of gradient adjustment required for optimisation per unit angular difference:

$$
G C _ { \mathrm { u n s e e n } } = \left\| \frac { \| a _ { \mathrm { u n s e e n } } \| _ { 2 } } { \theta _ { \mathrm { u n s e e n } } } \frac { \partial L _ { \mathrm { u n s e e n } } } { \partial a _ { \mathrm { u n s e e n } } } \right\| _ { 1 } ,\tag{3}
$$

where $L _ { \mathrm { u n s e e n } } ~ = ~ { \textstyle { \frac { 1 } { 2 } } } \theta _ { \mathrm { u n s e e n } } ^ { 2 }$ is the optimisation loss for reducing the angular difference between $a _ { \mathrm { u n s e e n } }$ and its nearest known representation $a _ { i ^ { * } }$ , and $\theta _ { \mathrm { u n s e e n } }$ denotes this angular difference, given by arccos $\left( { \frac { a _ { \mathrm { u n s e e n } } ^ { \top } a _ { i ^ { * } } } { \| a _ { \mathrm { u n s e e n } } \| _ { 2 } \| a _ { i ^ { * } } \| _ { 2 } } } \right)$ . The gradient $\frac { \partial L _ { \mathrm { u n s e e n } } } { \partial a _ { \mathrm { u n s e e n } } }$ indicates how $a _ { \mathrm { u n s e e n } }$ should be adjusted to reduce the angular difference. Dividing this gradient by $\theta _ { \mathrm { u n s e e n } }$ measures the gradient adjustment per unit angular difference, while multiplying by $\| a _ { \mathrm { u n s e e n } } \| _ { 2 }$ compensates for the scale of the representation. On this basis, we next quantify the Gradient-Cost Ratio as follows:

$$
G C R ^ { ( t ) } = \frac { G C _ { \mathrm { u n s e e n } } } { G C _ { \mathrm { b i r t h } } ^ { ( t ) } } ,\tag{4}
$$

where $G C _ { \mathrm { b i r t h } } ^ { ( t ) }$ is the Gradient Cost of integrating cluster $h ^ { ( t ) }$ with its sibling cluster, computed by minimising the angular difference between the closest pair of representations from these two clusters, which defines their birth distance. When $G C R ^ { ( t ) } > 1$ , meaning that integrating $a _ { \mathrm { u n s e e n } }$ into $h ^ { ( t ) }$ requires a greater Gradient Cost than integrating $h ^ { ( t ) }$ with its sibling cluster, we hypothesise that $a _ { \mathrm { u n s e e n } }$ is relatively difficult to integrate into $\bar { h } ^ { ( t ) }$ . Conversely, when $G C R ^ { ( t ) } < 1$ meaning that integrating $a _ { \mathrm { u n s e e n } }$ into $h ^ { ( t ) }$ requires a lower Gradient Cost than integrating $h ^ { ( t ) }$ with its sibling cluster, we hypothesise that $a _ { \mathrm { u n s e e n } }$ is relatively easy to integrate into $h ^ { ( t ) }$ . Under our hypothesis, $G C R ^ { ( t ) }$ therefore quantifies the relative difficulty of integrating $a _ { \mathrm { u n s e e n } }$ into $h ^ { ( t ) }$

Then, HCNA adjusts the birth distance of $h ^ { ( t ) } ( \mathrm { i . e . } d _ { \mathrm { b i r t h } } ^ { ( t ) } )$ according to $G C R ^ { ( t ) }$ to obtain an extrapolation distance $d _ { \mathrm { e x t r a } } ^ { ( t ) }$ as follows:

$$
d _ { \mathrm { e x t r a } } ^ { ( t ) } = \frac { d _ { \mathrm { b i r t h } } ^ { ( t ) } } { G C R ^ { ( t ) } } .\tag{5}
$$

A higher integration difficulty between $a _ { \mathrm { u n s e e n } }$ and $h ^ { ( t ) }$ contracts the empirical space of $\dot { h } ^ { ( t ) }$ in the direction of $a _ { \mathrm { u n s e e n } } .$ Conversely, a lower integration difficulty expands the empirical space in that direction. For a given $a _ { \mathrm { u n s e e n } } .$ , its extrapolation space of $h ^ { ( t ) }$ is defined as the space covered by all known representations in $h ^ { ( t ) }$ within $d _ { \mathrm { e x t r a } } ^ { ( t ) } .$ . Notably, in our formulation, the extrapolation space of $h ^ { ( t ) }$ is specific to each unseen representation, since a different $G C R ^ { ( t ) }$ and $d _ { \mathrm { e x t r a } } ^ { ( t ) }$ are calculated for each one.

Lastly, the unseen representation $a _ { \mathrm { u n s e e n } }$ is regarded as lying within $h ^ { ( t ) } \mathrm { s }$ extrapolation space if $d ( a _ { \mathrm { u n s e e n } } , a _ { i ^ { * } } ) \ <$ $d _ { \mathrm { e x t r a } } ^ { ( t ) }$ . If this condition is satisfied, the candidate secondorder pattern represented by cluster $h ^ { ( t ) }$ , including $h ^ { ( t ) } \mathrm { s }$ semantic interpretation provided by HCCM, is recognised as applying to the unseen utterance $x _ { \mathrm { u n s e e n } }$ . In this way, HCNA navigates all candidate second-order patterns for $x _ { \mathrm { u n s e e n } }$ to progressively identify all applicable ones.

## 4. EXPERIMENTS

## 4.1. Experimental Setup

The speaker recognition network used in our experiments is the ResNetSE34L model published by J. S. Chung et al. [22]. This model is trained on the VoxCeleb2 development set [3] using the angular prototypical loss [22]. We use the VoxCeleb1 test set [9] as known utterances for discovering second-order patterns, while using the VoxCeleb2 test set [3] as unseen samples. Among hierarchical representation clusters obtained using SLINK, those containing fewer than 2000 representations are excluded to limit the number of secondorder patterns. For unseen utterances in the VoxCeleb2 test set, predefined semantic classes related to speaker gender and nationality are collected from the Internet for use in HCCM.

## 4.2. Results

Table 1 compares the performance of the system (i.e. SLINK + HCCM + HCNA) on the second-order pattern recognition task under three HCNA extrapolation settings. HCNA w/o extrap. determines cluster membership based on the empirical space. HCNA GCR extrap. relies on the extrapolation space estimated using the GCR defined in Section 3.3. HCNA fixed extrap. determines cluster membership by extrapolating the empirical space using a fixed factor.

The first evaluation metric in Table 1, extrapolation magnitude, measures, under an extrapolation setting, the average extent to which the clusters regarded as second-order patterns are extrapolated beyond their empirical spaces. Path completeness measures, across all unseen representations, the average proportion of candidate clusters that are successfully traversed (i.e. recognised). Furthermore, some second-order patterns recognised as applicable to an unseen utterance carry the semantic classes interpreted by HCCM, such as individual semantic classes like UK and conjunctive semantic classes like UK&male [4]. The third metric, Atomic semantic recall, measures the proportion of all individual semantic classes pre-labelled for the unseen utterance that are recovered from the individual semantic classes involved in HCCM’s interpretations of second-order patterns. Gender atomic recall and nation atomic recall calculate the same metric using only individual classes related to gender and nationality, respectively.

Table 1. Evaluation of the second-order pattern recognition system under different HCNA extrapolation settings.
<table><tr><td>Metric</td><td>HCNA w/o extrap.</td><td>HCNA fixed extrap.</td><td>HCNA GCR extrap.</td></tr><tr><td>Extrap. magnitude</td><td></td><td>7.16%</td><td>7.16%</td></tr><tr><td>Path completeness</td><td>11.68%</td><td>24.05%</td><td>34.19%</td></tr><tr><td>Atomic recall</td><td>0.5119</td><td>0.5423</td><td>0.5669</td></tr><tr><td>Gender atomic recall</td><td>98.73%</td><td>98.58%</td><td>98.18%</td></tr><tr><td>Nation atomic recall</td><td>3.64%</td><td>9.88%</td><td>15.20%</td></tr></table>

As shown in Table 1, path completeness and atomic semantic recall increase from 11.68% and 0.5119 without an extrapolation mechanism, to 24.05% and 0.5423 with a fixed extrapolation magnitude of 7.16%, and then to 34.19% and 0.5669 with GCR-based extrapolation of the same 7.16% magnitude. Gender atomic recall remains approximately 98%, whereas nation atomic recall increases from 3.64% to 9.88% and 15.20%, with fixed and GCR-based extrapolation yielding the latter two values, separately. These observations demonstrate benefits of adaptively adjusting the extrapolation spaces of clusters using the GCR, rather than uniformly enlarging their empirical spaces or leaving them unchanged.

## 5. CONCLUSION

This work proposes a new task, called second-order pattern recognition, built upon speaker recognition. In this task, second-order patterns that XAI techniques “believe” characterise how the network discriminates and recognises first-order patterns (i.e. speaker identities) from known utterances are used to determine which of them apply to unseen utterances. A baseline system integrating SLINK, HCCM, and HCNA is developed, with SLINK and HCCM used to discover and interpret second-order patterns, and HCNA used to subsequently recognise their applicability to unseen utterances. The proposed HCNA introduces Gradient Cost and Gradient Cost Ratio (GCR) to estimate the extrapolation spaces of representation clusters in order to determine whether the representation of an unseen utterance lies within these spaces, thereby identifying the applicable second-order patterns. Experimental results show that HCNA’s GCR-based extrapolation improves performance on the second-order pattern recognition task, increasing path completeness from 11.68% to 34.19% and atomic semantic recall from 0.5119 to 0.5669. To show the benefit of adaptively extrapolating the empirical spaces of clusters using the GCR rather than a fixed scale, we also report that GCR-based extrapolation outperforms fixed extrapolation under the same extrapolation magnitude of 7.16%.

## 6. REFERENCES

[1] Anil K. Jain, Robert P. W. Duin, and Jianchang Mao, “Statistical pattern recognition: A review,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 1, pp. 4–37, 2000.

[2] Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton, “ImageNet classification with deep convolutional neural networks,” Advances in Neural Information Processing Systems, vol. 25, 2012.

[3] Joon Son Chung, Arsha Nagrani, and Andrew Zisserman, “VoxCeleb2: Deep Speaker Recognition,” in Interspeech, 2018, pp. 1086–1090.

[4] Yanze Xu, Wenwu Wang, and Mark D. Plumbley, “Explainable AI in Speaker Recognition–Making Latent Representations Understandable,” arXiv preprint arXiv:2604.23354, 2026.

[5] Yanze Xu, Mark D. Plumbley, and Wenwu Wang, “Explainable AI in Speaker Recognition–Attention Map Visualisation and Evaluation,” arXiv preprint arXiv:2606.22901, 2026.

[6] Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba, “Learning deep features for discriminative localization,” in Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, 2016, pp. 2921–2929.

[7] Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra, “Grad-cam: Visual explanations from deep networks via gradient-based localization,” in Proceedings of the IEEE international conference on computer vision, 2017, pp. 618–626.

[8] Josef Perner and Heinz Wimmer, ““John thinks that Mary thinks that. . . ” attribution of second-order beliefs by 5-to 10-year-old children,” Journal of Experimental Child Psychology, vol. 39, no. 3, pp. 437–471, 1985.

[9] Arsha Nagrani, Joon Son Chung, and Andrew Zisserman, “VoxCeleb: A Large-Scale Speaker Identification Dataset,” in Interspeech, 2017, pp. 2616–2620.

[10] Daniel Mullner, “Modern hierarchical, agglomerative¨ clustering algorithms,” arXiv preprint arXiv:1109.2378, 2011.

[11] John C Gower and Gavin JS Ross, “Minimum spanning trees and single linkage cluster analysis,” Journal of the Royal Statistical Society: Series C (Applied Statistics), vol. 18, no. 1, pp. 54–64, 1969.

[12] Robin Sibson, “SLINK: An optimally efficient algorithm for the single-link cluster method,” The Computer Journal, vol. 16, no. 1, pp. 30–34, 1973.

[13] Leland McInnes, John Healy, and Steve Astels, “HDB-SCAN: Hierarchical density based clustering,” The Journal of Open Source Software, vol. 2, no. 11, pp. 205, 2017.

[14] Pamela J. Haley and Donald Soloway, “Extrapolation limitations of multilayer feedforward neural networks,” in International Joint Conference on Neural Networks (IJCNN). 1992, vol. 4, pp. 25–30, IEEE.

[15] Xiaoliang Wu, Chau Luu, Peter Bell, and Ajitha Rajan, “Explainable attribute-based speaker verification,” arXiv preprint arXiv:2405.19796, 2024.

[16] Chau Luu, Peter Bell, and Steve Renals, “Leveraging speaker attribute information using multi task learning for speaker verification and diarization,” arXiv preprint arXiv:2010.14269, 2020.

[17] Chau Luu, Steve Renals, and Peter Bell, “Investigating the contribution of speaker attributes to speaker separability using disentangled speaker representations,” in Interspeech. 2022, pp. 610–614, ISCA.

[18] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze, “Deep clustering for unsupervised learning of visual features,” in Proceedings of the European Conference on Computer Vision (ECCV), 2018.

[19] Thomas Dopierre, Christophe Gravier, Julien Subercaze, and Wilfried Logerais, “Few-shot pseudo-labeling for intent detection,” in Proceedings ofthe 28th International Conference on Computational Linguistics. 2020, pp. 4993–5003, International Committee on Computational Linguistics.

[20] Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool, “SCAN: Learning to classify images without labels,” in Proceedings of the European Conference on Computer Vision (ECCV), 2020, pp. 268–285.

[21] Fionn Murtagh and Pedro Contreras, “Algorithms for hierarchical clustering: An overview, II,” Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery, vol. 7, no. 6, pp. e1219, 2017.

[22] Joon Son Chung, Jaesung Huh, Seongkyu Mun, Minjae Lee, Hee-Soo Heo, Soyeon Choe, Chiheon Ham, Sunghwan Jung, Bong-Jin Lee, and Icksang Han, “In Defence of Metric Learning for Speaker Recognition,” in Interspeech, 2020, pp. 2977–2981.