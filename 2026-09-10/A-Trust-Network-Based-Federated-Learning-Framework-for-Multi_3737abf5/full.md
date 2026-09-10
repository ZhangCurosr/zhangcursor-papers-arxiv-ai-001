# A Trust-Network-Based Federated Learning Framework for Multi-Center Aging Clock Prediction

Chunxu Zhang1, Bo Li1, Wenliang Wang2, Yang Liu1,4, Di Jiang1,4, Yuan Huang2, Yo-ichi Nabeshima⁵ Akinori Yamamura⁵, Bo Yang6, Qiang Yang1,3\*

1\*PolyU Academy for Artificial Intelligence, The Hong Kong   
Polytechnic University, Hung Hom, Hong Kong, China.   
2Quantum Life, Hong Kong, China.   
3Department of Data Science and Artificial Intelligence, The Hong   
Kong Polytechnic University, Hung Hom, Hong Kong, China.   
4Department of Computing, The Hong Kong Polytechnic University,   
Hung Hom, Hong Kong, China.   
5Department of Aging Science and Medicine, Graduate School of   
Medicine, Kyoto University, Kyoto, Japan.   
6College of Computer Science and Technology, Jilin University,   
Changchun, Jilin, China.

\*Corresponding author(s). E-mail(s): profqiang.yang@polyu.edu.hk; Contributing authors: chunxu.zhang@polyu.edu.hk; cnroselearn@gmail.com; barry@quantumlife.tech; yang-veronica.liu@polyu.edu.hk; di-prof.jiang@polyu.edu.hk; christine@quantumlife.tech; nabeshima.yoichi.7n@kyoto-u.ac.jp; yamamura.akinori.4y@kyoto-u.ac.jp; ybo@jlu.edu.cn;

## Abstract

Aging clocks quantify biological aging and provide a means to characterize individual health status. What kinds of protein interactions are important for determining a high-quality aging clock? Are these interactions zeroth-order or higher-order? Addressing these questions requires learning from large and diverse molecular datasets distributed across multiple medical centers. However, privacy and security constraints often prevent individual-level data from being centrally

shared, resulting in isolated data silos. Federated learning offers a natural way to enable collaborative learning without centralizing raw data, but its application to this setting faces four key challenges: First, limited local sample sizes constrain reliable predictive modeling and make local generative modeling particularly difficult at data-limited centers. Second, inter-center collaboration is often governed by sparse and directional trust relations rather than a globally trusted coordinator. Third, aging clocks are primarily formulated as discriminative predictors and should retain age prediction while supporting interpretation of the learned age-related patterns. Finally, heterogeneous cross-center data can drive model drift and weaken previously acquired knowledge during continued collaborative learning. To address these challenges, we propose TNFL, a trustnetwork-based federated learning framework for multi-center modeling, which we used to further answer the above biological questions. Specifically, TNFL organizes federated learning as progressive model propagation over directed pairwise trust relations, where a trust relation indicates whether a center can trust another center with information privacy and security. We build a model progressively along a path of trust relations without centralized aggregation. TNFL combines an age-aware mixture-of-experts model with generative replay, using generated pseudo-samples to preserve previously learned information and reduce forgetting and model drift. Experiments across multiple molecular datasets show that TNFL enables effective aging-clock prediction with limited local data. TNFL provides interpretable age-dependent prediction patterns and maintains stable performance across interaction orders without systematic forgetting. To answer the above biological questions concerning protein relevance to aging and the order of protein interactions, we analyze the model-identified pairwise protein interactions using TNFL and then investigate how these relationships organize at higher orders through functional and network analyses. The identified interactions repeatedly form coordinated higher-order subnetworks spanning multiple aging-related biological systems, with several proteins repeatedly appearing across different subnetworks. These findings indicate that the model captures molecular relationships that extend beyond isolated pairwise associations and exhibit coherent higher-order biological organization associated with aging.

Keywords: Biological Aging Clock, Trust Network, Sequential Federated Learning

Aging clocks estimate biological age from molecular measurements and provide quantitative indicators of aging-related biological changes [1, 2]. These models capture inter-individual variation in aging and are associated with functional decline, agerelated diseases, and mortality risk [3, 4]. High-quality aging clocks can help identify which protein interactions are important for age prediction and characterize both individual protein effects and higher-order interactions. Investigating these questions increasingly relies on large and diverse datasets that capture variation across populations [5]. However, such datasets are often distributed across multiple medical centers [6] and cannot be centrally shared [7, 8]. This setting therefore calls for a federated learning framework that can learn jointly from data distributed across different centers.

However, applying federated learning on multi-center aging-clock modeling faces four key challenges. First, limited local data pose a fundamental constraint on both predictive and generative modeling. Small centers may lack sufficient samples to train reliable aging clocks independently, while the same data scarcity also makes local generative modeling difficult. Second, cross-center collaboration is restricted by partial and asymmetric trust. In practice, centers may permit model communication only with selected partners, and these relations can be directional rather than globally shared among participating centers. Third, aging-clock modeling calls for discriminative and interpretable prediction. Aging clocks should maintain reliable predictive performance while also enabling interpretation of the age-related patterns learned by the model. Finally, cross-center heterogeneity introduces model drift and knowledge forgetting. Continued adaptation to different local data distributions may shift the model toward recently encountered patterns and weaken information acquired from earlier centers. Together, these challenges characterize the capabilities required for a federated learning framework to support reliable multi-center aging-clock modeling.

Standard federated learning relies on a central server trusted by all participating centers for model coordination and aggregation, an assumption that does not hold under sparse and directional inter-center trust. Beyond this conventional setting, trust-related federated [9–11], decentralized federated learning [12, 13], and sequential federated learning [14-16] have explored alternative collaboration mechanisms. These approaches each address part of the challenges arising in multi-center aging-clock modeling, but do not provide a single framework that addresses them together. In parallel, machine-learning and deep-learning studies on aging clocks have increasingly moved beyond age prediction toward biological interpretation [17, 18], including the identification of age-relevant molecular features [19] and their associated biological pathways [20-22]. However, how model-identified molecular interactions organize across different orders, particularly into higher-order relationships, remains less explored. Taken together, existing work does not yet provide a unified approach that simultaneously supports effective multi-center collaboration and examines which protein interactions are relevant to aging-clock prediction and how they organize across different orders. We therefore propose TNFL to address both aspects within a unified multi-center framework.

Specifically, TNFL organizes federated learning as progressive model propagation over directed pairwise trust relations without relying on a globally trusted aggregation server. As illustrated in Fig. 1(a), directed pairwise trust relations define which model transmissions are permitted between centers and collectively form the trust network. Each edge governs an authorized model transmission between two centers, and TNFL constructs valid trust sequences over this network. Fig. 1(b) illustrates how the evolving model is progressively updated along an admissible trust sequence, with each visited center continuing from the model state accumulated through preceding updates. This allows centers with limited local data to benefit from information incorporated earlier in the sequence. For aging-clock modeling, TNFL employs an age-aware mixture-of-experts predictor. Its routing behavior supports interpretation of age-dependent prediction patterns. TNFL further incorporates generative replay to preserve information across heterogeneous center updates and reduce model drift and knowledge forgetting. Experiments across multiple molecular datasets show that TNFL remains effective when local data are limited. TNFL provides interpretable agedependent prediction patterns and also maintains stable performance across different interaction orders without systematic forgetting.

![](images/676031c65ec571b0ff55c3aa0a8fe5fffdad597666272952ca3dbebc4e2fec50.jpg)  
Fig. 1: Trust-constrained model propagation in TNFL. (a) Directed pairwise trust relations define which model transmissions are permitted between centers, forming the trust network over which collaboration can proceed. (b) TNFL realizes progressive cross-center learning along an admissible trust sequence. The aging-clock model is updated at each visited center, with its model capability progressively enhanced as information from more centers is incorporated.

To further address the biological questions concerning which protein relationships are relevant to aging-clock prediction and whether these relationships extend to higher-order interactions, we first characterize the biological coherence of the protein interactions identified by TNFL. Functional analyses show that these interactions are associated with multiple aging-related biological processes, while several pairs are also supported by known functional associations. We then examine whether these pairwise relationships further organize at higher orders through complementary analyses based on age prediction and expert-routing outputs across different population settings. Across these analyses, the identified protein groups repeatedly form coordinated higher-order subnetworks spanning multiple aging-related biological systems, with several proteins repeatedly appearing across different subnetworks. Together, these results show that the protein relationships identified by the model are not limited to isolated pairwise associations but exhibit coherent higher-order organization associated with aging. Importantly, these patterns emerge without incorporating prior protein-interaction or pathway knowledge during model training, further supporting their biological coherence.

## 1 Results

In Section 1.1, we use TNFL to investigate the biological questions of which proteins are relevant to aging-clock prediction and how their effects and interactions are organized across different orders. In Section 1.2, we evaluate TNFL computationally against the key challenges of multi-center aging-clock modeling, focusing on predictive performance and robustness.

## 1.1 Biological Findings from TNFL

Biological analyses are conducted using the main TNFL implementation on the UKB-Center proteomic dataset. We first use the age-dependent routing behavior of AgeMoE to identify proteins associated with aging-related prediction patterns. We then examine these molecular effects across increasing interaction orders, from individual proteins to pairwise interactions and higher-order protein organizations, to characterize both their biological relevance and their broader coordination in aging.

## 1.1.1 Individual Protein Relevance to Aging-Clock Prediction

We first examine the age-dependent routing behavior of the AgeMoE predictor used in the main TNFL implementation and use it to identify individual proteins associated with aging-related prediction patterns. In AgeMoE, each expert is a learnable prediction sub-module, and a routing network assigns sample-specific weights to combine the outputs of different experts. We therefore analyze these routing weights to examine how the model organizes its predictions across age groups. Samples in the global test set are grouped into age intervals, and the average routing weight assigned to each expert is computed within each group.

As shown in Fig. 2(a), experts display distinct age-dependent routing patterns: some maintain relatively stable contributions across ages, whereas others receive systematically higher or lower weights in older or younger groups. For example, in the UKB-Center dataset, the routing weights of Experts 1 and 2 decrease with age, whereas those of Experts 3 and 4 increase. These results indicate that AgeMoE adjusts the relative contributions of its prediction sub-modules according to age, providing an interpretable view of how its predictive behavior changes across age groups.

To further connect these age-dependent routing patterns to molecular features, we analyze how individual protein values influence the routing weight assigned to Expert 3 in the UKB-Center proteomic dataset. Expert 3 is selected because its routing weight increases with sample age (Fig. 2(a)), making it informative for examining molecular features associated with this age-dependent model behavior. To assess protein-specific effects, each protein is systematically varied across ten quantile levels while keeping all other features fixed, and the resulting changes in the routing weight of Expert 3 are recorded. We quantify each protein's influence as the signed change in Expert 3 routing weight induced by increasing that protein from its lowest to highest quantile level while keeping all other features fixed, and rank proteins according to this score.

As shown in Fig. 2(b), we highlight a subset of the top 20 proteins with positive effects on Expert 3 routing, for which higher protein values lead to higher routing weights. Because Expert 3 is increasingly utilized at older ages, these proteins are associated with a model-routing pattern that becomes more prominent with age. Several highlighted proteins, including GDF15, GFAP, and NEFL, have previously been reported to increase with age [23–25], supporting the consistency of the learned routing patterns with known age-associated molecular changes.

![](images/50156a32a33900a17dc7b88c0e680b2b2660759f6b904723392a156a56cb7365.jpg)  
(a) UKB-Center

![](images/c809e4559b8c7924262a17c346fc9958b9eae7918276a7042236639d8b3ef290.jpg)  
(b) Top-ranked proteins with monotonically increasing expert routing weights

![](images/8c026db348b49c8adb3f770a77bbc8b9ecab5772f98a8d5ac8210748f422c9cf.jpg)  
(c) Functional and pathway enrichment of top-100 proteins  
Fig. 2: Visualization of expert routing patterns across samples (a), top-ranked proteins showing a positive relationship with Expert 3 routing weight, where higher protein values correspond to higher weight (b), and functional and pathway enrichment of the top-100 proteins ranked by their influence on expert routing (c).

We next examine the broader biological relevance of these model-identified proteins. Using the UKB-Center proteomic dataset, we selected the top 100 proteins ranked by their influence on expert routing for downstream biological characterization. Functional enrichment analysis using Metascape [26] revealed 20 significantly enriched ontology clusters, which can be summarized into three major biological themes: Extracellular Matrix (ECM) Structural Integrity, Deregulated Nutrient Sensing, and Neuro-degeneration and Inflammaging (Fig. 2(c)).

We further assessed biological relevance by comparing these proteins against curated aging-related databases. Several proteins overlap with GenAge [27] and the SASP Atlas [28], including SOD2, ELN, EFEMP1, COL6A3, GDF15, and PTX3. In addition, we evaluated protein localization and tissue origin using secretome annotations and data from the Human Protein Atlas [29]. The results indicate that most proteins are secreted or ECM-associated, with additional expression in brain and other major tissues such as vascular, liver, lung, and skin. Single-cell annotations further suggest expression across neural, stromal, and immune cell types. Collectively, these analyses show that the model-identified proteins are functionally enriched in aging-related processes and span multiple tissues and cell types relevant to systemic aging.

## 1.1.2 Pairwise Protein Interactions

We next extend the analysis from individual protein effects to pairwise protein interactions identified by the model. The top-10 model-identified protein pairs were further examined using STRING functional association analysis [30]. Three pairs, (GIP, CGA), (INSL3, CGA), and (GIP, INSL3), showed known STRING-supported associations related to endocrine and metabolic signaling. The remaining pairs showed no direct STRING links but exhibited coherent functional convergence. GFAP-related pairs were associated with neuroinflammatory and neuroendocrine processes [31], while (GFAP, NPPB) reflected neural and cardiac stress signaling [32]. Other pairs, including (EDA2R, CGA) and CCDC80-related pairs, involved inflammatory [33], endocrine, and extracellular matrix remodeling functions. These results indicate that the identified pairwise interactions capture biologically coherent relationships that are not limited to direct interactions already represented in curated databases, motivating further analysis of whether these pairwise relationships organize into higher-order protein subnetworks.

## 1.1.3 Higher-Order Organization of Protein Interactions

We therefore investigated whether model-identified pairwise interactions repeatedly organize into higher-order protein subnetworks. Higher-order groups were examined through STRING analysis from complementary analytical perspectives, including protein coordination associated with final biological age predictions and with latent AgeMoE routing behavior, as well as analyses over the full population and agerestricted young subsets. These complementary settings allow us to examine whether coordinated protein patterns recur across different views of the learned aging signals rather than being specific to a single analysis setting.

![](images/9ba2fa289d21fc909557c996389cb7cba022368385752dc887c1456e46370b9a.jpg)  
(a) Protein coordination under fullpopulation expert routing $( P = 5 . 4 7 \times 1 0 ^ { - 6 } )$

![](images/cf48f35702d262f3ade8fe951f5a214f79faa605bfe8f0467d694a2072ee323e.jpg)  
(b) Protein coordination under fullpopulation age prediction $( P = 6 . 8 3 \times 1 0 ^ { - 6 } )$

![](images/7c408d03845f2c026e1d57b1ae5860557946f10f8c7858f1ab75319da1ea179f.jpg)  
(c) Protein coordination under young-subset expert routing $( P = 5 . 8 5 \times 1 0 ^ { - 6 } )$

![](images/e645c99ff144af277b72124b6a0ba461e4e08ab828f687deae2dd75e6e8d4a61.jpg)  
(d) Protein coordination under young-subset age prediction $( P = 2 . 1 7 \times 1 0 ^ { - 7 } )$  
Fig. 3: Biological interpretation of STRING-derived synergistic protein subnetworks identified under complementary analytical perspectives (a)-(d) on the UKB-Center dataset. In subnetworks (a)-(d), nodes represent proteins and edges denote STRING functional associations, with thicker edges indicating stronger association confidence. Recurrent proteins appearing across multiple subnetworks are highlighted in red and marked with +. The reported P values correspond to STRING functional association enrichment, assessing whether the observed connectivity within a protein group exceeds that expected for random protein sets with comparable background connectivity.

Across these complementary perspectives, the identified subnetworks repeatedly converged on coherent aging-related biological organizations. One recurrent organization centered on coordinated neurovascular-endocrine modules involving neural injury markers (GFAP, NEFL), extracellular matrix remodeling proteins (ELN, LTBP2), and endocrine-metabolic regulators (GIP, FSHB) (Fig. 3(a)) [34]. A second organization further incorporated hypothalamic and neuroendocrine regulators, including AGRP and OXT, linking hormonal coordination with metabolic and neural processes (Fig. 3(b)) [35]. Another recurrent module emphasized systemic stress and cardiometabolic signaling through recurrent inclusion of GDF15 and NPPB (Fig. 3(c)) [36, 37]. The most integrated organization ultimately converged on broader multi-system coordination simultaneously involving inflammatory chemokines, neural injury markers, endocrine regulators, stress-response proteins, and vascular remodeling factors (Fig. 3(d)) [38, 39].

Notably, several proteins, including GFAP, NEFL, GDF15, GIP, ELN, LTBP2, and CXCL17, appeared repeatedly across these subnetworks, suggesting shared hub-like signals that connect multiple aging-related biological processes. Together, the recurrent appearance of these proteins and subnetworks indicates higher-order coordination across neural, vascular, endocrine, metabolic, inflammatory, and stress-related systems. This organization is in line with the hallmarks-of-aging framework [40], which links molecular and cellular hallmarks to broader integrative dysfunctions during aging.

Taken together, these analyses show that the model-identified protein interactions are not limited to isolated pairwise associations, but repeatedly organize into higherorder subnetworks spanning multiple aging-related biological systems. Importantly, these patterns emerge without incorporating prior protein-interaction or pathway knowledge during model training. These findings demonstrate that aging-related molecular effects captured by TNFL extend from individual proteins and pairwise interactions to coherent higher-order organization. Additional details of the biological analyses are provided in Appendix A.

## 1.2 Computational Evaluation of TNFL

We next evaluate TNFL from the computational perspective, focusing on predictive performance and robustness under multi-center learning. Predictive performance is assessed across limited-data centers and different aging-clock backbones, while robustness is examined across varying interaction orders, heterogeneous center conditions, and trust-network structures. Together, these evaluations characterize how effectively TNFL addresses the key computational challenges of multi-center aging-clock modeling.

## 1.2.1 Predictive Performance

We first evaluate whether TNFL improves aging-clock prediction when individual centers have limited local data, compared with independent local training and using server-based FedAvg [41] as a centralized federated learning reference. Performance is measured by mean absolute error (MAE) on each center's local test set 1. Local training and FedAvg provide reference settings outside the trust-network protocol, whereas TNFL is evaluated through trust-network-based sequential propagation. For this comparison, we instantiate multiple valid single-pass trust sequences by randomly permuting client indices so that every center is visited once, with each consecutive model transmission governed by a directed trust relation. Results for TNFL are averaged across these sampled sequences to reduce dependence on any particular propagation order. Experiments are conducted on three multi-center datasets covering multiple omics modalities and client configurations: UKB-Center and UKB-Imbalance from the UK Biobank [42], and GEO-Methylation from the Gene Expression Omnibus (GEO) [43]. Results are summarized in Fig. 4(a)−(c).

![](images/c0a715061824a021bec1373a16d055c9abb072c8f008725ea344f99847e74214.jpg)  
(a) UKB-Center

![](images/470d6d2661739b24e24433010ca4d099b930b5398aeedb107acf49396ac3df01.jpg)  
(b) UKB-Imbalance

![](images/d9e57d9760476e203e5a82c8c5613471226f42ecb87b9f02034781356c151be4.jpg)

<table><tr><td>Local training</td><td></td><td>FedAvg</td><td>TNFL</td></tr></table>

Local trainingFedAvg TNFL (a)-(c) compare local training, server-based FedAvg, and TNFL on each center's local test set.

(c) GEO-Methylation  
![](images/771c4b313689799e40ae03c9f269906f1507f60b5f60fdc639bd067c89e5624a.jpg)  
(d) UKB-Center

![](images/444061d3be644d1bf4b5db66da8c6e86986f4bf6f0849e31f1285712f555daf6.jpg)  
(e) UKB-Imbalance

![](images/5aaaabde5f95c99bda0c049e5fb4a51f968decec08e87c13a6035f5f21c57272.jpg)  
(f) GEO-Methylation  
(d)-(f) compare different aging clock backbones trained with the TNFL propagation strategy on each center's local test set. Circles indicate the corresponding local-training results for each backbone.

![](images/08526062d64434003db1ce7faeabf788879d2681a2d11e7c12c3ae22aeb09c35.jpg)  
(g) UKB-Center

![](images/b01ed08697eb8125d9b274562b7e1dbd99190651890cf1650b9adf003cf9e251.jpg)  
(i) GEO-Methylation  
Fig. 4: Predictive performance of TNFL and its compatibility with different agingclock backbones across multi-center datasets. Panels (a)-(c) compare TNFL with local training and server-based FedAvg, whereas panels (d)-(i) evaluate different agingclock backbones under the TNFL propagation strategy. AgeMoE denotes the main implementation with generative replay.

![](images/ed460886d72fd8a827b1c6922014fdcbe73ec80c748321a629092354572ad727.jpg)  
(h) UKB-Imbalance  
(g)-(i) compare global-test MAE across aging clock backbones trained with the TNFL propagation strategy. Box plots summarize variability across random interaction sequences.

Across datasets, the results support three main findings. First, TNFL consistently achieves lower MAE than independent local training, demonstrating the value of integrating information from distributed centers. Second, TNFL achieves predictive performance comparable to server-based FedAvg without relying on central aggregation. On the UKB-Center dataset, the average MAE reduction relative to local training is 0.69 years for TNFL and 0.70 years for server-based FedAvg, showing that trust-network-based pairwise propagation can provide effective collaborative modeling. Third, TNFL further improves over FedAvg under more heterogeneous configurations, with additional average MAE reductions of 0.12 years on UKB-Imbalance and 0.34 years on GEO-Methylation, indicating that TNFL remains effective under greater cross-center heterogeneity. These results demonstrate that TNFL enables effective collaborative aging-clock prediction when individual centers are limited by local data, while maintaining performance competitive with centralized federated learning.

We further examine whether these predictive benefits extend across different aging-clock backbones. Specifically, we consider linear regression [44], XGBoost [45], multilayer perceptrons (MLP) [46], and transformers [47] as alternative aging-clock backbones within the TNFL propagation strategy. The main TNFL implementation uses AgeMoE as the discriminative predictor together with generative replay for crosscenter knowledge preservation. Performance is assessed on local test sets to capture center-specific predictive ability and on a global test set to evaluate generalization across centers. To account for variability across interaction orders, results are averaged over multiple randomly sampled center sequences.

As shown in Fig. 4(d)-(f), all evaluated aging-clock backbones improve over their local-training counterparts when trained with the trust-network-constrained TNFL propagation strategy, indicating that TNFL is not tied to a specific aging-clock architecture. In addition, the main TNFL implementation consistently achieves the strongest overall performance and exhibits lower variance on global test sets across different interaction sequences (Fig. 4(g)-(i)). It achieves median MAE reductions of 0.54 years on UKB-Center, 0.55 years on UKB-Imbalance, and 2.55 years on GEO-Methylation. Notably, XGBoost also benefits substantially from the TNFL propagation strategy, achieving a 2.60-year MAE reduction on GEO-Methylation relative to local training. These results show that the trust-network-based learning mechanism can accommodate diverse discriminative aging-clock backbones, while AgeMoE provides the main predictive implementation used for subsequent analysis.

To further examine the contribution of generative replay to cross-center predictive generalization, we perform an ablation study by removing the generative component. Removing the generative component reduces cross-client generalization, particularly under settings with more pronounced distributional heterogeneity, supporting its role in preserving previously acquired information during cross-center learning. Detailed results are provided in Appendix B. Together, these results show that TNFL maintains effective predictive performance under limited local data, across different aging-clock backbones, and with heterogeneous cross-center distributions.

## 1.2.2 Robustness Evaluation

We next evaluate the robustness of TNFL under variations in interaction order, center-specific measurement conditions, and trust-network structures. We first examine whether sequential cross-center learning remains stable across different interaction orders and preserves previously acquired knowledge. Following the sequence-level instantiation used in the baseline evaluation, each sampled interaction order corresponds to a valid single-pass trust sequence in which all centers are visited once and direct model transmission occurs only between consecutive centers. We evaluate different aging-clock backbones trained with the TNFL propagation strategy across multiple randomly sampled interaction orders for each dataset. Three complementary metrics are used to characterize order sensitivity, sequential performance degradation, and knowledge forgetting, with detailed definitions and calculation procedures provided in Appendix C.

Client-level MAE range. We measure robustness for each client as the range of MAE values, computed as the maximum minus minimum MAE across interaction orders on the client's local test set. Smaller ranges indicate more stable center-specific performance under different propagation sequences. As shown in Fig. 5(a)-(c), the AgeMoE-based main implementation equipped with generative replay exhibits stable performance, with ranges comparable to or smaller than those of alternative agingclock backbones. The ranges remain below 0.5 years across datasets and drop below 0.3 years on UKB-Imbalance, indicating limited sensitivity of center-specific prediction to interaction order.

Cumulative MAE degradation. We further assess whether global predictive performance deteriorates during sequential propagation. For each interaction order, this metric is calculated by summing only the positive increases in global-test MAE between consecutive intermediate model states, i.e., max(current MAE - previous MAE, 0), accumulated over the propagation process. This metric captures the extent of performance degradation introduced by successive center updates; lower values indicate more stable sequential updates. As shown in Fig. 5(d)−(f), the main TNFL implementation exhibits minimal cumulative degradation, with values below 0.2 years on the two UK Biobank datasets and near zero on GEO-Methylation. These results indicate that successive center updates do not lead to substantial cumulative deterioration in global predictive performance.

![](images/08676be1e467201c3df11a4adcdb8b40811d797a382820da966112fc10babab4.jpg)  
(a) UKB-Center

![](images/8764b9d85ce787cc740b8ff44f6f1b1057397bab186bf634fd78e9d21b738b62.jpg)  
(b) UKB-Imbalance

![](images/f92a20d12c3d24a0201d82280856251a5f76aa86c1c98f83046d1560b87ee232.jpg)  
(c) GEO-Methylation

![](images/9603a004d8ead6981dd12c7c26eaa5b15955752104cad2c04a864e813dc501d2.jpg)  
(d) UKB-Center

![](images/30eb1ad6ceb0213faafa73a9e07a3103e20be436ee490de62a7dba9bfb54f6d4.jpg)  
(e) UKB-Imbalance

![](images/55d5e6a80000fdd526b36d6ca7bc825084ef0bd868c7dd35387978d8bbc4fdc1.jpg)  
(f) GEO-Methylation

![](images/10bef03031d691c8c8a3489d5aeb77022d6183c3e0000c2241d7f5161f1da903.jpg)  
(g) UKB-Center

![](images/7d288410825f4f0c2821374a9cb63422181871029ea255982c28fe0361071411.jpg)  
(h) UKB-Center

Fig. 5: Interaction-order robustness of aging clock models under TNFL propagation. (a)-(c) show the client-level MAE range, computed as the maximum minus minimum MAE across sampled interaction orders on local test sets; lower values indicate greater robustness to interaction-order changes. (d)-(f) show cumulative MAE degradation on the global test set, computed by accumulating only MAE increases between successive center updates; lower values indicate more stable performance progression. (g) shows forgetting rates on UKB-Center, computed as the relative MAE change from an intermediate model to the final model for clients in the first half of each sequence; positive values indicate performance degradation after subsequent updates, whereas zero or negative values indicate knowledgepreservation or improvement. (h) further visualizes step-wise forgetting dynamics on UKB-Center across all sampled interaction sequences, where each point summarizes the median forgetting rate for clients visited at a given step and evaluated after later updates; values below zero indicate that later updates do not systematically degrade earlier client performance.

Forgetting rate. We define forgetting for a client visited at a given step as the relative change in MAE from the intermediate model obtained after that client update to the final model produced at the end of the same interaction order. Specifically, it is calculated as (final MAE-intermediate MAE)/intermediate MAE×100%, where both MAE values are evaluated on the same client's local test set. Positive values therefore indicate that performance on an earlier-visited client becomes worse after subsequent center updates, suggesting forgetting of previously learned center-specific information. To focus on early-stage effects, this metric is computed on the first half of clients in each sequence, where such information is most likely to be overwritten by later updates. We use UKB-Center for this analysis because its larger number of clients and diverse sample characteristics provide a suitable setting to observe potential forgetting effects. Results show that the main TNFL implementation consistently maintains or improves performance on earlier-visited clients after subsequent center updates (Fig. 5(g)).

We further analyze step-wise forgetting dynamics on UKB-Center, where each step denotes the position of a visited center within an interaction sequence and the MAE immediately after that center update serves as the reference for later evaluations. For each visited step and later evaluation step, Fig. 5(h) reports the median forgetting rate across all sampled interaction sequences, rather than a single sequence-specific trajectory. The median trajectories remain mostly below zero, indicating that later center updates do not systematically degrade performance on previously visited centers and often further improve it.

These results show that TNFL remains stable across different interaction orders and largely preserves information acquired from previously visited centers. We next examine whether this robustness extends to variation in center-specific measurements. Robustness to simulated measurement variability. To evaluate how TNFL handles practical variations in data collection, we simulate measurement differences across clients, including factors such as instrument calibration, experimental procedures, and batch effects. Each feature is perturbed in two ways: a client-specific shift, representing systematic deviations shared by all samples within a center, and sample-level random noise, capturing local measurement fluctuations. Despite these perturbations, the main TNFL implementation shows only limited performance degradation, indicating that predictive performance remains stable under simulated center-specific measurement variability. Full results of this perturbation analysis are provided in Appendix D.

Robustness across trust-network structures. Finally, we evaluate whether TNFL remains effective when collaboration is constrained by more complex directed trustnetwork structures rather than single-pass interaction sequences. We simulate model propagation over network-based trust structures, where centers may participate multiple times and propagation sequences are not restricted to a single client permutation. Under these simulated conditions, TNFL maintains predictive performance comparable to that observed with randomly sampled interaction orders, and the relative performance ranking among aging-clock backbones remains largely consistent. These results indicate that the trust-network-based propagation mechanism remains effective across different feasible collaboration structures and more complex multi-center interaction patterns. Detailed experimental results are provided in Appendix E.

Together, these analyses show that TNFL maintains robust behavior across variations in interaction order, center-specific measurement conditions, and feasible trust-network structures.

## 2 Discussion

Summary of Main Findings. The biological analyses show that TNFL captures aging-related molecular patterns across multiple interaction orders. Age-dependent expert routing identifies individual proteins associated with aging-clock prediction, and these proteins are enriched in aging-related biological processes and supported by external biological annotations. Extending beyond individual effects, the model further identifies biologically coherent pairwise protein interactions that repeatedly organize into higher-order subnetworks spanning neural, vascular, endocrine, metabolic, inflammatory, and stress-related systems. From the computational perspective, TNFL achieves effective aging-clock prediction when local data are limited, remains compatible with different predictive backbones, and benefits from generative replay for cross-center generalization. It also exhibits stable behavior across different interaction orders, preserves information acquired from previously visited centers, and remains robust under simulated measurement variability and different trust-network structures. Together, these findings show that TNFL supports reliable multi-center aging-clock learning while enabling biological analysis of protein effects and interactions from individual to higher orders.

Comparison with Existing Work. Standard federated learning provides the canonical paradigm for collaborative model training across distributed centers, typically relying on a central server to coordinate local updates and aggregate them into a shared global model [41]. Beyond this conventional setting, trust-related FL incorporates trust into collaborative learning to characterize client reliability or regulate communication among selected parties [9-11], with OPS [11] further supporting collaboration over a directed trust graph. Decentralized FL removes the fixed central aggregator and instead coordinates learning through model exchange, aggregation, or synchronization among connected clients [12, 13]. Sequential FL transfers models across centers so that information can accumulate through successive local updates [14–16]; CWT [14] and FedELMY [16] adopt cumulative cross-center training, while FedDC [15] transfers multiple models with periodic aggregation, and FedELMY further regularizes local updates relative to the preceding model to reduce drift. A detailed comparison of these paradigms with TNFL is summarized in Table 1. These approaches address different aspects of cross-center collaboration, but none is designed to jointly meet the requirements of multi-center aging-clock modeling considered here. In contrast, TNFL uses directed pairwise trust relations to determine feasible model propagation without relying on a globally trusted aggregation server. Along this propagation process, cumulative cross-center learning allows data-limited centers to benefit from previously accumulated information, while generative replay helps preserve acquired knowledge across heterogeneous center updates.

Machine-learning and deep-learning studies on aging clocks have also increasingly moved beyond age prediction toward biological interpretation [17, 18]. Some studies identify molecular features that contribute most strongly to age prediction and examine their biological relevance, for example by analyzing selected CpG sites or their associations with aging-related phenotypes [19]. Other approaches further relate model-identified molecular features to genes, biological processes, and pathways [20, 21], while more recent work has examined how these aging-related molecular patterns vary across age stages or population subgroups [22]. These studies have substantially improved the interpretability of aging-clock models, but they mainly focus on which molecular features are important and what biological functions or patterns they represent. Our analysis further examines aging-related molecular organization across different orders, covering individual protein effects as well as pairwise and higher-order protein relationships. This allows us to characterize not only which proteins are relevant to aging-clock prediction, but also how their relationships extend beyond isolated pairwise associations into higher-order organization.

Table 1: Comparison of representative federated learning methods for four key challenges in multi-center aging-clock modeling. √indicates that the corresponding challenge is explicitly addressed by a dedicated mechanism, whereas X indicates that no such mechanism is provided
<table><tr><td>Category</td><td>Representative Method</td><td>Limited Local Data</td><td>Partial Asymmetric Trust</td><td>Prediction and Interpretation</td><td>Model Drift and Knowledge Forgetting</td></tr><tr><td rowspan="3"></td><td>TrustFL [9]</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Trust-related FL TrustNetFL [10]</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>OPS [11]</td><td>x</td><td>√</td><td>x</td><td>x</td></tr><tr><td rowspan="2">Decentralized FL</td><td>Swarm Learning [12]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Decentralized FedAvg [13]</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td rowspan="3">Sequential FL</td><td>CWT [14]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>FedDC [15]</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>FedELMY [16]</td><td>√</td><td>x</td><td>x</td><td>√</td></tr><tr><td>Ours</td><td>TNFL</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

Taken together, TNFL addresses the key challenges of multi-center aging-clock modeling and enables a unified investigation of which proteins are relevant to agingclock prediction and how their effects and interactions are organized across different orders.

Limitations and Future Directions in Aging Clock Modeling. The proposed trust-network-based framework achieves effective performance across multiple molecular aging-clock datasets, but several limitations remain. From an algorithmic perspective, the current framework assumes that the trust network is specified before collaborative learning and that model propagation proceeds along feasible trust sequences derived from this network. Future work could investigate adaptive routing under dynamic trust relations, where subsequent propagation decisions are adjusted according to the evolving learning state or collaboration context. It would also be valuable to develop model- or data-dependent termination criteria that determine when additional cross-center propagation provides limited incremental benefit, allowing the sequential learning process to terminate adaptively rather than relying on a predefined propagation process. Second, using chronological age as the supervision signal may favor features with strong empirical associations with age, which may limit the diversity of captured aging-related signals. Future work should therefore consider alternative prediction targets beyond chronological age, such as healthspan or phenotypic age, to better reflect functional aspects of biological aging. Third, the current analysis considers each omics modality separately, which captures only part of the molecular complexity underlying aging. Integrating multi-omics data in future work may provide a more comprehensive representation of molecular changes across biological layers and reduce dependence on any single data source. Finally, future work could extend TNFL to real-world multi-center deployments to further evaluate its behavior under naturally occurring data heterogeneity, institutional governance policies, and communication constraints.

## 3 Methods

This section describes TNFL for multi-center aging-clock modeling. We first formulate the multi-center problem setting and the computational requirements that motivate the framework design. Based on these requirements, TNFL is developed around two complementary aspects: how collaborative learning is organized under constrained inter-center trust, and how an aging-clock model is progressively learned while retaining predictive utility, interpretability, and previously acquired information across centers. We then formalize the trust-network-based learning framework and its model components, followed by the training and optimization procedures. Finally, we characterize the proposed trust relation relative to participant-level behavioral models and establish the security properties arising along valid trust sequences. The section concludes with the experimental design and evaluation settings used throughout the study.

## 3.1 Problem Formulation and Multi-Center Challenges

Problem Definition. We formulate the aging clock task as a supervised regression problem, where the goal is to predict an individual's chronological age from highdimensional molecular measurements, such as proteomic or DNA methylation features. Formally, let $\textbf { x } \in \ \mathbb { R } ^ { d }$ denote the molecular feature vector of a given sample, and let $y \in \mathbb { R }$ represent the corresponding chronological age. The objective is to learn a mapping $\mathcal { F } _ { \Theta } : \mathbb { R } ^ { d }  \mathbb { R }$ from molecular features to chronological age by minimizing a predefined loss function, typically the mean squared error, over the training data.

Multi-Center Setting. Let there be K centers, each holding a local dataset $\mathcal { D } _ { k } =$ $\left( \mathbf { x } _ { i } , y _ { i } \right) _ { i = 1 } ^ { n _ { k } } ,$ where $\mathbf { x } _ { i } \in \mathbb { R } ^ { d }$ denotes the molecular features of sample i and $y _ { i } \in$ R its chronological age, and $n _ { k }$ is the number of samples at center k. The datasets remain locally held without direct sharing of individual-level data, and may differ across centers in sample size and molecular feature distributions. Cross-center collaboration is conducted through model communication subject to the relations among participating centers.

Challenges in Multi-Center Aging-Clock Modeling. The above setting gives rise to four coupled challenges. Limited Local Data can restrict reliable aging-clock training at individual centers and may also be insufficient for local generative modeling, creating a need to leverage information across centers. Such collaboration is further constrained by Partial and Asymmetric Trust, where model communication may be permitted only between selected and directional pairs of centers. Meanwhile, Discriminative and Interpretable Prediction requires the aging clock to retain age prediction while supporting analysis of the learned age-related patterns. Finally, crosscenter heterogeneity gives rise to Model Drift and Knowledge Forgetting, as continued adaptation to center-specific distributions may shift the model away from previously learned patterns and weaken knowledge acquired from other centers.

These challenges motivate TNFL as a unified framework for progressive agingclock learning over constrained inter-center relations. Rather than training each center independently or relying on a globally trusted coordinator, TNFL allows an evolving model to accumulate information through authorized cross-center propagation. Within this process, the aging clock remains a discriminative predictor whose internal routing behavior can be interpreted, while generative replay helps retain information acquired from previously visited centers as the model adapts to heterogeneous local distributions. Together, these considerations shape the overall design of TNFL.

## 3.2 Directed Trust Relations and Trust-Network Formulation

To formalize the constrained inter-center relations described above, we introduce a directed, pairwise, and message-specific trust relation for model communication in TNFL. Each relation specifies the obligations associated with an authorized transmission between two centers, and the collection of such relations forms the directed trust network used by TNFL.

Protocol Message. Let A and B denote two participating centers. We use $X _ { A  B }$ to denote the protocol-authorized message that carries the model parameters transmitted from A to B in a pairwise interaction. The concrete parameter components included in $X _ { A  B }$ depend on the instantiated TNFL implementation and training procedure. The message does not include raw local data, individual-level molecular measurements, local gradients, explicit parameter-update vectors, or local training computations, all of which remain local to the corresponding center. Here, $X _ { A  B }$ refers specifically to a pairwise message transmitted during the training process; release of the final trained model is specified separately in the TNFL protocol.

Definition 1 (Directed Pairwise Trust Relation). For a protocol-authorized transmission of message $X _ { A  B }$ from center A to center B, Trust $( A , B , X _ { A  B } )$ denotes the directed, pairwise, and message-specific trust relation governing this transmission. This relation imposes the following obligations:

1. Sender-side authenticity. Center A agrees to transmit $X _ { A  B }$ to center B and guarantees that it does not poison, forge, or maliciously tamper with $X _ { A  B }$

2. Receiver-side confidentiality. Center B shall not use $X _ { A  B }$ received from center A to infer private information contained in or represented by $X _ { A  B }$

Remark 1. If $X _ { A  B }$ is derived from a model state influenced by previously participating centers, the receiver-side confidentiality obligation also applies to private information from those centers insofar as it is represented in $X _ { A  B }$ Remark 2. Center A may optionally encrypt $X _ { A  B }$ such that downstream centers of B cannot directly access $X _ { A  B }$ or infer the private information contained in it.

The directed trust relation is non-transitive; specifically, Trust $( A , B , X _ { A  B } )$ and Trust $( B , C , X _ { B  C } )$ do not imply a direct trust relation between A and $C .$ Trust Network. Let $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { K } \}$ denote the set of participating centers. The directed trust network is represented as

$$
\mathcal { G } = ( \nu , \mathcal { E } ) ,\tag{1}
$$

where $( v _ { i } , v _ { j } ) \ \in \mathcal { E }$ indicates that the protocol-authorized message $X _ { v _ { i }  v _ { j } }$ can be transmitted from $v _ { i }$ to $v _ { j }$ under Trust $( v _ { i } , v _ { j } , X _ { v _ { i }  v _ { j } } )$ . The trust network is treated as a given collaboration structure determined by pre-established relationships among the participating centers. TNFL organizes collaborative training through feasible modelpropagation sequences over this network, as described next.

## 3.3 Trust-Network-Based Federated Learning Framework

## 3.3.1 Framework Overview

Building on the directed trust network defined above, TNFL organizes multi-center learning as a progressive model-propagation process over feasible trust-constrained sequences. Rather than training separate models at individual centers or relying on centralized aggregation, an evolving model state is successively transmitted and updated across centers under pre-established directed trust relations. Each participating center therefore continues from the knowledge accumulated through preceding updates and further adapts the model using its local data.

Within this propagation process, the main implementation of TNFL employs AgeMoE as the discriminative aging-clock predictor, enabling age estimation while exposing age-dependent routing behavior for interpretation. A generative replay mechanism is further maintained alongside the aging-clock model to retain information acquired from previously visited centers during continued adaptation to heterogeneous local distributions. The overall architecture of TNFL is illustrated in Fig. 6.

## 3.3.2 Trust-Constrained Training Sequence

Given the directed trust network $\mathcal { G } = ( \nu , \mathcal { E } )$ , TNFL performs collaborative training along a finite sequence of centers

$$
S = [ s _ { 1 } , s _ { 2 } , \ldots , s _ { T } ] , \qquad s _ { t } \in \mathcal { V } ,\tag{2}
$$

![](images/4a04cfc0209e8669e30158c57cd98877906a7594a1aa6ca91fc859a8f5f4c203.jpg)  
Fig. 6: Sequential training procedure of TNFL. The AgeMoE aging clock and generative model are jointly propagated across centers along a valid trust sequence composed of directed trust relations. At each client, the generative model first synthesizes pseudo-samples approximating previously encountered client distributions, followed by prediction-consistency filtering using the received aging clock. The filtered pseudosamples are then combined with local data to update the AgeMoE aging clock, while the generative model is updated using local client data.

where each consecutive pair must correspond to a pre-established directed trust relation:

$$
( s _ { t } , s _ { t + 1 } ) \in \mathcal { E } , \qquad t = 1 , \dots , T - 1 .\tag{3}
$$

Equivalently, the protocol-authorized transmission from $s _ { t }$ to $s _ { t + 1 }$ is governed by

$$
\mathrm { T r u s t } ( s _ { t } , s _ { t + 1 } , X _ { s _ { t }  s _ { t + 1 } } ) .\tag{4}
$$

The sequence neither creates nor extends trust relations and it only composes preestablished directed trust relations into a feasible training route.

## 3.3.3 Sequential Training Protocol

Given a valid training sequence $S = [ s _ { 1 } , \dotsc , s _ { T } ]$ , TNFL maintains an evolving model state that is successively updated by the centers appearing in S. The first center $s _ { 1 }$ starts from the initial model parameters $\Theta _ { 0 }$ and performs local training on $\mathcal { D } _ { s _ { 1 } }$ . For each subsequent step $t = 2 , \ldots , T$ , center $s _ { t }$ receives the protocol-authorized message $X _ { s _ { t - 1 }  s _ { t } }$ from the preceding center under the corresponding directed trust relation. Let $\Theta _ { t - 1 }$ denote the model state available to center $s _ { t } ,$ with $\Theta _ { 0 }$ denoting the initial state for $s _ { 1 }$ . Each center then updates the received model using its local dataset:

$$
\Theta _ { t } = \operatorname { U p d a t e } \left( \Theta _ { t - 1 } , \mathcal { D } _ { s _ { t } } \right) , \qquad t = 1 , \ldots , T ,\tag{5}
$$

where $\mathrm { U p d a t e } ( \cdot )$ denotes the local training process. Each center therefore continues training from the model state accumulated through preceding centers rather than training an independent model from scratch. For $t < T ,$ the updated parameters $\Theta _ { t }$ are carried by the protocol message $X _ { s _ { t }  s _ { t + \cdot } }$ and transmitted to $s _ { t + 1 }$ under the corresponding directed trust relation. All transmissions follow the protocol-authorized message scope defined above, while local data and local optimization information remain at the corresponding center. In the main implementation, the auxiliary generative model is additionally propagated and updated together with the aging-clock model, as detailed in the subsequent training formulation.

## 3.3.4 Protocol Termination and Model Release

Protocol Termination. Given a predefined finite trust-constrained training sequence $S = [ s _ { 1 } , \dotsc , s _ { T } ]$ , the current TNFL protocol terminates after center $s _ { T }$ completes the final local update, yielding the final aging-clock model $\Theta _ { T }$ . The stopping condition is therefore defined by completion of the prescribed sequence, which may include repeated visits to a center when permitted by the trust-network topology.

Final Model Release. After termination, the final aging-clock model $\Theta _ { T }$ is released to the participating centers according to a predefined protocol-level release policy which can be represented as

$$
{ \mathrm { R e l e a s e } } ( \Theta _ { T } , \mathcal { V } ) .\tag{6}
$$

This release is treated as an output of the collaborative protocol rather than an additional pairwise training transmission. It therefore does not require the terminal center $s _ { T }$ to establish a direct trust relation with every participating center, nor does it imply that $s _ { T }$ is globally trusted. Accordingly, final model release is governed separately from the pairwise trust relations that constrain model propagation during training.

## 3.4 Model Architecture

## 3.4.1 AgeMoE Aging Clock Model

To model heterogeneous molecular patterns that arise across distributed client datasets, we implement an age-aware Mixture-of-Experts (AgeMoE) aging clock that maps tabular omics profiles to a scalar age prediction. The model combines a lightweight Transformer encoder with a soft MoE regression head, allowing multiple experts to specialize in different predictive patterns while a routing network dynamically determines their contributions for each sample. This design enables the model to capture diverse age-related signals across heterogeneous clients while maintaining a flexible and expressive prediction framework.

Input Representation. Omics datasets generally consist of high-dimensional numerical measurements, such as protein abundances or DNA methylation levels, along with a small number of categorical annotations, for example sex or tissue. We integrate both numerical and categorical features to capture complementary age-related signals that are informative for the aging clock. Directly combining raw numerical values with categorical indicators can hinder learning due to differences in scale and representation. To address this, we first project numerical features $\mathbf { x } ^ { \mathrm { n u m } } \in \mathbb { R } ^ { d _ { 1 } }$ into a shared embedding space:

$$
{ \bf e } _ { \mathrm { n u m } } = f _ { \theta _ { \mathrm { n u m } } } ( \bf x ^ { \mathrm { n u m } } ) ,\tag{7}
$$

where $f _ { \theta _ { \mathrm { n u m } } }$ is the mapping function parametrized with $\theta _ { \mathrm { n u m } } .$ which can generally be implemented as a simple linear transformation. For the categorical features ${ \bf x } ^ { \mathrm { c a t } } \in { }$ $\mathbb { R } ^ { d _ { 2 } }$ , we map each category $\mathbf { x } _ { c } ^ { \mathrm { c a t } } \in \mathbb { R }$ to a dense embedding vector using learnable feature-specific embedding tables $E m b _ { c }$ •·

$$
\mathbf { e } _ { \mathrm { c } } = E m b _ { c } ( \mathbf { x } _ { c } ^ { \mathrm { c a t } } ) , \quad c = 1 , . . . , d _ { 2 } .\tag{8}
$$

We then aggregate all embeddings into a unified representation:

$$
{ \bf h } _ { 0 } = \frac { 1 } { d _ { 2 } } \left( \sum _ { c = 1 } ^ { d _ { 2 } } { \bf e } _ { c } \mathrm { ~ + ~ } { \bf e } _ { \mathrm { n u m } } \right) .\tag{9}
$$

This unified representation $\mathbf { h } _ { 0 }$ serves as the input for subsequent model components, enabling integrated processing of both numerical and categorical features.

Transformer Encoder Backbone. To capture complex relationships among integrated feature representations, we employ a Transformer encoder as a flexible feature extractor. The self-attention mechanism enables the model to adaptively reweight feature interactions and learn expressive representations from high-dimensional omics inputs. Formally, given the aggregated input representation ${ \bf h } _ { 0 }$ , we first add a learnable positional embedding and pass the result through the Transformer encoder:

$$
\mathbf { h } = \mathrm { T r a n s f o r m e r E n c o d e r } _ { \boldsymbol { \theta } _ { t } } ( \mathbf { h } _ { 0 } \ + \ \mathbf { p } ) ,\tag{10}
$$

where p denotes the positional embedding and h represents the encoded feature vector produced by the Transformer backbone. The positional embedding introduces a consistent structural signal to the input representation, which facilitates stable representation learning across encoder layers. The resulting representation h is then used as the input to the subsequent MoE regression head for age prediction.

Mixture-of-Experts Regression for Aging Prediction. To model heterogeneous aging patterns in omics data, the AgeMoE aging clock employs a MoE regression module on top of the backbone representation. Since aging signals arise from multiple biological processes and exhibit diverse patterns across individuals, a single predictor may not capture all variability. To address this, we implement multiple expert heads and employ a learnable routing network that assigns input-dependent weights to each expert, enabling the model to combine specialized predictions into a final age estimate.

Given the encoded feature representation h from the Transformer backbone, we first compute routing weights using a routing network:

$$
\boldsymbol { \pi } = g _ { \boldsymbol { \theta } _ { r } } ( { \mathbf { h } } ) \in \mathbb { R } ^ { E } ,\tag{11}
$$

where $g _ { \theta _ { r } } ( \cdot )$ denotes the routing function and $E$ is the number of experts. We apply a softmax function to obtain normalized routing weights satisfying $\textstyle \sum _ { e = 1 } ^ { \bar { E } } \pi ^ { e } = 1$ . These weights determine how strongly each expert contributes to the final prediction.

We implement each expert as an independent regression head that maps the shared representation to an age estimate. For expert $e = 1 , . . . , E$ , the predicted age is

$$
\hat { y } ^ { e } = s _ { \phi _ { e } } ( \mathbf { h } ) .\tag{12}
$$

We then combine expert outputs with routing weights to produce the final prediction:

$$
\hat { y } = \sum _ { e = 1 } ^ { E } \pi ^ { e } \cdot \hat { y } ^ { e } .\tag{13}
$$

Loss Functions for AgeMoE Aging Clock. We train the AgeMoE predictor with two complementary objectives. The first is a base regression loss, which ensures accurate age estimation:

$$
\mathcal { L } _ { r e g } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } ( \hat { y } _ { i } ~ - ~ y _ { i } ) ^ { 2 }\tag{14}
$$

where $n _ { k }$ denotes the number of samples in center $k , \hat { y } _ { i }$ is the predicted age, and $y _ { i }$ is the true age for sample i.

Second, we incorporate an age-aware routing regularizer to encourage samples with similar ages to exhibit similar routing distributions. This constraint promotes smoother expert specialization across the age spectrum and helps stabilize the routing behavior of the MoE model. Let $\pmb { \pi } _ { i } \in \mathbb { R } ^ { E }$ denote the routing weights for sample i. For each sample, we consider its l nearest neighbors in the age space and measure the difference between their routing distributions using the symmetric KL divergence, whose symmetric formulation removes directional bias and provides a balanced measure of distributional similarity:

$$
D _ { \mathrm { s y m K L } } ( \pmb { \pi } _ { i } , \pmb { \pi } _ { j } ) = D _ { \mathrm { K L } } ( \pmb { \pi } _ { i } | | \pmb { \pi } _ { j } ) \ + \ D _ { \mathrm { K L } } ( \pmb { \pi } _ { j } | | \pmb { \pi } _ { i } ) .\tag{15}
$$

Then, the routing regularization loss is defined as:

$$
\mathcal { L } _ { t r } = \frac { 1 } { n _ { k } } \sum _ { i = 1 } ^ { n _ { k } } \frac { 1 } { l } \sum _ { j \in \mathcal { N } _ { l } ( i ) } \omega _ { i j } \cdot D _ { \mathrm { s y m K L } } ( \pi _ { i } , \pi _ { j } ) ,\tag{16}
$$

where $\mathcal { N } _ { l } ( i )$ denotes the set of l nearest neighbors of sample i in the age space and

$$
\omega _ { i j } = \exp ( - \frac { \left| y _ { i } - y _ { j } \right| } { \tau } ) ,\tag{17}
$$

assigns larger weights to pairs with closer ages, with τ controlling the decay rate.

The overall loss for training the AgeMoE aging clock is:

$$
\mathcal { L } _ { A g e M o E } ( \Theta ) = \mathcal { L } _ { r e g } + \lambda \mathcal { L } _ { t r } ,\tag{18}
$$

where $\Theta = \{ \theta _ { \mathrm { n u m } } , \{ E m b _ { c } \} _ { c = 1 } ^ { d _ { 2 } } , \theta _ { t } , \theta _ { r } , \{ \phi _ { e } \} _ { e = 1 } ^ { E } \}$ denotes the set of all parameters in the aging clock model, including the numerical feature mapping, categorical embedding tables, Transformer encoder, routing network, and expert heads, and λ is a hyperparameter that balances the regression and regularization terms.

## 3.4.2 Generative Replay for Cross-Center Knowledge Preservation

As the evolving aging-clock model is sequentially updated across heterogeneous centers, continued adaptation to new local data may weaken information acquired from previously visited centers. To preserve such information during cross-center learning, TNFL incorporates generative replay through an auxiliary conditional generative model. The generator synthesizes age-conditioned pseudo-samples representing information accumulated from preceding centers, which are replayed together with local data during subsequent model updates.

We implement the generative replay module using a conditional variational autoencoder (CVAE) that models the conditional distribution of omics features given age-related information. In addition to the age value, we introduce a discrete age-bin identifier b defined over the global age range with a fixed bin width. Conditioning on both the continuous age value and the age-bin indicator allows the model to capture coarse age-level information while preserving fine-grained variation within each bin. The CVAE consists of an encoder and a decoder parameterized by neural networks. The encoder maps the input sample and its conditioning variables to a latent representation:

$$
q _ { \psi } ( \mathbf { z } | \mathbf { x } , y , b ) ,\tag{19}
$$

where z denotes the latent variable. The decoder reconstructs the feature vector from the latent variable together with the conditioning information:

$$
p _ { \varphi } ( \mathbf { x } | \mathbf { z } , y , b ) .\tag{20}
$$

The model is trained by minimizing the following objective:

$$
\mathcal { L } _ { c v a e } ( \Psi ) = | | \mathbf { x } - \hat { \mathbf { x } } | | _ { 2 } ^ { 2 } \ + \ \beta D _ { \mathrm { K L } } ( q _ { \psi } ( \mathbf { z } | \mathbf { x } , y , b ) | | \mathcal { N } ( 0 , I ) ) ,\tag{21}
$$

where $\Psi \ = \ \{ \psi , \varphi \}$ denotes all parameters in the generative model, including the encoder parameters and the decoder parameters, and ê denotes the reconstructed feature vector generated by the decoder. The first term measures the reconstruction error between the input features and their reconstruction, while the second term regularizes the latent distribution toward a standard Gaussian prior. After training, the CVAE can generate pseudo-samples by sampling $\mathbf { z } \sim \mathcal { N } ( 0 , I )$ and decoding it together with specified age conditions $( y , b )$ . These synthesized samples are replayed alongside local data during subsequent model updates to help preserve information acquired from previously visited centers.

## 3.5 Training and Optimization

## 3.5.1 Joint Training of Aging Clock and Generative Model

In TNFL, the aging-clock and generative-model parameters are transmitted between successive centers through protocol-authorized messages along directed trust relations. To preserve information from previously visited clients while adapting to the current client's data, we use the generative replay mechanism introduced above. The generative model produces pseudo-samples approximating previously encountered client distributions, and the aging clock uses these samples together with local data during model updates. This strategy helps preserve previously acquired information as the aging clock and generator are updated sequentially across centers.

At the initial step, center $s _ { 1 }$ is assigned the model parameters $\Theta _ { 0 }$ and $\Psi _ { 0 }$ . At each subsequent step $t ~ > ~ 1$ , center $s _ { t }$ receives the protocol-authorized message $X _ { s _ { t - 1 } \dotsc s _ { t } } = ( \Theta _ { t - 1 } , \Psi _ { t - 1 } )$ from the preceding center $s _ { t - 1 }$ . For the initial center $s _ { 1 }$ pseudo-sample generation and prediction-consistency filtering are omitted because no preceding-center model state is available. The joint training procedure at each subsequent center consists of three steps:

## 1. Pseudo-sample generation and prediction-consistency filtering

We first generate pseudo-samples using the received generative model to represent information from previously visited clients. Specifically, we sample $ { n _ { p s e } }$ latent vectors $\mathbf { z } _ { i } \sim \mathcal { N } ( 0 , I )$ and decode feature vectors conditioned on target ages $y _ { i }$ and age-bin identifiers $b _ { i }$

$$
\hat { \mathbf { x } } _ { i } \sim p _ { \varphi _ { t - 1 } } ( \mathbf { x } | \mathbf { z } _ { i } , y _ { i } , b _ { i } ) , \quad i = 1 , . . . , n _ { p s e } ,\tag{22}
$$

where target ages and age-bin identifiers are sampled from the age range maintained by the generator. Because generated samples may vary in quality, we further apply prediction-consistency filtering to retain reliable pseudo-samples. The aging clock received from the preceding center serves as a reference predictor, $\hat { y } _ { r e f } ( \hat { \mathbf { x } } _ { i } ) = \mathcal { F } _ { \boldsymbol { \Theta } _ { t - 1 } } ( \hat { \mathbf { x } } _ { i } )$ . A pseudo-sample is kept if its predicted age is sufficiently consistent with the conditioning label:

$$
| \hat { y } _ { r e f } ( \hat { \mathbf { x } } _ { i } ) - y _ { i } | \leq \epsilon .\tag{23}
$$

The pseudo-samples satisfying this condition are retained to form the filtered set $\tilde { \mathcal { D } } _ { s _ { t } }$ . This filtering step excludes pseudo-samples with inconsistent age predictions and improves the reliability of generated supervision during subsequent training.

## 2. Aging clock update

After obtaining the filtered pseudo-samples $\tilde { \mathcal { D } } _ { s _ { t } } .$ we update the aging clock using them together with the current center's data $\mathcal { D } _ { s _ { t } }$ . The combined dataset allows the model to learn from the current center while retaining information from previously encountered client distributions. The aging clock parameters are optimized by minimizing the AgeMoE objective:

$$
\Theta _ { t } = \arg \operatorname* { m i n } _ { \Theta } \mathcal { L } _ { A g e M o E } \big ( \Theta ; \mathcal { D } _ { s _ { t } } \cup \tilde { \mathcal { D } } _ { s _ { t } } \big ) .\tag{24}
$$

Through this update, the generated samples provide complementary supervision for preserving information acquired from previously visited clients during sequential model updates.

## 3. Generative model update

Finally, we update the generative model from the received parameters $\Psi _ { t - 1 }$ using the current center's data $\mathcal { D } _ { s _ { t } }$ . This sequential update incorporates the current center's data distribution into the propagated generator for subsequent interactions. The model parameters are optimized by minimizing the CVAE objective:

$$
\Psi _ { t } = \arg \operatorname* { m i n } _ { \Psi } \mathcal { L } _ { c v a e } ( \Psi ; \mathcal { D } _ { s _ { t } } ) .\tag{25}
$$

The updated generative model is then passed along the training sequence together with the aging clock.

Overall, this joint training procedure allows the aging clock and generative model to be updated together as they propagate across centers through protocol-authorized transmissions under directed trust relations. Filtered pseudo-samples support replay from previously encountered client distributions, while sequential generator updates incorporate new local data for subsequent interactions.

## 3.5.2 Sequential Optimization Formulation

Given a valid training sequence ${ \cal S } = [ s _ { 1 } , \dots , s _ { T } ]$ , the TNFL training process can be summarized as a sequence of recursively initialized local optimization steps. Let $\mathcal { D } _ { s _ { t } }$ denote the local dataset of the center visited at step $t ,$ and let $\tilde { \mathcal { D } } _ { s _ { t } }$ denote the filtered pseudo-sample set used for generative replay. For the initial center $s _ { 1 }$ , no replay set from preceding centers is available, and we define $\tilde { \mathcal { D } } _ { s _ { 1 } } = \varnothing$ . For each subsequent step $t > 1$ , the filtered replay set is constructed as

$$
\begin{array} { r } { \tilde { \mathcal { D } } _ { s _ { t } } = \left\{ \left( \hat { x } _ { i } , y _ { i } \right) \big | \hat { x } _ { i } \sim p _ { \phi _ { t - 1 } } ( x \mid z _ { i } , y _ { i } , b _ { i } ) , \big | F _ { \Theta _ { t - 1 } } ( \hat { x } _ { i } ) - y _ { i } \big | \leq \epsilon \right\} . } \end{array}\tag{26}
$$

The aging-clock parameters are updated as

$$
\Theta _ { t } = \arg \operatorname* { m i n } _ { \Theta } \mathcal { L } _ { \mathrm { A g e M o E } } \left( \Theta ; \mathcal { D } _ { s _ { t } } \cup \tilde { \mathcal { D } } _ { s _ { t } } \right) , \qquad \Theta _ { t } ^ { ( 0 ) } = \Theta _ { t - 1 } ,\tag{27}
$$

and the generative-model parameters are updated as

$$
\Psi _ { t } = \arg \operatorname* { m i n } _ { \Psi } \mathcal { L } _ { \mathrm { c v a e } } \left( \Psi ; \mathcal { D } _ { s _ { t } } \right) , \qquad \Psi _ { t } ^ { ( 0 ) } = \Psi _ { t - 1 } ,\tag{28}
$$

Algorithm 1 Sequential Federated Training over a Directed Trust Network   
Require: Center datasets $\{ \mathcal { D } _ { k } \} _ { k = 1 } ^ { K }$ , initial parameters $\Theta _ { 0 } , \ \Psi _ { 0 } .$ number of pseudo  
samples $\boldsymbol { n _ { p s e } }$ , filtering threshold €, training sequence $\boldsymbol { S } = [ s _ { 1 } , . . . , s _ { T } ]$   
Ensure: Final aging-clock model $\Theta _ { T }$ and generative-model parameters $\Psi _ { T }$   
1: Initialize aging clock parameters $\Theta  \Theta _ { 0 }$   
2: Initialize generative model parameters $\Psi  \Psi _ { 0 }$   
3: for $t = 1$ to $T$ do   
4: Receive the authorized model-parameter message for center $s _ { t }$   
5: $\Theta _ { t } ^ { ( 0 ) }  \Theta$   
6: $\Psi _ { t } ^ { ( 0 ) }  \Psi$   
7: Pseudo-sample generation   
8: for $i = 1$ to $\boldsymbol { n _ { p s e } }$ do   
9: Sample latent vector $\mathbf { z } _ { i } \sim \mathcal { N } ( 0 , I )$   
10: Sample replay age conditions $( y _ { i } , b _ { i } )$   
11: Generate a pseudo sample $\hat { \mathbf { x } } _ { i }$ with Eq. (22)   
12: end for   
13: Prediction-consistency filtering   
14: Compute $\hat { y } _ { r e f } ( \hat { \mathbf { x } } _ { i } ) = \mathcal { F } _ { \Theta _ { t } ^ { ( 0 ) } } ( \hat { \mathbf { x } } _ { i } )$ for all generated pseudo-samples   
15: Construct the filtered pseudo-sample set $\tilde { \mathcal { D } } _ { s _ { t } }$ according to Eq. (23)   
16: Aging clock update   
17: Update aging clock parameters $\Theta _ { t }$ by optimizing the AgeMoE objective in   
Eq. (24)   
18: Set propagated aging clock parameters $\Theta  \Theta _ { t }$   
19: Generative model update   
20: Update generative model parameters $\Psi _ { t }$ by optimizing the CVAE objective in   
$\operatorname { E q . }$ (25)   
21: Set propagated generative model parameters $\Psi  \Psi _ { t }$   
22: end for   
23: $\Theta _ { T }  \Theta , \Psi _ { T }  \Psi$   
24: Final model release   
25: Release $\Theta _ { T }$ to all participating centers in V according to the predefined protocol  
level release policy in Eq. (6)   
26: return $\Theta _ { T }$

for $t = 1 , \ldots , T ,$ where $\Theta _ { 0 }$ and $\Psi _ { 0 }$ denote the initial aging-clock and generative-model parameters, respectively. Here, $\Theta _ { t }$ and $\Psi _ { t }$ denote the model parameters after the update at step t, while $\Theta _ { t } ^ { ( 0 ) }$ and $\bar { \Psi _ { t } ^ { ( 0 ) } }$ denote their local initialization before optimization. For $t > 1$ , these initializations are inherited from the protocol-authorized message transmitted by the preceding center under the corresponding directed trust relation. After the final update, TNFL outputs the aging-clock model $\Theta _ { T }$ The complete update procedure is summarized in Algorithm 1.

![](images/d13f78c0451f8fd735608d1a022037a729e2b998ac4afdf9eea5d20f270c3ea5.jpg)  
Fig. 7: Illustration of the sender-side authenticity and receiver-side confidentiality obligations under Trust $( A , B , X _ { A  B } )$

Table 2: Comparison of trust with semi-honest and fully honest models.
<table><tr><td>Setting</td><td>Requires A not to poison? not to poison? not to infer? not to infer?</td><td>Requires B</td><td>Requires A</td><td>Requires B</td></tr><tr><td>Both A and B are semi-honest</td><td>Yes</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Both A and B are fully honest</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Trust  $( A , B , X _ { A  B } )$ </td><td>Yes</td><td>No</td><td>No</td><td>Yes</td></tr></table>

## 3.6 Trust Characterization and Security Properties

The directed trust relation defines asymmetric and message-specific obligations for the sender and receiver, and therefore differs in scope from conventional participant-level behavioral models. This subsection clarifies this distinction and establishes a security property arising along sequences of separately established pairwise trust relations.

## 3.6.1 Relation to Semi-Honest and Fully Honest Models

Semi-honest and fully honest models characterize the behavior of an individual participant. A semi-honest participant follows the prescribed protocol and does not poison, forge, or tamper with protocol messages, but may use legitimately obtained information to infer private information. A fully honest participant additionally refrains from such inference. In contrast, Trust(A, $B , X _ { A  B } )$ is a directed and pairwise relation that assigns different obligations to the sender and receiver of a specific message. As illustrated in Fig. 7, the relation imposes sender-side authenticity on A and receiverside confidentiality on B. Building on this relation-level characterization, Table 2 applies the semi-honest and fully honest models to both A and B and contrasts their requirements with those imposed by Trust $( A , B , X _ { A  B } )$

## 3.6.2 Security Properties along Trust Sequences

TNFL composes multiple directed trust relations through sequential model propagation, whereas Definition 1 specifies a receiver-side confidentiality obligation at the level of an individual transmission. This subsection analyzes the security property induced by this composition and establishes downstream non-inference for upstream centers along a valid trust sequence.

Proposition 1 (Downstream Confidentiality along a Trust Sequence). Let

$$
{ \cal S } = [ v _ { 1 } , v _ { 2 } , \dots , v _ { T } ]
$$

be a sequence of participating centers, where $X _ { v _ { i }  v _ { i + 1 } }$ denotes the message transmitted from $v _ { i }$ to $v _ { i + 1 }$ . For every $i \in \{ 1 , . . . , T - 1 \}$ , the information about preceding model states exposed to $v _ { i + 1 }$ through the TNFL training protocol is contained in $X _ { v _ { i }  v _ { i + 1 } } . \mathrm { ~ I f ~ }$

$$
\mathrm { T r u s t } ( v _ { i } , v _ { i + 1 } , X _ { v _ { i }  v _ { i + 1 } } )
$$

holds for every $i \in \{ 1 , . . . , T - 1 \}$ , then, for any $1 \leq i < j \leq T$ , center $v _ { j }$ does not use its received message to infer the private information of center $v _ { i }$

Proof. We prove the proposition by induction on the number of centers n in a sequence prefix.

Base case $( n = 2 ) . \mathrm { ~ B y ~ }$

$$
\mathrm { T r u s t } ( v _ { 1 } , v _ { 2 } , X _ { v _ { 1 }  v _ { 2 } } )
$$

and the confidentiality requirement in Definition 1, center $v _ { 2 }$ does not use $X _ { v _ { 1 }  v _ { 2 } }$ to infer the private information of center $v _ { 1 }$ . Therefore, the proposition holds for the first two centers.

Induction hypothesis (n). Assume that, for the sequence prefix

$$
( v _ { 1 } , v _ { 2 } , \ldots , v _ { n } ) ,
$$

for any $1 \leq i < j \leq n .$ center $v _ { j }$ does not use its received message to infer the private information of center $v _ { i }$

Induction step $( n + 1 )$ . By the induction hypothesis, the proposition holds for the first n centers. Moreover, by

$$
\mathrm { T r u s t } ( v _ { n } , v _ { n + 1 } , X _ { v _ { n }  v _ { n + 1 } } ) ,
$$

the confidentiality requirement in Definition 1 ensures that $v _ { n + 1 }$ does not use $X _ { v _ { n }  v _ { n + 1 } }$ to infer the private information of center $v _ { n }$ . By Remark 1, $v _ { n + 1 }$ also does not use $X _ { v _ { n }  v _ { n + 1 } }$ to infer the private information of any preceding center $v _ { i } .$ where $1 \leq i < n$ . Therefore, for every $1 \leq i \leq n .$ center $v _ { n + 1 }$ does not use $X _ { v _ { n }  v _ { n + 1 } }$ to infer the private information of center $v _ { i }$ . Hence, the proposition holds for the first $n + 1$ centers.

By mathematical induction, for any $1 \leq i < j \leq T ,$ center $v _ { j }$ does not use its received message to infer the private information of center $v _ { i }$ □

## 3.7 Experimental Design and Evaluation Setup

Following the methodological formulation above, we next describe the experimental design used to evaluate TNFL. This includes the datasets and multi-center construction, model configurations, and evaluation settings used throughout the experiments.

## 3.7.1 Datasets and Multi-Center Construction

We use two molecular aging-clock datasets covering distinct omics modalities: proteomic data from the UK Biobank and DNA methylation data curated from the Gene Expression Omnibus (GEO). These datasets provide complementary molecular modalities for constructing the multi-center evaluation settings used in this study.

## Data Source

UK Biobank Proteomics Dataset. The UK Biobank proteomics dataset consists of 46,988 samples, with each sample characterized by expression measurements of 2,920 plasma proteins. The chronological age of participants ranges from 39 to 72 years.

GEO DNA Methylation Dataset. The DNA methylation dataset is assembled from publicly available studies deposited in the Gene Expression Omnibus (GEO), a widely used repository for functional genomics data. This dataset comprises 6,392 samples profiled using the Illumina HumanMethylation450 BeadChip array, with methylation levels measured at 485,514 CpG sites and participant ages spanning from 20 to 80 years. These samples were collected and curated through the EWAS Data Hub hosted by the National Genomics Data Center, which aggregates and standardizes DNA methylation array data and associated metadata from multiple public studies.

## Data Preprocessing

For both datasets, quality control and preprocessing are performed prior to model training. Molecular features with missing values in more than 10% of samples are first removed to mitigate the impact of sparsely observed signals. For the remaining features, missing values are imputed using mean values computed within the corresponding sample group; for the DNA methylation dataset, imputation is conducted separately within each tissue to preserve tissue-specific methylation patterns. After filtering and imputation, molecular features are standardized to ensure comparable scales across samples. For the UK Biobank proteomics dataset, standardization is performed across all samples, whereas for the GEO methylation dataset it is carried out within each tissue type to account for inherent tissue-specific differences in molecular measurements.

In addition, for the GEO dataset, CpG sites are further filtered based on their age association. For each site, Pearson correlations between methylation levels and chronological age are computed within individual tissues, and a weighted average correlation is obtained using tissue sample proportions as weights. Sites are retained only if their weighted correlation exceeds a data-driven threshold determined from the upper quantile of the correlation distribution and if the threshold is satisfied in at least 50% of tissues. This procedure preserves CpG sites with consistent age-related signals across multiple tissues while reducing noise from weakly associated loci.

## Multi-Center Construction

![](images/26ba67e3697d57c8ac3a77c77c266ce52fc6c8e20bdc70ad59360668143da39d.jpg)  
(a) UKB-Center

![](images/3f73f2c1d9984b53c3f81666c65e95c40b91fa2017e7dc2a9752710a1068e685.jpg)  
(b) UKB-Imbalance

![](images/c608fe78a2694ff91ef40a51aff871d84e23655b404087c210166abd00b999c3.jpg)  
(c) GEO-Methylation

![](images/2d4cd21e9bbbf7adf1ff6ef89ef3dea63288d1f4e7c997587f5887f5955319e3.jpg)  
(d) UKB-AgeSplit  
Fig. 8: Distribution of client sample sizes in constructed multi-center datasets.

To emulate realistic multi-center learning environments, we construct federated clients by partitioning each dataset according to practical data-distribution scenarios. A summary of the constructed multi-center settings is provided in Fig. 8, which presents the sample size distribution across clients for each dataset.

UKB-Center. For the UK Biobank proteomics dataset, samples are grouped based on their recruitment assessment centers. This partitioning naturally yields 18 clients corresponding to geographically distributed collection sites, reflecting the data silos typically encountered in large-scale biomedical collaborations.

UKB-Imbalance. In addition, to evaluate model robustness under severe client-level data imbalance, we construct an alternative setting derived from the UK Biobank dataset. In this scenario, a subset comprising 21,600 samples is organized into 10 clients with highly skewed sample sizes. The resulting distribution exhibits substantial heterogeneity, with the smallest client containing only 300 samples, thereby mimicking extreme imbalance conditions that may arise in real-world collaborative studies.

GEO-Methylation. For the GEO DNA methylation dataset, the overall sample size is comparatively smaller and the data originate from multiple studies with heterogeneous sample compositions. To reflect these characteristics, the dataset is partitioned into 4 clients with markedly imbalanced sample sizes, representing multicenter environments in which participating centers contribute datasets of varying scales.

## 3.7.2 Model Configurations

All aging clock models are formulated as supervised regression models that map highdimensional molecular feature vectors to chronological age. To ensure a controlled comparison within the TNFL framework, model architectures and hyperparameters are fixed across centers within each experiment.

Conventional Machine Learning Models. Linear regression and XGBoost are used as representative non-neural aging clock backbones. Linear regression models age as a linear function of molecular features with an l1-regularized objective, providing a simple low-capacity reference. XGBoost serves as a stronger tree-based baseline by learning an additive ensemble of regression trees under squared-error loss.

Deep Learning Models. We further consider plain MLP and Transformer backbone algorithms under the TNFL pipeline. The MLP consists of three fully connected layers with GeLU activations, serving as a standard feedforward network for tabular omics data. The Transformer employs a lightweight self-attention encoder with multiple attention heads and position-wise feedforward layers, adapted for tabular input.

AgeMoE Aging Clock. Our AgeMoE aging clock consists of a Transformer-based encoder followed by a mixture-of-experts regression head. A CVAE-based generative replay module synthesizes pseudo-samples for replay during subsequent training within the TNFL pipeline. These pseudo-samples, conditioned on age and discrete age-bin identifiers and filtered to retain only those whose predicted ages match the conditioning label within a tolerance ε (set to 1 year), are used alongside the current client's real data to train the aging clock. The specific model and training configurations are as follows.

For the UK Biobank experiments, we adopt a lightweight configuration. The aging clock uses a single-layer Transformer encoder with hidden dimension 64, one attention head, and feed-forward dimension 128, followed by a MoE head with 6 experts of hidden size 256. Training is performed with a batch size of 512 and a learning rate of 1e – 4, with a local budget of 40 epochs per stage and early stopping (patience of 5 epochs). The generative replay module produces pseudo-samples whose number is defined relative to the size of the current client's dataset. Specifically, the replay ratio increases linearly from 0.3 to 0.5 over the training process, meaning that the number of pseudo-samples grows from 30% to 50% of the real samples at each client. The generated samples are distributed across 5-year age bins to ensure coverage of the age range. The CVAE used for generation employs a latent variable of dimension 32; both the encoder and decoder are implemented as two-layer MLPs with hidden dimensions 256 and 128, and take as input the molecular features concatenated with a 16-dimensional conditioning vector. The model is trained with a learning rate of $5 e - 4 ,$ a batch size of 256, and 30 training epochs at each client.

For the GEO experiments, we adopt a higher-capacity configuration to accommodate increased data heterogeneity. The aging clock uses a Transformer encoder with hidden dimension 256, 4 attention heads, and a feed-forward dimension of 512, followed by a MoE head with 12 experts of hidden size 512. Training is performed with a batch size of 512 and a learning rate of $1 e - 4$ , with up to 80 local epochs at each client and early stopping (patience of 20 epochs). The replay mechanism follows the same design as above, where the number of pseudo-samples is defined relative to the size of the current client's dataset. The replay ratio increases linearly from 0.2 to 0.5, meaning that the number of generated samples grows from 20% to 50% of the real samples at each client. The generated samples are distributed across 5-year age bins. The CVAE used for generation employs a latent variable of dimension 128; both the encoder and decoder are implemented as two-layer MLPs with hidden dimensions 512 and 256, and take as input the molecular features concatenated with a 64-dimensional conditioning vector. The model is trained with a learning rate of 3e — 4, a batch size of 512, and 40 training epochs at each client.

Training Configuration Consistency. For neural network-based aging clocks, including the MLP, Transformer, and AgeMoE models, we keep the training protocol, optimizer, and major hyperparameter settings consistent where applicable, while retaining model-specific training objectives. This design ensures that performance differences primarily reflect model architecture rather than unrelated training settings. Conventional machine learning baselines, including linear regression and XGBoost, are trained using their standard optimization procedures with fixed hyperparameter settings.

## 3.7.3 Evaluation Setup

Data Splits. For each client, available samples are randomly partitioned into training, validation, and test sets with a ratio of 70:10:20. Local test sets are used to evaluate client-specific predictive performance, while a global test set is constructed by aggregating the test samples from all clients to assess overall generalization.

Comparison Methods. We compare TNFL with two reference settings. Local training independently trains an aging clock at each center using only its local data, providing a no-collaboration reference, whereas FedAvg provides a standard serverbased federated learning reference with centralized model aggregation. TNFL is evaluated under the trust-network-constrained setting defined above, where model propagation follows valid trust-constrained sequences composed of feasible pairwise trust relations.

Evaluation Metrics. Predictive performance is evaluated using the mean absolute error (MAE). The metric is computed on both local test sets and the global test set to characterize client-level predictive accuracy as well as overall model performance. Interaction-order robustness is further assessed using client-level MAE range, cumulative MAE degradation, and forgetting rate, with detailed definitions provided in Appendix C.

Interaction Order Generation. To examine the sensitivity of the TNFL protocol to interaction order across valid trust-constrained sequences, multiple valid singlepass trust sequences are instantiated from random client permutations. Specifically, 500 sequences are sampled for the UKB-Center dataset, 50 sequences for the UKB-Imbalance dataset, and 24 sequences for the GEO-Methylation dataset. The number of sequences is determined in accordance with the client scale of each dataset; for the GEO-Methylation dataset with four clients, all possible interaction permutations are evaluated. Model performance is evaluated under each sequence to quantify the robustness of TNFL to different interaction orders.

Computational Environment. All experiments are conducted on a server equipped with an NVIDIA RTX 5090 GPU (32 GB memory) and an Intel Xeon Platinum 8470Q CPU with 25 virtual cores. The software environment is based on Python 3.12 running on Ubuntu 22.04, with model training implemented using PyTorch 2.8.0 and CUDA 12.8 for GPU acceleration.

## 4 Declarations

• Funding: This work was supported in part by the PolyU Start-up Fund (No. P0059983), the Presidential Young Scholar Scheme (Project No. P0056638), the Research Institute for Artificial Intelligence of Things (RIAIoT) at The Hong Kong Polytechnic University (Project No. P0059914), and the Research Institute for Federated Learning (4CG00) at The Hong Kong Polytechnic University. This work also received support from the Joint Research Centre for AI Aging Science and Medicine under the PolyU Academy for Artificial Intelligence. The Department of Aging Science and Medicine is an endowed department supported and funded by GAKKEN HOLDINGS CO., LTD. and Medical Care Service Inc.

• Competing interests: The authors declare no competing interests.

• Data availability statement: The data used in this study include publicly available and controlled-access datasets. DNA methylation data were obtained from publicly available GEO-derived resources. The UK Biobank data are available under an approved application and are not publicly available due to access restrictions.

• Code availability: The complete code used in this study is provided in the Supplementary Materials accompanying this manuscript.

• Consent for publication: Not applicable. This study does not report identifiable individual-level information.

• Author contribution: Qiang Yang conceived the main idea and supervised the study. Chunxu Zhang and Bo Li designed the method, developed the model, conducted experiments, analyzed results, and drafted the manuscript. Liu Yang, Di Jiang and Bo Yang provided methodological suggestions and contributed to manuscript revision. Wenliang Wang and Yuan Huang contributed to biological analysis, functional annotation, and interpretation of model-identified molecular signals. Yo-ichi Nabeshima and Akinori Yamamura provided expert biological interpretation from the perspective of aging biology and reviewed the biological relevance of the identified molecular modules and higher-order organization. All authors reviewed and approved the manuscript.

## Appendix A Details About Biological Analysis of Key Molecular Features

## A.1 Selection of Key Omics Features via Age-Dependent Expert Sensitivity

To identify important molecular features associated with aging, we leveraged the routing weight of an expert that increases with age, providing an aging-related reference representation. This age-dependent behavior indicates that the routing weight of this expert reflects an age-associated routing pattern, which serves as a data-driven basis for assessing feature contributions to aging-related model behavior.

For each protein (or CpG site in the methylation data), we conducted a controlled perturbation analysis. Specifically, each feature was independently varied across ten quantile levels derived from its empirical distribution, while all other features were held fixed. The resulting changes in expert routing weight were recorded. Feature influence was quantified as the difference in expert routing weight between the highest and lowest quantile levels of each feature, and features were subsequently ranked based on this score.

## A.2 Pathway-Gene Composition of Enriched Functional Clusters

To complement the pathway enrichment analysis of the top-ranked proteins derived from the UKB-Center proteomic dataset, we provide the full gene-level composition of all significantly enriched ontology clusters. Specifically, each of the 20 enriched pathways identified in the main text is further decomposed to explicitly list the constituent proteins contributing to each functional module. This enables direct inspection of how individual proteins map onto biologically coherent processes.

## A.3 Model-based Identification of Synergistic Protein Pairs

To investigate coordinated protein effects associated with aging, we analyzed pairwise protein interactions learned by the aging clock model. Based on the important proteins identified in the previous section, we constructed candidate protein pairs by exhaustively combining these proteins into all possible two-protein combinations. We then evaluated how each protein pair jointly influenced the biological age estimated by the model on the global test population.

For a protein pair $( i , j )$ , we quantified its interaction effect by comparing model outputs under four perturbation states generated on the global test population. Specifically, we set the values of the corresponding proteins to zero across all samples to simulate suppression while leaving all remaining proteins unchanged. The four conditions included: both proteins suppressed $( f _ { 0 0 } )$ , only protein i retained at its original values while protein $j$ was suppressed $\left( f _ { 1 0 } \right)$ , only protein $j$ retained while protein i was suppressed $\left( f _ { 0 1 } \right)$ , and both proteins retained at their original values $\left( f _ { 1 1 } \right)$ . Based on these four conditions, we defined the pairwise synergy score as:

$$
\mathrm { S y n e r g y } _ { i j } = f _ { 1 1 } - f _ { 1 0 } - f _ { 0 1 } + f _ { 0 0 } .\tag{A1}
$$

A positive synergy score indicates that the joint contribution of the two proteins exceeds the sum of their individual effects, suggesting coordinated influence on agingrelated prediction patterns captured by the model. In contrast, negative synergy scores suggest redundant or antagonistic relationships between the two proteins. We ranked all candidate protein pairs according to their synergy scores and focused the main-text analysis on the top 10 synergistic protein pairs.

## A.4 Model-based Discovery of Higher-Order Protein Interactions

Beyond pairwise protein interactions, we further investigated whether the aging clock model captures coordinated higher-order protein organization associated with aging. To this end, we constructed candidate protein sets from model-identified important proteins for subsequent joint analysis of their collective effects.

To characterize protein coordination from multiple complementary biological perspectives, we performed higher-order synergy analysis under four analytical settings. Specifically, we considered interactions computed from either the final biological age prediction output or the latent expert-routing output of the AgeMoE model, enabling the analysis to capture both phenotype-level coordination and latent functional specialization patterns. In addition, we evaluated these interactions using either the full population distribution or age-restricted young subsets. The full-population setting prioritizes globally stable coordination patterns shared across heterogeneous individuals, whereas the young-subset setting is more sensitive to early-stage or age-dependent interaction signals that may be diluted in the overall population. Together, these complementary settings allow the analysis to capture both globally conserved and context-specific higher-order aging organization.

The group construction process follows a beam-search-style strategy that incrementally builds protein sets from informative starting pairs. Specifically, candidate protein pairs with strong pairwise synergy are first used as initial seeds. These seed pairs are then progressively expanded by adding one protein at a time, forming larger candidate groups. At each expansion step, candidate groups are evaluated based on their overall collective effect under the model, and only a subset of the most promising group configurations is retained for further expansion. Across the four analytical settings, this procedure yields a total of 16 representative protein groups.

## A.5 Biological Analysis on Methylation Data

We perform complementary biological analyses on the GEO DNA methylation dataset to assess whether the molecular patterns observed in the proteomic data extend across modalities. Feature importance is defined based on the magnitude of changes in expert routing weights induced by systematic perturbations, consistent with the definition used in the main text. Based on this criterion, the top 300 CpG sites are selected for downstream analysis, representing the most influential methylation features associated with expert routing behavior.

We first characterize the genomic distribution of these CpG sites using Illumina HumanMethylation450K hg19 annotation [48]. All CpGs are successfully mapped and are predominantly located in gene-associated regions and CpG islands, indicating a non-random distribution across genomic elements. We then map these CpGs to approximately 245 genes and perform functional enrichment and network analyses using Metascape [26] and STRING [30]. As shown in Fig. A1 (a), the enriched terms include cell adhesion, developmental processes, cell-cycle regulation, MAPK signaling, and immune-related functions. To further characterize these features, we perform external annotation analyses. As shown in Fig. A1 (b), DisGeNET [49] associations show links to hematological traits such as neutrophil and basophil counts, suggesting relevance to immune-related processes. Transcription factor target analysis further identifies enrichment for FOXO3 targets [50], a transcription factor previously implicated in longevity and aging-related regulation, which is summarized in Fig. A1 (c).

![](images/10912a8394b287f0532f87b64012ff640787422aeadc2acb06c09e92e5d2a8fb.jpg)

(a) Pathway and process enrichment analysis.  
![](images/93413754916b0914934c8b4330178cc69c4ff4c465c644e1b8bc9524e307ea76.jpg)

![](images/7144286d6c4306ae719a435e12d9d36b39cbfebb138710cf8fba3a9b224e5c26.jpg)  
(c) Summary of enrichment analysis in Transcription Factor Targets  
(b) Summary of enrichment analysis in DisGeNET.  
Fig. A1: Functional enrichment analysis of aging-related signals identified from the GEO methylation dataset.

Overall, these results show that functionally relevant biological processes are identified in both modalities, suggesting that the model captures meaningful molecular signals across different omics data types.

## Appendix B Analysis of the Generative Component

The AgeMoE model integrates a generative replay module that preserves information from previously visited clients and augments local training with pseudo-samples during cross-client updates. To assess its contribution under the TNFL training process, we compare the full model with a variant without the generative module, while keeping all other training settings unchanged. As the generative replay component influences how information from different clients is accumulated during collaborative training, its effect is most clearly reflected in the model's overall generalization across centers. We therefore evaluate performance on the global test set aggregated from all clients.

Experiments are conducted on the three datasets used in the main study, including the UKB-Center, UKB-Imbalance, and GEO-Methylation datasets. In addition, we construct an additional dataset from the UK Biobank cohort to simulate stronger cross-client distributional differences, referred to as UKB-AgeSplit. In this dataset, samples are partitioned into clients according to age intervals of five years (e.g., 40–45, 45–50), resulting in six clients, each containing individuals within a specific age range. This setting reflects practical scenarios in which medical centers serve distinct patient populations, such as pediatric or geriatric hospitals, leading to systematic differences in data distributions across centers.

![](images/10efe7792db8d263aa36179d0aa8ceb573c0193814d534aba8d17d0514ee0c60.jpg)  
(a) UKB-Center

![](images/515a12cd517aea4a7a9f23364346328a0604f07a64653b02d10a2d187c12b8fd.jpg)  
(b) UKB-Imbalance

![](images/1b42c31523167707ee2ed8884963c4fff09cf812ca52872da94d01c5c8269d73.jpg)  
(c) GEO-Methylation

![](images/52fcb967897edfaab3c5883a9da2fcba059316e7beb8795638ae3951fe86b0bb.jpg)  
(d) UKB-AgeSplit  
Fig. B2: Performance comparison of AgeMoE with (w/ CVAE) and without (w/o CVAE) the generative component on global test sets across multiple datasets.

Fig. B2 reports the global test performance of the two model variants across all datasets. Incorporating the generative component leads to improved global predictive performance, with larger gains observed for the GEO-Methylation dataset and the age-partitioned UK Biobank dataset, both of which exhibit stronger cross-client distributional differences. These observations indicate that the generative component is particularly beneficial when training involves heterogeneous client data. By retaining representative information from previously visited clients and augmenting local updates with pseudo-samples, the generative model helps maintain coverage of the overall data distribution and preserve information acquired from earlier clients, leading to improved global generalization.

## Appendix C Details on Robustness Metrics

This section provides formal definitions and calculation procedures for the three robustness metrics used to evaluate TNFL under varying interaction orders over the trust network. The metrics quantify how sensitive the model's predictive performance is to different sequences of client interactions.

## C.1 Client-level MAE Range

For each client $k ,$ we compute the range of MAE values across all interaction orders on the local test set. Let $\mathrm { \bar { M A E } } _ { k } ^ { ( i ) }$ denote the MAE of client k under interaction order $i ,$ then the client-level range is defined as:

$$
{ \mathrm { R a n g e } } _ { k } = \operatorname* { m a x } _ { i } { \mathrm { M A E } } _ { k } ^ { ( i ) } - \operatorname* { m i n } _ { i } { \mathrm { M A E } } _ { k } ^ { ( i ) } .\tag{C2}
$$

Smaller ${ \mathrm { R a n g e } } _ { k }$ indicates more stable performance across interaction sequences. To visualize these ranges, we aggregate ${ \mathrm { R a n g e } } _ { k }$ values across all clients for each model and plot them as boxplots, which show the distribution of per-client MAE variability. Each box represents the spread of ${ \mathrm { R a n g e } } _ { k }$ values across clients, providing an intuitive comparison of robustness among models.

## C.2 Cumulative MAE Degradation

Cumulative MAE degradation quantifies how sequential center updates affect the overall model performance. To capture the effect across all clients rather than individual local behaviors, MAE is measured on the global test set. Let $\mathrm { M A E } _ { \mathrm { g l o b a l } } ^ { ( i , j ) }$ denote the global MAE after the j-th client update in the i-th interaction order. Then the cumulative degradation for order i is defined as:

$$
\mathrm { D e g r a d a t i o n } ^ { ( i ) } = \sum _ { j = 2 } ^ { T } \mathrm { m a x } ( 0 , \mathrm { M A E } _ { \mathrm { g l o b a l } } ^ { ( i , j ) } - \mathrm { M A E } _ { \mathrm { g l o b a l } } ^ { ( i , j - 1 ) } ) ,\tag{C3}
$$

where $T$ is the number of update steps. Lower Degradation $\mathbf { \Psi } ( i )$ indicates that the sequential updates do not substantially increase error. To visualize these cumulative degradation values, we collect Degradation(i) across all interaction orders for each model and plot them as boxplots. Each box represents the distribution of cumulative MAE increases across sequences, providing a concise comparison of how different models accumulate errors during sequential updates.

## C.3 Forgetting Rate

Forgetting quantifies the relative performance loss of clients over the course of sequential updates. To focus on the retention of knowledge at the client level, we compare each center's MAE immediately after its own update with its MAE under the final model produced at the end of the same propagation sequence. Specifically, for client k appearing at position $j$ in order $i ,$ let $\mathrm { M A E } _ { k } ^ { ( i , j ) }$ and $\mathrm { M A E } _ { k } ^ { ( i , T ) }$ denote the MAE after its own update and at the final model, respectively. The forgetting rate is then defined as:

$$
\mathrm { F o r g e t t i n g } _ { k } ^ { ( i ) } = \frac { \mathrm { M A E } _ { k } ^ { ( i , T ) } - \mathrm { M A E } _ { k } ^ { ( i , j ) } } { \mathrm { M A E } _ { k } ^ { ( i , j ) } } \times 1 0 0 \%\tag{C4}
$$

Positive values indicate that the client's performance deteriorates over subsequent updates, while negative values suggest improvement or retention of knowledge. To focus on early-stage effects, we calculate $\mathrm { F o r g e t t i n g } _ { k } ^ { ( i ) }$ only for the first half of clients in each sequence, as these clients are more likely to be affected by subsequent updates. To visualize client-level forgetting, we collect $F _ { k } ^ { ( i ) }$ for all clients in the first half of each sequence and plot them as boxplots. Each box captures the distribution of these forgetting rates across sequences, enabling comparison of how different models vary in client-level performance retention.

## Appendix D Details About Performance Under Simulated Measurement Variability

To assess TNFL's stability under simulated multi-center measurement variability, we conducted a perturbation study using the UKB-Center dataset. We simulate measurement variability by modifying each feature with two components: (1) Client-level shift: a systematic deviation shared by all samples from a given center, modeling biases arising from instrument calibration, experimental protocols, batch effects, or site-specific measurement tendencies; (2) Sample-level noise: an independent fluctuation applied to individual samples, capturing variability due to sample processing, measurement precision, or local technical noise. For each feature, the perturbed value was:

$$
{ \bf x } _ { p e r t u r b e d } = { \bf x } _ { o r i g i n a l } + \mathrm { c l i e n t . s h i f t } + \mathrm { s a m p l e . n o i s e } ,\tag{D5}
$$

with the perturbation magnitudes set to introduce controlled inter- and intra-center variability (client shift \~35% of the global feature standard deviation, sample noise \~12%). This choice introduces controlled deviations representing plausible inter- and intra-center differences.

Predictive performance was evaluated on individual client and global test sets, with results summarized in Table D1. Minimal performance decline was observed under these perturbations, with the global mean MAE increasing by only 0.07 (from 2.33 on the original data to 2.40 on the perturbed data), indicating that TNFL maintains stable predictive performance under the simulated measurement variability.

## Appendix E Details About Performance Across Trust-Network Structures

In the main experiments, interaction sequences are constructed by randomly sampling permutations of clients, providing a controlled setting to evaluate order-dependent variability when each client participates once in a training sequence. To further examine how explicit trust-network structures affect model behavior, we conduct additional experiments on the UKB-Center dataset using trust-network-derived interaction sequences. Specifically, we generate random trust networks over clients with controlled sparsity ranging from 0.1 to 0.5 to simulate varying degrees of inter-client trust connectivity, and then identify propagation sequences of variable lengths that cover all clients while respecting directed trust relations. These sequences may include repeated visits to centers when necessary, reflecting feasible interaction patterns in sparsely connected trust networks. The resulting sequences are used in TNFL without modifying the underlying sequential update mechanism or training protocol.

Table D1: MAE of TNFL on individual clients and the global test set under original and perturbed data. Results are split into two groups for readability.
<table><tr><td>Client</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td rowspan="3"></td></tr><tr><td>Original</td><td>2.2714</td><td>2.2662</td><td>2.2637</td><td>2.2990</td><td>2.2330</td><td>2.1098</td><td>2.4706</td><td>2.3462</td><td>2.4036</td></tr><tr><td>Perturbed</td><td>2.3878</td><td>2.3677</td><td>2.3244</td><td>2.3749</td><td>2.3070</td><td>2.1959</td><td>2.5332</td><td>2.4181</td><td>2.5120</td></tr><tr><td>Client</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td><td>15</td><td>16</td><td>17</td><td>Global</td></tr><tr><td>Original</td><td>2.3677</td><td>2.4112</td><td>2.2739</td><td>2.3568</td><td>2.2513</td><td>2.2841</td><td>2.4482</td><td>2.2943</td><td>2.5144</td><td>2.3259</td></tr><tr><td>Perturbed</td><td>2.4128</td><td>2.4621</td><td>2.3313</td><td>2.4651</td><td>2.2944</td><td>2.3450</td><td>2.5705</td><td>2.3496</td><td>2.5892</td><td>2.4023</td></tr></table>

In this setting, we focus on predictive performance and robustness to interaction patterns. Experiments on model compatibility with different aging-clock backbones are not repeated, as they primarily evaluate the flexibility of the TNFL propagation strategy with respect to model choice and are not directly determined by how interaction sequences are instantiated. All experiments are conducted on the UKB-Center dataset, which provides a representative setting due to its relatively large number of clients and heterogeneous data distributions across centers.

We first compare predictive performance across four settings: independent local training, FedAvg, TNFL with randomly sampled interaction orders, and TNFL with trust-network-derived interaction sequences. As shown in Fig. E3 (a), the performance patterns remain consistent across these settings. TNFL achieves similar predictive performance under both sequence-generation settings, while maintaining its relative advantages over local training and remaining competitive with FedAvg.

We further evaluate robustness using the same three metrics as in the main study. As shown in Fig. E3 (b)-(d), the models exhibit consistent stability patterns under trust-network-derived interaction sequences. TNFL maintains low variability across interaction arrangements, limited cumulative degradation, and no observable increase in forgetting effects. The cumulative degradation metric aggregates performance changes along the propagation sequence and is therefore naturally dependent on sequence length; since trust-network-derived interaction sequences may involve repeated visits to clients and thus longer propagation sequences, the absolute magnitude is not directly comparable in scale to that under random orders. Nevertheless, the observed cumulative degradation remains small, with a median value below 0.5, indicating stable performance progression throughout training. Overall, these results are consistent with those obtained under randomly sampled interaction orders.

![](images/6c0c2e0b5449e6f22d113163ab21823c3cbe2828e28c714d17b5024f2e44db87.jpg)  
(a) Multi-center aging clock performance

![](images/da6fbce0e34e517c3ea2cbe9d6e8db94573416a4ea69cc43634be049d8c66a31.jpg)  
(b) Robustness: Client performance range

![](images/6c3ec9c7d444108dc3675dbaf6c66375ae6b59d4bacf347797f8796512278ff1.jpg)  
(c) Robustness: Cumulative degradation

![](images/a3a40eb0e2ad29c44851508362891e704163e866380db1f188a4301624f988f1.jpg)  
(d) Robustness: Forgetting rate  
Fig. E3: Performance comparison and robustness evaluation under trust-networkderived interaction sequences.

Overall, these results suggest that TNFL shows broadly consistent behavior across different valid interaction sequences instantiated from the underlying trust network, with predictive performance and robustness metrics exhibiting similar trends across random-permutation and trust-network-derived sequence-generation schemes.

## References

[1] Jylhävä, J., Pedersen, N.L., Hägg, S.: Biological age predictors. EBioMedicine 21, 29–36 (2017)

[2] Argentieri, M.A., Xiao, S., Bennett, D., Winchester, L., Nevado-Holgado, A.J., Ghose, U., Albukhari, A., Yao, P., Mazidi, M., Lv, J., et al.: Proteomic aging clock predicts mortality and risk of common age-related diseases in diverse populations. Nature medicine 30(9), 2450–2460 (2024)

[3] Fransquet, P.D., Wrigglesworth, J., Woods, R.L., Ernst, M.E., Ryan, J.: The epigenetic clock as a predictor of disease and mortality risk: a systematic review and meta-analysis. Clinical epigenetics 11(1), 62 (2019)

[4] Warner, B., Ratner, E., Datta, A., Lendasse, A.: A systematic review of phenotypic and epigenetic clocks used for aging and mortality quantification in humans. Aging (Albany NY) 16(17), 12414 (2024)

[5] Rutledge, J., Oh, H., Wyss-Coray, T.: Measuring biological age using omics data. Nature reviews genetics 23(12), 715–727 (2022)

[6] Min, M., Egli, C., Dulai, A.S., Sivamani, R.K.: Critical review of aging clocks and factors that may influence the pace of aging. Frontiers in aging 5, 1487260 (2024)

[7] Bonomi, L., Huang, Y., Ohno-Machado, L.: Privacy challenges and research opportunities for genomic data sharing. Nature genetics 52(7), 646–654 (2020)

[8] Wan, Z., Hazel, J.W., Clayton, E.W., Vorobeychik, Y., Kantarcioglu, M., Malin, B.A.: Sociotechnical safeguards for genomic data privacy. Nature Reviews Genetics 23(7), 429–445 (2022)

[9] Zhang, X., Li, F., Zhang, Z., Li, Q., Wang, C., Wu, J.: Enabling execution assurance of federated learning at untrusted participants. In: IEEE INFOCOM 2020-IEEE Conference on Computer Communications, pp. 1877–1886 (2020). IEEE

[10] Chen, D., Wang, K., Sundar, A.P., Li, F.: Trustnetfl: enhancing federated learning with trusted client aggregation for improved security. In: Proceedings of the Twenty-fourth International Symposium on Theory, Algorithmic Foundations, and Protocol Design for Mobile Networks and Mobile Computing, pp. 534–539 (2023)

[11] He, C., Tan, C., Tang, H., Qiu, S., Liu, J.: Central server free federated learning over single-sided trust social networks. arXiv preprint arXiv:1910.04956 (2019)

[12] Warnat-Herresthal, S., Schultze, H., Shastry, K.L., Manamohan, S., Mukherjee, S., Garg, V., Sarveswara, R., Händler, K., Pickkers, P., Aziz, N.A., et al.: Swarm learning for decentralized and confidential clinical machine learning. Nature

[13] Sun, T., Li, D., Wang, B.: Decentralized federated averaging. IEEE Transactions on Pattern Analysis and Machine Intelligence 45(4), 4289–4301 (2022)

[14] Chang, K., Balachandar, N., Lam, C., Yi, D., Brown, J., Beers, A., Rosen, B., Rubin, D.L., Kalpathy-Cramer, J.: Distributed deep learning networks among institutions for medical imaging. Journal of the American Medical Informatics Association 25(8), 945–954 (2018)

[15] Kamp, M., Fischer, J., Vreeken, J.: Federated learning from small datasets. In: The Eleventh International Conference on Learning Representations

[16] Wang, N., Deng, Y., Feng, W., Fan, S., Yin, J., Ng, S.-K.: One-shot sequential federated learning for non-iid data by enhancing local model diversity. In: Proceedings of the 32nd ACM International Conference on Multimedia, pp. 5201–5210 (2024)

[17] Galkin, F., Mamoshina, P., Kochetov, K., Sidorenko, D., Zhavoronkov, A.: Deepmage: a methylation aging clock developed with deep learning. Aging and disease 12(5), 1252 (2021)

[18] Lima Camillo, L.P., Lapierre, L.R., Singh, R.: A pan-tissue dna-methylation epigenetic clock based on deep learning. npj Aging 8(1), 4 (2022)

[19] Vijayakumar, K.A., Cho, G.-w.: Pan-tissue methylation aging clock: recalibrated and a method to analyze and interpret the selected features. Mechanisms of Ageing and Development 204, 111676 (2022)

[20] Martínez-Enguita, D., Dwivedi, S.K., Jörnsten, R., Gustafsson, M.: Ncae: datadriven representations using a deep network-coherent dna methylation autoencoder identify robust disease and risk factor signatures. Briefings in Bioinformatics 24(5), 293 (2023)

[21] Prosz, A., Pipek, O., Börcsök, J., Palla, G., Szallasi, Z., Spisak, S., Csabai, I.: Biologically informed deep learning for explainable epigenetic clocks. Scientific Reports 14(1), 1306 (2024)

[22] Lin, A., Giosan, I., Aparicio, A., Guo, T., Melnikas, M., Balagué-Dobón, L., Carreras-Gallo, N., Hassouneh, S.A.-D., Seale, K., Kowalewski, A., et al.: Deepstrataage: an interpretable deep-learning clock that reveals stage-and sexdivergent dna methylation aging dynamics. npj Aging 12(1), 62 (2026)

[23] Tanaka, T., Biancotto, A., Moaddel, R., Moore, A.Z., Gonzalez-Freire, M., Aon, M.A., Candia, J., Zhang, P., Cheung, F., Fantoni, G., et al.: Plasma proteomic signature of age in healthy humans. Aging cell 17(5), 12799 (2018)

[24] Pereira, J.B., Janelidze, S., Smith, R., Mattsson-Carlgren, N., Palmqvist, S., Teunissen, C.E., Zetterberg, H., Stomrud, E., Ashton, N.J., Blennow, K., et al.: Plasma gfap is an early marker of amyloid-β but not tau pathology in alzheimer's disease. Brain 144(11), 3505–3516 (2021)

[25] Lewczuk, P., Ermann, N., Andreasson, U., Schultheis, C., Podhorna, J., Spitzer, P., Maler, J.M., Kornhuber, J., Blennow, K., Zetterberg, H.: Plasma neurofilament light as a potential biomarker of neurodegeneration in alzheimer's disease. Alzheimer's research & therapy 10(1), 71 (2018)

[26] Zhou, Y., Zhou, B., Pache, L., Chang, M., Khodabakhshi, A.H., Tanaseichuk, O., Benner, C., Chanda, S.K.: Metascape provides a biologist-oriented resource for the analysis of systems-level datasets. Nature communications 10(1), 1523 (2019)

[27] Tacutu, R., Craig, T., Budovsky, A., Wuttke, D., Lehmann, G., Taranukha, D., Costa, J., Fraifeld, V.E., De Magalhaes, J.P.: Human ageing genomic resources: integrated databases and tools for the biology and genetics of ageing. Nucleic acids research 41(D1), 1027–1033 (2012)

[28] Basisty, N., Kale, A., Jeon, O.H., Kuehnemann, C., Payne, T., Rao, C., Holtz, A., Shah, S., Sharma, V., Ferrucci, L., et al.: A proteomic atlas of senescenceassociated secretomes for aging biomarker development. PLoS biology 18(1), 3000599 (2020)

[29] Uhlén, M., Fagerberg, L., Hallström, B.M., Lindskog, C., Oksvold, P., Mardinoglu, A., Sivertsson, Å., Kampf, C., Sjöstedt, E., Asplund, A., et al.: Tissue-based map of the human proteome. Science 347(6220), 1260419 (2015)

[30] Szklarczyk, D., Gable, A.L., Nastou, K.C., Lyon, D., Kirsch, R., Pyysalo, S., Doncheva, N.T., Legeay, M., Fang, T., Bork, P., et al.: The string database in 2021: customizable protein-protein networks, and functional characterization of user-uploaded gene/measurement sets. Nucleic acids research 49(D1), 605–612 (2021)

[31] Yang, Z., Wang, K.K.: Glial fibrillary acidic protein: from intermediate filament assembly and gliosis to neurobiomarker. Trends in neurosciences 38(6), 364–374 (2015)

[32] Daniels, L.B., Maisel, A.S.: Natriuretic peptides. Journal of the American college of cardiology 50(25), 2357–2368 (2007)

[33] Barbera, M.C., Guarrera, L., Re Cecconi, A.D., Cassanmagnago, G.A., Vallerga, A., Lunardi, M., Checchi, F., Di Rito, L., Romeo, M., Mapelli, S.N., et al.: Increased ectodysplasin-a2-receptor eda2r is a ubiquitous hallmark of aging and mediates parainflammatory responses. Nature Communications 16(1), 1898 (2025)

[34] Cai, W., Zhang, K., Li, P., Zhu, L., Xu, J., Yang, B., Hu, X., Lu, Z., Chen, J.: Dysfunction of the neurovascular unit in ischemic stroke and neurodegenerative diseases: an aging effect. Ageing research reviews 34, 77–87 (2017)

[35] Yoo, E.-S., Yu, J., Sohn, J.-W.: Neuroendocrine control of appetite and metabolism. Experimental & molecular medicine 53(4), 505–516 (2021)

[36] Wang, D., Day, E.A., Townsend, L.K., Djordjevic, D., Jørgensen, S.B., Steinberg, G.R.: Gdf15: emerging biology and therapeutic applications for obesity and cardiometabolic disease. Nature Reviews Endocrinology 17(10), 592–607 (2021)

[37] Zois, N.E., Bartels, E.D., Hunter, I., Kousholt, B.S., Olsen, L.H., Goetze, J.P.: Natriuretic peptides in cardiometabolic regulation and disease. Nature Reviews Cardiology 11(7), 403–412 (2014)

[38] Li, X., Li, C., Zhang, W., Wang, Y., Qian, P., Huang, H.: Inflammation and aging: signaling pathways and intervention therapies. Signal transduction and targeted therapy 8(1), 239 (2023)

[39] Chung, H.Y., Kim, D.H., Lee, E.K., Chung, K.W., Chung, S., Lee, B., Seo, A.Y., Chung, J.H., Jung, Y.S., Im, E., et al.: Redefining chronic inflammation in aging and age-related diseases: proposal of the senoinflammation concept. Aging and disease 10(2), 367 (2019)

[40] López-Otín, C., Blasco, M.A., Partridge, L., Serrano, M., Kroemer, G.: Hallmarks of aging: An expanding universe. Cell 186(2), 243–278 (2023)

[41] McMahan, B., Moore, E., Ramage, D., Hampson, S., Arcas, B.A.: Communication-efficient learning of deep networks from decentralized data. In: Artificial Intelligence and Statistics, pp. 1273–1282 (2017). Pmlr

[42] Sudlow, C., Gallacher, J., Allen, N., Beral, V., Burton, P., Danesh, J., Downey, P., Elliott, P., Green, J., Landray, M., et al.: Uk biobank: an open access resource for identifying the causes of a wide range of complex diseases of middle and old age. PLoS medicine 12(3), 1001779 (2015)

[43] Edgar, R., Domrachev, M., Lash, A.E.: Gene expression omnibus: Ncbi gene expression and hybridization array data repository. Nucleic acids research 30(1), 207–210 (2002)

[44] Hastie, T.: The elements of statistical learning: data mining, inference, and prediction. springer (2009)

[45] Chen, T., Guestrin, C.: Xgboost: A scalable tree boosting system. In: Proceedings of the 22nd Acm Sigkdd International Conference on Knowledge Discovery and Data Mining, pp. 785–794 (2016)

[46] Rumelhart, D.E., Hinton, G.E., Williams, R.J.: Learning representations by backpropagating errors. nature 323(6088), 533–536 (1986)

[47] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

[48] Bibikova, M., Barnes, B., Tsan, C., Ho, V., Klotzle, B., Le, J.M., Delano, D., Zhang, L., Schroth, G.P., Gunderson, K.L., et al.: High density dna methylation array with single cpg site resolution. Genomics 98(4), 288–295 (2011)

[49] Piñero, J., Ramírez-Anguita, J.M., Saüch-Pitarch, J., Ronzano, F., Centeno, E., Sanz, F., Furlong, L.I.: The disgenet knowledge platform for disease genomics: 2019 update. Nucleic acids research 48(D1), 845–855 (2020)

[50] Flachsbart, F., Caliebe, A., Kleindorp, R., Blanché, H., Eller-Eberstein, H., Nikolaus, S., Schreiber, S., Nebel, A.: Association of foxo3a variation with human longevity confirmed in german centenarians. Proceedings of the National Academy of Sciences 106(8), 2700–2705 (2009)