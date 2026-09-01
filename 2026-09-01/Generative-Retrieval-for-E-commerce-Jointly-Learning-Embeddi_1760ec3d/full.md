# Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster

Songtao Fang Alibaba Group HangZhou, CHINA fangsongtao.fst@taobao.com

Jin Zhang   
Alibaba Group   
HangZhou, CHINA   
zj146372@taobao.com   
Zihao Xu   
Alibaba Group   
HangZhou, CHINA   
xiyu.xzh@taobao.com   
Shaowei Wei   
Alibaba Group   
HangZhou, CHINA   
leqian.wsw@taobao.com   
Zhuojun Wang   
Alibaba Group   
HangZhou, CHINA   
jerry.wangzj@taobao.com

## Abstract

With the development of large language models (LLMs), generative retrieval is becoming increasingly important in e-commerce scenarios. Current mainstream approaches typically use a two-stage training strategy: first train a product embedding model, and then learn a codebook that maps embeddings to product IDs. This cascaded approach sufers from two major issues: (1) error accumulation—if the embedding model in the first stage produces biased representations, the codebook in the second stage cannot correct these errors, degrading final retrieval performance; and (2) codebook learning relies solely on product embeddings and lacks modeling of query-to-product and product-to-product interactions. As a result, products belonging to the same cluster may be assigned inconsistent IDs by the codebook, further hurting retrieval accuracy. To address these problems, we propose a novel method that jointly trains the embedding model and the codebook, and incorporates same product cluster information as an additional supervision sig nal. Experimental results demonstrate that our method significantly improves e-commerce retrieval performance while simultaneously enhancing both embedding and codebook learning.

## CCS Concepts

• Information systems → Online shopping; Retrieval tasks and goals; Language models; Novelty in information retrieval.

## Keywords

Large Language Model, Information Retrieval, E-Commerce Search

## ACM Reference Format:

Songtao Fang, Zihao Xu, Shaowei Wei, Jin Zhang, and Zhuojun Wang. 2026. Generative Retrieval for E-commerce: Jointly Learning Embedding and Codebook with Same Product Cluster. In Proceedings of the ACM Web Conference 2026 (WWW ’26), April 13–17, 2026, Dubai, United Arab Emirates. ACM, New York, NY, USA, 4 pages. https://doi.org/10.1145/3774904.3792862

## 1 Introduction

Generative retrieval (GR) has emerged as a promising approach in information retrieval. Unlike sparse or dense methods—which rely on shallow semantic matching and often fail to capture implicit intent[9, 10] or handle colloquial queries (e.g., “gifts for a 10-yearold who loves science”). GR leverages large generative models to better interpret user intent and produce more relevant results.

GR can directly produce document identifiers (DocIDs) from user queries using the capabilities of pre-trained models. DPI[12] introduced transformer-based autoregressive models that preprocess documents into hierarchical identifiers using k-means clustering. Other approaches, such as MERGE[16], LETTER[15], Tiger[7], and UniSearch[1], learn DocIDs for retrieval based on RQ-VAE[7] or VQ-VAE[14]. However, mainstream approaches employ a two-stage pipeline: first learning product embeddings, then deriving a codebook to map these embeddings to DocIDs. This introduces two fundamental limitations. First, error accumulation: inaccuracies in the learned embeddings are propagated and amplified during codebook training. Second, insuficient interaction modeling: the codebook relies solely on static embeddings and thus fails to capture query–product and inter-product interactions, often resulting in inconsistent product IDs assignments even among semantically equivalent products within the same product cluster (i.e., variations of the same product from diferent suppliers or in diferent sizes).

To address these issues, we propose a novel joint training framework that simultaneously optimizes both the product embedding and the codebook. Our framework uses product cluster assignments as auxiliary supervision, which explicitly enhances the semantic consistency among products within the same cluster. By jointly training both components, our approach eliminates the error propagation inherent in conventional two-stage pipelines and optimizes them under a unified objective. The main contributions of this paper are as follows:

(1) We propose a joint training framework to optimize product embedding and the codebook, addressing error accumulation in traditional two-stage methods.

(2) By leveraging product cluster information as additional supervision, we enhance both the semantic consistency of IDs and the quality of embeddings for identical products.

(3) Extensive experiments on multiple e-commerce retrieval tasks show that our method outperforms existing approaches.

![](images/cfa609ab6e9f926e2caabcd5480e51beeddbca548a6a7fcd83ef1fc6d3e52536.jpg)  
Figure 1: Architecture of our model, consisting of Product Identifier Training and LLM Training.

## 2 Related Work

Sparse and Dense Retrieval. Sparse retrieval methods like BM25[8] use inverted indexes and TF-IDF–style term matching for eficient lookup, but they struggle with semantic variation and term mis match. Dense retrieval approaches such as DPR[3] and ColBERT[4] leverage neural embeddings and pretrained models like BERT[2] to capture semantic relationships and improve accuracy. However, both rely on fixed features or learned representations that can fail to capture complex query intent. Large language models, by contrast, excel at understanding complex queries—an advantage that neither sparse nor dense retrieval consistently provides.

Generative Retrieval. Generative retrieval has emerged as a promising paradigm in information retrieval. DSI[12] transforms documents into DocIDs and utilizes transformers for end-to-end retrieval. GenRet[11] uses an Encoder-Decoder structure to generate ID tokens step-by-step. MERGE[16] introduces a novel framework that employs multi-level relevance learning within an RQ-VAE to generate document identifiers. GRAM[6] introduces a novel approach to e-commerce retrieval, leveraging large language models to generate shared codes for both queries and products. Mainstream methods typically adopt a two-stage pipeline, first learning embeddings and then mapping them to product IDs, which often leads to error accumulation and lacks efective interaction modeling.

## 3 Model

In this section, we present our framework, including the Product Identifier Training and the LLM Training. The product retrieval task can be formalized as the process of retrieving a relevant product � for a user query � from a large catalog of products P. Each product $p \in { \mathcal { P } } \mathrm { i }$ is associated with rich multimodal attributes, such as title, description, and category, which can be represented as a token sequence $t = \left\{ t _ { 1 } , \ldots , t _ { | d | } \right\}$ where |�| is the total number of tokens in the product info. To enable eficient end-to-end retrieval, we first adopt a product tokenization strategy that maps each product � to a discrete token sequence $\boldsymbol { z } = \left\{ z _ { 1 } , \ldots , z _ { l } , \ldots , z _ { L } \right\}$ , where each token $z _ { l }$ is a �-way categorical variable $\langle z _ { l } \in \left\{ 1 , 2 , \ldots , K \right\}$ , and � denotes the fixed length of the product identifier. Crucially, each product � also belongs to a group of the same product �(�) �(�)—a group of functionally or visually identical items (e.g., diferent sellers ofering the same SKU)—which provides valuable semantic supervision. Then we train an LLM to learn the mapping from user queries to product IDs.

## 3.1 Product Identifier Training

The Product Identifier Training is used to jointly train embeddings and codebooks.

Query&Product Encoding.Given a query and the corresponding product ${ \boldsymbol { p } } ,$ we first extract their semantic features $h _ { \mathit { p r o d u c t } }$ and $h _ { q u e r y }$ using an embedding model and a deep neural network(DNN) encoder. Additionally, we randomly sample � products from the same product cluster �(�) to which the product belongs, extract their semantic features $h _ { 1 } , h _ { 2 } , \ldots , h _ { m }$ , and compute their mean to obtain $h _ { c l u s t e r }$ . Mathematically, this can be expressed as follows:

$$
h _ { \mathrm { c l u s t e r } } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } h _ { i }\tag{1}
$$

Then we apply the Mean Squared Error (MSE) loss to enforce consistency between the product embedding ℎ <sub>������</sub> and the cluster-level representation $h _ { c l u s t e r } { : }$

$$
\mathcal { L } _ { m s e } = M S E ( h _ { p r o d u c t } , h _ { c l u s t e r } )\tag{2}
$$

Semantic Embedding Residual Quantization.The latent semantic embedding $h _ { \mathit { p r o d u c t } }$ is quantized into a code sequence using a hierarchy of codebooks consisting of � levels , where � represents the length of the identifier. At each quantization level �, there is a codebook $C _ { l } = \{ e _ { k } ^ { l } \} _ { k = 1 } ^ { K }$ , where $e _ { k } ^ { l } \in \bar { \mathbb { R } ^ { d } }$ represents a learnable code embedding and � denotes the codebook size. At each code level, we find the most similar code embedding with the semantic residual and assigns the product with the corresponding code index. The residual quantization can be formulated as:

$$
\begin{array}{c} \begin{array} { r } { \left\{ { \boldsymbol z } _ { l } = \arg \operatorname* { m i n } \| { \bf r } _ { l - 1 } - { \bf e } _ { k } ^ { l } \| ^ { 2 } , \quad { \bf e } _ { k } ^ { l } \in C _ { l } , \right.} \\ { { \bf r } _ { l } = { \bf r } _ { l - 1 } - { \bf e } _ { z _ { l } } } \end{array}   \end{array}\tag{3}
$$

where, $z _ { l }$ is the code index chosen from the �-th codebook, $r _ { l - 1 }$ is the semantic residual from the last level. In the first step, we set $r _ { 0 } { = } h _ { \mathit { p r o d u c t } }$ . After the residual quantization, we obtain the quantized identifier $\boldsymbol { z } = \left[ z _ { 1 } , \ldots , z _ { l } , \ldots , z _ { L } \right]$ and the quantized representation $\begin{array} { r } { \hat { z } = \sum _ { l = 0 } ^ { L - 1 } e _ { z _ { l } } } \end{array}$ . The quantized representation is fed into a DNN de coder $D \ ( h _ { r e c o n } \ = \ D ( \hat { z } ) )$ to reconstruct the input $h _ { \mathit { p r o d u c t } }$ via a reconstruction loss:

$$
\mathcal { L } _ { r e c o n } = | | h _ { j p r o d u c t } - h _ { r e c o n } | | ^ { 2 }\tag{4}
$$

The $\scriptstyle { \mathcal { L } } _ { R Q - V A E }$ objective combines reconstruction loss $\mathcal { L } _ { r e c o n }$ with codebook commitment loss $\mathcal { L } _ { r q } \colon$

$$
\left\{ \begin{array} { l l } { \mathcal { L } _ { R Q - V A E } = \mathcal { L } _ { r e c o n } + \mathcal { L } _ { r q } , w h e r e , } \\ { \mathcal { L } _ { r q } = \sum _ { l = 0 } ^ { L } ( | | s g [ r _ { l } ] - e _ { z _ { l } } ^ { l } | | ^ { 2 } + \mu | | r _ { l } - s g [ e _ { z _ { l } } ^ { l } ] | | ^ { 2 } ) } \end{array} \right.\tag{5}
$$

where ��[·] is the stop-gradient operation, and � is the coeficient to balance the strength between the optimization of code embeddings and encoder.

Contrastive Learning. To achieve joint training of the codebook and embedding model, we utilize InfoNCE loss[5] on two key components: the representation $h _ { \mathit { p r o d u c t } }$ generated by the embedding model and the reconstructed embedding representations ℎ<sub>�����</sub>. Both components share the same InfoNCE learning objective:

$$
\mathcal { L } _ { \mathrm { I n f o N C E } } = - \log \frac { \exp \left( s ( h _ { q u e r y } , p ^ { + } ) / \tau \right) } { \sum _ { j = 1 } ^ { N } \exp \left( s ( h _ { q u e r y } , p ^ { i } ) / \tau \right) }\tag{6}
$$

where $\tau , h _ { q u e r y } ,$ and $\mathcal { P }$ denote the temperature parameter, query representation and product representation. In the two components, � represents $h _ { \mathit { p r o d u c t } }$ and $h _ { r e c o n } .$ , respectively. The positive $p ^ { + }$ is the relevant product to query, and other irrelevant product are negatives. These negatives can be either hard-negatives or in-batch negatives (products of other instances in the same batch). �(·) is the relevance score of query and product, measured by the cosine similarity between their respective representations.

Overall Loss. The training loss for our model is summarized as:

$$
\mathcal { L } _ { \mathrm { m o d e l } } = \mathcal { L } _ { \boldsymbol { R } \boldsymbol { Q } - \boldsymbol { V } \boldsymbol { A } \boldsymbol { E } } + \alpha \mathcal { L } _ { \mathrm { I n f o N C E } } + \beta \mathcal { L } _ { m s e }\tag{7}
$$

where � and $\beta$ are hyper-parameters to control the strength of contrastive learning and same product cluster constraints, respectively.

## 3.2 LLM Training

LLM Training is designed to learn the mapping from user queries to product IDs. Assuming the length of the product identifier is 4, and using the trained embedding and VAE model, each product is assigned an ID sequence such as: $[ < \tt a - 1 > , < \tt b - 0 > , < \tt c - 6 > , < \tt d - 2 > ]$ . Each element in this sequence, such as <a-1>, represents a distinct special token added to the vocabulary for training. Since the newly introduced special tokens are unfamiliar to LLMs, we adopt a two-stage progressive learning strategy to better achieve semantic alignment and remodeling. In the first stage, we train the model to map product information (such as titles, attributes, and product descriptions) to product IDs, enabling it to preliminarily grasp the semantic information of the new tokens. In the second stage, we further train the model to map user queries to product IDs. Through this staged learning approach, the model can more efectively master the semantic characteristics of the new tokens, thereby improving retrieval performance. Finally, we train the sequence-to-sequence transformer model with the following loss function:

$$
\mathcal { L } _ { s f t } = - \sum _ { t = 1 } ^ { T } l o g P ( y _ { t } | y _ { < t } , i n p u t )\tag{8}
$$

where the input represents product information in Stage 1, and represents user query in Stage 2. � denotes the length of product ID, $y _ { t }$ is the �-th token of the target product ID, $y _ { < t }$ represents the previously generated tokens.

## 4 Experiments

## 4.1 Datasets Description

We use 20 million product data from Alibaba’s internal e-commerce platform to train embeddings and codebooks. Similarly, we use these 20 million products to learn product-to-ID mappings for LLM stage1 training. Additionally, we construct a mapping of 40 million queries to product IDs for stage2 training. We also constructed validation and test sets, each containing 10,000 queries, evenly distributed between regular and implicit intent queries to ensure comprehensive model evaluation.

## 4.2 Baseline Models

To investigate the efectiveness of our model, we use three baseline categories: sparse, dense, and generative retrieval models. The detailed introductions are listed as follows:

• BM25[8]: It enhances relevance scoring by refining term weights based on the TF-IDF feature.

• DPR[3]: This approach enhances retrieval efectiveness by capturing deeper contextual relationships between queries and documents. To further support long text sequences, we utilize the GTE embedding [17] as encoder.

• DSI[12]: It represents documents using hierarchical K-means clustering results.

$\mathrm { T i g e r } _ { \mathrm { r q - v a e } } [ 7 ]$ : It quantizes semantic information into code sequence for LLM-based generative recommendation. Here, we focus solely on comparing the RQ-VAE component.

## 4.3 Evaluation Metrics

We evaluate our model using Recall@K(%), which measures the fraction of relevant items retrieved in the top-K results, reflecting the model’s ability to identify relevant candidates. We also evaluate the quality of the codebook using the Average Length of the Shared ID Prefixes (ALSP) among items within the same product cluster.

Table 1: Performance comparison of diferent models
<table><tr><td>Model</td><td>Recall@1</td><td>Recall@10</td><td>Recall@100</td><td>ALSP</td></tr><tr><td>BM25</td><td>1.23</td><td>2.66</td><td>9.80</td><td>一</td></tr><tr><td>DPR</td><td>3.57</td><td>7.09</td><td>24.24</td><td>1</td></tr><tr><td>DSI</td><td>3.79</td><td>7.99</td><td>23.91</td><td>2.89</td></tr><tr><td> $\mathrm { T i g e r } _ { \mathrm { r q - v a e } }$ </td><td>4.33</td><td>8.84</td><td>26.38</td><td>3.92</td></tr><tr><td>ours</td><td>4.49</td><td>9.90</td><td>30.71</td><td>4.42</td></tr><tr><td>ours-cluster</td><td>4.39</td><td>8.91</td><td>28.47</td><td>4.01</td></tr></table>

## 4.4 Experiment Settings

For product identifier training, we utilize the gte embedding model[17] as the shared query and product embedding model. In our RQ-VAE implementation, the codebook is configured with 5 layers (�=5), each containing 32 entries (�=32), and the embedding size for each entry is set to 768. Additionally, up to 5 products are sampled from the same product cluster corresponding to each product and we use the AdamW optimizer with a learning rate of 1e-4. We follow[7] to set � as 0.25, � is set as 0.5 � and � is set to as 0.5. For LLM training, we employ Qwen2.5-7B[13] to learn the mapping from product and query to product IDs with a learning rate of 5e-5.

## 4.5 Results and Analysis

The experimental results are shown in Table 1. ours-cluster denotes our model with the product-cluster constraint removed. Overall, the experimental results indicate that the proposed model outperforms all baselines on a large-scale real-world dataset. Specifically, compared to sparse (e.g., BM25) and dense retrieval methods (e.g., DPR), our model significantly outperforms them on the dataset. Sparse retrieval relies on keyword matching, while dense retrieval is constrained by the construction of positive and negative samples and cannot leverage general world knowledge. These limitations make both approaches struggle to handle complex or implicit-intent queries efectively.

Compared with generative retrieval methods, our model exhibits superior efectiveness and practicality. DSI and ${ \mathrm { T i g e r } } _ { { \mathrm { r q } } - { \mathrm { v a e } } } ;$ which rely on item embeddings for clustering and residual quantization, tend to amplify inherent biases in the embedding space. Furthermore, their non-end-to-end learning paradigm introduces information loss, degrading overall retrieval performance. As evidenced by the ASLP metric, items within the same product cluster exhibit significantly higher common prefixes than those generated by other generative methods. Crucially, when the cluster-based vector constraints are removed, both ASLP and other key metrics exhibit substantial degradation. This demonstrates that the product-cluster constraint provides a powerful, semantically rich supervision signal for learning consistent item ID representations. By aligning item IDs with meaningful semantic groupings, our approach reduces ambiguity in subsequent LLM training, enabling more coherent and hierarchical semantic structures in the ID space. By comparing the performance of ours-cluster with ${ \mathrm { T i g e r } } _ { \mathrm { r q - v a e } }$ , we can observe that end-to-end training brings performance improvements, as the interaction between queries and products during training helps mitigate the bias inherent in codebook learning that relies solely on embeddings.

## 5 Conclusion

In this paper, we propose a joint-training framework for generative retrieval that simultaneously optimizes product embedding and codebook, using same product cluster information to enforce ID consistency. Our method eliminates error accumulation in twostage pipelines and significantly outperforms existing approaches on real-world e-commerce data. Ablations confirm that both joint training and cluster supervision are crucial for enhancing retrieval accuracy and improving product ID consistency within the same cluster.

## References

[1] Jiahui Chen, Xiaoze Jiang, Zhibo Wang, Quanzhi Zhu, Junyao Zhao, Feng Hu, Kang Pan, Ao Xie, Maohua Pei, Zhiheng Qin, et al. 2025. UniSearch: Rethink ing Search System with a Unified Generative Architecture. arXiv preprint arXiv:2509.06887 (2025).

[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter ofthe association for computational linguistics: human language technologies, volume 1 (long and short papers). 4171–4186.

[3] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick SH Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense Passage Retrieval fo Open-Domain Question Answering.. In EMNLP (1). 6769–6781.

[4] Omar Khattab and Matei Zaharia. 2020. Colbert: Eficient and efective passage search via contextualized late interaction over bert. In Proceedings ofthe 43rd International ACM SIGIR conference on research and development in Information Retrieval. 39–48.

[5] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018).

[6] Ming Pang, Chunyuan Yuan, Xiaoyu He, Zheng Fang, Donghao Xie, Fanyi Qu, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo, et al. 2025. Generative Retrieval and Alignment Model: A New Paradigm for E-commerce Retrieval. In Companion Proceedings ofthe ACM on Web Conference 2025. 413–421.

[7] Shashank Rajput, Nikhil Mehta, Anima Singh, Raghunandan Hulikal Keshavan, Trung Vu, Lukasz Heldt, Lichan Hong, Yi Tay, Vinh Tran, Jonah Samost, et al. 2023. Recommender systems with generative retrieval. Advances in Neural Information Processing Systems 36 (2023), 10299–10315.

[8] Stephen Robertson, Hugo Zaragoza, et al. 2009. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends® in Information Retrieval 3, 4 (2009), 333–389.

[9] Rulin Shao, Rui Qiao, Varsha Kishore, Niklas Muennighof, Xi Victoria Lin, Daniela Rus, Bryan Kian Hsiang Low, Sewon Min, Wen-tau Yih, Pang Wei Koh, et al. 2025. ReasonIR: Training Retrievers for Reasoning Tasks. arXiv preprint arXiv:2504.20595 (2025).

[10] Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighof, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S Siegel, Michael Tang, et al. 2024. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. arXiv preprint arXiv:2407.12883 (2024).

[11] Weiwei Sun, Lingyong Yan, Zheng Chen, Shuaiqiang Wang, Haichao Zhu, Pengjie Ren, Zhumin Chen, Dawei Yin, Maarten de Rijke, and Zhaochun Ren. 2023. Learning to Tokenize for Generative Retrieval. CoRR abs/2304.04171 (2023).

[12] Yi Tay, Vinh Tran, Mostafa Dehghani, Jianmo Ni, Dara Bahri, Harsh Mehta, Zhen Qin, Kai Hui, Zhe Zhao, Jai Gupta, et al. 2022. Transformer memory as a diferentiable search index. Advances in Neural Information Processing Systems 35 (2022), 21831–21843.

[13] Qwen Team et al. 2024. Qwen2 technical report. arXiv preprint arXiv:2407.10671 2, 3 (2024).

[14] Aaron Van Den Oord, Oriol Vinyals, et al. 2017. Neural discrete representation learning. Advances in neural information processing systems 30 (2017).

[15] Wenjie Wang, Honghui Bao, Xinyu Lin, Jizhi Zhang, Yongqi Li, Fuli Feng, See-Kiong Ng, and Tat-Seng Chua. 2024. Learnable item tokenization for generative recommendation. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management. 2400–2409.

[16] Fuwei Zhang, Xiaoyu Liu, Xinyu Jia, Yingfei Zhang, Shuai Zhang, Xiang Li, Fuzhen Zhuang, Wei Lin, and Zhao Zhang. 2025. Multi-level Relevance Document Identifier Learning for Generative Retrieval. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 10066–10080.

[17] Xin Zhang and Y Zhang. 2024. mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval,(2024). URL https://arxiv. org/abs/2407.19669 (2024).