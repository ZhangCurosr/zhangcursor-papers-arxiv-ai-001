# LingLan: An Advancing Traditional Chinese Medicine Diagnosis LLM with Multimodal Data

Zheng Chen<sup>1</sup>, Zhicheng Du<sup>1</sup>, Haoxuan Li<sup>1</sup>, Yingshan Liang<sup>1</sup>, Peiwu Qin<sup>2†</sup>

<sup>1</sup>Tsinghua Shenzhen International Graduate School, Tsinghua University, Shenzhen, China

<sup>2</sup>Guangdong Provincial Laboratory of Traditional Chinese Medicine Hengqin, Guangdong, China

Abstract—Though artificial intelligence (AI) increasingly transforms modern medicine, its integration into Traditional Chinese Medicine (TCM) has been relatively slow, primarily due to TCM’s reliance on holistic, subjective diagnostic methods—namely Inspection, Auscultation and Olfaction, Inquiry, and Palpation(I-AOI-P)—which are difficult to align with quantitative, standardized medical systems. In this work, we introduce a Unification Framework for Multimodal Data (UFMD), which automatically processes tongue and pulse images into structured, clinically standard descriptions, integrating multi-source diagnostic information into a unified digital record of I-AOI-P process. Building on this structured data, we create LingLan-14B, a TCMspecific large language model fine-tuned via supervised learning to emulate the diagnostic logic and workflow of I-AOI-P process. Experimental results show that our method significantly enhances diagnostic accuracy, achieving a relative improvement of 103.5% over the baseline (62.72% vs. 30.82%) and reaching an F1-score of up to 82%.

Index Terms—Multimodal Large Language Model, Traditional Chinese Medicine Diagnosis, Supervised Fine-Tuning

## I. INTRODUCTION

In recent years, the rapid advancement of artificial intelligence (AI) has profoundly reshaped the diagnostic and treatment models, as well as the service ecosystem, of modern medicine. From precision surgical robots to large language model (LLM) based auxiliary diagnostic systems, AI has progressively integrated into the entire chain of healthcare services [1–5]. However, Traditional Chinese Medicine (TCM), with its emphasis on holism, syndrome differentiation and treatment, and its reliance on the subjective experiential diagnostics of Inspection, Auscultation and Olfaction, Inquiry, and Palpation (I-AOI-P, shown in Fig. 1) struggles to fully synchronize with the modern medical system characterized by quantitative analysis and standardized pathways, resulting in a relatively lagging pace of development.

Concurrently, against the backdrop of an accelerating aging population and the increasing burden of chronic diseases, societal demand for TCM’s ”preventive treatment of disease” philosophy and personalized health management is steadily growing, which highlights the urgency of innovating TCM diagnostic and treatment models [6]. However, current AIassisted TCM diagnosis faces challenges of insufficient accuracy and stability, making it difficult to support reliable clinical application. On one hand, TCM syndrome differentiation depends on the comprehensive judgment of multi-modal information such as tongue and pulse images, yet existing algorithms focus more on inquiry information, encountering bottlenecks in fusing multi-source heterogeneous data and modeling nonlinear relationships. On the other hand, publicly available TCM datasets are limited in scale and uneven in quality, failing to meet the demand for high-granularity, multidimensional data required for individualized diagnosis and treatment. Therefore, there is a pressing need to bridge this gap by combining algorithmic innovations with the construction of high-quality data resources.

![](images/b1f6f2752d6d58755186a36d63890db42e2223bebffc57cf3812fe25d1e93de1.jpg)  
Fig. 1: (a) Diagnosis with only inquiry will result in lowreliability results. (b) Diagnosis with I-AOI-P Diagnosis will yield more accurate results.

To address the aforementioned challenges, we propose an innovative multi-modal large language model (MLLM) based framework for intelligent diagnosis in TCM. The core innovation of this framework lies in establishing an automated, structured pipeline that processes multi-source information into a unified knowledge representation. Specifically, the framework acquires the patient’s tongue diagnosis images and pulse diagnosis waveform images, which are directly input into the system. It then outputs standardized textual descriptions of tongue manifestations (e.g., pale-red tongue proper, thin white coating) and pulse manifestations (e.g., wiry pulse, thready pulse) that conform to TCM clinical terminology, thereby objectifying and digitizing the information obtained from inspection and palpation. Ultimately, data from the I-AOI-P method are integrated into a unified, structured case representation, forming a complete digital record of the I-AOI-P process.

Based on the digital records, we perform post-training on a pre-trained LLM, which involves domain-adaptive fine-tuning aimed at aligning the model’s internal reasoning mechanisms with the specialized knowledge and diagnostic logic of TCM. By learning from the unified structured representations, the model is enhanced to perform better in TCM analysis. The main highlights of our contributions are as follows:

• Unification Framework for Multimodal Data (UFMD): We present a novel framework for unifying TCM multimodal clinical record data, which leverages a MLLM and incorporates human-in-the-loop verification to integrate TCM medical record information from diverse sources into structured data.

• LingLan-14B: TCM Diagnosis LLM: Based on the unique I-AOI-P diagnostic process of TCM, we reframe the supervised fine-tuning (SFT) of the model from the perspective of its data framework, enabling the LLM to more effectively learn the underlying logic and workflow of I-AOI-P process. We will release the weight parameters of LingLan-14B after acceptance.

• Experiment Study: Performance and ablation experiments demonstrate that SFT based on I-AOI-P diagnosis greatly enhances the model’s ability to diagnose TCM, achieving a high improvement of 103.5% over the baseline and achieving an F1-score as high as 82%.

## II. RELATED WORK

1) Multimodal Data Unification: Current approaches to TCM data acquisition frequently integrate database compilation with manual curation. For instance, Liu et al. [7] aggregate prescription, herbal, ingredient, and target data from 14 authoritative TCM databases. They employ natural language processing (NLP), specifically the BioBERT model, to extract and validate entity relationships from over 30 million documents, supplemented by manual curation of clinical records to ensure comprehensiveness. Similarly, Li et al. [8] integrate multiple authoritative databases, manually curate clinical and ancient text data, and apply biomedical NLP techniques to extract and refine relationships from extensive literature. However, the inefficiency in cleansing large-scale real-world data and the inherently time-intensive nature of manual curation processes still exist.

2) TCM Diagnosis Improvement: Many studies urge to improve TCM diagnosis with AI [9–11]. He et al. [12] propose the OpenTCM system, which improves the accuracy and reliability of TCM diagnosis by combining LLM, knowledge graphs (KG), and graph-based retrieval enhancement generation (GraphRAG). Yang et al. [13] construct a large-scale TCM corpus and develop the TCM-GPT-7B model, which significantly improve its accuracy in TCM examinations and diagnostic tasks. However, these approaches have overlooked the comprehensive consideration of I-AOI-P diagnosis, which results in limited generalization capability of the models in practical applications.

## III. METHODOLOGY

## A. Unification Framework for Multimodal Data

With the deep integration of information technology into the medical field, the format of TCM clinical records is undergoing a fundamental shift from traditional paper-based documentation to comprehensive electronic systems. However, a prominent challenge in this transition is the significant heterogeneity of TCM medical data. This heterogeneity is primarily characterized by multi-source and multi-modal data types, including image-based tongue diagnosis data, pulse diagnosis image data, and unstructured or semi-structured information from paper-based or electronic medical records. The effective aggregation, integration, and in-depth utilization of these diverse data forms remain a considerable challenge, especially in the digitization of vast historical paper records and images, which often still relies on inefficient manual extraction methods, thereby hindering the full potential of data-driven clinical research and application in TCM.

To address this problem, we propose a unified data fusion framework. The core of this framework involves the introduction of an advanced DeepSeek-OCR model [14]. The contextual optical compression technology employed by this model enables the intelligent parsing of medical record images, image-based tongue diagnosis data, pulse diagnosis image data containing complex layouts and handwritten text, directly outputting high-quality structured data. Furthermore, we combine manual proofreading with expert verification in the framework to ensure the reliability and authenticity of the data. Fig. 2 illustrates the structure of UFMD. This approach achieves an ”end-to-end” automated conversion of photos and paperbased medical record data, significantly enhancing the overall efficiency of extraction and integration from multimodal heterogeneous data to standardized, computable information, thereby laying a solid foundation for subsequent construction of high-quality TCM clinical datasets.

To mitigate data extraction anomalies and erroneous generation caused by hallucination in MLLM, we propose a novel semantic relevance verification mechanism. For an electronic medical record datum $D _ { e x }$ extracted by MLLM and its corresponding raw text datum $D _ { r a w } ,$ we ensure the semantic relevance of the results by computing the cosine similarity between their sentence embeddings generated by a pre-trained Sentence-BERT (SBERT) model.

The Sentence-BERT model maps an input sentence to a fixed-dimensional dense vector space. Formally, the embedding of a sentence S is given by:

$$
\mathbf { e } _ { S } = \mathbf { S } \mathbf { B } \mathbf { E } \mathbf { R } \mathbf { T } ( S )\tag{1}
$$

where SBERT(·) denotes the forward pass of the Sentence-BERT model, and $\mathbf { e } _ { S }$ is the resulting sentence embedding vector.

The semantic similarity between $D _ { e x }$ and $D _ { r a w }$ is then calculated as the cosine similarity between their respective

![](images/ff040ea795edfda1cf73cda8d2fdff7bd07226e70e3064780065a4c7bac7cc16.jpg)  
Fig. 2: An overview of UFMD: (A) Extract features from multimodal data via MLLM templates. (B) Refine data through MLLM review and expert validation. (C) Store verified structured data in persistent database.

embeddings:

$$
\begin{array} { r } { \mathrm { s i m } ( D _ { e x } , D _ { r a w } ) ^ { \mathrm { S B E R T } } = \mathrm { c o s } ( \mathbf { e } _ { D _ { e x } } , \mathbf { e } _ { D _ { r a w } } ) } \\ { = \frac { \mathbf { e } _ { D _ { e x } } \cdot \mathbf { e } _ { D _ { r a w } } } { \left\| \mathbf { e } _ { D _ { e x } } \right\| \left\| \mathbf { e } _ { D _ { r a w } } \right\| } } \end{array}\tag{2}
$$

The proposed verification mechanism operates as follows:

1) Text Preprocessing: The extracted text $D _ { e x }$ and the raw source text $D _ { r a w }$ undergo standard text preprocessing, including tokenization and cleaning (e.g., removing extra whitespaces, handling special characters).

2) Sentence Embedding Generation: Each preprocessed text is fed into the Sentence-BERT model to obtain its fixed-dimensional semantic vector representation, ${ \bf e } _ { D _ { e x } }$ and $\mathbf { e } _ { D _ { r a w } }$

3) Similarity Calculation: The cosine similarity between the two embedding vectors is computed, yielding a score between -1 and 1, where scores closer to 1 indicate higher semantic relevance.

4) Relevance Thresholding: A predefined threshold θ is applied to the similarity score. If $\sin ( D _ { e x } , \bar { D _ { r a w } } ) ^ { \mathrm { S B E R T } } \geq \theta .$ , the extracted data $D _ { e x }$ is considered semantically consistent with the source $D _ { r a w } ;$ otherwise, it is flagged for potential hallucination or error.

This mechanism leverages the contextual understanding capabilities of Sentence-BERT, providing a robust assessment of relevance, especially for complex TCM narratives where paraphrasing and synonymy are common.

Furthermore, to ensure the accuracy and reliability of the core data content, we implemented a rigorous manual proofreading process. To further improve data processing efficiency, we introduced MLLM to constrain the uncertainty of generative data through information comparison, thereby enhancing data accuracy. This human-centric approach, augmented by AI assistance, involved domain experts systematically verifying the extracted and structured information against the original records. Through this meticulous manual proofreading, we ensured data accuracy while successfully completing the data transformation and structuring process for over 120,000 TCM clinical records.

## B. LingLan: SFT Model for TCM Diagnosis

To enhance the diagnostic capability of a general-purpose LLM in TCM, we employ a Parameter-Efficient Fine-Tuning (PEFT) approach based on Low-Rank Adaptation (LoRA) [15] in Qwen3-14B [16], which has high performance in Chinese tasks. This strategy allows us to effectively adapt the pretrained model to the specialized knowledge and language patterns inherent in TCM clinical records, while avoiding the prohibitive computational cost associated with full parameter fine-tuning.

Our methodology emphasizes the systematic integration of multimodal diagnostic data, particularly focusing on tongue diagnosis description and palpation diagnosis description extracted from UFMD. Specifically, the training dataset was meticulously curated and partitioned into training and heldout test sets. Each clinical record was reformatted using a structured template that explicitly incorporates quantitative feature information from tongue images and pulse waveform images alongside inquiry information. This structured representation forces the model to establish clinically meaningful relationships between the objective findings from inspection and palpation and the subjective information from Inquiry, thereby directly aligning the learning objective with the holistic reasoning process of TCM diagnosis. By learning from this integrated representative descriptions, the model develops the capability to correlate specific tongue characteristics (e.g., tongue proper color, coating texture) and pulse patterns (e.g., wiry, thready) with corresponding TCM syndromes. This approach significantly improves the model’s proficiency in analyzing patient symptoms and conducting syndrome differentiation based on the complete I-AOI-P diagnosis, moving beyond a reliance on textual descriptions of symptoms alone.

In our experimental setup, the fine-tuning process was efficiently configured using the PEFT library, with the key hyperparameters and hardware environment detailed in Table I for a direct comparison against the BianCang-14B-Instruct model [17]. This comparative analysis highlights the superior efficiency of our PEFT-based approach, which significantly reduces training time and computational resource requirements while maintaining competitive performance metrics. The tabulated data clearly demonstrates the advantages of our method in terms of parameter efficiency and accelerated convergence.

## IV. EXPERIMENT

## A. Experimental settings

1) Dataset: We use a 20% subset of data from the TCM medical record dataset as a reserved test set for our experiments. We employ stratified sampling to maintain consistency in class distribution, stratifying by diagnostic label to ensure that the proportion of each diagnosis is approximately the same in the test and training sets. We also handle rare classes: if a class has only one sample, it is excluded from the test set; if there are two samples, at least one is included. This prevents evaluation bias caused by a class appearing only in either the training or test set.

TABLE I: Configurations and Training Environment Comparison
<table><tr><td>Parameter / Environment</td><td>LingLan-14B (Ours)</td><td>BianCang-14B [17]</td></tr><tr><td>Base Model LoRA Rank (r) LoRA Alpha (α) LoRA Dropout Target Modules Learning Rate Batch Size Training Epochs</td><td>Qwen3-14B 8 32 0.05 all-linear 1 × 10−4 1 1</td><td>Qwen2.5-14B 2</td></tr><tr><td>Hardware (GPUs) Training Time</td><td>NVIDIA RTX 5090 32GB 12 hours</td><td>8 × NVIDIA A100 40GB 605 hours</td></tr></table>

To ensure the authority of the diagnostic classification, the diagnoses in this study are fully classified in strict accordance withthe Classification and Codes of Diseases and Patterns of Traditional Chinese Medicine(GB/T 15657-2021) approved by the China National Standardization Administration. The classification results and category statistics are shown in Fig. 3 and Table II.

![](images/49caa8530d10d0b164d59ab69b5b03066701fe6cc69a162f48e23a9901807710.jpg)  
Fig. 3: Distribution of Test Data

2) Baselines: We compare our fine-tuned model with the original model Qwen3-14B and open-source models with higher parameter levels. Among them, Qwen3-32B [16], DeepSeek-R1-Distill-32B and DeepSeek-R1(671B) [18], as general-purpose large-scale models trained on Chinese corpora, have better understanding of Chinese content and are therefore suitable choices for comparison. We also include a model specifically designed for TCM diagnosis, BianCang-14B-Instruct [17], in the experiment to compare LingLan’s performance with the professional TCM-LLM model.

3) Metrics: In order to compare the performance, we choose to use basic metrics such as accuracy, recall, and F1- score to evaluate the model’s fundamental TCM prediction ability. Furthermore, during our experiments, we found that general-purpose models, lacking quantifier constraints, tend to provide broader answers, leading to data inaccuracies. Therefore, based on the specific characteristics of TCM diagnosis, we designed the following two new metrics:

• Unconstrained Diagnosis Indicator(UDI): This metric measures the predictive ability of LLM for specific patient examination results and individual needs in the absence of any constraints, aiming to test the predictive power of LLM in generalized scenarios.

TABLE II: Classicifation Categories of Test Set
<table><tr><td>Diagnosis type</td><td>Abbreviation</td><td>Count</td></tr><tr><td>Internal medicine</td><td>BN</td><td>11708</td></tr><tr><td>Surgery</td><td>BW</td><td>699</td></tr><tr><td>Gynecology</td><td>BF</td><td>6561</td></tr><tr><td>Pediatrics</td><td>BE</td><td>154</td></tr><tr><td>Ophthalmology</td><td>BY</td><td>84</td></tr><tr><td>Otorhinolaryngology</td><td>BR</td><td>2208</td></tr><tr><td>Orthopedics</td><td>BG</td><td>2958</td></tr></table>

$$
\mathrm { U D I } = \frac { U _ { T P } + U _ { T N } } { U _ { T P } + U _ { T N } + U _ { F P } + U _ { T N } }\tag{3}
$$

• Post Constrained Diagnosis Indicator(PCDI): To test the predictive ability of LLM for more specific symptoms, we added instruction constraints to the LLM during the testing process and observed its diagnostic predictive ability after providing these constraints.

$$
\mathrm { P C D I } = { \frac { 2 P _ { T P } } { P _ { T P } + P _ { T N } + P _ { F P } + P _ { T N } } }\tag{4}
$$

## B. Results

1) Overall Performance: Table III compares the performance of the fine-tuned model and the baseline models on the TCM medical record test set. The results clearly demonstrate that LingLan-14B model achieves state-of-the-art diagnosis performance, significantly outperforming all baseline models across all evaluation metrics, including Accuracy, Recall, F1- score, UDI, and PCDI. This substantial improvement, particularly the more than doubling in Accuracy and F1-score compared to some larger models, underscores the effectiveness of our fine-tuning approach for the specialized domain of TCM medical records. The superior performance establishes a strong foundation for the subsequent discussion on the specific contributions and implications of our method.

To quantitatively evaluate the classification efficacy of our fine-tuned model, we present the confusion matrix heatmaps of our model and BianCang-14B-Instruct in Fig. 4. The matrices illustrate the alignment between the true labels (vertical axis) and the model’s predicted labels (horizontal axis) across all categories (BM represents the misclassified label). A predominant concentration of high values along the main diagonal is immediately evident, signifying strong predictive accuracy. Statistics show that LingLan’s recall accuracy reaches 82.03%, compared to 69.51% of BianCang-14B-Instruct. For more detail, our model correctly classified 2,850 instances for category BN, 5,406 for BF, and 1,703 for BG, as indicated by the intensely colored corresponding diagonal cells. While minor misclassifications are present in the off-diagonal elements—such as confusions between BN and BW (703 cases), BN and BY (487 cases), and BF and BG (1,103 cases)—their relatively low magnitudes compared to the diagonal values underscore the model’s overall robustness. The clear diagonal dominance visualized by the color gradient confirms that our fine-tuned model achieves high classification performance, effectively learning discriminative features for accurate syndrome differentiation in TCM diagnostics.

TABLE III: Experiment Result of Diagnosis Performance
<table><tr><td>Model</td><td>Acc. Recall</td><td>F1-score</td><td>UDI</td><td>PCDI</td></tr><tr><td>Qwen3-14B [16]</td><td>0.3082 0.4109</td><td>0.4112</td><td>0.3106</td><td>0.2875</td></tr><tr><td>Qwen3-32B [16]</td><td>0.3324 0.4293</td><td>0.4098</td><td>0.3647</td><td>0.3312</td></tr><tr><td>DeepSeek-R1-Distill-32B [18]</td><td>0.2700 0.3774</td><td>0.3591</td><td>0.2824</td><td>0.2776</td></tr><tr><td>DeepSeek-R1(671B) [18]</td><td>0.3097 0.4356</td><td>0.4425</td><td>0.3627</td><td>0.3030</td></tr><tr><td>BianCang-14B-Instruct [17]</td><td>0.3266 0.6951</td><td>0.6949</td><td>0.5015</td><td>0.4394</td></tr><tr><td>LingLan-14B (Ours)</td><td>0.6272 0.8203</td><td>0.8202</td><td>0.6325</td><td>0.6103</td></tr></table>

![](images/e1f16e92cc6927ca9308716fd8d3b914510c10208c76167f9ddb293880a470c7.jpg)  
(a)

![](images/eef863401140029656896e9a8b1a9bfb75d2f2a3543d8fbcf2b2d2c983158a63.jpg)  
(b)  
Fig. 4: Confusion matrix of (a) LingLan-14B (b) BianCang-14B-Instruct

2) Ablation Study: We conduct an ablation study to validate the effectiveness of our novel training set structure in predicting TCM diagnosis. In addition, to better demonstrate the assistance of structured data in TCM diagnosis, we calculate the accuracy and recall of the diagnosis and syndrome differentiation results separately. We fine-tune the same baseline model Qwen3-14B with the same parameters using the raw data without any data structure processing, and test the model using the same test set. Table IV shows that the results of finetuning with structured data are significantly better than those without.

## V. CONCLUSION

In this paper, we propose UFMD, an innovative MLLMbased framework for intelligent diagnosis in TCM, addressing the challenges of integrating holistic multimodal I-AOI-P diagnostic information with quantitative medical systems. We also present LingLan-14B, a TCM-specific LLM fine-tuned via supervised learning to emulate the I-AOI-P diagnostic workflow, which reaching SOTA diagnostic performance. Experimental results demonstrate that our method significantly enhances diagnostic accuracy, validating the effectiveness of our structured data integration and fine-tuning strategy.

## VI. FUTURE WORK

We must state that due to the involvement of a substantial amount of patient personal privacy data, the dataset utilized in this study cannot be made publicly available at this time. In the future, we endeavor to release an anonymized version of the data after undergoing a formal privacy security assessment and obtaining appropriate patient authorizations. Furthermore, owing to cybersecurity constraints, we were unable to empirically evaluate the performance of non-open-source models on TCM diagnosis task. Our future work will explore the feasibility of conducting security-compliant testing for such models without compromising patient data privacy. Additionally, while our fine-tuning methodology yielded significant improvements on smaller-scale models, we plan to extend its application to larger models in subsequent research to further validate its effectiveness and scalability.

TABLE IV: Ablation Experiment Result
<table><tr><td>Model</td><td>Diag  $\underline { { A c c } }$ </td><td>DiagR</td><td> $\overline { { \mathrm { S y n } \_ { \mathrm { D i f f } } _ { A c c } } }$ </td><td>Syn_DiffR</td></tr><tr><td>SFTwithout</td><td>0.3790</td><td>0.6542</td><td>0.2460</td><td>0.4626</td></tr><tr><td>SFTwith</td><td>0.6334</td><td>0.8277</td><td>0.6210</td><td>0.6020</td></tr></table>

## REFERENCES

[1] Lei Liu, Xiaoyan Yang, Junchi Lei, Xiaoyang Liu, Yue Shen, Zhiqiang Zhang, Peng Wei, Jinjie Gu, Zhixuan Chu, Zhan Qin, and Kui Ren, “A survey on medical large language models: Technology, application, trustworthiness, and future directions,” CoRR, vol. abs/2406.03712, 2024.

[2] Haochun Wang, Sendong Zhao, Zewen Qiang, Nuwa Xi, Bing Qin, and Ting Liu, “Beyond direct diagnosis: Llmbased multi-specialist agent consultation for automatic diagnosis,” CoRR, vol. abs/2401.16107, 2024.

[3] Huimin Wang, Wai-Chung Kwan, Kam-Fai Wong, and Yefeng Zheng, “Coad: Automatic diagnosis through symptom and disease collaborative generation,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, Anna Rogers, Jordan L. Boyd-Graber, and Naoaki Okazaki, Eds. 2023, pp. 6348–6361, Association for Computational Linguistics.

[4] Zhijie Bao, Wei Chen, Shengze Xiao, Kuang Ren, Jiaao Wu, Cheng Zhong, Jiajie Peng, Xuanjing Huang, and Zhongyu Wei, “Disc-medllm: Bridging general large language models and real-world medical consultation,” CoRR, vol. abs/2308.14346, 2023.

[5] Zhongyu Wei, Qianlong Liu, Baolin Peng, Huaixiao Tou, Ting Chen, Xuanjing Huang, Kam-Fai Wong, and Xiangying Dai, “Task-oriented dialogue system for automatic diagnosis,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 2: Short Papers, Iryna Gurevych and Yusuke Miyao, Eds. 2018, pp. 201–207, Association for Computational Linguistics.

[6] Qifei Zou, Yitong Chen, Huanxin Qin, Rui Tang, Taojian Han, Ziyi Guo, Juanjuan Zhao, and Delin Xu, “The role and mechanism of tcm in the prevention and treatment of infectious diseases,” Frontiers in Microbiology, vol. 14, pp. 1286364, 2023.

[7] Zhi Liu, Jiaxi Yang, Kui Chen, Tao Yang, Xiaochen Li, Bingjie Lu, Dianzheng Fu, Zeyu Zheng, and Changyong Luo, “TCM-KDIF: an information interaction framework driven by knowledge-data and its clinical application in traditional chinese medicine,” IEEE Internet Things J., vol. 11, no. 11, pp. 20002–20014, 2024.

[8] Xu Li, Jing Ren, Wen Zhang, Zhiming Zhang, Jinchao Yu, Jiawei Wu, He Sun, Shuiping Zhou, Kaijing Yan, Xijun Yan, et al., “Ltm-tcm: a comprehensive database

for the linking of traditional chinese medicine with modern medicine at molecular and phenotypic levels,” Pharmacological Research, vol. 178, pp. 106185, 2022.

[9] Jiawen Liu, Feixiang Ge, Fufeng Li, Xinlei Li, Linqiang Hu, Yan Rong, and Rongrong Zhu, “A hybrid cnn-transformer network for fine-grained tongue multiattribute classification with tongue group attribute mask training in traditional chinese medicine diagnosis,” in Advanced Intelligent Computing Technology and Applications - 21st International Conference, ICIC 2025, Ningbo, China, July 26-29, 2025, Proceedings, Part XI, De-Shuang Huang, Bo Li, Haiming Chen, and Chuanlei Zhang, Eds. 2025, vol. 15852 of Lecture Notes in Computer Science, pp. 467–479, Springer.

[10] Yang Jin and Zhiwan Wang, “Analysis of artificial intelligence in traditional chinese medicine diagnosis,” in Proceedings of the 2023 4th International Symposium on Artificial Intelligence for Medicine Science, ISAIMS 2023, Chengdu, China, October 20-22, 2023. 2023, pp. 947–952, ACM.

[11] Chunyang Ruan, Yingpei Wu, Yun Yang, and Guangsheng Luo, “Semantic-aware graph convolutional networks for clinical auxiliary diagnosis and treatment of traditional chinese medicine,” IEEE Access, vol. 9, pp. 8797–8807, 2021.

[12] Jinglin He, Yunqi Guo, Lai Kwan Lam, Waikei Leung, Lixing He, Yuanan Jiang, Chi Chiu Wang, Guoliang Xing, and Hongkai Chen, “Opentcm: A graphragempowered llm-based system for traditional chinese medicine knowledge retrieval and diagnosis,” CoRR, vol. abs/2504.20118, 2025.

[13] Guoxing Yang, Jianyu Shi, Zan Wang, Xiaohong Liu, and Guangyu Wang, “TCM-GPT: efficient pre-training of large language models for domain adaptation in traditional chinese medicine,” CoRR, vol. abs/2311.01786, 2023.

[14] Haoran Wei, Yaofeng Sun, and Yukun Li, “Deepseekocr: Contexts optical compression,” CoRR, vol. abs/2510.18234, 2025.

[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, and Weizhu Chen, “Lora: Low-rank adaptation of large language models,” CoRR, vol. abs/2106.09685, 2021.

[16] Haiquan Zhao, Chenhan Yuan, Fei Huang, Xiaomeng Hu, Yichang Zhang, An Yang, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin, Baosong Yang, Chen Cheng, Jialong Tang, Jiandong Jiang, and Yuxin Zhou, “Qwen3guard technical report,” CoRR, vol. abs/2510.14276, 2025.

[17] Sibo Wei, Xueping Peng, Yi-fei Wang, Jiasheng Si, Weiyu Zhang, Wenpeng Lu, Xiaoming Wu, and Yinglong Wang, “Biancang: A traditional chinese medicine large language model,” CoRR, vol. abs/2411.11027, 2024.

[18] DeepSeek-AI, “Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning,” CoRR, vol. abs/2501.12948, 2025.