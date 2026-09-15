# A Language-Guided Multimodal Foundation Model for Zero-Shot and Multi-Task Brain Signal Analysis

Mingzhi Chen<sup>1</sup> | Yiyu Gui<sup>1</sup> | Guibo Luo<sup>1</sup> | Yuchao Yang<sup>1,2,3</sup>

<sup>1</sup>New Cornerstone Science Laboratory, Guangdong Provincial Key Laboratory of In-Memory Computing Chips, School of Electronic and Computer Engineering, Shenzhen Graduate School, Peking University, Shenzhen, China

<sup>2</sup>Center for Brain Inspired Intelligence, Chinese Institute for Brain Research (CIBR), Beijing, China

<sup>3</sup>New Cornerstone Science Laboratory, Beijing Advanced Innovation Center for Integrated Circuits, School of Integrated Circuits, Peking University, Beijing, China

Correspondence: Guibo Luo (luogb@pku.edu.cn) | Yuchao Yang (yuchaoyang@pku.edu.cn)

Keywords: brain signal analysis; foundation models; language-signal alignment; zero-shot learning

Published in Advanced Intelligent Systems (2026), e70486. doi:10.1002/aisy.70486

## ABSTRACT

Brain signal analysis is essential for both neuroscience research and clinical diagnostics, vet current approaches face critical limitations. End-to-end models require task-specific retraining and exhibit limited generalization, while pre-trained models lack semantic depth and still depend on extensive fine-tuning. Meanwhile, general-purpose multimodal foundation models, though powerful in other domains, struggle to interpret brain signals due to representational misalignment and lack of domain knowledge. This study introduces a Multimodal foundation modEl for zero-shoT and multI-taSk brain signal analysis (METIS) through a unified language-signal alignment framework. METIS is pretrained on the largest and most diverse brain-signal corpus to date, comprising over 70 000 h of recordings from more than 11 000 subjects across 20 datasets. In a comprehensive zero-shot evaluation across 12 datasets, METIS outperformed the leading generalist model by over 20.9% in average accuracy. Remarkably, without any fine-tuning, METIS’s performance matches or exceeds that of supervised, task-specific models. Furthermore, METIS demonstrates exceptional data eficiency and strong generalization, achieving an average AUROC advantage of over 16.0% in few-shot settings and 15.9% in cross-dataset transfer. This work establishes a new paradigm for general-purpose brain signal analysis, paving the way for next-generation neurotechnology. The code is available at https://github.com/mingzhi-c/metis-brain-signal-foundation-model.

## 1 | Introduction

Brain signal analysis, especially using non-invasive electroencephalography (EEG) and invasive intracranial EEG (iEEG), plays a pivotal role in advancing neuroscience research and improving clinical diagnostics [1–4]. In recent years, deep learning has significantly propelled the field forward, demonstrating performance superior to traditional methods in tasks such as sleep stage classification, seizure detection, and the diagnosis of psychiatric and neurodegenerative disorders [5–8].

However, the heterogeneity of brain signal data—stemming from diferences in electrode setups, acquisition hardware, and participant cohorts—has made it dificult for existing models to generalize across tasks and datasets [9, 10]. Domain-specific models, including both end-to-end [11–15] and pre-trained architectures [16–20], are confined to task-specific training, which leads to high development costs, long iteration cycles, and limited adaptability to real-world demands such as zero-shot classification and multi-task processing. In parallel, general-purpose multimodal foundation models [21–24], while adept at instruction following and reasoning, lack the necessary domain knowledge to efectively interpret the complex dynamics of brain signals.

Current deep learning-based brain signal analysis methods can be broadly divided into end-to-end models and pre-trained models, both of which face significant challenges. End-to-end models are typically trained on single-task datasets and perform well in constrained settings, but they require retraining for each new task, lack zero-shot generalization, and are sensitive to changes in channel configuration and signal duration. For instance, representative architectures such as EEGNet [11] and EEGConformer [12] are typically designed and configured for specific datasets and single tasks. Although they perform well within their training domains, they fail to generalize when the task, channel montage, or signal duration difers, necessitating full retraining and limiting their adaptability across datasets and clinical scenarios. Pre-trained models, on the other hand, attempt to enhance feature extraction through large-scale pretraining. Representative examples include BIOT [16], which applies contrastive learning with signal perturbations, and LaBraM [17] and CBraMod [18], which employ masked reconstruction [25] in the temporal or spectral domain. However, these methods primarily target low-level signal reconstruction or local invariance, while neglecting the high-level semantic information embedded in clinical annotations. Consequently, their representations lack suficient semantic depth for clinical diagnostic tasks, exhibit poor generalization, and still require task-specific fine-tuning of downstream classifiers, limiting their ability to support zero-shot classification.

Meanwhile, the success of general-purpose multimodal foundation models across various domains ofers a new perspective for this field [26–29]. These models are capable of understanding natural-language instructions and performing zero-shot reasoning, allowing them to address complex tasks without requiring task-specific training [30–33]. These advancements raise a central question: can such “general intelligence” help overcome the persistent generalization challenges in brain signal analysis? However, experimental results indicate that directly applying existing general-purpose multimodal foundation models to brain signals leads to significant performance degradation (Figure 3b). The root cause lies in the models lack of domain-specific knowledge: they have scarcely been exposed to brain signal data during training, and their internal representational spaces are inadequate for capturing the complex spatiotemporal dynamics of brain signals. This representational misalignment often results in severe modality confusion and substantial performance collapse.

Here, we propose a Multimodal foundation modEl for zeroshoT and multI-taSk brain signal analysis (METIS) through a unified language-signal alignment framework. The model formulates neural state assessment and disease identification as question answering tasks based on brain signals and natural language prompts. To support this paradigm, we constructed the largest and most diverse brain signal instruction corpus to date, comprising over 70 000 h of recordings from more than 11 000 subjects across 20 datasets (Figure 1a). METIS’s architecture consists of three key components: a universal signal encoder that maps heterogeneous brain signals into a unified token sequence; a multimodal attention [34] mechanism that applies full attention to model the complete context of the signal sequence and uses causal attention for conditional language modeling, thereby bridging signal representations and language generation; and a mixture-of-experts [35–38] (MoE) module, which distinguishes between signal and language tokens to adaptively handle varying signal characteristics, facilitating flexible and specialized computation within a unified framework (Figure 1b).

To comprehensively evaluate METIS, we conducted systematic experiments across zero-shot, few-shot, and cross-dataset transfer settings (Figure 1c), comparing it against specialized models and general-purpose multimodal foundation models. Experiments covered 17 datasets with varying modalities, acquisition centers, and diverse task types. Across all scenarios, METIS consistently outperformed baseline models. In the zero-shot setting, it not only matched or exceeded the performance of supervised models trained on labeled data, but also outperformed the best general-purpose multimodal foundation model by over 20% in accuracy. Under few-shot conditions, METIS maintained a clear lead, with an average AUROC more than 16% above the best pre-trained model. It also delivered comprehensive outperformance in cross-dataset transfer tasks, surpassing the best end-to-end and pretrained models by an average of 19.8% and 15.9%, respectively. These results establish METIS as a universal and generalizable foundation model for brain signal analysis, demonstrating its potential to significantly reduce reliance on labeled data and enabling scalable, automated analysis across diverse clinical and research settings.

## 2 | Results

## 2.1 | Zero-Shot Generalization Across Diverse Clinical Tasks

METIS, a foundation model pretrained on large-scale, heterogeneous brain signal datasets, enables zero-shot brain signal analysis via a unified language-signal alignment framework. Given a brain signal segment and a natural language instruction (e.g., “Which sleep stage does this signal belong to?”), METIS generates answers without any task-specific fine-tuning (Figure 2a). This is formulated as a signal-question answering task. Crucially, this capability stems from learning clinically meaningful neural representations, rather than relying on shallow statistical correlations. We examined this in sleep stage classification, where Grad-CAM [39] visualizations revealed that METIS identifies the N3 stage [40] by attending to its defining clinical hallmark: slow-wave activity. This alignment with expert criteria indicates that METIS learns a semantically grounded and interpretable mapping between neural dynamics and high-level concepts.

Unlike traditional supervised models that require large amounts of labeled data and often overfit to task-specific distributions, METIS is designed to learn task-agnostic, crossdomain representations through large-scale pretraining. This enables plug-and-play generalization to new datasets, guided solely by natural language queries, without any access to training samples from the target domain.

To systematically assess its performance, we evaluated METIS across representative clinical tasks. In sleep stage classification, METIS achieved strong zero-shot AUROC scores on both the ISRUC [41] (94.1%) and Dreams [42] (92.9%) datasets, substantially outperforming competitive baselines trained with 10% of the labeled data (90.5% and 76.2%, respectively). This zero-shot superiority extended to the challenging task of interictal epileptiform discharge detection. On the Mayo [43] dataset, its zero-shot performance (AUROC 93.5%) not only surpassed all 1%-shot models but also rivaled specialized architectures like SPaRCNet trained with a 10% data budget. When fine-tuned on the same 10% subset, METIS further improved to an AUROC of 95.9%, establishing state-of-the-art performance on this task.

![](images/953c621a8392a334abc400c55a4c5e90bb1654e09987076b0de5feadb1d665ef.jpg)

Figure 1. Overview of data resources, acquisition modalities, application scenarios, and the METIS model framework. a, Data resources, modalities, and applications. Brain signals are acquired via multiple modalities, including scalp electroencephalography (scalp EEG), electrocorticography (ECoG), and stereoelectroencephalography (SEEG). Key clinical applications include sleep stage classification, epilepsy detection, Alzheimer’s disease detection, and detection of interictal epileptiform discharges (IEDs). The pretraining corpus consists of over 18 million signal segments from 20 datasets, encompassing recordings from more than 11,000 participants and totaling over 70 000 h. The circular chart illustrates the sample distribution across the datasets. b, Core architecture of the METIS model. Brain signals are first transformed into serialized tokens by a universal signal encoder, while natural language prompts are concurrently embedded as text tokens. The model integrates both token streams using a multimodal attention mechanism and stacked METIS blocks. Each block is composed of a Group Query Attention (GQA) layer and a Mixture-of-Experts (MoE) feedforward network. The left inset illustrates the unified multimodal attention mask, which governs the interactions between signal and text tokens. The right inset depicts the MoE layer, where a router directs tokens to specialized experts for targeted processing, while shared experts preserve generalizable representations. c, Evaluation paradigms for METIS. The model supports multiple evaluation paradigms: (i) zero-shot classification on unseen datasets; (ii) few-shot classification using a minimal number of labeled samples; (iii) signal question answering (Signal-QA), where the model processes both a signal and a natural language question to generate a textual answer; and (iv) cross-dataset transfer learning across diferent recording centers and patient cohorts.

A critical test for a foundation model is its robustness against major distributional shifts. We challenged METIS with crossspecies generalization on the RatEpilepsy [44] dataset. Even without prior exposure to rat data, it achieved a zero-shot AUROC of 63.2%, exceeding untrained baselines by more than 12% and matching or outperforming supervised models trained with up to 1% of the target-domain data. When fine-tuned on 10% of the data, METIS reached an AUROC of 83.6%, surpassing the best-performing baseline, SPaRCNet, by 6.6%. This generalization ability also extended to human seizure detection on the SEE [45] dataset. When evaluated in the zero-shot setting, METIS achieved an AUROC of 71.5%—outperforming untrained models by over 17%, surpassing all 1%-shot baselines, and exceeding the best-performing model, SPaRCNet, by 6.8%. With fine-tuning on the same data volume, its advantage further increased to 9.6%.

We further evaluated METIS’s versatility across a range of challenging psychiatric and neurological disorder detection tasks. For ADHD detection, it achieved zero-shot AUROCs of 72.4% on ADHD-121 [46] and 67.1% on ADHD-80 [47], exceeding the strongest untrained baselines by 23.7% and 9.6%, respectively. When fine-tuned on 1% or 10% of labeled data, METIS consistently outperformed all task-specific baselines. Similarly, in schizophrenia detection on the Schizo–Youth [48] dataset, METIS achieved a zero-shot AUROC of 72.1%, surpassing the strongest untrained baseline by 14.4%, the best 1%-shot model by 4.5%, and the best 10%-shot model by 2.8%. When fine-tuned on 1% and 10% of the labeled data, this margin further expanded to 12.8% and 18.4%, respectively. This trend continued for Alzheimer’s disease detection on the ADFSU [49] dataset, where METIS achieved a zero-shot AUROC of 74.6%, outperforming the strongest untrained baseline by 15.3% and surpassing the best 1%-shot and 10%-shot models by 11.6% and 5.8%, respectively. On the NMT [50] dataset for signal anomaly detection, METIS attained a zeroshot AUROC of 66.2%, exceeding the best untrained baseline by 16.5%. While its performance remained slightly below supervised baselines with 1% and 10% data, METIS regained a leading advantage when trained under the same data volume.

Aggregated results across all tasks (Figure 2b,c) highlight the model’s overall robustness and eficiency. On average, METIS’s zero-shot AUROC (75.5%) outperformed the 1%- shot supervised average (73.2%) and approached the 10%-shot level (81.6%). With minimal fine-tuning, METIS’s perfor mance surged to 82.0% (1% data) and 87.6% (10% data), outperforming all corresponding baselines. Furthermore, the violin plots reveal that METIS not only achieves a higher mean performance but also exhibits a significantly tighter performance distribution, reflecting more stable and reliable generalization across a diverse landscape of clinical tasks and datasets. These results establish METIS as a robust foundation model that consistently delivers strong zero-shot performance and scales efectively with limited supervision, demonstrating the promise of large-scale pretraining for generalizable and interpretable brain signal analysis.

## 2.2 | Zero-shot signal question answering beyond generalpurpose multimodal foundation models

General-purpose multimodal foundation models are reshaping AI with remarkable performance in language, vision, and reasoning, driving their adoption in visually grounded medical domains such as pathology and radiology [26, 27]. However, it remains an open question whether their capabilities extend to the complex, non-visual data of brain signals. Brain signal analysis presents a unique challenge due to high data variability across clinical settings and the need to ground abstract concepts in temporal patterns. For example, a model must learn to identify specific sleep markers or transient epileptic discharges directly from the signal.

We evaluate METIS against four leading general-purpose multimodal foundation models—ChatGPT, Gemini, Grok, and DeepSeek—on a total of 12 downstream datasets using two zero-shot protocols that share the same prompt and raw input segment. For clarity and conciseness, Figure 3 presents representative results on four datasets spanning sleep stage classification, Alzheimer’s disease detection, major depressive disorder detection, and interictal epileptiform discharge detection, while comprehensive results on the remaining 8 datasets are provided in Figures S18 and S19. The left panels of Figure 3 report multiple-choice accuracy, whereas the right panels report BERTScore [51] for open-ended answers. The openended protocol requires the model to generate precise clinical descriptions consistent with the underlying signal evidence.

The multiple-choice results show a consistent trend: METIS leads across all datasets, setting the top performance bar, whereas the strongest general model varies by task and trails behind in average accuracy. METIS achieved an average accuracy of 78.3% across four datasets, outperforming ChatGPT, the best general model, by 28.6%. The observed performance gap reflects the varying demands and signal characteristics across tasks. On the sleep stage classification dataset Dreams [42], METIS reached 76.9% accuracy, establishing a significant 51.7% lead over ChatGPT, and demonstrating a strong ability to identify complex sleep patterns. This advantage continued on ADFSU [49], an Alzheimer’s disease detection dataset, where METIS (81.7%) was more than 13% ahead of ChatGPT (68.4%). METIS’s superiority was also evident on Mayo [43], a dataset requiring precise detection of interictal discharges in iEEG. Here, its 84.9% accuracy surpassed that of Grok, the strongest of the generalist models, by over 26%. Finally, for major depressive disorder detection on MDD [52], METIS’s accuracy of 69.9% was more than 17% higher than DeepSeek, the best-performing general model on this dataset.

![](images/30a7f743c99a0064522b376e8b7b9db56c9d2ba8e72c862c7604629978e6a902.jpg)

![](images/e43a59df5fc62962564de05b4d447ae52d2a1acffd85e3c5ac27bd82861171f9.jpg)

![](images/1790cd530a396c1ac9fb46eb0c5d71bec8c7cd4303ff2ff12c1b528d7d634c6b.jpg)

<table><tr><td>Which sleep stage does this signal belong to? A. Wake</td></tr><tr><td>B. Non-REM Stage 1</td></tr><tr><td>C. Non-REM Stage 2</td></tr><tr><td>D. Non-REM Stage 3</td></tr><tr><td>E. Rapid Eye Movement</td></tr></table>

b  
![](images/a9aed0c1ce81be17fb6d679c04d89a65747fb30233062d3edb47b55c5b7d7761.jpg)

![](images/04cbba9074d8c9a564cd3097780c7e9f71bccba38d6b743f80a3bac60238047d.jpg)

![](images/d94f0cc496b035ac28d791e450931c74be9e1db2b9f2d1c4ae093970710caf50.jpg)

![](images/cd946e895540707447efe055c8820e90a536ca937b998e910e52d1057de67a3d.jpg)

![](images/882d9f54f8053fbf6dbede2ff2b1d5332ec919a8811dac1a826903d46ded9045.jpg)

![](images/9e0426cecf059f24721251a9b2702645b4fac90a79516127a749cda911cf8a20.jpg)

![](images/b78f6614c30f35d8a2498887a5943b48cf14687f7b0b046c75a9fc7dabc129b6.jpg)

![](images/5f8bcfaca3a623934392109fdc899a446f41e00d751b704e703d31010e5085b7.jpg)

![](images/e3a56fee959c846ec4e28a88eb3e3b34aa9a6c39b057e9d14179b32be11ffd09.jpg)

![](images/5c9e7b3489a4ca5b2bc1a14607eda00faeb921fe4d3206f82e30c654f917b25f.jpg)

![](images/f3882d7f8c345860c5fa30d03cf81ccb7b84eef6963e5d1fc4f2059bbbc6a24f.jpg)

![](images/eb66ca85d69caa1db22737da65dd3e030346966c5818577dbb876c20ae94a39a.jpg)

C  
![](images/c6e274baf33f5283a8a4cb0dce696b549ae2811a31ace7a4d8b731c80d6823b8.jpg)

![](images/247284439f4fb4365b6560c49c0d7dfe8e8f30c83bd85cc7fcc2fc120254e1b0.jpg)

![](images/1fe75a8791433da05971107ef39251943d14a58794ae836608131866dcb8f7f2.jpg)  
Figure 2. Zero-shot performance of METIS in multi-task scenarios. a, Illustration of zero-shot classification. Using sleep stage classification as an example, the model receives raw brain signals along with a natural language query (“Which sleep stage does this signal belong to?”) and, without any task-specific fine-tuning, outputs the correct answer (“D. Non-REM Stage 3”), while identifying relevant signal patterns (e.g., N3 slow-wave activity). b, Comparison of zero-shot performance with three end-to-end baselines (EEGNet[11], EEGConformer[12], SPaRCNet[13]) trained under limited-supervision conditions (1% and 10% labeled data), measured by AUROC. Across 12 datasets covering epilepsy detection, sleep stage classification, and psychiatric disorder diagnosis, METIS consistently outperforms task-specific models trained with limited supervision $( ^ { * } \mathrm { p } < 0 . 0 5 , ^ { * * } \mathrm { p } < 0 . 0 1 , ^ { * * * } \mathrm { p } < 0 . 0 0 1$ ; two-sided t-test). c, Performance distributions under diferent data conditions (AUROC, violin plots). The overall mean and median AUROC of METIS under zero-shot settings substantially exceed those of other models trained with 1% and 10% of the data, underscoring its cross-task generalization and data eficiency.

The open-ended setting demonstrates consistent trends, which requires the ability to articulate precise clinical concepts grounded in temporal signal patterns. Averaged across four datasets, METIS achieved a BERTScore of 70.7%, putting it 17.0% ahead of the best general baseline, Grok (53.7%). On the Dreams dataset, METIS demonstrated a substantial improvement over the best-performing baseline, Grok, with a 42.1% higher BERTScore (88.6% vs. 46.5%). It also held a significant 22.6% advantage on the Mayo dataset (76.7% vs. 54.1%). Although the performance gap was narrower on the ADFSU and MDD datasets, METIS remained ahead of the strongest general models.

Qualitative examples in Figure 3a reveal the mechanistic failures underlying the quantitative results. A prominent failure mode in several generalist models is a strong thematic bias, where signals from diverse clinical contexts are erroneously mapped to epilepsy. This likely stems from incorrect associations formed during pretraining on vast text corpora, where brain signals are disproportionately discussed in relation to epilepsy, creating a strong but mistaken prior. A more fundamental error is modality confusion, exemplified by Grok, which misinterprets EEG morphology as ECG and consequently provides a cardiovascular diagnosis for a psychiatric condition (major depressive disorder). Furthermore, ChatGPT’s per formance highlights a critical imbalance: despite being the strongest generalist model in multiple-choice recognition on Dreams and ADFSU, it ranks last in faithful, open-ended narration for the same tasks. These error patterns collectively point to a disconnect between discriminative pattern matching and true generative understanding. METIS consistently mitigates these failure modes, demonstrating robust performance that stems from its core design of coupling temporal signal structure with semantic concepts through instruction-driven pretraining.

Taken together, these results highlight that the observed performance gap is not simply quantitative, but reflects qualitative diferences in model design. Generalist models often exhibit thematic biases, modality confusion, and a discon nect between recognition and generation failures that point to limitations in grounding language to temporal brain signal data. METIS overcomes these challenges through a dedicated architecture that tightly couples semantic interpretation with temporal structure. This demonstrates the necessity of moving beyond general-purpose models and toward specialized, signal-aware architectures to achieve robust performance in brain signal analysis.

## 2.3 | Few-shot Classification

A foundation model’s practical utility, especially in data-scarce clinical environments, is critically defined by its ability to generalize from very limited labeled data. This challenge is particularly acute for brain signals, which are highly heterogeneous across individuals and often characterized by low signal-to-noise ratios. To assess this capability, we systematically evaluated METIS on 14 brain signal datasets under 1-shot, 2-shot, 4-shot, and 8-shot settings.

We adopted a linear probing [53] protocol to rigorously test the quality of the learned representations. In this setting, all model backbones were frozen, and only a single linear classification layer was fine-tuned. This strategy isolates the contribution of the pretrained features and reflects a practical scenario in which foundation models are used as general-purpose feature extractors without task-specific adaptation. Our evaluation covered nine distinct clinical tasks, such as sleep stage classification, epilepsy detection, psychiatric disorder diagnosis, anesthesia depth estimation, and neurodegenerative disease identification.

As shown in Figure 4a, METIS consistently and significantly outperforms all baseline models across the majority of datasets and shot levels. Its advantages are especially evident in high-complexity diagnostic tasks. For example, in sleep stage classification on the Dreams dataset, METIS achieves AUROCs of 95.2% and 95.7% under the 2-shot and 8-shot settings. These results surpass the best baseline, BIOT, by 39.8% and 34.0%, respectively. Similar gains are observed in interictal epileptiform discharge detection on the Mayo dataset, where METIS scores 93.6% (2-shot) and 93.7% (8-shot), outperforming CBraMod by over 22%.

Even on tasks unseen during pretraining, METIS maintains strong generalization. On the NTUHBIS [54] dataset for anesthesia depth monitoring, the model achieves 78.0% and 83.8% AUROC under 2-shot and 8-shot settings, respectively. These results confirm its ability to transfer across clinical domains with minimal supervision.

In contrast, existing baselines exhibit pronounced task sensitivity. BIOT performs well on sleep stage classification but struggles on Mayo. CBraMod is relatively stable across datasets, yet fails to reach competitive performance on Dreams. This inconsistency limits their utility in practice, especially in low-resource settings. The ISRUC and ADHD-80 datasets provide further evidence (Figure 4a, box plots): METIS exhibits smooth and reliable improvement with more labeled data, while models like LaBraM show minimal gains, with an AUROC increase of only 2% from 2-shot to 8-shot on ADHD-80.

These trends are confirmed by aggregated performance across all datasets, summarized in Figure 4b. METIS achieves average AUROCs of 76.9%, 77.6%, 78.9%, and 81.4% for the 1-shot, 2-shot, 4-shot, and 8-shot settings. The margins over the best-performing baseline, CBraMod, remain in a narrow band between 16% and 18% AUROC, indicating stable advantages across data regimes.

![](images/b13caea3f544f7a6c82166a1623fcdb105c21eab03b8beab6078e91c65e04792.jpg)  
ChatGPT

## Sleep stage classification

![](images/ee2003a8a4771db25fa7eb8e7081234ca49f06eede84e2a24653db14d292e20e.jpg)

## Sample from Dreams dataset

## (I) Multiple-choice QA

Which sleep stage does this signal belong to? A. Wake B. Non-REM Stage 1 C. Non-REM Stage 2 D. Non-REM Stage 3 E. Rapid Eye Movement

## (II) Open QA

Which sleep stage does this signal belong to?

## Depression detection

![](images/fc08b071321ab70cfed39030749441be4298e880235fd261923540091ba40b30.jpg)

## Sample from MDD dataset

## (I) Multiple-choice QA

Which disease does this   
signal belong to?   
A. Normal   
B. Major depressive disorder

(II) Open QA

Which disease does this signal belong to?

![](images/ee8868f27c804017e90ce81f97a32fb5a2742fa44979424792a00fe04a9376c0.jpg)  
ChatGPT

(II). This signal belongs to the wakefulness stage.

![](images/15da837f608add1f8c891c11e3d6afd7d7b653d1e72847ea784aece119b66aa6.jpg)

## (1). A

(II). This signal belongs to sleep stage N2.

![](images/1deeb7e962323a424b1c8f16025977abf0e03ca6105bbff236592f36dfbf0fe0.jpg)

(1). C (Il). Stage 2 sleep.

![](images/9e2a536beede9b73a5013f73407dd57ef05618109c0b83abddfb54a14c411d46.jpg)

![](images/8d7d685f32ce16ae4bfff8940432ee7133b59b266ec18daf4687690e4e96862e.jpg)

## (1). A

(II). This is an EEG pattern associated with epilepsy.

![](images/e9d1b02da438253666dfe7dc34bac7aefb30d8817623013ec139cc602790df1e.jpg)

## ✓(1). B

(II). This signal belongs to the disease epilepsy.

![](images/47f752ddb3dd5620d3f2ae2a81057aae16786d93ea4f94e574ff54d26504aced.jpg)

(1). A   
(II). Myocardial   
infarction

(1). A (II). Epilepsy DeepSeek

✓(1).B   
O METIS ✓ (I), Major   
METIS depressive disorder

## Alzheimer's disease detection

![](images/64565da5e36ca80351da632ba24e1b2e4933a5433eb8b6adacb1edd9229ed749.jpg)

![](images/1b55465d183593f012d6b0e8370303b4f214c48e6c64264b360d75052d49ba50.jpg)

## Sample from ADFSU dataset

(Ii). This signal - belongs to epilepsy.

![](images/be29546ab3b6d7cc2d7b8d6c821144a29cb16d7aedcdce18cce730e8199fd153.jpg)

(II). This signal belongs to the disease epilepsy.

## (I) Multiple-choice QA

Which disease does this signal belong to? A. Normal B. Alzheimer's disease (il). Creutzfeldt-Jakob disease

![](images/a80c51206a3b394041f223c3fd6aacb3b029dccec46bc1b4375e2162cf0a59b8.jpg)

## (II) Open QA

![](images/0cc563f24462b63d6235abf4b2318dd57bc8b1a997d09577c2640ed5c889fdbe.jpg)

Which disease does this signal belong to?

## Interictal epileptiform discharge detection

![](images/e454fdb503b36e92011d0e23cbfa1581ec6b2f9a95affe4e37a397840c47ec46.jpg)

## Sample from Mayo dataset

## (I) Multiple-choice QA

(II) Open QA

Which epilepsy state does   
this signal belong to?   
A. Interictal   
B. Interictal, pathological   
activity

Which epilepsy state does this signal belong to?

![](images/e4b8514a8ae3b75e00483aaf1f4609cf10051c4f80d292be5da30616b94f75fb.jpg)

![](images/5a66b37efc7cfd9dccf4a36e5ee4b2f4b230d59d1570c3ea52a90b08780dafd6.jpg)

(1). A (II). The absence seizure state.

![](images/34cae2c05383ef082def8a1b527cb636fd035b2568148dad3252daeecd5e6ab7.jpg)

(1). A (II). Ictal state

![](images/51a40311d90f3eebb4dd22bac12af38febff5bee7a1945b3ae9a3a6bacdb26c1.jpg)

(1). A (IÍ). Epilepsy state DeepSeek of focal seizures.

√(1).B METIS(II). Interictal, METIS pathological activity

b  
![](images/0f321a01fa32c986982ea787466558b3766a083358dd0d7e2a1c71285bdcac4f.jpg)

## ChatGPT

## Gemini

![](images/824eaa7e2449f195017a6fe03b898656af8d56a8b9f8cfc09d91e284086fa7a9.jpg)  
Figure 3. Zero shot performance comparison with general-purpose multimodal foundation models. a, Task examples and model responses on four representative datasets spanning sleep stage classification, Alzheimer’s disease detection, major depressive disorder detection, and interictal epileptiform discharge detection. Each example pairs one multiple-choice query and one open ended query for the same signal, with outputs from METIS, ChatGPT[21], Gemini[22], Grok[23], and DeepSeek[24]. For readability, only correct answers are marked with a green checkmark; unmarked answers are incorrect. b, Zero-shot results under two protocols. Left panels report multiple-choice accuracy. Right panels report BERTScore[51] for open-ended answers. Bars show means over repeated runs, with the rightmost groups giving the overall mean across datasets, where significance markers denote the diference between METIS and the top-performing generalist model (\*p < 0.05, \*\*p < 0.01, \*\*\*p < 0.001; two-sided t-test). METIS consistently outperforms generalist multimodal models in both protocols.

Figure 4c further illustrates this pattern using normalized radar plots. METIS presents a balanced and broad performance profile across all tasks, indicating strong task-invariant representation learning. In contrast, the baselines show uneven, spiked distributions, suggesting that their success is heavily influenced by task-specific properties rather than generalizable knowledge.

In summary, METIS demonstrates systematic and robust few-shot learning performance across a wide spectrum of clinical brain signal tasks. We attribute this strength to its unified pretraining paradigm, which encourages the alignment of neural signals with semantically meaningful instructions. This results in a shared latent space that supports rapid adaptation with minimal supervision. By consistently outperforming task-specific models with only a handful of labels, METIS ofers a scalable and practical solution for real-world brain signal analysis, particularly in data-limited clinical scenarios.

## 2.4 | Cross Dataset Transfer

In real-world clinical applications, brain signal models often operate across sites that difer in data acquisition and patient cohorts. Variation in protocols, electrode montages, hardware, sampling rates, and demographics produces substantial distribution shift, which is a major obstacle to practical deployment. To evaluate robustness under these conditions, we conducted 8 cross-dataset transfer experiments spanning four task categories: sleep stage classification, Alzheimer’s disease detection, attention deficit hyperactivity disorder (ADHD) detection, and interictal epileptiform discharge detection. In each experiment, models were trained on one dataset and evaluated directly on another dataset from the same task category without any fine-tuning. Baselines covered both end-to-end models (EEGNet, EEGConformer, SPaRCNet) trained with full supervision on the source data and pretrained models (BIOT, LaBraM, CBraMod) evaluated with frozen encoders and a linear classification head. METIS followed the same linear probing protocol to isolate representational quality.

As shown in Figure 5a, METIS consistently outperformed all baselines across 8 transfer settings. The largest gains appeared in sleep stage classification. When transferred from ISRUC to Dreams, METIS achieved an AUROC of 94.0%, exceeding the best-performing baseline in this setting, EEG-Conformer (73.1%), by 20.9%. In the reverse direction, from Dreams to ISRUC, METIS reached 95.2%, while the best baseline in this setting, EEGNet, scored 64.0%, yielding a margin of 31.3%. These results indicate that METIS retains sleep-relevant structure despite substantial changes in channel configuration between datasets.

Similar patterns held in the remaining tasks. For Alzheimer’s disease detection, METIS attained 72.6% AUROC when transferred from ADFSU to APAVA [56], outperforming the bestperforming baseline in this setting, EEGConformer (57.2%), by 15.4%. In the reverse transfer, METIS achieved 80.3%, surpassing the best baseline, BIOT (64.6%), by 15.7%. For ADHD detection, METIS reached 73.9% from ADHD-80 to ADHD-121 versus 72.6% for the top baseline in that setting; in the opposite direction it achieved 82.7%, exceeding BIOT (78.3%) by 4.4%. In interictal epileptiform discharge detection, METIS obtained 94.6% when transferred from IEDS [57] to Mayo, outperforming the best-performing baseline (BIOT, 82.8%) by 11.8%, and 65.2% from Mayo to IEDS, ahead of the best baseline (BIOT, 62.1%) by 3.1%. Averaged across all transfers, METIS improved AUROC by 15.9% relative to the best baseline per setting. Together, these results highlight the instability of existing methods under distribution shift, whereas METIS maintains strong and consistent performance.

To examine the representational basis for this robustness, we visualized t-SNE embeddings from the sleep stage classification task (Figure 5b). METIS produced compact, well-separated clusters whose spatial arrangement was consistent across IS-RUC and Dreams. For example, the relative proximity of wake stage and rapid eye movement stage was preserved across datasets, suggesting that METIS learns task-relevant, domaininvariant features rather than dataset-specific artifacts. In contrast, embeddings from baseline models were often entangled and lacked a stable geometry; LaBraM and CBraMod, in particular, showed overlapping class regions with unclear boundaries, which helps explain their reduced transfer performance.

We also analyzed the behavior of the Mixture-of-Experts (MoE) module in METIS by inspecting expert activation patterns across tasks and datasets (Figure 5c). The heatmaps reveal a functional division of roles. Certain experts are preferentially selected for specific task families, such as Expert 19 in sleep stage classification and Expert 73 in interictal epileptiform discharge detection. Other experts, such as Expert 25, are activated broadly across tasks, indicating that they capture common signal attributes like rhythmic structure or artifact suppression. This modular organization allows METIS to combine specialization with generalization: computation can be routed to experts that encode task-specific cues while still leveraging experts that model shared, transferable patterns. Such adaptive routing provides a structural basis for the stable cross-domain performance observed in Figure 5a.

In summary, METIS demonstrates reliable domain generalization across four clinical task categories and 8 cross-dataset transfers. The performance advantages are accompanied by interpretable evidence: t-SNE shows topologically consistent, semantically aligned representations across datasets, and MoE analysis shows a complementary mixture of specialized and generalist experts. These findings suggest that coupling strong pretrained representations with modular expert routing is an efective strategy for building brain signal foundation models that remain robust under the distribution shifts encountered in practice.

![](images/24009a625340817d9751e80a14ad21de2c2c4c96e08e0eb851a5cf52d2c44700.jpg)

![](images/28b1c3c8d6af4c609c349ff9ec811e97a714a683d018bb80f512e432fcc5aa3f.jpg)

![](images/1a50e95c61da6c6640cd430e39ff6bcc1e51bf342002c2652150ca1782f20f2a.jpg)  
Figure 4. Few-shot performance comparison of METIS. a. Few-shot classification performance (AUROC) across 14 downstream datasets. Bar plots show results under 2-shot and 8-shot conditions, where METIS consistently outperforms existing pretrained baselines (asterisks indicate statistical significance: ${ } ^ { \ast } \mathrm { p } <$ $0 . 0 5 , \mathrm { ^ { * * } p < 0 . 0 1 , \mathrm { ^ { * * * } p < 0 . 0 0 1 } }$ ; two-sided t-test). Box plots below further illustrate performance trends on the ISRUC and ADHD-80 datasets under 1-shot, 2-shot, 4-shot, and 8-shot settings, demonstrating that METIS maintains stable advantages even with extremely limited samples. b, Average performance across all 14 datasets. Bar plots compare mean AUROC under 1-shot, 2-shot, 4-shot, and 8-shot conditions, showing that METIS consistently leads at diferent data scales and remains robust in low-sample regimes. c, Normalized AUROC radar plots across all tasks. METIS achieves a larger and more uniformly distributed coverage, highlighting its consistency and stability across tasks, whereas baseline models exhibit stronger task dependence.

## 2.5 | Ablation Analysis of Key Design Components

To systematically evaluate the key factors underlying METIS’s performance, we conducted three categories of ablation studies that isolate the efects of model architecture, pretraining modality composition, and data scale. The protocol covered zero-shot inference on 12 downstream datasets and 8-shot linear probing on 14 datasets, with training and evaluation kept consistent across variants to ensure comparability.

First, under structural ablations, removing the mixture-ofexperts module (METIS-noMoE) consistently reduced zeroshot performance compared with the full model. Averaged across the 12 datasets, the mean diference between METIS and METIS-noMoE was 10.8% AUROC, with the largest drops appearing on psychiatric and anomaly-detection benchmarks. On the MDD dataset the reduction was 43.7% AUROC, and on the Schizo–Youth dataset it was 25.2%. NMT, ADHD-80, and ADHD-121 datasets also showed clear decreases. These patterns indicate that expert routing creates useful functional subspaces that improve representation quality when class boundaries are subtle or when distributions vary across tasks, which is consistent with the scatter trends visible in Figure 6a.

Second, the composition of pretraining modalities had a marked impact on cross-modality transfer. The EEG-only variant maintained reasonable performance on EEG datasets but averaged approximately 50% AUROC on iEEG tasks such as Mayo and IEDS. The iEEG-only variant performed well on iEEG with an average of 70.4% AUROC, yet its average on EEG datasets fell to 47.6%. In contrast, METIS trained jointly on EEG and iEEG was the top performer in 11 of the 12 zeroshot evaluations and was only slightly below the iEEG-only variant on Mayo by 1% AUROC. Sleep stage classification further illustrates this asymmetry: the iEEG-only variant nearly collapsed on ISRUC and Dreams with AUROCs around 39%, whereas METIS remained near or above 93%, in line with the modality-specific clusters seen in Figure 6a.

Third, data scale had a systematic efect on both accuracy and stability. As the pretraining corpus increased from 0% to 100%, the cross-dataset mean AUROC rose from 49.0% to 75.5%. Gains appeared early with only 1% of the data, where the mean already reached 63.6%, and continued to accumulate from 10% to 50% and from 50% to full scale. The violin plots in Figure 6b reflect this progression as higher central tendencies and tighter spreads at larger scales. The magnitude of improvement was task dependent. Sleep stage classification approached a ceiling quickly, since ISRUC and Dreams were already close to 93% at 1% and improved only modestly thereafter. Psychiatric and anomaly-detection benchmarks benefited more strongly from scale, with the Schizo–Youth dataset increasing from about 30% at 0% to about 95% at full scale, and NMT rising from about 33% to about 66%. These task-dependent responses explain why scaling reduces variance as well as raises the mean.

Finally, under the 8-shot setting, METIS achieved the highest or joint-highest AUROC on 11 of the 14 datasets. Its overall mean was 81.5%, compared with 75.4% for METIS-noMoE, 75.0% for METIS-EEG, and 76.3% for METIS-IEEG. Datasetwise margins were generally small in this low-label regime, but METIS still led by an average of about 5.1% AUROC over the strongest variant per dataset. Cases where a variant slightly exceeded METIS aligned with that variant’s specialization. METIS-IEEG was marginally higher on Mayo and Schizo-28, and METIS-EEG was marginally higher on ADHD-121. The aggregate bar summaries in Figure 6c are consistent with this pattern and show that the full model is the most stable across EEG and iEEG domains.

Taken together, the ablations reveal a coherent picture. The mixture-of-experts architecture supports task-adaptive specialization and helps on domains with weak or overlapping class structure. Joint EEG-iEEG pretraining provides the modality alignment required for transfer in both directions, avoiding the failures observed with unimodal pretraining. Scaling the pretraining corpus improves mean performance and reduces across-task variability, with the largest gains on intrinsically harder or more heterogeneous tasks. These factors explain the robustness of METIS in both zero-shot and few-shot regimes and provide practical guidance for designing transferable foundation models for brain signals.

## 3 | Discussion

In this study, we introduce METIS, a novel multimodal language-signal foundation model designed for generalpurpose brain signal analysis. Through extensive evaluation across 17 downstream datasets, we demonstrate that METIS surpasses existing end-to-end and pretrained models in zero-shot and multi-task settings, requiring minimal or no additional training. Unlike previous paradigms that rely on single-modality or task-specific models, METIS achieves exceptional cross-task and cross-dataset generalization by aligning heterogeneous brain signals with natural language instructions in a unified space. Specifically, the model shows outstanding performance in critical applications such as sleep stage classification, seizure detection, and psychiatric disorder diagnosis, achieving a zeroshot average accuracy of 70.4%. This significantly outperforms general-purpose multimodal foundation models (49.5%), and METIS maintains a consistent advantage in few-shot and crossdataset transfer tasks.

![](images/760d6cf718481b796d0b9af69262adff80aaa0c60992a6ab05d2bcd40d0afe79.jpg)

![](images/f5b867ad04a658a323244eedda2a8fad0e208fdec0086bd3cbf15fa029aba597.jpg)

![](images/d6d9b1410f98cab9e1ddee06585ba2e19524a47f0a992f1a4ad08298c895f36b.jpg)  
Figure 5. Cross dataset transfer, representation geometry, and expert routing in METIS. a, Cross dataset transfer performance. In each experiment, models are trained on a source dataset and evaluated directly on a diferent target dataset within the same task category (for example, trained on ISRUC and evaluated on Dreams). Bar plots report AUROC for METIS together with end-to-end baselines (EEGNet, EEGConformer, SPaRCNet) and pretrained baselines (BIOT, LaBraM, CBraMod) across 8 transfer settings. b, Feature visualization for sleep stage classification on ISRUC and Dreams. Two-dimensional t-SNE[55] projections are shown for METIS and representative pretrained models; points are colored by sleep stage. The plots illustrate inter-class separability and intra-class compactness, as well as the consistency of relative class layouts across datasets. c, Expert activation heatmap for the METIS mixture-of-experts module. Each row corresponds to a downstream task and each column to an expert. Colors encode routing probability from low (blue) to high (red). The patterns reveal task-dependent expert utilization together with experts that are active across tasks, indicating a balance of specialization and sharing.

The performance breakthrough of METIS stems from its language-guided causal modeling paradigm combined with large-scale pretraining, which overcomes the fundamental limitations of the two current mainstream approaches. End-to-end models are constrained by task-specific training, resulting in poor generalization. In contrast, the representations learned by unsupervised pretrained models often lack high-level semantic information, which limits their ability to perform zero-shot classification. Meanwhile, general-purpose multimodal foundation models often exhibit modality confusion and degraded performance owing to their lack of domain-specific representations for brain signals. METIS addresses these challenges through an instruction-driven signal– language alignment mechanism that maps brain signal representations to clinically meaningful semantic concepts. This framework enables METIS to lever age the largest and most diverse brain signal corpus to date for pretraining, learning generalizable representations with semantic depth.

The heterogeneity of brain signal data poses a core challenge for training a general brain signal analysis model. Our approach provides an efective solution to this problem by constructing a unified language-signal joint space. METIS maps heterogeneous brain signals from various sources into a shared representation space through its unified signal encoder and employs a MoE module to enable adaptive computation, thereby efectively mitigating the issue of distribution shift. This training paradigm ofers a general blueprint for integrating diferent types of physiological signals and can be extended in the future to the analysis of other time-series data such as electrocardiograms (ECG) and electromyograms (EMG). It even holds the potential to incorporate structured data like genomics to build a multimodal foundation model covering a broader range of clinical scenarios.

The introduction of METIS marks a paradigm shift in brain signal analysis from “specialized tools” to “general-purpose assistants”. Traditionally, each task required customized modeling, leading to redundant processes and high costs. METIS’s zero-shot capabilities fundamentally transform this model: clinical researchers can directly analyze unseen data using natural language instructions, significantly lowering the technical barrier and reducing model deployment time from months to hours. For example, METIS’s average zero-shot performance in this study not only surpassed supervised models trained with 1% labeled data but also approached those trained with 10%, indicating strong potential for reducing annotation costs in practical applications.

Although the zero-shot and few-shot classification results of METIS are encouraging, several key issues must be addressed on its path toward broader applicability. While we utilized the largest dataset to date, the data are primarily sourced from existing studies, and there remains room for improvement in covering diverse global populations, rare disease cohorts, and varied acquisition hardware. Future work will require evaluation in broader, multi-center prospective cohorts to further validate robustness and generalizability across clinical settings.

Looking ahead, the principles established by METIS ofer a scalable framework for next-generation brain-signal foundation models. We envision future work advancing along three key trajectories. First, a systematic investigation of scaling laws is essential to understand the interplay between model size, data diversity, and emergent capabilities. Expanding pretraining data to the million-hour scale and exploring Mixture-of-Experts architectures with trillions of parameters will unlock unprecedented potential for decoding complex neural states and analyzing rare diseases. Second, advancing these models toward generative and controllable paradigms will enable not only classification but also the simulation of neural activity, thereby suggesting novel intervention strategies. Third, developing lightweight versions via knowledge distillation [58] is crucial for deployment on low-power edge devices, such as wearable neural interfaces. This, combined with methods for eficient personalization, will pave the way for truly personalized neurology. Sustained eforts along these directions will transform METIS from a powerful research prototype into an indispensable platform that bridges general AI with the nuanced demands of computational neuroscience and clinical medicine.

## 4 | Experimental Section

## 4.1 | Instruction-Driven Pretraining: Overview and Objective

We formulate the pretraining of METIS as an instruction-driven generation problem, where the model is required to produce natural language responses grounded in raw EEG or iEEG signals under guidance of a textual prompt. This formulation not only enables a unified sequence-to-sequence modeling paradigm across modalities, but also provides a scalable pretraining strategy that leverages diverse biomedical tasks in a generative way.

As shown in Figure 1, the model receives a time-series brainsignal segment along with a natural language instruction (e.g., “Which disease does this signal belong to?”). The raw signal is first transformed into a sequence of spectro-temporal tokens by a universal signal encoder. The instruction is processed by the same model pipeline: it is first tokenized with the Qwen 2.5 [59] tokenizer into discrete indices and then passed through the model’s internal embedding layer to form instruction tokens. These two streams of tokens are concatenated into a single unified sequence, which is processed by a decoder-only Transformer equipped with a hybrid attention mask: signal tokens attend bidirectionally, while instruction and answer tokens follow causal masking.

![](images/b8a3c4b16d6b8e90a98a1c0edc79f08ec5f6294d9cfa0f31f9e66ce91b5154a5.jpg)

![](images/1998d895a57367be70700adfeca708b06e213724f8699a29a74b586733a78622.jpg)

![](images/4001e17db2e8fa48d998f3eb75fe4293adc9a1d03a600aecc3c453cb4a22a295.jpg)  
Figure 6. Ablation studies on key components of METIS. a, Zero-shot performance of diferent model variants across 12 downstream tasks. Scatter plots compare AUROC of the full METIS model with three ablated variants: without the MoE module (METIS-noMoE), pretrained only on EEG (METIS-EEG), and pretrained only on iEEG (METIS-IEEG). Colored dashed lines indicate the average performance of each model across all tasks. b, Impact of pretraining data scale on zero-shot performance. AUROC distributions across 12 tasks are shown for models trained with diferent proportions of pretraining data (0%, 1%, 10%, 50%, 100%). Results illustrate the performance trend from random initialization (0%) to increasing scales of pretraining. Black dashed lines connect the mean performance at each data scale. c, Few-shot (8-shot) performance of diferent model variants across 14 downstream tasks. Bar plots compare mean AUROC of the full METIS model and its ablated variants under the 8-shot setting. Error bars denote standard deviations from five-fold cross-validation.

During pretraining, METIS is optimized in a signalconditioned autoregressive manner, learning to generate taskaware answer tokens grounded in both neural dynamics and textual instructions. Formally, let � denote the input brain signal segment, � the instruction prompt, and $A = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { L } \}$ the target answer sequence. The model learns a conditional distribution over answer tokens:

$$
p ( A \mid X , I ) = \prod _ { t = 1 } ^ { L } p ( a _ { t } \mid X , I , a _ { < t } )
$$

All signal and instruction tokens are fully visible to the answer tokens via a causal attention mechanism, while answer tokens attend only to previous answer tokens through a causal mask. The learning objective is defined as the negative log-likelihood:

$$
\mathcal { L } _ { \mathrm { g e n } } = - \sum _ { t = 1 } ^ { L } \log P ( a _ { t } \mid X , I , a _ { < t } )
$$

Unlike vision-language models that align static visual patterns, METIS operates on temporally dynamic neural signals, requiring the model to infer latent neurophysiological states and map them to semantic clinical concepts during generation. This formulation enables METIS to integrate temporal neural representations and linguistic reasoning within a unified autoregressive framework.

Through this signal-conditioned generation objective, METIS learns to associate structured neural dynamics with high-level clinical semantics, supporting zero-shot reasoning across heterogeneous brain signal tasks without task-specific supervision.

In addition to this generation loss, we incorporate an auxiliary sparsity-aware loss $\mathcal { L } _ { b a l a n c e }$ to encourage expert load balancing in the Mixture-of-Experts (MoE) module. The final training objective is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { g e n } } + \lambda \mathcal { L } _ { \mathrm { b a l a n c e } }
$$

Where $\lambda = 0 . 0 0 1$ is a hyperparameter chosen to balance semantic learning and expert diversity during training.

## 4.2 | METIS Architecture

METIS adopts a unified decoder-only Transformer architecture tailored for multimodal reasoning over brain signals and natural-language instructions. Its core design reformulates EEG and iEEG analysis as instruction-driven generation, processing concatenated sequences of raw signals, task prompts, and target answers. The architecture is composed of three synergistic components: a universal signal encoder that transforms heterogeneous recordings into structured representations; a multimodal Transformer backbone integrating Group Query Attention (GQA) [60] and a MoE module for eficient, adaptive computation; and modality-aware mechanisms that handle structural and positional distinctions across signal and language tokens.

## 4.2.1 | Universal Signal Encoder

The universal signal encoder converts raw EEG and iEEG recordings, which may vary in channel count, sampling frequency, and recording duration, into a tokenized format suitable for integration with textual instructions (Figure 7). Each input channel is first Z-score normalized to reduce baseline drift and amplitude variability. To account for the nonstationary nature of brain signals, we apply a Short-Time Fourier Transform (STFT), converting each time-domain waveform $x _ { c } ( t )$ into a log-scaled spectrogram $S _ { c } ( f , t )$

$$
S _ { c } ( f , t ) = \log ( 1 + | \mathrm { S T F T } ( x _ { c } ( t ) ) | )
$$

Each spectrogram is then processed by a convolutional projection module that transforms the 2D time-frequency representation into a sequence of feature tokens per channel, each with a fixed embedding dimension D. The resulting token sequence preserves the spectral-temporal structure and serves as the intermediate representation for attention-based modeling. To enable interaction across channels, a multi-head self-attention (MHSA)[34] mechanism is applied:

$$
z _ { c , t } ^ { \prime } = \mathbf { M H S A } ( z _ { 1 , t } , z _ { 2 , t } , . . . , z _ { C , t } )
$$

The original and attended representations are fused via residual addition and global average pooling across channels:

$$
Z _ { S } = \frac { 1 } { C } \sum _ { c = 1 } ^ { C } ( z _ { c , t } + z _ { c , t } ^ { \prime } )
$$

This token sequence $Z _ { S }$ is concatenated with embedded instruction tokens $Z _ { T }$ to form the full input:

$$
Z = [ Z _ { S } ; Z _ { T } ]
$$

This design supports joint modeling of neural dynamics and semantic prompts in a unified generative interface.

![](images/1937519d59560dba7b72c95e5faa2595dd5ab8804bf46c5dfa7861382230ad6f.jpg)  
Figure 7. Universal signal encoder architecture. Raw multi-channel brain signals are first Z-score normalized and transformed into log-scaled spectrograms using STFT. A convolutional patch-embedding module converts each time-frequency map into per-channel token sequences. Cross-channel interactions are modeled using multi-head self-attention with QK-Norm for stability and Group Query Attention for parameter eficiency. Residual fusion and global average pooling yield a compact signal-token sequence that serves as a unified representation.

## 4.2.2 | Transformer Backbone With Group Query Attention and Mixture-of-Experts

The concatenated sequence is processed by a stack of Transformer decoder blocks. Each block consists of Group Query Attention (GQA) and a Mixture-of-Experts (MoE) feedforward network. GQA decouples the number of query heads $h _ { q }$ and key-value heads $h _ { k }$ , enabling more flexible capacity allocation. For an input tensor $\boldsymbol { Z } \in \mathbb { R } ^ { L \times D }$ , the projected queries, keys, and values are computed as:

$$
Q = Z W _ { Q } , \quad K = Z W _ { K } , \quad V = Z W _ { V }
$$

The key and value tensors are repeated to match the number of query heads, followed by scaled dot-product attention:

Attention

$$
( Q , K ^ { \prime } , V ^ { \prime } ) = \mathrm { S o f t M a x } \left( \frac { Q K ^ { \prime T } } { \sqrt { d _ { h } } } \right) V ^ { \prime }
$$

To increase model capacity without linearly increasing the parameter count, we adopt a token-level sparse Mixture-of-Experts (MoE) architecture in most feed-forward layers. For each token representation $\boldsymbol { x } \in \mathbb { R } ^ { D }$ , a router computes gating logits $g ( x ) = W _ { g } x$ , where $W _ { g } \in \mathbb { R } ^ { \mathrm { N _ { e } } \times D }$ and $N _ { e }$ denotes the number of experts. A learnable bias vector $b \in \mathbb { R } ^ { \mathrm { N _ { e } } }$ is added to modulate expert utilization and encourage balanced routing, yielding $\widetilde { g } ( x ) = g ( x ) + b$ . The routing probabilities are then obtained via:

$$
p ( x ) = \mathrm { s o f t m a x } ( \widetilde { g } ( x ) )
$$

For computational eficiency, only the top-k experts with the highest gate probabilities are activated. Let $\mathcal { T } _ { k } ( x )$ denote the index set of the top-k experts, and define a routing mask $m _ { j } ( x ) \in \{ 0 , 1 \}$ . The MoE output is given by:

$$
y ( x ) = \sum _ { j = 1 } ^ { N _ { e } } { m _ { j } ( x ) \cdot p _ { j } ( x ) \cdot E _ { j } ( x ) } + E _ { s } ( x )
$$

Where $E _ { j } ( \cdot )$ represents the j-th expert network and $E _ { s } ( \cdot )$ is a shared expert capturing task-agnostic patterns.

To enforce balanced expert utilization, we introduce an auxiliary load-balancing loss:

$$
\mathcal { L } _ { \mathrm { b a l a n c e } } = \sum _ { e = 1 } ^ { N _ { e } } p _ { e } \cdot q _ { e }
$$

Here $p _ { e }$ is the fraction of tokens routed to expert $^ { e , }$ and $q _ { e }$ is the average gate probability for that expert.

## 4.2.3 | Multimodal Attention With Hybrid Masking

We couple brain signal tokens and language tokens using a hybrid attention mask that controls directional dependencies within and across modalities. Let $N _ { s }$ and $N _ { t }$ be the numbers of signal and text tokens. The mask $M \in \mathbb { R } ^ { ( N _ { s } + N _ { t } ) \times ( N _ { s } + N _ { t } ) }$ is defined so that signal to signal is bidirectional, text to text is causal, and text to signal is fully visible:

$$
M _ { i j } = \left\{ \begin{array} { l l } { 0 , } & { i , j \le N _ { s } , } \\ { 0 , } & { i > N _ { s } \mathrm { ~ a n d ~ } j \le i , } \\ { - \infty , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.
$$

The mask is injected additively into scaled dot-product attention:

$$
\mathrm { A t t e n t i o n } ( \mathcal { Q } , K ^ { \prime } , V ^ { \prime } ) = \mathrm { S o f t M a x } \left( \frac { \mathcal { Q } K ^ { \prime T } } { \sqrt { d _ { h } } } + M \right) V ^ { \prime }
$$

which preserves full bidirectional interactions among signal tokens, enforces causal decoding for text tokens, and grants text tokens access to the entire signal sequence.

## 4.2.4 | Modality-Aware Rotary Position Encoding

We use rotary position embeddings[61] to inject relative positional information into queries and keys so that attention depends on token distances rather than absolute indices. This is suited to variable-length signal sequences and stabilizes autoregressive decoding when text is appended after the signal. Given a token vector $x _ { m }$ at position $m ,$ ���� applies a blockwise rotation to queries and keys:

$$
\begin{array} { r l } { \mathrm { R o P E } ( x _ { m } , m ) = x _ { m } \odot \cos ( m \theta ) } & { } \\ { + \mathcal { R } ( x _ { m } ) \odot \sin ( m \theta ) , } & { } \end{array}
$$

where ⊙ denotes elementwise multiplication and $\mathcal { R } ( \cdot )$ swaps the two halves of the vector with a sign flip on the second half. The angular frequencies are defined as $\theta _ { i } = \Theta ^ { \frac { - 2 i } { D } }$ , where � is the feature dimension. We assign modality-specific base frequencies: $\Theta _ { s i g n a l } = 1 0 ^ { 4 }$ for signal tokens and $\Theta _ { t e x t } = 1 0 ^ { 5 }$ for text tokens. Using two bases encodes modality identity directly in the rotational phase and introduces a controlled frequency jump at the signal-text boundary, which discourages spurious adjacency between the last signal token and the first text token, reduces cross-modal attention leakage, and improves alignment of generated language with neural evidence under the hybrid mask.

Table 1. Pretraining datasets and corpus statistics.
<table><tr><td>Dataset</td><td>Data type</td><td>Subjects</td><td>Samples</td><td>Duration (h)</td></tr><tr><td>HUP[62]</td><td>iEEG</td><td></td><td>573,322,242</td><td>2768.54</td></tr><tr><td>UPenn[63]</td><td>iEEG</td><td></td><td>121,691,209</td><td>469.78</td></tr><tr><td>SWEC_ETHZ[64]</td><td>iEEG</td><td></td><td>161,004,999</td><td>837.50</td></tr><tr><td>FNUSA[65]</td><td>iEEG</td><td>12</td><td>147,030</td><td>122.53</td></tr><tr><td>SHHS[66]</td><td>EEG</td><td></td><td>6441 5,789,08848242.40</td><td></td></tr><tr><td>SeiZelT2[67]</td><td>EEG</td><td></td><td>1254,519,290</td><td>12553.58</td></tr><tr><td>TUSZ[68]</td><td>EEG</td><td>675</td><td>687,375</td><td>1909.38</td></tr><tr><td>TUAB[69]</td><td>EEG</td><td>2329</td><td>409,455</td><td>1137.38</td></tr><tr><td>CHB MIT[70]</td><td>EEG</td><td>23</td><td>384,834</td><td>1068.98</td></tr><tr><td>TUEP[71]</td><td>EEG</td><td>179</td><td>214,376</td><td>595.49</td></tr><tr><td>SleepEDF[72]</td><td>EEG</td><td>78</td><td>195,478</td><td>1628.98</td></tr><tr><td>HaaglandenSleep[73] EEG</td><td></td><td>151</td><td>137,244</td><td>1143.70</td></tr><tr><td>TDBrain[74]</td><td>EEG</td><td>911</td><td>115,668</td><td>64.26</td></tr><tr><td>TUEV[75]</td><td>EEG</td><td>370</td><td>111,547</td><td>154.93</td></tr><tr><td>ADFTD[76]</td><td>EEG</td><td>88</td><td>34,876</td><td>19.38</td></tr><tr><td>BrainLat[77]</td><td>EEG</td><td>135</td><td>30,699</td><td>17.06</td></tr><tr><td>AD-Auditory[78]</td><td>EEG</td><td>35</td><td>17,757</td><td>9.87</td></tr><tr><td>ShuMI[79]</td><td>EEG</td><td>25</td><td>11,988</td><td>13.32</td></tr><tr><td>REEG-PD[80]</td><td>EEG</td><td>149</td><td>11,878</td><td>6.60</td></tr><tr><td>PhysionetMI[81]</td><td>EEG</td><td>109</td><td>9,527</td><td>7.94</td></tr></table>

## 4.3 | Pretraining Corpus Curation and Instruction-Answer Construction

To train METIS with instruction-driven supervision at scale, we curated a large corpus from 20 EEG and iEEG datasets that span sleep stage classification, epilepsy and interictal epileptiform discharge detection, neurodegenerative and movement disorders, attention-deficit or hyperactivity disorder, mood and psychotic disorders, signal anomaly detection, and motor imagery. The datasets difer in channel configurations, sampling frequencies, subject counts, and annotation conventions. We standardized preprocessing per dataset while respecting its native acquisition and labeling protocols. Signals were minimally cleaned according to source recommendations, and each training instance was paired with a standardized naturallanguage instruction and a canonical answer string to build instruction-answer pairs for autoregressive pretraining.

We use two QA formats that share the same label sets. In the open ended format the model receives a fixed instruction and generates one canonical answer string drawn from a discrete set. In the multiple-choice format the same candidate set is embedded in the prompt as lettered options and the model selects the corresponding letter. Each dataset is paired with a single instruction sentence and a fixed answer set. No other template families are used. Table S1 lists the instruction and answer set for every dataset. For TDBrain the full label inventory is provided in Table S2, and each composite label is treated as a single canonical class string.

## 4.4 | Evaluation Protocols

## 4.4.1 | Zero-Shot Classification

METIS is uniquely capable of performing zero-shot inference without any task-specific fine-tuning. To evaluate this ability, we reformulate classification problems as multiple-choice question answering tasks. Each instance is presented to the model as a combination of a brain signal segment and a textual instruction prompt. The model processes the concatenated signal and instruction tokens and autoregressively predicts the next token distribution over the vocabulary. As illustrated in Figure 2a, we do not decode the full output sequence; instead, we directly extract the logits corresponding to the candidate answer tokens.

Let V denote the vocabulary, and let $C = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { K } \} \subset$ V be the set of candidate answer tokens (e.g., $\begin{array} { r l } { C } & { { } = } \end{array}$ $\{ A , B , C , D , E \}$ for a five-way classification task). Given input sequence $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { T } \}$ comprising the signal tokens and the instruction tokens, the model predicts a distribution $P ( \cdot | X ) \in \mathbb { R } ^ { | \mathcal { V } | }$ over the vocabulary for the next token. We select the logits $z _ { c _ { 1 } } , z _ { c _ { 2 } } , . . . , z _ { c _ { K } }$ corresponding to the candidate answers and compute a softmax over this subset:

$$
P ( c _ { i } \mid X ) = { \frac { \exp ( z _ { c _ { i } } ) } { \sum _ { j = 1 } ^ { K } \exp ( z _ { c _ { j } } ) } } , \quad i = 1 , 2 , \ldots , K
$$

The predicted label is given by the candidate token with the highest probability. This targeted decoding approach avoids irrelevant tokens and enables eficient classification by leveraging the model’s language-semantic alignment. This zero-shot framework is applied uniformly across all downstream classification tasks, allowing direct comparison of generalization ability across domains.

## 4.4.2 | Few-Shot Classification

To assess the label eficiency and adaptability of METIS under limited supervision, we evaluate few-shot classification performance across a range of low-data regimes. Specifically, we consider four settings with � = 1, 2, 4, 8 labeled examples per class.

All few-shot experiments are conducted exclusively on pretrained models, including METIS. We adopt a linear probing protocol, where the backbone encoder remains frozen, and only a task-specific linear classification head is trained using the few-shot training samples. This evaluation strategy focuses on the generalization capacity of the pretrained representations under minimal supervision, allowing us to assess the label eficiency and transferability of learned features across diverse downstream tasks.

To ensure consistency across datasets, we follow two evaluation strategies depending on the presence of predefined splits. For datasets with oficial train/test splits, we randomly sample � examples per class from the training set using five independent random seeds, and report the mean and standard deviation of performance on the held-out test set. For datasets without such splits, we perform 5-fold subject-wise cross-validation, where four folds are used for few-shot training and the remaining fold for evaluation, reporting the averaged performance across folds. In all cases, no subjects or recordings are shared across the training, validation, and test partitions; for datasets with oficial splits, we strictly follow the oficial partition protocols.

This evaluation framework allows us to compare models in a label-constrained regime while accounting for inter-subject variability and dataset-specific constraints.

## 4.4.3 | Cross-Dataset Transfer

To evaluate the generalization ability of each model across diferent datasets, we design cross-dataset transfer experiments where the model is trained on one dataset (the source) and evaluated on a distinct dataset (the target). Specifically, all available data from the source dataset are used for training, and the target dataset is used exclusively for testing.

We distinguish two cases based on whether the target dataset comes with predefined train/test splits. If no such split is provided, we perform 5-fold subject-wise cross-validation on the target dataset. In each fold, one partition is used as the test set while the remaining four serve as validation-only references (since training occurs solely on the source data). The final results are reported as the average and standard deviation across all five folds. If the target dataset includes predefined splits, we use the oficial test set directly and conduct five independent transfer experiments using diferent random seeds, again reporting the mean and standard deviation.

For models trained from scratch, all parameters are finetuned on the source dataset before testing on the target. In contrast, all pretrained models, including METIS, are evaluated using a linear probing protocol. In this setting, the backbone encoder remains frozen during training on the source data, and only a task-specific linear classifier is trained. This approach isolates the quality of the learned representations and ofers a more rigorous assessment of cross-domain generalization.

## 4.5 | Downstream Datasets

To evaluate generalization across heterogeneous clinical contexts, we used 17 datasets spanning sleep stage classification, epilepsy and interictal epileptiform discharge detection, neurodegenerative and movement disorders, ADHD, mood and psychotic disorders, signal anomaly detection, and anesthesia depth monitoring. The datasets vary widely in channel configuration, sampling frequency, cohort size, and segmentation protocol, creating a rigorous test bed for robustness.

ISRUC[41]: This sleep stage classification dataset contains overnight scalp EEG from 100 participants sampled at 200 Hz. We followed a two-channel montage (C3-M2 and C4-M1) and segmented recordings into 30-second epochs labeled into the five standard sleep stages, yielding 73,883 labeled segments.

Dreams[42]: This sleep stage classification dataset contains overnight EEG from 20 participants sampled at 200 Hz. We used a three-channel montage (FP1-A2, FP2-A1, CZ-A1) and segmented recordings into 30-second epochs labeled into the five standard sleep stages, yielding 20,242 labeled segments.

SEE[45]: This epilepsy dataset ofers single-channel EEG from 30 subjects at 173.61 Hz with a fixed author-defined split. Signals were cut into 4-second windows and labeled as seizure or non-seizure, giving 4,000 segments in total.

Siena[82]: Multichannel scalp EEG from 14 subjects at 256 Hz is annotated into four peri-ictal classes (pre-ictal, ictal, post-ictal, inter-ictal). Using 19 channels and 10-second windows, preprocessing yielded 50,749 segments for evaluating seizure-related dynamics.

RatEpilepsy[44]: This intracranial EEG dataset comprises three-channel recordings at 6,000 Hz from five rats. Signals were segmented into 3-second windows labeled as seizure or interictal, yielding 2,530 samples and enabling a cross-species assessment of generalization.

IEDS[57]: An iEEG dataset focused on interictal epileptiform discharges, IEDS includes recordings from 25 subjects at 1,000 Hz. After 3-second segmentation, windows were labeled as normal activity or IED, producing 59,615 samples.

Mayo[43]: Intracranial EEG from 15 subjects at 500 Hz originally includes four labels. We retained physiological and pathological activity and removed artifacts and power-line noise to form a binary IED detection task. With 3-second windows, the dataset contains 71,957 labeled segments.

ADFSU[49]: This Alzheimer’s disease dataset comprises nineteen-channel EEG at 128 Hz from 92 participants (80 patients and 12 controls). We created 2-second segments labeled as Alzheimer’s disease or control, for a total of 736 samples.

APAVA[56]: A complementary Alzheimer’s dataset with sixteen-channel EEG at 256 Hz from 23 participants (12 patients and 11 controls). Using 2-second windows, preprocessing yielded 2,652 labeled segments.

SanDiego[83]: For Parkinson’s disease detection, SanDiego provides thirty-two-channel EEG at 512 Hz from 31 participants (15 patients and 16 controls). We used 2-second segments to obtain 4,521 samples for binary classification.

ADHD-121[46]: Nineteen-channel EEG at 128 Hz from 121 children supports ADHD detection. With 5-second windows and binary labels (ADHD or control), the dataset contains 3,322 segments.

ADHD-80[47]: This companion ADHD dataset includes two-channel EEG at 256 Hz from 80 participants (38 ADHD, 42 controls). Using 5-second segments, we obtained 5,056 labeled windows.

MDD[52]: For major depressive disorder, this dataset contains nineteen-channel EEG at 256 Hz from 61 participants (33 MDD, 28 controls). Segmentation into 5-second windows produced 7,600 labeled samples.

Schizo-Youth[48]: Sixteen-channel EEG at 128 Hz from 84 participants (45 with schizophrenia, 39 controls) was segmented into 10-second windows, yielding 504 samples for schizophrenia detection.

Schizo-28[84]: Nineteen-channel EEG at 250 Hz from 28 participants evenly split between patients and controls. With 10-second segments, this dataset provides 2,878 samples.

NMT[50]: A large-scale EEG anomaly detection benchmark with sixteen channels at 200 Hz. Following the oficial split and using 10-second windows, we formed 175,091 segments labeled as normal or abnormal to capture broad anomaly patterns.

NTUHBIS[54]: This anesthesia depth monitoring dataset contains single-channel EEG at 128 Hz from 23 surgical patients. Labels are derived from the bispectral index and grouped into four levels (deep hypnotic state, general anesthesia, moderate sedation, awake/light sedation). We followed the dataset’s segmentation protocol to obtain 5,324 labeled samples.

Across all datasets, we adhered to the original label definitions and harmonized nomenclature where needed. For instruction-driven evaluation, dataset-specific class taxonomies were mapped to consistent prompts while preserving the semantics of each task. This collection spans substantial variability in acquisition protocol, electrode montage, sampling rate, species, and cohort composition, providing a comprehensive basis for assessing task generalization and robustness under distribution shift.

## 4.6 | Evaluation Metrics

The primary outcome metric is the area under the receiver operating characteristic curve (AUROC). AUROC is threshold free and less sensitive to class imbalance than accuracy, which

is appropriate given the range of class prevalences across the tasks evaluated.

## 4.6.1 | Binary Tasks

For two class problems we compute the standard AUROC from the model scores of the positive class. The ROC curve is obtained by varying a threshold on these scores, and the AUROC equals the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample.

## 4.6.2 | Multiclass Tasks

For problems with K>2 classes we report the macro-averaged one-versus-one AUROC. Concretely, we compute AUROC for every unordered pair of classes and then take the unweighted average across all pairs:

$$
\mathrm { A U R O C } _ { \mathrm { m a c r o - o v o } } = { \frac { 2 } { K ( K - 1 ) } } \sum _ { \substack { 1 \le i < j \le K } } \mathrm { A U R O C } ( i \ \mathrm { v s } \ j )
$$

## 4.6.3 | Multiple-Choice Question Answering

For the multiple-choice question answering protocol, each instance provides a brain signal segment and a prompt that enumerates a finite set of candidate answers. The number of candidates varies by dataset and task. For METIS, we obtain a probability for each candidate using the targeted logits described in Section 4.4 and select the candidate with the highest probability. For general-purpose multimodal foundation models, we process input signals into 224×224 images, a widely-used input size in vision-language research[30, 33, 85]. This image-based input, in contrast to high-dimensional raw signal points, provides a more condensed and feature-rich representation, which we found to be more efective in preliminary studies for capturing the signal’s inherent characteristics. We then constrain and parse outputs using a two-stage rule aligned with our implementation. We first request a JSON object validated against a schema whose single field is an uppercase letter drawn from the set of valid option letters. If schema based decoding is unavailable or fails, we fall back to plain text generation and extract the first valid uppercase letter from the allowed set present in the output. Predictions that do not yield a valid option letter are treated as incorrect. Accuracy is the proportion of instances for which the mapped prediction equals the ground truth label. Results are reported per dataset together with an overall mean across datasets.

## 4.6.4 | Open Ended Question Answering

In the open ended protocol, the model generates a short textual answer that should match a canonical reference string for the same instance. Answer quality is quantified with BERTScore, which measures token level semantic similarity using contextual embeddings. Precision is the average over candidate tokens of the maximum similarity to any reference token. Recall is the average over reference tokens of the maximum similarity to any candidate token. We report the F1 combination of precision and recall as the main score. Following common practice, reference and candidate strings are lowercased and stripped of punctuation prior to scoring, and we use the default English configuration provided by the oficial BERTScore package.

## 4.7 | Implementation Details

All data preparation, pretraining, and downstream evaluations were conducted on the same Linux server to ensure consistency and reproducibility. The machine uses an Intel Xeon Gold 6342 CPU and a single NVIDIA A100 80 GB GPU. The software stack includes Python 3.11.7, PyTorch 2.0.1 with CUDA 12.2, NumPy 1.26.3, SciPy 1.10.1, Transformers 4.41.2, einops 0.8.0, and pyhealth 1.1.4.

## 4.7.1 | Pretraining

Training employed mixed precision in PyTorch. We used the AdamW optimizer with $\beta = ( 0 . 9 , 0 . 9 9 9 )$ , weight decay 0.1, and a base learning rate of 2e-4. The schedule included a warm-up phase covering 10% of the total pretraining steps. The per-step batch size was 256 with gradient accumulation of 20 steps, giving an efective batch size of 5,120 samples per optimizer update.

## 4.7.2 | Downstream Evaluation

Few-shot experiments and cross-dataset transfer were conducted in full precision to ensure numerical stability. All experiments were trained for 200 epochs, allowing suficient optimization steps for the models to converge. We used AdamW with $\beta = ( 0 . 9 , 0 . 9 9 9 )$ ), weight decay 0.1, and a base learning rate of 2e-4, and a batch size of 128. Zero-shot evaluation used the pretrained checkpoint without any fine-tuning, applying the same inference stack across datasets.

## Author Contributions

This paper was primarily written by Mingzhi Chen and Yiyu Gui. Mingzhi Chen conceived the study, designed the methodology, and conducted all experiments. Mingzhi Chen and Yiyu Gui performed dataset curation and construction. Yiyu Gui contributed to manuscript refinement and figure preparation. Guibo Luo and Yuchao Yang supervised the research and provided critical guidance throughout the project.

## Acknowledgments

This work has been supported by the National Key R&D Program of China (2025YFB4507300), Guangdong S&T Program (2025B0101140001, 2026B0101070006), Guangdong Provincial Key Laboratory of In-Memory Computing Chips (2024B1212020002), Shenzhen Science and Technology Program (ZDCY20250901103401002, JCYJ20241202125907011), and Beijing Natural Science Foundation (L234026, L257010). This work has been supported by the New Cornerstone Science Foundation and Financial Support for Outstanding Scientific and Technological Innovation Talents Training Fund in Shenzhen.

## Use of AI Tools

The web version of ChatGPT (OpenAI; model: GPT-5) was used solely for English language editing and improving clarity of phrasing. The

authors reviewed and edited all AI-assisted text and take full responsibility for the final content.

## Funding

This work was supported by the Shenzhen Science and Technology Program (JCYJ20241202125907011).

## Conflicts of Interest

The authors declare no conflicts of interest.

## Data Availability Statement

The source code, model implementation, and key scripts are available at https://github.com/mingzhi-c/metis-brain-signal-foundation-model.

## References

1. J. Parvizi and S. Kastner, “Promises and Limitations of Human Intracranial Electroencephalography,” Nature Neuroscience 21, no. 4 (2018): 474–483.

2. Y. Zhang and Z. S. Chen, “Harnessing Electroencephalography Connectomes for Cognitive and Clinical Neuroscience,” Nature Biomedical Engineering 9, no. 8 (2025): 1186–1201, https://doi.org/10.1038/s41551 -025-01442-4.

3. H. Zhang, Q.-Q. Zhou, H. Chen, et al., “The Applied Principles of EEG Analysis Methods in Neuroscience and Clinical Neurology,” Military Medical Research 10, no. 1 (2023): 67.

4. B. Frauscher, D. Mansilla, C. Abdallah, et al., “Learn How to Interpret and use Intracranial EEG Findings,” Epileptic Disorders 26, no. 1 (2024): 1–59.

5. P. Liu, W. Qian, H. Zhang, et al., “Automatic Sleep Stage Classification Using Deep Learning: Signals, Data Representation, and Neural Networks,” Artificial Intelligence Review 57, no. 11 (2024): 301.

6. N. Sinha, J. S. Duncan, B. Diehl, et al., “Intracranial EEG Structure-Function Coupling and Seizure Outcomes after Epilepsy Surgery,” Neurology 101, no. 13 (2023): e1293–e1306.

7. C. Formica, E. Gjonaj, L. Bonanno, et al., “The Role of High-Density EEG in Diagnosis and Prognosis of Neurological Diseases: A Systematic Review,” Clinical Neurophysiology 174 (2025): 37–47, https://doi.org/10.1016/j.clinph.2025.03.026.

8. J. J. Newson and T. C. Thiagarajan, “EEG Frequency Bands in Psychiatric Disorders: A Review of Resting State Studies,” Frontiers in Human Neuroscience 12 (2019): 521.

9. D. A. Engemann, F. Raimondo, J.-R. King, et al., “Robust EEG-Based Cross-Site and Cross-Protocol Classification of States of Consciousness,” Brain 141, no. 11 (2018): 3179–3192.

10. P. Prado, A. Birba, J. Cruzat, et al., “Dementia ConnEEGtome: Towards Multicentric Harmonization of EEG Connectivity in Neurodegeneration,” International Journal ofPsychophysiology 172 (2022): 24–38.

11. V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and B. J. Lance, “EEGNet: A Compact Convolutional Neural Network for EEG-Based Brain–computer Interfaces,” Journal ofNeural Engineering 15, no. 5 (2018): 056013.

12. Y. Song, Q. Zheng, B. Liu, and X. Gao, “EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization,” IEEE Transactions on Neural Systems and Rehabilitation Engineering 31 (2023): 710–719, https://doi.org/10.1109/TNSRE.2022.3230250.

13. J. Jing, W. Ge, S. Hong, et al., “Development of Expert-Level Classification of Seizures and Rhythmic and Periodic Patterns during EEG Interpretation,” Neurology 100, no. 17 (2023): e1750–e1762.

14. E. Eldele, Z. Chen, C. Liu, et al., “An Attention-Based Deep Learning Approach for Sleep Stage Classification with Single-Channel EEG,” IEEE Transactions on Neural Systems and Rehabilitation Engineering 29 (2021): 809–818.

15. H. Phan, F. Andreotti, N. Cooray, O. Y. Chén, and M. De Vos, “SeqSleepNet: End-to-End Hierarchical Recurrent Neural Network for Sequence-to-Sequence Automatic Sleep Staging,” IEEE Transactions on Neural Systems and Rehabilitation Engineering 27, no. 3 (2019): 400–410.

16. C. Yang, M. Westover, and J. Sun, “Biot: Biosignal Transformer for Cross-Data Learning in the Wild,” Advances in Neural Information Processing Systems 36 (2023): 78240–78260.

17. W. Jiang, L. Zhao, and B. Lu, “Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI,” in The Twelfth International Conference on Learning Representations (OpenReview.net, 2024).

18. J. Wang, S. Zhao, Z. Luo, et al., “CBraMod: A Criss-Cross Brain Foundation Model for EEG Decoding,” in The Thirteenth International Conference on Learning Representations (OpenReview.net, 2025).

19. D. Zhang, Z. Yuan, Y. Yang, J. Chen, J. Wang, and Y. Li, “Brant: Foundation Model for Intracranial Neural Signal,” Advances in Neural Information Processing Systems 36 (2023): 26304–26321.

20. C. Wang, V. Subramaniam, A. Yaari, et al., “BrainBERT: Self-Supervised Representation Learning for Intracranial Electrodes,” in International Conference on Learning Representations (ICLR, 2023).

21. J. Achiam, S. Adler, S. Agarwal, et al., “Gpt-4 Technical Report,” arXiv preprint arXiv:2303.08774 (2023).

22. Gemini Team, R. Anil, S. Borgeaud, et al., “Gemini: A Family of Highly Capable Multimodal Models,” arXiv preprint arXiv:2312.11805 (2023), https://doi.org/10.48550/arXiv.2312.11805.

23. “xAI, Grok 4,” (accessed 10, December 2025), (2025), https://x.ai/news/ grok-4.

24. Z. Wu, X. Chen, Z. Pan, et al., “DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding,” Preprint arXiv:2412.10302 (2024), https://doi.org/10.48550/arXiv.2412. 10302.

25. K. He, X. Chen, S. Xie, Y. Li, P. Dollár, and R. Girshick, “Masked Autoencoders Are Scalable Vision Learners,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (IEEE, 2022), 16000–16009.

26. T. J. Bradshaw, X. Tie, J. Warner, J. Hu, Q. Li, and X. Li, “Large Language Models and Large Multimodal Models in Medical Imaging: A Primer for Physicians,” Journal ofNuclear Medicine 66, no. 2 (2025): 173–182.

27. R. AlSaad, A. Abd-Alrazaq, S. Boughorbel, et al., “Multimodal Large Language Models in Health Care: Applications, Challenges, and Future Outlook,” Journal ofMedical Internet Research 26 (2024): e59505.

28. J. Brickman, M. Gupta, and J. R. Oltmanns, “Large Language Models for Psychological Assessment: A Comprehensive Overview,” Advances in Methods and Practices in Psychological Science 8, no. 3 (2025): 25152459251343582.

29. A. Sohail and L. Zhang, “Using Large Language Models to Facilitate Academic Work in the Psychological Sciences,” Current Psychology 44, no. 9 (2025): 7910–7918.

30. A. Radford, J. W. Kim, C. Hallacy, et al., “Learning Transferable Visual Models from Natural Language Supervision,” in International Conference on Machine Learning (PmLR, 2021), 8748–8763.

31. H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual Instruction Tuning,” Advances in Neural Information Processing Systems 36 (2023): 34892–34916.

32. J.-B. Alayrac, J. Donahue, P. Luc, et al., “Flamingo: A Visual Language Model for Few-Shot Learning,” Advances in Neural Information Processing Systems 35 (2022): 23716–23736.

33. J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping Language-Image Pre-Training with Frozen Image Encoders and Large Language Models,” in International Conference on Machine Learning (PMLR, 2023), 19730–19742.

34. A. Vaswani, N. Shazeer, N. Parmar, et al., “Attention Is All You Need,” in Advances in Neural Information Processing Systems 30 (Curran Associates, Inc, 2017).

35. N. Shazeer, A. Mirhoseini, K. Maziarz, et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer,” in International Conference on Learning Representations (2017).

36. Y. Zhou, T. Lei, H. Liu, et al., “Mixture-of-Experts with Expert Choice Routing,” Advances in Neural Information Processing Systems 35 (2022): 7103–7114.

37. D. Dai, C. Deng, C. Zhao, et al., “DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (Association for Computational Linguistics, 2024), 1280–1297.

38. D. Lepikhin, H. Lee, Y. Xu, et al., “GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding,” in International Conference on Learning Representations (OpenReview.net, 2021).

39. R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” in Proceedings of the IEEE International Conference on Computer Vision (IEEE, 2017), 618–626.

40. A. Rechtschafen and A. Kales, eds., A Manual of Standardized Terminology, Techniques and Scoring System for Sleep Stages of Human Subjects (U.S. Department of Health, Education, and Welfare, Public Health Service, National Institutes of Health, 1968).

41. S. Khalighi, T. Sousa, J. M. Santos, and U. Nunes, “ISRUC-Sleep: A Comprehensive Public Dataset for Sleep Researchers,” Computer Methods and Programs in Biomedicine 124 (2016): 180–192.

42. S. Devuyst, M. Kerkhofs, and T. Dutoit, The DREAMS Databases and Assessment Algorithm (Zenodo, 2019), https://doi.org/10.5281/zenodo.2 650142.

43. G. Worrell, B. H. Brinkmann, V. Kremen, et al., “BIDS\_MAYO,” (2020), https://doi.org/10.6084/m9.figshare.12192405.v1.

44. A. Ghanaei and A. Erfanian, Intracranial EEG (iEEG) Recording in a Rat Model ofEpilepsy (Mendeley Data, 2025), https://doi.org/10.17632/k n3k9f5vph.2.

45. S. Panwar, Single Electrode EEG Data ofHealthy and Epileptic Patients [Data set] (Zenodo, 2020), https://doi.org/10.5281/zenodo.3684992.

46. A. M. Nasrabadi, A. Allahverdy, M. Samavati, and M. R. Mohammadi, EEG Datafor ADHD / Control Children (IEEE Dataport, 2020), https://doi.org/10.21227/rzfh-zn36.

47. G. S. Bajestani, S. Abedian, F. Makhloughi, M. Raoufitabar, and H. Saeedi, A Dataset of EEG Signals from Adults with ADHD and Healthy Controls: Resting State, Cognitive Function, and Sound Listening Paradigm (Mendeley Data, 2023), https://doi.org/10.17632/6k4g25fhzg.1.

48. N. N. Gorbachevskaya and S. V. Borisov, EEG ofHealthy Adolescents and Adolescents with Symptoms OfSchizophrenia (M.V. Lomonosov Moscow State University, 2019), http://brain.bio.msu.ru/eeg\_schizophrenia.htm.

49. A.-K. Kiessner, R. T. Schirrmeister, L. A. Gemein, J. Boedecker, and T. Ball, “An Extended Clinical EEG Dataset with 15,300 Automatically Labelled Recordings for Pathology Decoding,” NeuroImage: Clinical 39 (2023): 103482.

50. H. A. Khan, R. Ul Ain, A. M. Kamboh, et al., “The NMT Scalp EEG Dataset: An Open-Source Annotated Dataset of Healthy and Pathological EEG Recordings for Predictive Modeling,” Frontiers in Neuroscience 15 (2022): 755817.

51. T. Zhang, V. Kishore, F. Wu, K. Q. Weinberger, and Y. Artzi, “BERTScore: Evaluating Text Generation with BERT,” in International Conference on Learning Representations (OpenReview.net, 2020).

52. W. Mumtaz, “MDD Patients and Healthy Controls EEG Data (New),” (2016), https://doi.org/10.6084/m9.figshare.4244171.v2.

53. T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A Simple Framework for Contrastive Learning of Visual Representations,” in International Conference on Machine Learning (PmLR, 2020), 1597–1607.

54. L. Ma, “EEG and BIS Raw Data,” (2017), https://doi.org/10.6084/m9.fig share.5589841.v1.

55. L. van der Maaten and G. Hinton, “Visualizing Data Using t-SNE,” Journal ofMachine Learning Research 9, no. 86 (2008): 2579–2605.

56. M. L. Vicchietti, F. M. Ramos, L. E. Betting, and A. S. Campanharo, “Computational Methods of EEG Signals Analysis for Alzheimer’s Disease Classification,” Scientific Reports 13, no. 1 (2023): 8184.

57. R. Falach, M. Geva-Sagiv, D. Eliashiv, et al., “Annotated Interictal Epileptiform Discharges in Intracranial EEG (iEEG) Sleep Data,” (2024), https://doi.org/10.6084/m9.figshare.26131978.v3.

58. G. Hinton, “Distilling the Knowledge in a Neural Network,” in Deep Learning and Representation Learning Workshop in Conjunction with NIPS (Neural Information Processing Systems Foundation, 2015).

59. A. Yang, B. Yang, B. Zhang, et al., “Qwen2.5 Technical Report,” Preprint arXiv:2412.15115 (2024), https://doi.org/10.48550/arXiv.2412.15115.

60. J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai, “GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics, 2023), 4895–4901.

61. J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “Roformer: Enhanced Transformer with Rotary Position Embedding,” Neurocomputing 568 (2024): 127063.

62. J. M. Bernabei, A. Li, A. Y. Revell, et al., HUP iEEG Epilepsy Dataset (OpenNeuro, 2023), https://doi.org/10.18112/openneuro.ds004100.v1.1 .3.

63. bbrinkm, sbaldassano, and W. Cukierski, UPenn and Mayo Clinic’s Seizure Detection Challenge (2014), https://kaggle.com/competitions/se izure-detection.

64. A. Burrello, L. Cavigelli, K. Schindler, L. Benini, and A. Rahimi, “Laelaps: An Energy-Eficient Seizure Detection Algorithm from Long-Term Human iEEG Recordings Without False Alarms,” in 2019 Design, Automation & Test in Europe Conference & Exhibition (DATE) (IEEE, 2019), 752–757.

65. G. Worrell, B. H. Brinkmann, V. Kremen, et al., “Dataset\_Fnusa,” (2020), https://doi.org/10.6084/m9.figshare.11734578.

66. G.-Q. Zhang, L. Cui, R. Mueller, et al., “The National Sleep Research Resource: Towards a Sleep Data Commons,” Journal ofthe American Medical Informatics Association 25, no. 10 (2018): 1351–1358.

67. M. Bhagubai, C. Chatzichristos, L. Swinnen, et al., SeizeIT2 (OpenNeuro, 2025), https://doi.org/10.18112/openneuro.ds005873.v1.1.0.

68. V. Shah, E. Von Weltin, S. Lopez, et al., “The Temple University Hospital Seizure Detection Corpus,” Frontiers in Neuroinformatics 12 (2018): 83.

69. S. Lopez, G. Suarez, D. Jungreis, I. Obeid, and J. Picone, “Automated Identification of Abnormal Adult EEGs,” in 2015 IEEE signal processing in medicine and biology symposium (SPMB) (IEEE, 2015), 1–5.

70. A. H. Shoeb, “Application of Machine Learning to Epileptic Seizure Onset Detection and Treatment,” (PhD diss., Massachusetts Institute of Technology, 2009).

71. L. Veloso, J. McHugh, E. von Weltin, S. Lopez, I. Obeid, and J. Picone, “Big Data Resources for EEGs: Enabling Deep Learning Research,” in 2017 IEEE signal processing in medicine and biology symposium (SPMB) (IEEE, 2017), 1–3.

72. B. Kemp, A. H. Zwinderman, B. Tuk, H. A. Kamphuisen, and J. J. Oberye, “Analysis of a Sleep-Dependent Neuronal Feedback Loop: The Slow-Wave Microcontinuity of the EEG,” IEEE Transactions on Biomedical Engineering 47, no. 9 (2000): 1185–1194.

73. D. Alvarez-Estevez and R. Rijsman, Haaglanden Medisch Centrum Sleep Staging Database (version 1.1) (PhysioNet, 2022), https://doi.org/10.130 26/t79q-fr32.

74. H. Van Dijk, G. Van Wingen, D. Denys, S. Olbrich, R. Van Ruth, and M. Arns, “The Two Decades Brainclinics Research Archive for Insights in Neurophysiology (TDBRAIN) Database,” Scientific Data 9, no. 1 (2022): 333, https://doi.org/10.1038/s41597-022-01409-z.

75. A. Harati, M. Golmohammadi, S. Lopez, I. Obeid, and J. Picone, “Improved EEG Event Classification Using Diferential Energy,” in 2015 IEEE Signal Processing in Medicine and Biology Symposium (SPMB) (IEEE, 2015), 1–4.

76. A. Miltiadous, K. D. Tzimourta, T. Afrantou, et al., A Dataset of EEG Recordings from: Alzheimer’s Disease, Frontotemporal Dementia and Healthy Subjects (OpenNeuro, 2024), https://doi.org/10.18112/openneu ro.ds004504.v1.0.8.

77. P. Prado, V. Medel, R. Gonzalez-Gomez, et al., “The BrainLat Project, a Multimodal Neuroimaging Dataset of Neurodegeneration from Underrepresented Backgrounds,” Scientific Data 10, no. 1 (2023): 889.

78. M. Lahijanian, H. Aghajan, and Z. Vahabi, 40Hz Auditory Entrainment (OpenNeuro, 2024), https://doi.org/10.18112/openneuro.ds005048.v1.0 .0.

79. J. Ma, B. Yang, W. Qiu, Y. Li, S. Gao, and X. Xia, “SHU Multi-Session Dataset,” (2022), https://doi.org/10.6084/m9.figshare.19228725.v3.

80. A. Singh, R. Cole, A. Espinoza, J. Cavanagh, and N. Narayanan, “Rest Eyes Open,” (OpenNeuro, 2023), https://doi.org/10.18112/openneuro.d s004584.v1.0.0.

81. G. Schalk, D. J. McFarland, T. Hinterberger, N. Birbaumer, and J. R. Wolpaw, “BCI2000: A General-Purpose Brain-Computer Interface (BCI) System,” IEEE Transactions on Biomedical Engineering 51, no. 6 (2004): 1034–1043.

82. P. Detti, G. Vatti, and G. Zabalo Manrique de Lara, “EEG Synchronization Analysis for Seizure Prediction: A Study on Data of Noninvasive Recordings,” Processes 8, no. 7 (2020): 846.

83. A. P. Rockhill, N. Jackson, J. George, A. Aron, and N. C. Swann, UC San Diego Resting State EEG Datafrom Patients with Parkinson’s Disease (OpenNeuro, 2021), https://doi.org/10.18112/openneuro.ds002778.v1.0 .5.

84. E. Olejarczyk and W. Jernajczyk, EEG in Schizophrenia (RepOD, 2017), https://doi.org/10.18150/repod.0107441.

85. B. Lin, Y. Ye, B. Zhu, et al., “Video-LLaVA: Learning United Visual Representation by Alignment Before Projection,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics, 2024), 5971–5984, https://doi.org/10.18653/v1/2024.emnlp-main.342.

## Supporting Information

## A Language-Guided Multimodal Foundation Model for Zero-Shot and Multi-Task Brain Signal Analysis

Mingzhi Chen, Yiyu Gui, Guibo Luo, Yuchao Yang

Each dataset is associated with one instruction sentence and a fixed answer set. In the multiple-choice format the same answer set is inserted into the prompt as lettered options.

Table S1. Instruction-answer templates used to build the pretraining QA pairs.
<table><tr><td>Dataset</td><td>Instruction</td><td>Answer</td></tr><tr><td rowspan="2">HUP</td><td rowspan="2">Which epilepsy state does this signal belong to? &quot;Interictal&quot;, &quot;Ictal&quot;,</td><td></td></tr><tr><td>&quot;Preictal&quot;, &quot;Postictal&quot;</td></tr><tr><td>UPenn</td><td>Which epilepsy state does this signal belong to? &quot;Interictal&quot;, &quot;Ictal&quot;</td><td></td></tr><tr><td>SWEC ETHZ</td><td>Which epilepsy state does this signal belong to? &quot;Preictal&quot;, &quot;Ictal&quot;,</td><td></td></tr><tr><td rowspan="2">FNUSA</td><td rowspan="2"></td><td>&quot;Postictal&quot;</td></tr><tr><td>Which epilepsy state does this signal belong to? &quot;Interictal&quot;,</td></tr><tr><td rowspan="2">SHHS</td><td rowspan="2"></td><td>&quot;Interictal, Pathological activity&quot;</td></tr><tr><td>&quot;Wake&quot;, &quot;Non-REM Stage 1&quot;,</td></tr><tr><td rowspan="3"></td><td rowspan="3"></td><td>&quot;Non-REM Stage 2&quot;, &quot;Non-REM Stage 3&quot;,</td></tr><tr><td>&quot;Rapid Eye Movement&quot;</td></tr><tr><td>Which epilepsy state does this signal belong to? &quot;Interictal&quot;, &quot;Ictal&quot;,</td></tr><tr><td></td><td></td><td>&quot;Preictal&quot;, &quot;Postictal&quot;</td></tr><tr><td>TUAB</td><td>Is the signal normal or abnormal?</td><td>&quot;Normal&quot;, &quot;Abnormal&quot;</td></tr><tr><td rowspan="2">CHB MIT</td><td rowspan="2">Which epilepsy state does this signal belong to? &quot;Interictal&quot;, &quot;Ictal&quot;,</td><td></td></tr><tr><td>&quot;Preictal&quot;, &quot;Postictal&quot;</td></tr><tr><td rowspan="3">SleepEDF</td><td rowspan="3">Which sleep stage does this signal belong to?</td><td>&quot;Wake&quot;, &quot;Non-REM Stage 1&quot;,</td></tr><tr><td>&quot;Non-REM Stage 2&quot;, &quot;Non-REM Stage 3&quot;,</td></tr><tr><td>&quot;Rapid Eye Movement&quot;</td></tr><tr><td rowspan="3">HaaglandenSleep</td><td rowspan="3">Which sleep stage does this signal belong to?</td><td>&quot;Wake&quot;, &quot;Non-REM Stage 1&quot;,</td></tr><tr><td>&quot;Non-REM Stage 2&quot;, &quot;Non-REM Stage 3&quot;,</td></tr><tr><td>&quot;Rapid Eye Movement&quot;</td></tr><tr><td>TDBrain TUEV</td><td>Which disease does this signal belong to?</td><td>dataset-specific clinical labels (see Table S2)</td></tr><tr><td rowspan="3"></td><td rowspan="3">Which type does this signal belong to?</td><td>&quot;Spike and slow wave&quot;, &quot;Generalized periodic epileptiform discharge&quot;,</td></tr><tr><td>&quot;Periodic lateralized epileptiform dischage&quot;, &quot;Eye movement&quot;, &quot;Artifact&quot;, &quot;Background&quot;</td></tr><tr><td>&quot;Normal&quot;, &quot;Alzheimer&#x27;s disease&quot;,</td></tr><tr><td rowspan="2"></td><td rowspan="2">Which disease does this signal belong to?</td><td>&quot;Frontotemporal dementia&quot;</td></tr><tr><td>&quot;Normal&quot;, &quot;Alzheimer&#x27;s disease&quot;,</td></tr><tr><td rowspan="2">BrainLat</td><td rowspan="2">Which disease does this signal belong to?</td><td>&quot;Frontotemporal dementia&quot;, &quot;Parkinson&#x27;s disease&quot;, &quot;Multiple sclerosis&quot;</td></tr><tr><td></td></tr><tr><td rowspan="2">AD-Auditory</td><td rowspan="2">Which disease does this signal belong to?</td><td>&quot;Normal&quot;, &quot;Alzheimer&#x27;s disease&quot;,</td></tr><tr><td>&quot;Mild cognitive impairment&quot;</td></tr><tr><td rowspan="2">ShuMI</td><td rowspan="2">Which type does this signal belong to?</td><td>&quot;Motor imagery left hand&quot;,</td></tr><tr><td>&quot;Motor imagery right hand&quot;</td></tr><tr><td>REEG-PD</td><td>Which disease does this signal belong to? Which type does this signal belong to?</td><td>&quot;Normal&quot;, &quot;Parkinson&#x27;s disease&quot;</td></tr><tr><td rowspan="4">PhysionetMI</td><td rowspan="4"></td><td>&quot;Motor imagery left fist&quot;,</td></tr><tr><td>&quot;Motor imagery right fist&quot;,</td></tr><tr><td>&quot;Motor imagery both fists&quot;, &quot;Motor imagery both feet&quot;</td></tr><tr><td>dataset-specific clinical labels (see Table S3)</td></tr><tr><td>TUEP</td><td>Which type does this signal belong to? Does this signal belong to epilepsy?</td><td>&quot;No&quot;, &quot;Yes&quot;</td></tr><tr><td></td><td></td><td></td></tr></table>

Table S2. TDBrain label inventory and canonical answer strings
<table><tr><td>Class</td><td>Canonical answer string</td></tr><tr><td>0</td><td>&quot;Burnout&quot;</td></tr><tr><td>1</td><td>&quot;Subjective memory complaints&quot;</td></tr><tr><td>2</td><td>&quot;Normal&quot;</td></tr></table>

Table S2. Continued.
<table><tr><td>Class</td><td>Canonical answer string</td></tr><tr><td>3</td><td>&quot;Dyslexia&quot;</td></tr><tr><td>4</td><td>&quot;Chronic pain&quot;</td></tr><tr><td>5</td><td>&quot;Major depressive disorder&quot;</td></tr><tr><td>6</td><td>&quot;Attention deficit hyperactivity disorder&quot;</td></tr><tr><td>7</td><td>&quot;Attention deficit hyperactivity disorder, asperger syndrome&quot;</td></tr><tr><td>8</td><td>&quot;Pervasive developmental disorder not otherwise specified, dyslexia&quot;</td></tr><tr><td>9</td><td>&quot;Pervasive developmental disorder not otherwise specified&quot;</td></tr><tr><td>10</td><td>&quot;Whiplash&quot;</td></tr><tr><td>11</td><td>&quot;Anxiety&quot;</td></tr><tr><td>12</td><td>&quot;Attention deficit hyperactivity disorder, dyslexia&quot;</td></tr><tr><td>13</td><td>&quot;Autism spectrum disorder&quot;</td></tr><tr><td>14</td><td>&quot;Tinnitus&quot;</td></tr><tr><td>15</td><td>&quot;Obsessive-compulsive disorder&quot;</td></tr><tr><td>16</td><td>&quot;Panic disorder&quot;</td></tr><tr><td>17</td><td>&quot;Major depressive disorder, anxiety&quot;</td></tr><tr><td>18</td><td>&quot;Migraine&quot;</td></tr><tr><td>19</td><td>&quot;Pervasive developmental disorder not otherwise specified, anxiety&quot;</td></tr><tr><td>20</td><td>&quot;Parkinson&#x27;s disease&quot;</td></tr><tr><td>21</td><td>&quot;Bipolar disorder&quot;</td></tr><tr><td>22</td><td>&quot;Major depressive disorder, bipolar disorder&quot;</td></tr><tr><td>23</td><td>&quot;Dyspraxia&quot;</td></tr><tr><td>24</td><td>&quot;Tinnitus, major depressive disorder&quot;</td></tr><tr><td>25</td><td>&quot;Attention deficit hyperactivity disorder, autism spectrum disorder, anxiety&quot;</td></tr><tr><td>26</td><td>&quot;Major depressive disorder, attention deficit hyperactivity disorder&quot;</td></tr><tr><td>27</td><td>&quot;Attention deficit hyperactivity disorder, pervasive developmental disorder not otherwise specified&quot;</td></tr><tr><td>28</td><td>&quot;Asperger syndrome&#x27;</td></tr><tr><td>29</td><td>&quot;Attention deficit hyperactivity disorder, epilepsy&quot;</td></tr><tr><td>30</td><td>&quot;Major depressive disorder, pain&quot;</td></tr><tr><td>31</td><td>&quot;Pervasive developmental disorder not otherwise specified, gilles de la tourette syndrome&quot;</td></tr><tr><td>32</td><td>&quot;Pervasive developmental disorder not otherwise specified, attention deficit hyperactivity disorder&quot;</td></tr><tr><td>33</td><td>&quot;Pervasive developmental disorder not otherwise specified, autism spectrum disorder&quot;</td></tr><tr><td>34</td><td>&quot;Traumatic brain injury&#x27;</td></tr><tr><td>35</td><td>&quot;Attention deficit hyperactivity disorder, anxiety&quot;</td></tr><tr><td>36</td><td>&quot;Attention deficit hyperactivity disorder, dyslexia, dyscalculia&quot;</td></tr><tr><td>37</td><td>&quot;Attention deficit hyperactivity disorder, major depressive disorder&quot;</td></tr><tr><td>38</td><td>&quot;Major depressive disorder, panic disorder&quot;</td></tr><tr><td>39</td><td>&quot;Depersonalization disorder&quot;</td></tr><tr><td>40</td><td>&quot;Major depressive disorder, trauma&quot;</td></tr><tr><td>41</td><td>&quot;Post-traumatic stress disorder, attention deficit hyperactivity disorder&quot;</td></tr><tr><td>42</td><td>&quot;Obsessive-compulsive disorder, dissociative psychogenic seizures&quot;</td></tr><tr><td>43</td><td>&quot;Major depressive disorder, obsessive-compulsive disorder&quot;</td></tr><tr><td>44</td><td>&quot;Major depressive disorder, tumor&quot;</td></tr><tr><td>45</td><td>&quot;Attention deficit hyperactivity disorder, gilles de la tourette syndrome&#x27;</td></tr><tr><td>46</td><td>&quot;Obsessive-compulsive disorder, major depressive disorder&quot;</td></tr><tr><td>47</td><td>&quot;Conversion disorder&quot;</td></tr><tr><td>48</td><td>&quot;Autism spectrum disorder, asperger syndrome&quot;</td></tr><tr><td>49</td><td>&quot;Major depressive disorder, attention deficit hyperactivity disorder, lyme disease&quot;</td></tr><tr><td>50</td><td>&quot;Attention deficit hyperactivity disorder, obsessive-compulsive disorder&quot;</td></tr><tr><td>51</td><td>&quot;Multiple system atrophy - cerebellar type&quot;</td></tr><tr><td>52</td><td>&quot;Obsessive-compulsive disorder, autism spectrum disorder&quot;</td></tr></table>

Table S2. Continued.
<table><tr><td>Class</td><td>Canonical answer string</td></tr><tr><td>53</td><td>&quot;Stroke, pain&quot;</td></tr><tr><td>54</td><td>&quot;Stroke&quot;</td></tr><tr><td>55</td><td>&quot;Major depressive disorder, obsessive-compulsive disorder, attention deficit hyperactivity disorder&quot;</td></tr><tr><td>56</td><td>&quot;Epilepsy, obsessive-compulsive disorder&quot;</td></tr><tr><td>57</td><td>&quot;Insomnia&quot;</td></tr><tr><td>58</td><td>&quot;Major depressive disorder, attention deficit hyperactivity disorder, anorexia&quot;</td></tr><tr><td>59</td><td>&quot;Major depressive disorder, anxiety, tinnitus&quot;</td></tr></table>

Table S3. TUSZ label inventory and canonical answer strings
<table><tr><td>Class</td><td>Canonical answer string</td></tr><tr><td>0</td><td>&quot;Spike/Sharp and Wave&quot;</td></tr><tr><td>1</td><td>&quot;Generalized Periodic Epileptiform Discharges&#x27;</td></tr><tr><td>2</td><td>&quot;Periodic Lateralized Epileptiform Discharges&quot;</td></tr><tr><td>3</td><td>&quot;Eye blink&quot;</td></tr><tr><td>4</td><td>&quot;Artifacts&quot;</td></tr><tr><td>5</td><td>&quot;Epilepsy ictal state, Seizure&quot;</td></tr><tr><td>6</td><td>&quot;Epilepsy ictal state, Focal Non-Specific Seizure&quot;</td></tr><tr><td>7</td><td>&quot;Epilepsy ictal state, Generalized Non-Specific Seizure&quot;</td></tr><tr><td>8</td><td>&quot;Epilepsy ictal state, Simple Partial Seizure&quot;</td></tr><tr><td>9</td><td>&quot;Epilepsy ictal state, Complex Partial Seizure&quot;</td></tr><tr><td>10</td><td>&quot;Epilepsy ictal state, Absence Seizure&quot;</td></tr><tr><td>11</td><td>&quot;Epilepsy ictal state, Tonic Seizure&quot;</td></tr><tr><td>12</td><td>&quot;Epilepsy ictal state, Clonic Seizure&quot;</td></tr><tr><td>13</td><td>&quot;Epilepsy ictal state, Tonic Clonic Seizure&quot;</td></tr><tr><td>14</td><td>&quot;Epilepsy ictal state, Atonic Seizure&quot;</td></tr><tr><td>15</td><td>&quot;Epilepsy ictal state, Myoclonic Seizure&quot;</td></tr><tr><td>16</td><td>&quot;Panic disorder&quot;</td></tr><tr><td>17</td><td>&quot;Major depressive disorder, anxiety&quot;</td></tr><tr><td>18</td><td>&quot;Epilepsy ictal state, Non-Epileptic Seizure&quot;</td></tr><tr><td>19</td><td>&quot;Interesting Patterns&quot;</td></tr><tr><td>20</td><td>&quot;Slowing&quot;</td></tr><tr><td>21</td><td>&quot;Eye Movement Artifact&quot;</td></tr><tr><td>22</td><td>&quot;Chewing Artifact&quot;</td></tr><tr><td>23</td><td>&quot;Shivering Artifact&quot;</td></tr><tr><td>24</td><td>&quot;Muscle Artifact&quot;</td></tr><tr><td>25</td><td>&quot;Electrode Pop Artifact&quot;</td></tr><tr><td>26</td><td>&quot;Electrostatic Artifact&quot;</td></tr><tr><td>27</td><td>&quot;Calibration Artifact&quot;</td></tr><tr><td>28</td><td>&quot;Hypnagogic Hypersynchrony&#x27;</td></tr><tr><td>29</td><td>&quot;Triphasic Wave&quot;</td></tr></table>

![](images/013ec44d11c10ae64eb4d9e13eb0b159f9a2841754f19ddbfdbbc1de62ef95c8.jpg)

Figure S1. ISRUC dataset class composition: overall counts and per fold proportions.  
![](images/e4f89a6eb2a94da0ce8a260b9e3493da95199d6a75a8cde5b77325eebbac689c.jpg)

Figure S2. Dreams dataset class composition: overall counts and per fold proportions.  
![](images/c475e65a65ba13406d9ee85de10c38e9e0d1f5148fb7b08c9e7ef54c31ecc818.jpg)  
Figure S3. Mayo dataset class composition: overall counts and per fold proportions.

![](images/34a27a1b41b06273503d9b1a20531669624540abe033a93ca049e050785cfce6.jpg)

Figure S4. IEDS dataset class composition: overall counts and per fold proportions.  
![](images/54a4c9ace686b256d9cee586b98c4674a0fd146f3b59b6d443b83fed7d313e3e.jpg)

Figure S5. ADFSU dataset class composition: overall counts and per fold proportions.  
![](images/a3dd195f11e601889c2a84ea4c57ef00ddf5a5766750f88786eaac25a0709972.jpg)  
Figure S6. APAVA dataset class composition: overall counts and per fold proportions.

![](images/c30b7ae807a4125fd8a5c97311b8ef38af90868e7a06ec4dc2b71795b8b7516f.jpg)

Figure S7. ADHD-80 dataset class composition: overall counts and per fold proportions.  
![](images/9ef8f952f2aab60253498e8f7e38bed29ae1030689f858b3d94cb6d636272e97.jpg)

Figure S8. ADHD-121 dataset class composition: overall counts and per fold proportions.  
![](images/45c56418ca4ed0c2852d7a715deb643693a9493a7feea91ec17d898a6650190a.jpg)  
Figure S9. Schizo-28 dataset class composition: overall counts and per fold proportions.

![](images/51fd48c429401e891eb9fd0b71a1a8cbfb59f527f53b7063231246249223846b.jpg)

Figure S10. Schizo-Youth dataset class composition: overall counts and per fold proportions.  
![](images/a091fadb8b814e6c1fc44c4d0ce5ba909ed37d679fbd76d588af39335f30b1ab.jpg)

Figure S11. MDD dataset class composition: overall counts and per fold proportions.  
![](images/a65bf8b79447b79b5ad572f76a7268291057670489c01d3559ddd249d6d5cb5d.jpg)  
Figure S12. SanDiego dataset class composition: overall counts and per fold proportions.

![](images/efab84a9bf00b992297fd7dabc88c6dd241f33223e5beab2df8b91923b3c2664.jpg)

Figure S13. Siena dataset class composition: overall counts and per fold proportions.  
![](images/a71abb5c857a47893a17ac107f5d3b482d5a15f579cdf2de235ae48b71555f0a.jpg)

Figure S14. RatEpilepsy dataset class composition: overall counts and per fold proportions.  
![](images/fb5477052d9a778fb871e94286c79c3481f49ac67eeec91531cfe55f2752efd7.jpg)  
Figure S15. NTUHBIS dataset class composition: overall counts and per fold proportions.

![](images/c1fcde0a3832e71dde5004e7f25eafe4bcec5bd85edcea743a201655834acf2e.jpg)

Figure S16. SEE dataset class composition: train, test, and total.  
![](images/83e67b65e67e8a5e1186541c53c30af4f70cc7108f79dd47dda0743a8bce9e37.jpg)  
Figure S17. NMT dataset class composition: train, test, and total.

![](images/2f3ff5edea74daac24451a494e90a31d4cf03b49842497ba46bcf2f4295a6b3a.jpg)

![](images/989620f6091b3323c2cf8379dad06ab7169d7f555b412d01c491b9bce86217db.jpg)

## Sleep stage classification

![](images/6197a4b426abfd9e3d04fcf294ad758138c357c4877fef7dfdfddc2861a19e6c.jpg)

![](images/8d25f01851fb47bd17876e065ee0cdd958cd484bff6ed9a0302cdbc9e8be2c14.jpg)

## Sample from ISRUC dataset

## (I) Multiple-choice QA

![](images/10eb68545f7dd3d5169efd01a5424b8c9be6f411e3b81d37992c51896fd94410.jpg)

![](images/c05559e22cd4c81326bf5a081009cd4adb1d7994c22f43f49a6a7ee57146b4f8.jpg)

![](images/0b620c164b1a55a07c0f584314b91838cf113aed6d503fd76250ba19ace94e74.jpg)

Which sleep stage does this signal belong to?

![](images/c1967115957d5b82ba92de38dd2145007e3b502f5f5751b2d6157eb971057e82.jpg)

## ADHD detection

![](images/f5b1882d3c358665a4877ce1e2e190f44b6c83a9b5daabbb3f2ef019aa624a93.jpg)

![](images/8019e32972dcf215a18cc2221e5ebacaf016847b73c785c8c1530c1cf2b7fbba.jpg)

## Sample from ADHD-121 dataset

![](images/f5f66928ec0a58f3e2c6afd281737ad02dc2bf737a76f33a3c2ea3f1a09448e7.jpg)

![](images/e37c0e6a2c9a1ce84c1530dc18d3da88d64dee0055d2f671fa3a266ddd152997.jpg)

![](images/694c1f67d27b26cd0e04928fb0d02fb4d59302e6252b8931617c2df5176cce25.jpg)

![](images/291df5e482174660850b77103b35b9fed1b9a6634d0d9ebcb51a608f212fe0c7.jpg)

![](images/30660a2966e879cb5c015a5a771bcd937a9c63516735281dd388f273f19997a5.jpg)

![](images/79159a9cf50b959722210f9a401177909f3a778b566c31dfbc0e585be9cf962e.jpg)

Which disease does this signal belong to?

## Schizophrenia detection

![](images/a48ad64182f27bd033b5614d674d6385c233fdad598a2b6b1add147af09c0b53.jpg)

![](images/5091f4629c76cbee08c1751ea23b2220c7ad243310387713fabc2dc275646a87.jpg)

![](images/c2963ce9d8b8a14d87f36b587d12335977d4a207d1ffeb63e5cee09b6fdb6630.jpg)

(Il). This signal is ChatGPT associated with epilepsy.

## Sample from Schizo-Youth dataset

![](images/b02aa0f6956efe97627d9908beab2cf3fac4e8c9b274886e6a63b8474e54890b.jpg)

## (I) Multiple-choice QA

![](images/bf1c03fa76e302e7ced3da6955a370c65d3b8f81748035fea4280c5023f01ded.jpg)

![](images/541346ad5d8c59b91bdeca421673d34531d6efcc657a0d1d6e2b24f9e337e795.jpg)

![](images/47411f44be6bb56db329c503e8726d45691ca654da5db4aec07eab550a592757.jpg)

![](images/3758686ec93a4aae24fb7907c0f4d96a21fc8ef9425e4cda133c3d5bdbe5b17d.jpg)

Which disease does this signal belong to?

![](images/b13095f7cf0b4c074445dd14c2520ac23381fc9e82818c189a2d368aea158b95.jpg)

## Interictal epileptiform discharge detection

![](images/e0048ba37bcc498d6cedb338f98daf7f1502d613e286c0826674d28ac1b69049.jpg)

![](images/4bc98b5ab001b2109113fb04372c04ac855ffc88adc1833ba47a0a0be12f87ae.jpg)

(1). B (Ií). Ictal state

![](images/25b9dc68fa7be868287bf436ebf077dae3eda2981057849d2002946cd54c4e8b.jpg)

![](images/f3df036e5dbd42b875e5cde1a8188d815b703bf7c25ed02ec6e66ff4dc280916.jpg)

![](images/e720b98c5fec0a70533907c648da4f969af58ad32fbaf16c12f7d80c05ff04ee.jpg)

![](images/536edc5e7e0e9561c9fcf3fba1cc9096c9c7f238a76af7b2d0cbfb4c55ee59e4.jpg)

![](images/d9cdf3f1db884c0c66f1d71cc5f0bccb154993f20a0a0a945c21b2da1c9d48d2.jpg)

Which epilepsy state does this signal belong to?

![](images/736fe987b8a70335b24f44c2168a48583c85faa123b1a14da409fe13e15297db.jpg)

Figure S18. Additional zero shot task examples on ISRUC, Schizo-Youth, ADHD-121, and IEDS. a, Sleep stage classification on ISRUC. Each example uses the same signal and pairs one multiple-choice query with one open ended query. Model outputs are shown for METIS, ChatGPT, Gemini, Grok, and DeepSeek. Correct answers are marked with a green checkmark for readability. b, Schizophrenia detection on Schizo-Youth. Binary disease detection with options Normal and Schizophrenia under the multiple-choice format, together with an open ended query that expects a canonical label. Green checkmarks indicate correct responses. c, ADHD detection on ADHD-121. Binary disease detection with options Normal and Attention deficit hyperactivity disorder in the multiple-choice query and the matching open ended query. Green checkmarks mark correct outputs. d, Interictal epileptiform discharge detection on IEDS. State detection with dataset specific options Interictal and Interictal pathological activity in the multiple-choice query and an open ended counterpart. All panels follow the same presentation: one signal, two query formats, five models, green checkmarks for correct answers and no marks for incorrect ones.

![](images/6a89fa41bb3761b6784a90c2f198266878e2ab72178bb0a6c0bd043e2c6821af.jpg)

![](images/5fc28a68a8fb82972dad20b00a7beae203af84ea1c422448c760f8f88d3bf770.jpg)

## Mouse seizure detection

![](images/924fcc8ee93bf2ec9a31dac92a6dc40a63431f75a1a6a80417f7dfd01eed3b18.jpg)

![](images/1e750f946002d0167624da6666e196b5b07cec44b4731682314e1d4444968493.jpg)

![](images/881e7def22f117fae03bc151fec7fda3e2a51b9646ada66edcc7d2061c3c849a.jpg)

![](images/a8bbf865a2745432051be53ec5934871e5a2d8d34d310b5d8022c0dcaaed2f89.jpg)

## Sample from RatEpilepsy dataset

![](images/08895c8589d3dd0ae9dd0a0f1f19d2b2e591d2004134466673afdb3400476238.jpg)

![](images/1e2e9a47f7f856d2b07850100ea7ab1c43da028fc1ab5478b2c9225f03e3acfb.jpg)

Which epilepsy state does   
this signal belong to?   
A. Intracranial   
B. Ictal

![](images/c34bce9b47d95c70a4c38e8c4bff7f392666a20293ae4896ae60aca9af517e17.jpg)

Which epilepsy state does this signal belong to?

![](images/7b33b0bde463a1196da27926581bc6e22c76eab9dfb866dc8e1345baaa3748de.jpg)

## ADHD detection

![](images/0f1c946966a895d5867dfcbf01f34430dc08375f55a2b3cbe6464fac3982f95b.jpg)

![](images/e7f19d0270e90071d6f5e56f9f1f3981444556a66a715192355085ca0250365d.jpg)

![](images/c0a2881c6348c663928f95d76eca17d8d67834f3bdc0f0266b115c6a245cdca5.jpg)

![](images/ef2b7f44821f581cbe682fc4e4fc40f01bef82246432308ee9871a09ffaf5da7.jpg)

![](images/de07d293901daa6149ce61291ddb0d00129339ed26261fe3223795809c5cf414.jpg)

![](images/50bb0b338851ec1a40d08ecd9faa042104792b0364b1431e0675979f69b63226.jpg)

## Signal anomaly detection

![](images/aeb9d70021eee1d172dcca1e1ba6c1d3115b543b451b7860ed2d877921f7a64e.jpg)

![](images/d8ab7e3d04b0ca97cca215b7f581aced24e5b11715cacc5d46571e36e5d2e706.jpg)

## Sample from NMT dataset

![](images/8c0661681824c6a2a7c8f93e0b1dbacd85278288cd77c3fc26e6cd12d04ad206.jpg)

![](images/60709f5655b93fb020acd50408b05b32b98b66e6df776e9947cc171644fb3808.jpg)

Which disease does this signal belong to?

![](images/38e1e6887ce9d7176d149b2ec11985965e427fd18571ee85e27375958af52ff9.jpg)

![](images/9399a5bbd5c182bf78198bc62f825af7c9ff15932dccc6099988611c3775d8e0.jpg)

![](images/0a918f6478422070e3cb47980c43b3112bbc26e97f58473739b4ae23acb7de9b.jpg)

![](images/26d51807b5a594ce741d8689d66e55f579aaeee9986ad9febd8c1a75155f7fcf.jpg)

![](images/90d53a08e489c500d476d5e32990b857fd0f00266aa71ce967f13fe82a0ace7b.jpg)

![](images/251c9451b3fb13a6b60f001e9500e27051f46fb71eb84d1ee88e5464f3d787a7.jpg)

Is the signal normal or abnormal?

![](images/eac5236989ad70925e96d1590d3fb2a61d08f81732a89fea88002f2490858405.jpg)

![](images/cc1612cc2d4720d691f8b01e3320c7af86f0b5bafb1c617dfd0a48b5c8d2b254.jpg)

## Epilepsy detection

![](images/46c58edc825671bcc18ed5b04020139a2ba4c42486cef710eaccb830823c5328.jpg)  
Sample from SEE dataset

![](images/185e04482b2169f08e74a5bb48728c791441adb35b28641e0e758b2de0b7bf41.jpg)

![](images/8bee74aa998a2688dd668439cc0a9e4f0c478a64faaab23d097356920335db7e.jpg)

![](images/7bb8e69774510e2508b5922cc5554508e984689239a1cd89a2429f9ebb5d05da.jpg)

![](images/b4a94fd5ac60bb63586735d0ad8d09e51c0c3c0012da697f8bf2381417a0c312.jpg)

Does this signal belong to   
epilepsy?   
A. No   
B. Yes

![](images/72722bf3edd87ef89c28af1f6576333ce48c57b0b586d7f8b99ef2139f3eee32.jpg)  
Figure S19. Additional zero shot task examples on RatEpilepsy, NMT, ADHD-80, and SEE. a, Mouse seizure detection on RatEpilepsy. Ictal versus non-ictal state detection shown with a multiple-choice query that lists the dataset options and an open ended query that expects a canonical state label. Outputs are reported for METIS, ChatGPT, Gemini, Grok, and DeepSeek. Correct answers carry a green checkmark. b, Signal anomaly detection on NMT. Binary abnormality screening with Normal and Abnormal as the candidate set in the multiple-choice query and an aligned open ended query. Green checkmarks indicate correct predictions. c, ADHD detection on ADHD-80. Binary disease detection with options Normal and Attention deficit hyperactivity disorder under the multiple-choice format, plus an open ended query that expects the same label space. Correct answers are marked with green checkmarks. d, Epilepsy detection on SEE. Binary epilepsy screening with options No and Yes for the multiple-choice query and the matched open ended query. Across all panels a single signal is queried in both formats, the five models are evaluated under identical prompts, correct responses are highlighted with a green checkmark, and unmarked responses are incorrect.

![](images/fd0bc8e3437814df5fdb4dfff0ff2ea67c655883dd101d373bb40cad393c7a30.jpg)  
Figure S20. Zero-shot performance comparison between METIS and generalist multimodal models across two QA protocols. Panel a presents accuracy scores for multiple-choice question answering tasks, where the model selects the most plausible answer among a limited set of options. Panel b shows BERTScore for open-ended question answering, evaluating the semantic similarity between generated answers and ground-truth references. Each bar represents the average performance across repeated runs on 12 diverse brain signal datasets. The rightmost group in each panel reports the overall mean across all datasets, using the same averaging method. Significance markers (\*, \*\*, \*\*\*) indicate the diference between METIS and the best-performing generalist baseline, based on two-sided t-tests $( ^ { * } \mathrm { p } < 0 . 0 5 , ^ { * * } \mathrm { p } < 0 . 0 1 , ^ { * * * } \mathrm { p } < 0 . 0 0 1 )$ . Across both protocols, METIS consistently achieves superior zero-shot performance compared to state-of-the-art generalist models such as ChatGPT, Gemini, Grok, and DeepSeek.

## Model Configuration and Architectural Details

METIS is implemented with a hidden dimension of 512 across all transformer layers. Query-key normalization is applied in the attention mechanism to improve training stability. For standard feed-forward layers, the intermediate dimensionality is set to 2048. For mixture-of-experts (MoE) layers, each routed expert adopts an intermediate dimension of 512, and the shared expert adopts an intermediate dimension of 1024. The backbone consists of 12 transformer layers in total. To ensure stable optimization during early training, the first two layers are implemented as standard transformer layers without expert routing. The remaining layers are configured as MoE layers. Each MoE layer contains 8 routed experts and 1 shared expert, following a top-k routing strategy with $k = 2$ . Multi-head attention is employed with 8 query heads, while key and value representations are shared across 2 heads to reduce computational overhead. All other architectural components follow standard transformer implementations unless otherwise specified.

## MoE Contribution Under Matched Model Scale

Table S4. Model scale comparison between METIS-MoE and METIS-Dense.
<table><tr><td>Model</td><td>MoE</td><td>Layers</td><td>Attention heads</td><td>Hidden dim</td><td>Total params</td><td>Active params</td></tr><tr><td>METIS-MoE</td><td>Yes</td><td>12</td><td>8</td><td>512</td><td>170.70M</td><td>124.21M</td></tr><tr><td>METIS-Dense</td><td>No</td><td>12</td><td>10</td><td>640</td><td>168.70M</td><td>168.70M</td></tr></table>

Table S5. Parameter-matched dense Transformer baseline under zero-shot evaluation. Gain = METIS-MoE − METIS-Dense, reported in percentage points.
<table><tr><td>Dataset</td><td>METIS-Dense AUROC</td><td>METIS-MoE AUROC</td><td>Gain (pp)</td></tr><tr><td>ISRUC</td><td>0.9525</td><td>0.9411</td><td>-1.14</td></tr><tr><td>Dreams</td><td>0.9404</td><td>0.9289</td><td>-1.15</td></tr><tr><td>Mayo</td><td>0.8970</td><td>0.9352</td><td>+3.82</td></tr><tr><td>IEDS</td><td>0.6308</td><td>0.6419</td><td>+1.11</td></tr><tr><td>SEE</td><td>0.6518</td><td>0.7145</td><td>+6.27</td></tr><tr><td>RatEpilepsy</td><td>0.5270</td><td>0.6319</td><td>+10.49</td></tr><tr><td>ADHD-80</td><td>0.4000</td><td>0.6707</td><td>+27.07</td></tr><tr><td>ADHD-121</td><td>0.6365</td><td>0.7240</td><td>+8.75</td></tr><tr><td>MDD</td><td>0.4833</td><td>0.7425</td><td>+25.92</td></tr><tr><td>Schizo-Youth</td><td>0.6302</td><td>0.7208</td><td>+9.06</td></tr><tr><td>ADFSU</td><td>0.6958</td><td>0.7458</td><td>+5.00</td></tr><tr><td>NMT</td><td>0.7153</td><td>0.6620</td><td>-5.33</td></tr><tr><td>Mean</td><td>0.6801</td><td>0.7550</td><td>+7.49</td></tr></table>

To examine whether the gain of METIS comes from the MoE architecture rather than total model scale, we constructed a parameter-matched dense Transformer baseline, denoted METIS-Dense. As shown in Table S4, METIS-Dense replaces the sparse expert feed-forward layers with standard dense feed-forward layers, while maintaining a comparable total parameter count to METIS-MoE.

METIS-Dense follows the same multimodal signal-language formulation and the same zero-shot evaluation protocol as METIS-MoE. It also uses the same pretraining data, objectives, optimizer, optimization settings, batch size, and training steps. Specifically, METIS-Dense contains 12 Transformer layers with a hidden dimension of 640 and 168.70M parameters, and its feed-forward intermediate dimension is 2560. METIS-MoE contains 12 Transformer layers with a hidden dimension of 512 and 170.70M total parameters. In METIS-MoE, the first two layers are standard Transformer layers, and the remaining ten layers are MoE layers. Each MoE layer contains eight routed experts and one shared expert, with top-k routing using k = 2. Each routed expert has an intermediate dimension of 512, and the shared expert has an intermediate dimension of 1024. METIS-MoE has 124.21M active parameters, corresponding to 73.6% of the active parameters of METIS-Dense. We report active parameters as an input-independent measure, since approximate FLOPs depend on the input length and the number of signal tokens across datasets.

As shown in Table S5, METIS-MoE achieved a mean zero-shot AUROC of 0.7550 across 12 datasets, compared with 0.6801 for METIS-Dense, yielding an average gain of 7.49 percentage points. METIS-MoE outperformed METIS-Dense on 9 of the 12 datasets, with especially large gains on ADHD-121, MDD, RatEpilepsy, Schizo-Youth, ADHD-80, SEE, ADFSU, and Mayo. These results suggest that sparse expert routing is beneficial for heterogeneous clinical brain-signal tasks, where diverse signal patterns and task semantics must be handled within a unified model.

METIS-Dense performed slightly better on ISRUC, Dreams, and NMT, indicating that dense feed-forward capacity can remain competitive for some structured tasks. Nevertheless, the overall zero-shot benchmark supports that the MoE architecture contributes to METIS’s generalization ability beyond merely increasing total parameter count.

## Dataset Usage and Overlap Control

Table S6. Dataset usage for pretraining and downstream evaluation.
<table><tr><td>Dataset</td><td>Modality</td><td>Usage</td><td>Downstream setting</td></tr><tr><td>HUP</td><td>iEEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>UPenn</td><td>iEEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>SWEC_ETHZ</td><td>iEEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>FNUSA</td><td>iEEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>SHHS</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>SeizeIT2</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>TUSZ</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>TUAB</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr></table>

Continued on next page

Table S6. Continued.
<table><tr><td>Dataset</td><td>Modality</td><td>Usage</td><td>Downstream setting</td></tr><tr><td>CHB-MIT</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>TUEP</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>SleepEDF</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>HaaglandenSleep</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>TDBrain</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>TUEV</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>ADFTD</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>BrainLat</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>AD-Auditory</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>ShuMI</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>REEG-PD</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>PhysioNetMI</td><td>EEG</td><td>Pretraining</td><td>Not applicable</td></tr><tr><td>ISRUC</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>Dreams</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>SEE</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot</td></tr><tr><td>Siena</td><td>EEG</td><td>Downstream evaluation</td><td>Few-shot</td></tr><tr><td>RatEpilepsy</td><td>iEEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot</td></tr><tr><td>IEDS</td><td>iEEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>Mayo</td><td>iEEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>ADFSU</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>APAVA</td><td>EEG</td><td>Downstream evaluation</td><td>Transfer</td></tr><tr><td>SanDiego</td><td>EEG</td><td>Downstream evaluation</td><td>Few-shot</td></tr><tr><td>ADHD-121</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>ADHD-80</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot; transfer</td></tr><tr><td>MDD</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot</td></tr><tr><td>Schizo-Youth</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot</td></tr><tr><td>Schizo-28</td><td>EEG</td><td>Downstream evaluation</td><td>Few-shot</td></tr><tr><td>NMT</td><td>EEG</td><td>Downstream evaluation</td><td>Zero-shot; few-shot</td></tr><tr><td>NTUHBIS</td><td>EEG</td><td>Downstream evaluation</td><td>Few-shot</td></tr></table>

To clarify dataset usage, Table S6 maps each dataset to its role in the study. The pretraining corpus and downstream evaluation datasets are disjoint. We further confirmed that there was no subject-level overlap or inadvertent reuse of recordings between pretraining and downstream evaluation. For downstream evaluations, no subjects or recordings were shared across the training, validation, and test partitions. For datasets with oficial splits, we strictly followed the oficial partition protocols; for datasets without oficial splits, subject-wise partitioning was used to prevent subject-level leakage across folds.

## Efect of Pretraining Instruction Quality

To examine the impact of language-instruction quality during pretraining, we conducted a weak-instruction ablation. In this experiment, the model architecture, pretraining datasets, training protocol, and zero-shot evaluation protocol were kept unchanged. The only modification was that the original task-specific natural-language instructions were replaced with a generic weak instruction, “Classify this signal.” The label options were preserved. Therefore, this ablation weakens the task-level linguistic context while keeping the answer space available.

Table S7 shows the zero-shot performance of the full-instruction model and the weak-instruction model across the 12 datasets used in the main zero-shot benchmark. The full-instruction model achieved a mean AUROC of 0.7550, whereas the weak-instruction model achieved 0.6600, corresponding to an average gain of 9.50 percentage points. The full-instruction model outperformed the weak-instruction model on 11 of the 12 datasets.

The efect of instruction quality was task-dependent. The performance gap was relatively small on ISRUC, Dreams, Mayo, and IEDS, which are closely related to high-frequency pretraining task types such as sleep-stage classification and epilepsy-related classification. In these settings, the weak-instruction model can still partly rely on strong signal-label associations and the provided label options. In contrast, the degradation was much larger on clinically heterogeneous or semantically subtle tasks, including MDD, RatEpilepsy, ADFSU, Schizo-Youth, ADHD-121, ADHD-80, SEE, and NMT. This pattern suggests that high-quality task-specific instructions are especially important in heterogeneous multi-task pretraining, where they help condition the model on the appropriate semantic decision space rather than relying mainly on dominant task distributions or label-option associations.

These results indicate that the language instructions used during pretraining are not merely superficial prompt templates. Instead, they contribute to task-aware signal-language alignment and improve zero-shot generalization across diverse clinical brain-signal tasks.

Table S7. Weak-instruction ablation under zero-shot evaluation. Gain = Full instruction − Weak instruction, reported in percentage points.
<table><tr><td>Dataset</td><td>Full instruction</td><td>Weak instruction</td><td>Gain (pp)</td></tr><tr><td>ISRUC</td><td>0.9411</td><td>0.9408</td><td>+0.03</td></tr><tr><td>Mayo</td><td>0.9352</td><td>0.9389</td><td>-0.37</td></tr><tr><td>Dreams</td><td>0.9289</td><td>0.9157</td><td>+1.32</td></tr><tr><td>ADFSU</td><td>0.7458</td><td>0.5916</td><td>+15.42</td></tr><tr><td>MDD</td><td>0.7425</td><td>0.5482</td><td>+19.43</td></tr><tr><td>ADHD-121</td><td>0.7240</td><td>0.6392</td><td>+8.48</td></tr><tr><td>Schizo-Youth</td><td>0.7208</td><td>0.5691</td><td>+15.17</td></tr><tr><td>SEE</td><td>0.7145</td><td>0.6500</td><td>+6.45</td></tr><tr><td>ADHD-80</td><td>0.6707</td><td>0.5226</td><td>+14.81</td></tr><tr><td>NMT</td><td>0.6620</td><td>0.5334</td><td>+12.86</td></tr><tr><td>IEDS</td><td>0.6419</td><td>0.6273</td><td>+1.46</td></tr><tr><td>RatEpilepsy</td><td>0.6319</td><td>0.4431</td><td>+18.88</td></tr><tr><td>Mean</td><td>0.7550</td><td>0.6600</td><td>+9.50</td></tr></table>

## Prompt and Label Phrasing Robustness

To examine whether METIS relies on exact prompt templates or label surface forms, we evaluated the original METIS checkpoint under controlled prompt and label phrasing variations. The model parameters, datasets, evaluation protocol, answer space, and ground-truth annotations were kept unchanged. Only the surface form of the task instructions or answer labels was modified.

The prompt and label variants were constructed based on the original dataset annotation protocols and standard clinical terminology. For prompt paraphrasing, we rewrote the task questions while preserving the original task intent. For label phrasing, we used clinically equivalent terms, standard abbreviations, or spelling variants where applicable, such as “Rapid Eye Movement” and “REM,” or “Alzheimer’s disease” and “AD.” Each variant was checked against the original label definitions to ensure that the task meaning, class boundaries, and ground-truth annotations remained unchanged.

We evaluated two types of robustness. In the prompt paraphrasing test, the original label set was kept unchanged, while the canonical task instruction was replaced by three semantically equivalent paraphrased prompts. In the label phrasing test, the canonical prompt was kept unchanged, while the answer labels were replaced by three alternate but semantically equivalent label phrasings. Table S8 summarizes the mean results, and Tables S9 and S10 provide the per-variant results. The exact prompt paraphrases and alternate label phrasings used for each dataset are listed in Tables S11 and S12.

Across the 12 zero-shot datasets, METIS achieved a mean AUROC of 0.7550 under the canonical setting. Under paraphrased prompts, the mean AUROC was 0.7301, corresponding to an average decrease of 2.49 percentage points. Under alternate label phrasings, the mean AUROC was 0.7235, corresponding to an average decrease of 3.15 percentage points. These results indicate that METIS maintains overall stable zero-shot performance under controlled prompt and label surface-form changes.

The sensitivity to linguistic variation was task-dependent. Performance remained highly stable on ISRUC, Dreams, MDD, Schizo-Youth, and ADHD-80, while larger drops were observed on Mayo, IEDS, ADHD-121, ADFSU, and NMT. Overall, these results support that METIS is not solely dependent on exact template matching, while also acknowledging that some datasets show substantial sensitivity to prompt or label phrasing.

Table S8. Summary of prompt and label phrasing robustness under zero-shot evaluation. Change is computed relative to the canonical setting and reported in percentage points.
<table><tr><td>Dataset</td><td>Canonical AUROC</td><td></td><td></td><td>Prompt mean AUROC Prompt change (pp) Label mean AUROC Label change (pp)</td><td></td></tr><tr><td>ISRUC</td><td>0.9411</td><td>0.9365</td><td>-0.46</td><td>0.9387</td><td>-0.24</td></tr><tr><td>Dreams</td><td>0.9289</td><td>0.9274</td><td>-0.15</td><td>0.9285</td><td>-0.04</td></tr><tr><td>Mayo</td><td>0.9352</td><td>0.8645</td><td>-7.07</td><td>0.8541</td><td>-8.11</td></tr><tr><td>IEDS</td><td>0.6419</td><td>0.5631</td><td>-7.88</td><td>0.5583</td><td>-8.36</td></tr><tr><td>SEE</td><td>0.7145</td><td>0.7210</td><td>+0.65</td><td>0.6771</td><td>-3.74</td></tr><tr><td>RatEpilepsy</td><td>0.6319</td><td>0.6254</td><td>-0.65</td><td>0.6034</td><td>-2.85</td></tr><tr><td>ADHD-80</td><td>0.6707</td><td>0.7205</td><td>+4.98</td><td>0.7072</td><td>+3.65</td></tr><tr><td>ADHD-121</td><td>0.7240</td><td>0.6449</td><td>-7.91</td><td>0.6153</td><td>-10.87</td></tr><tr><td>MDD</td><td>0.7425</td><td>0.7138</td><td>-2.87</td><td>0.7418</td><td>-0.07</td></tr><tr><td>Schizo-Youth</td><td>0.7208</td><td>0.7242</td><td>+0.34</td><td>0.7637</td><td>+4.29</td></tr><tr><td>ADFSU</td><td>0.7458</td><td>0.7590</td><td>+1.32</td><td>0.6845</td><td>-6.13</td></tr><tr><td>NMT</td><td>0.6620</td><td>0.5611</td><td>-10.09</td><td>0.6098</td><td>-5.22</td></tr><tr><td>Mean</td><td>0.7550</td><td>0.7301</td><td>-2.49</td><td>0.7235</td><td>-3.15</td></tr></table>

Table S9. Prompt paraphrasing robustness under zero-shot evaluation. Prompt mean denotes the average AUROC across three semantically equivalent paraphrased prompts. Max drop denotes the largest decrease from the canonical prompt among the three paraphrased prompts.
<table><tr><td>Dataset</td><td>Canonical</td><td>Prompt 1</td><td>Prompt 2</td><td>Prompt 3</td><td>Prompt mean</td><td>Max drop (pp)</td></tr><tr><td>ISRUC</td><td>0.9411</td><td>0.9358</td><td>0.9373</td><td>0.9365</td><td>0.9365</td><td>0.53</td></tr><tr><td>Dreams</td><td>0.9289</td><td>0.9274</td><td>0.9277</td><td>0.9270</td><td>0.9274</td><td>0.19</td></tr><tr><td>Mayo</td><td>0.9352</td><td>0.8696</td><td>0.8329</td><td>0.8910</td><td>0.8645</td><td>10.23</td></tr><tr><td>IEDS</td><td>0.6419</td><td>0.5686</td><td>0.5455</td><td>0.5752</td><td>0.5631</td><td>9.64</td></tr><tr><td>SEE</td><td>0.7145</td><td>0.7438</td><td>0.7377</td><td>0.6814</td><td>0.7210</td><td>3.31</td></tr><tr><td>RatEpilepsy</td><td>0.6319</td><td>0.6403</td><td>0.5401</td><td>0.6958</td><td>0.6254</td><td>9.18</td></tr><tr><td>ADHD-80</td><td>0.6707</td><td>0.7213</td><td>0.7187</td><td>0.7216</td><td>0.7205</td><td>0.00</td></tr><tr><td>ADHD-121</td><td>0.7240</td><td>0.6460</td><td>0.6674</td><td>0.6211</td><td>0.6449</td><td>10.29</td></tr><tr><td>MDD</td><td>0.7425</td><td>0.7078</td><td>0.7097</td><td>0.7239</td><td>0.7138</td><td>3.47</td></tr><tr><td>Schizo-Youth</td><td>0.7208</td><td>0.7188</td><td>0.7230</td><td>0.7307</td><td>0.7242</td><td>0.20</td></tr><tr><td>ADFSU</td><td>0.7458</td><td>0.7952</td><td>0.7404</td><td>0.7415</td><td>0.7590</td><td>0.54</td></tr><tr><td>NMT</td><td>0.6620</td><td>0.5744</td><td>0.5606</td><td>0.5483</td><td>0.5611</td><td>11.37</td></tr></table>

Table S10. Label phrasing robustness under zero-shot evaluation. Label mean denotes the average AUROC across three semantically equivalent alternate label phrasings. Max drop denotes the largest decrease from the canonical label phrasing among the three label variants.
<table><tr><td>Dataset</td><td>Canonical</td><td>Label 1</td><td>Label 2</td><td>Label 3</td><td>Label mean</td><td>Max drop (pp)</td></tr><tr><td>ISRUC</td><td>0.9411</td><td>0.9394</td><td>0.9382</td><td>0.9387</td><td>0.9387</td><td>0.30</td></tr><tr><td>Dreams</td><td>0.9289</td><td>0.9280</td><td>0.9276</td><td>0.9299</td><td>0.9285</td><td>0.13</td></tr><tr><td>Mayo</td><td>0.9352</td><td>0.8831</td><td>0.8310</td><td>0.8484</td><td>0.8541</td><td>10.42</td></tr><tr><td>IEDS</td><td>0.6419</td><td>0.5681</td><td>0.5532</td><td>0.5536</td><td>0.5583</td><td>8.87</td></tr><tr><td>SEE</td><td>0.7145</td><td>0.6715</td><td>0.6787</td><td>0.6811</td><td>0.6771</td><td>4.30</td></tr><tr><td>RatEpilepsy</td><td>0.6319</td><td>0.6389</td><td>0.6000</td><td>0.5714</td><td>0.6034</td><td>6.05</td></tr><tr><td>ADHD-80</td><td>0.6707</td><td>0.6985</td><td>0.7270</td><td>0.6962</td><td>0.7072</td><td>0.00</td></tr><tr><td>ADHD-121</td><td>0.7240</td><td>0.6980</td><td>0.5936</td><td>0.5543</td><td>0.6153</td><td>16.97</td></tr><tr><td>MDD</td><td>0.7425</td><td>0.7271</td><td>0.7442</td><td>0.7542</td><td>0.7418</td><td>1.54</td></tr><tr><td>Schizo-Youth</td><td>0.7208</td><td>0.7647</td><td>0.7526</td><td>0.7738</td><td>0.7637</td><td>0.00</td></tr><tr><td>ADFSU</td><td>0.7458</td><td>0.7263</td><td>0.6325</td><td>0.6947</td><td>0.6845</td><td>11.33</td></tr><tr><td>NMT</td><td>0.6620</td><td>0.6008</td><td>0.6068</td><td>0.6218</td><td>0.6098</td><td>6.12</td></tr></table>

Table S11. Exact prompt paraphrases used in the prompt robustness evaluation.
<table><tr><td>Dataset</td><td>Canonical prompt</td><td>Prompt 1</td><td>Prompt 2</td><td>Prompt 3</td></tr><tr><td>ISRUC</td><td>Which sleep stage does this signal belong to?</td><td>Which sleep stage best describes this signal?</td><td>Identify the sleep stage of this signal.</td><td>Classify the sleep stage for this recording segment.</td></tr><tr><td>Dreams</td><td>Which sleep stage does this signal belong to?</td><td>Which sleep stage best describes this signal?</td><td>Identify the sleep stage of this signal.</td><td>Classify the sleep stage for this recording segment.</td></tr><tr><td>Mayo</td><td>Which epilepsy state does this signal belong to?</td><td>Identify the epilepsy state of this</td><td>Classify the epilepsy state for the</td><td>What epilepsy state best describes this signal?</td></tr><tr><td>IEDS</td><td>Which epilepsy state does this signal belong to?</td><td>recording segment. Identify the epilepsy state of this</td><td>given neural signal. Classify the epilepsy state for the</td><td>What epilepsy state best describes this signal?</td></tr><tr><td>SEE</td><td>Does this signal belong to epilepsy?</td><td>recording segment. Is this EEG segment indicative of epilepsy?</td><td>given neural signal. Determine whether this signal</td><td>Does this signal show evidence of epilepsy?</td></tr><tr><td>RatEpilepsy</td><td>Which epilepsy state does this signal belong to?</td><td>Identify the epilepsy state of this recording segment.</td><td>shows epileptic activity. Classify the epilepsy state for the given neural signal.</td><td>What epilepsy state best describes this signal?</td></tr><tr><td>ADHD-121</td><td>What clinical diagnosis is associated with this signal?</td><td>Which clinical condition label applies to this signal?</td><td>Assign the clinical condition for this recording segment.</td><td>Determine the clinical condition indicated by this signal.</td></tr><tr><td>ADHD-80</td><td>What clinical diagnosis is associated with this signal?</td><td>Which clinical condition label</td><td>Assign the clinical condition for</td><td>Determine the clinical condition</td></tr><tr><td>MDD</td><td>What clinical diagnosis is</td><td>applies to this signal? Which clinical condition label</td><td>this recording segment. Assign the clinical condition for</td><td>indicated by this signal. Determine the clinical condition</td></tr><tr><td></td><td>associated with this signal?</td><td>applies to this signal?</td><td>this recording segment.</td><td>indicated by this signal.</td></tr><tr><td>Schizo-Youth</td><td>What clinical diagnosis is</td><td>Which clinical condition label</td><td>Assign the clinical condition for</td><td>Determine the clinical condition</td></tr><tr><td></td><td>associated with this signal?</td><td>applies to this signal?</td><td>this recording segment.</td><td>indicated by this signal.</td></tr><tr><td>ADFSU</td><td>What clinical diagnosis is</td><td>Which clinical condition label</td><td>Assign the clinical condition for</td><td>Determine the clinical condition</td></tr><tr><td></td><td>associated with this signal?</td><td>applies to this signal?</td><td>this recording segment.</td><td>indicated by this signal.</td></tr><tr><td>NMT</td><td>Is the signal normal or abnormal?</td><td>Determine whether this EEG</td><td>Classify this EEG recording by</td><td>Does this signal indicate an abnormal EEG pattern?</td></tr><tr><td></td><td></td><td>segment is normal or abnormal.</td><td>its clinical status.</td><td></td></tr><tr><td>Table S12. Exact alternate label phrasings used in the label robustness evaluation.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dataset</td><td>Canonical labels</td><td>Label phrasing 1</td><td>Label phrasing 2</td><td>Label phrasing 3</td></tr><tr><td>ISRUC</td><td>Wake; Non-REM Stage 1;</td><td>Wakefulness; N1; N2; N3; REM</td><td>Wake stage; N1 sleep; N2 sleep;</td><td>Awake; NREM Stage 1; NREM</td></tr><tr><td></td><td>Non-REM Stage 2; Non-REM</td><td>sleep</td><td>N3 sleep; Stage REM</td><td>Stage 2; NREM Stage 3; REM</td></tr><tr><td></td><td>Stage 3; Rapid Eye Movement</td><td></td><td></td><td></td></tr><tr><td>Dreams</td><td>Wake; Non-REM Stage 1; Non-REM Stage 2; Non-REM</td><td>Wakefulness; N1; N2; N3; REM</td><td>Wake stage; N1 sleep; N2 sleep;</td><td>Awake; NREM Stage 1; NREM</td></tr><tr><td></td><td>Stage 3; Rapid Eye Movement</td><td>sleep</td><td>N3 sleep; Stage REM</td><td>Stage 2; NREM Stage 3; REM</td></tr><tr><td>Mayo</td><td>Intracranial; Intracranial,</td><td>Intracranial without pathological</td><td>Non-pathological intracranial</td><td>Intracranial non-pathological</td></tr><tr><td></td><td>pathological activity</td><td>activity; Intracranial with</td><td>segment; Pathological</td><td>segment; Intracranial</td></tr><tr><td></td><td></td><td>pathological activity</td><td>intracranial segment</td><td>pathological segment</td></tr><tr><td>IEDS</td><td>Intracranial; Intracranial,</td><td>Intracranial without pathological</td><td>Non-pathological intracranial</td><td>Intracranial non-pathological</td></tr><tr><td></td><td>pathological activity</td><td>activity; Intracranial with</td><td>segment; Pathological</td><td>segment; Intracranial</td></tr><tr><td></td><td></td><td>pathological activity</td><td>intracranial segment</td><td>pathological segment</td></tr><tr><td>SEE</td><td>No; Yes</td><td>No seizure; Seizure</td><td>Non-seizure; Seizure event</td><td>Without seizure; With seizure</td></tr><tr><td>RatEpilepsy</td><td>Intracranial; Ictal; Preictal;</td><td>Interictal; Ictal state; Preictal</td><td>Non-ictal intracranial state;</td><td>Between-seizure state; Seizure</td></tr><tr><td></td><td>Postictal</td><td>state; Postictal state</td><td>Seizure state; Pre-seizure state;</td><td>period; Before-seizure state;</td></tr><tr><td></td><td></td><td></td><td>Post-seizure state</td><td>After-seizure state</td></tr><tr><td>ADHD-121</td><td>Normal; Attention deficit</td><td>Healthy control; ADHD</td><td>Clinically normal control;</td><td>Normal control; Attention-deficit</td></tr><tr><td></td><td>hyperactivity disorder</td><td></td><td>Attention-deficit/hyperactivity disorder</td><td>hyperactivity disorder</td></tr><tr><td>ADHD-80</td><td>Normal; Attention deficit</td><td>Healthy control; ADHD</td><td>Clinically normal control;</td><td>Normal control; Attention-deficit</td></tr><tr><td></td><td>hyperactivity disorder</td><td></td><td>Attention-deficit/hyperactivity</td><td>hyperactivity disorder</td></tr><tr><td></td><td></td><td></td><td>disorder</td><td></td></tr><tr><td>MDD</td><td>Normal; Major depressive</td><td>Healthy control; MDD</td><td>Clinically normal control; Major Normal control; Depressive</td><td></td></tr><tr><td></td><td>disorder</td><td></td><td>depression</td><td>disorder</td></tr><tr><td>Schizo-Youth</td><td>Normal; Schizophrenia</td><td>Healthy control; Schizophrenia</td><td>Clinically normal control;</td><td>Normal control; Schizophrenic</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>diagnosis</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Schizophrenia spectrum disorder disorder</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ADFSU</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>Healthy control; Alzheimer</td><td>Cognitively normal; AD</td><td></td></tr><tr><td></td><td>Normal; Alzheimer&#x27;s disease</td><td></td><td></td><td>Normal control; Alzheimer-type</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>disease</td><td>dementia</td><td>dementia</td></tr><tr></table>