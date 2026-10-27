# Decision Model Factory: Building Specialized Sovereign Decision Models

Yuma Ichikawa<sup>1,2</sup>

Decision needs vary with each application, language, and deployment environment. The Decision Model Factory turns those needs into specialized, sovereign models on demand. A brief recipe guides the training, calibration, verification, and evaluation of a selected backbone, delivering a deployable decision model under the operator’s control. The same workflow creates a new specialist whenever the requirements change. Across Japanese decision tasks, compact deployment, and long-context passage ranking, five factory-built models improve upon their backbones; three surpass the strongest results in a fixed public Japanese benchmark comparison. A dedicated runner computes only permitted-answer probabilities and reuses shared context, more than doubling serving speed on matched multi-question workloads with essentially unchanged decision scores. Together, the factory and runner make fast, task-specific decision models practical to build and deploy whenever needed.

![](images/31c9dbb00a08646c53638a890a8bc52abbcf2cbc16f9080b6f53f61ccca1430c.jpg)  
Figure 1. Japanese decision quality on the System One Mosaic Benchmark (S1MB). Red marks show factory-built DF models, hollow circles their backbones before specialization, and black marks public results [1].

## 1 Introduction

Many workflows require a decision, not a paragraph: does a clause violate policy, where should a ticket be routed, or which passage supports an answer? Large language models (LLMs) can reason over the evidence, but free-form generation adds decoding and parsing without directly yielding calibrated probabilities for the permitted outcomes. Decision models [2] instead return a distribution over those outcomes in one forward pass, so applications can act on confident predictions and refer uncertain cases for review.

![](images/1b21aa048221a6a5928f017012cc9cdfb4b8031aff8fb1473369a2a358073d1a.jpg)  
Figure 2. Decision Model Factory. A model recipe passes through five stages to produce a verified model package and a record of the build.

Table 1. The DF family and S1MB Japanese Task Avg before and after specialization. JA denotes Japanese-focused recipes; DF-Long uses about 4 billion active parameters per token.
<table><tr><td>Model</td><td>Gemma 4 backbone Recipe</td><td></td><td>Before</td><td>After</td></tr><tr><td>DF-JA-Large</td><td>31B</td><td>Japanese-focused</td><td>58.99</td><td>67.63</td></tr><tr><td>DF-Large</td><td>31B</td><td>General</td><td>58.99</td><td>66.98</td></tr><tr><td>DF-Long</td><td>26B-A4B</td><td>General, long inputs</td><td>51.58</td><td>63.57</td></tr><tr><td>DF-Small</td><td>E4B</td><td>General, distilled</td><td>31.97</td><td>52.96</td></tr><tr><td>DF-JA-Small</td><td>E4B</td><td>Japanese-focused, distilled</td><td>31.97</td><td>52.49</td></tr></table>

The challenge is to make the correct decision model available for each job. Languages, policies, answer labels, document lengths, and serving budgets vary across applications. Japanese contract review and ticket routing on a central processing unit (CPU), for example, call for diferent specializations. Organizations may also need to retain control of their training data, model weights, and deployment environment. We use sovereign for a model that the operator can build and serve under its own control. Repeating a separate training and deployment project for every requirement is dificult to sustain.

Decision Model Factory makes specialization a reproducible build that can be launched whenever a new need arises. Drawing on the model-factory concept [3], a recipe specifies the backbone, decision task, training inputs, and serving target. The Factory runs training, calibration, export verification, and evaluation, then delivers a verified package for graphics processing unit (GPU) or CPU serving (Figure 2). Its stages can run unattended on the operator’s computing system. When a language, task, or deployment constraint changes, the recipe changes, not the pipeline.

We demonstrate this approach with five models from three Gemma 4 backbones [4] (Table 1). Their recipes produce Japanese specialists, compact distilled variants, and a long-context model that also ranks passages. Three exceed the strongest public score in our fixed Japanese comparison. The accompanying decision runner serves multiple judgments over shared evidence more than twice as fast as the generalpurpose path in matched workloads. Together, they ofer a practical way to build and deploy the next specialized decision model as needs evolve.

## 2 From a Language Model to a Decision Model

A decision request consists of a state s (e.g., a ticket or document) and a question q that specifies the task instructions and a set of K possible answers. We consider three answer types: Noul for binary yes-or-no decisions, Choice for selecting among named alternatives, and Score for ratings on an ordered scale with values $v _ { 1 } < \cdots < v _ { K }$ . A fixed template converts (s, q) into a prompt x that presents the state first and assigns each answer option a code $c _ { k } \ ( \mathrm { e . g . } , \mathsf { A } \ \mathrm { o r } \ \mathsf { B } )$ . Each code corresponds to a single token in the backbone model’s vocabulary.

Let h(x) denote the backbone’s final hidden state at the last position of the prompt, and let $w _ { k }$ denote the row of its output matrix corresponding to code $c _ { k }$ . We compute the logit for option k as $z _ { k } ( x ) = g ( w _ { k } ^ { \top } h ( x ) )$ , where g represents any additional output transformation applied by the backbone. Following the option-logit readout of Geuter et al. [5], we define the decision distribution as

$$
p _ { \theta } ( k \mid s , q ) = \frac { \exp \bigl ( z _ { k } ( x ) / \tau _ { a } \bigr ) } { \sum _ { j = 1 } ^ { K } \exp \bigl ( z _ { j } ( x ) / \tau _ { a } \bigr ) } , \ k = 1 , \ldots , K ,\tag{1}
$$

where θ denotes the model parameters, a specifies the answer type, and $\tau _ { a } > 0$ represents the corresponding calibration temperature. For Score answers, we additionally report the expected value $\textstyle \sum _ { k } p _ { \theta } ( k \mid s , q ) v _ { k }$ This option-based readout operates over the allowed answers and directly reuses the backbone’s tokenizer and output layer, requiring no additional classification head.

We adapt the backbone using low-rank adaptation (LoRA) [6]. For each selected weight matrix $W _ { 0 } \in \mathbb { R } ^ { m \times n }$ , we keep $W _ { 0 }$ frozen and learn a low-rank update $\Delta W = B A$ , where $B \in \mathbb { R } ^ { m \times r } , A \in \mathbb { R } ^ { r \times n }$ and $r \ll \operatorname* { m i n } ( m , n )$ . During training, we use the option distribution before temperature calibration, together with a target distribution y over the allowed answers. The training loss is

$$
\mathcal { L } = - \sum _ { k = 1 } ^ { K } y _ { k } \log p _ { \theta } ( k \mid s , q ) + \lambda \mathbf { 1 } [ a = \operatorname { S c o r e } ] \times \left| \sum _ { k = 1 } ^ { K } ( p _ { \theta } ( k \mid s , q ) - y _ { k } ) v _ { k } \right| .\tag{2}
$$

The indicator activates the second term only for Score tasks, where it penalizes errors in the expected value with a weight of $\lambda \geq 0$ . The target distribution y is either a one-hot encoding of the ground-truth answer or a distillation mixture $y = ( 1 - \bar { \beta } ) y ^ { \mathrm { l a b e l } } + \beta p ^ { \mathrm { t e a c h e r } }$ , where p<sup>teacher</sup> denotes the teacher’s distribution over the allowed answers and $\beta \in [ 0 , 1 ]$ controls its contribution [7].

After training, the Factory selects the update strength

$$
W ( { \boldsymbol { \sigma } } ) = W _ { 0 } + { \boldsymbol { \sigma } } \Delta W , ~ { \boldsymbol { \sigma } } \in \Sigma ,\tag{3}
$$

by evaluating each candidate in Σ on held-out development decisions. At the selected strength, the Factory then fits a separate calibration temperature $\tau _ { a }$ for each answer type [8]. Finally, it verifies that the exported package reproduces the selected readout within a predefined tolerance.

## 3 Decision Model Factory

Each build begins with a recipe specifying the backbone model, decision dataset, optional teacher model, adapter configuration, and deployment formats. The preparation stage validates the answer types and sets aside development examples. Training then learns the low-rank update using the objective in Equation (2), followed by the selection of the adapter strength and calibration temperatures. During export, the selected update is merged into the backbone, and the resulting model is verified to produce consistent readouts between training and serving. For mixture-of-experts (MoE) backbones, this verification is performed separately for each expert. Finally, the runner evaluates the exported model on public benchmarks. The resulting package includes the model weights, prompt template, option codes, and calibration temperatures required for deployment.

The pipeline supports unattended execution and can resume from interruptions. Each stage records the recipe, code, data hashes, and selected configurations, ensuring that evaluation results can be traced to the exact build that produced them. These results can then inform subsequent recipes, while all builds follow the same training and serving workflow.

We develop five model variants using distinct recipes. The Japanese-focused models difer in their training data mixtures and adapter ranks, while the compact models learn from a larger decision teacher through distillation. DF-Long additionally incorporates long documents and relevance judgments into its training. Training examples include demonstrations embedded in the question [9], distracting content surrounding relevant evidence [10], and instructions embedded in documents that the model must interpret as content rather than commands [11]. Despite these diferences, all five models share a common build pipeline and serving interface.

## 4 A Purpose-Built Decision Runner

The General-purpose language model serves by computing logits over the entire vocabulary, even when a decision only requires a small set of allowed answers. Similarly, processing multiple questions about the same state independently can lead to redundant computations over shared evidence. Our decision runner addresses both ineficiencies.

Option-only output. Instead of projecting onto the full vocabulary, the runner projects the final hidden state h(x) onto only the output-matrix rows $w _ { 1 } , \ldots , w _ { K }$ corresponding to the allowed answers. It then applies the backbone’s output transformation and computes the decision probabilities using Equation (1) and the exported calibration temperatures. Consequently, the output-projection cost scales with the number of allowed answers rather than the vocabulary size, while the resulting probabilities remain equivalent to those of the full-vocabulary readout, up to floating-point rounding.

Shared-state execution. When multiple questions share the same state, the runner computes the attention keys and values for that state once and reuses the cached prefix across questions. For the MoE backbone, it further captures the prompt-processing pass in a GPU execution graph to reduce kernel-launch overhead. The CPU implementation similarly accesses only the output-matrix rows required for the allowed answers. For backbones that support reusing earlier attention states in later layers, the runner can also skip unnecessary computations at intermediate prompt positions [4].

## 5 Experiments

## 5.1 Evaluation Setting

Models and tasks. We evaluate the five models listed in Table 1, with training configurations detailed in Appendix A. The evaluated revisions of the System One Mosaic Benchmark (S1MB) [12] comprise 8,067 judgments across 27 Japanese benchmarks and 26,269 judgments across 137 English benchmarks. Each benchmark score is adjusted for its task-specific trivial baseline and clipped to [0, 1]. We compute Task Avg by first averaging scores within each answer type, then taking the equally weighted average across the three types and multiplying the result by 100.

DF-Long supports inputs of up to 262,144 tokens. We evaluate its passage-ranking performance on Japanese Question Answering with Retrieval Augmentation (JQaRA) [13] and the Japanese tracks of MIRACL [14] and Multilingual Long-Document Retrieval (MLDR) [15]. Candidates are ranked by the probability assigned to the afirmative option in a Noul relevance judgment, following an approach analogous to the relevance-token scoring used by monoT5 [16]. We assess robustness against embedded instructions designed to redirect the model’s answer, using 300 held-out states for each of English, Japanese, and Portuguese. We also evaluate calibration on 600 development decisions using expected calibration error (ECE), defined as the weighted average of the absolute diferences between confidence and accuracy across bins.

Comparisons. For S1MB, we use the oficial evaluator and 8-bit model packages. To isolate the efects of specialization, we evaluate each backbone before adaptation using the same readout and packaging as its corresponding DF model. Our comparison with publicly available results is based on a fixed snapshot of 34 complete evaluations retrieved on 18 October 2026 [1]. For passage ranking, we compare DF-Long against Ruri v3 [17] and bge-reranker-v2-m3 [18], using the same candidate passages for all methods.

Serving measurements. We measure serving performance on a single NVIDIA B200 GPU or 16 Intel Xeon CPU cores, using freshly generated synthetic support tickets. All reported timings are medians over repeated batches after warm-up. In the runner ablation, we hold the serving engine and model weights fixed to isolate the efects of the runtime optimizations. Cross-model latency and bulk-throughput experiments use the configurations specified in their respective figures.

## 5.2 Results

Decision quality. Specialization consistently improves the Japanese Task Avg across all backbones (Figure 1). With approximately 4 billion active parameters per token, DF-JA-Large, DF-Large, and DF-Long achieve scores of 67.63, 66.98, and 63.57, respectively, all surpassing the highest score of 60.05 in the fixed public snapshot. The distilled small models exhibit the largest improvements: DF-Small and DF-JA-Small increase their scores from 31.97 to 52.96 and 52.49, respectively, both exceeding the published score of 49.77 achieved by Saihai 8B [19]. Specialization also yields substantial gains on English tasks, improving DF-Long from 52.14 to 63.55 and DF-Small from 36.49 to 47.30.

Long-context serving and ranking. DF-Long extends the same decision interface to passage retrieval and long-context evidence processing (Table 2). Despite activating substantially fewer parameters, it slightly outperforms DF-Large on English decision tasks. It also achieves higher passage-ranking scores than both retrieval baselines across all three Japanese benchmarks. In the robustness evaluation, DF-Long successfully ignores 899 out of 900 embedded instructions across three languages. Its expected calibration error (ECE) is 0.024 on the development decisions used for temperature fitting. DF-Long also handles lengthy inputs eficiently, answering a question about a 27K-token document in just 0.41 s on the evaluated GPU.

Table 2. DF-Long across tasks. S1MB uses 8-bit weights; other results use bfloat16 (BF16) on one NVIDIA B200. nDCG@10 is normalized discounted cumulative gain at rank ten (×100). Ruri: Ruri v3; BGE: bge-reranker-v2-m3. Calibration uses development data.
<table><tr><td>Benchmark or test</td><td>Metric</td><td>DF-Long</td><td>Comparison or scope</td></tr><tr><td>S1MB English</td><td>Task Avg</td><td>63.55</td><td>DF-Large: 62.31</td></tr><tr><td>S1MB Japanese</td><td>Task Avg</td><td>63.57</td><td>Best public snapshot result: 60.05</td></tr><tr><td>JQaRA</td><td>nDCG@10</td><td>90.1</td><td>Ruri: 86.9; BGE: 67.8</td></tr><tr><td>MIRACL Japanese</td><td>nDCG@10</td><td>93.7</td><td>Ruri: 91.8; BGE: 90.1</td></tr><tr><td>MLDR Japanese</td><td>nDCG@10</td><td>98.9</td><td>Ruri: 97.7; BGE: 97.6</td></tr><tr><td>Embedded instructions</td><td>Ignored (%)</td><td>99.9</td><td>899 of 900 trials; three languages</td></tr><tr><td>Development decisions</td><td>ECE</td><td>0.024</td><td>600 decisions</td></tr></table>

Table 3. Matched serving comparison for DF-Long in BF16 on one NVIDIA B200. The general-purpose path uses full-vocabulary readout and one request per question. Speedup is the latency ratio, or the throughput ratio for the concurrent workload.
<table><tr><td>Workload</td><td>General path</td><td>Decision runner</td><td>Speedup</td></tr><tr><td>Eight questions about one state</td><td>71.7 ms</td><td>29.4 ms</td><td>2.44×</td></tr><tr><td>Eight states, one question each</td><td>68.8 ms</td><td>30.9 ms</td><td>2.23×</td></tr><tr><td>Four states, eight questions each</td><td>364 decisions/s</td><td>570 decisions/s</td><td>1.57×</td></tr><tr><td>One question</td><td>16.3 ms</td><td>15.0 ms</td><td>1.09×</td></tr></table>

Serving eficiency. The runner delivers its largest speedups when processing multiple decisions (Table 3). For eight questions sharing the same state, latency decreases from 71.7 ms to 29.4 ms, yielding a 2.44× speedup. Even when processing eight independent states, the runner achieves a 2.23× speedup. With four concurrent states and eight questions per state, throughput reaches 570 decisions per second.

DF-Long achieves a favorable balance between decision quality and inference latency (Figure 3). Its latency approaches that of DF-Small while delivering substantially higher scores on both Japanese and English tasks. In a separate cross-model measurement, DF-Long processes an eight-question batch in approximately 28 ms, which is about 40% faster than DF-Large. Further tuning of concurrency and the batched-token budget increases bulk throughput to approximately 2,328 decisions per second for the general-recipe E4B model and 380 decisions per second for DF-Large on a single GPU (Figure 4). On 16 CPU cores, the E4B model achieves 11.2 decisions per second.

Importantly, these runtime optimizations preserve the decision readout. Using identical DF-Long BF16 weights, the runner and standard inference path difer by at most 0.2 points in their end-to-end S1MB English, S1MB Japanese, and JQaRA scores. The corresponding S1MB results using separate 8-bit model packages are reported in Table 2.

## 5.3 Discussion

These results highlight the importance of task-specific specialization alongside backbone scale. Despite activating only about 4 billion parameters per token, DF-Long outperforms its unspecialized 31 billion parameter backbone on Japanese decision tasks. Similarly, the compact distilled models achieve improvements of more than 20 points without increasing their backbone size. By incorporating long-context and relevance-judgment data, DF-Long further extends the range of supported applications while retaining the same decision interface and serving infrastructure.

The decision runner makes these specialized models practical to deploy. Shared-prefix execution is particularly efective when multiple questions refer to the same evidence, while option-only output reduces computation even for independent requests.

## 6 Related Work

Model factories aim to streamline model development through reusable components and versioned experiments [3]. Jev [2] introduces a decision interface based on probabilities assigned to predefined answers. Li and Wagle [20] investigates structured answer-sufix scoring, while Geuter et al. [5] combines

DECISION QUALITY AND LATENCY English decisions  
![](images/1b45c6158021006bbbafca94180c4126c4836ae6523c687212ab0e1ff3173c01.jpg)

![](images/d68a9f77f156d15f3201b5ca450d05554840580d2aa00535025d5fbc87c85e9f.jpg)  
Latency for eight questions about one state (ms)  
Figure 3. S1MB Task Avg versus eight-question latency. Quality scores use 8-bit model packages; latency uses BF16 and the decision runner on one NVIDIA B200. Timings are separate from Table 3.  
SERVING THROUGHPUT GPU · NVIDIA B200

![](images/aace61f5475ecf13992b0594519c690135a3400c6107601c1f3f8d71170da050.jpg)  
CPU · 16 Xeon cores  
64 questions per batch

![](images/816569dfc51a56f096ab3f4d6757bea9d03b361db2544d68be40ff050432649f.jpg)  
Bar length: tuned/default rate · Black tick: 1.00× · Right labels: decisions/s

Figure 4. Serving throughput after tuning concurrency and the batched-token budget. Each bar compares tuned throughput with its own default setting; labels give the measured decisions per second.

option-token logits with low-rank fine-tuning to develop open decision models. For Japanese decision tasks, Tateno [19] presents a specialized model and its benchmark evaluation. Building on these directions, the Decision Model Factory provides a unified workflow for the reproducible specialization, verification, and eficient deployment of models that share a common probabilistic decision interface.

## 7 Conclusion and Outlook

We presented the Decision Model Factory, a framework for building and deploying specialized, operatorcontrolled decision models as application requirements evolve. Through five DF models, we demonstrated that a single workflow can support Japanese-focused, distilled, and long-context specializations without requiring separate training or serving systems. The improvements in decision quality, together with the eficiency gains from the dedicated runner, demonstrate the practical value of combining model specialization with decision-specific inference. The recipe-based workflow also provides a path toward supporting additional tasks and languages while keeping training data, model weights, and deployment under the operator’s control.

A promising direction for future work is to integrate PHOTON [21] into the Decision Model Factory. PHOTON’s hierarchical latent-stream architecture is designed to support eficient long-context and multi-query processing, making it particularly relevant to applications that require multiple decisions over shared documents. Incorporating PHOTON into the existing specialization, verification, and serving workflow could enable more memory-eficient processing of large numbers of parallel judgments. We leave this integration for future work.

## References

[1] Yuichi Tateno. s1mb-result: Public evaluation results. Hugging Face dataset, 2026. URL https: //huggingface.co/datasets/hotchpotch/s1mb-result. Snapshot 4c0e5d5, retrieved 18 October 2026.

[2] Diogo Almeida. Introducing System One models & Jev. TypeSafe AI Blog, September 2026. URL https://typesafe.ai/blog/introducing-system-one-models-and-jev. 15 September 2026.

[3] Joe Rowell and Eiso Kant. The hidden engineering behind foundation model building. Poolside Blog, July 2025. URL https://poolside.ai/blog/introducing-the-model-factory. 17 July 2025.

[4] Gemma Team. Gemma 4 Technical Report. arXiv:2607.02770, 2026.

[5] Jonathan Geuter, Andrew Cheng, and David Alvarez-Melis. Releasing the inner Jev: A simple recipe for open decision models. arXiv:2610.17205, 2026.

[6] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[7] Geofrey Hinton, Oriol Vinyals, and Jef Dean. Distilling the knowledge in a neural network. arXiv:1503.02531, 2015.

[8] Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1321–1330. PMLR, 2017. URL https://proceedings.mlr.press/ v70/guo17a.html.

[9] Tom Brown, Benjamin Mann, Nick Ryder, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877– 1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

[10] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl\_a\_00638.

[11] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. arXiv:2302.12173, 2023.

[12] Yuichi Tateno. S1MB: System One Mosaic Benchmark. Software and evaluation datasets, 2026. URL https://github.com/hotchpotch/S1MB. Evaluator revision a77c30c; Japanese dataset revision 59b66dd.

[13] Yuichi Tateno. JQaRA: Japanese question answering with retrieval augmentation. Hugging Face dataset, 2024. URL https://huggingface.co/datasets/hotchpotch/JQaRA.

[14] Xinyu Zhang, Nandan Thakur, Odunayo Ogundepo, Ehsan Kamalloo, David Alfonso-Hermelo, Xiaoguang Li, Qun Liu, Mehdi Rezagholizadeh, and Jimmy Lin. MIRACL: A multilingual retrieval dataset covering 18 diverse languages. Transactions of the Association for Computational Linguistics, 11:1114–1131, 2023. doi: 10.1162/tacl\_a\_00595.

[15] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding: Multilinguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.findings-acl.137.

[16] Rodrigo Nogueira, Zhiying Jiang, Ronak Pradeep, and Jimmy Lin. Document ranking with a pretrained sequence-to-sequence model. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 708–718. Association for Computational Linguistics, 2020. doi: 10.18653/v1/ 2020.findings-emnlp.63.

[17] Hayato Tsukagoshi and Ryohei Sasano. Ruri: Japanese general text embeddings. arXiv:2409.07737, 2024. Ruri v3 reranker model card: https://huggingface.co/cl-nagoya/ruri-v3-reranker-310m.

[18] Beijing Academy of Artificial Intelligence. bge-reranker-v2-m3. Hugging Face model card, 2024. URL https://huggingface.co/BAAI/bge-reranker-v2-m3. Accessed 24 October 2026.

[19] Yuichi Tateno. Release of the Japanese decision model “saihai” and a Japanese decision model benchmark. A Day in the Life, October 2026. URL https://secon.dev/entry/2026/10/15/ 100000-saihai-japanese-decision-benchmark/. 15 October 2026. In Japanese; title translated.

[20] Yinheng Li and Justin Wagle. LLM-as-Jev: LLMs are already Jev-style decision models—when and how to fine-tune them. arXiv:2610.02076v3, 2026.

[21] Yuma Ichikawa, Naoya Takagi, Takumi Nakagawa, Yuzi Kanazawa, and Akira Sakai. PHOTON: Hierarchical autoregressive modeling for lightspeed and memory-eficient language generation. In

Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 38365–38383. Association for Computational Linguistics, July 2026. doi: 10.18653/v1/2026.acl-long.1778.

Table 4. Training settings for the DF models. A dash indicates that no teacher is used.
<table><tr><td></td><td>DF-JA-Large</td><td>DF-Large</td><td>DF-Long</td><td></td><td>DF-Small DF-JA-Small</td></tr><tr><td>Adapter rank r</td><td>16</td><td>32</td><td>32</td><td>16</td><td>16</td></tr><tr><td>Learning rate</td><td>10⁻4</td><td>10⁻4</td><td>10⁻4 / 5 × 10−5</td><td>10⁻4</td><td>10-4</td></tr><tr><td>Effective batch size</td><td>32</td><td>32</td><td>32 / 16</td><td>16</td><td>16</td></tr><tr><td>Maximum prompt (tokens)</td><td>8,192</td><td>8,192</td><td>8,192 /32,768</td><td>8,192</td><td>8,192</td></tr><tr><td>Teacher weight  $\beta$ </td><td></td><td></td><td></td><td>0.5</td><td>0.5</td></tr></table>

## A Main Hyperparameters

For all models, we train LoRA adapters on the attention and feed-forward projections for one epoch, including the fused expert weights of Gemma 4 26B-A4B. All training runs use a 5% warm-up fraction. The Score loss weight is fixed at λ = 1, and the candidate adapter strengths are $\Sigma = \{ 0 . 5 , 0 . 7 5 , 1 \}$ . Table 4 summarizes the hyperparameters that vary across builds. For DF-Long, paired values correspond to the initial and long-input training stages, respectively. Efective batch size denotes the number of training decisions per optimizer step.

For the evaluated Gemma 4 backbones, the output transformation described in Section 2 uses logit soft-capping: $z _ { k } ( x ) = c \mathrm { t a n h } \big ( w _ { k } ^ { \top } h ( x ) \big / c \big )$ with $c = 3 0$ . The runner applies the same transformation to the selected option logits. Of 18 DF-Long output-projection comparisons in BF16 precision, 17 produce bit-identical logits. The remaining comparison difers by only one BF16 rounding step, with probability diferences below $2 \times 1 0 ^ { - 8 }$ . For Gemma 4 E4B, the final 18 layers reuse attention states computed in earlier layers, enabling the prompt-position optimization described in Section 4.