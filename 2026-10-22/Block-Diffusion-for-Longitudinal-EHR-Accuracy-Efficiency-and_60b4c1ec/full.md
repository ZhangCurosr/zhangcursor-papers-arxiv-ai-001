# Block Diffusion for Longitudinal EHR: Accuracy, Efficiency, and Representation Trade-offs

Ashvin Gupta<sup>1</sup> Joshua Placidi<sup>1</sup> Brendan Delaney<sup>1</sup> Alessandra Russo<sup>1</sup>

<sup>1</sup>Imperial College London

## Abstract

Autoregressive (AR) models have proved effective for zero-shot prediction from longitudinal electronic health records, but impose strictly sequential generation on trajectories containing temporally clustered and interdependent clinical events. We investigate block-based masked diffusion (BD) as an alternative on MIMIC-IV under matched tokenisation, context length, model capacity, and evaluation conditions. Across block sizes, BD exhibits a clear quality–efficiency trade-off, with small-block configurations retaining much of the predictive performance of AR while enabling substantially faster inference. A representative B=8 configuration achieves approximately 1.43× faster evaluation. Although generative rollout remains below AR in most settings, frozen linear probes on BD and AR representations achieve near-identical performance across clinical prediction tasks, indicating that BD learns comparably informative clinical representations. Increasing the denoising budget yields diminishing returns, while alternative decoding strategies provide limited gains. Overall, our results show that block diffusion is a viable and computationally attractive approach to longitudinal EHR modelling, with further improvements in iterative decoding offering a promising route toward closing the remaining generative performance gap.

## 1 Introduction

Foundation models for structured electronic health record (EHR) data have predominantly adopted autoregressive (AR) sequence modelling. By generating possible future patient trajectories, these models have demonstrated strong zero-shot performance across clinical prediction tasks, in some settings approaching or exceeding task-specific supervised baselines [1, 2, 3, 4]. However, AR modelling imposes a strict left-to-right factorisation with each event generated conditionally on all preceding events before subsequent events can be considered. Longitudinal EHRs are only partly sequential in this sense. Clinical activity often occurs in temporally concentrated bursts, making groups of nearby events a natural target for partially parallel generation (Appendix A.9), and in our MIMIC-IV data, 47.4% of recorded timestamps contain at least eight tokens (Appendix A.4). This strict factorisation also carries an inference cost, as token-by-token decoding limits parallelism and requires sequential computation for each generated event. This motivates generative objectives that relax strict next-token prediction and instead allow groups of clinical events to be modelled jointly.

Discrete diffusion language models provide one such alternative by learning to reconstruct corrupted tokens through iterative denoising [5]. These models have shown strong generative and representationlearning performance in other structured scientific domains, including protein sequence modelling and molecular graph generation [6, 7]. Block diffusion (BD) introduces an AR factorisation across blocks while denoising tokens jointly within each block, enabling partial parallelism while retaining causal conditioning on previously generated context [8]. Prior work has shown a quality–parallelism trade-off: larger blocks allow more parallel generation but generally degrade likelihood and generation quality, whereas smaller blocks more closely approach AR performance. Discrete diffusion has previously been applied to EHRs for synthetic data generation [9, 10], but its behaviour relative to strong AR foundation models for zero-shot clinical prediction remains less well understood.

In this work, we conduct a controlled empirical comparison between AR and block-diffusion modelling of tokenised MIMIC-IV patient trajectories. We ask three questions: (1) how do block size and denoising budget affect the trade-off between predictive performance and inference efficiency; (2) when diffusion rollout underperforms AR, does the gap arise from poorer learned representations or from the iterative generation process; and (3) can alternative decoding strategies recover performance lost during blockwise generation?

Our contributions are:

1. We show that block diffusion is a competitive alternative for longitudinal EHR prediction, achieving rollout performance close to AR while providing up to a 1.43× evaluation speedup in our primary operating regime.

2. We show that AR and block-diffusion models encode similarly predictive clinical information, with frozen linear probes achieving near-identical downstream performance across tasks despite a larger gap during generative rollout.

3. We provide a systematic study of how block size and denoising budget affect the quality– efficiency trade-off of block diffusion on EHR sequences, and find that alternative decoding strategies are not sufficient to close the remaining performance gap to AR.

## 2 Methods

## 2.1 Models and generative prediction

We represent each longitudinal EHR as a discrete token sequence $\boldsymbol { x } = ( x _ { 1 } , \dots , x _ { L } )$ , where each token corresponds to a clinical event, quantised measurement, demographic attribute, or discretised time interval from a vocabulary V. Both AR and BD models operate on this same temporal sequence.

We compare an AR language model with a capacity-matched BD model. The AR baseline is a GPT-2–style decoder with causal self-attention [11], trained using standard next-token cross-entropy:

$$
\mathcal { L } _ { \mathrm { A R } } = - \sum _ { i = 1 } ^ { L - 1 } \log p _ { \theta } ( x _ { i + 1 } \mid x _ { \le i } )\tag{1}
$$

For BD, we follow BD3-LM [8]. The trajectory is partitioned into blocks, with causal conditioning across blocks and masked diffusion within each active block. For a clean block $x _ { 0 }$ , we sample $t \sim \mathcal { U } ( 0 . 0 0 1 , 0 . 9 9 9 )$ and obtain $x _ { t }$ by retaining each token independently with probability $\alpha _ { t } =$ cos $( \pi t / 2 )$ and replacing it with [MASK] otherwise. The model is trained using the continuous-time masked diffusion objective:

$$
\mathcal { L } _ { \mathrm { M D L M } } = - \mathbb { E } _ { x _ { 0 } , t , x _ { t } } \left[ \frac { - \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \sum _ { i = 1 } ^ { L } \mathbf { 1 } \left[ x _ { t } ^ { i } = \left[ \mathrm { M A S K } \right] \right] \log p _ { \theta } \left( x _ { 0 } ^ { i } \mid x _ { t } , x ^ { ( < k ) } \right) \right]\tag{2}
$$

where $x ^ { ( < k ) }$ denotes the clean preceding blocks. Independent noise levels are sampled across blocks.

At inference time, both models use KV caching [12]. AR generates token by token, whereas block diffusion initialises each new block as fully masked, iteratively denoises it, and then commits the completed block to the cache. For zero-shot clinical prediction, generation continues until the model predicts one of the preconfigured end tokens, or a maximum number of generation steps is reached. We perform 20 Monte Carlo rollouts per patient and use the fraction in which the positive endpoint occurs first as the prediction score. We report AUROC, AUPRC, and wall-clock evaluation time.

## 2.2 Data and experiments

We use MIMIC-IV v2.2 [13] and follow the tokenisation framework proposed in Renc et al. [2], with further details in the paper. We evaluate ICU mortality, ICU readmission, ICU admission, and hospital mortality. To isolate the effect of the generative formulation, AR and diffusion models use identical tokenisation, data splits, and context length of 1024, with matched capacity: 6 transformer layers, hidden dimension 768, 12 attention heads, rotary positional embeddings, and the same optimizer schedule. We train separate BD models with block sizes $B \in \{ 2 , 4 , 8 , \bar { 1 6 } , 3 2 , 6 4 , 1 2 8 \}$ and retain ICU mortality as the primary task for ablations.

We first study how within-block parallelism affects predictive performance and inference efficiency. For the primary block-size comparison, we set the number of reverse steps to $S = B / 2$ , fixing the nominal denoising budget at $S / B = 0 . 5$ denoiser evaluations per generated token while varying the degree of joint within-block prediction. To separate block size from the amount of iterative refinement, we perform fixed-block denoising sweeps for $B = 4 , B = 8 ,$ , and $B = 1 2 8$ . These experiments test whether additional reverse steps can recover performance lost at larger block sizes.

We next test whether differences in generative performance are already reflected in the learned representations. We freeze the AR and $B = 8$ diffusion backbones and fit linear classifiers to their final-token hidden representations for all four clinical outcomes. This removes the generative decoding procedure and directly measures the predictive information encoded by each model. Mean pooling yielded lower performance for both models.

Our default BD decoder uses ancestral random unmasking according to the reverse masking schedule. Let t > s denote consecutive reverse times, with $\alpha _ { t }$ the token-retention probability at time t. Each currently masked position is revealed with probability $\begin{array} { r } { p _ { \mathrm { u n m a s k } } = \frac { \alpha _ { s } - \hat { \alpha } _ { t } } { 1 - \alpha _ { t } } . \mathrm { A t } \ : B \stackrel { \cdot } { = } 8 , S = 4 , } \end{array}$ we additionally evaluate two inference-time alternatives. Confidence-based unmasking preserves the same expected reveal budget: if $n _ { \mathrm { m a s k } }$ positions remain masked, we reveal $k \approx n _ { \mathrm { m a s k } } p _ { \mathrm { \iota } }$ unmask positions, selecting those whose sampled token receives the highest model probability [14, 15]. ReMDM-style remasking allows previously revealed positions to be masked and reconsidered during subsequent reverse steps [16]. On non-final steps, revealed positions are remasked independently with probability $\begin{array} { r } { \sigma = \operatorname* { m i n } \left( \eta , \frac { 1 - \alpha _ { s } } { \alpha _ { t } } \right) } \end{array}$ , where η controls the maximum remasking rate and we evaluate $\eta \in \{ 0 . 0 5 , 0 . 1 , 0 . 2 \}$

## 3 Results

BD provides competitive generative performance across the four downstream tasks, although AR remains stronger on three of four outcomes (Figure 1). Hospital mortality is the exception, but an increase in AUROC comes at the cost of AUPRC (Appendix A.6). Importantly, this gap largely disappears when evaluating the learned representations directly with frozen linear probes achieving near-identical AUROC for AR and BD across tasks. This indicates that BD captures similarly predictive clinical information, and suggests that much of the remaining gap emerging during rollout rather than representation learning.

Under a fixed denoising budget, smaller blocks provide the strongest quality–efficiency trade-off. The BD configuration of $B \overset { \cdot } { = } 8 , S = 4$ remained approximately 0.03 AUROC below AR, while achieving a 1.43× evaluation speedup. Performance generally declines at larger block sizes, and very large blocks eventually lose the latency advantage despite requiring fewer sequential forward passes, reflecting the increased cost of each blockwise denoising step.

Increasing the number of denoising steps initially improves predictive performance, but gains saturate beyond a moderate budget while evaluation cost continues to rise. Alternative decoding strategies do not materially change this picture. ReMDM-style revision produces only small, metric-dependent changes, while confidence-based unmasking degrades AUROC (Appendix A.9).

## 4 Discussion

We hypothesised that masked diffusion might be well suited to longitudinal EHR modelling because clinical activity often occurs in temporally concentrated bursts, where the exact ordering of nearby events may be less informative than their joint occurrence. Our results support block diffusion as a practical alternative to strictly autoregressive generation. Small-block BD configurations retain much of the predictive performance of AR while providing faster evaluation, with up to a 1.43× speedup in our primary operating regime. This benefit is not monotonic with block size, however. Large blocks reduce sequential depth but increase per-step compute, producing a U-shaped latency curve and reproducing the quality–parallelism trade-off observed in prior BD work [8].

![](images/a152f249cf6e2c10def465c0b5e4fa077d559f4c0508dfa4a15e812db0fc93ac.jpg)

![](images/ed1b9f84a43be79be5e4873ce99210397cbb3ef6bec25acc1b03b575df47a595.jpg)

![](images/44609c5174f84646d246c8934bf53479484802f1146e36e8745b954ada4c2de6.jpg)

![](images/9c76021fae3ce07b005b034e90c7dab275be08fc27d6d46936673ad72b7a7239.jpg)  
Figure 1: Quality, efficiency, and representation trade-offs of block diffusion. Top-left: BD–AR AUROC difference under rollout and linear probing. Top-right: AUROC across denoising budgets. Bottom: ICU mortality AUROC and evaluation time across block sizes at $S / B = 0 . 5$ . Dashed lines denote AR.

Increasing the denoising budget improves performance initially, but the gains plateau, suggesting that insufficient iterative refinement alone does not explain the remaining gap to AR. More strikingly, this discrepancy is largely absent under frozen linear probing, where AR and BD representations achieve near-identical predictive performance across downstream tasks. This suggests that BD learns similarly predictive clinical representations, and that much of the remaining difference arises during iterative generation rather than representation learning.

Our decoding ablations further support this interpretation. Confidence-based unmasking [14, 15] degrades performance, consistent with evidence that it can prematurely commit locally easy predictions before resolving dependencies important to the global trajectory [17]. ReMDM-style revision [16] enables iterative refinement but provides no meaningful gain. These results suggest that the performance gap is unlikely to be resolved through revision of token commitment alone.

One possible source of this mismatch is the use of a fixed block size. Each block must resolve a predetermined number of positions even though clinically meaningful event groups need not align with these boundaries. This may be particularly relevant for EHR trajectories, where the number of events associated with a clinical episode varies substantially. Variable-length diffusion generation therefore represents a natural direction for future work [18, 19, 20].

More generally, EHR structure may motivate diffusion formulations that differ from those developed for natural language. One possibility is to generate events between successive time markers jointly, treating temporally co-occurring events as unordered or partially ordered sets. Alternative corruption processes, including uniform discrete diffusion [21, 22], may also be worth exploring because they require the model to distinguish corrupted clinical events from plausible clean ones rather than relying exclusively on a dedicated mask state.

Finally, generative endpoint prediction captures only one use of diffusion models for EHRs. Their ability to condition on partially observed sequences could support infilling missing events, reconstructing incomplete records, or exploring counterfactual trajectories. Overall, our results suggest that block diffusion is a promising and computationally attractive framework for longitudinal EHR modelling, while also showing that improved decoding remains important for closing the residual gap to AR zero-shot prediction.

## References

[1] Haresh Rengaraj Rajamohan, Xiang Gao, Weicheng Zhu, Shih-Lun Huang, Long Chen, Kyunghyun Cho, Cem M. Deniz, and Narges Razavian. Foundation Models for Clinical Records at Health System Scale. 7 2025.

[2] Pawel Renc, Yugang Jia, Anthony E. Samir, Jaroslaw Was, Quanzheng Li, David W. Bates, and Arkadiusz Sitek. Zero shot health trajectory prediction using transformer. npj Digital Medicine, 7(1):256, 9 2024. ISSN 2398-6352. doi: 10.1038/s41746-024-01235-0.

[3] Shane Waxler, Paul Blazek, Davis White, Daniel Sneider, Kevin Chung, Mani Nagarathnam, Patrick Williams, Hank Voeller, Karen Wong, Matthew Swanhorst, Sheng Zhang, Naoto Usuyama, Cliff Wong, Tristan Naumann, Hoifung Poon, Andrew Loza, Daniella Meeker, Seth Hain, and Rahul Shah. Generative Medical Event Models Improve with Scale. 11 2025.

[4] Artem Shmatko, Alexander Wolfgang Jung, Kumar Gaurav, Søren Brunak, Laust Hvas Mortensen, Ewan Birney, Tom Fitzgerald, and Moritz Gerstung. Learning the natural history of human disease with generative transformers. Nature, 9 2025. ISSN 0028-0836. doi: 10.1038/s41586-025-09529-3.

[5] Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and Effective Masked Diffusion Language Models. 11 2024.

[6] Xinyou Wang, Zaixiang Zheng, Fei Ye, Dongyu Xue, Shujian Huang, and Quanquan Gu. Diffusion Language Models Are Versatile Protein Learners. 10 2024.

[7] Clement Vignac, Igor Krawczuk, Antoine Siraudin, Bohan Wang, Volkan Cevher, and Pascal Frossard. DiGress: Discrete Denoising diffusion for graph generation. 5 2023.

[8] Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models. 5 2025.

[9] Muhang Tian, Bernie Chen, Allan Guo, Shiyi Jiang, and Anru R. Zhang. Reliable Generation of Privacy-preserving Synthetic Electronic Health Record Time Series via Diffusion Models. 12 2024.

[10] Jun Han, Zixiang Chen, Yongqian Li, Yiwen Kou, Eran Halperin, Robert E. Tillman, and Quanquan Gu. Guided Discrete Diffusion for Electronic Health Record Generation. 6 2024.

[11] Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language Models are Unsupervised Multitask Learners. 2019.

[12] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Anselm Levskaya, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently Scaling Transformer Inference. 11 2022.

[13] Alistair Johnson, Lucas Bulgarelli, Tom Pollard, Brian Gow, Benjamin Moody, Steven Horng, Leo Anthony Celi, and Roger Mark. MIMIC-IV. PhysioNet, 10 2024. doi: 10.13026/kpb9-mt58. URL https://doi.org/10.13026/kpb9-mt58.

[14] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion Large Language Models. 8 2025.

[15] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T. Freeman. MaskGIT: Masked Generative Image Transformer. 2 2022.

[16] Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Remasking Discrete Diffusion Models with Inference-Time Scaling. 2 2026.

[17] Dueun Kim and Albert No. The Confidence Shortcut: A Reasoning Failure Mode of Masked Diffusion Models. 5 2026.

[18] Marton Havasi, Brian Karrer, Itai Gat, and Ricky T. Q. Chen. Edit Flows: Flow Matching with Edit Operations. 11 2025.

[19] Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Shansan Gong, Yansong Feng, Zhenguo Li, Wei Bi, Guorui Zhou, and Lingpeng Kong. DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas. 2 2026.

[20] Jaeyeon Kim, Lee Cheuk-Kit, Carles Domingo-Enrich, Yilun Du, Sham Kakade, Timothy Ngotiaoco, Sitan Chen, and Michael Albergo. Any-Order Flexible Length Masked Diffusion. 9 2025.

[21] Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dallatorre, Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot, and Volodymyr Kuleshov. Simple Guidance Mechanisms for Discrete Diffusion Models. 5 2025.

[22] Subham Sekhar Sahoo, Justin Deschenaux, Aaron Gokaslan, Guanghan Wang, Justin Chiu, and Volodymyr Kuleshov. The Diffusion Duality. 12 2025.

## A Appendix

## A.1 Responsible Use

This study evaluates block diffusion for retrospective prediction from longitudinal electronic health records and does not establish suitability for clinical deployment. Experiments are limited to MIMIC-IV, and performance may not generalise across institutions, patient populations, or care settings. Generative predictions may exhibit uncertainty, miscalibration, and clinically consequential errors. Before any clinical application, models would require external validation, subgroup fairness assessment, calibration analysis, and prospective evaluation under appropriate clinical oversight. Outputs should not be used to guide patient care without such validation and human supervision and as such, this work should be used for research purposes only.

## A.2 Acknowledgements

This work is supported by UK Research and Innovation (UKRI AI Centre for Doctoral Training in Digital Healthcare grant number EP/Y030974/1), NIHR Imperial Biomedical Research Centre and the NIHR Invention for Innovation grant “Artificial Intelligence to support cancer early diagnosis in general practice" (NIHR207533).

## A.3 Decoding with AR vs BD

![](images/ad94166a74ea961cf25027f1dbf10bceb25788a3aaa90b226a3d7563a209b512.jpg)  
Figure 2: Illustration of AR and block-diffusion decoding for longitudinal EHR trajectories. Clinical events frequently occur in temporally clustered groups. AR generates each event sequentially, whereas BD preserves causal generation across blocks while jointly denoising multiple positions within the active block.

## A.4 MIMIC-IV tokens

![](images/1187a62acddc0985b7c02ad0bdd560f5ab29a0cbf19ec14e7c644dd9638f0ca0.jpg)  
Figure 3: Clinical event multiplicity in MIMIC-IV. Survival distribution of the number of nontime-gap event tokens sharing a recorded timestamp in the ETHOS representation.

## A.5 Block diffusion objective and sampling

Let a tokenised EHR trajectory be $\boldsymbol { x } ~ = ~ \left( x _ { 1 } , \ldots , x _ { L } \right)$ , partitioned into K contiguous blocks $x = ( x ^ { ( 1 ) } , \dots , x ^ { ( K ) } )$ of size B. Following BD3-LM [8], the sequence likelihood factorises au-

Table 1: Multiplicity of clinical event tokens sharing a timestamp in the MIMIC-IV / ETHOS training set. N is the number of non-time-gap event tokens at a timestamp.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of timestamps</td><td>23 274 367</td></tr><tr><td>Mean events per timestamp Median</td><td>16.16 6</td></tr><tr><td>25th / 75th percentile</td><td>3 /20</td></tr><tr><td> $P ( N > 1 )$ </td><td>96.8%</td></tr><tr><td> $P ( N \ge 4 )$ </td><td>67.6%</td></tr><tr><td> $P ( N \geq 8 )$ </td><td>47.4%</td></tr><tr><td> $P ( N \ge 1 \dot { 6 } )$ </td><td>29.8%</td></tr></table>

toregressively across blocks,

$$
p _ { \theta } ( x ) \prod _ { k = 1 } ^ { K } p _ { \theta } ! \left( x ^ { ( k ) } \mid x ^ { ( < k ) } \right) ,\tag{3}
$$

where $x ^ { ( < k ) }$ denotes the completed preceding blocks. Within each active block, tokens are modelled jointly using masked discrete diffusion.

For a clean block $x _ { 0 } ^ { ( k ) }$ , we sample a continuous diffusion time $t \sim \mathcal { U } ( \epsilon , 1 - \epsilon )$ with $\epsilon = 0 . 0 0 1$ , using the cosine retention schedule

$$
\alpha _ { t } = \cos \left( \frac { \pi t } { 2 } \right) .\tag{4}
$$

Each token is independently retained with probability $\alpha _ { t }$ and otherwise replaced by the absorbing mask state $m = [ \mathrm { M A S K } ]$

$$
q ( x _ { t } ^ { i } \mid x _ { 0 } ^ { i } ) \alpha _ { t } , \delta _ { x _ { 0 } ^ { i } } + ( 1 - \alpha _ { t } ) , \delta _ { m } .\tag{5}
$$

Noise levels are sampled independently across blocks. The denoiser predicts the original tokens in the corrupted active block while conditioning on the clean preceding blocks. For absorbing-state masked diffusion, the continuous-time objective reduces to the weighted masked-language-modelling loss [5],

$$
\mathcal { L } _ { \mathrm { M D L M } } ^ { ( k ) } - \mathbb { E } * t , x _ { t } ^ { ( k ) } \left[ \frac { - \alpha _ { t } ^ { \prime } } { 1 - \alpha _ { t } } \sum * i : x _ { t } ^ { ( k ) , i } = m \log p _ { \theta } \left( x _ { 0 } ^ { ( k ) , i } \mid x _ { t } ^ { ( k ) } , x ^ { ( < k ) } \right) \right] .\tag{6}
$$

The total objective averages this loss across blocks. During training, the active block is observed only in corrupted form, while completed blocks provide clean causal context, matching the conditioning structure available during KV-cached blockwise generation.

At inference time, the next block is initialised entirely as [MASK] and denoised over reverse times $1 = t _ { S } > \cdot \cdot \cdot > t _ { 0 } = 0$ . For a reverse transition from t to $s < t ,$ , a currently masked position is revealed with probability

$$
p _ { \mathrm { u n m a s k } } ( s , t ) \frac { \alpha _ { s } - \alpha _ { t } } { 1 - \alpha _ { t } } .\tag{7}
$$

Selected positions sample their token identity from $p _ { \theta } ( x _ { 0 } ^ { i } \mid x _ { t } ^ { ( k ) } , x ^ { ( < k ) } )$ , while unselected positions remain masked. On the final step all remaining masks are resolved, after which the completed block is committed to the KV cache and generation proceeds to the next block.

For confidence-based decoding [15, 14], we preserve approximately the same reveal budget,

$$
k _ { \mathrm { r e v e a l } } \approx n _ { \mathrm { m a s k } } , p _ { \mathrm { u n m a s k } } ( s , t ) ,\tag{8}
$$

but commit the $k _ { \mathrm { r e v e a l } }$ positions whose sampled token receives the highest model probability. ReMDM-style revision [16] instead allows already revealed positions to be reconsidered. On non-final reverse steps, revealed positions are remasked independently with probability

$$
\sigma ( s , t ) \operatorname* { m i n } \left( \eta , \frac { 1 - \alpha _ { s } } { \alpha _ { t } } \right) ,\tag{9}
$$

where $\eta \in { 0 . 0 5 , 0 . 1 , 0 . 2 }$ controls the maximum remasking rate.

For a block size B and S denoising steps, we use $S / B$ as the nominal number of sequential denoiser evaluations per generated token. Our primary block-size comparison sets $S = B / 2 ,$ , giving $S / B = 0 . 5$ . This is a normalisation of sequential decoding depth rather than exact compute: the wall-clock cost of each forward pass still depends on the active block size.

## A.6 Hospital-mortality AUROC versus AUPRC

Hospital mortality exhibits an unusual divergence between discrimination metrics. BD $( B { = } 8 , S { = } 4 )$ outperforms AR in rollout AUROC $( \Delta = \stackrel { \smile } { + } 0 . 0 3 9 , 9 5 \% { \bf C I } \left[ + 0 . 0 2 7 , + 0 . 0 5 0 \right] )$ , but substantially underperforms in AUPRC (∆=−0.102, 95% $\mathrm { C I } \left[ - 0 . 1 2 4 , - \dot { 0 . } 0 8 0 \right] )$ . In contrast, last-token linear probes on the same checkpoints are nearly identical (AR 0.936, BD 0.933), suggesting that this discrepancy arises during generative scoring rather than from a difference in representation quality.

For each patient, the rollout score $\hat { p } _ { \mathrm { d e a t h } }$ is the fraction of 20 Monte Carlo trajectories in which MEDS\_DEATH is generated before the competing endpoint. The two models produce markedly different score distributions. AR is conservative, assigning a score of zero to $2 3 . 7 \%$ of true deaths and 86.7% of survivors. These missed deaths are therefore tied with a large mass of score-zero survivors, limiting pairwise ranking performance. However, the relatively few patients receiving high AR scores are precise, with precision@100 of 0.74.

BD produces a less sparse score distribution. Only 2.5% of true deaths receive a score of zero, and the mean death score increases from 0.242 under AR to 0.353. This improves separation between deaths and survivors and consequently increases AUROC. However, BD also assigns nonzero death probability to 60.9% of survivors, compared with only 13.3% for AR. This increase in false-positive mass reduces precision, particularly among the highest-ranked predictions (precision@100 0.47), explaining why higher AUROC is accompanied by substantially lower AUPRC.

Table 2 summarises the resulting instance-level score distributions on the shared test split (n=55,353, 1,135 deaths).

Table 2: Hospital-mortality rollout scores on the test set. AUROC/AUPRC brackets are 95% instancebootstrap CIs. Score statistics are Monte Carlo P(MEDS\_DEATH) over 20 samples.
<table><tr><td></td><td>AR</td><td>BD  $( B { = } 8 , S { = } 4 )$ </td></tr><tr><td>AUROC</td><td>0.844 [0.831, 0.858]</td><td>0.883 [0.873, 0.892]</td></tr><tr><td>AUPRC</td><td>0.306 [0.276, 0.335]</td><td>0.204 [0.183, 0.227]</td></tr><tr><td>Precision@100</td><td>0.74</td><td>0.47</td></tr><tr><td>Mean score (deaths / survivors)</td><td>0.242 / 0.015</td><td>0.353 / 0.090</td></tr><tr><td>Score = 0 (deaths / survivors)</td><td>23.7% / 86.7%</td><td>2.5% / 39.1%</td></tr></table>

## A.7 Quality–efficiency operating points

![](images/a50b6497560a899f1e8b9ecb589a13ea2104d38407ce1e7dd640508654462542.jpg)  
Figure 4: Predictive performance versus evaluation speed on ICU mortality. Each point denotes a block-diffusion configuration with a particular block size and denoising budget. The x-axis shows evaluation speedup relative to the autoregressive (AR) baseline, while the y-axis shows AUROC. The highlighted $\bar { B ^ { } } = 8 , S = 4$ configuration provides a representative operating point, achieving approximately 1.43× faster evaluation while retaining much of the predictive performance of AR.

## A.8 AUPRC results

![](images/693e7220eb6ca4d7cb04aeb4265a34d2410a1f0b3a43cbde383dd91df083f6c6.jpg)

![](images/d44e87e6834ad2a86014c6b1785cb256144397890c75a4daae77ee2e47735dbb.jpg)

![](images/c98f06ca0264822a47f5d23a95dbe68d3087f83d2946a7c3020a1a8340f767f5.jpg)  
Figure 5: Left: Difference in AUPRC between block diffusion and AR under generative rollout and frozen linear probing across four clinical tasks. Middle: ICU mortality AUPRC as block size varies under the fixed normalised denoising budget $S / B = 0 . 5$ . Dashed lines denote the AR baseline. Shaded regions and error bars indicate 95% bootstrap confidence intervals. Right: ICU mortality AUPRC as the normalized denoising budget $S / B$ increases for fixed block sizes.

## A.9 Decoding Interventions

Table 3: ICU mortality decoding strategies for block size B=8 and S=4 diffusion steps. Metrics are medians over completed full-validation runs where available; Confidence uses the available full split run. Brackets are 95% CIs over the three random seeds.
<table><tr><td>Strategy</td><td>AUROC</td><td>AUPRC</td></tr><tr><td>Random</td><td>0.8476 [0.8331, 0.8614]</td><td>0.3440 [0.3123, 0.3792]</td></tr><tr><td>ReMDM  $\eta = 0 . 0 5$ </td><td>0.8421 [0.8268, 0.8562]</td><td>0.3338 [0.3028, 0.3679]</td></tr><tr><td>ReMDM  $\eta = 0 . 1 0$ </td><td>0.8446 [0.8307, 0.8590]</td><td>0.3515 [0.3176, 0.3871]</td></tr><tr><td> $\mathrm { R e M D M } \eta = 0 . 2 0$ </td><td>0.8433 [0.8287, 0.8580]</td><td>0.3474 [0.3158, 0.3800]</td></tr><tr><td>Confidence</td><td>0.8089 [0.7921, 0.8257]</td><td>0.3487 [0.3154, 0.3844]</td></tr></table>