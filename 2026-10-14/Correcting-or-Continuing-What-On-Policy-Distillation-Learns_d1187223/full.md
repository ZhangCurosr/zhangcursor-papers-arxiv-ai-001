# Correcting or Continuing? What On-Policy Distillation Learns from Student Errors

Shuo Yang<sup>1,2,3</sup> Zheyu Zhang<sup>1,3</sup> Mahito Sugiyama<sup>2</sup> Gjergji Kasneci<sup>1,3</sup>

<sup>1</sup>Technical University of Munich <sup>2</sup>National Institute of Informatics <sup>3</sup>Munich Center for Machine Learning

shuo.yang@tum.de, zheyu.zhang@tum.de, mahito@nii.ac.jp, gjergji.kasneci@tum.de

## Abstract

On-policy distillation (OPD) trains a student on its own rollouts under a frozen teacher, mitigating the distribution mismatch of off-policy supervision. While prior work has primarily trained on problems a student can largely solve, we ask whether OPD can improve a student when training is restricted to problems it currently fails. We evaluate five representative OPD mechanisms with Qwen-3 and Phi-4 models under a unified competition-mathematics protocol. We find that OPD does not improve accuracy on this hard subset, with most mechanisms instead degrading performance despite continued decreases in training KL divergence. All eight configurations show a decline, and the two-sided sign test gives p = 0.0078. In contrast, the same pipeline improves accuracy by 4.6 percentage points when training is not concentrated on failures, showing that the failure is conditional on the training regime rather than systematic. These findings suggest that OPD is effective when supervision is dominated by correct student prefixes, but can reinforce existing errors when training is concentrated on problems the student currently fails. <sup>1</sup>

## 1 Introduction

Knowledge distillation aims to train a student model to imitate a stronger teacher model (Hinton et al., 2015). In on-policy distillation (OPD), the student generates its own outputs for a given task, and token-level supervision is computed from the Kullback–Leibler (KL) divergence between the teacher and student logits (Agarwal et al., 2024; Gu et al., 2024). This differs from earlier approaches that distill from a fixed corpus of teacher-generated samples rather than from the student’s own rollouts (Kim and Rush, 2016). On-policy distillation has recently been proposed as an alternative to reinforcement learning with verifiable rewards for training small reasoning models (Lu and Thinking Machines Lab, 2025; Ko et al., 2025; Shao et al., 2024; DeepSeek-AI, 2025).

![](images/7d7514be9f04bba9205ed9af23ccfd4aeb8be6098aa9b0bbcd0acac58120ddf0.jpg)  
Figure 1: Closer to the teacher does not imply more correct. When the student’s prefix $y _ { < t }$ already contains an error, on-policy distillation drives $p _ { S }$ toward p<sub>T</sub>(· | x, y<sub><t</sub>), reducing KL divergence yet failing to improve accuracy relative to the frozen student.

Reported successes share a common regime in which supervision is dominated by correct student prefixes, either because the student is a tuned model or because much of the training distribution lies within its existing capabilities. We instead ask whether OPD can improve a student when training is restricted to problems it currently fails to solve, so that supervision is dominated by incorrect prefixes, as shown in Figure 1. Therefore, we evaluate five mechanisms under a unified experimental protocol: a trained prompt rewriter, a soft-prefix encoder, a residual-stream offset, a per-token crossattention steering vector, and full fine-tuning. Furthermore, we consider both forward and reverse KL objectives, as well as direct imitation. Our contributions are as follows:

• Characterizing when improvement remains possible. We show that further gains are concentrated on problems the student currently fails to solve, suggesting that the remaining gap is primarily informational rather than a consequence of limited model capacity.

• Establishing a robust negative result. Across five mechanisms, OPD fails to improve and often degrades accuracy, consistently across implementations with a sign test reaching $p = 0 . 0 0 7 8$

• Identifying a conditional failure regime. The same pipeline significantly improves a weaker student on an easier benchmark $( p = 9 . 5 \times 1 0 ^ { - 4 } )$ demonstrating that the negative result depends on the student and task regime rather than the training pipeline itself.

• Tracing the failure. Our analyses provide evidence that supervision from incorrect prefixes can reinforce existing errors, while direct imitation of reference solutions does not recover the lost performance. We further identify three evaluation artifacts that can produce spurious positive results in this setting.

Our findings suggest that OPD is effective when applied to a student whose rollouts are broadly correct, but can fail when supervision is concentrated on the narrow set of problems where the student already makes mistakes. This distinction becomes critical when applying OPD to already capable models.

## 2 Problem Setup

We formulate reverse-KL on-policy distillation as minimizing the KL divergence from the student distribution to the teacher distribution along trajectories sampled from the student:

$$
\mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } \sim p _ { S } ( \cdot | \boldsymbol { x } ) } \sum _ { t } \mathrm { K L } ( p _ { S } ( \cdot | \boldsymbol { x } , \boldsymbol { y } _ { < t } ) | | p _ { T } ( \cdot | \boldsymbol { x } , \boldsymbol { y } _ { < t } ) ) .\tag{1}
$$

where $p _ { S }$ denotes the current student policy and $p _ { T }$ denotes a frozen teacher conditioned on the same prompt x and student-generated prefix $y _ { < t }$ For steering-based methods, the student backbone remains frozen, and $p _ { S }$ is induced by composing the backbone with a trainable perturbation module, such as a soft prefix, a residual-stream offset, or a cross-attention steering vector.

The forward-KL variant uses the same on-policy sampling distribution, $y \sim p _ { S } ( \cdot \mid x )$ , but reverses the arguments of the divergence:

$$
\mathbb { E } _ { \boldsymbol { x } , \boldsymbol { y } \sim p _ { T } ( \cdot | \boldsymbol { x } ) } \sum _ { t } \mathrm { K L } ( p _ { T } ( \cdot | \boldsymbol { x } , \boldsymbol { y } _ { < t } ) \| p _ { S } ( \cdot | \boldsymbol { x } , \boldsymbol { y } _ { < t } ) ) .\tag{2}
$$

Privileged self-distillation instead conditions the teacher on both the input x and the reference solution $y ^ { \star }$ , providing supervision from the correct solution path. Table 3 in Appendix B summarizes the distinction. Our main experiments use the unprivileged formulation, in which the teacher conditions only on the original input and the student’s sampled prefix. We introduce reference-conditioned supervision only as a stress test in Section 3.3.

## 3 Experiment and Results

## 3.1 The Elicitation Ceiling

We first quantify how much external supervision could improve the student in principle. We define the elicitation ratio as

$$
\eta _ { \mathrm { a c c } } = \frac { \mathrm { A c c } _ { \mathrm { m e t h o d } } - \mathrm { A c c } _ { \mathrm { b a s e } } } { \mathrm { A c c } _ { \mathrm { p r i v } } - \mathrm { A c c } _ { \mathrm { b a s e } } } ,\tag{3}
$$

where $\operatorname { A c c } _ { \operatorname { p r i v } }$ is the accuracy obtained when the frozen student is provided with the full worked reference solution in context. We use the worked solution rather than an answer-only string because the former provides the information required to solve the problem, whereas the latter primarily measures answer copying. The denominator therefore represents the improvement available through direct access to the reference solution.

Because every mechanism in Section 3.2 decreases rather than improves accuracy, $\eta _ { \mathrm { a c c } }$ is negative for all methods. More importantly, none of the methods recovers a positive fraction of the available elicitation ceiling. On 800 problems sampled from the training distribution, evaluated with thinking enabled and a 16k-token budget, the student solves 89.6% without external supervision. As shown in Table 1, 684 problems are therefore already solved by the student and provide no accuracy headroom. The remaining 116 problems contain nearly all of the available headroom: providing the reference solution increases accuracy by 26.9 percentage points, with a 95% confidence interval of 21.3–33.0 percentage points. On the hardest bucket, the corresponding gain reaches 50 percentage points, compared with only 3.9 percentage points overall.

The substantial improvement from referenceconditioned supervision on the hardest problems indicates that the student retains sufficient capacity to solve these problems when provided with the relevant information. The remaining failure of OPD therefore points to a gap in eliciting existing capability rather than a lack of model capacity.

A second, independent measurement provides consistent evidence. Under a leakage-free constraint, clean prompt rewriting attains an accuracy ceiling of only about two percentage points. Furthermore, the teacher-likelihood reward used to select rewrites is negatively correlated with answer correctness, yielding a Spearman correlation of $\rho = - 0 . 4 1$ . This suggests that higher teacher likelihood does not reliably identify correct rewrites. Full details are provided in Appendix B.

<table><tr><td>Bucket (base acc.)</td><td>n</td><td>Base</td><td>Priv.</td><td>∆(pp)</td></tr><tr><td>= 0</td><td>57</td><td>0.0</td><td>17.5</td><td>+17.5</td></tr><tr><td>(0,0.5)</td><td>13</td><td>25.0</td><td>75.0</td><td>+50.0</td></tr><tr><td>[0.5, 1)</td><td>46</td><td>64.7</td><td>96.7</td><td>+32.0</td></tr><tr><td>= 1 (saturated)</td><td>684</td><td>100.0</td><td>100.0</td><td>0.0</td></tr></table>

Table 1: Headroom from the reference solution sits on problems the student fails.

We therefore focus our OPD experiments on a hard subset of 1,034 problems selected from 8,000 candidates. Each selected problem has a base accuracy of at most 0.75. We split these problems into 904 training examples and 130 held-out examples.

## 3.2 subIsolation Study Across Mechanisms

Setup. We distill Qwen3-1.7B-Instruct using a frozen Qwen3-4B-Instruct teacher (Yang et al., 2025) on the 904-problem hard subset. We vary the distillation mechanism, KL direction, and training implementation, comparing our in-house implementation with the standard TRL trainer. Figure 2 reports the change in AIME24 (Zhang and Math-AI, 2024) pass@1 accuracy relative to the untouched student. We use 30 problems, grade boxed answers with a math verifier, and allow up to 16k generated tokens.

A second, independently implemented codebase evaluates activation-level steering with the frozen student. It uses three mechanisms: a soft prefix, a residual-stream offset, and a per-token steering vector derived from cross-attention over the problem’s hidden states. The corresponding results are reported in Table 2. Appendix C provides implementation details for all five mechanisms, Appendix B reports the trained prompt-rewriter results, and Appendix F provides the complete hyperparameter and hardware configuration.

Results. All evaluated configurations yield nonpositive accuracy changes. Varying model capacity, KL direction, gating and warm-up, teacher scale, or training framework does not reverse this direction. Meanwhile, token-level KL decreases throughout training even when held-out accuracy declines.

We assess the consistency of this effect using a two-sided sign test (Dixon and Mood, 1946) under a symmetric null of zero effect. With five matched configurations, all five accuracy changes are negative, yielding a two-sided $p = 0 . 0 6 3$ and a onesided $p = 0 . 0 3 1$ . Including the three additional negative steering configurations in Table 2 gives eight negative signs out of eight and $p = 0 . 0 0 7 8$ (two-sided). We exclude the fourth steering configuration because its effect is consistent with recovery from an induced baseline deficit rather than a genuine capability change.

![](images/29931a74d2d2b4557c6d51c8d332f0b3e81c385a2fcf86c7eba59c612da2a328.jpg)  
Figure 2: Accuracy change relative to the frozen student on the competent-student hard subset. All evaluated mechanisms yield non-positive changes. Single-run results are shown except for the TRL configuration, for which three training seeds and their mean are reported. The same pipeline improves weaker students, as discussed in Section 3.2.

Individual configurations show substantially greater variability in magnitude. Across three TRL training seeds, the accuracy changes are −6.7, −20.0, and 0.0 percentage points, with a mean of −8.9 and a standard deviation of 10.2 points. None of the seeds yields a positive change. Re-evaluating the frozen student alone gives accuracies of 46.7%, 40.0%, and 40.0%, with the sampling temperature of 0.6 changing the outcome on 10 of 30 AIME24 problems relative to greedy evaluation. These results motivate reporting the consistency of the direction rather than relying on the magnitude from any single run. Training seeds are treated as repeated measurements of the same configuration and are therefore not counted as independent observations in the sign test. Appendix C provides further details.

<table><tr><td>Configuration</td><td>Metric</td><td>∆</td></tr><tr><td>Residual offset / forward KL (8B)</td><td>Macro</td><td>-2.2</td></tr><tr><td>Token attention</td><td></td><td></td></tr><tr><td>Sparse-accuracy RL (→) Privileged imitation (4B)</td><td>Macro</td><td>-11.1</td></tr><tr><td>OPD after imitation (4B)</td><td>AIME24 -11.7 AIME24 +1.7</td><td></td></tr><tr><td></td><td></td><td></td></tr></table>

Table 2: Independent steering results (avg@2). Teacher sizes are in parentheses and “–” denotes no teacher.

Positive controls. To test whether the observed failure reflects a systematic issue in our implementation, we apply the same TRL reverse-KL pipeline to settings in which OPD has previously been shown to improve performance. Qwen2.5-0.5B-Instruct, distilled from Qwen2.5-1.5B-Instruct (Qwen et al., 2025) on GSM8K prompts (Cobbe et al., 2021), improves from 43.6% to 48.2% accuracy over 1,319 problems, corresponding to a statistically significant gain at $p = 9 . 5 \times 1 0 ^ { - 4 }$ . Phi-4-mini-instruct (Abouelenin et al., 2025), distilled from a same-size Phi-4-mini-reasoning teacher (Xu et al., 2025), improves from 67% to 74% on MATH-500 (Lightman et al., 2023). This second control is not competence-matched and therefore serves only as an additional positive control. Full details are provided in Appendix D.

These positive controls provide evidence against a general defect in the training pipeline. However, because they vary student capability, benchmark difficulty, and model family simultaneously, they do not isolate the factor responsible for the observed difference.

## 3.3 Failure Mode: Supervision on an Incorrect Prefix

Our setting differs from those in which OPD has previously succeeded: the student is already capable, receives no supervised warm-up, and is trained only on problems it currently fails. This creates a failure mode inherent to the on-policy objective in Equation 1. The KL supervision is computed on the student’s sampled prefix $y _ { < t }$ . When this prefix contains an incorrect step, the teacher is conditioned on an already erroneous state rather than on the correct solution path. The resulting supervision therefore cannot directly correct the error that generated the prefix and may instead reinforce the student’s existing trajectory. Figure 2 is consistent with this mechanism: training KL decreases while held-out accuracy also declines.

Stress test. We ask whether the failure is simply caused by the absence of a gold solution as an explicit anchor. Providing supervision directly from the reference solution does not resolve the problem. Row three of Table 2 decreases accuracy by 11.7% in a single run, which we interpret as evidence of direction rather than effect size given the seed variability above. A subsequent on-policy KL step recovers accuracy only to the untouched baseline.

A related soft-prefix analysis offers further evidence. Across 94 problems, the learned prefixes are nearly collinear, with pairwise cosine similarities of 0.998–0.999, whereas the underlying problem representations yield only 0.78 and a single direction accounts for 70–73% of their variance. As the checkpoint corresponding to row three was not retained, these results constitute consistent rather than direct evidence. We therefore interpret the observed recovery as the removal of a self-induced deficit rather than the elicitation of new capability.

Evaluation artifacts. Three artifacts can produce spurious positive results in this setting. Permissive regex matching can overstate gains: an apparent 11.7% improvement fell to 0.8% after requiring a verified boxed answer. An insufficient generation budget can truncate reasoning and drive both baseline and treatment accuracy to 0, producing a misleading null result. Finally, answer leakage through unaudited rewrites can inflate apparent gains by 10–20 %. Appendix E provides details.

## 4 Conclusion

Across five mechanisms and three training objectives, OPD does not reliably improve a student when training is restricted to problems it currently fails. The negative direction is consistent across eight configurations, while a positive control improves accuracy by 4.6 percentage points under the same pipeline, indicating that the failure depends on the training regime rather than on an implementation defect. These results delineate the boundary of standard OPD: it is effective when supervision is dominated by correct student prefixes, but can reinforce existing errors when supervision is concentrated on incorrect prefixes.

Future work should isolate student capability from task difficulty under matched conditions, test whether outcome-based reinforcement learning exhibits a similar failure mode, and develop objectives that correct erroneous prefixes rather than imitate the teacher on them.

## Limitations

Our findings highlight the potential risks of applying on-policy distillation without carefully assessing its effects on model capabilities. In particular, training primarily on problems that the student cannot solve may reinforce erroneous reasoning rather than improve accuracy, potentially propagating errors to downstream applications. However, our negative evidence is drawn from Qwen3, and is evaluated primarily on one hard-mathematics subset. Its transferability to other settings therefore remains unknown. AIME24 contains only thirty items, therefore we accordingly claim directional consistency rather than effect magnitude throughout. Furthermore, training and evaluating multiple distillation configurations incurs computational costs and associated environmental impacts, although we do not quantify energy consumption in this work. These considerations underscore the importance of evaluating both task-level performance and error patterns before deploying models trained with on-policy distillation.

## Ethical Considerations

The findings of this work raise broader questions about responsibility in the development and evaluation of AI systems. Distillation can make a model appear to be learning from a more capable teacher, yet improvements in optimization metrics do not necessarily translate into greater practical competence. Treating such metrics as proxies for capability may create misplaced confidence in models whose errors remain unresolved. This is particularly consequential in settings where model outputs inform human judgments or decisions, as errors may be accepted or propagated when users overestimate a system’s reliability. Our work highlights the importance of critically examining what training procedures actually teach models, rather than assuming that successful optimization implies meaningful capability gains. We propose that our results will motivate greater scrutiny of the assumptions underlying model evaluation and deployment.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Hassan Awadalla, Bach Nguyen, Jian-Min Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, Dong Chen, Dong-Dong Chen, Junkun Chen, Weizhu Chen, Yen-Chun Chen, Yi-Ling Chen, Qi Dai, Xi-Yang Dai, Ru-Chao Fan, and

55 others. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. ArXiv, abs/2503.01743.

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Tianqi Chen, Bing Xu, Chiyuan Zhang, and Carlos Guestrin. 2016. Training deep nets with sublinear memory cost. ArXiv, abs/1604.06174.

Karl Cobbe, Vineet Kosaraju, Mo Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John D. Schulman. 2021. Training verifiers to solve math word problems. ArXiv, abs/2110.14168.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. ArXiv, abs/2501.12948.

Wilfrid Joseph Dixon and Alexander McFarlane Mood. 1946. The statistical sign test. Journal ofthe American Statistical Association, 41 236:557–66.

Bradley Efron. 1979. Bootstrap methods: Another look at the jackknife. Annals ofStatistics, 7:1–26.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. 2026. Revisiting on-policy distillation: Empirical failure modes and simple fixes. Preprint, arXiv:2603.25562.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. 2024. MiniLLM: Knowledge distillation of large language models. In The Twelfth International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Xiaodong Song, and Jacob Steinhardt. 2021. Measuring mathematical problem solving with the math dataset. ArXiv, abs/2103.03874.

Geoffrey E. Hinton, Oriol Vinyals, and Jeffrey Dean. 2015. Distilling the knowledge in a neural network. ArXiv, abs/1503.02531.

Yoon Kim and Alexander M. Rush. 2016. Sequencelevel knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1317–1327, Austin, Texas. Association for Computational Linguistics.

Jongwoo Ko, Tianyi Chen, Sungnyun Kim, Tianyu Ding, Luming Liang, Ilya Zharkov, and Se-Young Yun. 2025. DistiLLM-2: A contrastive approach boosts the distillation of LLMs. In Forty-second International Conference on Machine Learning.

Jongwoo Ko, Sungnyun Kim, Tianyi Chen, and Se-Young Yun. 2024. DistiLLM: Towards streamlined distillation for large language models. In Forty-first International Conference on Machine Learning.

Weize Kong, Spurthi Hombaiah, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. 2024. PRewrite: Prompt rewriting with reinforcement learning. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 2: Short Papers), pages 594–601, Bangkok, Thailand. Association for Computational Linguistics.

Woosuk Kwon, Zhuo-Han Li, Si-Yuan Zhuang, Ying Sheng, Lian-Min Zheng, Cody Hao Yu, Joseph E. Gonzalez, Haotong Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Proceedings of the 29th Symposium on Operating Systems Principles.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John D. Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. ArXiv, abs/2305.20050.

Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. In International Confer ence on Learning Representations.

Kevin Lu and Thinking Machines Lab. 2025. On-policy distillation. Thinking Machines Lab: Connectionism blog.

Quinn Mcnemar. 1947. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12:153–157.

Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Frederick Diamos, Erich Elsen, David García, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. 2017. Mixed precision training. ArXiv, abs/1710.03740.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Charles Edward Spearman. 2015. The proof and measurement of association between two things. International journal of epidemiology, 39 5:1137–50.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Can-Wen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2019.

Transformers: State-of-the-art natural language processing. In Conference on Empirical Methods in Natural Language Processing.

Yan Xie, Sijie Zhu, Tiansheng Wen, Bo Chen, and Yifei Wang. 2026. On the position bias of on-policy distillation. Preprint, arXiv:2606.22600.

Haoran Xu, Baolin Peng, Hany Hassan Awadalla, Dongdong Chen, Yen-Chun Chen, Mei Gao, Young Jin Kim, Yun-Sheng Li, Li-Liang Ren, Yelong Shen, Shuohang Wang, Weijian Xu, Jian-Feng Gao, and Weizhu Chen. 2025. Phi-4-mini-reasoning: Exploring the limits of small reasoning language models in math. ArXiv, abs/2504.21233.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. 2024. Large language models as optimizers. In The Twelfth International Conference on Learning Representations.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. 2026. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. Preprint, arXiv:2602.12125.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. 2024. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. 2025. Does reinforcement learning really incentivize reasoning capacity in LLMs beyond the base model? In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yifan Zhang and Team Math-AI. 2024. American invitational mathematics examination (aime) 2024.

Yifan Zhang and Team Math-AI. 2025. American invitational mathematics examination (aime) 2025.

Zheyu Zhang, Shuo Yang, and Gjergji Kasneci. 2026. Consolidating rewarded perturbations for llm posttraining. arXiv preprint arXiv:2605.31494.

Siyan Zhao, Zhihui Xie, Mengchen Liu, Jing Huang, Guan Pang, Feiyu Chen, and Aditya Grover. 2026. Self-distilled reasoner: On-policy self-distillation for large language models. In Forty-third International Conference on Machine Learning.

## A Related Work

On-policy distillation and related generalized distillation objectives train a student on its own rollouts under the supervision of a frozen teacher (Agarwal et al., 2024; Gu et al., 2024; Ko et al., 2024, 2025). Existing studies typically initialize the student from a base or weakly tuned model that has already received supervised warm-up (Lu and Thinking Machines Lab, 2025; Yang et al., 2025; Yu et al., 2024). Recent work has identified several pathologies of on-policy supervision, including degraded supervision as the student’s rollouts diverge from the teacher’s distribution (Xie et al., 2026) and unreliable guidance when a generated prefix leaves the teacher’s support (Fu et al., 2026). These studies primarily diagnose such failures and propose modifications to the objective. In contrast, we investigate under what conditions the standard objective remains effective, evaluating multiple distillation mechanisms and validating our findings with both a positive control and a stress test.

Several recent extensions modify the standard distillation objective or its information available to the teacher. Reward-weighted generalizations can allow a student to surpass its teacher (Yang et al., 2026). Our results concern the standard KL-based and imitation objectives on a single challenging domain and therefore do not directly address such reward-weighted formulations. Privileged selfdistillation instead conditions the teacher on the reference solution (Zhao et al., 2026). We study the standard teacher-student setting without privileged information and consider reference-conditioned supervision only as a stress test.

Our experiments also include prompt rewriting as discrete forms of intervention (Yang et al., 2024; Kong et al., 2024). Our results highlight the need to examine rewriters for answer leakage when evaluating their effectiveness. Finally, our identified failure mode arises from supervision computed over the student’s own sampled tokens, whereas reinforcement learning with verifiable rewards typically provides supervision from the final outcome (Shao et al., 2024; DeepSeek-AI, 2025). Whether outcome-based reinforcement learning exhibits an analogous failure mode remains an open question, as does the broader question of whether such training induces genuinely new reasoning capabilities or primarily reinforces capabilities already present in the model’s distribution (Yue et al., 2025; Zhang et al., 2026).

## B Setup and Diagnostics

<table><tr><td>Aspect</td><td>OPD</td><td>OPSD</td></tr><tr><td>Teacher input</td><td>x only</td><td>x and  $y ^ { \star }$ </td></tr><tr><td>Gold anchor</td><td>None</td><td>Reference path  $y ^ { \star }$ </td></tr><tr><td>Positive gain</td><td>Distillation</td><td>Elicitation + write-in</td></tr></table>

Table 3: OPD lacks OPSD’s privileged answer anchor, making its training problem harder.

![](images/fa4a9d661296f1b657b9c67c2a08760d03ffd2f248776fb8bb7691074c05fe91.jpg)  
Figure 3: Accuracy with and without the reference solution in context, by the student’s own base accuracy, with thinking enabled and a 16k-token budget.

Ceiling by rewriting. Figure 3 plots the bucketed ceiling discussed. On MATH-500 we measure the accuracy a best-of-K oracle rewriter can recover under a boundary that excludes any rewrite containing a solution or a leaked final answer. An initial bound of twenty points traces entirely to unterminated reasoning traces that leaked the answer. Once that boundary is enforced cleanly the bound falls to about two points, and permitting a short strategy hint without a solution raises it again to 8.5 to 10.5 points. No trained rewriter reproduced even this relaxed bound without either failing to update or re-leaking the answer. A KL-based reward used to select among rewrites is negatively correlated with downstream accuracy. We measure this with the Spearman rank coefficient (Spearman, 2015)

$$
\rho = 1 - \frac { 6 \sum _ { i = 1 } ^ { n } d _ { i } ^ { 2 } } { n ( n ^ { 2 } - 1 ) } ,\tag{4}
$$

where $d _ { i }$ is the difference between the reward rank and the accuracy rank of rewrite i among n rewrites, giving $\rho ~ = ~ - 0 . 4 1$ here, therefore, optimizing agreement with the teacher actively selects against correct rewrites rather than for them.

Trained rewriters. Under the clean boundary, a trained rewriter changed accuracy by only about one point, a result indistinguishable from noise, and a later rewriter checkpoint instead lowered AIME24 accuracy from 45.8% to 26.7% under foursample averaging.

Privileged context under a short budget. Under a shorter, non-thinking budget the same reference solution becomes a net negative, since a long solution turns into interference once it can no longer fit the available context.

## C Isolation Study Details

![](images/34f376d683a914ea927862ab78844e79b18d6e4c93b97bf8c21520b62c9ac42f.jpg)  
(a)-(d) leave θ untouched; (e) updates every weight

Figure 4: The five mechanisms, ordered by the extent to which each modifies the frozen model. The first four leave all parameters of θ unchanged, whereas only full fine-tuning updates them.
<table><tr><td>Configuration</td><td>Base</td><td>Meth.</td><td>∆</td><td>W/L/T</td><td>p</td></tr><tr><td>Prefix, OPSD-style KL</td><td>26.7</td><td>13.3</td><td>-13.3</td><td>2/6/22</td><td>0.29</td></tr><tr><td>Prefix, OPD (4B)</td><td>26.7</td><td>20.0</td><td>-6.7</td><td>0/2/28</td><td>0.50</td></tr><tr><td>Full FT, forward KL</td><td>40.0</td><td>30.0</td><td>-10.0</td><td>2/5/23</td><td>0.45</td></tr><tr><td>Full FT, reverse KL</td><td>46.7</td><td>36.7</td><td>-10.0</td><td>2/5/23</td><td>0.45</td></tr><tr><td>TRL reverse KL, seed 42</td><td>46.7</td><td>40.0</td><td>-6.7</td><td>3/5/22</td><td>0.73</td></tr><tr><td>TRL reverse KL, seed 0</td><td>40.0</td><td>20.0</td><td>-20.0</td><td>0/6/24</td><td>0.031</td></tr><tr><td>TRL reverse KL, seed 1</td><td>40.0</td><td>40.0</td><td>0.0</td><td>3/3/24</td><td>1.00</td></tr></table>

Table 4: AIME24 pass@1 accuracy with thirty problems for every configuration: accuracy in percent, delta, items won, lost, and tied by the method, and the exact twosided McNemar p-value. Each configuration re-samples its own frozen baseline.

Sign tests. Table 4 lists every cell in full, and Figure 5 plots the seed spread discussed in Section 3.2. The sign test treats each configuration’s sign as a fair coin under a null of no effect: five negative outcomes out of five gives a one-sided $p = 0 . 0 3 1$ , or 0.063 two-sided. Adding the three negative steering rows gives eight out of eight, at a two-sided $p = 0 . 0 0 7 8$ . We exclude row four of Table 2 throughout, since we read it as recovery from an induced deficit rather than as an independent configuration. All configurations share the same thirty items, thus this test establishes sign consistency across recipes rather than independence across 150 separate problems. Seeds and later reevaluations re-measure configurations that are already counted and are accordingly never added a second time. No single item is degraded by all five configurations at once.

![](images/1d224f6df7d5dbb928950c428499a74b27c7aee83aa3d9c388da736a5984a21b.jpg)  
Figure 5: Three training seeds of the TRL configuration. The frozen student itself varies between 40.0 and 46.7% across evaluation runs, and its per-problem verdict changes on ten of thirty items.

Per-cell significance. For each cell we also report an exact McNemar test (Mcnemar, 1947) on the items where the method and the frozen student disagree. Writing b for the number of items the method alone solves and c for the number the base model alone solves, with $n = b + c , $

$$
p _ { \mathrm { t w o } } = \operatorname* { m i n } \left( 1 , 2 \sum _ { i = 0 } ^ { \operatorname* { m i n } ( b , c ) } { \binom { n } { i } } 2 ^ { - n } \right)\tag{5}
$$

is the exact two-sided p-value under a null that each disagreement is equally likely to favor either side. This is the test behind every p reported in Table 4. Confidence intervals instead use a nonparametric bootstrap (Efron, 1979): resampling the n graded items with replacement B times, computing the mean paired delta $\begin{array} { r } { \Delta _ { b } ^ { * } = \frac { 1 } { n } \sum _ { i } ( y _ { i } ^ { \mathrm { m e t h o d } } - y _ { i } ^ { \mathrm { b a s e } } ) } \end{array}$ on each resample, and reporting the 2.5th and 97.5th percentiles of $\{ \Delta _ { b } ^ { * } \} _ { b = 1 } ^ { B }$ with $B = 1 0 { , } 0 0 0$

AIME25 re-evaluation. Re-evaluating the two full fine-tuning checkpoints on AIME25 (Zhang and Math-AI, 2025) under the identical protocol gives a delta of +3.3 points for both, each a single additional solved item at $p = 1 . 0 $ , the opposite sign of their AIME24 deltas of −10.0 and −6.7 points. Pooling both benchmarks for a sample of sixty returns the direction to negative, at −3.3

and −1.7 points respectively, again without significance. We read this as a concrete illustration of the single-benchmark noise floor rather than as evidence against the overall direction, and we do not add these evaluations to the sign test above.  
![](images/4d1e8e5755ae24301a23758d89ec79db907002e9ed6056ada36cacb15b857550.jpg)  
Figure 6: Trained soft prefixes are nearly collinear across ninety-four problems, drawn from all thirty AIME24 items plus sixty-four further hard-subset items, unlike the pooled problem inputs or an untrained encoder of identical architecture. Static embeddings serve as a proxy input for the checkpoints trained with contextual pooling. A high cosine under this proxy remains informative even so.

Prefix collapse. Across these ninety-four problems the trained prefixes are nearly collinear, as Figure 6 shows, and the update from the imitation phase to the on-policy phase is itself a shared shift rather than a per-problem correction: the pairwise cosine between these shift vectors is 0.999. The learned prefix thus behaves as a global offset rather than as an instance-conditioned intervention.

Steering codebase. Figure 4 shows where each mechanism attaches to the frozen backbone. A small encoder maps each problem to a soft prefix, to a single offset added to every prompt embedding, or to a per-token offset produced by crossattention over hidden states of the problem’s own tokens, while the student itself stays entirely frozen throughout. The privileged-imitation row of Table 2 trains this offset with cross-entropy against the reference solution directly, and the following row applies a further on-policy KL step on top of it.

<table><tr><td>Control</td><td>Base</td><td>Student</td><td>∆</td><td>n</td><td>p</td></tr><tr><td>Qwen2.5-0.5B, GSM8K</td><td>43.6</td><td>48.2</td><td>+4.6</td><td>1319</td><td>9.5×10−4</td></tr><tr><td>Phi-4-mini, MATH-500</td><td>67.0</td><td>74.0</td><td>+7.0</td><td>100</td><td>0.21</td></tr><tr><td>Phi-4-mini, AIME24</td><td>10.0</td><td>13.3</td><td>+3.3</td><td>30</td><td>1.00</td></tr></table>

Table 5: Positive controls under the identical TRL reverse-KL pipeline. For GSM8K the student wins 196 items and loses 135, with a 95% confidence interval of 2.0 to 7.4 points. For Phi-4-mini on MATH-500 the student wins 15 items and loses 8, with a 95% confidence interval of −2.0 to 16.0 points.

## D Positive Controls

GSM8K. Table 5 summarizes both controls in full. We train with TRL reverse KL at β = 1, on promptonly data drawn from the 7,473 GSM8K (Cobbe et al., 2021) training problems, for one epoch, under an 8k generation cap, and we evaluate greedily on the 1,319 test problems.

Phi-4-mini. This tokenizer-family control distills Phi-4-mini-instruct (Abouelenin et al., 2025) from Phi-4-mini-reasoning (Xu et al., 2025), both 3.8B parameters and sharing a vocabulary, on 800 problems from the MATH training set (Hendrycks et al., 2021) for 200 steps. This control does not constitute a competence-matched replication of our main setting. The teacher is of the same size as the student rather than larger, MATH-500 is initially unsaturated for this particular student, and most rollouts were truncated at the 8k cap rather than terminating naturally. The MATH-500 gain does not reach statistical significance, and the AIME24 gain reduces to a single additional item. We therefore interpret this control only as a directional signal that the observed pattern is not specific to Qwen3.

## E Evaluation Artifacts

Under regex extraction of the last number within a short generation budget, an apparent AIME24 gain of 11.7 percentage points, from 14.2% to 25.8%, decreased to merely 0.8 percentage points, from 18.3% to 19.2%, once boxed-answer verification was instead required on the same checkpoint. A 2k-token budget truncates the student’s reasoning before it reaches a final answer. Consequently, under this budget the frozen baseline itself scores zero, which can be readily mistaken for a genuine effect if the baseline is not evaluated under the same budget. Finally, an unaudited rewriter that generated unterminated reasoning traces or stray boxed answers in its rewrites inflated apparent gains by ten percentage points under imitation and by twenty percentage points under oracle selection. Both apparent gains disappeared once the leakage boundary was properly enforced.

<table><tr><td></td><td>Isolation</td><td>Seed repeats</td><td>GSM8K</td><td>Phi-4</td></tr><tr><td>Student</td><td>Qwen3-1.7B</td><td>Qwen3-1.7B</td><td>Qwen2.5-0.5B</td><td>Phi-4-mini</td></tr><tr><td>Teacher</td><td>Qwen3-4B</td><td>Qwen3-4B</td><td>Qwen2.5-1.5B</td><td>Phi-4-reason.</td></tr><tr><td>Train set</td><td>hard-904</td><td>hard-904</td><td>GSM8K-7473</td><td>MATH-800</td></tr><tr><td>Steps run</td><td>200</td><td>200</td><td>934, 1 epoch</td><td>200</td></tr><tr><td>Learning rate</td><td>1.0e-5</td><td>1.0e-5</td><td>2.0e-5</td><td>1.0e-5</td></tr><tr><td>Batch × accum.</td><td>1 × 4</td><td>1 × 4</td><td>2 × 4</td><td>1 × 4</td></tr><tr><td>Warmup steps</td><td>20</td><td>20</td><td>28</td><td>10</td></tr><tr><td>Train sampling</td><td>0.6/0.95/20</td><td>0.6/0.95/20</td><td>0.6/0.95/20</td><td>0.6/0.95/20</td></tr><tr><td>Eval decoding</td><td>sampled</td><td>sampled</td><td>greedy</td><td>greedy</td></tr><tr><td>Gen. cap, train</td><td>16384</td><td>16384</td><td>8192</td><td>8192</td></tr><tr><td>Thinking mode</td><td>on</td><td>on</td><td>off</td><td>off</td></tr><tr><td>Attention impl.</td><td>sdpa</td><td>sdpa</td><td>sdpa</td><td>eager</td></tr><tr><td>Seed</td><td>42</td><td>0,1</td><td>42</td><td>42</td></tr></table>

Table 6: Hyperparameters shared by every TRL DistillationTrainer run in the paper, reverse KL at $\beta = 1$ weight decay 0.01, gradient clip 1.0, the AdamW optimizer (Loshchilov and Hutter, 2017) with $\beta _ { 1 } ~ = ~ 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9 , \epsilon = 1 0 ^ { - 8 }$ , and a linear learning-rate decay. Sampled evaluation uses temperature 0.6 with top-p 0.95 and top-k 20.
<table><tr><td></td><td>Fwd KL</td><td>Rev KL</td><td>Prefix, OPD</td><td>Prefix, OPSD</td></tr><tr><td>Updates</td><td>full params</td><td>full params</td><td>prefix, 64 tok.</td><td>prefix, 96 tok.</td></tr><tr><td>Imitation warm-up</td><td>none</td><td>none</td><td>300 steps</td><td>500 steps</td></tr><tr><td>KL direction</td><td>forward</td><td>reverse</td><td>forward</td><td>OPSD-style</td></tr><tr><td>KL support</td><td>top-64, clip 5</td><td>top-64, clip 5</td><td>top-64, clip 5</td><td>top-64, clip 5</td></tr><tr><td>Train rollout cap</td><td>16384</td><td>16384</td><td>4096</td><td>4096</td></tr><tr><td>Steps run, evaluated</td><td>400</td><td>800, at 500</td><td>400</td><td>400</td></tr><tr><td>Batch × accum.</td><td>1 × 1</td><td>4 × 1</td><td>4 × 1</td><td>1 × 1</td></tr><tr><td>Learning rate</td><td>1.0e-5</td><td>1.0e-5</td><td>2.5e-5</td><td>2.5e-5</td></tr><tr><td>Weight decay, clip</td><td>0.01, 1.0</td><td>0.01, 1.0</td><td>0.01,0.5</td><td>0.01, 0.5</td></tr><tr><td>Gated on errors</td><td>on</td><td>off</td><td>on</td><td>n/a</td></tr></table>

Table 7: Hyperparameters for the in-house engine behind the remaining four rows of Table 4, all with a constant learning rate, no warmup, and the AdamW optimizer (Loshchilov and Hutter, 2017). Evaluation always uses a 16384-token cap regardless of the training rollout cap above. A shorter training cap is a property of the training setup only.

## F Reproducibility

This appendix lists the hardware, software, and training hyperparameters behind every result in the paper.

Hardware and software. Every training and evaluation run uses a single compute node with one NVIDIA H200 NVL GPU, 141 GB of device memory, an AMD EPYC 9654 CPU with 96 cores, and 755 GiB of system memory, running NVIDIA driver 580.178.04 with system CUDA 12.9. The student and teacher always share this one GPU. The software stack is Python 3.11, PyTorch 2.10.0 built against CUDA 12.8, Transformers 4.57.6 (Wolf et al., 2019), TRL 1.13.0, and vLLM 0.18.0 (Kwon et al., 2023) for sampling-heavy evaluation. All training uses bf16 mixed precision (Micikevicius et al., 2017) with gradient checkpointing (Chen et al., 2016). Table 6 lists hyperparameters for every TRL-trained run, and Table 7 for the in-house engine behind the remaining four isolation-study rows.