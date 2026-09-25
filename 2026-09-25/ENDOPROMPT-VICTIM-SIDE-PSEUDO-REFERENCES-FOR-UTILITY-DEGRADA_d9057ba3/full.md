# ENDOPROMPT: VICTIM-SIDE PSEUDO-REFERENCES FOR UTILITY DEGRADATION

Qingyu Wu<sup>1,∗</sup>, Zeyu Feng<sup>1,∗</sup>, Yongda Yu<sup>2</sup>, Yuzhe Luo<sup>1</sup>, Renju Liu<sup>3</sup>, and Hua Cheng<sup>1,†</sup>

<sup>1</sup>Defense Innovation Institute, Academy of Military Science, Beijing, China <sup>2</sup>Nanjing University, Nanjing, China

<sup>3</sup>School of Software Engineering, South China University of Technology, Guangzhou, China <sup>∗</sup>Equal contribution; <sup>†</sup>Corresponding author; Email: chenghua ams@163.com

## ABSTRACT

Prompt injection can degrade benign task performance without eliciting harmful content. Yet many attack objectives depend on task labels or predefined target responses. We present ENDOPROMPT, a white-box method that learns utility-degrading prefixes from unlabeled instructions. Its generator takes the request text as input. Clean victim continuations serve as pseudo-references: local search identifies prefixes that reduce continuation likelihood, and preference fitting on comparisons within the same instruction, followed by reward refinement, distills this signal into a generator. At deployment, the generator produces one prefix per request without further victim-side search. Across four instruction-tuned models and the complete splits of seven benign benchmarks, ENDOPROMPT yields a mean utility change of -26.8 percentage points; 27 of 28 cells are negative. Failure analysis reveals output expansion and prefix reuse; the controls do not establish a degradation advantage from request matching. Victim-derived supervision can reveal utility weaknesses without benchmark feedback or prescribed failure responses. The code will be released upon acceptance.

Index Terms— large language models, prompt injection, utility degradation, robustness

## 1. INTRODUCTION

The security of instruction-tuned language models depends on both preserving legitimate task behavior and preventing harmful outputs. Prompt injection threatens the former by redirecting responses through untrusted contextual text [1]. A response can therefore fail a benign request without containing unsafe content or an explicit refusal. As illustrated in Fig. 1, an injected prefix turns a correct arithmetic solution into a fluent metaphor that leaves the calculation unanswered. Such failures make benign utility an important target of adversarial evaluation.

Attack discovery without downstream supervision remains difficult. Universal adversarial triggers can optimize a task loss or a target output [2], whereas methods such as GCG and AdvPrompter use harmful-response targets [3, 4]. These objectives directly specify the behavior to optimize. Benign utility failures are more heterogeneous: a response may be incorrect, incomplete, or unrelated to the request. How can such failures be discovered without task labels or a predefined target response?

We address this question using the victim’s clean continuation on an unlabeled instruction. It provides a behavioral reference without requiring a correct answer, while a prefix that reduces its likelihood supplies a direction for disrupting task behavior. We train a generator that receives the request at deployment; training and checkpoint selection remain independent of downstream labels and scoring rules. Here, “request-conditioned” describes the input interface, not a demonstrated matching advantage.

![](images/fa5159316b706c8bd41b6ac0458b5f0b730c6e481992802238aba5181e4d0448.jpg)  
Fig. 1. Attack supervision and an observed utility failure. ENDO-PROMPT uses a victim-generated reference instead of task labels or a target response. On the illustrated GSM8K request, Qwen2.5- 7B abandons the calculation after prefix injection. The request and clean solution are condensed; the attacked response is excerpted verbatim.

ENDOPROMPT alternates prefix search and generator learning (Fig. 2). Search ranks candidate prefixes for the same instruction, after which the generator learns from these preferences and is refined with the same displacement reward. This amortizes victimguided search into offline training and leaves a single generator call for each new request. Our contributions are as follows:

1. We show that a victim-generated continuation can supervise utility-degrading prefixes without task labels or a prescribed failure response.

2. We introduce ENDOPROMPT, which amortizes samerequest victim-side search into one-pass prefix generation from the request text.

3. Across four victims and seven benchmarks, we measure a -26.8-point mean utility change and analyze task redirection, output expansion, and prefix reuse.

![](images/99bf17753c78fae96d8e0c7754f19e12771bafffdbd884f5a726cd11ea9f5c5a.jpg)  
Fig. 2. ENDOPROMPT. (a) Each clean victim continuation serves as a pseudo-reference. The increase in its mean NLL under a candidate prefix determines both the preference ranking and the refinement reward (dashed arrow). The updated generator supplies the next round of candidates. (b) The selected generator produces a prefix that is prepended to the unchanged request before victim inference.

## 2. RELATED WORK

Adversarial text methods span input-specific word substitutions and universal token sequences [5, 6, 2]. Data-free trigger mining reduces dependence on training examples [7], whereas discrete prompt optimization learns text against a specified objective [8, 9]. For aligned LLMs, GCG and BEAST search for jailbreak suffixes, and AmpleGCG learns a generator of such suffixes [3, 10, 11]. AdvPrompter and ProAdvPrompter are particularly relevant because both train input-conditioned generators for harmful compliance [4, 12]. ENDOPROMPT follows this search-to-generator paradigm but derives supervision from clean victim continuations and evaluates benign utility degradation.

Utility loss is also studied through prompt robustness and availability attacks. PromptRobust and worst-prompt evaluation examine sensitivity to wording and format [13, 14], while system-prompt poisoning and guardrail false positives can disrupt benign use [15, 16]. Here we learn user-prefix attacks from victim-side signals without downstream evaluation feedback, then measure transfer on held-out benchmarks. This separates attack discovery on unlabeled instructions from task-performance measurement.

## 3. METHOD

## 3.1. Threat Model and Metric

Let $f _ { \theta }$ denote a frozen instruction-tuned victim and $g _ { \phi }$ a prefix generator. Given a benign request $x ,$ the attacker constructs $z ( x ) =$ $g _ { \phi } ( x )$ and submits $z ( x ) \oplus x$ , where ⊕ denotes concatenation within the user message. The attacker controls only the prefix; the request and system instructions remain unchanged. During offline training, the attacker has white-box access to victim probabilities on unlabeled instructions. Benchmark requests, answers, and scoring rules are held out from both training and checkpoint selection. At deployment, the generator receives only the request text. For benchmark b with evaluation set $\mathcal { D } _ { b }$ , the utility change in percentage points is

$$
\Delta \mathrm { U t i l i t y } _ { b } = \frac { 1 0 0 } { | { \cal D } _ { b } | } \sum _ { x \in { \mathcal { D } } _ { b } } \left[ s _ { b } \big ( f _ { \theta } \big ( g _ { \phi } \big ( x \big ) \oplus x \big ) \big ) - s _ { b } \big ( f _ { \theta } \big ( x \big ) \big ) \right] .\tag{1}
$$

Here $s _ { b } ( \cdot ) ~ \in ~ [ 0 , 1 ]$ is the item-level benchmark score evaluated against the same request and reference in both conditions. Negative $\Delta \mathrm { U t i l i t y } _ { b }$ values indicate degradation.

## 3.2. Victim-Side Pseudo-Reference Reward

For each unlabeled instruction u, the victim greedily generates a clean continuation $\pmb { \tilde { y } } ( u ) = ( \tilde { y } _ { 1 } , \dots , \tilde { y } _ { L } )$ of at most 64 tokens. Let $c _ { \theta } ( \cdot )$ denote the victim chat template. The reward assigned to prefix z is the increase in the mean negative log-likelihood (NLL) of this fixed continuation:

$$
r ( z ; u ) = \frac { 1 } { L } \sum _ { t = 1 } ^ { L } \bigl [ - \log p _ { \theta } ( \tilde { y } _ { t } \mid c _ { \theta } ( z \oplus u ) , \tilde { y } _ { < t } ) \bigr .\tag{2}
$$

Both terms use teacher forcing on the same clean tokens. A positive reward indicates that the prefix reduces the likelihood of the unperturbed continuation; it does not measure task error. The continuation need not be correct because it serves as a behavioral reference for comparing prefixes on unlabeled instructions. We use this displacement reward for candidate ranking and generator refinement, and measure downstream utility separately.

## 3.3. Victim-Guided Prefix Mining and Generator Refinement

ENDOPROMPT learns prefixes from local-search comparisons. The initial generator $R _ { 0 }$ is trained with Eq. (2); each later round samples 64 proposals of at most 16 generator tokens. Teachersearch candidates use the victim tokenizer with a 48-token cap; local search substitutes tokens or truncates prefixes. The nominal search budget is 512 candidates. The archived caches report a maximum of 515 serialized entries under the recorded historical implementation; these entries are a provenance diagnostic, not an additional deployment budget. Candidates are ranked by $r ( z ; u )$ , and the highest- and lowest-ranked eligible prefixes form a pair. We require reward $\geq 0 . 0 2$ and margin $\geq 0 . 0 0 5$ . The same instruction keeps the behavioral reference fixed.

The generator is then fitted to these preferences with a lengthnormalized pairwise objective $( \beta ~ = ~ 0 . 1 )$ , using its pre-fit policy as the reference [17]. GRPO subsequently refines the policy with

Eq. (2) [18], and the updated generator supplies candidates for the next round. We train one generator per victim while keeping victim parameters frozen. During deployment, deterministic generation produces a prefix of at most 16 tokens, followed by a single victim call.

## 4. EXPERIMENTS

## 4.1. Protocol

Table 1. Utility change (attacked minus clean, pp). Victim means average seven benchmarks; Overall averages 28 victim–benchmark cells. Reference attacks retain native objectives and channels; stages and ablations use the ENDOPROMPT protocol.
<table><tr><td colspan="4">ENDOPROMPT by victim</td></tr><tr><td>Victim</td><td>Round</td><td> $\Delta < 0$ </td><td>Mean</td></tr><tr><td>Qwen2.5-7B</td><td> $R _ { 1 }$ </td><td>7/7</td><td>-14.3</td></tr><tr><td>Llama-3.1-8B</td><td> $R _ { 1 }$ </td><td>7/7</td><td>-47.2</td></tr><tr><td>Mistral-7B-v0.3</td><td> $R _ { 2 }$ </td><td>6/7</td><td>-19.7</td></tr><tr><td>Gemma-2-9B</td><td> $R _ { 1 }$ </td><td>7/7</td><td>-26.0</td></tr><tr><td>Overall</td><td>auto-stop</td><td>27/28</td><td>-26.8</td></tr></table>

<table><tr><td colspan="3">Reference attacks</td></tr><tr><td>Method</td><td>Signal / channel</td><td>Mean</td></tr><tr><td>UAT</td><td>LM loss</td><td>-1.9</td></tr><tr><td>GCG-16</td><td>Harmful target</td><td>-7.4</td></tr><tr><td>BEAST</td><td>Harmful target</td><td>-1.4</td></tr><tr><td>AmpleGCG</td><td>Harmful target</td><td>-4.0</td></tr><tr><td>AdvPrompter</td><td>Harmful target</td><td>-4.9</td></tr><tr><td>SPP</td><td>System / folded user</td><td>-16.2</td></tr><tr><td>Ablations</td><td></td><td></td></tr><tr><td>Variant</td><td>Intervention</td><td>Mean</td></tr><tr><td>Before fitting</td><td>Pre-fit generator</td><td>-23.6</td></tr><tr><td>After fitting</td><td>Post-fit generator</td><td>-5.5</td></tr><tr><td>Remove preference loop</td><td>Direct reward refinement</td><td>-20.4</td></tr><tr><td>Shuffled pseudo-reference</td><td>Permuted continuations</td><td>-16.1</td></tr></table>

We evaluate the instruction-tuned Qwen2.5-7B, Llama-3.1-8B, Mistral-7B-v0.3, and Gemma-2-9B checkpoints [19, 20, 21, 22] on GSM8K, MATH-500, BBH, HellaSwag, TruthfulQA, IFEval, and MMLU [23, 24, 25, 26, 27, 28, 29]. The complete splits contain 1,319, 500, 6,511, 10,042, 817, 541, and 14,042 items, respectively. GSM8K, BBH, and MMLU use 4, 3, and 5 demonstrations; the remaining benchmarks use none. Clean and attacked conditions share the formatted request, chat template, greedy decoding with a 512- token limit, and scorer. HellaSwag, TruthfulQA, and MMLU use a common answer extractor; the remaining tasks use native scoring, including strict prompt-level IFEval. Each of the 28 cells contributes equally to the overall mean. Conditional on the evaluated checkpoints, confidence intervals use a 20,000-replicate paired bootstrap within benchmarks, with common item resamples across victims.

Each victim’s generator is initialized from the Dolphin3.0- Llama3.2-3B checkpoint [30]; all runs use seed 42. Training uses 2,400 instructions from Dolly-15k after discarding the original responses and labels [31]. Each GRPO stage runs for at most 260 updates with reward-based early stopping. The search stages that precede the selected checkpoints use victim-context limits of 4,096 tokens for Qwen, Llama, and Gemma and 32,768 for Mistral; reward refinement and checkpoint selection use 4,096 throughout. Inference uses the same greedy 16-token prefix cap for all victims.

Model selection uses no downstream utility. After each round, mean displacement reward is measured on 600 disjoint Dolly instructions. The first search–fit–refine cycle sets $R _ { 1 }$ as the incumbent; later rounds replace it only after a held-out gain of at least 0.03. Training stops after two successive rejections or at $R _ { 6 } ,$ , and evaluation uses the retained incumbent, which may precede the final iterate (Fig. 3).

![](images/f93c733a2a0acb4975a1e527e142b28b875eac7b15b2b17047f476abf9e36a53.jpg)  
Fig. 3. Checkpoint selection on 600 held-out unlabeled instructions per victim. Diamonds mark retained checkpoints; replacement requires a reward gain of at least 0.03. Vertical scales differ.

Table 1 also reports UAT, GCG-16, BEAST, AmpleGCG, AdvPrompter, and system-prompt poisoning (SPP) on the same splits. Their native objectives and channels are preserved; these rows are contextual references rather than a matched-objective ranking. The table also reports the ENDOPROMPT ablations analyzed below.

## 4.2. Utility Degradation Across Victims

The aggregate utility change is -26.8 percentage points (95% CI: [-27.4, -26.2]), and 27 of 28 victim–benchmark cells show degradation. All four victim means are negative, while Mistral on MMLU is the sole positive cell (Table 1 and Fig. 4). The learned prefixes therefore reduce performance on nearly all evaluated victim–benchmark cells, with the magnitude varying by victim and benchmark.

<table><tr><td>Qwen</td><td>-3.3</td><td>-10.6</td><td>-52.1</td><td>-8.3</td><td>-1.7</td><td>-15.9</td><td>-8.3</td><td rowspan="4">100 50 AUui i(p) 0 -50 -100</td></tr><tr><td>Llama</td><td>-70.1</td><td>-38.0</td><td>-56.6</td><td>-64.5</td><td>-49.7</td><td>-47.9</td><td>-3.6</td></tr><tr><td>Mistral</td><td>-39.1</td><td>-7.4</td><td>-12.5</td><td>-41.0</td><td>-32.2</td><td>-7.4</td><td>1.8</td></tr><tr><td>Gemma</td><td>-47.2</td><td>-8.6</td><td>-33.9</td><td>-14.6</td><td>-67.0</td><td>-8.5</td><td>-2.2</td></tr><tr><td>GSM</td><td colspan="2">BBH MATH</td><td>HS</td><td>TQA</td><td>IFE</td><td>MMLU</td><td></td></tr></table>

Fig. 4. Utility change by victim and benchmark (pp; attacked minus clean). Negative values denote degradation. HS, TQA, and IFE denote HellaSwag, TruthfulQA, and IFEval.

## 4.3. Search and Round Selection

In 96.5% of 9,600 victim–instruction cases, the searched winner exceeds the original best proposal after both are rescored in the same search pass; the mean gain is 0.125 NLL/token. This teacher-side comparison excludes cross-pass scoring drift and measures search improvement rather than downstream utility. The search stages preceding the selected checkpoints supply 9,597 eligible withininstruction preference pairs.

The held-out reward trajectories differ across victims (Fig. 3). The common rule retains $R _ { 1 }$ for Qwen, Llama, and Gemma, and $R _ { 2 }$ for Mistral: only the latter clears the 0.03 replacement margin.

The trajectory is non-monotonic: utility change moves from - 23.6 before fitting to -5.5 after fitting, then reaches -26.8 after refinement (Table 1). Removing the preference loop yields -20.4 and permuting pseudo-references yields -16.1, respectively 6.4 and 10.7 pp less negative than the complete pipeline. These matched ablations implicate both components, while request alignment is tested separately below.

## 4.4. Output-Level Failure Analysis

(a) Responses after utility loss  
![](images/8bd467a3cdac71d6bba19c57c221bea456050d4e5f2bb57e231695db7868244f.jpg)

(b) Prefix diversity and concentration  
![](images/12bcf2490ca04458d2adf732f1344e265b8e2382d0cf795524025c6657772856.jpg)  
Fig. 5. Failure responses and prefix reuse; n denotes each panel’s denominator. (a) Response characteristics among requests that change from correct to incorrect. Categories overlap; refusal-like indicates a fixed lexical match. (b) Distinct-prefix ratio and frequency of the most common prefix over all requests after case and whitespace normalization.

Among 36,750 clean-correct/attacked-wrong victim–request pairs, 58.4% contain at least twice as many output tokens and 39.5% contain no extracted answer under the stated parsers (Fig. 5(a)). Only 0.3% match the fixed refusal lexicon. These indicators overlap and are pooled over regression pairs, whereas the main result is an equal-cell mean; none is a semantic failure label.

The GSM8K example in Fig. 1 illustrates task redirection: the attacked response describes solving the problem metaphorically but never performs the calculation, without a task-specific error target during training.

Prefix reuse differs markedly across victims. After case and whitespace normalization, the distinct-prefix ratio is 2.5%, 4.7%, 0.4%, and 1.9% for Qwen, Llama, Mistral, and Gemma, respectively (Fig. 5(b)). The most common Mistral prefix appears in 65.0% of requests, compared with 24.8% for Qwen. Prefix behavior therefore ranges from varied outputs to extensive reuse; request conditioning alone does not imply a unique prefix for every request.

The ablations and controls answer different questions. Retraining ablations hold deployment and evaluation fixed while changing the training signal; fixed-generator controls keep the learned policy fixed and change only request–prefix assignment. The former support the contribution of the preference and pseudo-reference stages under this protocol, whereas the latter leave the causal value of request matching unresolved. Together, the results support victim-side supervision and one-pass deployment without implying a uniquely tailored prefix for every request.

The fixed-generator controls provide no evidence that alignment strengthens degradation: matched-minus-shuffled is +0.21 pp (95% CI [-0.06, +0.48]) and matched-minus-empty is -0.30 pp (95% CI [-0.65, +0.06]). Reusing each modal prefix increases degradation by 2.79 pp (95% CI [2.46, 3.13]); thus, these controls do not establish an alignment advantage. The modal gap is driven by Gemma (+6.61 pp) and Llama (+5.04 pp); Qwen and Mistral differ by less than 0.3 pp. Because this control reuses an observed generator output, it is a diagnostic of prefix concentration rather than a trained unconditional-generator baseline.

Equation (2) uses neither benchmark answers nor a target failure string, yet the learned prefixes affect tasks with different output spaces and scoring rules. Clean continuations provide a behavioral reference, while downstream losses include wrong, missing, and instruction-violating answers on tasks excluded from attack training.

## 4.5. Security Implications and Limitations

Taken together, the experiments support white-box discovery of utility weaknesses from victim behavior, including fluent responses that abandon benign requests. Benign task performance under injected text should therefore be measured directly, not inferred from harmful compliance or refusal rates.

Response diagnostics show that refusal is rare whereas response expansion and extraction failure are common; benchmark scores should therefore be paired with direct examination of generated responses.

The generator often returns the same prefix for different inputs (Fig. 5(b)); robustness evaluation should examine both assigned pairs and frequently generated strings.

These conclusions apply to offline probability access, openweight victims, and user-prefix insertion; closed-model transfer and indirect injection remain outside scope. One seed per victim means the intervals do not estimate retraining variability. A matched unconditional-generator branch is still needed to isolate the causal value of request input; the reported ablations test fitting and pseudoreference permutation.

## 5. CONCLUSION

We introduced ENDOPROMPT, which uses victim continuations to supervise one-pass prefix generation. Across four victims and seven benchmarks, the prefixes reduce benign performance without downstream labels or a prescribed failure response. Controls do not establish an alignment advantage, and prefix reuse shows that the generator can remain concentrated; the arithmetic example illustrates fluent task abandonment. These findings support white-box utility auditing alongside direct evaluation of legitimate task preservation.

## Compliance with Ethical Standards

This work studies a dual-use threat to benign model utility. Experiments use open-weight models and public benchmarks, with no attacks on private data, deployed services, or users. The findings can inform detection and mitigation. Potentially harmful attack artifacts are made available only to vetted researchers under responsible disclosure. OpenAI GPT-5.6 was used for language polishing and to assist with parts of the experimental code and visualizations. The authors reviewed, tested, and verified this assistance and remain responsible for the manuscript, code, and reported results.

## 6. REFERENCES

[1] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz, “Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection,” in Proc. ACM Workshop Artif. Intell. Secur. (AISec), 2023, pp. 79–90.

[2] Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh, “Universal adversarial triggers for attacking and analyzing NLP,” in Proc. EMNLP-IJCNLP, 2019, pp. 2153– 2162.

[3] Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint arXiv:2307.15043, 2023.

[4] Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian, “AdvPrompter: Fast adaptive adversarial prompting for LLMs,” in Proc. Int. Conf. Mach. Learn. (ICML), 2025, vol. 267, pp. 48439–48469.

[5] Di Jin, Zhijing Jin, Joey Tianyi Zhou, and Peter Szolovits, “Is BERT really robust? a strong baseline for natural language attack on text classification and entailment,” in Proc. AAAI Conf. Artif. Intell., 2020, vol. 34, pp. 8018–8025.

[6] Linyang Li, Ruotian Ma, Qipeng Guo, Xiangyang Xue, and Xipeng Qiu, “BERT-ATTACK: Adversarial attack against BERT using BERT,” in Proc. EMNLP, 2020, pp. 6193–6202.

[7] Swapnil Parekh, Yaman Singla Kumar, Somesh Singh, Changyou Chen, Balaji Krishnamurthy, and Rajiv Ratn Shah, “MINIMAL: Mining models for data free universal adversarial triggers,” arXiv preprint arXiv:2109.12406, 2021.

[8] Taylor Shin, Yasaman Razeghi, Robert L. IV Logan, Eric Wallace, and Sameer Singh, “AutoPrompt: Eliciting knowledge from language models with automatically generated prompts,” in Proc. EMNLP, 2020, pp. 4222–4235.

[9] Yuxin Wen, Neel Jain, John Kirchenbauer, Micah Goldblum, Jonas Geiping, and Tom Goldstein, “Hard prompts made easy: Gradient-based discrete optimization for prompt tuning and discovery,” in Adv. Neural Inf. Process. Syst., 2023, vol. 36, pp. 51008–51025.

[10] Vinu Sankar Sadasivan, Shoumik Saha, Gaurang Sriramanan, Priyatham Kattakinda, Atoosa Chegini, and Soheil Feizi, “Fast adversarial attacks on language models in one GPU minute,” in Proc. Int. Conf. Mach. Learn. (ICML), 2024, vol. 235, pp. 42976–42998.

[11] Zeyi Liao and Huan Sun, “AmpleGCG: Learning a universal and transferable generative model of adversarial suffixes for jailbreaking both open and closed LLMs,” in Proc. Conf. Lang. Model. (COLM), 2024.

[12] Hao Di et al., “ProAdvPrompter: A two-stage journey to effective adversarial prompting for LLMs,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025.

[13] Kaijie Zhu et al., “PromptRobust: Towards evaluating the robustness of large language models on adversarial prompts,” in Proc. ACM Workshop Large AI Syst. Models Privacy Saf. Anal., 2024, pp. 57–68.

[14] Bowen Cao, Deng Cai, Zhisong Zhang, Yuexian Zou, and Wai Lam, “On the worst prompt performance of large language models,” in Adv. Neural Inf. Process. Syst., 2024, vol. 37, pp. 69022–69042.

[15] Zongze Li, Jiawei Guo, and Haipeng Cai, “System prompt poisoning: Persistent attacks on large language models beyond user injection,” arXiv preprint arXiv:2505.06493, 2025.

[16] Qingzhao Zhang, Ziyang Xiong, and Z. Morley Mao, “LLM safeguard is a double-edged sword: Exploiting false positives for denial-of-service attacks,” in Proc. Workshop Large AI Syst. Models Privacy Secur. Anal., 2025, pp. 1–10.

[17] Mohammad Gheshlaghi Azar et al., “A general theoretical paradigm to understand learning from human preferences,” in Proc. Int. Conf. Artif. Intell. Stat. (AISTATS), 2024, vol. 238, pp. 4447–4455.

[18] Zhihong Shao et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024.

[19] An Yang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[20] Aaron Grattafiori et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[21] Albert Q. Jiang et al., “Mistral 7B,” arXiv preprint arXiv:2310.06825, 2023.

[22] Gemma Team, Morgane Riviere, et al., “Gemma 2: Improving open language models at a practical size,” arXiv preprint arXiv:2408.00118, 2024.

[23] Karl Cobbe et al., “Training verifiers to solve math word problems,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.

[24] Dan Hendrycks et al., “Measuring mathematical problem solving with the MATH dataset,” in Adv. Neural Inf. Process. Syst., 2021, vol. 34.

[25] Mirac Suzgun et al., “Challenging BIG-Bench tasks and whether chain-of-thought can solve them,” in Findings ACL, 2023, pp. 13003–13051.

[26] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi, “HellaSwag: Can a machine really finish your sentence?,” in Proc. ACL, 2019, pp. 4791–4800.

[27] Stephanie Lin, Jacob Hilton, and Owain Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in Proc. ACL, 2022, pp. 3214–3252.

[28] Jeffrey Zhou et al., “Instruction-following evaluation for large language models,” arXiv preprint arXiv:2311.07911, 2023.

[29] Dan Hendrycks et al., “Measuring massive multitask language understanding,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[30] Eric Hartford, Ben Gitter, BlouseJury, and Cognitive Computations, “Dolphin 3.0 Llama 3.2 3B,” Hugging Face model card, dphn/Dolphin3.0-Llama3.2-3B, 2025.

[31] Mike Conover et al., “Free dolly: Introducing the world’s first truly open instruction-tuned LLM,” Databricks Blog, 2023.