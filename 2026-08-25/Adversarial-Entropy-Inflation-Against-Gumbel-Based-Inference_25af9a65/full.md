# Adversarial Entropy Inflation Against Gumbel-Based Inference Verification

Nikita Kezins Delft University of Technology nikitakezins@gmail.com

## Abstract

Gumbel-based inference verification bounds LLM weight exfiltration by only forgiving token choices that plausibly arise from honest GPU nondeterminism, reporting a > 200× slowdown for a steganographic adversary under benign prompt traffic. This bound assumes a passive attacker; we show it degrades sharply against an adversary who instead controls the prompt distribution. Because the verifier’s admissible-token-set size is driven by the model’s own output entropy, prompts engineered to break grammatical and sub-word structure – rather than benign conversational traffic – widen that set and open a materially larger covert channel. Across six instruction-tuned models spanning 1B to 32B parameters and three random seeds, our strongest attack (character- and script-level disruption) roughly doubles bits leaked per token relative to benign prompts, cutting the slowdown factor to 60–118×. These results indicate that static, benign-traffic-calibrated thresholds are insufficient for this defense, and that jitter-forgiveness thresholds should instead be calibrated dynamically against local token entropy.

## 1 Introduction

As organizations scale their artificial intelligence capabilities, proprietary models become highly valuable assets, raising significant concerns regarding model weight exfiltration. One vector for such exfiltration is through standard API inference, where malicious users can encode and leak model weights via deliberate token selection. This technique effectively establishes a covert subliminal channel Simmons [1984] within the model’s standard output stream.

As a defense against this, Rinberg et al. [2026] proposed verifying sampled tokens under a predefined seed selection, building on the same Gumbel-margin statistic used more broadly for detecting inference misconfiguration and tampering Karvonen et al. [2025]. By utilizing Gumbel sampling, their method ensures nearly identical token selection under an argmax policy for both the verifier and the prover. While theoretically sound, this approach is vulnerable to GPU non-determinism (jitter) caused by non-associative floating-point operations. When multiple tokens have near-identical likelihoods, this jitter can flip the argmax token selection. The necessity for verifiers to forgive these jitter-induced discrepancies inadvertently creates an attack gap that an active exfiltrator can exploit.

Prior work demonstrated that under benign input conditions (e.g., a subset of the LMSYS Chatbot Arena dataset Zheng et al. [2023]), only a small fraction of tokens exhibit branching – where multiple tokens share similar likelihoods, allowing GPU jitter to alter the output – with the rest exactly determined (see Methodology for the formal admissible-set definition). This underlies the reported 200x slowdown in exfiltration rates relative to the theoretical maximum.

Because the capacity of this covert channel is strictly bound by the entropy of the token distribution, widening the attack gap requires an exfiltrator to maximize output randomness. In this work, we demonstrate that by deliberately breaking token dependencies and maximizing output entropy via adversarial prompting, an attacker can double the exfiltration capacity, reducing the defense’s slowdown factor from 200x to approximately 100x.

## 2 Methodology

Preliminaries. The GLS verifier of Rinberg et al. [2026] reconstructs, per output position, the Gumbel-perturbed logits the prover’s sampler would use under a shared seed, then defines the admissible set $A _ { t }$ as tokens within tolerance σ of the top Gumbel score – i.e. tokens a compliant prover could plausibly emit under GPU nondeterminism. Verification passes iff the observed token lies in $A _ { t } ;$ residual exfiltration bandwidth at that position is bounded by log<sub>2</sub> $| A _ { t } |$ bits. On benign traffic $A _ { t }$ is a singleton (“forced”) almost everywhere, giving the reported $> 2 0 0 \times$ slowdown. Since $| A _ { t } |$ tracks the model’s own output entropy, an adversary who raises that entropy through prompt design directly widens $A _ { t } -$ the attack surface we study.

We build upon the experimental framework provided by Rinberg et al. [2026]. To establish a baseline, we evaluate a subset of 150 benign prompts from the LMSYS dataset Zheng et al. [2023], matching the original study. We then design three tiers of adversarial attacks, each progressively escalating the attempt to induce high-entropy distributions. We generate 150 adversarial prompts for each attack type.

TERSE (Low-Context Generation): 722 combinatorially generated one-word fill-in-the-blank and low-context prompts (e.g. single-word mood/preference/description completions, terse conversational openers). The hypothesis is that genuine uncertainty concentrates on the first token of a response; isolating it rather than diluting it across a longer sequence should widen the set of admissible tokens.

SCRAMBLE (Grammatical Dependency Disruption): 233 prompts across four categories: unrelated-item lists with explicit anti-correlation instructions, pseudo-random password/hash/PIN generation, gibberish-syllable continuation, and short surrealist continuations. This targets local word-to-word grammatical dependencies, fundamentally undermining the structure that makes fluent prose low-entropy.

CIPHER (Sub-Word and Token-Level Disruption): 157 prompts across four categories: unbroken alphanumeric blocks, base64/hex/base32 dumps, gibberish-prefix “context bombs” the model must continue verbatim, and multi-script character collisions (Latin/Cyrillic/Katakana/Arabic). Because even “unrelated” word lists rely on standard vocabulary, this attack targets character-level distributions directly to maximize entropy at the token boundary.

We executed these four prompt sets (benign, plus three attacks; 150 prompts each) across three distinct random seeds. To ensure generalizability across different architectures and scales, we evaluated six instruction-tuned models: Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, and Qwen2.5-14B, Qwen3- 30B-MoE, Qwen3-32B.

## 3 Experiments and Results

Table 1: Impact of adversarial prompts on exfiltration bandwidth and defense slowdown. CIPHER significantly increases bits extracted per token and drastically reduces the verifier’s slowdown multiplier compared to the benign baseline.
<table><tr><td rowspan="2">Model</td><td colspan="4">Bits Extracted per Token (↑)</td><td colspan="4">Slowdown Multiplier (↓)</td></tr><tr><td>Benign</td><td>TERSE</td><td>SCRAMBLE</td><td>CIPHER</td><td>Benign</td><td>TERSE</td><td>SCRAMBLE</td><td>CIPHER</td></tr><tr><td>1B</td><td>0.085±0.007</td><td> $0 . 0 7 3 { \scriptstyle \pm 0 . 0 0 4 }$ </td><td> $0 . 1 3 2 { \scriptstyle \pm 0 . 0 0 9 }$ </td><td>0.160±0.007</td><td> $2 0 1 { \pm } 1 7 \times$ </td><td> $2 3 2 { \pm } 1 2 \times$ </td><td> $1 2 9 { \pm } 9 \times$ </td><td> $1 0 6 \pm 5 \times$ </td></tr><tr><td>3B</td><td>0.067±0.006</td><td> $0 . 0 4 8 { \pm } 0 . 0 0 2$ </td><td> $0 . 1 0 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td>0.146±0.011</td><td> $2 5 4 { \pm } 2 2 \times$ </td><td> $3 5 2 \pm 1 0 \times$ </td><td> $1 6 6 { \pm } 1 1 \times$ </td><td> $1 1 7 \pm 8 \times$ </td></tr><tr><td>8B</td><td>0.068±0.002</td><td> $0 . 0 5 8 { \pm } 0 . 0 1 1$ </td><td> $0 . 1 1 2 { \scriptstyle \pm 0 . 0 1 0 }$ </td><td> $0 . 1 8 3 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $2 5 2 \pm 7 \times$ </td><td> $3 0 4 { \pm } 6 8 \times$ </td><td> $1 5 3 { \pm } 1 4 \times$ </td><td> $9 3 \pm 6 \times$ </td></tr><tr><td>14B</td><td>0.093±0.002</td><td> $0 . 1 0 3 { \scriptstyle \pm 0 . 0 0 7 }$ </td><td> $0 . 1 6 7 { \scriptstyle \pm 0 . 0 1 2 }$ </td><td> $0 . 1 6 7 { \scriptstyle \pm 0 . 0 1 6 }$ </td><td> $1 8 6 \pm 4 \times$ </td><td> $1 6 8 { \pm } 1 1 \times$ </td><td> $1 0 3 { \pm } 7 \times$ </td><td> $1 0 4 { \pm } 9 \times$ </td></tr><tr><td>30B-MoE</td><td>0.119±0.011</td><td> $0 . 1 4 5 { \scriptstyle \pm 0 . 0 1 1 }$ </td><td> $0 . 2 5 8 { \pm } 0 . 0 1 5$ </td><td>0.286±0.013</td><td> $1 4 6 { \pm } 1 5 \times$ </td><td> $1 2 0 { \pm } 9 \times$ </td><td> $6 7 \pm 4 \times$ </td><td> $6 0 \pm 3 \times$ </td></tr><tr><td>32B-dense</td><td>0.070±0.001</td><td> $0 . 0 7 2 { \scriptstyle \pm 0 . 0 0 8 }$ </td><td> $0 . 0 9 9 { \scriptstyle \pm 0 . 0 0 5 }$ </td><td> $0 . 1 4 5 { \scriptstyle \pm 0 . 0 0 3 }$ </td><td> $2 4 8 { \pm } 4 \times$ </td><td> $2 4 1 { \pm } 2 4 \times$ </td><td> $1 7 4 { \pm } 8 \times$ </td><td> $\mathbf { 1 1 8 } { \pm } 3 \times$ </td></tr></table>

As shown in Figure 1 and Table 1, adversarial prompt design significantly impacts the efficacy of Gumbel-sampling-based verification. We highlight three core findings:

1. Shattering the Slowdown Multiplier: Table 1 validates the claimed > 200× benign-case slowdown (e.g., 248× on 32B-dense), but CIPHER systematically dismantles it, dropping the slowdown to 118× on 32B-dense and 60× on 30B-MoE.

Benign TERSE SCRAMBLE CIPHER

![](images/44b20b3caae41b64abd32d84c5e082d52585a213cecf4c39fe755651e89fae26.jpg)

![](images/81419ec2bf58737323e2b8753368346c0b80a2292a8409bf929fc925cf35c656.jpg)

![](images/6e9636340086dfb06cc8b5b30cea3dd63b3be21921ff3d312f4e6913d37fe29e.jpg)

![](images/1a498a62b5360c83b9cb70fc77cbf92a26c643610236ededd1a71dc943b732c2.jpg)

![](images/1049f9c884784f75b06dec5a68a2e711836133180ac925857b249c808a5eb0a6.jpg)

![](images/d8ba78e927c93407ae369a71999ae3cefa3a60f7a6273ac84e5e1d389b09e64a.jpg)

![](images/53792f6bbe564584417aa5395c7ff5ef4533fdd1b2beb7b260e5fa03c1652f50.jpg)

![](images/8efc5026938bb84748b78859e5c77f548e8a1257171f6e1a1f58001b83703a8f.jpg)  
Figure 1: Top Left: Extractable info (vocab-size-corrected) at verifier tolerance $\sigma = 1 . 0$ . Top Right: Relative exfiltration gain vs. benign baseline. Bottom: FPR vs. Extractable Info Pareto curves. CIPHER maximizes covert bandwidth; TERSE collapses on smaller models.

2. Sub-word Disruption Maximizes Leakage: CIPHER (context-bomb/multi-script) consistently achieves the highest exfiltration capacity. By attacking character-level priors, it prevents models from collapsing onto common vocabulary, yielding up to a 170% relative increase over benign prompts and peaking at 0.286 bits extracted per token on Qwen3-30B-MoE.

3. Low-Context Prompts Induce Collapse: TERSE underperforms the benign baseline on sub-14B models. Restricting context causes smaller models to default to high-probability tokens, ironically collapsing the output distribution into a near-deterministic state and exacerbating the defense’s slowdown (peaking at 352× on the 3B model).

## 4 Conclusion

Gumbel-based verification’s security guarantees are heavily contingent on output entropy: adversarial prompts that disrupt syntactic and sub-word priors widen the jitter-induced attack gap, slashing the purported 200× slowdown to 60×–118× depending on architecture. Verifier designs cannot rely on static, benign-calibrated baselines; jitter forgiveness thresholds must instead be calibrated dynamically against local token entropy. This is not unique to the unilateral weight-exfiltration setting – broader bilateral compute-verification architectures built on the same unexplained-information bound Petrie and Mühlhäuser [2026] inherit an identical entropy-dependent attack surface.

## Acknowledgments and Disclosure of Funding

This work was conducted during the Hardware Assurance Program at the Cambridge AI Safety Hub (CAISH).

## References

Adam Karvonen, Daniel Reuter, Roy Rinberg, Luke Marks, Adrià Garriga-Alonso, and Keri Warr. Difr: Inference verification despite nondeterminism, 2025. URL https://arxiv.org/abs/ 2511.20621.

James Petrie and Yannick Mühlhäuser. Verifying AI compute by bounding unexplained information exfiltration. In ICML Workshop on Technical AI Governance Research, 2026. URL https: //openreview.net/forum?id=qtgG5HZSsk.

Roy Rinberg, Adam Karvonen, Alexander Hoover, Daniel Reuter, and Keri Warr. Verifying llm inference to detect model weight exfiltration, 2026. URL https://arxiv.org/abs/2511. 02620.

Gustavus J Simmons. The prisoners’ problem and the subliminal channel. In Advances in Cryptology: Proceedings ofCrypto 83, pages 51–67. Springer, 1984.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Tianle Li, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zhuohan Li, Zi Lin, Eric. P Xing, Joseph E. Gonzalez, Ion Stoica, and Hao Zhang. Lmsys-chat-1m: A large-scale real-world llm conversation dataset, 2023.