COMPRESSION TRINITY: EXPLORING SPARSITY, QUANTIZATION, AND LOW-RANK APPROXIMATIONS FOR LLM COMPRESSION

by

Mohammad Mozafari

A thesis submitted in conformity with the requirements for the degree of Doctor of Philosophy Department of Computer Science University of Toronto

© Copyright 2026 by Mohammad Mozafari

# Compression Trinity: Exploring Sparsity, Quantization, and Low-Rank Approximations for LLM Compression

Mohammad Mozafari   
Doctor of Philosophy   
Department of Computer Science   
University of Toronto   
2026

## Abstract

Prohibitive computational and environmental costs impede the scalable deployment of Large Language Models (LLMs). Traditional compression techniques (sparsity, quantization, low-rank approximations) are typically applied in isolation, limiting efectiveness as each hits an accuracy-eficiency wall. Addressing this is critical for sustainable LLM deployment.

This thesis proposes the “Compression Trinity,” a unified framework applying these pillars jointly. By leveraging sparsity to reduce computation, quantization to minimize memory bandwidth, and low-rank approximations to recover accuracy, the framework unlocks new eficiency frontiers. We demonstrate these orthogonal methods are complementary, overcoming the limitations of isolated strategies.

To accelerate pretraining, we apply the Trinity to the optimizer and model architecture. MKOR approximates curvature via block-diagonal sparsity and low-rank inversion, maintaining numerical stability for quantized states. It reduces curvature update complexity from $\mathcal { O } ( d ^ { 3 } )$ to $\mathcal { O } ( d ^ { 2 } )$ , accelerating convergence by up to 1.85× over KFAC. SLOPE accelerates training by up to 1.25× via a double-pruned backward pass for N:M sparsity, using low-rank “lazy” adapters in the final 1% of training to recover accuracy.

For post-training compression, we progressively explore the Trinity to improve deployment eficiency. OP-TIMA establishes sparsity limits under strict resource constraints, stabilizing static masks in a zero-training regime by formulating weight reconstruction as globally optimal column-wise quadratic programs, improving zero-shot accuracy by up to 3.97%. Given a fine-tuning budget, PATCH breaks the ceiling of static masks by learning a dynamic hybrid sparsity ratio between 0% and 50%, yielding up to 1.38× speedups. Finally, to overcome the structural limits of sparsity, SLIM realizes the full Compression Trinity in one shot using mathematically derived low-rank adapters to recover information lost to quantization and sparsity, improving accuracy by up to 5.66% over state-of-the-art methods and outperforming uncompressed dense models at equal parameter budgets by 0.6%.

Together, these contributions demonstrate that the joint application of the Compression Trinity is essential for unlocking the next generation of eficient, scalable, high-performance LLMs.

## Acknowledgements

First and foremost, I would like to express my deepest gratitude to my family. To my father, Gholamhossein, and my mother, Pouran, whose unwavering love, sacrifices, and encouragement have been the foundation of everything I have achieved. To my sister, Maryam, for her constant support and for always believing in me. This thesis is as much yours as it is mine.

I would like to thank my supervisor, Professor Maryam Mehri Dehnavi, for providing the environment and resources that made this research possible.

I am grateful to my thesis committee members, Professor Angela Demke Brown, Professor Dan Roy, Professor Nandita Vijaykumar, and Professor Murat Erdogdu, for their valuable feedback and thoughtful questions that strengthened this work. I would also like to extend my thanks to Professor Dan Alistarh for serving as the external appraiser and for taking the time to carefully evaluate this thesis.

I owe a special debt of gratitude to Amir Yazdanbakhsh, who has been a mentor, collaborator, and source of inspiration throughout much of my research journey. His guidance and generosity with his time have profoundly shaped the way I approach problems. I am also grateful to Zhao Zhang for being a wonderful collaborator and for the many productive discussions we shared.

I would like to acknowledge my undergraduate supervisors, Professor Maryam Sabbaghiyan and Professor Amir Masoud Rabiei, who first sparked my passion for research and set me on this path. Their early mentorship laid the groundwork for everything that followed.

I have been fortunate to share the lab with exceptional friends and colleagues who made the journey both intellectually stimulating and enjoyable. I would like to thank Behrooz Zarebavani, Lucas Wilkinson, Younes Hourri, Kazem Cheshmi, Saeed Soori, Bangtian Liu, Avery Laird, Arya Rafii, Victor Kamel, Ray Hung, Kasra Jahankhani, Lucy Farcnik, Milad Khanchi, and Amirhossein Elmi for the countless conversations, collaborations, and moments of friendship.

To everyone who has contributed to this work, whether through direct collaboration or simply through friendship and encouragement, thank you.

## ēïà ŌǍűçŬόǇ

ėόˁ ،êàïŪ  Ŋ ،Ě <sup>ō</sup>ąώĜ ŔʉŘ ïíąόĶ ù ،ë Ŵ̭<sup>Ʌ̘</sup>ęĎ ȵ ،ý Ō<sup>ʷ</sup> <sub></sub>ŏόȼ ïîόĜ ïà .Ě <sup>Žʾ</sup> ĕ<sup>ń</sup> ýà ĘíàŪŊçόŰ Ě <sup>ō</sup>î̦ Ĺ àï íũʠ ņ<sup>ŉ</sup>àíïî όĲ ù ðçŬόǇ ë ĸŌ<sup>ʷ</sup>ſŻ<sup>ʥɼ</sup> ă İàĠĝ ، ïçόȵà ïí ٓ êç  ŬtųŬ Ɋ<sup>ķ</sup> Ęïàũʦ<sup>ɓ</sup> ėόˁ ،Ě <sup>ō</sup>Ġĝ ،ýŏόʚàũʠ ïà .éό Ǉà ĘíŪŊ ëό  ǒ ēçόʘíïùç<sub></sub>ŬόǇí ĕ<sup>ń</sup>ç̪ Ť êç  Ŭųě êç  όű ėȢĻù ņ <sup>ŉ</sup> ēçόʘ ĕ<sup>ń</sup>Ō<sup>Ǎ</sup>ưí ù çόʘ ēïçόīàî όĲ ،õĺ ïí ņ <sup>ŉ</sup> ſɌ<sup>ȶ</sup> .éό Ǉç̪ Ƌ <sub>ِ</sub>ê àٓ ïà ،éό Ǉà ëό  ǒ <sub>ِ</sub>ê àٓ ïà ėόˁ Ę ïàî<sub></sub>όĜà êç̪ό  ɓ ėĿ ėƪçόűï ë ĸà .éό Ǉà ĘîŬɊ<sup>˄</sup><sup>Ĝ</sup> éό Ǉí ëό  ǒ ėĿ ë Ŵűàí êç̪  Ťà ïà Ęçό˭ ġ  t<sup>Ə</sup> ù ĘíŪŊ ëό  ǒ ،é ǥçόű ĠّĪŹĶ àï ċό <sup>ʙ</sup>ù Ō<sup>ʷ</sup> ë ĸà ýç̩ <sup>ň</sup>à ėόˁ ņ<sup>ŉ</sup>ą<sub></sub>όĜçƠόȟà ù ĢƈɅ êíïù  à Ěό ٓ àŕ<sup>Œ</sup>ŏόţçόŰ ėĿ ،ēũ Ż<sup>ʙ</sup>í ēŔʉŘ ĚĠĝ ïũɌ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ،Ě ç̪ƍ<sup>ʙ</sup>àï íç<sub></sub>ŬόǇà ïà .ýïà Ō<sup>Ǎ</sup>űçŬόǇ ė<sub></sub>Ŀç̪ ƍ<sup>ʥɆ</sup>

ïũɌ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ù ïąόĶŪόƞ ĕ<sup>ɑ</sup> <sup>ň</sup>ù ç<sub></sub>Ŭįî<sub></sub>όĜą<sub></sub>όĜ ïũɌ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ،ēùï êí ïũ  Ɍ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ،êùàŌ  <sup>ʷ</sup> ĕ<sup>ʵĶ</sup>í ęĎ̩ <sup>ň</sup>à ïũ ٓ Ɍ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ،ýà ėƪçόűï êàïùàí  é ٔŮ <sup>ųʘ</sup> ýĠźʁ<sup>̘</sup> ēç ̧όȗà ïà ù Ō<sup>ʸ</sup>̭ <sup>ķ</sup> ûç̪ό˒،îύĜíŕό<sup>Ʀ</sup>Ō<sup>ʷ</sup>à ë ĸà ýçƠɍ W<sup>ƎǇ</sup>à é ǥžόǔ ėόˁ êç  όű ė<sub></sub>Ŀç̭ķî<sub></sub>όĜà öï ï ēçόʘ ñό <sup>Ŗ</sup>Ō<sup>ʷ</sup> ù î<sub></sub>Ŭʥ <sup>Ƌ</sup> ïïà ēçόʘíïũʠ ïąόĜ ŏόţçόŰ ėĿ ،ùî όȵùíïà èàïžό ǔ öĠ Ő ėƪçόűï ë ĸà ſŻ<sup>Ğ</sup>í ĕ<sup>ɂ</sup>ïŌ<sup>ʷ</sup> ēàŌ<sup>ʷ</sup> ėόˁ ĕ<sup>ż</sup><sup>Ğ</sup>ù ù ĕ<sup>ȇ</sup>ïçόŰ èą  ύĜ ïïà ñ<sup>Ȧ</sup> <sup>Ĺ</sup> ðŌ <sup>ʷ</sup> îόĜ ŏόţçόŰ ėĿ ïç<sub></sub>ŬɊ<sup>ŹƘ</sup>àٓêí ïũ  Ɍ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ïà ë Ŵƒ<sup>Ʌ</sup> .ýïàí àï êç  <sub></sub>ŬWųĶà .Ě <sup>Žʾ</sup> ĕ<sup>ń</sup> ņ<sup>ŉ</sup>àíïî όĲ ،î<sub></sub>ύĜíũʦ<sup>Ť</sup>

ëό  ǒ ýç̫όƮà <sub>ٔ</sub>ėʧ ɗ<sup>ǥ</sup>ĠĄ ù ïçƠʩόɓ ،ņ <sup>ŉ</sup> Ġĝ ،ýà ĕ <sup>ɚʙ</sup>ù Ō<sup>ʷ</sup>ĠźɎǒ ïà ēà Ęî̪όɼ ñ<sup>Ʌň</sup> ïí ėόˁ Ě <sup>Žʾ</sup> ĕ<sup>ń</sup> ñ<sup>Ʌň</sup>êàí  Ō<sup>ʷ</sup>Ġźǃà ïçŬVį àï ýà Ę Ō<sup>ʷ</sup>ù ðçŬόǇ ă İàĠĝ ë Ŵƒ<sup>Ʌɵ</sup> .éό Ǉà ė<sub>W</sub>ŵ ǥçόű êŪό  Ʀŕό<sup>Ʀ</sup>í çً̦Ƅʥ<sup>ɼ</sup> àï ü ٔ<sup>ě</sup>ç̭όǒ üό ّ <sup>Ű</sup> ėĿ ëό  ǒ ðŌ<sup>Ǎ</sup>Ĝ ،é Ğù òç̧ƈW ǥà ïí êç  ̭ķà ēî<sub></sub>Ŭʥ<sup>Ť</sup>ùç̩ό<sup>ȿ</sup> ù çόʘ ņ <sup>ŉ</sup>ç̪ƍ<sup>ʙ</sup>àï .éό Ǉà ĘíŪŊ .ýïà Ō<sup>Ǎ</sup>űçŬόǇ ،îόű ûîόĜ ù íï êç̪  ŤçŬόǒ ėόˁ ņ<sup>ŉ</sup>àùàŕ<sup>Œ</sup> ٔĘî<sub></sub>όĜ ïçόű ēçόʘũ˻Ŭ <sup>Ȧ˲</sup>ù Ęî<sub></sub>όĜ ïïà ēïçƠʩόɓŏόţçόŰ ėĿ Ǹěàï ŪٔŊàï ïà

àï ċό <sup>ʙ</sup>ù Ō<sup>ʷ</sup> ėĿ øç Ŭųűà ēçόʘ ėŀŏŢ ë ŴŬ<sup>ɊɅ</sup> <sup></sup> ėόˁ ،ĕ<sup>ȯƄ</sup><sup>įs</sup>ï íũȨɌ<sup>ǒ</sup>Ġźǃà ïũɌ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ù êç  Ŭȶ çŬόǁ Ě<sup>ō</sup>Ġĝ ïũɌ<sup>Ğ</sup>ùŌ<sup>ʷ</sup> ،ýà ĕ<sup>ɂ</sup>ç<sub></sub>ŬǇïçόī êàïùí î  Ŭ<sub></sub>įçόűà ïà .íç̫ œ ç<sub></sub>Ŭį àï îόĶà ņٓ <sup>ŉ</sup>ïí ėʁ<sup>ň</sup>à ĕٓ <sup>ń</sup>ç̪ Ť ēç<sub></sub>Ŭį Ō<sup>ʷ</sup> ï êç  ̭ķà ë ŴŬ<sup>Ɋ</sup> ēçόʘ ņ <sup>ŉ</sup>ç̪ƍ<sup>ʙ</sup>àï .Ě <sup>ō</sup>ç̪ Ť ĕ<sup>ń</sup> ņ<sup>ŉ</sup>àíïî όĲ ،î<sub></sub>ύĜíàí ïàŕ<sup>Œ</sup>ĠźɎǒ ë ĸà ïí àĠĝ ù î<sub></sub>ŬŶŰ ùŕ<sup>Œ</sup>àŌ<sup>ʷ</sup> ëό  ǒïí ñ<sup>Ʌ</sup> è ّîόƘ Ěό<sup>ʞ</sup> ù ïąύĜŌ<sup>ʷ</sup> ĕ<sup>ŧƻ</sup> ŏɴȔ ïà Ěό<sup>ʞ</sup> àï ĠźɎǒ ë ĸà ėόˁýà ė <sub></sub>ŵǇàí àï ņ <sup>ŉ</sup>ç<sub></sub>ŬXųŬǇà ņ<sup>ŉ</sup>àïçƠʩόɓ ù êç  <sub></sub>ŬόǇùí ąόĜ ĕ<sup>ʝ</sup>çˮɒķąόĶ ïà Ěό ٓ <sup>ʞ</sup> ïç̩ <sup>Ƒ</sup><sup>Ğ</sup>à ė<sub></sub>Ŀç<sub></sub>ƒɅ<sup>ƎǇ</sup>ũʠ ïũ Ż<sup>˄</sup><sup>Ĝ</sup>ù ،ĕ<sup>ȯƄ</sup><sup>Ğ</sup>ï ąύĜïà ،íŕό ٓ <sup>ƚ</sup> ēïùà ،ũŻ<sup>Ɨ</sup> êç  Ŭų˹Ŭ<sup>į</sup> ،ēïžόǲ îŬȧǲ ،ĕ<sup>ŧ</sup> <sup>ɗǥ</sup> Ě<sup>Ȃ</sup>çόī ،ēïũόʠ ñ<sup>ķ</sup>ŪŊ ،êũ  Ɍ<sup>ŹŬ</sup><sup>˄ƼĜ</sup>ù ðçόīŪόƚ ،ņ<sup>ŉ</sup>àŪŊ ôïà ï ïùŔʉœ ïà .î<sub></sub>ŬŶŰ çόű é ĞçόĲï èç <sup></sup>̨ʂ<sup>Ƭ</sup> ù çόʘ ēïçƠʩόɓ ،çόʘũ˻Ŭ <sup>Ȧ˲</sup>ŏόţçόŰ ėĿ ĕ<sup>ŧƻȵ</sup> ë Ŵ̭<sup>ǥ</sup>Ġźǃà ù ĕ<sup>ɑ</sup> <sup>ň</sup>çόŰ íęĎŬǒ ،ʺŬŶŰïçόĲ ĕ<sup>ɂ</sup>Ūόƚ ،ņ<sup>ŉ</sup>ç̩ <sup>ň</sup>ç̫ɸ ēĠĪʾ،Ǹěçόʘ ēï ،üό<sup>Ķ</sup>çόī .ýïà Ō<sup>Ǎ</sup>űçŬόǇïç̪ Ƌ ņ <sup>ŉ</sup> .Ě <sup>Žʾ</sup> ĕ<sup>ń</sup> ņ<sup>ŉ</sup>àíïî όĲ ė<sub></sub>Ŀç̪ ƍ<sup>ʥɆ</sup> ،ĕ<sup>ń</sup>Ō<sup>Ǎ</sup>ưí ù ĕ<sup>żǇ</sup>ùí ąόĜ çًόĲĠŐ ėǾ ù Ě <sup>Ž</sup> <sup>ȦƄɊ</sup> ēïçƠʩόɓ Ɓ<sup>Ŋ</sup>ŏόţ ïà ėǾ ،î<sub></sub>όĜà ė<sub></sub>ŵǇàí ĕ<sup>ŧʟʆ</sup> Ō<sup>ʷ</sup>à ë ĸà ïí ėόˁ ņ<sup>ŉ</sup>ç̭όʾ ĕ<sup>ń</sup>ç̪ Ť ïà

Dedicated to my parents,

Gholamhossein Mozaffari

and Pouran Shaban Ashini

for their endless love and sacrifice.

،ýïíąόĶ ù ïîόĜ ėĿ Ě î̦ Ĺ

![](images/8c519d70eee7c2e045a1312ebbb4af0c2e881bd4b154d1a271a962d7c6c9f66b.jpg)

## Contents

Acknowledgements iii   
Introduction 1   
1.1 LLM Life-cycle Stages 1   
1.2 Compression Techniques 2   
1.2.1 Sparsity 2   
1.2.2 Quantization 3   
1.2.3 Low-rank Approximations 3   
1.3 The Failure of Isolated Compression 3   
1.4 Thesis Contributions and Roadmap . 4   
2 Background 7   
2.1 The Transformer Bottleneck: Linear Layers 7   
2.2 Hardware Constraints and the Roofline Model 9   
2.2.1 Memory Hierarchy and Tensor Cores 9   
2.2.2 The Roofline Model 10   
2.2.3 Sparse Tensor Core Acceleration . 10   
2.3 The Solution Space: Two Frontiers of Acceleration 11   
2.4 Strategy 1: Algorithmic Eficiency 12   
2.4.1 First-Order Methods 12   
2.4.2 Second-Order Methods 12   
2.4.3 The Computational Barrier and Structured Approximations . 13   
2.5 Strategy 2: Hardware Eficiency and LLM Regimes 13   
2.5.1 The Compute-Bound Regimes: Training and Prefill 14   
2.5.2 The Memory-Bound Regime: Inference Decoding 14   
2.6 The Compression Trinity as a Solution Framework 15   
2.7 The Case for a Joint Approach 16   
3 MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates 17   
3.1 Introduction 17   
3.2 Background 19   
3.3 Methodology 20   
3.3.1 The MKOR Algorithm 20   
3.3.2 Hybrid MKOR 22   
3.3.3 MKOR Convergence and Stability 23   
3.4 Experimental Results 24   
3.5 Conclusion 31   
4 SLOPE: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs 32   
4.1 Introduction 32   
4.2 Additional Related Work 34   
4.3 Sparse Plus Low-rank Pretraining of LLMs 35   
4.3.1 Double-pruned Backward Pass 35   
4.3.2 Lazy Low-rank Adapters 36   
4.3.3 Sparse Kernels 37   
4.3.4 SLOPE Runtime Optimization 38   
4.4 Experimental Results 39   
4.4.1 End-to-end Speedup and Memory Saving: Pretraining and Inference 39   
4.4.2 Pretraining Accuracy Results 41   
4.4.3 Ablation Studies 44   
4.5 Conclusion 46   
5 OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction 48   
5.1 Introduction 48   
5.2 Additional Related Work 50   
5.3 Preliminaries 51   
5.4 OPTIMA: Optimal Weight Updates via Quadratic Programming 52   
5.4.1 Reformulation as a Quadratic Program with Linear Constraints 52   
5.4.2 Reformulation as an Unconstrained Quadratic Program 53   
5.4.3 Solving the Quadratic Programs 54   
5.4.4 Eficient Implementation 54   
5.5 Experiments 55   
5.6 Conclusion and Limitations . 59   
6 PATCH: Learnable Tile-Level Hybrid Sparsity for LLMs 67   
6.1 Statement of Contributions 67   
6.2 Introduction 67   
6.3 Additional Related Work 68   
6.3.1 Pruning methods 68   
6.3.2 Complementary compression techniques 69   
6.4 Preliminaries 70   
6.5 PATCH 71   
6.6 Eficient deployment of PATCH 73   
6.7 Experiments 73   
6.7.1 Model Quality Results 74   
6.7.2 Understanding the components of PATCH . 75   
6.7.3 Speedup and memory savings 77   
6.8 Conclusion and Limitations . 78   
7 SLIM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight   
Compression 81   
7.1 Introduction 81   
7.2 Related work 82   
7.2.1 Pruning 82   
7.2.2 Quantization 83   
7.2.3 Low-rank Adapters 83   
7.2.4 Sparse Plus Low-Rank Matrix Decomposition . 83   
7.3 Preliminaries 84   
7.4 Quantized sparse plus low-rank approximation of LLMs 85   
7.4.1 SLIM-Quant quantization method 86   
7.4.2 SLIM-LoRA low-rank adapters 87   
7.4.3 Low-rank adapter quantization . 89   
7.4.4 Optional Post-Compression Fine-Tuning 89   
7.5 Experimental results 90   
7.6 Conclusion 97   
8 Conclusion and Future Work 99   
8.1 Summary of Contributions 99   
8.1.1 The Trinity in Training Dynamics 99   
8.1.2 The Trinity in Post-Training and Inference 99   
8.2 Exploratory Frameworks and Open Research 100   
8.2.1 BEAM: Blockwise Error Minimization for One-shot Compression of LLMs 100   
8.2.2 LEAP: Learnable End-to-End Adaptive Pruning of LLMs 101   
8.2.3 SLICE: Selecting Layer-wise Configurations for Matryoshka-Style LLMs 101   
8.3 Limitations of the Current Approach 101   
8.4 Future Research Directions 102   
8.5 Closing Remarks 102   
A Supplementary Material for MKOR 115   
A.1 GLUE Results 115   
A.2 Derivation of NGD Approximations 115   
A.3 Numerical Instability of Second-order Methods 117   
A.4 Sensitivity to Learning Rate 118   
A.5 Scalability . 118   
A.6 Decaying Eigenvalues and Rank-1 Approximations 118   
A.7 Training Accuracy Experiments 119   
A.8 Knee-Point Learning Rate Scheduler 119   
A.9 Proofs 120   
B Supplementary Material for SLOPE 126   
B.1 Comparison with Dynamic Sparsity: SR-STE 126   
B.2 cuSPARSELt Initialization Overhead: Static vs. Dynamic Sparsity 127   
B.3 BERT-Large-Uncased: Pretraining and Downstream Evaluation . 127   
B.4 Performance overhead of bidirectional mask 128   
B.5 Sparsity ratio analysis of double-pruned backward pass 129   
B.6 Sensitivity to the choice of pruning matrix . 129   
B.7 Implementation details 130   
B.8 Task-specific GLUE results . 132   
B.9 Integration with Flash Attention 132   
B.10 Comparison with dense models . 133   
B.11 Zero-shot GLUE results for GPT 133   
B.12 Extended SR-STE and FST implementation details 134   
B.13 Comparison of Depth and Width Pruning 135   
B.14 Proofs 136   
C Supplementary Material for OPTIMA 140   
C.1 Calibration dataset size sensitivity 140   
D Supplementary Material for PATCH 142   
D.1 STOICC Integration . 142   
D.2 Per Task Results . 144   
D.3 Tile Transfer Learning 147   
E Supplementary Material for SLIM 148   
E.1 Notations 148   
E.2 Input Quantization 148   
E.3 Additional fine-tuning results 149   
E.4 Language modeling experiments 149   
E.5 Additional Sparse and Quantized Results . 153   
E.6 Sparsity vs. quantization 154   
E.7 Additional speedup results 155   
E.8 Fine-tuning costs 156   
E.9 Memory reduction analysis 157   
E.10 Computation reduction analysis . 158   
E.11 Compression costs 158   
E.12 Rank analysis 159   
E.13 Efects of calibration sample count 159   
E.14 Sensitivity to calibration dataset 159   
E.15 Sparsity analysis 160   
E.16 Group quantization challenges 161

## List of Tables

1.1 The Sparsity Paradox: Comparison ofLLaMA-2-7B zero-shot accuracy under standard Post-  
Training Compression. Sparsity yields lower compression ratios (2x) yet results in signifi  
cantly higher accuracy loss compared to Quantization (4x), highlighting its destructive nature. 3   
1.2 Average Zero-shot Accuracy of LLaMA-2-7B on 8 tasks (MMLU, PIQA, ARC-Easy, ARC-  
Challenge, WinoGrande, OpenBookQA, RACE, HellaSwag) at 8x compression ratio. Single  
pillar methods fail to retain capabilities, while multi-pillar methods (Compression Trinity)   
recover accuracy 4   
3.1 The computation and communication complexity and memory overhead of the state-of-the-art   
implementations of the first- and second-order (second-order optimizers are written in bold).   
The division by 2 in MKOR is because MKOR uses half-precision computations. The com  
plexity of KFAC-based methods depends on layer dimensions while SNGD methods mostly   
depend on the batch size. In transformers, due to the scaling of the batch size by the sequence   
length, batch sizes and layer dimensions are comparable, making both KFAC- and SNGD  
based methods more expensive than SGD. 22   
3.2 List properties of the models, datasets, and settings used in our experiments 25   
3.3 BERT-Large Uncased results on SQuAD v1.1 question answering task 26   
3.4 BERT-Large Uncased results on the GLUE classification tasks. We report the average of the   
metrics of diferent GLUE tasks (accuracy, F1 score, etc) for easier comparison. 26   
3.5 Per-GPU memory usage (in GB) for MKOR, KFAC/KAISA, LAMB, and SGD on BERT-  
Large-Uncased pre-training and ResNet-50 training on ImageNet. 29   
4.1 Comparative analysis of end-to-end pretraining and inference speedup (×) comparison be  
tween SLOPE and the latest work (FST) on accelerating pretraining with 2:4 sparsity (ICML   
2024) [62]. The baseline is dense PyTorch implementation of the models with CUBLAS   
backend. Note that the lack of inference speedup in FST is because of the final dense pretrain  
ing during the final iterations, resulting in a dense model for inference. E-SR-STE stands for   
Extended SR-STE. 40   
4.2 Comparative analysis of end-to-end memory reductions (×) during training and inference   
between SLOPE and the latest work (FST) on accelerating pretraining with 2:4 sparsity (ICML   
2024) [62]. Values greater than 1.00× show memory overhead. 41   
4.3 GPT2-Small accuracy results on zero-shot tasks. Adapter rank is the ratio of the low-rank   
adapter to the hidden dimension of the model. For Extended SR-STE, we have used a de  
cay factor of $6 \times 1 0 ^ { - 6 } ,$ , since it resulted in the lowest perplexity in OpenWebText. The best   
performing sparse configuration is highlighted in bold. 42   
4.4 SQuAD-v1.1 accuracy and GLUE results on BERT-Large-Uncased with diferent adapter ranks 43   
4.5 SQuAD-v1.1 accuracy results on BERT-Large-Uncased for diferent sparsity settings. 44   
4.6 End-to-end speedup (×) before and after eficient implementation of low-rank adapters. . 45   
4.7 End-to-end speedup (×) before and after splitting the upsample matrix. In both cases, the   
optimization discussed in Table 4.6 is used. 46   
4.8 SQuADv1.1 results on BERT-Large-Uncased for diferent pruned modules. 46   
5.1 Key hyperparameters used in OPTIMA. 56   
5.2 Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 50% unstruc  
tured sparsity. OPTIMA consistently improves the accuracy of the models across diferent   
tasks. 60   
5.3 Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 2:4 sparsity.   
In this experiment, only the layers in the MLP part of the transformer are pruned, and the self  
attention layers are dense, resulting in an end-to-end sparsity ratio of 38% to 41%. OPTIMA   
consistently improves the accuracy of the models across diferent tasks. Please note that Prox-  
Sparse pruning is limited to 2:4 sparsity, and hence our unstructured sparsity experiments do   
not include it. . 61   
5.4 Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 60% unstruc  
tured sparsity. OPTIMA consistently improves the accuracy of the models across diferent   
tasks. 62   
5.5 Qwen-2.5 family perplexity on WikiText2 and accuracy on zero-shot downstream tasks for   
50% unstructured sparsity. OPTIMA consistently improves the accuracy of the models across   
diferent tasks. 63   
5.6 Qwen-2.5 perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 60% un  
structured sparsity. OPTIMA consistently improves the accuracy of the models across difer  
ent tasks. 64   
5.7 Qwen-2.5 perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 2:4 spar  
sity. In this experiment, only the layers in the MLP part of the transformer are pruned, and   
the self-attention layers are dense, resulting in an end-to-end sparsity ratio of 38% to 41%.   
OPTIMA consistently improves the accuracy of the models across diferent tasks. Please note   
that ProxSparse pruning is limited to 2:4 sparsity, and hence our unstructured sparsity experi  
ments do not include it. . 65   
5.8 Comparison of OPTIMA with other optimizers without convergence guarantees (ADAM).   
ADAM can lead to suboptimal solutions (Gemma 3 1B) or divergence of the model (OPT   
125M). . 66   
6.2 Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2   
dataset) for diferent pruning methods. By jointly optimizing the location of dense tiles and   
the sparsity pattern within the sparse tiles, PATCH<sup>Joint</sup> allows for a continuous sparsity ratio   
for the models, providing a flexible tradeof between sparsity and model quality. 74   
6.1 Hyper-parameters used for PATCH<sup>Joint</sup> and PATCH<sup>Tile</sup> across sparsity ratios. All hyper param  
eters were tuned on Qwen-2.5-0.5B. 74   
6.3 Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2   
dataset) for diferent pruning methods. By only optimizing the location of dense tiles while   
keeping sparsity pattern within the sparse tiles frozen, PATCH<sup>Tile</sup> provides a memory eficient   
variant for PATCH<sup>Joint</sup>, allowing for a continuous sparsity ratio for the models and providing   
a flexible tradeof between sparsity and model quality. . 75   
6.4 Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2   
dataset) for PATCH, Wanda, and SparseGPT. For models with less than or equal to 1B param  
eters, PATCH<sup>Joint</sup> optimizes both dense tile locations and sparsity patterns, while for larger   
models PATCH<sup>Tile</sup> optimizes only dense tile locations with frozen sparsity patterns, both us  
ing Dense/2:4 Tiles pattern allowing continuous sparsity ratios and flexible tradeofs between   
sparsity and model quality. Wanda and SparseGPT are unstructured pruning methods. . . 76   
6.5 Impact of PATCH’s tile size across sparsity levels (↓ is better). The efect of tile size on model   
quality is not significant, showing PATCH’s robustness against tile size. 76   
6.6 Global sparsity yields better quality by concentrating pruning in less important blocks and   
preserving density elsewhere (↓ is better). 76   
6.7 Impact of fixed 2:4 mask selection for PATCH<sup>Tile</sup>, compared with joint optimization (↓ is   
better). PATCH<sup>Joint</sup> achieves the lowest perplexity overall, while for PATCH<sup>Tile</sup>, MaskLLM   
provides the best frozen mask. 77   
7.1 Average zero-shot accuracy of LLaMA-2 and OPT models with 50% sparsity and 4-bit   
weight quantization. Best Method∗ indicates the best quantization method out of Group   
AbsMax, AWQ, OmniQuant, and AfineQuant. ↑ indicates better performance. 91   
7.2 Efects of fine-tuning on the average zero-shot accuracy of LLaMA-2 models with 50% spar  
sity and 4-bit weight quantization. ↑ indicates better performance. 92   
7.3 Accuracy results of OPTIMA weight update mechanism with SLIM-LoRA. ↑ indicates better   
performance. 92   
7.4 Average accuracy (↑ indicates better) across eight zero-shot downstream tasks (including   
RACE [74] and HellaSwag [161]) and WikiText2 perplexity (↓ indicates better) of compressed   
models with 4-bit weight-only quantization. Please note that using LoRA adds additional   
parameters to the model. 93   
7.5 Average zero-shot accuracy of LLaMA-2 and OPT models with pruning. The quantization is   
disabled in this experiment. ↑ indicates better performance. 95   
7.6 Average zero-shot accuracy of LLaMA-2 and OPT models with quantization. The sparsity is   
disabled in this experiment. ↑ indicates better performance. 96   
7.7 LLaMA-2 family of models speedup (×) using SLIM compared to original dense unquantized   
model on NVIDIA RTX-3060. ↑ shows higher speedup. 97   
A.1 BERT-Large-Uncased Results on the GLUE classification tasks. 115   
A.2 Number of epochs necessary for convergence in diferent optimizers for ResNet-50 on CI-  
FAR10. MKOR is the least sensitive optimizer to learning rate, converging in almost the   
same number of iterations for a wide range of learning rate, while other optimizers either   
diverge (D) or converge to a local-minimum (∗ superscript). 118   
B.1 End-to-end slow-down of Bi-directional Mask [166] in comparison to the dense baseline. . . 128   
B.2 GLUE results for each task in the experiments discussed in Section 4.4. 132   
B.3 Speedup of SLOPE and FlashAttention-2 (FA2) on OPT models. 132   
B.4 Performance comparison across diferent GPT models, sparsity methods, and LoRA ranks on   
various tasks. E-SR-STE stands for Extended SR-STE. 133   
B.5 Performance comparison of GPT models using diferent sparsity methods and LoRA ranks on   
GLUE tasks. E-SR-STE stands for Extended SR-STE. 133   
B.6 Description of Key Terms . 134   
B.7 Model Configurations for LLaMA-2 7B 136   
B.8 Model Configurations for Gemma-9B 137   
B.9 Model Configurations for Gemma-2B 138   
D.1 Throughput of LLaMA-2 7B with mixed sparsity compared to the dense model. Measure  
ments taken on an A6000 GPU with batch size 16. Throughput is reported in tokens pro  
cessed/sec 144   
D.2 Throughput of LLaMA-2 7B with mixed sparsity compared to the dense model. Measure   
ments taken on an A100 GPU with batch size 16. Throughput is reported in tokens processed/sec.144   
D.3 Model quality (task accuracy across eight zero-shot tasks, reported in %) for Qwen-2.5 0.5B   
with diferent pruning methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity pat  
terns, enabling a flexible sparsity-quality tradeof. 145   
D.4 Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-2 7B   
with diferent pruning methods. PATCH<sup>Tile</sup> optimizes tile-based sparsity, enabling a flexible   
sparsity-quality tradeof. 145   
D.5 Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-3.1 8B   
with diferent pruning methods. PATCH<sup>Tile</sup> optimizes tile-based sparsity, enabling a flexible   
sparsity-quality tradeof. 146   
D.6 Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-3.2   
1B with diferent pruning methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity   
patterns, enabling a flexible sparsity-quality tradeof. 146   
D.7 Model quality (accuracy across eight zero-shot tasks) for Gemma-3 1B with diferent pruning   
methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity patterns, enabling a flexible   
sparsity-quality tradeof. 147   
D.8 Perplexity (↓) under diferent tile prior initializations. All priors yield nearly identical per  
formance, suggesting that the global sparsity target allows dynamic reallocation of sparsity   
during training, overriding the influence of fixed initialization. 147   
E.1 Key notation definitions used in the experimental results (Section 7.5). 148   
E.2 Average zero-shot accuracy of LLaMA-2 and OPT models with 4-bit weight and 8-bit input   
quantization. ↑ indicates better performance. 149   
E.3 Efects of fine-tuning on the average zero-shot accuracy of LLaMA-2 and OPT models with   
50% sparsity and 4-bit weight quantization. ↑ indicates better performance. 150   
E.4 Perplexity of LLaMA-2 and OPT models with 2:4 sparsity and 4-bit weight quantization   
on WikiText-2 dataset language modeling task. ↓ indicates better performance. 150   
E.5 Perplexity of LLaMA-2 and OPT models with 4-bit weight and 8-bit input quantization. ↓   
indicates better performance. 151   
E.6 Perplexity of LLaMA-2 and OPT models with unstructured sparsity and 4-bit weight quan  
tization on WikiText-2 dataset language modeling task. ↓ indicates better performance. . . 151   
E.7 Perplexity of LLaMA-2 and OPT models with pruning on WikiText-2 dataset language mod  
eling task. The quantization is disabled in this experiment. ↓ indicates better performance. . 152   
E.8 Perplexity of LLaMA-2 and OPT models with quantization on WikiText-2 dataset language   
modeling task. The sparsity is disabled in this experiment. ↓ indicates better performance. . 153   
E.9 Average zero-shot accuracy of LLaMA-2 and OPT models with 2:4 sparsity and 4-bit weight   
quantization. ↑ indicates better performance. 154   
E.10 Average zero-shot accuracy of diferent models using diferent pruning and quantization schemes.   
↑ indicates better performance. Combining sparsity and quantization provides better accuracy   
results in comparison to solely using quantization. 154   
E.11 Perplexity of diferent models on WikiText-2 dataset using diferent pruning and quantization   
schemes. ↓ indicates better performance. Combining sparsity and quantization provides better   
accuracy results in comparison to solely using quantization. 155   
E.12 The required time for fine-tuning the models with a single H100 GPU on 300,000 tokens from   
the C4 dataset with a batch size of 64 and a sequence length of 1024. 157   
E.13 Theoretical memory reduction (×) of diferent compression methods across various OPT and   
LLaMA models. In Quantized SLIM , the low-rank adapters are also quantized.(↓ indicates   
better performance.) . 157   
E.14 Compute (FLOP) reduction ratios (×) of diferent compression methods across various OPT   
and LLaMA models. In Quantized SLIM , the low-rank adapters are also quantized. (↑ indi   
cates better performance.) . 158   
E.15 The required compression time for diferent models and compression methods using a single   
H100 GPU. 159   
E.16 Perplexity of diferent models on WikiText-2 dataset using SLIM-LoRA with 4-bit quantiza  
tion using SLIM-Quant with diferent calibration datasets. ↓ indicates better performance. . . 160   
E.17 Group quantization slow-down (×) on diferent LLaMA-2 and LLaMA-3.1 models. ↓ indi  
cates worse. 161

## List of Figures

2.1 The compute graph of a standard Transformer block, highlighting the Self-Attention and Feed-  
Forward Network (FFN) sub-layers. 8   
2.2 Computational time breakdown for LLaMA-3.1-8B during training and inference, as profiled   
on a single NVIDIA H100 GPU. The ”linear” component is the largest single bottleneck. The   
batch size, input sequence length, and generation sequence length are set to 4, 1024, and 1024   
respectively. . 9   
2.3 Roofline Model comparison between NVIDIA A100 and H100 GPUs. The H100 (orange) of   
fers significantly higher peak compute (FLOPs). However, memory bandwidth has not scaled   
proportionally, shifting the ”knee” of the curve to the right. This implies that algorithms on   
H100 require a higher arithmetic intensity to escape the memory-bound region compared to   
the A100. 11   
3.1 MKOR for layer m on a single worker. The inputs of MKOR are the activations $A _ { t } ^ { m }$ , the   
gradients of the loss function with respect to the inputs $G _ { t } ^ { m }$ , and the gradients of the loss   
function with respect to the weights ∇ mL. The output is the update values $\Delta W ^ { m }$ 20   
3.2 The pre-training loss of BERT-Large-Uncased using diferent optimizers. 26   
3.3 Test accuracy of ResNet-50 on ImageNet for MKOR, KAISA, and SGD on 64 GPUs. 27   
3.4 The sensitivity of MKOR and KAISA for BERT-Large-Uncased and an Autoencoder model   
(a) and the efect of inversion frequency on the convergence properties of these models (b). . 28   
3.5 Per-step breakdown of diferent optimizers on BERT-Large-Uncased (a) and ResNet-50 (b).   
The times reported in these graphs reflect only the optimizer computations. The majority of   
the training time is spent on the model’s forward and backward passes, which are identical   
across all optimizers and are not included here. 29   
3.6 Rank-1 error for activation and input gradient covariance matrices for BERT-Large-Uncased   
pre-training (a, b) and ResNet-50 on ImageNet (c, d). 30   
4.1 The sparse training pipeline in SLOPE. Here, X, Y, and W denote the input, output, and   
the weight tensors for a specific layer, respectively. ∇ L represents the gradient of the loss   
function. L and R are the low-rank terms that are introduced only in the final 1% iterations.   
Superscript R shows row-wise pruning using N:M scheme and R, C shows both column   
and row-wise N:M sparsification, leading to extra imposed zeros. Blue elements represent   
non-zero values, while white elements represent pruned values, and red elements indicate   
additional zeros introduced during the backward pass. 34   
4.2 Validation perplexity of GPT2-Small and GPT2-Large on OpenWebText. $\gamma _ { w }$ shows the value   
of the decay factor parameter in Extended SR-STE (FST). 42   
4.3 (a) The speedup achieved using cuSPARSELt backend in PyTorch for Attention $( d _ { o u t } = d _ { i n } )$   
Upsample $( d _ { o u t } = 4 d _ { i n } )$ and Downsample $\begin{array} { r } { ( d _ { o u t } = \frac { d _ { i n } } { 4 } ) } \end{array}$ matrices with a batch size of 2048.   
(b) The cosine similarity of the low-rank adapters and the converged adapters for diferent   
layers in the model. The cosine similarities are averaged among the 24 layers of BERT-Large   
Uncased. 43   
4.4 The speedup achieved by low-rank adapters in comparison to a dense matrix-multiplication. 45   
5.1 OPTIMA generates a shared Hessian among the diferent columns of the pruned weight using   
a small calibration dataset. Then, the weights in diferent columns will be updated in parallel   
using a QP solver and the shared Hessian. 50   
5.2 Relative error reduction on OPTIMA in comparison to Wanda, SparseGPT, and Thanos for   
LLaMA-3.2 1B. 56   
6.1 Illustration of the PATCH learning process for generating tile-level hybrid masks. Each tile is   
parameterized by a learnable distribution and sampled with Gumbel Softmax to produce $\tilde { M } _ { \mathrm { t i l e } }$   
The dense probability is expanded and merged with a 2:4 mask $\tilde { M } _ { 2 : 4 }$ , which can be fixed or   
jointly learned during training, yielding $\tilde { M } .$ . The final mask assigns each tile to remain dense   
or follow the 2:4 pattern, enabling flexible sparsity across the weight matrix. 69   
6.2 Layer-wise sparsity allocation under diferent global sparsity budgets for various models. PATCH   
achieves the target global sparsity while flexibly distributing pruning across transformer layers. 77   
6.3 Sparsity distribution across Attention and MLP layers under varying global sparsity budgets   
in Qwen-2.5 0.5B. 78   
6.4 Sparsity distribution across Attention and MLP layers under varying global sparsity budgets   
in Gemma-3 1B. . 79   
6.5 Sparsity distribution across Attention and MLP layers under varying global sparsity budgets   
in LLaMA-3.2 1B. 80   
7.1 The SLIM weight compression pipeline consists of three main steps: (1) Quantizing weights   
using the symmetric SLIM-Quant algorithm, producing quantized weights $\mathcal { W } ^ { \mathcal { Q } }$ and quantiza  
tion error $E _ { Q } { \mathrm { : } }$ ; (2) Sparsifying quantized weights $\mathcal { W } ^ { \mathcal { Q } }$ through a pruning method, resulting   
in compressed weights $\mathcal { W } ^ { \mathcal { C } }$ and sparsity error $E _ { S } ;$ (3) Mitigating compression errors through   
SLIM saliency-based low-rank approximation, generating left and right low-rank adapters L   
and R. Optionally, these adapters can be fine-tuned with sparse quantized weights frozen to   
further enhance model accuracy. 85   
7.2 Accuracy results of the OPT family across diferent compression methods (↑ indicates bet   
ter performance). At equal parameter size, SLIM outperforms both dense models and other   
compression techniques, demonstrating that model compression with SLIM yields superior   
performance under the same budget. 94   
A.1 Approximations in second-order methods. 116   
A.2 Maximum and minimum eigenvalues (a) and the condition number (b) of the right factors in   
KFAC when training ResNet-50 on CIFAR-10. As illustrated, the minimum eigenvalues of   
the factors in KFAC approach zero, meaning that the factors are singular, and hence have large   
condition numbers, making numerical inversion of them complex and numerically unstable. . 122   
A.3 Scalability of MKOR. 123   
A.4 Average covariance rank-1 approximation error for ResNet-50 in diferent iterations 123   
A.5 Training time for distributed first- and second-order optimizers SGD, MKOR, KAISA, and   
HyLo on BERT-Large-Cased on IMDB (a), BERT-Base-Cased on SQuAD (b), and AlexNet on   
CIFAR-100 (c). In all the experiments, MKOR outperforms other optimizers in convergence   
speed. 124   
A.6 Training accuracy vs. the number of epochs for distributed first- and second-order optimizers   
SGD, MKOR, KAISA, and HyLo on BERT-Large-Cased on IMDB (a), BERT-Base-Cased on   
SQuAD (b), and AlexNet on CIFAR-100 (c). In all the experiments, MKOR outperforms other   
optimizers in convergence rate. 125   
B.1 Average mask diference between each iteration and the converged sparsity pattern in GPT2-   
Small pretraining using SR-STE. The highlighted area shows the ratio of the resources used   
for updating weights that are pruned and not used in the inference of the model. 127   
B.2 The setup and multiplication time for square matrices using the cuSPARSELt SpMM backend. 128   
B.3 Training loss of BERT-Large-Uncased on WikiCorpus dataset for phase 1 and 2. . 129   
B.4 The imposed sparsity ratio when pruning the weight matrices in the backward pass. 130   
B.5 Validation perplexity on GPT2-Small pretraining for 100,000 iterations for diferent matrix   
pruning settings. Pruning the output gradients leads to divergence within a few iterations and   
hence is not reported. . 131   
B.6 Comparison of the loss of depth and width pruning methods. 139   
C.1 Sensitivity analysis for the number of calibration samples for diferent pruning methods. . 141   
E.1 SLIM speedup for LLaMA-2 family of models on NVIDIA A100-40GB GPUs. 156   
E.2 Sensitivity analysis for the rank of the adapter (a) and the number of calibration samples (b)   
for diferent one-shot compression methods. For Naive-LoRA and SLIM-LoRA, we have   
used the SLIM-Quant quantization method, and for the SparseGPT, we have used the Group   
quantization version of OPTQ. 160   
E.3 Sparsity analysis on LLaMA-2-13B model using perplexity on WikiText-2 dataset. ↓ indicates   
better performance. 161

## Chapter 1

## Introduction

Large Language Models (LLMs) have become foundational tools in modern artificial intelligence, demonstrating remarkable capabilities in text generation, reasoning, and multi-modal tasks [14, 31, 139]. However, these capabilities come at a significant cost. Training state-of-the-art models consumes enormous computational, memory, and environmental resources [115, 14], and deploying them for inference remains a major challenge due to their massive memory footprint and high computational demands.

To mitigate these overheads, various model compression techniques have been proposed [58]. Historically, these methods have often been applied in isolation, focusing on a single tool such as sparsity (removing parameters) [36], quantization (reducing parameter precision) [37], or low-rank approximations (factoring parameter matrices) [61]. This isolated approach is inherently limiting, as it fails to address the multi-faceted nature of the eficiency bottleneck.

This thesis argues that these methods must be applied jointly. We introduce a conceptual framework, which we term the Compression Trinity, built upon the three fundamental pillars of sparsity, quantization, and low-rank approximations. These methods are highly complementary. While all three pillars contribute to reducing overall memory and compute overheads, they play distinct roles in balancing hardware constraints with model expressivity. Specifically, sparsity and quantization directly target the primary bottlenecks of modern hardware: sparsity reduces the computational load, while quantization minimizes memory band width requirements. Low-rank approximations, in turn, act as the critical algorithmic counterweight. By eficiently projecting parameters into lower-dimensional spaces, they restore lost accuracy and model capacity without reintroducing the hardware overheads that sparsity and quantization eliminated. We demonstrate that the joint application of this Compression Trinity across the diferent phases of an LLM’s life-cycle is the key to unlocking new frontiers of eficiency.

To understand the context for these contributions, we must first consider the distinct stages of an LLM’s development and deployment.

## 1.1 LLM Life-cycle Stages

The life-cycle of a large language model can be broadly divided into two primary stages<sup>1</sup>:

• Pretraining: This is the most computationally expensive phase, where the model is trained from scratch on massive, web-scale datasets to learn general-purpose language representations [119, 30].

• Inference: This is the deployment stage, where the trained model is used to generate predictions for new user inputs. In many real-world applications, inference must be performed with low latency and on resource-constrained hardware.

Each stage presents unique opportunities for acceleration. During the pretraining phase, this can be achieved by either accelerating the training process itself or by improving the optimizer to converge faster. For the inference stage, post-training compression can be applied to make the deployed model more eficient. These opportunities, though diferent on the surface, share a common computational bottleneck, dense matrix matrix multiplications, and can therefore share the same fundamental principles for compression, which we will cover next.

## 1.2 Compression Techniques

The core of our approach relies on the three main techniques of the Compression Trinity: sparsity, quantization, and low-rank approximation. Let us now define the three pillars of the Compression Trinity.

## 1.2.1 Sparsity

Sparsity is a compression technique that involves identifying and removing (i.e., setting to zero) the least important weights in a neural network, thereby reducing the total number of parameters and floating-point operations (FLOPs) [75, 55].

This technique is broadly categorized into two types. Unstructured sparsity removes arbitrary, individual weights from a matrix. While it ofers high flexibility and can often achieve high compression ratios with minimal accuracy loss, it is notoriously dificult to accelerate in practice. Its irregular memory access patterns are not hardware-friendly, meaning they cannot be executed eficiently on modern parallel accelerators like GPUs, which are designed to process data in large, regular chunks [151]. Conversely, structured sparsity removes entire blocks of parameters, such as full rows, columns, or filter channels [67, 84]. This regular structure is hardware-friendly but often damages model accuracy significantly, as it removes entire features indiscriminately.

A new family of semi-structured sparsity patterns has emerged to bridge this gap. A prominent example is N:M sparsity, which enforces that N out of every M consecutive weights are non-zero (e.g., 2:4 sparsity) [137, 62]. This pattern is flexible enough to preserve accuracy while being regular enough for hardware acceleration on modern GPUs [108].

However, finding the optimal sparsity pattern and updating the remaining non-zero weights to compensate for the removed elements remains a challenging task [35, 36]. To illustrate the severity of this challenge, we can compare sparsity against quantization. Table 1.1 demonstrates that even at a modest 2x compression ratio (50% sparsity), removing connections damages the model more than aggressive 4-bit quantization, which ofers 4x compression. This counter-intuitive result, that a method providing less compression yields lower accuracy, highlights that sparsity is the most destructive pillar of the Trinity, necessitating the dedicated optimization strategies we propose in Chapter 5 and Chapter 6.

Table 1.1: The Sparsity Paradox: Comparison of LLaMA-2-7B zero-shot accuracy under standard Post-Training Compression. Sparsity yields lower compression ratios (2x) yet results in significantly higher accuracy loss compared to Quantization (4x), highlighting its destructive nature.
<table><tr><td>Method</td><td>Compression Ratio</td><td>Bit-width / Density</td><td>Avg Accuracy</td></tr><tr><td>Dense Baseline</td><td>1x</td><td>FP16 / 100%</td><td>54.61%</td></tr><tr><td>4-bit Quantization *</td><td>4x</td><td>INT4 / 100%</td><td>53.63%</td></tr><tr><td>2:4 Sparsity†</td><td>2x</td><td>FP16 / 50%</td><td>48.62%</td></tr></table>

Best among AbsMax and OPTQ  
<sup>†</sup> Best among SparseGPT, Wanda, Thanos, ProxSparse, and MaskLLM

## 1.2.2 Quantization

Quantization reduces the numerical precision of the numbers used to represent the model’s weights and, in some cases, activations. For example, parameters are typically trained in 32-bit (FP32) or 16-bit (FP16/BF16) floating-point formats, but quantization can compress them down to 8-bit integers (INT8), 4-bit integers (INT4), or even lower bit-widths [27, 37].

This reduction in precision has two primary benefits: it saves memory (e.g., 4-bit quantization yields an 8x memory reduction over 32-bit weights) and allows for the use of faster, specialized compute units (like INT8 tensor cores) that can perform integer arithmetic much faster than floating-point operations.

The main challenge in quantization is that the representation capabilities of the numbers are reduced exponentially with the bit-width. This can lead to large accuracy degradation in low bit-width schemes, especially in the presence of outlier values, which are common in LLMs [81]. Finding the best way to map high-precision weights to a low-bit representation and updating those weights to minimize the resulting error is a non-trivial task [37]. As with sparsity, only relying on quantization limits the total compression ratio of the models, and other methods should be combined with it to push eficiency further.

## 1.2.3 Low-rank Approximations

Low-rank approximations are based on the observation that the large weight matrices in LLMs are often overparameterized and have a low ”intrinsic rank.” This redundancy can be exploited by decomposing a large weight matrix $W \in \mathbb { R } ^ { m \times k }$ into the product of two smaller, ”thin” matrices, $\boldsymbol { L } \in \mathbb { R } ^ { m \times r }$ and $R \in \mathbb { R } ^ { r \times k }$ where the rank $r \ll m , k$ . This technique, popularized by Low-Rank Adaptation (LoRA) [61], can reduce the memory and compute overhead of LLMs significantly.

However, low-rank approximations are not very efective in compressing matrices that are inherently highrank, and applying them too aggressively can lead to significant errors. Relying on low-rank approximation alone is insuficient, as it cannot capture the fine-grained, high-rank information that sparsity or high-precision quantization can preserve.

## 1.3 The Failure of Isolated Compression

As mentioned in Section 1.1, modern deployment scenarios impose strict constraints on memory and latency. While the individual pillars of compression—sparsity and quantization—are well-studied, existing literature often treats them as independent solutions.

However, empirical evidence suggests that pushing any single pillar to the extreme results in catastrophic accuracy loss. Table 1.2 illustrates this breakdown on the LLaMA-2-7B model. When we attempt to achieve an 8x compression ratio using only quantization (2-bit) or only sparsity (87.5%), the model’s reasoning capabilities collapse, with accuracy dropping to near-random chance (≈ 31%).

Table 1.2: Average Zero-shot Accuracy of LLaMA-2-7B on 8 tasks (MMLU, PIQA, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA, RACE, HellaSwag) at 8x compression ratio. Single-pillar methods fail to retain capabilities, while multi-pillar methods (Compression Trinity) recover accuracy.
<table><tr><td>Method</td><td>Average Accuracy</td></tr><tr><td>Dense Baseline (FP16)</td><td>54.61%</td></tr><tr><td>Single-Pillar Compression (8x Ratio) 水</td><td></td></tr><tr><td>2-bit Quantization</td><td>31.81%</td></tr><tr><td>87.5% Unstructured Sparsity†</td><td>31.24%</td></tr><tr><td>Multi-Pillar Compression (8x Ratio)</td><td></td></tr><tr><td>4-bit Quantization + 2:4 Sparsity</td><td>47.97%</td></tr><tr><td>4-bit Quantization + 50% Unstructured Sparsity</td><td>52.38%</td></tr></table>

Best among AbsMax and OPTQ  
<sup>†</sup> Best among SparseGPT, Wanda, and Thanos

In contrast, the table demonstrates that a hybrid approach, combining moderate 4-bit quantization with moderate 50% sparsity, recovers the majority of the accuracy (52.38%) while achieving the same compression ratio. This observation forms the core motivation of this thesis: compression is not a singular optimization problem, but a multi-dimensional balancing act.

## 1.4 Thesis Contributions and Roadmap

As discussed in Section 1.2, each of the compression techniques, sparsity, quantization, and low-rank approximation, cannot efectively compress LLMs alone and will hit an accuracy or eficiency wall at some point. This thesis argues and demonstrates that the joint application of the Compression Trinity is the key to unlocking new frontiers of eficiency. We show that by combining hardware-friendly sparsity and aggressive quantization, we can achieve massive reductions in compute and memory. We then use low-rank approximations as a controllable, highly eficient method to add back a small number of dense parameters, compensating for the joint compression error and restoring model accuracy.

Before detailing the novel methods that prove this thesis, Chapter 2 will first provide a comprehensive technical background.

The thesis is structured to explore the Compression Trinity across the full life-cycle of an LLM. We begin by applying the Trinity to the Pretraining phase (Chapter 3 and Chapter 4). We then perform a deep dive into the Sparsity pillar, the most destructive component of the Trinity, exploring both layer-wise (Chapter 5) and end-to-end (Chapter 6) optimization regimes. Finally, we integrate all three pillars for a holistic Post-Training Compression solution (Chapter 7). These contributions are detailed as follows:

• Chapter 3 MKOR: We lay the foundation for the Compression Trinity in the expensive pretraining phase. MKOR (Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates) is a novel second-order optimizer that leverages all three pillars. It approximates the second-order information as a block-diagonal sparse matrix, and then leverages rank-1 updates (a low-rank approximation)

to compute the inverse of its covariance matrices. Crucially, this joint formulation is exceptionally stable, allowing the inverse factors to be computed in a quantized 16-bit format, whereas other methods require 32-bit numbers for numerical stability. By applying the Trinity to the optimizer’s internal computations, we reduce the complexity of second-order updates from $\mathcal { O } ( d ^ { 3 } )$ to $\mathcal { O } ( d ^ { 2 } )$ and communication from $O ( d ^ { 2 } )$ to O(d), where d is the hidden dimension of the model. As a result, MKOR accelerates pretraining by outperforming state-of-the-art optimizers like KFAC by up to 1.85x on BERT-Large training.

• Chapter 4 SLOPE: We further accelerate the pretraining phase by applying the Compression Trinity directly to the linear layers. SLOPE (A Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining method) introduces a framework that jointly applies sparsity and low-rank approximations from the start. It accelerates sparse pretraining by introducing a novel double-pruned backward pass that enables N:M sparsity acceleration in both forward and backward passes. To recover accuracy lost from sparsity, we introduce low-rank adapters only during the final 1% of pretraining iterations. This ”lazy” insertion minimizes overhead while maximizing accuracy. By creating a base model that is already sparse and low-rank, SLOPE produces a model that is not only eficient for inference (1.34x speedup) but is also an ideal and stable candidate for the final pillar, quantization, which we apply in the post-training stage. This approach accelerates end-to-end training and inference of models like OPT-66B by up to 1.14x and 1.34x, respectively, and reduces training memory by 0.77x.

• Chapter 5 OPTIMA: We transition to the post-training phase, addressing the strict scenario where end-to-end training is infeasible. OPTIMA (Optimal One-shot Pruning via Quadratic Programming) focuses on perfecting the sparsity pillar under a ”zero-training” constraint. It formulates the one-shot, post-pruning weight update as a series of independent, row-wise Quadratic Programs (QPs) that share a common layer Hessian. This allows us to find the per-row globally optimal update given the Hessian, minimizing the reconstruction error from pruning. By creating the most accurate and stable sparse model possible, OPTIMA serves as a critical enabling step, producing a high-fidelity model that can withstand the subsequent application of aggressive quantization and low-rank approximations. OPTIMA acts as a drop-in replacement for the update step in methods like Wanda or SparseGPT, consistently improving zero-shot performance by up to 3.97% absolute accuracy without any fine-tuning.

• Chapter 6 PATCH: We address the scenario where a fine-tuning budget is available to push performance further. PATCH (Pruning with a Learnable Tile-level Configuration) overcomes the limitations of the layer-wise mask detection used in OPTIMA. It introduces a hybrid sparsity framework that partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask. This creates a continuous, efective sparsity ratio between 0% and 50%, balancing accuracy in critical regions with acceleration elsewhere. While PATCH focuses on enhancing this single pillar, it is designed to be fully composable with the other pillars of the Trinity. We explicitly demonstrate that it can be combined with joint quantization and low-rank approximation methods, such as SLIM, to further boost the accuracy and stability of the final, jointly compressed model. On LLaMA-2 7B, the PATCH framework alone delivers 1.18x-1.38x end-to-end speedup while improving accuracy by up to 2.96% compared to state-of-the-art 2:4 pruning.

• Chapter 7 SLIM: We present the complete fulfillment of the Compression Trinity in a one-shot setting. SLIM (One-shot Quantization and Sparsity with Low-rank Approximation) holistically integrates all three pillars to solve the ”compounded error” problem. It applies aggressive, hardware-friendly quantization and semi-structured sparsity, then compensates for the combined error using a novel saliency function that allows us to mathematically compute optimal low-rank adapter values in one shot. This joint approach improves accuracy by up to 5.66% on LLaMA-2-7B (combining 4-bit quantization and 2:4 sparsity) and achieves up to 4.3x layer-wise speedup.

Finally, Chapter 8 concludes the thesis by summarizing its key findings, reflecting on the impact of the Compression Trinity framework, and discussing potential avenues for future research. The code, pre-trained checkpoints, and interactive visualizations for the methods presented in this thesis are centralized at our re search hub<sup>2</sup>.

## Chapter 2

## Background

This chapter establishes the technical foundations necessary to understand the contributions of this thesis. We begin by identifying the computational bottleneck that dominates modern LLM workloads: the dense linear layers within the transformer architecture (Section 2.1). We then examine the hardware constraints that govern the execution of these layers on modern GPUs, introducing the memory hierarchy, specialized compute units, and the Roofline model that together determine whether a workload is limited by compute or by memory bandwidth (Section 2.2). With both the algorithmic bottleneck and the physical constraints established, we formalize the two orthogonal strategies for acceleration, reducing the number of training iterations through better optimization and reducing the cost of each iteration through hardware-aware compression (Section 2.3). We explore each strategy in turn: Section 2.4 surveys the landscape of first- and second-order optimizers, motivating the need for structured approximations that make curvature information tractable at scale; Section 2.5 characterizes the distinct computational regimes of LLM training and inference, revealing why diferent life cycle stages demand diferent compression techniques. Finally, Section 2.6 and Section 2.7 introduce the Compression Trinity as the unifying solution framework for this thesis and argue that the joint application of sparsity, quantization, and low-rank approximations is essential to overcome the limitations of any single technique applied in isolation.

## 2.1 The Transformer Bottleneck: Linear Layers

The Transformer has become the foundational architecture for virtually all modern Large Language Models (LLMs), demonstrating unparalleled scaling and performance on a wide array of language tasks [146]. In practice, an LLM is constructed by stacking a large number of identical Transformer blocks, often numbering in the dozens or even hundreds. This repetitive, block-based design means that the computational profile of a single block is representative of the model’s entire computational load. Understanding the specific operations within this block is therefore the first step toward identifying the primary opportunities for optimization.

A standard Transformer block, as illustrated in Figure 2.1, is composed of two primary sub-components. The first is the Self-Attention mechanism, which allows the model to weigh the importance of diferent tokens in a sequence relative to each other. The second is a position-wise Feed-Forward Network (FFN), which is typically a two- or three-layer Multi-Layer Perceptron (MLP) that provides the majority of the model’s representational capacity. In modern architectures, this FFN often takes the form of a SwiGLU variant [130]. These two components work in tandem, with the attention mechanism handling the aggregation of sequential information and the FFN processing that information at each token’s position.

![](images/5ab6c6d69279e3ff3c6bee15ca3ebdc92497626b445234446528b4e6b1691ed2.jpg)  
Figure 2.1: The compute graph of a standard Transformer block, highlighting the Self-Attention and Feed-Forward Network (FFN) sub-layers.

Connecting this architectural graph to its underlying mathematical operations reveals a critical insight: both sub-components are dominated by linear layers, which are implemented as dense matrix-matrix multiplications (GEMMs). The Self-Attention mechanism, for instance, computes its Query (Q), Key (K), and Value (V) representations through three independent linear layers, and a final Output (O) projection is applied after the attention scores are aggregated. Similarly, the SwiGLU FFN is composed of an Up-projection layer, a Gate layer, and a Down-projection layer. Consequently, the vast majority of computations and parameters in the Transformer block are contained within these dense matrix multiplications.

Each linear layer, parameterized by a weight matrix $W ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ , participates in three distinct matrix multiplications during a single training step. Given an input activation matrix $X \in \mathbb { R } ^ { b \times d _ { \mathrm { i n } } }$ , where b is the efective batch size (batch size × sequence length), the forward pass computes the layer’s output as $Y = X W ^ { T }$ . During backpropagation, two additional GEMMs are required: the weight gradient computation, $\nabla _ { W } \mathcal { L } = ( \nabla _ { Y } \mathcal { L } ) ^ { T } X$ , which determines how the weights should be updated, and the input gradient computation, $\nabla _ { X } \mathcal { L } = ( \nabla _ { Y } \mathcal { L } ) W$ , which propagates the error signal to the preceding layer. Crucially, while the forward pass multiplies the input by $W ^ { T }$ , the input gradient computation multiplies by W itself. This transpose relationship poses a fundamental challenge for structured compression: a sparsity pattern that is hardware-friendly along the rows of W (as required for the forward pass) may not be hardware-friendly along its columns (as required for the backward pass). This asymmetry is a central obstacle that we address directly in Chapter 4.

This architectural analysis is confirmed by empirical performance profiling, as shown in Figure 2.2. The ”linear” component is by itself the largest single computational bottleneck, consuming approximately 51.8% of the total training time for a model like LLaMA-3.1-8B. The ”Attention” component, which contains the self-attention computations with dynamic, data-dependent operations excluding the linear layers, accounts for another 9.6%. While the Attention mechanism’s unique properties present their own optimization challenges, this thesis will focus on the linear layers. As the largest and most dominant bottleneck, composed of standard static-weight GEMMs, these linear layers represent one of the most critical and impactful targets for optimization.

The evidence from both the architectural design and the performance profile establishes a clear conclusion: accelerating the dense linear layers is the central challenge for LLM eficiency. However, identifying the mathematical operations is only half the picture. To understand why these operations become bottlenecks, one must analyze the physical constraints of the hardware on which they execute. The following section will introduce the fundamental principles of GPU architecture and the Roofline model that govern the performance of these linear layers.

![](images/61b8b8b3f46d2f773676cdc0234c3b76c4090bf62af864b6eba29767fbb3a59b.jpg)  
Figure 2.2: Computational time breakdown for LLaMA-3.1-8B during training and inference, as profiled on a single NVIDIA H100 GPU. The ”linear” component is the largest single bottleneck. The batch size, input sequence length, and generation sequence length are set to 4, 1024, and 1024 respectively.

## 2.2 Hardware Constraints and the Roofline Model

While the Transformer architecture defines the operations to be performed, the execution speed is dictated by the underlying hardware. Modern LLMs are trained and deployed almost exclusively on Graphics Processing Units (GPUs), which are massive throughput-oriented processors. To understand the eficiency bottlenecks discussed in this thesis, we must first establish a model of how these devices process data, specifically focusing on the memory hierarchy, specialized compute units, and the theoretical limits defined by the Roofline model.

## 2.2.1 Memory Hierarchy and Tensor Cores

The computational pipeline of a GPU is constrained by two primary resources: Memory Bandwidth (how fast data can be moved) and Compute Throughput (how fast data can be processed).

Memory Hierarchy: Data movement in a GPU follows a hierarchy of speed and capacity. The bulk of model parameters and activations reside in High Bandwidth Memory (HBM), which ofers high capacity (e.g., 80GB on an NVIDIA H100) but relatively high latency and limited bandwidth compared to on-chip memory. To perform an operation, data must be moved from HBM to the smaller, faster L2 cache, and finally to the Streaming Multiprocessors’ (SM) shared memory and registers (SRAM). The cost of moving data from HBM is orders of magnitude higher than moving it within the chip. Consequently, algorithms that reuse loaded data multiple times (high data locality) are significantly more eficient than those that stream data for single use.

Tensor Cores vs. CUDA Cores: Historically, GPUs relied on general-purpose ”CUDA Cores” for arithmetic. However, the rise of Deep Learning necessitated specialized hardware. Modern architectures (e.g., NVIDIA Ampere and Hopper) feature Tensor Cores, specialized execution units designed essentially to perform one operation: matrix-multiply-and-accumulate $( D = A \times B + C )$ in a single clock cycle.

While standard CUDA cores operate on scalars or small vectors, Tensor Cores consume entire $8 \times 1 6$ or 16×16 matrices per instruction [105, 106]. This specialization allows Tensor Cores to deliver throughputs over an order of magnitude higher than CUDA cores. For example, on the NVIDIA H100, Tensor Cores introduce support for the FP8 data format, doubling the peak throughput compared to the standard BF16 [148] format used in previous generations like the A100. This massive disparity means that any algorithm not utilizing Tensor Cores eficiently leaves the vast majority of the GPU’s potential performance on the table.

## 2.2.2 The Roofline Model

The interaction between the memory bandwidth and compute throughput is formalized by the Roofline Model [152]. This model visualizes the theoretical peak performance of an algorithm based on its Arithmetic Intensity (I), defined as the number of floating-point operations (FLOPs) performed for every byte of data transferred from memory:

$$
I = { \frac { \mathrm { T o t a l } \mathrm { F L O P s } } { \mathrm { T o t a l } \mathrm { B y t e s } \mathrm { T r a n s f e r r e d } } }\tag{2.1}
$$

As illustrated in Figure 2.3, the Roofline model divides performance into two distinct regions:

• Memory-Bound Region (Sloped Line): When arithmetic intensity is low, the GPU’s compute units starve while waiting for data from HBM. In this region, performance is strictly limited by memory bandwidth. Improving performance here requires moving fewer bytes (e.g., via Quantization).

• Compute-Bound Region (Flat Line): When arithmetic intensity is high, data is loaded once and reused many times, keeping the compute units fully saturated. In this region, performance is limited by the peak FLOPs of the Tensor Cores. Improving performance here requires doing fewer operations (e.g., via Sparsity).

The ”knee” of the curve represents the transition point. Figure 2.3 highlights a critical trend in hardware evolution: the gap between compute and memory is widening. The NVIDIA H100 boasts significantly higher peak FLOPs than the A100 (especially with FP8), pushing the ”roof” higher. However, memory bandwidth has scaled much more slowly. This shifts the knee to the right, meaning that modern GPUs require increasingly higher arithmetic intensity to achieve peak utilization. This hardware reality fundamentally shapes the strategies required for optimization: Pretraining (high intensity) sits firmly under the compute roof, while Inference Decoding (low intensity) is trapped under the memory slope.

## 2.2.3 Sparse Tensor Core Acceleration

While standard Tensor Cores provide massive throughput for dense matrix multiplications, they perform redundant calculations if the weight matrix contains zeros. To address this, modern NVIDIA architectures introduced Sparse Tensor Cores designed to accelerate fine-grained structured sparsity, specifically the 2:4 pattern.

Roofline Model: NVIDIA GPUs (FP16 Tensor Core, Dense)  
![](images/65912d0cf91a38e1f71c489c8d1220b11f651cd5a56a9c1037a23162f2d86124.jpg)  
Figure 2.3: Roofline Model comparison between NVIDIA A100 and H100 GPUs. The H100 (orange) offers significantly higher peak compute (FLOPs). However, memory bandwidth has not scaled proportionally, shifting the ”knee” of the curve to the right. This implies that algorithms on H100 require a higher arithmetic intensity to escape the memory-bound region compared to the A100.

Mechanism and Format: The 2:4 structured sparsity format enforces a rigid constraint: in every contiguous block of 4 elements along the reduction dimension, at least 2 elements must be zero. This allows the hardware to compress the sparse matrix by 50%, storing only the non-zero values in a packed format alongside small metadata indices that record the original positions of the retained elements. During execution, the Sparse Tensor Core reads these indices to select only the relevant entries from the corresponding dense operand, skipping all multiplications that would involve a zero. Because the core performs the same operation in the same number of clock cycles but processes only half as many non-zero elements, it efectively doubles the theoretical peak throughput compared to the equivalent dense precision.

The Backward Pass Challenge: While 2:4 sparsity ofers massive theoretical gains, applying it to training is non-trivial. The Sparse Tensor Core requires the sparsity to exist along the reduction dimension. In the forward pass, this aligns naturally. However, in the backward pass, we must multiply by the transpose of the weights, misaligning the sparsity pattern relative to the hardware’s expected read order. This structural alignment challenge is a primary motivation for the custom pretraining strategies developed in Chapter 4 (SLOPE).

## 2.3 The Solution Space: Two Frontiers of Acceleration

Given the dominance of Linear layers identified in Section 2.1 and the rigid hardware constraints defined in Section 2.2, we can now formalize the optimization landscape. To minimize the total time required to train an

LLM or process a request, we can decompose the cost into two multiplicative factors:

$$
{ \mathrm { T o t a l ~ T i m e } } = \underbrace { \left( { \mathrm { N u m b e r ~ o f ~ I t e r a t i o n s } } \right) } _ { \mathrm { S a m p l e ~ E f f i c i e n c y } } \times \underbrace { \left( { \mathrm { T i m e ~ P e r ~ I t e r a t i o n } } \right) } _ { \mathrm { H a r d w a r e ~ E f f i c i e n c y } }\tag{2.2}
$$

This decomposition reveals two orthogonal frontiers for acceleration, each requiring distinct strategies:

1. Strategy 1: Sample Eficiency (Reducing the Number of Iterations). This strategy is exclusive to the training regime. If we can improve the optimization algorithm to converge in fewer steps, we reduce the total time even if the cost of each step remains constant. This motivates the study of advanced optimizers.

2. Strategy 2: Hardware Eficiency (Reducing the Time Per Iteration). This strategy applies to both training and inference. To reduce the cost of a single iteration, we must attack the hardware bottlenecks identified in the Roofline model: we must either reduce the FLOPs (for compute-bound operations) or reduce the memory trafic (for memory-bound operations).

The following sections will explore these two frontiers in detail, starting with Algorithmic Eficiency via advanced optimizers, followed by Hardware Eficiency via the diferent operating regimes of LLMs.

## 2.4 Strategy 1: Algorithmic Eficiency

Having defined the physical arena in which our models execute, we turn to the first frontier of acceleration: Sample Eficiency. If the hardware limits how fast we can compute a single update, we must strive to compute fewer updates overall. This brings us to the domain of optimization algorithms, which dictate the path a model takes through the loss landscape from random initialization to convergence.

## 2.4.1 First-Order Methods

Standard LLM training relies almost exclusively on first-order optimization methods, particularly adaptive gradient variants such as Adam [70] and AdamW [86]. These methods utilize the gradient vector $g = \nabla \mathcal { L } ( \theta )$ which represents the slope of the loss function. Geometrically, first-order methods approximate the loss landscape locally as a hyperplane (a linear approximation).

While computationally eficient—requiring only $\mathcal O ( d )$ memory and compute for d parameters—this linear approximation ignores the curvature of the landscape. In the highly non-convex and ill-conditioned optimization landscapes typical of deep neural networks, this blindness to curvature leads to two primary ineficiencies. In ”narrow valley” regions where the curvature is high in one direction and low in another, first-order methods tend to oscillate across the valley rather than moving down its floor, wasting iterations. Additionally, without knowledge of the local scale (curvature), determining the optimal step size (learning rate) is dificult, often requiring extensive tuning and warm-up schedules.

## 2.4.2 Second-Order Methods

Second-order methods address these limitations by incorporating the Hessian matrix $H = \nabla ^ { 2 } \mathcal { L } ( \theta )$ , which contains the second-order partial derivatives. By using the Hessian, these methods approximate the loss locally

as a quadratic function (a ”bowl”) rather than a plane. This allows for the computation of the Newton step:

$$
\Delta \theta = - H ^ { - 1 } g\tag{2.3}
$$

The Newton step uses the inverse Hessian to automatically rescale the gradient. It takes larger steps in directions of low curvature (flat plateaus) and smaller, more cautious steps in directions of high curvature (steep clifs). Theoretically, this property, known as afine invariance, allows second-order methods to converge in significantly fewer iterations than their first-order counterparts.

## 2.4.3 The Computational Barrier and Structured Approximations

Despite their theoretical superiority, exact second-order methods are intractable for LLMs due to the sheer size of the Hessian. For a model with d parameters, H is a d×d matrix. For a modest 7B parameter model, storing H would require exabytes of memory, and inverting it $( \mathcal { O } ( d ^ { 3 } ) )$ is computationally impossible. This presents a classic eficiency trade-of: second-order methods ofer high sample eficiency but sufer from catastrophic hardware ineficiency.

To make second-order optimization feasible, we must rely on Structured Approximations. The most prominent approach in deep learning is Kronecker-Factored Approximate Curvature (KFAC) [93].<sup>1</sup> KFAC approximates the Fisher Information Matrix (a proxy for the Hessian) not as a single dense matrix, but layerwise. It assumes the Hessian for a given layer can be approximated as the Kronecker product of two much smaller matrices, $H \approx A \otimes G$ , where A relates to the layer’s inputs and G to its output gradients.

Inverting this Kronecker product is eficient because $( A \otimes G ) ^ { - 1 } = A ^ { - 1 } \otimes G ^ { - 1 }$ . This reduces the inversion cost from $\mathcal { O } ( d ^ { 3 } )$ to the cubic size of the layer’s width $( { \bf e . g . } , \mathcal { O } ( w i d t h ^ { 3 } ) )$ ). However, even with KFAC, the memory cost of maintaining these curvature factors remains prohibitive for modern LLMs, often tripling the memory footprint compared to Adam [93, 116].

This unsolved challenge serves as the motivation for Chapter 3 (MKOR). We argue that Low-Rank factorization (like KFAC) is merely the starting point. To truly bridge the gap, achieving the sample eficiency of second-order methods with the hardware eficiency of first-order methods, we must apply the full Compression Trinity to the optimizer itself, combining Low-Rank updates with Sparsity and Quantization on the optimizer states.

## 2.5 Strategy 2: Hardware Eficiency and LLM Regimes

While advanced optimizers attack the ”Sample Eficiency” frontier to shorten training, they ofer no benefit during inference, where the number of steps is fixed by the user’s generation length. To accelerate inference and to further speed up training, we must attack the second frontier: Hardware Eficiency.

However, ”eficiency” is not a static target. As predicted by the Roofline model (Section 2.2.2), the bottleneck shifts dramatically depending on the operational regime. LLM execution is bifurcated into two dis tinct computational profiles: the Compute-Bound regime (dominating Training and Inference Prefill) and the Memory-Bound regime (dominating Inference Decoding).

## 2.5.1 The Compute-Bound Regimes: Training and Prefill

The Training phase<sup>2</sup> and the Inference Prefill phase share a fundamental characteristic: massive parallelism. In Training, the model processes large batches of sequences simultaneously. Similarly, in the Prefill phase of inference (processing the user’s prompt), the model computes attention and feed-forward outputs for all input tokens at once.

In these scenarios, the Arithmetic Intensity is high. The weight matrices are loaded from HBM to the chip once and reused across thousands of tokens (large batch size × sequence length). Consequently, both Training and Prefill sit firmly on the flat, Compute-Bound plateau of the Roofline model. In this region, the GPU’s compute units are fully saturated, and memory bandwidth is not the limiting factor.

Therefore, optimizing Training and Prefill requires strategies that strictly reduce the total number of operations (FLOPs). This makes Sparsity the premier accelerator for this regime. By skipping calculations for zero-valued weights, sparsity directly lowers the compute ceiling required to process the batch, translating to faster training steps and lower prompt latency.

## 2.5.2 The Memory-Bound Regime: Inference Decoding

Once the prompt is processed, the model enters the Decode phase, generating one token at a time. This phase is inherently sequential; the output of step t is required to compute step t + 1, preventing parallelization across the sequence dimension.

In this regime, the Arithmetic Intensity collapses. To generate a single token (or a small batch of tokens), the GPU must load the entire model (often 100GB+ for large models) from HBM to the chip, perform a single matrix-vector multiplication, and then discard the weights. The data reuse is minimal. Consequently, the Decode phase falls deep into the Memory-Bound slope of the Roofline model.

Here, the compute units (Tensor Cores) sit idle for the vast majority of execution time, starving for data. Reducing FLOPs (via Sparsity) provides diminishing returns because computation is not the bottleneck. Instead, acceleration is strictly defined by how fast data can be moved. This makes Quantization the dominant accelerator for decoding. By reducing the bit-width of the weights (e.g., from 16-bit to 4-bit), we reduce the data volume by 75%, efectively quadrupling the speed at which the model can be fed to the compute units.

The contrast between Training and Prefill, which require FLOP reduction, and Decoding, which requires Bandwidth reduction, reinforces the need for the Compression Trinity. No single compression technique can address both bottlenecks simultaneously: sparsity alone leaves the memory-bound decode phase untouched, while quantization alone cannot accelerate the compute-bound training and prefill phases. It is precisely the joint application of complementary techniques, sparsity to cut FLOPs, quantization to cut data movement, and low-rank approximations to recover lost accuracy, that enables a unified eficiency strategy across the entire LLM lifecycle.

## 2.6 The Compression Trinity as a Solution Framework

To address the specific bottlenecks and strategies identified for both pretraining and inference, we introduce the ”Compression Trinity,<sup>3</sup>” the conceptual framework for this thesis. This framework is built upon the three fundamental pillars of model compression: Sparsity, Quantization, and Low-Rank Approximations. Rather than viewing these as independent techniques, we posit that they are highly complementary tools that, when applied jointly, provide a comprehensive solution to the challenges of LLM eficiency. This section will introduce each pillar and map it to the two acceleration strategies defined in Section 2.4 and Section 2.5.

The first pillar, Sparsity, is the process of identifying and removing (pruning) the least important parameters from a weight matrix W, thereby reducing its efective size. This technique can be broadly categorized by the pattern of removed weights: unstructured sparsity removes individual weights, structured sparsity removes entire blocks (e.g., rows or columns), and semi-structured N:M sparsity enforces a fine-grained pattern, such as 2 non-zero weights out of every 4. Sparsity is the primary implementation of Compute Bound Optimizations (Reduce FLOPs). By reducing the number of non-zero parameters, it directly lowers the arithmetic cost, making it ideal for compute-bound regimes like pretraining. Simultaneously, by shrinking the storage requirement, it also assists with the Memory Bound Regime (Reduce Data Movement).

The second pillar, Quantization, is the process of reducing the numerical precision of the weights W (and sometimes activations) from high-precision formats like 32-bit floating point (FP32) to low-precision formats like 8-bit or 4-bit integers (INT8 or INT4). To maintain accuracy, especially in the presence of outlier values common in LLMs [27], this is often applied on a per-group basis. Quantization is the most powerful tool for Memory Bound Regimes (Reduce Data Movement). By reducing the bit-width of W from 16 (for BF16) to 4, it cuts the memory bandwidth requirement by 75%. This directly attacks the bottleneck of the memory-bound decode phase. While specialized hardware (e.g., FP8 Tensor Cores) allows quantization to also accelerate computation, its dominant benefit remains the massive reduction in memory trafic.

The third pillar, Low-Rank Approximations, is based on the hypothesis that the large weight matrices in LLMs are over-parameterized and have a low intrinsic rank [2]. This pillar exploits this redundancy by representing a large matrix as the product of two smaller, ”thin” matrices (e.g., W ≈ LR or ∆W = LR). This pillar is the most versatile of the three, acting as a bridge between Hardware Eficiency and Sample Eficiency.

For Hardware Eficiency (Strategy 2), representing W with far fewer parameters (e.g., d×r+r×d, where r ≪ d) reduces both the FLOPs and the memory footprint. Additionally, as an eficient means to mitigate accuracy degradation from sparsity and quantization, low-rank approximations provide a computationally cheap mechanism for representing fully dense matrices with unrestricted element values.

Finally, for Strategy 1 (Reduce Iterations), the pillars work together. While Low-Rank approximations typically provide the mathematical structure necessary to approximate curvature (e.g., factorizing the Hessian matrix), making these advanced optimizers computationally practical often requires the full Compression Trin ity. As we will demonstrate with MKOR (Chapter 3), solely relying on low-rank structure is often insuficient for fitting complex optimizer states into GPU memory. Instead, we must apply Sparsity and Quantization on top of the low-rank factors. Thus, Strategy 1 is not the domain of a single pillar, but rather the ultimate synthesis where all three pillars are combined to enable smarter, faster optimization.

## 2.7 The Case for a Joint Approach

The core argument of this thesis is that applying these pillars in isolation is an inherently limited approach. Each technique, when pushed to its extreme, hits a fundamental wall: aggressive sparsity causes catastrophic accuracy degradation, aggressive quantization fails due to the sensitivity of outlier parameters, and low-rank approximation alone cannot capture the full rank information necessary for a model’s capabilities.

However, when viewed through the lens of our identified strategies, these pillars become complementary pieces of a unified puzzle. Sparsity maximizes FLOP reduction. Quantization maximizes Bandwidth reduction. Low-Rank Approximation provides a mathematical structure to recover accuracy and enable advanced optimization.

It is important to clarify, however, that while the ultimate goal of this thesis is the joint application of the Compression Trinity, achieving this combination requires that each individual pillar be robust enough to support the others. If a single pillar is brittle, combining it with others leads to compounded errors and performance collapse. Therefore, not all chapters in this thesis focus on the simultaneous application of all three pillars. Instead, we adopt a staged approach: we dedicate specific chapters (specifically OPTIMA and PATCH) to pushing the boundaries of the Sparsity pillar in isolation. These contributions are necessary foundational steps, transforming sparsity from a fragile technique into a robust building block capable of withstanding the additional pressure of joint Quantization and Low-Rank approximation in the final unification (e.g., SLIM).

This thesis will demonstrate how this joint framework can be tailored to solve the distinct challenges of the LLM life-cycle:

1. Pretraining (Compute & Sample Eficiency): To solve the compute-bound pretraining problem, we advocate for a two-pronged approach. We use Strategy 1 by developing an advanced optimizer that leverages all three pillars of the Trinity to approximate curvature and accelerate convergence (as explored in MKOR [99], Chapter 3). Simultaneously, we use Strategy 2 by jointly applying Sparsity and Low-Rank Approximations to the weights during training (as explored in SLOPE [101], Chapter 4).

2. Inference (Memory Eficiency & Accuracy): For the post-training inference problem, the goal is to holistically apply all three pillars. We first establish a stable foundation by perfecting structured sparsity (as explored in OPTIMA, Chapter 5) and adapting it for modern hardware (as explored in PATCH, Chapter 6). With this foundation, we finally demonstrate the complete fulfillment of the Trinity: a oneshot method that jointly applies Sparsity, Quantization, and Low-Rank approximation to simultaneously attack compute, memory, and accuracy recovery (as explored in SLIM, Chapter 7).

In conclusion, this chapter has established the core problem (Linear layers), the physical constraints (Roofline model and Memory Hierarchy), the algorithmic opportunity (Sample Eficiency), and the Compression Trinity as our unified solution framework. The following chapters will now present the novel contributions of this thesis, demonstrating the efectiveness of this joint framework in unlocking new frontiers of eficiency.

Chapter 3

# MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates

Publication and Contributions. The content of this chapter is based on the paper “MKOR: Momentum-Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates” [99], published at the Conference on Neural Information Processing Systems (NeurIPS), 2023. This work was conducted in collaboration with Sikan Li, Zhao Zhang, and Maryam Mehri Dehnavi. Mohammad Mozafari conceived the algorithm, led the implementation, and designed and executed the experiments. Sikan Li assisted with conducting experiments. Zhao Zhang and Maryam Mehri Dehnavi supervised the project and contributed to the writing and revision of the manuscript.

## 3.1 Introduction

As established in Chapter 1, the pretraining phase of Large Language Models (LLMs) is dominated by massive computational costs. To mitigate this, we employ the second strategy of the Compression Trinity: reducing the total number of training iterations required for convergence. Second-order optimization methods have gained significant attention for this purpose, as they utilize curvature information, specifically the inverse of the Hessian matrix, to precondition gradients, thereby achieving higher convergence rates than first-order counterparts like SGD or Adam. However, these methods face a critical scalability wall. Since the size of the Hessian scales quadratically with the model parameters, computing and storing the exact Hessian and its inverse is computationally intractable for modern deep neural networks (DNNs), necessitating the use of approximation techniques.

Beyond computational complexity, the memory footprint of optimizer states presents a prohibitive bottleneck at the scale of LLMs. Standard first-order optimizers like Adam already require maintaining multiple state tensors (e.g., momentum and variance) for every model parameter, often consuming more GPU memory than the model weights themselves. Second-order methods exacerbate this issue by requiring the storage of curvature information, such as the Hessian or its factors. For models with billions of parameters, the memory required to store these high-precision curvature matrices frequently exceeds the capacity of modern hardware accelerators. Consequently, applying second-order optimization to LLMs requires not only approximating the curvature to reduce computational costs but also aggressively compressing them to fit within memory constraints.

Existing approximation methods attempt to make second-order optimization feasible but typically address only one aspect of the eficiency bottleneck. One common approach is Natural Gradient Descent (NGD) [4], which substitutes the Hessian with the Fisher Information Matrix (FIM) [8]. To handle the memory limitations of large models, algorithms like Kronecker-Factored Approximate Curvature (KFAC) [93] approximate the FIM using block-diagonal sparsity, where each block corresponds to a layer. However, inverting these blocks remains computationally expensive, scaling with $\mathcal { O } ( d ^ { 3 } )$ where d is the layer dimension. Consequently, KFAC implementations [9, 145, 117, 113, 132, 116] must update the curvature information infrequently (e.g., every 100-1000 iterations), which damages convergence rates. Alternative methods like SNGD [124, 157, 102] and KBFGS [45] attempt to shift the complexity to the batch dimension $( \mathcal { O } ( b ^ { 3 } )$ or $\mathcal { O } ( b d ^ { 2 } ) )$ . While efective for small batches, these fail in Transformer models [146] where the efective batch size scales with sequence lengths that can reach thousands of tokens [138]. Thus, existing methods hit a wall: they are either too computationally heavy due to matrix inversion or fail to scale with sequence length.

To overcome these limitations, we apply the Compression Trinity directly to the optimizer’s internal computations. We present MKOR, a Momentum-Enabled Kronecker-Factorization-Based Optimizer with Rank-1 Updates. MKOR unifies the Sparsity pillar, inherited through block-diagonalization, with the Low-Rank Approximation pillar, which approximates the inverse of the covariance blocks using rank-1 updates via the Sherman-Morrison identity. This formulation fundamentally alters the eficiency landscape, reducing the inversion complexity from $\mathcal { O } ( d ^ { 3 } )$ to $O ( d ^ { 2 } )$ while simultaneously alleviating the communication bottleneck. Unlike standard second-order methods that require synchronizing large inverse factors $( \mathcal { O } ( d ^ { 2 } ) )$ , MKOR synchronizes only the rank-1 approximation vectors, reducing communication costs to $\mathcal O ( d )$ . These eficiency gains allow MKOR to update second-order information up to 100 times more frequently compared to state-ofthe-art implementations like KAISA [116] and HyLo [102]. Crucially, unlike recent attempts like Eva [163] which store vectors but sacrifice momentum, MKOR fully preserves momentum information while achieving $O ( d ^ { 2 } )$ complexity.

The most closely related method to MKOR is Eva [163], which also targets the scalability bottleneck of KFAC by maintaining only rank-1 approximation vectors rather than full covariance factor inverses. This gives Eva a memory overhead of only $O ( d )$ , significantly lower than MKOR’s $O ( d ^ { 2 } )$ . However, this aggressive memory reduction comes at a fundamental cost: Eva discards the accumulated momentum of the inverse factors entirely, retaining only the most recent rank-1 snapshot. In contrast, MKOR preserves full momentum history in its factor inverses via the Sherman-Morrison update (Equation 3.5 and Equation 3.6), which exponentially averages past curvature information through the $\gamma$ decay term. As we demonstrate in Section 3.4, this distinction has practical consequences: Eva fails to converge to the target accuracy on several benchmarks where MKOR succeeds, suggesting that the memory savings come at the expense of optimization stability. MKOR thus occupies a deliberate design point in the memory–convergence tradeof, investing $O ( d ^ { 2 } )$ memory to retain curvature history that proves essential for reliable convergence at scale.

However, low-rank approximation alone is insuficient to fully resolve the scalability challenge. While it reduces the computational cost of inversion, it does not inherently address the memory overhead of storing the KFAC factors. To tackle this, we integrate the Quantization pillar. By performing computations and storage in half-precision, we significantly reduce the memory footprint of the curvature factors. This is non-trivia because standard second-order methods rely on operations like Cholesky decomposition or matrix inversion, which are numerically unstable in low precision. By pivoting to rank-1 updates via the Sherman-Morrison identity, MKOR replaces these unstable operations with simple matrix-vector products. These operations are inherently more robust to quantization noise, enabling us to store and compute curvature in half-precision without divergence.

While second-order methods provide a significant advantage in the initial phase of training, their relative benefit over first-order methods can diminish in later stages. To maximize eficiency, we introduce a hybrid variant, MKOR-H. This method combines the rapid initial convergence of second-order optimization with the low computational overhead of first-order methods. By utilizing a loss-reduction-rate-based switching mech anism, MKOR-H automatically transitions between regimes to ensure optimal resource utilization throughout the entire pretraining process.

Our experiments demonstrate that MKOR successfully validates the eficacy of applying the Compression Trinity to the optimizer. MKOR outperforms state-of-the-art distributed second- and first-order methods by up to 2.57×, reducing the training time of BERT-Large-Uncased from 8 hours to 3 hours on 64 A100 GPUs. Additionally, it achieves new state-of-the-art metrics on the GLUE dataset, successfully converging in settings where other second-order methods such as KFAC fail.

## 3.2 Background

Training a neural network involves solving an optimization problem to find the optimal values for a set of weights $\mathcal { W } = \{ W ^ { m } \} _ { m = 1 } ^ { M }$ , where M is the number of layers in the network and $W ^ { m }$ is a matrix in $\mathbb { R } ^ { d \times d }$ Second-order methods precondition the weights of the network with the inverse of the Hessian for better convergence rates. Block-diagonal approximations of NGD methods replace the Hessian with the block-diagonal FIM as shown in Equation 3.1, where $\boldsymbol { w ^ { m } } \in \mathbb { R } ^ { d ^ { 2 } }$ is the vector representation of $W ^ { m } , F ^ { m }$ is the block corresponding to that layer and L is the loss function. Martens [92] shows that the FIM matches the Gauss-Newton matrix under certain conditions.

$$
w ^ { m } : = w ^ { m } - \alpha ( F ^ { m } ) ^ { - 1 } \nabla _ { w ^ { m } } \mathcal { L }\tag{3.1}
$$

KFAC-based methods reformulate the FIM block as the Kronecker product of two matrices. Equation 3.2 shows the update rule in KFAC, where $\mathcal { L }$ is the loss function and $( L _ { t } ^ { m } ) ^ { - 1 }$ and $( R _ { t } ^ { m } ) ^ { - 1 }$ are the inverses of the left and right factors, respectively.

$$
\begin{array} { r } { \boldsymbol { W } ^ { m } : = \boldsymbol { W } ^ { m } - \alpha \big ( \boldsymbol { L } _ { t } ^ { m } \big ) ^ { - 1 } \nabla _ { \boldsymbol { W } ^ { m } } \mathcal { L } \big ( \boldsymbol { R } _ { t } ^ { m } \big ) ^ { - 1 } } \end{array}\tag{3.2}
$$

$( L _ { t } ^ { m } ) ^ { - 1 }$ and $( R _ { t } ^ { m } ) ^ { - 1 }$ in Equation 3.2 are computed using Equation 3.3 and Equation 3.4, respectively, where $a ^ { m }$ is the activation value of a sample at layer $m ,$ , and $g _ { m } = \nabla _ { a ^ { m - 1 } } \mathcal { L }$ and $\gamma$ incorporate the momentum feature to avoid extreme changes in the factors.

$$
L _ { t } ^ { m } = \gamma L _ { t - 1 } ^ { m } + ( 1 - \gamma ) \mathbb { E } [ g _ { t } ^ { m } g _ { t } ^ { m T } ]\tag{3.3}
$$

$$
R _ { t } ^ { m } = \gamma R _ { t - 1 } ^ { m } + ( 1 - \gamma ) \mathbb { E } [ a _ { t } ^ { m - 1 } a _ { t } ^ { m - 1 } ^ { T } ]\tag{3.4}
$$

![](images/bc74bc1932496ee63f099cd5115ef92692e828da1b0ff46d1f4caef2904626ee.jpg)

Figure 3.1: MKOR for layer m on a single worker. The inputs of MKOR are the activations $A _ { t } ^ { m }$ , the gradients of the loss function with respect to the inputs $G _ { t } ^ { m }$ , and the gradients of the loss function with respect to the weights $\nabla _ { \boldsymbol { W } ^ { m } } \mathcal { L }$ . The output is the update values $\Delta W ^ { m }$

## 3.3 Methodology

In this section, we first present the MKOR algorithm, its computation and communication complexity, then present hybrid MKOR (MKOR-H), and finally discuss MKOR’s convergence and stability.

## 3.3.1 The MKOR Algorithm

Algorithm 1 summarizes the MKOR optimizer for a single layer and Figure 3.1 shows the workflow. For each layer (line 1 in Algorithm 1) MKOR updates the second-order information and preconditions the gradients, and at the end the backend optimizer updates the weight using the preconditioned gradients (line 14 in Algorithm 1).

Rank-1 Approximation. For the rank-1 approximations of the covariance matrices, we use the average of the values across all the samples, i.e. $\mathbf { a } _ { \mathbf { t } } ^ { \mathbf { m } - 1 } = \mathbb { E } [ a _ { t } ^ { m - 1 } ]$ and $\mathbf { g } _ { \mathbf { t } } ^ { \mathbf { m } } = \mathbb { E } [ g _ { t } ^ { m } ]$ (lines 2 and 3 in Algorithm 1 and Figure 3.1-a). $( A _ { t } ^ { m - 1 } ) _ { : , i } ^ { - 1 }$ and $( G _ { t } ^ { m } ) _ { : , i } ^ { - 1 }$ show the $i ^ { t h }$ column of $( A _ { t } ^ { m - 1 } ) ^ { - 1 }$ and $( G _ { t } ^ { m } ) ^ { - 1 }$ respectively, where $A _ { t } ^ { m - 1 }$ and $G _ { t } ^ { m }$ are the activations and the gradients of layer m respectively.

Norm-Based Stabilizer. The values in the factor inverses in second-order methods can become large or vanish due to extremely large or small values in activations and gradients, leading to numerical instabilities and over/underflows. Since the inverse of the factors are directly multiplied by the gradients to find the update values, it can cause oscillations or even divergence. MKOR uses a norm-based stabilizer to detect the numerical instability and addresses it by modifying the inverse of the factors accordingly (lines 5 and 6 in Algorithm 1 and Figure 3.1-b). More details on the norm-based stabilizer are in Section 3.3.3.

Algorithm 1 MKOR Algorithm for a Single Layer m   
Input: $A _ { t } ^ { m - 1 } , G _ { t } ^ { m } , W _ { t - 1 } ^ { m }$   
Output: ${ W } _ { t } ^ { m }$   
1: if $m \in$ Second Order Layers then   
2: $\begin{array} { r } { \mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m - 1 } } \gets \frac { 1 } { b } \sum _ { i = 1 } ^ { b } ( A _ { t } ^ { m - 1 } ) _ { : , i } } \end{array}$ ▷ Approx: $A _ { t } ^ { m - 1 } A _ { t } ^ { m - 1 } { } ^ { T } \approx \mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m - 1 } } \mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m - 1 } } { } ^ { T }$   
3: $\begin{array} { r } { \mathbf { g _ { t } ^ { m } }  \frac { 1 } { b } \sum _ { i = 1 } ^ { b } ( G _ { t } ^ { m } ) _ { : , i } } \end{array}$ ▷ Approx: $G _ { t } ^ { m } G _ { t } ^ { m T } \approx \mathbf { g _ { t } ^ { m } } \mathbf { g _ { t } ^ { m T } }$   
4: $\mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m - 1 } } , \mathbf { g } _ { \mathrm { t } } ^ { \mathrm { m } } \gets \mathrm { A l l R e d u c e } ( \mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m - 1 } } , \mathbf { g } _ { \mathrm { t } } ^ { \mathrm { m } } )$ ▷ Synchronize Approximations   
5: $L t \hat { \mathbf { \Omega } } ^ { - 1 } \mathbf { \Omega } ^ { m ^ { - 1 } } \gets \mathbf { i f } \vert { L _ { t - 1 } ^ { m } } ^ { - 1 } \vert > \epsilon$ then ${ \zeta { L _ { t - 1 } ^ { m } } ^ { - 1 } } + ( 1 - \zeta ) I$ else $L _ { t - 1 } ^ { m } { } ^ { - 1 }$ ▷ Norm-Based Stabilization   
6: ${ R _ { t - 1 } ^ { \hat { m } } } ^ { - 1 } \gets \mathbf { i f } \ | { R _ { t - 1 } ^ { m } } ^ { - 1 } | > \epsilon$ then ${ \zeta { R _ { t - 1 } ^ { m } } ^ { - 1 } } + ( 1 - \zeta ) I$ else $R _ { t - 1 } ^ { m } { } ^ { - 1 }$   
7: $\begin{array} { r } { L _ { t } ^ { m - 1 } \gets \gamma L _ { t - 1 } ^ { \hat { m } } ^ { - 1 } + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) \mathbf { g } _ { t } ^ { \mathbf { m } T } L _ { t - 1 } ^ { \hat { m } } \mathbf { \Sigma } _ { \mathbf { g } _ { t } ^ { \mathbf { m } } } ) } L _ { t - 1 } ^ { \hat { m } } \mathbf { \Sigma } _ { \mathbf { g } _ { t } ^ { \mathbf { m } } \mathbf { g } _ { t } ^ { \mathbf { m } T } L _ { t - 1 } ^ { \hat { m } } } ^ { - 1 } - 1 } \mathbf { g } _ { t } ^ { \mathbf { m } } \mathbf { g } _ { t } ^ { \mathbf { m } T } L _ { t - 1 } ^ { \hat { m } }  \end{array}$ ▷ SM-Based Factor   
Inversion   
8: $\mathbf { \Pi } _ { R _ { t } ^ { m - 1 } } ^ { \mathrm { { s r o u n } } }  \gamma R _ { t - 1 } ^ { \hat { m } } ^ { \mathrm { { \alpha } } } + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) \mathbf { a } _ { \mathrm { t } } ^ { \mathrm { m T } } R _ { t - 1 } ^ { \hat { m } } ^ { \mathrm { { \alpha } } } \mathbf { \bar { \mathbf { \alpha } } } _ { \mathbf { a } _ { t } ^ { \mathrm { m } } } ^ { - 1 } ) } R _ { t - 1 } ^ { \hat { m } } - \mathbf { \Pi } _ { \mathbf { a } _ { t } ^ { \mathrm { m } } \mathbf { a } _ { t } ^ { \mathrm { m } } } ^ { - 1 } R _ { t - 1 } ^ { \hat { m } } - \mathbf { \Pi }$   
9: $\Delta \hat { W _ { t } ^ { m } } \gets L _ { t } ^ { m - 1 } \nabla _ { W ^ { m } } \mathcal { L } R _ { t } ^ { m - 1 }$ ▷ Precondition Gradients   
10: $\begin{array} { r } { \Delta W _ { t } ^ { m } \gets \frac { | \nabla _ { W ^ { m } } \mathcal { L } | } { \Delta \hat { W _ { t } ^ { m } } } \Delta \hat { W _ { t } ^ { m } } } \end{array}$ ▷ Rescale Gradients   
11: else   
12: $\Delta W _ { t } ^ { m } \gets \nabla _ { W ^ { m } } { \mathcal { L } }$   
13: end if   
14: $W _ { t } ^ { m } \gets \mathrm { O p t i m i z e r . s t e p } ( \Delta W _ { t } ^ { m } , W _ { t - 1 } ^ { m } )$   
Return: $W _ { t } ^ { m } .$

SM-Based Inverter. MKOR directly modifies the inverse of the left and right factors using rank-1 updates, while using the momentum for better convergence. If E $[ g ^ { m } g ^ { m ^ { T } } ]$ is approximated using a rank-1 matrix $\mathbf { g } ^ { \mathbf { m } } \mathbf { g } ^ { \mathbf { m } ^ { \mathbf { I } } }$ and using the Sherman-Morrison identity, Equation 3.5 is obtained (line 7 in Algorithm 1 and Figure 3.1-c).

$$
L _ { t } ^ { m - 1 } = \gamma L _ { t - 1 } ^ { m } { } ^ { - 1 } + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) { \bf g _ { t } ^ { \bf m } } ^ { T } L _ { t - 1 } ^ { m } { } ^ { - 1 } { \bf g _ { t } ^ { \bf m } } ) } L _ { t - 1 } ^ { m } { } ^ { - 1 } { } ^ { \bf g _ { t } ^ { \bf m } } { \bf g _ { t } ^ { \bf m } } ^ { T } L _ { t - 1 } ^ { m } { } ^ { - 1 }\tag{3.5}
$$

Furthermore, if Equation 3.4 is approximated using $\mathbb { E } [ a _ { t } ^ { m - 1 } a _ { t } ^ { m - 1 } ] \approx \mathbf { a _ { t } ^ { m } } \mathbf { a _ { t } ^ { m } } ^ { \mathbf { T } }$ with a similar derivation, Equation 3.6 is obtained (line 8 in Algorithm 1 and Figure 3.1-c).

$$
R _ { t } ^ { m - 1 } = \gamma R _ { t - 1 } ^ { m } { } ^ { - 1 } + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) { \bf a } _ { \bf t } ^ { \bf m T } R _ { t - 1 } ^ { m } { } ^ { - 1 } { \bf a } _ { \bf t } ^ { \bf m } ) } R _ { t - 1 } ^ { m } { } ^ { - 1 } { \bf a } _ { \bf t } ^ { \bf m } { \bf a } _ { \bf t } ^ { \bf m ^ { T } } R _ { t - 1 } ^ { m } { } ^ { - 1 }\tag{3.6}
$$

Rescaling Gradients. Preconditioning the gradients using the computed factors can change gradient norms. Sometimes, these changes interfere with the efect of the learning rate on the training process. To alleviate this and to make learning rate schedulers more efective, the preconditioned gradients are scaled so that their norm matches the original norms (line 10 in Algorithm 1 and Figure 3.1-d).

Complexity Analysis. MKOR reduces the memory, communication, and computation costs for factor inversion. Table 3.1 compares the overheads of diferent optimizers. (1) Computation Complexity. MKOR inverts the left and right factors in Equation 3.2 using Equation 3.5 and Equation 3.6, both of which can be computed using matrix-vector multiplications, and have $\mathcal { O } ( d ^ { 2 } )$ computation complexity, in contrast to KFAC and SNGD methods that need $\mathcal { O } ( d ^ { 3 } )$ and $\mathcal { O } ( b ^ { 3 } )$ complexity to invert matrices in $\mathbb { R } ^ { d \times d }$ and $\mathbb { R } ^ { b \times b }$ respectively. (2) Communication Complexity. The only data that is synchronized among diferent workers in MKOR is the two rank-1 approximations that have 2d elements. With quantization, this size can be halved. In KFAC, the activation and gradient covariance matrices and the inversion of left and right factors need to be synchronized between all the workers, leading to $4 d ^ { 2 }$ data transfers. In SNGD , the activations and gradients are synchronized, leading to 2bd data transfers and the inverted kernels are broadcast, resulting in $b ^ { 2 }$ data transfers. Reducing the communication complexity of MKOR from quadratic to linear results in better performance on large number of workers. (3) Memory Overhead. MKOR needs to store the inverse of the left and right factors and two rank-1 approximation vectors, leading to $2 d ^ { 2 } + 2 d$ memory overhead, and using half-precision computations further reduces this. KFAC stores the activation and gradient covariance matrices and the left and right factors, leading to $4 d ^ { 2 }$ memory overhead. SNGD stores the activations, the gradients, and the kernels they use as second-order information, leading to $2 b d + b ^ { 2 }$ memory complexity.

Table 3.1: The computation and communication complexity and memory overhead of the state-of-the-art implementations of the first- and second-order (second-order optimizers are written in bold). The division by 2 in MKOR is because MKOR uses half-precision computations. The complexity of KFAC-based methods depends on layer dimensions while SNGD methods mostly depend on the batch size. In transformers, due to the scaling of the batch size by the sequence length, batch sizes and layer dimensions are comparable, making both KFAC- and SNGD-based methods more expensive than SGD.
<table><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>ComputationalComplexity</td><td rowspan=1 colspan=1>Memory Overhead</td><td rowspan=1 colspan=1>CommunicationComplexity</td></tr><tr><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( d ^ { 2 } + b d ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 d ^ { 2 } / 2 ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 d / 2 ) } }$ </td></tr><tr><td rowspan=1 colspan=1>SNGD (HyLo)</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( b ^ { 3 } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 b d + b ^ { 2 } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 b d + b ^ { 2 } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>KFAC (KAISA)</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( d ^ { 3 } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 4 d ^ { 2 } ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 4 d ^ { 2 } ) } }$ </td></tr><tr><td rowspan=1 colspan=1>Eva</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( d ^ { 2 } + b d ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 d ) } }$ </td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( 2 d ) } }$ </td></tr><tr><td rowspan=1 colspan=1>SGD (Momentum)</td><td rowspan=1 colspan=1>-</td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( d ^ { 2 } ) } }$ </td><td rowspan=1 colspan=1>-</td></tr><tr><td rowspan=1 colspan=1>ADAM / LAMB</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\overline { { \mathcal { O } ( d ^ { 2 } ) } }$ </td><td rowspan=1 colspan=1></td></tr></table>

It is worth comparing MKOR’s complexity profile directly with Eva, as both methods share the same $O ( d ^ { 2 } + b d )$ computational complexity and achieve linear communication costs. The key distinction lies in the memory–accuracy tradeof. Eva achieves $O ( d )$ memory by storing only the rank-1 approximation vectors and discarding the factor inverses after each preconditioning step. MKOR, by contrast, retains the full $d \times d$ inverse factors in half-precision, resulting in $O ( d ^ { 2 } / 2 )$ memory but enabling momentum accumulation across iterations. While this is a meaningful increase in memory, it remains substantially lower than $\mathrm { K F A C " s } O ( 4 d ^ { 2 } )$ overhead and, as shown in Section 3.4, translates to more stable convergence, particularly in settings where Eva’s lack of momentum leads to suboptimal solutions or failure to reach the target accuracy.

## 3.3.2 Hybrid MKOR

We observed that second-order methods, including MKOR, usually accelerate training more during the first iterations of the training time, and as the loss flattens, their advantage over their first-order counterparts be comes less noticeable. This is because the second-order information of the loss functions approach identity near convergence points. Thus we designed a hybrid second- and first-order optimizer with a loss decrease ratebased switching method (MKOR-H). MKOR-H evaluates the changes in the loss function in diferent iterations and switches back to first-order methods if needed for an eficient trade-of between the costly second-order updates and their benefits for convergence.

## 3.3.3 MKOR Convergence and Stability

Inversion Frequency. Due to the high factor inversion costs in KFAC- and SNGD-based methods, researchers use the stale factor approach, which updates the inverted factors every f iterations and reuses the results in the other iterations in their preconditioning to reduce the computation and communication costs. The reciprocal of inversion frequency, f, varies from a few 100s to a few 1000s. Our experiments show tha in average-sized models such as ResNet-50 [56], in an iteration that includes the inversion of factors, the cost of KAISA and HyLo is 150× more than an SGD iteration that reuses stale factors. Furthermore, more than 98% of the total cost in those iterations are spent on matrix inversion.

The stale factors approach can lead to good preconditioners if the loss function landscape does not vary significantly in each iteration. However, this is a strong assumption and doesn’t necessarily hold in practice. Also, increasing the inversion frequency can benefit the convergence rate of the second-order methods. In addition, our experiments show that using stale factors can lead to converging to local minima in the loss function and damage the generalization of the model.

Numerical Stability. In second-order techniques, we need to invert or find the roots of matrices of diferent sizes, which are usually not full-rank, resulting in numerical issues. The KFAC implementation uses singular value decomposition (SVD) of the factors and masks the eigenvalues that are close to zero to deal with singular matrix inversion issues. In practice, the eigenvalues of the left and right factors in KFAC-based methods computed from Equation 3.3 and Equation 3.4 are increased manually by adding µI to each of them to improve numerical stability $( \mu > 0$ is called the damping factor), but MKOR doesn’t need such numerical fixes. Furthermore, HyLo uses two decomposition methods to sample the batch of inputs, namely KID and KIS. KID requires inverting matrices in $\mathbb { R } ^ { b \times b }$ of rank min(b, d), thus for batch sizes larger than d in a specific layer, the method fails.

Unlike SVD or other iterative methods used for factor inversion, MKOR doesn’t sufer from numerical instabilities that rise from large condition numbers. MKOR has a single scalar division, in which the denominator is guaranteed to be non-zero based on Theorem 3.3.1, eliminating the numerical over/under-flow possibility and the need for damping factors (required by other second-order methods for computational stability).

Lemma 3.3.1. The factors computed using Equation 3.5 and Equation 3.6 are all positive-definite.

Anil et al. [5] suggest using double precision representation of numbers to avoid numerical instabilities in inverting or computing the roots of matrices. This approach adds more costs to the matrix inversion and increases the time complexity of the main bottleneck in second-order methods.

MKOR does not need higher precision computations, and can use half-precision floating point operations to reduce costs significantly. This will improve the memory utilization and reduce the communication costs in GPUs by 2× while using cheaper computation blocks for half-precision operations. Theorem 3.3.2 shows an upper bound on the quantization error efect in the MKOR updates.

Lemma 3.3.2. Assuming that the maximum quantization error is $\epsilon ,$ the maximum number in matrices and vectors is m, and the dimension of the vectors and matrices are d and $d \times d$ respectively, the quantization error ofEquation 3.5 and Equation 3.6 is $O ( ( \gamma + 4 \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } } m ^ { 3 } d ^ { 2 } ) \epsilon )$

Exploding Gradients Problem. In second-order methods, where the gradients are preconditioned by various factors, the exploding gradient problem is worsened. Our experiments show that in first-order methods, by choosing a learning rate that doesn’t lead to divergence in the first few iterations, explosion in gradients almost never occurs. On the other hand, in second-order methods, we observe that the explosion can occur at any iteration, and both KFAC and SNGD implementations are prone to this problem. This can lead to ripples in accuracy and divergence.

One of the main approaches for solving the exploding gradient problem is choosing small values for the learning rate, limiting the convergence rate significantly. In particular, small learning rates damage the second order methods and make them almost as performant as their first-order counterparts.

Considering that SGD is more robust against the exploding gradients and taking advantage of the direct control of MKOR on the inverse of the factors, the factors in MKOR are modified to lean toward SGD once the possibility of exploding gradients is detected using Equation 3.7 and Equation 3.8, where ζ is a hyperparameter that controls the amount of information from the original factors that needs to be saved in the new factors.

$$
\hat { L _ { t } ^ { m } } = \zeta L _ { t } ^ { m } + ( 1 - \zeta ) I\tag{3.7}
$$

$$
\hat { R _ { t } ^ { m } } = \zeta R _ { t } ^ { m } + ( 1 - \zeta ) I\tag{3.8}
$$

By expanding Equation 3.2 with the new factors, we will get Equation 3.9, which reduces the loss based on Theorem 3.3.3. The first term in the right-hand side of Equation 3.9 is the KFAC term, the second and third terms are the left and right preconditioned versions, and the last term is the SGD term.

$$
\begin{array} { r l } & { \hat { L } ^ { m ^ { - 1 } } \nabla _ { W ^ { m } } \mathcal { L } \hat { R } ^ { m ^ { - 1 } } = \zeta ^ { 2 } L ^ { m ^ { - 1 } } \nabla _ { W ^ { m } } \mathcal { L } R ^ { m ^ { - 1 } } } \\ & { \qquad + \zeta ( 1 - \zeta ) L ^ { m ^ { - 1 } } \nabla _ { W ^ { m } } \mathcal { L } + \zeta ( 1 - \zeta ) \nabla _ { W ^ { m } } \mathcal { L } R ^ { m ^ { - 1 } } + ( 1 - \zeta ) ^ { 2 } \nabla _ { W ^ { m } } \mathcal { L } } \end{array}\tag{3.9}
$$

Lemma 3.3.3. Given a diferentiable function $\mathcal { L } ( w )$ with first-order Taylor series approximation $\hat { \mathcal { L } } ( w \mathrm { ~ - ~ }$ $\Delta w ) = \mathcal { L } ( w _ { 0 } ) - \Delta w ^ { T } \nabla _ { w } \mathcal { L } ( w _ { 0 } )$ around point $w _ { 0 } ,$ , assuming that at point $w _ { 0 }$ the second-order derivative of the function $\mathcal { L } ( w )$ is given as $\nabla _ { w } ^ { 2 } \mathcal { L } ( w _ { 0 } ) = H = L \otimes R ,$ , where L and R are positive-semi-definite matrices, for a value of $\begin{array} { r } { \mathbf { \dot { \Sigma } } \Delta w = ( ( \zeta L ^ { - 1 } + ( 1 - \zeta ) I ) \otimes ( \zeta R ^ { - 1 } + ( 1 - \zeta ) I ) ) \nabla \mathcal { L } ( w _ { 0 } ) } \end{array}$ , the inequality $\hat { \mathcal { L } } ( w _ { 0 } - \Delta w ) < \mathcal { L } ( w _ { 0 } )$ holds.

While this modification can avoid exploding gradients, overusing it with small values of ζ will convert MKOR to SGD. MKOR uses a factor norm-based metric that observes the infinity norm of the factors, and if they are greater than a specific threshold, the process of factor modification will be triggered.

## 3.4 Experimental Results

In this section, we demonstrate the performance of MKOR on a large language model using diferent benchmarks, and analyze the timing of diferent components in diferent first- and second-order algorithms. For results on more models and training sets, please refer to Appendix A.

Experiment Setup. For the BERT-Large-Uncased pre-training and fine-tuning experiments, we use up to 64 A100 GPUs on the Polaris [6] cluster, which has 560 nodes each with 4 NVIDIA A-100 GPUs with NVLink interconnects. The rest of the experiments are conducted on the Mist cluster [22], with 54 nodes each having 4 NVIDIA V100 GPUs with 32GB memory and NVLink inter-node connections. Each training experiment is conducted 5 times and the median timing is reported; reported accuracies are the median across multiple runs. Table 3.2 summarizes the models, datasets, and GPU architectures used in our experiments. For BERT-Large-Uncased pre-training, we use the same hyperparameters as [112], with factors in KAISA updated every 50 iterations and factors in MKOR and MKOR-H updated every 10 iterations. For ResNet-50, we follow the hyperparameters from [116], with MKOR factors updated every 10 iterations and the learning rate decaying by a factor of 2 at the end of epochs 25, 35, 40, 45, 50, 55, and 56. Our code base is publicly available at https://github.com/Mohammad-Mozaffari/mkor.

Table 3.2: List properties of the models, datasets, and settings used in our experiments.
<table><tr><td rowspan=1 colspan=2>Model</td><td rowspan=1 colspan=3>Dataset</td><td rowspan=1 colspan=2>GPU</td></tr><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>#Parameters</td><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=1>Train</td><td rowspan=1 colspan=1>Test</td><td rowspan=1 colspan=1>Arch</td><td rowspan=1 colspan=1>#</td></tr><tr><td rowspan=1 colspan=1>BERT-Large-Uncased</td><td rowspan=1 colspan=1>335.1M</td><td rowspan=1 colspan=1>Wikipedia -BookCorpus</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A100</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>ResNet-50</td><td rowspan=1 colspan=1>25.5M</td><td rowspan=1 colspan=1>ImageNet</td><td rowspan=1 colspan=1>1.2M</td><td rowspan=1 colspan=1>50k</td><td rowspan=1 colspan=1>V100</td><td rowspan=1 colspan=1>64</td></tr><tr><td rowspan=1 colspan=1>AlexNet</td><td rowspan=1 colspan=1>20.3M</td><td rowspan=1 colspan=1>CIFAR-100</td><td rowspan=1 colspan=1>50K</td><td rowspan=1 colspan=1>10K</td><td rowspan=1 colspan=1>V100</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>BERT-Base-Cased</td><td rowspan=1 colspan=1>108.9M</td><td rowspan=1 colspan=1>SQuAD v1.1</td><td rowspan=1 colspan=1>87.6K</td><td rowspan=1 colspan=1>10.6K</td><td rowspan=1 colspan=1>V100</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>BERT-Large-Cased</td><td rowspan=1 colspan=1>335.1M</td><td rowspan=1 colspan=1>IMDB</td><td rowspan=1 colspan=1>25K</td><td rowspan=1 colspan=1>25K</td><td rowspan=1 colspan=1>V100</td><td rowspan=1 colspan=1>4</td></tr></table>

Large Language Models. We pre-train BERT-Large Uncased and fine-tune it for diferent question-answering and text classification tasks. We use a setup similar to KAISA [116] for pre-training and fine-tuning. We use Fused LAMB [159] as the state-of-the-art first-order baseline. Similar to KAISA [116], for the pre-training process, we use the English Wikipedia [150] and the Toronto BookCorpus [170] dataset. These datasets were used in the original BERT pre-training; the latter dataset is not fully available, which results in a small reduction in the baseline accuracies achieved in our experiments from the original BERT results. Following KAISA [116], due to the time-intensive process of hyperparameter tuning for the first phase of pre-training, we report the efectiveness of MKOR in the second phase of pre-training only while using the checkpoints of the first phase generated using the LAMB optimizer. As expected, the computation, communication, and memory complexity of HyLo is high, and the Khatri-Rao-based Interpolative Decomposition (KID) approximation method, the main idea of HyLo, cannot be executed because a single sample cannot fit into the 40GB memory of an A100 GPU. In addition, HyLo doesn’t support gradient accumulation due to its memory complexity, depending on the batch size; in LLMs such as BERT, the batch sizes are as large as 64k.

For the question answering task, we fine-tune the pre-trained BERT checkpoints on the SQuAD v1.1 [123] dataset. Table 3.3 shows the F1 Score achieved using diferent optimizers and compares their convergence rate and speedups.<sup>2</sup> Convergence is defined as the number of iterations it takes the model to reach the same accuracy as the first-order optimizer. The vanilla MKOR and KAISA both converge after 1000 iterations, while the LAMB optimizer requires 1, 563 steps. Considering that each step in MKOR is faster than KAISA, MKOR achieves an end-to-end speedup. MKOR-H will converge in 600 steps, reducing the number of steps in LAMB by 2.6×, while achieving the same accuracy. In addition, it achieves 2.57× speedup over the LAMB optimizer and 1.75× speedup over KAISA. As another second-order baseline, we consider Eva, which converges in 1000 iterations, and MKOR-H achieves 1.69× speedup over it.

For classification tasks, we fine-tune BERT on the GLUE [147] dataset. Table 3.4 compares the results for diferent classification tasks in the GLUE dataset. MKOR with 1500 steps achieves a new state-of-the-art accuracy in GLUE dataset on BERT-Large Uncased, and MKOR and MKOR-H with 600 steps achieve the same average metric as the baseline, while reducing the number of steps by a factor of 2.6×. MKOR and MKOR-H both achieve 2.57× end-to-end speedup. After training KAISA for 1,563 steps, the model does not converge to the baseline average accuracy, while slowing down the convergence by 0.89×. Eva requires 1000 steps to converge to the target average metric, being 1.69× slower than MKOR-H with 600 steps and 1.24% less accurate than MKOR with 1500 steps (it is noteworthy that the accuracy of the model plateaus when using more iterations with Eva).

Table 3.3: BERT-Large Uncased results on SQuAD v1.1 question answering task
<table><tr><td rowspan=1 colspan=1>Metric</td><td rowspan=1 colspan=1>LAMB</td><td rowspan=1 colspan=1>KAISA</td><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>MKOR-H</td><td rowspan=1 colspan=1>Eva</td></tr><tr><td rowspan=1 colspan=1>F1</td><td rowspan=1 colspan=1>90.44</td><td rowspan=1 colspan=1>90.44</td><td rowspan=1 colspan=1>90.50</td><td rowspan=1 colspan=1>90.64</td><td rowspan=1 colspan=1>90.55</td></tr><tr><td rowspan=1 colspan=1># Iterations</td><td rowspan=1 colspan=1>1,563</td><td rowspan=1 colspan=1>1,000</td><td rowspan=1 colspan=1>1,000</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>1,000</td></tr><tr><td rowspan=1 colspan=1>Time (h)</td><td rowspan=1 colspan=1>7.97</td><td rowspan=1 colspan=1>5.71</td><td rowspan=1 colspan=1>5.25</td><td rowspan=1 colspan=1>3.10</td><td rowspan=1 colspan=1>5.24</td></tr><tr><td rowspan=1 colspan=1>Speedup (×)</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.39</td><td rowspan=1 colspan=1>1.51</td><td rowspan=1 colspan=1>2.57</td><td rowspan=1 colspan=1>1.52</td></tr></table>

BERT-Large-Uncased Training Loss

![](images/9483f142c5fe67bab2512e987b56f5c145ae263fdf74ff13a5d09ee98926ec4a.jpg)

![](images/80e15f914beff5ce1e4ddc9398ed0f3505085aff5ea95f679c5f997bd273a4c1.jpg)  
Time (h)  
Figure 3.2: The pre-training loss of BERT-Large-Uncased using diferent optimizers.

Per Figure 3.2, which shows the pre-training error during the training of BERT, MKOR decreases the error in fewer iterations in comparison to KAISA, Eva, and LAMB, leading to faster convergence. From Table 3.3 and Table 3.4, MKOR-H converges in only 600 steps.

ResNet-50 Experiments. We train ResNet-50, a convolutional neural network with more than 25M parameters, on ImageNet, an image classification task with more than 1.2M samples. The same setup is used in [116], and SGD is used as the first-order baseline.

The target accuracy in this experiment is 75.9%. MKOR converges to this target accuracy in 57 epochs, while SGD, the first-order baseline, achieves this accuracy in 88 epochs. MKOR achieves 1.49× speedup over

Table 3.4: BERT-Large Uncased results on the GLUE classification tasks. We report the average of the metrics of diferent GLUE tasks (accuracy, F1 score, etc) for easier comparison.
<table><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>LAMB</td><td rowspan=1 colspan=1>KAISA</td><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>MKOR-H</td><td rowspan=1 colspan=1>Eva</td></tr><tr><td rowspan=1 colspan=1>Iterations</td><td rowspan=1 colspan=1>1,563</td><td rowspan=1 colspan=1>1,563</td><td rowspan=1 colspan=1>1,500</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>1000</td></tr><tr><td rowspan=1 colspan=1>Time (h)</td><td rowspan=1 colspan=1>7.97</td><td rowspan=1 colspan=1>8.93</td><td rowspan=1 colspan=1>7.88</td><td rowspan=1 colspan=1>3.10</td><td rowspan=1 colspan=1>3.10</td><td rowspan=1 colspan=1>5.24</td></tr><tr><td rowspan=1 colspan=1>Speedup (×)</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>1.01</td><td rowspan=1 colspan=1>2.57</td><td rowspan=1 colspan=1>2.57</td><td rowspan=1 colspan=1>1.52</td></tr><tr><td rowspan=1 colspan=1>Average Metric</td><td rowspan=1 colspan=1>0.8023</td><td rowspan=1 colspan=1>0.796</td><td rowspan=1 colspan=1>0.8214</td><td rowspan=1 colspan=1>0.8078</td><td rowspan=1 colspan=1>0.811</td><td rowspan=1 colspan=1>0.809</td></tr></table>

SGD. KAISA, the second-order baseline converges in 54 epochs, but due to its expensive steps, MKOR still converges 1.04× faster than KAISA. We do not compare to HyLo because HyLo is not able to achieve the target accuracy for ResNet (it reaches 75.6% with tuning as reported in [102] and our experiments confirm it). As shown, the efect of complexity reduction and improvement in performance in MKOR is less obvious in ResNet because the model dimension (d) is smaller compared to LLMs such as BERT. Please see Table 3.1 for comparison of complexity between methods.

We could not reproduce the ResNet-50 results of Eva [163] on ImageNet because the hyperparameters are not reported. We tried to tune Eva on multiple settings and none converged to desired accuracy. It is important to note Eva is not comparing results with the most eficient implementation of KFAC. The KFAC version used in Eva is from [93], dated to 2015. A number of followup works, mentioned in Section 3.1, have provided faster implementations of KFAC. We use KAISA [116], the state-of-the-art implementation of KFAC. Also from discussions with KAISA authors and our own experiments the optimal inversion frequency for KFAC is 200. Eva uses an inversion frequency of 50 for KFAC, which makes KFAC slower.

## ResNet-50 Test Accuracy

![](images/84417b553632767f3d7364b229459f4f38a4896a32929ee1438c3a6cfea239d2.jpg)  
(a)

![](images/088efe82300d1852885e5cc8172b7e14e54355d53a2f9f14d523b865cc52faa9.jpg)  
(b)  
Figure 3.3: Test accuracy of ResNet-50 on ImageNet for MKOR, KAISA, and SGD on 64 GPUs.

Inversion Frequency. Due to the low computation complexity of the updates on MKOR, the factor inversion frequency (f) in MKOR is in the range of 10. Figure 3.4-a shows that while the average iteration cost in KAISA is heavily dependent on the inversion frequency, MKOR’s cost is almost independent of the inversion frequency. Also Figure 3.4-b shows that increasing the inversion frequency leads to higher convergence rate. In addition, using stale factors may result in converging to a local minima. Hence, in MKOR we increase the convergence rate by updating the factors more frequently, without afecting the per-iteration cost, leading to end-to-end speedups in training. We use a simple autoencoder [128] on CIFAR-100 [71] in this experiment.

Performance Analysis. We compare the performance of diferent parts of the optimizers to illustrate the bottlenecks and advantages of diferent methods. The training process for an optimizer has three steps: factor computation, precondition, and update weights. Figure 3.5 shows the time spent on each task in diferent optimizers on two models; BERT-Large-Uncased, a transformer-based LLM with large sequence length and ResNet-50, a CNN. Since first-order optimizers such as SGD, ADAM, and LAMB don’t require factorization and preconditioning, their optimization time is only spent in updating the weights. In ResNet-50, since the model size is larger compared to the batch size, the factor computation and inversion is more expensive for KAISA compared to HyLo. This cost is significantly reduced in MKOR.

![](images/804ce84e90440f18126f3e2a2e568517831edc2fbc267ebadfa85ee01b3eca6b.jpg)

![](images/f63b8fd363225e7850fb4efbfdb96839cd021e6f7b60f7bcb76ded6a96185f16.jpg)

(a) - Average Time per Iteration  
![](images/747b510fa50919d45fe3ada6dcf8153ba657cdc9dff632c17d6663202e050f2c.jpg)

![](images/516c17dd931df61713c3f418b33234587a6900c9e43c381e0b547eda38fa7fc3.jpg)  
(b) - Test Loss  
Figure 3.4: The sensitivity of MKOR and KAISA for BERT-Large-Uncased and an Autoencoder model (a) and the efect of inversion frequency on the convergence properties of these models (b).

For BERT-Large-Uncased, because of the large size of the model, the factor inversion time for KAISA is large. Also, due to the large sequence length value in this model, the kernel inversion time for HyLo is comparable to KAISA’s inversion time. But as expected, because of its low computational complexity, the aforementioned cost in our method is much smaller than the total training time, leading to speedups. It is important to note that HyLo diverges in this training process, hence convergence time is not reported for HyLo.

The preconditioning and weight updates for the diferent methods are similar; hence, not much variation is observed.

Memory Overheads. The memory overheads of MKOR in comparison to other optimizers are reported in Table 3.5. It can be observed that all the second-order methods have significant memory overheads compared to the first-order methods, but MKOR’s overhead is up to 1.5× lower than KFAC/KAISA.

Approximation Error Experimental Results. Due to the low-rank properties of the covariance matrices, MKOR utilizes rank-1 approximations of the covariance matrices to accelerate the computations and communication in KFAC-based optimizers. Here, we aim to theoretically and experimentally support this choice.

![](images/8b8c5f249dba15ae6e1642a4d38ab2db2070ee59a7499dbd4fecece93d0f46cd.jpg)  
(a)

![](images/bd479e49b5a55289f209cb994b088a234a2b7c8db5e85325cfa7ae26742ed7af.jpg)  
(b)  
Figure 3.5: Per-step breakdown of diferent optimizers on BERT-Large-Uncased (a) and ResNet-50 (b). The times reported in these graphs reflect only the optimizer computations. The majority of the training time is spent on the model’s forward and backward passes, which are identical across all optimizers and are not included here.

Table 3.5: Per-GPU memory usage (in GB) for MKOR, KFAC/KAISA, LAMB, and SGD on BERT-Large-Uncased pre-training and ResNet-50 training on ImageNet.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>KFAC/KAISA</td><td rowspan=1 colspan=1>LAMB</td><td rowspan=1 colspan=1>SGD</td></tr><tr><td rowspan=1 colspan=1>ResNet-50</td><td rowspan=1 colspan=1>3.88</td><td rowspan=1 colspan=1>5.83</td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1>3.01</td></tr><tr><td rowspan=1 colspan=1>BERT</td><td rowspan=1 colspan=1>23.34</td><td rowspan=1 colspan=1>29.97</td><td rowspan=1 colspan=1>12.80</td><td rowspan=1 colspan=1>一</td></tr></table>

As shown in Figure 3.6, our experiments show that the covariance matrices can be approximated with rank-1 matrices with low error and higher rank approximations are unnecessary in practice. Figure 3.6 shows the error distribution of the optimal rank-1 approximation methods of the covariance matrices in ResNet-50 and BERT-Large-Uncased pre-training. Our extensive tests on well-known benchmarks show this property holds for all models and we have not come across a benchmark that does not have low-rank covariance matrices.

Approximation Error Analysis and Extension to Higher Ranks. Small batch sizes and over parameterization of networks will lead to low-rank covariance matrices in DNNs. Let’s consider the covariance matrix $C = X X ^ { T }$ , where $C \in R ^ { d \times d }$ is the covariance matrix and $X \in R ^ { d \times b }$ is a matrix in which each column corresponds to a single sample and d and b are the sample dimension and the per-GPU batch size respectively. Rank of the covariance matrix is $m i n ( b , d )$ . If the per-GPU batch sizes are small, the covariance matrices in each GPU will be low-rank. Rank-1 approximation methods can work well in these scenarios. If the batch sizes in each GPU are large, we observe that the covariance matrices will stay low-rank. The underlying reason for this observation is that current neural networks are over-parameterized, and as a result, diferent features in the covariance matrices of the activations and the output gradients won’t be linearly independent, resulting in low-rank covariance matrices.

Extending MKOR to Higher Ranks: Furthermore, one can extend MKOR to use higher-rank covariance matrices. Let’s assume that $\begin{array} { r } { C = \sum _ { i = 1 } ^ { r } c _ { i } c _ { i } ^ { T } } \end{array}$ where r is the rank of the covariance matrix C. We can apply SMW identity to compute $C _ { 1 } ^ { n e w } = ( C ^ { o l d } + c _ { 1 } c _ { 1 } ^ { T } ) ^ { - 1 }$ with $O ( d ^ { 2 } )$ computational complexity. Then we can compute $C _ { 2 } ^ { n e w } = ( C _ { 1 } ^ { n e w } + c _ { 2 } c _ { 2 } ^ { T } ) ^ { - 1 }$ using SMW identity with $O ( d ^ { 2 } )$ computational complexity. We can continue the same pattern by computing $C _ { i } ^ { n e w } = ( C _ { i - 1 } ^ { n e w } + c _ { i } c _ { i } ^ { T } ) ^ { - 1 }$ . The total computation complexity of this process will be $O ( r d ^ { 2 } )$ . We should add this cost to the cost of computing the low-rank approximation of C which requires an SVD. Using SVD kills the main advantage of using low-rank computations, since the computational complexity of applying SVD is the same as inverting the factors directly. We could not find any cheaper way to compute low-rank approximations of the covariance matrices, except for the rank-1 approximation used in this chapter.

BERT Activation Error Distribution  
![](images/cd6ae8c6de9e797828c1246ea42b1f91948a11b683c7d3097b98515220c7223f.jpg)  
(a)

BERT Input Gradient Error Distribution  
![](images/d42876fada3d973431c23fd670ebb7b01bbe91f5a4de76a1d5c8790eec65692b.jpg)  
(b)

ResNet-50 Activation Error Distribution  
![](images/5bb85323b99355b9bcee2313c55546db083af4e077b840ff619fe196a48d9f84.jpg)  
(c)

ResNet-50 Input Gradient Error Distribution  
![](images/18878d71587b1ad56afca7bae9a0911ce70a55b0349883879d92ca6f41a7b3cf.jpg)  
(d)  
Figure 3.6: Rank-1 error for activation and input gradient covariance matrices for BERT-Large-Uncased pretraining (a, b) and ResNet-50 on ImageNet (c, d).

## 3.5 Conclusion

In this chapter, we presented MKOR, a scalable second-order optimizer that efectively executes the second strategy of the Compression Trinity: accelerating pretraining by reducing the number of required iterations. By applying the Trinity’s pillars directly to the optimizer’s internal mechanics, MKOR overcomes the historical bottlenecks of second-order methods. We leveraged low-rank approximations via rank-1 updates to reduce inversion complexity from $\mathcal { O } ( d ^ { 3 } )$ to $O ( d ^ { 2 } )$ , and employed stability mechanisms to enable lower-precision communication, reducing overheads to $\mathcal O ( d )$ . Our experiments confirm that MKOR significantly outperforms state-of-the-art first- and second-order optimizers, delivering up to 2.57× faster training for large language models.

However, accelerating convergence is only half of the pretraining equation. While MKOR reduces the total number of steps, the computational cost per step remains dominated by the dense matrix multiplications inherent to the Transformer architecture. To unlock the full potential of the Compression Trinity during the pretraining phase, we must also address Strategy 1: reducing the fundamental FLOPs and memory trafic of the linear layers themselves. In the next chapter, we introduce ${ \mathrm { S L o P E } } ,$ which extends the Trinity from the optimizer to the model weights, jointly applying sparsity and lazy low-rank adapters to accelerate the training dynamics without sacrificing model quality.

# SLOPE: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs

Publication and Contributions. The content of this chapter is based on the paper “SLOPE: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs” [101], published at the Thirteenth International Conference on Learning Representations (ICLR) 2025. This work was conducted in collaboration with Amir Yazdanbakhsh, Zhao Zhang, and Maryam Mehri Dehnavi. Mohammad Mozafari was the lead contributor, responsible for the algorithm design, implementation, and experimental evaluation. Amir Yazdanbakhsh and Maryam Mehri Dehnavi supervised the project and contributed to the writing and revision of the manuscript. Zhao Zhang provided additional supervisory guidance.

## 4.1 Introduction

Following our exploration of optimizer-level acceleration in Chapter 3, we now turn to the first strategy of the Compression Trinity: accelerating the computational cost of each individual training iteration. Large Language Models (LLMs) require massive resources for their life-cycle stages, specifically pretraining [119] on high-quality text [40, 44] and fine-tuning on downstream tasks [147, 123]. These phases are dominated by three intensive matrix multiplications per layer: the forward pass, the backward pass for input gradients, and the backward pass for weight gradients. To execute Strategy 1 (Accelerating Per-Iteration Computation), we must reduce the FLOPs and memory trafic for all three operations. The first pillar of our Trinity, sparsity, ofers a path forward [58]. While unstructured sparsity lacks hardware support [151] and rigid block-structured sparsity damages accuracy [67, 84, 25], semi-structured N:M sparsity (e.g., 2:4, where 2 out of 4 consecutive elements are set to zero) strikes a balance. It is flexible enough to preserve model quality while being structured enough for acceleration via NVIDIA’s Sparse Tensor Cores [108], with algorithms rapidly evolving for these patterns [69, 89, 10].

However, applying N:M sparsity to the training phase faces a critical ”transposability” bottleneck. While N:M sparsity successfully accelerates the forward pass, it fails to accelerate the backward pass because the row-wise N:M structure is destroyed when the weight matrix is transposed. Prior attempts to address this have focused on finding ”transposable masks” that maintain structure in both orientations [63, 166, 62]. Unfortunately, these methods often require expensive search algorithms or enforce rigid constraints that signif icantly reduce model accuracy. Paradoxically, the overhead of these complex mask searches can result in severe training slow-downs, up to 8.4× [62]. Alternative approaches that change the sparsity mask dynamically [29, 168, 69, 89] also introduce computational overheads and waste resources training weights that are eventually pruned.

To strengthen the sparsity pillar for pretraining, we propose a novel double-pruned backward pass formulation with theoretical convergence guarantees. Instead of enforcing the restrictive condition that a mask must be inherently transposable, our approach allows the forward pass to use a standard N:M mask. In the backward pass, we transpose the weight matrix first and then impose a new N:M sparsity pattern. This formulation allows the weight matrices to exhibit a much wider range of sparsity patterns compared to rigid transposable masks, leading to significantly improved accuracy while enabling acceleration in both directions.

While resolving the compute bottleneck, aggressive sparsity can still lead to an accuracy gap compared to dense models. To bridge this gap without sacrificing eficiency, we integrate the third pillar of the Trinity: Low-Rank Approximations. Previous methods often resort to dense fine-tuning to recover accuracy [142, 62], but this converts the model back to a dense state, negating all memory and compute savings during inference. The intuition behind this integration lies in the observation that, while the parameter space of LLMs is vast, learning efectively occurs on a manifold of much lower intrinsic dimension [77, 61]. This suggests that full-rank updates are not strictly necessary for recovering the expressivity lost to pruning.

However, combining these pillars is non-trivial. Theoretically, a low-rank adapter LR is a dense matrix; adding it to a sparse weight matrix W $( W ^ { \prime } = W + L R )$ would cause ”fill-in,” where strictly zero elements become non-zero, efectively destroying the sparsity pattern and its associated hardware benefits. To harness the power of both without this collision, we propose Lazy Low-Rank Adapters. We treat the sparse weights and dense adapters as parallel computational paths rather than a merged tensor. Furthermore, unlike standard adapters, ours are ”lazy” because they are introduced only during the final 1% of pretraining iterations. Our experiments show that these adapters converge noticeably faster compared to the original model parameters at the same parameter count. This approach improves the accuracy of the models while ensuring the base model remains sparse and eficient for deployment.

We present SLOPE, a Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining method for LLMs that jointly leverages these pillars. Key contributions of SLOPE are:

• Double-Pruned backward pass → We propose to transpose an already sparsified N:M weight matrix (forward pass) before imposing another round of N:M sparsity (backward pass). This improves model quality and eliminates the overhead of searching for transposable masks.

• Lazy Low-Rank adapters → We utilize the low-rank pillar to recover accuracy by introducing additional parameters with minimal compute and memory overheads, strictly for the last 1% of pretraining iterations (see Figure 4.1).

• Optimized CUDA kernels → We jointly optimize NVIDIA 2:4 sparse kernels and low-rank calls through eficient tiling and scheduling. Our highly-optimized CUDA kernels result in 1.25× end-to-end training speedup and 1.54× inference speedup on LLMs with billions of parameters, while reducing training and inference memory footprints by up to 0.63× and 0.61×, respectively.

![](images/7ee8593173b900ebf9b9eb31eed283014f0f9b41ae092776352c88b351cea84e.jpg)  
Figure 4.1: The sparse training pipeline in SLOPE. Here, X, Y, and W denote the input, output, and the weight tensors for a specific layer, respectively. ∇ L represents the gradient of the loss function. L and R are the low-rank terms that are introduced only in the final 1% iterations. Superscript R shows row-wise pruning using N:M scheme and R, C shows both column and row-wise N:M sparsification, leading to extra imposed zeros. Blue elements represent non-zero values, while white elements represent pruned values, and red elements indicate additional zeros introduced during the backward pass.

## 4.2 Additional Related Work

Model pruning. Pruning the models has been one of the most efective methods to reduce the complexity of LLMs [58]. One can pretrain the LLMs sparsely [34] or the pruning can happen after a dense pretraining [55, 75], possibly followed by a fine-tuning stage to recover part of the lost accuracy [39, 51]. Pruning the models after pretraining can be costly [127, 52] and typically fails to maintain their accuracy [36, 136]. While the sparse pretraining methods improve the accuracy of the model, they either use unstructured sparsity patterns that cannot be accelerated with the current hardware [142] or have significant overheads when searching for and applying their structured sparse masks [63, 166, 137].

Low-rank adapters. Low-rank adapters have emerged as a promising method to reduce the fine-tuning costs associated with pre-trained LLMs and enable more eficient task switching [61]. Diferent quantization and initialization schemes have been proposed to reduce their overheads in LLM fine-tuning [28, 48]. Adding low-rank factors to sparse matrices is a low-weight mechanism widely used to improve the accuracy of approx imations of dense matrices [12]. In machine learning, the sparse plus low-rank approximations are limited to attention heads [103, 17] and pruning after pretraining [104, 80], and the sparse plus low-rank pretraining has not been investigated. Additionally, the sparse plus low-rank fine-tuning work does not provide acceleration in both forward and backward pass of the fine-tuning process. Furthermore, the low-rank adapters in these works are added at the beginning of the fine-tuning process, adding extra overheads to the fine-tuning process.

## 4.3 Sparse Plus Low-rank Pretraining of LLMs

Equation 4.1, Equation 4.2, and Equation 4.3 depict the formulas for the forward and backward pass of the i-th linear layer in a neural network. Here, the weight tensor is denoted as $\mathcal { W } _ { i } \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ and the input tensor is denoted as $\boldsymbol { \mathcal { X } } _ { i } \in \mathbb { R } ^ { b \times d _ { i n } }$ . The forward pass generates an output tensor represented as $\mathcal { V } _ { i } \in \mathbb { R } ^ { b \times d _ { o u t } }$ In all equations, $d _ { i n }$ and $d _ { o u t }$ refer to the input and output dimensions of the respective layer and b refers to the batch size.

$$
\mathrm { F W D } \mapsto \mathscr { V } _ { i } = \mathscr { X } _ { i } \mathscr { W } _ { i } ^ { T }\tag{4.1}
$$

$$
\mathbf { B W D } - 1 \mapsto \nabla _ { W _ { i } } \mathcal { L } = \nabla _ { Y _ { i } } \mathcal { L } ^ { T } \mathcal { X } _ { i }\tag{4.2}
$$

$$
\mathbf { B W D } - 2 \mapsto \nabla _ { X _ { i } } \mathcal { L } = \nabla _ { Y _ { i } } \mathcal { L } \mathcal { W } _ { i }\tag{4.3}
$$

The dimension along which N:M pruning occurs corresponds to the reduction dimension in Matrix-Matrix multiplication. Without this restriction, the sparse Matrix-Matrix operation can not be accelerated on GPU [111]. With this restriction in mind, to leverage weight sparsity in forward and backward pass, one needs to prune elements along the columns of $\mathcal { W } _ { i } ^ { T }$ in Equation 4.1 (FWD) and $\mathcal { W } _ { i }$ in Equation 4.3. To satisfy this requirement, it is necessary to prune elements of the weight tensor $\mathcal { W } _ { i }$ along both row and column dimensions.

## 4.3.1 Double-pruned Backward Pass

Various approaches can be used to exploit N:M sparsity during both the forward and backward passes. For example, one may prune the activation tensor $\mathcal { X } _ { i }$ in FWD along the row dimension and $\mathcal { W } _ { i }$ in BWD-2 along the column dimension. Although diverse combinations exist for pruning, our focus in this study is primarily on the sparsification of weight tensors for two reasons: (a) the sparsification of weight tensors directly impacts the resource required for model storage and serving, and (b) our initial findings indicate that pruning weight tensors during both forward and backward passes has a comparatively lesser adverse impact on the overall end-to-end model quality. More details on our experiments can be found in Section B.6. As such, we present a double-pruned backward pass formulation that can productively accelerate FWD and BWD-2 computations.

In addition, we prove that such materialization of pruned weight tensors, despite being lossy<sup>1</sup>, exhibits convergence properties. For the rest of this chapter, we represent the weight tensor subjected to row-wise pruning as $\mathcal { W } _ { i } ^ { R }$ , while the concurrent row-wise and column-wise pruning (double-pruned) is presented as $\mathcal { W } _ { i } ^ { R , C }$ . We rewrite the training equations to accommodate these modifications, with proposed changes highlighted in blue:

$$
\mathrm { F W D } \mapsto \mathcal { V } _ { i } = \chi _ { i } \mathcal { W } _ { i } ^ { R ^ { T } }\tag{4.4}
$$

$$
\mathbf { B W D } - 1 \mapsto \nabla _ { W _ { i } } \mathcal { L } = \nabla _ { Y _ { i } } \mathcal { L } ^ { T } \mathcal { X } _ { i }\tag{4.5}
$$

$$
\mathbf { B W D } - 2 \mapsto \nabla _ { X _ { i } } \mathcal { L } = \nabla _ { Y _ { i } } \mathcal { L } \mathcal { W } _ { i } ^ { R , C }\tag{4.6}
$$

Using this formulation for training, we can accelerate both forward and backward passes owing to the existence of N:M sparsity along both dimensions of weight tensors (see Figure 4.1).

Memory Footprint Analysis. Inducing N:M structured sparsity not only improves computational eficiency of GEMM operations but also reduces the memory footprint for storing sparse tensors. It is noteworthy, however, that the storage of auxiliary meta-data becomes necessary, containing information about the locations of nonzero elements in a supporting matrix. Equation 4.7 delineates the requisite number of bits for storing the indices in the N:M sparsity format, where ⌈.⌉ denotes the ceiling function. We present the detailed results on the memory footprint reduction in Section 4.4.

$$
n _ { i n d e x } ^ { N : M } = \left\lceil l o g \left( \binom { M } { N } \right) \right\rceil\tag{4.7}
$$

Convergence Analysis. Theorem 4.3.1 (proof in Section B.14) shows the additional sparsity resulting from double pruning to an initially row-wise N:M pruned matrix. Following this lemma, we quantify the increased sparsity induced by double pruning with 1:2, 2:4, and 2:8 sparsity patterns as 12.5%, 9.375%, and 3.39%, respectively. This observation underscores that as the value of M in N:M increases, the surplus of zero elements in a double-pruned matrix diminishes. This reduction in zero elements consequently implies a decrease in computational errors, enhancing the robustness of the computations. We expound further insights into thi phenomenon in Section B.5.

Lemma 4.3.1. Consider a randomly initialized matrix A. Following our notations, we denote the row-wise pruned version of A by $A ^ { R }$ and the joint column- and row-wise pruned version of A by $A ^ { R , C }$ . We use $D ( . )$ to present the density ratio ofa matrix. Equation 4.8 shows the additional zero elements in matrix A that are introduced by double-pruning, where $\begin{array} { r } { s = \frac { N } { M } } \end{array}$

$$
D ( A ^ { R } ) - D ( A ^ { R , C } ) = \sum _ { j = N + 1 } ^ { M } { \binom { M } { j } } s ^ { j } ( 1 - s ) ^ { M - j } \frac { j - N } { M }\tag{4.8}
$$

Theorem 4.3.2 states that the dynamic alteration of the column-wise mask in Equation 4.5 during each training iteration does not exert a detrimental impact on the convergence of the optimizer. This phenomenon can be attributed to the equivalence between the left-hand side of Equation 4.9, which corresponds to Equation 4.3 [BWD-2], and the averaging efect achieved through multiple training iterations of backpropagation with distinct sparsity masks. However, for arbitrary values of N and M, Equation 4.4 and Equation 4.5 can be used in the training with convergence guarantee (proof in Section B.14). The sparsity mask is chosen randomly at initialization, i.e. all the weights have the same probability of being zero or non-zero. This is because at initialization the location of weights with larger magnitude is arbitrary. After choosing the sparsity mask at initialization, we keep the mask fixed throughout the entire training process. This policy ensures that each element in the weight has the same probability of being non-zero at initialization and satisfies the random mask assumption in Theorem 4.3.1.

Theorem 4.3.2. Assuming a loss function $\mathcal { L } ( \mathcal { W } _ { \rangle } , \mathcal { X } _ { \rangle } )$ for a random sample $X _ { i } ,$ and considering a random mask $M _ { i } ,$ Equation 4.9 holds, where $E [ . ]$ is the expectation operator and ⊙ is the element-wise multiplication.

$$
E _ { X _ { i } } [ \nabla _ { X _ { i } } \mathcal { L } ( W _ { i } , X _ { i } ) ] = \frac { M } { N } E _ { M _ { i } } [ E _ { X _ { i } } [ \nabla _ { Y _ { i } } \mathcal { L } ( W _ { i } , X _ { i } ) ( M \odot W _ { i } ) ] ]\tag{4.9}
$$

## 4.3.2 Lazy Low-rank Adapters

Pruning weight tensors in FWD and BWD-2 computations is desirable for computational eficiency but may have detrimental impact on quality. To mitigate this adverse impact on model quality, we augment the doublypruned weight matrix with a low-rank matrix. The decomposition of the doubly-pruned weight matrix, combined with the low-rank matrix, maintains the computational eficiency of sparse matrix-matrix multiplication during forward and backward passes. Simultaneously, this approach holds promise in alleviating the adverse efects of double pruning on overall model quality.

Considering the dense weight matrix, denoted by $W _ { d e n s e } ~ \in ~ \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ , Equation 4.10 illustrates the proposed matrix decomposition. In this expression, $W _ { s p a r s e } \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ signifies a doubly-pruned matrix and $\mathcal { L } \in \mathbb { R } ^ { d _ { o u t } \times r }$ and $\mathcal { R } \in \mathbb { R } ^ { r \times d _ { i n } }$ are components of the low-rank approximation. The variable r denotes the rank of this low-rank approximation and r functions as a hyperparameter that controls the trade-ofs between memory footprint, computational eficiency, and model quality.

$$
\mathcal { W } _ { d e n s e } = \mathcal { W } _ { s p a r s e } + \mathcal { L } \mathcal { R }\tag{4.10}
$$

The matrix decomposition of doubly-pruned matrix combined with a low-rank matrix approximation reduces the memory footprint of W from $d _ { i n } d _ { o u t }$ to $\begin{array} { r } { d _ { i n } d _ { o u t } \frac { N } { M } + ( d _ { i n } + d _ { o u t } ) r } \end{array}$ , where $r < < m i n ( d _ { i n } , d _ { o u t } )$ The computational complexity of dense Matrix-Matrix multiplication, however, changes from $b d _ { i n } d _ { o u t }$ to $\begin{array} { r } { b d _ { i n } d _ { o u t } \frac { N } { M } + b ( d _ { i n } + d _ { o u t } ) r } \end{array}$ . Given the substantially smaller value of r in comparison to $b , d _ { i n }$ , and $d _ { o u t } ,$ our formulation efectively reduces both memory footprint and computational complexity of Matrix-Matrix multiplication by a factor of $\textstyle { \frac { M } { N } } \times$

We empirically show that the convergence rate of low-rank adapters surpasses that of sparse weights. We attribute this behavior to the notably lower parameter counts inherent in low-rank adapters. Leveraging this observation, we incorporate low-rank adapters exclusively during the final 1% of the training iterations. This confined usage of low-rank adapters results in additional reduction of training cost, specifically in terms of total number of operations. We term the proposed usage of low-rank adapters in the final steps of the training as lazy low-rank adapters (see Figure 4.1).

## 4.3.3 Sparse Kernels

cuSPARSELt is a CUDA library designed explicitly for sparse Matrix-Matrix multiplication, where one operand undergoes pruning with the 2:4 sparsity pattern. However, this library does not ofer APIs for other algebraic routines such as addition and assignment for sparse tensors. We now delve into the details of diferent kernels for training and overview our implementation methodology.

Algorithm 2 shows the training process of a single linear layer taken from an attention-based model. We assume the use of weight decay in the optimizers, and subsequently design the requisite sparse APIs to facilitate the optimizer operations. The training starts with matrix initialization (line 2) and setting up sparse formats to store weight tensors and their corresponding transpose (line 3 and 4). Then, for every mini-batch in the training set, we compute the forward pass following Equation 4.4 (line 8). As part of the backward pass, the derivative of the loss function with respect to the output activation is computed (line 10). Subsequently, the gradients of the loss function with respect to the input activation (line 11) and the weight tensor (line 12) are computed using Equation 4.5 and Equation 4.2, respectively. In order to circumvent the necessity of updating weights with zero values and mitigate the associated memory footprint overhead, we employ a strategy wherein we mask the gradients for pruned weights. The computed values are stored in a sparse format (line 13). Next, in order to implement weight decay in the optimizer and mitigate the impact of gradient scaling, we compute the value of $\scriptstyle { \frac { 1 } { \gamma } } \nabla _ { W } { \mathcal { L } } + \alpha W$ (line 15). Here, α is the weight decay applied in the optimizer, while γ denotes the gradient scaling factor for numerical stability during the half-precision backward pass. The updated values for the weight tensor are calculated according to the optimizer update rule (line 16). Finally, the value of weight tensor and its transpose are updated directly in a sparse format (line 17 and line 18). More details about the implementation of the custom kernels used in Algorithm 2 can be found in Section B.7.

Algorithm 2 Accelerated Sparse Pretraining Algorithm for a Linear Layer   
1: Input: Weight W, training set D, weight decay α, gradient scaling factor γ.   
2: Output: Updated weight Wnew.   
3: backend.init() ▷ Initialize backend   
4: WSparseTranspose ←backend.setup(W<sup>T</sup>) ▷ Setup transpose for sparse matrix multiplication   
5: WSparse ←backend.setup(W) ▷ Setup sparse weight matrix   
6: sparseMask ←(WSparse ̸= 0) ▷ Element-wise mask for sparsity   
7: for each training example (X, Y<sup>ˆ</sup>) ∈ D do   
8: Forward Pass:   
9: Y ←backend.spmm(X, WSparseTranspose)   
10: Backward Pass:   
11: ∇<sub>Y</sub>L ← gradOutput ▷ Gradient w.r.t. output   
12: gradInput ←backend.spmm(gradOutput, WSparse)   
13: gradWeight ←backend.matmul(gradOutput<sup>T</sup>, X)   
14: gradWeightSparse ←backend.pruneAndCompress(gradWeight, sparseMask)   
15: Optimizer with Weight Decay:   
16: g ←backend.sparseAdd(gradWeightSparse, WSparse, <sup>1</sup><sub>γ</sub> , α)   
17: Wnew ←optimizer.updateWeight(g)   
18: backend.updateSparseMatrix(WSparse, Wnew)   
19: backend.updateSparseMatrix(WSparseTranspose, Wnew<sup>T</sup>)   
20: end for   
21: Return: Wnew.

## 4.3.4 SLOPE Runtime Optimization

While SLOPE improves the training and inference of LLMs by introducing sparse weights and low-rank adapters, a naïve implementation can hinder its full performance improvement. Specifically, cuSPARSELt [110] SpMM kernels exhibit sensitivity to input and weight tensor shapes, and introducing low-rank adapter at inference can increase the number of calls during the forward pass of each linear layer. This section covers our approach to optimize SLOPE’s implementation and further improve model performance.

Eficient tiling of upsample tensors. Figure 4.3-(a) showcases the speedup achieved by the cuSPARSELt backend across a range of tensor shapes commonly used in LLMs. While the speedup of SpMM in downsample tensors increases gradually as their sizes increase, the speedup of upsample tensors drops of at around hidden dimension = 4000. To overcome this limitation, we tile the upsample tensor into multiple smaller matrices of equal size, each of which benefits from improved speedup when multiplied by the input using 2:4 sparsity. By tuning the size of the tiles, we discovered that the best performance can be achieved by using square tiles. The results of these multiplications are then concatenated. This optimization, as detailed in Section 4.4.3, leads to a 12% improvement in inference speed and a 4% increase in training speed with SLOPE.

Eficient kernel for combined SpMM+low-rank adapters. A straightforward implementation of low-rank adapters requires four kernel calls: one for sparse matrix multiplication, two for low-rank computations, and one for adding the results. In addition, our experiments demonstrate that multiplying matrices with low-rank adapters does not scale proportionally with the adapter’s rank, leading to significant overheads due to their low arithmetic intensity (see Section 4.4.3). To address this, we introduce two optimizations: (1) concatenating the downsample tensor to the sparse weight tensor, reducing kernel calls and increasing arithmetic intensity as in Equation 4.11-left, and (2) leveraging a cuBLAS fused matrix multiplication and addition kernel, minimizing cache access and kernel calls as in Equation 4.11-right. As demonstrated in Section 4.4.3, these optimizations collectively contribute to a speedup improvement of up to 6% in the end-to-end inference speed.

$$
[ \mathcal V _ { 1 } | \mathcal V _ { 2 } ] = \mathcal X [ \mathcal W ^ { T } | \mathcal L ] ; \qquad \mathcal V = \mathcal V _ { 2 } \mathcal R + \mathcal V _ { 1 }\tag{4.11}
$$

## 4.4 Experimental Results

This section evaluates the eficacy of SLOPE in accelerating the pretraining while achieving memory savings. Due to the substantial computational resources required for LLM pretraining, our accuracy evaluation is primarily focused on smaller-scale LLMs up to 774M parameters. However, the speedup and memory reduction results extend to a wider range of models, from 2.6B up to 66B parameters.

Experiment Setup. Our experiments were conducted on the Narval and Mist clusters at Compute Canada [22] and the Lonestar 6 cluster at the Texas Advanced Computing Center [141]. Each Narval node is equipped with four Nvidia A100 GPUs (40GB), Mist nodes feature four Nvidia V100 GPUs (32GB), and Lonestar 6 node have three Nvidia A100 GPUs (40GB). For accuracy experiments, we emulated 2:4 and N:M sparsity using custom-designed, low-overhead CUDA kernels to prune weights in both the forward and backward passes, utilizing a mixture of available resources across clusters since model accuracy is not hardware-dependent. Speedup and memory saving experiments were conducted on a single A100 GPU in the Narval cluster over 1000 iterations, reporting the median to mitigate outlier efects; memory reduction experiments were run five times with the median reported. We employed the default hyperparameters from the NVIDIA BERT codebase [112] and the FlashAttention GPT codebase [26, 24]. Training BERT-Large-Uncased required approximately 32 hours on 64 A100-64GB GPUs, while pretraining GPT2-Small/Large took 32 and 111 hours on 64 V100-32GB GPUs, respectively.

## 4.4.1 End-to-end Speedup and Memory Saving: Pretraining and Inference

We evaluate the speedup and memory reduction by SLOPE during pretraining and inference across LLMs with diferent model parameter sizes. To demonstrate the scalability and eficiency of our method, we conducted extensive benchmarking on OPT (2.6 B to 66 B), LLaMA-3-8B and Mistral-v0.3-7B models. In all the experiments, we have enabled FlashAttention-2 [24] ( Section B.9 presents detailed ablation study on the impact of FlashAttention). To mitigate the impact of outliers, we conducted 1,000 iterations for each speedup experi ment and reported the median value. For the memory reduction experiments, we performed five independent runs and similarly reported the median outcome. These methodologies were chosen to provide a more reliable measure of central tendency in our results <sup>2</sup>.

We compared our method against dense pretraining and inference directly in PyTorch, which uses eficient cuBLAS backend. As the sparse pretraining benchmark, we compare our work against Fully Sparse Training (FST) [62], the state-of-the-art 2:4 pretraining method and the only semi-structured sparse pretraining work that provides end-to-end speedups. Note that methods targeting LLM pretraining with N:M sparsity often sufer from ineficiency due to mask search overheads and/or compression setup. Section B.4 and Section B.2 detail the profiling in Bi-Mask [166] and FST [62], which similarly use N:M sparsity on both forward and backward passes.

Table 4.1: Comparative analysis of end-to-end pretraining and inference speedup (×) comparison between SLOPE and the latest work (FST) on accelerating pretraining with 2:4 sparsity (ICML 2024) [62]. The baseline is dense PyTorch implementation of the models with CUBLAS backend. Note that the lack of inference speedup in FST is because of the final dense pretraining during the final iterations, resulting in a dense model for inference. E-SR-STE stands for Extended SR-STE.
<table><tr><td rowspan="2">MODEL</td><td rowspan="2">METHOD</td><td rowspan="2">TRAINING No ADAPTER (r = 0)</td><td colspan="3">INFERENCE</td></tr><tr><td>No ADAPTER (r = 0)</td><td>1.56% ADAPTER</td><td>6.25% ADAPTER</td></tr><tr><td rowspan="2">OPT-66B</td><td>SLoPE</td><td>1.20</td><td>1.46</td><td>1.43</td><td>1.40</td></tr><tr><td>FST</td><td>1.06</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-30B</td><td>SLoPE</td><td>1.22</td><td>1.53</td><td>1.53</td><td>1.50</td></tr><tr><td>FST</td><td>1.07</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-13B</td><td>SLoPE</td><td>1.25</td><td>1.54</td><td>1.39</td><td>1.36</td></tr><tr><td>FST</td><td>1.10</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-6.6B</td><td>SLoPE</td><td>1.21</td><td>1.46</td><td>1.46</td><td>1.43</td></tr><tr><td>FST</td><td>1.11</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-2.6B</td><td>SLoPE</td><td>1.13</td><td>1.31</td><td>1.25</td><td>1.18</td></tr><tr><td>FST</td><td>1.09</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">LLAMA-3-8B</td><td>SLoPE</td><td>1.16</td><td>1.35</td><td>1.33</td><td>1.32</td></tr><tr><td>FST</td><td>1.09</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">MISTRAL-v0.3-7B</td><td>SLoPE</td><td>1.15</td><td>1.34</td><td>1.32</td><td>1.31</td></tr><tr><td>FST</td><td>1.07</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

Notably, our approach, SLOPE, diverges significantly from recent work Fully Sparse Training (FST) [62] in three key aspects. Firstly, we comprehensively prune all weights in the model, encompassing both MLP and Self-Attention modules, whereas FST only prunes weights in the MLP modules. Secondly, FST employs dynamic transposable weights, which introduce additional computation and memory overhead during training. Thirdly, FST necessitates dense fine-tuning (∼17% of pretraining), thereby negating their speedup advantages during inference. In contrast, our approach achieves eficient and accurate large language models during both training and inference without such limitations.

SLOPE Speedup for Pretraining and Inference. Table 4.1 summarizes the speedups achieved by our method during both training and inference. Since over 99% of training occurs without low-rank adapters, the training speedup is largely independent of the adapter rank. Conversely, inference speedup is directly influenced by the adapter rank. Given the varying hidden dimensions across diferent model sizes, we report the inference speedup for various adapter rank ratios: $\frac { a d a p t e r - r a n k } { h i d d e n - d i m e n s i o n }$

Figure 4.3-(a) illustrates that cuSPARSELt achieves higher speedups for large matrices until it reaches its maximum performance capacity (2×). A similar trend is observed in the pretraining and inference speedups of the models. For small matrices used in low-rank adapters, the lower arithmetic intensity of low-rank adapter multiplication results in higher overhead relative to sparse multiplication. This is because low arithmetic intensity limits the full utilization of GPU resources, leading to ineficiencies.

SLOPE Memory Reduction in Pretraining and Inference. For training, the memory consumption of a dense model includes weights, gradients, and optimizer states, amounting to $4 \times 1 6$ bits for weights, $4 \times 1 6$ bits for gradients, and $2 \times 4 \times 3 2$ bits for optimizer states. The sparse model, however, stores non-zero weights and indices twice (for both weights and transposed weights), along with a binary mask, gradients, and reduced optimizer states. This adds up to $2 \times ( 1 6 + 3 )$ bits (weights and transposed weights), 4 × 8 bits (binary mask), $2 \times 1 6$ bits (gradients), and $2 \times 2 \times 3 2$ bits (optimizer states). Consequently, the memory footprint during training is reduced by 68%. For inference, a dense model requires storing weights with a total memory cost of

Table 4.2: Comparative analysis of end-to-end memory reductions (×) during training and inference between SLOPE and the latest work (FST) on accelerating pretraining with 2:4 sparsity (ICML 2024) [62]. Values greater than 1.00× show memory overhead.
<table><tr><td>MODEL</td><td>METHOD</td><td>TRAINING No ADAPTER (r = 0)</td><td>No ADAPTER (r = 0)</td><td>INFERENCE 1.56% ADAPTER</td><td>6.25% ADAPTER</td></tr><tr><td rowspan="2">OPT-66B</td><td>SLoPE</td><td>0.67</td><td>0.63</td><td>0.65</td><td>0.70</td></tr><tr><td>FST</td><td>1.27</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-30B</td><td>SLoPE</td><td>0.67</td><td>0.61</td><td>0.63</td><td>0.69</td></tr><tr><td>FST</td><td>1.17</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-13B</td><td>SLoPE</td><td>0.68</td><td>0.51</td><td>0.62</td><td>0.68</td></tr><tr><td>FST</td><td>1.16</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-6.6B</td><td>SLoPE</td><td>0.68</td><td>0.60</td><td>0.62</td><td>0.68</td></tr><tr><td>FST</td><td>1.19</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">OPT-2.6B</td><td>SLoPE</td><td>0.67</td><td>0.62</td><td>0.64</td><td>0.70</td></tr><tr><td>FST</td><td>1.18</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">LLAMA-3-8B</td><td>SLoPE</td><td>0.63</td><td>0.66</td><td>0.69</td><td>0.71</td></tr><tr><td>FST</td><td>1.17</td><td>1.00</td><td>1.00</td><td>1.00</td></tr><tr><td rowspan="2">MISTRAL-v0.3-7B</td><td>SLoPE</td><td>0.68</td><td>0.66</td><td>0.69</td><td>0.65</td></tr><tr><td>FST</td><td>1.15</td><td>1.00</td><td>1.00</td><td>1.00</td></tr></table>

4 × 16 bits. In contrast, our sparse model optimizes memory usage by storing only the non-zero weights and their indices, resulting in $2 \times 1 6$ bits for non-zeros and three bits for indices (see Equation 4.7). This leads to a 54% reduction in memory usage during inference.

Table 4.2 presents the memory reduction for diferent low-rank adapter ranks and OPT, LLaMA-2, and Mistral model variants. The memory reduction is slightly less than the theoretical expectation, primarily because of additional memory usage from other model components, such as layer norms, and dense model parameters.

## 4.4.2 Pretraining Accuracy Results

To assess the impact of SLOPE on model accuracy, we conducted pretraining experiments across various models and datasets. In all experiments, the classification heads and the first linear layer following the input are dense.

GPT2 (Small/Large). We pretrained both the small (117 M parameters) and large (774 M parameters) variants of GPT2 [120] on the OpenWebText dataset [44]. For a fair comparison, we evaluate the models on MMLU [57], Arc Challenge [21], and OpenBookQA [97] zero-shot tasks implemented in Language Model Evaluation Harness [41]. Additionally, we evaluate the validation perplexity of the models following the same experimental settings described in FlashAttention [26, 24]. We compare SLOPE against two state-of-the-art sparse pretraining methods, including (a) Wanda [136] → a one-shot pruning technique, (b) Extended SR-STE [168, 62] → a dynamic mask pretraining method for N:M sparsity, which serves as the foundation of follow-up work [63, 166, 62]. Please note that SR-STE only supports stochastic gradient descent optimization, and FST extended it to other optimizers. We use the extension provided by FST in our work, and call it Extended SR-STE. The diference between Extended SR-STE and FST is that FST requires dense pretraining (fine-tuning) in the last 17% of pretraining and only prunes the MLP layers of the model, while SR-STE is fully sparse and prunes both the MLP and the Self-Attention layers of the model.

Figure 4.2 compares the validation perplexity <sup>3</sup> and zero-shot accuracy of GPT2-Small and GPT2-Large across a range of sparse pretraining methods with diferent hyperparameters. We have additionally added lazy low-rank adapters to Extended SR-STE [168] to show the efectiveness of our approach in other methods and also compare both methods with more similar settings. While a gap in perplexity consistently exists between sparse and dense models, SLOPE achieves a lower perplexity compared to Wanda [136] and Extended SR-STE. Additionally, Table 4.3 summarizes the achieved accuracy of the models on zero-shot tasks, showing that SLOPE is consistently achieving a higher accuracy in comparison to Extended SR-STE. Moreover, adding lazy low-rank adapters can benefit both static and dynamic training methods. This improved accuracy stems from SLOPE’s eficient allocation of the training budget. Specifically, Extended SR-STE, with its dynamic pruning masks, expends a significant portion of its training budget (e.g. gradient updates) updating weights that may be ultimately pruned and not used at inference, leading to wasted resources. Section B.1 provides further details and supporting evidence for this observation. Additional validation results for GPT experiments on GLUE dataset are also provided in Section B.11 and Section B.10.

![](images/9da904f75243be557da9b132077dcd77d0304ad36c292883c2913e546de0ff9f.jpg)

![](images/3eb1642524527e3d99c3c4e7b4c673b0f1224b18454972dc2e0e9ad15e5f0369.jpg)  
Figure 4.2: Validation perplexity of GPT2-Small and GPT2-Large on OpenWebText. $\gamma _ { w }$ shows the value of the decay factor parameter in Extended SR-STE (FST).

Table 4.3: GPT2-Small accuracy results on zero-shot tasks. Adapter rank is the ratio of the low-rank adapter to the hidden dimension of the model. For Extended SR-STE, we have used a decay factor of $6 \times 1 0 ^ { - 6 }$ , since it resulted in the lowest perplexity in OpenWebText. The best performing sparse configuration is highlighted in bold.
<table><tr><td>METHOD</td><td>ADAPTER RANK</td><td>MMLU↑</td><td>ARC CHALLENGE↑</td><td>OPEN- BooκQA↑</td><td>WINO- GRANDE↑</td><td>HELLA- SWAG↑</td><td>MATHQA↑</td><td>PIQA↑</td><td>RACE↑</td></tr><tr><td>DENSE</td><td>N/A</td><td>22.9</td><td>20.7</td><td>16.2</td><td>50.6</td><td>28.5</td><td>21.8</td><td>59.8</td><td>28.4</td></tr><tr><td rowspan="3">SLoPE</td><td>2.1%</td><td>23.0</td><td>19.3</td><td>16.4</td><td>50.8</td><td>27.5</td><td>20.8</td><td>57.6</td><td>27.2</td></tr><tr><td>0.05%</td><td>23.0</td><td>19.4</td><td>16.2</td><td>50.5</td><td>27.4</td><td>20.8</td><td>57.5</td><td>27.1</td></tr><tr><td>0</td><td>23.0</td><td>19.3</td><td>16.0</td><td>50.1</td><td>27.5</td><td>20.8</td><td>57.4</td><td>27.1</td></tr><tr><td>EXTENDED</td><td>2.1%</td><td>24.2</td><td>18.3</td><td>14.2</td><td>47.5</td><td>26.9</td><td>21.4</td><td>55.2</td><td>24.2</td></tr><tr><td rowspan="2">SR-STE</td><td>0.05%</td><td>24.1</td><td>18.4</td><td>14.2</td><td>47.5</td><td>26.8</td><td>21.2</td><td>54.5</td><td>24.2</td></tr><tr><td>0</td><td>24.1</td><td>18.3</td><td>12.6</td><td>47.5</td><td>26.9</td><td>21.2</td><td>54.8</td><td>24.0</td></tr></table>

BERT-Large-Uncased. We pretrain BERT-Large-Uncased [30] (355 M parameters) and fine-tune it for var ious question-answering and text classification tasks, following a similar approach to [112, 99, 116] for both pretraining and fine-tuning. Section B.3 provides details on the pretraining and fine-tuning process. We eval uate the performance of BERT-Large-Uncased on the SQuAD v1.1 [123] and GLUE [147] tasks. We report the average metric score for GLUE and present the task-specific metrics in Section B.8. Please note that in all the experiments corresponding to BERT-Large-Uncased, when using Wanda, we have fine-tuned the model after pruning to improve the accuracy of the models, since using Wanda alone led to extremely low accuracy results.

Table 4.4: SQuAD-v1.1 accuracy and GLUE results on BERT-Large-Uncased with diferent adapter ranks. GLUE results are reported as the average metric score across all tasks. r denotes the ratio of the low-rank adapter to the hidden dimension (1024).
<table><tr><td>DATASET</td><td>DENSE</td><td> $r = 0$ </td><td> $r = 0 . 3 9 \%$ </td><td> $r = 1 . 5 6 \%$ </td><td> $r = 6 . 2 5 \%$ </td></tr><tr><td>SQuAD</td><td>90.44</td><td>89.1</td><td>89.1</td><td>89.2</td><td>89.5</td></tr><tr><td>GLUE</td><td>80.22</td><td>77.4</td><td>77.7</td><td>77.8</td><td>78.2</td></tr></table>

Efects of Low-rank Adapters. To understand the impact of low-rank adapters on pretraining performance, we conducted ablations using low-rank adapter ranks of 4, 16, and 64 for 1% of the total number of iterations. These ranks represent up to 6.25% of the model’s hidden dimension. Table 4.4 shows the results of these settings on SQuAD and GLUE downstream tasks. We present per-task metrics for GLUE in Section B.8. As expected, adding low-rank adapters improves the model’s final accuracy across all tasks. Additionally, higher ranks improve the model’s performance at the cost of increased computational requirements. It is also worth noting that incorporating low-rank adapters only in the final iterations (1% of total iterations) is suficient to recover pretraining accuracy.

Convergence Rate of Low-rank Adapters. We hypothesized that low-rank adapters would converge faster due to their significantly fewer learnable parameters. To test this, we introduced low-rank adapters in the second phase of BERT-Large-Uncased pretraining and monitored their convergence rate. Figure 4.3 shows the cosine similarity of the adapters, with the downsample adapter converging rapidly within 100 iterations and the upsample adapter converging slightly slower. Despite this, limiting training to 100 iterations still yields comparable results on downstream tasks.

![](images/f8f76e006532042ae8804653ad674254ea71c123083817d0a6fc811768963539.jpg)  
(a)

Low-Rank Adapter Similarity with Converged Weight  
![](images/180b5ea96b3ac4eb3c3fccbb410139f5bacba3897434ee1033c610db8a5ddad3.jpg)  
(b)  
Figure 4.3: (a) The speedup achieved using cuSPARSELt backend in PyTorch for Attention $( d _ { o u t } = d _ { i n } )$ Upsample $( d _ { o u t } = 4 d _ { i n } )$ and Downsample $\begin{array} { r } { ( d _ { o u t } = \frac { d _ { i n } } { 4 } ) } \end{array}$ matrices with a batch size of 2048. (b) The cosine similarity of the low-rank adapters and the converged adapters for diferent layers in the model. The cosine similarities are averaged among the 24 layers of BERT-Large-Uncased.

Efects of Mixed N:M sparsity. To study the sensitivity of diferent blocks to varying sparsity ratios and to assess their relative importance, we experiment across a range of configurations: (a) [2:4-2:4] → uniformly applying 2:4 sparsity across all layers (b) [2:4-2:8] → applying 2:4 sparsity pattern to the first 12 blocks and a 2:8 sparsity pattern to the last 12 blocks and (c) [2:8-2:4] → we reverse the sparsity ratios for the first and last 12 blocks. Note that, to reduce computational costs, we use the same dense checkpoint for Phase-1 in all settings and a low-rank adapter of rank 40 for all models. We also replicate this experiment using Wanda [136] and report the comparison results.

Table 4.5: SQuAD-v1.1 accuracy results on BERT-Large-Uncased for diferent sparsity settings.
<table><tr><td>SPARSITY PATTERN (FIRST 12 BLOCKs - LAST 12 BLOCKS)</td><td>SQuAD SLoPE</td><td>SQuAD WANDA</td><td>GLUE SLoPE</td><td>GLUE WANDA</td></tr><tr><td>2:4-2:4</td><td>90.17</td><td>89.93</td><td>79.08</td><td>78.84</td></tr><tr><td>2:4-2:8</td><td>89.85</td><td>89.55</td><td>79.03</td><td>77.24</td></tr><tr><td>2:8-2:4</td><td>89.67</td><td>86.57</td><td>75.92</td><td>69.08</td></tr></table>

Table 4.5 summarizes the GLUE and SQuAD results for these settings. As the results show, increasing the sparsity ratio reduces the accuracy of the model on all tasks. But when the first 12 blocks of the model are pruned, the accuracy drop is significantly higher, especially on the GLUE dataset. We conclude that the first blocks of the model are more sensitive to sparsity during pretraining, but one can sparsify the last blocks of LLMs more aggressively. We observe a similar pattern in Wanda results as well, but Wanda performs consistently worse than SLOPE in these cases.

Efects of sparsification on diferent modules. Each block in LLMs consists of a self-attention module and an MLP module, each containing multiple linear layers. We have analyzed the sensitivity of SLOPE to pruning each of those modules. Our results in Section 4.4.3 demonstrate that SLOPE can sustain competitive quality results while pruning all modules in the model.

## 4.4.3 Ablation Studies

Low-Rank Adapter Performance: Scaling and Arithmetic Intensity. As discussed in Section 4.3.4, the computation time of low-rank adapters does not scale linearly with their rank. This section provides experimental results to illustrate this behavior in more detail. The computational complexity of low-rank matrix multiplications is $\mathcal { O } ( b r d )$ , where b, r, and d represent the batch size, low-rank, and input/output dimensions of the layer, respectively. Based on this complexity, we expect the computation time to be a linear function of r. In other words, reducing r by a factor of α should result in a corresponding α-fold reduction in computation time. However, in practice, this linearity does not hold. This deviation arises because the assumption underlying this expectation – that matrix multiplication is compute-bound – is not always true. Specifically, the arithmetic intensity of the operation can fall below the machine’s balance point, as described in the Roofline model [152] in Section 2.2.2. Figure 4.4 shows the speedup achieved for diferent low-rank values using Py-Torch’s matrix multiplication function, which relies on the CUBLAS backend [107]. The figure demonstrates that the achieved speedups are significantly lower than the ideal linear scaling, particularly when reducing the rank. Moreover, it is evident that as the matrix dimensions increase, the gap between the ideal speedup and the observed speedup diminishes. This behavior can be attributed to the increased arithmetic intensity for larger matrices, leading to better utilization of tensor cores.

Low-Rank Matrix Multiplication Speedup  
![](images/da2c72746f48cbc8868a9c9be3c6db1c85a86ee163b0878872ff1999694e3711.jpg)  
Figure 4.4: The speedup achieved by low-rank adapters in comparison to a dense matrix-multiplication.

Table 4.6: End-to-end speedup (×) before and after eficient implementation of low-rank adapters.
<table><tr><td rowspan="2">MODEL</td><td colspan="2">1.56% ADAPTER</td><td colspan="2">6.25% ADAPTER</td></tr><tr><td>BEFORE</td><td>AFTER</td><td>BEFORE</td><td>AFTER</td></tr><tr><td>OPT-66B</td><td>1.15</td><td>1.20</td><td>1.12</td><td>1.19</td></tr><tr><td>OPT-30B</td><td>1.13</td><td>1.18</td><td>1.10</td><td>1.16</td></tr><tr><td>OPT-13B</td><td>1.11</td><td>1.10</td><td>1.09</td><td>1.10</td></tr><tr><td>OPT-6.6B</td><td>1.07</td><td>1.12</td><td>1.06</td><td>1.11</td></tr><tr><td>OPT-2.6B</td><td>1.01</td><td>1.06</td><td>0.97</td><td>1.00</td></tr></table>

Eficient Low-rank Adapter Implementation. As discussed in Section 4.3.4, a naïve implementation of low-rank adapters can lead to significant performance overheads due to the increased number of kernel launches and the low arithmetic intensity of their multiplications. To address these issues, we introduced two key optimizations: (1) concatenating one of the low-rank adapters with the sparse weights, and (2) fusing the multiplication of the other low-rank adapter with the subsequent result addition. These optimizations reduce kernel calls and increase arithmetic intensity, leading to more eficient utilization of GPU resources. Ta ble 4.6 summarizes the speedup improvements achieved with these optimizations, demonstrating an inference speedup increase of up to 6%.

Eficient Weight Tiling Implementation. We observed that the dimensions and aspect ratios of matrices significantly influence system speedup ( Section 4.3.4). To mitigate this, we implemented a matrix tiling strategy, dividing upsample matrices into multiple square matrices. This approach significantly improves performance, as shown in Table 4.7. Our results demonstrate that matrix tiling can enhance training speed by up to 4% and inference speed by up to 12%, highlighting its efectiveness in optimizing system performance.

SLOPE Sensitivity to Pruning Diferent Modules in Transformer. LLMs typically consist of two main modules: the MLP and the self-attention. The attention module’s weights are represented as a matrix in R<sup>d</sup>×<sup>3d</sup>, while the MLP uses weights in $\mathbb { R } ^ { d \times 4 d }$ and $\mathbb { R } ^ { 4 d \times d }$ , where d denotes the hidden dimension. To investigate the impact of sparsity on these modules, we conducted two experiments during Phase-2 of BERT-Large-Uncased pretraining: (a) [MLP] → pruning only MLP modules, and (b) [MLP + SELF-ATTENTION] → pruning both MLP and self-attention modules. Table 4.8 presents the SQuAD and GLUE results for these settings. A expected, we observe a consistent, albeit slight, decrease in model quality as more modules are sparsified. The marginal decrease in performance suggests that models are relatively insensitive to the specific modules being pruned when using our SLOPE pretraining method. This observation underscores the robustness of our approach and its ability to maintain competitive quality across diverse sparsity configurations.

Table 4.7: End-to-end speedup (×) before and after splitting the upsample matrix. In both cases, the optimization discussed in Table 4.6 is used.
<table><tr><td rowspan="2">MODEL</td><td colspan="2">TRAINING</td><td colspan="2">INFERENCE No ADAPTER</td><td colspan="2">INFERENCE 1.56% ADAPTER</td><td colspan="2">INFERENCE 6.25% ADAPTER</td></tr><tr><td>BEFORE</td><td>AFTER</td><td>BEFORE</td><td>AFTER</td><td>BEFORE</td><td>AFTER</td><td>BEFORE</td><td>AFTER</td></tr><tr><td>OPT-66B</td><td>1.10</td><td>1.13</td><td>1.22</td><td>1.34</td><td>1.20</td><td>1.31</td><td>1.19</td><td>1.30</td></tr><tr><td>OPT-30B</td><td>1.09</td><td>1.14</td><td>1.23</td><td>1.32</td><td>1.18</td><td>1.28</td><td>1.16</td><td>1.27</td></tr><tr><td>OPT-13B</td><td>1.10</td><td>1.12</td><td>1.23</td><td>1.30</td><td>1.10</td><td>1.30</td><td>1.10</td><td>1.12</td></tr><tr><td>OPT-6.6B</td><td>1.08</td><td>1.08</td><td>1.21</td><td>1.19</td><td>1.12</td><td>1.13</td><td>1.11</td><td>1.12</td></tr><tr><td>OPT-2.6B</td><td>1.03</td><td>1.02</td><td>1.02</td><td>1.07</td><td>1.06</td><td>1.05</td><td>1.00</td><td>1.00</td></tr></table>

Table 4.8: SQuADv1.1 results on BERT-Large-Uncased for diferent pruned modules.
<table><tr><td>PRUNED MODULES</td><td>SQuAD</td><td>GLUE</td></tr><tr><td>DENSE</td><td>90.44</td><td>80.22</td></tr><tr><td>MLP</td><td>90.28</td><td>79.03</td></tr><tr><td>MLP + SELF-ATTENTION</td><td>89.35</td><td>77.72</td></tr></table>

## 4.5 Conclusion

In this chapter, we presented SLOPE, a method that successfully executes the first strategy of the Compression Trinity: accelerating the computational cost of each pretraining iteration. By innovatively combining the sparsity pillar (via the double-pruned backward pass) and the low-rank pillar (via lazy adapters), SLOPE overcomes the rigidity of traditional sparse training. It delivers eficient N:M sparsity acceleration in both forward and backward passes while recovering model capacity through targeted low-rank updates. Our results demonstrate that this joint approach achieves up to 1.25× speedup in pretraining and 1.54× speedup in inference, while reducing memory footprints by 0.63× and 0.61×, respectively.

Together with MKOR (Chapter 3), these contributions conclude our exploration of the Pretraining lifecycle stage. We have demonstrated that the Compression Trinity can efectively accelerate training by attacking the problem from two orthogonal angles: reducing the number of iterations via a Trinity-enhanced optimizer (MKOR), and reducing the cost per iteration via Trinity-enhanced weight structures (SLOPE).

The narrative now shifts to the second major stage of the LLM life-cycle: Post-Training Compression for Inference. While SLOPE produces eficient sparse models, the ultimate goal of the Trinity is to jointly apply all three pillars, i.e., Sparsity, Low-Rank, and Quantization, to maximize inference eficiency on commodity hardware. However, applying these aggressive compression techniques simultaneously to a pre-trained, static model introduces a new challenge: compounded error. Before we can achieve the full Trinity in a one-shot inference setting, we must first establish a stable foundation. The next chapter introduces OPTIMA, where we rigorously perfect the sparsity pillar to withstand the pressures of joint compression.

Chapter 5

# OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction

Publication and Contributions. The content of this chapter is based on the paper “OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction,” [98] 2025. This work was conducted in collaboration with Samuel Kushnir, Amir Yazdanbakhsh, and Maryam Mehri Dehnavi. Mohammad Mozafari conceived the project, led the implementation, and designed and executed the experiments. Samuel Kushnir contributed to the design of the algorithm. Amir Yazdanbakhsh and Maryam Mehri Dehnavi supervised the project and contributed to the writing and revision of the manuscript.

## 5.1 Introduction

Having addressed the computational bottlenecks of pretraining in Chapter 3 and Chapter 4, we now turn our attention to the second major stage of the LLM life-cycle: inference. As discussed in Chapter 2, eficient inference is primarily constrained by memory bandwidth. While the Compression Trinity advocates for the joint application of sparsity, quantization, and low-rank approximations, blindly applying these methods to a pre-trained model carries significant risk. This is particularly true for sparsity, which we identified in Section 1.3 as the most structurally destructive pillar of the Trinity. Unlike quantization, which preserves the network’s topology, sparsity deletes connections entirely. If this structural skeleton is flawed, no amount of subsequent quantization or low-rank adaptation can recover the lost information.

Therefore, before we can integrate the full Trinity, we must first maximize the accuracy of the sparse foundation. In this chapter, we operate under a strict resource-constrained regime: we assume the practitioner requires a ”one-shot” solution with no access to the full training pipeline or budget for backpropagation. The goal is to determine the mathematical limit of reconstruction accuracy achievable using only a small calibration dataset and static, layer-wise optimization.

Post-training one-shot pruning [58], which removes parameters from a pretrained model using only a small calibration dataset, ofers a potential solution. However, current methods are often forced to choose between eficiency and reconstruction optimality. We can categorize existing approaches into two tiers. The first tier consists of fast, metric-based selectors like Wanda [136], magnitude pruning [53], and ProxSparse [83]. While computationally cheap, they perform no weight updates to compensate for removed connections, treating weights as independent and ignoring the correlations captured by the loss landscape curvature. Con versely, principled second-order approaches like Optimal Brain Surgeon [54] theoretically recover accuracy but are computationally infeasible at modern LLM scales. As a result, the second tier includes methods like SparseGPT [36] and Thanos [64], which attempt to adjust the remaining weights. However, to maintain speed, these methods rely on greedy approximations (e.g., iterative coordinate descent or localized Cholesky updates) rather than solving for the global optimum. Consequently, they leave significant performance on the table, an error that becomes significant when compounded with the quantization noise introduced in later chapters.<sup>1</sup>

To resolve this, we introduce OPTIMA, a practical one-shot post-training pruning framework that combines layer-wise optimality with accelerator-grade eficiency. Distinct from prior heuristics, we formulate the weight update not as an approximation, but as an exact constrained Convex Quadratic Program (QP). This approach draws a direct parallel to the second-order optimization methods discussed in Chapter 2 (e.g., KFAC). Just as KFAC leverages the curvature of the loss landscape (via the Fisher Information Matrix) to improve training convergence over first-order methods, OPTIMA utilizes the exact curvature of the layer-wise objective (via the Hessian matrix) to minimize pruning error.

The core of our methodology relies on a precise reformulation of the layer-wise reconstruction step. We observe that after fixing a binary mask for a weight matrix, the layer-wise output reconstruction (least-squares) objective decomposes across columns. We exploit a fundamental algebraic property of Transformer linear layers: while the linear constraints difer for each column (dictated by the mask), every column in the same layer shares the same Hessian matrix $H = X ^ { \top } X$ . Unlike previous works that approximate this Hessian to simplify computation, we use this exact structure to formulate the update for each column as a small constrained QP. This shared-Hessian structure allows us to guarantee per-column global optimality for the reconstruction objective, strictly outperforming greedy heuristics without making additional assumptions about weight independence.

Realizing this formulation in practice requires careful numerical and systems engineering. We adopt a first-order primal–dual QP solver (rAPDHG [88]) that is well-suited to our constrained problems, as its critical operations reduce to eficient matrix–vector products with the shared Hessian. We further avoid explicit dense equality matrices by enforcing fixed entries via tight bounds, accumulate layer Hessians incrementally from calibration sequences to save memory, and solve columns in batches so thousands of small QPs are processed in parallel. These implementation choices make OPTIMA not only theoretically principled but also practical to run on a single accelerator.

We evaluate OPTIMA across multiple model families (LLaMA, Gemma, and others) and sparsity regimes, including unstructured and 2:4 semi-structured sparsity. OPTIMA is designed to be modular; it acts as a drop-in replacement for the weight-update step in existing mask selectors (e.g., Wanda, SparseGPT, Thanos), consistently improving their zero-shot performance. Across six zero-shot downstream benchmarks in the Language Model Evaluation Harness, we observe up to 3.97% absolute gains on downstream tasks without any post-pruning fine-tuning.

In summary, our contributions are:

• We present a column-wise QP reformulation of the post-training reconstruction problem that yields percolumn global optimality under a shared-Hessian model and is provably equivalent to the least-squares objective after mask selection (Section 5.4).

![](images/214b3828ea48baef2379be9b9590763445288f00deb35d45148716e8983c5c85.jpg)  
Figure 5.1: OPTIMA generates a shared Hessian among the diferent columns of the pruned weight using a small calibration dataset. Then, the weights in diferent columns will be updated in parallel using a QP solver and the shared Hessian.

• We design and implement an accelerator-friendly QP solver pipeline that accumulates a single Hessian per layer, enforces mask constraints via bounds, batches thousands of column QPs, and leverages rAPDHG/M-PAX for eficient execution on GPUs/TPUs (detailed in Algorithm 3).

• We demonstrate the modularity of OPTIMA, showing it can be used as a drop-in weight-update step with common mask selection algorithms (Wanda, SparseGPT, Thanos), consistently improving their accuracy without fine-tuning (Section 5.5).

• We provide extensive empirical evidence and practical measurements. OPTIMA yields substantial average accuracy gains across tasks and model sizes (up to 3.97%), demonstrates robustness at high sparsity (up to 60%), and can prune billion-parameter models on a single H100 in less than 40 hours.

## 5.2 Additional Related Work

Model pruning compresses trained neural networks by eliminating redundant weights, thereby lowering computational and memory requirements during deployment. The field primarily divides into two categories: layer-wise pruning, exemplified by Optimal Brain Surgeon (OBS) [55], and end-to-end pruning, represented by Optimal Brain Damage (OBD) [75]. We review these approaches in the following subsections, beginning with layer-wise methods.

Layer-wise Model Pruning. Layer-wise pruning optimizes models by targeting redundancies within individual layers, assuming that local error reductions aggregate to minimize overall model degradation. Optimal Brain Surgeon (OBS) [55] formalizes this by identifying the least salient weight per layer and adjusting remaining weights to ofset its removal [35]. However, OBS’s computational intensity hinders its application to billion-parameter LLMs, necessitating approximations. SparseGPT [36] pioneered scaling OBS to LLMs by framing pruning as sparse regression problems solved approximately, trading some accuracy for eficiency. Thanos [64] refines this with multi-column pruning to cut approximation errors. In contrast, Wanda [136] employs a saliency metric combining weight magnitudes and activation data from calibration sets, yielding strong results with minimal pruning time. Nonetheless, Wanda lacks mechanisms to update weights post-pruning, opening avenues for enhancements, particularly in end-to-end methods that consider global interactions.

End-to-end Model Pruning. Unlike layer-wise methods, end-to-end pruning, exemplified by Optimal Brain Damage (OBD) [75], identifies least-important weights globally by leveraging second-order derivatives of the loss function, yielding higher accuracy than OBS. However, computing these derivatives is resource-intensive, demanding approximations [99]. WoodFisher [134] employs Kronecker factorization to approximate the Hessian, easing computation but still faltering at LLM scales. More recently, MaskLLM [33] sidesteps secondorder information by recasting pruning as a classification problem solved via standard optimizers like AdamW [86], achieving top performance at 2:4 sparsity. ProxSparse [83] reduces the costs of MaskLLM by using regularizers instead of training the model on a classification task, trading accuracy for speed. Yet, its optimization demands far exceed those of one-shot pruning, constraining real-world use and highlighting the value of integrating with other compression strategies.

Other Model Compression Methods. In addition to pruning, several orthogonal techniques enable model compression and can be integrated with pruning for compounded benefits. Quantization reduces parameter precision to lower-bit representations, as surveyed in [43, 125], minimizing memory footprint without severe accuracy loss.

Low-rank adapters, such as those in [100, 48, 101], decompose weight matrices into lower-dimensional factors, while knowledge distillation [46] transfers knowledge from larger teacher models to compact students. These methods complement pruning by addressing diferent aspects of redundancy, paving the way for hybrid frameworks in advanced compression research.

## 5.3 Preliminaries

Post-training pruning (PTP) compresses pre-trained models without retraining, using a small calibration dataset to produce a sparse model that preserves performance. To make PTP tractable, the problem is decomposed into independent layer-wise subproblems. For layer l, the goal is to find a binary sparsity mask $\mathbf { M } _ { l }$ and updated weights $\hat { \mathbf { W } } _ { l }$ that minimize the output reconstruction error given original weights $\mathbf { W } _ { l }$ and input activations $\mathbf { X } _ { l }$ This task can be formulated as in Equation 5.1, where $\odot$ denotes the Hadamard product, and $\mathbf { M } _ { l }$ is a binary tensor of the same shape as $\mathbf { W } _ { l }$ with 0s for pruned weights and 1s for retained ones. Equation 5.1 is solved sequentially across layers, with $\mathbf { X } _ { l }$ as the pruned output from layer l − 1. Finding the optimal $\mathbf { M } _ { l }$ is NP-hard, motivating heuristics.

$$
\begin{array} { r l } & { \underset { { \mathbf { M } _ { l } } , \hat { \mathbf { W } _ { l } } } { \mathrm { a r g m i n } } \| { \mathbf { X } _ { l } } { \mathbf { W } _ { l } } - { \mathbf { X } _ { l } } \big ( { \mathbf { M } _ { l } } \odot \hat { { \mathbf { W } } } _ { l } \big ) \| _ { F } ^ { 2 } } \\ & { ~ { \mathbf { M } _ { l } } , \hat { \mathbf { W } } _ { l } } \end{array}\tag{5.1}
$$

A common heuristic decouples mask selection from weight updates. After selecting $\mathbf { M } _ { l }$ (e.g., by magnitude), the problem simplifies to Equation 5.2, which is a convex least-squares problem, but solving it directly is computationally expensive for large LLM weights.

$$
\operatorname* { m i n } _ { \hat { \mathbf { W } } _ { l } } \| \mathbf { X } _ { l } \mathbf { W } _ { l } - \mathbf { X } _ { l } \big ( \mathbf { M } _ { l } \odot \hat { \mathbf { W } } _ { l } \big ) \| _ { F } ^ { 2 }\tag{5.2}
$$

Consequently, many methods employ strategies to circumvent the expensive weight update step. For example, Wanda [136] avoids weight updates altogether, simply setting the selected weights to zero. However, other methods such as SparseGPT [36] and Thanos [64] adopt a compromise, performing a more complex

update but only on a small subset of the weights. These heuristics trade of optimality for computational feasibility.

## 5.4 OPTIMA: Optimal Weight Updates via Quadratic Programming

To overcome the challenges of weight update in LLM pruning, we propose OPTIMA, a novel approach that enables the eficient and optimal update of all remaining weights once the pruning mask $\mathbf { M } _ { l }$ has been chosen.

We achieve this by reformulating the least-squares problem as a set of independent Quadratic Programs (QPs) that can be solved in parallel on hardware accelerators like GPUs or TPUs using iterative methods. Specifically, we derive both a linearly constrained QP formulation and an equivalent unconstrained formulation. While the unconstrained form can be useful for optimizers restricted to such problems or in cases where it can be solved more eficiently, our implementation focuses on the constrained QP formulation, which is more amenable to GPU/TPU acceleration.

## 5.4.1 Reformulation as a Quadratic Program with Linear Constraints

As discussed in Section 5.3, our goal is to minimize the problem defined in Equation 5.2. The Frobenius norm objective function in Equation 5.2 is separable by the columns of the weight matrix $\cdot ^ { 2 }$ We can therefore solve the optimization problem for each column independently.

Let $\mathbf { w } _ { j }$ be the j-th column of the original weight matrix $\mathbf { W } _ { l }$ , and let $\hat { \mathbf { w } } _ { j }$ be the corresponding column in the updated matrix $\hat { \mathbf { W } } _ { l }$ . The mask for this column is $\mathbf { m } _ { j }$ . The optimization for this single column can be formulated as in Equation 5.3.

$$
\operatorname* { m i n } _ { \hat { \mathbf { w } } _ { j } } \| \mathbf { X } _ { l } \mathbf { w } _ { j } - \mathbf { X } _ { l } \big ( \mathbf { m } _ { j } \odot \hat { \mathbf { w } } _ { j } \big ) \| _ { 2 } ^ { 2 }\tag{5.3}
$$

By defining the change in the weight column as $\Delta \mathbf { w } _ { j } \ = \ ( \mathbf { m } _ { j } \odot { \hat { \mathbf { w } } } _ { j } ) \ - \mathbf { w } _ { j }$ , the objective can then be rewritten in terms of this change as in Equation 5.4 in standard quadratic form.

$$
\operatorname* { m i n } _ { \Delta \mathbf { w } _ { j } } \| - \mathbf { X } _ { l } \Delta \mathbf { w } _ { j } \| _ { 2 } ^ { 2 } = \operatorname* { m i n } _ { \Delta \mathbf { w } _ { j } } \Delta \mathbf { w } _ { j } ^ { T } ( \mathbf { X } _ { l } ^ { T } \mathbf { X } _ { l } ) \Delta \mathbf { w } _ { j }\tag{5.4}
$$

The constraints on $\Delta \mathbf { w } _ { j }$ in Equation 5.4 are determined by the mask $\mathbf { m } _ { j }$ . Let $S _ { j }$ be the set of indices where the mask is zero (i.e., weights to be pruned). For each index $i \in S _ { j }$ , the corresponding entry in the updated weight vector, $( \hat { \mathbf { w } } _ { j } ) _ { \mathrm { ? } }$ <sub>i</sub>, must be zero. This imposes a linear constraint on the change vector, as shown in Equation 5.5.

$$
( \mathbf { m } _ { j } \odot \hat { \mathbf { w } } _ { j } ) _ { i } = 0 \implies ( \Delta \mathbf { w } _ { j } ) _ { i } = - ( \mathbf { w } _ { j } ) _ { i } \quad \forall i \in \mathcal { S } _ { j }\tag{5.5}
$$

The entries of $\Delta { \bf w } _ { j }$ for the unpruned weights (where $m _ { i j } = 1 )$ remain as free variables to be optimized. For each column $j$ of the weight matrix, we have a QP of the form represented in Equation 5.6, where $\mathbf { H } = \mathbf { X } _ { l } ^ { T } \mathbf { X } _ { l }$ is the Hessian matrix, which is positive semi-definite and shared across all column-wise problems. The fact that the Hessian is shared among all columns, and only the constraints change, makes it very easy to parallelize on accelerators such as GPUs and TPUs.

CHAPTER 5. OPTIMA: OPTIMAL ONE-SHOT PRUNING FOR LLMS VIA QUADRATIC PROGRAMMING RECONSTRUCTION

$$
\begin{array} { r l } { \underset { \Delta \mathbf { w } _ { j } } { \mathrm { m i n i m i z e } } } & { \Delta \mathbf { w } _ { j } ^ { T } \mathbf { H } \Delta \mathbf { w } _ { j } } \\ { \mathrm { s u b j e c t ~ t o } } & { ( \Delta \mathbf { w } _ { j } ) _ { i } = - ( \mathbf { w } _ { j } ) _ { i } , \forall i \in \mathcal { S } _ { j } } \end{array}\tag{5.6}
$$

## 5.4.2 Reformulation as an Unconstrained Quadratic Program

As an alternative to the constrained formulation in Equation 5.6, we can reformulate each column-wise problem as an unconstrained quadratic program. This can be useful in settings where solvers are optimized for unconstrained problems or when eliminating constraints enables more eficient optimization. Although our implementation adopts the constrained approach for reasons discussed below, we include the unconstrained version for completeness.

The key idea is to eliminate the equality constraints in Equation 5.5 by substituting them directly into the objective. For a given column $j ,$ define $\mathcal { T } _ { j }$ as the set of indices where the mask is one (i.e., unpruned weights), and let $S _ { j }$ denote the complement set (i.e., pruned weights, where the mask is zero).

We reorder the entries of the change vector $\Delta { \bf w } _ { j }$ and the shared Hessian matrix $\mathbf { H } = \mathbf { X } _ { l } ^ { T } \mathbf { X } _ { l }$ based on this partitioning, as shown in Equation 5.7.

$$
\Delta \mathbf { w } _ { j } = \left[ \begin{array} { l } { \Delta \mathbf { w } _ { \mathcal { T } _ { j } } } \\ { \Delta \mathbf { w } _ { \mathcal { S } _ { j } } } \end{array} \right] , \quad \mathbf { H } = \left[ \begin{array} { l l } { \mathbf { H } _ { \mathcal { T } _ { j } \mathcal { T } _ { j } } } & { \mathbf { H } _ { \mathcal { T } _ { j } \mathcal { S } _ { j } } } \\ { \mathbf { H } _ { \mathcal { S } _ { j } \mathcal { T } _ { j } } } & { \mathbf { H } _ { \mathcal { S } _ { j } \mathcal { S } _ { j } } } \end{array} \right]\tag{5.7}
$$

As established in Equation 5.5, the entries of $\Delta \mathbf { w } _ { j }$ corresponding to $S _ { j }$ are fixed: $( \Delta \mathbf { w } _ { j } ) _ { i } = - ( \mathbf { w } _ { j } ) _ { i }$ <sub>i</sub> for all $i \in S _ { j }$ . Substituting these fixed values into the quadratic objective yields the expanded form in Equation 5.8.

$$
\Delta \mathbf { w } _ { j } ^ { T } \mathbf { H } \Delta \mathbf { w } _ { j } = \Delta \mathbf { w } _ { T _ { j } } ^ { T } \mathbf { H } _ { T _ { j } T _ { j } } \Delta \mathbf { w } _ { T _ { j } } + 2 \Delta \mathbf { w } _ { T _ { j } } ^ { T } \mathbf { H } _ { T _ { j } S _ { j } } \Delta \mathbf { w } _ { S _ { j } } + \Delta \mathbf { w } _ { S _ { j } } ^ { T } \mathbf { H } _ { S _ { j } S _ { j } } \Delta \mathbf { w } _ { S _ { j } }\tag{5.8}
$$

Since $\Delta \mathbf { w } _ { S _ { j } } = - \mathbf { w } _ { S _ { j } }$ , we substitute this to obtain the unconstrained objective in Equation 5.9.

$$
\operatorname* { m i n } _ { \Delta \mathbf { w } _ { \mathcal { T } _ { j } } } \left( \Delta \mathbf { w } _ { \mathcal { T } _ { j } } ^ { T } \mathbf { H } _ { \mathcal { T } _ { j } \mathcal { T } _ { j } } \Delta \mathbf { w } _ { \mathcal { T } _ { j } } - 2 \Delta \mathbf { w } _ { \mathcal { T } _ { j } } ^ { T } \mathbf { H } _ { \mathcal { T } _ { j } S _ { j } } \mathbf { w } _ { S _ { j } } + \mathbf { w } _ { S _ { j } } ^ { T } \mathbf { H } _ { S _ { j } S _ { j } } \mathbf { w } _ { S _ { j } } \right)\tag{5.9}
$$

The final term in Equation 5.9 is constant with respect to the optimization variable $\Delta \mathbf { w } _ { \mathbb { Z } _ { j } }$ and can therefore be omitted. This results in the unconstrained quadratic program in Equation 5.10.

$$
\begin{array} { r l } { \underset { \Delta \mathbf { w } _ { \mathbb { T } _ { j } } } { \mathrm { m i n i m i z e } } } & { \Delta \mathbf { w } _ { \mathbb { Z } _ { j } } ^ { T } \mathbf { Q } _ { j } \Delta \mathbf { w } _ { \mathbb { Z } _ { j } } + \mathbf { c } _ { j } ^ { T } \Delta \mathbf { w } _ { \mathbb { Z } _ { j } } } \end{array}\tag{5.10}
$$

where the problem-specific matrix and vector are defined as:

$$
\begin{array} { r } { \mathbf Q _ { j } = \mathbf H _ { \mathcal I _ { j } \mathcal I _ { j } } , \quad \mathbf c _ { j } = - 2 \mathbf H _ { \mathcal I _ { j } \mathcal S _ { j } } \mathbf w _ { \mathcal S _ { j } } } \end{array}\tag{5.11}
$$

This formulation eliminates the need for explicit constraints, but introduces column-dependent variation in problem dimensions. Specifically, the size of $\mathbf { Q } _ { j }$ and $\mathbf { c } _ { j }$ varies with the number of unpruned weights in each column. Consequently, the unconstrained QPs have heterogeneous shapes and objectives across columns, making them more dificult to batch and parallelize eficiently on accelerators like GPUs or TPUs. This motivates our choice to adopt the constrained formulation in Equation 5.6, where the problem structure is uniform and well-suited for high-throughput parallel execution.

```latex
Algorithm 3 Layer-wise Pruning with Batched Column-wise Quadratic Programming
Input: Pre-trained LLM M, calibration data X, pruning masks $\mathcal { M } _ { \mathrm { a s k } }$ , QP solver S, batch size B.
Output: Pruned and updated LLM M<sup>ˆ</sup> , updated masks $\hat { \mathcal { M } } _ { \mathrm { a s k } } .$
1 for each layer L in the LLM M do
2 Initialize Hessian estimate $\mathbf { H }  0 .$ ▷ Initialize covariance matrix
3 for each calibration sample $x \in \mathbf { X }$ do
4 $y  L ( x )$ ▷ Forward pass for one sequence
5 $\mathbf { \bar { H } }  \mathbf { \bar { H } } + y ^ { T } y$ ▷ Accumulate covariance
6 end for
7 Store intermediate inputs $\{ \mathbf { X } _ { \mathbf { W } } \mid \mathbf { W } \in L \}$ from a forward pass of $L ( \mathbf { X } )$
8 for each weight matrix W in layer L do
9 Retrieve corresponding mask $\mathbf { M } \in \mathcal { M } _ { \mathrm { a s k } }$
10 Partition the columns of W into batches of size $B .$
11 for each batch of columns $\{ \mathbf { w } _ { j } \} _ { j = 1 } ^ { B }$ in parallel do
12 for each column ${ \bf w } _ { j }$ in the batch do
13 ${ \cal { S } } _ { j }  \{ i \mid { \bf { M } } _ { j , i } = 0 \}$ ▷ Indices of pruned entries
14 Define QP:
min $\Delta \mathbf { w } _ { j } ^ { T } \mathbf { H } \Delta \mathbf { w } _ { j }$
∆w<sub>j</sub> (5.12)
s.t. $( \Delta \mathbf { w } _ { j } ) _ { i } = - ( \mathbf { w } _ { j } ) _ { i } , \ \forall i \in S _ { j }$
15 end for
16 $\{ \Delta \mathbf { w } _ { j } \} _ { j = 1 } ^ { B }  S ( \mathbf { H } , \{ \mathbf { w } _ { j } \} _ { j = 1 } ^ { B } , \{ S _ { j } \} _ { j = 1 } ^ { B } )$
17 Update weights: $\mathbf { w } _ { j }  \mathbf { w } _ { j } + \Delta \mathbf { w } _ { j }$ ∀j
18 end for
19 end for
20 $\mathbf { X } \gets L ( \mathbf { X } )$ ▷ Update activations for next layer
21 end for
Return: Updated model $\hat { \mathcal { M } } ,$ updated masks $\hat { \mathcal { M } } _ { \mathrm { a s k } } .$
```

## 5.4.3 Solving the Quadratic Programs

With the constrained QP formulation established, we now select a solver, whose eficiency is crucial for runtime and scalability on parallel hardware like GPUs and TPUs. Our QP, with its shared Hessian H and simple bounds, suits specialized modern solvers. We adopt the state-of-the-art Restarted Accelerated Primal-Dual Hybrid Gradient (rAPDHG) algorithm [88], a first-order method efective here for three reasons: (1) its bottleneck—matrix-vector multiplications with H and its transpose—runs eficiently on GPUs/TPUs; (2) it achieves provably optimal linear convergence; and (3) a high-performance, open-source JAX-based imple mentation is available in MPAX [87], designed for GPU/TPU execution. This enables parallel solving of thousands of column-wise QPs, leveraging the shared structure.

## 5.4.4 Eficient Implementation

Naively implementing the optimization problem in Equation 5.6 is computationally expensive and incurs substantial memory overhead. These costs, however, can be greatly reduced through a series of optimization techniques. In the following, we describe the strategies we employ to solve the QPs eficiently on a single GPU, even for very large LLMs. Additionally, a detailed algorithm of our implementation is provided in Algorithm 3.

Equality Constraints. Directly encoding the constraints from Equation 5.5 into the standard quadratic objective leads to a prohibitively large matrix of equalities, even though these constraints merely fix individual variables to constant values. To avoid constructing such large matrices, we instead enforce the constraints by setting upper and lower bounds on the corresponding variables. In particular, fixing the bounds of $( \Delta w _ { j } )$ <sub>i</sub> to $- ( w _ { j } ) _ { i }$ <sub>i</sub> efectively locks the variable to the desired value, without incurring the overhead of explicit equality matrices.

Batching QP Problems. In memory-limited scenarios, the optimization problems for all columns of the weight matrices may not fit on a single GPU. To address this, we employ a batching strategy that solves a subset of QP problems at a time. This approach reduces memory overhead while still leveraging the eficiency of solving multiple QPs in parallel. As a result, our method enables pruning of large LLMs even on a single GPU.

Hessian calculation. For each layer, the Hessian matrix can be estimated as the covariance of the dense model’s outputs across multiple sequences. Suppose the output tensor is $Y \in \mathbb { R } ^ { b \times s \times d }$ , where b is the number of sequences, s is the sequence length, and d is the output dimension of the layer. To compute the covariance directly, we would first reshape Y into $\hat { Y } \in \mathbb { R } ^ { b s \times d }$ , efectively stacking all tokens from all sequences into a single matrix, and then evaluate $\hat { Y } ^ { T } \hat { Y }$

While this formulation is straightforward, it requires storing the full Y in accelerator memory, which becomes prohibitively expensive for large b and s, often causing out-of-memory errors. To make the computation feasible, we observe that the covariance can be accumulated incrementally. Specifically, Y can be decomposed into b smaller matrices, $y _ { i } \in \mathbb { R } ^ { s \times d }$ , each corresponding to the output of a single sequence. Instead of materializing $\hat { Y }$ , we compute $y _ { i } ^ { T } y _ { i }$ for each sequence separately and sum the results as in $\begin{array} { r } { H \approx \sum _ { i = 1 } ^ { b } y _ { i } ^ { T } y _ { i } } \end{array}$ This decomposition yields the same result as computing $\hat { Y } ^ { T } \hat { Y }$ directly, but avoids the need to store the entire $Y$ at once, making the approach scalable to very large LLMs.

## 5.5 Experiments

Model, datasets, and evaluation. We evaluate OPTIMA on LLaMA 3.1, LLaMA 3.2 [31], Gemma 2 [140], and Gemma 3 [139] family of models. Model accuracy is assessed on a range of zero-shot downstream tasks, including MMLU [57], Piqa [13], Arc-Easy, Arc-Challenge [21], WinoGrande [126], and OpenBookQA [97], all of which are commonly used to evaluate LLM compression [100, 136]. For zero-shot evaluations, we utilize the Language Model Evaluation Harness [41] framework. In line with prior work [136, 36, 100], we also report the perplexity of the models on a language modeling task on the WikiText2 [94] dataset.

Baselines. We compare OPTIMA against state-of-the-art one-shot pruning methods, including Wanda [136], SparseGPT [36], Thanos [64], and ProxSparse [83] and show how OPTIMA can improve the performance of all these pruning methods across diferent models and datasets. The sensitivity of OPTIMA to the calibration dataset size can be found in Section C.1. In terms of memory reductions and speedup, our method is guaranteed to achieve the same performance as other pruning methods such as Wanda and SparseGPT, since the sparsity pattern in these methods stays intact.

Experiment Setup. Following previous work [36, 136, 100, 64], we use 128 samples, each with 2048 tokens from the C4 dataset [121] for calibration. We set the relative and absolute tolerance of the rAPDHG QP solver in MPAX to 0.01 and the maximum number of iterations to 100,000. If the optimizer does not converge within this budget for most problems, or the final error of a layer is larger than the initial error, OPTIMA skips updating that layer. Table 5.1 summarizes the key hyperparameters. For all baselines, we either use their publicly available checkpoints or reproduce results with default hyperparameters.

![](images/5e16dc02ab96af7f5ac82967513aeb5401712e3118cfc4341df5b147c78993c0.jpg)  
Layer Number  
Figure 5.2: Relative error reduction on OPTIMA in comparison to Wanda, SparseGPT, and Thanos for LLaMA-3.2 1B.

Table 5.1: Key hyperparameters used in OPTIMA.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Calibration Samples</td><td>128</td></tr><tr><td>Tokens per Sample Dataset for Calibration</td><td>2048 C4</td></tr><tr><td>Relative Tolerance (rAPDHG)</td><td>0.01</td></tr><tr><td>Absolute Tolerance (rAPDHG)</td><td>0.01</td></tr><tr><td>Maximum Iterations (rAPDHG)</td><td>100,000</td></tr><tr><td>ADAM Learning Rate</td><td> $\{ 1 0 ^ { - 2 } , 1 0 ^ { - 3 } , 1 0 ^ { - 4 } , 1 0 ^ { - 5 } \}$ </td></tr><tr><td>ADAM Weight Decay</td><td>0</td></tr></table>

Model Quality. We evaluate the accuracy of OPTIMA and other state-of-the-art pruning methods across 2:4 and unstructured sparsity benchmarks. Wanda is a mask selection algorithm that does not provide any weight update mechanism for the weights. SparseGPT and Thanos, on the other hand, update the weight values in addition to searching for the best mask. We couple OPTIMA weight update with the masks generated using each of these methods and compare the resulting performance of the models.

Table 5.2 summarizes the performance metrics for Wanda, SparseGPT, and Thanos with and without the OPTIMA update mechanism for 50% unstructured sparsity. It can be seen that models pruned with OPTIMA weight update scheme consistently outperform the methods using weight update methods, providing up to

1.80% average accuracy improvement across six downstream tasks (Gemma-3-1B).

Table 5.3 presents the results of pruning transformer models using 2:4 semi-structured sparsity. In these experiments, we applied pruning exclusively to the weight matrices in the multilayer perceptron (MLP) components, leaving the self-attention layers dense. This approach yielded sparse models with an overall sparsity of 38% to 41%. We adopted this selective pruning strategy to maintain model accuracy above a practical threshold, as 2:4 sparsity significantly impacts performance, potentially rendering fully sparse models inefective. Our results demonstrate that our proposed OPTIMA update mechanism consistently outperforms other methods under 2:4 sparsity, achieving superior accuracy.

Higher Sparsity Ratios. To assess the robustness of OPTIMA at more aggressive compression levels, we extend our evaluation to 60% unstructured sparsity. Table 5.4 presents the perplexity and zero-shot accuracy metrics across the same models and tasks. OPTIMA continues to deliver consistent improvements over the baseline pruning methods, with average accuracy gains of up to 2.53% across the downstream tasks (LLaMA-3.2-1B).

These enhancements are particularly notable at higher sparsity ratios, where pruning a larger portion of weights introduces greater reconstruction error. By optimally readjusting the remaining weights through our QP formulation, OPTIMA efectively mitigates this error, leading to lower perplexity and higher downstream performance compared to Wanda, SparseGPT, or Thanos individually. For example, on LLaMA-3.2-3B, OP-TIMA increases Wanda’s average accuracy from 38.77% to 42.74%, highlighting its ability to preserve model utility under extreme sparsity conditions.

Extended Evaluation on the Qwen-2.5 Model Family. To further validate the robustness and generalizability of OPTIMA, we conduct additional experiments on the Qwen-2.5 family of models, with sizes ranging from 0.5B to 14B parameters. These models were not included in the preceding analysis, and this evaluation serves to confirm that OPTIMA’s benefits apply across diferent model architectures.

We evaluate performance across three distinct settings, mirroring the main experiments: 50% unstructured sparsity (Table 5.5), 60% unstructured sparsity (Table 5.6), and 2:4 semi-structured sparsity (Table 5.7).

Unstructured Sparsity (50% and 60%). At 50% unstructured sparsity (Table 5.5), OPTIMA consistently improves zero-shot performance across all Qwen-2.5 model sizes and for all mask selection methods (Wanda, SparseGPT, and Thanos). For example, on the Qwen-2.5 3B model, OPTIMA boosts the average accuracy of Wanda from 54.02% to 55.33% and SparseGPT from 54.70% to 55.69%. These gains demonstrate that our OPTIMA reconstruction successfully recovers accuracy lost during the pruning step.

The advantages of OPTIMA are even more pronounced at the more aggressive 60% sparsity ratio, as shown in Table 5.6. At this level, pruning introduces a more significant reconstruction error, providing a greater opportunity for OPTIMA to recover performance. This is especially clear on the Qwen-2.5 3B model, where OPTIMA improves Wanda’s average accuracy from 43.67% to 47.86% (a 4.19% absolute gain) and Thanos’s from 48.45% to 49.98% (a 1.53% gain).

Semi-Structured Sparsity (2:4). In the 2:4 semi-structured sparsity setting (Table 5.7), where pruning is applied only to the MLP layers, OPTIMA provides clear improvements for most models, particularly in the 1.5B and 3B range. For instance, it improves the average accuracy of the 3B model pruned with Wanda from 49.48% to 50.63% and the 1.5B model from 46.01% to 47.26%.

On the larger 7B and 14B models, the results are more varied, with performance difering based on the underlying mask selector. This suggests a complex interaction between mask selection heuristics and OPTIMA reconstruction for structured sparsity at this scale, which could be a valuable avenue for future investigation.

Overall, these experiments on the Qwen-2.5 family reinforce the findings from the preceding sections. They confirm that OPTIMA is a broadly applicable and efective method for enhancing model accuracy postpruning, delivering its most significant and consistent gains in high-sparsity unstructured regimes.

Comparison with Alternative Optimizers. While our constrained QP solver leverages theoretical guarantees for convergence and optimality, we also compare it against ADAM [70], a popular first-order optimizer without such assurances for quadratic problems. We reformulate the weight update as a mean squared error (MSE) minimization problem and use ADAM for solving it. Optimizers such as ADAM do not guarantee convergence, and are sensitive to their hyperparameters. For each layer, we do an exhaustive search with 4 diferent learning rates ranging from $1 0 ^ { - 2 } ~ \mathrm { t o } ~ 1 0 ^ { - 5 }$ , each with a linear learning rate scheduler and choose the best configuration for final weight update.

Table 5.8 illustrates this on Gemma 3 1B and OPT 125M [165] under 50% unstructured sparsity. We show two examples in Table 5.8, showing that ADAM results in suboptimal solutions. To further test the limitations of optimizers without convergence guarantees, we test ADAM on OPT-125M, and observe that it leads to divergence of the model. On Gemma 3 1B, ADAM yields competitive results in some cases (e.g., slightly lower perplexity for SparseGPT+ADAM at 27.12 versus OPTIMA’s 27.35), but OPTIMA achieves higher overall accuracy (e.g., 44.01% for Wanda+OPTIMA versus 43.72% for Wanda+ADAM). However, on smaller models like OPT 125M, ADAM exhibits instability, leading to divergence and dramatically higher perplexity (e.g., 205.82 for Wanda+ADAM versus 35.44 for Wanda+OPTIMA). This underscores the risks of using non-specialized optimizers for our column-wise QPs, where suboptimal or unstable solutions can degrade model quality. OPTIMA’s use of provably convergent methods like rAPDHG ensures reliable and superior weight updates, making it a more robust choice for post-training pruning.

Layer-wise Error Improvement. To provide a deeper insight into how OPTIMA improves the accuracy of the models, we compare the layer-wise error of diferent layers in LLaMA-3.2 1B during pruning with and without OPTIMA. Figure 5.2 shows the relative output error improvement of all the pruned layers in the model, defined as $\frac { M S E ( Y _ { \mathrm { O P T I M A } } , Y _ { \mathrm { d e n s e } } ) } { M S E ( Y _ { \mathrm { o t h e r } } , Y _ { \mathrm { d e n s e } } ) }$ , where MSE denotes the mean squared error across the calibration dataset. Figure 5.2 shows that OPTIMA consistently improves the layer-wise error of other methods, resulting in superior accuracy on the downstream tasks.

Pruning Time Analysis. To evaluate the computational eficiency of OPTIMA, we measured the time required to prune various language models. The pruning process was conducted on a single NVIDIA H100 GPU with 80GB of memory. Our measurements show that pruning times vary with model size: smaller models like LLaMA 3.2 1B and Gemma 3 1B each required approximately 2.5 h, Gemma 2 2B took 5.5 h, LLaMA 3.2 3B needed 7.0 h, and the larger LLaMA 3.1 8B model required up to 40.0 h.

The results indicate that pruning time scales with model size, reflecting the computational complexity of OPTIMA’s pruning algorithm, which adapts to the architectural diferences across models. The consistency in pruning times for models of similar size (e.g., LLaMA 3.2 1B and Gemma 3 1B) highlights the robustness of OPTIMA in handling diverse model architectures eficiently.

## 5.6 Conclusion and Limitations

In this chapter, we explored the limits of the sparsity pillar under a strict resource-constrained regime. Recognizing that end-to-end training is not always feasible, we developed OPTIMA to determine the mathematical upper bound of reconstruction accuracy possible using only a small calibration dataset. By reformulating post-training weight reconstruction as globally optimal, column-wise Quadratic Programs (QPs) and leveraging the shared-Hessian structure, we achieved massive parallelism on standard GPUs without the need for backpropagation.

Our results demonstrate that this principled approach pays significant dividends. OPTIMA functions as a drop-in weight-update step for common mask selectors, improving zero-shot accuracy across various LLM families by up to 3.97 percentage points without any fine-tuning. Crucially, these gains persist even at high sparsity levels (≥ 60%), proving that it is possible to create a highly sparse model that retains the fidelity of the original dense network.

However, while OPTIMA minimizes the reconstruction error for any given mask, it remains bound by two fundamental limitations:

1. Layer-wise Pruning Sub-optimality: It must operate within the constraints of a fixed, pre-determined sparsity pattern (like 2:4), which may not align with the model’s true information distribution.

2. The Accuracy Gap: Even with optimal reconstruction, a gap often remains between the sparse model and the dense baseline, suggesting that sparsity alone—without the aid of other Trinity pillars—has reached its ceiling in the zero-shot regime.

To address the first limitation, the next chapter introduces PATCH. We transition from the ”no-training” regime of OPTIMA to a ”fine-tuning” regime, where we utilize a training budget to learn a flexible, hybrid sparsity structure that dynamically preserves density where it matters most. To address the second limitation (the accuracy gap), we will revisit OPTIMA’s findings in Chapter 7, where we demonstrate that re-introducing the Low-Rank pillar can bridge the remaining distance to dense performance.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Mask Selection</td><td rowspan="2">Weight Update</td><td rowspan="2">Perplexity</td><td colspan="8">Metrics (%)</td></tr><tr><td>MMLU</td><td>PIQA</td><td>Arc-E</td><td>Arc-C</td><td>Wino</td><td>OpenQA|</td><td colspan="2">Average</td></tr><tr><td rowspan="8">LLaMA 3.1 8B</td><td>Dense</td><td>一</td><td>5.84</td><td>63.57</td><td>80.09</td><td>81.44</td><td>51.37</td><td>73.48</td><td>33.40</td><td colspan="2">63.89</td></tr><tr><td>Wanda</td><td></td><td>9.64</td><td>47.79</td><td>75.68</td><td>72.56</td><td>40.70</td><td>70.09</td><td>27.40</td><td colspan="2">55.70</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>9.37</td><td>48.85</td><td>76.71</td><td>73.82</td><td>42.32</td><td>70.32</td><td>28.20</td><td colspan="2">56.70</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>9.30</td><td>51.32</td><td>76.19</td><td>73.02</td><td>41.27</td><td>70.88</td><td>29.40</td><td colspan="2">57.01</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>9.33</td><td>49.31</td><td>76.61</td><td>74.28</td><td>42.83</td><td>70.88</td><td>28.20</td><td colspan="2">57.02</td></tr><tr><td>Thanos</td><td>Thanos</td><td>9.27</td><td>50.36</td><td>77.04</td><td>74.92</td><td>42.58</td><td>70.96</td><td>30.00</td><td colspan="2">57.64</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>9.35</td><td>50.17</td><td>76.50</td><td>74.16</td><td>41.89</td><td>70.24</td><td>28.40</td><td colspan="2">56.89</td></tr><tr><td>Dense</td><td></td><td>9.75</td><td>36.92</td><td>74.27</td><td>65.53</td><td>31.31</td><td>60.30</td><td>26.20</td><td colspan="2">49.09</td></tr><tr><td rowspan="6">LLaMA 3.2 1B</td><td>Wanda</td><td></td><td></td><td>26.35</td><td></td><td></td><td>23.81</td><td>54.62</td><td>18.00</td><td colspan="2">40.01</td></tr><tr><td></td><td>一</td><td>23.51</td><td>27.69</td><td>65.18</td><td>52.10 52.61</td><td>24.74</td><td>55.64</td><td>20.20</td><td colspan="2">41.33</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>18.84</td><td>25.71</td><td>67.08</td><td>54.29</td><td>26.54</td><td>57.70</td><td>22.00</td><td colspan="2"></td></tr><tr><td>SparseGPT OPTIMA</td><td>SparseGPT SparseGPT</td><td>18.84 18.09</td><td>26.95</td><td>67.85 68.01</td><td>54.59</td><td>25.85</td><td>56.91</td><td>24.00</td><td colspan="2">42.35 42.72</td></tr><tr><td>Thanos</td><td>Thanos</td><td>19.70</td><td>25.37</td><td>67.63</td><td>52.99</td><td>27.13</td><td>54.38</td><td>22.20</td><td colspan="2">41.62</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>18.77</td><td>25.99</td><td>68.23</td><td>53.49</td><td>26.45</td><td>55.88</td><td>21.60</td><td colspan="2">41.94</td></tr><tr><td rowspan="7">LLaMA 3.2 3B</td><td>Dense</td><td></td><td>7.81</td><td>54.13</td><td>76.55</td><td>74.28</td><td>42.75</td><td>69.38</td><td>30.60</td><td colspan="2">57.95</td></tr><tr><td>Wanda</td><td>1</td><td>12.92</td><td>40.79</td><td>72.03</td><td>65.45</td><td>32.34</td><td>63.69</td><td>25.40</td><td colspan="2">49.95</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>12.24</td><td>43.11</td><td>72.47</td><td>66.50</td><td>33.53</td><td>66.38</td><td>26.20</td><td colspan="2">51.37</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>12.32</td><td>37.96</td><td>73.45</td><td>65.19</td><td>33.02</td><td>66.38</td><td>25.20</td><td colspan="2">50.20</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>12.43</td><td>40.54</td><td>73.45</td><td>66.37</td><td>35.07</td><td>66.69</td><td>26.20</td><td colspan="2">51.39</td></tr><tr><td>Thanos</td><td>Thanos</td><td>12.26</td><td>40.11</td><td>72.80</td><td>64.77</td><td>32.85</td><td>67.72</td><td>26.60</td><td colspan="2">50.81</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>12.40</td><td>41.51</td><td>73.23</td><td>65.07</td><td>34.39</td><td>67.25</td><td>27.00</td><td colspan="2">51.41</td></tr><tr><td rowspan="7">Gemma 3 1B</td><td>Dense</td><td>1</td><td>14.17</td><td>24.95</td><td>74.81</td><td>71.93</td><td>35.41</td><td>58.72</td><td>28.80</td><td colspan="2">49.10</td></tr><tr><td>Wanda</td><td></td><td>32.96</td><td>22.97</td><td>67.19</td><td>61.03</td><td>26.37</td><td>55.72</td><td>20.00</td><td colspan="2">42.21</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>28.90</td><td>23.96</td><td>69.48</td><td>62.84</td><td>28.58</td><td>56.83</td><td>22.40</td><td colspan="2">44.01</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>28.34</td><td>24.85</td><td>68.88</td><td>60.94</td><td>26.62</td><td>55.49</td><td>21.40</td><td colspan="2">43.03</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>27.35</td><td>25.73</td><td>69.75</td><td>60.90</td><td>27.82</td><td>56.35</td><td>22.00</td><td colspan="2">43.76</td></tr><tr><td>Thanos</td><td>Thanos</td><td>28.65</td><td>23.09</td><td>69.75</td><td>62.16</td><td>27.99</td><td>56.51</td><td>23.80</td><td colspan="2">43.88</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>28.14</td><td>24.70</td><td>69.64</td><td>63.43</td><td>27.39</td><td>55.96</td><td>23.20</td><td colspan="2">44.05</td></tr><tr><td rowspan="7">Gemma 2 2B</td><td>Dense</td><td>一</td><td>68.69</td><td>49.33</td><td>78.24</td><td>80.22</td><td>46.93</td><td>68.82</td><td>31.40</td><td colspan="2">59.16</td></tr><tr><td>Wanda</td><td></td><td>327.45</td><td>34.17</td><td>74.16</td><td>69.78</td><td>34.30</td><td>62.83</td><td>26.40</td><td colspan="2">50.27</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>215.63</td><td>34.86</td><td>73.99</td><td>71.38</td><td>32.59</td><td>61.96</td><td>25.80</td><td colspan="2">50.10</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>234.68</td><td>35.59</td><td>73.61</td><td>69.99</td><td>34.22</td><td>65.82</td><td>28.20</td><td colspan="2">51.24</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>241.09</td><td>37.59</td><td>73.83</td><td>70.62</td><td>35.07</td><td>64.72</td><td>27.80</td><td colspan="2">51.60</td></tr><tr><td>Thanos</td><td>Thanos</td><td>276.97</td><td>30.62</td><td>73.18</td><td>67.72</td><td>33.62</td><td>63.22</td><td>26.80</td><td colspan="2">49.19</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>250.15</td><td>32.72</td><td>73.72</td><td>68.81</td><td>34.13</td><td>63.85</td><td>26.40</td><td colspan="2">49.94</td></tr><tr><td rowspan="7">LLaMA 3.1 8B</td><td>Dense</td><td>一</td><td>5.84</td><td>63.57</td><td>80.09</td><td>81.44</td><td>51.37</td><td>73.48</td><td>33.40</td><td>63.89</td></tr><tr><td>Wanda</td><td></td><td>13.54</td><td>43.42</td><td>73.18</td><td>69.23</td><td>35.32</td><td>67.32</td><td>25.80</td><td>52.38</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>12.58</td><td>45.45</td><td>73.39</td><td>69.57</td><td>36.18</td><td>68.90</td><td>25.20</td><td>53.12</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>12.37</td><td>45.62</td><td>73.83</td><td>69.15</td><td>35.84</td><td>69.22</td><td>25.60</td><td>53.21</td></tr><tr><td></td><td>SparseGPT OPTIMA</td><td>12.54</td><td>46.04</td><td>73.72</td><td>69.95</td><td>36.77</td><td>69.61</td><td>27.00</td><td>53.85</td></tr><tr><td>Thanos</td><td>Thanos</td><td>12.66</td><td>44.39</td><td>73.94</td><td>69.57</td><td>36.18</td><td>68.90</td><td>25.20</td><td>53.03</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>12.80</td><td>44.41</td><td>74.05</td><td>69.95</td><td>36.43</td><td>68.59</td><td>25.60</td><td>53.17</td></tr><tr><td rowspan="11">LLaMA 3.2 1B</td><td>Dense</td><td></td><td>9.75</td><td>36.92</td><td>74.27</td><td>65.53</td><td>31.31</td><td>60.30</td><td>26.20</td><td>49.09</td></tr><tr><td>Wanda</td><td></td><td>30.43</td><td>23.32</td><td>63.55</td><td>47.56</td><td>23.63</td><td>55.25</td><td>15.00</td><td>38.05</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>48.23</td><td>24.80</td><td>66.10</td><td>58.04</td><td>23.55</td><td>55.25</td><td>19.80</td><td>41.26</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>21.98</td><td>23.05</td><td>65.45</td><td>52.15</td><td>25.17</td><td>57.62</td><td>17.60</td><td>40.17</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>21.40</td><td>23.40</td><td>65.72</td><td>52.78</td><td>25.51</td><td>57.06</td><td>18.60</td><td>40.51</td></tr><tr><td>Thanos</td><td>Thanos</td><td>22.80</td><td>24.09</td><td>65.67</td><td>51.68</td><td>25.00</td><td>52.96</td><td>17.60</td><td>39.50</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>22.26</td><td>23.41</td><td>65.34</td><td>52.22</td><td>23.72</td><td>55.96</td><td>16.80</td><td>39.58</td></tr><tr><td>ProxSparse</td><td></td><td>41.95</td><td>23.64</td><td>61.21</td><td>42.38</td><td>22.53</td><td>53.67</td><td>16.00</td><td>36.57</td></tr><tr><td></td><td>ProxSparse OPTIMA</td><td>28.53</td><td>23.07</td><td>63.38</td><td>47.90</td><td>22.53</td><td>54.78</td><td>16.40</td><td>38.01</td></tr><tr><td rowspan="10">LLaMA 3.2 3B</td><td>Dense</td><td></td><td>7.81</td><td>54.13</td><td>76.55</td><td>74.28</td><td>42.75</td><td>69.38</td><td>30.60</td><td>57.95</td></tr><tr><td>Wanda</td><td></td><td>18.51</td><td>34.30</td><td>70.73</td><td>60.69</td><td>30.72</td><td>61.17</td><td>24.80</td><td>47.07</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>16.64</td><td>37.15</td><td>70.78</td><td>61.95</td><td>31.14</td><td>62.51</td><td>24.60 25.00</td><td>48.02 48.27</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>16.19</td><td>36.13 38.03</td><td>70.29 70.84</td><td>63.01</td><td>30.46</td><td>64.72</td><td>25.60</td><td>48.92</td></tr><tr><td></td><td>SparseGPT OPTIMA</td><td>16.36</td><td></td><td></td><td>63.17</td><td>32.17</td><td>63.69</td><td></td><td></td></tr><tr><td>Thanos</td><td>Thanos</td><td>16.24</td><td>35.55</td><td>70.35</td><td>61.28</td><td>29.78</td><td>63.30</td><td>24.20 25.60</td><td>47.41</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>16.49</td><td>35.72</td><td>70.62</td><td>62.04</td><td>30.97</td><td>63.22</td><td></td><td>48.03</td></tr><tr><td>ProxSparse</td><td></td><td>19.50</td><td>24.66</td><td>68.12</td><td>56.31</td><td>27.82</td><td>58.56</td><td>20.00</td><td>42.58</td></tr><tr><td>ProxSparse OPTIMA</td><td></td><td>18.28</td><td>31.76</td><td>69.53</td><td>60.27</td><td>28.84</td><td>60.30</td><td>20.60</td><td>45.22</td></tr><tr><td rowspan="9">Dense Wanda Gemma</td><td></td><td></td><td>14.17</td><td>24.95</td><td>74.81 65.51</td><td>71.93</td><td>35.41</td><td>58.72</td><td>28.80</td><td>49.10</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>60.74 23.25</td><td>23.74 23.25</td><td>63.38</td><td>56.78 51.14</td><td>22.35 24.06</td><td>52.72 54.30</td><td>19.80 18.20</td><td>40.15 39.06</td></tr><tr><td>SparseGPT SparseGPT</td><td></td><td>44.87</td><td>24.83</td><td>66.76</td><td>57.70</td><td>23.29</td><td>55.96</td><td>19.40</td><td>41.32</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>42.66</td><td>25.11</td><td>66.27</td><td>58.96</td><td>23.89</td><td>55.80</td><td>20.60</td><td>41.77</td></tr><tr><td>Thanos</td><td>Thanos</td><td>48.50</td><td>25.23</td><td>65.89</td><td>59.30</td><td>23.12</td><td>53.59</td><td>20.80</td><td>41.32</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>44.91</td><td>25.83</td><td>66.00</td><td>58.63</td><td>23.29</td><td>54.70</td><td>20.00</td><td>41.41</td></tr><tr><td>ProxSparse</td><td></td><td>41.02</td><td>23.01</td><td>66.00</td><td>54.34</td><td>22.44</td><td>55.88</td><td>20.20</td><td>40.31</td></tr><tr><td>ProxSparse OPTIMA</td><td></td><td>52.99</td><td>24.13</td><td>64.74</td><td>53.70</td><td>22.61</td><td>52.25</td><td>17.00</td><td>39.07</td></tr><tr><td rowspan="10">Gemma 22B</td><td>Dense</td><td></td><td>68.69</td><td>49.33</td><td>78.24</td><td>80.22</td><td>46.93</td><td>68.82</td><td>31.40</td><td>59.16</td></tr><tr><td>Wanda Wanda</td><td></td><td>421.01</td><td>34.34</td><td>71.33</td><td>68.10</td><td>30.97</td><td>61.40</td><td>26.40</td><td>48.76</td></tr><tr><td></td><td>OPTIMA</td><td>229.69</td><td>34.44</td><td>71.87</td><td>68.90</td><td>33.87</td><td>62.27</td><td>25.00</td><td>49.39</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>251.71</td><td>32.84</td><td>71.76</td><td>68.73 67.47</td><td>32.42 32.17</td><td>61.88 63.38</td><td>23.40 24.40</td><td>48.51 48.66</td></tr><tr><td>SparseGPT OPTIMA Thanos Thanos</td><td></td><td>227.99 256.58</td><td>32.77 31.02</td><td>71.76 70.73</td></tr></table>

Table 5.2: Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 50% unstructured sparsity. OPTIMA consistently improves the accuracy of the models across diferent tasks.

Table 5.3: Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 2:4 sparsity. In this experiment, only the layers in the MLP part of the transformer are pruned, and the self-attention layers are dense, resulting in an end-to-end sparsity ratio of 38% to 41%. OPTIMA consistently improves the accuracy of the models across diferent tasks. Please note that ProxSparse pruning is limited to 2:4 sparsity, and hence our unstructured sparsity experiments do not include it.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Mask Selection</td><td rowspan="2">Weight Update</td><td rowspan="2">Perplexity</td><td colspan="7">Metrics (%)</td></tr><tr><td>MMLU</td><td>PIQA</td><td>Arc-E</td><td>Arc-C</td><td>Wino</td><td>OpenQA|</td><td>Average</td></tr><tr><td rowspan="8">LLaMA 3.1 8B</td><td>Dense</td><td>一</td><td>5.84</td><td>63.57</td><td>80.09</td><td>81.44</td><td>51.37</td><td>73.48</td><td>33.40</td><td>63.89</td></tr><tr><td>Wanda</td><td></td><td>21.65</td><td>31.98</td><td>69.53</td><td>61.11</td><td>27.30</td><td>61.09</td><td>21.40</td><td>45.40</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>17.56</td><td>33.96</td><td>71.60</td><td>63.76</td><td>29.35</td><td>66.06</td><td>22.60</td><td>47.89</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>15.44</td><td>35.32</td><td>71.55</td><td>62.88</td><td>31.66</td><td>68.19</td><td>24.20</td><td>48.96</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>15.64</td><td>32.44</td><td>71.87</td><td>63.97</td><td>33.11</td><td>67.56</td><td>24.60</td><td>48.93</td></tr><tr><td>Thanos</td><td>Thanos</td><td>15.91</td><td>35.22</td><td>72.09</td><td>65.28</td><td>33.19</td><td>67.40</td><td>23.40</td><td>49.43</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>16.09</td><td>34.48</td><td>72.03</td><td>64.69</td><td>33.02</td><td>68.51</td><td>22.80</td><td>49.25</td></tr><tr><td>Dense</td><td></td><td>9.75</td><td>36.92</td><td></td><td>65.53</td><td></td><td>60.30</td><td></td><td></td></tr><tr><td rowspan="6">LLaMA 3.2 1B</td><td>Wanda</td><td>一</td><td>71.53</td><td>22.95</td><td>74.27</td><td>39.48</td><td>31.31 18.77</td><td>50.43</td><td>26.20 12.20</td><td>49.09 33.92</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>41.50</td><td>23.52</td><td>59.68</td><td>44.53</td><td>20.65</td><td>52.57</td><td>14.80</td><td>36.45</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>48.00</td><td>23.02</td><td>62.62</td><td>43.48</td><td>21.76</td><td>52.09</td><td>17.40</td><td>36.64</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>38.05</td><td>22.95</td><td>62.08 63.38</td><td>43.52</td><td>20.48</td><td>53.28</td><td>19.60</td><td>37.20</td></tr><tr><td>Thanos</td><td>Thanos</td><td>46.78</td><td>23.25</td><td>62.57</td><td>44.49</td><td>21.59</td><td>53.20</td><td>16.60</td><td>36.95</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>40.54</td><td>23.02</td><td>62.95</td><td>44.53</td><td>21.67</td><td>53.91</td><td>17.40</td><td>37.25</td></tr><tr><td rowspan="8">LLaMA 3.2 3B</td><td>Dense</td><td></td><td>7.81</td><td>54.13</td><td></td><td>74.28</td><td>42.75</td><td>69.38</td><td>30.60</td><td>57.95</td></tr><tr><td>Wanda</td><td></td><td>31.13</td><td>25.53</td><td>76.55 65.23</td><td>47.90</td><td>22.70</td><td>55.25</td><td>16.00</td><td>38.77</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>23.56</td><td>31.20</td><td>67.41</td><td>53.96</td><td>24.57</td><td>59.51</td><td>19.80</td><td>42.74</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>22.00</td><td>31.27</td><td>69.37</td><td>53.66</td><td>26.02</td><td>61.33</td><td>21.00</td><td>43.78</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>22.67</td><td>29.58</td><td>68.77</td><td>54.80</td><td>24.74</td><td>62.35</td><td>20.60</td><td>43.47</td></tr><tr><td>Thanos</td><td>Thanos</td><td>22.48</td><td>29.23</td><td>67.63</td><td>55.01</td><td>26.02</td><td>57.85</td><td>19.20</td><td>42.49</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>22.28</td><td>31.43</td><td>67.90</td><td>55.26</td><td>24.91</td><td>59.67</td><td>20.60</td><td>43.30</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="7">Gemma 3 1B</td><td>Dense Wanda</td><td>一</td><td>14.17</td><td>24.95</td><td>74.81</td><td>71.93</td><td>35.41</td><td>58.72</td><td>28.80</td><td>49.10</td></tr><tr><td>Wanda</td><td>1</td><td>90.48</td><td>23.04</td><td>62.19</td><td>49.75</td><td>18.60</td><td>50.99</td><td>15.20 16.40</td><td>36.63</td></tr><tr><td></td><td>OPTIMA</td><td>64.79</td><td>23.34 24.58</td><td>64.09</td><td>52.86</td><td>20.48</td><td>51.93</td><td></td><td>38.18</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>60.91</td><td>23.72</td><td>65.34</td><td>51.98</td><td>21.93</td><td>51.14</td><td>16.60</td><td>38.60</td></tr><tr><td>Thanos</td><td>SparseGPT OPTIMA</td><td>56.27</td><td></td><td>66.21</td><td>52.44</td><td>22.53</td><td>52.96</td><td>17.60</td><td>39.24</td></tr><tr><td>Thanos</td><td>Thanos</td><td>62.22</td><td>24.62</td><td>64.53</td><td>52.86</td><td>20.65</td><td>52.17</td><td>18.80</td><td>38.94</td></tr><tr><td></td><td>OPTIMA</td><td>56.78</td><td>24.44</td><td>64.85</td><td>55.18</td><td>22.01</td><td>54.85</td><td>19.80</td><td>40.19</td></tr><tr><td rowspan="7">Gemma 2 2B</td><td>Dense</td><td>1</td><td>68.69</td><td>49.33</td><td>78.24</td><td>80.22</td><td>46.93</td><td>68.82</td><td>31.40</td><td>59.16</td></tr><tr><td>Wanda Wanda</td><td></td><td>757.47</td><td>23.36</td><td>65.78</td><td>56.10</td><td>21.59</td><td>52.64</td><td>19.80 20.00</td><td>39.88 41.46</td></tr><tr><td></td><td>OPTIMA</td><td>435.10</td><td>24.37</td><td>66.59</td><td>58.50</td><td>21.93</td><td>57.38</td><td></td><td></td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>488.25</td><td>24.49</td><td>68.50</td><td>57.45</td><td>25.00</td><td>58.96</td><td>25.00</td><td>43.23</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>451.46</td><td>25.89</td><td>68.88</td><td>58.50</td><td>26.28</td><td>58.01</td><td>24.20</td><td>43.63</td></tr><tr><td>Thanos</td><td>Thanos</td><td>523.61</td><td>23.69</td><td>68.23</td><td>58.12</td><td>23.89</td><td>58.33</td><td>21.20</td><td>42.24</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>497.75</td><td>23.12</td><td>67.74</td><td>57.07</td><td>23.38</td><td>59.27</td><td>20.60</td><td>41.86</td></tr><tr><td rowspan="7">Qwen 2.5 0.5B</td><td>Dense</td><td></td><td>13.08</td><td>47.36</td><td>69.97</td><td>64.18</td><td>29.18</td><td>55.80</td><td>24.40</td><td>48.48</td></tr><tr><td>Wanda</td><td></td><td>24.00</td><td>30.52</td><td>64.09</td><td>57.41</td><td>24.06</td><td>54.38</td><td>19.80</td><td>41.71</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>22.70</td><td>26.14</td><td>64.58</td><td>57.79</td><td>25.26</td><td>56.04</td><td>22.00</td><td>41.97</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>20.33</td><td>29.38</td><td>64.74</td><td>56.52</td><td>24.15</td><td>56.20</td><td>20.60</td><td>41.93</td></tr><tr><td></td><td>SparseGPT OPTIMA</td><td>19.54</td><td>27.68</td><td>65.13</td><td>56.99</td><td>24.66</td><td>55.33</td><td>20.60</td><td>41.73</td></tr><tr><td>Thanos</td><td>Thanos</td><td>20.85</td><td>28.94</td><td>65.40</td><td>55.93</td><td>24.40</td><td>56.35</td><td>21.60</td><td>42.10</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>20.41</td><td>30.00</td><td>64.69</td><td>56.10</td><td>24.40</td><td>55.41</td><td>22.20</td><td>42.13</td></tr><tr><td rowspan="7">Qwen 2.5 1.5B</td><td>Dense</td><td></td><td>9.28</td><td>59.70</td><td>75.73</td><td>75.34</td><td>40.96</td><td>63.14</td><td>32.20</td><td>57.84</td></tr><tr><td>Wanda</td><td></td><td>14.45</td><td>44.76</td><td>71.22</td><td>66.62</td><td>31.74</td><td>59.91</td><td>24.80</td><td>49.84</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>12.85</td><td>45.61</td><td>72.36</td><td>66.62</td><td>32.34</td><td>61.80</td><td>24.60</td><td>50.55</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>13.09</td><td>46.80</td><td>71.65</td><td>66.75</td><td>33.62</td><td>62.27</td><td>25.60</td><td>51.12</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>12.76</td><td>46.96</td><td>71.82</td><td>65.45</td><td>33.02</td><td>61.80</td><td>26.20</td><td>50.87</td></tr><tr><td>Thanos</td><td>Thanos</td><td>13.17</td><td>48.40</td><td>71.76</td><td>66.84</td><td>33.70</td><td>62.83</td><td>27.20</td><td>51.79</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>12.89</td><td>48.21</td><td>72.03</td><td>67.26</td><td>33.53</td><td>62.04</td><td>26.20</td><td>51.55</td></tr><tr><td rowspan="7">Qwen 2.5 3B</td><td>Dense</td><td></td><td>8.03</td><td></td><td></td><td>77.31</td><td>44.88</td><td>68.43</td><td></td><td></td></tr><tr><td>Wanda</td><td></td><td>11.39</td><td>65.00 49.09</td><td>78.35 73.23</td><td>71.46</td><td>38.48</td><td>65.43</td><td>29.20 26.40</td><td>60.53</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>10.59</td><td>52.00</td><td>74.37</td><td>72.18</td><td>38.05</td><td>66.77</td><td>28.60</td><td>54.02</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>10.74</td><td>52.49</td><td>74.65</td><td>71.34</td><td>36.86</td><td>64.64</td><td>28.20</td><td>55.33</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>10.57</td><td>53.92</td><td>75.35</td><td>70.83</td><td>38.31</td><td>66.14</td><td>29.60</td><td>54.70 55.69</td></tr><tr><td>Thanos</td><td>Thanos</td><td>10.64</td><td>52.61</td><td>75.52</td><td>70.54</td><td>36.69</td><td>66.61</td><td>28.40</td><td></td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>10.52</td><td>52.11</td><td>75.46</td><td>70.12</td><td>37.29</td><td>66.69</td><td>28.20</td><td>55.06 54.98</td></tr><tr><td rowspan="7">Qwen 2.5 7B</td><td>Dense</td><td></td><td>6.85</td><td>71.76</td><td>78.73</td><td>80.51</td><td>48.38</td><td>72.61</td><td>33.40</td><td>64.23</td></tr><tr><td>Wanda</td><td></td><td>8.62</td><td>65.89</td><td>77.31</td><td>75.08</td><td>40.53</td><td>70.17</td><td>30.80</td><td>59.96</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>8.33</td><td>66.17</td><td>77.69</td><td>76.43</td><td>42.66</td><td>71.27</td><td>30.60</td><td>60.80</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>8.42</td><td>66.09</td><td>78.07</td><td>75.34</td><td>42.75</td><td>71.11</td><td>31.00</td><td>60.73</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>8.36</td><td>65.78</td><td>77.64</td><td>75.63</td><td>42.92</td><td>71.51</td><td>31.60</td><td>60.85</td></tr><tr><td>Thanos</td><td>Thanos</td><td>8.49</td><td>66.21</td><td>77.86</td><td>74.71</td><td>42.32</td><td>70.17</td><td>30.40</td><td>60.28</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>8.46</td><td>66.23</td><td>77.58</td><td>76.22</td><td>44.45</td><td>71.19</td><td>31.20</td><td>61.15</td></tr><tr><td rowspan="7">Qwen 2.5 14B</td><td>Dense</td><td>一</td><td>5.30</td><td>77.62</td><td>81.28</td><td>82.24</td><td>55.80</td><td>75.14</td><td>34.40</td><td>67.75</td></tr><tr><td>Wanda</td><td></td><td>7.30</td><td>69.84</td><td>79.16</td><td>81.02</td><td>51.28</td><td>73.72</td><td>34.60</td><td>64.94</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>7.18</td><td>69.29</td><td>79.43</td><td>81.19</td><td>52.30</td><td>73.80</td><td>33.80</td><td>64.97</td></tr><tr><td>SparseGPT SparseGPT</td><td></td><td>7.24</td><td>69.83</td><td>79.60</td><td>80.98</td><td>51.02</td><td>72.93</td><td>32.80</td><td>64.53</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>7.14</td><td>69.71</td><td>79.54</td><td>81.19</td><td>51.79</td><td>73.80</td><td>33.60</td><td>64.94</td></tr><tr><td>Thanos</td><td>Thanos</td><td>7.25</td><td>70.57</td><td>79.87</td><td>80.18</td><td>49.15</td><td>73.09</td><td>32.20</td><td>64.17</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>7.19</td><td>70.16</td><td>79.60</td><td>81.57</td><td>51.37</td><td>73.48</td><td>33.00</td><td>64.86</td></tr><tr><td rowspan="7">Qwen 2.5 0.5B</td><td>Dense</td><td></td><td>13.08</td><td>47.36</td><td>69.97</td><td>64.18</td><td>29.18</td><td>55.80</td><td>24.40</td><td>48.48</td></tr><tr><td>Wanda</td><td></td><td>83.42</td><td>23.02</td><td>59.96</td><td>43.81</td><td>18.09</td><td>50.28</td><td>12.80</td><td>34.66</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>51.97</td><td>23.16</td><td>60.72</td><td>46.25</td><td>20.14</td><td>51.78</td><td>16.40</td><td>36.41</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>40.56</td><td>22.90</td><td>61.59</td><td>48.40</td><td>21.25</td><td>52.80</td><td>16.80</td><td>37.29</td></tr><tr><td></td><td>SparseGPT OPTIMA</td><td>36.77</td><td>23.06</td><td>62.13</td><td>48.74</td><td>21.33</td><td>53.99</td><td>17.40</td><td>37.77</td></tr><tr><td>Thanos</td><td>Thanos</td><td>44.29</td><td>23.78</td><td>62.02</td><td>48.65</td><td>21.33</td><td>52.25</td><td>17.80</td><td>37.64</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>41.92</td><td>23.59</td><td>61.86</td><td>46.80</td><td>22.35</td><td>53.75</td><td>19.60</td><td>37.99</td></tr><tr><td rowspan="7">Qwen 2.5 1.5B</td><td>Dense</td><td></td><td>9.28</td><td>59.70</td><td>75.73</td><td>75.34</td><td>40.96</td><td>63.14</td><td>32.20</td><td>57.84</td></tr><tr><td>Wanda</td><td></td><td>58.38</td><td>27.25</td><td>65.18</td><td>54.50</td><td>24.74</td><td>53.04</td><td>17.20</td><td>40.32</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>23.81</td><td>30.99</td><td>66.87</td><td>56.44</td><td>24.91</td><td>56.83</td><td>18.40</td><td>42.41</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>21.92</td><td>33.56</td><td>67.36</td><td>58.08</td><td>27.47</td><td>57.14</td><td>21.60</td><td></td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>19.35</td><td>31.44</td><td>67.79</td><td>56.27</td><td>27.22</td><td>59.27</td><td>22.40</td><td>44.20 44.07</td></tr><tr><td>Thanos</td><td>Thanos</td><td>27.07</td><td>33.66</td><td>67.63</td><td>57.49</td><td>27.73</td><td>56.67</td><td>20.60</td><td>43.96</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>23.64</td><td>35.97</td><td>67.14</td><td>57.66</td><td>26.19</td><td>58.09</td><td>20.80</td><td>44.31</td></tr><tr><td rowspan="7">Qwen 2.5 3B</td><td>Dense</td><td></td><td>8.03</td><td></td><td>78.35</td><td>77.31</td><td>44.88</td><td>68.43</td><td></td><td></td></tr><tr><td>Wanda</td><td></td><td>22.06</td><td>65.00 28.07</td><td>67.14</td><td>60.86</td><td>27.39</td><td>58.17</td><td>29.20 20.40</td><td>60.53 43.67</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>15.67</td><td>37.22</td><td>70.24</td><td>63.55</td><td>30.89</td><td>61.64</td><td>23.60</td><td></td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>14.82</td><td>43.16</td><td>71.60</td><td>64.35</td><td>32.59</td><td>63.30</td><td>23.20</td><td>47.86</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>14.50</td><td>40.25</td><td>72.20</td><td>64.90</td><td>33.62</td><td>63.69</td><td>24.00</td><td>49.70 49.78</td></tr><tr><td>Thanos</td><td>Thanos</td><td>14.90</td><td>40.76</td><td>71.38</td><td>63.26</td><td>30.63</td><td>61.25</td><td>23.40</td><td>48.45</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>14.42</td><td>42.58</td><td>71.27</td><td>64.73</td><td>32.68</td><td>63.61</td><td>25.00</td><td>49.98</td></tr><tr><td rowspan="7">Qwen 2.5 7B</td><td>Dense</td><td></td><td>6.85</td><td>71.76</td><td>78.73</td><td>80.51</td><td>48.38</td><td>72.61</td><td>33.40</td><td>64.23</td></tr><tr><td>Wanda</td><td></td><td>14.09</td><td>54.58</td><td>72.03</td><td>71.68</td><td>37.03</td><td>66.46</td><td>25.40</td><td>54.53</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>11.15</td><td>55.49</td><td>73.99</td><td>73.86</td><td>37.88</td><td>67.96</td><td>26.20</td><td>55.90</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>10.86</td><td>56.63</td><td>74.92</td><td>73.36</td><td>40.61</td><td>67.25</td><td>25.80</td><td>56.43</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>10.53</td><td>55.55</td><td>75.46</td><td>73.78</td><td>40.70</td><td>66.93</td><td>26.60</td><td>56.50</td></tr><tr><td>Thanos</td><td>Thanos</td><td>11.07</td><td>59.54</td><td>74.70</td><td>73.44</td><td>40.44</td><td>69.22</td><td>26.40</td><td>57.29</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>10.74</td><td>58.90</td><td>75.35</td><td>72.69</td><td>40.10</td><td>69.46</td><td>27.00</td><td>57.25</td></tr><tr><td rowspan="7">Qwen 2.5 14B</td><td>Dense 一</td><td></td><td>5.30</td><td>77.62</td><td>81.28</td><td>82.24</td><td>55.80</td><td>75.14</td><td>34.40</td><td>67.75</td></tr><tr><td>Wanda</td><td></td><td>11.16</td><td>61.38</td><td>75.41</td><td>74.12</td><td>42.15</td><td>71.51</td><td>29.20</td><td>58.96</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>9.69</td><td>61.74</td><td>75.57</td><td>75.34</td><td>41.98</td><td>73.09</td><td>29.40</td><td>59.52</td></tr><tr><td>SparseGPT SparseGPT</td><td></td><td>9.22</td><td>62.83</td><td>76.66</td><td>76.18</td><td>44.45</td><td>72.14</td><td>29.60</td><td>60.31</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>8.97</td><td>62.22</td><td>76.93</td><td>76.47</td><td>44.54</td><td>71.67</td><td>29.00</td><td>60.14</td></tr><tr><td>Thanos</td><td>Thanos</td><td>9.14</td><td>63.03</td><td>77.20</td><td>76.05</td><td>43.77</td><td>71.98</td><td>29.80</td><td>60.31</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>8.99</td><td>60.30</td><td>76.39</td><td>76.30</td><td>43.77</td><td>72.14</td><td>30.60</td><td>59.92</td></tr><tr><td rowspan="7">Qwen 2.5 0.5B</td><td>Dense</td><td></td><td>13.08</td><td>47.36</td><td>69.97</td><td>64.18</td><td>29.18</td><td>55.80</td><td>24.40</td><td>48.48</td></tr><tr><td>Wanda</td><td></td><td>41.30</td><td>27.88</td><td>61.75</td><td>48.99</td><td>23.38</td><td>52.72</td><td>14.00</td><td>38.12</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>27.61</td><td>25.57</td><td>63.76</td><td>51.18</td><td>22.18</td><td>53.28</td><td>15.80</td><td>38.63</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>27.15</td><td>24.83</td><td>62.79</td><td>49.41</td><td>22.35</td><td>52.33</td><td>17.20</td><td>38.15</td></tr><tr><td></td><td>SparseGPT OPTIMA</td><td>25.77</td><td>23.38</td><td>62.95</td><td>51.30</td><td>22.95</td><td>54.70</td><td>17.00</td><td>38.71</td></tr><tr><td>Thanos</td><td>Thanos</td><td>27.58</td><td>24.31</td><td>62.68</td><td>49.92</td><td>21.42</td><td>51.78</td><td>16.60</td><td>37.78</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>26.26</td><td>23.60</td><td>62.79</td><td>51.22</td><td>22.18</td><td>54.30</td><td>16.60</td><td>38.45</td></tr><tr><td rowspan="7">Qwen 2.5 1.5B</td><td>Dense</td><td></td><td>9.28</td><td>59.70</td><td>75.73</td><td>75.34</td><td>40.96</td><td>63.14</td><td>32.20</td><td>57.84</td></tr><tr><td>Wanda</td><td>一</td><td>21.92</td><td>39.95</td><td>67.25</td><td>61.11</td><td>28.84</td><td>58.09</td><td>20.80</td><td>46.01</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>17.14</td><td>39.96</td><td>69.26</td><td>62.33</td><td>30.29</td><td>58.72</td><td>23.00</td><td>47.26</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>17.24</td><td>41.05</td><td>69.64</td><td>63.09</td><td>30.63</td><td>61.17</td><td>23.00</td><td>48.10</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>16.52</td><td>35.64</td><td>69.91</td><td>62.75</td><td>29.35</td><td>60.93</td><td>23.00</td><td>46.93</td></tr><tr><td>Thanos</td><td>Thanos</td><td>17.56</td><td>42.83</td><td>68.55</td><td>61.53</td><td>29.10</td><td>58.96</td><td>21.40</td><td>47.06</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>16.83</td><td>40.31</td><td>69.75</td><td>62.67</td><td>30.38</td><td>58.56</td><td>25.00</td><td>47.78</td></tr><tr><td rowspan="7">Qwen 2.5 3B</td><td>Dense</td><td></td><td>8.03</td><td></td><td></td><td>77.31</td><td>44.88</td><td>68.43</td><td></td><td></td></tr><tr><td>Wanda</td><td></td><td>17.14</td><td>65.00 46.68</td><td>78.35 70.08</td><td>64.77</td><td>31.66</td><td>61.48</td><td>29.20 22.20</td><td>60.53 49.48</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>14.08</td><td>46.55</td><td>71.44</td><td>64.90</td><td>31.31</td><td>64.17</td><td>25.40</td><td></td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>14.06</td><td>43.79</td><td>72.03</td><td>66.16</td><td>31.31</td><td>64.96</td><td>25.20</td><td>50.63</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>13.57</td><td>43.36</td><td>71.44</td><td>66.62</td><td>31.91</td><td>65.27</td><td></td><td>50.58</td></tr><tr><td>Thanos</td><td>Thanos</td><td>14.35</td><td>41.41</td><td>71.49</td><td>61.70</td><td>29.27</td><td>64.01</td><td>27.00 25.20</td><td>50.93</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>13.75</td><td>43.56</td><td>70.73</td><td>64.06</td><td>30.97</td><td>64.09</td><td>25.40</td><td>48.85 49.80</td></tr><tr><td rowspan="7">Qwen 2.5 7B</td><td>Dense</td><td></td><td>6.85</td><td>71.76</td><td>78.73</td><td>80.51</td><td>48.38</td><td>72.61</td><td>33.40</td><td></td></tr><tr><td>Wanda</td><td></td><td>11.47</td><td>61.03</td><td>74.48</td><td>75.34</td><td>42.15</td><td>68.82</td><td>27.80</td><td>64.23 58.27</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>11.80</td><td>53.90</td><td>74.10</td><td>71.97</td><td>38.05</td><td>67.96</td><td>26.40</td><td>55.40</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>10.21</td><td>60.30</td><td>75.57</td><td>75.59</td><td>41.38</td><td>71.51</td><td>28.20</td><td>58.76</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>10.92</td><td>53.90</td><td>74.32</td><td>72.43</td><td>37.46</td><td>69.61</td><td>27.80</td><td>55.92</td></tr><tr><td>Thanos</td><td>Thanos</td><td>10.45</td><td>60.12</td><td>74.54</td><td>75.08</td><td>41.13</td><td>69.93</td><td>28.80</td><td>58.27</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>11.13</td><td>54.90</td><td>73.94</td><td>70.83</td><td>35.15</td><td>69.06</td><td>26.00</td><td>54.98</td></tr><tr><td rowspan="7">Qwen 2.5 14B</td><td>Dense</td><td></td><td>5.30</td><td>77.62</td><td>81.28</td><td>82.24</td><td>55.80</td><td>75.14</td><td>34.40</td><td>67.75</td></tr><tr><td>Wanda</td><td></td><td>9.70</td><td>65.82</td><td>76.99</td><td>76.89</td><td>45.39</td><td>73.56</td><td>31.80</td><td>61.74</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>8.90</td><td>67.03</td><td>77.31</td><td>77.82</td><td>46.76</td><td>74.27</td><td>32.60</td><td>62.63</td></tr><tr><td>SparseGPT SparseGPT</td><td></td><td>9.02</td><td>67.45</td><td>77.58</td><td>77.61</td><td>44.62</td><td>73.88</td><td>32.60</td><td>62.29</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>8.82</td><td>67.33</td><td>77.58</td><td>77.82</td><td>44.28</td><td>73.88</td><td>32.00</td><td>62.15</td></tr><tr><td>Thanos</td><td>Thanos</td><td>9.06</td><td>66.31</td><td>77.64</td><td>77.90</td><td>46.25</td><td>72.77</td><td>31.20</td><td>62.01</td></tr><tr><td>Thanos</td><td>OPTIMA</td><td>8.92</td><td>66.07</td><td>77.97</td><td>77.44</td><td>45.65</td><td>72.93</td><td>31.80</td><td>61.98</td></tr><tr><td rowspan="7">Gemma 3 1B</td><td>Dense</td><td></td><td>14.17</td><td>24.95</td><td>74.81</td><td>71.93</td><td>35.41</td><td>58.72</td><td>28.80</td><td>49.10</td></tr><tr><td>Wanda</td><td></td><td>32.96</td><td>22.97</td><td>67.19</td><td>61.03</td><td>26.37</td><td>55.72</td><td>20.00</td><td>42.21</td></tr><tr><td>Wanda</td><td>ADAM</td><td>29.25</td><td>23.16</td><td>69.04</td><td>62.71</td><td>27.73</td><td>57.46</td><td>22.20</td><td>43.72</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>28.90</td><td>23.96</td><td>69.48</td><td>62.84</td><td>28.58</td><td>56.83</td><td>22.40</td><td>44.01</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>28.34</td><td>24.85</td><td>68.88</td><td>60.94</td><td>26.62</td><td>55.49</td><td>21.40</td><td>43.03</td></tr><tr><td>SparseGPT ADAM</td><td></td><td>27.12</td><td>24.74</td><td>69.53</td><td>61.36</td><td>27.05</td><td>54.78</td><td>22.20</td><td>43.28</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>27.35</td><td>25.73</td><td>69.75</td><td>60.90</td><td>27.82</td><td>56.35</td><td>22.00</td><td>43.76</td></tr><tr><td rowspan="7">OPT 125M</td><td>Dense</td><td></td><td>27.67</td><td>22.85</td><td>62.84</td><td>43.56</td><td>19.45</td><td>49.88</td><td>16.40</td><td>35.83</td></tr><tr><td>Wanda</td><td></td><td>39.50</td><td>22.92</td><td>61.15</td><td>39.94</td><td>19.88</td><td>52.17</td><td>14.00</td><td>35.01</td></tr><tr><td>Wanda</td><td>ADAM</td><td>205.82</td><td>25.63</td><td>57.02</td><td>34.13</td><td>17.66</td><td>50.51</td><td>13.00</td><td>32.99</td></tr><tr><td>Wanda</td><td>OPTIMA</td><td>35.44</td><td>23.02</td><td>61.66</td><td>42.93</td><td>19.11</td><td>50.12</td><td>14.60</td><td>35.24</td></tr><tr><td></td><td>SparseGPT SparseGPT</td><td>36.88</td><td>23.00</td><td>61.97</td><td>40.99</td><td>19.71</td><td>53.59</td><td>14.60</td><td>35.64</td></tr><tr><td>SparseGPT ADAM</td><td></td><td>224.34</td><td>23.15</td><td>56.75</td><td>35.65</td><td>17.49</td><td>47.36</td><td>12.20</td><td>32.10</td></tr><tr><td>SparseGPT OPTIMA</td><td></td><td>35.61</td><td>23.85</td><td>62.37</td><td>42.28</td><td>19.97</td><td>52.25</td><td>15.40</td><td>36.02</td></tr></table>

Table 5.4: Model perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 60% unstructured sparsity. OPTIMA consistently improves the accuracy of the models across diferent tasks.

Table 5.5: Qwen-2.5 family perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 50% unstructured sparsity. OPTIMA consistently improves the accuracy of the models across diferent tasks.

Table 5.6: Qwen-2.5 perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 60% unstructured sparsity. OPTIMA consistently improves the accuracy of the models across diferent tasks.

Table 5.7: Qwen-2.5 perplexity on WikiText2 and accuracy on zero-shot downstream tasks for 2:4 sparsity. In this experiment, only the layers in the MLP part of the transformer are pruned, and the self-attention layers are dense, resulting in an end-to-end sparsity ratio of 38% to 41%. OPTIMA consistently improves the accuracy of the models across diferent tasks. Please note that ProxSparse pruning is limited to 2:4 sparsity, and hence our unstructured sparsity experiments do not include it.

Table 5.8: Comparison of OPTIMA with other optimizers without convergence guarantees (ADAM). ADAM can lead to suboptimal solutions (Gemma 3 1B) or divergence of the model (OPT 125M).

# PATCH: Learnable Tile-Level Hybrid Sparsity for LLMs

## 6.1 Statement of Contributions

The content of this chapter is derived from the paper “PATCH: Learnable Tile-Level Hybrid Sparsity for LLMs” [59]. This research was a collaborative efort with Younes Hourri, and we are co-first authors with equal contribution.

The specific breakdown of contributions is as follows:

• Mohammad Mozafari: Conceived the original concept of learnable hybrid sparsity and implemented the initial version of the codebase. Designed and executed the experiments regarding quantization, Low-Rank Adaptation (LoRA), and the fine-tuning (FT) comparisons.

• Younes Hourri: Formulated the mathematical framework for the tile-level probability distributions (specifically Equation 6.3 and Equation 6.4), extended the codebase with additional capabilities, and conducted the primary pruning experiments. He was also responsible for the hardware acceleration implementation and throughput analysis.

• Joint Contributions: Both authors collaborated closely on the writing and revision of the manuscript. Maryam Mehri Dehnavi held a supervisory position in this work.

## 6.2 Introduction

In the previous chapter, we established OPTIMA to maximize the accuracy of sparse models under a strict ”no-training” constraint. We demonstrated that for a fixed mask structure, one can mathematically solve for the optimal weights. However, OPTIMA, and indeed any layer-wise pruning and weight reconstruction method, eventually hits an accuracy ceiling imposed by the rigidity of the mask itself. To break this ceiling and fully refine the sparsity pillar, we must relax the resource constraints. In this chapter, we transition from the ”zero training” regime of OPTIMA to a learnable regime, utilizing a training budget to move beyond optimizing values within a fixed pattern and address the pattern itself. We need a method that bridges the gap between the high accuracy of flexible unstructured pruning and the hardware acceleration of rigid structured patterns.

Currently, sparsity techniques operate at two extremes, neither of which is suficient for the Compression

Trinity. Unstructured sparsity allows non-zero elements to appear anywhere, theoretically matching dense model accuracy due to its flexibility in allocation [136, 36, 1]. However, its irregular memory access patterns hinder acceleration on modern hardware like GPUs, preventing practical speedups [154, 32]. Conversely, semistructured sparsity ofers practical acceleration but imposes strict layout expectations. Specifically, we focus on the 2:4 sparse pattern supported by NVIDIA Ampere and Hopper architectures, as detailed in Chapter 2. This format requires every block of four contiguous weights to contain at least two zeros to utilize sparse Tensor Cores. This ”one-size-fits-all” approach enforces a uniform 50% sparsity ratio across all layers, failing to account for the varying sensitivity of diferent network components. This often leads to significant accuracy loss when models are pruned using one-shot methods [136, 36, 64, 83] or end-to-end learned masks [33]. Recent studies confirm that sparsity should be allocated non-uniformly for optimal performance [158, 149, 76], yet standard 2:4 sparsity locks the model into a fixed allocation.

To bridge this gap and create a truly flexible sparsity pillar, we propose Pruning with a Learnable Tile-level Configuration for Hybrid Sparsity (PATCH). Instead of forcing the entire model to adhere to a rigid sparse structure, PATCH introduces a hybrid mask that partitions each weight matrix into hardware-friendly tiles. Through a learnable masking process—enabled by our relaxed training budget—PATCH designates each tile as either dense (0% sparsity) or 2:4 sparse (50% sparsity). This adaptive approach allows the matrix to realize an efective global sparsity ratio anywhere between 0% and 50%, dynamically balancing accuracy in critical regions with hardware-friendly sparsity elsewhere. While the PATCH methodology is generalizable to any block-based sparsity pattern (such as 4:8 or custom N:M ratios), we focus our evaluation on 2:4 sparsity as it is the only pattern currently supported by native acceleration on commodity GPUs (see Chapter 2).

We enable this flexibility through two distinct optimization strategies. For maximum accuracy, we employ a joint optimization method that tunes both the sparsity pattern within the 2:4 tiles and the tile-level configurations during training. For scenarios with tighter compute budgets, we ofer a variant that tunes only the location of the dense tiles while keeping the initial 2:4 mask fixed. Importantly, unlike theoretical hybrid methods that never see deployment, PATCH is fully compatible with tile-level sparsity acceleration libraries and compilers such as STOICC [122]. This makes it the first hybrid sparsity method to demonstrate practical speedups on commodity hardware. For instance, on LLaMA-2 7B running on a consumer-grade A6000 GPU, PATCH achieves 1.18×–1.38× end-to-end speedup over the dense baseline while improving accuracy by 0.37%–2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM. Note that this chapter focuses exclusively on refining the sparsity pillar; the integration of PATCH with quantization and low-rank approximation is presented in Chapter 7, where we demonstrate the combined eficacy of the Compression Trinity.

## 6.3 Additional Related Work

## 6.3.1 Pruning methods

Pruning is one of the most widely studied approaches for compressing deep neural networks, with the goal of removing redundant parameters while preserving accuracy. Classical pruning methods can be broadly categorized into local (layer-wise) and global (end-to-end) strategies.

Local pruning. Local approaches prune each layer independently, typically by minimizing reconstruction error within that layer. A seminal example is Optimal Brain Surgeon (OBS) [54, 35], which leverages secondorder information to identify and remove weights while updating the remaining parameters to compensate for loss. While highly principled, the quadratic cost of computing and inverting the Hessian makes OBS infeasible for large models.

![](images/4b7c714cae1f97a85da36be4c77d0c1253f8ce6ee33ccce0425eb29a3da0018e.jpg)  
Figure 6.1: Illustration of the PATCH learning process for generating tile-level hybrid masks. Each tile is parameterized by a learnable distribution and sampled with Gumbel Softmax to produce $\tilde { M } _ { \mathrm { t i l e } }$ . The dense probability is expanded and merged with a 2:4 mask $\tilde { M } _ { 2 : 4 }$ , which can be fixed or jointly learned during training, yielding M<sup>˜</sup> . The final mask assigns each tile to remain dense or follow the 2:4 pattern, enabling flexible sparsity across the weight matrix.

Recent work adapts these ideas to LLM-scale pruning. SparseGPT [36] formulates layer-wise pruning as a sparse regression problem, enabling eficient approximations of OBS that scale to billion-parameter models. Thanos [64] further improves accuracy by employing multi-column approximations to reduce error accumulation. Wanda [136], on the other hand, discards explicit weight updates and instead uses a simple magnitudeactivation criterion with calibration data, yielding competitive quality with extremely fast runtimes. Despite their eficiency, local methods often sufer from limited capacity to recover accuracy since pruning decisions ignore cross-layer dependencies.

Global pruning. Global approaches aim to jointly optimize pruning decisions across layers, typically leading to better overall trade-ofs. Optimal Brain Damage (OBD) [75] is an early global method that estimates weight saliency using the diagonal Hessian. Extensions such as WoodFisher [134] approximate the Hessian via Kronecker factorizations, making computation more tractable but still challenging for modern LLMs [99].

More recent approaches bypass costly second-order computations. MaskLLM [33] formulates pruning as a binary classification task (keep vs. prune) and solves it using standard optimizers such as AdamW [86], achieving strong results even under hardware-friendly structured sparsity (e.g., 2:4). ProxSparse [83] instead adopts a proximal regularization framework, reducing the overhead of MaskLLM while trading of some pruning accuracy. These works highlight the tension between pruning quality and eficiency: global methods often achieve higher accuracy but remain more computationally expensive than simple one-shot local pruning.

## 6.3.2 Complementary compression techniques

Beyond pruning, several orthogonal compression techniques are widely used and can be combined with sparsity for additional gains. Quantization reduces the bit precision of parameters and activations, e.g., from 32-bit floating point to 8- or 4-bit integers, thereby reducing memory footprint and accelerating inference [43, 125].

Low-rank adaptation methods decompose weight matrices into smaller factors, efectively reducing parameter counts while maintaining expressivity. Recent approaches such as LQ-LoRA [48], SLiM [100], and SLoPe [101] demonstrate that low-rank structures can be used both for eficient fine-tuning and for direct model compression.

Finally, knowledge distillation [46] transfers knowledge from a large teacher model to a smaller student, yielding compact models that retain much of the teacher’s performance. These methods are complementary to pruning, and hybrid frameworks that integrate sparsity, quantization, and low-rank factorization represent a promising direction for achieving high compression ratios without sacrificing accuracy.

## 6.4 Preliminaries

Diferentiable Sampling. Sampling from a categorical distribution is inherently non-diferentiable, which poses challenges for gradient-based optimization. The Gumbel Softmax [66] addresses this by combining the Gumbel-Max reparameterization trick together with a softmax relaxation. The reparameterization expresses the sampling process by decoupling the deterministic log-probabilities $p \in \mathbb R ^ { n }$ from the stochastic perturbations $z \in \mathbb { R } ^ { n }$ introduced by Gumbel noise, which emulate random draws from the distribution. The subsequent softmax yields a diferentiable approximation to categorical sampling:

$$
{ \mathrm { G S } } ( p ; \tau ) _ { k } = { \frac { \exp ( ( p _ { k } + z _ { k } ) / \tau ) } { \sum _ { j } \exp ( ( p _ { j } + z _ { j } ) / \tau ) } }\tag{6.1}
$$

where $z _ { k } = - \log ( - \log ( u _ { k } ) )$ with $u _ { k }$ ∼ Uniform(0, 1). The resulting vector ${ \bf G S } ( p ; \tau ) \in \mathbb { R } ^ { n }$ is a soft index vector whose entries $\operatorname { G S } ( p ; \tau ) _ { I }$ represent the relaxed probability of selecting class k.

Additionally, the temperature parameter τ controls the hardness of the sampled index. Lower values of τ yield a more peaked distribution, causing $\mathrm { G S } ( p )$ to converge to a one-hot vector as $\tau  0$

Learnable 2:4 Mask. MaskLLM [33] formulates 2:4 mask selection as a learnable probabilistic process over the six possible patterns. The underlying weights remain fixed, while training shifts the categorical distribution to favor masks that preserve better pruning performance. The mask for each four consecutive elements can be parameterized with a vector $p \in \mathbb { R } ^ { 6 \times 1 }$ . Scaling this vector to a weight matrix $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ will result in $P _ { 2 : 4 } \in \mathbb { R } ^ { 6 \times \frac { d _ { 1 } d _ { 2 } } { 4 } }$ as the mask search parameters. The resulting mask can be computed as in Equation 6.2, where $\tilde { M } _ { 2 : 4 } \in [ 0 , 1 ] ^ { d _ { 1 } \times d _ { 2 } }$ denotes the 2:4 soft mask, obtained as a weighted average over the candidate masks, and $S \in \mathbb { R } ^ { 6 \times 4 }$ is the matrix containing these six candidates as its rows.

$$
\tilde { M } _ { 2 : 4 } = \tt { r e s h a p e } ( G S ( P _ { 2 : 4 } ; \tau , \kappa ) \times S , \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } } )\tag{6.2}
$$

A scaling factor κ is also introduced in Equation 6.1, where it multiplies the logits p before adding the Gumbel noise z, thereby controlling their relative influence. Small κ values let the noise dominate, encouraging exploration across candidate masks, while larger κ values amplify the logits and make the sampling more deterministic.

## 6.5 PATCH

To overcome the rigidity of fixed 50% 2:4 sparsity, we introduce PATCH. PATCH learns a structured mask— optimized on top of frozen weights—that is partitioned into tiles, where each tile decides whether its corresponding weights remain dense or are pruned with a 2:4 pattern. This design preserves accuracy in sensitive regions while exploiting hardware-accelerated sparsity elsewhere. Unlike fixed 2:4 sparsity, which enforces the same pattern across all weights, PATCH adapts at the tile level by assigning dense tiles to critical regions and sparse tiles elsewhere.

Finding the optimal allocation of dense tiles (value 1) and sparse tiles (2:4 pattern) within a mask is a combinatorially dificult problem, as the number of possible configurations grows rapidly with the number of tiles across the LLM. By also modeling this problem as a probabilistic sampling process, and adjusting the probability of each tile (and the 2:4 patterns within sparse tiles), PATCH can eficiently explore the space of configurations and converge toward masks that balance accuracy and sparsity. The mask distributions are learned end-to-end by training the Gumbel–Softmax logits while keeping the model weights frozen. We address this challenge by formulating mask selection as two coupled subproblems: (1) selecting which tiles are dense or sparse, and (2) choosing the 2:4 sparsity pattern within sparse tiles.

Tile-based pruning of LLMs. We associate each parameter matrix $W \in \mathbb { R } ^ { d _ { 1 } \times d _ { 2 } }$ with a grid of tile-level distributions, each parameterized by a learnable logit. Collectively, these form $P _ { \mathrm { t i l e } } \in \mathbb { R } ^ { \frac { d _ { 1 } } { b _ { 1 } } \times \frac { d _ { 2 } } { b _ { 2 } } }$ , where each entry specifies the unnormalized score of keeping the corresponding $b _ { 1 } \times b _ { 2 }$ tile fully dense. To create a two-class distribution (keep dense vs. prune), we concatenate a fixed zero to each logit, yielding $[ P _ { \mathrm { t i l e } } , 0 ] \in$ $\mathbb { R } ^ { \frac { d _ { 1 } } { b _ { 1 } } \times \frac { d _ { 2 } } { b _ { 2 } } \times 2 }$ . After applying Gumbel–Softmax, we broadcast the dense probabilities across their respective $b _ { 1 } \times$ b<sub>2</sub> region (since the weighted average of the two outcomes reduces to $p _ { \mathrm { d e n s e } } \cdot 1 + p _ { \mathrm { p r u n e } } \cdot 0 = p _ { \mathrm { d e n s e } } )$ , so that all elements of a tile receive the same mask value. Formally,

$$
\begin{array} { r } { \tilde { M } _ { \mathrm { t i l e } } = \mathbf { G S } ( [ P _ { \mathrm { t i l e } } , 0 ] ; \tau , \kappa ) _ { : , : , 0 } \otimes \mathbf { 1 } . } \end{array}\tag{6.3}
$$

This yields the tile-level mask $\tilde { M } _ { \mathrm { t i l e } } \in [ 0 , 1 ] ^ { d _ { 1 } \times d _ { 2 } }$ in Equation 6.3, where $\mathbf { 1 } \in \mathbb { R } ^ { b _ { 1 } \times b _ { 2 } }$ is an all-ones matrix and ⊗ denotes the Kronecker product.

Joint optimization with sparse mask. To fully determine the efective sparsity pattern, the tile-level mask must be combined with the fine-grained 2:4 mask. Assuming that the 2:4 mask $\tilde { M } _ { 2 : 4 }$ is generated using Equation 6.2, PATCH combines it with the tile mask $\tilde { M } _ { \mathrm { t i l } \epsilon }$ as shown in Equation 6.4. The resulting soft mask interpolates between dense and sparse behavior: values of $\tilde { M } _ { \mathrm { t i l e } }$ close to one make the tile predominantly dense, while values close to zero shift the tile toward the soft 2:4 mask pattern defined by $\tilde { M } _ { 2 : 4 }$ . Thus, $\tilde { M }$ can be understood as a per-tile weighted average of the dense option and the 2:4 patterns, with $\tilde { M } _ { \mathrm { t i l e } }$ determining the relative contribution of each. An overview of the process is provided in Figure 6.1.

$$
\tilde { M } = \tilde { M } _ { \mathrm { t i l e } } + \left( 1 - \tilde { M } _ { \mathrm { t i l e } } \right) \odot \tilde { M } _ { 2 : 4 }\tag{6.4}
$$

Learning masks with targeted sparsity. PATCH uses a novel regularization term to achieve a flexible 0%– 50% sparsity ratio across the model by controlling the number of dense tiles. Unlike traditional regularization methods like weight decay, which produce non-deterministic sparsity ratios, our term penalizes deviations from the target sparsity, enabling precise control. This global sparsity approach prunes sensitive linear layers less aggressively while setting redundant weight elements to zero, ofering greater flexibility than fixed perlayer sparsity. We directly compare global versus per-layer sparsity regularization in Section 6.7.

Algorithm 4 Joint Tile & 2:4 Mask Learning   
Input: Weight matrix W, tile size $( b _ { 1 } , b _ { 2 } )$ , sparsity target $\rho ,$ training steps $T ,$ loss hyperparameters $\lambda _ { 1 } , \lambda _ { 2 } ,$   
temperature schedule $\{ \tau _ { t } \} _ { t = 1 } ^ { T }$ , scaling schedule $\{ \kappa _ { t } \} _ { t = 1 } ^ { \bar { T } } .$   
Output: Learned pruning masks $\mathbf { M } ^ { \star }$ , pruned weights $\widehat { \mathbf { W } } .$   
1 Initialize tile logits $\mathbf { P } _ { \mathrm { t i l e } } \in \mathbb { R } ^ { \frac { d _ { 1 } } { b _ { 1 } } \times \frac { d _ { 2 } } { b _ { 2 } } } ,$   
2 Initialize $\mathbf { P } _ { \mathrm { t i l e } }$ with one-shot prior.   
3 Initialize diferentiable 2:4 parameters $\mathbf { P } _ { 2 : 4 } \in \mathbb { R } ^ { 6 \times \frac { d _ { 1 } d _ { 2 } } { 4 } }$   
4 for $t = 1 \  \ T$ do   
5 $\tilde { \mathbf { M } } _ { \mathrm { t i l e } }  \mathbf { G S } ( [ \mathbf { P } _ { \mathrm { t i l e } } , 0 ] ; \tau _ { t } , \kappa _ { t } ) _ { : , : , 0 } \otimes \mathbf { 1 } _ { b _ { 1 } \times b _ { 2 } }$ ▷ Dense soft tile mask   
6 $\dot { \mathbf { M } } _ { 2 : 4 }  \mathrm { E }$ quation $_ { 6 . 2 }$ ▷ Diferentiable 2:4 mask   
7 $\tilde { \mathbf { M } } _ { i } \gets \tilde { \mathbf { M } } _ { \mathrm { t i l e } } + \left( 1 - \tilde { \mathbf { M } } _ { \mathrm { t i l e } } \right) \odot \tilde { \mathbf { M } } _ { 2 : 4 }$ ▷ Merge masks   
8 Compute loss:   
$\begin{array} { r l } { \mathcal { L } = \mathcal { L } _ { L M } ( \boldsymbol { x } ; \tilde { \boldsymbol { \mathbf { M } } } \odot \boldsymbol { \mathbf { W } } ) } & { { } + \lambda _ { 1 } \left\| \displaystyle \frac { \sum _ { i } \tilde { \mathbf { M } } _ { i } } { \sum _ { i } \| \boldsymbol { \mathbf { W } } _ { i } \| _ { 0 } } - \boldsymbol { \rho } \right\| _ { 1 } \ - \lambda _ { 2 } \frac { \sum _ { i } \| \tilde { \mathbf { M } } _ { i } \odot \boldsymbol { \mathbf { W } } _ { i } \| _ { 2 } ^ { 2 } } { \sum _ { i } \| \boldsymbol { \mathbf { W } } _ { i } \| _ { 2 } ^ { 2 } } } \end{array}$   
9 Update $\mathbf { P } _ { \mathrm { t i l e } } , \mathbf { P } _ { 2 : 4 }$ via backpropagation.   
10 end for   
11 $\mathbf { M } _ { \mathrm { t i l e } } ^ { \star }  \mathbf { 1 } [ \mathbf { P } _ { \mathrm { t i l e } } > 0 ] \otimes \mathbf { 1 } _ { b _ { 1 } \times b _ { 2 } }$ ▷ Hard tile mask   
12 $\mathbf { M } _ { 2 : 4 } ^ { \star } $ select best 2:4 mask from $\mathbf { P } _ { 2 : 4 } .$   
13 $\mathbf { M } _ { i } ^ { \star } \gets \mathbf { M } _ { \mathrm { t i l e } } ^ { \star } + ( 1 - \mathbf { M } _ { \mathrm { t i l e } } ^ { \star } ) \odot \mathbf { M } _ { 2 : 4 } ^ { \star } .$   
14 $\widehat { \mathbf { W } } \xleftarrow { } \mathbf { W } \odot \mathbf { M } _ { i } ^ { \star }$ ▷ Final pruned weights   
Return: Learned mask $\mathbf { M } ^ { \star }$ , pruned weights $\widehat { \mathbf { W } } .$

Training objective. The overall training objective, as shown in Equation 6.5, of PATCH combines three components: the standard modeling loss, a sparsity regularization term that enforces the target density of the model $\rho ,$ and a weight regularization term (as in MaskLLM) that promotes larger weight magnitudes and gradient propagation. Formally,

$$
\mathcal { L } = \mathcal { L } _ { L M } \left( x ; \tilde { M } _ { i } \odot W _ { i } \right) + \lambda _ { 1 } \left. \frac { \sum _ { i } \tilde { M } _ { i } } { \sum _ { i } \lVert W _ { i } \rVert _ { 0 } } - \rho \right. _ { 1 } - \lambda _ { 2 } \frac { \sum _ { i } \lVert \tilde { M } _ { i } \odot W _ { i } \rVert _ { 2 } ^ { 2 } } { \sum _ { i } \lVert W _ { i } \rVert _ { 2 } ^ { 2 } }\tag{6.5}
$$

Following MaskLLM, we progressively decrease τ and increase κ during training so that the Gumbel-Softmax distribution converges to a clear one-hot choice of mask by the end of training.

Inference. After training, the sign of each logit in $P _ { \mathrm { t i l e } }$ determines the final mask. Since a zero logit is concatenated to represent the sparse class (Equation 6.3), positive values correspond to the dense option, while negative values correspond to the sparse option. The complete procedure is outlined in Algorithm 4.

Memory eficient PATCH. To further reduce overhead, PATCH can be run in a memory-eficient manner by freezing the sparse mask parameters and optimizing only the tile-level decisions. This reduces the number of learnable parameters to $\frac { d _ { 1 } d _ { 2 } } { b _ { 1 } b _ { 2 } }$ . While this lighter formulation limits mask-selection flexibility and can reduce performance as seen in Table 6.7, it makes training feasible under strict memory constraints, such as fitting an

8B model on a single 80GB GPU. We denote this version of PATCH by PATCH<sup>Tile</sup> and the joint optimization version of PATCH by PATCH<sup>Joint</sup>.

## 6.6 Eficient deployment of PATCH

Executing PATCH requires handling hybrid sparse–dense tiles, a capability not supported by existing GPU libraries. Current tools either focus exclusively on dense computation (e.g., cuBLAS [109], dense CUTLASS [23], OpenAI Triton [143]), or restrict support to fixed 2:4 sparsity (e.g., cuSPARSELt [110], sparse CUT-LASS). STOICC [122] lifts these limitations by extending Triton with hybrid tile-level sparsity, making it a suitable backend for accelerating PATCH.

Similar to Triton, STOICC employs an inspector that benchmarks candidate kernel configurations for each sparsity ratio, identifying the most hardware-eficient tile size for the target GPU. On NVIDIA A100 and A6000 GPUs, our experiments show that the optimal configurations are consistently drawn from 128×128 or its subdivisions (e.g., 128×64, 64×128, 64×64). In practice, this means that regardless of the sparsity ratio or the layer shape, the chosen 128×128 granularity guarantees that STOICC’s autotuned tiles can be applied consistently. Unless otherwise specified, we adopt these hardware-friendly tile sizes in all PATCH experiments. Further implementation details are provided in Section D.1.

## 6.7 Experiments

Model, dataset and evaluation. We evaluate PATCH across diverse transformer architectures, including the Qwen-2.5 [118], Gemma 3 [139], and LLaMA-2 [144] and 3 [31] model families, spanning 500M to 8B parameters. Following the dataset size and configurations in MaskLLM [33], masks are trained for 2000 steps with a batch size of 256 on sequences with a length of 4096 tokens from the SlimPajama dataset [135].

Following previous LLM compression work [100, 33], we evaluate the models on eight zero-shot down stream tasks: PIQA [13], ARC-Easy and ARC-Challenge [21], Winogrande [126], OpenBookQA [97], RACE [74], HellaSwag [161], and MMLU [57] using the Language Model Evaluation Harness [42] framework. Additionally, similar to previous work [100, 36, 136], we evaluate the models on a language modeling task using the WikiText2 [95] dataset with a sequence length of 4096, comparing against established baselines in the following sections.

Baselines. To evaluate PATCH against established 2:4 sparsity pruning techniques, we compare it with the state-of-the-art learnable method MaskLLM [33], as well as one-shot methods including Wanda [136], SparseGPT [36], Thanos [64], ProxSparse [83] and magnitude pruning [53]. For one-shot pruning methods, following the default configurations in each paper, we prune the models over 128 samples from the C4 dataset.

The publicly available MaskLLM pruned checkpoints are limited to LLaMA-2 7B and LLaMA-3.1 8B models. To ensure a fair comparison across all models, we implemented MaskLLM in PyTorch and replicated its results for additional architectures presented in this study.

We faced a similar challenge with ProxSparse as well, where only the LLaMA-2-7B and LLaMA-3.1-8B checkpoints are publicly available. We have pruned other models with their oficial code base using their default hyperparameters for comparison.

Experiment Setup. All masks are trained using the HuggingFace Trainer API [153] for 2000 steps with a global batch size of 256 and a sequence length of 4096, processing 2B tokens from the SlimPajama corpus

Table 6.2: Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2 dataset) for diferent pruning methods. By jointly optimizing the location of dense tiles and the sparsity pattern within the sparse tiles, PATCH<sup>Joint</sup> allows for a continuous sparsity ratio for the models, providing a flexible tradeof between sparsity and model quality.
<table><tr><td rowspan="2">Sparsity Method</td><td rowspan="2"></td><td rowspan="2">Pattern</td><td colspan="2">Qwen-2.5 0.5B</td><td colspan="2">LLaMA-3.2 1B</td><td colspan="2">Gemma-3 1B</td></tr><tr><td></td><td>Acc (% ↑) PPL (↓)</td><td>Acc (% ↑)</td><td>PPL (↓)</td><td></td><td>Acc (% ↑) PPL (↓)</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>46.00</td><td>12.08</td><td>47.70</td><td>9.06</td><td>47.01</td><td>11.67</td></tr><tr><td rowspan="6">50%</td><td>Magnitude</td><td>2:4</td><td>30.16</td><td>6734.97</td><td>29.66</td><td>563.44</td><td>31.66</td><td>5005.56</td></tr><tr><td>Wanda</td><td>2:4</td><td>32.97</td><td>72.48</td><td>31.61</td><td>78.18</td><td>34.16</td><td>69.41</td></tr><tr><td>SparseGPT</td><td>2:4</td><td>34.81</td><td>36.59</td><td>35.55</td><td>32.73</td><td>35.58</td><td>44.59</td></tr><tr><td>Thanos</td><td>2:4</td><td>31.31</td><td>37.32</td><td>35.71</td><td>33.03</td><td>35.09</td><td>62.63</td></tr><tr><td>ProxSparse</td><td>2:4</td><td>32.05</td><td>111.05</td><td>33.55</td><td>49.33</td><td>36.63</td><td>90.50</td></tr><tr><td>MaskLLM</td><td>2:4</td><td>39.33</td><td>15.22</td><td>41.04</td><td>12.93</td><td>41.84</td><td>12.82</td></tr><tr><td>45%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>40.29</td><td>14.57</td><td>42.08</td><td>12.23</td><td>42.80</td><td>11.96</td></tr><tr><td>35%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>41.15</td><td>13.84</td><td>42.72</td><td>11.67</td><td>43.30</td><td>11.48</td></tr><tr><td>25%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>42.39</td><td>13.47</td><td>43.81</td><td>11.00</td><td>44.07</td><td>11.17</td></tr></table>

[135]. Training is accelerated via data parallelism across a single node with 4 H100 GPUs. In this setup, PATCH<sup>Joint</sup> requires 4.5 and 6 GPU hours on the 0.5B and 1B models, respectively, while PATCH<sup>Tile</sup> requires 21 and 24 GPU hours on the 7B and 8B models. The hyperparameters for PATCH<sup>Joint</sup> and PATCH<sup>Tile</sup> are summarized in Table 6.1, tuned on Qwen-2.5-0.5B. For the 2:4 mask parameters, we follow the configuration from MaskLLM [33].

Table 6.1: Hyper-parameters used for PATCH<sup>Joint</sup> and PATCH<sup>Tile</sup> across sparsity ratios. All hyper parameters were tuned on Qwen-2.5-0.5B.
<table><tr><td>Sparsity</td><td>Method</td><td>Optimizer</td><td>Logits Init</td><td>Gumbel Scaling</td><td>Gumbel</td><td>Prior(Strength)</td><td>Sparse Reg.</td><td>Weight Reg.</td></tr><tr><td>25%</td><td>PATCHJoint</td><td>Adam(0.001)</td><td>N(0, 0.014)</td><td>25 → 350</td><td>2 → 0.05</td><td>SparseGPT(3)</td><td>7</td><td>10</td></tr><tr><td>35%</td><td>PATCHJoint</td><td>Adam(0.001)</td><td>N(0, 0.014)</td><td>25 → 350</td><td>2 → 0.05</td><td>SparseGPT(3)</td><td>7</td><td>10</td></tr><tr><td>45%</td><td>PATCHJoint</td><td>Adam(0.001)</td><td>N(0, 0.014)</td><td>25 → 350</td><td>4 → 0.05</td><td>SparseGPT(3)</td><td>7</td><td>10</td></tr><tr><td>25%</td><td>PATCHTile</td><td>Adam(0.0001)</td><td>N(0, 0.014)</td><td>100 → 500</td><td>2 → 0.05</td><td>SparseGPT(3)</td><td>3</td><td>0.1</td></tr><tr><td>35%</td><td>PATCHTile</td><td>Adam(0.0001)</td><td>N(0, 0.014)</td><td>100 → 500</td><td>2 → 0.05</td><td>SparseGPT(3)</td><td>3</td><td>0.1</td></tr><tr><td>45%</td><td>PATCHTile</td><td>Adam(0.0001)</td><td>N(0, 0.014)</td><td>100 → 500</td><td>2 → 0.05</td><td>SparseGPT(3)</td><td>3</td><td>0.1</td></tr></table>

## 6.7.1 Model Quality Results

Joint sparse and dense tile optimization. For smaller models like Qwen-2.5 0.5B, LLaMA-3.2 1B, and Gemma-3 1B, we apply the joint variant PATCH<sup>Joint</sup>, which simultaneously optimizes dense tile locations and sparsity patterns within sparse tiles. This approach enables efective performance.

The average accuracy of the models across eight zero-shot downstream tasks and their perplexity on the WikiText2 dataset is reported in Table 6.2. The results demonstrate that PATCH<sup>Joint</sup> provides a flexible tradeof between sparsity ratio and model quality, narrowing the performance gap to dense models while ensuring hardware-friendly inference. A similar pattern holds for larger models using a memory-eficient variant, as explored next.

Table 6.3: Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2 dataset) for diferent pruning methods. By only optimizing the location of dense tiles while keeping sparsity pattern within the sparse tiles frozen, PATCH<sup>Tile</sup> provides a memory eficient variant for PATCH<sup>Joint</sup>, allowing for a continuous sparsity ratio for the models and providing a flexible tradeof between sparsity and model quality.
<table><tr><td></td><td>Sparsity Method</td><td>Pattern</td><td colspan="2">LLaMA-2 7B</td><td colspan="2">LLaMA-3.1 8B</td></tr><tr><td></td><td></td><td></td><td>Acc (% ↑)</td><td>PPL (↓)</td><td>Acc (% ↑) PPL (↓)</td><td></td></tr><tr><td>0% 50%</td><td>Dense</td><td></td><td>54.61</td><td>5.12</td><td>60.31</td><td>5.84</td></tr><tr><td rowspan="6"></td><td>Magnitude</td><td>2:4</td><td>43.44</td><td>54.39</td><td>35.93</td><td>765.92</td></tr><tr><td>Wanda</td><td>2:4</td><td>44.30</td><td>11.15</td><td>41.77</td><td>21.29</td></tr><tr><td>SparseGPT</td><td>2:4</td><td>45.09</td><td>10.12</td><td>45.53</td><td>15.11</td></tr><tr><td>Thanos</td><td>2:4</td><td>44.80</td><td>11.19</td><td>45.72</td><td>16.09</td></tr><tr><td>ProxSparse</td><td>2:4</td><td>45.92</td><td>9.18</td><td>45.14</td><td>15.17</td></tr><tr><td>MaskLLM</td><td>2:4</td><td>48.62</td><td>6.78</td><td>52.80</td><td>8.58</td></tr><tr><td>45%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>48.99</td><td>6.55</td><td>53.60</td><td>8.20</td></tr><tr><td>35%</td><td> ${ \mathrm { P A T C H } } ^ { \mathrm { T i l e } }$ </td><td>Dense/2:4 Tiles</td><td>50.08</td><td>6.18</td><td>55.28</td><td>7.89</td></tr><tr><td>25%</td><td> ${ \mathrm { P A T C H } } ^ { \mathrm { T i l e } }$ </td><td>Dense/2:4 Tiles</td><td>51.58</td><td>5.86</td><td>56.48</td><td>7.34</td></tr></table>

Memory-eficient tile selection. For larger models such as LLaMA-2 7B and LLaMA-3.1 8B, we employ the memory-eficient variant PATCH<sup>Tile</sup>, which freezes the fine-grained sparse weight structure while optimizing dense tile selections.

Table 6.3 summarizes the average accuracy of the models across eight downstream tasks in addition to their perplexity on the WikiText2 dataset for diferent sparsity ratios, illustrating that $\mathrm { P A T C H ^ { \mathrm { T i l e } } }$ delivers a comparable flexible sparsity-quality tradeof when using a high-quality frozen 2:4 mask.

Overall, across Table 6.2 and Table 6.3, PATCH consistently surpasses one-shot methods like Wanda, SparseGPT, and magnitude pruning due to its end-to-end training on large corpora. While MaskLLM also trains end-to-end on a large dataset, its fixed 2:4 sparsity ratio limits achievable accuracy and perplexity. In contrast, PATCH overcomes this limitation with flexible dense tile allocation, achieving accuracy gains and perplexity reductions from 45% to 25% sparsity that progressively align with dense model performance. The full per-task accuracy results are provided in Section D.2.

Comparison with unstructured sparsity In this section, we compare the quality of the models pruned with PATCH against other unstructured sparsity methods. Table 6.4 summarizes the average accuracy of the models across eight downstream tasks and the model perplexity on WikiText2 dataset. The results indicate that while unstructured sparsity consistently outperforms the hybrid sparsity, the gap between the two is not significant, showing that PATCH is helping to bridge the gap between unstructured sparsity and semi-structured sparsity.

## 6.7.2 Understanding the components of PATCH

This subsection examines the design choices driving PATCH’s performance by analyzing its behavior across various configurations on the Qwen-2.5 0.5B model.

Tile size. We initially assess the impact of tile size on PATCH’s performance, fixing hyperparameters to those optimized for 128×128 tiles. Table 6.5 reveals that 4 × 4 tiles maximize model quality through finer sparse-dense control, though larger tile sizes show minimal variation, suggesting robustness. However, smaller tiles may hinder hardware eficiency, requiring a balance with hardware specifications.

Table 6.4: Model quality (average accuracy across eight zero-shot tasks and perplexity on WikiText2 dataset) for PATCH, Wanda, and SparseGPT. For models with less than or equal to 1B parameters, PATCH<sup>Joint</sup> optimizes both dense tile locations and sparsity patterns, while for larger models $\mathrm { P A T C H ^ { \mathrm { T i l e } } }$ optimizes only dense tile locations with frozen sparsity patterns, both using Dense/2:4 Tiles pattern allowing continuous sparsity ratios and flexible tradeofs between sparsity and model quality. Wanda and SparseGPT are unstructured pruning methods.
<table><tr><td>Sparsity Method</td><td></td><td>Pattern</td><td colspan="2">Qwen-2.5 0.5B</td><td colspan="2">LLaMA-3.2 1B</td><td colspan="2">Gemma-3 1B</td><td colspan="2">LLaMA-2 7B</td><td colspan="2">LLaMA-3.1 8B</td></tr><tr><td></td><td></td><td></td><td>Acc (% ↑)</td><td>PPL (↓)</td><td>Acc (% ↑) PPL (↓)</td><td></td><td></td><td>Acc (% ↑) PPL (↓)</td><td>Acc (% ↑) PPL (↓)</td><td></td><td>Acc (% ↑) PPL (↓)</td><td></td></tr><tr><td>45%</td><td>PATCH</td><td>Dense/2:4 Tiles</td><td>40.29</td><td>14.57</td><td>42.08</td><td>12.23</td><td>42.80</td><td>11.96</td><td>48.99</td><td>6.55</td><td>53.60</td><td>8.20</td></tr><tr><td>45%</td><td>Wanda</td><td>Unstructured</td><td>41.45</td><td>18.81</td><td>40.76</td><td>16.56</td><td>42.87</td><td>25.38</td><td>52.72</td><td>6.36</td><td>55.67</td><td>8.24</td></tr><tr><td>45%</td><td>SparseGPT</td><td>Unstructured</td><td>42.31</td><td>17.65</td><td>42.66</td><td>15.01</td><td>43.52</td><td>22.26</td><td>52.77</td><td>6.46</td><td>56.70</td><td>8.21</td></tr><tr><td>35%</td><td>PATCH</td><td>Dense/2:4 Tiles</td><td>41.15</td><td>13.84</td><td>42.72</td><td>11.67</td><td>43.30</td><td>11.48</td><td>50.08</td><td>6.18</td><td>55.28</td><td>7.89</td></tr><tr><td>35%</td><td>Wanda</td><td>Unstructured</td><td>43.46</td><td>15.04</td><td>44.60</td><td>11.95</td><td>45.50</td><td>16.98</td><td>54.37</td><td>5.87</td><td>58.68</td><td>7.02</td></tr><tr><td>35%</td><td>SparseGPT</td><td>Unstructured</td><td>44.66</td><td>14.79</td><td>45.62</td><td>11.68</td><td>45.45</td><td>16.92</td><td>54.18</td><td>5.92</td><td>58.81</td><td>7.07</td></tr><tr><td>25%</td><td>PATCH</td><td>Dense/2:4 Tiles</td><td>42.39</td><td>13.47</td><td>43.81</td><td>11.00</td><td>44.07</td><td>11.17</td><td>51.58</td><td>5.86</td><td>56.48</td><td>7.34</td></tr><tr><td>25%</td><td>Wanda</td><td>Unstructured</td><td>45.70</td><td>13.70</td><td>46.50</td><td>10.46</td><td>46.56</td><td>15.14</td><td>54.60</td><td>5.65</td><td>59.80</td><td>6.54</td></tr><tr><td>25%</td><td>SparseGPT</td><td>Unstructured</td><td>45.28</td><td>13.63</td><td>46.52</td><td>10.42</td><td>46.37</td><td>15.05</td><td>54.71</td><td>5.68</td><td>59.52</td><td>6.55</td></tr></table>

Table 6.5: Impact of PATCH’s tile size across sparsity levels (↓ is better). The efect of tile size on model quality is not significant, showing PATCH’s robustness against tile size.  
Table 6.6: Global sparsity yields better quality by concentrating pruning in less important blocks and preserving density elsewhere (↓ is better).
<table><tr><td>Sparsity (0.5B)</td><td>128</td><td>64</td><td>32</td><td>16</td><td>8</td><td>4</td></tr><tr><td>45%</td><td>14.57</td><td>14.66</td><td>14.70</td><td>14.67</td><td>14.70</td><td>14.55</td></tr><tr><td>35%</td><td>13.84</td><td>14.08</td><td>14.15</td><td>14.03</td><td>14.01</td><td>13.72</td></tr><tr><td>25%</td><td>13.47</td><td>13.54</td><td>13.52</td><td>13.53</td><td>13.40</td><td>13.11</td></tr></table>

<table><tr><td>Sparsity (0.5B)</td><td>Global</td><td>Layer-wise</td></tr><tr><td>45%</td><td>14.57</td><td>15.17</td></tr><tr><td>35%</td><td>13.84</td><td>14.48</td></tr><tr><td>25%</td><td>13.47</td><td>13.95</td></tr></table>

Joint vs. tile-only mask search. We then analyze the impact of fixing the 2:4 masks and optimizing only tile masks. Table 6.7 shows that among frozen 2:4 masks, MaskLLM provides the strongest results. On the other hand, one-shot pruning methods perform comparably at higher sparsity levels but diverge at lower sparsity, with SparseGPT emerging as the best overall. When comparing against our full approach, joint optimization of both tile and 2:4 masks consistently outperforms tile-only training across sparsity ratios. Nevertheless, tileonly training remains a practical alternative for larger models in resource-constrained settings, as also reflected in Table 6.3.

Sparsity allocation. We analyze how sparsity is allocated across transformer blocks under a global target. Across models, deeper transformer blocks are pruned far less, while the initial blocks also tend to receive lighter pruning depending on the architecture. By contrast, the middle blocks consistently absorb most of the sparsity, suggesting that they contain more redundancy (Figure 6.2). We compare this flexible allocation to enforcing sparsity uniformly at the layer level. As shown in Table 6.6, global targets deliver better results by pruning more aggressively in redundant layers while preserving capacity in sensitive ones. In contrast, layer-wise targets impose uniform sparsity that can over-prune critical components [79, 156, 78, 158].

On top of variation across depth, sparsity is also distributed unevenly across the individual linear layers within each transformer block. Figure 6.3 breaks down the allocation into the query, key, value, and output matrices of the attention module, as well as the up, gate, and down matrices of the MLP for the Qwen 2.5 0.5B model. The up, gate, and down layers absorb most of the sparsity and largely explain the overall allocation pattern seen in Figure 6.2. In contrast, the attention module is treated as more critical. The key and value matrices are never pruned, while the output matrix shows moderate pruning at higher global sparsity targets. The query matrix is pruned the most, suggesting it is the least important within the attention submodule.

Table 6.7: Impact of fixed 2:4 mask selection for PATCH<sup>Tile</sup>, compared with joint optimization (↓ is better). PATCH<sup>Joint</sup> achieves the lowest perplexity overall, while for PATCH<sup>Tile</sup>, MaskLLM provides the best frozen mask.
<table><tr><td>Sparsity (0.5B)</td><td>MaskLLM</td><td>SparseGPT (w/o weight update)</td><td>Wanda</td><td>Magnitude</td><td>PATCHJoint</td></tr><tr><td>45%</td><td>15.06</td><td>21.84</td><td>21.83</td><td>21.33</td><td>14.57</td></tr><tr><td>35%</td><td>14.55</td><td>17.29</td><td>17.96</td><td>19.90</td><td>13.84</td></tr><tr><td>25%</td><td>14.17</td><td>14.89</td><td>15.09</td><td>16.05</td><td>13.47</td></tr></table>

![](images/aed92f85a5e600fa4340a8a2d28a253e77af936291f5231c0c13223ee14678fc.jpg)

![](images/c8fe480981269939f8a0fe1fdc5994dd05c4708a622b7043cd8d10395a0e4161.jpg)

![](images/dace3cd7bff0cf36f18cee3653ecdc3022232292085b087a88c13e918b71ac44.jpg)  
Figure 6.2: Layer-wise sparsity allocation under diferent global sparsity budgets for various models. PATCH achieves the target global sparsity while flexibly distributing pruning across transformer layers.

Additionally, we provide the sparsity distributions for the Gemma-3-1B (Figure 6.4) and Llama-3.2-1B (Figure 6.5) models, as referenced in the main text. Similar to the Qwen-2.5 0.5B model, the patterns observed here indicate that MLP layers (up, gate, and down matrices) are pruned more aggressively, absorbing the majority of sparsity. In contrast, the self-attention layers are treated as more critical, with key and value matrices remaining largely dense or unpruned, while the query matrix experiences the highest pruning within the attention submodule, and the output matrix shows moderate pruning under higher global sparsity targets. This consistent behavior across models underscores the redundancy in MLP components and the sensitivity of attention mechanisms.

## 6.7.3 Speedup and memory savings

We evaluate the inference eficiency of the LLaMA-2 7B model pruned with PATCH using the STOICC [122] compiler. With a batch size of 16 on an A6000 GPU, we observe end-to-end throughput improvements of 1.18×, 1.27×, and 1.38× at sparsity levels of 25%, 35%, and 45%, respectively, compared to the dense baseline. At the same sparsity levels, the model’s GPU memory footprint during inference is also reduced, dropping to 0.76×, 0.68×, and 0.59× of the fully dense model, respectively. These results underscore the trade-of between accuracy retention and the computational savings enabled by sparsity.

![](images/77dde896f9dd35fafdcb1b02b315dba54b256e5459d890a6e5c33c9793d8c428.jpg)  
Figure 6.3: Sparsity distribution across Attention and MLP layers under varying global sparsity budgets in Qwen-2.5 0.5B.

## 6.8 Conclusion and Limitations

In this chapter, we explored the upper limits of the sparsity pillar when a training budget is available. We introduced PATCH, a hybrid sparsity framework that breaks the rigidity of the layer-wise static masks used in Chapter 5. By partitioning weight matrices into tiles designated as either dense or 2:4 sparse, PATCH enables adaptive sparsity ratios between 0% and 50%, dynamically balancing accuracy in critical regions with hardware acceleration elsewhere.

Experiments across models up to 8B parameters show that PATCH consistently improves accuracy over state-of-the-art 2:4 pruning methods while achieving up to 1.38× end-to-end speedup on consumer-grade GPUs. These results demonstrate the promise of hybrid sparsity as a practical approach to eficient LLM inference and motivate future work on broader sparsity formats, integration with quantization, and co-design with hardware kernels.

While PATCH ofers superior accuracy-eficiency trade-ofs, it is important to situate it within the broader ”compute budget” narrative of this thesis. Unlike OPTIMA, which targeted the zero-training regime, PATCH requires a fine-tuning budget. The learnable masking process introduces computational overhead that makes it unsuitable for instant, on-device adaptation. However, as established in Section 1.4, these two chapters represent complementary solutions for diferent deployment scenarios: OPTIMA maximizes performance when training is impossible, whereas PATCH maximizes performance when resources permit.

With PATCH, we have now refined the sparsity pillar to its logical conclusion in both static and dynamic regimes. We have made it highly accurate (via hybrid masks) and physically fast (via tile-level kernels). Yet, we have applied this pillar in isolation. As noted in the ”Sparsity Paradox” (Table 1.1), even optimized sparsity eventually hits an accuracy wall that cannot be overcome by simply removing fewer weights. To unlock the next frontier of performance, we must stop treating sparsity as a solo actor. In the next chapter, we present the culmination of this thesis: SLIM. There, we integrate our optimized sparsity findings with aggressive quantization and low-rank approximations, finally realizing the full potential of the Compression Trinity in a

![](images/34322edbcc2b9bcdf2d7c12c35f04ffdb9b53fa30adf69d7c88c3c5ed2adee33.jpg)  
Figure 6.4: Sparsity distribution across Attention and MLP layers under varying global sparsity budgets in Gemma-3 1B.  
unified, one-shot framework.

![](images/f701a87f1b245a22e4c9bd415bb4e6a5ccceb072a04a34d06c8868c5e1206496.jpg)

![](images/0c38f4c8a6c1002e3231a64b1aeb63d4dbb3f0e3e5ca5c3a53ea2462941c1039.jpg)

![](images/54946d2fab801a7af16ceb093913aba43eebdaccc3177f1fb3268a37e9149770.jpg)

![](images/b2d08a4f1469ec81eb60374a1c8b9e3bcd124393222f4ae3ceff9be8af5b40b0.jpg)

![](images/b7fd42a13a084d5a691b473b54887c2205356879ad7e5a0dc131bc930b8ca760.jpg)

![](images/402c3cd633fe44edb5eb8275f95fcad672d4aedae3b0aedf0c1eeb77a7686935.jpg)

![](images/bad18d47d560fac4c10f23717ccda8f60f81b299fcf03bcead5ac6112dbc99cc.jpg)  
Figure 6.5: Sparsity distribution across Attention and MLP layers under varying global sparsity budgets in LLaMA-3.2 1B.

# SLIM: One-shot Quantization and Sparsity with Low-rank Approximation for LLM Weight Compression

Publication and Contributions. The content of this chapter is based on the paper “SLiM: One-shot Quan tized Sparse Plus Low-rank Approximation of LLMs” [100], published at Forty-Second International Confer ence on Machine Learning (ICML), 2025. This work was conducted in collaboration with Amir Yazdanbakhsh and Maryam Mehri Dehnavi. Mohammad Mozafari was the lead contributor, responsible for the algorithm design, implementation, and experimental evaluation. Amir Yazdanbakhsh and Maryam Mehri Dehnavi su pervised the project and contributed to the writing and revision of the manuscript.

## 7.1 Introduction

In the preceding chapters, we pushed the Sparsity pillar to its limits. With OPTIMA (Chapter 5), we established the mathematical upper bound for reconstruction with layer-wise pruning and weight update, and with PATCH (Chapter 6), we broke the rigidity of those masks using end-to-end learnable hybrid patterns. However, as identified in the ”Sparsity Paradox” (Section 1.3), sparsity is the most destructive pillar. Even with the optimized skeletons provided by OPTIMA and PATCH, a performance gap remains because we are removing information that cannot be fully recovered by weight adjustment alone. Additionally, with a perfected sparse model, relying on a single compression technique limits the potential for eficiency. To bridge this gap and fully conquer the memory bandwidth bottleneck, we must stop treating sparsity in isolation. We must integrate it with the remaining two pillars of the Compression Trinity: Quantization and Low-Rank Approximation.

This chapter presents the culmination of this thesis: SLIM, a unified framework thatjointly applies hardwarefriendly sparsity, quantization, and low-rank approximations in a single one-shot step. Bringing these three pillars together presents a formidable challenge: compounded error. When aggressive sparsity (removing weights) meets aggressive quantization (reducing precision), the errors do not merely add up; they exacerbate each other, leading to a catastrophic drop in model capability [36, 129, 90, 81, 49]. Traditional recovery methods rely on costly retraining (e.g., Quantization-Aware Training [127, 114]), which violates the ”resource constrained” principles we explored in Chapter 5. Conversely, existing one-shot methods like SparseGPT [36] struggle to combine structured patterns (like 2:4 sparsity) with low-bit quantization, failing to arrest the accuracy slide.<sup>1</sup>

To resolve these limitations and realize the full Trinity without retraining, we propose SLIM. We decompose the problem into three synchronized sub-tasks, ensuring that each pillar supports the others rather than conflicting with them.

1. Quantization: We prioritize uniform quantization for hardware eficiency. While SLIM is compatible with any standard quantization kernel (e.g., Group MinMax or AbsMax), we introduce SLIM-Quant, a probabilistic and tractable reformulation that finds the optimal quantization parameters. This allows users to either leverage existing quantization standards or utilize our optimizer to minimize the error floor before sparsity is even applied.

2. Sparsity: We apply hardware-friendly pruning to the quantized weights to create the eficient sparse structure. Crucially, the SLIM framework is agnostic to the specific mask selection algorithm. This allows us to seamlessly integrate masks generated by Wanda [136], OPTIMA, PATCH, or future stateof-the-art selectors, ensuring the framework remains relevant as pruning metrics evolve.

3. Low-Rank Approximation: Finally, we deploy the third pillar not just for compression, but as a mathematically derived error-correction mechanism. This is the critical innovation that closes the accuracy gap left by Chapter 5 and Chapter 6. We propose SLIM-LoRA, a one-shot low-rank adaptation method designed to compensate for the aggregated error. Unlike standard adapters that require iterative training to find optimal values [61, 104, 28, 48, 80], we develop a saliency function that is both invertible and additive. These properties enable us to analytically compute the optimal low-rank adapter values that minimize the compression-induced error in one shot, eliminating the need for any retraining overhead.

By solving the compounded error problem through this joint formulation, SLIM shifts the Pareto frontier of eficiency. It achieves what neither OPTIMA nor PATCH could do alone: recovering close to dense model accuracy while maintaining high compression rates. Compared to state-of-the-art methods, SLIM achieves an average accuracy improvement of 5.66% on LLaMA-2-7B under 2:4 sparsity and 4-bit quantization. Uniquely, it delivers higher model accuracy at the same total bit budget compared to existing techniques (up to 0.5%) and even outperforms uncompressed dense models at equal parameter budgets (up to 0.6%). Beyond accuracy, SLIM demonstrates the practical value of the Trinity, achieving up to 3.78× and 3.75× layer-wise speedup on NVIDIA RTX3060 and A100 GPUs, respectively. For cases requiring maximal performance, we also support an optional lightweight PEFT method, providing up to an 1.66% additional accuracy improvement.

## 7.2 Related work

SLIM combines model pruning and quantization for compression, complemented by zero-shot low-rank adapters to recover lost accuracy. This section reviews related work on these topics.

## 7.2.1 Pruning

Eliminating redundant weights reduces computation and memory costs during inference. Optimal Brain Damage (OBD) [75] leverages second-order information of the loss function to identify the least important weights but is computationally prohibitive for large language models (LLMs) [99]. WoodFisher [134] approximates the Hessian matrix using Kronecker Factorization to mitigate this overhead but struggles to scale to LLMs.

Optimal Brain Surgeon (OBS) [54] evaluates weight matrices layer-wise using the layer-wise Hessian matrix to preserve layer outputs. However, the cubic growth in the cost of inverting the layer-wise Hessian with model size renders this approach impractical for LLMs. Optimal Brain Compression (OBC) [35] addresses the OBS-defined compression problem using a greedy algorithm, while SparseGPT reformulates it as a sparse regression problem. Wanda introduces a lightweight method based on weight and activation magnitudes to identify unimportant weights without updating their values.

## 7.2.2 Quantization

Quantizing all elements in a matrix is challenging due to the significant impact of outliers on the model [27]. Group quantization [3, 47] addresses this by quantizing small groups of a weight matrix with a shared quantization parameter, but it introduces challenges discussed in Section E.16.

AbsMax [65] with round-to-nearest (RTN) is the simplest quantization scheme for matrix elements. OPTQ [37] minimizes layer-wise error using an approach akin to OBS. AWQ [81] shifts the challenge of quantizing salient weights to activations, while SmoothQuant [155] balances quantization error between weights and activations, enabling input quantization. OmniQuant [129] improves accuracy with learnable clipping and channel scaling. AfineQuant leverages equivalent afine transformations to reduce quantization error, and QuaRot [7] uses rotations to eliminate outliers during quantization.

Advanced methods like JSQ [49] jointly prune and quantize weights to 8 bits but struggle to recover accuracy in low bit-width quantization, limiting their utility.

## 7.2.3 Low-rank Adapters

Low-rank adapters were first introduced to LLMs to reduce the overhead of fine-tuning [61, 101]. Q-LoRA [28] extended this approach by quantizing weights before fine-tuning, allowing the process to recover accuracy lost during quantization. LQ-LoRA [48] further improved Q-LoRA by initializing the adapters using the SVD of the quantization error. LoSparse [80] has a similar approach as LQ-LoRA, but for sparsity, initializing the low-rank adapters to the norm of the pruning error. RoSA [104] expands the learning capability of the model by adding both low-rank and sparse adapters to the model. This approach adds an extra sparse matrix multiplication to the inference, increasing the adapter overhead even further. However, all these methods require hundreds of millions of tokens for fine-tuning, making them costly and not comparable to one-shot pruning and quantization methods, or methods that use much shorter fine-tuning phases.

L<sup>2</sup>QER [162] avoids fine-tuning by using one-shot low-rank adapters to mitigate quantization error. However, it performs poorly when combined with sparsity, resulting in a significant accuracy gap between the compressed and dense models.

## 7.2.4 Sparse Plus Low-Rank Matrix Decomposition

The decomposition of a matrix into the sum of a sparse component and a low-rank component is a classical problem in signal processing and optimization, most prominently studied under the framework of Robust Principal Component Analysis (RPCA) [15]. Given an observed matrix M, RPCA seeks to recover $M =$ $L _ { 0 } + S _ { 0 } ,$ , where $L _ { 0 }$ is low-rank and $S _ { 0 }$ is sparse, by solving the convex program known as Principal Component

CHAPTER 7. SLIM: ONE-SHOT QUANTIZATION AND SPARSITY WITH LOW-RANK APPROXIMATION FOR LLM WEIGHT COMPRESSION

Pursuit (PCP):

$$
\operatorname* { m i n } _ { L , S } \| L \| _ { * } + \lambda \| S \| _ { 1 } \quad \mathrm { s u b j e c t t o } \quad L + S = M ,\tag{7.1}
$$

where $\| \cdot \|$ denotes the nuclear norm and λ is a regularization parameter. Candès et al. [15] proved that under mild incoherence conditions on the low-rank component, this convex relaxation exactly recovers both $L _ { 0 }$ and $S _ { 0 }$ even when the sparse errors are arbitrarily large in magnitude. Chandrasekaran et al. [16] provided complementary recovery guarantees through rank-sparsity incoherence conditions, while Zhou et al. [169] extended the theory to the noisy setting (Stable PCP), showing that the decomposition remains robust when the observation also contains a small dense noise term, i.e., $M = L _ { 0 } + S _ { 0 } + N _ { 0 }$

The structural parallel between RPCA and model compression was first explored in the context of deep convolutional networks by Yu et al. [160], who decomposed weight matrices into sparse-plus-low-rank form us ing greedy bilateral decomposition and showed that this representation achieves better accuracy-compression trade-ofs than either pure pruning or pure low-rank factorization in isolation. More recently, the connection to LLM compression has been made explicit. OATS [164] formulates post-training weight compression as a sparse-plus-low-rank decomposition problem, scaling the weights by the second moment of input embeddings before decomposition to preserve outlier features. HASSLE-free [91] established a unified framework show ing that several existing LLM pruning methods, including Wanda and Magnitude Pruning, can be viewed as special cases of an alternating minimization procedure for the sparse-plus-low-rank objective. Concurrently, 3BASiL [11] proposed a three-block ADMM formulation that jointly optimizes the sparse and low-rank components, ofering improved convergence guarantees over alternating minimization.

SLIM’s formulation $W \approx W ^ { C } + L R ,$ , where $W ^ { C }$ is the sparse (and quantized) component and LR is the low-rank correction, can be viewed through the lens of this decomposition tradition. However, SLIM difers from standard RPCA approaches in several important ways. First, the sparse component in SLIM is additionally quantized, introducing a structured noise that is absent in classical RPCA. Second, rather than solving for S and $L$ jointly via convex optimization, SLIM employs a sequential pipeline (quantize, then sparsify, then compute the low-rank correction), which enables the use of of-the-shelf pruning and quantization algorithms but forgoes the joint optimality guarantees of PCP. Third, SLIM’s saliency-weighted SVD (Equation 7.12) introduces input statistics into the decomposition, a data-dependent weighting that has no direct analogue in standard RPCA but shares the spirit of the outlier-aware scaling in OATS [164]. Bertsimas et al. [12] further studied the sparse-plus-low-rank decomposition from a discrete optimization perspective, providing mixed integer programming formulations that could, in principle, be adapted to the compression setting.

## 7.3 Preliminaries

Model Compression. Model compression reduces the compute and memory demands of large models while maintaining predictive accuracy by minimizing output diferences between compressed and original models. However, directly optimizing these diferences across the entire model is computationally infeasible due to the high dimensionality of neural networks. Optimal Brain Surgeon (OBS) [54] simplifies this challenge by focusing on minimizing output discrepancies layer by layer, using calibration datasets.

OBS applies a layer-wise approach to compress feed-forward layers eficiently. Denoting compressed matrices with a superscript $C ,$ for a layer with input $\mathcal { X } \in \mathbb { R } ^ { b \times d _ { i n } }$ , weight $\mathcal { W } \in \mathbb { R } ^ { d _ { i n } \times d _ { o u t } }$ , and output $\mathcal { Y } \in \mathbb { R } ^ { b \times d _ { o u t } }$ , it minimizes output diferences by optimizing Equation 7.2. This method ensures compression fidelity and has become foundational for many modern compression techniques.

![](images/a47c722502111e29c5a0fce75b62dafc655915bcaba86e5322e724e4e1f4d0c0.jpg)  
Figure 7.1: The SLIM weight compression pipeline consists of three main steps: (1) Quantizing weights using the symmetric SLIM-Quant algorithm, producing quantized weights $\mathcal { W } ^ { \mathcal { Q } }$ and quantization error $E _ { Q } \mathrm { ; }$ (2) Sparsifying quantized weights $\mathcal { W } ^ { \mathcal { Q } }$ through a pruning method, resulting in compressed weights $\mathcal { W } ^ { \mathcal { C } }$ and sparsity error $E _ { S } ; ( 3 )$ Mitigating compression errors through SLIM saliency-based low-rank approximation, generating left and right low-rank adapters $L$ and R. Optionally, these adapters can be fine-tuned with sparse quantized weights frozen to further enhance model accuracy.

$$
\operatorname* { m i n } _ { \mathcal { W } ^ { C } } | \mathcal { V } ^ { C } - \mathcal { V } | ^ { 2 } = \operatorname* { m i n } _ { \mathcal { W } ^ { C } } | \mathcal { X } ( \mathcal { W } ^ { C } - \mathcal { W } ) | ^ { 2 }\tag{7.2}
$$

Symmetric Quantization. Symmetric quantization is a core technique for reducing model size and boosting computational eficiency. It computes the quantized matrix $\begin{array} { r } { \mathcal { M } ^ { Q } \propto r o u n d ( \frac { \mathcal { M } } { \alpha } ) } \end{array}$ , where α is a scaling factor based on the range or norm of the matrix. This scaling ensures $\mathcal { M } ^ { Q }$ values stay within the representable range, enabling eficient matrix multiplications with minimal overhead. However, its efectiveness depends on selecting α carefully, as this choice significantly impacts precision.

AbsMax, the most common symmetric quantization method, selects α as the matrix’s maximum absolute value, ensuring all values remain within the target range. Unfortunately, it is highly sensitive to outliers; a single large value can inflate α, reducing the precision of most quantized weights. For zero-centered, bellcurved distributions typical in LLMs, AbsMax maps many weights to zero, leading to significant quantization errors.

Group quantization [3, 47] tackles AbsMax’s outlier sensitivity by assigning separate scaling factors to subgroups of the weight matrix. This approach captures local variations in weight magnitudes, reducing quan tization error for non-uniform distributions. However, storing multiple scaling factors increases memory usage, and subgroup-specific dequantization increases computational complexity, potentially slowing inference. The challenges of using group quantization are discussed in Section E.16.

## 7.4 Quantized sparse plus low-rank approximation of LLMs

To achieve efective compression of LLMs while preserving accuracy, SLIM combines quantization, pruning, and saliency-based low-rank adapters into an integrated pipeline. First, SLIM applies SLIM-Quant , a novel scheme designed to minimize quantization error, laying the foundation for subsequent pruning using methods such as Wanda [136]. Finally, low-rank adapters are introduced to reduce the impact of compression errors from both quantization and pruning, ensuring minimal accuracy loss. The overall process is illustrated in Figure 7.1, providing a visual summary of how these components interact to achieve efective model compression. In the following sections, we dive into the details of each step, highlighting the innovations and contribution of SLIM-Quant , the pruning strategy, and the saliency-based low-rank adapters.

## 7.4.1 SLIM-Quant quantization method

SLIM adopts symmetric weight quantization due to its low dequantization and memory overhead and ease of implementation. Denoting the quantized matrices by $Q$ superscript, Equation 7.3 shows the symmetric quantization formula for q-bit quantization, where α is the quantization scaling parameter and $c l i p ( . )$ operator clips the input to values between [−1, 1].

$$
{ \mathcal { W } } ^ { Q } = r o u n d ( c l i p ( { \frac { \mathcal { W } } { \alpha } } ) ) 2 ^ { q - 1 }\tag{7.3}
$$

The objective of quantization is to reduce the weight reconstruction error shown in Equation 7.4, where the ∗ superscript shows the optimal value. But the objective function in Equation 7.4 is not convex, and to the best of our knowledge, does not have a closed form solution.

$$
\alpha ^ { * } = \arg \operatorname* { m i n } _ { \alpha } \| \mathcal { W } ^ { Q } - \mathcal { W } \| ^ { 2 } = \arg \operatorname* { m i n } _ { \alpha } \| r o u n d ( c l i p ( \frac { \mathcal { W } } { \alpha } ) ) 2 ^ { q - 1 } - \mathcal { W } \| ^ { 2 }\tag{7.4}
$$

To solve the mean squared error (MSE) problem in Equation 7.4, we propose a probabilistic reformulation as shown in Equation 7.5, where $Q ( . )$ and $Q ^ { - 1 } ( . )$ are the quantization and dequantization functions respectively, and $f ( . )$ is the probability distribution function (PDF) of the weight elements.

$$
\alpha ^ { * } = \arg \operatorname* { m i n } _ { \alpha } E _ { Q } = \arg \operatorname* { m i n } _ { \alpha } | | \mathcal { W } ^ { Q } - \mathcal { W } | | ^ { 2 } = \arg \operatorname* { m i n } _ { \alpha } \int _ { - \infty } ^ { \infty } f ( x ) | Q ^ { - 1 } ( Q ( x ) ) - x | ^ { 2 } d x\tag{7.5}
$$

By incorporating the quantization formula from Equation 7.3 into Equation 7.5, we can simplify the integration into the sum of two terms based on the absolute value of the data: the quantization error for absolute values less than α (Equation 7.6) and the clipping error for absolute values larger than α (Equation 7.7). Here, $f _ { a b s } ( . )$ represents the probability density function (PDF) of the absolute value of the weights. Equation 7.8 presents the simplified version of Equation 7.5.

$$
E _ { q u a n t } ( \alpha ) = \int _ { 0 } ^ { \alpha } f _ { a b s } ( x ) | \alpha \times r o u n d ( \frac { x } { \alpha } ) \times 2 ^ { 1 - q } - x | ^ { 2 } d x\tag{7.6}
$$

$$
E _ { c l i p } ( \alpha ) = \int _ { \alpha } ^ { \infty } f _ { a b s } ( x ) | \alpha - x | ^ { 2 } d x\tag{7.7}
$$

$$
\alpha ^ { * } = \arg \operatorname* { m i n } _ { \alpha } E _ { Q } ( \alpha ) = \arg \operatorname* { m i n } _ { \alpha } E _ { q u a n t } ( \alpha ) + E _ { c l i p } ( \alpha )\tag{7.8}
$$

Equation 7.8 can be solved theoretically by diferentiating the objective function with respect to α, provided the probability density function (PDF) of the weight distribution is known. However, the weight distribution of neural networks rarely conforms to standard PDFs. To verify this, we tested various candidate distributions, including Gaussian, Laplace, Pareto, q-Gaussian, and Weibull, as they are commonly used in modeling natural data. Unfortunately, none of these matched the observed weight distributions accurately. This discrepancy underscores the need for a more adaptable method, motivating the data-driven approach we adopt in SLIM-Quant .

Algorithm 5 SLIM-Quant Algorithm   
1 Input: Weight magnitude PDF $f _ { a b s } ,$ , high resolution step size $\eta _ { h i g h }$ , low resolution step size $\eta _ { l o w }$   
2 weight matrix W, quantization bitwidth $q .$   
3 Output: Quantized weight matrix $\mathcal { W } _ { q u a n t } .$   
4   
Function EstimateError(α)   
5 $\begin{array} { r } { E _ { q u a n t } ( \alpha ) = \int _ { 0 } ^ { \alpha } f _ { a b s } ( x ) | \alpha \times \mathrm { r o u n d } ( \frac { x } { \alpha } ) \times 2 ^ { 1 - q } - x | ^ { 2 } d x } \end{array}$   
6 $\begin{array} { r } { E _ { c l i p } ( \alpha ) = \int _ { \alpha } ^ { \infty } f _ { a b s } ( x ) | \alpha - x | ^ { 2 } d x } \end{array}$   
7 return $E _ { q u a n t } + E _ { c l i p }$   
8   
end function   
9 $E \gets \mathrm { E m p t y D i }$ ctionary() ▷ Initialize error dictionary   
10 for for α in $\mathtt { r a n g e } ( 0 , M , \eta _ { l o w } )$ do   
11 $E ( \alpha ) \gets \mathtt { E s t i }$ mateError(α)   
12 end for   
13 $\alpha _ { l o w }$ ← arg min<sub>α</sub> $E ( \alpha )$   
14 for for α in $\mathtt { r a n g e } ( \alpha _ { l o w } - \eta _ { l o w } , \alpha _ { l o w } + \eta _ { l o w } , \eta _ { h i g h } )$ do   
15 E(α) ←EstimateError $( \alpha )$   
16 end for   
17 ${ \boldsymbol { \alpha } } ^ { * } \gets \mathrm { a r g }$ min<sub>α</sub> $E ( \alpha )$   
18 $\mathcal { W } _ { q u a n t } \overset {  } {  } \mathrm { r o u n d } ( \mathrm { c l i p } ( \frac { \mathcal { W } } { \alpha ^ { * } } ) ) \times 2 ^ { q - 1 }$ ▷ Apply optimal quantization   
19 Return: $\mathcal { W } _ { q u a n t } .$

To address the absence of a closed-form weight PDF, we employ numerical integration on the weight histogram to solve Equation 7.8. To enhance eficiency, we adopt a multi-grid strategy: starting with 10 uniform samples in the range (0, max(W)), the grid is iteratively refined around the region of minimum error. This iterative process converges to the optimal α with minimal computational overhead. The full procedure is detailed in Algorithm 5.

## 7.4.2 SLIM-LoRA low-rank adapters

The use of a low-rank adapter to compensate for compression errors can be situated within the broader framework of sparse-plus-low-rank matrix decomposition, a problem studied extensively in the Robust PCA litera ture [15, 16]. In this classical formulation, a matrix is expressed as the sum of a sparse component and a low rank component, with provable recovery guarantees under suitable incoherence conditions. Yu et al. [160] applied this decomposition paradigm to compress deep neural network weights, and recent works such as OATS [164] and HASSLE-free [91] have extended it to LLMs. SLIM adapts this principle to the joint compression setting: rather than solving a single convex program, we derive the low-rank component analytically from the compression error using a saliency-weighted SVD, which enables a one-shot solution without itera tive optimization. We detail this procedure below.

After quantizing the model using SLIM-Quant , we sparsify it using an of-the-shelf one-shot pruning method such as Wanda. The combined efects of quantization and pruning of a weight matrix can be modeled as additive noise, such that $\mathcal { W } ^ { C } = \mathcal { W } + E _ { Q } + E _ { S }$ , where $E _ { Q } = \mathcal { W } - \mathcal { W } ^ { Q }$ and $E _ { S } = \mathcal { W } ^ { C } - \mathcal { W } ^ { Q }$ are the quantization and sparsity errors respectively. To mitigate these errors, we introduce low-rank adapters that adjust the compressed weights such that $\mathcal { W } \approx \mathcal { W } ^ { C } + \mathcal { L } \mathcal { R }$ , where $\mathcal { L } \in \mathbb { R } ^ { d _ { i n } \times r }$ and $\mathcal { R } \in \mathbb { R } ^ { r \times d _ { o u t } }$ are the low-rank adapters and r is the adapter rank.

CHAPTER 7. SLIM: ONE-SHOT QUANTIZATION AND SPARSITY WITH LOW-RANK APPROXIMATION FOR LLM WEIGHT COMPRESSION

A straightforward approach minimizes the total error norm between W and $\mathcal { W } ^ { C }$ , focusing solely on reducing the error magnitude while ignoring the saliency of individual elements in the weight matrix. We call this method Naive-LoRA as it overlooks the significance of individual elements in the weight matrix. However, this method is suboptimal and can be substantially improved.

To address the limitations of Naive-LoRA, we propose a novel low-rank approximation formulation that integrates weight saliency and uses a carefully designed saliency function to determine optimal adapters. The saliency function (F) in our formulation needs to satisfy two key properties. First, it needs to be invertible, enabling the retrieval of low-rank adapters from their saliency. Second, it must be additive, meaning $\forall A , B$ $F ( A + B ) = F ( A ) + F ( B )$ . The additive property is crucial for isolating the saliency of low-rank adapters from the compressed matrix and distinguishing the saliency of the error from that of the original weights. These properties ensure that the saliency function can efectively isolate and optimize the contribution of low-rank adapters, forming the foundation of our proposed formulation.

Assuming that there exists an additive invertible saliency function $F : \mathbb { R } ^ { d _ { i n } \times d _ { o u t } }  \mathbb { R } ^ { d _ { i n } \times d _ { o u t } }$ , we need to solve Equation 7.9 to find the optimal adapters. By using the additive property of the saliency function $F ( . )$ , we can simplify Equation 7.9 to Equation 7.10.

$$
\mathcal { L } , \mathcal { R } = \arg \operatorname* { m a x } _ { \mathcal { L } , \mathcal { R } } | | F ( \mathcal { W } ^ { C } + \mathcal { L } \mathcal { R } ) | | ^ { 2 } = \arg \operatorname* { m i n } _ { \mathcal { L } , \mathcal { R } } | | F ( \mathcal { W } - ( \mathcal { W } ^ { C } + \mathcal { L } \mathcal { R } ) ) | | ^ { 2 }\tag{7.9}
$$

$$
\mathcal { L } , \mathcal { R } = \mathop { \operatorname { a r g m i n } } _ { \mathcal { L } , \mathcal { R } } | | F ( \mathcal { W } - \mathcal { W } ^ { C } ) - F ( \mathcal { L } \mathcal { R } ) | | ^ { 2 } = \mathop { \operatorname { a r g m i n } } _ { \mathcal { L } , \mathcal { R } } | | F ( - ( E _ { Q } + E _ { S } ) ) - F ( \mathcal { L } \mathcal { R } ) | | ^ { 2 }\tag{7.10}
$$

Now, we can find $F ( \mathcal { L } \mathcal { R } )$ by computing the SVD of $F ( - ( E _ { Q } + E _ { S } ) )$ , and using the invertibility property of $F$ , we can obtain the exact value of L and $\mathcal { R }$

The saliency function used in SLIM must satisfy three essential criteria—invertibility, additivity, and the efective utilization of input and weight statistics—to optimize weight importance during compression. Recent works such as Wanda, AWQ, LLM.int8(), and $\mathrm { L ^ { 2 } Q E R }$ suggest that the product of the magnitude of the weights and activations is a useful metric for identifying important weights during pruning and quantization. Motivated by this observation, we propose a saliency function formulation for $F$ that meets these criteria and leverages weight-activation interactions for efective compression.

To incorporate input statistics into the saliency function, we define $F ( \mathcal { W } ) \triangleq d i a g ( \mathbf { x } ) \mathcal { W }$ , where $\mathbf { x } \in \mathbb { R } ^ { d _ { i \tau } }$ represents the average absolute value of inputs from a calibration set. This formulation ensures that the saliency function efectively weights the matrix elements based on their significance during compression, facilitating a more accurate approximation. By replacing $F ( \mathcal W )$ in Equation 7.10, the optimization problem transforms into a computationally eficient solution using singular value decomposition, followed by an inverse saliency transformation to derive the left low-rank adapter (Equation 7.12).

$$
\mathcal { L } , \mathcal { R } = \arg \operatorname* { m i n } _ { \mathcal { L } , \mathcal { R } } | | - d i a g ( \mathbf { x } ) ( E _ { Q } + E _ { S } ) - d i a g ( \mathbf { x } ) \mathcal { L } \mathcal { R } | | ^ { 2 }\tag{7.11}
$$

$$
d i a g ( \mathbf { x } ) \mathcal { L } , \mathcal { R } = - S V D ( d i a g ( \mathbf { x } ) ( E _ { Q } + E _ { S } ) )\tag{7.12}
$$

We refer to this method of computing saliency-based low-rank adapters as SLIM-LoRA, a practical and eficient approach tailored for addressing compression errors in large language models. To ensure numerical stability and guarantee the invertibility of the saliency function, an identity matrix with small values can be added to $d i a g ( { \bf x } )$ . This adjustment is equivalent to uniformly shifting all elements of x and ensures that the saliency function remains robust even when x contains near-zero elements. Algorithm 6 provides a comprehensive overview of the steps involved in computing saliency-based low-rank adapters using SLIM-LoRA,

Algorithm 6 SLIM-LoRA Saliency-based Low-rank Adapter Computation   
Input: Original weight W, compressed weight $\mathcal { W } ^ { C }$ , calibration input X.   
Output: Saliency-based low-rank adapters ${ \mathcal { L } } , { \mathcal { R } } .$   
1 $E _ { C }  E _ { Q } + E _ { S } = \mathcal { W } ^ { C } - \mathcal { W }$ ▷ Compute error   
2 $\tilde { \mathbf { x } } \gets \mathrm { m e a n } ( \mathcal { X } )$ ▷ Average over all the samples   
3 $\mathbf { x } \gets \tilde { \mathbf { x } } \mathrm { + m i n } \big ( | \tilde { \mathbf { x } } | \big )$ ▷ Shift values to avoid zeros in x   
4 ${ \cal S } _ { \cal C }  \mathrm { d i a g } ( { \bf x } ) { \cal E } _ { \cal C }$ ▷ Compute error saliency   
5 $\tilde { \mathcal { L } } , \tilde { \mathcal { R } } \gets \mathrm { S V D } ( \tilde { S } _ { C } )$ ▷ Low-rank approximation   
6 $\mathcal { L } \gets \mathrm { d i a g } ( 1 / \mathbf { x } ) \tilde { \mathcal { L } }$ ▷ Converting saliency to weight   
7 $\mathcal { R }  \tilde { \mathcal { R } }$   
Return: ${ \mathcal { L } } , { \mathcal { R } } .$

ensuring reproducibility and clarity.

## 7.4.3 Low-rank adapter quantization

While pruning and quantizing the weights significantly reduce the model’s computation and memory requirements (∼8× memory footprint reduction), incorporating full-precision low-rank adapters reintroduces over head, partially ofsetting these gains. To address this, we apply 4-bit quantization to compress the adapters. This step ensures that the compression eficiency achieved through weight pruning and quantization is preserved, while maintaining the performance benefits of the low-rank adapters.

Quantizing low-rank adapters poses unique challenges due to the long-tailed distribution of their elements, which limits the efectiveness of advanced non-group quantization methods, such as SLIM-Quant. To address this, we adopt an AbsMax group quantization scheme for the adapters, where groups of 128 elements share the same quantization parameter. By grouping elements, this method efectively captures the distribution’s variability while minimizing quantization error, striking a balance between accuracy and compression. This approach not only reduces the adapter overhead by 4× but ensures that their contribution to overall model compression and performance is retained; as demonstrated in our experimental evaluation.

## 7.4.4 Optional Post-Compression Fine-Tuning

Fine-tuning large language models post-compression has many challenges because the high parameter count and memory demands of traditional methods make them computationally prohibitive. For example, using a simple optimizer such as ADAMW leads to 4× additional memory overhead to store gradient and optimizer states, rendering these approaches impractical for compressed models. Thus, parameter-eficient fine-tuning is essential for preserving the benefits of compression while avoiding excessive computational and memory costs. This necessity is further highlighted by the results in Section 7.5, which illustrate the overheads of traditional fine-tuning and the advantages of parameter-eficient alternatives.

To overcome the challenges of fine-tuning compressed models, SLIM employs parameter-eficient lowrank adapters as the only tunable components during the fine-tuning phase. During this optional phase, SLIM freezes the sparse and quantized weights, enabling focused fine-tuning solely on the adapters. If the adapters are quantized, SLIM uses a straight-through estimator (STE) for quantization-aware fine-tuning and reduces its overheads with custom quantization and dequantization kernels implemented in Triton. This parameter-eficient fine-tuning method allows rapid accuracy improvements for the compressed model, requir ing only a short fine-tuning phase over thousands of tokens. By limiting the fine-tuning process to a small subset of parameters, SLIM significantly reduces computational requirements while ensuring the model can adapt efectively to new data or tasks. This approach maintains the benefits of compression while enabling eficient adaptation, as demonstrated by the significant improvements achieved during fine-tuning.

## 7.5 Experimental results

Models, Datasets, and Evaluation. We evaluate SLIM on the OPT [165] and LLaMA-2 [144] model families, both of which serve as standard baselines in model compression studies [90, 36, 136]. Model accuracy is assessed on a range of zero-shot downstream tasks, including MMLU [57], Piqa [13], Arc-Easy, Arc-Challenge [21], WinoGrande [126], and OpenBookQA [97]. For zero-shot evaluations, we utilize the Language Model Evaluation Harness [41] framework. In line with prior work [136, 36, 90], we also report the perplexity of the models on a language modeling task on the WikiText2 [94] dataset, provided in Section E.4.

Baselines. We compare SLIM against state-of-the-art one-shot pruning methods, including Wanda [136], SparseGPT [36], and Magnitude Pruning [52], as well as one-shot quantization techniques like OPTQ [37], OmniQuant [129], AfineQuant [90], $\mathrm { L } ^ { 2 } \mathrm { Q E R }$ [162], and AbsMax. Additionally, we extend Joint Sparsification and Quantization (JSQ) [49] to support 4-bit weight quantization and include it in our experiments. To ensure fairness, we use the optimal hyperparameters reported for each method, or the default hyperparameters if not explicitly reported. For a thorough description of the notations used to show the diferent variants of SLIM, please see Table E.1 in Section E.1. The hyperparameters used in our experiments are detailed below.

To the best ofour knowledge, $\mathrm { L } ^ { 2 } \mathrm { Q E R }$ is the only compression method utilizing zero-shot low-rank adapters to enhance model accuracy. Our approach, SLIM, significantly diverges from $\mathrm { L } ^ { 2 } \mathrm { Q E R }$ in several key aspects. First, we employ saliency-based low-rank adapters to mitigate compression loss in quantized and sparse mod els, whereas $\mathrm { L ^ { 2 } Q E R }$ is tailored exclusively for quantization, resulting in reduced accuracy when combined with sparsity, as demonstrated in the subsequent sections. Second, we introduce SLIM-Quant , which lowers the overhead and complexity of group quantization compared to methods like $\mathrm { L } ^ { 2 } \mathrm { Q E R }$ . Finally, SLIM compresses and fine-tunes low-rank adapters eficiently to minimize overhead. In contrast, $\mathrm { L } ^ { 2 } \mathrm { Q E R }$ relies on fullprecision low-rank adapters, which incur additional overhead and do not benefit from the parameter-eficient fine-tuning proposed in our work.

Experiment Setup. Similar to Wanda, SparseGPT, and OPTQ, SLIM uses 128 sequences sampled from the C4 [121] dataset for calibration, and 300,000 tokens from C4 for all fine-tuning experiments. SLIM-Quant uses a histogram of weight elements to find the optimal scaling factor, with the number of bins set to max(512, min $( \frac { d _ { i n } \times d _ { o u t } } { 1 0 0 0 } , 2 0 , 0 0 0 ) \rangle$ to achieve an accurate approximation. All quantization experiments fol low a 4-bit weight-only scheme with a group size of 128, consistent with prior work (OPTQ, OmniQuant, AfineQuant, etc.). For experiments involving Naive-LoRA and SLIM-LoRA, the adapter rank is set to 10% of the model’s hidden dimension unless stated otherwise. Fine-tuning is performed with the HuggingFace Trainer [153] using the AdaFactor [131] optimizer with linear learning rate scheduling and default parameters. We use BFloat-16 [148] on NVIDIA A100 GPUs, with a local batch size of 1 and gradient accumulation fac tor of 64 to reduce memory overhead. Weight updates for sparse and/or quantized weights and corresponding biases are disabled during fine-tuning.

Accuracy results. We evaluate the accuracy of SLIM and other state-of-the-art pruning and quantization methods across 2:4 and unstructured sparsity benchmarks, highlighting SLIM’s superiority in Table 7.1. SparseGPT and Group $\mathrm { O P T Q } .$ designed to work together, achieve competitive performance. For other advanced quantization methods, we pruned models using Wanda and quantized the sparse checkpoints with Group AbsMax, AWQ, OmniQuant, and AfineQuant, reporting the best results (detailed in Section E.5). Notably, methods like OmniQuant and AfineQuant struggle to quantize OPT-350M, often resulting in NaN values. Moreover, AWQ, OmniQuant, AfineQuant, and L<sup>2</sup>QER encounter out-of-memory (OOM) errors when compressing models on a single A100-40GB GPU. While JSQ performs well for the LLaMA-2 family, its dificulty compressing the OPT family limits its broader applicability.

Table 7.1: Average zero-shot accuracy of LLaMA-2 and OPT models with 50% sparsity and 4-bit weight quantization. Best Method∗ indicates the best quantization method out of Group AbsMax, AWQ, OmniQuant, and AfineQuant. ↑ indicates better performance.
<table><tr><td>Pruning/LoRA</td><td>Weight</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>Method</td><td>Quantization</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>-</td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>2:4 Sparsity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude</td><td>Group AbsMax</td><td>32.19</td><td>31.94 33.38</td><td>33.82 38.75</td><td>33.43</td><td>34.81 44.32</td><td>34.68 45.64</td><td>44.64</td><td>44.18</td></tr><tr><td>SparseGPT</td><td>Group OPTQ</td><td>33.70</td><td></td><td></td><td>40.15</td><td></td><td></td><td>45.49</td><td>51.05</td></tr><tr><td>Wanda</td><td>Best Method*</td><td>33.39</td><td>32.79</td><td>38.43</td><td>40.00</td><td>43.41</td><td>44.07</td><td>44.86</td><td>48.94</td></tr><tr><td>JSQ L2QER</td><td>JSQ</td><td>31.98</td><td>31.13</td><td>36.34</td><td>31.79</td><td>41.33</td><td>37.38</td><td>45.34</td><td>49.45</td></tr><tr><td></td><td>Group AbsMax</td><td>33.34</td><td>31.68</td><td>36.68</td><td>38.11</td><td>41.37</td><td>O0M</td><td>43.77</td><td>00M</td></tr><tr><td>Naive-LoRA</td><td>SLıM-Quant</td><td>34.28</td><td>33.38</td><td>38.36</td><td>41.21</td><td>44.91</td><td>45.25</td><td>48.45</td><td>51.94</td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>34.62</td><td>34.36</td><td>40.61</td><td>42.73</td><td>45.99</td><td>46.09</td><td>51.15</td><td>54.94</td></tr><tr><td>SLiM-LoRAQ</td><td>SLıM-Quant</td><td>34.43</td><td>34.30</td><td>40.11</td><td>42.37</td><td>46.33</td><td>46.24</td><td>51.02</td><td>53.55</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude</td><td>Group AbsMax</td><td>33.34</td><td>33.51</td><td>32.12</td><td>39.90</td><td>36.44</td><td>32.33</td><td>47.03</td><td>51.04</td></tr><tr><td>SparseGPT</td><td>OPTQ</td><td>35.10</td><td>35.13</td><td>38.72</td><td>43.43</td><td>46.97</td><td>47.38</td><td>51.09</td><td>55.94</td></tr><tr><td>Wanda</td><td>Best Method*</td><td>35.11</td><td>33.89</td><td>41.02</td><td>42.89</td><td>46.52</td><td>46.84</td><td>53.62</td><td>56.76</td></tr><tr><td>JSQ</td><td>JSQ</td><td>32.05</td><td>31.09</td><td>39.53</td><td>33.35</td><td>41.04</td><td>31.80</td><td>52.08</td><td>57.00</td></tr><tr><td>L2QER</td><td>Group AbsMax</td><td>34.45</td><td>34.45</td><td>38.38</td><td>41.28</td><td>45.08</td><td>00M</td><td>50.60</td><td>OOM</td></tr><tr><td>Naive-LoRA</td><td>SLıM-Quant</td><td>34.77</td><td>34.23</td><td>40.40</td><td>43.37</td><td>46.64</td><td>47.30</td><td>51.52</td><td>55.33</td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>35.20</td><td>35.32</td><td>41.85</td><td>43.48</td><td>47.08</td><td>47.96</td><td>54.26</td><td>57.85</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>35.35</td><td>35.13</td><td>41.74</td><td>43.63</td><td>47.16</td><td>47.86</td><td>54.18</td><td>57.33</td></tr></table>

The progression from Naive-LoRA to SLIM-LoRA and $\mathbf { S L I M - L o R A } ^ { Q }$ demonstrates the benefits of incorporating weight saliency into low-rank adapters and applying quantization for reducing overhead. While Naive-LoRA improves model accuracy across diferent sizes, SLIM-LoRA achieves additional gains by effectively leveraging the saliency of the weights in the adapter design. Extending this, SLIM-LoRA<sup>Q</sup> applies quantization to the low-rank adapters, further minimizing overhead with minimal impact on accuracy, adding negligible improvements or degradation to the accuracy of the model.

Table 7.2 demonstrates how lightweight fine-tuning (FT) improves the accuracy of both SLIM-LoRA and Naive-LoRA, with SLIM-LoRA exhibiting greater gains due to its saliency-aware design. Further details on the fine-tuning process and its overhead are provided in Section E.8, illustrating its practicality for enhancing compressed model performance.

Integration with Weight Update. Chapter 5 proposes a method to compute the optimal per-layer weight updates given a calibration dataset. After determining the low-rank adapter values in SLIM, we can find the optimal weight values by solving $\begin{array} { r } { \mathcal { W } ^ { C * } \ = \ \arg \operatorname* { m i n } _ { \mathcal { W } ^ { C } } \| \mathcal { X } \mathcal { W } ^ { C } \ - \ \mathcal { X } ( \mathcal { W } - \mathcal { L } \mathcal { R } ) \| } \end{array}$ using the QP solvers introduced in OPTIMA. As shown in Table 7.3, our results indicate that applying the compression trinity with OPTIMA as the weight update and SLIM-LoRA as the low-rank adapters can further boost the accuracy of the models.

Table 7.2: Efects of fine-tuning on the average zero-shot accuracy of LLaMA-2 models with 50% sparsity and 4-bit weight quantization. ↑ indicates better performance.
<table><tr><td>Pruning/LoRA Method</td><td>Weight Quantization</td><td>LLaMA-2 7B</td><td>13B</td></tr><tr><td>Dense</td><td>–</td><td>56.6</td><td>60.8</td></tr><tr><td>50% 2:4  $\mathrm { N a i v e - L o R A + F T }$   $\mathrm { S L I M - L o R A + F T }$   $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant SLrM-Quant SLıM-Quant</td><td>50.89 52.12 48.31</td><td>55.70 56.60 56.50</td></tr><tr><td>50% Unstructured  $\mathrm { N a i v e - L o R A + F T }$   $\mathrm { S L I M - L o R A + F T }$   $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant  $\mathbf { S L I M - Q u a n t }$   $\mathbf { S L I M - Q u a n t }$ </td><td>52.90 54.69 53.57</td><td>57.08 57.96 57.78</td></tr></table>

Table 7.3: Accuracy results of OPTIMA weight update mechanism with SLIM-LoRA. ↑ indicates better performance.
<table><tr><td rowspan="2">Pruning/LoRA Method</td><td rowspan="2">Weight Quantization</td><td colspan="2">LLaMA-2</td></tr><tr><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>-</td><td>56.6</td><td>60.8</td></tr><tr><td>50% 2:4</td><td></td><td></td><td></td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>51.02</td><td>53.55</td></tr><tr><td> $\operatorname { S L I M - L o R A } ^ { Q } + \operatorname { O P T I M A }$ </td><td>SLıM-Quant</td><td>51.62</td><td>53.84</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SL1M-Quant</td><td>54.18</td><td>57.33</td></tr><tr><td> $\operatorname { S L I M - L o R A } ^ { Q } + \operatorname { O P T I M A }$ </td><td>SLrM-Quant</td><td>54.32</td><td>57.45</td></tr></table>

Integration with Hybrid Sparsity. As demonstrated in Chapter 6, PATCH improves upon rigid semi-structured sparsity by enabling adaptive, tile-level density. While the standard implementation of SLIM presented in this chapter utilizes Wanda [136] for eficient pruning, the modular design of the Compression Trinity allows us to substitute this component with more advanced sparsity operators. In this section, we integrate PATCH into the SLIM pipeline to evaluate the impact of hybrid sparsity on the fully compressed model.

We apply the SLIM-LoRA error correction mechanism on top of the hybrid masks generated by PATCH. For these specific experiments, we utilize Group AbsMax quantization to isolate the benefits of the hybrid sparsity pattern when combined with standard quantization schemes. Table 7.4 reports the results on LLaMA-2 7B and LLaMA-3.1 8B. The results demonstrate that combining the flexible sparsity of PATCH with quantization and low-rank approximation enables controllable tradeofs between compression ratio and model quality. This confirms that the enhancements made to the sparsity pillar in the previous chapter translate directly to improved flexibility in the joint compression setting.

Table 7.4: Average accuracy (↑ indicates better) across eight zero-shot downstream tasks (including RACE [74] and HellaSwag [161]) and WikiText2 perplexity (↓ indicates better) of compressed models with 4-bit weight-only quantization. Please note that using LoRA adds additional parameters to the model.
<table><tr><td rowspan="2">Sparsity Method</td><td rowspan="2"></td><td rowspan="2">Pattern</td><td rowspan="2">LoRA</td><td colspan="2">LLaMA-2-7B</td><td colspan="2">LLaMA-3.1-8B</td></tr><tr><td>Acc (% ↑) PPL (↓)</td><td></td><td>Acc (% ↑) PPL (↓)</td><td></td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>=</td><td>54.61</td><td>5.12</td><td>60.31</td><td>5.84</td></tr><tr><td>50%</td><td>MaskLLM 2:4</td><td></td><td>=</td><td>47.98</td><td>7.64</td><td>51.12</td><td>9.92</td></tr><tr><td>45%</td><td>PATCHTile</td><td>Dense/2:4 Tiles -</td><td></td><td>48.19</td><td>7.34</td><td>52.47</td><td>9.68</td></tr><tr><td>45%</td><td>PATCHTile</td><td>Dense/2:4 Tiles SLiM-LoRA</td><td></td><td>50.71</td><td>6.83</td><td>54.04</td><td>9.12</td></tr><tr><td>35%</td><td>PATCHTile</td><td>Dense/2:4 Tiles -</td><td></td><td>49.38</td><td>6.92</td><td>53.81</td><td>9.26</td></tr><tr><td>35%</td><td>PATCHTile</td><td>Dense/2:4 Tiles SLiM-LoRA</td><td></td><td>51.91</td><td>6.42</td><td>55.70</td><td>8.37</td></tr><tr><td>25%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td> -</td><td>50.45</td><td>6.57</td><td>55.45</td><td>8.69</td></tr><tr><td>25%</td><td>PATCHTile</td><td>Dense/2:4 Tiles SLiM-LoRA</td><td></td><td>52.62</td><td>6.11</td><td>56.99</td><td>7.77</td></tr></table>

Comparison of large compressed and small dense models. This section compares large compressed models with dense models of equivalent parameter size, ofering guidelines for configuration selection under hardware constraints. We focus on 2:4 sparsity due to its hardware acceleration support and evaluate the OPT model family, which spans a wide range of sizes for comprehensive analysis.

We analyze model performance by plotting average accuracy against parameter size, calculated as detailed in Section E.9. This visualization enables a direct performance comparison between models with an equal number of bits.

Figure 7.2 presents the accuracy results of the OPT model family across diferent compression methods. The x-axis represents the model parameter size in gigabytes, while the y-axis denotes accuracy (higher is better). The results demonstrate that SLIM-LoRA<sup>Q</sup>, both with and without fine-tuning, consistently outperforms dense models and other compression techniques at the same parameter size. Notably, compressed models achieve higher accuracy than dense models of equivalent size, highlighting the efectiveness of the proposed method. This trend underscores the advantage of SLIM-LoRA<sup>Q</sup> in maximizing model eficiency under strict hardware constraints.

Speedup. Leveraging sparsity and quantization enhances GPU resource utilization, enabling faster model inference. Following Wanda’s experimental setup, we evaluate the speedup achieved across diferent model layers and sizes. Similar to Wanda, AWQ, and QuaRot [7], we focus on consumer-grade GPUs and conduct our experiments on NVIDIA RTX 3060 GPUs. Speedup results for NVIDIA A100 GPUs are provided in Section E.7.

SLIM achieves notable speedups through optimized sparse and quantized matrix multiplication, utilizing Sparse Marlin [38] integrated with vLLM [73]. For inference, we adopt small batch sizes during decoding, as recommended by prior works [154, 167]. Dense Quantized Marlin or PyTorch kernels handle the lowrank adapters based on their quantization status. Table 7.7 highlights the speedup achieved across diferent LLaMA-2 layers compared to dense, unquantized models. Larger matrices, such as those in self-attention and feed-forward modules, consistently yield greater speedups, aligning with trends detailed in Section E.7.

Sparse-only results. To evaluate the isolated impact of sparsity on model accuracy, we disable quantization and benchmark Magnitude Pruning, SparseGPT, and Wanda, alongside low-rank approximations like Wanda-SVD and SLIM . Our experiments assess both 50% unstructured sparsity and 2:4 structured sparsity patterns.

Accuracy vs. Model Parameter Size (2:4 Sparsity)  
![](images/af8ec17e6cbdc58b2e54fb51db9eceb1090244b2ab449f7cbfae650ff229b328.jpg)  
Figure 7.2: Accuracy results of the OPT family across diferent compression methods (↑ indicates better performance). At equal parameter size, SLIM outperforms both dense models and other compression techniques, demonstrating that model compression with SLIM yields superior performance under the same budget.

Table 7.5 shows the accuracy results for sparse models. Magnitude Pruning performs the worst, while Wanda and SparseGPT achieve comparable results, with larger accuracy gaps for semi-structured sparsity. Low-rank adapters improve accuracy, with SLIM leveraging saliency-based approximation for superior performance. A brief fine-tuning phase further boosts the accuracy of low-rank approximations.

Table 7.5: Average zero-shot accuracy of LLaMA-2 and OPT models with pruning. The quantization is disabled in this experiment. ↑ indicates better performance.
<table><tr><td>Pruning/LoRA</td><td colspan="5">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Method</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>2:4 Sparsity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude</td><td>32.6</td><td>31.8</td><td>35.4</td><td>33.9</td><td>36.4</td><td>30.7</td><td>31.2</td><td>32.0</td></tr><tr><td>SparseGPT</td><td>33.8</td><td>33.2</td><td>37.7</td><td>41.3</td><td>45.2</td><td>45.6</td><td>47.3</td><td>52.3</td></tr><tr><td>Wanda</td><td>34.0</td><td>32.5</td><td>38.3</td><td>40.5</td><td>43.2</td><td>44.1</td><td>46.1</td><td>49.7</td></tr><tr><td>SLiM-Naive</td><td>34.1</td><td>34.1</td><td>40.4</td><td>42.8</td><td>46.0</td><td>45.9</td><td>51.6</td><td>55.8</td></tr><tr><td>SLiM-Naive + FT</td><td>34.8</td><td>34.5</td><td>41.3</td><td>43.4</td><td>46.5</td><td>47.2</td><td>52.4</td><td>56.9</td></tr><tr><td>SLiM-LoRA</td><td>34.5</td><td>32.9</td><td>40.7</td><td>43.1</td><td>46.4</td><td>46.3</td><td>51.4</td><td>56.1</td></tr><tr><td>SLiM-LoRA + FT</td><td>35.1</td><td>34.9</td><td>41.5</td><td>43.8</td><td>46.5</td><td>47.3</td><td>51.6</td><td>56.4</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude</td><td>33.3</td><td>33.7</td><td>34.0</td><td>40.6</td><td>35.8</td><td>30.9</td><td>32.6</td><td>31.9</td></tr><tr><td>SparseGPT</td><td>35.5</td><td>35.1</td><td>39.6</td><td>43.5</td><td>47.4</td><td>47.8</td><td>53.3</td><td>57.3</td></tr><tr><td>Wanda</td><td>35.0</td><td>34.5</td><td>41.1</td><td>42.9</td><td>46.5</td><td>46.8</td><td>52.7</td><td>57.2</td></tr><tr><td>SLıM-Naive</td><td>35.3</td><td>35.2</td><td>41.9</td><td>44.1</td><td>47.5</td><td>47.8</td><td>54.9</td><td>58.5</td></tr><tr><td>SLiM-Naive + FT</td><td>35.74</td><td>35.7</td><td>42.7</td><td>44.6</td><td>47.8</td><td>48.4</td><td>54.9</td><td>58.7</td></tr><tr><td>SLiM-LoRA</td><td>35.2</td><td>35.1</td><td>42.0</td><td>44.1</td><td>47.7</td><td>48.2</td><td>55.0</td><td>58.8</td></tr><tr><td>SLiM-LoRA + FT</td><td>35.9</td><td>35.7</td><td>42.5</td><td>44.7</td><td>47.7</td><td>48.4</td><td>55.0</td><td>58.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Quantization-only results. To evaluate the impact of SLIM-Quant and low-rank compensation in SLIM, we conduct experiments without sparsity, testing quantization schemes like Group AbsMax, OPTQ, AWQ, OmniQuant, AfineQuant, L<sup>2</sup>QER, and SLIM-Quant . To enhance accuracy, we add low-rank adapters to SLIM-Quant and Group AbsMax, optimizing either error saliency (SLIM-LoRA) or reconstruction error norm (Naive-LoRA). Other quantization methods cannot incorporate low-rank adapters due to conflicting weight/activation update rules.

Table 7.6 presents the quantization results. Adding low-rank adapters to Group AbsMax significantly boosts model accuracy, outperforming most advanced methods. While SLIM-Quant alone is not designed for high accuracy, its integration with SLIM variants achieves results comparable to or better than Group AbsMax with low-rank adapters, highlighting the value of co-design in compression methods. Furthermore, a lightweight fine-tuning phase with SLIM-Quant delivers state-of-the-art accuracy.

Table 7.6: Average zero-shot accuracy of LLaMA-2 and OPT models with quantization. The sparsity is disabled in this experiment. ↑ indicates better performance.
<table><tr><td>Quantization</td><td>Low-rank</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>Method</td><td>Adapter</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td></td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>OPTQ</td><td></td><td>35.64</td><td>36.46</td><td>42.83</td><td>44.20</td><td>47.46</td><td>48.24</td><td>53.53</td><td>59.80</td></tr><tr><td>AWQ</td><td></td><td>36.16</td><td>31.83</td><td>42.98</td><td>45.28</td><td>48.45</td><td>48.76</td><td>53.97</td><td>00M</td></tr><tr><td>OmniQuant</td><td></td><td>35.46</td><td>NaN</td><td>42.15</td><td>44.71</td><td>46.65</td><td>00M</td><td>54.33</td><td>O0M</td></tr><tr><td>AffineQuant</td><td></td><td>35.73</td><td>NaN</td><td>42.62</td><td>44.92</td><td>47.91</td><td>OOM</td><td>54.52</td><td>00M</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Group AbsMax</td><td></td><td>35.45</td><td>36.67</td><td>42.57</td><td>44.79</td><td>48.30</td><td>48.49</td><td>55.56</td><td>60.12</td></tr><tr><td>Group AbsMax</td><td>L²QER</td><td>34.75</td><td>35.63</td><td>40.60</td><td>44.22</td><td>46.90</td><td>00M</td><td>55.95</td><td>O0M</td></tr><tr><td>Group AbsMax</td><td>SLıM-Naive</td><td>36.30</td><td>36.58</td><td>43.07</td><td>45.13</td><td>48.26</td><td>48.72</td><td>56.23</td><td>60.53</td></tr><tr><td>Group AbsMax</td><td>SLiM-LoRA</td><td>36.18</td><td>36.72</td><td>42.89</td><td>45.65</td><td>48.45</td><td>48.89</td><td>55.99</td><td>60.16</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLıM-Quant</td><td>=</td><td>31.98</td><td>36.46</td><td>36.19</td><td>40.08</td><td>45.61</td><td>38.27</td><td>31.11</td><td>30.51</td></tr><tr><td>SLıM-Quant</td><td>SLıM-Naive</td><td>35.29</td><td>36.02</td><td>42.48</td><td>45.01</td><td>47.75</td><td>48.38</td><td>55.96</td><td>60.85</td></tr><tr><td>SLıM-Quant</td><td>SLiM-LoRA</td><td>35.69</td><td>36.42</td><td>42.59</td><td>45.26</td><td>48.18</td><td>48.52</td><td>56.26</td><td>60.59</td></tr><tr><td>SLıM-Quant</td><td>SLiM-LoRA + FT</td><td>35.91</td><td>36.61</td><td>43.29</td><td>45.58</td><td>48.29</td><td>49.04</td><td>56.51</td><td>60.65</td></tr></table>

Additional Experiments. We provide additional experiments for a comprehensive evaluation in the appendix.

The Language Modeling Experiments (Section E.4) evaluate SLIM across sparse and quantized, sparseonly, and quantized-only models on WikiText-2. The results align with the accuracy trends reported in the preceding sections, further validating the efectiveness of SLIM .

The Fine-tuning Costs (Section E.8) show that SLIM reduces fine-tuning overhead from over 36 days for 13B parameter models to just 14 hours on a single GPU, demonstrating its practicality and eficiency.

We provide a comparison between Sparsity vs. Quantization (Section E.6) to show that combining 50% sparsity and 4-bit quantization helps achieve better compression results in comparison to solely using 2-bit quantization, while maintaining a similar compression ratio (∼8×).

Additional speedup results for SLIM on NVIDIA A100-40GB GPUs are provided in the Additional Speedup Results (Section E.7). A theoretical analysis of computation and memory reductions can be found in the Computation Reduction Analysis (Section E.10) and Memory Reduction Analysis (Section E.9), highlighting the eficiency of SLIM .

Compression Costs (Section E.11) details the time required to compress models of various sizes across diferent methods. Rank Analysis (Section E.12) explores how rank choices in low-rank adapters impact computational and memory costs, as well as model accuracy. Sparsity Analysis (Section E.15) analyzes the efects of diferent sparsity ratios on model compression. Lastly, Efects of Calibration Sample Count (Section E.13) evaluates the influence of calibration sample counts on the accuracy of calibration-based methods.

Table 7.7: LLaMA-2 family of models speedup (×) using SLIM compared to original dense unquantized model on NVIDIA RTX-3060. ↑ shows higher speedup.
<table><tr><td>Model Size</td><td>Batch Size</td><td>LoRA Type</td><td>Self- Attention</td><td>Up- Projection</td><td>Down Projection</td></tr><tr><td rowspan="4">7B</td><td rowspan="2">16</td><td>FP16</td><td>1.99</td><td>2.58</td><td>3.23</td></tr><tr><td>INT4</td><td>1.00</td><td>2.29</td><td>2.58</td></tr><tr><td rowspan="2">32</td><td>FP16</td><td>2.06</td><td>2.53</td><td>3.40</td></tr><tr><td>INT4 FP16</td><td>1.13 1.54</td><td>2.27 1.70</td><td>2.97 1.85</td></tr><tr><td rowspan="5">13B</td><td>64</td><td>INT4</td><td>0.96</td><td>1.62</td><td>1.76</td></tr><tr><td rowspan="2">16 32</td><td>FP16 INT4</td><td>2.18 1.28</td><td>2.53 3.24</td><td>2.60 3.17</td></tr><tr><td>FP16</td><td>2.23</td><td>2.68</td><td>2.91</td></tr><tr><td rowspan="2">64</td><td>INT4</td><td>1.43</td><td>2.96</td><td>3.20</td></tr><tr><td>FP16</td><td>1.38</td><td>1.78</td><td>1.67</td></tr><tr><td rowspan="5">70B</td><td>16</td><td>INT4 FP16</td><td>1.21 2.18</td><td>1.69 2.86</td><td>1.65 2.75</td></tr><tr><td rowspan="2">32</td><td>INT4</td><td>3.11</td><td>3.99</td><td>3.79</td></tr><tr><td>FP16</td><td>2.00</td><td>2.63</td><td>2.67 3.39</td></tr><tr><td rowspan="2">64</td><td>INT4</td><td>2.75</td><td>3.19</td><td></td></tr><tr><td>FP16 INT4</td><td>1.38 1.51</td><td>1.70 1.77</td><td>1.86 1.94</td></tr></table>

## 7.6 Conclusion

In this chapter, we presented SLIM, the complete fulfillment of the Compression Trinity framework. We began this thesis by identifying the ”Sparsity Paradox,” the observation that removing weights is structurally destructive and, when applied in isolation, leads to early accuracy collapse. SLIM resolves this paradox. By seamlessly integrating optimized uniform quantization (SLIM-Quant), hardware-friendly sparsity, and crucially mathematically derived low-rank error correction (SLIM-LoRA), we have turned the Low-Rank pillar into a restorative force. It recovers the information lost by the aggressive application of the first two pillars, solving the ”compounded error” challenge that has historically hindered joint compression.

An important direction for strengthening the theoretical foundations of SLIM is to draw more explicitly on the guarantees provided by Robust PCA theory. The classical PCP framework [15] establishes that the sparse-plus-low-rank decomposition is exactly recoverable under incoherence conditions, and the Stable PCP extension [169] shows that this recovery is robust to additional dense noise, a property that could be leveraged to account for quantization error. Translating these guarantees to the LLM compression setting would re quire verifying whether the incoherence assumptions hold for pre-trained weight matrices and characterizing the interaction between quantization noise, sparsity patterns, and low-rank structure. Furthermore, replacing SLIM’s current sequential pipeline with a joint optimization formulation, as explored by OATS [164], HASSLE-free [91], and 3BASiL [11], could yield tighter error bounds and potentially improve accuracy by avoiding the suboptimality inherent in sequential decomposition. Such a formulation would need to incorporate the quantization constraint, extending the standard sparse-plus-low-rank problem to a “quantized sparse-plus-low-rank” decomposition, an open problem that merits further investigation.

SLIM not only shifts the Pareto frontier, outperforming dense models at equal parameter budgets, but also serves as the final proof of our core thesis: that eficiency is not a singular optimization problem, but a multidimensional balancing act. We have traversed the complete arc of this life-cycle: from accelerating pretraining dynamics (Chapter 3, Chapter 4) to establishing the limits of sparsity in both static (Chapter 5) and dynamic (Chapter 6) regimes, and finally achieving a unified, one-shot solution for deployment (Chapter 7).

The next and final chapter will summarize these contributions, discuss the broader implications of the Compression Trinity for the future of eficient AI, and outline potential avenues for further research.

## Chapter 8

## Conclusion and Future Work

## 8.1 Summary of Contributions

This thesis has argued that the eficiency bottleneck in Large Language Models (LLMs) is not merely a resource constraint, but a methodological failure to integrate complementary compression principles. We have established that the “eficiency wall” encountered by isolated techniques is methodological rather than fundamental. By jointly applying the “Compression Trinity,” sparsity, quantization, and low-rank approximations, we have demonstrated that the distinct hardware bottlenecks of compute FLOPs, memory bandwidth, and parameter redundancy can be attacked simultaneously.

Crucially, we established that the Trinity is not merely a post-training optimization tool; it is a fundamental framework applicable to the entire life-cycle of the model.

## 8.1.1 The Trinity in Training Dynamics

We demonstrated that the training dynamics themselves are compressible. By selectively applying pillars of the Trinity during optimization, we proved that high-fidelity, dense updates are not strictly necessary for convergence.

• Sparse, Low-Rank, and Quantized Optimization (MKOR): In Chapter 3, we addressed the prohibitive computational penalty of second-order optimization. MKOR validates the Trinity’s utility in training by combining block diagonal Sparsity with Low-Rank approximations (rank-1 updates) and Quantization (to stabilize memory footprint). This approach reduces curvature update complexity from $\mathcal { O } ( d ^ { 3 } )$ to O(d<sup>2</sup>), accelerating convergence by up to 2.57× compared to first-order baselines and 1.75× compared to KFAC.

• Sparse and Low-Rank Training (SLOPE): In Chapter 4, we validated the concept of “Lossy Training” by integrating Sparsity and Low-Rank principles. By enforcing N:M sparsity in the backward pass and delaying low-rank recovery (“lazy” adapters) to the final 1% of training, we showed that the training process can withstand significant information loss without sacrificing final model accuracy.

## 8.1.2 The Trinity in Post-Training and Inference

For deployed models, we systematically dismantled the primary barriers to compression, compounded error and structural rigidity, resulting in a unified inference framework.

• Solving Compounded Error (OPTIMA): OPTIMA (Chapter 5) utilized global optimization to stabilize the Quantization pillar. By establishing that weight reconstruction must be solved via column-wise Quadratic Programs (QPs) with a shared Hessian, OPTIMA provides the foundational stability necessary to withstand aggressive compression.

• Breaking Rigidity (PATCH): PATCH (Chapter 6) advanced the Sparsity pillar by breaking the rigidity of hardware-enforced patterns. By introducing learnable tile-level hybrid sparsity, we proved that sparsity ratios can be continuous and adaptive (0–50%) rather than discrete, preserving density in information-critical layers.

• Unified Inference (SLIM): The empirical validation of the full framework culminates in SLIM (Chapter 7). SLIM represents the simultaneous integration of the full Trinity, aggressive Quantization, semi-structured Sparsity, and saliency-based Low-Rank adapters, during the inference phase. It shifts the Pareto frontier of model eficiency, demonstrating that a fully compressed model can improve accuracy by up to 5.66% over state-of-the-art methods and, in specific configurations, outperform uncompressed dense models at equivalent parameter budgets. The industrial significance of these findings, particularly the necessity of 2:4 sparsity in modern production stacks, was subsequently featured in our technical analysis on the PyTorch blog<sup>1</sup>.

## 8.2 Exploratory Frameworks and Open Research

The core pillars of the Compression Trinity, sparsity, quantization, and low-rank approximations, provide a rigorous foundation for model eficiency. However, the application of these principles need not be confined to the rigid structures of traditional academic publication cycles. In parallel with the formal chapters of this thesis, we have developed agile, exploratory frameworks that extend the Trinity into new domains of adaptability and rapid deployment. These works, released as open-research contributions, demonstrate the flexibility of our methodology in addressing emerging challenges in the LLM landscape.

## 8.2.1 BEAM: Blockwise Error Minimization for One-shot Compression of LLMs

Standard post-training compression techniques (e.g., GPTQ, Wanda) often hit an accuracy ceiling because they optimize weights locally (layer-wise) without accounting for the non-linear interactions across the full transformer block. Conversely, full model fine-tuning (e.g., LoRA) is computationally expensive and requires curated datasets.

To bridge this gap, we introduced BEAM<sup>2</sup>, a framework for one-shot compression that requires no endto-end retraining. BEAM re-frames the compression problem by treating the intermediate activations of the original, uncompressed model as the “ground truth” for the compressed model. By splitting the LLM into independent transformer blocks and optimizing the compressed weights to minimize the feature reconstruction error of each block, BEAM captures the non-linear dependencies lost by simple layer-wise techniques. Crucially, this method is orthogonal to the specific compression type; it serves as a universal refinement stage that can recover up to 4.34% accuracy on sparse and quantized models using a single GPU in under four hours.

## 8.2.2 LEAP: Learnable End-to-End Adaptive Pruning of LLMs

In Chapter 6, we explored hybrid structural sparsity to balance hardware eficiency with information retention. However, imposing any structure (even a hybrid one) inherently limits the model’s expressivity compared to unstructured pruning.

$\mathbf { L E A P } ^ { 3 }$ challenges the assumption that unstructured sparsity must be static or heuristically determined (e.g., magnitude-based pruning). Instead, LEAP introduces a fully diferentiable masking mechanism where the binary inclusion of every parameter is treated as a learnable latent variable during training. By relaxing the discrete mask into a continuous probability distribution and applying straight-through estimation, LEAP allows the model to dynamically “evolve” its own sparsity pattern end-to-end. This approach reveals tha optimal sparsity is not fixed; it shifts during training as the model specializes, suggesting that the “Trinity” can eventually include topology as a learnable parameter alongside weights.

## 8.2.3 SLICE: Selecting Layer-wise Configurations for Matryoshka-Style LLMs

The prevailing paradigm in LLM deployment is “one size fits all”: a model is compressed to a fixed target (e.g., 4-bit, 50% sparsity) and deployed. This rigidity is ineficient for dynamic environments where hardware availability changes in real-time.

SLICE<sup>4</sup> extends the concept of Matryoshka Representation Learning to the compression configuration itself. SLICE trains a single “super-model” capable of operating at multiple eficiency tiers simultaneously. By solving for a nested set of configurations (e.g., a 2-bit core nested within a 4-bit shell), SLICE enables elastic deployment: the same model can instantaneously shed layers or precision bits to meet strict latency deadlines, or expand to full capacity when resources permit. This work points toward a future where the Compression Trinity is not a static compilation step, but a dynamic runtime state.

## 8.3 Limitations of the Current Approach

While the Compression Trinity ofers a robust framework, the specific implementations proposed in this thesis are subject to constraints that afect their immediate generalizability and ease of adoption.

Hardware Coupling and Portability. Our methods, particularly PATCH and the semi-structured sparsity utilized in SLOPE and SLIM, are currently tightly coupled to the NVIDIA Sparse Tensor Core architecture (2:4 sparsity). While efective on dominant hardware, the transferability of these specific patterns to non-NVIDIA accelerators (e.g., TPUs, AMD MI-series) or general-purpose CPUs remains unproven. Furthermore, the reliance on custom CUDA and Triton kernels introduces a significant software portability barrier, efectively restricting these optimizations to advanced engineering environments.

Training Instability Risks. Although MKOR introduces stabilizers to mitigate exploding gradients, secondorder optimizers remain inherently more sensitive to hyperparameters than robust first-order methods like AdamW. The introduction of additional hyperparameters for curvature approximation imposes a tuning burden on practitioners, potentially ofsetting the wall-clock speed gains in experimental settings.

Pipeline Complexity. The “Compression Trinity” introduces significant engineering overhead. Deploying a pipeline that requires simultaneous quantization, pruning, and low-rank adaptation (as in SLIM) is considerably more complex to implement, debug, and maintain than simpler post-training quantization techniques (e.g., INT8/FP8). This complexity represents a barrier to adoption for practitioners seeking “plug-and-play” solutions.

Fairness and Diferential Impact on Underrepresented Populations. Throughout this thesis, compression quality is measured by aggregate accuracy on standard benchmarks, yet matching the uncompressed model’s overall accuracy does not guarantee uniform performance across all subpopulations. Pruning, quantization, and low-rank factorization remove model capacity that may disproportionately encode knowledge about underrepresented groups, a phenomenon that aggregate metrics can mask entirely. For instance, a compressed language model may preserve perplexity on high-resource languages such as English while exhibiting signif icant degradation on low-resource languages that were sparsely represented in the fine-tuning or calibration datasets. Similarly, in classification tasks, accuracy on minority demographic groups may sufer even when the overall metric remains stable. Because our calibration and evaluation pipelines rely on datasets that pre dominantly reflect majority populations, we cannot rule out that the proposed compression methods amplify existing biases or introduce new disparities. A rigorous fairness audit, disaggregating performance across languages, dialects, demographic groups, and downstream tasks, is an important direction that lies outside the scope of this work but is essential before deploying compressed models in high-stakes applications.

Scale of Evaluation. Our empirical validation focused on models in the 125M to 70B parameter range (e.g., OPT, LLaMA-2/3). As scaling laws push frontier models into the trillion-parameter regime, emergent behaviors or shifting bottlenecks (e.g., massive cross-node communication overheads) may alter the efectiveness of these compression techniques, a domain this thesis leaves unexplored.

## 8.4 Future Research Directions

Compressing the Context Window. This thesis focused heavily on Linear layers, which currently dominate compute. However, as sequence lengths grow to 1M+ tokens, the Attention mechanism and Key-Value (KV) cache become the dominant memory bottlenecks. Future work must extend the Trinity to the context window: quantizing dynamic KV states, inducing sparsity in attention patterns (e.g., via Sliding Window or Block-Sparse Attention), and applying low-rank approximations to the attention heads themselves to enable infinitecontext reasoning on commodity hardware.

Hardware-Algorithm Co-design. Current sparse formats incur a memory overhead for storing metadata (indices), which can negate compression gains at lower bit-widths. Future compression research cannot occur in a software vacuum; it requires a co-design approach to propose “hardware-defining” sparse formats. We envision a move toward algorithmic sparsity where the pattern is deterministic or predicted, eliminating the need for explicit index storage and further reducing the memory footprint.

The Trinity for Activations. While we successfully compressed weights, activation tensors remain a challenge during the prefill phase. Future work should investigate applying PATCH-like learnable masks to dynamic activation tensors, enabling “activation sparsity” that can accelerate the compute-bound prefill phase without requiring full retraining.

## 8.5 Closing Remarks

Ultimately, this thesis posits that eficiency is not merely a post-hoc optimization, but a fundamental design constraint. We have argued that the “eficiency wall” is a methodological artifact, dissolvable by the joint application of sparsity, quantization, and low-rank approximations.

By proving that the “Compression Trinity” can be integrated into every stage of the LLM life-cycle, from the training dynamics of MKOR to the inference engines of SLIM, we pave the way for a new paradigm of model design. In this paradigm, models are not simply made smaller; they are architected from the ground up to be dense in knowledge yet sparse in computation, rendering high-intelligence AI fundamentally more accessible and sustainable.

## Bibliography

[1] Abhinav Agarwalla, Abhay Gupta, Alexandre Marques, Shubhra Pandit, et al. Enabling High-Sparsity Foundational LLaMA Models with Eficient Pretraining and Deployment. arXiv preprint arXiv:2405.03594, 2024.

[2] Armen Aghajanyan, Sonal Gupta, and Luke Zettlemoyer. Intrinsic Dimensionality Explains the Efectiveness of Language Model Fine-Tuning. In Proceedings of the 59th Annual Meeting of the Associa tion for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7319–7328, Online, August 2021. Association for Computational Linguistics.

[3] Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, et al. QSGD: Randomized Quantization for Communication-Eficient Stochastic Gradient Descent. In NeurIPS, 2017.

[4] Shun-Ichi Amari. Natural Gradient Works Eficiently in Learning. Neural Computation, 10(2):251– 276, 1998.

[5] Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, et al. Scalable Second Order Optimization for Deep Learning. arXiv preprint arXiv:2002.09018, 2020.

[6] Argonne Leadership Computing Facility. Polaris. https://www.alcf.anl.gov/polaris.

[7] Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, et al. QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. In NeurIPS, 2024.

[8] Nihat Ay. On the Locality of the Natural Gradient for Learning in Deep Bayesian Networks. Information Geometry, pages 1–49, 2020.

[9] Jimmy Ba, Roger Grosse, and James Martens. Distributed Second-Order Optimization Using Kronecker-Factored Approximations. In ICLR, 2017.

[10] Abhimanyu Rajeshkumar Bambhaniya, Amir Yazdanbakhsh, Suvinay Subramanian, Sheng-Chun Kao, et al. Progressive Gradient Flow for Robust N:M Sparsity Training in Transformers. arXiv preprint arXiv:2402.04744, 2024.

[11] Kayhan Behdin, Mehdi Makni, and Rahul Mazumder. 3BASiL: An algorithmic framework for sparse plus low-rank compression of LLMs. arXiv preprint arXiv:2603.01376, 2025.

[12] Dimitris Bertsimas, Ryan Cory-Wright, and Nicholas AG Johnson. Sparse Plus Low Rank Matrix Decomposition: A Discrete Optimization Approach. JMLR, 2023.

[13] Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. PIQA: Reasoning About Physical Commonsense in Natural Language. In AAAI, 2020.

[14] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, et al. Language Models Are Few-Shot Learners. Advances in Neural Information Processing Systems, 33:1877–1901, 2020.

[15] Emmanuel J. Candès, Xiaodong Li, Yi Ma, and John Wright. Robust Principal Component Analysis? Journal ofthe ACM, 58(3):1–37, 2011.

[16] Venkat Chandrasekaran, Sujay Sanghavi, Pablo A. Parrilo, and Alan S. Willsky. Rank-Sparsity Incoherence for Matrix Decomposition. SIAM Journal on Optimization, 21(2):572–596, 2011.

[17] Beidi Chen, Tri Dao, Eric Winsor, Zhao Song, et al. Scatterbrain: Unifying Sparse and Low-Rank Attention Approximation. arXiv preprint arXiv:2110.15343, 2021.

[18] Stanley F. Chen, Douglas Beeferman, and Roni Rosenfeld. Evaluation Metrics for Language Models. Technical report, Carnegie Mellon University, 1998.

[19] Stanley F Chen, Douglas Beeferman, and Roni Rosenfeld. Evaluation Metrics for Language Models. Carnegie Mellon University, 1998.

[20] Zhaodong Chen, Zheng Qu, Yuying Quan, Liu Liu, et al. Dynamic N:M Fine-Grained Structured Sparse Attention Mechanism. In PPoPP, 2023.

[21] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, et al. Think You Have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457, 2018.

[22] Compute Canada. Compute Canada. https://computecanada.ca/.

[23] NVIDIA Corporation. CUTLASS 4.2.0: CUDA Templates for Linear Algebra Subroutines. https: //github.com/NVIDIA/cutlass, 2025. Also see: Kerr, A., Merrill, D., Demouth, J., Tran, J. ”CUTLASS: Fast Linear Algebra in CUDA C++”, NVIDIA blog, Dec. 2017.

[24] Tri Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. In ICLR, 2024.

[25] Tri Dao, Beidi Chen, Kaizhao Liang, Jiaming Yang, et al. Pixelated Butterfly: Simple and Eficient Sparse Training for Neural Network Models. In ICLR, 2022.

[26] Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, et al. FlashAttention: Fast and Memory-Eficient Exact Attention with IO-Awareness. In NeurIPS, 2022.

[27] Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-Bit Matrix Multi plication for Transformers at Scale. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, NeurIPS, pages 30318–30332. Curran Associates, Inc., 2022.

[28] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient Finetuning of Quantized LLMs. In NeurIPS, 2023.

[29] Tim Dettmers and Luke Zettlemoyer. Sparse Networks from Scratch: Faster Training without Losing Performance. arXiv preprint arXiv:1907.04840, 2019.

[30] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-Training of Deep Bidirectional Transformers for Language Understanding. In NAACL, pages 4171–4186, 2019.

[31] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, et al. The LLaMA 3 Herd of Models. arXiv preprint arXiv:2407.21783, 2024.

[32] Ruibo Fan, Xiangrui Yu, Peijie Dong, Zeyu Li, et al. SpInfer: Leveraging Low-Level Sparsity for Eficient Large Language Model Inference on GPUs. In Proceedings of the Twentieth European Conference on Computer Systems, pages 243–260, 2025.

[33] Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, et al. MaskLLM: Learnable Semi Structured Sparsity for Large Language Models. In NeurIPS, 2024.

[34] Jonathan Frankle, Gintare Karolina Dziugaite, Daniel Roy, and Michael Carbin. Linear Mode Connectivity and the Lottery Ticket Hypothesis. In ICML, 2020.

[35] Elias Frantar and Dan Alistarh. Optimal Brain Compression: A Framework for Accurate Post-Training Quantization and Pruning. NeurIPS, 35:4475–4488, 2022.

[36] Elias Frantar and Dan Alistarh. SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. In ICML, 2023.

[37] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate Quantization for Generative Pre-Trained Transformers. In ICLR, 2022.

[38] Elias Frantar, Roberto L Castro, Jiale Chen, Torsten Hoefler, et al. MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models. arXiv preprint arXiv:2408.11743, 2024.

[39] Trevor Gale, Erich Elsen, and Sara Hooker. The State of Sparsity in Deep Neural Networks. arXiv preprint arXiv:1902.09574, 2019.

[40] Leo Gao, Stella Biderman, Sid Black, Laurence Golding, et al. The Pile: An 800GB Dataset of Diverse Text for Language Modeling. arXiv preprint arXiv:2101.00027, 2020.

[41] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, et al. A Framework for Few-Shot Language Model Evaluation, 07 2024.

[42] Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, et al. The Language Model Evaluation Harness, 07 2024.

[43] Amir Gholami, Sehoon Kim, Zhen Dong, Zhewei Yao, et al. A Survey of Quantization Methods for Eficient Neural Network Inference. In Low-Power Computer Vision, pages 291–326. Chapman and Hall/CRC, 2022.

[44] Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. OpenWebText Corpus, 2019.

[45] Donald Goldfarb, Yi Ren, and Achraf Bahamou. Practical Quasi-Newton Methods for Training Deep Neural Networks. Advances in Neural Information Processing Systems, 33:2386–2396, 2020.

[46] Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge Distillation: A Survey. International Journal of Computer Vision, 129(6):1789–1819, 2021.

[47] Park Gunho, Park Baeseong, Kwon Se Jung, Kim Byeongwook, et al. nuQmm: Quantized MatMul for Eficient Inference of Large-Scale Generative Language Models. arXiv preprint arXiv:2206.09557, 2022.

[48] Han Guo, Philip Greengard, Eric P Xing, and Yoon Kim. LQ-LoRA: Low-Rank Plus Quantized Matrix Decomposition for Eficient Language Model Finetuning. In ICLR, 2024.

[49] Jinyang Guo, Jianyu Wu, Zining Wang, Jiaheng Liu, et al. Compressing Large Language Models by Joint Sparsification and Quantization. In ICML, 2024.

[50] Vineet Gupta, Tomer Koren, and Yoram Singer. Shampoo: Preconditioned Stochastic Tensor Optimization. In ICML, pages 1842–1850. PMLR, 2018.

[51] Song Han, Huizi Mao, and William J Dally. Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Hufman Coding. arXiv preprint arXiv:1510.00149, 2015.

[52] Song Han, Jef Pool, John Tran, and William Dally. Learning Both Weights and Connections for Eficient Neural Network. NeurIPS, 2015.

[53] Song Han, Jef Pool, John Tran, and William Dally. Learning Both Weights and Connections for Eficient Neural Network. Advances in Neural Information Processing Systems, 28, 2015.

[54] Babak Hassibi, David Stork, and Gregory Wolf. Optimal Brain Surgeon: Extensions and Performance Comparisons. NeurIPS, 1993.

[55] Babak Hassibi, David G Stork, and Gregory J Wolf. Optimal Brain Surgeon and General Network Pruning. In IEEE International Conference on Neural Networks, pages 293–299. IEEE, 1993.

[56] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

[57] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, et al. Measuring Massive Multitask Language Understanding. arXiv preprint arXiv:2009.03300, 2020.

[58] Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, et al. Sparsity in Deep Learning: Pruning and Growth for Eficient Inference and Training in Neural Networks. JMLR, 2021.

[59] Younes Hourri, Mohammad Mozafari, and Maryam Mehri Dehnavi. PATCH: Learnable Tile-Level Hybrid Sparsity for LLMs. arXiv preprint arXiv:2509.23410, 2025.

[60] Jeremy Howard and Sebastian Ruder. Universal Language Model Fine-Tuning for Text Classification. In ACL, pages 328–339, 2018.

[61] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, et al. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR, 2022.

[62] Yuezhou Hu, Kang Zhao, Weiyu Huang, Jianfei Chen, et al. Accelerating Transformer Pre-Training with 2:4 Sparsity. In ICML, 2024.

[63] Itay Hubara, Brian Chmiel, Moshe Island, Ron Banner, et al. Accelerated Sparse Neural Training: A Provable and Eficient Method to Find N:M Transposable Masks. NeurIPS, 2021.

[64] Ivan Ilin and Peter Richtarik. Thanos: A Block-Wise Pruning Algorithm for Eficient Large Language Model Compression. arXiv preprint arXiv:2504.05346, 2025.

[65] Benoit Jacob, Skirmantas Kligys, Bo Chen, Menglong Zhu, et al. Quantization and Training of Neural Networks for Eficient Integer-Arithmetic-Only Inference. In CVPR, 2018.

[66] Eric Jang, Shixiang Gu, and Ben Poole. Categorical Reparameterization with Gumbel-Softmax. In ICLR, 2017.

[67] Yu Ji, Ling Liang, Lei Deng, Youyang Zhang, et al. TETRIS: Tile-Matching the Tremendous Irregular Sparsity. NeurIPS, 2018.

[68] Keller Jordan, Yuchen Jin, Vlado Boza, You Jiacheng, Franz Cesista, Laker Newhouse, and Jeremy Bernstein. Muon: An Optimizer for Hidden Layers in Neural Networks, 2024.

[69] Sheng-Chun Kao, Amir Yazdanbakhsh, Suvinay Subramanian, Shivani Agrawal, et al. Training Recipe for N:M Structured Sparsity with Decaying Pruning Mask. arXiv preprint arXiv:2209.07617, 2022.

[70] Diederik P Kingma and Jimmy Ba. Adam: A Method for Stochastic Optimization. In ICLR, 2015.

[71] Alex Krizhevsky, Geofrey Hinton, et al. Learning Multiple Layers of Features from Tiny Images. Technical Report Tr-2009, University of Toronto, 2009.

[72] Alex Krizhevsky, Ilya Sutskever, and Geofrey E Hinton. ImageNet Classification with Deep Convolutional Neural Networks. Communications ofthe ACM, 60(6):84–90, 2017.

[73] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, et al. Eficient Memory Management for Large Language Model Serving with PagedAttention. In SOSP, 2023.

[74] Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, et al. RACE: Large-Scale ReAding Comprehension Dataset from Examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel, editors, EMNLP, pages 785–794, Copenhagen, Denmark, September 2017. Association for Computational Linguistics.

[75] Yann LeCun, John Denker, and Sara Solla. Optimal Brain Damage. Advances in Neural Information Processing Systems, 2, 1989.

[76] Jaeho Lee, Sejun Park, Sangwoo Mo, Sungsoo Ahn, et al. Layer-Adaptive Sparsity for the Magnitude-Based Pruning, 2021.

[77] Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. Measuring the Intrinsic Dimension of Objective Landscapes. In ICLR, 2018.

[78] Lujun Li, Peijie Dong, Zhenheng Tang, Xiang Liu, Qiang Wang, Wenhan Luo, Wei Xue, Qifeng Liu, Xiaowen Chu, and Yike Guo. Discovering Sparsity Allocation for Layer-Wise Pruning of Large Language Models. Advances in Neural Information Processing Systems, 37:141292–141317, 2024.

[79] Wei Li, Lujun Li, Mark Lee, and Shengjie Sun. Adaptive Layer Sparsity for Large Language Models via Activation Correlation Assessment. Advances in Neural Information Processing Systems, 37:109350– 109380, 2024.

[80] Yixiao Li, Yifan Yu, Qingru Zhang, Chen Liang, et al. LoSparse: Structured Compression of Large Language Models Based on Low-Rank and Sparse Approximation. In ICML, 2023.

[81] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, et al. AWQ: Activation-Aware Weight Quantization for On-Device LLM Compression and Acceleration. MLSys, 2024.

[82] Hong Liu, Sang Michael Xie, Zhiyuan Li, and Tengyu Ma. Same Pre-Training Loss, Better Down stream: Implicit Bias Matters for Language Models. In ICML, 2023.

[83] Hongyi Liu, Rajarshi Saha, Zhen Jia, Youngsuk Park, et al. ProxSparse: Regularized Learning of Semi-Structured Sparsity Masks for Pretrained LLMs. arXiv preprint arXiv:2502.00258, 2025.

[84] Zechun Liu, Haoyuan Mu, Xiangyu Zhang, Zichao Guo, et al. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. In ICCV, 2019.

[85] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. In ICLR, 2017.

[86] Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. In ICLR, 2019.

[87] Haihao Lu, Zedong Peng, and Jinwen Yang. MPAX: Mathematical Programming in JAX. arXiv preprint arXiv:2412.09734, 2024.

[88] Haihao Lu and Jinwen Yang. A Practical and Optimal First-Order Method for Large-Scale Convex Quadratic Programming. arXiv preprint arXiv:2311.07710, 2023.

[89] Yucheng Lu, Shivani Agrawal, Suvinay Subramanian, Oleg Rybakov, et al. STEP: Learning N:M Structured Sparsity Masks from Scratch with Precondition. arXiv preprint arXiv:2302.01172, 2023.

[90] Yuexiao Ma, Huixia Li, Xiawu Zheng, Feng Ling, et al. AfineQuant: Afine Transformation Quantization for Large Language Models. arXiv preprint arXiv:2403.12544, 2024.

[91] Mehdi Makni, Kayhan Behdin, Zheng Xu, Natalia Ponomareva, and Rahul Mazumder. A Unified Framework for Sparse Plus Low-Rank Matrix Decomposition for LLMs. In The Second Conference on Parsimony and Learning (Proceedings Track), 2025.

[92] James Martens. New Insights and Perspectives on the Natural Gradient Method. Journal of Machine Learning Research, 21(1):5776–5851, 2020.

[93] James Martens and Roger Grosse. Optimizing Neural Networks with Kronecker-Factored Approximate Curvature. In ICML, pages 2408–2417. PMLR, 2015.

[94] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Models, 2016.

[95] Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer Sentinel Mixture Mod els. arXiv preprint arXiv:1609.07843, 2016.

[96] Paulius Micikevicius, Dusan Stosic, Neil Burgess, Marius Cornea, et al. FP8 Formats for Deep Learning. arXiv preprint arXiv:2209.05433, 2022.

[97] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In EMNLP, 2018.

[98] Mohammad Mozafari, Samuel Kushnir, Maryam Mehri Dehnavi, and Amir Yazdanbakhsh. OPTIMA: Optimal One-Shot Pruning for LLMs via Quadratic Programming Reconstruction. arXiv preprint arXiv:2512.13886, 2025.

[99] Mohammad Mozafari, Sikan Li, Zhao Zhang, and Maryam Mehri Dehnavi. MKOR: Momentum Enabled Kronecker-Factor-Based Optimizer Using Rank-1 Updates. In NeurIPS, 2023.

[100] Mohammad Mozafari, Amir Yazdanbakhsh, and Maryam Mehri Dehnavi. SLiM: One-Shot Quantized Sparse Plus Low-Rank Approximation of LLMs. In ICML, 2025.

[101] Mohammad Mozafari, Amir Yazdanbakhsh, Zhao Zhang, and Maryam Mehri Dehnavi. SLoPe: Double-Pruned Sparse Plus Lazy Low-Rank Adapter Pretraining of LLMs. In ICLR, 2025.

[102] Baorun Mu, Saeed Soori, Bugra Can, Mert Gürbüzbalaban, et al. HyLo: A Hybrid Low-Rank Natural Gradient Descent Method. In Proceedings of the International Conference on High Performance Computing, Networking, Storage and Analysis, pages 1–16, 2022.

[103] Tan Nguyen, Vai Suliafu, Stanley Osher, Long Chen, et al. FMMformer: Eficient and Flexible Transformer via Decomposed Near-Field and Far-Field Attention. In NeurIPS, 2021.

[104] Mahdi Nikdan, Soroush Tabesh, and Dan Alistarh. RoSA: Accurate Parameter-Eficient Fine-Tuning via Robust Adaptation. In ICML, 2024.

[105] NVIDIA. NVIDIA A100 Tensor Core GPU Architecture, 2020. Version 1.0.

[106] NVIDIA. NVIDIA Hopper Architecture Whitepaper, 2022. Accessed via GTC 2022 presentation materials.

[107] NVIDIA, Péter Vingelmann, and Frank H.P. Fitzek. CUDA, Release: 10.2.89, 2020.

[108] NVIDIA Corporation. NVIDIA Ampere Architecture In-Depth. https://developer.nvidia. com/blog/nvidia-ampere-architecture-in-depth.

[109] NVIDIA Corporation. NVIDIA cuBLAS. https://docs.nvidia.com/cuda/cublas/.

[110] NVIDIA Corporation. NVIDIA cuSPARSELt. https://docs.nvidia.com/cuda/ cusparselt/index.html.

[111] NVIDIA Corporation. NVIDIA cuSPARSELt Functions. https://docs.nvidia.com/cuda/ cusparselt/functions.html.

[112] NVIDIA Corporation. NVIDIA Deep Learning Examples. https://github.com/NVIDIA/ DeepLearningExamples.

[113] Kazuki Osawa, Yohei Tsuji, Yuichiro Ueno, Akira Naruse, et al. Large-Scale Distributed Second-Order Optimization Using Kronecker-Factored Approximate Curvature for Deep Convolutional Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12359–12367, 2019.

[114] Eunhyeok Park, Sungjoo Yoo, and Peter Vajda. Value-Aware Quantization for Training and Inference of Neural Networks. In ECCV, 2018.

[115] David Patterson, Joseph Gonzalez, Quoc V. Le, Chen Liang, et al. Carbon Emissions and Large Neural Network Training, 2021.

[116] J Gregory Pauloski, Qi Huang, Lei Huang, Shivaram Venkataraman, et al. KAISA: An Adaptive Second-Order Optimizer Framework for Deep Neural Networks. In SC, 2021.

[117] J Gregory Pauloski, Zhao Zhang, Lei Huang, Weijia Xu, et al. Convolutional Neural Network Training with Distributed K-FAC. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–12. IEEE, 2020.

[118] Qwen, An Yang, Baosong Yang, et al. Qwen2.5 Technical Report, 2025.

[119] Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving Language Understanding by Generative Pre-Training. OpenAI, 2018.

[120] Alec Radford, Jefrey Wu, Rewon Child, David Luan, et al. Language Models Are Unsupervised Multitask Learners. OpenAI Blog, 1(8):9, 2019.

[121] Colin Rafel, Noam Shazeer, Adam Roberts, Katherine Lee, et al. Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. arXiv e-prints, 2019.

[122] Arya Rafii, Victor Kamel, and Maryam Mehri Dehnavi. Stoicc. https://paramathic.github. io/stoicc-docs/, 2025.

[123] Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv preprint arXiv:1606.05250, 2016.

[124] Yi Ren and Donald Goldfarb. Eficient Subsampled Gauss-Newton and Natural Gradient Methods for Training Neural Networks. arXiv preprint arXiv:1906.02353, 2019.

[125] Babak Rokh, Ali Azarpeyvand, and Alireza Khanteymoori. A Comprehensive Survey on Model Quantization for Deep Neural Networks in Image Classification. ACM Transactions on Intelligent Systems and Technology, 14(6):1–50, 2023.

[126] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. Communications of the ACM, 64(9):99–106, 2021.

[127] Victor Sanh, Thomas Wolf, and Alexander Rush. Movement Pruning: Adaptive Sparsity by Fine-Tuning. NeurIPS, 2020.

[128] Jürgen Schmidhuber. Deep Learning in Neural Networks: An Overview. Neural Networks, 61:85–117, 2015.

[129] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, et al. OmniQuant: Omnidirectionally Cali brated Quantization for Large Language Models. In ICLR, 2024.

[130] Noam Shazeer. GLU Variants Improve Transformer. arXiv preprint arXiv:2002.05202, 2020.

[131] Noam Shazeer and Mitchell Stern. Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. In ICML, 2018.

[132] Shaohuai Shi, Lin Zhang, and Bo Li. Accelerating Distributed K-FAC with Smart Parallelism of Computing and Communication Tasks. In 2021 IEEE 41st International Conference on Distributed Computing Systems (ICDCS), pages 550–560. IEEE, 2021.

[133] Seongjin Shin, Sang-Woo Lee, Hwijeen Ahn, Sungdong Kim, et al. On the Efect of Pretraining Corpora on In-Context Learning by a Large-Scale Language Model. arXiv preprint arXiv:2204.13509, 2022.

[134] Sidak Pal Singh and Dan Alistarh. WoodFisher: Eficient Second-Order Approximation for Neural Network Compression. NeurIPS, 2020.

[135] Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, et al. SlimPajama: A 627B Token Cleaned and Deduplicated Version of RedPajama. https://bit.ly/slimpajamas, 2023.

[136] Mingjie Sun, Zhuang Liu, Anna Bair, and J Zico Kolter. A Simple and Efective Pruning Approach for Large Language Models. In ICLR, 2024.

[137] Wei Sun, Aojun Zhou, Sander Stuijk, Rob Wijnhoven, et al. DominoSearch: Find Layer-Wise Fine-Grained N:M Sparse Schemes from Dense Neural Networks. In NeurIPS, 2021.

[138] Yi Tay, Mostafa Dehghani, Samira Abnar, Yikang Shen, et al. Long Range Arena: A Benchmark for Eficient Transformers. In ICLR, 2021.

[139] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, et al. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786, 2025.

[140] Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, et al. Gemma 2: Improving Open Language Models at a Practical Size. arXiv preprint arXiv:2408.00118, 2024.

[141] Texas Advanced Computing Center. Lonestar 6. https://tacc.utexas.edu/systems/ lonestar6/.

[142] Vithursan Thangarasa, Abhay Gupta, William Marshall, Tianda Li, et al. SPDF: Sparse Pre-Training and Dense Fine-Tuning for Large Language Models. arXiv preprint arXiv:2303.10464, 2023.

[143] Philippe Tillet, H. T. Kung, and David Cox. Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations. In MAPL 2019: Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages, 2019.

[144] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, et al. LLaMA 2: Open Foundation and Fine-Tuned Chat Models. arXiv preprint arXiv:2307.09288, 2023.

[145] Yuichiro Ueno, Kazuki Osawa, Yohei Tsuji, Akira Naruse, et al. Rich Information Is Afordable: A Systematic Performance Analysis of Second-Order Optimization Using K-FAC. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 2145– 2153, 2020.

[146] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, et al. Attention Is All You Need. Ad vances in Neural Information Processing Systems, 30, 2017.

[147] Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, et al. GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding. In ICLR, 2019.

[148] Shibo Wang and Pankaj Kanwar. BFloat16: The Secret to High Performance on Cloud TPUs. http: //bit.ly/3WEtCGm, 2019.

[149] Wenxuan Wang and Zhaopeng Tu. Rethinking the Value of Transformer Components, 2020.

[150] Wikipedia. Wikipedia Corpus. https://meta.wikimedia.org/wiki/Data\_dump\_ torrents#English\_Wikipedia.

[151] Lucas Wilkinson, Kazem Cheshmi, and Maryam Mehri Dehnavi. Register Tiling for Unstructured Sparsity in Neural Network Inference. PLDI, 2023.

[152] Samuel Williams, Andrew Waterman, and David Patterson. Roofline: An Insightful Visual Performance Model for Multicore Architectures. Communications ofthe ACM, 2009.

[153] Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, et al. Transformers: State-of-the-Art Natural Language Processing. In EMNLP (System Demonstrations), pages 38–45, 2020.

[154] Haojun Xia, Zhen Zheng, Yuchao Li, Donglin Zhuang, et al. Flash-LLM: Enabling Cost-Efective and Highly-Eficient Large Generative Model Inference with Unstructured Sparsity. arXiv preprint arXiv:2309.10285, 2023.

[155] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, et al. SmoothQuant: Accurate and Eficient Post Training Quantization for Large Language Models. In ICML, 2023.

[156] Peng Xu, Wenqi Shao, Mengzhao Chen, Shitao Tang, Kaipeng Zhang, Peng Gao, Fengwei An, Yu Qiao, and Ping Luo. BESA: Pruning Large Language Models with Blockwise Parameter-Eficient Sparsity Allocation. arXiv preprint arXiv:2402.16880, 2024.

[157] Minghan Yang, Dong Xu, Zaiwen Wen, Mengyun Chen, et al. Sketchy Empirical Natural Gradient Methods for Deep Learning. arXiv preprint arXiv:2006.05924, 2020.

[158] Lu Yin, You Wu, Zhenyu Zhang, Cheng-Yu Hsieh, et al. Outlier Weighed Layerwise Sparsity (OWL): A Missing Secret Sauce for Pruning LLMs to High Sparsity. In ICML, 2024.

[159] Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, et al. Large Batch Optimization for Deep Learning: Training BERT in 76 Minutes. In ICLR, 2020.

[160] Xiyu Yu, Tongliang Liu, Xinchao Wang, and Dacheng Tao. On Compressing Deep Models by Low Rank and Sparse Decomposition. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 7370–7379, 2017.

[161] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, et al. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

[162] Cheng Zhang, Jianyi Cheng, George A Constantinides, and Yiren Zhao. LQER: Low-Rank Quantiza tion Error Reconstruction for LLMs. arXiv preprint arXiv:2402.02446, 2024.

[163] Lin Zhang, Shaohuai Shi, and Bo Li. EVA: A General Vectorized Approximation Framework for Second-Order Optimization, 2023.

[164] Stephen Zhang and Vardan Papyan. OATS: Outlier-aware pruning through sparse and low rank decomposition. In The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[165] Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, et al. OPT: Open Pre-Trained Transformer Language Models. arXiv preprint arXiv:2205.01068, 2022.

[166] Yuxin Zhang, Yiting Luo, Mingbao Lin, Yunshan Zhong, et al. Bi-Directional Masks for Eficient N:M Sparse Training. arXiv preprint arXiv:2302.06058, 2023.

[167] Ningxin Zheng, Bin Lin, Quanlu Zhang, Lingxiao Ma, et al. SparTA: Deep-Learning Model Sparsity via Tensor-with-Sparsity-Attribute. In OSDI, 2022.

[168] Aojun Zhou, Yukun Ma, Junnan Zhu, Jianbo Liu, et al. Learning N:M Fine-Grained Structured Sparse Neural Networks from Scratch. arXiv preprint arXiv:2102.04010, 2021.

[169] Zihan Zhou, Xiaodong Li, John Wright, Emmanuel J. Candès, and Yi Ma. Stable Principal Component Pursuit. In IEEE International Symposium on Information Theory (ISIT), pages 1518–1522. IEEE, 2010.

[170] Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, et al. Aligning Books and Movies: Towards Story-Like Visual Explanations by Watching Movies and Reading Books. In Proceedings ofthe IEEE International Conference on Computer Vision, pages 19–27, 2015.

## Appendix A

# Supplementary Material for MKOR

In this chapter, we report the GLUE results achieved on each task in Appendix A.1. Next, we discuss the derivation of KFAC and SNGD approximation methods in Appendix A.2. We then discuss some of the features of MKOR and other optimizers and back them up with quantitative data in Appendix A.3 and Appendix A.4. We provide scalability results of MKOR in Appendix A.5 and provide more data to back up the low-rank features of the covariance matrices in Appendix A.6. In Appendix A.7, we analyze MKOR and other optimizers on the training tasks as non-convex optimizers, only concerning their performance on training tasks. We also describe the knee-point learning rate scheduler in Appendix A.8. We conclude this chapter by proving the lemmas used in the preceding sections.

## A.1 GLUE Results

We discussed the speedup achieved using MKOR on the GLUE dataset in Appendix 3.4. For completeness, Table A.1 shows the metrics achieved in each of the diferent GLUE tasks on BERT-Large-Uncased trained on diferent optimizers.

Table A.1: BERT-Large-Uncased Results on the GLUE classification tasks.
<table><tr><td rowspan=1 colspan=1>Optimizer</td><td rowspan=1 colspan=1>Iterati-ons</td><td rowspan=1 colspan=1>MNLI(acc)</td><td rowspan=1 colspan=1>QQP(F1)</td><td rowspan=1 colspan=1>QNLI(acc)</td><td rowspan=1 colspan=1>SST-2(acc)</td><td rowspan=1 colspan=1>COLA(mcc)</td><td rowspan=1 colspan=1>STS-B(corr)</td><td rowspan=1 colspan=1>MRPC(F1)</td><td rowspan=1 colspan=1>RTE(acc)</td><td rowspan=1 colspan=1>Avera-ge</td></tr><tr><td rowspan=1 colspan=1>LAMB</td><td rowspan=1 colspan=1>1,563</td><td rowspan=1 colspan=1>0.841</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.913</td><td rowspan=1 colspan=1>0.919</td><td rowspan=1 colspan=1>0.516</td><td rowspan=1 colspan=1>0.875</td><td rowspan=1 colspan=1>0.812</td><td rowspan=1 colspan=1>0.664</td><td rowspan=1 colspan=1>0.8023</td></tr><tr><td rowspan=1 colspan=1>KAISA</td><td rowspan=1 colspan=1>1,563</td><td rowspan=1 colspan=1>0.821</td><td rowspan=1 colspan=1>0.854</td><td rowspan=1 colspan=1>0.900</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.489</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.888</td><td rowspan=1 colspan=1>0.617</td><td rowspan=1 colspan=1>0.796</td></tr><tr><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>1,500</td><td rowspan=1 colspan=1>0.844</td><td rowspan=1 colspan=1>0.879</td><td rowspan=1 colspan=1>0.916</td><td rowspan=1 colspan=1>0.923</td><td rowspan=1 colspan=1>0.523</td><td rowspan=1 colspan=1>0.892</td><td rowspan=1 colspan=1>0.905</td><td rowspan=1 colspan=1>0.690</td><td rowspan=1 colspan=1>0.8214</td></tr><tr><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>0.833</td><td rowspan=1 colspan=1>0.878</td><td rowspan=1 colspan=1>0.904</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.494</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.893</td><td rowspan=1 colspan=1>0.653</td><td rowspan=1 colspan=1>0.8078</td></tr><tr><td rowspan=1 colspan=1>MKOR-H</td><td rowspan=1 colspan=1>600</td><td rowspan=1 colspan=1>0.838</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.911</td><td rowspan=1 colspan=1>0.921</td><td rowspan=1 colspan=1>0.502</td><td rowspan=1 colspan=1>0.886</td><td rowspan=1 colspan=1>0.898</td><td rowspan=1 colspan=1>0.657</td><td rowspan=1 colspan=1>0.811</td></tr><tr><td rowspan=1 colspan=1>Eva</td><td rowspan=1 colspan=1>1000</td><td rowspan=1 colspan=1>0.839</td><td rowspan=1 colspan=1>0.877</td><td rowspan=1 colspan=1>0.907</td><td rowspan=1 colspan=1>0.914</td><td rowspan=1 colspan=1>0.499</td><td rowspan=1 colspan=1>0.890</td><td rowspan=1 colspan=1>0.904</td><td rowspan=1 colspan=1>0.650</td><td rowspan=1 colspan=1>0.809</td></tr></table>

## A.2 Derivation of NGD Approximations

Natural Gradient Descent (NGD) In NGD, which is a second-order method, we use the inverse of the Fisher Information Matrix (FIM) as a preconditioner to the gradients as shown in Figure A.1-a. Equation 3.1 shows the update rule of NGD, where $F ^ { m }$ is the FIM block corresponding to block $m .$ . Equation A.1 shows the definition of FIM for an arbitrary layer in our model, where $x _ { i }$ is the $i ^ { t h }$ sample in the batch.

![](images/8de4934a0526603b2ddf39b4e9bbdb420e0f811a23af23eebeba679c1820a240.jpg)  
c. SNGD Based Approximatoin  
Figure A.1: Approximations in second-order methods.

$$
F ^ { m } = \frac { 1 } { b } \sum _ { i = 1 } ^ { b } \nabla _ { x _ { i } } \ell ( \mathcal { W } , x _ { i } ) \nabla _ { x _ { i } } \ell ( \mathcal { W } , x _ { i } ) ^ { T }\tag{A.1}
$$

Kronecker Factorization (KFAC). KFAC methods reformulate the FIM block as the Kronecker product of two matrices as shown in Equation A.2 where $\begin{array} { r } { g ^ { m } = \nabla _ { x ^ { m } } \mathcal { L } } \end{array}$ and $a ^ { m }$ is the vector form of the activation output of layer m and E is the expectation operator and $x ^ { m }$ is the input matrix of layer $m .$ . Please note that we have used the mixed-product property of Kronecker multiplication for getting the right hand value.

$$
F ^ { m } = \mathbb { E } [ ( g ^ { m } \otimes a ^ { m - 1 } ) ( g ^ { m } \otimes a ^ { m - 1 } ) ^ { T } ] = \mathbb { E } [ ( g ^ { m } g ^ { m T } ) \otimes ( a ^ { m - 1 } a ^ { m - 1 } ) ]\tag{A.2}
$$

Furthermore, we assume that $\begin{array} { r } { { \mathbb { E } } [ ( g ^ { m } g ^ { m T } ) \otimes ( a ^ { m - 1 } a ^ { m - 1 } ^ { T } ) ] \approx { \mathbb { E } } [ g ^ { m } g ^ { m T } ] \otimes { \mathbb { E } } [ a ^ { m - 1 } a ^ { m - 1 } ^ { T } ] } \end{array}$ , which is a strong assumption, but helps us simplify the computation further. Using the inversion property of Kronecker multiplication, we can compute the inverse of FIM using Equation A.3.

$$
F ^ { m - 1 } w ^ { m } = \mathbb { E } [ g ^ { m } g ^ { m T } ] ^ { - 1 } \otimes \mathbb { E } [ a ^ { m - 1 } a ^ { m - 1 ^ { T } } ] ^ { - 1 } w ^ { m }\tag{A.3}
$$

By using the mixed Kronecker matrix-vector product property, we can get the update value in Equation 3.2, which is illustrated in Figure A.1-b. We refer to $L ^ { m }$ and $R ^ { m }$ as the left and right factors respectively. Adding momentum to the left and right factors and denoting the iteration number with a subscript to the factors, we will get Equation 3.3 and Equation 3.4.

Sherman-Morrison-Woodbury-Based Natural Gradient Descent (SNGD). In this method, the SMW identity is used for approximating the inverse of $( F ^ { m } + \mu I ) \in \mathbb { R } ^ { d ^ { 2 } \times d ^ { 2 } }$ , where $\mu$ is a damping factor used in the preconditioning. Equation A.4 shows the process of computing the inverse of the FIM for a single layer in the network, where $A ^ { m } \in \mathbb { R } ^ { d \times b }$ is the batch of activations of layer l and $G ^ { m } \in \mathbb { R } ^ { d \times b }$ is the batch of gradients of the loss function with respect to the inputs of that layer and $\boldsymbol { U } = [ \nabla _ { W ^ { m } } \mathcal { L } ( \mathcal { W } , x _ { 1 } ) , . . . , \nabla _ { W ^ { m } } \mathcal { L } ( \mathcal { W } , x _ { b } ) ] ^ { T } \in \mathbb { R } ^ { d ^ { 2 } \times }$ b is the concatenation of the gradients of the loss function with respect to the parameters of that layer and $\odot$ shows the Hadamard element-wise product. In this method, a kernel matrix in $\mathbb { R } ^ { b \times b }$ is inverted, as shown in Figure A.1-c.

$$
( F ^ { m } + \mu I ) ^ { - 1 } = \frac { 1 } { \mu } ( I - U ^ { m } ( { A ^ { m - 1 } } ^ { T } A ^ { m - 1 } \odot G ^ { m } { } ^ { T } G ^ { m } + \mu I ) ^ { - 1 } U ^ { m } { } ^ { T } )\tag{A.4}
$$

## A.3 Numerical Instability of Second-order Methods

In Appendix 3.3.3, we discussed that in second-order methods, multiple matrix inversion or root-finding algorithms need to be executed, which make the second-order methods prone to numerical instabilities. Furthermore, we discussed that left and right factors in second-order methods have large condition numbers, resulting in further issues in inversion. Figure A.2 shows the eigenvalues of the right factor and its condition number for ResNet-50 model on CIFAR-10 dataset on KFAC algorithm. Even when using damping factors and filtering out extremely small eigenvalues, the condition number of these matrices is large, motivating the use of double precision computations for avoiding numerical instabilities.

MKOR, on the other hand, doesn’t sufer numerical instabilities when inverting such matrices, and its computational complexity isn’t dependent on the condition number either.

Table A.2: Number of epochs necessary for convergence in diferent optimizers for ResNet-50 on CIFAR10. MKOR is the least sensitive optimizer to learning rate, converging in almost the same number of iterations for a wide range of learning rate, while other optimizers either diverge (D) or converge to a local-minimum (∗ superscript).
<table><tr><td rowspan=1 colspan=1>Learning Rate $\mathrm { O p t i m i z e r } \frown$ </td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.1</td><td rowspan=1 colspan=1>0.01</td></tr><tr><td rowspan=1 colspan=1>MKOR</td><td rowspan=1 colspan=1>94</td><td rowspan=1 colspan=1>79</td><td rowspan=1 colspan=1>78</td><td rowspan=1 colspan=1>76</td></tr><tr><td rowspan=1 colspan=1>KAISA</td><td rowspan=1 colspan=1>112</td><td rowspan=1 colspan=1>100</td><td rowspan=1 colspan=1>90</td><td rowspan=1 colspan=1>89*</td></tr><tr><td rowspan=1 colspan=1>HyLo</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>123*</td><td rowspan=1 colspan=1>98</td><td rowspan=1 colspan=1>150*</td></tr><tr><td rowspan=1 colspan=1>SGD</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>108</td><td rowspan=1 colspan=1>145*</td></tr></table>

## A.4 Sensitivity to Learning Rate

Learning rate is one of the main hyperparameters in machine learning (ML) that can directly afect the convergence time of optimization, and ML practitioners have to spend a lot of time tuning this hyperparameter. More specifically, in first-order methods, a large learning rate can easily lead to divergence and numerical instability, and in second-order methods, large learning rates can lead to exploding gradients as discussed in Appendix 3.3.3. Using small learning rates can lead to slow convergence in both first- and second-order methods and can even jeopardize the main advantage of second-order methods, which is their faster convergence rate.

One of the main advantages of our method is its robustness against a wide range of learning rates. As Table A.2 shows, first-order methods are extremely sensitive to the learning rate, and the second-order methods are prone to ripples and divergent for a larger range of learning rates, and lose their performance for small learning rates. Our method, on the other hand, will converge with a high convergence rate for a wide range of learning rates, and by directly modifying the inverse of factors as discussed in Appendix 3.3.3 can find a proper equilibrium between first- and second-order methods. This table shows that our method is the least sensitive to the learning rate values and can make the job of ML practitioners for tuning this hyperparameter extremely easy.

## A.5 Scalability

Figure A.3 shows the strong scalability of MKOR on BERT-Large-Uncased on up to 64 GPUs.

## A.6 Decaying Eigenvalues and Rank-1 Approximations

Figure A.4 shows that the eigenvalues of the factors will decay as the model converges, making rank-1 approximations more efective. The reason behind the decay in the eigenvalues is that the weights are initialized randomly and the neurons work independently in the beginning of the training, but as the model converges, the neurons become more dependent on each other and thus the activation and input gradients will become linearly dependent. This is also reflected by some large error values in the distributions in Figure 3.6[a, b, c, d]. The factors in MKOR are initialized with identity, starting MKOR from a first-order method. As a result, MKOR is more robust against noise in the approximations in the first iterations (the approximation error does not noticeably afect the factors when replacing $L _ { t - 1 } ^ { m - 1 - 1 }$ and $R _ { t - 1 } ^ { m } { } ^ { - 1 }$ in Equation 3.5 and Equation 3.6 with identity). But as the model converges, the factors in MKOR will be mostly shaped by the training samples, making MKOR more reliable on less erroneous approximations, and the decaying eigenvalues of the factors help MKOR with that.

## A.7 Training Accuracy Experiments

To evaluate MKOR as an optimizer that tries to minimize a specific objective function, we have considered the case of only minimizing the loss function of models on diferent tasks and set all the other optimizer parameters such as weight decay to zero, since using a non-zero weight decay adds a quadratic term to the loss function and using diferent weight decays for diferent optimizers leads to optimizing diferent objective functions, which might be considered unfair.

Recent work has shown the advantage of second-order methods over their first-order counterparts on multiple CNN tasks [102, 116], such as residual networks [56]. In our training accuracy experiment, we use another CNN benchmark, AlexNet [72] with more than 20M parameters on CIFAR-100 [71] consisting of 50K training and 10K validation images of 100 classes. Figure A.5-c and Figure A.6-c show the convergence properties of diferent optimizers. MKOR is 1.26×, 1.31×, and 1.58× faster than HyLo-KIS, SGD, and KAISA respectively. The reason for the low convergence speed of KAISA is that we needed to use small learning rates for avoiding exploding gradients in it, which has damaged its convergence rate.

BERT is a large language model with two variants, BERT-Base with more than 108M parameters and BERT-Large with more than 335M parameters. As shown in Figure A.5-a and Figure A.6-a, we have finetuned BERT-Large on the IMDB dataset which is a text classification task with 25K training and 25K test samples. MKOR outperforms SGD and HyLo-KIS by a speedup factor of 1.22× and 1.43× respectively. We have also fine-tuned BERT-Base on SQuAD dataset, which is a question answering task with 87.6K training and 10.6K test samples. MKOR achieves 1.26× and 1.56× speedup over SGD and HyLo respectively. Using a wide range of learning rates, KAISA could not converge on any of our BERT experiments which are based on the HuggingFace implementation of BERT, and the reason for lack of convergence of KAISA is exploding gradients.

## A.8 Knee-Point Learning Rate Scheduler

<sup>1</sup> While using large learning rates is crucial for utilizing the higher convergence rate of second-order methods, it is necessary to reduce the learning rate after some iterations so that the model converges, and the number of iterations for changing the learning rate is not known in advance. In practice, machine learning practitioners will manually find the number of iterations by trial and error or use predefined functions [60, 85] that don’t necessarily work ideally in all optimization problems. We used knee-point learning rate scheduler in Appendix A.7.

To fully utilize the potentials of the optimizers, we have designed a learning rate scheduler that monitors the rate of improvement in accuracy or decrease in the loss function value, and based on that decides when to decrease the learning rate. The scheduler detects knee-points in the accuracy/loss and decreases the learning rate when a knee-point is observed.

By definition, knee-points are defined as the points where the average accuracy/loss rate is less than β times the increment/decrement in the accuracy/loss since using the current learning-rate. For averaging the accuracy/loss rate, we use an exponential moving average, and $\beta$ is a hyperparameter that we can choose to show how much the scheduler can tolerate lack of improvement to detect the accuracy/loss.

## A.9 Proofs

## Theorem 3.3.1:

Proof. Given a positive definite matrix $J _ { t - 1 }$ and a vector $j$ and scalar $0 < \gamma < 1$ , we show that Equation A.5 results in a positive-definite matrix.

$$
{ J _ { t } } ^ { - 1 } = \gamma J _ { t - 1 } { } ^ { - 1 } + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) j ^ { T } J _ { t - 1 } { } ^ { - 1 } j ) } { J _ { t - 1 } } ^ { - 1 } { j j ^ { T } J _ { t - 1 } } ^ { - 1 }\tag{A.5}
$$

Since $\gamma > 0$ and $J _ { t - 1 }$ is a positive-definite matrix, $\gamma J _ { t - 1 }$ is also positive-definite.

Also, since $J _ { t - 1 }$ is a positive-definite matrix, ∀x $\neq 0 : x ^ { T } J _ { t - 1 } x > 0$

$$
j ^ { T } L _ { t - 1 } ^ { - 1 } j > 0 \xrightarrow { 0 < \gamma < 1 } 1 + \gamma ( 1 - \gamma ) j ^ { T } J _ { t - 1 } ^ { - 1 } j > 0\tag{A.6}
$$

Now we show that ${ J _ { t - 1 } } ^ { - 1 } { j j ^ { T } } { J _ { t - 1 } } ^ { - 1 }$ is also positive-definite.

$$
\forall x \neq 0 : x ^ { T } J _ { t - 1 } { } ^ { - 1 } j j ^ { T } J _ { t - 1 } { } ^ { - 1 } x = ( j ^ { T } J _ { t - 1 } ^ { - 1 } x ) ^ { T } ( j ^ { T } J _ { t - 1 } ^ { - 1 } x ) = \Vert ( j ^ { T } J _ { t - 1 } ^ { - 1 } x ) \Vert ^ { 2 } > 0\tag{A.7}
$$

Since both matrices on the right-hand-side of Equation A.5 are positive-definite and the sum of two positive definite matrices is a positive-definite matrix, the left-hand-side of it will be also positive definite. □

## Theorem 3.3.3:

Proof. By defining $P = ( \zeta L ^ { - 1 } + ( 1 - \zeta ) I ) \otimes ( \zeta R ^ { - 1 } + ( 1 - \zeta ) I )$ , Equation A.8 will hold.

$$
\hat { \mathcal { L } } ( w _ { 0 } - \Delta w ) - \mathcal { L } ( w _ { 0 } ) = - \nabla \mathcal { L } ( w _ { 0 } ) ^ { T } P \nabla \mathcal { L } ( w _ { 0 } )\tag{A.8}
$$

Now, we will show that P is a positive-semi-definite matrix. By using the associativity property of Kronecker multiplication, we can represent $P$ as in Equation A.9. Please note that diferent identity matrices in Equation A.9 have diferent shapes.

$$
P = \zeta ^ { 2 } L ^ { - 1 } \otimes R ^ { - 1 } + \zeta ( 1 - \zeta ) L ^ { - 1 } \otimes I + \zeta ( 1 - \zeta ) I \otimes R ^ { - 1 } + I\tag{A.9}
$$

Since matrices L and R are positive-semi-definite, the Kronecker products $L ^ { - 1 } \otimes R ^ { - 1 } , L ^ { - 1 } \otimes I .$ , and $I \otimes R ^ { - 1 }$ are also positive-semi-definite. As a result, for any non-zero vector $x , x ^ { T } P x > 0$ . So, based on Equation $\begin{array} { r } { \mathrm { A } . 8 , - \nabla \mathcal { L } ( w _ { 0 } ) ^ { T } P \nabla \mathcal { L } ( w _ { 0 } ) } \end{array}$ . As a result, Equation A.10 holds.

$$
\hat { \mathcal { L } } ( w _ { 0 } - \Delta w ) < \mathcal { L } ( w _ { 0 } )\tag{A.10}
$$

Theorem 3.3.2

Proof. Considering the quantization of matrix $J$ and vector $j$ in Equation A.11 and assuming the maximum quantization error is ϵ and the maximum values in vector $j$ and matrix $J$ is $m _ { : }$ , we can consider one of the three possible cases:

1. Vector-Vector Dot Product: The resulting error is $O ( 2 \epsilon m ^ { 2 } d )$ because the error of each multiplication can at most be $2 m \epsilon + \epsilon ^ { 2 }$ and by adding the d multiplications, the maximum error can be $( 2 m + 1 ) d \epsilon$ .

2. Vector-Matrix Product: The resulting error is $O ( 2 \epsilon m ^ { 2 } d )$ because a vector-matrix product can be considered as multiple independent vector-vector dot products.

3. Vector-Vector Outer Product: The resulting error is $O ( 2 \epsilon m )$ because each element in the resulting matrix is computed using a multiplication with the maximum error of $2 \epsilon m ^ { 2 } + \epsilon ^ { 2 }$

$$
\gamma J + \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } ( 1 + \gamma ( 1 - \gamma ) j ^ { T } J j ) } J j j ^ { T } J ^ { T }\tag{A.11}
$$

The error imposed by quantizing $\gamma J$ is at most $\gamma \epsilon .$ . The quantization error in the denominator of the fraction in Equation A.11 is negligible in comparison to 1 and won’t change the growth order of the final error term. The quantization error of $J j$ is $O ( 2 m ^ { 2 } d \epsilon )$ as discussed earlier in the case of vector-matrix product. $J j j ^ { T } J ^ { T } = ( J j ) ( J j ) ^ { T }$ is a vector-vector outer product, resulting in $O ( 4 m ^ { 3 } d ^ { 2 } )$ error.

So the final error quantization error is $O ( ( \gamma + 4 \frac { ( 1 - \gamma ) } { \gamma ^ { 2 } } m ^ { 3 } d ^ { 2 } ) \epsilon )$

ResNet-50 - CIFAR-10 - Eigenvalues of Right Factor  
![](images/9678bc10e4dd808c4f6c6a1aed5ef588c1527fdb5bad168abd3e901c54609b0f.jpg)  
(a)

ResNet-50 - CIFAR-10 - Condition Number of Right Factor  
![](images/1ec047bfaa1569ebdd5ac22a5980c93a6343299e5a8305959bd4d24955f16c0a.jpg)  
(b)  
Figure A.2: Maximum and minimum eigenvalues (a) and the condition number (b) of the right factors in KFAC when training ResNet-50 on CIFAR-10. As illustrated, the minimum eigenvalues of the factors in KFAC approach zero, meaning that the factors are singular, and hence have large condition numbers, making numerical inversion of them complex and numerically unstable.

![](images/43753f56ebd6acd662ec2ed379303d3f43ee55b6f54f12562d192a81001e7440.jpg)  
Figure A.3: Scalability of MKOR.

![](images/11e0c68f4efd057d5b12746a104c45798ad632f43b7e7ba98839050b6b7bd822.jpg)  
Figure A.4: Average covariance rank-1 approximation error for ResNet-50 in diferent iterations

![](images/c9f81909b12e8421525d62212604720985f9e6a3ac1d4cc552d0a67c99791639.jpg)  
(a)

![](images/969ade398815d507c37c830d4769a691ebc8189c400311ef7697701b29c73d63.jpg)

(b)  
![](images/26da9de914f7bbad918f3cd36635548896650a74d39afe322d9a4b1989282cf8.jpg)  
(c)  
Figure A.5: Training time for distributed first- and second-order optimizers SGD, MKOR, KAISA, and HyLo on BERT-Large-Cased on IMDB (a), BERT-Base-Cased on SQuAD (b), and AlexNet on CIFAR-100 (c). In all the experiments, MKOR outperforms other optimizers in convergence speed.

![](images/82792040ea49e8b74d63c1cc3009b3953b3b8c7af922b893e3a2cc8f8595f6c7.jpg)  
(a)

![](images/08e22a9f35f07282aaf38311b675c62daead846a91573ba4f4f5ab051502326e.jpg)

(b)  
![](images/48d8ac9e20c195247f35eb0b117aad12fbe36b734f097252396f3120f3ef2e6a.jpg)  
(c)  
Figure A.6: Training accuracy vs. the number of epochs for distributed first- and second-order optimizers SGD, MKOR, KAISA, and HyLo on BERT-Large-Cased on IMDB (a), BERT-Base-Cased on SQuAD (b), and AlexNet on CIFAR-100 (c). In all the experiments, MKOR outperforms other optimizers in convergence rate.

## Appendix B

# Supplementary Material for SLOPE

This chapter provides supplementary material for the SLOPE chapter. We begin with a comparison against dynamic sparsity using SR-STE in Appendix B.1, followed by an analysis of the cuSPARSELt initialization overhead in Appendix B.2. We present BERT-Large-Uncased pretraining and downstream evaluation results in Appendix B.3, and discuss the performance overhead of the bidirectional mask in Appendix B.4. Appendix B.5 analyzes the sparsity ratio in the double-pruned backward pass, and Appendix B.6 examines sensitivity to the choice of pruning matrix. Implementation details are provided in Appendix B.7, task-specific GLUE results in Appendix B.8, and the integration with Flash Attention in Appendix B.9. We compare with dense models in Appendix B.10 and conclude with proofs of the lemmas used in the chapter.

## B.1 Comparison with Dynamic Sparsity: SR-STE

We pretrained GPT2-Small (Appendix 4.4.2) using the SR-STE method [168] and reported the perplexity results in Figure 4.2. SR-STE aims to mitigate the Sparse Architecture Divergence (SAD) by dynamically adjusting the sparsity mask throughout training. We have tested various decay factor hyperparameters to find the optimal optimization strategy for SR-STE.

To understand the performance gap between SR-STE and SLOPE (our method) for the same training budget, we analyzed the mask dynamics in SR-STE. We plotted the average number of mask elements changes during training compared to the final converged mask sparsity pattern. High mask change values indicate that training resources are spent on updating weights that ultimately get pruned and do not necessarily contribute to the final model accuracy.

Figure B.1 shows this average mask diference per iteration relative to the converged model. As training progresses, the mask diference decreases, demonstrating SR-STE’s convergence to a specific sparsity pattern. However, in SLOPE, where all resources are dedicated to optimizing weights under a static mask<sup>1</sup>, SR-STE’s dynamic approach leads to wasted computation (represented by the area under the curve in Figure B.1). Consequently, for the same training budget, SLOPE achieves a lower perplexity in comparison to SR-STE due to its static mask approach.

![](images/9a4c51298c4224c967d7939cd1cf14c4ab76a0e6372dbf53c53a49597aed9f3a.jpg)  
Figure B.1: Average mask diference between each iteration and the converged sparsity pattern in GPT2-Small pretraining using SR-STE. The highlighted area shows the ratio of the resources used for updating weights that are pruned and not used in the inference of the model.

## B.2 cuSPARSELt Initialization Overhead: Static vs. Dynamic Sparsity

This section analyzes the time breakdown of the cuSPARSELt SpMM pipeline, highlighting the significant overheads associated with dynamically changing sparsity masks. The cuSPARSELt SpMM operation consists of two main phases: (1) Setup and (2) Matrix Multiplication. The setup phase involves initializing matrix handles and compressing the 2:4 sparse matrix. This compression copies non-zero values into a contiguous memory layout and generates indices for those values. The matrix multiplication phase leverages this metadata to perform the sparse matrix-matrix multiplication

Figure B.2 shows the setup and multiplication time for square matrices using the cuSPARSELt SpMM backend. As evident from the figure, the setup overhead is significantly larger than the actual matrix multiplication time. For SLOPE, which employs static sparsity masks, the setup cost is incurred only once and becomes negligible compared to the numerous matrix multiplications performed during training and inference. However, for dynamic sparsity patterns, such as Fully Sparse Training [62], Bidirectional Masks [166], and other similar methods[63, 137, 89, 168], this setup overhead can be substantial, leading to reduced speedup (as observed in Appendix 4.4.1 for Fully Sparse Training) or slowdowns in some configurations (as discussed in Appendix B.4).<sup>2</sup>

## B.3 BERT-Large-Uncased: Pretraining and Downstream Evaluation

BERT-Large-Uncased pretraining consists of two phases, as illustrated in Figure B.3. Phase 1 comprises 7,038 iterations with a global batch size of 65,536 and a sequence length of 128. Phase 2 includes 1,563 iterations with a global batch size of 32,768 and a sequence length of 512.

cuSPARSELt SpMM Time Breakdown  
![](images/b15cc2f19ab8068bca11796e9ad0edc3201320b6cf81fa6d37c903e19450ed9d.jpg)  
Figure B.2: The setup and multiplication time for square matrices using the cuSPARSELt SpMM backend.

Table B.1: End-to-end slow-down of Bi-directional Mask [166] in comparison to the dense baseline.
<table><tr><td>MODEL</td><td>DATASET</td><td>SLOW-DOWN (×)</td></tr><tr><td>MOBILENET v2</td><td>CIFAR10</td><td>5.08</td></tr><tr><td>RESNET-32</td><td>CIFAR10</td><td>5.07</td></tr><tr><td>VGG19</td><td>CIFAR10</td><td>8.41</td></tr><tr><td>RESNET-18</td><td>IMAGENET</td><td>3.66</td></tr><tr><td>RESNET-50</td><td>IMAGENET</td><td>3.01</td></tr></table>

Figure B.3 shows the training loss for both phases under diferent sparsity settings. We observe that higher sparsity ratios generally lead to higher training loss in both phases. Interestingly, the loss/perplexity gap does not directly correlate with the observed accuracy drops in downstream tasks [19, 82, 133].

We evaluated the pretrained BERT-Large-Uncased models on the SQuAD v1.1 [123] and GLUE [147] benchmarks. SQuAD v1.1, a comprehensive question-answering dataset based on Wikipedia, is widely used for LLM training. We report the F1 score for SQuAD throughout the chapter. GLUE, a diverse benchmark for natural language understanding tasks, provides a single aggregated score across various challenges, facilitating model comparisons. The chapter presents the average metric score for GLUE, while task-specific metrics are detailed in Appendix B.8.

## B.4 Performance overhead of bidirectional mask

Table B.1 presents the runtime results of Bidirectional Masks [166], a state-of-the-art N:M sparsity method. Our analysis demonstrates that the mask search and associated overheads of this approach result in significant slowdowns compared to dense baselines. For these experiments, we utilized the repository provided in [166] and employed the same models used in their evaluation.

BERT-Large-Uncased Pretraining Phase 1  
![](images/02eb06cb579efcafc3c520af3fc32870235118593aca8a237e52b18dc3ad4188.jpg)

BERT-Large-Uncased Pretraining Phase 2  
![](images/ec745e7571941fe8eac5168e9858e739f02c04225d08fa4b43273967f0dbfa25.jpg)  
Figure B.3: Training loss of BERT-Large-Uncased on WikiCorpus dataset for phase 1 and 2.

## B.5 Sparsity ratio analysis of double-pruned backward pass

As described in Appendix 4.3.1, our proposed sparse pretraining approach involves pruning weights in both the forward and backward passes. During the backward pass, we apply both row-wise and column-wise pruning, which introduces additional zero values to the column-wise pruned weight matrices used in the forward pass. Theorem 4.3.1 demonstrates that the resulting sparsity ratio can be calculated using Equation 4.8. Figure B.4 visualizes the imposed sparsity ratios for various N:M sparsity patterns. As expected, smaller N/M ratios lead to lower imposed sparsity ratios. Moreover, in most cases, the imposed sparsity ratio is significantly smaller than the original matrix’s density ratio.

## B.6 Sensitivity to the choice of pruning matrix

In linear layers, three matrices are involved in the forward and backward passes: the input, the output gradient, and the weights. Pruning each of these matrices can have distinct efects on model performance.

To identify the optimal pruning strategy, we conducted an experiment where we pretrained GPT2-Small for 100,000 iterations (a quarter of the full pretraining) while systematically applying both static and dynamic pruning to each of the three matrices. Static pruning involves generating a random mask at initialization and applying it throughout training. Dynamic pruning, on the other hand, prunes matrices based on their magnitude at each iteration. For dynamic pruning, the dense matrix values are computed and stored, and then pruned at every step.

Imposed Sparsity in Row-wise and Column-wise Pruning  
![](images/d399f4dd835e983188d9998015ad488dabd53b2e64ff7b10bf379a7c9dc7616d.jpg)  
Figure B.4: The imposed sparsity ratio when pruning the weight matrices in the backward pass.

Figure B.5 presents the validation perplexity for these experiments. Notably, pruning the output gradient led to model divergence after a few iterations and is not shown in the figure.

Analysis. As shown in Figure B.5, static pruning consistently achieved lower perplexities. This behavior suggests that focusing computational resources on elements that remain active throughout training can lead to improved performance. Furthermore, pruning weights resulted in lower perplexities compared to pruning inputs, indicating that weights are generally a better target for pruning.

Intuition. Pruning weights is analogous to removing connections between neurons. Pruning activation tensors is similar to introducing a non-linear function (akin to max-pooling) before each linear layer. Pruning output gradients, however, lacks practical justification and introduces errors into the backward pass, leading to model divergence.

## B.7 Implementation details

This section details the implementation of the custom functions and CUDA kernels used in Algorithm 2 to facilitate eficient sparse training.

Initialization, sparse matrix setup, and SpMM kernels. Before utilizing the cuSPARSELt APIs, a crucial initialization phase ensures proper configuration of essential variables for our computational task. Follow ing initialization, we configure the sparse data formats tailored for sparse matrices. This involves initializing matrix descriptors, pruning the matrices, and compressing them into a more compact representation. cuS PARSELt employs an automated search to determine the optimal kernel for executing SpMM. While setting up these sparse data formats incurs a non-negligible computational cost, this overhead is mitigated by the repetitive nature of matrix multiplications during the training process.

![](images/a28d99c49b9ebd83300c0971a12bc5f93d957590bcb72ccd3b14f3ae01718374.jpg)  
Figure B.5: Validation perplexity on GPT2-Small pretraining for 100,000 iterations for diferent matrix prun ing settings. Pruning the output gradients leads to divergence within a few iterations and hence is not reported.

Prune and compress. The gradient of the loss function with respect to the weights requires pruning using the same mask as the weight matrix. Consequently, it contains 50% extra zero values in the dense format. To address this redundancy, we developed an optimized CUDA kernel, integrated into PyTorch, that masks the gradients accordingly, eliminating the storage of unnecessary data and reducing memory usage. The output of this operation is a new matrix in $\mathbb { R } ^ { d _ { o u t } \times \frac { d _ { i n } } { 2 } }$

Sparse matrix addition. The cuSPARSELt sparse data format does not natively support addition operations. However, for matrices A and B sharing the same sparsity patterns, we developed an optimized CUDA kernel seamlessly integrated into the PyTorch training workflow. This kernel eficiently computes linear combinations of the form $\beta A + \gamma B ,$ , where $\beta$ and γ are arbitrary user-defined constants. This functionality is particularly useful for adding sparse weights to gradients in optimizers that utilize weight decay.

Update Sparse Matrix. After the optimizer updates the weight tensor values based on its rules, we need to update the sparse matrix format to reflect these changes. We implemented an optimized CUDA kernel that copies the weight tensors from the PyTorch format into the cuSPARSELt data type, enabling eficient storage and manipulation of sparse weights.

Table B.2: GLUE results for each task in the experiments discussed in Appendix 4.4.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Phase</td><td rowspan="2">Rank</td><td rowspan="2">First 12 Blocks</td><td rowspan="2">Last 12 Blocks</td><td rowspan="2">CoLA (mcc)</td><td rowspan="2">SST-2 (acc)</td><td rowspan="2">MRPC (f1)</td><td rowspan="2">STS-B (corr)</td><td rowspan="2">QQP (f1)</td><td rowspan="2">RTE (acc)</td><td rowspan="2">MNLI (acc)</td><td rowspan="2">QNLI (acc)</td></tr><tr><td></td></tr><tr><td>Dense SLoPE</td><td>1,2</td><td>0</td><td>2:4</td><td>2:4</td><td>51.6</td><td>91.9</td><td>81.2</td><td>87.5</td><td>87.8</td><td>66.4</td><td>84.1</td><td>91.3</td></tr><tr><td>MLP Mixer Only</td><td>2</td><td>0</td><td>2:4</td><td>2:4</td><td>41.8</td><td>91.4</td><td>88.7</td><td>87.2</td><td>85.9</td><td>65</td><td>82.1</td><td>90.1</td></tr><tr><td>SLoPE MLP Mixer + Self-Attention</td><td>2</td><td>0</td><td>2:4</td><td>2:4</td><td>38.8</td><td>90.4</td><td>85.9</td><td>86.4</td><td>85.9</td><td>63.5</td><td>81.5</td><td>89.3</td></tr><tr><td>SLoPE with Non-Lazy Adapters</td><td>2</td><td>40</td><td>2:4</td><td>2:4</td><td>43.3</td><td>90.8</td><td>89</td><td>87</td><td>86</td><td>64.6</td><td>82.3</td><td>89.6</td></tr><tr><td>SLoPE with Non-Lazy Adapters</td><td>2</td><td>40</td><td>2:8</td><td>2:4</td><td>29</td><td>89.7</td><td>83.7</td><td>85.6</td><td>85.2</td><td>66.8</td><td>79.9</td><td>87.4</td></tr><tr><td>SLoPE with Non-Lazy Adapters</td><td>2</td><td>40</td><td>2:4</td><td>2:8</td><td>44.1</td><td>91.1</td><td>89.8</td><td>86.6</td><td>86.3</td><td>62.5</td><td>82.3</td><td>89.6</td></tr><tr><td>SLoPE</td><td>1,2</td><td>0</td><td>2:4</td><td>2:4</td><td>37.9</td><td>91.4</td><td>85.4</td><td>86.6</td><td>85.8</td><td>62.5</td><td>80.7</td><td>88.6</td></tr><tr><td>SLoPE</td><td>1,2</td><td>4</td><td>2:4</td><td>2:4</td><td>38.5</td><td>91.4</td><td>85.8</td><td>86.8</td><td>85.8</td><td>63.9</td><td>80.8</td><td>88.4</td></tr><tr><td>SLoPE</td><td>1,2</td><td>16</td><td>2:4</td><td>2:4</td><td>39.2</td><td>91.3</td><td>86.4</td><td>86.6</td><td>86</td><td>63.5</td><td>80.8</td><td>88.2</td></tr><tr><td>SLoPE</td><td>1,2</td><td>64</td><td>2:4</td><td>2:4</td><td>42.7</td><td>90.3</td><td>85.1</td><td>86.8</td><td>85.7</td><td>66.4</td><td>80.3</td><td>88.5</td></tr><tr><td>WANDA</td><td>N/A</td><td>0</td><td>2:4</td><td>2:4</td><td>43.0</td><td>91.4</td><td>88.3</td><td>86.9</td><td>86.1</td><td>63.5</td><td>81.9</td><td>89.6</td></tr><tr><td>WANDA</td><td>N/A</td><td>0</td><td>2:8</td><td>2:4</td><td>4.6</td><td>0.88</td><td>81.3</td><td>81</td><td>83.3</td><td>53.8</td><td>76.7</td><td>83.9</td></tr><tr><td>WANDA</td><td>N/A</td><td>0</td><td>2:4</td><td>2:8</td><td>42.1</td><td>91.7</td><td>84.4</td><td>87.2</td><td>85.6</td><td>63.5</td><td>81.5</td><td>81.9</td></tr></table>

Table B.3: Speedup of SLOPE and FlashAttention-2 (FA2) on OPT models.
<table><tr><td rowspan="2">MODEL SIZE</td><td colspan="3">TRAINING</td><td colspan="3">INFERENCE</td><td colspan="2">INFERENCE</td><td colspan="2">INFERENCE</td></tr><tr><td>FA2</td><td>SLoPE</td><td>SLoPE + FA2</td><td>FA2</td><td>SLoPE</td><td>SLoPE + FA2</td><td>FA2</td><td>SLoPE + FA2</td><td>FA2</td><td>SLoPE + FA2</td></tr><tr><td>66B</td><td>1.28</td><td>1.13</td><td>1.53</td><td>1.36</td><td>1.34</td><td>1.99</td><td>1.31</td><td>1.95</td><td>1.30</td><td>1.91</td></tr><tr><td>30B</td><td>1.36</td><td>1.14</td><td>1.66</td><td>1.46</td><td>1.32</td><td>2.24</td><td>1.28</td><td>2.24</td><td>1.27</td><td>2.20</td></tr><tr><td>13B</td><td>1.47</td><td>1.12</td><td>1.84</td><td>1.61</td><td>1.30</td><td>2.48</td><td>1.30</td><td>2.24</td><td>1.12</td><td>2.19</td></tr><tr><td>6.7B</td><td>1.60</td><td>1.08</td><td>1.94</td><td>1.71</td><td>1.21</td><td>2.50</td><td>1.13</td><td>2.50</td><td>1.12</td><td>2.45</td></tr><tr><td>2.6B</td><td>2.26</td><td>1.05</td><td>2.56</td><td>2.47</td><td>1.07</td><td>3.23</td><td>1.05</td><td>3.09</td><td>1.00</td><td>2.92</td></tr></table>

## B.8 Task-specific GLUE results

The GLUE benchmark [147] comprises eight distinct natural language understanding classification tasks. While Appendix 4.4 presented the average GLUE score as a measure of overall model performance, this section provides a more detailed analysis by presenting the complete task-specific results for each training setting in Table B.2.

## B.9 Integration with Flash Attention

To show the compatibility of SLOPE with other optimization methods, we integrate SLOPE with FlashAttention-2 [24] and show that these approaches are orthogonal in practice and can boost the performance of the model separately. Table B.3 summarizes the speedup achieved with and without SLOPE or FlashAttention-2. As it can be observed, each of these methods can improve the speed of the model both in training and inference, and adding them together will increase the speedup even further.

## B.10 Comparison with dense models

To compare the performance of sparse models with dense models of the same size, we have conducted an experiment with GPT2-Small, in which we have reduced the number of transformer blocks in the model to half of GPT2-Small. We call this new configuration GPT2-Half. Table B.4 and Table B.5 summarize the accuracy results for GPT2-Half on diferent zero-shot downstream tasks.

It can be observed that SLOPE outperforms GPT2-Half on average, while dynamic sparse training methods, such as SR-STE perform worse than it. Additionally, it is clear that adding low-rank adapters to the model improves the accuracy of all sparse pretraining methods.

Table B.4: Performance comparison across diferent GPT models, sparsity methods, and LoRA ranks on various tasks. E-SR-STE stands for Extended SR-STE.
<table><tr><td>Model</td><td>Method</td><td>LoRA (r)</td><td>MMLU</td><td>Arc Challenge</td><td>Open Book QA</td><td>Average</td></tr><tr><td>GPT2-Small</td><td>Dense</td><td>r = 0</td><td>22.9</td><td>20.7</td><td>16.2</td><td>19.94</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 0</td><td>23.0</td><td>19.3</td><td>16.0</td><td>19.43</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 0.05%</td><td>23.0</td><td>19.4</td><td>16.2</td><td>19.53</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 2.1%</td><td>23.0</td><td>19.3</td><td>16.4</td><td>19.57</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 0</td><td>24.1</td><td>18.3</td><td>12.6</td><td>18.33</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 0.05%</td><td>24.1</td><td>18.4</td><td>14.2</td><td>18.90</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 2.1%</td><td>24.2</td><td>18.3</td><td>14.2</td><td>18.90</td></tr><tr><td>GPT2-Half</td><td>Dense</td><td>r = 0</td><td>22.9</td><td>19.5</td><td>16.0</td><td>19.47</td></tr></table>

## B.11 Zero-shot GLUE results for GPT

We have tested the accuracy of the models on the zero-shot GLUE tasks in Language Model Evaluation Harness [41]. Table B.5 summarizes the achieved GLUE results by diferent models. It can be seen that SLOPE outperforms SR-STE and GPT2-Half on average. Additionally, SR-STE performs better than GPT2-Half in GLUE task.

Table B.5: Performance comparison of GPT models using diferent sparsity methods and LoRA ranks on GLUE tasks. E-SR-STE stands for Extended SR-STE.
<table><tr><td>Model</td><td>Method</td><td>LoRA (r)</td><td>CoLA</td><td>MNLI (m)</td><td>MNLI (mm)</td><td>MRPC</td><td>QNLI</td><td>QQP</td><td>RTE</td><td>SST2</td><td>Avg</td></tr><tr><td>GPT2-Small</td><td>Dense</td><td>r = 0</td><td>0</td><td>32.4</td><td>33.2</td><td>66.9</td><td>50.3</td><td>51.8</td><td>49.8</td><td>59.3</td><td>43.2</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 0</td><td>0</td><td>34.3</td><td>34.0</td><td>72.5</td><td>50.0</td><td>48.5</td><td>50.0</td><td>52.3</td><td>42.8</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 0.05%</td><td>0</td><td>34.3</td><td>34.1</td><td>72.6</td><td>49.8</td><td>48.8</td><td>50.9</td><td>52.3</td><td>42.9</td></tr><tr><td>GPT2-Small</td><td>SLoPE</td><td>r = 2.1%</td><td>0</td><td>34.3</td><td>34.0</td><td>71.6</td><td>50.0</td><td>49.0</td><td>52.0</td><td>52.6</td><td>43.1</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 0</td><td>0</td><td>33.6</td><td>33.9</td><td>57.1</td><td>50.7</td><td>50.4</td><td>55.2</td><td>54.7</td><td>42.5</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 0.05%</td><td>0</td><td>33.1</td><td>33.6</td><td>57.9</td><td>51.0</td><td>50.5</td><td>55.4</td><td>55.0</td><td>42.6</td></tr><tr><td>GPT2-Small</td><td>E-SR-STE</td><td>r = 2.1%</td><td>0</td><td>33.3</td><td>33.5</td><td>58.2</td><td>51.0</td><td>50.5</td><td>55.2</td><td>55.2</td><td>42.6</td></tr><tr><td>GPT2-Half</td><td>Dense</td><td>r = 0</td><td>0.0</td><td>33.9</td><td>33.8</td><td>53.6</td><td>51.1</td><td>47.7</td><td>56.7</td><td>50.6</td><td>41.1</td></tr></table>

## B.12 Extended SR-STE and FST implementation details

Before we proceed with the details of Extended SR-STE and FST, we clarify the notations used in this chapter and the FST paper [62] in Table B.6

Table B.6: Description of Key Terms
<table><tr><td rowspan=1 colspan=1>Term</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Sparse Pretraining</td><td rowspan=1 colspan=1>Common notation used in SLoPe and FST, indicating the use ofsparse weights during pretraining.</td></tr><tr><td rowspan=1 colspan=1>Dense Finetuning</td><td rowspan=1 colspan=1>Notation used in the FST paper, indicating an extended pretrainingphase.</td></tr><tr><td rowspan=1 colspan=1>Downstream Finetuning</td><td rowspan=1 colspan=1>Performance after pretraining concludes, used to finetune themodel for specific downstream tasks.</td></tr><tr><td rowspan=1 colspan=1>FST</td><td rowspan=1 colspan=1>Extended pretraining technique focused on dense finetuning.</td></tr><tr><td rowspan=1 colspan=1>Extended SR-STE</td><td rowspan=1 colspan=1>Variation of sparse pretraining extended with additional fine-tuning.</td></tr></table>

We compare SLOPE with FST exclusively for training speedups and memory savings. Comparing pretraining quality between SLOPE and FST is less meaningful because the final models produced by these methods difer significantly in the number of parameters. Specifically, FST produces a dense model after sparse pretraining and dense finetuning (99% sparse pretraining + 1% sparse + low-rank adaptation), while SLOPE produces a sparse model augmented with lightweight low-rank adapters. The number of parameters in the FST model is approximately 2× larger than in SLOPE, which makes a direct quality comparison imbalanced.

We compare SLoPe with Extended SR-STE in terms of model quality, focusing on understanding the dynamics between static and dynamic masking under an equal number of parameters. This allows for a fair, ”apple-to-apple” comparison between the methods (iso-params). We refer to this method as ”Extended SR-STE” because, while the original SR-STE approach was designed for use with SGD, the FST paper extended it to support other optimizers.

The FSTand Extended SR-STE code are available in Listing B.1 and Listing B.2 respectively.

```python
def forward(ctx, x, weight, weight_sparse, weight_sparse_T, bias):
2 ctx.save_for_backward(input, weight_sparse_T, bias)
ctx.shape = x.shape
x = x.view(-1, x.shape[-1])
output = torch.mm(x, weight_sparse.t())
if bias is None:
return output.view(*ctx.shape[:-1], -1)
8 else:
9 return output.view(*ctx.shape[:-1], -1) + bias
10
11 def backward(ctx, grad_output):
12 grad_output = grad_output.half()
13 x, weight_T, bias = ctx.saved_tensors
14 grad_input = grad_weight = grad_bias = None
15 if ctx.needs_input_grad[0]:
16 if grad_output.stride() == (0, 0, 0):
17 grad_output = torch.ones_like(grad_output, device=grad_output.device
, dtype=grad_output.dtype)
```

```python
18 grad_output = grad_output.view(-1, grad_output.shape[-1])
19 grad_input = torch.mm(grad_output, weight_T.t()).view(
20 ctx.shape)
21 if ctx.needs_input_grad[1]:
22 x = x.view(-1, input.shape[-1])
23 grad_output = grad_output.view(-1, grad_output.shape[-1])
24 grad_weight = torch.mm(to_sparse_semi_structured(grad_output.t(), MVUE24
=True), x)
25 if ctx.needs_input_grad[2]:
26 grad_bias = grad_output.sum(0)
27 return grad_input, grad_weight, None, None, grad_bias
```

Listing B.1: FST Algorithm

```python
def forward(ctx, input, weight, mask, weight_factor):
sparse_weight = weight.clone().detach()
sparse_weight[mask] = 0.
ctx.save_for_backward(input, sparse_weight, weight_factor * mask * weight)
output = torch.matmul(input, sparse_weight.t())
output = output.clone()
8 return output
9
10 @staticmethod
11 def backward(ctx, grad_output):
12 input, weight, weight_addition_term = ctx.saved_tensors
13 input_shape = input.shape
14 if input.dim() == 3:
15 new_batch_size = input_shape[0] * input_shape[1]
16 input = input.reshape(new_batch_size, -1)
17 grad_output = grad_output.reshape(new_batch_size, -1)
18 grad_output, grad_output_mask = prune_column_wise(grad_output)
19 grad_weight = torch.matmul(grad_output.t(), input)
20 grad_weight += weight_addition_term
21
22 grad_input = torch.matmul(grad_output, weight)
23 grad_input = grad_input.reshape(input_shape)
24 return grad_input, grad_weight, None
```  
Listing B.2: Extended SR-STE Algorithm. The weights are stored as dense and are pruned on-the-fly.

## B.13 Comparison of Depth and Width Pruning

Depth pruning refers to reducing the number of layers in a model, while width pruning means reducing the size of the weights inside each layer in the model. We have conducted an experiment with depth and width pruning on LLaMA-2-7B [144] and Gemma-2-2B and Gemma-2-9B [140] to compare the efects of depth and width pruning on the performance of the models. The configurations used for the models are summarized in Table B.7, Table B.9, Table B.8. Similar to [62], we reduced the aspect ratio of the Up-Sample and Down-Sample modules to half. Please note that this mechanism gives an advantage to width pruning methods, as the number of parameters in the Self-Attention modules remain intact.

Table B.7: Model Configurations for LLaMA-2 7B
<table><tr><td>Pruning Method</td><td>Attributes</td></tr><tr><td>Baseline</td><td>base_emb_dim: 4096 base_num_query_heads: 32 base_num_kv_heads: 32 base_mlp_dim: 11008 base_num_decoder_layers: 32 head_dim: 128</td></tr><tr><td>Depth Pruning</td><td>base_emb_dim: 4096 base_num_query_heads: 32 base_num_kv_heads: 32 base_mlp_dim: 11008 base_num_decoder_layers: 16 # half the number of layers head_dim: 128</td></tr><tr><td>Width Pruning</td><td>base_emb_dim: 4096 base_num_query_heads: 32 base_num_kv_heads: 32 base_mlp_dim: 5504 # half the number of dimensions base_num_decoder_layers: 32 head_dim: 128</td></tr></table>

Preliminary retraining loss curves, as shown in Figure B.6 suggest no significant diference between depth pruning and width-pruning during pretraining. Interestingly, in some cases, depth-pruning appears to outperform width-pruning.

## B.14 Proofs

## Theorem 4.3.1

Proof. Considering a matrix with N : M column-wise pruned sparsity pattern, we want to prune the matrix using N : M sparsity pattern row-wise as well. Let’s define random variable X as the number of added nonzeros to M row-wise consecutive elements and $Y$ as the number of non-zeros in M row-wise consecutive elements.

$$
E [ X ] = \sum _ { i = 1 } ^ { M - N } P r [ X = i ] i\tag{B.1}
$$

Replacing $P r [ X = i ] = P r [ Y = N + i ]$ in Equation B.1, we will get Equation B.2, where we used a change in dummy variable $j = N + i$

$$
E [ X ] = \sum _ { i = 1 } ^ { M - N } P r [ Y = N + i ] i = \sum _ { j = N + 1 } ^ { M } P r [ Y = j ] ( j - N )\tag{B.2}
$$

Considering the definition of $Y$ , it can be inferred that random variable Y has binomial distribution with a success probability of $\textstyle { \frac { N } { M } }$ . As a result Equation B.3 shows the probability mass distribution of Y.

Table B.8: Model Configurations for Gemma-9B
<table><tr><td>Pruning Method</td><td>Attributes</td></tr><tr><td>Baseline</td><td>base_emb_dim: 3584 base_num_query_heads: 16 base_num_kv_heads: 8 base_mlp_dim: 14336 base_num_decoder_layers: 20 # merged local and global attention head_dim: 256</td></tr><tr><td>Depth Pruning</td><td>base_emb_dim: 3584 base_num_query_heads: 16 base_num_kv_heads: 8 base_mlp_dim: 14336 base_num_decoder_layers: 10 # half the merged layers head_dim: 256</td></tr><tr><td>Width Pruning</td><td>base_emb_dim: 3584 base_num_query_heads: 16 base_num_kv_heads: 8 base_mlp_dim: 7168 # half the number of dimensions base_num_decoder_layers: 20 # merged local and global attention head_dim: 256</td></tr></table>

$$
P r [ Y = j ] = { \binom { M } { j } } s ^ { j } ( 1 - s ) ^ { M - j } ; s \triangleq \frac { N } { M }\tag{B.3}
$$

By replacing Equation B.3 in Equation B.2, we will get Equation B.4.

$$
E [ X ] = \sum _ { j = N + 1 } ^ { M } { \binom { M } { j } } s ^ { j } ( 1 - s ) ^ { M - j } ( j - N )\tag{B.4}
$$

Let’s define random variable $Z$ as the added sparsity ratio to the matrix by the extra pruning. Since X was the number of added non-zeros in M consecutive elements, $\begin{array} { r } { E [ Z ] = \frac { 1 } { M } E [ X ] } \end{array}$ , and hence:

$$
E [ Z ] = D ( A ^ { R } ) - D ( A ^ { R , C } ) = \sum _ { j = N + 1 } ^ { M } { \binom { M } { j } } s ^ { j } ( 1 - s ) ^ { M - j } { \frac { j - N } { M } }\tag{B.5}
$$

## Theorem 4.3.2

Proof. In an optimization problem, we are aiming to find the optimal solution to Equation B.6.

$$
\operatorname* { m i n } _ { W _ { i } } E _ { X } [ { \mathcal { L } } ( X , W _ { i } ) ]\tag{B.6}
$$

When using backpropagation, which is based on the chain rule in derivation, we compute the gradient in Equation B.7.

$$
E _ { X } [ \nabla _ { X _ { i } } \mathcal { L } ( X , W _ { i } ) ] = E _ { X } [ \nabla _ { Y _ { i } } \mathcal { L } W ]\tag{B.7}
$$

Let’s define random variable M as a uniformly random mask of 0’s and 1’s. The mask will be 1 at each point with a probability of $\textstyle { \frac { N } { M } }$ . Let’s define $O \triangleq E [ M ]$ . O is a matrix of all ${ \frac { N } { M } } \mathbf { \bar { s } } .$ As a result $\begin{array} { r } { { \cal { O } } \odot { \cal { W } } = \frac { { \cal { N } } } { { \cal { M } } } { \cal { W } } } \end{array}$

Table B.9: Model Configurations for Gemma-2B
<table><tr><td>Pruning Method</td><td>Attributes</td></tr><tr><td>Baseline</td><td>base_emb_dim: 2304 base_num_query_heads: 8 base_num_kv_heads: 4 base_mlp_dim: 9216 base_num_decoder_layers: 12 # merged local and global attention head_dim: 256</td></tr><tr><td>Depth Pruning</td><td>base_emb_dim: 2304 base_num_query_heads: 8 base_num_kv_heads: 4 base_mlp_dim: 9216 base_num_decoder_layers: 6 # half the merged layers head_dim: 256</td></tr><tr><td>Width Pruning</td><td>base_emb_dim: 2304 base_num_query_heads: 8 base_num_kv_heads: 4 base_mlp_dim: 4608 # half the number of dimensions base_num_decoder_layers: 12 # merged local and global attention head_dim: 256</td></tr></table>

$$
E _ { X } [ \nabla _ { Y _ { i } } \mathcal { L } W ] = E _ { X } [ \nabla _ { Y _ { i } } \mathcal { L } ( \frac { M } { N } O \odot W ) ] = E _ { X } [ \nabla _ { Y _ { i } } \mathcal { L } ( \frac { M } { N } E _ { M } M \odot W ) ]\tag{B.8}
$$

By using the linearity of derivation and expectation operators, we can get the result in Equation B.9, which proves the theorem.

$$
E _ { X } [ \nabla _ { X _ { i } } \mathcal { L } ( X , W _ { i } ) ] = \frac { M } { N } E _ { M } [ E _ { X } [ \nabla _ { Y _ { i } } \mathcal { L } ( M \odot W ) ] ]\tag{B.9}
$$

![](images/3aab264f3e253a9a3102137b1f2b6e96c42a66f86387bd8e622e86b6e5fc261a.jpg)

![](images/00fc7d794441940e5523e921bf6f9d9fc2f1a2b6095314923a2625cd282c946a.jpg)

![](images/7dfff89d46c2989dafcf0676847fd5add42b3ebd0beda94a826f7ab9da4297bd.jpg)  
Figure B.6: Comparison of the loss of depth and width pruning methods.

## Appendix C

## Supplementary Material for OPTIMA

This chapter provides supplementary material for the OPTIMA chapter. Appendix C.1 evaluates the sensitivity of OPTIMA to the size of the calibration dataset.

## C.1 Calibration dataset size sensitivity

Similar to previous work (SparseGPT, Wanda, Thanos), OPTIMA leverages a set of calibration data from the C4 dataset to prune the models. Figure C.1 shows the perplexity of LLaMA-3.2-1B on WikiText2 dataset when pruning the models with various numbers of calibration samples. Our results indicate that unlike the other methods (Wanda and SparseGPT) that have stochastic behavior as the number of samples increases, OPTIMA shows consistent improvement in model quality. However, the improvements are not significant, suggesting robustness to dataset size.

Calibration Data Sensitivity Analysis  
![](images/fdbec668b0aa93ee31b3ced553d4f43ca8ce7ed4a63017f2b0132ee6cc5e65b3.jpg)  
Figure C.1: Sensitivity analysis for the number of calibration samples for diferent pruning methods.

# Appendix D

# Supplementary Material for PATCH

This chapter provides supplementary material for the PATCH chapter. Appendix D.1 describes the integration with STOICC. Appendix D.2 reports per-task accuracy results, and Appendix D.3 explores tile transfer learning across diferent models.

## D.1 STOICC Integration

Triton [143] enables developers to write eficient GPU kernels with a Python-like syntax, but it natively supports only dense matrix operations and cannot handle sparsity. To accelerate the mixed-tile format produced by PATCH, we employ the STOICC compiler [122]. STOICC extends Triton with a sparse code-generation backend that allows tiles within a matrix to be either dense or sparse, enabling mixed execution within a single matrix multiplication.

We rely on STOICC’s inspector to autotune both tile sizes and execution schedules (i.e., alternative kernel execution schemes such as split-K parallelism) for the prefill and decoding stages of LLM inference. Matrix compression and metadata generation are determined by the chosen tile size, which must remain consistent across both stages. To address this, we first autotune the decoding stage, which is the primary bottleneck of autoregressive generation, since it is executed once per generated token (e.g., 128 times for 128 new tokens), unlike the single pass of prefill. The optimal tile size identified for decoding is then fixed and reused for prefill, where we perform a second round of autotuning over the remaining independent parameters.

In contrast, for fully 2:4 sparse matrices, compression is independent of the block size, so they can be autotuned in the same way as dense kernels in Triton without this coupling constraint.

The pseudocode outlining this process, including the handling of dense, fully 2:4 sparse, and mixedsparsity modules, is provided in Listing D.1.

1 def tune\_and\_convert\_model(M, backend\_name):   
2 // backend\_name {"STOICC", "cuSPARSELt"}   
3 2\_4\_backend = select\_2\_4\_backend(backend\_name)   
4   
5 // create all configs & schedules to tune over   
6 base\_configs = STOICC.create\_configs()   
7 inspector = Inspector()   
8   
9 for each module in M:

```asm
10 s = get_sparsity_ratio(module.weight)
11
12 // Keep dense Torch (cuBLAS) module
13 if s == 0:
14 continue
15
16 // Use STOICC or cuSPARSELt for fully 2:4
17 elif s == 0.5:
18 c = 2_4_backend.compress(module.weight)
19 new_module = 2_4_backend.create_module(c)
20 replace(module, new_module)
21 continue
22
23 else:
24 decoding_input = Tensor(BS, module.weight.shape[1])
25 prefill_input = Tensor(BS * SL, module.weight.shape[1])
26
27 // Tune on decoding input first
28 inspector.set_configs(base_configs)
29 best_cfg_dec = inspector.inspect(
30 decoding_input,
31 module.weight,
32 isASparse=False)
33 BN = best_cfg_dec["BLOCK_N"]
34 BK = best_cfg_dec["BLOCK_K"]
35
36 // Tune on prefill using decoding tile sizes
37 prefill_cfg = STOICC.create_configs(BLOCK_N=BN, BLOCK_K=BK)
38 inspector.set_configs(prefill_cfg)
39 best_cfg_pre = inspector.inspect(
40 prefill_input,
41 module.weight,
42 isASparse=False)
43
44 c = inspector.compress(module.weight, BN, BK)
45 mixed_module = MixedModule(c, best_cfg_dec, best_cfg_pre)
46 replace(module, mixed_module)
47
48 return M
```  
PseudoCode D.1: Tuning and Converting Model Weights to Mixed Format.

Table D.1 reports the measured throughput (tokens processed per second) of LLaMA-2 7B at sparsity levels of 45%, 35%, and 25% with a batch size of 16 on an A6000 GPU. To reduce CPU overhead from launching Triton kernels in PyTorch, we executed generation through CUDA graphs, capturing both the prefill and decoding stages. With sparsity ratios between 25% and 45%, our heterogeneous approach achieves 1.18×– 1.38× end-to-end acceleration over the dense baseline. We also report timings on A100 in Table D.2.

Table D.1: Throughput of LLaMA-2 7B with mixed sparsity compared to the dense model. Measurements taken on an A6000 GPU with batch size 16. Throughput is reported in tokens processed/sec.
<table><tr><td>Sparsity</td><td>Prefill length</td><td>Tokens generated</td><td>Throughput (tok/s)</td><td>Speedup vs. dense</td></tr><tr><td>0%</td><td>128</td><td>128</td><td>1023.80</td><td>1.00×</td></tr><tr><td>25%</td><td>128</td><td>128</td><td>1212.79</td><td>1.18×</td></tr><tr><td>35%</td><td>128</td><td>128</td><td>1304.46</td><td>1.27×</td></tr><tr><td>45%</td><td>128</td><td>128</td><td>1410.20</td><td>1.38×</td></tr><tr><td>0%</td><td>128</td><td>1024</td><td>435.42</td><td>1.00×</td></tr><tr><td>25%</td><td>128</td><td>1024</td><td>493.33</td><td>1.13×</td></tr><tr><td>35%</td><td>128</td><td>1024</td><td>515.39</td><td>1.18×</td></tr><tr><td>45%</td><td>128</td><td>1024</td><td>542.87</td><td>1.25×</td></tr></table>

Table D.2: Throughput of LLaMA-2 7B with mixed sparsity compared to the dense model. Measurements taken on an A100 GPU with batch size 16. Throughput is reported in tokens processed/sec.
<table><tr><td>Sparsity</td><td>Prefill length</td><td>Tokens generated</td><td>Throughput (tok/s)</td><td>Speedup vs. dense</td></tr><tr><td>0%</td><td>128</td><td>128</td><td>1876.24</td><td>1.00×</td></tr><tr><td>25%</td><td>128</td><td>128</td><td>2002.02</td><td>1.07×</td></tr><tr><td>35%</td><td>128</td><td>128</td><td>2088.98</td><td>1.11×</td></tr><tr><td>45%</td><td>128</td><td>128</td><td>2180.88</td><td>1.16×</td></tr><tr><td>0%</td><td>128</td><td>1024</td><td>812.55</td><td>1.00×</td></tr><tr><td>25%</td><td>128</td><td>1024</td><td>864.66</td><td>1.06×</td></tr><tr><td>35%</td><td>128</td><td>1024</td><td>885.90</td><td>1.09×</td></tr><tr><td>45%</td><td>128</td><td>1024</td><td>907.12</td><td>1.12×</td></tr></table>

## D.2 Per Task Results

This appendix provides detailed per-task accuracy results for the models evaluated in Appendix 6.7, covering eight zero-shot downstream tasks: MMLU, PIQA, ARC-Easy, ARC-Challenge, Winogrande, OpenBookQA, RACE, and HellaSwag. The results are presented for each model at various sparsity levels and pruning methods, including our proposed PATCH<sup>Joint</sup> and PATCH<sup>Tile</sup> variants, alongside baseline methods such as Magnitude, Wanda, SparseGPT, Thanos, ProxSparse, and MaskLLM. These tables complement the average accuracy and perplexity results reported in Table 6.2 and Table 6.3 of this chapter, ofering a granular view of model performance across individual tasks.

For smaller models (Qwen-2.5 0.5B, LLaMA-3.2 1B, and Gemma-3 1B), we report results using the PATCH<sup>Joint</sup> variant, which jointly optimizes dense tile locations and sparsity patterns within sparse tiles. For larger models (LLaMA-2 7B and LLaMA-3.1 8B), we report results using the memory-eficient PATCH<sup>Tile</sup> variant, which optimizes dense tile selections with a fixed 2:4 sparsity mask. The per-task accuracies highlight the efectiveness of our approaches in maintaining robust performance across diverse tasks, even at high sparsity levels, compared to baseline methods.

The following tables detail the per-task accuracies for each model:

• Qwen-2.5 0.5B: Table D.3 presents the per-task accuracies for the PATCH<sup>Joint</sup> variant and baselines at 0% and 50% sparsity, with PATCH<sup>Joint</sup> evaluated at 25%, 35%, and 45% sparsity.

• LLaMA-2 7B: Table D.4 shows the per-task accuracies for the PATCH<sup>Tile</sup> variant and baselines, with PATCH<sup>Tile</sup> evaluated at 25%, 35%, and 45% sparsity.

• LLaMA-3.1 8B: Table D.5 provides the per-task accuracies for the PATCH<sup>Tile</sup> variant and baselines, with PATCH<sup>Tile</sup> at 25%, 35%, and 45% sparsity.

• LLaMA-3.2 1B: Table D.6 reports the per-task accuracies for the PATCH<sup>Joint</sup> variant and baselines, with PATCH<sup>Joint</sup> at 25%, 35%, and 45% sparsity.

• Gemma-3 1B: Table D.7 details the per-task accuracies for the PATCH<sup>Joint</sup> variant and baselines, with PATCH<sup>Joint</sup> at 25%, 35%, and 45% sparsity.

These results enable a deeper analysis of the task-specific performance trends, demonstrating the flexibility and robustness of PATCH<sup>Joint</sup> and PATCH<sup>Tile</sup> in achieving high accuracy across diverse tasks while maintaining hardware-friendly sparsity patterns.

Table D.3: Model quality (task accuracy across eight zero-shot tasks, reported in %) for Qwen-2.5 0.5B with diferent pruning methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity patterns, enabling a flexible sparsity-quality tradeof.
<table><tr><td>Sparsity</td><td>Method</td><td>Pattern</td><td>MMLU</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>WinoG.</td><td>OBQA</td><td>RACE</td><td>HellaS.</td><td>Avg</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>47.71</td><td>70.24</td><td>64.48</td><td>29.52</td><td>56.20</td><td>24.20</td><td>35.02</td><td>40.63</td><td>46.00</td></tr><tr><td>50%</td><td>Magnitude</td><td>2:4</td><td>23.00</td><td>54.24</td><td>31.23</td><td>19.20</td><td>49.96</td><td>13.60</td><td>23.44</td><td>26.59</td><td>30.16</td></tr><tr><td rowspan="5"></td><td>Wanda</td><td>2:4</td><td>24.43</td><td>58.71</td><td>43.18</td><td>17.75</td><td>51.62</td><td>12.20</td><td>26.32</td><td>29.58</td><td>32.97</td></tr><tr><td>SparseGPT 2:4</td><td></td><td>22.93</td><td>60.77</td><td>46.60</td><td>20.82</td><td>52.88</td><td>14.00</td><td>29.57</td><td>30.93</td><td>34.81</td></tr><tr><td>Thanos</td><td>2:4</td><td>22.97</td><td>60.17</td><td>45.37</td><td>19.20</td><td>53.59</td><td>15.20</td><td>31.00</td><td>31.31</td><td>34.85</td></tr><tr><td>ProxSparse 2:4</td><td></td><td>23.00</td><td>57.34</td><td>40.53</td><td>18.26</td><td>48.62</td><td>14.00</td><td>25.65</td><td>29.02</td><td>32.05</td></tr><tr><td>MaskLLM</td><td>2:4</td><td>25.11</td><td>67.03</td><td>56.57</td><td>23.98</td><td>52.57</td><td>20.20</td><td>33.30</td><td>35.90</td><td>39.33</td></tr><tr><td>45%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>27.39</td><td>68.44</td><td>59.13</td><td>25.77</td><td>53.67</td><td>19.80</td><td>32.15</td><td>35.99</td><td>40.29</td></tr><tr><td>35%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>29.04</td><td>68.88</td><td>60.40</td><td>26.37</td><td>55.09</td><td>20.40</td><td>32.44</td><td>36.58</td><td>41.15</td></tr><tr><td>25%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>30.89</td><td>69.15</td><td>62.79</td><td>29.10</td><td>55.33</td><td>20.00</td><td>34.16</td><td>37.71</td><td>42.39</td></tr></table>

Table D.4: Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-2 7B with diferent pruning methods. PATCH<sup>Tile</sup> optimizes tile-based sparsity, enabling a flexible sparsity-quality tradeof.
<table><tr><td>Sparsity Method</td><td></td><td>Pattern</td><td>MMLU</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>WinoG.</td><td>OBQA</td><td>RACE</td><td>HellaS.</td><td>Avg</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>41.82</td><td>78.07</td><td>76.35</td><td>43.52</td><td>69.06</td><td>31.40</td><td>39.52</td><td>57.13</td><td>54.61</td></tr><tr><td>50%</td><td>Magnitude</td><td>2:4</td><td>25.82</td><td>70.02</td><td>61.78</td><td>30.12</td><td>61.01</td><td>21.80</td><td>31.48</td><td>45.45</td><td>43.44</td></tr><tr><td rowspan="5"></td><td>Wanda</td><td>2:4</td><td>25.80</td><td>71.00</td><td>63.80</td><td>30.29</td><td>61.09</td><td>25.20</td><td>35.50</td><td>41.75</td><td>44.30</td></tr><tr><td>SparseGPT 2:4</td><td></td><td>26.17</td><td>70.73</td><td>63.80</td><td>30.63</td><td>65.04</td><td>24.00</td><td>37.13</td><td>43.18</td><td>45.09</td></tr><tr><td>Thanos</td><td>2:4</td><td>25.27</td><td>70.78</td><td>63.43</td><td>30.97</td><td>64.56</td><td>23.80</td><td>36.46</td><td>43.11</td><td>44.80</td></tr><tr><td>ProxSparse 2:4</td><td></td><td>26.77</td><td>71.60</td><td>65.70</td><td>33.02</td><td>62.90</td><td>24.20</td><td>35.31</td><td>47.84</td><td>45.92</td></tr><tr><td>MaskLLM</td><td>2:4</td><td>27.65</td><td>74.76</td><td>69.44</td><td>35.58</td><td>65.04</td><td>26.80</td><td>38.56</td><td>51.15</td><td>48.62</td></tr><tr><td>45%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>27.28</td><td>75.41</td><td>70.16</td><td>35.84</td><td>65.27</td><td>27.60</td><td>38.76</td><td>51.61</td><td>48.99</td></tr><tr><td>35%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>29.93</td><td>76.71</td><td>70.88</td><td>36.95</td><td>65.67</td><td>28.20</td><td>39.33</td><td>52.96</td><td>50.08</td></tr><tr><td>25%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>32.33</td><td>76.99</td><td>72.81</td><td>38.57</td><td>68.27</td><td>29.80</td><td>39.52</td><td>54.34</td><td>51.58</td></tr></table>

Table D.5: Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-3.1 8B with diferent pruning methods. PATCH<sup>Tile</sup> optimizes tile-based sparsity, enabling a flexible sparsity-quality tradeof.
<table><tr><td>Sparsity</td><td>Method</td><td>Pattern</td><td>MMLU</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>WinoG.</td><td>OBQA</td><td>RACE</td><td>HellaS.</td><td>Avg</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>63.57</td><td>80.09</td><td>81.44</td><td>51.37</td><td>73.48</td><td>33.40</td><td>39.14</td><td>60.02</td><td>60.31</td></tr><tr><td>50%</td><td>Magnitude</td><td>2:4</td><td>23.06</td><td>63.82</td><td>45.33</td><td>25.94</td><td>53.91</td><td>15.20</td><td>26.70</td><td>33.49</td><td>35.93</td></tr><tr><td></td><td>Wanda</td><td>2:4</td><td>27.85</td><td>68.88</td><td>58.33</td><td>26.71</td><td>60.93</td><td>19.00</td><td>33.78</td><td>38.70</td><td>41.77</td></tr><tr><td></td><td>SparseGPT 2:4</td><td></td><td>31.82</td><td>70.46</td><td>63.85</td><td>31.74</td><td>64.56</td><td>21.60</td><td>37.22</td><td>42.99</td><td>45.53</td></tr><tr><td></td><td>Thanos</td><td>2:4</td><td>34.23</td><td>70.40</td><td>63.13</td><td>31.40</td><td>63.61</td><td>23.20</td><td>37.03</td><td>42.75</td><td>45.72</td></tr><tr><td></td><td>ProxSparse 2:4</td><td></td><td>29.89</td><td>71.71</td><td>62.63</td><td>33.28</td><td>58.56</td><td>23.80</td><td>35.22</td><td>46.03</td><td>45.14</td></tr><tr><td></td><td>MaskLLM</td><td>2:4</td><td>42.47</td><td>77.04</td><td>73.15</td><td>40.19</td><td>68.43</td><td>28.80</td><td>38.28</td><td>54.04</td><td>52.80</td></tr><tr><td>45% 35%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>47.32</td><td>77.96</td><td>73.61</td><td>41.89</td><td>68.03</td><td>29.00</td><td>36.56</td><td>54.44</td><td>53.60</td></tr><tr><td></td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>51.15</td><td>77.97</td><td>76.14</td><td>42.41</td><td>69.46</td><td>31.40</td><td>38.18</td><td>55.54</td><td>55.28</td></tr><tr><td>25%</td><td>PATCHTile</td><td>Dense/2:4 Tiles</td><td>52.95</td><td>77.75</td><td>77.57</td><td>44.62</td><td>70.56</td><td>31.80</td><td>39.90</td><td>56.69</td><td>56.48</td></tr></table>

Table D.6: Model quality (task accuracy across eight zero-shot tasks, reported in %) for LLaMA-3.2 1B with diferent pruning methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity patterns, enabling a flexible sparsity-quality tradeof.
<table><tr><td>Sparsity Method</td><td></td><td>Pattern</td><td>MMLU</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>WinoG.</td><td>OBQA</td><td>RACE</td><td>HellaS.</td><td>Avg</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>37.57</td><td>74.54</td><td>65.53</td><td>31.32</td><td>60.62</td><td>26.40</td><td>37.89</td><td>47.76</td><td>47.70</td></tr><tr><td>50%</td><td>Magnitude</td><td>2:4</td><td>23.31</td><td>53.81</td><td>27.74</td><td>18.94</td><td>51.38</td><td>11.80</td><td>24.02</td><td>26.26</td><td>29.66</td></tr><tr><td></td><td>Wanda</td><td>2:4</td><td>22.90</td><td>58.11</td><td>37.08</td><td>19.20</td><td>49.09</td><td>13.20</td><td>25.17</td><td>28.11</td><td>31.61</td></tr><tr><td></td><td>SparseGPT 2:4</td><td></td><td>22.93</td><td>61.43</td><td>45.03</td><td>22.35</td><td>54.93</td><td>15.80</td><td>29.86</td><td>32.08</td><td>35.55</td></tr><tr><td></td><td>Thanos</td><td>2:4</td><td>23.12</td><td>62.40</td><td>44.91</td><td>21.76</td><td>54.30</td><td>16.00</td><td>31.10</td><td>32.09</td><td>35.71</td></tr><tr><td></td><td>ProxSparse 2:4</td><td></td><td>22.96</td><td>60.83</td><td>39.44</td><td>20.31</td><td>51.54</td><td>16.80</td><td>25.17</td><td>31.37</td><td>33.55</td></tr><tr><td></td><td>MaskLLM</td><td>2:4</td><td>26.28</td><td>69.10</td><td>57.41</td><td>25.85</td><td>55.48</td><td>21.40</td><td>32.82</td><td>39.94</td><td>41.04</td></tr><tr><td>45% 35%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>23.81</td><td>70.89</td><td>60.77</td><td>27.22</td><td>56.27</td><td>22.80</td><td>34.07</td><td>40.78</td><td>42.08</td></tr><tr><td></td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>25.13</td><td>71.32</td><td>60.27</td><td>29.18</td><td>57.06</td><td>22.00</td><td>34.64</td><td>42.17</td><td>42.72</td></tr><tr><td>25%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>28.59</td><td>71.44</td><td>61.57</td><td>28.67</td><td>58.25</td><td>23.20</td><td>35.22</td><td>43.52</td><td>43.81</td></tr></table>

Table D.7: Model quality (accuracy across eight zero-shot tasks) for Gemma-3 1B with diferent pruning methods. PATCH<sup>Joint</sup> optimizes dense tile locations and sparsity patterns, enabling a flexible sparsity-quality tradeof.
<table><tr><td>Sparsity</td><td>Method</td><td>Pattern</td><td>MMLU</td><td>PIQA</td><td>ARC-E</td><td>ARC-C</td><td>WinoG.</td><td>OBQA</td><td>RACE</td><td>HellaS.</td><td>Avg</td></tr><tr><td>0%</td><td>Dense</td><td>-</td><td>24.95</td><td>75.03</td><td>71.84</td><td>34.90</td><td>58.64</td><td>28.60</td><td>34.83</td><td>47.26</td><td>47.01</td></tr><tr><td>50%</td><td>Magnitude</td><td>2:4</td><td>23.08</td><td>59.79</td><td>37.29</td><td>17.66</td><td>50.59</td><td>14.00</td><td>22.87</td><td>27.97</td><td>31.66</td></tr><tr><td></td><td>Wanda</td><td>2:4</td><td>23.96</td><td>59.52</td><td>48.02</td><td>18.34</td><td>51.22</td><td>14.20</td><td>27.85</td><td>30.18</td><td>34.16</td></tr><tr><td></td><td>SparseGPT 2:4</td><td></td><td>23.62</td><td>62.79</td><td>49.83</td><td>19.03</td><td>51.54</td><td>15.20</td><td>30.62</td><td>31.99</td><td>35.58</td></tr><tr><td></td><td>Thanos</td><td>2:4</td><td>23.44</td><td>62.24</td><td>48.86</td><td>18.34</td><td>50.12</td><td>15.60</td><td>30.81</td><td>31.28</td><td>35.09</td></tr><tr><td></td><td>ProxSparse 2:4</td><td></td><td>23.10</td><td>64.25</td><td>50.72</td><td>21.59</td><td>53.43</td><td>18.00</td><td>29.09</td><td>32.86</td><td>36.63</td></tr><tr><td></td><td>MaskLLM</td><td>2:4</td><td>25.03</td><td>69.91</td><td>60.27</td><td>27.65</td><td>56.27</td><td>21.20</td><td>34.55</td><td>39.84</td><td>41.84</td></tr><tr><td>45% 35%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>23.54</td><td>71.65</td><td>63.97</td><td>27.47</td><td>57.30</td><td>23.60</td><td>33.49</td><td>41.39</td><td>42.80</td></tr><tr><td></td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>25.38</td><td>72.31</td><td>63.80</td><td>27.39</td><td>56.67</td><td>24.00</td><td>34.74</td><td>42.07</td><td>43.30</td></tr><tr><td>25%</td><td>PATCHJoint</td><td>Dense/2:4 Tiles</td><td>25.45</td><td>71.87</td><td>66.16</td><td>30.55</td><td>57.85</td><td>22.80</td><td>34.55</td><td>43.33</td><td>44.07</td></tr></table>

## D.3 Tile Transfer Learning

We also test whether initializing tile logits with priors from one-shot pruning methods improves performance, as done in MaskLLM [33]. In our case, the initialization is derived from one-shot pruning with unstructured sparsity. We initialize tiles that retain more nonzeros after unstructured pruning with positive logits (favoring dense assignment), while the remaining tiles receive negative logits, controlled by a strength parameter. The number of tiles initialized as dense is selected such that the overall layer-wise sparsity target is satisfied. As shown in Table D.8, the choice of prior has little impact on final performance: all priors yield nearly identical perplexity, with random initialization often performing best. This is likely because the global sparsity target enables dynamic reallocation of sparsity across layers during training, overriding the efect of any fixed initialization. For consistency with prior work, we adopt SparseGPT initialization in all experiments.

Table D.8: Perplexity (↓) under diferent tile prior initializations. All priors yield nearly identical performance, suggesting that the global sparsity target allows dynamic reallocation of sparsity during training, overriding the influence of fixed initialization.
<table><tr><td>Sparsity (0.5B)</td><td>Nothing</td><td>SparseGPT</td><td>Wanda</td><td>Magnitude</td><td>Random</td></tr><tr><td>45%</td><td>14.80</td><td>14.57</td><td>14.50</td><td>14.48</td><td>14.51</td></tr><tr><td>35%</td><td>13.97</td><td>13.84</td><td>13.87</td><td>13.85</td><td>13.79</td></tr><tr><td>25%</td><td>13.47</td><td>13.47</td><td>13.37</td><td>13.44</td><td>13.33</td></tr></table>

## Appendix E

## Supplementary Material for SLIM

This chapter provides supplementary material for the SLIM chapter. We begin by defining the notations used throughout this work in Appendix E.1. Appendix E.2 evaluates input quantization, and Appendix E.3 present additional fine-tuning results. Language modeling experiments are reported in Appendix E.4, with additional sparse and quantized accuracy results in Appendix E.5. Appendix E.6 compares sparsity and quantization, while Appendix E.7 provides additional speedup results. Appendix E.8 details the fine-tuning costs, and theoretical analyses of memory and computation reductions are provided in Appendix E.9 and Appendix E.10, respectively. Appendix E.11 reports compression costs, Appendix E.12 explores the impact of rank choices on low-rank adapters, and Appendix E.14 examines sensitivity to the calibration dataset. Appendix E.15 analyze the efects of diferent sparsity ratios, and Appendix E.16 discusses the challenges of group quantization.

## E.1 Notations

Table E.1 details the key notations, particularly for Appendix 7.5.

Table E.1: Key notation definitions used in the experimental results (Appendix 7.5).
<table><tr><td rowspan=1 colspan=1>Term</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Naive-LoRA</td><td rowspan=1 colspan=1>A one-shot low-rank adapter that minimizes the norm of the difference be-tween the original and the compressed weights.</td></tr><tr><td rowspan=1 colspan=1>SLiM-LoRA</td><td rowspan=1 colspan=1>A saliency-based one-shot low-rank adapter that minimizes the saliency ofthe difference between the original and the compressed weights.</td></tr><tr><td rowspan=1 colspan=1>Q (Superscript)</td><td rowspan=1 colspan=1>Q indicates that the compression method quantizes the low-rank adapters aswell.</td></tr><tr><td rowspan=1 colspan=1>+ FT</td><td rowspan=1 colspan=1>+ FT shows a short fine-tuning phase on 300,000 tokens from the C4 dataset.</td></tr></table>

## E.2 Input Quantization

We evaluate SLIM with 8-bit input quantization to assess its impact on accuracy. We use AbsMax uniform quantization with a single parameter per input tensor and apply FP8 format [96] for weight quantization. The choice between E4M3 and E5M2 depends on the tensor’s maximum value; if it exceeds E4M3’s range, we switch to E5M2 for greater expressivity. Next, we examine how input quantization afects model accuracy.

Table E.2 presents accuracy results for diferent SLIM variants with input quantization. A comparison with Table E.3, which reports accuracy without input quantization, reveals minimal accuracy loss, demonstrating SLIM’s robustness. For further validation, we extend these experiments to language modeling tasks (Appendix E.4).

Table E.2: Average zero-shot accuracy of LLaMA-2 and OPT models with 4-bit weight and 8-bit input quantization. ↑ indicates better performance.
<table><tr><td rowspan="2">Pruning/LoRA Method</td><td rowspan="2">Weight Quantization</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>–</td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>50% 2:4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathbf { S L I M - L o R A }$ </td><td>SLıM-Quant</td><td>34.85</td><td>34.27</td><td>40.29</td><td>42.58</td><td>45.78</td><td>46.21</td><td>50.99</td><td>54.66</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLıM-Quant</td><td>35.28</td><td>34.33</td><td>41.14</td><td>43.29</td><td>46.44</td><td>47.33</td><td>51.77</td><td>56.28</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>34.30</td><td>33.85</td><td>39.92</td><td>41.99</td><td>46.08</td><td>45.94</td><td>50.70</td><td>53.56</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>34.92</td><td>34.80</td><td>41.66</td><td>43.69</td><td>46.03</td><td>46.87</td><td>50.26</td><td>56.28</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLiM-LoRA</td><td>SLiM-Quant</td><td>35.12</td><td>34.86</td><td>41.94</td><td>43.53</td><td>47.27</td><td>47.70</td><td>54.28</td><td>57.82</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLıM-Quant</td><td>35.18</td><td>35.30</td><td>42.37</td><td>44.02</td><td>47.01</td><td>48.52</td><td>54.43</td><td>57.70</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>35.26</td><td>34.67</td><td>41.48</td><td>43.46</td><td>47.25</td><td>47.76</td><td>53.91</td><td>57.16</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>35.52</td><td>35.31</td><td>42.66</td><td>44.50</td><td>47.08</td><td>48.53</td><td>53.23</td><td>57.55</td></tr></table>

## E.3 Additional fine-tuning results

To complement the results in Appendix 7.5, we provide accuracy measurements for PEFT-based fine-tuning of low-rank adapters on the OPT and LLaMA-2 model families in Table E.3 while showing the accuracy results without fine-tuning for comparison. The results confirm the previously observed trend: lightweight fine-tuning enhances the accuracy of all baselines, with SLIM-LoRA achieving the most significant improvements due to its saliency-based design.

## E.4 Language modeling experiments

We evaluate all benchmarks from Appendix 7.5 and Appendix E.2 on the WikiText2 language modeling task. Table E.4 and Table E.6 show perplexity results for 4-bit quantized models with 2:4 and unstructured sparsity, respectively. Table E.5 summarizes the results for 8-bit input quantization. To examine sparsity and quantization independently, Table E.7 and Table E.8 report results for pruning-only and quantization-only models. Consistent with Appendix 7.5, SLIM achieves superior performance across all settings.

Table E.3: Efects of fine-tuning on the average zero-shot accuracy of LLaMA-2 and OPT models with 50% sparsity and 4-bit weight quantization. ↑ indicates better performance.
<table><tr><td>Pruning/LoRA Method</td><td rowspan="2">Weight</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>Quantization</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>–</td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>50% 2:4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Naive-LoRA</td><td>SLıM-Quant</td><td>34.28</td><td>33.38</td><td>38.36</td><td>41.21</td><td>44.91</td><td>45.25</td><td>48.45</td><td>51.94</td></tr><tr><td> $\mathrm { N a i v e - L o R A + F T }$ </td><td>SLıM-Quant</td><td>34.41</td><td>34.70</td><td>39.72</td><td>42.88</td><td>46.16</td><td>46.76</td><td>50.89</td><td>55.70</td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>34.62</td><td>34.36</td><td>40.61</td><td>42.73</td><td>45.99</td><td>46.09</td><td>51.15</td><td>54.94</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLıM-Quant</td><td>35.03</td><td>34.58</td><td>41.11</td><td>43.35</td><td>46.71</td><td>47.25</td><td>52.12</td><td>56.60</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>34.43</td><td>34.30</td><td>40.11</td><td>42.37</td><td>46.33</td><td>46.24</td><td>51.02</td><td>53.55</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>34.92</td><td>34.85</td><td>41.84</td><td>43.87</td><td>46.31</td><td>46.91</td><td>48.31</td><td>56.50</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Naive-LoRA</td><td>SLıM-Quant</td><td>34.77</td><td>34.23</td><td>40.40</td><td>43.37</td><td>46.64</td><td>47.30</td><td>51.52</td><td>55.33</td></tr><tr><td> $\mathrm { N a i v e - L o R A + F T }$ </td><td>SLıM-Quant</td><td>35.70</td><td>35.47</td><td>41.89</td><td>44.16</td><td>47.08</td><td>47.78</td><td>52.90</td><td>57.08</td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>35.20</td><td>35.32</td><td>41.85</td><td>43.48</td><td>47.08</td><td>47.96</td><td>54.26</td><td>57.85</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLıM-Quant</td><td>35.59</td><td>35.71</td><td>42.37</td><td>44.58</td><td>47.69</td><td>48.26</td><td>54.69</td><td>57.96</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>35.35</td><td>35.13</td><td>41.74</td><td>43.63</td><td>47.16</td><td>47.86</td><td>54.18</td><td>57.33</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>35.65</td><td>35.67</td><td>42.74</td><td>44.54</td><td>47.48</td><td>48.40</td><td>53.57</td><td>57.78</td></tr></table>

Table E.4: Perplexity of LLaMA-2 and OPT models with 2:4 sparsity and 4-bit weight quantization on WikiText-2 dataset language modeling task. ↓ indicates better performance.
<table><tr><td rowspan="2">Pruning/LoRA Method</td><td rowspan="2">Weight Quantization</td><td colspan="5">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td></td><td>27.66</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>5.12</td><td>4.57</td></tr><tr><td></td><td>Group AbsMax</td><td>5.1E2</td><td>4.4E2</td><td>1.2E3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude SparseGPT</td><td>Group OPTQ</td><td>78.18</td><td>59.86</td><td>27.36</td><td>1.3E3 18.62</td><td>3.6E2 15.31</td><td>4.9E2 13.25</td><td>86.34 15.01</td><td>8.98 8.97</td></tr><tr><td>Wanda</td><td>Group AbsMax</td><td>1.8E2</td><td>1.3E2</td><td>32.76</td><td>24.48</td><td>17.29</td><td>16.86</td><td>13.46</td><td>8.70</td></tr><tr><td>Wanda</td><td>AWQ</td><td>9.3E1</td><td>8.1E5</td><td>29.56</td><td>22.91</td><td>16.28</td><td>16.72</td><td>12.79</td><td>00M</td></tr><tr><td>Wanda</td><td>OmniQuant</td><td>9.7E1</td><td>NaN</td><td>33.61</td><td>25.89</td><td>19.09</td><td>00M</td><td>12.77</td><td>00M</td></tr><tr><td>Wanda</td><td>AffineQuant</td><td>9.7E1</td><td>NaN</td><td>30.32</td><td>1.6E3</td><td>16.85</td><td>00M</td><td>12.21</td><td>O0M</td></tr><tr><td>JSQ</td><td>JSQ</td><td>3.5E3</td><td>2.5E4</td><td>67.36</td><td>3.2E3</td><td></td><td>5.5E2</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>22.50</td><td></td><td>11.69</td><td>8.05</td></tr><tr><td>Naive-LoRA</td><td>Group AbsMax</td><td>69.23</td><td>50.02</td><td>20.52</td><td>16.05</td><td>12.83</td><td>13.12</td><td>8.04</td><td></td></tr><tr><td>Naive-LoRA</td><td>SLıM-Quant</td><td>83.08</td><td>58.69</td><td>27.06</td><td>20.92</td><td>14.29</td><td>13.20</td><td></td><td>6.38</td></tr><tr><td>Naive-LoRA + FT</td><td>SLıM-Quant</td><td>51.82</td><td>38.84</td><td>20.59</td><td>16.19</td><td>13.13</td><td>12.55</td><td>8.19 6.96</td><td>7.09 6.01</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>57.91</td><td>50.09</td><td>19.64</td><td>15.65</td><td>12.71</td><td>12.13</td><td>7.56</td><td>6.50</td></tr><tr><td>SLiM-LoRA + FT</td><td>SLıM-Quant</td><td>44.03</td><td>37.32</td><td>18.25</td><td>14.89</td><td>12.68</td><td>12.06</td><td>6.70</td><td>6.60</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLıM-Quant</td><td>53.09</td><td>46.96</td><td>19.62</td><td>16.01</td><td>12.48</td><td>12.15</td><td>7.75</td><td>6.96</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>42.80</td><td>37.39</td><td>18.38</td><td>15.40</td><td>12.65</td><td>12.35</td><td>7.08</td><td>6.36</td></tr></table>

Table E.5: Perplexity of LLaMA-2 and OPT models with 4-bit weight and 8-bit input quantization. ↓ indicates better performance.
<table><tr><td rowspan="2">Pruning/LoRA Method</td><td rowspan="2">Weight Quantization</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>-</td><td>27.66</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>5.12</td><td>4.57</td></tr><tr><td>50% 2:4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLiM-LoRA</td><td>SLrM-Quant</td><td>48.4</td><td>49.6</td><td>16.6</td><td>16.2</td><td>12.9</td><td>12.3</td><td>7.2</td><td>6.5</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLiM-Quant</td><td>39.8</td><td>37.5</td><td>18.3</td><td>15.5</td><td>12.8</td><td>12.1</td><td>6.6</td><td>5.8</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLiM-Quant</td><td>54.2</td><td>50.8</td><td>20.8</td><td>16.8</td><td>13.0</td><td>12.4</td><td>7.8</td><td>7.0</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLrM-Quant</td><td>43.4</td><td>39.1</td><td>19.3</td><td>16.0</td><td>13.1</td><td>12.6</td><td>7.1</td><td>5.8</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLiM-LoRA</td><td>SL1M-Quant</td><td>36.8</td><td>31.1</td><td>16.8</td><td>14.0</td><td>11.7</td><td>10.9</td><td>6.1</td><td>5.4</td></tr><tr><td> $\mathrm { S L I M - L o R A + F T }$ </td><td>SLıM-Quant</td><td>33.8</td><td>28.6</td><td>16.5</td><td>14.0</td><td>12.0</td><td>11.5</td><td>5.9</td><td>5.2</td></tr><tr><td> $\mathbf { S L I M - L o R A } ^ { Q }$ </td><td>SLiM-Quant</td><td>39.5</td><td>31.3</td><td>17.3</td><td>14.2</td><td>11.8</td><td>10.9</td><td>6.3</td><td>5.6</td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>35.6</td><td>29.1</td><td>17.0</td><td>14.3</td><td>12.2</td><td>11.7</td><td>6.2</td><td>5.5</td></tr></table>

Table E.6: Perplexity of LLaMA-2 and OPT models with unstructured sparsity and 4-bit weight quantization on WikiText-2 dataset language modeling task. ↓ indicates better performance.
<table><tr><td rowspan="2">Pruning/LoRA Method</td><td rowspan="2">Weight Quantization</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>=</td><td>27.66</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>5.12</td><td>4.57</td></tr><tr><td></td><td></td><td>3.2E2</td><td>1.1E2</td><td>3.2E3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude SparseGPT</td><td>Group AbsMax Group OPTQ</td><td>42.60</td><td>34.19</td><td>21.41</td><td>3.6E2 14.30</td><td>7.2E2 12.15</td><td>5.4E3 11.26</td><td>17.18 8.28</td><td>6.77 5.92</td></tr><tr><td>Wanda</td><td>Group AbsMax</td><td>62.64</td><td>39.60</td><td>19.93</td><td>15.01</td><td>12.31</td><td>12.46</td><td>6.80</td><td>5.75</td></tr><tr><td>Wanda</td><td>AWQ</td><td>42.49</td><td>3.8E5</td><td>18.80</td><td>14.67</td><td>12.17</td><td>12.34</td><td>7.28</td><td>00M</td></tr><tr><td>Wanda</td><td>OmniQuant</td><td>43.55</td><td>NaN</td><td>20.58</td><td>15.82</td><td>13.29</td><td>00M</td><td>7.40</td><td>00M</td></tr><tr><td>Wanda</td><td>AffineQuant</td><td>43.66</td><td>NaN</td><td>19.40</td><td>14.94</td><td>12.39</td><td>00M</td><td></td><td></td></tr><tr><td>JSQ</td><td>JSQ</td><td>3.2E3</td><td>1.9E4</td><td>23.88</td><td></td><td></td><td></td><td>7.21</td><td>O0M</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>2.3E2</td><td>15.13</td><td>1.2E5</td><td>6.63</td><td>5.73</td></tr><tr><td></td><td></td><td></td><td>30.99</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Naive-LoRA</td><td>Group AbsMax</td><td>40.37</td><td></td><td>17.02</td><td>13.91</td><td>11.68</td><td>11.38</td><td>6.12</td><td>5.28</td></tr><tr><td>Naive-LoRA Naive-LoRA + FT</td><td>SLıM-Quant</td><td>46.66</td><td>33.90</td><td>19.46</td><td>15.36</td><td>12.16</td><td>11.41</td><td>6.56</td><td>5.58</td></tr><tr><td></td><td>SLıM-Quant</td><td>38.05</td><td>29.27</td><td>17.52</td><td>14.39</td><td>12.28</td><td>11.84</td><td>6.10</td><td>5.28</td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>39.62</td><td>31.51</td><td>16.52</td><td>13.65</td><td>11.42</td><td>10.82</td><td></td><td>5.36</td></tr><tr><td>SLiM-LoRA + FT</td><td>SLıM-Quant</td><td>34.92</td><td>28.67</td><td>16.16</td><td>13.66</td><td>11.83</td><td>11.47</td><td>6.16 5.36</td><td></td></tr><tr><td>SLiM-LoRAQ</td><td>SLıM-Quant</td><td>38.79</td><td>30.16</td><td>16.64</td><td>13.82</td><td>11.43</td><td>10.80</td><td>6.26</td><td>5.19 5.58</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { S L I M - L o R A } ^ { Q } + \mathrm { F T }$ </td><td>SLıM-Quant</td><td>35.17</td><td>28.31</td><td>16.46</td><td>13.96</td><td>11.42</td><td>10.80</td><td>5.94</td><td>5.46</td></tr></table>

Table E.7: Perplexity of LLaMA-2 and OPT models with pruning on WikiText-2 dataset language modeling task. The quantization is disabled in this experiment. ↓ indicates better performance.
<table><tr><td colspan="2">Pruning/LoRA</td><td colspan="4">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Method</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>27.66</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>5.12</td><td>4.57</td></tr><tr><td>2:4 Sparsity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Magnitude</td><td>341.5</td><td>417.1</td><td>427.2</td><td>1.2E3</td><td>264.1</td><td>4.0E4</td><td>9.1E4</td><td>2.0E5</td></tr><tr><td>SparseGPT</td><td>60.7</td><td>50.7</td><td>23.8</td><td>17.2</td><td>14.1</td><td>12.9</td><td>10.2</td><td>8.3</td></tr><tr><td>Wanda</td><td>81.6</td><td>116.0</td><td>27.8</td><td>21.4</td><td>16.0</td><td>16.4</td><td>12.0</td><td>8.5</td></tr><tr><td>Naive-LoRA</td><td>46.9</td><td>45.0</td><td>18.8</td><td>15.2</td><td>12.5</td><td>12.9</td><td>8.1</td><td>6.5</td></tr><tr><td>Naive-LoRA + FT</td><td>39.6</td><td>35.1</td><td>15.0</td><td>16.3</td><td>12.7</td><td>12.3</td><td>6.5</td><td>5.7</td></tr><tr><td>SLiM-LoRA</td><td>45.2</td><td>43.6</td><td>18.6</td><td>15.0</td><td>12.4</td><td>12.6</td><td>7.3</td><td>6.2</td></tr><tr><td>SLiM-LoRA + FT</td><td>37.1</td><td>33.7</td><td>17.0</td><td>14.2</td><td>12.4</td><td>12.1</td><td>6.4</td><td>5.8</td></tr><tr><td colspan="9">50% Unstructured</td></tr><tr><td>Magnitude</td><td>193.4</td><td>97.8</td><td>1.7E3</td><td>265.2</td><td>968.7</td><td>2.4E4</td><td>9.9E4</td><td>1.1E5</td></tr><tr><td>SparseGPT</td><td>36.7</td><td>31.8</td><td>17.6</td><td>13.4</td><td>11.5</td><td>11.1</td><td>6.5</td><td>5.6</td></tr><tr><td>Wanda</td><td>39.3</td><td>36.4</td><td>18.3</td><td>14.3</td><td>12.0</td><td>12.3</td><td>6.4</td><td>5.4</td></tr><tr><td>Naive-LoRA</td><td>33.3</td><td>29.1</td><td>16.3</td><td>13.5</td><td>11.5</td><td>11.2</td><td>6.2</td><td>5.4</td></tr><tr><td>Naive-LoRA + FT</td><td>31.9</td><td>27.5</td><td>16.3</td><td>13.8</td><td>12.0</td><td>11.6</td><td>5.8</td><td>5.1</td></tr><tr><td>SLiM-LoRA</td><td>32.7</td><td>29.0</td><td>15.9</td><td>13.2</td><td>11.2</td><td>10.8</td><td>5.9</td><td>5.2</td></tr><tr><td>SLiM-LoRA + FT</td><td>31.0</td><td>26.8</td><td>15.5</td><td>13.1</td><td>11.6</td><td>11.0</td><td>5.8</td><td>4.7</td></tr></table>

Table E.8: Perplexity of LLaMA-2 and OPT models with quantization on WikiText-2 dataset language modeling task. The sparsity is disabled in this experiment. ↓ indicates better performance.
<table><tr><td>Quantization</td><td>Low-rank</td><td colspan="5">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>Method</td><td>Adapter</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td>I</td><td>27.66</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>5.12</td><td>4.57</td></tr><tr><td>OPTQ</td><td></td><td>33.0</td><td>24.4</td><td>16.0</td><td>13.0</td><td>11.3</td><td>10.3</td><td>6.1</td><td>4.9</td></tr><tr><td>AWQ</td><td></td><td>29.1</td><td>2.7E5</td><td>14.9</td><td>12.7</td><td>11.0</td><td>10.2</td><td>6.0</td><td>00M</td></tr><tr><td>OmniQuant</td><td></td><td>30.2</td><td>NaN</td><td>15.8</td><td>13.3</td><td>11.6</td><td>00M</td><td>5.7</td><td>00M</td></tr><tr><td>AffineQuant</td><td>=</td><td>28.7</td><td>NaN</td><td>14.9</td><td>12.6</td><td>11.0</td><td>00M</td><td>5.7</td><td>00M</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Group AbsMax</td><td></td><td>35.1 30.4</td><td>23.3 22.9</td><td>15.5 15.1</td><td>12.9 12.7</td><td>11.1 11.0</td><td>10.3 10.2</td><td>5.4</td><td>4.7</td></tr><tr><td>Group AbsMax Group AbsMax</td><td>Naive-LoRA SLiM-LoRA</td><td>29.3</td><td>22.8</td><td>15.0</td><td>12.7</td><td>10.9</td><td>10.2</td><td>5.3 5.2</td><td>4.7 4.7</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLıM-Quant</td><td></td><td>1.4E3</td><td>26.0</td><td>1.7E3</td><td>33.1</td><td>31.0</td><td>6.7E2</td><td>1.3E5</td><td>7.8E4</td></tr><tr><td>SLıM-Quant</td><td>Naive-LoRA</td><td>32.1</td><td>24.1</td><td>15.6</td><td>13.4</td><td>11.2</td><td>10.5</td><td>5.4</td><td>4.8</td></tr><tr><td>SLıM-Quant</td><td>SLiM-LoRA</td><td>30.8</td><td>23.1</td><td>15.2</td><td>12.9</td><td>11.1</td><td>10.3</td><td>5.4</td><td>4.8</td></tr><tr><td>SLıM-Quant</td><td>SLiM-LoRA + FT</td><td>30.7</td><td>23.5</td><td>15.3</td><td>13.3</td><td>11.6</td><td>10.0</td><td>5.3</td><td>4.7</td></tr></table>

## E.5 Additional Sparse and Quantized Results

In Appendix 7.5, we provided the accuracy results for diferent pruning and quantization methods. When using Wanda for pruning, we only reported the best quantization method out of Group AbsMax, AWQ, OmniQuant, and AfineQuant. For completeness, we have provided the accuracy achieved by each of these quantization methods separately in Table E.9.

Methods like OmniQuant and AfineQuant encounter dificulties in quantizing OPT-350M, resulting in NaN values. Additionally, approaches such as AWQ, OmniQuant, and AfineQuant cause memory issues (OOM) when attempting to compress the models on a single A100-40GB GPU.

Table E.9: Average zero-shot accuracy of LLaMA-2 and OPT models with 2:4 sparsity and 4-bit weight quantization. ↑ indicates better performance.
<table><tr><td>Pruning/LoRA Method</td><td rowspan="2">Weight</td><td colspan="5">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Quantization</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Dense</td><td></td><td>35.9</td><td>37.1</td><td>43.4</td><td>45.5</td><td>48.3</td><td>48.7</td><td>56.6</td><td>60.8</td></tr><tr><td>2:4 Sparsity</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wanda</td><td>Group AbsMax</td><td>33.27</td><td>32.79</td><td>37.47</td><td>39.45</td><td>42.95</td><td>43.64</td><td>43.89</td><td>48.94</td></tr><tr><td>Wanda</td><td>AWQ</td><td>33.33</td><td>31.50</td><td>38.43</td><td>40.00</td><td>43.41</td><td>44.07</td><td>44.86</td><td>O0M</td></tr><tr><td>Wanda</td><td>OmniQuant</td><td>33.37</td><td>NaN</td><td>37.35</td><td>39.39</td><td>41.50</td><td>00M</td><td>43.95</td><td>00M</td></tr><tr><td>Wanda</td><td>AffineQuant</td><td>33.39</td><td>NaN</td><td>37.48</td><td>33.51</td><td>42.88</td><td>00M</td><td>44.62</td><td>00M</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wanda</td><td>Group AbsMax</td><td>34.67</td><td>33.89</td><td>40.38</td><td>42.77</td><td>45.88</td><td>46.60</td><td>51.76</td><td>56.76</td></tr><tr><td>Wanda</td><td>AWQ</td><td>35.11</td><td>31.57</td><td>41.02</td><td>42.89</td><td>46.52</td><td>46.84</td><td>50.68</td><td>00M</td></tr><tr><td>Wanda</td><td>OmniQuant</td><td>34.85</td><td>NaN</td><td>39.84</td><td>42.16</td><td>44.67</td><td>00M</td><td>50.51</td><td>00M</td></tr><tr><td>Wanda</td><td>AffineQuant</td><td>34.64</td><td>NaN</td><td>41.23</td><td>42.68</td><td>46.05</td><td>OOM</td><td>53.62</td><td>O0M</td></tr></table>

## E.6 Sparsity vs. quantization

A natural question that arises when compressing models is whether it is more eficient to reduce the model size through pruning or quantization. To answer this question, we conduct a set of experiments, which evaluate the perplexity of diferent models under three diferent conditions, all with around 8× model size reduction factor: (1) 2-bit weight quantization with no sparsity, (2) 4-bit weight quantization with 50% unstructured sparsity, and (3) 4-bit weight quantization with 50% 2:4 sparsity. We have used SLIM-LoRA with SLIM-Quant in all the experiments. The accuracy and perplexity results of these experiments are summarized in Table E.10 and Table E.11, showing that combining sparsity and quantization yields better results in compar ison to quantization-only settings with lower bitwidth.

Table E.10: Average zero-shot accuracy of diferent models using diferent pruning and quantization schemes. ↑ indicates better performance. Combining sparsity and quantization provides better accuracy results in com parison to solely using quantization.
<table><tr><td colspan="7">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Quantization</td><td>Sparsity</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>2-bit</td><td></td><td>33.5</td><td>32.5</td><td>38.5</td><td>39.2</td><td>43.8</td><td>44.4</td><td>42.4</td><td>44.9</td></tr><tr><td>4-bit</td><td>2:4</td><td>34.6</td><td>34.4</td><td>40.6</td><td>42.7</td><td>46.0</td><td>46.1</td><td>51.2</td><td>54.9</td></tr><tr><td>4-bit</td><td>50% Unstructured</td><td>35.2</td><td>35.3</td><td>41.9</td><td>43.5</td><td>47.1</td><td>48.0</td><td>54.3</td><td>57.9</td></tr></table>

Table E.11: Perplexity of diferent models on WikiText-2 dataset using diferent pruning and quantization schemes. ↓ indicates better performance. Combining sparsity and quantization provides better accuracy results in comparison to solely using quantization.
<table><tr><td colspan="7">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Quantization</td><td>Sparsity</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>2-bit</td><td></td><td>116.2</td><td>169.7</td><td>35.1</td><td>27.1</td><td>16.2</td><td>15.0</td><td>12.5</td><td>11.7</td></tr><tr><td>4-bit</td><td>2:4</td><td>47.5</td><td>45.6</td><td>18.8</td><td>15.7</td><td>12.4</td><td>12.1</td><td>7.2</td><td>6.5</td></tr><tr><td>4-bit</td><td>50% Unstructured</td><td>36.3</td><td>29.9</td><td>16.3</td><td>13.7</td><td>11.4</td><td>10.8</td><td>6.0</td><td>5.4</td></tr></table>

## E.7 Additional speedup results

Appendix 7.5 presents the speedup of SLIM on consumer-grade GPUs, while this section provides results on NVIDIA A100-40GB GPUs. Figure E.1 summarizes the speedup for the LLaMA-2 and LLaMA-3.1 model families, including LLaMA-3.1-405B, highlighting SLIM’s scalability to large models. As with consumergrade devices, larger models achieve higher speedups. However, for smaller models like LLaMA-2-7B, Sparse Marlin’s sparse quantized matrix multiplication kernels lead to a slowdown on A100 GPUs, which does not occur on consumer-grade GPUs and is not specific to SLIM .

![](images/02b97bba7fe0b95ba16749ea3025f089f403f50ec98fc74bee819241d530ef0c.jpg)  
Figure E.1: SLIM speedup for LLaMA-2 family of models on NVIDIA A100-40GB GPUs.

## E.8 Fine-tuning costs

Fine-tuning compressed models can recover lost accuracy, but the high parameter count leads to substantial time and memory costs. In our experiments, we fine-tuned models with low-rank adapters, where the quantized weights are frozen and only the adapters are fine-tuned. This results in a more parameter-eficient approach, reducing both memory and computational costs. When no low-rank adapter is used, the straight-through estimator (STE) fine-tunes the quantized weights.

Table E.12 presents the fine-tuning results for 300,000 tokens from the C4 dataset, using a batch size of 64 and sequence length of 1024 on a single H100 GPU. Fine-tuning models without low-rank adapters took 12 hours for 125M parameter models and over 36 days for 13B parameter models. Given these high costs, completing fine-tuning was challenging with our limited resources. In contrast, using low-rank adapters and freezing the sparse quantized weights made fine-tuning more eficient, enabling us to report accuracy results in Table 7.1.

Table E.12: The required time for fine-tuning the models with a single H100 GPU on 300,000 tokens from the C4 dataset with a batch size of 64 and a sequence length of 1024.
<table><tr><td rowspan="2">Pruning Method</td><td rowspan="2">Weight Quantization</td><td colspan="5">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Magnitude SparseGPT</td><td>Group AbsMax OPTQ</td><td>12h</td><td>43h</td><td>164h</td><td>361h</td><td>866h</td><td>867h</td><td>842h</td><td>844h</td></tr><tr><td>Wanda</td><td>Group AbsMax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SLiM-Naive</td><td>SLıM-Quant</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>3h</td><td>6h</td><td>8h</td><td>16h</td><td>18h</td><td></td><td></td></tr><tr><td>SLiM-LoRA</td><td>SLıM-Quant</td><td>1.5h</td><td></td><td></td><td></td><td></td><td></td><td>14h</td><td>14h</td></tr></table>

## E.9 Memory reduction analysis

SLIM prunes and quantizes the models and adds additional low-rank adapters to them. Additionally, it supports quantization methods for the low-rank adapters to reduce their overheads. In the following, we provide an analysis of the memory reduction when using SLIM and other pruning and quantization methods.

Assuming the hidden dimension of a model is d and the low-rank adapter ratio used in the model is of rank $r < 1$ . Furthermore, by denoting the number of transformer blocks with n and the vocabulary size of the model by V and by denoting the ratio of the up-projection and down-projection layers in the model by a, we can get the memory reduction as the ratio of $\frac { \mathrm { C o m p r e s s e d M o d e l S i z e } } { \mathrm { D e n s e M o d e l S i z e } }$ from Equation E.1.

$$
{ \mathrm { M e m o r y ~ R e d u c t i o n } } = { \frac { n ( 4 d ^ { 2 } / 2 + 4 \times 2 d ^ { 2 } r + 2 d ^ { 2 } a / 2 + 2 d ( d r + d r a ) ) + d V } { n ( 4 d ^ { 2 } + 2 d ^ { 2 } a ) + d V } }\tag{E.1}
$$

Table E.13 summarizes the memory reduction of diferent pruning and quantization methods. Please note that when using low-rank adapters (in Naive-LoRA and SLIM-LoRA), we assume a rank of $r = 0 . 1$

Table E.13: Theoretical memory reduction (×) of diferent compression methods across various OPT and LLaMA models. In Quantized SLIM , the low-rank adapters are also quantized.(↓ indicates better performance.)
<table><tr><td rowspan="2">Compression Method</td><td colspan="5">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>SparseGPT + OPTQ</td><td>0.40</td><td>0.30</td><td>0.25</td><td>0.17</td><td>0.15</td><td>0.14</td><td>0.15</td><td>0.14</td></tr><tr><td>Wanda + AbsMax</td><td>0.40</td><td>0.30</td><td>0.25</td><td>0.17</td><td>0.15</td><td>0.14</td><td>0.15</td><td>0.14</td></tr><tr><td>Naive-LoRA + AbsMax</td><td>0.50</td><td>0.42</td><td>0.38</td><td>0.31</td><td>0.30</td><td>0.29</td><td>0.31</td><td>0.30</td></tr><tr><td>SLiM-LoRA + SLiM-Quant</td><td>0.50</td><td>0.42</td><td>0.38</td><td>0.31</td><td>0.30</td><td>0.29</td><td>0.31</td><td>0.30</td></tr><tr><td>SLiM-LoRAQ + SLiM-Quant</td><td>0.42</td><td>0.33</td><td>0.28</td><td>0.20</td><td>0.19</td><td>0.18</td><td>0.19</td><td>0.18</td></tr></table>

## E.10 Computation reduction analysis

SLIM and other compression methods reduce the number of floating point operations (FLOPs) at the inference of models. Additionally, the low-rank adapters used in SLIM and Wanda SVD can add additional computational overheads to the inference of the models. Following JSQ [49], in this section, we provide an analysis of the FLOP reduction in the inference of diferent methods. It is noteworthy that even though quantization can reduce the memory overhead of models, since all the computations are done in floating point format, it does not lead to a reduction in the computation of the inference.

Assuming the hidden dimension of a model is d and the low-rank adapter ratio used in the model is of rank r < 1. Furthermore, by denoting the number of transformer blocks with n and the vocabulary size of the model by V and by denoting the ratio of the up-projection and down-projection layers in the model by a, we can get the FLOP reduction as the ratio of Dense Inference FLOP Count from Equation E.2, where b is the batch Compressed Inference FLOP Count size, and is canceled in the numerator and the denominator of the equation.

$$
\mathrm { F L O P ~ R e d u c t i o n } = { \frac { n ( 4 b d ^ { 2 } + 2 b d ^ { 2 } a ) + b d V } { n ( 4 b d ^ { 2 } / 2 + 4 \times 2 b d ^ { 2 } r + 2 b d ^ { 2 } a / 2 + 2 b ( d ^ { 2 } r + d ^ { 2 } r a ) ) + b d V } }\tag{E.2}
$$

Table E.14 summarizes the FLOP reduction of diferent compression methods. As can be seen, the overhead of adding the low-rank adapters (r = 0.1) in SLIM-LoRA and Naive-LoRA is not significant.

Table E.14: Compute (FLOP) reduction ratios (×) of diferent compression methods across various OPT and LLaMA models. In Quantized SLIM , the low-rank adapters are also quantized. (↑ indicates better performance.)
<table><tr><td rowspan="2">Compression Method</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>SparseGPT + OPTQ</td><td>1.52</td><td>1.66</td><td>1.75</td><td>1.91</td><td>1.94</td><td>1.96</td><td>1.95</td><td>1.97</td></tr><tr><td>Wanda + AbsMax</td><td>1.52</td><td>1.66</td><td>1.75</td><td>1.91</td><td>1.94</td><td>1.96</td><td>1.95</td><td>1.97</td></tr><tr><td>Naive-LoRA + AbsMax</td><td>1.32</td><td>1.39</td><td>1.43</td><td>1.50</td><td>1.51</td><td>1.52</td><td>1.49</td><td>1.49</td></tr><tr><td>SLiM-LoRA + SLiM-Quant</td><td>1.32</td><td>1.39</td><td>1.43</td><td>1.50</td><td>1.51</td><td>1.52</td><td>1.49</td><td>1.49</td></tr><tr><td>SLiM-LoRAQ + SLiM-Quant</td><td>1.32</td><td>1.39</td><td>1.43</td><td>1.50</td><td>1.51</td><td>1.52</td><td>1.49</td><td>1.49</td></tr></table>

## E.11 Compression costs

The computational cost of compression methods varies depending on their complexity. While all approaches can compress a single layer at a time, the memory usage is similar across methods, as each stores only one layer in the GPU’s global memory. Techniques like Wanda, which rely on matrix multiplication, are faster than more complex methods like SparseGPT, which computes the inverse Hessian matrix for each layer. Adding low-rank adapters to Wanda-SVD and SLIM increases computational complexity due to the need for singular value decomposition (SVD), making them comparable to SparseGPT in terms of computation.

Table E.15 summarizes the time required to compress various models using the discussed methods. Methods incorporating low-rank adapters (SLIM and Wanda-SVD) generally take longer to compress due to their higher complexity. Interestingly, SparseGPT’s compression time is comparable to methods with low-rank adapters, despite only performing pruning and quantization. The saliency-based approach in SLIM does not add significant overhead compared to Wanda-SVD, maintaining eficiency despite its added complexity.

Table E.15: The required compression time for diferent models and compression methods using a single H100 GPU.
<table><tr><td>Pruning</td><td colspan="3">Weight</td><td colspan="3">OPT</td><td colspan="3">LLaMA-2</td></tr><tr><td>Method</td><td>Quantization</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>Magnitude</td><td>AbsMax</td><td>1s</td><td>1s</td><td>1s</td><td>1s</td><td>2s</td><td>4s</td><td>2s</td><td>4s</td></tr><tr><td>SparseGPT</td><td>OPTQ</td><td>1m</td><td>2m</td><td>5m</td><td>11m</td><td>22m</td><td>41m</td><td>25m</td><td>46m</td></tr><tr><td>Wanda</td><td>SLıM-Quant</td><td>0.5m</td><td>1m</td><td>3m</td><td>5m</td><td>8m</td><td>13m</td><td>8m</td><td>14m</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Wanda-SVD</td><td>SLrM-Quant</td><td>1m</td><td>2m</td><td>7m</td><td>13m</td><td>33m</td><td>60m</td><td>38m</td><td>67m</td></tr><tr><td>SLIM</td><td>SLıM-Quant</td><td>1m</td><td>2m</td><td>7m</td><td>13m</td><td>34m</td><td>63m</td><td>39m</td><td>68m</td></tr></table>

## E.12 Rank analysis

The key hyperparameter in low-rank approximation is the rank of the adapters. While increasing the rank reduces approximation error, it also leads to higher computational and memory overhead. Therefore, it is crucial to analyze the trade-of between the accuracy improvements and the overhead introduced by the chosen approximation rank.

Assuming the rank of the low-rank adapter is rd, where $r < 1$ is a fixed factor and d is the dimension of the weights in a square feed-forward layer, the low-rank adapters are represented as $\mathcal { L } , \mathcal { R } ^ { T } \in \mathbb { R } ^ { d \times r d }$ , resulting in a memory overhead of $\mathcal { O } ( 2 r d ^ { 2 } )$ for storing them. To compute X LR, where $\boldsymbol { \mathcal { X } } \in \mathbb { R } ^ { b \times d }$ is the input with a batch size of $b ,$ the computational complexity is $\mathcal { O } ( 2 b r d ^ { 2 } )$ . Given that the original memory and computational complexity of the layer are $\mathcal { O } ( d ^ { 2 } )$ and $\mathcal { O } ( b d ^ { 2 } )$ , respectively, the overhead introduced by the low-rank adapters becomes negligible when $r \ll 1$

Figure E.2-a shows the average zero-shot accuracy of the OPT-6.7B and LLaMA-2-7B models for various ranks. As expected, increasing the rank leads to improved model accuracy. Based on these results, a rank of $r = 0 .$ 1 provides a substantial boost in accuracy without introducing significant overhead to inference.

## E.13 Efects of calibration sample count

Similar to previous work (SparseGPT, Wanda, AWQ, OmniQuant, and AfineQuant), SLIM leverages a set of calibration data from the C4 dataset to assess weight saliency for pruning and low-rank approximations. Figure E.2-b illustrates the perplexity of LLaMA-2-7B using varying numbers of calibration samples. As shown, SLIM demonstrates low sensitivity to the number of calibration samples, making it efective even in scenarios with limited data.

## E.14 Sensitivity to calibration dataset

Similar to other pruning and quantization methods such as Wanda, SparseGPT, OPTQ, and AWQ, SLIM relies on a calibration dataset to evaluate weight saliency. The C4 [121] and SlimPajama [135] datasets are among the most commonly used calibration sets for LLM compression. Table E.16 presents the perplexity results for

![](images/2d7a90a5e1c56864d7cedd0e9868dfaded084188a1134e6c895d83482e50877e.jpg)  
(a)

![](images/badf55a688cc952a7e7476d14e913e1df983d84f3ab082a928532dd02d0fe7ff.jpg)  
(b)  
Figure E.2: Sensitivity analysis for the rank of the adapter (a) and the number of calibration samples (b) for diferent one-shot compression methods. For Naive-LoRA and SLIM-LoRA, we have used the SLIM-Quant quantization method, and for the SparseGPT, we have used the Group quantization version of OPTQ.

SLIM-LoRA and SLIM-Quant across diferent calibration datasets. The results indicate that SLIM is largely insensitive to the choice of dataset, achieving comparable accuracy regardless of the calibration dataset used.

Table E.16: Perplexity of diferent models on WikiText-2 dataset using SLIM-LoRA with 4-bit quantization using SLIM-Quant with diferent calibration datasets. ↓ indicates better performance.
<table><tr><td>Calibration</td><td colspan="6">OPT</td><td colspan="2">LLaMA-2</td></tr><tr><td>Dataset</td><td>125M</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>7B</td><td>13B</td></tr><tr><td>50% 2:4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>C4</td><td>57.91</td><td>50.09</td><td>19.64</td><td>15.65</td><td>12.71</td><td>12.13</td><td>7.56</td><td>6.50</td></tr><tr><td>SlimPajama</td><td>46.27</td><td>44.77</td><td>19.35</td><td>16.04</td><td>12.56</td><td>12.32</td><td>7.15</td><td>6.49</td></tr><tr><td>50% Unstructured</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>C4</td><td>39.62</td><td>31.51</td><td>16.52</td><td>13.65</td><td>11.42</td><td>10.82</td><td>6.16</td><td>5.36</td></tr><tr><td>SlimPajama</td><td>36.49</td><td>29.94</td><td>16.64</td><td>14.08</td><td>11.61</td><td>11.02</td><td>5.99</td><td>5.34</td></tr></table>

## E.15 Sparsity analysis

To analyze the impact of sparsity on model accuracy, we conduct experiments on LLaMA-2-13B with 4-bit quantization, pruning it to varying sparsity ratios. Figure E.3 presents the perplexity results for SLIM-LoRA with SLIM-Quant , SparseGPT with OPTQ, and Wanda with Group AbsMax. As expected, increasing the sparsity ratio leads to higher perplexity, indicating a trade-of between compression and accuracy. Notably, SLIM-LoRA combined with SLIM-Quant maintains competitive accuracy up to 60% sparsity, whereas other methods experience noticeable degradation at lower sparsity levels.

![](images/1d28153db23d7aa89830a941287a5897ed4a7e2e35dfb5d8ed2735cd37b943fd.jpg)  
Figure E.3: Sparsity analysis on LLaMA-2-13B model using perplexity on WikiText-2 dataset. ↓ indicates better performance.

## E.16 Group quantization challenges

Group quantization allows sharing the same quantization parameters for a small group of the elements in the quantized matrix, leading to smaller error. But, using group quantization adds additional challenges to the training and inference of the model, e.g. more complicated implementation and additional memory and compute overheads.

The state-of-the-art group quantization GPU kernel, dense and sparse Marlin [38], consists of thousands of lines of CUDA code optimized for only a limited number of GPU architectures, showcasing the amount of efort needed to implement a version of group quantization. Furthermore, other libraries and frameworks, such as Triton [143] and CUTLASS [23] do not provide support for 4-bit group quantization, limiting its flexibility and possibility of modification.

Furthermore, using group quantization can lead to an additional overhead during matrix multiplication, since more parameters need to be loaded for dequantizing each group. As an example, Table E.17 shows the slow-down of using group quantization on the down-projection matrices in diferent LLaMA-2 and LLaMA-3.1 models on a NVIDIA A100-40GB GPU, with a batch size of 16.

Table E.17: Group quantization slow-down (×) on diferent LLaMA-2 and LLaMA-3.1 models. ↓ indicates worse.
<table><tr><td>Model</td><td>LLaMA-2-7B</td><td>LLaMA-2-13B</td><td>LLaMA-2-70B</td><td>LLaMA-3.1-405B</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Slow-Down (×)</td><td>0.94</td><td>0.95</td><td>0.95</td><td>0.94</td></tr></table>