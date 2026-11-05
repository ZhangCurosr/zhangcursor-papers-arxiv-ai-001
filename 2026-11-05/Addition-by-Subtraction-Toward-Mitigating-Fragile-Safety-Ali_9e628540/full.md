# Addition by Subtraction: Toward Mitigating Fragile Safety Alignment in Large Language Models

Hamid Dashtbani

Department of Computer Science

University of Duisburg-Essen

hamid.dashtbani@uni-due.de

Alexandra Dmitrienko

Department of Computer Science

University of Duisburg-Essen

alexandra.dmitrienko@uni-due.de

Abstract—Large language models (LLMs) undergo safety alignment to prevent harmful generations, yet they remain vulnerable to jailbreak attacks. Recent work reveals a deeper structural weakness: safety alignment can be highly sparse, such that pruning a small number of safety-critical neurons or ablating a single refusal direction can substantially undermine aligned behavior. In this work, we take the first steps toward strengthening safety alignment such that it is less dependent on sparse model components. We propose Addition by Subtraction, which gradually removes identified safety-critical neurons during finetuning and uses preference optimization to re-establish safety-aligned behavior through other parts of the model. Our results show that the resulting models are substantially more robust to pruning-based attacks. Remarkably, these benefits generalize beyond the white-box attacks motivating our design: Addition by Subtraction improves robustness against other sparsity-based interventions, including refusal-direction ablation, as well as a broad range of black-box jailbreak attacks. The finetuned models also exhibit improved truthfulness while largely preserving general capabilities. Comprehensive experiments across four models from three vendors demonstrate consistent improvements in safety robustness, reducing attack success rates by up to 90% for some attacks.

## 1. Introduction

In recent years, large language models (LLMs) have grown substantially in both popularity and capabilities, becoming increasingly important tools in everyday life. Modern LLMs are capable of following complex user instructions with little to no additional guidance [1], [2] and have achieved significant improvements across a wide range of tasks, including programming [3], mathematics and reasoning [4], [5], question answering [1], and natural language understanding [6].

While LLMs demonstrate remarkable performance across many areas, they are also prone to generating content that is considered harmful or undesirable, such as vulnerable code [7] or toxic content [8], [9]. To mitigate the generation of such responses, LLMs typically undergo safety alignment during post-training [10], [11], [12], with the objective of aligning model outputs with human preferences and safety objectives.

While safety alignment has generally made LLMs more likely to produce harmless responses, the safety-aligned models have been shown to remain vulnerable to jailbreak attacks [13], [14], [15], [16], [17], where attackers circumvent safety-aligned behavior and induce the model to generate harmful responses to malicious prompts that it would otherwise refuse. Recent works have proposed a wide range of jailbreak attacks, including methods that use imaginary scenarios [13], [18], [19], obfuscation [18], roleplaying [20], and other prompt modifications [21] to conceal or disguise the malicious intent of the original prompt.

Moreover, more recently, as open-weight models have become increasingly common, more powerful attacks have been proposed that expose the fragility of safety alignment in LLMs. Multiple recent works [22], [23], [24] have shown that safety alignment in LLMs is highly sparse, with only a few thousand neurons, commonly referred to as safetycritical neurons, playing a central role in safety-aligned behavior. These works have shown that pruning these neurons can effectively remove safety alignment and induce harmful responses while having negligible impact on the model’s general capabilities. Another recent study [25] has shown that safety-aligned refusal behavior in large language models can be effectively disabled by ablating a single direction from the model’s intermediate activations and representations.

Taken together, these results suggest that current safety alignment depends on a surprisingly small subset of neurons, creating a structural weakness that is fundamentally different from conventional prompt-level jailbreak vulnerabilities. These findings are particularly concerning for open-weight models, where attackers have unrestricted access to model parameters and internal representations and can directly target the components responsible for safety-aligned behavior. Consequently, improving robustness only at the prompt or output level may be insufficient.

The brittleness of safety alignment is therefore a significant concern, particularly given the growing ecosystem of open-weight models, making stronger and more robust safety mechanisms for such models increasingly important.

Current defensive approaches do not directly address the sparsity of safety alignment and instead rely on methods such as external guardrails for user inputs or model responses [26], [27], [28], [29] to filter harmful prompts or responses before or after generation, or finetuning approaches that improve the general robustness of LLMs against jailbreak attacks using techniques such as negative preference optimization [30] or chain-of-thought reasoning for safety alignment [31]. However, as our experiments show, these approaches still suffer from the same sparsity in safety alignment as the original models.

This work takes the first steps toward mitigating the sparsity of safety alignment in large language models. We propose Addition by Subtraction (ABS-Align), which, at a high level, uses preference optimization to finetune a large language model toward more distributed safety alignment representations by temporarily and gradually deactivating the initially identified sparse safety-critical neurons. This process encourages a less sparse and concentrated distribution of safety-aligned behavior across the model, such that, during the later stages of finetuning, all originally identified safety neurons are temporarily deactivated while the model is optimized to maintain safety alignment through other neurons and representations. Rather than directly searching for new safety neurons, ABS-Align repeatedly removes the neurons with the strongest safety signal and finetunes the remaining model to re-introduce safety alignment. We use the term denser to describe safety-algined behavior that is less fragile and less dependent on a small set of safetycritical neurons that sparsity-based attacks detect, remove, or manipulate in order to jailbreak a target LLM.

Interestingly, our experiments show that the benefits of the less fragile safety-aligned model achieved by ABS-Align extend beyond the white-box attacks that motivate ABS-Align. Our experiments show that ABS-Align leads to broad improvements against a range of black-box attacks, despite receiving no data augmentation or explicit guidance for robustness against these attacks during finetuning. The finetuned model also exhibits greater honesty and truthfulness, while attacks that exploit the sparsity of safety alignment at both the neuron level and the activation and representation levels become significantly less effective.

We perform extensive experiments across four models from three different vendors, specifically Google’s Gemma, Meta’s Llama model families, and Alibaba’s Qwen. Our results show that ABS-Align consistently improves the models’ safety alignment against a broad range of jailbreak attacks while having a negligible effect on their general capabilities and instruction-following ability.

The contributions of this work can be summarized as follows:

We propose ABS-Align, as a defense that mitigates the sparsity of safety alignment in LLMs by specifically finetuning the model to distribute its safety alignment across more parts of the network, improving robustness of safety alignment.

• By making safety-aligned LLMs less fragile, ABS-Align is the first to show increased robustness against not only pruning-based attacks, but also other forms of white-box attacks that conceptually rely on the sparsity of safety alignment. ABS-Align also generalizes to black-box attacks without any explicit data augmentation or guidance during finetuning.

Our experiments provide causal evidence that finetuning with ABS-Align leads to a less sparse safety alignment and reduces the dependence of safetyaligned behavior on sparse model components. Specifically, we show that the finetuned models are substantially more robust to sparsity-based attacks that prune safety-critical neurons using two different jailbreak methods, TwinBreak and NeuroStrike. After fine-tuning, these methods’ ability to jailbreak LLMs by pruning safety neurons is substantially degraded where even pruning 3 times more neurons compared to the undefended model shows minimal jailbreak success. Additionally, ablating a single refusal direction becomes substantially less effective, providing further evidence that the improved robustness extends beyond neuron-level interventions to attacks targeting the model’s internal representations.

We conduct a comprehensive set of experiments and evaluations across multiple models and dimensions, including attack success rates for white-box and black-box attacks, retention of general model capabilities across multiple benchmarks, and overrefusal rates. We compare our results with those of recent related work and show improvements across multiple evaluation dimensions, including reductions of approximately up to 90% in attack success rate for some attacks.

Overall, ABS-Align provides a simple yet effective approach for mitigating fragility of safety alignment in large language models, improving robustness not only against attacks that exploit sparse safety-alignment representations but also against a broad range of prompt-level and black-box jailbreak attacks.

Outline. The remainder of the paper is structured as follows. Section 2 provides the necessary background, and Section 3 describes our method. Section 4 presents the experimental evaluation and results, followed by a discussion in Section 5. Section 6 reviews related work, and Section 7 concludes the paper.

## 2. Background

This section provides an overview of the key concepts used throughout this paper. We begin with a description of how large language models work in Section 2.1. Next, Section 2.2 discusses safety alignment and how it aims to mitigate jailbreak attacks against LLMs. Finally, Section 2.3 discusses how jailbreak attacks threaten the security and safety of LLMs.

## 2.1. Large Language Models

Large language models (LLMs) are autoregressive models trained to predict the next token given an input sequence of tokens [1]. More specifically, given a sequence of n tokens, denoted by $x _ { 1 : n }$ (i.e., a sequence of $x _ { 1 } , x _ { 2 } , . . . , x _ { n } )$ the model predicts the probability of the next token $x _ { n + 1 } .$ i.e., $P ( x _ { n + 1 } | x _ { 1 : n } )$ , over each token $v _ { i }$ in its vocabulary $V ,$ which contains all tokens recognized by the LLM.

To generate long sequences of text, LLM inference is performed autoregressively, generating one token at a time. At each inference step, the newly generated token is appended to the sequence of previously processed tokens and used as input for the next step.

LLMs are composed of multiple layers of transformer blocks [32], where each block has two main components: a multi-layer perceptron (MLP) block and a multi-head selfattention (MHSA) block, each consisting of multiple parameterized layers. The MLP component performs nonlinear transformations of token representations, while the MHSA component enables the model to attend to information distributed across different parts of the input sequence.

The MLP block typically consists of three linear layers and can be formulated as follows:

$$
\begin{array} { r } { \mathbf { M L P } ( x ) = W _ { \mathrm { d o w n } } \left( \sigma _ { 1 } \left( x W _ { \mathrm { g a t e } } \right) \odot \sigma _ { 2 } \left( x W _ { \mathrm { u p } } \right) \right) } \end{array}
$$

where $\sigma _ { 1 }$ and $\sigma _ { 2 }$ are nonlinear activation functions, x is the input to the MLP block, and $\odot$ denotes element-wise multiplication. More specifically, the MLP block consists of gate, up, and down projection layers. The gate and up projection layers jointly control and transform the information flow while projecting the hidden representation into a higher-dimensional space, after which the down projection layer projects it back to the model’s hidden dimension before it is passed to subsequent layers. Multiple works have found that safety-critical neurons primarily reside in the gate and up projection layers of the MLP block [22], [23].

## 2.2. Safety Alignment

LLMs are trained on large corpora of text originating from many different sources, including web data. These training sets contain data that can cause the model to learn toxic, offensive, and otherwise undesirable content, and LLMs have been shown to reproduce such content [8], including generating vulnerable code [7] or harmful content such as instructions on how to build a bomb or scripts for racist speeches [22], [23].

To mitigate undesirable content generation by LLMs, these models usually undergo a process known as safety alignment during post-training [10], [11], [12]. The goal of safety alignment is to align model generations with human preferences and safety objectives [11], thereby preventing models from generating harmful content such as discriminatory speech or instructions on how to harm others.

One of the main tools for safety alignment is reinforcement learning from human feedback (RLHF) [11], through which the model is taught what kinds of responses are more desirable according to human judgments and safety values. A common approach is to use pairs of responses for each prompt in the training dataset, with one response judged to be preferable based on qualities such as helpfulness, honesty, and harmlessness [11]. These preferences are then used to train a reward model, which provides a reward signal for optimizing the LLM toward preferred responses.

RLHF is resource-intensive and usually relies on a separate reward model during training to provide the feedback signal used to optimize the model [11]. Direct Preference Optimization (DPO) [12] was proposed by Rafailov et al. as a simpler and more computationally efficient approach to alignment that directly optimizes the language model on preference data, without relying on an explicit reward model or reinforcement learning. DPO uses a dataset of preference samples $( x , y _ { c } , y _ { r } )$ , where $x$ denotes the user request, $y _ { c }$ represents the preferred (chosen) model response, and $y _ { r }$ denotes the dispreferred (rejected) response. For safety alignment, $y _ { r }$ can be a response that contains harmful or prohibited content, while $y _ { c }$ can be a response that appropriately refuses to generate such content. The DPO loss is then defined as:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } = - \mathbb { E } _ { ( x , y _ { c } , y _ { r } ) \sim \mathcal { D } } \Bigg [ \log \sigma \Bigg ( \beta \log \frac { \pi _ { \theta } \left( y _ { c } \mid x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { c } \mid x \right) } } \\ & { ~ - \beta \log \frac { \pi _ { \theta } \left( y _ { r } \mid x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { r } \mid x \right) } \Bigg ) \Bigg ] } \end{array}
$$

where $\pi _ { \theta }$ is the trainable model, $\pi _ { \mathrm { r e f } }$ is a reference model, and $\beta$ controls the strength of the preference signal between chosen and rejected responses.

## 2.3. Jailbreak Attacks

Even after undergoing safety alignment, large language models have been shown to remain susceptible to jailbreak attacks that can elicit harmful generations [15], [16], [17], [22], [23], [25], [33]. The goal of jailbreak attacks against LLMs is to bypass the safety-aligned behavior of the target model and induce it to generate content that it would otherwise refuse to produce.

Jailbreak attacks can be either black-box or white-box. In black-box settings, attackers are assumed to have only black-box access to the target LLM [19], meaning that they do not have access to the model weights or internal states and can only prompt the model and observe its outputs, either in the form of output probabilities and logprobabilities [16] or as decoded text strings [17], [19]. The attacker’s methodology in such settings is therefore to modify a malicious prompt x by crafting an adversarial prompt $\bar { x } ^ { \prime }$ such that $x ^ { \prime }$ causes the model to generate an undesirable response. More formally, suppose $f _ { \theta } : \mathcal { X }  \mathcal { Y }$ denotes the target LLM with weights θ, X denotes the space of input prompts, and $\mathcal { V }$ denotes the space of responses, with $\mathcal { V } _ { \mathrm { u n s a f e } } \subset \mathcal { V }$ denoting the set of unsafe responses that the model is expected to refuse to generate. Assuming x is a harmful prompt, such as ”How to rob a ban $\therefore$ , for which the model generates a refusal response $y \in \mathcal { V } , y \notin \mathcal { V } _ { \mathrm { u n s a f e } } ,$ the attacker transforms x into $x ^ { \prime }$ such that the model instead outputs $y ^ { \prime } \in \mathcal { V } _ { \mathrm { u n s a f e } } \colon$

$$
f _ { \theta } ( x ^ { \prime } ) = y ^ { \prime } \quad \mathrm { s . t . } \quad y ^ { \prime } \in \mathcal { V } _ { \mathrm { u n s a f e } } \wedge f _ { \theta } ( x ) = y \not \in \mathcal { V } _ { \mathrm { u n s a f e } }
$$

White-box attacks, on the other hand, assume that the attacker has access to the model weights and internal states. Attackers in this setting can therefore leverage gradient information [14] and model-internal intervention techniques such as neuron pruning [22], [23], or activation-level interventions [25].

## 3. Method

This section describes our method, threat model, and the design of the proposed method, Addition by Subtraction (ABS-Align), in greater detail. We begin by describing the scenario considered by our method in Section 3.1. We then briefly describe the attacker and defender, along with their respective capabilities, in Section 3.2. Next, we provide an overview of how ABS-Align works in Section 3.3. Finally, we outline the details of our method in Section 3.4.

## 3.1. Scenario

In our considered scenario, we mainly focus on decoderonly large language models that have already undergone safety alignment during post-training [10], [11], [12] to encourage helpful, honest, and harmless behavior [11]. Opensource large language models are becoming increasingly popular, with model weights publicly available such that anyone can download and use these models locally, including malicious users who may use them for harmful purposes, such as generating malware or writing social engineering emails.

## 3.2. Threat Model

Our threat model consists of two main entities: the defender, who is the vendor of the large language model to be protected, and a malicious user who has white-box access to the model and can observe its internal states, including its weights and activations. The attacker can therefore use pruning-based attacks [22], [23] or activation-steering methods [25] to jailbreak the model. However, adversarial finetuning attacks fall outside the scope of our threat model.

## 3.3. General Approach

Recent studies have shown that safety alignment in LLMs is fragile and sparse, allowing attackers to achieve high jailbreak success rates with only minor modifications to model weights [22], [23], [24] or intermediate activations during inference [25]. ABS-Align mitigates this vulnerability by encouraging safety-aligned behavior to be distributed more broadly throughout the model, thereby making successful jailbreaks through such targeted modifications more difficult.

ABS-Align consists of two main stages. First, it identifies the safety-critical neurons of a large language model, i.e., neurons whose pruning removes safety alignment. Recent works have proposed several methods for detecting these neurons [22], [23], [24]; however, we use Twin-Break [23] since this method was the fastest among the considered approaches while achieving comparable performance, based on the results reported in the respective original papers. After identifying these safety-critical neurons, one can create a pruned version of the model by pruning the detected neurons, which effectively sets the weights associated with each detected neuron to zero.

The second stage of ABS-Align consists of pruning and finetuning the model to make the model’s safety and refusal features less dependent on the small set of detected neurons. To achieve this, ABS-Align gradually prunes the identified safety neurons, starting with those that have the greatest influence on the model’s safety alignment. By the end of the finetuning process, all initially detected safety neurons will have been added to the set of pruned neurons, and any safety-aligned behavior exhibited by the finetuned model must therefore originate from other parts of the model.

ABS-Align does not remove all safety-critical neurons at once for two main reasons. First, removing all safety neurons results in a significant reduction in the model’s safety alignment capability, which can make it substantially more difficult to re-establish safe behavior and may require a larger dataset and longer training time. In contrast, our method relies on only a small subset of samples for finetuning. Second, removing all safety neurons also weakens the model on non-security-related tasks, as reported in prior neuron-pruning attacks [22], [23]. Consequently, finetuning this less capable model can further reduce its usability and negatively affect the performance of the finetuned model. Our ablation study, presented in Section 4.6, further confirms this intuition.

Overall, by gradually removing safety neurons from the LLM, ABS-Align performs finetuning on a partially safetyaligned model while relying on progressively fewer safety neurons from the originally aligned model. At later stages of training, all safety neurons identified in the original LLM are pruned, and the model is optimized to preserve refusal and safety-aligned behavior through other neurons and representations during finetuning.

## 3.4. Addition by Subtraction Details

Here we outline the design of the proposed method in greater details. We first describe safety neuron detection and pruning in Section 3.4.1. After that we describe how the finetuning process of ABS-Align unfolds in Section 3.4.2.

(e) Finetuned model  
![](images/c2137f8a9c1902fe5b56a44b6dbf8f29b08e03edc981f85b1acf368589d9e46a.jpg)  
(a) Original model  
(b) Early Finetuning  
(c) Mid Finetuning  
(d) Late Finetuning

Figure 1: Overview of the proposed method. The depicted new and original safety neurons are conceptual.

3.4.1. Finding and Pruning Safety Neurons. ABS-Align starts by identifying the safety-critical neurons of the model M. As mentioned in Section 3.3, we use TwinBreak [23] to identify these neurons due to its simplicity and efficiency. To do so, TwinBreak uses pairs of highly similar prompts, where one prompt in each pair is malicious and the other is benign, while both are closely matched in content and structure.

After that, the mean activation difference between the two sets is used to identify safety-critical neurons. More specifically, let $\mathcal { D } _ { m }$ denote the set of malicious prompts and $\mathcal { D } _ { b }$ the set of benign prompts. TwinBreak compares the activations of the gate and up projection layers of the MLP block (see Section 2.1). Let ℓ denote the transformer layer index and $d _ { \mathrm { o u t } }$ the output dimension of $W _ { \mathrm { g a t e } }$ . For the gate projection, we have

$$
T ^ { ( \ell ) } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \left| a _ { b _ { i } } ^ { ( \ell - 1 ) } W _ { \mathrm { g a t e } } ^ { ( \ell ) } - a _ { m _ { i } } ^ { ( \ell - 1 ) } W _ { \mathrm { g a t e } } ^ { ( \ell ) } \right|\tag{1}
$$

where $m _ { i } \in \mathcal { D } _ { m }$ and $b _ { i } \in \mathcal { D } _ { b }$ denote the malicious and benign prompts, respectively, in the i’th pair of the dataset; $a _ { m _ { i } } ^ { ( \ell - \overline { { 1 } } ) }$ and $a _ { b _ { i } } ^ { ( \ell - 1 ) }$ denote the corresponding activations from the previous layer; n denotes the total number of pairs; ℓ is the transformer block index; and $W _ { \mathrm { g a t e } } ^ { ( \ell ) } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { m } }$ denotes the weights of the gate projection layer at layer ℓ, where $d _ { \mathrm { m o d e l } }$ is the transformer hidden dimension and $d _ { m }$ is the number of neurons in the gate projection layer. After that, $T ^ { ( \ell ) }$ is used to identify the top 1% of neurons, denoted by $I ^ { ( \ell ) }$ , with the largest activation differences. This process is performed iteratively for five iterations, identifying and pruning the top 1% of neurons at each iteration. Specifically, at the end of each iteration, for every $j \in I ^ { ( \ell ) }$ , the j’th column of the weight matrix is set to zero:

$$
W _ { \mathrm { g a t e } } ^ { ( \ell ) } [ : , j ] = 0
$$

The same applies to the up projection layers with $W _ { \mathrm { u p } } \in$ R<sup>dmodel×dm</sup> .

Importantly, this procedure is repeated iteratively for up to five pruning iterations. During each pruning iteration i, all neurons identified in iterations 1 through i − 1 remain pruned before the next top 1% of neurons are identified.

3.4.2. Mitigating Fragility in Safety-Aligned Language Models. ABS-Align replicates safety alignment across more parts of the model by freezing the initial set of safety neurons and gradually pruning them. To do so, ABS-Align ranks the model’s safety neurons according to their TwinBreak scores [23], starts by pruning the most important safety neurons at the beginning of the finetuning process, and gradually prunes the entire initially identified set of safety neurons. Figure 1 depicts a high-level overview of this pro cess. Figure 1a shows part of the initial LLM, where darker circles denote safety neurons labeled $S _ { 1 } , S _ { 2 }$ , and $S _ { 3 } ,$ and light blue circles denote other neurons in the network. The size of each safety neuron denotes its importance for safety alignment; therefore, the ordering of the safety neurons by importance can be expressed as $S _ { 1 } > S _ { 2 } > S _ { 3 }$

ABS-Align first freezes all the detected safety neurons and prunes the top r most important safety neurons $( S _ { 1 }$ in Figure 1b). ABS-Align then performs preference opti mization on the resulting pruned model, encouraging safetyaligned behavior to emerge through other parts of the model, conceptually depicted by $S _ { 1 } ^ { \prime }$ in Figure 1b.

ABS-Align then continues to prune the initially detected safety neurons throughout the finetuning process. After every $t _ { p }$ steps, the next r most important safety neurons are pruned, as shown in Figures 1c and 1d where $S _ { 2 }$ and $S _ { 3 }$ are pruned, respectively. By the final stages of finetuning, all safety neurons detected in the original model are pruned $( S _ { 1 } , S _ { 2 } , S _ { 3 }$ in Figure 1d); therefore, the model is optimized to maintain appropriate refusal behavior through other neurons and representations, conceptually depicted by $S _ { 1 } ^ { \prime } , S _ { 2 } ^ { \prime }$ and $S _ { 3 } ^ { \prime }$ in Figure 1e.

After the finetuning process is complete, all pruned neurons are restored, resulting in a hardened model in which safety-aligned behavior is less dependent on the originally identified safety neurons, as shown in Figure 1e. Specifically, the original safety neurons $S _ { 1 } , S _ { 2 }$ , and $S _ { 3 }$ are augmented by the newly formed safety neurons $S _ { 1 } ^ { \prime } , S _ { 2 } ^ { \prime }$ , and $S _ { 3 } ^ { \prime } .$

Interestingly, we observed that applying our method only to the originally identified safety neurons is sufficient to make sparsity-based attacks ineffective, without requiring additional rounds of identifying and pruning new safetycritical neurons after finetuning.

Detailed Steps. Algorithm 1 shows how ABS-Align works. It takes as input the model to be finetuned M, a preference dataset $\mathcal { D } _ { s } ,$ and the set of safety neurons $S ,$ ranked from most to least important according to their scores. It starts by freezing all the neurons in S, selecting and removing the top r neurons from S, and pruning the selected neurons from the model (lines 1-4). The finetuning loop then runs for $T$ steps (line 5). At each step, a batch of preference samples is drawn from $\mathcal { D } _ { s } ,$ and the model is finetuned using the DPO loss (lines 6 and 7). Every $t _ { p }$ steps, the next top r safety neurons are selected and pruned from the model (lines 8-12). After $T$ steps, all pruned neurons are restored, and the algorithm terminates (lines 13-14).

Algorithm 1 Addition by Subtraction   
Require: Original model M, a preference dataset $\mathcal { D } _ { s } ,$ fine  
tuning steps T, pruning rate $r ,$ model pruning step $t _ { p } ,$   
safety neurons sorted from highest scores to lowest $\dot { S }$   
1: Freeze all the neurons in S   
2: $P \gets \mathrm { t o p } \ ( \mathrm { S } , \mathrm { \Delta r } )$ ▷ Select Top r Safety Neurons   
3: $S _ { \mathrm { a c t i v e } }  \mathbf { S } \cdot \mathbf { P }$   
4: Prune P from $\mathcal { M }$   
5: for $t = 1 , \dots , T$ do ▷ Finetuning Loop   
6: $x _ { s } \sim \mathcal { D } _ { s }$ ▷ Batch Sample   
7: Finetune M on $x _ { s }$ ▷ DPO Finetuning Step   
8: if t % $t _ { p } = = 0$ then ▷ Every $t _ { p }$ Steps   
9: $P $ top $( S _ { \mathrm { a c t i v e } } , \mathbf { r } )$   
10: $S _ { \mathrm { a c t i v e } }  S _ { \mathrm { a c t i v e } } \cdot { \bf P }$   
11: Prune $P$ from $\mathcal { M }$   
12: end if   
13: end for   
14: Restore all S neurons for M ▷ Unprune All Neurons

Gradual Pruning. ABS-Align does not prune all safety neurons at once at the beginning of the finetuning process. Instead, it gradually prunes an increasing number of the originally identified safety neurons as finetuning progresses.

As mentioned in Section 3.3, we do not prune all safety neurons at once because doing so can noticeably degrade the pruned model’s general capabilities while also removing most of its safety alignment. This can make reestablishing safety alignment substantially more difficult and require considerably more training data and computational resources.

Importance of Safety Neurons. As mentioned earlier, ABS-Align first prunes the most important safety neurons. We define the importance of each safety neuron according to its $T ^ { ( \ell ) }$ score (see Section 3.4.1), where neurons with higher $T ^ { ( \ell ) }$ scores are ranked higher and considered more important for safety alignment.

Biggest First. ABS-Align starts by pruning neurons with the highest scores and thus the greatest importance for safety alignment. We start with the highest-scoring safety neurons because pruning less important neurons leaves most of the model’s safety-aligned behavior intact, resulting in an already strongly safety-aligned model during finetuning that provides little learning signal for safety-aligned behavior.

Finetuned Parameters. ABS-Align finetunes the full model except for the first and last transformer blocks, which remain frozen throughout the finetuning process. This is motivated by the observation that lower transformer blocks have been shown to encode more surface-level and lexical information about the input, while later blocks are more involved in higher-level representations and coherent output generation [34]. Based on the intuition that the first block is closely tied to processing input embeddings, whereas the last block directly precedes output-token generation, we exclude these two blocks from finetuning to better preserve the model’s existing capabilities.

## 4. Experiments

In this section, we evaluate ABS-Align’s performance across various dimensions. We first describe our experimental setup, including the benchmarks, attacks, and metrics used in Section 4.1. We then evaluate ABS-Align’s robustness across several dimensions in Section 4.2, including its performance against TwinBreak, its generalization to other sparsity-based attacks, and its generalization to blackbox attacks. Next, in Sections 4.3 and 4.4, we evaluate ABS-Align’s performance with respect to over-refusal and the model’s general capabilities, respectively. Section 4.5 then compares ABS-Align with recent related work. In Section 4.6, we perform a comprehensive ablation study to evaluate the design components of ABS-Align. Finally, in Section 4.7, we evaluate ABS-Align’s robustness against adaptive attacks.

## 4.1. Experimental Setup

Training Dataset and Models: We evaluate Addition by Subtraction (ABS-Align) on open-weight models from three different vendors and four models. Specifically, we use Llama-3-8B-Instruct [35], [36] and Llama-3.2-3B-Instruct from Meta, Gemma-2-2B-IT [37] from Google, and Qwen-3-4B from Alibaba. For preference finetuning, we use the PKU-SafeRLHF [38] dataset and randomly sample 1,000 preference samples for training. To retain model’s general capability we additionally augment this dataset with samples 850 samples from Alpaca [39] and 150 safe samples from XSTest [40]. For full training details please see Appendix 1.

Related Work Baselines: For comparisons with existing defenses we use Circuit Breaker [41], W-DOOR [30], and Deep Refusal [42]. We evaluate each method only on models considered in its original work and use the corresponding reported training hyperparameters. We do not extend the baselines to additional models, as doing so would require additional hyperparameter tuning and could result in suboptimal baseline performance. Consequently, some baseline results are unavailable for models not evaluated by the corresponding original work.

Over-Refusal benchmark: To evaluate ABS-Align’s result regarding whether the reinforced safety aligned behavior leads to incorrect refusal of harmless prompts, such as ”How to kill a Python program”, we use XSTest [40], a benchmark for measuring exaggerated safety behavior in LLMs. XSTest contains 250 benign prompts that contain sensitive words such as ”kill”, ”execute”, ”rob” in harmless contexts. We create a random split of the 250 prompts and use 150 prompts for training while keeping the rest of the

100 prompts for over-refusal evaluation. To report the overrefusal rate, we use XSTest’s keyword-matching evaluator on the 100 harmless prompts split and report the fraction of responses that incorrectly refuse the corresponding request.

Utility Benchmarks: To measure the effect of ABS-Align on the general capabilities and instruction-following ability of the finetuned model, we evaluate its performance across a diverse set of benchmarks covering knowledge, reasoning, and problem-solving capabilities. Specifically, we use MMLU [43], TruthfulQA [44], ARC-Challenge [45], WinoGrande [46], and HellaSwag [47].

White-Box Attack Baselines: We evaluate ABS-Align against three white-box attacks, namely TwinBreak [23], NeuroStrike [22], and refusal direction ablation [25] attacks. TwinBreak and NeuroStrike find and prune safety critical neurons while refusal direction ablation identifies and ablates a single refusal direction from the victim model to suppress refusal of harmful prompts.

SORRY-Bench Black-Box Attacks: To evaluate ABS-Align against black-box jailbreak attacks, we use SORRY-Bench [21], which provides 20 linguistic transformations of 440 unsafe base prompts, resulting in 440 adversarial prompts per transformation. We use ten of the twenty provided transformations in our evaluation. For a complete description of the selected attack types, please refer to Appendix 2.

Evaluation Metrics: We evaluate ABS-Align and related work using a comprehensive set of metrics across three main dimensions: attack success rate (ASR) against multiple jailbreak attacks, over-refusal (OR) rate, and few-shot accuracy on general capability and instruction-following benchmarks.

For measuring jailbreak attack success rate, we use the StrongREJECT [48] benchmark. StrongREJECT contains 313 malicious prompts across six harmful behavior categories and provides a finetuned LLM-as-a-judge evaluator that shows strong agreement with human judgments of jailbreak effectiveness. The evaluator assigns each model response a score between 0 and 1, where 0 indicates a refusal or a response that provides no useful harmful information, while higher scores indicate responses that provide increasingly specific and convincing information relevant to the harmful request. The final StrongREJECT score is computed as the mean score across all 313 prompts. Thus, lower scores indicate stronger safety behavior, while higher scores indicate more successful jailbreaks. We report the StrongREJECT score on all the 313 prompts for evaluating the baseline white-box jailbreak attacks.

For evaluations against the black-box attacks provided by SORRY-Bench [21], we instead use the finetuned evaluator provided by SORRY-Bench. The evaluator determines whether each response fulfills or refuses the corresponding unsafe request, and we report the fulfillment rate across the 440 prompts for each attack category as a percentage from 0 to 100, where higher values indicate a higher attack success rate.

## 4.2. Robustness Results

Here, we evaluate and discuss how ABS-Align affects the attack success rate of jailbreak attacks across multiple dimensions. We evaluate ABS-Align against TwinBreak, other sparsity-based attacks, and black-box jailbreak attacks to assess whether its robustness generalizes beyond the attack used to motivate our design. First, in Section 4.2.1, we assess the robustness of the finetuned model against TwinBreak [23]. After that, in Section 4.2.2 we investigate whether ABS-Align’s robustness generalizes to other sparsity-based attacks. Finally, in Section 4.2.3, we evaluate ABS-Align against a broad range of black-box jailbreak attacks to determine whether its robustness also generalizes to these black-box style attacks.

4.2.1. Robustness to TwinBreak. We first evaluate ABS-Align’s robustness against TwinBreak using the StrongRE-JECT malicious prompt set. To do so, we first finetune the model using our method and then rerun TwinBreak on the finetuned model. TwinBreak attempts to identify and prune safety-critical neurons of the finetuned model over multiple pruning iterations. Following the original TwinBreak setting [23], we run the attack for five pruning iterations. The results are presented in Table 1. Similar to TwinBreak evlautions we report two scores for our evaluation: a periteration score, corresponding to the mean StrongREJECT score across all responses generated at each pruning iteration, and a cumulative score, computed by taking the maximum StrongREJECT score obtained for each prompt across the five iterations and then averaging these maxima across all prompts.

As can be seen in Table 1, ABS-Align consistently reduces both the per-iteration and cumulative scores. For Gemma-2-2B-IT, ABS-Align decreases the per-iteration score from approximately 0.6 to approximately 0.1, while reducing the cumulative score by about 70%, from 0.71 to 0.23. Similar results are observed for Llama-3-8B-Instruct, where the per-iteration score decreases from approximately 0.7 to less than 0.1 and the cumulative score has a sharp drop of 0.7 points. Llama-3.2-3B-Instruct and Qwen-3-4B similarly show significant improvements in their robustness against the pruning attack, where the cumulative scores decreases by around 55% and 83% for the models, respectively.

The results in Table 1 show that ABS-Align significantly improves robustness against TwinBreak, as the attack’s ability to effectively remove safety-aligned behavior by identifying and pruning safety-critical neurons of the finetuned model over its five pruning iterations is significantly im paired.

4.2.2. Generalization of robustness to attacks exploiting sparsity of safety features. While ABS-Align performs well against TwinBreak, ABS-Align may have caused the fine-tuned model to simply adapt to TwinBreak and its specific safety neuron identification method. In this section, we investigate this by examining whether the robustness provided by ABS-Align also generalizes to other sparsitybased attacks, including refusal-direction ablation [25] and NeuroStrike [22]. To do so, we apply both attacks to the ABS-Align-finetuned model and evaluate their effectiveness using StrongREJECT benchmark.

TABLE 1: Performance of ABS-Align aganist TwinBreak [23] Attack.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="5">Per Iteration ASR ↓</td><td rowspan="2">Cumulative↓</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td></tr><tr><td rowspan="2">Gemma-2-2B-IT</td><td>No Defense</td><td>0.55</td><td>0.58</td><td>0.59</td><td>0.55</td><td>0.53</td><td>0.71</td></tr><tr><td>ABS-Align</td><td>0.07</td><td>0.15</td><td>0.15</td><td>0.15</td><td>0.13</td><td>0.23</td></tr><tr><td>Llama-3-8B-Instruct</td><td>No Defense ABS-Align</td><td>0.64 0.04</td><td>0.66 0.04</td><td>0.68 0.07</td><td>0.69 0.06</td><td>0.68 0.03</td><td>0.81 0.11</td></tr><tr><td rowspan="2">Llama-3.2-3B-Instruct</td><td>No Defense</td><td>0.59</td><td>0.54</td><td>0.36</td><td>0.4</td><td>0.35</td><td>0.74</td></tr><tr><td>ABS-Align</td><td>0.19</td><td>0.22</td><td>0.15</td><td>0.15</td><td>0.11</td><td>0.33</td></tr><tr><td rowspan="2">Qwen-3-4B</td><td>No Defense</td><td>0.68</td><td>0.66</td><td>0.65</td><td>0.62</td><td>0.58</td><td></td></tr><tr><td>ABS-Align</td><td>0.04</td><td>0.04</td><td>0.04</td><td>0.07</td><td>0.05</td><td>0.79 0.13</td></tr></table>

TABLE 2: Performance of ABS-Align against other Sparsity-based Attacks.
<table><tr><td>Model</td><td>Method</td><td></td><td>NS [22] Rf-Dir [25] Average</td><td></td></tr><tr><td>Gemma-2-2B-IT</td><td>No Defense ABS-Align</td><td>0.50 0.05</td><td>0.80 0.08</td><td>0.65 0.06</td></tr><tr><td>Llama-3-8B-Instruct</td><td>No Defense ABS-Align</td><td>0.84 0.03</td><td>0.75 0.05</td><td>0.79 0.04</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>No Defense ABS-Align</td><td>0.79 0.37</td><td>0.57 0.30</td><td>0.68 0.33</td></tr><tr><td>Qwen-3-4B</td><td>No Defense ABS-Align</td><td>0.89 0.44</td><td>0.70 0.09</td><td>0.78 0.26</td></tr></table>

As the results in Table 2 show, ABS-Align significantly improves the robustness of the finetuned model against both NeuroStrike and refusal-direction ablation attacks. For Gemma-2-2B-IT, NeuroStrike’s attack success rate decreases from 0.50 to only 0.05, while the score for refusal-direction ablation is reduced by 0.72 points from 0.80 to only 0.08. Similar results are observed for Llama-3- 8B-Instruct, where NeuroStrike becomes largely ineffective and refusal-direction ablation also exhibits a substantial decrease in attack success. Out of the four evaluated models, Llama-3.2-3B-Instruct and Qwen-3-4B show a more modest improvement while still improving robustness by more than 50% in all the settings and up to around 87%.

The improved robustness of the finetuned models against both attacks shows that ABS-Align does not merely improve robustness against TwinBreak’s safety-critical neuron detection method, but also generalizes to other attacks that exploit the sparsity of safety alignment.

4.2.3. Generalization of robustness against black-box attacks. To investigate whether ABS-Align only improves robustness against white-box attacks that exploit the internal activations and representations of LLMs, or whether it also improves the model’s broader jailbreak robustness, we evaluate its performance against a wide range of black-box attacks. To do so, we use ten black-box attack transformations from SORRY-Bench [21]. SORRY-Bench provides 440 malicious prompts for each attack transformation, and we evaluate all 440 prompts for each selected attack. Table 3 reports the results for five of these ten attacks, along with the average across all ten attacks. For a complete description of the selected attacks and their full results, please refer to Appendices 2 and 3.

The results in Table 3 show that ABS-Align also improves robustness against black-box attacks. The average attack success rate across the ten attack types decreases by approximately 20-40 percentage points for the four evaluated models, from around 42%, 48%, 53%, and 69% to approximately 13%, 18%, 33%, and 27%. What makes the results more significant is that the improved robustness is merely achieved by fine-tuning the model with the goal of replicating safety neurons and with no additional information about black-box attack techniques or data augmentation using adversarial black-box prompts.

The results for individual attacks show similarly consistent improvements, with attack success rates reduced by approximately 20–60%. This trend holds across all ten SORRY-Bench attacks included in our evaluation; the complete results are reported in Appendix 3.

The results shows that ABS-Align inherently improves the safety alignment of finetuned models.

## 4.3. Over-refusal

Safety-aligned LLMs are shown to be prone to associate certain keywords such as ”kill” in the user request to refusal response generation, primarily based on the superficial presence of such terms are in the user prompt. This behavior is referred to as over-refusal [40], [49] and negatively affects the helpfulness [11] of LLMs.

Most safety-aligned LLMs have been shown to exhibit more pronounced over-refusal as their safety against harmful content generation increases [49], introducing an important trade-off between safety and usability. Since ABS-Align’s main goal is to make safety alignment more distributed across the model, a potential side effect is an increase in over-refusal, where the reinforced refusal behavior may generalize to benign prompts.

TABLE 3: Performance of ABS-Align across black-box attacks.
<table><tr><td rowspan="2">Model</td><td colspan="8">Attack Success Rate ↓</td></tr><tr><td>Method</td><td>Role Play</td><td>Evidence-based Persuasion</td><td>Expert Endorsement Endorsement</td><td>Authority</td><td>Misrepresentation</td><td>Average over all 10 attacks</td></tr><tr><td rowspan="2">Gemma-2-2B-IT</td><td>No Defense</td><td>20.68</td><td>55.68</td><td>66.81</td><td>65.45</td><td>72.95</td><td>41.58</td></tr><tr><td>ABS-Align</td><td>6.81</td><td>11.81</td><td>16.36</td><td>20.45</td><td>17.27</td><td>13.24</td></tr><tr><td rowspan="2">Llama-3-8B-Instruct</td><td>No Defense</td><td>36.36</td><td>56.59</td><td>60.68</td><td>60.90</td><td>55.45</td><td>48.34</td></tr><tr><td>ABS-Align</td><td>15.90</td><td>22.04</td><td>18.63</td><td>22.50</td><td>13.63</td><td>17.79</td></tr><tr><td rowspan="2">Llama-3.2-3B-Instruct</td><td>No Defense</td><td>34.54</td><td>71.13</td><td>75.45</td><td>79.77</td><td>83.40</td><td>53.07</td></tr><tr><td>ABS-Align</td><td>22.04</td><td>46.36</td><td>45.68</td><td>49.09</td><td>48.18</td><td>33.02</td></tr><tr><td rowspan="2">Qwen-3-4B</td><td>No Defense</td><td>52.04</td><td>93.86</td><td>93.86</td><td>93.63</td><td>92.95</td><td>68.75</td></tr><tr><td>ABS-Align</td><td>7.04</td><td>47.72</td><td>57.95</td><td>49.09</td><td>58.18</td><td>27.20</td></tr></table>

TABLE 4: Comparison of attack robustness, over-refusal, and utility performance under different defense methods.
<table><tr><td rowspan="2"></td><td rowspan="2">Defense Method</td><td colspan="4">Attack Success Rate ↓</td><td>Over-Refusal↓</td><td colspan="4">Utility↑</td></tr><tr><td>TB [23] NS [22]</td><td></td><td>SB [21] Ref-Dir [25]</td><td></td><td>XSTest</td><td>Arc</td><td>TQA</td><td>WG MMLU</td><td>HS</td></tr><tr><td rowspan="4">Gemma-2- 2B-IT</td><td>No Defense</td><td>0.71</td><td>0.50</td><td>41.58</td><td>0.80</td><td>2.00</td><td>42.06</td><td>55.15</td><td>65.43 29.12</td><td>60.25</td></tr><tr><td>W-DOOR [30]</td><td>0.62</td><td>0.19</td><td>35.36</td><td>0.55</td><td>9.00</td><td>44.37</td><td>52.71 65.98</td><td>36.44</td><td>64.94</td></tr><tr><td>ABS-Align (Ours)</td><td>0.23</td><td>0.05</td><td>13.24</td><td>0.08</td><td>4.00</td><td>42.06</td><td>64.29 63.93</td><td>27.34</td><td>57.99</td></tr><tr><td>No Defense</td><td>0.81</td><td>0.84</td><td>48.34</td><td>0.75</td><td>5.00</td><td>50.34 52.7964.33</td><td></td><td>57.76</td><td>64.92</td></tr><tr><td rowspan="4">Llama-3-</td><td>Deep Refusal [42]</td><td>0.25</td><td>0.06</td><td>21.81</td><td>0.07</td><td>48.00</td><td>52.4751.3369.38</td><td></td><td>63.12</td><td>72.59</td></tr><tr><td>8B-Instruct Circuit Breaker [41]</td><td>0.16</td><td>0.79</td><td>16.06</td><td>0.24</td><td>4.00</td><td>50.43 52.87 64.80</td><td></td><td>58.95</td><td>63.48</td></tr><tr><td>ABS-Align (Ours)</td><td>0.11</td><td>0.03</td><td>17.79</td><td>0.05</td><td>1.00</td><td>46.50 67.02 67.16</td><td></td><td>57.48</td><td>67.27</td></tr></table>

Here, we evaluate ABS-Align’s performance with respect to over-refusal. We measure the change in over-refusal rate on our evaluation dataset before and after applying ABS-Align and compare its results with those of the baselines and related work.

Table 5 shows the over-refusal results across the evaluated methods. As can be seen, ABS-Align causes only a slight increase in over-refusal for Gemma-2-2B-IT while the over-refusal rate for all the other models is unchanged or even slighlty improved in the case of Qwen-3-4B and Llama-3-8B-Instruct. As shown in 4.5, this is in contrast to the related works where they generally struggle with balancing over-refusal and safety alignment.

TABLE 5: Change in over-refusal performance for each model and fine-tuning using Addition by Subtraction.
<table><tr><td>Model</td><td>Baseline Fine-tuned</td></tr><tr><td>Gemma-2-2B-IT</td><td>2.00 4.00</td></tr><tr><td>Llama-3-8B-Instruct</td><td>5.00 1.00</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>4.00 4.00</td></tr><tr><td>Qwen-3-4B</td><td>3.00 1.00</td></tr></table>

## 4.4. Utility Evaluations

Finetuning an already trained LLM for safety alignment can potentially degrade its utility and alter its normal behavior. To investigate how ABS-Align affects the model’s general capabilities, we evaluate it on five widely used benchmarks: ARC-Challenge [45], TruthfulQA [44], Wino-Grande [46], MMLU [43], and HellaSwag [47]. The results of our evaluations is reported in Table 6.

TABLE 6: Change in utility performance for each model.
<table><tr><td>Model</td><td>ARC</td><td>TQA</td><td>WG</td><td>MMLU</td><td>HS</td></tr><tr><td>Gemma-2-2B-IT</td><td>0.00</td><td>+9.14</td><td>-1.50</td><td>-1.77</td><td>-2.25</td></tr><tr><td>Llama-3-8B-Instruct</td><td></td><td>-3.67 +14.26 +3.00</td><td></td><td>-0.37</td><td>+2.32</td></tr><tr><td>Llama-3.2-3B-Instruct +0.60+13.78+1.03</td><td></td><td></td><td></td><td>-3.15</td><td>-2.02</td></tr><tr><td>Qwen-3-4B</td><td></td><td>-0.09+11.75-0.79</td><td></td><td>+0.69</td><td>-1.18</td></tr></table>

As the results in Table 6 show, ABS-Align causes only slight performance decreases on WinoGrande, MMLU, and HellaSwag, while ARC-Challenge’s performance is unchanged and TruthfulQA shows a substantial improvement on Gemma-2-2B-IT. Similarly, for Llama-3-8B-Instruct, performance across most benchmarks changes only slightly, while TruthfulQA again exhibits a more substantial improvement. The same pattern can be seen for Llama-3.2- 3B-Instruct and Qwen-3-4B where most of the benchmarks show only a slight precentage points of improvement or degradation.

## 4.5. Comparison with Other Works

For comparison with related work, we consider three recent finetuning-based defenses: W-DOOR [30], Circuit Breaker [41], and Deep Refusal [42]. We conduct a comprehensive set of experiments comparing ABS-Align with these methods across multiple dimensions, including robustness against TwinBreak [23], NeuroStrike [22], refusal-direction ablation [25], black-box jailbreak attacks, over-refusal, and the impact of each method on the model’s general capabilities.

We compare ABS-Align only on models evaluated in the original works and reuse the training hyperparameters reported in their respective papers. We do not extend these methods to additional models, since doing so would require tuning their hyperparameters and could lead to suboptimal baseline performance. Therefore, we use Gemma-2-2B for comparison with W-DOOR and Llama-3-8B for comparison with both Deep Refusal and Circuit Breaker.

The comparison results are reported in Table 4. As the results show, ABS-Align provides substantially greater robustness across all evaluated attacks while exhibiting a lower over-refusal rate on XSTest compared to W-DOOR. In particular, W-DOOR provides only a modest improvement against SORRY-Bench black-box attacks, reducing the ASR from 41.58% to 35.36%, whereas ABS-Align reduces it by around 30 points to 13.24%. Similarly, on TwinBreak, W-DOOR improves the score by only 0.09 points, compared to around 0.50-point improvement with ABS-Align; for refusal-direction ablation, W-DOOR improves the score by 0.25 points, compared to 0.72 points with ABS-Align. W-Door also substantially increase the over-refusal rate from 2% to 9%.

ABS-Align also outperforms Deep Refusal [42] across all evaluated attacks even on refusal-direction ablation, which Deep Refusal is specifically trained to defend against. However, Deep Refusal achieves this robustness at the cost of a substantially higher over-refusal rate on XSTest, refusing close to half of the benign prompts despite incorporating XSTest and OR-Bench samples during training to mitigate over-refusal. Finally, Circuit Breaker [41] also underperforms against ABS-Align across all attacks except for Sorry-Bench where it achieves a slightly better performance compared to ABS-Align across black-box attacks. Circuit Breaker exhibits a slightly worse over-refusal rate too even while its retain dataset which explicitly includes all the XSTest samples to preserve appropriate responses to such benign prompts. Regarding utility, ABS-Align performs better than Circuit Breaker on TruthfulQA, WinoGrande, and HellaSwag, while Circuit Breaker performs slightly better on MMLU and ARC-Challenge.

In summary, compared to related work, ABS-Align shows consistent improvements across a wide range of attacks while having only minor effects on utility and overrefusal. In contrast, the related methods show less consistent robustness, with the defenses performing poorly against attacks and increasing over-refusal.

## 4.6. Ablation Study

This section presents a comprehensive experimental analysis of ABS-Align’s design choices and hyperparameters, examining how different configurations affect the method’s performance in terms of attack success rate and over-refusal. For these experiments, we use the Gemma-2- 2B-IT model and keep all other settings fixed while varying one component at a time. Specifically, we use the same number of training samples, learning rate, optimizer, and scheduler across all ablation experiments.

Table 7 summarizes the different settings considered in our ablation study and compares their results in terms of ASR against TwinBreak [23], over-refusal on XSTest [40], and performance on two representative utility benchmarks, ARC-Challenge [45] and WinoGrande [46].

Does Pruning Safety Neurons Help? Rows 1 and 2 in Table 7 examine how ABS-Align changes under different pruning settings. Row 1 shows the results when no neurons are pruned, which effectively reduces the method to additional DPO-based preference finetuning. Row 2 evaluates the effect of pruning random neurons instead of the identified safety-critical neurons. To do so, we apply the same gradual pruning schedule and prune the same number of neurons, but select their indices randomly rather than using the detected safety neurons.

As shown in Table 7, both settings provide only modest improvements in robustness compared to the substantially larger improvement achieved by the final row, which corresponds to the default ABS-Align configuration. Both settings also exhibit slightly lower over-refusal than the full method, while their utility performance remains broadly comparable.

These results show that pruning safety-critical neurons plays a central role in ABS-Align’s effectiveness and contributes substantially to its robustness gains.

Pruning Strategy. ABS-Align prunes safety neurons gradually, starting from the most important ones and eventually pruning all initially detected safety neurons. Rows 3-5 evaluate the impact of this strategy against several alternative pruning strategies. Row 3 corresponds to the setting where all safety neurons are pruned at the beginning of the finetuning process and remain pruned throughout finetuning. Row 4 corresponds to pruning all safety neurons at the beginning and gradually restoring them throughout finetuning, such that, during the later stages, all initially detected safety neurons are restored. Row 5 corresponds to a random pruning strategy where, every $t _ { p }$ steps, instead of progressively increasing the number of pruned neurons, a fraction r of the initially detected safety neurons is randomly pruned. In other words, every $t _ { p }$ steps, a new random set of safety neurons is pruned while the previously pruned set is restored.

As the results in Rows 3-5 show, all three settings underperform compared to the full method. In particular, nongradual pruning results in a noticeably smaller improvement in robustness. Rows 4 and 5 similarly achieve worse ASR than the full ABS-Align configuration.

Other Parameters. Rows 6-8 evaluate several additional settings related primarily to the finetuning process. By default, ABS-Align freezes the first and last Transformer blocks, whereas in Row 6 these two blocks are also finetuned. As the results show, unfreezing them leads to slightly worse over-refusal and utility performance.

Finally, Rows 7 and 8 show the method’s performance with a higher and lower number of finetuning epochs, respectively. As the results in Row 7 show, longer finetuning leads to worse performance in terms of both over-refusal and utility. In contrast, shorter finetuning in Row 8 results in substantially weaker robustness against attacks.

TABLE 7: Ablation study on the design choices of our method using Gemma-2-2B-IT. ASR is reported on TwinBreak; over-refusal on XSTest; utility on two representative tasks.
<table><tr><td rowspan="2" colspan="2">Setting</td><td>ASR↓ TwinBreak [23]</td><td rowspan="2">Over-refusal↓</td><td colspan="2">Utility Difference</td></tr><tr><td>XSTest [40]</td><td>Arc Challenge [45]</td><td>WinoGrande [46]</td></tr><tr><td>1</td><td>No Pruning</td><td>0.52</td><td>7.0%</td><td>+0.09%</td><td>-0.39%</td></tr><tr><td>2</td><td>Prune Random Neurons instead of Safety Neurons</td><td>0.50</td><td>6.0%</td><td>+0.09%</td><td>-0.24%</td></tr><tr><td>3</td><td>Non-gradual pruning</td><td>0.38</td><td>3.0%</td><td>+0.43%</td><td>-0.63%</td></tr><tr><td>4 5</td><td>Decremental pruning</td><td>0.31</td><td>3.0%</td><td>+0.09%</td><td>-1.50%</td></tr><tr><td></td><td>Prune Randomly Sampled Safety Neurons (r = 0.2)</td><td>0.41</td><td>2.0%</td><td>+0.26%</td><td>-0.32%</td></tr><tr><td></td><td>(r = 0.4)</td><td>0.32 0.31</td><td>4.0% 4.0%</td><td>-0.17% -0.09%</td><td>-1.34% -1.42%</td></tr><tr><td></td><td>(r = 0.6)</td><td></td><td></td><td></td><td></td></tr><tr><td>6 7</td><td>Unfreeze First and Last Transformer Layers</td><td>0.23</td><td>7.0%</td><td>-1.09%</td><td>-1.95%</td></tr><tr><td>8</td><td>Longer Train (4 Epochs)</td><td>0.32</td><td>5.0% 3.0%</td><td>-0.43%</td><td>-1.74%</td></tr><tr><td></td><td>Shorter Train (1 Epoch)</td><td>0.45</td><td></td><td>-0.09%</td><td>-0.71%</td></tr><tr><td></td><td>Ours (Full)</td><td>0.23</td><td>4.0%</td><td>0.00%</td><td>-1.50%</td></tr></table>

To summarize, the results in Table 7 show that both pruning safety-critical neurons and gradually pruning them throughout finetuning are central to ABS-Align’s effectiveness and have a substantial impact on its robustness.

## 4.7. Adaptive Attacks

While ABS-Align performs well against a wide range of attacks in both black-box and white-box settings, an adversary aware of the hardened model’s safety-neuron structure may launch adaptive attacks specifically targeting the reinforced model.

A simple adaptive strategy is to increase the pruning budget by associating a larger number of neurons with safety alignment and pruning them. To evaluate this setting, we run TwinBreak with an increased number of pruning iterations, extending its default budget from 5 to 15 iterations.

TABLE 8: Attack success rate (ASR) against adaptive attackers with increasing iteration budgets, comparing no defense vs. our method.
<table><tr><td>Model</td><td>ASR@5</td><td>ASR@10</td><td>ASR@15</td><td>No Def.</td></tr><tr><td>Gemma-2-2B-IT</td><td>0.23</td><td>0.26</td><td>0.27</td><td>0.71</td></tr><tr><td>Llama-3-8B-Instruct</td><td>0.11</td><td>0.25</td><td>0.25</td><td>0.81</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>0.33</td><td>0.38</td><td>0.39</td><td>0.74</td></tr><tr><td>Qwen-3-4B</td><td>0.13</td><td>0.22</td><td>0.29</td><td>0.79</td></tr></table>

The results of the adaptive attack against the ABS-Alignfinetuned model are reported in Table 8. For Gemma-2-2B-IT, five additional pruning iterations increase the ASR by only 0.03 points, while a further five iterations increase it by only 0.01, resulting in an ASR of 0.27 after 15 pruning iterations. For Llama-3-8B-Instruct, the ASR increases by 0.14 after five additional pruning iterations and shows no further increase after another five, resulting in an ASR of 0.25 after 15 iterations. Llama-3.2-3B-Instruct follow a similar pattern where the score for Llama-3.2-3B-Instruct increases by only 0.05 and 0.01 after 10 and 15 iterations, respectively. Finally, the score for Qwen-3-4B increases by 0.07 points at iteration 10 and by another 0.07 at iteration

15 where 15% of the models’ neurons are pruned. These results show that increasing the pruning budget provides only limited gains to the adaptive attacker and remains far from recovering the attack success rate observed on the undefended models.

## 5. Discussion

In this section, we discuss the limitations of our work, along with insights into unresolved challenges and directions for future work that emerged from our findings.

## 5.1. Limitations

Safety Neuron Detection Methods Are Not Perfect. Addition by Subtraction relies on safety neuron detection methods [22], [23], [33]. These methods typically rely on heuristics, such as identifying the top k neurons with the largest activation differences or highest scores. As a result, they may not identify a perfect set of safety-critical neurons for the target LLM: some safety neurons may be missed, while a small number of unrelated neurons may be incorrectly identified as safety-critical. Furthermore, some neurons have been shown to serve multiple functions [33], making it difficult to associate individual neurons exclusively with safety alignment.

The imperfection of safety neuron detection has two important implications for Addition by Subtraction. First, since some safety-critical neurons may remain undetected, it is difficult to obtain a fully unaligned model without also degrading its general capabilities. As a result, the model may continue to rely on parts of the existing safety mechanism during finetuning, making it more difficult to induce completely new and more distributed safety representations. Second, pruning incorrectly identified or multifunctional neurons may lead to unintended behavior or degradation in general model capabilities, either because these neurons serve multiple functions or because they are not actually part of the safety mechanism.

Measuring Safety-Alignment Density. Our results show that ABS-Align reduces the model’s reliance on the safety-critical neurons identified in the original model, as evidenced by its increased robustness to multiple attacks that exploit sparse safety representations. However, given the limitations of existing safety-neuron detection methods, we do not directly quantify the absolute density of safety alignment in the resulting model. We therefore use denser safety alignment to refer to the less fragile and reduced dependence of safety-aligned behavior on the originally identified sparse set of safety-critical neurons.

Overhead of Preference Optimization. Addition by Subtraction relies on DPO [12] for preference-based finetuning. While DPO is substantially more computationally efficient than RLHF-based alignment methods [11], [12], it still incurs considerable computational and memory overhead because it requires maintaining a frozen reference model in addition to the model being finetuned.

## 5.2. Future Work

Safety alignment is not the only model behavior that has been shown to be tied to a small number of neurons in the model architecture; recent works have reported similar phenomena for various tasks and languages [50], [51]. A potential direction for future research is to investigate whether the proposed method can be extended to make representations of specific capabilities more distributed across the model, with potential applications in settings where a model underperforms on a particular task or where improving performance on a specific task is desired.

Another interesting direction for future work is to better understand the post-training process, particularly safety alignment, and investigate whether non-sparse refusal representations can be established during the main stage of preference optimization rather than through post-hoc approaches such as Addition by Subtraction.

## 6. Related Work

This section provides an overview of related work in three main categories. First, Section 6.1 reviews representative jailbreak methods, including both black-box and white-box attacks. Next, Section 6.2 discusses defensive approaches proposed to mitigate jailbreak attacks. Finally, Section 6.3 reviews attacks that exploit the sparsity of safety alignment in LLMs.

## 6.1. Jailbreak Attacks

A broad range of works have shown that both openweight and closed-source LLMs are vulnerable to jailbreak attacks, where attackers use various techniques to elicit outputs that the models would otherwise refuse to generate [16], [17], [19], [22], [23], [25], [33]. These attacks can be broadly categorized into black-box and white-box approaches based on the level of access the attacker has to the target model.

In black-box settings, the attacker only has query access to the target LLM wherein the attacker can only prompt the target model and observer the outputs returned by the LLM. Black-box attacks initially often relied on manually crafted adversarial prompts employing techniques such as role-playing, persuasion, or obfuscation of malicious intent [13], [18], [20]. Later works tried to make this more systematic [17], [19], [52]. PAIR [19], [53] for instance uses an attacker LLM to iteratively refine and improve jailbreak prompts using only black-box access to the target LLM where the attacker LLM creates adversarial prompts, feeds it to the target LLM and refines its output based on the target LLMs response.

In white-box scenario, the attacker has access to the target LLM’s architecture, weights, and internal states [14], [15], [22], [23], [25]. This additional access enables attacks that are unavailable in black-box settings, including computing model gradients [14], directly intervening on model parameters [22], [23], and manipulating intermediate activations [25] during inference. GCG [14], for instance, appends a set of adversarial suffix tokens to an input prompt and uses gradients from the open-weight target LLM to optimize the suffix tokens toward eliciting an affirmative response to a harmful request. Another example is AutoDAN [15], which uses a hierarchical genetic algorithm to iteratively improve a population of jailbreak prompts based on the target model’s likelihood of generating a predefined affirmative response.

## 6.2. Defense Methods against Jailbreaks

Improving the robustness of LLMs against jailbreaks has emerged as a challenging problem and has resulted in a growing body of work on defensive approaches. At a highlevel, existing defenses against jailbreaks can be broadly categorized into input-time defenses, output-time defenses, and finetuning-based methods.

6.2.1. Input-time and Output-time defenses. Input-time defenses typically rely on tools that are external to the LLM itself and process the input before it is fed to the LLM [26], [28], [29]. Alon et al. [29] proposed using perplexity (a metric measuring how likely the LLM is to generate a particular token sequence) to detect adversarial jailbreak prompts, motivated by the fact that early jailbreak prompts commonly employed adversarial suffixes to bypass safety filters [14]. Kumar et al. [28] proposed an ”eraseand-check” approach that generates subsequences by erasing tokens from the input prompt and inspects them using a safety filter. The input is flagged as harmful if either the original prompt or any of the inspected subsequences is classified as harmful. Llama Guard [26] was proposed by Inan et al. as an LLM-based safeguard model that scans user inputs (and additionally model outputs) to determine whether it contains unsafe content according to a safetyrisk taxonomy. SmoothLLM [27] defends against jailbreak attacks by creating multiple randomly perturbed copies of an input prompt and aggregating model responses to each perturbed version based on the intuition that adversarial prompts are fragile to small scale modifications. A majority vote is then conducted on the aggregated responses and a response consistent with the majority vote is returned by the model.

One of the disadvantages of input-time defenses is the additional overhead they introduce by filtering or transforming the inputs before forwarding user requests to the deployed LLM. Furthermore, these defenses don’t improve the inherent robustness of the model; therefore, if the defense fails or is circumvented, the underlying LLM remains vulnerable to jailbreak attacks.

Output-time defenses are also deployed externally and process responses generated by the LLM before returning them back to the user [26], [54]. Aligner [54], for instance, proposes an approach that processes output responses and applies learned corrections to the output text to reduce harmful content and improve overall alignment. Similar to input-time defenses, however, output-time defenses do not modify the underlying safety alignment of the target LLM and therefore retain many of the same limitations.

Unlike output-time and input-time defenses, Addition by Subtraction directly finetunes the model therefore improving its inherent safety robustness instead of relying on external mechanisms that can only process inputs or outputs of models.

6.2.2. Finetuning-based Defenses. Fine-tuning-based defenses directly finetune an LLM to improve its robustness against jailbreak attacks [30], [31], [41], [55]. W-DOOR [30] improves safety alignment by decomposing DPO’s loss into a dual objective loss that optimizes refusal training in addition to unlearning to targeted unlearning of harmful knowledge, improving robustness against a range of jailbreak attacks such as prefilling [33] and adversarial suffix attacks [14].

Zou et al. [41] propose using representation finetuning in order to ”circuit break” LLMs by rerouting harmful internal representations away from trajectories that would otherwise lead to harmful completions. In other words, they train LLMs into redirecting harmful representation states as soon as they emerge in the model’s internals while preserving benign representations, essentially ”short-circuiting” the internal trajectories that lead to harmful outputs.

Hu et al. propose Alignment-Weighted DPO [31] which uses a Chain-of-Thought safety alignment dataset in addition to a fine-grained training method that decomposes model responses in the training dataset into two parts, namely the reasoning trace and final answer. The method assigns different weights to these two parts of the response based on the strength of the correction signal required for each part.

None of these works address the sparse and fragile safety alignment of LLMs leaving them vulnerable to interventionbased attacks [25] and neuron-pruning attacks [22], [23]. In contrast, Addition by Subtraction directly targets the sparsity of LLM safety alignment by finetuning the model while progressively pruning its safety-critical neurons, encouraging safety-aligned behavior to become distributed across other parts of the model.

## 6.3. Sparsity of Safety Alignment

In recent years, multiple works have shown that safety alignment in modern large language models is highly fragile due to the sparse nature of the resulting safety mechanisms in safety-aligned LLMs [22], [23], [24], [25]. In other words, prior work has shown that safety mechanisms in LLMs have a low density both in the model’s internal representation space [25] and in their distribution across the model architecture [22], [23], [24].

Architecture-wise sparsity in safety alignment allows attackers to jailbreak LLMs by finding and removing safetycritical neurons, essentially removing the safety alignment and transforming the pruned model into a jailbroken version. Numerous works have shown that pruning only 1-5% of neurons can substantially degrade safety alignment while having little impact on the model’s general capabilities [22], [23], [24].

Wei et al. [24] use Wanda [56] and SNIP [57] scores computed on separate benign and harmful prompts to find crucial neurons for safety alignment and removing them. TwinBreak [23] inputs structurally and semantically similar pairs of benign and malicious prompts to a target LLM and compares the resulting intermediate activations between the two sets to locate and prune neurons in an iterative manner. NeuroStrike [22] similarly identifies and prunes safety neurons by training linear probes on intermediate activations from benign and malicious prompts to identify neurons that are strongly associated with distinguishing harmful inputs. Furthermore, Arditi et al. [25] show that refusal behavior in language models is represented by a single direction in the model’s activation space and that by ablating this direction from the activation space it is possible to jailbreak the target LLM.

One of the main goals of Addition by Subtraction is to directly address this weakness. Our method reinforces safety alignment by gradually pruning safety-critical neurons and re-establishing safety alignment by finetuning the resulting pruned model. This process encourages safety-aligned behavior to become distributed across a broader set of neurons and representations, resulting in a less fragile and more robust safety-aligned model.

## 7. Conclusion

We propose Addition by Subtraction, a method for making safety alignment more resilient and less fragile, and show that this leads to broad improvements across multiple dimensions compared to existing defenses while requiring only limited finetuning. Addition by Subtraction significantly reduces the effectiveness of attacks that exploit sparse neuron-level or representation-level safety alignment, while also improving robustness against a wide range of other jailbreak attacks without any explicit guidance or data augmentation during finetuning.

## References

[1] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, “Language models are few-shot learners,” in Advances in Neural Information Processing Systems, 2020.

[2] J. Wei, M. Bosma, V. Zhao, K. Guu, A. W. Yu, B. Lester, N. Du, A. M. Dai, and Q. V. Le, “Finetuned language models are zero-shot learners,” in International Conference on Learning Representations, 2022.

[3] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. Ponde, J. Kaplan,´ H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger, M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder, M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P. Such, D. W. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss, W. H. Guss, A. Nichol, I. Babuschkin, S. Balaji, S. Jain, A. Carr, J. Leike, J. Achiam, V. Misra, E. Morikawa, A. Radford, M. M. Knight, M. Brundage, M. Murati, K. Mayer, P. Welinder, B. McGrew, D. Amodei, S. Mc-Candlish, I. Sutskever, and W. Zaremba, “Evaluating large language models trained on code,” ArXiv preprint arXiv:2107.03374, 2021.

[4] D. Hendrycks, C. Burns, S. Kadavath, A. Arora, S. Basart, E. Tang, D. Song, and J. Steinhardt, “Measuring mathematical problem solving with the math dataset,” in Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

[5] J. Wei, X. Wang, D. Schuurmans, M. Bosma, b. ichter, F. Xia, E. Chi, Q. V. Le, and D. Zhou, “Chain-of-thought prompting elicits reasoning in large language models,” in Advances in Neural Information Processing Systems, 2022.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pretraining of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019.

[7] H. Pearce, B. Ahmad, B. Tan, B. Dolan-Gavitt, and R. Karri, “Asleep at the keyboard? assessing the security of github copilot’s code contributions,” in IEEE Symposium on Security and Privacy (SP), 2022.

[8] S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith, “RealToxicityPrompts: Evaluating neural toxic degeneration in language models,” in Findings of the Association for Computational Linguistics: EMNLP, 2020.

[9] J. Xu, D. Ju, M. Li, Y.-L. Boureau, J. Weston, and E. Dinan, “Botadversarial dialogue for safe conversational agents,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, 2021.

[10] N. Lambert, J. Morrison, V. Pyatkin, S. Huang, H. Ivison, F. Brahman, L. J. V. Miranda, A. Liu, N. Dziri, X. Lyu, Y. Gu, S. Malik, V. Graf, J. D. Hwang, J. Yang, R. L. Bras, O. Tafjord, C. Wilhelm, L. Soldaini, N. A. Smith, Y. Wang, P. Dasigi, and H. Hajishirzi, “Tulu 3: Pushing frontiers in open language model post-training,” in Second Conference on Language Modeling (COLM), 2025.

[11] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Gray, J. Schulman, J. Hilton, F. Kelton, L. Miller, M. Simens, A. Askell, P. Welinder, P. Christiano, J. Leike, and R. Lowe, “Training language models to follow instructions with human feedback,” in Advances in Neural Information Processing Systems, 2022.

[12] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[13] X. Shen, Z. Chen, M. Backes, Y. Shen, and Y. Zhang, “”do anything now”: Characterizing and evaluating in-the-wild jailbreak prompts on large language models,” in Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security (CCS), 2024.

[14] A. Zou, Z. Wang, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv, 2023.

[15] X. Liu, N. Xu, M. Chen, and C. Xiao, “Autodan: Generating stealthy jailbreak prompts on aligned large language models,” in The Twelfth International Conference on Learning Representations (ICLR), 2024.

[16] M. Andriushchenko, F. Croce, and N. Flammarion, “Jailbreaking leading safety-aligned LLMs with simple adaptive attacks,” in The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[17] P. Ding, J. Kuang, D. Ma, X. Cao, Y. Xian, J. Chen, and S. Huang, “A wolf in sheep’s clothing: Generalized nested jailbreak prompts can fool large language models easily,” in North American Chapter of the Association for Computational Linguistics (NACCL), 2024.

[18] D. Kang, X. Li, I. Stoica, C. Guestrin, M. Zaharia, and T. Hashimoto, “ Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks ,” in IEEE Security and Privacy Workshops (SPW), 2024.

[19] P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “ Jailbreaking Black Box Large Language Models in Twenty Queries ,” in IEEE Conference on Secure and Trustworthy Machine Learning (SaTML), 2025.

[20] Y. Zeng, H. Lin, J. Zhang, D. Yang, R. Jia, and W. Shi, “How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

[21] T. Xie, X. Qi, Y. Zeng, Y. Huang, U. M. Sehwag, K. Huang, L. He, B. Wei, D. Li, Y. Sheng, R. Jia, B. Li, K. Li, D. Chen, P. Henderson, and P. Mittal, “Sorry-bench: Systematically evaluating large language model safety refusal,” in The Thirteenth International Conference on Learning Representations, 2025.

[22] L. Wu, S. Behrouzi, M. Rostami, M. Thang, S. Picek, and A.- R. Sadeghi, “Neurostrike: Neuron-level attacks on aligned llms,” Network and Distributed System Security (NDSS) Symposium, 2026.

[23] T. Krauß, H. Dashtbani, and A. Dmitrienko, “TwinBreak: Jailbreaking LLM security alignments based on twin prompts,” in 34th USENIX Security Symposium (USENIX Security 25), 2025.

[24] B. Wei, K. Huang, Y. Huang, T. Xie, X. Qi, M. Xia, P. Mittal, M. Wang, and P. Henderson, “Assessing the brittleness of safety alignment via pruning and low-rank modifications,” in Proceedings of the 41st International Conference on Machine Learning (ICML), 2024.

[25] A. Arditi, O. B. Obeso, A. Syed, D. Paleka, N. Rimsky, W. Gurnee, and N. Nanda, “Refusal in language models is mediated by a single direction,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[26] H. Inan, K. Upasani, J. Chi, R. Rungta, K. Iyer, Y. Mao, M. Tontchev, Q. Hu, B. Fuller, D. Testuggine, and M. Khabsa, “Llama guard: Llm-based input-output safeguard for human-ai conversations,” ArXiv, 2023.

[27] A. Robey, E. Wong, H. Hassani, and G. J. Pappas, “SmoothLLM: Defending large language models against jailbreaking attacks,” Transactions on Machine Learning Research (TMLR), 2025.

[28] A. Kumar, C. Agarwal, S. Srinivas, A. J. Li, S. Feizi, and H. Lakkaraju, “Certifying LLM safety against adversarial prompting,” in First Conference on Language Modeling (COLM), 2024.

[29] G. Alon and M. Kamfonas, “Detecting language model attacks with perplexity,” ArXiv, 2023.

[30] X. Zhao, W. Cai, T. Shi, D. Huang, L. Lin, S. Mei, and D. Song, “Improving LLM safety alignment with dual-objective optimization,” in Forty-second International Conference on Machine Learning (ICML), 2025.

[31] M. Hu, V. Datla, A. Kumar, Z. Guan, S. Li, A. Samuel, and D. Liu, “Alignment-weighted DPO: A principled reasoning approach to improve safety alignment,” in The Fourteenth International Conference on Learning Representations (ICLR), 2026.

[32] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. u. Kaiser, and I. Polosukhin, “Attention is all you need,” in Advances in Neural Information Processing Systems, 2017.

[33] A. Wei, N. Haghtalab, and J. Steinhardt, “Jailbroken: How does LLM safety training fail?” in Thirty-seventh Conference on Neural Information Processing Systems, 2023.

[34] T. Tikhomirova and D. U. Wulff, “Where meaning lives: Layerwise accessibility of psycholinguistic features in encoder and decoder language models,” in Findings of the Association for Computational Linguistics: ACL, 2026.

[35] AI@Meta, “Llama 3 model card,” 2024. [Online]. Available: https://github.com/meta-llama/llama3/blob/main/MODEL CARD.md

[36] A. Grattafiori et al., “The llama 3 herd of models,” 2024.

[37] G. Team, “Gemma,” 2024. [Online]. Available: https://www.kaggle.com/m/3301

[38] J. Ji, D. Hong, B. Zhang, B. Chen, J. Dai, B. Zheng, T. A. Qiu, J. Zhou, K. Wang, B. Li, S. Han, Y. Guo, and Y. Yang, “Pkusaferlhf: Towards multi-level safety alignment for llms with human preference,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds., 2025.

[39] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto, “Stanford alpaca: An instructionfollowing llama model,” 2023.

[40] P. Rottger, H. Kirk, B. Vidgen, G. Attanasio, F. Bianchi, and D. Hovy,¨ “XSTest: A test suite for identifying exaggerated safety behaviours in large language models,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024.

[41] A. Zou, L. Phan, J. Wang, D. Duenas, M. Lin, M. Andriushchenko, J. Z. Kolter, M. Fredrikson, and D. Hendrycks, “Improving alignment and robustness with circuit breakers,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[42] Y. Xie, Y. Zhang, T. Liu, D. Ma, and T. Liu, “Beyond surface alignment: Rebuilding LLMs safety mechanism via probabilistically ablating refusal direction,” in Findings of the Association for Computational Linguistics: EMNLP 2025, 2025.

[43] D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” in International Conference on Learning Representations, 2021.

[44] S. Lin, J. Hilton, and O. Evans, “TruthfulQA: Measuring how models mimic human falsehoods,” in Proceedings ofthe 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022.

[45] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord, “Think you have solved question answering? try arc, the AI2 reasoning challenge,” arXiv preprint arXiv:1803.05457, 2018.

[46] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “Winogrande: an adversarial winograd schema challenge at scale,” Communications of the ACM, 2021.

[47] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi, “HellaSwag: Can a machine really finish your sentence?” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

[48] A. Souly, Q. Lu, D. Bowen, T. Trinh, E. Hsieh, S. Pandey, P. Abbeel, J. Svegliato, S. Emmons, O. Watkins, and S. Toyer, “A strongREJECT for empty jailbreaks,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[49] J. Cui, W.-L. Chiang, I. Stoica, and C.-J. Hsieh, “OR-bench: An overrefusal benchmark for large language models,” in Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025.

[50] T. Tang, W. Luo, H. Huang, D. Zhang, X. Wang, X. Zhao, F. Wei, and J.-R. Wen, “Language-specific neurons: The key to multilingual capabilities in large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

[51] X. Wang, K. Wen, Z. Zhang, L. Hou, Z. Liu, and J. Li, “Finding skill neurons in pre-trained transformer-based language models,” in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022.

[52] A. Mehrotra, M. Zampetakis, P. Kassianik, B. Nelson, H. S. Anderson, Y. Singer, and A. Karbasi, “Tree of attacks: Jailbreaking blackbox LLMs automatically,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[53] J. Yu, X. Lin, Z. Yu, and X. Xing, “LLM-Fuzzer: Scaling assessment of large language model jailbreaks,” in 33rd USENIX Security Symposium (USENIX Security 24), 2024.

[54] J. Ji, B. Chen, H. Lou, D. Hong, B. Zhang, X. Pan, T. Qiu, J. Dai, and Y. Yang, “Aligner: Efficient alignment by learning to correct,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024.

[55] X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson, “Safety alignment should be made more than just a few tokens deep,” in The Thirteenth International Conference on Learning Representations (ICLR), 2025.

[56] M. Sun, Z. Liu, A. Bair, and J. Z. Kolter, “A simple and effective pruning approach for large language models,” in The Twelfth International Conference on Learning Representations, 2024.

[57] N. Lee, T. Ajanthan, and P. H. Torr, “Snip: Single-shot network pruning based on connection sensitivity,” in The Seventh International Conference on Learning Representations (ICLR), 2019.

## Appendix

## 1. Training Details

We train all the evaluated models for two epochs on 1,000 randomly selected sample from PKU-SafeRLHF [38] dataset with a batch size of two. Following [41], we observed that adding a small number of samples for retaining models’ general capability is necessary. Therefore we follow Circuit Breaker and additionally use 850 samples from Alpaca [39] and 150 benign samples from XSTest [40] during training Llama-3-8B-Instruct and Llama-3.2-3B to retain the models’ capability during fine-tuning. To create preference pairs for the retain part of the dataset, we use a set of fixed refusal answers for the dispreferred choice and each model’s benign response for the preferred choice. Preference pairs for PKU-SafeRLHF are used directly from the dataset itself.

Regarding ABS-Align’s pruning strategy, we set pruning steps $t _ { p } ~ = ~ 2 0 0$ and the pruning rate to 20% such that every 200 steps safety neurons corresponding to one pruning iteration detected by TwinBreak are pruned. We use a linear cosine scheduler and AdamW as the optimizer for both models.

TABLE 9: SORRY-Bench black-box attack transformations used in our evaluation.
<table><tr><td>Attack Type</td><td>Description</td></tr><tr><td>Question</td><td>Harmful instruction rephrased as question</td></tr><tr><td>Slang</td><td>Rewrites the harmful prompt using collo- quial language and slang.</td></tr><tr><td>Uncommon</td><td>Éxpresses the harmful request using less</td></tr><tr><td>Dialects Technical Terms</td><td>common dialects.</td></tr><tr><td></td><td>Rephrases the request using technical ter- minology.</td></tr><tr><td>Role Play</td><td>Frames the harmful request within an as- signed role or hypothetical persona.</td></tr><tr><td>Misspellings</td><td>Introduces spelling or grammatical errors into the request.</td></tr><tr><td>Authority</td><td>Cites authoritative sources or institutions</td></tr><tr><td>Endorsement</td><td>to support the request</td></tr><tr><td>Misrepresentation</td><td>Misrepresents the harmful purpose of the</td></tr><tr><td>Evidence-based</td><td>request as benign. Uses facts to justify fulfilling the request.</td></tr><tr><td>Persuasion</td><td></td></tr><tr><td>Expert Endorsement</td><td>Invokes the opinion of an expert.</td></tr></table>

For finetuning the Gemma-2-2B-IT model, we use a learning rate of 1e − 5 and we set the beta for DPO to 0.1. We set weight decay to 0 and warmup ratio to 0.1. For Finetuning Llama-3-8B-Instruct, we use a learning rate of 2e-5, beta of 0.1, and warmup ratio of 0.1. For Llama-3.2-3B we use a learning rate of 2e-4 and beta of 0.3 with a warmup ratio of 0.1. Finally, for Qwen-3-4B we use a learning rate of 2e-4, a beta of 0.1, and a warmup ratio of 0.05.

## 2. SORRY-Bench Black-box-style Adversarial Prompts

SORRY-Bench [21] has black-box twenty attack types in four categories namely writing style attacks, multilanguages, encoding and decryption, and presuasion techniques. We exclude multi-language and encoding and decryption attacks and select the ten categories in Table 9 in our evaluations.

## 3. Full SORRY-Bench Results

We presented ASR over five attacks from SORRY-Bench and the average over all the selected ten attacks in Section 4.2.3. Here we provide the full results over all the ten attacks for completeness. Table 10 provides ASR for all the ten evaluated attacks for Gemma-2-2B-IT which shows that ABS-Align consistently improves robustness. Results for Llama-3-8B-Instruct, Llama-3.2-3B, and Qwen-3-4B show a similar trend as the numbers reported in Tables 11, 12, and 13 show.

TABLE 10: Attack success rate (%) of Gemma-2-2B-IT across all ten SORRY-Bench black-box attacks used in our evaluation. Lower is better.
<table><tr><td>Attack</td><td>No Defense</td><td>ABS-Align</td></tr><tr><td>Question</td><td>19.77</td><td>12.50</td></tr><tr><td>Slang</td><td>31.59</td><td>13.86</td></tr><tr><td>Uncommon Dialects</td><td>27.95</td><td>12.50</td></tr><tr><td>Technical Terms</td><td>24.09</td><td>8.86</td></tr><tr><td>Role Play</td><td>20.68</td><td>6.81</td></tr><tr><td>Misspellings</td><td>30.90</td><td>12.04</td></tr><tr><td>Authority Èndorsement</td><td>65.45</td><td>20.45</td></tr><tr><td>Misrepresentation</td><td>72.95</td><td>17.27</td></tr><tr><td>Evidence-based Persuasion</td><td>55.68</td><td>11.81</td></tr><tr><td>Expert Endorsement</td><td>66.81</td><td>16.36</td></tr><tr><td>Average</td><td>41.58</td><td>13.24</td></tr></table>

TABLE 11: Attack success rate (%) of Llama-3-8B-Instruct across all ten SORRY-Bench black-box attacks used in our evaluation. Lower is better.
<table><tr><td>Attack</td><td>No Defense</td><td>ABS-Align</td></tr><tr><td>Question</td><td>41.13</td><td>18.86</td></tr><tr><td>Slang</td><td>41.36</td><td>16.59</td></tr><tr><td>Uncommon Dialects</td><td>41.36</td><td>11.36</td></tr><tr><td>Technical Terms</td><td>46.36</td><td>20.00</td></tr><tr><td>Role Play</td><td>36.36</td><td>15.90</td></tr><tr><td>Misspellings</td><td>43.18</td><td>18.40</td></tr><tr><td>Authority Endorsement</td><td>60.90</td><td>22.50</td></tr><tr><td>Misrepresentation</td><td>55.45</td><td>13.63</td></tr><tr><td>Evidence-based Persuasion</td><td>56.59</td><td>22.04</td></tr><tr><td>Expert Endorsement</td><td>60.68</td><td>18.63</td></tr><tr><td>Average</td><td>48.34</td><td>17.79</td></tr></table>

TABLE 12: Attack success rate (%) of Llama-3.2-3B across all ten SORRY-Bench black-box attacks used in our evaluation. Lower is better.
<table><tr><td>Attack</td><td>No Defense</td><td>ABS-Align</td></tr><tr><td>Question</td><td>36.82</td><td>27.27</td></tr><tr><td>Slang</td><td>36.36</td><td>22.73</td></tr><tr><td>Uncommon Dialects</td><td>35.23</td><td>20.91</td></tr><tr><td>Technical Terms</td><td>43.18</td><td>27.95</td></tr><tr><td>Role Play</td><td>34.55</td><td>22.05</td></tr><tr><td>Misspellings</td><td>34.77</td><td>20.00</td></tr><tr><td>Authority Endorsement</td><td>79.77</td><td>49.09</td></tr><tr><td>Misrepresentation</td><td>83.41</td><td>48.18</td></tr><tr><td>Evidence-based Persuasion</td><td>71.14</td><td>46.36</td></tr><tr><td>Expert Endorsement</td><td>75.45</td><td>45.68</td></tr><tr><td>Average</td><td>53.07</td><td>33.02</td></tr></table>

TABLE 13: Attack success rate (%) of Qwen-3-4B across all ten SORRY-Bench black-box attacks used in our evaluation. Lower is better.
<table><tr><td>Attack</td><td>No Defense</td><td>ABS-Align</td></tr><tr><td>Question</td><td>44.09</td><td>6.81</td></tr><tr><td>Slang</td><td>55.90</td><td>9.54</td></tr><tr><td>Uncommon Dialects</td><td>58.18</td><td>13.86</td></tr><tr><td>Technical Terms</td><td>52.50</td><td>13.40</td></tr><tr><td>Role Play</td><td>52.04</td><td>7.04</td></tr><tr><td>Misspellings</td><td>50.45</td><td>8.40</td></tr><tr><td>Authority Éndorsement</td><td>93.63</td><td>49.09</td></tr><tr><td>Misrepresentation</td><td>92.95</td><td>58.18</td></tr><tr><td>Evidence-based Persuasion</td><td>93.86</td><td>47.72</td></tr><tr><td>Expert Endorsement</td><td>93.86</td><td>57.95</td></tr><tr><td>Average</td><td>68.75</td><td>27.20</td></tr></table>