# Calibration-Aware Uncertainty Cascades for Eficient Heterogeneous Model Collaboration

Yilin Zhang<sup>∗</sup>, Han Jiang<sup>∗</sup>, Cai Xu, Ying Liu, Wei Zhao

School of Computer Science and Technology, Xidian University

Xi’an, China

ylzhang\_3@stu.xidian.edu.cn, han522708@gmail.com, cxu@xidian.edu.cn, ying210281@163.com, ywzhao@mail.xidian.edu.cn

## Abstract

Heterogeneous model collaboration seeks to exploit the complementary strengths of diferent models to balance predictive performance and inference cost. Existing approaches typically rely either on trained routers, which tie routing decisions to a fixed task and model pool, or on raw-confidence cascades, whose thresholds lack consistent reliability semantics across heterogeneous models. Consequently, these approaches adapt poorly to changing model pools and deployment budgets. We propose Calibration-Aware Uncertainty Cascades (CAUC), a simple post-hoc framework that independently calibrates each model’s confidence and selects deployment policies using validation data. The resulting calibrated confidence scores establish a common reliability scale for accepting an early prediction, invoking a stronger model, or selectively combining model outputs. This unified decision criterion decouples deployment policies from any particular model pool or operating budget. We further show theoretically that calibration gives confidence thresholds an explicit selective-risk interpretation, whereas uncalibrated scores ofer no comparable reliability guarantee. Extensive experiments demonstrate that, across six language benchmarks, CAUC achieves an average relative accuracy improvement of 1.9% over strong-modelonly inference while avoiding approximately 47% of strongmodel calls. On image classification benchmarks, it maintains or improves predictive performance while reducing measured GFLOPs by up to 57%.

## Introduction

Modern AI deployments rely on heterogeneous models with diferent capabilities and computational costs. Compact models can handle many inputs eficiently on resourceconstrained devices or low-cost servers, whereas larger models generally provide stronger predictive performance at substantially higher latency and computation. Always invoking the strongest model is therefore wasteful when less expensive models can already solve many inputs correctly. The central deployment question is how much computation each input should receive: whether to stop with the current model, invoke a stronger one, or combine complementary predictions. This trade-of arises in both edge–cloud vision systems that coordinate device and server inference and LLM services that select among models with diferent capabilities and prices (Kag et al. 2023; Chen, Zaharia, and Zou 2024). The objective is consequently not accuracy alone, but predictive performance under a computation or service budget.

A common approach is to train a router that selects which model should process each query. Recent routers learn query–model compatibility from benchmark outcomes, preference data, or query and model representations, sometimes using contrastive objectives (Shnitzer et al. 2024; Ong et al. 2025; Chen et al. 2024; Zhuang et al. 2025). Although these methods can exploit rich supervision, their decision rules are typically coupled to the training tasks, candidate model pool, and cost definition. Adding or replacing a model, or changing the deployment budget, may therefore require new performance labels and router retraining. This dependence limits their flexibility in model libraries whose members and operating costs evolve over time.

Uncertainty-based cascades provide a lightweight alternative: models are evaluated in increasing order of cost, and a confidence threshold determines whether the system stops or invokes a stronger model (Jitkrittum et al. 2023; Gupta et al. 2024; Ramírez, Birch, and Titov 2024). However, raw confidence is often an unreliable proxy for correctness. The same score can correspond to substantially diferent empirical accuracies across models and datasets, making the behavior of a shared threshold unpredictable. Alternatively, tuning a separate threshold for every model complicates model replacement and cascade expansion. Conventional cascades also discard earlier predictions after deferral, even when their errors are complementary to those of later models.

To address these limitations, we propose Calibration-Aware Uncertainty Cascades (CAUC), a post-hoc framework for eficient heterogeneous model collaboration. CAUC independently calibrates the confidence of each model, establishing a common reliability scale for accepting an early prediction or invoking a stronger model. For deferred inputs, CAUC selectively combines model outputs when their predictions are complementary. This separation between model-specific reliability estimation and deployment policy allows models to be replaced or added through lightweight post-hoc calibration and policy updates, without training a task-specific router or tuning a separate stopping threshold for every model. We further extend CAUC to multi-model cascades through recursive fusion, allowing predictions from earlier models to contribute to the final decision rather than being discarded

![](images/8c1d4311e31cdf82c255ebc40b37f0704e838d8fd8ead069360c338d88a2c526.jpg)  
Figure 1: Comparison ofcollaboration paradigms for heterogeneous models. (a) Trained routers need labeled routing data and retraining after model-pool changes. (b) Raw-confidence cascades use model-specific thresholds and discard earlier predictions upon deferral. (c) CAUC independently calibrates models to enable a shared threshold across the cascade. It selectively fuses retained logits when they are complementary; otherwise, it uses the final model’s prediction.

## after deferral.

Our contributions are summarized as follows:

• We propose CAUC, a post-hoc framework that independently calibrates heterogeneous models and uses calibrated confidence as a common reliability scale for early acceptance, model escalation, and selective output fusion.

• By aligning confidence semantics across heterogeneous models, CAUC accommodates model replacement and cascade expansion through lightweight calibration and policy updates, rather than router retraining or joint retuning of model-specific thresholds.

• We provide theoretical justification for calibrated confidence thresholding, showing that calibration gives thresholds an explicit selective-risk interpretation and supports near-optimal decisions under a cost-sensitive objective.

• Extensive experiments on 6 LLM benchmarks and 3 image classification datasets demonstrate that CAUC achieves superior performance-cost trade-ofs across diverse model combinations and deployment budgets.

## Related Work

Model Routing. Model routing selects one model from a candidate pool before inference. Existing approaches differ mainly in how they estimate query–model compatibility. Benchmark Routing learns correctness predictors from model outcomes on benchmark datasets, whereas RouteLLM derives pairwise routing decisions from preference data (Shnitzer et al. 2024; Ong et al. 2025). Representation-based routers seek more transferable compatibility signals: RouterDC jointly learns query and model embeddings through dual contrastive objectives, while GraphRouter explicitly models relations among tasks, queries, and LLMs as a heterogeneous graph (Chen et al. 2024; Feng, Shen, and You 2025). EmbedLLM decouples reusable model representations from downstream routing decisions, whereas RadialRouter uses a lightweight structured encoder to capture query–model relations eficiently (Zhuang et al. 2025; Jin et al. 2025). Collectively, these methods exploit specialization within a model pool, but depend on an auxiliary compatibility predictor learned for particular tasks and candidate models. CAUC addresses a diferent decision stage: rather than selecting a model before inference, it decides whether to stop after observing a prediction, using calibrated correctness estimates without training a separate query router.

Cascades and Adaptive Invocation. Model cascades invoke models in increasing order of cost and stop once an intermediate prediction is considered reliable. This principle appears in adaptive image inference through learned exits or transitions between classifiers, and in language-model systems that optimize service order and stopping policies under a budget (Bolukbasi et al. 2017; Huang et al. 2018; Chen, Zaharia, and Zou 2024; Dekoninck, Baader, and Vechev 2025). The central design problem is the deferral criterion. Confidence-based deferral is efective only when the score is suficiently related to correctness and downstream recoverability (Jitkrittum et al. 2023); recent LLM cascades therefore study token-level uncertainty, post-hoc representations, and generation margins as alternatives to a neural query router (Gupta et al. 2024; Ramírez, Birch, and Titov 2024). These approaches reduce strong-model calls, but their decision statistics and suitable thresholds can vary substantially across tasks and models. Conventional cascades also replace the early prediction after deferral, even when its errors are complementary to those of the stronger model. CAUC follows the cascade paradigm but calibrates the stopping statistic explicitly and retains selective fusion only as an action for deferred, ambiguous examples.

Confidence Calibration Uncertainty-based collaboration requires more than ranking easy and dificult inputs: a threshold should have a consistent reliability meaning. Post-hoc calibration aligns predictive confidence with empirical correctness without retraining the base model. Temperature scaling provides a strong baseline for image classifiers, while adaptive temperature scaling extends this idea to languagemodel option logits (Guo et al. 2017; Xie et al. 2024). Selective prediction complements calibration by measuring the risk among accepted examples as coverage changes (Geifman and El-Yaniv 2017); ensemble and fusion methods further show that complementary predictions can improve uncertainty when additional computation is available (Jiang, Ren, and Lin 2023; Liu et al. 2025; Zhou, Shelhamer, and Pleiss 2025). Prior cascade studies primarily use uncertainty as a ranking or deferral signal. CAUC instead treats calibrated correctness probability as a common interface across heterogeneous models, giving its stopping threshold an explicit selective-risk interpretation and allowing the policy to be retuned without jointly retraining the models or a router.

## Calibration-Aware Uncertainty Cascade

## Problem Formulation

We consider a prediction task with a finite candidate label set Y, where $K = | \mathcal { \boldsymbol { y } } |$ |. Each input x is paired with a target $y \in \mathcal { V }$ In image classification, the candidates are visual classes; in multiple-choice question answering, they are the answer options provided for the question. Both settings therefore produce a score for every candidate and return the highestscoring one. For simplicity, we explain the method using a two-model cascade consisting of a small model $f _ { s }$ and a large model $f _ { l } .$ . Model $m \in \{ s , l \}$ produces logits $\breve { z } _ { m } ( x ) \in \mathbb { R } ^ { \breve { K } }$ and predicts $\hat { y } _ { m } ( x ) = \arg \operatorname* { m a x } _ { k } z _ { m , k } ( x )$

The small model is always evaluated first. The system can accept its prediction or defer the input to the large model, after which it returns either the large-model prediction or a combination of both outputs. Let $c _ { s }$ and $c _ { l }$ denote the incremental costs of evaluating the small and large models. If $\rho$ is the fraction of inputs sent to the large model, the expected inference cost is

$$
\bar { c } = c _ { s } + \rho c _ { l } .\tag{1}
$$

Our goal is to reduce $\rho$ and hence c¯ while maintaining or improving predictive performance relative to always using the large model.

## Calibrated Confidence for Cascading

Due to diferences in architecture, model scale, and data distribution, the same raw maximum probability may correspond to diferent empirical accuracies across heterogeneous models, making such probabilities not directly comparable. CAUC addresses this problem by first calibrating each model so that confidence has a common correctness interpretation. It then uses calibrated confidence to decide whether the cascade should stop at the small model or invoke the large model. For deferred inputs, CAUC further checks whether the smallmodel output is complementary enough to retain. We first describe this two-model procedure and then introduce a recursive extension for longer model chains.

The cascade decision requires a score that means the same thing for both models. Native softmax probabilities often fail this requirement: one model can be overconfident while another is underconfident, so a shared raw-confidence threshold can accept samples with very diferent error rates. We therefore fit each model independently on a held-out calibration set $\mathcal { D } _ { \mathrm { c a l } } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { \mathrm { c a l } } }$ . Specifically, model $m$ receives one scalar temperature $\bar { T _ { m } } > 0$ , selected by minimizing its negative log likelihood on $\mathcal { D } _ { \mathrm { c a l } }$

$$
T _ { m } = \arg \operatorname* { m i n } _ { T > 0 } - \frac { 1 } { N _ { \mathrm { c a l } } } \sum _ { i = 1 } ^ { N _ { \mathrm { c a l } } } \log \left[ \operatorname { s o f t m a x } ( z _ { m } ( x _ { i } ) / T ) _ { y _ { i } } \right] .\tag{2}
$$

Both the image and language-model experiments use this standard scalar temperature scaling. For a new input x, the calibrated probability vector $\pi _ { m }$ and confidence $p _ { m }$ are

$$
\begin{array} { l } { \pi _ { m } ( x ) = \displaystyle \mathrm { s o f t m a x } ( z _ { m } ( x ) / T _ { m } ) , \medskip } \\ { p _ { m } ( x ) = \displaystyle \operatorname* { m a x } _ { k } \pi _ { m , k } ( x ) . } \end{array}\tag{3}
$$

Here, $p _ { m } ( x )$ estimates the probability that model m’s prediction on x is correct. Since scaling the logits by a positive temperature preserves their ordering, calibration changes the confidence score without changing the predicted label.

CAUC uses the observed reliability of the large model as its stopping target. Let

$$
\widehat { A } _ { l } = \frac { | \{ i : \hat { y } _ { l } ( x _ { i } ) = y _ { i } \} | } { N _ { \mathrm { c a l } } } , \qquad \tau _ { 0 } = \widehat { A } _ { l } ,\tag{4}
$$

where $\widehat { A } _ { l }$ is the large model’s empirical accuracy on $\mathcal { D } _ { \mathrm { c a l } }$ and $\tau _ { 0 }$ is the shared stopping threshold. CAUC accepts $\hat { y } _ { s } ( x )$ when $p _ { s } ( x ) \geq \tau _ { 0 } ;$ otherwise, it evaluates $f _ { l } .$ . The rule has a direct interpretation: the small model answers only when its estimated probability of being correct reaches the reference accuracy observed for the large model. Since each confidence is calibrated to the same event, replacing either model requires fitting only its scalar temperature and recomputing the corresponding calibration statistics, rather than training an additional decision model.

## Selective Collaboration after Deferral

Deferring an input does not imply that the small-model output is useless. Its prediction may correct errors made by the large model, but indiscriminate fusion can also replace correct large-model answers. CAUC decides whether to retain the small model by measuring this trade-of on $\mathcal { D } _ { \mathrm { c a l } }$ . Among calibration samples for which the small model is more confident than the large model, $p _ { s } ( x _ { i } ) > p _ { l } ( x _ { i } )$ , let $N _ { \mathrm { g a i n } }$ count cases where the small model is correct and the large model is wrong, and let $N _ { \mathrm { l o s s } }$ count the reverse cases. We define the calibration-set complementarity rate as

$$
\mathrm { C R } ( s , l ) = \frac { N _ { \mathrm { g a i n } } - N _ { \mathrm { l o s s } } } { N _ { \mathrm { c a l } } } .\tag{5}
$$

A positive value indicates that trusting the more confident small model produces more corrections than harmful replacements. CAUC therefore enables fusion only when $\mathrm { C } \bar { \mathrm { R } } ( s , l ) > 0$ ; otherwise, every deferred input uses the largemodel prediction directly. This sign test is performed once on the calibration set and introduces no learned decision network.

When fusion is enabled, we first place heterogeneous logits on comparable scales. Let $\ell _ { m } ( \bar { x } ) = z _ { m } ( x ) \bar { / } T _ { m }$ be the calibrated logits. We estimate their global mean $\mu _ { m }$ and standard deviation $\sigma _ { m }$ on $\mathcal { D } _ { \mathrm { c a l } }$ and form $\pmb { q } _ { m } ( \boldsymbol { x } ) \ =$ $( \ell _ { m } ( x ) - \mu _ { m } ) / ( \sigma _ { m } + \varepsilon )$ , where $\varepsilon > 0$ prevents division by zero. For a deferred sample, CAUC computes

$$
\begin{array} { r l } & { z _ { \mathrm { f u s e } } ( x ) = \displaystyle \frac { p _ { s } ( x ) \pmb { q } _ { s } ( x ) + p _ { l } ( x ) \pmb { q } _ { l } ( x ) } { p _ { s } ( x ) + p _ { l } ( x ) } , } \\ & { \hat { y } _ { \mathrm { f u s e } } ( x ) = \arg \displaystyle \operatorname* { m a x } _ { k } z _ { \mathrm { f u s e } , k } ( x ) . } \end{array}\tag{6}
$$

Thus, $\pmb { q } _ { s }$ and $\pmb q _ { l }$ provide standardized class evidence, while $p _ { s }$ and $p _ { l }$ determine how strongly each model contributes on the current input. In summary, CAUC fits independent temperatures, sets the large-model reference threshold and complementarity sign on $\mathcal { D } _ { \mathrm { c a l } }$ , accepts reliable small-model predictions, and applies either direct fallback or selective fusion to the remaining inputs. These components form one calibration-driven cascade without a task-specific selector.

## Recursive Fusion Extension

CAUC selectively fuses two model outputs after deferral. To preserve useful evidence across a longer model chain, we extend it to CAUC-RF, which recursively accumulates calibrated outputs while retaining the same stopping rule. Suppose M models $f _ { 1 } , \ldots , f _ { M }$ are ordered from inexpensive to expensive, and let $\ell _ { j } = z _ { j } / T _ { j }$ denote the individually calibrated logits of model $j . \mathsf { C \bar { A } U \bar { C } { \cdot } R F }$ extends the endpoint fusion by maintaining a recursive score state. It starts with $\pmb { r } _ { 1 } = \pmb { \ell } _ { 1 }$ and, after evaluating model $j \geq 2$ , updates

$$
r _ { j } = \frac { r _ { j - 1 } / \alpha _ { j } + \ell _ { j } / \beta _ { j } } { 2 } , \qquad \alpha _ { j } , \beta _ { j } > 0 .\tag{7}
$$

The pair temperatures $\alpha _ { j }$ and $\beta _ { j }$ are fitted sequentially on $\mathcal { D } _ { \mathrm { c a l } }$ by minimizing the NLL of softmax $( r _ { j } )$ . They control the relative scale of the accumulated evidence and the newly called model. The confidence after stage j is

$$
p ^ { ( j ) } ( x ) = \operatorname* { m a x } _ { k } \mathrm { s o f t m a x } ( \pmb { r } _ { j } ( x ) ) _ { k } .\tag{8}
$$

Unlike the main CAUC rule, this extension can adapt its operating point to an explicit budget. On a calibrationindependent cascade-validation set $\mathcal { D } _ { \mathrm { v a l } }$ , it selects one

![](images/168c6789ddcfaf92e0666d753866ff15df4cb1c66254028c08fb24b865bf5603.jpg)  
Figure 2: CAUC-RF recursively fuses calibrated outputs using learned pairwise temperatures and applies a shared con fidence threshold for early stopping.

threshold shared by all nonfinal stages:

$$
\tau _ { B } \in \arg \operatorname* { m a x } _ { \tau : \widehat { c } _ { \mathrm { v a l } } ( \tau ) \leq B } \widehat { A } _ { \mathrm { v a l } } ( \tau ) ,\tag{9}
$$

where B is the deployment budget, ${ \widehat c _ { \mathrm { v a l } } ( \tau ) }$ is the measured cascade cost, and $\widehat { A } _ { \mathrm { v a l } } ( \tau )$ is the corresponding accuracy on $\mathcal { D } _ { \mathrm { v a l } }$ . Inference stops at the first stage satisfying $p ^ { ( j ) } \geq \tau _ { B } ;$ the final stage always returns its recursive prediction. We report this recursive, budget-tuned extension as CAUC-RF. It retains CAUC’s calibrated-confidence interface and adds recursive evidence accumulation without introducing a learned selector.

## Theoretical Analysis

We establish two theoretical analysis for calibrated thresholding in CAUC. The first relates the stopping threshold to accepted-prediction accuracy, while the second proves its near-optimality under a cost-sensitive objective.

## Why Calibration Matters

Let $P \in [ 0 , 1 ]$ be a model’s calibrated confidence on a randomly drawn input, and let C equal one when its prediction is correct and zero otherwise. The model’s actual accuracy at confidence p is $\eta ( p ) = \operatorname* { P r } ( C = 1 \mid P = p )$ . We summarize its average calibration error by $\epsilon = \bar { \mathbb { E } } | \eta ( P ) - P |$ . For a stopping threshold τ, let $q _ { \tau } = \mathrm { P r } ( P \geq \tau ) > 0$ be the fraction of inputs accepted at this stage.

Proposition 1 (Calibration controls accepted accuracy) For every $\tau \in [ 0 , 1 ]$

$$
\operatorname* { P r } ( C = 1 \mid P \geq \tau ) \geq \tau - \frac { \epsilon } { q _ { \tau } } .\tag{10}
$$

Proof. The accepted accuracy is $\mathbb { E } [ \eta ( P ) \mid P \geq \tau ]$ . The mean reported confidence in this group is at least τ, while its mean calibration gap is at most $\epsilon / q _ { \tau }$ . Subtracting this gap gives Eq. 10. □

This bound gives the stopping threshold an accuracy interpretation. Without calibration, a model could report confidence one on every input while having arbitrarily low accuracy, so the same threshold would provide no reliability guarantee. Calibration controls the accuracy of accepted samples; confidence discrimination determines how many samples can stop early.

<table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Method</td><td colspan="2">MMLU</td><td colspan="2">LogiQA</td><td colspan="2">MathQA</td><td colspan="2">MedMCQA</td><td colspan="2">PIQA</td><td colspan="2">SocialIQA</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td rowspan="2">Single-Model</td><td>Ministral-3-8B</td><td>75.94</td><td>8.00</td><td>57.45</td><td>8.00</td><td>44.44</td><td>8.00</td><td>64.06</td><td>8.00</td><td>86.34</td><td>8.00</td><td>73.61</td><td>8.00</td></tr><tr><td>Gemma-4-26B</td><td>79.08</td><td>26.00</td><td>58.22</td><td>26.00</td><td>45.89</td><td>26.00</td><td>65.48</td><td>26.00</td><td>89.61</td><td>26.00</td><td>76.44</td><td>26.00</td></tr><tr><td rowspan="5">Router</td><td>RouterDC</td><td>78.89</td><td>25.58</td><td>58.22</td><td>26.00</td><td>44.46</td><td>23.80</td><td>65.34</td><td>25.18</td><td>89.61</td><td>26.00</td><td>76.44</td><td>26.00</td></tr><tr><td>Benchmark Routing</td><td>78.34</td><td>16.28</td><td>58.06</td><td>14.47</td><td>45.33</td><td>15.55</td><td>64.81</td><td>16.60</td><td>87.81</td><td>35.35</td><td>75.22</td><td>13.60</td></tr><tr><td>RouteLLM</td><td>78.38</td><td>15.98</td><td>58.37</td><td>20.00</td><td>44.62</td><td>20.28</td><td>64.42</td><td>15.98</td><td>87.43</td><td>16.67</td><td>74.91</td><td>16.17</td></tr><tr><td>GraphRouter</td><td>76.22</td><td>9.66</td><td>56.68</td><td>11.20</td><td>43.65</td><td>8.04</td><td>64.45</td><td>13.50</td><td>86.67</td><td>9.23</td><td>73.61</td><td>8.89</td></tr><tr><td>EmbedLLM</td><td>78.09</td><td>18.78</td><td>56.68</td><td>19.40</td><td>44.62</td><td>15.97</td><td>65.48</td><td>21.80</td><td>89.06</td><td>19.71</td><td>75.54</td><td>19.57</td></tr><tr><td rowspan="3">Cascade</td><td>FrugalGPT</td><td>78.61</td><td>17.73</td><td>58.53</td><td>25.21</td><td>45.70</td><td>17.27</td><td>64.74</td><td>20.59</td><td>88.08</td><td>21.44</td><td>74.82</td><td>20.72</td></tr><tr><td>Margin Sampling</td><td>79.80</td><td>20.34</td><td>58.68</td><td>25.13</td><td>47.37</td><td>24.79</td><td>66.83</td><td>21.20</td><td>89.61</td><td>21.06</td><td>76.17</td><td>20.54</td></tr><tr><td>Post-Hoc-Embed</td><td>78.81</td><td>21.02</td><td>58.22</td><td>24.41</td><td>46.13</td><td>25.14</td><td>65.38</td><td>20.89</td><td>88.63</td><td>20.86</td><td>76.08</td><td>20.51</td></tr><tr><td>Ensemble</td><td>Asymmetric Duo</td><td>80.65</td><td>34.00</td><td>59.91</td><td>34.00</td><td>48.98</td><td>34.00</td><td>68.54</td><td>34.00</td><td>90.37</td><td>34.00</td><td>76.57</td><td>34.00</td></tr><tr><td rowspan="2">Ours</td><td>CAUC</td><td>80.57</td><td>20.44</td><td>59.60</td><td>24.51</td><td>47.98</td><td>24.20</td><td>67.97</td><td>20.77</td><td>90.04</td><td>20.00</td><td>76.39</td><td>20.43</td></tr><tr><td>CAUC-RF</td><td>80.51</td><td>20.44</td><td>59.91</td><td>24.51</td><td>47.70</td><td>24.20</td><td>67.90</td><td>20.77</td><td>90.10</td><td>20.00</td><td>76.53</td><td>20.43</td></tr></table>

Table 1: Two-model multiple-choice results for Ministral-3-8B → Gemma-4-26B. Accuracy is in percentage points; Cost is the average inference cost under the shared accounting. Bold indicates the highest overall accuracy, and underlining the highest accuracy among policies that can terminate without evaluating both models for every example.

## Near-Optimality of Unified-Threshold Cascading

Consider the decision after the small model has been evaluated. Let $P _ { s }$ be its calibrated confidence and let $\eta _ { s } ( p )$ be its true probability of being correct when it reports confidence $p .$ Accepting the small model then has expected error $1 - \eta _ { s } ( p )$ . Deferring to the large model incurs

$$
L _ { \mathrm { l a r g e } } = r _ { l } + \lambda c _ { l } ,\tag{11}
$$

where $r _ { l }$ is the large model’s error rate, $c _ { l }$ is its additional inference cost, and $\lambda \geq 0$ specifies how strongly cost is penalized. We assume this fallback loss does not vary with $P _ { s } .$ . For the nontrivial case $\tau ^ { \star } = 1 - L _ { \mathrm { l a r g e } } \in [ 0 , \dot { 1 } ]$ , the better action is to accept the small model when $\eta _ { s } ( p ) \geq \tau ^ { \star }$

Proposition 2 (Unified-threshold regret) If the small model is perfectly calibrated, so that $\eta _ { s } ( p ) = p ,$ accepting it when $\bar { P _ { s } } \geq \tau ^ { \star }$ is optimal. With average calibration error $\epsilon _ { s } = \mathbb { E } | \eta _ { s } ( P _ { s } ) - P _ { s } |$ , the excess expected loss over an oracle that knows $\eta _ { s }$ satisfies

$$
\mathcal { R } _ { \mathrm { t h r e s h o l d } } - \mathcal { R } _ { \mathrm { o r a c l e } } \leq \epsilon _ { s } .\tag{12}
$$

Proof. The threshold rule and oracle difer only when $P _ { s }$ and $\eta _ { s } ( P _ { s } )$ fall on opposite sides of $\tau ^ { \star }$ . Whenever this happens, the extra loss is no larger than $| \eta _ { s } ( P _ { s } ) - P _ { s }$ |. Averaging over inputs gives Eq. 12; perfect calibration makes the gap zero. □

The threshold depends on the fallback error and deployment cost, not on the small model architecture. It can therefore be reused after replacing the small model, provided that the new model is calibrated and the fallback setting remains unchanged. The CAUC default $\tau _ { 0 } = \widehat { A } _ { l }$ corresponds to matching the large model’s error when cost is not explicitly penalized; CAUC-RF instead selects a budget-specific threshold on cascade-validation data.

## Experiments

## Experimental Setup

Benchmarks. The LLM evaluation uses six multiplechoice benchmarks: MMLU (Hendrycks et al. 2021), LogiQA (Liu et al. 2020), MathQA (Amini et al. 2019), MedMCQA (Pal, Umapathi, and Sankarasubbu 2022), PIQA (Bisk et al. 2020), and SocialIQA (Sap et al. 2019). The vision evaluation uses Caltech256 (Grifin, Holub, and Perona 2007), iWildCam from WILDS (Koh et al. 2021), and ImageNet (Russakovsky et al. 2015); iWildCam includes indistribution (ID) and out-of-distribution (OOD) test settings.

Baselines and inference setting. Baselines include individual small and large models as cost and performance endpoints. The routing baselines are RouterDC (Chen et al. 2024), Benchmark Routing (Shnitzer et al. 2024), RouteLLM (Ong et al. 2025), GraphRouter (Feng, Shen, and You 2025), and EmbedLLM (Zhuang et al. 2025). We compare FrugalGPT (Chen, Zaharia, and Zou 2024), Margin Sampling (Ramírez, Birch, and Titov 2024), and Post-Hoc-Embed (Gupta et al. 2024) as cascades, and Asymmetric Duo (Zhou, Shelhamer, and Pleiss 2025) as an ensemble. Tunable routers use performance-first settings, whereas cascade baselines use approximately cost-matched operating points. All policies use the same examples and saved model outputs.

For datasets with test labels, we divide the original validation set equally into $\mathcal { D } _ { \mathrm { c a l } }$ and $\mathcal { D } _ { \mathrm { v a l } }$ , reserving the test set for final evaluation. PIQA and MedMCQA lack test labels, so their original validation sets serve as test sets. For each, an equally sized pool from the training set is divided equally into $\bar { \mathcal { D } } _ { \mathrm { c a l } }$ and $\mathcal { D } _ { \mathrm { v a l } }$

Evaluation metrics. LLM quality uses exact-match multiple-choice accuracy. Caltech256 and ImageNet use top-1 accuracy, whereas ID and OOD iWildCam use macro-F1 due to class imbalance. Language-model cost is nominal parameter count in billions, a uniform proxy when invocation prices are unavailable. Vision-model cost uses forward-pass GFLOPs from TorchVision metadata, normalized within each cascade so that the larger model costs 1.0×.

<table><tr><td rowspan="2">Paradigm</td><td rowspan="2">Method</td><td colspan="2">Caltech256</td><td colspan="2">iWildCam</td><td colspan="2">iWildCam-OOD</td><td colspan="2">ImageNet</td></tr><tr><td>Acc.</td><td>Rel. GFLOPs</td><td>F1</td><td>Rel. GFLOPs</td><td>F1</td><td>Rel. GFLOPs</td><td>Acc.</td><td>Rel. GFLOPs</td></tr><tr><td>Single-Model</td><td>MnasNet-0.75 ResNet-50</td><td>84.84 88.70</td><td>0.05 1.00</td><td>33.09 38.48</td><td>0.05 1.00</td><td>22.07 23.04</td><td>0.05 1.00</td><td>70.94 80.59</td><td>0.05 1.00</td></tr><tr><td>Router</td><td>Benchmark Routing EmbedLLM</td><td>87.48 87.52</td><td>0.76 0.63</td><td>36.49 36.25</td><td>0.58 0.56</td><td>23.63 22.41</td><td>0.40 0.46</td><td>79.04 79.86</td><td>0.85 0.87</td></tr><tr><td>Cascade</td><td>Post-Hoc-Embed Margin Sampling</td><td>87.24 88.85</td><td>0.47 0.45</td><td>36.14 38.23</td><td>0.64 0.57</td><td>22.03 23.11</td><td>0.69 0.63</td><td>77.68 79.99</td><td>0.55 0.55</td></tr><tr><td>Ensemble</td><td>Asymmetric Duo</td><td>89.96</td><td>1.05</td><td>38.89</td><td>1.05</td><td>25.16</td><td>1.05</td><td>80.49</td><td>1.05</td></tr><tr><td>Ours</td><td>CAUC CAUC-RF</td><td>89.81 90.03</td><td>0.43 0.43</td><td>39.17 38.98</td><td>0.56 0.56</td><td>26.08 25.64</td><td>0.64 0.64</td><td>80.35 80.70</td><td>0.55 0.55</td></tr></table>

Table 2: Two-model image classification results for MnasNet-0.75 → ResNet-50. Accuracy and macro-F1 are percentages; relative GFLOPs are normalized to ResNet-50. Bold and underlining indicate the highest and second-highest predictive performance, respectively, for each dataset setting.

## Main Results on Language Models

Across the six tasks in Table 1, CAUC-RF always improves over large-only inference, while Base improves on five and is efectively tied on SocialIQA. The stronger CAUC variant gains between 0.09 and 2.49 percentage points. Base and CAUC-RF average 70.43% and 70.44% accuracy at a mean cost of 21.73, compared with 69.12% at 26.00 for large-only inference. Asymmetric Duo reaches 70.84% at cost 34.00, so CAUC approaches ensemble accuracy below the large-model endpoint cost.

Figure 3 shows full budget sweeps on two representative tasks. Base and CAUC-RF share a routing schedule, so their separation at identical costs isolates the prediction rule. Both occupy the strongest high-accuracy region, with gains largely saturating at moderate cost while competing cascades plateau at lower accuracy.

## Main Results on Image Models

In Table 2, CAUC achieves the best predictive performance in all four settings, with CAUC-RF leading on Caltech256 and ImageNet and Base on both iWildCam settings. Relative to always-on Asymmetric Duo, CAUC improves every reported metric while reducing relative GFLOPs from 1.05 to 0.43–0.64. CAUC also outperforms ResNet-50 on Caltech256 and both iWildCam settings, while CAUC-RF improves ImageNet accuracy by 0.12 points. The two variants have identical per-dataset costs, isolating the efect of their prediction policies.

CAUC was most compute-eficient when the small model was already accurate. On Caltech256 and ImageNet, it stopped more examples after MnasNet-0.75 and surpassed both routers at 0.43 and 0.55 relative GFLOPs. On iWild-Cam, MnasNet-0.75 achieved only 33.09 and 22.07 macro-F1 in the ID and OOD settings, raising CAUC’s costs to 0.56 and 0.64. CAUC remained cost-competitive on ID but exceeded both routers on OOD, where single-model routing avoids the cascade’s double evaluation of deferred examples.

![](images/d5859aad10037b5afb1cecc68384b6c71b8a37b48360659c6c6ac3f115c693a4.jpg)  
Figure 3: Accuracy–cost trade-ofs across operating points on MMLU and PIQA.

## Calibration and Threshold Reliability

Figure 4(a) shows that the uncalibrated MnasNet-0.75 curve lies above the diagonal across most confidence bins, indicating understated empirical correctness. Temperature scaling with T = 0.542 moves the curve near the diagonal without changing the predicted class. Panel (b) shows that MnasNet-0.75, EficientNet-B0, ShufleNetV2-2.0x, and ViT-L/16 remain near a common confidence–accuracy relation after calibration. Calibrated scores therefore approximate correctness probabilities on a shared scale, supporting one reliability threshold across the cascade. Figures 4(c) and (d) show that calibration makes high-confidence coverage increase more consistently with model compute. The shared cutof therefore measures coverage at a common reliability target rather than model-specific score scaling.

## Complementarity-Guided Fusion

Using the same saved two-model outputs, we compare two CAUC variants, with either direct deferral or confidenceweighted fusion, and test whether the calibration-set complementarity rate predicts which is better. Across all six pairs in Table 3, negative complementarity corresponds to higher direct-deferral accuracy, whereas positive complementarity corresponds to higher fusion accuracy. The zeropoint rule therefore selects the better fixed fallback in every case, after which CAUC-RF improves accuracy by 0.05–0.56 points. Figure 5 extends this comparison across model pairs and datasets. Although the signs predominantly align, some points cross the zero-gain boundary, supporting Eq. 5 while showing that complementarity rate is an admission signal rather than a guarantee.

![](images/c32b4a0b7532bfbaae43d615fe29ba63ec6851c26edb1fa60d24eb1f3bc317a4.jpg)  
(a) MnasNet-0.75 confidence before and after calibration.

![](images/7ab48948c9e9d16545cc4d6e25a7cac145d29a28d16118e9ce563c845590779a.jpg)  
(b) Accuracy of four models at matched calibrated confidence.

![](images/a26b21521c56f2fa22ea0d8958623771c871c4cb0d79331883d4c13772116bb8.jpg)  
(c) High-confidence sample counts before calibration.

![](images/97d6400bdaca7725631da461a49fb6c24a7d898b22f4e2e29b3086674c66728c.jpg)  
(d) High-confidence sample counts after calibration.

Figure 4: Calibration reliability and high-confidence coverage on ImageNet. High-confidence counts use [0.85, 1.00], and the orange curve denotes model GFLOPs.
<table><tr><td>Model Pair</td><td>Direct</td><td>Fuse</td><td>CAUC-RF</td><td>CR (%)</td></tr><tr><td>Ministral-3-3B → Ministral-3-8B</td><td>75.94</td><td>75.14</td><td>75.99</td><td>-1.70</td></tr><tr><td>Qwen3-VL-4B → Ministral-3-8B</td><td>75.94</td><td>76.48</td><td>76.73</td><td>+0.54</td></tr><tr><td>Qwen3-4B → Qwen3-8B</td><td>74.93</td><td>74.66</td><td>75.09</td><td>-0.52</td></tr><tr><td>Ministral-3-3B → Qwen3-8B</td><td>74.93</td><td>75.50</td><td>75.64</td><td>+0.59</td></tr><tr><td>Qwen3-4B → Ministral-3-14B</td><td></td><td>78.7476.64</td><td>78.98</td><td>-1.70</td></tr><tr><td>Qwen3.5-4B → Ministral-3-14B</td><td></td><td>78.74 78.87</td><td>79.43</td><td>+0.20</td></tr></table>

Table 3: Confidence-weighted fusion for six LLM pairs on MMLU. Accuracy and CR are in percentage points. Bold marks the better of direct deferral and fusion; CAUC-RF is the cost-constrained reference.

## Scaling to Longer Model Cascades

We scale an ImageNet cascade from one to seven models, retaining Swin-V2-L as the endpoint, and measure accuracy and cost. Table 4 shows that both policies minimize cost with four models, reducing the large-only cost by 63.0% for CAUC and 58.8% for CAUC-RF. CAUC-RF reaches its highest cascade accuracy of 86.48% with five models, 0.26 points below large-only inference at less than half the cost. Further stages fail to improve accuracy and raise cost, indicating that marginal interception no longer ofsets forward-pass cost; CAUC-RF remains more accurate than Base at a growing cost premium. A fixed three-model LLM baseline comparison appears in the supplementary material.

Complementarity Rate vs. Realized Fusion Accuracy Gain  
![](images/589cd170e794d6ace4604ad073e081d9d074aa5e04f8c2042f9417d6de48cdbd.jpg)  
Figure 5: Calibration-set complementarity rate versus realized accuracy gain from confidence-weighted fusion over direct deferral. Each point represents one dataset–model-pair evaluation across six LLM tasks; dashed lines indicate zero complementarity and zero gain.

<table><tr><td rowspan="2"># Models Newly Added Model</td><td rowspan="2"></td><td colspan="2">CAUC</td><td colspan="2">CAUC-RF</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td>1</td><td>Swin-V2-L (large only)</td><td>86.74</td><td>47.80</td><td>86.74</td><td>47.80</td></tr><tr><td>2</td><td>MnasNet-0.75</td><td>86.22</td><td>28.93</td><td>86.26</td><td>28.93</td></tr><tr><td>3</td><td>EfficientNet-B0</td><td>86.14</td><td>21.74</td><td>86.22</td><td>22.61</td></tr><tr><td>4</td><td>ResNet-50</td><td>86.05</td><td>17.69</td><td>86.30</td><td>19.72</td></tr><tr><td>5</td><td>ConvNeXt-S</td><td>85.96</td><td>18.83</td><td>86.48</td><td>21.75</td></tr><tr><td>6</td><td>Swin-V2-B</td><td>85.73</td><td>19.55</td><td>86.34</td><td>24.12</td></tr><tr><td>7</td><td>ConvNeXt-B</td><td>85.66</td><td>21.92</td><td>86.31</td><td>27.61</td></tr></table>

Table 4: ImageNet results as the cascade grows from one to seven models, with Swin-V2-L fixed as the final endpoint. Each row adds the indicated model to the preceding cascade; the one-model row denotes large-only inference.

## Conclusion

CAUC uses independently calibrated confidence as a shared decision interface for heterogeneous model collaboration. It supports early acceptance, selective deferral, and confidenceweighted fusion without training a pool-specific router, while CAUC-RF recursively integrates predictions from earlier models in longer cascades. Our analysis links calibration error to accepted-sample reliability and bounds the regret of unified-threshold decisions. Across six LLM benchmarks, CAUC improves the accuracy–cost trade-of while avoiding about 47% of strong-model calls. Across three image datasets, it maintains or improves predictive performance while reducing measured GFLOPs by up to 57%. Fusion and multi-model experiments further demonstrate the benefits of model complementarity and the diminishing returns of adding excessive cascade stages.

## References

Amini, A.; Gabriel, S.; Lin, S.; Koncel-Kedziorski, R.; Choi, Y.; and Hajishirzi, H. 2019. MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms. In Proceedings of NAACL-HLT, 2357–2367. Association for Computational Linguistics.

Bisk, Y.; Zellers, R.; Le Bras, R.; Gao, J.; and Choi, Y. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, 7432–7439.

Bolukbasi, T.; Wang, J.; Dekel, O.; and Saligrama, V. 2017. Adaptive Neural Networks for Eficient Inference. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 527–536. PMLR.

Chen, L.; Zaharia, M.; and Zou, J. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research. Featured Certification.

Chen, S.; Jiang, W.; Lin, B.; Kwok, J.; and Zhang, Y. 2024. RouterDC: Query-Based Router by Dual Contrastive Learning for Assembling Large Language Models. In Advances in Neural Information Processing Systems, volume 37, 66305– 66328. Curran Associates, Inc.

Dekoninck, J.; Baader, M.; and Vechev, M. 2025. A Unified Approach to Routing and Cascading for LLMs. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 12987–13010. PMLR.

Feng, T.; Shen, Y.; and You, J. 2025. GraphRouter: A Graphbased Router for LLM Selections. In Yue, Y.; Garg, A.; Peng, N.; Sha, F.; and Yu, R., eds., International Conference on Learning Representations, volume 2025, 26186–26203.

Geifman, Y.; and El-Yaniv, R. 2017. Selective Classification for Deep Neural Networks. In Advances in Neural Information Processing Systems, volume 30.

Grifin, G.; Holub, A.; and Perona, P. 2007. Caltech-256 Object Category Dataset. Technical Report CNS-TR-2007- 001, California Institute of Technology.

Guo, C.; Pleiss, G.; Sun, Y.; and Weinberger, K. Q. 2017. On Calibration of Modern Neural Networks. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, 1321–1330. PMLR.

Gupta, N.; Narasimhan, H.; Jitkrittum, W.; Rawat, A. S.; Menon, A. K.; and Kumar, S. 2024. Language Model Cascades: Token-Level Uncertainty and Beyond. In International Conference on Learning Representations.

Hendrycks, D.; Burns, C.; Basart, S.; Zou, A.; Mazeika, M.; Song, D.; and Steinhardt, J. 2021. Measuring Massive Multitask Language Understanding. In International Conference on Learning Representations.

Huang, G.; Chen, D.; Li, T.; Wu, F.; van der Maaten, L.; and Weinberger, K. Q. 2018. Multi-Scale Dense Networks for Resource Eficient Image Classification. In International Conference on Learning Representations.

Jiang, D.; Ren, X.; and Lin, B. Y. 2023. LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 14165–14178. Toronto, Canada: Association for Computational Linguistics.

Jin, R.; Shao, P.; Wen, Z.; Wu, J.; Feng, M.; Zhang, S.; and Tao, J. 2025. RadialRouter: Structured Representation for Eficient and Robust Large Language Models Routing. In Findings of the Association for Computational Linguistics: EMNLP 2025, 14587–14600. Association for Computational Linguistics.

Jitkrittum, W.; Gupta, N.; Menon, A. K.; Narasimhan, H.; Rawat, A. S.; and Kumar, S. 2023. When Does Confidence-Based Cascade Deferral Sufice? In Advances in Neural Information Processing Systems, volume 36.

Kag, A.; Fedorov, I.; Gangrade, A.; Whatmough, P.; and Saligrama, V. 2023. Eficient Edge Inference by Selective Query. In International Conference on Learning Representations.

Koh, P. W.; Sagawa, S.; Marklund, H.; Xie, S. M.; Zhang, M.; Balsubramani, A.; Hu, W.; Yasunaga, M.; Phillips, R. L.; Gao, I.; Lee, T.; David, E.; Stavness, I.; Guo, W.; Earnshaw, B.; Haque, I.; Beery, S. M.; Leskovec, J.; Kundaje, A.; Pierson, E.; Levine, S.; Finn, C.; and Liang, P. 2021. WILDS: A Benchmark of in-the-Wild Distribution Shifts. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 5637–5664. PMLR.

Liu, C.; Quan, X.; Pan, Y.; Wu, W.; Chen, X.; and Lin, L. 2025. Cool-Fusion: Fuse Large Language Models without Training. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10617–10627. Vienna, Austria: Association for Computational Linguistics.

Liu, J.; Cui, L.; Liu, H.; Huang, D.; Wang, Y.; and Zhang, Y. 2020. LogiQA: A Challenge Dataset for Machine Reading Comprehension with Logical Reasoning. In Proceedings of the Twenty-Ninth International Joint Conference onArtificial Intelligence, 3622–3628.

Ong, I.; Almahairi, A.; Wu, V.; Chiang, W.-L.; Wu, T.; Gonzalez, J. E.; Kadous, M. W.; and Stoica, I. 2025. RouteLLM: Learning to Route LLMs from Preference Data. In International Conference on Learning Representations.

Pal, A.; Umapathi, L. K.; and Sankarasubbu, M. 2022. MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical Domain Question Answering. In Proceedings of the Conference on Health, Inference, and Learning, volume 174 of Proceedings of Machine Learning Research, 248–260. PMLR.

Ramírez, G.; Birch, A.; and Titov, I. 2024. Optimising Calls to Large Language Models with Uncertainty-Based Two-Tier Selection. In Conference on Language Modeling.

Russakovsky, O.; Deng, J.; Su, H.; Krause, J.; Satheesh, S.; Ma, S.; Huang, Z.; Karpathy, A.; Khosla, A.; Bernstein, M.; Berg, A. C.; and Fei-Fei, L. 2015. ImageNet Large Scale Vi-

sual Recognition Challenge. International Journal of Computer Vision, 115(3): 211–252.

Sap, M.; Rashkin, H.; Chen, D.; Le Bras, R.; and Choi, Y. 2019. Social IQa: Commonsense Reasoning about Social Interactions. In Proceedings ofEMNLP-IJCNLP, 4463–4473. Association for Computational Linguistics.

Shnitzer, T.; Ou, A.; Silva, M.; Soule, K.; Sun, Y.; Solomon, J.; Thompson, N.; and Yurochkin, M. 2024. Large Language Model Routing with Benchmark Datasets. In Conference on Language Modeling.

Xie, J.; Chen, A. S.; Lee, Y.; Mitchell, E.; and Finn, C. 2024. Calibrating Language Models with Adaptive Temperature Scaling. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 18128–18138. Miami, Florida, USA: Association for Computational Linguistics.

Zhou, T. G.; Shelhamer, E.; and Pleiss, G. 2025. Asymmetric Duos: Sidekicks Improve Uncertainty. In Advances in Neural Information Processing Systems.

Zhuang, R.; Wu, T.; Wen, Z.; Li, A.; Jiao, J.; and Ramchandran, K. 2025. EmbedLLM: Learning Compact Representations of Large Language Models. In Yue, Y.; Garg, A.; Peng, N.; Sha, F.; and Yu, R., eds., International Conference on Learning Representations, volume 2025, 76913–76926.

## Appendix

This appendix provides supplementary details for CAUC. Section A presents the complete algorithms for the Base and Recursive Fusion variants. Section B describes the models, dataset splits, baselines, and three-model setting. Section C reports additional language-model results for two- and threemodel cascades. Section D compares method-specific setup times. Finally, Section E extends the unified-threshold analysis to confidence-dependent fallback quality and proves the corresponding performance-gap bound.

## A Overall Algorithm Description

Algorithm 1 Base Calibration-Aware Uncertainty Cascade   
Input: Calibration set $\mathcal { D } _ { \mathrm { c a l } } ;$ ordered models $f _ { 1 } , \dots , f _ { M } ;$ test   
input x   
Output: Prediction ${ \hat { y } } ( x )$   
1: for $j = 1 , \dots , \bar { M }$ do   
2: Fit $T _ { j } > 0$ by minimizing the NLL on $\mathcal { D } _ { \mathrm { c a l } }$   
3: S $\begin{array} { r } { \mathrm { e t } \bar { \ell _ { j } } = z _ { j } \dot { / } T _ { j } , \pi _ { j } = \mathrm { s o f t m a x } ( \ell _ { j } ) , } \end{array}$ , and $p _ { j } = \operatorname* { m a x } _ { k } \pi _ { j , k }$   
4: Estimate global $\mu _ { j } , \sigma _ { j }$ of $\ell _ { j }$ on $\mathcal { D } _ { \mathrm { c a l } }$   
5: end for   
6: τ ← calibration accuracy of the final model $f _ { M }$   
7: ${ \mathcal { S } } \gets \{ M \}$   
8: for $j = 1 , \dotsc , M - 1$ do   
9: Compute $\widehat { \mathrm { C R } } ( j , M ) = ( N _ { \mathrm { g a i n } } - N _ { \mathrm { l o s s } } ) / | \mathcal { D } _ { \mathrm { c a l } } |$   
10: i $\widehat { \mathrm { \Gamma } } \widehat { \mathrm { C R } } ( j , M ) > 0$ then   
11: $s  s \cup \{ j \}$   
12: end if   
13: end for   
14: for $j = 1 , \dots , M - 1$ do   
15: Evaluate $f _ { j } ( x )$ and compute $\ell _ { j } ( x ) , \pi _ { j } ( x ) , p _ { j } ( x )$   
16: if $p _ { j } ( x ) \geq \dot { \tau } _ { 0 }$ then   
17: return arg max<sub>k</sub> $\ell _ { j , k } ( x )$   
18: end if   
19: end for   
20: Evaluate $f _ { M } ( x )$ and compute $\ell _ { M } ( x ) , \pi _ { M } ( x ) , p _ { M } ( x )$   
21: $q _ { j } ( x ) \gets \bar { ( \ell _ { j } ( x ) - \mu _ { j } ) / ( \bar { \sigma } _ { j } + \varepsilon ) }$ for every $j \in \mathcal S$   
22: $z _ { \mathrm { f u s e } } ( x ) \gets \left( \sum _ { j \in S } p _ { j } ( x ) q _ { j } ( x ) \right) / \left( \sum _ { j \in S } p _ { j } ( x ) \right)$   
23: return arg max<sub>k</sub> $z _ { \mathrm { f u s e } , k } ( x )$

Algorithms 1 and 2 summarize the two CAUC variants as complete ofline–online procedures. Both variants first fit an independent temperature for every model on the calibration set, so that confidence has a comparable correctness interpretation across heterogeneous architectures, and then evaluate models from inexpensive to expensive. Base uses the final model’s calibration accuracy as its early-exit threshold, retains an earlier model for deferred-example fusion only when its empirical complementarity rate is positive, and fuses the retained standardized logits using calibrated confidence weights. RF instead recalibrates and averages the accumulated prediction whenever a new model is invoked, and selects one shared stopping threshold on the independent cascade-validation set under budget B. All temperatures, normalization statistics, complementarity decisions, fusion parameters, and stopping thresholds are fitted once before deployment; final evaluation labels are never used by either algorithm.

Algorithm 2 CAUC with Recursive Fusion (RF)   
Input: Calibration set $\mathcal { D } _ { \mathrm { c a l } } ;$ cascade-validation set $\mathcal { D } _ { \mathrm { v a l } } ;$ ordered   
models $f _ { 1 } , \ldots , f _ { M } ;$ costs $c _ { 1 } , \ldots , c _ { M } ;$ budget B; test input x   
Output: Prediction yˆ(x)   
1: for $j = 1 , \dots , M$ do   
2: Fit $T _ { j } > 0$ by minimizing the NLL on $\mathcal { D } _ { \mathrm { c a l } }$ and set   
$\ell _ { j } = \tilde { z _ { j } } / T _ { j }$   
3: end for   
4: Set $r _ { 1 }  \ell _ { 1 }$   
5: for $j = 2 , \dots , M$ do   
6: Fit $\alpha _ { j } , \beta _ { j } > 0$ on $\mathcal { D } _ { \mathrm { c a l } }$ by minimizing the NLL of   
softma $\dot { \iota } ( ( r _ { j - 1 } / \alpha _ { j } + \ell _ { j } / \beta _ { j } ) / 2 )$   
7: $r _ { j } \gets ( r _ { j - 1 } / \alpha _ { j } + \ell _ { j } / \beta _ { j } ) / 2$   
8: end for   
9: $\tau _ { B } \gets \mathrm { a r g } \operatorname* { m a x } _ { \tau : \widehat { c } _ { \mathrm { v a l } } ( \tau ) \leq B } \widehat { A } _ { \mathrm { v a l } } ( \tau )$ on $\mathcal { D } _ { \mathrm { v a l } }$   
10: Evaluate $f _ { 1 } ( x )$ and set $\overline { { r } }  \ell _ { 1 } ( x )$   
11: for $j = 1 , \dots ,$ M do   
12: i $\mathbf { f } \ j > 1$ then   
13: Evaluate $f _ { j } ( x )$ and set $r  ( r / \alpha _ { j } + \ell _ { j } ( x ) / \beta _ { j } ) / 2$   
14: end if   
15: $p \gets \mathrm { m a x } _ { k }$ softmax $( r ) _ { k }$   
16: $\mathbf { i f } \ p \ge \tau _ { B } \ o r j = M$ then   
17: return arg max<sub>k</sub> $r _ { k }$   
18: end if   
19: end for

## B Additional Experimental Details

Models. The language experiments use models from the DeepSeek-R1-Distill, Gemma 4, Ministral 3, Qwen3, Qwen3-VL, and Qwen3.5 families. The vision experiments use models from the ConvNeXt, Eficient-Net/EficientNetV2, MnasNet, ResNet, ShufleNetV2, Swin-V2, and ViT families.

Dataset splits. For MMLU, LogiQA, MathQA, SocialIQA, Caltech256, and iWildCam, which provide labeled test partitions in our evaluation setup, we divide the original validation set equally into a calibration set and a cascadevalidation set for threshold selection, and reserve the oficial test set for final evaluation. The oficial iWildCam ID and OOD test partitions are retained as two separate final evaluation settings. PIQA and MedMCQA do not provide test labels, so their original validation sets become the final evaluation sets; for each dataset, we sample an equally sized pool from its training set and divide that pool equally between calibration and threshold selection. ImageNet likewise lacks public test labels, but its larger validation set permits a direct three-way split: 25% for calibration, 25% for threshold selection, and 50% for final evaluation. Thus, no final evaluation example is used to fit temperatures, choose fusion parameters, or select stopping thresholds.

<table><tr><td rowspan="2" colspan="2">Method</td><td colspan="2">MMLU</td><td colspan="2">LogiQA</td><td colspan="2">MathQA</td><td colspan="2">MedMCQA</td><td colspan="2">PIQA</td><td colspan="2">SocialIQA</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td rowspan="3">Naive</td><td>Qwen3.5-4B Ministral-3-14B</td><td>72.34 78.74</td><td>4.00</td><td>58.53</td><td>4.00</td><td>36.18</td><td>4.00</td><td>58.10</td><td>4.00</td><td>85.69</td><td>4.00</td><td>76.35</td><td>4.00</td></tr><tr><td></td><td>82.22</td><td>14.00 31.00</td><td>56.53 59.75</td><td>14.00</td><td>50.08</td><td>14.00</td><td>66.48</td><td>14.00</td><td>85.85</td><td>14.00</td><td>75.67</td><td>14.00</td></tr><tr><td>Gemma-4-31B</td><td></td><td></td><td></td><td>31.00</td><td>50.75</td><td>31.00</td><td>69.46</td><td>31.00</td><td>92.55</td><td>31.00</td><td>75.85</td><td>31.00</td></tr><tr><td rowspan="3">Router</td><td>RouterDC</td><td>81.83</td><td>28.86</td><td>58.53</td><td>4.22</td><td>50.08</td><td>14.00</td><td>69.78</td><td>28.63</td><td>92.55</td><td>30.99</td><td>76.35</td><td>4.00</td></tr><tr><td>GraphRouter</td><td>72.57</td><td>4.79</td><td>57.76</td><td>8.35</td><td>36.18</td><td>4.00</td><td>58.88</td><td>5.81</td><td>85.91</td><td>4.78</td><td>76.21</td><td>4.09</td></tr><tr><td>EmbedLLM</td><td>79.04</td><td>19.10</td><td>59.91</td><td>17.77</td><td>50.62</td><td>16.39</td><td>67.93</td><td>19.49</td><td>90.21</td><td>23.57</td><td>75.99</td><td>19.68</td></tr><tr><td>Cascade</td><td>FrugalGPT</td><td>80.20</td><td>25.30</td><td>59.14</td><td>13.33</td><td>50.82</td><td>30.66</td><td>67.47</td><td>18.29</td><td>91.40</td><td>24.46</td><td>76.35</td><td>21.78</td></tr><tr><td rowspan="2">CAUC (Ours)</td><td>Base</td><td>82.95</td><td>24.65</td><td>60.68</td><td>25.14</td><td>51.91</td><td>32.02</td><td>70.17</td><td>27.02</td><td>91.78</td><td>23.61</td><td>78.42</td><td>18.09</td></tr><tr><td>RF</td><td>82.97</td><td>25.10</td><td>59.45</td><td>26.72</td><td>52.41</td><td>32.26</td><td>70.77</td><td>27.35</td><td>92.22</td><td>24.35</td><td>78.78</td><td>19.28</td></tr></table>

Table A1: Three-model results for Qwen3.5-4B → Ministral-3-14B → Gemma-4-31B. Accuracy is reported in percentage points, and Cost follows the common evaluation accounting. Bold marks the best accuracy in each task.

Baselines. Small-only evaluates the inexpensive model on every example and serves as the minimum-cost endpoint, whereas Large-only always evaluates the stronger model and serves as the single-model performance endpoint. RouterDC learns query and model embeddings with sample– model and sample–sample contrastive objectives, then routes each query according to their learned compatibility (Chen et al. 2024). Benchmark Routing reuses per-model correctness outcomes on existing benchmarks to train binary performance predictors for selecting a model on a new query (Shnitzer et al. 2024). RouteLLM learns a strong-versusweak model routing decision from human preference data so that easier queries can be sent to the cheaper model (Ong et al. 2025). GraphRouter represents tasks, queries, and LLMs in a heterogeneous graph and predicts the performance and cost attributes of query–model edges for routing (Feng, Shen, and You 2025). EmbedLLM first derives compact reusable representations of candidate LLMs from their question–answer behavior and then trains a lightweight downstream router on those representations (Zhuang et al. 2025). FrugalGPT learns an adaptive ordering and stopping policy that calls LLM services sequentially while satisfying a user-specified budget (Chen, Zaharia, and Zou 2024). Margin Sampling uses the small model’s margin between its two highest class scores as a nonparametric confidence measure and defers low-margin examples to the large model (Ramírez, Birch, and Titov 2024). Post-Hoc-Embed trains a post-hoc deferral predictor from token-level uncertainty and representation features to decide when the stronger model should be invoked (Gupta et al. 2024). Asymmetric Duo evaluates a large model together with a smaller sidekick and combines their predictions by learned weighted averaging (Zhou, Shelhamer, and Pleiss 2025); unlike a router or cascade, it therefore incurs both model costs on every example.

## C Additional Language-Model Results

Table A1 evaluates the three-model Qwen3.5-4B → Ministral-3-14B → Gemma-4-31B chain, whereas Figure A1 completes the budget sweeps for the main paper’s two-model Ministral-3-8B → Gemma-4-26B system on the other four tasks.

For the three-model setting, Table A1 shows that the stronger CAUC variant outperforms the largest single model on five of six tasks, with gains from 0.75 points on MMLU to 2.92 points on SocialIQA; on PIQA it is 0.33 points lower. CAUC attains the best result on every task except PIQA. Averaged across tasks, Base and RF reach 72.65% and 72.77% accuracy at costs 25.09 and 25.84, respectively, compared with 71.76% at cost 31.00 for Gemma-4-31B.

![](images/6f90ef71d64aad484f9177b99976b19e3da614d0f12e56aae01d1091ce7447eb.jpg)  
Figure A1: Two-model accuracy–cost trade-ofs for Ministral-3-8B → Gemma-4-26B across operating points on LogiQA, MathQA, MedMCQA, and SocialIQA, complementing the MMLU and PIQA results reported in the main paper.

For the two-model setting, Figure A1 shows Base and RF in the upper accuracy region at moderate-to-high costs on all four tasks. Their advantage is clearest on MathQA and MedMCQA, while both variants remain stable near 60% accuracy on LogiQA. SocialIQA is the closest comparison: Margin Sampling is competitive at intermediate costs, but CAUC retains the highest endpoint. Thus, the selected results reflect the broader accuracy–cost frontier.

## D Method-Specific Setup Time

<table><tr><td>Category</td><td>Method</td><td>Setup Time (s) ↓</td></tr><tr><td rowspan="3">Router</td><td>RouterDC</td><td>4,891</td></tr><tr><td>Benchmark Routing RouteLLM</td><td>156 161</td></tr><tr><td>GraphRouter</td><td>449</td></tr><tr><td rowspan="3">Cascade</td><td>EmbedLLM</td><td>841</td></tr><tr><td>FrugalGPT</td><td>3,777</td></tr><tr><td>Margin Sampling Post-Hoc-Embed</td><td>16 209</td></tr><tr><td rowspan="2">CAUC (Ours)</td><td>Base</td><td></td></tr><tr><td>RF</td><td>12 20</td></tr></table>

Table A2: Ofline runtime of method-specific policy construction. Time includes router training or cascade calibration, fitting, and threshold selection as required by each method. Base-model forward passes and deployment inference are excluded. Bold marks the lowest setup time.

Table A2 isolates the overhead of constructing the routing or cascade policy from the cost of running the underlying models. Base has the lowest setup time at 12 seconds, while RF requires 20 seconds; the additional time comes from selecting its budget-dependent operating policy. Margin Sampling is the closest baseline at 16 seconds. The learned routers require 156–4,891 seconds, making Base and RF respectively 13.0–407.6× and 7.8–244.6× faster than this group. The cascade baselines span 16–3,777 seconds: lightweight score thresholding is inexpensive, whereas methods with broader search or fitting procedures incur substantially larger ofline cost. CAUC therefore adds little method-specific preparation beyond calibration and, for RF, development-set policy selection.

## E Generalized Guarantee for Unified Thresholds

The second theoretical proposition in the main paper assumes that the fallback loss is constant. Here we define its two risks formally and extend the result to confidence-dependent fallback quality. The main-paper bound follows as a special case.

## Setup and Risk Definitions

Let $P _ { s } \in [ 0 , 1 ]$ be the small model’s calibrated confidence, let $C _ { s }$ indicate whether its prediction is correct, and define

$$
\eta _ { s } ( p ) = \mathrm { P r } ( C _ { s } = 1 \mid P _ { s } = p ) , \qquad \ell _ { s } ( p ) = 1 - \eta _ { s } ( p ) .\tag{A1}
$$

The fallback is the fixed policy applied after deferral, such as the large-model prediction or a predetermined fusion rule. If its conditional error is $r _ { \mathrm { f b } } ( p )$ , its cost-sensitive loss is

$$
\begin{array} { r } { L _ { \mathrm { f b } } ( p ) = r _ { \mathrm { f b } } ( p ) + \lambda c _ { l } , } \end{array}\tag{A2}
$$

where $c _ { l }$ is the incremental large-model cost and $\lambda \geq 0$ is its loss weight.

Fix a reference loss $\bar { L } \in [ 0 , 1 ]$ and let $\bar { \tau } = 1 - \bar { L } .$ . The unified-threshold policy accepts the small model when $P _ { s } \ge$ τ¯. Its risk is

$$
\mathcal { R } _ { \mathrm { t h r e s h o l d } } = \mathbb { E } [ \ell _ { s } ( P _ { s } ) \mathbf { 1 } \{ P _ { s } \geq \bar { \tau } \} + L _ { \mathrm { f b } } ( P _ { s } ) \mathbf { 1 } \{ P _ { s } < \bar { \tau } \} ] .\tag{A3}
$$

We compare it with a score-aware oracle that knows the two conditional loss functions but, like the threshold policy, may use only $P _ { s }$ to choose between acceptance and fallback. It does not observe the true label of an individual example. Its risk is

$$
\mathcal { R } _ { \mathrm { o r a c l e } } = \mathbb { E } [ \operatorname* { m i n } \{ \ell _ { s } ( P _ { s } ) , L _ { \mathrm { f b } } ( P _ { s } ) \} ] .\tag{A4}
$$

## Performance Guarantee

## Proposition A1 (Unified-threshold performance gap)

$$
\begin{array} { r } { \epsilon _ { s } = \mathbb { E } | \eta _ { s } ( P _ { s } ) - P _ { s } | , } \\ { \Delta _ { \mathrm { f b } } ( \bar { L } ) = \mathbb { E } | L _ { \mathrm { f b } } ( P _ { s } ) - \bar { L } | . } \end{array}\tag{A5}
$$

Then

$$
0 \leq \mathcal { R } _ { \mathrm { t h r e s h o l d } } - \mathcal { R } _ { \mathrm { o r a c l e } } \leq \epsilon _ { s } + \Delta _ { \mathrm { f b } } ( \bar { L } ) .\tag{A6}
$$

Proof. For a fixed confidence $p ,$ define the true and surrogate diferences between the acceptance and fallback losses as

$$
\begin{array} { l } { { d ( p ) = 1 - \eta _ { s } ( p ) - { \cal L } _ { \mathrm { f b } } ( p ) , } } \\ { { \widehat { d } ( p ) = 1 - p - \bar { \cal L } . } } \end{array}\tag{A7}
$$

The oracle accepts when $d ( p ) \leq 0$ , whereas the threshold policy accepts when $\widehat { d } ( p ) \leq 0$ , equivalently when $p \geq \bar { \tau }$ . Let $\rho ( p )$ denote the threshold policy’s conditional excess loss. If the two policies agree, then $\rho ( p ) = 0$ . If they disagree, then $d ( p ) { \widehat { d } } ( p ) \leq 0$ and $\rho ( p ) = | d ( p ) |$ , which implies

$$
\begin{array} { l } { \rho ( p ) \leq | d ( p ) - \widehat { d } ( p ) | } \\ { \leq | \eta _ { s } ( p ) - p | + | L _ { \mathrm { f b } } ( p ) - \bar { L } | . } \end{array}\tag{A8}
$$

Taking expectation over $P _ { s }$ proves the upper bound. Nonnegativity follows because the oracle minimizes conditional loss at every confidence value. □

## Relation to the Main-Paper Result

The main paper considers

$$
L _ { \mathrm { f b } } ( p ) = \bar { L } = L _ { \mathrm { l a r g e } } = r _ { l } + \lambda c _ { l } .\tag{A9}
$$

Then $\bar { \tau } = 1 - L _ { \mathrm { l a r g e } } = \tau ^ { \star }$ and $\Delta _ { \mathrm { f b } } ( \bar { L } ) = 0$ . Equations A3 and A4 therefore give the precise meanings of $\mathcal { R } _ { \mathrm { t h r e s h o l d } }$ and $\mathcal { R } _ { \mathrm { o r a c l e } }$ in the main paper, while Proposition A1 reduces to

$$
\mathcal { R } _ { \mathrm { t h r e s h o l d } } - \mathcal { R } _ { \mathrm { o r a c l e } } \leq \epsilon _ { s } .\tag{A10}
$$

Under perfect calibration, the threshold rule is consequently optimal among stopping policies based only on $P _ { s }$

For direct large-model fallback with $\lambda = 0 , \tau ^ { \star } = 1 - r _ { l } =$ $A _ { l } ,$ and CAUC substitutes the calibration-set estimate $\widehat { A } _ { l } .$ If the fallback instead uses fusion, $\Delta _ { \mathrm { f b } } ( \bar { L } )$ captures variation and mismatch in its conditional quality. Thus, a unified threshold is near-optimal when both calibration error and fallback variation are small. This result concerns only the stopping decision; it does not establish optimality of the fusion rule or the recursive CAUC-RF extension, nor does it include finite-sample estimation error.