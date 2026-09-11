# ActMap: Single-Pass Uncertainty Quantification from Generation-Time Activation Maps

Jacopo Dardini

Roberta Calegari

University of Bologna, Italy

## Abstract

Practical uncertainty quantification (UQ) for large language models must decide, from a single generation, whether a specific answer should be trusted. Existing methods either sample multiple generations, read only output-token probabilities, or reduce the model’s internal computation to a single hidden state. We introduce ActMap, a white-box representation that compresses the generation-time hidden-state trajectory (every layer, every generated token) into a fixed $1 2 \times 3 2 \times 1 2 8$ tensor of temporal-statistic channels that preserves structure across transformer depth and pooled hidden coordinates. The map is captured during the generation pass with no measurable overhead, has a fixed shape across model depths and hidden sizes, and occupies 96 KiB: a compact artifact that can be retained for audit-relevant generations and probed directly, with occlusion analysis localizing the classifier’s signal to mid-depth regions of the map. A lightweight classifier, instantiated as a compact Vision Transformer, reads an estimated correctness probability from each map in a fraction of a millisecond; capacity-matched MLPs perform comparably, indicating the representation itself carries the result. Trained and evaluated in-domain on short-answer QA, direct-answer math, and summarization factuality with three instruction-tuned 7–8B models, ActMap consistently outperforms sampling, token-probability, attention, and embedding baselines, and matches ACT-ViT, a detector trained on dense activation tensors 67× larger, at essentially the same mean AUROC with lower calibration error on ten of twelve pairs. The resulting score supports abstention, routing, and selective verification from a single generation, making it a practical primitive for scalable oversight ofdeployed models.

## Intr<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n

Large language models answer questions, solve problems, and summarize documents with fluent confidence whether or not they are right. Deployments that act on these outputs need a per-answer reliability signal: a score that says, for this specific generation, how likely it is to be correct. This answer-level uncertainty quantification (UQ) problem, detecting hallucinated or otherwise unreliable answers, is increasingly the gatekeeper for abstention, retrieval fallback, escalation, and human review (Kadavath et al., 2022; Farquhar et al., 2024), and the signal that decides where scarce verification efort goes when human oversight cannot scale to every generation (Bowman et al., 2022).

Existing UQ methods occupy three regimes, each with a structural gap. Sampling-based black-box methods such as semantic entropy measure disagreement across multiple generations (Kuhn et al., 2023; Farquhar et al., 2024); they capture meaning-level ambiguity but require N sampled generations per query, which raises total decoding compute and throughput requirements even when parallel sampling hides latency. Grey-box methods read output-token probabilities (perplexity, mean token entropy, or learned functions of the output distribution (Bar-Shalom et al., 2026)) from a single pass, but see only the final projection of the model’s computation. White-box methods look inside the model, yet most reduce the internal state to a single vector: a probe on the last token’s hidden state (Azaria and Mitchell, 2023; Marks and Tegmark, 2024), a pooled sentence representation, or aggregated logits at one self-evaluation position (Xiao et al., 2026). Between “one vector” and “ten regenerations” lies almost everything the model computed while producing the answer: how representations difer across depth and across the hidden space, and how they evolve over generated tokens; many existing methods collapse or omit much of this structure.

We propose ActMap, a representation designed to keep it. During a single generation, ActMap records the hidden state of every transformer layer at every generated token, then compresses this variable-size $L \times T \times D$ trajectory into a fixed 12 × 32 × 128 tensor: twelve channels of temporal statistics over the token axis, with adaptive pooling mapping the layer and hidden-dimension axes to fixed sizes. The result is compact (96 KiB per generation), fixed in shape across model depths and hidden sizes, and structured: rows index transformer depth, columns index pooled hidden coordinates, and channels are aligned temporal statistics. A learned classifier, in our main experiments a compact Vision Transformer, maps each tensor to an estimated correctness probability. The classifier never sees the generated text or output token probabilities: correctness is predicted from the internal state alone, with no extra model calls.

Our primary setting is in-domain deployment: an operator serves a fixed model on a fixed task, labels a set of generations once, and then assigns every subsequent answer a single correctness score, on which downstream decisions such as abstention, escalation, routing, or selective verification can be thresholded. Our main contributions are as follows:

• Re<sub>p</sub>resentation. A fixed-size, multi-channel activation-map summary of the full generation trajectory (layers, generated tokens, hidden dimensions, and activation dynamics), computable during one generation pass for any decoder-only transformer; ablations show robustness to the classifier choice.

• Method. A single-pass supervised UQ method with negligible added inference cost: capture is not measurably slower than plain decoding, and scoring is one forward pass of a 2.4M-parameter classifier.

• Evaluation. A unified comparison against eight baselines organized in an explicit black-/grey-/white-box taxonomy, on four tasks and three open-weight 7–8B models under a shared balanced protocol. The comparison includes ACT-ViT, evaluated with its complete published per-pair architecture sweep; ActMap reaches essentially the same mean AUROC from a 67× smaller representation with a single fixed classifier configuration.

• Anal<sub>y</sub>sis. Ablations and controls that identify crosslayer, pooled-coordinate structure as the primary source of predictive signal; an occlusion analysis of where the classifier draws that signal; transfer experiments across datasets, tasks, generators, and model scale (with clearly reported near-chance results); and calibration, selectiveprediction, and cost analyses.

## R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>o</sub>rk

Sampling-based black-box UQ. Semantic entropy samples several answers to the same query, clusters them by bidirectional entailment, and computes entropy over the resulting meaning classes (Kuhn et al., 2023; Farquhar et al., 2024). It needs no access to model internals, but each scored query requires N sampled generations (here N=10) plus entailment inference.

Token-probability (grey-box) methods. Sequence perplexity and mean token entropy summarize the output distribution of a single generation; P(True)-style self-evaluation elicits the probability the model assigns to its own answer being correct (Kadavath et al., 2022). LOS-Net learns a detector over the full sequence of next-token distributions (Bar-Shalom et al., 2026), evidence that learned functions of output distributions can outperform hand-crafted statistics. All of these observe only the model’s final projection onto the vocabulary.

Single-vector white-box probes. Linear probes on one hidden state can recover truthfulness information (Azaria and Mitchell, 2023; Marks and Tegmark, 2024), and layeraggregated logits at a self-evaluation token improve calibration (Xiao et al., 2026). These approaches collapse the token axis entirely (one position) or the layer axis (one pooled vector), assuming the reliability signal is localized.

Structured white-box methods. EigenScore measures the diferential entropy of sampled-response embeddings in internal space (Chen et al., 2024); RAUQ aggregates attention through uncertainty-relevant heads, unsupervised, in a single pass (Vazhentsev et al., 2026); TAD learns attentionbased features of conditional dependency between generation steps (Vazhentsev et al., 2025). ACT-ViT is the closest representation-level comparison: it trains a vision transformer directly over padded layer-by-token activation tensors (Bar-Shalom et al., 2025). Unlike methods that select attention statistics or embedding geometry, ACT-ViT and ActMap both preserve joint depth and token structure. They difer in what is retained: ACT-ViT keeps a dense tensor whose width tracks the generator’s hidden size, reads at most the first 100 generated tokens, and couples the detector to the generator through a model-specific adapter; ActMap compresses the full trajectory into a fixed-shape map that is identical in geometry for every generator.

Summarization factuality. Summary factuality is judged against the source document and is known to be graded and multi-dimensional rather than binary (Maynez et al., 2020). We use MiniCheck sentence-level verification for CNN/DailyMail (Tang et al., 2024) and assign one factual or non-factual label to each summary. This supports the same correctness-prediction interface for long-form generation, although summaries near the label boundary remain dificult.

Positioning. ActMap keeps what each family discards: it scores the single produced answer from one pass (vs. sampling), reads the pre-projection trajectory (vs. grey-box), preserves the depth axis and summarizes evolution over generated tokens (vs. single-vector probes), and exposes a broad structured summary for a learned classifier to mine rather than committing to one signal (Elhage et al., 2022). Viewed as a monitor, ActMap extends work that reads safety-relevant signals directly from internal activations (Burns et al., 2023). Table 2 places every baseline in this taxonomy.

## M<sub>e</sub>th<sub>o</sub>d

## Generation-Time Trajectory

Let a decoder-only transformer with L layers and hidden size D generate an answer of T tokens. During decoding we record, via forward hooks on every layer, the hidden state of each newly generated token, giving a trajectory

$$
H \in \mathbb { R } ^ { L \times T \times D } ,\tag{1}
$$

where $H _ { \ell , t }$ is the layer-ℓ representation of generated token t. To bound memory, each hidden vector is reduced online from D to $D ^ { \prime } { = } 1 2 8$ coordinates by contiguous adaptive average pooling (each pooled coordinate is the mean of a fixed contiguous block of hidden dimensions), so the stored trajectory is $L \times T \times 1 2 8$ . Generation itself is unchanged: one decoding pass in vLLM (Kwon et al., 2023), no extra samples.

## A<sub>c</sub>ti<sub>va</sub>ti<sub>o</sub>n<sub>-</sub>M<sub>ap</sub> C<sub>o</sub>n<sub>s</sub>tr<sub>uc</sub>ti<sub>o</sub>n

The trajectory varies in T (answers have diferent lengths) and, across models, in L and D. ActMap converts it into a fixed tensor $M \in \bar { \mathbb { R } } ^ { C \times L ^ { \prime } \times D ^ { \prime } }$ with C=12, L<sup>′</sup>=32, D<sup>′</sup>=128 in three steps (Figure 1).

(1) Temporal statistics. The token axis is summarized by twelve channels, each a function $\mathbb { R } ^ { T } $ R applied independently at every (layer, pooled-coordinate) location: segment means (4: means over the four consecutive quarters of the answer, locating activation mass in answer time), final states (2: last-token state and mean over the final eight tokens, the information a last-token probe would see), dispersion (2: standard deviation and maximum over tokens), drift and slope (2: last-minus-first diference and least-squares slope against token index), and magnitude and dynamics (2: perlayer RMS norm, broadcast over coordinates, and mean absolute token-to-token diference). Every channel is a statistic over the whole token axis, so the output is independent of T by construction; one-token answers set the dispersion, slope, and dynamics channels to zero.

(2) Layer pooling. The layer axis is adaptively averagepooled to L<sup>′</sup>=32 rows, aligning models with diferent depths $( L \in \{ 3 2 , 3 6 , 6 4 \}$ here) onto a common axis while preserving depth ordering.

(3) Normalization. Each channel is standardized to zero mean and unit variance over its $3 2 \times 1 2 8$ entries, so the classifier sees spatial patterns within a channel rather than raw magnitudes, and channels are on a common scale. Maps are stored in float16 (96 KiB per generation).

We deliberately avoid interpreting individual cells; the representation’s role is to preserve where in depth, where in the pooled hidden space, and when in answer time activity difers between correct and incorrect generations.

## C<sub>o</sub>rr<sub>ec</sub>tn<sub>ess</sub> Cl<sub>ass</sub>ifi<sub>e</sub>r

The detector $f _ { \theta }$ maps a tensor M to an estimated correctness probability $p ( \mathrm { c o r r e c t } ) = \sigma ( f _ { \theta } ( M ) )$ ; the uncertainty score of a generation is $u = 1 - p ( \mathrm { c o r r e c t } )$ ). We instantiate $f _ { \theta }$ as a compact Vision Transformer (Dosovitskiy et al., 2021) over $4 \times 1 6$ patches of the map (2.4M parameters; six pre-norm blocks, embedding width 192, six heads, a class token) with factorized row/column positional embeddings, so that depth and coordinate identity are preserved; remaining details appear in the appendix. The representation is not tied to this choice (see Ablations); all main-table results use the Vision Transformer.

Table 1: Datasets, pre-balancing split sizes, and label types.
<table><tr><td>Dataset</td><td>Rows</td><td>Test t Label</td><td></td></tr><tr><td>TriviaQA (no ctx.)</td><td>50,000</td><td>5,051</td><td>answer correctness</td></tr><tr><td>NQ-Open</td><td>50,000</td><td>5,064</td><td>answer correctness</td></tr><tr><td>GSM8K (direct)</td><td>8,792</td><td>879</td><td>numeric correctness</td></tr><tr><td>CNN/DailyMail</td><td>50,000</td><td>5,000</td><td>summary factuality</td></tr></table>

Training minimizes binary cross-entropy on maps with binary correctness labels, using AdamW (learning rate $1 0 ^ { - 3 }$ weight decay 0.05), cosine decay with 5 warm-up epochs, at most 80 epochs with early stopping on validation AUROC (patience 20), batch size 64, Gaussian input noise $_ { ( \sigma = 0 . 0 8 ) }$ and mixup (α=0.2), and three seeds {42, 123, 456}. Reported probabilities are raw sigmoid outputs; deployment at a base rate diferent from the training distribution may require prior correction (see Calibration). Scoring a stored map is a single forward pass of the small network; no additional LLM call is made.

## Ex<sub>p</sub>erimental Setu<sub>p</sub>

## T<sub>as</sub>k<sub>s,</sub> M<sub>o</sub>d<sub>e</sub>l<sub>s, a</sub>nd G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n

We evaluate on four datasets covering three task families and two output-length regimes (Table 1): TriviaQA (no context) and NQ-Open for short-answer factual QA (Joshi et al., 2017; Kwiatkowski et al., 2019), GSM8K for mathematical problem solving in direct-answer mode (Cobbe et al., 2021), and CNN/DailyMail for long-form summarization factuality (See et al., 2017). Generators are Qwen3-8B (Qwen Team, 2025), Llama-3.1-8B-Instruct (Llama Team, AI @ Meta, 2024), and Mistral-7B-Instruct-v0.3 (Jiang et al., 2023); Qwen3-32B (64 layers) is used for an in-domain and model-scale transfer study on the three short-form tasks. Primary answer generation is greedy (temperature 0) in vLLM with a 4,096-token context, 32 new tokens for short-answer tasks and GSM8K (direct answer, no explicit reasoning), and 384 for summaries. In total, the study produces more than 476,000 generations with captured trajectories.

## L<sub>a</sub>b<sub>e</sub>l<sub>s a</sub>nd S<sub>p</sub>lit<sub>s</sub>

For TriviaQA and NQ-Open, a response is correct if its normalized answer exactly matches a normalized gold alias or attains token F1 of at least 0.8 against any alias; GSM8K uses exact numeric match. CNN/DailyMail summaries are split into sentences and each sentence is verified against the source article with MiniCheck (Flan-T5-Large); a summary is labeled factual if every sentence is supported at threshold 0.5. Splits are disjoint by source key. Because per-model accuracy varies, all supervised training and all reported metrics use per-(dataset, model, split) balanced indices with equal numbers of correct and incorrect generations. All supervised detectors (ActMap, TAD, and ACT-ViT) train on the same balanced training rows, use the same balanced validation rows for their respective model-selection protocols, and are evaluated on the identical balanced test rows.

![](images/4c589d2ce1e3736e97f65597eac20999d21639867fb88beef525bfeae45cb77c.jpg)  
Figure 1: The ActMap pipeline. The LLM generates its answer normally while hooks capture the per-layer, per-token hiddenstate trajectory; temporal statistics, pooling, and per-channel standardization compress it into a fixed 12 × 32 × 128 map, from which a compact Vision Transformer classifier reads p(correct). Channels shown: token standard deviation, temporal slope, first segment mean (real Qwen3-8B GSM8K generation).

## B<sub>ase</sub>li<sub>ne</sub> T<sub>axonomy</sub>

Table 2 summarizes all methods. The black-box baseline is Semantic Entropy (10 sampled generations at temperature 1.0, top-p 0.9, clustered by bidirectional DeBERTa-MNLI entailment). Grey-box baselines are sequence perplexity, mean token entropy (MTE), and P(True) self-evaluation. White-box baselines are TAD and ACT-ViT (supervised, like ActMap), RAUQ, and EigenScore (over 10 sampledresponse embeddings). We follow each baseline’s published protocol; for ACT-ViT this includes the authors’ full 24- configuration architecture sweep on the shared balanced splits and seeds (details in the appendix).

## M<sub>e</sub>t<sub>r</sub>i<sub>cs an</sub>d E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> A<sub>xes</sub>

We report AUROC (↑; ranking quality across all thresholds), AUPRC (↑; precision–recall performance, with a 0.5 baseline on the balanced splits), and 10-bin expected calibration error (ECE, ↓; the gap between predicted confidence and empirical correctness) (Guo et al., 2017). Supervised detectors report the mean over seeds {42, 123, 456}; the maximum seed standard deviation of ActMap AUROC is 0.026 (GSM8K, smallest split) and below 0.010 on all splits above 1,700 test rows. The evaluation separates (i) in-domain performance, training and testing on the same (dataset, model); (ii) cross-dataset transfer within a model; (iii) cross-task transfer between short-form correctness and long-form factuality; (iv) cross-generator transfer; and (v) model-scale transfer (Qwen3-8B↔32B).

## M<sub>a</sub>i<sub>n</sub> R<sub>esu</sub>lt<sub>s</sub>

Table 3 reports the in-domain matrix over all (model, dataset, method) triplets.

ActMap leads every non-tensor baseline. ActMap attains the highest AUROC and AUPRC on all twelve (model, dataset) pairs against the sampling, token-probability, attention, and embedding baselines. Its mean AUROC over those twelve pairs is .825, against .790 for TAD and .741 for the best training-free baseline (MTE).

Baseline categories behave consistently. Grey-box statistics are strong on short-answer QA, where a wrong answer usually coincides with a difuse output distribution, but weaker on direct-answer mathematical correctness, where an incorrect numeric answer may still be produced with a concentrated token distribution: the best grey-box AUROC on GSM8K trails ActMap by .031–.087. Sampling-based Semantic Entropy pays for its ten extra generations without matching single-pass grey-box statistics on these balanced splits, consistent with its score reflecting question ambiguity rather than answer-specific reliability. Among the nontensor white-box baselines, supervised TAD is consistently the strongest; unsupervised RAUQ and EigenScore sit between grey-box statistics and the supervised methods.

Long-form factuality is the hardest regime. On CNN/DailyMail, absolute scores drop for all methods. Summary factuality is graded: a mostly supported summary may contain one unsupported clause, making factual/non-factual labels dificult near the boundary. Training-free methods are barely better than chance here, and Semantic Entropy is at chance, since whole-response equivalence clustering is poorly matched to long-form output; only the supervised methods extract usable signal, with ActMap ahead of TAD on all three models, within .012 AUROC of ACT-ViT on every model, and keeping ECE below .05 throughout.

## C<sub>o</sub>m<sub>p</sub>r<sub>ess</sub>i<sub>o</sub>n Pr<sub>ese</sub>r<sub>ves</sub> th<sub>e</sub> D<sub>e</sub>n<sub>se-</sub>T<sub>e</sub>n<sub>so</sub>r Si<sub>g-</sub> <sub>na</sub>l

We evaluate ACT-ViT with its complete published 24- configuration sweep per pair rather than as a fixed-score baseline. Ranking quality is nearly identical: ActMap leads on seven of twelve pairs (one decided beyond three decimals) and ACT-ViT on five. Mean AUROC is .825 versus .823; the largest gaps are comparable (.048 and .044).

Table 2: Method taxonomy. “Extra gen.” counts generated responses beyond the scored answer. Per-answer latencies, which are specific to our implementation and hardware stack, are reported in the appendix.
<table><tr><td>Method</td><td>Access</td><td>Extra gen.</td><td>Information used</td><td>Supervision</td><td>Stored artifact</td></tr><tr><td>Semantic Entropy (Kuhn et al., 2023)</td><td>black-box</td><td>10</td><td>sampled answer texts</td><td>none</td><td>samples</td></tr><tr><td>Perplexity</td><td>grey-box</td><td>0</td><td>output token probabilities</td><td>none</td><td></td></tr><tr><td>Mean token entropy</td><td>grey-box</td><td>0</td><td>output token distributions</td><td>none</td><td></td></tr><tr><td>P(True) (Kadavath et al., 2022)</td><td>grey-box</td><td>1 short pass</td><td>self-evaluation token probability</td><td>none</td><td></td></tr><tr><td>TAD (Vazhentsev et al., 2025)</td><td>white-box</td><td>0</td><td>attention + token probabilities</td><td>trained</td><td>attn. features</td></tr><tr><td>RAUQ (Vazhentsev et al., 2026)</td><td>white-box</td><td>0</td><td>attention + token probabilities</td><td>none</td><td>attn. stats</td></tr><tr><td>EigenScore (Chen et al., 2024)</td><td>white-box</td><td>10</td><td>sampled-response embeddings</td><td>none</td><td>embeddings</td></tr><tr><td>ACT-ViT (Bar-Shalom et al., 2025)</td><td>white-box</td><td>0</td><td>padded activation tensor</td><td>trained</td><td>activations</td></tr><tr><td>ACTMAP (ours)</td><td>white-box</td><td>0</td><td>full hidden-state trajectory</td><td>trained</td><td>96 KiB map</td></tr></table>

Table 3: In-domain results on balanced test splits (AUROC ↑ / AUPRC ↑ / ECE ↓; mean over three seeds for trained methods). Bold marks the best value for that pair. ECE for methods without native probabilities uses a fixed monotone score-to-probability mapping. CNN/DailyMail EigenScore covers the label subset with sampled-response embeddings.
<table><tr><td>Model</td><td>Method</td><td colspan="3">TriviaQA</td><td colspan="3">NQ-Open</td><td colspan="3">GSM8K</td><td colspan="3">CNN/DM</td></tr><tr><td></td><td></td><td>AUROC</td><td>AUPRC</td><td>ECE</td><td></td><td>AUROC AUPRC ECE</td><td></td><td>AUROC</td><td>AUPRC ECE</td><td></td><td>AUROC</td><td>AUPRC ECE</td><td></td></tr><tr><td>Qwen3-8B</td><td>ACTMAP (ours)</td><td>.887</td><td>.883</td><td>.028</td><td>.838</td><td>.802</td><td>.057</td><td>.791</td><td>.781</td><td>.126</td><td>.705</td><td>.692</td><td>.033</td></tr><tr><td></td><td>ACT-ViT</td><td>.861</td><td>.856</td><td>.059</td><td>.855</td><td>.827</td><td>.088</td><td>.835</td><td>.844</td><td>.100</td><td>.717</td><td>.703</td><td>.083</td></tr><tr><td></td><td>Semantic Entropy</td><td>.772</td><td>.678</td><td>.397</td><td>.752</td><td>.682</td><td>.382</td><td>.719</td><td>.651</td><td>.359</td><td>.510</td><td>.516</td><td>.367</td></tr><tr><td></td><td>Perplexity</td><td>.837</td><td>.836</td><td>.116</td><td>.761</td><td>.753</td><td>.079</td><td>.744</td><td>.753</td><td>.134</td><td>.579</td><td>.569</td><td>.028</td></tr><tr><td></td><td>Mean token entropy</td><td>.847</td><td>.841</td><td>.186</td><td>.760</td><td>.751</td><td>.116</td><td>.755</td><td>.758</td><td>.142</td><td>.582</td><td>.570</td><td>.061</td></tr><tr><td></td><td>P(True)</td><td>.817</td><td>.773</td><td>.367</td><td>.729</td><td>.691</td><td>.396</td><td>.685</td><td>.704</td><td>.364</td><td>.534</td><td>.506</td><td>.497</td></tr><tr><td></td><td>TAD</td><td>.837</td><td>.814</td><td>.126</td><td>.823</td><td>.782</td><td>.305</td><td>.761</td><td>.734</td><td>.118</td><td>.684</td><td>.668</td><td>.058</td></tr><tr><td></td><td>RAUQ</td><td>.813</td><td>.771</td><td>.329</td><td>.792</td><td>.740</td><td>.335</td><td>.750</td><td>.692</td><td>.335</td><td>.583</td><td>.579</td><td>.357</td></tr><tr><td></td><td>EigenScore</td><td>.742</td><td>.694</td><td>.124</td><td>.730</td><td>.674</td><td>.140</td><td>.714</td><td>.718</td><td>.157</td><td>.573</td><td>.555</td><td>.469</td></tr><tr><td>Llama-3.1-8B AcrMAP (ours)</td><td></td><td>.880</td><td>.878</td><td>.039</td><td>.861</td><td>.840</td><td>.066</td><td>.931</td><td>.939</td><td>.086</td><td>.769</td><td>.754</td><td>.044</td></tr><tr><td></td><td>ACT-ViT</td><td>.832</td><td>.833</td><td>.072</td><td>.848</td><td>.823</td><td>.098</td><td>.912</td><td>.906</td><td>.061</td><td>.766</td><td>.753</td><td>.103</td></tr><tr><td></td><td>Semantic Entropy</td><td>.802</td><td>.744</td><td>.386</td><td>.758</td><td>.719</td><td>.368</td><td>.575</td><td>.582</td><td>.228</td><td>.483</td><td>.488</td><td>.418</td></tr><tr><td></td><td>Perplexity</td><td>.830</td><td>.834</td><td>.169</td><td>.795</td><td>.793</td><td>.150</td><td>.812</td><td>.772</td><td>.177</td><td>.576</td><td>.560</td><td>.054</td></tr><tr><td></td><td>Mean token entropy</td><td>.844</td><td>.843</td><td>.171</td><td>.796</td><td>.792</td><td>.155</td><td>.827</td><td>.793</td><td>.233</td><td>.579</td><td>.559</td><td>.111</td></tr><tr><td></td><td>P(True)</td><td>.792</td><td>.780</td><td>.316</td><td>.707</td><td>.707</td><td>.274</td><td>.844</td><td>.867</td><td>.079</td><td>.557</td><td>.542</td><td>.405</td></tr><tr><td></td><td>TAD</td><td>.849</td><td>.839</td><td>.099</td><td>.824</td><td>.798</td><td>.214</td><td>.845</td><td>.819</td><td>.076</td><td>.732</td><td>.716</td><td>.079</td></tr><tr><td></td><td>RAUQ</td><td>.827</td><td>.799</td><td>.350</td><td>.817</td><td>.782</td><td>.355</td><td>.770</td><td>.684</td><td>.383</td><td>.580</td><td>.565</td><td>.366</td></tr><tr><td></td><td>EigenŠcore</td><td>.713</td><td>.692</td><td>.125</td><td>.782</td><td>.754</td><td>.181</td><td>.535</td><td>.527</td><td>.298</td><td>.606</td><td>.587</td><td>.384</td></tr><tr><td>Mistral-7B</td><td>ACTMAP (ours)</td><td>.859</td><td>.851</td><td>.048</td><td>.905</td><td>.877</td><td>.041</td><td>.782</td><td>.813</td><td>.146</td><td>.689</td><td>.671</td><td>.039</td></tr><tr><td></td><td>ACT-ViT</td><td>.885</td><td>.881</td><td>.066</td><td>.911</td><td>.878</td><td>.124</td><td>.769</td><td>.763</td><td>.159</td><td>.689</td><td>.664</td><td>.077</td></tr><tr><td></td><td>Semantic Entropy</td><td>.711</td><td>.614</td><td>.445</td><td>.777</td><td>.699</td><td>.410</td><td>.728</td><td>.673</td><td>.288</td><td>.495</td><td>.510</td><td>.413</td></tr><tr><td></td><td>Perplexity</td><td>.777</td><td>.763</td><td>.070</td><td>.853</td><td>.857</td><td>.050</td><td>.751</td><td>.745</td><td>.142</td><td>.516</td><td>.514</td><td>.041</td></tr><tr><td></td><td>Mean token entropy</td><td>.787</td><td>.767</td><td>.144</td><td>.850</td><td>.854</td><td>.172</td><td>.750</td><td>.745</td><td>.131</td><td>.517</td><td>.513</td><td>.089</td></tr><tr><td></td><td>P(True)</td><td>.774</td><td>.730</td><td>.390</td><td>.731</td><td>.663</td><td>.426</td><td>.575</td><td>.622</td><td>.504</td><td>.577</td><td>.556</td><td>.500</td></tr><tr><td></td><td>TAD</td><td>.788</td><td>.748</td><td>.141</td><td>.891</td><td>.856</td><td>.337</td><td>.771</td><td>.768</td><td>.153</td><td>.676</td><td>.661</td><td>.082</td></tr><tr><td></td><td>RAUQ</td><td>.761</td><td>.724</td><td>.329</td><td>.879</td><td>.858</td><td>.341</td><td>.757</td><td>.759</td><td>.343</td><td>.521</td><td>.518</td><td>.363</td></tr><tr><td></td><td>EigenScore</td><td>.681</td><td>.676</td><td>.418</td><td>.855</td><td>.828</td><td>.298</td><td>.597</td><td>.585</td><td>.433</td><td>.556</td><td>.541</td><td>.208</td></tr></table>

ActMap reaches this parity from 49,152 values per generation against the 3.3M values of ACT-ViT’s dense 8×100×4096 tensor, with one fixed classifier for all pairs while the sweep selects diferent architectures for diferent pairs, and with lower ECE on ten of twelve pairs (mean .063 vs. .091). These results indicate that temporal-statistic compression preserves the uncertainty signal in the dense activation tensor while producing a fixed, generator-agnostic map.

## Tr<sub>a</sub>n<sub>s</sub>f<sub>e</sub>r <sub>a</sub>nd G<sub>e</sub>n<sub>e</sub>r<sub>a</sub>li<sub>za</sub>ti<sub>o</sub>n

A detector is deployed either in-domain, trained for a fixed generator and task (the previous section), or in transfer, where target labels are unavailable and a detector trained elsewhere must generalize. We quantify the latter along four axes with the classifier frozen after source training (Table 4).

Within-task transfer is strong: TriviaQA → NQ-Open stays close to matched in-domain training and ahead of the frozen TAD baseline, though TAD transfers with better calibration. Beyond the task boundary the picture changes: transfer from pooled QA to GSM8K is weak and model-dependent, and both directions between short-answer correctness and summary factuality are near chance. Cross-generator transfer trains on the pooled maps of two generators with the dataset held fixed and evaluates on the held-out third (a macro mean over four datasets, three targets, and three seeds); it is near chance. Model scale behaves the same way: transfer between Qwen3-8B and Qwen3-32B is near chance in both directions despite the shared model family and tokenizer, while a detector trained in domain on Qwen3-32B is at least as strong as at 8B (macro .868 AUROC, with the largest gain on GSM8K, .791 → .870). These results concern transfer of the learned decision boundary. Shared map geometry alone does not align boundaries across tasks, generators, or scales.

Table 4: Frozen-detector transfer (AUROC ↑ / AUPRC ↑ / ECE ↓ on balanced target test splits; no target-domain training). Rows are macro means over their constituent target pairs and three seeds; model-scale and in-domain-scale rows average over the three short-form datasets.
<table><tr><td>Axis</td><td>Source → target</td><td>AUROC AUPRC ECE</td><td></td><td></td></tr><tr><td>Cross-dataset</td><td>TriviaQA → NQ-Open (3 models)</td><td>.837</td><td>.821</td><td>.138</td></tr><tr><td rowspan="3">Cross-task</td><td>frozen TAD, same transfer</td><td>.804</td><td>.766</td><td>.053</td></tr><tr><td>TriviaQA+NQ → GSM8K</td><td>.617</td><td>.645</td><td>.304</td></tr><tr><td>TriviaQA+NQ → CNN/DM</td><td>.512</td><td>.513</td><td>.463</td></tr><tr><td rowspan="4">Model scale</td><td>CNN/DM → TriviaQA+NQ</td><td>.537</td><td>.532</td><td>.256</td></tr><tr><td>Cross-generator leave-one-generator-out (12 targets)</td><td>.510</td><td>.516</td><td>.337</td></tr><tr><td>Qwen3-8B → 32B</td><td>.520</td><td>.518</td><td>.366</td></tr><tr><td>Qwen3-32B → 8B</td><td>.517</td><td>.522</td><td>.403</td></tr><tr><td></td><td>Scale (in-dom.) Qwen3-32B, short-form</td><td>.868</td><td>.860</td><td>.065</td></tr></table>

Table 5: Ablations and controls on TriviaQA × Qwen3-8B (test AUROC, mean over three seeds; seed std ≤ .010 for every variant). Variants requiring re-captured trajectories use the retained class-balanced split intersection, on which the full configuration scores .886 (vs. .887 in Table 3).
<table><tr><td>Configuration</td><td>AUROC</td></tr><tr><td>Full AcTMAP (12 ch., 32 × 128, Vision Transformer)</td><td>.886</td></tr><tr><td>Representation scope</td><td></td></tr><tr><td>last-token channels only</td><td>.879</td></tr><tr><td>single mean-pooled final-layer state</td><td>.754</td></tr><tr><td>Pooling structure (controls)</td><td></td></tr><tr><td>hidden-coordinate permutation before pooling</td><td>.713</td></tr><tr><td>Gaussian projection instead of pooling</td><td>.863</td></tr><tr><td>Construction choices</td><td></td></tr><tr><td>any one channel group removed (worst-best of 5)</td><td>.881-.890</td></tr><tr><td>1 / 8 temporal segments (default 4)</td><td>.889 / .889</td></tr><tr><td>resolution 12 × 16 × 64</td><td>.877</td></tr><tr><td>resolution 12 × 64 × 256</td><td>.886</td></tr><tr><td>global normalization (not per-channel)</td><td>.889</td></tr><tr><td>Classifier on identical maps</td><td></td></tr><tr><td>logistic regression (flattened)</td><td>.877</td></tr><tr><td>MLP (matched parameters)</td><td>.892</td></tr><tr><td>MLP (4× parameters)</td><td>.893</td></tr><tr><td>Training-set size</td><td></td></tr><tr><td>10% / 25% / 50% of train</td><td>.815 / .855 / .875</td></tr><tr><td>Sanity controls</td><td></td></tr><tr><td>permuted labels (expect ≈.5)</td><td>.510</td></tr><tr><td>answer-length-only predictor</td><td>.597</td></tr></table>

A new deployment therefore requires target-domain labels; otherwise the monitor degrades silently under shift, which is itself an oversight risk.

## Abl<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

All ablations run on TriviaQA × Qwen3-8B with the maintable protocol and three seeds (Table 5); they answer four questions.

Where does the gain come from? From the structure preserved across depth and pooled hidden coordinates; no single statistic explains it. Collapsing the map to one mean-pooled final-layer vector, the representation prior white-box probes use, costs .13 AUROC. A map built from the last-token channels alone recovers nearly all of the full map’s performance, as expected on short answers where the final state can summarize the preceding computation. Temporal summaries add little beyond the last-token channels in both output-length regimes (CNN/DailyMail × Qwen3-8B, summaries up to 384 tokens: .704 vs. .705), and no single channel group is critical. The temporal channels are thus a compact mechanism for reducing variable-length trajectories to a fixed shape; the predictive signal lies in the cross-layer, pooled-coordinate structure.

Does the pooling scheme matter? Yes. Permuting hidden coordinates before pooling preserves the marginal activation values but destroys the coordinate grouping; it is the most damaging representation variant, costing more than collapsing the map to a single final-layer vector. Random Gaussian projections of matched size also lose ground. Performance depends on the consistent coordinate grouping induced by contiguous pooling; dimension reduction alone does not preserve the signal.

D<sub>oes pe</sub>rf<sub>o</sub>rm<sub>a</sub>n<sub>ce</sub> d<sub>epe</sub>nd <sub>o</sub>n th<sub>e c</sub>l<sub>ass</sub>ifi<sub>e</sub>r <sub>a</sub>r<sub>c</sub>hit<sub>ec</sub>t<sub>u</sub>r<sub>e</sub>? No: logistic regression, a capacity-matched MLP, and a 4× MLP all perform comparably on identical maps, and the capacity-matched MLP in fact slightly outperforms the Vision Transformer; on this ablation setting, classifier choice has little efect relative to the representation. The remaining construction choices (segment count, map resolution, normalization) shift AUROC by at most .010.

How much supervision is needed? On TriviaQA × Qwen3-8B, 10% of the training data (about 3,800 balanced examples; AUROC .815) beats Semantic Entropy and Eigen-Score; 25% (.855) beats every evaluated training-free baseline. As expected, a detector fit to permuted labels falls to chance, and an answer-length-only predictor stays well below the full map.

## D<sub>ep</sub>th<sub>-</sub>Wi<sub>se</sub> L<sub>oca</sub>li<sub>za</sub>ti<sub>o</sub>n <sub>o</sub>f Pr<sub>e</sub>di<sub>c</sub>ti<sub>ve</sub> Si<sub>g</sub>n<sub>a</sub>l

We localize the predictive signal the classifier uses within the map: on TriviaQA × Qwen3-8B, we occlude one depth-band × coordinate-band region at a time and re-evaluate the frozen detector (baseline .887 AUROC; occluding the entire map collapses it to .500). The classifier is most sensitive to midnetwork depth bands (occlusion drops of .003–.013), while the earliest and latest bands are individually more redundant; integrated-gradients and attention-rollout attributions agree on the same mid-depth concentration (Figure 2). This agreement localizes the detector’s signal without implying an explicit correctness representation in the generator, and is consistent with probing literature placing semantic and truthfulness information in intermediate layers (Azaria and Mitchell, 2023; Marks and Tegmark, 2024).

## C<sub>a</sub>libr<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd S<sub>e</sub>l<sub>ec</sub>ti<sub>ve</sub> Pr<sub>e</sub>di<sub>c</sub>ti<sub>o</sub>n

Abstention thresholds require calibrated probabilities, not only good ranking (Guo et al., 2017). ActMap’s raw sigmoid outputs are the best- or near-best-calibrated on short-answer QA and CNN/DailyMail. The exception is GSM8K, whose training split is an order of magnitude smaller: ActMap’s ECE rises to .086–.146 and ACT-ViT or MTE is better calibrated there (Table 3). Raw-score ECE comparisons favor trained probabilistic detectors by construction. As a diagnostic, we fit one scalar temperature per method on half of the Qwen3-8B × TriviaQA test predictions and report ECE on the other half: ActMap’s optimal temperature is ≈1.0 (ECE .027 → .024), while MTE needs strong sharpening (T=0.50) and still leaves ECE at .158.

![](images/2c3f00e81bc01585866f58e043818929db6b9d75a065f38e6a472d8c3b59f37f.jpg)

![](images/3e05886da2d999ed62172cd05d765678f71b58c519c484be172b6d6d43e19aa2.jpg)

![](images/9ad3355c82a58cec769a5c06fa69c4616ad199a308c7080614740cb8eb89bd90.jpg)

![](images/bc30ab51a55eecbd084e767d10ab8d555f45ed04b3d140b8bcb00a491ffb953b.jpg)  
Figure 2: Occlusion atlas for TriviaQA × Qwen3-8B (mean over three seeds): AUROC drop from zeroing one depth × coordinate band, with integrated-gradients (IG) and attention-rollout views of the same grid. All views concentrate in mid-depth bands.

![](images/8b132d7c48fb5e9a3f82cc2e58295be7e417ff617c397b837fb5e89d5a40e81b.jpg)  
Figure 3: Risk–coverage for Qwen3-8B: selective risk (error among retained answers) vs. coverage on shared balanced test rows. Markers: 80% and 90% coverage.

Balanced splits also difer from deployment prevalence, so we simulate prevalence shift by class-conditionally resampling each balanced test set to the model’s natural test accuracy (100 replicates; temperature fit on 256 held-out target examples each). Where natural accuracy is near the balanced regime (TriviaQA, .53–.66; CNN/DailyMail, .54–.63), raw ECE stays at .02–.09 and selective prediction remains useful (18–38% coverage at 5% risk on TriviaQA). Where accuracy collapses (NQ-Open, .17–.23; Mistral-7B GSM8K, .07), ECE rises to .15–.24 and temperature scaling does not repair it: the error is a prior shift requiring prior correction, not a sharpness error.

The deployment-relevant summary is risk–coverage behavior (Geifman and El-Yaniv, 2017): how much generation volume can be cleared automatically at a target error rate, concentrating human review on the remainder (Figure 3). On Qwen3-8B TriviaQA, ActMap retains 18.3% coverage at 5% risk versus 14.3% for ACT-ViT and 6.2% for MTE. CNN/DailyMail is substantially harder: no method provides useful coverage at 5% risk; at 80% coverage ActMap and

ACT-ViT are comparable (.432 and .427) against .476 for MTE.

## Com<sub>p</sub>utational Cost

Measurements use vLLM on one NVIDIA L40S with Qwen3-8B × TriviaQA; matched baseline latencies are in the appendix. Capture: forward hooks pool every layer online; hooked and unhooked decoding both averaged about 7 ms/prompt (five batches of 64), so overhead is within run-to-run variation. Storage: one 12 × 32 × 128 float16 map is 96 KiB (85–96× smaller than a 32-token float16 trajectory and 67× smaller than ACT-ViT’s dense tensor); 38k TriviaQA generations occupy 3.5 GiB as maps versus 235 GiB as dense tensors. Scoring: classifier inference takes .024 ms/map (batch 256), and training converges in under 20 GPU-minutes per seed. ACT-ViT takes .066 ms/answer with 2.1× the parameters and 25× the peak memory; its CNN/DailyMail configuration trains for about 3 GPU-hours per pair before the 24-configuration sweep. Semantic Entropy and EigenScore require 10 additional generations per answer.

## Limit<sub>a</sub>ti<sub>o</sub>n<sub>s a</sub>nd Ethi<sub>ca</sub>l C<sub>o</sub>n<sub>s</sub>id<sub>e</sub>r<sub>a</sub>ti<sub>o</sub>n<sub>s</sub>

Limitations. ActMap requires white-box access to generation-time hidden states: self-hosted or providerinstrumented models, not closed APIs. It is supervised: each deployment regime needs its own labeled generations. Label sources are imperfect (alias matching misses paraphrases; MiniCheck inherits its judge’s errors), and the score is correlational: high confidence means the trajectory resembles previously correct generations. It does not establish truth. Balanced-split evaluation difers from deployment prevalence (see Calibration). Our experiments use greedy decoding on English tasks; robustness across decoding strategies and temperatures is left to future work. In-domain results cover 7–8B generators on four tasks and Qwen3-32B on three short-form tasks; ablations cover two pairs.

Ethical considerations. A correctness score can reduce overreliance on fluent wrong answers, but a miscalibrated score can become a safety veneer. Under prevalence shift on NQ-Open and Mistral-7B GSM8K, ECE reaches .15–.24 and temperature scaling does not repair it. The score should guide verification rather than replace it. Stored maps require the source text’s access controls and retention limits and provide a re-scorable audit trail for consequential answers.

## C<sub>o</sub>n<sub>c</sub>l<sub>us</sub>i<sub>o</sub>n

ActMap converts generation-time hidden activations into a fixed-size representation over depth and pooled hidden coordinates. Trained in-domain, its lightweight classifier outperforms every evaluated non-tensor baseline on all twelve 7– 8B pairs and matches dense activation-tensor learning from a 67× smaller, generator-agnostic artifact with better calibration. Capture adds no measurable overhead; results survive classifier swaps and hold in-domain at 32B scale. The main limitation is transfer: decision boundaries do not yet align across tasks, generators, or scales. Code, replication materials, and the 476,372-map dataset will be released on Hugging Face upon publication.

## R<sub>e</sub>f<sub>erences</sub>

Amos Azaria and Tom Mitchell. The internal state of an LLM knows when it’s lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 967– 976, 2023.

Guy Bar-Shalom, Fabrizio Frasca, Yaniv Galron, Yftah Ziser, and Haggai Maron. Beyond token probes: Hallucination detection via activation tensors with ACT-ViT. In Advances in Neural Information Processing Systems (NeurIPS), 2025.

Guy Bar-Shalom, Fabrizio Frasca, Derek Lim, Yoav Gelberg, Yftah Ziser, Ran El-Yaniv, Gal Chechik, and Haggai Maron. Beyond next token probabilities: Learnable, fast detection of hallucinations and data contamination on LLM output distributions. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

Samuel R. Bowman, Jeeyoon Hyun, Ethan Perez, Edwin Chen, Craig Pettit, Scott Heiner, Kamile Lukoši˙ ut¯ e,˙ Amanda Askell, et al. Measuring progress on scalable oversight for large language models, 2022. arXiv:2211.03540.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. In Proceedings of the 11th International Conference on Learning Representations (ICLR), 2023.

Chao Chen, Kai Liu, Ze Chen, Yi Gu, Yue Wu, Mingyuan Tao, Zhihang Fu, and Jieping Ye. INSIDE: LLMs’ internal states retain the power of hallucination detection. In Proceedings of the 12th International Conference on Learning Representations (ICLR), 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Łukasz Kaiser, Matthias Plappert,

Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. arXiv:2110.14168.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In Proceedings of the 9th International Conference on Learning Representations (ICLR), 2021.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, et al. Toy models of superposition, 2022. arXiv:2209.10652.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems (NeurIPS), volume 30, pages 4878–4887, 2017.

Chuan Guo, Geof Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural networks. In Proceedings ofthe 34th International Conference on Machine Learning (ICML), pages 1321–1330, 2017.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7B, 2023. arXiv:2310.06825.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1601–1611, 2017.

Saurav Kadavath, Tom Conerly, Amanda Askell, et al. Language models (mostly) know what they know, 2022. arXiv:2207.05221.

Lorenz Kuhn, Yarin Gal, and Sebastian Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. In Proceedings ofthe 11th International Conference on Learning Representations (ICLR), 2023.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles (SOSP), pages 611–626, 2023.

Llama Team, AI @ Meta. The Llama 3 herd of models, 2024. arXiv:2407.21783.

Samuel Marks and Max Tegmark. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In Proceedings of the Conference on Language Modeling (COLM), 2024.

Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. On faithfulness and factuality in abstractive summarization. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1906–1919, 2020.

Qwen Team. Qwen3 technical report, 2025. arXiv:2505.09388.

Abigail See, Peter J. Liu, and Christopher D. Manning. Get to the point: Summarization with pointer-generator networks. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL), pages 1073–1083, 2017.

Liyan Tang, Philippe Laban, and Greg Durrett. MiniCheck: Eficient fact-checking of LLMs on grounding documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 8818–8847, 2024.

Artem Vazhentsev, Ekaterina Fadeeva, Rui Xing, Gleb Kuzmin, Ivan Lazichny, Alexander Panchenko, Preslav Nakov, Timothy Baldwin, Maxim Panov, and Artem Shelmanov. Unconditional truthfulness: Learning unconditional uncertainty of large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 35673– 35694, 2025.

Artem Vazhentsev, Lyudmila Rvanova, Gleb Kuzmin, Ekaterina Fadeeva, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin, and Artem Shelmanov. Eficient hallucination detection for LLMs using uncertainty-aware attention heads. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026.

Zeguan Xiao, Diyang Dou, Boya Xiong, Yun Chen, and Guanhua Chen. Enhancing uncertainty estimation in LLMs with expectation of aggregated internal belief. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026.

## T<sub>ec</sub>hni<sub>ca</sub>l A<sub>ppe</sub>ndi<sub>x</sub>

This appendix gives the implementation details for the representation and classifier, specifies the ACT-ViT reproduction protocol, and reports measured costs. It also records the software environment and retained artifacts. The main paper contains all claims and results needed to assess the paper.

## A E<sub>xac</sub>t A<sub>c</sub>ti<sub>va</sub>ti<sub>o</sub>n<sub>-</sub>M<sub>ap</sub> C<sub>o</sub>n<sub>s</sub>tr<sub>uc-</sub> tion

For one generated answer, let $H \in \mathbb { R } ^ { L \times T \times D }$ contain the hidden state from every transformer block and generated token. Hidden coordinates are first reduced to $D ^ { \prime } = 1 2 8$ by one-dimensional adaptive average pooling over contiguous coordinate intervals. Denote the resulting tensor by $X \in$ $\mathbb { R } ^ { L \times T \times D ^ { \prime } }$

The twelve channels below are computed independently for every layer ℓ and pooled coordinate d. For clarity, $x _ { t } =$ $X _ { \ell , t , d }$ and $\bar { x } = T ^ { - 1 } \sum _ { t } x _ { t }$

1. Four segment means. The token indices are divided by integer boundaries obtained from linspace(0, T, 5). Each channel is the mean within one consecutive segment; an empty segment for a very short answer is replaced by its nearest valid one-token interval.

2. Final state. $x _ { T - 1 }$

3. Final-window mean. $\begin{array} { r } { \frac { 1 } { \operatorname* { m i n } ( 8 , T ) } \sum _ { t = \operatorname* { m a x } ( 0 , T - 8 ) } ^ { T - 1 } x _ { t } . } \end{array}$

4. Temporal standard deviation. The sample standard deviation over tokens. It is zero for $T = 1$

5. Temporal maximum. max<sub>t</sub> x<sub>t</sub>.

6. Endpoint drift. ${ x } _ { T - 1 } - { x } _ { 0 }$

7. Least-squares slope.

$$
\frac { \sum _ { t } ( t - \bar { t } ) ( x _ { t } - \bar { x } ) } { \operatorname* { m a x } \{ \sum _ { t } ( t - \bar { t } ) ^ { 2 } , 1 0 ^ { - 8 } \} } ,\tag{2}
$$

set to zero for $T = 1$

8. Layer RMS. Unlike the other channels, this value is computed once per layer over all tokens and pooled coordinates: $\| X _ { \ell , : , : } \| _ { 2 } / \sqrt { T D ^ { \prime } }$ . It is then broadcast across the coordinate axis.

$$
\begin{array} { r l r } { { 9 } . \ \mathbf { M e a n } } & { \mathbf { a b s o l u t e } } & { \mathbf { t e m p o r a l } } & { \mathbf { d i f f e r e n c e } . } \\ { ( T \ - \ 1 ) ^ { - 1 } \sum _ { t = 1 } ^ { T - 1 } | x _ { t } \ - \ x _ { t - 1 } | , } & { \mathrm { s e t } } & { \mathrm { t o } \quad \mathrm { z e r o } \quad \mathrm { f o r } } \\ { T = 1 . } \end{array}
$$

Stacking the channels gives $S \in \mathbb { R } ^ { 1 2 \times L \times 1 2 8 }$ . Contiguous adaptive average pooling maps the ordered layer axis to 32 rows. Finally, every channel is standardized independently within each example:

$$
M _ { c } = \frac { S _ { c } - \mu ( S _ { c } ) } { \mathrm { m a x } \{ \mathrm { s t d } ( S _ { c } ) , 1 0 ^ { - 6 } \} } .\tag{3}
$$

Table 6: Why each hand-designed channel family is included. These working hypotheses are evaluated by the grouped ablations in the main paper; they do not prescribe what the classifier must use.
<table><tr><td>Channel family</td><td>Intended observable</td></tr><tr><td>Segment means</td><td>Where activation mass lies during the an- swer and how its depth profile changes over time.</td></tr><tr><td>window</td><td>Final state and The endpoint representation and its re- cent context, comparable to last-token and</td></tr><tr><td>Dispersion</td><td>pooled-state probes. Variation within an answer and rare acti-</td></tr><tr><td>Drift and slope</td><td>vation excursions that may separate stable from conflicted or unstable generations. Whether activation patterns accumulate, re-</td></tr><tr><td>Layer RMS</td><td>solve, or change direction during the answer. Layer-wise activation energy independent of coordinate, retaining a coarse depth pro-</td></tr><tr><td>ences</td><td>file. Temporal differ- Local token-to-token changes that a global mean, endpoint, or linear slope can miss.</td></tr></table>

The stored map M has shape $1 2 \times 3 2 \times 1 2 8$ and float16 size 96 KiB. All statistics are computed in float32 before storage.

## A<sub>.</sub>1 D<sub>es</sub>i<sub>gn</sub> R<sub>a</sub>ti<sub>ona</sub>l<sub>e</sub> f<sub>or</sub> th<sub>e</sub> T<sub>we</sub>l<sub>ve</sub> Ch<sub>anne</sub>l<sub>s</sub>

We chose a small, hand-designed channel set and fixed it before the final benchmark evaluation. We do not claim that this set is unique or optimal. The channels summarize four aspects of a generated answer: when a pattern occurs, endpoint state, global dispersion, and temporal change. Segment means and endpoint channels retain coarse phase information. Standard deviation and maximum describe the distribution. Drift and slope capture long-range direction, while temporal diferences capture local movement. All statistics use the same captured hidden states, require no extra model call, and produce a fixed-size output for answers of diferent lengths.

ActMap tests whether a temporal statistic can carry diferent information at diferent depths and coordinates. The representation therefore keeps the layer and pooled-coordinate axes. In the grouped ablations, removing one channel family changes AUROC only slightly, whereas collapsing the depthresolved map causes a much larger loss. This result supports the full structured map but does not show that any single channel family is essential.

The same per-example standardization is applied to every retained channel. It removes absolute scale diferences between channels and examples but keeps the relative pattern across layers and pooled coordinates. The globalnormalization control in the main paper tests whether this pattern adds information beyond the overall activation scale.

## A<sub>.</sub>2 F<sub>o</sub>rm<sub>a</sub>l R<sub>ep</sub>r<sub>ese</sub>nt<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd C<sub>o</sub>m<sub>pu</sub>t<sub>a-</sub> ti<sub>ona</sub>l P<sub>roper</sub>ti<sub>es</sub>

For a trajectory $H \ \in \ \mathbb { R } ^ { L \times T \times D }$ , let ${ P _ { D } } : \mathbb { R } ^ { D } \ :  \ : \mathbb { R } ^ { 1 2 8 }$ denote contiguous adaptive average pooling over hidden coordinates. Applying it independently to every (ℓ, t) gives

$$
X _ { \ell , t , : } = P _ { D } ( H _ { \ell , t , : } ) , \qquad X \in \mathbb { R } ^ { L \times T \times 1 2 8 } .\tag{4}
$$

For channel $c ,$ let $\phi _ { c }$ be one of the twelve scalar trajectory statistics listed above. The unpooled channel maps are

$$
S _ { c } [ \ell , d ] = \phi _ { c } \mathopen { } \mathclose \bgroup \left( X _ { \ell , 0 : T - 1 , d } \aftergroup \egroup \right) , \qquad S \in \mathbb { R } ^ { 1 2 \times L \times 1 2 8 } .\tag{5}
$$

Let $A _ { L }$ be ordered adaptive average pooling from L rows to 32 rows, and let $\mathcal { N }$ denote independent per-channel spatial standardization. The stored ActMap is therefore the deterministic operator

$$
M = \Phi ( H ) = \mathcal { N } ( A _ { L } ( S ) ) \in \mathbb { R } ^ { 1 2 \times 3 2 \times 1 2 8 } .\tag{6}
$$

The learned detector is a separate function

$$
\hat { p } = f _ { \theta } ( \Phi ( H ) ) , \qquad \hat { p } \in [ 0 , 1 ] ,\tag{7}
$$

trained to estimate the correctness label for a fixed generator and task. Φ defines the reusable representation. We do not assume that $f _ { \theta }$ or its decision boundary transfers without target-domain supervision.

The output shape is independent of L, T, and D for a decoder-only transformer whose hidden states can be captured. The map keeps layer and coordinate order, so the classifier can use their relative structure. During generation, the implementation reduces each hidden state to 128 pooled coordinates. It applies the twelve statistics after the pooled sequence is complete. The stored map has 12 · 32 · 128 values regardless of output length, instead of O(LTD) full hiddenstate values. Because segment boundaries depend on the final token count, the current implementation temporarily keeps the pooled O(LT · 128) sequence until generation ends. The ablations test whether this compression retains enough information for correctness prediction.

## B Prim<sub>a</sub>r<sub>y</sub> Cl<sub>ass</sub>ifi<sub>e</sub>r

The main experiments use the compact transformer in Table 7. A convolution whose stride equals its kernel size forms non-overlapping learned patches. It turns the 32 × 128 map into an 8 × 8 grid of 64 tokens. A learned classification token is prepended. Learned row and column embeddings preserve both spatial axes without a full 64-position table.

Attention dropout is 0.1. MLP and readout dropout are 0.3, while positional dropout is 0.15. Stochastic-depth probability increases linearly from 0 in the first block to 0.05 in the sixth. The patch-projection weights use Xavier uniform initialization. The class token and factorized positional embeddings use a truncated normal distribution with standard deviation 0.02; remaining linear and normalization layers use PyTorch defaults.

The parameter accounting is: 147,648 for patch projection, 3,456 for class and position parameters, 2,225,664 across the six encoder blocks, 384 for the final layer normalization, and 24,833 for the readout head. The sigmoid of the scalar logit is the reported estimated correctness probability.

## C O<sub>p</sub>timiz<sub>a</sub>ti<sub>o</sub>n <sub>a</sub>nd M<sub>o</sub>d<sub>e</sub>l S<sub>e</sub>l<sub>ec</sub>ti<sub>o</sub>n

Each (generator, dataset) detector is trained independently on its balanced training split with the hyperparameters in Table 8. Validation and test splits are also balanced and disjoint by source key. We use binary cross-entropy with logits. The positive-class weight is $N _ { - } / N _ { + }$ , which equals 1.0 for every balanced training split in the main experiments.

Gaussian noise is sampled independently for every training map before batching. For each ActMap training batch, a single λ ∼ Beta(0.2, 0.2) mixes maps with a random permutation; the loss is the corresponding convex combination of the two binary losses. Validation and test maps receive no augmentation. The checkpoint is replaced whenever validation AUROC strictly improves; ties retain the earlier epoch. Training stops after 20 epochs without improvement. The selected checkpoint is then evaluated exactly once on the test split. Main-table values are arithmetic means over the three seeds; ECE uses ten equal-width probability bins.

The learning-rate multiplier at zero-indexed epoch e is

$$
\eta ( e ) = \left\{ \begin{array} { l l } { ( e + 1 ) / 5 , } & { e < 5 , } \\ { 0 . 0 1 + 0 . 4 9 5 \left[ 1 + \cos \left( \pi \frac { e - 5 } { 8 0 - 5 } \right) \right] , } & { e \geq 5 . } \end{array} \right.\tag{8}
$$

## C<sub>.</sub>1 ACT<sub>-</sub>ViT R<sub>ep</sub>r<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n Pr<sub>o</sub>t<sub>oco</sub>l

We evaluate ACT-ViT using the authors’ released architecture. Each generated trajectory is converted to the published $L _ { \mathrm { e f f } } ~ = ~ 8 ~ \mathrm { b y } ~ N _ { \mathrm { e f f } } ~ = ~ 1 0 0$ dense activation tensor. As in the released preprocessing, the output-token axis is sliced or zero-padded to the fixed $N _ { \mathrm { m a x } } = 1 0 0$ positions, and the zero-padded layer axis is max-pooled to eight groups. We retain the released model-specific linear-adapter branch and zero-padding rule. We search the full 24-configuration grid used by the authors: hidden dimensions {128, 1024}, transformer depths {1, 3}, weight decays $\{ 1 , 1 0 ^ { - 3 } \}$ , and patch sizes {(1, 1), (8, 1), (4, 2)}. All configurations use four attention heads, dropout 0.3, learning rate $1 0 ^ { - 3 }$ , 15 epochs, and the published batch size 128. Optimization uses AdamW and binary cross-entropy, with a cosine schedule and a 10% step warm-up. The released patience of 30 exceeds the 15- epoch sweep horizon, so every configuration runs for all 15 epochs.

We select the configuration by validation AUROC for seed 42, retrain that configuration with seeds 123 and 456, and report the three-seed mean. For each seed, the checkpoint is replaced whenever validation AUROC strictly improves; ties retain the earlier epoch. We evaluate the selected checkpoint once on the test split. This matches the authors’ test metric at the best-validation epoch; the test split is not consulted during training or model selection. ACT-ViT receives the same complete balanced training, validation, and test rows as the other supervised methods; we do not apply the upstream 10,000-example preprocessing cap. Generators, saved responses, correctness labels, supervision, splits, and random seeds are fixed across methods. Only the detector and its published model-selection protocol difer.

Table 7: Exact classifier architecture. The total number of trainable parameters is 2,401,985.
<table><tr><td>Component</td><td>Configuration</td><td>Output</td></tr><tr><td>Input</td><td>12 channels</td><td> $1 2 \times 3 2 \times 1 2 8$ </td></tr><tr><td>Patch projection</td><td>Conv2d, 4 × 16, stride 4 × 16</td><td> $6 4 \times 1 9 2$ </td></tr><tr><td>Position</td><td>class + factorized 8 row/8 column</td><td> $6 5 \times 1 9 2$ </td></tr><tr><td>Encoder</td><td>6 pre-norm blocks</td><td> $6 5 \times 1 9 2$ </td></tr><tr><td>Attention</td><td> $6 \mathrm { h e a d s } , \mathrm { h e a d w i d t h } 3 2$ </td><td> $6 5 \times 1 9 2$ </td></tr><tr><td>Block MLP</td><td> $1 9 2 \to 5 7 6 \to 1 9 2 , { \mathrm { G E L U } }$ </td><td> $6 5 \times 1 9 2$ </td></tr><tr><td>Readout</td><td> $\mathrm { L N } , 1 9 2  1 2 8  1 , \mathrm { G E L U }$ </td><td>scalar logit</td></tr></table>

Table 8: Training hyperparameters used by every main-table ActMap detector.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Initial learning rate</td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>Weight decay</td><td>0.05</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Maximum epochs</td><td>80</td></tr><tr><td>Warm-up</td><td>5 epochs, linear</td></tr><tr><td>Post-warm-up schedule</td><td>cosine, final multiplier 0.01</td></tr><tr><td>Early-stopping patience</td><td>20 epochs</td></tr><tr><td>Selection metric</td><td>validation AUROC</td></tr><tr><td>Gradient-norm clipping</td><td>1.0</td></tr><tr><td>Gaussian input noise</td><td> $\sigma = 0 . 0 8$ </td></tr><tr><td>Mixup</td><td> $\alpha = 0 . 2$ </td></tr><tr><td>Seeds</td><td> $4 2 , 1 2 3 , 4 5 6$ </td></tr></table>

Measured cost. Under the scoring protocol of the main paper (batch 256, 300 timed repeats, median per-item latency, single NVIDIA L40S), the validation-selected TriviaQA × Qwen3-8B configuration (hidden dimension 128, depth 1, weight decay $1 0 ^ { - 3 }$ , patch size 4 × 2; 5.16M parameters) scores at .066 ms per answer with a peak of 4,949 MiB of CUDA memory for one batch; the ActMap classifier (2.40M parameters) measured identically scores at .024 ms with a 197 MiB peak. One packed ACT-ViT input tensor $( 8 \times 1 0 0 \times 4 0 9 6$ , float16) is 6.25 MiB against ActMap’s 96 KiB map. Training the selected configuration for all three seeds took 1.4–73.7 GPU-minutes per short-form pair and 157.8–195.9 GPU-minutes per CNN/DailyMail pair on one L40S, excluding the 24-configuration sweep that selects it. The corresponding three-seed ActMap training takes about 54–57 GPU-minutes per TriviaQA pair and about 63 GPUminutes per CNN/DailyMail pair; unlike ACT-ViT, it uses one fixed configuration and no architecture sweep.

Table 9: In-domain ActMap results for Qwen3-32B on balanced test splits (mean over three seeds).
<table><tr><td>Dataset</td><td>AUROC</td><td>AUPRC</td><td>ECE</td></tr><tr><td>TriviaQA</td><td>.893</td><td>.892</td><td>.056</td></tr><tr><td>NQ-Open</td><td>.841</td><td>.816</td><td>.057</td></tr><tr><td>GSM8K</td><td>.870</td><td>.873</td><td>.082</td></tr></table>

## C<sub>.</sub>2 Tr<sub>a</sub>n<sub>s</sub>f<sub>e</sub>r S<sub>cope a</sub>nd Tr<sub>a</sub>in<sub>a</sub>bl<sub>e</sub> Ad<sub>ap</sub>t<sub>e</sub>r<sub>s</sub>

Transfer freezes the detectors and uses no target labels or adapter updates. This is stricter than in-domain training. A shared input shape does not guarantee the same correctness boundary across tasks or generators. The in-domain ACT-ViT comparison trains its published model-specific adapter; transfer freezes that adapter. Its near-chance results measure zero-shot reuse. Lightweight target adaptation may improve them, but remains outside this study.

## D R<sub>ep</sub>r<sub>o</sub>d<sub>uc</sub>ti<sub>o</sub>n En<sub>v</sub>ir<sub>o</sub>nm<sub>e</sub>nt <sub>a</sub>nd R<sub>e</sub>t<sub>a</sub>in<sub>e</sub>d Artif<sub>ac</sub>t<sub>s</sub>

The experiments used Python 3.12 with package versions pinned in a lockfile. Each run manifest records the operating system, CUDA and PyTorch versions, visible accelerator, configuration, and input-artifact identifiers. We ran all experiments on NVIDIA L40S GPUs with 48 GiB of device memory. Experiments with 7–8B models used one L40S; Qwen3-32B experiments used two. The cloud VMs used x86-64 Ubuntu Noble images. CPU model and host memory varied across generation and training jobs.

## D<sub>.</sub>1 D<sub>e</sub>t<sub>erm</sub>i<sub>n</sub>i<sub>sm</sub>

The reproduction runner fixes the relevant random seeds and uses deterministic PyTorch settings. These controls support repeated runs on the recorded software and hardware stack; exact bitwise agreement across stacks is not claimed.

Before test aggregation, each seed retains the full run configuration, the classifier constructor arguments, the validation-selected model state, the per-epoch loss and validation history, and the aggregate test metrics with one score and probability per balanced test row. The configuration includes split counts, optimizer, augmentation, determinism settings, and source-artifact metadata. The three checkpoints for each (generator, dataset) pair are the frozen detectors used in the transfer experiments.

Table 10: Measured per-answer latency beyond producing the answer, on our implementation and hardware stack.
<table><tr><td>Method</td><td>Added latency</td></tr><tr><td>Perplexity / MTE</td><td>negligible post-processing</td></tr><tr><td>P(True)</td><td>17.2 ms</td></tr><tr><td>Semantic Entropy</td><td>85 ms + 50–291 ms NLI</td></tr><tr><td>TAD</td><td>125.6ms</td></tr><tr><td>RAUQ</td><td>146.2 ms</td></tr><tr><td>EigenScore</td><td>85 ms + 329 ms cov.</td></tr><tr><td>ACT-ViT</td><td>.066 ms score</td></tr><tr><td>ACTMAP</td><td>no measurable capture; .024 ms score</td></tr></table>

## E Cl<sub>ass</sub>ifi<sub>e</sub>r C<sub>o</sub>ntr<sub>o</sub>l<sub>s</sub>

The classifier ablation keeps the maps and data splits fixed. Logistic regression maps all 49,152 entries to one logit (49,153 parameters). The matched MLP flattens the same map, applies one GELU hidden layer and dropout, and emits one logit. Its hidden width matches the primary classifier’s parameter budget. The 4× control uses four times that budget. Every architecture receives the same input information, so the comparison tests classifier capacity and architecture without changing the representation.

## F M<sub>easure</sub>d P<sub>er-</sub>A<sub>nswer</sub> L<sub>a</sub>t<sub>enc</sub>i<sub>es</sub>

Table 10 reports latency after the scored answer has been produced on our stack (vLLM, one NVIDIA L40S, Qwen3- 8B × TriviaQA, 32 new tokens). The values can change with the serving stack, kernels, batch size, and output length. For Semantic Entropy and EigenScore, sampling cost is the measured 8.5 ms marginal decode time times 10 samples. NLI cost increases with output length; parallel sampling can reduce wall-clock time by using more memory. Figure 4 relates these measurements to mean ranking quality.

## G TAD L<sub>a</sub>t<sub>ency</sub> P<sub>ro</sub>t<sub>oco</sub>l

The TAD benchmark restores the fitted oficial two-stage checkpoint for Qwen3-8B on TriviaQA. We time the evaluation path used in the paper: teacher-forced BF16 inference with eager attention, extraction of top-10 all-layer/all-head attention and token probabilities, and the fitted Ridge readout. Loading and training are excluded. After eight warm-up rows, we score 128 balanced test rows twice on each of two L40S GPUs. Each GPU runs an isolated single-GPU workload. The two means are 126.25 and 124.86 ms, or

![](images/1dc71a511164e3e5210031575003ce21b9bf960605513ae60339915b654f99b9.jpg)  
Added latency per scored generation (ms, log scale)  
Figure 4: Added per-answer latency (log scale; our stack, Table 10) against mean in-domain AUROC over the twelve 7–8B model–dataset pairs of the main paper. Latencies are implementation- and hardware-specific.

125.6 ms overall. Across 512 scores, feature extraction averages 110.6 ms and readout 14.9 ms. Total latency has a median of 120.8 ms and a 95th percentile of 158.9 ms. The experiment archive retains the raw timings and software versions.