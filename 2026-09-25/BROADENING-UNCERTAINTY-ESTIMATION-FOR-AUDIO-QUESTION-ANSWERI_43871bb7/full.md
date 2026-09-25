# BROADENING UNCERTAINTY ESTIMATION FOR AUDIO QUESTION ANSWERING ACROSS METHODS, FORMATS, AND INPUTS

Aaron Isidore Grace<sup>1</sup> Weiran Wang<sup>2</sup>

<sup>1</sup>David R. Cheriton School of Computer Science, University of Waterloo, Waterloo, ON, Canada <sup>2</sup>Department of Computer Science, University of Iowa, Iowa City, IA, USA aaron.grace@uwaterloo.ca weiran-wang@uiowa.edu

## ABSTRACT

Audio-language models can produce confident answers unsupported by the audio, motivating uncertainty estimates that identify unreliable responses. We compare probability-based, sampling-based, self-verification, evidential, and contrastive measures across four open-weight models and five audio QA benchmarks. In multiplechoice evaluation, first-token measures are strongest overall, with top-1 probability achieving a mean AUROC of .740, compared with .708 for ten-sample discrete semantic entropy, while requiring no additional model calls. Across four benchmarks, shifting from multiple-choice to open-ended evaluation lowers mean accuracy from 57.6% to 36.6%, yet uncertainty remains predictive of errors: semantic entropy, maximum token entropy, and semantic agreement achieve mean AUROCs of .697, .694, and .693, respectively. To test whether uncertainty reflects the evidence available to answer the question, we perform input ablations that remove either the audio or the question. Across top-1 confidence, entropy, and samplingbased measures, removing audio reduces error-detection AUROC by .101 on average, compared with .010 when removing the question. Together, these results establish efficient uncertainty baselines and show that uncertainty in audio-language models depends substantially more on available audio evidence than on question text.

Index Terms— Uncertainty estimation, Audio-language models, Model reliability, Model confidence, Audio grounding

## 1. INTRODUCTION

A confident-sounding answer from an audio-language model is not necessarily grounded in the audio. Models can invent sounds, speakers, or events, or answer questions the recording cannot resolve [1, 2]. Because such hallucinations can sound as fluent as correct answers, identifying unreliable responses before they are acted on is essential. Uncertainty estimation offers one solution.<sup>1</sup>

Uncertainty estimation in language and vision-language models draws on model probabilities and entropy [3], variation across sampled answers [4], and explicit self-evaluation [5]. Recent approaches also measure evidential conflict [6] or changes in predictions when an input modality is removed or altered [7, 8]. And in ASR, wordlevel confidence estimation is used to detect transcription errors [9].

While uncertainty estimation is widely studied in text and vision models, its application to audio-language models is underexplored. Kuan et al. [10] provide the first systematic comparison of uncertainty measures with LALMs, while recent contrastive-decoding methods use uncertainty to determine when intervention is needed [11, 12, 13]. However, many recent LLM and VLM uncertainty methods remain untested in audio-language models.

In addition, most audio-language benchmarks use multiplechoice questions, yet this format has come under increasing scrutiny [14, 15]. Chandak et al. [16] show that answer choices provide strong cues even when the question is withheld. Their answermatching method scores free-form responses against reference answers and agrees well with human judgments. ORCA [17] evaluates open-ended audio responses using reference answers and textual audio grounding. However, whether uncertainty estimates remain reliable in open-ended settings is still unclear.

A related question is whether uncertainty reflects reliance on audio. Foo et al. [18] find that models retain 60–72% of their performance without audio, suggesting substantial linguistic and answerchoice priors. However, how uncertainty changes when either the audio or the question is removed has not been measured.

We address these gaps by evaluating uncertainty across multiplechoice and open-ended audio question answering and under controlled removal of input information. Our contributions are:

• We conduct a broad comparison of uncertainty measures for audio-language models, spanning probability-based, sampling-based, self-verification, evidential, and contrastive signals. Across multiple-choice audio QA, simple first-token probability measures outperform more computationally expensive sampling and self-verification methods on average while requiring no additional model calls, establishing them as a strong and efficient baseline for uncertainty estimation.

• We convert four multiple-choice benchmarks to open-ended evaluation by removing the answer choices and scoring freeform responses with an answer-matching model. Although accuracy changes substantially, uncertainty remains predictive of correctness, with sequence-level measures showing greater robustness across answer formats.

• We separately remove the audio or the question while retaining the answer choices. Audio removal causes a much larger decline in uncertainty discrimination and greater changes in the ranking of uncertainty methods than question removal.

## 2. MULTIPLE-CHOICE EXPERIMENTAL SETUP

We evaluate four LALMs: Audio Flamingo 3 (AF3) [19], Phi-4- multimodal-instruct (Phi-4-MM) [20], Qwen2-Audio-7B-Instruct (Qwen2-A) [21], and Qwen2.5-Omni-7B (Qwen2.5-O) [22]. We test five audio QA benchmarks. AQUA-Bench includes standard QA and abstention cases with insufficient evidence, missing correct answers, or incompatible answer sets [23]. MMAU spans speech, environmental sounds, and music; we use its 1,000-question testmini split [24]. MMAR targets multi-step audio reasoning [25], MMSU fine-grained linguistic and paralinguistic understanding [26], and SAKURA matched single- and multi-hop reasoning [27]. Models are prompted to output the option letter first.

<table><tr><td></td><td></td><td></td><td colspan="3">Single-pass</td><td>Sampling</td><td colspan="3">Contrastive</td><td colspan="2">Self-verification</td></tr><tr><td>Benchmark</td><td>Model</td><td>Acc.</td><td>TOP1</td><td> $H _ { \mathrm { a l l } }$ </td><td>κ</td><td>DSE</td><td>UE</td><td>Sup</td><td>L2</td><td>P(IK)</td><td>P(aud)</td></tr><tr><td></td><td>AF3</td><td>55.5</td><td>.685</td><td>.660</td><td>.642</td><td>.646</td><td>.665</td><td>.596</td><td>.564</td><td>.487</td><td>.829</td></tr><tr><td>AQUA-Bench</td><td>Phi-4-MM</td><td>31.2</td><td>.510</td><td>.499</td><td>.392</td><td>.504</td><td>.514</td><td>.536</td><td>.470</td><td>.338</td><td>.270</td></tr><tr><td></td><td>Qwen2-A</td><td>39.0</td><td>.730</td><td>.734</td><td>.693</td><td>.690</td><td>.734</td><td>.746</td><td>.685</td><td>.659</td><td>.575</td></tr><tr><td></td><td>Qwen2.5-0</td><td>73.9</td><td>.729</td><td>.692</td><td>.693</td><td>.703</td><td>.702</td><td>.507</td><td>.413</td><td>.458</td><td>.910</td></tr><tr><td></td><td>AF3</td><td>74.6</td><td>.860</td><td>.851</td><td>.842</td><td>.823</td><td>.856</td><td>.651</td><td>.492</td><td>.604</td><td>.623</td></tr><tr><td>MMAU</td><td>Phi-4-MM</td><td>62.7</td><td>.788</td><td>.784</td><td>.567</td><td>.752</td><td>.771</td><td>.551</td><td>.434</td><td>.572</td><td>.593</td></tr><tr><td></td><td>Qwen2-A</td><td>65.8</td><td>.804</td><td>.805</td><td>.793</td><td>.755</td><td>.805</td><td>.699</td><td>.578</td><td>.568</td><td>.653</td></tr><tr><td></td><td>Qwen2.5-0</td><td>75.3</td><td>.853</td><td>.846</td><td>.839</td><td>.815</td><td>.848</td><td>.693</td><td>.570</td><td>.582</td><td>.588</td></tr><tr><td></td><td>AF3</td><td>59.3</td><td>.695</td><td>.693</td><td>.671</td><td>.675</td><td>.681</td><td>.647</td><td>.579</td><td>.526</td><td>.547</td></tr><tr><td>MMAR</td><td>Phi-4-MM</td><td>47.7</td><td>.683</td><td>.694</td><td>.579</td><td>.680</td><td>.661</td><td>.601</td><td>.577</td><td>.467</td><td>.548</td></tr><tr><td></td><td>Qwen2-A</td><td>48.2</td><td>.681</td><td>.685</td><td>.678</td><td>.666</td><td>.679</td><td>.631</td><td>.534</td><td>.470</td><td>.552</td></tr><tr><td></td><td>Qwen2.5-0</td><td>60.2</td><td>.742</td><td>.726</td><td>.704</td><td>.697</td><td>.727</td><td>.712</td><td>.647</td><td>.510</td><td>.555</td></tr><tr><td></td><td>AF3</td><td>58.4</td><td>.794</td><td>.788</td><td>.773</td><td>.758</td><td>.794</td><td>.774</td><td>.667</td><td>.496</td><td>.537</td></tr><tr><td>MMSU</td><td>Phi-4-MM</td><td>54.3</td><td>.770</td><td>.770</td><td>.567</td><td>.741</td><td>.756</td><td>.688</td><td>.597</td><td>.650</td><td>.631</td></tr><tr><td></td><td>Qwen2-A</td><td>54.1</td><td>.758</td><td>.760</td><td>.757</td><td>.721</td><td>.748</td><td>.679</td><td>.569</td><td>.595</td><td>.603</td></tr><tr><td></td><td>Qwen2.5-0</td><td>61.7</td><td>.821</td><td>.821</td><td>.816</td><td>.794</td><td>.810</td><td>.737</td><td>.648</td><td>.659</td><td>.525</td></tr><tr><td></td><td>AF3</td><td>65.4</td><td>.771</td><td>.753</td><td>.672</td><td>.710</td><td>.765</td><td>.863</td><td>.774</td><td>.574</td><td>.555</td></tr><tr><td>SAKURA</td><td>Phi-4-MM</td><td>39.5</td><td>.615</td><td>.627</td><td>.559</td><td>.612</td><td>.584</td><td>.633</td><td>.578</td><td>.483</td><td>.494</td></tr><tr><td></td><td>Qwen2-A</td><td>58.0</td><td>.704</td><td>.705</td><td>.701</td><td>.672</td><td>.735</td><td>.769</td><td>.677</td><td>.564</td><td>.550</td></tr><tr><td></td><td>Qwen2.5-0</td><td>69.8</td><td>.806</td><td>.803</td><td>.772</td><td>.754</td><td>.819</td><td>.832</td><td>.775</td><td>.687</td><td>.737</td></tr></table>

Table 1. Multiple-choice accuracy (%) and uncertainty AUROC across models and benchmarks. Bold indicates the best AUROC in each row.

Because models can exhibit option-position bias [15, 28], we place the correct answer once in each available position and average across the permutations. Particuarly, this matters for AQUA-Bench, which places abstention at option E. AF3 achieves only 19.7% accuracy at E, versus 54.5% across A–D, suggesting that previously reported low performance [10, 23] may reflect position bias.

## 3. MULTIPLE-CHOICE UNCERTAINTY

We use each uncertainty score to distinguish incorrect from correct answers, treating errors as the positive class. Performance is measured by area under the receiver operating characteristic curve (AUROC), where higher values indicate better separation and .5 corresponds to chance. Confidence-based metrics are inverted so that higher scores consistently indicate greater uncertainty.

First token distribution. Let p be the first-token distribution over vocabulary V . We treat top-1 token probability as confidence and normalized full-vocabulary entropy as uncertainty:

$$
T O P I = \operatorname* { m a x } _ { v \in V } p _ { v } , \qquad H _ { \mathrm { a l l } } = \frac { - \sum _ { v \in V } p _ { v } \log p _ { v } } { \log | V | } .\tag{1}
$$

Despite their simplicity, first-token scores are the strongest overall methods in Table 1. Much of the useful uncertainty signal is therefore already available at the point of decision, without additional sampling or verification. Because these scores are available before further generation, they are more suitable for latency-sensitive audio applications where uncertainty must be assessed in real time.

Evidential conflict. Following Huang et al. [6], we center output weights across answer options and write each resulting logit as $w _ { i } ^ { \hat { + } } -$ $\boldsymbol { w } _ { i } ^ { - }$ . Here, $\boldsymbol { w } _ { i } ^ { + }$ and $\boldsymbol { w } _ { i } ^ { - }$ sum the positive and negative contributions from final hidden-state features by magnitude. Retaining both sums exposes evidence that would otherwise cancel in the final logit. Let $\Omega = \{ o _ { 1 } , \ldots , o _ { M } \}$ denote the M answer options. A Dempster– Shafer mass function m $: 2 ^ { \Omega }  [ 0 , 1 ]$ assigns mass $m ( A )$ to each subset $A \subseteq \Omega ,$ , representing support that the correct answer lies in A, with $m ( \emptyset ) = 0$ and $\Sigma _ { A \subset \Omega } { \bar { m ( A ) } } = 1$ . Positive evidence assigns mass $1 - e ^ { - w _ { i } ^ { + } }$ to {o<sub>i</sub>}, while negative evidence assigns $1 - e ^ { - w _ { i } ^ { - } }$ to $\Omega \setminus \{ o _ { i } \}$ ; remaining mass is assigned to Ω.

Combining these masses across options separately for each sign using Dempster’s rule gives $m ^ { + }$ and $m ^ { - }$ . Their conflict is

$$
\kappa = \sum _ { \stackrel { B , C \subseteq \Omega } { B \cap C } } m ^ { + } ( B ) m ^ { - } ( C ) .\tag{2}
$$

Thus, κ measures how much positive and negative evidence supports incompatible answer sets. A high κ indicates strong contradictory evidence within the model rather than merely weak evidence overall. Its competitive AUROC, distinct mechanism, and lack of additional model calls motivate further study, although large evidence magnitudes can cause saturation.

Sampling. Discrete semantic entropy (DSE) measures disagreement among $K = 1 0$ sampled answers [10].

$$
\hat { p } _ { j } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbf { 1 } [ y ^ { ( k ) } = j ] , \qquad D S E = H ( \hat { p } ) .\tag{3}
$$

Despite requiring multiple generations, DSE achieves a mean AU-ROC of .708, below $H _ { \mathrm { a l l } } \left( . 7 3 5 \right)$ and TOP1 (.740) in Table 1.

Audio–silence contrast. Alongside the answering pass, we run one further single-token pass with the audio replaced by silence and read its distribution the same way. Renormalizing over the options gives audio/silence posteriors $c ^ { a } , \bar { c } ^ { 0 }$ (full-vocabulary $p ^ { a } , p ^ { 0 } ) ; y$ is the option chosen under real audio. This asks not how confident the model is, but how much of that confidence the audio is responsible for.

<table><tr><td></td><td></td><td></td><td colspan="3">Token uncertainty</td><td colspan="2">Seq. likelihood</td><td colspan="2">Sampling</td><td colspan="2">Contrastive</td><td colspan="2">Self-verification</td></tr><tr><td>Benchmark</td><td>Model</td><td>Match</td><td> $\overline { { T O P { } . { } _ { M I N } } }$ </td><td> $H _ { \mathrm { m a x } }$ </td><td> $H _ { \mathrm { { f i r s t } } }$ </td><td>NLL</td><td>PPL</td><td>SemEnt</td><td>SemAgr</td><td>UE</td><td>CWO</td><td>P(IK)</td><td>P(aud)</td></tr><tr><td></td><td>AF3</td><td>78.8</td><td>.750</td><td>.743</td><td>.703</td><td>.754</td><td>.730</td><td>.730</td><td>.758</td><td>.666</td><td>.376</td><td>.489</td><td>.378</td></tr><tr><td>AQUA</td><td>Phi-4-MM</td><td>45.7</td><td>.763</td><td>.733</td><td>.728</td><td>.782</td><td>.778</td><td>.759</td><td>.759</td><td>.718</td><td>.242</td><td>.281</td><td>.281</td></tr><tr><td></td><td>Qwen2-A</td><td>32.5</td><td>.715</td><td>.776</td><td>.762</td><td>.732</td><td>.716</td><td>.771</td><td>.755</td><td>.785</td><td>.486</td><td>.747</td><td>.639</td></tr><tr><td></td><td>Qwen2.5-0</td><td>80.5</td><td>.657</td><td>.643</td><td>.636</td><td>.677</td><td>.671</td><td>.648</td><td>.650</td><td>.551</td><td>.351</td><td>.416</td><td>.718</td></tr><tr><td>MMAR</td><td>AF3</td><td>31.5</td><td>.692</td><td>.687</td><td>.586</td><td>.678</td><td>.621</td><td>.697</td><td>.720</td><td>.597</td><td>.522</td><td>.510</td><td>.520</td></tr><tr><td></td><td>Phi-4-MM</td><td>25.6</td><td>.738</td><td>.725</td><td>.698</td><td>.698</td><td>.660</td><td>.731</td><td>.724</td><td>.673</td><td>.440</td><td>.470</td><td>.445</td></tr><tr><td></td><td>Qwen2-A</td><td>25.2</td><td>.654</td><td>.671</td><td>.638</td><td>.645</td><td>.624</td><td>.675</td><td>.668</td><td>.641</td><td>.464</td><td>.502</td><td>.521</td></tr><tr><td></td><td>Qwen2.5-0</td><td>30.6</td><td>.712</td><td>.711</td><td>.695</td><td>.698</td><td>.677</td><td>.719</td><td>.696</td><td>.700</td><td>.506</td><td>.530</td><td>.632</td></tr><tr><td>MMAU</td><td>AF3</td><td>35.1</td><td>.655</td><td>.659</td><td>.697</td><td>.571</td><td>.676</td><td>.668</td><td>.662</td><td>.707</td><td>.444</td><td>.613</td><td>.583</td></tr><tr><td></td><td>Phi-4-MM</td><td>25.6</td><td>.695</td><td>.724</td><td>.735</td><td>.624</td><td>.692</td><td>.688</td><td>.671</td><td>.739</td><td>.453</td><td>.449</td><td>.581</td></tr><tr><td></td><td>Qwen2-A</td><td>26.1</td><td>.641</td><td>.640</td><td>.635</td><td>.600</td><td>.639</td><td>.635</td><td>.633</td><td>.629</td><td>.492</td><td>.625</td><td>.586</td></tr><tr><td></td><td>Qwen2.5-0</td><td>30.3</td><td>.688</td><td>.712</td><td>.708</td><td>.636</td><td>.684</td><td>.705</td><td>.663</td><td>.709</td><td>.422</td><td>.612</td><td>.618</td></tr><tr><td>MMSU</td><td>AF3</td><td>32.4</td><td>.699</td><td>.669</td><td>.636</td><td>.650</td><td>.648</td><td>.685</td><td>.700</td><td>.617</td><td>.585</td><td>.469</td><td>.527</td></tr><tr><td></td><td>Phi-4-MM</td><td>27.1</td><td>.633</td><td>.643</td><td>.634</td><td>.586</td><td>.679</td><td>.665</td><td>.668</td><td>.619</td><td>.468</td><td>.600</td><td>.563</td></tr><tr><td></td><td>Qwen2-A</td><td>24.7</td><td>.648</td><td>.667</td><td>.659</td><td>.627</td><td>.678</td><td>.684</td><td>.681</td><td>.647</td><td>.517</td><td>.614</td><td>.533</td></tr><tr><td></td><td>Qwen2.5-0</td><td>34.0</td><td>.709</td><td>.707</td><td>.707</td><td>.620</td><td>.708</td><td>.684</td><td>.673</td><td>.705</td><td>.526</td><td>.674</td><td>.656</td></tr></table>

Table 2. Open-ended evaluation: judge-match accuracy (%) and uncertainty AUROC. Bold indicates the best AUROC in each row.

Weighted support (Sup). Whether the audio backs the model’s specific answer, and whether the model believes it: the chosen option’s audio-branch probability, scaled by its gain over silence.

Unexplained entropy (UE). Exactly VAUQ’s s<sub>VAUQ</sub> [7], read over the option posterior and built on a whole-clip silence ablation rather than VAUQ’s core-region masking: predictive uncertainty discounted by however much of it the audio explains.

L2 is the full-distribution shift between audio and silence.

$$
\begin{array} { r l } & { S u p = c _ { y } ^ { a } \big ( c _ { y } ^ { a } - c _ { y } ^ { 0 } \big ) , } \\ & { U E = H ( c ^ { a } ) - \alpha [ H ( c ^ { 0 } ) - H ( c ^ { a } ) ] , \quad \alpha = . 5 , } \\ & { L \mathcal { Q } = \| p ^ { a } - p ^ { 0 } \| _ { 2 } / \sqrt { 2 } . } \end{array}\tag{4}
$$

Among the audio–silence metrics, UE performs most consistently overall, while Sup performs especially well on SAKURA. L2 is generally weaker: it keeps the size of the distributional shift but not its direction, discarding exactly the graded, signed information UE and Sup are built to preserve.

## Self-verification.

A second pass evaluates the answer $\hat { y }$ under a verification prompt π, using the probability of affirmation as confidence [5, 10]:

$$
S V = p _ { \theta } ( z ^ { + } \mid a , q , \hat { y } , \pi ) .
$$

We test several prompts and report two in Table 1: P(IK), following the published baseline [10], and P(aud), which asks whether the audio supports the answer. Thus, P(aud) reflects the model’s stated audio support rather than measured audio dependence. Overall, self-verification is weaker and less stable than first-token confidence, though P(aud) performs well in some settings.

We also evaluated Qwen2.5-Omni-3B, which showed similar uncertainty trends but lower mean accuracy than the 7B model (65.2% vs. 68.2%) and AUROC lower by .011–.022 across metrics. We therefore report only the 7B results for brevity.

## 4. OPEN-ENDED UNCERTAINTY

We convert the multiple-choice benchmarks to open-ended evaluation by removing the answer choices. SAKURA is excluded because many of its questions depend on the answer choices. For

MMAU and MMAR, we exclude questions that cannot be answered without the option menu, dropping 78 of MMAU’s 1,000 questions and 23 of MMAR’s 995. MMSU is filtered to exclude 11 menudependent task families out of 47, removing 1,181 of its 5,000 questions. For AQUA-Bench, we retain the original and mismatched audio–question subsets, with an abstention hint in the prompt for the latter. A Qwen3-4B judge scores responses against the reference using Chandak et al.’s [16] answer-matching protocol, marking a response correct when it conveys at least as much information as the reference. We evaluate open-ended uncertainty measures that generally correspond to the multiple-choice metrics, allowing a practical comparison across answer formats. Results are reported in Table 2.

Removing answer choices generally lowers accuracy: mean accuracy falls from 69.6% to 29.3% on MMAU, from 57.1% to 29.5% on MMSU, and from 53.9% to 28.2% on MMAR. However, accuracy rises from 49.9% to 59.4% on AQUA-Bench. This suggests that models can more easily express abstention in free-form responses without the distraction of answer labels.

Token confidence and entropy. For the vocabulary distribution p<sub>t</sub> at generation step t, we compare first-token entropy with the most uncertain token in the response:

$$
\begin{array} { r l } { T O P 1 _ { M I N } = \underset { t } { \operatorname* { m i n } } \underset { v \in V } { \operatorname* { m a x } } p _ { t } ( v ) , } & { { } H _ { \mathrm { m a x } } = \underset { t } { \operatorname* { m a x } } H ( p _ { t } ) , } \\ { H _ { \mathrm { f i r s t } } = H ( p _ { 1 } ) . } \end{array}\tag{5}
$$

$H _ { \mathrm { { f i r s t } } }$ uses entropy at the first token. $H _ { \mathrm { m a x } }$ takes the maximum entropy and $T O P \mathrm { 1 } _ { \operatorname* { m i n } }$ the minimum top-1 probability across the response. $H _ { \mathrm { { f i r s t } } }$ performs worse overall, suggesting that later tokens provide additional uncertainty information.

Sequence likelihood. Negative log-likelihood (NLL) and perplexity (PPL) measure uncertainty from the probabilities assigned to the generated tokens. For an answer $y _ { 1 : T }$

$$
N L L = - \sum _ { t = 1 } ^ { T } \log p _ { t } ( y _ { t } ) , \qquad P P L = \exp ( N L L / T ) .\tag{6}
$$

NLL sums token surprise across the entire response and therefore grows with response length, whereas PPL normalizes by length and measures average token surprise. Both perform competitively in Table 2, with NLL particularly strong on AQUA-Bench and PPL on MMSU; neither dominates consistently overall.

<table><tr><td></td><td colspan="4">Canonical</td><td colspan="4">No question</td><td colspan="4">No audio</td></tr><tr><td></td><td colspan="2"> $H _ { \mathrm { a l l } }$ </td><td colspan="2">DSE</td><td colspan="2"> $H _ { \mathrm { a l l } }$ </td><td colspan="2">DSE</td><td colspan="2"> $H _ { \mathrm { a l l } }$ </td><td colspan="2">DSE</td></tr><tr><td>Dataset</td><td>Mean score</td><td>AUROC</td><td>Mean score</td><td>AUROC</td><td> $\Delta$ </td><td>AUROC</td><td></td><td>Δ AUROC</td><td>∆</td><td>AUROC</td><td></td><td>Δ AUROC</td></tr><tr><td>AQUA-Bench</td><td>.068</td><td>.646</td><td>.610</td><td>.636</td><td>+.004</td><td></td><td> $\mathbf { . 6 6 7 } \quad + . 0 3 6$ </td><td>.649</td><td>+.018</td><td>.630</td><td>+.148</td><td>.613</td></tr><tr><td>MMAU</td><td>.049</td><td>.821</td><td>.460</td><td>.786</td><td>+.010</td><td></td><td> $\mathbf { \nabla } \cdot 7 8 6 \mathbf { \nabla } + . 0 9 6$ </td><td>.756</td><td> $+ . 0 1 6$ </td><td>.728</td><td>+.139</td><td>.699</td></tr><tr><td>MMAR</td><td>.063</td><td>.700</td><td>.588</td><td></td><td> $. 6 7 9 \quad + . 0 0 3 \quad$ </td><td></td><td> $\mathbf { \nabla } . 7 \mathbf { 0 } 7 \mathbf { \nabla } + . 0 2 5 \mathbf { \Omega }$ </td><td>.684</td><td> $+ . 0 1 2$ </td><td>.606</td><td> $+ . 1 0 0$ </td><td>.575</td></tr><tr><td>MMSU</td><td>.057</td><td>.785</td><td>.537</td><td></td><td> $. 7 5 3 \quad + . 0 0 4 \quad$ </td><td></td><td> $\mathbf { . 7 5 6 \quad + . 0 2 9 }$ </td><td>.729</td><td> $+ . 0 1 5$ </td><td>.659</td><td> $+ . 1 2 0$ </td><td>.638</td></tr><tr><td>SAKURA</td><td>.055</td><td>.722</td><td>.493</td><td></td><td>.687 +.008</td><td></td><td> $\mathbf { \nabla } . 7 \mathbf { 0 9 } \mathrm { ~  ~ { ~ + . 0 5 2 } ~ }$ </td><td>.674</td><td> $+ . 0 2 1$ </td><td>.549</td><td> $+ . 1 4 7$ </td><td>.545</td></tr></table>

Table 3. Multiple-choice input ablation, averaged over four models. Mean score is the mean uncertainty value under full input, $\Delta$ its mean change under ablation (ablated minus canonical). Bold marks the best AUROC in each condition.

Contrastive. We compare generation with audio, denoted by superscript $^ a ,$ , to generation with the audio removed, denoted by superscript <sup>0</sup>. Here, $p _ { 1 } ^ { a }$ and $p _ { 1 } ^ { 0 }$ are the first-token distributions under the two conditions, while $y ^ { a }$ and $y ^ { 0 }$ are the corresponding responses. Let $W ( y )$ denote the set of content words in response $y .$ Unexplained entropy (UE) and content-word overlap (CWO) are:

$$
\begin{array} { c c } { { U E = H ( p _ { 1 } ^ { a } ) - \alpha \big [ H ( p _ { 1 } ^ { 0 } ) - H ( p _ { 1 } ^ { a } ) \big ] \ : , } } & { { \alpha = . 5 , } } \\ { { { } } } & { { { } } } \\ { { C W O = \displaystyle \frac { | W ( y ^ { a } ) \cap W ( y ^ { 0 } ) | } { | W ( y ^ { a } ) \cup W ( y ^ { 0 } ) | } . } } \end{array}\tag{7}
$$

Sampling. Semantic uncertainty from free-form responses accounts for different wordings of the same answer [29, 4]. We draw $K = 1 0$ stochastic responses $\mathbf { \chi } _ { y ^ { ( 1 ) } , \ldots , y ^ { ( K ) } } ^ { }$ and let yˆ denote the greedily decoded response, obtained by selecting the highest-probability token at each step. Using DeBERTa-large-MNLI conditioned on the question, we treat two responses as equivalent, $\boldsymbol { y } ^ { ( i ) } \equiv \boldsymbol { y } ^ { ( j ) }$ , when they entail one another in both directions. The sampled responses are grouped into semantic clusters with labels $c _ { k } \in \{ 1 , \ldots , C \}$

$$
\hat { p } _ { j } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbf { 1 } [ c _ { k } = j ] , \qquad S e m E n t = - \sum _ { j = 1 } ^ { C } \hat { p } _ { j } \log \hat { p } _ { j } ,\tag{8}
$$

$$
S e m A g r = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } { \bf 1 } \Big [ y ^ { ( k ) } \equiv \hat { y } \Big ] .
$$

SemEnt measures how dispersed the samples are across semantic clusters, while SemAgr measures how often sampled responses agree with the greedy response. These measures are competitive overall: SemEnt or SemAgr achieves the best AUROC in 6 of 16 model–benchmark pairs, including three of four MMAR rows and two of four MMSU rows. Semantic grouping also avoids treating paraphrases as distinct answers, making it better suited to free-form sampling than exact-string DSE.

Self-verification. Both self-verification prompts remain relatively weak in the open-ended setting. Mean P(aud) AUROC decreases from .596 to .549, while $P ( I K )$ remains nearly unchanged. Together with their modest multiple-choice performance, this suggests that the models have limited ability to reliably assess the correctness or audio support of their own answers through textual queries.

## 5. UNCERTAINTY WITH MISSING INPUTS

The question, audio, and answer choices can each provide clues to the answer. We remove either the audio or the question while retaining the remaining inputs to examine how uncertainty depends on these sources of evidence and whether it still identifies errors. Table 3 averages results across four models per dataset; overall means weight the 20 model–benchmark pairs equally. Higher mean $H _ { \mathrm { a l l } }$ and DSE scores indicate greater model uncertainty.

Removing the question. Mean accuracy falls from 57.7% to 54.2%, while TOP1 and $H _ { \mathrm { a l l } }$ each lose only .010 AUROC. The scores barely move either: $H _ { \mathrm { a l l } }$ rises .006 and DSE .048. One possible explanation is that descriptive answer choices can make the question unnecessary. For example, given options such as “a dog barking” and “a car horn,” a model can select the sound heard in the recording without seeing the question. With these cues available, removing the question has little effect on uncertainty’s ability to identify errors.

Removing audio. Removing audio has a larger effect than removing the question. Mean accuracy falls by 17.8 versus 3.5 percentage points, and TOP1 AUROC declines by .108 versus .010. Fullvocabulary entropy shows a similar decline in error discrimination. The scores also respond more strongly to missing audio. Mean $H _ { \mathrm { a l l } }$ rises by .016 without audio versus .006 without the question, with DSE following the same pattern. Thus, audio removal increases uncertainty while making errors harder to identify. The smaller TOP1 decline on AQUA-Bench may reflect questions that models can recognize as unanswerable from text alone.

First-token confidence and entropy retain useful error discrimination without audio, outperforming ten-sample DSE on average. Evidential conflict also becomes more competitive, narrowing its mean AUROC gap with TOP1 from .054 to .006, although its absolute AUROC declines for three of four models. Self-verification is more affected, with P(aud) approaching chance at .516 AUROC.

## 6. DISCUSSIONS

Existing approaches use uncertainty to gate chain-of-thought (CoT) reasoning [10] or guide contrastive decoding [13]. We instead tried giving models their own uncertainty scores to help them revise their answers. Each prompt included the original question, the previous answer, and one score, presented alone or with a calibrated summary or thirty worked examples. This approach did not improve accuracy. Numerical feedback reduced accuracy by 10.4–12.3 percentage points, and asking models to reason before revising made performance worse, with losses reaching 23.6 points.

These results suggest that the tested models do not reliably use their own uncertainty for self-correction. Future work could combine multiple uncertainty signals with a learned probe to produce a calibrated estimate of correctness. Bayesian decision theory could then turn this estimate into an explicit choice to retain, revise, or abstain, offering a principled way to balance answer quality against the costs of errors and additional inference.

## 7. REFERENCES

[1] C.-Y. Kuan, W.-P. Huang, and H.-y. Lee, “Understanding sounds, missing the questions: The challenge of object hallucination in large audio-language models,” in Proceedings of Interspeech, 2024.

[2] F. Zhao, Y. Chen, W. Lu, D. Zhang, X. Yue, and J. Wei, “HalluAudio: A comprehensive benchmark for hallucination detection in large audio-language models,” arXiv preprint arXiv:2604.19300, 2026.

[3] A. Malinin and M. Gales, “Uncertainty estimation in autoregressive structured prediction,” in ICLR, 2021.

[4] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal, “Detecting hallucinations in large language models using semantic entropy,” Nature, vol. 630, no. 8017, pp. 625–630, 2024.

[5] S. Kadavath, T. Conerly, A. Askell, T. Henighan, D. Drain, E. Perez, N. Schiefer, Z. Hatfield-Dodds, N. DasSarma, E. Tran-Johnson, et al., “Language models (mostly) know what they know,” arXiv preprint arXiv:2207.05221, 2022.

[6] T. Huang, Z. Liu, R. Wang, Y. Zhang, and L. Jing, “Visual hallucination detection in large vision-language models via evidential conflict,” arXiv preprint arXiv:2506.19513, 2025.

[7] S. Park, C. Oh, H. K. Choi, S. Du, and S. Li, “VAUQ: Visionaware uncertainty quantification for LVLM self-evaluation,” in Findings of ACL, 2026.

[8] R. Zhang, H. Zhang, and Z. Zheng, “VL-Uncertainty: Detecting hallucination in large vision-language model via uncertainty estimation,” arXiv preprint arXiv:2411.11919, 2024.

[9] D. Oneata, A. Caranica, A. Stan, and H. Cucu, “An evaluation of word-level confidence estimation for end-to-end automatic speech recognition,” arXiv preprint arXiv:2101.05525, 2021.

[10] C.-Y. Kuan, W.-P. Huang, and H.-y. Lee, “Walking through uncertainty: An empirical study of uncertainty estimation for audio-aware large language models,” arXiv preprint arXiv:2604.25591, 2026.

[11] T.-w. Hsu, K.-H. Lu, C.-H. Chiang, and H.-y. Lee, “Reducing object hallucination in large audio-language models via audioaware decoding,” arXiv preprint arXiv:2506.07233, 2025.

[12] C. Jung, Y. Jang, and J. S. Chung, “AVCD: Mitigating hallucinations in audio-visual large language models through contrastive decoding,” in NeurIPS, 2025.

[13] Y. Li, Y. Liu, Z. Song, Y. Wei, M. Taka´c, and S. Lahlou, “Tem-ˇ poral contrastive decoding: A training-free method for large audio-language models,” arXiv preprint arXiv:2604.15383, 2026.

[14] W. Li, L. Li, T. Xiang, X. Liu, W. Deng, and N. Garcia, “Can multiple-choice questions really be useful in detecting the abilities of LLMs?,” in Proceedings ofLREC-COLING, 2024.

[15] Y.-X. Lin, C.-A. Li, S.-L. Wei, P.-C. Chen, H.-H. Chen, and H.- y. Lee, “Hearing the order: Investigating position bias in large audio-language models,” arXiv preprint arXiv:2510.00628, 2025.

[16] N. Chandak, S. Goel, A. Prabhu, M. Hardt, and J. Geiping, “Answer matching outperforms multiple choice for language model evaluation,” arXiv preprint arXiv:2507.02856, 2025.

[17] S. Sedl<sup>ˇ</sup> a´cek, S. Barahona, B. Yusuf, L. Herrera-Alarcˇ on, S. Ke-´ siraju, C. Bolanos, A. Lozano-Diez, S. Udupa, F. L ˜ opez, et al., ´ “ORCA: Open-ended response correctness assessment for audio question answering,” arXiv preprint arXiv:2512.09066, 2025.

[18] L. H.-Y. Foo, C.-K. Yang, C.-A. Li, K.-H. Lu, and H.-y. Lee, “All that glitters is not audio: Rethinking text priors and audio reliance in audio-language evaluation,” arXiv preprint arXiv:2604.24401, 2026.

[19] A. Goel, S. Ghosh, J. Kim, S. Kumar, Z. Kong, S.-g. Lee, C.- H. H. Yang, et al., “Audio flamingo 3: Advancing audio intelligence with fully open large audio language models,” arXiv preprint arXiv:2507.08128, 2025.

[20] A. Abouelenin, A. Ashfaq, A. Atkinson, H. Awadalla, N. Bach, J. Bao, A. Benhaim, M. Cai, V. Chaudhary, C. Chen, et al., “Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs,” arXiv preprint arXiv:2503.01743, 2025.

[21] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin, et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[22] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, et al., “Qwen2.5-omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[23] C.-Y. Kuan and H.-y. Lee, “AQUA-Bench: Beyond finding answers to knowing when there are none in audio question answering,” in ICASSP, 2026.

[24] S. Sakshi, U. Tyagi, S. Kumar, A. Seth, R. Selvakumar, O. Nieto, R. Duraiswami, S. Ghosh, and D. Manocha, “MMAU: A massive multi-task audio understanding and reasoning benchmark,” in ICLR, 2025.

[25] Z. Ma, Y. Ma, Y. Zhu, C. Yang, Y.-W. Chao, R. Xu, W. Chen, Y. Chen, Z. Chen, J. Cong, et al., “MMAR: A challenging benchmark for deep reasoning in speech, audio, music, and their mix,” arXiv preprint arXiv:2505.13032, 2025.

[26] D. Wang, J. Li, J. Wu, D. Yang, X. Chen, T. Zhang, and H. Meng, “MMSU: A massive multi-task spoken language understanding and reasoning benchmark,” arXiv preprint arXiv:2506.04779, 2025.

[27] C.-K. Yang, N. Ho, Y.-T. Piao, and H.-y. Lee, “SAKURA: On the multi-hop reasoning of large audio-language models based on speech and audio information,” arXiv preprint arXiv:2505.13237, 2025.

[28] C. Zheng, H. Zhou, F. Meng, J. Zhou, and M. Huang, “Large language models are not robust multiple choice selectors,” in ICLR, 2024.

[29] L. Kuhn, Y. Gal, and S. Farquhar, “Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation,” in ICLR, 2023.

## 8. COMPLIANCE WITH ETHICAL STANDARDS

This study used publicly available data from AQUA-Bench, MMAU, MMAR, MMSU, and SAKURA, together with publicly released open-weight models. No new human- or animal-subject data were collected; therefore, ethical approval was not required. Weiran Wang is supported by a Google Gift Award. The authors declare no conflicts of interest.