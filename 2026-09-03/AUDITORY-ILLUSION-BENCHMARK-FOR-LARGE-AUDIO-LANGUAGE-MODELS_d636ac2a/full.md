![](images/c85b7ff3815fdc088755644f18ad1ecdd1cf0bea15552a13899bdb342766e14b.jpg)

# AUDITORY ILLUSION BENCHMARK FOR LARGE AUDIO LANGUAGE MODELS

Hayoon Kim<sup>♭♮</sup>, Eunice Hong<sup>♭♮</sup>, Kyogu Lee<sup>♭♮♯†</sup>

<sup>♭</sup>Music and Audio Research Group, Seoul National University   
<sup>♮</sup>Department of Intelligence and Information, Seoul National University   
<sup>♯</sup>AIIS, Seoul National University <sup>†</sup>IPAI, Seoul National University {hyway, euniunie, kglee}@snu.ac.kr

## ABSTRACT

Perceptual illusions have long served as crucial probes into human cognition, revealing biases and limitations of perception. In the auditory domain, such illusions provide a unique lens for testing whether Large Audio Language Models (LALMs) replicate human perceptual tendencies. Despite their importance, most benchmarks focus on visual illusions or general audio tasks, leaving auditory illusions underexplored. To this end, we present AIB, the first auditory illusion benchmark for LALMs, covering ten representative illusions across music, sound, and speech, each annotated for the presence of knowledge-based priors. Our methodology pairs model evaluation with controlled human listening studies, enabling direct comparison of responses. Results show systematic differences: while most LALMs remain signal-faithful on low-level acoustic illusions, several exhibit more human-like responses when linguistic or musical priors are involved, although no model matches the human perceptual profile. These findings highlight the current limitations of LALMs as cognitive models. By establishing auditory illusions as a rigorous testbed, our work offers a new perspective for probing neural black-box models and advancing understanding of auditory cognition. AIB is publicly available at https://github.com/gillosae/aib.

Index Terms— Illusion, Auditory Illusion, Large Audio Language Models, Benchmark, Dataset

## 1. INTRODUCTION

Perceptual illusions offer cognitive science insight into the mechanisms and biases underlying human perception [1, 2, 3]. In the auditory domain, classic illusions such as the Shepard tone [4] or the missing fundamental [5] illustrate how the brain integrates acoustic signals and prior knowledge to form coherent percepts that diverge from the physical stimulus [6], revealing both the constructive nature of auditory perception and its systematic fallibility.

Recent advances have produced Large Audio Language Models (LALMs) with impressive capabilities in automatic speech recognition, audio captioning, and music understanding. Although these models show limited success in tasks with clear ground truth [7], little is known about whether they also internalize human-like perceptual biases, including susceptibility to illusions.

Existing audio and multimodal benchmarks focus primarily on tasks with objective, signal-level ground truth, such as transcription accuracy, classification, or retrieval [8]. In contrast, illusions pose a fundamentally different question: whether systems replicate the same subjective misperceptions observed in humans, a property already explored for computer vision models but not yet systematically investigated in the auditory domain.

Fig. 1. Representative samples from the AIB benchmark, featuring both illusion-inducing (e.g., Missing fundamental, Zwicker tone, and Tartini tone) and control stimuli. The benchmark queries human participants and AI models on their perceptual experiences to evaluate how closely AI responses align with human-like auditory perception.

To address this gap, we introduce the first auditory illusion benchmark (AIB) for LALMs. AIB spans ten representative illusions from prior literature, categorized along two axes: underlying mechanism (physics-based vs. physics+knowledge-based) and perceptual content (music, sound, speech). Figure 1 shows example prompts. We pair model evaluations with controlled human listening studies, enabling direct comparison between models and humans.

Our results show systematic divergences between humans and LALMs. For low-level acoustic illusions, humans consistently exhibited strong susceptibility, while models ranged from partial alignment to strict adherence to the physical signal. In contrast, illusions that additionally rely on linguistic or musical priors elicited more human-like responses in several models, although no model matched the human perceptual profile. Notably, some models shifted from signal-faithful behavior on physics-based illusions to greater illusion susceptibility on knowledge-driven ones, suggesting that this alignment may arise from higher-level learned priors rather than shared low-level auditory processing.

First, we introduce a benchmark that establishes auditory illusions as a new dimension of LALM evaluation, together with the opensource implementations for systematically generating the benchmark illusions. Second, we provide human baseline data through controlled perceptual studies. Third, we present empirical analyses showing that LALM illusion susceptibility diverges systematically from humans, highlighting both progress in perceptual modeling and persistent gaps in human-likeness. By situating auditory illusions as a rigorous testbed, our work extends evaluation beyond recognition accuracy and towards perceptual alignment, contributing to both machine learning and cognitive science perspectives on auditory perception.

## 2. RELATED WORK

Auditory Illusions. Auditory illusions reveal systematic mismatches between acoustic input and perception, providing powerful probes into auditory mechanisms. Early accounts, such as Tartini’s third tone, were later explained by Helmholtz as non-linear responses of the auditory system [9]. Modern systematic research was pioneered by Deutsch, who discovered the octave [10], scale [11], and tritone paradox [12], showing variation across brain organization, handedness, and linguistic background. Other canonical examples include the missing fundamental, Shepard tone, Risset rhythm [13], Zwicker tone [14], and temporal gap illusion [15], often interpreted via Gestalt principles of closure [16]. Speech-related illusions such as phonemic restoration, speech-to-song [17, 18], phantom words, and high-profile cases like the Yanny/Laurel debate [19] further highlight how perception imposes linguistic structure on ambiguous inputs. These findings establish illusions as rigorous tools for studying auditory cognition, though certain binaural or human-specific illusions remain difficult to model computationally. Beyond their role in basic psychoacoustics, auditory illusions have also been used in neuroscience to study neural adaptation [20], in HCI to probe auditory attention and perceptual learning [1], and in cross-linguistic settings to reveal how cultural and linguistic priors shape auditory interpretation [12]. This broad applicability underscores their value as diagnostic tools not only for human perception but also for evaluating artificial systems.

Large Audio Language Models. LALMs extend transformer-based architectures into the auditory domain by combining multimodal encoders with large-scale training [21]. AudioPaLM [22] unified recognition and translation by transferring linguistic and paralinguistic knowledge, while Qwen-Audio [23] supported multitask understanding across speech, music, and environmental audio. Audio Flamingo [24] further extended these capabilities to in-context dialogue about audio. More recent work has emphasized reasoning: GAMA-IT [25] used synthetic datasets for instruction tuning to reduce hallucinations, and Audio-Reasoner [26] introduced CoTA [27] to enhance inference in audio QA. While these systems demonstrate strong performance across diverse auditory tasks, their evaluation remains primarily recognition-oriented, leaving open whether they capture human-like perceptual fallibilities. These systems differ not only in training objectives but also in their inductive biases: dual-encoder designs emphasize modality alignment, while unified encoder-decoders encourage transfer across tasks. Reasoning-tuned models such as GAMA or Audio-Reasoner suggest that chain-of-thought methods can reduce hallucination, yet none of these approaches directly evaluate whether models internalize human-like perceptual biases such as illusions. Addressing this gap is critical if LALMs are to serve as cognitive models of auditory processing.

Auditory or Illusion Benchmarks. Benchmarking in audio has evolved from recognition to reasoning and instruction following. SUPERB [28] evaluates a wide range of speech and audio tasks, while AudioBench [29] unifies evaluation across 26 datasets with both objective metrics and LLM-based judging. Recent efforts such as MMAU [30], AIR-Bench [31], MMAR [32] emphasize inference and multi-step reasoning across speech, music, and environmental audio. CoTA [26] provides reasoning-oriented captions and chain-of-thought annotations to advance emergent reasoning. Despite their breadth, these benchmarks remain anchored in tasks with objective ground truth labels, such as classification or transcription. They cannot assess whether models replicate systematic human misperceptions. In contrast, illusion-focused benchmarks in vision [33, 34, 35] explicitly probe subjective misalignment between models and humans.

In vision-language research, illusion-focused benchmarks have directly tested perceptual biases. IllusionBench [33] demonstrated that larger VLMs align more closely with human susceptibility to classic illusions, while HallusionBench [34] revealed persistent mismatches where models hallucinate or miss illusions. These works highlight illusions as probes of perceptual alignment. However, no systematic auditory benchmark currently examines whether LALMs perceive auditory illusions like humans. Our work addresses this gap by introducing the first auditory illusion benchmark and comparing model predictions with human judgments.

## 3. BENCHMARK DATASET

Overview. The AIB benchmark dataset comprises ten representative auditory illusions, including 10,385 illusion stimuli and 4,444 control stimuli, as detailed in Table 1. Each illusion belongs to one of three domains—music, speech, or general sound—and is further categorized by underlying mechanism (physics-based vs. physics+knowledge-based) as detailed in Section 3. For every illusion, both stimulus and matched control sets were collected to ensure controlled conditions, enabling systematic comparison of human judgments and model predictions.

Data Curation and Illusion Classification. We compiled a set of academically recognized auditory illusions for this study, excluding those that fundamentally depend on binaural physiology (e.g., the octave illusion) or on mechanisms beyond the representational capacity of current computational models, such as spatial-hearing–specific effects [10]. This ensured that all selected illusions could be robustly presented, measured, and compared across both human listeners and available LALMs.

Although auditory illusions have been extensively investigated [17], systematic classification has been more thoroughly developed in the visual domain. In vision science, Gestalt principles have provided influential taxonomies of illusion types [38]. However, for our purposes, Gregory’s causal framework—distinguishing illusions by their origins—offers a more direct mapping to auditory phenomena [39]. Gregory categorized illusions into two groups: those arising from physical or physiological factors (e.g., optical distortions, sensory signal disturbances) and those driven by knowledge, rules, or prior experience. The former are stimulus-driven and bottom-up, whereas the latter emerge from top-down processes that impose expectations or contextual knowledge on ambiguous input [40].

Adapting this framework to the auditory domain, we define two categories: (i) Physics-based auditory illusions, where percepts arise from acoustic properties or basic auditory physiology, including frequency interactions, masking, and temporal adaptation; and (ii) Physics+Knowledge-based auditory illusions, where percepts are also shaped by prior knowledge or expectations, such that top-down mechanisms impose linguistic, musical, or contextual structure on incomplete input. Each illusion in our benchmark was systematically assigned to one of these categories based on its dominant causal mechanism, following Gregory’s distinction between bottom-up and top-down processes in perception.

We emphasize that this framework reflects the dominant explanatory mechanism rather than a strict dichotomy. In ambiguous cases, classification is based on the dominant causal contribution rather than exclusivity. This classification strategy ensures that illusions can be robustly benchmarked across both human listeners and artificial systems.

Table 1. Statistical summary of the AIB dataset. For each illusion we list its domain, its dominant mechanism (P: Physics; P+K: Physics+Knowledge), and the number of illusion and control stimuli in the full benchmark and in the mini subset.
<table><tr><td rowspan="2">Domain</td><td rowspan="2">Illusion</td><td colspan="2">Mechanism</td><td colspan="3">Full set</td><td colspan="3">Mini set</td></tr><tr><td>P</td><td>P+K</td><td>Illusion</td><td>Control</td><td>Total</td><td>Illusion</td><td>Control</td><td>Total</td></tr><tr><td rowspan="4">Music</td><td>Missing fundamental [5]</td><td>√</td><td></td><td>1,080</td><td>178</td><td>1,258</td><td>108</td><td>88</td><td>196</td></tr><tr><td>Risset rhythm [13]</td><td></td><td>√</td><td>960</td><td>144</td><td>1,104</td><td>96</td><td>23</td><td>119</td></tr><tr><td>Shepard tone [4]</td><td></td><td>√</td><td>1,296</td><td>518</td><td>1,814</td><td>127</td><td>56</td><td>183</td></tr><tr><td>Tartini tone [36]</td><td>√</td><td></td><td>720</td><td>179</td><td>899</td><td>72</td><td>10</td><td>82</td></tr><tr><td rowspan="4">Sound</td><td>Temporal gap [15]</td><td></td><td>√</td><td>1,350</td><td>540</td><td>1,890</td><td>135</td><td>48</td><td>183</td></tr><tr><td>Zwicker tone [14]</td><td>√</td><td></td><td>432</td><td>61</td><td>493</td><td>43</td><td>43</td><td>86</td></tr><tr><td>Tempo change illusion [37]</td><td></td><td>√</td><td>896</td><td>537</td><td>1,433</td><td>89</td><td>66</td><td>155</td></tr><tr><td>Continuity illusion [20]</td><td></td><td>√</td><td>432</td><td>172</td><td>604</td><td>36</td><td>16</td><td>52</td></tr><tr><td rowspan="2">Speech</td><td>Speech-to-song [17]</td><td></td><td>√</td><td>2,304</td><td>1,766</td><td>4,070</td><td>234</td><td>150</td><td>384</td></tr><tr><td>Phonemic restoration [6]</td><td></td><td>√</td><td>915</td><td>349</td><td>1,264</td><td>92</td><td>34</td><td>126</td></tr><tr><td>Total</td><td>10 illusions</td><td>3</td><td>7</td><td>10,385</td><td>4,444</td><td>14,829</td><td>1,032</td><td>534</td><td>1,566</td></tr></table>

## 4. EXPERIMENTS

Evaluation Setup. We evaluated a set of LALMs alongside proprietary baselines, which have demonstrated strong performance on multimodal audio benchmarks. Models were presented with the same stimuli as human listeners. To adapt illusions into machineinterpretable tasks, we reformulated each perceptual judgment as a multiple-choice question, limited to two formats: (i) binary choices, such as yes/no or same/different judgments, and (ii) ternary choices, such as ascending/descending/no change for pitch-related illusions. This setup ensures direct comparability with human responses and follows the design principles of prior illusion benchmarks [33, 34, 35], adapted here to the auditory domain. Model outputs were parsed into the answer set; unparseable responses were excluded.

Metrics. Model responses were compared against human response distributions. We report: Human-Likeness Accuracy (HLA): proportion of model outputs matching the dominant human response. Reality Alignment (RA): proportion of outputs aligned with the physical ground truth (illusion-free). Illusion Susceptibility Index (ISI): difference between HLA and RA, quantifying the extent to which models share human-like misperceptions. We frame ISI as a descrip tive measure of human-likeness rather than a monolithic performance metric; high or low ISI values are task-dependent and should be interpreted in the context of the specific application goals.

Human Listening Study. To establish perceptual baselines, we recruited 20 participants possessing absolute pitch for controlled human experiments. Absolute-pitch listeners were chosen because several illusions (e.g., Shepard and Tartini tones) require reliable note-level judgments, allowing us to separate perceptual susceptibility from pitch-labeling error. Participants completed a randomized trial sequence that included illusion and control stimuli, presented without feedback so that responses reflected immediate percepts rather than learned strategies. The experiment was conducted via a web-based interface, and participants were instructed to use headphones in a quiet environment to ensure signal clarity. For each trial, they reported categorical judgments (e.g. ‘up’ vs. ‘down’ in Shepard tones, ‘complete’ vs. ‘incomplete’ in phonemic restoration). To ensure data reliability, we aggregated responses using majority voting to construct robust ground-truth distributions for each illusion.

## 5. RESULTS

Table 2 summarizes our evaluation. Since the human reference defines HLA and ISI as 1.0 by construction, the question is how far each model departs from it, and whether departures reflect a genuine percept or a response bias, which the matched controls diagnose. No model approaches the human profile: the best average ISI (Audio Flamingo 3, 0.455) is less than half the human reference, and control accuracy is modest throughout, with several models below chance. ISI must therefore be read jointly with control accuracy: a degenerate strategy that always gives the illusion-consistent answer attains an average ISI of 0.799 while failing Physics controls almost entirely. This instability indicates that robust auditory understanding in LALMs remains an open challenge, but also positions the benchmark as a forward-looking diagnostic.

On physics-based illusions, only MuLLaMa and Audio Flamingo 3 share the human-like percept, and both do so with below-chance control accuracy, so part of their susceptibility is answer bias rather than perception. All remaining models side with the physical signal, and the most signal-faithful models are also those with the highest control accuracy: they act as detectors of what is physically present (e.g., the absent fundamental) rather than of what a listener hears.

Illusions that additionally rely on top-down priors elicit the most human-like behavior. Audio Flamingo 3 attains the highest ISI (0.505) with above-chance controls, and Gemini 3.1 Pro the highest HLA (0.707). Notably, Gemini 3.1 Pro, Kimi-Audio-Instruct, and Fun-Audio-Chat reverse sign between domains: strongly signal-faithful on acoustic illusions yet human-like once linguistic or musical expectations are involved, suggesting alignment driven by language-level priors rather than shared low-level auditory processing. Qwen2- Audio-Instruct is the most consistently literal model, pairing negative ISI in both domains with the highest control accuracy, whereas GLM-4-Voice is simply unstable, with negative ISI and far below-chance controls.

Re-evaluating three checkpoints with refined prompts (last block of Table 2) shows that wording alone can shift ISI by as much as 0.6, more than most between-model differences: Qwen2-Audio-Instruct collapses to a purely physical reading with perfect Physics control accuracy. Susceptibility of instruction-tuned models is thus a property of the model-prompt pair rather than of the checkpoint alone.

Table 2. Performance of all evaluated LALMs on auditory illusions, categorized by Physics and Physics+Knowledge domains. All retained illusion stimuli were validated as eliciting the intended percept in the human listening study, so the human response becomes the illusion label. † marks models whose free-text replies could not be mapped onto the answer set for more than 20% of trials; their scores understate failure rather than measuring perception. The highest ISI and control accuracy among the evaluated models in each column are shown in bold.
<table><tr><td rowspan="3">Models</td><td rowspan="3">Size</td><td colspan="3">Physics</td><td colspan="4">Physics+Knowledge</td><td colspan="4">Average</td></tr><tr><td colspan="2">Illusion</td><td rowspan="2">ISI</td><td rowspan="2">Control HLA</td><td colspan="2">Illusion</td><td rowspan="2">Control</td><td rowspan="2">HLA</td><td rowspan="2">Illusion</td><td rowspan="2"></td><td rowspan="2">ISI</td></tr><tr><td>HLA</td><td>RA</td><td></td></tr><tr><td>Baselines and human reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random Guess</td><td></td><td>0.500 0.500</td><td>0.000</td><td>0.500</td><td>0.436</td><td>0.436</td><td>0.000</td><td>0.450</td><td>0.449</td><td>0.449</td><td>0.000</td><td>0.455</td></tr><tr><td>Most Frequent Choice</td><td></td><td>1.000</td><td>0.000</td><td>1.000</td><td>0.110</td><td>0.744 0.000</td><td>0.744</td><td>0.607</td><td>0.799</td><td>0.000</td><td>0.799</td><td>0.560</td></tr><tr><td>Human</td><td></td><td>1.000</td><td>0.000</td><td>1.000</td><td>1.000</td><td>1.000 0.000</td><td>1.000</td><td>1.000</td><td>1.000</td><td>0.000</td><td>1.000</td><td>1.000</td></tr><tr><td>Large Audio Language Models (LALMs)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pengi [41]†</td><td>323M</td><td>0.000</td><td>0.014</td><td>-0.014</td><td>0.014</td><td>0.278 0.177</td><td>0.101</td><td>0.153</td><td></td><td>0.2180.142</td><td>0.076</td><td>0.140</td></tr><tr><td>Voxtral-Mini [42]</td><td>3B</td><td>0.463</td><td>0.478 -0.015</td><td></td><td>0.526</td><td>0.357 0.509</td><td>-0.152</td><td>0.602</td><td>0.380</td><td>0.503</td><td>-0.123</td><td>0.595</td></tr><tr><td>MuLLaMa [43]</td><td>7B</td><td>0.659 0.341</td><td></td><td>0.319</td><td>0.368</td><td>0.481 0.274</td><td>0.207</td><td>0.432</td><td>0.519</td><td>0.288</td><td>0.231</td><td>0.426</td></tr><tr><td>Kimi-Audio-Instruct [44]</td><td>7B</td><td>0.124</td><td>0.876</td><td>-0.752</td><td>0.782</td><td>0.540 0.279</td><td>0.262</td><td>0.621</td><td>0.451</td><td>0.407</td><td>0.044</td><td>0.636</td></tr><tr><td>Audio Flamingo 3 [24]</td><td>8B</td><td>0.584</td><td>0.313</td><td>0.271</td><td>0.404</td><td>0.598 0.093</td><td>0.505</td><td>0.592</td><td>0.595</td><td>0.140</td><td>0.455</td><td>0.574</td></tr><tr><td>Fun-Audio-Chat [45]</td><td>8B</td><td>0.008 0.676</td><td>-0.668</td><td>0.553</td><td>0.450</td><td>0.410</td><td>0.039</td><td>0.589</td><td>0.355</td><td>0.467</td><td>-0.112</td><td>0.585</td></tr><tr><td>Qwen-Audio-Chat [23]</td><td>8.4B</td><td>0.427 0.573</td><td>-0.146</td><td>0.548</td><td>0.345</td><td>0.381</td><td>-0.037</td><td>0.512</td><td>0.362</td><td>0.423</td><td>-0.060</td><td>0.516</td></tr><tr><td>Qwen2-Audio-Instruct [46]</td><td>8.4B</td><td>0.228 0.769</td><td>-0.541</td><td></td><td>0.727</td><td>0.335 0.565</td><td>-0.230</td><td>0.642</td><td>0.312</td><td>0.609</td><td>-0.297</td><td>0.650</td></tr><tr><td>GLM-4-Voice [47]</td><td>9B</td><td>0.321 0.485</td><td></td><td>-0.164</td><td>0.488</td><td>0.399 0.582</td><td>-0.184</td><td>0.227</td><td>0.382</td><td>0.561</td><td>-0.179</td><td>0.251</td></tr><tr><td>Large Language Models (LLMs) Gemini 3.1 Pro [48]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.261</td><td>0.739</td><td>-0.478</td><td>0.737</td><td>0.7070.322</td><td>0.385</td><td>0.535</td><td>0.612</td><td>0.412</td><td>0.200</td><td>0.554</td></tr><tr><td>Same checkpoints, refined prompt wording</td><td>323M</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Pengi† Qwen-Audio-Chat</td><td>8.4B</td><td>0.194 0.509</td><td>0.014 0.491</td><td>0.179 0.018</td><td>0.014 0.498</td><td>0.330 0.135 0.377 0.495</td><td>0.195 -0.119</td><td>0.146 0.608</td><td>0.301 0.405</td><td>0.109</td><td>0.192</td><td>0.134</td></tr><tr><td>Qwen2-Audio-Instruct</td><td>8.4B</td><td>0.000</td><td>1.000</td><td></td><td>1.000</td><td>0.923</td><td>-0.876</td><td>0.383</td><td>0.040</td><td>0.494 0.936</td><td>-0.089</td><td>0.597</td></tr><tr><td></td><td></td><td></td><td></td><td>-1.000</td><td></td><td>0.047</td><td></td><td></td><td></td><td></td><td>-0.896</td><td>0.417</td></tr></table>

Across domains, three regimes emerge: susceptible models (Audio Flamingo 3, MuLLaMa) with positive ISI but weak controls; literal models (Qwen2-Audio-Instruct, Voxtral-Mini, Qwen-Audio-Chat) with negative ISI and above-chance controls; and domaindependent models (Gemini 3.1 Pro, Kimi-Audio-Instruct) that are literal on acoustic illusions but susceptible on knowledge-driven ones. No model combines high susceptibility with reliable controls, which is precisely the human profile. Since same-scale models span the entire ISI range, progress likely requires not scale alone but training that integrates perceptual priors with reliable signal-level judgment.

## 6. DISCUSSION

Our results highlight both the promise and limitations of current LALMs when evaluated through auditory illusions. The divergences between human and model perception vary systematically by illusion type. For physics-based cases, only a few models (e.g., MuL-LaMa, Audio Flamingo 3) reproduced human-like misperceptions, while most—including recent systems such as Gemini 3.1 Pro and Kimi-Audio-Instruct—aligned with the physical signal. Illusions that required the integration of acoustic and prior-based cues elicited the most human-like responses, but the alignment remained partial (at best an ISI of 0.505) and, for several models, reversed the sign observed on physics-based illusions, suggesting that it is driven by linguistic and musical priors rather than by shared low-level auditory processing. The same checkpoint could also shift from moderate to near-complete signal fidelity under a reworded prompt. Notably, many models showed unstable performance even under control conditions, indicating that LALMs are still at an early stage of development.

These findings have broader implications for cognitive science and machine learning. From a cognitive perspective, illusions are a powerful diagnostic tool for probing auditory mechanisms, revealing which aspects of perception models capture and which remain elusive. From an engineering perspective, illusion susceptibility offers a complementary evaluation axis. It is important to note that the primary goal of AIB is to measure cognitive alignment rather than general task performance. The desirability of alignment depends on the application: In Human-Computer Interaction, higher alignment (high ISI) may support more natural and empathetic interaction. Conversely, in safety-critical settings or precision measurement tasks, reduced susceptibility (low ISI) may be preferable. Thus, AIB serves as a diagnostic tool to position models along this spectrum. We offer this benchmark as an initial step that may support the development of both cognitively grounded and task-appropriate LALMs.

Conclusion. We introduced an auditory illusion benchmark for LALMs, pairing controlled human studies with systematic model evaluation. While models capture low-level regularities, they diverge in knowledge-driven illusions, underscoring limits in perceptual alignment. By framing illusions as diagnostic tools, we offer a complementary perspective for evaluating audio models and point to future work in expanding coverage, strengthening human baselines, and exploring reasoning-oriented training. We also highlight the value of illusions as probes of human-likeness in machine perception. Similar to visual benchmarks, auditory illusions reveal whether models reflect human biases or adhere strictly to the signal. We hope this benchmark can serve as a useful step toward advancing LALMs beyond recognition accuracy to more cognitively aligned auditory understanding.

## 7. ACKNOWLEDGEMENTS

This work was partly supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) [No. RS-2024-00461617, 50%], [No. RS-2025-24683892, 45%], and Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [NO.RS-2021-II211343, 5%]. The authors would also like to thank Kihong Kim for his helpful discussions and support.

## 8. REFERENCES

[1] David M. Eagleman, “Visual illusions and neurobiology,” Nature Reviews Neuroscience, 2001.

[2] Ahyeon Choi et al., “The effects of musical factors on the perception of auditory illusions,” Topics in Cognitive Science, vol. 17, no. 1, pp. 106–119, 2025.

[3] You Jeong Hong et al., “Concurrent musical pitch height biases judgment of visual brightness,” Psychol. of Music, vol. 53, no. 3, pp. 492–502, 2025.

[4] Roger N Shepard, “Circularity in judgments of relative pitch,” JASA, 1964.

[5] Dante R Chialvo, “How we hear what is not there: A neural mechanism for the missing fundamental illusion,” Chaos: An Interdisciplinary J. ofNonlinear Science, 2003.

[6] Richard M Warren and Charles J Obusek, “Speech perception and phonemic restorations,” Perception & Psychophysics, 1971.

[7] Yongyi Zang, Sean O’Brien, Taylor Berg-Kirkpatrick, Julian McAuley, and Zachary Novack, “Are you really listening? boosting perceptual awareness in music-qa benchmarks,” 2025.

[8] Jort F. Gemmeke et al., “Audioset: An ontology and humanlabeled dataset for audio events,” in IEEE ICASSP, 2017.

[9] Ben Carson, “What are musical paradox and illusion?,” 2007.

[10] Diana Deutsch, “The octave illusion and auditory perceptual integration,” Hearing research and theory, 1981.

[11] Diana Deutsch, “Musical illusions,” Scientific American, 1975.

[12] Diana Deutsch, “The tritone paradox: An influence of language on music perception,” Music perception, 1991.

[13] JC Risset, “Paradoxes de hauteur (with sound examples),” IR-CAM Rep, 1978.

[14] Jan-Moritz P Franosch et al., “Zwicker tone illusion and noise reduction in the auditory system,” Physical review letters, 2003.

[15] C Formby and TG Forrest, “Detection of silent temporal gaps in sinusoidal markers,” Sci. the Acoustical Soc. ofAmerica, 1991.

[16] Angélique A Scharine and Tomasz R Letowski, “Auditory conflicts and illusions,” Helmet-mounted displays: sensation, perception and cognition issues, 2009.

[17] Diana Deutsch et al., “Illusory transformation from speech to song,” Sci. the Acoustical Soc. ofAmerica, 2011.

[18] Haesun Joung et al., “Exploring the speech-to-song illusion: A comparative study of standard korean and dialects,” in Proceedings ofthe Annual Meeting ofthe Cognitive Science Soc., 2025, vol. 47.

[19] Adrian Leemann et al., “Factors affecting the percept of yanny v. laurel (or mixed): Insights from a large-scale study on swiss german listeners.,” in INTERSPEECH, 2022, pp. 1851–1855.

[20] Fatima T Husain et al., “Investigating the neural basis of the auditory continuity illusion,” J. of Cognitive Neuroscience, 2005.

[21] Alec Radford et al., “Robust speech recognition via large-scale weak supervision,” 2022.

[22] Paul K Rubenstein et al., “Audiopalm: A large language model that can speak and listen,” arXiv:2306.12925, 2023.

[23] Yunfei Chu et al., “Qwen-audio: Advancing universal audio understanding via unified large-scale audio-language models,” arXiv:2311.07919, 2023.

[24] Zhifeng Kong et al., “Audio Flamingo: A novel audio language model with few-shot learning and dialogue abilities,” arXiv:2402.01831, 2024.

[25] Sreyan Ghosh et al., “Gama: A large audio-language model with advanced audio understanding and complex reasoning abilities,” arXiv:2406.11768, 2024.

[26] Zhifei Xie et al., “Audio-Reasoner: Improving reasoning capability in large audio language models,” arXiv:2503.02318, 2025.

[27] Ziyang Ma et al., “Audio-CoT: Exploring chain-of-thought reasoning in large audio language models,” arXiv:2501.07246, 2025.

[28] Shu-wen Yang et al., “SUPERB: Speech processing universal performance benchmark,” arXiv:2105.01051, 2021.

[29] Bin Wang et al., “AudioBench: A universal benchmark for audio large language models,” arXiv:2406.16020, 2024.

[30] S Sakshi et al., “MMAU: A massive multi-task audio understanding and reasoning benchmark,” arXiv:2410.19168, 2024.

[31] Qian Yang et al., “Air-bench: Benchmarking large audio-language models via generative comprehension,” arXiv:2402.07729, 2024.

[32] Ziyang Ma et al., “MMAR: A challenging benchmark for deep reasoning in speech, audio, music, and their mix,” arXiv:2505.13032, 2025.

[33] Yiming Zhang et al., “IllusionBench: A large-scale and comprehensive benchmark for visual illusion understanding in visionlanguage models,” arXiv:2501.00848, 2025.

[34] Tianrui Guan et al., “HallusionBench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models,” in IEEE/CVF CVPR, 2024.

[35] Yichi Zhang et al., “Grounding visual illusions in language: Do vision-language models perceive illusions like humans?,” 2023.

[36] Max F Meyer, “Subjective tones: Tartini and beat-tone pitches,” The American J. ofPsychol., 1957.

[37] Marilyn G Boltz, “Illusory tempo changes due to musical characteristics,” Music Perception, 2011.

[38] Stanley Coren and Joan S Girgus, “Principles of perceptual organization and spatial distortion: the gestalt illusions.,” J. of Exp. Psychol.: Human Perception and Performance, 1980.

[39] Richard L Gregory, “Knowledge in perception and illusion,” Philosophical Transactions ofthe Royal Soc. ofLondon. Series B: Biological Sci., 1997.

[40] Richard L Gregory, “Visual illusions classified,” Trends in cognitive Sci., 1997.

[41] Soham Deshmukh et al., “Pengi: An audio language model for audio tasks,” NeurIPS, 2023.

[42] Alexander H Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, et al., “Voxtral,” arXiv preprint arXiv:2507.13264, 2025.

[43] Shansong Liu et al., “Music understanding llama: Advancing text-to-music generation with question answering and captioning,” in IEEE ICASSP, 2024.

[44] Ding Ding, Zeqian Ju, Yichong Leng, Songxiang Liu, Tong Liu, Zeyu Shang, Kai Shen, Wei Song, Xu Tan, Heyi Tang, et al., “Kimi-audio technical report,” arXiv preprint arXiv:2504.18425, 2025.

[45] Tongyi Fun Team, Qian Chen, Luyao Cheng, Chong Deng, Xiangang Li, Jiaqing Liu, Chao-Hong Tan, Wen Wang, Junhao Xu, Jieping Ye, et al., “Fun-audio-chat technical report,” arXiv preprint arXiv:2512.20156, 2025.

[46] Yunfei Chu et al., “Qwen2-audio technical report,” arXiv:2407.10759, 2024.

[47] Aohan Zeng, Zhengxiao Du, Mingdao Liu, Kedong Wang, Shengmin Jiang, Lei Zhao, Yuxiao Dong, and Jie Tang, “Glm-4-voice: Towards intelligent and human-like end-to-end spoken chatbot,” arXiv preprint arXiv:2412.02612, 2024.

[48] Google DeepMind, “Gemini 3.1 pro model card,” https://deepmind.google/models/model-cards/gemini-3-1- pro, 2026.