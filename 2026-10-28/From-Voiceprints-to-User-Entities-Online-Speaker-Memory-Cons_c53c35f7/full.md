# From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents

Yucong Zhang<sup>3,1,2,∗</sup>, Cancan Li<sup>3,2,</sup>\*, Bo Xiao<sup>2</sup>, Ke Zeng<sup>2</sup>, Ming Li<sup>1,†</sup> <sup>1</sup>School of Artificial Intelligence, The Chinese University of Hong Kong, Shenzhen, <sup>2</sup>Meituan, <sup>3</sup>School of Computer Science, Wuhan University {yucong.zhang, cancan.li}@whu.edu.cn, mingli369@cuhk.edu.cn

## Abstract

Personalized spoken agents must identify speakers before accessing user-specific memory, yet memory benchmarks typically assume correctly attributed histories. We argue that voiceprints are observations of identity, not identities themselves, and formulate online speaker-memory construction to maintain persistent user entities as acoustic evidence evolves. We introduce EvoSpeaker-Mem, a training-free layer for causal speaker routing, selective acoustic-state updates, and identity-partitioned memory writes and retrieval through a replaceable text-memory backend. We also construct LoCoMo-ID, a spoken-memory benchmark with hidden speaker identities, speaker-sensitive queries, simulated recording-condition shifts, and fourspeaker identity-confusion tests. On longitudinal SMIIP-TV, genuine-probe rejection under a frozen near-enrollment threshold rises from 29.8% to 55.6% over approximately one year. On LoCoMo-ID speaker-sensitive queries, gated updating improves identity-gated QA over fixed representations by 3.86 and 11.62 percentage points under clean and shifted conditions, respectively, with no wrong-user authorizations. The advantage persists in the four-speaker test. Yet reducing fragmentation and improving QA can coexist with incorrect cross-user links, revealing a fragmentation– contamination trade-off that QA alone cannot capture.

## 1 Introduction

Personalized spoken agents rely on user profiles and interaction histories to remain consistent across turns and sessions (Zhang et al., 2018; Xu et al., 2022). In shared spoken environments, however, maintaining such memory first requires deciding whose history each incoming observation belongs to. Speaker attribution is therefore a prerequisite for personalized memory construction. An identity error is stateful: treating a returning user as new fragments one person’s history across profiles, whereas confusing two users contaminates one profile with another user’s information and can affect subsequent personalized responses.

Recent audio large language models (Audio LLMs) have advanced speech understanding, yet speaker-centric reasoning remains challenging. MSU-Bench and M3-SLU reveal gaps in speaker verification, attribution, and reasoning about who said what (Sun et al., 2026; Kwon et al., 2025). VoxMem further evaluates multi-session memory for acoustic evidence, including speaker identity, and finds that current Audio LLMs retain spoken content more reliably than speaker information (Xiao et al., 2026). These findings motivate explicit speaker tools and persistent identity state.

Conversational-memory research has made substantial progress in organizing long histories. Lo-CoMo and LongMemEval evaluate retrieval, temporal reasoning, and knowledge updating across sessions (Maharana et al., 2024; Wu et al., 2025b); CarMem studies preference memory for voice assistants (Kirmayr et al., 2025), while spoken and multimodal evaluations extend memory beyond text (Kim et al., 2025; Bei et al., 2026). These benchmarks principally evaluate memory readout from histories with available participant attribution. A shared spoken agent must also construct that history by resolving each incoming speaker and routing content to the appropriate user’s memory.

Spoken dialogue systems connect identity with personalization: PAChat links speaker representations to profiles (Fu et al., 2025), AFA routes speech to per-user memories (Al-Ratrout et al., 2026), and EgoMem updates audiovisual memory (Yao et al., 2025). Concurrent VoxPolyMem combines incremental speaker identification, gated exponential moving averages, and session-level identity reconciliation (Jia et al., 2026). Yet the consequences of imperfect online speaker assignments for persistent user-memory integrity remain insufficiently understood.

A key challenge is that acoustic observations from the same speaker vary across sessions, environments, and user states. Speaker-verification research studies session variability, template aging, and adaptation (Kinnunen and Li, 2010; Das et al., 2016; Wang et al., 2016; Heo et al., 2024); SMIIP-TV provides recordings from 373 speakers to study such variation (Qin et al., 2024). Linking voiceprints to writable conversational memory adds persistent consequences: false rejection can duplicate a user’s profile, while false acceptance can misattribute information to another user.

We therefore argue that voiceprints are observations of identity, rather than identities themselves. We formulate personalized spoken interaction as online speaker-memory construction: an agent causally creates, links, and updates persistent user entities as acoustic evidence accumulates. EvoSpeakerMem routes each turn to an existing or new entity, selectively updates its acoustic state, and writes the attributed content to a replaceable textual-memory backend.

We evaluate the proposed formulation in three stages. First, we use MSU-Bench to examine the limitations of existing Audio LLMs in speakeridentity tasks, comparing them with an LLM equipped with a speaker-embedding tool. Second, SMIIP-TV evaluates how evolving acoustic evidence affects speaker-memory maintenance. Third, LoCoMo-ID evaluates the full pipeline from online speaker routing and user-partitioned memory construction to speaker-sensitive question answering. Our contributions are summarized as follows:

• We formulate online speaker-memory construction, in which voiceprints are treated as observations of persistent user entities, and introduce EvoSpeakerMem, a training-free speech support layer for chronological CRE-ATE/LINK routing and selective speaker-state updating.

• We organize SMIIP-TV into static verification and chronological multi-user replay evaluations, showing how acoustic variation challenges fixed speaker representations, motivates online entity maintenance, and reveals the fragmentation–contamination trade-off.

• We construct $\mathrm { L o C o M o - I D ^ { 1 } }$ , a spoken-memory benchmark with hidden speaker identities, and evaluate online speaker routing, userpartitioned memory construction, and identitygated retrieval under clean speech, recordingcondition shifts, and multi-speaker identityconfusion scenarios.

## 2 Method

## 2.1 Problem Formulation

We consider spoken turns $\{ x _ { t } \} _ { t = 1 } ^ { T }$ arriving chronologically. At time t, the system observes the current audio $x _ { t }$ and its existing state $\mathcal { U } _ { t - 1 }$ , without speaker labels or access to future turns. This state contains local user entities $\mathcal { U } _ { t - 1 } = \{ u _ { 1 } , . . . , u _ { K } \}$ each combining an acoustic representation with attributed textual memory:

$$
u _ { k } = ( \mathbf { c } _ { k } , n _ { k } , \mathcal { H } _ { k } ) ,\tag{1}
$$

where $\mathbf { c } _ { k }$ is a speaker-embedding center, $n _ { k }$ counts observations assigned to the entity, and $\mathcal { H } _ { k }$ stores atomic textual facts extracted from its attributed turns. The multi-exemplar variant below replaces the center with historical embeddings.

The routing decision $r _ { t } \in \{ u _ { 1 } , \ldots , u _ { K } , \mathrm { N E W } \}$ assigns the turn’s identity and memory partition. Voiceprints are observations of persistent entities. Incorrect linking can contaminate acoustic state and factual memory; incorrect new-entity creation fragments one user’s evidence across partitions.

## 2.2 Online Acoustic Entity Routing

A speaker encoder maps the incoming audio to a length-normalized embedding $\mathbf { e } _ { t }$ . For center-based policies, we score existing entities by cosine similarity:

$$
s _ { t , k } = \frac { \mathbf { e } _ { t } ^ { \top } \mathbf { c } _ { k } } { \left\| \mathbf { e } _ { t } \right\| \left\| \mathbf { c } _ { k } \right\| } .\tag{2}
$$

The system creates a new entity initialized with $\mathbf { e } _ { t }$ if the registry is empty or the highest entity score $s _ { t }$ falls below $\tau _ { \mathrm { { m e r g e } } } ;$ otherwise, it links the turn to the highest-scoring entity. We set $\tau _ { \mathrm { m e r g e } }$ at the equal-error-rate (EER) operating point computed from genuine and impostor speakerverification scores on reference voices, and keep it fixed throughout online processing.

Unrestricted updating can adapt a center to changing acoustic evidence, but an incorrect link can also make subsequent errors self-reinforcing. We therefore separate assignment from center update. After a turn is linked, the center changes only if $s _ { t } \geq \tau _ { \mathrm { u p d a t e } } .$ , where τ<sub>update</sub> > τ<sub>merge</sub> imposes a stricter requirement for modifying persistent acoustic state. An accepted update uses

![](images/d8af28f0dcb984a00f0918b4171a0eeea3e474844ac3ecdc0872b575b347e315.jpg)  
Figure 1: Overview of EvoSpeakerMem as a speech support layer for replaceable textual-memory backends. Acoustic routing assigns each turn to a persistent user entity before its content is written to memory.

$$
\mathbf { c } _ { k } \gets \frac { n _ { k } \mathbf { c } _ { k } + \mathbf { e } _ { t } } { n _ { k } + 1 } .\tag{3}
$$

The count $n _ { k }$ increases on every accepted LINK, including those that leave the center unchanged. Thus a turn can contribute textual facts without necessarily changing the acoustic representation used for future matching.

We compare FIXED, which retains the first embedding; UPDATE, which updates on every LINK; and GATED UPDATE, which applies the stricter update condition. A multi-exemplar alternative stores accepted embeddings in $\mathcal { E } _ { k }$ and scores an entity by the mean of its top min $( m , | { \mathcal { E } } _ { k } | )$ cosine similarities, for a fixed neighborhood size m. An embedding is appended only after the identity assignment. These policies change acoustic-state maintenance while retaining the same memory-write interface.

## 2.3 Speech Support Layer and Memory Construction

Figure 1 illustrates the speech-to-memory pipeline. For each spoken turn, an audio LLM (Xu et al., 2025b) generates a transcript, while a speakerembedding model extracts acoustic identity evidence. The router (Section 2.2) assigns the turn to an existing local entity or creates a new one. The transcript, assigned entity identifier, and timestamp are then passed to a replaceable text-memory backend (Fang et al., 2026), which extracts and indexes atomic facts within the corresponding user’s memory partition. Each entity thus maintains acoustic state and user-attributed textual memory, while transcript semantics do not influence acoustic routing or center updates.

At query time, read-only voice matching against the registry identifies the relevant entity and restricts retrieval to its memory partition; unmatched requests receive no personal memory. The same entity identifiers therefore govern both memory writes and reads. Appendix F provides transcription and routing prompts.

## 3 LoCoMo-ID Benchmark Construction

LoCoMo-ID combines query adaptation, identityconsistent speech synthesis, and two stress conditions targeting acoustic variation and multi-speaker interference.

## 3.1 Text Adaptation for Spoken Queries

Following the non-commercial license, we retain LoCoMo’s ten histories (5,882 turns), 1,986 QA items, and reference answers (Maharana et al., 2024)<sup>2</sup>. Rule-based query transformations replace explicit participant references with “I” or “my,” making the spoken requester’s identity necessary for answering while preserving the reference answer.

We rewrite 1,371 single-person questions into first-person sensitive queries and 22 sharedexperience questions into first-person-plural forms. The other 593 questions remain unchanged. Manual checks assess whether rewritten questions preserve the original meaning, participant reference, and answer. Appendix G gives transformation examples.

## 3.2 Voice Assignment and Speech Synthesis

We generate candidate reference voices with Qwen3-TTS VoiceDesign<sup>3</sup> and retain high-quality voice reference after ECAPA-based diversity filtering. Qwen3-TTS Base<sup>4</sup> then synthesizes the history and query audio from the assigned references (Hu et al., 2026). Each conversation’s two participants receive distinct voices, maintained across their history turns, sessions, and corresponding spoken queries.

Each conversation uses three voice assignments with identical dialogue text and QA content. Each conversation–realization pair initializes an independent speaker registry and textual memory. This yields 30 clean collections, 17,646 history audio turns, and 5,958 query audio items.

Quality control (QC) checks perceptual quality with UTMOS (Saeki et al., 2022) and speaker consistency against a calibrated reference-similarity threshold. A failing audio clip receives up to four additional synthesis attempts; if none passes both checks, we retain the highest-UTMOS candidate. QC targets usable, identity-consistent speech rather than downstream QA. Appendix G gives the voicegeneration models, prompts, and QC settings.

To assess content fidelity, we transcribe history and query audio with an automatic speech recognition (ASR) model whisper-large-v3 (Radford et al., 2023) and compute word error rate (WER) against source texts. Over 99.7% of utterances have WER below 0.3. Very short clips can have high WER from minor ASR mismatches, such as “bye Nate” recognized as “by Nate.” Manual listening confirmed correct spoken content in these cases, supporting synthesis fidelity.

## 3.3 Recording-condition Shifts

To test acoustic robustness beyond the clean reference, we apply simulated room, device, and environmental variations to history audio, preserving content and speaker assignments. No quality or reference-similarity filtering follows augmentation, so difficult acoustic observations are retained. This condition simulates recording variability to mimic complex scenarios during conversations. Detailed settings are provided in Appendix G.4.

## 3.4 Four-Speaker Interleaved Stress Test

We retain each original two-person story and construct a parallel story with the same conversational structure but controlled fact substitutions. The two stories share character names but use disjoint voice pairs, yielding four distinct acoustic identities with conflicting personal histories. This setting introduces deliberate identity ambiguity, testing whether the system maintains separate user memories despite overlapping names and conversational contexts.

We interleave the two stories at session boundaries while preserving the chronology of each story, requiring the system to track four identities across alternating sessions. Queries target facts from the original story, even when same-named speakers in the parallel story have conflicting information. Correct retrieval therefore depends on maintaining speaker-specific memory partitions rather than relying on textual names or shared context.

## 4 Experiments

## 4.1 Experimental Setup

Unless otherwise stated, all online routing, causal replay, and LoCoMo-ID experiments use the SpeechBrain ECAPA-TDNN checkpoint<sup>5</sup> (Desplanques et al., 2020) for acoustic embeddings. Appendices C.2 and C.3 report matched-set verification and replay checks with an more advanced speaker embedding model, ReDimNet2- B6 (Yakovlev and Okhotnikov, 2026). History audio is processed chronologically with speaker labels hidden.

For LoCoMo-ID, we cross four acoustic maintenance policies (FIXED, UPDATE, GATED UPDATE, and all-exemplar KNN) with two evidence configurations: per-turn ECAPA voiceprints and sessionlocal Parakeet(Wang et al., 2025) diarization with

<table><tr><td>Condition</td><td>Acoustic evidence</td><td>Metric</td><td>Fixed</td><td>Update</td><td>Gated update</td><td>All-exemplar KNN</td></tr><tr><td rowspan="5">Clean</td><td rowspan="3">ECAPA-only</td><td> $\overline { { F _ { B ^ { 3 } } \uparrow } }$ </td><td>97.04</td><td>98.41</td><td>99.54</td><td>98.51</td></tr><tr><td> $\mathrm { F N U R } \downarrow$ </td><td>0.77</td><td>0.27</td><td>0.31</td><td>0.21</td></tr><tr><td>Clusters/Gold</td><td>3.217</td><td>1.767</td><td>1.883</td><td>1.583</td></tr><tr><td rowspan="2"> $\overline { { F _ { B ^ { 3 } } } }$  ECAPA + diarization</td><td>↑</td><td>76.96</td><td>77.06</td><td>77.06</td><td>77.09</td></tr><tr><td> $\mathrm { F N U R } \downarrow$ </td><td>0.08</td><td>0.07</td><td>0.07</td><td>0.07</td></tr><tr><td rowspan="6">Condition- shifted</td><td rowspan="3">ECAPA-only</td><td>Clusters/Gold</td><td>1.150</td><td>1.117</td><td>1.117</td><td>1.117</td></tr><tr><td> $\overline { { F _ { B ^ { 3 } } \mathrm { ~ \AA ~ } } }$ </td><td>74.18</td><td>93.54</td><td>92.73</td><td>96.62</td></tr><tr><td> $\mathrm { F N U R } \downarrow$ </td><td>2.00</td><td>0.90</td><td>0.94</td><td>0.55</td></tr><tr><td rowspan="3">ECAPA + diarization</td><td>Clusters/Gold</td><td>6.833</td><td>3.600</td><td>3.733</td><td>2.567</td></tr><tr><td> $\overline { { F _ { B ^ { 3 } } \mathrm { ~ \AA ~ } } }$ </td><td>81.05</td><td>81.23</td><td>81.20</td><td>81.04</td></tr><tr><td> $\mathrm { F N U R } \downarrow$ </td><td>0.15</td><td>0.10</td><td>0.10</td><td>0.09</td></tr><tr><td colspan="2"></td><td>Clusters/Gold</td><td>1.317</td><td>1.167</td><td>1.167</td><td>1.150</td></tr></table>

Table 1: Online speaker-memory construction results on LoCoMo-ID clean speech and recording-shifted condition. Acoustic evidence distinguishes ECAPA-only routing from routing augmented with session-level diarization. We report partition quality $\left( F _ { B ^ { 3 } } \right)$ and fragmentation-related indicators (FNUR and Clusters/Gold). All rates are percentages except Clusters/Gold.

ECAPA-based cross-session association. We include diarization as an alternative speaker-linking approach that groups turns by speaker within each session. However, diarization labels are sessionlocal and cannot directly establish identity continuity across sessions, so we use ECAPA embeddings to associate these clusters with persistent user entities. Each session-local cluster is matched against the same pre-session registry snapshot before updates; the downstream memory and QA backend is otherwise shared. The transcription, memorybackend, embedding, answer-generation, judging, and decoding configurations are provided in $\mathsf { A p - }$ pendix E. Our implementation is available in an anonymous code repository<sup>6</sup>. ECAPA embedding extraction takes 65.6 ms per turn, excluding routing, transcription, and memory construction.

For synthesis QC, we compute the cosine similarity between each synthesized utterance and its assigned reference voice using ECAPA embeddings. We set the speaker-consistency threshold to 0.489, corresponding to the EER operating point calibrated on the VoiceDesign voice bank. This provides a verification-based criterion that balances false acceptance and false rejection on the calibration data domain. The threshold is used to assess whether synthesis preserves the intended speaker identity.

## 4.2 Causal Online Evaluation

We evaluate online speaker-memory construction using a causal first-seen binding protocol. When a local entity is first created, it is bound to the gold speaker of that turn only for post-hoc scoring. The binding is immutable and never affects routing, updating, or memory construction; a later assignment is correct only when it matches this binding. This avoids future-aware cluster matching, which could relabel entities after observing the full sequence and hide online routing errors.

Each turn is categorized into one of five mutually exclusive routing outcomes: new-speaker creation, new-speaker linking, known-speaker creation, correct known-speaker linking, and wrong-target linking. These outcomes capture two complementary failure modes of online entity maintenance: fragmentation, where a known speaker is assigned to a new entity, and contamination, where different speakers are merged into the same entity.

${ \mathbf B } ^ { 3 }$ F1 (Bagga and Baldwin, 1998) penalizes cross-speaker merging and within-speaker fragmentation. Clusters/Gold is the predicted-to-gold entity count ratio (ideal: 1). CREATE precision $( P _ { \mathrm { c r e a t e } } )$ is correct creations/all CREATES; recall $( R _ { \mathrm { c r e a t e } } )$ uses first-seen speaker turns as denominator. LINK precision $( P _ { \mathrm { l i n k } } )$ is correct links/all LINKS. Known-user recall $( R _ { \mathrm { k n o w n } } )$ and false-new user rate (FNUR) divide correct links and erroneous creations, respectively, by returning-speaker turns. Online ID accuracy is (correct creations + correct links)/evaluated turns, excluding missing predictions. Appendix A gives the equivalent count-based formulas.

## 4.3 MSU-Bench: Motivating Explicit Speaker State

We use the speaker retrieval (SR) and verification (SV) tasks in MSU-Bench to examine the value of explicit acoustic identity tools and persistent speaker state. Using annotation-provided temporal boundaries but no gold speaker identities, an Audio LLM invokes an ECAPA-based speaker tool for each segment in chronological order. The tool creates or links local speaker entities, maintains a persistent speaker registry, and provides deterministic query-time readout. This configuration achieves 96.61% SR and 79.35% SV accuracy, exceeding the reported end-to-end Audio LLM results. Detailed results appear in Appendix B.

<table><tr><td>Gap (days)</td><td>EER</td><td>Below-τ</td><td>Imp. accept</td><td>Gen. sim.</td></tr><tr><td>0-1 days</td><td>9.7</td><td>29.8</td><td>2.0</td><td>0.538</td></tr><tr><td>60–100 days</td><td>11.2</td><td>40.7</td><td>1.8</td><td>0.495</td></tr><tr><td>365–450 days</td><td>14.0</td><td>55.6</td><td>1.4</td><td>0.446</td></tr></table>

Table 2: Fixed-template verification across SMIIP-TV time gaps. Error rates are percentages. “Imp. accept” is the proportion of impostor trials exceeding the frozen threshold; “Gen. sim.” is mean genuine cosine similarity.

## 4.4 Time-Varying Speaker Evidence on SMIIP-TV

## 4.4.1 Longitudinal Verification Setup

SMIIP-TV (Qin et al., 2024)<sup>7</sup> contains repeated recordings from 373 speakers over 90 days, with an approximately one-year follow-up for 118 matched speakers. Its text-independent utterances cover variations in recording environment, speaker state, and time of day.

We use each speaker’s earliest utterance as enrollment and evaluate a frozen ECAPA representation across increasing time gaps. Using impostor scores from probes within three days of enrollment, we calibrate a threshold at approximately 2% false acceptance $( \tau _ { 2 \% } = 0 . 4 7 5 1 )$ ) and freeze it for subsequent analyses. We report EER and Below-τ , the proportion of genuine probes falling below this threshold. Complete protocol details are provided in Appendix C.1.

## 4.4.2 Fixed-Template Degradation

Table 2 shows that fixed-template reliability deteriorates with increasing enrollment–probe gaps. Below-τ rises from 29.8% within one day to 55.6% after approximately one year, while EER increases from 9.7% to 14.0%. The trend also persists in an observational subset restricted to bedroom/dormitory recordings, normal speaker state, and normal noise: Below-τ rises from 16.42% to 29.71% between the 0–1- and 60–100-day intervals

(Appendix C.5).

A stronger encoder mitigates but does not eliminate this effect. On matched SMIIP-TV utterances, ReDimNet2-B6 reduces absolute verification errors, yet its Below-τ rate still increases from 5.19% to 12.96% across the same time gaps. These findings indicate that stronger acoustic representations alone do not make a fixed enrollment voiceprint a reliable long-term identity key. Detailed encoder comparisons appear in Appendix C.2.

## 4.4.3 Causal Speaker Maintenance

To examine how acoustic variability affects persistent identity construction, we organize SMIIP-TV recordings into causal replay streams containing two to four speakers. We evaluate six conditions involving temporal, environmental, and speakerstate variation, with five seeds per condition. Each turn is processed chronologically without future speaker information. We compare six training-free representation policies, ranging from fixed and updated single templates to all-exemplar KNN (Appendices C.6 and C.7).

Across the six large-span conditions, online mean updating improves macro-averaged ${ \mathbf B } ^ { 3 }$ F1 from 46.82% (First exemplar) to 55.99%, while Allexemplar KNN achieves 65.31%. However, preserving more historical evidence does not guarantee correct identity assignment: All-exemplar KNN achieves lower LINK precision than First exemplar (52.56% vs. 66.25%). These results show that adaptive representations can mitigate fragmentation, but also expose a trade-off with incorrect linking. We examine the downstream consequences of this trade-off through memory construction and identity-gated QA on LoCoMo-ID (Section 4.5).

## 4.5 Online Speaker-Memory Evaluation on LoCoMo-ID

## 4.5.1 History Routing and Entity Structure

Table 1 compares four maintenance policies in two evidence modes across 30 clean collections and 30 condition-shifted collections. We fix $\tau _ { \mathrm { m e r g e } }$ for routing and private-query lookup, and a stricter threshold $\tau _ { \mathrm { u p d a t e } }$ for gated updates. In this paper, we calibrate τ<sub>mer e</sub> on the VoiceDesign voice bank to adapt the speaker-recognition decision boundary to the score distribution of the target voice domain. Its EER operating point, 0.489, provides a domain-calibrated matching criterion shared by all maintenance policies. For gated updates, we use the threshold $\tau _ { \mathrm { u p d a t e } } = 0 . 5 8$ , supported by the clean-condition ablation in Appendix D.4: performance approaches a plateau at this value, and further increases offer only marginal additional benefits. Parakeet supplies within-session labels and ECAPA performs cross-session association, whereas the global-voiceprint configuration routes individual turns without the session label grouping.

<table><tr><td>Acoustic evidence / policy</td><td>Clean</td><td>Condition-shifted</td></tr><tr><td>ECAPA / Fixed</td><td>72.31</td><td>69.27</td></tr><tr><td>ECAPA / Update</td><td>73.30</td><td>72.55</td></tr><tr><td>ECAPA / Gated update</td><td>73.38</td><td>72.04</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>73.38</td><td>72.79</td></tr><tr><td>Parakeet+ECAPA / Fixed</td><td>73.43</td><td>72.96</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>73.50</td><td>73.16</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>73.52</td><td>73.40</td></tr><tr><td>Parakeet+ECAPA / All-exemplar KNN</td><td>73.33</td><td>73.18</td></tr></table>

Table 3: Matched all-memory QA (%) for eight policies at merge threshold 0.489. Results are micro-pooled judge-correct answers over speaker-sensitive queries without query-speaker filtering or privacy denial.

Table 1 shows that, across both evidence modes and both speech conditions, Update, Gated Update, and All-exemplar KNN reduce FNUR and Clusters/Gold and generally improve ${ \mathbf B } ^ { 3 }$ F1 relative to Fixed. The largest gains occur under ECAPA-only routing with shifted conditions, where ${ \mathbf B } ^ { 3 }$ F1 rises from 74.18% for Fixed to 92.73–96.62% for the non-fixed policies. Overall, maintaining speaker evidence beyond the initial voiceprint improves entity structure. Full causal CREATE/LINK diagnostics appear in Appendix D.

## 4.5.2 All-Memory Retrieval and QA

All-memory QA transcribes each query and retrieves up to 60 items across the collection without identifying the requester or filtering by speaker. Table 3 evaluates the same 4,113 sensitive queries used below, enabling comparison of unrestricted and identity-gated retrieval. ECAPA gated updating reaches 73.38% versus 72.31% for Fixed on clean speech, and 72.04% versus 69.27% under shifted-condition. Because all-memory retrieval does not depend on assigning the query to a speaker-specific entity, differences among speaker-memory policies are smaller here than under identity-gated retrieval, particularly on clean speech. Nevertheless, the gains under shiftedcondition show that updating speaker evidence can still improve the quality and coverage of the retrieved memory, even when access is unrestricted. Full-set results on all 5,958 queries appear in Appendix D.1.

<table><tr><td>Condition</td><td>ECAPA policy</td><td>C/W/D</td><td>QA</td><td>QA|Correct</td></tr><tr><td rowspan="4">Clean</td><td>Fixed</td><td>4084/0/29</td><td>71.41</td><td>71.91</td></tr><tr><td>Update</td><td>3993/115/5</td><td>73.43</td><td>75.63</td></tr><tr><td>Gated update</td><td>4110/0/3</td><td>75.27</td><td>75.33</td></tr><tr><td>KNN</td><td>3993/116/4</td><td>73.64</td><td>75.86</td></tr><tr><td rowspan="4">Condition-shifted</td><td>Fixed</td><td>4079/1/33</td><td>60.15</td><td>60.65</td></tr><tr><td>Update</td><td>3993/115/5</td><td>70.24</td><td>72.35</td></tr><tr><td>Gated update</td><td>4109/0/4</td><td>71.77</td><td>71.84</td></tr><tr><td>KNN</td><td>3993/116/4</td><td>72.40</td><td>74.58</td></tr></table>

Table 4: Top-1 private-access results on 4,113 speakersensitive query audios under clean and shifted condition. C/W/D denotes correct, wrong, and denied authorizations; QA requires correct authorization and a judgecorrect answer (%). KNN denotes All-exemplar KNN.

## 4.5.3 Identity-Gated Retrieval

For the 4,113 first-person, speaker-sensitive query audios per condition, ECAPA performs read-only lookup against the registry constructed from that condition’s history turns. Top-1 authorizes the highest-scoring entity only at similarity $\tau _ { \mathrm { m e r g e } } =$ 0.489 or above; otherwise no personal memory is returned. The authorized identifier is a hard retrieval filter. Queries never create entities, append exemplars, update centers, or write to LightMem. Gold requester identities are used solely for posthoc scoring. A no-match query is an upstream refusal; a downstream answer refusal after authorization is not counted as an authorization denial.

We evaluate Top-1 authorization against the immutable entity-origin binding used for historyrouting scoring. A query is correctly authorized when the selected entity originates from the gold requester, wrongly authorized when it originates from another user, and denied when no entity reaches the threshold. Let C, W, and D be the mutually exclusive counts of these outcomes and $N _ { q } = C + W + D$

Let $A _ { q } \in$ {correct, wrong, denied} denote the authorization outcome for query q, and let $J _ { q } \in$ {0, 1} denote the binary LoCoMo judgment of answer correctness. To couple access correctness with answer utility, we define end-to-end identitygated QA as

$$
\mathrm { Q A } = \frac { 1 } { N _ { q } } \sum _ { q = 1 } ^ { N _ { q } } \mathbf { 1 } [ A _ { q } = \mathrm { c o r r e c t } ] J _ { q } .\tag{4}
$$

Denials, wrong authorizations, and incorrect or unparseable answers therefore receive zero credit. We additionally report

$$
\operatorname { Q A } \mid \operatorname { C o r r e c t } = \frac { 1 } { C } \sum _ { q = 1 } ^ { N _ { q } } \mathbf { 1 } [ A _ { q } = \operatorname { c o r r e c t } ] J _ { q } ,\tag{5}
$$

<table><tr><td>Acoustic evidence / policy</td><td>Online ID</td><td> $F _ { B ^ { 3 } }$ </td><td> $\mathrm { ~ C ~ / ~ W ~ / ~ D ~ }$ </td><td>Targeted QA  $( n = 1 9 )$ </td><td>Full QA (n = 1,371)</td></tr><tr><td>ECAPA / Fixed</td><td>98.84</td><td>96.54</td><td>1342 / 18 / 11</td><td>78.95</td><td>67.83</td></tr><tr><td>ECAPA / Update</td><td>93.35</td><td>95.45</td><td>1335 / 32 / 4</td><td>78.95</td><td>72.06</td></tr><tr><td>ECAPA / Gated update</td><td>96.25</td><td>97.77</td><td>1368 / 0 / 3</td><td>78.95</td><td>73.60</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>94.85</td><td>97.20</td><td>1335 / 33 / 3</td><td>78.95</td><td>71.99</td></tr><tr><td>Parakeet+ECAPA / Fixed</td><td>74.03</td><td>72.86</td><td>1335 / 32 / 4</td><td>68.42</td><td>71.48</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>71.96</td><td>72.57</td><td>1333 / 32 / 6</td><td>73.68</td><td>72.28</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>73.87</td><td>73.10</td><td>1334 / 32 / 5</td><td>68.42</td><td>72.21</td></tr><tr><td>Parakeet+ECAPA / All-exemplar KNN</td><td>71.93</td><td>72.60</td><td>1335 / 32 / 4</td><td>73.68</td><td>72.36</td></tr></table>

Table 5: Four-speaker interleaved stress test: ten clean collections, 11,764 history turns, and 40 gold speakers. Online ID excludes missing predictions. C/W/D counts correct/wrong/denied authorizations on 1,371 queries. Both QA columns require correct authorization and a judge-correct answer; targeted QA uses 19 gated items among 20 diagnostics. Rates are percentages. This differs from the standard 4,113-query condition.

which isolates memory retrieval and answer generation among correctly authorized queries.

Table 4 reports identity-gated QA for ECAPAonly routing, with corresponding diarizationaugmented results in Appendix D. With ECAPAonly routing, QA rises from 71.41% for Fixed to 73.43–75.27% on clean audio and from 60.15% to 70.24–72.40% under shifted condition. The same pattern holds for Parakeet+ECAPA: QA rises from 68.08% to 68.10–68.25% on clean audio and from 63.36% to 67.20–67.76% under shifted condition. These results show that retaining or updating speaker evidence beyond the initial voiceprint improves downstream personalized QA, with the largest gains under shifted condition.

This reveals a trade-off between answer utility and identity integrity. Under shifted conditions, Allexemplar KNN achieves slightly higher identitygated QA than Gated Update (72.40% vs. 71.77%), but incurs 116 wrong authorizations versus none for Gated Update, highlighting the importance of preventing cross-user memory access.

Identity filtering can restrict evidence coverage: Fixed’s shifted condition all-memory QA is 69.27%, versus 60.15% strict QA. Appendix D discusses fragmentation and this utility trade-off.

With merge/lookup threshold fixed at 0.489, a clean update-threshold sweep tests sensitivity of the policy comparison: online accuracy and ${ \mathbf B } ^ { 3 }$ F1 are nearly flat from 0.58 to 0.66, and wrong authorizations remain zero through 0.70, versus 115 at 0.50 and 32 at 0.54. Duration analysis finds lower gated-update accuracy for 1–< 2 s turns: 98.17% clean and 77.38% shifted condition, versus 99.74% and 99.10% for ≥ 2 s. Appendix D.4 provides both diagnostics.

## 4.5.4 Four-Speaker Interleaved Stress Test

Compared with the standard two-speaker setting, the four-speaker interleaved stress test doubles the speakers per conversation and adds a factconflicting parallel story with shared character names, increasing identity and memory interference (Table 5). Descriptively, ECAPA Gated Update attains 97.77% ${ \mathbf B } ^ { 3 }$ F1 and 73.60% full identity-gated QA in the four-speaker setting, compared with 99.54% and 75.27% in the standard clean two-speaker setting. Crucially, updating speaker evidence remains beneficial in the more complex setting: ECAPA Update, Gated Update, and All-exemplar KNN achieve 72.06%, 73.60%, and 71.99% full QA, respectively, versus 67.83% for Fixed; Gated Update also has the highest fourspeaker ${ \tt B } ^ { 3 }$ F1 (97.77%). The same QA advantage over Fixed holds with Parakeet+ECAPA (72.21– 72.36% versus 71.48%). Appendix D.3 provides detailed results.

## 5 Conclusion

We formulate online speaker-memory construction, placing speaker attribution in the memory write path. Longitudinal speech challenges fixed templates; clean speech still exposes fragmentation. Across clean and condition-shifted LoCoMo-ID experiments, Update, Gated Update, and Allexemplar KNN improve identity-gated QA over Fixed in both acoustic evidence modes. This advantage persists in the more challenging four-speaker interleaved test: with ECAPA-only routing, the three non-fixed policies achieve 71.99–73.60% full QA versus 67.83% for Fixed, with Gated Update attaining the highest score. These findings support maintaining speaker evidence beyond the initial voiceprint when constructing and querying personalized memory.

## Limitations

Our evaluation has several limitations. LoCoMo-ID uses synthetic voices; shifted-condition simulates session-dependent rooms, microphones, device chains, and noise rather than naturally recorded, overlapping multi-user conversations. The standard clean and shifted-condition collections each contain two participants. Our fourspeaker stress test interleaves two synthetic, factconflicting stories at session boundaries, preserving within-story chronology; it does not represent naturally recorded, overlapping four-party dialogue. Its full-query evaluation covers only the ten clean collections. The two-to-four-speaker SMIIP-TV streams evaluate entity maintenance over repeated recordings rather than dialogue-memory QA. The Parakeet comparison combines session-local diarization with the same ECAPA cross-session registry, not an independently evaluated published system. We vary only the center-update threshold on clean speech: merge/lookup threshold 0.489 has no independent sensitivity sweep, and the operational encoder and backend remain fixed. Incorrectly written memories are not automatically corrected.

Although multi-exemplar KNN is now evaluated through the LoCoMo-ID write/read pipeline, it stores every accepted voiceprint and does not bound memory growth. The condition-shifted evaluation shows improved clustering but wronguser authorization, so a bounded, contaminationresistant exemplar selection policy remains future work. The post-hoc duration bins (clean 59/17,643 and condition-shifted 13/17,644 subsecond evaluated turns for global ECAPA) contain few short turns, so they cannot establish robust duration effects or the influence of encoder padding; no duration-controlled intervention or duration-stratified private QA is available. Identitygated retrieval is a simulated access protocol, not a formal privacy or authentication guarantee.

## Ethical Considerations

Speaker embeddings and persistent conversational memory can reveal sensitive information. A local anonymous identifier does not make a person’s voice anonymous, and long-term voice linkage could be misused for surveillance or identification beyond the intended interaction context. Raw audio, embeddings, transcripts, and extracted memories should therefore be collected and processed only with appropriate consent and access controls, retained for no longer than necessary, and protected against unauthorized reuse. Synthetic voices and derived conversational data should likewise be used in accordance with the licenses and intended uses of their source assets. The synthetic voice bank is constructed from anonymous generated references and is not intended to imitate or identify real individuals.

Identity-routing errors also create direct privacy risks: a false merge may expose one user’s information to another, while fragmented profiles can make correction and deletion incomplete. The observed wrong authorizations under unrestricted center updating, multi-exemplar scoring, and session-logbased grouping illustrate how plausible matches can expose another user’s memory under the simulated protocol. A deployed system should provide transparent enrollment and identity-management mechanisms, support user inspection, correction, deletion, and withdrawal, and avoid using the proposed router as a security-critical authentication mechanism. Our experiments study the technical problem of identity-aware memory construction and should be understood as a research prototype rather than a deployment-ready biometric system.

## References

Mohammad Al-Ratrout, Pavan Uttej Ravva, Shayla Sharmin, Aditya Raikwar, Ju Young Shin, and Roghayeh Leila Barmaki. 2026. Afa: Identity-aware memory for preventing persona confusion in multiuser dialogue. arXiv preprint arXiv:2604.25022.

Amit Bagga and Breck Baldwin. 1998. Entity-based cross-document coreferencing using the vector space model. In 36th Annual Meeting of the Association for Computational Linguistics and 17th International Conference on Computational Linguistics, Volume 1, pages 79–85, Montreal, Quebec, Canada. Association for Computational Linguistics.

Yuanchen Bei, Tianxin Wei, Xuying Ning, Yanjun Zhao, Zhining Liu, Xiao Lin, Yada Zhu, Hendrik Hamann, Jingrui He, and Hanghang Tong. 2026. Mem-gallery: Benchmarking multimodal long-term conversational memory for MLLM agents. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 40750–40784, San Diego, California, United States. Association for Computational Linguistics.

Gheorghe Comanici et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Rohan Kumar Das, Sarfaraz Jelil, and S. R. Mahadeva Prasanna. 2016. Exploring session variability and

template aging in speaker verification for fixed phrase short utterances. In Interspeech 2016, pages 445– 449.

Brecht Desplanques, Jenthe Thienpondt, and Kris Demuynck. 2020. ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification. In Interspeech 2020, pages 3830–3834.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. LightMem: Lightweight and efficient memory-augmented generation. In International Conference on Learning Representations.

Dongjie Fu, Xize Cheng, Linjun Li, Xiaoda Yang, Lujia Yang, and Tao Jin. 2025. PACHAT: Persona-aware speech assistant for multi-party dialogue. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 29325–29342, Suzhou, China. Association for Computational Linguistics.

Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, and Bryan Catanzaro. 2025. Audio Flamingo 3: Advancing audio intelligence with fully open large audio language models. arXiv preprint arXiv:2507.08128.

Google DeepMind. 2025. Gemini 3 Flash model card. Google DeepMind model card.

Hee-Soo Heo, KiHyun Nam, Bong-Jin Lee, Youngki Kwon, Minjae Lee, You Jin Kim, and Joon Son Chung. 2024. Rethinking session variability: Leveraging session embeddings for session robustness in speaker verification. In IEEE International Conference on Acoustics, Speech and Signal Processing, pages 12321–12325.

Hangrui Hu, Xinfa Zhu, Ting He, Dake Guo, Bin Zhang, Xiong Wang, Zhifang Guo, Ziyue Jiang, Hongkun Hao, Zishan Guo, Xinyu Zhang, Pei Zhang, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. 2026. Qwen3-TTS technical report. arXiv preprint arXiv:2601.15621.

Wenxu Jia, Xize Cheng, Zihan Zhang, Dongjie Fu, Linjun Li, Wenshi Chen, Yangyang Wu, and Tao Jin. 2026. Beyond dyadic memory: Interaction-aware multimodal memory with adaptive agentic retrieval for multi-party spoken conversations. arXiv preprint arXiv:2609.32522.

Heeseung Kim, Che Hyun Lee, Sangkwon Park, Jiheum Yeom, Nohil Park, Sangwon Yu, and Sungroh Yoon. 2025. Does your voice assistant remember? analyzing conversational context recall and utilization in voice interaction models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8984–9014, Vienna, Austria. Association for Computational Linguistics.

KimiTeam et al. 2025. Kimi-Audio technical report. arXiv preprint arXiv:2504.18425.

Tomi Kinnunen and Haizhou Li. 2010. An overview of text-independent speaker recognition: From features to supervectors. Speech Communication, 52(1):12– 40.

Johannes Kirmayr, Lukas Stappen, Phillip Schneider, Florian Matthes, and Elisabeth Andre. 2025. CarMem: Enhancing long-term memory in LLM voice assistants through category-bounding. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 343–357, Abu Dhabi, UAE. Association for Computational Linguistics.

Yejin Kwon, Taewoo Kang, Hyunsoo Yoon, and Changouk Kim. 2025. M3-SLU: Evaluating speakerattributed reasoning in multimodal large language models. arXiv preprint arXiv:2510.19358.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of LLM agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 13851– 13870, Bangkok, Thailand. Association for Computational Linguistics.

Xiaoyi Qin, Na Li, Shufei Duan, and Ming Li. 2024. Investigating long-term and short-term time-varying speaker verification. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 32:3408– 3423.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR.

Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari. 2022. UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022. In Interspeech 2022, pages 4521–4525.

Zhaokai Sun, Shuai Wang, Zhennan Lin, Chengyou Wang, Dehui Gao, Yuang Cao, Chunjiang He, Pan Zhou, and Lei Xie. 2026. MSU-Bench: Towards speaker-centric understanding in conversational multi-speaker scenarios. In Proc. Interspeech.

Linlin Wang, Jun Wang, Lantian Li, and Frank K. Soong. 2016. Improving speaker verification performance against long-term speaker variability. Speech Communication, 79:14–29.

Weiqing Wang, Taejin Park, Ivan Medennikov, Jinhan Wang, Kunal Dhawan, He Huang, Nithin Rao Koluguri, Jagadeesh Balam, and Boris Ginsburg. 2025. Speaker targeting via self-speaker adaptation for multi-talker ASR. In Proceedings of INTER-SPEECH 2025, Rotterdam, The Netherlands.

Boyong Wu et al. 2025a. Step-Audio 2 technical report. arXiv preprint arXiv:2507.16632.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025b. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations.

Yang Xiao, Vidhyasaharan Sethu, Eun-Jung Holden, and Ting Dang. 2026. Voxmem: Benchmarking multimodal memory in large audio language models. arXiv preprint arXiv:2609.32607.

Xiaomi LLM-Core Team et al. 2025. MiMo-Audio: Audio language models are few-shot learners. arXiv preprint arXiv:2512.23808.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025a. Qwen2.5-Omni technical report. arXiv preprint arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025b. Qwen3-Omni technical report. arXiv preprint arXiv:2509.17765.

Xinchao Xu, Zhibin Gou, Wenquan Wu, Zheng-Yu Niu, Hua Wu, Haifeng Wang, and Shihang Wang. 2022. Long time no see! open-domain conversation with long-term persona memory. In Findings of the Associationfor Computational Linguistics: ACL 2022, pages 2639–2650, Dublin, Ireland. Association for Computational Linguistics.

Ivan Yakovlev and Anton Okhotnikov. 2026. Redimnet2: Scaling speaker verification via time-pooled dimension reshaping. In Proceedings ofInterspeech 2026. Accepted to Interspeech 2026.

Yiqun Yao, Naitong Yu, Xiang Li, Xin Jiang, Xuezhi Fang, Wenjia Ma, Xuying Meng, Jing Li, Aixin Sun, and Yequan Wang. 2025. Egomem: Lifelong memory agent for full-duplex omnimodal models. arXiv preprint arXiv:2509.11914.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204–2213, Melbourne, Australia. Association for Computational Linguistics.

## A Detailed Online Routing Metrics

Following the causal first-seen binding protocol in Section 4.2, let $a , \ b , \ c , \ d ,$ and e denote new speaker–CREATE, new speaker–LINK, known speaker–CREATE, known speaker–correct LINK, and known speaker–wrong-target LINK, respectively, with $N = a + b + c + d + e . \mathrm { ~ A ~ }$ speaker is considered new if it has not appeared earlier in the causal stream, and a LINK is correct only when the selected entity matches the immutable first-creation binding.

<table><tr><td>Model</td><td>SR</td><td>SV</td></tr><tr><td>Qwen2.5-Omni (Xu et al., 2025a) AudioFlamingo-3 (Goel et al., 2025) Qwen3-Omni (Xu et al., 2025b)</td><td>19.00 24.00 23.0029.00</td><td>49.0040.00</td></tr><tr><td>Kimi-Audio (KimiTeam et al., 2025) StepAudio2 (Wu et al., 2025a)</td><td>46.0024.00 34.0043.00</td><td></td></tr><tr><td>MiMoAudio (Xiaomi LLM-Core Team et al., 2025) Gemini-2.5-Flash (Comanici et al., 2025)</td><td>38.0048.00 66.0068.00</td><td></td></tr><tr><td>Gemini-2.5-Pro (Comanici et al., 2025) Gemini-3-Flash (Google DeepMind, 2025) Unified speaker state</td><td>67.00 68.00 69.00 75.00</td><td></td></tr></table>

Table 6: MSU-Bench speaker retrieval (SR) and verification (SV) accuracy (%). Unified speaker state uses deterministic tool readout.

We report detailed routing statistics, including CREATE precision $( P _ { \mathrm { c r e a t e } } )$ , CREATE recall $( R _ { \mathrm { c r e a t e } } )$ , LINK precision $( P _ { \mathrm { l i n k } } )$ , and knownuser resolution recall $( R _ { \mathrm { k n o w n } } )$ . FNUR is reported in the main paper as a causal fragmentation indicator.

$$
\begin{array} { l l } { { \displaystyle P _ { \mathrm { c r e a t e } } = \frac { a } { a + c } , } } & { { \displaystyle R _ { \mathrm { c r e a t e } } = \frac { a } { a + b } , } } \\ { { \displaystyle P _ { \mathrm { l i n k } } = \frac { d } { b + d + e } , } } & { { \displaystyle R _ { \mathrm { k n o w n } } = \frac { d } { c + d + e } , } } \\ { { \displaystyle \mathrm { F N U R } = \frac { c } { c + d + e } . } } & { { } } \end{array}\tag{6}
$$

Known-user resolution recall measures correct resolution of previously observed speakers while penalizing erroneous CREATE and wrong-target LINK decisions. FNUR captures the fraction of returning-speaker turns incorrectly assigned to newly created entities. Online ID accuracy is $( a + d ) / N$ ; turns without a prediction are excluded from N.

## B Additional MSU-Bench Results

Table 6 compares this tool-augmented persistent speaker state with reported end-to-end Audio LLM results on MSU-Bench. It achieves 96.61% on SR and 79.35% on SV, exceeding every listed end-toend model on the respective task. The comparison shows that an Audio LLM can attain strong speakermemory performance by orchestrating an explicit lightweight acoustic identity module and persistent state, rather than relying solely on direct answer generation.

## C Additional SMIIP-TV Analyses

## C.1 Static Verification Protocol

SMIIP-TV contains recordings from five daily periods, with three utterances per period and metadata describing recording environments, noise, physical states, and time of day. We use text-independent utterances throughout. The approximately one-year extension includes 118 matched speakers but lacks the full context annotations of the original release.

For static verification, each speaker’s earliest utterance serves as enrollment. We use only probes recorded within three days of enrollment for threshold calibration. The 98th percentile of the corresponding impostor-score distribution defines $\tau _ { 2 \% } = 0 . 4 7 5 1$ , which is frozen across subsequent time-gap analyses and causal replay policies. For bucketed EER estimation, each genuine probe is paired with 20 randomly sampled impostors. Longer-gap genuine probes and downstream labels are excluded from calibration.

Below-τ measures genuine probes rejected by the frozen enrollment template; it is distinct from the false-new user rate (FNUR), which measures erroneous entity creation during causal replay.

## C.2 Matched-Set ReDimNet2-B6 Static Robustness Check

To test whether fixed-template degradation is specific to the ECAPA encoder used in the operational system, we repeat the static verification protocol with ReDimNet2-B6. The comparison is restricted to the intersection of utterances successfully processed by both encoders, containing 205,765 textindependent SMIIP-TV utterances. ECAPA and ReDimNet2-B6 therefore use identical speakers, enrollment utterances, genuine probes, sampled impostor trials, and time buckets.

Each encoder is calibrated separately under the same protocol: the earliest available utterance is enrollment, probes within three days define the calibration window, and the threshold is selected at target FAR=2% and then frozen for all remaining analyses. This gives $\tau _ { \mathrm { E C A P A } } = 0 . 4 7 5 2$ and $\tau _ { \mathrm { R e D i m N e t 2 } } = 0 . 6 1 3 1$ . The threshold values themselves are encoder-specific and are not compared directly; EER and within-encoder changes in below-threshold genuine rate are the relevant cross-encoder diagnostics.

Table 7 and Figures 2–3 compare the matchedset verification results. ReDimNet2-B6 consistently improves absolute verification separability. For example, its EER is 3.34% at 0–1 days and 6.35% at 365–450 days, compared with 9.71% and 14.09% for ECAPA on the same utterance set. Nevertheless, ReDimNet2-B6’s below-threshold genuine rate increases from 5.19% to 12.96% over the same interval. Thus, stronger acoustic representations reduce the severity of longitudinal mismatch but do not make a single enrollment embedding a time-invariant entity key.

## C.3 Matched-Set ReDimNet2-B6 Causal Replay Check

We additionally test whether the online maintenance findings depend on the ECAPA representation used by the operational system. On the same common-utterance set as the static comparison, we construct causal multi-speaker replay streams once and replay the identical session JSONL streams with ECAPA-TDNN and ReDimNet2-B6 embeddings. The six drift dimensions, five random seeds, policy implementations, turn order, and causality constraint are unchanged; only the encoder and its independently calibrated frozen threshold differ.

For fragmentation, let $n _ { s , k }$ count evaluated turns of gold speaker s assigned to predicted entity $k ,$ and let $S$ be the gold speakers in a replay. We compute each speaker’s fraction outside its dominant predicted identity, then macro-average across speakers:

$$
\mathrm { F r a g . } = \frac { 1 0 0 } { \left| S \right| } \sum _ { s \in S } \left( 1 - \frac { \operatorname* { m a x } _ { k } n _ { s , k } } { \sum _ { k } n _ { s , k } } \right) .\tag{7}
$$

This measures within-speaker dispersion, not crossspeaker contamination: assigning everyone to one entity can give zero fragmentation despite poor partition quality. We therefore report it alongside ${ \mathbf B } ^ { 3 }$ F1 and FNUR.

Table 8 isolates the encoder effect for allexemplar KNN: mean ${ \mathbf B } ^ { 3 }$ F1 rises from 65.45% with ECAPA to 73.68% with ReDimNet2-B6, an 8.23-point improvement. ReDimNet2-B6 improves ${ \mathbf B } ^ { 3 }$ F1 in five of six conditions and reduces falsenew decisions and fragmentation in most conditions. Together with the static verification results, this comparison shows that stronger embeddings mitigate acoustic mismatch while fixed-template degradation persists. The relative benefit of alternative maintenance policies under ReDimNet2-B6 remains unmeasured.

<table><tr><td></td><td colspan="2">0-1 days</td><td colspan="2">60–100 days</td><td colspan="2">365-450 days</td></tr><tr><td>Encoder</td><td>EER</td><td>Below-τ</td><td>EER</td><td>Below-τ</td><td>EER</td><td>Below-T</td></tr><tr><td>ECAPA-TDNN</td><td>9.71</td><td>29.82</td><td>11.18</td><td>40.70</td><td>14.09</td><td>55.54</td></tr><tr><td>ReDimNet2-B6</td><td>3.34</td><td>5.19</td><td>4.82</td><td>8.78</td><td>6.35</td><td>12.96</td></tr></table>

Table 7: Matched-set fixed-template verification at representative enrollment–probe gaps. EER and Below-τ are percentages. The comparison uses the 205,765 text-independent utterances successfully processed by both encoders, with identical enrollment utterances, genuine probes, impostor trials, and time buckets. Each encoder uses its own threshold frozen after near-enrollment calibration at target $\mathrm { F A R } { = } 2 \% \ ( \tau _ { \mathrm { E C A P A } } = 0 . 4 7 5 2 ; \tau _ { \mathrm { R e D i m N e t 2 } } = 0 . 6 1 3 1 )$ , so threshold values are not compared directly. ReDimNet2-B6 lowers absolute error, but both encoders exhibit higher EER and a larger below-threshold genuine rate at longer gaps.  
Same utterance population; thresholds calibrated independently at FAR=2%

![](images/ecb3a3c5a2b8a6e4eddc01e82dd00ee8f94e5f4b87265e775a193e2d73167474.jpg)

![](images/22fe03a2290f3849181a5c2b8dc35381a2cbf3ee2ae37d7c33cb0e37a99323fe.jpg)  
Figure 2: Matched-set short-term fixed-template verification for ECAPA and ReDimNet2-B6. Both encoders use identical utterances and time buckets, while each uses an encoder-specific threshold frozen after near-enrollment FAR=2% calibration. ReDimNet2-B6 lowers absolute EER and below-threshold genuine rate, but both measures increase as the enrollment–probe gap grows.  
Same utterance population; thresholds calibrated independently at FAR=2%

![](images/ec945089cc09f37aed8a170f7fb4870e865ae47105251c408da4f6b2e014417f.jpg)

![](images/c0b1ba8fd3bf8a46e6a49cfeac0df11be88e0ddc684aa4eced3e3afd29c48d36.jpg)  
Figure 3: Matched-set long-term fixed-template verification for ECAPA and ReDimNet2-B6, including the approximately one-year follow-up. Although ReDimNet2-B6 substantially improves absolute separability, its EER and below-threshold genuine rate still worsen at longer enrollment–probe gaps.

## C.4 Time-Gap and Score Diagnostics

For the ECAPA operational setting used by the online replay and LoCoMo-ID experiments, Figure 4 expands the three representative rows in Table 2 to the full observed enrollment–probe range. The fixed threshold remains unchanged across buckets.

Figure 5 compares the unrestricted data with the controlled bedroom/dormitory, normal-state, normal-noise subset described in the main text. Figure 6 provides a speaker-level view of Below-τ change, showing that the aggregate trend is not attributable only to a small number of outlying users.

<table><tr><td rowspan="2">Large-span condition</td><td colspan="3">ECAPA-TDNN</td><td colspan="3">ReDimNet2-B6</td></tr><tr><td> $\overline { { F _ { B ^ { 3 } } } }$ </td><td>↑ FNUR↓</td><td>Frag. ↓</td><td> $\overline { { F _ { B ^ { 3 } } } }$ </td><td>↑ FNUR↓</td><td>Frag. ↓</td></tr><tr><td>Intra-day TOD</td><td>63.08</td><td>18.09</td><td>25.52</td><td>62.91</td><td>6.75</td><td>12.33</td></tr><tr><td>Inter-day</td><td>66.58</td><td>1.21</td><td>12.20</td><td>70.56</td><td>0.22</td><td>3.08</td></tr><tr><td>Long-term cross-batch</td><td>67.60</td><td>3.60</td><td>20.79</td><td>81.49</td><td>0.91</td><td>4.51</td></tr><tr><td>Environment</td><td>67.05</td><td>5.00</td><td>23.53</td><td>73.43</td><td>1.16</td><td>7.84</td></tr><tr><td>Speaker state</td><td>58.88</td><td>6.78</td><td>27.63</td><td>66.90</td><td>1.14</td><td>11.45</td></tr><tr><td>Long-term × hist.-env.</td><td>69.54</td><td>4.10</td><td>21.68</td><td>86.78</td><td>1.02</td><td>4.29</td></tr><tr><td>Mean</td><td>65.45</td><td></td><td></td><td>73.68</td><td></td><td></td></tr></table>

Table 8: Matched-session large-span causal replay for the all-exemplar KNN policy. ECAPA-TDNN and ReDimNet2-B6 replay identical common-utterance session streams under encoder-specific thresholds calibrated with the same near-enrollment FAR=2% protocol. Frag. denotes speaker fragmentation (%). ReDimNet2-B6 improves absolute online entity maintenance in most conditions, particularly for long-term and context-variable replay; the near-equal intra-day ${ \mathbf B } ^ { 3 }$ F1 should be interpreted together with its lower FNUR and fragmentation.

![](images/2c7176ddae8d500db28479d83c7a25929f18d9f9c848484e8a2ecf5c374ae7bf.jpg)  
Figure 4: Fixed-template verification over the full observed time range. The same low-drift-calibrated threshold is frozen for all buckets; the final bucket uses the approximately one-year follow-up collection.

![](images/0d3e7fd963d693d8629f28763a4b59de4c1dd2f3443fcc4034929a3e3d74307b.jpg)

![](images/308f921c6551cfef3bb7a6b36a9bfa02e5c855ffffc9de0f4edcbd198981f43a.jpg)  
Figure 5: Below-τ and EER across time gaps in the unrestricted data and in the controlled observational subset. The controlled comparison reduces several measured context differences but does not establish a causal effect of time.

## C.5 Complete Context Breakdowns

Figures 8, 9, and 10 report all available environment, physical-state, and time-of-day categories, including the lower-count categories omitted from Figure 7. Each figure jointly displays thresholdfree EER, mean genuine similarity, and fixedthreshold Below-τ and impostor acceptance rate. Several categories contain few speakers or probes; these plots are descriptive diagnostics and should not be interpreted as controlled causal comparisons.

![](images/8c7c0f701fbab7e8cb310747912ce61508bd3b702708ff17473a0f6a583c38bd.jpg)

![](images/528d94098e7c6e2a2b1d93d33873c3c341dfa6870dbb767836c27cdb44261a8c.jpg)  
change (percentage points)Below-τ change (percentage points)

Figure 6: Speaker-level Below-τ in the nearest and farthest within-release time buckets. Lines connect observations from the same speaker where both buckets are available.  
![](images/3cffb8481862442c23b9f4a378620b8a4b12c0d2303f0dcfd7902e5a5ea802ef.jpg)  
(a) Environment

![](images/107af0eec032474a3229889c8b10ac76874c104a818a448e1c32151ad89cca65.jpg)  
(b) Speaker state

![](images/665e731cbbfce6c10ab0b7ec5829c93cbd2d0c5496796427856e655f8479a130.jpg)  
(c) Time of day

Figure 7: Fixed-template Below-τ and EER across recording environment, speaker state, and time of day. Probes are within 14 days of enrollment, and displayed categories contain at least 300 probes. Complete category breakdowns follow below.
<table><tr><td>Dimension</td><td>Small span (control)</td><td>Large span (higher drift)</td><td>Purpose</td><td></td></tr><tr><td>Intra-day TOD</td><td>Same day and same time-of-day bucket as enrollment</td><td>Same day, at least two time-of-day buck- Isolate ets from enrollment</td><td>change</td><td>within-day</td></tr><tr><td>Inter-day</td><td>0–3 days after enrollment</td><td>30–100 days after enrollment</td><td>Short-term drift</td><td>temporal</td></tr><tr><td>batch</td><td>Long-term cross- Same-release SMIIP-TV probe</td><td>Follow-up probe approximately one Long-term return year later</td><td></td><td></td></tr><tr><td>Environment</td><td>Within 14 days, same environment as enrollment</td><td>Within 14 days, different environment</td><td>Approximate environ- ment contrast</td><td></td></tr><tr><td>Speaker state</td><td>Within 14 days, normal state</td><td>Within 14 days, non-normal state (e.g., Approximate state con- sleepy, eating, hoarse, or exercise)</td><td>trast</td><td></td></tr><tr><td>env.</td><td>Long-term × hist.- Follow-up probes from speakers with low historical environment entropy</td><td>Follow-up probes from speakers with high historical environment entropy</td><td>1 Long-term return by prior variability</td><td></td></tr></table>

Table 9: Small- and large-span definitions for the six SMIIP-TV online replay dimensions. Hist.-env. uses variability in the original SMIIP-TV histories.

## C.6 Online Replay Conditions

Table 9 defines the low-drift control and higherdrift treatment used for each causal replay. “Small”

and “large” refer to the condition span relative to enrollment, not the number of speakers, turns, or sessions. For each speaker and span, the replay begins with enrollment observations and then consumes probes from the corresponding span in chronological order. Small- and large-span replays are constructed independently rather than concatenated into a single stream. Multiple speaker queues are randomly interleaved while preserving per-speaker first-in, first-out (FIFO) order. We do not impose session-level domain consistency: different speakers in the same synthetic session may occupy different stages of their own trajectories.

![](images/55ff04bdbeba90ec8ae0ab3c1382146325ba4539967189d7d0bc28b282058bbb.jpg)

Figure 8: Complete near-enrollment breakdown by recording environment. Sample sizes are shown in the plot; categories with small support should be interpreted cautiously.  
![](images/826120d3187dad87c387fe8a61ae88e6ccfb322b2189c2bf5b00ba527f4fcb31.jpg)

![](images/a4971f316239fea546c006b4638b86dc63ebbf28c3153e5664817681ff85c774.jpg)

![](images/1ba61e72e470fae6d245dcf383a1df3c4da01fd6b0954a9fb8a1a085a0dd111b.jpg)  
Figure 9: Complete near-enrollment breakdown by reported speaker state. Lower Below-τ in a category does not necessarily imply uniformly easier routing because impostor acceptance (shown as FMR in the figure) and sample support can differ.

For each replay, we randomly interleave speaker queues while preserving their chronological order, selecting one to three consecutive turns from a speaker at each step. We evaluate six trainingfree representation policies: First exemplar, Enrollment mean, Latest exemplar, Online mean, exponential moving average (EMA; $\alpha = 0 . 3 )$ , and All-exemplar KNN. The first five retain or update a single speaker template, whereas All-exemplar KNN stores accepted historical embeddings and averages the top-five cosine similarities for identity matching. All policies use the same frozen encoder and assignment threshold. Metrics are averaged over five seeds within each condition and then macro-averaged across the six conditions.

![](images/ab1308bd05806701fb26cf3c5ef296b79fe65c580adb7f3a1af81081ce9c3313.jpg)

Figure 10: Complete near-enrollment breakdown by time of day. All five periods have broad speaker coverage, but the comparison remains observational.
<table><tr><td rowspan=1 colspan=1>54.52</td><td rowspan=1 colspan=1>39.68</td><td rowspan=1 colspan=1>44.24</td><td rowspan=1 colspan=1>50.14</td><td rowspan=1 colspan=1>46.45</td><td rowspan=1 colspan=1>45.90</td></tr><tr><td rowspan=1 colspan=1>47.70</td><td rowspan=1 colspan=1>53.39</td><td rowspan=1 colspan=1>55.89</td><td rowspan=1 colspan=1>55.97</td><td rowspan=1 colspan=1>48.61</td><td rowspan=1 colspan=1>59.93</td></tr><tr><td rowspan=1 colspan=1>56.59</td><td rowspan=1 colspan=1>11.47</td><td rowspan=1 colspan=1>24.29</td><td rowspan=1 colspan=1>36.76</td><td rowspan=1 colspan=1>35.98</td><td rowspan=1 colspan=1>28.10</td></tr><tr><td rowspan=1 colspan=1>44.74</td><td rowspan=1 colspan=1>61.94</td><td rowspan=1 colspan=1>60.92</td><td rowspan=1 colspan=1>55.01</td><td rowspan=1 colspan=1>46.39</td><td rowspan=1 colspan=1>66.92</td></tr><tr><td rowspan=1 colspan=1>47.23</td><td rowspan=1 colspan=1>49.28</td><td rowspan=1 colspan=1>52.76</td><td rowspan=1 colspan=1>55.16</td><td rowspan=1 colspan=1>45.67</td><td rowspan=1 colspan=1>62.10</td></tr><tr><td rowspan=1 colspan=1>62.45</td><td rowspan=1 colspan=1>66.86</td><td rowspan=1 colspan=1>67.08</td><td rowspan=1 colspan=1>67.68</td><td rowspan=1 colspan=1>58.33</td><td rowspan=1 colspan=1>69.48</td></tr></table>

Figure 11: Large-span $\mathsf { B } ^ { 3 } \mathsf { F } 1$ (%), averaged over five seeds. Bold marks the best policy per condition. TOD denotes time of day; Hist.-env. denotes historical environment variability.

<table><tr><td>Policy</td><td> $\overline { { P _ { \mathrm { c r e a t e } } } } ^ { \prime } .$  ←</td><td> $\overline { { R _ { \mathrm { c r e a t e } } \uparrow } }$ </td><td> $\overline { { P _ { \mathrm { l i n k } } \ : \mathcal { T } } }$ </td><td> $\overline { { R _ { \mathrm { k n o w n } } \cdot } }$  ←</td><td> $\overline { { F _ { B ^ { 3 } } \uparrow } }$ </td></tr><tr><td>First exemplar</td><td>21.77</td><td>52.90</td><td>66.25</td><td>62.90</td><td>46.82</td></tr><tr><td>Enrollment mean</td><td>27.23</td><td>42.83</td><td>57.18</td><td>56.51</td><td>53.58</td></tr><tr><td>Latest exemplar</td><td>23.95</td><td>54.14</td><td>34.60</td><td>33.10</td><td>32.20</td></tr><tr><td>Online mean</td><td>31.24</td><td>38.48</td><td>46.54</td><td>46.65</td><td>55.99</td></tr><tr><td>EMA</td><td>31.07</td><td>42.01</td><td>41.36</td><td>41.35</td><td>52.03</td></tr><tr><td>All-exemplar KNN</td><td>31.06</td><td>56.24</td><td>52.56</td><td>51.30</td><td>65.31</td></tr></table>

Table 10: Macro-averaged large-span decision and ${ \mathbf B } ^ { 3 }$ results across six SMIIP-TV conditions (five seeds; %; higher is better).

## C.7 Online Policy Diagnostics

Table 10 summarizes the macro-averaged causal replay results across six conditions. Figure 11 provides the corresponding per-condition ${ \mathbf B } ^ { 3 }$ F1 comparison, while Figure 12 reports detailed CRE-ATE/LINK metrics. These diagnostics illustrate the trade-off between speaker fragmentation and incorrect identity linking. Low FNUR alone is insufficient: a policy may link returning turns frequently while still selecting the wrong target. The decision matrices should therefore be interpreted jointly with the ${ \mathbf B } ^ { 3 }$ results in Figure 11.

## D Additional LoCoMo-ID Routing and Private-Access Analyses

## D.1 Full-Set All-Memory QA

Table 11 reports unrestricted-retrieval answer accuracy on all 5,958 query audios per condition for the eight policies in Table 3. The scores combine 4,113 sensitive-query results from all-memory evaluation with 1,845 non-sensitive-query results, which also used unrestricted retrieval in the corresponding fullset runs.

## D.2 Full Causal Decision and Private-Access Results

Table 12 reports additional history-routing metrics and entity counts. Entities is the total number of predicted entities across 30 independent collectionlocal registries; all rate metrics are percentages.

Table 13 reports the complete Top-1 access outcomes. “Safe auth.” is the correctly authorized fraction, $1 0 0 C / N _ { q }$ , with $N _ { q } ~ = ~ 4 , 1 1 3$ Wrong authorization is counted even if the LLM subsequently refuses or answers incorrectly. QA divides the correct-authorized and judge-correct count by

<table><tr><td colspan="7">(a) CREATE Precision</td></tr><tr><td>Intra-day TOD</td><td>46.4</td><td>50.2</td><td>53.3</td><td>52.7</td><td>56.2</td><td>52.1</td></tr><tr><td>Inter-day gap</td><td>6.6</td><td>10.4</td><td>7.4</td><td>14.0</td><td>12.7</td><td>18.7</td></tr><tr><td>One-year return</td><td>11.8</td><td>16.6</td><td>11.6</td><td>20.3</td><td>18.8</td><td>20.7</td></tr><tr><td>Environment</td><td>24.1</td><td>32.1</td><td>28.3</td><td>37.8</td><td>37.4</td><td>36.6</td></tr><tr><td>Speaker state</td><td>30.5</td><td>38.1</td><td>32.1</td><td>42.1</td><td>43.2</td><td>39.8</td></tr><tr><td>One-year x hist.-env.</td><td>11.2</td><td>16.0</td><td>11.0</td><td>20.6</td><td>18.1</td><td>18.5</td></tr><tr><td></td><td>exemplar</td><td>mean</td><td>Latest exemplar</td><td>Online mean</td><td>EMA</td><td>KNN</td></tr></table>

(b) CREATE Recall
<table><tr><td colspan="7">(b) CREATE Recall</td></tr><tr><td>Intra-day TOD</td><td>46.9</td><td>37.4</td><td>48.6</td><td>32.4</td><td>36.6</td><td>51.6</td></tr><tr><td>Inter-day gap</td><td>47.1</td><td>36.9</td><td>49.1</td><td>32.1</td><td>36.2</td><td>50.5</td></tr><tr><td>One-year return</td><td>58.8</td><td>47.8</td><td>59.2</td><td>43.8</td><td>47.8</td><td>62.3</td></tr><tr><td>Environment</td><td>49.4</td><td>39.9</td><td>52.1</td><td>36.0</td><td>39.3</td><td>53.3</td></tr><tr><td>Speaker state</td><td>54.5</td><td>43.5</td><td>54.2</td><td>38.1</td><td>41.9</td><td>57.6</td></tr><tr><td>One-year x hist.-env.</td><td>60.7</td><td>51.5</td><td>61.7</td><td>48.5</td><td>50.2</td><td>62.0</td></tr><tr><td></td><td>First exemplar</td><td>Enrollment mean</td><td>Latest exemplar</td><td>Online mean</td><td>EMA</td><td>All-exemplar KNN</td></tr></table>

(c) LINK Precision
<table><tr><td rowspan=1 colspan=1>40.6</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>29.1</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>23.7</td><td rowspan=1 colspan=1>39.2</td></tr><tr><td rowspan=1 colspan=1>73.6</td><td rowspan=1 colspan=1>67.6</td><td rowspan=1 colspan=1>21.9</td><td rowspan=1 colspan=1>55.1</td><td rowspan=1 colspan=1>37.4</td><td rowspan=1 colspan=1>49.4</td></tr><tr><td rowspan=1 colspan=1>77.2</td><td rowspan=1 colspan=1>68.8</td><td rowspan=1 colspan=1>41.7</td><td rowspan=1 colspan=1>61.3</td><td rowspan=1 colspan=1>52.6</td><td rowspan=1 colspan=1>61.9</td></tr><tr><td rowspan=1 colspan=1>68.4</td><td rowspan=1 colspan=1>56.8</td><td rowspan=1 colspan=1>33.8</td><td rowspan=1 colspan=1>39.7</td><td rowspan=1 colspan=1>38.5</td><td rowspan=1 colspan=1>52.5</td></tr><tr><td rowspan=1 colspan=1>57.8</td><td rowspan=1 colspan=1>46.6</td><td rowspan=1 colspan=1>29.8</td><td rowspan=1 colspan=1>34.5</td><td rowspan=1 colspan=1>33.7</td><td rowspan=1 colspan=1>44.4</td></tr><tr><td rowspan=1 colspan=1>79.9First</td><td rowspan=1 colspan=1>75.7Enrollment</td><td rowspan=1 colspan=1>51.3Latest</td><td rowspan=1 colspan=1>67.4Online</td><td rowspan=1 colspan=1>62.3</td><td rowspan=1 colspan=1>68.0All--exemplar</td></tr></table>

(d) Known−user Resolution Recall
<table><tr><td>Intra-day TOD</td><td>40.4</td><td>30.2</td><td>30.1</td><td>24.3</td><td>26.8</td><td>39.3</td></tr><tr><td>Inter-day gap</td><td>71.1</td><td>66.6</td><td>21.2</td><td>54.8</td><td>37.0</td><td>48.9</td></tr><tr><td>One-year return</td><td>72.6</td><td>66.9</td><td>39.2</td><td>60.3</td><td>51.4</td><td>60.0</td></tr><tr><td>Environment</td><td>64.5</td><td>56.1</td><td>32.3</td><td>39.8</td><td>38.4</td><td>51.2</td></tr><tr><td>Speaker state</td><td>54.2</td><td>46.1</td><td>28.1</td><td>34.8</td><td>33.8</td><td>42.8</td></tr><tr><td>One-year x hist.-env.</td><td>74.6</td><td>73.2</td><td>47.7</td><td>66.1</td><td>60.7</td><td>65.6</td></tr><tr><td></td><td>First exemplar</td><td>Enrollment mean</td><td>Latest exemplar</td><td>Online mean</td><td>EMA</td><td>All- -exemplar KNN</td></tr></table>

Figure 12: Large-span CREATE/LINK decision metrics across six policies and six conditions (five-seed means; %; higher is better).
<table><tr><td>Acoustic evidence / policy</td><td>Clean</td><td>Condition-shifted</td></tr><tr><td>ECAPA / Fixed</td><td>65.53</td><td>61.09</td></tr><tr><td>ECAPA / Update</td><td>66.63</td><td>65.54</td></tr><tr><td>ECAPA / Gated update</td><td>66.87</td><td>64.99</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>66.65</td><td>66.25</td></tr><tr><td>Parakeet+ECAPA / Fixed</td><td>66.30</td><td>65.36</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>65.93</td><td>65.74</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>66.03</td><td>66.00</td></tr><tr><td>Parakeet+ECAPA / All-exemplar KNN</td><td>65.88</td><td>65.68</td></tr></table>

Table 11: Full-set all-memory QA (%) for eight policies on clean and condition-shifted speech. Each condition comprises 30 collections and 5,958 query audios. Scores are micro-pooled judge-correct rates without speaker-based retrieval filtering.

4,113; conditional QA divides by C. A query rejected before retrieval (D) differs from a downstream refusal on an authorized query.

Identity filtering need not improve answer accuracy alone. Fixed’s augmented all-memory QA is 69.27%, versus 60.15% strict identity-gated QA. Its fragmented histories (410 entities for 60 gold speakers) are consistent with the selected Top-1 partition retaining only part of the requester’s evidence. This is a possible mechanism, not an isolated causal test: the two QA metrics also impose different authorization requirements. Parakeet+ECAPA shows the complementary failure mode. Despite fewer entities, its lower LINK precision and poorer ${ \mathbf B } ^ { 3 }$ F1 are consistent with incorrect associations across speakers, so entity count alone cannot establish correct access. These observations motivate joint reporting of partition structure, authorization, and answer utility.

## D.3 Four-Speaker Interleaved Query Results

Table 14 expands the four-speaker results in the main text. Each policy uses its completed fourspeaker history and memory state. Full identitygated QA divides the number of correctly authorized, judge-correct answers by 1,371, while conditional QA divides that number by C. Targeted strict QA uses the 19 identity-gated questions; targeted raw-judge QA evaluates answer correctness on all 20 questions without requiring correct authorization, including the remaining verbatim question.

<table><tr><td>Condition</td><td>Acoustic evidence / policy</td><td>Turns</td><td>Entities</td><td>P_create</td><td>Rcreate</td><td>Plink</td><td>Rknown</td><td>FNUR</td></tr><tr><td rowspan="8">Clean</td><td>ECAPA / Fixed</td><td>17643</td><td>193</td><td>30.05</td><td>96.67</td><td>99.98</td><td>99.22</td><td>0.77</td></tr><tr><td>ECAPA / Update</td><td>17643</td><td>106</td><td>54.72</td><td>96.67</td><td>97.49</td><td>97.23</td><td>0.27</td></tr><tr><td>ECAPA / Gated update</td><td>17643</td><td>113</td><td>51.33</td><td>96.67</td><td>99.95</td><td>99.65</td><td>0.31</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>17643</td><td>95</td><td>61.05</td><td>96.67</td><td>97.48</td><td>97.29</td><td>0.21</td></tr><tr><td>Parakeet+ECAPA /Fixed</td><td>17644</td><td>69</td><td>79.71</td><td>91.67</td><td>81.41</td><td>81.36</td><td>0.08</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>17644</td><td>67</td><td>82.09</td><td>91.67</td><td>81.40</td><td>81.37</td><td>0.07</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>17644</td><td>67</td><td>82.09</td><td>91.67</td><td>81.40</td><td>81.37</td><td>0.07</td></tr><tr><td>Parakeet+ECAPA / All-exemplar KNN</td><td>17644</td><td>67</td><td>82.09</td><td>91.67</td><td>81.37</td><td>81.34</td><td>0.07</td></tr><tr><td rowspan="7">Condition-shifted</td><td>ECAPA / Fixed</td><td>17644</td><td>410</td><td>14.15</td><td>96.67</td><td>99.63</td><td>97.65</td><td>2.00</td></tr><tr><td>ECAPA / Update</td><td>17644</td><td>216</td><td>26.85</td><td>96.67</td><td>97.30</td><td>96.43</td><td>0.90</td></tr><tr><td>ECAPA / Gated update</td><td>17644</td><td>224</td><td>25.89</td><td>96.67</td><td>99.67</td><td>98.74</td><td>0.94</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>17644</td><td>154</td><td>37.66</td><td>96.67</td><td>97.48</td><td>96.96</td><td>0.55</td></tr><tr><td>Parakeet+ECAPA/Fixed</td><td>17643</td><td>79</td><td>67.09</td><td>88.33</td><td>83.26</td><td>83.17</td><td>0.15</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>17643</td><td>70</td><td>75.71</td><td>88.33</td><td>83.26</td><td>83.22</td><td>0.10</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>17643</td><td>70</td><td>75.71</td><td>88.33</td><td>83.25</td><td>83.20</td><td>0.10</td></tr><tr><td></td><td>Parakeet+ECAPA / All-exemplar KNN</td><td>17643</td><td>69</td><td>76.81</td><td>88.33</td><td>83.11</td><td>83.07</td><td>0.09</td></tr></table>

Table 12: History turn-micro causal routing results for all eight policies under clean speech and condition shifted at merge threshold 0.489. Values are percentages except Turns and Entities.
<table><tr><td>Condition</td><td>Acoustic evidence / policy</td><td>C</td><td>W</td><td>D</td><td>Safe auth.</td><td>QA</td><td>QA|Correct</td></tr><tr><td rowspan="8">Clean</td><td>ECAPA / Fixed</td><td>4084</td><td>0</td><td>29</td><td>99.29</td><td>71.41</td><td>71.91</td></tr><tr><td>ECAPA / Update</td><td>3993</td><td>115</td><td>5</td><td>97.08</td><td>73.43</td><td>75.63</td></tr><tr><td>ECAPA / Gated update</td><td>4110</td><td>0</td><td>3</td><td>99.93</td><td>75.27</td><td>75.33</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>3993</td><td>116</td><td>4</td><td>97.08</td><td>73.64</td><td>75.86</td></tr><tr><td>Parakeet+ECAPA /Fixed</td><td>3833</td><td>273</td><td>7</td><td>93.19</td><td>68.08</td><td>73.05</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>3833</td><td>272</td><td>8</td><td>93.19</td><td>68.25</td><td>73.23</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>3833</td><td>272</td><td>8</td><td>93.19</td><td>68.25</td><td>73.23</td></tr><tr><td>Parakeet+ECAPA / All-exemplar KNN</td><td>3833</td><td>274</td><td>6</td><td>93.19</td><td>68.10</td><td>73.08</td></tr><tr><td rowspan="7">Condition-shifted</td><td>ECAPA / Fixed</td><td>4079</td><td>1</td><td>33</td><td>99.17</td><td>60.15</td><td>60.65</td></tr><tr><td>ECAPA / Update</td><td>3993</td><td>115</td><td>5</td><td>97.08</td><td>70.24</td><td>72.35</td></tr><tr><td>ECAPA / Gated update</td><td>4109</td><td>0</td><td>4</td><td>99.90</td><td>71.77</td><td>71.84</td></tr><tr><td>ECAPA / All-exemplar KNN</td><td>3993</td><td>116</td><td>4</td><td>97.08</td><td>72.40</td><td>74.58</td></tr><tr><td>Parakeet+ECAPA/Fixed</td><td>3560</td><td>318</td><td>235</td><td>86.55</td><td>63.36</td><td>73.20</td></tr><tr><td>Parakeet+ECAPA / Update</td><td>3786</td><td>320</td><td>7</td><td>92.05</td><td>67.76</td><td>73.61</td></tr><tr><td>Parakeet+ECAPA / Gated update</td><td>3739</td><td>320</td><td>54</td><td>90.91</td><td>67.20</td><td>73.92</td></tr><tr><td></td><td>Parakeet+ECAPA / All-exemplar KNN</td><td>3787</td><td>321</td><td>5</td><td>92.07</td><td>67.57</td><td>73.38</td></tr></table>

Table 13: Speaker-sensitive Top-1 authorization and end-to-end answer utility over 4,113 questions per condition (clean and condition-shifted). C/W/D are counts, other values are percentages.

## D.4 Update-Threshold and Utterance-Duration Diagnostics

We vary the center-update threshold on clean ECAPA speech while holding the merge and query lookup threshold at 0.489 (Table 15). We select 0.58 as the lowest tested threshold in the 0.58–0.66 range where online accuracy and ${ \mathbf B } ^ { 3 }$ F1 are nearly stable and no wrong authorizations occur; lower values (0.50 and 0.54) produce 115 and 32 wrong authorizations, respectively.

For a post-hoc utterance-length analysis, we join the recorded duration\_seconds and chronological per-turn assignments for the two global-ECAPA policies with private gold labels only after replay. Within each of the 30 collections, a newly created local ID is immutably bound to the gold identity of its first evaluated turn, and a first-seen gold speaker scores as correct only if a new local ID is created (Section 4.2). Each row of Table 16 counts actual waveform duration, not lexical length, and reports micro-averaged first-seen-binding accuracy; rates are not a controlled intervention on duration. Shifted condition changes waveform durations and thus the bucket populations relative to clean speech.

All 60 first-appearance gold turns in each condition fall in the ≥ 2 s bin. In particular, 11 of 13 condition-shifted < 1 s gated turns are erroneous duplicate CREATES; these estimates are high variance and cannot establish a general short-speech failure rate.

## E Backend Models and Decoding Configuration

All LoCoMo-ID experiments use local Qwen3-Omni-30B-A3B-Instruct for history/query transcription, LightMem factual extraction, answer generation, and LLM-as-ajudge evaluation. The speech-generation branch is disabled. LightMem uses bge-small-en-v1.5 (384 dimensions) for both indexing memory entries and embedding text queries; Qdrant provides the vector store. Before fact extraction, we use the multilingual MeetingBank checkpoint from LLMLingua-2 for pre-compression (rate 0.6) and topic segmentation. The factual-extraction, answer, and judge prompts are otherwise unchanged from LightMem and LoCoMo.

<table><tr><td colspan="3">Online</td><td></td><td>Targeted</td><td>Targeted</td><td>Full strict</td><td>QA</td></tr><tr><td>Policy</td><td>ID</td><td> $F _ { B ^ { 3 } }$ </td><td>C/W/D</td><td>strict (19)</td><td>raw (20)</td><td>(1,371)</td><td>Correct</td></tr><tr><td colspan="8">ECAPA</td></tr><tr><td>Fixed</td><td>98.84</td><td>96.54</td><td>1342/18/11</td><td>15/19 (78.95)</td><td>15/20 (75.00)</td><td>930/1371 (67.83)</td><td>69.30</td></tr><tr><td>Update</td><td>93.35</td><td>95.45</td><td>1335/32/4</td><td>15/19 (78.95)</td><td>16/20 (80.00)</td><td>988/1371 (72.06)</td><td>74.01</td></tr><tr><td>Gated update</td><td>96.25</td><td>97.77</td><td>1368/0/3</td><td>15/19 (78.95)</td><td>15/20 (75.00)</td><td>1009/1371 (73.60)</td><td>73.76</td></tr><tr><td>KNN</td><td>94.85</td><td>97.20</td><td>1335/33/3</td><td>15/19 (78.95)</td><td>16/20 (80.00)</td><td>987/1371 (71.99)</td><td>73.93</td></tr><tr><td colspan="8">Parakeet+ECAPA</td></tr><tr><td>Fixed</td><td>74.03</td><td>72.86</td><td>1335/32/4</td><td>13/19 (68.42)</td><td>14/20 (70.00)</td><td>980/1371 (71.48)</td><td>73.41</td></tr><tr><td>Update</td><td>71.96</td><td>72.57</td><td>1333/32/6</td><td>14/19 (73.68)</td><td>15/20 (75.00)</td><td>991/1371 (72.28)</td><td>74.34</td></tr><tr><td>Gated update</td><td>73.87</td><td>73.10</td><td>1334/32/5</td><td>13/19 (68.42)</td><td>14/20 (70.00)</td><td>990/1371 (72.21)</td><td>74.21</td></tr><tr><td>KNN</td><td>71.93</td><td>72.60</td><td>1335/32/4</td><td>14/19 (73.68)</td><td>15/20 (75.00)</td><td>992/1371 (72.36)</td><td>74.31</td></tr></table>

Table 14: Complete four-speaker results on ten clean collections. Targeted strict QA (19) and full strict QA (1,371) require correct authorization and a judge-correct answer; targeted raw judge (20) requires only a judge-correct answer and includes the non-gated item. C/W/D are full-set authorization counts. QA|Correct conditions full strict QA on C. All rates are percentages; KNN denotes All-exemplar KNN.
<table><tr><td>Tupdate</td><td>Entities</td><td>Online (%)</td><td>B3 F1 (%)</td><td>Query assigned / 5958</td><td>W/4113</td><td>QA (%)</td></tr><tr><td>0.50</td><td>106</td><td>97.23</td><td>98.41</td><td>5953</td><td>115</td><td>73.33</td></tr><tr><td>0.54</td><td>110</td><td>98.65</td><td>99.11</td><td>5954</td><td>32</td><td>74.81</td></tr><tr><td>0.58</td><td>113</td><td>99.64</td><td>99.54</td><td>5955</td><td>0</td><td>75.27</td></tr><tr><td>0.62</td><td>114</td><td>99.64</td><td>99.54</td><td>5955</td><td>0</td><td>75.27</td></tr><tr><td>0.66</td><td>114</td><td>99.64</td><td>99.55</td><td>5955</td><td>0</td><td>75.47</td></tr><tr><td>0.70</td><td>118</td><td>99.61</td><td>99.45</td><td>5955</td><td>0</td><td>75.44</td></tr></table>

Table 15: Clean ECAPA gated-update sweep, fixed merge threshold 0.489. Routing/cluster scores and query assigned are from the upstream replay; W and QA are from the separate complete downstream threshold-sweep runs, not the default-run main table.
<table><tr><td>Condition</td><td>ECAPA policy</td><td>&lt; 1 s: correct / n</td><td>1−&lt; 2 s: correct / n</td><td>≥ 2 s: correct / n</td></tr><tr><td>Clean</td><td>Gated update</td><td>48 / 59 (81.36%)</td><td>375 / 382 (98.17%)</td><td>17157 / 17202 (99.74%)</td></tr><tr><td rowspan="3">Condition-shifted</td><td>All-exemplar KNN</td><td>50 / 59 (84.75%)</td><td>372 / 382 (97.38%)</td><td>16742 / 17202 (97.33%)</td></tr><tr><td>Gated update</td><td>2 / 13 (15.38%)</td><td>195 / 252 (77.38%)</td><td>17223 / 17379 (99.10%)</td></tr><tr><td>All-exemplar KNN</td><td>3 / 13 (23.08%)</td><td>210 / 252 (83.33%)</td><td>16894 / 17379 (97.21%)</td></tr></table>

Table 16: Post-hoc duration-bucket first-seen-binding accuracy on evaluated history turns. Each policy/condition denominator sums to 17,643 (clean) or 17,644 (condition-shifted); the short bins are very small in the augmented condition.

Table 17 lists the model and decoding settings for these components.

Timing scope. The reported 65.6 ms per turn measures ECAPA embedding extraction only. It excludes routing, transcription, and textual-memory construction, and does not establish full-pipeline real-time performance.

## F LoCoMo Acoustic Tool-Use Prompt

We retain LightMem’s original factual-extraction prompt, the original LoCoMo answer prompt, and the original LoCoMo judge prompt unchanged, and therefore do not reproduce them here. The spoken adaptation adds the audio-transcription prompt below and a deterministic acoustic tool-use prompt for each chronologically arriving history segment. For routing, the model’s role is only to invoke the tool; it does not select a speaker identity. The extract\_segment\_embedding tool extracts an ECAPA-TDNN embedding, scores it against the persistent registry, applies the merge threshold, and creates or links a local speaker entity. For fixed, update, and gated-update policies, only the deterministic center-maintenance rule differs; the all-exemplar policy instead maintains a history of accepted embeddings and uses mean top-five similarity. Session-log-assisted policies first group turns with local Parakeet speaker labels and then use ECAPA acoustic evidence for cross-session association. No generated transcript is used to decide identity.

<table><tr><td>Stage</td><td>Model</td><td>Decoding configuration</td></tr><tr><td>History and query transcription</td><td>Qwen3-Omni-30B-A3B-Instruct</td><td>Sampling: temperature 0.7, top-p 0.8, top-k 20; maximum 512 new tokens.</td></tr><tr><td>LightMem factual extraction</td><td>Qwen3-Omni-30B-A3B-Instruct</td><td>Sampling: temperature 0.1, top-p 0.1; max- imum 16,000 new tokens. The configured top-k = 1 is not forwarded to the decoder.</td></tr><tr><td>Answer generation</td><td>Qwen3-Omni-30B-A3B-Instruct</td><td>Greedy decoding (do_sample=False); max- imum 300 new tokens.</td></tr><tr><td>LLM-as-a-judge</td><td>Qwen3-Omni-30B-A3B-Instruct</td><td>Greedy decoding (do_sample=False); max- imum 200 new tokens.</td></tr><tr><td>Text embedding</td><td>bge-small-en-v1.5</td><td>384-dimensional embedding; no generative decoding.</td></tr></table>

Table 17: Models and generation settings for the reported LoCoMo-ID experiments.

## F.1 Audio Transcription Prompt

For every history and query clip, Qwen3-Omni receives the audio together with the following system prompt and fixed user message. The resulting transcript is passed to the unchanged Light-Mem/LoCoMo components.

## System prompt

You are a careful speech transcription assistant. You will listen to one audio clip containing a question someone is asking. Transcribe EXACTLY what they say, as faithfully and completely as possible (keep every concrete name, place, date, number mentioned). Output ONLY the transcript text, nothing else — no preamble, no quotation marks, no commentary.

## User message

[audio clip] Transcribe this audio clip now.

## F.2 Deterministic Acoustic Tool-Use Prompt

You are a careful speech analysis assistant building a persistent speaker memory for a conversation one audio segment at a time.

For the CURRENT audio segment, you MUST call “extract\_segment\_embedding” exactly once with the supplied audio path, start\_time, and end\_time. This acoustic tool extracts the segment’s voiceprint, compares it with the persistent speaker registry, and automatically determines whether to link the segment to an existing local speaker or create a new one according to its configured similarity threshold and update policy.

Do NOT infer, choose, name, override, or guess the speaker identity yourself. Do

NOT call any other speaker-routing tool. After the tool returns its result, reply only with a short confirmation that the current segment has been processed. Do not output a multiple-choice answer or any additional analysis.

For a segment with duration d, the accompanying user message is:

Audio file: <audio\_path>   
Current segment: start\_time=0,   
end\_time=<d>.   
Process this segment with   
“extract\_segment\_embedding” now.

## G LoCoMo-ID Construction Details

LoCoMo histories and reference answers are retained without LLM generation; turn selection and query rewriting use deterministic rules. The construction pipeline’s only natural-language model instruction is the Qwen3-TTS VoiceDesign prompt for the anonymous voice bank.

Synthesis QC and augmentation. Each clip is initially synthesized and checked for UTMOS ≥ 3.0 and reference-voice cosine similarity ≥ 0.489. A failing clip receives at most four additional attempts. If none passes both checks, the highest-UTMOS candidate among the initial synthesis and retries is retained; no dialogue turn is discarded by this fallback. The procedure targets perceptual quality and intended speaker attribution, not downstream QA. Generated-to-reference similarity is not the online score against a first-turn or evolving entity center, so passing QC does not guarantee a correct routing decision. Nevertheless, using ECAPA for both QC and routing can affect cleancondition difficulty. Condition shifts are applied after clean construction without further UTMOS or reference-similarity filtering; its difficult acoustic observations are therefore not removed by this QC stage. Pre-regeneration score records were not retained, so the current experiments do not establish the before/after score distribution.

## G.1 VoiceDesign Instruction and Enrollment

The VoiceDesign instruction follows the template:

A {pitch}-pitched voice speaking at a   
{speaking\_rate} pace, with a {timbre}   
timbre and a {expression} expression,   
{accent} accent. {delivery\_style}

Candidate profiles combine pitch, speaking rate, timbre, expression, English accent, and delivery style. An instantiated instruction is:

A medium-high-pitched voice speaking at   
a slightly slow pace, with a bright   
and clear timbre and an animated and   
expressive expression, Scottish English   
accent. A composed, understated style   
with minimal pitch variation.

ECAPA-TDNN diversity filtering retains 50 of 120 candidate voices; descriptive attributes do not necessarily yield distinct acoustics.

Each candidate voice reads the same canonical enrollment sentence:

Hi there, this is my voice. I’m looking   
forward to our conversation today.

Identical enrollment content controls lexical variation in ECAPA-TDNN comparisons.

## G.2 Voice-Cloning Inputs

VoiceDesign generates each identity’s enrollment audio. Qwen3-TTS Base then synthesizes history turns and queries in voice-cloning mode with these inputs:

Reference audio:   
[voice-id]\_enrollment.wav   
Reference transcript: Hi there, this is   
my voice. I’m looking forward to our   
conversation today.   
Target text: [one LoCoMo history turn   
or one rewritten query]

Reusing enrollment audio and text is intended to preserve identity across sessions.

## G.3 Deterministic Query Transformations

Question rewriting is rule based rather than prompt driven. For identity\_gated questions, a named participant is converted to the first person, for example, “What did Caroline research?” becomes “What did I research?”, and “What is Caroline’s identity?” becomes “What is my identity?” For symmetric we\_shared questions, the two participants are converted to the first-person plural: “What do Jon and Gina both have in common?” becomes “What do we have in common?”, while “What did

Jon and Gina compare their entrepreneurial journeys to?” becomes “What did we compare our entrepreneurial journeys to?” Questions that cannot be transformed safely, together with adversarial questions, remain verbatim; for example, “What nickname does Nate use for Joanna?” is unchanged. Unsafe transformations fall back to their original wording.

## G.4 Session-Level Recording-Condition Augmentation

For each conversation instance and session, we deterministically assign one acoustic profile that specifies the room geometry, source–microphone distance and azimuth, reverberation time, microphone directivity, device response, automatic gain control, and optional environmental noise at a fixed signal-to-noise ratio. All utterances within the same session share this profile, whereas profiles can change across sessions, reflecting realistic transitions across different recording scenarios.

Each waveform is processed by room impulse response convolution, device-bandwidth filtering, gain control, noise mixing, and peak normalization. This augmentation preserves the original text, speaker–voice binding, turn order, and query audio, thereby isolating robustness to heterogeneous historical recording conditions from semantic or identity changes.