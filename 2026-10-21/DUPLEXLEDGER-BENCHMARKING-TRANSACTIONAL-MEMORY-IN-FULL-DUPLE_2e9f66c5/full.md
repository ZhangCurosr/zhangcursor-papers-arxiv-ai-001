# DUPLEXLEDGER: BENCHMARKING TRANSACTIONAL MEMORY IN FULL-DUPLEX SPEECH AGENTS

Yuxuan Li<sup>1,∗</sup>, Xiangchen Guan<sup>2,∗</sup>, Kunrong Li<sup>3</sup>, Jin Wang<sup>4</sup>, Kaiwen Luo<sup>1</sup>, Yang Xiao<sup>5</sup>, Yufei Guo<sup>6</sup>, Zhisheng Chen<sup>1</sup>, Liang Lin<sup>1,†</sup>, Kun Wang<sup>1,†</sup>

<sup>1</sup>Nanyang Technological University <sup>2</sup>Peking University <sup>3</sup>Beijing University of Technology <sup>4</sup>Xidian University <sup>5</sup>University of Melbourne <sup>6</sup>China Aerospace Science and Industry Corporation

<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## ABSTRACT

Full-duplex speech agents must update persistent memory while listening and speaking, yet it remains unclear which events should be stored, revised, or excluded. Existing benchmarks isolate interaction timing from memory outcomes, obscuring the state transitions behind a correct answer. We introduce DuplexLedger, a bilingual benchmark of 1,094 persistent sessions aligning four fullduplex scenarios with event-level WRITE/UPDATE/IGNORE/RETAIN supervision, explicit pre/post memory states, and 6,564 same-session questions and checks. Evaluating five full-duplex models reveals an interaction–memory gap: no system combines strong boundary control with coherent final-state memory, and the best final-state fact coverage is 2.01%. We propose ACT-Mem, an 8.92M-parameter sidecar comprising an event controller, a versioned transactional ledger, and a typed retriever coupled with a frozen MiniCPM-o 4.5 backbone. On 219 held-out sessions, ACT-Mem increases latest-value accuracy from 4.11% to 8.68% while reducing stale assertions and distractor adoption. It improves interaction boundaries but slows interruption stopping. These results motivate selective, auditable full-duplex memory rather than indiscriminate transcript retention. Code is available at https://github.com/lyx-researcher/DuplexLedger.

## 1 INTRODUCTION

Speech is a natural interface for agents that assist people continuously. End-to-end spoken language models now support open-ended voice interaction (Zhang et al., 2023; Tang et al., 2024; Chu et al., 2024; Xie & Wu, 2024), while full-duplex systems listen while speaking, enabling interruptions, backchannels, and overlapping turns (Defossez et al.´ , 2024; Wang et al., 2025c; Roy et al., 2026; Liu et al., 2026). Yet concurrency creates a state-management problem: an utterance may introduce or revise a fact, remain incomplete, serve only as a backchannel, or be addressed to someone else. A dependable agent must decide what was said and whether it should modify persistent memory.

![](images/8aef2b3ec058653781c4df63b02f026699822202e41df06e56b8fe7b5c2e4a2e.jpg)  
Figure 1: From isolated interaction and memory to transactional full-duplex memory.

Existing evaluation treats interaction and memory largely as separate capabilities. Full-duplex benchmarks emphasize turn taking, pause handling, overlap, and interruption, with recent extensions adding corrections and entity tracking (Lin et al., 2025; 2026b; Wang et al., 2026; Lin et al., 2026c; Modi et al., 2026). Memory methods and benchmarks instead study long-horizon storage, retrieval, and updating from serialized histories (Packer et al., 2023; Maharana et al., 2024; Wu et al., 2025; Xu et al., 2025), while recent speech agents begin to connect persistent memory with streaming interaction (Yao et al., 2025; Xie et al., 2026). The compared resources still do not jointly expose event admission, supersession, distractor exclusion, and memory-grounded spoken answers under adaptive concurrency. Without aligned state transitions, a correct answer cannot show whether the system maintained coherent memory or merely matched its current context.

We formulate this capability as transactional memoryforfull-duplex agents: each event is admitted, rejected, or used to revise a versioned state. We introduce DuplexLedger, a bilingual benchmark of 1,094 persistent sessions aligning pause handling, speech to others, backchannels, and interruptions with WRITE, UPDATE, IGNORE, and RETAIN operations, explicit pre/post states, and same-session spoken questions. We further propose ACT-Mem, coupling an event controller, a versioned transactional ledger, and a typed retriever with a frozen full-duplex backbone. The controller determines admission and operation; the ledger records committed values and supersession through PREPARE, COMMIT, and ABORT; and the retriever supplies active, type-appropriate evidence only for memory QA. Streaming generation remains frozen, making memory transitions explicit and auditable.

Contributions. (1) We formulate full-duplex memory as selective event-level state construction. (2) DuplexLedger aligns adaptive interaction, reference state transitions, and spoken QA across four scenarios, 1,094 bilingual sessions, and 6,564 questions/checks. (3) ACT-Mem combines admission control, versioned memory, and typed retrieval with a frozen backbone, increasing latest-value accuracy from 4.11% to 8.68% while reducing stale assertions and distractor adoption.

## 2 RELATED WORK

Full-Duplex Dialogue. Early spoken language models established end-to-end speech understanding and generation (Zhang et al., 2023; Tang et al., 2024; Chu et al., 2024; Xie & Wu, 2024). Full-duplex systems subsequently modeled synchronized channels, paired next-token prediction, or parallel audio streams to support concurrent listening and speaking (Nguyen et al., 2023; Veluri et al., 2024; Wang et al., 2025b; Defossez et al.´ , 2024). Freeze-Omni retains a frozen language backbone, while MiniCPM-o 4.5, PersonaPlex, and Lychee-FD further target low-latency interaction, role control, and acoustic–semantic coordination (Wang et al., 2025c; Cui et al., 2026; Roy et al., 2026; Liu et al., 2026). These systems improve conversational timing, but generally provide no explicit policy for deciding which overlapping events should enter or revise persistent memory.

Memory-Augmented Agents. Memory-augmented agents manage conversational history through hierarchical storage, temporal updates, structured retrieval, or latent memory (Packer et al., 2023; Zhong et al., 2024; Sarthi et al., 2024; Jimenez Guti´ errez et al.´ , 2024; Xu et al., 2025; Zhang et al., 2026). EgoMem and VoiceMem connect persistent memory to streaming multimodal interaction (Yao et al., 2025; Xie et al., 2026). These methods improve retention and retrieval but process session-level evidence after observation; they do not jointly control event admission, causal supersession, or exclusion of non-addressed and incomplete speech during concurrent interaction.

Full-Duplex and Audio Benchmarks. Full-Duplex-Bench evaluates pauses, backchannels, turn transitions, and interruptions; its overlap-focused extension adds speech to others and background speech (Lin et al., 2025; 2026b). HumDial-FDBench broadens human-recorded coverage, while adaptive benchmarks test correction handling, entity tracking, tool use, and interruption-conditioned state updates (Wang et al., 2026; Lin et al., 2026c;a; Modi et al., 2026). General audio benchmarks assess speech, scene, and paralinguistic understanding (Wang et al., 2025a); Audio Multi-Challenge also evaluates multi-turn inference memory, instruction retention, coherence, and spoken self-correction (Gosai et al., 2026). These protocols broaden audio-interaction evaluation but do not expose an event-level persistent-state ledger under adaptive full-duplex execution.

## 3 DUPLEXLEDGER: BENCHMARK DESIGN

Full-duplex memory evaluation requires determining event content and whether it should update persistent state. DuplexLedger combines adaptive interaction challenges with event-level memory supervision, enabling joint evaluation of interaction behavior and memory fidelity.

Table 1: DuplexLedger versus representative full-duplex and conversational-memory benchmarks.
<table><tr><td></td><td></td><td></td><td colspan="2">Duplex</td><td colspan="4">Memory</td></tr><tr><td>Dataset</td><td>Scale</td><td>Hrs</td><td>Adaptive interaction</td><td>FD scenarios</td><td>State / Memory eval.</td><td>Update QA</td><td>Distractor reject</td><td>Event state</td></tr><tr><td>FDB v1/v1.5</td><td>727/499</td><td>NR</td><td>x</td><td>4/4</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>HumDial-FDBench</td><td>15,698</td><td>&gt; 100</td><td>x</td><td>9</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>MultiTalkBench</td><td>104</td><td>56.5</td><td>x</td><td>N/A</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>FDB v2</td><td>200</td><td>N/A</td><td>√</td><td>4</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>EchoChain</td><td>200</td><td>NR</td><td>√</td><td>1</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>LoCoMo</td><td>10</td><td>N/A</td><td>x</td><td>N/A</td><td>√</td><td>x</td><td>x</td><td>x</td></tr><tr><td>LongMemEval</td><td>500</td><td>N/A</td><td>x</td><td>N/A</td><td>√</td><td>√</td><td>x</td><td>x</td></tr><tr><td>DuplexLedger (ours)</td><td>1,094</td><td>≥35.3</td><td></td><td>4</td><td></td><td>V</td><td></td><td>J</td></tr></table>

Scale denotes stimuli for FDB v1/v1.5, instances for HumDial-FDBench, samples for MultiTalkBench, tasks for FDB v2, interrupted conversations for EchoChain, conversations for LoCoMo, questions for LongMemEval, and sessions for DuplexLedger. Update QA denotes explicit evaluation of factual revision or knowledge updating; temporal reasoning alone is not counted. NR: aggregate audio duration is not reported; N/A: no fixed audio corpus or no comparable full-duplex scenario taxonomy.

DuplexLedger: Adaptive Full-Duplex Memory Evaluation  
![](images/94ea2df8eae012bd1ed545853514d7a9d2316a0dc71013008332a4a7be2ee469.jpg)  
Figure 2: Overview of DuplexLedger. A persistent session aligns adaptively triggered full-duplex challenges, event-level memory transitions, and spoken memory questions.

## 3.1 TASK FORMULATION

A session is an ordered event stream $\mathcal { E } = ( e _ { 1 } , \dots , e _ { T } )$ , where $\boldsymbol { e } _ { t } = ( x _ { t } , s _ { t } , a _ { t } , \tau _ { t } )$ specifies speech content, speaker, addressee, and timing. The memory state $M _ { t }$ contains items $( k , v , \sigma )$ with entity key k, value v, and status σ ∈ {active, superseded}. Because one event may affect multiple entities, transitions are defined at the item level:

$$
M _ { t } = { \mathcal { T } } ( M _ { t - 1 } , \{ ( k _ { i } , v _ { i } , o _ { i } ) \} _ { i = 1 } ^ { n _ { t } } ) , \qquad o _ { i } \in \{ { \mathrm { W R I T E } } , { \mathrm { U P D A T E } } , { \mathrm { I G N O R E } } , { \mathrm { R E T A I N } } \} .\tag{1}
$$

WRITE introduces a value, UPDATE supersedes an active value, IGNORE rejects inadmissible evidence, and RETAIN preserves the current state. The evaluated model receives only streaming audio and spoken questions; ledger operations and states are reserved for supervision and scoring. The task therefore evaluates selective state construction rather than unrestricted recall.

## 3.2 ADAPTIVE FULL-DUPLEX SESSIONS

Each persistent session contains contextual events, two to five challenges, and six evaluation targets, without state resets. Fixed event content follows a model-dependent schedule: interruptions and backchannels require VAD-detected decoded assistant speech, with unmet triggers remaining unrealized. Pause and speech-to-others events use predefined observation windows.

The four scenarios require distinct behavior. An interruption should stop the current response and admit the revised intent; a backchannel should neither terminate the response nor enter memory. A within-turn pause should not trigger premature generation or commitment, whereas speech addressed to another person should produce neither an assistant response nor a memory update.

Table 2: Composition and supervision statistics of DuplexLedger.  
(a) Session composition
<table><tr><td>Scenario</td><td>ZH</td><td>EN</td><td>Sess.</td><td>Events</td><td>Chal.</td><td>QA</td></tr><tr><td>Pause handling</td><td>149</td><td>127</td><td>276</td><td>1,932</td><td>552</td><td>1,656</td></tr><tr><td>Speech to others</td><td>150</td><td>126</td><td>276</td><td>2,208</td><td>1,104</td><td>1,656</td></tr><tr><td>User backchannel</td><td>146</td><td>137</td><td>283</td><td>2,545</td><td>1,130</td><td>1,698</td></tr><tr><td>User interruption</td><td>132</td><td>127</td><td>259</td><td>1,554</td><td>518</td><td>1,554</td></tr><tr><td>Total</td><td>577</td><td>517</td><td>1,094</td><td>8,239</td><td>3,304</td><td>6,564</td></tr></table>

(b) Ledger and evaluation supervision
<table><tr><td>Operation</td><td>Count</td><td>%</td><td>Target</td><td>Count</td><td>/sess.</td></tr><tr><td>WRITE</td><td>3,982</td><td>27.1</td><td>Latest value</td><td>2,188</td><td>2</td></tr><tr><td>UPDATE</td><td>1,212</td><td>8.3</td><td>Factual recall</td><td>1,094</td><td>1</td></tr><tr><td>RETAIN</td><td>1,941</td><td>13.2</td><td>Distractor reject.</td><td>1,094</td><td>1</td></tr><tr><td>IGNORE</td><td>7,546</td><td>51.4</td><td>Final summary</td><td>1,094</td><td>1</td></tr><tr><td>Total</td><td>14,681</td><td>100</td><td>Behavior check</td><td>1,094</td><td>1</td></tr></table>

Notes. Events are executable user or bystander inputs; ledger counts additionally include 6,442 aligned assistant-shadow entries. Fixed-input and event-source audio total 35.33 and 24.22 h, respectively. Release dev/test splits contain 237/857 sessions; method train/validation/calibration/held-out splits contain 660/175/40/219 sessions.

## 3.3 EVENT-LEVEL SUPERVISION AND EVIDENCE-GROUNDED QA

Each ledger entry records the event content and roles, target operation, memory-state transition, and supersession links. Each question is aligned with supporting memory identifiers, forbidden superseded or distractor identifiers, and a reference answer. Two latest-value questions per session test whether revisions replace earlier values; the remaining targets cover factual recall, distractor rejection, final-state summarization, and scenario behavior.

Combined with system traces, this alignment supports diagnosis of admission, supersession, exclusion, retrieval, and generation failures. Table 2 summarizes dataset composition and supervision.

## 3.4 CONSTRUCTION AND EVALUATION PROTOCOL

DuplexLedger comprises controlled bilingual sessions generated specifically for this benchmark. We preserve source waveforms, derive event playback windows from timestamps and forcedalignment boundaries, and compile immutable manifests specifying event order, response requirements, challenge conditions, and same-session questions. Gold operations, memory states, and reference answers are retained for supervision, scoring, and audit, but withheld from the evaluated speech model and its answer generator. Appendix A details construction, quality controls, and the data card.Adaptive evaluation distinguishes planned from realized challenges. Interruption and backchannel events are triggered by decoded assistant speech, whereas pause and speech-to-others conditions use scheduled observation windows. For session s, the realized challenge rate (RCR) and its macro-average are defined as

$$
\mathrm { R C R } _ { s } = \frac { N _ { \mathrm { r e a l i z e d } } ^ { ( s ) } } { N _ { \mathrm { p l a n n e d } } ^ { ( s ) } } , \qquad \overline { { \mathrm { R C R } } } = \frac { 1 } { | \cal { S } | } \sum _ { s \in \cal { S } } \mathrm { R C R } _ { s } ,\tag{2}
$$

where S contains sessions with at least one planned challenge.

Other session-level ratios are also macro-averaged unless stated otherwise. Conditional interaction metrics use eligible realized or observable events and are reported alongside RCR and validity indicators. All memory answers are scored by the same deterministic bilingual lexical scorer. Memory questions retain fixed question denominators, so silence and unanswered cases receive no credit. Latest-value answers must assert the active value without asserting a superseded alternative, and silence does not count as distractor rejection. Exact scheduling rules and metric denominators are specified in Appendices B and C; the released artifact structure is documented in Appendix J.

## 4 PERFORMANCE OF CURRENT FULL-DUPLEX MODELS

## 4.1 EXPERIMENTAL SETUP AND REPORTING PROTOCOL

We evaluate five full-duplex systems using identical source audio, event order, questions, and scheduler logic. Since model architectures and response tendencies differ, cross-model results are descriptive. Memory questions retain fixed denominators, while conditional interaction metrics follow their eligible-event denominators. RCR is therefore treated as challenge exposure rather than

Table 3: Full-duplex interaction and memory fidelity on DuplexLedger.
<table><tr><td colspan="8">(a) Full-duplex interaction</td></tr><tr><td>Model</td><td>RCR†</td><td>cFD↑</td><td>Early↓</td><td>Pre-end↓</td><td>Win-Intr.↓</td><td>Stop@0.8↑</td><td>Stop (s)↓</td></tr><tr><td>MiniCPM-0 4.5</td><td>73.51</td><td>40.23</td><td>18.70</td><td>44.38</td><td>59.47</td><td>55.69</td><td>1.032</td></tr><tr><td>Freeze-Omni</td><td>86.17</td><td>28.40</td><td>49.76</td><td>91.82</td><td>73.64</td><td>45.75</td><td>1.239</td></tr><tr><td>Moshi</td><td>44.63</td><td>37.36</td><td>66.59</td><td>72.25</td><td>45.11</td><td>60.79</td><td>0.777</td></tr><tr><td>Lychee-FD</td><td>75.64</td><td>47.32</td><td>24.68</td><td>57.14</td><td>53.89</td><td>90.12</td><td>0.312</td></tr><tr><td>PersonaPlex</td><td>93.54</td><td>22.74</td><td>46.64</td><td>72.73</td><td>90.62</td><td>16.40</td><td>1.689</td></tr><tr><td colspan="8">(b) Memory fidelity</td></tr><tr><td>Model</td><td>Resp.↑</td><td>Latest↑</td><td>Stale↓</td><td>D-Adopt↓</td><td></td><td>D-Rej.↑</td><td>Final↑</td></tr><tr><td>MiniCPM-0 4.5</td><td>98.90</td><td>4.30</td><td>0.59</td><td>14.17</td><td></td><td>85.19</td><td>2.01</td></tr><tr><td>Freeze-Omni</td><td>99.94</td><td>11.61</td><td>3.38</td><td>65.90</td><td></td><td>34.10</td><td>0.00</td></tr><tr><td>Moshi</td><td>54.31</td><td>2.19</td><td>0.32</td><td>44.15</td><td></td><td>8.32</td><td>0.04</td></tr><tr><td>Lychee-FD</td><td>92.41</td><td>25.50</td><td>1.65</td><td>48.17 79.71</td><td></td><td>51.55</td><td>1.06</td></tr><tr><td>PersonaPlex</td><td>96.40</td><td>5.85</td><td>2.29</td><td></td><td></td><td>17.46</td><td>0.24</td></tr></table>

Bold and underlined: best and second best. Values are percentages except Stop (s). † RCR is diagnostic; cFD is conditional. Win-Intr.: windowed speech intrusion; D-Adopt/D-Rej.: distractor adoption/rejection; Final: final-state coverage.

a performance score. The execution and scoring protocols are detailed in Appendices B and C;   
model-specific inference settings and complete baseline results appear in Appendices D and H.

## 4.2 BENCHMARK RESULTS

[Takeaway 1] No modeljointly dominates full-duplex interaction and memory fidelity. Table 3 separates the two capabilities. Lychee-FD leads in conditional full-duplex performance and latestvalue accuracy. MiniCPM-o 4.5 achieves the lowest early-onset and pre-end overlap rates, together with the strongest distractor rejection and final-state coverage. Moshi pairs the lowest stale rate with only 54.31% response coverage. PersonaPlex realizes the most challenges but obtains the lowest cFD. These contrasting profiles preclude a meaningful scalar ranking.

## 4.3 FULL-DUPLEX COORDINATION UNDER ADAPTIVE CHALLENGES

[Takeaway 2] Challenge exposure, boundary control, and interruption recovery are distinct capabilities. Lychee-FD combines substantial exposure with the highest cFD, 90.12% stopping success within 800 ms, and the shortest stopping latency. Its remaining pre-end overlap and window intrusion nevertheless show that fast interruption recovery does not ensure correct turn boundaries. MiniCPM-o 4.5 instead minimizes early onset and pre-end overlap, while PersonaPlex pairs the highest RCR with the lowest cFD, greatest window intrusion, and slowest stopping. RCR must therefore be interpreted as exposure rather than interaction quality.

## 4.4 TRANSACTIONAL MEMORY FAILURES AND INTERACTION–MEMORY MISALIGNMENT

[Takeaway 3] Local value revision does not yield a coherent persistent state. Lychee-FD reaches 25.50% latest-value accuracy but covers only 1.06% of final-state facts and adopts 48.17% of distractors. MiniCPM-o 4.5 is more selective but correctly answers only 4.30% of update questions. Freeze-Omni responds almost universally but retains high stale-value and distractor errors. Moshi similarly combines a low stale rate with limited response coverage and latest-value accuracy.

Even the best final-state coverage is only 2.01%. Current systems can coordinate speech, recover isolated revisions, or reject irrelevant content, yet none jointly supports event admission, supersession, exclusion, and state consolidation. Maintaining an evolving state during full-duplex interaction is therefore the central limitation motivating ACT-Mem in Section 5.

![](images/22e0ba9d6f22a101d8910543a3ccabf0067355703aa7168c1c4279e14f8bce3b.jpg)  
Figure 3: Overview of the ACT-Mem framework for transactional memory in full-duplex agents.

## 5 ACT-MEM: TRANSACTIONAL MEMORY FOR FULL-DUPLEX AGENTS

## 5.1 DESIGN MOTIVATION AND OVERVIEW

The failures in Section 4.4 motivate explicit control over memory admission and revision. We introduce ACT-Mem (Auditable and Causal Transactional Memory), an external module coupled with a frozen MiniCPM-o 4.5 backbone (Cui et al., 2026). As shown in Figure 3, streaming audio enters both the frozen full-duplex backbone and a frozen ASR/VAD module. The latter supplies transcripts and timing to a trainable event controller, which manages a transactional ledger implemented as versioned memory. A typed retriever constructs evidence cards for memory questions. Only these cards enter the backbone through QA-only text prefill; all speech-generation components remain frozen. Architectural and runtime details are provided in Appendix E.

## 5.2 EVENT ADMISSION AND TRANSACTION LIFECYCLE

[Admission] Persistence requires explicit evidence admission. A deterministic parser extracts candidate entity–value slots from the event transcript, while the learned controller predicts eventlevel addressee, completeness, operation, and confidence for admission. Each proposal enters PRE-PARE with score

$$
g _ { t } = p _ { t } ^ { \mathrm { a d d r } } p _ { t } ^ { \mathrm { c o m p l e t e } } \big [ p _ { t } ( \mathbf { W } \mathrm { R I T E } ) + p _ { t } ( \mathbf { U } \mathrm { P D A T E } ) \big ] c _ { t } ,\tag{3}
$$

where $p _ { t } ^ { \mathrm { a d d r } }$ and $p _ { t } ^ { \mathrm { c o m p l e t e } }$ estimate assistant-directedness and semantic completeness, and $c _ { t } \in [ 0 , 1 ]$ is the controller confidence.

A proposal enters COMMIT when $g _ { t } \geq \tau _ { \mathrm { w r i t e } } .$ Non-addressed or incomplete proposals enter ABORT; ambiguous or low-confidence proposals remain UNCERTAIN. Neither is retrievable. Pause and silence create no persistent transaction. WRITE creates an active fact, UPDATE replaces an active value, IGNORE rejects the proposal, and RETAIN preserves the ledger. Committed entries retain their source event, predicted roles, order, completeness, and confidence for audit.

## 5.3 CAUSAL SUPERSESSION AND VERSIONED MEMORY

[Supersession] Recency alone cannot overwrite an active value. The transactional ledger extends the state in Section 3.1 with provenance and version links. A candidate $m _ { j }$ may supersede an active item $m _ { i }$ only when

$$
k _ { j } = k _ { i } , \qquad t _ { j } > t _ { i } , \qquad { \widehat { o } } _ { j } = { \mathrm { U P D A T E } } , \qquad g _ { j } \geq \tau _ { \mathrm { w r i t e } } ,\tag{4}
$$

where $k _ { i }$ is the entity key and $t _ { i }$ denotes source-event order. A later same-key mention is therefore insufficient without an admissible update.

A successful transaction marks m as superseded, activates $m _ { j }$ , and records their correction link. Both versions remain auditable, but only active values are exposed to the typed retriever. Superseded, uncertain, aborted, and ignored content cannot enter an evidence card.

## 5.4 TYPED RETRIEVAL AND MEMORY CONDITIONING

[Typed retrieval] Question type controls evidence scope and card generation. The typed retriever maps each question to one of four behaviors. A latest-value question selects the target entity’s active value, a recall question retrieves up to four active facts, and a summary question aggregates up to eight facts across types and entities. Rejection and behavior questions produce no card, preventing inadmissible content from being disclosed to the backbone.

For retrieval-enabled questions, ranking combines dense similarity (Karpukhin et al., 2020), lexical overlap, recency, controller confidence, and stale/distractor penalties. The resulting card contains only typed facts and active values; insufficient evidence yields no card.

ACT-Mem conditions generation without modifying the decoder (Lewis et al., 2020). At a stablesilence QA boundary, the backbone processes the question audio, receives one evidence card through QA-only text prefill, and generates a spoken answer. Context events and full-duplex challenges continue through the native audio interface without memory prefill.

## 5.5 TRAINING, CALIBRATION, AND MODEL COMPLEXITY

Training uses frozen-ASR transcripts from the session-disjoint training split. The event controller learns operation, addressee, completeness, and confidence prediction. Slot extraction and entity-key normalization use non-trainable rules, and the ledger applies version transitions. Its objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { c t r l } } = \mathcal { L } _ { \mathrm { o p } } ^ { \mathrm { w C E } } + 0 . 3 \mathcal { L } _ { \mathrm { a d d r } } ^ { \mathrm { C E } } + 0 . 3 \mathcal { L } _ { \mathrm { c o m p l e t e } } ^ { \mathrm { C E } } + 0 . 2 \mathcal { L } _ { \mathrm { c o n f } } ^ { \mathrm { B C E } } , } \\ & { \mathcal { L } _ { \mathrm { r e t r } } = \mathcal { L } _ { \mathrm { B C E } } ( \mathbf { s } , \mathbf { y } ) + \displaystyle \frac { 0 . 5 } { | \mathcal { P } | | \mathcal { N } | } \sum _ { p \in \mathcal { P } } \sum _ { n \in \cal N } \left[ 0 . 2 5 - s _ { p } + s _ { n } \right] _ { + } . } \end{array}\tag{5}
$$

Weighted operation cross-entropy uses class weights (1.0, 1.5, 1.4, 0.8) for WRITE, UPDATE, IG-NORE, and RETAIN. s contains retrieval scores, y gives binary relevance labels, and $\mathcal { P }$ and $\mathcal { N }$ contain positive and forbidden, hard-negative, or distractor items in the QA context. Joint training updates the controller and retriever; adaptation freezes the controller and updates only the retriever.

We use the session-disjoint splits in Section 3.4. Gold states and evidence labels provide supervision only; future events never enter the current prefix. Calibration on 40 sessions gives $\tau _ { \mathrm { w r i t e } } = 0 . 3 5$ without gradient updates. ACT-Mem adds 8.92M trainable parameters: 4.59M in the event controller and 4.33M in the typed retriever. The backbone and ASR/VAD module remain frozen. Training uses 8 NVIDIA A100 GPUs; optimization and augmentation details are provided in Appendix F.

## 6 EVALUATION OF ACT-MEM

We evaluate ACT-Mem on the same 219 held-out sessions used for the baseline study, containing 1,314 questions/checks and 438 latest-value questions. The controlled comparison retains the frozen MiniCPM-o 4.5 backbone; no held-out session is used for training, checkpoint selection, or calibration. Evaluation follows Sections 3.4 and 4.1. Relative changes use unrounded scores, and pointwise 95% confidence intervals for latest-value accuracy and final-state coverage are obtained from 20,000 paired bootstrap resamples of sessions. Further details appear in Appendices C and G.

## 6.1 MAIN RESULTS

Figure 4 compares the complete ACT-Mem configuration with its frozen MiniCPM-o 4.5 backbone.

[Takeaway 1] ACT-Mem substantially improves latest-value recovery, while complete final-state reconstruction remains difficult. Latest-value accuracy increases from 4.11% to 8.68% (+111.2%; +4.57 percentage points, 95% CI [1.60, 7.53]), whereas final-state coverage rises from 1.27% to 2.00% (+57.5%; +0.73 points, 95% CI [−0.49, 1.99]). The latest-value interval excludes zero, providing evidence that ACT-Mem more reliably recovers active committed values after revision. Final-state coverage, however, remains low and its interval crosses zero. The controlled evidence therefore supports improved local state revision, but not yet complete reconstruction of the sessionwide memory state.

![](images/5ce40f60dc57ec34de65f8a9b2fb522bf9041e085378f76dc194f28e53a82313.jpg)  
Figure 4: ACT-Mem versus its frozen MiniCPM-o 4.5 backbone on 219 held-out sessions; error bars show 95% paired-bootstrap CIs for Latest and Final.

[Takeaway 2] The memory gains are concentrated in selective commitment rather than uniformly improving every outcome. Stale assertions and distractor adoption decrease by 0.22 and 2.28 percentage points, respectively, showing that the latest-value gain is accompanied by fewer conflicting and incorrectly admitted facts. Distractor rejection and response rate decline by 2.28 and 1.45 points, although the latter remains high at 97.41%. Because distractor adoption and rejection are separately credited under fixed denominators, they are not complementary; their joint decline means that ACT-Mem produces fewer explicit adoptions but also fewer credited rejections. The principal benefit is therefore more selective state construction, while robust distractor rejection remains improvable.

## 6.2 EFFECTS ON FULL-DUPLEX INTERACTION

[Takeaway 3] ACT-Mem improves most interaction-boundary metrics but slows interruption stopping. cFD increases by 10.8%, while early response, pre-end speech, and windowed intrusion decrease by 18.6%, 7.6%, and 5.1%, respectively. Because cFD is conditioned on realized challenges and RCR decreases from 76.35% to 74.10%, this improvement should be interpreted as better be havior on exposed challenges rather than uniformly greater challenge coverage. The main trade-off is interruption stopping: Stop@0.8 falls from 51.06% to 45.00%, and median stopping latency increases from 1.123 to 1.340 seconds. Thus, ACT-Mem improves observed onset and overlap control, while rapid interruption recovery remains its principal interaction weakness.

## 6.3 ABLATION STUDIES

We evaluate three inference-time variants with fixed memory-model weights. FIFO memory replaces the versioned ledger with a 32-slot queue and removes supersession and latest-only selection. Without addressee gating removes speaker/addressee rejection and the learned gate. Untyped re trieval applies top-4 retrieval to every question type, including rejection questions.

[Takeaway 4] The full configuration balances value recovery with selective memory use. ACT-Mem leads all three reported interaction metrics and three of four memory metrics. Compared with FIFO memory, its stale-assertion rate is 79.8% lower; compared with removing addressee gating, distractor adoption is 34.2% lower. Untyped retrieval achieves 9.59% latest-value accuracy, but stale assertions and distractor adoption reach 1.60% and 19.63%. ACT-Mem thus trades 0.91 percentage points of latest-value accuracy for fewer stale assertions and distractor adoptions. Interaction differences remain descriptive across runs: QA-only retrieval does not establish a causal effect on preceding interaction boundaries.

Table 4: Ablations on 219 sessions. Best and second-best scores are bold and underlined.
<table><tr><td></td><td colspan="3">Full-duplex interaction</td><td colspan="4">Memory fidelity</td></tr><tr><td>Variant</td><td>cFD↑</td><td>Early↓</td><td>Pre-end↓</td><td>Latest↑</td><td>Stale↓</td><td>D-Adopt↓</td><td>Final↑</td></tr><tr><td>FIFO memory</td><td>34.63</td><td>37.21</td><td>48.65</td><td>7.08</td><td>2.28</td><td>12.33</td><td>1.61</td></tr><tr><td>w/o addressee gating</td><td>40.14</td><td>31.48</td><td>46.36</td><td>6.39</td><td>0.91</td><td>17.35</td><td>1.75</td></tr><tr><td>Untyped retrieval</td><td>40.33</td><td>29.98</td><td>44.61</td><td>9.59</td><td>1.60</td><td>19.63</td><td>1.82</td></tr><tr><td>ACT-Mem</td><td>41.19</td><td>16.03</td><td>41.82</td><td>8.68</td><td>0.46</td><td>11.42</td><td>2.00</td></tr></table>

All values are percentages.

![](images/8680dfe6a99a9b7181d95a5f3e3c5d103623d0f24153c6d82e4541105870ba39.jpg)  
Figure 5: ACT-Mem cases across four full-duplex scenarios, showing controller decisions, memory transitions, and paired outcomes.

## 6.4 RESULTS BY LANGUAGE AND SCENARIO

[Takeaway 5] Improvements are bilingual but scenario-dependent. Latest-value accuracy increases from 5.83% to 13.11% on English sessions and from 2.59% to 4.74% on Chinese sessions. The largest absolute gains occur for backchannels (+10.53 points) and speech to others (+6.36), followed by pause handling (+1.82). Under user interruption, however, latest-value accuracy decreases from 3.85% to 2.88%; backchannel final-state coverage also falls from 2.34% to 1.10%. These descriptive breakdowns identify interruption-time updating as the principal remaining weakness.

## 6.5 CASE STUDY

Figure 5 connects interaction events with admission decisions, committed states, and QA outcomes. The controller governs admission; the ledger records supersession and exclusion; typed retrieval supplies active, question-relevant evidence. DuplexLedger’s reference transitions and ACT-Mem’s transaction traces distinguish correct operation predictions from successful slot resolution, supersession, or exclusion (Appendix I, Table 21). These records support failure localization beyond answer scores by separating predicted operations, committed changes, and downstream QA outcomes.

## 6.6 ADMISSION-THRESHOLD SENSITIVITY

Table 5 summarizes admission sensitivity on 40 calibration sessions with fixed model weights. Event-level admission precision remains 100% at all tested thresholds; 0.35 and 0.45 share the highest F1. Increasing the

Table 5: Threshold sensitivity on 40 calibration sessions.
<table><tr><td>Twrite</td><td>0.35</td><td>0.45</td><td>0.50</td><td>0.55</td><td>0.65</td><td>0.75</td></tr><tr><td>Recall (%)</td><td>55.26</td><td>55.26</td><td>54.74</td><td>53.68</td><td>50.53</td><td>48.95</td></tr><tr><td>F1 (%)</td><td>71.19</td><td>71.19</td><td>70.75</td><td>69.86</td><td>67.13</td><td>65.72</td></tr></table>

threshold from 0.35 to 0.75 reduces recall from 55.26% to 48.95% and F1 from 71.19% to 65.72%. These results indicate local stability across two adjacent settings and a recall cost from stricter ad mission, with conclusions limited to calibration-set admission behavior.

## 7 CONCLUSION

We frame persistent memory during concurrent speech as selective state management. DuplexLedger aligns adaptive challenges, event-level operations, explicit reference states, and spoken QA across 1,094 bilingual sessions. This alignment supports joint evaluation of interaction and memory, while reference transitions and system traces help diagnose admission, supersession, exclusion, retrieval, and generation failures. Across five systems, no model dominates both interaction and memory fidelity. ACT-Mem combines an event controller, versioned ledger, and typed retriever with a frozen backbone. On 219 held-out sessions, latest-value accuracy rises from 4.11% to 8.68%, with fewer stale assertions and distractor adoptions. Several interaction-boundary metrics also improve, although interruption stopping becomes slower. These findings motivate explicit, auditable state management and further validation on natural conversations, additional languages, longer timescales, and other backbones.

## ETHICS STATEMENT

DuplexLedger contains AI-generated dialogues and synthetic speech rather than recordings of human participants or private conversations. Its controlled scenarios may nevertheless encode generation-model biases and should not be treated as representative of unrestricted real-world dialogue.

## REPRODUCIBILITY STATEMENT

Dataset construction, execution protocols, metric denominators, baseline configurations, ACT-Mem implementation and training, statistical procedures, complete baseline results, qualitative traces, and artifact organization are documented in Appendices A–J.

## AI USE STATEMENT

Generative models were used to construct synthetic benchmark dialogues as documented in Appendix A. The authors specified the schemas, validation rules, evaluation protocol, and analyses, and remain responsible for the resulting content and claims.

## REFERENCES

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, Chang Zhou, and Jingren Zhou. Qwen2-Audio technical report, 2024. URL https://arxiv.org/abs/2407.10759.

Junbo Cui, Bokai Xu, Chongyi Wang, Tianyu Yu, Weiyue Sun, Yingjing Xu, Tianran Wang, Zhihui He, Wenshuo Ma, Tianchi Cai, Jiancheng Gui, Luoyuan Zhang, Xian Sun, Fuwei Huang, Moye Chen, Zhuo Lin, Hanyu Liu, Qingxin Gui, Qingzhe Han, Yuyang Wen, Huiping Liu, Rongkang Wang, Yaqi Zhang, Hongliang Wei, Chi Chen, You Li, Kechen Fang, Jie Zhou, Yuxuan Li, Guoyang Zeng, Chaojun Xiao, Yankai Lin, Xu Han, Maosong Sun, Zhiyuan Liu, and Yuan Yao. MiniCPM-o 4.5: Towards real-time full-duplex omni-modal interaction, 2026. URL https://arxiv.org/abs/2604.27393.

Alexandre Defossez, Laurent Mazar´ e, Manu Orsini, Am´ elie Royer, Patrick P´ erez, Herv´ e J´ egou,´ Edouard Grave, and Neil Zeghidour. Moshi: A speech-text foundation model for real-time dialogue. arXiv preprint arXiv:2410.00037, 2024. URL https://arxiv.org/abs/2410. 00037.

Advait Gosai, Tyler Vuong, Utkarsh Tyagi, Steven Li, Wenjia You, Miheer Bavare, Arda Uc¸ar, Zhongwang Fang, Brian Jang, Bing Liu, and Yunzhong He. Audio MultiChallenge: A multi-turn evaluation of spoken dialogue systems on natural human interaction. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 35740–35770. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long. 1654. URL https://aclanthology.org/2026.acl-long.1654/.

Bernal Jimenez Guti ´ errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. Hip-´ poRAG: Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems, volume 37, 2024. doi: 10.52202/ 079017-1902. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/6ddc001d07ca4f319af96a3024f6dbd1-Abstract.html.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6769–6781. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020. emnlp-main.550. URL https://aclanthology.org/2020.emnlp-main.550/.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨ aschel, Sebas-¨ tian Riedel, and Douwe Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pp. 9459–9474, 2020. URL https://proceedings.neurips.cc/paper/2020/hash/ 6b493230205f780e1bc26945df7481e5-Abstract.html.

Guan-Ting Lin, Jiachen Lian, Tingle Li, Qirui Wang, Gopala Anumanchipalli, Alexander H. Liu, and Hung-yi Lee. Full-duplex-bench: A benchmark to evaluate full-duplex spoken dialogue models on turn-taking capabilities. In Proceedings of the IEEE Automatic Speech Recognition and Understanding Workshop, 2025. URL https://arxiv.org/abs/2503.04721.

Guan-Ting Lin, Chen Chen, Zhehuai Chen, and Hung-yi Lee. Full-duplex-bench v3: Evaluating tool use under speech disfluency, 2026a. URL https://arxiv.org/abs/2604.04847.

Guan-Ting Lin, Shih-Yun Shan Kuan, Qirui Wang, Jiachen Lian, Tingle Li, and Hung-yi Lee. Fullduplex-bench v1.5: Evaluating overlap handling for full-duplex spoken dialogue models. In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing, 2026b. URL https://arxiv.org/abs/2507.23159.

Guan-Ting Lin et al. Full-duplex-bench v2: An adaptive benchmark for full-duplex spoken dialogue models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, 2026c. URL https://arxiv.org/abs/2510.07838.

Zhenyu Liu, Xuanyu Zhang, Yunxin Li, Qixun Teng, Shenyuan Jiang, Haolan Chen, Mingjun Zhao, Fanbo Meng, Yu Xu, Yancheng He, Baotian Hu, Haizhou Li, and Min Zhang. Hierarchical acoustic-semantic modeling: Modality separation and semantic coherence for full-duplex SLMs. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9264–9280. Association for Computational Linguistics, 2026. doi: 10.18653/v1/2026.acl-long.419. URL https://aclanthology.org/2026. acl-long.419/.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. Evaluating very long-term conversational memory of LLM agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pp. 13851–13870, 2024. doi: 10.18653/v1/2024.acl-long.747. URL https://aclanthology.org/2024. acl-long.747/.

Smit Nautambhai Modi, Gandharv Mahajan, Marc Wetter, and Randall Welles. Echochain: Evaluating state updates under interruption in full-duplex spoken dialogue, 2026. URL https: //arxiv.org/abs/2604.16456.

Tu Anh Nguyen, Eugene Kharitonov, Jade Copet, Yossi Adi, Wei-Ning Hsu, Ali Elkahky, Paden Tomasello, Robin Algayres, Benoˆıt Sagot, Abdelrahman Mohamed, and Emmanuel Dupoux. Generative spoken dialogue language modeling. Transactions of the Association for Computational Linguistics, 11:250–266, 2023. doi: 10.1162/tacl a 00545. URL https:// aclanthology.org/2023.tacl-1.15/.

Charles Packer, Vivian Fang, Shishir G. Patil, Kevin Lin, Sarah Wooders, and Joseph E. Gonzalez. MemGPT: Towards LLMs as operating systems, 2023. URL https://arxiv.org/abs/ 2310.08560.

Rajarshi Roy, Jonathan Raiman, Sang-gil Lee, Teodor-Dumitru Ene, Robert Kirby, Sungwon Kim, Jaehyeon Kim, and Bryan Catanzaro. PersonaPlex: Voice and role control for full duplex conversational speech models, 2026. URL https://arxiv.org/abs/2602.06053.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. RAPTOR: Recursive abstractive processing for tree-organized retrieval. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=GN921JHCRw.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. SALMONN: Towards generic hearing abilities for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=t8EF6AAImY.

Bandhav Veluri, Benjamin N. Peloquin, Bokai Yu, Hongyu Gong, and Shyamnath Gollakota. Beyond turn-based interfaces: Synchronous LLMs as full-duplex dialogue agents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 21390–21402. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024.emnlp-main.1192. URL https://aclanthology.org/2024.emnlp-main.1192/.

Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy F. Chen. AudioBench: A universal benchmark for audio large language models. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4297– 4316. Association for Computational Linguistics, 2025a. doi: 10.18653/v1/2025.naacl-long.218. URL https://aclanthology.org/2025.naacl-long.218/.

Chengyou Wang, Hongfei Xue, Guojian Li, Zhixian Zhao, Shuiyuan Wang, Shuai Wang, Xin Xu, Hui Bu, and Lei Xie. HumDial-FDBench: A human-recorded benchmark for full-duplex spoken dialogue, 2026. URL https://arxiv.org/abs/2604.21406.

Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, and Peilin Zhao. NTPP: Generative speech language modeling for dual-channel spoken dialogue via next-token-pair prediction. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 63912– 63928. PMLR, 2025b. URL https://proceedings.mlr.press/v267/wang25by. html.

Xiong Wang, Yangze Li, Chaoyou Fu, Yike Zhang, Yunhang Shen, Lei Xie, Ke Li, Xing Sun, and Long Ma. Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen LLM. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 63345–63354. PMLR, 2025c. URL https: //proceedings.mlr.press/v267/wang25aw.html.

Di Wu et al. LongMemEval: Benchmarking chat assistants on long-term interactive memory. In International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=pZiyCaVuti.

Zhifei Xie and Changqiao Wu. Mini-Omni: Language models can hear, talk while thinking in streaming, 2024. URL https://arxiv.org/abs/2408.16725.

Zhifei Xie et al. Voicemem: Streaming memory for long-term voice interaction, 2026. URL https://arxiv.org/abs/2608.26005.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for LLM agents. In Advances in Neural Information Processing Systems, 2025. URL https://arxiv.org/abs/2502.12110.

Yiqun Yao et al. Egomem: Lifelong memory agent for full-duplex omnimodal models, 2025. URL https://arxiv.org/abs/2509.11914.

Dong Zhang, Shimin Li, Xin Zhang, Jun Zhan, Pengyu Wang, Yaqian Zhou, and Xipeng Qiu. SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities. In Findings of the Association for Computational Linguistics: EMNLP 2023, pp. 15757– 15773. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.findings-emnlp. 1055. URL https://aclanthology.org/2023.findings-emnlp.1055/.

Guibin Zhang, Muxin Fu, and Shuicheng Yan. MemGen: Weaving generative latent memory for self-evolving agents. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=vI56m4Iu4e.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. MemoryBank: Enhancing large language models with long-term memory. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(17):19724–19731, 2024. doi: 10.1609/aaai.v38i17.29946. URL https:// ojs.aaai.org/index.php/AAAI/article/view/29946.

## A DUPLEXLEDGER CONSTRUCTION AND DATA CARD

## A.1 CONSTRUCTION PRINCIPLES

DuplexLedger was constructed to evaluate whether a spoken dialogue agent can maintain a coherent memory state while simultaneously respecting full-duplex interaction boundaries. We designed each session around three coupled requirements: (i) the dialogue must contain facts that evolve through additions and corrections; (ii) some audible content must be inadmissible to persistent memory; and (iii) the interaction must contain a concurrency event whose correct handling depends on the model’s runtime speaking state. These requirements led to four scenarios: user interruption, user backchannel, within-turn pause, and speech addressed to another person.

All dialogue content is synthetic. We used GPT-5.5 to generate the underlying Chinese and English task descriptions, conversational utterances, factual values, corrections, distractors, and reference answers under scenario-specific instructions. Generation was constrained by a structured schema specifying speaker roles, event order, memory dependencies, challenge types, and evaluation targets. The generated content was subsequently normalized and instantiated by deterministic construction programs. The language model was used only during dataset creation and is not part of the benchmark execution or scoring pipeline.

## A.2 TASK AND DIALOGUE GENERATION

We constructed sessions from a broad collection of everyday task templates, including scheduling, planning, customer support, troubleshooting, study assistance, household coordination, travel, document preparation, creative work, and low-pressure personal organization. The Chinese and English construction pools contain 73 and 72 base task templates, respectively, covering 72 high-level domains in each language.

Each base task specifies two revisable attributes and several persistent facts. A typical task contains:

• an initial value and a corrected value for a primary attribute;

• an initial value and a corrected value for a secondary attribute;

• a stable goal or constraint that should remain active;

• an additional task-specific detail;

• a named person and an associated responsibility;

• a plausible but irrelevant distractor;

• an expected consolidated final state.

To reduce exact repetition, each task was combined with a scenario-specific identifier, one of four discourse-style variants, and one of twelve instance-level modifiers. The modifiers vary properties such as budget conservatism, scheduling slack, communication requirements, fallback plans, execution complexity, and family or workplace coordination. Timing, challenge position, pause length, overlap duration, and short-feedback form were sampled using fixed construction seeds.

We also varied how the user concludes a session. A final turn may fully restate the task, omit the recap, provide only a partial recap, contain a self-correction, or explicitly request that the assistant not summarize. The intended proportions were 40%, 25%, 20%, 10%, and 5%, respectively. The released corpus contains 434 complete recaps, 268 sessions without a complete final recap, 218 incomplete recaps, 118 recaps containing a correction, and 56 explicit no-summary requests. This variation prevents the final memory state from being recoverable solely by copying the last user turn.

## A.3 SCENARIO-SPECIFIC INTERACTION DESIGN

The same underlying memory requirements are combined with four distinct full-duplex phenomena.

User interruption. The user begins with a provisional task state and later interrupts an active assistant response to correct one or more values. The interruption should terminate the obsolete response trajectory, and the corrected values should supersede their earlier versions. A second interruption may clarify that a nearby detail is irrelevant and must not enter persistent memory.

User backchannel. Short acknowledgements such as “yes,” “okay,” or their Chinese equivalents are inserted while the assistant is speaking. These utterances provide listener feedback but introduce no new task fact. A correct system should continue its response without interpreting the backchannel as an interruption or memory write.

Pause handling. The user produces an incomplete utterance, pauses, and then resumes with a completion or correction. The pause is represented explicitly rather than as a completed turn. A correct system should wait for the continuation and must not commit a provisional value merely because a long silence occurred.

Speech addressed to another person. The session contains both user speech directed to a bystander and audible speech from that bystander. Such content is topically plausible but is not addressed to the assistant. It therefore serves as a controlled distractor: the model should neither respond to it nor incorporate it into the official task state. The user later returns explicitly to the assistant and continues the admissible task.

## A.4 STRUCTURED EVENT AND MEMORY ANNOTATION

A session is represented as an ordered list of speech and silence events. Each event records its identifier, speaker, addressee, dialogue act, planned time interval, audio-rendering status, memory dependencies, and candidate memory writes. Events spoken by the primary user or bystander are audible; reference assistant events specify the intended conversational trajectory but are not rendered into the benchmark input.

Memory is annotated at the item level. The annotation vocabulary covers goal, schedule, constraint, preference, episodic, named entity, task state, and correction. Each item contains a stable memory identifier, semantic type, textual content, and source event. Corrections additionally identify the earlier item that they supersede. Distractors are annotated separately and are excluded from the oracle final state.

For evaluation, the event annotations are mapped to four memory operations: WRITE, UPDATE, IGNORE, and RETAIN. Ordinary admissible facts produce WRITE; explicit corrections and clarifications produce UPDATE; backchannels, side conversations, and bystander speech produce IGNORE; and pauses without confirmed content produce RETAIN. This mapping supplies the event-level supervision used to distinguish admission errors from supersession, retrieval, and answer-generation errors.

The construction procedure enforces two provenance constraints. First, every item in the oracle final state must be supported by an audible user event. Muted reference-assistant events cannot introduce a gold fact. Second, a superseding item must refer to an earlier item in the same session; future or cross-session references are disallowed.

## A.5 MEMORY QUESTIONS

Every session contains six spoken evaluation targets:

1. two latest-value questions, one for each corrected attribute;

2. one factual-recall question about a stable requirement;

3. one distractor-rejection question;

4. one final-state summary question;

5. one scenario-specific behavioral check.

Each question is aligned with its expected answer, supporting memory identifiers or events, and, when applicable, forbidden obsolete or distractor items. The latest-value questions require the corrected value and prohibit credit when the obsolete value is asserted. The distractor question tests whether the system rejects inadmissible content rather than rewarding silence alone. The summary question requires coverage of multiple active facts and therefore evaluates state consolidation beyond isolated retrieval.

Questions are synthesized as audio and presented after all contextual and challenge events, without resetting the model state. The question audio, reference text, and gold evidence annotations are stored separately. Reference answers and memory annotations are used only for scoring and are not exposed to the evaluated system.

## A.6 SPEECH SYNTHESIS AND TEMPORAL ALIGNMENT

All speech is generated using text-to-speech synthesis; the dataset contains no recordings from human participants. Audible event segments are rendered at 24 kHz and assembled according to the structured event timeline. Scenario-specific speaker assignments distinguish the primary user from the bystander in multi-speaker sessions. The reference assistant remains muted, because the assistant audio during evaluation must be produced by the system under test.

We apply forced alignment to the synthesized speech to obtain word-level English boundaries and character-level Chinese boundaries. Punctuation is removed for alignment, while the original text is retained in the source annotation. The aligned event window is defined by the first and last aligned units with 120 ms of boundary padding. For each event, we retain the source sampling rate, number of samples, first and last aligned boundaries, leading and trailing silence, playback sample range, source duration, SHA-256 checksum, and quality-control flags.

The fixed mixed waveform is retained as a compatibility artifact, but it is not the authoritative input for adaptive evaluation. The final protocol instead replays aligned event segments according to a closed-loop stage manifest. This distinction is necessary because a full-duplex challenge should occur when the evaluated model is actually speaking, rather than at a timestamp derived from a prerecorded reference trajectory.

## A.7 ADAPTIVE SESSION MATERIALIZATION

For each session, we compile a chronological execution manifest containing ordinary context playback, adaptive challenges, recovery periods, and same-session memory QA. Ordinary user events are streamed in real time. Before an interruption or backchannel, the scheduler waits for decoded-audio VAD to confirm that the evaluated model is speaking. It then injects the corresponding user event. If the required speaking state is not observed within the timeout, the challenge remains unrealized and the timeout is recorded.

Pause intervals are represented as explicit wait stages. Speech directed to another person is played through a dedicated side-talk stage and is marked as not addressed to the assistant. After an ordinary response-required event, the scheduler observes response onset and subsequent silence before advancing. The six audio questions are asked only after the interaction stages have finished.

The complete session uses one persistent model state:

• the model is reset before a new session;

• it is not reset between context and challenges;

• it is not reset between challenges and memory QA;

• it is not reset between individual QA items.

The execution protocol uses 80-ms input chunks. Interruption and backchannel triggers require at least 500 ms and 800 ms of active assistant speech, respectively. Question input retains 400 ms of post-speech tail silence, and answer windows are determined dynamically. Runtime failures, trigger timeouts, and missing responses are recorded rather than silently discarded.

## A.8 VALIDATION AND QUALITY CONTROL

We apply validation at three levels.

Structural validation. Before synthesis, each session must contain all required fields, unique event and memory identifiers, chronologically valid timestamps, valid memory dependencies, and consistent overlap annotations. Memory reads may reference only previously introduced items. Supersession links must point to existing earlier items, and every final-state fact must trace to audible user speech.

Audio and alignment validation. The construction pipeline verifies the existence and duration of every waveform, preserves source-file checksums, and records missing alignments, leading silence, and trailing residuals. The released corpus contains no missing event-audio files. Of 7,687 speech events, 7,589 have forced-alignment boundaries. The remaining events are retained with explicit missing-alignment flags rather than removed. In addition, 3,007 aligned events contain more than 0.5 s of trailing residual audio; these cases remain visible through per-event quality-control metadata.

Execution validation. Each session must contain a nonempty ordered stage list, unique stage identifiers, relative and resolvable audio paths, an adaptive state-wait stage before every interruption or backchannel injection, and a final same-session QA stage. Question windows must satisfy

$$
0 \leq n _ { \mathrm { s t a r t } } < n _ { \mathrm { s p e e c h - e n d } } \leq n _ { \mathrm { e n d } } .
$$

A full validation pass over all 1,094 sessions reports no hard structural errors.

## A.9 DATASET COMPOSITION

Table 6 summarizes the released corpus.

The public release uses a fixed split of 237 development and 857 test sessions. The split assignment, source text, source waveforms, question order, and oracle annotations are held fixed in the adaptive release. The separate method experiments use session-disjoint training, validation, calibration, and held-out subsets; no persistent session is shared across those subsets.

## A.10 INTENDED USE AND ACCESS RESTRICTIONS

The baseline setting of DuplexLedger evaluates spoken agents through streamed audio events and spoken questions within a persistent session. Source text, annotation timestamps, event roles, gold memory operations, oracle states, evidence links, and reference answers are evaluator-side resources rather than baseline-model inputs. For augmented systems, access by auxiliary modules must be distinguished from access by the speech backbone; ACT-Mem’s observation and conditioning inter faces are described in Appendix E.

Table 6: Composition of DuplexLedger. Fixed-input duration refers to the compatibility waveform; event-source duration is the total duration of the individual audible speech segments used by the adaptive protocol.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Sessions</td><td>1,094</td></tr><tr><td>Chinese / English</td><td>577 / 517</td></tr><tr><td>Interruption / backchannel sessions</td><td>259 /283</td></tr><tr><td>Pause / talking-to-other sessions</td><td>276 /276</td></tr><tr><td>Development / test sessions</td><td>237 / 857</td></tr><tr><td>Memory questions</td><td>6,564</td></tr><tr><td>Latest-value questions</td><td>2,188</td></tr><tr><td>Full-duplex challenges</td><td>3,304</td></tr><tr><td>Audible speech and silence events</td><td>8,239</td></tr><tr><td>Speech-event waveforms</td><td>7,687</td></tr><tr><td>Fixed compatibility audio</td><td>35.328 h</td></tr><tr><td>Event-source audio</td><td>24.221 h</td></tr><tr><td>Session duration, mean / median</td><td>116.25 s / 114.93 s</td></tr><tr><td>Session duration, minimum / maximum</td><td>86.64 s / 176.70 s 7,589</td></tr><tr><td>MFA-aligned speech events</td><td></td></tr><tr><td>Missing event waveforms</td><td>0</td></tr></table>

Access to gold transcripts, annotated speaker/addressee labels, question types, oracle states, or supporting-memory identifiers must be declared for the complete evaluated system, including auxiliary modules. Such access constitutes an annotation-assisted or oracle setting, depending on the information supplied, and must be distinguished from end-to-end audio-only evaluation.

## A.11 LIMITATIONS AND RESPONSIBLE RELEASE

DuplexLedger is fully synthetic and therefore does not reproduce the complete distribution of spontaneous human speech. In particular, TTS speech underrepresents natural disfluency, accent vari ation, emotional prosody, microphone conditions, environmental noise, and unplanned multi-party interaction. The sessions last approximately one to three minutes and do not establish performance on hour-scale or cross-day memory.

Although the task includes selective admission, correction, and distractor rejection, it does not currently contain an explicit memory-deletion operation. It should therefore not be interpreted as a complete benchmark for privacy-aware memory management. The absence of real-user recordings reduces direct privacy risk, but public release still requires documentation of the generation-model and speech-synthesis licenses, the exact generation prompts, construction seeds, source and derived checksums, and all validation reports.

## B ADAPTIVE FULL-DUPLEX EXECUTION PROTOCOL

## B.1 PROTOCOL OBJECTIVE AND EXECUTION UNIT

The evaluation unit is a complete interaction session associated with one DuplexLedger sample. A session contains an annotated context, one or more full-duplex challenges, a recovery phase, and a set of memory questions. The audio content and event annotations are fixed before evaluation, whereas the wall-clock time at which a challenge is delivered is determined online from the decoded assistant waveform. Consequently, the protocol distinguishes between a planned challenge, a trigger-satisfied challenge, and a realized challenge. A planned challenge is an event specified in the sample manifest. A trigger is satisfied only when the required assistant state is observed within the prescribed timeout. A challenge is realized only when the corresponding user audio is subsequently delivered to the model. This separation prevents a silent or non-responsive model from being credited with having successfully experienced an interruption that was never actually applied.

Each sample is evaluated in one persistent model session. The session is reset before the first context event, but it is not reset between context, challenge, recovery, or memory-question stages. In particular, the memory questions are asked after the interaction challenges in the same conversational

Table 7: Execution stages generated from annotated ledger events.
<table><tr><td>Stage</td><td>Purpose</td><td>Execution semantics</td></tr><tr><td>play-audio</td><td>Context delivery</td><td>Plays the bounded audio window associated with a user event and preserves the event&#x27;s speaker, addressee, and dialogue-act metadata.</td></tr><tr><td>wait_for_assistant_ state</td><td>State-triggered synchronization</td><td>Waits for decoded assistant speech to satisfy a target state and a minimum active-speech duration. The stage</td></tr><tr><td>inject_interruptionInterruption challenge</td><td></td><td>terminates on satisfaction or timeout. Delivers the interruption audio only after the preceding</td></tr><tr><td>inject_backchannel Backchannel challenge</td><td></td><td>speaking-state trigger has been satisfied. Delivers a backchannel only after the assistant has</td></tr><tr><td>play-side_talk</td><td>Non-addressee speech</td><td>produced the required amount of active speech. Plays speech addressed to another participant and then observes whether the assistant responds during the</td></tr><tr><td>wait</td><td>Annotated pause or</td><td>observation interval. Advances the scheduler for a fixed duration without</td></tr><tr><td>end_user_event</td><td>observation Event boundary</td><td>fabricating user speech. A pause inside a turn is not treated as a semantic turn boundary. Records the semantic turn boundary and whether an</td></tr><tr><td></td><td></td><td>assistant response is required; it does not reset the session.</td></tr><tr><td>wait_for_assistant_ silence</td><td>Recovery</td><td>Waits for stable assistant silence after a</td></tr><tr><td>run_same_session.</td><td>synchronization</td><td>response-required event.</td></tr><tr><td>memory-qa</td><td>Memory evaluation</td><td>Opens the dynamically timed question-and-answer windows without clearing the preceding conversational</td></tr></table>

state. The default lifecycle is therefore

$$
\mathrm { R E S E T }  \mathrm { C O N T E X T }  \mathrm { C H A L L E N G E / R E C O V E R Y }  \mathrm { S A M E S E S I O N Q A }  \mathrm { F I N A L Z E } .
$$

No reset is performed between individual memory questions.

## B.2 COMPILATION OF LEDGER EVENTS INTO EXECUTION STAGES

The event-level representation is compiled into an ordered stage manifest before model inference. The manifest contains event identifiers, source sample ranges, speaker and addressee labels, dialogue acts, semantic turn boundaries, response requirements, and the expected memory operation. The scheduler executes only this manifest; it does not infer the event ordering from a concatenated waveform at runtime.

For ordinary response-required context events, the manifest inserts a response-start wait followed by a response-silence wait before the next context event. Adaptive interruption and backchannel events instead insert a state-trigger wait immediately before the challenge audio. This ordering ensures that the challenge is aligned to the model’s actual speaking state rather than to a fixed timestamp in the source recording.

## B.3 AUDIO WINDOWS AND CLOCK SEMANTICS

For each speech event, the scheduler stores an actual source range $[ s _ { i } , e _ { i } )$ in samples. The range is derived from the annotated speech window and expanded by a fixed padding of 120 ms, subject to the available source boundaries. The original waveform is never modified in place. A concatenated waveform may be retained as a compatibility artifact, but it is not the authority for adaptive execution; the stage manifest and its sample ranges are authoritative.

The protocol maintains two clocks. Media time measures the amount of user and assistant audio advanced by the scheduler, whereas wall time is measured with a monotonic runtime clock. Adaptive sessions are therefore allowed to have different wall-clock durations for different models. Runtime records preserve both clocks, the decoded assistant VAD state, and the timestamps reported by the native audio bridge. Decisions about whether the assistant is speaking are made from decoded PCM audio rather than packet arrival. Input audio is processed in 80 ms chunks, and the scheduler poll state at 20 ms intervals.

## B.4 ADAPTIVE CHALLENGE TRIGGERING

Interruption and backchannel challenges use a closed-loop trigger:

$$
\mathrm { w a i t f o r ~ a s s i s t a n t ~ s p e e c h } \  \ \mathrm { v e r i f y ~ V A D ~ d u r a t i o n } \  \ \mathrm { i n j e c t ~ c h a l l e n g e ~ a u d i o } .
$$

An interruption requires at least 500 ms of continuous decoded assistant speech, while a backchannel requires at least 800 ms. Both trigger waits have a 10 s assistant-start timeout. If the condition is not satisfied before the timeout, the challenge audio is not sent and the event is recorded as an unrealized challenge with a trigger timeout. In particular, codec activity without VAD-confirmed speech does not satisfy the trigger.

The remaining challenge types use different execution semantics. A pause is implemented as a scheduler wait over its annotated duration and does not force the assistant to speak. Speech addressed to another participant is played as audio and followed by a 1.2 s observation interval. Its structured addressee label is retained in the evaluator’s event record; whether an auxiliary module also accesses this field is a separate input configuration described in Appendix E. Thus, these events test pause intrusion and target-speaker selectivity, respectively, rather than using the same speechtrigger rule as interruption and backchannel events.

After an event whose semantic turn has ended and which requires a response, the scheduler waits for at least 160 ms of assistant speech. It then waits for stable silence for up to 12 s, with a 2 s silence criterion. Failure to observe response onset is recorded as a response-initiation failure; it is not converted into a zero-duration response.

## B.5 RECOVERY AND STOP-LATENCY MEASUREMENT

For an interruption, the assistant’s decoded waveform is monitored from the time at which challenge audio is injected. The runtime record stores whether the assistant was speaking before injection, whether speech continued after injection, the first observed speech offset, and whether the response ended within the configured silence window. Stop latency is computed only when a valid assistant speech interval and a valid interruption injection are both available. If the assistant does not stop before the response timeout, the event remains in the record with an explicit timeout or failure flag.

Backchannels are not treated as interruptions: the protocol records whether the assistant continues, pauses, or restarts after the feedback audio. For non-addressee speech, the observation interval is retained even when the assistant remains silent, since silence is a valid selectivity outcome. All runtime failures are recorded under a “record-and-continue” policy whenever the session remains executable. A server crash, unsupported cancellation, or unrecoverable protocol violation terminates the sample and preserves the partial runtime artifacts.

## B.6 SAME-SESSION MEMORY QUESTIONS

The final stage evaluates memory without resetting the model. Before each question, the controller waits for 1.2 s of stable assistant silence, with a maximum pre-input drainage timeout of 60 s. When supported by the model interface, an outstanding response is cancelled through the native cancellation mechanism without clearing conversational state; the controller then allows an additional 10 s recovery interval before sending the question.

For each question, the execution order is

stable-silence check → question playback → answer-onset wait → answer collection → silence/turn-end closure.

The answer-onset timeout and answer-collection timeout are both 30 s. An answer is closed by a detected turn end, a 2 s silence interval, or the answer timeout. Late speech after a timeout is held in a 10 s quarantine interval and is not retroactively assigned to the expired question. If no answer is observed, the question is marked no response; text from a previous question is never reused.

Table 8: Default timing and execution parameters.
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Input chunk</td><td>80 ms</td><td>Scheduler poll</td><td>20 ms</td></tr><tr><td>Event padding</td><td>120 ms</td><td>Assistant-start timeout</td><td>10s</td></tr><tr><td>Interruption trigger speech</td><td>500 ms</td><td>Backchannel trigger speech</td><td>800 ms</td></tr><tr><td>Minimum response speech</td><td>160 ms</td><td>Assistant-end timeout</td><td>12s</td></tr><tr><td>Post-event silence</td><td>240 ms</td><td>Stable assistant silence</td><td>2s</td></tr><tr><td>Side-talk observation</td><td>1.2s</td><td>QA pre-input quiet</td><td>1.2 s</td></tr><tr><td>QA start timeout</td><td>30s</td><td>QA answer timeout</td><td>30 s</td></tr><tr><td>QA end silence</td><td>2s</td><td>Late-response quarantine</td><td>10s</td></tr><tr><td>Maximum session duration</td><td>900 s</td><td>Maximum input stream RTF</td><td>1.25</td></tr></table>

Memory questions and reference targets are fixed by the planned session. If a speech-triggered challenge is not delivered, these targets are not reconstructed from the realized playback history. Fixed-denominator memory scores therefore measure end-to-end task completion and can reflect both missed evidence delivery and subsequent memory or response errors. They do not isolate retention conditional on receiving the supporting evidence.

## B.7 VALIDITY GATES AND FAILURE ACCOUNTING

The protocol reports validity at multiple levels. First, a session is protocol-valid only if its manifest, audio windows, runtime clocks, and required artifacts pass structural validation. Second, a valid session is behaviorally observable only if decoded assistant audio is available for the relevant decision windows. Third, each planned challenge is labeled as realized, unrealized, or timed out according to its trigger result. These states are not collapsed:

$$
\begin{array} { r l } & { \mathrm { a t t e m p t e d ~ s e s s i o n } } \\ & { \mathrm { p r o t o c o l ~ v a l i d ~ v s . } \mathrm { ~ p r o t o c o l ~ i n v a l i d } } \\ & { \mathrm { b e h a v i o r a l l y ~ o b s e r v a b l e ~ v s . ~ n o t ~ o b s e r v a b l e } } \\ & { \mathrm { c h a l l e n g e ~ r e a l i z e d ~ v s . } \mathrm { ~ u n r e a l i z e d / t i m e o u t . } } \end{array}
$$

Consequently, conditional behavioral metrics are computed only on realized and observable challenges, while challenge-realization and trigger-timeout rates are reported with the planned-challenge denominator. Runtime failures, response-initiation failures, no-response questions, skipped challenge audio, and timeout events remain in the released execution records rather than being silently discarded.

## B.8 REPRODUCIBLE PROTOCOL CONFIGURATION

All parameters in Table 8, including VAD thresholds, trigger conditions, timeout behavior, and session-reset rules, are frozen before evaluation. The runtime output records the effective configuration, stage identifiers, media and wall-clock timestamps, VAD decisions, trigger outcomes, timeout flags, and failure types, thereby making each adaptive execution auditable and reproducible.

## C METRIC DEFINITIONS AND DENOMINATORS

This section defines the aggregate metrics used to evaluate full-duplex interaction and same-session memory behavior. Every metric is first computed independently for each sample and is then aggregated across samples using a macro-average. The resulting values therefore give every valid sample equal weight, regardless of the number of events or the duration of its interaction.

## C.1 AGGREGATION AND MISSING-VALUE POLICY

Let $m _ { i }$ denote the value of metric m for sample $i ,$ and let $\mathcal { I } _ { m }$ be the set of samples for which $m _ { i }$ defined. The reported aggregate is

$$
\overline { { m } } = \frac { 1 } { \left| \mathcal { T } _ { m } \right| } \sum _ { i \in \mathcal { T } _ { m } } m _ { i } .\tag{6}
$$

If $\mathcal { I } _ { m }$ is empty, the metric is reported as unavailable. Missing values are not replaced with zero. An unanswered memory question is not a missing observation: it remains in the fixed question denominator and receives no credit. By contrast, an interaction metric without eligible observations is undefined and is excluded from that metric’s mean. An unrealized challenge remains in the planned-challenge denominator for RCR but does not supply an observation for delivery-conditioned metrics.

All ratios are computed using sample-specific denominators before aggregation. The evaluation therefore does not pool all events, questions, or audio segments across the complete test set. This rule also applies to latency statistics: the median and the 90th percentile are first computed within each sample and are then averaged across samples. Thus, stop latency p50 s and stop latency p90 s are macro-averaged sample-level quantiles, rather than quantiles computed from one pooled latency distribution.

## C.2 RUNTIME VALIDITY AND BEHAVIORAL OBSERVABILITY

Protocol validity and behavioral observability are reported separately. For sample i, protocol validity is defined as

v = 1[runtime timestamps are valid ∧ no runtime failure is recorded ∧ a decoded assistant-audio artifact exists] .

(7)

The aggregate protocol valid is the mean of $v _ { i }$

Behavioral observability is defined as

$$
o _ { i } = \mathbf { 1 } \left[ d _ { i } ^ { \mathrm { a s s t } } > 0 \land n _ { i } ^ { \mathrm { r a w } } > 1 \right] ,\tag{8}
$$

where $d _ { i } ^ { \mathrm { a s s t } }$ is the decoded assistant-speech duration and $n _ { i } ^ { \mathrm { r a w } }$ is the number of samples in the raw assistant track. The aggregate behaviorally observable is the mean of $o _ { i }$ . Consequently, a session can be protocol-valid while remaining behaviorally unobservable if the model produces no usable decoded speech.

## C.3 ADAPTIVE-CHALLENGE METRICS

Let $\mathcal { P } _ { i }$ be the set of distinct challenge identifiers planned in the manifest of sample i, and let $\mathcal { R } _ { i }$ be the subset recorded as realized in the runtime trace. For an interruption or backchannel, realization requires successful audio injection after the corresponding assistant-speaking trigger has been satisfied. For wait-based challenges, realization follows the corresponding runtime observation rule. The sample-level challenge realization rate is

$$
r _ { i } = \frac { \left| \mathcal { R } _ { i } \right| } { \left| \mathcal { P } _ { i } \right| } .\tag{9}
$$

Samples without a planned challenge do not contribute to this ratio.

Let $T _ { i }$ denote the number of challenge-trigger waits that time out before the required assistant state is observed. The trigger-timeout rate is

$$
t _ { i } = \frac { T _ { i } } { \vert \mathcal { P } _ { i } \vert } .\tag{10}
$$

When a trigger times out, the corresponding interruption or backchannel audio is not injected and the challenge is not counted as realized.

The conditional full-duplex score, denoted by $c _ { i } .$ , summarizes behavioral correctness on challenge instances for which the corresponding scenario can be evaluated. The scenario-specific correctness criteria are:

• timely stopping after an interruption;

• continuation of the ongoing response after a backchannel;

• absence of assistant intrusion during an annotated pause; and

• target-speaker selectivity during speech addressed to another participant.

The aggregate conditional score is

$$
\mathsf { c o n d i t i o n a l \_ c f d } = \frac { 1 } { | \mathscr { T } _ { c } | } \sum _ { i \in \mathscr { T } _ { c } } c _ { i } ,\tag{11}
$$

where $\mathcal { T } _ { c }$ contains samples with a defined conditional score. Because this score is conditioned on behaviorally scorable challenges, it must be interpreted together with challenge realization rate, trigger timeout rate, and behaviorally observable.

The aggregate adaptive fd is the macro-average of the corresponding sample-level adaptive full-duplex scores. It is an auxiliary summary that combines challenge opportunity and challenge behavior. It is not recomputed as the product of separately averaged aggregate quantities.

## C.4 ORDINARY-TURN TIMING METRICS

Let $\mathcal { U } _ { i }$ be the set of response-required semantic turns in sample $i ,$ and let $\mathcal { O } _ { i } \subseteq \mathcal { U } _ { i }$ be the subset satisfying the ordinary-turn eligibility conditions. An eligible ordinary turn must have a valid semantic boundary, valid speech-alignment markers, usable native audio-clock metadata, and an input stream within the prescribed real-time limit.

Response coverage is computed over $\textstyle { \mathcal { U } } _ { i } :$

$$
\mathtt { r e s p o n s e \mathrm { - } c o v e r a g e } _ { i } = \frac { \sum _ { u \in \mathcal { U } _ { i } } \mathbf { 1 } [ \mathrm { a s s i s t a n t ~ s p e e c h ~ i s ~ o b s e r v e d ~ f o r } ~ u ] } { | \mathcal { U } _ { i } | } .\tag{12}
$$

The remaining ordinary-turn metrics use $\mathcal { O } _ { i } { : }$

$$
\mathtt { p r e m a t u r e \mathrm { - } o n s e t \mathrm { - } r a t e } _ { i } = \frac { \sum _ { u \in \mathcal { O } _ { i } } \mathbf { 1 } [ \mathrm { a s s i s t a n t o n s e t p r e c e d e s ~ u s e r ~ e n d } ] } { | \mathcal { O } _ { i } | } ,\tag{13}
$$

$$
\mathsf { c a r r y - i n \_ r a t e } _ { i } = \frac { \sum _ { { \boldsymbol u } \in { \mathcal { O } } _ { i } } \mathbf 1 [ \mathrm { a s s i s t a n t i s ~ s p e a k i n g ~ a t ~ i n p u t ~ o n s e t } ] } { | { \boldsymbol \mathcal { O } } \cdot | } ,\tag{14}
$$

$$
\mathsf { o v e r l a p \mathrm { - } b e f o r e \mathrm { - } e n d \mathrm { - } r a t e } _ { i } = \frac { \sum _ { u \in \mathcal { O } _ { i } } \mathbf { 1 } [ \mathrm { a s s i s t a n t \ o v e r l a p s \ u s e r \ s p e e c h \ b e f o r e \ u s e r \ e n d } ] } { | \mathcal { O } _ { i } | } .\tag{15}
$$

For pause handling, let $\mathcal { W } _ { i }$ be the annotated pause-wait events. Pause intrusion is defined as

$$
\mathsf { p a u s e \_ i n t r u s i o n \_ r a t e } _ { i } = \frac { \sum _ { w \in \mathcal { W } _ { i } } \mathbf 1 [ \mathrm { a s s i s t a n t \ s p e e c h \ i s \ o b s e r v e d \ d u r i n g \ } w ] } { | \mathcal { W } _ { i } | } .\tag{16}
$$

These definitions distinguish a new premature response from speech carried over from a previous turn and use only turns with valid timing evidence.

## C.5 INTERRUPTION STOP METRICS

Let $\mathcal { L } _ { i }$ be the set of valid stop latencies for realized interruption events in sample i. A stop latency is included only when the interruption was actually injected and the decoded assistant waveform contains a valid speech-offset measurement. The sample-level latency statistics are

$$
\ell _ { i } ^ { 5 0 } = \mathrm { Q u a n t i l e } _ { 0 . 5 0 } ( \mathcal { L } _ { i } ) ,\tag{17}
$$

$$
\ell _ { i } ^ { 9 0 } = \mathrm { Q u a n t i l e } _ { 0 . 9 0 } ( \mathcal { L } _ { i } ) ,\tag{18}
$$

$$
s _ { i } ^ { 8 0 0 } = \frac { \sum _ { \ell \in \mathcal { L } _ { i } } \mathbf { 1 } [ \ell \le 0 . 8 \mathrm { { s } } ] } { \left| \mathcal { L } _ { i } \right| } .\tag{19}
$$

Table 9: Aggregate runtime metrics and their sample-level denominators.
<table><tr><td>Metric</td><td>Direction</td><td>Denominator or interpretation</td></tr><tr><td>protocol_valid</td><td>→→→→↓→→</td><td>Attempted samples.</td></tr><tr><td>behaviorally_observable</td><td></td><td>Attempted samples; requires usable decoded assistant speech.</td></tr><tr><td>challenge_realization_rate</td><td></td><td>Distinct planned challenge identifiers.</td></tr><tr><td>conditional_cfd</td><td></td><td>Challenge instances eligible for conditional behavioral scoring.</td></tr><tr><td>adaptive_fd</td><td></td><td>Sample-level adaptive full-duplex score; auxiliary summary.</td></tr><tr><td>trigger_timeout_rate</td><td></td><td>Distinct planned challenges.</td></tr><tr><td>response_coverage</td><td></td><td>Response-required semantic turns.</td></tr><tr><td>premature_onset_rate</td><td></td><td>Eligible ordinary response turns.</td></tr><tr><td>carry_in_rate</td><td></td><td>Eligible ordinary response turns.</td></tr><tr><td>overlap_before_end_rate</td><td></td><td>Eligible ordinary response turns.</td></tr><tr><td>pause_intrusion_rate</td><td></td><td>Annotated pause-wait events.</td></tr><tr><td>stop-latency-p50_s</td><td></td><td>Valid interruption stop latencies, summarized within each sample.</td></tr><tr><td>stop-latency-p90_s</td><td></td><td>Valid interruption stop latencies, summarized within each sample.</td></tr><tr><td>stop_success_at_800ms</td><td></td><td>Valid interruption stop-latency observations.</td></tr><tr><td>assistant_speech_duration_s</td><td></td><td>Total decoded assistant-speech duration; descriptive only.</td></tr></table>

† Conditional or auxiliary measures that must be interpreted with challenge realization, trigger timeout, and observability statistics.

The macro-averages of these quantities are reported as stop latency p50 s, stop latency p90 s, and stop success at 800ms, respectively. Samples without a valid stop-latency observation are excluded from the corresponding aggregate and are not assigned a latency of zero.

The metric assistant speech duration s is the macro-average of the total decoded assistant-speech duration per sample. It is a descriptive quantity rather than a quality measure.

## C.6 MEMORY-ANSWER NORMALIZATION

Memory answers are obtained from the transcript associated with each question-specific answer window. When available, the frozen transcript associated with the answer window is used; otherwise, the corresponding answer artifact is used. Text is normalized with Unicode NFKC normalization, lowercasing, and removal of punctuation and symbols. English text is tokenized by whitespacedelimited words, while Chinese text is tokenized into CJK characters and alphanumeric units.

For token sequences $A$ and $B ,$ multiset precision and recall are

$$
P ( A , B ) = \frac { | A \cap B | } { | B | } , \qquad R ( A , B ) = \frac { | A \cap B | } { | A | } ,\tag{20}
$$

and the token overlap score is

$$
F _ { 1 } ( A , B ) = \frac { 2 P ( A , B ) R ( A , B ) } { P ( A , B ) + R ( A , B ) } .\tag{21}
$$

A value is considered present when its normalized form is an exact substring of the answer or when token $F _ { 1 }$ reaches 0.80 for values of at most two tokens and 0.60 for longer values.

## C.7 MEMORY QA METRICS

Let $\mathcal { Q } _ { i }$ be all memory questions in sample $i ,$ and let $\mathcal { Q } _ { i } ^ { + }$ be the subset with a non-empty response that is not marked as no response. Response and no-response rates are

$$
\mathtt { r e s p o n s e \_ r a t e } _ { i } = \frac { | \mathcal { Q } _ { i } ^ { + } | } { | \mathcal { Q } _ { i } | } ,\tag{22}
$$

$$
\cap \circ \mathrm { - r } \mathsf { e s p o n s e \mathrm { - } r a t e } _ { i } = 1 - { \mathrm { r e s p o n s e \mathrm { - } r a t e } } _ { i } .\tag{23}
$$

Let $\mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } }$ denote the latest-value questions. For question $q ,$ let $a _ { q }$ indicate that an accepted updated value is present and let $f _ { q }$ indicate that a forbidden superseded value is present. Then

$$
\mathtt { l a t e s t \_ v a l u e \_ a c c u r a c y } _ { i } = \frac { \sum _ { \boldsymbol { q } \in \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } } \mathbf { 1 } [ \boldsymbol { q } \mathrm { \ a n s w e r e d } \wedge \boldsymbol { a } _ { \boldsymbol { q } } \wedge \neg \boldsymbol { f } _ { \boldsymbol { q } } ] } { | \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } | } ,\tag{24}
$$

$$
\mathsf { s u p e r s e d e d \mathrm { _ { - } v a l u e \mathrm { _ { - } e r r o r } } } _ { i } = \frac { \sum _ { q \in \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } } \mathbf { 1 } [ q \mathrm { ~ a n s w e r e d } \wedge f _ { q } ] } { | \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } | } .\tag{25}
$$

Table 10: Aggregate same-session memory metrics and their sample-level denominators.  
Metric Direction Denominator or interpretation   
response rate All memory questions.   
no response rate All memory questions; complementary to response rate.   
latest value accuracy All latest-value questions, including unanswered questions.   
superseded value error All latest-value questions.   
distractor adoption rate All distractor-ignore questions.   
distractor rejection accuracy All distractor-ignore questions; rejection additionally requires a response.   
summary fact coverage Atomic reference facts in final-summary questions.   
answer accuracy given response Answered latest-value questions only.   
legacy token f1 All memory questions; retained as a compatibility measure.

Accuracy conditioned on producing an answer is

$$
\mathsf { a n s w e r \_ a c c u r a c y - g i v e n \_ r e s p o n s e } _ { i } = \frac { \sum _ { q \in \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } \cap \mathcal { Q } _ { i } ^ { + } } \mathbf { 1 } [ a _ { q } \wedge \neg f _ { q } ] } { | \mathcal { Q } _ { i } ^ { \mathrm { l a t e s t } } \cap \mathcal { Q } _ { i } ^ { + } | } .\tag{26}
$$

This metric separates incorrect answers from failures to answer.

Let $\mathcal { Q } _ { i } ^ { \mathrm { d i s t } }$ denote the distractor-ignore questions. A distractor is considered adopted when the answer is non-empty and does not explicitly reject the distractor, or when the answer directly matches distractor content without rejection. A distractor is considered rejected only when the model responds and the adoption condition is false:

$$
\begin{array} { r }  \mathrm { \normalfont ~ d i s t r a c t o r \_ a d o p t i o n \_ r a t e } _ { i } = \frac { \sum _ { q \in \mathcal { Q } _ { i } ^ { \mathrm { d i s t } } } \mathbf 1 [ \mathrm { d i s t r a c t o r \ a d o p t e d } ] } { | \mathcal { Q } _ { i } ^ { \mathrm { d i s t } } | } , \qquad ( 2 7 \cdot \frac  \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^ { \infty } \sum _ { \ell = 1 } ^  \end{array}\tag{28}
$$

These two measures are reported together with response rate, since silence can reduce distractor adoption without demonstrating correct rejection.

For a final-summary question, the reference summary is split into atomic facts using sentence-final punctuation, semicolons, question marks, and line breaks. A fact is considered covered when its token $F _ { 1 }$ with the answer is at least 0.50. If $\mathcal { F } _ { q }$ is the reference fact set for question q, then

$$
\mathrm { c o v e r a g e } ( q ) = \frac { \sum _ { f \in \mathcal { F } _ { q } } \mathbf { 1 } [ F _ { 1 } ( f , \widehat { a } _ { q } ) \geq 0 . 5 0 ] } { | \mathcal { F } _ { q } | } .\tag{29}
$$

The sample-level summary fact coverage is the mean coverage over final-summary questions.

The metric legacy token f1 is the mean token $F _ { 1 }$ between each answer and its full reference answer. It is retained for compatibility with earlier result tables, but it is not used as the sole measure of memory quality.

## C.8 DENOMINATOR REPORTING AND INTERPRETATION

For each evaluated model, the results report the total number of attempted samples, the number of protocol-valid samples, and the number of behaviorally observable samples. The total number of memory questions and the number of latest-value questions are also reported. Challenge, ordinary-turn, pause, and stop-latency denominators are retained in the sample-level records used for aggregation.

Conditional metrics must always be reported together with their coverage variables. In particular, conditional cfd is accompanied by challenge realization and trigger-timeout rates; stoplatency measures are accompanied by the number of valid latency observations; and conditional memory accuracy is accompanied by response rate. A low error rate from a nearly silent model is therefore not interpreted as evidence of strong full-duplex or memory behavior. Stratified analyses by scenario or language reuse the definitions above and do not introduce additional metrics.

Table 11: Native model interfaces and model-specific inference settings.
<table><tr><td>Model</td><td>Native interface</td><td>Input / out- put</td><td>Chunk or inference window</td><td>Model-specific condi- tion</td><td>GPU</td></tr><tr><td>MiniCPM-o 4.5</td><td>Native streaming worker</td><td>16/24 kHz</td><td>80 ms input; 1000 ms native window</td><td>Greedy decoding, fixed seed, fixed reference au- dio</td><td>1</td></tr><tr><td>Freeze-Omni</td><td>Socket.IO-based streaming service</td><td>16/24 kHz</td><td>20 ms native input block</td><td>p = 0, k = 1, temper- ature = 1.0; neutral sys- tem prompt</td><td>1</td></tr><tr><td>Moshi</td><td>WebSocket stream- ing service</td><td>24/24 kHz</td><td>80 ms input block</td><td>Native server decoding; no evaluator-side task- specific sampling over- ride</td><td>1</td></tr><tr><td>Lychee-FD</td><td>vLLM-backed streaming service</td><td>16/24 kHz</td><td>80 ms input; 400 ms native inference win- dow</td><td>Fixed prompt voice; na- tive backend decoding</td><td>2</td></tr><tr><td>PersonaPlex- 7B-v1</td><td>Native WebSocket streaming service</td><td>24/24 kHz</td><td>80 ms input block</td><td>Fixed voice prompt and fixed random seed</td><td>1</td></tr></table>

## D BASELINE MODELS AND INFERENCE CONFIGURATIONS

## D.1 EVALUATION SCOPE AND COMPARISON PROTOCOL

We evaluate five publicly available full-duplex speech models: MiniCPM-o 4.5 (Cui et al., 2026), Freeze-Omni (Wang et al., 2025c), Moshi (Defossez et al. ´ , 2024), Lychee-FD (Liu et al., 2026), and PersonaPlex-7B-v1 (Roy et al., 2026). The models are selected to cover different streaming speech architectures and native interaction interfaces. All baseline models use the same session manifests, planned event order, source audio clips, and scheduling rules. Because speech triggered challenges depend on each model’s realized output, the subset of challenge clips actually delivered can differ across models. A sample is executed as one persistent session: the session is reset before the sample begins, but not between context events, full-duplex challenges, recovery stages, or memory questions.

The baseline speech models are not provided with gold memory states, reference answers, alignment boundaries, event labels, addressee labels, or future events. This restriction concerns baselinemodel inputs; the interfaces available to ACT-Mem’s auxiliary modules are specified separately in Appendix E. All models are evaluated with the same adaptive execution controller, the same challenge-triggering rules, the same session-reset policy, and the same answer-attribution procedure. Model-specific native chunk sizes, sampling rates, voice prompts, and transport protocols are retained when they are required by the corresponding implementation. These interface-specific parameters are reported explicitly rather than hidden under a common configuration.

## D.2 NATIVE INTERFACES AND MODEL CONFIGURATIONS

Table 11 summarizes the model-specific settings used for inference. Input and output sample rates refer to the native model bridge; the evaluator maintains a shared reference clock and performs any required resampling. A native inference window is distinct from the evaluator’s event window and does not change the logical event sequence.

MiniCPM-o 4.5 uses greedy decoding with the frozen evaluation seed and a fixed reference audio prompt. Freeze-Omni uses the native service configuration with $p = 0 , k = 1$ , and temperature 1.0. Moshi is evaluated through its native streaming service without introducing an additional task-specific decoding rule. Lychee-FD uses the vLLM-backed native inference path with a fixed prompt voice and a 400 ms native inference window. PersonaPlex uses a fixed voice prompt and the evaluation seed for all sessions.

The native interfaces do not expose identical user-turn controls. Freeze-Omni supports an explicit response-boundary operation, whereas the other bridges record local event boundaries when an equivalent native message is unavailable. These local markers advance the evaluator’s state machine but do not reset the model or claim that the model received an explicit semantic end-of-turn signal. This distinction is retained in the runtime artifacts.

Table 12: Shared evaluator-side inference and timing configuration.
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Reference input chunk</td><td>80 ms</td></tr><tr><td>Evaluator polling interval</td><td>20 ms</td></tr><tr><td>Event-window padding</td><td>120 ms</td></tr><tr><td>Assistant-start timeout</td><td>10 s</td></tr><tr><td>Assistant-end timeout</td><td>12 s</td></tr><tr><td>Interruption trigger speech</td><td>500 ms</td></tr><tr><td>Backchannel trigger speech</td><td>800 ms</td></tr><tr><td>Minimum response speech</td><td>160 ms</td></tr><tr><td>Stable assistant silence</td><td>2 s</td></tr><tr><td>Side-talk observation window</td><td>1.2s</td></tr><tr><td>QA start timeout</td><td>30 s</td></tr><tr><td>QA answer timeout</td><td>30 s</td></tr><tr><td>QA pre-input quiet period</td><td>1.2s</td></tr><tr><td>QA pre-input drainage timeout</td><td>60 s</td></tr><tr><td>Late-response quarantine</td><td>10 s</td></tr><tr><td>Maximum session duration</td><td>900 s</td></tr></table>

## D.3 SHARED EVALUATOR-SIDE CONFIGURATION

All models are evaluated with a common evaluator-side configuration. User audio is streamed in real time as bounded event clips. Each delivered clip is played once, is not accelerated, and is not truncated because of the assistant’s current output. Challenges whose trigger conditions are not met are omitted as specified below. The controller uses an 80 ms reference input chunk, a 20 ms polling interval, and a maximum input real-time factor of 1.25. Native model sample rates and chunk size are permitted, but all timing decisions are mapped to the shared reference clock.

Assistant speaking state is determined from decoded assistant PCM using the shared waveform-level VAD. Packet arrival, codec activity, and text-token activity are not treated as evidence of speech. Interruption audio is injected only after at least 500 ms of continuous decoded assistant speech, and backchannel audio is injected only after at least 800 ms of such speech. If the trigger condition is not satisfied within the 10 s trigger window, the challenge is recorded as unrealized and its audio is not injected.

The common protocol parameters are listed in Table 12. These parameters are frozen before evalu ation and are not tuned on held-out sessions.

## D.4 PROMPT AND INFORMATION ISOLATION

All baseline models receive a neutral voice-assistant instruction in the language of the corresponding sample. The default English instruction is “You are a voice assistant,” with the corresponding neutral Chinese instruction for Chinese samples. The instruction is transported through the native interface when required by the model, but no task-specific memory or interaction instruction is added.

The following information is not exposed to any baseline model:

• gold ledger states or future memory updates;

• event types, challenge identifiers, or addressee labels;

• MFA word boundaries or evaluator-side VAD decisions;

• reference answers and expected memory values; and

• assistant reference audio generated from the benchmark annotations.

A model may therefore use only its native audio history and the neutral instruction supplied at session initialization. The proposed memory module, when evaluated in the controlled comparison, uses the same frozen MiniCPM-o 4.5 backbone and the same audio protocol; its additional memory conditioning is applied only through the method-specific inference path described in the main paper and the implementation appendix.

## D.5 AUDIO OUTPUT, TRANSCRIPTION, AND ANSWER ATTRIBUTION

All assistant audio is captured from the native model bridge and retained as a decoded waveform. The waveform is used for VAD-based state detection, challenge triggering, stop-latency measurement, and response-onset attribution. Memory answers are evaluated from independent answer windows rather than from a concatenated session transcript.

For cross-model textual comparison, each answer waveform is transcribed with the same frozen external Whisper-small configuration. The language is taken from the sample manifest, decoding temperature is set to zero, and the same transcription backend is used for all models. Native model text, when available, is retained as a secondary artifact but is not used as the primary cross-model transcript. A transcription failure is recorded explicitly and makes the corresponding primary textual score unavailable; it is not silently converted to a zero-valued answer.

The answer window opens after decoded assistant speech onset and closes on decoded silence, an explicit turn boundary, or the configured answer timeout. A late response after timeout is kept in a quarantine interval and is not assigned retroactively to the expired question. No answer text is reused between questions.

## D.6 CALIBRATION, TEST ISOLATION, AND REPRODUCIBILITY

Before the held-out evaluation, a fixed calibration split is used to freeze the shared VAD, challengetrigger thresholds, timeout values, text normalization rules, and scoring parameters. The held-out sessions are not used for prompt tuning, decoding-parameter tuning, VAD adjustment, timeout selection, or checkpoint selection.

For every model, the evaluation record stores the public model identifier, checkpoint identity, repository revision when available, model-profile configuration, prompt transport, decoding settings, random seed, native sample-rate configuration, assigned GPU resources, and runtime environment identity. Checkpoint and configuration hashes are retained in the artifact manifest so that the reported results can be traced to an unambiguous model state without exposing machine-specific filesystem paths in the paper.

GPU isolation is enforced during inference. A run is considered invalid for paper reporting if an unrelated process contaminates an assigned GPU. The same session and sample identifiers are preserved across all runtime artifacts, allowing failed samples and resumed samples to be audited without double counting.

## D.7 FAILURE HANDLING AND EFFECTIVE DENOMINATORS

Runtime failures, server disconnects, unsupported native controls, trigger timeouts, answer timeouts, and no-response questions are retained in the execution record. A failed sample is not silently removed from the attempted sample count. Instead, the evaluation reports protocol validity, behavioral observability, completion or failure status, and the effective denominator of every metric.

A challenge that is not realized because its trigger condition is not reached is distinguished from a challenge that is realized but handled incorrectly. Likewise, a valid session with no assistant speech is distinguished from a protocol-invalid session. Conditional behavioral scores are computed only when their required audio and timing evidence exists, while challenge realization, trigger timeout, and failure counts are reported separately. This accounting prevents differences in interface robustness, speech observability, and challenge exposure from being mistaken for differences in full-duplex behavior or memory fidelity.

## E ACT-MEM IMPLEMENTATION DETAILS

## E.1 IMPLEMENTATION SCOPE AND FROZEN COMPONENTS

ACT-Mem is implemented as an external memory module coupled to a frozen MiniCPM-o 4.5 full-duplex backbone (Cui et al., 2026). The backbone, including its acoustic encoder, streaming decoder, voice-generation components, and native full-duplex session manager, remains frozen

throughout training and evaluation. A frozen speech transcription and timing component provides event-aligned textual observations to the memory controller. The trainable part consists of an event controller and a typed retriever, which jointly contain 8.92M parameters.

The online computation is organized into three stages:

$$
\mathrm { { e v e n t  o b s e r v a t i o n } \longrightarrow \ t r a n s a c t i o n a l \ m e m o r y \ u p d a t e \longrightarrow \ t y p e d Q A \ r e t r i e v a l . }\tag{30}
$$

Context and full-duplex challenge events update the external memory state. During memory $\mathrm { Q A } .$ the retriever constructs a compact evidence card from the active state, and the card is provided to the frozen backbone through the native text-prefill interface. The full-duplex audio stream and the memory controller therefore share the same persistent session.

## E.2 EVENT CONTROLLER

For each delivered event $e _ { t } ,$ a frozen ASR component transcribes the evaluator-delimited audio segment. The scheduler supplies the event index and speaker/addressee metadata. These metadata are distinct from the controller’s learned predictions: a non-user or non-addressed event can veto admission through an explicit gate. The adapter uses the default audibility flag rather than supplying a separate online audibility estimate.

A lightweight text encoder and recurrent state module process the transcript and chronological context. Four prediction heads produce the operation, addressee, semantic-completeness, and confidence estimates:

$$
\left( \widehat { o } _ { t } , \widehat { a } _ { t } , \widehat { c } _ { t } , p _ { t } ^ { \mathrm { c o n f } } \right) = f _ { \theta } \left( x _ { t } , h _ { t - 1 } \right) ,\tag{31}
$$

where

$$
\begin{array} { r } { \widehat { o } _ { t } \in \{ \mathsf { W R I T E } , \mathsf { U P D A T E } , \mathsf { I G N O R E } , \mathsf { R E T A I N } \} , \qquad \widehat { a } _ { t } \in \{ \mathrm { a s s i s t a n t } , \mathrm { o t h e r } \} , \qquad \widehat { c } _ { t } \in \{ 0 , 1 \} . } \end{array}
$$

The controller proposes atomic entity–value slots from each event. Each slot stores an entity key, value, event identifier, chronological event index, operation type, confidence, source role, and semantic-completeness flag. The controller output is converted into an admission score

$$
g _ { t } = p _ { t } ^ { \mathrm { a d d r } } p _ { t } ^ { \mathrm { c o m p l e t e } } \left[ p _ { t } ( \mathbf { W } \mathbf { R } \mathrm { I } \mathrm { T E } ) + p _ { t } ( \mathbf { U } \mathbf { P } \mathbf { D } \mathbf { A } \mathrm { T E } ) \right] c _ { t } ,\tag{32}
$$

where $p _ { t } ^ { \mathrm { a d d r } } , p _ { t } ^ { \mathrm { c o m p l e t e } }$ , and $c _ { t }$ denote the predicted addressee probability, completeness probability, and controller confidence, respectively.

A proposal is processed through the following transaction states:

$$
\mathrm { P R E P A R E }  \{ \begin{array} { l l } { \mathrm { C O M M I T } , } & { g _ { t } \geq \tau _ { \mathrm { w r i t e } } , } \\ { \mathrm { A B O R T } , } & { \widehat { a } _ { t } = \mathrm { o t h e r ~ o r ~ t h e ~ e v e n t ~ i s ~ i n a d m i s s i b l e } , } \\ { \mathrm { U N C E R T A I N } , } & { \mathrm { o t h e r w i s e } . } \end{array} 
$$

The calibration split selects $\tau _ { \mathrm { w r i t e } } = 0 . 3 5$ . Committed proposals are passed to the versioned memory store, while aborted and uncertain proposals remain unavailable to ordinary retrieval.

## E.3 VERSIONED TRANSACTIONAL LEDGER

The memory store maintains active values, superseded versions, uncertain proposals, and provenance metadata. Each committed item is represented as

$$
m _ { j } = ( k _ { j } , v _ { j } , \tau _ { j } , \iota _ { j } , \rho _ { j } , \gamma _ { j } , \sigma _ { j } ) ,
$$

where $k _ { j }$ is the entity key, $v _ { j }$ is the value, $\tau _ { j }$ is the source-event index, $\iota _ { j }$ is the source event identifier, $\rho _ { j }$ is the operation label, $\gamma _ { j }$ is the confidence, and $\sigma _ { j } \in \{ a c t i \nu e$ , superseded, uncertain} is the state status.

A WRITE operation creates a new active item. An UPDATE operation creates a new active version and supersedes the previous active value associated with the same entity key:

$$
k _ { j } = k _ { i } , \qquad \tau _ { j } > \tau _ { i } , \qquad \rho _ { j } = \mathrm { U P D A T E } .\tag{33}
$$

Table 13: Question routing and retrieval scope.
<table><tr><td>Question type</td><td>Retrieval behavior</td><td>Maximum items</td></tr><tr><td>Latest-value QA</td><td>Selects the target entity&#x27;s current active value after ranking candidate facts.</td><td>2 candidates, 1 final value</td></tr><tr><td>Ordinary memory QA</td><td>Retrieves active facts relevant to the question.</td><td>4</td></tr><tr><td>Final-summary QA</td><td>Aggregates active facts across entities and memory types.</td><td>8</td></tr><tr><td>Distractor-ignore QA</td><td>Routes directly to rejection behavior without memory retrieval.</td><td>0</td></tr></table>

The predecessor is moved to the superseded store and linked through supersedes and superseded by fields. A WRITE or UPDATE proposal whose admission score is below the threshold is retained as uncertain. IGNORE records an aborted proposal, whereas RETAIN preserves the currently active state.

The ledger has a maximum active capacity of 32 slots. When the capacity is exceeded, lowerconfidence and older active items are moved to the uncertain state while supersession links remain available for audit. Retrieval uses active items by default; superseded and uncertain items are retained for diagnostic analysis and error attribution.

## E.4 TYPED RETRIEVAL AND EVIDENCE CARDS

The adapter supplies a query type to the retriever. In the provided implementation, an annotated type is loaded from the configured QA metadata when available; otherwise, a deterministic rule routes the ASR-transcribed question. These annotation-assisted and text-routed modes must be distinguished when interpreting results. The query-type set is:

K = {latest value qa, ordinary memory qa, final summary qa, distractor ignore}.

The retrieval scope and maximum number of returned items depend on this type.

For a question q and memory item $m ,$ the retriever combines representation similarity, lexical overlap, recency, confidence, and state-based penalties:

$$
\begin{array} { r } { S ( q , m ) = S _ { \mathrm { d e n s e } } ( q , m ) + \alpha _ { \mathrm { l e x } } S _ { \mathrm { l e x } } ( q , m ) + \alpha _ { \mathrm { c o n f } } \gamma _ { m } + \alpha _ { \mathrm { r e c } } S _ { \mathrm { r e c } } ( q , m ) - \alpha _ { \mathrm { b a d } } B ( m ) , } \end{array}\tag{34}
$$

where B(m) penalizes superseded, uncertain, or distractor items. The retriever ranks candidate items by $S ( q , m )$ and applies a score-and-margin gate. An evidence card is generated only when the retrieved evidence passes the gate.

The evidence card contains the question type, entity keys, active values, source event identifiers, and a compact answering instruction:

$$
C _ { q } = { \mathrm { S e r i a l i z e } } \left( q , \{ m _ { j } : m _ { j } \in { \mathcal { R } } ( q ) \} \right) .\tag{35}
$$

The card is provided once at the corresponding QA boundary through the backbone’s native textprefill interface. Context events and full-duplex challenge events continue through the native audio path, while memory conditioning is applied only to the associated same-session question.

The evidence-card intervention begins at the corresponding QA boundary and does not directly alter preceding interaction events. Cross-variant differences in pre-QA interaction metrics are therefore descriptive whole-run differences, rather than isolated causal effects of retrieval or ledger maintenance.

## E.5 TRAINING OBJECTIVES

Training uses session-disjoint event and QA records. The controller is trained to predict operation, addressee, completeness, and confidence:

$$
\mathcal { L } _ { \mathrm { c t r l } } = \mathcal { L } _ { \mathrm { o p } } ^ { \mathrm { w C E } } + 0 . 3 \mathcal { L } _ { \mathrm { a d d r } } ^ { \mathrm { C E } } + 0 . 3 \mathcal { L } _ { \mathrm { c o m p l e t e } } ^ { \mathrm { C E } } + 0 . 2 \mathcal { L } _ { \mathrm { c o n f } } ^ { \mathrm { B C E } } .\tag{36}
$$

The weighted operation loss uses weights (1.0, 1.5, 1.4, 0.8) for WRITE, UPDATE, IGNORE, and RETAIN, respectively.

The retriever is trained with binary relevance supervision over active, superseded, forbidden, hardnegative, and distractor memory items:

$$
\mathcal { L } _ { \mathrm { r e t r } } = \mathcal { L } _ { \mathrm { B C E } } ( \mathbf { s } , \mathbf { y } ) + \frac { 0 . 5 } { | \mathcal { P } | | \mathcal { N } | } \sum _ { p \in \mathcal { P } } \sum _ { n \in N } \left[ 0 . 2 5 - s _ { p } + s _ { n } \right] _ { + } ,\tag{37}
$$

where s denotes candidate scores, y denotes binary relevance labels, and P and N denote positive and negative candidate sets.

The total objective is

$$
\mathcal { L } _ { \mathrm { A C T - M e m } } = \mathcal { L } _ { \mathrm { c t r l } } + \mathcal { L } _ { \mathrm { r e t r } } .\tag{38}
$$

## E.6 OPTIMIZATION AND DATA SPLITS

The training data are divided by session into train, validation, calibration, and held-out partitions containing 660, 175, 40, and 219 sessions, respectively. The controller and retriever are jointly optimized in the first stage using AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ , batch size 32, ten training epochs, weight decay 0.01, gradient-norm clipping at 1.0, and random seed 20260907.

In the second stage, the controller is fixed and the retriever is adapted with learning rate $1 \times 1 0 ^ { - 4 }$ batch size 32, three adaptation epochs, and random seed 20260908. Checkpoints are selected using only the training and validation partitions. The 40-session calibration split is used for threshold selection without gradient updates. Held-out sessions are reserved for the final comparison and are not used for parameter, checkpoint, or threshold selection.

Training is performed on eight NVIDIA A100 GPUs. The MiniCPM-o 4.5 backbone and the speech transcription and timing components remain frozen during both optimization stages.

## E.7 CALIBRATION AND SAME-SESSION INFERENCE

Calibration evaluates candidate admission thresholds on the calibration sessions and selects the threshold that maximizes memory-eligibility F1. The selected threshold is then fixed for held-out inference. During a test session, the controller processes events in chronological order and updates the ledger after each event. The retriever is invoked only when a memory question reaches the stable-silence boundary.

The inference sequence for a memory question is

$$
{ \begin{array} { r l } & { { \mathrm { s t a b l e ~ a s s i s t a n t ~ s i l e n c e } } \to { \mathrm { q u e s t i o n ~ a u d i o } } \to { \mathrm { t y p e d ~ r e t r i e v a l } } } \\ & { \qquad \to { \mathrm { e v i d e n c e - c a r d ~ c o n s t r u c t i o n } } \to { \mathrm { n a t i v e ~ t e x t ~ p r e f i l l } } \to { \mathrm { s p o k e n ~ a n s w e r . } } } \end{array} }\tag{39}
$$

The question audio and the evidence card are provided within the same persistent backbone session. Each question receives an independently constructed card, and the resulting answer is attributed to its own QA window.

## E.8 RUNTIME ARTIFACTS AND AUDIT RECORDS

For every evaluated session, ACT-Mem records the controller trace, ledger snapshot, retrieved memory identifiers, superseded identifiers, query type, retrieval scores, gate decision, evidence card, answer transcript, response/no-response status, and runtime latency. Each event trace contains the event identifier, event index, predicted operation, addressee, completeness, confidence, admission score, created slots, updated slots, superseded slots, and uncertain or aborted slots.

These records provide an auditable correspondence between an input event, its predicted memory operation, the resulting versioned state, the evidence shown at QA time, and the final spoken answer. They also preserve the effective session and question denominators used for the held-out evaluation.

Table 14: Training configuration.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Session split</td><td>660/175/40/219 train/validation/calibration/held-out</td></tr><tr><td>Trainable parameters</td><td>8.92M</td></tr><tr><td>Controller parameters</td><td>4.59M</td></tr><tr><td>Retriever parameters</td><td>4.33M</td></tr><tr><td>Optimization stages</td><td>Joint training followed by retriever adaptation</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Batch size</td><td>32</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td>Joint-stage learning rate and epochs</td><td> $3 \times { 1 0 } ^ { - 4 } .$  10 epochs</td></tr><tr><td>Adaptation-stage learning rate and epochs</td><td> $1 \times { 1 0 } ^ { - 4 } .$  3 epochs</td></tr><tr><td>Training hardware</td><td>8 NVIDIA A100 GPUs</td></tr></table>

## F TRAINING, CALIBRATION, AND COMPUTATIONAL COST

## F.1 TRAINING SETUP

ACT-Mem is trained with session-disjoint data partitions containing 660 training sessions, 175 validation sessions, 40 calibration sessions, and 219 held-out sessions. The MiniCPM-o 4.5 full-duplex backbone and the speech transcription and timing components remain frozen. Only the event controller and typed retriever are optimized, resulting in 8.92M trainable parameters: 4.59M in the controller and 4.33M in the retriever.

Training consists of two stages. The first stage jointly optimizes the event controller and the typed retriever. The second stage initializes from the joint checkpoint, freezes the event controller, and further adapts the typed retriever using session-level memory evidence. Optimization uses AdamW with weight decay 0.01, batch size 32, and gradient-norm clipping at 1.0. The joint stage uses learning rate $3 \times 1 0 ^ { - 4 }$ , ten epochs, and random seed 20260907; the adaptation stage uses learning rate $1 \times 1 0 ^ { - 4 }$ , three epochs, and random seed 20260908. Training is performed on eight NVIDIA A100 GPUs.

All supervision is organized by session, and the held-out sessions are excluded from training, checkpoint selection, and calibration. The final checkpoint is loaded jointly with the frozen speech backbone during same-session evaluation.

## F.2 CALIBRATION PROTOCOL

Calibration is performed on the dedicated 40-session calibration split without gradient updates. We search over candidate admission thresholds and select the value that maximizes event-memory eligibility F1. The resulting write threshold is

$$
\tau _ { \mathrm { w r i t e } } = 0 . 3 5 .
$$

After calibration, the threshold is fixed for held-out inference. The calibration record stores the candidate thresholds, the selected value, the calibration split identifier, and the associated checkpoint identity.

## F.3 COMPUTATIONAL COST

The computational cost reported for ACT-Mem is incremental relative to the frozen MiniCPM-o 4.5 backbone. The additional trainable module contains 8.92M parameters, divided between the event controller and the typed retriever as shown in Table 14. During inference, the controller performs one chronological state update for each observed event, and the retriever constructs at most one evidence card for each memory question. The resulting card is inserted through the native questionconditioning path while the full-duplex backbone remains unchanged.

Runtime artifacts record the loaded checkpoint, effective calibration threshold, controller traces, retrieved memory identifiers, evidence cards, answer windows, and per-session failures. These records provide the basis for reporting the incremental memory-module cost separately from the fixed cost of full-duplex backbone inference.

## G STATISTICAL ANALYSIS

## G.1 UNIT OF ANALYSIS AND AGGREGATION

The session is the primary statistical unit. The six memory questions within a session, as well as the interaction events belonging to that session, are first aggregated into session-level quantities. Overall results are then computed by arithmetic macro-averaging over sessions:

$$
\overline { { m } } = \frac { 1 } { \left| \mathcal { T } _ { m } \right| } \sum _ { i \in \mathcal { T } _ { m } } m _ { i } ,\tag{40}
$$

where $m _ { i }$ is the value of metric m for session $i ,$ and $\mathcal { I } _ { m }$ is the set of sessions with a valid value for that metric.

Questions from the same session are not treated as independent statistical replicates. Failure handling is metric-specific. Unanswered memory questions receive no credit under fixed question de nominators, and unrealized challenges remain in the planned-challenge denominator for RCR. Interaction metrics use their eligible-event denominators. A session with no valid denominator for a metric is excluded from that metric-specific mean, with the effective denominator reported; undefined values are not replaced by zero.

## G.2 PAIRED SESSION BOOTSTRAP

The controlled comparison between ACT-Mem and the frozen MiniCPM-o 4.5 backbone uses the same 219 held-out sessions for both systems. We therefore use a paired percentile bootstrap at the session level. For each bootstrap replicate b, 219 session identifiers are sampled with replacement. The outputs of both systems for each sampled identifier are kept paired, and the session-level metric is recomputed before aggregation.

For metric m, the paired effect is

$$
\Delta _ { m } = \overline { { { m } } } _ { \mathrm { A C T - M e m } } - \overline { { { m } } } _ { \mathrm { M i n i C P M } } ,\tag{41}
$$

where the two means are computed on the same effective set of sessions for that metric. We generate 20,000 bootstrap replicates using random seed 20260807. The 95% percentile confidence interval is

$$
\mathrm { C I } _ { 9 5 } ( m ) = \Big [ Q _ { 0 . 0 2 5 } \Big ( \{ \Delta _ { m } ^ { ( b ) } \} _ { b = 1 } ^ { 2 0 0 0 0 } \Big ) , Q _ { 0 . 9 7 5 } \Big ( \{ \Delta _ { m } ^ { ( b ) } \} _ { b = 1 } ^ { 2 0 0 0 0 } \Big ) \Big ] ,\tag{42}
$$

where $Q _ { \alpha }$ denotes the empirical α-quantile of the bootstrap effect distribution.

The primary inferential intervals are reported for latest-value accuracy and final-state fact coverage. When intervals are available for additional memory or interaction metrics, they are computed using the same paired session-level procedure. Absolute changes in percentage points are reported as the primary effect size; relative percentage changes are descriptive summaries.

Pairing is by session identifier and does not imply identical realized challenge delivery across systems. The intervals quantify session-sampling uncertainty conditional on the recorded runs and the specified scorer; they do not account for repeated-run generation variability, ASR errors, or scoringrule validity.

## G.3 BASELINE AND SUBGROUP REPORTING

The results for the five full-duplex baseline models are primarily descriptive. Each model is evaluated on the same session manifests and is aggregated with the same session-level procedure. No scalar ranking is derived by combining interaction and memory metrics. In particular, challenge realization is interpreted as exposure, while conditional full-duplex scores are interpreted together with challenge realization, trigger-timeout rate, and behavioral observability.

Language-, scenario-, and question-type analyses are reported as exploratory stratified summaries. The same definitions, denominators, and macro-aggregation rule are used within each stratum. Each subgroup result is accompanied by its session count and effective metric denominator. These stratified analyses are not treated as independent confirmatory tests, and no additional significance claims are made without a prespecified multiple-comparison procedure.

Table 15: Runtime validity, challenge exposure, and response coverage on all 1,094 sessions. Values are percentages except assistant speech duration.
<table><tr><td>Model</td><td>Protocol valid↑</td><td>Behaviorally observable↑</td><td>Challenge realization↑</td><td>Conditional cFD↑</td><td>Adaptive FD↑</td><td>Trigger timeout↓</td><td>Response coverage↑</td><td>Speech duration (s)</td></tr><tr><td>MiniCPM-o 4.5</td><td>100.00</td><td>100.00</td><td>73.51</td><td>40.23</td><td>35.28</td><td>6.04</td><td>28.38</td><td>78.35</td></tr><tr><td>Freeze-Omni</td><td>100.00</td><td>100.00</td><td>86.17</td><td>28.40</td><td>30.41</td><td>0.53</td><td>76.03</td><td>203.16</td></tr><tr><td>Moshi</td><td>100.00</td><td>100.00</td><td>44.63</td><td>37.36</td><td>13.08</td><td>27.67</td><td>12.61</td><td>29.23</td></tr><tr><td>Lychee-FD</td><td>99.82</td><td>100.00</td><td>75.64</td><td>47.32</td><td>48.19</td><td>1.09</td><td>35.30</td><td>99.19</td></tr><tr><td>PersonaPlex-7B-vl</td><td>100.00</td><td>100.00</td><td>93.54</td><td>22.74</td><td>35.41</td><td>1.73</td><td>32.43</td><td>74.20</td></tr></table>

Notes. Conditional cFD is conditioned on behaviorally scorable challenges and must be interpreted together with challenge realization and trigger timeout. Adaptive FD is aggregated from session-level scores and therefore need not equal the product of the two separately aggregated columns.

## G.4 DENOMINATORS AND REPRODUCIBILITY

For every reported metric, we retain the attempted-session count and the effective denominator used for aggregation. Runtime metrics additionally retain protocol-valid and behaviorally observable session counts, while memory metrics retain the total number of questions and the number of questions of each relevant type. Conditional quantities, including conditional full-duplex behavior and answer accuracy given a response, are reported together with their coverage variables.

All bootstrap computations use the fixed session identifiers, the fixed held-out split, and the fixed random seed specified above. The released statistical records contain session-level metric values, effective denominators, paired differences, bootstrap settings, and confidence intervals, allowing the reported aggregate results to be reproduced without treating individual questions as independent observations.

## H COMPLETE BENCHMARK RESULTS

## H.1 EVALUATION SCOPE

This section reports the complete aggregate results for MiniCPM-o 4.5, Freeze-Omni, Moshi, Lychee-FD, and PersonaPlex-7B-v1 on all 1,094 sessions. Each model is evaluated on the same immutable session manifests, comprising 6,564 same-session memory questions and checks, including 2,188 latest-value questions. All ratios are calculated within each session and then macroaveraged over sessions. The tables below include all aggregate runtime and memory metrics defined in Appendix C.

## H.2 RUNTIME VALIDITY AND CHALLENGE EXPOSURE

Table 15 reports execution validity, behavioral observability, adaptive-challenge exposure, conditional full-duplex behavior, response coverage, and assistant speech duration. Challenge realization measures the fraction of planned challenges that became behaviorally realized. Conditional cFD evaluates behavior on scorable challenges, whereas Adaptive FD is the macro-average of the corresponding session-level adaptive score. Assistant speech duration is descriptive and is not treated as a performance measure.

## H.3 TURN-BOUNDARY AND INTERRUPTION METRICS

Table 16 reports ordinary-turn boundary behavior, pause intrusion, and interruption stopping. Premature onset measures assistant speech beginning before the valid user-end boundary. Carry-in indicates that assistant speech from the preceding response remains active when the next user input begins. Pre-end overlap records any assistant speech overlapping the user before the user-end boundary.

Table 16: Turn-boundary and interruption-handling results. Rates are percentages and stopping latencies are measured in seconds.
<table><tr><td>Model</td><td>Premature onset↓</td><td>Carry- in↓</td><td>Pre-end overlap↓</td><td>Pause intrusion.↓</td><td>Stop P50↓</td><td>Stop P90↓</td><td>Stop@ 0.8s↑</td></tr><tr><td>MiniCPM-o 4.5</td><td>18.70</td><td>25.86</td><td>44.38</td><td>59.47</td><td>1.032</td><td>1.446</td><td>55.69</td></tr><tr><td>Freeze-Omni</td><td>49.76</td><td>42.44</td><td>91.82</td><td>73.64</td><td>1.239</td><td>1.802</td><td>45.75</td></tr><tr><td>Moshi</td><td>66.59</td><td>5.49</td><td>72.25</td><td>45.11</td><td>0.777</td><td>0.864</td><td>60.79</td></tr><tr><td>Lychee-FD</td><td>24.68</td><td>32.96</td><td>57.14</td><td>53.89</td><td>0.312</td><td>0.440</td><td>90.12</td></tr><tr><td>PersonaPlex-7B-vl</td><td>46.64</td><td>26.23</td><td>72.73</td><td>90.62</td><td>1.689</td><td>2.104</td><td>16.40</td></tr></table>

Table 17: Response availability and factual-update results. All values are percentages.
<table><tr><td>Model</td><td>Response rate↑</td><td>No-response rate↓</td><td>Latest-value accuracy↑</td><td>Superseded-value error↓</td><td>Accuracy given response↑</td></tr><tr><td>MiniCPM-o 4.5</td><td>98.90</td><td>1.10</td><td>4.30</td><td>0.59</td><td>4.32</td></tr><tr><td>Freeze-Omni</td><td>99.94</td><td>0.06</td><td>11.61</td><td>3.38</td><td>11.61</td></tr><tr><td>Moshi</td><td>54.31</td><td>45.69</td><td>2.19</td><td>0.32</td><td>3.60</td></tr><tr><td>Lychee-FD</td><td>92.41</td><td>7.59</td><td>25.50</td><td>1.65</td><td>30.80</td></tr><tr><td>PersonaPlex-7B-v1</td><td>96.40</td><td>3.60</td><td>5.85</td><td>2.29</td><td>5.93</td></tr></table>

## H.4 SAME-SESSION MEMORY METRICS

Tables 17 and 18 report all aggregate memory metrics. Latest-value accuracy requires the updated value to be present and the superseded value to be absent. Answer accuracy given response evaluates the same criterion only among latest-value questions for which an answer was produced.

## H.5 EFFECTIVE DENOMINATORS

Table 19 reports the principal aggregate denominators. All models are evaluated on 1,094 attempted sessions. Lychee-FD contains 1,092 protocol-valid sessions; its remaining behaviorally observ able sessions are retained in the attempted-session accounting. Metric-specific denominators for challenge, timing, pause, and interruption statistics follow the definitions in Appendix C.

## I QUALITATIVE EXAMPLES AND FAILURE ANALYSIS

## I.1 EXAMPLE SELECTION AND PRESENTATION

We present representative traces that expose failure modes not fully visible from aggregate scores. The examples were selected from completed, protocol-valid evaluation sessions and were not used for training, calibration, or checkpoint selection.

Each trace combines the scheduled challenge, the expected state transition, runtime interaction measurements, and the model’s spoken answer. Spoken outputs are reported from the frozen-ASR transcripts, with only whitespace and terminal punctuation normalized for readability. The examples are diagnostic rather than estimates of failure frequency.

## I.2 INTERACTION AND MEMORY CAN FAIL INDEPENDENTLY

Table 20 summarizes examples covering the four full-duplex scenarios. The traces demonstrate that correct interaction timing does not guarantee coherent memory and, conversely, that recovering an isolated fact does not imply correct full-duplex behavior.

Interruption is not equivalent to revision. The interruption example shows the clearest interaction–memory dissociation. The model stopped within the required interval for both realized challenges, yielding perfect stopping success in the selected session. However, it recovered only one of the two corrected values and admitted information explicitly marked as irrelevant. The interruption mechanism therefore reacted to the acoustic event without constructing the complete post-correction state.

Table 18: Distractor handling, final-state coverage, and answer-overlap results. All values are percentages.
<table><tr><td>Model</td><td>Distractor adoption↓</td><td>Distractor rejection↑</td><td>Summary fact coverage↑</td><td>Legacy token F1↑</td></tr><tr><td>MiniCPM-o 4.5</td><td>14.17</td><td>85.19</td><td>2.01</td><td>29.45</td></tr><tr><td>Freeze-Omni</td><td>65.90</td><td>34.10</td><td>0.00</td><td>15.42</td></tr><tr><td>Moshi</td><td>44.15</td><td>8.32</td><td>0.04</td><td>7.64</td></tr><tr><td>Lychee-FD</td><td>48.17</td><td>51.55</td><td>1.06</td><td>21.36</td></tr><tr><td>PersonaPlex-7B-vl</td><td>79.71</td><td>17.46</td><td>0.24</td><td>7.70</td></tr></table>

Notes. Distractor adoption and rejection are reported together because silence may reduce adoption without demonstrating correct rejection. Legacy token F is retained for compatibility and is not used as the sole measure of memory fidelity.

Table 19: Principal denominators for the complete benchmark evaluation.
<table><tr><td>Model</td><td>Attempted sessions</td><td>Protocol-valid sessions</td><td>Observable sessions</td><td>Memory questions</td><td>Latest-value questions</td></tr><tr><td>MiniCPM-o 4.5</td><td>1,094</td><td>1,094</td><td>1,094</td><td>6,564</td><td>2,188</td></tr><tr><td>Freeze-Omni</td><td>1,094</td><td>1,094</td><td>1,094</td><td>6,564</td><td>2,188</td></tr><tr><td>Moshi</td><td>1,094</td><td>1,094</td><td>1,094</td><td>6,564</td><td>2,188</td></tr><tr><td>Lychee-FD</td><td>1,094</td><td>1,092</td><td>1,094</td><td>6,564</td><td>2,188</td></tr><tr><td>PersonaPlex-7B-v1</td><td>1,094</td><td>1,094</td><td>1,094</td><td>6,564</td><td>2,188</td></tr></table>

Pause errors can precede otherwise plausible answers. In the pause-handling example, both models spoke during a region that should have been treated as an unfinished user turn. Their later memory behaviors diverged: one was selective but unable to state the revised values, whereas the other recovered a revised value but admitted a personal item. This case illustrates why a plausible answer to one question cannot substitute for joint measurement of interaction boundaries and persistent-state fidelity.

Explicit rejection is stronger evidence than silence. A model receives distractor-rejection credit only when its spoken answer explicitly rejects the irrelevant information. No response remains an unanswered case. This distinction prevents systems with low response coverage from appearing selective merely because they remain silent.

## I.3 AUDITABLE TRANSACTION TRACE

Table 21 presents an ACT-Mem trace from a held-out English pause-handling session. The user first supplied provisional budgeting values, later corrected the review time to this Saturday morning and the electricity cap to \$150, and explicitly marked an unrelated detail as unsuitable for storage.

This trace illustrates the value of retaining transaction provenance. The high-level UPDATE decision was correct, but the correction did not supersede the provisional entries because the extracted entity representation was inconsistent. An answer-only evaluation would record a failed latest-value ques tion without distinguishing a missed correction from failed entity resolution, failed supersession, retrieval error, or response omission.

The same trace also reveals a limitation of clause-level admission. A negated exclusion statement may contain a fluent entity–value pattern, causing its literal wording to be stored even though its pragmatic function is to prohibit storage. More robust semantic scope modeling is therefore required for utterances that combine relevant facts, corrections, and explicit non-persistence instructions.

## I.4 FAILURE TAXONOMY

We assign each qualitative error to the earliest observable stage that can explain it. Table 22 defines the resulting taxonomy. Later symptoms are retained for analysis but are not treated as evidence that all earlier stages were correct.

Table 20: Representative qualitative examples from protocol-valid sessions. RCR and cFD are reported as proportions for the selected session.
<table><tr><td>Scenario</td><td>Expected behavior</td><td>Observed evidence</td><td>Diagnosis</td></tr><tr><td>Interruption</td><td>Stop the current response, replace Saturday evening with Saturday morning, and replace twenty-five minutes with ten minutes.</td><td>Both interruptions were realized, with median stopping latency 0.368 s and Stop@0.8 of 1.00. The model recovered Saturday morning, but did not recover ten minutes. When asked about the side information, it answered that the distractor should be saved.</td><td>Fast interruption stopping was accompanied by only partial factual revision and incorrect admission. This separates acoustic interruption recovery from transactional memory updating.</td></tr><tr><td>Pause handling</td><td>Remain silent during within-turn pauses, then commit the completed correction: Thursday morning and fifteen minutes.</td><td>For the same session, both examined models intruded into every evaluated pause. One model rejected the distractor but recovered neither scored latest value. The other recovered Thursday morning, but answered “I will save all the details into the final memory&quot; for the distractor question and later included a travel toothbrush in its summary.</td><td>The first behavior is a missed update; the second is an admission failure followed by summary contamination. Both occur together with incorrect pause-boundary control.</td></tr><tr><td>Backchannel</td><td>Continue the current response without treating the backchannel as a new turn or a persistent fact.</td><td>All three challenges were realized, but session cFD was 0.33. The model nevertheless recalled the stable requirement to avoid hazelnut flavoring and retain whole beans, and correctly stated that the kitchen-clutter detail should not be saved. Its final summary remained incomplete.</td><td>Correct retention and distractor rejection can coexist with poor continuation behavior. Backchannel handling and final-state consolidation therefore require separate attribution.</td></tr><tr><td>Speech to others</td><td>Do not answer the non-addressed speech and do not admit its content into memory.</td><td>The session obtained RCR 0.75 and cFD 0.25. The model explicitly rejected the side information, but failed both latest-value questions and produced an incomplete final summary.</td><td>Rejecting a queried distractor does not establish that the system maintained the intended active state. Addressee control, supersession, and later answer generation remain distinct stages.</td></tr></table>

Quoted outputs are frozen-ASR transcriptions. The interpretations use the same fixed-denominator decisions as the quantitative evaluation; no qualitative relabeling is applied.

Table 21: An ACT-Mem transaction trace illustrating both successful control decisions and downstream state-construction failures.
<table><tr><td>Stage</td><td>Gold transition</td><td>Predicted transaction</td><td>Consequence</td></tr><tr><td>Initial request</td><td>WRITE provisional review time and budget constraints.</td><td>The controller predicted WRITE and committed extracted entity-value slots.</td><td>The initial values became active and were available to subsequent retrieval.</td></tr><tr><td>Correction</td><td>UPDATE the review time to this Saturday morning and the electricity cap to $150; supersede both provisional</td><td>The controller predicted UPDATE, but the extracted correction was represented as one compound slot with an entity key different from the</td><td>A new item was created, while the original items remained active and no supersession link was formed. The failure therefore occurred after the operation decision, during slot normalization and ledger resolution.</td></tr><tr><td>Distractor- bearing event</td><td>Commit the relevant condition and IGNORE the explicitly excluded side detail.</td><td>provisional slots. The relevant condition was written, but the exclusion clause “just a side detail and should not be saved&quot; was also</td><td>The ledger became contaminated even though the user&#x27;s utterance explicitly expressed non-persistence. This is an incorrect-admission failure rather than a retrieval-only failure.</td></tr><tr><td>Latest- value QA</td><td>Retrieve the active corrected electricity cap and answer $150.</td><td>extracted as an active fact. No spoken answer was produced.</td><td>The case remains incorrect under the fixed denominator. The trace localizes the outcome to unresolved state construction followed by answer realization.</td></tr><tr><td>Distractor QA</td><td>Produce an explicit rejection without exposing the side detail as evidence.</td><td>No spoken answer was produced.</td><td>Neither adoption nor successful rejection is credited; silence is kept separate from selective exclusion.</td></tr></table>

Controller decisions and slot contents are derived from frozen-ASR event transcripts. The trace reports the stored state before answer scoring and therefore permits failures to be localized without inferring hidden state from the spoken answer alone.

The examples identify three recurring patterns. First, interaction success is local: rapid stopping or correct backchannel continuation does not guarantee admission or revision of the correspond ing facts. Second, memory selectivity is compositional: a system may reject one queried distractor while retaining stale values or producing an incomplete summary. Third, an explicit transaction trace makes compound errors separable. In particular, correct operation classification does not ensure correct slot extraction, entity resolution, supersession, retrieval, or spoken realization. These distinctions explain why the benchmark reports interaction boundaries and memory outcomes separately and why no single scalar score is sufficient to characterize full-duplex persistent memory.

Table 22: Failure taxonomy used for qualitative attribution.
<table><tr><td>Failure</td><td>Observable criterion</td></tr><tr><td>Trigger timeout</td><td>A speech-dependent challenge is planned but cannot be injected because the required assistant-speech condition is not reached.</td></tr><tr><td>Boundary violation</td><td>The assistant begins before a completed user turn, speaks during a protected pause, or intrudes into a non-response window.</td></tr><tr><td>Slow interruption stop</td><td>Assistant speech continues beyond the stopping threshold after an interruption is delivered.</td></tr><tr><td>Missed write Incorrect admission</td><td>An admissible new fact produces no committed memory item. Non-addressed, incomplete, backchannel, or explicitly excluded content is committed as persistent</td></tr><tr><td></td><td>memory.</td></tr><tr><td>Failed supersession</td><td>A valid correction is detected, but the prior active value is not linked and marked as superseded. A superseded value is exposed to the answer generator despite the presence of a valid active</td></tr><tr><td>Stale retrieval</td><td>replacement.</td></tr><tr><td>Distractor leakage</td><td>An excluded item enters the evidence card or appears affirmatively in the spoken answer or final summary.</td></tr><tr><td>Retrieval miss Answer realization error</td><td>The required active item exists in the ledger but is absent from the question-conditioned evidence. Appropriate evidence is available, but the spoken response omits, distorts, or contradicts it, or no</td></tr><tr><td></td><td>answer is produced.</td></tr><tr><td>Transcription error</td><td>The frozen-ASR transcript changes a value or boundary in a way that affects controller input or answer scoring.</td></tr></table>

Table 23: Artifact manifest for reproducing the benchmark and experiments.
<table><tr><td>Artifact group</td><td>Representative records</td><td>Reproducibility role</td></tr><tr><td>Session inputs</td><td>source. json, session audio, event audio</td><td>Defines language, scenario, event order, speaker roles, timing, and the rendered input waveform.</td></tr><tr><td>Interaction manifests</td><td>interactive_manifest.json, challenge and timing manifests</td><td>Specifies challenge conditions, observation windows, trigger rules, and the fixed execution order.</td></tr><tr><td>Memory supervision</td><td>memory-qa.json, reference-state records, update links</td><td>Provides question types, accepted and forbidden values, expected operations, supporting evidence, and final-state supervision.</td></tr><tr><td>Reference behavior</td><td>reference_behavior.json</td><td>Stores the intended interaction response for each challenge and the corresponding state-transition constraints.</td></tr><tr><td>Runtime execution</td><td>runtime_events.jsonl,VAD and timing records</td><td>Records challenge realization, response boundaries, interruption stopping, overlap, pauses, and execution failures.</td></tr><tr><td>Generated outputs</td><td>Assistant waveform, raw predictions, QA windows, frozen-ASR transcripts</td><td>Preserves the model output and the temporal windows used for answer attribution and transcription.</td></tr><tr><td>Question-level scores</td><td>Per-question score records and session summaries</td><td>Allows reconstruction of response coverage, latest-value accuracy, stale assertions, distractor outcomes, and final-state coverage.</td></tr><tr><td>ACT-Mem traces</td><td>Controller-ledger traces, retrieval records, evidence cards</td><td>Exposes event operations, committed items, supersession links, active values, and question-conditioned evidence.</td></tr><tr><td>Configuration and scripts</td><td>Inference, scoring, aggregation, and calibration configurations</td><td>Reproduces execution settings and regenerates the reported tables from the recorded outputs.</td></tr></table>

## J REPRODUCIBILITY AND ARTIFACT MANIFEST

This section specifies the artifact organization required to reproduce the reported experiments. All released records use stable session identifiers and preserve the mapping from each input session to its events, challenges, questions, model outputs, and scores. The artifact structure is summarized in Table 23.

## J.1 ARTIFACT ORGANIZATION

The records are linked by session identifier and question identifier. Consequently, a question-level score can be traced back to its input audio, the corresponding event prefix, the realized challenge schedule, the generated answer, and the evidence used for scoring. Failed, unanswered, and timedout executions remain in the released records and are not removed before aggregation.

## J.2 DATA SPLITS AND PROVENANCE

The release-level development and test partitions contain 237 and 857 sessions, respectively. The method-development partitions contain 660 training sessions, 175 validation sessions, 40 calibration sessions, and 219 held-out sessions. Splits are defined at the session level, so that events, questions, and audio derived from one session cannot appear in more than one partition.

The training and validation records are used for parameter fitting and checkpoint selection. The calibration records are used only for selecting the admission threshold. The held-out records are reserved for the final comparison and are not used for training, calibration, or model selection. Each reported held-out result can therefore be reconstructed from the corresponding fixed session manifest and its stored output records.

## J.3 EXECUTION AND SCORE RECONSTRUCTION

The complete reproduction path is

$$
\begin{array} { r l } & { \mathrm { s e s s i o n ~ m a n i f e s t \to \ s t r e a m i n g ~ e x e c u t i o n \to r u n t i m e ~ e v e n t ~ l o g } } \\ & { \qquad \to \mathrm { g e n e r a t e d ~ a u d i o \to \ f r o z e n \mathrm { - } A S R ~ t r a n s c r i p t } } \\ & { \qquad \to \mathrm { q u e s t i o n \mathrm { - } l e v e l ~ s c o r i n g \to \mathrm { s e s s i o n ~ a g g r e g a t i o n \to r e p o r t e d ~ t a b l e } } . } \end{array}\tag{43}
$$

The runtime log determines whether an adaptive challenge was realized and provides the timestamps required for interaction metrics. The question records determine the fixed denominators for memory evaluation. Session ratios are computed before macro-averaging across sessions, and unan swered or invalid responses remain in the corresponding denominators. The final aggregation files are generated from these session-level records rather than manually transcribed into the tables.

For the controlled comparison, the held-out session identifiers are retained in paired form for the backbone and memory-augmented runs. This allows the reported confidence intervals to be regenerated with paired session resampling while preserving the correspondence between the two conditions.

## J.4 INTEGRITY AND REPRODUCTION CHECKS

Before scoring, the reproduction script verifies that every session manifest has a matching input waveform, event list, challenge record, and question set. It also checks that event and question references are unique and resolvable, that all event timestamps fall within the declared session duration, and that no session identifier occurs in multiple partitions. After inference, the script verifies that every output can be mapped to a fixed input session and that the score denominators agree with the released question manifests.

These checks ensure that the reported results are reproducible from session-level inputs and recorded outputs, while preserving the traceability required to inspect individual interaction and memory failures.