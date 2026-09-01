# Auditing Anonymous AI Models: A Four-Stage Protocol for Black-Box Identity Verification

Yisen Xi

Independent Researcher, Beijing, China

## Abstract

The 2025–2026 AI market has seen a wave of stealth releases: frontier models launched anonymously on developer platforms under codenames. For their users, identity determines data-handling terms, supply-chain risk, and capability expectations. No validated methodology exists for black-box identity verification of anonymous models: practitioner checklists lack accuracy evidence, and self-identification is untrustworthy by design. We propose a four-stage forensic audit protocol for API-served models. Stage 0 reconstructs launch-time configuration from archived platform snapshots (Internet Archive), exposing preview–production drift. Stage 1 fingerprints configuration (context, output ceiling, reasoning, modality) against the platform catalog. Stage 2 tests tokenizer identity with a cross-length diferential that rejects short-prompt collisions. Stage 3 corroborates with behavioral probes. We test declaration consistency on 10 known-identity releases (7 exact, 2 precision-diferences, 1 partial, 0 counter-directional), not end-to-end identification under anonymity. Identification is validated prospectively on a flagship case whose 2026-08-23 analysis pointed to the GLM-5.3 version line and whose oficial reveal confirmed those family and version-line inferences (deployment variant was not pre-asserted; Flash was consistent post-reveal), and on three Stage-0-only cases where the protocol produced a graded hypothesis or declined rather than guessed. A standard-library-only implementation is provided as supplementary material.

Keywords: anonymous model audit, digital forensics, black-box fingerprinting, tokenizer diferential, configuration forensics, AI supply-chain risk

## 1. Introduction

## 1.1. The problem

On August 20, 2026, the reasoning model Ox Alpha appeared on the OpenRouter API platform: a 1,048,576-token context window, multimodal input, tool calling, price zero, and no vendor attribution: a “stealth model. . . developed and operated by a third-party provider who has chosen to remain anonymous.” Such anonymity is not incidental: anonymous models are engineered not to reveal themselves (approximately 300 community identityattack probes elicited zero credible self-identifications), and platform infrastructure obfuscates usage data. Yet identity is operationally decisive for users: data-retention terms are channel- and provider-specific, supply-chain risk (who serves the model? on what infrastructure?) is uninsurable without attribution, and capability expectations (what family? what training era?) shape engineering decisions.

This is not a single-case curiosity. Between 2025 and 2026 we enumerate 20 stealth-release cases from nine vendors across three distribution channels (built from public platform records and archived snapshots; full list in Supplementary Material S1); of these, 10 have both a launch-time specification record and a known identity and form the retrospective sample in Section 4. Anonymous models have appeared among high-usage entries on developer platforms. The verification problem (given an anonymous model, determine its true identity) is therefore a digital-forensics and serving-integrity task: reconstruct identity from artifacts the serving stack cannot easily fake, and produce an auditable evidence trail for adoption and compliance decisions. Independent audits of claimed-oficial third-party endpoints likewise find widespread identity mismatch (Zhang et al., 2026b): 45.83% of fingerprint checks failed on shadow APIs. This is evidence that API model identity cannot be taken on trust even when a brand is advertised. It also connects to established security concerns: model-extraction attacks showed that API access leaks model functionality (Tramèr et al., 2016), supply-chain attacks on open-source dependencies are a documented, growing threat class (Ohm et al., 2020), and frontier-model identity determines the trust basis of every downstream integration (OpenAI, 2023).

## 1.2. Why not ask the model? Why not trust disclosures?

Self-identification is a contaminated signal: anonymous models are system-engineered to self-present only under their codename, and community tests show both successful suppression (zero self-confessions under adversarial prompting) and unreliable self-claims (local deployments of the same family misidentifying as Claude). Vendor and platform disclosures are authoritative when they occur, but they are the output of an adoption process that may never happen (one corpus case has remained anonymous for over eight months), and disclosures are not auditable. Practitioner protocols (Reed, 2026) provide a disclosure-evidence ladder and reproducible test protocol but publish no accuracy evidence, no retrospective validation, and no failure-mode analysis.

## 1.3. Contributions

We contribute: (1) a four-stage forensic audit protocol for blackbox identity verification of anonymous models, including a Stage 0 archivedconfiguration forensics method (time-consistent specification baseline via Internet Archive / record extracts) and a Stage 2 cross-length tokenizer differential test with a documented collision failure mode; (2) retrospective declaration-consistency validation on 10 stealth releases with known identities (7 exact, 2 precision-diferences, 1 partial, 0 counter-directional); (3) prospective blind validation on the flagship Ox Alpha case, whose 2026-08-23 dated measurement artifacts supported a Zhipu GLM-5.3 versionline attribution later confirmed by the oficial reveal (ZAI GLM-5.3-Flash, 2026-08-26); deployment variant was not pre-asserted, plus a two-outcome tracking design for a long-running anonymous model (Big Pickle); (4) negative controls (known-backend re-acceptance; non-family collision rejection) and “cannot-attribute” demonstrations; (5) a standard-library-only implementation (audit-tool) plus an auditor worksheet template, submitted as supplementary material; and (6) scoped portability to branded-base and serving-switch audits, distinguished from unvalidated end-to-end product studies.

## 1.4. Scope and disclosure

The protocol is for settings where vendor or platform disclosure is absent, delayed, or disputed; it does not replace disclosure when one exists. All experiments used public API endpoints with zero-cost or vendor-approved access and respect platform terms. Findings are labeled “identity unconfirmed, reproducibility limited” where applicable (per provider data-retention terms). The flagship case was re-verified against the known model after oficial reveal (Section 5.1: constant tokenizer diferential of 75 tokens across all four paired probes, four days post-reveal).

## 2. Background and Related Work

## 2.1. Model identification in AI systems

Prior work on model identification falls into four streams.

AI-generated content detection (Gehrmann et al., 2019; Mitchell et al., 2023; Bao et al., 2024) attributes authorship of text, not model provenance.

Watermarking and output fingerprinting (Kirchenbauer et al., 2023; Szyller et al., 2021; Hu et al., 2024) require model-side cooperation or pre-existing fingerprints, absent in anonymous previews. Related work on IP protection fingerprints (Xu et al., 2025), adversarial robustness of LLM fingerprints (Nasery et al., 2025), and robustness of fingerprint-style inference (Szyller et al., 2022) shares the reliability concern but still targets cooperative settings.

Active black-box fingerprinting sends crafted queries and classifies responses. LLMmap (Pasquini et al., 2025) combines expertise-informed probes with a classifier over the LLM version behind an application. ZeroPrint (Shao et al., 2026, 2025) estimates output-distribution gradients and provides LeaFBench for copyright auditing. Bruckner (2026) verifies a claimed OpenRouter endpoint against an enrolled reference using single-token answer distributions. FLIPS (Richardeau et al., 2026) targets instance-level IDs via pseudo-random sequences at finer granularity than family attribution.

These fingerprinting lines assume a cooperative or non-resistant target: response style and output statistics that stealth models are engineered to suppress (Section 1.1). They also omit prospective blind validation, explicit decline outputs, and launch-time baselines. We therefore rank deployment properties the target cannot easily fake: archived declarations (Stage 0), catalog fields (Stage 1), tokenizer diferentials (Stage 2) over behavioral corroboration (Stage 3).

Gateway substitution audits test whether the served backend matches what was advertised. Prior membership-inference and extraction work shows that API access leaks model information (Shokri et al., 2017; Jagielski et al.,

2020; Tramèr et al., 2016; Carlini et al., 2024). IRIS (Zhang et al., 2026a) detects stream substitution and routing dilution; Stability Monitor (Leshin et al., 2026) flags serving changes from output-distribution drift. A concurrent shadow-API audit (Zhang et al., 2026b) reports that 45.83% of audited endpoints advertising frontier brands failed LLMmap fingerprint checks, with large utility gaps even when branding matched.

Those audits answer whether a declared backend is faithful or when it changed. Our setting is diferent: who serves when no truthful declaration exists. Cross-provider serving deviations nonetheless overlap with our Stage 1/2 signals.

Across these strands, none validates on deliberately anonymized models, none pairs a declaration-consistency sample with a prospective blind test, and none treats cannot attribute as a first-class output. Table 1 summarizes the gap.

## 2.2. Tokenizer-based attribution

Tokenizer fingerprints have been used for model-family attribution in community practice (e.g., counting prompt tokens across candidate models to detect shared tokenizers). Two limitations are known but under-documented: (i) short prompts can produce coincidental equal token counts across diferent tokenizers (we document empirical counterexamples: three non-candidate families exhibited constant diferentials on short prompts that collapsed under long-text testing); and (ii) gateway-level usage reporting can be transformed by the intermediary, so raw token counts require diferential (betweenmodel) rather than absolute interpretation. Our cross-length diferential test (Section 3.3) addresses both limitations.

## 2.3. Practitioner protocols and community forensics

Reed (2026) proposes a disclosure-evidence ladder for stealth models: evidence channels (platform disclosure, provider behavior, model card, etc.) graded as confirmed, inferred, or unknown, plus a reproducible test protocol over technical and behavioral signals. The protocol has no retrospective validation, no failure-mode analysis, and relies on disclosures rather than active measurement. Shrey (2026) conducted a large community forensics campaign on a single flagship case (600+ requests, 13.5M prompt tokens): 44- string tokenizer diferentials, knowledge-boundary tests, context-limit sweeps, and modality tests, concluding GLM-5 generation attribution. The constant tokenizer-count ofset on that case was first reported there; our contribution is the cross-length constancy criterion, collision rejection, and protocolization, not the original observation. That campaign validated the power of tokenizer diferentials at scale but was case-specific, ad hoc, and, as the oficial reveal showed, less precise than configuration-layer evidence at the version level (the campaign’s knowledge-boundary inference pointed to the base checkpoint; configuration evidence pointed to the version line that the reveal confirmed). Our protocol systematizes these dispersed techniques into a staged, validated, tooled procedure, and adds the Stage 0 time-consistent baseline that the community campaign lacked.

## 2.4. Positioning

Relative to the systems in Section 2.1–2.3, we difer mainly in four places: a declaration-consistency sample (retrospective N=10), prospective blind testing (including a confirmed flagship case), an explicit cannot attribute output, and a standard-library-only implementation. Table 1 situates these gaps against the closest related systems (signals and mechanisms are discussed in Section 2.1–2.3). Scope relative to vendor disclosure is stated in Section 1.4.

Table 1: Comparison with related identification / serving-integrity methods.
<table><tr><td>Method</td><td>Setting</td><td>Coop.</td><td>Prospective</td><td>Decline</td></tr><tr><td>Reed (2026)</td><td>Stealth releases</td><td>Partial</td><td>No</td><td>Graded labels only</td></tr><tr><td>LLMmap (Pasquini et al., 2025)</td><td>Known models behind apps</td><td>Yes</td><td>No</td><td>Open-set ≠ decline</td></tr><tr><td>ZeroPrint (Shao et al., 2025, 2026)</td><td>Fingerprint/ copyright</td><td>Yes</td><td>No</td><td>No</td></tr><tr><td>Bruckner (2026)</td><td>Claimed OpenRouter</td><td>Yes</td><td>No</td><td>Vs enrolled reference</td></tr><tr><td>FLIPS (Richardeau et al., 2026)</td><td>endpoints Instance/ config ID</td><td>Yes</td><td>No</td><td>Classifier only</td></tr><tr><td>IRIS (Zhang et al., 2026a)</td><td>Gateway substitution</td><td>Yes</td><td>No</td><td>Deviation flag</td></tr><tr><td>Shadow-API audit (Zhang et al., 2026b)</td><td>Claimed third-party</td><td>Yes</td><td>No</td><td>Mismatch vs official</td></tr><tr><td>Stability Monitor (Leshin</td><td>APIs Serving change detection</td><td>Yes</td><td>Change-points only</td><td>Change ≠ identity</td></tr><tr><td>et al., 2026) This work</td><td>Anonymous APIs</td><td>No (Stages 0–2)</td><td>Yes (Ox Alpha)</td><td>Yes</td></tr></table>

Note. Coop. = needs cooperative behavioral or distributional responses. Prospective = anonymous pre-reveal test later scored against an oficial reveal. Decline = first-class cannot attribute (or equivalent audit stop) rather than forced classification.

## 3. The Audit Protocol

The protocol has four stages (Stages 0–3), with a mandatory cross-length guard inside Stage 2 (Fig. 1). It is implemented in audit-tool (zero thirdparty dependencies; keys read from environment, never logged or stored). The intended user is an IT-security or supply-chain auditor who must decide whether to adopt an anonymous endpoint before a vendor disclosure exists; each stage produces artifacts intended to form an auditable evidence chain consistent with forensic incident-response practice (NIST, 2006). For prospective claims, auditors should commit graded conclusions (e.g., SHA-256 of a prediction card) to a public timestamped channel before reveal; dated measurement logs alone are weaker under adversarial scrutiny of self-held clocks.

![](images/72b7535a8a480d92b9a3c713c7f0f7622b07f87ccbecef6c6ceb4b603e461229.jpg)  
Figure 1: Four-stage forensic protocol for anonymous-model identity verification. Stage 0 reconstructs the launch-time configuration from archived platform records. Stage 1 shrinks the live catalog to a configuration-compatible candidate pool. Stage 2 tests tokenizer identity with a mandatory cross-length diferential (short-prompt constancy is not evidence). Stage 3 corroborates with behavior. Insuficient evidence yields cannot attribute rather than a forced guess.

## 3.1. Stage 0: Configuration-declaration forensics (time-consistent baseline)

Goal. Establish what the target declared at launch time, and whether the declaration drifted over time.

Procedure. (1) Locate archived snapshots of the model’s platform record page (Internet Archive; also platform /models endpoints) at launch and at regular intervals; coverage of those pages is the critical dependency here (Ainsworth et al., 2011). (2) Extract configuration fields: context window, max output, reasoning controls, modality (text/image/video/audio), supported parameters. (3) Compare launch-time vs. current declarations; flag drift (we document version drift between preview and production configurations in a flagship case: preview 1,048,576 → production 1,310,720 context, with modality constant, see Section 5). (4) Record the launch-time configuration as the comparison baseline for Stages 1–3.

Rationale. Configuration is a deployment property, not a model-weights property; it varies across snapshots and channels. Using current specifications to test a model whose preview configuration difered invites false negatives (we document this as a retrospective method-correction lesson, Section 4.1). Stage 0 converts the “time-point scanning” principle of disclosure auditing into operational forensics. Table 2 records the two central drift cases used in this study; the full extraction sheet is Supplementary Material S3. Auditors are expected to produce a portable worksheet (Supplementary Material S2) rather than an informal notebook.

Table 2: Stage 0 drift cases (launch-time vs later declaration; full extracts in S3).
<table><tr><td>Case</td><td>Launch ctx</td><td>Later / official</td><td>Role</td></tr><tr><td>Ox Alpha (preview) 1,048,576</td><td></td><td>1,310,720 (GLM-5.3-Flash)</td><td>Prospective baseline</td></tr><tr><td>Polaris Alpha</td><td>262,144</td><td>400K (GPT-5.1)</td><td>Retrospective partial /drift lesson</td></tr></table>

## 3.2. Stage 1: Configuration fingerprinting against the catalog

Goal. Shrink the candidate pool from the platform’s full model catalog to a small set of configuration-compatible candidates.

Procedure. (1) Pull the platform’s full model catalog (e.g., OpenRouter api/v1/models). (2) Build the target’s configuration fingerprint from the launch-time record (Stage 0): context window, max output, reasoning controls, modality, supported parameters. (3) Match the fingerprint against all catalog entries; report the candidate set as a layered shrinkage, not a single pool: matching on context window and max output alone (the two most common fields) shrank the flagship-case catalog from 422 entries to 6 (a 70× reduction); adding modality produced a unique match. The conservative pool (context + max output) is the candidate set carried into Stage 2, since it preserves recall over vendor families; the unique modality-level match is recorded as a strong auxiliary signal with a caveat: in the flagship case, the anonymous target declared video input while its true public counterpart did not; post-hoc, this was an early leakage of the variant’s multimodal capability ahead of the public catalog’s declaration, but at audit time it was indistinguishable from declaration noise. Stage 1 therefore reports both pools and flags modality divergence between target and candidates rather than discarding candidates on it. (4) Record modality separately as a scarce time-invariant signal: modality is less subject to context inflation and version drift than context length (retrospective evidence: modality identified a variant that context length could not), while its declaration can lag serving capability, making Stage 0’s launch-time baseline the arbiter of which declaration is authoritative.

## 3.3. Stage 2: Cross-length tokenizer diferential test (mandatory guard)

Goal. Establish tokenizer identity (same tokenizer implies same family/generation) with a false-positive-resistant test.

Procedure. (1) For the target and each candidate, query the same prompt and record prompt\_tokens from usage fields; compute the diference. (2) Repeat across ≥3 texts spanning short (identity questions), long Chinese, and long English texts (mandatory: at least one long text per language family). (3) Constancy rule (ε = 0): the target and candidate are treated as sharing a tokenizer only if the diferential is exactly equal across all retained probes (same tokenizer implies the diference equals the fixed hidden-systemprompt injection delta, which is prompt-length invariant). Near-matches are not accepted. (4) Probe-stability rule (protocol-level, applied before the constancy test): for each probe id, if the target endpoint returns inconsistent prompt\_tokens across repeated identical rounds, discard that probe from the constancy assessment for all candidates; do not tune the probe set after seeing family outcomes. (5) Counterexample rule: if a constant diferential appears on short prompts only and collapses on long texts, classify as collision (diferent tokenizers coincidentally agreeing on short token counts), not evidence.

Failure-mode documentation. We empirically document three collision cases on non-family endpoints (step-3.7-flash, hy3, kimi-k3): on short identity prompts they exhibited constant (or near-constant) diferentials (75, 72, and 2 tokens respectively) that dissolved under the archived 20-probe ox-full battery (Fig. 2 source JSONL): step-3.7-flash ranged over 69–77, hy3 over 68–77, and kimi-k3 over –1 to 5, while the true family held a constant 75 across every paired probe of that battery and across the five canonical crosslength probes (short×3 + long Chinese + long English). A separate 16-probe archival run shows the same family constant at 75 on 15/16 probes; applying the probe-stability rule above, a style probe (style\_en) was discarded because the target alone was round-unstable (105 then 20 tokens) while candidates remained stable at 30) before computing constancy. Target-side instability is itself a serving-path signal (template, caching, or non-deterministic accounting), not evidence for or against a candidate family. Without the mandatory multi-length condition, the short-prompt collisions would have produced false positives. Same-generation siblings that do share a tokenizer (in the flagship run, glm-5.2 with glm-5.3) remain constant across lengths; that is family-level identity, not a collision, and version-line separation is left to Stage 1. Hence cross-length constancy is the criterion; shortprompt constancy alone is not evidence. For the short-prompt-only comparison in Table 3 we use a near-constant acceptance rule (diferential spread $\leq 2$ tokens across the three identity/cutof probes counts as constant), reflecting the coarser quantization of short prompts; under this rule kimi-k3’s $\Delta \in \{ 2 , 4 \}$ pattern is admitted as a collision, and the short-only rule yields two true acceptances plus three false positives.

Collision ablation. The mandatory multi-length condition is the central component of Stage 2, not a stylistic precaution. Replaying the flagship-case ox-full run under two decision rules (Fig. 2; Table 3) shows that shortprompt-only constancy admits three non-family false positives, while the full multi-length rule admits only the true family. Multi-length structure is what converts a heuristic into a criterion; with a single short string, collision and identity are indistinguishable.

![](images/eea3cbd8a5ab34f5ff581625f64b751244397152a2c62cda040a1d1f1b82eced.jpg)  
Figure 2: Tokenizer diferentials (ox → candidate) from the archived ox-full JSONL (2026-08-23); ∆ = prompt\_tokens(ox) – prompt\_tokens(candidate). Left: three short-probe $\Delta$ points; right: full-battery $\Delta$ range (min–max) over paired probes.

Table 3: Stage 2 outcomes on the five Fig. 2 endpoints.
<table><tr><td>Decision rule</td><td>TP</td><td>FP</td><td>TN</td><td>FN</td></tr><tr><td>Short-prompt only</td><td>2</td><td>3</td><td>0</td><td>0</td></tr><tr><td>Full multi-length  $( \varepsilon = 0 )$ </td><td>2</td><td>0</td><td>3</td><td>0</td></tr></table>

Note. Source: ox-full JSONL, 2026-08-23; true family = {glm-5.2, glm-5.3}. Short Accept includes kimi-k3’s near-constant short pattern $( \Delta \in \{ 2 \}$ 4}); Full Accept uses $\varepsilon = 0$ over paired ox-full probes.

Beyond the five Fig. 2 endpoints in Table 3, three further non-family endpoints in the same JSONL (ernie-, mimo-, and qwen-line) are rejected by the full multi-length rule (archive-wide: TP = 2, FP = 0, TN = 6, FN = 0); they also fail exact short ε = 0 constancy and are outside the Fig. 2 collision panel.

Attribution boundary. Tokenizer identity establishes family and generation with high confidence; it does not establish exact checkpoint, quantization, or deployment variant (we document a case where tokenizer matched a generation while the true model was a new-architecture variant of that generation (same tokenizer family, diferent architecture; Section 5)). The diferential design also neutralizes a gateway-level threat: intermediaries can transform absolute usage reports, so only between-model diferences on identical prompts are treated as evidence. We use SentencePiece-style subword segmentation (Sennrich et al., 2016; Kudo and Richardson, 2018) as the reference model of tokenizer behavior: subword segmentation is deterministic given the tokenizer and prompt, so cross-length constancy of the diferential follows from tokenizer identity, and any deviation is diagnostic.

Formal basis. The test’s logic can be stated precisely. Let a serving endpoint E wrap a model with tokenizer $T$ and prepend a fixed hidden system prompt $s _ { E }$ to every user prompt $u ,$ reporting ptok $ \mathbf { \sigma } _ { \cdot E } ( u ) \ = \ T ( s _ { E } \cdot u )$ where $\mathrm { p t o k } _ { E } ( u )$ denotes the prompt-token count reported by endpoint E for prompt u, and · denotes string concatenation. Under (A1) deterministic segmentation and (A2) faithful arithmetic reporting, and provided the prompt assembly is boundary-stable (the template closes with a fixed delimiter so that segmentation of u is independent of the preceding context), $T ( s _ { E } \cdot u ) = T ( s _ { E } ) + T ( u )$ . Consequently, for any two endpoints that share a tokenizer,

$$
\mathrm { p t o k } _ { E _ { 1 } } ( u ) - \mathrm { p t o k } _ { E _ { 2 } } ( u ) = T ( s _ { E _ { 1 } } ) - T ( s _ { E _ { 2 } } ) \equiv \Delta \quad \mathrm { f o r ~ a l l ~ } u ,
$$

That is, the diferential is a prompt-length-invariant constant equal to the diference in hidden-injection lengths. The diagnostically useful direction is the contrapositive: if two prompts $u _ { 1 }$ and $u _ { 2 }$ yield non-constant diferentials (under the idealization $\varepsilon = 0$ , i.e., zero reporting noise), then at least one of (a) diferent tokenizers, (b) non-fixed injection, or (c) reporting transformation must hold; each of which independently disqualifies same-backend attribution. The converse (cross-length constancy implying the same tokenizer) is not a general theorem; it is only a necessary consequence of tokenizer identity under (A1)–(A2). We use it operationally because short-prompt collisions are common, whereas multi-length collisions are empirically rare in our archives (Fig. 2). Joint collision risk over the k length bands of the battery $( k = 5$ in the canonical cross-length probe set) need not factor into independent per-band probabilities (tokenizers often share BPE merge structure, so collisions can be positively correlated); multi-length rejection power is therefore treated as an empirical operating characteristic rather than a multiplicative probability claim. Two boundary conditions further delimit the claim: a server-side template change alters $s _ { E }$ and thereby shifts $\Delta$ (the version-drift event detected by Stage 0), and an (A2)-violating gateway can distort absolute counts (neutralized by the diferential design). Empirically, cross-length constancy held for the true family through reveal (Section 5.1), and the documented short-prompt collisions were rejected by the multi-length rule (Table 3).

## 3.4. Stage 3: Behavioral probes and corroboration

Goal. Corroborate or discriminate candidates using behavior that candidates cannot easily fake: (1) knowledge boundaries (model launch-time recall: knowing an earlier model but not the candidate generation’s launch, noisy, and we rank it below configuration and tokenizer evidence after the flagship case’s reveal contradicted a knowledge-boundary inference; calibration research shows models partially encode what they do not know, Kadavath et al. (2022), but self-reported identity remains unreliable for deliberately anonymized models, Section 1.2); (2) upstream error codes and refusal patterns (provider-specific HTTP error bodies, content-filter patterns); (3) reasoning controls (/nothink behavior, reasoning-efort parameter handling); (4) modality capability tests (image/video/audio handling; audio failure vs. image success discriminated variant profiles in the flagship case); (5) multilingual behavior (weak discriminator alone).

## 3.5. Confidence grading and output discipline

Attribution outputs carry a graded confidence: (i) family (vendor/generation), (ii) version line, (iii) variant (flash/turbo/max), (iv) exact checkpoint (rarely attainable). The protocol explicitly supports “noanswer” outputs: when evidence is insuficient (no discriminating signal, no configuration match, scarce signals uninformative at the time point), the correct output is cannot attribute, not a forced guess. We demonstrate three Stage-0-only cases (Section 5.3) where the protocol produced a graded hypothesis or cannot attribute rather than a forced identification; in one case, post-hoc verification confirmed the decline was correct.

## 3.6. Threat model, detection boundaries, and portability

Adversary classes. The protocol is designed against four classes (classification adapted from standard threat-modeling practice, Shostack, 2014):

• A-idle (honest anonymous deployment): no adversary; the operator withholds identity but reports configuration truthfully. Baseline scenario: Stages 0–2 operate at full strength.

• A-declare (declaration manipulation): the operator falsifies the catalog record (inflated context, wrong modality) to mislead fingerprinting. Mitigations: Stage 0 anchors the launch-time declaration from an independent archive (the attacker must have falsified the record at launch and maintained the forgery across snapshots), and several declared properties are behaviorally testable: advertised context by boundary-probing, advertised modality by modality tests (Stage 3). Residual risk: undeclarable serving properties (quantization, inference stack) are untestable black-box, an acknowledged boundary.

• A-mimic (backend mimicry): a provider serves a diferent model behind the record of the advertised one (impersonation or undisclosed swap), the ML-supply-chain analogue of backdoored or swapped thirdparty components (Gu et al., 2017). Detected when the served backend’s configuration/tokenizer fingerprints diverge from the declaration or from reference samples. This is the division of labor with gatewaysubstitution auditing (Section 2.1): IRIS-class monitors detect that a substitution occurred with statistical guarantees; our Stages 1–2 attribute which model is now being served, without enrollment. Realworld shadow-API measurements illustrate a related boundary: fingerprint agreement with a claimed brand need not preserve task utility (Zhang et al., 2026b report large MedQA drops for advertised Gemini endpoints): identity match is not behavioral fidelity, so Stage 3 remains a cross-check rather than a substitute for Stages 0–2.

• A-evasive (adversarial suppression): the operator engineers the model to resist identity leakage; the stealth-release case (Section 1.1: zero self-identifications under \~300 community probes). Defeated by construction: Stages 0–2 consume deployment properties (declarations, catalog entries, usage arithmetic) that do not depend on model co operation. Residual risk: a full-stack proxy that forwards queries to a diferent model’s API while rewriting usage fields; against this, the protocol ofers a cross-stage inconsistency alarm: when Stage 2 attributes family X but Stage 3 behavior (self-identification content, vendor markers) is characteristic of family Y, the attribution is flagged as proxy-suspect. The alarm’s positive direction (stages agree) was observed post-reveal (Section 5.1). Its negative direction is exercised in controlled Experiment E1 (Section 6.1; Supplementary Material S4): forging prompt\_tokens to match a declared GLM endpoint while serving a Kimi backend makes Stage 2 accept the declared family (constant $\Delta { = } 0 )$ , while Stage 3 identity probes return Moonshot/Kimi self-reports, triggering the alarm in both ofline JSONL replay and live OpenRouter runs.

Detection boundaries. Stage 0 trusts the archive’s integrity (Wayback compromise is out of scope); Stage 2 trusts arithmetic reporting up to the diferential design (A2 violations afecting both models symmetrically cancel; asymmetric forgery triggers the consistency alarm above); no stage detects inference-stack changes below token-count resolution: quantization, kernel, or hardware swaps that preserve token counts are invisible to all three stages. That layer sits beside change-detection systems that fingerprint output distributions over time (Stability Monitor, Leshin et al., 2026; they detect that something changed; we attribute what it is); together they cover a fuller serving-integrity audit.

Portability. Primary scenario: oficial anonymity (no declaration; inference from observable behavior). Secondary: shadow-API impersonation (a provider masquerading as a known model). Extended scenarios (Section 6): branded-model base attribution (declared model ̸= actual backbone, e.g., coding products built on third-party open backbones) and silent-downgrade detection (ad-funded free tools that declare a premium model but serve a cheaper variant under load, detectable because downgrade changes tokenizer/configuration fingerprints). The tool’s configuration and tokenizer layers are reusable across these scenarios with the same probe sets.

## 4. Retrospective Validation

## 4.1. What the retrospective sample tests, and what it does not

The 10 retrospective cases with known identities (six vendors: OpenAI, xAI, Xiaomi, Zhipu AI, NVIDIA, Mistral) test whether launch-time declared specifications (record pages) match the true models’ oficial specifications: that is, declaration consistency and version drift, the evidential foundation Stages 0–1 rest on. The sample is conditioned on having both an archived launch-time specification and a later known identity (Supplementary Material S1); cases lacking either are out of scope for Table 4. That filter can favor semi-disclosed releases: selection on verifiability, not a random draw from all stealth endpoints, so Table 4 estimates channel reliability under observable records, not population identification accuracy. Because most corpus cases were semi-disclosed (the record page itself named the true model) or oficially revealed, these cases do not exercise the identification pipeline under anonymity: the protocol ran blind against unknown identity only in the prospective flagship case (Section 5.1), supported by the collision ablation and the head-to-head comparison (Sections 3.3, 4.4). We state this scope explicitly; the retrospective sample validates the evidence channel, not end-to-end identification accuracy. The validation itself surfaced three method corrections, reported here as lessons:

1. v0.1 error: using current-listing models as comparison baseline systematically underestimated match rates (1/7 exact) because later models served as the comparison; v0.2 correction: compare against the true identity’s launch-time oficial specifications (thirdparty/oficial mixed), raising exact-match to 5/7.

2. v0.3 correction (first-party oficial sources): re-verified all specifications against first-party oficial sources (vendor docs, arXiv) where available.

3. v0.4 sample expansion (platform enumeration): systematic platform enumeration (Wayback; OpenRouter namespace; OpenCode Zen snapshots) added three cases (Andromeda Alpha, Bert-Nebulon Alpha, Pony Alpha record), N=7→10, and corrected the earlier claim that Pony had no record page (it does: 200K context, semi-disclosed).

## 4.2. Results

Final retrospective outcomes (N=10, one codename = one case): 7 exact matches, 2 precision-diferences, 1 partial match, 0 counterdirectional (Table 4).

Verdict rubric (applied uniformly). Comparisons are against the oficial declaration’s semantics, not a single numeric tolerance on raw integers. Exact: the record matches the vendor/platform oficial specification under that specification’s own rounding or banding (e.g., an oficial “1M” context covers a 1,048,576-class declaration; “1.8M” covers 1,840,000). Precision-diference: both sides publish exact integer windows that difer by <1% with no oficial rounded band that absorbs the gap (Quasar/Optimus: 1,048,576 vs GPT-4.1’s 1,047,576). Partial: identity direction is consistent, but a documented preview→production (or snapshot→card) drift changes a compared field (Polaris). Counter-directional: record points to a diferent vendor/family than the reveal (none in this sample).

Table 4: Retrospective declaration-consistency sample (N=10).
<table><tr><td>Codename</td><td>Record ctx</td><td>Official</td><td>Disclosure</td><td>Verdict</td></tr><tr><td>Quasar Alpha</td><td>1,048,576</td><td>GPT-4.1</td><td>Platform</td><td>Precision (0.1%)</td></tr><tr><td>Optimus Alpha</td><td>1,048,576</td><td>(1,047,576) GPT-4.1</td><td>Platform</td><td>Precision (0.1%)</td></tr><tr><td>Sonoma Dusk</td><td>2,000,000</td><td>(1,047,576) Grok 4 Fast</td><td>Semi</td><td>Exact</td></tr><tr><td>Alpha Sherlock Think</td><td>1,840,000</td><td>(2M) Grok 4.1 Fast</td><td>Semi</td><td>Exact</td></tr><tr><td>Alpha Hunter Alpha</td><td>1,048,576</td><td>(1.8M) MiMo-V2-Pro</td><td>Semi+V</td><td>Exact</td></tr><tr><td>Healer Alpha</td><td>262,144 +</td><td>(1M) MiMo-V2-Omni</td><td>Semi+V</td><td>Exact</td></tr><tr><td>Polaris Alpha</td><td>modality 262,144</td><td>GPT-5.1 (400K)</td><td>Semi</td><td>Partial (drift)</td></tr><tr><td>Pony Alpha</td><td>200K</td><td>GLM-5 (200K)</td><td>Semi+V</td><td>Exact</td></tr><tr><td>Andromeda Alpha</td><td>128K</td><td>Nemotron Nano 2 VL</td><td>Semi</td><td>Exact</td></tr><tr><td>Bert-Nebulon Alpha</td><td>256K</td><td>Mistral Large 3</td><td>Semi</td><td>Exact</td></tr></table>

Note. Verdicts follow the rubric above (not end-to-end identification under anonymity). Disclosure: Platform = platform reveal; Semi = record semi-disclosure; Semi+V = semi-disclosure plus vendor claim.

Zero counter-directional cases: no record pointed toward a wrong vendor. A second same-launch codename, Sonoma Sky Alpha (same true model as case 3, launched the same day), is excluded because no independent recordspecification snapshot was archived for it; including it as an eleventh case would double-count one specification match.

## 4.3. Candidate-pool shrinkage (Stage 1 quantification)

For the prospective flagship case (not one of the Table 4 rows), the Stage 1 matching procedure against the full catalog (N=422 models) was instrumented field-by-field (Fig. 3). The shrinkage funnel: context-window equality (1,048,576) retains 48/422 (11.4%); adding max-output equality (131,072) retains 6 (1.4%); adding the reasoning profile (mandatory reasoning, default efort = max) retains 3 (0.7%); the supported-parameters set is identical for all 3; and adding modality (text+image+video→text) yields a single-model candidate set, a 422→1 reduction (99.8% shrinkage) from configuration fields alone. Notably, the final filter that achieves uniqueness is modality, the signal we identify as scarce and time-invariant (F4): context and output ceilings are shared by several models, but the image+video input combination was unique in the catalog at that time point. The protocol therefore reports L2 (size 6) as the Stage-2 candidate pool (recall-preserving) and L4 uniqueness as an auxiliary strong signal with the declaration-lag caveat of Section 3.2. Total query cost of the full audit (both protocol sides, all probes) is on the order of 10<sup>2</sup> API calls and $\sim 1 0 ^ { 4 }$ prompt tokens, at the flagship’s free-window pricing, \$0; at the confirmed model’s list price, under a few cents. Attribution cost is thus dominated by latency, not spend.

![](images/ad27e9e0ffdd03889d782d449a67d950c5488e088e250d7351fe5a9ead758093.jpg)  
Figure 3: Stage 1 layered shrinkage for Ox Alpha against the 2026-08-23 Open-Router catalog snapshot (422 models). L2 is the conservative Stage-2 pool; L4 modality uniqueness is auxiliary.

## 4.4. Findings

• F1 (no counter-evidence): all 10 cases are same-vendor or compatible: the record-specification channel never pointed to a wrong vendor.

• F2 (declaration trustworthiness, revised): 9/10 launch-time records matched oficial specs → “declaration is truthful” holds within the same-version semantics; Polaris is the exception (snapshot 262K vs. production 400K) → declarations drift across versions.

• F3 (context inflation is a cross-generation artifact, not samegeneration): Polaris appears “inflated” only against later-generation comparisons (GPT-5.6’s 1.05M); against the same-generation GPT-5.1 oficial 400K it is a snapshot-evolution artifact. Specification signals are valid in launch-time semantics → supports Stage 0’s time-consistent baseline.

• F4 (modality is a scarce time-invariant signal): Healer’s full modality (text+image+audio+video) matched MiMo-V2-Omni’s oficial multimodal claim; modality is immune to context inflation and version drift, a more robust signal than context length (implemented in Stage 1).

• F5 (vendor claim texts are minable primary evidence): Xi aomi’s oficial release material explicitly stated “Hunter Alpha shown below is an early anonymous version of mimo-v2-pro”; vendor oficial materials are authoritative primary sources for stealth identity (“early anonymous version”/“early snapshot”/“stealth” as searchable phrases).

## 4.5. Head-to-head: response-based vs. configuration-based identification

We test whether deployment properties outperform response behavior on a controlled head-to-head, using LLMmap’s released eight-query apparatus and dataset (52 models, mid-2024). We ran the original queries (3 rounds each, 48 queries, zero errors) against GLM-5.3-Flash and GLM-5.3 on the vendor’s oficial endpoint.

On the response channel, single-run outputs were unstable: 14 of 16 model-probe pairs difered across three identical queries. Refusal style, answer format, and self-reported identity did not separate the two versions (both answered as “GLM by Z.ai”). The flash variant mis-stated its context window as 128K tokens (catalog and Stage 0 baseline: 1,310,720). LLMmap’s classifier is trained on 52 mostly ≤10B open models plus ten closed APIs, frozen before July 2024, with near-zero Chinese-vendor coverage, outside the 2025–2026 stealth corpus. Self-identification succeeded here only because the endpoints cooperated; the same probes elicited zero credible selfidentifications from anonymous Ox Alpha (\~300 community attempts).

On the configuration and tokenizer channel, Stage 1 split the versions via reasoning-mandatory flags; Stage 2 separated the family from non-candidates in \~10 queries each.

LLMmap’s published strength is cross-vendor identification within its training distribution (>95% on 42 models). Same-family version discrimination after a model cutof is a harder task. What this head-to-head shows is narrower than “configuration beats response” in general: response-style fingerprinting fails on post-cutof, out-of-distribution stealth models: instability, cooperative self-ID only, and no version separation, whereas configuration and tokenizer evidence remain usable without retraining. Indistribution LLMmap performance is outside this claim.

## 4.6. Negative controls: known backends and non-family collisions

Beyond the retrospective declaration sample, two control regimes bound false acceptance.

(C1) Known-backend re-verification. After the flagship reveal, the same Stage 2 probes were re-run against the now-public z-ai/glm-5.3-flash OpenRouter endpoint and the vendor-direct API (Section 5.1). Diferentials versus the archived anonymous measurements remained constant at 75 across short and long probes on both endpoints, showing that the Stage 2 criterion accepts the true backend when identity is known, and that the anonymouswindow fingerprint was not a stealth-only artifact.

(C2) Non-family rejection. The three collision endpoints in Section 3.3 / Fig. 2 / Table 3 are negative controls for the short-prompt failure mode: under a short-only rule they would be accepted; under the mandatory multilength rule they are rejected. Additional non-GLM candidates in the same archive (e.g., qwen-, ernie-, and mimo-line endpoints in the ox-full JSONL) likewise fail cross-length constancy and are not reported as family matches. Together, C1 and C2 show that Stage 2 is not a one-sided acceptor: it holds for the true family and rejects documented collisions.

## 5. Prospective Blind Validation

## 5.1. Flagship case: Ox Alpha (prediction → reveal)

The protocol was applied to a live anonymous model (Ox Alpha, Open-Router, launched 2026-08-20). On 2026-08-23, dated measurement artifacts (catalog snapshot, configuration fingerprint database, and Stage 2 JSONL under supplementary/evidence/) supported an attribution to Zhipu AI (Z.ai), GLM-5.3 version line, on three evidence streams (timeline: Fig. 4). No separate timestamped prediction-deposit file was created that day; the prospective claim is exactly what those artifacts encode. Deployment variant (flash/turbo/max) was not pre-asserted, consistent with contemporaneous community forensics that treated exact serving/quantization/variant as unproven (Shrey, 2026).

• Stage 1 (configuration): catalog-wide matching (422 models) reduced the pool to a unique cross-vendor configuration-equivalence class {ox-alpha, glm-5.3, glm-latest}, a Zhipu GLM version-line signal.

• Stage 2 (tokenizer): cross-length diferential test over five texts (short×3 + long Chinese + long English): constant diferential of 75 tokens vs. glm-5.3 and glm-5.2 across all texts: family-level identity, unique among non-GLM candidates (the three non-family collisions in §3.3 collapsed under the long-text condition). Version-line separation is from Stage 1, not from the tokenizer.

• Stage 3 (behavior): modality capability (image/video) + reasoning controls matched the GLM serving profile; knowledge-boundary tests suggested an earlier knowledge cutof (this signal later proved less precise than configuration evidence, see below). Modality was recorded as a scarce catalog signal, not as a registered Flash-variant claim.

Oficial reveal (2026-08-26): Zhipu AI confirmed Ox Alpha = GLM-5.3-Flash (a new-architecture multimodal variant of the GLM-5 generation), released weights, and the stealth entry was delisted. Family and version-line layers matched the 2026-08-23 inferences; Flash as the deployment variant is a post-reveal consistency, not a scored pre-assertion (Table 5):

Table 5: Prospective flagship case: 2026-08-23 analysis vs oficial reveal (Ox Alpha = GLM-5.3-Flash).
<table><tr><td>Layer</td><td>2026-08-23 analysis</td><td>Reveal</td><td>Verdict</td></tr><tr><td>Family / generation</td><td>Zhipu GLM</td><td>GLM-5.3-Flash (GLM-5 gen.)</td><td>Confirmed</td></tr><tr><td>Version line</td><td>GLM-5.3</td><td>GLM-5.3-Flash</td><td>Confirmed</td></tr><tr><td>Variant</td><td>not pre-asserted</td><td>Flash (native</td><td>Post-reveal</td></tr><tr><td></td><td></td><td>multimodal)</td><td>consistency only</td></tr></table>

![](images/1821fd6f152e1a9afb99313987a74f7a16194faa1141520733847efec1d9cb5d.jpg)  
Figure 4: Launch → 2026-08-23 analysis prediction → oficial reveal → postreveal re-verification, with Stage 0 context drift called out.

Methodological lessons from the reveal. (1) Configuration evidence outperformed knowledge-boundary evidence at version level: the community forensics campaign’s knowledge-boundary inference pointed to the base checkpoint (GLM-5), while our configuration evidence pointed to the version line confirmed by the reveal. We therefore rank knowledge boundaries below configuration and tokenizer evidence (revised Stage 3 ordering). (2) Tokenizer identity ̸= architecture identity: GLM-5.3-Flash shares the GLM-5- generation tokenizer but is a fully new architecture (linear+sparse attention hybrid, 18B active of 320B total); tokenizer identity established the family, not the checkpoint, consistent with our documented attribution boundary (§3.3). (3) Version drift between preview and production: the preview declared 1,048,576 context; the production model declares 1,310,720 (a Polarispattern replay). Stage 0’s time-consistent baseline is therefore not a formality; it is required for version-level inference. (4) Community cross-validation: an independent community forensics campaign (Shrey, 2026; 44-string differentials, 600+ requests) reached the same family-level conclusion through independent methods, and a second independent stylometry study (Du, 2026) likewise favored the GLM-5.3 line while noting it cannot distinguish a base checkpoint from post-training, a system prompt, or a shared serving stack. Both campaigns corroborate the family; our configuration layer provided the finer-grained (version-line) inference, and our tokenizer diferential addresses the serving-stack confound those studies self-report.

Post-reveal re-verification on the known model. Four days after the reveal (2026-08-27), we re-ran Stage 1–2 against two known endpoints: the now-public z-ai/glm-5.3-flash OpenRouter endpoint and the same model served directly on the vendor’s oficial API, and compared prompt-token counts with the archived pre-reveal measurements of the anonymous model (same probes, same platform, four days apart). The recheck used the four probes that had stable paired counts in the pre-reveal archive under the probe-stability rule (identity\_zh, identity\_en, long\_zh, long\_en); the fifth canonical short probe (cutof) was not re-paired because the post-reveal flash run did not repeat that wording, so we do not claim a five-of-five recheck. The diferential was constant at exactly 75 tokens across all four paired probes on both endpoints, including both long-text conditions (identity\_zh 102→27/27; identity\_en 102→27/27; long\_zh 198→123/123; long\_en 197→122/122), and every probe returned stable counts across repeated rounds. This closes a three-endpoint loop: anonymous (gateway) → public (gateway) → oficial (vendor-direct), with token-identical serving across all three: the fingerprints are stable across the anonymity boundary and are not artifacts of the stealth window or of the gateway’s serving configuration. The re-verification also surfaced a contrast at Stage 3: both public endpoints self-report as “Z.ai GLM” while the anonymous model suppressed self-identification under identical probing, confirming that self-identification suppression is a deliberate, revocable serving-layer policy rather than a model property (Section 1.2). Full log: supplementary evidence archive.

## 5.2. Long-running anonymous case: Big Pickle (two-outcome tracking)

A second live case, Big Pickle on OpenCode Zen, tests the protocol when ground truth may never arrive. Public Zen catalog timelines show a stealth slot occupied by Code Supernova from 2025-10-02, then rotated to Big Pickle from 2025-11-18; as of the 2026-08-28 audit window the endpoint had remained anonymous for more than eight months (Supplementary Material S1, cases 19–20).

Stage 0–1 case card (archived 2026-08-23; zero API key). The Zen-facing declaration used for comparison is a 200K context window. Against first-party oficial specifications available at that date:
<table><tr><td>Candidate</td><td>Official ctx</td><td>Match 200K</td></tr><tr><td>GLM-5</td><td>200K</td><td>yes</td></tr><tr><td>GLM-5.1</td><td>200K</td><td>yes</td></tr><tr><td>GLM-5.2</td><td>1M</td><td>no (unless Polaris-style drift)</td></tr><tr><td>Claude-family guesses</td><td>varied</td><td>inconclusive</td></tr></table>

Archived prediction (deposited 2026-08-24 03:16; target still anonymous; prospective, not post-reveal). Most-consistent pool: GLM-5 / GLM-5.1 family by configuration exclusion. A GLM-5.2 lead was downgraded on the 1M-vs-200K contradiction (weak without an independent drift archive). Confidence: low-to-medium; Stage 1 narrows; Stage 2–3 were not executed because the Zen channel did not expose usable prompt\_tokens usage fields to the auditor’s available access (data-availability bound). Full case card: Supplementary Material, Big Pickle prediction note.

Two-outcome design. (i) If later revealed, the archived prediction is scored against the reveal. (ii) If never revealed, the protocol’s value is the time-stamped evidence trail itself: candidate pool, excluded lines, confidence ceiling, and the explicit reason Stages 2–3 stopped. This is the operational meaning of cannot fully attribute under channel constraints, not a forced guess at a checkpoint.

## 5.3. “No-answer” demonstrations: attribution boundaries

To establish that the protocol declines rather than guesses, we ran Stage-0-only attributions on three delisted cases. Two produced graded, low-information hypotheses, not forced identifications: (i) Cypher (1M context; GPT-4.1 line hypothesis; medium confidence via exclusion); (ii) Owl (1M context; a directional inference later confirmed as LongCat-2.0). The remaining case is a deliberate decline: (iii) Aurora (128K context, cannot attribute, low confidence, correctly declined because 128K is a common configuration at that time point). No post-hoc evidence contradicted the Aurora decline. We report all three because they show the output discipline, graded hypothesis or cannot-attribute, rather than a closed-set guess.

## 6. Portability and Controlled Adversarial Experiments

The protocol’s configuration and tokenizer layers transfer to adjacent audit tasks where “declared vs. actual” divergence is the audit object. Beyond methodological transfer, we ran two controlled experiments that exercise the threat-model residual and the serving-switch use case (full logs: Supplementary Material S4; reproducible script under supplementary/experiments/).

## 6.1. Experiment E1 — Usage-forging proxy (A-evasive negative direction)

Setup. Declared family A = GLM-5.3; actual backend B = Kimi. The adversary forges prompt\_tokens on every probe to equal A’s honest counts (so Stage 2 sees $\Delta { = } 0$ vs an enrolled A reference) while returning B’s response text.

Ofline replay (archived multi-model JSONL, 2026-08-23): Stage 2 on the forged endpoint is constant at $\Delta { = } 0$ (ACCEPT same-tokenizer as declared). Without forgery, honest B vs A diferentials fall in 71–73 (REJECT). Stage 3 identity probes on B self-report as Kimi / Moonshot AI; marker mismatch and identity-content divergence both fire → cross-stage alarm = true.

Live OpenRouter (declared $\mathrm { z - a i / g l m \mathrm { - } 5 . 3 }$ , backend moonshotai/kimik2): same pattern; forged Stage 2 constant $\Delta { = } 0 ;$ honest backend diferentials non-constant (range –9. . . –3); Stage 3 returns Kimi/Moonshot selfidentification → cross-stage alarm = true.

Interpretation. Asymmetric usage forgery can fool Stage 2 alone; the protocol’s defense is the Stage 2–3 inconsistency alarm, not Stage 2 in isolation. Stages 0–2 still defeat non-cooperative self-identification suppression; defeating a full-stack usage forger requires retaining Stage 3 as a cross-check.

## 6.2. Experiment E2 — Silent serving switch (downgrade simulation)

Setup. An endpoint keeps a fixed declared label. Phase 1 serves premium backend A; phase 2 silently switches to cheaper backend B. The auditor holds an enrolled reference of A and monitors Stage 2 diferentials on a fixed probe set.

Ofline replay (A=glm-5.3, B=kimi-k3): phase-1 diferentials vs reference are constant 0; after switch, diferentials jump to 71–73 and lose constancy → switch detected = true.

Live OpenRouter $( \mathrm { A { = } z { \mathrm { - } } a i / g l m { \mathrm { - } } 5 . 3 } .$ , B=openai/gpt-4o-mini): phase-1 constant 0 across five probes (including long ZH/EN); phase-2 diferentials span –7. . . 23 (non-constant) → switch detected = true.

Interpretation. Longitudinal Stage 2 detects backend swaps that change token accounting, without vendor cooperation. This is a controlled simulation of silent downgrade / load-shedding substitution, not a scrape of a commercial ad-funded IDE, so product-specific routing policies remain out of scope. Combined with Section 5.1’s anonymity-boundary stability $( \Delta \mathrm { = } 7 5 $ across anonymous → public → vendor-direct), Stage 2 is shown to be both stable when the backend is fixed and sensitive when the backend changes.

## 6.3. Branded-model base attribution (methodological transfer)

Products increasingly ship with branding over third-party open-weight backbones. Stage 1–2 apply with a changed candidate pool (branded target vs open-weight catalog). We do not claim a branded-product case study here; Experiments E1–E2 and Section 4.5 already show the arithmetic an auditor would reuse.

## 7. Discussion

## 7.1. Relation to disclosure

Vendor and platform disclosures remain authoritative when present. The protocol fills the gap before disclosure: it can expose preview–production configuration drift (Stage 0; two cases here) and leave an auditable trail for adoption decisions. On the flagship case, community forensics and our protocol agreed on the family; configuration evidence reached the version line; the vendor reveal settled the remaining layers.

## 7.2. Limitations

Validation as an evidence pyramid, not a single accuracy number. End-to-end blind identification under anonymity is demonstrated on one flagship case (Ox Alpha; Section 5.1). That case sits atop criterion-level evidence (cross-length collision ablation, Fig. 2), controlled negative checks (E1/E2; Section 6), declaration-channel validation (N=10; Section 4), and cannot-attribute demonstrations, not a substitute for a multi-model prospective panel. We claim a validated protocol design with this layered support, not a population identification rate.

Validation covers free-window stealth models only; the paid-stealth cell is empty. Confirmation of identity still depends on eventual disclosure; for never-revealed models the protocol can only emit graded outputs (by design). Platform-reported usage and catalogs can be transformed (Section 2.2); we use diferentials rather than absolute counts. Stage 3 is noisier than Stages 0–2 and was ranked downward after the flagship reveal. The prospective identification test is a single comprehensive case; the two-outcome Big Pickle design reduces, but does not remove, single-case dependence. For Ox Alpha, prospective support is encoded in dated measurement artifacts rather than a separately hashed public prediction deposit (Section 5.1); a weaker commitment than the protocol now recommends. Catalogs and terms change, so outputs are time-point valid only.

## 7.3. Implications

For auditors, the protocol is a due-diligence check before adopting an anonymous endpoint and a longitudinal monitor for serving changes. The deliverable is an evidence trail (snapshots, catalog matches, tokenizer diferentials, graded confidence), not a classifier score. Where vendors publish Model Cards or platform declarations, Stages 0–1 test whether the served model matches what was declared.

For platforms, anonymity does not hide configuration or tokenizer signals accessible through ordinary API use. Future work should expand platform coverage (Zen endpoints need usable usage fields), fill the paid-stealth cell, and, with institutional access, replicate E2-style switch detection inside production gateways.

## 8. Conclusion

Anonymous API endpoints are increasingly released without vendor attribution, yet adopters need auditable identity evidence for supply-chain and data-handling decisions. We specified a four-stage forensic protocol, validated declaration consistency on ten known-identity stealth releases, and prospectively confirmed a flagship case at family and version-line levels after oficial reveal (deployment variant not pre-asserted). Controlled replays show that forged usage fields trigger a cross-stage alarm and that silent backend switches surface in longitudinal Stage 2. A standard-library audit tool and evidence archive accompany the paper. When disclosure is absent or delayed, measured deployment properties still support due diligence: not certainty, but a traceable audit trail.

## 9. Availability

• Tool (audit-tool): Python implementation using the standard library only; API keys are read from environment variables and never logged or stored. Source is included in the supplementary archive. The accompanying CLI currently implements Stages 1–2 (catalog fingerprint + cross-length diferential) and Stage 3 probes; Stage 0 (Wayback reconstruction) is a documented procedure (Table 2; Supplementary S3), not a packaged crawler.

• Case corpus: the enumerated stealth releases (Table 4 subsample plus remaining cases) are listed as Supplementary Material S1.

• Auditor worksheet: portable forensic output template as Supplementary Material S2.

• Stage 0 extracts: Ox Alpha and Polaris launch-time vs later declarations as Supplementary Material S3.

• Controlled experiments: E1 (usage-forging proxy) and E2 (silent serving switch) logs and script as Supplementary Material S4 (evidence/experiments/, experiments/run\_controlled\_experiments.py).

• Evidence archive: dated 2026-08-23 measurement artifacts for Ox Alpha (catalog snapshot, configuration fingerprint database, Stage 2 JSONL): these encode the prospective GLM-5.3-line attribution; there is no separately hashed public prediction-deposit file for that day (a weaker commitment than the protocol now recommends; Section 3). The Big Pickle case card (evidence/big-pickle-pre diction-20260823.md, deposited 2026-08-24 while the endpoint remained anonymous) is retained as a true prospective record for the never-revealed / long-running design. Raw query logs (credentials removed), tokenizer-diferential measurements (Fig. 2 source: evidence/audit-oxfull-20260823-225759.jsonl; 16-probe archival set with probe-unstable style\_en rounds: evidence/probes-16/), record-page extracts, and post-reveal re-verification logs referenced in Sections 4–5 are included under supplementary/evidence/. A journal submission may attach the full archive as editorial supplementary material; the public deposit is archived at Zenodo with DOI https://doi.org/10.5281/zenodo.22210928.

• Figures: Fig. 1–4 are provided as vector PDF (Elsevier preferred) plus SVG sources; EM upload copies as Figure\_1.pdf–Figure\_4.pdf; 1000 dpi PNG also available.

## CRediT authorship contribution statement

Yisen Xi: Conceptualization; Methodology; Software; Investigation; Data curation; Writing – original draft; Writing – review & editing; Visualization.

## Declaration of competing interest

The author declares that he has no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used an AI-assisted writing environment (Cursor) for language polishing of the manuscript text. After using this tool, the author reviewed and edited the content as needed and takes full responsibility for the content of the published article. No generative AI tool was used to create or alter figures. No AI tool was used to generate experimental data.

## References

Ainsworth, S.G., AlSum, A., SalahEldeen, H., Weigle, M.C., Nelson, M.L., 2011. How much of the web is archived?, in: Proceedings of the 11th ACM/IEEE Joint Conference on Digital Libraries (JCDL 2011), pp. 133– 136. doi:10.1145/1998076.1998100.

Bao, G., Zhao, Y., Teng, Z., Yang, L., Zhang, Y., 2024. Fast-DetectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature, in: Proceedings of the International Conference on Learning Representations (ICLR 2024). arXiv:2310.05130.

Bruckner, T., 2026. One token is enough: Fingerprinting and verifying large language models from single-token output distributions arXiv:2607.10252.

Carlini, N., Paleka, D., Dvijotham, K.D., et al., 2024. Stealing part of a production language model, in: Proceedings of the 41st International Conference on Machine Learning (ICML 2024), pp. 5680–5705. arXiv:2403.06634.

Du, K., 2026. Ox-Alpha-Stylometry: An exploratory writing-fingerprint study of the anonymous stealth/ox-alpha model. GitHub repository. URL: https://github.com/ItsKaiwenDu/Ox-Alpha-Stylometry. accessed 31 August 2026; independent community stylometry study.

European Union, 2024. Regulation (EU) 2024/1689 of the European Parliament and of the Council of 13 June 2024 laying down harmonised rules on artificial intelligence (Artificial Intelligence Act), Art. 53, Annexes XI– XII. Oficial Journal of the European Union, OJ L, 2024/1689. URL: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX: 32024R1689.

Gehrmann, S., Strobelt, H., Rush, A.M., 2019. GLTR: Statistical detection and visualization of generated text, in: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 111–116. doi:10.18653/v1/P19-3019, arXiv:1906.04043.

Gu, T., Dolan-Gavitt, B., Garg, S., 2017. BadNets: Identifying vulnerabilities in the machine learning model supply chain, in: NeurIPS 2017 Workshop on Machine Learning and Computer Security. arXiv:1708.06733.

Hu, Z., Chen, L., Wu, X., Wu, Y., Zhang, H., Huang, H., 2024. Unbiased watermark for large language models, in: Proceedings of the International Conference on Learning Representations (ICLR 2024). arXiv:2310.10669.

Internet Archive, 2026. Wayback Machine snapshots of OpenRouter model record pages (Stage 0 launch-time configuration baseline). URL: https: //web.archive.org. accessed 28 August 2026.

Jagielski, M., Carlini, N., Berthelot, D., Kurakin, A., Papernot, N., 2020. High accuracy and high fidelity extraction of neural networks, in: Proceedings of the 29th USENIX Security Symposium (USENIX Security 20), pp. 1345–1362. URL: https://www.usenix.org/conference/usenixsecuri ty20/presentation/jagielski.

Kadavath, S., Conerly, T., Askell, A., et al., 2022. Language models (mostly) know what they know arXiv:2207.05221.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., Goldstein, T., 2023. A watermark for large language models, in: Proceedings of the 40th

International Conference on Machine Learning (ICML 2023), pp. 17061– 17084. arXiv:2301.10226.

Kudo, T., Richardson, J., 2018. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing, in: Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 66–71. doi:10.18653 /v1/D18-2012, arXiv:1808.06226.

Leshin, J., Shah, M., Timmis, I., Kang, D., 2026. Behavioral fingerprints for LLM endpoint stability and identity, in: Proceedings of the ACM Conference on AI and Agentic Systems (CAIS 2026), pp. 1327–1331. doi:10.1145/3786335.3813194, arXiv:2603.19022.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C.D., Finn, C., 2023. Detect-GPT: Zero-shot machine-generated text detection using probability curvature, in: Proceedings of the 40th International Conference on Machine Learning (ICML 2023), pp. 24950–24962. arXiv:2301.11305.

Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I.D., Gebru, T., 2019. Model cards for model reporting, in: Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* 2019), pp. 220–229. doi:10.1145/3287560.3287596, arXiv:1810.03993.

Nasery, A., Contente, E., Kaz, A., Viswanath, P., Oh, S., 2025. Are robust LLM fingerprints adversarially robust? arXiv:2509.26598.

NIST, 2006. Guide to integrating forensic techniques into incident response. Technical Report NIST Special Publication 800-86. National Institute of Standards and Technology. doi:10.6028/NIST.SP.800-86.

NTIA, 2024. Dual-use foundation models with widely available model weights: Report. Technical Report. U.S. Department of Commerce, National Telecommunications and Information Administration. URL: https: //www.ntia.gov/sites/default/files/publications/ntia-ai-ope n-model-report.pdf.

Ohm, M., Plate, H., Sykosch, A., Meier, M., 2020. Backstabber’s knife collection: A review of open source software supply chain attacks, in: Proceedings of the 17th International Conference on Detection of Intrusions

and Malware, and Vulnerability Assessment (DIMVA 2020), Springer. pp. 23–43. doi:10.1007/978-3-030-52683-2\_2, arXiv:2005.09535.

OpenAI, 2023. GPT-4 technical report arXiv:2303.08774.

OpenRouter, 2026. Platform records and API documentation used for Stage 0–1 configuration forensics. URL: https://openrouter.ai/docs/api/a pi-reference/models/get-models. accessed 28 August 2026; 2026-08 snapshot: 422 models in /api/v1/models.

Pasquini, D., Kornaropoulos, E.M., Ateniese, G., 2025. LLMmap: Fingerprinting for large language models, in: Proceedings of the 34th USENIX Security Symposium (USENIX Security 25), pp. 299–318. URL: https: //www.usenix.org/conference/usenixsecurity25/presentation/pa squini, arXiv:2407.15847.

Reed, J.R., 2026. Stealth AI models: How to evaluate hidden model identity. Personal blog, jonathanrreed.com. URL: https://jonathanrreed.com/ blog/stealth-models/. published 5 December 2025; updated 27 August 2026; accessed 28 August 2026; practitioner protocol, not peer-reviewed.

Richardeau, G., Dashyan, G., Le Merrer, E., Trédan, G., 2026. FLIPS: Instance-fingerprinting for LLMs via pseudo-random sequences arXiv:2606.03330.

Sennrich, R., Haddow, B., Birch, A., 2016. Neural machine translation of rare words with subword units, in: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1715–1725. doi:10.18653/v1/P16-1162, arXiv:1508.07909.

Shao, S., Li, Y., He, Y., Yao, H., Yang, W., Tao, D., Qin, Z., 2025. SoK: Large language model copyright auditing via fingerprinting arXiv:2508.19843.

Shao, S., Li, Y., Yao, H., Chen, Y., Yang, Y., Qin, Z., 2026. Reading between the lines: Towards reliable black-box LLM fingerprinting via zeroth-order gradient estimation, in: Proceedings of the ACM Web Conference 2026, pp. 2637–2648. doi:10.1145/3774904.3792196, arXiv:2510.06605.

Shokri, R., Stronati, M., Song, C., Shmatikov, V., 2017. Membership inference attacks against machine learning models, in: Proceedings of the

2017 IEEE Symposium on Security and Privacy (IEEE S&P), pp. 3–18. doi:10.1109/SP.2017.41.

Shostack, A., 2014. Threat Modeling: Designing for Security. Wiley.

Shrey, A., 2026. Ox Alpha identification: Black-box model forensics. GitHub repository. URL: https://github.com/LuD1161/ox-alpha-identific ation-public. accessed 28 August 2026; community forensics campaign; not peer-reviewed.

Szyller, S., Atli, B.G., Marchal, S., Asokan, N., 2021. DAWN: Dynamic adversarial watermarking of neural networks, in: Proceedings of the 29th ACM International Conference on Multimedia (MM 2021), pp. 4417–4425. doi:10.1145/3474085.3475591, arXiv:1906.00830.

Szyller, S., Zhang, R., Liu, J., Asokan, N., 2022. On the robustness of dataset inference arXiv:2210.13631.

Tramèr, F., Zhang, F., Juels, A., Reiter, M.K., Ristenpart, T., 2016. Stealing machine learning models via prediction APIs, in: Proceedings of the 25th USENIX Security Symposium (USENIX Security 16), pp. 601–618. URL: https://www.usenix.org/conference/usenixsecurity16/technical -sessions/presentation/tramer, arXiv:1609.02943.

Xu, Z., Yue, X., Wang, Z., et al., 2025. Copyright protection for large language models: A survey of methods, challenges, and trends arXiv:2508.11548.

Zhang, Y., Jiang, Y., Chen, Z., Backes, M., Shen, X., Zhang, Y., 2026b. Real money, fake models: Deceptive model claims in shadow APIs arXiv:2603.01919. preprint; CISPA Helmholtz Center for Information Security.

Zhang, Y., Zhang, Z.H., Qin, H., 2026a. Which model is actually serving you? IRIS: Budgeted black-box auditing of model substitution and routing dilution in LLM gateways arXiv:2607.20860.

Corpus statement. The phenomenon-level enumeration in the introduction (20 stealth releases, nine vendors, three distribution channels; Supplementary Material S1) was assembled from public sources only: platform record pages, Internet Archive snapshots, vendor release materials, and community documentation. Section 4 tabulates the 10 cases that have both a

launch-time specification record and a known identity; the remaining enumerated cases lack one of those two conditions and are out of scope for the declaration-consistency sample. No non-public data were used.