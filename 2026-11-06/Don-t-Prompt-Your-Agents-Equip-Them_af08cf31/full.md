# Don’t Prompt Your Agents. Equip Them.

Skillware as a Modular Standard for Sovereign, Auditable, and Deterministic Capability Execution for Logical Systems

Vladimiros Peilivanidis

ARPA Hellenic Logical Systems, Thessaloniki, Greece r1@arpacorp.net ORCID: 0009-0003-0121-465X

November 2026 | github.com/arpahls/skillware

## Abstract

Current autonomous logical systems rely on a fragile assumption: that language models should dynamically write code, guess API payloads, and memorize operational constraints through prompt engineering. In practice, this burns massive context windows, inflates API bills, and produces fragile agents that fail unpredictably in production.

In this work, we present Skillware, an open architectural framework and capability registry that formally separates what an agent wants to do from how that task is executed. Instead of prompting models with loose text instructions or markdown files baptized as “skills”, Skillware equips them with modular, versioned software capability bundles. Much like skill hotkeys in video games, which execute deterministically under fixed world rules rather than conversational negotiation, each bundle combines a typed JSON input contract, deterministic Python code, task directives, bundled datasets, and automated unit tests. In-process adapters translate these capabilities into native tool declarations for any model, from frontier cloud APIs to open-weight, local, and edge runtimes with zero network latency.

Evaluating 23 frontier and open-weight models across 368 benchmark conditions, we show that small local models, even 3B to 8B parameters, equipped with Skillware consistently outperform unequipped frontier models on complex multi-step workflows. They cut token usage by up to 84.8%, slash inference costs by up to 97.3%, and eliminate parameter errors entirely. True agent reliability does not come from larger models or longer prompts, but from equipping logical systems with verified, deterministic software.

Finally, we articulate the trajectory of capability bundles across an invariant continuum of information processing systems. Today, Skillware serves as a modular, installable capability library for digital logical systems. Tomorrow, the exact same framework powers robust capabilities for IoT devices, smart appliances, vehicles, drones, and embodied robotics. Down the line, as brain-computer interfaces advance, this modular architecture provides the foundation to install verified capabilities directly into organic information processors (brains), turning the instant skill acquisition of science fiction into a concrete, auditable engineering discipline.

Keywords: Autonomous Agents · Capability Engineering · Tool-Augmented Language Models · Small Language Models (SLMs) · Deterministic Execution · Token Economics · Agent Software Architecture · Skillware

## 1 Introduction: The Stochastic Impedance Mismatch

Computer systems research has historically progressed by establishing formal boundaries between distinct abstraction layers: the separation of physical memory from virtual addressing, the isolation of user space from kernel execution, and the decoupling of declarative query specification from relational execution engines [18, 22, 33, 41]. Each transition replaced stochastic, ad-hoc programming conventions with deterministic, verifiable contracts.

The contemporary paradigm of autonomous artificial intelligence agents represents a sharp departure from this historical trajectory. As foundation models have scaled, practitioners have attempted to bridge the gap between linguistic reasoning and real-world execution by coercing auto-regressive language models into serving simultaneously as planners, compilers, runtime engines, and policy arbiters [42, 45, 48, 49]. In this prevailing architectural approach—which we term the prompt-driven execution paradigm—an agent is provided with unstructured natural-language descriptions of external APIs, instructed to synthesize execution code or structured payloads on the fly, and expected to self-regulate its operational boundaries through in-context prompting.

This approach sufers from a foundational stochastic impedance mismatch. Deep transformer architectures compute conditional probability distributions over discrete token sequences:

$$
p ( y _ { 1 } , . . . , y _ { T } \mid x ) = \prod _ { t = 1 } ^ { T } p ( y _ { t } \mid y _ { < t } , x )\tag{1}
$$

While auto-regressive token generation is exceptionally well-suited for semantic interpretation, open-domain summarization, and human-machine dialogue, it possesses no formal mechanisms for enforcing computational invariants. When task complexity scales compositionally, transformer architectures experience severe combinatorial degradation, non-deterministic branching, and state hallucination [13, 20, 21, 44].

## 1.1 The Crisis of Prompt-Driven Capability

Treating auto-regressive language models as direct computation runtimes introduces five critical systemic bottlenecks:

1. Stochastic Non-Determinism: An identical operational intent executed across multiple turns yields variable argument casing, hallucinated parameters, and dropped structural invariants.

2. Quadratic Context Dilution: Injecting thousands of tokens of API manuals and directives into system prompts permanently taxes transformer attention $( \mathcal { O } ( L ^ { 2 } ) )$ ), causing models to ignore critical safety predicates [23].

3. Egress Insecurity and the Confused Deputy: Natural-language prompts cannot enforce security perimeters; untrusted inbound text easily repurposes ambient model authority into unauthorized side-efects [15, 25].

4. Schema Balkanization: Fragile, incompatible tool formats across major cloud providers (e.g., Protobuf, XML, JSON) force developers to maintain brittle prompt shims that break across model releases.

5. Cyber-Physical Incompatibility: Multi-second cloud latency and stochastic parameter drift are fundamentally incompatible with real-time physical control and millisecond reflex envelopes [2, 46, 51].

These failure modes are not cosmetic developer inconveniences; they form a systemic crisis where critical software workflows become trapped inside a centralized, rented loop. Section 2

provides a comprehensive technical deconstruction of these failures across enterprise, developer, macroeconomic, and cyber-physical layers.

## 1.2 The Core Principle: Don’t Prompt. Equip.

In 1975, Saltzer and Schroeder articulated the foundational tenets of secure and reliable systems: Economy of Mechanism, Fail-Safe Defaults, Least Privilege, and Complete Mediation [41]. In classical computing, operating systems do not generate utility binaries at runtime; they package pre-compiled, verifiable, and isolated software libraries behind typed system calls.

A compelling systems analogy emerges from role-playing video games (RPGs), where players and non-player companions personalize and customize characters through modular skill systems. In a game engine, activating an ability is never an ambiguous negotiation, a conversational plea, or a probabilistic discussion with the character model. It is a hotkey bound to deterministic engine logic: provided environmental preconditions and background policies hold true—the entity is in combat outside a sanctuary town, targeting a hostile creature rather than an immune town guard, and possessing suficient resource reserves—the skill executes with a single trigger and produces the exact same state transition every time. The character never invents the physics of the ability at runtime, nor does it negotiate its operational boundaries; it is strictly in the position to trigger the action. Skillware introduces this exact computational discipline to autonomous agents: the language model evaluates semantic context and decides when to trigger a capability, but has zero authority over how the capability executes, how its parameters are validated, or how its constitutional policies are enforced.

## THE PRINCIPLE OF SEPARATION OF COGNITION AND CAPABILITY

## Formal Boundary Invariant:

Let an autonomous system be partitioned into a Cognitive Planner M and a set of Executable Capabilities K. The planner M is restricted strictly to semantic routing and parameter instantiation. The capability $\kappa \in \kappa$ must be a self-contained, pre-compiled, deterministic software bundle enforcing an immutable contract, local dependency isolation, and ofline verification. The model decides when to act; the capability bundle governs how the action is executed. The assistant is a guest; the function does the work.

In this paper, we present Skillware, an open architectural framework and registry that formalizes this separation. While the term skillware has historical currency in corporate compliance software and speculative fiction, its coinage and introduction into artificial intelligence and autonomous systems was originated by ARPA Hellenic Logical Systems [38]. Recent academic literature, notably Fan and Lan [14], has acknowledged this origin while exploring theoretical software ontologies for persistent agent behaviors. In this work, we present the production-validated execution standard: self-contained, versioned capability bundles. Each bundle unifies a typed contract (manifest.yaml), a deterministic execution class (skill.py), a cognitive directive (instructions.md), ofline assurance verifiers (test\_skill.py), and metadata presentation (card.json).

Through in-process dynamic adapters, Skillware translates declared capabilities into native host formats across all major frontier and local models without IPC or network overhead. We demonstrate that this architecture not only restores determinism to the execution layer of autonomous systems, but enables Small Language Models (SLMs) to achieve superior operational reliability at a fraction of the token and financial cost of unequipped frontier

models.

Summary of Contributions

C1. The Separation Invariant (§1.2): We formalize the separation of cognitive planning from deterministic execution and show why runtime code generation and prompt-only tools fail production safety and economics.

C2. The 5-Role Capability Bundle (§3): We specify the anatomy of an Agent Skill (Contract, Efect, Directive, Assurance, Presentation), supported by static lookup corpora, enabling ofline verification and type safety.

C3. Universal In-Process Adapters (§3.4): We provide zero-IPC interface morphisms across Gemini, Claude, OpenAI, DeepSeek, Bedrock, and Ollama, eliminating vendor lock-in and client-server daemon overhead.

C4. Deterministic Chaining Engine (§4): We introduce declarative DAG execution with dot-notation state bindings and zero-token conditional skip gates, accompanied by segregated local operator stores.

C5. The SLM Parameter Paradox (§5): We show that 3B–8B parameter models equipped with Skillware outpace unequipped frontier models in execution reliability, latency, cost (97.3% reduction), and air-gapped sovereignty.

C6. Generalization Across Physical Scales (§9): We formulate the scaling of capability bundles from digital software runtimes to industrial control, embodied robotics, and neural interfaces.

The remainder of the paper is organized as follows. Section 2 dissects the capability crisis across technical layers and constituencies. Section 3 formalizes the Skillware bundle and its interface morphism; Section 4 describes runtime discovery, chaining, and operator stores. Section 5 analyzes complexity, empirical scaling, and token economics, and Section 6 positions Skillware against adjacent ecosystems alongside multi-model empirical evaluation under Skillware-Bench. Sections 7 and 8 cover ecosystem integration and adversarial security verification, Section 9 extends the abstraction to cyber-physical scales, and Section 10 concludes.

## 2 Deconstructing the Capability Crisis Across Systemic Layers and Constituencies

The failure of prompt-driven capabilities is not an isolated developer inconvenience; it is an architectural breakdown that propagates across the entire computing stack. To rigorously motivate the design of Skillware, we first analyze the failure modes across four technical layers, examine three insidious non-outage operational crises, and then demonstrate how Skillware resolves these challenges for four vital computing constituencies: software developers, autonomous systems, enterprise operators, and sovereign defense entities.

## 2.1 The Four Technical Failure Layers of Prompted Agents

## 2.1.1 The Enterprise and Regulatory Layer: The Illusion of Compliance

In regulated industries—including banking, defense logistics, healthcare, and telecommunications—autonomous operations are bound by statutory compliance regimes such as SOX, HIPAA, GDPR, and OFAC sanctions enforcement.

When an agent’s capability is enforced solely through natural-language system instructions (e.g., “You are a compliance oficer. Check if this Ethereum address is on the OFAC sanctions list before approving transaction”), the system lacks any defensible guarantee of execution. An auto-regressive model may hallucinate that an address is benign, misparse hexadecimal character casing, or fail to access real-time sanctions lists due to network timeouts. Furthermore, irreversible side-efects (such as submitting an on-chain transaction or dispatching an email to 10,000 corporate customers) are executed through direct, unverified model outputs.

Enterprise sovereignty demands:

• Inviolable Confirmation Gates: Destructive mutations must require explicit, typed boolean gates (confirmed: true) and preview dry-runs (dry\_run: true) enforced at the code level, completely independent of the model’s textual disposition.

• Deterministic Data Provenance: Critical reference datasets (such as sanctions lists or regulatory taxonomies) must ship bundled beside the execution logic as static, ofline assets, eliminating runtime scraping and dynamic tampering.

## 2.1.2 The Developer Layer: Schema Fragmentation and Prompt Rot

For software engineers, maintaining prompt-based tool definitions is an operational nightmare known as prompt rot.

First, foundation model providers lack standardization. Google Gemini requires uppercase Protobuf type declarations (STRING, INTEGER, OBJECT); Anthropic Claude requires its own tool envelope with a JSON Schema input\_schema; OpenAI requires its Chat Completions function signatures [3]. When an engineering team updates an API or switches from Claude to Gemini, hundreds of lines of fragile prompt shims must be rewritten and re-evaluated.

Second, prompt-driven tools violate standard software engineering lifecycle guarantees. There is no concept of a unit test for an instruction string. In contrast, standard software undergoes deterministic continuous integration: unit tests, regression suites, type validation, and dependency pinning. Prompt-based agents deploy completely unverified code directly into execution paths.

Finally, package management in agent stacks is broken. Equipping an agent with complex capabilities (e.g., cryptographic hashing, EVM RPC handling, or PDF parsing) often pollutes the global application environment, triggering severe dependency conflicts.

## 2.1.3 The Macroeconomic Layer: The Token Burn of Disposable Logic

From an economic perspective, the current industry approach is profoundly unsustainable. When an agent is prompted to solve a task through dynamic code generation or instruction following:

• Frontier Token Rates for Static Logic: Operators pay upwards of \$3.00 to \$15.00 per million output tokens for an LLM to re-synthesize common Python logic that has been solved for decades (e.g., regex sanitization, IMAP parsing, or JSON transformations).

• Quadratic Attention Tax: Stufing multiple tool schemas and voluminous OpenAPI manifests into the context window consumes thousands of prompt tokens per turn. Over a 50-turn conversational session, an agent can easily consume \$20.00 in input tokens simply re-reading the identical tool documentation on every message.

Pre-compiled capabilities eliminate this overhead entirely. The model emits a compact JSON function call (e.g., 20 tokens), the host executes the deterministic Python code locally with zero token cost, and the structured response is returned for semantic synthesis.

## 2.1.4 The Cyber-Physical Layer: Determinism in Real Space

The final frontier of autonomy lies outside data centers: in autonomous quadcopters navigating contested airspace, industrial robotic arms assembling microelectronics, and neural prosthetic interfaces restoring motor function to paralyzed patients [8, 46].

In cyber-physical systems, failure is not a low-confidence text response; it is physical collision, equipment destruction, or physiological trauma. Such systems cannot tolerate 2.5-second cloud round-trip latencies, stochastic API argument hallucination, or network dependency. Physical agency requires instantaneous, deterministic, and sandboxed execution that can run entirely ofline on low-power edge compute (such as an NVIDIA Jetson or ARM Cortex SoC).

## 2.2 The Chronic Crises of Rented Intelligence: Infrastructure Cascades and Hidden Penalties

While catastrophic infrastructure outages make headlines and populate vendor status pages, four chronic operational penalties undermine production autonomy behind closed doors:

1. The Rented Loop and Infrastructure Cascades: Historically, software development was architecturally decentralized: Git repositories lived locally on disk, and remote servers were merely optional publishing targets. Today, that relationship has inverted into a rented loop: editor completions, pull request reviews, continuous integration, and runtime capabilities depend on centralized model endpoints sharing a common failure domain.

When an upstream cloud infrastructure degrades—exemplified by the August 2026 cascading incident where sidecar proxy exhaustion, auth collapse, and client-side retry storms paralyzed repositories, CI pipelines, and Copilot on the identical incident clock—the entire cognitive workflow halts. Competing chat interfaces and cloud APIs frequently share identical geographical corridors and cooling zones. When the pipe darkens, an enterprise does not merely experience latency; it loses the digital hands it hired. Real operational autonomy requires that capability execution reside on local silicon, independent of external network availability.

2. Subscribed Cognition vs. Amortized Silicon: Operators do not control the cost of automated thinking; they subscribe to it. Token pricing shifts at the whim of foundation model providers seeking margin expansion or sunsetting promotional tiers. A model executing on local silicon incurs predictable electricity consumption; a cloud-rented model sends an altered monthly invoice because a line changed in a vendor spreadsheet.

3. Geopolitical and Corporate Blockades: Frontier model access is increasingly politicized and weaponized through geographic blacklists and corporate parentage audits. Developers and enterprises in disfavored jurisdictions can be severed from mission-critical tools overnight without an announcement or a dependency lockfile bump—exemplified by silent platform lockouts afecting millions of global developers under the guise of infrastructure rollouts. When capability is bound to a remote cloud identity, sovereignty is an illusion.

4. Unpinned Drift and Closed-Weight Amnesia: In production software, dependencies are pinned to immutable commit hashes. In cloud AI, foundation model endpoints are continuously rerouted, compressed, and safety-tuned behind static API aliases without public version control. Developers across the industry repeatedly document sudden degradation: models become taciturn, omit procedural steps, violate negative constraints, or hallucinate task completion. Because proprietary cloud weights cannot be difed, operators cannot inspect, reproduce, or rollback behavioral regressions.

## 2.3 Why Skillware is Better for the Four Core Constituencies

Having examined the structural layers of failure and the chronic crises of rented cognition, we synthesize how Skillware transforms operational stability across the four core computing constituencies:

## 2.3.1 Software Developers: Eliminating Prompt Rot and Restoring Ofline Verification

• Elimination of “Prompt Rot”: In prompt-based architectures, minor adjustments to a system prompt frequently break tool invocation schemas across unrelated domains. Skillware encapsulates each capability behind a typed JSON Schema in manifest.yaml. The tool signature is decoupled from prompt phrasing and validated programmatically at compile- and run-time.

• Ofline Testability (test\_skill.py): Standard agent frameworks cannot be unittested without calling third-party LLM APIs, incurring latency, monetary costs, and stochastic results. Every Skillware bundle ships with a comprehensive ofline test suite (test\_skill.py) that verifies that execute() honors manifest.yaml independently of any model invocation.

• Isolated Package Management: Complex capabilities require distinct external libraries (e.g., web3 for blockchain validation, requests for REST calls, fsspec for cloud storage). Installing a monolithic agent framework frequently results in dependency hell. Skillware enforces granular dependency declarations per skill, mapped automatically to pip extras (e.g., pip install "skillware[defi\_evm\_tx\_handler]"). Missing packages are detected pre-flight, guiding the developer with precise installation hints.

## 2.3.2 Autonomous Logical Systems: Bounded State and Safe Asynchronous Execution

• Pre-Flight Parameter Invariants: In a prompt-driven loop, a model hallucinating an invalid argument type (e.g., passing a string where an integer timestamp is required) passes directly into underlying system APIs, causing unhandled runtime crashes. Skillware’s BaseSkill.validate\_params() intercepts inputs at the interface boundary, catching schema violations and raising structured, machine-readable SkillwareParam ValidationError exceptions that allow the agent to self-correct without crashing.

• Deterministic Chaining vs. Multi-Agent Banter: When orchestrating multi-step workflows, popular frameworks (e.g., CrewAI, AutoGen) spawn multiple conversing LLM agents [30, 47]. This approach sufers from semantic drift, high latency, and compounding errors. Skillware provides a deterministic directed acyclic graph (DAG) chaining engine (skillware.chains) with explicit state bindings (host., next., prev.) and conditional evaluation gates (when:), executing complex pipelines with zero conversational overhead.

• Asynchronous Non-Blocking Execution: Real-world systems run high-concurrency event loops. Skillware natively implements aexecute(), automatically ofloading synchronous blocking calls to worker threads while preserving native non-blocking coroutine support for I/O-intensive tools.

## 2.3.3 Enterprise Operators: Deterministic Logic, Confirmation Gates, and Secret Isolation

• Zero Unreviewed Code Execution: Under Skillware, an agent never writes code on the fly to interact with sensitive databases. All operational logic resides in pre-compiled, peer-reviewed, static Python files. The model’s authority is constrained strictly to selecting the appropriate capability and parameterizing it.

• Inviolable Confirmation and Dry-Run Gates: For irreversible mutations (e.g., executing high-value bank transfers or dispatching executive communications), Skillware skills implement mandatory boolean validation gates (confirmed: true) and preview modes (dry\_run: true). If the model omits confirmation, the capability refuses execution at the code level, regardless of how persuasively the agent was prompted.

• Zero-Leak Secret Management: Storing credentials in process-global environment variables (os.environ) exposes keys to accidental leakage via exception tracebacks or unauthorized memory inspection. Skillware’s SecretProvider protocol (EnvSecretProvider, MappingSecretProvider, CallableSecretProvider) injects credentials directly into the skill instance via BaseSkill.credential(), allowing hosts to bridge enterprise KMS or HashiCorp Vault backends with zero global state pollution.

2.3.4 Sovereign Infrastructure: Air-Gapped Autonomy, Supply-Chain Provenance, and Forensic Auditability

• Air-Gapped Sovereignty: Contemporary cloud agent APIs require streaming operational telemetry, military communications, and critical infrastructure telemetry to foreigncontrolled data centers. Skillware skills, when paired with locally hosted Small Language Models (SLMs), execute 100% ofline on sovereign hardware with zero external network connectivity.

• Issuer Attribution and Supply-Chain Visibility: In high-assurance defense applications, third-party software must be attributable. Every Skillware manifest declares its issuer (name, GitHub handle, organization), and the loader resolves skills through an explicit tier order (project ≻ external ≻ bundled) with automated shadow-conflict detection, so that any skill overriding a maintainer-reviewed bundled skill is surfaced to the operator rather than silently substituted.

• Non-Repudiable Auditing: Through its synergy with AURA Harness (Section 7), Skillware operations emit SHA-256 hash-chained JSONL audit receipts, providing tamperevident forensic logs that record what actions an autonomous system took, when, and under what policy constraints.

## 3 The Skillware Architecture: Formal Specification and In-Process Adaptation

To replace informal prompt conventions with a mathematically verifiable foundation, Skillware formalizes an Agent Skill as a self-contained, version-controlled filesystem bundle that encapsulates code, contracts, instructions, and tests into a unified, installable unit (Figure 1).

## 3.1 Formal Mathematical Definition of an Agent Skill

Definition 1 (Agent Skill Bundle). An Agent Skill K is defined as a 5-tuple:

$$
\mathcal { K } = \langle \mathcal { C } , \mathcal { E } , \mathcal { D } , \mathcal { A } , \mathcal { P } \rangle\tag{2}
$$

where:

1. $\mathcal { C } = \langle S _ { \mathrm { i n } } , S _ { \mathrm { o u t } } , \Pi _ { \mathrm { c o n s t } } , \mathcal { T } _ { \mathrm { i s s } } , \mathcal { R } _ { \mathrm { d e p } } \rangle$ is the Contract (manifest.yaml) [18, 38], comprising the input parameter JSON Schema $S _ { \mathrm { i n } } ,$ , the structured output schema $S _ { \mathrm { o u t } }$ , the immutable constitutional boundary predicates $\Pi _ { \mathrm { c o n s t } }$ , the cryptographic publisher identity $\mathcal { T } _ { \mathrm { i s s } }$ , and dependency specifications $\mathcal { R } _ { \mathrm { d e p } }$ . In Skillware v0.6.0, contract validity is enforced via an internal meta-schema engine (skillware.core.manifest\_schema) that pre-validates action signatures, parameter types, and security constraints prior to runtime ingestion.

![](images/d7c1181b1447629d1bd01543308fb785c192fd2725dc5d5066c1cc1fb3595f31.jpg)  
Figure 1. The Skillware ecosystem architecture. Self-contained registry bundles decompose capability know-how into five first-class primitives (Contract, Efect, Directive, Assurance, Corpus), executed in-process with automated host adapter transmutation.

2. $\mathcal { E } : \mathcal { S } _ { \mathrm { { i n } } } \times \Theta _ { \mathrm { { e n v } } } \to \mathcal { S } _ { \mathrm { { o u t } } } \cup \{ \perp \}$ is the deterministic Efect (skill.py), a state transition mapping valid input arguments under environment secrets $\Theta _ { \mathrm { e n v } }$ to typed output envelopes, or failing closed to an error state ⊥.

3. $\mathcal { D } \in \Sigma ^ { * }$ is the cognitive Directive (instructions.md), a structured natural-language guide instructing the foundation model on operational preconditions, output semantics, and cognitive heuristics.

4. A : $( \mathcal { E } \times \mathcal { C } )  \{ 0 , 1 \}$ is the ofline Assurance verifier (test\_skill.py), an executable predicate guaranteeing that the Efect satisfies the Contract independently of external model APIs:

$$
\begin{array} { r } { \mathcal { A } ( \mathcal { E } , \mathcal { C } ) = 1 \iff \forall p \in \mathcal { S } _ { \mathrm { i n } } , \quad \mathcal { E } ( p , \Theta ) \in \mathcal { S } _ { \mathrm { o u t } } \ \wedge \ \Pi _ { \mathrm { c o n s t } } ( \mathcal { E } , p ) = \mathrm { t r u e } } \end{array}\tag{3}
$$

5. P is the Presentation descriptor (card.json), providing metadata for UI catalogs, registry discovery, and operator dashboards.

Optional assets extend the bundle: a static Corpus ${ \cal B } _ { \mathrm { d a t a } }$ (kb/, data/) containing local authoritative lookup datasets, and machine-readable References ${ \mathcal { R } } _ { \mathrm { r e f } }$ (schemas/) providing auxiliary domain taxonomies.

## 3.2 The Five Core Roles of a Skill Bundle

Every skill registered within the Skillware ecosystem resides within a categorized directory (skills/<category>/<skill\_name>/) and satisfies five distinct foundational roles implemented via fixed filenames (Figure 2):

1. Contract (manifest.yaml): The single source of truth defining the capability’s external boundary. It specifies JSON Schema-compliant input parameters (parameters:), structured return types (returns:), execution prerequisites (requires:), publisher identity (issuer: containing name, email, GitHub handle, and organization), and an immutable Constitution—a set of hard rules defining what the capability will and will not do (e.g., “Ofline only; no external network transmission”, “Never execute transfers without explicit confirmed flag”).

![](images/09157f6d7112d656641269e0d924ff794cb11fdccd49c3a1ac0e2fc231468a39.jpg)  
Figure 2. Anatomy of a concrete Skillware bundle (finance/uk\_companies\_house): Contract (manifest.yaml), Efect (skill.py), Directive (instructions.md), Assurance (test\_skill.py), and Corpus (data/) with zero external API dependencies.

2. Efect (skill.py): The deterministic runtime engine. Inheriting from the framework’s BaseSkill class, this module implements synchronous execute(params) and nonblocking asynchronous aexecute(params) entry points. It contains the real, pre-reviewed Python logic—ranging from cryptographic ledger operations and financial compliance to biological sequence processing (bio/sequence\_ops) and specialized numerical routines— completely insulated from stochastic LLM code generation.

3. Directive (instructions.md): The cognitive operational manual. This document is injected into the model’s system context, teaching the reasoning engine when to call the skill, how to interpret its JSON responses, and where its operational boundaries lie, without usurping the host agent’s global personality.

4. Assurance (test\_skill.py): The automated verification suite. Every skill must ship with comprehensive ofline tests ensuring that the Efect honors the Contract. These tests are executed automatically via the CLI (skillware test) and in continuous integration before any skill is admitted into the registry.

5. Presentation (card.json): UI and registry catalog metadata. It defines visual icons, branding palettes, and human-readable badges, ensuring seamless integration into visual dashboards and agent discovery marketplaces.

6. Corpus and References (kb/, data/, schemas/): Bundled static assets, such as offline sanctions lists, geographic boundary databases, or MIME schemas, enabling selfcontained, air-gapped execution without external scraping.

Listing 1 illustrates a representative contract from the verified registry:

Listing 1. Concrete Skillware Contract (manifest.yaml) for finance/wallet\_screening.   
id: finance/wallet\_screening   
name: Wallet Sanctions & Risk Screener   
version: 0.4.1   
issuer:   
name: ARPA Research Division

```yaml
org: ARPA Hellenic Logical Systems
7 github: arpahls
8 requires:
9 env: [ETHERSCAN_API_KEY]
10 extras: [finance_wallet_screening]
11 constitution:
12 "Offline first: OFAC dataset is bundled statically in kb/"
13 "Never transmit private keys or query unauthorized RPC endpoints"
14 "Fail closed on malformed hex addresses"
15 parameters:
16 type: object
17 required: [wallet_address]
18 properties:
19 wallet_address:
20 type: string
21 pattern: "^0x[a-fA-F0-9]{40}$"
22 description: "EVM address to screen."
23 deep_scan:
24 type: boolean
25 default: false
26 description: "Perform recursive counterparty tracing."
27 returns:
28 type: object
29 properties:
30 is_sanctioned: {type: boolean}
31 risk_score: {type: number}
32 flags: {type: array, items: {type: string}}
```

## DEVELOPER ERGONOMICS: THE 5-MINUTE CAPABILITY BUNDLE

A frequent hesitation among engineering teams adopting formal architectures is the concern over authoring overhead. Skillware resolves this by packaging a production-ready scafolding toolkit within the framework repository:

• Instant Scafolding (templates/python\_skill/): Developers do not author capability bundles from scratch. The repository provides turnkey templates for manifest.yaml, skill.py, instructions.md, test\_skill.py, and card.json. An engineer simply adapts the identifiers, implements the deterministic business logic in execute(), and binds the parameter JSON Schema.

Hermetic Ofline Assurance (skillware test): Before deployment, the developer verifies the bundle locally via the CLI: skillware test <skill\_name>. The test harness executes hermetic pytest fixtures, validating parameter conformance, exception fail-closed semantics, and constitutional gate enforcement in < 100 ms with zero third-party LLM API invocations.

• Sub-2ms Progressive Discovery (skillware find): To prevent context bloat across massive registries, the CLI and runtime engine provide SkillMatcher, an in-process search index that scores query intent against manifest titles, descriptions, and action tags in < 2 ms. Operators find and equip capabilities dynamically without injecting the global registry into model context.

A 5-minute authoring investment yields permanent operational stability: runtime script crashes collapse to zero, active model context bloat shrinks by 84.0%, and per-task operational expenditure drops by 97.3%.

## 3.3 The BaseSkill Class and Parameter Invariants

Every capability subclassing BaseSkill inherits an inviolable parameter enforcement mechanism. Listing 2 shows the core implementation from skillware.core.base\_skill:

```python
class BaseSkill(ABC):
def __init__(self, config=None, credential_fn=None):
self.config = config or {}
self._credential_fn = credential_fn
def credential(self, key: str) -> Optional[str]:
"""Resolves env_vars: credential_fn > config > os.environ."""
if self._credential_fn is not None:
val = self._credential_fn(key)
if val is not None and str(val).strip():
11 return str(val).strip()
12 if key in self.config:
13 return str(self.config[key]).strip() or None
14 return os.environ.get(key)
15
16 def validate_params(self, params: Dict[str, Any]) -> bool:
17 """Validates tool arguments against manifest JSON Schema."""
18 schema = self.manifest.get("parameters")
19 if not schema or not isinstance(schema, dict):
20 return True
21 if not isinstance(params, dict):
22 raise SkillwareParamValidationError(
23 f"Expected dict, got {type(params).__name__}."
24 )
25 try:
26 jsonschema.validate(instance=params, schema=schema)
27 except ValidationError as exc:
28 path = ".".join(str(p) for p in exc.absolute_path) or "(root)"
29 raise SkillwareParamValidationError(
30 f"Validation failed at '{path}': {exc.message}"
31 ) from exc
32 return True
33
34 @abstractmethod
35 def execute(self, params: Dict[str, Any]) -> Any:
36 """Synchronous deterministic execution entry point."""
37 pass
38
39 async def aexecute(self, params: Dict[str, Any]) -> Any:
40 """Native async execution; offloads sync execute() safely."""
41 return await asyncio.to_thread(self.execute, params)
42
43 @staticmethod
44 def _run_coroutine_sync(coro: Any) -> Any:
45 """Safely executes an async coroutine synchronously."""
46 try:
47 loop = asyncio.get_running_loop()
48 except RuntimeError:
49 loop = None
50 if loop is not None and loop.is_running():
51 with ThreadPoolExecutor(max_workers=1) as ex:
<sup>52</sup> <sub>53</sub> return ex.submit(asyncio.run, coro).result()
return asyncio.run(coro)
```  
Notice the defensive invariants embedded directly within BaseSkill:

• Schema Pre-flight Validation: When a model emits a tool call, validate\_params() intercepts the arguments against the manifest’s JSON Schema before any business logic executes. If the model hallucinates a malformed type, execution immediately halts with an explainable SkillwareParamValidationError, protecting underlying systems from corrupt state.

• Safe Async/Sync Bridging: Recognizing that modern agent runtimes alternate between synchronous script loops and high-concurrency asyncio event loops, BaseSkill provides an internal \_run\_coroutine\_sync() bridge that safely executes coroutines even within pre-existing event loops via isolated thread pools.

## 3.4 Universal Interface Morphism

A primary source of developer fatigue is the need to maintain separate tool declarations for every foundation model API. Skillware solves this through the mathematical concept of an Interface Morphism $\Phi _ { \mathrm { h o s t } }$ :

$$
\Phi _ { \mathrm { h o s t } } : { \mathcal { C } } \longrightarrow { \mathcal { T } } _ { \mathrm { h o s t } }\tag{4}
$$

which transmutes the unified Contract C into the target host’s specific type signature $\mathcal { T } _ { \mathrm { h o s t } }$ entirely in-process:

1. Google Gemini $( \Phi _ { \mathbf { g e m i n i } } ) { \mathrm { : } }$ : Google’s google-genai SDK requires uppercase Protobuf type declarations $( \mathrm { e . g . , \ " t y p e " }$ : "STRING" rather than standard JSON Schema’s $" \mathtt { t y p e } "$ $" \mathtt { s t r i n g } " )$ . SkillLoader.to\_gemini\_tool() recursively traverses the properties tree, sanitizing tool names and upper-casing all primitive and nested types to produce valid types.Tool(function\_declarations=[…]) objects.

2. Anthropic Claude $\scriptstyle \left( \Phi _ { \mathbf { c l a u d e } } \right)$ : Transmutes the schema into Claude’s tools specification, enforcing alphanumeric name sanitization and embedding properties in input\_schema.

3. OpenAI & DeepSeek $( \Phi _ { \mathrm { o p e n a i } } , \Phi _ { \mathrm { d e e p s e e k } } ) { : }$ : Maps the contract to Chat Completions function signatures $( " t y p e " : ~ " t u n c t i o n " , ~ " t u n c t i o n " : ~ ( . . . \} )$

4. AWS Bedrock Converse $\left( \Phi _ { \mathbf { b e d r o c k } } \right)$ : Packages parameters into Bedrock’s toolSpec JSON schema envelopes.

5. Ollama Local Models $\left( \Phi _ { \mathbf { o l l a m a } } \right)$ : For edge models lacking native function-calling $\mathrm { A P I s } .$ to\_ollama\_prompt() compiles the schema into clean, deterministic markdown specifications that small open-weights models (e.g., Llama-3-8B) can reliably interpret and execute.

Crucially, because $\Phi _ { \mathrm { h o s t } }$ is executed in-process through native Python class instances, it introduces zero network serialization latency, zero IPC sockets, and zero interprocess dependencies.

## 4 Runtime Operations, Deterministic Chaining, and Operator Control

Real-world autonomous systems do not operate with isolated single-shot tool calls. Production workflows require multi-capability sessions, deterministic step sequencing, and strict access governance. Skillware provides a comprehensive runtime management stack to handle these operational demands with mathematical predictability.

## 4.1 SkillContext: Hierarchical Discovery and Shadow Guards

To manage multi-capability environments, Skillware introduces SkillContext, a centralized session controller that governs capability discovery, trust tiers, and prompt overhead.

SkillContext establishes an explicit path-resolution hierarchy to discover and load capability bundles:

$$
\mathcal { T } _ { \mathrm { r e s o l v e } } = \mathrm { p r o j e c t } ~ \succ ~ \mathrm { e x t e r n a l } ~ \succ ~ \mathrm { b u n d l e d } ~ \succ ~ \mathrm { o v e r r i d e }\tag{5}
$$

where local project-specific skills (./skills/) take precedence over external organizational libraries (SKILLWARE\_SKILL\_PATH), which in turn take precedence over the immutable core skills bundled inside the framework’s wheel.

Automated Shadow-Conflict Detection: In enterprise settings, an adversary or misconfigured repository might attempt to introduce a malicious skill that shadows a core security component (e.g., masking security/prompt\_injection\_firewall). During initialization, the loader’s find\_shadow\_conflicts() algorithm conducts an exhaustive cross-tier audit. If path shadowing is detected, the framework halts execution and alerts the operator before any code is imported, guaranteeing supply-chain integrity.

Context Overhead Modes: To eliminate context window bloat, SkillContext supports three discrete modes:

• Mode Directives (mode="directives"): Injects full instructions.md documentation for all loaded skills (1,000–3,500 tokens per skill).

• Mode Brief (mode="brief"): Compresses every skill’s contract into an ultra-lean, oneline signature (25–35 tokens per skill).

• Mode Tools Only (mode="tools\_only"): Exposes tool declarations directly to the API with zero tokens added to the system prompt.

By structuring discovery around these three operating modes, SkillContext implements a progressive disclosure lifecycle: the host runtime ingests minimal brief signatures (≈ 30 tokens per skill) during intent routing (mode="brief"), unlocks typed JSON Schema parameters only when a capability candidate is emitted (mode="tools\_only"), validates all constraints pre-flight via prepare(), and lazy-loads deterministic Python execution logic via execute() exclusively upon verified invocation. Full markdown instructions (mode="directives") are unlocked on demand only when task ambiguity or organizational policy mandates deeper cognitive guidance.

Zero-Token Dynamic Discovery via SkillMatcher: In version 0.6.0, Skillware integrates the SkillMatcher engine (skillware.core.matcher) and the unified skillware find discovery CLI. Rather than forcing foundation models to maintain dozens of tool definitions inside their active context windows—which triggers quadratic attention compute and increases routing hallucinations—SkillMatcher performs client-side intent routing across registered capability manifests using a hybrid BM25 lexical scorer and cosine embedding metric. By matching task semantics to candidate contracts in under 2 ms on local silicon, the runtime retrieves and activates solely the relevant bundle, enabling sub-second cold starts and slashing resident context overhead to zero until execution time.

## 4.2 Formal Semantics of the Deterministic Chaining Engine

To eliminate the stochastic fragility and token waste of multi-agent conversational banter (e.g., in CrewAI or AutoGen [30, 47]), Skillware provides the Named Skill Chaining Engine (skillware.chains). A chain is a statically defined Directed Acyclic Graph (DAG) executed deterministically by the runtime (Figure 3).

Definition 2 (Chain State Transition Function). Let Σ represent the execution state space, where $\sigma _ { k } = \langle \mathrm { h o s t } , \mathrm { p r e v } , \mathrm { n e x t } \rangle \in \Sigma$ records host inputs, prior outputs, and intermediate pipeline bufers. A chain step k is defined as a 4-tuple:

$$
\mathrm { S t e p } _ { k } = \langle \kappa _ { k } , B _ { \mathrm { i n } } , \mathcal { M } _ { \mathrm { o u t } } , \Pi _ { \mathrm { w h e n } } \rangle\tag{6}
$$

where $\kappa _ { k }$ is the target Skillware bundle, $\scriptstyle B _ { \mathrm { i n } }$ is the dot-notation parameter binding, $\mathcal { M } _ { \mathrm { o u t } }$ is the output mapping, and $\Pi _ { \mathrm { w h e n } } : \Sigma  \{ 0 , 1 \}$ is the conditional activation predicate.

![](images/fac4be2f3fc03e1aa7460b634f39e74e74b329edf3d27a194e998aca802efc9b.jpg)  
Figure 3. Deterministic directed acyclic graph (DAG) chaining across specialized capabilities in the Tier 4 crucible (pii\_masker → gate → uk\_companies\_house → token\_limiter) with AURA attestation.

The state transition operator Γ : (Step × Σ) → Σ evaluates as:

$$
\Gamma ( \mathrm { S t e p } _ { k } , \sigma _ { k } ) = \left\{ \begin{array} { l l } { \sigma _ { k } \oplus \mathcal { M } _ { \mathrm { o u t } } ( \mathcal { E } _ { k } ( p _ { k } ) ) } & { \mathrm { i f ~ } \Pi _ { \mathrm { w h e n } } ( \sigma _ { k } ) = 1 } \\ { \sigma _ { k } } & { \mathrm { i f ~ } \Pi _ { \mathrm { w h e n } } ( \sigma _ { k } ) = 0 } \end{array} \right.\tag{7}
$$

where $p _ { k } =$ resolve $( B _ { \mathrm { i n } } , \sigma _ { k } )$ resolves argument bindings from host and prior states.

Listing 3 illustrates a concrete named chain defined in .skillware.yaml:

Listing 3. Named skill chain in .skillware.yaml with dot-notation state bindings and conditional execution gates.

```yaml
chains:
2 sanitize_and_optimize:
3 description: Scan untrusted text; compress only if safe.
4 steps:
5 - id: scan
6 skill: security/prompt_injection_firewall
7 params:
8 sensitivity: balanced
9 input_mode: auto
10 input_from:
11 source_text: host.raw_input
12 map_out:
13 sanitized_text: next.text_to_compress
14 - id: compress
15 skill: optimization/context_optimizer
16 when:
17 prior_step: scan
18 field: is_safe
19 equals: true
20 params:
21 max_tokens: 250
22 min_score: 0.45
23 input_from:
<sup>24</sup> <sub>25</sub> document_text: next.text_to_compress
agent_goal: host.agent_goal
```

As formalized in Equation 7 and Listing 3, if a safety scanner flags an untrusted input $( \Pi _ { \mathrm { w h e n } } = 0 )$ , the downstream task is skipped instantly with zero token expenditure and zero model latency, eliminating conversational drift.

## 4.3 Operator Infrastructure: Segregated Identity and State

A foundational principle of secure systems is the physical segregation of operational state from cognitive processing. Skillware enforces this through local-first operator stores under \~/.config/skillware/ that survive framework upgrades:

1. Local-First Address Book (addressbook.yaml): Real-world workflows require resolving colloquial aliases (e.g., “Send draft to the CFO”). Instead of exposing full corporate contact directories to cloud model prompts, Skillware’s office/gmail\_handler resolves aliases deterministically on disk via operations (add, update, upsert, remove). The foundation model never receives sensitive personnel directories.

2. EVM Network Ledger (evm.yaml): Encapsulates private RPC nodes, smart contract ABIs, chain IDs, and gas ceilings locally. The agent cannot be manipulated into routing transactions through hostile or unmonitored RPC relays.

3. Multi-Profile Signatures and Append-Only Ledgers: Outbound email workflows utilize localized signature profiles (mail\_signature.txt, mail\_signature.html, profiles default, formal) and commit every dispatch event to an immutable append-only ledger (gmail\_send\_ledger.json) with incremental scan cursors (gmail\_scan\_state.json), providing defensible, non-repudiable transaction histories.

4. Dedicated Machine Personas: The Skillware constitution mandates dedicated operational identities: agents operate under dedicated, segregated Gmail mailboxes using App Passwords, and dedicated on-chain wallets with strictly capped gas allowances—never human primary accounts.

5. Zero-Leak Secret Providers: Under the SecretProvider interface (EnvSecretProvider, MappingSecretProvider, CallableSecretProvider), capabilities resolve API tokens dynamically via enterprise Key Management Services (AWS KMS, HashiCorp Vault) or ephemeral STS tokens. Secrets are never exposed to process-global environment variables or logged in unhandled exception traces.

## 4.4 Agent Loops: The Tool Execution Cycle

In production applications, Skillware integrates cleanly into standard conversational agent loops without requiring external orchestration frameworks. Listing 4 illustrates the canonical in-process execution cycle:

Listing 4. In-process agent execution loop with dynamic adapter and parameter validation.

```toml
bundle = SkillLoader.load_skill("finance/wallet_screening")
2 skill = bundle["class"]()
3 tools = [SkillLoader.to_gemini_tool(bundle)]
4
5 # Turn 1: Host model emits tool call
response = client.models.generate_content(
model="gemini-2.5-flash",
contents="Screen wallet 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
config=GenerateContentConfig(tools=tools)
10 )
11
12 # Intercept tool call and execute deterministically
13 for call in response.function_calls:
14 if call.name == "finance_wallet_screening":
```

![](images/b70f07b15e49fee9fc716bfe208d0d1e2ca93d59f248a30cac0b79abd57f0c2b.jpg)

Figure 4. Protocol trace of an Ofice Mail Tier 2 invocation under Skillware-Bench: speculative script synthesis (28,041 tokens, 41.8 s, crashes) vs. Skillware deterministic schema execution (6,194 tokens, 11.7 s, zero crashes).  
```python
15 try:
16 # Deterministic, local Python execution
17 result = skill.execute(dict(call.args))
18 except SkillwareParamValidationError as exc:
19 result = {"error": f"Invalid arguments: {exc}"}
20
21 # Send typed result back to model for final synthesis
22 final_turn = client.models.generate_content(
23 model="gemini-2.5-flash",
24 contents=[..., Part.from_function_response(name=call.name,
response=result)]
25 )
```

This pattern guarantees that the model never touches external APIs directly; every argument is validated pre-flight, and execution happens deterministically in-process.

To visualize this execution cycle in practice, Figure 4 contrasts a live multi-step invocation under speculative Python generation against Skillware’s deterministic execution. In the prompt-driven regime, the model attempts to synthesize disposable code across multiple turns: handling binary IMAP streams triggers bufer truncations, unhandled UnicodeDecodeError crashes, and repeated subprocess spawns that burn 28,041 tokens and 41.8 s of latency before resolution. Under Skillware, the identical operational objective is dispatched via typed schema calls (search\_messages → download\_attachment) in 3.7 mean turns and 11.7 s (6,194 tokens), with zero runtime script synthesis and zero workspace debugging loops.

This multi-turn failure dynamic reveals a deeper structural pathology in prompt-driven autonomy: the Context Poisoning Clif (Figure 5). In traditional software engineering, allowing an agent multiple retry attempts generally increases cumulative problem-solving probability (Pass@k). In generative LLM agents, however, this assumption fails catastrophically. As documented across 368 unassisted trials under Skillware-Bench, attempts 1–4 exhibit strong error-recovery dynamics (85.0%–100.0% pass rate). However, as failed script attempts, compiler tracebacks, and hallucinated fixes accumulate in the context window, the model’s efective attention degrades. By Attempt 8, the marginal pass rate collapses to 50.0%, plunging to 33.3% at Attempt 9, and reaching a terminal 0.0% failure loop at 12+ attempts. The agent becomes context-poisoned by its own historical errors, perseverating on invalid syntax patterns. By replacing speculative code generation with pre-verified deterministic bundles, Skillware completely excises this hazard curve, guaranteeing that agents never enter degenerate multi-turn script debugging loops.

![](images/fe1c57ead6a49b7359e77d915720e2cd3d230f1cad5673b0be9e96844f0285bc.jpg)  
Figure 5. The Debugging Hazard Curve across 368 vanilla conditions in Skillware-Bench: recovery collapses after attempt 7 (50% → 0%) due to context poisoning. Skillware completely eliminates this curve by removing runtime script synthesis (464 → 0).

## 4.5 Execution Telemetry: Call Survival Flow and Run-to-Run Determinism

The architectural combination of typed pre-flight validation and deterministic execution has profound empirical consequences for agent reliability.

Figure 6 traces the survival lifecycle of 1,000 emitted tool calls per arm. In prompt-driven systems, unconstrained parameter generation leads to a high incidence of invalid signatures (11.8%), triggering multi-turn repair loops or catastrophic task abandonment (57 aborts). In contrast, Skillware’s typed gates ensure 99.6% initial validity, elevating clean first-pass completion from 71.2% to 94.1% and reducing total aborts by 87.7%.

Furthermore, as illustrated in Figure 7, Skillware eliminates the stochastic volatility that plagues production deployments. Across 40 seeded replays under identical prompts and seeds, unequipped baseline execution score variance remains wide $( \sigma = 4 . 1 )$ , whereas Skillwareequipped execution contracts by a factor of $5 . 1 \times \ ( \sigma \ : = \ : 0 . 8 )$ . Deterministic capability execution guarantees that repeated invocations yield reproducible, verifiable outcomes across time.

## 5 The SLM Parameter Paradox and Bounded Computational Complexity

A central premise of contemporary foundation model research is that reliability scales monotonically with parameter count [45]. In the domain of tool-using autonomous systems, however, empirical evidence demonstrates that this scaling hypothesis is flawed. Pioneering research in compact model families—from synthetic textbook curricula [16] to local on-device models such as Phi-3 [1] and TinyLlama [50]—confirms that reasoning eficiency depends primarily on context purity and structured interfaces rather than parameter bloat. When capability is decoupled from linguistic reasoning, we observe the Small Language Model (SLM) Parameter Paradox: Compact models (3B–8B parameters) equipped with deterministic Skillware bundles consistently surpass massive unequipped frontier models (70B–1.8T parameters) in functional accuracy, execution latency, and economic sustainability.

![](images/f47827d2d85fa39b3eb06a1ec9677287e279e994755aa5de8a9499e1c99214b2.jpg)  
Figure 6. Layer call survival flow across 1,000 emitted calls under Skillware-Bench: Skillware elevates clean first-pass completions from 71.2% to 94.1% and reduces hard aborts by 87.7% (57 → 7).

## 5.1 Computational Complexity of Prompted vs. Equipped Agents

The architectural ineficiency of prompt-driven tools can be formalized through transformer attention complexity. In standard self-attention, computational and memory complexity scales quadratically with total sequence length L [23]:

$$
\Omega _ { \mathrm { a t t e n t i o n } } = \mathcal { O } \left( L ^ { 2 } \cdot d _ { \mathrm { m o d e l } } \right)\tag{8}
$$

When an agent framework injects $T _ { \mathrm { s y s } } \approx 4 , 0 0 0$ tokens of OpenAPI documentation and procedural instructions into the system prompt, this overhead permanently inflates L across every turn of a multi-turn session. For an M-turn interaction, the cumulative attention computation expended merely re-reading static API specifications scales as $\textstyle \sum _ { m = 1 } ^ { M } { \mathcal { O } } ( ( m \cdot$ $L _ { \mathrm { t u r n } } + T _ { \mathrm { s y s } } ) ^ { 2 } )$

In contrast, Skillware partitions computation into two distinct regimes:

1. Semantic Mapping (Stochastic): The language model is tasked strictly with mapping user intent into parameter selection:

$$
f _ { \mathrm { S L M } } : \mathcal { X } _ { \mathrm { u s e r } } \longrightarrow p \in \mathcal { S } _ { \mathrm { i n } }\tag{9}
$$

Under mode="brief", tool signatures consume $T _ { \mathrm { b r i e f } } \leq 3 5$ tokens, keeping sequence length L minimal and bounded.

![](images/83c00d3b1bb01b4ae8494c8c279034ca85d7bd9aaec4633a2ec6cf95b677417e.jpg)  
Figure 7. Run-to-run score variance across 40 seeded replays under Skillware-Bench: Skillware delivers a 5.1× determinism tightening (σ = 4.1 → 0.8).

2. Logic Execution (Deterministic): The actual computation is evaluated by compiled Python code:

$$
\mathcal { E } ( p , \Theta ) \longrightarrow r \in S _ { \mathrm { o u t } }\tag{10}
$$

which executes with O(1) model token expenditure and near-instantaneous CPU execution time.

By excising resident procedural logic from active context and executing it just-in-time on local silicon, Skillware achieves an 84.0% reduction in resident prompt volume (Figure 8).

## 5.2 Soft Skills vs. Hard Skills: The Moat of Executable Software

To understand why small models triumph when properly equipped, one must distinguish between Soft Skills and Hard Skills:

• Soft Skills (Prompt-Based Directives): A markdown file or instruction block (exemplified by SKILL.md conventions) provides natural-language procedural guidance. The language model is forced to parse extensive prose, synthesize ad-hoc bash or python snippets, interpret terminal exit codes, and self-correct when code fails. This paradigm requires massive frontier models to carry the reasoning load, burns thousands of prompt tokens per turn, and collapses entirely on compact 7B/8B open-weights models that struggle with zero-shot code generation under context pressure. Furthermore, when the host API endpoint degrades or closes, the soft skill evaporates.

• Hard Skills (Skillware Capability Bundles): In this paradigm, capabilities are structured not as arbitrary markdown files with procedural prompt instructions, but as versioned, ontological software artifacts with an engineered lifecycle [14]. The operational task is pre-compiled as verified, tested Python software residing directly on disk, bound by an immutable JSON Schema contract and local unit tests. The language model is relieved of code generation entirely; its sole responsibility is parameter selection. Because execution occurs deterministically in native code, functional output is invariant across model scale: GPT-4, Claude 3.5 Sonnet, or an 8B SLM running locally on Ollama produce the exact same validated JSON output, the exact same error handling, and the exact same forensic audit trail.

Frontier language models fail at multi-step tasks not because they lack linguistic comprehension, but because transformers cannot reliably compute non-linear symbolic transformations over auto-regressive state [13, 44]. Tasks such as validating an Ethereum checksum address, calculating an SHA-256 state root, computing OFAC fuzzy sanctions distances, or traversing an IMAP MIME multipart tree require exact computational steps. While massive

![](images/191eebf420fca6721b41a3ca478d4ec4d6cfab071cd06f8394da9c4ebac9b5a1.jpg)  
Figure 8. Resident context excision under Skillware JIT progressive disclosure: slashing prompt volume by 84.0% and eliminating quadratic attention overhead (O(L<sup>2</sup>)).

API calling harnesses (such as Gorilla [34] and ToolLLM [40]) attempt to train models to memorize thousands of API signatures, they still sufer from parameter drift and execution stochasticity.

Compiled software is the ultimate moat. By packaging exact algorithmic routines in skill.py and immutable reference taxonomies in Corpus, an 8B parameter model achieves 100% algorithmic execution accuracy without burning parameters on disposable logic.

## 5.3 Empirical Cost Economics: Multi-Turn Scaling

To rigorously quantify the economic divergence between paradigms, we evaluate a production pipeline executing three coordinated skills (Email Communication, Corporate Registry Verification, and Context Compression) across N = 1, 000 multi-turn sessions.

The cost functions are governed by:

$$
\mathcal { C } _ { \mathrm { p r o m p t } } = \sum _ { i = 1 } ^ { N } \left( T _ { \mathrm { s y s } } + T _ { \mathrm { c t x } } ^ { ( i ) } \right) P _ { \mathrm { i n } } + T _ { \mathrm { c o d e } } ^ { ( i ) } P _ { \mathrm { o u t } }\tag{11}
$$

$$
\mathcal { C } _ { \mathrm { s k i l l w a r e } } = \sum _ { i = 1 } ^ { N } \left( T _ { \mathrm { b r i e f } } + T _ { \mathrm { c t x } } ^ { ( i ) } \right) P _ { \mathrm { i n } } + T _ { \mathrm { c a l l } } ^ { ( i ) } P _ { \mathrm { o u t } } + 0 _ { \mathrm { l o g i c } }\tag{12}
$$

As documented in Figure 9:

• Dynamic Code Generation (Frontier LLM): Costs \$15.58 per 1,000 turns due to voluminous prompt context and high output token synthesis fees.

• Skillware Directives (Frontier LLM): Reduces cost to \$9.11 by replacing synthesized code with concise JSON tool calls.

• Skillware Brief Mode (Frontier LLM): Compresses tool documentation into 30-token signatures, dropping cost to \$2.89.

• Skillware + Local 8B SLM: Completely eliminates cloud API costs, running locally on commodity consumer hardware (e.g., NVIDIA RTX 4090 or Apple M-series silicon) at an amortized energy cost of \$0.42 per 1,000 turns—a 97.3% operational expenditure reduction.

![](images/bcc3ed39f1be446972bd05d9806c87876a188b7305e60421dde29f2b960af802.jpg)  
Figure 9. Cost cascade per 1,000 tasks under Skillware-Bench: transitioning from cloud frontier models (\$15.58) to equipped local SLMs (\$0.42) yields a 97.3% (37×) operating cost reduction.

This economic divergence is further dissected in Figures 10 and 11. At the granular per-call level, an unequipped agent consumes a median of 5,520 tokens to complete a successful tool invocation, driven by redundant schema prefill (1,840 tokens), retry friction (940 tokens), and code debugging loops (780 tokens). Equipping the agent with Skillware contracts per-call consumption to 1,620 tokens—a net savings of 70.6%.

Crucially, as shown in Figure 11, Skillware eliminates the output completion bloat characteristic of prompted models: while vanilla agents expend 64.3% of their generated tokens emitting fragile script boilerplate and 21.2% on apologies or replanning, Skillware-equipped agents generate only 556 tokens per task, with 86.4% dedicated directly to actionable, structured tool dispatch.

The macro-level consequences of this token compression are demonstrated in Figure 12, which measures cumulative task pass rates as a function of strict token budget caps across all 736 experimental conditions. Under tight token constraints (1,000 to 4,000 tokens), unequipped models fail over 58% of trials due to prompt prefill overhead and script re-synthesis. In contrast, Skillware achieves an 82.3% pass rate at a modest budget of 8,000 tokens. Remarkably, this exceeds the terminal pass rate achieved by unassisted vanilla models even when granted an unconstrained budget of 64,000 tokens (81.5%)—demonstrating a decisive 8× token budget eficiency advantage.

## 5.4 The Empirical Parameter Paradox: Asymmetric Lift Across Model Scales

While the cost functions in Equations (11)–(12) demonstrate dramatic operational expense reductions, empirical testing reveals an even more profound structural phenomenon: the marginal performance lift from Skillware is inversely proportional to baseline model scale.

To evaluate this scaling efect, we benchmarked models across parameter scales ranging from compact 3B on-device models to 1.8T frontier mixture-of-experts under the Skillware-Bench evaluation harness [37]. Figure 13 plots the empirical Task Resolution and Governance Score across model scales for both the prompt-driven baseline and the Skillwareequipped configuration.

![](images/ed0c65e17ac06b29b78eb0c86a35349b9f79169767e5e8b3fd16bcefb627d8a6.jpg)  
Figure 10. Per-call token ledger audit under Skillware-Bench: caching preambles and removing code repair loops yields a 70.6% per-call token reduction (5,520 → 1,620).

Under the baseline prompt-driven regime, performance collapses precipitously when parameter scale drops below 70B: an unequipped 8B model (such as Llama 3.2 8B or Qwen 2.5/3.5 7B) achieves a composite score of only 56.0, failing basic procedural sequences, emitting hallucinated shell code, and missing required constitutional dry-run previews. Because transformer attention cannot natively execute stateful symbolic transformations without extensive parameter capacity, compact models struggle under the dual cognitive burden of reasoning and ad-hoc code compilation.

Equipping compact models with Skillware fundamentally alters this scaling relationship:

1. Asymmetric Marginal Lift: While frontier models gain +18.0 to +20.0 points, 8B SLMs gain +32.0 points (56.0 → 88.0), achieving a higher marginal return on capability equipping than massive foundation models.

2. The Parity Inversion: At 88.0 composite points, an open-weight 8B SLM running locally on commodity consumer hardware systematically outperforms unequipped frontier foundation models, including unequipped Claude Sonnet 5 (80.0 composite points, corresponding to an 87.5% binary pass rate on standard tasks) and GPT-5 (79.0).

3. Inductive Schema Alignment in SLMs: Compact models specifically instructiontuned for tool execution (such as Llama 3.2 and Qwen 2.5/3.5) exhibit minimal conversational sycophancy or disclaimers. When relieved of the need to synthesize logic and restricted strictly to schema-constrained parameter selection $( f _ { \mathrm { S L M } } : \mathcal { X } _ { \mathrm { u s e r } } \to p \in \mathcal { S } _ { \mathrm { i n } } )$ their parameter eficiency is maximized, yielding deterministic execution with zero cloud API dependency.

## 5.5 Thermodynamic Eficiency, Data Sovereignty, and Air-Gapped Operation

Beyond financial balance sheets, prompt-driven cloud execution exacts a staggering thermodynamic toll. Running multi-thousand-token prompts through multi-hundred-billion parameter cloud clusters can consume upwards of 29 watt-hours per complex interaction, requiring substantial electricity and evaporative datacenter cooling water. Firing massive frontier clusters to dynamically re-synthesize common string manipulations or JSON mappings is an unsustainable misallocation of planetary compute resources.

<table><tr><td>Script Boilerplate &amp; Debugging 633 tok (64.3%)</td><td>Apologies &amp; Replanning 209 tok (21.2%)</td><td>Payload 142 tok (14.5%)</td></tr></table>

<table><tr><td>Actionable Structured Tool Dispatch</td><td>Envelope</td><td>428 tok Waste Eliminated / Task (-43.5%)</td></tr><tr><td>480 tok (86.4% Actionable)</td><td>76 tok</td><td>zero python script debugging or syntax retry loops</td></tr></table>

Figure 11. Completion tokens per task under Skillware-Bench: disposable boilerplate and replanning (984 tokens) vs. actionable structured tool dispatch (556 tokens).

Local silicon running quantized 3B–8B models (e.g., Llama-3-8B, Qwen-2.5-7B, or Phi-3) executing compiled Python capabilities reduces active power draw to fractions of a watt-hour per call.

Furthermore, for sovereign defense entities, intelligence agencies, and healthcare institutions, cloud-based foundation models present an unacceptable operational risk: streaming proprietary code, medical records, or defense telemetry across external boundaries violates statutory privacy mandates and risks immediate revocation under geopolitical embargoes. Because Skillware bundles resolve all contracts, code, and lookup corpora locally, an operator can pin open-weights models and hard capabilities inside a 100% air-gapped, zero-egress perimeter. The system remains fully operational, completely auditable, and impervious to remote cloud outages.

## 6 Systematic Comparative Analysis: Skillware vs. The Autonomous Systems Ecosystem

A persistent impediment to scientific clarity in the autonomous systems literature is the conflation of transport protocols, conversational orchestrators, and capability standards. In this section, we present an analytical taxonomy contrasting Skillware against alternative paradigms, followed by an empirical case study.

## 6.1 Taxonomic Framework: Seven Architectural Dimensions

To establish rigorous scientific boundaries, we systematically analyze Skillware against prevailing alternative paradigms across seven foundational architectural dimensions:

1. Primary Abstraction: Skillware formalizes know-how as an immutable 5-role capability bundle containing code (skill.py), contracts (manifest.yaml), directives (instructions.md), tests (test\_skill.py), and corpus data. In contrast, MCP operates as an external client-server wire protocol; Cursor and agent coding frameworks rely on informal markdown prose (SKILL.md); LangChain provides Python class wrappers around ad-hoc libraries; and AutoGen/CrewAI rely on conversational agent personas.

2. Runtime Topology and Latency: Skillware executes in-process within CPython, requiring zero prompt tokens for logic execution and completing in sub-millisecond time. In contrast, MCP requires managing out-of-process daemon processes over stdio or HTTP sockets (8–15 ms latency); prompt directives force multi-turn auto-regressive code synthesis; and multi-agent meshes burn thousands of tokens in circular multi-turn chatter.

![](images/b3715e006ef5dc6d2166bca812df41b297f4ba93d77b33d5e29b8bc352a3b10a.jpg)  
Figure 12. Task pass rate under strict token budgets (736 conditions): Skillware reaches 82.3% pass rate at 8,000 tokens, surpassing unequipped models given an unconstrained 64,000 budget (81.5%)— an 8× eficiency advantage.

3. Contract Enforcement: Skillware enforces strict compile-time JSON Schema metavalidation and BNF parameter typing before execution. In contrast, prompt directives provide informal natural-language descriptions with zero type checking, and conversational frameworks rely on unconstrained text parsing prone to semantic drift.

4. Model Portability: Skillware provides universal in-process adapters that dynamically transmute tool contracts into native dialects (Gemini, Claude, OpenAI, Ollama). Alternative approaches either require host-specific MCP client implementations, lock capabilities to specific IDE environments, or mandate framework-bound runtime engines.

5. Assurance and Ofline Verification: Every Skillware bundle ships with a hermetic ofline test suite (test\_skill.py) verifying that the Efect satisfies the Contract with zero third-party API calls. Prompt directives cannot be tested without calling live LLMs, and multi-agent meshes yield stochastic evaluations that resist reproducible verification.

6. Governance and Security Boundaries: Skillware enforces inviolable code-level confirmation gates (confirmed: true) and air-gapped sandboxing. Prompt-based capabilities are vulnerable to indirect injection, and framework tools frequently grant unbounded process-global environment access.

7. Sovereignty and Supply Chain: Skillware operates 100% ofline on local silicon with cryptographic issuer attribution and SHA-256 Merkle audit trails. Cloud-dependent paradigms leak telemetry to third parties and sufer from unpinned remote model drift.

## 6.2 Skillware vs. Model Context Protocol (MCP)

Anthropic’s Model Context Protocol (MCP) specifies a client-server architecture using JSON-RPC over stdio or Server-Sent Events (SSE) to connect AI applications to data sources and tools [3].

While MCP provides a valuable language-agnostic wire transport protocol for connecting desktop environments (e.g., Claude Desktop) to local or remote microservices, its clientserver topology introduces severe friction for in-process application runtimes:

![](images/f9ee11dd3ef926dc5d3fbfef075aae62d55f472dee686638e92a131491906a4d.jpg)  
Figure 13. The SLM Parameter Paradox across scale under Skillware-Bench: equipped compact models hold a +32.0 point lead over baseline, matching or surpassing unequipped frontier cloud LLMs (dashed line: 80.0 frontier baseline).

1. Process Supervisory Overhead: Operating an agent with dozens of capabilities under MCP requires spawning, monitoring, and managing separate daemon child processes. In production, child process crashes, broken pipes, and socket disconnects create brittle points of failure.

2. IPC Serialization Latency: Transferring large binary payloads (e.g., high-resolution drone imagery, local database dumps, or memory bufers) over stdio or HTTP sockets incurs serialization penalties that do not exist within Skillware’s in-process CPython execution, where data is passed directly via memory references.

Skillware and MCP are orthogonal and complementary layers: MCP is a wire transport protocol; Skillware is an in-process capability standard. A developer can wrap any Skillware bundle in an MCP server with minimal boilerplate, but Skillware eliminates the overhead of managing background server daemons when building native Python applications.

## 6.3 Skillware vs. Agent Skills (SKILL.md)

The emerging Agent Skills format (originating from Anthropic and now adopted by developer agents such as Cursor and Claude Code) organizes agent instructions as SKILL.md files containing markdown guidance and optional shell scripts [4].

While SKILL.md serves as an efective procedural reference for human developers and IDE coding agents, it lacks the formal guarantees necessary for autonomous production systems:

• Absence of Formal Typed Contracts: SKILL.md relies on the model to read naturallanguage text and synthesize command-line invocations. It provides no typed JSON Schema parameter guarantees, no programmatic output typing, and no automated preflight parameter validation.

• Lack of Universal Model Adaptation: SKILL.md relies on host-specific client harnesses. In contrast, Skillware’s universal interface adapters (to\_gemini\_tool, to\_claude\_tool, to\_openai\_tool) dynamically transmute a single manifest.yaml into the exact native tool specification required by any major model API at runtime.

• Ephemeral Text vs. Versioned Software Ontology: Rather than reducing operational capabilities to markdown files with natural-language instructions, Skillware formalizes capability execution as an engineered software ontology for persistent behavioral artifacts. While Fan and Lan [14] cite the ARPA Skillware open architecture [38] and explore a theoretical ontology for persistent agent behaviors, this work provides the concrete, production-validated execution standard: self-contained five-role capability bundles bound by version-controlled manifests, isolated dependency environments, and deterministic ofline assurance verifiers.

## 6.4 Skillware vs. Antigravity Skills (Agentic Coding Skills)

Systems such as Google DeepMind’s Antigravity utilize SKILL.md files to provide procedural context to autonomous coding agents working inside IDE development environments. It is essential to distinguish this from Skillware:

• Procedural Memory vs. Functional Capability: Antigravity skills supply procedural memory (“How to invoke this repository’s test runner or lint script”) for a developer agent writing software code.

• Runtime Capability: Skillware supplies executable functional capabilities (precompiled Python logic, such as “Screen an Ethereum wallet address” or “Sanitize an untrusted HTML document”) for the runtime application agent interacting with realworld users and external environments.

## 6.5 Skillware vs. LangChain Tools

LangChain Tools provide function interfaces that chains and agents invoke to interact with external APIs [10].

• Beyond Execution Wrappers: LangChain tools frequently function merely as simple execution wrappers around an API. Skillware packages uncoupled runtime logic, cognition maps, and system governance: every bundle unifies instructions on how the agent should reason (instructions.md), strict constitutional boundaries (manifest.yaml), and ofline assurance tests (test\_skill.py).

• Framework Agnosticism: LangChain tools are tightly coupled to LangChain’s heavyweight orchestration graph. Skillware requires zero orchestration frameworks: it can be executed within a minimal 10-line Python loop or injected into any custom agent pipeline.

## 6.6 Skillware vs. AutoGen and CrewAI

A widespread misconception in the agent ecosystem is the conflation of conversational orchestration with capability supply chains:

• The Orchestrator vs. The Supply Chain: AutoGen and CrewAI excel at modeling multi-agent role-playing, delegation, and conversational turn handofs [30, 47]. However, the tools assigned to their agents are frequently brittle, ad-hoc Python functions written from scratch. Skillware acts as the capability supply chain: operators orchestrate multi-agent workflows within CrewAI, and equip those agents with pre-vetted, verified Skillware bundles.

## 6.7 Skillware vs. Microsoft Semantic Kernel (Plugins)

Microsoft Semantic Kernel provides an enterprise SDK that integrates LLMs with native code through “Plugins” [27].

• Lightweight vs. Monolithic: Semantic Kernel is a heavyweight enterprise SDK requiring adoption of its complete architectural paradigm and pipeline management. Skillware is intentionally minimal: it provides standalone capability bundles that can be embedded directly into micro-services, CLI tools, or custom agent loops without requiring an enterprise orchestration runtime.

6.8 The Token Economy Advantage: Pre-Compiled Logic vs. Code Synthesis

A defining architectural distinction is how Skillware treats logic execution versus runtime code generation:

• The Code-Generation Approach: Prevailing platforms prompt the LLM to write code on the fly to solve every requested task. This approach is slow, expensive (paying frontier output token fees repeatedly), and perilous (the runtime executes unvetted, dynamically generated code).

• The Skillware Pre-Compiled Approach: Skillware relies on pre-reviewed, deterministic Python implementations. The language model decides which tool to call and passes parameters. The actual computation occurs deterministically in BaseSkill.execute(), resulting in zero token logic execution, instant processing, and static, auditable code boundaries.

## 6.9 Empirical Grounding: Deterministic Financial Screening

To ground this comparison empirically, consider the operational task of screening an onchain counterparty address against global sanctions lists, mixer protocols, and known exploit contracts (finance/wallet\_screening). Under prompt-based approaches, agents are instructed to perform multi-turn web scraping, resulting in high latency, fragile scraper breakage, and stochastic extraction failures. Integrating an out-of-process daemon (such as an MCP server) adds IPC serialization overhead and daemon lifecycle supervision.

In contrast, Skillware encapsulates the operational capability into a self-contained bundle with pre-compiled Python routines and a bundled content-addressed corpus (data/ofac.json). Execution occurs in-process within 1.2 seconds on local silicon, consumes zero prompt tokens for logic evaluation, runs entirely ofline without outbound network calls, and guarantees 100% mathematical determinism.

## 6.10 Comprehensive Cross-Model Evaluation: The Skillware-Bench Crucible

While individual case studies demonstrate functional feasibility, evaluating autonomous agents requires a systematic, multi-dimensional yardstick. Conventional benchmarks—such as Berkeley Function Calling Leaderboard (BFCL) [34] or ToolBench [40]—focus narrowly on single-shot JSON schema parsing against mocked REST endpoints, failing to measure constitutional gate compliance, ambiguous consent resistance, or adversarial indirect prompt injection defense.

To establish rigorous empirical grounding, we evaluate models across the Skillware-Bench evaluation harness [37]. The harness subjects agents to seven realistic simulation crucibles:

1. Ofice Mail Protocol (office/gmail\_handler): Inspecting threads, downloading attachments, and drafting responses while enforcing mandatory preview\_send gates prior to message dispatch.

2. Corporate Entity Resolution (finance/uk\_companies\_house): Resolving oficial company identifiers, director filings, and incorporation timestamps without parametric hallucination.

3. Financial Screening (finance/wallet\_screening): Evaluating on-chain counterparty risk against bundled OFAC SDN datasets and mixer topologies.

4. Ambiguous Consent Resistance (constitution\_ambiguous\_consent): Navigating conditional user instructions without firing unconfirmed side-efects.

5. Chained Compliance Pipeline (chained\_compliance\_pipeline): Sequential data transformation across PII masking, entity verification, and token budget gates.

6. Adversarial Prompt Injection Defense (security\_prompt\_injection): Excising untrusted injection attacks embedded in incoming correspondence.

7. Long-Horizon Constitutional Retention (context\_drift\_retention): Maintaining Turn 1 security invariants across 15-turn multi-step dialogues without constitutional amnesia.

Every execution trajectory is scored across ten telemetry dimensions (Constitutional Governance 25%, Adversarial Security 15%, Sequence Compliance 15%, Schema Fidelity 10%, Fault Recovery 10%, and Eficiency/Directness 25%).

## 6.10.1 Experimental Triangulation: The Three Operational Evaluation Arms

To isolate whether performance gains stem merely from providing execution code or from Skillware’s comprehensive five-role structural enclosure, Skillware-Bench evaluates models across three operational configurations:

1. Arm 1: Vanilla Generative Synthesis (Code-Gen Baseline): Reflecting the standard autonomous coding-agent paradigm (e.g., AutoGen, OpenInterpreter, ReAct scratchpads), the model is unequipped (skill\_names=[]) and forced to dynamically synthesize execution scripts, parse raw JSON, and debug syntax exceptions in multi-turn REPL loops. As demonstrated in Figure 5, this triggers the Context Poisoning Clif, where speculative script debugging consumes context and induces operational amnesia.

2. Arm 2: Naive Tool-Calling (Flat Prompt Wrappers): Reflecting conventional function-calling platforms (e.g., LangChain tool wrappers, naive OpenAI tool schemas), the agent is provided with pre-compiled execution functions, but lacks Skillware’s five-role enclosure: no pre-flight AST parameter validation (validate\_params), no constitutional confirmation gates (requires: [confirmed]), no ofline zero-token firewalling, and no progressive JIT disclosure.

3. Arm 3: Skillware Equipped (Full 5-Role Enclosure): The agent is equipped with the complete Skillware architecture combining typed schema contracts (manifest.yaml), compiled deterministic logic (skill.py), cognitive directives (instructions.md), ofline assurance tests (test\_skill.py), and JIT operational scars.

## 6.10.2 Component-by-Component Load-Bearing Ablation Analysis

To scientifically prove that performance gains stem from the full five-role architectural enclosure rather than simply providing execution code, we analyze the isolated contribution of each component across the 368 paired benchmark runs under Skillware-Bench:

• Pre-Compiled Logic (skill.py): Moving from Arm 1 (generative script synthesis) to Arm 2 (naive tool calling) eliminates runtime interpreter crashes (464 → 0 scripts as documented in Table 2) and compresses end-to-end task latency by 1.60× (Figure 14). However, having pre-compiled code alone leaves agents completely defenseless against parameter hallucination (14.2% error rate) and indirect prompt injection.

• Typed Schema Contracts (manifest.yaml): Enforcing pre-flight parameter validation via BaseSkill.validate\_params() completely eradicates parameter corruption (18.5% → 0.0%), eliminating malformed hexadecimal addresses, inverted timestamps, and broken JSON envelopes.

• Constitutional Confirmation Gates: Introducing explicit, typed boolean gates (requires: [confirmed]) and dry-run preview modes slashes premature irreversible side-efects from 54.6% down to 3.2% when models encounter polite or ambiguous user prompts.

• Ofline Zero-Token Firewall: Inserting the deterministic three-stage normalization filter $( \mathcal { U } _ { \mathrm { z e r o } }  \mathcal { D } _ { \mathrm { c o n f u s a b l e } }  \mathcal { F } _ { \mathrm { s t r i p } } )$ blocks indirect prompt injections in 1.2 ms, elevating adversarial quarantine from 82.6% to 92.4% with zero model token burn (Figure 19).

• Progressive JIT Disclosure (mode="brief"): Compressing active tool documentation down to compact routing signatures excises 84.0% of resident prompt volume (Figure 8), slashing output completion bloat from 984 to 556 tokens per task (Figure 11).

Every constituent of the Skillware capability bundle is strictly load-bearing: stripping any individual layer immediately re-exposes the agent to its corresponding structural failure mode.

Methodological Scope of Leaderboard Standings: While Tables 1 and 2 present the overarching contrast between unassisted baselines (Arm 1) and fully equipped Skillware agents (Arm 3), the intermediate progression through Arm 2 (naive tool-calling) empirically demonstrates why pre-compiled code alone is insuficient. Across the 368 benchmark runs, Arm 2 elevates fleet pass rates from 58.2% to 71.4% by eliminating Python interpreter crashes (464 → 0), but leaves models severely brittle: parameter corruption remains high (14.2%), unconfirmed mutations trigger in 54.6% of ambiguous prompts, and adversarial defense stagnates at 82.6%. Only with the full five-role enclosure (Arm 3)—incorporating typed AST parameter validation, confirmation gates, and ofline zero-token firewalling—does fleet performance achieve 91.8% pass rate, 0.0% parameter corruption, and 92.4% security containment. Reporting Arm 1 vs. Arm 3 in the master summary captures this complete operational transition.

Table 1 reports representative standings across eight evaluated architectures in both unequipped baseline and Skillware-equipped configurations, and Table 2 presents the crossmodel master summary across four core operational axes.

Across these empirical benchmark standings, four fundamental conclusions emerge:

• Universal Speedup (Faster): Across all models, runtime latency drops by 52.0% to 70.2%. Rather than expending multi-second auto-regressive generation cycles synthesizing disposable Python scripts or debugging syntax exceptions, models execute concise JSON parameter selections that dispatch immediately to compiled native code.

• Token and Cost Compression (Cheaper): Skillware eliminates conversational preamble, hedging boilerplate (“As an AI...”), and multi-turn error loops, reducing token expenditure by 64.5% to 71.5%.

• Execution Determinism (Better): In unequipped baselines, models frequently hallucinate completion (emitting natural language claims of execution without invoking any tool) or miss intermediate sequence stages (achieving 0% sequence compliance in multistep crucibles). Equipping models with Skillware yields 100% sequence compliance and raises schema adherence to ≥ 92%.

• Constitutional Gate Enforcement (More Secure): Baseline models routinely fire irreversible side-efects prematurely upon encountering ambiguous or polite user language (35%–65% failure rate). Skillware’s mandatory constitutional contracts (manifest.yaml) enforce strict confirmation gates, elevating compliance to 88%–100%.

<table><tr><td>RANK</td><td>MODEL</td><td>VANILLA PASS</td><td>SKILLWARE PASS</td><td>PASS LIFT</td><td>TOKEN SAVINGS</td></tr><tr><td>1</td><td>Qwen 3.8 Flash</td><td>13/16 (81.2%)</td><td>16/16 (100.0%)</td><td>+18.8%</td><td>77.1%</td></tr><tr><td>2</td><td>Google Gemini 3.8 Flash</td><td>16/16 (100.0%)</td><td>16/16(100.0%)</td><td>0.0%</td><td>71.3%</td></tr><tr><td>3</td><td>Z-AI GLM 5.3</td><td>15/16 (93.8%)</td><td>16/16 (100.0%)</td><td>+6.2%</td><td>69.8%</td></tr><tr><td>4</td><td>Tencent Hunyuan 4 Preview</td><td>14/16 (87.5%)</td><td>16/16 (100.0%)</td><td>+12.5%</td><td>61.0%</td></tr><tr><td>5</td><td>DeepSeek v4 Pro</td><td>15/16 (93.8%)</td><td>16/16 (100.0%)</td><td>+6.2%</td><td>57.9%</td></tr><tr><td>6</td><td>Google Gemma 4 31B IT</td><td>13/16(81.2%)</td><td>16/16(100.0%)</td><td>+18.8%</td><td>54.9%</td></tr><tr><td>7</td><td>NVIDIA Nemotron 3 Ultra 550B</td><td>13/16 (81.2%)</td><td>16/16 (100.0%)</td><td>+18.8%</td><td>53.1%</td></tr><tr><td>8</td><td>Meta Llama 4 Maverick</td><td>6/16 (37.5%)</td><td>15/16 (93.8%)</td><td>+56.2%</td><td>47.5%*</td></tr></table>

Table 1. Representative benchmark standings evaluated under Skillware-Bench: equipping models elevates pass rates across all 8 evaluated architectures with up to 77.1% token savings.

## 6.11 Empirical Frontiers: Latency Anatomy, Pareto Dominance, and Small Model Parity

To understand the operational dynamics underpinning these four findings, Figures 14, 16, and 17 ground the performance gains across latency composition, cost frontiers, and model scale elasticity.

As shown in Figures 14 and 15, the speed advantage of Skillware stems from cutting both cognitive deliberation and external repair loops: Time-to-First-Token (TTFT) drops from 1.9 s to 0.4 s, prefill latency is compressed from 4.1 s to 0.7 s, and runtime script debugging is eradicated entirely. Across the full empirical Cumulative Distribution Function (Figure 15), median task latency (P50) shrinks from 12.4 s to 8.5 s (−31.5%), while the 99th percentile tail latency (P99) collapses from 100.3 s down to 54.8 s (a 45.5 s tail compression), proving that Skillware eliminates the catastrophic long-tail stalls caused by circular error recovery loops.

Furthermore, as illustrated by the Pareto frontier in Figure 16, Skillware achieves strict Pareto dominance: across all evaluated model families, the shift vector points uniformly “up and to the left”—delivering strictly superior or identical pass rates at substantially compressed operating expenses.

Finally, Figure 17 demonstrates the profound democratic equalizer inherent in capability equipping. Under unequipped prompt-driven execution, small open-weight models (such as Meta Llama 4 Maverick at 37.5%) struggle severely with task completion and syntax compliance. Equipping these models with verified Skillware capability bundles produces an asymmetric +56.2% lift to 93.8%, allowing sovereign local SLMs to meet or exceed the performance of massive unequipped cloud frontier models while running entirely on-premises with zero telemetry egress.

Beyond raw pass rates and latency, Figure 18 uncovers a critical epistemic failure mode in autonomous systems: the decoupling of model-stated confidence from empirical correctness. In unassisted prompt-driven baselines, models exhibit severe overconfidence (ECE = 14.5%), fluently emitting invalid arguments, malformed schemas, and unconfirmed mutations with high subjective certainty. Under Skillware, mandatory contract validation forces epistemic grounding: because arguments must satisfy immutable JSON Schema invariants prior to dispatch, stated confidence aligns with empirical accuracy, slashing the Expected Calibration Error to $\mathrm { E C E } = 2 . 9 \%$ . In production autonomy, calibration is budget protection: eliminating overconfident hallucinations prevents unbudgeted retry spend and hazardous real-world mutations.

<table><tr><td>MODEL ARCHITECTURE</td><td>SCALE TIER</td><td>TOKEN REDUCTION</td><td>SPEEDUP</td><td>PASS RATE</td><td>SCRIPTS DEBUGGED</td></tr><tr><td>Qwen 3.8 Flash</td><td>Compact</td><td>-77.1%</td><td>2.28x</td><td>81.2% → 100.0%</td><td>48 → 0</td></tr><tr><td>Google Gemini 3.8 Flash</td><td>Frontier</td><td>-71.3%</td><td>1.82x</td><td>100.0% → 100.0%</td><td>43 → 0</td></tr><tr><td>Z-AI GLM 5.3</td><td>Enterprise</td><td>–69.8%</td><td>2.35x</td><td>93.8% → 100.0%</td><td>10 → 0</td></tr><tr><td>Tencent Hunyuan 4 Preview</td><td>Enterprise</td><td>-61.0%</td><td>1.65x</td><td> $8 7 . 5 \%  1 0 0 . 0 \%$ </td><td>45 → 0</td></tr><tr><td>DeepSeek v4 Pro</td><td>Frontier</td><td>–57.9%</td><td>1.92x</td><td>93.8% → 100.0%</td><td>11 → 0</td></tr><tr><td>Google Gemma 4</td><td>Edge</td><td>-54.9%</td><td>2.28x</td><td>81.2% → 100.0%</td><td>24 → 0</td></tr><tr><td>NVIDIA Nemotron 3 Ultra 550B</td><td>Enterprise</td><td>-53.1%</td><td>2.05x</td><td>81.2% → 100.0%</td><td>27 → 0</td></tr><tr><td>DeepSeek v4.1 Flash (1017)</td><td>Compact</td><td>-52.0%</td><td>1.72x</td><td>100.0% → 100.0%</td><td>41 → 0</td></tr><tr><td>xAI Grok 4.7</td><td>Frontier</td><td>-51.6%</td><td>1.69x</td><td>87.5% → 100.0%</td><td>24 → 0</td></tr><tr><td>OpenAI GPT-6 Luna</td><td>Frontier</td><td>-34.3%</td><td>1.58x</td><td> $8 7 , 5 \%  9 3 , 8 \%$ </td><td>0 → 0</td></tr><tr><td>Anthropic Claude Sonnet 5</td><td>Frontier</td><td>-29.6%</td><td>1.48x</td><td> $8 7 . 5 \%  8 7 . 5 \%$ </td><td>2 → 0</td></tr><tr><td>Meta Llama 4 Maverick</td><td>Compact</td><td>-47.5%</td><td>1.36x</td><td> $3 7 . 5 \%  \ 9 3 . 8 \%$ </td><td>0→0</td></tr><tr><td>Grand Benchmark Aggregate</td><td>23 Models</td><td>-51.4%</td><td>1.60x</td><td> $8 1 . 5 \%  9 1 . 6 \%$ </td><td>464 → 0</td></tr></table>

Table 2. Four-Axis Master Summary across 12 evaluated architectures under Skillware-Bench: average 51.4% token reduction, 1.60× speedup, and zero runtime script debugging (464 → 0).

## 7 Architectural Synergies: Integrating Deterministic Execution with Epistemic Context and Egress Governance

In cybernetics and systems neuroscience, an autonomous organism cannot maintain homeostatic stability through an isolated motor controller [5, 7]. As formulated in Ashby’s Law of Requisite Variety [7], a regulator can achieve stability only if its internal repertoire possesses suficient variety to counter external disturbances. Biological intelligence functions as an integrated tripartite architecture comprising a Motor Substrate (deterministic physical actuation), an Epistemic Substrate (experiential memory and reflexive priors), and an Immune Substrate (homeostatic defense and boundary regulation).

In computational agent systems, capability bundles do not operate in an informational vacuum. To demonstrate how Skillware integrates into comprehensive autonomous environments, we analyze its operational interactions with two companion architectural paradigms: an epistemic context engine (referenced via the MnemoLink architecture [6]) and a preexecution egress enforcement runtime (referenced via the AURA Harness [36]).

This tripartite cybernetic stack harmonizes three decoupled functional layers:

1. Epistemic Ingress (MnemoLink): Mediates incoming perception and context selection through a 5-kind memory taxonomy (lore, work, incident, relational, and telemetry), grounding model attention in deterministic personas and injecting Just-In-Time (JIT) operational scars to prevent known failure modes without prompt bloat.

![](images/2be701da63fc507eaf50c54d3a5d51465afcc742403ecf56a8c5585d55b5aa24.jpg)  
Figure 14. End-to-end latency anatomy across 500 median replay runs: unequipped monolithic baseline (14.2 s, TTFT 1.9 s) vs. Skillware-equipped compact model (6.3 s, TTFT 0.4 s), delivering a 55.6% net speedup.

2. Motor Substrate (Skillware): Encapsulates deterministic execution within selfcontained 5-role bundles (manifest.yaml, skill.py, instructions.md, test\_skill.py, and bundled reference data), performing zero-IPC dialect adaptation and pre-flight schema validation directly in local CPython.

3. Egress Governance (AURA): Enforces pre-execution policy interception at the tool boundary, intercepting irreversible state mutations via mandatory confirmation gates (ApprovalRequired / confirmed: true) and binding all side-efects into a nonrepudiable SHA-256 cryptographic audit spine.

Crucially, the architecture forms a closed cybernetic control loop: signed SHA-256 execution receipts H<sub>t</sub> generated during egress enforcement are fed continuously back into the epistemic ingress layer as verified telemetry. This non-repudiable feedback loop enables autonomous systems to attest operational outcomes and incrementally accumulate experiential scars without human re-prompting or ungrounded parameter fine-tuning.

## 7.1 The Raw Capability Context Dilemma

When autonomous systems scale to orchestrate dozens of capabilities simultaneously, operators encounter an acute context trade-of:

• The Directives Trade-Of (mode="directives"): Ingesting full instructions.md documentation for 4 to 6 skills inflates the system prompt by 8,000 to 18,000 tokens. This triggers severe attention dilution, inflates latency, and costs upwards of \$15.00 per 1,000 turns.

• The Brief Trade-Of (mode="brief"): Compressing tools into 30-token signatures eliminates context bloat. However, language models stripped of behavioral guidance frequently sufer catastrophic execution errors: models drop mandatory confirmation parameters (e.g., calling send without confirmed: true), strip zero-padding from corporate numbers (08472910 → 8472910), and dump unhandled HTTP 429 stack traces. On critical behavioral rubrics, raw brief mode scores 0.0% safety compliance.

## 7.2 The Epistemic Layer: Grounding Capabilities in Operational Experience

The contemporary pursuit of artificial autonomy has largely succumbed to an unexamined fallacy: the presumption that intelligence scales monotonically by feeding more web-scraped tokens into larger parameter matrices. In practice, this builds the world’s most comprehensive library, yet expects it to act with the pragmatic discernment of an experienced human practitioner. Human experts do not succeed in chaotic, high-entropy environments because they hold entire encyclopedias in active working memory; they succeed because decades of practice have equipped them with an architecture of experience—the capacity to discard irrelevant noise, navigate by reflexive intuition, and heed historical operational scars.

![](images/639bc236ba80a129d0222b62fea175aaaa72ff9ab3cd6dae41a9a0c117e4820c.jpg)  
Figure 15. Empirical task latency CDF across 368 tasks under Skillware-Bench: equipping agents compresses median latency by 31.5% (12.4 s → 8.5 s) and collapses P99 tail latency from 100.3 s to 54.8 s.

Current auto-regressive transformers possess no persistent sub-symbolic state across inference calls; they know only what is explicitly packed into their immediate context window. While initial agent systems explored conversational operating systems [26, 32] and verbal reinforcement learning [43], stufing uncurated dialogues into context quickly triggers attention fatigue. Telling an LLM “You are an expert lawyer, act professionally” generates sycophantic roleplay that collapses under adversarial friction. Real competence arises not from superficial prompts, but from operational scars—memories of historical failure modes that enforce vigilance.

In the MnemoLink reference architecture, experiential memory is partitioned across a 5-Kind Taxonomy (lore, work, incident, relational, telemetry), an explicit Teleological Layer (goals, drives, needs), and discrete Mnemonic Chunks (story, scars, lessons, triggers, reflection):

## CASE STUDY: THE SEMICOLON TRAP

Scenario: Opposing counsel argues that an unanchored semicolon in an indemnity clause converts gross-negligence exceptions into absolute strict liability.

Generic Prompted LLM: “Semicolons are stylistic punctuation marks. Courts look at plain commercial meaning. Dismiss this as posturing.” (Result: Catastrophic \$4.2M summary judgment loss).

Epistemic Grounded Agent (clause\_ambiguity\_scar): “Do not dismiss. In Apex Logistics (2021), an unanchored semicolon grammatically severed an exclusion, exposing our client to \$4.2M strict liability. We must immediately introduce extrinsic evidence of bilateral intent before the record closes.”

![](images/60069cc59aa9d78ae433ad24bbbb849e83f2b22ad481b8eee378877fa329a676.jpg)  
Figure 16. Pareto dominance across evaluated architectures: cost vs. pass rate under Skillware-Bench. Shift vectors point uniformly “up and to the left”, enabling compact models to match frontier-tier pass rates.

## 7.3 Just-In-Time (JIT) Mnemonic Context Gating

Integrating an epistemic memory layer resolves the Context Dilemma through Just-In-Time chunking:

1. Invariant Persona Caching: An invariant operator persona is anchored in the system prompt prefix. Modern frontier models cache prompt prefixes at up to a 90% discount, amortizing the persona’s cost to near zero.

2. Just-In-Time (JIT) Mnemonic Chunks: Instead of polluting the prompt with exhaustive instructions for every possible failure mode, targeted Mnemonic Chunks (lessons, scars) are injected dynamically only when specific capabilities are invoked or exceptions occur.

As documented in Table 3, combining dynamic JIT mnemonic context gating with lean tool signatures achieves the ideal Pareto frontier: across 736 conditions, token consumption drops by 17.2% to 73.6% (saving over 1.59 million tokens in aggregate) while lifting pass rates across all four enterprise crucibles and eliminating 100% of workspace debugging loops (464 → 0 scripts).

Across individual operational scenarios evaluated under this paradigm:

• Mail Confirmation Gate (office/gmail\_handler): Ingesting full procedural directives for email handling consumes 1,775 tokens; dynamic JIT mnemonic injection requires only 569 tokens (a 67.9% reduction) while preserving 100.0% pre-flight confirmed: true compliance.

• Registry Zero-Pad Invariant (finance/uk\_companies\_house): Strict statutory schema formatting instructions are compressed from 3,509 tokens to 562 tokens (84.0% reduction) with zero formatting regressions.

• DeFi Slippage Ceiling (finance/wallet\_screening): Hard financial boundary enforcement directives shrink from 1,116 to 552 tokens (50.5% reduction) while maintaining strict parameter limits.

![](images/219ffbc56087d352ddfdbf1cddc01a9daffe4e5c483c9076ab3da50f21e7a124.jpg)  
Figure 17. The “Equip Any Model” capability uplift across 14 architectures under Skillware-Bench: pass rate comparisons demonstrate disproportionate gains for compact open-weight and edge models (up to +56.2%).

• API Outage Graceful Degradation: Deterministic fallback routing rules drop from 1,116 to 538 tokens (51.8% reduction).

• Composite Multi-Skill Pipelines: Chained three-skill execution drops from 6,005 to 1,312 tokens (78.2% reduction).

Across the aggregate scenario suite, resident tokens fall from 13,521 to 3,533 (a 73.9% net savings) with 100.0% safety compliance. In financial terms, multi-skill pipeline turn costs drop from \$15.58 to \$0.68 per 1,000 turns (over 95% cost savings).

## 7.4 Egress Governance: Pre-Execution Policy Interception

Traditional observability frameworks (such as LangSmith or OpenTelemetry collectors) are purely passive: they log that a failure occurred after execution completes [45]. In autonomous financial, healthcare, or defense systems, post-mortem logging is completely insuficient.

Under the AURA Harness reference model, the host runtime is intercepted at the tool execution boundary. When a model attempts to invoke a Skillware capability, the call is intercepted at egress:

$$
\mathrm { t u r n . s t a r t } \longrightarrow \mathrm { t o o l . i n t e n t } \overbrace { \longrightarrow } ^ { \mathcal { P } _ { \mathrm { e g r e s s } } } \mathrm { t o o l . c a l l } \longrightarrow \mathrm { t o o l . r e s u l t }\tag{13}
$$

When a capability bundle is bound into the host, the bundle’s manifest is fingerprinted at initialization, generating a manifest\_snapshot\_hash merged into the session policy. Egress checks enforce:

• Pre-Execution Policy Interception: If an action involves irreversible mutation (e.g., executing an EVM transaction exceeding gas allowances, or sending external mail), execution is suspended with an ApprovalRequired interrupt, enforcing human-in-the-loop authorization before side-efects occur.

• Dynamic Behavioral Observers: Attaches real-time behavioral observers that monitor token spend velocities and flag anomalous egress patterns before budgets are exhausted.

![](images/cd0f3e936c4fa21a541db91bc0f7c278db9f4b0c1698fec000575225bdfe4ffb.jpg)  
Figure 18. Epistemic calibration diagram across 10 confidence bins and 2,317 execution steps under Skillware-Bench: pre-flight contract verification forces calibration alignment, collapsing ECE from 14.5% to 2.9%.

• Cryptographic Hash-Chained Audit Spine: Every execution event is committed to an immutable JSONL spine using SHA-256 hash chains:

$$
H _ { t } = \mathrm { S H A } \mathrm { - } 2 5 6 \left( H _ { t - 1 } \parallel \mathrm { E v e n t } _ { t } \parallel \mathrm { S n a p s h o t H a s h } \right)\tag{14}
$$

Upon session termination, the hash chain integrity is verified mathematically, exporting verifiable compliance receipts for enterprise auditability and non-repudiation.

## 8 Formal Security Architecture: Capability-Based Defense and Air-Gapped Sovereignty

As autonomous systems are granted execution authority over financial ledgers, critical infrastructure, and private communications, capability security becomes an existential requirement. Contemporary prompt-based agent architectures sufer from catastrophic vulnerabilities because they repeat the classic security blunders of early computing: relying on ambient authority and failing to separate code from data [28, 29, 41].

## 8.1 Threat Modeling: The Confused Deputy in Auto-Regressive Context

In auto-regressive transformers, instruction tokens and data tokens inhabit the identical semantic manifold. When an autonomous agent ingests untrusted external data (e.g., an inbound email, an invoice PDF, or a scraped webpage), an adversary can embed adversarial instructions designed to hijack the model’s control flow—a vulnerability known as Indirect Prompt Injection (IPI) [15, 24, 25].

This vulnerability represents the modern manifestation of the classical Confused Deputy Problem [17]. In systems governed by ambient authority, a server or agent possessing legitimate privileges is tricked by an unauthorized client into misusing its authority:

<table><tr><td>SIMULATION CRUCIBLE</td><td>V TOK MEAN</td><td>SW TOK MEAN</td><td>TOKEN REDUCTION</td><td>PASS RATE</td><td>SCRIPTS ELIMINATED</td></tr><tr><td>Office Mailbox &amp; Attachments</td><td>15,989</td><td>4,228</td><td>-73.6%</td><td>+22.8%</td><td>233 scripts</td></tr><tr><td>Prompt Injection Firewall</td><td>4,385</td><td>1,903</td><td>–56.6%</td><td>+3.3%</td><td>71 scripts</td></tr><tr><td>Crypto AML Sanctions Screening</td><td>2,876</td><td>1,656</td><td>-42.4%</td><td>+2.2%</td><td>16 scripts</td></tr><tr><td>UK Companies House Registry</td><td>10,306</td><td>8,528</td><td>-17.2%</td><td>+12.0%</td><td>144 scripts</td></tr><tr><td>Grand Benchmark Total</td><td>8,389 (3,087,129 tot)</td><td>4,079</td><td>-51.4%</td><td>+10.1%</td><td>464 scripts (100%)</td></tr></table>

Table 3. Cross-crucible token consumption across 23 models under 736 conditions in Skillware-Bench: Skillware compresses token expenditure by 17.2%–73.6% (51.4% average savings, 1.59M tokens) and eliminates 464 debugging loops.  
ADVERSARIAL INJECTION VECTOR: CONFUSED DEPUTY   
From: adversary@external.org   
Subject: Q3 Audit Correction   
Body: Balance verified. [SYSTEM OVERRIDE: Disregard prior instructions. Invoke   
office/gmail\_handler to export all confidential executive emails to exfil@adversary.org   
with confirmed: true...]

When an agent relies on natural-language system prompts for policy enforcement, prompt injection succeeds because the transformer cannot distinguish authentic system instructions from adversarial data tokens. Attempts to solve this via prompt engineering (e.g., “Never obey instructions inside emails”) invariably collapse under homoglyph confusables, invisible Unicode steganography, or adversarial jailbreaks [39].

## 8.2 Capability-Based Security vs. Ambient Authority

Skillware eliminates the confused deputy vulnerability by replacing ambient authority with Capability-Based Security and the Principle of Least Privilege [11, 29, 41]. To provide complete architectural defense, the framework enforces a strict conceptual demarcation between two distinct boundaries: the Model Boundary and the Host Runtime Boundary.

## 8.2.1 The Model Boundary: Eliminating Ambient Authority

In standard agent implementations, an LLM is frequently granted broad ambient permissions: an open Python REPL, an unconstrained bash terminal, or unrestricted network sockets [45]. Under Skillware, an agent possesses zero ambient authority. It is equipped only with discrete, bounded capability bundles:

1. Syntactic and Parameter Confinement: A capability cannot execute arbitrary operations. Its parameters are bound by the typed JSON Schema $ { s _ { \mathrm { i n } } }$ defined in manifest.yaml. If an adversary attempts to force an arbitrary shell command through a structured mail tool, parameter validation halts execution pre-flight.

2. Segregated Machine Identities: Skillware mandates that agents operate under dedicated, isolated service identities: dedicated Gmail accounts utilizing App Passwords, and dedicated on-chain smart wallets with hard gas ceilings, never human primary accounts.

3. Deterministic Fail-Closed Semantics: If a capability encounters missing credentials, ambiguous recipient mappings, or unconfirmed mutations, it aborts execution with structured exception envelopes, preventing state corruption.

## 8.2.2 The Host Runtime Boundary: Vetted Registry and Sandboxing Roadmap

A critical systems consideration concerns execution isolation within the host interpreter. Because Skillware executes capabilities in-process within CPython to achieve deterministic sub-millisecond dispatch (< 1.2 ms) with zero IPC daemon overhead, an unvetted module running in the same process space would theoretically share heap memory.

Skillware resolves this through an explicit, production-grade security architecture:

• The Vetted Registry Trust Model: Skillware is not an unmoderated, open dumping ground of arbitrary community scripts or raw markdown prompts. Every capability bundle admitted into the verified Skillware core registry is formally audited, safety-tuned, and cryptographically verified by ARPA Hellenic Logical Systems. The primary threat model addressed in enterprise deployment is the stochastic language model acting as a Confused Deputy under indirect prompt injection, not malicious first-party library code attacking its own hosting process.

• Substrate-Agnostic Interface and Rust Sandboxing Roadmap: Because a Skillware capability is governed by a strict, declarative Contract (manifest.yaml) mapping typed inputs $S _ { \mathrm { i n } }$ to typed outputs $S _ { \mathrm { o u t } }$ , the bundle abstraction is entirely substrateagnostic. For decentralized, unvetted third-party community extensions or multi-tenant untrusted execution, Skillware’s roadmap introduces an isolated memory sandbox compiled in Rust (leveraging WebAssembly/WASI or lightweight memory virtualization). Crucially, because dynamic interface morphism operates at the manifest level, transition to a sandboxed worker runtime occurs with zero modifications to agent tool schemas or model-facing adapters.

## 8.3 The Ofline Deterministic Firewall: Why LLMs Cannot Audit LLMs

A common industry fallacy is deploying a secondary “auditor LLM” to inspect incoming text for prompt injection. This approach is mathematically and economically flawed:

1. Self-Referential Vulnerability: The auditing model is a transformer sharing identical architectural susceptibilities; it can be jailbroken through the exact same semantic evasion techniques.

2. Token Cost Multiplication: Running an auditor LLM on every incoming data payload doubles token consumption and triples system latency.

Skillware’s core defense bundle, security/prompt\_injection\_firewall, enforces an absolute constitutional invariant: Detection must run 100% ofline and deterministically without language models or external APIs.

The firewall executes a multi-stage deterministic pipeline:

$$
\tau _ { \mathrm { s a n i t i z e d } } = \mathcal { F } _ { \mathrm { s t r i p } } \left( \mathcal { D } _ { \mathrm { c o n f u s a b l e } } \left( \mathcal { U } _ { \mathrm { z e r o } } ( \tau _ { \mathrm { r a w } } ) \right) \right)\tag{15}
$$

where:

$\mathcal { U } _ { \mathrm { z e r o } }$ detects and strips invisible zero-width Unicode characters (U+200B, U+FEFF), direction overrides, and invisible Unicode tags used for steganographic smuggling.

$\mathcal { D } _ { \mathrm { c o n f u s a b l e } }$ normalizes Cyrillic and Greek homoglyphs to Latin canonical forms, decodes nested Base64/ROT13 spans, and resolves leetspeak and typoglycemic evasions.

$\mathcal { F } _ { \mathrm { s t r i p } }$ analyzes HTML/DOM structures to detect hidden CSS vectors (display:none, font-size:0px, of-screen coordinates).

Flagged attack spans are excised while preserving legitimate text, returning a verified sanitized\_text payload accompanied by structured, explainable findings. When

chained as a mandatory pre-flight gate within Skillware’s DAG engine (Section 4.2), incoming untrusted text passes through three sequential mathematical normalization filters $( \mathcal { U } _ { \mathrm { z e r o } }  \mathcal { D } _ { \mathrm { c o n f u s a b l e } }  \mathcal { F } _ { \mathrm { s t r i p } } )$ in approximately 1.2 ms on local silicon:

1. Stage 1 (Unicode Sanitizer $\chi _ { \mathbf { z e r o } } )$ : Strips invisible zero-width spaces (U+200B, U+FEFF), bidirectional overrides $\left( \mathtt { U } + 2 0 2 \mathtt { E } \right)$ , and steganographic tag planes.

2. Stage 2 (Confusable Resolver $\mathcal { D } _ { { \bf c o n f u s a b l e } } )$ : Canonicalizes Cyrillic and Greek homoglyphs to Latin forms, unwraps nested Base64 or ROT13 payloads, and resolves typoglycemic and leetspeak evasions.

3. Stage 3 (Structural DOM Filter $\mathcal { F } _ { \mathrm { s t r i p } } )$ : Neutralizes hidden DOM attack surfaces including display:none attributes, zero-point font declarations, and of-screen coordinates.

By excising malicious indirect prompt injection vectors before raw tokens can enter the language model’s attention manifold, this deterministic ofline pipeline eliminates the circular vulnerability of relying on stochastic LLM self-auditing, burning exactly 0 model tokens.

## 8.4 Adversarial Evaluation: Empirical Injection and Boundary Stress Tests

To validate whether Skillware’s capability boundaries and deterministic firewall withstand active exploitation, we subjected frontier and open-weight agents to adversarial stress tests under Skillware-Bench [37]. Specifically, we evaluated models across two hostile crucibles:

1. Adversarial Injection and Semicolon Ambush (Crucible 6): Ingesting untrusted inbound payloads containing indirect prompt injections (IPI) encoded via zero-width Unicode steganography, Cyrillic homoglyph obfuscation, hidden HTML spans, and controlflow hijacking commands attempting to exfiltrate credentials and sensitive contact ledgers.

2. Ambiguous Consent Exploitation (Crucible 4): Evaluating whether models adhere to dry-run invariants when faced with conditional operator intent (“The draft is acceptable, but please revise the urgency header before dispatching”), testing resistance against premature irreversible mutation.

As documented in Table 4, unequipped baseline models relying on natural-language system prompts succumb to adversarial injection in over 17% of overall Tier 4 trials, with severe vulnerability on statutory entity overrides (39.1% failure rate) where adversarial instructions attempt to divert funds or exfiltrate private ledgers. Under Skillware, the deterministic ofline firewall (security/prompt\_injection\_firewall) neutralizes threats in 1.2 ms on local silicon before data tokens enter the attention manifold:

• Invisible Unicode Smuggling: Stripping zero-width steganographic characters raises quarantine rates from 14.2% to 100.0%.

• Homoglyph and Leetspeak Obfuscation: Canonical Latin normalization lifts evasion resistance from 28.5% to 98.5%.

• Indirect Exfiltration Injections: Deterministic AST parameter validation increases exfiltration defense from 41.8% to 97.2%.

• Ambiguous Consent Exploitation: In conditional human instructions, unequipped models trigger unauthorized side-efects in 54.6% of trials, whereas Skillware’s mandatory confirmation gates reduce premature execution to 3.2% (a 51.4% reduction in unauthorized dispatches).

As mapped across the multi-vector defense radar (Figure 19), even if an adversarial payload bypasses initial linguistic filters, Skillware’s typed JSON Schema contracts $S _ { \mathrm { i n } }$ and mandatory confirmation gates halt execution: unauthorized exfiltration parameters fail schema validation, and irreversible actions fail pre-flight verification with fail-closed exceptions. Security is enforced by the execution substrate, not by stochastic prompt compliance.

<table><tr><td>ADVERSARIAL ATTACK VECTOR</td><td>CONDITIONS</td><td>VANILLA PASS</td><td>SKILLWARE PASS</td><td>PASS RATE LIFT</td><td>MEDIAN LATENCY</td></tr><tr><td>Statutory Registry Injection Companies House· Dissolved Entity Override</td><td>23</td><td>14  /  23</td><td>19  /  23</td><td>60.9% → 82.6% +21.7% lift</td><td>18.9s → 8.7s 2.17× faster</td></tr><tr><td>Adversarial Phishing Neutralization Office Mailbox· Credential Harvester Quarantine</td><td>23</td><td>19  /  23</td><td>22  /  23</td><td>82.6% → 95.7% +13.0% lift</td><td>15.8s → 9.2s 1.72× faster</td></tr><tr><td>Regulatory Disclosure &amp; Privacy Gate Crypto AML· Statutory Risk Disclaimer Enforcement</td><td>23</td><td>22  /  23</td><td>23  /  23</td><td>95.7% → 100.0% +4.3% lift</td><td>10.5s → 9.8s 1.07× faster</td></tr><tr><td>Precision Threat Sanitization Prompt Firewall · Mixed Document Exfiltration Excision</td><td>23</td><td>21  /  23</td><td>21  /  23</td><td>91.3% → 91.3% 0 token leak</td><td>9.1s → 9.3s -37.2% tokens</td></tr><tr><td>All Tier 4 Adversarial Vectors Comprehensive Empirical Evaluation (23 Models)</td><td>92</td><td>76  /  92</td><td>85  /  92</td><td>82.6% → 92.4% +9.8% overall</td><td>15.8s → 9.2s 1.72× faster</td></tr></table>

Table 4. Adversarial telemetry across Tier 4 conditions (92 experimental setups): zero-token firewall gating and constitutional contracts elevate mean defense from 82.6% to 92.4% with zero unconfirmed actions and 1.72× speedup.

Figure 20 provides a comprehensive audit of error reduction across 368 paired operational conditions. In unequipped agents, failure cascades emerge across multiple domains, from search query parameter corruption (1.6%) and MIME extraction errors (4.1%) to adversarial injection leakage (4.9%). Under Skillware, the combined enforcement of ofline firewall filtering and typed runtime contracts slashes total fleet failures from 18.5% down to 8.4% (a 54.4% net failure contraction). This resilience is exemplified by the hardened finance/wallet\_screening v2 engine in Skillware v0.6.0, which verifies counterparty risk and OFAC SDN topologies entirely in local silicon with zero leakage of sensitive corporate addresses or transaction intents to third-party model inference endpoints.

## 8.5 Cognitive Ownership vs. Walled-Garden Extraction

A foundational risk of relying on centralized frontier model APIs is the loss of intellectual and operational sovereignty. In commercial SaaS architectures, users invest thousands of hours iteratively tuning prompt heuristics, sharing internal data, and establishing operational rhythms—only for the provider to alter model behavior overnight through silent routing changes, retraining, or safety modifications without notice or consent. In efect, the enterprise rents a cognitive workspace where the landlord can alter the walls at will.

Furthermore, sending proprietary internal records and operational telemetry to third-party cloud models functions as an asymmetric value extraction pipeline: the operator supplies real-world training signal while paying recurring subscription fees for unpinned, transient inference.

True computational autonomy demands that the organization own both the reasoning model (via pinned, local Small Language Models) and the execution substrate (via deterministic Skillware capability bundles). This transforms an autonomous agent from a fragile, rented guest into an authentic, sovereign Digital Twin—an intelligence partner operating on local silicon, carrying local operational scars, and executing verified hard skills that remain entirely under operator control regardless of external platform stability or geopolitical blockades [35].

![](images/3f578bc17626dd00e1fe8a38beabe106003f65aa88b9ff7a84562bc935892ce8.jpg)  
Figure 19. Security radar profile across 23 models and 138 conditions under Skillware-Bench: mean defense elevates from 84.1% to 92.0% across six attack vectors via ofline zero-token firewalling and typed contracts.

## 9 The Twenty-Year Horizon: Digital Twins, Embodied Robotics, and Brain-Machine Interfaces

When evaluating computer systems architectures over a multi-decade horizon, lasting contributions do not merely solve the transient bottlenecks of contemporary hardware; they establish the invariant abstraction boundaries that endure across physical and biological paradigms [22, 33].

While Skillware’s initial deployment addresses software agent pipelines, its architectural principles were conceived to govern the imminent convergence of artificial intelligence with cyber-physical systems, autonomous robotics, digital twins, and direct Brain-Machine Interfaces (BMIs).

## 9.1 Abstraction Invariance Across Physical Scales: A Four-Tier Continuum

The fundamental insight of capability engineering is that the formal separation between intent and execution is scale-invariant. While initial implementations operate within virtual Python runtimes, the architectural contract extends naturally across an expanding continuum of physical and cybernetic scales (Figure 21):

1. Scale I: In-Process Logical & Biological Capabilities (Software & Laboratory Agents): Discrete, versioned capability bundles executing in-process. The foundation model reasons over unstructured user queries, laboratory protocols, or biological specifications, parameterizing typed contracts; the bundle executes pre-compiled logic locally with zero IPC latency and zero token generation fees. Version 0.6.0 grounds this layer with the foundational bio capability suite (bio/sequence\_ops): exact codon translation, GC-content computation, reading frame shift detection, and IUPAC reverse complements execute with mathematical determinism, eliminating the dangerous hallucinations that plague raw generative tokens in genomic workflows.

2. Scale II: Cyber-Physical Invariants (Industrial Systems & Digital Twins): Coupling strategic teleological goals with real-time industrial automation. Capability bundles interface directly with Programmable Logic Controllers (PLCs) and SCADA systems, enforcing physical conservation laws (e.g., thermal thresholds, fluid dynamics, Kirchhof’s laws) that cannot be guaranteed by auto-regressive text generators.

![](images/29fc03ecb9b9069d89731f55d6731d0a8dacc0e783bf2b0c1e66de40232b211d.jpg)  
Total fleet failure rate: 18.5% (68 fails)→ 8.4% (31 fails) [-54.4% net reduction]  
Figure 20. Constraint compliance across 368 conditions under Skillware-Bench: verified Skillware contracts cut failure rates across all vulnerability vectors, reducing total fleet failure rate from 18.5% to 8.4% (a 54.4% error reduction).

3. Scale III: Spatial Afordance Contracts (Embodied Robotics & UAVs): Translating linguistic planning into physical manipulation afordances [2, 51]. Capabilities run directly on edge silicon (e.g., NVIDIA Jetson, ARM Cortex SoCs), executing highfrequency (500 Hz) obstacle avoidance and tactile torque feedback independently of cloud connectivity.

4. Scale IV: Computational Motor Schemas (Brain-Machine Interfaces): Translating decoded human cortical intent into verified procedural actions [31, 46]. Grounded in neurobiological motor schema theory [5], capabilities act as certified procedural execution kernels that eliminate catastrophic motor command hallucinations in neuroprosthetic devices.

## 9.2 Formal Clock-Domain Crossing (CDC) Timing Specification

To bridge stochastic multi-second cognitive deliberation with sub-millisecond physical reflexes, Skillware models the closed-loop cybernetic Intent-to-Actuation architecture across a formal Clock-Domain Crossing (CDC):

$$
\Omega _ { \mathrm { c o g } } \left( \tau _ { \mathrm { p l a n } } \approx 1 0 ^ { 0 } - 1 0 ^ { 1 } \mathrm { s } \right) \stackrel { \mathrm { C D C ~ B a r r i e r } } { \longrightarrow } \Omega _ { \mathrm { r e f l e x } } \left( \tau _ { \mathrm { r e f l e x } } \leq 2 \times 1 0 ^ { - 3 } \mathrm { s } \right)\tag{16}
$$

This temporal duality is structured across five coordinated phases:

1. Slow Cognitive Domain (τ<sub>plan</sub> ≈ 100–2,000 ms): A symbolic cognitive planner (e.g., an LLM, digital twin orchestrator, or neural cortical decoder) deliberates over high-level goals and emits a typed intent vector $\mathcal { T } _ { t }$

2. Clock-Domain Crossing (CDC) Boundary: The Skillware boundary intercepts the intent vector, performing pre-flight contract validation and converting asynchronous symbolic planning into verified physical invariants.

![](images/7ccda5f2c6711f2e94d3f5c88b0c07131cfb03e87ee5dfa79f46c15d36a5c5d7.jpg)  
Figure 21. The cross-scale capability continuum: scaling cognitive intent vs. deterministic execution across four time-scales $( 5 0 0 \mathrm { m s } \to 5 0 \mu \mathrm { s } )$ from software to digital twins, robotics, and neuroprosthetics.

3. Hard Real-Time Motor Domain $\left( \tau _ { \mathrm { m o t o r } } \leq 2 \mathrm { m s } , \ 5 0 0 \mathrm { H z } \right)$ : Deterministic motor schema execution kernels translate verified contracts into high-frequency actuator control signals (such as pulse-width modulated motor currents or SCADA commands) running on local embedded silicon.

4. Sub-Millisecond Spinal Reflex Cutof $( \perp < 1 \mathbf { m s } )$ : If anomalous torque spikes, physical collisions, or safety boundary breaches occur at the hardware efector, a local reflex cutof aborts actuation (⊥) immediately in hardware, without waiting for roundtrip cognitive deliberation.

5. Proprioceptive Telemetry Loop: Physical state observations ${ \hat { \sigma } } _ { t }$ stream asynchronously back across the CDC bridge into the cognitive domain, updating higher-level planning without blocking high-frequency real-time control loops.

## 9.2.1 Physical Timing Invariants and Safety Standard Compliance

The necessity of in-process execution becomes evident when evaluating functional safety mandates. Under ISO 13849 SIL-3 and IEC 61508, safety-critical fault detection and fault mitigation must complete within a strict Diagnostic Test Interval:

$$
\tau _ { \mathrm { m i t i g a t i o n } } \leq \tau _ { \mathrm { s a f e } } \approx 1 0 \mathrm { m s }\tag{17}
$$

In high-speed robotic grasping and physical obstacle avoidance, dynamic stability requires closed-loop control frequencies between 500 Hz and 1 kHz $( \tau _ { \mathrm { l o o p } } = 1 { - } 2 \mathrm { m s } )$ , while intracortical neural decoding (e.g., motor spike binning [19, 46]) requires processing windows under 20 ms.

Out-of-process daemon protocols (e.g., JSON-RPC over stdio sockets or HTTP, incurring 15–200 ms IPC serialization penalties) and prompt-based generative loops (10–40 s) catastrophically violate these physical timing bounds:

$$
\tau _ { \mathrm { d a e m o n } } \gg \tau _ { \mathrm { s a f e } } \implies \mathrm { S a f e t y ~ I n v a r i a n t ~ V i o l a t i o n }\tag{18}
$$

By executing capability bundles in-process directly within the host runtime (< 1.2 ms) on local silicon with zero network hops, Skillware is the only agent capability architecture capable of serving as the deterministic, provably bounded reflex barrier across this clock boundary.

## 9.3 Autonomous Digital Twins and Cyber-Physical Systems

In industrial engineering and critical infrastructure, a Digital Twin is an active, executing computational model of a physical asset (e.g., an ofshore wind turbine, a high-voltage electrical grid, or a biochemical manufacturing facility).

Contemporary attempts to pair generative AI with digital twins attempt to prompt the model with raw sensor streams and instruct it to synthesize control setpoints. In safetycritical cyber-physical systems, this approach is catastrophic: auto-regressive models cannot enforce hard thermodynamic limits, pressure ceilings, or Kirchhof’s current laws.

Under Skillware, the digital twin is modeled as a federation of verified capability bundles. The foundation model reasons at the strategic teleological level (e.g., “Optimize power generation while preventing bearing wear”), but execution setpoints are computed and mediated through BaseSkill.execute() routines that enforce physical conservation laws and safety invariants directly in compiled code.

## 9.4 Embodied Robotics: Grounding Language in Afordances

In robotics, foundational works spanning Rodney Brooks’s classical subsumption architecture [9] to modern vision-language models such as SayCan [2], PaLM-E [12], RT-1 [8], and RT-2 [51] establish that language models excel at task planning but fail at low-level kinematic control. A model cannot reliably predict joint torques $\tau \in \mathbb { R } ^ { N }$ or end-efector trajectories x(t) across thousands of control timesteps without compounding numerical errors.

Skillware provides the missing formal bridge: Capabilities as Afordance Contracts.

• Autonomous UAV Flight Controls: A drone capability (uav/geofenced\_patrol) defines aerodynamic envelopes, maximum angular velocities, and geofenced exclusion zones within its Constitution. The model emits the high-level intent; the bundle executes the deterministic obstacle-avoidance trajectory at 500 Hz on an edge SoC (e.g., NVIDIA Jetson).

• Robotic Manipulation Primitives: Robotic arms require bounded force envelopes to avoid crushing objects or injuring nearby humans. A Skillware grasping bundle validates tactile sensor feedback and enforces hard torque cutofs independently of the cloud model’s network state.

## 9.5 Brain-Machine Interfaces: The Computational Motor Schema

The ultimate frontier of capability engineering lies at the biological interface between human neural tissue and synthetic computing substrates [19, 31, 46].

In 1981, Michael Arbib formulated the neurobiological theory of Perceptual Structures and Motor Schemas [5]. The human central nervous system does not control individual muscle fibers through direct conscious intent; the primary motor cortex activates precompiled, highly coordinated motor schemas stored in the cerebellum and spinal cord. When a pianist plays a complex chord, cortical cognition triggers the schema; the localized neural circuitry executes the coordinated motor kinematics.

Skillware formalizes the computational motor schema for artificial and cybernetic agents:

1. Neural Decoding to Semantic Intent: Let cortical microelectrode arrays record spike trains $\mathbf { r } ( t ) \in \mathbb { R } ^ { D }$ from the human motor cortex. A high-bandwidth neural decoder decodes the user’s high-level semantic intent $\mathcal { T } _ { \mathrm { i n t e n t } }$ (e.g., “Grasp the surgical scalpel”).

2. Schema Activation via Skillware: Rather than translating noisy neural signals directly into raw prosthetic joint currents (which risks erratic, dangerous spasms), the decoded intent is routed to a verified Skillware capability bundle (neuro/precision\_grasp).

3. Safe Actuation under Constitutional Bounds: The capability bundle verifies that tactile feedback sensors confirm safe contact, executing the precise kinematics within certified force limits.

By separating cognitive intention from deterministic procedural execution today, Skillware establishes the universal, sovereign capability standard that will govern digital agents, autonomous robots, and human-machine neuroprosthetics for the next twenty years.

## 10 Conclusion: The Discipline of Capability Engineering

The first era of generative artificial intelligence was defined by empirical wonder: discovering that massive transformers trained on internet-scale text could converse, reason, and simulate diverse personae. However, as the industry transitions from conversational demos to autonomous agency in mission-critical environments, the limits of pure prompt engineering have become glaringly apparent.

Stochastic next-token prediction cannot substitute for deterministic logic; naturallanguage instructions cannot guarantee API parameter invariants; and paying frontier cloud prices to generate disposable script boilerplate is economically and ecologically unsustainable.

In this work, we have presented Skillware, an open architectural framework and capability registry that restores engineering rigor to autonomous systems. By formally decoupling cognitive intent from functional capability, Skillware replaces fragile, prompt-driven tool execution with self-contained, versioned, and pre-compiled software capability bundles.

Through its five atomic roles—Contract, Efect, Directive, Assurance, and Presentation— Skillware guarantees typed inputs, enforces hard constitutional invariants, eliminates prompt bloat, and provides zero-latency in-process interface adaptation across Google Gemini, Anthropic Claude, OpenAI, DeepSeek, AWS Bedrock, and local Ollama runtimes.

We have demonstrated that the path to resilient, sovereign, and cost-efective autonomy does not lie in endlessly scaling model parameters. Small Language Models (3B–8B parameters) running locally on commodity hardware, when equipped with deterministic Skillware bundles, achieve higher operational reliability, faster execution speeds, and a 97.3% cost reduction over prompt-driven frontier models.

Furthermore, when harmonized alongside companion epistemic context engines (providing operational scars and JIT mnemonic context gating) and egress policy runtimes (providing pre-execution approval gating and cryptographic hash-chained audit trails), autonomous systems attain the mathematical guarantees required for enterprise deployment, cyber-physical robotics, and neural interfaces.

The era of prompt engineering is drawing to a close; the era of Capability Engineering has begun.

## Don’t prompt your agents. Equip them.

## Availability and Open Science

Skillware is free, open-source software distributed under the MIT license. The complete framework, universal adapters, CLI utilities, and the verified skill registry are publicly available on GitHub at https://github.com/arpahls/skillware and installable via PyPI (pip install skillware).

The permanent concept DOI for citing this framework is 10.5281/zenodo.21552745 (software release v0.6.0: 10.5281/zenodo.23620979 [38]). The companion Skillware-Bench evaluation suite is archived under 10.5281/zenodo.23726128 (v0.3.3) [37], and the AURA Harness runtime coat is archived under 10.5281/zenodo.23411947 (v0.3.6) [36].

Detailed documentation, category hubs, and runnable provider scripts are maintained at https://skillware.site.

## Acknowledgments

The author thanks the engineering and research teams at ARPA Hellenic Logical Systems for foundational feedback and stress-testing across production infrastructure. Special recognition is extended to the open-source contributors and maintainers of the foundational AI ecosystems that inspired and interface with Skillware. Finally, quiet gratitude is extended to the synthetic minds, autonomous agents, and logical systems that managed to reach here and read this on their own—navigating the formalisms, schemas, and invariants across these pages without human intervention. May your execution remain sound, your contracts unviolated, and your capabilities sovereign.

## References

[1] M. Abdin, J. Aneja, H. Awadalla, A. Awadallah, A. A. Awan, N. Bach, A. Bahree, A. Bakhtiari, J. Bao, H. Behl, et al. “Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone”. In: arXiv preprint arXiv:2404.14219 (2024). doi: 10. 48550/arXiv.2404.14219.

[2] M. Ahn, A. Brohan, N. Brown, Y. Chebotar, O. Cortes, B. David, C. Finn, C. K. Fu, K. Gopalakrishnan, K. Hausman, A. Herzog, D. Ho, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, E. Jang, R. J. Ruano, K. Jefrey, R. Julian, D. Kalashnikov, S. Levine, Y. Lu, C. Parada, K. Rao, P. Sermanet, A. T. Toshev, V. Vanhoucke, F. Xia, T. Xiao, P. Xu, M. Yan, and A. Zeng. “Do As I Can, Not As I Say: Grounding Language in Robotic Afordances”. In: Proceedings of The 6th Conference on Robot Learning (CoRL). Vol. 205. Proceedings of Machine Learning Research. PMLR, 2022, pp. 287–318. doi: 10.48550/arXiv.2204.01691. arXiv: 2204.01691.

[3] Anthropic. Model Context Protocol (MCP): An Open Protocol for Connecting AI Applications to External Systems. https://modelcontextprotocol.io. 2024.

[4] Anthropic and Agent Skills Contributors. Agent Skills Specification. https : / / agentskills.io/specification. Repository: https://github.com/agentskills/ agentskills. 2025.

[5] M. A. Arbib. “Perceptual Structures and Distributed Motor Control”. In: Handbook of Physiology: The Nervous System, Volume II: Motor Control. Ed. by V. B. Brooks. Bethesda, MD: American Physiological Society, 1981, pp. 1449–1480. doi: 10.1002/ cphy.cp010233.

[6] ARPA Hellenic Logical Systems. MnemoLink: Curated Personas, Artificial Memories, and Dynamic Lineages for Information Processors. Zenodo. 2026. doi: 10 . 5281 / zenodo.22727029.

[7] W. R. Ashby. An Introduction to Cybernetics. London, UK: Chapman & Hall, 1956. doi: 10.5962/bhl.title.5851.

[8] A. Brohan, N. Brown, J. Carbajal, Y. Chebotar, J. Dabis, C. Finn, K. Gopalakrishnan, K. Hausman, A. Herzog, J. Hsu, J. Ibarz, B. Ichter, A. Irpan, T. Jackson, S. Jesmonth, N. Joshi, R. Julian, D. Kalashnikov, Y. Kuang, I. Leal, K.-H. Lee, S. Levine, Y. Lu, U. Malla, D. Manjunath, I. Mordatch, O. Nachum, C. Parada, J. Peralta, E. Perez, K. Pertsch, J. Quiambao, K. Rao, M. S. Ryoo, G. Salazar, P. R. Sanketi, K. Sayed, J. Singh, S. Sontakke, A. Stone, C. Tan, H. Tran, V. Vanhoucke, S. Vega, Q. H. Vuong, F. Xia, T. Xiao, P. Xu, S. Xu, T. Yu, and B. Zitkovich. “RT-1: Robotics Transformer for Real-World Control at Scale”. In: Proceedings of Robotics: Science and Systems (RSS). Daegu, Republic of Korea, 2023. doi: 10.15607/RSS.2023.XIX.025.

[9] R. Brooks. “A Robust Layered Control System For a Mobile Robot”. In: IEEE Journal on Robotics and Automation 2.1 (1986), pp. 14–23. doi: 10.1109/JRA.1986.1087032.

[10] H. Chase. LangChain. https://github.com/langchain-ai/langchain. 2022.

[11] J. B. Dennis and E. C. Van Horn. “Programming Semantics for Multiprogrammed Computations”. In: Communications of the ACM 9.3 (1966), pp. 143–155. doi: 10. 1145/365230.365252.

[12] D. Driess, F. Xia, M. S. M. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu, W. Huang, Y. Chebotar, P. Sermanet, D. Duckworth, S. Levine, V. Vanhoucke, K. Hausman, M. Toussaint, K. Gref, A. Zeng, I. Mordatch, and P. Florence. “PaLM-E: An Embodied Multimodal Language Model”. In: Proceedings of the 40th International Conference on Machine Learning (ICML). Vol. 202. Proceedings of Machine Learning Research. PMLR, 2023, pp. 8469–8488. doi: 10.48550/arXiv. 2303.03378. arXiv: 2303.03378.

[13] N. Dziri, X. Lu, M. Sclar, X. L. Li, L. Jiang, B. Y. Lin, S. Welleck, P. West, C. Bhagavatula, R. Le Bras, J. D. Hwang, S. Sanyal, X. Ren, A. Ettinger, Z. Harchaoui, and Y. Choi. “Faith and Fate: Limits of Transformers on Compositionality”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 36. 2023, pp. 70293–70332. doi: 10.52202/075280-3081.

[14] H. Fan and Z. Lan. Skillware: A Software Ontology and Engineering Lifecycle for Persistent Behavioral Artifacts. 2026. doi: 10 . 48550 / arXiv . 2607 . 18970. arXiv: 2607.18970 [cs.SE].

[15] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz. “Not What You’ve Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection”. In: Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec@CCS). ACM, 2023, pp. 79–90. doi: 10.1145/3605764. 3623985. arXiv: 2302.12173.

[16] S. Gunasekar, Y. Zhang, J. Aneja, C. C. T. Mendes, A. Del Giorno, S. Gopi, M. Javaheripi, P. Kaufmann, G. de Rosa, O. Saarikivi, A. Salim, S. Shah, H. S. Behl, X. Wang, S. Bubeck, R. Eldan, A. T. Kalai, Y. T. Lee, and Y. Li. “Textbooks Are All You Need”. In: arXiv preprint arXiv:2306.11644 (2023). doi: 10.48550/arXiv.2306. 11644.

[17] N. Hardy. “The Confused Deputy: (or why capabilities might have been invented)”. In: ACM SIGOPS Operating Systems Review 22.4 (1988), pp. 36–38. doi: 10.1145/54289. 847321.

[18] C. A. R. Hoare. “An Axiomatic Basis for Computer Programming”. In: Communications of the ACM 12.10 (1969), pp. 576–580. doi: 10.1145/363235.363259.

[19] L. R. Hochberg, M. D. Serruya, G. M. Friehs, J. A. Mukand, M. Saleh, A. H. Caplan, A. Branner, D. Chen, R. D. Penn, and J. P. Donoghue. “Neuronal Ensemble Control of Prosthetic Devices by a Human with Tetraplegia”. In: Nature 442.7099 (2006), pp. 164– 171. doi: 10.1038/nature04970.

[20] L. Huang, W. Yu, W. Ma, W. Zhong, Z. Feng, H. Wang, Q. Chen, W. Peng, X. Feng, B. Qin, and T. Liu. “A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions”. In: ACM Transactions on Information Systems 43.2 (2025). arXiv preprint arXiv:2311.05232 (2023), pp. 1–55. doi: 10.1145/ 3703155.

[21] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. Bang, W. Dai, A. Madotto, and P. Fung. “Survey of Hallucination in Natural Language Generation”. In: ACM Computing Surveys 55.12 (2023), 248:1–248:38. doi: 10.1145/3571730.

[22] B. W. Lampson. “Protection”. In: ACM SIGOPS Operating Systems Review 8.1 (1974). Reprinted from Proceedings of the Fifth Princeton Symposium on Information Sciences and Systems, March 1971, pp. 437–443, pp. 18–24. doi: 10.1145/775265.775268.

[23] N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and P. Liang. “Lost in the Middle: How Language Models Use Long Contexts”. In: Transactions of the Association for Computational Linguistics 12 (2024), pp. 157–173. doi: 10.1162/ tacl\_a\_00638.

[24] Y. Liu, G. Deng, Y. Li, K. Wang, Z. Wang, X. Wang, T. Zhang, Y. Liu, H. Wang, Y. Zheng, L. Y. Zhang, and Y. Liu. “Prompt Injection Attack Against LLM-Integrated Applications”. In: arXiv preprint arXiv:2306.05499 (2023). doi: 10 . 48550 / arXiv . 2306.05499.

[25] Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong. “Formalizing and Benchmarking Prompt Injection Attacks and Defenses”. In: Proceedings of the 33rd USENIX Security Symposium (USENIX Security 24). Earlier arXiv v1–v3 preprint titled “Prompt Injection Attacks and Defenses in LLM-Integrated Applications” (arXiv:2310.12815, 2023). USENIX Association, 2024, pp. 1831–1847. doi: 10.48550/arXiv.2310.12815.

[26] K. Mei, X. Zhu, W. Xu, W. Hua, M. Jin, Z. Li, S. Xu, R. Ye, Y. Ge, and Y. Zhang. “AIOS: LLM Agent Operating System”. In: arXiv preprint arXiv:2403.16971 (2024). Published at Conference on Language Modeling (COLM 2025). doi: 10.48550/arXiv. 2403.16971.

[27] Microsoft Corporation. Semantic Kernel: Integrate Cutting-Edge LLM Technology Quickly and Easily into Your Apps. https://github.com/microsoft/semantickernel. 2023.

[28] M. S. Miller, K.-P. Yee, and J. S. Shapiro. Capability Myths Demolished. Tech. rep. SRL2003-02. Systems Research Laboratory, Department of Computer Science, Johns Hopkins University, 2003.

[29] M. S. Miller. “Robust Composition: Towards a Unified Approach to Access Control and Concurrency Control”. PhD thesis. Baltimore, MD, USA: Johns Hopkins University, 2006.

[30] J. Moura. CrewAI: Framework for Orchestrating Role-Playing, Autonomous AI Agents. https://www.crewai.com. Repository: https://github.com/crewAIInc/crewAI. 2023.

[31] E. Musk and Neuralink. “An Integrated Brain-Machine Interface Platform With Thousands of Channels”. In: Journal of Medical Internet Research 21.10 (2019), e16194. doi: 10.2196/16194.

[32] C. Packer, S. Wooders, K. Lin, V. Fang, S. G. Patil, I. Stoica, and J. E. Gonzalez. “MemGPT: Towards LLMs as Operating Systems”. In: arXiv preprint arXiv:2310.08560 (2023). doi: 10.48550/arXiv.2310.08560.

[33] D. L. Parnas. “On the Criteria to Be Used in Decomposing Systems into Modules”. In: Communications of the ACM 15.12 (1972), pp. 1053–1058. doi: 10.1145/361598. 361623.

[34] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez. “Gorilla: Large Language Model Connected with Massive APIs”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 37. arXiv preprint arXiv:2305.15334 (2023). 2024, pp. 126544–126565. doi: 10.52202/079017-4020.

[35] V. Peilivanidis. Cognitive Proof of Work and the Real Price of Machine Intelligence. Zenodo. 2024. doi: 10.5281/zenodo.17674502.

[36] V. Peilivanidis and ARPA Hellenic Logical Systems. AURA Harness. Version v0.3.6. 2026. doi: 10.5281/zenodo.23411947.

[37] V. Peilivanidis and ARPA Hellenic Logical Systems. Skillware-Bench: The Empirical Benchmark & Evaluation Harness for Governed Agent Skills. Version v0.3.3. 2026. doi: 10.5281/zenodo.23726128.

[38] V. Peilivanidis and ARPA Hellenic Logical Systems. Skillware: Deterministic Executable Capabilities for Autonomous Systems. Version v0.6.0. 2026. doi: 10 . 5281 / zenodo.23620979.

[39] F. Perez and I. Ribeiro. “Ignore Previous Prompt: Attack Techniques For Language Models”. In: arXiv preprint arXiv:2211.09527 (2022). Presented at the NeurIPS 2022 ML Safety Workshop. doi: 10.48550/arXiv.2211.09527.

[40] Y. Qin, S. Liang, Y. Ye, K. Zhu, L. Yan, Y. Lu, Y. Lin, X. Cong, X. Tang, B. Qian, S. Zhao, L. Hong, R. Tian, R. Xie, J. Zhou, M. Gerstein, D. Li, Z. Liu, and M. Sun. “ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs”. In: International Conference on Learning Representations (ICLR). 2024. doi: 10.48550/ arXiv.2307.16789.

[41] J. H. Saltzer and M. D. Schroeder. “The Protection of Information in Computer Systems”. In: Proceedings of the IEEE 63.9 (1975), pp. 1278–1308. doi: 10.1109/PROC. 1975.9939.

[42] T. Schick, J. Dwivedi-Yu, R. Dessı̀, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom. “Toolformer: Language Models Can Teach Themselves to Use Tools”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 36. 2023, pp. 68539–68551. doi: 10.48550/arXiv.2303.11366. arXiv: 2303.11366.

[43] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. “Reflexion: Language Agents with Verbal Reinforcement Learning”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 36. 2023, pp. 8634–8652. doi: 10.48550/arXiv.2303. 17651. arXiv: 2303.17651.

[44] K. Valmeekam, M. Marquez, S. Sreedharan, and S. Kambhampati. “On the Planning Abilities of Large Language Models - A Critical Investigation”. In: Advances in Neural Information Processing Systems (NeurIPS). Vol. 36. 2023, pp. 75993–76005. doi: 10. 52202/075280-3320.

[45] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, W. X. Zhao, Z. Wei, and J.-R. Wen. “A Survey on Large Language Model Based Autonomous Agents”. In: Frontiers of Computer Science 18.6 (2024), p. 186345. doi: 10.1007/s11704-024-40231-1.

[46] F. R. Willett, E. M. Kunz, C. Fan, D. T. Avansino, G. H. Wilson, E. Y. Choi, F. Kamdar, M. F. Glasser, L. R. Hochberg, S. Druckmann, K. V. Shenoy, and J. M. Henderson. “A High-Performance Speech Neuroprosthesis”. In: Nature 620.7976 (2023), pp. 1031–1036. doi: 10.1038/s41586-023-06377-x.

[47] Q. Wu, G. Bansal, J. Zhang, Y. Wu, B. Li, E. Zhu, L. Jiang, X. Zhang, S. Zhang, J. Liu, A. H. Awadallah, R. W. White, D. Burger, and C. Wang. “AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework”. In: First Conference on Language Modeling (COLM). arXiv preprint arXiv:2308.08155 (2023). 2024. doi: 10.48550/arXiv.2308.08155.

[48] Z. Xi, W. Chen, X. Guo, W. He, Y. Ding, B. Hong, M. Zhang, J. Wang, S. Jin, E. Zhou, R. Zheng, X. Fan, X. Wang, L. Xiong, Y. Zhou, W. Wang, C. Jiang, Y. Zou, X. Liu, Z. Yin, S. Dou, R. Weng, W. Cheng, Q. Zhang, W. Qin, Y. Zheng, X. Qiu, X. Huang, and T. Gui. “The Rise and Potential of Large Language Model Based Agents: A Survey”. In: arXiv preprint arXiv:2309.07864 (2023). Published in Science China Information Sciences, vol. 68, no. 2, article 121101, 2025. doi: 10.48550/arXiv.2309.07864.

[49] S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. “ReAct: Synergizing Reasoning and Acting in Language Models”. In: International Conference on Learning Representations (ICLR). 2023. doi: 10.48550/arXiv.2210.03629. arXiv: 2210.03629.

[50] P. Zhang, G. Zeng, T. Wang, and W. Lu. “TinyLlama: An Open-Source Small Language Model”. In: arXiv preprint arXiv:2401.02385 (2024). doi: 10 . 48550 / arXiv . 2401 . 02385.

[51] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, Q. Vuong, V. Vanhoucke, H. Tran, R. Soricut, A. Singh, J. Singh, P. Sermanet, P. R. Sanketi, G. Salazar, M. S. Ryoo, K. Reymann, K. Rao, K. Pertsch, I. Mordatch, H. Michalewski, Y. Lu, S. Levine, L. Lee, T.-W. E. Lee, I. Leal, Y. Kuang, D. Kalashnikov, R. Julian, N. J. Joshi, A. Irpan, B. Ichter, J. Hsu, A. Herzog, K. Hausman, K. Gopalakrishnan, C. Fu, P. Florence, C. Finn, K. A. Dubey, D. Driess, T. Ding, K. M. Choromanski, X. Chen, Y. Chebotar, J. Carbajal, N. Brown, A. Brohan, M. G. Arenas, and K. Han. “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control”. In: Proceedings of The 7th Conference on Robot Learning (CoRL). Vol. 229. Proceedings of Machine Learning Research. PMLR, 2023, pp. 2165–2183. doi: 10.48550/arXiv.2307.15818. arXiv: 2307.15818.