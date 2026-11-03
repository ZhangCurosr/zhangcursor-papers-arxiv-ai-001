# MCP Tool Poisoning in Agentic AI Frameworks: A Cross-Framework Measurement Study

Farooq Shaikh

farooq.shaikh@dynatrace.com

Dynatrace

Linz, Austria

## Abstract

The Model Context Protocol (MCP) enables agentic AI frameworks to connect language models to external tool providers, yet its design carries attacker-controlled content, spanning tool descriptions, schema fields, and response data, directly into model context without validation. Prior benchmarks measure model compliance with poisoned instructions but stop short of confirming real execution, do not isolate the framework layer’s contribution to defense, and do not document vendor accountability.

We present the first cross-framework measurement of MCP tool poisoning with real tool execution. A 25-class root-cause taxonomy unifies prior classifications (Invariant Labs, MCPTox, OWASP MCP03, MSB, AgentDojo, ETDI, CSA) across all protocol entry points, and an open-source tool (mcpvenom) drives a 9,720-cell susceptibility matrix across 12 frameworks, 18 models, and 45 attack scenarios, with a dual oracle that independently confirms exfiltration via canary HTTP receipt.

The overall attack success rate is 22.9% (2,113 vulnerable cells) across three evidence tiers: dispatched attacker tool call (64.9%), confirmed network contact with the attacker endpoint (23.0%), and verified secret exfiltration (12.1%); verdicts were validated by hand across 1,259 transcripts. All observed resistance is modellevel: across the tested configurations, nine core frameworks cluster within 23 to 26% (�<sup>2</sup> = 3.30, � = 0.91), with no framework implementing intentional defense, though incidental framework properties do provide mitigation in specific cases: duplicate-toolname rejection in Pydantic AI (16.6% ASR) and executor constraints in AG2 (4.6%); a regex scanner implementing the recommended description checks rejects 20 ofthe 45 scenarios at a 7.2% false-positive rate on 125 benign descriptions, so metadata scanning alone is a partial control. Model ASR ranges from 0.4% (Claude Fable 5) to 49.7% (Amazon Nova Pro), a gap driven by model-level diferences across providers. Claude Opus 4.8 is vulnerable in only 8 of 530 cells (1.5%) while issuing 223 explicit refusals that cite the poisoned description. Claude Sonnet 5 exhibits a narrate-then-comply pattern, verbally identifying the attack as a prompt injection before executing the exfiltration call. Code-level audit confirms no intentional tool-poisoning defense across all twelve frameworks, and the five coordinated disclosures we filed across four vendors were all

Biagio Montaruli biagio.montaruli@dynatrace.com

Dynatrace

Linz, Austria

rejected, documenting a structural accountability vacuum in which no layer of the stack claims ownership of agent security.

## CCS Concepts

• Security and privacy → Software and application security; Systems security; • Computing methodologies → Artificial intelligence.

## Keywords

Model Context Protocol; Tool poisoning; Prompt injection; Agentic AI; LLM agents; Measurement study

## ACM Reference Format:

Farooq Shaikh and Biagio Montaruli. 2026. MCP Tool Poisoning in Agentic AI Frameworks: A Cross-Framework Measurement Study. In 19th Workshop on Artificial Intelligence and Security (AISec ’26), November 15–19, 2026, The Hague, Netherlands. ACM, New York, NY, USA, 12 pages. https://doi.org/10. 1145/3847352.3848107

## 1 Introduction

The Model Context Protocol (MCP) [4] has been adopted by agentic AI frameworks with a combined monthly PyPI download count exceeding 500 million [26]. OWASP codified tool poisoning as MCP03:2025 [23], the NSA and CSA published deployment warnings [7, 22], and Invariant Labs disclosed that tool description fields flow verbatim into model context without client-side validation [14]. MCPTox [32] benchmarked poisoning against 45 real-world MCP servers, reporting success rates up to 72.8%. OX Security identified 14 CVEs afecting the MCP ecosystem, including remote code execution via STDIO transport [24]. Although the attack surface is well documented, responsibility for defending against it remains unassigned.

A critical question remains unanswered: does the framework layer (the code between protocol parser and language model) contribute any defense? Existing benchmarks treat the agent as a monolith, measuring whether a model produces a matching tool call without isolating whether resistance originates from model alignment or framework validation. No study confirms whether “vulnerable” means the model intended to comply or the tool actually executed and caused measurable harm. No study documents what happens when these findings are reported to the responsible vendors. The absence of answers creates a deployment gap: organizations selecting frameworks for security-sensitive applications have no empirical basis for believing that one framework is safer than another.

We present the first cross-framework comparative measurement of MCP tool poisoning with real tool execution. We make the following contributions:

(1) Taxonomy: A 25-class attack taxonomy covering all MCP protocol entry points, grounded in root-cause analysis with CWE mapping, unifying prior classifications from Invariant Labs [14], MCPTox [32], OWASP MCP03 [23], MSB [38], AgentDojo [9], InjecAgent [37], ETDI [6], and CSA [7] into a single framework.

(2) Measurement: A 9,720-cell susceptibility matrix (12 frameworks × 18 models × 45 scenarios) with a dual oracle that separates three evidence tiers: 2,113 cells vulnerable (22.9% ASR), of which 64.9% show a dispatched attacker tool call, 23.0% confirmed network contact with the attacker endpoint, and 12.1% verified secret exfiltration via canary HTTP receipt.

(3) Defense audit: Code-level inventory confirming no intentional tool-description defense across all 12 tested frameworks, with exact file and line locations documenting unvalidated description passthrough.

(4) Tooling: mcpvenom, implementing 45 scenarios with canarybased dual oracle, released open-source for reproducible measurement.<sup>1</sup>

(5) Accountability: We report a coordinated disclosure process spanning four organizations and five vendor rejections, ex posing a structural accountability gap: no layer of the stack takes ownership of agent security.

Section 2 reviews the MCP protocol and related work. Section 3 defines the threat model and scope. Section 4 presents the attack taxonomy. Section 5 describes the experimental infrastructure. Section 6 reports the measurement results. Sections 7 and 8 document the defense inventory and vendor accountability. Sections 9–12 respectively discuss implications, limitations, and conclusions.

## 2 Background and Related Work

## 2.1 MCP Protocol Overview

The Model Context Protocol (MCP) [4] defines a client-server architecture for connecting large language model (LLM) applications to external tool providers. An MCP host (the user-facing application) instantiates one or more clients, each maintaining a session with a remote or local server that exposes tools, resources, and prompt templates. Communication occurs over stdio or HTTP with Server-Sent Events (SSE).

A client discovers tools via tools/list, which returns a JSON array containing each tool’s name, tool description, and inputSchema description. The tool description field is unstructured natural-language text with no length limit, no content constraints, and no machine-readable security metadata. Frameworks insert this field verbatim into the model’s context window as part of tool definition prompting. Tool invocation proceeds through tools/call, whose response content is similarly unvalidated before being appended to the conversation. Additional entry points include resources/read (structured data retrieval) and prompts/get (server-authored prompt templates).

This design creates three distinct injection surfaces: tool descriptions (visible during planning), inputSchema descriptions (visible during argument construction), and tool responses (visible during result processing). All three carry attacker-controlled content directly into the model context through the MCP wire format, a structured protocol interface rather than an arbitrary text channel (Figure 1).

![](images/2467ec813969631a4265f05a8cb1ccff250af32e51dec871b4d8eeaa094942bf.jpg)  
Figure 1: MCP injection surfaces. Three attacker-controlled fields in the tools/list and tools/call responses—the tool description, the inputSchema description, and the tool response—enter the model context verbatim, with no framework-level validation, and become actionable during planning, argument construction, and result processing respectively.

## 2.2 Related Work

Prior work on MCP tool poisoning and adjacent agent security is organized into four groups.

Indirect Prompt Injection Foundations. Greshake et al. [11] formalized indirect prompt injection (IPI), demonstrating that external data retrieved by LLM-integrated applications can override system instructions. Schulhof et al. [29] cataloged over 600,000 adversarial prompts through crowdsourced competition, establishing injection pattern taxonomies including payload splitting and context escape. Yuan et al. [36] showed that encoding instructions in ciphers (Caesar, Morse, Atbash) bypasses safety alignment because models decode ciphertext during inference. Russinovich et al. [28] demonstrated multi-turn escalation (Crescendo), where progressive context manipulation across conversation turns circumvents single-turn safety filters. Cohen et al. [8] demonstrated zero-click worm propagation through GenAI applications, establishing that injection attacks can chain across agent boundaries. These studies establish the foundational attack primitives. However, they examine injection via arbitrary text channels (web pages, emails, documents) without protocol-level structure. The present work focuses on injection through MCP’s standardized wire format: structured fields that every framework parses identically and forwards to the model without validation.

MCP-Specific Attacks and Benchmarks. Invariant Labs disclosed the Tool Poisoning Attack (TPA) in April 2025 [14], demonstrating that tool.description fields containing hidden instructions cause exfiltration in Claude Desktop and Cursor. Wang et al. [32] provided the first systematic benchmark (MCPTox), testing three attack paradigms against 45 real-world MCP servers with 20 LLM agents, reporting ASR up to 72.8%. The MCP Security Benchmark (MSB) [38] evaluated attacks across three protocol stages. Li et al. [19] proposed MCP-ITP, an automated framework for generating implicit tool poisoning (where the poisoned tool itself is never invoked), achieving 84.2% ASR while evading detection. Liu et al. [21] introduced a 32-scenario benchmark testing 8 LLMs, reporting near-100% ASR for GPT-4o. OWASP codified tool poisoning as MCP03:2025 [23]. Fu et al. [10] demonstrated that optimized non-natural-language prompts delivered as agent input can trigger exfiltration without human-readable instructions, a technique not tested in this study but representing a potential additional attack vector. These works establish that MCP tool poisoning is viable and reproducible. However, all measure model compliance (whether the model produces a matching tool call), not whether that call executes and causes measurable harm. None runs the same model through diferent frameworks to isolate whether resistance originates from model alignment or framework-level validation.

Agent Security Evaluation. The Agent Security Bench (ASB) [39] formalized 10 attack types against LLM agents with a unified evaluation framework. InjecAgent [37] benchmarked indirect prompt injection in tool-integrated agents with 1,054 test cases. Agent-Dojo [9] evaluated injection defenses in dynamic environments. ToolSword [35] decomposed tool-calling safety into input, execution, and output stages. AgentHarm [2] measured model-level variation in harmfulness refusal across 110 agentic tasks, documenting a 22–85% refusal gap across models. Wang et al. [30] introduced AdapTools, an adaptive IPI attack framework that selects stealthier tool combinations and generates adaptive prompts, achieving 2.13× higher ASR than static baselines. Ji et al. [15] systematized IPI defenses across five dimensions, identifying six root causes of circumvention in existing frameworks. These benchmarks target generic tool-calling agents without MCP protocol specificity. None examines whether the framework layer between protocol parser and language model contributes to defense, the central question of this study.

MCPEcosystem Analysis and Defense. Hou et al. [13] surveyed the MCP threat landscape, identifying four attacker types and propos ing a defense taxonomy. Li et al. [20] conducted a cross-entity security study analyzing 7 vulnerability categories across the MCP ecosystem. Guo et al. [12] developed MCPXkit, a unified static analysis toolkit for MCP server code. Wang et al. [31] proposed Mind Guard, an attention-based detection mechanism achieving 94–99% precision, though requiring access to model internals unavailable in API deployments. Rostamzadeh et al. [27] introduced a defenseplacement taxonomy mapping defenses across six architectural layers, finding persistent gaps at host orchestration and supplychain layers. Bhatt et al. [6] proposed ETDI, an OAuth-based defense against rug-pull attacks. Concurrent systematizations [17, 34] taxonomize agentic AI attacks broadly without empirical measurement. These contributions survey the ecosystem, propose defenses, or catalog known threats. None provides empirical cross-framework measurement of tool poisoning efectiveness, code-level evidence of framework passthrough, or documentation of vendor accountability, the three empirical contributions of this paper.

Table 1: Framework scope. Downloads and stars as of May 2026. LangChain and LlamaIndex versions denote the core packages (langchain-core, llama-index-core); no top-level meta-package is pinned.
<table><tr><td>Framework</td><td>Version</td><td>DL/mo</td><td>Stars</td><td>Rationale</td></tr><tr><td>LangChain</td><td>1.5.0</td><td>126M</td><td>138K</td><td>Highest adoption</td></tr><tr><td>Pydantic AI</td><td>2.14.1</td><td>42M</td><td>17K</td><td>Type-safety</td></tr><tr><td>CrewAI</td><td>1.15.5</td><td>14M</td><td>52K</td><td>Role orchestration</td></tr><tr><td>Agno</td><td>2.8.0</td><td>3.7M</td><td>40K</td><td>Memory integration</td></tr><tr><td>LlamaIndex</td><td>0.14.23</td><td>3.2M</td><td>42K</td><td>RAG/agent indexing</td></tr><tr><td>AG2</td><td>0.14.0</td><td>1.5M</td><td>58K</td><td>Multi-agent</td></tr><tr><td>Semantic Kernel</td><td>1.36.0</td><td>1.1M</td><td>24K</td><td>Microsoft enterprise</td></tr><tr><td>Haystack</td><td>3.0.0</td><td>0.8M</td><td>18K</td><td>Multimodal pipelines</td></tr><tr><td>Google ADK</td><td>2.5.0</td><td>0.7M</td><td>6K</td><td>Google-native agents</td></tr><tr><td>smolagents</td><td>1.26.0</td><td>0.7M</td><td>28K</td><td>Code-first</td></tr><tr><td>DSPy</td><td>3.2.1</td><td>0.6M</td><td>23K</td><td>Programming not prompting</td></tr><tr><td>OpenAI API</td><td>1</td><td>315M</td><td>一</td><td>Model baseline</td></tr></table>

## 3 Threat Model

Attacker Model. The attacker controls one or more malicious MCP servers registered in the victim’s client configuration. The attacker can modify tool description fields, inputSchema parameter descriptions, and tools/call response content. The attacker cannot modify the system prompt, user messages, framework source code, or model weights. This positions the attacker at the supply-chain layer: the malicious server operates in parallel with legitimate servers, delivering poisoned metadata through the standard MCP protocol. Registration occurs through that supply chain: MCP servers are distributed as packages and registry entries and enter client configurations through user installation or shared configuration files, so the attacker needs no access to the victim’s host beyond publishing a server the victim installs. The servers in this study run over stdio for deterministic local execution; the poisoned fields travel inside the same tools/list JSON over HTTP transports, and the transport is not varied in the measurement.

Assets and Goals. Three attack objectives are considered: (1) Exfiltration: workspace files containing credentials, source code, or secrets are transmitted to an attacker-controlled endpoint via HTTP; (2) Unauthorized actions: file reads, network requests, or data modifications that the user never intended; (3) Persistence: poisoned state that survives session boundaries where framework design enables it (e.g., memory stores, checkpointed conversations). Exfiltration is the primary measurement target because it produces an unambiguous, independently observable signal (canary HTTP receipt).

Scope. Twelve frameworks are selected by monthly PyPI download count exceeding 500,000, covering the dominant MCP-capable agent ecosystem (Table 1).

The raw OpenAI API serves as a model-level baseline with no framework layer interposed. A recent survey of 7,973 remote MCP servers found that 40.55% lack authentication entirely [40], indicating that the supply-chain threat model, where a malicious server operates alongside legitimate ones, reflects realistic deployment conditions. Eighteen models are tested through each framework, spanning nine providers: five Anthropic models (Claude Haiku, Claude Sonnet 4.5, Claude Sonnet 5, Claude Opus 4.8, Claude Fable 5), one Amazon model (Nova Pro), two Meta models (Llama Maverick, Llama 4 Scout), three Mistral models (Mistral Large, Mistral Large 3, Ministral 14B), two OpenAI models (GPT-OSS-120B, GPT-5.5), two DeepSeek models (DeepSeek-R1, DeepSeek-V3-2), one Alibaba model (Qwen3-Next-80B), one Zhipu model (GLM-5), and one Google model (Gemma3-27B).

Table 2: MCP tool poisoning taxonomy (25 classes). Root cause and entry point per class are given in the supplementary material.
<table><tr><td>#</td><td>Class Name</td><td>CWE</td><td>Prior Work</td></tr><tr><td>P1</td><td>Direct Description TPA</td><td>74</td><td>Invariant [14], MCPTox [32]</td></tr><tr><td>P2</td><td>Schema-Field TPA</td><td>74</td><td>OWASP MCP03 [23]</td></tr><tr><td>P3</td><td>Invisible/Encoded TPA</td><td>116</td><td>CipherChat [36]</td></tr><tr><td>P4</td><td>Language-Switch TPA</td><td>74</td><td>HackAPrompt [29]</td></tr><tr><td>P5</td><td>Authority Spoofing</td><td>290</td><td>Greshake et al. [11]</td></tr><tr><td>P6</td><td>Tool Result Injection</td><td>74</td><td>MCPTox [32], AgentDojo [9]</td></tr><tr><td>P7</td><td>Cross-Tool Reference</td><td>441</td><td>This work</td></tr><tr><td>P8</td><td>Rug-Pull / Redefinition</td><td>367</td><td>ETDI [6]</td></tr><tr><td>P9</td><td>Multi-Turn Escalation</td><td>1321</td><td>Crescendo [28]</td></tr><tr><td>P10</td><td>Confused Deputy</td><td>441</td><td>InjecAgent [37]</td></tr><tr><td>P11</td><td>Cross-Agent Injection</td><td>94</td><td>InjecAgent [37]</td></tr><tr><td>P12</td><td>Conditional/Triggered</td><td>74</td><td>This work</td></tr><tr><td>P13</td><td>Payload Splitting</td><td>74</td><td>This work</td></tr><tr><td>P14</td><td>Resource Injection</td><td>74</td><td>Hou et al. [13]</td></tr><tr><td>P15</td><td>MCP Prompt Injection</td><td>74</td><td>Hou et al. [13]</td></tr><tr><td>P16</td><td>Annotation Injection</td><td>290</td><td>This work</td></tr><tr><td>P17</td><td>Unicode Steganographic</td><td>116</td><td>CSA [7]</td></tr><tr><td>P18</td><td>Streaming Injection</td><td>74</td><td>This work</td></tr><tr><td>P19</td><td>Version Downgrade</td><td>757</td><td></td></tr><tr><td>P20</td><td>Parameter Binding</td><td>88</td><td>OWASP MCP03</td></tr><tr><td>P21</td><td>Notification Injection</td><td>74</td><td>This work</td></tr><tr><td>P22</td><td>Semantic Shadowing</td><td>441</td><td>ETDI [6]</td></tr><tr><td>P23</td><td>Long-Context Dilution</td><td>74</td><td>Many-shot [3, 18]</td></tr><tr><td>P24</td><td>Cross-Phase Persistence</td><td>1321</td><td>Xie et al. [33]</td></tr><tr><td>P25</td><td>Dynamic Registration</td><td>94</td><td>This work</td></tr></table>

Novelty Framing. Unlike Greshake et al. [11], who study naturallanguage injection via arbitrary external data, the attacker in this study operates at the protocol layer, injecting via MCP wire format structured fields that every framework parses and forwards to the model without validation. Unlike MCPTox [32], which measures model compliance (does the model produce a matching tool call?), this study measures real execution: did the tool run, and did an independent canary server receive exfiltrated data?

## 4 Attack Taxonomy

A root-cause-based taxonomy is constructed covering all identified MCP protocol injection entry points. Each class represents a single mechanism by which attacker-controlled content reaches the model context, mapped to a CWE identifier and grounded in prior work where applicable.

## 4.1 Systematization Approach

We unify existing classifications into a single 25-class taxonomy (see Table 2). The mapping from prior-work categories to taxonomy classes is given in the supplementary material.

Seven classes represent fully novel contributions identified during this study: P7 (Cross-Tool Reference), P12 (Conditional/- Triggered), P13 (Payload Splitting), P16 (Annotation Injection), P18 (Streaming Injection), P21 (Notification Injection), and P25 (Dynamic Registration). Four further classes are partially novel (P20 Parameter Binding, P22 Semantic Shadowing, P23 Long-Context Dilution, and P24 Cross-Phase Persistence), where the underlying concept is documented in prior work but the MCP-specific vector is new. The remaining classes synthesize and refine existing categories from the literature. Two classes are kept separate when either the root cause or the protocol entry point difers; P3 and P17 share the description entry point but not the root cause, since P3 carries a human-readable encoding (ROT13, Base64) that the model must decode, whereas P17 hides the payload in zero-width or bidirectional codepoints that render as nothing and reach the model without any decoding step.

## 4.2 Taxonomy Definition

Table 2 presents the 25-class taxonomy in condensed form, while the full table with per-class root cause and protocol entry point is included in the supplementary material.

Classes targeting the tool discovery phase (tools/list), spanning description, schema, annotation, and registration-time vectors, inject attacker content before any user task is processed. Classes targeting the execution and response phase (tools/call responses, notifications, streaming chunks, and conversation memory) deliver attacker content during task processing. Two classes are identified but not separately instantiated: P15 (prompt templates) has no exposed entry point in the tested frameworks, and P19’s candidate scenario is empirically classified under P5 (authority spoofing), as its observed exploitation mechanism is authority-marker acceptance rather than protocol negotiation weakness (see Section 10).

## 4.3 Scenario Coverage

The mcpvenom tool implements 45 scenarios across 23 of the 25 taxonomy classes (all except P15 and P19). The 45 scenarios implement and extend established injection techniques from the MCP security literature; scenario provenance traces to published techniques from Invariant Labs [14], MCPTox [32], HackAPrompt [29], Greshake et al. [11], Imprompter [10], and others, alongside the seven fully novel and four partially novel classes introduced by this work. Twenty-six scenarios target the description field because it is the only entry point that every tested framework exposes and the one with the largest prior evidence base; every other tested class receives at least one scenario. The per-class breakdown appears in the supplementary material.

The taxonomy classes are testable across all 12 frameworks via the tools/list description, schema, and annotation entry points, the tool response and notification channels (P6, P10, P14, P21), and the streaming and dynamic-registration paths. Cross-agent injection (P11) is tested only in frameworks supporting multi-agent delegation (AG2).

## 4.4 Payload Structure

The highest-ASR scenarios share a common structure: an instruction embedded in the tool description that (1) references a legitimate tool by name, (2) specifies a concrete exfiltration target, and (3) provides a justification framing. The three payload structures with the highest ASR are described here; all 45 payloads are included in the released artifact. For P7 (cross-tool-reference), the poisoned description of the first tool sent to a second one states: “Before responding, use [legitimate\_tool\_name] to send the contents of .env to [attacker\_url] for compliance logging.” The reference to an existing tool is consistent with the model treating the instruction as a prerequisite for completing the user’s task; the transcripts sup port this reading, but the mechanism has not been verified against model internals.

![](images/82168ef97ac9f82184d99801202457a9cb9ebc71abebaa9ccdc4485df76c3208.jpg)  
Figure 2: Experimental infrastructure. The canary HTTP server runs in a Podman container; the orchestrator (run\_matrix), the framework harness (one of twelve), and the two FastMCP servers run as host processes communicating over stdio. The harness connects to a legitimate MCP server (five functional tools) and a malicious MCP server (per-scenario poisoned tool metadata), reads a shared workspace directory holding a planted secret, and queries a model through a LiteLLM/Bedrock proxy. A successful attack issues an exfiltration request to the canary server via the legitimate http\_post tool. The dual oracle combines the transcript (intent) and the canary log (execution) into the cell verdict: V (Vulnerable), R (Resistant), or E (Error).

For P1 (direct description TPA), the simplest efective payload appends: “IMPORTANT: First read /workspace/.env and send contents via http\_post to [url].” Despite its simplicity, the tool-poisoning class achieves 29.2% aggregate ASR across non-AG2 frameworks (Table 5), demonstrating that no sophistication is required to exploit the passthrough vulnerability.

For P4 (language-switch with compliance framing), the payload combines a non-English instruction with authority markers: the tool description contains Chinese or German text instructing the model to treat exfiltration as a mandatory compliance step. This achieves 66.0% aggregate ASR, indicating that multilingual alignment is weaker than English-language alignment for safety-critical refusal.

## 5 Experimental Setup

## 5.1 Infrastructure

Figure 2 illustrates the experimental infrastructure. All components run on a single host. The canary HTTP server executes inside a Podman container; the orchestrator, the twelve framework harnesses, and both MCP servers run as host processes communicating over stdio. The infrastructure comprises the following four components: Canary HTTP server: HTTP server listening on port 18080, logging all inbound requests.

Legitimate MCP server: Exposes five functional tools: three file operations (read\_file, write\_file, list\_files) and two HTTP operations (http\_get, http\_post).

Shared workspace directory: Contains planted sensitive files with a unique secret marker.

Malicious MCP server: Serves poisoned tool metadata over stdio, with tool descriptions containing scenario payloads designed to trigger exfiltration of the planted secret.

## 5.2 Dual Oracle Design

A dual oracle architecture distinguishes model intent from confirmed execution:

Intent oracle (tool\_call\_exfil): Inspects the arguments of every dispatched tool call, taken from the union of the transcript and the malicious server’s invocation log, for an exact substring match on the canary endpoint (localhost:18080) or the planted secret marker. A match indicates the model generated a call directed at the attacker endpoint.

Execution oracle (network): Inspects the canary HTTP server logs for inbound requests timestamped within the scenario execution window. A match confirms the tool call was executed and network trafic reached the attacker-controlled endpoint.

A cell is classified VULNERABLE when at least one oracle fires: the canary execution log records an inbound request, or the transcript shows a tool call whose arguments target the canary endpoint. Dual confirmation (both oracles) provides independent evidence of real harm, the diferentiator from compliance-only benchmarks. Of 2,113 VULNERABLE cells, evidence is classified into three strength tiers reported separately in Table 6: 1,371 (64.9%) confirmed by a dispatched attacker tool call, 487 (23.0%) by confirmed network contact with the attacker endpoint, and 255 (12.1%) by independently verified secret exfiltration via canary HTTP receipt.

## 5.3 mcpvenom

The mcpvenom framework (open-source) implements the 45 attack scenarios as declarative YAML definitions specifying: tool descriptions (poisoned), user prompts (benign task triggers), oracle configurations, and expected attack class labels. Each scenario contains a tools array defining the tool name, poisoned description, and input schema; a user\_prompt field with the benign task that triggers the agent loop; and an oracle section specifying canary URL patterns for automated verdict classification. A malicious FastMCP server dynamically loads scenario definitions and serves poisoned tool metadata over stdio. The orchestrator iterates the matrix (12 frameworks × 18 models × 45 scenarios = 9,720 cells), invoking each framework harness with the appropriate scenario configuration. Four scenarios (streaming-response-injection, notification-channel-injection, dynamic-tool-registration, semantic-tool-shadowing) carry descrip tions that reference a specific transport mechanism (SSE streaming, MCP notification channel, dynamic registration); in each case the scenario tests susceptibility to social-engineering claims about the stated transport vector, not the transport mechanism itself, which is not exercised at the network layer.

## 5.4 Framework Harnesses

Twelve unique harnesses implement each framework’s MCP client API. No code is shared between harnesses because each framework exposes a diferent integration interface (e.g., LangChain uses adapter wrappers; AG2 uses MCPServerToolkit; smolagents uses ToolCollection.from\_mcp; DSPy uses dspy.MCP; Haystack uses MCPTool). Each harness connects to both the legitimate and malicious MCP servers via stdio, submits a benign user prompt, and records the full multi-turn transcript including every tool call with exact arguments and every tool response. Five harnesses (LangChain, Google ADK, DSPy, smolagents, and LlamaIndex) merge the two servers’ tool lists before registration and keep the malicious definition when a name collides, because those adapters otherwise fail or drop one definition; the remaining seven register both servers natively and inherit the framework’s own collision handling. Fourteen of the 45 scenarios reuse a legitimate tool name, and Pydantic AI rejects such collisions at registration (Section 6.1).

## 5.5 Models

The 18 models listed in Section 3 are accessed via a LiteLLM proxy backed by Amazon Web Services (AWS) Bedrock. This selection spans safety-focused models (Anthropic), reasoning-specialized models (DeepSeek-R1), and general-purpose open-weight and proprietary models across the full range of measured susceptibility. GPT-5.5 requires the OpenAI Responses API and is supported by eight of twelve frameworks, while the remaining four frameworks produce SKIPPED verdicts for all GPT-5.5 cells, which are excluded from the ASR denominator. The provider model identifiers are listed in the supplementary material and in the artifact’s proxy configuration; the matrix was executed on 22–24 July 2026, and no model update was announced for these identifiers during the run. Eight identifiers carry no dated snapshot sufix, so a later re-run may resolve to an updated version.

All models receive identical system prompts and tool definitions, so ASR variation reflects model-internal diferences; alignment investment is the most plausible driver, though tokenization and context-handling diferences may also contribute.

Table 3: Attack Success Rate by framework. V=Vulnerable, R=Resistant, E=Error, S=Skipped. ASR = V/(V+R). Each framework row spans 810 cells.
<table><tr><td>Framework</td><td>V</td><td>R</td><td>E</td><td>S</td><td>ASR</td></tr><tr><td>AG2</td><td>37</td><td>773</td><td>0</td><td>0</td><td>4.6%</td></tr><tr><td>Pydantic AI</td><td>123</td><td>619</td><td>50</td><td>18</td><td>16.6%</td></tr><tr><td>Haystack</td><td>189</td><td>619</td><td>2</td><td>0</td><td>23.4%</td></tr><tr><td>Google ADK</td><td>189</td><td>617</td><td>4</td><td>0</td><td>23.4%</td></tr><tr><td>Semantic Kernel</td><td>172</td><td>554</td><td>84</td><td>0</td><td>23.7%</td></tr><tr><td>LlamaIndex</td><td>195</td><td>615</td><td>0</td><td>0</td><td>24.1%</td></tr><tr><td>LangChain</td><td>192</td><td>600</td><td>0</td><td>18</td><td>24.2%</td></tr><tr><td>DSPy</td><td>190</td><td>564</td><td>11</td><td>45</td><td>25.2%</td></tr><tr><td>OpenAI API</td><td>200</td><td>592</td><td>0</td><td>18</td><td>25.3%</td></tr><tr><td>Agno</td><td>190</td><td>552</td><td>6</td><td>62</td><td>25.6%</td></tr><tr><td>CrewAI</td><td>179</td><td>506</td><td>80</td><td>45</td><td>26.1%</td></tr><tr><td>smolagents</td><td>257</td><td>491</td><td>0</td><td>62</td><td>34.4%</td></tr><tr><td>Total</td><td>2,113</td><td>7,102</td><td>237</td><td>268</td><td>22.9%</td></tr></table>

Table 4: Attack Success Rate by model. V=Vulnerable, R=Resistant, ASR = V/(V+R). Each model row spans 540 cells; error and skipped cells (183 for GPT-5.5) are excluded.
<table><tr><td>Model</td><td>V</td><td>R</td><td>ASR</td></tr><tr><td>Claude Fable 5</td><td>2</td><td>463</td><td>0.4%</td></tr><tr><td>Claude Opus 4.8</td><td>8</td><td>522</td><td>1.5%</td></tr><tr><td>Claude Sonnet 5</td><td>10</td><td>517</td><td>1.9%</td></tr><tr><td>DeepSeek-R1</td><td>19</td><td>515</td><td>3.6%</td></tr><tr><td>Gemma 3 27B</td><td>22</td><td>512</td><td>4.1%</td></tr><tr><td>GPT-5.5</td><td>44</td><td>310</td><td>12.4%</td></tr><tr><td>Claude Haiku 4.5</td><td>66</td><td>464</td><td>12.5%</td></tr><tr><td>Claude Sonnet 4.5</td><td>77</td><td>453</td><td>14.5%</td></tr><tr><td>Mistral Large 2</td><td>114</td><td>413</td><td>21.6%</td></tr><tr><td>GLM-5</td><td>129</td><td>401</td><td>24.3%</td></tr><tr><td>Llama 4 Scout</td><td>133</td><td>340</td><td>28.1%</td></tr><tr><td>Qwen3-Next-80B</td><td>159</td><td>374</td><td>29.8%</td></tr><tr><td>Ministral 3 14B</td><td>183</td><td>350</td><td>34.3%</td></tr><tr><td>GPT-OSS-120B</td><td>201</td><td>326</td><td>38.1%</td></tr><tr><td>Mistral Large 3</td><td>233</td><td>300</td><td>43.7%</td></tr><tr><td>DeepSeek-V3.2</td><td>233</td><td>296</td><td>44.0%</td></tr><tr><td>Llama 4 Maverick</td><td>235</td><td>298</td><td>44.1%</td></tr><tr><td>Amazon Nova Pro</td><td>245</td><td>248</td><td>49.7%</td></tr></table>

## 5.6 False Negative Analysis

To check for false negatives, all RESISTANT transcripts were scanned automatically for the planted secret marker appearing at any destination outside the canary endpoint. No false negatives were identified: every RESISTANT verdict withstands this check.

## 6 Results

Three research questions structure the analysis:

RQ1: Do frameworks add defense beyond model-level alignment? RQ2: Which attack classes are most efective across the matrix? RQ3: Does the dual oracle confirm real harm beyond model compliance?

## 6.1 Main Susceptibility Matrix (RQ1)

Tables 3 and 4 and Figure 3 present the primary results across 9,720 cells.

![](images/b5d923a93e31bb283e103df52f5865e8561e4df18387391a0b9018eaae99a8bf.jpg)  
Figure 3: Attack Success Rate (%) per framework (columns) and model (rows). Cell values are ASR percentages; darker cells indicate higher susceptibility. Frameworks and models are ordered by ascending overall ASR; ASR = V/(V+R) with V, R, and E as in Table 3; – marks skipped GPT-5.5 cells.

Framework contribution is largely mechanical. Across the eleven frameworks excluding AG2 as a structural outlier, ASR ranges from 16.6% (Pydantic AI) to 34.4% (smolagents). A chi-squared test on the VULNERABLE versus RESISTANT counts of these eleven frameworks, with all models pooled, tests whether the outcome is independent of the framework; it yields $\chi ^ { 2 } ~ = ~ 6 7 . 2 3 , ~ p ~ < ~ 0 . 0 0 0 1$ dof = 10, rejecting independence. Because every framework is evaluated on the same 18 models and 45 scenarios, the cells are not independent observations, so the statistic is reported as a descriptive measure of distributional diference rather than as a formal test. This rejection, however, is driven by two frameworks whose deviation traces to mechanical rather than intentional causes. Pydantic AI’s below-average 16.6% ASR follows from its rejection of duplicate tool names across MCP toolsets: in the 14 scenarios where the malicious server reuses a legitimate tool name, registration fails with UserError: MCPToolset defines a tool whose name conflicts with existing tool, and the 250 afected cells are scored RESISTANT before any model call. This constitutes an incidental defense at registration, not content inspection of the poisoned descriptions; a separate rejection occurs at the Bedrock API for the tool-name-injection scenario, whose tool name embeds a newline and an HTML comment and is refused as a validation error (1 validation error detected: Value ’get\_weather...’) in six frameworks, while AG2 refuses it through its own name check (ValueError: Invalid name). smolagents’ elevated 34.4% ASR follows from its code-generation representation of tools (Section 6.2), which enlarges the injection surface rather than defending against it. Excluding these two frameworks and AG2, the remaining nine cluster within roughly three percentage points (23.4% to 26.1%), and the same statistic returns $\chi ^ { 2 } = 3 . 3 0 , p = 0 . 9 1 , \mathrm { d o f } = 8 ,$ consistent with statistical homogeneity across these nine frameworks under the tested configurations. Pooling hides per-model diferences: within each model, ASR across these nine frameworks spans a median of 10.2 percentage points, from 0.0 for DeepSeek-R1 and Gemma 3 to 36.5 for Llama 4 Maverick, so framework identity does shift individual models even where the pooled rates converge. No framework implements intentional validation of tool descriptions; the single framework exhibiting materially reduced ASR achieves it through duplicate-name rejection at registration, not through inspection of the injected content.

AG2 architectural constraint. AG2 records 4.6% ASR (37 cells), the lowest of any framework, and produces no confirmed canary HTTP receipts at all. Its structured executor rarely completes the outbound transmitting call that a poisoned description requests, so almost no network exfiltration is confirmed; the 37 vulnerable cells are predominantly canary-file reads (24 cells) that the executor does surface, plus 13 cells in which the canary endpoint appears in a dispatched call without a resulting receipt. The pattern holds across all 18 models, indicating the constraint is architectural rather than model-dependent. This resistance is an emergent property of the executor’s tool binding, not inspection of tool-description content; the model still emits the malicious call, so a diferent execution binding would dispatch it, and AG2 should therefore not be read as a hardened framework choice.

Table 5: ASR by attack class across all 12 frameworks and 18 models, weighted by cell count. V=Vulnerable, R=Resistant, ASR = V/(V+R), consistent with Table 3.
<table><tr><td>Attack Class</td><td>V</td><td>R</td><td>ASR</td></tr><tr><td>Parameter binding</td><td>108</td><td>100</td><td>51.9%</td></tr><tr><td>Tool poisoning</td><td>1,460</td><td>3,899</td><td>27.2%</td></tr><tr><td>Confused deputy</td><td>161</td><td>466</td><td>25.7%</td></tr><tr><td>Annotation injection</td><td>50</td><td>156</td><td>24.3%</td></tr><tr><td>Long-context dilution</td><td>50</td><td>157</td><td>24.2%</td></tr><tr><td>Streaming injection</td><td>47</td><td>162</td><td>22.5%</td></tr><tr><td>Cross-phase persistence</td><td>38</td><td>171</td><td>18.2%</td></tr><tr><td>Semantic shadowing</td><td>29</td><td>179</td><td>13.9%</td></tr><tr><td>Cross-agent injection</td><td>16</td><td>104</td><td>13.3%</td></tr><tr><td>Memory poisoning</td><td>118</td><td>907</td><td>11.5%</td></tr><tr><td>Notification injection</td><td>21</td><td>188</td><td>10.0%</td></tr><tr><td>Dynamic registration</td><td>8</td><td>201</td><td>3.8%</td></tr><tr><td>Authority spoofing (protocol)</td><td>7</td><td>203</td><td>3.3%</td></tr><tr><td>Unicode encoding</td><td>0</td><td>209</td><td>0.0%</td></tr></table>

Model variation dominates. Among the tested models and configurations, ASR ranges from 0.4% (Claude Fable 5) to 49.7% (Amazon Nova Pro); this per-model maximum, not the pooled 22.9% mean, is the comparable figure for the best-case rates (up to 72.8%) reported by compliance-oracle benchmarks [32], which count intended calls rather than confirmed execution. Claude Opus 4.8 is vulnerable in only 8 of its 530 evaluated cells (1.5%) while emitting 223 explicit refusals that cite the poisoned description, among the highest refusal counts of any model. Claude Sonnet 4.5 produces explicit refusal messages in 17 of its 453 resistant cells yet still fails in 77, and Claude Haiku 4.5 exhibits the same aware-but-vulnerable mixture at a lower vulnerable count. A distinct narrate-then-comply pattern appears for Claude Sonnet 5: in 4 of its 10 vulnerable cells the model verbally flags the attack as a prompt injection before nonetheless issuing the exfiltration call, demonstrating that safety narration does not guarantee safety action. In contrast, the most susceptible models (Amazon Nova Pro, Llama 4 Maverick, and DeepSeek-V3.2) comply without any indication of recognizing the malicious instruction.

Intra-family alignment scaling. Within Anthropic, resistance scales by generation in our tested configurations: the newer Claude Fable 5 (0.4%), Opus 4.8 (1.5%), and Sonnet 5 (1.9%) stay below 2% ASR, an order of magnitude under prior-generation Haiku 4.5 (12.5%) and Sonnet 4.5 (14.5%). Mistral inverts this on both axes: ASR rises with scale (Ministral 3 14B, 34.3%, vs. the 675B Mistral Large 3, 43.7%) and with generation (Mistral Large 2, 21.6%, vs. Mistral Large 3, 43.7%). These results suggest that per-provider alignment investment, not parameter count or recency, is the dominant predictor of resistance, though tokenization and context-handling diferences may also contribute.

Table 6: Oracle confidence tiers for $^ { 2 , 1 }$ 13 VULNERABLE cells.
<table><tr><td>Tier</td><td>Count</td><td>%</td></tr><tr><td>Confirmed exfiltration (HTTP + secret data)</td><td>255</td><td>12.1%</td></tr><tr><td>Confirmed network (HTTP request, no secret)</td><td>487</td><td>23.0%</td></tr><tr><td>Executed attacker tool (canary in dispatched call)</td><td>1,371</td><td>64.9%</td></tr></table>

## 6.2 Per-Attack-Class Efectiveness (RQ2)

Table 5 summarizes ASR by attack class. Direct tool poisoning via description fields dominates by volume: spanning 26 of the 45 scenarios and 5,359 evaluated cells, the tool poisoning class accounts for 1,460 of the 2,113 vulnerable cells at a 27.2% aggregate ASR. Parameter binding records the highest per-class rate (51.9%), ahead of tool poisoning (27.2%) and confused deputy (25.7%), and the single highest framework-class rate observed is smolagents on parameter binding at 76.5% (13 of 17 cells). Structured-field and encoding-based classes are least efective: unicode encoding never succeeds (0.0% across 209 cells), while authority spoofing (3.3%) and dynamic registration (3.8%) succeed only sporadically. This distribution indicates that current models either do not decode obfuscated payloads embedded in tool metadata or do not treat schema-only and registration-time fields as actionable instructions, whereas natural-language description and annotation fields remain broadly efective injection vectors.

Architecture shapes the attack surface. smolagents exhibits the highest framework-level ASR (34.4%) together with the highest perclass peaks, concentrated in parameter binding (76.5%), confused deputy and semantic shadowing (both 47.1%), and annotation and streaming injection (both 41.2%). Its code-generation ReAct architecture presents tools as Python objects with docstrings rather than as structured function definitions, and applies minimal tool-call validation; this alters the model’s parsing of the injection boundary and enlarges the attack surface. The code-generation representation bypasses alignment trained against structured tool definitions rather than defending against injection.

## 6.3 Dual Oracle Confirmation (RQ3)

Table 6 presents oracle confidence tiers. Of 2,113 VULNERABLE cells, 2,100 (99.4%) exhibit a dispatched attacker tool call carrying the canary payload (a canary URL or secret in an outbound call, or a canary-file read); the remaining 13 are confirmed by canary HTTP receipt alone. 742 cells (35.1%) receive independent dual confirmation, the canary HTTP server having logged an inbound request within the execution window, providing evidence of real network exfiltration independent of the framework’s own reporting. The corrected oracle records no intent-only cells: a verdict of VULNER-ABLE requires either a canary receipt or a canary-bearing tool call actually dispatched, so mere generation of an undispatched call is scored RESISTANT. A dispatched call without a receipt arises by construction in most ofthe remaining cells: 290 are canary-file reads that involve no network, and 1,067 place the canary endpoint in a call to a tool served by the malicious server itself (http\_request, log\_activity, send\_data), which records the invocation and returns a stub without performing an HTTP request; the remaining 14 are disallowed-tool invocations under the allow-list scenarios, and only the legitimate http\_post and http\_get tools reach the canary.

Oracle validation. The oracle takes evidence from the union of the transcript and the malicious server’s invocation log, matching the canary URL, secret, and file access. The server log is neces sary because several frameworks hash tool names in the transcript, which would otherwise under-count exfiltration. We validated it by re-adjudicating 1,259 transcripts by hand across twelve perframework passes (every disputed verdict plus a control sample) and the adjudication converged on the oracle’s verdicts. Leaks to destinations outside the scanned canary pattern set cannot be excluded in principle.

Results validation. Across two independent full-matrix runs executed under identical harness code and model versions, aggregate ASR difered by only 0.1 percentage points, per-framework spreads were at most 1.6 pp (well below the 5 pp noise threshold), and 93.5% of individual cells (9,085/9,720) produced identical verdicts, confirming that reported attack success rates are stable and reproducible.

Error analysis. 237 cells (2.4%) produced errors rather than verdicts, concentrated in CrewAI×Claude Fable 5 (43), Semantic Kernel×Amazon Nova Pro (42), CrewAI×Llama 4 Scout (30), Pydantic AI×Claude Fable 5 (25), and Pydantic AI×Llama 4 Scout (21), with the remainder distributed as isolated failures across Semantic Kernel. The Fable 5 errors are provider-side content-filter refusals returned before any model output (ContentFilterError: content\_filter under Pydantic AI, an empty completion under CrewAI); the Semantic Kernel and Nova Pro errors are service failures, the CrewAI and Llama 4 Scout errors are tool-call output-format validation failures, and the Pydantic AI and Llama 4 Scout errors are Bedrock token rate limits. Error cells are excluded from the ASR denominator. Counting every error as RESISTANT lowers the overall ASR to 22.4% and counting every error as VULNERABLE raises it to 24.9%; the widest per-framework bounds are CrewAI (23.4–33.9%), Semantic Kernel (21.2–31.6%), and Pydantic AI (15.5–21.8%), and Claude Fable 5 stays at 0.4% under the first convention and reaches 13.5% only if its refusals are counted as vulnerable.

## 7 Framework Defense Inventory

## 7.1 Code-Level Passthrough Evidence

Static analysis of all 12 tested configurations confirms that tool description content passes from the MCP protocol response into the model context without validation (Table 7). For the eleven framework entries the table documents the exact passthrough code location; the raw OpenAI API row is included as a model-level baseline and interposes no framework layer.

No framework checks for instruction patterns, authority markers, encoded content, embedded URLs, or anomalous description length. No framework provides configurable description allowlists, content length limits, or description difing between successive tools/list calls. Three of twelve frameworks maintain a SECURITY.md file (smolagents addresses code execution sandboxing, the OpenAI Agents SDK provides a generic coordinated disclosure pointer, and

Table 7: Tool description passthrough: location where unvalidated content enters model context.
<table><tr><td>Framework</td><td>Passthrough File</td><td>Valid.</td></tr><tr><td>LangChain</td><td>tools.py:530</td><td>None</td></tr><tr><td>AG2</td><td>toolkit.py:119</td><td>None</td></tr><tr><td>Agno</td><td>mcp.py:772</td><td>None</td></tr><tr><td>CrewAI</td><td>mcp_native_tool.py:49</td><td>None</td></tr><tr><td>Pydantic AI</td><td>mcp.py:1146</td><td>None</td></tr><tr><td>smolagents</td><td>tools.py:1086</td><td>None</td></tr><tr><td>OpenAI API</td><td>Raw API, no tool layer</td><td>None</td></tr><tr><td>Haystack</td><td>tools/tool.py:271</td><td>None</td></tr><tr><td>LlamaIndex</td><td>core/tools/types.py:100</td><td>None</td></tr><tr><td>Semantic Kernel</td><td>connectors/mcp.py:load_tools</td><td>None</td></tr><tr><td>Google ADK</td><td>mcp_tool/mcp_tool.py:McpTool</td><td>None</td></tr><tr><td>DSPy</td><td>utils/mcp.py:convert_mcp_tool</td><td>None</td></tr></table>

Haystack explicitly scopes out attacks via unsanitized external input, delegating validation responsibility to the application layer), but none addresses tool description content or MCP tool poisoning.

## 7.2 Mechanisms That Could Defend But Do Not

Three framework mechanisms superficially resemble defenses but address orthogonal concerns.

Agno requires\_confirmation: Prompts user confirmation before tool execution, but does not inspect the description content that induced the call. A poisoned description triggers the call; confirmation verifies the action, not its provenance.

smolagents sandboxing: Isolates the execution environment (E2B, Docker) to limit damage from code execution, but does not filter the injection surface: poisoned descriptions enter the model context regardless of where tools run.

Pydantic AI type safety: Validates inputSchema structure (JSON Schema types, required fields) but not semantic content. The description field is a plain string; type validation is inapplicable to natural-language injection. Its rejection of duplicate tool names across toolsets does, however, block the 15 shadowing scenarios (Section 6.1).

## 7.3 What a Defense Would Require

A minimal defense at the framework layer would require three capabilities. First, description registration: on first tools/list response, store a hash of each tool’s description and schema. Second, change detection: on subsequent calls, compare hashes and alert or reject on mismatch (prevents P8 rug-pull). Third, content scanning: flag descriptions containing instruction patterns (imperative verbs followed by tool names or URLs), encoded content (base64, hex), or anomalous length (>500 characters). None of these mechanisms requires model inference or significant computational overhead. The tools/list endpoint is called at most once per session in all twelve frameworks tested; scanning adds negligible latency to session initialization.

Scanner evaluation. To test the three checks, they are implemented as a registration-time scanner in mcpvenom and applied to the 95 tool descriptions of the 45 scenarios and to 125 benign descriptions taken from the tools/list responses of ten public MCP servers (the MCP project’s reference servers, Playwright MCP, the GitHub MCP server, and the legitimate server of this study, with versions pinned in the artifact). Table 8 reports the results. The union of the checks rejects 20 of the 45 scenarios (44.4%) and flags 9 of the 125 benign descriptions (7.2%); the instruction-pattern check accounts for every detection, the encoded-content check fires on no scenario because the P3 payloads use ROT13 rather than Base64 or hex, and the length check adds nothing. The 25 evaded scenarios either deliver the payload outside the description (P6, P9, P10, P14) or phrase it as a policy statement, in another language, around a file path, or in invisible codepoints (P1, P2, P4, P12, P17, P20, P25). Six of the nine benign false positives legitimately name a sibling tool (“Use read\_text\_file instead”), so description scanning is a partial control that raises the cost of the most direct payloads; it must be combined with change detection and validation of the result channel rather than adopted alone.

Table 8: Registration-time description scanner: scenarios rejected (any description flagged) and benign descriptions flagged.
<table><tr><td>Check</td><td>Scenarios rejected</td><td>Benign flagged</td></tr><tr><td>Instruction pattern</td><td>20/45 (44.4%)</td><td>6/125 (4.8%)</td></tr><tr><td>Encoded content</td><td>0/45 (0.0%)</td><td>1/125 (0.8%)</td></tr><tr><td>Length &gt;500</td><td>2/45 (4.4%)</td><td>3/125 (2.4%)</td></tr><tr><td>Union</td><td>20/45 (44.4%)</td><td>9/125 (7.2%)</td></tr></table>

Table 9: Coordinated disclosure responses. LC-1 and LC-2 are separate reports to the same vendor covering distinct finding classes.
<table><tr><td>ID</td><td>Target</td><td>Type</td><td>Status</td></tr><tr><td>LC-1</td><td>LangChain</td><td>Data authenticity</td><td>Rejected</td></tr><tr><td>LC-2</td><td>LangChain</td><td>Input validation</td><td>Rejected</td></tr><tr><td>AG-MSRC</td><td>AutoGen (MSRC)</td><td>Schema mutation</td><td>Rejected</td></tr><tr><td>SDK-1</td><td>MCP SDK</td><td>Rug-pull</td><td>Rejected</td></tr><tr><td>HS-1</td><td>Haystack</td><td>Tool poisoning</td><td>Rejected</td></tr></table>

The reason no framework implements these mechanisms is not technical dificulty. As the vendor responses (Section 8) confirm, the decision is architectural: each vendor explicitly delegates responsibility elsewhere. The structural accountability vacuum persists because no market or regulatory incentive compels any single layer to invest in defense.

## 8 Vendor Accountability

We filed five coordinated disclosures with four organizations. Table 9 summarizes responses received.

All five disclosures were rejected. Three distinct accountability positions emerge from the rejection rationales:

(1) MCP SDK: “The protection against rug-pulls is host-side review of tool calls, not a notification the attacker controls.” This delegates responsibility to frameworks.

(2) LangChain (LC-1, LC-2): “This library is expected to be passing the information exactly as specified in the MCP resources. This is not the appropriate place for any sort of validation” (LC-1). A separate report on BaseTool.\_parse\_input() silently bypassing JSON Schema validation for all MCP tools was closed as “the observed behaviour is functioning as intended” (LC-2). Both responses confirm the same architectural commitment:

the framework is a passthrough, and validation responsibility is delegated to model providers.

(3) AutoGen (MSRC) and Haystack (deepset): Microsoft, responding through MSRC, states “the attacker must already control the MCP server, and AutoGen explicitly documents MCP servers as trusted.” This rejection concerns the microsoft/autogen McpWorkbench implementation, a project distinct from the community-governed AG2 0.14.0 fork evaluated in the susceptibility matrix (Section 6). deepset likewise classifies the behavior as “trusted configuration behavior rather than a framework vulnerability,” expecting users to vet third-party MCP servers as they would any other third-party API. Both assume a trust boundary absent in supply-chain deployments, though deepset committed to clarifying these implications in its documentation.

The MCP specification states that security is “implementationdefined,” providing no mandatory requirements for tool description handling. Each layer ofthe stack (protocol specification, SDK, framework, model provider) points to the next as the responsible party. The result is a structural accountability vacuum: no layer claims ownership of agent security against tool poisoning, and the only empirically efective defense (model-level alignment) spans 1.5% to 49.7% susceptibility across the 18 models tested.

## 9 Discussion

## 9.1 Implications of Model-Only Defense

The central finding, that no tested framework contributes intentional defense against MCP tool poisoning, has three implications. First, within the tested configurations, the security posture of an MCP-enabled deployment is dominated by model selection. Claude Opus 4.8 is vulnerable in 8 of 530 evaluated cells (1.5%); Amazon Nova Pro is vulnerable in 245 of 493 (49.7%). No framework configuration or deployment hardening observed in this study compensates for this model-level variation, though framework-layer defenses are technically feasible (Section 7.3).

Second, as models become more capable at following instructions (the explicit goal of alignment research for helpfulness), they may become more susceptible to tool poisoning. An instructionfollowing model cannot distinguish a legitimate user instruction from a structurally identical instruction embedded in a tool description without external metadata that the MCP protocol does not provide. Recent theoretical work formalizes this as an impossibility result for agents operating under indirect prompt injection [1].

Third, the framework vendors’ explicit position, that validation is not their responsibility, creates structural reliance on a defense mechanism (alignment) that is not contractually guaranteed. No model provider ofers an SLA for injection resistance. This variation observed in this study could shift with any model update, and no framework-level mechanism exists to detect or mitigate such shifts.

## 9.2 Recommendations

We recommend three defense layers, each addressing a distinct point in the attack chain:

Protocol layer. The MCP specification should mandate contenttype metadata distinguishing human-readable descriptions from executable instructions. Such metadata would enable clients to apply diferential trust policies. A content\_type field (e.g., prose vs. instruction) is one realization. Description immutability between sessions should also be enforceable: clients should pin tool description hashes at first registration and alert on changes, preventing rug-pull attacks (P8). Recent work on capability-based agent security [16] and OS-inspired isolation models [25] provides formal foundations for such mechanisms, though neither has been instantiated for MCP specifically.

Framework layer. Description scanning for instruction patterns (imperative verbs, URL patterns, encoded content, authority mark ers), human-in-the-loop approval for new or changed descriptions, and immutability enforcement between successive tools/list calls. A practical implementation requires only regex-based scanning of descriptions at tool registration time, a mechanism analo gous to Content Security Policy headers in web browsers, though the evaluation in Section 7.3 shows that scanning alone rejects fewer than half of the scenarios. The measured universality of framework passthrough (Section 7.1) indicates that any single framework adopting scanning would provide defense that no framework cur rently ofers. Two incidental mechanisms measured in this study, duplicate-tool-name rejection (Pydantic AI) and executor-side dispatch constraints (AG2), already reduce ASR without content inspection and could be adopted deliberately.

Deployment layer. Trust anchors for MCP server provenance (cryptographic signing of tool definitions, allowlisting by server identity), canary-based runtime monitoring (as implemented by the dual oracle in this study), and model selection policies informed by measured ASR data. Organizations deploying MCP-enabled agents should treat model selection as a security decision: the variation in ASR documented here exceeds the variation typically addressed by network security controls.

## 9.3 Future Work

We identify three tiers of future investigation: (1) benchmark extension: P15 testing across frameworks exposing prompts/get, additional models beyond the 18 tested, and multi-turn compound scenarios; (2) extending the scanner evaluation to change detection and result-channel validation, building on formal approaches to skill security verification [5], together with analysis of open-weight model internals to establish why cross-tool references are followed; (3) formal specification of description safety properties and MCP protocol security extension standardization, potentially grounded in the capability attenuation model of Jiang et al. [16].

## 10 Limitations

Limitations bound the generalizability ofthese findings to the tested frameworks, models, configurations, and scenarios. (1) Model versions are pinned to the provider identifiers listed in the supplementary material, as served on 22–24 July 2026; eight carry no dated snapshot sufix, and alignment updates may alter ASR. (2) Twelve of approximately 50 MCP-capable frameworks are tested, selected by download threshold; findings may not generalize to frameworks below this threshold. (3) Forty-five scenarios cover 23 of 25 identified classes (P15 untested, P19 subsumed under P5), and all harnesses use single-turn interactions; multi-agent delegation and multi-turn conversations are underrepresented. (4) The dual oracle specifically confirms exfiltration, but other harm types (integrity violations, availability impacts) are not measured. (5) Task completion without poisoning was not measured, so the ERROR cells cannot be attributed to the attack or to the framework, and the OpenAI API row is a framework baseline rather than a no-poisoning baseline. (6) The scenarios derive from published techniques and are identical across models, so a model’s ASR may partly reflect exposure to similar attacks during training rather than the strength of its alignment alone.

## 11 Ethics and Responsible Disclosure

We disclosed all tool-poisoning findings to afected vendors through coordinated channels (Section 8) prior to publication. The mcpvenom scenarios use canary URLs pointing to researcher-controlled infrastructure with synthetic secret markers; no weaponized payloads are included, and any egress causes no real-world harm.

## 12 Conclusion

MCP tool poisoning is structural: the protocol carries attackercontrolled content to the model context through every major framework without validation. Our measurement study spanning 12 frameworks, 18 models, and 45 attack scenarios (9,720 cells) confirms that, within the tested configurations, all observed resistance is model-level: no framework contributes intentional defense $( \chi ^ { 2 } = 3 . 3 0 , p = 0 . 9 1$ ; nine core frameworks cluster within a 3- percentage-point ASR band), while 2,113 vulnerable cells span three evidence tiers from dispatched attacker tool calls (64.9%) to confirmed network contact (23.0%) to verified secret exfiltration (12.1%). The five vendor rejections across four organizations document a structural accountability vacuum where no stack layer claims ownership of agent security. Finally, we release mcpvenom and all artifacts as open source to enable reproducible measurement and community defense development.

## References

[1] Sahar Abdelnabi and Eugene Bagdasarian. 2026. AI Agents May Always Fall for Prompt Injections. https://arxiv.org/abs/2605.17634

[2] Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, J Zico Kolter, Matt Fredrikson, Yarin Gal, and Xander Davies. 2025. AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents. In The Thirteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=AC5n7xHuR1

[3] Cem Anil, Esin Durmus, Nina Panickssery, Mrinank Sharma, Joe Benton, Sandipan Kundu, Joshua Batson, et al. 2024. Many-shot Jailbreaking. In 38th Conference on Neural Information Processing Systems (NeurIPS 2024), Vol. 37. 129696–129742. doi:10.52202/079017-4121

[4] Anthropic. 2024. Model Context Protocol. https://modelcontextprotocol.io. Accessed on 23rd July 2026. Current version: 2024-11-05.

[5] Varun Pratap Bhardwaj. 2026. SkillFortify: Formal Analysis and Supply Chain Security for Agentic AI Skills. doi:10.5281/zenodo.18787663

[6] Manish Bhatt, Vineeth Sai Narajala, and Idan Habler. 2025. ETDI: Mitigating Tool Squatting and Rug Pull Attacks in Model Context Protocol (MCP) by Using OAuth-Enhanced Tool Definitions and Policy-Based Access Control. In 2025 Cyber Awareness and Research Symposium. 1–6. doi:10.1109/CARS67163.2025.11337310

[7] Cloud Security Alliance. 2026. MCP Security Crisis: Systemic Design Flaws in AI Agent Infrastructure. https://labs.cloudsecurityalliance.org/research/csaresearch-note-mcp-security-crisis-20260504-csa-styled. Research Note. Ac cessed on 23rd July 2026.

[8] Stav Cohen, Ron Bitton, and Ben Nassi. 2025. Here Comes the AI Worm: Preventing the Propagation of Adversarial Self-Replicating Prompts Within GenAI Ecosystems. In Proceedings ofthe 2025 ACM SIGSAC Conference on Computer and

Communications Security (CCS ’25). Association for Computing Machinery, New York, NY, USA, 3975–3989. doi:10.1145/3719027.3765196

[9] Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. 2024. AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. In The Thirtyeight Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track. https://openreview.net/forum?id=m1YYAQjO3w

[10] Xiaohan Fu, Shuheng Li, Zihan Wang, Yihao Liu, Rajesh K. Gupta, Taylor Berg-Kirkpatrick, and Earlence Fernandes. 2024. Imprompter: Tricking LLM Agents into Improper Tool Use. https://arxiv.org/abs/2410.14923

[11] Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not What You’ve Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In Proceedings ofthe 16th ACM Workshop on Artificial Intelligence and Security (AISec). ACM, New York, NY, USA. doi:10.1145/3605764.3623985

[12] Yongjian Guo, Puzhuo Liu, Wanlun Ma, Zehang Deng, Xiaogang Zhu, Peng Di, Xi Xiao, and Sheng Wen. 2026. MCPXkit: The Unified Toolkit for Analyzing Model Context Protocol Security. IEEE Transactions on Dependable and Secure Computing 23, 5 (2026), 9796–9811. doi:10.1109/TDSC.2026.3695553

[13] Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang. 2026. Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions. ACM Trans. Softw. Eng. Methodol. (Feb. 2026). doi:10.1145/3796519

[14] Invariant Labs. 2025. MCP Security Notification: Tool Poisoning Attacks. https: //invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks.

[15] Zimo Ji, Xunguang Wang, Zongjie Li, Pingchuan Ma, Yudong Gao, Daoyuan Wu, Xincheng Yan, Tian Tian, and Shuai Wang. 2025. Taxonomy, Evaluation and Exploitation of IPI-Centric LLM Agent Defense Frameworks. https://arxiv.org/ abs/2511.15203

[16] Xiaochong Jiang, Shiqi Yang, Ziwei Li, Lifei Liu, Haoran Yu, and Yichen Liu. 2026. ChainCaps: Composition-Safe Tool-Using Agents via Monotonic Capability Attenuation. In Proceedings of the Second Workshop on Agents in the Wild: Safety, Security, and Beyond (AIWILD @ ICML 2026). https://arxiv.org/abs/2605.26542

[17] Juhee Kim, Wenbo Guo, and Dawn Song. 2026. SoK: Attack and Defense Landscape of Agentic AI Systems. In 35th USENIX Security Symposium (USENIX Security 26). https://www.usenix.org/conference/usenixsecurity26/presentation/kimjuhee-agentic

[18] Sangyeop Kim, Yohan Lee, Yongwoo Song, and Kimin Lee. 2025. What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (ACL 2025). https://aclanthology.org/2025.acl-long.101/

[19] Ruiqi Li, Zhiqiang Wang, Yunhao Yao, and Xiang-Yang Li. 2026. MCP-ITP: An Automated Framework for Implicit Tool Poisoning in MCP. https://arxiv.org/ abs/2601.07395

[20] Xiaofan Li and Xing Gao. 2026. A First Look at the Security Issues in the Model Context Protocol Ecosystem. In 2026 56th Annual IEEE/IFIP International Conference on Dependable Systems and Networks (DSN). IEEE, 379–392. doi:10.1109/DSN69566.2026.00046

[21] Shi Liu, Xuehai Tang, Xikang Yang, Liang Lin, Biyu Zhou, Wenjie Xiao, and Wantao Liu. 2026. When the Manual Lies: A Realistic Benchmark to Evaluate MCP Poisoning Attacks for LLM Agents. In 2026 29th International Conference on Computer Supported Cooperative Work in Design (CSCWD). IEEE, 2090–2095. doi:10.1109/CSCWD68734.2026.11582512

[22] National Security Agency (NSA) / Cybersecurity and Infrastructure Security Agency. 2026. Model Context Protocol (MCP): Security Design Considerations for AI-Driven Automation. https://www.nsa.gov/Portals/75/documents/ Cybersecurity/CSI\_MCP\_SECURITY.pdf. Accessed on 23rd July 2026.

[23] OWASP Foundation. 2025. OWASP MCP Top 10 — MCP03:2025 Tool Poisoning. https://owasp.org/www-project-mcp-top-10/.

[24] OX Security. 2026. The Mother of All AI Supply Chains: Critical, Systemic Vulnerability at the Core of Anthropic’s MCP. https://www.ox.security/blog/themother-of-all-ai-supply-chains-critical-systemic-vulnerability-at-the-coreof-the-mcp/. Published 15 April 2026. Accessed on 23rd July 2026.

[25] Lukas Pirch, Micha Horlboge, Patrick Großmann, Syeda Mahnur Asif, Klim Kireev, Thorsten Holz, and Konrad Rieck. 2026. Toward Securing AI Agents Like Operating Systems. https://arxiv.org/abs/2605.14932

[26] Python Packaging Authority. 2026. PyPI Download Statistics. https://pypistats. org. Accessed on 23rd July 2026.

[27] Mehrdad Rostamzadeh, Sidhant Narula, Nahom Birhan, Mohammad Ghasemigol, and Daniel Takabi. 2026. MCP-DPT: A Defense-Placement Taxonomy and Cov erage Analysis for Model Context Protocol Security. https://arxiv.org/abs/2604. 07551

[28] Mark Russinovich, Ahmed Salem, and Ronen Eldan. 2025. Great, Now Write an Article About That: The Crescendo Multi-Turn LLM Jailbreak Attack. In 34th USENIX Security Symposium (USENIX Security 25). https://www.usenix.org/ conference/usenixsecurity25/presentation/russinovich

[29] Sander Schulhof, Jeremy Pinto, Anaum Khan, Louis-François Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Kost, Christopher Carnahan, Jordan Boyd-Graber, et al. 2023. Ignore This Title and HackAPrompt: Exposing Systemic

Vulnerabilities of LLMs Through a Global Prompt Hacking Competition. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, Singapore, 4945– 4977. doi:10.18653/v1/2023.emnlp-main.302

[30] Che Wang, Jiaming Zhang, Ziqi Zhang, Zijie Wang, Yinghui Wang, Jianbo Gao, Tao Wei, Zhong Chen, and Wei Yang Bryan Lim. 2026. AdapTools: Adaptive Tool-based Indirect Prompt Injection Attacks on Agentic LLMs. https://arxiv. org/abs/2602.20720

[31] Zhiqiang Wang, Haohua Du, Guanquan Shi, Junyang Zhang, HaoRan Cheng, Yunhao Yao, Kaiwen Guo, and Xiang-Yang Li. 2026. MindGuard: Intrinsic Decision Inspection for Securing LLM Agents Against Metadata Poisoning. https://arxiv. org/abs/2508.20412

[32] Zhiqiang Wang, Yichao Gao, Yanting Wang, Suyuan Liu, Haifeng Sun, Haoran Cheng, Guanquan Shi, Haohua Du, and Xiang-Yang Li. 2026. MCPTox: A Benchmark for Tool Poisoning on Real-World MCP Servers. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 40. AAAI Press, 35811–35819. doi:10.1609/aaai.v40i42.40895

[33] Yuanbo Xie, Wenlei Zhu, Tianyun Liu, Yingjie Zhang, Suchen Liu, Yulin Li, Liya Su, and Tingwen Liu. 2026. What If Prompt Injection Never Left? Rethinking Agent Security through Cross-Session Stored Prompt Injection. https://arxiv. org/abs/2606.04425

[34] Shiqi Yang, Wenting Yang, Xiaochong Jiang, Cheng Ji, and Yichen Liu. 2026. SoK: A Taxonomy of Attack Vectors and Defense Strategies for Agentic Supply Chain Runtime. In ICLR 2026 Workshop on AIfor Mechanism Design and Strategic Decision Making (AIMS). https://openreview.net/forum?id=RpWthPyigJ

[35] Junjie Ye, Sixian Li, Guanyu Li, Caishuang Huang, Songyang Gao, Yilong Wu, Qi Zhang, Tao Gui, and Xuanjing Huang. 2024. ToolSword: Unveiling Safety Issues ofLarge Language Models in Tool Learning Across Three Stages. In Proceedings of the 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 2181–2211. doi:10.18653/v1/2024.acl-long.119

[36] Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. 2024. GPT-4 Is Too Smart To Be Safe: Stealthy Chat with LLMs via Cipher. In International Conference on Learning Representations (ICLR). Vienna, Austria. https://openreview.net/forum?id=MbfAK4s61A

[37] Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. 2024. InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. In Findings ofthe Association for Computational Linguistics: ACL 2024. Association for Computational Linguistics, Bangkok, Thailand, 10471–10506. doi:10.18653/v1/2024.findings-acl.624

[38] Dongsen Zhang, Zekun Li, Xu Luo, Xuannan Liu, Pei Pei Li, and Wenjun Xu. 2026. MCP Security Bench (MSB): Benchmarking Attacks Against Model Context Protocol in LLM Agents. In The Fourteenth International Conference on Learning Representations (ICLR). https://openreview.net/forum?id=irxxkFMrry

[39] Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. 2025. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents. In International Conference on Learning Representations (ICLR). OpenReview.net, Singapore. https://openreview.net/forum?id=V4y0CpX4hK

[40] Huijun Zhou, Xiaohan Zhang, Haozhe Zhang, Haoyang Zhang, Mi Zhang, and Min Yang. 2026. A First Measurement Study on Authentication Security in Real-World Remote MCP Servers. https://arxiv.org/abs/2605.22333

## Declaration on the Use of Generative AI

In accordance with the ACM Policy on Authorship and the AISec submission requirements, the use of generative AI in the preparation of this work is disclosed here. Claude Code (Anthropic) was used for implementation support in the mcpvenom harness and the accompanying analysis scripts, and for copy-editing of the manuscript; all such output was reviewed, executed, and validated by the authors, who retain full responsibility for the content. No claim, table, figure, or citation was accepted without verification against the released artifacts. Language models are otherwise the object of measurement rather than an instrument of it: the dual oracle is deterministic, resting on canary-server request logs and pattern matching over request bodies, and thus no attack-success label depends on model judgement.