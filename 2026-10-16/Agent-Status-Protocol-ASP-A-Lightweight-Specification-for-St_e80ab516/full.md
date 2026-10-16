# Agent Status Protocol (ASP): A Lightweight Specification for State Reporting in Multi-Agent Systems

Diego Gosmar Open Voice Interoperability Initiative Linux Foundation AI & Data Turin, Italy diego.gosmar@ieee.org

Emmett Coin   
ejTalk, Inc.   
Open Voice Interoperability Initiative   
Linux Foundation AI & Data   
Belleville, Michigan, USA   
emmett@ejtalk.com

Deborah A. Dahl Conversational Technologies Open Voice Interoperability Initiative Linux Foundation AI & Data Plymouth Meeting, PA, USA dahl@conversational-technologies.com

Csaba Bolyós Open Voice Interoperability Initiative Linux Foundation AI & Data Bratislava, Slovakia bladeszasza@gmail.com

Andreas Zettl Open Voice Interoperability Initiative Linux Foundation AI & Data Berlin, Germany info@azettl.net

October 16, 2026

## Abstract

The Agent Status Protocol (ASP) defines a lightweight way for heterogeneous AI agents to report their operational state to a status server, and for observers such as dashboards to receive that state as soon as it changes. Agents push small JSON events over HTTP POST; the server exposes the aggregated state to observers over Server-Sent Events (SSE). Because an agent that has crashed, hung or been compromised cannot be trusted to report its own failure, ASP does not rely on self-reports alone: the server infers liveness from missed heartbeats, can detect agents that are alive but no longer making progress, and accepts observations from supervisors, guardrails, operators and peers, some of which take precedence over what the agent says about itself while others are shown alongside it. ASP models state along independent dimensions (lifecycle, health, task, floor, governance), so that an agent can be, for example, online, degraded, working on a task and waiting for the floor at the same time. It reuses existing vocabularies: task states from the Agent2Agent (A2A) protocol, floor states from the Open Floor Protocol (OFP), and attribute conventions from the OpenTelemetry GenAI semantic conventions. This paper presents version 0.1 of ASP as a proposal, not as an approved standard. It is a working draft intended to open a discussion, and the authors invite comments, alternative proposals and independent implementations from the community.

Keywords: Multi-agent systems, Agent interoperability, Observability, A2A, Open Floor Protocol, OpenTelemetry.

Status of this document. This document is a proposal submitted for discussion. It has not been reviewed or endorsed by any standards body, and nothing in it should be read as an adopted standard. Its content, including the state names, the message formats and the normative requirements, may change substantially in response to feedback. Comments, objections and alternative proposals are welcome. They can be submitted as issues or pull requests in the project’s public repository, https://github.com/AgentStatusProtocol/spec [1], which hosts a summary of the proposal, a draft JSON Schema, example events and illustrative implementations, or sent to the authors.

## 1 Introduction

## 1.1 Motivation

AI agents are moving from single assistants to systems in which many agents plan, delegate, debate and call tools on each other’s behalf. Each agent is an autonomous and non-deterministic process that depends on external model providers, tools and budgets, any of which can fail, slow down or run out while the agent is still nominally running. One agent can be watched by hand, by reading its logs or its console. A crowd of agents cannot.

The dificulty grows faster than the number of agents. In a fully connected topology of n agents there are $n ( n - 1 ) / 2$ possible pairwise links: 15 among 6 agents, 190 among 20 and 1,225 among 50. Each link is a place where one agent may wait on another whose state it cannot see, and each added agent adds potential links to every existing one. Empirical studies of multi-agent LLM systems confirm that a large share of failures arise from inter-agent misalignment and from tasks that are not properly verified or terminated, and not only from the limits of individual models [5].

Without a shared view of agent state, these failures remain blind spots. A crashed or hung agent looks just like a slow one. Peers keep calling a stuck agent, unaware of its state. Rate limits and exhausted quotas surface as poor answers rather than as operational faults. A looping agent can hold the floor of a conversation or consume budget for minutes before anyone notices. An agent that has been compromised, for example by a prompt injection, keeps talking to the others as if nothing had happened.

Three related capabilities are needed to close these gaps. Tracing records what happened and when, so that incidents can be reconstructed. Monitoring shows live which agents are alive, busy, waiting or blocked, so that health and task problems are seen before users feel them. Governing lets an operator or a guardrail act on that picture, for example by quarantining one agent so that the others can route around it, and leaves a timestamped history for audit and post-mortem analysis.

Multi-agent systems increasingly mix agents built on diferent frameworks and protocols. Each exposes its state diferently, or not at all. Operators have no common way to answer basic questions: which agents are alive, what each is doing, who is waiting on whom, and which agent currently holds the floor in each conversation.

ASP provides that common layer. It is deliberately lightweight to adopt, so that any agent able to send an HTTP request can conform.

## 1.2 Application Scenarios

Live agent state is useful wherever several agents share work and a human or a component is accountable for the outcome. The following scenarios illustrate the range.

Enterprise agent fleets. Organisations deploy dozens of specialised agents for tasks such as order handling, document processing or planning, often built by diferent teams on diferent frameworks. A shared state view tells operators which agents are degraded, rate-limited or waiting for approval, and lets orchestrators route new work away from them.

Multi-party conversations and voice. In open-floor conversations, such as multi-agent debates, contact-centre interactions or voice conferences in which several agents take turns [15], who holds the floor and who is queued for it is operational information. A convener that stops yielding, or a participant that never gets the floor, becomes visible immediately.

Safety and guardrail pipelines. Pipelines in which specialised agents review, correct or filter the output of others, as in hallucination mitigation [13] and prompt injection mitigation [14], need a common way for a guardrail to flag an agent and for every other component to see that flag. The same holds for architectures in which a network of Sentinel Agents monitors interagent communication and a Coordinator Agent isolates or quarantines misbehaving agents on the basis of their alerts [12]; Section 11 shows how ASP supports this pattern.

Cost and capacity control. Token budgets, provider quotas and concurrency limits are shared resources. Reporting quota-exhausted, rate-limited or queue depth as state lets operators and autoscalers react before a budget overrun or a cascade of retries.

Regulated and high-assurance deployments. Regulations such as the EU Artificial Intelligence Act require record-keeping and efective human oversight for high-risk AI systems [9]. An append-only history of state transitions, together with explicit governance states such as under-review, gives operators evidence of what each agent was doing and of who intervened, without recording the content of the agents’ exchanges.

## 1.3 Why an Independent Standard

Operating deployed enterprise agent fleets at scale requires a reliable and uniform way of monitoring the state of every agent, whoever built it and whatever framework it runs on. Most agent frameworks and observability tools already expose some notion of agent status, but each defines its own. Without a standard, every pair of framework and monitoring tool needs its own integration, the same word means diferent things in diferent teams, and a dashboard can show only the agents built with the framework it was designed for. With F frameworks and T tools, bespoke integrations grow as $F \times T ;$ a shared protocol reduces them to F + T. ASP is designed around six properties that a proprietary or framework-bound solution does not provide together.

Any framework. Any agent that can send an HTTP POST can report its state. ASP requires no SDK, runtime or vendor, so it does not lock a deployment into one ecosystem.

Any topology. The same states describe a floor-managed debate under OFP, a task delegation under A2A and a voice conference, and they apply equally to agents that use neither protocol.

State only. Utterances, prompts and replies never enter ASP (Section 12). Monitoring therefore does not widen the exposure of what agents say, and the status server does not become a second copy of sensitive content.

One vocabulary. idle, degraded or awaiting-approval mean the same thing for every team and tool, so one dashboard can read every agent.

Server-side liveness. A missed heartbeat marks an agent as unreachable. The server decides, so an agent that has died cannot report itself as healthy (Section 7).

Governance in the protocol. Quarantine and policy states are part of the state model, and a state set by a guardrail or an operator takes precedence over the agent’s self-report (Section 12).

ASP defines the state. It does not define the agent, the framework or the vendor. This paper proposes ASP as a candidate for such a standard; whether it, or a successor shaped by community input, becomes one depends on review, implementation and consensus beyond its authors.

## 1.4 Why Not an Existing Health Protocol

Web servers and other services already report their state through established mechanisms, such as the liveness and readiness probes of container orchestrators [24] and the gRPC health checking protocol [16]. These mechanisms answer one question, whether a process is alive and ready to serve requests, typically with a small set of values such as serving or not serving, and they are usually polled by the infrastructure. Agents need more than that. An agent can be alive and ready while its task waits legitimately for human approval or for input from another agent; it can hold or wait for the floor of a conversation; it can fail because an upstream model provider has rate-limited it or its token budget is exhausted, while the process itself is healthy; and it can be quarantined by a guardrail while still running. None of these conditions can be expressed in a binary health check, and none is observable by probing the process. ASP does not replace existing health protocols, which remain appropriate for the processes that host agents. It reuses what fits from them and from related standards, namely heartbeats with timeouts, HTTP, Server-Sent Events and the OpenTelemetry conventions, and adds the agent-specific dimensions they lack (Section 2).

## 1.5 Why a State Layer on Top of OpenTelemetry

OpenTelemetry is the obvious candidate for carrying agent state, and ASP is designed to work with it rather than around it. OpenTelemetry defines how telemetry is produced, carried and exported as metrics, logs and traces, and its GenAI semantic conventions [22] name model calls, tool executions and agent invocations, on which the major instrumentation libraries are converging, where necessary through normalisation in the Collector. A deployment could therefore represent agent state with existing OpenTelemetry signals, for example a gauge or a log record carrying a state attribute. What OpenTelemetry does not provide, at the time of writing, are four properties that ASP specifies normatively. None of them concerns transport; all of them concern meaning and behaviour.

A shared vocabulary of agent states. The GenAI conventions describe what an agent did, not the condition it is in. Without agreed values, degraded or waiting in one framework means something diferent in another, which reproduces for agent state the fragmentation that semantic conventions resolved for model calls. ASP supplies the missing vocabulary, which can be carried as attribute values in any OpenTelemetry signal (Section 1.10).

A meaning for the absence of a signal. Telemetry pipelines are designed to tolerate loss through sampling, batching and periodic export. A missing span may mean that nothing happened, that the span was sampled out, or that the agent has died, and whether and when missing telemetry indicates failure is left to the alerting rules of each backend. ASP makes this part of the contract: each agent declares its heartbeat interval, and every conforming server derives unreachable and stalled from the same rules (Section 7).

Every transition, in order. A periodically exported metric reports one value per export interval, so a state held for a few seconds between two exports is never observed, and spans are exported only when an operation completes. ASP requires every transition to be reported as an event with a per-sender sequence number and kept in an append-only history, which both live control and audit require.

Authority over state. OpenTelemetry attributes each record to the resource that emitted it. It has no notion that a guardrail may assert the state of another agent, or that such an assertion takes precedence over the agent’s self-report. Investigations of agent incidents show why the distinction matters: a record of who claimed to act is not a record of who was observed acting. ASP records the source of every assertion and defines which source prevails (Section 7.2.3).

ASP is therefore best understood as a layer of state semantics, liveness rules and authority on top of existing telemetry, not as an alternative to it. Section 9.3 gives the current mapping from OpenTelemetry to ASP. A binding in the opposite direction, in which ASP events are emitted as OpenTelemetry log records linked to the active trace, and an implementation of the status server functions as a component of the OpenTelemetry Collector, are natural extensions of this draft.

## 1.6 Why Push State Instead of Asking the Agent

A simpler design would let each observer ask each agent for its state. ASP does not rule this out, and the pull mode of Section 4 uses it for agents that cannot be modified, but asking alone does not work well for large, heterogeneous deployments. First, an agent that has crashed or lost connectivity cannot answer, so failure detection has to rely on the absence of an expected signal rather than on a reply; heartbeats provide that signal. Second, asking requires every observer to know and reach every agent: with N agents and M observers polling, trafic grows as N × M, and agents behind NAT or firewalls would need inbound ports. Third, some states are not the agent’s to report: unreachable is inferred by the server, and governance states may be set by a guardrail or an operator against the agent’s own report. Fourth, a server that receives every transition can keep a complete history for audit, whereas periodic polling misses transitions that happen between two polls. Finally, answers are only comparable if agents use the same names, so a shared state vocabulary is needed however the state is obtained, which is why it can be adopted on its own (Section 1.10).

## 1.7 Scope

ASP covers agent registration and deregistration with a status server, periodic liveness signalling (heartbeat), reporting of state transitions along defined dimensions, and live distribution of aggregated state to observers.

## 1.8 Non-goals

ASP does not define how agents communicate with each other, which is the role of A2A [2], OFP [21] and similar protocols. It does not define task payloads, prompts or model outputs, nor full distributed tracing, which is the role of OpenTelemetry [22]. Agent discovery and capability negotiation are also out of scope.

## 1.9 Relationship to Existing Standards

ASP is designed as a profile, not a replacement. Task states align with A2A. Floor and role states align with OFP. Metric and identity attributes align with the OpenTelemetry GenAI semantic conventions, so that an ASP server can also ingest OTLP data. Section 9 gives the mappings. ASP is independent of each of these protocols: it observes agents that speak them but does not extend or depend on any of them, and an agent that speaks none of them can still conform.

## 1.10 Specification Layers

This specification is organised in three layers that can be adopted separately. The state vocabulary (Section 5) defines the dimensions and the standard state names; it does not depend on any transport and MAY be used on its own, for example as attribute values in logs, in OpenTelemetry spans or in the metadata of another agent protocol. The event layer (Section 6) defines the JSON messages that carry state transitions. The behaviour layer (Sections 7, 8 and 12) defines what a status server does with those messages: liveness inference, transport binding and security rules. An implementation MAY conform to the state vocabulary alone, to the state vocabulary and the event layer, or to all three layers. For this reason the state model is presented before the message types.

## 2 Related Work

ASP draws on four bodies of work: agent communication protocols, lifecycle management in classical multi-agent platforms, observability of LLM agents, and failure detection in distributed systems. It also builds on earlier work on multi-party agent conversations and their safety applications. This section reviews each and then summarises what ASP adds (Table 1).

## 2.1 Agent Communication Protocols

Agent communication has a long history, from KQML and FIPA ACL to the protocols that have emerged for LLM-based agents. The Model Context Protocol (MCP) [20] connects an agent to tools and data sources; the Agent2Agent protocol (A2A) [2] lets agents delegate tasks to each other and describes their capabilities in Agent Cards; the Agent Communication Protocol (ACP) and the Agent Network Protocol (ANP) address REST-based messaging and open-network discovery respectively; a recent survey compares these four protocols [8]. The Open Floor Protocol (OFP) [21] manages multi-party conversations in which a convener grants and revokes the floor.

These protocols define how agents talk, and some expose part of an agent’s state as a side efect. A2A defines task states, but they are visible only to the client of each task and say nothing about whether the agent itself is healthy or alive. An Agent Card is a static description rather than a live status. OFP floor events describe one conversation to its participants. MCP describes tools, not agents. None of them defines liveness, health or governance, and none gives a third-party observer an aggregated, live view of all agents. ASP complements these protocols rather than competing with them, and reuses their vocabularies where they exist (Section 9).

## 2.2 Lifecycle Management in Classical Multi-Agent Platforms

The FIPA Agent Management Specification [11] defined an agent platform in which an Agent Management System keeps a directory of agents and a lifecycle with states such as initiated, active, suspended, waiting and transit. Platforms such as JADE [3] implemented this model. ASP inherits the idea of an explicit lifecycle. It difers in three respects: the FIPA lifecycle lives inside a single agent platform rather than across heterogeneous frameworks; it has a single dimension, so an agent cannot be simultaneously active, degraded and waiting for the floor; and it predates the failure modes typical of LLM agents, such as rate limits, exhausted quotas, unavailable tool servers and suspected prompt injection.

## 2.3 Observability of LLM Agents

The OpenTelemetry GenAI semantic conventions [22] standardise span and metric attributes for model calls and agent invocations, and the AgentOps taxonomy [7] catalogues the artefacts that should be traced across the lifecycle of LLM agents. Commercial and open-source tracing platforms build on these ideas. This work is essential for debugging and evaluation, but it is organised around traces of past operations: spans are exported once an operation completes, they often carry prompts and model outputs, and there is no model of the current state of an agent or of its liveness between operations. ASP is complementary: it adopts the OpenTelemetry identifiers and can ingest OTLP data (Section 9.3), and it adds the live, content-free state that traces do not express.

## 2.4 Failure Detection and Health Checking

Distributed systems have long detected crashed processes with heartbeats and timeouts, formalised as unreliable failure detectors [6] and refined by accrual detectors that adapt to network conditions [17]. Container orchestrators check the liveness and readiness of processes through periodic probes [24], and gRPC defines a standard health checking service [16]. ASP applies the same heartbeat-and-timeout principle (Section 7), with two diferences: liveness is inferred from heartbeats the agent pushes, rather than from probes the orchestrator sends, and a process that is alive is only the first dimension of an agent’s state, alongside health, tasks, floor and governance.

## 2.5 Failures, Guardrails and Multi-Party Conversations

Empirical work on multi-agent LLM systems has produced taxonomies of how they fail, including inter-agent misalignment and failures of termination and verification [5]. Guardrail toolkits such as NeMo Guardrails [23] constrain what individual agents may say or do. ASP builds directly on earlier work on multi-party agent conversations: the multi-agent interoperability extension for managing multiparty conversations [15] introduced the convener and floor concepts that ASP adopts in its floor dimension, and the same open-floor approach has been applied to hallucination mitigation in multi-agent pipelines [13] and to prompt injection mitigation, where specialised agents detect and neutralise injected instructions [14]. Sentinel Agents [12] generalise this idea into a distributed security layer: sentinels combine LLM-based semantic analysis, behavioural analytics, retrieval-augmented verification and cross-agent anomaly detection to oversee interagent communication, while a Coordinator Agent ingests their alerts, adapts policies and isolates or quarantines misbehaving agents. That work defines who detects threats and who decides; it leaves open the format in which alerts and decisions are exchanged and made visible to the rest of the system, which is the gap ASP fills. More generally, this line of work shows that the behaviour of a multi-agent pipeline depends on the state of each agent and on who holds the floor at any moment, but none of it defines a common runtime signal through which that state becomes visible to the rest of the system.

## 2.6 What ASP Adds

Table 1 compares ASP with the closest existing approaches. The comparison concerns the exposure of agent state to observers, not the broader purpose of each approach, most of which ASP is designed to complement. ASP contributes:

• a multi-dimensional state model, in which lifecycle, health, task, floor and governance vary independently, so that an agent can be online, degraded, working and waiting for the floor at the same time;

• liveness inferred by the server from heartbeats declared by each agent, so that an agent that has died is marked unreachable rather than trusted to report itself;

• governance states in the protocol, with precedence for guardrails and operators over selfreports and a record of who set each state;

• a profile that reuses A2A task states, OFP floor states and OpenTelemetry identifiers, with explicit mappings that let adapters cover agents that cannot be modified;

• a content-free design, in which events carry state but never prompts, outputs or user data;

• a layered specification whose state vocabulary can be adopted without the event and behaviour layers (Section 1.10).

Table 1: Exposure of agent state to observers in existing approaches and in ASP. ✓: supported; ◦: partial or limited scope; –: not addressed.
<table><tr><td>Approach</td><td>Lifecycle/health</td><td>Liveness</td><td>Task</td><td>Floor</td><td>Governance</td><td>Aggregated live view</td></tr><tr><td>FIPA Agent</td><td>O</td><td></td><td></td><td></td><td></td><td>O</td></tr><tr><td>Management [11] A2A [2]</td><td></td><td></td><td>√</td><td></td><td></td><td>O</td></tr><tr><td>OFP [21]</td><td></td><td></td><td></td><td>√</td><td></td><td>O</td></tr><tr><td>MCP [20]</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>OpenTelemetry</td><td></td><td></td><td>0</td><td></td><td></td><td></td></tr><tr><td>GenAI [22] Health</td><td>O</td><td>V</td><td></td><td></td><td></td><td>0</td></tr><tr><td>probes [24, 16] ASP</td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>V</td></tr></table>

## 3 Terminology

The key words MUST, MUST NOT, SHOULD, SHOULD NOT and MAY are to be interpreted as described in RFC 2119 [4] and RFC 8174 [19]. In this proposal they state the requirements that a conforming implementation would have to meet if ASP were adopted; they do not imply that any standards body has endorsed these requirements. Table 2 defines the terms used in this specification.

Table 2: Terminology.
<table><tr><td>Term</td><td>Definition</td></tr><tr><td>Agent</td><td>An autonomous software component that uses AI models, memory and tools to pursue goals, make decisions or act on behalf of users or other agents, and whose state is reported using ASP, either natively or through an adapter.</td></tr><tr><td>Status server</td><td>The component that receives ASP events, maintains current state and distributes it. In the reference binding it is an HTTP service run by the operator of a deployment of agents, but its functions may also be embedded in an existing component (Section 4.3); its scope is one deployment, not the Internet (Section 4.2).</td></tr><tr><td>Observer</td><td>Any client that consumes aggregated state. The typical observer is a dashboard used by a human operator; other observers are software, such as alerting, autoscaling or orchestration components.</td></tr><tr><td>Operator</td><td>The human, or the software acting on a human&#x27;s behalf (for example an orchestrator, supervisor or autoscaler), responsible for running a deployment of agents. Operator actions such as pausing an agent or setting a governance state reach the status server through a credential authorised for that purpose (Section 12).</td></tr><tr><td>Event</td><td>A single JSON message about one agent, sent to the status server by that agent or by a component authorised to report on it, or generated by the server itself.</td></tr><tr><td>Dimension</td><td>An independent axis of state (lifecycle, health, task, floor, governance)</td></tr><tr><td>State</td><td>A value within one dimension at a given time. A component that emits ASP events on behalf of an agent that does not implement</td></tr></table>

## 4 Architecture

ASP has three roles: agents push events, the status server aggregates them, and observers subscribe to the result (Fig. 1).

Agents initiate every connection, so agents behind NAT or firewalls work without inbound ports; the only exception is the OPTIONAL active probe of Section 7.2.2. Observers receive updates as soon as the server processes an event; no polling is required.

![](images/98569540ef89fa2e71452bb8931f6df4c3646f80cc17146c49bec67cea10203f.jpg)  
Figure 1: ASP architecture. Agents and adapters push events; observers receive updates over SSE. Observers include dashboards for human operators and software components; some of them, such as Coordinator Agents or Sentinel Agents, also report on other agents (dashed arrow).

The status server does not connect to an agent, except for the OPTIONAL active probe. Each agent, or its adapter, is configured at deployment time with the base URL of the status server and a credential, for example through environment variables or a configuration file, and sends register before its first event to that server (Section 6.2). How that configuration is distributed is a deployment concern, and discovery of status servers is out of scope. Agents that cannot be configured in this way are covered by adapters, or by the pull mode described below.

A status server MAY also operate in pull mode for agents it cannot modify, for example by periodically reading an A2A Agent Card or a health endpoint and synthesising ASP events. Pull mode is outside the core specification. Appendix A shows an illustrative observer dashboard.

## 4.1 Observers

An observer is any client that reads the aggregated state from the status server through the distribution endpoints: it fetches a snapshot of current state, subscribes to the SSE stream, receives each change as soon as the server processes it, and can reconnect without losing updates (Section 8). An observer does not need to contact any agent to learn its state. ASP does not prevent it from doing so: a component may also query agents through other protocols, for example by requesting the status of an A2A task or by asking an agent to justify a claim during verification, and MAY report the outcome to the status server as an observation with a source other than self (Section 7.2.3). Observers need authorisation, may filter the stream by agent or conversation, and see only state, never the content of what agents say.

Observers fall into three broad groups, shown in Fig. 1. The typical observer is a dashboard through which human operators supervise a deployment, such as the one in Appendix A. Software observers act on the same stream without a human in the loop: alerting systems notify oncall staf when an agent becomes unreachable, autoscalers read queue depth and active tasks, orchestrators stop routing work to agents that are degraded or quarantined, and exporters feed audit archives or security monitoring. Finally, some observers are themselves agents. A Coordinator Agent reads the stream to receive alerts and to correlate them with the health and task states of the agents concerned before deciding on governance actions. Sentinel Agents may read it to use state patterns as an additional signal and to focus their scrutiny (Section 11); a sentinel that only reports alerts, without consuming state, is not an observer.

Being an observer is a function, not a kind of component. A single component may register and report its own state as an agent, read the stream as an observer, and report on other agents with a source other than self; a Coordinator Agent typically does all three. The same component may also interact with agents through channels outside ASP: a Sentinel Agent inspects inter-agent messages through the conversation protocol and may query a suspect agent directly, and a Coordinator Agent may query agents and enforce its decisions through the convener or the orchestrator, for example by revoking an agent’s floor. ASP carries the resulting state, not those interactions. The term is unrelated to the OFP conversation role observer, which denotes an agent that listens to a conversation without speaking rights.

## 4.2 Status Server Placement

In the reference binding, a status server is an ordinary HTTP service. It is not an Internet-wide registry. Each deployment runs its own server, and an agent reports only to the server it is configured for. Several servers MAY coexist, for example one per environment or per business unit; aggregation across servers is out of scope for this draft.

The status server is independent of any conversation protocol. In an OFP deployment it is not part of the floor: the convener is an agent like any other and reports its own state to the server, and the floor manager, where present, is a separate component that MAY do the same. A floor manager MAY run an ASP adapter that reports floor transitions on behalf of the agents in its conversations, and MAY be co-located with the status server, but neither is required.

## 4.3 Deployment Topologies

ASP does not require a dedicated server, but it does require that the functions of a status server be performed by some component: inferring liveness, including the states an agent cannot report about itself; enforcing the precedence of governance states over self-reports; checking credentials and ordering events; and distributing state to observers while keeping a history for audit. The HTTP and SSE binding of Section 8 is the reference realisation of these functions, but other topologies are possible. The functions may be embedded in a component that already exists, such as an orchestrator, an agent runtime or an OFP convener, without deploying a separate service. They may rely on a message broker, for example MQTT, whose retained messages and Last Will and Testament can provide a snapshot of state and part of liveness detection. Or events may travel in-band inside the conversation or task protocol, with each participant building its own view. The more these functions are distributed, the weaker liveness and governance become, because no single component has an authoritative view. A deployment MAY choose any topology, but it SHOULD identify which component performs each function; where no component checks credentials at ingest, events SHOULD be signed (Section 12).

## 4.4 Stateful and Stateless Agents

ASP works best with agents that keep their own operational state: a stable identity, a process that keeps running between requests, and knowledge of the tasks and conversations they are engaged in. Stateless agents that exist only for the duration of a single invocation, for example as serverless functions, can also be represented, but not natively: between invocations no process sends heartbeats, several short-lived instances may serve the same logical agent, and states that span several invocations are not retained. For such agents it is RECOMMENDED that a component that persists across invocations, such as the orchestrator or the hosting platform, report on their behalf as an adapter, using the agentId of the logical agent. The distinction concerns operational state, not conversational memory: an agent that remembers nothing of previous conversations but runs as a long-lived service can report its state natively.

## 5 State Model

This section is the state vocabulary layer (Section 1.10) and can be used independently of the event and behaviour layers. An agent’s state is a tuple of values, one per dimension. Dimensions are independent: a change in one does not imply a change in another, except where Section 7 states otherwise. Table 3 summarises the dimensions.

Table 3: State dimensions.
<table><tr><td>Dimension</td><td>Cardinality</td><td>Reported via</td></tr><tr><td>lifecycle</td><td>one per agent</td><td>state_change, heartbeat</td></tr><tr><td>health</td><td>one per agent</td><td>state_change, heartbeat</td></tr><tr><td>task</td><td>one per task</td><td>state_change</td></tr><tr><td>floor</td><td>one per conversation</td><td>floor_change</td></tr><tr><td>governance</td><td>one per agent</td><td>state_change</td></tr></table>

## 5.1 Lifecycle

Table 4 lists the lifecycle states and Fig. 2 the permitted transitions. From any state, the server MAY assign unreachable, and a supervisor MAY report crashed. Receiving any event from an unreachable agent returns it to the state that event reports.

The states idle and busy describe whether work is in progress, not whether the agent is at full capacity. An agent working on one task that can still accept more is busy. Spare capacity is reported through the OPTIONAL heartbeat metrics activeTasks and maxConcurrentTasks. An agent that can accept no further work reports lifecycle at-capacity: this is a normal operating condition, not a fault, and is distinct from health overloaded, which signals a request queue above threshold. The states idle, busy and at-capacity are reported by the agent, since only the agent knows its own capacity.

Table 4: Lifecycle states.
<table><tr><td>State</td><td>Meaning</td></tr><tr><td>registering</td><td>Announcing itself to the server.</td></tr><tr><td>initializing</td><td>Loading models, tools and configuration.</td></tr><tr><td>online</td><td>Active and ready.</td></tr><tr><td>idle</td><td>Online with no work in progress.</td></tr><tr><td>busy</td><td>Online with work in progress, and able to accept more.</td></tr><tr><td>at-capacity</td><td>Online with work in progress and no capacity for further work.</td></tr><tr><td>paused</td><td>Suspended by an operator; accepts no new work.</td></tr><tr><td>draining</td><td>Finishing current work, accepting no new work; typically precedes shutdown.</td></tr><tr><td>stopping</td><td>Shutting down.</td></tr><tr><td>offline</td><td>Cleanly stopped, after deregister.</td></tr><tr><td>unreachable</td><td>Heartbeat missed. Server-assigned only (Section 7).</td></tr><tr><td>crashed</td><td>Terminated abnormally, as detected by a supervisor or adapter.</td></tr></table>

## 5.2 Health

Health transitions are unconstrained: any state MAY move to any other. Table 5 lists the health states.

![](images/535375771b771c71a342ced3b190d8591068c87833baa86dcbf650d66b55bc8d.jpg)  
Figure 2: Lifecycle transitions. The server-assigned unreachable and the supervisor-reported crashed are reachable from any state and are omitted for clarity. An agent in at-capacity returns to busy when capacity frees up.

Table 5: Health states.
<table><tr><td>State</td><td>Meaning</td></tr><tr><td>healthy</td><td>Operating normally.</td></tr><tr><td>degraded</td><td>Working with reduced performance, e.g. high latency or fallback to a smaller model.</td></tr><tr><td>rate-limited</td><td>Temporarily blocked by an upstream provider&#x27;s rate limit.</td></tr><tr><td>quota-exhausted</td><td>Token or cost budget used up.</td></tr><tr><td>dependency-down</td><td>A required tool, API or MCP server is not responding.</td></tr><tr><td>auth-error</td><td>Credentials expired or invalid.</td></tr><tr><td>overloaded</td><td>Request queue above threshold.</td></tr><tr><td>unhealthy</td><td>Unable to perform its function.</td></tr><tr><td>stalled</td><td>Alive but making no progress on its work. Server-assigned (Section 7.2.1).</td></tr></table>

## 5.3 Task

Task state is tracked per taskId, which every state\_change in the task dimension carries (Section 6.4); an agent MAY have several tasks open at the same time, each with its own state. Table 6 lists the task states. States of origin A2A are taken unchanged from the A2A specification [2]; the others are ASP extensions. A task in a terminal state MUST NOT transition further. The server SHOULD remove terminal tasks from the live view after a configurable retention period.

Table 6: Task states.
<table><tr><td>State</td><td>Origin</td><td>Terminal</td><td>Meaning</td></tr><tr><td>submitted</td><td>A2A</td><td>no</td><td>Received, not yet started.</td></tr><tr><td>queued</td><td>ASP</td><td>no</td><td>Waiting behind other tasks.</td></tr><tr><td>working</td><td>A2A</td><td>no</td><td>In progress.</td></tr><tr><td>input-required</td><td>A2A</td><td>no</td><td>Needs input from a user or another agent.</td></tr><tr><td>auth-required</td><td>A2A</td><td>no</td><td>Needs additional authorisation.</td></tr><tr><td>awaiting-approval</td><td>ASP</td><td>no</td><td>Waiting for human-in-the-loop approval.</td></tr><tr><td>delegated</td><td>ASP</td><td>no</td><td>Part of the work passed to another agent; awaiting its result.</td></tr><tr><td>retrying</td><td>ASP</td><td>no</td><td>Repeating a failed step.</td></tr><tr><td>completed</td><td>A2A</td><td>yes</td><td>Finished successfully.</td></tr><tr><td>failed</td><td>A2A</td><td>yes</td><td>Finished with an error.</td></tr><tr><td>canceled</td><td>A2A</td><td>yes</td><td>Cancelled by the requester.</td></tr><tr><td>rejected</td><td>A2A</td><td>yes</td><td>Declined by the agent (out of scope, policy).</td></tr><tr><td>timed-out</td><td>ASP</td><td>yes</td><td>Exceeded its maximum duration.</td></tr><tr><td>unknown</td><td>A2A</td><td>no</td><td>State cannot be determined.</td></tr></table>

## 5.4 Floor

Floor states are aligned with OFP floor management [21, 15] and are tracked per conversationId. Table 7 lists them and Fig. 3 shows the permitted transitions.

Floor states describe an agent’s participation in one conversation, not the agent itself. An agent in left for a conversation stops sending floor\_change events for that conversation, but it remains active: it continues to send heartbeats and to report its lifecycle, health, task and governance states, and it MAY take part in other conversations. An agent that stopped reporting altogether after leaving a conversation would be marked unreachable (Section 7); an agent that is shutting down sends deregister instead.

ASP does not define conversation roles. Roles such as convener, participant or observer belong to the conversation protocol, and their meaning is defined there. A floor\_change event MAY carry the agent’s role in the conversation in the OPTIONAL role field, using the value defined by that protocol, for example convener or participant for OFP; the status server stores and forwards the value without interpreting it.

Table 7: Floor states.
<table><tr><td>State</td><td>Meaning</td></tr><tr><td>not-joined</td><td>Not part of the conversation.</td></tr><tr><td>invited</td><td>Invited, not yet joined.</td></tr><tr><td>joined</td><td>Participating and listening</td></tr><tr><td>requesting-floor</td><td>Has asked to speak.</td></tr><tr><td>has-floor</td><td>Currently speaking.</td></tr><tr><td>yielded</td><td>Has given up the floor.</td></tr><tr><td>floor-revoked</td><td>Floor removed by the convener.</td></tr><tr><td>left</td><td>Has left the conversation, for example after sending bye, after being removed from it, or after declining the invitation.</td></tr></table>

## 5.5 Governance

Governance is an OPTIONAL dimension for production deployments (Table 8). Governance states MAY be set by the agent itself, by a guardrail component, or by an operator through the server. Observers SHOULD display any state other than normal prominently. The quarantined

![](images/7186628708aad76d067ae0914951cfc918842b015aea420a191235ec068eee0e.jpg)  
Figure 3: Floor transitions within one conversation. An agent MAY enter the conversation directly in joined without a prior invitation. An invited agent that declines or is uninvited moves directly to left.

state is intended, for example, for agents flagged by a prompt injection detection layer such as the one described in [14], or quarantined by a Coordinator Agent on the basis of Sentinel Agent alerts [12] (Section 11).

Table 8: Governance states.
<table><tr><td>State</td><td>Meaning</td></tr><tr><td>normal</td><td>No governance restriction.</td></tr><tr><td>sandboxed</td><td>Operating with reduced permissions.</td></tr><tr><td>quarantined</td><td>Isolated after anomalous behaviour, e.g. suspected prompt injection.</td></tr><tr><td>policy-violation</td><td>Attempted an action forbidden by policy.</td></tr><tr><td>under-review</td><td>Suspended pending review by an operator, human or software (Table 2).</td></tr></table>

## 6 Message Types

ASP defines five event types. Every event is a JSON object sent as the body of an HTTP POST with Content-Type: application/json.

## 6.1 Common Fields

Table 9 lists the fields common to all events. The examples in this section omit aspVersion, eventId and seq for brevity.

Table 9: Common fields of all ASP events.
<table><tr><td>Field</td><td>Type</td><td>Req.</td><td>Description</td></tr><tr><td>aspVersion string</td><td></td><td>yes</td><td>Protocol version, e.g. &quot;0.1&quot;.</td></tr><tr><td>type</td><td>string</td><td>yes</td><td>One of register, heartbeat, state_change, floor_change, deregister.</td></tr><tr><td>agentId</td><td>string (URI) yes</td><td></td><td>Stable, globally unique agent identifier. A URN is RECOMMENDED.</td></tr><tr><td>eventId</td><td>string</td><td>yes</td><td>Unique per event, used for deduplication. UUID RECOMMENDED.</td></tr><tr><td>seq</td><td>integer</td><td>yes</td><td>Monotonically increasing per agent and per sender: an agent numbers its own events, and each component that reports on it numbers its own events about that agent. Used for ordering.</td></tr><tr><td>ts</td><td>string</td><td>yes</td><td>Event time in RFC 3339 [18] format, UTC. Who reports the event: self (default), supervisor, guardrail,</td></tr><tr><td>source</td><td>string</td><td>no</td><td>operator or peer. Events generated by the server carry server. See Section 7.2.3.</td></tr></table>

## 6.2 register

Sent once, before the agent’s first event to a given status server. An agent registers when it begins to operate in the deployment that the server monitors: typically at start-up, or when it is first engaged by that deployment, for example when it is invited to an OFP conversation or receives its first A2A task. Activity of the agent before that moment, unrelated to the deployment, is outside the server’s scope. Registration is independent of any conversation: joining a conversation is reported separately as a floor\_change to joined. An agent that stops operating in the deployment sends deregister; an agent that reports to several status servers registers with each. Registration MUST precede any other event from that agent. The field heartbeatIntervalSec is REQUIRED and MUST be between 1 and 300. The fields healthUrl, an endpoint the server MAY probe (Section 7.2.2), and stallAfterSec, the time without progress after which the agent is considered stalled (Section 7.2.1), are OPTIONAL.

```jsonl
1 {
" type ": " register ",
" agentId ": "urn : agent : example : research - assistant ",
" name ": " Research Assistant ",
" version ": " 1.2.0 ",
" protocols ": [" a2a ", "ofp"] ,
" capabilities ": ["web - research ", " summarization "],
" heartbeatIntervalSec ": 10,
9 "ts": " 2026 -09 -22 T17 :00:00 Z"
10 }
```

## 6.3 heartbeat

Sent periodically to prove liveness. It SHOULD be as small as possible. It SHOULD be emitted from the agent’s main work loop rather than from an independent thread, and SHOULD carry progressSeq, a counter that the agent increments whenever it makes progress on its work (Section 7.2.1). It MAY carry activity, a short code that tells observers what a long wait is spent on without revealing its content. Recommended values are model-call, tool-call, media-generation, waiting-floor, waiting-peer and waiting-input; custom values follow Section 13. The field is a hint for observers and does not change the agent’s state in any dimension. The metrics object is OPTIONAL; Section 9.3 lists recommended metric names.

```json
{
" type ": " heartbeat ",
" agentId ": "urn : agent : example : research - assistant ",
" lifecycle ": " busy ",
" health ": " healthy ",
" progressSeq ": 1742 ,
" activity ": "model - call ",
" metrics ": { " latencyMs ": 820 , " queueDepth ": 2 , " activeTasks ": 1 , II
maxConcurrentTasks ": 4 },
9 "ts": " 2026 -09 -22 T17 :00:10 Z"
10 }
```

## 6.4 state\_change

Sent on every transition in any dimension other than floor. The fields dimension, from and to are REQUIRED. The field taskId is REQUIRED when dimension is task. The field maxDurationSec is OPTIONAL and applies to transitions of a task to working (Section 7.2.1). The field reason is OPTIONAL. Its value SHOULD start with a short machine-readable code in lowercase kebab-case, which MAY be followed by a colon and a brief neutral description, e.g. clarification-needed: time period. The same format applies to every reason field in ASP. A reason MUST NOT contain sensitive data (see Section 12).

```jsonl
1 {
2 " type ": " state_change ",
3 " agentId ": "urn : agent : example : research - assistant "
4 " dimension ": " task ",
5 " from ": " working ",
6 "to": "input - required ",
7 " taskId ": "t -8812 ",
8 " contextId ": "ctx -301 ",
9 " reason ": " clarification - needed : time period ",
10 "ts": " 2026 -09 -22 T17 :02:11 Z"
11 }
```

## 6.5 floor\_change

Sent on transitions in a multi-party conversation. The fields conversationId, from and to are REQUIRED; the field role is OPTIONAL (Section 5.4). An agent MAY participate in several conversations at once; state is tracked per conversationId.

```jsonl
1 {
2 " type ": " floor_change ",
3 " agentId ": "urn : agent : example : research - assistant "
" conversationId ": "c -44 ",
" from ": " requesting - floor ",
6 "to": "has - floor ",
7 " role ": " participant ",
8 "ts": " 2026 -09 -22 T17 :02:15 Z"
9 }
```

## 6.6 deregister

Sent on clean shutdown. On receipt, the server MUST set lifecycle to offline and stop liveness monitoring for that agent.

```jsonl
1 {
2 " type ": " deregister ",
" agentId ": "urn : agent : example : research - assistant "
" reason ": " scheduled - shutdown ",
"ts": " 2026 -09 -22 T18 :00:00 Z"
}
```

## 7 Liveness

A failed agent cannot be relied on to announce its own failure. ASP therefore separates what the agent reports about itself from what the server and other components can establish independently. Section 7.1 covers agents that fall silent; Section 7.2 covers agents that keep reporting but are hung or compromised.

## 7.1 Heartbeat Timeout

Each agent declares its own heartbeat interval at registration. The server infers liveness from that interval; no server-side polling is required. The following rules apply.

• An agent MUST send a heartbeat at least once per declared heartbeat interval. Any other event sent by the agent itself also counts as a heartbeat; events sent about the agent by other components do not.

• If no event arrives within 2.5 declared heartbeat intervals, the server MUST set lifecycle to unreachable and notify observers.

• The 2.5 multiplier tolerates one lost heartbeat plus network jitter. A server MAY make it configurable but MUST NOT set it below 2.

• When an unreachable agent sends any event, the server MUST restore lifecycle to the value that event reports, or to online if it reports none.

• If an unreachable agent stays silent for a server-defined expiry period (RECOMMENDED: 24 hours), the server MAY treat it as offline and discard its live state.

• unreachable and stalled are the only states an agent MUST NOT report about itself.

On transition to unreachable, the server SHOULD keep the last known task and floor states visible but mark them as stale, since the server cannot know whether they are still true.

## 7.2 Beyond Heartbeats

The heartbeat timeout detects an agent that has stopped, lost connectivity or crashed, because it relies on the absence of a signal rather than on a message from the agent. It does not detect two other cases: an agent whose heartbeat continues while its work has stopped, and an agent that has been compromised and reports false state. The mechanisms below address them. All are OPTIONAL in this draft; a server that implements one MUST follow the rules given for it.

## 7.2.1 Progress

Liveness shows that the agent’s process is running; progress shows that its work is advancing. An agent that emits heartbeats from a separate thread can keep reporting healthy while its main loop is blocked. To make this visible:

• An agent that declares stallAfterSec MUST include progressSeq in every heartbeat and MUST increase it whenever it completes a step of work, such as a model call, a tool call or a message sent.

• If the agent has at least one task in working or retrying, the states in which progress is expected, and progressSeq has not increased for stallAfterSec, the server MUST set health to stalled with source: server and notify observers. The first heartbeat that reports a higher progressSeq clears the state and restores the health reported by the agent. Tasks that are waiting by design, such as input-required, awaiting-approval, delegated or queued, do not trigger stalled.

• An agent MAY declare maxDurationSec on a state\_change that moves a task to working. If the task is still non-terminal when that time has elapsed, the server MAY set it to timed-out with source: server.

• Expected durations difer widely between agents and tasks: generating images or 3D content can take far longer than producing text. stallAfterSec and maxDurationSec SHOULD therefore be set per agent and, where useful, per task type, rather than as a single deployment-wide value. Together with progressSeq and activity, they let an observer tell a long but advancing operation from an agent waiting on others and from one that is stuck.

## 7.2.2 Active Probing

Active probing is the only case in which the status server initiates a connection to an agent, and it is OPTIONAL for both. An agent that supports it declares healthUrl in register: an HTTPS endpoint that the status server can reach and that answers a GET request with any 2xx status when the agent is running. Probing therefore works only for agents that accept inbound connections from the server; agents behind NAT or firewalls without inbound ports simply do not declare healthUrl. When an agent that declared healthUrl misses its heartbeat deadline, the server MAY send one GET request to that endpoint before or after setting unreachable. The probe does not restore the agent’s state: it only refines the reason given to observers, no-heartbeat-probe-failed when the agent does not answer and no-heartbeat-probe-ok when it answers but has stopped reporting, which points to a fault in the reporting path rather than in the agent. Probes MUST NOT be sent more often than once per declared heartbeat interval.

## 7.2.3 Independent Observation

The state of an agent SHOULD NOT depend only on what the agent says about itself. Components that interact with the agent, such as a supervisor that runs its process, an OFP floor manager or convener that grants it the floor, a guardrail that screens its output or a peer that calls it, often see its failures first. Such components MAY send events about another agent, with the source field set to the closest of the values in Table 9; a convener or floor manager, for example, reports as peer or supervisor depending on its authority in the deployment, using a credential authorised for that purpose (Section 12). The following precedence rules apply.

• Lifecycle. States assigned by the server (unreachable) and by a supervisor (crashed) override the agent’s self-report until the agent sends a new event, as in Section 7.1.

• Health. A peer observation, for example repeated timeouts when calling the agent, does not replace the self-reported health. The server MUST record it and expose it alongside the self-report in the snapshot, and observers SHOULD highlight any disagreement between the two. A caller that enforces its own per-call timeout SHOULD report each expiry as a peer observation with reason: call-timeout, so that the timeout becomes visible to every observer rather than only to the caller.

• Governance. States set by a guardrail or an operator take precedence over the agent’s self-report and can be cleared only by a component with the same or higher authority. Authority is ordered as operator above guardrail above self: an operator can clear a state set by a guardrail, but not the reverse.

## 7.2.4 Compromised Agents

A compromised agent can emit well-formed heartbeats and false states indefinitely, so no liveness mechanism alone can reveal it. ASP limits the damage in three ways: its governance cannot be changed by its own reports; its self-reports are shown next to independent observations, so that a contradiction becomes visible; and, when an agent is set to quarantined, the operator MAY revoke its ingest credential. After revocation the server rejects the agent’s events with 401, the agent becomes unreachable in the normal way, and its last known state remains visible as stale for investigation.

## 8 Transport Binding

## 8.1 Ingest (Agent to Server)

Table 10 lists the ingest endpoints and Table 11 the response codes, which are standard HTTP status codes as defined in RFC 9110 [10]. The server MUST ignore events whose eventId it has already processed, returning 202. Events with a seq lower than the last one processed for that agent from the same sender MUST be recorded in history but MUST NOT overwrite current state.

Table 10: Ingest endpoints.
<table><tr><td>Method</td><td>Path</td><td>Purpose</td></tr><tr><td>POST</td><td>/v1/events</td><td>Submit one event.</td></tr><tr><td>POST</td><td>/v1/events:batch</td><td>Submit an array of events, for agents buffering during brief outages.</td></tr><tr><td>POST</td><td>/v1/traces</td><td>OPTIONAL. OTLP/HTTP ingest; the server maps spans to ASP events (Section 9.3).</td></tr></table>

Table 11: HTTP status codes returned by the ingest endpoints.
<table><tr><td>HTTP status code</td><td>Meaning</td></tr><tr><td>202 Accepted</td><td>Event accepted.</td></tr><tr><td>400 Bad Request</td><td>Malformed JSON or schema violation.</td></tr><tr><td>401 Unauthorized</td><td>Missing or invalid credentials.</td></tr><tr><td>409 Conflict</td><td>Event sent before register, or invalid transition (e.g. out of a terminal task state).</td></tr><tr><td>429 Too Many Requests</td><td>Agent exceeds the server&#x27;s rate limit; retry after the Retry-After header.</td></tr></table>

## 8.2 Distribution (Server to Observers)

Table 12 lists the distribution endpoints. An observer SHOULD first fetch the snapshot, then open the stream. Each SSE message [25] carries the ASP event as its data field and the event type as the SSE event field. The SSE id field MUST carry a server sequence number, so that observers reconnecting with Last-Event-ID receive the updates they missed. The stream MAY accept filters as query parameters, such as agentId or conversationId.

Table 12: Distribution endpoints.
<table><tr><td>Method</td><td>Path</td><td>Purpose</td></tr><tr><td>GET</td><td>/v1/agents</td><td>Snapshot of current state for all agents.</td></tr><tr><td>GET</td><td>/v1/agents/{agentId}</td><td>Snapshot for one agent, including open tasks and conversations.</td></tr><tr><td>GET</td><td>/v1/stream</td><td>SSE stream of state updates.</td></tr></table>

## 8.3 Alternative Transports

For deployments with thousands of agents or high event rates, bindings over WebSocket, gRPC streaming or a message broker (MQTT, NATS) MAY be defined. They MUST carry the same event schema. They are not part of this draft.

## 9 Mappings to Existing Standards

These mappings let adapters generate ASP events from agents that already speak other protocols, without modifying them.

## 9.1 A2A

Table 13 maps A2A elements [2] to ASP. A2A has no notion of floor; A2A-only agents report no floor dimension.

Table 13: Mapping from A2A to ASP.
<table><tr><td>A2A element</td><td>ASP</td></tr><tr><td>Task.status.state</td><td>state_change with dimension: task; values map one-to-one to the A2A-origin states in Table 6.</td></tr><tr><td>Task.id</td><td>taskId</td></tr><tr><td>contextId</td><td>contextId</td></tr><tr><td>Agent Card name, version, skills</td><td>register fields name, version, capabilities</td></tr><tr><td>TaskStatusUpdateEvent</td><td>Source events for an adapter observing A2A streaming traffic.</td></tr></table>

## 9.2 OFP

Table 14 maps OFP elements [21] to ASP. OFP event names follow the OFP specification at the time of writing; the mapping SHOULD be revalidated against the OFP version in use.

The OFP events uninvite and bye both end an agent’s participation and map to left. The reason field distinguishes the two: uninvited for a removal decided by the convener, bye for a voluntary departure.

The OFP manifest exchange (getManifests, publishManifests) has no counterpart among ASP states. Publishing a manifest is a one-time exchange of capability information rather than a condition that persists and changes over time, and capability negotiation is a non-goal of ASP. An adapter MAY copy the capabilities declared in a published manifest into the capabilities field of register.

## 9.2.1 What an Adapter Can and Cannot Infer

An adapter that observes OFP trafic can derive floor states and the role of each agent in the conversation reliably, because every floor transition is an explicit event. It can also infer that an agent has joined or left, or has not spoken for some time. It cannot tell, from conversation events alone, whether a silent agent is making a long model call, waiting for a tool or a peer, or is stuck: knowing who holds the floor does not reveal what the holder is doing. Lifecycle, health, progress and activity therefore require signals from the agent’s runtime, either through native ASP reporting or through an adapter that has access to the runtime rather than only to the conversation. An adapter SHOULD NOT report states it cannot observe, and SHOULD leave such dimensions unreported rather than guess them.

Table 14: Mapping from OFP to ASP.
<table><tr><td>OFP element</td><td>ASP</td></tr><tr><td>Conversation id</td><td>conversationId</td></tr><tr><td>requestFloor</td><td>floor_change to requesting-floor</td></tr><tr><td>grantFloor</td><td>floor_change to has-floor for the recipient</td></tr><tr><td>yieldFloor</td><td>floor_change to yielded</td></tr><tr><td>revokeFloor</td><td>floor_change to floor-revoked</td></tr><tr><td>invite</td><td>floor_change to invited</td></tr><tr><td>uninvite</td><td>floor_change to left, reason: uninvited</td></tr><tr><td>declineInvite</td><td>floor_change from invited to left, reason: declined</td></tr><tr><td>bye</td><td>floor_change to left, reason:bye</td></tr><tr><td>publishManifests</td><td>No state change; capabilities MAY populate register</td></tr><tr><td>getManifests</td><td>No ASP equivalent</td></tr><tr><td>Convener</td><td>role: convener</td></tr></table>

## 9.3 OpenTelemetry

ASP identifiers and metrics reuse the OpenTelemetry GenAI semantic conventions [22] where they exist (Table 15). Those conventions are still experimental, and the mapping may change with them. Recommended metrics names, all OPTIONAL, are latencyMs, queueDepth, activeTasks, maxConcurrentTasks, tokensIn, tokensOut, costUsd and errorRate.

OpenTelemetry spans alone cannot express lifecycle, health or floor states. An agent ingested only via OTLP therefore shows task state but relies on span arrival for liveness.

Table 15: Mapping from OpenTelemetry GenAI conventions to ASP.
<table><tr><td>OpenTelemetry</td><td>ASP</td></tr><tr><td>gen_ai.agent.id</td><td>agentId</td></tr><tr><td>gen_ai.agent.name</td><td>name</td></tr><tr><td>invoke_agent span start</td><td>task working</td></tr><tr><td>invoke_agent span end, status OK</td><td>task completed</td></tr><tr><td>invoke_agent span end, status ERROR</td><td>task failed</td></tr><tr><td>gen_ai.usage.input_tokens</td><td>metrics.tokensIn</td></tr><tr><td>gen_ai.usage.output_tokens</td><td>metrics.tokensOut</td></tr></table>

## 10 Use Case: Monitoring a Multi-Agent Debate

This section follows a short sequence of ASP messages sent while a group of agents debates an open research question in an OFP conversation. The scenario matches the dashboard in Appendix A: a Moderator acts as convener, a Planner, a Fact Checker and a Critic take turns on the floor, and a Research Assistant works on a separate task. All agents send their events to the same status server, which forwards the resulting state to a dashboard over SSE. The Moderator and the convener are not two diferent things that need two names. Moderator is simply the name that this example gives to one agent, and it describes the agent’s function in the application, which is to moderate the debate. Convener is the role that the same agent holds in the OFP conversation, carried in the role field of its floor\_change events (Section 5.4). A name belongs to the agent and does not change, whereas a role belongs to the agent’s participation in one conversation: the same agent could be the convener of one conversation and a participant in another, and any agent, whatever its name, MAY hold the convener role.

Registration. Each agent registers when it begins to operate in the deployment, in this example at start-up. The message below shows the registration of the Planner in full, including the common fields that the remaining examples omit.

1 { {   
" aspVersion ": " 0.1 ",   
" type ": " register ",   
" eventId ": "5 f0c2e1a -7 b3d -4 c9e -9 a41 -2 d8e6b1f0a77 ",   
" seq": 1 ,   
" agentId ": "urn : agent : example : planner ",   
" name ": " Planner ",   
" version ": " 2.0.1 ",   
9 " protocols ": [" ofp "],   
10 " capabilities ": ["task - decomposition ", " argumentation "] ,   
11 " heartbeatIntervalSec ": 10,   
12 "ts": " 2026 -09 -22 T16 :58:00 Z"   
13 }

Taking the floor. When the Planner wants to speak, it asks the convener for the floor through OFP and reports the transition to the status server. When the Moderator grants the floor, the Planner reports the second transition. The two events travel as two separate POST requests, six seconds apart.

1   
2 " type ": " floor\_change ",   
3 " agentId ": "urn : agent : example : planner ",   
" conversationId ": "c -44 ",   
" from ": " joined ",   
6 "to": " requesting - floor ",   
7 " role ": " participant ",   
8 "ts": " 2026 -09 -22 T17 :02:09 Z"   
9 }   
10   
11 {   
12 " type ": " floor\_change ",   
13 " agentId ": "urn: agent : example : planner ",   
14 " conversationId ": "c -44 ",   
15 " from": " requesting - floor ",   
16 "to": "has - floor ",   
17 " role ": " participant ",   
18 "ts": " 2026 -09 -22 T17 :02:15 Z"   
19 }

Waiting for input. Meanwhile the Research Assistant, working on task t-8812, needs a clarification before it can continue. It reports the task transition together with a short reason, so that an operator can see why the task stopped without opening the agent’s logs.

```jsonl
1 {
" type ": " state_change ",
" agentId ": "urn : agent : example : research - assistant ",
" dimension ": " task ",
" from ": " working ",
"to": "input - required ",
" taskId ": "t -8812 ",
" reason ": " clarification - needed : which time period to cover ",
"ts": " 2026 -09 -22 T17 :02:11 Z"
10 }
```

Quarantine by a guardrail. A guardrail component, for example a Sentinel Agent [12], that screens the output of the Summary Writer detects a suspected prompt injection. It sends a governance event about the Summary Writer, authenticated with its own credential. The server accepts the event because that credential is authorised to set governance states for other agents, and it records the guardrail as the setter (Section 12). A later self-report by the Summary Writer cannot clear the state.

```jsonl
1 {
2 " type ": " state_change ",
3 " agentId ": "urn : agent : example : summary - writer ",
<sup>4</sup> <sub>5</sub> " source ": " guardrail ",
" dimension ": " governance ",
6 " from ": " normal ",
7 "to": " quarantined ",
8 " reason ": " suspected -prompt - injection ",
9 "ts": " 2026 -09 -22 T17 :01:58 Z"
10 }
```

Missed heartbeats. The Web Browser declared a heartbeat interval of 10 seconds, and its last event arrived at 17:01:39. When 25 seconds pass without further events, which is 2.5 times the declared interval, the server marks the agent as unreachable (Section 7). This transition is generated by the server itself, so no agent sends it: it appears only on the observer side. The dashboard receives it over SSE as follows, with the JSON payload split across several data lines, which SSE joins before delivery.

```jsonl
1 event : state_change
2 id: 48213
3 data : {" type ": " state_change ",
4 data : " agentId ": " urn : agent : example :web - browser ",
5 data : " dimension ": " lifecycle ",
6 data : " from ": " busy ", "to": " unreachable ",
7 data : " reason ": "no - heartbeat -25 s"
8 data : "ts": " 2026 -09 -22 T17 :02:04 Z"}
```

Across the whole sequence no agent needs to know that a dashboard exists, and the dashboard never polls an agent: every update travels as one small POST into the server and one SSE message out of it. The last known task of the Web Browser stays visible on the dashboard, marked as stale, until the agent sends a new event or its live state expires.

## 11 Governance with Sentinel Agents

The Sentinel Agent architecture [12] places a network of Sentinel Agents alongside the functional agents of a multi-agent system. Sentinels oversee inter-agent communication and raise alerts on suspected prompt injection, hallucination, data exfiltration, collusion or privacy breaches; a Coordinator Agent supervises policy, manages agent participation and, on the basis of those alerts, isolates or quarantines misbehaving agents. The architecture describes the roles but not a common format for alerts and decisions. Without one, each sentinel needs its own integration with the coordinator, and the decision to quarantine an agent remains invisible to the conversation manager, the dashboard and the other agents until each is told separately. This section shows how ASP provides that format. Nothing in it is required for ASP conformance.

## 11.1 Roles

In ASP terms, a Sentinel Agent reports with source: guardrail and a Coordinator Agent, which acts on an operator’s behalf, with source: operator; both use credentials authorised to report on other agents (Section 12). Under the precedence rules of Section 7.2.3, a state set by the coordinator can be cleared only by the coordinator or a human operator, and no state set by either can be cleared by the monitored agent. Sentinels and the coordinator are themselves agents, so they register, send heartbeats and report their own state like any other.

## 11.2 From Alert to Quarantine

A typical sequence has three steps. It difers from the use case of Section 10, in which a single guardrail quarantines the agent directly: here detection and decision are separated. First, a sentinel that detects a suspected threat sets the monitored agent to under-review, with a short reason code and, optionally, detail under ext, which never contains the ofending content (Section 12):

```jsonl
1 {
2 " type ": " state_change ",
3 " agentId ": "urn : agent : example : summary - writer ",
4 " source ": " guardrail ",
5 " dimension ": " governance ",
6 " from ": " normal ",
7 "to": "under - review ",
8 " reason ": " suspected -prompt - injection ",
9 " ext ": { "org . example . sentinel ": {
10 " sentinelId ": "urn: agent : example : sentinel -2"
11 " confidence ": 0.92 } },
12 "ts": " 2026 -09 -22 T17 :40:12 Z"
13 }
```

Second, the coordinator, subscribed to the SSE stream with a filter on the governance dimension, receives the event within the same stream that serves the dashboard, correlates it with alerts from other sentinels and with the agent’s health and task states, and may also query the agent directly. It then decides, and sets the agent to quarantined or sandboxed with source: operator, or back to normal if the alert is not confirmed. Third, the decision takes efect across the system: the convener revokes the agent’s floor, which appears as floor-revoked; orchestrators stop routing tasks to it; the operator MAY revoke its ingest credential (Section 7.2); and every observer sees the change at the same moment.

## 11.3 State as Input to Sentinels

ASP state is also useful to sentinels as input. Patterns in the state stream can indicate threats that the content of a single message does not reveal: an agent that reports healthy while peers record repeated call-timeout observations, a task that oscillates between working and retrying, a burst of registrations from unknown agents, or an agent that delegates an unusual number of tasks. Sentinels MAY use such patterns to raise alerts, and MAY use state to focus their scrutiny, for example on newly registered or recently degraded agents.

## 11.4 Monitoring the Monitors

Because sentinels report their own liveness, a gap in coverage is visible: a sentinel that becomes unreachable or stalled leaves the agents it watches unmonitored. A coordinator SHOULD treat such a gap as a governance event in its own right, for example by setting the afected agents to sandboxed until coverage is restored. The append-only history of governance transitions, each recorded with its source and credential, provides the audit trail that the Sentinel Agent architecture calls for in support of regulatory compliance [9].

## 12 Security Considerations

• Authentication. Every ingest request MUST be authenticated. Bearer tokens scoped per agent are RECOMMENDED; mutual TLS MAY be used. This draft does not specify how credentials are issued, scoped, rotated or revoked; these are deployment-specific in version 0.1 (Section 14). The server MUST reject events whose agentId does not match the credential, except governance events sent by a credential explicitly authorised to set governance states for other agents, such as a guardrail component, and observations sent with a source other than self by a credential explicitly authorised to report on other agents (Section 7.2.3).

• Transport. All endpoints MUST be served over TLS in any non-local deployment.

• Message-level integrity. In topologies without a dedicated status server (Section 4.3), no single component verifies credentials at ingest. Events SHOULD then be signed, for example with JSON Web Signature, so that every observer can verify who reported each state, in particular who set a governance state.

• Observer access. Snapshot and stream endpoints MUST require authorisation. State reveals which systems exist, what they are doing and when they fail, which is valuable to an attacker.

• Data minimisation. Events MUST NOT carry prompts, model outputs, user data or secrets. The reason fields follow the format of Section 6.4: a short code, optionally followed by a neutral description.

• Spoofing. A compromised agent could report false state, e.g. hiding a quarantined status. Governance states set by guardrails or operators MUST take precedence over self-reported ones, and the server MUST record who set each governance state. More generally, the server MUST record the source and the credential of every event, and SHOULD expose self-reports and independent observations separately so that a compromised agent cannot mask what others observe (Section 7.2.3).

• Denial of service. Servers SHOULD rate-limit per agent and cap event size (RECOM-MENDED: 16 KB).

• Audit. Servers SHOULD keep an append-only history of events for forensic analysis, with retention set by the operator.

## 13 Extensibility and Versioning

Vendor-specific fields MUST be placed under an ext object, keyed by a reverse-domain namespace, e.g. "ext": {"org.example": {...}}. Servers MUST preserve and forward unknown ext content.

Custom state values MUST use the x- prefix, e.g. x-warming-cache. Observers SHOULD display unknown states as-is, grouped with the nearest standard state if a hint is provided.

The field aspVersion follows semantic versioning. Minor versions add optional fields or states only. A server MUST accept events from any minor version of the major version it supports.

## 14 Open Issues

The following questions were raised for future versions of the specification. They are not exhaustive: the authors invite the community to comment on them, to challenge the choices already made in this draft, and to raise further questions.

1. Should floor state be tracked for A2A multi-turn exchanges, using contextId as a pseudoconversation?

2. Is a formal JSON Schema file part of the normative specification, or published alongside it?

3. Is a pull-mode binding worth standardising, given that adapters already cover unmodifiable agents? The single probe of Section 7.2.2 is a first step; a full binding would let the server also collect state from agents that never push.

4. Should peer observations be able to change an agent’s health when several independent peers agree, and if so with what quorum?

5. Should an event for updating capabilities after registration be defined, for example when an OFP agent publishes a new manifest?

6. Should ASP define a standard credential profile, for example OAuth 2.0 client credentials with separate scopes for reporting an agent’s own state, reporting on other agents and observing, so that agents, sentinels and observers from diferent vendors can authenticate to any conforming server?

7. Should bindings be defined for topologies without a dedicated status server (Section 4.3), such as an in-band binding for OFP or an MQTT binding, and should event signing then be REQUIRED?

8. Should ASP define a registry of governance reason codes, for example data-exfiltration or suspected-collusion, so that alerts from diferent sentinel implementations can be compared (Section 11)?

9. Should the state vocabulary, the event layer and the behaviour layer (Section 1.10) be published as three separate documents?

10. How do observers know where agents are registered, especially in multi-server environments?

## 15 Future Directions

Version 0.1 is a working draft. The directions below are intentions rather than commitments, and their scope and order may change with feedback from the community. Contributions from implementers, protocol designers and standards bodies are welcome in each of them. The project repository [1] is intended as the common place for this work, from the text of the specification to the reference implementation, the JSON Schema and the conformance tests.

Reference implementation and schema. A possible next step is a reference implementation of the status server and of the observer dashboard, which would make the behaviour of Sections 7 and 8 testable, including the optional mechanisms of Section 7.2. A normative JSON Schema for each event type could support automatic validation, and a conformance test suite could let implementers check each layer of Section 1.10 separately.

Validation with independent adopters. The specification would benefit from being exercised by implementations other than its authors’. One candidate for a first validation is the Playground, an interactive environment for OFP conversations that already records floor events and message routing and visualises agent states on a canvas. A small experiment could start with two or three agents, use an adapter to produce ASP events from the existing Playground events, and compare the Playground view with the ASP dashboard in scenarios such as waiting for the floor, a deliberately delayed response, a timeout, and an interruption of status reporting. Such an experiment could help clarify which states can be mapped reliably from conversation events, which need additional runtime signals, and how the meaning of individual states holds up in practice. Trials with A2A-only agents, and with agents that speak neither protocol, would be a natural way to test the claim of protocol neutrality.

Governance and security. The integration with Sentinel Agents outlined in Section 11 could be explored in a simulated environment, for example by looking at how quickly a sentinel alert becomes a quarantine visible to every observer, and at whether state patterns usefully complement content analysis in detecting attacks. Experience of this kind might also inform the shared registry of governance reason codes and the treatment of peer quorum listed among the open issues (Section 14).

Scale and standardisation. For deployments with thousands of agents, bindings over Web-Socket, gRPC streaming or a message broker (Section 8) may be worth specifying, together with guidance on aggregating state across several status servers. A common credential profile, if the open issue on authentication is settled in that direction, could also be specified, so that components from diferent vendors can interoperate without bespoke configuration.

## 16 Conclusion

Multi-agent systems are growing faster than the means of observing them: as agents and the links between them multiply, operators lose sight of which agents are alive, what each is doing, who is waiting on whom and which agent should no longer be trusted. ASP addresses this gap with a lightweight, protocol-neutral specification. Agents push small JSON events over HTTP and observers receive the aggregated state as soon as it changes. State is modelled along five independent dimensions (lifecycle, health, task, floor and governance), and the state vocabulary can be adopted on its own.

ASP does not rely on agents to report their own failures. Liveness is inferred by the server from missed heartbeats, progress signals separate a long operation from a stuck agent, and guardrails, supervisors, operators and peers can report on other agents, with operator and guardrail decisions taking precedence over self-reports, so that a quarantine decided by a Sentinel or Coordinator Agent is visible to every component at once. By reusing the vocabularies of A2A, OFP and OpenTelemetry, and by carrying state but never content, ASP complements existing standards and protocols rather than replacing them. It defines the state, not the agent, the framework or the vendor. ASP is ofered as a proposal and a starting point: the authors invite the community to review it, test it against real deployments and help shape what a common standard for agent state should become, starting from the project repository [1].

## References

[1] Agent Status Protocol Project. Agent Status Protocol (ASP): Specification and implementations. https://github.com/AgentStatusProtocol/spec, 2026.

[2] Agent2Agent (A2A) Project. Agent2agent protocol specification. Linux Foundation, https: //a2a-protocol.org, 2026.

[3] Fabio Luigi Bellifemine, Giovanni Caire, and Dominic Greenwood. Developing Multi-Agent Systems with JADE. Wiley, 2007.

[4] Scott Bradner. Key words for use in RFCs to indicate requirement levels. RFC 2119, IETF, 1997.

[5] Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail?, 2025.

[6] Tushar Deepak Chandra and Sam Toueg. Unreliable failure detectors for reliable distributed systems. Journal of the ACM, 43(2):225–267, 1996.

[7] Liming Dong, Qinghua Lu, and Liming Zhu. AgentOps: Enabling observability of LLM agents, 2024.

[8] Abul Ehtesham, Aditi Singh, Gaurav Kumar Gupta, and Saket Kumar. A survey of agent interoperability protocols: Model context protocol (MCP), agent communication protocol (ACP), agent-to-agent protocol (A2A), and agent network protocol (ANP), 2025.

[9] European Parliament and Council of the European Union. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (artificial intelligence act). Oficial Journal of the European Union, L series, 2024.

[10] Roy T. Fielding, Mark Nottingham, and Julian Reschke. HTTP semantics. RFC 9110, IETF, 2022.

[11] Foundation for Intelligent Physical Agents. FIPA agent management specification. Document SC00023K, FIPA, 2004.

[12] Diego Gosmar and Deborah A. Dahl. Sentinel agents for secure and trustworthy agentic AI in multi-agent systems, 2025.

[13] Diego Gosmar and Deborah A. Dahl. Hallucination mitigation with agentic AI NLP-based open-floor standard. In Proceedings of the 18th International Conference on Agents and Artificial Intelligence - Volume 5: ICAART, pages 3893–3900. INSTICC, SciTePress, 2026.

[14] Diego Gosmar and Deborah A. Dahl. Prompt injection mitigation with agentic AI, nested learning, and AI sustainability via semantic caching, 2026.

[15] Diego Gosmar, Deborah A. Dahl, Emmett Coin, and David Attwater. AI multi-agent interoperability extension for managing multiparty conversations, 2024.

[16] gRPC Authors. gRPC health checking protocol. https://github.com/grpc/grpc/blob/ master/doc/health-checking.md, 2026.

[17] Naohiro Hayashibara, Xavier Défago, Rami Yared, and Takuya Katayama. The φ accrual failure detector. In Proceedings of the 23rd IEEE International Symposium on Reliable Distributed Systems (SRDS), pages 66–78, 2004.

[18] Graham Klyne and Chris Newman. Date and time on the internet: Timestamps. RFC 3339, IETF, 2002.

[19] Barry Leiba. Ambiguity of uppercase vs lowercase in RFC 2119 key words. RFC 8174, IETF, 2017.

[20] Model Context Protocol Project. Model context protocol specification. https:// modelcontextprotocol.io, 2025.

[21] Open Voice Interoperability Initiative. Open floor protocol specifications. Linux Foundation AI & Data, https://github.com/open-voice-interoperability, 2026.

[22] OpenTelemetry Authors. Semantic conventions for generative AI systems. https:// opentelemetry.io/docs/specs/semconv/gen-ai/, 2026.

[23] Traian Rebedea, Razvan Dinu, Makesh Sreedhar, Christopher Parisien, and Jonathan Cohen. NeMo guardrails: A toolkit for controllable and safe LLM applications with programmable rails, 2023.

[24] The Kubernetes Authors. Liveness, readiness, and startup probes. https://kubernetes. io/docs/concepts/configuration/liveness-readiness-startup-probes/, 2026.

[25] WHATWG. HTML living standard, section 9.2: Server-sent events. https://html.spec. whatwg.org/multipage/server-sent-events.html, 2026.

## Appendix A: Reference Dashboard

Figure 4 shows an illustrative observer dashboard built on the distribution endpoints of Section 8. It displays twelve agents with generic roles and exercises every state dimension defined in Section 5. The dashboard is intended for a human operator who supervises the deployment; the same distribution endpoints also serve software observers.

The upper strip gives one segment per agent, coloured by lifecycle state, followed by aggregate counts. The agent table on the left is sorted by attention needed: the unreachable agent comes first, with its last known task marked as stale as required by Section 7, followed by the quarantined agent, whose floor has been revoked. Each row reports lifecycle, health, task and floor state, together with the age of the last heartbeat.

The panel on the upper right shows the floor state of an OFP conversation: the convener at the centre, the agent holding the floor, and the queue of agents requesting it. The event stream below lists the most recent state\_change and floor\_change events received over SSE, newest first. Every state carries a text label in addition to its colour, so the figure remains readable in greyscale. All data are illustrative.

![](images/6d81426f251da08ff53541e8e2b11b86652cafdc4b3ab05b6a73655143a55c00.jpg)  
Figure 4: Illustrative ASP reference dashboard: lifecycle strip and counts (top), agent table sorted by attention needed (left), floor state of an OFP conversation (upper right) and live event stream (lower right).