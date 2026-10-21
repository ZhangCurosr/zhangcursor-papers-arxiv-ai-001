# From Code Generator to Engineering Companion: A Plant-aware Agent for PLC Programming

Joschka Kersting, Gesa Benndorf

Fraunhofer IOSB-INA

Lemgo, Germany

{ . i ae ak @th s .dn s - ea er, ng hi g ns dok r .nb.c f oe f rs n} be r fio a o uj

Abstract—Programmable Logic Controller (PLC) coding remains a manual, IDE-centric task that demands constant context switching between documentation, project files, and code editors. Recent work has demonstrated that retrieval-augmented large language models can generate vendor-compliant Structured Text at high compilation rates [1], yet these systems treat code generation as a one-shot pipeline: the engineer submits a query and receives code, with no persistent knowledge of the specific plant. We present a plant-aware conversational copilot that extends this paradigm in three directions: plant-specific context ingestion, a chat-first interaction model for explanation and debugging, and a graduated tool-access ladder from chat to deployment. The system is designed for real plant infrastructure and keeps the engineer in control of each step that produces real-world effects.

Index Terms—LLM, PLC, Structured Text, IEC 61131-3, Engineering Copilot, Plant Context, Human-in-the-Loop 1

## I. INTRODUCTION

Every production facility runs on control logic authored by automation engineers who navigate a demanding workflow: read the I/O specification, open the vendor IDE, locate the relevant function-block documentation, write Structured Text (ST) code, compile, fix errors, and repeat. Large language models have begun to reduce this friction. Prior work [1] demonstrated that a retrieval-augmented coding assistant for Mitsubishi Electric’s GX Works3 environment can generate vendor-compliant ST code at high compilation rates using only a structured knowledge base of function-block definitions and documentation excerpts, without exposing proprietary code to cloud providers.

Yet that earlier system operates as a pipeline. The engineer submits a natural-language task, the model generates code, and the compilation server validates it. There is no persistent knowledge of the specific plant, let alone an iterative dialogue, and no path from chat to deployment.

The gap we address is not code generation per se, but the engineering interaction model. A copilot that knows the plantspecific I/O list, labels, and existing program fragments can generate code that is immediately appropriate for the target namespace. Moreover, in our setting we find that agents often

only write correct code when they know the real-world setup of the plant. Together, these aspects form the difference between a generic coding tool and an engineering companion. Unlike generic copilots (e.g., GitHub Copilot on a CODESYS MCP server), our contribution is not the chat mechanism itself but its coupling with plant-specific context retrieval and a domaintailored, safety-graduated tool ladder.

Contributions

• Plant-aware context ingestion that handles the specific plant’s I/O list, device labels, module configuration, and existing program fragments.

• A conversational interaction model for explaining, debugging, design discussion, and code modification.

• Graduated tool access with four explicit levels: Chat, Code, Compile, and Deploy.

## II. RELATED WORK

LLM-based PLC code generation. One line of work establishes iterative generation pipelines: a user-guided approach combines compiler, grammar-checker, and formal verification feedback, raising ST generation success on a manufacturing testbed from 47% to 72% [2]. A fully automated multi-agent successor adds RAG and Chain-of-Thought planning with dedicated syntax and verification agents; on a benchmark of 23 programming tasks (16 easy, 7 medium), the framework achieves a verifiable rate of 68% on easy tasks and 42.3% on medium tasks with GPT-4o [3]. A vendor-aware framework targets ST fragmentation across two proprietary platforms with a four-stage pipeline (planning, case retrieval, API recommendation, IDE-guided validation), reporting high compilation pass rates on a large benchmark at low average pertask cost [4]. A complementary fine-tuning direction applies iterative online Direct Preference Optimization, using compiler feedback and a specialist LLM as synthetic annotators to construct preference pairs without manual labelling [7]; a separate system extends generation beyond ST to Sequential Function Charts, achieving 75–94% syntactically valid output [8].

LLM-based DCS engineering. A series of incremental contributions culminating in Spec2Control established a complete LLM workflow for Distributed Control System (DCS) control logic: an initial study evaluated LLM capabilities across 100 PLC/DCS prompts in ten categories [9]; a followup integrated vendor function-block libraries via RAG [10];

TABLE I
<table><tr><td>System</td><td>Plant context</td><td>Code gen</td><td>Compile</td><td>Deploy</td><td>Chat-first</td><td>Multi-vendor</td><td>Target platform(s)</td></tr><tr><td>LLM4PLC [2]</td><td></td><td>√</td><td>√</td><td>一</td><td></td><td>一</td><td>Siemens SCL</td></tr><tr><td>Agents4PLC [3]</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>IEC 61131-3 (generic)</td></tr><tr><td>AutoPLC [4]</td><td></td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>Siemens SCL, CODESYS ST</td></tr><tr><td>Spec2Control [5]</td><td></td><td>√</td><td>一</td><td>√</td><td></td><td></td><td>ABB</td></tr><tr><td>Siemens Copilot [6]</td><td></td><td>√</td><td>√</td><td>√</td><td></td><td></td><td>Siemens TIA</td></tr><tr><td>Prior work [1]</td><td></td><td>√</td><td>√</td><td>1</td><td></td><td></td><td>Mitsubishi iQ-R</td></tr><tr><td>This work</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td><td>Mitsubishi iQ-R</td></tr></table>

a third variant generated code from P&ID diagrams using vision-capable models [11]. Spec2Control presents an endto-end LLM workflow to generate graphical IEC 61131- 3 Function Block Diagrams directly from natural-language control narratives and deploy them to an automation controller [5]. It produces FBDs from proprietary ABB libraries and achieves 98.6% correct strategy connections across 65 test cases from 10 narratives, with a 94–96% reduction in engineering effort [5]. The key distinction from our work is the interaction model: Spec2Control is a batch workflow in which the engineer is consulted at low-confidence steps, whereas our system is conversational-first and leaves each tool action to explicit engineer initiation.

Commercial engineering copilots. Several vendors have shipped conversational code assistance integrated into their proprietary IDEs [6], [12], [13], showing that major automation vendors are productizing this paradigm in commercial tools. These products lack controlled evaluation datasets; Spec2Control characterises them as targeting textual notations with limited automation support [5].

Human-in-the-loop and graduated autonomy. HITL patterns for LLM agents are well established in orchestration frameworks [14], [15], where agents pause before high-stakes tool calls pending explicit approval; human-guided development workflows following this pattern have been deployed in production software engineering [16]. Recent work formalises agent autonomy as a five-level spectrum of user roles— from full operator control to passive observation—providing a principled vocabulary for autonomy-level design [17]. Our contribution applies this design pattern to the industrial PLC domain, where an uncontrolled LLM-to-hardware action can cause physical damage and production downtime.

Evaluation methodology. Existing work measures primarily compilation success and formal verification rates [2]– [4]; Spec2Control adds GQM-structured quality and effort metrics [5]. For conversational copilots, however, metrics capturing interaction efficiency (turns to task completion), error distribution by correction cause, and cognitive load remain absent—gaps that our planned evaluation aims to address.

## III. SYSTEM DESIGN

The system extends the RAG-enhanced compilation architecture from [1] with three new layers: a persistent plantcontext retrieval segment, a conversational interaction model, and a graduated tool-access mechanism.

## A. Conversational Engineering Interface

The copilot presents a persistent, multi-turn chat interface. Unlike the single-query model of [1], the engineer can engage in extended dialogue: asking the system to explain an existing program fragment, identify potential interlock conflicts, discuss design alternatives, or propose modifications before any code is generated.

![](images/3001ff919588cf819716d6535ded4d97a2185150b344030920f055dbde9fc880.jpg)  
Fig. 1. The prototype web application used to control generation, compilation, and plant-aware context. The left panel exposes generation settings and MCP actions, while the main pane shows the conversation and generated ST code.  
Figure 1 shows the web application that orchestrates the

generation workflow, the plant context toggle, and the manual compile and deployment actions.

Code generation is explicitly request-triggered. When the engineer asks for code, the model draws on all context layers; in subsequent turns the engineer can ask for modifications, and the model maintains the generated program as a shared artefact in the conversation.

## B. Plant Context Integration

We define plant-awareness as the system’s ability to ground generated code in the target plant’s I/O list, label namespace, module configuration, and physical signal topology. The key architectural addition over [1] is a dedicated plant-context retrieval segment. On project initialization, the engineer provides:

• I/O list: signal names, addresses, data types, and descriptions for all inputs and outputs.

• Device labels: the complete GX Works3 label table, including internal variables and inter-program signals.

• Module configuration: rack layout, CPU type, and special-function module assignments.

• Existing program fragments: current ST programs and function-block instances already deployed on the plant.

These materials are chunked, embedded, and indexed in a dedicated plant-context retriever, separate from the general function-block and specification retrievers established in [1]. At query time, a four-retriever strategy is used: library definitions, specification rules, general context, and plant-specific context. Hence, similarity search operates within homogeneous chunk spaces and avoids cross-source interference.

## C. Graduated Tool Access

Safety in industrial environments demands that LLMgenerated artefacts are never silently applied to running machinery. The copilot enforces four explicit levels of tool access:

• Level 1 – Advisory: text only, no tool call.

• Level 2 – Code generation: code is displayed in the chat interface on explicit request.

• Level 3 – Compilation: the engineer triggers the compilation server and receives structured diagnostics.

• Level 4 – Deployment: the engineer explicitly confirms deployment to the PLC.

Each level transition produces real-world effects of increasing consequence. The graduated structure ensures that advisory interactions carry no risk of unintended hardware modification.

## IV. DEMONSTRATION AND PRELIMINARY RESULTS

## A. Setup

Figure 3 shows the physical setup used in the demonstration. The demonstration plant is a sorting machine that, based on air pressure and light sensors, can push metal discs of different colors onto a conveyor belt. The colors are chrome, red and black. Since these colors reflect differing amounts of light, the sensors can be configured to detect them with different thresholds. Another actuator with sensors can put inserts into the discs. Users may configure their own sorting app that configures a mixture of colored discs with or without inserts using AI. The plant can thus be reconfigured based on users current needs. The prototype may serve as a demonstration for more complex plants, e.g., in the automotive industry.

![](images/7f20284f54e7aec0513ef9494c9b2b1e7de3de77a455b2975e75478805b875d1.jpg)  
Fig. 2. System architecture for our ST coding assistant with plant-aware retrieval, user-triggered compilation and PLC control via MCP (based on [1]).

The demo setup uses a conveyor belt that transports discs to the right where, having passed all magazines, they are sorted into three boxes. Chrome, as the most reflective color, comes first, followed by red and black. Chrome and red are actively sorted by a pneumatic cylinder that pushes them off the belt, while black discs are passively sorted by gravity. In Fig. 3, the plant is pushing red off the conveyor belt.

The plant is equipped with a Mitsubishi Electric iQ-R series CPU programmed via a prototype web interface and GX Works3. Plant context was ingested as an I/O list, a label table, and existing ST program fragments. Moreover, the actual plant setup had to be explained to the model, since the I/O list and labels alone do not capture the physical layout and signal semantics. That is, the model had to be informed where magazines are, when they are loaded (i. e. the corresponding sensor must be triggered as TRUE) and that the discs are pushed on a conveyor belt to the right, where actuators in a certain order are situated.

![](images/af62f24eac8ff1895cfd336fb27cbb804ec755f738961dfe0e7bc70b4a9dab9a.jpg)  
Fig. 3. Demonstration plant setup at a research test facility providing metal discs in different colors, inserts, sensors, and actuators for sorting them. The plant is equipped with a Mitsubishi Electric iQ-R series CPU (model R04).

## B. Example Interaction Trace

The following excerpt shows the intended interaction pattern for plant-aware code generation and validation.

Engineer: Are there blue discs available?

Copilot (Advisory – L1): No. The plant context defines only chrome, red, and black discs. There is no blue disc magazine, sensor, or sorting path.

Engineer: Generate the sequence red, red with insert, chrome, black with insert, chrome.

Copilot (Code gen – L2): The generated Structured Text uses the plant’s actual labels and control flow, including the black-disc branch gated by BLACK\_MAG\_FUELLSTAND and INSERT\_MAG\_FUELLSTAND.

Engineer: [Triggers compilation]

Copilot (Compile – L3): GX Works3 compilation successful.

Engineer: Can the black disc also receive an insert? Copilot (Code gen – L2): Yes. The same insertstation logic applies to the black disc before it continues on the black path. Code: ...

Engineer: [Triggers compilation]

Copilot (Compile – L3): GX Works3 compilation successful.

Engineer: [Triggers deployment to PLC, chooses to delete latches and variable status; Starts PLC run]

Copilot (Deploy – L4): Code deployed to PLC. The plant is now executing the new sequence. See Fig. 3.

## V. DISCUSSION & CONCLUSION

This paper presented a plant-aware conversational copilot for PLC engineering that extends the RAG-enhanced code generation architecture of prior work [1] with persistent plantcontext ingestion, a chat-first multi-turn interaction model, and a graduated tool-access ladder from chat to deployment. By retrieving plant-specific signal names at query time, the copilot is intended to generate code that is more appropriate for the target plant’s namespace. A structured quantitative evaluation comparing plant-aware and generic-context generation on matched task sets is planned.

Limitations include that the current system is demonstrated on a single plant within Mitsubishi Electric’s iQ-R ecosystem. Cross-vendor generalizability is not in the scope. An evaluation methodology should capture functional correctness, which requires hardware-in-the-loop testing or domain-expert review beyond compilation success. Human evaluation from a single organizational context may limit generalizability.

Future Work includes (A) execution feedback connecting PLC runtime signals and the copilot context, (B) multi-PLC orchestration for cross-controller installations, (C) graphical IEC 61131-3 extensions for Ladder diagram generation and legacy AWL code, and (D) a quantitative evaluation across multiple industrial-size plants with plant-awareness metrics.

## REFERENCES

[1] J. Kersting, M. Rummel, and G. Benndorf, “Vendor-aware industrial agents: RAG-enhanced LLMs for secure on-premise PLC code generation,” in 2026 IEEE International Conference on Industrial Technology (ICIT), 2026.

[2] M. Fakih, R. Dharmaji, Y. Moghaddas, G. Quiros Araya, O. Ogundare, and M. A. Al Faruque, “LLM4PLC: Harnessing large language models for verifiable programming of PLCs in industrial control systems,” in Proc. 46th Int. Conf. Software Engineering: Software Engineering in Practice (ICSE-SEIP), Lisbon, Portugal, 2024, pp. 192–203.

[3] Z. Liu et al., “Agents4PLC: Automating closed-loop PLC code generation and verification in industrial control systems using LLM-based agents,” arXiv:2410.14209, Oct. 2024.

[4] D. Yang, A. Wu, T. Zhang, L. Zhang, F. Liu, X. Lian, Y. Ren, J. Tian, and X. Che, “AutoPLC: Generating vendor-aware structured text for programmable logic controllers,” arXiv:2412.02410, Dec. 2024.

[5] H. Koziolek, T. Braun, V. Ashiwal, S. Linsbauer, M. A. Hansen, and K. Grøtterud, “Spec2Control: Automating PLC/DCS control-logic engineering from natural language requirements with LLMs — A multiplant evaluation,” arXiv:2510.04519, Oct. 2025.

[6] Siemens, “Engineering copilot TIA standard (managed-service),” https: //www.siemens.com/EngineeringCopilotTIA, 2024, 2026-05-07.

[7] A. Haag et al., “Training LLMs for generating IEC 61131-3 structured text with online feedback,” arXiv:2410.22159, Oct. 2024.

[8] O. Glick et al., “LLM4SFC: Sequential function chart generation via large language models,” arXiv:2512.06787, Dec. 2025.

[9] H. Koziolek, S. Gruener, and V. Ashiwal, “ChatGPT for PLC/DCS control logic generation,” in Proc. IEEE ETFA, 2023.

[10] H. Koziolek et al., “LLM-based and retrieval-augmented control code generation,” in Proc. 1st Int. Workshop LLMs for Code, 2024, pp. 22–29.

[11] H. Koziolek and A. Koziolek, “LLM-based control code generation using image recognition,” in Proc. 1st Int. Workshop LLMs for Code, 2024, pp. 38–45.

[12] Beckhoff, “TwinCAT chat: Using innovative chatbot technology in TwinCAT engineering,” https://www.beckhoff.com/, Apr. 2023.

[13] B&R Automation, “Automation studio copilot,” https://www. br-automation.com/, 2024.

[14] LangChain, “Human-in-the-loop middleware,” https://docs.langchain. com/, 2025.

[15] OpenAI, “Human-in-the-loop — OpenAI agents SDK,” https://platform. openai.com/docs/, 2025.

[16] W. Takerngsaksiri et al., “Human-in-the-loop software development agents,” arXiv:2411.12924, 2025.

[17] K. J. K. Feng et al., “Levels of autonomy for AI agents,” arXiv:2506.12469, Jun. 2025.