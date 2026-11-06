# Create Your Own Work Stack: Deploying Self-hosted AI and Tooling in a Tech Startup

Saad Ahmed Bazaz<sup>∗†</sup>, Talha Amjad<sup>∗</sup>, Abdurrehman Subhani<sup>∗</sup>, Sohaib Ahmed Bazaz,

Abdullah Aamir

Grayhat Developers (Pvt) Ltd

{bazaz, talha.amjad, subhani, sohaib.bazaz, abdullah.aamir}@grayhat.studio

## Abstract

Self-hosting AI is often presented as a way to reduce inference costs and increase control, but deployment introduces its own costs. We report our experience deploying a hybrid AI stack at Grayhat Developers, a small software company with an AI-intensive development workflow. Our initial motivation was to counter the growing cost of commercial AI usage: Claude/- Codex incurred approximately \$325 per month, with more than 25.93 billion recorded tokens. We experimented with local open-source models as replacements for commercial coding agents, but after learning that their performance and inference speed were insuficient, we adopted a hybrid approach in which commercial agents remained responsible for complex tasks while local models handled lower-complexity tasks, organizational information retrieval, and synchronization with self-hosted tools including Outline, Plane, and ERP-Next.

We estimated that approximately 30–40% of our commercial AI workload was associated with context transfer, documentation synchronization, task management, and related operational coordination. By moving this work to local models, we estimate that approximately 90% of this coordination workload could be handled without commercial inference, corresponding to an estimated 27–36% reduction in overall commercial-AI workload or cost-equivalent usage. Our first internal deployment nevertheless failed to achieve adoption because it introduced UX friction. Replacing it with a centrally configured deployment substantially improved feedback and adoption. We conclude that local inference is most useful not as a wholesale replacement for commercial AI, but as part of a hybrid architecture that assigns tasks according to their capability requirements, cost, context demands, and operational sensitivity.

## Introduction

Often, the biggest believers are the biggest critics.

We are heavy users of AI at Grayhat. We’re a team of under 40 people who build software for startups in San Francisco and develop our own products, and AIassisted development has become part of our normal workflow. We were therefore not looking for a reason to use less AI. We were looking for a way to use it without allowing the cost and operational complexity to grow with every additional task.

Our immediate problem was commercial AI usage. At one point, five Claude Pro seats could be exhausted around the middle of a billing period. For some routine work, the economics became uncomfortable enough that assigning the work to a human developer was cheaper than continuing to consume commercial model capacity.

What followed was not a clean engineering project. We tried things that did not work, changed direction several times, and discovered that some of our assumptions about AI adoption were wrong — ultimately documenting a deployment pattern and empirical retrospective showing how a small software organization can use local inference as an organizational-context and orchestration layer while retaining commercial models for capability-intensive development tasks. This can be explained as reminiscent to the Kübler-Ross model of grief (Kübler-Ross 1969).

## Our Experience

## Denial: The Icarus of Vibe-coding

In early 2026, we took on a series of ambitious hypercasual game development projects with delivery cycles ranging from one to two weeks per game. The projects ran in parallel and involved fewer people than we would traditionally have assigned to that amount of work. Hu man involvement was concentrated primarily around reviewing outputs, QA, and final validation. This created a highly AI-dependent workflow.

<sup>∗</sup>These authors contributed equally.

We tried to keep much of our project context in code and repositories. This worked reasonably well for developers who already understood the project, but it soon disconnected the rest of the organization from that context, including designers and managers.

Our initial hypothesis was that AI would primarily reduce development costs. We instead found friction in collaboration and management, so we connected the AI to MCP servers (Model Context Protocol 2026) for each self-hosted tool: Outline (Outline 2026) for documentation, Plane (Plane 2026) for ticket management, and ERPNext (Frappe Technologies Pvt. Ltd. 2026) for general management and organizational information.

Table 1: AI-assisted development before introducing a local LLM.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Monthly commercial AI cost</td><td>$325</td></tr><tr><td>Recorded token usage</td><td>25.93B+</td></tr><tr><td>Estimated coordination workload</td><td>30-40%</td></tr><tr><td>Quality</td><td>Very High</td></tr></table>

The resulting AI usage was already substantial. Seven Claude Team/Claude Code seats cost approximately \$175 per month, while seven Codex subscriptions added approximately \$140; usage-based credits added approximately \$10. Across these services, recorded activity exceeded 25.93 billion tokens at a recurring cost of approximately \$325 per month (Table 1).

Based on repeated workflow observation, approximately 20% of a typical usage window was consumed by documentation synchronization and a further 10– 15% by task and project-management synchronization. We therefore estimated that 30–40% of workload involved context transfer, tool descriptions, MCP calls, and operational synchronization rather than coding or higher-value reasoning. This was an operational estimate rather than a controlled token measurement. Prior work similarly identifies substantial context overhead in MCP-based tool use. (Jayanti and Han 2026)

## Anger: Open-source AI and Its Frustrations

Our first proposed solution was simple: try to cut out the paid models entirely.

We experimented with self-hosted models, including Kimi (Kimi Team 2025) and DeepSeek (DeepSeek-AI 2024), using various harnesses and open-source IDEs on real development tasks.

The experience was frustrating for the engineers, who had prior experience with paid cloud models. The models were useful for some tasks, but the gap became obvious when we asked them to perform more complicated coding work. A model could produce reasonable snippets and handle smaller requests while still being unsuitable as a replacement for the coding agents our developers were already using.

The gap was particularly noticeable for repositorylevel work requiring large context windows and fast inference. On realistic consumer hardware, the models were either too slow or insuficiently capable of understanding existing code, dependencies, and conventions without afecting unrelated parts of the application. (Jimenez et al. 2024; Zhang et al. 2024)

## Bargaining: Deciding on Tasks and Using a Hybrid Approach

Instead of trying to replace our commercial coding agents, we took a more hybrid approach. The concept of a single, universal AI model for everyday grunt work was flawed - many tasks, similar to the ones in nature, do not require 100% of brain power at all times. We therefore decided which tasks could be handled by local models and which required the capabilities of commercial coding agents.

Approach to the hybrid work stack We had already deployed Outline, Plane, and ERPNext using Coolify (CoolLabs 2026) on a machine running on our premises. We tied together authentication for these tools using Authentik (Authentik 2026). We then self hosted the MCP servers that connected our AI systems to internal tools on the same machine. The hardware configuration is provided in Appendix A.

We used a separate machine for AI workloads. Its hardware and software configuration is also provided in Appendix A.

We started with Gemma 3 2.7B (Gemma Team 2025), and later shifted to the latest available Gemma model at the time of deployment. The LLM was deployed as llama.cpp instances managed by GPUStack (GPUStack Authors 2026). The AI chat interface (explained in sections ahead) was connected to the AI inference server through local IP.

The process, highlighted in Figure 1, was roughly as follows:

1. Humans use local models to retrieve information from Outline and Plane, summarize it, and transform it into a form that can be used by coding agents, such as files in the repository like AGENTS.md.

2. Commercial coding agents remain the preferred option for complex coding tasks and repository-level development. Tasks are written to files in the repository for the coding agents to use.

3. Humans use local models to read the tasks produced by coding agents and translate them into tickets that can be pushed to Plane.

## Depression: People Still Don’t Use It

Our first internal AI interface, called version 1, was LobeHub (LobeHub 2026). We set it up on our own devices, and created a tutorial for other users to follow to connect it to our MCP infrastructure. After QA, we released the system for external testing with the rest of the company.

However, the team did not adopt it. The first few demos were great, but people did not actually use it regularly. We initially thought the reason was that people were accustomed to the quality of commercial AI products and therefore found our implementation lower quality.

It turned out that the biggest hurdle was UX. Despite being a team of engineers, users found it dificult to configure the AI environment reliably. They needed to know which MCP servers were available, how to configure them, and how to authenticate against the systems they wanted to access, especially in a local environment. Commercial AI products had an advantage here because much of this complexity was hidden from the user.

![](images/3ecf663a19d0820dd26dee6b257abef877c0414f4e1a8398c8ddb2e18ffeddf2.jpg)  
Figure 1: Overview of the hybrid AI workflow process. The self-hosted AI resembles a "moat" around the organization’s internal tools.

Table 2: Categorized feedback for version 1 from 11 structured employee interviews.
<table><tr><td>Category</td><td>Comments</td></tr><tr><td>Setup &amp; Onboarding</td><td>8</td></tr><tr><td>AI Performance &amp; Model Quality</td><td>5</td></tr><tr><td>Reliability, Uptime &amp; Guardrails</td><td>27</td></tr><tr><td>Tool Integrations</td><td></td></tr><tr><td>UX &amp; Trust in Output</td><td>3</td></tr><tr><td>Feature Requests</td><td>1</td></tr><tr><td>Positive / Neutral Sentiment Total</td><td>2 28</td></tr></table>

## Acceptance: Self-hosted LibreChat and Pre-configured MCPs

We replaced LobeHub with a self-hosted LibreChat (LibreChat 2026) deployment, called version 2. With LibreChat, we moved configuration to a central deployment rather than ofloading it to users. Users authenticate new MCPs when first accessing the application.

The resulting configuration is shown in Appendix B.

This reduced the number of decisions required before use and improved feedback and adoption from the team.

Table 3: Categorized feedback for version 2 from unstructured user feedback on Discord, with change from version 1.
<table><tr><td>Category</td><td>Comments</td><td>Change</td></tr><tr><td>Setup &amp; Onboarding</td><td>3</td><td>-62.5%</td></tr><tr><td>AI Performance &amp; Model Quality</td><td>6</td><td>+20.0%</td></tr><tr><td>Reliability, Uptime &amp; Guardrails</td><td>7</td><td>+250.0%</td></tr><tr><td>Tool Integrations</td><td>6</td><td>-14.3%</td></tr><tr><td>UX &amp; Trust in Output</td><td>5</td><td>+66.7%</td></tr><tr><td>Feature Requests</td><td>7</td><td>+600.0%</td></tr><tr><td>Positive / Neutral Sentiment</td><td>5</td><td>+150.0%</td></tr><tr><td>Total</td><td>39</td><td>+39.3%</td></tr></table>

## Learnings

The original motivation was reducing commercial AI usage and its associated cost. Our observations suggest that approximately 30–40% of the commercial-AI workload was associated with context synchronization and operational coordination, and approximately 90% of that workload could be handled by local inference. This corresponds to an estimated displacement of 27–36% of overall commercial-AI workload, or approximately \$88– 117 per month when expressed as a cost-equivalent fraction of the observed \$325 monthly commercial-AI spend (Table 4). The cost still does not drop dead to zero because of equipment costs, electricity cost, and maintenance costs (USD per hour) by the engineers themselves. We view it as moving some costs from recurring SaaS and model usage into infrastructure and engineering overhead, while gaining more control over the systems and the information flowing through them.

Feedback was categorized into seven issue types: setup and onboarding, AI performance and model quality, reliability and guardrails, tool integrations, UX and trust in output, feature requests, and positive or neutral sentiment. Each distinct reported observation was counted once within its corresponding category. Feed back for v1 is visible in Table 2, while feedback for v2 is visible in Table 3. The change in feedback between v1 and v2 is also summarized in Table 3. Because v1 and v2 feedback were collected using diferent mechanisms and at diferent stages of deployment, these counts are descriptive rather than a controlled measure of improvement.

## Lessons From the Deployment

Networking in our internal network was dificult. Setting up routing for LLM inference to a local machine was harder than expected. Server uptime was also a problem — running the server 24/7 was not needed so we added scheduling, and we added an uninterruptible power supply (UPS) to avoid downtime during power outages.

Table 4: Estimated commercial-AI cost-equivalent exposure avoided after introducing a local LLM.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Monthly commercial AI cost baseline Estimated coordination workload</td><td>$325</td></tr><tr><td rowspan="3">Coordination workload handled locally Estimated overall workload displaced</td><td>30-40%</td></tr><tr><td>~90% 27-36%</td></tr><tr><td>~$88-$117/month</td></tr><tr><td>Cost-equivalent workload displaced Local LLM operating cost</td><td>$10-$30/month</td></tr><tr><td>Quality</td><td>High</td></tr></table>

We also had to add monitoring tools such as Beszel (Henry G. D. 2026), as well as analytics within LibreChat itself, to catch errors early. This meant that, in addition to operational features, we had to build observability as well, and that diverted engineering attention from other commercial work.

Overall, we had to adopt a self-hosting mindset as an entire team. Everyone became responsible for being aware of the servers and for considering what kind of work was being given to each AI model, instead of blindly trusting a commercial model to handle everything. For a small company, this tradeof needs to be made deliberately.

We also learned that local inference is not always the best option. Research into stronger open-source coding models such as GLM-4 (Zeng et al. 2024) continues, but for a small startup, the hardware and setup costs may not be worthwhile for every workload. At the same time, commercial model usage can become expensive when large amounts of context are repeatedly processed. But as mankind and capitalism demand costoptimization at every stage in tech, higher engineering speed demands that token cost be optimized. So, local inference does not have to replace commercial inference. The two can work in parallel.

This small example illustrates the distinction: I don’t need my model to know Einstein’s theory of relativity to write a simple Jira ticket. Assigning simpler tasks to smaller local models can therefore reduce cost without requiring the organization to give up stronger commercial models for tasks where they provide a meaningful advantage. (Chen, Zaharia, and Zou 2023; Ong et al. 2024)

Context is part of the cost. The cost of an AI system is not just the cost of generating an answer. If the model needs to retrieve large amounts of information before it can answer, the retrieval and context-construction process becomes part of the workload.

AI adoption can fail because of UX even when the AI works. Our LobeHub deployment was a useful example. There was no fundamental technical blocker preventing people from using it. The system simply required too much efort from users before it became useful.

The move also reduced our dependence on external AI providers for internal organizational information. Keeping some retrieval and processing local means that not all internal context needs to be sent to an external model provider, hence organizational data can more or less be kept more private than otherwise.

## Limitations and Future Work

Our current setup is still fairly small and manually managed.

The division between local and external models is based largely on our experience with diferent classes of tasks rather than an automated routing system. We would like to make this more systematic, potentially using a lightweight classifier or policy layer, like Jev (Almeida 2026), to determine which model should handle a request.

We are also interested in separating model responsibilities further. One possible direction is to use a larger local model primarily for reasoning while using a smaller local model for tool selection and execution. This could reduce the amount of expensive inference required for routine tool interactions.

Another direction is deeper integration with the coding agents our developers already use. At present, the internal AI system and the coding agents are largely separate systems. We would like them to communicate directly, allowing a coding agent to request organizational context from our local agents without requiring a person to manually move that information between systems. However, autonomous behavior might cause another increase in token usage. A query-manager agent, potentially also running locally or at the edge (on the device where the query was originally made), could sit on top of the system and delegate tasks to the coding agent and internal tool agent.

Our cost measurements are operational rather than controlled experiments. We did not establish a formal baseline before beginning the deployment, and several changes happened at approximately the same time. The cost figures therefore describe our experience rather than establishing that self-hosting will produce the same savings for another organization.

We plan to create a replicable environment, provisionally called “WorkStack,” as a one-click deployment. This would allow us to reproduce the environment more consistently and measure future changes against a clearer baseline.

## References

Almeida, D. 2026. Introducing System One Models & Jev. TypeSafe AI Blog.

Authentik. 2026. Authentik. Open-source identity provider and authentication platform.

Chen, L.; Zaharia, M.; and Zou, J. 2023. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.

CoolLabs. 2026. Coolify. https://github.com/ coollabsio/coolify.

DeepSeek-AI. 2024. DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. arXiv preprint arXiv:2401.02954.

Frappe Technologies Pvt. Ltd. 2026. ERPNext. https: //github.com/frappe/erpnext.

Gemma Team. 2025. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786.

GPUStack Authors. 2026. GPUStack: A GPU Cluster Manager for High-Performance AI Model Serving. Open-source software.

Henry G. D. 2026. Beszel. https://github.com/ henrygd/beszel.

Jayanti, M. A.; and Han, X. Y. 2026. Enhancing Model Context Protocol (MCP) with Context-Aware Server Collaboration. arXiv:2601.11595.

Jimenez, C. E.; Yang, J.; Wettig, A.; Yao, S.; Pei, K.; Press, O.; and Narasimhan, K. 2024. SWE-bench: Can Language Models Resolve Real-World GitHub Issues? In The Twelfth International Conference on Learning Representations.

Kimi Team. 2025. Kimi k1.5: Scaling Reinforcement Learning with LLMs. arXiv preprint arXiv:2501.12599.

Kübler-Ross, E. 1969. On Death and Dying. New York: Macmillan.

LibreChat. 2026. LibreChat. https://github.com/ danny-avila/LibreChat.

LobeHub. 2026. LobeHub. https://github.com/ lobehub/lobehub. Version 2.2.14.

Model Context Protocol. 2026. Model Context Protocol Specification. Online specification. Accessed: September 8, 2026.

Ong, I.; Almahairi, A.; Wu, V.; Chiang, W.-L.; Wu, T.; Gonzalez, J. E.; Kadous, M. W.; and Stoica, I. 2024. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665.

Outline. 2026. Outline. https://github.com/outline/ outline.

Plane. 2026. Plane. https://github.com/makeplane/ plane.

Zeng, A.; Xu, B.; Wang, B.; Zhang, C.; Yin, D.; Zhang,

D.; Rojas, D.; Feng, G.; Zhao, H.; Lai, H.; Yu, H.; Wang,

H.; Sun, J.; Zhang, J.; Cheng, J.; Gui, J.; Tang, J.;

Zhang, J.; Sun, J.; Li, J.; Zhao, L.; Wu, L.; Zhong, L.;

Liu, M.; Huang, M.; Zhang, P.; Zheng, Q.; Lu, R.; Duan,

S.; Zhang, S.; Cao, S.; Yang, S.; Tam, W. L.; Zhao, W.;

Liu, X.; Xia, X.; Zhang, X.; Lv, X.; Liu, X.; Zhang, X.;

Song, X.; Song, X.; Liu, X.; An, Y.; Xu, Y.; Niu, Y.;

Yang, Y.; Li, Y.; Bai, Y.; Dong, Y.; Qi, Z.; Wang, Z.;

Yang, Z.; Du, Z.; Hou, Z.; and Wang, Z. 2024. Chat-GLM: A Family of Large Language Models from GLM-130B to GLM-4 All Tools. arXiv:2406.12793.

Zhang, S.; Zhao, H.; Liu, X.; Zheng, Q.; Qi, Z.; Gu, X.; Zhang, X.; Dong, Y.; and Tang, J. 2024. NaturalCodeBench: Examining Coding Performance Mismatch on HumanEval and Natural User Prompts. arXiv:2405.04520.

## A Infrastructure Configuration

Self-hosted Tools Machine

• Operating system: Ubuntu 24.04.1 LTS

• Architecture: x86\_64

• Kernel: 6.8.0-54-generic

• CPU: 6 cores, Intel Core i5 8th Generation

• Memory: 16 GB DDR4 RAM

• Storage: 500 GB SSD

## AI Workload Machine

• Operating system: Ubuntu 25.10

• Architecture: x86\_64

• Kernel: 6.17.0-35-generic

• CPU: 20 cores, Intel Core i5 13th Generation

• GPU: NVIDIA RTX 4080 16 GB

• Memory: 32 GB DDR5 RAM

• Storage: 2 TB NVMe SSD

## B Version 2 LibreChat Interface

Figure 2 shows the self-hosted LibreChat deployment with pre-configured MCP connectors, tool-specific agents, and a routing agent.

![](images/608f6a8322e4f323553e81be5f0791b05641d266ec2bdff566117eeac92a839d.jpg)  
Figure 2: Self-hosted LibreChat with pre-configured MCP connectors and tool-specific agents for Plane, Outline, and ERPNext.