![](images/52148adf3099398bc070a87bde76e606363bcb6f771c20c4a62a59667ddd44b0.jpg)

# AGENTBUG-SMITH : AUTOMATICALLY REPRODUCING REAL-WORLD HARNESS BUGS IN AGENTIC SYSTEMS

Yiming Cheng The University of Chicago eaminchan@uchicago.edu

Mengshi Zhang and Zihao Chen TensorBlock, Inc. {mengshiz, zihaoc}@tensorblock.co

Yiling Lou University of Illinois Urbana-Champaign yilingl@illinois.edu

Alfin Wijaya Rahardja   
Fudan University   
24212010055@m.fudan.edu.cn   
Zhenpeng Chen   
Tsinghua University   
zpchen@tsinghua.edu.cn

## ABSTRACT

Agent harness bugs exhibit unique characteristics and remain challenging for state-of-the-art software agents to repair. Progress in this area is further hindered by existing benchmarks, which contain only a small and fixed number of executable harness bugs while requiring hundreds of human hours to construct. This work presents AGENTBUG-SMITH, an automated harness bug reproduction approach that continuously discovers and reproduces real-world harness bugs from open-source agentic systems. Across different backbone LLMs, AGENTBUG-SMITH consistently outperforms existing bug reproduction techniques designed for general software, achieving 10.67% - 27.56% higher success rates of reproducing harness bugs. By applying AGENTBUG-SMITH to open-source agentic systems in the wild, we construct LIVE-HARNESS-BENCH, a live and extensible benchmark that currently contains 200 reproducible harness bugs. We further demonstrate the utility of LIVE-HARNESS-BENCH through two downstream applications. First, we use LIVE-HARNESS-BENCH as the evaluation benchmark to systematically evaluate state-of-the-art software agents, revealing their limited capabilities in repairing real-world harness bugs. Second, we use LIVE-HARNESS-BENCH as a knowledge base of real-world harness bug fixes, from which reusable repair skills can be distilled to improve existing software agents, increasing their harness-bug repair rates by 6.32%. Together, AGENTBUG-SMITH and LIVE-HARNESS-BENCH establish a scalable foundation for continuously evaluating and improving software agents on harness bug repair, turning real-world agent failures into executable evaluation instances and reusable knowledge for harness improvement, thus contributing to the ultimate goal of recursively self-improving agents.

## AGENTBUG-SMITH LIVE-HARNESS-BENCH

## 1 INTRODUCTION

Large Language Model (LLM) agents are rapidly emerging as a new software paradigm and have been increasingly adopted across diverse domains (Yang et al., 2024; Kannan et al., 2024; Zeng et al., 2023; Gottweis et al., 2026; Xiao et al., 2025). Agentic systems are commonly composed of backbone LLMs and their surrounding harnesses, where the harness serves as a software infrastructure that works with LLMs to jointly shape the agent behavior. Modern agents involve increasingly complex and large-scale harness implementations (e.g., 200K lines of code in OpenClaw (ope, 2026a)) to support diverse functionalities such as context management, security guardrails, orchestration, and tool integration. Accordingly, there have been growing research efforts (Li et al., 2026) on harness engineering, demonstrating that improving the harness alone can substantially enhance agent effectiveness, even when the underlying backbone LLM remains unchanged. More recently, advances in self-evolving harnesses (Lee et al., 2026) and recursive self-improvement (Liu et al., 2026) have further elevated the harness from a static supporting layer to an evolving component of agentic systems, underscoring the critical role of harness quality in determining the agent capabilities.

The growing complexity of agent harnesses inevitably makes harness implementation bugs an important source of failures in agentic systems. Recent studies (Rahardja et al., 2025; Zhang et al., 2026b; Chen et al., 2026) have revealed that modern agent applications and frameworks suffer from diverse implementation bugs in harnesses. Automatically detecting and repairing such bugs is therefore becoming increasingly critical for building reliable agentic systems. However, prior work (Rahardja et al., 2025) has shown that existing software agents (Yang et al., 2024; Xia et al., 2024; Zhang et al., 2024) achieve substantially lower success rates in repairing harness bugs than general software bugs (e.g., 4.67% versus 40.67%). This difficulty stems in part from the distinct characteristics of harness bugs, which arise from agent-specific structures and behaviors, such as complex workflow orchestration and interactions with external resources (e.g., tools, services, and model providers). These characteristics introduce failure modes and repair challenges that are uncommon in traditional software systems. Taken together, automated harness bug repair represents an emerging and distinct challenge in improving agent reliability.

However, we still lack comprehensive benchmarks of harness bugs. Most benchmarks of agent fail ures (Cemri et al., 2025; Zhu et al., 2025) focus on failures stemming from backbone LLMs rather than harness implementation bugs. While there have been increasing effort to investigate harness im plementation bugs (Zhang et al., 2026b; Chen et al., 2026), existing bug collections lack executable environments and artifacts for reproducing the agent failures. To date, the only executable bench mark of reproducible harness bugs, AgentIssue-Bench (Rahardja et al., 2025), contains only a small and fixed set of 43 harness bugs, which cannot support comprehensive or rigorous evaluation due to the limited number of bugs and the potential risk of contamination from exposure to model training data. In addition, manually reproducing harness bugs is labor-intensive and time-consuming, e.g., 150 manual hours were spent in reproducing a small number of harness bugs (Rahardja et al., 2025).

While recent approaches such as SWE-Factory (Guo et al., 2025) and SWE-bench-Live (Zhang et al., 2026a) have automated the reproduction of bugs reported in GitHub Issues, they are primarily designed for general software systems. Harness bugs in agentic systems pose unique reproduction challenges, as agent harnesses extensively interact with external resources, including LLM providers, tools, network services, and dynamically changing environments, while their failures often manifest only under specific execution states and interaction contexts. These characteristics fundamentally distinguish harness bugs from conventional software bugs and pose substantial challenges to directly applying existing automated bug reproduction techniques to agentic systems.

Technique. This work proposes AGENTBUG-SMITH, an automated harness bug reproduction approach that continuously discovers and reproduces real-world harness bugs from open-source agents. AGENTBUG-SMITH is built as a multi-agent system to streamline reproduction pipeline without any manual intervention, facilitating three phases including (i) harness bug related issues identification, (ii) agent execution environment construction, and (iii) failure-reproducing harness test generation.

Evaluation. We perform extensive evaluation to show the effectiveness of AGENTBUG-SMITH. First, based on our manual checking, AGENTBUG-SMITH achieves a high accuracy of automatically identifying agent repositories (i.e., 95% accuracy) and harness bug related issues (i.e., 92%). Second, AGENTBUG-SMITH consistently achieves higher success rate of reproducing harness bugs compared state-of-the-art bug reproduction techniques that are designed for general software (i.e., SWE-Factory and SWE-bench-Live), with 10.67% - 27.56% percentage-point improvements across all studied backbone LLMs. Third, our ablation analysis further confirms the effectiveness of both environment and test generation components in AGENTBUG-SMITH compared to baselines.

Benchmark and Downstream Application. Continuously applying AGENTBUG-SMITH to opensource agentic systems in the wild can enabling scalable construction of a benchmark of executable harness bugs, LIVE-HARNESS-BENCH. Its current release contains 200 reproducible harness bugs collected from real-world agent repositories, covering diverse harness components and spanning over wide time ranges. We further demonstrate the usage of LIVE-HARNESS-BENCH in two downstream applications. First, we use LIVE-HARNESS-BENCH as the evaluation benchmark to systematically evaluate state-of-the-art software agents, revealing their limited capabilities in repairing realworld harness bugs. Second, we use LIVE-HARNESS-BENCH as a knowledge base of real-world harness bug fixes, from which reusable repair skills can be distilled to improve existing software agents, increasing their harness-bug repair success rate by 6.32%. Together, AGENTBUG-SMITH and LIVE-HARNESS-BENCH establish a scalable foundation for continuously evaluating and improving software agents on harness bug repair, turning real-world agent failures into executable evaluation instances and reusable knowledge for harness improvement, thereby contributing to the ultimate goal of recursively self-improving agents.

## 2 RELATED WORK

Failures in Agentic Systems. Substantial research efforts have been devoted to analyzing and understanding failures in agentic systems. Most existing studies or benchmarks (Zhang et al., 2025; Zhu et al., 2025; Cemri et al., 2025; Bouzenia & Pradel, 2025) focus on the runtime failures stemming from the underlying LLMs but not the failures caused by implementation bugs in the agent harness layer. Rahardja et al. Rahardja et al. (2025) performed the first study to investigate the implementation bugs in agentic systems, summarizing diverse categories of bugs across different agent components and building the first executable benchmark of agent bugs, AgentIssue-Bench. More recently, there have been increasing effort in investigating the bugs in agent harnesses (Zhang et al., 2026b; Chen et al., 2026; Zhu et al., 2026; Shao et al., 2024; Xue et al., 2025), they do not provide executable environments or artifacts to trigger the failure. To date, AgentIssue-Bench is the only benchmark of executable harness bugs, but only includes a small and fixed number of bugs (i.e., 43) and is constructed with huge manual effort (i.e., 150 hours). To date, we still lack large-scale and comprehensive benchmarks of reproducible agent harness bugs.

Automated Bug Benchmark Construction. Existing techniques have automated the construction of software bug benchmarks and datasets (Yang et al., 2026; Jain et al., 2025), but primarily focus on synthesizing large numbers of artificial bugs through mutation rather than reproducing real-world bugs. Such datasets are often used as training data for model improvement. In this work, we instead focus on constructing benchmarks of real-world bugs to enable more realistic bug distribution in practice. While there has been prior work (Zhang et al., 2026a; Guo et al., 2025; Badertdinov et al., 2026; Tomassi et al., 2019; Pan et al., 2024) that automatically construct benchmarks from real-world bugs in open-source software (e.g., mostly Github issues), they are designed for general software. In contrast, this work specifically focuses on reproducing real-world bugs in agentic systems, which pose unique reproduction challenges due to the extensive interactions between agent harnesses and external resources.

## 3 AGENTBUG-SMITH

In this section, we present AGENTBUG-SMITH, a fully automated approach that reproduces realworld harness bugs from open-source agentic systems and thus can continuously construct a growing benchmark. Following widely-adopted construction pipelines for general-software bug benchmarks (swe, 2026; Guo et al., 2025; Zhang et al., 2026a)), we use GitHub issues as the source of real-world agent failures. Figure 1 presents the overview of AGENTBUG-SMITH, which is built as multi-agent systems covering the following three key stages. (i) Harness Bug Identification, an agentic pipeline that discovers high-quality GitHub repositories of agentic systems and mines issues related to harness bugs (Section 3.1); (ii) Agent Execution Environment Construction, an agentic pipeline that builds environment containers required for agent execution and bug reproduction (Section 3.2); and (iii) Failure-Reproducing Harness Test Generation, an agentic pipeline that synthesizes the harness test that can exactly trigger the issue-described agent failure (Section 3.3).

## 3.1 HARNESS BUG IDENTIFICATION

As the open-source agent ecosystem undergoes rapid and continuous development, static benchmarks can quickly become outdated. To capture the dynamics of this highly active community in real time, we design an automated harness bug identification pipeline to continuously discover

![](images/5e7ba99a68750c8eff367a249cf2b49e9dff514da724d58ab5005ed3b412b2d0.jpg)

Figure 1: Overview of AGENTBUG-SMITH.  
![](images/e9a5ce73580e655524ede1afe2c2d1064ea2e6410591e967b587b3a25da7b3f7.jpg)  
Figure 2: Harness Bug Identification in AGENTBUG-SMITH.

emerging open-source agent repositories and identify GitHub issues related to agent harness bugs.   
Figure 2 illustrates the overall process, which consists of the following two key steps.

Agent Repository Collection. To maintain an up-to-date list of high-quality and widely used agent repositories, we incorporate a dual-stream repository discovery mechanism, including: (1) Curated Initialization, which performs a one-time, large-batch initialization of the repository pool based on community-curated awesome lists of GitHub agentic systems (kyr, 2026; jen, 2026; sla, 2026; e2b, 2026; jim, 2026; roh, 2026; geo, 2026; aih, 2026; fou, 2026; shu, 2026; hyp, 2026; wan, 2026; ten, 2026; xlite dev, 2024); (2) Incremental Stream Update, which continuously processes near-real-time GitHub Archive event streams in small batches to discover active and emerging agent repositories based on developer interactions. The repositories identified through these two streams are merged into a unified candidate pool, allowing the initial curated collection to be continuously expanded with newly emerging projects.

To ensure the quality of selected repositories, we apply two inclusion criteria. First, repositories must demonstrate sufficient community adoption (i.e., at least 50 GitHub stars) and contain tests, provid ing basic support for subsequent environment construction and reproduction test generation. Detailed filtering rules are in Appendix A. Second, repositories must follow commonly adopted agent structures, involving components such as tools, orchestration, and provider integrations. Specifically, following common agent and harness components defined in prior work (Guo et al., 2026; Li et al., 2026; Liu et al., 2024; Meng et al., 2026; Wang et al., 2024a), we design an LLM-based judge to analyze each candidate repository and check whether it implements an agentic system. The detailed specification provided to the LLM judge is included in Appendix B.

Harness Bug Mining. AGENTBUG-SMITH then mines GitHub issues related to harness bugs from each agent repository. First, to ensure the quality of selected issues, we follow a set of rules (detailed in Appendix C) requiring each issue to form a valid issue-pull request (PR) pair, where the associated PR has been successfully merged and contains functional code changes. Moreover, as issues, PRs, and commits often exhibit many-to-many relationships, we further perform one-to-one matching (detailed in Appendix E) to identify the unique buggy and patched commits for each issue.

Second, as revealed in previous work (Rahardja et al., 2025), not all GitHub issues in agent repositories are related to harness implementation bugs. In fact, a non-trivial proportion of GitHub issues involve common bugs (e.g., utility bugs) that can also occur in general software systems. Therefore, to ensure that our constructed benchmark focuses specifically on harness bugs rather than being mixed with general software bugs, we leverage LLMs to inspect each issue and its patch, retaining an issue only if its patch occurs in a harness component. In particular, we follow the common definition of agent harness components (Guo et al., 2026; Li et al., 2026; Meng et al., 2026; Ning et al., 2026), e.g., orchestration, memory, or tools, and design a harness bug specification for the LLM judge (detailed prompt in Appendix D).

![](images/1b9b31c790d0c3b5a8ed87d8563b1334023862aee8d10f6153a150999d3984e4.jpg)  
Figure 3: Environment Construction and Failure-Reproducing Test Generation.

After this stage, AGENTBUG-SMITH returns a set of candidate harness bugs, whose metadata include the issue description text, the linked PR, the buggy commit, and the patched commit.

## 3.2 AGENT EXECUTION ENVIRONMENT CONSTRUCTION

Compared to general software, agentic systems often depend heavily on external resources (e.g., model providers, tools, and external knowledge bases), making the setup of a proper execution environment a critical step in reproducing user-reported harness bugs. While existing techniques (Guo et al., 2025; Zhang et al., 2026a; Hu et al., 2026) can generate execution environments for general software, AGENTBUG-SMITH introduces an environment generation pipeline that automatically synthesizes a Dockerfile (D) for constructing an isolated execution environment specifically tailored to agent-dependent execution resources. In particular, to enforce strict isolation during evaluation, AGENTBUG-SMITH injects mock service environment variables at the container level, transparently rerouting external model client invocations to localized test harnesses without requiring intrusive modifications to the underlying codebase. As illustrated in Figure 3, AGENTBUG-SMITH adopts the following steps for environment construction.

Rule-Guided Initialization. Rather than relying on LLMs to generate Dockerfiles from scratch, AGENTBUG-SMITH first performs a rule-based scanning routine that parses standard package descriptors, including dependency manifests, build configurations, and CI/CD workflows, to automatically infer runtime configurations (e.g., programming language, version constraints, package manager, installation commands, and test runner). Based on the collected context, AGENTBUG-SMITH generates an initial Dockerfile $\mathcal { D } _ { i n i t }$ from an official minimal image with a single LLM API invocation, thereby incurring only minimal LLM inference cost.

Iterative Environment Refinement. The initial Dockerfile $\mathcal { D } _ { i n i t }$ can be imperfect (e.g., omitting a system library, specifying an incompatible package version, or placing the repository on an incorrect import path). AGENTBUG-SMITH therefore incorporates a tool-using agent to iteratively refine D based on build error messages. If the build succeeds, AGENTBUG-SMITH further samples a set of existing tests (e.g., 20) and executes them as an environment smoke test. We intentionally use a lightweight sample rather than the full test suite, as the goal at this stage is to validate basic environment executability rather than repository correctness. Running the full test suite would not only incur substantial overhead during iterative refinement, but could also introduce false-positive environment failures from buggy, flaky, or external-resource-dependent tests. Accordingly, assertion failures are excluded from the diagnostic feedback, allowing the agent to focus on environmentrelated failures such as collection, import, and dependency errors. The refinement loop terminates when the environment is successfully constructed or the maximum number of attempts is reached. In particular, the environment is considered successfully constructed once the image builds successfully and the tests can pass the execution.

## 3.3 FAILURE-REPRODUCING HARNESS TEST GENERATION

A failure-reproducing test is a test that fails on the buggy commit with the same failure reported in the corresponding issue description, while passing on the patched commit. In practice, some bugfixing PRs contain both the patch and an in-patch test (IPT), where the IPT is submitted alongside the patch to validate its correctness and can therefore naturally serve as a failure-reproducing test. However, IPTs are not always available: according to our statistics, less than half (47.11%) of the issues contain IPTs. Restricting reproduction only to issues with IPTs would therefore substantially limit the scope of reproducible bugs and bias the resulting benchmark toward harness bugs for which developers explicitly provided tests. To broaden the coverage of reproducible harness bugs to issues both with and without IPTs, AGENTBUG-SMITH introduces an agentic workflow that automatically synthesizes failure-reproducing tests for a given harness bug, even when an IPT is absent.

The LLM is provided with the corresponding context (e.g., the buggy code repository, issue description, and developer-submitted patch) to generate an initial test T that triggers the failure described in the issue. To ensure faithful reproduction, the generated test must import and exercise the actual buggy implementation rather than replacing the target functionality with mocks. Mocking is permitted only at model-client or network boundaries to isolate test execution from nondeterministic external networks and third-party model APIs, which is consistent with common testing practices in agentic systems (Hasan et al., 2026). In contrast, functions under test cannot be mocked, ensuring that the reproduced failure originates from the actual buggy implementation.

Joint Optimization of Test and Environment. A generated test is considered failure-reproducing only if (i) it fails on the buggy commit with the same failure reported in the issue description and (ii) it passes on the patched commit. If either condition is not satisfied, AGENTBUG-SMITH iteratively refines the test T based on its execution outcomes until both conditions are satisfied or the maximum number of iterations is reached. However, optimizing the test T alone can sometimes be insufficient, as unsuccessful reproduction may stem from misalignment between the generated test T and the environment Dockerfile D. For example, a generated test may exercise an incorrect assertion while the constructed environment simultaneously imports an installed package instead of the checkedout source. To address such cases, beyond test-only optimization, AGENTBUG-SMITH optionally performs joint optimization to co-refine the test T and the environment Dockerfile D. Appendix F presents an example where such joint optimization successfully brings the test into a fail-to-pass state (i.e., failing on the buggy commit while passing on the patched commit), whereas test-only optimization cannot.

## 4 EVALUATION

## 4.1 EFFECTIVENESS OF HARNESS BUG IDENTIFICATION

This section first evaluates the effectiveness of the first component (i.e., harness bug identification) in AGENTBUG-SMITH, as it serves as the foundation for all the subsequent components.

Settings. For continuous harness bug identification, we primarily use cost-effective models (i.e., GPT-4.1-mini) in AGENTBUG-SMITH, as this stage continuously processes a large volume of issues and thus requires cost-effective model inference for scalable deployment. As there is no existing technique specifically designed for harness bug identification, we compare our approach against vanilla LLM invocation as the baseline. To evaluate identification accuracy, we randomly sample 132 out of 225 GitHub issues (corresponding to a 95% confidence level and a 5% margin of error) identified as harness-bug-related by our approach and the vanilla baseline. Two annotators then independently label whether each selected repository is an agent repository and whether each selected issue is related to a harness bug. The resulting Cohen’s kappa is 0.88, indicating high inter-annotator agreement (Landis & Koch, 1977). In addition, given the inherent randomness of LLM inference, we evaluate the stability of our approach by independently classifying each identified issue five times and reporting the set agreement $S _ { \mathrm { s e t } }$ and normalized entropy $S _ { \mathrm { e n t r o p y } }$ (formulas in Appendix H).

Results. As shown in Table 1, AGENTBUG-SMITH achieves high accuracy in identifying both agent repositories (i.e., 95% repository-level accuracy) and harness bug related issues (i.e., 92% issue-level accuracy). Moreover, our approach demonstrates higher stability across repeated executions than vanilla LLM invocation, further supporting the reliability of the issues identified by AGENTBUG-

Table 1: Accuracy and Stability of Harness Bug Identification.
<table><tr><td rowspan="2">Method</td><td colspan="2">Stability</td><td colspan="2">Accuracy</td></tr><tr><td>Sset</td><td>Sentropy</td><td>Repo. Acc.</td><td>Issue Acc.</td></tr><tr><td>Vanilla LLM</td><td>0.72</td><td>0.89</td><td>0.35</td><td>0.46</td></tr><tr><td>AGENTBUG-SMITH</td><td>0.92</td><td>0.94</td><td>0.95</td><td>0.92</td></tr></table>

SMITH. Overall, these results demonstrate that AGENTBUG-SMITH can accurately and consistently identify harness bug related issues in practice.

## 4.2 REPRODUCTION SUCCESS RATE

This section evaluates the bug reproduction success rate of AGENTBUG-SMITH and compares it with state-of-the-art bug reproduction techniques that are designed for general software systems.

Baselines and Settings. We include two state-of-the-art bug reproduction techniques, SWE-Factory (Guo et al., 2025) and SWE-bench-Live (Zhang et al., 2026a), as our baselines and directly adopt their released implementations. Following previous work (Guo et al., 2025), we evaluate all studied techniques using three different and widely used backbone LLMs, including GPT-4.1-mini (GPT, 2025), Kimi-k2.5 (kim, 2026), and DeepSeek-v3.2 (dee, 2025), considering the cost-effectiveness required for continuous application to a large volume of GitHub issues. Since SWE-Factory and SWE-bench-Live are designed to reproduce general software bugs from GitHub issues, they do not include components for identifying harness bug related issues. Therefore, for a fair comparison, we apply all studied approaches to the same set of 225 harness bug-related issues identified by the first component of AGENTBUG-SMITH in the previous section.

Results. As shown in Table 2, despite the inherent difficulty of reproducing real-world harness bugs, AGENTBUG-SMITH successfully reproduces up to 28.00% of the harness bugs, substantially outperforming existing baselines (e.g., 10.67%-27.56% percentage-point improvements) at acceptably higher costs (e.g., 0.56\$-2.44\$ per issue). This result is particularly encouraging given that the best baseline can only achieve up to 13.78% reproduction success rate. Notably, such improvements are consistent across all backbone LLMs, demonstrating the generality of AGENTBUG-SMITH. We further analyze the overlap among the issues reproduced by all techniques (across all backbone LLMs) in Figure 4. Overall, AGENTBUG-SMITH successfully reproduces 75 bugs, including 45 unique bugs that cannot be reproduced by either baseline. These results further demonstrate the unique effectiveness of AGENTBUG-SMITH and highlight the necessity of developing agent-oriented bug reproduction techniques. Furthermore, to validate automatically generated tests truly reproduce the reported failure rather than merely satisfying fail-to-pass, we also perform manual inspection of all the 75 successfully-reproduced bugs, showing 74 (98.67%) of them are valid. Nevertheless, to better understand the limitations of AGENTBUG-SMITH, Appendix G further performs failure analysis of unsuccessful reproduction.

Table 2: Reproduction Success Rate.
<table><tr><td>Method</td><td>Success rate</td><td>Avg. $</td></tr><tr><td colspan="3">GPT-4.1-mini</td></tr><tr><td>SWE-Factory SWE-bench-Live AGENTBUG-SMITH</td><td>21/225 (9.33%) 6/225 (2.67%) 45/225 (20.00%)</td><td>0.09 0.41 0.56</td></tr><tr><td colspan="3">Kimi-k2.5</td></tr><tr><td>SWE-Factory SWE-bench-Live</td><td>19/225 (8.44%) 6/225 (2.67%)</td><td>0.37 0.66</td></tr><tr><td>AGENTBUG-SMITH</td><td>46/225 (20.44%)</td><td>0.91</td></tr><tr><td>SWE-Factory</td><td>DeepSeek-v3.2 31/225 (13.78%)</td><td>0.31</td></tr><tr><td>SWE-bench-Live AGENTBUG-SMITH</td><td>1/225 (0.44%) 63/225 (28.00%)</td><td>1.05 2.44</td></tr></table>

![](images/038ab537c51a0374cd54329b03c019c3e14cb401d1e0566398595cd4eb1086c2.jpg)  
Figure 4: Overlapping Analysis.

## 4.3 ABLATION ANALYSIS

This section performs the ablation analysis to investigate the effectiveness of the remaining two components (e.g., environment construction and test generation) in AGENTBUG-SMITH.

Effectiveness of Environment Construction. In Table 3, the column “Env. Build” compares the effectiveness of environment construction in all studied techniques. In particular, AGENTBUG-SMITH substantially outperforms both baselines across all backbone models, i.e., 10.67% - 35.11% percentage-point improvements in success rate of environment construction. These results suggest that effectively constructing execution environments for agentic systems requires explicitly accounting for their agent-specific dependencies and execution characteristics.

Table 3: Effectiveness of Environment Construction and Test Generation.
<table><tr><td>Method</td><td>Env. Build</td><td>#Test Gen.|</td><td>w/IPT</td><td>wo/ IPT</td></tr><tr><td colspan="5">GPT-4.1-mini</td></tr><tr><td>SWE-Factory</td><td>36/225 (16.00%)</td><td>21</td><td>21</td><td>0</td></tr><tr><td>SWE-bench-Live</td><td>33/225 (14.67%)</td><td>6</td><td>6</td><td>0</td></tr><tr><td>AGENTBUG-SMITH</td><td>84/225 (37.33%)</td><td>45</td><td>24</td><td>21</td></tr><tr><td colspan="5">Kimi-k2.5</td></tr><tr><td>SWE-Factory</td><td>27/225 (12.00%)</td><td>19</td><td>19</td><td>0</td></tr><tr><td>SWE-bench-Live</td><td>29/225 (12.89%)</td><td>6</td><td>6</td><td>0</td></tr><tr><td>AGENTBUG-SMITH</td><td>53/225 (23.56%)</td><td>46</td><td>20</td><td>26</td></tr><tr><td colspan="5">DeepSeek-v3.2</td></tr><tr><td>SWE-Factory</td><td>44/225 (19.56%)</td><td>31</td><td>31</td><td>0</td></tr><tr><td>SWE-bench-Live</td><td>14/225 (6.22%)</td><td>1</td><td>1</td><td>0</td></tr><tr><td>AGENTBUG-SMITH</td><td>93/225 (41.33%)</td><td>63</td><td>34</td><td>29</td></tr></table>

Effectiveness of Test Generation. In Table 3, the “#Test Gen.” column reports the overall number of issues that can be successfully reproduced with failure-reproducing tests, while the “w/ IPT” and “w/o IPT” columns report the numbers of successfully reproduced issues that originally come with and without in-patch tests, respectively. Notably, AGENTBUG-SMITH is capable of generating failure-reproducing tests for issues without any in-patch tests, whereas both baselines fail to reproduce any such issues when no in-patch tests are available as references. This gap demonstrates the effectiveness of the failure-reproducing test generation component, which enables AGENTBUG-SMITH to reproduce a broader scope of harness bugs.

## 4.4 LIVE-HARNESS-BENCH: STATISTICS AND DOWNSTREAM APPLICATION

The evaluation above demonstrates the effectiveness of AGENTBUG-SMITH in reproducing agent harness bugs. Building on this capability, we further apply AGENTBUG-SMITH to construct LIVE-HARNESS-BENCH, a live, large-scale benchmark of executable harness bugs. The current release of LIVE-HARNESS-BENCH contains 200 executable harness bugs and can be continuously and automatically extended with newly emerging harness bugs through AGENTBUG-SMITH.

Statistics of LIVE-HARNESS-BENCH. Table 4 summarizes the complexity of the harness bugs in LIVE-HARNESS-BENCH in terms of the scale of the corresponding code repositories and the developersubmitted gold patches. Overall, the reproduced bugs involve large codebase and non-trivial code changes, highlighting the substantial effort required to navigate the repositories, localize the faults, and implement the corresponding fixes. Figure 5 further shows the distribution of bugs in LIVE-HARNESS-BENCH across different harness components. In particular, LIVE-HARNESS-BENCH provides diverse coverage of all critical agent harness components, for example, with 31.5% of the bugs involving tool registries and action interfaces and 30.5% involving context and memory management. Moreover, Figure 6 presents the temporal distribution of the reproduced bugs, demonstrating consistent coverage across time. Enabled by AGENTBUG-SMITH, LIVE-HARNESS-BENCH can also incorporate recent harness bugs, including those reported as recently as August 2026. Going forward, periodically applying AGENTBUG-SMITH to continuously extend LIVE-HARNESS-BENCH with newly reported bugs can help mitigate concerns about contamination from model training data.

Table 4: Benchmark Statistics.
<table><tr><td>Level</td><td>#Item</td><td>Average</td><td>Median</td></tr><tr><td rowspan="2">Repo.</td><td>LoC</td><td>123k</td><td>83k</td></tr><tr><td>Files</td><td>479</td><td>343</td></tr><tr><td rowspan="2">Patch</td><td>Files</td><td>3.8</td><td>3.0</td></tr><tr><td>Hunks Lines</td><td>19.8 202.9</td><td>12.0 102.5</td></tr></table>

![](images/c29a71b3d4b503bf31377400cba300549c87e188b3abc977f456e33bf747f744.jpg)  
Figure 5: Bug Distribution across Harness Components.

![](images/a970087cb4ea5acdf8b02acd2f83fd6b8a9fb93dca227dbc3f9effbc15906b61.jpg)  
Figure 6: Temporal Distribution of Issue Creation Times in LIVE-HARNESS-BENCH.

Downstream Application I: Evaluation Benchmark. One straightforward application of LIVE-HARNESS-BENCH is to serve as a benchmark for systematically evaluating the capabilities of existing software agents in fixing harness bugs. Table 5 presents the performance of three widely used software agents, mini-SWE-agent (min, 2026), OpenHands (ope, 2026b), and AutoCodeRover (Zhang et al., 2024), with GPT-4.1-mini on LIVE-HARNESS-BENCH. The “plausibly resolved” column refers to cases where the generated patch passes the failure-reproducing tests. Following the previous work (Rahardja et al., 2025) and the common practice in program repair (Xia & Zhang, 2024), we further manually inspect whether the plausible patches are semantically equivalent to the developer-submitted gold patch, which is presented in the “correctly resolved” column. Overall, state-of-the-art software agents exhibit limited resolution rates on LIVE-HARNESS-BENCH (e.g., at most 9.00%), especially compared with their resolution rates reported on general software bugs, e.g., 40.67% on SWE-Bench Verified (swe, 2025). These results highlight the challenges of fixing real-world harness bugs and the importance of constructing reproducible harness bug benchmarks for rigorous evaluation. Detailed analysis of the resolution rates across harness components is in Appendix J.

Table 5: Effectiveness of Software Agents on LIVE-HARNESS-BENCH.
<table><tr><td rowspan=1 colspan=1>Agent</td><td rowspan=1 colspan=1>Plausiblyresolved%</td><td rowspan=1 colspan=1>Correctlyresolved%</td><td rowspan=1 colspan=1>Localization %File-level Function-level</td><td rowspan=1 colspan=1>Avg.$Cost</td></tr><tr><td rowspan=1 colspan=1>mini-SWE-agent</td><td rowspan=1 colspan=1>19.50</td><td rowspan=1 colspan=1>9.00</td><td rowspan=1 colspan=1>75.12        68.66</td><td rowspan=1 colspan=1>0.38</td></tr><tr><td rowspan=1 colspan=1>OpenHands</td><td rowspan=1 colspan=1>17.50</td><td rowspan=1 colspan=1>8.50</td><td rowspan=1 colspan=1>81.00        66.00</td><td rowspan=1 colspan=1>0.03</td></tr><tr><td rowspan=1 colspan=1>AutoCodeRover</td><td rowspan=1 colspan=1>6.00</td><td rowspan=1 colspan=1>3.50</td><td rowspan=1 colspan=1>37.19        31.66</td><td rowspan=1 colspan=1>0.04</td></tr></table>

Downstream Application II: Enhancement. Given the limited capabilities of existing software agents in fixing harness bugs (as shown in Application I), another application of LIVE-HARNESS-BENCH is to serve as a knowledge base for enhancing existing software agents on harness bug repair. To validate this potential, we split LIVE-HARNESS-BENCH into training and test sets. To avoid overfitting and data leakage, we strictly ensure that harness bugs from the same agent repository do not appear in both the training and test sets. As a result, the training set contains 121 harness bugs, while the remaining 79 harness bugs form the test set. Specifically, we implement a basic skill distillation pipeline that iteratively generates and refines a textual skill (SKILL.md) over the training instances. We then apply the distilled skill to an existing software agent, mini-SWEagent, which achieves the best effectiveness in our evaluation in Application I. The detailed skill distillation design is in Appendix I. As shown in Table 6, the distilled skill improves the correct resolution rate of the off-the-shelf software agent by 6.32%, while achieving 31.81% function-level localization accuracy. Appendix K presents an example illustrating how the distilled skill enables mini-SWE-agent to fix a harness bug that it cannot resolve without the distilled skill. Overall, given the continuously growing nature of LIVE-HARNESS-BENCH, it can serve as a scalable data foundation for continuously improving software agents on harness bug repair. More broadly, as harness improvement constitutes an important building block toward recursively self-improving agents,

LIVE-HARNESS-BENCH provides a foundation for advancing this capability through continuously accumulated real-world harness failures and fixes.  
Table 6: Effectiveness of Distilled Skills.
<table><tr><td>Setting</td><td>Plausibly resolved %</td><td>Correctly resolved%</td><td colspan="2">Localization % File-level Function-level</td></tr><tr><td>W/o skill</td><td>35.44</td><td>1.27</td><td>79.32</td><td>34.60</td></tr><tr><td>W/ skill</td><td>41.77</td><td>7.59</td><td>92.31</td><td>66.41</td></tr></table>

## 5 CONCLUSION

This work presents AGENTBUG-SMITH, an automated harness bug reproduction approach that continuously discovers and reproduces real-world harness bugs from open-source agentic systems. AGENTBUG-SMITH substantially outperforms existing bug reproduction techniques designed for general software. Building on AGENTBUG-SMITH, we automatically construct LIVE-HARNESS-BENCH, a live and extensible benchmark that currently contains 200 reproducible harness bugs and can be continuously expanded with newly emerging bugs. Our two downstream application experiments further demonstrate that LIVE-HARNESS-BENCH can serve as both a rigorous evaluation benchmark and a reusable knowledge base for evaluating and improving existing software agents in fixing harness bugs, thereby contributing to the broader goal of recursively self-improving agents.

## REFERENCES

Gpt-4.1-mini, 2025. https://openai.com/index/gpt-4-1/.

Deepseek-v3.2, 2025. https://api-docs.deepseek.com/news/news251201/.

Swe-bench verified, 2025. https://www.swebench.com/verified.html.

Awesome chinese llm, 2026. https://github.com/AiHubCN/Awesome-Chinese-LLM.

Awesome ai agents, 2026. https://github.com/e2b-dev/awesome-ai-agents.

Awesome-foundation-agents, 2026. https://github.com/FoundationAgents/awesom e-foundation-agents.

Awesome ai in finance, 2026. https://github.com/georgezouq/awesome-ai-in-f inance.

Awesome llm-powered agent, 2026. https://github.com/hyp1231/awesome-llm-p owered-agent.

Awesome-ai-agents, 2026. https://github.com/Jenqyang/Awesome-AI-Agents.

Awesome ai agents: Tools, resources, and projects, 2026. https://github.com/jim-sch woebel/awesome\_ai\_agents.

Kimi-k2.5, 2026. https://www.kimi.ai/ai-models/kimi-k2-5/.

Awesome agents, 2026. https://github.com/kyrolabs/awesome-agents.

mini-swe-agent, 2026. https://github.com/swe-agent/mini-swe-agent.

openclaw, 2026a. https://github.com/openclaw/openclaw.

openhands, 2026b. https://github.com/OpenHands/openhands.

Awesome ai apps, 2026. https://github.com/rohitg00/awesome-ai-apps.

Awesome llm apps, 2026. https://github.com/shubhamsaboo/awesome-llm-apps.

Awesome ai agents, 2026. https://github.com/slavakurilyak/awesome-ai-age nts.

Swe leaderboards, 2026. https://www.swebench.com.

Awesome llmops, 2026. https://github.com/tensorchord/Awesome-LLMOps.

Awesome llm resources, 2026. https://github.com/WangRongsheng/awesome-LLM -resources.

Ibragim Badertdinov, Alexander Golubev, Maksim Nekrashevich, Anton Shevtsov, Simon Karasik, Andrei Andriushchenko, Maria Trofimova, Daria Litvintseva, and Boris Yangel. Swe-rebench: An automated pipeline for task collection and decontaminated evaluation of software engineering agents. Advances in Neural Information Processing Systems, 38, 2026.

Islem Bouzenia and Michael Pradel. Understanding software engineering agents: A study of thought-action-result trajectories. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 2846–2857. IEEE, 2025.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya G. Parameswaran, Dan Klein, Kannan Ramchandran, Matei A. Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025, 2025.

Jingyi Chen, Songqiang Chen, Hengcheng Zhu, Jialun Cao, Jiasi Shen, and Shing-Chi Cheung. Understanding agent-reactive bugs at the model-harness boundary: An empirical study of llm agent issue reports. arXiv preprint arXiv:2607.15684, 2026.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Petar Sirkovic, Artiom Myaskovsky, Grzegorz Glowaty, Felix Weissenberger, Alessio Orlandi, Dan Popovici, Anil Palepu, Keran Rong, Ryutaro Tanno, Khaled Saab, Fan Zhang, Jacob Blum, Andrew Carroll, Kavita Kulkarni, Nenad Tomašev, Dina Zverinski, Ivor Rendulic, Elahe Vedadi, Florian Hasler, Luka Rimanic, Marina Boia, Ivan Budiselic, Ben Feinstein, Mathias Bellaiche, Tom Sheffer, Jan Freyberg, Jeremy Ratcliff, Ottavia Bertolli, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R. D. Costa, José R. Penadés, Gary Peltz, Yossi Matias, James Manyika, Demis Hassabis, Yunhan Xu, Pushmeet Kohli, Annalisa Pawlosky, Alan Karthikesalingam, and Vivek Natarajan. Accelerating scientific discovery with co-scientist. Nature, 655(8122):487–496, May 2026. ISSN 1476-4687. doi: 10.1038/s4 1586-026-10644-y. URL http://dx.doi.org/10.1038/s41586-026-10644-y.

Jianyuan Guo, Zhiwei Hao, Chengcheng Wang, Cheng Fan, Tingzhang Luo, Hongguang Li, Ying Gao, Hefei Mei, Jiankun Peng, Rongjian Xu, Minjing Dong, Han Im Wu, Mengyu Zheng, Kai Han, Shiqi Wang, Chang Xu, and Yunhe Wang. From question answering to task completion: A survey on agent system and harness design. arXiv preprint arXiv:2606.20683, 2026.

Lianghong Guo, Yanlin Wang, Caihua Li, Wei Tao, Pengyu Yang, Jiachi Chen, Haoyu Song, Duyu Tang, and Zibin Zheng. Swe-factory: Your automated factory for issue resolution training data and evaluation benchmarks. arXiv preprint arXiv:2506.10954, 2025.

Mohammed Mehedi Hasan, Hao Li, Emad Fallahzadeh, Gopi Krishnan Rajbahadur, Bram Adams, and Ahmed E Hassan. An empirical study of testing practices in open source ai agent frameworks and agentic applications. Empirical Software Engineering, 31(5):124, 2026.

Ruida Hu, Chao Peng, Junjielong Xu, and Cuiyun Gao. Repo2run: Automated building executable environment for code repository at scale. Advances in Neural Information Processing Systems, 38:32679–32718, 2026.

Naman Jain, Jaskirat Singh, Manish Shetty, Liang Zheng, Koushik Sen, and Ion Stoica. R2e-gym: Procedural environments and hybrid verifiers for scaling open-weights swe agents. arXiv preprint arXiv:2504.07164, 2025.

Shyam Sundar Kannan, Vishnunandan L. N. Venkatesh, and Byung-Cheol Min. SMART-LLM: smart multi-agent robot task planning using large language models. In IEEE/RSJ International Conference on Intelligent Robots and Systems, IROS 2024, Abu Dhabi, United Arab Emirates,

October 14-18, 2024, pp. 12140–12147. IEEE, 2024. doi: 10.1109/IROS58592.2024.10802322. URL https://doi.org/10.1109/IROS58592.2024.10802322.

J. Richard Landis and Gary G. Koch. The measurement of observer agreement for categorical data. Biometrics, 33(1):159–174, 1977. doi: 10.2307/2529310.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Metaharness: End-to-end optimization of model harnesses. arXiv preprint arXiv:2603.28052, 2026.

Junjie Li, Xi Xiao, Yunbei Zhang, Chen Liu, Lin Zhao, Xiaoying Liao, Yingrui Ji, Janet Wang, Yingqiang Ge, Weijie Xu, Xi Fang, Xiang Xu, Tianchen Zhao, Youngeun Kim, Jihun Hamm, Tianyang Wang, and Chandan Reddy. Agent harness engineering: A survey, 2026. URL https: //openreview.net/pdf?id=eONq7FdiHa.

Junwei Liu, Kaixin Wang, Yixuan Chen, Xin Peng, Zhenpeng Chen, Lingming Zhang, and Yiling Lou. Large language model-based agents for software engineering: A survey. CoRR, abs/2409.02977, 2024. doi: 10.48550/ARXIV.2409.02977. URL https://doi.org/ 10.48550/arXiv.2409.02977.

Shuaiqi Liu, Zhengkai Lin, Yuxiang Zhang, Yuanyi Ren, Yue Wu, Yongbin Li, Zheng Wang, Zhihang Fu, and Jieping Ye. The path to recursive self-improving agents: Foundation, framework, and future directions. 2026.

Qianyu Meng, Yanan Wang, Liyi Chen, Wei Wu, Yihang Li, Wenyuan Jiang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, and Yao Hu. Agent harness for large language model agents: A survey. 2026. doi: 10.20944/preprints202604.0428.v3. URL https: //www.preprints.org/manuscript/202604.0428/v3.

Xuying Ning, Katherine Tieu, Dongqi Fu, Tianxin Wei, Zihao Li, Yuanchen Bei, Jiaru Zou, Mengting Ai, Zhining Liu, Ting-Wei Li, et al. Code as agent harness. arXiv preprint arXiv:2605.18747, 2026.

Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, and Yizhe Zhang. Training software engineering agents and verifiers with swe-gym. arXiv preprint arXiv:2412.21139, 2024.

Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, and Yiling Lou. Can agent fix agent issues? In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 49002–49028. Curran Associates, Inc., 2025. doi: 10.52202/085713-1638. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/45f0a b2d264a057113492b35c5ef47f2-Paper-Conference.pdf.

Yuchen Shao, Yuheng Huang, Jiawei Shen, Lei Ma, Ting Su, and Chengcheng Wan. Are llms correctly integrated into software systems? arXiv preprint arXiv:2407.05138, 2024.

David A Tomassi, Naji Dmeiri, Yichen Wang, Antara Bhowmick, Yen-Chuan Liu, Premkumar T Devanbu, Bogdan Vasilescu, and Cindy Rubio-González. Bugswarm: Mining and continuously growing a dataset of reproducible failures and fixes. In 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), pp. 339–349. IEEE, 2019.

Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, Wayne Xin Zhao, Zhewei Wei, and Jirong Wen. A survey on large language model based autonomous agents. Frontiers of Computer Science, 18(6), March 2024a. ISSN 2095-2236. doi: 10.1007/s11704-024-40231-1. URL http://dx.doi.org/10.10 07/s11704-024-40231-1.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024b.

Chunqiu Steven Xia and Lingming Zhang. Automated program repair via conversation: Fixing 162 out of 337 bugs for \$0.42 each using chatgpt. In Maria Christakis and Michael Pradel (eds.), Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, ISSTA 2024, Vienna, Austria, September 16-20, 2024, pp. 819–831. ACM, 2024. doi: 10.1145/3650212.3680323. URL https://doi.org/10.1145/3650212.3680323.

Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, and Lingming Zhang. Agentless: Demystifying llm-based software engineering agents. arXiv preprint arXiv:2407.01489, 2024.

Ying Xiao, Jie Huang, Ruijuan He, Jing Xiao, Mohammad Reza Mousavi, Yepang Liu, Kezhi Li, Zhenpeng Chen, and Jie M. Zhang. Fairmedqa: Benchmarking bias in large language models for medical question answering. CoRR, abs/2505.19562, 2025. doi: 10.48550/ARXIV.2505.19562. URL https://doi.org/10.48550/arXiv.2505.19562.

liyucheng09 etc xlite dev. Awesome-llm-inference: A curated list of awesome llm inference papers with codes, 2024. URL https://github.com/xlite-dev/Awesome-LLM-Inferen ce. Open-source software available at https://github.com/xlite-dev/Awesome-LLM-Inference.

Ziluo Xue, Yanjie Zhao, Shenao Wang, Kai Chen, and Haoyu Wang. A characterization study of bugs in llm agent workflow orchestration frameworks. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 3369–3380. IEEE, 2025.

John Yang, Carlos Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

John Yang, Kilian Lieret, Carlos Jimenez, Alexander Wettig, Kabir Khandpur, Yanzhe Zhang, Binyuan Hui, Ofir Press, Ludwig Schmidt, and Diyi Yang. Swe-smith: Scaling data for soft ware engineering agents. Advances in Neural Information Processing Systems, 38, 2026.

Fanlong Zeng, Wensheng Gan, Yongheng Wang, Ning Liu, and Philip S. Yu. Large language models for robotics: A survey. CoRR, abs/2311.07226, 2023. doi: 10.48550/ARXIV.2311.07226. URL https://doi.org/10.48550/arXiv.2311.07226.

Linghao Zhang, Shilin He, Chaoyun Zhang, Yu Kang, Bowen Li, Chengxing Xie, Junhao Wang, Maoquan Wang, Yufan Huang, Shengyu Fu, et al. Swe-bench goes live! Advances in Neural Information Processing Systems, 38, 2026a.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. Which agent causes task failures and when? on automated failure attribution of LLM multi-agent systems. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net, 2025.

Xiaowen Zhang, Hannuo Zhang, and Shin Hwei Tan. Understanding bugs in modern agentic frameworks: A study of symptoms, root causes, and triggering conditions. arXiv preprint arXiv:2604.08906, 2026b.

Yuntong Zhang, Haifeng Ruan, Zhiyu Fan, and Abhik Roychoudhury. Autocoderover: Autonomous program improvement. In Proceedings of the 33rd ACM SIGSOFT International Symposium on Software Testing and Analysis, pp. 1592–1604, 2024.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: LLM agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 19632–19642, 2024. doi: 10.1609/aaai.v38i17.29936.

Kunlun Zhu, Zijia Liu, Bingxuan Li, Muxin Tian, Yingxuan Yang, Jiaxun Zhang, Pengrui Han, Qipeng Xie, Fuyang Cui, Weijia Zhang, Xiaoteng Ma, Xiaodong Yu, Gowtham Ramesh, Jialian Wu, Zicheng Liu, Pan Lu, James Zou, and Jiaxuan You. Where LLM agents fail and how they can learn from failures. CoRR, abs/2509.25370, 2025.

Xinxue Zhu, Jiacong Wu, Xiaoyu Zhang, Tianlin Li, Yanzhou Mu, Juan Zhai, Chao Shen, Chunrong Fang, and Yang Liu. Bugs in modern llm agent frameworks: An empirical study. In Proceedings of the 34th ACM International Conference on the Foundations of Software Engineering, pp. 1588– 1592, 2026.

## A REPOSITORY SELECTION RULES

<table><tr><td colspan="2">Dual-stream collection</td></tr><tr><td>Stream Rule</td><td></td></tr><tr><td>Static</td><td>Query GitHub for curated awesome-lists (awesome-agent, awesome-llm, awesome-ai-agents, ...) and expand hits with the keyword family.</td></tr><tr><td>Incremental Merge</td><td>Ingest GHArchive hourly files, extract repo . name, apply the same family. Deduplicate the two streams.</td></tr></table>

<table><tr><td>Keyword family</td><td>Examples</td></tr><tr><td>Agent identity</td><td>agent, assistant, chatbot</td></tr><tr><td>LLM / providers</td><td>LLM, OpenAI, Anthropic, model names</td></tr><tr><td>Tools and memory</td><td>tool calling, memory, RAG, vector</td></tr><tr><td>Control</td><td>planner, orchestration, multi-agent</td></tr><tr><td>Prompting</td><td>prompt, ReAct, few-shot</td></tr><tr><td>Frameworks</td><td>CrewAI, AutoGen, LangChain, MetaGPT, AutoGPT</td></tr></table>

We seed from community-curated awesome-lists of agentic systems and expand them with a finite keyword family. The family can later be extended by mining frequent terms from repositories that survive verification.

<table><tr><td>Event</td><td>Counts as active</td></tr><tr><td>PushEvent</td><td>code was pushed</td></tr><tr><td>PullRequestEvent,IssuesEvent</td><td>a PR or issue moved</td></tr><tr><td>WatchEvent,ForkEvent</td><td>a star or fork</td></tr><tr><td>CreateEvent,ReleaseEvent</td><td>a ref or release</td></tr><tr><td>MemberEvent,PublicEvent,DeleteEvent</td><td>membership, visibility, or deletion</td></tr></table>

We ingest GHArchive hourly files. A repository is active if it appears in recent events above (pushes, pull requests, issues, stars, forks, releases). Activity is not equated with code change: a star and a push both indicate that the repository is currently visible, which is the signal we want for emerging systems.

<table><tr><td colspan="2">Filtering gates</td></tr><tr><td colspan="2">Gate Keep if</td></tr><tr><td>Stars ≥50</td><td></td></tr><tr><td>README parseable</td><td></td></tr><tr><td>In-repo tests</td><td>at least one path matches the detector below (issue-agnostic; a hard gate)</td></tr></table>

## B AGENTIC SYSTEM SPECIFICATION

This is the document given to the repository judge.

Definition. An Agentic System is a codebase that implements or contains an LLM-driven agent. Its operational logic is driven by a large language model and exhibits some of autonomous planning, memory management, environmental perception, and tool use. If a repository satisfies some of the components below, it should be considered an Agentic System.

Core components.

<table><tr><td>Component</td><td>What it covers</td></tr><tr><td>LLM“brain”</td><td>task decomposition, scheduling, conversational or action memory</td></tr><tr><td>Perception</td><td>user input, events, files, sensors, or other environment signals</td></tr><tr><td>Action / tooling</td><td>search, shell, databases, third-party APIs, plugins</td></tr><tr><td>Orchestration</td><td>single-agent loop or multi-agent collaboration</td></tr><tr><td>Provider integration</td><td>SDKs, API keys, or model configuration</td></tr><tr><td>Runtime artifacts</td><td>optional: Docker, prompts, tests, example scenarios</td></tr></table>

<table><tr><td colspan="3">Distinguishing features.</td></tr><tr><td>Feature</td><td>Why it differs from conventional software</td><td></td></tr><tr><td>Nondeterminism External dependence</td><td>identical inputs may yield different outputs providers, tools, and resources change frequently</td><td></td></tr><tr><td>Cross-component failures</td><td></td><td>faults span LLM calls, memory, tools, and workflow</td></tr><tr><td>Prompt / context</td><td></td><td>prompt libraries and context length are first-class</td></tr><tr><td>Workflow orientation</td><td></td><td>loops and state checks may hang or run infinitely</td></tr><tr><td></td><td></td><td></td></tr><tr><td colspan="3">Hooks. A repository is likely an Agentic System if it contains:</td></tr><tr><td># Hook</td><td>Typical evidence</td><td></td></tr><tr><td>1</td><td>LLM provider</td><td>openai, anthropic, or a custom wrapper</td></tr><tr><td>2</td><td>Prompts</td><td>prompt/,templates/,prompts/</td></tr><tr><td>3</td><td>Memory</td><td>vector DB, session store, history module</td></tr><tr><td>4</td><td>Tools</td><td>tools/,plugins/,tool_wrappers/</td></tr><tr><td>5</td><td>Control loop</td><td>planner, scheduler, agent loop</td></tr><tr><td>6 7</td><td>Model config</td><td>model name, token limit, API key, context length</td></tr><tr><td></td><td>Language</td><td>README mentions of agent, planner, assistant loop, tool invocation</td></tr><tr><td colspan="3">Checklist. Three or more “yes&quot;: likely. Five or more: highly likely. The judge emits owner/repo and drops documentation collections, tutorials, paper lists, and standalone tool libraries.</td></tr><tr><td>ID Question</td><td colspan="2"></td></tr><tr><td>AS1</td><td colspan="2"></td></tr><tr><td></td><td colspan="2">README or code mentions agent, planner, tool invocation, memory, or LLM?</td></tr><tr><td>AS2</td><td colspan="2">Depends on an LLM-provider SDK?</td></tr><tr><td>AS3</td><td colspan="2">Prompt / template directory, or prompt-management code?</td></tr><tr><td>AS4</td><td colspan="2">Memory / session / vector store?</td></tr><tr><td>AS5</td><td colspan="2"></td></tr><tr><td>AS6</td><td colspan="2">Wrappers or calls to external tools or plugins? Orchestration or an agent loop (plannēr / scheduler)?</td></tr></table>

## C ISSUE SELECTION RULES

<table><tr><td colspan="2">Keep if</td></tr><tr><td>Criterion Closure Description</td><td>Rule state_reason = completed; drop not_planned and null non-empty body after stripping whitespace</td></tr><tr><td>Patch size In-patch tests</td><td>some linked PR is merged into main or master; drop develop / release added + deleted lines ≥ 20 recorded with the same detector as in-repo tests, never required Drop if the patch is exclusively</td></tr><tr><td colspan="2">Patterns</td></tr><tr><td>Kind Documentation</td><td>docs/,README,*.md,*.rst</td></tr><tr><td>Lock / generated</td><td>package-lock.json,yarn.lock,poetry.lock,Pipfile.lock, Cargo.lock,go.sum,*.pb.go,dist/,build/</td></tr><tr><td>Vendored Binary / assets Formatting</td><td>vendor/,third_party/,node_modules/ images, archives, PDF, audio/video, .onnx, .pt, .pth, .h5, .pickle ≥ 20 changed lines of which &lt; 5% have non-whitespace content</td></tr><tr><td>Stored fields</td><td></td></tr><tr><td>Field repo, timestamps</td><td>Role which Agentic System, when crawled</td></tr><tr><td>issue meta</td><td>number, title, url, body, labels</td></tr><tr><td>linked PR</td><td>number, merge flag, base branch</td></tr><tr><td>base_sha</td><td>buggy snapshot (pre-merge mainline)</td></tr><tr><td>head_sha</td><td>patched snapshot (PR tip)</td></tr><tr><td>patch</td><td>unified diff; may be missing</td></tr><tr><td>in-repo tests</td><td>existing_test_paths</td></tr><tr><td>in-patch tests</td><td>test_paths_in_patch;optional</td></tr><tr><td>ai_judgment</td><td>Harness Bug decision and raw response</td></tr></table>

## D HARNESS BUG SPECIFICATION

## This is the document given to the issue judge.

Definition. A Harness Bug is a user-reported problem (bug report or feature request) in an Agentic System that concerns agent-specific execution machinery: LLM-provider integration, tool invocation, memory, LLM operation, workflows, and utilities.

Taxonomy. Six categories and twenty sub-categories. Category F is listed so that utility issues—failures that also arise in conventional software—can be excluded.

<table><tr><td>ID</td><td>Sub-category</td><td>Typical failure</td></tr><tr><td colspan="3">A. Incompatibility with LLM providers</td></tr><tr><td>A.1</td><td>Incompatible dependencies</td><td>missing or misused provider SDKs (e.g., OpenAI, LiteLLM)</td></tr><tr><td>A.2 A.3</td><td>Unsupported models Incompatible parameters</td><td>cannot bind popular models (GPT-4, Claude, DeepSeek, .. .) unexpected or missing provider arguments</td></tr><tr><td colspan="3">B. Tool-related issues</td></tr><tr><td>B.1 B.2 B.3</td><td>Tool dependencies Tool configuration</td><td>missing libraries or binaries needed to register or run tools retriever / embedder / retrieval-mode misconfiguration</td></tr><tr><td>B.4 C. Memory-related issues</td><td>Tool implementation Misused interfaces</td><td>bugs in tool or RAG logic, including helpers bad arguments, serialization, or LLM–tool binding</td></tr><tr><td>C.1 C.2 C.3 Memory</td><td>Initialization Content errors</td><td>DB / workspace reset failures, inconsistent state bad message attributes, serialization, non-primitive types broken internal or external memory-stack modules</td></tr><tr><td colspan="3"></td></tr><tr><td></td><td>dependencies</td><td></td></tr><tr><td>D.1</td><td>D. LLM operation issues</td><td></td></tr><tr><td>D.2 Token usage</td><td>Model access</td><td>wrong binding or missing credentials max tokens, pricing, or token-accounting failures</td></tr><tr><td>D.3</td><td>Output handlers</td><td>empty, malformed, or refusal responses mishandled</td></tr><tr><td>D.4</td><td>Model dependencies</td><td>missing tokenization or provider-client libraries</td></tr><tr><td>D.5</td><td></td><td>overflow or incorrect length accounting</td></tr><tr><td>D.6</td><td>Context length</td><td></td></tr><tr><td></td><td>Prompts</td><td>missing, stale, or poorly managed prompts</td></tr><tr><td>E. Workflow issues</td><td></td><td></td></tr><tr><td>E.1</td><td>Scheduling / loops</td><td>hangs, infinite loops, skipped steps</td></tr><tr><td></td><td></td><td></td></tr><tr><td>F. Utility issues</td><td></td><td></td></tr><tr><td>F.1</td><td>Implementation</td><td>UI, Docker, logging, unrelated imports</td></tr><tr><td>F.2</td><td>Dependencies</td><td>non-agent libraries or internal circular imports</td></tr><tr><td>F.3</td><td>Configuration</td><td>I/O paths, encoding, IPs, telemetry</td></tr></table>

Checklist. Accepted if closed with a developer patch, at least one row is yes, and it is not a utility-only issue.

<table><tr><td>ID</td><td>Question</td><td>If yes</td></tr><tr><td>HB1</td><td>Mentions an LLM provider, model name, SDK, or API key?</td><td>include</td></tr><tr><td>HB2</td><td>Mentions prompt content, templates, or prompt management?</td><td>include</td></tr><tr><td>HB3</td><td>Reports memory symptoms (missing history, corrupt storage, init failures)?</td><td>include</td></tr><tr><td>HB4</td><td>Involves tool invocation, parameters, configuration, or implementation?</td><td>include</td></tr><tr><td>HB5</td><td>Describes workflow anomalies (hangs, loops, repeated actions)?</td><td>include</td></tr><tr><td>HB6</td><td>Patch changes LLM calls, memory, tool wrappers, prompts, or orchestration?</td><td>include</td></tr></table>

## E ONE-TO-ONE ISSUE–PR–COMMIT MATCHING

Issues, PRs, and commits are many-to-many and would otherwise duplicate the same fix. For each PR p we collapse the set of k commits into a squash commit $c _ { p } ^ { s } ,$ , build a bipartite graph $G =$ $( \mathcal { P } , \mathcal { C } ^ { s } , E )$ , and keep only one-to-one edges

$$
{ \mathcal { C } } ( p ) = \bigcup _ { j = 1 } ^ { k } \{ c _ { j } \} \ \longrightarrow \ c _ { p } ^ { s } \qquad E ^ { \star } = \big \{ ( p , c ^ { s } ) \in E \ \big | \ \deg _ { G } ( p ) = \deg _ { G } ( c ^ { s } ) = 1 \big \} .\tag{1}
$$

With issue–PR incidence $E _ { I P } ,$ , each instance is $( i , p , c ^ { s } ) \in \mathcal { D }$ where $( p , c ^ { s } ) \in E ^ { \star }$ and $( i , p ) \in E _ { I P }$ The squash commit is the patched snapshot; the PR base commit is the buggy snapshot.

## F EXAMPLE OF JOINT OPTIMIZATION

AgentScope issue #1297 (https://github.com/agentscope-ai/agentscope/iss ues/1297) reports that a disconnected Studio hook raises a ConnectionError after retrying and crashes the agent. Its developer patch logs the failure and returns, but it contains no IPT. This instance exposes why independent artifact refinement can stall when both the import path and the test oracle are misaligned.

<table><tr><td colspan="2">Observed trace for AgentScope issue #1297</td></tr><tr><td colspan="2"></td></tr><tr><td>Point in pipeline Before co-fix</td><td>Observed evidence All 25 attempts across five epochs to optimizing the test still remained f2 f (i.e., the test fails on both buggy and patched commits). In the final attempt, the</td></tr><tr><td>Full-generation co-fix</td><td>patched run still loaded the hook from site-packages, while the test patched a logger location that did not correspond to the implementation&#x27;s observable behavior. With no IPT to restore, one response changed the container to an editable install with /app/ src first on PYTHONPATH, and rewrote the test to simulate a failed</td></tr><tr><td>Re-verification</td><td>requests.post call and assert that the real agent reply completes after the expected retries. The same command returned exit code 1 on the buggy snapshot and returned exit code 0 after applying the patch, yielding the first f2p (i.e., failed on the buggy commit and passing on the patched commit) outcome for the instance.</td></tr></table>

This trace demonstrates the role of full-generation co-fix after test-only rounds have failed: one final diagnosis can reconcile the container’s source path and the test’s behavioral oracle. It is a representative execution trace, not a component-wise causal ablation; without evaluating the two crossed combinations, we do not claim that each individual edit is independently necessary.

## G FAILURE-MODE ANALYSIS

Figure 7 summarizes the major failure modes observed in unsuccessful reproduction runs of AGENTBUG-SMITH. Moreover, some failure distributions vary across backbone models, suggesting that some failure modes are model-dependent. The results also indicate that further improvements may require targeted enhancements at different stages of the reproduction pipeline, such as more robust dependency resolution, environment construction, and execution-feedback collection. In particular, future efforts could prioritize improving dependency installation and Docker image construction, as well as preserving diagnostic logs when container disk exhaustion occurs.

Table 7: Failure modes of unsuccessful reproduction runs.
<table><tr><td>Failure mode</td><td>DeepSeek-v3.2</td><td>GPT-4.1-mini</td><td>Kimi-k2.5</td></tr><tr><td colspan="4">Single cause: environment</td></tr><tr><td>Dependency installation</td><td>44</td><td>33</td><td>19</td></tr><tr><td>Docker build configuration</td><td>19</td><td>15</td><td>40</td></tr><tr><td>Missing imports or incompatible APIs</td><td>15</td><td>25</td><td>7</td></tr><tr><td>Python packaging metadata</td><td>15</td><td>27</td><td></td></tr><tr><td>Native build or compiler</td><td>12</td><td>28</td><td>45</td></tr><tr><td>Container disk exhaustion</td><td>12</td><td>0</td><td>40</td></tr><tr><td>Missing verifier or Docker logs</td><td>11</td><td>5</td><td>56</td></tr><tr><td colspan="4">Single cause: test</td></tr><tr><td>Oracle or exercised behavior</td><td>17</td><td>30</td><td>4</td></tr><tr><td>Test command crashed or timed out</td><td>4</td><td>9</td><td>1</td></tr><tr><td colspan="4">Coupled cause</td></tr><tr><td>Environment and test together</td><td>13</td><td>8</td><td>3</td></tr><tr><td>Total unsuccessful runs</td><td>162</td><td>180</td><td>179</td></tr></table>

## H STABILITY METRIC

To check stability, we independently classify each retained issue $m { = } 5$ times. Let $A _ { i }$ be the set of issues labeled Yes in run $i ,$ and let $p _ { j }$ be the fraction of Yes answers for issue j. We report set agreement and normalised entropy

$$
S _ { \mathrm { s e t } } = { \frac { \left| \bigcap _ { i = 1 } ^ { m } A _ { i } \right| } { \left| \bigcup _ { i = 1 } ^ { m } A _ { i } \right| } } \qquad S _ { \mathrm { e n t r o p y } } = 1 - { \frac { 1 } { n } } \sum _ { j = 1 } ^ { n } H ( p _ { j } ) ,\tag{2}
$$

where $H ( p ) = - p \log _ { 2 } p - ( 1 - p ) \log _ { 2 } ( 1 - p )$ is binary entropy. Both scores are 1 when every run returns the same label for every issue.

## I SKILL DISTILLATION PROCESS

This section presents the concrete process of skill distillation, which basically follow the common pipelines design in previous work (Zhao et al., 2024; Wang et al., 2024b). The skill π is a humanreadable SKILL.md document paired with a set of lightweight pattern scanners. It is neither a fine-tuned model but a plain-text instruction that the agent reads as part of its prompt. The pipeline refines π iteratively over the training set through the following steps.

1. Initialization. The pipeline begins with an empty skill, and the agent attempts each training issue $i \in \mathcal { T } _ { \mathrm { t r } }$ with no guidance beyond its default prompt.

2. Failure diagnosis. When the agent fails on a training issue, the pipeline diagnoses whether the failure matches a recognizable bug shape. For example, an overly narrow string predicate, an unhandled empty payload at an API boundary, a missing fallback branch, or a dropped keyword argument at a call site.

3. Generic scanner proposal. If a failure matches a known shape, the skill is updated by adding or tightening a generic scanner: a short instruction that tells the agent what syntactic or semantic pattern to look for and what minimal edit to apply. Crucially, proposed updates are constrained to remain within thefeasible set:

$\Pi _ { \mathrm { s a f e } } = \{ \pi :$ : π contains no repository name, file path, issue ID, or gold-patch text	, (3) so that the skill cannot memorize any specific codebase. Any proposed update that references the current repository is automatically rejected:

$$
\pi _ { t + 1 } = { \left\{ \begin{array} { l l } { \pi ^ { \prime } } & { { \mathrm { i f ~ } } \pi ^ { \prime } \in \Pi _ { \mathrm { s a f e } } , } \\ { \pi _ { t } } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{4}
$$

4. Freezing. After one pass through all training issues, the skill π is frozen. No further updates are made before evaluation on the held-out test set.

## J DETAILED ANALYSIS OF FIXING EFFECTIVENESS

Figure 7 compares the fixing rate of studied software agents across the bugs of different harness components. Overall, tool related bugs are the one with highest fixing rate for all studied agents. Figure8 shows the unique and overlap of correctly resolved issues across the studied software agents on LIVE-HARNESS-BENCH. All the studied agents collectively resolved a total of 24 unique issues, with mini-SWE-agent and OpenHands demonstrating the highest overall efficacy by participating in 18 and 17 total resolutions respectively. mini-SWE-agent achieved the strongest independent problem-solving capacity with 7 total resolutions.

![](images/45d020583162f9e473b2fb7fb7abb9a3052f5d25e147a8d3854a244504fade55.jpg)  
Figure 7: Effectiveness of Agents on LIVE-HARNESS-BENCH.

![](images/2e226855a671dc598c8de457af681d11115ae85382d8899e2fef36d2a840577d.jpg)  
Figure 8: Venn Diagram of Resolved Bugs.

## K CASE: HARNESS-SDK #362

This is one of the five new held-out fail-to-pass successes. C5 was not specialized to this repository.

Issue   
strands-agents/harness-sdk#362 reports that system\_prompt is not passed into   
structured\_output. Agent.structured\_output\_async already stores   
self.system\_prompt, then calls the model with only the schema and the messages, so a   
fail-to-pass spy still sees system\_prompt=None.   
Without the skill   
src/strands/agent/agent.py · structured\_output\_async()   
events = self.model.structured\_output(output\_model, self.messages)   
Function-level localization misses this call. Typical attempts edit Bedrock caching, rewrite a   
formatter, or emit no patch.   
Generated patch   
src/strands/agent/agent.py:460 · C5, plumb a missing keyword   
events = self.model.structured\_output(output\_model,   
self.messages)   
+ events = self.model.structured\_output(   
+ output\_model, self.messages, system\_prompt=self.system\_prompt)

```python
Golden patch
harness-sdk#466 · structured_output_async()
events = self.model.structured_output(output_model,
self.messages)
+ events = self.model.structured_output(
+ output_model, self.messages, system_prompt=self.system_prompt)
```

The generated call-site edit is identical to gold. The original fail-to-pass test turns green, the AI judge marks the patch correct, and function-level localization flips from miss to hit. The other four new successes have the same chain: capability → blamed site → one-line legal edit → F2P that matches gold.