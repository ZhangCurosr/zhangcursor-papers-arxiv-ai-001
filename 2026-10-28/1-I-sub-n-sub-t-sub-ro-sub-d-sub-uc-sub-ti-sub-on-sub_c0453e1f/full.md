# Benchmarkin<sub>g</sub> LLM A<sub>g</sub>ents for Behaviorall<sub>y</sub> Ade<sub>q</sub>uate Test G<sub>enera</sub>ti<sub>on</sub> f<sub>rom</sub> S<sub>o</sub>ft<sub>ware</sub> R<sub>equ</sub>i<sub>remen</sub>t<sub>s</sub>

<sup>BINHANG</sup> Q<sup>I</sup>, National University of Singapore, Singapore

<sup>YIMING</sup> <sup>LIU</sup>, Shanghai Jiao Tong University, China and Shanghai Innovation Institute, China

<sup>YUN</sup> <sup>LIN∗</sup>, Shanghai Jiao Tong University, China

<sup>YANG</sup>J<sup>IE</sup> <sup>ZHOU</sup>, National University of Singapore, Singapore

<sup>HAILONG</sup> <sup>SUN</sup>, Beihang University, China

J<sup>IN</sup> <sup>SONG</sup> <sup>DONG</sup>, National University of Singapore, Singapore

As large language model (LLM) agents increasingly compile natural-language requirements into repositorylevel code, requirement-to-test generation grows in importance to provide validation constraints (or harness) to guide code generation. Yet it remains unclear whether current agents can generate tests that validate the required behavior when the implementation of that behavior is withheld. Existing test-generation benchmarks mostly focus on white-box testing, whereas the requirement–test-related study usually requires evaluation on semantic alignment between the given documents and the generated tests.

We introduce Req2Test-Bench, a benchmark containing 5,086 execution-validated instances recovered from developer-authored behavior-driven development (BDD) artifacts in 8 open-source projects, spanning 5 programming languages and 4 BDD frameworks. Each instance pairs a natural-language requirement in Gherkin with its developer-written Test Implementation. Given the requirement, target interfaces of the required behavior, and a repository view that withholds either the target implementation or all production implementations, an agent must generate a complete, potentially multi-file Test Implementation. Solving the task requires bridging the gap between a high-level requirement and its test-code correspondence regarding projectspecific knowledge: crystallizing what to test, stated in natural language, into how to test it with the project’s own fixtures, helpers, and interfaces. We evaluate the semantic diference between generated and ground-truth tests by comparing their production API calls, code coverage, and mutant kills. Evaluation of 8 agent–model– reasoning configurations reveals a substantial gap between execution success and behavioral adequacy, which is measured strictly as kill-completeness: a kill-complete test kills every mutant that the ground-truth test kills. The best-performing configuration, Codex with DeepSeek-V4.1-Flash, produces passing tests for 82.8% of instances, yet only 33.8% are behaviorally adequate. We find that (1) scaling up agents yields sharply diminishing returns in behavioral adequacy relative to generation cost; (2) under-specified requirements cause the information gap, including ambiguous setup and implicit expected outcomes; and (3) requirement-grounded harnesses that steer agent non-determinism ofer a promising direction for improvement.

## 1 I<sub>n</sub>t<sub>ro</sub>d<sub>uc</sub>ti<sub>on</sub>

Large language model (LLM) agents increasingly translate natural-language software requirements into repository-wide implementations [38]. As code generation accelerates, validation becomes a bottleneck: developers need executable evidence that generated software satisfies the requested behavior rather than merely compiling or appearing plausible. Requirement-derived tests provide behavioral checks for code-generation harnesses, supporting candidate selection and iterative refinement [6, 16]. Yet a generated test may pass while omitting meaningful checks or relying on tautological assertions [18, 33]. The challenge is sharper when tests must be written before an implementation of the required behavior is available [23]. We thus ask: can current LLM agents generate complete, executable, and behaviorally adequate tests from software requirements when the target implementation is withheld? We examine it through three questions:

![](images/8202942c533da26d4bc47491343762ebc3f00b848b0a04ed81d4b4f30780e405.jpg)  
Fig. 1. Overview of the Req2Test-Bench task. Each instance provides a requirement composed of enclosing Feature context and a target Scenario, together with designated target interfaces and project context exposed through either a target-hidden or an all-hidden view. Both views withhold the target implementation. Targethidd<sub>en re</sub>t<sub>a</sub>i<sub>ns o</sub>th<sub>er perm</sub>it<sub>e</sub>d <sub>pro</sub>d<sub>uc</sub>ti<sub>on</sub> i<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>ons, w</sub>h<sub>ereas a</sub>ll<sub>-</sub>hidd<sub>en w</sub>ithh<sub>o</sub>ld<sub>s</sub> th<sub>em.</sub> A<sub>n</sub> LLM agent generates a complete, potentially multi-file Test Implementation. After generation, an evaluator runs the test a<sub>g</sub>ainst the com<sub>p</sub>lete <sub>p</sub>roduction im<sub>p</sub>lementation<sub>,</sub> recordin<sub>g</sub> sta<sub>g</sub>ed functional outcomes and com<sub>p</sub>arin<sub>g</sub> it<sub>s pro</sub>d<sub>uc</sub>ti<sub>on-</sub>API <sub>ca</sub>ll<sub>s, co</sub>d<sub>e coverage, an</sub>d <sub>mu</sub>t<sub>an</sub>t kill<sub>s w</sub>ith th<sub>ose o</sub>f th<sub>e groun</sub>d<sub>-</sub>t<sub>ru</sub>th T<sub>es</sub>t I<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on.</sub>

• Q1: Does scaling up LLM agents reliably and proportionally improve behavioral adequacy?

• Q2: If not, what gaps limit the progress?

• Q3: What directions are promising to narrow the gap, guiding the development of LLM agents? In this work, we define the requirement and the behavioral adequacy as follows:

Definition 1. (Requirement) A requirement is a natural-language statement of one software behavior expected from a software system, stated independently of the code that realizes it. It is concrete enoug<sup>h</sup> to <sup>d</sup>etermine a testing o<sup>b</sup>jective, typica<sup>ll</sup>y <sup>b</sup>y stating t<sup>h</sup>e re<sup>l</sup>evant con<sup>d</sup>itions, t<sup>h</sup>e triggering i<sub>n</sub>t<sub>erac</sub>ti<sub>on,</sub> <sub>an</sub>d th<sub>e</sub> <sub>expec</sub>t<sub>e</sub>d <sub>o</sub>b<sub>serva</sub>bl<sub>e</sub> <sub>ou</sub>t<sub>come.</sub>

Definition 2 (Behavioral Adequacy). A test is behaviorally adequate for a requirement if it exercises t<sup>h</sup>e <sup>b</sup>e<sup>h</sup>avior prescri<sup>b</sup>e<sup>d</sup> <sup>b</sup>y t<sup>h</sup>at requirement an<sup>d</sup> correct<sup>l</sup>y c<sup>h</sup>ec<sup>k</sup>s t<sup>h</sup>e expecte<sup>d</sup> outcomes in t<sup>h</sup>e project<sup>’</sup>s context.

Existing LLM-based test-generation studies and benchmarks start from prescribed methods, classes, or source files [18, 36, 41], from code changes [15], from issue reports [1, 26], or from behavior inferred by exploring a running application [3]. Consequently, their results alone do not sufice to answer these questions.

We instead study requirement-to-test generation, which is requirement-driven: an independently stated requirement determines the testing objective, and the target implementation is withheld from the agent. Related requirement–test resources address parts of this problem; however, they have complementary limitations: 1 Some synthesize or manually adapt requirement or test artifacts during benchmark construction [24, 25]; 2 others associate a coarse requirement artifact, such as a project-level requirement, with an entire test file rather than with the complete Test Implementation for one individual, concrete behavior [21, 24].

To address these limitations, Req2Test-Bench combines authentic requirements, complete ground-truth Test Implementations for individual behaviors, execution-validated mappings, and behavioral evaluation under controlled implementation visibility. We instantiate the task using artifacts from behavior-driven-development (BDD) repositories. BDD projects express requirements in Gherkin, where a Feature describes a capability and groups related Scenarios, while a Scenario expresses a concrete behavioral example through an ordered sequence of Steps [35], as shown in Figure 1. Their availability across open-source projects and programming languages, without restriction to a particular test level, supports large-scale and diverse benchmark construction [21].

Accordingly, for each Req2Test-Bench instance, we operationally define one requirement $r _ { i }$ as the pair of an enclosing Feature $F _ { i }$ and a target Scenario $S _ { i }$ . We formalize the requirement-to-test generation task as:

$$
T _ { i } ^ { g e n } = { \mathcal { A } } { \left( r _ { i } , \ Z _ { i } ^ { \alpha } \right) } , \qquad r _ { i } \triangleq { \langle F _ { i } , S _ { i } \rangle } , \qquad { \mathit { T } } _ { i } ^ { \alpha } \triangleq { \bigl \langle } { \mathit { T } } _ { i } ^ { \mathrm { t a r } } , C _ { \alpha } ( P _ { i } ) { \bigr \rangle } .\tag{1}
$$

The agent-visible project information ${ \mathcal { I } } _ { i } ^ { \alpha }$ combines the designated target interfaces ${ \cal T } _ { i } ^ { \mathrm { t a r } }$ with available project context $C _ { \alpha } ( P _ { i } ) . \ J _ { i } ^ { \mathrm { t a r } }$ provides method declarations derived from first-party public production calls made directly by the ground-truth test’s action and check Steps. They provide interface-localization cues, but do not enumerate the auxiliary APIs, fixtures, or helpers needed to construct the test. $C _ { \alpha } ( P _ { i } )$ of project snapshot $P _ { i }$ is exposed through one of two views, with $\alpha \in \{ \mathrm { T H } , \mathrm { A H } \}$ : 1 In the target-hidden view $C _ { \mathrm { T H } } ( P _ { i } )$ , the selected production bodies realizing the target behavior, including relevant first-party delegation, are replaced with declaration-preserving placeholders. $\bullet$ In the all-hidden view $C _ { \mathrm { A H } } ( P _ { i } )$ , implementation bodies throughout the project’s declared production-code scope are likewise replaced. The output $T _ { i } ^ { g e n }$ generated by an LLM agent $\mathcal { A }$ is one complete, potentially multi-file Test Implementation containing the test-side artifacts needed to build, bind, execute, and check $S _ { i } .$ . After generation, $T _ { i } ^ { g e n }$ from either view is evaluated by executing it against the same complete production implementation in a separate environment. As illustrated by Figure 1, $r _ { i }$ pairs the Feature “Adding a new Currency ...” with the Scenario “Site should be using currency $\mathbf { \bar { ' } U S D     } _ { \mathbf { --- } } ^ { \mathbf { 3 } } . \mathbf { \bar { \Gamma } } _ { i } ^ { \mathrm { t a r } }$ includes StoreContextInterface::getStore() and CurrencyInterface::getId(), and $T _ { i } ^ { g e { \dot { n } } }$ spans three PHP files.

To construct Req2Test-Bench, we recover the developer-written Test Implementation of each Scenario from the Step Definitions that the native BDD framework binds to its Steps at runtime, validate each recovered mapping by execution, and freeze a per-instance mutant pool with the kills of the ground-truth test. Req2Test-Bench’s validated mapping dataset contains 5,086 Requirement– Test Implementation instances from 1,415 Features in 8 projects, spanning 5 languages, 6 domains, 4 BDD frameworks, and component-, integration-, and system-level tests.

We evaluate 8 agent–model–reasoning configurations that vary the coding agent (Aider, Open-Hands, Claude Code, and Codex CLI), the LLM (DeepSeek-V4.1-Flash, GPT-6 Luna, GPT-6 Sol, and Claude Sonnet 5.5), and the reasoning efort. Beyond execution success (Pass@1), we assess behavioral adequacy with the developers’ own test as the reference for the required behavior. Our primary metric, KillComplete@1, is the share of instances whose generated test both passes and kills every mutant that its ground-truth test kills, i.e., is kill-complete.

The experimental results show that current agents produce passing tests far more often than behaviorally adequate ones. The best-passing configuration, Codex CLI with DeepSeek-V4.1-Flash at high reasoning efort, passes 82.8% of instances but is kill-complete on only 33.8%. Three findings emerge: 1 Returns diminish steeply: The returns in behavioral adequacy obtained by scaling up agents (e.g., increasing LLM scale) diminish with generation cost. For Codex CLI, moving from GPT-6 Luna to DeepSeek-V4.1-Flash and then to GPT-6 Sol increases the generation cost by factors of 6.0 and 4.9, yet changes KillComplete@1 by only +5.5 percentage points (pp) and then −2.5 pp $( 2 8 . 3 \%  3 3 . 8 \%  3 1 . 3 \% )$ . 2 Information gap between high-level requirements and concrete tests: 90–99% of passing but kill-incomplete tests fall short where the requirement leaves the setup ambiguous or the expected outcome tacit. 3 A requirement-grounded harness is required for reducing uncertainty in test generation: The best of three generations is kill-complete 8.6–10.2 pp more often than a single one, headroom for harnesses that exploit the non-determinism of LLM agents and can discriminate non-trivial from trivial diferences between tests.

T<sub>a</sub>bl<sub>e</sub> 1<sub>.</sub> C<sub>ompar</sub>i<sub>son o</sub>f <sub>resources pa</sub>i<sub>r</sub>i<sub>ng requ</sub>i<sub>remen</sub>t<sub>-</sub>lik<sub>e</sub> i<sub>npu</sub>t<sub>s w</sub>ith <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th t<sub>es</sub>t <sub>co</sub>d<sub>e.</sub> “S<sub>ource</sub>” indicates w<sup>h</sup>et<sup>h</sup>er an arti<sup>f</sup>act was preserved <sup>f</sup>rom t<sup>h</sup>e subject project. <sup>“</sup>Hierarc<sup>h</sup>y<sup>”</sup> indicates test <sup>l</sup>eve<sup>l</sup>s and ex<sub>p</sub>licit re<sub>q</sub>uirement relations.
<table><tr><td rowspan="2">Resource</td><td rowspan="2">Source</td><td rowspan="2">Granularity</td><td rowspan="2">Evaluation Criteria</td><td rowspan="2">Hierarchy</td><td colspan="2">Size</td></tr><tr><td>#Pairs</td><td>#PLs</td></tr><tr><td>E2EDev [25]</td><td>Synthetic</td><td>Scenario ↔ Test Impl.</td><td>N/A</td><td>N/A</td><td>703</td><td>1</td></tr><tr><td>GivenWhenThen [21]</td><td>Real</td><td>Feature ↔ Test Files</td><td>N/A</td><td>N/A</td><td>2,289</td><td>3</td></tr><tr><td>AutoGlue [34]</td><td>Real</td><td>Step ↔ Method</td><td>Ref.-based</td><td>N/A</td><td>1,307</td><td>1</td></tr><tr><td>DevBench [24]</td><td>Manual</td><td>Project ↔ Test Files</td><td>Prog.-based</td><td>N/A</td><td>22</td><td>4</td></tr><tr><td>REQ2TEST-BENCH</td><td>Real</td><td>Scenario ↔ Test Impl.</td><td>Prog. &amp; Ref.-based</td><td>√</td><td>5,086</td><td>5</td></tr></table>

We make the following contributions:

• We formulate repository-level requirement-to-test generation as a requirement-driven, interfaceassisted task in which an LLM agent must synthesize one complete, potentially multi-file Test Implementation while the target implementation is withheld.

• We construct and release Req2Test-Bench, comprising 5,086 authentic, execution-validated Requirement–Test Implementation instances from 8 projects in 5 programming languages, and measure behavioral adequacy relative to the developers’ own tests.

• We evaluate eight agent–model–reasoning configurations that difer in the agent scafold, the LLM, or the reasoning efort, uncover sharply diminishing returns and a persistent information gap, and identify promising directions for future research.

The code, data, and leaderboard are available in our repository [32].

## 2 R<sub>e</sub>l<sub>a</sub>t<sub>e</sub>d W<sub>or</sub>k

Test-generation benchmarks. Existing test-generation benchmarks usually derive the test objective from prescribed code, a defect context, or an observed application. Methods2Test [36] mines focal-method–JUnit-test pairs, while TestBench [41] and TestGenEval [18] evaluate classor file-level test generation using execution, coverage, and fault-detection or mutation metrics. SWT-Bench [26] asks for a test that reproduces an issue report, and AutoE2E [3] infers features by exploring a running web application before generating end-to-end tests. These works provide important execution-based protocols, but their task boundary is supplied by focal code, a codechange or fault context, or an observed application. Req2Test-Bench studies the complementary requirement-defined setting, in which a developer-authored Requirement determines the target behavior and an agent must realize it as a complete, project-compatible Test Implementation.

Re uirement–test resources. Table 1 compares the closest works that associate requirement artifacts with test code. E2EDev [25] provides LLM-generated requirements, BDD Scenarios, and Step Definitions for web applications. GivenWhenThen [21] mines developer-authored Feature and Step Definition files at scale, but its scenario records contain file-level artifacts rather than an execution-validated, Scenario-specific Test Implementation. DevBench [24] includes benchmark authored PRDs and executable acceptance- and unit-testing tasks with mixed-provenance tests, but does not map individual requirement criteria to individual ground-truth tests. Req2Test-Bench instead recovers developer-authored Feature–Scenario artifacts and their complete ground-truth Test Implementations from native BDD runtime bindings, validates each mapping through positive

![](images/c2d783878cb616117162b125894085c76df865d17410abb09662f34d1f73bfe2.jpg)

Fi<sub>g.</sub> 2<sub>.</sub> BDD-<sub>g</sub>r<sub>ou</sub>nd<sub>e</sub>d m<sub>e</sub>t<sub>a</sub>-m<sub>o</sub>d<sub>e</sub>l <sub>o</sub>f Re<sub>q</sub>2Test-Bench<sub>.</sub> A R<sub>equ</sub>ir<sub>e</sub>m<sub>e</sub>nt <sub>pa</sub>ir<sub>s a</sub>n <sub>e</sub>n<sub>c</sub>l<sub>os</sub>in<sub>g</sub> F<sub>ea</sub>t<sub>u</sub>r<sub>e</sub> with a tar<sub>g</sub>et Scenario. A Feature is illustrated b<sub>y</sub> Scenarios. Each Scenario contains an ordered se<sub>q</sub>uence of Ste<sub>p</sub>s matched to executable Ste<sub>p</sub> Definitions and is automated b<sub>y</sub> one com<sub>p</sub>lete<sub>,</sub> <sub>p</sub>otentiall<sub>y</sub> multi-file Test Im<sub>p</sub>lementation. A Test Im<sub>p</sub>lementation contains its matched Ste<sub>p</sub> Definitions and their hel<sub>p</sub>er code. De<sub>p</sub>endenc<sub>y</sub> and variant relations ma<sub>y</sub> connect Features or Scenarios.

and negative executions, and uses the mapping for ground-truth-relative API, coverage, and mutation comparisons. Thus, Req2Test-Bench contributes the execution-validated, Scenariolevel Requirement–Test Implementation mapping and behavior-aware evaluation needed to study complete Scenario realization, complementing prior work on Gherkin-to-code generation.

Requirements-based and BDD test generation. Requirements-based testing has long derived tests from structured or natural-language specifications. UMTG [37] derives executable acceptance tests from use cases and a domain model, while Specmate [12] and CiRA [11] derive test designs from acceptance criteria or conditional relations. Recent LLM-based systems generate executable artifacts from requirements or intermediate behavioral specifications. AutoGlue [34] generates one Java glue method for a target Gherkin Step using its Feature and Scenario context together with retrieved project artifacts. Ferreira et al. [10] generate Gherkin Scenarios and Cypress tests from user stories and application HTML, whereas AToMIC [13] generates Gherkin Scenarios, Page Objects, and mobile UI tests from JIRA requirements and code changes. Other systems use DSL requirements, validation intentions, or product-requirement knowledge to guide testing around prescribed architectures or focal code [14, 27, 31, 39]. These studies establish the value of requirements and project knowledge, but difer in generation unit and task boundary. Among these, AutoGlue is the closest task-level neighbor: it maps one Step to one glue method, whereas Req2Test-Bench evaluates the complete, potentially multi-file Test Implementation required to automate one Scenario. Other systems target specialized UI-automation pipelines or guide test generation around prescribed architectures or focal implementations.

## 3 Re<sub>q</sub>2Test-Bench

Req2Test-Bench recovers execution-validated pairs of Requirements and Test Implementations from developer-authored BDD artifacts. This section defines the artifact meta-model, dataset construction, and evaluation criteria.

## 3<sub>.</sub>1 Re<sub>q</sub>uirement-to-Test Artifact Meta-Mode

Each instance � supplies the task inputs of Equation 1, namely the Requirement $r _ { i } = \langle F _ { i } , S _ { i } \rangle$ , the target interfaces $\bar { \varGamma } _ { i } ^ { \mathrm { t a r } }$ , and both views of the project snapshot $P _ { i }$ . It also holds evaluator-only data: the developer-authored ground-truth Test Implementation $T _ { i } ^ { \mathrm { g t } }$ and its API calls, coverage, and mutant kills, against which a generated Test Implementation $\dot { T } _ { i } ^ { \mathrm { g e n } }$ is compared.

Entity. Figure 2 shows the BDD-grounded meta-model: 1 A Feature provides a developer-authored capability description and groups related Scenarios. $\textcircled { \pmb { \theta } }$ Each Scenario expresses a concrete behavioral example through ordered Steps. 3 Steps Given, When, and Then conventionally express context, an action or event, and an expected outcome. An And continues the preceding role [35]. Each efective Step in a retained instance binds uniquely to an executable Step Definition. 4 A Test Implementation is the project-local test-side closure needed to bind, execute, and check the target Scenario: its matched Step Definitions and the helper code that they require. Each Test Implementation has a test level, namely component, integration, or system testing.

Relation. The meta-model defines five relation types. 1 illustrated\_by links a Feature to its Scenarios; 2 contains links a Scenario to its ordered Steps and a Test Implementation to its constituent test-side artifacts; 3 matched\_by links a Step to its selected Step Definition. Multiple Steps may use the same Step Definition, and Test Implementations may overlap. 4 The optional, directed depends\_on relation records prerequisite or decomposition relations between Features or between Scenarios. 5 The symmetric variant\_of relation connects artifacts expressing the same core intent under diferent inputs, preconditions, configurations, or expected outcomes.

## 3<sub>.</sub>2 Dataset Construction

We construct Req2Test-Bench in five stages. Stage I assembles candidate repositories and their BDD artifacts. Stage II selects projects for execution and filters their Scenarios. Stage III recovers and validates Requirement–Test Implementation mappings, identifies target interfaces, and constructs the agent views. Stage IV constructs the mutation evidence and establishes mutation scorability. Stage V labels test levels and requirement relations.

Stage I: Candidate discovery and artifact collection. We assembled candidate GitHub repositories containing developer-authored Gherkin .feature files and BDD test code, using framework dependencies, configuration files, and Step Definitions as static evidence. Four execution frameworks are Cucumber (Java and Ruby), Behave (Python), Behat (PHP), and Godog (Go). An initial survey of the 100 highest-starred discovered repositories with more than 100 recorded stars identified potentially usable Scenarios with static Step Definition matches in 46 repositories. Further discovery added candidates beyond this survey. For candidates taken forward, we pinned revisions and collected feature files, Step Definitions, framework configuration, hooks, fixtures, test data, and helpers. This stage yields candidate repositories and their artifacts for execution checks in Stage II.

Stage II: Project admission and Requirement filtering. We favored higher recorded GitHub star counts, more recent last-push dates, and larger BDD suites, measured by their numbers of feature files and Scenarios. These preferences were balanced against domain and testing diversity. The eight selected projects span five programming languages and cover six domains: e-commerce (CoreShop, PrestaShop, and Sylius), document processing (python-docx), a blockchain SDK (Algorand SDK), IoT services (Kapua), an online petitions service (e-petitions), and agent software (coddy-agent). We sought both direct library tests and multistep workflows involving shared fixtures, database state, and service/API interactions.

We removed empty, comment-only, and unparsable feature files and Scenarios that need a browser or a live service, and retained Features with nonempty titles and Scenarios with nonempty Step sequences. Applicable Background Steps are included in execution order, forming the Scenario’s efective Steps. A Scenario Outline remains one instance: its required Examples rows are executed after parameter substitution and combined into one instance-level outcome. We provisioned each project at its pinned revision and invoked its native BDD runner. A Scenario was retained only if the project initialized, every efective Step bound uniquely to a non-pending Step Definition, and the Scenario passed in isolation across three runs. We excluded build or initialization failures, undefined or ambiguous Steps, timeouts, and unstable outcomes without rewriting project-authored requirements or tests to make them pass. This execution gate follows repository-scale benchmark construction practice [20, 25].

Sta<sub>g</sub>e III: Test-Im<sub>p</sub>lementation recover<sub>y,</sub> tar<sub>g</sub>et-interface identification<sub>,</sub> and a<sub>g</sub>ent views. Test Implementation Recovery. For each retained Scenario, we recorded the Step Definition selected by the runner for every efective Step. Starting from these runtime bindings, we recovered the transitive project-local test-side closure, including helpers, but excluding production code, third-party framework code, and the testing infrastructure that the whole suite shares. Shared definitions and supporting artifacts can belong to more than one closure. We validated each candidate closure in two derived workspaces sharing the same production revision, execution command, configuration, and project-global BDD infrastructure. Among project-local test-side artifacts, the positive workspace retains only the recovered closure and invariant infrastructure, and the target Scenario must pass. In the negative workspace, which withholds the closure, the runner must still initialize, but the target Scenario must not pass. Both outcomes must hold across three runs. Only validated closures become ground-truth Test Implementations $T _ { i } ^ { \mathrm { g t } }$

Target Interface Identification. We identify target interfaces by Step-attributed direct-call tracing of ground-truth tests. We select Steps When and Then, excluding Background and other context Steps. We retain public production methods called directly from test-side code during these Steps, excluding transitive production calls and constructors invoked only from check Steps. Method declarations are deduplicated and grouped by declaring type to obtain ${ \cal T } _ { i } ^ { \mathrm { t a r } }$

This restriction reduces setup-related disclosure while retaining concrete interaction and observation points. Only declarations are supplied, not ground-truth call arguments, invocation order, assertions, or test code. The result is an observed testing boundary, not an exhaustive semantic localization. Instances with an empty set are retained, in total 234 instances. Their action and check Steps call no production method directly, e.g., an e-petitions test acts through an in-process HTTP client, and excluding them would remove 85 of the 98 e-petitions instances.

Agent Views. For target-hidden, a source-level mapping uses the Requirement, ground-truth bindings, and pinned code and configuration to identify the bodies realizing the target behavior, including relevant delegation but not unrelated supporting infrastructure. The interface list alone is not the body-removal list. Language-aware transformations replace mapped bodies while preserving declarations, and all-hidden applies them throughout the declared production scope.

We then materialize $C _ { \mathrm { T H } } ( P _ { i } )$ and $C _ { \mathrm { A H } } ( P _ { i } )$ with the same Requirement, interface annotations, permitted contract information, and retained infrastructure. Both views remove the Step Definitions and helpers of all Scenarios. Both keep the testing infrastructure that the whole suite shares, such as hooks, fixtures, and test data. Neither shows runtime traces, coverage, or mutation records.

Stage IV: Mutation data and scorability. We implement established mutation-operator families as language-specific source transformations [17, 19, 30]. We keep mutant generation separate from test execution, so that the ground-truth and generated tests can run against the same recorded mutants using each project’s native BDD runner. Operator families include conditional-boundary changes, negation, arithmetic and logical-operator replacement, constant and return-value replacement, and method-call removal. The supported operators and language-specific adaptations are recorded per project rather than assumed identical across languages. Mutants are restricted to the declared production scope, excluding test scafolding, generated, third-party, and other out-of-scope code. The versioned manifest records each mutant’s source location, change, and stable identity.

Let $M _ { P _ { i } }$ denote the selected production-mutant manifest and $C _ { i } ^ { \mathrm { g t } }$ the production file–line locations covered by $T _ { i } ^ { \mathrm { g t } }$ on the unmodified snapshot. For the source locations $\mathcal { L } ( m )$ assigned to mutant � by the project’s fixed coverage mapping, the instance pool is

$$
U _ { i } ^ { \mathrm { m u t } } = \left\{ m \in M _ { P _ { i } } \ | \ \mathcal { L } ( m ) \cap C _ { i } ^ { \mathrm { g t } } \neq \emptyset \right\} .\tag{2}
$$

Thus, pool membership depends on the ground-truth test’s coverage.

T<sub>a</sub>bl<sub>e</sub> 2<sub>.</sub> St<sub>a</sub>ti<sub>s</sub>ti<sub>cs o</sub>f Re<sub>q</sub>2Test<sub>-</sub>Bench<sub>.</sub> L<sub>e</sub>ft<sub>:</sub> t<sub>o</sub>t<sub>a</sub>l<sub>s over a</sub>ll i<sub>ns</sub>t<sub>ances.</sub> Ri<sub>g</sub>ht<sub>: m</sub>i<sub>n</sub>i<sub>mum, max</sub>i<sub>mum, an</sub>d <sub>mean</sub> <sub>p</sub>er instance. C / I / S: com<sub>p</sub>onent<sub>,</sub> inte<sub>g</sub>ration<sub>,</sub> and s<sub>y</sub>stem level. A <sub>g</sub>round-truth kill is an instance–mutant <sub>pa</sub>i<sub>r</sub> <sub>w</sub>h<sub>ose</sub> <sub>mu</sub>t<sub>an</sub>t th<sub>e</sub> <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th t<sub>es</sub>t kill<sub>s.</sub>
<table><tr><td>Statistic Total</td></tr><tr><td>Benchmark</td></tr><tr><td>Projects 8 Programming languages 5</td></tr><tr><td>BDD frameworks 4</td></tr><tr><td>Instances 5,086 Features 1,415</td></tr><tr><td>Multi-file Test Impl. 1,741</td></tr><tr><td>Test levels (C / I / S) 588 / 2,066 / 2,432 depends_on pairs 634</td></tr><tr><td>variant_of pairs 3,584</td></tr><tr><td>Mutation operators 11 Ground-truth kills 247,585</td></tr></table>

<table><tr><td>Statistic</td><td>Min</td><td>Max</td><td>Avg.</td></tr><tr><td>Feature</td><td></td><td></td><td></td></tr><tr><td>Scenarios per Feature</td><td>1</td><td>64</td><td>4.75</td></tr><tr><td>Words per Feature</td><td>1</td><td>151</td><td>28.97</td></tr><tr><td>Scenario</td><td></td><td></td><td></td></tr><tr><td>Steps per Scenario</td><td>2</td><td>77</td><td>10.96</td></tr><tr><td>Words per Scenario</td><td>11</td><td>762</td><td>110.30</td></tr><tr><td>Test Impl.</td><td></td><td></td><td></td></tr><tr><td>Files per Test Impl.</td><td>1</td><td>25</td><td>5.29</td></tr><tr><td>Step defs. per Test Impl.</td><td>1</td><td>36</td><td>7.37</td></tr><tr><td>Lines per Test Impl.</td><td>37</td><td>8,681</td><td>1,763.50</td></tr><tr><td>Mutation</td><td></td><td></td><td></td></tr><tr><td>Ground-truth kills per instance</td><td>2</td><td>588</td><td>120.71</td></tr><tr><td>Assertion-dependent kills per instance</td><td>0</td><td>258</td><td>34.05</td></tr></table>

We execute $T _ { i } ^ { \mathrm { g t } }$ separately against every mutant in $U _ { i } ^ { \mathrm { m u t } }$ and record an outcome for each instance– mutant pair. Pair-level evidence is necessary because one project mutant can occur in multiple instance pools and have diferent outcomes for diferent ground-truth tests. For a test passing on the unmodified snapshot, a mutant is killed when its execution produces a test failure under the declared outcome policy, and it survives when the execution passes. A ground-truth run that times out, or fails before any Step executes, is an infrastructure outcome, and its pair is excluded. The valid ground-truth kills form $K _ { i } ^ { \mathrm { g t } }$ . Furthermore, mutants killed by the ground-truth test but survived by its assertion-stripped counterpart form an assertion-dependent subset for secondary analysis of the removed checks.

Stage V: Test levels and requirement relations. Stage V labels test levels and requirement relations, and neither label changes the task.

Test levels. We label a ground-truth Test Implementation by how its Step Definitions reach the production code: 1 component-level when they instantiate production classes directly; 2 integrationlevel when they obtain real services from the application’s own wiring, such as a dependencyinjection container or a command bus; and 3 system-level when they drive the assembled application only through its external interface, such as its command-line entry point or the documented entry point of a library. Among our collected test implementations, 4,276 tests are named by the oficial developers, e.g., Kapua’s separate unit and integration runners, and our labels agree for 4,266 of these 4,276 instances. None is end-to-end test, as Stage II excludes Scenarios that need a browser.

Requirement relations. Scenario � depends\_on Scenario � when � performs the behavior that � checks and relies on it, as a precondition or as one of its parts. For example, a Kapua Scenario that tests changing a user’s password first creates a password credential in a Given Step, and another Scenario performs the same creation and checks that it succeeds. As developers write such shared behavior either in a Background or among a Scenario’s own Steps, we look in both: �’s Given and When Steps must run every Step Definition that � runs before its checks. An LLM judge (DeepSeek-V4.1-Flash) then removes pairs in which � reuses only setup that � does not itself check. Scenarios whose Given and When Steps run the same Step Definitions, difering only in data or expected outcomes, are variant\_of each other. This yields 634 depends\_on relations and 3,584 variant\_of pairs.

Tab<sup>l</sup>e 3. Project-<sup>l</sup>eve<sup>l</sup> brea<sup>k</sup>down o<sup>f</sup> Req2Test-Bench: instances, per-instance measurements o<sup>f</sup> t<sup>h</sup>e Require-<sub>men</sub>t <sub>an</sub>d it<sub>s groun</sub>d<sub>-</sub>t<sub>ru</sub>th T<sub>es</sub>t I<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on, an</sub>d <sub>mu</sub>t<sub>a</sub>ti<sub>on</sub> d<sub>a</sub>t<sub>a.</sub> M<sub>eans are over</sub> th<sub>e eva</sub>l<sub>ua</sub>t<sub>e</sub>d i<sub>ns</sub>t<sub>ances.</sub>
<table><tr><td></td><td></td><td colspan="3">Instances</td><td colspan="5">Mean per evaluated instance</td><td colspan="3">Mutation</td></tr><tr><td>Project</td><td>Language / framework</td><td>Valid.</td><td>Eval.</td><td>Step defs.</td><td>Steps</td><td>Target I/F</td><td>Files</td><td>Lines</td><td>API</td><td>Mutants</td><td>Pairs</td><td>Kill (%)</td></tr><tr><td>coddy-agent</td><td>Go, godog</td><td>282</td><td>274</td><td>835</td><td>5.8</td><td>3.3</td><td>2.5</td><td>1,357</td><td>8.6</td><td>20,243</td><td>227,886</td><td>24.8</td></tr><tr><td>Algorand SDK</td><td>Java, Cucumber</td><td>117</td><td>111</td><td>188</td><td>4.1</td><td>6.2</td><td>2.2</td><td>542</td><td>7.5</td><td>7,766</td><td>12,853</td><td>58.4</td></tr><tr><td>Kapua</td><td>Java, Cucumber</td><td>605</td><td>330</td><td>415</td><td>6.3</td><td>9.5</td><td>1.5</td><td>2,286</td><td>18.9</td><td>48,911</td><td>33,000</td><td>41.8</td></tr><tr><td>CoreShop</td><td>PHP, Behat</td><td>422</td><td>334</td><td>371</td><td>16.1</td><td>3.9</td><td>13.8</td><td>2,631</td><td>44.0</td><td>10,205</td><td>171,174</td><td>28.2</td></tr><tr><td>PrestaShop</td><td>PHP, Behat</td><td>1,352</td><td>348</td><td>713</td><td>21.5</td><td>13.8</td><td>8.2</td><td>2,911</td><td>35.0</td><td>71,670</td><td>34,355</td><td>29.4</td></tr><tr><td>Sylius</td><td>PHP, Behat</td><td>1,950</td><td>319</td><td>552</td><td>11.9</td><td>1.9</td><td>3.4</td><td>1,215</td><td>2.0</td><td>12,373</td><td>256,324</td><td>26.1</td></tr><tr><td>python-docx</td><td>Python, behave</td><td>239</td><td>237</td><td>466</td><td>2.9</td><td>3.1</td><td>2.1</td><td>368</td><td>6.0</td><td>2,225</td><td>40,359</td><td>74.4</td></tr><tr><td>e-petitions</td><td>Ruby, Cucumber</td><td>119</td><td>98</td><td>166</td><td>10.6</td><td>0.2</td><td>4.0</td><td>652</td><td>0.4</td><td>5,997</td><td>105,193</td><td>13.7</td></tr><tr><td>Total</td><td></td><td>5,086</td><td>2,051</td><td>3,706</td><td>11.0</td><td>6.0</td><td>5.3</td><td>1,763</td><td>18.7</td><td>179,390</td><td>881,144</td><td>28.1</td></tr></table>

Benchmark statistics. Stages I–V yield 5,086 execution-validated instances from 1,415 Features in eight projects. Table 2 summarizes them, and Table 3 breaks them down by project.

## 3<sub>.</sub>3 E<sub>va</sub>l<sub>ua</sub>ti<sub>on</sub> C<sub>r</sub>it<sub>er</sub>i<sub>a</sub>

A generated test must first be a valid, executable implementation and must then exercise the behavior required by the scenario. We therefore report two complementary layers of metrics: stagebased execution outcomes that identify where generation or evaluation fails, and ground-truth-relative behavioral comparison that provides evidence beyond a passing execution.

Req2Test-Bench saves the generated test files in a separate copy of the project containing the complete production implementation. Req2Test-Bench then executes the target Scenario using the project’s native BDD runner and records its outcome and behavioral measurements. For each agent and view, Req2Test-Bench evaluates one final generated test per instance in a predeclared population of size �. Intermediate edits, local checks, and repairs belong to the same generation run and do not count as additional evaluation candidates.

3.3.1 Stage-based execution outcomes. For each instance with a recorded result, we report success or the first stage at which test generation or evaluation fails. Failures caused by benchmark infrastructure are recorded separately and excluded from the following categories:

• Generation failure: the agent returns no usable test code, no writable file, or an invalid submission format.

• Build failure: the generated test files cannot be parsed, compiled, imported, or initialized with the project’s test environment. This category also covers interpreted languages.

• Binding failure: the test framework cannot bind every Scenario step to a Step Definition, or encounters an undefined, ambiguous, or pending step.

• Execution failure: the scenario does not complete due to an exception, crash, or timeout, without an assertion-failure outcome.

• Assertion failure: the generated test runs and an assertion fails against the unmodified produc tion implementation.

• Pass: the selected Scenario completes with every efective step passing and the test command exits successfully.

Let E be the attempted instances with recorded outcomes in the six categories above, after documented infrastructure exclusions, $N _ { \mathrm { e x e c } } = | \mathcal { E } |$ , and $o _ { i }$ the outcome of instance �. For $N _ { \mathrm { e x e c } } > 0$

the execution success rate is

$$
\mathrm { P a s s } @ 1 = \frac { 1 } { N _ { \mathrm { e x e c } } } \sum _ { i \in \mathcal { E } } 1 [ o _ { i } = \mathrm { p a s s } ] .\tag{3}
$$

All failure rates use the same denominator $N _ { \mathrm { e x e c } }$ . We report $N , N _ { \mathrm { e x e c } }$ , and the number and reasons for excluded instances.

3.3.2 Ground-truth-based behavioral comparison. Pass@1 captures whether a generated test executes successfully, but it does not quantify the extent to which the test exercises and checks the behavior required by the target Scenario. In the extreme, a Scenario may pass even when all generated Step-Definition bodies are empty.

Therefore, Req2Test-Bench further compares the passing generated test with its developerwritten ground-truth test along three dimensions: ① API-call F1 and ② coverage Jaccard describe the exercised behavior, while ③ mutant-kill recall assesses fault detection. Our primary end-to-end measure combines execution success with complete mutant-kill recall. For each dimension �, let $G T _ { i } ^ { s }$ and $G e n _ { i } ^ { s }$ denote the ground-truth and generated-test sets.

API-call F1. The API sets identify public production methods directly called by the ground-truth or generated Test Implementation, including setup, action, and check Steps. Production-to-production calls and calls made only by retained testing infrastructure are excluded. For ${ G T _ { i } ^ { \mathrm { a p i } } \neq \emptyset }$

$$
\mathrm { A P I F } 1 _ { i } = 2 \times \frac { \lvert G e n _ { i } ^ { \mathrm { a p i } } \cap G T _ { i } ^ { \mathrm { a p i } } \rvert } { \lvert G e n _ { i } ^ { \mathrm { a p i } } \rvert + \lvert G T _ { i } ^ { \mathrm { a p i } } \rvert } .\tag{4}
$$

Coverage Jaccard. Coverage compares sets of executed code lines across all production files:

$$
\mathrm { C o v J a c c a r d } _ { i } = \frac { \left| { G e n _ { i } ^ { \mathrm { c o v } } \cap G T _ { i } ^ { \mathrm { c o v } } } \right| } { \left| { G e n _ { i } ^ { \mathrm { c o v } } \cup G T _ { i } ^ { \mathrm { c o v } } } \right| } .\tag{5}
$$

High coverage overlap does not establish that the required outcomes are asserted.

Mutant-kill recall. Within the fixed pool $U _ { i } ^ { \mathrm { m u t } } , G T _ { i } ^ { \mathrm { m u t } }$ is $K _ { i } ^ { \mathrm { g t } }$ restricted to retained instance–mutant pairs, and $G e n _ { i } ^ { \mathrm { { m u t } } }$ contains the generated test’s kills on those same pairs. Timeout and infrastructure exclusions are reported with counts. All ground-truth kills in the scored scope are used, not only the assertion-dependent subset. For $G T _ { i } ^ { \mathrm { m u t } } \neq \emptyset$

$$
\mathrm { M u t R e c a l l } _ { i } = \frac { \lvert G e n _ { i } ^ { \mathrm { m u t } } \cap G T _ { i } ^ { \mathrm { m u t } } \rvert } { \lvert G T _ { i } ^ { \mathrm { m u t } } \rvert } .\tag{6}
$$

End-to-end success. Let ${ \cal T } _ { \mathrm { m u t } }$ be the planned instances with supported, nonempty ground-truth kill sets. We define ground-truth-kill-complete pass@1 as

$$
\mathrm { K i l l C o m p l e t e } @ 1 = \frac { 1 } { N } \sum _ { i \in J _ { \mathrm { m u t } } } 1 \left[ o _ { i } = \mathrm { p a s s } \ \wedge \ \mathrm { M u t R e c a l l } _ { i } = 1 \right] .\tag{7}
$$

The denominator is the complete planned population �, and a non-passing instance is never killcomplete. A requirement’s text may leave project-specific setup and expected outcomes implicit. We therefore use the developer-written ground-truth test as a concrete reference for evaluating behavioral adequacy (Definition 2). A passing test that misses any evaluated ground-truth kill is kill-incomplete. It may still satisfy the requirement’s text under a defensible interpretation, but it misses faults that the developers’ test detects, so it is not adequate in the project’s context.

Aggregation and interpretation. Within each project, we average API F1, coverage Jaccard, and mutant-kill recall over all � instances (API F1 over those whose ground truth calls a production method directly), with a non-passing test scoring 0, so that these means are end-to-end measures like Pass@1 and KillComplete@1. Project scores are then averaged with equal weight. These means describe each agent’s passing tests, whereas KillComplete@1 uses the planned population. None of these comparisons establishes semantic equivalence or exhaustive requirement satisfaction.

3.3.3 Eficiency. For each configuration and view, we report generation time, LLM calls, input and output tokens, and monetary cost where recorded. Costs of post-generation test execution and ofline mutation analysis are reported separately. In addition to per-instance summaries and totals, we report generation cost per execution pass and per ground-truth-kill-complete pass.

## 4 Ex<sub>p</sub>erimental Desi<sub>g</sub>n

In this section, we formulate the research questions, introduce the evaluated approaches, specify how each approach is executed, and describe how the resulting observations are analyzed.

## 4.1 Research Questions

Our empirical study addresses the following research questions: RQ1 and RQ2 answer question Q1 of the introduction, RQ3 and RQ4 question Q2, and Section 5.5 question Q3.

RQ1 (Efectiveness): How efectively do LLM agents solve Req2Test tasks in Req2Test-Bench? This question investigates the current capability of state-of-the-art LLM agents to transform naturallanguage requirements into executable and behaviorally adequate tests.

RQ2 (Eficiency). How eficiently do the LLM agents solve the Req2Test tasks, and what efectiveness– eficiency trade-ofs do they exhibit? We measure monetary cost, end-to-end time, model requests, and tokens per generation, and relate cost to efectiveness through the cost per kill-complete pass. RQ3 (Associated Factors). What requirement-, test-, and repository-level characteristics are associated with successful Req2Test generation? This question investigates whether current agents perform consistently across the heterogeneous settings represented by Req2Test-Bench. We compare the all-hidden and target-hidden views on the same instances, and relate success to characteristics of the Requirement, its ground-truth Test Implementation (including its test level), the hidden code, and the Scenario’s prerequisites among the depends\_on relations.

RQ4 (Failure Modes). Whatfailure modes do current agents exhibit, and what limitations do these failures reveal? We first identify the evaluation stage at which an unsuccessful generation fails. We then derive failure modes bottom-up from a stratified sample, explaining both why failing tests fail and why passing tests miss ground-truth kills.

## 4<sub>.</sub>2 E<sub>va</sub>l<sub>ua</sub>t<sub>e</sub>d M<sub>o</sub>d<sub>e</sub>l<sub>s an</sub>d A<sub>gen</sub>t<sub>s</sub>

We evaluate eight agent–model–reasoning configurations using four LLM backbones: ① DeepSeek-V4.1-Flash (DS) [8], ② GPT-6 Luna [28], ③ GPT-6 Sol [28], and ④ Claude Sonnet 5.5 [4]. The four general-purpose coding agents are ① OpenHands (openhands-sdk 1.34.0) [38], ② Aider (0.86.2) [2], ③ Codex CLI (0.153.4; 0.157.1 for GPT-6 Sol) [29], and ④ Claude Code (2.1.274) [7] Reasoning efort is set to disabled or high, depending on the configuration. We call each combination of agent, LLM, and reasoning efort a configuration and name it by its agent and LLM, e.g., Codex/DS. Codex/DS (of) runs DS without reasoning. Claude Code/Sonnet runs Claude Sonnet 5.5 at high efort with an output limit of 128,000 tokens per request, and Claude Code/DS uses a limit of 32,000 tokens.

Aider and OpenHands run DS in non-thinking mode, with limits of 10 reflections and 200 LLM responses, respectively. Codex CLI and Claude Code each allow up to 200 LLM requests. Each agent receives the target requirement and corresponding target interfaces, and uses its native mechanisms for repository inspection, file editing, and command execution. We retain the native interaction workflow of each agent rather than forcing all agents into an identical implementation interface.

Tab<sup>l</sup>e 4. Overa<sup>ll</sup> e<sup>f</sup>ectiveness: means over eig<sup>h</sup>t projects. A non-passing test counts as 0 in simi<sup>l</sup>arity co<sup>l</sup>umns.
<table><tr><td rowspan="2">System</td><td rowspan="2">Model</td><td rowspan="2">Reasoning View</td><td rowspan="2"></td><td colspan="2">End-to-end success (%)</td><td colspan="3">Similarity to ground truth (%)</td></tr><tr><td></td><td>Pass@1 KillComplete@1 API F1 Coverage J Mutation R</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Aider</td><td rowspan="2">DS</td><td rowspan="2">off</td><td>Target-hidden</td><td>9.7</td><td>4.5</td><td>6.6</td><td>8.0</td><td>8.5</td></tr><tr><td>All-hidden</td><td>9.7</td><td>4.4</td><td>6.7</td><td>8.0</td><td>8.7</td></tr><tr><td rowspan="2">OpenHands DS</td><td rowspan="2"></td><td rowspan="2">off</td><td>Target-hidden</td><td>32.7</td><td>18.1</td><td>26.4</td><td>30.9</td><td>31.2</td></tr><tr><td>All-hidden</td><td>26.5</td><td>16.6</td><td>21.2</td><td>25.2</td><td>25.3</td></tr><tr><td rowspan="2">Claude Code DS</td><td rowspan="2"></td><td rowspan="2">high</td><td>Target-hidden</td><td>80.5</td><td>34.4</td><td>62.0</td><td>73.2</td><td>73.3</td></tr><tr><td>All-hidden</td><td>73.7</td><td>30.7</td><td>57.4</td><td>67.9</td><td>67.1</td></tr><tr><td rowspan="2">Claude Code Sonnet 5.5</td><td rowspan="2"></td><td rowspan="2">high</td><td>Target-hidden</td><td>80.9</td><td>35.9</td><td>61.7</td><td>76.0</td><td>75.6</td></tr><tr><td>All-hidden</td><td>78.4</td><td>34.8</td><td>59.1</td><td>74.0</td><td>73.7</td></tr><tr><td rowspan="2">Codex CLI</td><td rowspan="2">DS</td><td rowspan="2">off</td><td>Target-hidden</td><td>57.6</td><td>23.6</td><td>41.9</td><td>53.1</td><td>53.4</td></tr><tr><td>All-hidden</td><td>47.4</td><td>21.3</td><td>34.2</td><td>44.0</td><td>44.0</td></tr><tr><td rowspan="2">Codex CLI</td><td rowspan="2">DS</td><td rowspan="2">high</td><td>Target-hidden</td><td>82.8</td><td>33.8</td><td>64.2</td><td>75.9</td><td>76.3</td></tr><tr><td>All-hidden</td><td>75.4</td><td>33.7</td><td>56.5</td><td>70.0</td><td>69.1</td></tr><tr><td rowspan="2">Codex CLI</td><td rowspan="2">GPT-6 Luna high</td><td rowspan="2"></td><td>Target-hidden</td><td>66.2</td><td>28.3</td><td>49.9</td><td>59.6</td><td>60.6</td></tr><tr><td>All-hidden</td><td>57.1</td><td>27.8</td><td>42.9</td><td>52.7</td><td>53.6</td></tr><tr><td rowspan="2">Codex CLI</td><td rowspan="2">GPT-6 Sol</td><td rowspan="2">high</td><td>Target-hidden</td><td>73.5</td><td>31.3</td><td>54.2</td><td>65.3</td><td>66.1</td></tr><tr><td>All-hidden</td><td>65.3</td><td>29.1</td><td>48.2</td><td>59.7</td><td>60.3</td></tr></table>

## 4<sub>.</sub>3 I<sub>mp</sub>l<sub>emen</sub>t<sub>a</sub>ti<sub>on</sub> D<sub>e</sub>t<sub>a</sub>il<sub>s</sub>

Generation. Each generation receives the Requirement, the target interfaces, and one repository view inside a bubblewrap sandbox [5] whose network reaches only the model provider, under a common 3,600 s wall-clock limit. Agents may inspect the view and run local commands, including their own tests wherever the sandbox can load the retained code, but no such run can exercise the target behavior: its production bodies are placeholders that fail with an explicit “implementation hidden” error, and the harness provisions no application runtime and returns no execution feedback. The instructions state that the final test runs against a separate, complete implementation. We collect the test files from the workspace dif and discard production edits. Each instance receives one scored generation per configuration and view. Generations interrupted by provider-side limits are re-run, and none is repeated because of its outcome. To ensure that no generation observes ground-truth-derived artifacts, we audited the trajectories for access to such files and regenerated the afected generations in a hardened sandbox under a rule fixed in advance.

Evaluation. The evaluator runs each final test once against the complete production code with the project’s native runner in a network-isolated container (600 s, coverage enabled), and runs every passing test against the instance’s mutant pool (Section 3.2).

Many Scenarios of a project are tested through the same Step Definitions, so their Test Implementations are near-duplicates. We therefore evaluate one per group of instances whose ground-truth Test Implementations use the same Step-Definition methods. All configurations run on these 2,051 instances from 780 Features in both views.

Statistical analysis. Brackets indicate 95% confidence intervals obtained by cluster bootstrap resampling of feature files within each project. For the paired comparisons, we compute one-sided �-values using exact McNemar tests or paired sign-flip tests.

## 5 R<sub>esu</sub>lt<sub>s</sub>

## 5.1 RQ1: Efectiveness

Passing is common, but kill-completeness is not. In Table 4, the best-passing configuration, Codex/DS, passes 82.8% of the target-hidden instances ([80.5, 84.8]) but is kill-complete on only

![](images/5c76b5404d39962af02acb4ceaa90294174a5a99c56a28b7360058b1c5252e1c.jpg)  
Fi<sub>g</sub>. 3. Pass@1 (bar hei<sub>g</sub>ht) and KillCom<sub>p</sub>lete@1 (dark <sub>p</sub>art), in <sub>p</sub>ercent, <sub>p</sub>er <sub>p</sub>roject, view, and confi<sub>g</sub>uration.

33.8% ([28.4, 38.1]). As shown in Figure 3, per project, only 6.7–66.2% of the passing tests of all configurations and views are kill-complete. The gap persists when the generation is repeated, although single verdicts vary (Section 5.5). A passing test of Codex/DS reproduces 91% of its ground-truth test’s kills on average, yet the gap between Pass@1 and KillComplete@1 is not merely a matter of demanding every kill. Under a relaxed criterion that accepts a test reproducing at least 95% of them, Codex/DS still succeeds on only 53.9% of the target-hidden instances, 28.9 percentage points (pp) below its Pass@1. High coverage similarity does not imply kill-completeness either. On the six projects other than coddy-agent and Kapua, 17.4–69.7% of the passing tests with a coverage Jaccard of at least 0.9 are kill-incomplete. Figure 4 shows a concrete example: the test of Codex/Sol passes, but it constructs the latent style directly in its own code instead of loading it from the project’s fixture file, and kills only 10 of the 110 mutants that the ground-truth test kills.

The gap persists when only check-dependent kills count. We empty the bodies of the groundtruth tests’ Then Step Definitions to identify mutants killed by the original tests but surviving the ablated tests. Across the seven DS and GPT-6 configurations, counting only these kills reduces the gap between Pass@1 and KillComplete@1 by only 24–37%, without changing the configurations’ KillComplete@1 ranking. When counting only kills inside the hidden target implementation, the gap also persists, e.g., 11.9 pp for Codex/DS.

Kill-incomplete tests mostly fall short on checking, not only on the route. We find agents fall short on both clauses of Definition 2, i.e., “exercises the behavior” and “checks expected outcomes”. For example, only 43.5% of the target-hidden instances get a passing Codex/DS test that executes every mutant that the developers’ test kills, and only 41.1% get one that kills every such mutant it executes. Of its kill-incomplete tests, 77.9% execute such a mutant and still pass, and only 22.1% miss only mutants on lines they never run. Thus, most kill-incomplete tests run the faulty code on their own path and check less than the developers’ test there.

Scafold and reasoning efort matter, but the costliest LLM is not the best. 1 Agent scafold: with DS and reasoning of, Codex CLI passes 57.6% of the target-hidden instances, against 32.7% for OpenHands and 9.7% for Aider. 2 Reasoning efort: for Codex/DS, high instead of no reasoning raises Pass@1 from 57.6% to 82.8% and KillComplete@1 from 23.6% to 33.8%. 3 LLM: with Codex CLI at high efort, DS passes more instances (82.8%) than the cheaper GPT-6 Luna (66.2%) and the costliest LLM, GPT-6 Sol (73.5%; Section 5.2). Its KillComplete@1 is 5.5 pp [2.2, 8.2] above Luna’s and 2.5 pp [−1.0, 5.4] above Sol’s. Under Claude Code, Claude Sonnet 5.5 reaches about the same

Binhan<sub>g</sub> Qi, Yimin<sub>g</sub> Liu, Yun Lin, Yan<sub>g</sub>jie Zhou, Hailon<sub>g</sub> Sun, and Jin Son<sub>g</sub> Don<sub>g</sub>

![](images/9e927d06f3cc3e4392202f15ea9cddd93a5ec3475925f3a18882f75a3d59ab1c.jpg)  
Fi<sub>g</sub>. 4. An illustrative exam<sub>p</sub>le from python-docx, sty-latent-props.feature:36 (tar<sub>g</sub>et-hidden view). L<sub>e</sub>ft<sub>:</sub> th<sub>e</sub> <sub>agen</sub>t’<sub>s</sub> i<sub>npu</sub>t <sub>an</sub>d it<sub>s</sub> t<sub>es</sub>t <sub>resu</sub>lt<sub>.</sub> Ri<sub>g</sub>ht<sub>:</sub> th<sub>e</sub> <sub>groun</sub>d<sub>-</sub>t<sub>ru</sub>th t<sub>es</sub>t <sub>an</sub>d th<sub>e</sub> t<sub>es</sub>t <sub>genera</sub>t<sub>e</sub>d b<sub>y</sub> C<sub>o</sub>d<sub>ex</sub>/S<sub>o</sub>l<sub>.</sub> Y<sub>e</sub>ll<sub>ow: w</sub>h<sub>a</sub>t th<sub>e</sub> R<sub>equ</sub>i<sub>remen</sub>t l<sub>eaves a</sub>b<sub>s</sub>t<sub>rac</sub>t <sub>an</sub>d h<sub>ow</sub> th<sub>e groun</sub>d t<sub>ru</sub>th fill<sub>s</sub> it<sub>; re</sub>d<sub>: w</sub>h<sub>ere</sub> th<sub>e genera</sub>t<sub>e</sub>d t<sub>es</sub>t constructs t<sup>h</sup>e object itse<sup>lf</sup>.

Pass@1 and KillComplete@1 as DS (80.9% and 80.5%, 35.9% and 34.4%). The leaderboard in our repository [32] lists the scores of every configuration.

Stron<sub>g</sub>er confi<sub>g</sub>urations <sub>g</sub>ain kill-com<sub>p</sub>lete tests b<sub>y p</sub>assin<sub>g</sub> more instances<sub>,</sub> not b<sub>y</sub> a hi<sub>g</sub>her kill-complete share of their passing tests. In the target-hidden view, Codex/DS has 336 more kill-complete tests than OpenHands/DS and 104 more than Codex/Luna, and the whole surplus comes from instances that only one of the two configurations passes. On the instances that both pass, their tests are kill-complete about equally often (51.5% against 54.4% for OpenHands/DS, and 42.6% against 44.0% for Codex/Luna).

Answer to RQ1. Agents often produce a passing test but far less often a behaviorally adequate one. The best-passing configuration, Codex/DS, passes 82.8% of target-hidden instances but is kill-complete on only 33.8%. The lead of Codex/DS over OpenHands/DS and Codex/Luna comes from passing more instances, not from a higher kill-complete share of its passing tests.

## 5.2 RQ2: Eficienc<sub>y</sub>

Cost varies about 30-fold, and efectiveness does not follow it. As shown in Table 5, a targethidden generation costs from \$0.039 for Codex/Luna to \$1.16 for Codex/Sol on average, although a Codex/Sol generation takes only 3.1 minutes and 16.8 LLM calls. Figure 5 orders the configurations by monetary cost: neither Pass@1 nor KillComplete@1 rises with it. In the target-hidden view, only Codex/Luna, Claude Code/DS, and Claude Code/Sonnet lie on the Pareto frontier of cost and KillComplete@1. Every other configuration costs more than one of them without a higher KillComplete@1.

Spending more yields sharply diminishing returns. Moving along this frontier from Codex/Luna to Claude Code/DS multiplies the cost by 4.0 and raises KillComplete@1 by 6.1 pp [2.8, 8.8]. Beyond that, additional spending buys no measurable gain. Codex/DS costs 1.5 times as much as Claude Code/DS for a KillComplete@1 that does not difer measurably (−0.6 pp [−2.4, 1.1]), and Codex/Sol 7.4 times as much for one that is no higher (−3.1 pp [−5.8, 0.1]). With Codex CLI, moving from GPT-6 Luna to DS multiplies the cost by 6.0 for 5.5 pp, and moving on to GPT-6 Sol by another 4.9 for no further gain: Sol costs 30 times as much as its sibling Luna for 3.0 pp [1.1, 5.0]. Reasoning efort also raises the price of success. It roughly doubles Codex/DS’s cost per generation for 10.2 pp more KillComplete@1, and a kill-complete test costs \$0.74 instead of \$0.55.

Table 5. Cost and resource usa<sub>g</sub>e <sub>p</sub>er <sub>g</sub>eneration in the tar<sub>g</sub>et-hidden view.
<table><tr><td rowspan="2"></td><td rowspan="2">Aider DS, off</td><td rowspan="2">OpenHands DS, off</td><td colspan="2">Claude Code</td><td colspan="4">Codex CLI</td></tr><tr><td></td><td>DS, high Sonnet, high</td><td></td><td></td><td>DS, off DS, high Luna, high Sol, high</td><td></td></tr><tr><td>$ per generation</td><td>0.061</td><td>0.187</td><td>0.158</td><td>0.680</td><td>0.121</td><td>0.236</td><td>0.039</td><td>1.16</td></tr><tr><td>$ per kill-complete test</td><td>1.56</td><td>1.21</td><td>0.49</td><td>1.99</td><td>0.55</td><td>0.74</td><td>0.15</td><td>3.92</td></tr><tr><td>Time (minutes)</td><td>2.0</td><td>5.7</td><td>6.2</td><td>2.7</td><td>4.1</td><td>8.7</td><td>3.2</td><td>3.1</td></tr><tr><td>LLM calls</td><td>2.0</td><td>137.8</td><td>88.9</td><td>22.3</td><td>96.6</td><td>102.4</td><td>23.6</td><td>16.8</td></tr></table>

![](images/54987479dc385c866a3692ca9d3b142c88edc3a10de7eb5007b364ad782aa0fe.jpg)

![](images/c5f4f476024587f36b0af398df3ca75c8ea49165022e802221400d5393e225b3.jpg)  
Fi<sub>g</sub>. 5. KillCom<sub>p</sub>lete@1 (filled) and Pass@1 (hollow) of the confi<sub>g</sub>urations, ordered b<sub>y</sub> their cost <sub>p</sub>er <sub>g</sub>eneration. T<sup>h</sup>e t<sup>h</sup>ic<sup>k l</sup>ine joins t<sup>h</sup>e Pareto <sup>f</sup>rontier o<sup>f</sup> t<sup>h</sup>e point estimates, i.e., t<sup>h</sup>e con<sup>f</sup>igurations w<sup>h</sup>ose Ki<sup>ll</sup>Comp<sup>l</sup>ete@1 <sub>no o</sub>th<sub>er reac</sub>h<sub>es a</sub>t th<sub>e same or</sub> l<sub>ower cos</sub>t<sub>.</sub>

Answer to RQ2. Performance of agents does not follow cost. Along the target-hidden Pareto frontier, 4.0 times the cost buys 6.1 pp of KillComplete@1, and more spending buys no measurable gain.

## 5.3 RQ3: Associated Factors

Hidin<sub>g</sub> the surroundin<sub>g</sub> im<sub>p</sub>lementations costs <sub>p</sub>asses more than kill-com<sub>p</sub>lete tests. As shown in Table 4, on the same instances, the all-hidden view lowers the Pass@1 of the five high-efort configurations by 2.5–9.1 pp (least for Claude Code/Sonnet), most of all on CoreShop (7.8–39.5 pp), but their KillComplete@1 by only 0.1–3.7 pp, which for Codex/DS, Codex/Luna, and Claude Code/Sonnet is too little to tell from no drop. Reading the surrounding production code thus mainly helps a test run. Without it, Codex/DS and Codex/Luna tests more often violate the runtime contracts of the interfaces that they call (see Section 5.4).

L<sub>onger</sub> S<sub>cenar</sub>i<sub>os an</sub>d l<sub>arger groun</sub>d<sub>-</sub>t<sub>ru</sub>th t<sub>es</sub>t<sub>s are</sub> h<sub>ar</sub>d<sub>er</sub> t<sub>o pass an</sub>d h<sub>ave more</sub> kill<sub>s</sub> t<sub>o</sub> reproduce. We rank each project’s instances by a characteristic and compare the third with the lowest values to the third with the highest (five high-efort configurations, target-hidden view).

![](images/ffbce1d2f1729b49b821d9ee312fec540173dd0d0e988c41388871ab2933d2cb.jpg)

![](images/b91b53b68490cf6802a88a211579a55bba7fb4d05f3b533d414871b2beccced5.jpg)  
Fi<sub>g</sub>. 6. (a) Where <sub>g</sub>enerations end, in <sub>p</sub>ercent of each confi<sub>g</sub>uration’s <sub>g</sub>enerations (<sub>p</sub>roject means over both views). Black ticks mark the end of the <sub>p</sub>assin<sub>g</sub> tests. (b) Wh<sub>y</sub>: failure modes of failin<sub>g</sub> <sub>g</sub>enerations and kill-incom<sub>p</sub>lete tests<sub>,</sub> in <sub>p</sub>ercent.

From the shortest to the longest Scenarios (a median of 5 against 15 Steps), Pass@1 falls from 82.8% to 70.8% and KillComplete@1 from 45.8% to 22.7%. Larger ground-truth tests show the same pattern, by lines, covered production lines, or kills: Pass@1 falls by 6–14 pp and KillComplete@1 by 24–32 pp. Repository-level characteristics matter less for kill-completeness. With more hidden code or more target interfaces, both metrics fall by 11–15 pp.

Answer to RQ3. The all-hidden view lowers Pass@1 by 2.5–9.1 pp but KillComplete@1 by only 0.1–3.7 pp.   
Longer Scenarios and larger ground-truth tests are harder to pass and have more kills to reproduce.

## 5.4 RQ4: Failure Modes

We analyze both failing generations, which produce no passing tests, and passing but kill-incomplete tests. For each of five configurations, we drew a stratified random sample of 40 cases of each type, yielding 200 failing generations and 200 kill-incomplete tests. We derived the failure modes in Figure 6(b) through a bottom-up analysis of the sampled cases’ outcomes and code, supplemented by missed-mutant analysis for kill-incomplete tests. We assigned labels using deterministic rules and an LLM judge (Claude Opus 5.5), with manual checks by the authors.

Failing generations mostly deliver no test or misuse the project’s interfaces. 1 No test delivered: As shown in Figure 6(a) (grey bar), Aider/DS and OpenHands/DS deliver no Test Implementation in 53.4% and 59.0% of their generations. Aider/DS’s reply mostly repeats itself up to its output limit, and OpenHands/DS mostly exhausts its 200 LLM calls on exploration, without writing a test file. As shown in Table 5, with the same call limit and LLM, Codex CLI and Claude Code deliver a test with about 100 calls. 2 Hallucinated or miswired interfaces (45% and 41% of the failing generations of Codex/DS and Codex/Luna, Figure 6(b), left): The test uses classes, methods, signatures, or services that do not exist in the project, such as Tools::link\_rewrite() in 40 Codex/Luna generated tests on PrestaShop, redefines existing Steps, or misuses the BDD framework. 3 Unmet runtime contracts (25% and 43%): The test calls existing interfaces in ways that their implementation rejects, e.g., an operation that the concrete class does not support. Declarations do not reveal such contracts, whereas the surrounding implementations often do. The mode is thus more frequent when all implementations are hidden (34% and 51%, against 13% and 32% in the target-hidden view). Another 13% and 15% fail on the information gap below.

Kill<sub>-</sub>i<sub>ncomp</sub>l<sub>e</sub>t<sub>e</sub> t<sub>es</sub>t<sub>s</sub> f<sub>a</sub>ll <sub>s</sub>h<sub>or</sub>t <sub>w</sub>h<sub>ere</sub> th<sub>e</sub> R<sub>equ</sub>i<sub>remen</sub>t l<sub>eaves</sub> i<sub>n</sub>f<sub>orma</sub>ti<sub>on uns</sub>t<sub>a</sub>t<sub>e</sub>d<sub>.</sub> This information gap accounts for 90–99% of each judged configuration’s kill-incomplete tests (Figure 6(b), right). In 120 of the 200 judged tests, the missing information is in the agent’s own workspace, e.g., in fixture files or helpers. In most of the remaining 80, the needed project convention of setup or check appears only in developers’ tests, which the benchmark removes from the agent’s workspace. 1 Ambiguous setup (over 50%): the Step names a state that can be established in several plausible ways, and the test takes a shorter but unexpected way than the ground truth, as in Figure 4. For example, on python-docx instances whose ground truth opens a fixture file, tests that also open one are kill-complete in 81.8% of cases, against 24.6% for tests that construct the state themselves. 2 Tacit expected outcome (up to 28%): the Then Step states only part of what the ground truth checks, and what else to check is tacit for developers with domain knowledge. For example, on 25 Algorand SDK Scenarios, the Steps name a few fields of a response, the ground truth compares the whole response, and only 9 of the 320 passing tests are kill-complete, all of them by Claude Code/Sonnet. Ambiguous setup and tacit expected outcomes are normal in requirements: a requirement describes behavior in terms of the domain [40], and the knowledge that completes it stays tacit with the developers [9]. An agent must recover this knowledge from the project, or the requirement must be refined to state it. 3 Hedging (10% for Codex/DS): fallbacks or broad exception handlers let Steps pass whatever the product does.

Answer to RQ4. Failing generations mostly deliver no test or misuse the project’s interfaces. Kill-incomplete tests fall short where the setup is ambiguous or the expected outcome tacit (90–99%).

## 5<sub>.</sub>5 Toward Re<sub>q</sub>uirement-Grounded Harnesses

Four preliminary studies investigate what a harness around the agent can supply.

Grounding tests in project knowledge targets ambiguous setup. An added prompt paragraph instructs the agent to establish each Given through the project’s own fixtures, helpers, and entry points, not by constructing objects. On 231 python-docx and Algorand SDK instances not used to design the paragraph, this raises KillComplete@1 by 6.1 pp, 15.0 pp, and 8.0 pp for Codex/DS, Codex/Luna, and Codex/Sol $( p \leq 0 . 0 0 5 )$ .

Enhancing the Requirement targets ambiguous setup and tacit expected outcomes. In domain language and naming no code, one author modifies the prompt to clarify what the ground-truth test sets up or checks beyond what a Step states. For example, the unchanged Step “the parsed Get Block response should have rewards pool . . . ” is clarified as “the parsed response matches the mock http response as a whole”. As a result, Codex/Luna’s kill-complete tests rise from 2 to 14 on 17 Algorand SDK Scenarios with tacit outcomes $( p < 0 . 0 0 1 )$ , and from 25 to 36 on 60 python-docx instances with an ambiguous Given. Enhancing requirements automatically with the project’s conventions, fixtures, and helpers is a potential direction.

Decomposing the Requirement targets misused project interfaces. Longer Scenarios are harder to pass (Section 5.3), but many of their Steps are shared with smaller Scenarios that they depend on. A harness could therefore decompose the target requirement and generate tests for the resulting smaller requirements first, and give them to the agent for the test generation of target requirement. We repeated test generation for 100 requirements from CoreShop and Kapua for which Codex/Luna had initially failed to produce passing tests. Retries with the unchanged prompt produced passing tests for 39 requirements due to the agent’s non-determinism. Augmenting the prompt with Codex/Luna’s own previously generated passing tests for smaller depends\_on requirements increased this number to 74 $( p < 0 . 0 0 1 )$ , suggesting that reusing these tests improves success beyond retries alone.

Exploiting non-determinism. Non-determinism makes a single generation unreliable but leaves room for improvement. We observed that the best of three generations, chosen by the evaluator, is kill-complete 8.6–10.2 pp more often than a single one (e.g., for Codex/DS, 44.8% against 36.2% on the repeated sample). A harness can exploit this only if it recognizes the behaviorally adequate candidate.

## 6 Th<sub>rea</sub>t<sub>s</sub> t<sub>o</sub> V<sub>a</sub>lidit<sub>y</sub>

Construct validity. KillComplete@1 measures behavioral adequacy against one developer-written test, so a kill-incomplete test may still validly test a narrower reading of the requirement. However, the gap persists when only check-dependent kills count, and most kill-incomplete tests run a faulty line and still pass (Section 5.1), so the gap does not stem only from the test’s setup route. Moreover, mutants are artificial faults, but their detection correlates with that of real faults [22]. Equivalent mutants, which no test can kill, do not lower any score, because we count only the mutants that the ground-truth test kills.

Internal validity. Due to the time budgets, the experiments did not repeat all agents’ test generations multiple times on the whole benchmark to reduce the efects of agents’ non-determinism. We choose five configurations (Codex/DS, Codex/Luna, Claude Code/DS, OpenHands/DS, and Aider/DS) to regenerate twice to assess the efects. Up to 24% of their pass verdicts and 17% of their kill-complete verdicts flip, but their Pass@1 and KillComplete@1 change by at most 4.5 pp. Additionally, the target-hidden view rests on an LLM-drafted, author-accepted map of the target implementation, which may hide slightly more or less than the code that realizes the target behavior. However, the all-hidden view does not depend on this map, and it shows the same gap between passing and kill-complete tests. Finally, the projects are public and may be in the training data of some LLMs. But memorized developer tests would raise KillComplete@1 and thus narrow, not widen, the gap that we report.

External validity. The eight projects were selected purposively, but they cover five languages, four BDD frameworks, six domains, and three test levels, and our reported gap appears in every one of them. Moreover, Gherkin Scenarios are more structured than most informal requirements. Informal requirements are likely to state even less of the setup and the expected outcome, so we expect the information gap to be larger for them, not smaller. Finally, both views in the benchmark remove the tests of all Scenarios, so that none reveals part of the ground truth. However, in practice, a developer who extends a test suite could read them. Our results thus concern writing a test without the tests of other Scenarios. We leave such cases in future work.

## 7 C<sub>onc</sub>l<sub>us</sub>i<sub>on</sub>

We presented Req2Test-Bench, a benchmark of 5,086 execution-validated requirement-to-test instances from 8 real projects. It evaluates a generated test by whether it passes and reproduces the fault detection of the developers’ own test. We evaluated eight agent–model–reasoning configurations, and found that the best-passing configuration passes 82.8% of the instances but is kill-complete on only 33.8%. Kill-incomplete tests mostly fall short where the requirement leaves the setup ambiguous or the expected outcome tacit. They take shortcuts instead of the project’s own fixtures and entry points, or check less than the developers’ tests. This gap persists even with costlier agent configurations, motivating requirement-grounded harnesses for test generation that guide requirement understanding, test validation, and project-specific knowledge acquisition.

## 8 D<sub>a</sub>t<sub>a</sub> A<sub>va</sub>il<sub>a</sub>bilit<sub>y</sub>

The benchmark code and evaluation results are available at https://github.com/code-philia/Req2Test-Bench, and the dataset is available at https://huggingface.co/datasets/code-philia/Req2Test-Bench.

## R<sub>e</sub>f<sub>erences</sub>

[1] Toufique Ahmed, Martin Hirzel, Rangeet Pan, Avraham Shinnar, and Saurabh Sinha. 2024. Tdd-bench verified: Can llms generate tests for issues before they get resolved? arXiv preprint arXiv:2412.02883 (2024).

[2] Aider. 2026. aider is AI pair programming in your terminal. https://github.com/Aider-AI/aider.

[3] Parsa Alian, Noor Nashid, Mobina Shahbandeh, Taha Shabani, and Ali Mesbah. 2025. Feature-driven end-to-end test generation. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 450–462.

[4] Anthropic. 2026. Claude Sonnet 5.5. https://www.anthropic.com/claude-sonnet-5-5.

[5] Bubblewrap Contributors. 2026. Low-level unprivileged sandboxing tool used by Flatpak and similar projects. https://github.com/containers/bubblewrap

[6] Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397 (2022).

[7] claude-code Authors. [n. d.]. Claude Code. https://github.com/anthropics/claude-code.

[8] DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. arXiv:2609.19969 [cs.CL] https://arxiv.org/abs/2609.19969

[9] Alessio Ferrari, Paola Spoletini, and Stefania Gnesi. 2016. Ambiguity and tacit knowledge in requirements elicitation interviews. Requir. Eng. 21, 3 (Sept. 2016), 333–355. doi:10.1007/s00766-016-0249-3

[10] Margarida Ferreira, Luis Viegas, Joao Pascoal Faria, and Bruno Lima. 2025. Acceptance test generation with large language models: An industrial case study. In 2025 IEEE/ACM International Conference on Automation ofSoftware Test (AST). IEEE, 1–11.

[11] Jannik Fischbach, Julian Frattini, and Andreas Vogelsang. 2021. Cira: A tool for the automatic detection of causa relationships in requirements artifacts. arXiv preprint arXiv:2103.06768 (2021).

[12] Jannik Fischbach, Andreas Vogelsang, Dominik Spies, Andreas Wehrle, Maximilian Junker, and Dietmar Freudenstein. 2020. Specmate: Automated creation of test cases from acceptance criteria. In 2020 IEEE 13th international conference on software testing, validation and verification (ICST). IEEE, 321–331.

[13] Pedro Luís Fonseca, Bruno Lima, and João Pascoal Faria. 2025. Streamlining Acceptance Test Generation for Mobile Applications Through Large Language Models: An Industrial Case Study. In 2025 40th IEEE/ACM International Conference on Automated Software Engineering (ASE). 3311–3321. doi:10.1109/ASE63991.2025.00273

[14] Mattheus França and Fabiane Benitti. 2026. Req2test: an approach for the automatic generation of unit tests from user stories using llms. In Proceedings ofthe 41st ACM/SIGAPP Symposium on Applied Computing. 1654–1663.

[15] Huimin Hu, Cristian Cadar, and Michael Pradel. 2026. DifTestGen: Change-Directed LLM-Based Testing for Exposing Behavioral Diferences. arXiv preprint arXiv:2607.16024 (2026).

[16] Dong Huang, Jie M Zhang, Michael Luck, Qingwen Bu, Yuhao Qing, and Heming Cui. 2023. Agentcoder: Multi-agentbased code generation with iterative testing and optimisation. arXiv preprint arXiv:2312.13010 (2023).

[17] Infection. [n. d.]. PHP Mutators. https://infection.github.io/guide/mutators.html.

[18] Kush Jain, Gabriel Synnaeve, and Baptiste Rozière. 2025. TestGenEval: A Real World Unit Test Generation and Test Completion Benchmark. In The Thirteenth International Conference on Learning Representations. https://openreview. net/forum?id=7o6SG5gVev

[19] Yue Jia and Mark Harman. 2011. An Analysis and Survey of the Development of Mutation Testing. IEEE Trans. Softw. Eng. 37, 5 (Sept. 2011), 649––678. doi:10.1109/TSE.2010.62

[20] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-world Github Issues?. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=VTF8yNQM66

[21] Luciano Belo de Alcântara Júnior and João Eduardo Montandon. 2026. GivenWhenThen: A Dataset of BDD Test Scenarios Mined from Open Source Projects. In Proceedings ofthe 23rd International Conference on Mining Software Repositories. 595–599.

[22] René Just, Darioush Jalali, Laura Inozemtseva, Michael D Ernst, Reid Holmes, and Gordon Fraser. 2014. Are mutants a valid substitute for real faults in software testing?. In Proceedings of the 22nd ACM SIGSOFT international symposium on foundations of software engineering. 654–665.

[23] Weiyu Kong, Yun Lin, Xiwen Teoh, Duc-Minh Nguyen, Ruofei Ren, Jiaxin Chang, Haoxu Hu, and Haoyu Chen. 2026. Compiling Large Multi-modal Requirement Documents into Runnable Software Systems: From an Agentic Test-Driven Perspective. Proc. ACM Softw. Eng. 3, ISSTA, Article ISSTA105 (Oct. 2026), 24 pages. doi:10.1145/3832196

[24] Bowen Li, Wenhan Wu, Ziwei Tang, Lin Shi, John Yang, Jinyang Li, Shunyu Yao, Chen Qian, Binyuan Hui, Qicheng Zhang, Zhiyin Yu, He Du, Ping Yang, Dahua Lin, Chao Peng, and Kai Chen. 2024. DevBench: A Comprehensive Benchmark for Software Development. arXiv:2403.08604v1 [cs.CL] https://arxiv.org/abs/2403.08604v1

[25] Jingyao Liu, Chen Huang, Zhizhao Guan, Wenqiang Lei, and Yang Deng. 2026. E2EDev: Benchmarking Large Language Models in End-to-End Software Development Task. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 35032–35068.

[26] Niels Mündler, Mark N Müller, Jingxuan He, and Martin Vechev. 2024. Swt-bench: Testing and validating real-world bug-fixes with code agents. Advances in Neural Information Processing Systems 37 (2024), 81857–81887.

[27] Zifan Nan, Zhaoqiang Guo, Kui Liu, and Xin Xia. 2025. Test Intention Guided LLM-Based Unit Test Generation. In 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). 1026–1038.

[28] OpenAI. 2026. Introducing GPT-6 Sol and Luna. https://openai.com/index/introducing-gpt-6-sol-and-luna/.

[29] OpenAI. 2026. Lightweight coding agent that runs in your terminal. https://github.com/openai/codex.

[30] PIT. [n. d.]. Java Mutation Operators. https://pitest.org/quickstart/mutators/.

[31] Binhang Qi, Yun Lin, Xinyi Weng, Yuhuan Huang, Chenyan Liu, Hailong Sun, Zhi Jin, and Jin Song Dong. 2026. Generating Project-Specific Test Cases with Requirement Validation Intention. Proc. ACM Softw. Eng. 3, ISSTA, Article ISSTA090 (Oct. 2026), 23 pages. doi:10.1145/3832181

[32] Req2Test-Bench Authors. 2026. Req2Test-Bench. https://github.com/code-philia/Req2Test-Bench

[33] Max Sch"afer, Sarah Nadi, Aryaz Eghbali, and Frank Tip. 2024. An Empirical Evaluation ofUsing Large Language Models for Automated Unit Test Generation. IEEE Trans. Softw. Eng. 50, 1 (Jan. 2024), 85––105. doi:10.1109/TSE.2023.3334955

[34] Xinyu Shi, Zhou Yang, and An Ran Chen. 2026. Bridging Behavior and Implementation: Automated Java Glue Code Generation for Behavior-Driven Development. arXiv:2607.19703 [cs.SE] https://arxiv.org/abs/2607.19703

[35] The Cucumber Open Source Project. 2026. Gherkin Reference. https://cucumber.io/docs/gherkin/reference/

[36] Michele Tufano, Shao Kun Deng, Neel Sundaresan, and Alexey Svyatkovskiy. 2022. Methods2Test: A dataset of focal methods mapped to test cases. In Proceedings ofthe 19th International Conference on Mining Software Repositories. 299–303.

[37] Chunhui Wang, Fabrizio Pastore, Arda Goknil, Lionel C Briand, and Zohaib Iqbal. 2015. UMTG: a toolset to automatically generate system test cases from use case specifications. In Proceedings ofthe 2015 10th Joint Meeting on Foundations of Software Engineering. 942–945.

[38] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Daniel Shao, Niklas Muennighof, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. 2025. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. In The Thirteenth International Conference on Learning Representations. https://openreview.net/forum?id=OJd3ayDDoF

[39] Chen Yang and Junjie Chen. 2026. Uncovering Business Logic Bugs via Semantics-Driven Unit Test Generation. arXiv:2604.23509 [cs.SE] https://arxiv.org/abs/2604.23509

[40] Pamela Zave and Michael Jackson. 1997. Four dark corners of requirements engineering. ACM Trans. Softw. Eng. Methodol. 6, 1 (Jan. 1997), 1–30. doi:10.1145/237432.237434

[41] Quanjun Zhang, Ye Shang, Chunrong Fang, Siqi Gu, Jianyi Zhou, and Zhenyu Chen. 2024. Testbench: Evaluating class-level test case generation capability of large language models. arXiv preprint arXiv:2409.17561 (2024).