# BobFlow: Towards Human–Agent Collaborative Data Science Using Dataflows

Jiadong Bai University of California, Irvine Irvine, California, USA jiadongb@ics.uci.edu

Yicong Huang University of Massachusetts Amherst Amherst, Massachusetts, USA yiconghuang@cs.umass.edu

Chen Li University of California, Irvine Irvine, California, USA chenli@ics.uci.edu

## ABSTRACT

Recent advances in large language models (LLMs) have established human–agent collaboration as a promising paradigm for data science. Efective collaboration requires a shared computational abstraction that enables agents to reason about processing logic while allowing human analysts to understand, guide, and refine the analy sis. Although scripts written in Python or SQL provide precise control, their syntax and execution details can be dificult for analysts to understand and modify. In this paper, we argue that dataflows are an efective abstraction for human–agent collaboration in data science. We introduce BobFlow, a system that combines the dataflow abstraction with the ReAct paradigm. It enables agents to iteratively construct and execute dataflows for data science tasks while allowing analysts to inspect the analysis and provide feedback. Our user study shows that BobFlow helps analysts understand, inspect, and verify agent-generated analyses. A thorough experimental evaluation across proprietary and open-weight LLMs shows accuracy score gains of up to 18.6 points at just over 1/3 of a script-based ReAct agent’s inference cost.

## PVLDB Reference Format:

Jiadong Bai, Yicong Huang, and Chen Li. BobFlow: Towards Human–Agent Collaborative Data Science Using Dataflows. PVLDB, 20(1): XXX-XXX, 2027.

doi:XX.XX/XXX.XX

## PVLDB Artifact Availability:

The source code, data, and/or other artifacts have been made available at https://github.com/Texera/bobflow.

## 1 INTRODUCTION

Recent advances in large language models (LLMs) are fundamentally transforming how humans and AI agents collaborate on data science tasks [34, 61]. Most commonly, agent-assisted data science has followed a human-led paradigm, where human analysts drive the analytical process, and AI agents assist with individual tasks, such as code generation, data transformation, and result interpreta tion [27, 50]. Recently, a growing number of LLM-powered agents have demonstrated the ability to plan and execute complex analytical tasks autonomously. Systems such as Claude Science [2],

OpenAI’s in-house data agent [38], and Google’s multi-agent Co-Scientist [18] exemplify this trend. As a result, data science is shifting toward an agent-led, human-guided paradigm, in which analysts increasingly supervise agent-driven analyses, evaluate analytical decisions, and provide high-level guidance.

This shift introduces new challenges for efective human–agent collaboration [27, 56]. In traditional human-led analytics, analysts maintain a natural understanding of the analytical process because they directly perform and control its individual steps. In contrast, when agents autonomously carry out complex, multi-step analyses, potentially involving multiple collaborating agents [18, 54, 61], human analysts may no longer be directly involved in the decisions and operations that produce the results. To efectively supervise these processes, humans must be able to quickly understand what agents have done, evaluate their analytical decisions, and provide timely feedback to guide subsequent steps [45, 62]. Moreover, analysts may wish to intervene in ongoing analyses or revisit earlier analytical states to explore alternative directions. This calls for interaction mechanisms that let humans eficiently understand, evaluate, and steer agent-driven analyses while preserving agents’ autonomy. The Need for a Shared Computational Abstraction. A central design choice for such human–agent collaboration is the computational abstraction, an intermediate representation through which humans and agents conduct and interact with an analysis. Beyond expressing computational intent, this abstraction determines how analytical steps and their dependencies are exposed, how humans can inspect and modify agent-generated analyses, and how agents can reason about intermediate results and plan subsequent actions. It must support human comprehension and agent reasoning.

Existing computational abstractions ofer diferent trade-ofs. Natural language is accessible but often lacks precise semantics [57]. General-purpose programming languages, such as Python, are expressive and precise, but require analysts to understand program syntax, variables, and execution state to inspect or modify an analysis [48, 55]. SQL provides a higher-level declarative abstraction with well-defined relational semantics, but does not naturally expose intermediate steps as independently inspectable and editable components. These abstractions express computations efectively, but none is designed specifically for the iterative inspection, modification, and feedback that human–agent collaboration requires.

For example, consider a movie-analysis task involving two CSV files: movies.csv, containing movie titles, genres, ratings, and release years, and budgets.csv, containing production budgets. An analyst asks, “Which highly rated movies have below-average budgets compared with movies released in the same decade?” As illustrated in Figure 1, natural language expresses the analytical intent concisely but leaves important decisions unspecified, such as the rating threshold and how to handle missing budgets. Python can precisely implement the analysis by joining the files, grouping movies by release decade, computing average budgets, and filtering the results. However, understanding or modifying the agent-generated script requires the analyst to navigate the code and reason about intermediate variables and their dependencies.

![](images/de5f9f3f83aa839f5363cb5fe3369ea2510125dfff61b57b24ee6c0c48ff2da7.jpg)  
Figure 1: Three abstractions of the same movie-analysis task. Natural language leaves decisions unspecified; Python is precise but harder to understand and modify; dataflows expose operators, dependencies, and intermediate results for targeted feedback.

Dataflows as a Shared Computational Abstraction. We argue that dataflows, represented as directed acyclic graphs (DAGs) of declarative operators and data dependencies, provide an efective shared abstraction for human–agent collaboration. Each operator represents an analytical step, such as scanning, filtering, joining, or aggregating data, while directed edges capture the dependencies between these steps. By combining well-defined computational semantics with an intuitive visual representation, dataflows enable agents to construct and reason about analyses while allowing human analysts to understand, inspect, and modify them.

As illustrated in Figure 1, dataflows ofer three key benefits. First, precise: each operator explicitly specifies an analytical operation, such as filtering movies with ratings above a threshold or computing the average budget for each release decade. Second, intuitive: the visual representation exposes analytical steps and their dependencies, allowing analysts to follow the analysis without navigating program syntax or intermediate variables. Third, easier to give feedback: analysts can inspect intermediate results and modify individual operators to refine the analysis. For example, an analyst can examine the average budget for each decade and ask the agent to compare movies by genre instead, allowing the agent to revise the relevant operators while preserving the remaining analysis.

Importantly, the same abstraction can benefit agents as well. Operators, schemas, intermediate results, and data dependencies provide structured context for reasoning about the analysis. High level operator declarations can avoid repeatedly presenting lengthy implementation scripts to the LLM, and dependencies can help select relevant observations during later interactions. Our experi ments show that, compared with the script abstraction, the dataflow abstraction can improve task accuracy scores and reduce inference cost simultaneously in most settings, with the largest observed gains reaching double-digit points in accuracy score and over 2.8× in inference-cost eficiency. These benefits can improve performance in human–agent collaboration on data science tasks.

Building on this abstraction, we present BobFlow, a data science system that enables human analysts and LLM agents to construct, execute, inspect, and refine dataflows collaboratively. It grounds agent actions in declarative operators and integrates them with an execution engine, retained results, conversation history, and a visual interface. Our contributions are:

• Dataflow abstraction (Section 3). We identify key human– agent collaboration requirements and formalize dataflows with declarative operators and explicit data dependencies.

• Dataflow ReAct framework (Section 4). It lets agents incrementally construct, modify, and execute dataflows through iterative reasoning and actions.

• Human–agent interaction (Section 5). Our context tracker maintains conversation and dataflow snapshots for interruption and exploration from historical states.

• Dataflow-aware context optimizations (Section 6). For agent reasoning, we summarize intermediate results with samples and statistics and prune irrelevant historical observations using dataflow dependencies.

• Experimental evaluation (Section 7). We comprehensively evaluate BobFlow and the underlying dataflow abstraction with a user study and experiments comparing BobFlow with a baseline script-based agent across proprietary and open-weight LLMs.

## 2 PRELIMINARIES

## 2.1 Data Science Tasks and Computational Abstractions

A data science task is a natural-language instruction that asks a question or requests an analysis over given data. For example, a human analyst submits this task over two CSV files:

“Using movies.csv and rating.csv, which Comedy movies receive a 5-star rating?”

This task asks for Comedy movies with at least one 5-star rating.

To solve a data science task, an analyst uses a computational abstraction to declare data-processing steps, execute them on an engine, and store and retrieve their results. Example abstractions include Python scripts and SQL queries. With Python, the analyst writes code specifying the processing steps, and a Python interpreter executes it. Results are retained in variables or saved to files and loaded later. With SQL, the analyst writes a query specifying the desired result, and a query engine executes a physical plan to compute it. Results are returned or stored in a table for later queries.

## 2.2 LLM-Based Agents for Data Science

Similar to human analysts, LLM agents solve data science tasks by declaring analytical steps through a computational abstraction, requesting their execution, and examining the results. For a given abstraction, we describe LLM agents through their environment, actions, and action paradigm [47, 52], as shown in Figure 2.

![](images/15465071caac7dfda59ae71bee0c1eabb6b1da7c375f565b903cc2f7fc2e3f98.jpg)  
Figure 2: Human–agent collaboration on data science tasks.

Environment. The environment provides an engine for executing a declared analysis and storage for data and execution results. For example, a Python interpreter executes scripts [51], with data files and intermediate results stored in memory or on disk for retrieval. Actions. An agent interacts with the environment through prede fined actions. Each action specifies an operation and its arguments, such as a script to execute or a result to retrieve. The available actions and their formats are described to the LLM. The environment carries out each action and returns an observation, such as an execution result or an error message.

Action Paradigm. An action paradigm describes how an agent uses an LLM to reason about a task and generate actions. ReAct [60], a common paradigm for data science agents, interleaves reasoning with actions [14, 22, 23, 36, 58]. In each step, the agent provides an LLM with a context including the task, available action descriptions, and a history of earlier reasoning, actions, and observations. The LLM reasons over this context and generates actions to be executed by the environment. The agent appends the generated reasoning, actions, and returned observations to the history for the next step. The loop ends when the agent completes the task or reaches a step limit.

## 2.3 Human–Agent Collaboration

Human–agent collaboration is a powerful paradigm because data science tasks often require iterative refinement of the analysis [15, 50]. We identify three collaboration primitives in Figure 2.

• Action Primitives. Allow both the analyst and agent to create and modify data-processing steps, execute them, and inspect results. For example, the analyst can change the rating condition to include 4-star ratings and retrieve the matching movie titles.

• Conversation Primitives. Enable two-way communication: the analyst provides tasks, feedback, and questions, while the agent reports its reasoning, actions, observations, and answers.

• Control Primitives. Allow the analyst to interrupt the agent, inspect history and earlier results, and resume from a previous version in a new branch while preserving subsequent analyses. In this paper, we want to answer the following question:

What is the right abstraction to enable efective human–agent collaboration on data science tasks?

## 3 IDENTIFYING THE RIGHT ABSTRACTION

In this section, we identify the requirements for an efective computational abstraction and examine how dataflows support human– agent collaboration in data science.

## 3.1 Requirements for the Right Abstraction

Data science tasks involve declaring and executing sequences of data-processing steps, which can be expressed through diferent computational abstractions. For efective human–agent collaboration, we identify three requirements for the abstraction:

• Requirement 1: Precise Semantics. Each data-processing step should have well-defined semantics that unambiguously specify its operation, enabling agents to execute analyses consistently and analysts to verify alignment with their analytical objectives.

• Requirement 2: Intuitive Representation. The abstraction should represent analytical steps through high-level operations that analysts can easily understand and verify, without exposing low-level implementation details, while remaining concise for agents to generate and manipulate.

• Requirement 3: Step-Level Execution Transparency. Each analytical step should remain identifiable during execution, with intermediate results accessible to both analysts and agents. The abstraction should expose step dependencies, allowing analysts to inspect individual operations, provide targeted feedback, and iteratively refine the analysis.

Table 1: Abstractions for human–agent collaboration.
<table><tr><td>Abstraction</td><td>Precise Semantics</td><td>Intuitive s Representation</td><td>Step-Level Transparency</td></tr><tr><td>Natural Language</td><td>X</td><td>√</td><td>X</td></tr><tr><td>Programming Language</td><td>√</td><td>x</td><td>√</td></tr><tr><td>SQL</td><td>√</td><td>x</td><td>x</td></tr><tr><td>Dataflow</td><td>√</td><td>√</td><td>√</td></tr></table>

Table 1 compares common abstractions in terms of these requirements. Natural language lacks precise semantics due to its inherent ambiguity and potential inconsistency across interpretations, while programming languages expose physical implementation details. SQL provides a precise, declarative specification, but does not expose the intermediate result of each declared step. Moreover, the optimizer may substantially rewrite the physical plan, so even a small declaration change can invalidate the execution structure and its intermediate results, complicating step-by-step iteration.

These limitations motivate an abstraction with precise semantics, intuitive representation, and step-level execution transparency.

## 3.2 The Dataflow Abstraction

We propose dataflows as an abstraction satisfying these requirements. We next specify this abstraction and explain its benefits. Declarative Operators. We define a declarative operator (or “operator” for short) as a logical specification of a data-processing operation, independent of its physical implementation. Its inputs and outputs are structured tables, and its declaration specifies how to produce the outputs from the inputs. A source operator takes no input tables; its declaration instead specifies how to produce output tables by loading data from files or external sources.

At execution time, each declarative operator can map to one physical implementation, which may difer across dataflow engines while preserving its data-processing semantics. We define a set of declarative operators that capture common data-processing semantics, as shown in Table 2, along with example physical implementations in Pandas or PySpark. We also define a custom declarative operator to represent data-processing semantics not covered by the predefined operators in the above set. A custom operator specifies the intended processing logic with a generated description, while the dataflow engine registers its physical implementation separately. For example, custom("extract entities from text") describes an operator that extracts entities from its input text, without specifying how the entity extraction is implemented.

Table 2: Declarative operators and example implementations.
<table><tr><td>Declarative operator</td><td>Pandas</td><td>PySpark</td></tr><tr><td>scan({path}, {options})</td><td>pd.read_csv(...)</td><td>spark.read  $\overline { { . \mathsf { c s v } ( \ldots ) } }$ </td></tr><tr><td>filter({condition})</td><td> $\mathsf { q u e r y ( \ldots ) }$ </td><td> $\mathsf { f i l t e r } ( \ldots )$ </td></tr><tr><td>distinct({columns})</td><td>drop_duplicates(...)</td><td>dropDuplicates  $( \dots )$ </td></tr><tr><td>sort({columns}, {orders})</td><td>sort_values(...)</td><td>orderBy(...)</td></tr><tr><td>limit({offset}, {count})</td><td>iloc[...]</td><td>offset(...).limit(...)</td></tr><tr><td>project({mapping})</td><td>[[...]].rename(...)</td><td>select(...)</td></tr><tr><td>cast({types})</td><td>astype(...)</td><td>select(...)</td></tr><tr><td>join({keys}, {mode})</td><td>merge(...)</td><td>join(...)</td></tr><tr><td>union()</td><td>pd.concat(...)</td><td>unionByName(...)</td></tr><tr><td>aggregate({groups},</td><td>groupby(...)</td><td>groupBy(...)</td></tr><tr><td>{measures})</td><td> $. \mathsf { a g g } ( \ldots )$ </td><td> $. \mathsf { a g g } ( \ldots )$ </td></tr><tr><td>custom({description})</td><td> $\overline { { { \mathsf { d f } } \cdot { \mathsf { a p p l y } } ( \ldots ) } }$ </td><td> $\overline { { \mathrm { { u d f } ( \dots \dots ) } } }$ </td></tr></table>

Data Edges. Declarative operators are connected through directed data edges, which specify how one operator’s output supplies another operator’s input. Formally, an edge $e _ { i j } = ( v _ { i } , v _ { j } , \sigma )$ denotes that the output table of operator $v _ { i } ,$ , with schema � specifying its column names and types, is supplied to operator $v _ { j } .$ . The edge thus records a data dependency and the table’s schema.

Dataflow. A dataflow is a directed acyclic graph (DAG), denoted as $D = \left( V , E \right)$ , where � is a set of declarative operators and � is a set of data edges. An operator with no outgoing data edges is a sink. Figure 3 shows a dataflow with eight declarative operators for model training and inference. Operators $v _ { 1 } - v _ { 3 }$ load labeled text from train.csv, remove rows with missing labels, and train a text classifier. The output of $v _ { 2 }$ also feeds $v _ { 4 } ,$ , which counts training examples by label. Operators $v _ { 5 } - v _ { 7 }$ load test.json, select its text column, and remove rows with missing text. The inference operator $v _ { 8 }$ consumes this cleaned inference data and a one-cell table containing the trained model produced by $v _ { 3 } ,$ then outputs predicted labels. Operators $v _ { 4 }$ and $v _ { 8 }$ are sinks of this dataflow.

Dataflow Execution. To execute a dataflow, the engine maps each declarative operator to one or more physical operators, connected by physical edges that carry data. Together, they form a DAG called a physical plan. Figure 3 illustrates an example dataflow and its physical plan mapped by a dataflow engine. The dataflow abstraction is independent of the physical implementation’s programming language (e.g., Python, Java, or Scala) and execution model (e.g., operator-by-operator or pipelined execution). The engine must preserve the dataflow’s declared semantics and data dependencies regardless of implementation. The engine also collects runtime profiles for physical operators, including execution status, processing latency and rate, and data volumes and transfer rates. These profiles are linked to the corresponding declarative operators, letting analysts and agents monitor individual analytical steps.

![](images/cc18a6520fa53fab33dfe126eb3d22b3523b17f30cbb60571be0273cdb56d461.jpg)  
Figure 3: A dataflow for ML model training and inference, with a corresponding physical plan.

The engine retains output tables, including sink outputs, for later inspection. We define a dataflow execution as $X = ( D , R )$ , where � is the submitted dataflow and � contains execution results indexed by declarative operator. For each $v _ { i } \in V$ , its result $r _ { i } \in R$ includes output tables on success or error traces on failure, together with the available runtime profiles of its physical operators.

## 3.3 The Dataflow Abstraction in Data Science

We illustrate how dataflows support humans and agents in constructing, refining, executing, and inspecting data science analyses. Declaring and Refining an Analysis. Humans and agents express data-processing logic using the predefined declarative operators listed in Table 2 or custom operators with natural-language descriptions and separately registered physical implementations. They compose complex analyses by connecting operators through data edges, allowing branches to share upstream results and combine outputs using operators such as join and union. The resulting DAG makes data dependencies explicit, enabling humans and agents to understand individual steps by inspecting their operator declarations and tracing upstream dependencies. For example, understanding the label counts produced by $v _ { 4 }$ in Figure 3 requires examining $\upsilon _ { 1 }  \upsilon _ { 2 }  \upsilon _ { 4 }$ . Similarly, downstream dependencies reveal which results may be afected by an edit. Changing the filter condition in �<sub>2</sub> may afect label counts, model training, and predictions, while leaving the test-data preparation branch unafected. Specifying and Inspecting Execution. Humans and agents execute selected declarative operators along with their required upstream dependencies. During execution, they can monitor operatorlevel runtime profiles, including execution status, processing rates, and data-transfer rates and volumes. After execution, they can inspect result tables and retained runtime profiles to verify results and evaluate performance, or examine error traces to diagnose failures.

## 4 BOBFLOW: A DATAFLOW REACT SYSTEM

In this section, we present BobFlow, a system that uses dataflows as a shared abstraction for human–agent collaboration in data science.

## 4.1 System Overview

Figure 4 shows BobFlow’s four components: a data lake, dataflow engine, LLM agent, and dataflow graphical user interface (GUI). Data Lake. We use a data lake as the persistent storage layer of BobFlow. It stores the data for analysis, such as files in object storage or file systems. It also stores system states, including dataflow execution results and metadata. Storage options include Apache Iceberg [6] tables, DBMSs [49], and key-value stores [42].

Dataflow Engine. A dataflow is executed by an engine, such as Apache Texera<sup>1</sup> [7], Apache Hamilton<sup>1</sup> [5], Dagster [11], or Dask [41]. During execution, the engine reads source data from the data lake and stores intermediate results.

LLM Agent. Bob is an LLM agent that collaborates with analysts through conversations, each maintaining a shared dataflow and interaction history. It connects to LLMs through APIs and uses ReAct to solve tasks with the dataflow abstraction. A Dataflow MCP component, based on the Model Context Protocol (MCP) [1], executes LLM-generated actions, supporting dataflow inspection and editing, custom operator implementation management, dataflow execution, and result retrieval. A context tracker records conversation history and dataflow changes, enabling analysts to inspect reasoning and results, revisit earlier states, and branch from historical steps.

Dataflow GUI. The GUI supports human–agent collaboration on a shared dataflow and conversation. Its dataflow canvas lets analysts construct and execute dataflows, inspect operator runtime status and results, and select operators or edges for feedback to Bob. The chat interface lets them discuss tasks with Bob, review its reasoning, and provide feedback in natural language.

![](images/43d8be447c87ed49b566424185c52ea07fbb3c6a29120a7db32cef1e9a29f4a2.jpg)  
Figure 4: System architecture of BobFlow.  
<sup>1</sup>Apache Hamilton and Apache Texera are currently undergoing incubation at the Apache Software Foundation.

## 4.2 Solving a Task Using Dataflow ReAct

Solving a data science task requires multiple iterations in which an analyst and an agent reason about a dataflow and its execution results, revise the operators, and execute the updated dataflow. To support this iterative process, BobFlow adopts ReAct [60] and uses the dataflow abstraction for the agent’s reasoning and actions. We call this framework Dataflow ReAct. Figure 5 illustrates how the agent solves the task in Section 2.1 using Dataflow ReAct. Using this example, we describe a step’s three phases (Assemble, Think, and Act), then explain how the agent iterates to compute an answer.

![](images/edf84bfe1a943666a2a0f6bde86b10c1769380d629e20c0b29204d314892a954.jpg)  
Figure 5: Dataflow ReAct life cycle for a data science task.

Definition 4.1 (Agent Actions). We define an action as a primitive operation exposed by BobFlow that an agent invokes with specific arguments to interact with the system. Table 3 lists the supported action types, which allow the agent to inspect and modify the dataflow and custom operator implementations, execute dataflows, and retrieve execution results.

Table 3: Actions for solving tasks using dataflows.
<table><tr><td>Action</td><td>Specifies</td><td>Application</td></tr><tr><td>DescribeReasoning</td><td>Reasoning</td><td>Display reasoning in chat.</td></tr><tr><td>AnswerTask</td><td>Answer</td><td>Display answer in chat.</td></tr><tr><td>GetDataflow</td><td>Operators, edges</td><td>Inspect operators, schemas, and edges.</td></tr><tr><td>SetDataflow</td><td>Dataflow changes</td><td>Add, delete, or update operators and edges.</td></tr><tr><td>GetCode</td><td>υ ∈ V</td><td>Retrieve registered physical implementation.</td></tr><tr><td>SetCode</td><td>Code</td><td>Register implementation for custom operators.</td></tr><tr><td>ExecuteDataflow</td><td>D&#x27;⊆D</td><td>Execute the specified dataflow.</td></tr><tr><td>GetExecution</td><td>v ∈ V</td><td>Retrieve full results, samples, statistics, or errors.</td></tr></table>

Initialization. When the agent receives the first data science task � in a new conversation, it initializes an empty dataflow � to represent the analysis throughout the Dataflow ReAct loop. This dataflow is shared by the agent and the analyst, allowing both to inspect and revise the same analysis. To retain the analyst’s instructions and the actions carried out during the analysis, the agent maintains a shared conversation history �. We model � as a chronological se quence of entries and initialize it with the built-in system prompt $P ,$ followed by the task $\tau \colon N = [ P , \tau ]$ ]. The system prompt $P$ contains: A goal that directs the LLM to solve the given data science task using the dataflow abstraction.

Available resources that Bob can access, including the dataflow, the declarative operator specifications in Table 2, the dataflow engine, the data lake, and the available actions in Table 3.

End-to-end examples that show how the LLM can solve a task by reasoning over the current conversation and generating actions in the expected formats.

Phase 1 (Assemble): Preparing the Context. In this phase, Bob constructs context � from conversation history � so that the LLM can decide what actions to take next. For each step, $A \ =$ $[ a _ { 1 } , a _ { 2 } , \ldots , a _ { k } ]$ and $O = \left[ o _ { 1 } , o _ { 2 } , \ldots , o _ { k } \right]$ denote the actions and their corresponding observations. History $N$ records action–observation pairs $( a _ { i } , o _ { i } )$ chronologically:

$$
N = [ P , \tau , ( a _ { 1 } , o _ { 1 } ) , ( a _ { 2 } , o _ { 2 } ) , . . . ] .
$$

Throughout this section, we assume that each step’s context contains the full conversation history, i.e., $C = N$ . Note that the dataflow � is not included directly in $C ,$ since recorded actions sufice to reconstruct it. In Figure $5 , C = [ P , \tau ]$ at $T _ { 2 } ;$ at $T _ { 3 } ,$ , it also contains the actions and observations from the initial scans.

Phase 2 (Think): Reasoning over the Context. In this phase, Bob sends � to the LLM, which generates action sequence $A$ to advance the analysis. The LLM can include DescribeReasoning to communicate its reasoning to the analyst. In our running example, at $T _ { 2 } ,$ it plans to inspect the input schemas and data through scans $v _ { 1 }$ and $v _ { 2 }$ before constructing the analysis. It generates actions to create scan operators $v _ { 1 }$ and $v _ { 2 } ,$ execute both operators, and retrieve their results. Through DescribeReasoning, the LLM explains how the scans guide the search for Comedy movies with 5-star ratings. Phase 3 (Act): Carrying Out the Actions. Bob executes $A$ sequentially and appends each action–observation pair $\left( { { a _ { i } } , { o _ { i } } } \right)$ to � when $a _ { i }$ completes. Observations may include dataflow changes, execution status, or retrieved results. $\mathrm { A t } T _ { 2 }$ , Bob displays the reasoning in chat, creates and executes the two scans through Dataflow MCP, and retrieves their schemas, sample tuples, and runtime profiles. The schemas reveal a shared join key, movie\_id. The samples reveal an extra title row to skip in $v _ { 2 }$ and both single and pipe-separated genres in $v _ { 1 } ,$ , such as Drama and Comedy|Romance.

The Loop: Iterating Assemble–Think–Act Steps. Together, the Assemble, Think, and Act phases form one step, as shown in Figure 6. Across steps, Bob maintains the evolving dataflow and uses history � to assemble the next context. The loop ends when the agent applies AnswerTask or reaches the step limit.

In Figure $^ { 5 , }$ at $T _ { 3 } ,$ the LLM uses the initial scan results to plan data cleaning and filtering. The agent registers a custom implementation via SetCode to split and unnest movie genres and retain Comedy movies. It also updates scan $v _ { 2 }$ to skip the extra title row in rating.csv, adds filter $v _ { 3 }$ for 5-star ratings, and adds custom operator $v _ { 4 }$ to process the output of movie scan $v _ { 1 } .$ The agent then executes $v _ { 3 }$ and $v _ { 4 }$ and retrieves their results. $\mathrm { A t } T _ { 4 } ,$ the results reveal that � identifies movies by movie\_id without titles, while $v _ { 3 }$ $v _ { 4 }$ provides Comedy movies with titles. The agent joins their outputs on movie\_id using $v _ { 5 }$ and retrieves the titles. At $T _ { 5 } ,$ , the LLM generates an AnswerTask action based on the retrieved titles, which Bob applies to return the answer and terminate the loop.

## 5 HUMAN INTERACTIONS WITH THE AGENT

Human analysts may need to interact with the agent during a Dataflow ReAct loop, for example, to redirect the analysis, ask a question, or modify the dataflow by adding or editing an operator. BobFlow supports these interactions.

## 5.1 Capturing Human Interactions in Context

Definition 5.1 (Human Interactions). Human analysts can interact with Bob through two interfaces in BobFlow: a chat interface and a dataflow canvas. $\mathrm { O n }$ the chat interface, BobFlow allows analysts to send tasks to Bob, which starts a Dataflow ReAct loop to handle them. On the dataflow canvas, BobFlow allows analysts to perform the actions related to the dataflow in Table 3 on the shared dataflow $D .$ To handle these actions from the analyst, the canvas supports inspecting and editing operators and edges, while the dataflow engine handles execution requests and the data lake handles execution result retrieval. We call each task submission or canvas action a human interaction. As described in Section 4.2, the initial task � is stored directly in �. After handling a subsequent interaction, the system appends an entry $( u _ { i } , o _ { u _ { i } } )$ to the history $N ,$ where $u _ { i }$ records the submitted task or canvas action and $o _ { u _ { i } }$ records the resulting observation. The extended history is in the form

$$
N = [ P , \tau , ( a _ { 1 } , o _ { 1 } ) , ( u _ { 1 } , o _ { u _ { 1 } } ) , ( a _ { 2 } , o _ { 2 } ) , ( u _ { 2 } , o _ { u _ { 2 } } ) , \dots ] .
$$

This extension allows Bob to incorporate the analyst’s interactions into subsequent Dataflow ReAct loops.

Definition 5.2 (Context Snapshot). A context snapshot $s _ { i } = \left( N _ { i } , D _ { i } \right)$ logically captures the state of human–agent collaboration at a particular point in time. Its conversation snapshot $N _ { i }$ contains the history up to that point, and its dataflow snapshot $D _ { i }$ captures the corresponding state of the shared dataflow.

Context Tracker. To capture and manage the evolving context of human–agent collaboration, a context tracker maintains snapshots as a tree $\mathcal { H } = ( V _ { H } , E _ { H } )$ and tracks a current snapshot from which the analyst and agent continue their analysis. The root �<sub>0</sub> represents the initial state with conversation history $N _ { 0 } = [ P ]$ and an empty dataflow, and is set as the current snapshot when a conversation begins. Each completed human interaction or agent action advances the collaboration from the current snapshot �<sub>�</sub> to a new snapshot $s _ { j } ,$ adding $( s _ { i } , s _ { j } )$ to $E _ { H }$ and making $s _ { j }$ the new current snapshot. The edge $( s _ { i } , s _ { j } )$ records the conversation entry introduced by the human interaction or agent action together with its changes to the dataflow. Rather than materializing the complete $N _ { j }$ and $D _ { j }$ for every snapshot, the context tracker records these incremental changes along the edges of H.

![](images/4ae8b223b2fd0035cebb0ffa23a49b6743138eab81636536b6d4650f1bda45cf.jpg)  
Figure 6: Assemble, Think, and Act in Dataflow ReAct.

## 5.2 Commands to Enable Human Interactions

To enable human interactions throughout the analysis, BobFlow provides two commands: interrupt and jump. As illustrated in Fig ure 7, interrupt allows an analyst to stop an ongoing Dataflow ReAct loop and interact with the current snapshot, while jump allows the analyst to return to an earlier snapshot and interact from that point.

![](images/e5495b213e133cc7c3cd4d988f3586df11ab0f42046048e75df9bcd5c6365768.jpg)  
Figure 7: Human interactions in Dataflow ReAct.

Command 1: Interrupting the Dataflow ReAct Loop. During a Dataflow ReAct loop, the analyst can monitor Bob’s actions, evolv ing dataflow, and execution results through the GUI. To provide feedback, modify the dataflow, or redirect the analysis, the analyst can issue an interrupt command to the agent to stop the loop. During Assemble, Bob stops context preparation without modifying the conversation or dataflow. During Think, it discards the pending LLM response without applying its proposed actions. During Act, it cancels the ongoing action, discards uncommitted changes, and re stores the latest completed snapshot. In all cases, the context tracker ensures that the analyst and agent can resume from a consistent snapshot $s _ { i } = \left( N _ { i } , D _ { i } \right)$

For example, in Figure 7, the analyst decides to redirect the analysis while Bob is identifying Comedy movies with 5-star ratings. The interruption sequence illustrates how the analyst interrupts the Dataflow ReAct loop after $^ { s _ { 8 } , }$ while the agent is applying the first action $a _ { 1 }$ of the illustrated step. The agent cancels this action, discards its uncommitted changes, and terminates the loop. The context tracker restores the conversation history $N _ { 8 }$ and dataflow $D _ { 8 } ,$ , allowing the analyst to intervene from a consistent snapshot. The agent then reports that it is ready.

Command 2: Jumping to a Snapshot. An analyst may revisit an earlier snapshot to provide feedback, ask questions, inspect results, or redirect the analysis. BobFlow allows the analyst to jump to any snapshot in H when no Dataflow ReAct loop is running.

When the analyst selects a snapshot $s _ { i } ,$ the context tracker restores its conversation history $N _ { i }$ and dataflow $D _ { i }$ using the incremental changes recorded in H. This can be achieved by reconstructing the selected snapshot from the root $s _ { 0 }$ or reverting changes from the current snapshot. Optionally, checkpointing can also be used to accelerate restoration. The tracker then sets $s _ { i }$ as the current snapshot and restores the corresponding conversation and dataflow in the GUI. Subsequent interactions can create new branches without modifying existing analyses, allowing analysts to navigate between snapshots and explore alternative directions.

For example, after interrupting the loop at �<sub>8</sub> in Figure 7, the analyst jumps to �<sub>7</sub> to inspect the two scans and their results. The tracker restores $N _ { 7 }$ and $D _ { 7 } ,$ , after which the analyst submits a new task (�<sub>1</sub>) requesting each movie’s average rating. This creates �<sub>9</sub> as a child of $s _ { 7 }$ and starts a new branch. The agent continues from the restored context, computes average ratings using the existing scans, and returns the answer at $s _ { 1 9 } .$ Both branches share the history up to $s _ { 7 } ,$ , while the original analysis at $s _ { 8 }$ remains accessible.

## 6 DATAFLOW-AWARE CONTEXT OPTIMIZATIONS

So far, we have assumed that context $C$ preserves the full conversation history �. Context composition afects both decision accuracy and token cost: relevant information supports reasoning, while unnecessary information increases cost and may distract the LLM [31, 59]. We relax this assumption and use two dataflow properties to optimize �: data properties of operator outputs to improve result representation, and topology properties of operator dependencies to select relevant historical results for analyst interactions.

## 6.1 Context Optimization with Data

During the Dataflow ReAct loop, the LLM inspects execution results to understand the data, check whether operators produce the intended output, and decide which actions to take next. To inspect these results, the LLM generates GetExecution actions, each specifying an operator. Bob retrieves the requested results and records them as observations for the next step’s context. The challenge is to represent these results in context, preserving information needed for accurate decisions without excessive token costs.

We compare representations and their trade-ofs with this task:

“Using movies.csv and rating.csv, which five

Comedy movies have the highest average ratings?”

To solve this task, the LLM first creates and executes two scan operators to inspect the movie and rating files. Figure 8 compares representations of the movie scan’s output.

Baseline Approach. Including an operator’s full output in the context gives the LLM access to all records for identifying data patterns. In the running example, the movie scan returns 10,000 records with single-genre values such as Comedy and multi-genre values such as Comedy|Romance and Comedy|Drama, helping the LLM recognize that filtering for Comedy movies must account for multiple genres. However, the full output incurs high token costs and may exceed the LLM’s context window. Truncating it, for example, to 5,000 characters reduces its size but may hide important patterns, leading to incorrect assumptions about the data and inappropriate downstream operations. If only single-genre values remain, the LLM may apply filter(genre='Comedy'), incorrectly excluding multi-genre Comedy movies.

Optimization: Samples and Result Statistics. In the dataflow abstraction, each operator produces a structured table with a schema specifying its columns and their types. We refer to these properties as data properties, which enable us to represent the output table with sample tuples and result statistics. Sample tuples provide the LLM with examples of the output data. Result statistics cover row and column counts, duplicate and all-null row counts, and per-column null, non-null, and distinct counts. Type-specific statistics include numerical minima and maxima, the five most frequent string values and their counts, Boolean true and false counts, and earliest and latest datetime values. Together, samples and statistics provide a compact, informative view of the output data, helping the LLM understand execution results accurately with fewer input tokens.

In the running example, Figure 8 shows excerpts of an opti mized representation with 20 sample tuples and result statistics. The value distribution of the genre column reports 150 occurrences of Comedy|Romance and 120 of Comedy|Drama. These values help the LLM recognize that it should split and unnest the genre column before filtering for Comedy movies. This representation provides the information needed to specify the filter correctly at a lower token cost than including all 10,000 records.

BobFlow computes result statistics through the dataflow engine once an operator’s output table is complete. Using the table’s schema, the engine computes type-specific column statistics and row-level statistics in a full-table scan, then stores them alongside the output in the data lake. Samples are retrieved directly from the stored table without separate precomputation.

To support these representations, we extend the GetExecution action in the Dataflow ReAct framework to allow the LLM to request either sample tuples with statistics or the full execution result. Based on the request, Bob retrieves the corresponding data from the data lake and records it as the action’s observation.

## 6.2 Context Optimization with Topology

As described in Section 4.1, an analyst can inspect execution results and select operators on the dataflow canvas to ask Bob questions or provide feedback. Such interactions may require only part of the history. For example, while Bob identifies the five Comedy movies with the highest average ratings, the analyst selects $v _ { 3 } ,$ which splits and unnests movie genres, and asks why the same movie appears in multiple output rows. To answer, Bob can examine $\boldsymbol { v _ { 3 } } ^ { \prime } \boldsymbol { s }$ genre rows and the output of its upstream movie scan $v _ { 1 } ,$ as shown in Figure 9. Since GetExecution results are recorded as observations in the history, retaining relevant observations can give the LLM context needed to respond while reducing token cost and distractions.

![](images/2b839a7181ba7d4d05da10374bb9945c0f0c633d0da9787cb7351087fc6e198d.jpg)  
Figure 8: Full output versus samples and result statistics.

Optimization: Pruning Irrelevant Observations. The directed edges in a dataflow encode dependencies between operators. We refer to these structural relationships as topology properties, which provide a way to identify which observations to retain. The analyst’s selection identifies the operators whose results the interaction concerns, and Bob uses these dependencies to identify all their upstream operators. During the Assemble phase immediately after the interaction, Bob retains in � the observations associated with relevant actions and interactions, replacing irrelevant observations with minimal markers of their occurrence.

![](images/80d14a276f723b28c06e59ce1680a86cfdfaadfa33a2838e4ca0e2d289c61444.jpg)  
Figure 9: Pruning irrelevant observations for � at $v _ { 3 } ,$

Formally, let $D = ( V , E )$ be the dataflow at the time of an analyst interaction �, and let $V _ { u } ~ \subseteq ~ V$ be the nonempty set of operators selected by the analyst for this interaction. An operator �<sub>�</sub> ∈ � is upstream of a selected operator $v _ { j } \in V _ { u }$ if there is a directed path from $v _ { i }$ to $v _ { j }$ in �. We define $D _ { u }$ as the sub-DAG containing the selected operators, all their upstream operators, and the edges in � between these operators. To construct �, Bob first derives $D _ { u }$ from the analyst’s selected operators $V _ { u }$ . Starting with an empty context $C ,$ Bob traverses the conversation history � in chronological order. For each action–observation pair $( a _ { i } , o _ { i } )$ , if $a _ { i }$ is a GetExecution action, Bob checks whether its specified operator � belongs to $D _ { u } .$ If so, Bob retains $( a _ { i } , o _ { i } )$ in $C ;$ otherwise, it retains $a _ { i }$ and replaces $o _ { i }$ with a marker. All other history entries are appended to � unchanged. In the example, $D _ { u }$ contains the selected operator $v _ { 3 } ,$ , its upstream movie scan $v _ { 1 } .$ and their connecting edge, as shown in Figure 9. The context � retains their GetExecution observations and replaces those for the rating scan, Comedy filter, and join with minimal markers. Together with the task, reasoning, and actions, these observations allow the LLM to explain that splitting and unnesting the genre column produces a row for each of a movie’s genres.

Trade-of: Prompt Caching. Many LLM providers ofer prompt caching, which reuses an unchanged initial token sequence and charges a lower rate for cached input tokens [40]. Pruning modifies earlier observations and can shorten this reusable prefix, so more retained input tokens incur the uncached rate. Consequently, the additional cost of uncached tokens may outweigh the savings from reducing the context size. When earlier conversation history is no longer cached or prompt caching is disabled, pruning reduces uncached input tokens and can lower inference cost.

## 7 EXPERIMENTS

We evaluated the effectiveness of the dataflow abstraction as a shared medium for human–agent collaboration in data science. We conducted a user study to evaluate how BobFlow and a script-based agent support analysts in understanding, verifying, and guiding agent-generated analyses during task completion and debugging. For agents, we evaluated BobFlow’s task accuracy and inference cost on three benchmarks, comparing it with the script-based agent, ablating both context optimizations, and adding representative case analyses and latency and memory measurements in human interactions.

## 7.1 Experimental Setup

Benchmarks. We used three benchmarks for table preparation, end-to-end analysis, and dependent interactions. PrepBench [57] contains 306 tasks, each requiring 3–18 transformation steps, over 829 CSV files; we used its clarified requests to evaluate table prepa ration. KramaBench [29] contains 104 tasks over 1,764 files from 24 heterogeneous sources, evaluating end-to-end data discovery, preparation, integration, and analysis. LongDS-Bench [58] evaluates multi-turn analysis; we used v1.1 Lite [64], with 24 sessions and 777 interactions across six domains, to assess state reuse, historical revis iting, and alternative analyses. Each benchmark turn corresponds to an interaction in which the analyst submits a request to the agent. BobFlow Implementation. We built BobFlow based on Apache Texera [7, 53], an open-source dataflow system, extending its components to implement the architecture in Figure 4. At the GUI layer, we added an interaction panel to each operator for analyst feedback and rendered the snapshot tree for revisiting snapshots. We imple mented Dataflow ReAct and added the context tracker with the data-property and topology-property optimizations. We configured LiteLLM v1.83.10 [32] as the LLM API proxy. We built a Dataflow MCP component to let Bob apply dataflow actions to the dataflow engine Amber [28] and interact with the data lake. At the data-lake layer, we stored analyst files in MinIO [35], operator outputs as Apache Iceberg [6] tables, and system metadata in PostgreSQL [49]. Baseline and Experimental Environment. We used smolagents CodeAgent v1.23.0 [43], a Python-based ReAct agent, as a baseline (Script ReAct). In each step, the LLM generated an ExecuteCode ac tion containing Python code executed in a shared environment, with printed output recorded as an observation. The baseline pro vides a built-in chat interface for submitting tasks and checking the agent’s reasoning, generated code, and execution output. Script ReAct follows the same ReAct paradigm as Dataflow ReAct but uses scripts to express and execute analyses, enabling comparison of the two abstractions. Experiments ran on a server with an AMD Ryzen 9 7900X (12 cores, 24 threads), 31 GiB RAM, Ubuntu 20.04.5 LTS, and Python 3.12.12. We accessed the LLMs through APIs.

## 7.2 Dataflow’s Benefits for Human Analysts

We conducted a user study to evaluate how the dataflow abstraction helps analysts collaborate with agents, understand and verify their analyses, and guide the analytical process. We recruited 16 partici pants with varying levels of data science experience: six reported advanced experience, three intermediate, five beginner, and two no prior experience. Fourteen reported familiarity with at least one programming language. We asked participants to complete two human–agent collaboration scenarios using BobFlow and Script ReAct. We used Claude Haiku 5.5 [3] in both systems in this study. Scenario 1: Starting a Task from Scratch. This scenario examined whether dataflows can support human analysts in collaborating with agents end to end on a data science task. Participants were given an Olympic history dataset [19] and asked to collaborate with the agent on a task to identify the continents with the most gold medals. Participants 1–9 used Script ReAct, while participants

10–16 used BobFlow. By analyzing traces of participants using the agent, we found that the dataflow abstraction helped participants collaborate more efectively with the agent on solving the given task. As shown in Figure 10, all participants using BobFlow reached the correct answer with their first request, whereas a third of the participants using Script ReAct needed further requests, and the agent in BobFlow used fewer iterations and less model inference time per task.

![](images/e80fcebfba360557f698310c08c416c9a0be33763bf115d3f8ecb4186b9a65a3.jpg)

![](images/f3dbde47d9c0207988a46bf6ba9a3571bda583610dabae7f84633a1841ff1ba8.jpg)  
Figure 10: User study results for Scenarios 1 (S1) and 2 (S2).

![](images/eae5e0eeebb070ddabcc35bd655121aeb6c6c269c726725571dac036b7a5b223.jpg)

Scenario 2: Debugging an Existing Task. Next, we examined how dataflows support interaction with the agent on an existing analysis. In this scenario, for each system, we prepared a pipeline generated by an agent for the Inside Airbnb dataset for San Francisco [24] and introduced an equivalent price-parsing mistake, causing unexpectedly low average rental prices. Each participant received the pipeline corresponding to their assigned system: participants 1–9 used BobFlow with a given dataflow, while participants 10–16 used Script ReAct with a given script. Participants were asked to identify and correct the mistake by interacting with the corresponding agent, without prior knowledge of the underlying cause. Our system traces again showed that the dataflow abstraction made interaction with the agent more efective. As shown in Figure 10, participants identified the bug at comparable rates in both systems, while the agent in BobFlow handled each interaction with less than half the iterations and about a third of the model inference time. User Evaluations. To better understand participants’ experiences with each system, we asked them to complete a questionnaire using a five-point Likert scale and analyzed their responses. The results shown in Table 4 echoed the benefits of dataflows for human–agent collaboration. In particular, participants found it much easier to follow data lineage across analytical steps with BobFlow, rating this item 1.9 points higher than with Script ReAct. They also found it easier to locate problematic steps, with a 0.8-point improvement, helping them identify where interaction with the agent was needed.

Table 4: Participant ratings of intuitiveness, transparency, and control (1 = strongly disagree; 5 = strongly agree). Means are shown; bold indicates the more favorable score; Δ = BobFlow − Script ReAct.
<table><tr><td>Dimension</td><td>Questionnaire item</td><td>Script ReAct BobFlow</td><td></td><td>∆</td></tr><tr><td>Intuitiveness</td><td>Easy to collaborate with the agent</td><td>3.9</td><td>4.0</td><td>+0.1</td></tr><tr><td rowspan="7">Transparency</td><td>Easy to grasp processed/produced tables</td><td>3.5</td><td>4.3</td><td>+0.8</td></tr><tr><td>Confident about what each step does</td><td>3.3</td><td>3.8</td><td>+0.5</td></tr><tr><td>Easy to view each step&#x27;s output tables</td><td>2.6</td><td>4.2</td><td>+1.6</td></tr><tr><td>Easy to follow data lineage</td><td>2.3</td><td>4.2</td><td>+1.9</td></tr><tr><td>Clear final-answer vs. exploratory steps</td><td>2.9</td><td>4.0</td><td>+1.1</td></tr><tr><td>Easy to verify each step&#x27;s output tables</td><td>2.7</td><td>3.7</td><td>+1.0</td></tr><tr><td>Hard to verify the agent&#x27;s analysis</td><td>3.8</td><td>3.6</td><td>-0.2</td></tr><tr><td rowspan="3">Control</td><td>Easy to locate problematic steps</td><td>2.6</td><td>3.4</td><td>+0.8</td></tr><tr><td>Confident only requested edits occur</td><td>3.3</td><td>3.8</td><td>+0.5</td></tr><tr><td>Easy to verify revised answers</td><td>3.4</td><td>3.8</td><td>+0.4</td></tr></table>

## 7.3 Dataflow’s Benefits for Agents

To evaluate the dataflow abstraction’s benefits for agents, we compared Script ReAct with BobFlow on KramaBench and PrepBench using the eight proprietary and open-weight LLMs [4, 12, 16, 17, 37, 39] listed in Table 5. BobFlow enabled the data-property and topology-property optimizations described in Section 6. We computed accuracy scores using each benchmark’s evaluator and expressed them on a 0–100 scale. For PrepBench, correctness required all output tables to match their references under the oficial comparator, which checked row counts, key-aligned rows, and columnspecific matching rules [57]. For KramaBench, we ran Script ReAct and BobFlow three times and reported the mean accuracy from the KramaBench evaluator [29]. We calculated inference cost from per-call token usage at OpenAI, Anthropic, and OpenRouter prices, reporting cost and token totals per benchmark run.

Table 5: Task accuracy scores and inference cost (\$); bold indicates the better value (higher accuracy, lower cost).
<table><tr><td rowspan="2">Benchmark Agent</td><td rowspan="2"></td><td colspan="2">GPT-5.6 Luna</td><td colspan="2">GPT-5.6 Terra</td><td colspan="2">GPT-5.6 Sol</td><td colspan="2">GPT-6.1 Sol</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td rowspan="2">KramaBench</td><td>Script ReAct</td><td>62.3</td><td>0.40</td><td>64.3</td><td>6.99</td><td>68.9</td><td>10.22</td><td>78.9</td><td>3.22</td></tr><tr><td>BobFlow</td><td>76.4</td><td>0.30</td><td>72.3</td><td>2.76</td><td>76.4</td><td>5.09</td><td>78.4</td><td>1.67</td></tr><tr><td rowspan="2">PrepBench</td><td>Script ReAct</td><td>69.9</td><td>1.21</td><td>71.9</td><td>24.56</td><td>79.4</td><td>45.97</td><td>90.5</td><td>17.33</td></tr><tr><td>BobFlow</td><td>83.3</td><td>0.99</td><td>86.3</td><td>8.67</td><td>88.6</td><td>16.50</td><td>90.8</td><td>6.33</td></tr></table>

<table><tr><td rowspan="2">Benchmark Agent</td><td rowspan="2"></td><td colspan="2">Gemini 3.8 Flash</td><td colspan="2">Gemini 4 Argon</td><td colspan="2">DeepSeek V4.1-Flash</td><td colspan="2">Claude Opus 5.5</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td></tr><tr><td rowspan="2">KramaBench</td><td>Script ReAct</td><td>67.0</td><td>10.31</td><td>77.7</td><td>11.40</td><td>68.4</td><td>0.62</td><td>77.9</td><td>4.76</td></tr><tr><td>BobFlow</td><td>81.9</td><td>6.38</td><td>76.2</td><td>6.15</td><td>73.1</td><td>0.66</td><td>77.1</td><td>4.35</td></tr><tr><td rowspan="2">PrepBench</td><td>Script ReAct</td><td>70.9</td><td>37.05</td><td>88.6</td><td>33.41</td><td>85.6</td><td>1.40</td><td>90.2</td><td>17.76</td></tr><tr><td>BobFlow</td><td>89.5</td><td>13.22</td><td>91.5</td><td>15.42</td><td>88.9</td><td>1.31</td><td>90.5</td><td>16.89</td></tr></table>

As shown in Table 5, BobFlow improved accuracy by 0.3–18.6 points in 13 of the 16 model–benchmark settings and reduced inference cost by 4.9–64.7% in 15 settings compared with Script ReAct. With Gemini 3.8 Flash on PrepBench, BobFlow improved accuracy by 18.6 points while reducing inference cost by 64.3%. BobFlow also decreased accuracy by 0.5–1.5 points in some settings, and these decreases fall within the random noise of LLM outputs.

We broke down the input and output token usage of Script ReAct and BobFlow to understand the cost diference. As shown in Figure 11, BobFlow generated 40.2–77.3% fewer output tokens than Script ReAct for GPT and Gemini across both benchmarks. BobFlow also used 1.1–62.3% fewer input tokens than Script ReAct in most settings. On KramaBench with DeepSeek V4.1-Flash, BobFlow used 9.3% fewer input tokens but generated 19.8% more output tokens than Script ReAct, contributing to its 6.5% higher inference cost.

![](images/db37abe068a90a77352de30c374e2d3183ed2a945204f48c7eaca22fce2c894a.jpg)  
Figure 11: Input and output tokens per benchmark run.

## 7.4 Efect of Data-Property Optimization

We evaluated data-property optimization (“opt1”) by running PrepBench on BobFlow with and without opt1. As shown in Table 6, BobFlow reduced inference cost by 14.2–31.1% across all eight models, while accuracy changed by −0.7 to +2.2 points.

Table 6: Data-property optimization on PrepBench: BobFlow without opt1 → BobFlow. Accuracy changes are absolute; cost changes are relative.
<table><tr><td>Model</td><td>Accuracy</td><td>∆Acc.</td><td>Cost ($)</td><td>∆ Cost</td></tr><tr><td>GPT-5.6 Luna</td><td>82.4 → 83.3</td><td>+0.9</td><td>1.21 → 0.99</td><td>-18.2%</td></tr><tr><td>GPT-5.6 Terra</td><td>85.9 → 86.3</td><td>+0.4</td><td>10.66 → 8.67</td><td>-18.7%</td></tr><tr><td>GPT-5.6 Sol</td><td>88.2 → 88.6</td><td>+0.4</td><td>21.35 → 16.50</td><td>-22.7%</td></tr><tr><td>GPT-6.1 Sol</td><td>88.6 → 90.8</td><td>+2.2</td><td>8.39 → 6.33</td><td>-24.6%</td></tr><tr><td>Gemini 3.8 Flash</td><td>90.2 → 89.5</td><td>-0.7</td><td>16.19 → 13.22</td><td>-18.3%</td></tr><tr><td>Gemini 4 Argon</td><td>90.2 → 91.5</td><td>+1.3</td><td>17.97 → 15.42</td><td>-14.2%</td></tr><tr><td>DeepSeek V4.1-Flash</td><td>86.9 → 88.9</td><td>+2.0</td><td>1.90 → 1.31</td><td>-31.1%</td></tr><tr><td>Claude Opus 5.5</td><td>88.9 → 90.5</td><td>+1.6</td><td>24.48 → 16.89</td><td>-31.0%</td></tr></table>

We also compared engine execution times with and without opt1 during end-to-end task solving to measure the overhead of computing result statistics. For tasks processing more than 1 GB of input and producing large results, opt1 increased execution time by about 10% with essentially unchanged worker memory usage, motivating further work on eficient statistics computation.

## 7.5 Efect of Topology-Property Optimization

We evaluated topology-property optimization (“opt2”) during human–agent interactions for task solving. In the absence of a dedicated benchmark for this setting, we used LongDS-Bench, which evaluates multi-turn analysis, to simulate human interactions, treating each session as a task and its requests as analyst interactions. We evaluated BobFlow with and without opt2 using three models: GPT-5.6 Luna, GPT-5.6 Terra, and DeepSeek V4.1-Flash.

During human–agent collaboration, analysts may continue an analysis while earlier prompts remain cached or resume it after the cache expires. Therefore, we considered two conditions for reusing cached prompts across interactions. In the warm-start condition, earlier prompts could be reused, and costs reflected the recorded cache usage. In the cold-start condition, earlier prompts were assumed to be unavailable; we estimated costs by charging the first model call of each interaction at uncached input rates while retaining recorded cache usage for subsequent calls.

Table 7: LongDS-Bench v1.1 Lite results; mean tokens in thousands per interaction.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Configuration</td><td rowspan="2">Accuracy</td><td colspan="2">Input Output</td><td colspan="2">Cost ($)</td></tr><tr><td>(K)</td><td>(K)</td><td></td><td>Cold Warm</td></tr><tr><td rowspan="2">GPT-5.6 Luna</td><td>BobFlow without opt2</td><td>42.9</td><td>126</td><td>1.4</td><td>9.73</td><td>3.54</td></tr><tr><td>BobFlow</td><td>42.9</td><td>54</td><td>1.9</td><td>5.43</td><td>4.85</td></tr><tr><td rowspan="2">GPT-5.6 Terra</td><td>BobFlow without opt2</td><td>54.6</td><td>99</td><td>1.2</td><td>88.33</td><td>29.52</td></tr><tr><td>BobFlow</td><td>58.4</td><td>45</td><td>1.5</td><td>49.37</td><td>43.47</td></tr><tr><td>DeepSeek</td><td>BobFlow without opt2</td><td>58.0</td><td>239</td><td>4.0</td><td>9.87</td><td>2.57</td></tr><tr><td>V4.1-Flash BobFlow</td><td></td><td>58.9</td><td>163</td><td>10.2</td><td>8.00</td><td>7.28</td></tr></table>

![](images/d58202aabe7dd90d68d3502c9ef4a466b2bdeb23caf3630ada5ba32f9de24ea3.jpg)  
Figure 12: Simplified traces of four KramaBench cases comparing BobFlow and Script ReAct.

![](images/1ca2aef5ce2f4a6c689fb8fc82c62e41de96ab77a3bd4c46623d96059745c545.jpg)  
Figure 13: Median context size (top) and cache-miss rate (bottom) versus the number of preceding interactions.

Table 7 compares accuracy, mean token usage, and total inference cost across all 777 interactions. Compared with BobFlow without opt2, BobFlow maintained or improved accuracy by up to 3.8 points and reduced input-token usage by 31.7–56.9%. According to the traces, after pruning, the agent sometimes generated additional actions to inspect earlier results and relevant parts of the dataflow. These additional actions contributed to the higher output-token usage with opt2. Opt2 also reduced estimated cold-start cost by 18.9–44.2% but increased warm-start cost by 37.0–182.9%.

Context Size Across Interactions. To examine how opt2 reduced input-token usage, we measured context size as the input length of the first model call for each interaction. The upper panel of Figure 13 reports the median across 22 sessions with at least 30 interactions; the horizontal axis counts preceding interactions. Opt2 limited context growth as interactions accumulated, reducing context size by 63.5–69.1% across the three models at the start of interaction 30. Trade-of with Prompt Caching. We observed the trade-of discussed in Section 6.2: opt2 reduced estimated inference cost when earlier prompts were not cached, but raised it when they could be reused from the cache. As shown in the lower panel of Figure 13, the cache-miss rate, the fraction of input tokens not served from the cache, rose at interaction 30 from 0.8–1.3% without opt2 to 20.2–42.9% with opt2. The higher cost of uncached inputs ofset savings from smaller contexts. DeepSeek also generated 2.5 times as many output tokens with opt2, further increasing its cost. These results suggest enabling opt2 to reduce inference cost when the cache expires or cannot be reused after a model switch, and leaving it disabled while the same model can reuse cached history.

## 7.6 Case Studies of the Two Abstractions

We studied LLM traces to see how agents’ use of the dataflow abstraction afected accuracy and cost. The four KramaBench cases in Figure 12 help explain the results in Section 7.3.

Case 1: BobFlow Answered the Task Correctly. This task asked for the average age of patients whose serous tumor samples were analyzed in a study. Both agents read the Case\_excluded column, but Script ReAct ignored it in subsequent computations and returned an incorrect average. BobFlow kept it in an intermediate table and used it to exclude flagged samples, answering correctly.

Observation 1. BobFlow helped the agent accurately reuse earlier results   
from the context, improving answer accuracy.

Case 2: Script ReAct Answered the Task Correctly. This task asked for the total amount defrauded across payment methods. BobFlow used a custom operator whose parser skipped amounts containing commas. The operator returned only the final sum, and BobFlow accepted this plausible but incorrect result without inspecting the parsed values. Script ReAct handled these amounts correctly and printed the parsed values, obtaining the correct total.

Observation 2. BobFlow let the agent trust plausible results despite insuficient runtime inspection, reducing accuracy.

Case 3: BobFlow Solved the Task at Lower Cost. This task asked which U.S. military branch had the largest median fraud loss. Script ReAct generated code using assumed column names; after the code failed, it inspected the table and corrected the code. BobFlow first inspected the table and used its observed structure to generate the analysis. Both agents answered correctly, but BobFlow avoided the failed attempt and reduced inference cost.

Observation 3. BobFlow inspected data first, reducing debugging cost. Case 4: Script ReAct Solved the Task at Lower Cost. This task asked for the ratio of peak atmospheric mass density during two periods. Before computing the ratio, BobFlow gathered large table previews, whereas Script ReAct inspected only a few rows. Both agents answered the question correctly, but the larger observations increased BobFlow’s input-token usage and inference cost.

Observation 4. BobFlow’s excessive previews raised cost, not accuracy.

## 7.7 Human Interaction Support

We evaluated how eficiently BobFlow maintains interaction histories and restores earlier analysis states as interactions accumulate. Memory Utilization. We replayed 207 recorded KramaBench conversations in one session to measure memory growth as their histories accumulated. After garbage collection, we measured JavaScript heap and external-bufer growth for BobFlow and Python memory for Script ReAct. Both measurements excluded initial memory, execution-engine memory, and Python execution variables. As shown in Figure 14, BobFlow’s history memory grew by 110 MiB after 1,239 interactions, roughly 95% less than Script ReAct’s 2.3 GiB. Script ReAct retained copies of model inputs at each step, while BobFlow recorded incremental changes.

![](images/98caeac1e6f09a1fafcfa21311da95d100f84737cf0fb68cfffd403431a5193e.jpg)  
Figure 14: Memory growth from interaction 100 onward.

Jump Latency and Tracking Overhead. To measure jump latency as the snapshot tree grew, we replayed the same 207 conversations and jumped to 50 uniformly sampled snapshots at intervals of about 250 snapshots. We measured each jump command’s end-toend latency, including restoration of the target snapshot. As shown in Figure 15 (left), median latency was 1.2 ms across 1,150 jumps and remained below 1.9 ms in each 500-snapshot interval.

![](images/526ff88571778ee7010cf65feaf34ace04e86eb9044270c6eed311e6c3ba7250.jpg)

![](images/877942d1fdea495ed8045785b10637736adee7a77f15cbf307ce0d67d77b881f.jpg)  
Figure 15: Jump latency (left) and memory with and without tracking (right) in replays of over 5,000 snapshots.

Using the same snapshots, we measured tracking overhead by comparing heap memory with and without the tracker. At 5,095 snapshots, tracking added 4.1 MiB of overhead (Figure 15, right).

## 8 RELATED WORK

ReAct Framework. ReAct [60] enables LLM agents to solve diverse tasks by interleaving reasoning and actions, with observations guiding later steps. For example, SWE-agent [59] resolves software issues by navigating repositories, editing code, and running tests; WebVoyager [20] completes web tasks through reasoning over webpage observations and browser actions; and ChemCrow [9] plans and executes syntheses with specialized chemistry tools. We design BobFlow as a Dataflow ReAct system for solving data science tasks. Agentic Systems for Data Processing. As LLMs improve in reasoning and tool use, more agentic systems are being built for data science tasks. Python scripts or SQL queries commonly serve as their computational abstraction. For example, CodeAct [51], DA-Agent [23], and DA-Studio [33] use Python, while Chat2Data [63] and SiriusBI [25] generate SQL queries. BobFlow instead formalizes dataflows as a computational abstraction for data science agents.

Another line of work optimizes how agents solve specific data-processing tasks. For data preparation, AutoPrep [13] and DataFlow [30] coordinate specialized agents to construct pipelines, while DeepPrep [14] uses execution-guided tree search to explore and revise them. With their emphasis on optimization, these systems focus on automated task solving. In BobFlow, we emphasize analyst–agent interaction throughout a task and explore optimization in this setting.

Human–Agent Collaboration in Data Science. Human–agent collaboration has become a focus of many proprietary systems and research prototypes, with interfaces for analysts often centered on dialogue or code. Dialogue-oriented interfaces in Claude Science [2], OpenAI’s in-house data agent [38], SiriusBI [25], and DataChat [26] let analysts express analytical requests in natural language. Code-oriented interfaces expose analytical steps for inspection and revision through generated code, notebooks, or diagrams derived from code. For example, Stepwise [27] and Chat-Pipe [10] expose analytical steps and generated code. Through notebooks, DocWrangler [46] lets analysts annotate outputs and refine document-processing pipelines, while Carnot [44] lets them inspect, modify, and execute operator DAGs. WaitGPT [56] presents interactive operation diagrams from code. Compared with these systems, BobFlow centers interaction on a shared visual dataflow that analysts and agents can inspect and modify, with a chat interface for conversation and context management. Systems such as Flowco [15] and DataFlow-Harness [21] use executable dataflow graphs to help analysts construct and revise analyses with agents. Compared with these works, we also study how agents can use dataflow properties to construct and optimize their context during collaboration. Our demonstration paper [8] showcased BobFlow’s interface and scenarios, whereas this paper presents the dataflow abstraction, Dataflow ReAct, the context optimizations, and their evaluation.

## 9 CONCLUSION

In this paper, we identify requirements for human–agent collaboration in data science and formalize dataflows as a shared abstraction. Dataflow ReAct in BobFlow lets agents iteratively construct, modify, and execute dataflows to solve data science tasks. We support human interactions by tracking conversation and dataflow snapshots for interruption and exploration. We optimize agent context with data properties and prune irrelevant observations using topology properties. Our user study shows that dataflows make collaboration and interaction with agents more efective and intuitive. Our experiments demonstrate that dataflows can substantially improve task accuracy while reducing inference cost in most evaluated settings. These findings support dataflows as a shared abstraction for human–agent collaboration.

## ACKNOWLEDGMENTS

We thank Dr. Xi Lu for her help. We thank the participants in our user study. This work was supported by NSF awards 2200274 and 2609582, NIH NIDDK award 2U24DK097771-11, and an SAP grant.

## REFERENCES

[1] Anthropic. 2024. Introducing the Model Context Protocol. https://www. anthropic.com/news/model-context-protocol Accessed October 18, 2026.

[2] Anthropic. 2026. Claude Science, an AI workbench for scientists, is now avail able. https://www.anthropic.com/news/claude-science-ai-workbench Accessed November 1, 2026.

[3] Anthropic. 2026. Introducing Claude Haiku 5.5. https://www.anthropic.com claude-haiku-5-5 Accessed November 1, 2026.

[4] Anthropic. 2026. Introducing Claude Opus 5.5. https://www.anthropic.com/ claude-opus-5-5 Accessed November 1, 2026.

[5] Apache Software Foundation. [n.d.]. Apache Hamilton. https://hamilton.apache. org/ Accessed November 1, 2026.

[6] Apache Software Foundation. [n.d.]. Apache Iceberg. https://iceberg.apache.org/ Accessed September 28, 2026.

[7] Apache Software Foundation. [n.d.]. Apache Texera. https://texera.apache.org/ Accessed October 31, 2026.

[8] Jiadong Bai, Yicong Huang, and Chen Li. 2026. Demonstration of BobFlow: Human-Agent Collaboration Using Dataflows on Data Science Tasks. Proc. VLDB Endow. 19, 12 (2026), 4718–4721. https://doi.org/10.14778/3827998.3828105

[9] Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. 2024. Augmenting large language models with chemistry tools. Nature Machine Intelligence 6, 5 (2024), 525–535. https://doi.org/10.1038/ s42256-024-00832-8

[10] Sibei Chen, Hanbing Liu, Waiting Jin, Xiangyu Sun, Xiaoyao Feng, Ju Fan, Xiaoyong Du, and Nan Tang. 2024. ChatPipe: Orchestrating Data Preparation Pipelines by Optimizing Human-ChatGPT Interactions. In Companion of the 2024 International Conference on Management ofData (SIGMOD/PODS 2024). ACM, 484–487. https://doi.org/10.1145/3626246.3654727

[11] Dagster Labs. [n.d.]. Dagster Documentation. https://docs.dagster.io/ Accessed September 29, 2026.

[12] DeepSeek. 2026. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash Accessed October 16, 2026.

[13] Meihao Fan, Ju Fan, Nan Tang, Lei Cao, Guoliang Li, and Xiaoyong Du. 2025. AutoPrep: Natural Language Question-Aware Data Preparation with a Multi-Agent Framework. Proc. VLDB Endow. 18, 10 (2025), 3504–3517. https://doi.org/ 10.14778/3748191.3748211

[14] Meihao Fan, Ju Fan, Yuxin Zhang, Shaolei Zhang, Xiaoyong Du, Jie Song, Peng Li, Fuxin Jiang, Tieying Zhang, and Jianjun Chen. 2026. DeepPrep: An LLM Powered Agentic System for Autonomous Data Preparation. Proc. VLDB Endow. 19, 11 (2026), 3371–3384. https://doi.org/10.14778/3836663.3836695

[15] Stephen N. Freund, Brooke Simon, Emery D. Berger, and EuniceJun. 2025. Flowco: Mixed-Initiative Authoring of Reliable End-to-End Data Analyses via Dataflow Graphs and LLMs. In Proceedings ofthe 38th Annual ACM Symposium on User Interface Software and Technology. ACM, 1–20. https://doi.org/10.1145/3746059. 3747636

[16] Google. 2026. Gemini 4 Argon: Built to Solve Your Hardest Problems. https://blog.google/innovation-and-ai/models-and-research/geminimodels/introducing-gemini-4-argon/ Accessed November 1, 2026.

[17] Google. 2026. Introducing Gemini 3.8 Flash and 3.8 Flash Cyber. https://blog.google/innovation-and-ai/models-and-research/gemini-models/3- 8-flash-and-3-8-flash-cyber/ Accessed November 1, 2026.

[18] Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, et al. 2026. Accelerating scientific discovery with Co-Scientist. Nature 655, 8122 (2026), 487–496. https://doi.org/10.1038/s41586-026-10644-y

[19] Randi Grifin. 2018. 120 Years of Olympic History: Athletes and Results. Kaggle. https://www.kaggle.com/datasets/heesoo37/120-years-of-olympic-historyathletes-and-results Accessed October 28, 2026.

[20] Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. 2024. WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, 6864–6890. https://doi.org/10.18653/ v1/2024.acl-long.371

[21] Runming He, Zhen Hao Wong, Hao Liang, Zimo Meng, Chengyu Shen, Xiaochen Ma, and Wentao Zhang. 2026. DataFlow-Harness: A Grounded Code-Agent Platform for Constructing Editable LLM Data Pipelines. https://doi.org/10. 48550/arXiv.2607.16617 arXiv:2607.16617 [cs.SE]

[22] Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Robert Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Yongxin Ni, Zhibin Gou, Zongze Xu, Yuyu Luo, and Chenglin Wu. 2025. Data Interpreter: An LLM Agent for Data Science. In Findings ofthe Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025 (Findings of ACL), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, 19796–19821. https:

//doi.org/10.18653/v1/2025.findings-acl.1016

[23] Yiming Huang,Jianwen Luo, Yan Yu, Yitong Zhang, Fangyu Lei, Yifan Wei, Shizhu He, Lifu Huang, Xiao Liu, Jun Zhao, and Kang Liu. 2024. DA-Code: Agent Data Science Code Generation Benchmark for Large Language Models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, 13487–13521. https://doi.org/10.18653/V1/2024.EMNLP-MAIN.748

[24] Inside Airbnb. 2026. San Francisco Listings, Reviews, and Neighbourhoods. https://insideairbnb.com/get-the-data/ September 14, 2026 snapshot. Accessed October 28, 2026.

[25] Jie Jiang, Haining Xie, Siqi Shen, Yu Shen, Zihan Zhang, Meng Lei, Yifeng Zheng, Yang Li, Chunyou Li, Danqing Huang, Yinjun Wu, Wentao Zhang, Bin Cui, and Peng Chen. 2025. SiriusBI: A Comprehensive LLM-Powered Solution for Data Analytics in Business Intelligence. Proc. VLDB Endow. 18, 12 (2025), 4860–4873. https://doi.org/10.14778/3750601.3750610

[26] Rogers Jefrey Leo John, Dylan Bacon, Junda Chen, Ushmal Ramesh, Jiatong Li, Deepan Das, Robert V. Claus, Amos Kendall, and Jignesh M. Patel. 2023. DataChat: An Intuitive and Collaborative Data Analytics Platform. In Companion ofthe 2023 International Conference on Management ofData, SIGMOD/PODS 2023, Seattle, WA, USA, June 18-23, 2023, Sudipto Das, Ippokratis Pandis, K. Selçuk Candan, and Sihem Amer-Yahia (Eds.). ACM, 203–215. https://doi.org/10.1145/3555041. 3589678

[27] Majeed Kazemitabaar, Jack Williams, Ian Drosos, Tovi Grossman, Austin Zachary Henley, Carina Negreanu, and Advait Sarkar. 2024. Improving Steering and Verification in AI-Assisted Data Analysis with Interactive Task Decomposition. In Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology, UIST 2024, Pittsburgh, PA, USA, October 13-16, 2024, Lining Yao, Mayank Goel, Alexandra Ion, and Pedro Lopes (Eds.). ACM, 92:1–92:19. https: //doi.org/10.1145/3654777.3676345

[28] Avinash Kumar, Zuozhi Wang, Shengquan Ni, and Chen Li. 2020. Amber: A Debuggable Dataflow System Based on the Actor Model. Proc. VLDB Endow. 13, 5 (2020), 740–753. https://doi.org/10.14778/3377369.3377381

[29] Eugenie Lai, Gerardo Vitagliano, Ziyu Zhang, Om Chabra, Sivaprasad Sudhir, Anna Zeng, Anton A. Zabreyko, Chenning Li, Ferdi Kossmann, Jialin Ding, Jun Chen, Markos Markakis, Matthew Russo, Weiyang Wang, Ziniu Wu, Michael J. Cafarella, Lei Cao, Samuel Madden, and Tim Kraska. 2026. KramaBench: A Benchmark for AI Systems on Data-to-Insight Pipelines over Data Lakes. In The Fourteenth International Conference on Learning Representations, ICLR 2026. https://openreview.net/forum?id=fZfUdeCC5X

[30] Hao Liang, Xiaochen Ma, Zhou Liu, Zhen Hao Wong, Zhengyang Zhao, Zimo Meng, Runming He, Chengyu Shen, Qifeng Cai, Zhaoyang Han, Meiyi Qiang, Yalin Feng, Tianyi Bai, Zewei Pan, Ziyi Guo, Yizhen Jiang, Jingwen Deng, Qijie You, Peichao Lai, Tianyu Guo, Chi Hsu Tsai, Hengyi Feng, Rui Hu, Wenkai Yu, Junbo Niu, Bohan Zeng, Ruichuan An, Lu Ma, Jihao Huang, Yaowei Zheng, Conghui He, Linpeng Tang, Bin Cui, Weinan E, and Wentao Zhang. 2025. DataFlow: An LLM-Driven Framework for Unified Data Preparation and Workflow Automation in the Era of Data-Centric AI. https://doi.org/10.48550/arXiv.2512.16676 arXiv:2512.16676 [cs.LG]

[31] Tobias Lindenbauer, Igor Slinko, Ludwig Felder, Egor Bogomolov, and Yaroslav Zharov. 2025. The Complexity Trap: Simple Observation Masking Is as Eficient as LLM Summarization for Agent Context Management. https://doi.org/10. 48550/arXiv.2508.21433 arXiv:2508.21433 [cs.SE]

[32] LiteLLM. [n.d.]. LiteLLM AI Gateway. https://docs.litellm.ai/docs/simple\_proxy Accessed October 31, 2026.

[33] Yizhe Liu, Shaolei Zhang, and Ju Fan. 2026. DA-Studio: An Agentic System for End-to-End Data Analysis. Proc. VLDB Endow. 19, 12 (2026), 4766–4769. https://doi.org/10.14778/3827998.3828117

[34] Yuyu Luo, Guoliang Li, Ju Fan, and Nan Tang. 2026. Data Agents: Levels, State of the Art, and Open Problems. In Companion ofthe 2026 International Conference on Management ofData (SIGMOD/PODS 2026). ACM, 571–579. https://doi.org/ 10.1145/3788853.3801878

[35] MinIO. [n.d.]. MinIO. https://www.min.io/ Accessed November 1, 2026.

[36] Jaehyun Nam, Jinsung Yoon, Jiefeng Chen, Raj Sinha, Jinwoo Shin, and Tomas Pfister. 2025. DS-STAR: Data Science Agent for Solving Diverse Tasks across Heterogeneous Formats and Open-Ended Queries. https://doi.org/10.48550/ arXiv.2509.21825 arXiv:2509.21825 [cs.AI]

[37] OpenAI. 2026. GPT-5.6: Frontier Intelligence That Scales with Your Ambition. https://openai.com/index/gpt-5-6/ Accessed October 16, 2026.

[38] OpenAI. 2026. Inside OpenAI’s in-house data agent. https://openai.com/index/ inside-our-in-house-data-agent/ Accessed November 1, 2026.

[39] OpenAI. 2026. Introducing GPT-6.1 Sol. https://openai.com/index/introducinggpt-6-1-sol/ Accessed November 1, 2026.

[40] OpenAI. 2026. Prompt Caching. https://developers.openai.com/api/docs/guides/ prompt-caching Accessed September 13, 2026.

[41] Matthew Rocklin. 2015. Dask: Parallel Computation with Blocked algorithms and Task Scheduling. In Proceedings ofthe 14th Python in Science Conference 2015 (SciPy 2015), Austin, Texas, July 6 - 12, 2015, Kathryn Huf and James Bergstra

(Eds.). scipy.org, 126–132. https://doi.org/10.25080/MAJORA-7B98E3ED-013

[42] RocksDB. [n.d.]. RocksDB: A Persistent Key-Value Store. https://rocksdb.org Accessed September 28, 2026.

[43] Aymeric Roucher, Albert Villanova del Moral, Thomas Wolf, Leandro von Werra, and Erik Kaunismäki. 2025. smolagents: a smol library to build great agentic systems. https://github.com/huggingface/smolagents

[44] Matthew Russo, Yash Agarwal, Tianyu Li, Zhuohan Gu, Michael Cafarella, Omar Khattab, Tim Kraska, and Samuel Madden. 2026. Carnot: Interpretable, Interactive, and Optimized Execution of Deep Research Queries. Proc. VLDB Endow. 19, 12 (2026), 4642–4645. https://doi.org/10.14778/3827998.3828086

[45] Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Michael Moor, Zicheng Liu, and Emad Barsoum. 2025. Agent Laboratory: Using LLM Agents as Research Assistants. In Findings ofthe Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 5977–6043. https://doi.org/10.18653/v1/2025.findings-emnlp.320

[46] Shreya Shankar, Bhavya Chopra, Mawil Hasan, Stephen Lee, Björn Hartmann, Joseph M. Hellerstein, Aditya G. Parameswaran, and Eugene Wu. 2025. Steering Semantic Data Processing With DocWrangler. In Proceedings ofthe 38th Annual ACM Symposium on User Interface Software and Technology. Association for Computing Machinery, 84:1–84:18. https://doi.org/10.1145/3746059.3747625

[47] Theodore Sumers, Shunyu Yao, Karthik R. Narasimhan, and Thomas L. Grifiths. 2024. Cognitive Architectures for Language Agents. Transactions on Machine Learning Research (2024). https://openreview.net/forum?id=1i6ZCvflQJ

[48] Alexander K. Taylor, Yicong Huang, Junheng Hao, Xinyuan Lin, Xiusi Chen, Wei Wang, and Chen Li. 2024. Data Science Tasks Implemented with Scripts versus GUI-Based Workflows: The Good, the Bad, and the Ugly. In 40th International Conference on Data Engineering, ICDE 2024 - Workshops, Utrecht, Netherlands, May 13-16, 2024. IEEE, 267–277. https://doi.org/10.1109/ICDEW61823.2024.00040

[49] The PostgreSQL Global Development Group. [n.d.]. About PostgreSQL. https: //www.postgresql.org/about/ Accessed September 28, 2026.

[50] Chenglong Wang, Bongshin Lee, Steven M. Drucker, Dan Marshall, and Jianfeng Gao. 2025. Data Formulator 2: Iterative Creation of Data Visualizations, with AI Transforming Data Along the Way. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. ACM, Article 677, 17 pages. https: //doi.org/10.1145/3706598.3713296

[51] Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. 2024. Executable Code Actions Elicit Better LLM Agents. In Proceedings ofthe 41st International Conference on Machine Learning (Proceedings ofMachine Learning Research), Vol. 235. PMLR, 50208–50232. https://proceedings.mlr.press/ v235/wang24h.html

[52] Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang, Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H. Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas Muennighof, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao Peng, Heng Ji, and Graham Neubig. 2025. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. In The Thirteenth International Conference on Learning Representations, ICLR 2025. https://arxiv.org/abs/2407.16741

[53] Zuozhi Wang, Yicong Huang, Shengquan Ni, Avinash Kumar, Sadeem Alsudais, Xiaozhen Liu, Xinyuan Lin, Yunyan Ding, and Chen Li. 2024. Texera: A System for Collaborative and Interactive Data Analytics Using Workflows. Proc. VLDB

Endow. 17, 11 (2024), 3580–3588. https://doi.org/10.14778/3681954.3682022

[54] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. 2024. AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations. In Proceedings of the First Conference on Language Modeling. https://openreview.net/forum?id=BAakY1hNKS

[55] Yifan Wu, Joseph M. Hellerstein, and Arvind Satyanarayan. 2020. B2: Bridging Code and Interactive Visualization in Computational Notebooks. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology. ACM, 152–165. https://doi.org/10.1145/3379337.3415851

[56] Liwenhan Xie, Chengbo Zheng, Haijun Xia, Huamin Qu, and Chen Zhu-Tian. 2024. WaitGPT: Monitoring and Steering Conversational LLM Agent in Data Analysis with On-the-Fly Code Visualization. In Proceedings ofthe 37th Annual ACM Symposium on User Interface Software and Technology. ACM, 1–14. https: //doi.org/10.1145/3654777.3676374

[57] Jingzhe Xu, Rui Wang, Jiannan Wang, and Guoliang Li. 2026. PrepBench: How Far Are We from Natural-Language-Driven Data Preparation? Proc. VLDB Endow. 19, 10 (2026), 2866–2879. https://doi.org/10.14778/3828612.3828638

[58] Kewei Xu, Xiaoben Lu, Shuofei Qiao, Zihan Ding, Haoming Xu, Lei Liang, and Ningyu Zhang. 2026. LongDS-Bench: On the Failure of Long-Horizon Agentic Data Analysis. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 13351– 13383. https://doi.org/10.18653/2026.emnlp-main.584

[59] John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. 2024. SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. In Advances in Neural Information Processing Systems, Vol. 37. https://doi.org/10.52202/079017-1601

[60] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net. https://openreview.net/forum?id=WE\_vluYUL-X

[61] Ziming You, Yumiao Zhang, Dexuan Xu, Yiwei Lou, Yandong Yan, Wei Wang, Huamin Zhang, and Yu Huang. 2025. DatawiseAgent: A Notebook-Centric LLM Agent Framework for Adaptive and Robust Data Science Automation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 1099–1123. https://doi. org/10.18653/v1/2025.emnlp-main.58

[62] Qiuhai Zeng, Claire Jin, Xinyue Wang, Yuhan Zheng, and Qunhua Li. 2025. AIRepr: An Analyst-Inspector Framework for Evaluating Reproducibility of LLMs in Data Science. In Findings of the Association for Computational Linguistics: EMNLP 2025. Association for Computational Linguistics, 10170–10201. https: //doi.org/10.18653/v1/2025.findings-emnlp.539

[63] Xinyang Zhao, Xuanhe Zhou, and Guoliang Li. 2024. Chat2Data: An Interactive Data Analysis System with RAG, Vector Databases and LLMs. Proc. VLDB Endow. 17, 12 (2024), 4481–4484. https://doi.org/10.14778/3685800.3685905

[64] ZJUNLP. 2026. LongDS-Bench. https://github.com/zjunlp/DataMind/blob/main/ longds/README.md Version 1.1 Lite. Accessed October 16, 2026.