# Application of curiosity driven exploration methods for hardware interference identification

Ludovic Matar <sup>1</sup>, Cl´ement Moulin-Frier<sup>1</sup>,Pierre-Yves Oudeyer<sup>1</sup>

<sup>1</sup>Flowers AI & CogSci Lab - National Institute for Research in Digital Science and Technology, Bordeaux, France {ludovic.matar, clement.moulin-frier,pierre-yves.oudeyer}@inria.fr

## Abstract

The transition from single-core to multi-core architectures in safety-critical embedded systems introduces significant challenges due to inter-core interference caused by contention for shared hardware resources. Such interference afects execution times and complicates the verification of strict temporal requirements, particularly in domains such as avionics where standards require comprehensive identification of interference sources. Existing interference analysis approaches, whether manual or modelbased, struggle to capture the full range of behaviors arising from the complex interactions among microarchitectural components. In this paper, we frame multi-core interference analysis as the exploration of a complex system behavior space. We propose the use of curiosity-driven exploration algorithms from artificial intelligence to systematically and eficiently cover the space of possible interference behaviors. Using a simulator-based environment, we show that the proposed approach achieves broader and more uniform behavioral coverage within a limited experimental budget compared to traditional pseudorandom program generation methods.

Keywords: Machine learning, Real-time application, Interference Analysis, Curiosity-Driven Learning

## 1 Introduction

The shift from single-core to multi-core architectures is essential in safety-critical embedded systems in multiple domains, such as aerospace and automotive, driven by both the need to enhance processor performance for increasingly demanding applications and adaptation to recent technology. However, this transition introduces new complexities due to tasks running in parallel competing for shared resources; particularly, hardware contention issues known as inter-core interference. As the occurrence of interference has a consequence on the execution time, the use of multi-core processors in critical real-time systems represents a challenge. Indeed, temporal requirements with high levels of confidence have to be met. In the avionics domain, the AMC 20-193 standard requires identifying all sources of interference that can occur when executing tasks in parallel and addressing them.

![](images/36d3666b869524c6c9cd6103bcc6d309d63f2200e4974b7051622db0b3482872.jpg)  
Figure 1: Common computer memory hierarchy [18]

Systems on chips face a general trade-of between the necessary time to access information stored in a memory resource and their storage capacity. As an element is closer to the CPU, its access time and storage capacity are smaller. This principle is illustrated in Figure 1. Because they are directly on the CPU, registers typically serve information almost instantly. Their storage capacity is very small, and only essential information is stored. When a core treats a memory instruction, it requests the main memory by default. However, the main memory often operates orders of magnitude slower than the CPU. Thus the cache, namely random static memory locates between the registers and the main memory. It has larger latencies than the registers, but remains considerably faster than main memory. Moreover, it has a higher storage capacity than the registers but smaller than the main memory. It stores areas of the main memory that are the most likely to be referenced. The main memory consists of DRAM cells that ofer a great storage capacity. Therefore, there may be a noticeable slowdown each time the CPU needs to access the main memory. As a consequence of this organization, competition for accessing stored data and instructions within shared caches and shared memory between CPU cores increases the execution time, and thus creates potential interference channels.

Definition 1 Interference is a phenomenon such that, for identical initial conditions, the execution time of an application $S _ { 1 }$ running in isolation $( S _ { 1 } , \ L _ { - } )$ on a platform difers from the execution time of $S _ { 1 }$ running with an application $S _ { 2 }$ on the platform $( S _ { 1 } , S _ { 2 } )$ .

Interference arises from contention on shared resources, e.g., when multiple cores compete to allocate information in cache or memory. Another case is the competition for accessing a shared bus, resulting in bandwidth limitations and delays in data transfer. To summarize, we can distinguish distinct categories of interference occurring because of contention on shared resources [20]:

• Cache contention: When multiple cores compete for space in shared caches, evicting each other’s data and causing cache misses.

• Memory contention: When multiple cores access a shared memory simultaneously (such as the main memory), leading to a bottleneck due to limited memory bandwidth.

• Bus contention: When cores compete for access to a shared bus, resulting in bandwidth limitations and delays in data transfer.

• I/O Contention: When multiple processes try to perform I/O operations at the same time, leading to delays and reduced I/O performance.

As there are multiple types of interference, there exist various types of interference sources to discover. In the following, we will briefly present existing methods that tackle real-time system challenges.

Various methods have been proposed to identify mechanisms that are sources of interference [14, 6, 8]. Classic interference analysis methods are either manual and empirical approaches or model-based approaches. The first ones involve manually analyzing the hardware documentation or further characterizing the hardware through experiments and measurements [3, 13, 22, 8, 16]. The others involve identifying interference from a model of the system through two principles: (1) capturing the hardwarelevel knowledge about the system to produce a model, and (2) using an analyzer tool which identifies the interference channels from the model [6, 2, 5]. Classical approaches to interference identification rely heavily on analyzing hardware datasheets, which are often large [17], complex, incomplete, or inconsistent. This makes the process dificult and error-prone, leading to uncertain results and possibly missed interference channels. This can lead, for example, to creating oversimplified models or test cases, resulting in missed interference scenarios or channels [7].

While existing approaches have enabled better anticipation of potential sources of interference, they remain limited in their ability to cover the entire space of possible behaviors of multi-core processor systems. Interference patterns arise from the complex non-linear interactions between a large number of micro-architectural components. In consequence, they are extremely sensitive to small variations in the concurrent programs that are being executed, which strongly limits the ability to predict the behavior of the systems analytically. Moreover, programs that are generated with pseudo-random procedures are unlikely to cover the entire space of possible behaviors.

In this sense, multi-core architectures can be considered complex systems.

A complex system is a system consisting of multiple entities interacting non-linearly with each other, giving rise to emergent phenomena that are dificult to predict [19]. Examples of complex systems in Nature are water molecules crystallizing into snowflakes, collective animal behavior forming complex patterns (e.g., bird flocks), or chains of amino acids forming complex 3D protein structures. To study such systems, scientists often need to build a model mapping their input parameters (e.g., a sequence of amino acids) to their corresponding observed behavior (e.g., the resulting 3D shape of a protein). Complex systems often sufer from so-called butterfly efects, where two slightly diferent initial conditions do not necessarily induce similar outcomes. Such systems also usu ally exhibit attractor efects, that is, a tendency of the system to evolve towards a particular set of states. Understanding and predicting the behavior of complex systems often requires uncovering the diversity of behaviors they can produce. This is usually a challenging problem, in particular when the parameter space is high-dimensional and when the mapping from the parameter space to the behavior space is highly non-linear. In particular, random exploration of the parameter space is usually inefi cient for uncovering the diversity of potential behaviors in most complex systems.

Recently, contributions in AI have proposed algorithms able to eficiently cover the space of behaviors of any complex system [9], namely, curiosity-driven exploration algorithms, which belong to the family of automated discovery algorithms. These algorithms have proven to be very eficient at uncovering a wide diversity of behaviors in many complex systems from diferent scientific domains, including computer science [10], physics [12], chemistry [15] and biology [11]. In this paper, we propose, for the first time to our knowledge, to apply these algorithms to the problem of discovering interference patterns in multicore processor systems. For this aim, we use a multi-core processor simulator, enabling us to prototype our methods without having to deal with the complexity of a real hardware architecture.

We show that our proposed method enables better discovery of the space of possible behaviors of the system with a limited experimental budget, compared to more standard methods based on the pseudo-random generation of programs.

• In section section motivation, we recall general principles regarding interference.

• Then in section 2, we highlight previous related AI works on real-time systems.

• In section 4 we unfold our exploration strategy step by step.

• In section 5, we present a summary of our findings, with results of diversity measures.

## 2 Related works

The interest in AI techniques within the real-time computing community is relatively recent. Although studies tackling interference on embedded platforms using AIbased techniques exist, to our knowledge, no studies have explicitly targeted their use to identify sources of interference. Nevertheless, we will review the key works in these related areas and explore how they might be adapted or inspire solutions for our problem.

Several works have focused on the use of Machine Learning to estimate Worst-Case Execution Times, whether for single-core platforms [1] or multi-core ones [4]. An interesting work closer to our objective is the Kryptonite approach [21], introducing a framework that synthesizes a maximally interfering environment for a Program Under Test (PUT) executing on a multi-core platform. In other words, the Worst-Case Program Interference is targeted. The approach relies on a set of code snippets, called gadgets, to hammer specific shared resources in the multi-core platform and create interference. Kryptonite then arranges the gadgets in order to maximize interference in two phases: firstly, a greedy approach is used to iteratively build a sequence of gadgets that increases execution times of the PUT. And secondly, a Reinforcement Learning (RL) algorithm is used to finetune the interfering environment. Although the objective is not the same, this approach motivates the idea of using an automated discovery algorithm on a multi-core platform to explore interference as it requires many interactions with the system. Indeed, a complex system is exploited, and the objective of reaching an optimum in an output space by exploring a code space is fulfilled. In contrast with an optimization method that minimizes a loss function, we propose a space coverage technique to induce a large diversity of behaviors in a given complex system. In the next section, we present our basic simulator upon which we will perform our exploration.

## 3 Description of our dual-core model

We justify the implementation of the simulator and briefly describe components and their behaviors. A minimum of two cores with shared memory components is, by definition, necessary to induce interference phenomena. Thus, our basic simulator model contains:

• two cores,

• their respective private L1 caches,

• a shared L2 cache unit and a DDR memory with its controller,

• an interconnect model that transmits instructions sent by the cache to the memory.

![](images/2b1b22d5c3a2a59c38291f326a1a0d963087ab538491b137327aa9ab2185f09b.jpg)  
Figure 2: Simulated dual-core architecture used in this paper. Each core has a private L1 cache and shares an L2 cache with the other. The L2 cache is connected to the main DDR memory via its DDR memory controller.

With such a configuration, interference arises in shared components, such as the L2 cache, the DDR memory controller, and the interconnect. In the following, we explain the functioning of some of the basic blocks of our simulator.

## 3.1 Core model

As interference is mostly due to memory mechanisms, the simulation of cores handles exclusively memory access instructions; that is, the instruction set is reduced to read and write operations in memory since, ultimately, only memory accesses are simulated. In the following, we denote the ”read” operations by RD and the ”write” operations by WR.

In our algorithms, for each core, a simple loop models the ”fetch and execute” cycle. At each CPU cycle, the core can:

1. wait for the end of an RD operation,

2. execute an RD,

3. execute a WR,

4. execute an instruction that does not perform memory accesses, which, in the case of the simulator, amounts to doing nothing.

As in real-life applications, we make use of a set of virtual addresses that maps to the set of physical addresses, that is, elements of the micro-architecture memory. Within our programs, virtual addresses are simply modeled as numbers. The function that associates a virtual address to a location in the DDR memory is given in A.

Within the pairs, the programs do not exchange data to be written or read, but are to be executable and interpretable; moreover, this data is not modeled as it does not play any role in interference phenomena, as opposed to instructions. Programs consist of RD and

WR at given addresses. The shipment is done via the private L1 caches, as instructions are not stored in either the cache or the memory. Thus, programs are concretized as simple dictionaries with the format {cycle:(type,address)} as follows. For each instruction in a program, its key is the cycle at which the considered instruction is sent to the simplified simulator.

Listing 1: Example of simplified assembly programs. program\_0 = {23:( ’ read ’, 11) ,37:( ’ read ’ ,17) ,49:( ’ write ’ ,6)} program\_1 = {12:( ’ write ’, 24) ,18:( ’ read ’ ,39) ,32:( ’ write ’ ,37)}

The maximum length of a program and the largest cycle for which an instruction is sent are fixed as parameters. This model is obviously very simplified compared to the actual operation of a physical target device. Waiting for the completion of a read memory access to continue code execution does not correspond to reality because the processor has various mechanisms specifically allowing it to hide these latencies. This wait is only necessary to preserve true data access dependencies (e.g., a program WR @x RD @x must be executed in this order and be preserved at run time to maintain program semantics).

## 3.2 Cache model

Each cache level is configurable in total size, cache line size, and associativity. By default, the behavior is of the ”write-back” type, that is, writing to lower-level memory occurs when a cache line is evicted. During the write operation, the line is simply marked as ”dirty” and is written to the lower-level memory during its eviction. It is also possible to employ a ”write-through” behavior in which writing to the lower-level memory is done immediately. The management of the eviction of cache lines is carried out by a PLRU whose role is to determine the line to be replaced based on the current state of the cache. Ideally, we would like to implement a behavior of the LRU (Least Recently Used) type, which would consist of eliminating the line used least recently in order to make the most of the locality principle <sup>1</sup>. However, this strategy is expensive to implement and we often prefer to use a simpler mechanism called Pseudo-LRU, which relies on a binary tree.

Thus, this algorithm includes a function that maintains the data structure with a binary tree and which allows choosing the next row to be evicted based on memory accesses. It also includes a function allowing the choice of the next cache row to be evicted using the information contained in the binary tree. The last level cache (L2 in figure 4) is shared by both memory hierarchies. There is no cache coherency management mechanism.

![](images/c68b1f8c84b3e3d6acad58e4fab15c7ef976378777c2bb42acff078d83f1b51a.jpg)  
Figure 3: Simplified state diagram assuming a single bank [14]

## 3.2.1 DDR model

Let’s recall that DRAM is organized in rows and columns of banks. Each bank contains an additional row called a row bufer. When information is accessed, its row is stored in the row bufer.

In our model, the memory consists of several banks, each of which contains a row bufer acting as a cache. Latencies are modeled and play an important role in the interference delay in the following way:

• Access to diferent rows from the same bank induces significant delays because of the temporal cost of changing rows in the bufer.

• Simultaneous accesses to distinct rows from the same bank induce larger temporal costs than simultaneous accesses to distinct banks. In other words, intrabank interference induces larger delays than interbank interference.

Each bank is managed by a timed state machine that reproduces the one described in [14] (see figure 3). The state of the banks of the DDR determines the completion dates of operations. Requests are issued by the DDR controller. Let’s note PRE as the operation that deactivates an open row of a given bank; this action writes the selected information from the row bufer to the memory. And ACT is the operation that activates a closed row by storing the information from the given row in the bufer. Additional information on the DDR is given in A.

## 3.3 DDR controller model

The memory controller is in charge of reordering memory access requests to maximize the memory access rate. It implements several queues: a queue for commands, a queue for read data, and a queue for the data being written. Prioritization rules are modeled. Indeed, the RD are prioritized over the WR in order to minimize waiting times, while preserving memory consistency (WR @x RD @x must be executed in this order). Instructions targeting lines that are already in row bufers are prioritized in order to avoid the penalty related to completing a PRE=>ACT sequence. RD and WR are processed in ”batches” to avoid the overhead related to the transition between RD and WR. Mechanisms of our DDR controller are described in A.1.

## 4 Overview of the exploration strategy

![](images/e012ee7692e6d233cc33870b87abd8270794ab739f997c51ecc0f4343789f317.jpg)  
Figure 4: Illustration of a complex system associated with multi-core hardware. On the left side of the figure, pairs of simplified assembly programs are represented in the parameter space Θ. Each of them is executed on the platform and corresponds to an outcome (behavior) on the right-hand side, representing the behavior space B. In our case, both the parameter space and the behavior space can be high-dimensional spaces, although in general, the parameter space has a larger dimension than the behavior space.

The Intrinsically Motivated Goal Exploration Process is an instance of automated discovery algorithms. This diversity-driven strategy aims to maximally cover a behavior space. During its exploration, a discovery agent iteratively samples goals uniformly in the behavior space and ofers appropriate parameters to reach them by leveraging previously found (parameters, outcome) relations. The diversity maximization is a side efect of the goal exploration process. The exploration consists of several episodes called ”experiments,” during which the agent continually improves its ability to discover new singularities.

For interference identification, the considered complex system is the simulated hardware platform. In contrast to Kryptonite, for example, which attempts to minimize/- maximize interference delays, an IMGEP instead aims to discover a large diversity of possible interference patterns. We wish to explore a behavior space B that characterizes interference patterns within the hardware by manipulating inputs from a parameter space Θ; see Figure 4. In our case, we establish a parameter space made of simplified assembly programs. It is essential, on the one hand, to formally model basic elements of the exploration, such as the parameter space Θ and the behavior space B, and on the other hand, to specify the agent’s internal models.

Definition 2 We call a ”pair of programs” a couple of two programs of code $S _ { 1 }$ and $S _ { 2 }$ allocated respectively to the $C _ { 1 }$ and $C _ { 2 }$ cores of an execution platform. $S _ { 1 }$ and $S _ { 2 }$ are independent code programs. In particular, they do not exchange data.

Definition 3 We call an ”experiment” the interaction with our complex system. It involves the parallel execution of the programs $S _ { 1 }$ and $S _ { 2 }$ , and the execution of $S _ { 1 }$ and $S _ { 2 }$ alone. All experiments are to be independent; this involves, for example, clearing the cache, DDR, and interconnect before every execution.

From an IMGEP perspective, an element of a behavior space is seen as a goal to achieve; thus, goals are target behaviors in the behavior space. In our experiments, a goal is a vector encoding a targeted interference pattern. The discovery agent samples goals to solve and uses its strategy to solve them. Two internal models are used for these respective aspects: one for the goal generation strategy ${ \mathcal { G } } ,$ and a goal strategy achievement model Π. As illustrated in Figure 5, the agent fills a database H to update its internal models during the exploration, and thus the acquired data is reused to extract potential solutions to solve other goals.

An IMGEP is a loop algorithm that iterates online by interacting with the considered complex system. The algorithm starts with a warming set of $N _ { \mathrm { i n i t } }$ iterations. This helps the exploration to escape poor knowledge regions.

This type of architecture highlights previous independent experiments stored in the database H. Using efficient sampling methods, this algorithm allows one to quickly reach a great diversity of outcomes in the behavior space. Moreover, its usage of the database prevents forgetting during exploration.

## 4.1 Parameter space

A pair of programs is required to run each experiment. As presented in Figure 5, the discovery agent generates a pair $\theta \in \Theta$ at each iteration. Following the general program constraints discussed in part 3.1, we recall that most sources of interference are due to memory mechanisms; therefore, we choose to produce pairs of simplified assembly instructions to constitute a parameter space Θ. Indeed, as presented before, only memory instructions

![](images/293d7e05689d32db7b3a8262f0c53e3ce946cbb825351d5d529ff953723a21ec.jpg)  
Figure 5: Schematic view of an IMGEP for our case of interference identification, based on the general description of IMGEP B.

RD and WR are modeled. To get closer to a real-life application, we can model two distinct sets of addresses for the cores, i.e., core 0: 0,20 and core 1: 21,40. As two distinct addresses can map to the same physical address, it will still be possible to observe interference.

With the maximum program length and largest shipment cycle being carefully chosen, we synthesize a parameter space of a reasonable cardinality to thus have a better understanding of the results.

## 4.2 Behavior space

All microarchitectural mechanisms are known, as the simulator is a white box. We wish to identify the ones responsible for interference. A set of relevant performance counters provides building blocks for the behavior space; see table 3.

<table><tr><td rowspan=1 colspan=1>Performance counter</td><td rowspan=1 colspan=1>Category</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>Processor cycles</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>Number of executed cycles</td></tr><tr><td rowspan=1 colspan=1>Instructions completed</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>Number of completed instructions</td></tr><tr><td rowspan=1 colspan=1>Decode stalled</td><td rowspan=1 colspan=1>General</td><td rowspan=1 colspan=1>Number of cycles in a waiting status</td></tr><tr><td rowspan=1 colspan=1>Cache misses</td><td rowspan=1 colspan=1>Cache</td><td rowspan=1 colspan=1>Number of cache misses L1/L2</td></tr><tr><td rowspan=1 colspan=1>Cache store allocates</td><td rowspan=1 colspan=1>Cache</td><td rowspan=1 colspan=1>Number of line allocations in L1/L2</td></tr><tr><td rowspan=1 colspan=1>Cache demand access</td><td rowspan=1 colspan=1>Cache</td><td rowspan=1 colspan=1>Number of requests for L1/L2</td></tr><tr><td rowspan=1 colspan=1>DDR store misses</td><td rowspan=1 colspan=1>DDR</td><td rowspan=1 colspan=1>Number of DDR stores</td></tr><tr><td rowspan=1 colspan=1>DDR load misses</td><td rowspan=1 colspan=1>DDR</td><td rowspan=1 colspan=1>Number of misses</td></tr><tr><td rowspan=1 colspan=1>DDR demand access</td><td rowspan=1 colspan=1>DDR</td><td rowspan=1 colspan=1>Number of requests</td></tr><tr><td rowspan=1 colspan=1>DDR store allocates</td><td rowspan=1 colspan=1>DDR</td><td rowspan=1 colspan=1>Number of line allocations</td></tr></table>

Table 1: Set of available performance counters in the dual-core simulator

Such performance counters can be clock cycles, row misses, instruction types, branch mispredictions, and the number of stalls. Indeed, the phenomena quantified by those performance counters allow for the description of conflicts occurring in shared resources and thus are directly related to interference. In general, information on the platform will always be needed in a context where interference is being addressed. If a platform is a black box, all the requirements of the certification file cannot be fulfilled.

Each element of the established behavior space is perceived as a potential goal for the discovery agent. A goal is defined as a pair $\boldsymbol { g } = \left( z _ { g } , \mathcal { L } _ { g } \right)$ with an embedding $z _ { g } \in B$ and an evaluation function $\mathcal { L } _ { g } = \mathcal { D } ( \cdot , z _ { g } )$ . Ideally, B has ”good properties” such as being a low-dimensional metric space. The metric property is important for the goal achievement strategy of the discovery agent. Indeed, one wants to evaluate the distances between some previous experiment outcomes z and a goal embedding $z _ { g }$ . For example: $\mathcal { L } _ { g } ( z ) \ : = \ : | | z - z _ { g } | | _ { 2 } .$ This aspect is essential for designing an eficient goal achievement strategy, as one wants to eficiently select previous experiment outcomes and their associated parameters that are potentially suited for the generation of new candidate parameters.

Because interference arises following microarchitectural behavior conflicts, we target shared resource performance counter diferences in values between the executions in isolation (iso) and the executions in non-isolation (non-iso). Given a pair of programs, we calculate for each selected performance counter its diference when executing in isolation on one core and in non-isolation on both cores. The features in table 2 constitute the behavior space B.

<table><tr><td>Features</td><td>Category</td><td colspan="6">Description</td></tr><tr><td>Processor cycles diff core</td><td>General</td><td>Difference</td><td>[nb of executed cycles]</td><td>[non-iso]</td><td></td><td>[nb of executed cycles]</td><td>[iso.]</td></tr><tr><td>Processor mutual</td><td>General</td><td>Difference</td><td>[nb of executed cycles]</td><td>[non-iso.</td><td>core 0 –</td><td></td><td>[nb of executed cycles[non-iso core 1]</td></tr><tr><td>Cache write misses</td><td>L2 Cache</td><td>Difference</td><td>[nb of L2 cache misses</td><td>[non-iso]</td><td></td><td>[nb of L2 cache misses]</td><td>][iso.]</td></tr><tr><td>Cache write hits</td><td>L2 Cache</td><td>Difference</td><td>[nb of L2 cache hits][non-iso]</td><td>1</td><td></td><td>[nb of L2 cache hits][iso.</td><td></td></tr><tr><td>Cache read misses</td><td>L2 Cache</td><td>Difference</td><td>[nb of L2 cache misses][non-iso]</td><td></td><td>1</td><td>[nb of L2 cache misses[iso.]</td><td></td></tr><tr><td>Cache read hits</td><td>L2 Cache</td><td>Difference</td><td>[nb of L2 cache hits] [non-iso]</td><td></td><td></td><td>[nb of L2 cache hits[[iso.</td><td></td></tr><tr><td>DDR misses</td><td>DDR</td><td>Difference</td><td>[nb of misses][non-iso]</td><td></td><td>[nb of misses]</td><td>[[iso.] for each pair (bank, row)</td><td></td></tr><tr><td>DDR hits</td><td>DDR</td><td>Difference</td><td>[nb of hits] [[non-iso]</td><td>[nb of hits][iso.]</td><td></td><td>for each pair (bank, row)</td><td></td></tr></table>

Table 2: Features constituting the behavior space B. For any performance counter C, we calculate two features $F _ { C }$ as non-iso. vs. iso. diferences $F _ { C } \equiv C [ \mathrm { n o n - i s o } ] - C [ \mathrm { i s o } . ]$ $\mathrm { ^ { 5 9 } i s o . } ^ { \mathrm { 5 9 } }$ corresponds to an execution with only one of the two cores, 0 or 1.

With $n _ { \mathrm { b a n k s } }$ banks and n<sub>rows</sub> rows, the dimension of B is given by $n = 3 + 4 \cdot 2 + 2 \cdot 2 \cdot n _ { \mathrm { b a n k s } }$ · n<sub>rows</sub> and $B \subset \mathbb { R } ^ { n }$ . If $n _ { \mathrm { { r o w s } } } = 3$ and $n _ { \mathrm { b a n k s } } = 4 ,$ we then have $B \subset \mathbb { R } ^ { 5 9 }$ . The performance of the exploration algorithm might decrease in case of a large behavior space dimension as a consequence of the curse of dimensionality, where the interpretation of distances loses its sense. Therefore, working with a parsimonious set of data might imply better results in terms of diversity measures. This chosen configuration of a behavior space is an Euclidean space. Therefore, we can measure distances between the current goal embedding $z _ { g }$ and known features z in our database. Let us remark that the performance counters associated with the features presented in table 2 are not systematically available when working with a physical device.

## 4.3 Goal sampling

![](images/f3063467a950e2575517030221ae4228cdc19e4c819a77be25dcb3c84dcaa5a1.jpg)  
Figure 6: Schematic view of a sampling process. Yellow points represent border outcomes from the already known region and allow the determination of the sampling region for the next experiments.

For the discovery agent, the concept of ”curiosity” is materialized by the sampling of goals. For the sampling of goals, the simplest case scenario is to sample them uniformly at random in a connex subregion of B including the set of already discovered points. As illustrated in Figure 6, choosing the sampling region slightly larger than a subregion including the set of already discovered points allows to target regions of G that have not been reached yet but are close to the already discovered ones.

We periodically set the sampling boundaries based on the database H, and we simply sample goals with a uniform distribution. For instance, we determine the minima and maxima of the discovered values for each 1- dimensional feature:

$$
\begin{array} { r l } & { \bullet \ m i n _ { \mathcal { B } } g : = \ \big ( m i n \{ z _ { 1 } , z \ \in \ { \mathcal { H } } \} , \cdots , m i n \{ z _ { \dim ( \mathcal { B } ) } , z \ \in \ { \mathcal { H } } \} \big ) } \\ & { \qquad { \mathcal { H } } \big \{ \} \big ) } \\ & { \bullet \ m a x _ { \mathcal { B } } g : = \big ( m a x \{ z _ { 1 } , z \ \in { \mathcal { H } } \} , \cdots , m a x \{ z _ { \dim ( \mathcal { B } ) } , z \ \in \ { \mathcal { H } } \} \big ) } \\ & { \qquad { \mathcal { H } } \big \{ \} \big ) } \end{array}
$$

Then, to establish the slightly larger sampling region, we fix two factors, e.g., $f _ { 1 } = 0 . 8 , f _ { 2 } = 1 . 2 ,$ that determine percentages of the previous minimal and maximal values for the new uniform distribution that will be used to sample goals $g \in B .$ . Since G is a multidimensional space, the resulting distribution will be the cartesian product of several 1-dimensional uniform distributions:

$$
\begin{array} { c } { { z _ { g } \sim \mathcal { U } ( [ f _ { 1 } \cdot ( m i n _ { B } g ) _ { 1 } , f _ { 2 } \cdot ( m i n _ { B } g ) _ { 2 } ] ) \otimes \cdots } } \\ { { \ldots } } \\ { { \ldots \otimes \mathcal { U } ( [ f _ { 1 } \cdot ( m i n _ { B } g ) _ { \dim ( B ) } , f _ { 2 } \cdot ( m a x _ { B } g ) _ { \dim ( B ) } ] ) } } \end{array}
$$

## 4.4 Goal achievement strategy

![](images/c4e20aea7d89843a71702e40829555a1aa6b79974d1e21f05426bdfcb093b085.jpg)  
Achievement Strategy with a kNN model

Figure 7: Illustration scheme of a goal achievement strategy. The left hand side illustrates the selection of outcomes stored in the database that will allow expansion, that is the synthesis of a new outcome closer from g (right hand side). Here, the two closest neighbors to g in the database H are selected, and their parameters are used to obtain the new outcome closer to g.

We present our population-based achievement strategy, which resembles an evolutionary algorithm, where the set of pairs of programs together with their respective outcomes is seen as a population. Thus, our algorithm implies the performance of mutations of the individuals.

At each step during an IMGEP exploration, the programs (parameters) and outcomes $( ( S _ { 1 } , S _ { 2 } ) , z ) \in \Theta \times B$ that make up the agent’s discoveries are stored in the database H. Because the exploration budget is low, we choose the agent to perform non-parametric learning, which means that the complexity of the model grows linearly with the size of the database H.

![](images/495ce96f8e5cc5d9628e00eb5a33236109a9afdbdd94cad020c773f223483458.jpg)

The goal achievement strategy is typically made of two steps with two respective operators. During the selection step, the discovery agent selects outcomes. The expansion step consists of using their corresponding parameters to produce another one, leading to an outcome close to the goal $^ { g , }$ see figure 7.

We use a $k N N ^ { 2 }$ model as a selection operator. This selection operator relies on an expansion operator, as kNN selects k promising outcomes from H and expands them by producing a candidate pair of programs to achieve g. In particular, we select the k closest outcomes to the given goal g within the database H using a well-designed evaluation function L. We then mix the k corresponding pairs of programs together to obtain a candidate pair of programs to achieve goal g.

We establish a weighted evaluation function so that all features are equally considered in the selection step:

$$
\mathcal { L } _ { g } ( z ) = \sum _ { F } \frac { 1 } { \operatorname* { m a x } _ { F } - \operatorname* { m i n } _ { F } } \cdot ( z _ { F } - g ) ^ { 2 } , \forall z , g \in \mathcal { B }
$$

For mixing the programs, we select contiguous parts from multiple programs, preserving timing, and ensuring the resulting program fits within the fixed maximum cycle. The length of each segment is chosen at random. The key parameters are the number of segments to produce the output program and the number of programs to mix.

To ensure more eficiency, one also apply a mutation operator following the mixing operator. The mutations performed are : 1. randomly change existing instructions, 2. delete random instructions, 3. add random instructions. Consequently, diferent patterns of the programs change, such as the number of instructions, number of memory accesses, and the set of used addresses.

In the particular case of k = 1, we simply select the closest observation z of g in the database H and apply the mutation operator to obtain the candidate parameter. The mixing and mutation operators are detailed in B and B.

During the first iterations of the algorithm, IMGEP typically outputs outcomes that are not diverse. To overcome the dificulty of the agent exploring during its first iterations, a warm-up made of several random iterations is performed. The agent initially induces poor diversity, and while the population stored in the database H grows, the agent is more likely to solve its goals. Hence, on average over time, goals targeted by the agent are not reached with the parameters θ outputted by IMGEP; meanwhile, it is a powerful method that leads to high-diversity discoveries.

## 5 Results

We run IMGEP for a total of $N = 1 0 ^ { 4 }$ iterations, and we initialize it with a warming set of $N _ { i n i t } = 1 0 ^ { 3 }$ iterations of random program synthesis. We choose to run it with $k =$ 1, 2 and 3 as parameters for the kNN model of the goal achievement strategy Π. Each program contains between 1 and 10 instructions. The maximum cycle for the last instruction to be sent is 60. As said previously, we model two distinct sets of addresses for the cores, i.e., core 0: 0,20 and core 1: 21,40. The selection of the simulator parameters chosen to run our experiments is available in A.

We first assess the diversity of the resulting datasets through diversity measures. Secondly, we explore the results and attempt to infer diagnostics from the data. Finally, we discuss approaches that would be more relevant for IMGEP.

## 5.1 Diversity evaluation

Figure 8: Diversity of miss ratio for both cores and all bank row pairs. Orange bars represent lower and upper confidence bounds using the Student’s t-distribution asymptotic confidence interval $\begin{array} { r } { \mathbb { P } ( \bar { X } - u _ { 1 - \frac { 1 } { \alpha } } \sqrt { \frac { \hat { v } } { n } } \le g ( \theta ) \le } \end{array}$ $\bar { X } + u _ { 1 - \frac { 1 } { \alpha } } \sqrt { \textstyle \frac { \hat { v } } { n } } ) \longrightarrow . 9 5$ with mean and unbiased standard deviation estimators.

We compare our method with two simple strategies: 1. A random exploration, that is, generating pairs of programs randomly. And 2. to be compared with IMGEP for k, an algorithm that selects k pairs of programs randomly from its database H, mixes them to obtain a single pair, and mutates it. Indeed, we would like to ensure that a larger IMGEP diversity does not occur simply because of the design of the mixing and mutation operators. For this matter, we aim to demonstrate that mixing and mutating random selections of programs from the same warming set does not lead to a significantly higher diversity than that of IMGEP runs. The mixing and mutation operators are the same as $\mathrm { I M G E P ` s }$ . Hence, for a fair comparison, the first $N _ { i n i t }$ iterations of the random exploration are shared by the IMGEP runs and by the operator combination exploration. We quantitatively assess each exploration process by simply establishing diversity measurements of the outcomes stored in the respective database H. In particular, we observe how the behavior space $\boldsymbol { B }$ is covered by the corresponding outcomes $o _ { 1 } , \cdots , o _ { n }$

In the following, we denote two particular subspaces of the behavior space as the time behavior space and hit/miss space. We construct the following measure suitable to assess diversity: The number of bins filled in a n-dimensional histogram. With $n =$ $3 + 4 \cdot 2 + 2 \cdot 2 \cdot n _ { \mathrm { b a n k s } }$ · n<sub>rows</sub>, being the dimension of each outcome we obtain after an experiment, see part $4 . 2$ . We denote the quantity corresponding to each axis $j$ of an outcome F by $F _ { j } \in \mathbb { R } , 1 \le j \le n$ The values of $\operatorname { i n f } ( F _ { j } )$ and sup $( F _ { j } )$ are carefully chosen so that $\forall F , F \in ] { \mathrm { i n f } } ( F _ { j } )$ , sup(F<sub>j</sub>)[, and that for every axis $j ,$ these values determin the smallest envelopp of possible values.

<table><tr><td rowspan=1 colspan=1>Performance counter</td><td rowspan=1 colspan=1>Inf</td><td rowspan=1 colspan=1>Sup</td></tr><tr><td rowspan=1 colspan=1>Processor cycles</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Handmade determined upperbound</td></tr><tr><td rowspan=1 colspan=1>Instructions completed</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>Decode stalled</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>Cache misses</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>Cache store allocates</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>Cache demand access</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>DDR store misses</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>DDR load misses</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>DDR demand access</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr><tr><td rowspan=1 colspan=1>DDR store allocates</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>Maximal number of instructions</td></tr></table>

Table 3: Set of available performance counters in the dual-core simulator

Let us denote a chosen partition of $\operatorname { l i n f } ( F _ { j } )$ , sup $( F _ { j } ) |$ by $x ( j ) , 1 \leq j \leq n$ with $n _ { j }$ points, such that : inf( $F _ { j } ) =$ $x ( j ) _ { 1 } < \cdot \cdot \cdot < x ( j ) _ { n _ { j } } = \operatorname* { s u p } ( F _ { j } )$ . For a family of partitions $( x ( 1 ) , \cdots , x ( n ) )$ , we define a bin as :

$$
\begin{array} { r l } & { b i n ( ( x ( 1 ) _ { k _ { 1 } } , \cdots , x ( n ) _ { k _ { n } } ) ) = [ x ( 1 ) _ { k _ { 1 } } , x ( 1 ) _ { k _ { 1 } + 1 } [ \times , } \\ & { \qquad \cdots \times [ x ( n ) _ { k _ { n } } , x ( n ) _ { k _ { n } + 1 } [ , } \\ & { \qquad \forall k _ { 1 } \in \mathbb { N } \cup [ 1 , n _ { 1 } [ , \cdots \ , \forall k _ { n } \in \mathbb { N } \cup [ 1 , n _ { n } [ } \end{array}\tag{1}
$$

We define ${ \mathcal { D } } _ { ( x ( 1 ) , \cdots , x ( n ) ) }$ , the positive measure of data clouds associated with the set of all bins with the partitions $x ( j )$ ). For a set of points discovered H from $\mathcal { H } ,$ , we have:

$$
\begin{array} { c } { { \mathcal { D } _ { ( x ( 1 ) , \cdots , x ( n ) ) } ( H ) = } } \\ { { = \displaystyle \sum _ { ( k _ { 1 } , \cdots , k _ { n } ) } \mathbf { 1 } _ { \{ \mathrm { c a r d } ( b i n ( x ( 1 ) k _ { 1 } , \cdots , x ( n ) k _ { n } ) \cap H ) > 0 \} } } } \end{array}\tag{2}
$$

For each dimension $j ,$ we choose the length of bin $l _ { j }$ such that $\begin{array} { r } { \frac { | x ( j ) _ { k _ { j } } - x ( j ) _ { k _ { j } + 1 } | } { 5 } = l _ { j } , \forall 1 \leq j \leq n , \forall k _ { j } . \mathcal { D } _ { ( x ( 1 ) , \cdots , x ( n ) ) } } \end{array}$ is a well-defined positive measure for the associated set of non-empty bins of a data cloud H.

![](images/4fe7a6ff557ae20e6f99759953c30df272b97bdeeb58e87d8a1f8c1a1f947fb7.jpg)

![](images/98f70cd76dfb50399897c98a963af0c90106a29d09e1bcf83adc383fe3206ae0.jpg)

![](images/6411b6421c2161b2a414188d4f4d050c4a645a867d4124a9b5fcffe364070516.jpg)  
Figure 9: On the two left plots, we visualize the diversity on the time behavior space and on hit/miss space, that is, how the number of bins filled in a multidimensional his togram increases during exploration. On the right plot, we visualize the diversity of the entire dataset, the dimension of the considered histogram is the number of features (59). Results of $N = 1 0 ^ { 4 }$ iterations with a warming set of $N _ { i n i t } = 1 0 0 0$ iterations of random program synthesis

For each value of $k ,$ which is the number of neighbors in the optimization policy achievement model, we show that IMGEP leads to a higher diversity than the operator combination exploration. Indeed, IMGEP finds many more distinct values of miss ratios across the distinct locations—pairs (bank, row)—in the main memory; see figure 8. Note that the discoveries of ”miss” occurrences probably help to induce more interference. Indeed, whenever requested data is missing in a DDR bufer, it has to be recharged, and this involves an additional delay. Moreover, figure 9 shows that the IMGEP diversity measurement increases faster on the entire behavior space than that of the operator combination exploration, and simultaneously on the distinct L2 & DDR miss/hit and execution time goal subspaces.

Generating programs with IMGEP allows one to observe much more interference than generating programs randomly. Interference occurs whenever the execution time in isolation difers from the execution time in nonisolation. To some extent, we consider that a pair of programs produces interference if |time[non-isolation] − time[isolation] $| \mathbf { \partial } > 0$ . In the example run presented in figure 10, the bars with axis 0 on histograms for values of —time[non-isolation] - time[isolation]— demonstrate that exploring the simulator with randomly generated programs produces interference about 4 times out of 10, whereas it is about 7 times out of 10 for IMGEP.

![](images/b6f7c649ff166da66f59bcb0e103d22550dcbb381c2059213a1873f47739e59b.jpg)  
Figure 10: Diversity for execution time behavior space. Results of $N = 1 0 ^ { 4 }$ IMGEP iterations with $k = 1$ and a warming set of $N _ { i n i t } = 1 0 0 0$ iterations of random program synthesis.

Information on the execution time is represented for each program in Figure 11. More precisely, we attribute to each pair of programs executed simultaneously on core 0 and core 1 two respective points on the distinct scatter plots to represent its couple (time[isolation], time[nonisolation]). This representation allows us to spot interference, that is, whenever points are not located on the red identity line. Points above the red line represent programs that are slowed down by the parallel execution on the other core, whereas points below the line represent programs which are accelerated, which means the execution time is faster when the program is executed together with another program. This is due to the absence of a coherency management mechanism. For instance, the following pair of programs results in the acceleration of both programs. One can show DDR bank and line numbers instead of addresses. Instructions of core 1 on cycles 36, 40, and 41 are processed earlier when the program on core 0 is executed because of its instructions on cycles 20 and 30 which load the lines {bank 1, row 1} and {bank 0, row 1} into their bufer. Hence, sequences $\mathrm { P R E } { = } { > } \mathrm { A C T }$ are avoided as the data is already in the bufer.

## Listing 2: Example of programs inducing mutual acceleration.

bk\_rw\_core0 = {20:{ ’ bk ’:1, ’rw ’:1} , 27:{ ’bk ’:3 , ’rw ’:0} ,   
30:{ ’bk ’:0 , ’rw ’:1} ,31: {’bk ’: 2, ’rw ’: 0},   
36: {’bk ’: 1, ’rw ’: 0} ,43: {’bk ’: 3, ’rw ’: 0},   
46: {’bk ’: 0, ’rw ’: 0}}   
bk\_rw\_core1 = {6:{ ’bk ’:2 , ’rw ’:1} , 8:{ ’bk ’:2 , ’rw ’:1} ,   
11:{ ’bk ’:3 , ’rw ’:1} ,18:{ ’ bk ’:1 , ’rw ’:2} ,   
27:{ ’bk ’:0 , ’rw ’:2} ,36:{ ’ bk ’:1 , ’rw ’:1} ,   
40:{ ’bk ’:1 , ’rw ’:1} , 42:{ ’bk ’:0 , ’rw ’:1} ,   
45:{ ’bk ’:3 , ’rw ’:1} , 47:{ ’bk ’:0 , ’rw ’:2}}   
core 1 mutuality 78 cycles , isolation : 162 cycles   
core 0 mutuality 108 cycles , isolation : 97 cycles

![](images/e21cc2cd2ddc979ae0f2138848bdc4428f66d808a00a21ceb340babe409a4e25.jpg)

![](images/84db07f5c2a7408e9b5a0d9b5d2032b8dbcf6c240a768edafe93b761d749e411.jpg)  
Figure 11: Diversity for execution time behavior space. Results of $N = 1 0 ^ { 4 }$ IMGEP iterations with $k = 1$ and a warming set of $N _ { i n i t } = 1 0 0 0$ iterations of random program synthesis.

In Figure 12, we represent values of couples (time[isolation, core 0] - time[non-isolation, core 0], time[isolation, core 1] - time[non-isolation, core 1]). Any point with an abscissa or ordinate diferent from 0 corresponds to a pair of programs presenting interference on core 0 or core 1, respectively. The figure clearly shows that the IMGEP cloud spreads more, indicating that our algorithm finds many more interference cases.

## 5.2 Analysis

The results we presented demonstrate through diversity measurements that automated exploration algorithms can cover eficiently an output space of a complex system associated with a simulated hardware model.

The data from which the exploration is performed provides only aggregated information and is not internal. While this implies that characterizing sources of interference with such a defined behavior space is limited, we spot an emerging structure in Figure 13, suggesting that pairs of programs with many requests to the same DDR locations induce a greater variety of interference delays.

Hence, the resulting pairs of programs causing interference can be further analyzed by humans, and our method may be completed to ensure that distinct points in the outcome space provide two distinct sources of interference.

![](images/a93ffa8a80e35c56b3d16ae77b4e5b208bc36aa576088c9075d1f66f7aad84b7.jpg)  
Figure 12: Spanning of IMGEP on time behavior space. Interference occurs on core 0 or 1 when the abscissa or ordinate is diferent from zero. Results of $N = 1 0 ^ { 4 }$ IMGEP iterations with $k = 1$ and a warming set of $N _ { i n i t } = 1 0 0 0$ iterations of random program synthesis.

![](images/b72527af893fe8dbfe9b63326abde31e78d8b4c27e167b032f410e207e4be738.jpg)

![](images/96a4478cb68851d56289934a70a804a7fc90acd3ac835ce8a8107049419115e9.jpg)  
Figure 13: (time[non-iso] - time[iso]) vs nonoverlap. We call non-overlap the quantity $f ( p _ { 0 } , p _ { 1 } ) =$ $\textstyle \sum _ { j }$ |(nb access row j , core 0) − (nb access row j core 1)|. This graph shows the number of access occurring in the same locations for both cores in the main memory is not enough to characterize interference.

We believe that creating a space that fully consists of sources of interference will allow discoveries of non-trivial access patterns. In addition, one wants to discover access patterns that are unknown and dificult to produce by specialized engineers. Furthermore, the resulting behavior space would not necessarily be a Euclidean space, and could, for example, be a graph, a tree, or a semantic space.

## 6 Conclusion

This work deals with the problem of interacting with SoCs via automated discovery algorithms to help the discovery of singular cases. The method relies on simple machine learning models to structure an autonomous agent that fixes its own goals, and on the establishment of a behavior and a parameter space.

The results demonstrate that automated discovery algorithms can interact successfully with a simplified simulated model of dual-core architecture and encourage the idea of exploring a real SoC. Indeed, the behavior space, which consists of performance counters, is explored eficiently with IMGEP, as it leads to a higher diversity than exploring the simulator with programs that are randomly generated.

This approach encourages the idea of exploring a behavior space that is more suited for characterizing of mechanisms of interference in order to discover non-trivial sources of interference. In further work, we will aim to develop a strategy that specifically identifies distinct mechanisms that are responsible for interference, rather than exhibiting a large diversity of interferences regardless of the reasons they occur. For this, we will distinguish whether the observation of an interference relies on the activation of a new mechanism. Hence, the objective will be to maximize the number of distinct types of interference.

## Appendices

## A Description of our dual-core model

![](images/77cd38a4e244f4f6b632c6d985c94d24f0bf00442c3c2733c50003b61af99b53.jpg)  
Table 4: Selection of the simulator parameters for our experiments

## A.1 Complementary information on the DDR and its controller

Our simplified DDR2 model implements the following operations and some are described in 4

• WR: Write

• RD: Read

• ACT: Activation of a closed row.

• PRE: Deactivation of an open row.

• REF: Refreshment of memory cells.

The functionning of both the DDR and the memory controller are coupled. At each cycle:

• The controller determines requests that are achieved, that is, if the current date is superior than the scheduled completion date. In that case, for an RD request, the controller calls a ”call back” function which signals to the core that this access is achieved. This callback is used by the core to allow the execution of a new instruction. No callback functionality is not implemented for RD requests.

• The controller treats requests that are located in its entry queue. For each request, it determines whether it can be treated by the DDR according to its state (handled by the DDR state machine) while ensuring minimal delay contraints. It then determines the ”best” request to treat depending on a established ranking to prioritize request inducing ”row hits”, WR over RD requests and arrival order (FIFO policy). THhe controller finally sends the ”best” request to the DDR. Memory accesses are done with burst lenght of 8 bytes.

The model is inspired from mechanisms described in [14].

## A.2 Interconnect model

The interconnect model can be described as follows:

1. It consists in maintainig a queue (FIFO) of memory requests and in executing request in the arrial order and after a delay corresponding to minimale access latency. A random relay of 2 cycles in added.

2. At each clock cycle, some number of requests in priority is unpiled from the queue among the ones that are executable during the current cycle. The requests are the, sent to the memory. The number of unpiled requests at each cycle is determined by the bus bandwidth and each the counting of the numberf of treated requests.

3. The requests that could not be treated are moved in the queue of requests that are to be treated in next cycles.

4. The interconnect clock cycle frequency and that of the cores are the same. One could introduce distinct clocks by calling the functions ”clock” in our codes at distinct cycles.

## B Intrinsically Motivated Goal Exploration Process

Parameters chosen for the IMGEP exploration are described on table 5. We describe below the algorithm designed to mix sequences of simplified assembly algorithms. As a reminder, the algorithms are dictionaries of form {cycle : (type, adress)}.

IMGEP parameters Description Value   
N<sub>init</sub> Number of iterations for the warming set 1000   
N − N<sub>init</sub> Total number of IMGEP iterations 9000   
k Number of neighbors in the kNN model for the goal achievement strategy Π. 1,2,3   
Number of mutations Number of mutation for each program in the model Π 5   
max sending cycle Maximum cycle for sending the instruction (cycle) 60   
Range instructions Minimal and maximal number of instructions 1-10   
core 0 Range of addresses 0-20   
core 1 Range of addresses 21-40

Table 5: The selection of IMGEP parameters for our experiments.  
Algorithm 1 Random Mixing of Instruction Sequences   
Require: A list of instruction programs S, number of parts ${ \overline { { P , } } }$   
random seed s, maximum cycle C<sub>max</sub>   
Ensure: A mixed instruction program M   
Initialize random generator with seed s   
Chunks ← ∅   
for all program $p \in S$ do   
Sort instructions of p by increasing cycle   
Split p into P contiguous parts   
for all part q of p do   
Append q to Chunks   
end for   
end for   
Randomly shufle Chunks   
Concatenate all elements of Chunks into list L   
Randomly select |L| distinct cycle values from [1, C<sub>max</sub>]   
Sort selected cycle values in increasing order   
for i ← 1 to |L| do   
Assign cycle cycles[i] to instruction L[i]   
end for   
Construct mixed program M from assigned cycles and instruc  
tions   
return M

Algorithm 2 Mutation operator   
// Mutate an instruction sequence by adding, deleting, or modifying   
instructions.   
Require: Instructions θ, Number of mutations N, Maximal cycle c,   
range for adresses a<sub>min</sub>, a<sub>max</sub>,maximal number of instructions i<sub>max</sub>   
Ensure: Initialize a copy θ of the intructions θ   
Ensure: Initialize set of available cycles freecycles: all cycles - used   
cycles   
for i ← 1 : N do   
if lenght(θ ) > 1 then   
Select a mutation type : t ∼ U([add, delete, modify])   
else   
Select a mutation type : t ∼ U([add, modify])   
end if   
if mutation type is add and freecycles non empty then   
newcycle ∼ U(freecycles)   
θ (newcycle) = {new instruction type,a}   
remove freecycles(newcycle)   
else if mutation type is delete and lengh $\operatorname { t } ( \theta ^ { ' } ) > 1$ then   
Delete θ (cycle to delete)   
set free freecycles(cycle to delete)   
else if Mutation type is modify and lenght $( \theta ^ { ' } ) > 1$ then   
Pick an instruction cycle to modify $t _ { o l d , } a _ { o l d } = \theta ^ { ' } ( \mathbf { c } )$   
modify choice m ∼ U([type, adress, both])   
if m = type then   
t<sub>new</sub> = write if t<sub>old</sub> == read else read   
θ<sup>′</sup>(c) = {type : t<sub>new</sub>, address : a<sub>old</sub>}   
else if m address then   
θ<sup>′</sup>(c) = {type : t<sub>old</sub>, address : $\mathcal { U } ( \{ a _ { m i n } , \cdot \cdot \cdot , a _ { m a x } \} ) \}$   
else   
//Change both type and address   
t<sub>new</sub> = write if t<sub>old</sub> == read else read   
θ<sup>′</sup>(c) = {type : t<sub>new</sub>, address : U({a<sub>min</sub>, · · · , a<sub>max</sub>})}   
end if   
end if   
if lenght $( \theta ^ { ' } ) > i _ { m a x }$ then   
Uniformally sample set of distinct lenght(θ ) − i instruc  
tions to delete   
Delete the instructions   
end if   
end for

## B.1 Diversity Evaluation

Another diversity measure, but ill-defined, is the sum of the squared euclidean distances between all pairs: $\begin{array} { r } { \mathcal { D } ( \mathcal { H } ) = \sum _ { i } \sum _ { j < i } \left| \left| o _ { i } - o _ { j } \right| \right| ^ { 2 } } \end{array}$ . This measure is not a well defined positive measure as $\mathcal { D } ( H _ { 1 } \cup H _ { 2 } ) \neq \mathcal { D } ( H _ { 1 } ) + \mathcal { D } ( H _ { 2 } )$ when $H _ { 1 } \cap H _ { 2 } = \emptyset$ in general. However, it provides a good intuition of the diversity, as a small perimeter cloud with small pairwise distances leads to a small diversity result whereas a large cloud perimeter involves larger pairwise distances and thus a larger diversity.

## References

[1] Abderaouf Nassim Amalou. “Machine learning for timing estimation”. Theses. Universit´e de Rennes, Dec. 2023. url: https : / / hal . science / tel - 04406029.

[2] Pierre Bieber et al. “A model-based certification approach for multi/many-core embedded systems”. In: 9th European Congress on Embedded Real Time Software and Systems (ERTS 2018). 9th European Congress on Embedded Real Time Software and Systems (ERTS 2018). Toulouse, France, Jan. 2018. url: https://hal.science/hal-01700857.

[3] Jingyi Bin et al. “Studying co-running avionic real-time applications on multi-core COTS architectures”. In: ERTS 2014 proceedings. Toulouse, France, Feb. 2014. url: https://hal.science/ hal-02271379.

[4] Armelle Bonenfant et al. “Early WCET Prediction using Machine Learning”. In: Proceedings of WCET 2017. OpenAccess Series in Informatics (OASIcs). ISBN : 978-3-95977-057-6. Dubrovnik, Croatia: OA-SICs, Dagstuhl Publishing, June 2017, 5:1–5:9. doi: 10.4230/OASIcs.WCET.2017.5. url: https:// hal.science/hal-03116285.

[5] Fr´ed´eric Boniol et al. “Modelling and analyzing multi-core COTS processors”. In: 11th European Congress on Embedded Real Time Software and Systems (ERTS 2022). Toulouse, France, June 2022. url: https://hal.science/hal-03761937.

[6] Fr´ed´eric Boniol et al. “PHYLOG certification methodology: a sane way to embed multi-core processors”. In: 10th European Congress on Embedded Real Time Software and Systems (ERTS 2020). Toulouse, France, Jan. 2020. url: https://hal. science/hal-02441323.

[7] Lorenzo Carletti et al. “Taking a closer look at memory interference efects in commercial-of-theshelf multicore SoCs”. In: Journal of Systems Architecture 167 (2025), p. 103487. issn: 1383-7621. doi: https://doi.org/10.1016/j.sysarc.2025. 103487. url: https://www.sciencedirect.com/ science/article/pii/S1383762125001596.

[8] C´edric Courtaud. “Caract´erisation de la sensibilit´e aux interf´erences m´emoire dans les syst\`emes temps-r´eels embarqu´es sur des plateformes multicoeurs”. Theses. Sorbonne Universit´e, Jan. 2020. url: https : / / theses . hal . science / tel - 03429679.

[9] Mayalen Etcheverry. “Curiosity-driven AI for Science : Automated Discovery of Self-Organized Structures”. Theses. Universit´e de Bordeaux, Nov. 2023. url: https://theses.hal.science/tel-04504878.

[10] Mayalen Etcheverry, Cl´ement Moulin-Frier, and Pierre-Yves Oudeyer. “Hierarchically-Organized Latent Modules for Exploratory Search in Morphogenetic Systems”. In: CoRR abs/2007.01195 (2020). arXiv: 2007 . 01195. url: https : / / arxiv . org / abs/2007.01195.

[11] Mayalen Etcheverry et al. “AI-driven automated discovery tools reveal diverse behavioral competencies of biological networks”. In: eLife 13 (Jan. 2025). Ed. by Arvind Murugan and Aleksandra M Walczak, RP92683. issn: 2050-084X. doi: 10 . 7554 / eLife.92683. url: https://doi.org/10.7554/ eLife.92683.

[12] Martin J. Falk et al. “Curiosity-driven search for novel nonequilibrium behaviors”. In: Phys. Rev. Res. 6 (3 July 2024), p. 033052. doi: 10 . 1103 / PhysRevResearch.6.033052. url: https://link. aps . org / doi / 10 . 1103 / PhysRevResearch . 6 . 033052.

[13] Sylvain Girbal, Jimmy Le Rhun, and Hadi Saoud. “METrICS: a Measurement Environment For Multi-Core Time Critical Systems”. In: 9th European Congress on Embedded Real Time Software and Systems (ERTS 2018). 9th European Congress on Embedded Real Time Software and Systems (ERTS 2018). Toulouse, France, Jan. 2018. url: https://hal.science/hal-02278292.

[14] Alfonso Mascare˜nas Gonz´alez. “DDR SDRAM Interference Minimization via Task and Memory Mapping in a Multi-objective Optimization context on Heterogeneous MPSoCs”. Theses. Toulouse, ISAE, Dec. 2022. url: https : / / hal . science / tel-05025151.

[15] Jonathan Grizou et al. “A curious formulation robot enables the discovery of a novel protocell behavior”. In: Science Advances 6.5 (2020), eaay4237. doi: 10 . 1126 / sciadv . aay4237. eprint: https : //www.science.org/doi/pdf/10.1126/sciadv. aay4237. url: https://www.science.org/doi/ abs/10.1126/sciadv.aay4237.

[16] Jean Guyomarc’H. “Analyse de syst\`emes tempsr´eels de sˆuret´e et mitigation de leurs interf´erences temporelles”. Theses. Universit´e Paris-Saclay, Oct. 2021. url: https://theses.hal.science/tel-03793814.

[17] Niklas Hauser and Jan Pennekamp. “Tool: Automatically Extracting Hardware Descriptions from PDF Technical Documentation”. In: 3.1 (2023). issn: 2770-5501. doi: 10 . 5070 / SR33162446. url: https : / / www . comsys . rwth - aachen . de / publication / 2023 / 2023 \_ hauser \_ tool - automatically - extracting - hardware / 2023 \_ hauser \_ tool - automatically - extracting - hardware.pdf.

[18] Ray Kuldeep. “Memory Hierarchy – How does computer memory work ?” In: SPEAR ITN (2024). url: https : / / spear - itn . eu / memory - hierarchy-how-does-computer-memory-work/.

[19] James Ladyman, James Lambert, and Karoline Wiesner. “What is a complex system?” In: European Journal for Philosophy of Science 3.1 (2013), pp. 33–67. issn: 1879-4920. doi: 10.1007/s13194- 012-0056-8. url: https://doi.org/10.1007/ s13194-012-0056-8.

[20] Tamara Lugo et al. “A Survey of Techniques for Reducing Interference in Real-Time Applications on Multicore Platforms”. In: IEEE Access 10 (Jan. 2022), pp. 21853–21882. doi: 10 . 1109 / ACCESS . 2022.3151891.

[21] Nikhilesh Singh et al. “Kryptonite: Worst-Case Program Interference Estimation on Multi-Core Embedded Systems”. In: ACM Transactions on Embedded Computing Systems 22 (Sept. 2023), pp. 1– 23. doi: 10.1145/3609128.

[22] Steven H. Vanderleest and Samuel R. Thompson. “Measuring the Impact of Interference Channels on Multicore Avionics”. In: CoRR abs/2101.02204 (2021). arXiv: 2101.02204. url: https://arxiv. org/abs/2101.02204.