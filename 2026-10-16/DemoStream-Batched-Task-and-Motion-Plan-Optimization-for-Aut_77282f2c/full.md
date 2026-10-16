# DemoStream: Batched Task-and-Motion-Plan Optimization for Automated Learning from Demonstration

Caelan Garrett<sup>1</sup>, Neel Jawale<sup>1</sup>, Fabio Ramos<sup>1,2</sup>

Abstract— Learning from demonstration is an effective strategy for endowing robots with manipulation capabilities but requires training on datasets comprised of massive amounts of demonstrations. Existing demonstration collection approaches either require constant human involvement, for example, via teleoperation or are specialized for a single task. Task and Motion Planning (TAMP) systems are able to autonomously solve and therefore generate demonstrations for general-purpose tasks; however, learning from their demonstrations has proven difficult due to multi-modality in their solutions. In this work, we build on recent work leveraging GPU acceleration for TAMP and propose DemoStream, a TAMP system that optimizes demonstration quality in order to improve behavior cloning success rates. First, we introduce batched TAMP sampling operations along with a new TAMP algorithm that correspondingly performs batched constraint satisfaction in order to explore many solutions in parallel. We conduct simulation experiments ablating different action spaces, cost functions, and batch sizes to analyze the effect of these choices on downstream policy success rates. Finally, we show that we can leverage the safety guarantees of TAMP to directly collect real-world demonstrations at an average rate of 66 demos per hour. See our website for policy rollout videos: demostream.github.io.

## I. INTRODUCTION

Foundation models such as Vision-Language-Action Models (VLAs) [1] show incredible promise for robot control but require vast amounts of robot data, which is not easily obtainable. Teleoperation is a simple strategy for collecting robot data in the form of demonstrations but incurs a large human cost per task, making it difficult to scale. To reduce the amount of human effort, recent work developed semiautomated demonstration generation systems that adapt a handful of human-teleoperated demonstrations to new initial states [2]. However, these approaches still require annotated human demonstrations per task, preventing them from scaling autonomously to new tasks, and are typically deployed only in simulation because they produce unsafe behaviors.

Task and Motion Planning (TAMP) [3] systems are able to autonomously solve new tasks using the same planning model. Prior work has used TAMP to generate demonstrations for imitation learning [4], [5], but due to the stochastic nature of prior TAMP algorithms, the resulting demonstrations have been difficult to learn from due to their subopti mality and multi-modality. Recent work on TAMP leverages hardware acceleration, for example, on Graphics Processing Units (GPUs), to more efficiently solve problems [6], [7], [8]. We directly investigate the impact of GPU acceleration not only on producing demonstrations quickly but also on optimizing them so that they are easier to learn from.

![](images/c2369873325ce8e7ebda7cf5d2e878716b32c5f8d8de89f1645a4022cb489429.jpg)  
Fig. 1: Real-World Policy Evaluation. DemoStream collected 75 real-world demonstrations and trained visuomotor policies with a success rate of 67%. Left: stack the red block on the blue block (Stack-Two). Center: drop the banana into the trash bin (Banana-Bin). Right: pack the apple and lime on the plate (Apple-Lime).

Our approach, DemoStream, first makes novel TAMP contributions in the form of batched streams, conditional samplers that generate constraint-satisfying outputs for a batch of input values, and batched computation graph optimization algorithms that compose multiple batched streams and cost functions in order to explore thousands of TAMP solutions in parallel. These representation and algorithmic innovations give us the flexibility to explore multiple TAMP cost functions and analyze their effect on demonstration success rate, demonstration length, and downstream policy success rate. Notably, we find that without solution optimization, TAMP systems still have high demonstration generation success rates but train policies with success rates of zero (Table I). Additionally, the choice of cost function matters; we find that it is easier to learn from demonstrations that minimize task-space rather than joint-space costs, both during TAMP and motion planning, likely because these costs minimize the motion of wrist-mounted cameras, shrinking the distribution of encountered observations. DemoStream is able to train successful visuomotor policies on a variety of tasks including multi-robot (Figure 5), long-horizon (Figure 6), agentic openworld (Figure 7), and real-world (Figure 1) tasks.

The contributions of this paper are:

• New TAMP batched samplers and computation graph optimizers for GPU-accelerated plan optimization;

• Automated demonstration generation via agentic, batched TAMP for end-to-end imitation learning;

• Analysis on the effect of action space, cost function, and batch size on policy learning success rates;

• Simulation experiments demonstrating long-horizon, multi-robot, and open-world DemoStream capabilities;

• Safe automated real-world demonstration collection.

## II. RELATED WORK

DemoStream builds on work at the intersection of demonstration generation for imitation learning and TAMP.

a) Demonstration Generation: We seek to acquire demonstration datasets of observation-action training examples to train visuomotor policies using Behavioral Cloning (BC) [9]. Rather than collecting these demonstrations through teleoperation [10], one strategy to reduce the amount of human effort involved is to teleoperate a handful of annotated demonstrations per task, adapt those demonstrations to new initial states, execute the adapted demonstrations, and retain the successful ones [2]. Non-trivial variations in the initial state induce large changes in the task-relevant workspace and safe configuration space, hampering the ability to successfully adapt demonstrations. Several approaches interleave skill-specific demonstration adaptation with motion planning to be robust to these variations [11], [12], [13], [14]. Still, these methods are task-specific and require new human demonstrations whenever the sequence of actions or involved objects change. In contrast, DemoStream is able to produce demonstrations for unseen tasks involving unseen objects without any human demonstrations.

b) Task and Motion Planning: Task and Motion Planning (TAMP) [3] jointly plans discrete & continuous robot actions, for example, which object it should pick and how it should safely execute the pick. Several TAMP algorithms [6], [8], [7] leverage GPU acceleration of some kind in order to more efficiently search over continuous action parameters. Of these, ScheduleStream [8] is closest to our approach; however, its use of GPU acceleration focuses on efficient multiarm trajectory collision checking. In contrast, we introduce the notion of a batched conditional sampler, where a single operation generates many independent values at a time.

Two approaches [4], [5] investigated generating demonstrations fully autonomously using TAMP. Both used existing CPU-only TAMP algorithms to generate demonstrations [15], [16] without direct attention to optimizing demonstrations for learnability. In this work, we find that successfully solving tasks alone is not enough for successful imitation learning, one must also produce high-quality demonstrations. Moreover, these approaches were never shown to generate real-world demonstrations in a scalable manner.

Human-In-The-Loop (HITL) TAMP [17] interleaves TAMP with conventional teleoperation where TAMP delegates control to a human teleoperator during contact-rich segments of a task, such as insertion actions. Partial autonomy enables a single human to simultaneously teleoperate multiple robot arms using a queueing system; however, a human is still present on every episode.

## III. BATCHED TAMP

Our goal is to develop a TAMP system that can autonomously generate high-quality demonstrations for imitation learning. To that end, we first provide background on the TAMP framework build off of (Section III-A). Then, we introduce batched streams, a new general-purpose TAMP representational capability that will allow us more efficiently optimize demonstration quality (Section III-B). In order to maximize the utility of batched streams, we present a new TAMP algorithm that composes batched streams in batched computation graphs (Section III-C).

## A. TAMP Preliminaries

We build our approach on top of ScheduleStream [8], a domain-independent language for planning & scheduling with continuous sampling operations. ScheduleStream extends Planning Domain Definition Language (PDDL) [18] by offering support for Python sampling operations that generate continuous values, formalized as streams,

To convey our approach, we will use the singlerobot <sup>1</sup> two-object packing task in Figure 6 (left) as a running example. In this task, the goal is to transfer both the red and blue blocks, which initially start on the table, to the plate in a manner that is efficient and avoids collisions. In this domain, the constants, i.e., primitive values, are the discrete names of objects = [red,blue,plate,table], continuous robot configurations q ∈ R<sup>d</sup>, and continuous object placement and grasp poses $\mathfrak { p } , \mathfrak { g } \in \mathrm { S E } ( 3 )$ . Discrete values are strings and continuous values are PyTorch tensors.

Relationships among constants are represented using functions. A function Function represents a relationship between a tuple of inputs, modeled as parameters "?x1 ... ?xN", and an output. A predicate Predicate is function where the output is either True or False. Predicates and functions specify a domain condition, denoted by cond, that conveys typing information about their parameters. In our running example, the following predicates declare constant typing information about their input parameters.

Conf = Predicate("?q"); Object = Predicate("?obj")   
Placement = Predicate("?obj ?p")   
Grasp = Predicate("?obj ?g")

Additionally, there are fluent functions that model changing state-variable values; At is the current robot configuration, Holding is what object the robot is holding (or None), Attached gives the parent frame of object "?obj" in the kinematic tree, and Pose is SE(3) pose of object "?obj" with respect to its parent frame.

At = Function(); Holding = Function()   
Attached = Function("?obj", cond=[Object("?obj")])   
Pose = Function("?obj", cond=[Object("?obj")])

States are assignments (denoted by <=) of output values to all function instances. Any undeclared function and predicate values are implicitly None and False respectively. The initial state in our example is:

where q0 is the initial robot configuration and p\_r, p\_b, p\_p are the initial poses of the red, blue, and plate objects. The goal conditions (denoted by ==) are:

<sup>1</sup>DemoStream also supports multiple robots (see the Multi-Arm task in Figure 5), going beyond prior TAMP demonstration generators [4], [5].

Actions describe legal decisions the system can take to change the current state. An action Action has a tuple of input parameters params, a list of equality conditions cond, a list of assignment effects eff, and a list of additive costs cost. An action can be applied to a state if all the conditions cond are satisfied by the state. After application, the assignments in the state are updated to have the values listed in the effects eff and the costs in cost are incurred.

In our example, we specify three actions [move, pick, place] although additional ones can readily be added. The move action has parameters for the current configuration "?q1", trajectory "?t", and new configuration "?q2".

move = Action(params="?q1 ?t ?q2",  
cond=[Motion("?q1 ?t ?q2"), At()=="?q1", ...],  
eff=[At()<="?q2"],  
cost=[Distance("?q1, ?q2")])

The pick action has parameters for the current robot configuration "?q1", object to be picked "?obj", current placement pose "?p", and new grasp pose "?g".

pick = Action(params="?q ?obj ?g ?p",   
cond=[Kin("?q ?obj ?g ?p"), Holding()==None,   
Pose("?obj")=="?p", At()=="?q"],   
eff=[Holding()<="?obj", Pose("?obj")<="?g",   
Attached("?obj")<=None])

The place action has parameters for the current robot configuration "?q1", object to be placed "?obj", current grasp pose "?g", new placement pose "?p", object to be placed upon "?o2", and the current pose of "?o2".

```javascript
place = Action(params="?q ?obj ?g ?p ?o2 ?p2",
cond=[Kin("?q ?obj ?g ?p"), Holding()=="?obj",
Support("?obj ?p ?o2 ?p2"), Pose("?o2")=="?p2",
Pose("?obj")=="?g", At()=="?q"] +
[∼Collision("?obj", Pose("?obj"), o, Pose(o))
for o in objects],
eff=[Holding()<=None, Pose("?obj")<="?p",
Attached("?obj")<="?o2"])
```

## B. Batch Streams

So far, we’ve modeled domains purely declaratively; however, we require a procedural component to generate values that serve as valid parameters params for actions [move, pick, place]. Additionally, these values must satisfy the following static predicates, which are present in function conditions and action preconditions cond. Grasp defines stable grasps "?g", Support defines placement poses "?p" supported by another pose "?p2", Kin models the kinematic constraint that the end-effector pose at configuration "?q" times grasp pose "?g" gives placement pose "?p", and Motion asserts configurations "?q1", "?q2" are the start and end of trajectory "?t".

Grasp = Predicate("?obj ?g", cond=[Object("?obj")])   
Support = Predicate("?obj ?p ?o2 ?p2", cond=[   
Placement("?obj ?p"), Placement("?o2 ?p2")])   
Kin = Predicate("?q ?obj ?g ?p", cond=[   
Conf("?q"), Grasp("?obj ?g"), Placement("?obj ?p")]   
Motion = Predicate("?q1 ?t ?q2", cond=[   
Conf("?q1"), Conf("?q2")])

PDDLStream [16] proposed generating predicatesatisfying constants by modeling streams, conditional generators that take in a tuple of input constants and yield tuples of output constants. A stream, denoted as Stream, specifies a predicate, a subset of the predicate’s parameters as inputs cond (the remaining parameters are outputs), and a Python function that returns a generator gen. For example, a grasps stream takes object "?obj" as input and yields grasp pose tensors for output parameter "?g".

```python
def grasp_gen(obj: str) \
-> Iterator[Tuple[Tensor]]: ...
grasps = Stream(Grasp, inps="?obj", gen=grasp_gen)
```

While it is possible to leverage GPU acceleration within grasp\_gen to produce many outputs, fundamentally streams consume only a single input at a time. This becomes limiting when we consider an Inverse Kinematics (IK) stream ik\_qs responsible for satisfying Kin. We would like to leverage advances in batched IK [19] to solve many IK subproblems simultaneously, for different grasp and placement combinations, but to do so, we must augment the stream contract. To that end, we introduce batch streams. A batch stream, BatchStream, specifies a batched Python function fn, which takes in a list of input tuples and outputs a list of lists of output tuples per input tuple, along with a maximum batch size batch. For example, the ik\_qs stream in Figure 2 takes N = 128 grasp and placement pairs as input and solves for configurations that satisfy them.

inp Kin("?q ?obj ?g ?p")   
(obj, g1, p1) [(q1 1, ) ... (q1 M, ) ]   
(obj, g2,p2) []   
ik\_qs   
… …   
(obj,gN,pN) batch = N = 128 [(qN 1,)]   
Grasp("?obj ?g"), Placement("?obj ?p") Conf("?q")

Fig. 2: Batch IK Stream. The ik\_qs stream takes a batch of N = 128 inverse kinematics subproblems as inputs and returns a list of configurations "?q" per subproblem, if any can be found.

In our domain, we batch the placement generation, IK, and motion streams placements, ik\_qs, and motions because they all take continuous values as inputs.

```python
def support_fn(inp: List[Tuple[str,str,Tensor]]) \
-> List[List[Tuple[Tensor]]]: ...
def ik_fn(inp: List[Tuple[str,Tensor,Tensor]]) \
-> List[List[Tuple[Tensor]]]: ...
def motion_fn(inp: List[Tuple[Tensor,Tensor]]) \
-> List[List[Tuple[Tensor]]]: .
placements = BatchStream(Support,
inps="?obj ?o2 ?p2", fn=support_fn, batch=inf)
ik_qs = BatchStream(Kin,
inps="?obj ?g ?p", fn=ik_fn, batch=256)
motions = BatchStream(Motion,
inps="?q1 ?q2", fn=motion_fn, batch=1)
```

We also batch functions that compute configuration distances (Distance) and placement collisions (Collision).

def distance(inp: List[Tuple[Tensor,Tensor]]) \   
-> List[float]:   
def collision(inp: List[Tuple[str,Tensor, \   
str,Tensor]]) -> List[bool]:   
Distance = Function("?q1 ?q2",   
fn=distance, batch=inf)   
Collision = Predicate("?obj1 ?p1 ?o2 ?p2",   
test=collision, batch=inf)

## C. Batch Planning

To effectively plan with the batch stream in Section V-C, we introduce a new planning algorithm BATCH-STREAM. It builds off of the LAZY-STREAM algorithm [8] in that it plans lazily prior to calling streams to avoid unnecessary expensive operations. Critically, it differs in how it calls streams once it has identified a plan in BATCH-OPTIMIZATION.

Algorithm 1 Batch Stream Planning   
1: procedure BATCH-STREAM(I, $\mathcal { G } , \mathcal { A } , \mathcal { S } ; c _ { \mathrm { m a x } } )$   
2: S ← LAZY-STREAMS(I, S) ▷ Create lazy versions of the streams   
3: while True do   
4: I ← APPLY-STREAMS(I, S) ▷ Propagate lazy stream outputs   
5: $\pi  \operatorname { s E A R C H } ( I , \mathcal { G } , \mathcal { A } )$ ▷ Solve a search subproblem   
6: if $\pi = \mathbf { N }$ one then   
7: return None ▷ Return failure   
8: $\psi $ RETRACE-STREAMS $( \mathcal { T } , S , \pi , \mathcal { G } )$ ▷ Lazy plan parameters   
9: $F \gets \mathrm { E X T R A C T - C O S T S } ( \pi )$ ▷ Extract action costs   
10: b ← BATCH-OPTIMIZATION $( \psi , F ; c _ { \mathrm { m a x } } )$ ▷ Bind parameters   
11: $\mathbf { i f } \ b _ { * } \neq$ None then   
12: return SUBSTITUTE $( \pi , b _ { * } )$ ▷ Return a solution   
13: $S \gets \mathrm { { U P D A T E - S T R E A M S } } ( S , \psi )$ ▷ Remove attempted streams

Algorithm 1 displays the pseudocode for BATCH-STREAM. It takes as input initial state I, goal conditions ${ \mathcal { G } } ,$ actions ${ \mathcal { A } } ,$ and batched streams S as well as a maximum cost $c _ { \mathrm { m a x } } \in [ 0 , \infty )$ . First, it creates a set of lazy batched streams S, with placeholder output values [8]. On each iteration, it constructs an augmented initial state $I \supseteq \mathcal { Z }$ using the lazy stream outputs. Then, it solves a finite search subproblem. If a plan π is found, it identifies the lazy action parameters along the plan and retraces the streams associated with them, producing a stream plan $\psi \subseteq S .$ A stream plan can be thought of as a serialized computation graph for binding action parameters. It also extracts the function costs F from the plan. Then, it calls BATCH-OPTIMIZATION, the key subroutine that attempts to bind values to these lazy action parameters using batched stream calls (Algorithm 2). The lowest-cost binding $b _ { * }$ is used to bind the lazy action parameters on π, and the resulting plan is returned. If no satisfying binding is found, the set of lazy streams S is updated to remove the streams attempted in stream plan ψ.

Algorithm 2 Batch Stream Optimization   
1: procedure BATCH-OPTIMIZATION(ψ, $F ; c _ { \mathrm { m a x } } )$   
2: B = [{ }] ▷ Initialize list of bindings with an empty binding   
3: for $\langle s , ? x , ? y \rangle \in \psi$ do ▷ Stream s, input $? x ,$ output $\mathring { : } y$   
4: $B ^ { \prime }  [ ]$ ▷ List of new bindings   
5: X ← [SUBSTITUTE(?x, b) for $b \in B ]$ ▷ Batch inputs   
6: for $\langle x ^ { \prime } , y ^ { \prime } \rangle \in \mathbf { z i p } ( \dot { X } , s . f i n ( X ) )$ do ▷ Batch stream call   
7: for b ∈ B do   
8: if SUBSTITUTE $\mathbf { \nabla } ( ? x , b ) = x ^ { \prime }$ then ▷ $x ^ { \prime }$ in binding b   
9: $B ^ { \prime }  B ^ { \prime } + [ b \cup \{ \uparrow y : y ^ { \prime } \} ] \quad \triangleright$ Augment binding   
10: $B  B ^ { \prime }$ ▷ Sort by increasing cost   
11: $C \gets [ 0 . ] * | B |$ ▷ Initialize zero cost per binding   
12: for $\langle \dot { f } , \dot { \ ? } z \rangle \dot { \in } \dot { F }$ do ▷ Function f, input ?z   
13: Z ← [SUBSTITUTE(?z, b) for $b \in B ]$ ▷ Batch inputs   
14: $C \gets \dot { C } + f . f n ( Z )$ ▷ Batch function call   
15: $\mathbf { i f } \ | B | = 0$ or min $. ( C ) > c _ { \mathrm { m a x } }$ then   
16: return None ▷ No satisfying bindings   
17: return B[argmin(C)] ▷ Return the best binding

Algorithm 2 displays the pseudocode for BATCH-OPTIMIZATION, a algorithm for batched computation graph optimization. Although we use it as a subroutine, it can be used for general-purpose constraint satisfaction to, for example, solve for collision-free simulation states. It takes in a stream plan ψ, a set of function costs F, and a maximum cost $c _ { \mathrm { m a x } }$ . It maintains a list of bindings B, mappings from lazy action parameters to values. It processes each stream instance $\langle s , ? x , ? y \rangle$ in the stream plan ψ, where ?x are input parameters and $? y$ are output parameters. On each iteration, it concatenates binding values for input parameters ?x into an input batch X. Then, it calls stream s, generating a batch of output values $y ^ { \prime }$ associated with input values $x ^ { \prime } \in X$ . A new list of bindings $B ^ { \prime }$ is created by joining the existing bindings B with the new parameter values $y ^ { \prime } .$

After all streams in ψ have been called, BATCH-OPTIMIZATION computes the cost of each binding C in batch. In practice, this is interleaved with stream evaluation, where a function is evaluated as soon as its parameter values can be bound, so that the lowest-cost bindings are greedily prioritized in batch streams. BATCH-OPTIMIZATION fails if there are no bindings or if the minimum binding cost is greater than the input maximum cost $c _ { \mathrm { m a x } } .$ . Otherwise, it returns the minimum cost binding in $B .$ In our implementation, BATCH-OPTIMIZATION runs on the CPU but each stream and function call runs on the GPU; however, all these operations could be performed on the GPU [7] to maximize performance at the expense of flexibility.

As an example, suppose SEARCH in BATCH-STREAM returns the following plan π for placing red on the plate.

plan = [move(q0,"?t1","?q1"),   
pick("?q1",red,"?g",p\_r),   
move("?q1","?t2","?q2"),   
place("?q2",red,"?g","?p",plate,p\_p)]

RETRACE-STREAMS and EXTRACT-COSTS will produce the interleaved stream plan $\psi$ and function costs F in Figure 3 corresponding to plan π. Suppose grasps and placements each generate 1,000 outputs per input. The bottleneck in the stream plan ψ is then $\mathtt { i k \_ q s }$ , which has a batch size of 128; however, through sorting by costs and filtering by constraints, the best 128 bindings are prioritized.

## IV. DEMONSTRATION GENERATION

Although in Section III we make novel representation and algorithmic contributions to TAMP itself, these contributions are motivated by investigating the impact of TAMP demon stration quality on visuomotor policy learning. Here, we describe how DemoStream solves TAMP problems to produce demonstration datasets for imitation learning (Section IV-A), is implemented (Section IV-B), supports multiple cost functions (Section IV-C) and motion planners (Section IV D), and integrates with coding agents (Section IV-E).

## A. TAMP Demonstrations

We are interested in generating datasets of observationaction trajectories that successfully complete a task. Here, observations consist of robot proprioception, including the joint positions, gripper position, and end-effector pose, as well as wrist and one or more external camera RGB images. We consider three action spaces in particular to understand whether the best configuration of DemoStream depends on the action space (Section V-A). Specifically, we experiment with joint-space absolute (Joint-Abs) and relative (Joint-Rel) actions as well as task-space relative actions (IK-Rel), implemented using an Inverse Kinematics (IK) controller.

![](images/3afb1363a60feff58224bad39f60f894d8ebadea2909db1713f8ed3ec76d93f3.jpg)  
Fig. 3: Batched Computation Graph. The computation graph (stream plan ψ) corresponding to a plan π involving picking the red block and placing it on the plate. The right column lists the number of bindings |B| after each batch stream call.

To produce a dataset, we first interpret the goal G, for example, using a Large Language Model (LLM) coding agent [20]. Then, we sample an initial state, convert it to a planning state I and solve a TAMP problem for a plan π that achieves the goal G. We execute plans open-loop, retain the resulting observations and actions as demonstrations, and save the demonstrations that reach a goal state We use a fixed set of actions A and streams S across all our experiments in Section V, although additional capabilities can easily be authored by LLM coding agents. After we generate 1,000 successful demonstrations, we train discriminative [10] and generative [21] supervised models that distill the TAMP solver by mapping directly from observations to actions.

We take advantage of batched simulated environments [22] to try hundreds of TAMP solutions in parallel across different initial states. Batched simulation negates the simulation bottleneck, leaving TAMP as the bottleneck at several seconds per successful demonstration (Figure V-C). While we could correspondingly batch TAMP to solve multiple problems simultaneously by modeling an initial-state-generating action and stream, this comes at the expense of optimizing solution quality given a finite amount of GPU memory.

## B. Stream Implementation

Like in prior work [19], we represent the collision volume of the robot as a union of spheres greedily optimized to cover its underlying mesh model [8]. We use cuRoboV2 [23] for batched multi-pose inverse kinematics, implementing ik\_fn. We apply GPU acceleration beyond motion planning by also using it during grasp and placement generation. We implement grasp\_fn by developing a vectorized antipodal grasp sampler that generates 10,000 antipodal points on an object, samples rolls about these axes, filters out grasps that never make contact with the object, and filters out grasps in collision with the object when fully open. We perform these collision checks efficiently by transforming just the spheres on the robot’s end-effector. Not all grasps will result in successful execution (Figure 7); however, through trial-and-error execution, only successful grasps are retained for policy learning. We adapt a similar strategy to implement placement\_fn by randomly generating 10,000 candidate placements of object "?obj" above "?o2" using their bounding boxes, filtering out placements that do not make contact with $" ? 0 2 "$ in the −z direction, and filtering out placements that are in collision with $" ? 0 2 "$ . We use asymmetric collision checking between movable objects to implement $\mathtt { C O 1 1 i s i o n }$ as the conjunction of collisions between spheres of object "?obj1" and the mesh of object $" ? 0 2 "$ and vice versa [8].

## C. Cost Functions

The choice of TAMP cost function to implement distance, along with how much computation is spent optimizing it, directly impacts downstream policy learning. Optimizing a cost function implicitly enforces regularity on the distribution of TAMP solutions and thus the distribution of demonstrations. We hypothesize that it is easier for policies to learn when the distribution of demonstrated actions per state is unimodal and concentrated, even for generative models like Diffusion Policy [21]. Concretely, the implementation of distance affects which grasps, placements, and configurations are selected within BATCH-OPTIMIZATION, serving as targets for motion planning.

In Section V-A, we perform an experiment ablating the following three cost functions and notice dramatic effects on policy learning: Zero cost, Joint-space distance, and endeffector Task-space distance. A zero cost function makes TAMP indifferent to the choice of solution, essentially sampling a feasible solution at random [5]. The joint-space distance is the most natural function because it correlates with the execution duration of the plan. The end-effector taskspace distance computes a weighted sum of the translation and rotation of an end-effector link. Minimizing this cost function minimizes the motion of every link rigidly connected to the end-effector, notably wrist-mounted cameras.

## D. Motion Planning

Building on our investigation of joint- versus taskspace TAMP cost functions (Section IV-C), we also explored multiple ways to plan low-level motions, implementing motion\_fn. We first consider an off-the-shelf motion planner: cuRoboV2’s joint-space trajectory optimizer (cuRoboV2) [23]. To have full control over planning, we also developed a batched sampling-based motion planner applicable both in joint and task space. It takes as a hyperparameter a batched cost function modeling collision and motion edge costs. The planner has two phases: a global exploration phase that creates a Probabilistic Roadmap (PRM) [24] in batch and a local exploitation phase that performs shortcutting [25] in batch. When planning in task space, we sample end-effector poses using batched forward kinematics on uniformly sampled configurations. We check collisions only with the endeffector collision spheres, and lazily map plans to joint space using iterative IK. We consider the following motion planners: joint-space PRM with joint-space costs (J-PRM), joint-space PRM with task-space costs (J-PRM:T), and taskspace PRM with task-space costs (T-PRM).

## E. Agentic Goals

We are interested in generating demonstrations for tasks with novel open-world goals, for example, those described by natural language. For example, consider the Rubiks-Near and Rubiks-Left RoboLab [26] tasks in Figure 7, where the goals are for the Rubik’s cube to be near and left of the bowl. With the rise of LLM coding agents, we adopt the strategy of OWL-TAMP [20] by parameterizing object-object goal predicates with a natural language prompt parameter. Critically, our approach differs in that the agent must implement a batched version of the test that is compatible with efficient planning. LLM coding agents can implement batched tests upfront using offline skills when the set of tasks is known or lazily online to respond to unseen goals. For simulation tasks with programmatic success tests, LLM coding agents can even read the implementation of the test and adapt them to support GPU-accelerated batched planning.

goal = [AgenticGoal("Rubik's cube left of the bowl", rubiks, Pose(rubiks), bowl, Pose(bowl))]

## V. EXPERIMENTS

We conducted experiments to understand how the following factors affect policy Success Rates (SR): 1) action space and cost function (Section V-A), 2) demonstration generator (Section V-B), 3) batch size (Section V-C), 4) task variation (Section V-D), 5) complicated objects (Section V-E), 6) realworld constraints (Section V-F).

Unless indicated otherwise, we used the IsaacLab [22] physics simulator, IK-Rel action space, collected 1,000 demonstrations using DemoStream, added [−0.01, +0.01] uniform noise to actions, used an IK batch size of 128, trained visuomotor BC-RNN policies [10] or Diffusion Policies [21] for 600 and 2,000 epochs, performed 50 rollouts per policy, and reported the best-performing checkpoint from the latter half of training.

## A. Action Space and Cost Function

Our first experiment studies the effect of TAMP (Section IV-C) and motion (Section IV-D) cost functions on policy learning performance in different action spaces (Section IV-A). Table I displays the results of the experiment. Minimizing task-space costs (Task) and using a task-space PRM (T-PRM) results in the highest success rates across action spaces. We also trained Diffusion Policy [21] on this data and found it had even higher success rates of 86% (Joint-Abs), 90% (Joint-Rel), and 86% (IK-Rel). The jointspace PRM with a task-space metric (J-PRM:T) outperforms cuRoboV2 and J-PRM, showing that even soft end-effector motion minimization is beneficial. The zero cost function (Zero), which is representative of prior work [5], fails to train policies with positive success rates in both the Joint-Abs and Joint-Rel action spaces. Additionally, joint-space costs with cuRoboV2 is representative of cuTAMP [6], which DemoStream also outperforms. We hypothesize that minimizing wrist-camera motion is critical for effective visuomotor pol icy learning and note that randomized planners can generate high-quality demonstrations via batched optimization.

TABLE I: Cost Ablation. BC-RNN policy success rates on the Stack-Three task across action spaces, cost functions, and motion planners. Minimizing task-space TAMP and motion costs maximizes policy success rate across action spaces.
<table><tr><td>Action Space</td><td>Cost</td><td>cuRoboV2</td><td>J-PRM</td><td>J-PRM:T</td><td>T-PRM</td></tr><tr><td rowspan="4">Joint-Abs</td><td>Zero</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td>Joint</td><td>22%</td><td>12%</td><td>32%</td><td>34%</td></tr><tr><td>Task</td><td>2%</td><td>0%</td><td>0%</td><td>76%</td></tr><tr><td>Zero</td><td>0%</td><td>0%</td><td>0%</td><td>0%</td></tr><tr><td rowspan="3">Joint-Rel</td><td>Joint</td><td>6%</td><td>10%</td><td>26%</td><td>38%</td></tr><tr><td>Task</td><td>0%</td><td>0%</td><td>0%</td><td>72%</td></tr><tr><td>Zero</td><td>46%</td><td>20%</td><td>34%</td><td>36%</td></tr><tr><td rowspan="2">IK-Rel</td><td>Joint</td><td>46%</td><td>56%</td><td>76%</td><td>42%</td></tr><tr><td>Task</td><td>72%</td><td>70%</td><td>78%</td><td>82%</td></tr></table>

## B. MimicGen Comparison

To understand how DemoStream compares to existing non-TAMP demonstration generation techniques, we compared DemoStream to IsaacLab [22]’s default implementation of MimicGen [2] using the reference demonstrations. Table II shows that although each MimicGen episode is faster than a DemoStream episode, the MimicGen success rate is substantially lower than that of DemoStream (35% vs. 92%) and as a result the time to collect a successful demo is higher (9s vs. 6s). Additionally, MimicGen both produced longer demonstrations because it doesn’t optimize for the best feasible demonstration per problem and trained a worseperforming policy. DemoStream outperformed MimicGen all while not requiring any annotated human demonstrations.

TABLE II: MimicGen vs. DemoStream. DemoStream results in higher demonstration and policy Success Rates (SR) than Mimic-Gen [2], without any task-specific annotated human demonstrations.
<table><tr><td>Method</td><td>Demo SR</td><td>Demo Length</td><td>Time/Demo</td><td>Policy SR</td></tr><tr><td>MimicGen [2]</td><td>35%</td><td>243 ± 16</td><td>9s</td><td>34%</td></tr><tr><td>DemoStream</td><td>93%</td><td>113 ± 6</td><td>6s</td><td>82%</td></tr></table>

## C. Batch Size Scaling

We conducted a scaling experiment to assess how DemoStream performs as the IK stream batch size is scaled from 1 to 256 by factors of 4. Here, an IK batch size of 1 is representative of the performance of prior CPU-only TAMP methods that produce just feasible solutions [5]. Figure 4 shows that as the batch size increases, the demo success rate increases, the mean demonstration length decreases and its distribution narrows, and the policy success rate increases. In this experiment, the mean and standard deviation of demonstration lengths correlate with policy success rate. To understand whether simply short demonstrations were easier to learn, we halved and quartered the DemoStream execution speed, resulting in longer demonstrations, but found that the policy success rate actually slightly increased to 90% and 92% respectively. Thus, mean demonstration length alone is not a good predictor of policy success rate, but Coefficient of Variation, the ratio of the standard deviation to the mean, correlates meaningfully with policy success rate.

![](images/3e495fac073245c344ffeaf3a617403a9fe4cdb9c76d878073ca9e713e1b7fc6.jpg)

![](images/69a58db05c094bf5155b88efb6241cc3536e3682e9ff54d204c485cfc129dca2.jpg)  
Fig. 4: Scaling Batch Size. The effect of scaling the IK batch size by multiples of 4. Left: the mean and standard deviation of the demonstration length. Right: the demonstration and policy Success Rate (SR). Large batch sizes decrease demonstration length and dispersion, resulting in higher policy success rates.

## D. TAMP Tasks

![](images/124ccfa1a7e7b88315d0ad04fb7bea8b0e855470d7245e18770c95653e8c4af2.jpg)  
Fig. 5: Stacking Tasks. Left: stack four blocks (Stack-Four, long-horizon). Center: stack three blocks in the bin (Stack-Bin, obstacles). Right: stack three blocks using two arms (Multi-Arm).

We tested DemoStream on more difficult TAMP tasks [16] requiring long horizons (Stack-Four), collision-aware motion (Bin-Stack), collision-aware placement (Pack-Two, 100% Blocked), varying task plans (Pack-Either, 50% Blocked), and multiple robots (Multi-Arm). We defined four stacking tasks (Figure 5): stack two (Stack-Two) and four blocks (Stack-Four), stack three blocks in a bin (Bin-Stack), stack three blocks using two arms (Multi-Arm). We used absolute joint-space actions in Multi-Arm. We also considered the following four packing tasks (Figure 6): pack two blocks on the plate (Pack-Two), pack either block on the plate (Pack-Either), and pack the red block where the blue block starts on the plate with 50% (50% Blocked) and (100% Blocked) probability. Table III shows that DemoStream is able to train high-performing policies despite increased task difficulty.

## E. RoboLab Tasks

To validate that DemoStream works on more complex objects and open-world goals (Rubiks-Near, Rubiks-Left), we experimented on several RoboLab [26] IsaacLab Arena tasks (Figure 7). The tasks are: place the Rubik’s cube in

![](images/dffd4eb3ee2d197c9cc09cd1f58a15cffda0e3f40a9adadfa6a5d2ef5fa96b2b.jpg)  
Fig. 6: Packing Tasks. Left: pack the two blocks on the plate (Pack-Two). Center: pack the red block on the plate (50% Blocked). Right: pack the red block on the obstructed plate (100% Blocked).

TABLE III: Stacking and Packing Results. The Demonstration Success Rate (Demo SR) and policy Success Rate (SR) across the stacking (Figure 5) and packing (Figure 6) tasks.
<table><tr><td>Task</td><td>Demo SR</td><td>BC-RNN SR</td><td>Diffusion SR</td></tr><tr><td>Stack-Two</td><td>100%</td><td>100%</td><td>90%</td></tr><tr><td>Stack-Four</td><td>72%</td><td>54%</td><td>64%</td></tr><tr><td>Bin-Stack</td><td>97%</td><td>82%</td><td>98%</td></tr><tr><td>Multi-Arm</td><td>54%</td><td>12%</td><td>50%</td></tr><tr><td>Pack-Two</td><td>93%</td><td>94%</td><td>96%</td></tr><tr><td>Pack-Either</td><td>94%</td><td>90%</td><td>82%</td></tr><tr><td>50% Blocked</td><td>99%</td><td>88%</td><td>86%</td></tr><tr><td>100% Blocked</td><td>99%</td><td>94%</td><td>90%</td></tr></table>

![](images/5f2ed2f11019d27d68803a186d01e6241c5d7943df00f72b5edbd560329aa667.jpg)  
Fig. 7: RoboLab Tasks. Left: put the Rubik’s cube and banana in the bowl (Rubiks-Banana). Center: place the mustard on the raisins (Mustard-Raisin). Right: drop the mustard in the bin (Mustard-Bin). the bowl (Rubiks-Bowl), place the Rubik’s cube near the bowl (Rubiks-Near), place the Rubik’s cube left of the bowl (Rubiks-Left), place the banana in the bowl (Banana-Bowl), place the Rubik’s cube and banana in the bowl (Rubiks-Banana), place the mustard bottle on the raisin box (Mustard-Raisin), and drop the mustard bottle in the bin (Mustard-Bin). Table IV shows that although DemoStream requires more attempts to stably grasp objects, it distills effective policies.

TABLE IV: RoboLab Results. Demonstration Success Rate and policy Success Rate (SR) on the RoboLab tasks (Figure 7).
<table><tr><td>Task</td><td>Demo SR</td><td>BC-RNN SR</td><td>Diffusion SR</td></tr><tr><td>Rubiks-Bowl</td><td>51%</td><td>86%</td><td>76%</td></tr><tr><td>Rubiks-Near</td><td>63%</td><td>96%</td><td>56%</td></tr><tr><td>Rubiks-Left</td><td>27%</td><td>48%</td><td>28%</td></tr><tr><td>Banana-Bowl</td><td>40%</td><td>78%</td><td>44%</td></tr><tr><td>Rubiks-Banana</td><td>3%</td><td>18%</td><td>22%</td></tr><tr><td>Mustard-Raisin</td><td>55%</td><td>88%</td><td>34%</td></tr><tr><td>Mustard-Bin</td><td>17%</td><td>86%</td><td>70%</td></tr></table>

## F. Real World

We conducted a real-world demonstration-collection and policy-learning experiment on a DROID [27]-style Franka Panda to understand DemoStream’s viability for direct real-world learning. The human input to the system is a command-line natural language prompt describing the task (Section IV-E). Unlike in simulation, TAMP cannot directly read the geometry and pose of objects. To address this, we adopted the strategy of Curtis et al. [28] for open-world TAMP, where object geometries are generically estimated from a point cloud segmented by SAM3 [29].

TABLE V: Real World Results. The Demonstration Success Rate (Demo SR), demonstration length, time per demo, and policy Success Rate (SR) on the real-world tasks (Figure 1).
<table><tr><td>Real Task</td><td>Demo SR</td><td>Demo Length</td><td>Time/Demo</td><td>Policy SR</td></tr><tr><td>Stack-Two</td><td>72%</td><td> $1 7 8 \pm 1 3$ </td><td>48s</td><td>65%</td></tr><tr><td>Banana-Bin</td><td>83%</td><td> $2 1 4 \pm 1 9$ </td><td>49s</td><td>75%</td></tr><tr><td>Apple-Lime</td><td>79%</td><td> $3 2 3 \pm 2 8$ </td><td>69s</td><td>60%</td></tr></table>

We considered three tasks (Figure 1): stack the red block on the blue block (Stack-Two), drop the banana in the bin (Banana-Bin), and pack the apple and lime on the plate (Apple-Lime). DemoStream collected 75 demonstrations per task at a throughput of 66 demonstrations per hour. The pipeline was autonomous without intervention aside from human-performed randomized resets and success labeling; for certain tasks, TAMP could automate these low-effort operations as well. We trained a Joint-Abs Diffusion Policy [21] per task. The policy’s observations were proprioception as well as 200x200 wrist and left-external camera images. We evaluated the final checkpoints over 20 task trials. Table V shows that DemoStream trains policies that are successful on average 67% of the time on purely real-world data. See our website for rollout videos: https://demostream.github.io. <sup>2</sup>

## REFERENCES

[1] J. Bjorck, F. Castaneda, N. Cherniadev, X. Da, R. Ding, L. Fan,˜ Y. Fang, D. Fox, F. Hu, S. Huang et al., “Gr00t n1: An open foundation model for generalist humanoid robots,” arXiv preprint arXiv:2503.14734, 2025.

[2] A. Mandlekar, S. Nasiriany, B. Wen, I. Akinola, Y. Narang, L. Fan, Y. Zhu, and D. Fox, “Mimicgen: A data generation system for scalable robot learning using human demonstrations,” in 7th Annual Conference on Robot Learning, 2023.

[3] C. R. Garrett, R. Chitnis, R. Holladay, B. Kim, T. Silver, L. P. Kaelbling, and T. Lozano-Perez, “Integrated Task and Motion Planning,”´ Annual Review of Control, Robotics, and Autonomous Systems, 2021.

[4] M. J. McDonald and D. Hadfield-Menell, “Guided imitation of task and motion planning,” in Conference on Robot Learning. PMLR, 2022, pp. 630–640.

[5] M. Dalal, A. Mandlekar, C. Garrett, A. Handa, R. Salakhutdinov, and D. Fox, “Imitating task and motion planning with visuomotor transformers,” arXiv preprint arXiv:2305.16309, 2023.

[6] W. Shen, C. Garrett, N. Kumar, A. Goyal, T. Hermans, L. P. Kaelbling, T. Lozano-Perez, and F. Ramos, “Differentiable gpu-parallelized task´ and motion planning,” in Robotics: Science and Systems, 2025.

[7] L. Chen, S. R. Iyer, and Z. Kingston, “Differentiable particle optimization for fast sequential manipulation,” arXiv preprint arXiv:2510.07674, 2025.

[8] C. Garrett and F. Ramos, “Schedulestream: Temporal planning with samplers for gpu-accelerated multi-arm task and motion planning & scheduling,” arXiv preprint arXiv:2511.04758, 2025.

[9] D. A. Pomerleau, “Alvinn: An autonomous land vehicle in a neural network,” in Advances in neural information processing systems, 1989, pp. 305–313.

<sup>2</sup>We disclose the use of Claude and ChatGPT to assist with code implementation, facilitating the experiments, and proofreading the manuscript.

[10] A. Mandlekar, D. Xu, J. Wong, S. Nasiriany, C. Wang, R. Kulkarni, L. Fei-Fei, S. Savarese, Y. Zhu, and R. Mart´ın-Mart´ın, “What matters in learning from offline human demonstrations for robot manipulation,” in Conference on Robot Learning (CoRL), 2021.

[11] C. R. Garrett, A. Mandlekar, B. Wen, and D. Fox, “Skillgen: Automated demonstration generation for efficient skill learning and deployment,” in 8th Annual Conference on Robot Learning, 2024.

[12] L. Yang, H. Suh, T. Zhao, B. P. Graesdal, T. Kelestemur, J. Wang, T. Pang, and R. Tedrake, “Physics-driven data generation for contact-rich manipulation via trajectory optimization,” arXiv preprint arXiv:2502.20382, 2025.

[13] C. Li, M. Xu, A. Bahety, H. Yin, Y. Jiang, H. Huang, J. Wong, S. Garlanka, C. Gokmen, R. Zhang et al., “Momagen: Generating demonstrations under soft and hard constraints for multi-step bimanual mobile manipulation,” arXiv preprint arXiv:2510.18316, 2025.

[14] K. Lin, A. Mandlekar, C. R. Garrett, N. Chernyadev, Y. Fang, R. Ding, Y. Xie, J. Tran, L. Fan, and Y. Zhu, “Humanoidmimicgen: Data generation for loco-manipulation via whole-body planning,” arXiv preprint arXiv:2605.27724, 2026.

[15] D. Hadfield-Menell, C. Lin, R. Chitnis, S. Russell, and P. Abbeel, “Sequential quadratic programming for task plan optimization,” in 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2016, pp. 5040–5047.

[16] C. R. Garrett, T. Lozano-Perez, and L. P. Kaelbling, “PDDLStream:´ Integrating Symbolic Planners and Blackbox Samplers,” in ICAPS, 2020.

[17] A. Mandlekar, C. Garrett, D. Xu, and D. Fox, “Human-in-the-loop task and motion planning for imitation learning,” in 7th Annual Conference on Robot Learning, 2023.

[18] D. McDermott, M. Ghallab, A. Howe, C. Knoblock, A. Ram, M. Veloso, D. Weld, and D. Wilkins, “PDDL: The Planning Domain Definition Language,” Yale Center for Computational Vision and Control, Tech. Rep., 1998.

[19] B. Sundaralingam, S. K. S. Hari, A. Fishman, C. Garrett, K. Van Wyk, V. Blukis, A. Millane, H. Oleynikova, A. Handa, F. Ramos, N. Ratliff, and D. Fox, “Curobo: Parallelized collision-free robot motion generation,” in 2023 IEEE International Conference on Robotics and Automation (ICRA), 2023, pp. 8112–8119.

[20] N. Kumar, F. Ramos, D. Fox, and C. R. Garrett, “Open-world task and motion planning via vision-language model inferred constraints,” in Workshop on Language and Robot Learning: Language as an Interface, 2024.

[21] C. Chi, Z. Xu, S. Feng, E. Cousineau, Y. Du, B. Burchfiel, R. Tedrake, and S. Song, “Diffusion policy: Visuomotor policy learning via action diffusion,” The Int’l Journal of Robotics Research, 2023.

[22] M. Mittal, P. Roth, J. Tigue, A. Richard, O. Zhang, P. Du, A. Serrano-Munoz, X. Yao, R. Zurbrugg, N. Rudin¨ et al., “Isaac lab: A gpuaccelerated simulation framework for multi-modal robot learning,” arXiv preprint arXiv:2511.04831, 2025.

[23] B. Sundaralingam, A. Murali, and S. Birchfield, “curobov2: Dynamicsaware motion generation with depth-fused distance fields for high-dof robots,” arXiv preprint arXiv:2603.05493, 2026.

[24] L. E. Kavraki, P. Svestka, J.-C. Latombe, and M. H. Overmars, “Probabilistic roadmaps for path planning in high-dimensional configuration spaces,” IEEE transactions on Robotics and Automation, vol. 12, no. 4, pp. 566–580, 1996.

[25] R. Geraerts and M. H. Overmars, “Creating high-quality paths for motion planning,” The international journal of robotics research, vol. 26, no. 8, pp. 845–863, 2007.

[26] X. Yang, R. Dagli, A. Zook, H. Hadfield, A. Goyal, S. Birchfield, F. Ramos, and J. Tremblay, “Robolab: A high-fidelity simulation benchmark for analysis of task generalist policies,” arXiv preprint arXiv:2604.09860, 2026.

[27] A. Khazatsky, K. Pertsch, S. Nair, A. Balakrishna, S. Dasari, S. Karamcheti, S. Nasiriany, M. K. Srirama, L. Y. Chen, K. Ellis et al., “Droid: A large-scale in-the-wild robot manipulation dataset,” arXiv preprint arXiv:2403.12945, 2024.

[28] A. Curtis, X. Fang, L. P. Kaelbling, T. Lozano-Perez, and C. R.´ Garrett, “Long-horizon manipulation of unknown objects via task and motion planning with estimated affordances,” in IEEE International Conference on Robotics and Automation (ICRA), 2022.

[29] N. Carion, L. Gustafson, Y.-T. Hu, S. Debnath, R. Hu, D. Suris Coll-Vinent, C. Ryali, K. V. Alwala, H. Khedr, A. Huang et al., “Sam 3: Segment anything with concepts,” in International conference on learning representations, vol. 2026, 2026, pp. 138 846–138 923.