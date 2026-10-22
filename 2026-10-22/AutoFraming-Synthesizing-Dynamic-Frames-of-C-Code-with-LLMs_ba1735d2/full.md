# AutoFraming: Synthesizing Dynamic Frames of C Code with LLMs

Han Zhou

University of Missouri–Kansas City

Yu Luo

Dianxiang Xu

Kansas City, USA

University of Central Missouri

hzb4f@umkc.edu

Warrensburg, USA

yuluo@ucmo.edu

University of Missouri–Kansas City

Kansas City, USA

dxu@umkc.edu

Abstract—Large language models (LLMs) have shown considerable promise in synthesizing formal specifications from source code. However, existing approaches primarily focus on functional contracts and loop invariants, while largely neglecting frame conditions, which specify the memory locations that may or may not be modified during execution and constitute an essential part of formal specifications. To address this limitation, we propose an LLM-based approach for synthesizing ACSL frame specifications alongside functional contracts from C code. Our approach analyzes code property graphs (CPGs) to extract frame-relevant information from the input code and leverages this information to guide the LLM in generating accurate frame conditions. The resulting specifications capture both region-separation constraints and data dependencies that govern variable modifications. Experimental results demonstrate the effectiveness of our approach. On a benchmark of 84 ACSL-compliant functions from non-trivial real-world systems, our method successfully synthesizes errorfree frame specifications for 92.9% of the functions, representing a 179% relative improvement over state-of-the-art techniques. On a second benchmark consisting of 51 functions, the success rate increases from 80.4% to 96.1%. Furthermore, when integrated as an auxiliary component into existing LLM-based ACSLsynthesis frameworks, our approach improves frame-specification generation performance by up to 61.9%.

Index Terms—deductive verification, frame specification, ACSL, Frama-C, code property graph, large language models, program analysis

## I. INTRODUCTION

Large language models (LLMs) have recently shown strong potential for generating formal specifications from source code. For C programs, several approaches [1]–[5] have explored using LLMs to synthesize ACSL specifications, often with feedback from static analyzers or verification tools. These techniques can generate useful functional specifications, such as preconditions, postconditions, and loop invariants, and can substantially reduce the manual effort required for deductive verification. However, little attention has been paid to frame conditions: the part of a specification that describes what memory a function may affect, what values those effects depend on, and what memory must remain independent. Without accurate frame information, a verified function cannot be safely reused by callers, because the caller cannot know which parts of the program state are preserved.

This limitation is especially important for C programs. Recent approaches such as AutoSpec [4] and Preguss [5] push LLM-based specification generation further by combining models with program information and verification feedback. However, their treatment of frame information remains limited. They primarily optimize for whether the generated specification can help discharge verification conditions, while frame completeness is not directly measured. In particular, evaluations based mainly on WP proof success may overlook missing memory-dependence information or imprecise aliasing assumptions, because a proof can succeed even when the contract does not fully describe what state a caller may rely on after the function returns. This gap is difficult to close for C: pointers, arrays, pointer arithmetic, loops, and library calls can hide which memory regions may overlap and which input values may influence each write. As a result, a generated contract may look correct to the verifier but still be too weak for precise modular reasoning. Section II shows a small example where an apparently verified contract still permits an invalid aliased call.

To address this problem, we propose AutoFraming, an approach for synthesizing frame specifications for C programs. AutoFraming analyzes frame-relevant information from a code property graph (CPG) [6], including possible memory interference, caller-visible writes, loop-indexed write ranges, and data dependencies. It then projects this information into a fixedframe skeleton that captures which memory regions must be kept independent and which inputs may influence each written location. The LLM is used to assemble the full contract and repair remaining verification failures, while the frame-critical structure is generated and preserved during refinement.

We evaluate AutoFraming on two benchmarks: 84 functions from real-world C systems and 51 academic verification examples. On the real-world benchmark, AutoFraming synthesizes error-free frame specifications for 92.9% of functions, a 179% relative improvement over the state-of-the-art baseline. On the academic benchmark, it improves the errorfree frame rate from 80.4% to 96.1%. When used as an auxiliary frame-synthesis component for existing specificationgeneration tools, AutoFraming further improves error-free frame performance by up to 61.9%. Ablation results show that the CPG-derived frame information and deterministic projections are essential to the improvement.

The contribution of this paper is summarized as follows: AutoFraming provides an effective approach to the novel problem of synthesizing error-free frame conditions with LLMs. It conducts frame analysis, including caller-visible roots, access windows, interfering pointer pairs, write-set locations, and dependence footprints. It then projects frame-related information into ACSL specification, including region separation for caller-visible memory regions and data dependence for writeset locations. Moreover, AutoFraming can be used in other ACSL synthesis frameworks, such as Preguss and AutoSpec, to improve their capability of synthesizing frame conditions.

## II. A MOTIVATING EXAMPLE

Consider the C function parse\_value, shown below, which reads two values from the memory region pointed to by buf and stores their sum in the memory location pointed to by eaten.

```c
Listing 1. Source code of the motivating example.
int parse_value(const uint16_t buf, uint16_t len,
,→ uint16_t <sub>*</sub>eaten) {
if (len < 2 || buf == NULL || eaten == NULL) return -1;
<sub>*</sub>eaten = buf[0];
<sub>*</sub>eaten = (uint16_t)(<sub>*</sub>eaten + buf[1]);
return 0;
}
```

To call parse\_value, the client needs to provide actual parameters for <sub>\*</sub>buf and <sub>\*</sub>eaten, such as the following:

```c
uint16_t buf[2] = { 3, 5 };
uint16_t out;
parse_value(buf, 2, &out);
```

Without a formal specification for parse\_value, clients have no precise way to determine whether a call satisfies the function’s assumptions or achieves its intended behavior. To address this issue, we can prompt an LLM to synthesize the following ACSL [7] specification directly from the source code of parse\_value:

```c
Listing 2. Preliminary ACSL specification generated by prompting an LLM
/<sub>*</sub>@ requires len > 0;
requires buf != \null;
ensures \result == 0 || \result == -1; <sub>*</sub>/
int parse_value(const uint16_t <sub>*</sub>buf, uint16_t len,
,→ uint16_t eaten);
```

The generated comment enclosed by /<sub>\*</sub>@ and <sub>\*</sub>/ defines the ACSL contract for parse\_value, including its precondition (the requires clauses) and postcondition (the ensures clause). The precondition specifies the conditions that callers must satisfy before invoking parse\_value; specifically, len must be positive and buf must be non-null. The postcondition specifies the property that must hold when parse\_value returns, namely that the return value is either 0 or -1.

The ACSL specification can be verified with Frama-C and one of its plug-ins (e.g., WP or Eva) [8]. Given a function and its ACSL contract, Frama-C generates a set of proof obligations, called goals, each expressed as a logical formula whose validity establishes a specific aspect of the contract. These goals are then dispatched to automated SMT solvers. The proportion of discharged goals, referred to as the goal ratio (proved/total), indicates the extent to which the contract has been verified. A goal ratio of 1 (i.e., a full proof) means that every proof obligation has been discharged, establishing that the implementation satisfies the specified contract for all executions. Otherwise, the result is only a partial proof, indicating that some proof obligations remain unresolved. It is important to note, however, that a full proof does not necessarily imply that the code is correct. Rather, it only guarantees conformance to the given specification. If the specification is incomplete or inaccurate, the implementation may still satisfy it while failing to capture the intended behavior. For example, Frama-C with the WP plugin reports a full proof for the preliminary LLM-generated ACSL contract shown above, even though the contract only partially captures the intended behavior of parse\_value. The generated precondition is overly restrictive. It requires len to be positive and buf to be non-null, even though parse\_value handles the cases where len is non-positive or buf is NULL without error. To improve the quality of synthesized ACSL specifications, existing approaches leverage the results of static code analysis to prompt LLMs and refine the specifications. The following ACSL specification is generated by Preguss [5], the state-ofthe-art method:

```c
Listing 3. ACSL specification generated by existing approaches
/<sub>*</sub>@ requires \valid_read(buf + (0..len-1));
requires \valid(eaten);
assigns <sub>*</sub>eaten;
ensures \result == 0 || \result == -1; <sub>*</sub>/
int parse_value(const uint16_t <sub>*</sub>buf, uint16_t len,
,→ uint16_t <sub>*</sub>eaten);
```

The requires clauses specify the validity conditions for buf and eaten. The assigns clause, assigns <sub>\*</sub>eaten;, defines the function’s frame condition by specifying that only the memory location referenced by eaten may be modified during execution. Consequently, all other program state, including buf, len, and the memory referenced by buf, must remain unchanged. More generally, the assigns clause consists of a list of variable names or memory locations (referred to as a write set or footprint).

However, the above frame specification is insufficient even though the verification result is a full proof. Consider the following call:

```c
uint16_t buf[2] = { 3, 5 };
parse_value(buf, 2, &buf[1]);
```

The call satisfies both precondition clauses, i.e., buf is readable and &buf[1] is valid. Although the intended value for <sub>\*</sub>eaten (i.e., buf[1]) is buf[0] + buf[1] = 8, the actual outcome is 3 + 3 = 6 because parse\_value reads buf[1] after writing through <sub>\*</sub>eaten, so the write clobbers buf[1] first and the call computes 3 + 3.

The problem arises because the same object, buf, is passed to both pointer parameters, buf and eaten. However, parse\_value operates correctly only when these pointers refer to separate, non-overlapping memory locations. Therefore, the ACSL specification must explicitly capture this separation requirement to prevent invalid function calls and ensure correct verification.

In summary, existing LLM-based approaches to ACSL specification synthesis do not explicitly address the frame problem, particularly dynamic frame conditions such as the memory separation requirement illustrated above. Dynamic frame conditions precisely specify the memory locations that a function is permitted to read or modify, ensuring that changes to one object cannot inadvertently affect the state of another unrelated object. The goal of this paper is to enable LLMs to synthesize ACSL specifications with accurate frame conditions for C programs that are ACSL-compliant, i.e., programs whose behavior can be completely specified in ACSL. The following specification for the example illustrates the desired result:

```c
Listing 4. ACSL specification with dynamic frames
/<sub>*</sub>@ requires \valid_read(buf + (0..len-1));
requires \valid(eaten);
requires \separated(buf + (0..len-1), eaten);
assigns <sub>*</sub>eaten
\from buf[0..len-1], len;
ensures \result == 0 || \result == -1; <sub>*</sub>/
int parse_value(const uint16_t <sub>*</sub>buf, uint16_t len,
,→ uint16_t eaten);
```

The additional requires clause, \separated(buf + (0..len-1), eaten); specifies that the memory region referenced by buf must be disjoint from the memory location referenced by eaten. The assigns clause, <sub>\*</sub>eaten \from buf[0..len-1], len; defines both the function’s write set and the data dependencies of the modified location, stating that only <sub>\*</sub>eaten may be updated and that its value depends solely on buf[0..len-1] and len. Synthesizing the region separation requirement and the data dependency of the write set is a challenging problem. Because C programs do not expose their memory footprint directly in the source syntax. Pointer aliasing, array ranges, pointer arithmetic, loops, globals, and library calls can hide both the regions that may interfere with one another and the inputs that may affect each write. Therefore, synthesizing accurate dynamic frames requires recovering caller-visible memory regions, separation assumptions, and write dependencies from program structure rather than treating frame clauses as ordinary text to be generated.

## III. THE DESIGN OF AUTOFRAMING

## A. Overview

AutoFraming leverages the code property graph (CPG) representation of C code to synthesize its frame specifications. Given an unannotated function, AutoFraming generates an ACSL specification whose frame part describes not only the memory locations that may be modified, but also the memory regions that must remain disjoint and the caller-visible inputs that may influence each write. Although AutoFraming outputs a complete ACSL specification, this paper focuses on the synthesis of frame conditions. Figure 1 shows the three-stage workflow of AutoFraming.

Stage 1: AutoFraming first constructs the call graph and analyzes functions in bottom-up order, ensuring that verified callee contracts are available when analyzing their callers. For each function $f ,$ AutoFraming builds a CPG and extracts the frame-related information $\Phi _ { f } .$ , which captures the callervisible roots of memory accesses, accessed regions of pointer roots, pointer pairs whose aliasing may interfere with callervisible writes, write-set locations used as frame carriers, and the dependence footprint of each write-set location.

![](images/ef5c3a3c6257a999218201c7372c47d8cb58c2b8ccc32f62492b5fd42fff0ec6.jpg)  
Fig. 1. AutoFraming Pipeline for CPG-guided Frame Specification Synthesis.

Stage 2: AutoFraming projects $\Phi _ { f }$ into a frame skeleton. The separation projection turns interfering pointer regions into disjointness requirements, while the dependence projection attaches caller-visible dependence footprints to write-set locations. AutoFraming treats projected frame obligations as required constraints. A candidate is rejected if it deletes or weakens them, but the current implementation may still allow equivalent or stronger rewritings that change the proof burden.

Stage 3: AutoFraming asks the LLM to assemble a verifiercheckable contract around the frame skeleton and uses Frama-C feedback to repair remaining failures. During refinement, candidate contracts are accepted only if they preserve the projected frame conditions, avoid degenerate frames, and make proof progress. This design lets AutoFraming use the LLM for auxiliary annotations and repair while keeping the framecritical content grounded in program analysis.

## B. CPG-Based Frame Analysis

For each function f, AutoFraming extracts from its CPG the information needed to synthesize the dynamic frame. A CPG overlays three program views on one graph [9]: the abstract syntax tree (AST), which exposes declarations and assignment targets; the control-flow graph (CFG), which exposes branches, loops, and execution paths; and the program dependence graph (PDG), which exposes data and control dependencies. This combination is essential because frame synthesis is not a purely syntactic task: write-set locations come from syntax, region extents often come from loops and guards, and dependence footprints require following value and control flow.

TABLE I  
THE FRAME-INFORMATION SET $\Phi _ { f }$ EXTRACTED FROM THE CPG.
<table><tr><td>Component Meaning</td><td></td><td>CPG view</td><td>Used by</td></tr><tr><td> $\mathcal { R } _ { f }$ </td><td>caller-visible roots: parameters and AST globals touched by f</td><td></td><td>Sep, Dep</td></tr><tr><td> $\Omega _ { f }$ </td><td>access window of each pointer root</td><td>AST, CFG</td><td>Sep, Dep</td></tr><tr><td> $\boldsymbol { \mathcal { T } _ { f } }$ </td><td>interfering pointer-root pairs requiring AST, PDG separation</td><td></td><td>Sep</td></tr><tr><td> $\mathcal { L } _ { f }$ </td><td>normalized write-set locations used as AST, CFG assigns carriers</td><td></td><td>Dep</td></tr><tr><td> $\Delta _ { f }$ </td><td>dependence footprint of each write set PDG location</td><td></td><td>Dep</td></tr></table>

AutoFraming represents the frame-related information as

$$
\Phi _ { f } = ( \mathcal { R } _ { f } , \Omega _ { f } , \mathcal { I } _ { f } , \mathcal { L } _ { f } , \Delta _ { f } ) .
$$

The first three components help address the region-separation issue: which caller-visible regions must be disjoint? The last two deal with the data dependency issue: what does each write set location depend on?

a) CPG-to-information extraction.: AutoFraming derives each component of $\Phi _ { f }$ by querying specific CPG nodes and edges rather than by passing the whole graph to the LLM. AST nodes provide syntactic roots and memory-access forms: parameter and global declarations identify caller-visible roots; dereference, array-subscript, field-access, assignment, and update nodes identify reads and writes; and declaration and cast nodes provide type information for alias compatibility. CFG edges provide the execution context of these accesses: loop headers, back edges, and branch guards are used to recover loop-indexed access ranges and conditional bounds. PDG edges provide dependence information: data-dependence edges connect values to the expressions they influence, while control-dependence edges connect guards to the writes whose execution they control. Together, these CPG views determine the five information components used by AutoFraming: callervisible roots, access windows, interfering pointer pairs, writeset locations, and dependence footprints.

b) Caller-visible roots.: $\mathcal { R } _ { f }$ contains the formal parameters and globals that the function may read or write, each recorded with its type. These roots are the only memory roots that can appear in the external frame of $f .$ Local variables are excluded because their storage does not escape the function call. Pointer roots are recorded with their pointedto types, since type compatibility determines whether two roots may denote overlapping storage. For a pointer-typed root $p$ with C type $^ { T * }$ , we call $T$ the pointed-to type of $p .$ For example, the pointed-to type of const uint16\_t <sub>\*</sub>buf is uint $^ \lfloor 6 \_ t$ , and the pointed-to type of uint16\_t <sub>\*</sub>eaten is also uint16\_t. Hence, buf and eaten may refer to overlapping uint $^ { 1 6 } - \mathtt { t }$ storage at a call site.

AutoFraming uses pointed-to types as a conservative aliascompatibility filter. Two pointer roots are considered compatible when their accesses may refer to the same kind of object storage. This includes roots with the same pointed-to type after removing qualifiers such as const and $\mathtt { v o l a t i l e }$ , roots accessed through byte-level pointer types such as char <sub>\*</sub> or unsigned char <sub>\*</sub>, roots accessed through generic void <sub>\*</sub> interfaces, and roots whose type information is weakened by casts. Conversely, roots with clearly different non-byte pointed-to types, such as int <sub>\*</sub> and double <sub>\*</sub>, are filtered out when the function contains no casts, byte-level accesses, or generic pointer operations that could make overlap relevant.

c) Access windows.: For each pointer root $p \in \mathcal { R } _ { f } ,$ $\Omega _ { f } ( p )$ records the entry-expressible region of $p$ that the function may access:

$$
\Omega _ { f } ( p ) = \mathsf { w i n d o w } \big ( \mathsf { A c c e s s e s } _ { f } ( p ) \big ) .
$$

The access set Accesses $_ f ( p )$ is collected by traversing AST memory-access nodes rooted at $p ,$ including dereferences, array subscripts, field accesses, and summarized library calls. It then uses CFG loop and guard edges to attach path conditions and ranges to these accesses. A constant access such as p[c] contributes offset $c ;$ a loop-indexed access such as $\mathrm { ~ p ~ } [ \mathrm { ~ i ~ } ]$ contributes the loop range recovered from the loop header and back edge; and an access guarded by a condition such as i $< \mathrm { ~ n ~ }$ contributes a range bounded by n when n is available at function entry. These windows allow AutoFraming to emit separation constraints over accessed regions rather than over pointer names alone.

d) Interfering pointer pairs.:

$$
\begin{array} { r } { \mathbb { Z } _ { f } = \{ \{ p , q \} \mid p , q \in \mathcal { R } _ { f } \wedge \mathsf { p t r } ( p ) \wedge \mathsf { p t r } ( q ) \qquad } \\ { \wedge \mathsf { c o m p a t i b l e } ( p , q ) \wedge \mathsf { i n t e r f e r e s } _ { f } ( p , q ) \} . } \end{array}
$$

Here, compatible $( p , q )$ applies the pointed-to-type filter: it keeps same-type, byte-level, generic-pointer, and cast-related pairs, and filters out clearly incompatible typed accesses. The predicate interferes $_ f ( p , q )$ checks whether aliasing the two roots can change a caller-visible write or the value written:

$$
\begin{array} { r } { \mathsf { i n t e r f e r e s } _ { f } ( p , q ) \triangleq \big ( \mathsf { w r i t e s } _ { f } ( p ) \wedge \mathsf { a f f e c t s W i t e } _ { f } ( q ) \big ) } \\ { \mathsf { V } \left( \mathsf { w r i t e s } _ { f } ( q ) \wedge \mathsf { a f f e c t s W i t e } _ { f } ( p ) \right) . } \end{array}
$$

A root satisfies afectsWrite when it either roots a write or is read into a value that flows to a write. This interference test prevents AutoFraming from over-restricting callers by separating unrelated pointer parameters.

e) Write-set locations.: $\mathcal { L } _ { f }$ contains the caller-visible locations that may be modified by $f ,$ normalized into ACSL assignable locations. AutoFraming first collects write targets from AST write nodes in the CPG, including assignment left-hand sides, array-cell updates, structure-field updates, and dereferences. We use w to denote one such write target. A target w is kept only when it can be expressed using a callervisible root in $\mathcal { R } _ { f }$ , such as a parameter or global. If a write is performed through a local pointer alias, AutoFraming uses CPG data-dependence edges only to rewrite the target back to its caller-visible root; the local alias itself is never emitted in an ACSL frame clause.

When a write target w is an array access indexed by a loop variable, for example $w = A [ i ]$ , AutoFraming uses CFG loop and guard edges to recover range $\mathbf { \Phi } _ { \mathrm { ~ \it ~ f ~ } } ( i ) \ = \ ( l o , h i )$ and normalizes the target to the whole loop-written range:

$$
\mathsf { n o r m } _ { f } ( w ) = \left\{ \begin{array} { l l } { A [ l o . . h i - 1 ] , } & { \mathrm { i f ~ } w = A [ i ] , } \\ { w , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

The write-set location set is therefore

$$
\mathcal { L } _ { f } = \{ \mathsf { n o r m } _ { f } ( w ) \mid w \in \mathsf { W r i t e s } _ { f } ^ { v i s } \} ,
$$

where ${ \mathsf { W i t e s } } _ { f } ^ { v i s }$ denotes the collected write targets that can be expressed using caller-visible roots. These locations supply the left-hand side of ACSL assigns clauses. In AutoFraming, however, the write location is not the main synthesis target; it is the carrier to which dependence information is attached.

f) Dependence footprints.: For each write set location $L \in \mathcal { L } _ { f } , \Delta _ { f } ( L )$ contains the caller-visible inputs whose values may influence the value written to $L \colon$

$$
\Delta _ { f } ( L ) = \{ r \in \mathcal { R } _ { f } \mid r \sim \sp { d a t a / c o n t r o l } _ { \bf \sf { W } \bar { 1 } \bar { t } \bar { \bf { e } } ( L ) } \} .
$$

The relation $ _ { f } ^ { d a t a / c o n t r o l }$ is computed from the PDG. It includes data dependencies, where a root contributes to the expression being written, and control dependencies, where a root controls whether a write occurs. Local intermediate variables are followed transitively but are not reported; the final dependence set contains only caller-visible inputs. Pointer reads are lifted to accessed regions using $\Omega _ { f }$ , so a loop that writes an output from buf[i] contributes the accessed region of buf, rather than only the syntactic root buf.

## C. Synthesis of Region-Separation Constraints

The first dynamic-frame synthesis issue is region separation: for pointer-accessed memory, a frame specification must identify the caller-visible regions that must be disjoint for the declared frame to be valid. If two such regions may overlap at a call site, a write through one pointer can affect a read or write through another, making the specified frame unsound or incomplete for modular reasoning.

Regional separation analysis is difficult for C code because the required disjointness is not explicit in the source text. The same function body can be called with separated or aliased actual arguments, and the relevant regions may be hidden behind pointer arithmetic, loop-indexed accesses, local pointer aliases, casts, or library calls. Thus, the synthesis problem is not merely to emit a \separated clause, but to determine which caller-visible pointer regions require such a clause and what region extent should be used.

AutoFraming uses frame information extracted from the CPG to synthesize region-separation requirements. The interfering-pair component $\mathcal { T } _ { f }$ identifies caller-visible pointer roots whose aliasing can affect a caller-visible write, while the access-window component $\Omega _ { f }$ identifies the region of each root that the function may access. The separation projection then maps each pair in $\mathcal { T } _ { f }$ and its corresponding access windows to an explicit caller-side \separated precondition.

For each $\{ p , q \} \in { \mathcal { T } } _ { f }$ , AutoFraming emits ${ \mathsf { S e p } } _ { f } ( p , q )$

$$
\begin{array} { r } { \begin{array} { r l } & { \rho _ { p } = \mathsf { r e g i o n } ( p , \Omega _ { f } ( p ) ) , } \\ & { \rho _ { q } = \mathsf { r e g i o n } ( q , \Omega _ { f } ( q ) ) , } \\ & { \mathsf { S e p } _ { f } ( p , q ) = \mathsf { r e q u i r e s \setminus s e p a r a t e d } ( \rho _ { p } , \rho _ { q } ) ; } \end{array} } \end{array}
$$

The function region converts a pointer root and its access window into an entry-expressible ACSL memory region, such as $\mathtt { p , p + ( 0 \_ . . n - 1 ) }$ , or another slice expressible at function entry. A constant offset contributes a singleton extent; a loopindexed offset contributes the loop range; and a guarded access contributes the guard bound when that bound is available at function entry.

Type compatibility alone is insufficient: many C functions take multiple pointers of the same type without requiring them to be disjoint. AutoFraming emits a separation constraint only when aliasing the two roots can affect the function’s caller-visible frame. This distinction is important for C APIs such as copy, parse, and decode routines, where source and destination pointers have compatible types but only some pairs need disjointness.

When an access window cannot be expressed at function entry, AutoFraming emits the strongest entry-expressible approximation available and sends the candidate to the verificationrepair stage. Such a candidate is not accepted merely because the clause was generated; it must still preserve the projected frame and discharge the relevant frame obligations.

## D. Synthesis of Data Dependencies for Write Sets

The second frame synthesis issue is data dependence for write sets: for each location that a function may modify, the frame specification must identify the caller visible inputs that may influence the value written there. An ACSL assigns clause without a \from footprint states where the function may write, but not what information the written value may depend on. Such a frame is incomplete for modular reasoning because callers cannot determine which parts of the prestate are relevant to the modified location. Inferring this footprint is difficult in C because the dependence of a written value is often separated from the final assignment. The value may flow through local temporaries, pointer aliases, loop carried updates, control predicates, globals, or summarized library calls before reaching the write. As a result, reading the left hand side and right hand side of the final assignment is not enough to recover the frame dependence.

AutoFraming handles this by using the write-set locations in $\mathcal { L } _ { f }$ as carriers and attaching to each one the dependence footprint $\Delta _ { f } ( L )$ extracted from the CPG. The projection therefore turns the internal value-flow information for each write into the ACSL form expected by modular verification.

For each normalized write set location $L \in { \mathcal { L } } _ { f }$ , the data dependence projection attaches $\Delta _ { f } ( L )$

$$
{ \frac { L \in { \mathcal { L } } _ { f } \qquad \Delta _ { f } ( L ) = \{ r _ { 1 } , \dots , r _ { m } \} \qquad m > 0 } { \mathsf { a s s i g n s ~ } L \setminus \pounds \operatorname { r c o m } r _ { 1 } , \dots , r _ { m } ; } } { \mathrm { ~ D E P } }
$$

If $\Delta _ { f } ( L ) = \emptyset$ , AutoFraming emits:

$$
\frac { L \in \mathcal { L } _ { f } } { a s s \mathrm { i } \mathrm { g n s ~ } L \mathrm { \backslash } \mathrm { f r o m ~ \backslash n o t h i n g } ; } \mathrm { D E P - C o N s T }
$$

The dependence footprint includes both data and control dependencies. A root is included in $\Delta _ { f } ( L )$ if its value flows into the expression written to $L ,$ or if it controls whether a write to L is executed. For example, in a branch that writes different constants to <sub>\*</sub>out depending on flag, the output depends on flag even though flag does not appear on the right-hand side of either assignment. Capturing this control dependence is necessary for the \from clause to describe the value written, not just the memory location modified.

The result of the two projections is a frame skeleton $S _ { f }$ The region separation projection contributes the \separated preconditions, and the data dependence projection contributes the assigns . . . \from clauses. Thus, AutoFraming’s synthesis target is not merely the static write set, but the dynamicframe information attached to that write set.

## E. Frame-constrained contract synthesis

The two projection rules produce a frame skeleton $S _ { f }$ consisting of \separated preconditions and assigns . . . \from clauses. This skeleton is the output of the frame-constrained projection step. AutoFraming does not ask the LLM to invent or reinterpret it. Instead, $S _ { f }$ is inserted into the prompt as a set of required frame obligations in the candidate contract.

Frama-C verifier, however, checks frame clauses only inside a complete ACSL contract. A frame skeleton can introduce proof obligations that require additional surrounding annotations. For example, a separated region $p + ( 0 . . n - 1 )$ requires preconditions establishing that the region is well-formed and valid at function entry; an assigned slice $A [ l o . . h i - 1 ]$ may require loop annotations showing that the loop writes only that range; and a \from footprint may require auxiliary assertions or invariants that expose how values flow through local temporaries. AutoFraming therefore asks the LLM to synthesize a contract envelope $E _ { f }$ around the locked frame skeleton:

$$
C _ { f } ^ { ( 0 ) } = E _ { f } \cup S _ { f } \qquad { \mathrm { w i t h } } \qquad S _ { f } \subseteq C _ { f } ^ { ( 0 ) } .
$$

The envelope may contain validity preconditions, simple functional postconditions, loop invariants, loop variants, loop assigns clauses, and auxiliary assertions needed by the verifier.

The frame information $\Phi _ { f }$ guides this envelope synthesis. Access windows $\Omega _ { f }$ suggest validity requirements such as \valid\_read for input regions and \valid for written regions. write-set locations in $\mathcal { L } _ { f }$ guide loop assigns annotations and frame-preserving loop invariants. Dependence footprints $\Delta _ { f } ( L )$ identify the inputs that are relevant to postconditions over assigned outputs. When a function calls another function, AutoFraming provides the verified callee contract as context, so the LLM can use the callee’s frame and functional clauses rather than re-analyzing the callee body.

This design separates the role of program analysis from the role of the LLM. Program analysis and frame-constrained projection synthesize the frame-critical content $S _ { f }$ . The LLM supplies only the verifier-facing envelope $E _ { f }$ needed to make $S _ { f }$ checkable in a complete ACSL contract. As a result, AutoFraming avoids relying on the LLM to guess dynamicframe clauses from source code alone, while still using the LLM to fill in auxiliary annotations and functional context that are necessary for verifier to discharge the generated frame.

## F. Frame-constrained refinement

The contract produced by frame-constrained synthesis already contains the projected frame conditions, but it may still fail the frame-error-free checks. These failures can arise because the surrounding contract envelope is incomplete: the candidate may lack validity preconditions needed to justify an accessed region, loop annotations needed to prove an assigned slice, or auxiliary assertions needed to expose a dependence relation to the verifier. AutoFraming therefore performs verifierguided refinement using feedback from Frama-C/WP and Eva.

The refinement loop is frame-constrained. The LLM may revise the contract to make the projected frame conditions checkable, but it may not delete, weaken, or replace them with degenerate frame clauses.

In each round, the LLM reads the function body, the current contract, the projected frame skeleton, and the verifier feedback, and proposes a revised contract. A revised candidate $C _ { f } ^ { ( k + 1 ) }$ is accepted only when the following conditions hold. First, the frame structure must be preserved. Every \separated precondition and every assigns . . . \from clause projected from $\Phi _ { f }$ must remain present. The LLM may strengthen these clauses, but it cannot delete them or replace them with a weaker frame.

Second, the candidate must not contain a degenerate frame. In particular, AutoFraming rejects candidates with missing assigns clauses, missing loop assigns clauses, or vacuous frames that hide caller-visible writes.

Third, the candidate must make strict proof progress: it must close at least one previously unresolved proof obligation without reopening any obligation that was already discharged.

The first two conditions prevent the LLM from satisfying the verifier by weakening the frame. The third keeps the repair loop making monotone progress. The loop ends when Frama-C/WP and EVA discharge the relevant frame conditions or when it reaches the maximum number of rounds N.

## IV. EVALUATION

We evaluate AutoFraming with respect to three questions.

RQ1 (Performance). How effectively does AutoFraming synthesize frame specifications?

RQ2 (Ablation). How do CPG-based frame analysis and frame-constrained projection contribute to the efficiency of AutoFraming?

RQ3 (Generality). Can AutoFraming improve existing frameworks’ capability of synthesizing frame specifications?

## A. Datasets

We use the following existing datasets:

a) $\mathcal { D } _ { \mathrm { a } } \colon R e a l - w o r l d C$ code : Nine real-world C programs from the Preguss public benchmark [5], across an embedded RTOS (Atomthreads), four Contiki OS subsystems (ctimer, etimer, process, trickle), a TLS X.509 certificate parser (x509- submodule-2), an authenticated-encryption mode (AES-CCM), and an embedded database (Antelope memb, Antelope relation). We include functions whose frame specifications are expressible in first-order ACSL without user-defined inductive predicates, yielding 84 benchmark functions. Functions requiring such predicates are excluded because their verification rests on the quality of a hand-written predicate, which is orthogonal to the frame synthesis we study in this paper. The same inclusion criterion is applied to all compared configurations. For transparency we also report the excluded set, where Preguss’s error-free frame rate is only 3.3% of the predicate-dependent functions, confirming that these cases are omitted because they need inductive-predicate reasoning outside our target problem, not because their exclusion advantages AutoFraming.

b) $\mathcal { D } _ { \mathrm { b } } \mathrm { : }$ Frama-C Problems: Fifty-one small C programs from the Frama-C tutorial suite widely used by formal verification works, organised into eight categories including arrays and loops, general wp problems, immutable arrays, loops, miscellaneous, more arrays, mutable arrays, pointers.

## B. Experimental Setup

For each function we measure two frame-quality metrics. Frame warning: a warning emitted by the Frama-C WP plugin indicating that the contract has no assigns clause or that the clause is incomplete. Frame fail: reported by the Frama-C WP plugin together with the Eva plugin; a function is frame-failing if it fails at least one $\mathtt { t y p e d \_ c f > \_ a s s i g n s \_ s }$ or $\mathtt { t y p e d \_ c f > \_ l o o p \_ a s s i g n s \_ t }$ goal during WP discharge, or if it fails the Eva dependence-set check. We classify a function as frame-error-free only when all three checks pass: (1) Frama-C emits no missing/incomplete assigns warning, (2) WP discharges every generated assigns and loop-assigns proof obligation within the timeout, and (3) Eva reports no dependency set mismatch for the write-set locations. Timeouts on frame conditions are counted as frame failures. We do not count a function as error-free merely because the contract is syntactically accepted. To confirm that strengthening a frame does not weaken the functional specification, we also report the proof ratio (goals proved / total goals) from the WP results.

To show that our results are not specific to a single model, we instantiate the pipeline identically with two different models, Gemini-2.5-Pro and GPT-5.2. Verification uses Z3 and Alt-Ergo with a 30-second per-goal timeout.

C. RQ1: How effectively does AutoFraming synthesize frame specifications?

RQ1 evaluates whether AutoFraming can synthesize frame specifications that are accepted by frame specific checks. We instantiate the same pipeline with two LLMs, Gemini-2.5- Pro and GPT-5.2, using the same configurations. We denote

AutoFraming as AF in the result tables. Accordingly, AF-GPT and AF-Gemini refer to AutoFraming instantiated with GPT-5.2 and Gemini-2.5-Pro, respectively. We compare against the strongest available ACSL specification generators on our benchmarks: Preguss on both benchmarks and AutoSpec on the academic benchmark. All methods are evaluated using the same frame-error-free criterion.

Tables II and III report the number of functions for which each method synthesizes a frame-error-free specification. On the real-world benchmark $D _ { a } ,$ AutoFraming substantially improves over Preguss: AF-Gemini succeeds on 78 of 84 functions (92.9%) and AF-GPT succeeds on 77 of 84 functions (91.7%), compared with 28 of 84 functions (33.3%) for Preguss. On the academic benchmark $D _ { b } ,$ AF-Gemini succeeds on 49 of 51 functions (96.1%) and AF-GPT succeeds on 48 of 51 functions (94.1%), compared with 41 of 51 functions (80.4%) for Preguss and 23 of 51 functions (45.1%) for AutoSpec.

These results show that frame synthesis exposes failures that are not captured by overall proof success alone. Preguss and AutoSpec can produce contracts that are useful for functional verification, but their frame clauses are often missing, incomplete, or insufficiently precise for the frame-specific checks used in this study. AutoFraming improves because it targets the frame conditions directly: region separation and data dependence for write-set locations are synthesized before the LLM synthesizes the rest of the contract.

Table IV further decomposes each method’s failures into frame warnings and frame proof failures. The breakdown shows that the main weakness of the baselines is frame incompleteness. On $D _ { a }$ , Preguss emits 51 frame warnings and has 5 frame failures, leaving only 28 functions frame-errorfree. AutoFraming sharply reduces these warnings: AF-GPT has 2 warnings and 5 frame failures, while AF-Gemini has 4 warnings and 2 frame failures. On $D _ { b } ,$ both AutoFraming variants leave only one frame warning, compared with 10 for Preguss and 28 for AutoSpec.

The few functions AutoFraming leaves unverified falls into two categories. The first is a limit of the verifier. For example, memb\_alloc writes m->count[0..m->num-1], an extent whose bound m->num is a mutable struct field; the synthesized frame names exactly this region and is correct, but the verifier cannot decide the completeness of a range whose bound is a runtime value.

The second failure type occurs when the LLM overrides the projected frame skeleton with its own reading of the code during contract synthesis. Although AutoFraming is designed to preserve the projected frame, the final candidate still passes through an LLM generation step. In a small number of cases, this generation changes the structure of the frame in a way that remains syntactically valid but becomes harder for WP to discharge. For example, set\_iv, our analysis gives its write set as a single contiguous region, assigns iv[0..15], and AF-Gemini keeps it: WP discharges the frame at once. AF-GPT, handed the very same input, re-derives the frame to mirror the four write statements in the body—iv[0], the thirteen bytes copied into $\mathrm { i } \mathrm { v } [ 1 \dots 1 3 ]$ by memcpy, iv[14], and iv[15], and emits them as four disjoint slices. This forces WP to re-establish the memcpy’s separation against four separate regions, which causes the timeout. That is a limitation of our approach because our approach still depends on LLM generation.

TABLE II  
ERROR-FREE FRAME COUNTS PER PROGRAM ON THE REAL-WORLD BENCHMARK D .AF-GPT AND AF-GEMINI DENOTE AUTOFRAMING INSTANTIATED WITH GPT-5.2 AND GEMINI-2.5-PRO, RESPECTIVELY.
<table><tr><td>Program</td><td>#Fns</td><td>LOC</td><td>Preguss</td><td>AF-GPT</td><td>AF-Gemini</td></tr><tr><td>AES-CCM</td><td>10</td><td>610</td><td>2</td><td>6</td><td>8</td></tr><tr><td>Antelope memb</td><td>1</td><td>55</td><td>0</td><td>1</td><td>0</td></tr><tr><td>Antelope relation</td><td>1</td><td>42</td><td>0</td><td>1</td><td>1</td></tr><tr><td>Atomthreads</td><td>10</td><td>830</td><td>1</td><td>10</td><td>9</td></tr><tr><td>Contiki ctimer</td><td>10</td><td>315</td><td>0</td><td>10</td><td>10</td></tr><tr><td>Contiki etimer</td><td>10</td><td>195</td><td>8</td><td>10</td><td>10</td></tr><tr><td>Contiki process</td><td>11</td><td>420</td><td>4</td><td>10</td><td>11</td></tr><tr><td>Contiki trickle</td><td>11</td><td>395</td><td>3</td><td>11</td><td>11</td></tr><tr><td>X.509 submod-2</td><td>20</td><td>1480</td><td>10</td><td>18</td><td>18</td></tr><tr><td>TOTAL</td><td>84</td><td>4342</td><td>28</td><td>77</td><td>78</td></tr><tr><td>frame-error-free rate</td><td></td><td></td><td>33.3%</td><td>91.7 %</td><td>92.9%</td></tr></table>

TABLE III

ERROR-FREE FRAME COUNTS PER CATEGORY ON THE ACADEMIC BENCHMARK D . AF-GPT AND AF-GEMINI DENOTE AUTOFRAMING INSTANTIATED WITH GPT-5.2 AND GEMINI-2.5-PRO, RESPECTIVELY.
<table><tr><td>Category</td><td>#Fns</td><td>AutoSpec</td><td>Preguss</td><td>AF-GPT</td><td>AF-Gemini</td></tr><tr><td>arrays_and_loops</td><td>5</td><td>3</td><td>4</td><td>5</td><td>5</td></tr><tr><td>general_wp_problems</td><td>12</td><td>10</td><td>10</td><td>12</td><td>12</td></tr><tr><td>immutable_arrays</td><td>8</td><td>0</td><td>8</td><td>7</td><td>7</td></tr><tr><td>loops</td><td>8</td><td>1</td><td>5</td><td>8</td><td>7</td></tr><tr><td>miscellaneous</td><td>5</td><td>2</td><td>4</td><td>4</td><td>5</td></tr><tr><td>more_arrays</td><td>3</td><td>0</td><td>2</td><td>3</td><td>3</td></tr><tr><td>mutable_arrays</td><td>2</td><td>0</td><td>0</td><td>2</td><td>2</td></tr><tr><td>pointers</td><td>8</td><td>7</td><td>8</td><td>7</td><td>8</td></tr><tr><td>TOTAL</td><td>51</td><td>23</td><td>41</td><td>48</td><td>49</td></tr><tr><td>frame-error-free rate</td><td></td><td>45.1%</td><td>80.4%</td><td>94.1%</td><td>96.1%</td></tr></table>

Answer to RQ1. AutoFraming synthesizes frame-error-free specifications for more than 90% of functions on both benchmarks and substantially outperforms existing ACSL generation frameworks under frame-specific checks. The improvement is consistent across two different LLMs, suggesting that the gain comes from the frame-synthesis design rather than from a single model choice.

## D. RQ2: How do CPG-derived frame information and frameconstrained projection contribute to the efficiency of AutoFraming?

To isolate the source of AutoFraming’s frame quality, we compare the full pipeline with three reduced variants. The first variant removes CPG-derived frame information but keeps frame-constrained projection and frame-constrained refinement. The second keeps the CPG information but removes frame-constrained projection. The third removes both CPG information and frame-constrained projection, leaving only LLM-based refinement.

TABLE IV  
FRAME-OUTCOME BREAKDOWN PER METHOD—FRAME WARNINGS, FRAME FAILURES, AND ERROR-FREE—WITH EACH ROW SUMMING TO #FNS. AF-GPT AND AF-GEMINI DENOTE AUTOFRAMING INSTANTIATED WITH GPT-5.2 AND GEMINI-2.5-PRO, RESPECTIVELY.
<table><tr><td colspan="3"></td><td rowspan="2">Frame Fail</td><td rowspan="2">Frame Error-free</td></tr><tr><td>Benchmark</td><td>Method</td><td>Frame Warning</td></tr><tr><td rowspan="3"> $\mathcal { D } _ { \mathrm { a } }$  (84)</td><td>Preguss</td><td>51</td><td>5</td><td>28</td></tr><tr><td>AF-GPT</td><td>2</td><td>5</td><td>77</td></tr><tr><td>AF-Gemini</td><td>4</td><td>2</td><td>78</td></tr><tr><td rowspan="4"> $\mathcal { D } _ { \mathrm { b } }$  (51)</td><td>AutoSpec</td><td>28</td><td>0</td><td>23</td></tr><tr><td>Preguss</td><td>10</td><td>0</td><td>41</td></tr><tr><td>AF-GPT</td><td>1</td><td>2</td><td>48</td></tr><tr><td>AF-Gemini</td><td>1</td><td>1</td><td>49</td></tr></table>

The full AutoFraming pipeline synthesizes frame-error-free specifications for 78 of 84 functions. Removing the frameconstrained projection while keeping the CPG information and the full frame-constrained refinement loop reduces the result to 54 of 84 functions, a loss of 24 functions. This drop shows that CPG information alone is not sufficient: without the projection step, the LLM must still infer how the extracted information should be translated into ACSL frame clauses.

The other reduced variants support the same conclusion. The configuration without frame-constrained projection reaches 54 functions, which is close to the refinement-only configuration with 53 functions. Removing the CPG-based frame information reduces the result further to 48 functions. Thus, neither CPG information alone nor frame-constrained refinement alone explains AutoFraming’s improvement. The main gain comes from the frame-constrained projection.

The two frame constraints govern where the variants fail. Most failures fall under the data dependence projection for write sets, in two forms. Some are datadependence errors: the model names the right locations but omits the \from dependence (e.g. assigns <sub>\*</sub>dest; instead of assigns <sub>\*</sub>dest \from <sub>\*</sub>src;), or it declares an unsound dependence, as when set\_iv gives \from nonce[0..12] that cannot be discharged through the library memcpy. Others are write-set errors, where the extent of the write set is wrong rather than merely its name: aes\_128\_set\_padded\_key and set\_key declare assigns round\_keys[0..10] but miss the inner array dimension—the sound frame is round\_keys[0..10][0..15]; atomCurrentContext, list\_head, and arg\_alloc omit the \result location entirely; and \_parse\_arc declares assigns \nothing for a function that writes two outputs. The remaining failures fall under region separation. For example, parse\_PolicyInformation and parse\_policyQualifierInfo declare the correct write set but hedge their precondition, weakening the \separated guarantee until the frame condition can no longer be discharged.

Answer to RQ2. The ablation shows that the frame-constrained projection is the central component of AutoFraming. Removing this projection reduces the frame-error-free count from 78/84 to 54/84, a loss of 24 functions. This indicates that CPG information and frame-constrained refinement are not sufficient by themselves; the extracted information must be projected into explicit ACSL frame conditions before the LLM assembles the final contract. The refinement-only configuration reaches a similar result (53/84), and removing CPGbased frame information lowers the result to 48/84. Thus, AutoFraming’s improvement comes primarily from the frameconstrained projection.

TABLE V  
RQ2: ABLATION ON $\mathcal { D } _ { \mathbf { a } }$ ERROR-FREE FRAME COUNTS FOR AUTOFRAMING AND THREE REDUCED VARIANTS.
<table><tr><td>Configuration</td><td>#Clean</td><td>Rate</td></tr><tr><td>LLM Refinement Only</td><td>53/84</td><td>63.1%</td></tr><tr><td>Without CPG derived info</td><td>48/84</td><td>57.1%</td></tr><tr><td>Without frame-constrained projection</td><td>54/84</td><td>64.3%</td></tr><tr><td>AutoFraming</td><td>78/84</td><td>92.9%</td></tr></table>

TABLE VI

RQ3: ERROR-FREE FRAME COUNTS AND GOAL-PROVE RATES WHEN AUTOFRAMING AUGMENTS EXISTING GENERATORS. “+LLMREFINE” DENOTES DIRECT LLM REFINEMENT WITHOUT AUTOFRAMING’S FRAME-CONSTRAINED PROJECTION, AND “+AUTOFRAMING” DENOTES REFINEMENT WITH AUTOFRAMING.
<table><tr><td>Benchmark Upstream</td><td></td><td>#Fns</td><td>Error-free frame</td><td>Goal-prove</td></tr><tr><td rowspan="3"> $\mathcal { D } _ { \mathrm { a } }$ </td><td>Preguss</td><td>84</td><td>28 (33.3%)</td><td>94.0%</td></tr><tr><td>Preguss+LLMRefine</td><td>84</td><td>52 (61.9%)</td><td>93.2%</td></tr><tr><td>Preguss+AutoFraming</td><td>84</td><td>80 (95.2%)</td><td>96.8%</td></tr><tr><td rowspan="6"> $\mathcal { D } _ { \mathrm { b } }$ </td><td>AutoSpec</td><td>51</td><td>23 (45.1%)</td><td>86.4%</td></tr><tr><td>AutoSpec+LLMRefine</td><td>51</td><td>36 (70.6%)</td><td>89.4%</td></tr><tr><td>AutoSpec+AutoFraming</td><td>51</td><td>50 (98.0%)</td><td>97.9%</td></tr><tr><td>Preguss</td><td>51</td><td>41 (80.4%)</td><td>96.0%</td></tr><tr><td>Preguss+LLMRefine</td><td>51</td><td>39 (76.5%)</td><td>93.7%</td></tr><tr><td>Preguss+AutoFraming</td><td>51</td><td>50 (98.0%)</td><td>97.5%</td></tr></table>

## E. RQ3: Can AutoFraming improve existing frameworks as a frame synthesis component?

RQ3 evaluates whether AutoFraming can be used as a modular frame-synthesis component for existing specification generators. Rather than replacing upstream tools such as Preguss and AutoSpec, AutoFraming augments their generated contracts by strengthening the frame conditions. Table VI reports the frame-error-free counts and goal-prove rates when AutoFraming augments Preguss and AutoSpec. Three observations show that AutoFraming improves the frame synthesis of these tools without degenerating their functional specification.

First, the total number of verification goals increases after applying AutoFraming: from 1378 to 1687 when comparing Preguss with Preguss+AutoFraming on $\mathcal { D } _ { \mathrm { a } } ,$ from 663 to 681 when comparing AutoSpec with AutoSpec+AutoFraming on ${ \mathcal { D } } _ { \mathrm { b } } ,$ and from 676 to 786 when comparing Preguss with Preguss+AutoFraming on $\mathcal { D } _ { \mathrm { b } }$ . The added goals come from the \separated preconditions and \from dependencies synthesized by AutoFraming. A weakened contract would tend to remove obligations rather than introduce additional ones.

Second, the goal-prove rate also increases despite the larger number of goals. On $\mathcal { D } _ { \mathrm { a } } ,$ Preguss+AutoFraming raises the goal-prove rate from 94.0% to 96.8%. On $\mathcal { D } _ { \mathrm { b } }$ , AutoSpec+AutoFraming raises the rate from 86.4% to 97.9%, and Preguss+AutoFraming raises it from 96.0% to 97.5%. Thus, AutoFraming discharges more proof obligations in absolute terms while also increasing the proved-to-total ratio.

Third, AutoFraming outperforms a direct LLM-refinement baseline. On $\mathcal { D } _ { \mathrm { b } } .$ , AutoSpec+LLMRefine reaches 36/51 frameerror-free functions (70.6%), while AutoSpec+AutoFraming reaches 50/51 (98.0%), a 27.4 percentage-point improvement. Similarly, Preguss+LLMRefine reaches 39/51 (76.5%), while Preguss+AutoFraming reaches 50/51 (98.0%), a 21.5 percentage-point improvement. This shows that the gain does not come from the LLM only refinement; it comes from AutoFraming’s frame-constrained projection and refinement. Answer to RQ3. AutoFraming generalizes as a framespecification refinement layer for existing ACSL generators. With Preguss and AutoSpec as upstream tools, it raises the frame-error-free rate to 95.2%–98.0% while also increasing the goal-prove rate and the absolute number of proved goals. This indicates that AutoFraming strengthens the frame specification rather than improving the frame score by weakening the surrounding contract. We do not claim that the goalprove rate establishes semantic completeness of the functional clauses; rather, it checks that the added frame clauses remain compatible with the evaluated verifier.

## V. RELATED WORK

## A. Program-Analysis-Based Specification Inference

Early work on specification inference mainly relies on program analysis to derive likely contracts, invariants, or behavioral properties from source code. These approaches can be broadly characterized by whether they reason from observed executions or from static program semantics.

Dynamic inference techniques generate candidate specifications from execution traces. Daikon [10], for example, instruments programs and infers likely invariants from recurring value patterns observed at program points such as function entries and exits. CLIPER [11] similarly mines temporal behavioral patterns from traces. Although such techniques are useful for discovering common runtime properties, their results depend heavily on input coverage and may overfit to observed executions. To improve generalization, later systems combine concrete traces with symbolic reasoning. DySy [12] integrates dynamic execution and symbolic execution, using counterexamples to refine inferred invariants, while SymInfer [13] combines concrete traces with symbolic states to produce more general specifications. SPEEDY [14] further connects runtime monitoring with Frama-C static analyzers to suggest specifications during development.

Static approaches infer specifications without executing the program, often aiming to cover all feasible paths. Kindspec [15], [16] applies symbolic execution to KernelC programs and synthesizes abstract preconditions and postconditions as logical contracts. TriCera [17] translates C programs into constrained Horn clauses and uses SMT solving to infer ACSL function contracts. AutoDeduct [18] also automates contract generation through symbolic execution and Horn clause reasoning. In contrast to these automated techniques, Araujo et al. [19] demonstrate a manual specification and verification process by translating avionics requirements into ACSL annotations and checking them with Frama-C. These works show the value of program analysis for contract inference, but they generally focus on functional behavior rather than frame specifications.

## B. LLM-Assisted Formal Specification Generation

Recent work has explored large language models as a flexible mechanism for generating formal specifications from code, natural language requirements, examples, or verifier feedback. For C programs, George et al. [1] prompt GPT-4 to synthesize ACSL contracts directly from source code, showing that LLMs can often produce specifications that capture intended behavior rather than merely restating implementation details. In followup work, George et al. [2] enrich the prompt with outputs from Frama-C Eva [20] and PathCrawler [21], demonstrating that static-analysis warnings and test-based traces can guide the model toward more useful contracts.

Several systems combine LLM generation with formalverification feedback. AutoSpec [4] constructs program information such as call graphs and iteratively refines LLMgenerated specifications using verifier feedback. Janßen et al. [3] evaluate ChatGPT on small C programs and show that generated loop invariants can help Frama-C and CPAchecker complete proofs, while also identifying common failure modes such as weak or invalid invariants. Guangyuan et al. [22] use GPT-3.5 to propose loop invariants and rely on CBMC counterexamples to guide refinement. ACInv [23] extracts loop context through static analysis, prompts an LLM to generate invariants, and uses an LLM-based evaluator to iteratively strengthen, weaken, or reject candidates. NL2Contract [24] translates informal natural-language hints into formal functional contracts with both preconditions and postconditions for automatic software verification. Preguss [5] improves the scalability of LLM-assisted specification synthesis by using potential runtime-error assertions to decompose large C verification tasks into prioritized units. These studies suggest that LLMs are most effective when combined with symbolic information and verifier feedback, but their primary focus remains on functional contracts and loop invariants.

LLM-based specification generation has also been studied beyond C and ACSL. For Java, Teuber and Beckert [25] use GPT to complete JML annotations and repair invalid contracts with feedback from the KeY verifier, while Greiner et al. [26] train CodeT5-based models to generate JML preconditions and postconditions from a large corpus of examples. SpecGen [27] adopts a two-stage process that first prompts an LLM to draft specifications and then uses mutation and heuristic selection to repair them. For Rust, SAFE [28] generates Verus specifications from function signatures and documentation, followed by iterative self-refinement. ClassInvGen [29] targets C++ class invariants and validates candidates through execution. In the smart-contract domain, PropertyGPT [30] retrieves examples from Certora reports to guide GPT-4 in generating Solidity properties, which are then checked and refined by custom provers. VeCoGen [31] studies the related problem of generating C programs from ACSL specifications, naturallanguage descriptions, and tests.

## VI. THREATS TO VALIDITY

a) Internal validity: Internal threats mainly concern LLM behavior and the implementation of our analysis rules. Since LLMs are trained on large code and specification corpora, benchmark or similar examples may have appeared during pretraining and could inflate performance. To mitigate this concern, our ablation with only LLM-based refinement, without CPG-derived information or frame-constrained projection, performs substantially worse, suggesting that the gains mainly come from the proposed frame analysis and rules. Another threat is that our CPG-based rules are derived from frame-relevant program information and expert knowledge, but may still miss uncommon C patterns such as complex pointer manipulation or unusual data dependencies. Missing rules or implementation errors may lead to incomplete frame specifications or unresolved verification goals.

b) Construct validity: Our main evaluation metric is error-free frame status, which is determined using Frama-C/WP diagnostics and Frama-C’s dependency analysis. This metric provides practical information that the generated frame specifications are accepted by the verification toolchain, but it is not a complete semantic oracle for all possible client contexts. For example, a generated frame specification may verify successfully while imposing preconditions that are stronger than necessary and may exclude some valid calls. Similarly, the goal-prove rate measures verifier discharge rather than specification minimality or user intent. Therefore, our results should be interpreted as verifier-accepted frame correctness under the evaluated toolchain.

c) External validity: Our experiments are conducted on 84 functions from real-world C systems and 51 academic verification examples. Although these benchmarks cover common frame-relevant patterns such as pointers, arrays, loops, and data dependencies, they may not represent all C programs. The results may not generalize to software with complex lowlevel features, such as unrestricted casts, concurrency, etc. In addition, our approach is designed for ACSL and Frama-C/WP; applying it to other specification languages, verification tools, or programming languages may require adapting the frame rules and verification workflow. Finally, LLM behavior can vary across model versions, prompts, and decoding settings, so future work should evaluate additional models and release prompts, generated specifications, tool versions, and verification scripts to support replication.

## VII. CONCLUSION

We have presented AutoFraming for synthesizing ACSL frame specifications for C programs. It extracts frame-related information and projects the information into a deterministic frame skeleton, and uses the LLM only to complete and repair the surrounding contract under verifier feedback. Our experimental results show that grounding LLM-based specification synthesis in explicit program structure and framespecific constraints leads to more complete and reusable frame specifications than existing generators. Future work will extend AutoFraming to richer heap abstractions, recursive data structures, inter-procedural library summaries, and additional specification languages and verifiers.

[1] G. Granberry, W. Ahrendt, and M. Johansson, “Seeking specifications: The case for neuro-symbolic specification synthesis,” 2025. [Online]. Available: https://arxiv.org/abs/2504.21061

[2] ——, “Specify what? Enhancing neural specification synthesis by symbolic methods,” in Integrated Formal Methods, N. Kosmatov and L. Kovacs, Eds. Cham: Springer Nature Switzerland, 2025, pp. 307–´ 325.

[3] C. Janßen, C. Richter, and H. Wehrheim, “Can ChatGPT support software verification?” 2023. [Online]. Available: https://arxiv.org/abs/ 2311.02433

[4] C. Wen, J. Cao, J. Su, Z. Xu, S. Qin, M. He, H. Li, S.-C. Cheung, and C. Tian, “Enchanting program specification synthesis by large language models using static analysis and program verification,” in Computer Aided Verification, A. Gurfinkel and V. Ganesh, Eds. Cham: Springer Nature Switzerland, 2024, pp. 302–328.

[5] Z. Wang, T. Lin, M. Chen, H. Li, M. Yang, X. Yi, S. Qin, Y. Luo, X. Li, B. Gu, L. Lu, and J. Yin, “A tale of 1001 loc: Potential runtime error-guided specification synthesis for verifying large-scale programs,” Proceedings of the ACM on Programming Languages, vol. 10, no. OOPSLA1, 2026.

[6] F. Yamaguchi, N. Golde, D. Arp, and K. Rieck, “Modeling and discovering vulnerabilities with code property graphs,” in 2014 IEEE symposium on security and privacy. IEEE, 2014, pp. 590–604.

[7] P. Baudin, J.-C. Filliatre, C. Marchˆ e, B. Monate, Y. Moy, and´ V. Prevosto, “ACSL: ANSI/ISO C specification language,” 2008– present, frama-C reference manual. [Online]. Available: https://frama-c. com/download/acsl.pdf

[8] P. Cuoq, F. Kirchner, N. Kosmatov, V. Prevosto, J. Signoles, and B. Yakobowski, “Frama-C: A software analysis perspective,” in SEFM, 2012, pp. 233–247.

[9] F. Yamaguchi, N. Golde, D. Arp, and K. Rieck, “Modeling and discovering vulnerabilities with code property graphs,” in IEEE S&P, 2014, pp. 590–604.

[10] M. D. Ernst, J. H. Perkins, P. J. Guo, S. McCamant, C. Pacheco, M. S. Tschantz, and C. Xiao, “The daikon system for dynamic detection of likely invariants,” Sci. Comput. Program., vol. 69, no. 1–3, p. 35–45, Dec. 2007. [Online]. Available: https://doi.org/10.1016/j.scico.2007.01. 015

[11] D. Lo, S.-C. Khoo, and C. Liu, “Efficient mining of iterative patterns for software specification discovery,” in Proceedings of the 13th ACM SIGKDD international conference on Knowledge discovery and data mining, 2007, pp. 460–469.

[12] C. Csallner, N. Tillmann, and Y. Smaragdakis, “Dysy: dynamic symbolic execution for invariant inference,” in Proceedings of the 30th International Conference on Software Engineering, ser. ICSE ’08. New York, NY, USA: Association for Computing Machinery, 2008, p. 281–290. [Online]. Available: https://doi.org/10.1145/1368088.1368127

[13] T. Nguyen, K. Nguyen, and H. Duong, “Syminfer: inferring numerical invariants using symbolic states,” in Proceedings of the ACM/IEEE 44th International Conference on Software Engineering: Companion Proceedings, ser. ICSE ’22. New York, NY, USA: Association for Computing Machinery, 2022, p. 197–201. [Online]. Available: https://doi.org/10.1145/3510454.3516833

[14] D. R. Cok and S. C. Johnson, “Speedy: An eclipse-based ide for invariant inference,” Electronic Proceedings in Theoretical Computer Science, vol. 149, p. 44–57, Apr. 2014. [Online]. Available: http://dx.doi.org/10.4204/EPTCS.149.5

[15] M. Alpuente, D. Pardo, and A. Villanueva, “Abstract contract synthesis and verification in the symbolic k framework,” Fundamenta Informaticae, vol. 177, no. 3-4, pp. 235–273, 2020.

[16] M. Alpuente and A. Villanueva, “Automated synthesis of software contracts with kindspec,” in Analysis, Verification and Transformation for Declarative Programming and Intelligent Systems: Essays Dedicated to Manuel Hermenegildo on the Occasion of His 60th Birthday. Springer, 2023, pp. 51–71.

[17] J. Amilon, “Automated inference of ACSL function contracts using tricera,” 2021.

[18] J. Amilon, D. Gurov, C. Lidstrom, M. Nyberg, G. Ung, and¨ O. Wingbrant, “Autodeduct: A tool for automated deductive verification of c code,” 2025. [Online]. Available: https://arxiv.org/abs/2501.10889

[19] F. Dordowsky, “An experimental study using ACSL and Frama-C to formulate and verify low-level requirements from a DO-178C compliant avionics project,” arXiv preprint arXiv:1508.03894, 2015.

[20] F.-C. D. Team, FramaC Eva Plug-in Manual, CEA LIST, 2025. [Online]. Available: https://frama-c.com/html/publications.html#eva

[21] ——, FramaC Pathcrawler Plug-in Manual, CEA LIST, 2025. [Online]. Available: https://frama-c.com/html/publications.html#pathcrawler

[22] G. Wu, W. Cao, Y. Yao, H. Wei, T. Chen, and X. Ma, “LLM meets bounded model checking: Neuro-symbolic loop invariant inference,” in Proceedings of the 39th IEEE/ACM International Conference on Automated Software Engineering, ser. ASE ’24. New York, NY, USA: Association for Computing Machinery, 2024, p. 406–417. [Online]. Available: https://doi.org/10.1145/3691620.3695014

[23] R. Liu, G. Li, M. Chen, L.-I. Wu, and J. Ke, “Enhancing automated loop invariant generation for complex programs with large language models,” arXiv preprint arXiv:2412.10483, 2024.

[24] C. Richter and H. Wehrheim, “Beyond postconditions: Can large language models infer formal contracts for automatic software verification?” arXiv preprint arXiv:2510.12702, 2025.

[25] S. Teuber and B. Beckert, “Next steps in LLM-Supported Java verification,” in 2025 IEEE/ACM 1st International Workshop on Neuro-Symbolic Software Engineering (NSE). IEEE, May 2025, p. 1–4. [Online]. Available: http://dx.doi.org/10.1109/NSE66660.2025.00007

[26] S. Greiner, N. Buhlmann, M. Ohrndorf, C. Tsigkanos, O. Nierstrasz,¨ and T. Kehrer, “Automated generation of code contracts: Generative ai to the rescue?” in Proceedings of the 23rd ACM SIGPLAN International Conference on Generative Programming: Concepts and Experiences, 2024, pp. 1–14.

[27] L. Ma, S. Liu, Y. Li, X. Xie, and L. Bu, “ SpecGen: Automated Generation of Formal Program Specifications via Large Language Models ,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). Los Alamitos, CA, USA: IEEE Computer Society, May 2025, pp. 16–28. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ICSE55347.2025.00129

[28] T. Chen, S. Lu, S. Lu, Y. Gong, C. Yang, X. Li, M. R. H. Misu, H. Yu, N. Duan, P. Cheng et al., “Automated proof generation for rust code via self-evolution,” arXiv preprint arXiv:2410.15756, 2024.

[29] C. Sun, V. Agashe, S. Chakraborty, J. Taneja, C. Barrett, D. Dill, X. Qiu, and S. K. Lahiri, “Classinvgen: Class invariant synthesis using large language models,” 2025. [Online]. Available: https: //arxiv.org/abs/2502.18917

[30] Y. Liu, Y. Xue, D. Wu, Y. Sun, Y. Li, M. Shi, and Y. Liu, “Propertygpt: Llm-driven formal verification of smart contracts through retrieval-augmented property generation,” in Proceedings 2025 Network and Distributed System Security Symposium, ser. NDSS 2025. Internet Society, 2025. [Online]. Available: http: //dx.doi.org/10.14722/ndss.2025.241357

[31] M. Sevenhuijsen, K. Etemadi, and M. Nyberg, “Vecogen: Automating generation of formally verified c code with large language models,” in 2025 IEEE/ACM 13th International Conference on Formal Methods in Software Engineering (FormaliSE). IEEE, 2025, pp. 101–112.