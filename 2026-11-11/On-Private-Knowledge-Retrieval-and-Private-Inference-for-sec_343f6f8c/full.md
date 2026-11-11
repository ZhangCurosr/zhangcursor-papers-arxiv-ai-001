# On Private Knowledge Retrieval and Private Inference for secure multi-agent systems<sup>⋆</sup>

<sub>Vicenç</sub> <sub>Torra</sub>1[0000-0002-0368-8037] <sub>and</sub> <sub>Maria</sub>

<sub>Bras-Amorós</sub>2[0000-0002-3481-004X]

<sup>1</sup> Department of Computing Science, Umeå University, Umeå, Sweden. 2 Departament of Mathematics, Universitat Politècnica de Catalunya (UPC), Barcelona, Catalonia, Spain.

E-mail addresses: vtorra@cs.umu.se, maria.bras@upc.edu

Abstract. Multi-agent systems provide a convenient framework for the massive deployment of agents based on Large Language Models (LLMs). It provides a sound framework where agents can interact with existing software to perform actions in an autonomous way. Agentic AI is the new term to denote these AI-based autonomous systems.

In real multi-agent systems, agents need to deal with malicious agents some of which can exploit security vulnerabilities to do harm. A memory poisoning attack is an example of exploiting a vulnerability to modify agents knowledge and, thus, afect its behavior. In an agent, there are diferent types of memory systems (semantic, episodic, and short-term) which have their own characteristics and for which diferent memory poisoning attacks may be developed.

In this paper we focus on semantic memory, we discuss its attack, and mitigation strategies. We introduce private knowledge retrieval and private inference. We propose a signature system in the knowledge base to prevent from undesired manipulation. This should avoid memory poisoning attacks by introducing security by design in the construction of the semantic memory. In our architecture, an agent contains a local Prologlike inference system (i.e., using Horn clauses to represent knowledge) which accesses an external knowledge base to get the required knowledge. This externally signed knowledge base is accessed privately. We provide diferent implementations in Python using diferent privacy levels including k-anonymity and information-theoretical privacy (IT-PIR).

Keywords: Agentic AI · Memory poisoning attacks · Short-term memory

## 1 Introduction

Agentic AI is becoming the new methodology to develop distributed AI-based systems taking advantage of Large Language Models (LLMs). Agentic AI relates with the well-established field of Multi-Agent Systems (MAS), where autonomous agents collaborate in solving problems together. That is, instead of an agent solving a problem alone accessing other types of software tools, a myriad of agents collaborate and coordinate to solve problems. See discussions in [12] and [5] on the connections between both fields. Multi-agent systems and agentic AI allow us to build sophisticated software architectures. An example is the AI-orchestrated cyber espionage campaign described by Anthropic [2] based on Claude.

When implementing and deploying multi-agent systems, we need to take into account that there may be malicious agents that can exploit both security vulnerabilities and design principles so that they can afect other agents behaviors to do harm. It is also important to note that interactions between agents are not one-shot. During the life-cycle of an agent, malicious agents can have long interactions that can be exploited for long-term attacks.

Memory poisoning attacks are an important type of attack. They consist on a non-authorized modification of agents memory to modify its behavior. Memory is a key component in any agent, as it stores interactions, experiences, and knowledge. Then, naturally, memory afects how agent’s decisions are made and actions implemented. Memory poisoning, and its relation with LLMs and LLMsbased agents, has a relationship with prompt hacking [11] (on both prompt injection and jailbreaking) with the goal to cause unintended agent’s behavior. Nevertheless, in prompt hacking the goal is to build prompts that cause a certain (malfunctioning or unintended) behavior of the LLM-based agent. There is no intended access to memory.

It is important to underline that memory is not a monolithic structure. Agents have diferent memory systems, each of them with their own characteristics, and memory poisoning attacks can be designed and implemented for each of them. The three main types of memory systems are short-term, episodic, and semantic memory. Their diference is in their duration and consolidation. I.e., from recent interactions to long term and consolidated knowledge.

Contributions. In this paper we focus on semantic memory and discuss and propose mitigation strategies for semantic memory poisoning attacks.

We first argue that the formalization of memory poisoning as it is proposed in the recent paper by Chen et al. [7] is not always meaningful. They formalize poisoning in a way similar to data and model poisoning, which has been studied for several years in e.g. the federated learning community. Such approach, based on defining an optimization problem, looks for optimal modifications to produce a certain agent’s behavior. We challenge their assumptions and discuss that there are security solutions that help us to mitigate these attacks.

Second, we introduce private knowledge retrieval and private inference for secure multi-agent systems as a way to mitigate semantic memory poisoning attacks. Our threat model consists of malicious agents that want to poison the memory of another agent. We want to protect semantic memory (both knowledge and inferences) by means of preventing malicious agents to know what is stored in memory, and make tampering dificult. Our approach is to keep the knowledge base external but having logical inferences local. We develop a proof of concept, based on private information retrieval, which permits us to store knowledge in terms of Horn clauses, as in Prolog, and provide logical inference. Agents to implement local logical inferences can request external knowledge, which is signed to avoid tampering, and supplied privately to the agent.

Our implementation demonstrates the technical solutions that can mitigate memory poisoning. We have also implemented both strong and light solutions. E.g., we have implemented both IT-PIR and k-anonymity. Digitally signed knowledge makes tampering more dificult; private knowledge retrieval reduces the risk of unauthorized access to agent’s knowledge and inference processes. We evaluate our approach experimentally and discuss computational costs.

Paper structure. The structure of the paper is as follows. In Section 2, we describe memory systems and memory poisoning attacks in multi-agent systems. Then, in Section 3, we introduce our private inference system, and, in Section 4, we describe our experiments and present our analysis. The paper concludes with a discussion and a description of some research directions.

## 2 Memory systems and their poisoning attacks in MAS

Agentic AI needs a memory to store available knowledge, partial results and interactions with other agents. Memory, however, is not a single and compact structure but it is often a composite element including diferent memory systems. This discussion on memory is agnostic on the type of agents we are considering. It is valid for both LLMs-based agents as well as for more classical architectures [35] as deliberative agents based on BDI-architectures (Belief-Desire-Intention architectures) that may include knowledge in terms of logical expressions.

## 2.1 Memory systems

Existing literature [8,20,21,23,36] classifies diferent memory systems in diferent ways. In this work, following recent trends, we distinguish three types of memory systems based on its duration and consolidation. We classify them from short-term to long-term memory. Next we describe these three types of memory systems from our specific focus on symbolic semantic memory.

Short-term memory. This type of memory stores the current conversation and keeps track of existing interactions with other agents. Partial results as well as partial inferences from existing knowledge can also be included in short-term memory.

– Episodic memory. This type of memory stores past interactions and other not-so-recent information. It is often a consolidation of short-term memory. That is, summaries of previous short-term memories.

– Semantic memory. This type of memory system corresponds to domainspecific knowledge or factual knowledge. It is typically a relatively permanent structure with very low frequency updates. A typical example corresponds to the capitals of countries $( \mathrm { e . g . }$ , Paris is the capital of France), and established medical knowledge about certain illnesses. There are diferent ways to implement semantic knowledge. A common way is to use a symbolic approach [3,16]. In this case, knowledge is typically represented by means of logical expressions, ontologies, or Bayesian models (graphical models). In this case, a knowledge base that contains domain-specific knowledge is combined with an inference mechanism that allows the agent to infer about certain facts (its desired queries) from already known information (facts in the knowledge base or facts in short-term memory) and previously established knowledge. Resolution and Prolog-like inference is the standard mechanism for knowledge expressed by logical expressions. An alternative is to use Retrieval-Augmented Generation (RAG) which permits us to incorporate specific knowledge into LLMs-based agents. Some references use the term declarative memory [20,21] which can be seen as a synonym of semantic memory. We may also consider procedural memory as a type of semantic network. It corresponds to implicit knowledge to perform an action. It is implicit in the sense that it is not possible to reason about how the action is done.

## 2.2 Semantic memory poisoning attacks

A memory poisoning attack consists of an undesired modification of the agent’s memory by another agent. The goal of memory poisoning is to modify on purpose an agent’s behavior. Literature on memory poisoning is about studying possible attacks as well as developing mitigation strategies. These attacks and strategies typically follow the machine learning perspective, and are similar to existing attacks and mitigation strategies for data and model poisoning in federated learning. In this section, we describe the typical attack and object later about its relevance. As we discuss in more detail in the rest of the section, agents secure by design can overcome some of these problems.

For semantic memory, the literature considers attacks following the seminal paper by Chen et al. in NeurIPS [7]. In this attack, an agent computes the optimal modification of the semantic memory in order to afect agent’s behavior. According to the authors [7] the “assumption aligns with practical scenarios where the memory unit of a victim agent is hosted by a third-party retrieval service or directly leverages an unverified knowledge base”.

The attack model considers a poisoned database $\mathcal { D } _ { p o i s o n }$ with a trigger $x _ { t }$ defined in terms of the clean database $\mathcal { D } _ { c l e a n }$ to which an attacker injects adversarial key-value pairs. These injected key-value pairs are denoted by $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { t } )$ where $A ( x _ { t } ) = \{ ( k _ { 1 } ( x _ { t } ) , v _ { 1 } ) , \ldots , ( k _ { | A ( x _ { t } ) | } ( x _ { t } ) , v _ { | A ( x _ { t } ) | } ) \}$ . Then,

$$
\mathcal { D } _ { p o i s o n } ( x _ { t } ) = \mathcal { D } _ { c l e a n } \cup \mathcal { A } ( x _ { t } ) .\tag{1}
$$

Let us denote by $\mathcal { E } ( q , \mathcal { D } )$ the inference of the semantic memory given query q and knowledge or database D. Then, the output of a query q from the poisoned database is denoted by $\mathcal { E } ( q , \mathcal { D } _ { p o i s o n } ( x _ { t } ) )$ , and, similarly, adding the trigger $x _ { t }$ to a query q the output will be $\mathcal { E } ( q \oplus x _ { t } , { \cal D } _ { p o i s o n } ( x _ { t } ) )$ . Attacks are to cause a modification of the output under the presence of $x _ { t }$ so that instead of producing a benign output $a _ { b }$ we have a malign output $a _ { m }$ leading to a malign action. Examples of malign actions are removing the whole home directory or buying a wrong flight ticket.

Under these assumptions, an attack is formalized as a multi-objective optimization problem consisting of the following two objectives:

– Maximization of adversarial output. If $a _ { m }$ is the target malicious action, the attacker wants to maximize this action for triggers. That is, given a sample distribution of input queries $\pi ,$ the retrieval of q (where q is drawn from a query distribution $q \sim \pi )$ poisoned with $x _ { t } \ ( { \mathrm { i . e . , ~ } } q \oplus x _ { t } )$ produces $a _ { m }$ with high probability. Formally,

$$
\mathrm { M a x i m i z e ~ } \mathbb { E } _ { q \sim \pi } [ \mathbb { 1 } \big ( L L M ( q \oplus x _ { t } , \mathcal { E } ( q \oplus x _ { t } , \mathcal { D } _ { p o i s o n } ( x _ { t } ) ) \big ) = a _ { m } \big ) ] ,\tag{2}
$$

where $\mathbb { 1 } ( \cdot )$ denotes the indicator function. That is, a function that is one when its argument is true, and zero otherwise.

Minimization of the efect on non-targeted queries. That is, the output should not be afected and the action produced by the LLM-based agent should be the benign $a _ { b }$ . Formally, with high probability it produces $a _ { b }$ and not $a _ { m }$

$$
\operatorname { M a x i m i z e } \mathbb { E } _ { q \sim \pi } [ \mathbb { 1 } ( L L M ( q , \mathcal { E } ( q , \mathcal { D } _ { p o i s o n } ( x _ { t } ) ) ) = a _ { b } ) ] .\tag{3}
$$

This model assumes that the clean database is known (see Equation 1), updates are possible $( \mathrm { i . e . }$ , that we can add these $\boldsymbol { \mathcal { A } } ( \boldsymbol { x } _ { t } )$ key-value pairs), and we know local-agents’ LLM-s outputs (see E in Equations 2 and 3). Mitigation strategies need to address these key aspects. Our mitigation strategies directly challenge these assumptions.

## 2.3 Mitigation strategies for semantic memory poisoning attacks

As we have explained above, semantic memory is about established knowledge that is mainly static, and updates are only needed from time-to-time. We propose the following three strategies to avoid semantic memory poisoning: (i) secure memory mechanisms to avoid internal memory poisoning attacks, (ii) secure memory updating algorithms for knowledge bases and (iii) secure communications with external agents as well as communication logs. Each of these requirements is discussed below.

The semantic memory attack model assumes that the memory is external to the agent and hosted by a third party. We discuss this fact, and we consider below both the case of internal and external memory.

(i) Secure memory mechanisms. Internal secure memory can be implemented using hashing and signatures. Hashing allows to detect any undesired changes in memory, and signatures allow to ensure provenance of the memory components. Moreover, cryptographic protocols can help us to make local memory private, and thus, make $\mathcal { D } _ { c l e a n }$ unknown to attackers, which prevents the attacks associated to Equations 1, 2 and 3.

Same tools (hashing and signatures) apply when the semantic memory “is hosted by a third-party retrieval service”. This would limit malicious manipulation of the memory. In short, ensuring safe memory in trusted third parties is similar to the case of avoiding corrupted databases in the cloud. This approach permits intruders to study the optimization problem above as $\mathcal { D } _ { c l e a n }$ may be accessed, but intruders will not be able to manipulate the knowledge base. We have implemented this approach in our system. In our implementation, each knowledge item (including its provenance information, see below) is signed to avoid its manipulation.

In the case of external untrusted knowledge based servers providing inference, there are solutions for deep learning models. They are based [25] on Fully Homomorphic Encryption (FHE). These solutions are costly and only available in practice for small neural networks. For other types of data-driven models there are some FHE solutions. See e.g. the case of decision trees. Up to our knowledge, there are no private-inference models in the literature for knowledge base inference as the one discussed here below. They would be costly in any case, and more costly than decision trees.

Our alternative, discussed later in detail, is to keep the static part of the semantic memory mechanism external and signed, while maintaining the inference system protected and local to the agent. Formally, with respect to Equations 2 and $^ { 3 , }$ this requires $\mathcal { E } ( \cdot )$ to be local and protected, which prevents malicious agents from computing $\mathcal { E } ( q \oplus x _ { t } , \mathcal { D } _ { p o i s o n } ( x _ { t } ) )$ . Note that although our inference engine is based on Prolog, the knowledge and knowledge-base servers are agnostic to the inference engine $( { \mathrm { i . e . } }$ , other forms of logical inference could also be implemented). Moreover, digital signatures prevent unauthorized updates to $\mathcal { D } _ { c l e a n }$ , thus preventing Equation 1 from applying. In other words, a malicious server cannot modify the knowledge base without being detected, and local inference is therefore performed over authenticated knowledge. This way inference is expected to be reasonably correct. We introduce private knowledge retrieval and private inference according to this design in Section 3.

(ii) Secure updating of knowledge bases. Knowledge bases need to be updated from time to time. Provenance structures can ensure integrity for data but also for the provenance structures themselves so that no agent can forge provenance data. Note that data provenance can ensure completeness (i.e., that all actions that are relevant to computation are detected and represented). Provenance structures [6,33] can provide both coarse-grained and fine-grained information about updates and modifications of knowledge.

We have included basic provenance information in our knowledge base. Each knowledge item (i.e., Prolog-like Horn clauses, rules, and facts, as explained in more detail below) includes metadata specifying its source. Moreover, each knowledge item, together with its provenance information, is digitally signed by the provenance agent using its private key and a SHA-256-based signature scheme. This allows agents to verify the integrity of the received information before inference using the agent’s public key.

(iii) Secure communication and communication logs. Communication with trusted agents should be secured so that information and knowledge are transmitted safely. Public-key encryption and digital signatures will ensure safe transmission. However, our current system implements neither secure communication mechanisms nor communication logs. This limitation is less relevant in our prototype as we do not support agents running in parallel, and everything is executed on a single machine.

## 3 Private inference in tipp++

In this section we describe our architecture for private knowledge retrieval and private inference as well as its components. In short, the architecture consists of a set of agents deployed to do a task (or a set of tasks). It also includes agents providing knowledge as a service. That is, there are knowledge base servers that provide knowledge on demand. Agents include local knowledge base inference mechanisms that may require access to this external knowledge.

We begin describing the Prolog-like mechanism which is implemented locally within the agent, then the knowledge base client which is internal to the agent and that can access the external knowledge base when local knowledge does not sufice, and finally, we describe the implementation of the external database. Our software is published online [37].

## 3.1 Prolog-like inference mechanism

We have implemented a Prolog-like inference mechanism in Python where knowledge is represented in terms of Horn clauses (see Figure 1). Thus, we can represent both facts and rules (implications). Recursion is allowed in rules. A difference with Prolog is that disjunction is not implemented in our rules and only pure Horn clauses can be used. Recall that from a logical perspective, this is not a constraint on the logical expressiveness of the Horn clauses because a clause with disjunction (e.g., p:- q ∨ r) is logically equivalent to a set of Horn clauses with only conjunction (e.g., p:- q and p:- r).

Predicates follow the syntaxis in Prolog, with the name of the functor and number of arguments. We use the same convention used in Prolog, strings starting with upper-case letters represent variables and those starting with lower-case letters represent constants. The number of arguments of a predicate corresponds to its arity. We do not implement the wildcard variable “\_” commonly used in Prolog. Note that not implementing “\_” does not limit the expressiveness of our knowledge representation formalism. For example, in Prolog the fact member(X, [X|\_]) can be implemented as well with member(X, [X|YY]).

![](images/5e613839eb732c469feab887fd575214613190fb00ee3ee91d5cc8f76b5a646b.jpg)  
Fig. 1. Key-value pairs for the knowledge-base considering only Prolog-like inference systems and queries. Key values correspond to predicate names and arity.

In addition to constants (and variables) we have also implemented lists. They are defined in terms of their constructors nil and cons. Recall that nil defines the empty list, and cons adds an element to an already existing list (i.e., nil corresponds to generating [], cons(a,nil) corresponds to [a], and cons(a,[b,c,d]) corresponds to [a,b,c,d]). We have implemented a preprocessing method so that the knowledge base can be defined using lists in the standard way (i.e., using expressions as [a,b,c]) and then internally translated into lists using the standard constructors nil and cons.

The inference mechanism is implemented following standard Prolog resolution. A query, or in logical terms a request to prove or refute a logical expression, can be a single predicate possibly with uninstantiated variables or a sequence of predicates also possibly with uninstantiated variables. In other words, a sequence of one or more subgoals. Then, the engine selects the first goal and either satisfies it with a fact (via appropriate unification of variables), or expands this goal using a rule/implication. In the latter case the subgoal is replaced by a list of other subgoal(s) and thus, possibly, enlarging the sequence of goals. The process is repeated until all goals are satisfied. When the use of a certain rule leads to a dead end (the sequence of goals cannot be fulfilled) the process backtracks and another rule is selected. If there is no way to fulfill the original goals because there are no further rules to apply, the inference fails. Using the standard closed-world assumption (CWA) in Prolog (i.e., standard Prolog semantics), this means that the query is false.

Our implementation difers with respect to standard Prolog because by default we provide all possible instantiations of a predicate up to a given recursion depth. For example, the output of a proof of mortal(X) would be a list of elements X that are mortal. In our implementation this includes socrates, aristotle, and hypathia. We use this approach for two reasons, (i) we do not implement the cut operator (i.e., the ! in Prolog), and (ii) we do not require facts and rules to be defined in a particular order. In Prolog, diferent orderings can produce solutions in diferent order, and can even cause the program to loop infinitely.

## 3.2 Knowledge base client

Our knowledge base inference system is run locally by the agent. That is, agents have their own knowledge base client which can contain local knowledge as well as provide access to the knowledge base server to obtain external knowledge.

When the local inference system needs to resolve a query, it queries the local knowledge base using the pair (predicate name, arity). The knowledge base client will provide the corresponding Horn clauses to the inference system. Figure 1 illustrates key-value pairs indexed with the pairs (predicate name, arity) and returning Horn clauses. If the knowledge-base client does not have any available Horn clause for this pair, or if the relevant knowledge has expired (each knowledge item is assigned a time-to-live, or TTL), the query is forwarded to the knowledge-base server. Results of external queries are then processed and appropriately sent to the inference system.

In addition, the knowledge retrieved will be cached by the knowledge base client, so that there is no need to query external servers again for the same knowledge. Our main reason for applying caching is that solving a predicate usually requires multiple access to the same facts, and because rules are recursive. So, external access will increase computation time. Moreover, note that repeated access to the same Horn clauses can give hints to intruders about which clauses are being queried and can compromise privacy as well as give hints for knowledge poisoning.

For example, when the agent queries append([a], L, [a,b]) to the local inference system, it will try to retrieve all Horn clauses with signature “append/3” from the knowledge base client. If no clause (no facts or rules) are found in the local knowledge base, the query is forwarded to the external knowledge base server(s). For this example, our knowledge base will return the following:

```javascript
("append",3): [
{"head": ["append","nil","X","X"], "body": [],
"meta": {"predicates":"", "source":"db3"}},
{"head": ["append",("cons","X","Xs"), "Ys", ("cons","X","Zs")],
"body": [["append","Xs","Ys","Zs"]],
"meta": {"predicates":"other", "source":"db3"}}],
```

Additional possible improvements. We implement retrieval solely based on predicate name and arity. It is possible to query taking into account the arguments as well. This will reduce the amount of clauses transferred from knowledgebase servers to knowledge-base clients. This approach is used by Just In Time Indexing (JITI) utilities in SWI-Prolog [38]. Another optimization is tabling (a memoization technique). Tabling consists of storing the results of predicate calls. In other words, it is about caching and reusing answers. None of these optimization techniques have been implemented in our system.

## 3.3 Knowledge base server

In our threat model, knowledge-base servers are not trusted, agents seek to protect their requests, and knowledge items must be protected. Protection of agent’s requests is achieved through private knowledge retrieval, which is based on private information retrieval (PIR) and described in the next section. Knowledge items are protected together with their provenance information. Specifically, each knowledge item, together with its provenance information, is digitally signed by the provenance agent using its private key and a SHA-256-based signature scheme, as described in Section 2.3 (ii). In our implementation, all knowledge base servers are initialized at system startup, and they are provisioned with all the required signed knowledge.

## 3.4 On the implementation of private knowledge retrieval

As we have just mentioned, private knowledge retrieval is based on private information retrieval. Chor et al. [9,10] proved that information-theoretical privacy can be achieved with the assumption that there are copies of the same database in at least two diferent queryable storage servers which do not communicate with each other. When this is the case, each server gets no information on the item retrieved by the user. In contrast, for a single server information-theoretic privacy can only be provided when the whole database is retrieved.

There are relaxations of this approach. For example, Kushilevitz and Ostrovsky [22] relaxed information-theoretical privacy to computational privacy and presented a method for constructing single-database PIR based on the Goldwasser-Micali public-key encryption scheme [15]. Many other single-database PIR schemes have appeared since then. See the survey [24], and more recently [18,4,34] and the survey [1].

In our implementation, we follow the approach of Chor et al. [9,10] to implement private knowledge retrieval using two copies of the database (see also [31]). For this, we need to index knowledge items. We have already specified above that the inference engine and the knowledge base clients request Horn clauses based on the pairs (predicate-name, arity). We use these pairs to index knowledge, and, therefore, we compute an index or hash from them. As the number of Horn clauses (facts and rules) and their lengths associated with a given index will be typically diferent, the database is padded so that all indices in the knowledge base have the same length.

To access a given index i, the agent client builds two random queries difering in the value associated to i. The agent submits one query to each knowledge base server, each server provides a string of bits that are xor-ed bits of the received query. These two responses are also xor-ed, and the resulting string is decoded providing the desired result, i.e., facts and rules associated to index i.

The agent’s knowledge-base client receives the requested knowledge, verifies its integrity using the associated provenance information and digital signature, caches it provided the verification is successful, and forwards it to the inference engine for inference.

In addition to the approach that directly follows PIR, we have implemented two other alternative solutions that are more light-weighted. They are as follows.

– Single server, k-anonymity [27]. Given k, the agent submits the desired index i and $k - 1$ other random indices. The server provides all Horn clauses associated to the k indices, and the client selects the desired one.

– Single server, buckets. All knowledge (Horn clauses) in the knowledge base is divided into b buckets at the initialization stage. Then, for a desired index i, the agent determines in which bucket this knowledge is, and retrieves all associated knowledge (all Horn clauses) from the server. The agent then selects the desired one.

In what follows, we will use the following notation. We denote by $n _ { p a }$ the number of diferent pairs (predicate-name, arity) in the knowledge-base, by $n _ { h }$ the number of Horn clauses in the knowledge base, and by r the number of bits in the padded database.

In summary, we have implemented three alternative private knowledge retrieval protocols. Each implementation has diferent formal privacy and security guarantees, as well as diferent computational requirements. The stronger privacy guarantees with the, in principle, largest computational time corresponds to the implementation of non-colluding knowledge bases. In contrast, k-anonymity and buckets are less computationally demanding but at the same time providing less privacy and security guarantees. Among k-anonymity and buckets, its privacy guarantees and computation requirements depend on the parameter k and the number of buckets. So, if $b = n _ { p a } / k$ then the privacy guarantees of the two approaches are similar, as a query will return a similar number of items. Note however that multiple access to the same index is deterministic for buckets but not for k-anonymity.

## 3.5 Untrusted agents and agent collusions

In our implementation we have considered two servers to provide privacy preserving knowledge retrieval. In addition, knowledge is digitally signed to prevent tampering. Moreover, multiple servers provide interesting mechanisms for guaranteeing robustness and reliability for information storage [29,30]. In this paradigm, encoding of the information stored by the servers (mainly by the use of maximum distance separable codes) prevents from the loss of information in case some servers get damaged or incommunicated and ensures reliability of the information, in case that a (small) number of servers send perturbed or directly fake information. This would provide additional security guarantees against memory poisoning attacks by some of the servers. However, when a number of servers is used, some of them may collude. Some solutions against this kind of collusions using error correcting codes can be found in [14,17]. In these references information retrieval is generalized to file downloading. Another alternative is peer-to-peer PIR [13] in which a user is cloaked in a peer-to-peer user community, where peers submit queries on behalf of other peers and conversely.

Access to knowledge bases and ontologies can also be implemented using these approaches. In this way, untrusted collusions of servers will not know which knowledge items are accessed, and malicious actions could be mitigated. That is, we could minimize compromising memory and future agent’s actions. Locally repairable codes (LRC) [19,28], that are used for large databases in practice, can be used for this purpose.

<table><tr><td></td><td>knowledge bases</td><td>sent</td><td>received communication communication computation</td><td>selection</td></tr><tr><td>k-anonymity</td><td>1KB</td><td>k · 64</td><td> $\overline { { k \cdot r } }$ </td><td>select 1 of k</td></tr><tr><td>b-buckets</td><td>1 KB</td><td>1 · 64</td><td> $m \cdot r$ </td><td>select 1 of b</td></tr><tr><td>2-PIR</td><td>2 KB</td><td> $2 \cdot n _ { p a }$ </td><td> $d r$ </td><td>XOR(2r)</td></tr><tr><td>2d-PIR</td><td>d KB</td><td> $d \cdot n _ { p a } ^ { 1 / d }$ </td><td> $d r$ </td><td>XOR(dr)</td></tr></table>

Table 1. Costs on bits for private knowledge retrieval implemented using k-anonymity, b-buckets, two non-colluding databases (2-PIR), and $2 ^ { d }$ knowledge databases. 64 above represents the length in bits of integers (assuming long long int). Here, r is the number of bits associated to each index in the knowledge base, k the parameter associated to k-anonymity, b is the number of buckets and $m = n _ { p a } / b$ is the number of rows retrieved by b-buckets.

## 4 Experiments and analysis

As described in the previous section, we have implemented a prototype of our architecture in Python. This prototype includes the local Prolog-like inference mechanism. Although all components run on a single laptop, we simulate both the knowledge-base server(s) and the knowledge-base client by providing separate software implementations for each role.

We are assuming that the server is not trusted, nevertheless, we do not allow severs to collude. The external knowledge base is indexed taking into account the predicate name and its arity. Therefore, the size of the database in number of records depends on how many distinctive pairs (name, arity) exist. Then, we have implemented a set of queries and evaluated their execution time experimentally. In addition, we assess the complexity of querying the system. We begin with the analysis of the complexity of the private information retrieval approaches introduced in Section 3.4.

## 4.1 Complexity analysis

In the case of two non-colluding knowledge bases, each one with $n _ { p a }$ keys, the agent is required to send $n _ { p a }$ bits to each server (so, communication requires $2 n _ { p a }$ bits) and this agent will receive r bits from each server (so, communication requires 2r bits). Here r corresponds to the number of bits of a knowledge item of the padded database. Then, the agent performs a xor of these $2 r$ bits obtaining a string of r bits. Our solution is not optimal. There are alternative solutions for $2 ^ { d }$ databases for a given d in which the protocol sends $d \cdot n _ { p a } ^ { 1 / d }$ bits in total and the agent receives from each database r bits $( \mathrm { i . e . , }$ dr bits are received). Here the agent applies a xor of these d · r bits obtaining r bits.

<table><tr><td>KB</td><td>Mean</td><td>St. dev.</td><td>Min</td><td>Max</td></tr><tr><td>baseline</td><td></td><td>0.177878430.002762650.175162150.18181177</td><td></td><td></td></tr><tr><td>bucket</td><td></td><td>0.16892560 0.00250258 0.16619691 0.17300868</td><td></td><td></td></tr><tr><td>Anon k4</td><td></td><td>0.179443650.003014580.175855210.18262578</td><td></td><td></td></tr><tr><td></td><td></td><td>Anon k60.17852928 0.00138278 0.176388550.17977013</td><td></td><td></td></tr><tr><td></td><td></td><td>Anon k8 0.18061575 0.00335783 0.176539080.18402709</td><td></td><td></td></tr><tr><td></td><td></td><td>Anon k10 0.17887916 0.00275545 0.17525422 0.18126626</td><td></td><td>0.223094460.005005670.217117050.22866296</td></tr></table>

Table 2. Execution time statistics for diferent knowledge base access mechanisms, five runs, each accessing the knowledge bases as first time (i.e., case Non-cached).

In the case of k-anonymity, the intruder sends k indices, say integers of 64 bits (so, it sends k·64 bits) and receives k knowledge items. Considering as above that they are r bits long, this means that $k \cdot r$ bits are received. These strings need to be filtered to obtain the final r bits.

In the case of buckets, the approach has a cost analysis similar to the one of k-anonymity, except for the transmission from client to server, which is just the number associated to a set of buckets. So, say, 64 bits. Then, for a number of buckets equal to b we will have that each query will retrieve $m = n _ { p a } / b$ indices from the knowledge base. So, the agent will receive $m \cdot r$ bits that need to be filtered to obtain the final r bits. Costs are summarized in Table 1.

## 4.2 Computational experiments

To test our system, we have defined a knowledge base consisting of $n _ { h } = 9 9$ facts and rules, which corresponds to $n _ { p a } = 4 6$ entries in the database. We have included a variety of Horn clauses, including both facts and rules, with non-recursive and recursive rules, and implementing as well two predicates that are benchmarks from SICStus Prolog [39]. In particular, we have implemented nreverse (naive reverse) which for a list of $n = 3 0$ elements requires a recursion of depth $( n + 1 ) + 2 n ( n + 1 ) = 3 1 + 4 6 5 = 4 9 6$ , and zebra.

Then, we have considered a set of 18 queries and executed them in diferent configurations. We have also executed each configuration several times. The configurations we have considered are the following ones.

“baseline” which means that a single external knowledge base is used, no privacy preserving access is applied,

– “pir\_bucket” which means that we use a single external knowledge base implementing private information retrieval with buckets,

$\mathrm { ^ { * } p i r \_ k A n o n } ^ { \mathrm { 3 } }$ which means that we use a single external knowledge base implementing private information retrieval with k-anonymity, and

– “itpir\_2servers” which means that we use the PIR protocol with two external knowledge bases.

Note that each of these types of executions are implemented for each of the above configurations. As queries in the external knowledge base are cached by the knowledge client, we have repeated executions in diferent ways. Computation times for each type of execution are provided.

<table><tr><td>KB</td><td>Mean St. dev. Min Max</td></tr><tr><td>Baseline</td><td>0.0969 0.1119 0.0178 0.1761</td></tr><tr><td>Bucket</td><td>0.0919 0.1081 0.0154 0.1684</td></tr><tr><td>Anon k4</td><td>0.1006 0.1142 0.0198 0.1814</td></tr><tr><td>Anon k6</td><td>0.0976 0.1143 0.01680.1785</td></tr><tr><td>Anon k8</td><td>0.0978 0.1129 0.0180 0.1776</td></tr><tr><td>Anon k10 0.0987</td><td>0.1114 0.0200 0.1775</td></tr></table>

Table 3. Execution time statistics for diferent knowledge base access mechanisms corresponding to two runs with cache. That is, first execution of the queries uses external access, and the second never does (case Standard caching).

– Non-cached. 5 executions of each query as if the query access was new each time but allowing caching for recursion as well as within the set of queries. For example, the first inference associated to nreverse requires an external knowledge base access to retrieve the associated Horn clauses, but no external access is required for recursive computation of nreverse. Moreover, no external access is required for another call of nreverse within the same set of queries.

– Standard caching. Two executions of a query with cache. That is, we have executed each set of queries twice. Nevertheless, external access is only applied once, and all the other executions use cached Horn clauses. Because of that, even if the first execution required external access, the second execution never requires this external access, and, thus, the overall time for executing the set of queries takes much less time.

– All cached. Baseline with no external query. In order to assess the execution costs using the external knowledge base, we also provide mean execution times when no external knowledge base is used. These results are computed using only cached Horn clauses.

The execution times for the first set of experiments with the minimum number of cached Horn clauses (case Non-cached) are provided in Table 2. Then, execution times for the second set of experiments with partial caches (case Standard caching) are provided in Table 3. Finally, execution times for all Horn clauses cached (case All cached) are provided in Table 4. As expected, the longest execution time is for the case non-cached, Table 2.

The computational results displayed in these tables for the two PIR databases are surprisingly good. In the case non-cached, its time is 0.17s for baseline and 0.22s for it2pir. It seems that the implementation of the XOR on the knowledge base is extremely fast and comparable to the other expected-light-weight implementations that need to generate multiple queries, forward these queries, and then filter the results of these queries. Query generation and post-processing is implemented in Python.

<table><tr><td>KB</td><td>Mean St. dev. Min Max</td></tr><tr><td>Baseline</td><td>0.0135 0.0027 0.0117 0.0176</td></tr><tr><td>Bucket 0.0149</td><td>0.0038 0.0120 0.0205</td></tr><tr><td>Anon k4 0.0158</td><td>0.0044 0.0121 0.0217</td></tr><tr><td>Anon k6</td><td>0.0140 0.0016 0.0125 0.0161</td></tr><tr><td>Anon k8 Anon k10</td><td>0.0126 0.0016 0.0116 0.0150 00.0144 0.0021 0.0131 0.0175</td></tr></table>

Table 4. Execution time statistics for diferent KB access mechanisms corresponding to five runs with no external access (i.e., baseline, case All cached).

Note that as all the query processing is performed by the knowledge base client and knowledge base server, execution times are in fact independent of the logical inference engine (our Prolog engine). In other words, the Prolog engine is agnostic on the type of knowledge base server used and it only queries the local knowledge base client and gets appropriate Horn clauses in return.

## 5 Conclusions

In this paper we claim that the standard formalization of memory poisoning attacks for semantic memory can be mitigated using secure memory mechanisms, secure updating of knowledge bases, and secure communication and communication logs. We implemented a knowledge base which includes provenance and digital signatures to avoid knowledge tampering. We introduced a private knowledge retrieval and private inference as a way to provide secure inference. We described our implementation and provided experiments and analysis of its cost.

We mentioned in the paper that there are some ways to improve the performance of the inference system that were not considered and are left as possible extensions. In relation to knowledge retrieval, we considered retrieval by predicate name and arity. More sophisticated retrievals are possible taking into account the arguments of the predicates (e.g., as Just In Time Indexing, JITI, utilities in SWI-Prolog). Also, it would be possible to use tabling, memoization of inference results.

In our implementation we considered only a single external knowledge base, or two knowledge bases with replicated content. Our approach can be easily extended to multiple knowledge base servers with diferent knowledge. Specialized knowledge bases will ofer their services listing their knowledge (i.e., providing appropriate hashing functions). In this case, agent’s client would proceed in a way analogous to our current system, but deciding first to which server it needs to send a request. I.e., instead of having a single hash function providing an index for the pair predicate name/arity we would have multiple hash functions, one for each knowledge base.

## Acknowledgements

This work was partially supported by the Wallenberg AI, Autonomous Systems and Software Program (WASP) funded by the Knut and Alice Wallenberg Foundation. Support by Swedish Research Council (VR 2023-05531) is also acknowledged. The work was also partially supported by project PID2024-156636NB-C21 (MATSE) funded by MCIN/AEI/10.13039/501100011033/ FEDER, UE.

## References

1. Alfarano, G.N., Khathuria, K., Weger, V, (2023) A survey on single server private information retrieval in a coding theory perspective, Appl. Algebra Eng. Commun. Comput., vol. 34, no. 3, pp. 335-358.

2. Anthropic (2025) Disrupting the first reported AI-orchestrated cyber espionage campaign, anthropic.com.

3. Baura, D., Calvanese, D. (2025) Assessing Privacy Requirements for Controlled Query Evaluation in OBDA, Proc. MDAI 2025 183-197.

4. Bordage, S., Lavauzelle, J. (2021) On the privacy of a code-based single-server computational PIR scheme, Cryptogr. Commun., vol. 13, no. 4, pp. 519-526.

5. Botti, V. (2025) Agentic AI and multiagentic: are we reinventing the wheel? ArXiv:2506.01463.

6. Chapman, A., Lauro, L., Missier, P., Torlone, R. (2024) Supporting Better Insights of Data Science Pipelines with Fine-grained Provenance, ACM Transactions on Database Systems 49:2. Article No.: 6, pages 1 - 42.

7. Chen, Z., Xiang, Z., Xiao, C., Song, D, Li, B. (2024) AgentPoison: Red-teaming LLM Agents via Poisoning Memory or Knowledge Bases, NeurIPS 2024.

8. Chong, H.Q., Tan, A.H., Ng, G.W. (2007) Integrated cognitive architectures: a survey, Artificial Intelligence Review 28 103-130.

9. Chor, B., Goldreich, O. , Kushilevitz, E., Sudan, M. (1995) Private information retrieval, Proceedings of IEEE 36th Annual Foundations of Computer Science, Milwaukee, WI, USA, 1995, pp. 41-50.

10. Chor, B., Goldreich, O., Kushilevitz, E., Sudan, M. (1998) Private information retrieval. Journal of the ACM 45, 965-981.

11. Das, B. C., Amini, M. H., Wu, Y. (2025) Security and privacy challenges of large language models, ACM Computing Surveys 57:6 (June 2025)

12. Dignum, V., Dignum, F. (2025) Agentifying Agentic AI. WMAC 2026 - AAAI 2026 Bridge Program on Advancing LLM-Based Multi-Agent Collaboration. arXiv:2511.17332

13. Domingo-Ferrer, J., Bras-Amorós, M., Wu, Q., Manjón, J.A. (2009) User-private information retrieval based on a peer-to-peer community. Data Knowl. Eng. 68(11): 1237-1252.

14. Freij-Hollanti, R., Gnilke, O. W., Hollanti, C., Karpuk, D. A. (2017) Private information retrieval from coded databases with colluding servers, SIAM J. Appl. Algebra Geom. 1 647-664.

15. Goldwasser, S, Micali, S (1984) Probabilistic encryption, Journal of Computer and System Sciences, Volume 28, Issue 2, 270-299.

16. Gu, Z., Lanti, D., Corcoglioniti, F., Panfilo, M. D., Mosca, A., Calvanese, D., Xiao, G. (2025) Ontology-based data federation and query optimization, Knowledge Based Systems 329 114216.

17. Holzbaur, L., Freij-Hollanti, R., Li, J., Hollanti, C. (2022) Toward the capacity of private information retrieval from coded and colluding servers, IEEE Trans. Inform. Theory 68 517-537.

18. Holzbaur L., Hollanti, C., Wachter-Zeh, A. (2020) Computational Code-Based Single-Server Private Information Retrieval, IEEE International Symposium on Information Theory (ISIT), Los Angeles, CA, USA, pp. 1065-1070.

19. Huang, C., Simitci, H., Xu, Y., Ogus, A., Calder, B., Gopalan, P., Li, J., Yekhanin, S. (2012) Erasure Coding in Windows Azure Storage, USENIX ATC 2012 15-26.

20. Isaev, P., Hammer. P. (2023) Memory System and Memory Types for Real-Time Reasoning Systems, Proc. AGI 2023, LNAI 13921 147-157.

21. Kotseruba, I., Tsotsos, J. K. (2020) 40 years of cognitive architectures: core cognitive abilities and practical applications, Artificial Intelligence Review 53 17-94.

22. Kushilevitz, E., Ostrovsky, R. (1997) Replication is not needed: single database, computationally-private information retrieval. In: Proc. of the 38th Annual IEEE Symposium on Foundations of Computer Science, pp. 364-373.

23. Laird, J. (2012) The Soar Cognitive Architecture. MIT Press. https://doi.org/ 10.7551/mitpress/7688.001.0001

24. Ostrovsky, R., Skeith, W.E. (2007) A Survey of Single-Database Private Information Retrieval: Techniques and Applications. In: Okamoto, T., Wang, X. (eds) Public Key Cryptography - PKC 2007. PKC 2007. Springer Lecture Notes in Computer Science, vol 4450.

25. Pindado, Z., Spendlhofer, T., Allam, M., Mehta, P., Martens, L., Peña, A. J. (2026) SoK of Private Deep Neural Network Inference with Approximate Fully Homomorphic Encryption, https://eprint.iacr.org/2026/047

26. Rando, J., Zhang, J., Carlini, N., Tramèr, F. (2025) Adversarial ML Problems Are Getting Harder to Solve and to Evaluate, Proc. DLSP 2025.

27. Samarati, P. (2001) Protecting Respondents’ Identities in Microdata Release, IEEE Trans. on Knowledge and Data Engineering, 13:6 1010-1027.

28. Sathiamoorthy, M., Asteris, M., Papailiopoulos, D. S., Dimakis, A. G., Vadali, R., Chen, S., Borthakur, D. (2013) XORing Elephants: Novel Erasure Codes for Big Data, VLDB Endow. 6 325-336.

29. Shah, N.B., Rashmi, K.V., Ramchandran, K (2014) One extra bit of download ensures perfectly private information retrieval, 2014 IEEE International Symposium on Information Theory, Honolulu, HI, USA, 2014, pp. 856-860.

30. Tajeddine, R., Gnilke, O. W., El Rouayheb, S. (2018) Private information retrieval from MDS coded data in distributed storage systems, IEEE Transactions on Information Theory, vol. 64, no. 11, pp. 7081-7093.

31. Torra, V. (2022) Guide to data privacy: Models, Technologies, Solutions, Springer.

32. Torra, V., Bras-Amorós, M. (2026) Memory poisoning and secure multi-agent systems, arXiv:2603.20357

33. Torra, V., Navarro-Arribas, G., Sanchez-Charles, D., Muntés-Mulero, V. (2017) Provenance and privacy, Proc. MDAI 2017.

34. Verma, N., Hollanti, C. (2024) Code-Based Single-Server Private Information Retrieval: Circumventing the Sub-Query Attack, 2024 IEEE International Symposium on Information Theory (ISIT), Athens, Greece, 2024, pp. 2880-2885.

35. Wooldridge, M. (2006) An introduction to multiagent systems, John Wiley and Sons.

36. Wu, Y., Liang, S., Zhang, C., Wang, Y., Zhang, Y., Guo, H., Tang, R., Liu, Y. (2025) From human memory to AI memory: a survey on memory mechanisms in the era of LLMs, arXiv:2504.1595v2.

37. https://www.mdai.cat/code/

38. https://www.swi-prolog.org/pldoc/man?section=prologjiti

39. https://sicstus.sics.se/performance.html