# Co<sup>2</sup>In: a Bi-level Memory Incremental Learning Framework with Knowledge Encoding, Consolidation, and Integration

Wenju Sun, Qingyong Li, Boyang Li, Wen Wang, Siyu Zhang, Yi Wei, Yangli-ao Geng

Abstract—Incremental learning (IL) aims to continually acquire new knowledge (plasticity) while retaining previously learned information (stability). However, striking a balance between plasticity and stability remains a significant challenge for intelligent systems. The human brain achieves exceptional balance, owing to various memory units that collaboratively encode and store information. Inspired by human memory mechanisms, this paper introduces an IL framework with knowledge enCoding, Consolidation, and Integration (Co<sup>2</sup>In). Co<sup>2</sup>In is designed as a bi-level memory architecture: a working memory for adaptive knowledge acquisition and a long-term memory dedicated to the persistent retention of information. The two memory modules work cooperatively during IL. The working memory first learns from new data to encode knowledge into parameters. Subsequently, Co<sup>2</sup>In performs a consolidation process to identify underlying patterns in the learned parameters and re-express them into a compact knowledge representation. Next, the knowledge representation and the identified patterns are transformed into separate network layers and integrated into the long-term memory. These designs empower Co<sup>2</sup>In to accumulate knowledge with high plasticity and stability. We evaluate Co<sup>2</sup>In on CIFAR-10, CIFAR-100, and Tiny-ImageNet under exemplar-free Class-IL and Task-IL settings. Experimental results show that Co<sup>2</sup>In achieves state-of-the-art performance with efficient memory consumption.

Index Terms—Incremental Learning, Continual Learning, Catastrophic Forgetting, Class-Incremental Learning.

✦

## 1 INTRODUCTION

In an ever-changing environment, incremental learning (IL) emerges as a critical challenge for intelligent systems to continually accumulate new knowledge while retaining previously acquired information [1]. In typical IL scenarios, a learner is required to learn a sequence of disjoint tasks. When learning a new task, only data from the new task are available. In this training schema, an effective IL learner should be able to flexibly learn and incorporate new information from the current task (plasticity) while retaining previously acquired old-task knowledge (stability). To strike a balance between plasticity and stability, a variety of IL strategies have been developed based on distinct technical mechanisms [2], [3]. Among them, memory-based IL methods have shown superior performance [4]–[7].

As illustrated in Fig. 1 (a), memory-based methods typically involve an auxiliary memory to store partial information from previous tasks. The information is used to relieve forgetting when learning new tasks. Depending on the type of information stored, memory-based methods can be further categorized into rehearsal-based and gradientmemory-based methods. Rehearsal-based methods [4], [7], [9]–[14] adopt an exemplar memory (or a generative model) to store (or generate) samples or features from old tasks, replaying them to reinforce old-task knowledge during new-task learning. Alternatively, gradient-memory-based methods [5], [6], [15]–[17] choose to monitor gradient directions that might impede old-task performance, updating the model only with gradients orthogonal to the stored ones.

Both rehearsal-based and gradient-memory-based methods adhere to a fundamental principle: the model must simultaneously acquire new knowledge while retaining old knowledge. However, knowledge acquiring and remembering typically interfere with each other, leading to the stability-plasticity dilemma [18]. For the rehearsal-based methods, fitting new tasks could overwrite old knowledge, particularly if the stored samples inadequately represent the data distribution of the old tasks, incurring compromised stability. On the other hand, gradient memory-based methods may hinder parameter updates along the direction of the steepest descent for loss minimization, especially in extreme cases where the entire gradient space is constrained, resulting in reduced plasticity.

In contrast, humans exhibit remarkable IL abilities, attributed to the multi-level structure of the human brain and effective memory procedures. Figure 1 (b) illustrates the classic Atkinson-Shiffrin human memory model [8]. In this schema, environmental information is first captured by sensory memory and temporarily stored in short-term memory. Through repeated rehearsals, the encoded information undergoes consolidation with the assistance of the prefrontal cortex, eventually leading to its permanent storage in the long-term memory [19], [20]. The layered structure effectively segregates the tasks of information encoding and storage into different brain organs, endowing human memory with an ideal balance of plasticity and stability.

![](images/6c542f6034fe287a770b89eda4ef3f8c45cd9ff4ba767906771bf80c746dba1e.jpg)  
Fig. 1. Diagram of memory-based IL methods, Atkinson-Shiffrin human memory model [8], and the proposed $C o ^ { 2 } I n$ . (a) Memory-based IL methods prevent the model from forgetting when learning by replaying old-task exemplars or restricting gradient directions. (b) In the human brain, information is temporally stored in short-term memory. After repeated rehearsal, the information will be permanently remembered in long-term memory. (c) Inspired by the human brain, $C o ^ { 2 } I n$ acquires new knowledge with working memory and integrates it into long-term memory for storage.

Inspired by the mechanism of human memory, we introduce a bi-level memory framework for IL. As illustrated in Fig. 1(c), the framework consists of a working memory (akin to the short-term memory in the human memory model) and a long-term memory, each implemented as neural networks but with distinct roles. The working memory dynamically acquires new knowledge, while the long-term memory consistently stores all learned information. This cognitively inspired framework decouples learning from remembering, ensuring sufficient plasticity and stability.

Based on the bi-level memory framework, we propose the knowledge enCoding, Consolidation, and Integration $( C o ^ { 2 } I n )$ strategy to continually accumulate knowledge. Upon encountering a new task, the working memory first encodes the new knowledge into trainable parameters. Acknowledging the inherent over-parameterization issue within these trained parameters [21], $C o ^ { 2 } .$ In executes a consolidation process to identify the underlying patterns of the trained parameters. The identified patterns serve as a set of basis to construct a parameter space that encapsulates valuable knowledge but with significantly lower dimensions. By projecting the trained parameters with respect to the pattern basis, $C o ^ { \mathrm { { 2 } } } I n$ generates a compact representation of valuable knowledge involved in the trained parameters. Then, the long-term memory re-organizes the knowledge representations and pattern basis of all learned tasks into independent layers, ultimately integrating them into a unified network. The unified network directly executes inference operations. Moreover, we equip $C o ^ { 2 } I n$ with a novel quadratic classifier, which establishes compact decision boundaries. As a result, $C o ^ { 2 } I n$ contains cross-task discriminability without necessitating the task identification of the query sample, thus enhancing its generalizability in task-agnostic scenarios [22]. The contributions of this paper are listed as follows:

Inspired by the human memory system, we propose $C o ^ { 2 } I n$ , a bi-level memory IL method that decouples knowledge acquisition and retention to balance plasticity and stability. A series of strategies are devised for knowledge encoding, consolidation, and integration within ${ \bf \ddot { \boldsymbol { C } } } \boldsymbol { o } ^ { 2 } I \boldsymbol { n } _ { \perp }$ , enabling effective IL with high memory and computational efficiency.

We propose a quadratic classifier to encourage $C o ^ { 2 } \mathrm { _ { \perp } }$ In to learn compact decision boundaries, which further extends its capability for task-agnostic scenarios, so $C o ^ { 2 } I n$ is applicable to both Task-IL and Class-IL.

$C o ^ { 2 } I n$ is evaluated on CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets. Our experimental results demonstrate that $C o ^ { 2 } I _ { 1 }$ n outperforms most state-ofthe-art baselines with lower memory usage.

A preliminary conference version of this work, referred to as BMKP, has been presented in [23]. This paper expands upon the conference version with several key enhancements: (1) We introduce a more efficient knowledge integration process, which markedly reduces computational overhead during inference. (2) A quadratic classifier is incorporated into the bi-level memory framework, notably improving the performance of $C o ^ { 2 } I n$ in exemplar-free Class-IL, an aspect not covered in BMKP. (3) We offer more experiments to comprehensively validate $C o ^ { 2 } I n$

## 2 RELATED WORK

## 2.1 Incremental Learning

According to the techniques for balancing stability and plasticity, current IL methods can be classified into three categories: memory-based, regularization-based, and architecture-based.

Memory-based methods typically preserve partial previous tasks information to mitigate catastrophic forgetting.

According to the stored information, memory-based methods can be further split into rehearsal-based methods and gradient-memory based methods. Some rehearsal-based methods [4], [7], [9], [13], [24]–[27] store old-task samples as exemplars, and replay them alongside new data during learning. Other rehearsal-based methods [14], [28]–[31] maintain a generative model capable of producing pseudosamples that mimic the distribution of old tasks, also known as generative replay. On the other hand, gradient-memory based methods [5], [10], [11], [15], [17], [32] adopt a different strategy by preserving the directions of gradients that could disrupt the performance on earlier tasks. These recorded gradients serve as constraints during model updates: only those gradient directions orthogonal to them are permitted for updating the parameters, thereby preserving previously acquired knowledge. Although $C \check { o } ^ { 2 } \checkmark$ In employs a bi-level memory structure, it avoids exemplar storage and gradient constraints by instead retaining low-rank knowledge patterns and representations.

Regularization-based methods alleviate forgetting by incorporating regularization terms into the loss function. Learning without Forgetting (LwF) [33] first applies knowledge distillation to regularize the model, aiming to align the current model with a teacher model representing the oldtask knowledge. As knowledge distillation is effective in reducing forgetting, this technique has become a standard IL baseline and has been widely adopted in many IL methods [33]–[37]. Another prominent regularization-based method is Elastic Weight Consolidation (EWC) [38]. EWC leverages the Fisher information matrix to estimate the importance of each parameter and punishes their changes during subsequent task training. Several IL methods follow a similar idea to alleviate forgetting, such as Synaptic Intelligence (SI) [39], oEWC [40], and Memory Aware Synapses (MAS) [41]. In this paper, we also propose a regularization term for $C o ^ { 2 } .$ In to encourage the model to reuse previously acquired knowledge patterns in new tasks.

Architecture-based methods address forgetting by employing specifically designed network structures. Some mask-based methods [42]–[45] learn a set of parameter masks that indicate the assignment of neurons. These masks facilitate the selective activation or deactivation of specific parts of the network for each task. Other expansion-based methods [31], [46]–[49] expand the width of networks to ensure sufficient network capacity for new-task learning. During inference, most architecture-based methods require explicit task identification of queries to adapt the parameters accordingly, which limits their application in taskagnostic scenarios, $\mathrm { e . g . }$ Class-IL. To address this issue, some architecture-based methods leverage one-class classifiers [31], [47], [50] or confidence [51] to achieve task-wise discrimination. Our $C o ^ { 2 } I n$ increases the network capacity by dynamically expanding pattern basis. Moreover, similar to one-class classifier based methods, it learns compact decision boundaries using a quadratic classifier, enabling task-agnostic inference without explicit task identifiers.

Additionally, various IL methods are inspired by the human cognitive process. Drawing inspiration from the phenomenon of synaptic expansion (neurons expand new synapses to acquire new information), Wang et al. [52] propose to expand new network branches when learning a new task. The parameters of the newly expanded branch are selectively merged with the original branch through knowledge distillation. Sarfraz et al. [53] suggest a multimemory system to mimic the human memory structure, including episodic memory, working memory, and longterm memory. $C o ^ { 2 } I n$ shares a similar motivation with these approaches but with different underlying mechanisms. In ${ \dot { C } } { \dot { o } } ^ { \dot { 2 } } I n ,$ all knowledge is stored in the long-term memory, while the working memory is solely responsible for acquiring knowledge. With this design, $C o ^ { 2 } \dot { I } n$ decouples the roles of learning and remembering into different memory units, thereby achieving both high plasticity and stability. In contrast, the responsibilities of knowledge acquiring and remembering in existing cognitive-inspired IL methods are still concentrated within the same model.

Some recent IL methods are designed for pretrained models, which benefit from a well-initialized, transferable representation space. Prompt-based methods insert learnable prompts into frozen backbones, where L2P [54] dynamically selects input-dependent prompts, DualPrompt [55] disentangles task-shared and task-specific prompts across layers, and CODA-Prompt [56] further enables compositional prompt generation. Adapter-based methods introduce lightweight modules for task adaptation, including EASE [57], which learns task-specific adapters via prototype completion to mitigate feature–classifier mismatch; SSIAT [58], which maintains a shared adapter and performs unified classifier retraining; and ACMAP [59], which improves scalability by merging multiple adapters. Other LoRA-based approaches constrain updates to low-rank subspaces, with InfoLoRA [60] constructing interference-aware subspaces and CL-LoRA [61] combining shared and task-specific LoRA modules. In this paper, the proposed $C o ^ { 2 } .$ In focuses on the training-from-scratch setting, while it can be naturally extended to pretrained-based IL by learning a low-rank residual branch, enabling seamless integration with parameterefficient adaptation paradigms.

## 2.2 Human Memory Mechanism

Human beings exhibit remarkable plasticity and stability throughout their lives, owing to the brain’s multi-layered memory structure and effective memory mechanisms. The classical Atkinson-Shiffrin model [8] posits that human memory includes three main units: sensory memory, shortterm memory, and long-term memory. Sensory memory acts as a cache for sensory activations like sounds or touches, transferring information to short-term memory when attention is concentrated. Short-term memory holds information for immediate use, and after repeated rehearsal, this information is consolidated into long-term memory for permanent storage. This hierarchical structure ensures both the plasticity and stability of the brain.

This paper draws inspiration from the human brain’s memory mechanisms. Leveraging the hierarchical structure of multi-level human memory [8], we propose a bi-level memory IL framework that separates knowledge acquisition and retention into specialized memory units, ensuring both plasticity and stability. guided by the cognitive mechanisms of the human memory process [20], we propose the knowledge encoding, consolidation, and integration strategy within the bi-level memory framework. This strategy ensures both memory and computational efficiency.

TABLE 1 Mathematical notations and their descriptions.  
Notation Description   
General Notations   
$\left. . \right. _ { 1 }$ $\ell _ { 1 }$ norm: sum of absolute values of all elements.   
$\lVert . \rVert _ { F }$ Frobenius norm: $\sqrt { \sum _ { i , j } | a _ { i j } | ^ { 2 } }$ for a matrix $A .$   
$\mathrm { S V D ( . ) }$ Singular Value Decomposition: $\mathrm { S V D } ( \pmb { X } ) = \pmb { U } \pmb { \Sigma } \pmb { V } ^ { \top }$ , where U and V are orthogonal, and Σ is diagonal.   
$\underline { L } _ { c e } ( . ) ^ { ' }$ The cross-entropy loss.   
$[ . , . ]$ The concatenation operation.   
Dataset and Layer-wise Features Transformation Notations   
$\mathcal { D } _ { t }$ Dataset of task $t \colon { \mathcal { D } } _ { t } = \{ ( x _ { k } , y _ { k } ) \} _ { k = 1 } ^ { n \tilde { t } } ,$ where n<sub>t</sub> is the sample size.   
$X ^ { l } \in \mathbb { R } ^ { p ^ { l } \times n }$ Input features of the l-th layer: $\mathbf { { \cal { X } } } ^ { l } = [ \pmb { x } _ { 1 } ^ { l } , \dots , \pmb { x } _ { k } ^ { l } , \dots , \pmb { x } _ { n } ^ { l } ] ,$ , where $\boldsymbol { x } _ { k } ^ { l } \in \mathbb { R } ^ { p ^ { l } }$ corresponds the feature for the k-th sample.   
$Z ^ { l } \in \mathbb { R } ^ { p ^ { l + 1 } \times n }$ Transformed features of the l-th layer: $Z ^ { l } = W ^ { l } X ^ { l }$   
$W ^ { l } \in \mathbb { R } ^ { p ^ { l + 1 } \times p ^ { l } }$ Parameters of the l-th layer.   
σ The parameter-free non-linear unit such as the ReLU activation: $\pmb { X } ^ { l + 1 } = \sigma ( \pmb { Z } ^ { l } ) .$   
$C o ^ { 2 } I n$ Specific Notations: Knowledge Representation and Pattern Basis   
$\boldsymbol { A }$ Knowledge representations of all tasks: $\mathcal { A } = \{ \mathcal { A } _ { 1 } , \mathcal { A } _ { 2 } , \cdot \cdot \cdot \} ,$ , where $\boldsymbol { \mathcal { A } } _ { t } = \{ A _ { t } ^ { 1 } , A _ { t } ^ { 2 } , . . . , A _ { t } ^ { L } \}$ is the knowledge   
representation specific to task t   
$\boldsymbol { A } _ { t } ^ { l } \in \mathbb { R } ^ { r _ { t } ^ { l } \times p ^ { l } }$ Knowledge representation of the l-th layer for task $t .$   
B<sub>t</sub> $B _ { t } = \{ B _ { t } ^ { 1 } , . . . , \mathbf { \dot { B } } _ { t } ^ { L } \}$ represents the pattern basis of all layers after task t.   
$\bar { B _ { t } ^ { l } } \in \mathbb { R } ^ { p ^ { l + 1 } \times r _ { t } ^ { l } }$ Pattern basis matrix of the l-th layer after learning task t.   
r<sup>l</sup> Number of retained basis for the l-th layer after task t.   
$\hat { \hat { W } } ^ { l } \in \mathbb { R } ^ { p ^ { l + 1 } \times p ^ { l } }$ The projection of $\mathbf { \mathbf { \mathbf { W } } } ^ { l }$ onto the basis of $\mathbf { \dot { \boldsymbol { B } } } ^ { l } \colon \hat { \boldsymbol { W } } ^ { l } = \boldsymbol { B } ^ { l } \boldsymbol { B } ^ { l } \mathbf { \bar { \Sigma } } ^ { \top } \mathbf { \boldsymbol { W } } ^ { l } .$   
$\tilde { \boldsymbol W } ^ { l } \in \mathbb R ^ { p ^ { l + 1 } \times p ^ { l } }$ The residual of the orthogonal projection of $W$ onto CKS: $\tilde { \pmb { W } } ^ { l } = \pmb { W } ^ { l } - \pmb { B } ^ { l } \pmb { B } ^ { l ^ { \top } } \pmb { W } ^ { l } .$

## 3 PROBLEM DEFINITION

Currently, two IL settings are drawing attention from the research community, namely task incremental learning (Task-IL) and class incremental learning (Class-IL) [22] :

Definition 3.1 (Task-IL). Consider a sequence of T classification tasks $\{ \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$ , each associated with a unique dataset $\mathcal { D } _ { t } = \{ ( \boldsymbol { x } _ { k } , y _ { k } ) \} _ { k = 1 } ^ { n _ { t } } ,$ where $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ and $\mathbf { \boldsymbol { y } } _ { k } \in \mathcal { N } _ { t }$ represents a inputlabel pair. It is critical to note that the classes for classification in different tasks are mutually exclusive, $i . e . , \mathcal { V } _ { i } \overset { } { \cap } \mathcal { V } _ { j } = \overset { \cdot } { \emptyset } , \forall i \neq j .$ The goal ofthe model is to incrementally learn these tasks, without the need to refer back to the data of previous tasks once they have been learned. During the test stage of Task-IL, each query sample is accompanied by its task identification to assist in classification.

Definition 3.2 (Class-IL). Class-IL shares the same setup as Task-IL, except during the testing stage, where task identification is no longer available for each query sample.

The proposed $C o ^ { 2 } I n$ is feasible for both Task-IL and Class-IL. Moreover, considering the application in privacysensitive scenarios, this paper focuses on the two settings under the challenging exemplar-free restriction [62], where no samples from previous tasks are retained for replaying.

To achieve this, we investigate layer-wise operations of neural networks, where features are progressively transformed across layers. By decomposing and preserving parameters layer-by-layer, the model achieves efficient IL while mitigating forgetting. Relevant notations for these operations are summarized in Table 1, organized into the following categories:

• General notations: commonly used mathematical operations and norms, such as $\ell _ { 1 }$ norm $( \left. . \right. _ { 1 } ) .$ , Frobenius norm $( \left\| . \right\| _ { F } ) .$ , singular value decomposition (SVD), and cross-entropy loss $( L _ { c e } ( . ) )$

Dataset and layer-wise feature transformations, including input features $( \boldsymbol { X } ^ { l } ) ,$ , transformed features $( Z ^ { l } )$ , parameters $( W ^ { l } )$ , and non-linear activation (σ).

• Notations specific for $C o ^ { 2 } I n \colon$ knowledge representations across tasks $( { \mathcal { A } } ) .$ , pattern basis $( \hat { B _ { t } } ) _ { i }$ , and projection-related terms such as $\hat { W } ^ { l }$ (projection of weights) and $\tilde { W } ^ { l }$ (residual projection).

## 4 BI-LEVEL MEMORY FRAMEWORK

A naive IL solution is to train an independent model for each task, but storing and running all task-specific models lead to increasing memory and computational overhead. Our proposed approach involves using low-rank matrices to compactly represent incremental knowledge learned from each additional task and gradually expand the network width. This forms the bi-level structure introduced in this section, a novel IL framework that separates knowledge acquisition and retention into two distinct memory units. Before describing the bi-level structure, we first elaborate on two knowledge spaces that serve as a guide for organizing knowledge within these memory units.

## 4.1 Knowledge Spaces

In the context of neural networks, we define knowledge as the inherent capability to map a specific input into the corresponding output. The network parameters can be thought of as an explicit representation of knowledge, and we define the space which resides in as Parameter Knowledge Space (PKS). PKS often exhibits redundancy in dimensions, as current deep neural models are typically overparameterized [21]. In light of this, we introduce Core Knowledge Space (CKS) as a subspace of PKS that contains all meaningful information but with much fewer dimensions.

![](images/749aa352707306d037619da23cbc16e58e411e734dddac08be4d8f4a2df48793.jpg)  
Fig. 2. Illustration of the learning process of $C o ^ { 2 } I n .$ (a) The task-t training process of $C o ^ { 2 } I n ,$ including knowledge encoding, knowledge consolidation, and knowledge integration steps. (b) In the knowledge consolidation step, Co<sup>2</sup>In recognizes new pattern basis and projects the parameters to form compact knowledge representations. (c) In the knowledge integration step, $C o ^ { 2 } I n$ reorganizes the long-term memory into a unified neural network.

We pursue a CKS by investigating the mechanism of neural networks. Consider an L-layered neural network with parameters ${ \mathcal W } ~ = ~ \{ W ^ { 1 } , . . . , \dot { W } ^ { L } \}$ , where $W ^ { l }$ is a $p ^ { l + 1 } { - } \dot { \boldsymbol { \mathrm { b y } } } { - } p ^ { l }$ matrix representing the parameters of the l-th layer. The network receives input features and executes the following feature transformation layer by layer:

$$
\pmb { Z } ^ { l } = \pmb { W } ^ { l } \pmb { X } ^ { l } \quad \pmb { X } ^ { l + 1 } = \sigma ( \pmb { Z } ^ { l } ) ,\tag{1}
$$

where $X ^ { l }$ is the input of layer l and $X ^ { l + 1 }$ is the output of layer l. It is worth emphasizing that, without loss of generality, we conceptualize the parameters $W ^ { l }$ as a matrix, given the inherent linearity within current network layers. Section 5.4 details the extension to convolutional layers.

Equation (1) evidence that the feature $Z ^ { l }$ is a linear combination of the columns of $W ^ { l }$ . This observation indicates that only those columns of $W ^ { l }$ that contribute to the combination of $Z ^ { l }$ contain valuable knowledge, i.e., parts of $W ^ { l }$ that reside within the column space of $\breve { Z } ^ { l }$ . To elucidate further, suppose that the columns of a ${ p ^ { l + 1 } { - } { \bf b y } { - } r ^ { l } ( r ^ { l } \ll p ^ { l + 1 } ) }$ matrix $B ^ { l }$ form an orthogonal basis for the space spanned by $Z ^ { l }$ , the first term in Eq. (1) can be re-expressed as:

$$
\begin{array} { r l } & { \boldsymbol { Z ^ { l } } = \boldsymbol { B ^ { l } B ^ { l } } ^ { \top } \boldsymbol { Z ^ { l } } = \boldsymbol { B ^ { l } B ^ { l } } ^ { \top } \boldsymbol { W ^ { l } } \boldsymbol { X ^ { l } } } \\ & { \qquad = \hat { W } ^ { l } \boldsymbol { X ^ { l } } } \\ & { \qquad = \boldsymbol { B ^ { l } A ^ { l } X ^ { l } } , } \end{array}
$$

which forms a linear space equivalent to the column space of $Z ^ { l } \colon \hat { W } ^ { l }$ transforms the given input $X ^ { l }$ to the expected output $\pmb { Z } ^ { l } .$ , implying that $\hat { W } ^ { l }$ encapsulates the most knowledge in $\mathbf { } W ^ { l } . \hat { A } ^ { l } = \bar { B } ^ { l ^ { \top } } \mathbf { } W ^ { l }$ stands for the coefficient matrix of $\hat { W } ^ { l }$ under the basis of $B ^ { l }$ , which can be interpreted as a compact knowledge representation of $W ^ { l }$ with respect to the basis $B ^ { l }$ . We conceptualize the space spanned by $Z ^ { l }$ as CKS and refer $B ^ { l }$ as pattern basis. One possible way to find such a $B ^ { l }$ is to perform singular value decomposition (SVD) of $\mathrm { S V D } ( Z ^ { l } ) = \mathbf { \hat { U } } ^ { l } \Sigma ^ { l } V ^ { l } ^ { \top }$ and let $B ^ { l } = \pmb { U } ^ { l }$

## 4.2 Bi-level Memory Structure

We propose a bi-level memory framework based on PKS and CKS. As depicted in Fig. 1 (c), The framework consists of the following two memory units:

Working memory is responsible for adaptively acquiring new knowledge in PKS. It is implemented as a L-layer neural network, with parameters denoted as $\mathcal { W } \overset { \cdot } { = } \{ W ^ { 1 } , . . . , W ^ { L } \}$

Long-term memory dynamically maintains an entire CKS during incremental learning, including a set of pattern basis $\vec { B } ~ = ~ \{ B ^ { 1 } , . . . , B ^ { \check { L } } \}$ for each layer, alongside the projected knowledge representations for all learned tasks $\boldsymbol { \mathcal { A } } = \{ \mathcal { A } _ { 1 } , . . . , \breve { \mathcal { A } } _ { T } \}$

The bi-level memory framework decouples the roles of knowledge acquiring and remembering into specialized memory units, thereby ensuring high plasticity and stability.

## 5 KNOWLEDGE ENCODING, CONSOLIDATION, AND INTEGRATION

Based on the bi-level memory framework, this section outlines an IL strategy with knowledge encoding, consolidation, and integration $( C o ^ { 2 } I n )$ . As illustrated in Fig. 2, during the knowledge encoding phase, the working memory fits with new-task data to encode knowledge into parameters. In the knowledge consolidation stage, Co<sup>2</sup>In projects the parameter-formed knowledge onto CKS to constitute a much more compact knowledge representation. Subsequently, in the knowledge integration step, the long-term memory re-organizes all knowledge representations and pattern basis into a unified neural network.

We will elaborate on knowledge encoding, consolidation, and integration steps in Section 5.1, 5.2, and 5.3, respectively. Additionally, in Section $5 . 4 ,$ we present a straightforward strategy for extending our methods to convolutional neural networks. Lastly, in Section 5.5, we propose a quadratic classifier that enhances the Class-IL ability of $C o ^ { \dot { 2 } } .$ In with compact decision boundaries.

## 5.1 Knowledge Encoding

Knowledge encoding aims to encapsulate new knowledge from data into the working-memory parameter.

During training, we incorporate a representation compaction regularizer to ensure that the working memory aligns with the existing pattern basis, formulated as follows:

$$
L _ { r e g } ( \mathcal { W } ) = \sum _ { \boldsymbol { W } ^ { l } \in \mathcal { W } } \frac { \| \tilde { \boldsymbol { W } } ^ { l } \| _ { F } ^ { 2 } } { \| \tilde { \boldsymbol { W } } ^ { l } \| _ { 1 } ^ { \top } \tilde { \boldsymbol { W } } ^ { l } \| _ { 1 } } ,
$$

where $\tilde { \mathbf { W } } ^ { l } = \mathbf { W } ^ { l } - B _ { t - 1 } ^ { l } { B _ { t - 1 } ^ { l } } ^ { \top } \mathbf { W } ^ { l }$ signifies the residual of the orthogonal projection of $W ^ { l }$ onto CKS. This regularizer serves two pivotal roles in enhancing performance and efficiency. Firstly, the numerator of Eq. (5.1) penalizes the Frobenius norm of the projection residual, which encourages knowledge encoding by leveraging previously learned patterns $B _ { t - 1 } ^ { l } ,$ thereby facilitating the transfer of old-task knowledge to current task learning. Secondly, the maximization of the $\ell _ { 1 }$ vector norm in the denominator encourages the columns of the residual to be co-linear:

$$
{ \left\| \tilde { W } ^ { l \top } \tilde { W } ^ { l } \right\| } _ { 1 } = \sum _ { i , j } | \tilde { \pmb { w } } _ { i } ^ { l \top } \tilde { \pmb { w } } _ { j } ^ { l } | ,
$$

thereby regularizing $\tilde { W } ^ { l }$ to be approximately low-rank $( \mathrm { i . e . }$ the rows approximately lie in a low-dimensional subspace). Empirical evidence in Fig. 8 has shown that with an appropriate weight, the regularizer significantly reduces memory overhead while improving the performance of $C o ^ { 2 } I n$

The overall loss function for knowledge encoding is

$$
{ \mathcal W } _ { t } \longleftarrow \underset { \mathcal W } { \arg \operatorname* { m i n } } L _ { c e } ( { \mathcal W } , { \mathcal D } _ { t } ) + { \lambda } L _ { r e g } ( { \mathcal W } ) ,\tag{2}
$$

where λ is a hyper-parameter weighting the regularizer.

## 5.2 Knowledge Consolidation

$C o ^ { 2 } .$ In next executes knowledge consolidation to project new knowledge from PKS onto CKS. To minimize the loss of information during projection, we initially recognize new pattern basis based on the column space of $\boldsymbol { \underline Z { _ t ^ { l } } }$ . Instead of directly performing SVD on $\begin{array} { r } { \pmb { Z _ { t } ^ { l } } = \mathbf { \bar { U } } _ { t } ^ { l } \pmb { \Sigma _ { t } ^ { l } } \pmb { V _ { t } ^ { l } } ^ { \top } } \end{array}$ and letting $B _ { t } ^ { l } = \dot { U } _ { t } ^ { l }$ , we perform SVD on the incremental knowledge learned from task $t ,$ represented by ${ Z _ { t } ^ { l } } - { B _ { t - 1 } ^ { l } } { B _ { t - 1 } ^ { l } } ^ { \top } { Z _ { t } ^ { l } } .$

$$
\begin{array} { r l } { { U _ { t } ^ { l } } { \Sigma _ { t } ^ { l } } { V _ { t } ^ { l } } ^ { \top } \longleftarrow \mathrm { S V D } \left( { Z _ { t } ^ { l } } - B _ { t - 1 } ^ { l } { B _ { t - 1 } ^ { l } } ^ { \top } Z _ { t } ^ { l } \right) } & { } \\ { { B _ { t } ^ { l } } \longleftarrow \lbrack { B _ { t - 1 } ^ { l } } , { U _ { t } ^ { l } } ] . } \end{array}\tag{3}
$$

![](images/83e743dae839f825077749947834f64e33e34b52a3a96595594c80df7b98f58d.jpg)  
Fig. 3. Diagram of the zero-padding operation during multiplying the task-t pattern basis $\boldsymbol { B } _ { t } ^ { l } \in \dot { \mathbb { R } } ^ { p ^ { l + 1 } \times r _ { t } ^ { l } }$ with previously learned task-j $( j < t )$ knowledge representation $\boldsymbol { A } _ { j } ^ { l } \in \mathbb { R } ^ { r _ { j } ^ { l } \times \dot { p } ^ { l } }$ in Eq. (6).

It is worth noting that $\pmb { U } _ { t } ^ { l }$ is guaranteed to be orthogonal to the existing basis $B _ { t - 1 } ^ { l }$ . The orthogonality stems from the operation of projecting $ { \boldsymbol { Z } } _ { t } ^ { l }$ onto the left null space of $B _ { t - 1 } ^ { l }$ , which is detailed in Theorem 8.1. Moreover, to keep the basis compact, we conduct basis selection with an introduced retained singular-value ratio<sup>1</sup>. Only the singular vector corresponding to retained values is picked as a basis.

Based on $\bar { \boldsymbol { B } } _ { t } ^ { l } , \boldsymbol { C o } ^ { 2 } I \boldsymbol { r }$ then projects $\mathbf { } W _ { t } ^ { l }$ onto CKS :

$$
A _ { t } ^ { l } \longleftarrow B _ { t } ^ { l ^ { \top } } W _ { t } ^ { l } .
$$

By applying this projection step to all layers, we obtain the updated pattern basis $\boldsymbol { B _ { t } } ~ \stackrel { \cdot } { = } ~ \{ B _ { t } ^ { 1 } , \ldots , \tilde { B } _ { t } ^ { L } \}$ and the knowledge representation $\mathcal { A } _ { t } = \{ A _ { t } ^ { 1 } , \ldots , A _ { t } ^ { L } \}$

The selection of singular vectors may exclude some valuable patterns associated with minor singular values, potentially resulting in a loss of knowledge. To mitigate this issue, we propose a refinement strategy: finetuning $\boldsymbol { A } _ { t }$ with $\mathcal { D } _ { t }$ in CKS, which can be formulated as follows:

$$
\underset { \ b { \mathcal A } _ { t } } { \underbrace { \mathcal A _ { t } } } \longleftarrow \underset { \ b { \mathcal A } _ { t } } { \underbrace { \arg \operatorname* { m i n } } } L _ { c e } ( \ b { \mathcal B } _ { t } \ b { \mathcal A } _ { t } , \ b { \mathcal D } _ { t } ) ,\tag{4}
$$

where $B _ { t } { A } _ { t } ~ = ~ \{ B _ { t } ^ { 1 } A _ { t } ^ { 1 } , . . . , B _ { t } ^ { L } A _ { t } ^ { L } \}$ denotes the layer-bylayer multiplication of pattern basis $B _ { t }$ and knowledge representation $\boldsymbol { \mathcal { A } } _ { t } .$ Upon completion of the finetuning, we transmit $\boldsymbol { A } _ { t }$ and $B _ { t }$ into the long-term memory for the subsequent knowledge integration.

## 5.3 Knowledge Integration

Next, $C o ^ { 2 } I n$ integrates all acquired knowledge to enable efficient inference. The key idea is to rearrange the multiplication order in line with the associative law:

$$
{ \cal Z } _ { t } ^ { l } { = } { \hat { W _ { t } } } ^ { l } { \pmb X } ^ { l } { = } ( { \pmb B } _ { t } ^ { l } { \pmb A } _ { t } ^ { l } ) { \pmb X } ^ { l } { = } { \pmb B } _ { t } ^ { l } ( { \pmb A } _ { t } ^ { l } { \pmb X } ^ { l } ) { \pmb X } ^ { l + 1 } { = } \sigma ( { \pmb Z } ^ { l } ) .\tag{5}
$$

Equation (5) provides an opportunity to integrate all knowledge representations into a separate layer:

$$
\begin{array} { r l } & { [ \boldsymbol { Z } _ { 1 } ^ { l } , \ldots , \boldsymbol { Z } _ { t } ^ { l } ] = \boldsymbol { B } _ { t } ^ { l } [ \boldsymbol { A } _ { 1 } ^ { l } \boldsymbol { X } _ { 1 } ^ { l } , \ldots , \boldsymbol { A } _ { t } ^ { l } \boldsymbol { X } _ { t } ^ { l } ] } \\ & { [ \boldsymbol { X } _ { 1 } ^ { l + 1 } , \ldots , \boldsymbol { X } _ { t } ^ { l + 1 } ] = \sigma \left( [ \boldsymbol { Z } _ { 1 } ^ { l } , \ldots , \boldsymbol { Z } _ { t } ^ { l } ] \right) . } \end{array}\tag{6}
$$

Here, for simplicity, we omit the zero-padding operation on $A _ { j } ^ { l }$ when $j ~ < ~ t .$ . This operation pads zeros for the superfluous dimensions in $A _ { i } ^ { l }$ (corresponding to the basis of $\mathbf { \bar { \mathbf { B } } } _ { t } ^ { l }$ that are never used by $\overset { \triangledown } { \boldsymbol { A } } _ { j } ^ { l } )$ , as depicted in Fig. 3.

![](images/8463fce29468087ae3e3a3081b5b323631351d1943921a50de99fd755ce3f526.jpg)  
Fig. 4. Diagram of converting a convolutional layer into a linear layer. $w ^ { \tilde { l } } , h ^ { l }$ , and $\mathit { \Pi } _ { c } { l }$ denote the width, height, and channel of the input tensor, respectively; $w ^ { l + 1 } , h ^ { l + 1 }$ , and $c ^ { \dot { l } + 1 }$ correspond to the width, height, and channel of the output feature tensor, respectively, and $k ^ { l }$ signifies the size of the convolution kernel.

Knowledge integration facilitates parallel inference across tasks. Given a set of query samples $X , \ C o ^ { 2 } I n$ begins by duplicating the samples t times, yielding inputs $[ X _ { 1 } ^ { 1 } , . . . , X _ { t } ^ { 1 } ]$ corresponding t tasks at the first layer. Each layer then transforms the feature according to Eq. (6), eventually generating the classification logits $[ Z _ { 1 } ^ { L } , . . . , Z _ { t } ^ { L } ]$ . The class associated with the highest logit value is selected as the final classification result:

$$
\hat { \mathbf { Y } } = \arg \operatorname* { m a x } [ { Z _ { 1 } ^ { L } , . . . , Z _ { t } ^ { L } } ] .\tag{7}
$$

Note that this parallel inference is necessary only for Class-IL or when a query batch contains data from multiple tasks. In typical Task-IL settings, where a query batch originates from a single task and the task ID is available, it suffices to compute only the logits associated with that specific task $( \mathrm { i } . \mathbf { e } . , Z _ { t } ^ { \hat { L } }$ for the query task t).

Knowledge integration significantly enhances computational efficiency. Consider a straightforward serial inference that directly re-composes each task parameter. Upon receiving query samples X, the serial approach first reconstructs the task-1 parameters via $B _ { t } \hat { A _ { 1 } } ,$ then loads them into working memory for predicting logits for task-1 classes ${ \cal Z } _ { 1 } ^ { L } \ = \ f ( \breve { X } ; B _ { t } { \mathcal A } _ { 1 } )$ . This prediction process is then repeated iteratively for other task’s knowledge representation, yielding logits $\{ Z _ { 1 } ^ { L } , . . . , Z _ { t } ^ { L } \}$ }. Notably, due to memory constraints, it is impractical to recompose all task parameters $\{ B _ { t } \mathcal { A } _ { 1 } , . . . , B _ { t } \mathcal { A } _ { t } \}$ simultaneously. Assuming $r _ { t } ^ { l }$ signifies the number of basis after learning task $t ,$ the computational complexity of serial prediction for the l-th layer within task t approximates $\mathcal { O } ( p ^ { l + 1 } p ^ { l } n + p ^ { l + 1 } r _ { t } ^ { l } p ^ { l } ) = \mathcal { O } ( ( r _ { t } ^ { l } { + } n ) p ^ { l ^ { 2 } } )$ , given that $p ^ { l } \approx p ^ { l + 1 }$ . Here, $p ^ { l }$ represents the input dimension for the l-th layer, and n is the number of samples. This serial prediction is time-consuming, especially for deeper neural network layers with high dimensionality (i.e., large $p ^ { l } )$

In contrast, parallel inference brought by knowledge integration is markedly more efficient. On one hand, the computational complexity in Eq. (5) reduces to $\mathcal { O } ( r _ { t } ^ { l } p ^ { l } n +$ $p ^ { l + 1 } \dot { r } _ { t } ^ { l } n ) ~ = ~ \mathcal { O } ( r _ { t } ^ { l } { \dot { n } } p ^ { l } )$ , which is significantly lower than that of the serial inference $( \mathcal { O } ( ( r _ { t } ^ { l } + n ) p ^ { l ^ { 2 } } ) )$ . On the other hand, Eq. (6) involves concatenated matrix multiplication, benefiting more from GPU acceleration.

Algorithm 1: Incremental learning of $C o ^ { 2 } I n$   
Input: Datasets $\{ \mathcal { D } _ { 1 } , \hdots , \mathcal { D } _ { T } \}$   
Output: Knowledge representations $A ^ { * } ;$ ; Pattern   
basis B   
1 Initialize $\mathcal { A } ^ { \ast } = \{ A ^ { \ast l } = \varnothing \} _ { l = 1 } ^ { L } , \mathcal { B } = \{ B ^ { l } = \varnothing \} _ { l = 1 } ^ { L } ;$   
2 for $t = 1 , \dots , T$ do   
3 Initialize $\overline { { \mathcal { A } _ { t } } } = \{ A _ { t } ^ { l } = \varnothing \} _ { l = 1 } ^ { L }$   
4 // knowledge encoding   
5 ${ \mathcal W } _ { t } \longleftarrow \arg \operatorname* { m i n } _ { L _ { c e } ( \mathcal W , \mathcal D _ { t } ) + \lambda L _ { r e g } ( \mathcal W ) }$   
W   
6 $/ /$ knowledge consolidation   
7 for $l = 1 , \ldots , L$ do   
8 $\overline { { U _ { t } ^ { l } \Sigma _ { t } ^ { l } { V _ { t } ^ { l } } ^ { \top } } } \longleftarrow \mathrm { S V D } \left( Z _ { t } ^ { l } - B ^ { l } B ^ { l } { } ^ { \top } Z _ { t } ^ { l } \right)$   
9 $B ^ { l } \gets [ B ^ { l } , U _ { t } ^ { l } ]$   
10 $A _ { t } ^ { l } \longleftarrow B ^ { l ^ { \top } } W _ { t } ^ { l }$   
11 A<sub>t</sub> ←− arg min $L _ { c e } ( B \mathcal { A } _ { t } , \mathcal { D } _ { t } )$   
$\sim { \cal A } _ { t }$   
12 $/ /$ knowledge integration   
13 for $l = 1 , \ldots , L$ do   
14 L $\overline { { { A ^ { * l } \longleftarrow \left[ A ^ { * l } , A _ { t } ^ { l } \right] } } }$   
15 return A<sup>∗</sup>, B

Upon knowledge integration finish, $C o ^ { 2 }$ In completes the learning of task t. The entire learning process of $C o ^ { 2 } I n$ is summarized in Algorithm 1.

## 5.4 Extension to Convolutional Neural Networks

In previous sections, the discussion revolved around the application of Co<sup>2</sup>In within fully connected neural networks. This section presents a straightforward strategy for extending our method to convolutional neural networks. The fundamental principle behind this extension involves converting network layers between convolution form and matrix-multiplication form.

Knowledge consolidation with matrix-multiplication form. Convolution is constituted by pixel-wise multiplications between the convolutional kernel and the image patches. This inherent linearity inspires us to reshape both the kernel and the patches into matrix form. As illustrated in Fig. $^ { 4 , }$ the l-th convolution layer with a kernel size of $c ^ { l + 1 } \times c ^ { l } \times k ^ { l } \times k ^ { l }$ is transformed into a matrix of dimensions $c ^ { l + 1 } \times ( c ^ { l } \times k ^ { l } \times k ^ { l } )$ . This transformation enables the seamless application of knowledge consolidation methods to convolutional layers without any modification.

Knowledge integration with convolution form. After knowledge consolidation, the convolution kernel of l-th layer is factorized into two matrices: a pattern basis matrix with the size of $c ^ { l + 1 } \times r _ { t } ^ { l }$ and a knowledge representation matrix with the size of $r _ { t } ^ { l } \times ( c ^ { l } \times k ^ { l } \times k ^ { l } )$ . An intriguing aspect of this decomposition is that the pattern basis can be reinterpreted as a $1 \times 1$ convolution kernel, while the knowledge representation can be converted into a convolution kernel with dimensions $\boldsymbol { r } _ { t } ^ { l } \times \boldsymbol { c } ^ { l } \times \boldsymbol { k } ^ { l } \times \boldsymbol { k } ^ { l }$ , as visualized in Fig. 2 (c). In line with the knowledge integration process on fully connected layers, all kernel-formed knowledge representations are encapsulated within a grouped convolutional layer to facilitate efficient inference, as shown in Fig. 5.

![](images/318d2d9c0f4b28b0ab70e2cf15ed13578bbf3fd5c480a349e2219770fbee26ff.jpg)  
Fig. 5. Diagram of the l-th convolutional layer of long-term memory after knowledge integration, which turns into a group convolutional layer (corresponding to knowledge representations) and a $1 \times 1$ convolutional layer (corresponding to pattern basis). Between them, we supplement an additional zero-channel padding operation to match the dimension.

![](images/9b310866d81f7374dbaa1cf83495c245aa84048a2ac8f78d20df56caef01fbb1.jpg)

![](images/42d2586e92737fb23f45d31e05b93507b19b9ae1e8e44037f9b3d6ce5d09c517.jpg)  
Fig. 6. Contour map for logits corresponding to a task-1 class (cluster with the red star) output by a piece-wise linear or quadratic classifier trained with task-1 samples. The linear classifier accurately identifies task-1 samples (a). Due to linear extrapolation, it produces relatively high logits for task-2 samples (b), which could result in misclassification. In contrast, the logits output by the quadratic classifier can not only contribute reasonable classification within task-1 samples (c) but is also suitable for task-2 samples (d).

## 5.5 Quadratic Classifier

$C o ^ { 2 } .$ In remains an inherent challenge in Class-IL: the lack of cross-task discriminability during inference because of the unavailability of task identification. Recall that Eq. (7) derives the classification result by comparing the prediction logits obtained from all t tasks, each functions as an autonomous subnetwork. However, as each subnetwork undergoes an independent training process, their output logits may not cross-task discriminative, potentially influencing the Class-IL performance of $C o ^ { 2 } I n$

We conducted an exploratory experiment through a toy IL dataset to assess the cross-task discriminability of outputs from these independently trained subnetworks. The toy dataset consists of six classes, each sampled from a Gaussian distribution. As depicted in Fig. 6, the first four classes (colored purple) belong to task 1, while the remaining two classes (colored green) come from task 2. A two-layer fully connected model was trained exclusively on task-1 data. Contour maps were subsequently generated to analyze the behavior of logits across both tasks. Fig. 6 (a) demonstrates that the logits enable accurate classification within task $^ { 1 , }$ evidenced by higher values for corresponding class samples and lower values for other samples within the same task. Conversely, as depicted in Fig. 6 (b), the logits for task-2 samples become disproportionately large, leading to misclassification as task-1 classes. This behavior highlights the infeasibility of directly concatenating these misleading logits described in Eq. (7).

One potential explanation for the above issue can be attributed to the “push” factor [63] inherent in the crossentropy loss. To be more specific, the cross-entropy loss can be decomposed into two components:

$$
\begin{array} { r l } & { L _ { c e } ( \boldsymbol { W } , \mathcal { D } _ { t } ) = \displaystyle \sum _ { ( \boldsymbol { x } _ { k } , \boldsymbol { y } _ { k } ) \in \mathcal { D } _ { t } } - \log \frac { \exp ( \boldsymbol { W } _ { \boldsymbol { y } _ { k } } ^ { L } \top \boldsymbol { x } _ { k } ^ { L - 1 } ) } { \sum _ { i } \exp ( \boldsymbol { W } _ { i } ^ { L } \top \boldsymbol { x } _ { k } ^ { L - 1 } ) } } \\ & { = \displaystyle \sum _ { ( \boldsymbol { x } _ { k } , \boldsymbol { y } _ { k } ) \in \mathcal { D } _ { t } } \underbrace { - \boldsymbol { W } _ { \boldsymbol { y } _ { k } } ^ { L } \top \boldsymbol { x } _ { k } ^ { L - 1 } } _ { \mathrm { p u l l t e r m } } + \underbrace { \log \sum _ { i } \exp ( \boldsymbol { W } _ { i } ^ { L } \top \boldsymbol { x } _ { k } ^ { L - 1 } ) } _ { \mathrm { p u s h t e r m } } , } \end{array}\tag{8}
$$

where $\boldsymbol { W } _ { i } ^ { L }$ denotes the parameter of the final layer corresponding to class i, and $\pmb { x } _ { k } ^ { L - 1 }$ represents the feature of $\scriptstyle { \mathbf { { \mathit { x } } } } _ { k }$ extracted by the penultimate layer. In the last line of Eq. (8), the first term behaves as a “pull” factor that attracts the feature towards the classifier prototype $W _ { y _ { k } } ^ { L }$ , while the second term serves as the “push” factor that repels the features from the prototypes of other classes $W _ { i } ^ { L }$ . However, the “push” factor is unbounded and tends to dominate the overall loss in data imbalance scenarios [64], such as incremental learning. Figure 6 (b) evidences how the pronounced effect of the “push” factor can incur misclassifications.

To address this issue, inspired by Deep-SVDD [65], we introduce a quadratic classifier for $C o ^ { 2 } \bar { I n }$ designed to emphasize the “pull” factor in the cross-entropy loss and enhance recognition of out-of-distribution (i.e., other tasks) classes. The quadratic classifier is defined as follows:

$$
\begin{array} { r } { Z ^ { L } = ( { W ^ { L } } ^ { \top } X ^ { L - 1 } - b ) ^ { 2 } , } \end{array}
$$

where b is a learnable centroid, $( . ) ^ { 2 }$ is element-wise square operation. Combining the quadratic classifier with the crossentropy loss yields:

$$
\begin{array} { r l r } {  { L _ { c e } ( \mathcal { W } , \mathcal { D } _ { t } ) = \sum _ { k } - \log \frac { \exp ( - ( W _ { y _ { k } } ^ { L } \tau _ { k } ^ { L - 1 } - b _ { y _ { k } } ) ^ { 2 } ) } { \sum _ { i } \exp ( - ( W _ { i } ^ { L } \tau _ { x } ^ { L - 1 } - b _ { i } ) ^ { 2 } ) } } } \\ & { = \sum _ { ( \boldsymbol { x } _ { k } , y _ { k } ) \in \mathcal { D } _ { t } } \underbrace { ( W _ { y _ { k } } ^ { L } \tau _ { x } ^ { L - 1 } - b _ { y _ { k } } ) ^ { 2 } } _ { \mathrm { p u l t e r m } } } \\ & { } & { + \underbrace { \log \sum _ { i } \exp ( - ( W _ { i } ^ { L } \tau _ { x } ^ { L - 1 } - b _ { i } ) ^ { 2 } ) } _ { \mathrm { p u s h t e r m } } , } \end{array}\tag{9}
$$

where $\mathbf { } _ { b _ { i } }$ denotes the centroid parameter corresponding to class i. In Eq. (9), the $\mathrm { \Delta ^ { \prime \prime } p u l l ^ { \prime \prime } }$ term aims to quadratically draw features into the vicinity of the class centroid $b _ { y _ { k } } .$ Conversely, the “push” term diminishes to zero as features diverge from other class centroid $b _ { i }$ . Therefore, by bounding the “push” factor, the quadratic classifier minimizes its impact on the overall loss. This aspect is empirically substantiated in Fig. 6 (c) and (d), where the quadratic classifier establishes a much tighter decision boundary compared to the linear classifier, ensuring cross-task discriminability.

The quadratic classifier is not exclusive to $C o ^ { 2 } \mathrm { { I n } }$ . We identify two key factors that influence its effectiveness. First, the stability of the method: the compact decision boundaries of the quadratic classifier make it highly sensitive to variations in the extracted features, as the logits change quadratically in response to these variations. Second, the model’s ability to learn new features, such as the ability to expand the basis in $C o ^ { 2 } \mathrm { { I n } }$ . The new features serve as a clear signal for distinguishing between new and old task classes. Methods that possess both of these characteristics, such as GPM, can benefit from combining with the quadratic classifier to strengthen their Class-IL capability, whereas other methods may not achieve the same improvement. For more details, please refer to Section 6.11.

## 6 EXPERIMENT

## 6.1 Settings

Datasets. We choose three datasets during the evaluation: CIFAR-10 [66], CIFAR-100 [66], and Tiny-ImageNet [67]. CIFAR-10 includes ten classes of $3 2 \times 3 2$ color images. We divide it into five incremental batches, with each batch containing samples of 2 classes. CIFAR-100 consists of 60000 $3 2 \times 3 2$ color images that fall into 100 classes. We split it into ten incremental batches in most of our experiments. Tiny-ImageNet comprises 200 categories of 64 × 64 color images. The 200 classes are divided into ten tasks equally, each including samples from 20 classes.

Comparison baselines. We select several classic and latest IL methods for comparison, including Gradient Episodic Memory (GEM) [10], Dark Experience Replay (DER and DER++) [4], Contrastive Continual Learning (Co2L) [25], Hierarchical Correlations Replay [24], NPCL [68], Learning without Forgetting (LwF) [33], Synaptic Intelligence (SI) [39], Orthogonal Weights Modification (OWM) [69], DeepInversion (DI) [29], ILCOC [47], Gradient Projection Memory (GPM) [5], Adam-NSCL [6], Prototype Augmentation and

![](images/6fb90d9b0f87ca5c1af834a1b7b1d6216662a79060626b91782ff0ac76d4d7dc.jpg)

![](images/811142d51b7336b8ca33a03b81c90bd369190f20b4c71fb0e08b68add49703f0.jpg)  
Fig. 7. Memory usage (MB) on CIFAR-100 and Tiny-ImageNet.

Self-Supervision (PASS) [70], Efficient Feature Transformations (EFT) [51], Always Be Dreaming (ABD) [14], Filter Atom Swapping (FAS) [71], Winning SubNetworks (WSN) [72], DCPOC [31], VDFD [36]. We also introduce a fundamental method named Joint, which is trained on all task data jointly. Joint does not adhere to the IL training setup, which is generally seen as the upper-bound performance. In addition, we report the results of the conference version of our method, referred to as BMKP [23]. BMKP can be thought of as a simplified $C o ^ { 2 } I n$ without the knowledge integration and the quadratic classifier.

Performance metrics. We follow [4], [10] and employ classification accuracy (ACC) on all learned tasks after IL as the primary metric for performance evaluation. To eliminate the stochasticity inherent in neural network training, we report the average performance of five runs with random seeds. We also provide memory usage in megabytes (MB) for certain methods that require additional storage beyond the network parameters. The additional storage includes exemplars for DER, gradient basis for GPM and Adam-NSCL, and long-term memory for BMKP and $C o ^ { 2 } I n$

Implementation details. To ensure a fair performance comparison, we evaluated the performance of all methods under similar memory constraints. For the projection-based methods (GPM, Adam-NSCL, BMKP, and ${ \mathsf { \bar { C } } } o ^ { 2 } I n )$ in which the memory usage depends on the width of the network, we follow [5] and employ a narrower version of ResNet18 [73]. In contrast, all the other baselines employ the standard ResNet18 as the network backbone. For rehearsal-based methods (GEM, DER, DER++, Co2L, HCR) that necessitate additional memory to preserve exemplars, we equip them with a buffer capable of storing up to 500 samples.

Hyper-parameters. All hyper-parameters are searched through a validation set obtained by sampling ten percent samples from the training set. We optimize $C o ^ { 2 } \bar { I _ { n } }$ with stochastic gradient descent (SGD) optimizer over 100 iterative epochs. We tailor the batch size to 32, 16, and 128 for CIFAR-10, CIFAR-100, and Tiny-ImageNet. The learning rates are configured at 2e-3, 2e-2, and 2e-2 for the three datasets, respectively. The balance weight λ is tuned within the range of {0, 0.1, 1, 10, 100}, with the optimal values selected as 0.1, 1, and 10 for CIFAR-10, CIFAR-100, and Tiny-ImageNet. Additionally, we adjust the retained singular-value ratio (Cf. Eq. (3)) across the range of {95%, 96%, 97%, 98%, 99%}, ultimately settling on values of 97%, 97%, and 98% for the three datasets, respectively. For more details, please refer to the code repository <sup>2</sup>.

TABLE 2  
Average accuracy (%) and the standard deviation over five runs with random seeds. Joint is the model jointly trained with data from all tasks. (-) means that the result was unavailable, due to the intractable training time by our implementation (GEM) or unsuitable setting (GPM and Adam-NSCL for Class-IL). (†) and (‡) imply the results are quoted from HCR and the origin paper, where the standard deviations is not provided. Bold and underlined denote the best and the second-best performance. Results with (\*) imply the leveraging of rotate class augmentation.
<table><tr><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Venue</td><td rowspan=1 colspan=1>Buffer</td><td rowspan=1 colspan=2>CIFAR-10Class-IL       Task-IL</td><td rowspan=1 colspan=2>CIFAR-100Class-IL       Task-IL</td><td rowspan=1 colspan=2>Tiny-ImageNetClass-IL       Task-IL</td></tr><tr><td rowspan=1 colspan=1>Joint</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>91.29</td><td rowspan=1 colspan=1>98.07</td><td rowspan=1 colspan=1>70.31</td><td rowspan=1 colspan=1>91.18</td><td rowspan=1 colspan=1>58.07</td><td rowspan=1 colspan=1>82.01</td></tr><tr><td rowspan=1 colspan=1>GEM [10]DER [4]DER++ [4]Co2L† [25]HCR†[24]NPCL‡ [68]</td><td rowspan=1 colspan=1>NIPS2017NIPS2020NIPS2020ICCV2021KBS2022NIPS2023</td><td rowspan=1 colspan=1>500500500500500500</td><td rowspan=1 colspan=1>23.66±1.270.57±1.073.48±0.374.2674.6471.34±1.5</td><td rowspan=1 colspan=1>85.14±2.193.41±0.394.10±0.395.9096.01</td><td rowspan=1 colspan=1>11.76±0.734.24±1.436.52±2.030.4533.7137.43</td><td rowspan=1 colspan=1>62.80±2.773.26±1.374.86±1.179.6180.55</td><td rowspan=1 colspan=1>17.75±1.119.38±1.420.1222.0515.29±1.0</td><td rowspan=1 colspan=1>51.78±0.951.91±0.753.0454.73</td></tr><tr><td rowspan=1 colspan=1>LwF [33]SI [39]OWM [69]DI [29]ILCOC[47]GPM [5]Adam-NSCL [6]PASS [70]EFT [51]ABD [14]FAS [71]WSN [72]DCPOC [31]VDFD [36]</td><td rowspan=1 colspan=1>TPAMI2017ICML2017NMI2019CVPR2020CVPRW2021ICLR2021CVPR2021CVPR2021CVPR2021ICCV2021ICLR2022ICML2022PR2023TPAMI2023</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>19.59±0.119.47±0.247.50±0.524.14±2.238.71±4.253.12±0.3*55.12±1.957.64±0.343.36±2.819.14±0.144.30±0.659.90±1.8*</td><td rowspan=1 colspan=1>91.91±0.769.40±1.981.00±0.194.46±0.694.74±0.386.58±0.987.23±0.486.07±0.2*95.09±0.395.11±0.390.89±1.392.99±0.490.43±0.394.24±0.5*</td><td rowspan=1 colspan=1>10.05±0.67.42±0.527.63±0.511.19±0.322.19±1.631.80±0.7*35.99±1.033.30±0.325.79±0.78.79±0.225.80±0.438.38±0.9*</td><td rowspan=1 colspan=1>63.78±4.362.21±2.660.64±0.468.43±2.169.87±1.070.93±0.965.69±0.277.30±0.4*79.29±0.674.83±0.570.89±0.681.10±0.772.20±0.283.30±0.2*</td><td rowspan=1 colspan=1>9.36±0.16.77±0.115.30±0.38.19±0.215.78±0.428.48±0.6*26.89±0.615.80±0.424.29±0.37.68±0.319.75±0.126.21±0.7*</td><td rowspan=1 colspan=1>58.61±1.860.91±1.340.89±0.366.12±0.954.09±0.759.84±0.259.98±0.762.87±0.4*63.88±0.546.76±0.660.10±0.267.39±0.753.08±0.266.62±0.3*</td></tr><tr><td rowspan=1 colspan=1>BMKP (ours)Co2In (ours)</td><td rowspan=1 colspan=1>CVPR2023</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>29.70±1.056.21±5.0*</td><td rowspan=1 colspan=1>94.49±0.295.21±0.3*</td><td rowspan=1 colspan=1>16.89±0.637.54±0.9</td><td rowspan=1 colspan=1> $7 9 . 6 2 { \pm } 0 . 8 $ 84.21±1.0</td><td rowspan=1 colspan=1>8.91±0.231.23±0.5</td><td rowspan=1 colspan=1>70.36±0.268.75±0.7</td></tr></table>

TABLE 3

Ablation study of Co<sup>2</sup>In on CIFAR-100 and Tiny-ImageNet.
<table><tr><td>Methods</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td></td><td>Class-IL</td><td>Task-IL</td><td>Class-IL</td><td>Task-IL</td></tr><tr><td>(a)  $C o ^ { 2 } .$  In w/o representation compaction regularizer</td><td>34.75±0.5</td><td>83.04±0.5</td><td>27.05±2.0</td><td>65.00±2.2</td></tr><tr><td>(b)  $C o ^ { 2 }$  In w/o basis updating</td><td>6.88±5.9</td><td>36.48±17.7</td><td>2.90±1.0</td><td>27.23±3.6</td></tr><tr><td>(c)  $C o ^ { 2 } I n$  w/o retraining</td><td>36.12±0.7</td><td>81.78±3.0</td><td>28.91±0.7</td><td>65.68±1.9</td></tr><tr><td>(d) Co2In w/o quadratic classifier</td><td>16.90±0.5</td><td>79.65±0.4</td><td>8.81±0.1</td><td>69.31±0.5</td></tr><tr><td>Co2In</td><td>37.54±0.9</td><td>84.21±1.0</td><td>31.23±0.5</td><td>68.75±0.7</td></tr></table>

## 6.2 Comparison Result

This section provides a comparison between $C o ^ { 2 } I n$ and baseline methods. The performance is presented in Table 2 for both Task-IL and Class-IL settings. Additionally, Fig. 7 illustrates the memory usage of various methods.

For the small-scale CIFAR-10 dataset, $C o ^ { 2 } .$ In is competitive with the rotate class augmentation technique<sup>3</sup>, showcasing the best Task-IL performance and the third-best Class-IL performance among exemplar-free methods. Our empirical findings suggest that the rotate class augmentation is particularly efficacious in enhancing performance in settings with limited numbers of categories. Section 6.5 details an in-depth examination of this augmentation.

For CIFAR-100, $C o ^ { 2 } I n$ achieves the second-best Class-IL performance (37.54%) and the best Task-IL performance (84.21%), surpassing the second-best VDFD by 0.91%. For Tiny-ImageNet, under the Task-IL setting, the conference version BMKP obtains the best classification accuracy (70.36%), while $C o ^ { 2 } \mathrm { _ { \perp } }$ In delivers the second-best performance (68.75%). The difference in performance can be attributed to the distinct classifier employed: $C o ^ { 2 }$ In relies on quadratic classifiers, while BMKP utilizes linear classifiers. Additionally, under the Class-IL setting, $C o ^ { 2 } I n$ attains the best performance (31.23%) on Tiny-ImageNet, surpassing the secondbest method by 2.75%.

Furthermore, the competitive performance of $C o ^ { 2 } I n$ is achieved with lower memory utilization. As depicted in Fig. 7, $C o ^ { 2 } I n$ utilizes 25.53 MB of memory when applied to CIFAR-100, which is lower than that of the secondbest method by a margin of 1.18 MB. Similarly, on Tiny-ImageNet, $C o ^ { 2 } \bar { I n }$ consumes 22.89 MB of memory, outperforming the second-best BMKP with 2.16 MB. These results underscore the effectiveness of $C o ^ { 2 } I n$ on its economical use of resources without compromising performance across diverse datasets and learning paradigms.

In conclusion, the aforementioned results substantiate that $C o ^ { 2 } .$ In strikes a good balance between plasticity and stability in incremental learning, attributed to its brain-inspired bi-level memory framework. Additionally, by compacting the decision boundaries, the quadratic classifier effectively boosts the Class-IL performance of $C o ^ { 2 } I n$

## 6.3 Ablation Study

This section presents ablation studies on each component.

Effectiveness of representation compaction regularizer. We start by introducing variation (a), which is derived from

![](images/ebf16fbb8297d30de53a4c4dfed9d04d37d0f8d5b5cbb8e3121aff3070cb0e42.jpg)  
Fig. 8. Incremental learning results of $C o ^ { 2 }$ In with different λ (first row) and SVD ratio (second row) on CIFAR-100 and Tiny-ImageNet.

$C o ^ { 2 } I n$ by setting the hyperparameter λ to 0. As demonstrated in the first row of Table 3, variation (a) exhibits a degraded performance relative to the original $C o ^ { 2 } I n$ . This observation reveals that the representation compaction regularizer is able to enhance the performance of $\dot { C o } ^ { 2 } I n$

Effectiveness of basis updating. In variation (b), knowledge consolidation does not incorporate the basis updating step. Consequently, the knowledge representations of variation (b) are acquired with respect to the task-1 pattern basis. The second row of Table 3 shows a decline in performance for variation (b) across all setups. This phenomenon can be attributed to the fact that basis updating expands the range of CKS. Without this expansion, the expression of knowledge would be severely constrained, i.e., low plasticity.

Effectiveness of retraining. Based on $C \bar { o } ^ { 2 } I n ,$ variation (c) excludes the retraining step (Cf. Eq. (4)) after knowledge projection. The third row of Table 3 shows that variation (c) exhibits slightly lower performance than $C o ^ { 2 } I n$ . This phenomenon is expected since retraining serves as a finetuning process that refines the model performance.

Effectiveness of quadratic classifier. In the final variation (d), the inherent quadratic classifier of $C o ^ { 2 } I n$ is substituted with a conventional linear classifier. As can be seen, the Class-IL performance of variation (d) experiences a significant drop, but the Task-IL performance remains consistent. This finding suggests that variation (d) may have a bias toward certain tasks, highlighting the importance of the quadratic classifier for the Class-IL ability of $\dot { C o } ^ { 2 } I n$

## 6.4 Analysis of Hyper-Parameters

Sensitivity analysis of λ (Cf. Eq. (2)). We first conduct experiments to evaluate the sensitivity of λ by setting different values (0, 1, 10, 100, 1000), and further investigate the effectiveness of the representation compaction regularizer.

The first row in Fig. 8 depicts the experimental results. when $\lambda = 0 ( \mathrm { i . e . , }$ , the representation compaction regularizer is not applied), $C o ^ { 2 } I n$ consumes the most memory but yields relatively lower accuracy on both Task-IL and Class-IL. As the value of λ rises, the classification accuracy correspondingly increases, peaking at an optimal point before gradually declining. This phenomenon reveals that the representation compaction regularizer improves the performance of $C o ^ { 2 } I n ,$ which we attribute to facilitating the transfer of learned knowledge. Furthermore, the memory usage of $C o ^ { 2 } I$ n consistently diminishes as λ grows, signifying that the regularizer efficiently improves the memory efficiency of $C o ^ { 2 } I n$ by encouraging the model to represent new knowledge using previous-established patterns.

Sensitivity analysis of SVD ratio (Cf. Eq. (3)). Subsequently, we evaluate Co<sup>2</sup>In with different SVD ratios: 0.95, 0.96, 0.97, 0.98, 0.99. The experimental results are depicted in the second row of Fig. 8. As the SVD ratio increases, both accuracy and memory usage remain stable and gradually increase monotonically. This phenomenon is owing to the fact that a larger SVD ratio allows $C o ^ { 2 } I _ { 1 }$ n to preserve more basis during knowledge consolidation, thereby retaining more knowledge. We empirically determine an SVD ratio of 0.97 for CIFAR100 and 0.98 for Tiny-ImageNet to balance the performance and memory usage.

## 6.5 Analysis of Rotate Class Augmentation

This section delves into the impact of the rotate class augmentation (RCA) on the performance of $C o ^ { 2 } I n$ . As shown in Table 4, RCA significantly boosts the performance of $C o ^ { 2 } I n$ on CIFAR-10, where each task contains only two classes. Without RCA, the model struggles to generalize due to overfitting on narrow task-specific features. For instance, distinguishing between cars and ships may cause the model to over-rely on background color (e.g., blue water for ships), which does not transfer well to unseen classes. RCA mitigates this by enforcing feature learning through self-supervised rotation tasks, leading to improved generalization and better Class-IL performance.

TABLE 4  
Incremental learning result (ACC %) of $C o ^ { 2 }$ In with and without self-supervised technique on CIFAR-10. “RCA” indicates the rotate class augmentation technique.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Methods</td><td rowspan=1 colspan=1>Class-IL</td><td rowspan=1 colspan=1>Task-IL</td></tr><tr><td rowspan=4 colspan=1>CIFAR-10</td><td rowspan=3 colspan=1>OWM [69]PASS [70]EFT [51]ABD [14]FAS [71]DCPOC [31]</td><td rowspan=2 colspan=1>47.50±0.5</td><td rowspan=2 colspan=1> $8 1 . 0 0 { \pm } 0 . 1 \ $  $8 6 . 0 7 { \scriptstyle \pm 0 . 2 }$  $9 5 . 0 9 { \pm } 0 . 3 $  $9 5 . 1 1 { \pm } 0 . 3 $ </td></tr><tr><td rowspan=1 colspan=1>57.64±0.3</td></tr><tr><td rowspan=1 colspan=1>43.36±2.844.30±0.6</td><td rowspan=1 colspan=1>90.89±1.3 $9 0 . 4 3 { \pm } 0 . 3 \ \qquad $ </td></tr><tr><td rowspan=1 colspan=1>BMKP (ours)Co2In (ours) $C o ^ { 2 } I n \ \mathrm { { w / \ R C A / o u r s } ) }$ </td><td rowspan=1 colspan=1>29.70±1.038.15±2.956.21±5.0</td><td rowspan=1 colspan=1> $9 4 . 4 9 { \pm } 0 . 2 $  $9 2 . 7 4 \pm 0 . 7 $  ${ \bf 9 5 . 2 1 } { \pm } { \bf 0 . 3 }$ </td></tr></table>

TABLE 5

Incremental learning result (ACC %) of ${ \mathsf { C o } } ^ { 2 }$ In with and without rotate class augmentation (RCA) with different numbers of tasks.
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Partition</td><td rowspan=1 colspan=1>CIFAR-100Class-IL (%)  Task-IL (%)</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>10-Split</td><td rowspan=1 colspan=1>37.54        84.21</td></tr><tr><td rowspan=1 colspan=1>Co2In w/ RCA</td><td rowspan=1 colspan=1>10-Split</td><td rowspan=1 colspan=1>29.58        84.18</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>25-Split</td><td rowspan=1 colspan=1>21.75        91.45</td></tr><tr><td rowspan=1 colspan=1>Co²In w/RCA</td><td rowspan=1 colspan=1>25-Split</td><td rowspan=1 colspan=1>27.37        91.03</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>50-Split</td><td rowspan=1 colspan=1>5.02         91.53</td></tr><tr><td rowspan=1 colspan=1>Co2In w/RCA</td><td rowspan=1 colspan=1>50-Split</td><td rowspan=1 colspan=1>21.73        94.56</td></tr><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Partition</td><td rowspan=1 colspan=1>Tiny-ImageNetClass-IL (%)  Task-IL (%)</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>10-Split</td><td rowspan=1 colspan=1>31.23        68.75</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n w / R C A$ </td><td rowspan=1 colspan=1>10-Split</td><td rowspan=1 colspan=1>24.07        63.25</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>25-Split</td><td rowspan=1 colspan=1>20.00        78.78</td></tr><tr><td rowspan=1 colspan=1>Co2In w/RCA</td><td rowspan=1 colspan=1>25-Split</td><td rowspan=1 colspan=1>22.00        76.98</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>50-Split</td><td rowspan=1 colspan=1>7.03         83.77</td></tr><tr><td rowspan=1 colspan=1>Co2In w/RCA</td><td rowspan=1 colspan=1>50-Split</td><td rowspan=1 colspan=1>14.40        85.38</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n$ </td><td rowspan=1 colspan=1>100-Split</td><td rowspan=1 colspan=1>3.74         92.01</td></tr><tr><td rowspan=1 colspan=1> $C o ^ { 2 } I n \ w / R C A$ </td><td rowspan=1 colspan=1>100-Split</td><td rowspan=1 colspan=1>8.41         92.17</td></tr></table>

To further investigate the effect of RCA, we report results on CIFAR-100 and Tiny-ImageNet with different task splits in Table 5. These results reveal that RCA is particularly effective when the number of classes per task is small. For instance, on CIFAR-100 with 50 tasks $( \mathrm { i } . \mathrm { e } . , 2$ classes per task), RCA boosts Class-IL accuracy from 5.02% to 21.73%. Similarly, on Tiny-ImageNet with 50 tasks, Class-IL improves from 7.03% to 14.40%, and with 100 tasks (2 classes per task), it improves from 3.74% to 8.41%. This effect diminishes when tasks already contain sufficient class diversity, such as 10-split CIFAR-100 or 10-split Tiny-ImageNet. Therefore, we recommend applying RCA alongside our $C o ^ { 2 } I n$ in scenarios where each task involves a limited number of classes.

## 6.6 Analysis of Memory Efficiency

This section analyzes the performance of models under different memory usage. For projection-based methods $( \mathrm { e . g . , }$ GPM, Adam-NSCL, BMKP, and $C o ^ { 2 } I n )$ , memory usage is controlled by scaling the width of the network backbone. For the rehearsal-based method (DER++), we evaluate its performance using both the standard and narrower versions of ResNet18 across different numbers of exemplars (200, 500, 1000, 2000). Figure 9 illustrates the Task-IL results on CIFAR-100 and Tiny-ImageNet. As can be seen, the performance of DER++ is sensitive to the number of exemplars. As the network grows larger, the effect of the exemplar number becomes more significant. However, as memory is limited, DER++ shows a substantial performance drop. In contrast, $C o ^ { 2 } \mathrm { j }$ In performs comparable memory efficiency to BMKP. Both methods achieve higher accuracy than others under the same memory budget. These findings demonstrate the effectiveness of $\dot { C o } ^ { 2 } I _ { \ l }$ n in memory-sensitive scenarios.

![](images/a5683b786f71a2b9b4a30d4a505fb3502cdab52e95f6dc7280be93cb8c049267.jpg)  
Fig. 9. Memory usage (MB) versus Task-IL performance (%) on CIFAR-100 and Tiny-ImageNet.

TABLE 6  
FLOPs for different methods on CIFAR-100 and Tiny-ImageNet.
<table><tr><td>Method</td><td>CIFAR-100 Class-IL Task-IL | Class-IL Task-IL</td><td>Tiny-ImageNet</td></tr><tr><td>LWF [33]</td><td> $\lvert 5 . 5 7 \times 1 0 ^ { 8 } 5 . 5 7 \times 1 0 ^ { 8 }$ </td><td> $\left| 2 . 2 2 \times 1 0 ^ { 9 } 2 . 2 2 \times 1 0 ^ { 9 } \right.$ </td></tr><tr><td>DER++ [4]</td><td> $5 . 5 7 \times 1 0 ^ { 8 } 5 . 5 7 \times 1 0 ^ { 8 }$ </td><td> $2 . 2 2 \times 1 0 ^ { 9 } 2 . 2 2 \times 1 0 ^ { 9 }$ </td></tr><tr><td> $C o ^ { 2 } I n \mathrm { ~ w ~ } / \mathrm { ~ o ~ }$  knowledge integration</td><td> $1 . 4 5 \times 1 0 ^ { 9 } 1 . 4 5 \times 1 0 ^ { 8 }$ </td><td>3.23×1093.23×108</td></tr><tr><td></td><td></td><td></td></tr><tr><td> $C o ^ { 2 } I n$ </td><td> $\mathbf { 3 . 4 9 \times 1 0 ^ { 8 } ~ 3 . 4 9 \times 1 0 ^ { 7 } }$ </td><td> $1 . 4 3 \times 1 0 ^ { 9 } 1 . 4 3 \times 1 0 ^ { 8 }$ </td></tr></table>

## 6.7 Analysis of Computational Efficiency

The knowledge integration is introduced to enhances the computational efficiency of $C o ^ { 2 } I n$ . To validate this, we compare the FLOPs of various methods, including LWF, $\mathrm { D E R + + } ,$ , and $C o ^ { 2 } I n$ with and without knowledge integration. As shown in Table 6, knowledge integration notably reduces the computational cost in both CIFAR-100 and Tiny-ImageNet. Specifically, $C o ^ { 2 } I n$ with knowledge integration achieves the lowest FLOPs, outperforming LWF and DER++ in both Class-IL and Task-IL settings. For example, in the CIFAR-100 Class-IL setting, $C o ^ { 2 } I n$ reduces FLOPs by nearly 40% compared to other methods. These results demonstrate that $C o ^ { 2 } { I n }$ achieves state-of-the-art performance with significantly fewer computations. The reduction in computational cost is even more pronounced in the Task-IL setting, highlighting the computational efficiency of $C o ^ { 2 } I n$

## 6.8 Analysis of Classification Balance

We propose the quadratic classifier to boost the Class-IL capability of ${ \dot { C } } o ^ { 2 } I n .$ To validate its effectiveness, we analyze the confusion matrices under Class-IL for finetuning, BMKP, and $C o ^ { 2 }$ In on CIFAR-100 and Tiny-ImageNet datasets. The confusion matrix reveals the model’s correct predictions (represented by the diagonal entries) and misclassifications (shown by off-diagonal entries). As illustrated in Fig. 10 (a) and (d), fine-tuning inclines to classify samples into new-task classes. Additionally, Fig. 10 (b) and (e) show that BMKP displays a bias towards the first task, showing that linear classifiers struggle to accurately identify newtask samples, akin to the toy-dataset example provided in Fig. 6. In contrast, Fig. 10 (c) and (f) demonstrate that the highlights in the confusion matrices of $C o ^ { 2 } I n$ concentrate on diagonal entries, proving that the quadratic classifier enables $C o ^ { 2 }$ In with task-wise balance classification ability.

(c) Co<sup>2</sup>In  
(a) Fine-tuning  
(b) BMKP  
![](images/e29d7b323b53b46e1d1c2debf1b2d13ca7d1619d3fdbeb5a4fe7a414a2ca79a8.jpg)

CIFAR -100  
![](images/bc57c86bc85578ad124f4f716e6aeff0be00a0ce7fa8dcfe066e21a3f86d3f34.jpg)

![](images/83c246e06e360ac5e6c6e5f3f34262a5b638a310286258d89ae7c056cceea5ad.jpg)

![](images/da13a45b8bf80ad821082b866b072bf0675c04fbdc542ab3a5d300712d6d1ef4.jpg)  
(d) Fine-tuning

![](images/9b9950a52cd09014f25a8ee2b182cde5ae5e98456cbe7b8f078a03e55cae714f.jpg)  
(e) BMKP

![](images/b6f0eba4182880af2b5415a2879db1f69a22592d03953c41a3ee2faa3abc2ac0.jpg)  
(f) Co<sup>2</sup>In  
Fig. 10. Confusion matrix of fine-tuning, BMKP, and $C o ^ { 2 }$ In on CIFAR-100 and Tiny-ImageNet.

## 6.9 Analysis of Plasticity and Stability

This paper introduces a bi-level memory framework for $C o ^ { 2 } I n$ that decouples the roles of knowledge acquiring and remembering into distinct memory units, promoting both plasticity and stability. In this section, we analyze the plasticity and stability by inspecting the performance trend of each task during incremental learning. The performance curves of DER++ [4], GPM [5], and $C o ^ { 2 } .$ In are depicted in Fig. 11. As evidenced in the figure, DER++ exhibits strong plasticity as it achieves high accuracy immediately once each task is finished. However, as subsequent tasks are introduced, performance sharply declines, since the stored partial old-task exemplars may not support the original data distribution and fail to prevent forgetting, highlighting the low stability of DER++. On the contrary, GPM maintains more stable performance across all tasks owing to its gradient projection mechanism. However, the peak performance for each task is lower compared to ${ \dot { C o ^ { 2 } } } I n$ and DER++, revealing that the gradient direction restriction leads to low plasticity of GPM. As for $C o ^ { 2 } I n ,$ the bi-level memory framework guarantees high plasticity by the working memory and non-degrading performance for each task by the longterm memory.

TABLE 7  
Class Incremental learning results $( A C C \% )$ on CIFAR100 for various numbers of tasks (5, 10, 20). Results with “†” stem from ABD [14].
<table><tr><td>Methods</td><td>5-split CIFAR-100</td><td>10-split CIFAR-100</td><td>20-split CIFAR-100</td></tr><tr><td>DGR [28]†</td><td>14.4</td><td>8.1</td><td>4.1</td></tr><tr><td>LwF [33]†</td><td>17.0</td><td>9.2</td><td>4.7</td></tr><tr><td>DI [29]†</td><td>18.8</td><td>10.9</td><td>5.7</td></tr><tr><td> $\mathrm { \ A B { \dot { D } } \Gamma [ { \dot { 1 } } 4 ] ^ { \dagger } }$ </td><td>43.9</td><td>33.7</td><td>20.0</td></tr><tr><td> $\mathrm { P A S S } ^ { \bullet } [ 7 0 ^ { \bullet } ]$ </td><td>45.2</td><td>30.8</td><td>17.4</td></tr><tr><td> $\mathrm { C R H D } [ 7 4 ]$ </td><td>46.1</td><td>34.3</td><td>21.8</td></tr><tr><td> $\bar { \mathrm { D C P O C } } [ 3 \bar { 1 } ]$ </td><td>33.1</td><td>27.5</td><td>20.5</td></tr><tr><td> $C o ^ { 2 } .$  In (ours)</td><td>42.5</td><td>37.5</td><td>22.5</td></tr></table>

## 6.10 Study on Number of Tasks

We first assess the Class-IL performance of $C o ^ { 2 } \mathrm { 1 }$ In across different task sequence lengths. Following Always Be Dreaming (ABD) [14], we conduct experiments on CIFAR-100 using sequences of 5, 10, and 20 tasks. As depicted in Table $^ { 7 , }$ both PASS and CRHD demonstrate better performance than $C o ^ { 2 } I n$ for the 5-task setting. Nevertheless, in longer task sequence (10 and 20) settings, Co<sup>2</sup>In surpasses all the baselines. These findings prove that $C o ^ { 2 } I n$ is effective for longer sequences of Class-IL tasks.

Next, we delve into the Task-IL performance of $C o ^ { 2 } I n$ with a longer sequence of tasks. Following previous works [6], [36], we conduct experiments on 20-split CIFAR-100 and 25-split Tiny-ImageNet, where 20-split CIFAR-100 is constructed by splitting CIFAR-100 into 20 tasks, and 25-split Tiny-ImageNet is constructed by splitting Tiny-ImageNet into 25 tasks. Table 8 presents the comparison results. As can be seen, our method achieves the highest accuracy of 87.76% on 20-split CIFAR-100, outperforming the secondbest performance by a margin of 1.92%. As for 25-split Tiny-ImageNet, both BMKP and $C o ^ { 2 } I n$ overwhelm other methods, with BMKP achieving the highest accuracy of 80.01% and $C o ^ { 2 } .$ In closely following with an accuracy of 78.78%. These results highlight the Task-IL ability of $\dot { C o ^ { 2 } } .$ In for longer sequences of tasks.

![](images/b7037da3c3e7a2f4508fe298d4a8e83f246897163a22a58fcd3262d6636a60c1.jpg)  
Fig. 11. Diagram of ACC (%) of each task during incremental learning on CIFAR-100 and Tiny-ImageNet.

TABLE 8  
Task Incremental learning results (ACC %) on 20-split CIFAR-100 and 25-split Tiny-ImageNet. Results with (†) stem from VDFD [36].
<table><tr><td>Methods</td><td>20-split CIFAR-100</td><td>25-split Tiny-ImageNet</td></tr><tr><td>InstAParam† [75]</td><td>51.04</td><td>34.64</td></tr><tr><td>PackNet† [42] BLIP† [76]</td><td>67.50</td><td>52.93</td></tr><tr><td>GEM† [10]</td><td>68.17</td><td>47.63</td></tr><tr><td></td><td>68.89</td><td></td></tr><tr><td>A-GEM† [11] EWC† [38]</td><td>61.91</td><td>53.32</td></tr><tr><td>MAS† [41]</td><td>71.66</td><td>52.33</td></tr><tr><td>SI† [39]</td><td>63.84</td><td>47.96</td></tr><tr><td>IMM† [77]</td><td>59.76</td><td>45.27</td></tr><tr><td>ADNSf [16]</td><td>57.94</td><td>46.81</td></tr><tr><td>LWF† [33]</td><td>77.33</td><td>59.77</td></tr><tr><td>iCaRLf [13]</td><td>74.38</td><td>56.57</td></tr><tr><td>PASS† [70]</td><td>75.75</td><td>56.35</td></tr><tr><td></td><td>74.43</td><td>56.78</td></tr><tr><td>OWM† [69]</td><td>68.47</td><td>49.98</td></tr><tr><td>Adam-NSCL† [6]</td><td>75.95</td><td>58.28</td></tr><tr><td>VDFD† [36]</td><td>85.84</td><td>65.99</td></tr><tr><td>GPM [5]</td><td>77.55</td><td>68.62</td></tr><tr><td>BMKP (ours)</td><td>85.81</td><td>80.01</td></tr><tr><td>Co2In (ours)</td><td>87.76</td><td>78.78</td></tr></table>

## 6.11 Study the Adaptability of Quadratic Classifier

This section investigates the application of the quadratic classifier to other IL methods and explores its adaptability. We select LWF, DER++, GPM, and FAS for this analysis. As shown in Table 9, when applied to low-stability methods such as LWF and DER++, the training becomes highly unstable. This instability may arise from the quadratic classifier’s tendency to learn compact decision boundaries, which makes it sensitive to feature variations. Even minor fluctuations in the features can lead to significant changes in the classification logits. This characteristic, however, makes it only suitable for high-stability methods. For example, the quadratic classifier yields significant improvement for

TABLE 9  
Accuracy (%) of methods with linear and quadratic classifiers.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Classifier</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td>Class-IL</td><td>Task-IL</td><td>Class-IL</td><td>Task-IL</td></tr><tr><td>LWF [33]</td><td>Linear</td><td>10.05±0.663.78±4.3</td><td></td><td>9.36±0.1 58.61±1.8</td><td></td></tr><tr><td rowspan="2">LWF [33] DER++ [4]</td><td>Quadratic</td><td>FAIL</td><td>FAIL</td><td>FAIL</td><td>FAIL</td></tr><tr><td>Linear</td><td>7.42±0.5</td><td>62.21±2.6</td><td>6.77±0.1</td><td>60.91±1.3</td></tr><tr><td rowspan="2">DER++ [4] GPM [5]</td><td>Quadratic</td><td>FAIL</td><td>FAIL</td><td>FAIL</td><td>FAIL</td></tr><tr><td>Linear</td><td>8.19±0.1</td><td>70.93±0.9</td><td>7.04±0.1</td><td>59.84±0.2</td></tr><tr><td rowspan="2">GPM [5]</td><td>Quadratic</td><td>17.04±1.061.11±0.8</td><td></td><td>15.92±0.3</td><td>53.15±0.4</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">FAS [71] FAS [71]</td><td>Linear Quadratic</td><td>27.52±1.472.93±0.9</td><td>25.79±0.7 70.89±0.6</td><td>24.63±0.359.18±0.3</td><td>|24.29±0.3 60.10±0.2</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Co2In (ours) Co2In (ours)</td><td>Linear</td><td></td><td>16.90±0.579.65±0.4</td><td>8.81±0.1 69.31±0.5</td><td></td></tr><tr><td>Quadratic</td><td>37.54±0.984.21±1.0</td><td></td><td></td><td>31.23±0.568.75±0.7</td></tr></table>

TABLE 10

Class-IL performance (%) of Co<sup>2</sup>In using linear vs. quadratic classifiers on CIFAR-100. “Number of Expanded Basis” indicates the maximum number of new basis allowed to be added in each task.
<table><tr><td>Number of Expanded Basis | Linear Classifier Quadratic Classifier</td><td></td><td></td></tr><tr><td>0</td><td>8.98</td><td>8.50</td></tr><tr><td>1248</td><td>12.96</td><td>24.85</td></tr><tr><td></td><td>13.84</td><td>36.14</td></tr><tr><td></td><td>14.17</td><td>36.70</td></tr><tr><td></td><td>14.86</td><td>37.10</td></tr><tr><td>No Restriction</td><td>16.90</td><td>37.54</td></tr></table>

TABLE 11  
Comparison of FLOPs, memory usage, and performance (Class-IL and Task-IL accuracy) on CIFAR-100 and Tiny-ImageNet.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Backbone</td><td colspan="4">CIFAR-100 Class-IL</td><td rowspan="3">Memory (MB)</td><td colspan="4">Tiny-ImageNet Class-IL</td><td rowspan="3">Memory (MB)</td></tr><tr><td>Accuracy (%) Flops</td><td></td><td>Task-IL Accuracy (%)</td><td>Flops</td><td>Accuracy (%)1</td><td>Flops</td><td>Task-IL Accuracy (%) Flops</td><td></td></tr><tr><td>LWF [33]</td><td>|standard ResNet18</td><td>10.05</td><td> $5 . 5 7 \times 1 0 ^ { 8 }$ </td><td>63.78</td><td> $5 . 5 7 \times 1 0 ^ { 8 }$ </td><td>42.50</td><td>9.36</td><td> $2 . 2 2 \times 1 0 ^ { 9 }$ </td><td>58.61</td><td> $2 . 2 2 \times 1 0 ^ { 9 }$ </td><td>42.50</td></tr><tr><td></td><td>DER++ [4]|standard ResNet18</td><td>36.52</td><td> $5 . 5 7 \times 1 0 ^ { 8 }$ </td><td>74.86</td><td> $5 . 5 7 \times 1 0 ^ { 8 }$ </td><td>48.35</td><td>19.38</td><td> $2 . 2 2 \times 1 0 ^ { 9 }$ </td><td>51.91</td><td> $2 . 2 2 \times 1 0 ^ { 9 }$ </td><td>65.94</td></tr><tr><td> $C o _ { \circ } ^ { 2 } I n$ </td><td>standard ResNet18</td><td>39.25</td><td> $3 . 2 4 \times 1 0 ^ { 9 }$ </td><td>84.78</td><td> $3 . 2 4 \times 1 0 _ { - } ^ { 9 }$ </td><td>309.32</td><td>33.25</td><td> $1 . 1 9 \times 1 0 ^ { 1 0 }$ </td><td></td><td> $1 . 1 9 \times 1 0 _ { _ { - } } ^ { 9 }$ </td><td></td></tr><tr><td> $C o ^ { 2 } I n$ </td><td>[narrower ResNet18</td><td>37.54</td><td> $3 . 4 9 \times 1 0 ^ { 8 }$ </td><td>84.21</td><td> $3 . 4 9 \times 1 0 ^ { 7 }$ </td><td>25.53</td><td>31.23</td><td> ${ \bf 1 . 4 3 \times 1 0 ^ { 9 } }$ </td><td>70.08 68.75</td><td> $\mathbf { 1 . 4 3 \times 1 0 ^ { 8 } }$ </td><td>270.81 23.74</td></tr></table>

TABLE 12  
Comparison with different numbers of non-decomposed layers.
<table><tr><td>一</td><td colspan="5">CIFAR-100</td></tr><tr><td rowspan="2">Non-decomp. layers</td><td colspan="2">Class-IL</td><td colspan="2">Task-IL</td><td rowspan="2">Memory (MB)</td></tr><tr><td>Accuracy (%)</td><td>Flops</td><td>Accuracy (%)</td><td>Flops</td></tr><tr><td>All decomposed</td><td>37.54</td><td> $3 . 4 9 \times 1 0 ^ { 8 }$ </td><td>84.21</td><td> $3 . 4 9 \times 1 0 ^ { 7 }$ </td><td>25.53</td></tr><tr><td>First 5</td><td>35.87</td><td> $2 . 6 2 \times 1 0 ^ { 8 }$ </td><td>83.37</td><td> $4 . 3 4 \times 1 0 ^ { 7 }$ </td><td>25.46</td></tr><tr><td>First 10</td><td>35.60</td><td> $2 . 1 3 \times 1 0 ^ { 8 }$ </td><td>81.09</td><td> $5 . 0 3 \times 1 0 ^ { 7 }$ </td><td>23.95</td></tr><tr><td>First 15</td><td>31.46</td><td> $1 . 4 \bar { 5 } \times 1 0 ^ { 8 }$ </td><td>75.33</td><td> $5 . 5 3 \times 1 0 ^ { 7 }$ </td><td>21.19</td></tr><tr><td>一</td><td colspan="5">Tiny-ImageNet</td></tr><tr><td rowspan="2">Non-decomp. layers</td><td colspan="2">Class-IL</td><td colspan="2">Task-IL</td><td rowspan="2">Memory (MB)</td></tr><tr><td>Accuracy (%)</td><td>Flops</td><td>Accuracy (%)</td><td>Flops</td></tr><tr><td>All decomposed</td><td>31.23</td><td> $1 . 4 3 \times 1 0 ^ { 9 }$ </td><td>68.75</td><td> $1 . 4 3 \times 1 0 ^ { 8 }$ </td><td>23.74</td></tr><tr><td>First 5</td><td>30.19</td><td> $6 . 0 6 \times 1 0 ^ { 8 }$ </td><td>67.11</td><td> $1 . 6 6 \times 1 0 ^ { 8 }$ </td><td>23.14</td></tr><tr><td>First 10</td><td>28.94</td><td> $5 . 9 3 \times 1 0 ^ { 8 }$ </td><td>65.82</td><td> $1 . 9 5 \times 1 0 ^ { 8 }$ </td><td>22.74</td></tr><tr><td>First 15</td><td>26.56</td><td> $4 . 7 2 \times 1 0 ^ { 8 }$ </td><td>63.23</td><td> $2 . 2 4 \times 1 0 ^ { 8 }$ </td><td>22.45</td></tr></table>

$C o ^ { 2 } I n ,$ with Class-IL performance increasing by 20.64% on CIFAR-100 and 22.42% on Tiny-ImageNet, respectively.

However, the performance gain for FAS is relatively modest. We attribute this discrepancy primarily to the constraint that its shared basis remains fixed after the first task, whereas $C \mathrm { o } ^ { 2 }$ In dynamically expands its basis throughout incremental learning. To validate this, we conducted experiments restricting basis expansion in $\mathrm { { C o ^ { 2 } I n . } }$ As shown in Table 10, without basis expansion, the quadratic classifier yields no improvement and may even degrade performance. As the number of expanded basis increases, the benefits of the quadratic classifier become more significant. This behavior aligns with GPM, where the projection mechanism encourages learning new, orthogonal weights, akin to basis expansion. We believe this dependence on basis expansion is intuitive: new basis captures task-specific features that help distinguish between old and new tasks.

## 6.12 Results of Co<sup>2</sup>In on Standard ResNet18

In the experiments, we selected a narrower version of ResNet18 as the backbone for $C o ^ { 2 } I n$ . This section analyzes its performance using the standard ResNet18 architecture. Table 11 presents the results of $C o ^ { 2 } ]$ n on CIFAR-100 and Tiny-ImageNet. On the standard ResNet18, $C o ^ { 2 } I n$ achieves the highest performance but incurs greater memory and computational costs. However, it also results in higher memory and computational overhead. We recommend using a narrower ResNet18 for $C o ^ { 2 } I n ,$ as it strikes a favorable trade-off between accuracy and computational/memory overhead.

## 6.13 Study on Number of Non-Decomposed Layers

This section investigates the overhead introduced by the bilevel memory framework. To this end, we freeze the first 5, 10, and 15 layers closest to the input after learning the first task and apply the knowledge consolidation and integration only for the remaining layers.

TABLE 13  
Comparison of pretrained model-based IL methods.
<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="3">Tiny-ImageNet</td></tr><tr><td>Class-IL Task-IL</td><td>Memory</td><td>Class-IL</td><td>Task-IL</td><td>Memory</td></tr><tr><td>ACMap [59]</td><td>89.91% 98.35%</td><td>63.28 MB</td><td>86.01%</td><td>96.12%</td><td>63.28 MB</td></tr><tr><td>InfoLoRÅ [60]</td><td>85.83%</td><td>98.62% 63.28 MB</td><td>84.62%</td><td>93.58%</td><td>63.28 MB</td></tr><tr><td>Naive LoRA</td><td>83.44%</td><td>98.03% 63.28 MB</td><td>79.30%</td><td>91.42%</td><td>63.28 MB</td></tr><tr><td>Co2In (ours)</td><td>85.96%</td><td>98.74% 63.98 MB</td><td>84.38%</td><td>96.72%</td><td>60.38 MB</td></tr></table>

The results in Table 12 reveal a clear trade-off between resource efficiency and model performance. Freezing more layers significantly reduces memory usage but also decreases performance. For instance, on CIFAR-100, freezing 15 layers reduced memory usage by approximately 17% and FLOPs by nearly 60%. However, a simultaneous performance decline is observed: Class-IL accuracy decreases by 6.08%, while Task-IL accuracy drops by 8.88%. Similar trends are observed on Tiny-ImageNet, where freezing 15 layers led to a 4.67% reduction in Class-IL accuracy.

We also observe an interesting yet reasonable phenomenon: as the number of non-decomposed layers decreases, FLOPs for Class-IL increase, while those for Task-IL decrease. This is because Class-IL requires all extended parameters to be processed in parallel, which results in higher FLOPs with more decomposed layers. In contrast, Task-IL only activates certain parts of the parameters, so a greater number of decomposed layers benefits from the accelerated knowledge integration, i.e., the reduction in FLOPs.

## 7 EXTENSION TO PRETRAINED MODEL-BASED IN-CREMENTAL LEARNING

Although this paper focuses on training from scratch, the proposed $C o ^ { 2 } \bar { I n }$ can be seamlessly extended to pretrained model-based incremental learning. In this setting, the pretrained model provides a strong and stable feature prior. Therefore, we freeze the pretrained backbone and learn a residual branch for each adaptable layer. For the l-th layer at task $t ,$ the effective weight is formulated as

$$
\boldsymbol { W } _ { t } ^ { l } = \boldsymbol { W } _ { p r e } ^ { l } + \Delta \boldsymbol { W } _ { t } ^ { l } ,
$$

where $\boldsymbol { W _ { p r e } ^ { l } }$ denotes the frozen pretrained weight, and $\Delta \mathbf { { W } } _ { t } ^ { l }$ denotes the task-specific residual to be learned.

The residual branch follows the same knowledge encoding, consolidation, and integration pipeline as $\mathsf { \Sigma } _ { C o ^ { 2 } I n }$ Specifically, the working memory initializes $\Delta \mathbf { { W } } _ { t } ^ { l }$ as a zero matrix with the same size as $\hat { W _ { p r e } ^ { l } }$ and optimizes it on the current task. After training, the learned residual is consolidated into the long-term memory by decomposing it into a pattern basis $B _ { t } ^ { l }$ and a compact knowledge representation $\mathbf { \dot { } } A _ { t } ^ { l }$ . The long-term memory then integrates the basis and representations from all learned tasks into separate layers $B ^ { \acute { l } }$ and $A ^ { l } ,$ enabling efficient parallel inference.

We follow ACMap [59] and evaluate $C o ^ { 2 } I$ n using ViT-B/16 pretrained on ImageNet as the backbone. Three representative baselines are selected: Naive LoRA, which trains an independent LoRA module for each task and merges them during inference; InfoLoRA [60], a LoRA-based IL method; and ACMap [59], an adapter-based IL method. $\mathrm { A s }$ shown in Table 13, with comparable memory consumption, $C o ^ { 2 } I n$ achieves the best Task-IL performance on both CIFAR-100 and Tiny-ImageNet, indicating that the proposed residual consolidation strategy effectively preserves taskspecific knowledge. For Class-IL, $C o ^ { 2 } I n$ obtains performance comparable to existing pretrained IL methods, while clearly outperforming Naive LoRA. These results show that $C o ^ { 2 } I$ n is effective for pretrained-model-based IL.

## 8 PROOF OF THE ORTHOGONALITY IN EQ. (3)

To support the orthogonality claim in Eq.(3), we present Theorem 8.1, which is derived from the properties of SVD.

Theorem 8.1. Let B be an $m \times r$ matrix with orthonormal columns, i.e., $B ^ { \top } B = I _ { r } .$ . Given any $m \times n$ matrix $Z ,$ perform the singular value decomposition:

$$
U \Sigma V ^ { \top } = Z - B B ^ { \top } Z ,
$$

where U is an $m \times k$ orthogonal matrix, Σ is a diagonal matrix containing the non-zero singular values, and V is $\textit { a k } \times \textit { n }$ orthogonal matrix. Then, the columns of U lie in the left null space of B, i.e.,

$$
B ^ { \top } U = { \bf 0 } .
$$

Proof.

$$
\begin{array} { r l } & { \boldsymbol { B } ^ { \top } \boldsymbol { U } = \boldsymbol { B } ^ { \top } \boldsymbol { U } \boldsymbol { \Sigma } \boldsymbol { \Sigma } ^ { - 1 } = \boldsymbol { B } ^ { \top } \boldsymbol { U } \boldsymbol { \Sigma } \boldsymbol { V } ^ { \top } \boldsymbol { V } \boldsymbol { \Sigma } ^ { - 1 } } \\ & { \qquad = \boldsymbol { B } ^ { \top } \left( \boldsymbol { Z } - \boldsymbol { B } \boldsymbol { B } ^ { \top } \boldsymbol { Z } \right) \boldsymbol { V } \boldsymbol { \Sigma } ^ { - 1 } = \left( \boldsymbol { B } ^ { \top } \boldsymbol { Z } - \boldsymbol { B } ^ { \top } \boldsymbol { Z } \right) \boldsymbol { V } \boldsymbol { \Sigma } ^ { - 1 } } \\ & { \qquad = \mathbf { 0 } \boldsymbol { V } \boldsymbol { \Sigma } ^ { - 1 } = \mathbf { 0 } . } \end{array}
$$

## 9 CONCLUSION

This work draws inspiration from the multi-level human memory system and proposes a bi-level memory framework for incremental learning. The bi-level memory framework consists of a working memory for acquiring new knowledge and a long-term memory for preserving previously learned information. With the separate memory units for knowledge acquiring and remembering, the bi-level memory framework ensures both high plasticity and stability. Furthermore, in light of the human memory mechanism, we devise a series of strategies for knowledge encoding, consolidation, and integration within the bi-level memory framework, which together constitute our $C o ^ { 2 } .$ In. Extensive experiments demonstrate that $C o ^ { 2 } I n$ effectively handles Task-IL and Class-IL with efficient memory and computation utilization.

Nonetheless, $C o ^ { 2 }$ In has some limitations, such as missing backward knowledge transfer and escalating memory consumption proportional to the number of tasks learned. In the future, we plan to improve $C o ^ { 2 } I n$ by integrating advanced pattern basis editing methods, such as meta-learning. These enhancements are expected to facilitate backward knowledge transfer and effectively curb the expansion of the pattern basis, further advancing the efficiency of $C o ^ { 2 } I n$

## ACKNOWLEDGMENTS

This work was supported in part by the National Natural Science Foundation of China under Grant 62276019, 62306028, 62501043, U22B2004, in part by the Fundamental Research Funds for the Central Universities under Grant 2022JBQY007, 2026XKRC005, in part by the Beijing Natural Science Foundation under Grant L231019, in part by the RIE2025 Industry Alignment Fund – Industry Collaboration Projects (IAF-ICP) (Award I2301E0026), administered by A\*STAR, supported by Alibaba Group and NTU Singapore through Alibaba-NTU Global e-Sustainability CorpLab (ANGEL), partially funded by National Research Foundation Fellowship (NRFF13-2021-0006), Singapore.

## REFERENCES

[1] D.-W. Zhou, Q.-W. Wang, Z.-H. Qi, H.-J. Ye, D.-C. Zhan, and Z. Liu, “Deep class-incremental learning: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 9851–9873, 2024.

[2] Z. Mai, R. Li, J. Jeong, D. Quispe, H. Kim, and S. Sanner, “Online continual learning in image classification: An empirical survey,” Neurocomputing, vol. 469, pp. 28–51, 2022.

[3] M. Delange, R. Aljundi, M. Masana, S. Parisot, X. Jia, A. Leonardis, G. Slabaugh, and T. Tuytelaars, “A continual learning survey: Defying forgetting in classification tasks,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.

[4] P. Buzzega, M. Boschini, A. Porrello, D. Abati, and S. CALDER-ARA, “Dark experience for general continual learning: A strong, simple baseline,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 15 920–15 930.

[5] G. Saha, I. Garg, and K. Roy, “Gradient projection memory for continual learning,” in International Conference on Learning Representations, 2021.

[6] S. Wang, X. Li, J. Sun, and Z. Xu, “Training networks in null space of feature covariance for continual learning,” in IEEE Conference on Computer Vision and Pattern Recognition, 2021, pp. 184–193.

[7] M. Boschini, L. Bonicelli, P. Buzzega, A. Porrello, and S. Calderara, “Class-incremental continual learning into the extended derverse,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 5, pp. 5497–5512, 2023.

[8] K. W. Spence and J. T. Spence, Psychology of learning and motivation. Academic Press, 1967.

[9] L. Bonicelli, M. Boschini, A. Porrello, C. Spampinato, and S. Calderara, “On the effectiveness of lipschitz-driven rehearsal in continual learning,” in Advances in Neural Information Processing Systems, 2022.

[10] D. Lopez-Paz and M. A. Ranzato, “Gradient episodic memory for continual learning,” in Advances in Neural Information Processing Systems, vol. 30, 2017, pp. 6467–6476.

[11] A. Chaudhry, M. Ranzato, M. Rohrbach, and M. Elhoseiny, “Efficient lifelong learning with a-gem,” in International Conference on Learning Representations, 2019.

[12] Y. Wu, Y. Chen, L. Wang, Y. Ye, Z. Liu, Y. Guo, and Y. Fu, “Large scale incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, 2019, pp. 374–382.

[13] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, “icarl: Incremental classifier and representation learning,” in IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 2001–2010.

[14] J. Smith, Y.-C. Hsu, J. Balloch, Y. Shen, H. Jin, and Z. Kira, “Always be dreaming: A new approach for data-free class-incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, October 2021.

[15] Z. Zhao, Z. Zhang, X. Tan, J. Liu, Y. Qu, Y. Xie, and L. Ma, “Rethinking gradient projection continual learning: Stability / plasticity feature space decoupling,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2023, pp. 3718–3727.

[16] Y. Kong, L. Liu, Z. Wang, and D. Tao, “Balancing stability and plasticity through advanced null space in continual learning,” in European Conference on Computer Vision, 2022, pp. 219–236.

[17] H. Liu and H. Liu, “Continual learning with recursive gradient optimization,” in International Conference on Learning Representations, 2022.

[18] M. Mermillod, A. Bugaiska, and P. BONIN, “The stabilityplasticity dilemma: investigating the continuum from catastrophic forgetting to age-limited learning effects,” Frontiers in Psychology, vol. 4, 2013.

[19] S. McKenzie and H. Eichenbaum, “Consolidation and reconsolidation: Two lives of memories?” Neuron, vol. 71, no. 2, pp. 224–233, 2011.

[20] M. T. R. van Kesteren and M. Meeter, “How to optimize knowledge construction in the brain,” npj Science of Learning, vol. 5, no. 1, p. 5, 2020.

[21] Y. Cheng, D. Wang, P. Zhou, and T. Zhang, “A survey of model compression and acceleration for deep neural networks,” arXiv preprint arXiv:1710.09282, 2017.

[22] G. M. Van de Ven, T. Tuytelaars, and A. S. Tolias, “Three types of incremental learning,” Nature Machine Intelligence, vol. 4, no. 12, pp. 1185–1197, 2022.

[23] W. Sun, Q. Li, J. Zhang, W. Wang, and Y.-a. Geng, “Decoupling learning and remembering: A bilevel memory rramework with knowledge projection for task-incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2023.

[24] Q. Wang, J. Liu, Z. Ji, Y. Pang, and Z. Zhang, “Hierarchical correlations replay for continual learning,” Knowledge-Based Systems, vol. 250, p. 109052, 2022.

[25] H. Cha, J. Lee, and J. Shin, “Co2l: Contrastive continual learning,” in IEEE International Conference on Computer Vision, October 2021, pp. 9516–9525.

[26] E. Arani, F. Sarfraz, and B. Zonooz, “Learning fast, learning slow: a general continual learning method based on complementary learning system,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview. net/forum?id=uxxFrDwrE7Y

[27] Y. Wei, J. Ye, Z. Huang, J. Zhang, and H. Shan, “Online prototype learning for online continual learning,” in IEEE International Conference on Computer Vision, October 2023, pp. 18 764–18 774.

[28] H. Shin, J. K. Lee, J. Kim, and J. Kim, “Continual learning with deep generative replay,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

[29] H. Yin, P. Molchanov, J. M. Alvarez, Z. Li, A. Mallya, D. Hoiem, N. K. Jha, and J. Kautz, “Dreaming to distill: data-free knowledge transfer via deepinversion,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2020.

[30] W. Hu, Z. Lin, B. Liu, C. Tao, Z. Tao, J. Ma, D. Zhao, and R. Yan, “Overcoming catastrophic forgetting via model adaptation,” in International Conference on Learning Representations, 2019.

[31] W. Sun, Q. Li, J. Zhang, D. Wang, W. Wang, and Y. ao Geng, “Exemplar-free class incremental learning via discriminative and comparable parallel one-class classifiers,” p. 109561, 2023.

[32] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations, 2015.

[33] Z. Li and D. Hoiem, “Learning without forgetting,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 40, no. 12, pp. 2935–2947, 2017.

[34] A. Douillard, M. Cord, C. Ollion, T. Robert, and E. Valle, “Podnet: Pooled outputs distillation for small-tasks incremental learning,” in European Conference on Computer Vision, 2020, pp. 86–102.

[35] X. Tao, X. Chang, X. Hong, X. Wei, and Y. Gong, “Topologypreserving class-incremental learning,” in European Conference on Computer Vision. Springer, 2020, pp. 254–270.

[36] X. Li, S. Wang, J. Sun, and Z. Xu, “Variational data-free knowledge distillation for continual learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–17, 2023.

[37] W. Shi and M. Ye, “Prototype reminiscence and augmented asymmetric knowledge aggregation for non-exemplar class-incremental learning,” in IEEE International Conference on Computer Vision, October 2023, pp. 1772–1781.

[38] J. Kirkpatrick, R. Pascanu, N. Rabinowitz, J. Veness, G. Desjardins, A. A. Rusu, K. Milan, J. Quan, T. Ramalho, A. Grabska-Barwinska et al., “Overcoming catastrophic forgetting in neural networks,” Proceedings of the National Academy of Sciences, vol. 114, no. 13, pp. 3521–3526, 2017.

[39] F. Zenke, B. Poole, and S. Ganguli, “Continual learning through synaptic intelligence,” in International Conference on Machine Learning, 2017, pp. 3987–3995.

[40] J. Schwarz, W. Czarnecki, J. Luketina, A. Grabska-Barwinska, Y. W. Teh, R. Pascanu, and R. Hadsell, “Progress & compress: A scalable framework for continual learning,” in International Conference on Machine Learning, 2018, pp. 4528–4537.

[41] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, “Memory aware synapses: Learning what (not) to forget,” in European Conference on Computer Vision, 2018, pp. 139–154.

[42] A. Mallya and S. Lazebnik, “Packnet: Adding multiple tasks to a single network by iterative pruning,” in IEEE Conference on Computer Vision and Pattern Recognition, 2018, pp. 7765–7773.

[43] C.-Y. Hung, C.-H. Tu, C.-E. Wu, C.-H. Chen, Y.-M. Chan, and C.-S. Chen, “Compacting, picking and growing for unforgetting continual learning,” in Advances in Neural Information Processing Systems, vol. 32, 2019, pp. 13 647–13 657.

[44] A. Mallya, D. Davis, and S. Lazebnik, “Piggyback: Adapting a single network to multiple tasks by learning to mask weights,” in European Conference on Computer Vision, 2018, pp. 67–82.

[45] T. Konishi, M. Kurokawa, C. Ono, Z. Ke, G. Kim, and B. Liu, “Parameter-level soft-masking for continual learning,” 2023.

[46] A. A. Rusu, N. C. Rabinowitz, G. Desjardins, H. Soyer, J. Kirkpatrick, K. Kavukcuoglu, R. Pascanu, and R. Hadsell, “Progressive neural networks,” arXiv preprint arXiv:1606.04671, 2016.

[47] W. Sun, J. Zhang, D. Wang, Y.-a. Geng, and Q. Li, “Ilcoc: An incremental learning framework based on contrastive one-class classifiers,” in IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2021, pp. 3580–3588.

[48] J. Yoon, E. Yang, J. Lee, and S. J. Hwang, “Lifelong learning with dynamically expandable networks.” in International Conference on Learning Representations, 2018.

[49] R. Hyder, K. Shao, B. Hou, P. Markopoulos, A. Prater-Bennette, and M. S. Asif, “Incremental task learning with incremental rank updates,” in European Conference on Computer Vision, 2022, pp. 566–582.

[50] W. Hu, Q. Qin, M. Wang, J. Ma, and B. Liu, “Continual learning by using information of each class holistically,” in AAAI Conference on Artificial Intelligence, 2021.

[51] V. K. Verma, K. J. Liang, N. Mehta, P. Rai, and L. Carin, “Efficient feature transformations for discriminative and generative continual learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2021, pp. 13 865–13 875.

[52] L. Wang, M. Zhang, Z. Jia, Q. Li, C. Bao, K. Ma, J. Zhu, and Y. Zhong, “Afec: Active forgetting of negative transfer in continual learning,” in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 22 379–22 391.

[53] F. Sarfraz, E. Arani, and B. Zonooz, “Sparse coding in a dual memory system for lifelong learning,” in AAAI Conference on Artificial Intelligence, vol. 37, no. 8, Jun. 2023, pp. 9714–9722.

[54] Z. Wang, Z. Zhang, C.-Y. Lee, H. Zhang, R. Sun, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Learning to prompt for continual learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2022, pp. 139–149.

[55] Z. Wang, Z. Zhang, S. Ebrahimi, R. Sun, H. Zhang, C.-Y. Lee, X. Ren, G. Su, V. Perot, J. Dy, and T. Pfister, “Dualprompt: Complementary prompting for rehearsal-free continual learning,” in European Conference on Computer Vision, 2022, pp. 631–648.

[56] J. S. Smith, L. Karlinsky, V. Gutta, P. Cascante-Bonilla, D. Kim, A. Arbelle, R. Panda, R. Feris, and Z. Kira, “Coda-prompt: Continual decomposed attention-based prompting for rehearsal-free continual learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2023, pp. 11 909–11 919.

[57] D.-W. Zhou, H.-L. Sun, H.-J. Ye, and D.-C. Zhan, “Expandable subspace ensemble for pre-trained model-based class-incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2024, pp. 23 554–23 564.

[58] Y. Tan, Q. Zhou, X. Xiang, K. Wang, Y. Wu, and Y. Li, “Semantically-shifted incremental adapter-tuning is a continual vitransformer,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2024, pp. 23 252–23 262.

[59] T. Fukuda, H. Kera, and K. Kawamoto, “Adapter merging with centroid prototype mapping for scalable class-incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2025, pp. 4884–4893.

[60] Y.-S. Liang and W.-J. Li, “Inflora: Interference-free low-rank adaptation for continual learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2024, pp. 23 638–23 647.

[61] J. He, Z. Duan, and F. Zhu, “Cl-lora: Continual low-rank adaptation for rehearsal-free class-incremental learning,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), June 2025, pp. 30 534–30 544.

[62] M. Masana, X. Liu, B. Twardowski, M. Menta, A. D. Bagdanov, and J. van de Weijer, “Class-incremental learning: survey and performance evaluation on image classification,” IEEE Transactions on Pattern Analysis and Machine Intelligence, pp. 1–20, 2022.

[63] Y. Yang, L. Xie, S. Chen, X. Li, Z. Lin, and D. Tao, “Do we really need a learnable classifier at the end of deep neural network?” arXiv preprint arXiv:2203.09081, 2022.

[64] Y. Yang, H. Yuan, X. Li, Z. Lin, P. Torr, and D. Tao, “Neural collapse inspired feature-classifier alignment for few-shot classincremental learning,” in International Conference on Learning Representations, 2023.

[65] L. Ruff, R. Vandermeulen, N. Goernitz, L. Deecke, S. A. Siddiqui, A. Binder, E. Muller, and M. Kloft, “Deep one-class classification,”¨ in International Conference on Machine Learning, 2018, pp. 4393– 4402.

[66] A. Krizhevsky, “Learning multiple layers of features from tiny images,” University of Toronto, 05 2012.

[67] Stanford, “Tiny imagenet challenge (cs231n),” 2015. [Online]. Available: http://tiny-imagenet.herokuapp.com/

[68] S. Jha, D. Gong, H. Zhao, and L. Yao, “Npcl: Neural processes for uncertainty-aware continual learning,” in Advances in Neural Information Processing Systems, 2023.

[69] G. Zeng, Y. Chen, B. Cui, and S. Yu, “Continual learning of contextdependent processing in neural networks,” Nature Machine Intelligence, vol. 1, no. 8, pp. 364–372, 2019.

[70] F. Zhu, X.-Y. Zhang, C. Wang, F. Yin, and C.-L. Liu, “Prototype augmentation and self-supervision for incremental learning,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2021, pp. 5871–5880.

[71] Z. Miao, Z. Wang, W. Chen, and Q. Qiu, “Continual learning with filter atom swapping,” in International Conference on Learning Representations, 2022.

[72] H. Kang, R. J. L. Mina, S. R. H. Madjid, J. Yoon, M. Hasegawa-Johnson, S. J. Hwang, and C. D. Yoo, “Forget-free continual learning with winning subnetworks,” in International Conference on Machine Learning, 2022.

[73] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[74] L. Yu, T. Hua, W. Yang, P. Ye, and Q. Liao, “Cdhd: Contrastive ereamer for hint distillation,” in IEEE International Conference on Acoustics, Speech and Signal Processing, 2023, pp. 1–5.

[75] H.-J. Chen, A.-C. Cheng, D.-C. Juan, W. Wei, and M. Sun, “Mitigating forgetting in online continual learning via instance-aware parameterization,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 17 466–17 477.

[76] Y. Shi, L. Yuan, Y. Chen, and J. Feng, “Continual learning via bitlevel information preserving,” in IEEE Conference on Computer Vision and Pattern Recognition, June 2021, pp. 16 674–16 683.

[77] S.-W. Lee, J.-H. Kim, J. Jun, J.-W. Ha, and B.-T. Zhang, “Overcoming catastrophic forgetting by incremental moment matching,” in Advances in Neural Information Processing Systems, vol. 30, 2017.

![](images/ced5ae91bdaf78716a31ef9a9ec995c26060cbe47ec90d155ac520eaa4487417.jpg)

Wenju Sun received the B.Sc. degree in Software Engineering from Beijing Jiaotong University, Beijing, China. He is currently pursuing his Ph.D. degree at the School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China. His research interests include computer vision and incremental learning.

![](images/3a974dbdbb0ea9244efbc9f3013064a6de7d1a2063bdb3cbc8cc7fc2f732f65f.jpg)

Qingyong Li received the B.Sc. degree in Computer Science and Technology from Wuhan University, Wuhan, China, in 2001, and his Ph.D. degree in Computer Science and Technology from the Institute of Computing Technology (ICT), Chinese Academy of Sciences (CAS), Beijing, China, in 2006. He is currently a Professor at Beijing Jiaotong University, Beijing. His current research interests include computer vision and artificial intelligence.

![](images/dc41fad2bae191f75f5043cd6a7c27eff2b25f82d1d3839a7451140c768bfc24.jpg)

Boyang Li received the B.Sc. degree in Computer Engineering from Nanyang Technological University, Singapore, in 2008, and his Ph.D. degree in Computer Science from the Georgia Institute of Technology, Georgia, US, in 2014. He is currently an Associate Professor at Nanyang Technological University, Singapore. His current research interests include computational narrative intelligence, multi-modal learning, and machine learning.

![](images/5bf2588ed20e7899aade1dc91d712836063953d09dae7055e5e8d08373e88e0b.jpg)

Wen Wang received the Ph.D. degree in Computer Science and Technology from the Institute of Computing Technology (ICT), Chinese Academy of Sciences (CAS), Beijing, China, in June 2017. Currently, she is an Associate Professor at Beijing Jiaotong University, Beijing. Her research interests mainly include computer vision, pattern recognition, and machine learning.

![](images/514dee813e1a16d94a3823b5762b1ba3f814f5a26fdf3a147266c11fe4374aec.jpg)

Siyu Zhang received the B.Sc. degree in Engineering Mechanics from Beijing Jiaotong University, Beijing, China. She is currently pursuing her M.S. degree at the School of Computer Science and Technology, Beijing Jiaotong University, Beijing, China. Her research interests include incremental learning and novel class discovery.

![](images/a526c73e2682dedbadd293220502085e6b5c073007bf6bd4e0a61ba689cd3691.jpg)

Yi Wei received the Ph.D. degree in Computer Science and Technology from the Institute of Computing Technology (ICT), Chinese Academy of Sciences (CAS), Beijing, China, in 2008. He is currently an Associate Professor at the School of Mathematics and Computer Science, Wuyi University, Fujian, China. His research interests include causal discovery, causal inference, causal learning, and their applications.

![](images/076e9af5787f790bbf66805b7389184e3896cd5f85d1285f67e1f8011f4c1897.jpg)

Yangli-ao Geng received his Ph.D. in Computer Science and Technology from Beijing Jiaotong University, Beijing, China, in 2021. He is currently an Assistant Professor at the School of Computer and Information Technology, Beijing Jiaotong University. His research interests include graph neural networks, spectral graph methods, and graph learning.