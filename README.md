# awesome-graph-reduction

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) ![Stars](https://img.shields.io/github/stars/ChandlerBang/awesome-graph-reduction?color=yellow)

[IJCAI 2024] This is a curated list of papers about graph reduction including graph condensation, graph coarsening, graph sparsification, graph summarization, etc.

If you want to add new entries, please make PRs with the same format.

This list serves as a complement to the survey below.

![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)[[A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation]](https://arxiv.org/abs/2402.03358)

<div align=center><img src="https://github.com/ChandlerBang/awesome-graph-reduction/blob/main/figs/graph_reduction.png" width="500" /></div>

If you find this repo helpful, we would appreciate it if you could cite our survey.

```
@article{hashemi2024comprehensive,
  title={A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation},
  author={Hashemi, Mohammad and Gong, Shengbo and Ni, Juntong and Fan, Wenqi and Prakash, B Aditya and Jin, Wei},
  journal={International Joint Conference on Artificial Intelligence (IJCAI)},
  year={2024}
}
```

## Graph Condensation / Graph Dataset Distillation

- [arXiv 2025] Efficient Graph Condensation via Gaussian Process. [[pdf]](https://arxiv.org/pdf/2501.02565) [[code]](https://github.com/WANGLin0126/GCGP)
- [arXiv 2025] (App) GraphDART: Graph Distillation for Efficient Advanced Persistent Threat Detection. [[pdf]](https://arxiv.org/pdf/2501.02796)
- [AAAI 2025] Bi-Directional Multi-Scale Graph Dataset Condensation via Information Bottleneck. [[pdf]](https://arxiv.org/abs/2412.17355) [[code]](https://github.com/RingBDStack/BiMSGC)
- [arXiv 2024] Contrastive Graph Condensation: Advancing Data Versatility through Self-Supervised Learning. [[pdf]](https://arxiv.org/abs/2411.17063)
- [arXiv 2024] Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training. [[pdf]](https://arxiv.org/pdf/2311.15772)
- [arXiv 2024] Bonsai: Gradient-free Graph Distillation for Node Classification. [[pdf]](https://arxiv.org/abs/2410.17579)
- [ECCV 2024] GSTAM: Efficient Graph Distillation with Structural Attention-Matching. [[pdf]](https://arxiv.org/pdf/2408.16871) [[code]](https://github.com/arashrasti96/GSTAM)
- [KDD 2024] Self-Supervised Learning for Graph Dataset Condensation. [[pdf]](https://dl.acm.org/doi/abs/10.1145/3637528.3671682) [[code]](https://github.com/wyx11112/SGDC)
- [arXiv 2024] Backdoor Graph Condensation. [[pdf]](https://arxiv.org/abs/2407.11025)
- [arXiv 2024] TinyGraph: Joint Feature and Node Condensation for Graph Neural Networks. [[pdf]](https://arxiv.org/abs/2407.08064)
- [arXiv 2024] RobGC: Towards Robust Graph Condensation. [[pdf]](https://arxiv.org/pdf/2406.13200)
- [KDD 2024] Graph Condensation for Open-World Graph Learning. [[pdf]](https://arxiv.org/html/2405.17003v1)
- [ICML 2024] Graph Condensation via Eigenbasis Matching. [[pdf]](https://arxiv.org/pdf/2310.09202.pdf) [[code]](https://github.com/liuyang-tian/GDEM)
- [ICLR 2024] Mirage: Model-Agnostic Graph Distillation for Graph Classification. [[pdf]](https://openreview.net/pdf?id=78iGZdqxYY) [[code]](https://github.com/idea-iitd/Mirage)
- [arXiv 2024] Calibrated Dataset Condensation for Faster Hyperparameter Search. [[pdf]](https://arxiv.org/pdf/2405.17535)
- [arXiv 2024] Federated Graph Condensation with Information Bottleneck Principles. [[pdf]](https://arxiv.org/pdf/2405.03911)
- [arXiv 2024] Rethinking and Accelerating Graph Condensation: A Training-Free Approach with Class Partition. [[pdf]](https://arxiv.org/pdf/2405.13707)
- [ECML PKDD 2024] Simple Graph Condensation. [[pdf]](https://arxiv.org/pdf/2403.14951.pdf)
- [arXiv 2024] Graph Data Condensation via Self-expressive Graph Structure Reconstruction. [[pdf]](https://arxiv.org/pdf/2403.07294v1.pdf) [[code]](https://www.dropbox.com/scl/fi/2aonyp5ln5gisdqtjimu8/GCSR.zip?rlkey=11cuwfpsf54wxiiktu0klud0x&dl=0)
- [ICML 2024] Navigating Complexity: Toward Lossless Graph Condensation via Expanding Window Matching. [[pdf]](https://arxiv.org/pdf/2402.05011.pdf) [[code]](https://github.com/NUS-HPC-AI-Lab/GEOM)
- [arXiv 2024] Two Trades is not Baffled: Condensing Graph via Crafting Rational Gradient Matching [[pdf]](https://arxiv.org/pdf/2402.04924.pdf) [[code]](https://github.com/NUS-HPC-AI-Lab/CTRL)
- [arXiv 2024] Disentangled Condensation for Large-scale Graphs. [[pdf]](https://arxiv.org/pdf/2401.12231.pdf) [[code]](https://github.com/BangHonor/DisCo)
- [TKDE 2024] Heterogeneous Graph Condensation. [[pdf]](https://ieeexplore.ieee.org/abstract/document/10423255) [[code]](https://github.com/jianjianGJ/hgcond)
- [WWW 2024] Globally Interpretable Graph Learning via Distribution Matching. [[pdf]](https://arxiv.org/abs/2306.10447)
- [WWW 2024] EXGC: Bridging Efficiency and Explainability in Graph Condensation. [[pdf]](https://arxiv.org/pdf/2402.05962.pdf) [[code]](https://github.com/MangoKiller/EXGC)
- [WWW 2024] Fast Graph Condensation with Structure-based Neural Tangent Kernel. [[pdf]](https://arxiv.org/pdf/2310.11046.pdf) [[code]](https://github.com/WANGLin0126/GCSNTK)
- [ICDE 2024] Graph Condensation for Inductive Node Representation Learning. [[pdf]](https://arxiv.org/pdf/2307.15967)
- [arXiv 2023] Attend Who is Weak: Enhancing Graph Condensation via Cross-Free Adversarial Training. [[pdf]](https://arxiv.org/pdf/2311.15772.pdf)
- [TKDE 2024] (App) PUMA: Efficient Continual Graph Learning with Graph Condensation. [[pdf]](https://arxiv.org/pdf/2312.14439) [[code]](https://github.com/superallen13/puma)
- [arXiv 2023] (App) Faster Hyperparameter Search for GNNs via Calibrated Dataset Condensation. [[pdf]](https://openreview.net/pdf?id=ohQPU2G3r3C)
- [arXiv 2022] Graph Condensation via Receptive Field Distribution Matching. [[pdf]](https://arxiv.org/pdf/2206.13697.pdf)
- [arXiv 2023] (App) FedGKD: Unleashing the Power of Collaboration in Federated Graph Neural Networks. [[pdf]](https://arxiv.org/pdf/2309.09517.pdf)
- [Applied Sciences 2023] GCARe: Mitigating Subgroup Unfairness in Graph Condensation through Adversarial Regularization. [[pdf]](https://www.mdpi.com/2076-3417/13/16/9166)
- [NeurIPS 2023] Fair Graph Distillation. [[pdf]](https://openreview.net/pdf?id=xW0ayZxPWs)
- [NeurIPS 2023] Structure-free Graph Condensation: From Large-scale Graphs to Condensed Graph-free Data. [[pdf]](https://arxiv.org/pdf/2306.02664.pdf) [[code]](https://github.com/Amanda-Zheng/SFGC)
- [NeurIPS 2023] Does Graph Distillation See Like Vision Dataset Counterpart? [[pdf]](https://openreview.net/pdf?id=VqIWgUVsXc) [[code]](https://github.com/RingBDStack/SGDD)
- [ICDM 2023] (App) CaT: Balanced Continual Graph Learning with Graph Condensation. [[pdf]](https://arxiv.org/pdf/2309.09455.pdf) [[code]](https://github.com/superallen13/CaT-CGL)
- [KDD 2023] Kernel Ridge Regression-Based Graph Dataset Distillation. [[pdf]](https://dl.acm.org/doi/10.1145/3580305.3599398) [[code]](https://github.com/pricexu/KIDD)
- [KBS 2023] Multiple sparse graphs condensation. [[pdf]](https://www.sciencedirect.com/science/article/pii/S0950705123006548) [[code]](https://github.com/jianjianGJ/MSGC)
- ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)[KDD 2022] Condensing Graphs via One-Step Gradient Matching. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3534678.3539429) [[code]](https://github.com/ChandlerBang/GCond)
- ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)[ICLR 2022] Graph Condensation for Graph Neural Networks. [[pdf]](https://openreview.net/forum?id=WLEx3Jo4QaB) [[code]](https://github.com/ChandlerBang/GCond)

<!--#### Applications
- [ICDM 2023] CaT: Balanced Continual Graph Learning with Graph Condensation. [[pdf]](https://arxiv.org/pdf/2309.09455.pdf) [[code]](https://github.com/superallen13/CaT-CGL)
- [arXiv 2023] FedGKD: Unleashing the Power of Collaboration in Federated Graph Neural Networks. [[pdf]](https://arxiv.org/pdf/2309.09517.pdf)
  -->

## Graph Coarsening / Clustering / Summary

### GNN-involved
- [Nature Communications] Coarse-graining network flow through statistical physics and machine learning. [[pdf]](https://www.nature.com/articles/s41467-025-56034-2) [[code]](https://github.com/3riccc/nfc_model)
- [NeurIPS 2024] (App) A Topology-aware Graph Coarsening Framework for Continual Graph Learning. [[pdf]](https://arxiv.org/pdf/2401.03077)
- [ICML 2024] (App) Translating Subgraphs to Nodes Makes Simple GNNs Strong and Efficient for Subgraph Representation Learning. [[pdf]](https://arxiv.org/pdf/2204.04510) [[code]](https://github.com/dongkwan-kim/S2N)
- [NN 2024] Graph Batch Coarsening framework for scalable graph neural networks. [[pdf]](https://www.sciencedirect.com/science/article/pii/S0893608024008608)
- [arXiv 2024] Feature Driven Graph Coarsening for Scaling Graph Representation Learning. [[pdf]](https://openreview.net/pdf?id=6VuTXirQIv)
- [DSAA 2024] Semi-Supervised Coarsening of Bipartite Graphs for Text Classification via Graph Neural Network. [[pdf]](https://ieeexplore.ieee.org/abstract/document/10722822)
- [ICLR 2024] StructComp: Substituting Propagation with Structural Compression in Training Graph Contrastive Learning. [[pdf]](https://arxiv.org/pdf/2312.04865) [[code]](https://github.com/szzhang17/StructComp/tree/main)
- [ICASSP 2024] Enhancing Performance of Coarsened Graphs with Gradient-Matching. [[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10448089)
- [WWW 2024] Graph-Skeleton: ～ 1% Nodes are Sufficient to Represent Billion-Scale Graph. [[pdf]](https://arxiv.org/pdf/2402.09565.pdf) [[code]](https://github.com/caolinfeng/GraphSkeleton)
- [arXiv 2024] Graph Coarsening with Message-Passing Guarantees. [[pdf]](https://arxiv.org/pdf/2405.18127)
- [Pacific Symposium on Biocomputing 2023] A Graph Coarsening Algorithm for Compressing Representations of Single-Cell Data with Clinical or Experimental Attributes. [[pdf]](https://psb.stanford.edu/psb-online/proceedings/psb23/chen_c.pdf) [[code]](https://github.com/ChenCookie/cytocoarsening)
- [WWW 2024] Graph Coarsening via Convolution Matching for Scalable Graph Neural Network Training. [[pdf]](https://arxiv.org/pdf/2312.15520.pdf) [[code]](https://github.com/amazon-science/convolution-matching)
- [arXiv 2023] ResolvNet: A Graph Convolutional Network with multi-scale Consistency. [[pdf]](https://arxiv.org/pdf/2310.00431.pdf)
- [ICML 2023] Featured Graph Coarsening with Similarity Guarantees. [[pdf]](http://proceedings.mlr.press/v202/kumar23a/kumar23a.pdf)
- [JMLR 2023] A Unified Framework for Optimization-Based Graph Coarsening. [[pdf]](https://www.jmlr.org/papers/volume24/22-1085/22-1085.pdf) [[code]](https://github.com/GraphCoarsening/Featured-Graph-Coarsening)
- [ICLR 2023] Serving Graph Compression for Graph Neural Networks. [[pdf]](https://openreview.net/pdf?id=T-qVtA3pAxG)
- [WWW 2022] (App) ALLIE: Active Learning on Large-scale Imbalanced Graphs. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3485447.3512229)
- [NeurIPS 2022] (App) SizeShiftReg: a Regularization Method for Improving Size-Generalization in Graph Neural Networks. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2022/file/ceeb3fa5be458f08fbb12a5bb783aac8-Paper-Conference.pdf) [[code]](https://github.com/DavideBuffelli/SizeShiftReg)
- [WWWc 2022] Scaling R-GCN Training with Graph Summarization. [[pdf]](https://arxiv.org/pdf/2203.02622.pdf)
- [ICLR 2021] Graph Coarsening with Neural Networks. [[pdf]](https://openreview.net/pdf?id=uxpzitPEooJ) [[blog]](https://iclr-blog-track.github.io/2022/03/25/coarsening/)
- [KDD 2021] Scaling Up Graph Neural Networks Via Graph Coarsening. [[pdf]](https://arxiv.org/pdf/2106.05150.pdf) [[code]](https://github.com/szzhang17/Scaling-Up-Graph-Neural-Networks-Via-Graph-Coarsening)
- [HiPC 2021] (App) DistMILE: A Distributed Multi-Level Framework for Scalable Graph Embedding. [[pdf]](https://ieeexplore.ieee.org/document/9680339)
- [ICWSM 2021] (App) MILE: A Multi-Level Framework for Scalable Graph Embedding. [[pdf]](https://arxiv.org/abs/1802.09612) [[code]](https://github.com/jiongqian/MILE)
- [ICLR 2021] Optimization-Based Algebraic Multigrid Coarsening
  Using Reinforcement Learning [[pdf]](https://arxiv.org/pdf/2106.01854.pdf) [[code]](https://github.com/compdyn/rl_grid_coarsen)
- [AAAI 2021] Unsupervised Learning of Graph Hierarchical Abstractions with Differentiable Coarsening and Optimal Transport. [[pdf]](https://arxiv.org/pdf/1912.11176.pdf) [[code]](https://github.com/matenure/OTCoarsening)
- [ICML 2020] Spectral Clustering with Graph Neural Networks for Graph Pooling. [[pdf]](https://arxiv.org/pdf/1907.00481.pdf) [[code]](https://github.com/FilippoMB/Spectral-Clustering-with-Graph-Neural-Networks-for-Graph-Pooling)
- [KBS 2020] Graph convolutional networks with multi-level coarsening for graph
  classification. [[pdf]](https://www.sciencedirect.com/science/article/pii/S0950705120300629)
- [ICML 2020] Learning Algebraic Multigrid Using Graph Neural Networks. [[pdf]](https://proceedings.mlr.press/v119/luz20a/luz20a.pdf) [[code]](https://github.com/ilayluz/learning-amg)
- [ICLR 2020] (App) GraphZoom: A multi-level spectral approach for accurate and scalable graph embedding. [[pdf]](https://arxiv.org/pdf/1910.02370.pdf) [[code]](https://github.com/cornell-zhang/GraphZoom)
- [AAAI 2018] (App) HARP: Hierarchical Representation Learning for Networks. [[pdf]](https://arxiv.org/abs/1706.07845)
<details>
<summary>non-GNN-involved</summary>

- [AISTATS 2020] Graph Coarsening with Preserved Spectral Properties. [[pdf]](https://arxiv.org/pdf/1802.04447.pdf)
- [NeurIPS 2019] A unifying framework for spectrum-preserving graph sparsification and coarsening. [[pdf]](https://proceedings.neurips.cc/paper_files/paper/2019/file/cd474f6341aeffd65f93084d0dae3453-Paper.pdf) [[code]](https://github.com/TheGravLab/A-Unifying-Framework-for-Spectrum-Preserving-Graph-Sparsification-and-Coarsening)
- [JMLR 2019] Graph reduction with spectral and cut guarantees. [[pdf]](https://arxiv.org/pdf/1808.10650.pdf) [[code]](https://github.com/loukasa/graph-coarsening/tree/v1.1)
- [Chaos 2018] Spectral coarse graining for random walk in bipartite networks. [[pdf]](https://arxiv.org/pdf/1209.1028.pdf)
- [ICML 2018] Spectrally approximating large graphs with smaller graphs. [[pdf]](https://arxiv.org/pdf/1802.07510.pdf)
- [ICDM 2018] NetGist: Learning to Generate Task-Based Network Summaries. [[pdf]](https://faculty.cc.gatech.edu/~badityap/papers/netgist-icdm18.pdf)
- [Signal Processing 2016] (App) A Multiscale Pyramid Transform for Graph Signals. [[pdf]](https://arxiv.org/pdf/1308.4942.pdf)
- [KDD 2014] Fast Influence-based Coarsening for Large Networks. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2623330.2623701)
- [ICSEE 2014] Graph summarization for attributed graphs. [[pdf]](https://ieeexplore.ieee.org/abstract/document/6948163)
- [arXiv 2013] Aggregation-based aggressive coarsening with polynomial smoothing. [[pdf]](https://arxiv.org/pdf/1307.6305.pdf)
- [SIAM 2012] Lean Algebraic Multigrid (LAMG): Fast Graph Laplacian Linear Solver. [[pdf]](https://arxiv.org/pdf/1108.1310.pdf)
- [SIAM 2011] Relaxation-Based Coarsening and Multiscale Graph Organization. [[pdf]](https://arxiv.org/pdf/1004.1220.pdf)
- [SIAM 2011] Algebraic Distance on Graphs. [[pdf]](https://jiechenjiechen.github.io/pub/algebraic_distance_long.pdf)
- [ICDE 2010] Discovery-driven graph summarization. [[pdf]](https://ieeexplore.ieee.org/abstract/document/5447830)
- [TPAMI 2007] Weighted Graph Cuts without Eigenvectors A Multilevel Approach. [[pdf]](https://ieeexplore.ieee.org/document/4302760)
- [Physical Review E 2005] Coarse-Graining and Self-Dissimilarity of Complex Networks. [[pdf]](https://arxiv.org/pdf/q-bio/0405011.pdf)
- [SIAM 1998] (App) A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs. [[pdf]](https://www.cs.utexas.edu/~pingali/CS395T/2009fa/papers/metis.pdf)
- [Bell System 1970] (App) An efficient heuristic procedure for partitioning graphs. [[pdf]](https://ieeexplore.ieee.org/abstract/document/6771089)
  </details>
<!-- - [ICLR 2018] FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling. [[pdf]](https://arxiv.org/pdf/1801.10247.pdf) [[code]](https://github.com/matenure/FastGCN)
- [ICML 2018] Stochastic Training of Graph Convolutional Networks with Variance Reduction. [[pdf]](https://arxiv.org/pdf/1710.10568.pdf) [[code]](https://github.com/thu-ml/stochastic_gcn)-->

<!-- - [IJCAI 2023] Gapformer: Graph Transformer with Graph Pooling for Node Classification. [[pdf]](https://www.ijcai.org/proceedings/2023/0244.pdf)
- [NeurIPS 2022] Hierarchical graph transformer with adaptive node sampling. [[pdf]](https://arxiv.org/pdf/2210.03930.pdf)-->

## Graph Sparsification / Sampling / Selection

### GNN-involved

<!-- - [arXiv 2024] Two Heads Are Better Than One:Boosting Graph Sparse Training via Semantic and Topological Awareness. [[pdf]](https://arxiv.org/pdf/2402.01242.pdf) [[code]](https://anonymous.4open.science/r/GST-0F15)
- [TPAMI 2023] Graph Neural Network Meets Sparse Representation: Graph Sparse Neural Networks via Exclusive Group Lasso. [[pdf]](https://ieeexplore.ieee.org/document/10149528)-->
- [KDD 2025] Large-Scale Spectral Graph Neural Networks via Laplacian Sparsification: Technical Report. [[pdf]](https://arxiv.org/pdf/2501.04570#page=13.84)
- [VLDB 2024] Size Does (Not) Matter? Sparsification and Graph Neural Network Sampling for Large-scale Graphs. [[pdf]](https://vldb.org/workshops/2024/proceedings/LSGDA/LSGDA24.06.pdf#page=1.31)
- [VLDB 2024] Demystifying Graph Sparsification Algorithms in Graph
  Properties Preservation. [[pdf]](https://arxiv.org/pdf/2311.12314) [[code]](https://github.com/yuhanchan/sparsification)
- [arXiv 2024] Graph Sparsification via Mixture of Graphs. [[pdf]](https://arxiv.org/pdf/2405.14260) [[code]](https://github.com/yanweiyue/MoG)
- [arXiv 2024] Spectral Greedy Coresets for Graph Neural Networks. [[pdf]](https://arxiv.org/pdf/2405.17404)
- [TKDE 2024] Graph Rewiring and Preprocessing for Graph Neural Networks Based on Effective Resistance. [[pdf]](https://ieeexplore.ieee.org/abstract/document/10521752)
- [KDD 2023] Interpretable Sparsification of Brain Graphs: Better Practices and Effective Designs for Graph Neural Networks. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/3580305.3599394) [[code]](https://github.com/motivationss/IGS)
- [TNNLS 2023] (App) Ricci Curvature-Based Graph Sparsification for Continual Graph Representation Learning. [[pdf]](https://ieeexplore.ieee.org/document/10225445)
- [NeurIPS 2023] On the Ability of Graph Neural Networks to Model Interactions Between Vertices. [[pdf]](https://arxiv.org/pdf/2211.16494.pdf) [[code]](https://github.com/noamrazin/gnn_interactions)
- [Nature Computational Science 2023] GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks. [[pdf]](https://www.nature.com/articles/s43588-023-00465-8) [[code]](https://github.com/dfdazac/grapes)
- [ICDM 2022] (App) Sparsified Subgraph Memory for Continual Graph Representation Learning. [[pdf]](https://ieeexplore.ieee.org/document/10027629) [[code]](https://github.com/QueuQ/SSM/issues)
- [ISCA 2022] SmartSAGE: Training Large-scale Graph Neural Networks using In-Storage Processing Architectures. [[pdf]](https://arxiv.org/pdf/2205.04711.pdf)
- [UAI 2022] Principle of Relevant Information for Graph Sparsification. [[pdf]](https://proceedings.mlr.press/v180/yu22c/yu22c.pdf) [[code]](https://github.com/SJYuCNEL/PRI-Graphs)
- [ICLR 2020] GraphSAINT: Graph Sampling Based Inductive Learning Method. [[pdf]](https://arxiv.org/pdf/1907.04931.pdf) [[code]](https://github.com/GraphSAINT/GraphSAINT)
- [ICDM 2020] Graph Sparsification with Generative Adversarial Network. [[pdf]](https://arxiv.org/pdf/2009.11736.pdf)
- [ICML 2020] (App) Robust Graph Representation Learning via Neural Sparsification. [[pdf]](https://proceedings.mlr.press/v119/zheng20d/zheng20d.pdf) [[code]](https://github.com/flyingdoog/PTDNet)
- [TOC 2020] Robust Graph Learning from Noisy Data. [[pdf]](https://arxiv.org/pdf/1812.06673.pdf) [[code]](https://github.com/sckangz/RGC)

<details>
<summary>non-GNN-involved</summary>

- [Information Sciences 2024] Generic network sparsification via degree- and subgraph-based edge sampling. [[pdf]](https://www.sciencedirect.com/science/article/pii/S0020025524010107)
- [STOC 2019] A General Framework for Graph Sparsification. [[pdf]](https://arxiv.org/pdf/1004.4080.pdf)
- [NeurIPS 2019] (Privacy) On Differentially Private Graph Sparsification and Applications. [[pdf]](https://papers.nips.cc/paper_files/paper/2019/file/e44e875c12109e4fa3716c05008048b2-Paper.pdf)
- [AISTATS 2016] Graph Sparsification Approaches for Laplacian Smoothing. [[pdf]](https://www.stat.berkeley.edu/~ryantibs/papers/lapsparse.pdf)
- [Circuits and Systems 2013] Kron Reduction of Graphs with Applications to Electrical Networks. [[pdf]](https://arxiv.org/pdf/1102.2950.pdf)
- [CVPR 2012] Non-negative low rank and sparse graph for semi-supervised learning. [[pdf]](https://zhouchenlin.github.io/Publications/2012-CVPR-NNLRS.pdf)
- [VLDB 2012] Densest Subgraph in Streaming and MapReduce. [[pdf]](https://vldb.org/pvldb/vol5/p454_bahmanbahmani_vldb2012.pdf)
- [VLDB 2012] Dense Subgraph Maintenance under Streaming Edge Weight Updates for Real-time Story Identification. [[pdf]](https://vldb.org/pvldb/vol5/p574_albertangel_vldb2012.pdf)
- [PODS 2012] Graph Sketches: Sparsification, Spanners, and Subgraphs. [[pdf]](https://people.cs.umass.edu/~mcgregor/papers/12-pods1.pdf)
- [STOC 2011] A General Framework for Graph Sparsification. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/1993636.1993647)
- [STOC 2011] Spectral Sparsification of Graphs [[pdf]](https://epubs.siam.org/doi/abs/10.1137/08074489X)
- [ICDM 2011] Identity Obfuscation in Graphs through the Information Theoretic Lens. [[pdf]](https://ieeexplore.ieee.org/document/5767905)
- [STOC 2008] Graph Sparsification by Effective Resistances. [[pdf]](https://arxiv.org/pdf/0803.0929.pdf)
- [STOC 2004] Nearly-linear time algorithms for graph partitioning, graph sparsification, and solving linear system. [[pdf]](https://dl.acm.org/doi/10.1145/1007352.1007372)
- [Internet Mathematics 2014] Ranking and Sparsifying a Connection Graph. [[pdf]](https://mathweb.ucsd.edu/~fan/wp/connectionj.pdf)
- [ACM 1994] Random sampling in cut, flow, and network design problems. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/195058.195422)
- [JACM 1997] Sparsification–A Technique for Speeding Up Dynamic Graph Algorithms. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/265910.265914)
- [STOC 1996] Approximating s-t minimum cuts in Õ(n2) time. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/237814.237827)
- [STOC 1994] Random sampling in cut, flow, and network design problems. [[pdf]](https://dl.acm.org/doi/10.1145/195058.195422)
  <!-- - - [ICLR 2020] DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. [[pdf]](https://arxiv.org/pdf/1907.10903.pdf) [[code]](https://github.com/DropEdge/DropEdge)
  [WSDM 2021] Learning to Drop: Robust Graph Neural Network via Topological Denoising. [[pdf]](https://arxiv.org/pdf/2011.07057.pdf) [[code]](https://github.com/flyingdoog/PTDNet)-->
    </details>

## Surveys & Benchmarks

Graph Reduction/ Summarization / Simplification

- [arXiv 2024] Extending Graph Condensation to Multi-Label Datasets: A Benchmark Study. [[pdf]](https://arxiv.org/pdf/2412.17961)
- [arXiv 2024] GC4NC: A Benchmark Framework for Graph Condensation on Node Classification with New Insights. [[pdf]](https://arxiv.org/abs/2406.16715) [[code]](https://github.com/Emory-Melody/GraphSlim/tree/main/benchmark)
- ![image](https://github.com/CurryTang/Towards-Graph-Foundation-Models-New-perspective-/assets/15672123/89a23a37-71d4-47f7-8949-7d859a41e369)**[IJCAI 2024] A Comprehensive Survey on Graph Reduction: Sparsification, Coarsening, and Condensation. [[pdf]](https://arxiv.org/abs/2402.03358)**
- [NeurIPS 2024] GC-Bench: [[pdf]](https://openreview.net/pdf?id=ScPgzCZ6Lo) [[code]](https://github.com/RingBDStack/GC-Bench)
- [arXiv 2024] GCondenser: Benchmarking Graph Condensation. [[pdf]](https://arxiv.org/pdf/2405.14246) [[code]](https://github.com/superallen13/GCondenser)
- [arXiv 2024] A Survey on Graph Condensation. [[pdf]](https://arxiv.org/abs/2402.02000)
- [arXiv 2024] Graph Condensation: A Survey. [[pdf]](https://arxiv.org/abs/2401.11720)
- [Communications of the ACM] Spectral Sparsification of Graphs: Theory and Algorithms. [[pdf]](http://cs-www.cs.yale.edu/homes/spielman/PAPERS/CACMsparse.pdf)
- [TAI 2023] A Comprehensive Survey on Graph Summarization with Graph Neural Networks. [[pdf]](https://arxiv.org/abs/2302.06114)
- [SeMA 2022] Graph coarsening: from scientific computing to machine learning. [[pdf]](https://link.springer.com/article/10.1007/s40324-021-00282-x)
- [CSR 2020] Multilayer network simplification: Approaches, models and methods [[pdf]](https://arxiv.org/abs/2004.14808)
- [JMLR 2018] Community Detection and Stochastic Block Models: Recent Developments. [[pdf]](https://jmlr.org/papers/volume18/16-480/16-480.pdf)
- [ACS 2018] Graph Summarization Methods and Applications: A Survey. [[pdf]](https://arxiv.org/abs/1612.04883)
- [HPDC 2016] Efficient Processing of Large Graphs via Input Reduction. [[pdf]](https://dl.acm.org/doi/pdf/10.1145/2907294.2907312)
- [VLDB 2005] Densest Subgraph Discovery on Large Graphs: Applications, Challenges, and Techniques. [[pdf]](https://www.vldb.org/pvldb/vol15/p3766-luo.pdf)
- [SIAM 1972] The Transitive Reduction of a Directed Graph. [[pdf]](https://www.cs.tufts.edu/comp/150FP/archive/al-aho/transitive-reduction.pdf)

Other related topics

- [arXiv 2023] Dataset Distillation: A Comprehensive Review. [[pdf]](https://arxiv.org/pdf/2301.07014.pdf) [[github]](https://github.com/Guang000/Awesome-Dataset-Distillation)
- [IJCAI 2023] A Survey on Dataset Distillation: Approaches, Applications and Future Directions. [[pdf]](https://arxiv.org/pdf/2305.01975.pdf)

## Toolkits

- Mongoose: [[pdf]](https://people.clas.ufl.edu/hager/files/mongoose.pdf) [[code]](https://github.com/ScottKolo/Mongoose)
- PyGSP: [[code]](https://github.com/epfl-lts2/pygsp)
- Graph-tool: [[code]](https://graph-tool.skewed.de/)
- Networkit: [[code]](https://networkit.github.io/)
