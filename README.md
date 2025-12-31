# Multi-Node-RAG-LLM-platform
Distributed RAG System with Continuous Learning on Kubernetes
-

### Short Description

A **production-grade Retrieval-Augmented Generation (RAG) platform** deployed on a **heterogeneous Kubernetes cluster (x86 + ARM64)**, enhanced with **feedback-driven fine-tuning**.

This project is built to demonstrate:
- Real world **ML-Ops system design**
- **Distributed systems** thinking
- Practical **LLM/RAG engineering**
- End-to-end **ML lifecycle** (inference -> feedback -> training -> deployment)

### Detailed Description

This project implements a distributed Retrieval-Augmented Generation (RAG) platform that enables users to query large document collections with accurate, context-aware responses. The system combines semantic retrieval with large language model inference to provide grounded answers while maintaining low latency and high scalability.

The platform is designed to run on a Kubernetes-based multi-node environment, supporting heterogeneous hardware and independent scaling of ingestion, retrieval, and inference components. It includes a feedback-driven learning loop that continuously improves response quality through dataset generation, fine-tuning, evaluation, and controlled model deployment.

The primary objective of this project is to build a reliable, scalable, and continuously improving document question-answering system that mirrors real-world production ML systems, emphasizing robustness, observability, and lifecycle management rather than isolated model performance.