# About Us

At AI Planet, we’ve built a secure and reliable AI platform aimed at empowering everyone, individuals and enterprises alike, to harness the full potential of AI. Our end-to-end GenAI open source frameworks are designed to simplify the development, evaluation and deployment of Gen AI applications, and autonomous agents. 

Our unique approach minimizes risks like AI “hallucinations” (inaccurate outputs) by enhancing factual accuracy and offering tailored customizable solutions. Beyond our technology, we’re fostering a global community of over 300,000 members committed to empowering the world with essential AI education and advancing secure AI for real-world impact. 

![410950520-11472570-6ac6-4a24-bf45-5b24f9edb645](https://github.com/user-attachments/assets/a3a81a41-4d93-4466-80c3-b3c1742232f4)

[![Contributor Guidance](https://img.shields.io/badge/Contributor_Guidance-blue?style=for-the-badge)](https://github.com/aiplanethub/Contributor-Guidance/blob/main/README.md)

# AI Planet - GSoC 2025 Ideas

## 💡 Project 1: Agentic RAG Implementation for OpenAGI Knowledge Base 

- **Difficulty:** Easy (~90 hours of project)

### Description

Implement a Retrieval-Augmented Generation (RAG) system within OpenAGI to enable agents to dynamically access and update a centralized knowledge base. This project will enhance agents' ability to retrieve contextual information during task execution, improving accuracy and reducing hallucinations.

### Objective

* Extend the Worker class to include RAG actions (e.g., RetrieveFromKnowledgeBase, UpdateKnowledgeBase)
* Connect RAG to OpenAGI’s TaskPlanner to prioritize knowledge retrieval during task decomposition.
* Build a vector database (e.g., FAISS or Chroma) for efficient similarity searches as part of the Knowledge base component. 
* Develop a hybrid retrieval system combining semantic search (via LLMs) and keyword-based lookup.
* Test with OpenAGI’s existing use cases and cookbooks. 
* Improve the TaskPlanner and Worker inference latency to adapt to this project. 

### Skills Required

Python, Vector databases (such as Qdrant, Chroma, Milvus), Langchain, Google Gemini or other LLM inference. 

### Mentor

- Tarun R Jain: 
  Data Scientist at AI Planet and Google Developer Expert in AI/ML. Also been part of Google Summer of Code 2023 at caMicroscope and Google Summer of Code 2024 at Red Hen Lab. 

- Mukesh Thawani 
  Engineering Leader. Mukesh is a seasoned engineering leader with over a decade of experience in backend development and NLP, specializing in building AI chatbots at scale. A graduate of Georgia Tech, he has consistently driven innovation by developing high-performance microservices with sub-100ms latency and implementing advanced message clustering systems using UMAP and HDBSCAN.

## 💡 Project 2: Self-Learning Mechanisms for Behavior Optimization

- **Difficulty:** Medium-Hard (~175 hours of project). *Default but negotiable size**

### Description

Enhance OpenAGI agents with self-learning capabilities, enabling them to optimize their behavior through RLHF feedback mechanism. 

### Objective

* Modify the TaskPlanner to incorporate a reward signal from the Evaluation module.
* Use OpenAGI’s Memory to store historical task outcomes for RL training.
* Design a lightweight RL loop where agents adjust strategies based on task success metrics (e.g., user satisfaction, accuracy).
* Implement a feedback mechanism (e.g., user ratings) to guide behavior optimization.
* Benchmark against OpenAGI’s existing [benchmark script](https://github.com/aiplanethub/openagi/blob/main/benchmark.py).
* Improve the TaskPlanner and Worker inference latency to adapt to this project. 

### Skills Required

Reinforcement learning, Python, Langchain, OOPs. 

### Mentor

- Anurag Gupta
  Have spent 7 years at Meta as a Technical Lead, he worked on high-scale platforms like WhatsApp, where he pioneered tailored user journeys for businesses, established robust product metrics, and led privacy and integrity reviews. 
  Anurag also played a pivotal role in mentoring over 20 engineers. Prior to Meta, he was one of the founding engineers at Zeta

## 💡 Project 3: Shared Knowledge Base & Cross-Agent Learning 

- **Difficulty:** Medium-Hard (~175 hours of project). *Default but negotiable size**

### Description

Develop a shared knowledge infrastructure for OpenAGI agents, enabling experience accumulation and cross-agent collaboration.

### Objective

- Extend the Memory class to support distributed storage (e.g., Redis or decentralized graph databases).
- Create a KnowledgeSync middleware for real-time updates between agents.
- Design access control protocols to manage shared knowledge permissions.
- Test with multi-agent workflows (e.g., Education Agent + Healthcare Agent sharing symptom-tracking data) 
- Integrate with OpenAGI’s existing Admin class for task coordination.
- Improve the TaskPlanner and Worker inference latency to adapt to this project. 

### Skills Required

Python, Database design, RAG, Langchain, Graph RAG

### Mentor

- Gurram Poorna Prudhvi
  Prudhvi has a deep expertise in NLP and building agents at scale. Having worked at Kore AI and contributed to advancements in the NLP and LLM space, he has published impactful research that has informed the development of cutting-edge AI solutions. 

## 💡 Project 4: Agent Communication Layer for Task Coordination 

- **Difficulty:** Medium (~175 hours of project). 

### Description

Build a communication protocol for OpenAGI agents to coordinate tasks, allocate resources, and persist state during complex workflows.

### Objective

- Extend the Admin class to manage inter-agent messaging 
- Develop a pub-sub system for real-time task updates.
- Implement conflict resolution algorithms for resource allocation.
- This layer manages interactions between agents, ensuring that information flows smoothly and efficiently building efficient multi agent architectures. 

### Skills Required

Python, crewAI, Langchain, Vector databases, Reddis. 

### Mentor

- Gourab Sinha
  Senior Software Developer. Gourab brings a wealth of software engineering expertise, gained from his time at AI Planet, Zetwerk, and Odessa. As a Lead Engineer at AI Planet, he focuses on developing impactful features and improving end-user experiences. 

- Mukesh Thawani 

## 💡 Project 5: Long-Term Memory Optimization 

- **Difficulty:** Medium (~175 hours of project). 

### Description

Improve OpenAGI’s existing LTM system by enhancing retrieval efficiency and memory compression.

### Objective

- Optimize the Memory class’s storage backend and feedback layer. 
- Integrate with the TaskPlanner to prioritize critical memories during task execution.
- Implement memory pruning algorithms to remove redundant data.
- Add temporal context tagging to memories (e.g., "last accessed").

### Skills Required

- Python, OpenAGI’s Memory component, Qdrant & Chroma vector database

### Mentor

- Tarun R Jain

## 💡 Project 6: Evaluating Agents

- **Difficulty:** Medium (~175 hours of project). 

### Description

Develop a comprehensive evaluation framework for OpenAGI agents to measure performance, robustness, and real-world applicability. This project will create custom metrics, automated testing pipelines, and comparative analysis tools to systematically assess agent behavior across diverse tasks.

### Objective

- Design domain-specific evaluation metrics (e.g., task success rate, hallucination frequency, response latency, resource efficiency).
- Integrate evaluation hooks into OpenAGI’s core components (TaskPlanner, Worker, Memory).
- Build an automated testing pipeline compatible with OpenAGI’s [benchmark script](https://github.com/aiplanethub/openagi/blob/main/benchmark.py).
- Implement comparative analysis tools for A/B testing between agent versions or configurations.
- Validate the framework through real-world scenarios (e.g., healthcare diagnostics vs. customer support use cases or cookbooks).

### Skills Required

Python, Evaluation Frameworks (Langchain Evaluation, RAGAS, DeepEval), CI/CD Pipelines, Statistical Testing.

### Mentor

- Gurram Poorna Prudhvi

## 💡 Project 7: Prompt Optimization using DSPy

- **Difficulty:** Easy (~90 hours of project)

### Description

Implement DSPy-based prompt optimization techniques to enhance OpenAGI's interaction quality and efficiency. This project focuses on programmatic prompt engineering to reduce manual tuning while improving task success rates.

### Objective

- Integrate DSPy framework with OpenAGI's TaskPlanner and Worker classes
- Develop optimization strategies for multi-step agent workflows
- Benchmark optimized prompts against existing configurations
- Create a library of reusable prompt signatures for common tasks
- Improve inference latency through prompt compression techniques

### Skills Required

Python, Prompt Engineering, DSPy Framework, LLM Inference

### Mentor

- Gurram Poorna Prudhvi

## 💡 Project 8: Multimodal Agents Workflows

- **Difficulty:** Easy (~90 hours of project)

### Description

Enable OpenAGI agents to process and generate multimodal outputs (text, images, audio) through enhanced workflow capabilities.

### Objective

- Extend Worker class to handle multimedia inputs/outputs
- Integrate multimodal models (e.g., CLIP, Whisper) with OpenAGI's toolset
- Develop evaluation metrics for multimodal task performance
- Create demonstration workflows combining vision+language capabilities
- Optimize resource allocation for memory-intensive multimodal tasks

### Skills Required

Python, Multimodal Models (CLIP/Whisper), Langchain, Data Processing

### Mentor

- Gourab Sinha

# OpenAGI and BeyondLLM

[OpenAGI](https://github.com/aiplanethub/openagi), developed by AI Planet, is an open-source framework designed to make human-like agents accessible to everyone, paving the way toward open agents and, eventually, AGI for all. It offers a flexible architecture that supports various tools capable of searching the internet, interacting with platforms like YouTube and GitHub, and more. OpenAGI also supports multiple Large Language Models (LLMs) in any sequence or a single LLM for all tasks, enhancing cost-effectiveness. Additionally, it features a task decomposer that assists in breaking down user queries into granular tasks, facilitating efficient multi-agent communication and reasoning capabilities. 
OPENAGI.AIPLANET.COM

[BeyondLLM](https://github.com/aiplanethub/beyondllm), also developed by AI Planet, is an open-source framework designed to streamline the development of Retrieval-Augmented Generation (RAG) and LLM applications. It offers an all-in-one toolkit for experimentation, evaluation, and deployment of RAG systems, simplifying the process with automated integration, customizable evaluation metrics, and support for various LLMs tailored to specific needs. BeyondLLM supports multiple LLMs, including ChatOpenAI, Gemini, and others, providing flexibility in model selection based on specific project requirements. It also provides observability features to monitor model performance, aiding in the optimization of RAG applications.
BEYONDLLM.AIPLANET.COM

# Miscellaneous Details

- Compute Costs: Covered by LuxProvide - Meluxina (HPC partner).
- Models: Primarily Google Gemini 2.0 Flash and Groq (for latency-sensitive tasks). Open-source models (e.g., Llama 3) will supplement domain-specific needs.

[![Contributor Guidance](https://img.shields.io/badge/Contributor_Guidance-blue?style=for-the-badge)](https://github.com/aiplanethub/Contributor-Guidance/blob/main/README.md)
