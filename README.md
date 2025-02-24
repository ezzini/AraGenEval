# Multi-Domain Detection of AI-Generated Text (M-DAIGT)

## Task Description
The **Multi-Domain Detection of AI-Generated Text (M-DAIGT)** shared task focuses on identifying AI-generated content across different domains, particularly **news articles** and **academic writing**. With the rapid advancement of large language models (LLMs), distinguishing human-written and AI-generated text has become a critical challenge. This task aims to contribute to research on information integrity and academic honesty.

## Subtasks

### 1. News Article Detection (NAD)
- Binary classification of news articles as **human-written** or **AI-generated**
- Evaluation on both **full articles** and **snippets**
- Covers various genres: **politics, technology, sports**, etc.

### 2. Academic Writing Detection (AWD)
- Binary classification of academic texts as **human-written** or **AI-generated**
- Evaluation on **student coursework** and **research papers**
- Covers multiple **academic disciplines and writing styles**

## Dataset

### Data Collection and Annotation
- **Human-written content**: Sourced from verified **news websites** and academic papers with proper permissions.
- **AI-generated content**: Created using multiple LLMs (**GPT-3.5, GPT-4, Claude, etc.**) with different **prompting strategies and generation settings**.

### Dataset Statistics
| Split            | Samples per Subtask |
|----------------|------------------|
| **Training**   | 10,000           |
| **Development** | 2,000            |
| **Test**       | 3,000            |

- **Balanced distribution** of human-written and AI-generated text.

## Evaluation Metrics

### Primary Metrics
- **Accuracy**
- **F1-score**
- **Precision**
- **Recall**

### Secondary Analysis
- Model robustness across different:
  - **Text lengths**
  - **Writing styles**
  - **Topic domains**
  - **Generation models**

## Timeline

| Milestone | Timeline (Relative to Conference Deadline) |
|-----------|----------------------------------------|
| Task Announcement | -4 months |
| Training Data Release | -3 months |
| Development Data Release | -2.5 months |
| Test Data Release | -1 month |
| System Submission Deadline | -2 weeks |
| Results Announcement | -1 week |
| System Description Paper Deadline | Conference Deadline |

## Organizers
- **Salima Lamsiyah**, University of Luxembourg, Luxembourg
- **Saad Ezzini**, Lancaster University, UK
- **Abdelkader Elmahdaouy**, Mohammed VI Polytechnic University, Morocco
- **Hamza Alami**, Sidi Mohamed Ben Abdellah University, Morocco
- **Abdessamad Benlahbib**, Sidi Mohamed Ben Abdellah University, Morocco
- **Samir El Amrany**, University of Luxembourg, Luxembourg
- **Salmane Chafik**, Mohammed VI Polytechnic University, Morocco
- **Hicham Hammouchi**, University of Luxembourg, Luxembourg

## Resources Required
- **Computing resources** for dataset generation and baseline models.
- **Annotation platform** for human verification.
- **Website** for task documentation and submission (hosted on **GitHub Pages**).
- **Submission system** on **Codalab**.
- **Evaluation scripts** for scoring models.

## Expected Impact

### 1. Research Contributions
- Novel detection methods for AI-generated text.
- Insights into **cross-domain detection** challenges.
- Understanding of **LLM fingerprints** and artifacts.

### 2. Practical Applications
- AI detection tools for **news organizations**.
- Academic **integrity support systems**.

### 3. Dataset Legacy
- A **high-quality benchmark dataset** for AI text detection.
- **Cross-domain evaluation resource** for future research.

## Baseline Systems
We will provide the following **baseline models**:
1. **Statistical Baseline**: TF-IDF + SVM
2. **Transformer-Based Baseline**: RoBERTa
3. **Evaluation scripts** and **submission format examples**

## Novelty and Significance
This shared task stands out due to:
1. **Multi-domain focus** enabling cross-domain insights.
2. **Comprehensive evaluation** across various text types and lengths.
3. **Consideration of multiple AI-generation sources**.
4. **Real-world application relevance**.

## Logistics and Support
- **Website Hosting**: GitHub Pages
- **Submission System**: Codalab
- **Communication Channels**:
  - Slack workspace
  - Mailing list
  - GitHub repository
- **Regular updates and participant support**

---

### 💡 Stay Updated
- Official **GitHub Repository**: [To be announced]
- Join our **Slack community**: [To be announced]
- Follow the latest **announcements and updates** on this page!
