# **AraGenEval**: Arabic Authorship Style Transfer and AI Generated Text Detection Shared Task 

### Hosted with [Arabic Natural Language Processing Conference (ArabicNLP 2025)](https://arabicnlp2025.sigarab.org)

## 1. Overview of the Shared Task
The rapid expansion of user-generated content across social media, digital news platforms, and online communication has created a growing demand for sophisticated Natural Language Processing (NLP) techniques to analyze and manipulate writing styles. Unlike general text style analysis [1], which focuses on broad linguistic features, **Authorship Style Transfer (AST)** aims to transform a given text to match the distinctive writing style of a specific author while preserving its original meaning [2]. This contrasts with traditional stylistic analysis, where the goal is to identify and characterize an author’s style rather than actively modify text to conform to it.  
In addition, recent advances in Arabic-based large language models have made it increasingly difficult to distinguish between human-written and AI-generated Arabic content [3]. We believe that Arabic style identification can help detect such content.  
This shared task seeks to promote research in Arabic AST, an area that remains relatively underdeveloped compared to other languages. Participants will develop models for one or more of the following subtasks:

1. Authorship Style Transfer (Text Generation) <br>
2. Authorship Identification (Multiclass Classification) <br>
3. ARATECT: Arabic AI-Generated Text Detection (Binary Classification) 

## 2. Motivation
Authorship style transfer and AI-generated text detection can be applied in various domains, including education, cultural adaptation, and social media content generation. The motivation for launching this shared task arises from the increasing presence of Arabic-language discussions on various socio-political and technological topics. Although authorship style transfer [4, 5] is explored in NLP, the Arabic domain presents distinct challenges:
- **Linguistic diversity**: Arabic exhibits significant variation, with multiple formal and dialectal forms.  
- **Contextual dependencies**: Performance shifts depending on the author style, requiring models to capture nuanced contextual cues.  
- **Limited labeled data**: Large-scale datasets for Arabic style transfer and detection are scarce.  
Our goal is to inspire researchers to tackle these challenges and enhance style transfer and detection techniques specifically for the Arabic language.

## 3. Data Collection and Creation

### 3.1 Authorship Style Transfer (Tasks 1 & 2)
- **Corpus**: 21 authors, 10 publicly accessible books each.  
- **Segmentation**: Each book divided into semantically coherent paragraphs.  
- **Parallelization**: GPT-4o mini2 rephrased selected paragraphs into a standardized formal style; pairs limited to ≤ 1900 tokens.  
- **Splits**: Train / Test / Validation as shown below.

#### Dataset Statistics

| Id | Author                  | Train | Test | Val |
|----|-------------------------|------:|-----:|----:|
|(1) | Ahmed Amin              | 2892  | 594  | 246 |
|(2) | Ahmed Taymour Pasha     |  804  | 142  |  53 |
|(3) | Ahmed Shawqi            |  596  |  46  |  58 |
|(4) | Ameen Rihani            | 1557  | 624  | 142 |
|(5) | Tharwat Abaza           |  755  | 191  |  90 |
|(6) | Gibran Khalil Gibran    |  748  | 240  |  30 |
|(7) | Jurji Zaydan            | 2762  | 562  | 326 |
|(8) | Hassan Hanafi           | 3735  |1002  | 548 |
|(9) | Robert Barr             | 2680  | 512  |  82 |
|(10)| Salama Moussa           |  984  | 282  | 119 |
|(11)| Taha Hussein            | 2371  | 534  | 253 |
|(12)| Abbas M. Al-Aqqad       | 1820  | 499  | 267 |
|(13)| Abdel Ghaffar Makawi    | 1520  | 464  | 396 |
|(14)| Gustave Le Bon          | 1515  | 358  | 150 |
|(15)| Fouad Zakaria           | 1771  | 294  | 125 |
|(16)| Kamel Kilani            |  399  | 109  |  25 |
|(17)| Mohamed H. Heikal       | 2627  | 492  | 260 |
|(18)| Naguib Mahfouz          | 1630  | 343  | 327 |
|(19)| Nawal El Saadawi        | 1415  | 382  | 295 |
|(20)| William Shakespeare     | 1236  | 358  | 238 |
|(21)| Yusuf Idris             | 1140  | 349  | 120 |

### 3.2 ARATECT: Arabic AI-Generated Text Detection (Subtask 3)
- **Human-Written Texts**: Collected from reputable Arabic news sites and verified literary sources; manually curated.  
- **AI-Generated Texts**: Produced using Arabic-compatible LLMs (e.g., Mistral, GPT-4, LLaMA) under diverse prompting strategies.  
- **Annotation**: Binary labels (human vs. AI) with domain coverage across news and literature.

## 4. Task Description

### 4.1 Subtask 1: Authorship Style Transfer
- **Goal**: Transform a formal input text into the style of a specified author while preserving semantics.  
- **Evaluation**: BLEU (primary), chrF (secondary).

### 4.2 Subtask 2: Authorship Identification
- **Goal**: Identify the author of a given text excerpt across diverse genres and periods.  
- **Evaluation**: Macro-F1 Score (primary), Accuracy (secondary).

### 4.3 Subtask 3:ARATECT
Focuses on one main domains:
- **Arabic News Text Detection (ArabicNewsGen)**
   - Full-length articles and short excerpts; genres include politics, economy, technology, sports.  
- **Evaluation**: F1-Score (primary), Accuracy (secondary).

## 5. Tentative Timeline
- **June 10, 2025**: Release of training data  
- **July 20, 2025**: Release of test data  
- **July 25, 2025**: End of evaluation cycle (test submissions close)  
- **July 30, 2025**: Final results released  
- **August 15, 2025**: Shared task papers due date  
- **August 25, 2025**: Notification of acceptance  
- **September 5, 2025**: Camera-ready versions due  
- **November 5–9, 2025**: Main Conference

## 6. Organizers’ Details
- **Shadi Abudalfa**, King Fahd University of Petroleum & Minerals  
- **Saad Ezzini**, King Fahd University of Petroleum & Minerals  
- **Ahmed Abdelali**, Saudi Data & AI Authority  
- **Salima Lamsiyah**, University of Luxembourg  
- **Hamzah Luqman**, King Fahd University of Petroleum & Minerals  
- **Mustafa Jarrar**, Hamad Bin Khalifa University / Birzeit University  
- **Mo El-Haj**, VinUniversity  
- **Abdelkader Elmahdaouy**, Mohammed VI Polytechnic University  
- **Hamza Alami**, Sidi Mohamed Ben Abdellah University  
- **Abdessamad Benlahbib**, Sidi Mohamed Ben Abdellah University  
- **Salmane Chafik**, Mohammed VI Polytechnic University  

## 7. Participation Guidelines
- For participation guidelines, please refer to [Participation Guidelines](guidelines.md).
- Comprehensive instructions for preparing and submitting your paper(s) are available at [Paper Submission Guidelines](PAPER.md).

## References
1. Hu et al. “Text style transfer: A review and experimental evaluation.” _ACM SIGKDD Explorations Newsletter_, 24(1), 2022.  
2. Shao et al. “Authorship style transfer with inverse transfer data augmentation.” _AI Open_, 5, 2024.
3. Alghamdi et al. "Distinguishing Arabic GenAI-generated Tweets and Human Tweets utilizing Machine Learning." Engineering, Technology & Applied Science Research, 14(5), 16720-16726, 2024.  
4. Patel et al. “Low-Resource Authorship Style Transfer: Can Non-Famous Authors Be Imitated?” _arXiv preprint arXiv:2212.08986_, 2022.  
5. Horvitz et al. “TinyStyler: Efficient Few-Shot Text Style Transfer with Authorship Embeddings.” _arXiv preprint arXiv:2406.15586_, 2024.  

---


### Logistics and Support
- **Website Hosting**: GitHub Pages
- **Submission System**: Codabench
- **Communication Channels**:
  - Slack workspace
  - Mailing list
  - GitHub repository
- **Regular updates and participant support**


### Stay Updated
- Official **GitHub Repository**: https://github.com/ezzini/AraGenEval
- Join our [Slack community](https://join.slack.com/t/arabicnlp2025-oqe5144/shared_invite/zt-39zsj6k6o-LP09lFRhLNuuHzkyahNRxw)
- Follow the latest **announcements and updates** on this page!

### Anti-Harassment policy
We uphold the [ACL Anti-Harassment Policy](https://www.aclweb.org/adminwiki/index.php?title=Anti-Harassment_Policy), and participants in this shared task are encouraged to reach out with any concerns or questions to any of the shared task organizers.

