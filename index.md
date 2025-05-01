# Shared Task on Multi-Domain Detection of AI-Generated Text (M-DAIGT)

### Task Overview
We invite researchers and practitioners to participate in the **Multi-Domain Detection of AI-Generated Text (M-DAIGT) Shared Task**, which focuses on detecting AI-generated text across multiple domains, specifically news articles and academic writing. With the growing prevalence of large language models, distinguishing human-written content from AI-generated text has become a critical challenge for information integrity and academic honesty.

### Subtasks
Participants are encouraged to develop models for one or both of the following subtasks:

1. **News Article Detection (NAD)**
   - Binary classification of news articles as either human-written or AI-generated
   - Evaluation on both full articles and article snippets
   - Coverage of various news genres, including politics, technology, sports, etc.

2. **Academic Writing Detection (AWD)**
   - Binary classification of academic texts as either human-written or AI-generated
   - Includes student coursework and research papers
   - Spans multiple academic disciplines and writing styles

### Dataset
The dataset for this shared task consists of:
- **Human-written content** sourced from verified news platforms and academic writing samples (with appropriate permissions)
- **AI-generated content** produced using multiple LLMs (e.g., GPT-3.5, GPT-4, Claude) with varied prompting strategies and generation parameters

**Dataset Statistics:**
- **Training set:** 10,000 samples per subtask
- **Development set:** 2,000 samples per subtask
- **Test set:** 3,000 samples per subtask
- **Balanced distribution** of human-written and AI-generated content

### Evaluation Metrics
Submissions will be evaluated using the following classification metrics:
- **Primary Metrics:** Accuracy, F1-score, Precision, Recall
- **Secondary Analysis:** Model robustness across different text lengths, writing styles, topic domains, and generation models

### Timeline

- **Training Data Ready:** March 31, 2025
- **Dev Data Ready:** May 1, 2025
- **Registration Deadline:** May 8, 2025
- **Evaluation Period:** May 8–15, 2025
- **Paper Submission Due:** June 1, 2025
- **Notification to Authors:** June 15, 2025
- **Task Overview Paper Due:** June 15, 2025
- **Camera-Ready Due:** June 30, 2025 (hard deadline; cannot be postponed)
- **Shared Task Presentation Co-located with RANLP 2025:** September 11 and September 12, 2025
  
(All deadlines are 11:59 PM UTC-12:00, "Anywhere on Earth")

### How to Participate
Participants must:
- Register for the shared task through our website 
- Follow updates via our mailing list
- Submit their models following the evaluation protocol and deadlines

### Organizers
- **Salima Lamsiyah**, University of Luxembourg, Luxembourg  
- **Saad Ezzini**, King Fahd University of Petroleum and Minerals, Saudi Arabia  
- **Abdelkader Elmahdaouy**, Mohammed VI Polytechnic University, Morocco  
- **Hamza Alami**, Sidi Mohamed Ben Abdellah University, Morocco  
- **Abdessamad Benlahbib**, Sidi Mohamed Ben Abdellah University, Morocco  
- **Samir El Amrany**, University of Luxembourg, Luxembourg  
- **Salmane Chafik**, Mohammed VI Polytechnic University, Morocco  
- **Hicham Hammouchi**, University of Luxembourg, Luxembourg
  
### Resources Provided
Participants will have access to:
- **Computing resources** for dataset generation and baseline models
- **Annotation platform** for human verification
- **Submission system** via CodaLab
- **Evaluation scripts** and submission format examples
- **Task documentation and updates** through GitHub and Slack
  
### Expected Impact
This shared task aims to:
1. **Advance research** in AI-generated text detection across multiple domains
2. **Develop real-world applications** to support news organizations and academic integrity initiatives
3. **Establish a benchmark dataset** for AI text detection research

### Baseline Systems
To support participants, we will provide:
1. **Simple Statistical Baseline** (TF-IDF + SVM)
2. **Transformer-Based Baseline** (RoBERTa)
3. **Evaluation scripts** and sample submission formats

### Novelty and Significance
This shared task differentiates itself from existing work by:
1. Covering two different domains for cross-domain analysis
2. Conducting comprehensive evaluations across various text types and lengths
3. Using multiple AI generation sources for content diversity
4. Addressing real-world applications in media and academia

### Logistics and Support
- **Website Hosting**: GitHub Pages
- **Submission System**: Codalab
- **Communication Channels**:
  - Slack workspace
  - Mailing list
  - GitHub repository
- **Regular updates and participant support**


### Stay Updated
- Official **GitHub Repository**: https://github.com/ezzini/M-DAIGT
- Join our **Slack community**: [To be announced]
- Follow the latest **announcements and updates** on this page!

### Anti-Harassment policy
We uphold the [ACL Anti-Harassment Policy](https://www.aclweb.org/adminwiki/index.php?title=Anti-Harassment_Policy), and participants in this shared task are encouraged to reach out with any concerns or questions to any of the shared task organizers.

