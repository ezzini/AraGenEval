# Call for Participation: Shared Task on Multi-Domain Detection of AI-Generated Text (M-DAIGT)**

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
- **Task Announcement:** [Conference deadline - 4 months]
- **Training Data Release:** [Conference deadline - 3 months]
- **Development Data Release:** [Conference deadline - 2.5 months]
- **Test Data Release:** [Conference deadline - 1 month]
- **System Submission Deadline:** [Conference deadline - 2 weeks]
- **Results Announcement:** [Conference deadline - 1 week]
- **System Description Paper Deadline:** [Conference deadline]

### How to Participate
Participants must:
- Register for the shared task through our website [link]
- Follow updates via our mailing list and Slack channel
- Submit their models following the evaluation protocol and deadlines

### Organizers
- **Salima Lamsiyah**, University of Luxembourg, Luxembourg  
- **Saad Ezzini**, Lancaster University, Lancaster, UK  
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
1. Covering multiple domains for cross-domain analysis
2. Conducting comprehensive evaluations across various text types and lengths
3. Using multiple AI generation sources for content diversity
4. Addressing real-world applications in media and academia

### Logistics & Support
- **Task website** hosting on GitHub
- **Submission system** via CodaLab
- **Community engagement** through Slack, mailing lists, and GitHub discussions
- **Regular updates** and participant support sessions

**Join us in tackling the challenge of AI-generated text detection!** For more details and registration, visit [website link].


