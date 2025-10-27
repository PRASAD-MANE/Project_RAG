
# Crowdfunding Insights with Open-Source RAG

## 1. Introduction

This repository contains the full implementation of a **Retrieval-Augmented Generation (RAG) system** built to analyze open-source crowdfunding campaign data.
The purpose of this project is to demonstrate how modern language models can be combined with vector retrieval techniques to derive contextual insights from structured and unstructured data.

This work was developed as part of the **“Crowdfunding Insights with Open-Source RAG”** assignment, which required:

* Designing an ethically sourced data pipeline,
* Building a retrieval-based knowledge system,
* Justifying all open-source components used,
* Providing an interactive interface for exploration.

---

## 2. Objectives

* Collect and clean a publicly available crowdfunding dataset.
* Generate textual “documents” summarizing patterns and statistics from the dataset.
* Implement a retrieval layer using **ChromaDB** and **text embeddings**.
* Integrate an open-source LLM (**Gemini 2.5 Flash**) for contextual question answering.
* Build an intuitive Streamlit UI so end-users can interact with the system in natural language.

---

## 3. Data Acquisition and Ethical Use

### 3.1 Source

* **Dataset:** “Crowdfunding Campaign Data” from [Kaggle Datasets](https://www.kaggle.com/datasets).
* **Fields used:**
  `project_id`, `name`, `category`, `goal`, `pledged`, `backers`, `country`, `description`, `launched_date`.
* **License:** Open for educational and non-commercial research use.

### 3.2 Ethical Considerations

* No personal or sensitive user data is included.
* The dataset is publicly released under Kaggle’s open data terms.
* All analysis is for research and learning; no redistribution of private content occurs.

---

## 4. Data Preprocessing

### 4.1 Cleaning Steps

1. **Load CSV:** Read using `pandas`.
2. **Handle Missing Values:**

   * Removed rows with missing campaign names or descriptions.
   * Filled numeric gaps with column means.
3. **Normalize Text:**

   * Converted to lowercase.
   * Removed punctuation and special characters.
4. **Feature Selection:**
   Retained only fields relevant for semantic analysis (`name`, `category`, `goal`, `pledged`, `description`, `backers`, `country`).
5. **Sampling:**
   Selected **2000 random rows** to stay within the free-tier API quota of Gemini while preserving diversity.

### 4.2 Output

The cleaned dataset was stored in `data/cleaned_crowdfunding.csv`.

---

## 5. Document Generation

### 5.1 Motivation

RAG systems operate on *documents*. Each document should contain meaningful, semantically coherent text.
Since each crowdfunding record is short, we combined and summarized groups of records into **10 large documents**, each containing **at least 400 words**.

### 5.2 Method

1. Grouped 2000 rows by category or thematic clusters.
2. Used Gemini 2.5 Flash (via its API) to generate textual summaries describing patterns such as:

   * Average funding goal and success rate per category.
   * Common features among successful campaigns.
   * Trends by geography.
3. Stored the resulting documents as text files (`generated_docs/doc_1.txt` … `doc_10.txt`).

### 5.3 Example Topics

* *Technology campaigns and innovation trends*
* *Arts & culture funding patterns*
* *Charitable causes and donor behavior*
* *Regional success rates and campaign lengths*

Each document serves as a contextual knowledge block for the RAG retriever.

---

## 6. RAG Pipeline Architecture

```
+--------------------------+
|   Crowdfunding Dataset   |
+-----------+--------------+
            |
            v
+--------------------------+
| Data Cleaning & Sampling |
+-----------+--------------+
            |
            v
+--------------------------+
| Document Generation (LLM)|
+-----------+--------------+
            |
            v
+--------------------------+
| Embedding Model (Gemini) |
+-----------+--------------+
            |
            v
+--------------------------+
| Vector Store (ChromaDB)  |
+-----------+--------------+
            |
            v
+--------------------------+
|  Retrieval + LLM Answer  |
+-----------+--------------+
            |
            v
+--------------------------+
| Streamlit User Interface |
+--------------------------+
```

---

## 7. Component Details

### 7.1 Embedding Model

**Model:** `text-embedding-004` (Gemini API)
**Justification:**

* High semantic accuracy for sentence-level embeddings.
* Free-tier accessible.
* Managed by Google; lightweight and reliable.

### 7.2 Vector Store

**Tool:** [ChromaDB](https://www.trychroma.com)
**Why ChromaDB?**

* Open-source, simple Python integration.
* Fast cosine-similarity search.
* Persistent or in-memory storage options.

### 7.3 Retrieval Mechanism

* Convert user query → embedding.
* Search top-K (default = 3) similar documents.
* Feed those documents to the LLM prompt as *context*.

### 7.4 Language Model

**Model:** `Gemini 2.5 Flash`
**Rationale:**

* Free-tier availability.
* Fast response and strong reasoning.
* Ideal for small-scale RAG experiments.

### 7.5 Contextual Answering Logic

```python
def get_insight_from_rag(query: str) -> str:
    context_docs = retrieve_context(query)
    prompt = f"Use the context below to answer:\n{context_docs}\n\nQuestion: {query}"
    return gemini_generate(prompt)
```

If no relevant document is found, the model replies:

> “The answer cannot be found within the provided dataset.”

---

## 8. Streamlit Application

The Streamlit interface provides a clean and professional user experience.

**Features**

* Text input for natural language queries.
* Adjustable Top-K context parameter.
* Displays retrieved context and generated answer.
* Automatically handles API connection and vector store initialization.

**Run Command**

```bash
streamlit run src/app.py
```

---

## 9. Project Structure

```
Project_RAG/
│
├── data/
│   ├── crowdfunding_data.csv
│   └── cleaned_crowdfunding.csv
│
├── generated_docs/
│   ├── doc_1.txt
│   ├── ...
│   └── doc_10.txt
│
├── src/
│   ├── app.py
│   ├── data_cleaning.py
│   ├── document_generator.py
│   ├── rag_service.py
│
├── requirements.txt
├── README.md
└── .env           # contains GEMINI_API_KEY
```

---

## 10. Installation and Execution Guide

### Step 1 – Clone Repository

```bash
git clone https://github.com/<username>/Crowdfunding-RAG.git
cd Crowdfunding-RAG
```

### Step 2 – Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### Step 3 – Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Configure API Key

**Option 1:** Environment variable

```bash
set GEMINI_API_KEY=your_key_here
```

**Option 2:** Temporary (for testing)
Add in `src/app.py`:

```python
os.environ["GEMINI_API_KEY"] = "your_key_here"
```

### Step 5 – Run the App

```bash
streamlit run src/app.py
```

### Step 6 – Interact

Enter queries such as:

* “What determines a successful crowdfunding campaign?”
* “Average funds raised in technology projects?”
* “Which category has the most backers?”
* “Trends in art and culture campaigns.”

---

## 11. Test Cases and Example Queries

| Test No | Query                                              | Expected Behavior                                                                               |
| ------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| 1       | “Which campaigns raised the most funds overall?”   | Retrieves summary from financial insights doc; LLM gives average or examples of top categories. |
| 2       | “What makes technology campaigns successful?”      | Pulls from the tech trends document and lists contributing factors.                             |
| 3       | “Average goal amount for health-related projects.” | Returns numeric estimate derived from summaries.                                                |
| 4       | “Do cultural campaigns attract many backers?”      | References arts & culture document.                                                             |
| 5       | “Which countries lead in crowdfunding?”            | Uses regional document to outline trends.                                                       |
| 6       | “Are short campaigns more likely to succeed?”      | Answers using context from success-pattern document.                                            |
| 7       | “Give insight on donor motivation.”                | Fetches from charity-focused document.                                                          |
| 8       | “Can you find info about AI campaigns?”            | If absent, system should respond that context is unavailable.                                   |

---

## 12. Troubleshooting

| Problem                                               | Cause                     | Solution                                                                      |
| ----------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------- |
| `Please set your GEMINI_API_KEY environment variable` | Key missing               | Add key to environment or `app.py`.                                           |
| `google.api_core.exceptions.ResourceExhausted: 429`   | Exceeded free quota       | Wait 24 hours or switch to local embedding model (e.g. SentenceTransformers). |
| Streamlit not opening in browser                      | Firewall or port conflict | Run with `streamlit run src/app.py --server.port 8502`.                       |
| Empty responses                                       | Context docs not found    | Check if generated documents exist in `generated_docs/`.                      |

---

## 13. Limitations

* **API Quota:** Gemini free tier restricts embedding calls and requests per minute.
* **Dataset Sample:** Only 2000 rows used due to token and cost constraints.
* **LLM Bias:** Responses rely on model interpretation; factual accuracy depends on context.
* **Compute:** Designed for educational scale, not large-scale production deployment.

---

## 14. Future Work

1. Replace Gemini embedding with open-source **SentenceTransformers** model (`all-MiniLM-L6-v2`).
2. Extend to full dataset for higher coverage.
3. Integrate vector visualizations with **Plotly** or **Altair** inside Streamlit.
4. Implement fine-tuning or prompt-engineering for domain-specific insights.
5. Add REST API endpoints using **FastAPI** for programmatic access.

---

## 15. Justification Summary

| Component           | Choice                          | Reason                            |
| ------------------- | ------------------------------- | --------------------------------- |
| **Dataset**         | Kaggle – Crowdfunding Campaigns | Public, ethical, diverse          |
| **LLM**             | Gemini 2.5 Flash                | Free, performant, API-based       |
| **Embedding Model** | `text-embedding-004`            | Accurate, lightweight             |
| **Vector Store**    | ChromaDB                        | Simple, open-source               |
| **Interface**       | Streamlit                       | Quick deployment and UX           |
| **Document Count**  | 10 (≥ 400 words each)           | Balance between context and quota |

---

## 16. Example Outputs

**Query:** “Which campaign categories have the highest success rate?”

**Context (retrieved excerpt):**

> Technology campaigns achieved 120 % of their funding goal on average, while food and beverage initiatives reached 95 % …

**Generated Insight:**

> Based on the dataset, technology and design campaigns show the highest success rates, often exceeding their original goals. Social and environmental causes attract steady support but have moderate completion ratios.

---
Example:
![alt text](image.png)
## 17. Conclusion

This project demonstrates the complete pipeline of a **Retrieval-Augmented Generation (RAG)** system using only open-source and free-tier components.
By combining cleaned crowdfunding data, ChromaDB vector retrieval, and Gemini 2.5 Flash LLM reasoning, it provides a practical framework for **context-driven analytics** on real-world datasets.

The repository showcases an end-to-end approach to:

* Data cleaning and transformation,
* Intelligent document creation,
* Semantic embedding and retrieval, and
* Contextual response generation through a professional Streamlit interface.
## ⚙️ Setup Instructions
 Clone the repo:
   ```bash
   git clone https://github.com/PRASAD-MANE/Project_RAG.git
   cd Project_RAG