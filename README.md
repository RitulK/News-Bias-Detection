# 📰 News Bias Detection & Semantic Similarity Analysis

This project aims to **automatically detect political bias** in news articles and **group similar articles** covering the same events using fine-tuned transformer models. It includes three core ML modules:  
1. **MPNET-based Similarity Detection Model**  
2. **LLaMA 3-based Bias Detection Model**  
3. **Fine-Tuned BERT Model for Bias Classification Enhancement**

The project also includes a fully functional **web platform** built with **Next.js**, a **FastAPI backend**, and **MongoDB** for storage and retrieval.

---

## 📌 Features

- 🔍 **Semantic Similarity Model** (MPNET)
- 🏛️ **Political Bias Detection Models** (LLaMA 3.2B & fine-tuned BERT)
- 🌐 **Web Interface** using Next.js
- 🚀 **FastAPI-based API** for model and DB communication
- 🗃️ **MongoDB** database for storing articles and metadata
- 📊 **Confusion Matrices** & Evaluation Stats for all models
- ⚙️ **End-to-End pipeline** for real-time bias analysis and event clustering

---

## 🧠 Technologies Used

| Layer        | Tech Stack                        |
|--------------|-----------------------------------|
| Frontend     | **Next.js**, TailwindCSS          |
| Backend API  | **FastAPI**                       |
| Database     | **MongoDB**, pymongo              |
| ML Models    | `sentence-transformers`, `transformers`, `datasets`, `torch`, `BERT`, `LLaMA` |
| Training     | MPNET + STSB Dataset (`stsb_multi_mt`) |
| Bias Models  | LLaMA 3.2B + Fine-Tuned BERT + custom political dataset |
| Deployment   | Can be containerized using Docker |

---
Application:
![image](https://github.com/user-attachments/assets/2df81d8e-77f3-4ad9-acc0-9a86ce30e87c)
![image](https://github.com/user-attachments/assets/59750ec4-3bcc-4b44-8f91-10a4bbb6937e)
![image](https://github.com/user-attachments/assets/eeb2249e-0403-47fd-8cc3-e153be3e7d80)
![image](https://github.com/user-attachments/assets/fe5d77bd-d7ad-4341-b72c-d87419021c97)
![image](https://github.com/user-attachments/assets/778ca41e-cf5f-41fa-90bd-d919a5ff2af8)
![image](https://github.com/user-attachments/assets/63b0ab20-9138-44cf-9142-c16d45c95ad6)


## 🛠️ Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/news-bias-detection.git
cd news-bias-detection
📦 Backend (API + Models)
Setup FastAPI
bash
Copy
Edit
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
Requirements (backend/requirements.txt)
txt
Copy
Edit
fastapi
uvicorn
pymongo
transformers[torch]
sentence-transformers

Datasets
⚙️ Model Training
🔧 MPNET - Similarity Detection
Base Model: mpnet-base
Dataset: PhilipMay/stsb_multi_mt (German subset)

Loss Function: CosineSimilarityLoss

Eval Metrics: Pearson: 0.7641, Spearman: 0.7619

🔧 LLaMA 3 & Fine-Tuned BERT - Bias Detection
LLaMA 3.2B used for deep contextual understanding.
BERT model fine-tuned on labeled political datasets.
Multi-model fusion boosts classification performance.

Evaluation Results

Similarity Detection Model:

Metric Score:
Pearson Correlation	0.7641
Spearman Correlation	0.7619

Bias Detection

Model Observation
LLaMA 3.2B	Strong performance on ideological separation
Fine-Tuned BERT	Helps with label refinement and stability

Functional Modules
Module	Description
API - FastAPI-based middleware for DB and model inference
Frontend - Next.js app for displaying article clusters and biases
Similarity Detection - MPNET fine-tuned for grouping related articles
Bias Classification - LLaMA 3.2B and fine-tuned BERT for political leanings
Pipeline - Handles preprocessing, inference, and storage

🌐 Frontend Setup (Next.js)
bash
Copy
Edit
cd frontend
npm install
npm run dev

This will launch the platform where:
1. Users can browse events
2. See grouped articles
3. View political bias predictions



📚 References :
1. STS Benchmark Dataset
2. The Media Bias Taxonomy
3. MAGPIE - Media Bias Analysis
4. MBIB - Media Bias Identification Benchmark

🧑‍💻 Authors
Ritul Kulkarni
Anish Ketkar
Reach out for collaboration or implementation queries!
