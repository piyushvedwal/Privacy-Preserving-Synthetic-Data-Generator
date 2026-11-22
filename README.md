# 🧠 Privacy-Preserving Synthetic Data Generator

A **comprehensive AI-powered system** for generating realistic **synthetic tabular data** with strong **privacy guarantees** using **Differential Privacy (DP)**.  
This application enables researchers, developers, and data scientists to **train, generate, and evaluate privacy-safe synthetic datasets** interactively.

---

## 🚀 Live Demo  
👉 **Try it here:** [https://piyush-synthetic-data-generator.streamlit.app/](https://piyush-synthetic-data-generator.streamlit.app/)

---

## 🧩 Key Features

### 🔧 Data Preprocessing  
- Automatic detection of numerical and categorical features  
- Missing value imputation  
- Data normalization & one-hot encoding  
- Intelligent feature type handling for mixed datasets  

### 🧠 Model Training  
- **CTGAN (Conditional Tabular GAN)** from **SDV (Synthetic Data Vault)**  
- Optional **Differential Privacy integration** using **Opacus** (DP-SGD)  
- Customizable privacy parameters (ε, δ) for tunable privacy-utility trade-off  

### 🧬 Synthetic Data Generation  
- Generate configurable numbers of **privacy-safe synthetic rows**  
- Preserve statistical relationships while hiding individual identities  

### 📊 Evaluation Metrics  

**Fidelity**  
- Kolmogorov–Smirnov (KS) Test  
- Chi-Square Test for categorical distributions  
- Correlation matrix similarity  
- Propensity score classifier  

**Privacy**  
- Membership inference attack (MIA) simulation  
- Differential Privacy (ε, δ) metrics  

**Utility**  
- Train-on-synthetic, test-on-real (ToS-ToR) ML evaluation  

### 📈 Visualization Dashboard  
- Side-by-side **distribution comparison plots**  
- **Correlation heatmaps** for real vs synthetic data  
- Intuitive metric summaries for quick interpretation  

---

## 🖥️ User Interface  
Built with **Streamlit**, offering an interactive and intuitive workflow:

1. **Upload Dataset** (CSV)  
2. **Train Model** (CTGAN + Optional Differential Privacy)  
3. **Generate Synthetic Data**  
4. **Evaluate & Visualize** results  
5. **Download Synthetic Dataset**  

---

## ⚙️ Installation

Clone the repository:
```bash
git clone https://github.com/piyushvedwal/Privacy-Preserving-Synthetic-Data-Generator.git
cd Privacy-Preserving-Synthetic-Data-Generator
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the app locally:
```bash
streamlit run app.py
```

---

## 🌍 Deployment Options

This app is ready for deployment on multiple platforms:

### ✅ **1. Streamlit Cloud (Recommended)**
- Easiest method — directly deploy via GitHub  
- Example live app: [https://piyush-synthetic-data-generator.streamlit.app/](https://piyush-synthetic-data-generator.streamlit.app/)

### 🐳 **2. Docker**
```bash
docker build -t synthetic-data-generator .
docker run -p 8501:8501 synthetic-data-generator
```

### ☁️ **3. Heroku**
Use the included `Procfile` for Heroku deployment:
```bash
git push heroku main
```

Detailed instructions available in [`DEPLOYMENT.md`](DEPLOYMENT.md).

---

## 📁 Project Structure

```
├── app.py                 # Streamlit user interface
├── data_preprocess.py     # Data cleaning, encoding, and feature detection
├── model_train.py         # CTGAN + Differential Privacy training module
├── evaluate.py            # Evaluation metrics for fidelity, privacy, and utility
├── requirements.txt       # Python dependencies
├── DEPLOYMENT.md          # Deployment guide (Streamlit, Docker, Heroku)
└── README.md              # Project documentation
```

---

## 🔒 Privacy Considerations

This tool employs **Differential Privacy (DP)** to provide **formal privacy guarantees**.  
The **privacy budget (ε)** defines the balance between privacy and data utility:

| ε (Epsilon) Range | Privacy Level | Utility |
|--------------------|---------------|----------|
| 0.1 – 1.0 | 🔐 Strong Privacy | ⚠️ Lower Utility |
| 5.0 – 10.0 | ⚖️ Moderate Privacy | ✅ Higher Utility |

Membership inference tests are used to evaluate **privacy leakage risk**.

---

## 📊 Example Output

| Metric Category | Example Metrics | Description |
|------------------|------------------|--------------|
| **Fidelity** | KS Similarity, Correlation Preservation | Measures realism of synthetic data |
| **Privacy** | Membership Inference AUC, ε (DP Budget) | Measures resistance to data leakage |
| **Utility** | ML Accuracy on Synthetic vs Real | Measures downstream usability |

Visual outputs include:
- Distribution overlap plots  
- Correlation matrix heatmaps  
- Privacy vs. Utility trade-off visualizations  

---

## 🧰 Dependencies

- **SDV (Synthetic Data Vault)**  
- **Opacus** (for Differential Privacy)  
- **PyTorch**, **scikit-learn**, **scipy**  
- **pandas**, **numpy**  
- **matplotlib**, **seaborn**  
- **streamlit**

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📜 License  
**MIT License** – Open for personal and research use.

---

## 👥 Authors  
- **[Ganesh Khetawat](https://github.com/gkganesh12)**: original author  
  🔗 [LinkedIn](https://www.linkedin.com/in/ganeshkhetawat/) • [GitHub](https://github.com/gkganesh12) • [Portfolio](http://ganeshkhetawat.unaux.com/portfolio)
- **[Piyush Vedwal](https://github.com/piyushvedwal)**: co-developer  

Original repository: [gkganesh12/Privacy-Preserving-Synthetic-Data-Generator](https://github.com/gkganesh12/Privacy-Preserving-Synthetic-Data-Generator)
