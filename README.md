# 🦠 DiseaseSpread: Predicting Disease Outbreak Using Epidemiological Data

> **SRM Institute of Science and Technology — Mini Project | Data Science | 2026**
> **SDG Goal 3: Good Health and Well-Being**

---

## 📌 Project Title
**DiseaseSpread: Multi-Factor Outbreak Risk Scoring System**

---

## 📝 Abstract
Disease outbreaks pose a significant threat to public health, particularly in densely populated regions like India. Early prediction of outbreak risk can enable timely intervention and resource allocation. This project, DiseaseSpread, develops a Multi-Factor Outbreak Risk Scoring (ORS) system using historical epidemiological data from India's IDSP surveillance program, combined with climate variables such as temperature, rainfall, and humidity, along with population density metrics. The project follows a structured data science pipeline covering data collection, preprocessing, exploratory data analysis, visualization, and predictive modeling. An Outbreak Risk Score is computed for each district by analyzing correlations between environmental conditions and historical disease incidence rates. Machine learning models including Random Forest are used to forecast case surges over a 2–4 week horizon. Results are visualized through choropleth maps and time-series plots to identify high-risk zones. This system aims to support public health authorities in proactive decision-making, directly contributing to UN SDG Goal 3: Good Health and Well-Being.

---

## ❗ Problem Statement
Early detection of disease outbreaks helps prevent large-scale spread. Traditional surveillance systems are reactive — they respond after outbreaks occur. This project aims to build a proactive system that analyzes historical health, climate, and demographic data to predict district-level outbreak risk before it escalates, providing a ranked Outbreak Risk Score to guide health resource allocation.

---

## 📊 Dataset Source
| Dataset | Source | Description |
|--------|--------|-------------|
| IDSP Weekly Bulletins | [mohfw.gov.in](https://idsp.mohfw.gov.in) | District-level disease case reports |
| ERA5 Climate Data | [Copernicus Climate Store](https://cds.climate.copernicus.eu) | Temperature, rainfall, humidity |
| Census 2011 India | [censusindia.gov.in](https://censusindia.gov.in) | Population density per district |
| WHO Disease Outbreak News | [who.int](https://www.who.int/emergencies/disease-outbreak-news) | Historical outbreak records |

---

## 🔄 Methodology / Workflow

```
Problem Identification
        ↓
Dataset Collection (IDSP + Climate + Census)
        ↓
Data Cleaning & Preprocessing
  - Handle missing weeks
  - Normalize case counts per 100K population
  - Align timestamps across datasets
        ↓
Exploratory Data Analysis
  - Correlation: climate vs disease spikes
  - Seasonal trends
  - Geographic clustering
        ↓
Data Visualization
  - Choropleth maps (district-level risk)
  - Time-series plots (outbreak waves)
  - Heatmaps & correlation matrices
        ↓
Model Development
  - Outbreak Risk Score (ORS) calculation
  - Random Forest / XGBoost for surge prediction
        ↓
Result Interpretation
  - Top 10 high-risk districts
  - Feature importance analysis
```

---

## 🛠️ Tools Used
- **Language:** Python 3.10+
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly, Folium
- **Machine Learning:** Scikit-learn, XGBoost
- **Geospatial:** GeoPandas
- **Notebook Environment:** Google Colab
- **Version Control:** Git & GitHub

---

## 📈 Results / Findings
> *(To be updated after model training and evaluation)*

- Identified top high-risk districts based on Outbreak Risk Score
- Found significant correlation between rainfall patterns and disease case spikes
- Achieved ~XX% accuracy in predicting surge events 2–4 weeks in advance
- Visualized seasonal outbreak patterns across Indian districts

---

## 👥 Team Members

| Name | Roll Number |
|------|------------|
| Subbu Vasanth K | RA2311026050079 | 
| Barathraj R | RA2311026050069|


---

## 📁 Repository Structure
```
MiniProject/
├── README.md
├── requirements.txt
├── docs/
│   ├── abstract.pdf
│   ├── problem_statement.pdf
│   └── presentation.pptx
├── dataset/
│   ├── raw_data/
│   └── processed_data/
├── notebooks/
│   ├── data_understanding.ipynb
│   ├── preprocessing.ipynb
│   └── visualization.ipynb
├── src/
│   ├── preprocessing.py
│   ├── analysis.py
│   └── model.py
├── outputs/
│   ├── graphs/
│   └── results/
└── report/
    └── mini_project_report.pdf
```



*SRM Institute of Science and Technology | Department of AIML | Batch 2027*
