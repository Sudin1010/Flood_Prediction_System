# 🌊 Flood Risk Prediction System

A machine learning-based system to predict flood risk (Low / Medium / High) using environmental and infrastructural factors.

---

## 📌 Objective

To develop a flood prediction system using machine learning techniques and implement the final model from scratch as per academic requirements.

---

## 📊 Dataset

- Dataset contains multiple environmental, climatic, and infrastructure-related features
- Example features:
  - MonsoonIntensity
  - ClimateChange
  - DamsQuality
  - DrainageSystems
  - Deforestation
  - Landslides
- Target:
  - FloodProbability

---

## ⚙️ Project Workflow

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing
   - Handling missing values
   - Feature scaling using StandardScaler
3. Model Comparison
   - Logistic Regression
   - Decision Tree
   - Random Forest
4. Final Model Selection
   - Logistic Regression (best performance)
5. Implementation from Scratch
6. Model Evaluation
7. Streamlit UI

---

## 🧠 Model Details

- Final Model: Logistic Regression
- Implemented from scratch using:
  - Sigmoid function
  - Gradient descent
  - Binary cross-entropy loss
- Input data is scaled using StandardScaler

---

## 📈 Output

The system predicts:

- 🟢 Low Flood Risk
- 🟡 Medium Flood Risk
- 🔴 High Flood Risk

Along with probability score.

## 📊 Flood Risk Classification

The model predicts flood risk as a probability (0 to 1) using a Logistic Regression model implemented from scratch.

Based on the predicted probability, flood risk is categorized as:

| Probability Range | Risk Level |
|------------------|-----------|
| 0.00 – 0.35      | 🟢 Low Flood Risk |
| 0.35 – 0.70      | 🟡 Medium Flood Risk |
| 0.70 – 1.00      | 🔴 High Flood Risk |

### Interpretation

- **Low Risk (0.00–0.35):**
  Minimal flood likelihood. Regular monitoring is sufficient.

- **Medium Risk (0.35–0.70):**
  Moderate flood possibility. Preventive measures and planning are recommended.

- **High Risk (0.70–1.00):**
  High probability of flood occurrence. Immediate preparedness and mitigation actions are required.

### Note

The prediction is based on the **combined influence of all environmental, climatic, and infrastructure factors**. Even moderate values across multiple features can lead to higher flood risk due to cumulative impact.

---

## 🖥️ Streamlit UI

Interactive dashboard where users can:
- Input environmental parameters
- Get real-time flood risk prediction

---

## 🚀 How to Run

### 1. Clone repository

```bash
git clone <your-repo-link>
cd Flood-Prediction-System