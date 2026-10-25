# 🛡️ AI Fraud Detection in Financial Transactions

## About the Project

Financial fraud is a major problem in digital payments and online banking. Detecting suspicious transactions early can help reduce potential losses and improve security.

I built this project to explore how Machine Learning can be used to identify transactions that may require further investigation. It uses Python, Scikit-learn, and Streamlit to train a classification model and display predictions through an interactive dashboard.

This is an educational project created as part of my learning journey in Artificial Intelligence and Machine Learning.

[🚀 Open AI Fraud Detection Dashboard](https://fraud-detection-ai-cc4wgrmz3l3hnxvytncclx.streamlit.app/)


## ✨ Features

- Train a Machine Learning model to classify transaction-like data.
- Upload a CSV file containing transaction details.
- Estimate the probability that a transaction may be fraudulent.
- Display potentially suspicious transactions for further review.
- Download the prediction results as a CSV file.
- Use a simple, interactive Streamlit dashboard.

## 🛠️ Technologies Used

- **Python** — main programming language
- **Pandas** — data processing and analysis
- **NumPy** — numerical calculations
- **Scikit-learn** — Machine Learning model
- **Streamlit** — interactive web dashboard
- **Git and GitHub** — version control and project hosting

## 📂 Project Structure

```text
fraud-detection-ai/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── LICENSE
└── .gitignore
```

The model and sample CSV files are generated when you run the training script.

## ⚙️ How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/nirajyadav-cse/fraud-detection-ai.git
cd fraud-detection-ai
```

### 2. Create a virtual environment

For macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

### 4. Train the Machine Learning model

```bash
python train_model.py
```

This generates the model files and a sample CSV for testing.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 📊 How to Use It

1. Start the Streamlit dashboard.
2. Download the sample transaction CSV from the application.
3. Upload the CSV to the dashboard.
4. Review the predicted labels and fraud probability estimates.
5. Download the results for further analysis.

The CSV must contain the required numeric features: `amount`, `time`, `merchant_risk`, `distance_from_home_km`, `failed_attempts_24h`, and `is_international`.

## 🧠 Machine Learning Approach

The project uses a Random Forest classifier from Scikit-learn.

The workflow includes generating sample data, preparing the input features, splitting the data into training and testing sets, training the model, evaluating its performance, and using the trained model to make predictions.

## ⚠️ Dataset and Limitations

The current version uses **synthetic data generated for demonstration purposes**. It does not use real banking or payment transaction records.

The model is intended for learning and portfolio demonstration only. Its predictions may be incorrect and must not be used to block real transactions or make financial decisions.

A real-world fraud detection system would require appropriate real-world data, rigorous testing, privacy protections, security controls, and continuous monitoring.

## 🚀 Future Improvements

- Train and evaluate the model on an appropriate, publicly available fraud dataset.
- Improve detection performance and reduce false positives.
- Add interactive charts and transaction analytics.
- Explore model explainability techniques.
- Deploy the dashboard online using Streamlit Community Cloud.

## 👨‍💻 Author

**Niraj Yadav**

B.Tech Computer Science and Engineering — Artificial Intelligence and Machine Learning

GitHub: [nirajyadav-cse](https://github.com/nirajyadav-cse)

---

*Built as a learning project to explore Machine Learning, fraud risk analysis, and interactive Python applications.*