# Penguin Predictor

## Purpose

Penguin Predictor is a machine learning application that predicts the species of a penguin based on biological measurements and metadata. The project uses a Support Vector Classifier (SVC) trained on the Penguin Size dataset from Kaggle.

## Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- Seaborn
- Joblib

## Struktur

```
cuddly-fortnight/
|-- src/
|  |-- assets
|  |   |-- adelie.jpg
|  |   |-- chinstrap.jpg
|  |   |-- gentoo.jpg
|  |-- data/penguins_size.csv
|  |-- analysis.ipynb
|  |-- app.py
|  |-- data_processing.py
|  |-- main.py
|  |-- model_training.py
|-- .gitignore
|-- model.pkl
|-- report.docx
|-- req.txt
|-- scaler.pkl
```

## Installation

In a terminal, place yourself in the folder 'cuddly-fortnight'.

```
python -m venv .venv
# windows
.venv/Scripts/activate

pip install -r req.txt
```

## Run the application

```
streamlit run src/main.py
```

## Exit the application

```
ctrl + c
```

## Analysis

In the folder cuddly-fortnight/src open the file analysis.ipynb and *Run all* for an exploratory data analysis, model comparison and evaluation.