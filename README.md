# Багатокласова класифікація типу шахрайства

Дослідження методів багатокласової класифікації для визначення типу шахрайства
на основі набору даних Bank Transaction Fraud Detection.

## Набір даних

Джерело: https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset

## Структура

```
fraud_type_classification/
├── data/
│   └── bank_fraud.csv
├── visualizations/
│   ├── eda/              графіки EDA
│   └── results/          матриці невідповідностей, t-SNE та таблиці
├── reports/              звіти по класифікації у форматі CSV
├── models/               збережені навчені моделі 
├── data_analysis.py      аналіз набору даних (EDA)
├── preprocessing.py      підготовка, балансування та кодування даних
├── models.py             навчання та оцінка методів класифікації
├── visualization.py      візуалізація підсумкових результатів
├── main.py               запуск 
└── requirements.txt
```

