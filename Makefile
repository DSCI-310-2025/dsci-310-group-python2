# ===========================
# Makefile for Banknote Classification Project
# This Makefile automates data loading, cleaning, EDA, modeling, and report generation.
# Run `make all` to execute the entire pipeline.
# ===========================

.PHONY: all clean eda model report results

all: test \
	data/original/BankNote_Authentication.csv \
	data/clean/BankNote_Authentication_Clean.csv \
 	results/EDA/BankNote_Authentication_eda_count_table.csv \
	results/EDA/BankNote_Authentication_eda_variance.png \
	results/EDA/BankNote_Authentication_eda_skewness.png \
	results/EDA/BankNote_Authentication_eda_curtosis.png \
	results/EDA/BankNote_Authentication_eda_entropy.png \
	results/model_results/BankNote_Authentication_mr_knn_cv.png \
	results/model_results/BankNote_Authentication_mr_confusion_matrix.png  \
	results/model_results/BankNote_Authentication_mr_classification_report.csv \
	reports/bill-classification-analysis.html

# run tests
test:
	pytest

# Ensure results directory exists
results:
	mkdir -p results

# Step 1: Load the data from the repository using the 01-load_data.py script
data/original/BankNote_Authentication.csv: scripts/01-load_data.py 
	python scripts/01-load_data.py --repo_id 267 --output_path=data/original/BankNote_Authentication.csv

# Step 2: Clean the loaded data using 02-clean_data.py
data/clean/BankNote_Authentication_Clean.csv: data/original/BankNote_Authentication.csv scripts/02-clean_data.py
	python scripts/02-clean_data.py --input_path=data/original/BankNote_Authentication.csv \
		--output_path=data/clean/BankNote_Authentication_Clean.csv

# Step 3: Perform EDA using 03-visualization.py
results/EDA/BankNote_Authentication_eda_count_table.csv \
results/EDA/BankNote_Authentication_eda_variance.png \
results/EDA/BankNote_Authentication_eda_skewness.png \
results/EDA/BankNote_Authentication_eda_curtosis.png \
results/EDA/BankNote_Authentication_eda_entropy.png: data/clean/BankNote_Authentication_Clean.csv scripts/03-visualization.py
	mkdir -p results/EDA
	python scripts/03-visualization.py --input_path=data/clean/BankNote_Authentication_Clean.csv --output_prefix=results/EDA/BankNote_Authentication_eda

# Step 4: Perform modeling using 04-modeling.py
results/model_results/BankNote_Authentication_mr_knn_cv.png \
results/model_results/BankNote_Authentication_mr_confusion_matrix.png  \
results/model_results/BankNote_Authentication_mr_classification_report.csv: data/clean/BankNote_Authentication_Clean.csv scripts/04-modeling.py
	mkdir -p results/model_results
	python scripts/04-modeling.py --input_path=data/clean/BankNote_Authentication_Clean.csv --output_prefix=results/model_results/BankNote_Authentication_mr

# Render quarto report in HTML and PDF
reports/bill-classification-analysis.html: analysis/bill-classification-analysis.qmd
	mkdir -p reports
	quarto render analysis/bill-classification-analysis.qmd --to html

# Clean target: Remove generated files
clean:
	rm -rf results/EDA results/model_results reports
	rm -rf reports/bill-classification-analysis.html
	rm -rf data/clean/*
	rm -rf data/original/*
	rm -rf data/eda/*
	rm -rf data/model_results/*
