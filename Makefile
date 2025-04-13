# ===========================
# Makefile for Banknote Classification Project
# This Makefile automates data loading, cleaning, EDA, modeling, and report generation.
# Run `make all` to execute the entire pipeline.
# ===========================

.PHONY: all clean EDA model report results

all: data/original/BankNote_Authentication.csv \
	data/clean/BankNote_Authentication_Clean.csv \
 	results/eda/BankNote_Authentication_EDA_count_table.csv \
	results/eda/BankNote_Authentication_EDA_variance.png \
	results/eda/BankNote_Authentication_EDA_skewness.png \
	results/eda/BankNote_Authentication_EDA_curtosis.png \
	results/eda/BankNote_Authentication_EDA_entropy.png \
	results/eda/BankNote_Authentication_EDA_p_values.csv \
	results/model_results/BankNote_Authentication_mr_knn_cv.png \
	results/model_results/BankNote_Authentication_mr_confusion_matrix.png  \
	results/model_results/BankNote_Authentication_mr_classification_report.csv \
	reports/bill-classification-analysis.html

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
results/eda/BankNote_Authentication_EDA_count_table.csv \
results/eda/BankNote_Authentication_EDA_variance.png \
results/eda/BankNote_Authentication_EDA_skewness.png \
results/eda/BankNote_Authentication_EDA_curtosis.png \
results/eda/BankNote_Authentication_EDA_entropy.png \
results/eda/BankNote_Authentication_EDA_p_values.csv: data/clean/BankNote_Authentication_Clean.csv scripts/03-visualization.py
	mkdir -p results/eda
	python scripts/03-visualization.py --input_path=data/clean/BankNote_Authentication_Clean.csv --output_prefix=results/eda/BankNote_Authentication_EDA

# Step 4: Perform modeling using 04-modeling.py
results/model_results/BankNote_Authentication_mr_knn_cv.png \
results/model_results/BankNote_Authentication_Analysis_shap_plot_1.png \
results/model_results/BankNote_Authentication_Analysis_shap_plot_2.png \
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
	rm -rf results/eda results/model_results reports
	rm -rf reports/bill-classification-analysis.html
	rm -rf data/clean/*
	rm -rf data/original/*
	rm -rf data/eda/*
	rm -rf data/model_results/*
