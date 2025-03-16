# Authentic vs Fake Banknote Classification

## Summary

The goal of this project is to determine whether a number of random bank notes can be accurately classified as authentic or fake. Through the use of a training set extracted from this dataset, we were able to create a KNN classification model and test our testing set to determine our accuracy. We then ran our unknown bank notes through our model and created visualizations to understand our bank notes better.

### Steps to Run the Analysis

First, run a git terminal and clone the repository: <br>
`git clone https://github.com/DSCI-310-2025/dsci-310-group-python2`

Change the current directory to the cloned repository: <br>
`cd dsci-310-group-python2`

Build a new docker instance: <br>
`docker build -t bill-analysis .`

Run the docker instance: <br>
`docker run -it --rm -v "${PWD}":/home/joyvan -p 8888:8888 bill-analysis`

Go to http://localhost:8888/ and run the analysis file `bill-classification-analysis.ipynb`

### Makefile

This project is automated using a Makefile. You can run individual steps or the entire pipeline using make all. You can also execute indivisual instances by calling: 
   -  make data/original/BankNote_Authentication.csv this loads your data from a repository 
   -  make data/clean/BankNote_Authentication_Clean.csv this cleans and filters your data in roder to get   specififc variables
    - These lines of code output speciffic figures or table you will need in exploratory data analysis part
      make results/EDA/BankNote_Authentication_eda_count_table.csv \ 
      make results/EDA/BankNote_Authentication_eda_variance.png \
      make results/EDA/BankNote_Authentication_eda_skewness.png \
      make results/EDA/BankNote_Authentication_eda_curtosis.png \
      make results/EDA/BankNote_Authentication_eda_entropy.png
    - These lines of code outputs individual KNN model results:
        results/model_results/BankNote_Authentication_mr_knn_cv.png \
        results/model_results/BankNote_Authentication_mr_confusion_matrix.png  \
        results/model_results/BankNote_Authentication_mr_classification_report.txt
    - This code renders the quarto file into pdf and html:
        reports/bill-classification-analysis.pdf
        reports/bill-classification-analysis.html

#### List of Dependencies Needed to Run

- python (version 3.11)
- jupyter (version 4.0.7)
- pandas (version 2.2.3)
- matplotlib (version 3.10.1)
- seaborn (version 0.13.2)
- scikit-learn (version 1.6.1)
- ucimlrepo (version 0.0.7)
- click(version 8.1.8)

#### Licenses

This project is licensed under the licenses specified in `LICENSE.md`. This includes the `MIT License` and the `CC BY-NC-ND 4.0 License`.

#### Contributors

- Danny Pirouz
- William Ho
- Sayana Imash
- Arad Sabet
