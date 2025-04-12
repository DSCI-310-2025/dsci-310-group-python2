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

Go to http://localhost:8888/ and go to the terminal and type `make all` to run the analysis. If you want to redo the analysis, you can run `make clean` to reset all the resulting files.

To view the resulting analysis, go to the `analysis` directory and open the `bill-classification-analysis.html` file. This will show the results and discussion of the bill classification analysis.

#### List of Dependencies Needed to Run

- python (version 3.11)
- jupyter (version 4.0.7)
- pandas (version 2.2.3)
- matplotlib (version 3.10.1)
- seaborn (version 0.13.2)
- scikit-learn (version 1.6.1)
- ucimlrepo (version 0.0.7)
- click (version 8.1.8)
- pytest (version 8.3.5)
- pandera (version 0.22.1)
- banknote-knn-utils (version 0.1.6)

#### Licenses

This project is licensed under the licenses specified in `LICENSE.md`. This includes the `MIT License` and the `CC BY-NC-ND 4.0 License`.

#### Contributors

- Danny Pirouz
- William Ho
- Sayana Imash
- Arad Sabet
