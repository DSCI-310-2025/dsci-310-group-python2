# Authentic vs Fake Bank Note Classification

## Summary

The goal of this project is to determine whether a given random bank note can be accurately classified as authentic or fake. Through the use of a training set extracted from this dataset, we were able to create a KNN classification model and test our testing set to determine our accuracy. We then ran our unknown label through our model and created visualizations to understand our bank note better.

### Steps to Run the Analysis

First clone the repository: <br>
`git clone https://github.com/DSCI-310-2025/dsci-310-group-python2`

Run a git terminal and change the directory to the cloned repository: <br>
`cd dsci-310-group-python2`

Build a new docker instance: <br>
`docker build -t bill-analysis .`

Run the docker instance: <br>
`docker run -it --rm -v "${PWD}":/home/joyvan -p 8888:8888 bill-analysis`

Go to http://localhost:8888/ and make changes via the jupyter notebook

#### List of Dependencies Needed to Run

- pandas 2.2.3
- matplotlib 3.10.1
- seaborn 0.13.2
- scikit-learn==1.6.1

#### Licenses

- MIT License

#### Contributers

- Danny Pirouz
- William Ho
- Sayana Imash
- Arad Sabet