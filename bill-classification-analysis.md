### Import all necessary libraries


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

### Loading Data From the Web


```python
# Loading data from the web
# Data was downloaded from https://www.kaggle.com/datasets/ritesaluja/bank-note-authentication-uci-data?resource=download
bill_url = "https://raw.githubusercontent.com/DSCI-310-2025/dsci-310-group-python2/refs/heads/main/data/BankNote_Authentication.csv"
bill_data = pd.read_csv(bill_url)
```

### Wrangle and Clean Data


```python
# See if there are missing values
missing_values = bill_data.isna().sum().sum()
print(missing_values)
```

    0


### Summary of Dataset


```python
# Create count table
count_table = bill_data.groupby('class').size().reset_index(name='Count')
count_table['Percentage'] = 100 * count_table['Count'] / len(bill_data)

count_table
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>762</td>
      <td>55.539359</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>610</td>
      <td>44.460641</td>
    </tr>
  </tbody>
</table>
</div>



#### Table 1. Authentic (0) to Fake (1) Proportion in Dataset


```python
# Split data into training and testing sets
bill_train, bill_test = train_test_split(bill_data, test_size=0.25, stratify=bill_data['class'], random_state=1)
# Take a look at first 5 rows
bill_train.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variance</th>
      <th>skewness</th>
      <th>curtosis</th>
      <th>entropy</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>892</th>
      <td>0.21431</td>
      <td>-0.69529</td>
      <td>0.87711</td>
      <td>0.29653</td>
      <td>1</td>
    </tr>
    <tr>
      <th>465</th>
      <td>-2.69890</td>
      <td>12.19840</td>
      <td>0.67661</td>
      <td>-8.54820</td>
      <td>0</td>
    </tr>
    <tr>
      <th>57</th>
      <td>5.80700</td>
      <td>5.00970</td>
      <td>-2.23840</td>
      <td>0.43878</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1166</th>
      <td>-2.19790</td>
      <td>-2.12520</td>
      <td>1.71510</td>
      <td>0.45171</td>
      <td>1</td>
    </tr>
    <tr>
      <th>223</th>
      <td>4.64640</td>
      <td>10.53260</td>
      <td>-4.58520</td>
      <td>-4.20600</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create class proportions table
class_proportions_table = bill_train.groupby('class').size().reset_index(name='Count')
class_proportions_table['Percentage'] = 100 * class_proportions_table['Count'] / len(bill_train)

class_proportions_table

# We notice that our "class" proportions were preserved when we split the data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
      <th>Count</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>572</td>
      <td>55.587949</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>457</td>
      <td>44.412051</td>
    </tr>
  </tbody>
</table>
</div>



#### Table 2. Mean of Factors


```python
# Compute mean for each factor
means_per_column = bill_train.iloc[:, :-1].mean().reset_index()
means_per_column.columns = ['Factor', 'Mean_Value']

means_per_column
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Factor</th>
      <th>Mean_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>variance</td>
      <td>0.398695</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skewness</td>
      <td>1.837843</td>
    </tr>
    <tr>
      <th>2</th>
      <td>curtosis</td>
      <td>1.462280</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy</td>
      <td>-1.192189</td>
    </tr>
  </tbody>
</table>
</div>



#### Table 3. Maximium of Factors


```python
# Compute max for each factor
max_per_column = bill_train.iloc[:, :-1].max().reset_index()
max_per_column.columns = ['Factor', 'Maximum_Value']

max_per_column
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Factor</th>
      <th>Maximum_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>variance</td>
      <td>6.8248</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skewness</td>
      <td>12.6247</td>
    </tr>
    <tr>
      <th>2</th>
      <td>curtosis</td>
      <td>17.6772</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy</td>
      <td>2.1625</td>
    </tr>
  </tbody>
</table>
</div>



#### Table 4. Minimum of Factors


```python
# Compute min for each factor
min_per_column = bill_train.iloc[:, :-1].min().reset_index()
min_per_column.columns = ['Factor', 'Minimum_Value']

min_per_column
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Factor</th>
      <th>Minimum_Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>variance</td>
      <td>-7.0421</td>
    </tr>
    <tr>
      <th>1</th>
      <td>skewness</td>
      <td>-13.6779</td>
    </tr>
    <tr>
      <th>2</th>
      <td>curtosis</td>
      <td>-5.2861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>entropy</td>
      <td>-8.5482</td>
    </tr>
  </tbody>
</table>
</div>



### Visualizations of the Dataset

### Figure 1. Distribution of Variance Grouped By Authentic or Fake


```python
# Plot histogram of variance column distribution

plt.figure(figsize=(8, 6))
sns.histplot(data=bill_train, x='variance', hue='class', element='bars', bins=30, kde=True)
plt.xlabel("Variance")
plt.ylabel("Value")
plt.title("Distribution of Variance Labeled as 0 or 1")
plt.legend(title="Class", labels=["1 (Fake)", "0 (Authentic)"])
plt.show()
```


    
![png](output_19_0.png)
    


### Figure 2. Distribution of Skewness Grouped By Authentic or Fake


```python
# Plot histogram of skewness column distribution

plt.figure(figsize=(8, 6))
sns.histplot(data=bill_train, x='skewness', hue='class', element='bars', bins=30, kde=True)
plt.xlabel("Skewness")
plt.ylabel("Value")
plt.title("Distribution of Skewness Labeled as 0 or 1")
plt.legend(title="Class", labels=["1 (Fake)", "0 (Authentic)"])
plt.show()
```


    
![png](output_21_0.png)
    


### Figure 3. Distribution of Curtosis Grouped By Authentic or Fake


```python
# Plot histogram of curtosis column distribution

plt.figure(figsize=(8, 6))
sns.histplot(data=bill_train, x='curtosis', hue='class', element='bars', bins=30, kde=True)
plt.xlabel("Curtosis")
plt.ylabel("Value")
plt.title("Distribution of Curtosis Labeled as 0 or 1")
plt.legend(title="Class", labels=["1 (Fake)", "0 (Authentic)"])
plt.show()
```


    
![png](output_23_0.png)
    


### Figure 4. Distribution of Entropy Grouped By Authentic or Fake


```python
# Plot histogram of entropy column distribution

plt.figure(figsize=(8, 6))
sns.histplot(data=bill_train, x='entropy', hue='class', element='bars', bins=30, kde=True)
plt.xlabel("Entropy")
plt.ylabel("Value")
plt.title("Distribution of Entropy Labeled as 0 or 1")
plt.legend(title="Class", labels=["1 (Fake)", "0 (Authentic)"])
plt.show()
```


    
![png](output_25_0.png)
    


##### Summary of Visualizations

The entropy histograms (Figure 4) have very similar distributions for authentic and fake bills as they both have their modes around 0. Since the distributions are very similar, it is very unlikely that entropy is a driving factor is determining authentic or fake bills. The same can be said for the curtosis histogram (Figure 3) as authentic and fake bills have similar distributions with their modes around 0 as well. Curtosis is also unlikely to be a driving force. The variance distributions (Figure 1) are very different. Both authentic and fake bills have a bell curve distribution with the fake bills having a mode around -2.5 and the authentic bills having a mode around 4. This suggests that variance is a very strong driving force in determining whether a bill is authentic or fake. The skewness histograms (Figure 2) have somewhat different distributions. The fake bills have their mode at around 2.5 while the authentic bills have their mode at around 8. The distributions overlap quite a bit suggesting that skewness is not as big of a driving factor as variance. These histograms will allow us to understand what we should expect with our unknown bill once the bill is identified. We can use these histograms to see if our expectations line up with the unknown bill.

## Data Analysis

### Import all necessary libraries


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report

```

### Cross validation

We tried standard scaler but got better results withought it.


```python
X_train = bill_train.drop('class', axis=1)  # features
y_train = bill_train['class']              # target

neighbors = range(1, 26)  # testing 1 to 25 neighbors
cv_scores = []

for k in neighbors:
    # Update the pipeline parameter for number of neighbors
    
    KNN = KNeighborsClassifier(n_neighbors= k)
    
    # Perform 5-fold cross-validation and record the mean accuracy
    scores = cross_val_score(KNN, X_train, y_train, cv=2)
    cv_scores.append(scores.mean())

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(neighbors, cv_scores, marker='o')
plt.xlabel('Number of Neighbors')
plt.ylabel('Mean CV Accuracy')
plt.title('KNN Accuracy vs. Number of Neighbors')
plt.grid(True)
plt.show()
```


    
![png](output_33_0.png)
    


### Results

seems like even with one neighbour we are getting the best result, so let's go ahead and use n_neighbours = 1 in our pipeline

### Applying our findings on test data


```python
X_test = bill_test.drop('class', axis=1)  # features
y_test = bill_test['class']              # target

FinalKNN = KNeighborsClassifier(n_neighbors= 1)

FinalKNN.fit(X_train,y_train)

FinalKNN.score(X_test,y_test)

```




    1.0



### Results
As we can see, we achieve score of 1. Which means we with no error we are predicting everything right. Lets explore that a bit more in terms of numbers of correct predictions. 

### Confusion Matrix


```python
# Make predictions on the test set
y_pred = FinalKNN.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Optionally, print a classification report for additional metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

    Confusion Matrix:
    [[190   0]
     [  0 153]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00       190
               1       1.00      1.00      1.00       153
    
        accuracy                           1.00       343
       macro avg       1.00      1.00      1.00       343
    weighted avg       1.00      1.00      1.00       343
    


### Interpretation:*  
Every test sample was classified correctly. There are no misclassifications, which implies that the model perfectly distinguishes between the two classes. 
### Key Metrics

- **Precision:**  
  - Class 0: 1.00  
  - Class 1: 1.00  
  Indicates that every instance predicted as a specific class was correct.

- **Recall:**  
  - Class 0: 1.00  
  - Class 1: 1.00  
  Indicates that the model successfully identified all instances for each class.

- **F1-Score:**  
  - Class 0: 1.00  
  - Class 1: 1.00  
  The harmonic mean of precision and recall, confirming overall perfect performance.

- **Support:**  
  - Class 0: 190 instances  
  - Class 1: 153 instances  
  The number of actual instances in each class within the test set.

- **Accuracy:**  
  - Overall: 1.00 (100%)  
  All 343 test samples were classified correctly.

## Conclusion

The classifier achieved **100% accuracy** on the test set, with perfect precision, recall, and F1-scores for both classes. 

