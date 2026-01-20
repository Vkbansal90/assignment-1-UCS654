by vishesh kumar bansal (102316085)
## Dataset and Imbalance
The dataset contains a `Class` column with two values:
- 0 = legitimate transaction
- 1 = fraudulent transaction

The minority class occurs very rarely, which makes it harder for models to learn useful patterns. A bar chart is plotted to show the imbalance before any processing.

## Balancing Strategy
Before training any models, the dataset is balanced using SMOTE (Synthetic Minority Oversampling Technique). SMOTE creates synthetic samples for the minority class instead of simply duplicating existing ones. This helps reduce bias and gives the model more chances to learn the characteristics of fraudulent transactions. After balancing, a second bar chart is shown to confirm the balanced class distribution.

The balancing is done before the train-test split to avoid extremely skewed training folds.

## Sampling Techniques
After splitting the balanced dataset, five different sampling techniques are applied only on the training portion:

- SMOTE (Oversampling)
- Random Oversampling (ROS)
- Random Undersampling (RUS)
- NearMiss (Undersampling)
- Tomek Links (Cleaning-based sampling)

These techniques adjust the distribution of classes in different ways. Oversampling adds minority samples, undersampling removes majority samples, and cleaning methods like Tomek Links remove borderline samples that may cause overlap between classes.

The reason for doing sampling only on the training data is to avoid data leakage and keep the test set representative.

## Models
Five machine learning models are used:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- XGBoost

These models were chosen because they cover different learning paradigms (linear, distance-based, tree-based, and boosting). Each sampling technique is tested with all models for comparison.

## Evaluation
Accuracy is used as the comparison metric. Although accuracy is not always ideal for fraud detection, it is useful here for comparing multiple sampler-model combinations in a controlled setup. The test set remains the same for every experiment.

A result table is created to show how each model performs under different sampling techniques. After that, the best sampler for each model and the best model for each sampler are identified. Finally, the overall best combination is reported.

## Key Observations
Some general patterns noted from the experiments:
- Balancing the dataset before training significantly improves results compared to leaving it imbalanced.
- Oversampling methods (such as SMOTE and ROS) tend to perform better for fraud detection tasks compared to pure undersampling.
- Undersampling reduces dataset size and can remove useful information, which sometimes harms performance.
- Random Forest and XGBoost usually perform better on structured datasets and appear to benefit the most from balanced training sets.
- Cleaning techniques like Tomek Links help reduce overlapping noise around decision boundaries.
## Results

After balancing the dataset using SMOTE and applying five different sampling techniques on the training data, each sampler was tested with five models. The accuracy table below summarizes the performance:

### Accuracy Table (in %)

| Model | SMOTE | ROS | RUS | NEAR | TOMEK |
|-------|-------|------|------|-------|--------|
| Logistic Regression | 91.62 | 91.62 | 91.62 | 91.62 | 91.62 |
| KNN | 85.60 | 85.60 | 85.60 | 85.60 | 85.60 |
| Decision Tree | 97.12 | 97.64 | 97.64 | 97.91 | 97.38 |
| Random Forest | **98.95** | 98.69 | 98.69 | 98.69 | 98.69 |
| XGBoost | 98.43 | 98.43 | 98.43 | 98.43 | 98.43 |

---

### Best Sampling Technique per Model

| Model | Best Sampling |
|-------|---------------|
| Logistic Regression | SMOTE |
| KNN | SMOTE |
| Decision Tree | NearMiss |
| Random Forest | SMOTE |


### Best sampling technique per model
- Logistic Regression → SMOTE  
- KNN → SMOTE  
- Decision Tree → NearMiss  
- Random Forest → SMOTE  
- XGBoost → SMOTE  

### Best model per sampling technique
- SMOTE → Random Forest  
- ROS → Random Forest  
- RUS → Random Forest  
- NearMiss → Random Forest  
- Tomek Links → Random Forest  

### Best performing combination
- Model: Random Forest  
- Sampling: SMOTE  
- Accuracy: 98.95%  

---

## Discussion of Results

From the experiment, Random Forest consistently performed the best across all sampling techniques, indicating that it is well-suited for the structure of this dataset. SMOTE emerged as the best sampling strategy for most models, likely because oversampling retains information compared to undersampling, which discards data.  

NearMiss showed strong performance only for Decision Tree, suggesting that boundary-focused undersampling can occasionally help simpler tree models learn more distinctive patterns. Tomek Links performed similarly to undersampling but did not surpass SMOTE.

Overall, handling class imbalance before modeling significantly improved results. Without balancing, the minority fraud class would be mostly ignored by the classifiers, even if the raw accuracy appeared high.

---

## Conclusion

Balancing the dataset followed by sampling-based experimentation provided a clearer picture of how different preprocessing strategies interact with different models. Random Forest combined with SMOTE produced the highest accuracy and represents the most effective combination for this dataset based on the metrics used.
