# Credit Card Fraud Detection: Balancing + Sampling Techniques

## Introduction
This project explores how handling class imbalance affects the performance of machine learning models in credit card fraud detection. Fraud datasets are typically highly imbalanced, where the number of legitimate transactions is far greater than fraudulent ones. If this imbalance is ignored, models tend to predict everything as the majority class and still achieve high accuracy, which is misleading for real fraud detection.

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

## Conclusion
Properly addressing class imbalance is crucial in fraud detection. Balancing the dataset early and then experimenting with sampling methods gives a better understanding of how different models behave under different data distributions. Without handling imbalance, accuracy alone becomes misleading and makes fraud detection appear easier than it actually is.

## Tools and Libraries
Python, scikit-learn, imbalanced-learn, XGBoost, pandas, numpy, matplotlib, seaborn

