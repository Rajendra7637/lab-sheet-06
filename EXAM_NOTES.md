# Exam Notes: how to explain the projects

The sheet says you must **demonstrate the project in the practical exam**.
Run both projects on your own computer before the exam, read each step,
and make sure you can answer the questions below in your own words.

## The 30-second explanation

> "I built a model that predicts [student marks / house price]. I loaded the data,
> cleaned it, split it into training and testing data, built a pipeline that fills
> missing values and scales the numbers, trained several algorithms, compared them
> with [R2, MAE, RMSE / accuracy, F1], saved the best one with Joblib, and wrote a
> script that predicts for new input."

## Questions you may be asked

**Data and preprocessing**
- *Why split into train and test data?* To test the model on data it has never seen.
  Good scores on training data alone can come from memorizing it (overfitting).
- *How did you handle missing values?* Median for numbers (not affected by extreme values),
  most frequent value for categories. Done inside the pipeline.
- *Why inside the pipeline and not before the split?* The fill values and scaling numbers
  are learned from the training data only. This prevents data leakage from the test data.
- *Why scale the features?* Features with big numbers (like area in thousands) would
  dominate small ones (like bedrooms). Scaling puts them on the same level.
  Tree models do not need it, but linear models do.
- *What is one-hot encoding?* A text category becomes one 0/1 column per category,
  because models need numbers. (Location "City Centre" becomes a column that is 1 or 0.)
- *What is an outlier and how did you find it?* A value very far from the rest.
  IQR rule: below Q1 - 1.5 x IQR or above Q3 + 1.5 x IQR. (Project 2)
- *What is cross-validation?* The training data is split into 5 parts. The model trains on 4
  and tests on 1, five times, and the scores are averaged. It shows the result is stable.

**Algorithms**
- *Linear / Ridge Regression:* fits a straight-line relationship. Ridge adds a penalty
  on big coefficients to avoid overfitting.
- *Decision Tree:* asks a series of yes/no questions on the features.
- *Random Forest:* many decision trees on random parts of the data, with their answers averaged.
- *Gradient Boosting:* trees built one after another, each correcting the mistakes of the previous.
- *Logistic Regression:* despite the name, it is a classification algorithm.
- *Why did the simple linear model do so well (or not)?* Look at your results and think:
  if the real relationship is close to a straight line, a simple model is enough.

**Metrics**
- *MAE:* average size of the mistake. *MSE/RMSE:* squared mistake and its square root
  (punishes big mistakes more). *R2:* share of the variation the model explains (1 is perfect).
- *MAPE (Project 2):* mistake as a percentage of the real price.
- *Accuracy:* share of correct predictions. *Precision:* of those predicted as a class, how many are right.
  *Recall:* of those really in a class, how many were found. *F1:* balance of precision and recall.
- *Confusion matrix:* rows are the real classes, columns are the predicted classes.
- *How can you tell the model is overfitting?* Train score is much higher than test score.

**Saving**
- *What does Joblib do?* It saves the whole trained pipeline (preprocessing and model) to a file,
  so `predict.py` can load it later without training again.

## Things the examiner may ask you to change live (practice these!)

1. Change `test_size` from 0.2 to 0.3 and see how the scores change.
2. Add another algorithm to the `models` dictionary (for example `KNeighborsRegressor`).
3. Change the median to the mean in `SimpleImputer`.
4. Predict for a new input with `predict.py`.
5. Explain one graph: what does it show and what do you conclude?

Tip: if you can do these five things without help, you understand the project.
