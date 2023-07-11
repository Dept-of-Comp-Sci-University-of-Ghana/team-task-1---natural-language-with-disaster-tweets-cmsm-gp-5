In this approach, we aimed to train a model to classify tweets as either disaster or not-disaster. The following steps were performed:

Reading and Exploring the Data:

The train and test datasets were read using pandas' read_csv function.
The data was explored by printing the information and the first few rows of each dataset.
Data Preprocessing:

The keyword and location columns were dropped from both the train and test datasets as they were not considered relevant for classification.
A function called cleanTxt was created to perform text cleaning on the tweets. This function removed mentions, hashtags, RTs, and URLs, and converted the text to lowercase.
The cleanTxt function was applied to the 'text' column of both the train and test datasets.
Splitting the Data:

The features (tweets) and the target variable (target column) were separated from the training dataset.
The training data was further split into training and validation sets using the train_test_split function from sklearn.
Feature Extraction:

An instance of CountVectorizer was created to convert the text data into numerical features.
The CountVectorizer was fitted on the training data to learn the vocabulary and transform the text data into a matrix of token counts.
Training and Evaluating the Model:

A Multinomial Naive Bayes model was created using MultinomialNB from sklearn.naive_bayes.
The model was trained on the training data (vectorized features) and corresponding target labels.
Predictions were made on the validation dataset using the trained model.
Evaluation metrics including accuracy, F1 score, precision, and recall were calculated using sklearn's metrics functions.

Logistic Regression Model:

A Logistic Regression model was created using LogisticRegression from sklearn.linear_model.
The logistic regression model was trained on the training data (vectorized features) and corresponding target labels.
Predictions were made on the validation dataset using the trained logistic regression model.
Evaluation metrics were calculated for the logistic regression model.
Visualizing the Results:

A confusion matrix was created using the validation predictions and actual target labels. It was visualized using seaborn's heatmap function.
A bar plot was created to visualize the counts of the target categories in the test dataset.

Main Findings
Discussion:
Both Multinomial NB and Logistic Regression models performed reasonably well in predicting the target variable. The Logistic Regression model slightly outperformed the Multinomial NB model in terms of accuracy, achieving an accuracy of 0.810 compared to 0.808.

Regarding the F1 score, which balances both precision and recall, the Logistic Regression model achieved a higher value of 0.767, while the Multinomial NB model scored 0.760. This indicates that the Logistic Regression model exhibited better overall performance in terms of the trade-off between precision and recall.

Precision represents the ability of a model to correctly classify positive instances, and recall measures the model's ability to identify all positive instances. The Multinomial NB model achieved a precision of 0.815, slightly higher than the Logistic Regression model's precision of 0.804. However, when it comes to recall, the Logistic Regression model outperformed the Multinomial NB model, achieving a recall of 0.733 compared to 0.712.