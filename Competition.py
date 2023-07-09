import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import seaborn as sns



# Reading the datasets
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

# Exploring the data
print(df_train.info())
print(df_test.info())

# Converting to DataFrame
df_train = pd.DataFrame(df_train)
df_test = pd.DataFrame(df_test)

# Printing out the top 5 elements of the dataset
print(df_train.head())
print(df_test.head())

# Dropping the keyword and location column
df_train = df_train.drop(['keyword', 'location'], axis=1)
df_test = df_test.drop(['keyword', 'location'], axis=1)

# Printing out the new dataset
print(df_train.head())
print(df_test.head())

# Creating function to clean tweets
def cleanTxt(text):
    text = re.sub(r'@[A-Za-z0-9]+', '', text)   # Removes @mention
    text = re.sub(r'#', '',text)    # Removing the hashtags
    text= re.sub(r'RT[\s]:+', '', text)  # Removing RT
    text = re.sub(r'https?://\S+|www\.\S+', '', text) #Removing all non english characters
    text = text.lower()
    return text

# Applying the cleantext function
df_train['text'] = df_train['text'].apply(cleanTxt)
df_test['text'] = df_test['text'].apply(cleanTxt)

# Printing out the new datasets
print(df_train.head())
print(df_test.head())

# Separate the features (X_train) and the target variable (y_train) from the training dataset
X_train = df_train['text']  # Features from training dataset
y_train = df_train['target']  # Target variable from training dataset

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create an instance of the CountVectorizer to convert text data into numerical features
vectorizer = CountVectorizer()

# Convert text data into numerical features for training dataset
X_train_vectorized = vectorizer.fit_transform(X_train)

# Create an instance of the Multinomial Naive Bayes model
model = MultinomialNB()

# Train the model using the training data
model.fit(X_train_vectorized, y_train)

# Convert text data into numerical features for validation dataset
X_val_vectorized = vectorizer.transform(X_val)

# Make predictions on the validation dataset using the trained model
predictions = model.predict(X_val_vectorized)

# Calculate evaluation metrics using the actual target values from the validation dataset
accuracy = accuracy_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)


# Separate the features (X_test) from the test dataset
X_test = df_test['text']  # Features from test dataset

# Convert text data into numerical features for test dataset
X_test_vectorized = vectorizer.transform(X_test)

# Make predictions on the test dataset using the trained model
test_predictions = model.predict(X_test_vectorized)

# Creating the target column
df_test['target'] = test_predictions


# Print the evaluation metrics
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)
print("Recall:", recall)

# Visualize the Confusion Matrix
cm = confusion_matrix(y_val, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Bar plot
labels = ['Disaster', 'Not-Disaster']
counts = [len(df_test[df_test['target'] == 1]), len(df_test[df_test['target'] == 0])]

plt.figure(figsize=(6, 4))
plt.bar(labels, counts)
plt.xlabel('Categories')
plt.ylabel('Counts')
plt.title('Category Counts in df_test')
plt.show()


df_test = df_test.drop('text', axis=1)
df_test.set_index('id', inplace=True)
df_test.to_csv('sample.csv')
