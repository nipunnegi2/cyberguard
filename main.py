

















# # import pandas as pd
# # import re
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score

# # # Function to clean and normalize text
# # def clean_text(text):
# #     """Clean and normalize text."""
# #     if isinstance(text, str):
# #         text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
# #         text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
# #         text = text.lower()  # Convert to lowercase
# #     return text

# # print("Loading data...", flush=True)
# # train_data = pd.read_csv("train.csv")
# # print("Data loaded.")

# # print("Preprocessing data...", flush=True)
# # train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo'].apply(clean_text)
# # train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo_cleaned'].fillna(" ")  # Replace NaN with empty strings
# # train_data['sub_category'] = train_data['sub_category'].fillna('Unknown')
# # print("Preprocessing completed.")

# # print("Handling rare categories...", flush=True)
# # category_counts = train_data['category'].value_counts()
# # valid_categories = category_counts[category_counts >= 2].index
# # train_data_filtered = train_data[train_data['category'].isin(valid_categories)]
# # print("Rare categories handled.")

# # print("Extracting features...", flush=True)
# # # Use a smaller vocabulary for testing
# # vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
# # X = vectorizer.fit_transform(train_data_filtered['crimeaditionalinfo_cleaned'])
# # y = train_data_filtered['category']
# # print("Features extracted.")

# # print("Splitting data...", flush=True)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# # print("Data split completed.")

# # print("Training model...", flush=True)
# # model = RandomForestClassifier(n_estimators=100, random_state=42)
# # model.fit(X_train, y_train)
# # print("Model training completed.")

# # print("Evaluating model...", flush=True)
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))



# # code 2

# # import pandas as pd
# # import re
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.model_selection import train_test_split
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics import classification_report, accuracy_score

# # # Function to clean and normalize text
# # def clean_text(text):
# #     """Clean and normalize text."""
# #     if isinstance(text, str):
# #         text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
# #         text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
# #         text = text.lower()  # Convert to lowercase
# #     return text

# # print("Loading data...", flush=True)
# # train_data = pd.read_csv("train.csv")

# # # Sample a smaller dataset for faster execution
# # train_data = train_data.sample(5000, random_state=42)  # Use a random sample of 5,000 rows
# # print("Data loaded.")

# # print("Preprocessing data...", flush=True)
# # train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo'].apply(clean_text)
# # train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo_cleaned'].fillna(" ")  # Replace NaN with empty strings
# # train_data['sub_category'] = train_data['sub_category'].fillna('Unknown')
# # print("Preprocessing completed.")

# # print("Handling rare categories...", flush=True)
# # category_counts = train_data['category'].value_counts()
# # valid_categories = category_counts[category_counts >= 2].index
# # train_data_filtered = train_data[train_data['category'].isin(valid_categories)]
# # print("Rare categories handled.")

# # print("Extracting features...", flush=True)
# # # Use a smaller vocabulary for testing
# # vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
# # X = vectorizer.fit_transform(train_data_filtered['crimeaditionalinfo_cleaned'])
# # y = train_data_filtered['category']
# # print("Features extracted.")

# # print("Splitting data...", flush=True)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# # print("Data split completed.")

# # print("Training model...", flush=True)
# # model = RandomForestClassifier(n_estimators=10, random_state=42)  # Use fewer trees for faster execution
# # model.fit(X_train, y_train)
# # print("Model training completed.")

# # print("Evaluating model...", flush=True)
# # y_pred = model.predict(X_test)
# # print("Accuracy:", accuracy_score(y_test, y_pred))
# # print("Classification Report:")
# # print(classification_report(y_test, y_pred))


import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Function to clean and normalize text
def clean_text(text):
    """Clean and normalize text."""
    if isinstance(text, str):
        text = re.sub(r'\W+', ' ', text)  # Remove non-alphanumeric characters
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
        text = text.lower()  # Convert to lowercase
    return text

print("Loading data...", flush=True)
train_data = pd.read_csv("train.csv")

# Sample a smaller dataset for faster execution
train_data = train_data.sample(10000, random_state=42)  # Use a random sample of 2,000 rows
print("Data loaded.")

print("Preprocessing data...", flush=True)
train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo'].apply(clean_text)
train_data['crimeaditionalinfo_cleaned'] = train_data['crimeaditionalinfo_cleaned'].fillna(" ")  # Replace NaN with empty strings
train_data['sub_category'] = train_data['sub_category'].fillna('Unknown')
print("Preprocessing completed.")

print("Handling rare categories...", flush=True)
category_counts = train_data['category'].value_counts()
valid_categories = category_counts[category_counts >= 2].index
train_data_filtered = train_data[train_data['category'].isin(valid_categories)]
print("Rare categories handled.")

print("Extracting features...", flush=True)
vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
X = vectorizer.fit_transform(train_data_filtered['crimeaditionalinfo_cleaned'])
y = train_data_filtered['category']
print("Features extracted.")

print("Splitting data...", flush=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Data split completed.")

print("Training model with class weights...", flush=True)
model = RandomForestClassifier(n_estimators=50, random_state=42, class_weight="balanced")  # More trees, handle imbalance
model.fit(X_train, y_train)
print("Model training completed.")

print("Evaluating model...", flush=True)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
