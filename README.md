# Disaster-Response-Pipeline-Project
ML Pipeline Description
1. Data Preprocessing
•	Load Data: The data is loaded from a SQLite database using read_sql_table.
•	Define Features and Target:
o	Features: X = df['message']
o	Targets: Y = df.iloc[:, 4:] (36 output categories).
2. Tokenization
A custom tokenize function:
•	Normalizes text by converting it to lowercase and removing special characters.
•	Tokenizes text into words.
•	Removes stopwords using NLTK.
3. Model Pipeline
The pipeline consists of:
1.	CountVectorizer: Converts text into a bag-of-words representation.
2.	TfidfTransformer: Applies TF-IDF to reweight word importance.
3.	MultiOutputClassifier: Trains a RandomForestClassifier for each target category.
4. Training and Hyperparameter Tuning
•	The data is split into training and test sets (80/20).
•	The model is trained on the training set using a Pipeline.
•	Hyperparameter tuning is performed using GridSearchCV:
python
Code kopieren
parameters = {
    'clf__estimator__n_estimators': [50, 100],
    'clf__estimator__min_samples_split': [2, 4]
}
________________________________________
Model Evaluation
The model is evaluated using precision, recall, and F1-score for each category. Example results:
Category	Precision	Recall	F1-Score
water	0.96	0.99	0.98
medical_help	0.93	0.99	0.96
shelter	0.93	0.99	0.96
Observations
•	The model performs well for majority classes.
•	Imbalanced categories (e.g., shops, tools) show poor recall due to insufficient data.
________________________________________
Future Improvements
1.	Handle Class Imbalance:
o	Use SMOTE or weighted loss functions to address imbalanced data.
2.	Model Optimization:
o	Test alternative models like XGBoost or LightGBM.
o	Include additional features (e.g., message length, genre).
3.	Enhanced Tokenization:
o	Use advanced NLP techniques like lemmatization or word embeddings.
4.	Web App Visualizations:
o	Add category-wise prediction probabilities.
o	Display message classification visually.
