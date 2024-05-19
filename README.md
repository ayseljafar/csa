# Customer Satisfaction | Sentiment Analysis

## This project explored customer sentiment in Trustpilot reviews using a comprehensive toolbox that combined web scraping, machine learning, and Natural Language Processing (NLP) techniques.

Data Acquisition and Preprocessing:

Web scraping with Beautiful Soup gathered a diverse dataset of Trustpilot reviews, including both textual content and metadata.
Rigorous data cleaning ensured a robust dataset for further analysis. This involved addressing missing values and employing text cleaning techniques.
Sentiment Classification and Interpretation:

We explored a wide range of machine learning models for sentiment classification, including Naive Bayes, Random Forest, Support Vector Machines (SVM), XGBoost, Logistic Regression, and Neural Networks. These models achieved impressive accuracy, reaching up to 94% on the test dataset.
Feature importance and SHAP analysis were employed to understand the factors influencing model predictions, providing insights into which aspects of the reviews (textual or metadata) were most impactful for each model.
Advanced NLP for Deeper Analysis:

To gain a deeper understanding of customer sentiment, NLP techniques were utilized:

Tokenization: Breaking down textual content into individual words for more granular analysis.
Vectorization: Converting tokens into numerical representations suitable for machine learning algorithms.
Word2Vec: Unveiling semantic relationships between words within reviews, enhancing both sentiment classification accuracy and cluster analysis.
Finally, K-means clustering, an unsupervised learning algorithm, identified recurring themes and patterns within the reviews. This provided a comprehensive view of customer sentiment and valuable insights for businesses seeking to improve customer satisfaction and engagement.

[You can view the Project here](https://customer-satisfaction.streamlit.app/)
