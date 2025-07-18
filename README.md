# Topic-Modeling-on-Twitter-Data
This machine learning project explore topic modelling on Twitter data using LDA and BERTopic. This includes data cleaning, EDA, model building, evaluation, and visualizations to discover latent themes in political tweets.

# NLP Topic Modelling on Twitter Data

This repository contains a topic modelling project completed as part of the ICT606 Machine Learning unit at Murdoch University. The project applies Natural Language Processing (NLP) techniques to analyze political Twitter data using Latent Dirichlet Allocation (LDA) and BERTopic models.

## 📌 Project Overview

Twitter is a rich source of real-time public opinion. In this project, we explore how topic modelling can uncover hidden themes in tweets. Two main models were implemented:

- **LDA (Latent Dirichlet Allocation)** from scikit-learn
- **BERTopic** using transformer-based embeddings

The project involves:
- Cleaning and preprocessing 5,000 sampled tweets
- Performing Exploratory Data Analysis (EDA)
- Applying and evaluating topic models
- Visualizing and interpreting the discovered topics

---

## Dataset

The dataset was sourced from Kaggle:  
🔗 [Twitter Sentiment Dataset](https://www.kaggle.com/datasets/saurabhshahane/twitter-sentiment-dataset)

- `cleaned_text`: The tweet text
- `category`: Sentiment label (-1 = negative, 0 = neutral, 1 = positive)

---

## Preprocessing Steps

1. Lowercasing
2. Removing URLs, mentions, hashtags
3. Removing punctuation and numbers
4. Tokenization
5. Stopword removal (NLTK)
6. Lemmatization (WordNetLemmatizer)
7. Filtering short words (<3 chars)

---

## Exploratory Data Analysis (EDA)

- Sentiment distribution plot
- Word clouds for each sentiment category
- Top 20 TF-IDF words
- Word2Vec embeddings visualized using t-SNE

---

## Topic Modelling

### LDA (Latent Dirichlet Allocation)

- Used `CountVectorizer` with parameters:  
  `max_df=0.95`, `min_df=10`, `stop_words='english'`
- Extracted 10 topics
- Visualized topic-word heatmaps
- Assigned dominant topic to each tweet

### BERTopic

- Uses BERT embeddings + UMAP + HDBSCAN
- Higher topic diversity score (0.86 vs 0.69 for LDA)
- Visualized intertopic distance map and topic bar charts

---

## 📈 Evaluation Metrics

| Metric           | LDA  | BERTopic |
|------------------|------|----------|
| Coherence Score  | 0.40 | 0.38     |
| Topic Diversity  | 0.69 | 0.86     |

- LDA had slightly higher coherence
- BERTopic showed greater topic diversity and semantic separation

---

## 📌 Key Insights

- Topics revealed strong focus on Indian politics (Modi, BJP, Congress)
- BERTopic captured more distinct themes like geopolitical issues and national pride
- Word embeddings improved semantic clustering in BERTopic


## 📚 References

- Becker et al. (2011). *Beyond trending topics: Real-world event identification on Twitter.*
- Singh et al. (2019). *Topic modelling and classification of tweets using NLP.*


## 🛠️ Tech Stack

- Python
- Scikit-learn
- BERTopic
- Pandas, Numpy
- NLTK, WordNet
- Matplotlib, Seaborn

