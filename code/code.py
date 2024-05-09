from textblob import TextBlob
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from collections import Counter
from textblob import TextBlob
import seaborn as sns
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import datetime


def analyze_sentiment(comment):
    sid = SentimentIntensityAnalyzer()
    sentiment_score = sid.polarity_scores(comment)
    return sentiment_score['compound']

def topicmodelinglda(df):
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')


    dtm = vectorizer.fit_transform(df['clean_comments'])

    lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
    lda_model.fit(dtm)
    no_top_words = 10
    display_topics(lda_model, vectorizer.get_feature_names_out(), no_top_words)
    
    

def create_wordcloud_for_topic(topic_idx, model, feature_names):
    topic_words = model.components_[topic_idx]
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(
        ' '.join([feature_names[i] for i in topic_words.argsort()[:-11:-1]]))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Topic {topic_idx + 1}')
    plt.axis('off')
    plt.show()

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx + 1}:")
        print(" ".join([feature_names[i]
              for i in topic.argsort()[:-no_top_words - 1:-1]]))
    for topic_idx in range(model.n_components):
        create_wordcloud_for_topic(topic_idx, model, feature_names)


def contains_suggestion(comment):
    suggestion_keywords = ['suggest', 'recommend','improve', 'should', 'could', 'need to']
    for keyword in suggestion_keywords:
        if keyword in comment.lower():
            return True
    return False


def contains_question(comment):
    return '?' in comment


def main():
    # Load the dataset from a CSV file
   # Replace with the actual path to your CSV file
    # student_data = pd.read_csv('new_data.csv', encoding='latin1')

    # Analyze and visualize sentiment for each student
    print("Enter the dataset name:")
    dataset_name = input().strip() # Taking dataset name as input for this to work dataset must be in the same directory as code
    student_data = pd.read_csv(dataset_name) # reading the dataset with assigned variable
    positive_percentages = []
    sentiments = student_data['Comments'].apply(lambda x: TextBlob(x).sentiment.polarity)
    sns.histplot(sentiments, bins=20)
    plt.xlabel('Sentiment Polarity')
    plt.ylabel('Frequency')
    plt.title('Distribution of Sentiment Polarity in Review Text')
    plt.show()

    for _, row in student_data.iterrows():
        
        comments = row['Comments']

        # Perform sentiment analysis
        sentiment_score = analyze_sentiment(comments)
        # Convert to percentage (normalized to [0, 100])
        positive_percentage = (sentiment_score + 1) * 50
        positive_percentages.append(positive_percentage)

        # Output: Show the percentage of positiveness
        # print(f"\nSentiment Analysis Result for {student_name}:")
        # print(f"Positive Percentage: {positive_percentage:.2f}%")

    # Visualize sentiment for all students in a ring graph
    average_positive_percentage = sum(positive_percentages) / len(positive_percentages)
    print(f"\nAverage Positive Percentage for All Students: {average_positive_percentage:.2f}%")
    labels = ['Positive', 'Negative']
    sizes = [positive_percentage, 100 - positive_percentage]
    colors = ['#4CAF50', '#FF5733']  # Green for positive, Orange for negative

    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Sentiment Analysis for the Event')
    plt.show()

    # Define colors for success and failure
    success_color = '#66c2a5'  # Green color for success
    failure_color = '#fc8d62'  # Red color for failure

# Calculate failure percentage
    failure_percentage = 100 - positive_percentage

# Plotting the bar plot
    plt.figure(figsize=(8, 6))
    plt.bar(['Success', 'Failure'], [positive_percentage,
            failure_percentage], color=[success_color, failure_color])
    plt.title('Overall Success of the Event')
    plt.ylabel('Percentage')
    plt.ylim(0, 100)
    plt.show()

    # analyze_and_visualize_sentiment(student_data)<===================================================
#===================================================================(TILL NOW IT IS TILL THIS FUNCTION |)
    stop_words = set(stopwords.words('english'))


# Combine all summaries into a single string
    summaries = ' '.join(student_data['Comments']).split()

# Filter out stopwords from the summaries
    filtered_summaries = [word for word in summaries if word.lower() not in stop_words]

# Count the remaining words
    summary_word_counts = Counter(filtered_summaries)
    common_summary_words = summary_word_counts.most_common(10)

# Plot the bar chart
    plt.bar(*zip(*common_summary_words))
    plt.xlabel('Word')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Common Words in Summary Text (Excluding Stopwords)')
    plt.xticks(rotation=45)
    plt.show()
    # topmost(student_data)<============================================================================
# ===================================================================(TILL NOW IT IS TILL THIS FUNCTION |)

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(
            token) for token in tokens if token.isalnum() and token not in stop_words]
        return ' '.join(tokens)

    student_data['clean_comments'] = student_data['Comments'].apply(preprocess_text)
    sia = SentimentIntensityAnalyzer()

    def get_sentiment_score(text):
        return sia.polarity_scores(text)['compound']

    student_data['sentiment_score'] = student_data['clean_comments'].apply(get_sentiment_score)

# Calculate Positive Sentiment Percentage
    positive_comments = student_data[student_data['sentiment_score'] > 0]
    positive_sentiment_percentage = (len(positive_comments) / len(student_data)) * 100

# Determine Overall Success
    if positive_sentiment_percentage > 50:
        print("The event was successful!")
    else:
        print("The event was not successful.")
    

    # df=outcome(student_data)<=========================================================================
# ===================================================================(TILL NOW IT IS TILL THIS FUNCTION |)

    text = ' '.join(student_data['Comments'])

    wordcloud = WordCloud(width=800, height=400).generate(text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Review Text')
    plt.show()

    # allwordcloud(student_data)<=======================================================================
# ===================================================================(TILL NOW IT IS TILL THIS FUNCTION |)


    topicmodelinglda(student_data)
    
    suggested_comments = student_data[student_data['Comments'].apply(contains_suggestion)]
    print("Comments Containing Suggestions:")
    print(len(suggested_comments))
    print(suggested_comments['Comments'])

    question_comments = student_data[student_data['Comments'].apply(contains_question)]


    # Display the comments containing questions
    print("Comments Containing Questions:")
    print(len(question_comments))
    print(question_comments['Comments'])
   
if __name__ == "__main__":
    
    
    main()
