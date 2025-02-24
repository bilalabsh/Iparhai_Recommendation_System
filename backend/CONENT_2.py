import os
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


# Preprocess text function with Porter Stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

# Joins all JSON files
def concatenate_json_files(folder_path):
    # List all JSON files in the folder
    json_files = [file for file in os.listdir(folder_path) if file.endswith('.json')]
    
    # Load and concatenate all JSON files into a single DataFrame
    dataframes = []
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        df = pd.read_json(file_path)  # Read the JSON file into a DataFrame
        dataframes.append(df)
    
    # Combine all DataFrames
    concatenated_data = pd.concat(dataframes, ignore_index=True)

    # Apply preprocessing to the questions
    concatenated_data['Question_processed'] = concatenated_data['Question'].apply(preprocess_text)

    # Remove duplicates based on the 'Question_processed' column, keeping the first occurrence
    concatenated_data = concatenated_data.drop_duplicates(subset='Question_processed', keep='first')

    return concatenated_data




# Assuming the following functions are already defined as per your initial code
def generate_random_questions(data, num_questions):
    return data.sample(num_questions)

def display_questions_and_options(questions, show_subtopic=False):
    for idx, row in questions.iterrows():
        if show_subtopic:
            print(f"Subtopic: {row['Sub_topic']}")  # Display subtopic information
        print(f"Question {row['QuestionNumber']}: \n{row['Question']}")
        for option, value in row['Options'].items():
            print(f"{option}: {value}")
        print("\n")

def collect_user_answers(questions):
    user_answers = {}
    for idx, row in questions.iterrows():
        answer = input(f"Your answer for Question {row['QuestionNumber']}: ").strip().upper()
        user_answers[row['QuestionNumber']] = answer
    return user_answers

def calculate_score(questions, user_answers):
    score = 0
    for idx, row in questions.iterrows():
        if user_answers[row['QuestionNumber']] == row['answer']:
            score += 1
    return score

def analyze_performance(questions, user_answers):
    subtopic_performance = {}
    for idx, row in questions.iterrows():
        subtopic = row['Sub_topic']
        if subtopic not in subtopic_performance:
            subtopic_performance[subtopic] = {'correct': 0, 'total': 0}
        subtopic_performance[subtopic]['total'] += 1
        if user_answers[row['QuestionNumber']] == row['answer']:
            subtopic_performance[subtopic]['correct'] += 1
    
    subtopic_scores = {subtopic: (data['correct'] / data['total']) for subtopic, data in subtopic_performance.items()}
    return subtopic_scores

def find_similar_questions(data, question_number, top_n=3):
    # Filter by subtopic
    subtopic = data[data['QuestionNumber'] == question_number]['Sub_topic'].values[0]
    filtered_data = data[data['Sub_topic'] == subtopic].reset_index(drop=True)  # Reset indices
    
    # Vectorize the processed questions
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(filtered_data['Question_processed'])
    
    # Find the index of the given question in the filtered DataFrame
    question_index = filtered_data[filtered_data['QuestionNumber'] == question_number].index[0]
    
    # Calculate cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[question_index], tfidf_matrix).flatten()
    
    # Get the indices of the top_n most similar questions
    similar_indices = cosine_similarities.argsort()[:-top_n-1:-1]
    
    # Get the top_n most similar questions
    similar_questions = filtered_data.iloc[similar_indices].copy()
    
    # Add similarity scores
    similar_questions['Similarity Score'] = cosine_similarities[similar_indices]
    
    return similar_questions

def recommend_for_wrong_questions(data, wrong_questions):
    recommended_questions = pd.DataFrame()

    if not wrong_questions:
        print("No wrong questions to recommend from.")
        return recommended_questions

    for question_number in wrong_questions:
        # Find 1 similar question for each wrong question
        similar_question = find_similar_questions(data, question_number, top_n=1)
        recommended_questions = pd.concat([recommended_questions, similar_question])

    return recommended_questions


def recommend_for_weak_subtopics(data, weak_subtopics):
    recommended_questions = pd.DataFrame()

    if not weak_subtopics:
        print("No weak subtopics identified.")
        return recommended_questions

    for subtopic in weak_subtopics:
        # Filter questions by subtopic and select 2 random questions
        subtopic_questions = data[data['Sub_topic'] == subtopic].sample(n=2, replace=True)
        recommended_questions = pd.concat([recommended_questions, subtopic_questions])

    return recommended_questions


def practice_more(data, weak_subtopics, wrong_questions):
    if not weak_subtopics and not wrong_questions:
        print("Practice complete! No weak subtopics or wrong questions identified.")
        return

    practice_more = input("Do you want to practice more? (yes/no): ").strip().lower()

    while practice_more == 'yes':
        # Recommend questions for wrong questions and weak subtopics
        wrong_question_recommendations = recommend_for_wrong_questions(data, wrong_questions)
        weak_subtopic_recommendations = recommend_for_weak_subtopics(data, weak_subtopics)

        # Combine both sets of recommendations
        recommended_questions = pd.concat([wrong_question_recommendations, weak_subtopic_recommendations])

        if recommended_questions.empty:
            print("No questions available for practice. Exiting.")
            break

        print("--------------------------------Practice---------------------------------\n")
        display_questions_and_options(recommended_questions, show_subtopic=True)

        # Collect user answers for the recommended questions
        recommended_user_answers = collect_user_answers(recommended_questions)


        print("--------------------------------Result---------------------------------\n")
        # Continue with performance evaluation and practice
        recommended_score = calculate_score(recommended_questions, recommended_user_answers)
        print(f"Your score for the recommended questions is: {recommended_score}/{len(recommended_questions)}")

        # Analyze performance for the recommended questions
        recommended_subtopic_scores = analyze_performance(recommended_questions, recommended_user_answers)
        sorted_recommended_subtopics = sorted(recommended_subtopic_scores.items(), key=lambda x: x[1])
        weakest_recommended_subtopics = [subtopic for subtopic, score in sorted_recommended_subtopics[:2]]

        print(f"Your weakest subtopics after practice are: {weakest_recommended_subtopics}\n\n")

        # Identify wrong answers from the recommended questions
        wrong_recommended_questions = [
            row['QuestionNumber']
            for idx, row in recommended_questions.iterrows()
            if recommended_user_answers[row['QuestionNumber']] != row['answer']
        ]

        # Update weak subtopics and wrong questions for the next round
        weak_subtopics = weakest_recommended_subtopics
        wrong_questions = wrong_recommended_questions

        # Ask if the user wants to practice more again
        practice_more = input("Do you want to practice more? (yes/no): ").strip().lower()

    print("Thank you for practicing! Goodbye.")

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Helper function to calculate precision, recall, F1 score
def calculate_metrics(questions, user_answers):
    y_true = [row['answer'] for _, row in questions.iterrows()]  # Correct answers
    y_pred = [user_answers[row['QuestionNumber']] for _, row in questions.iterrows()]  # User answers

    # Convert to binary for metrics calculation
    y_true_binary = [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]

    # Metrics
    precision = precision_score(y_true_binary, [1]*len(y_true_binary))  # Binary assumption
    recall = recall_score(y_true_binary, [1]*len(y_true_binary))
    f1 = f1_score(y_true_binary, [1]*len(y_true_binary))

    return precision, recall, f1


# Helper function to calculate MAP
def calculate_map(questions, user_answers):
    y_true = [row['answer'] for _, row in questions.iterrows()]  # Correct answers
    y_pred = [user_answers[row['QuestionNumber']] for _, row in questions.iterrows()]  # User answers

    # Calculate precision at each correct answer
    precision_at_k = []
    correct_count = 0
    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        if true == pred:
            correct_count += 1
            precision_at_k.append(correct_count / (i + 1))  # Precision@K

    # Mean Average Precision
    mean_average_precision = np.mean(precision_at_k) if precision_at_k else 0
    return mean_average_precision


# Helper function to calculate NDCG
def calculate_ndcg(questions, user_answers):
    y_true = [row['answer'] for _, row in questions.iterrows()]  # Correct answers
    y_pred = [user_answers[row['QuestionNumber']] for _, row in questions.iterrows()]  # User answers

    # Binary relevance
    relevance = [1 if pred == true else 0 for pred, true in zip(y_pred, y_true)]

    # Calculate DCG
    def dcg(scores):
        return sum(score / np.log2(idx + 2) for idx, score in enumerate(scores))

    # DCG for user answers
    dcg_score = dcg(relevance)

    # Ideal DCG (sorted by relevance)
    ideal_dcg_score = dcg(sorted(relevance, reverse=True))

    # NDCG
    ndcg_score = dcg_score / ideal_dcg_score if ideal_dcg_score > 0 else 0
    return ndcg_score


def main():


    n = 10  # Number of questions to ask

    folder_path = "datset_json" 
    data = concatenate_json_files(folder_path)

    # Generate random questions
    questions = generate_random_questions(data, n)

    # Display questions and collect user answers
    print("--------------------------------QUIZ---------------------------------\n")
    display_questions_and_options(questions, show_subtopic=True)
    user_answers = collect_user_answers(questions)

    print("--------------------------------Result---------------------------------\n")
    # Calculate score
    score = calculate_score(questions, user_answers)
    print(f"Your score is: {score}/{n}")

    # Analyze performance
    subtopic_scores = analyze_performance(questions, user_answers)
    sorted_subtopics = sorted(subtopic_scores.items(), key=lambda x: x[1])
    weakest_subtopics = [subtopic for subtopic, score in sorted_subtopics[:2]]
    strongest_subtopic = sorted_subtopics[-1][0]

    print(f"Your strongest subtopic is: {strongest_subtopic}")
    print(f"Your weakest subtopics are: {weakest_subtopics}\n\n")

    # Identify wrong answers
    wrong_questions = [
        row['QuestionNumber']
        for idx, row in questions.iterrows()
        if user_answers[row['QuestionNumber']] != row['answer']
    ]

    # Calculate metrics
    precision, recall, f1 = calculate_metrics(questions, user_answers)
    mean_average_precision = calculate_map(questions, user_answers)
    ndcg_score = calculate_ndcg(questions, user_answers)

    print("----------Performance Metrics----------")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Mean Average Precision (MAP): {mean_average_precision:.2f}")
    print(f"NDCG (Normalized Discounted Cumulative Gain): {ndcg_score:.2f}\n")

    # Ask if the user wants to practice more
    practice_more(data, weakest_subtopics, wrong_questions)


if __name__ == "__main__":
    main()
