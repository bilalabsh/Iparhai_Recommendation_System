from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from motor.motor_asyncio import AsyncIOMotorClient  # MongoDB async driver
from bson import ObjectId
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()  # This loads the .env file

# ------------------- NLTK Setup -------------------
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ------------------- FastAPI Setup -------------------
app = FastAPI()

# Enable CORS (for React Native Expo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- MongoDB Connection -------------------
MONGO_URI = os.getenv("MONGO_URI")
# print(MONGO_URI)
DATABASE_NAME = "App"
COLLECTION_NAME = "questions"

client = AsyncIOMotorClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

# ------------------- Helper Functions -------------------


# Convert ObjectId to string for JSON serialization
def serialize_document(doc):
    doc["_id"] = str(doc["_id"])
    return doc


# Text Preprocessing Function
def preprocess_text(text):
    """Lowercase text, remove punctuation, stopwords, and apply stemming."""
    text = text.lower()
    text = "".join([char for char in text if char.isalnum() or char.isspace()])
    text = " ".join([word for word in text.split() if word not in stop_words])
    text = " ".join([stemmer.stem(word) for word in text.split()])
    return text


# Load and preprocess questions from MongoDB
async def load_questions():
    """Loads and preprocesses questions from MongoDB."""
    cursor = collection.find({})
    questions = await cursor.to_list(length=None)  # Convert cursor to list

    if not questions:
        raise HTTPException(status_code=404, detail="No questions found in database")

    for question in questions:
        if "Question" in question:
            question["Question_processed"] = preprocess_text(question["Question"])

    # Serialize ObjectIds
    questions = [serialize_document(q) for q in questions]

    return questions


# ------------------- Pydantic Models -------------------
class AnswerRequest(BaseModel):
    user_answers: dict  # Example: {"1": "A", "2": "C"}


# ------------------- API Endpoints -------------------


# Get Random Questions
@app.get("/questions/{count}")
async def get_questions(count: int):
    """Fetches a set number of random questions."""
    questions = await load_questions()
    random_questions = random.sample(questions, min(count, len(questions)))
    return {"questions": random_questions}


# Submit Answers and Evaluate
@app.post("/submit-answers/")
async def submit_answers(request: AnswerRequest):
    """Evaluates user answers and provides recommendations."""
    user_answers = request.user_answers
    correct_count = 0
    wrong_questions = []
    subtopic_performance = {}

    questions = await load_questions()

    for question in questions:
        q_num = str(question["QuestionNumber"])
        if q_num in user_answers:
            if user_answers[q_num] == question["answer"]:
                correct_count += 1
            else:
                wrong_questions.append(q_num)

            # Track subtopic performance
            subtopic = question["Sub_topic"]
            if subtopic not in subtopic_performance:
                subtopic_performance[subtopic] = {"correct": 0, "total": 0}
            subtopic_performance[subtopic]["total"] += 1
            if user_answers[q_num] == question["answer"]:
                subtopic_performance[subtopic]["correct"] += 1

    # Compute subtopic weakness
    subtopic_scores = {
        k: v["correct"] / v["total"] for k, v in subtopic_performance.items()
    }
    weakest_subtopics = sorted(subtopic_scores, key=subtopic_scores.get)[:2]

    return {
        "score": correct_count,
        "total_questions": len(user_answers),
        "weakest_subtopics": weakest_subtopics,
        "wrong_questions": wrong_questions,
    }


# Find Similar Questions Using TF-IDF
async def find_similar_questions(question_number, top_n=3):
    """Finds top-N similar questions using TF-IDF."""
    questions = await load_questions()

    question_data = next(
        (q for q in questions if q["QuestionNumber"] == question_number), None
    )

    if not question_data:
        raise HTTPException(status_code=404, detail="Question not found")

    subtopic = question_data["Sub_topic"]
    filtered_questions = [q for q in questions if q["Sub_topic"] == subtopic]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(
        [q["Question_processed"] for q in filtered_questions]
    )

    question_index = next(
        (
            i
            for i, q in enumerate(filtered_questions)
            if q["QuestionNumber"] == int(question_number)
        ),
        None,
    )
    if question_index is None:
        raise HTTPException(
            status_code=404, detail="Question not found in filtered data"
        )

    cosine_similarities = cosine_similarity(
        tfidf_matrix[question_index], tfidf_matrix
    ).flatten()
    similar_indices = cosine_similarities.argsort()[: -top_n - 1 : -1]

    similar_questions = [filtered_questions[i] for i in similar_indices]

    for i, q in enumerate(similar_questions):
        q["Similarity Score"] = float(cosine_similarities[similar_indices[i]])

    return similar_questions


# Recommend Questions Based on Previous Mistakes
@app.get("/recommend-wrong-questions/")
async def recommend_wrong_questions(wrong_questions: str):
    """Recommends questions based on previous wrong answers."""
    wrong_list = wrong_questions.split(",")  # Convert "1,2,3" to list

    recommended = []
    for q in wrong_list:
        recommended += await find_similar_questions(q, top_n=1)

    return {"recommended_questions": recommended}


# Recommend Questions from Weak Subtopics
@app.get("/recommend-weak-subtopics/")
async def recommend_weak_subtopics(weak_subtopics: str):
    """Suggests questions from weak subtopics."""
    weak_list = weak_subtopics.split(",")

    questions = await load_questions()
    recommended = []

    for subtopic in weak_list:
        subtopic_questions = [q for q in questions if q["Sub_topic"] == subtopic]
        sampled_questions = random.sample(
            subtopic_questions, min(2, len(subtopic_questions))
        )
        recommended += sampled_questions

    return {"recommended_questions": recommended}


# ------------------- Run the Server -------------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
