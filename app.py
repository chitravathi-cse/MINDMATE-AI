import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="MINDMATE AI", page_icon="🤖")

st.title("🤖 MINDMATE AI")
st.subheader("Your Smart Student Assistant")
st.write("Ask me anything about studies, AI and careers!")

# Language selection
language = st.selectbox(
    "Choose your language",
    ["English", "Kannada", "Telugu"]
)

# Questions and answers
data = {
    "English": {
        "greeting": {
            "questions": ["hello", "hi", "good morning", "hey"],
            "answer": "Hello! 👋 Welcome to MINDMATE AI. How can I help you?"
        },
        "ai": {
            "questions": ["what is artificial intelligence",
                          "define AI", "explain AI"],
            "answer": "Artificial Intelligence (AI) is a technology that helps machines think, learn and solve problems like humans."
        },
        "study": {
            "questions": ["give me study tips", "how to study",
                          "help me study"],
            "answer": "📚 Make a timetable, study one topic at a time, take short breaks and revise daily."
        },
        "quiz": {
            "questions": ["give me a quiz", "ask me a question",
                          "test my knowledge"],
            "answer": "🧠 Quiz time! What is the full form of AI? A) Artificial Intelligence B) Automatic Internet C) Advanced Information"
        },
        "career": {
            "questions": ["career guidance", "which skills should I learn",
                          "suggest a career"],
            "answer": "🚀 Learn Python, communication, problem-solving, AI and web development. Build small projects to improve your skills."
        },
        "motivation": {
            "questions": ["motivate me", "i feel sad",
                          "give me motivation"],
            "answer": "✨ Believe in yourself! Every small step takes you closer to your dream. Never give up!"
        }
    },

    "Kannada": {
        "greeting": {
            "questions": ["ನಮಸ್ಕಾರ", "ಹಲೋ", "ಹಾಯ್"],
            "answer": "ನಮಸ್ಕಾರ! 👋 MINDMATE AI ಗೆ ಸ್ವಾಗತ. ನಿಮಗೆ ಹೇಗೆ ಸಹಾಯ ಮಾಡಲಿ?"
        },
        "ai": {
            "questions": ["ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಎಂದರೇನು",
                          "ಎಐ ಎಂದರೇನು"],
            "answer": "ಕೃತಕ ಬುದ್ಧಿಮತ್ತೆ ಎಂದರೆ ಮನುಷ್ಯರಂತೆ ಯೋಚಿಸಲು ಮತ್ತು ಕಲಿಯಲು ಯಂತ್ರಗಳಿಗೆ ಸಹಾಯ ಮಾಡುವ ತಂತ್ರಜ್ಞಾನ."
        },
        "study": {
            "questions": ["ಓದಲು ಸಲಹೆ ನೀಡಿ", "ಹೇಗೆ ಓದಬೇಕು"],
            "answer": "📚 ವೇಳಾಪಟ್ಟಿ ಮಾಡಿ, ಪ್ರತಿದಿನ ಓದಿ ಮತ್ತು ಓದಿದ ವಿಷಯಗಳನ್ನು ಪುನರಾವರ್ತಿಸಿ."
        },
        "quiz": {
            "questions": ["ಪ್ರಶ್ನೆ ಕೇಳಿ", "ರಸಪ್ರಶ್ನೆ ನೀಡಿ"],
            "answer": "🧠 AI ಯ ಪೂರ್ಣ ರೂಪ ಏನು? A) Artificial Intelligence B) Automatic Internet"
        },
        "career": {
            "questions": ["ಉದ್ಯೋಗ ಮಾರ್ಗದರ್ಶನ ನೀಡಿ",
                          "ಯಾವ ಕೌಶಲ್ಯ ಕಲಿಯಬೇಕು"],
            "answer": "🚀 Python, ಸಂವಹನ ಕೌಶಲ್ಯ, AI ಮತ್ತು ವೆಬ್ ಡೆವಲಪ್ಮೆಂಟ್ ಕಲಿಯಿರಿ."
        },
        "motivation": {
            "questions": ["ನನಗೆ ಪ್ರೇರಣೆ ನೀಡಿ", "ನನಗೆ ಬೇಸರವಾಗಿದೆ"],
            "answer": "✨ ನಿಮ್ಮ ಮೇಲೆ ನಂಬಿಕೆ ಇಡಿ. ಪ್ರತಿಯೊಂದು ಸಣ್ಣ ಪ್ರಯತ್ನವೂ ಯಶಸ್ಸಿನತ್ತ ಕರೆದೊಯ್ಯುತ್ತದೆ!"
        }
    },

    "Telugu": {
        "greeting": {
            "questions": ["నమస్కారం", "హలో", "హాయ్"],
            "answer": "నమస్కారం! 👋 MINDMATE AI కి స్వాగతం. నేను మీకు ఎలా సహాయం చేయగలను?"
        },
        "ai": {
            "questions": ["కృత్రిమ మేధస్సు అంటే ఏమిటి",
                          "AI అంటే ఏమిటి"],
            "answer": "కృత్రిమ మేధస్సు అనేది మనుషుల్లా ఆలోచించడానికి మరియు నేర్చుకోవడానికి యంత్రాలకు సహాయపడే సాంకేతికత."
        },
        "study": {
            "questions": ["చదువు చిట్కాలు చెప్పండి",
                          "ఎలా చదవాలి"],
            "answer": "📚 టైమ్ టేబుల్ తయారు చేసుకోండి, ప్రతిరోజూ చదవండి మరియు నేర్చుకున్న విషయాలను మళ్లీ చదవండి."
        },
        "quiz": {
            "questions": ["నాకు ప్రశ్న అడగండి",
                          "క్విజ్ ఇవ్వండి"],
            "answer": "🧠 AI పూర్తి రూపం ఏమిటి? A) Artificial Intelligence B) Automatic Internet"
        },
        "career": {
            "questions": ["ఉద్యోగ మార్గదర్శనం ఇవ్వండి",
                          "ఏ నైపుణ్యాలు నేర్చుకోవాలి"],
            "answer": "🚀 Python, కమ్యూనికేషన్, AI మరియు వెబ్ డెవలప్‌మెంట్ నేర్చుకోండి."
        },
        "motivation": {
            "questions": ["నన్ను ప్రోత్సహించండి",
                          "నాకు బాధగా ఉంది"],
            "answer": "✨ మీపై మీకు నమ్మకం ఉంచండి. ప్రతి చిన్న ప్రయత్నం మిమ్మల్ని విజయానికి దగ్గర చేస్తుంది!"
        }
    }
}

# AI response matching using NLP
def get_response(user_input, lang):

    questions = []
    answers = []

    for intent in data[lang].values():
        for question in intent["questions"]:
            questions.append(question)
            answers.append(intent["answer"])

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5)
    )

    vectors = vectorizer.fit_transform(
        questions + [user_input]
    )

    similarity = cosine_similarity(
        vectors[-1], vectors[:-1]
    )

    best_match = similarity.argmax()

    if similarity[0][best_match] < 0.12:
        return "Sorry! I don't understand. Please ask another question."

    return answers[best_match]


# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.write(user_input)

    response = get_response(user_input, language)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()