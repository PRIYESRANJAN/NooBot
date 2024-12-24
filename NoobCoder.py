import json
import random
import numpy as np
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Load intents file
with open('intents.json', encoding='utf-8', errors='ignore') as file:
    intents = json.load(file)

# Prepare data
lemmatizer = WordNetLemmatizer()
patterns = []
responses = []
tags = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        responses.append(intent['responses'])
        tags.append(intent['tag'])

# Vectorize the patterns
vectorizer = TfidfVectorizer(tokenizer=nltk.word_tokenize, stop_words='english')
X = vectorizer.fit_transform(patterns)

# Create labels for the tags
tag_to_index = {tag: index for index, tag in enumerate(set(tags))}
y = np.array([tag_to_index[tag] for tag in tags])

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X, y)

# Function to predict the class of a user input
def predict_class(sentence):
    sentence_vector = vectorizer.transform([sentence])
    prediction = model.predict(sentence_vector)
    return prediction[0]

# Function to get a response based on the predicted class
def get_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "I don't understand."

# Function to log chat history to CSV
def log_chat(user_input, bot_response):
    log_file = 'chat_log.csv'
    new_entry = pd.DataFrame([[user_input, bot_response]], columns=['User   ', 'Bot'])
    
    if os.path.isfile(log_file):
        new_entry.to_csv(log_file, mode='a', header=False, index=False)
    else:
        new_entry.to_csv(log_file, mode='w', header=True, index=False)

# Streamlit interface
def main():
    st.set_page_config(page_title="Chatbot", page_icon="🤖")
    
    # Background video
    st.markdown(
        """
        <style>
        .video-background {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: -1;
            overflow: hidden;
        }
        .video-background video {
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }
        .stApp {
            background-color: rgba(0, 0, 0, 0.5); /* Semi-transparent background */
            color: white;
        }
        </style>
        <div class="video-background">
            <video autoplay loop muted>
                <source src="https://media.gettyimages.com/id/2164160009/video/artificial-intelligence-technology-data-computer-programmer-metaverse-digitally-generated.mp4?s=mp4-640x640-gi&k=20&c=3bgG54fC5JCpHJIv7UBhTB9S3j1QdZq1oadO5E86uxU=" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select an option", ["Home", "Connect Us", "Syllabus", "Question Bank"])

    # Main content area based on dropdown selection
    if option == "Home":
        st.title("Welcome to the Chatbot!")
        st.write("Type your message below to chat with the bot.")
        
        user_input = st.text_input("You: ")

        if st.button("Send"):
            if user_input:  # Correctly indented
                predicted_tag_index = predict_class(user_input)
                predicted_tag = list(tag_to_index.keys())[predicted_tag_index]
                response = get_response(predicted_tag)

                # Log the chat history
                log_chat(user_input, response)

                # Display the bot's response
                st.write(f"Bot: {response}")
            else:
                                st.warning("Please enter a message before sending.")

    elif option == "Connect Us":
        st.title("Connect Us")
        st.write("You can reach us at: priyesranjan@gmail.com")

    elif option == "Syllabus":
        st.title("Syllabus")
        branch = st.selectbox("Select Branch", ["IOT", "CSE", "ECE", "EE", "CE", "ME"])
        semester = st.selectbox("Select Semester", ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])

        # Syllabus links for all branches
        syllabus_links = {
            "IOT": {
                "1st": "https://docs.google.com/uc?export=download&id=1w73Q_Lf5D5N21hDs5NUf1OlrOmQU9FDs",
                "2nd": "https://docs.google.com/uc?export=download&id=1qG35rzdLe3q09S2m8LL-FUy5RBiQmytd",
                "3rd": "https://docs.google.com/uc?export=download&id=12Lg062__HyX6viANPcvWLqjuceauVuWO",
                "4th": "https://docs.google.com/uc?export=download&id=1wIcdRxufNPHgFk0E2I7nUfh_UPxpbyyn",
                "5th": "https://docs.google.com/uc?export=download&id=1RKtWrEztEHzocNpSEBRNLnl5m9kIGiMP",
                "6th": "https://docs.google.com/uc?export=download&id=15jyzYmg3wBGyWOjXgGVKiS_gVybSQU4e",
                "7th": "https://docs.google.com/uc?export=download&id=16sDI-Ozo7hqSkdSgilcJQYxFsG5lvX3r",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            },
            "CSE": {
                "1st": "https://docs.google.com/uc?export=download&id=1w73Q_Lf5D5N21hDs5NUf1OlrOmQU9FDs",
                "2nd": "https://docs.google.com/uc?export=download&id=1qG35rzdLe3q09S2m8LL-FUy5RBiQmytd",
                "3rd": "https://docs.google.com/uc?export=download&id=12Lg062__HyX6viANPcvWLqjuceauVuWO",
                "4th": "https://docs.google.com/uc?export=download&id=1wIcdRxufNPHgFk0E2I7nUfh_UPxpbyyn",
                "5th": "https://docs.google.com/uc?export=download&id=1RKtWrEztEHzocNpSEBRNLnl5m9kIGiMP",
                "6th": "https://docs.google.com/uc?export=download&id=15jyzYmg3wBGyWOjXgGVKiS_gVybSQU4e",
                "7th": "https://docs.google.com/uc?export=download&id=16sDI-Ozo7hqSkdSgilcJQYxFsG5lvX3r",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            },
            "ECE": {
                "1st": "https://docs.google.com/uc?export=download&id=1Z_uIqFFve_WNWZmW7ghbo0TmngOhNGQc",
                "2nd": "https://docs.google.com/uc?export=download&id=1pGoXGR0RfXfRvkeJy_nxiA-E_YIFy35b",
                "3rd": "https://docs.google.com/uc?export=download&id=1cDvVPlLDC0ziZCUz31rtqs8cSij8XLhq",
                "4th": "https://docs.google.com/uc?export=download&id=1Odjrqru3hIkCSBD4tutxWRoE4ruSMz-m",
                                "5th": "https://docs.google.com/uc?export=download&id=1iZE2-nGPyevgeM7HE8Xt5SnKqzSTNfLh",
                "6th": "https://docs.google.com/uc?export=download&id=14QlVfyR70AgRzyDaLmOBDbVLoLUb7T5z",
                "7th": "https://docs.google.com/uc?export=download&id=1IZAgJM44r5P-fuohdinxDU2aU7G4fMep",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            },
            "EE": {
                "1st": "https://docs.google.com/uc?export=download&id=1EvWsW_9VfN8cCGrTLX_nSSbIwftSMqz9",
                "2nd": "https://docs.google.com/uc?export=download&id=1x5XvGTCMP3OQ4gOD5X8BXdQTZOIHVmtW",
                "3rd": "https://docs.google.com/uc?export=download&id=1WRjNBTJUJqA4JGzuVdnBW7hbpC5o93lP",
                "4th": "https://docs.google.com/uc?export=download&id=1ZnYPXGOvopwlD4A_bQrHqR5K-x43dnpe",
                "5th": "https://docs.google.com/uc?export=download&id=1dnFf5X1z1z6tiGk7sDTZIN9mhGHTxcJ9",
                "6th": "https://docs.google.com/uc?export=download&id=1-sT272zCMxVqp9jL4_80zoClEzE--o0r",
                "7th": "https://docs.google.com/uc?export=download&id=1_xcPKZvCT1avsZBypJIQ2OzMbNAcKJ78",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            },
            "CE": {
                "1st": "https://docs.google.com/uc?export=download&id=1kQg7FO2r3a2yyZGZ9ThQ9cruxBaSAc5E",
                "2nd": "https://docs.google.com/uc?export=download&id=1mtMQm6BFIHb6d7LD_VXGwwARX2TH48tJ",
                "3rd": "https://docs.google.com/uc?export=download&id=1_MCFnRu4--eFk63Kcl3DOVAMmOdTidSu",
                "4th": "https://docs.google.com/uc?export=download&id=1AxnC52LbSoqMIVA9NZk_dkPsE0LPPNln",
                "5th": "https://docs.google.com/uc?export=download&id=1o0ieHKUGRgFj6lyIfE0k8bD_YccgRsS1",
                "6th": "https://docs.google.com/uc?export=download&id=199QSq3RHl5imQrwdAY13usQwo-jILtk9",
                "7th": "https://docs.google.com/uc?export=download&id=1l6r73fBntjjMDE3DnoCJ6YOuJKUsOfpG",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            },
            "ME": {
                "1st": "https://docs.google.com/uc?export=download&id=1ZI7Ssb0rpAKG0xByWHRZCL8GAHhGIhQJ",
                "2nd": "https://docs.google.com/uc?export=download&id=12KVc8Se7WrHMWlNCT75scOcasv5hqI5i",
                "3rd": "https://docs.google.com/uc?export=download&id=1ad6J8hAehBRfHZJy2-lJ_7r9N51bVgBR",
                "4th": "https://docs.google.com/uc?export=download&id=1-Hd_veE0lx39kingmX0JfME4KIZgQIDw",
                                "5th": "https://docs.google.com/uc?export=download&id=1vmeOiGWlVW8a353acZDYVquz1B2_2MA9",
                "6th": "https://docs.google.com/uc?export=download&id=1bwGcRxw0H3qPJ6rfEwMiiRhc2e8bzcz7",
                "7th": "https://docs.google.com/uc?export=download&id=1xqHKy-9Ae4mxoaYQgMMs7Ww9wg4ORaUp",
                "8th": "https://beubiharsyllabus.blogspot.com/undefined",
            }
        }

        # Display syllabus based on selected branch and semester
        if branch and semester:
            if branch in syllabus_links and semester in syllabus_links[branch]:
                st.write(f"Syllabus for {branch} - {semester} Semester:")
                st.markdown(f"[Click here for the syllabus]({syllabus_links[branch][semester]})")
            else:
                st.write("Syllabus not available for the selected branch and semester.")

    elif option == "Question Bank":
        st.title("Question Bank")
        branch = st.selectbox("Select Branch", ["IOT", "CSE", "ECE", "EE", "CE", "ME"])
        semester = st.selectbox("Select Semester", ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"])

        # Question bank links for all branches
        question_bank_links = {
            "IOT": {
                "1st": "https://example.com/iot/1st-semester-question-bank",
                "2nd": "https://example.com/iot/2nd-semester-question-bank",
                "3rd": "https://example.com/iot/3rd-semester-question-bank",
                "4th": "https://example.com/iot/4th-semester-question-bank",
                "5th": "https://example.com/iot/5th-semester-question-bank",
                "6th": "https://example.com/iot/6th-semester-question-bank",
                "7th": "https://example.com/iot/7th-semester-question-bank",
                "8th": "https://example.com/iot/8th-semester-question-bank",
            },
            "CSE": {
                "1st": "https://example.com/cse/1st-semester-question-bank",
                "2nd": "https://example.com/cse/2nd-semester-question-bank",
                "3rd": "https://example.com/cse/3rd-semester-question-bank",
                "4th": "https://example.com/cse/4th-semester-question-bank",
                "5th": "https://example.com/cse/5th-semester-question-bank",
                "6th": "https://example.com/cse/6th-semester-question-bank",
                "7th": "https://example.com/cse/7th-semester-question-bank",
                "8th": "https://example.com/cse/8th-semester-question-bank",
            },
            "ECE": {
                "1st": "https://example.com/ece/1st-semester-question-bank",
                "2nd": "https://example.com/ece/2nd-semester-question-bank",
                "3rd": "https://example.com/ece/3rd-semester-question-bank",
                "4th": "https://example.com/ece/4th-semester-question-bank",
                "5th": "https://example.com/ece/5th-semester-question-bank",
                "6th": "https://example.com/ece/6th-semester-question-bank",
                "7th": "https://example.com/ece/7th-semester-question-bank",
                "8th": "https://example.com/ece/8th-semester-question-bank",
            },
            "EE": {
                "1st": "https://example.com/ee/1st-semester-question-bank",
                "2nd": "https://example.com/ee/2nd-semester-question-bank",
                "3rd": "https://example.com/ee/3rd-semester-question-bank",
                "4th": "https://example.com/ee/4th-semester-question-bank",
                "5th": "https://example.com/ee/5th-semester-question-bank",
                "6th": "https://example.com/ee/6th-semester-question-bank",
                                "7th": "https://example.com/ee/7th-semester-question-bank",
                "8th": "https://example.com/ee/8th-semester-question-bank",
            },
            "CE": {
                "1st": "https://example.com/ce/1st-semester-question-bank",
                "2nd": "https://example.com/ce/2nd-semester-question-bank",
                "3rd": "https://example.com/ce/3rd-semester-question-bank",
                "4th": "https://example.com/ce/4th-semester-question-bank",
                "5th": "https://example.com/ce/5th-semester-question-bank",
                "6th": "https://example.com/ce/6th-semester-question-bank",
                "7th": "https://example.com/ce/7th-semester-question-bank",
                "8th": "https://example.com/ce/8th-semester-question-bank",
            },
            "ME": {
                "1st": "https://example.com/me/1st-semester-question-bank",
                "2nd": "https://example.com/me/2nd-semester-question-bank",
                "3rd": "https://example.com/me/3rd-semester-question-bank",
                "4th": "https://example.com/me/4th-semester-question-bank",
                "5th": "https://example.com/me/5th-semester-question-bank",
                "6th": "https://example.com/me/6th-semester-question-bank",
                "7th": "https://example.com/me/7th-semester-question-bank",
                "8th": "https://example.com/me/8th-semester-question-bank",
            }
        }

        # Display question bank based on selected branch and semester
        if branch and semester:
            if branch in question_bank_links and semester in question_bank_links[branch]:
                st.write(f"Question Bank for {branch} - {semester} Semester:")
                st.markdown(f"[Click here for the question bank]({question_bank_links[branch][semester]})")
            else:
                st.write("Question bank not available for the selected branch and semester.")

if __name__ == "__main__":
    main()
