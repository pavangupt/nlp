import os
import tensorflow as tf
import gdown
import tensorflow_hub as hub
# Dataset for Question-Answering
from tensorflow.python.framework.test_ops import none
import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForQuestionAnswering
from PIL import Image
import speech_recognition as sr

speech_r = sr.Recognizer()

qa_image = Image.open('question_answering.jpg')
sa_img = Image.open('sentiment_analysis.jpg')
st_img = Image.open('speechtotext.png')

tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

#df = pd.read_csv("CoQA_data.csv")
#random_num = np.random.randint(0, len(df))
#question = df["question"][random_num]
#text = df["text"][random_num]

#df = pd.read_json("http://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json")
#df = df.drop("version", axis=1)

# dataset for sentiment analysis
#df_sent = pd.read_csv("sms_spam.csv")

# model for sentiment analysis
sent_model_url = "https://drive.google.com/uc?id=1--eULExMNhEKGiY4zZmdSB7dvMwh0nOX"
sent_model_path = './Best_model_emotion.h5'

# downloading model and tokenzier for speech to txt
#speech_tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
#speech_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


#def load_data():
 #   global df
#    # required columns in our dataset
#    columns = ["text", "question", "answer"]

#    comp_list = []
#    for index, row in df.iterrows():
#        for i in range(len(row["data"]["questions"])):
#            temp_list = []
#            temp_list.append(row["data"]["story"])
#            temp_list.append(row["data"]["questions"][i]["input_text"])
#            temp_list.append(row["data"]["answers"][i]["input_text"])
#            comp_list.append(temp_list)
#    new_df = pd.DataFrame(comp_list, columns=columns)
#    # saving the dataframe to csv file for further loading
#    new_df.to_csv("CoQA_data.csv", index=False)
#    return new_df


def store_pred(text, pred, val):
    return none


def load_model():
    global sent_model_path
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(sent_model_path):
        if os.path.getsize(sent_model_path) >= 2770000:
            st.warning("model already there haha")

    else:
        try:
            weights_warning = st.warning("Downloading %s..." % sent_model_path)
            gdown.download(sent_model_url, output=sent_model_path)
            st.warning('download finished')
        finally:
            st.write('thanks for the patience')

    hub_layer = hub.KerasLayer(
        "https://tfhub.dev/google/universal-sentence-encoder/4")
    sent_model = tf.keras.models.load_model(sent_model_path, custom_objects={'KerasLayer': hub_layer})
    return sent_model


@st.cache(suppress_st_warning=True)
def sent_anal_app(texts):
    # loading the model
    sent_model = load_model()
    # predicting using pretrained model
    sent_prediction = sent_model.predict([texts])

    if round(sent_prediction[0][0]) == 1:
        val = "neg"
        st.title("This is negative sentence 😞")
    else:
        st.title("This is Positive Sentence 😉 ")
        val = "pos"

    store_pred(text=texts, pred=sent_prediction[0][0], val=val)
    return


# def speechtotext(audiofile):
#   file = audiofile
#    speech, rate = librosa.load(file, sr=16000)  # sampling rate is 16000
#   # converting vectors into pytorch tensors
#    input_values = speech_tokenizer(speech, return_tensors="pt").input_values
#
#   # storing logits (Non-normalized prediction)
#    logits = speech_model(input_values).logits

# store predicted ids
#    predicted_ids = torch.argmax(logits, dim=-1)
# decoding the array
#   transcription = speech_tokenizer.decode(predicted_ids[0])
#    return transcription

def speech_recognizer(file):
    audio_file = file
    with sr.AudioFile(audio_file) as source:
        audio = speech_r.record(source)

    try:
        textdata = speech_r.recognize_google(audio)
        print("Your Audio Text is:", textdata)

    except sr.UnknownValueError:
        print("Audio Error!")

    except sr.RequestError as e:
        print("Could not be able to request from google API in the moments! ")


def question_answer(text: object, question: object) -> object:
    # tokenize question and text as a pair
    input_ids = tokenizer.encode(question, text)

    # string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # segment IDs
    # first occurences of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)
    # number of tokens in segment A (question)
    num_seg_a = sep_idx + 1
    # number of tokens in segment B (text)
    num_seg_b = len(input_ids) - num_seg_a

    # list of 0s and 1s for segment embeddings
    segment_ids = [0] * num_seg_a + [1] * num_seg_b
    assert len(segment_ids) == len(input_ids)

    # model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))

    # tokens with the highest start and end scores
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start + 1, answer_end + 1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]

    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."

    return answer.capitalize()


def question_answer_condition(text, question):
    while True:
        question_answer(text, question)
        flag = True
        flag_N = False
        while flag:
            response = st.text_input('Do you want to ask another question based on this text (Y/N)?', 'Y')
            st.write("You have selected:-", response)
            if response[0] == "Y":
                question = st.text_area("Enter the next Questions")
                flag = False
            elif response[0] == "N":
                print("Thank you for your time!")
                flag = False
                flag_N = True
        if flag_N == True:
            break


# sent_analysis_model= load_model("model_sent_analysis.h5")

def main():
    option = st.sidebar.selectbox("Which NLP Task You Want To Perform!",
                                  ("Question_Answering", "Sentiment_Analysis", "Speech-To-Text"))
    st.sidebar.write("You selected:--", option)
    # new_dfs = preprocess.load_data()
    if option == "Question_Answering":
        st.sidebar.image(qa_image, caption="Question Answering")
        st.subheader("Question Answering Model")
        texts = st.text_area("Enter Your Text/Paragraph")
        questions = st.text_area("Enter Your Questions")
        if st.button("Predict"):
            answers = question_answer(texts, questions)
            st.success(answers)

    if option == "Sentiment_Analysis":
        st.sidebar.image(sa_img, caption="Sentiment Analysis")
        st.title("Sentiment Analysis Model (Emotion Detection!)")

        texts = st.text_area("Enter the text", """This is the sample text! """)
        if st.button("Predict"):
            result = sent_anal_app(texts)
            st.success(result)

    if option == "Speech-To-Text":
        st.sidebar.image(st_img, caption="Speech To Text Conversion")
        st.title('Speech to Text Conversion')
        audio_file = st.file_uploader("Please upload your Audio file (wav. format)!")

        st.audio(audio_file)
        st.write("Sample Audio for the reference!")
        if st.button('Convert'):
            if audio_file is not None:
                audio_text = speech_recognizer(audio_file)
                st.success(audio_text)


if __name__ == "__main__":
    main()
