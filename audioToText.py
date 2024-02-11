from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.llms import HuggingFaceHub, openai
import speech_recognition as sr1
from dotenv import load_dotenv, get_key
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain_openai import OpenAI
from getpass import getpass
import streamlit as st
import numpy as np
# import cors
import librosa
import os

load_dotenv()

# st.set_option('server.enableCORS', True)

# OPENAI_API_KEY = getpass()
os.environ["OPENAI_API_KEY"] = get_key(key_to_get="OPENAI_API_KEY", dotenv_path=".env")


# hf = HuggingFacePipeline.from_model_id(
#     model_id="google/mobilebert-multilingual-uncased",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 10},
# )

# Function to extract acoustic features from audio

try:
    ner = HuggingFaceHub(
    repo_id = "HuggingFaceH4/zephyr-7b-beta",
    task = "text-generation",
    model_kwargs={"max_new_tokens": 250, "temperature": 0.1},
    huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
    )
except:
    print("Using openai for ner")
    ner = openai.OpenAI(OPENAI_API_KEY=get_key(key_to_get="OPENAI_API_KEY", dotenv_path=".env"))

    

st.set_page_config(page_title="Q&A demo", page_icon="🌍")
st.header("Langchain Application")

class ListOutputParser(BaseOutputParser):
    def parse(self, response: str):
        res = response.split(",")
        return [x.strip() for x in res]
    


# Prompt template for initial medical conditions and their severity
initial_prompt_template = PromptTemplate(
    input_variables=["input"],
    template="""
    You are an AI assistant that helps users facing medical emergencies. 
    At the end of this message there is input by user which is facing some symptoms maybe for a medical condition, provide only medical disorders or diseases which are most probable with their severity also explain why do you came on that conclusion (high severity is such that immediate need of medical attention, such that user can be in danger if not treated immediately, medium is such that user needs medical attention but not immediately, low is such that user can wait for some time before getting medical attention.), 
    which the user might be facing, separated by commas in order of severity.

    {input}
    """
)

firstaid_prompt_template = PromptTemplate(
    input_variables=["prediction","userssymptoms"],
    template="""
    {prediction} is the most probable medical condition that the user might be facing. The user is facing the following symptoms: {userssymptoms}.
    Please provide the first aid for the condition {prediction}.
    """ 
)

summary_prompt_template = PromptTemplate(
    input_variables=["response"],
    template="""
    {response}
    from the above given input, summarize and extract all medical disorders and diseases along with each of their severity levels such that ach pair should be in the format of severity - condition name, other than alphabets in the response there must only contain hyphen for pairs and comma after pairs starting directly with pairs.
    For example: cardiac arrest - high, dehydration - medium, etc.
    There are 4 levels of severity: very high, high, medium, low.

    """,
    output_parser=ListOutputParser()
)

try:
    llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
    huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
    )
except:
    print("Using openai for LLM")
    llm = openai.OpenAI(OPENAI_API_KEY=get_key(key_to_get="OPENAI_API_KEY", dotenv_path=".env"))

base_chain = LLMChain(llm=llm, prompt=initial_prompt_template, output_key="response")

final_chain = LLMChain(llm=ner, prompt=summary_prompt_template, output_key="result")

firstaid_chain = LLMChain(llm=llm, prompt=firstaid_prompt_template, output_key="firstaid")

def get_response(question):
    res = base_chain.invoke({"input": question})
    final_res = final_chain.invoke({"response": res["response"]})
    print(res["response"])
    firstaid_res = firstaid_chain.invoke({"prediction": final_res["result"][0], "userssymptoms": question})
    return final_res["result"], res["response"], firstaid_res["firstaid"]


input_text = st.text_input("Enter the question: ", key=input)

submit_button = st.button("Ask the question")

if submit_button:
    st.subheader("The response is:")
    if input_text == "":
        st.write("Please enter a question")
    else:
        res = get_response(input_text)
        st.subheader("First aid for the condition")
        st.write(res[1].strip())
        st.subheader("Summary of the medical conditions and their severity")
        st.write(res[2].strip())



def extract_acoustic_features(audio_data, sr):
    y = audio_data.astype(np.float32)

    # Extract pitch using librosa
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    mean_pitch = np.nanmean(pitches)

    # Extract intensity (loudness) using root mean square (RMS) amplitude
    rms = librosa.feature.rms(y=y)
    mean_intensity = np.mean(rms)

    # Extract zero crossing rate (ZCR)
    zcr = librosa.feature.zero_crossing_rate(y=y)
    mean_zcr = np.mean(zcr)

    # Calculate duration of speech
    duration = librosa.get_duration(y=y, sr=sr)

    # Calculate average pause duration (silence duration)
    pauses = librosa.effects.split(y)
    pause_durations = np.diff(pauses) / sr
    if len(pause_durations) > 0:
        mean_pause_duration = np.mean(pause_durations)
    else:
        mean_pause_duration = 0

    # Calculate speech rate (words per minute)
    # Assuming average word length is 5 characters
    num_words = len(text.split())
    speech_rate = (num_words / duration) * 60 if duration > 0 else 0

    return mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, speech_rate

# give button to upload audio file
audio_file = st.file_uploader("Upload audio file", type=["wav"])

text = ""
if audio_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.wav", "wb") as f:
        f.write(audio_file.read())

    # Load the audio data using librosa
    audio_data, sr = librosa.load("temp.wav", sr=None)
       
    # Transcribe the audio
    recognizer = sr1.Recognizer()
    with sr1.AudioFile("temp.wav") as source:
        audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        st.write(f"Transcribed text: {text}")

    mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, speech_rate = extract_acoustic_features(audio_data, sr)
    st.write(f"Mean pitch: {mean_pitch}")
    st.write(f"Mean intensity: {mean_intensity}")
    st.write(f"Mean zero crossing rate: {mean_zcr}")
    st.write(f"Duration: {duration}")
    st.write(f"Mean pause duration: {mean_pause_duration}")
    st.write(f"Speech rate: {speech_rate}")
    res = get_response(text)
    st.subheader("First aid for the condition")
    st.write(res[1].strip())
    st.subheader("Summary of the medical conditions and their severity")
    st.write(res[2].strip())
    st.write("The response is:")
    st.write(res[0])

    # Delete the temporary file
    os.remove("temp.wav")
