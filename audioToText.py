from langchain_community.llms import HuggingFaceHub
import speech_recognition as sr
from dotenv import load_dotenv, get_key
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import numpy as np
import librosa


load_dotenv()

# Function to extract acoustic features from audio
def extract_acoustic_features(audio_data, sr):
    # Convert audio data to floating-point format
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

ner = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"max_new_tokens": 250, "temperature": 0.1},
    huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
)

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

llm = HuggingFaceHub(
    # repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    # task="text-generation",
      repo_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={
        # "max_new_tokens": 512,
        "max_new_tokens": 250,
        "top_k": 30,
        "temperature": 0.1,
        "repetition_penalty": 1.03,
    },
    huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
)

base_chain = LLMChain(llm=llm, prompt=initial_prompt_template, output_key="response")

final_chain = LLMChain(llm=ner, prompt=summary_prompt_template, output_key="result")

firstaid_chain = LLMChain(llm=llm, prompt=firstaid_prompt_template, output_key="firstaid")

# def get_response(question):
    # res = base_chain.invoke({"input": question})
    # final_res = final_chain.invoke({"response": res["response"]})
    # print(res["response"])
    # firstaid_res = firstaid_chain.invoke({"prediction": final_res["result"][0], "userssymptoms": question})
    # return (final_res["result"], res["response"], firstaid_res["firstaid"])
# llm = HuggingFaceHub(
#     repo_id="openai-community/gpt2",
#     task="text-generation",
#     model_kwargs={"max_new_tokens": 512, "temperature": 0.1},
#     huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
# )
chat_llm = ChatHuggingFace(llm=llm)  # Adjust model as needed

def get_response(question):
    # Initial prompt as a system message
    base_res = chat_llm.run(
        messages=[
            {"type": "system", "content": initial_prompt_template.template.format(input=question)}
        ]
    )

    # NER model as a system message
    final_res = chat_llm.run(
        messages=[{"type": "system", "content": base_res["messages"][0]["content"]}]
    )

    # First aid prompt as a system message
    firstaid_res = chat_llm.run(
        messages=[
            {
                "type": "system",
                "content": firstaid_prompt_template.template.format(
                    prediction=final_res["result"][0], userssymptoms=question
                ),
            }
        ]
    )

    return final_res["result"], base_res["messages"][0]["content"], firstaid_res["result"]
 
 
recognizer = sr.Recognizer()

button = st.button("Record audio")

if button:
    with sr.Microphone() as source:
        st.write("Please speak something to analyze your anxiety level:")
        audio_data = recognizer.listen(source)  # Record audio once

        st.write("Audio recorded")

        try:
            text = recognizer.recognize_google(audio_data)
            st.write("You said:", text)

            # Extract acoustic features (ensure `speech_rate` is returned)
            mean_pitch, mean_intensity, mean_zcr, duration, mean_pause_duration, speech_rate = extract_acoustic_features(
                audio_data, sr
            )

            st.write("Analyzing audio...")
            st.write("Speech rate:", speech_rate)

            # Classify speech rate (adjust thresholds as needed)
            if speech_rate > 200:
                st.write("Person is speaking very fast")
            elif speech_rate > 150:
                st.write("Person is speaking fast")
            elif speech_rate > 100:
                st.write("Person is speaking at a normal pace")
            else:
                st.write("Person is speaking slowly")

        except Exception as e:
            st.write("Error:", e)



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
        st.subheader("Medical conditions and their severity")
        st.write(res[0].strip())
        st.subheader("Summary of the medical conditions and their severity")
        st.write(res[2].strip())
