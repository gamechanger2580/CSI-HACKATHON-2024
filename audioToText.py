from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv, get_key
import streamlit as st
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

ner = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    model_kwargs={"max_new_tokens": 250, "temperature": 0.1},
    huggingfacehub_api_token=get_key(key_to_get="HUGGINGFACEHUB_API_KEY", dotenv_path=".env")
)

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

# Prompt template for summarizing medical conditions and severity pairs
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

base_chain = LLMChain(llm=llm, prompt=initial_prompt_template, output_key="response")

final_chain = LLMChain(llm=ner, prompt=summary_prompt_template, output_key="result")

def get_response(question):
    res = base_chain.invoke({"input": question})
    final_res = final_chain.invoke({"response": res["response"]})
    print(res["response"])
    return final_res["result"]

st.set_page_config(page_title="Q&A demo", page_icon="🌍")
st.header("Langchain Application")

input_text = st.text_input("Enter the question: ", key=input)

submit_button = st.button("Ask the question")

if submit_button:
    st.subheader("The response is:")
    if input_text == "":
        st.write("Please enter a question")
    else:
        response = get_response(input_text)
        st.write(response)
