from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

load_dotenv()

def main():

    information = """
    Elon Musk vision towards spacex and tesla.
    """

    summary_prompt_template = """
    Summarize the following information and add the vision of its future:
    1. vision
    2. mission
    3. values
    4. goals
    5. challenges
    {information}
    """
    summary_prompt_template = PromptTemplate(
        template=summary_prompt_template,
        input_variables=["information"]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # llm = ChatOllama(model="gemma4:12b-it-qat", temperature=0)

    llm_chain = summary_prompt_template | llm
    print("Calling Ollama (this can take a while on a 12B model)...", flush=True)
    result = llm_chain.invoke({"information": information})
    print(result.content, flush=True)

if __name__ == "__main__":
    main()
