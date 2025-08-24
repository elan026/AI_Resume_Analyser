from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_resume_chain():
    model_name = "google/flan-t5-base"  # small, CPU-friendly

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

    llm = HuggingFacePipeline(pipeline=pipe)

    template = """
    You are an AI resume analyzer. Compare the following resume with the job description
    and provide a structured analysis.

    Resume: {resume_text}
    Job Description: {job_description}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["resume_text", "job_description"],
        template=template,
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
