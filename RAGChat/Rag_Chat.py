#working, return the insights along with the referred reviews
from flask import Flask, request, jsonify
from flask_cors import CORS
import datetime
import re
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import os

from langchain_ibm import ChatWatsonx

# Langsmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = ""
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = ""

# Pinecone API key and index setup
os.environ["PINECONE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
index_name = ''

# Set WatsonX credentials
os.environ["WATSONX_APIKEY"] = ""
os.environ["WATSONX_URL"] = "https://us-south.ml.cloud.ibm.com"
os.environ["WATSONX_PROJECT_ID"] = ""

# WatsonX model parameters
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 900,
    "min_new_tokens": 1,

}

# Initialize WatsonX LLM
chat = ChatWatsonx(
    model_id="ibm/granite-13b-chat-v2",
    url=os.environ["WATSONX_URL"],
    project_id=os.environ["WATSONX_PROJECT_ID"],
    params=parameters,
)

# Initializing the Vector Store
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Flask app setup
app = Flask(__name__)
CORS(app)

# Templates
retriever_template = """You are a retriever tasked with retrieving the most relevant documents to the question and other parameters. 
Rating:**{rating}** 
User's Question:$$ {question} $$ 
Date range:^^{date}^^
"""

retriever_prompt = ChatPromptTemplate.from_template(retriever_template)

retriever_chain = (
    retriever_prompt
    | RunnableLambda(lambda x: x.messages[0].content if hasattr(x, 'messages') and x.messages else x)
    | RunnableLambda(lambda x: {
        "question": re.search(r"\$\$(.+)\$\$", x).group(1).strip(),
        "rating": [int(r) for r in re.search(r"\*\*(.+)\*\*", x).group(1).split(',')],
        "date": re.search(r"\^\^(.+)\^\^", x).group(1).split(',')
    } if re.search(r"\$\$(.+)\$\$", x) and re.search(r"\*\*(.+)\*\*", x) and re.search(r"\^\^(.+)\^\^", x) else x)
    | RunnableLambda(lambda x: vectorstore.similarity_search_with_score(str(x["question"]), k=10, filter={"rating": {"$in": x["rating"]}, "time": {"$in": x["date"]}}))
)

insights_template = """You must take the retrieved data and generate an answer to the user's question. 
Make sure that you only use information from the retrieved data and nowhere else.
Retrieved data: {retrieved_data}
Question: {question}
"""

insights_prompt = ChatPromptTemplate.from_template(insights_template)

insights_chain = (
    RunnablePassthrough.assign(retrieved_data=retriever_chain)
    | insights_prompt
    | chat 
    | StrOutputParser()
)

# Function to run the chat with WatsonX
def chat_watsonx(user_query, rating, date_str):
    retrieved_data_tuple = retriever_chain.invoke({"question": user_query, "rating": rating, "date": date_str})
    retrieved_data_dicts = [{"page_content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in retrieved_data_tuple]
    insights_response = insights_chain.invoke({"question": user_query, "rating": rating, "date": date_str, "retrieved_data": retrieved_data_dicts})
    return {"insights": insights_response, "retrieved_data": retrieved_data_dicts}

def create_date_str(start_date, end_date):
    start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    date_list = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
    return ','.join([date.strftime("%d-%m-%Y") for date in date_list])

@app.route('/reviews-chat', methods=['POST'])
def reviews_chat():
    data = request.json
    user_query = data.get('question', '')
    rating = data.get('rating', '')
    start_date = data.get('startDate', '')
    end_date = data.get('endDate', '')
    date_str = create_date_str(start_date, end_date)
    response = chat_watsonx(user_query, rating, date_str)
    return jsonify(response)


# OpenAI API key and embeddings setup
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=os.environ["OPENAI_API_KEY"])

if __name__ == '__main__':
    app.run(debug=True, port=8000)
