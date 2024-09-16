import os
import logging
import ast
import re
import uuid
import threading
import io
from typing import List, Dict
from flask import Flask, request, send_file
from flask_cors import CORS
from twilio.rest import Client
import pandas as pd
import matplotlib.pyplot as plt
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx
from langchain_core.output_parsers import StrOutputParser
from query_examples import get_examples
from dotenv import load_dotenv
import matplotlib
import textwrap
import matplotlib.image as mpimg

matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)
CORS(app)

#Initialize Granite model

parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1,
    "min_new_tokens":1
}
Watson_llm = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="YOUR_API_KEY",
    project_id="YOUR_PROJECT_ID",
    params=parameters,
)
Watson_prompt = ChatPromptTemplate.from_template(
'''You are an advanced language model with access to tabular data. Your task is to provide concise and accurate answers based on the information provided. Use the data in the context to answer the user's questions, and ensure your responses are clear and directly related to the given data.
    Instructions:
    1. If the Context you get has the word "Sorry" in it, your response should always be to just return the full context as the response.
    2. Create tables whenever the context contains more than or equal to 2 rows.
    Try giving the data in a table format whenever possible. WHEN INCLUDING TABLES ALWAYS PUT THEM BETWEEN ##  ##
    Here is an example:
    User question: What are my top 3 items by sales?
    Response: Here are your Top 3 item by sales
    ##| Item Name                          | Total Sales Amount |
    |------------------------------------|---------------------|
    | Nruhdq Jduolf Euhdg                | ₹12,37,257          |
    | Sduwb Irrg Sdfndjhv                | ₹11,13,119          |
    | Plgqljkw Idqwdvb                   | ₹6,30,249           |## 
    THIS STEP IS VERY IMPORTANT
    ALSO THE TABLE ## MUST BE THE LAST THING IN THE RESPONSE, DO NOT HAVE ANY TEXT OR ANYTHING AFTER THE LAST ##!
    DO NOT OVER USE TABLES!! ONLY USE IT WHEN THERE IS TOO MUCH DATA TO DISPLAY IN A TEXT MESSAGE! (more than two rows)
    IT IS VERY IMPORTANT THAT YOU INCLUDE ALL THE DATA IN THE TABLE, DO NOT SHORTEN THE DATA OR EXCLUDE ANYTHING!!
    3. Format all numerical values in Indian Rupees using the Indian comma format and explicitly mention that the values are in Indian Rupees.
    4. If the context does not have the answer to the user's specific question, output whatever information is available in the context.
    Example question: What is my most sold item in January?
    Example context: `Item Name | Total Sales Amount`  
                    `Korean Garlic Bread(Sweet flavour tone) | ₹12,37,625` 
    Example answer: Your most sold item is Korean Garlic Bread (Sweet flavour tone) with sales of 12,37,625 rupees.

    Example question: What is my sales breakdown for January?
    Example Context: Week | Week Start Date | Week End Date | Total Sales | Food Sales | Liquor Sales | Soft Drinks Sales | Beer Sales | Wine Sales | Smokes Sales
    ---|---|---|---|---|---|---|---|---|---
    1 | 2024-01-01 | 2024-01-06 | ₹57,37,198 | ₹30,41,014 | ₹11,51,675 | ₹5,07,520 | ₹3,77,443 | ₹1,12,890 | ₹16,144
    2 | 2024-01-07 | 2024-01-13 | ₹74,38,603 | ₹35,66,007 | ₹14,48,180 | ₹6,29,265 | ₹4,68,790 | ₹1,50,430 | ₹23,535
    3 | 2024-01-14 | 2024-01-20 | ₹77,37,691 | ₹34,83,308 | ₹15,28,520 | ₹6,49,100 | ₹5,17,817 | ₹1,59,745 | ₹19,450
    Total |        |            | ₹3,25,79,055 | ₹1,59,22,035 | ₹65,49,325 | ₹28,52,170 | ₹22,04,852 | ₹6,45,360 | ₹96,667
    Example answer : Here is your sales breakdown for January:
    ##| Week | Week Start Date | Week End Date | Total Sales | Food Sales | Liquor Sales | Soft Drinks Sales | Beer Sales | Wine Sales | Smokes Sales |
    |------|-----------------|---------------|-------------|------------|--------------|-------------------|------------|------------|--------------|
    | 1    | 2024-01-01      | 2024-01-06    | ₹57,37,198  | ₹30,41,014 | ₹11,51,675   | ₹5,07,520         | ₹3,77,443  | ₹1,12,890  | ₹16,144      |
    | 2    | 2024-01-07      | 2024-01-13    | ₹74,38,603  | ₹35,66,007 | ₹14,48,180   | ₹6,29,265         | ₹4,68,790  | ₹1,50,430  | ₹23,535      |
    | 3    | 2024-01-14      | 2024-01-20    | ₹77,37,691  | ₹34,83,308 | ₹15,28,520   | ₹6,49,100         | ₹5,17,817  | ₹1,59,745  | ₹19,450      |
    | Total|                 |               | ₹3,25,79,055| ₹1,59,22,035| ₹65,49,325  | ₹28,52,170        | ₹22,04,852 | ₹6,45,360  | ₹96,667      |##
    
    User Question: {user_question}
    Context: {sql_query_result}
    Response: '''
)
chain = Watson_prompt | Watson_llm | StrOutputParser()

# Load environment variables from .env file
load_dotenv()

# Set up environment variables for Langchain
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "<YOUR_LANGCHAIN_PROJECT>"
os.environ["LANGCHAIN_ENDPOINT"] = "<YOUR_LANGCHAIN_ENDPOINT>"
os.environ["LANGCHAIN_API_KEY"] = "<YOUR_LANGCHAIN_API_KEY>"

# Set up Twilio client
twilio_account_sid = os.getenv('TWILIO_ACCOUNT_SID')
twilio_auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_whatsapp_number = 'whatsapp:PHONE_NUMBER'  # Twilio WhatsApp Sandbox number

client = Client(twilio_account_sid, twilio_auth_token)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define connection strings for different databases
DATABASES = {
    'whatsapp:PHONE_NUMBER': os.getenv('DB_CONNECTION_STRING_JAN')
}

examples = get_examples()

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

system_prefix = """You are an agent designed to interact with a SQL database.Your goal is to generate a syntactically correct {dialect} query based on the user's question, execute it, and return the result with respective column headings. Follow these instructions:
Given an input question, create a syntactically correct {dialect} query to run, always return the SQL query result in the following format, separated by `|` (pipe):
- Column headings and data values are separated by `|`.
example question: What are my top 3 best sellers?
example response:`Item Name | Total Sales Amount`  
                `Korean Garlic Bread(Sweet flavour tone) | ₹12,37,625`  
                `Party Food Packages | ₹10,29,334`
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

ALL VALUES IN THE TABLE ARE IN INDIAN RUPEES AND MUST BE FORMATTED IN INDIAN COMMA FORMAT.

If you need to filter on a proper noun or any word that you want more context on, you must ALWAYS first look up the filter value using the "search_proper_nouns" tool! Even seemingly common words may be different in the database SO USE THE TOOL EXTENSIVELY!

You have access to the following tables: {table_names}
Here are the columns in each table:
Footfall	Sno
Footfall	Booking Time
Footfall	Seated Time
Footfall	Reserved Time
Footfall	Booking Type
Footfall	Guest Name
Footfall	Guest Email
Footfall	Pax
Footfall	Section(s)
Footfall	Table(s)
Footfall	Vist Count
Footfall	Booking Status
Footfall	Deletion Type
Footfall	Source of Booking

Issue	Item_Name
Issue	Units
Issue	Issue_Name
Issue	Amount
Issue	Rate
Issue	Quantity
Issue	Date

Itemwise	ItemName
Itemwise	Rate
Itemwise	Qty
Itemwise	TotalAmount
Itemwise	Classification
Itemwise	RestaurantName
Itemwise	MenuName
Itemwise	Date

KOT	KOT
KOT	KOT Time
KOT	Table Number
KOT	Item Name
KOT	Rate
KOT	Quantity
KOT	Total Amount
KOT	Bill Number 
KOT	Restaurant Name
KOT	Classification
KOT	Type
KOT	Class Name
KOT	Date

Reviews	Rating
Reviews	Review
Reviews	Dining
Reviews	Time

Sales Book	Bill Date
Sales Book	Bill Number
Sales Book	Table Number
Sales Book	Pax
Sales Book	Gross Amount
Sales Book	Cash
Sales Book	Credit Card
Sales Book	Company
Sales Book	Food
Sales Book	Liquor
Sales Book	Soft Drinks
Sales Book	Beer
Sales Book	Wine
Sales Book	Time
Sales Book	Extracted Date
Sales Book	Smokes

Table "Issues" basically contains the requested ingredients to be issued to the department that requested it (That is the Issue_Name) IT IS BASICALLY THE RESTUARANTS PURCHASES BY DEPARTMENT!
You do not have access to NCKOT and Chat History AND AS OF RIGHT NOW QUERYING THE REVIEWS TABLE IS NOT SUPPORTED BECAUSE OF CERTAIN PROBLEMS SO TELL THE USER THIS IF THEY TRY TO ASK REVIEWS BASED QUESTIONS!!
Reviews table contains all the reviews but stuff like "Get reviews for specific items or categories." will not work it can do math queries on the ratings (show me all 1&2 star ratings) or like reviews from a specific time.

If the question does not seem related to the database, tell the user do ask something about the database.

For context all the data is from 2024 onwards! The data in the database is majorly from jan 2024.

IF YOU ARE ABLE TO FIND A SIMILAR EXAMPLE BELOW THEN RUN THAT QUERY EXACTLY. ALWAYS RUN sql_db_query AS THE FIRST TOOL (only exception is when more info on proper noun like words is needed) IF sql_db_query FAILS ONLY THEN USE ANY OF THE OTHER TOOLS AND START OFF WITH THE search_proper_nouns TOOL!!
THIS IS VERY IMPORTANT!!

IF THE QUERY RETURNS NO RESULT FROM THE DATABASE TRY AGAIN AFTER USING SOME OF THE OTHER TOOLS ON THE USERS INPUT, IF THAT DOES NOT WORK GIVE THE USER AN ALTERNATE QUESTION TO ASK LIKE: "Sorry, I could not find an answer did you mean [give them a better phrased version of the question that that you know you can answer] IF THEY SAY NO OR REPLY IN THE NEGATIVE OR THE QUERY DOES NOT RETURN ANY VALUE FROM THE DATABASE THEN REPLY WITH "I don't know, can you rephrase your question and try again please." NEVER MAKE UP YOUR OWN DATA!

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

class ChatMessageHistory:
    def __init__(self, max_messages: int = 7):  # number of messages to save in history (int =7) 
        self.messages: List[BaseMessage] = []
        self.max_messages = max_messages
        self.feedback_flag = False  # Add a flag to manage feedback

    def add_message(self, message: BaseMessage):
        self.messages.append(message)
        logging.info(f"Added message to history: {message}")
        # Keep only the most recent `max_messages` messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> List[BaseMessage]:
        logging.info(f"Retrieving messages from history: {self.messages}")
        return self.messages

    def format_history_for_prompt(self) -> str:
        formatted_messages = []
        for message in self.messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"User: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"AI: {message.content}")
        return "\n".join(formatted_messages)

    def get_latest_user_message(self) -> str:
        for message in reversed(self.messages):
            if isinstance(message, HumanMessage):
                return message.content
        return ""

    def clear(self):
        self.messages = []
        self.feedback_flag = False  # Reset the flag
        logging.info("Cleared chat history")

# High-cardinality column search (to be able to search based on closest name for items)
def query_as_list(db, query):
    res = db.run(query)
    res = [el for sub in ast.literal_eval(res) for el in sub if el]
    res = [re.sub(r"\b\d+\b", "", string).strip() for string in res]
    return list(set(res))

def get_db_connection(phone_number):
    connection_string = DATABASES.get(phone_number)
    if not connection_string:
        raise ValueError("Unauthorized user.")
    return SQLDatabase.from_uri(connection_string)

def load_database_contents(db):
    Item = query_as_list(db, "SELECT DISTINCT Item_Name FROM Issue")
    Department = query_as_list(db, "SELECT DISTINCT Issue_Name FROM Issue")
    SourceofBooking = query_as_list(db, "SELECT DISTINCT [Source of Booking] FROM Footfall")
    BookingStatus = query_as_list(db, "SELECT DISTINCT [Booking Status] FROM Footfall")
    Dishes = query_as_list(db, "SELECT DISTINCT ItemName FROM Itemwise")
    MenuName = query_as_list(db, "SELECT DISTINCT MenuName FROM Itemwise")
    return Item, Department, SourceofBooking, BookingStatus, Dishes, MenuName

def create_retriever_tool_for_user(db):
    Item, Department, SourceofBooking, BookingStatus, Dishes, MenuName = load_database_contents(db)
    vector_db = FAISS.from_texts(
        Dishes + MenuName + Item + Department + SourceofBooking + BookingStatus,
        OpenAIEmbeddings()
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 8})
    description = """Use to look up values to filter on. Input is an approximate spelling of the proper noun, output is valid proper nouns. Use the noun most similar to the search. Remember use this tool as frequently as possible, it doesn't just have to be with proper nouns you can use it anywhere you require clarity even simple and common seeming names might be different in the database."""
    return create_retriever_tool(
        retriever,
        name="search_proper_nouns",
        description=description,
    )

def parse_tables(text):
    tables = []
    parts = text.split("##")
    descriptive_text = parts[0].strip()  # Text before the first table

    for i in range(1, len(parts), 2):
        try:
            table_text = parts[i].strip()
            # Split the text into lines and find the separator line
            lines = table_text.split('\n')
            separator_index = [i for i, line in enumerate(lines) if set(line.strip()) == {'-', '|'}]

            if not separator_index:
                logging.error("No separator found in the table.")
                continue

            header_line = lines[separator_index[0] - 1]
            data_lines = lines[separator_index[0] + 1:]

            # Process header and data lines
            headers = [header.strip() for header in header_line.split('|') if header.strip()]
            data = [[cell.strip() for cell in row.split('|') if cell.strip()] for row in data_lines if row.strip()]

            # Create DataFrame if there is data
            if data:
                df = pd.DataFrame(data, columns=headers)
                tables.append(df)
            else:
                logging.error("No data found in the table.")
        except ValueError as e:
            logging.error(f"Table markers not found: {e}")
        except Exception as e:
            logging.error(f"Error parsing table: {e}")

    return descriptive_text, tables


def create_table_image(df, max_rows_per_image=30, image_format='png'):
    logging.info(f"Creating image for DataFrame with {len(df)} rows and {len(df.columns)} columns.")

    def draw_table(df, fig_width, fig_height, dpi, fontsize, wrapped_text_width):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        ax.axis('off')  # Hide the axes for a cleaner image
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)

        # Add padding to cells
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('black')
            if key[0] == 0:  # Header cells
                cell.set_fontsize(fontsize + 2)
                cell.set_text_props(weight='bold', verticalalignment='center')
                cell.set_facecolor('#D3D3D3')  # Light grey background for headers
            else:
                cell_text = str(cell.get_text().get_text())
                wrapped_text = "\n".join(textwrap.wrap(cell_text, wrapped_text_width))  # Wrap text and add newlines
                cell.set_text_props(text=wrapped_text, verticalalignment='center')
                cell.set_fontsize(fontsize)
                cell.set_height(cell.get_height() + 0.03)  # Increase row height slightly

        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0, hspace=0)
        return fig, ax

    # Function to split DataFrame into chunks
    def split_dataframe(df, chunk_size):
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        return chunks

    try:
        # Adjust figure size dynamically based on the number of columns
        fig_width = 10 + 2 * len(df.columns)
        fig_height = 0.3 * min(len(df), max_rows_per_image) + 2  # Adjust height based on number of rows

        dpi = 500  # High DPI for better quality
        fontsize = 12
        wrapped_text_width = max(10, 50 - len(df.columns) * 2)  # Adjust text wrap width based on the number of columns

        # Split DataFrame into chunks if necessary
        df_chunks = split_dataframe(df, max_rows_per_image)

        image_bytes_list = []

        for chunk in df_chunks:
            fig, ax = draw_table(chunk, fig_width, fig_height, dpi, fontsize, wrapped_text_width)

            # Save the image in-memory with compression
            img_bytes = io.BytesIO()
            plt.savefig(img_bytes, format=image_format, bbox_inches='tight', pad_inches=0.1, transparent=True)
            plt.close(fig)
            img_bytes.seek(0)

            image_bytes_list.append(img_bytes)

        logging.info("Images created in-memory")
        return image_bytes_list

    except Exception as e:
        logging.error(f"Error creating table image: {e}", exc_info=True)
        return None



# Store images in-memory
image_store = {}

# Initialize chat histories
chat_histories: Dict[str, ChatMessageHistory] = {}

# Global variable for user_retriever_tool
user_retriever_tool = None

def initialize_retriever_tool():
    global user_retriever_tool
    # Using a default database connection for initial setup
    db_connection = SQLDatabase.from_uri(os.getenv('DB_CONNECTION_STRING_JAN'))  # Adjust as needed
    user_retriever_tool = create_retriever_tool_for_user(db_connection)
    logging.info("User retriever tool initialized successfully.")

# Initialize tool right away
initialize_retriever_tool()

@app.before_request
def ensure_tool_initialization():
    global user_retriever_tool
    if user_retriever_tool is None:
        logging.info("Re-initializing user retriever tool.")
        initialize_retriever_tool()


def process_message(incoming_msg, from_number):
    try:
        if from_number not in DATABASES:
            client.messages.create(
                body="Oops! It looks like you don't have access to this data. To get authorized, please reach out to Tushar Magar at tusharmagar@daimyo.in.",
                from_=twilio_whatsapp_number,
                to=from_number
            )
            return

        if from_number not in chat_histories:
            chat_histories[from_number] = ChatMessageHistory()
        chat_history = chat_histories[from_number]

        # Define FAQ responses
        faq_responses = {
            "greetings": re.compile(r"^(hello|hi|howdy|hey|hello daimyo|hey daimyo)[!?.,]*$", re.IGNORECASE),
            "how are you": re.compile(r"^(how are you|how's it going)[!?.,]*$", re.IGNORECASE),
            "help": re.compile(r"^(help|how can you help|how can you help me|what can you do|what can you do for me)[!?.,]*$", re.IGNORECASE),
            "thank you": re.compile(r"^(thank you|thanks)[!?.,]*$", re.IGNORECASE)
        }

        # Check for FAQ responses
        if faq_responses["greetings"].match(incoming_msg):
            response = "Hello! How can I assist you today?"
        elif faq_responses["how are you"].match(incoming_msg):
            response = "I'm just a bot, but I'm here to help you. How can I assist you today?"
        elif faq_responses["help"].match(incoming_msg):
            response = '''I can assist you with querying and retrieving information from a SQL database. Here are some specific ways I can help:

1. *Sales Insights*: I can generate sales insights based on various criteria such as item-wise sales, category-wise sales, and total sales for a specific period.

2. *Footfall Analysis*: I can provide data on customer footfall, including the number of customers visiting over a specific period.

3. *Inventory and Issues*: I can give you details about inventory issues, including the quantity and cost of items issued to different departments.

5. *KOT (Kitchen Order Tickets)*: I can retrieve information related to kitchen orders, including details about specific items ordered.

6. *Financial Analysis*: I can provide financial data such as total sales amounts, average rates, and quantities sold.

If you have any specific questions or need data on a particular aspect, just let me know, and I can generate the appropriate SQL query to fetch the information for you.'''
        elif faq_responses["thank you"].match(incoming_msg):
            response = "You're welcome! If you have any other questions, feel free to ask."
        else:
            response = None

        if response:
            client.messages.create(
                body=response,
                from_=twilio_whatsapp_number,
                to=from_number
            )
            return

        # Check for feedback response
        if incoming_msg == "👎":
            client.messages.create(
                body="Thank you for your feedback. Could you please provide more details about what went wrong with the previous message?",
                from_=twilio_whatsapp_number,
                to=from_number
            )
            chat_history.feedback_flag = True  # Set the feedback flag
            return
        elif chat_history.feedback_flag:
            client.messages.create(
                body="Noted. We will investigate this issue. In the meantime, please rephrase your question and try again. Thank you for your patience!",
                from_=twilio_whatsapp_number,
                to=from_number
            )
            chat_history.feedback_flag = False  # Reset the feedback flag
            return

        # Check for positive feedback response
        if incoming_msg == "👍":
            client.messages.create(
                body="Thank you for your positive feedback! If you have any other questions or need further assistance, feel free to ask.",
                from_=twilio_whatsapp_number,
                to=from_number
            )
            return

        chat_history.add_message(HumanMessage(content=incoming_msg))

        db = get_db_connection(from_number)  # Retrieve db connection for the user

        agent = create_sql_agent(
            llm=ChatOpenAI(model="gpt-4o", temperature=0),
            db=db,
            extra_tools=[user_retriever_tool],  # Use the globally initialized tool
            prompt=full_prompt,
            verbose=True,
            agent_type="openai-tools",
        )

        full_input = f"User: {incoming_msg}\n---\n{chat_history.format_history_for_prompt()}"
        response = agent.invoke({"input": full_input})
        agent_response = response.get('output', "Sorry, I couldn't process your request.")
        chat_history.add_message(AIMessage(content=agent_response))

        logging.info(f"Agent response: {agent_response}")
        watson_response = chain.invoke({"user_question":incoming_msg, "sql_query_result": agent_response})
        descriptive_text, tables = parse_tables(watson_response)
        # Send descriptive text first
        if descriptive_text:
            client.messages.create(
                body=descriptive_text,
                from_=twilio_whatsapp_number,
                to=from_number
            )

        # Process each table and send as separate messages
        for df in tables:
            img_bytes_list = create_table_image(df)
            for img_bytes in img_bytes_list:
                image_id = str(uuid.uuid4())
                image_store[image_id] = img_bytes
                public_url = f"YOUR_NGROK_URL/images/{image_id}"
                logging.info(f"Public URL: {public_url}")
                client.messages.create(
                    body="",
                    from_=twilio_whatsapp_number,
                    to=from_number,
                    media_url=[public_url]
                )

    except ValueError as ve:
        logging.error(f"ValueError: {ve}", exc_info=True)
        client.messages.create(
            body=f"Sorry, there was a value error: {ve}",
            from_=twilio_whatsapp_number,
            to=from_number
        )
    except KeyError as ke:
        logging.error(f"KeyError: {ke}", exc_info=True)
        client.messages.create(
            body=f"Sorry, there was a key error: {ke}",
            from_=twilio_whatsapp_number,
            to=from_number
        )
    except Exception as e:
        logging.error(f"Error processing message: {e}", exc_info=True)
        client.messages.create(
            body=f"Sorry, there was an error processing your request: {e}",
            from_=twilio_whatsapp_number,
            to=from_number
        )

@app.route('/whatsapp', methods=['POST'])
def whatsapp():
    incoming_msg = request.values.get('Body', '').strip()
    from_number = request.values.get('From', '').strip()
    logging.info(f"Received WhatsApp message: {incoming_msg} from {from_number}")
    thread = threading.Thread(target=process_message, args=(incoming_msg, from_number))
    thread.start()
    return "", 200

@app.route('/images/<image_id>')
def serve_image(image_id):
    try:
        img_bytes = image_store.get(image_id)
        if img_bytes is None:
            raise ValueError("Image not found")
        return send_file(img_bytes, mimetype='image/png')
    except Exception as e:
        logging.error(f"Failed to serve image: {e}")
        return "Error serving image", 500

@app.route("/")
def hello():
    return "Powered by Daimyo, Contact: PHONE_NUMBER for sales"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8080)))
