# Database Chat System

## Overview
This system allows users to ask natural language questions via a chat interface. The questions are processed, converted into SQL queries, and then executed on a connected database. The results are formatted and presented back to the user.

## Database Chat Flow

Here’s how the system works:

1. **User asks a question**: The user sends a natural language query via WhatsApp.
2. **Twilio forwards the message**: Twilio API handles the message and forwards it to the server.
3. **Flask API processes the request**: The server receives the request and sends the question to the SQL Query Generator Model.
4. **SQL Query Generator creates a query**: The model converts the user's query into an SQL statement.
5. **Database executes the query**: The SQL query is run, and the results are retrieved.
6. **Watsonx Model generates a response**: The results are used by the Watsonx model to generate a user-friendly response, including tables if necessary.
7. **Response is returned to the user**: The final response is sent back to the user via WhatsApp.

![Database Chat Flow](images/database_chat.png)

## Features

- Natural language to SQL query conversion
- Dynamic response generation using IBM Watsonx
- Seamless integration with WhatsApp via Twilio
- Automatic table generation from query results
