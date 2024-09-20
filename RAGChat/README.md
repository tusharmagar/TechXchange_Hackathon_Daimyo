# RAG Chat

## Overview

This system analyzes customer reviews and provides actionable insights for businesses. It utilizes the Watsonx Granite model for processing, LangChain for natural language processing and retrieval, and Twilio for notifications. The system also leverages Pinecone for vector storage and retrieval.

![RAG chat](/images/RAG_Chat.png)

## Dashboard
![Dashboard1](/images/Dashboard1.png)
![Dashboard2](/images/Dashboard2.png)

## Components

### 1. Next.js Frontend

- **Endpoint**: `/reviews-chat`
- **Function**: Receives POST requests with review data and user queries, processes them, and returns insights.

### 2. Watsonx Granite Model

- **ChatWatsonx**: Processes reviews and generates insights using the IBM Watsonx model.

### 3. Database

- **Pinecone**: Vector database that stores review data and supports similarity searches.

## Workflow

1. **Review Collection**: Collect reviews from Google Reviews (with plans to expand to other sources).
2. **Review Storage**: Store collected reviews in a vector database.
3. **Dashboard Metrics**: Currently, the dashboard displays dummy data.
4. **RAG Chat Insights**: When RAG Chat is run, it performs retrieval-augmented generation (RAG) on the vector database, sending insights along with the relevant reviews that were referenced.

