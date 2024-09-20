# RAG Chat

## Overview

This system analyzes customer reviews and provides actionable insights for businesses. It utilizes the Watsonx Granite model for processing, LangChain for natural language processing and retrieval, and Twilio for notifications. The system also leverages Pinecone for vector storage and retrieval.

![RAG chat](/images/RAG_Chat.png)

## Dashboard
![Dashboard1](/images/Dashboard1.png)
![Dashboard2](/images/Dashboard2.png)

## Components

### 1. Flask App

- **Endpoint**: `/reviews-chat`
- **Function**: Receives POST requests with review data and user queries, processes them, and returns insights.

### 2. Watsonx Granite Model

- **ChatWatsonx**: Processes reviews and generates insights using the IBM Watsonx model.

### 3. Database

- **Pinecone**: Vector database that stores review data and supports similarity searches.

## Workflow

1. **Review Collection**: Collect reviews from various sources (e.g., social media, websites).
2. **Data Processing**: Send collected reviews to the Watsonx Granite model for analysis.
3. **Data Storage**: Store processed data in Pinecone for vector-based retrieval.

