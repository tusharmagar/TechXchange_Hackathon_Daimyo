# Custom Alerts

## Overview
This system monitors Google Reviews for specific places and handles them through a series of automated processes. The system fetches and processes new reviews, generates suggested replies using an AI model, and sends notifications or daily summaries via WhatsApp. It also stores reviews and notification statuses in a database.

![Custom Alerts](Custom_alerts.png)



## System Flow

### 1. Fetch and Store Reviews
- **Review Fetching**:
  - **Trigger**: The script starts by calculating a cutoff timestamp, which is 8 hours before the current time.
  - **Process**: Reviews are fetched from the Google Maps API for each place, starting from the cutoff timestamp.
  - **Storage**: Each review is inserted into a SQL database, ensuring it doesn't already exist. If the review meets certain criteria, a notification is sent.

### 2. Generate and Send Notifications
- **Review Processing**:
  - **Text Sanitization**: Review text is sanitized and URLs are shortened.
  - **Suggested Reply**: A suggested reply is generated using the IBM Watsonx model.
  - **Notification**: The review details, including the suggested reply, are sent to designated WhatsApp numbers using Twilio. The message is sent via a predefined template.

### 3. Send Daily Summary
- **Summary Calculation**:
  - **Timestamp**: Calculate the start and end timestamps for the previous day.
  - **Query Execution**: Aggregate review data for the previous day from the database.
  - **Notification**: A summary of review statistics is sent to WhatsApp numbers associated with each place.
  - **Logging**: Sent notifications are logged in the database to prevent duplicate summaries for the same day.

### 4. Main Execution Flow
- **Concurrency**: Uses `ThreadPoolExecutor` to fetch reviews concurrently for multiple places.
- **Daily Summary Execution**: After fetching reviews, the system generates and sends daily summaries.

## Features
- **Automated Review Fetching**: Retrieves new reviews from Google Maps and processes them.
- **Dynamic Response Generation**: Uses IBM Watsonx to generate personalized replies to reviews.
- **Notification System**: Sends review notifications and daily summaries via WhatsApp using Twilio.
- **Database Integration**: Stores reviews and notification statuses in a SQL database.
- **Concurrency Handling**: Efficiently handles multiple places and reviews using threading.
