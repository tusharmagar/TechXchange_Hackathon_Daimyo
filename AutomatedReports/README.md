# Overview

This system is designed to analyze and report on customer reviews, providing valuable insights for businesses. It leverages the Watsonx Granite model to process reviews, Twilio for WhatsApp notifications, and a custom-built report generator.

![Report Generation](/images/Report_generation.png)
## Components and Functionality

### Watsonx Granite Model

- **Processes Batches of Reviews**: Handles both positive and negative reviews.
- **Extracts Key Information**: Includes sentiment analysis and key details from the reviews.

### Database

- **Storage**: Stores processed review data for further analysis and reporting.

### Report Generator Code

Generates detailed review summaries, including:

- **Overall Sentiment Analysis**: Summarizes the general sentiment of the reviews.
- **Common Themes and Topics**: Identifies prevalent themes and topics discussed in the reviews.
- **Key Phrases and Keywords**: Highlights important phrases and keywords.
- **Rating Distributions Over Time**: Shows how ratings have varied over a specified period.
- **Weekly Comparisons**: Compares review metrics on a weekly basis.

### Twilio

- **WhatsApp Notifications**: Sends generated reports via WhatsApp to designated recipients.

## Workflow

1. **Review Collection**: Reviews are collected from various sources such as social media and websites.
2. **Data Processing**: Reviews are sent to the Watsonx Granite model for processing.
3. **Data Storage**: Processed review data is stored in the database.
4. **Report Generation**: The report generator analyzes the stored data and creates detailed reports.
5. **Notification**: Generated reports are sent via WhatsApp using Twilio.
