import json
import sys
import logging
from datetime import datetime, timedelta
import time
from google_maps_reviews import ReviewsClient
from twilio.rest import Client
from sqlalchemy import create_engine, text
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyshorteners
from langchain_core.prompts import ChatPromptTemplate
from langchain_ibm import ChatWatsonx
from langchain_core.output_parsers import StrOutputParser

# Set up logging to the terminal only
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s:%(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.info("Script started at " + str(datetime.now()))

#Initialze granite model
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "repetition_penalty": 1,
    "min_new_tokens":1
}
Watson_llm = ChatWatsonx(
    model_id="ibm/granite-34b-code-instruct",
    url="https://us-south.ml.cloud.ibm.com",
    apikey="YOUR API KEY",
    project_id="YOUR PROJECT ID",
    params=parameters,
)
prompt = ChatPromptTemplate.from_template(
    """Write a reply to this review left by a customer about our restaurant include their first name in the review if it seems relevant, be warm and friendly. Don't start with dear (name) and don't sign off, keep it like a text message.: 
    Review: {Review}
    Full Name: {Name} 
    Rating: {Rating}
    Reply:"""
)
chain = prompt | Watson_llm | StrOutputParser()

# Initialize the ReviewsClient with your API key
client = ReviewsClient(api_key='API KEY')

# Initialize URL shortener
s = pyshorteners.Shortener()

# Calculate the cutoff timestamp (8 hours before the current time)
def get_cutoff_timestamp():
    now = datetime.utcnow()
    cutoff_time = now - timedelta(hours=8)
    cutoff_timestamp = int(time.mktime(cutoff_time.timetuple()))
    logging.info(f"Cutoff timestamp: {cutoff_timestamp} (UTC time: {cutoff_time})")
    return cutoff_timestamp

# Convert a datetime to a Unix timestamp
def datetime_to_timestamp(dt):
    return int(time.mktime(dt.timetuple()))

# List of Place IDs with their corresponding names
PLACE_IDS = {
    'Place_id1': 'Restaurant_1',
    'Place_id2': 'Restaurant_2',
}

# Map of place names to WhatsApp phone numbers
PHONE_NUMBERS = {
    'Restaurant 1': ['whatsapp:<PHONE_NUMBER_1>', 'whatsapp:<PHONE_NUMBER_2>'],
    'Restaurant 2': ['whatsapp:<PHONE_NUMBER_1>', 'whatsapp:<PHONE_NUMBER_2>']
}

# Database connection using SQLAlchemy
engine = create_engine('mssql+pymssql://<DB_USER>:<DB_PASSWORD>@<DB_HOST>/<DB_NAME>', echo=False)

# Twilio configuration
account_sid = '<TWILIO_ACCOUNT_SID>'
auth_token = '<TWILIO_AUTH_TOKEN>'
twilio_client = Client(account_sid, auth_token)

# Messaging service SID
messaging_service_sid = '<TWILIO_MESSAGING_SERVICE_SID>'

# Your approved template's ContentSid
notification_content_sid = "<WHATSAPP_NOTIFICATION_CONTENT_SID>"
daily_summary_content_sid = "<WHATSAPP_DAILY_SUMMARY_CONTENT_SID>"

# Function to sanitize the review text
def sanitize_review_text(review_text):
    if review_text:
        return review_text.replace('\n', ' ').replace('\r', ' ')
    return review_text

# Function to shorten URLs
def shorten_url(url):
    try:
        return s.tinyurl.short(url)
    except Exception as e:
        logging.error(f"Failed to shorten URL {url}: {e}")
        return url

# Function to send notification via Twilio using a template
def send_template_notification(review, place_name):
    phone_numbers = PHONE_NUMBERS.get(place_name, [])
    for phone_number in phone_numbers:
        try:
            review_datetime_utc = datetime.strptime(review['review_datetime_utc'], '%m/%d/%Y %H:%M:%S')
            review_datetime_ist = review_datetime_utc + timedelta(hours=5, minutes=30)
            review_time_ist = review_datetime_ist.strftime('%H:%M')

            # Shorten the review link
            short_link = shorten_url(review['review_link'])

            # If review text is missing, replace it with "NO REVIEW WRITTEN"
            sanitized_text = sanitize_review_text(review['review_text'][:200]) if review['review_text'] else "NO REVIEW WRITTEN"

            # Generate suggested reply
            suggested_reply = generate_suggested_reply(sanitized_text, review['author_title'], review['review_rating'])

            content_variables = json.dumps({
                "1": review['author_title'],
                "2": str(review['review_rating']) + " star",
                "3": review_time_ist,
                "4": sanitized_text,
                "5": short_link,
                "6": place_name,
                "7": suggested_reply
            })

            logging.info(f"Sending message to {phone_number} with variables: {content_variables}")

            message = twilio_client.messages.create(
                to=phone_number,
                content_sid=notification_content_sid,
                content_variables=content_variables,
                messaging_service_sid=messaging_service_sid
            )

            logging.info(f"Notification sent to {phone_number} for review ID: {review['review_id']} with SID: {message.sid}")

        except Exception as e:
            logging.error(f"Failed to send notification to {phone_number}: {e}")

# Function to insert a review into the database
def insert_review(review, place_name):
    review_questions = review.get("review_questions")
    if review_questions and review_questions != 'None':
        review_questions = json.dumps(review_questions)
    else:
        review_questions = '{}'

    logging.info(f"Review Data: {json.dumps(review, default=str)}")

    sql_insert = text('''
        INSERT INTO GoogleReviews_watsonx (
            place_name, google_id, review_id, author_title, author_id, 
            review_text, review_questions, review_link, review_rating, 
            review_datetime_utc, author_reviews_count, author_ratings_count, 
            sentiment_score
        )
        SELECT 
            :place_name, :google_id, :review_id, :author_title, :author_id, 
            :review_text, :review_questions, :review_link, :review_rating, 
            :review_datetime_utc, :author_reviews_count, :author_ratings_count, 
            :sentiment_score
        WHERE NOT EXISTS (
            SELECT 1 FROM GoogleReviews_watsonx WHERE review_id = :review_id
        )
    ''')


    try:
        with engine.connect() as connection:
            result = connection.execute(sql_insert, {
                "place_name": place_name,
                "google_id": review["google_id"],
                "review_id": review["review_id"],
                "author_title": review["author_title"],
                "author_id": review["author_id"],
                "review_text": review.get("review_text", ""),
                "review_questions": review_questions,
                "review_link": review["review_link"],
                "review_rating": review["review_rating"],
                "review_datetime_utc": review["review_datetime_utc"],
                "author_reviews_count": review.get("author_reviews_count", 0),
                "author_ratings_count": review.get("author_ratings_count", 0),
                "sentiment_score": review.get("sentiment_score", None)
            })

            if result.rowcount > 0:
                connection.commit()
                logging.info(f"Inserted review ID: {review['review_id']}")

                review_text = review.get('review_text', '')
                if review['review_rating'] <= 3 or (review_text and len(review_text) > 100):
                    logging.info(f"Review ID {review['review_id']} meets criteria for notification.")
                    send_template_notification(review, place_name)
                else:
                    logging.info(f"Review ID {review['review_id']} does not meet criteria for notification.")
            else:
                logging.info(f"Review ID {review['review_id']} already exists in the database.")
    except Exception as e:
        logging.error(f"Error inserting review: {e}")

# Function to fetch and store reviews
def fetch_and_store_reviews(place_id, place_name, cutoff_timestamp):
    logging.info(f"Fetching reviews for Place ID: {place_id} ({place_name})")
    
    try:
        reviews_data = client.get_reviews(place_id, cutoff=cutoff_timestamp, reviewsLimit=0)

        if not reviews_data or 'reviews_data' not in reviews_data[0]:
            logging.info(f"No new reviews found for Place ID: {place_id} ({place_name})")
            return

        reviews = reviews_data[0].get('reviews_data', [])

        for review in reviews:
            review_timestamp = datetime_to_timestamp(datetime.strptime(review['review_datetime_utc'], '%m/%d/%Y %H:%M:%S'))
            if review_timestamp > cutoff_timestamp:
                logging.info(f"Inserting review ID: {review.get('review_id')}")
                insert_review(review, place_name)

    except Exception as e:
        logging.error(f"Error fetching reviews for Place ID: {place_id} ({place_name}): {e}")

    logging.info(f"Finished fetching reviews for Place ID: {place_id} ({place_name})")

# Function to check if a summary was already sent today
def summary_already_sent(place_name):
    query = text('''
        SELECT COUNT(*) FROM SentNotification_watsonx 
        WHERE place_name = :place_name 
        AND notification_type = 'GoogleReviewsDailySummary' 
        AND sent_date = CAST(GETDATE() AS DATE)
    ''')
    with engine.connect() as connection:
        result = connection.execute(query, {"place_name": place_name}).scalar()
    return result > 0

# Function to log sent notification
def log_sent_notification(place_name):
    sql_insert = text('''
        INSERT INTO SentNotification_watsonx (place_name, notification_type, sent_date)
        VALUES (:place_name, 'GoogleReviewsDailySummary', CAST(GETDATE() AS DATE))
    ''')
    try:
        with engine.connect() as connection:
            connection.execute(sql_insert, {"place_name": place_name})
            connection.commit()
    except Exception as e:
        logging.error(f"Error inserting sent notification: {e}")

# Updated daily summary function with logging logic
def send_daily_summary():
    logging.info("Generating daily summary...")

    # Calculate the start and end UTC timestamps for "yesterday" in IST
    now = datetime.utcnow()
    start_utc = (now - timedelta(days=2)).replace(hour=18, minute=30, second=0, microsecond=0)
    end_utc = (now - timedelta(days=1)).replace(hour=18, minute=30, second=0, microsecond=0)

    logging.info(f"Calculated start_utc: {start_utc}, end_utc: {end_utc}")

    for place_name in PLACE_IDS.values():
        if summary_already_sent(place_name):
            logging.info(f"Daily summary for {place_name} has already been sent today.")
            continue

        # Query the reviews data
        query = text('''
            SELECT
                COUNT(*) AS total_reviews,
                CAST(ROUND(AVG(CAST(review_rating AS FLOAT)), 1) AS DECIMAL(3, 1)) AS average_rating,
                SUM(CASE WHEN review_rating = 5 THEN 1 ELSE 0 END) AS five_star_reviews,
                SUM(CASE WHEN review_rating = 4 THEN 1 ELSE 0 END) AS four_star_reviews,
                SUM(CASE WHEN review_rating = 3 THEN 1 ELSE 0 END) AS three_star_reviews,
                SUM(CASE WHEN review_rating = 2 THEN 1 ELSE 0 END) AS two_star_reviews,
                SUM(CASE WHEN review_rating = 1 THEN 1 ELSE 0 END) AS one_star_reviews
            FROM GoogleReviews
            WHERE place_name = :place_name 
            AND review_datetime_utc >= :start_utc
            AND review_datetime_utc < :end_utc
        ''')

        with engine.connect() as connection:
            result = connection.execute(query, {"place_name": place_name, "start_utc": start_utc, "end_utc": end_utc}).fetchone()

        if result:
            total_reviews = result[0]
            average_rating = round(result[1], 1) if result[1] else 0
            five_star_reviews = result[2]
            four_star_reviews = result[3]
            three_star_reviews = result[4]
            two_star_reviews = result[5]
            one_star_reviews = result[6]

            content_variables = json.dumps({
                "1": place_name,
                "2": (datetime.utcnow() - timedelta(days=1)).strftime('%d %b %Y'),
                "3": str(total_reviews),
                "4": str(average_rating),
                "5": str(five_star_reviews),
                "6": str(four_star_reviews),
                "7": str(three_star_reviews),
                "8": str(two_star_reviews),
                "9": str(one_star_reviews)
            })

            phone_numbers = PHONE_NUMBERS.get(place_name, [])
            for phone_number in phone_numbers:
                try:
                    message = twilio_client.messages.create(
                        to=phone_number,
                        content_sid=daily_summary_content_sid,
                        content_variables=content_variables,
                        messaging_service_sid=messaging_service_sid
                    )
                    logging.info(f"Daily summary sent to {phone_number} for {place_name} with SID: {message.sid}")
                    log_sent_notification(place_name)  # Log the sent notification after sending
                except Exception as e:
                    logging.error(f"Failed to send daily summary to {phone_number}: {e}")

# Main function to run both review fetching and daily summary
def main():
    cutoff_timestamp = get_cutoff_timestamp()

    # Fetch reviews
    with ThreadPoolExecutor(max_workers=len(PLACE_IDS)) as executor:
        futures = [executor.submit(fetch_and_store_reviews, place_id, place_name, cutoff_timestamp) for place_id, place_name in PLACE_IDS.items()]
        for future in as_completed(futures):
            try:
                future.result()  # This will re-raise any exception from the thread
            except Exception as e:
                logging.error(f"Error during parallel execution: {e}")

    # Send daily summary
    send_daily_summary()

def generate_suggested_reply(review_text, author_name, rating):
    try:
        response = chain.invoke({"Review": review_text, "Name": author_name, "Rating": rating})
        clean_response = response.replace('\n', '')
        return clean_response
    except Exception as e:
        logging.error(f"Failed to generate suggested reply: {e}")
        return "We appreciate your feedback and will strive to improve."


if __name__ == "__main__":
    main()
    logging.info("Review fetching and daily summary process completed.")