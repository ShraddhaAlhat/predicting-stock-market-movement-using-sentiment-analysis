import praw
import json
from datetime import datetime, timezone

# Set up your Reddit API credentials
reddit = praw.Reddit(
    client_id='eOumPJu1adSnaFQknhnmOw',       # Replace with your client ID
    client_secret='dJ4tmfjcTgiRe1DcVO4l5T__g7kYBQ',  # Replace with your client secret
    user_agent='nvidia-stock-scraper-v1',     # Replace with your user agent
)

# List of subreddits to scrape
subreddits = ['stocks', 'investing', 'nvidia', 'wallstreetbets', 'stockmarket', 'techstock',
              "pennystocks", "finance", "financialindependence", "Daytrading",
              "cryptocurrency", "LongTermInvesting", "ValueInvesting", "ETF",
              "TechNews"]

# Keyword related to Nvidia
nvidia_keywords = ["NVIDIA", "NVDA", "NVIDIA stock", "NVIDIA shares", "NVIDIA Corporation"]

# Prepare a list to hold all post data
post_data = []

# Set to track unique post IDs
seen_post_ids = set()

# Loop through each subreddit and search for Nvidia-related keywords
for subreddit_name in subreddits:
    for keyword in nvidia_keywords:
        print(f"Fetching posts for: {keyword} in subreddit: {subreddit_name}")

        try:
            # Search for posts containing the keyword
            top_posts = reddit.subreddit(subreddit_name).search(keyword, limit=800)  # Fetch up to 800 posts related to the keyword

            # Extract post details
            for post in top_posts:
                if post.id not in seen_post_ids:  # Check if post ID is unique
                    seen_post_ids.add(post.id)  # Add post ID to the set to mark it as seen

                    post_info = {
                        "stock_name": "NVIDIA",  # Hardcoded as we're searching only for Nvidia-related posts
                        "title": post.title,
                        "text": post.selftext,
                        "post_id": post.id,
                        "date": datetime.fromtimestamp(post.created_utc, timezone.utc).strftime('%Y-%m-%d %H:%M:%S'),  # Convert UTC to readable date
                        "upvotes": post.score,
                        "num_comments": post.num_comments  # Fetch only the number of comments
                    }

                    post_data.append(post_info)

        except Exception as e:
            print(f"Error fetching posts from r/{subreddit_name}: {str(e)}")  # Print the error message if something goes wrong

# Define the file path for saving the JSON data
file_path = 'nvidia_stock_posts.json'

# Save the data to a JSON file
with open(file_path, 'w') as json_file:
    json.dump(post_data, json_file, indent=4)

print(f"Data has been saved to {file_path}")
