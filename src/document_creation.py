import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Gemini API key
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY not found. Please add it in the .env file.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Define 10 document themes
themes = [
    "Overall summary of crowdfunding campaigns and key statistics",
    "Analysis of successful campaigns and their success factors",
    "Insights into failed campaigns and common challenges",
    "Trends in Film & Video campaigns",
    "Trends in Music campaigns",
    "Design category campaigns and innovation patterns",
    "Food campaigns and sustainability trends",
    "Country-wise performance comparison of campaigns",
    "Patterns in backers and pledges distribution",
    "Trends in campaign goals vs actual pledged outcomes"
]

# Load cleaned dataset
df = pd.read_csv("data/cleaned_crowdfunding.csv")

# Create docs folder if it doesn't exist
os.makedirs("docs", exist_ok=True)

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# Generate 10 detailed documents
for i, theme in enumerate(themes, start=1):
    sample = df.sample(min(2000, len(df)), random_state=i)
    snippet = sample.head(100).to_string(index=False)

    prompt = f"""
    You are a data analyst working on crowdfunding insights.
    Write a detailed 400–600 word report about: {theme}

    Use the dataset snippet below as reference data.
    Include:
    - Clear subheadings (Overview, Trends, Observations, Examples, Conclusion)
    - Insights about success or failure patterns
    - Trends across categories, goals, and pledges
    - A short conclusion summarizing key points

    Dataset sample:
    {snippet}
    """

    response = model.generate_content(prompt)
    text = response.text.strip()

    with open(f"docs/document_{i}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Document {i} generated: {theme}")

print("All 10 documents generated and saved in docs/ folder.")
