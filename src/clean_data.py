import pandas as pd
import os

DATA_PATH = os.path.join("data", "crowdfunding.csv")
OUTPUT_PATH = os.path.join("data", "cleaned_crowdfunding.csv")

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"Original rows: {len(df)}")

required_columns = [
    'ID', 'name', 'category', 'main_category', 'currency',
    'deadline', 'goal', 'launched', 'pledged', 'state',
    'backers', 'country', 'usd pledged'
]
df = df[required_columns]

df = df.dropna(subset=['goal', 'pledged', 'state', 'main_category'])
df['backers'] = df['backers'].fillna(0)
df['category'] = df['category'].fillna('Unknown')
df['country'] = df['country'].fillna('Unknown')

numeric_cols = ['goal', 'pledged', 'usd pledged', 'backers']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df = df.dropna(subset=['goal', 'pledged'])

df['launched'] = pd.to_datetime(df['launched'], errors='coerce')
df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')
df['duration_days'] = (df['deadline'] - df['launched']).dt.days
df = df[df['duration_days'] > 0]

df['name'] = df['name'].str.strip()
df['main_category'] = df['main_category'].str.strip().str.title()
df['state'] = df['state'].str.lower()
df['is_successful'] = df['state'] == 'successful'

print("\nQuick summary after cleaning:")
print(df.describe(include='all'))
print(f"\nFinal rows after cleaning: {len(df)}")

os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
print(f"\n✅ Cleaned data saved to: {OUTPUT_PATH}")
print("Data cleaning completed.")