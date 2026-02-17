import pandas as pd

# Load CSV files
df = pd.read_csv("implicitmis_sd_implied.csv")   # reference dataset
df1 = pd.read_csv("dataset.csv")                 # target dataset

# Preprocess: list of main_posts (drop NaN, ensure string)
main_posts = (
    df["main_post"]
    .dropna()
    .astype(str)
    .tolist()
)

def format_text(full_text):
    if not isinstance(full_text, str):
        return full_text

    for main_post in main_posts:
        if full_text.startswith(main_post):
            comment = full_text[len(main_post):].strip()
            return f"Post: {main_post}\nComments: {comment}"

    # No match → return original text
    return full_text


# Apply once (linear, not quadratic in practice)
df1["text"] = df1["text"].apply(format_text)


df1.to_csv("dataset_formatted.csv", index=False, encoding="utf-8")
