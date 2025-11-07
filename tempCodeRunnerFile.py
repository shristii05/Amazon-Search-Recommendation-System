import streamlit as st
import pandas as pd
from PIL import Image
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_data
def load_data():
    df = pd.read_csv("amazon_product.csv") 
    df["Title"] = df["Title"].fillna("")
    df["Description"] = df["Description"].fillna("")
    df["Text"] = df["Title"] + " " + df["Description"]
    return df

amazon_df = load_data()

# NLP preprocessing
stemmer = SnowballStemmer("english")
nltk.download('punkt', quiet=True)

def tokenize_stem(text):
    tokens = nltk.word_tokenize(text.lower())
    stems = [stemmer.stem(t) for t in tokens]
    return stems

@st.cache_resource
def build_tfidf(df):
    vectorizer = TfidfVectorizer(tokenizer=tokenize_stem, stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["Text"])
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_tfidf(amazon_df)

# Search function
def search_products(query, top_n=5):
    if not query.strip():
        return pd.DataFrame()
    query_vec = vectorizer.transform([query])
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = cosine_sim.argsort()[-top_n:][::-1]
    results = amazon_df.iloc[top_indices][["Title", "Description","Category"]].copy()
    results["Similarity"] = cosine_sim[top_indices]
    return results

# Recommendation function
def recommend_products(selected_index, top_n=5):
    cosine_sim_matrix = cosine_similarity(tfidf_matrix[selected_index], tfidf_matrix).flatten()
    indices = cosine_sim_matrix.argsort()[-top_n-1:][::-1]
    indices = [i for i in indices if i != selected_index][:top_n]
    recommended = amazon_df.iloc[indices][["Title", "Description","Category"]].copy()
    recommended["Similarity"] = cosine_sim_matrix[indices]
    return recommended


# App UI

img = Image.open("img.webp") 
st.image(img, width=600)
st.title("🛍️ Amazon Product Search & Recommendations")
st.markdown("""
Search for a product and get **top matches** along with **recommended similar products**.
""")

query = st.text_input("🔎 Enter a product name or description:")
submit = st.button("Search")

if submit:
    if not query.strip():
        st.warning("Please enter a product name or description.")
    else:
        results = search_products(query, top_n=10)
        if results.empty:
            st.error("No matching products found 😕")
        else:
            st.success(f"Top {len(results)} products for '{query}':")
            for i, row in results.iterrows():
                st.markdown(f"### {row['Title']}")
                st.write(f"**Category:** {row.get('Category','N/A')}")
                st.write(f"**Description:** {row.get('Description','')[:300]}...")
                st.progress(min(1.0, row['Similarity']))
                st.write("---")
            # Recommended Section
            st.subheader("💡 Recommended Products based on top match")
            top_product_index = results.index[0]  
            recommended = recommend_products(top_product_index, top_n=5)
            for i, row in recommended.iterrows():
                st.markdown(f"### {row['Title']}")
                st.write(f"**Category:** {row.get('Category','N/A')}")
                st.write(f"**Description:** {row.get('Description','')[:300]}...")
                st.progress(min(1.0, row['Similarity']))
                st.write("---")
