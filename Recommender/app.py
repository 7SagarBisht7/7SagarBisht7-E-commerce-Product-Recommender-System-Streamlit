import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import re


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_spacy():
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        return nlp
    except Exception:
        return None

st.set_page_config(page_title="E‑Commerce Product Recommender", layout="wide")

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    return df

def safe_str(x):
    try:
        return str(x)
    except Exception:
        return ""

def simple_clean(text: str) -> str:
    text = safe_str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_data(show_spinner=False)
def preprocess_data(raw: pd.DataFrame, use_spacy: bool) -> pd.DataFrame:

    colmap = {
        'Uniq Id': 'ID',
        'Product Id': 'ProdID',
        'Product Rating': 'Rating',
        'Product Reviews Count': 'ReviewCount',
        'Product Category': 'Category',
        'Product Brand': 'Brand',
        'Product Name': 'Name',
        'Product Image Url': 'ImageURL',
        'Product Description': 'Description',
        'Product Tags': 'Tags',
        'Product Contents': 'Contents'
    }
    df = raw.copy()
    df.rename(columns={k:v for k,v in colmap.items() if k in df.columns}, inplace=True)

    keep = ['ID','ProdID','Rating','ReviewCount','Category','Brand','Name','ImageURL','Description','Tags']
    df = df[[c for c in keep if c in df.columns]]

    for c in ['Rating','ReviewCount']:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    for c in ['Category','Brand','Description','Tags','Name','ImageURL']:
        if c in df.columns:
            df[c] = df[c].fillna(" ")

    if 'ID' in df.columns:
        df['ID'] = df['ID'].astype(str).str.extract(r'(\d+)')
        df['ID'] = pd.to_numeric(df['ID'], errors='coerce').fillna(0).astype(int)
    else:
        df['ID'] = 0

    if 'ProdID' in df.columns:
        df['ProdID'] = df['ProdID'].astype(str).str.extract(r'(\d+)')
        df['ProdID'] = pd.to_numeric(df['ProdID'], errors='coerce').fillna(0).astype(int)
    else:
        df['ProdID'] = np.arange(1, len(df)+1)

    cat = df.get('Category', pd.Series([""]*len(df)))
    br  = df.get('Brand', pd.Series([""]*len(df)))
    desc= df.get('Description', pd.Series([""]*len(df)))

    if use_spacy:
        nlp = get_spacy()
        if nlp is None:
            use_spacy = False

    if use_spacy:
        from spacy.lang.en.stop_words import STOP_WORDS
        toks = []
        for txt in (cat + " " + br + " " + desc):
            doc = nlp(safe_str(txt).lower())
            words = [t.text for t in doc if t.is_alpha and t.text not in STOP_WORDS]
            toks.append(" ".join(words))
        tags = toks
    else:
        tags = (cat + " " + br + " " + desc).apply(simple_clean)

    df['Tags'] = tags

    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    df['ReviewCount'] = pd.to_numeric(df['ReviewCount'], errors='coerce').fillna(0).astype(int)

    return df

@st.cache_resource(show_spinner=False)
def build_content_index(df: pd.DataFrame) -> Tuple[TfidfVectorizer, any, dict]:
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Tags'])
    name_to_index = {n:i for i,n in enumerate(df['Name'])}
    return vectorizer, X, name_to_index

@st.cache_resource(show_spinner=False)
def build_user_item(df: pd.DataFrame):
    user_item = df.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    return user_item

# ========== Recommenders ==========
def recommend_trending(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    rec = (
        df.groupby(['Name','Brand','ImageURL'], as_index=False)
          .agg(Rating=('Rating','mean'), ReviewCount=('ReviewCount','max'))
          .sort_values(['Rating','ReviewCount'], ascending=[False, False])
          .head(top_n)
    )
    return rec

def recommend_content(df: pd.DataFrame, X, name_to_index, item_name: str, top_n: int = 10) -> pd.DataFrame:
    if item_name not in name_to_index:
        return pd.DataFrame(columns=['Name','Brand','ImageURL','Rating','ReviewCount','Score'])
    idx = name_to_index[item_name]
    sims = cosine_similarity(X[idx], X).ravel()
    order = np.argsort(-sims)
    order = order[order != idx][:top_n]
    rec = df.iloc[order][['Name','Brand','ImageURL','Rating','ReviewCount']].copy()
    rec['Score'] = sims[order]
    return rec

def recommend_collab(df: pd.DataFrame, user_item: pd.DataFrame, target_user_id: int, top_n: int = 10) -> pd.DataFrame:
    if target_user_id not in user_item.index:
        return pd.DataFrame(columns=['Name','Brand','ImageURL','Rating','ReviewCount','ProdID'])

    A = user_item.values
    target_idx = user_item.index.get_loc(target_user_id)
    sims = cosine_similarity(A[target_idx:target_idx+1], A).ravel()
    sims[target_idx] = 0

    target_ratings = user_item.iloc[target_idx]
    not_rated = target_ratings == 0
    if not not_rated.any():
        return pd.DataFrame(columns=['Name','Brand','ImageURL','Rating','ReviewCount','ProdID'])

    weighted_scores = sims @ A 
    scores = pd.Series(weighted_scores, index=user_item.columns)

    top_items = scores[not_rated].sort_values(ascending=False).head(top_n).index.tolist()
    rec = (df[df['ProdID'].isin(top_items)]
           .drop_duplicates(subset=['ProdID'])
           [['ProdID','Name','Brand','ImageURL','Rating','ReviewCount']]
           .head(top_n)
           .copy())
    return rec

def recommend_hybrid(df: pd.DataFrame, X, name_to_index, user_item, item_name: str, target_user_id: int,
                     top_n: int = 10, content_weight: float = 0.5, collab_weight: float = 0.5) -> pd.DataFrame:
    c = recommend_content(df, X, name_to_index, item_name, top_n*2)  # take more then trim
    u = recommend_collab(df, user_item, target_user_id, top_n*2)

    if not c.empty:
        c['c_score'] = c['Score'] / (c['Score'].max() if c['Score'].max() else 1.0)
    else:
        c['c_score'] = []

    if not u.empty:
        u['u_score'] = u['Rating'] / (u['Rating'].max() if u['Rating'].max() else 1.0)
    else:
        u['u_score'] = []

    c_small = c[['Name','Brand','ImageURL','Rating','ReviewCount','c_score']]
    u_small = u[['Name','Brand','ImageURL','Rating','ReviewCount','u_score']]

    merged = pd.merge(c_small, u_small, on=['Name','Brand','ImageURL','Rating','ReviewCount'], how='outer')
    merged['c_score'] = merged['c_score'].fillna(0)
    merged['u_score'] = merged['u_score'].fillna(0)
    merged['FinalScore'] = content_weight*merged['c_score'] + collab_weight*merged['u_score']
    out = merged.sort_values('FinalScore', ascending=False).head(top_n).copy()
    return out

# ========== UI ==========
st.title("🛒 E‑Commerce Product Recommender")
st.caption("Deploy-ready Streamlit app. Supports Trending, Content-Based, Collaborative, and Hybrid recommendations.")

with st.sidebar:
    st.header("Settings")
    data_path = st.text_input("Path to TSV dataset", value="data/walmart_reviews_5k.tsv",
                              help="Place your TSV in the 'data' folder and set the relative path here.")
    use_spacy = st.toggle("Use spaCy for tag cleaning (slower)", value=False)
    algo = st.selectbox("Recommendation type", ["Trending (Rating-based)", "Content-based", "Collaborative", "Hybrid"])
    top_n = st.slider("Top N", 3, 30, 10)

try:
    raw = load_data(data_path)
except Exception as e:
    st.error(f"Could not load dataset from '{data_path}'. Upload a TSV below or fix the path. Error: {e}")
    uploaded = st.file_uploader("Or upload your TSV file", type=["tsv","txt","csv"])
    if uploaded:
        raw = pd.read_csv(uploaded, sep="\t")
    else:
        st.stop()

df = preprocess_data(raw, use_spacy=use_spacy)

vectorizer = X = name_to_index = None
user_item = None

if algo in ("Content-based", "Hybrid"):
    vectorizer, X, name_to_index = build_content_index(df)
if algo in ("Collaborative", "Hybrid"):
    user_item = build_user_item(df)

col_controls = st.columns(3)
with col_controls[0]:
    if algo in ("Content-based", "Hybrid"):
        options = sorted(df['Name'].dropna().unique().tolist())
        default = options[0] if options else ""
        item_name = st.selectbox("Pick a product you like", options, index=0 if options else None)
    else:
        item_name = None

with col_controls[1]:
    if algo in ("Collaborative", "Hybrid"):
        user_ids = sorted(df['ID'].unique().tolist())
        target_user_id = st.selectbox("Pick a user ID", user_ids, index=0 if user_ids else None)
    else:
        target_user_id = None

with col_controls[2]:
    if algo == "Hybrid":
        content_weight = st.slider("Content weight", 0.0, 1.0, 0.6, 0.05)
        collab_weight = 1.0 - content_weight
        st.write(f"Collaborative weight = **{collab_weight:.2f}**")
    else:
        content_weight = collab_weight = None

if algo == "Trending (Rating-based)":
    recs = recommend_trending(df, top_n=top_n)
elif algo == "Content-based":
    if not df.empty and item_name:
        recs = recommend_content(df, X, name_to_index, item_name, top_n=top_n)
    else:
        recs = pd.DataFrame()
elif algo == "Collaborative":
    if not df.empty and target_user_id is not None:
        recs = recommend_collab(df, user_item, int(target_user_id), top_n=top_n)
    else:
        recs = pd.DataFrame()
else:  
    if not df.empty and item_name and target_user_id is not None:
        recs = recommend_hybrid(df, X, name_to_index, user_item, item_name, int(target_user_id),
                                top_n=top_n, content_weight=content_weight, collab_weight=collab_weight)
    else:
        recs = pd.DataFrame()

st.markdown("---")
st.subheader("Recommendations")

if recs is None or recs.empty:
    st.info("No recommendations found. Try a different product/user or check your dataset.")
else:
    for _, row in recs.iterrows():
        with st.container(border=True):
            cols = st.columns([1, 4])
            with cols[0]:
                img = row.get("ImageURL", "")
                if isinstance(img, str) and img.strip():
                    st.image(img, use_container_width=True)
            with cols[1]:
                st.markdown(f"**{row.get('Name','(no name)')}**")
                st.write(f"Brand: {row.get('Brand','-')}  •  ⭐ {row.get('Rating',0)}  •  {int(row.get('ReviewCount',0))} reviews")
                if 'Score' in row:
                    st.write(f"Similarity: {row['Score']:.3f}")
                if 'FinalScore' in row:
                    st.write(f"Hybrid score: {row['FinalScore']:.3f}")

    csv = recs.to_csv(index=False).encode('utf-8')
    st.download_button("Download results as CSV", data=csv, file_name="recommendations.csv", mime="text/csv")
