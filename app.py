import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
import plotly.express as px

# Load environment variables
load_dotenv()

# Initialize the Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Define the model
model = "gemini-embedding-001"

def get_embedding(text: str):
    """Generate embedding for a given text and return as numpy array."""
    response = client.models.embed_content(model=model, contents=[text])
    embedding_obj = response.embeddings[0]
    # Convert ContentEmbedding object to numpy array
    return np.array(embedding_obj.values)

# --- Page Config ---
st.set_page_config(page_title="AI Job Matcher", page_icon="🤖", layout="wide")
st.markdown("""
<style>
body {background-color: #1e1e1e; color: #f0f0f0; font-family: 'Arial', sans-serif;}
h1 {color: #00ffff; text-align: center;}
.stButton>button {background-color: #00bfff; color:white;}
.css-1aumxhk {background-color: #2b2b2b; border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>🤖 AI Job Matcher</h1>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("Upload & Settings")
resumes_files = st.sidebar.file_uploader("Upload Multiple Resumes (PDF)", type="pdf", accept_multiple_files=True)
jobs_file = st.sidebar.file_uploader("Upload Job Descriptions CSV", type="csv")
top_n_keywords = st.sidebar.slider("Top Keywords to Highlight", min_value=3, max_value=20, value=10)
score_filter = st.sidebar.slider("Minimum Matching Score (%)", min_value=0, max_value=100, value=0)
show_resume_preview = st.sidebar.checkbox("Show Resume Text Preview", value=False)

# --- Main Matching ---
if st.button("Match Resumes"):
    if not resumes_files:
        st.warning("Please upload at least one resume!")
    elif jobs_file is None:
        st.warning("Please upload a jobs CSV file!")
    else:
        jobs_df = pd.read_csv(jobs_file)
        if "Job Title" not in jobs_df.columns or "Job Description" not in jobs_df.columns:
            st.error("CSV must have columns: Job Title, Job Description")
        else:
            final_results = []
            progress_bar = st.progress(0, text="Processing Resumes")
            total_resumes = len(resumes_files)

            for i, resume_file in enumerate(resumes_files):
                with pdfplumber.open(resume_file) as pdf:
                    resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

                if show_resume_preview:
                    with st.expander(f"Preview Resume: {resume_file.name}"):
                        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

                resume_emb = get_embedding(resume_text)

                job_scores = []
                matched_keywords_list = []

                for job_desc in jobs_df["Job Description"]:
                    job_emb = get_embedding(job_desc)
                    # Compute cosine similarity
                    score = np.dot(resume_emb, job_emb) / (np.linalg.norm(resume_emb) * np.linalg.norm(job_emb))
                    job_scores.append(round(score*100, 2))

                    job_keywords = [w.lower() for w in job_desc.split() if len(w) > 3]
                    resume_words = [w.lower() for w in resume_text.split()]
                    matched_keywords = list(set(job_keywords) & set(resume_words))[:top_n_keywords]
                    matched_keywords_list.append(", ".join(matched_keywords))

                temp_df = jobs_df.copy()
                temp_df["Matching Score (%)"] = job_scores
                temp_df["Resume"] = resume_file.name
                temp_df["Matched Keywords"] = matched_keywords_list
                final_results.append(temp_df)

                progress_bar.progress((i + 1)/total_resumes, text=f"Processing {resume_file.name}")

            final_df = pd.concat(final_results).sort_values(by="Matching Score (%)", ascending=False).reset_index(drop=True)
            final_df = final_df[final_df["Matching Score (%)"] >= score_filter]
            st.success("✅ Matching Completed!")

            # --- Tabs ---
            tab1, tab2, tab3 = st.tabs(["Dashboard", "Resume Preview", "Download Results"])

            # --- Dashboard Tab ---
            with tab1:
                for resume_name in final_df["Resume"].unique():
                    resume_df = final_df[final_df["Resume"] == resume_name].sort_values(by="Matching Score (%)", ascending=False)
                    st.markdown(f"### Resume: {resume_name}")

                    top_job = resume_df.iloc[0]
                    st.markdown(f"""
                    <div style='padding:15px; border-radius:10px; background-color:#2b2b2b; margin-bottom:10px;'>
                    <h4>🏆 Top Job: {top_job['Job Title']}</h4>
                    <p>Matching Score: <b>{top_job['Matching Score (%)']}%</b></p>
                    <p>Matched Keywords: {top_job['Matched Keywords']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    fig = px.bar(
                        resume_df,
                        x="Matching Score (%)",
                        y="Job Title",
                        orientation="h",
                        text="Matching Score (%)",
                        hover_data=["Matched Keywords"],
                        color="Matching Score (%)",
                        color_continuous_scale="darkmint"
                    )
                    fig.update_layout(yaxis={'categoryorder':'total ascending'}, xaxis_range=[0,100], height=400, paper_bgcolor="#1e1e1e", plot_bgcolor="#1e1e1e", font_color="#f0f0f0")
                    st.plotly_chart(fig)

            # --- Resume Preview Tab ---
            with tab2:
                for resume_file in resumes_files:
                    with pdfplumber.open(resume_file) as pdf:
                        resume_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                    with st.expander(f"Preview Resume: {resume_file.name}", expanded=False):
                        st.text(resume_text[:3000] + ("..." if len(resume_text) > 3000 else ""))

            # --- Download Tab ---
            with tab3:
                excel_file = "resume_job_matches.xlsx"
                # Export Excel using openpyxl explicitly
                final_df.to_excel(excel_file, index=False, engine='openpyxl')

                # Use context manager to read the file for download
                with open(excel_file, "rb") as f:
                    st.download_button("📥 Download Results as Excel", data=f.read(), file_name=excel_file)
