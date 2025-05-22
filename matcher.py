from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_similarity(resume_text, job_text):
    embeddings = model.encode([resume_text, job_text], convert_to_tensor=True)
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    return round(similarity.item() * 100, 2)  # percentage similarity
