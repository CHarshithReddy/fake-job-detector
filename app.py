import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import scipy.sparse as sp
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Fake Job Detector", layout="wide")

@st.cache_resource
def load_models():
    with open('src/model_lr.pkl','rb') as f:
        lr = pickle.load(f)
    with open('src/model_rf.pkl','rb') as f:
        rf = pickle.load(f)
    with open('src/tfidf.pkl','rb') as f:
        tfidf = pickle.load(f)
    with open('src/numeric_features.pkl','rb') as f:
        numeric_features = pickle.load(f)
    return lr, rf, tfidf, numeric_features

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
lr, rf, tfidf, NUMERIC_FEATURES = load_models()

# Sidebar
st.sidebar.title("Fake Job Detector")
st.sidebar.markdown("Detects fraudulent job postings using NLP + Machine Learning")
page = st.sidebar.radio("Navigate", ["Detector", "Analytics"])

if page == "Detector":
    st.title("Fake Job Posting Detector")
    st.markdown("Paste a job description below to check if it's real or fraudulent")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input("Job title",
                              placeholder="e.g. Software Engineer")
    with col2:
        employment_type = st.selectbox("Employment type",
            ["Full-time","Part-time","Contract","Temporary","Other"])

    company_profile = st.text_area("Company profile", height=100,
        placeholder="Describe the company...")
    description = st.text_area("Job description", height=200,
        placeholder="Paste the full job description here...")
    requirements = st.text_area("Requirements", height=100,
        placeholder="Required skills and experience...")

    if st.button("Analyse posting"):
        if description:
            combined = f"{title} {company_profile} {description} {requirements}"
            clean = re.sub(r'[^a-z\s]','',
                    ' '.join([w for w in combined.lower().split()
                               if w not in stop_words]))
            text_vec = tfidf.transform([clean])

            numeric_vals = [[
                len(description),
                len(description.split()),
                1 if company_profile else 0,
                1 if requirements else 0,
                0,
                0,
                sum(1 for w in ['urgent','guaranteed','no experience',
                    'work from home','earn money','weekly pay',
                    'unlimited income','be your own boss']
                    if w in combined.lower()),
                1 if employment_type == 'Part-time' else 0
            ]]

            X = sp.hstack([text_vec, sp.csr_matrix(numeric_vals)])

            lr_prob = lr.predict_proba(X)[0][1]
            rf_prob = rf.predict_proba(X)[0][1]
            avg_prob = (lr_prob + rf_prob) / 2
            pred = "FAKE" if avg_prob > 0.3 else "REAL"

            st.divider()
            c1, c2, c3 = st.columns(3)
            if pred == "FAKE":
                c1.error(f"Prediction: FAKE JOB")
            else:
                c1.success(f"Prediction: REAL JOB")
            c2.metric("Fraud probability", f"{avg_prob*100:.1f}%")
            c3.metric("Confidence",
                      "High" if abs(avg_prob-0.5)>0.3 else "Medium")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_prob*100,
                title={'text': "Fraud probability (%)"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': "crimson" if pred=="FAKE"
                            else "steelblue"},
                    'steps': [
                        {'range':[0,30],'color':'#EAF3DE'},
                        {'range':[30,60],'color':'#FAEEDA'},
                        {'range':[60,100],'color':'#FCEBEB'}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

            suspicious_found = [w for w in
                ['urgent','guaranteed','no experience',
                 'work from home','earn money','weekly pay',
                 'unlimited income','be your own boss']
                if w in combined.lower()]
            if suspicious_found:
                st.warning(f"Suspicious phrases found: "
                          f"{', '.join(suspicious_found)}")
        else:
            st.warning("Please enter a job description!")

else:
    st.title("Analytics Dashboard")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total postings", "17,880")
    c2.metric("Fake postings", "866 (4.84%)")
    c3.metric("Model ROC-AUC", "0.989")

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.bar(
            x=['Real','Fake'], y=[17014, 866],
            color=['Real','Fake'],
            color_discrete_map={'Real':'steelblue','Fake':'crimson'},
            title='Real vs Fake job postings'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        emp_fraud = {
            'Employment type': ['Part-time','Other','Full-time',
                                'Contract','Temporary'],
            'Fraud rate': [0.093, 0.066, 0.042, 0.029, 0.009]
        }
        import pandas as pd
        emp_data = pd.DataFrame(emp_fraud)
        fig2 = px.bar(emp_data, x='Employment type', y='Fraud rate',
                      title='Fraud rate by employment type',
                      color='Fraud rate',
                      color_continuous_scale=['steelblue','crimson'])
        st.plotly_chart(fig2, use_container_width=True)