
import streamlit as st
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


st.set_page_config(
    page_title="AI Customer Support Ticket Triage",
    page_icon="🎫",
    layout="wide"
)

st.title("AI Customer Support Ticket Triage")
st.write(
    "Classify customer support tickets, identify urgency, "
    "and generate suggested responses using machine learning."
)


# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("tickets.csv")


# Train models
@st.cache_resource
def train_models(data):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(data["clean_ticket"])

    category_model = LogisticRegression(max_iter=1000)
    category_model.fit(X, data["category"])

    urgency_model = LogisticRegression(max_iter=1000)
    urgency_model.fit(X, data["urgency"])

    return vectorizer, category_model, urgency_model


try:
    df = load_data()

    vectorizer, category_model, urgency_model = train_models(df)

except Exception as e:
    st.error("Could not load the dataset or train the models.")
    st.code(str(e))
    st.stop()


priority_map = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}

responses = {
    "billing": (
        "We are checking your billing issue "
        "and will assist you shortly."
    ),
    "technical": (
        "Our technical team is looking into the issue. "
        "Please try again shortly."
    ),
    "account": (
        "We will help you secure and restore access "
        "to your account."
    ),
    "product": (
        "We are reviewing your product-related concern "
        "and will assist you."
    )
}


# Dashboard
st.subheader("Ticket Dashboard")

col1, col2, col3 = st.columns(3)

col1.metric("Total Training Tickets", len(df))

high_urgency = len(
    df[df["urgency"].isin(["high", "critical"])]
)

col2.metric("High/Critical Training Tickets", high_urgency)

col3.metric(
    "Ticket Categories",
    df["category"].nunique()
)

with st.expander("View category distribution"):
    st.bar_chart(df["category"].value_counts())


# Ticket submission
st.subheader("Analyze a Customer Ticket")

ticket_text = st.text_area(
    "Enter the customer's issue",
    placeholder="Example: I was charged twice for my order.",
    height=150
)

if st.button("Analyze Ticket", type="primary"):

    if not ticket_text.strip():
        st.warning("Please enter a ticket before analyzing.")

    else:
        ticket_vector = vectorizer.transform([ticket_text])

        predicted_category = category_model.predict(
            ticket_vector
        )[0]

        predicted_urgency = urgency_model.predict(
            ticket_vector
        )[0]

        category_confidence = max(
            category_model.predict_proba(ticket_vector)[0]
        )

        urgency_confidence = max(
            urgency_model.predict_proba(ticket_vector)[0]
        )

        confidence = min(
            category_confidence,
            urgency_confidence
        )

        priority = priority_map.get(predicted_urgency, 1)

        suggested_response = responses.get(
            predicted_category,
            "Your issue has been received. "
            "Our support team will assist you."
        )

        st.divider()
        st.subheader("Analysis Results")

        result_col1, result_col2 = st.columns(2)

        result_col1.metric(
            "Predicted Category",
            str(predicted_category).title()
        )

        result_col2.metric(
            "Predicted Urgency",
            str(predicted_urgency).title()
        )

        st.write(f"**Priority level:** {priority} / 4")
        st.write(f"**Prediction confidence:** {confidence * 100:.2f}%")

        if confidence < 0.60:
            st.warning(
                "Low confidence — human review required."
            )

            review_data = pd.DataFrame([{
                "ticket": ticket_text,
                "predicted_category": predicted_category,
                "predicted_urgency": predicted_urgency,
                "confidence": round(confidence, 2)
            }])

            review_file = "review_log.csv"
            review_data.to_csv(
                review_file,
                mode="a",
                header=not os.path.exists(review_file),
                index=False
            )

        else:
            st.success(
                "Prediction confidence is acceptable."
            )

        st.subheader("Suggested Customer Response")
        st.info(suggested_response)
