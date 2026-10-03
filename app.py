from  flask  import Flask, render_template, request
import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

df = pd.read_csv("tickets.csv")

counter_file = "ticket_counter.txt"

if os.path.exists(counter_file):
    with open(counter_file, "r") as f:
        next_ticket_id = int(f.read())
else:
    next_ticket_id = len(df) + 1

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    df["clean_ticket"]
)

category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X,
    df["category"]
)

urgency_model = LogisticRegression(
    max_iter=1000
)

urgency_model.fit(
    X,
    df["urgency"]
)

priority_map = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}

responses = {
    "billing":
    "We are checking your billing issue and will assist you shortly.",

    "technical":
    "Our technical team is looking into the issue. Please try again shortly.",

    "account":
    "We will help you secure and restore access to your account.",

    "product":
    "We are reviewing your product-related concern and will assist you."
}


@app.route("/", methods=["GET", "POST"])
def home():

    result = None

    if request.method == "POST":

        ticket_text = request.form["ticket"]

        ticket_vector = vectorizer.transform(
            [ticket_text]
        )

        predicted_category = category_model.predict(
            ticket_vector
        )[0]

        predicted_urgency = urgency_model.predict(
            ticket_vector
        )[0]

        category_confidence = max(
            category_model.predict_proba(
                ticket_vector
            )[0]
        )

        urgency_confidence = max(
            urgency_model.predict_proba(
                ticket_vector
            )[0]
        )

        confidence = min(
            category_confidence,
            urgency_confidence
        )

        priority = priority_map[
            predicted_urgency
        ]

        global next_ticket_id

ticket_id = (
    "TICKET-" +
    str(next_ticket_id).zfill(3)
)

next_ticket_id += 1

with open(counter_file, "w") as f:
    f.write(str(next_ticket_id))

suggested_response = responses[
            predicted_category
        ]


 if confidence < 0.60:

            review_data = pd.DataFrame([{

                "ticket_id": ticket_id,

                "ticket": ticket_text,

                "predicted_category":
                    predicted_category,

                "predicted_urgency":
                    predicted_urgency,

                "confidence":
                    round(confidence, 2)

            }])
      

            review_data.to_csv(
                "review_log.csv",
                mode="a",
                header=not os.path.exists(
                    "review_log.csv"
                ),
                index=False
            )

            review_message = (
                "Low confidence - "
                "Human review required"
            )

        else:

            review_message = (
                "Prediction confidence is acceptable"
            )

        result = {

            "ticket_id": ticket_id,

            "category":
                predicted_category,

            "urgency":
                predicted_urgency,

            "priority":
                priority,

            "confidence":
                round(confidence * 100, 2),

            "response":
                suggested_response,

            "review":
                review_message
        }

    total_tickets = len(df)

    high_urgency = len(
        df[
            df["urgency"].isin(
                ["high", "critical"]
            )
        ]
    )

    categories = (
        df["category"]
        .value_counts()
        .to_dict()
    )

    return render_template(
        "index.html",
        result=result,
        total_tickets=total_tickets,
        high_urgency=high_urgency,
        categories=categories
    )


if __name__ == "__main__":
    app.run(debug=True)
