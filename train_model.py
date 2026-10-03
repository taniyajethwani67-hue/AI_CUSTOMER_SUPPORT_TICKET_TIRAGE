import pandas as pd
import re


 

data = {
    "ticket": [
        "I was charged twice for my subscription",
        "My payment failed while buying the product",
        "I need a refund for my order",
        "There is an incorrect charge on my bill",
        "My credit card payment is not working",

        "The application keeps crashing",
        "The website is showing an error",
        "My app is not loading",
        "The website stopped working",
        "I cannot open the application",

        "I cannot login to my account",
        "I forgot my password",
        "Please help me reset my password",
        "My account is locked",
        "I cannot access my profile",

        "The product I received is damaged",
        "I received the wrong product",
        "The product is missing from my order",
        "I want information about this product",
        "The product quality is poor",

        # Extra examples for our own ideas
        "Someone accessed my account and I cannot login",
        "My account has been hacked",
        "I was charged an extra amount",
        "The application is completely down"
    ],

    "category": [
        "billing", "billing", "billing", "billing", "billing",
        "technical", "technical", "technical", "technical", "technical",
        "account", "account", "account", "account", "account",
        "product", "product", "product", "product", "product",
        "account", "account", "billing", "technical"
    ],

    "urgency": [
        "high", "high", "medium", "high", "high",
        "high", "high", "medium", "high", "medium",
        "high", "medium", "medium", "high", "medium",
        "high", "high", "medium", "low", "medium",
        "critical", "critical", "medium", "critical"
    ]
}



df = pd.DataFrame(data)



df.to_csv("tickets.csv", index=False)


print("====================================")
print("AI CUSTOMER SUPPORT TICKET TRIAGE")
print("====================================")

print("\nHistorical ticket data created successfully!")

print("Total tickets:", len(df))

print("\nCategories:")
print(df["category"].value_counts())

print("\nUrgency levels:")
print(df["urgency"].value_counts())



def clean_text(text):


    text = text.lower()


    text = re.sub(r"[^a-zA-Z\s]", "", text)


    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_ticket"] = df["ticket"].apply(clean_text)


print("\n====================================")
print("TEXT CLEANING COMPLETED")
print("====================================")

print(df[["ticket", "clean_ticket"]].head())




print("\n")
print("CUSTOM URGENCY SYSTEM")
print("")

print("Urgency levels used:")
print("Low")
print("Medium")
print("High")
print("Critical")



df["ticket_id"] = [
    "TICKET-" + str(i).zfill(3)
    for i in range(1, len(df) + 1)
]


print("\nTicket IDs created successfully.")

print(df[["ticket_id", "ticket", "category", "urgency"]].head())



df.to_csv("tickets.csv", index=False)



print("\nDataset saved as tickets.csv")


from sklearn.feature_extraction.text import TfidfVectorizer



vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(
    df["clean_ticket"]
)


print("\n====================================")
print("STEP 3: TF-IDF FEATURES")
print("====================================")

print("TF-IDF conversion completed!")

print("Number of tickets:", X.shape[0])

print("Number of text features:", X.shape[1])




priority_map = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4
}


df["priority"] = df["urgency"].map(priority_map)


print("\n====================================")
print("CUSTOM PRIORITY SYSTEM")
print("====================================")

print(df[[
    "ticket_id",
    "urgency",
    "priority"
]].head())



df.to_csv(
    "tickets.csv",
    index=False
)

print("\nUpdated tickets.csv saved successfully!")


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.25,
    random_state=42,
    stratify=df["category"]
)

X_train = X[train_idx]
X_test = X[test_idx]




y_category_train = df.loc[
    train_idx, "category"
]

y_category_test = df.loc[
    test_idx, "category"
]


y_urgency_train = df.loc[
    train_idx, "urgency"
]

y_urgency_test = df.loc[
    test_idx, "urgency"
]




category_model = LogisticRegression(
    max_iter=1000
)

category_model.fit(
    X_train,
    y_category_train
)

print("\n====================================")
print("CATEGORY MODEL TRAINED")
print("====================================")

print("The model can predict:")
print("Billing")
print("Technical")
print("Account")
print("Product")




urgency_model = LogisticRegression(
    max_iter=1000
)

urgency_model.fit(
    X_train,
    y_urgency_train
)

print("\n====================================")
print("URGENCY MODEL TRAINED")
print("====================================")

print("The model can predict:")
print("Low")
print("Medium")
print("High")
print("Critical")


category_predictions = category_model.predict(
    X_test
)

urgency_predictions = urgency_model.predict(
    X_test
)


print("\n====================================")
print("MODEL PREDICTIONS")
print("====================================")

print("Category predictions:")
print(category_predictions)

print("\nUrgency predictions:")
print(urgency_predictions)


print("\n=================================")


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt


print("\n====================================")
print("CATEGORY MODEL EVALUATION")
print("====================================")

print("\nClassification Report - Category:")

print(
    classification_report(
        y_category_test,
        category_predictions,
        zero_division=0
    )
)




category_cm = confusion_matrix(
    y_category_test,
    category_predictions,
    labels = category_model.classes_

)

category_display = ConfusionMatrixDisplay(
    confusion_matrix=category_cm,
    display_labels=category_model.classes_
)

category_display.plot()
plt.title("Category Confusion Matrix")

plt.savefig(
    "static/category_confusion_matrix.png",
    bbox_inches="tight"
)

plt.show()
plt.close()




print("\n====================================")
print("URGENCY MODEL EVALUATION")
print("====================================")

print("\nClassification Report - Urgency:")

print(
    classification_report(
        y_urgency_test,
        urgency_predictions,
        zero_division=0
    )
)




urgency_cm = confusion_matrix(
    y_urgency_test,
    urgency_predictions,
    labels=urgency_model.classes_
)

urgency_display = ConfusionMatrixDisplay(
    confusion_matrix=urgency_cm,
    display_labels=urgency_model.classes_
)

urgency_display.plot()
plt.title("Urgency Confusion Matrix")


plt.show()













 



