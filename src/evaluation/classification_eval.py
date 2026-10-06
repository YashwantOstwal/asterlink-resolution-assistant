import json
import instructions

from typesafe_sdk import  TypeSafeClient

from dotenv import load_dotenv
load_dotenv()

client = TypeSafeClient()

with open(
    "docs/development_tickets.json",
    "r",
    encoding="utf-8",
) as file:
    development_tickets = json.load(file)["new_tickets"] 
    
    cnt = 0
    for development_ticket in development_tickets:
        response = client.system_one(
            state={development_ticket["complaint"]},
            questions=instructions.CLASSIFICATION_QUESTIONS
        )
        cnt+= response.choices["product"].choice == development_ticket["product"]
        cnt+= response.choices["severity"].choice  == development_ticket["severity"]
        cnt+= response.choices["category"].choice == development_ticket["category"]
        cnt+= response.choices["customer_sentiment"].choice == development_ticket["customer_sentiment"]
    print(cnt / (len(development_tickets) * 4))


with open(
    "docs/heldout_test_tickets.json",
    "r",
    encoding="utf-8",
) as file:
    development_tickets = json.load(file)["new_tickets"] 
    
    cnt = 0
    for development_ticket in development_tickets:
        response = client.system_one(
            state={development_ticket["complaint"]},
            questions=instructions.CLASSIFICATION_QUESTIONS
        )
        cnt+= response.choices["product"].choice == development_ticket["product"]
        cnt+= response.choices["severity"].choice  == development_ticket["severity"]
        cnt+= response.choices["category"].choice == development_ticket["category"]
        cnt+= response.choices["customer_sentiment"].choice == development_ticket["customer_sentiment"]
    print(cnt / (len(development_tickets) * 4))


