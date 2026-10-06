import instructions

from typesafe_sdk import  TypeSafeClient

from dotenv import load_dotenv
load_dotenv()

client = TypeSafeClient()

def classify_complaint(complaint: str):
    response = client.system_one(
        state={complaint},
        questions=instructions.CLASSIFICATION_QUESTIONS,
    )
    return response.choices
    