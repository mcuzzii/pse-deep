from cohere import ClientV2
import os
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

co = ClientV2(os.getenv("COHERE_API_KEY"))
response = co.models.list()
pprint(response)