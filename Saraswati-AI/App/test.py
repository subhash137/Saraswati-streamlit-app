#AIzaSyDnX_IIMWOYdcNIYedTP346PYMjK7dqlp0

from langchain_google_vertexai import VertexAI
import vertexai
from langchain_google_vertexai import ChatVertexAI

vertexai.init(project="nlp1-427616", location="us-central1")

llm = VertexAI(model_name="gemini-pro")
output = llm.invoke(["How to make a pizza?"])
print(output)