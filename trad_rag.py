# Traditional RAG with PgVector using Azure OpenAI

from agno.agent import Agent
from agno.embedder.azure_openai import AzureOpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.azure import AzureOpenAI
from agno.vectordb.pgvector import PgVector, SearchType
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure Azure OpenAI
def get_azure_model():
    return AzureOpenAI(
        id=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    )

# Database configuration
db_url = "postgresql+psycopg2://ai:ai@localhost:5532/ai"

# Create a knowledge base of PDFs from URLs
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
    vector_db=PgVector(
        table_name="recipes",
        db_url=db_url,
        search_type=SearchType.hybrid,
        embedder=AzureOpenAIEmbedder(
            id=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        ),
    ),
)

# Load the knowledge base: Comment after first run as the knowledge base is already loaded
knowledge_base.load(upsert=True)

agent = Agent(
    model=get_azure_model(),
    knowledge=knowledge_base,
    # Enable RAG by adding context from the `knowledge` to the user prompt.
    add_references=True,
    # Set as False because Agents default to `search_knowledge=True`
    search_knowledge=False,
    markdown=True,
)

agent.print_response(
    "How do I make chicken and galangal in coconut milk soup", stream=True
)
