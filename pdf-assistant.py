import typer
from typing import List, Optional
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.storage.agent.postgres import PgAgentStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2, PgVector
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
from phi.llm.groq import Groq 

import os
from dotenv import load_dotenv
load_dotenv()


#First run the command below in the command prompt, 
# that will update the models vector dimension to 768:

#docker exec -it pgvector psql -U ai -d ai -c "ALTER TABLE ai.recipes ALTER COLUMN embedding TYPE vector(768);"
embedder = SentenceTransformerEmbedder(batch_size=100, model="sentence-transformers/all-mpnet-base-v2", dimensions=768)

# Set up the database URL
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


# Create a knowledge base from pdf
knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#   vector_db = PgVector2(collection="recipes", db_url=db_url, embedder=HuggingfaceCustomEmbedder(model='sentence-transformers/all-MiniLM-L6-v2'))
    vector_db = PgVector2(collection="recipes", db_url=db_url, embedder=embedder)
)

# Load the knowledge base
knowledge_base.load()

# Create a storage
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    if not new:
       existing_run_ids: List[str] = storage.get_all_run_ids(user)
       if len(existing_run_ids) > 0:
           run_id = existing_run_ids[0]

    assistant = Assistant(
        llm= Groq(model="llama-3.3-70b-versatile",name="Groq",embedder = embedder),
        run_id=run_id,
        knowledge_base=knowledge_base,
        storage=storage,
        user_id=user,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to use the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,  
    )

    if run_id is None:
        run_id = assistant.run_id
        print(f"Assistant run_id: {run_id}")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

    if __name__ == "__main__":
        typer.run(pdf_assistant)

# "Assistance" and "PgVector2" still 
# using the "OpenAI" as a default LLM models for embedding and search