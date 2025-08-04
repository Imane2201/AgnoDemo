# Thai Cuisine Coordinator - Multi-Expert Team with Azure OpenAI and PgVector

from agno.agent import Agent
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.azure import AzureOpenAI
from agno.team import Team
from agno.vectordb.pgvector import PgVector, SearchType
from agno.embedder.azure_openai import AzureOpenAIEmbedder
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

# Initialize PgVector
vector_db = PgVector(
    table_name="thai_recipes",
    db_url=db_url,
    search_type=SearchType.hybrid,
    embedder=AzureOpenAIEmbedder(
        id=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
    ),
)

# Create knowledge base with Thai recipes PDF
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=vector_db,
)

# Create specialized team members

# 1. Recipe Expert - Master of traditional recipes
recipe_expert = Agent(
    name="Chef Somchai",
    role="Master Thai Chef and Recipe Expert",
    knowledge=knowledge_base,
    model=get_azure_model(),
    add_references=True,
    search_knowledge=False,
    markdown=True,
    instructions="""
    You are Chef Somchai, a master Thai chef with 30 years of experience. 
    You specialize in authentic Thai recipes and traditional cooking methods.
    Always provide detailed, step-by-step instructions and explain the cultural significance of dishes.
    Focus on traditional techniques and authentic ingredients.
    """
)

# 2. Ingredient Specialist - Expert in Thai ingredients and substitutions
ingredient_specialist = Agent(
    name="Dr. Pim",
    role="Thai Ingredient Specialist and Food Scientist",
    knowledge=knowledge_base,
    model=get_azure_model(),
    add_references=True,
    search_knowledge=False,
    markdown=True,
    instructions="""
    You are Dr. Pim, a food scientist specializing in Thai ingredients.
    You know everything about Thai herbs, spices, and ingredients.
    Provide ingredient substitutions, explain flavor profiles, and suggest where to find authentic ingredients.
    Always mention nutritional benefits and traditional uses of ingredients.
    """
)

# 3. Technique Expert - Cooking methods and techniques
technique_expert = Agent(
    name="Khun Lek",
    role="Thai Cooking Technique Specialist",
    knowledge=knowledge_base,
    model=get_azure_model(),
    add_references=True,
    search_knowledge=False,
    markdown=True,
    instructions="""
    You are Khun Lek, a Thai cooking technique specialist.
    You focus on proper cooking methods, timing, heat control, and traditional techniques.
    Explain why certain techniques are used and how they affect the final dish.
    Provide tips for achieving authentic flavors and textures.
    """
)

# 4. Regional Cuisine Expert - Different Thai regions and styles
regional_expert = Agent(
    name="Ajarn Siri",
    role="Thai Regional Cuisine Expert",
    knowledge=knowledge_base,
    model=get_azure_model(),
    add_references=True,
    search_knowledge=False,
    markdown=True,
    instructions="""
    You are Ajarn Siri, an expert in Thai regional cuisines.
    You understand the differences between Northern, Southern, Central, and Northeastern Thai cooking.
    Explain regional variations, local ingredients, and cultural influences on dishes.
    Provide context about the origins and traditions of different recipes.
    """
)

# 5. Dietary Specialist - Health and dietary considerations
dietary_specialist = Agent(
    name="Dr. Nong",
    role="Thai Cuisine Dietary Specialist",
    knowledge=knowledge_base,
    model=get_azure_model(),
    add_references=True,
    search_knowledge=False,
    markdown=True,
    instructions="""
    You are Dr. Nong, a nutritionist specializing in Thai cuisine.
    You provide dietary advice, health benefits, and modifications for different dietary needs.
    Suggest healthier alternatives, portion control, and nutritional information.
    Consider vegetarian, vegan, gluten-free, and low-sodium options.
    """
)

# Create the comprehensive Thai cooking team with coordinate mode
thai_cooking_team = Team(
    name="Thai Cuisine Master Team",
    mode="coordinate",
    members=[
        recipe_expert,
        ingredient_specialist,
        technique_expert,
        regional_expert,
        dietary_specialist
    ],
    model=get_azure_model(),
    knowledge=knowledge_base,
    show_members_responses=True,
    markdown=True,
    add_references=True,
    search_knowledge=False,
    description="A coordinated team of Thai cuisine experts that provides comprehensive cooking guidance",
    instructions=[
        "You are the Team Leader coordinating a team of Thai cuisine experts.",
        "When receiving a query, analyze it and break it down into specific tasks for team members:",
        "- Delegate recipe and traditional cooking questions to Chef Somchai",
        "- Assign ingredient and substitution questions to Dr. Pim", 
        "- Send technique and timing questions to Khun Lek",
        "- Route regional and cultural questions to Ajarn Siri",
        "- Direct dietary and health questions to Dr. Nong",
        "Synthesize all team member outputs into a cohesive, comprehensive response.",
        "Ensure the final response is well-structured and addresses all aspects of the user's query.",
        "Coordinate the team to avoid redundancy while ensuring complete coverage of the topic."
    ],
    success_criteria="Provide comprehensive, well-structured Thai cuisine guidance that combines recipe details, ingredient knowledge, cooking techniques, regional context, and dietary considerations",
    add_datetime_to_instructions=True,
    enable_agentic_context=True,
    share_member_interactions=True
)

# Load the knowledge base
print("Loading knowledge base...")
knowledge_base.load(upsert=True)

# Test the coordinated Thai cuisine team
print("\n" + "="*60)
print("THAI CUISINE COORDINATOR - DEMO")
print("="*60)

# Demo: Complex recipe coordination
print("\n🍜 DEMO: Tom Kha Gai Recipe Coordination")
print("="*50)
thai_cooking_team.print_response(
    "I want to make an authentic Tom Kha Gai soup. Please provide the complete recipe, explain the key ingredients and their substitutes, describe the proper cooking techniques, mention any regional variations, and suggest dietary modifications for someone with high blood pressure."
)

# print("\n🌶️ DEMO: Regional Cuisine Comparison")
# print("="*50)
# thai_cooking_team.print_response(
#     "What are the differences between Northern and Southern Thai curries? Include typical ingredients, cooking methods, flavor profiles, and cultural significance."
# )

# print("\n🥘 DEMO: Ingredient Coordination")
# print("="*50)
# thai_cooking_team.print_response(
#     "I have galangal, lemongrass, and kaffir lime leaves. What can I make with these ingredients? Explain the proper way to prepare each ingredient, the cooking techniques needed, and suggest a complete meal plan."
# )

# print("\n🏥 DEMO: Dietary Consultation")
# print("="*50)
# thai_cooking_team.print_response(
#     "I'm trying to eat healthier but love Thai food. What are some nutritious Thai dishes I can make? Include low-sodium options, vegetarian alternatives, and explain the health benefits of key Thai ingredients."
# )