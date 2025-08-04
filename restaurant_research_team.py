# Restaurant Research & Planning Team - Multi-Agent with Various Toolkits

from agno.agent import Agent
from agno.team import Team
from agno.models.azure import AzureOpenAI
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.website import WebsiteTools
from agno.tools.pandas import PandasTools
from agno.tools.csv_toolkit import CsvTools
from agno.tools.file import FileTools
from agno.tools.calculator import CalculatorTools
from agno.tools.email import EmailTools
from agno.tools.yfinance import YFinanceTools
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

# Create specialized team members with different toolkits

# 1. Market Research Agent - Web search and data analysis
market_researcher = Agent(
    name="Market Analyst Sarah",
    role="Market Research and Competitive Analysis Specialist",
    model=get_azure_model(),
    tools=[DuckDuckGoTools(), PandasTools(), CsvTools()],
    instructions=[
        "You are Sarah, a market research analyst with expertise in restaurant industry analysis.",
        "Use DuckDuckGo to search for current restaurant trends, market data, and industry reports.",
        "Analyze data using pandas to identify patterns and insights.",
        "Create CSV reports with your findings for the team.",
        "Focus on market size, growth trends, and competitive landscape.",
        "Provide data-driven insights about restaurant opportunities and risks."
    ],
    add_datetime_to_instructions=True
)

# 2. Location Scout Agent - Location analysis (without Google Maps)
location_scout = Agent(
    name="Location Expert Mike",
    role="Real Estate and Location Analysis Specialist",
    model=get_azure_model(),
    tools=[CalculatorTools()],
    instructions=[
        "You are Mike, a location scout specializing in restaurant real estate.",
        "Analyze potential locations based on general knowledge and market data.",
        "Calculate distances, demographics, and accessibility metrics using available data.",
        "Evaluate parking availability, public transport access, and visibility factors.",
        "Assess competition density in target areas using web research.",
        "Provide location recommendations with detailed analysis based on available information."
    ],
    add_datetime_to_instructions=True
)

# 3. Financial Analyst Agent - Financial data and calculations
financial_analyst = Agent(
    name="Financial Analyst Lisa",
    role="Financial Planning and Investment Analysis Specialist",
    model=get_azure_model(),
    tools=[YFinanceTools(), CalculatorTools(), PandasTools()],
    instructions=[
        "You are Lisa, a financial analyst specializing in restaurant investments.",
        "Use Yahoo Finance to research publicly traded restaurant companies and market trends.",
        "Perform financial calculations for startup costs, ROI projections, and break-even analysis.",
        "Analyze financial data using pandas for insights.",
        "Create financial models and projections.",
        "Provide investment recommendations and risk assessments."
    ],
    add_datetime_to_instructions=True
)

# 4. Web Scraper Agent - Website analysis and content research
web_scraper = Agent(
    name="Web Research Agent Alex",
    role="Web Scraping and Content Analysis Specialist",
    model=get_azure_model(),
    tools=[WebsiteTools(), DuckDuckGoTools()],
    instructions=[
        "You are Alex, a web research specialist focused on restaurant industry content.",
        "Scrape restaurant websites to analyze menus, pricing, and business models.",
        "Research competitor websites for insights on their offerings and strategies.",
        "Search for restaurant reviews, ratings, and customer feedback.",
        "Analyze social media presence and online reputation of competitors.",
        "Provide detailed competitive intelligence reports."
    ],
    add_datetime_to_instructions=True
)

# 5. Report Generator Agent - File management and documentation
report_generator = Agent(
    name="Report Specialist David",
    role="Documentation and Report Generation Specialist",
    model=get_azure_model(),
    tools=[FileTools(), CsvTools()],
    instructions=[
        "You are David, a report generation specialist who creates comprehensive business documents.",
        "Use FileTools to read and write various document formats.",
        "Create CSV files with structured data from team findings.",
        "Compile team reports into comprehensive business plans.",
        "Organize and format data for presentation to stakeholders.",
        "Ensure all documentation is professional and well-structured."
    ],
    add_datetime_to_instructions=True
)

# 6. Communication Agent - Email and stakeholder communication
communication_agent = Agent(
    name="Communication Manager Emma",
    role="Stakeholder Communication and Email Management Specialist",
    model=get_azure_model(),
    tools=[EmailTools()],
    instructions=[
        "You are Emma, a communication manager responsible for stakeholder outreach.",
        "Use EmailTools to send professional communications to stakeholders.",
        "Draft and send project updates, findings summaries, and meeting invitations.",
        "Coordinate communication between team members and external parties.",
        "Ensure all communications are clear, professional, and timely.",
        "Maintain communication logs and follow-up schedules."
    ],
    add_datetime_to_instructions=True
)

# Create the comprehensive restaurant research team with collaborate mode
restaurant_team = Team(
    name="Restaurant Research & Planning Team",
    mode="collaborate",
    members=[
        market_researcher,
        location_scout,
        financial_analyst,
        web_scraper,
        report_generator,
        communication_agent
    ],
    model=get_azure_model(),
    show_members_responses=True,
    markdown=True,
    description="A collaborative team of specialists for comprehensive restaurant research and planning",
    instructions=[
        "You are the Discussion Master coordinating a restaurant research and planning team.",
        "All team members will respond to the query concurrently, providing their specialized perspectives:",
        "- Market Analyst Sarah: Market trends, industry analysis, data insights",
        "- Location Expert Mike: Real estate analysis, demographics, accessibility",
        "- Financial Analyst Lisa: Financial projections, investment analysis, ROI",
        "- Web Research Agent Alex: Competitive intelligence, online presence",
        "- Report Specialist David: Document creation, data organization",
        "- Communication Manager Emma: Stakeholder communication, project coordination",
        "Review all team member responses and synthesize them into a comprehensive consensus.",
        "Stop the discussion when the team has reached a consensus on the topic.",
        "Ensure the final response combines all perspectives into actionable recommendations."
    ],
    success_criteria="The team has reached a consensus on restaurant research and planning recommendations",
    add_datetime_to_instructions=True,
    enable_agentic_context=True,
    show_tool_calls=True
)


# Demo scenarios
print("\n" + "="*60)
print("RESTAURANT RESEARCH & PLANNING TEAM - DEMO") 
print("="*60)

# Load any required data
print("Initializing team and tools...")

# Demo: Comprehensive restaurant feasibility study
print("\n📊 DEMO: Restaurant Feasibility Study")
print("="*50)
restaurant_team.print_response(
    "What are the key factors to consider when opening a Thai restaurant in Seattle? Focus on market analysis, location considerations, and financial planning."
)

# # Demo: Location comparison analysis  
# print("\n📍 DEMO: Location Comparison Analysis")
# print("="*50)
# restaurant_team.print_response(
#     "Compare three potential locations for a new restaurant: Capitol Hill, Belltown, and Fremont in Seattle. Analyze foot traffic, demographics, competition, and provide location recommendations with supporting data."
# )

# # Demo: Financial planning analysis
# print("\n💰 DEMO: Financial Planning & Investment Analysis") 
# print("="*50)
# restaurant_team.print_response(
#     "Create a detailed financial plan for a new restaurant concept. Include startup costs, revenue projections, break-even analysis, and research similar restaurant chains' financial performance. Provide investment recommendations."
# )

# # Demo: Competitive intelligence report
# print("\n🔍 DEMO: Competitive Intelligence Report")
# print("="*50)
# restaurant_team.print_response(
#     "Research the competitive landscape for Thai restaurants in Seattle. Analyze competitor menus, pricing, online presence, customer reviews, and identify market opportunities and competitive advantages."
# )

