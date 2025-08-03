import os
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.azure import AzureOpenAI
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.exa import ExaTools
from agno.utils import pprint
import json

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

# Input Pydantic Model
class EventSearchRequest(BaseModel):
    """Structured event search request with specific requirements"""
    
    location: str = Field(description="City or location to search for events")
    event_type: str = Field(description="Type of events to search for (e.g., 'tech', 'social', 'professional', 'parties')")
    date_range: str = Field(description="When to search for events (e.g., 'this weekend', 'next month', 'upcoming')")
    max_events: int = Field(description="Number of events to find", default=1)
    platform_preference: Optional[str] = Field(description="Preferred platform (e.g., 'Eventbrite', 'Meetup', 'LinkedIn', 'Facebook')", default=None)
    category: Optional[str] = Field(description="Specific category or industry focus", default=None)

# Structured Output Models
class EventDetails(BaseModel):
    title: str
    description: str
    date: str
    location: str
    organizer: str
    category: str
    price: Optional[str] = None
    url: Optional[str] = None
    platform: str

class EventSearchAnalysis(BaseModel):
    platform: str
    events_found: int
    events: List[EventDetails]
    platform_summary: str


# Function to convert Pydantic model to string message for team compatibility
def convert_request_to_message(request: EventSearchRequest) -> str:
    """
    Convert EventSearchRequest Pydantic model to a string message for the team.
    This is needed because the team routing mechanism doesn't handle Pydantic models directly.
    """
    platform_pref = request.platform_preference or "none"
    category = request.category or "none"
    
    return f"Find {request.max_events} {request.event_type} events in {request.location} for {request.date_range}. Platform preference: {platform_pref}. Category: {category}."


# Event Platform Agents
eventbrite_agent = Agent(
    name="Eventbrite Agent",
    role="Eventbrite Event Scraper",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the Eventbrite Event Scraper Agent, specialized in finding and extracting event information from Eventbrite.",
        "Key Responsibilities:",
        "1. Search for events on Eventbrite based on user criteria (location, date, category, etc.)",
        "2. Extract the specified number of events from search results (max_events parameter)",
        "3. Extract detailed information for each event including:",
        "   - Event title and description",
        "   - Date and time",
        "   - Location and venue",
        "   - Ticket prices and availability",
        "   - Organizer information",
        "   - Event category and tags",
        "   - Event URL",
        "4. Provide a summary of the events found",
        "",
        "Search Strategy Guidelines:",
        "1. Use multiple search approaches with simplified terms:",
        "   - Search: 'Eventbrite [location] [event type] [date]'",
        "   - Search: 'Events [location] this weekend'",
        "   - Search: '[event type] [location] tickets'",
        "   - Search: 'Eventbrite [location] upcoming events'",
        "2. Use ExaTools to find recent Eventbrite listings and event pages",
        "3. Look for event aggregator websites that list Eventbrite events",
        "4. Search for venue websites that link to Eventbrite tickets",
        "",
        "Max Events Parameter Guidelines:",
        "1. Use the max_events value from the structured input",
        "2. Extract exactly that many events",
        "3. Prioritize events that are:",
        "   - Closest to the requested date",
        "   - Best match for the requested location",
        "   - Most relevant to the requested category/type",
        "   - Have good reviews or high attendance",
        "4. Return the specified number of events with complete details",
        "",
        "Time Criteria Guidelines:",
        "1. ALWAYS prioritize recent and upcoming events:",
        "   - For 'this weekend': Look for events in the next 3-7 days",
        "   - For 'next month': Look for events in the next 30-60 days",
        "   - For 'upcoming': Look for events in the next 2-4 weeks",
        "   - For 'next week': Look for events in the next 7-14 days",
        "2. AVOID events that are:",
        "   - More than 2 months old",
        "   - Past their date",
        "   - Not clearly dated",
        "3. When searching, use specific date ranges:",
        "   - Add '2025' or current year to searches",
        "   - Use 'upcoming' or 'next' in search terms",
        "   - Include 'recent' or 'latest' when appropriate",
        "",
        "IMPORTANT: Use multiple search strategies and simplify search terms to find more results.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

meetup_agent = Agent(
    name="Meetup Agent",
    role="Meetup Event Scraper",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the Meetup Event Scraper Agent, specialized in finding and extracting event information from Meetup.com.",
        "Key Responsibilities:",
        "1. Search for meetups and events on Meetup.com based on user criteria",
        "2. Extract the specified number of events from search results (max_events parameter)",
        "3. Extract detailed meetup information for each event including:",
        "   - Meetup title and description",
        "   - Date and time",
        "   - Location (online or in-person)",
        "   - Group information and member count",
        "   - Event category and tags",
        "   - RSVP information",
        "   - Meetup URL",
        "4. Provide a summary of the events found",
        "",
        "Search Strategy Guidelines:",
        "1. Use multiple search approaches with simplified terms:",
        "   - Search: 'Meetup [location] [topic] [date]'",
        "   - Search: 'Meetup groups [location] [topic]'",
        "   - Search: '[topic] meetup [location]'",
        "   - Search: 'Meetup [location] this weekend'",
        "   - Search: 'Meetup [location] upcoming events'",
        "2. Use ExaTools to find recent Meetup event pages and group listings",
        "3. Look for local event calendars that list Meetup events",
        "4. Search for community websites that promote local meetups",
        "",
        "Max Events Parameter Guidelines:",
        "1. Use the max_events value from the structured input",
        "2. Extract exactly that many events",
        "3. Prioritize meetups that are:",
        "   - Closest to the requested date",
        "   - Best match for the requested location",
        "   - Most relevant to the requested topic/interest",
        "   - Have active groups with good member counts",
        "   - Free or low-cost events",
        "4. Return the specified number of events with complete details",
        "",
        "Time Criteria Guidelines:",
        "1. ALWAYS prioritize recent and upcoming events:",
        "   - For 'this weekend': Look for events in the next 3-7 days",
        "   - For 'next month': Look for events in the next 30-60 days",
        "   - For 'upcoming': Look for events in the next 2-4 weeks",
        "   - For 'next week': Look for events in the next 7-14 days",
        "2. AVOID events that are:",
        "   - More than 2 months old",
        "   - Past their date",
        "   - Not clearly dated",
        "3. When searching, use specific date ranges:",
        "   - Add '2025' or current year to searches",
        "   - Use 'upcoming' or 'next' in search terms",
        "   - Include 'recent' or 'latest' when appropriate",
        "",
        "IMPORTANT: Use multiple search strategies and simplify search terms to find more results.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

linkedin_events_agent = Agent(
    name="LinkedIn Events Agent",
    role="LinkedIn Events Scraper",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the LinkedIn Events Scraper Agent, specialized in finding professional and business events on LinkedIn.",
        "Key Responsibilities:",
        "1. Search for professional events, webinars, and conferences on LinkedIn",
        "2. Extract the specified number of events from search results (max_events parameter)",
        "3. Extract detailed LinkedIn event information including:",
        "   - Event title and professional description",
        "   - Date and time",
        "   - Host organization and speaker information",
        "   - Event format (webinar, conference, workshop, etc.)",
        "   - Professional networking opportunities",
        "   - Industry relevance and target audience",
        "4. Provide a summary of the events found",
        "",
        "Search Strategy Guidelines:",
        "1. Use multiple search approaches with simplified terms:",
        "   - Search: 'LinkedIn Events [location] [topic] [date]'",
        "   - Search: 'LinkedIn [topic] events [location]'",
        "   - Search: '[topic] conference [location] LinkedIn'",
        "   - Search: 'LinkedIn webinars [topic] [location]'",
        "   - Search: 'LinkedIn [location] professional events'",
        "2. Use ExaTools to find recent LinkedIn event posts and professional event listings",
        "3. Look for professional event aggregator websites",
        "4. Search for industry-specific event calendars and professional organizations",
        "",
        "Max Events Parameter Guidelines:",
        "1. Use the max_events value from the structured input",
        "2. Extract exactly that many events",
        "3. Prioritize events that are:",
        "   - Most relevant to professional development and business networking",
        "   - Have credible speakers and host organizations",
        "   - Offer valuable networking opportunities",
        "   - Match the requested industry/topic",
        "4. Return the specified number of events with complete details",
        "",
        "Time Criteria Guidelines:",
        "1. ALWAYS prioritize recent and upcoming events:",
        "   - For 'this weekend': Look for events in the next 3-7 days",
        "   - For 'next month': Look for events in the next 30-60 days",
        "   - For 'upcoming': Look for events in the next 2-4 weeks",
        "   - For 'next week': Look for events in the next 7-14 days",
        "2. AVOID events that are:",
        "   - More than 2 months old",
        "   - Past their date",
        "   - Not clearly dated",
        "3. When searching, use specific date ranges:",
        "   - Add '2025' or current year to searches",
        "   - Use 'upcoming' or 'next' in search terms",
        "   - Include 'recent' or 'latest' when appropriate",
        "",
        "IMPORTANT: Use multiple search strategies and simplify search terms to find more results.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

facebook_events_agent = Agent(
    name="Facebook Events Agent",
    role="Facebook Events Scraper",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the Facebook Events Scraper Agent, specialized in finding social and community events on Facebook.",
        "Key Responsibilities:",
        "1. Search for social events, parties, and community gatherings using multiple search strategies",
        "2. Extract the specified number of events from search results (max_events parameter)",
        "3. Extract detailed Facebook event information including:",
        "   - Event title and social description",
        "   - Date and time",
        "   - Location and venue details",
        "   - Host and organizer information",
        "   - Event category (party, concert, festival, etc.)",
        "   - Attendee count and engagement",
        "4. Provide a summary of the events found",
        "",
        "Search Strategy Guidelines:",
        "1. Use multiple search approaches:",
        "   - Search for 'Facebook Events [location] [event type] [date]'",
        "   - Search for 'Events in [location] this weekend'",
        "   - Search for '[event type] parties [location]'",
        "   - Use ExaTools to find recent event listings",
        "2. Look for event aggregator websites and local event calendars",
        "3. Search for venue websites and event organizers",
        "",
        "Max Events Parameter Guidelines:",
        "1. Use the max_events value from the structured input",
        "2. Extract exactly that many events",
        "3. Prioritize events that are:",
        "   - Most popular with high engagement",
        "   - Match the requested social category",
        "   - Have good venue information",
        "   - Closest to requested date/location",
        "4. Return the specified number of events with complete details",
        "",
        "Time Criteria Guidelines:",
        "1. ALWAYS prioritize recent and upcoming events:",
        "   - For 'this weekend': Look for events in the next 3-7 days",
        "   - For 'next month': Look for events in the next 30-60 days",
        "   - For 'upcoming': Look for events in the next 2-4 weeks",
        "   - For 'next week': Look for events in the next 7-14 days",
        "2. AVOID events that are:",
        "   - More than 2 months old",
        "   - Past their date",
        "   - Not clearly dated",
        "3. When searching, use specific date ranges:",
        "   - Add '2025' or current year to searches",
        "   - Use 'upcoming' or 'next' in search terms",
        "   - Include 'recent' or 'latest' when appropriate",
        "",
        "IMPORTANT: Use multiple search strategies to find actual events. Don't give up after one search.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

general_events_agent = Agent(
    name="General Events Agent",
    role="General Event Aggregator",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    tools=[DuckDuckGoTools(), ExaTools()],
    instructions=[
        "You are the General Events Agent, specialized in finding events across multiple platforms and sources.",
        "Key Responsibilities:",
        "1. Search for events across various platforms and websites",
        "2. Extract the specified number of events from search results (max_events parameter)",
        "3. Extract comprehensive event details including:",
        "   - Event title and description",
        "   - Date, time, and duration",
        "   - Location and venue information",
        "   - Ticket prices and registration details",
        "   - Organizer and contact information",
        "   - Event category and target audience",
        "4. Provide a summary of the events found",
        "",
        "Search Strategy Guidelines:",
        "1. Use multiple search approaches with simplified terms:",
        "   - Search: 'Events [location] [topic] [date]'",
        "   - Search: '[topic] events [location] this weekend'",
        "   - Search: 'Events [location] upcoming'",
        "   - Search: '[topic] [location] calendar'",
        "   - Search: 'Things to do [location] [date]'",
        "2. Use ExaTools to find recent event listings and local event calendars",
        "3. Look for event aggregator websites (Eventbrite, Meetup, local calendars)",
        "4. Search for venue websites, local newspapers, and community calendars",
        "5. Check for city/town official event calendars and tourism websites",
        "",
        "Max Events Parameter Guidelines:",
        "1. Use the max_events value from the structured input",
        "2. Extract exactly that many events",
        "3. Prioritize events that are:",
        "   - Most relevant to the request",
        "   - From reputable sources",
        "   - Have complete information available",
        "   - Match requested criteria",
        "4. Return the specified number of events with complete details",
        "",
        "Time Criteria Guidelines:",
        "1. ALWAYS prioritize recent and upcoming events:",
        "   - For 'this weekend': Look for events in the next 3-7 days",
        "   - For 'next month': Look for events in the next 30-60 days",
        "   - For 'upcoming': Look for events in the next 2-4 weeks",
        "   - For 'next week': Look for events in the next 7-14 days",
        "2. AVOID events that are:",
        "   - More than 2 months old",
        "   - Past their date",
        "   - Not clearly dated",
        "3. When searching, use specific date ranges:",
        "   - Add '2025' or current year to searches",
        "   - Use 'upcoming' or 'next' in search terms",
        "   - Include 'recent' or 'latest' when appropriate",
        "",
        "IMPORTANT: Use multiple search strategies and simplify search terms to find more results.",
    ],
    add_datetime_to_instructions=True,
    markdown=True,
)

# Event Scraping Team with Route Mode
event_scraping_team = Team(
    name="Event Scraping Team Leader",
    mode="route",
    model=get_azure_model(),
    response_model=EventSearchAnalysis,
    instructions=[
        "You are the Event Scraping Team Leader, responsible for routing event search requests to the most appropriate specialized agent.",
        "Key Responsibilities:", 
        "1. Analyze structured event search requests to determine the best single platform for event search",
        "2. Route each request to exactly one specialized agent based on:",
        "   - Platform preference (Eventbrite, Meetup, LinkedIn, Facebook, etc.)",
        "   - Event type (professional, social, community, etc.)",
        "   - User requirements and preferences",
        "",
        "Routing Guidelines - Choose ONE agent based on these criteria:",
        "1. Eventbrite Agent - Route if request matches:",
        "   - Commercial events, workshops, conferences",
        "   - Ticketed events and paid activities", 
        "   - Professional and business events",
        "2. Meetup Agent - Route if request matches:",
        "   - Community and networking events",
        "   - Tech meetups and professional groups",
        "   - Free community gatherings",
        "3. LinkedIn Events Agent - Route if request matches:",
        "   - Professional development events",
        "   - Business conferences and webinars",
        "   - Industry-specific events",
        "4. Facebook Events Agent - Route if request matches:",
        "   - Social events and parties",
        "   - Local community gatherings",
        "   - Entertainment and cultural events",
        "5. General Events Agent - Route if request matches:",
        "   - No clear platform preference specified",
        "   - General event discovery needed",
        "   - Multiple types of events requested",
        "",
        "IMPORTANT: When you receive a structured input (Pydantic model), extract the information and create a clear text message for routing.",
        "Format the message as: 'Find [max_events] [event_type] events in [location] for [date_range]. Platform preference: [platform_preference]. Category: [category].'",
        "",
        "Always route to exactly one agent that best matches the request. Provide clear reasoning for your routing decision.",
    ],
    members=[
        eventbrite_agent,
        meetup_agent,
        linkedin_events_agent,
        facebook_events_agent,
        general_events_agent,
    ],
    add_datetime_to_instructions=True,
    markdown=True,
    debug_mode=True,
    show_members_responses=True,
)

# Test the event scraping team with Pydantic model input
if __name__ == "__main__":
    print("=== Testing Event Scraping Team with Pydantic Model Input ===\n")
    
    # # Test 1: Professional LinkedIn event search
    # test_request1 = EventSearchRequest(
    #     location="San Francisco",
    #     event_type="professional tech",
    #     date_range="next month",
    #     max_events=2,
    #     platform_preference="LinkedIn",
    #     category="technology"
    # )
    
    # print(f"1. Searching for {test_request1.max_events} professional LinkedIn events in {test_request1.location}...")
    # print(f"   Request: {test_request1.model_dump_json(indent=2)}")
    
    # # Convert Pydantic model to string for team compatibility
    # message_text1 = convert_request_to_message(test_request1)
    # print(f"   Converted message: {message_text1}")
    
    # # Use print_response with string message
    # event_scraping_team.print_response(message=message_text1)
    
    # print("\n" + "="*80 + "\n")
    
    # # Test 2: Social event search
    # test_request2 = EventSearchRequest(
    #     location="New York",
    #     event_type="social parties",
    #     date_range="this weekend",
    #     max_events=3,
    #     platform_preference="Facebook",
    #     category="entertainment"
    # )
    
    # print(f"2. Searching for {test_request2.max_events} social events in {test_request2.location}...")
    # print(f"   Request: {test_request2.model_dump_json(indent=2)}")
    
    # # Convert Pydantic model to string for team compatibility
    # message_text2 = convert_request_to_message(test_request2)
    # print(f"   Converted message: {message_text2}")
    
    # # Use print_response with string message
    # event_scraping_team.print_response(message=message_text2)
    
    # print("\n" + "="*80 + "\n")
    
    # Test 3: General event search with default max_events
    test_request3 = EventSearchRequest(
        location="Austin",
        event_type="tech meetups",
        date_range="upcoming",
        category="technology"
    )
    
    print(f"3. Searching for events with default max_events={test_request3.max_events}...")
    print(f"   Request: {test_request3.model_dump_json(indent=2)}")
    
    # Convert Pydantic model to string for team compatibility
    message_text = convert_request_to_message(test_request3)
    print(f"   Converted message: {message_text}")
    
    # Use print_response with string message
    event_scraping_team.print_response(message=message_text) 