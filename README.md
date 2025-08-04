# AgnoDemo - Multi-Agent Systems Implementation

A comprehensive showcase of Agno framework implementations demonstrating different team modes, agent specializations, and real-world applications.

## File Analysis

### `thai_cuisine_coordinator.py`

**Agno Implementation**: Coordinate Mode Team with RAG Knowledge Base

**Core Components**:
- **5 Specialized Agents**: Each with distinct roles (Chef Somchai, Dr. Pim, Khun Lek, Ajarn Siri, Dr. Nong)
- **Knowledge Base**: `PDFUrlKnowledgeBase` with Thai recipes PDF
- **Vector Database**: `PgVector` with hybrid search type
- **Embedder**: `AzureOpenAIEmbedder` using text-embedding-ada-002
- **Team Mode**: `mode="coordinate"` for task delegation and synthesis

**Agno Features Used**:
- `Agent` with `add_references=True` and `search_knowledge=False`
- `Team` with `enable_agentic_context=True` and `share_member_interactions=True`
- `PDFUrlKnowledgeBase` with `upsert=True` for knowledge loading
- Coordinate mode task delegation to specialized agents

**Technical Flow**:
1. Knowledge base loads Thai recipes into PgVector table
2. Team leader analyzes queries and delegates to appropriate agents
3. Agents use RAG to retrieve relevant recipe information
4. Team leader synthesizes responses into comprehensive guidance

### `restaurant_research_team.py`

**Agno Implementation**: Collaborate Mode Team with Multiple Toolkits

**Core Components**:
- **6 Business Agents**: Market Analyst, Location Scout, Financial Analyst, Web Scraper, Report Generator, Communication Manager
- **Tool Integration**: `DuckDuckGoTools`, `YFinanceTools`, `PandasTools`, `WebsiteTools`, `FileTools`, `EmailTools`, `CalculatorTools`
- **Team Mode**: `mode="collaborate"` for concurrent agent responses

**Agno Features Used**:
- `Agent` with specialized tool assignments per role
- `Team` with `show_members_responses=True` and `markdown=True`
- `add_datetime_to_instructions=True` for temporal context
- Collaborate mode for parallel agent execution

**Technical Flow**:
1. All agents receive the same query concurrently
2. Each agent uses their specialized tools to gather information
3. Team leader reviews all responses and builds consensus
4. Final synthesis combines all perspectives

### `event_scraping_team.py`

**Agno Implementation**: Route Mode Team with Structured Output

**Core Components**:
- **5 Platform Agents**: Eventbrite, Meetup, LinkedIn, Facebook, General
- **Response Model**: `EventSearchAnalysis` with `EventDetails` Pydantic models
- **Team Mode**: `mode="route"` for intelligent agent selection
- **Tools**: `DuckDuckGoTools` and `ExaTools` for web search

**Agno Features Used**:
- `Agent` with `response_model=EventSearchAnalysis`
- `Team` with routing logic for platform-specific agent selection
- `add_name_to_instructions=True` for agent identity
- Route mode for single-agent task execution

**Technical Flow**:
1. Team leader analyzes request to determine optimal platform
2. Routes to single most appropriate agent
3. Agent performs platform-specific event search
4. Returns structured `EventSearchAnalysis` response

### `event_scraping_team_pydantic.py`

**Agno Implementation**: Advanced Structured Input/Output with Pydantic

**Core Components**:
- **Input Model**: `EventSearchRequest` with validation
- **Output Model**: `EventSearchAnalysis` with `EventDetails`
- **Conversion Function**: `convert_request_to_message()` for team compatibility
- **Route Mode**: Intelligent routing with structured criteria

**Agno Features Used**:
- Pydantic models for type-safe input/output
- `Agent` with `response_model` for structured responses
- `Team` with enhanced routing instructions
- Structured data validation and conversion

**Technical Flow**:
1. Pydantic model validates input request
2. Conversion function transforms to team-compatible message
3. Route mode selects appropriate platform agent
4. Agent returns structured Pydantic response

### `run_pgvector.ps1`

**Agno Implementation**: Vector Database Infrastructure

**Purpose**: Automated PgVector container setup for RAG systems

**Technical Components**:
- Docker container with PostgreSQL and PgVector extension
- Volume persistence for vector data
- Port mapping for database connectivity
- Environment variables for database configuration


## Agno Framework Concepts Demonstrated

### Team Modes
1. **Coordinate Mode**: Task delegation and synthesis (`thai_cuisine_coordinator.py`)
2. **Collaborate Mode**: Concurrent agent responses (`restaurant_research_team.py`)
3. **Route Mode**: Intelligent agent selection (`event_scraping_team.py`, `event_scraping_team_pydantic.py`)

### Agent Specialization
- **Role-based Agents**: Each agent has specific expertise and tools
- **Tool Integration**: Specialized toolkits per agent role
- **Response Models**: Pydantic models for structured output
- **Instructions**: Detailed agent behavior specifications

### Knowledge Management
- **RAG Systems**: PDF knowledge base with vector embeddings
- **Vector Search**: PgVector with hybrid search capabilities
- **Embedding Generation**: Azure OpenAI text embeddings
- **Knowledge Loading**: Upsert operations for data persistence

### Tool Integration
- **Search Tools**: DuckDuckGo, Exa for web search
- **Data Tools**: Pandas, CSV for data processing
- **Financial Tools**: YFinance for market data
- **Communication Tools**: Email for stakeholder outreach
- **Web Tools**: Website scraping for competitive intelligence


This implementation demonstrates advanced Agno framework usage with multiple team modes, specialized agents, comprehensive tool integration, and production-ready knowledge management systems.

---

**Imane Labbassi**  
*Building intelligent multi-agent systems with Agno*