# AI Agent Tools in Azure AI Foundry 🛠️

## Available Tools

### 1. File Search Tool
The File Search tool enables agents to search through document repositories using Azure AI Search's vector capabilities.

```python
from azure.ai.projects.models import FileSearchTool

# Create agent with file search
agent = project_client.agents.create(
    name="health-advisor",
    tools=[FileSearchTool(name="document-search")]
)
```

### 2. Code Interpreter Tool
The Code Interpreter tool allows agents to execute Python code for data analysis and calculations.

```python
from azure.ai.projects.models import CodeInterpreterTool

# Create agent with code interpreter
agent = project_client.agents.create(
    name="health-calculator",
    tools=[CodeInterpreterTool(name="python-calculator")]
)
```

### 3. Bing Grounding Tool
Ground agent responses in real-time web data using Bing Search.

```python
from azure.ai.projects.models import BingGroundingTool

# Create agent with Bing grounding
agent = project_client.agents.create(
    name="health-researcher",
    tools=[BingGroundingTool(name="web-search")]
)
```

## Tool Combinations
Agents can use multiple tools together for enhanced capabilities:

```python
# Create multi-tool agent
agent = project_client.agents.create(
    name="health-advisor-pro",
    tools=[
        FileSearchTool(name="document-search"),
        CodeInterpreterTool(name="python-calculator"),
        BingGroundingTool(name="web-search")
    ]
)
```

For implementation examples, see:
- [Code Interpreter Example](../2-notebooks/2-agent_service/2-code_interpreter.ipynb)
- [File Search Example](../2-notebooks/2-agent_service/3-file-search.ipynb)
- [Bing Grounding Example](../2-notebooks/2-agent_service/4-bing_grounding.ipynb)
