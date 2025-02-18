# Azure Prerequisites 🚀

Before starting with Azure AI Foundry, ensure you have the following resources and configurations:

## Required Azure Resources

### 1. Azure AI Foundry Project 【†L1】
- [Create an AI Foundry project](https://learn.microsoft.com/azure/ai-foundry/get-started)
- Configure project settings and permissions
- Set up Azure AD authentication

### 2. Azure AI Search 【†L4】
- [Set up Azure AI Search](https://learn.microsoft.com/azure/search/search-create-service-portal)
- Configure vector search capabilities
- Set up access keys and endpoints

### 3. Azure Cosmos DB 【†L5】
- [Create a Cosmos DB account](https://learn.microsoft.com/azure/cosmos-db/nosql/quickstart-portal)
- Enable vector search
- Configure connection strings

### 4. Azure Database for PostgreSQL 【†L6】
- [Deploy PostgreSQL Flexible Server](https://learn.microsoft.com/azure/postgresql/flexible-server/quickstart-create-server-portal)
- Enable pgvector extension
- Set up connection strings

### 5. Azure API Management 【†L7】
- [Create an APIM instance](https://learn.microsoft.com/azure/api-management/get-started-create-service-instance)
- Configure AI endpoints
- Set up policies and security

### 6. Azure Logic Apps 【†L8】
- [Create a Logic App](https://learn.microsoft.com/azure/logic-apps/quickstart-create-first-logic-app-workflow)
- Configure workflow connections
- Set up event triggers

### 7. Azure Functions 【†L9】
- [Create a Function App](https://learn.microsoft.com/azure/azure-functions/functions-create-first-function-vs-code)
- Configure application settings
- Set up deployment

## Environment Variables

Configure these variables in your `.env` file:

```bash
# Project Configuration
PROJECT_CONNECTION_STRING=     # Azure AI Foundry project connection string
MODEL_DEPLOYMENT_NAME=         # Primary model deployment name
EMBEDDING_MODEL_DEPLOYMENT_NAME= # Embedding model deployment name
SERVERLESS_MODEL_NAME=         # Serverless model name

# Integration Settings
BING_CONNECTION_NAME=         # Bing grounding tool connection
AZURE_TRACING_GEN_AI_CONTENT_RECORDING_ENABLED=true  # Enable telemetry
AZURE_SDK_TRACING_IMPLEMENTATION=opentelemetry       # Set tracing implementation
```

## Authentication Setup

1. Install Azure CLI:
```bash
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

2. Login to Azure:
```bash
az login
```

3. Set subscription:
```bash
az account set --subscription <subscription-id>
```

For detailed setup instructions, see the [Azure AI Foundry documentation](https://learn.microsoft.com/azure/ai-foundry/get-started).
