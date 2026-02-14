## Full-stack Data & AI System For Swift, Precise Analytics

### Project Overview

This project is designed with an ultra focus on aligning business requirements and policy constraints with technical scalability.
The objective covers the breadth of a unified analytics and intelligence system that supports customer behavior analysis, decision support, and conversational insights. The solution integrates data engineering, analytics, feature engineering, and AI-driven interfaces, delivered through a backend feature and an intuitive frontend inrerface.

<div align="center">
  <img src="https://github.com/fredie7/accenture_assignment/blob/main/project-images/system%20design.png?raw=true" />
  <br>
   <sub><b>Oroject Architecture</b></sub>
</div>

### Technology Stack
- Data & Backend: Python (Pandas, FastAPI)
- Analytics & Feature Engineering: Statistical analysis, clustering (K-Means)
- AI: LLMs, RAG, LangChain, LangGraph
- Frontend: Next.js
- Architecture: ETL pipelines, star schema modeling, multi-agent AI

### Data Engineering & ETL Pipeline
The project began with the design of an ETL pipeline, following standard data engineering best practices.

#### Data Ingestion & Architecture
Raw data was ingested into a data lake–style folder structure to preserve history and serve the medallion architecture:
- Bronze layer: source data ingestion
- Silver layer: Cleaned and standardized datasets with quality checks
- Gold layer: Business-ready fact and dimension tables optimized for analytics
This structure ensures traceability, scalability, and governance for downstream analytics and AI use cases.

#### Data Cleaning & Quality Handling
The Silver layer transformation involved extensive data quality checks, including handling of missing values, duplicates, data types, and schema consistency.
One of such occurrences was in the treatment of the currency feature - Based on domain knowledge of Nordic markets, EUR was treated as the base currency.
Missing values were first replicated into a separate column (currency_imputed) for financial lineage and auditability.. The primary currency column was then imputed with EUR. All currency codes were normalized to uppercase ISO standards.

#### Duplicate Transactions
Approximately 2,000 duplicate transaction_ids were identified. These were investigated using timestamps, transaction amounts, and customer IDs to detect potential late-arriving data or pipeline retries.
The latest occurence of timestamp for the duplicate transaction_id's were retained to preserve transactional integrity, while the remaining duplicate transacion_id's were dropped. Transactions having no customer_id's were dropped since keeping them defets the purpose of customer analytics, plus they constitute  minority of the wholr breadth of transactions.
#### Data Quality Alert
An anomaly was detected where all customer emails appeared identical.
In line with standard data governance practices, this issue was flagged and documented for review as it may indicate upstream masking errors.

#### Data Modeling: Gold Layer
Once the data was cleaned, it was transformed into a star schema to support analytics, reporting, and AI workloads.
Fact & Dimension Design
- Fact table: Transactions (measures such as amount, exchange rate, time attributes)
- Dimension tables: Customers, Dates, Categories, Currencies

<div align="center">
  <img src="https://github.com/fredie7/accenture_assignment/blob/main/project-images/model.png?raw=true" />
  <br>
   <sub><b>Kimball-style star schema</b></sub>
</div>

#### Slowly Changing Dimension - Type 2
The Customer dimension was modeled as a Slowly Changing Dimension Type 2, anticipating changes such as country or account attributes over time.
This ensures:
- historical accuracy,
- resilience to upstream changes,
- consistency in time-based analytics.
Final quality checks were performed across all fact and dimension tables before exposing the data for downstream consumption.

### Customer Level Insights & Segmentation
Before establishing a many-to-one relationship between the data that drove this line of work, the above features had been chosen to unravel customer level insights and customer behaviour through the following metrics and business questions:

#### Transaction Frequency
- Chosen to measure customer engagement and loyalty.
- Supports: Who are our most engaged customers?
- Value: Enables retention strategies and loyalty program design.

#### Total Spend per Customer
- Chosen as a proxy for customer lifetime value.
- Supports: Who drives most of the revenue? Where should VIP efforts focus?
- Value: Guides revenue concentration analysis and premium customer treatment.

#### Average Transaction Amount
- Chosen to capture purchasing behavior, not just total value.
- Supports: Do customers prefer small frequent purchases or large baskets?
- Value: Informs pricing and discount strategies.

#### Category Preference
- Chosen to understand what customers actually buy.
- Supports: Which products drive loyalty and repeat purchases?
- Value: Enables personalized marketing and cross-sell opportunities.

#### Customer Segmentation
- Chosen to group customers by behavior instead of treating everyone the same.
- Supports: How should marketing, pricing, and retention differ by segment?
- Value: Turns raw metrics into insightful customer personas.

### AI-Multi-Agentic Decision Support System
The AI solution was architected around a Retrieval-Augmented Generation (RAG) approach, where policy compliance and business rules are embedded and retrieved contextually at query time. This ensures that responses are rooted in approved policies and compliance constraints.

On top of the RAG layer, a multi-agent architecture was implemented. Each agent is specialized for a distinct responsibility, such as policy interpretation or customer analytics, and exposed as a tool within the system. It's built to connect with two knowledge sources:
- The structured & cleaned business data stored in te ware house: where agentic tools connect with to perform their distinct functions, and
- The unstructured policy document: which is compressed as embeddings in the Fiss dtbase.

Every step of the way the RAG was tested on its bility to address polivy lookups as predefined. this came with concurrent testing of the vector database and the retrieval chain. Yhe agentic tools which queried the data in the warehouse were lso tested for issues of column definition mismatches.

A supervisory agent, following a ReAct-style reasoning pattern, evaluates the user’s request, determines the appropriate action, and routes the query to the correct agent. This design improves the reliability, transparency, and explainability of responses.



<div align="center">
  <img src="https://github.com/fredie7/accenture_assignment/blob/main/project-images/ai-tool-calls.png?raw=true" />
  <br>
   <sub><b>AI Tool Calls</b></sub>
</div>

<table align="center">
  <tr>
    <td align="center">
      <img src="https://github.com/fredie7/accenture_assignment/blob/main/project-images/conversation_1.png?raw=true" height="300"><br>
      <sub><b>CLient & AI Interaction</b> </sub>
    </td>
    <td align="center">
      <img src="https://github.com/fredie7/accenture_assignment/blob/main/project-images/conversation_2.png?raw=true" height="300"><br>
      <sub><b>CLient & AI Interaction</b> </sub>
    </td>
  </tr>
</table>

To operationalize the system, a FastAPI backend was developed to serve as the interface between stakeholders and the AI services. All endpoints were tested for correctness, performance, and reliability. Finally, a Next.js frontend was built to allow non-technical stakeholders to interact with the system conversationally and generate accurate, business-ready insights without requiring technical expertise.

#### Key Assumptions and Trade-offs

Several pragmatic assumptions were made to balance realism, scope, and delivery quality within the constraints of the assignment. It was assumed that the provided datasets represent a reliable snapshot of customer and transaction behavior, and that missing or inconsistent values could be addressed using standard data engineering best practices without materially distorting business outcomes. For example, missing currency values were imputed using EUR as a base currency, based on domain context of Nordic markets, while preserving lineage for auditability. Also the eamil feature was compltely dropped since it wouldn't account for any business usecase.

From a modeling perspective, a Kimball-style star schema was selected to optimize analytical clarity and downstream BI and AI use cases. While this approach favors simplicity and query performance, it trades off some normalization and flexibility that might be preferred in highly operational or rapidly changing domains. Similarly, customer segmentation was implemented using an unsupervised K-Means model to emphasize interpretability and speed over model complexity, acknowledging that more advanced models could yield incremental gains at the cost of explainability.

For the AI component, the focus was placed on architecture and integration rather than prompt optimization or model fine-tuning. The RAG pipeline uses a limited policy corpus and a single vector store, which is sufficient for demonstrating compliance-aware reasoning but does not yet reflect enterprise-scale document volumes or retrieval strategies.

#### What I Would Improve with More Time

With additional time, the first priority would be to strengthen data quality and governance by introducing automated data validation frameworks (e.g., schema enforcement and anomaly detection) and more robust Slowly Changing Dimension handling for customer attributes.

On the analytics side, the customer behavior models could be enhanced by incorporating temporal features and supervised learning approaches for churn or fraud risk prediction, supported by proper model evaluation and monitoring.

As business volume and data complexity grow, the solution is designed to scale naturally using Azure-native services and Databricks. The current architecture intentionally focuses on clarity and correctness, while remaining fully compatible with cloud-scale execution.

On the data platform side, the ETL pipeline and star-schema model can be migrated to Azure Data Lake Storage and Databricks to support distributed processing, incremental loads, and large-scale analytics. This enables the platform to handle increased transaction volumes, more complex transformations, and advanced analytics use cases without redesign.

For the AI system, improvements would include expanding the document corpus, implementing hybrid retrieval strategies, adding response confidence scoring, and introducing evaluation metrics for RAG accuracy and latency. Finally, the platform could be productionized further through CI/CD pipelines, enhanced observability, and role-based access controls to better reflect an enterprise consulting delivery.




#### Guidelines on how to run the project on your computer:

First, please, ensure that you have python and node installed on your computer

- Then fork the repo
- Go into the project directory:
- cd accenture_assignment

To do all intallations and run from an independent virtual environment(recommended):
- python -m venv venv

Assess virtual environment:
- source venv\bin\activate (hit enter)

Once you're in, install uv:
- pip install uv

- Install all the required dependences:
- uv pip install -r requirements.txt'

To run the ETL:

- cd data_warehouse
- cd etl
- cd gold
run each of the following to loaf or refresh the data:
- python dim_categories.py
- python dim_customers.py
- python dim_dates.py
- python dim_currencies.py
- python fact_transactions.py

- to run the AI:
Return to the root of this project - accenture_assignment
- Then:
- cd AI
- cd app
- run:
- uvicorn main:app --reload

Once it starts, you may want to interact with the front end:
- Split your terminal or cd into the project from another terminal
- Start from the project's root - accenture_assignment
- cd frontend
- cd app
- cd business-agent-ui

To have access to alll supportive javascript packages, run: 
- npm install
- npm install remark-gfm
- npm install react-markdown
- cd app (again)
run:
- npm run dev
- Go to the browser:
enter:
- http://localhost:3000/         (Ensure that you do not have any program running on this port in the mean tim)


