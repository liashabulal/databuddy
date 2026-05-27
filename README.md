# DataBuddy 🐾
An LLM-powered EDA agent that reasons about your data and decides 
which analyses to run — no code required.

🔴 [Live Demo](https://huggingface.co/spaces/liashabulal/databuddy) · 
📄 [Technical Report](EDA_Agent__Technical_Report.pdf)

## What is DataBuddy?
DataBuddy is a dataset-agnostic Exploratory Data Analysis agent. 
Load any Hugging Face dataset by name, ask a question in plain English, 
and the agent decides which analytical tools to run, in what order, 
and how to interpret the results.

Unlike a fixed EDA script, DataBuddy reasons about your specific dataset 
— adapting its analysis to the structure, types, and quality of the data 
it receives.

## Why is this interesting?
Every dataset is different. A fixed EDA script runs the same operations 
regardless of whether they make sense. DataBuddy treats EDA as a 
reasoning problem and then the agent observes the data, selects appropriate 
tools, sequences them logically, and synthesises results into actionable 
insights.

## How it works
- **17 specialised tools** — each covering a specific EDA operation 
  (shape, missing values, distributions, correlations, outliers, 
  visualisations, preprocessing recommendations)
- **LangGraph ReAct agent** — reasons about which tools to call and 
  in what order based on the user's question
- **Closure pattern** — dataset is injected into tools at runtime, 
  keeping data and logic cleanly separated
- **MemorySaver** — maintains conversation state across multi-turn 
  queries within a session
- **Gradio UI** — conversational interface with inline visualisations

## Tech Stack
- LangChain + LangGraph
- Google Gemini (gemini-3.1-flash-lite)
- Hugging Face Datasets
- Gradio
- Pandas, Matplotlib, Seaborn

## How to Run
1. Clone this repo
2. Install dependencies: `pip install -r requirements.txt`
3. Get a free Gemini API key from 
   [Google AI Studio](https://aistudio.google.com)
4. Run: `python app.py`
5. Enter your API key and any Hugging Face dataset name in the UI

## Validated On
- `scikit-learn/iris`
- `mstz/titanic`  
- `scikit-learn/adult-census-income`

## Author
Built as part of an independent research project investigating 
LLM agent decision-making in automated data analysis.
