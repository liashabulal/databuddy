import pandas as pd
from datasets import load_dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import gradio as gr
from tool_descriptions import DESCRIPTIONS

# ── Dataset loader ──
def load_hf_dataset(dataset_name: str, split: str = "train") -> pd.DataFrame:
    """Loads any Hugging Face dataset by name and returns it as a pandas DataFrame."""
    dataset = load_dataset(dataset_name, split=split)
    df = dataset.to_pandas()
    df = df.replace(["?", "NA", "N/A", "unknown", "none", "-", ""], pd.NA)
    return df

# ── Tools ──
def make_tools(df, descriptions):

    @tool
    def get_shape(input: str = "") -> str:
        """Use this first. Returns the number of rows and columns in the dataset."""
        return f"Rows: {df.shape[0]}, Columns: {df.shape[1]}"

    @tool
    def get_missing_values(input: str = "") -> str:
        """Returns count and percentage of missing values for each column."""
        missing = df.isnull().sum()
        pct = (missing / len(df) * 100).round(2)
        result = pd.DataFrame({"missing": missing, "percent": pct})
        return result[result["missing"] > 0].to_string()

    @tool
    def get_dtypes(input: str = "") -> str:
        """Returns the data type of each column in the dataset."""
        return df.dtypes.to_string()

    @tool
    def get_cardinality(input: str = "") -> str:
        """Returns the number of unique values in each column."""
        return df.nunique().to_string()

    @tool
    def get_descriptive_stats(input: str = "") -> str:
        """Returns descriptive statistics like mean, std, min, max for numeric columns."""
        return df.describe().to_string()

    @tool
    def get_value_counts(column: str) -> str:
        """Returns value counts for a specific column. Input must be a column name."""
        if column not in df.columns:
            return f"Column '{column}' not found."
        return df[column].value_counts().to_string()

    @tool
    def get_class_imbalance(column: str) -> str:
        """Returns class distribution and imbalance ratio for a target column. Input must be a column name."""
        if column not in df.columns:
            return f"Column '{column}' not found."
        counts = df[column].value_counts()
        ratio = (counts / counts.sum() * 100).round(2)
        return pd.DataFrame({"count": counts, "percent": ratio}).to_string()

    @tool
    def get_correlation_matrix(input: str = "") -> str:
        """Returns the correlation matrix for all numeric columns."""
        return df.select_dtypes(include="number").corr().round(3).to_string()

    @tool
    def get_target_correlation(column: str) -> str:
        """Returns correlation of all numeric columns with a target column. Input must be a column name."""
        if column not in df.columns:
            return f"Column '{column}' not found."
        corr = df.select_dtypes(include="number").corr()[column].drop(column)
        return corr.sort_values(ascending=False).to_string()

    @tool
    def get_group_target_analysis(input: str) -> str:
        """Returns mean target value grouped by a categorical column. Input format: 'group_col,target_col'"""
        try:
            group_col, target_col = [c.strip() for c in input.split(",")]
            return df.groupby(group_col)[target_col].mean().round(3).to_string()
        except:
            return "Input format must be: 'group_col,target_col'"

    @tool
    def detect_outliers_iqr(column: str) -> str:
        """Detects outliers in a numeric column using IQR method. Input must be a column name."""
        if column not in df.columns:
            return f"Column '{column}' not found."
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        outliers = df[(df[column] < q1 - 1.5 * iqr) | (df[column] > q3 + 1.5 * iqr)]
        return f"Outliers in '{column}': {len(outliers)} rows ({(len(outliers)/len(df)*100):.2f}%)"

    @tool
    def plot_histogram(column: str) -> str:
        """Plots a histogram for a numeric column. Input must be a column name."""
        import matplotlib.pyplot as plt
        if not column or column.strip() == "":
            return "Please provide a column name."
        if column not in df.columns:
            return f"Column '{column}' not found."
        fig, ax = plt.subplots()
        df[column].dropna().plot(kind="hist", ax=ax)
        ax.set_title(f"Histogram of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        path = f"/tmp/{column}_histogram.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        return path  # ── FIXED: was returning wrong path

    @tool
    def plot_correlation_heatmap(input: str = "") -> str:
        """Plots a heatmap of the correlation matrix for numeric columns."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            return "Not enough numeric columns to generate a correlation heatmap."
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("/tmp/correlation_heatmap.png", dpi=100, bbox_inches='tight')
        plt.close()
        return "/tmp/correlation_heatmap.png"

    @tool
    def plot_group_target(input: str) -> str:
        """Plots mean target value by group as a bar chart. Input format: 'group_col,target_col'"""
        import matplotlib.pyplot as plt
        if not input or input.strip() == "":
            return "Input format must be: 'group_col,target_col'"
        try:
            group_col, target_col = [c.strip() for c in input.split(",")]
            fig, ax = plt.subplots()
            df.groupby(group_col)[target_col].mean().plot(kind="bar", ax=ax)
            ax.set_title(f"Mean {target_col} by {group_col}")
            ax.set_xlabel(group_col)
            ax.set_ylabel(f"Mean {target_col}")
            plt.tight_layout()
            path = f"/tmp/{group_col}_{target_col}_group.png"
            plt.savefig(path, dpi=100, bbox_inches='tight')
            plt.close()
            return path
        except:
            return "Input format must be: 'group_col,target_col'"

    @tool
    def plot_boxplot(column: str) -> str:
        """Plots a boxplot for a numeric column. Input must be a column name."""
        import matplotlib.pyplot as plt
        if not column or column.strip() == "":
            return "Please provide a column name."
        if column not in df.columns:
            return f"Column '{column}' not found."
        fig, ax = plt.subplots()
        df[[column]].dropna().plot(kind="box", ax=ax)
        ax.set_title(f"Boxplot of {column}")
        ax.set_ylabel(column)
        plt.tight_layout()
        path = f"/tmp/{column}_boxplot.png"
        plt.savefig(path, dpi=100, bbox_inches='tight')
        plt.close()
        return path

    @tool
    def get_preprocessing_recommendations(input: str = "") -> str:
        """Analyzes the dataset and returns preprocessing recommendations."""
        recommendations = []
        missing = df.isnull().sum()
        for col in df.columns:
            if missing[col] > 0:
                pct = missing[col] / len(df) * 100
                if pct > 50:
                    recommendations.append(f"DROP '{col}' — {pct:.1f}% missing")
                elif df[col].dtype == "object":
                    recommendations.append(f"IMPUTE '{col}' with mode — {pct:.1f}% missing")
                else:
                    recommendations.append(f"IMPUTE '{col}' with median — {pct:.1f}% missing")
        for col in df.select_dtypes(include="number").columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outliers = df[(df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)]
            if len(outliers) > 0:
                recommendations.append(f"OUTLIERS in '{col}': {len(outliers)} rows — consider scaling or capping")
        for col in df.select_dtypes(include="object").columns:
            n = df[col].nunique()
            if n == 2:
                recommendations.append(f"ENCODE '{col}' with label encoding — binary column")
            elif n <= 10:
                recommendations.append(f"ENCODE '{col}' with one-hot encoding — {n} categories")
            else:
                recommendations.append(f"ENCODE '{col}' with target/frequency encoding — high cardinality ({n})")
        return "\n".join(recommendations) if recommendations else "No major preprocessing issues found."
    # swap descriptions
    for t in [get_shape, get_missing_values, get_dtypes, get_cardinality,
              get_descriptive_stats, get_value_counts, get_class_imbalance,
              get_correlation_matrix, get_target_correlation, get_group_target_analysis,
              detect_outliers_iqr, plot_histogram, plot_correlation_heatmap,
              plot_group_target, plot_boxplot, get_preprocessing_recommendations]:
        t.description = descriptions[t.name]
        
    return [
        get_shape, get_missing_values, get_dtypes, get_cardinality,
        get_descriptive_stats, get_value_counts, get_class_imbalance,
        get_correlation_matrix, get_target_correlation, get_group_target_analysis,
        detect_outliers_iqr, plot_histogram, plot_correlation_heatmap,
        plot_group_target, plot_boxplot, get_preprocessing_recommendations
    ]

# ── Agent initializer — now takes API key from user ──
def initialize_agent(api_key, dataset_name, model_name, description_level):
    api_key = api_key.strip()
    dataset_name = dataset_name.strip()
    if not api_key:
        return None, "Please enter your Gemini API key."
    if not dataset_name:
        return None, "Please enter a dataset name."
    try:
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key
        )
        df = load_hf_dataset(dataset_name)
        descriptions = DESCRIPTIONS[description_level]
        tools = make_tools(df, descriptions)
        agent = create_react_agent(llm, tools, checkpointer=MemorySaver())
        return agent, f"✅ Dataset '{dataset_name}' loaded — {df.shape[0]} rows, {df.shape[1]} columns. Ask me anything!"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"

# ── Chat runner ──
def run_chat(message, history, agent):
    if agent is None:
        return "Please load a dataset first.", [] , []
    try:
        config = {"configurable": {"thread_id": "eda_session"}}
        output = ""
        images = []
        tool_sequence = []
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config
        ):
            if "agent" in chunk:
                for msg in chunk["agent"]["messages"]:
                    if msg.content:
                        for block in msg.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                output += block["text"] + "\n"
                    if msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            tool_sequence.append(tool_call["name"])
                            
            if "tools" in chunk:
                for msg in chunk["tools"]["messages"]:
                    if msg.content and msg.content.endswith(".png"):
                        images.append(msg.content)
        return output, images, tool_sequence
    except Exception as e:
        return f"❌ Error: {str(e)}", [], []
        
# ── Experiment endpoint ──
def run_experiment(api_key: str, dataset_name: str, description_level: str, question: str)-> tuple[str, list, str]:
    try:
        agent, status = initialize_agent(api_key, dataset_name, "gemini-3.1-flash-lite", description_level)
        if agent is None:
            return "", [], f"FAILED: {status}"
        text, images, tool_sequence = run_chat(question, [], agent)
        if not text and not tool_sequence:
            return "", [], "FAILED: empty response"
        return text, tool_sequence, "OK"
    except Exception as e:
        return "", [], f"FAILED: {str(e)}"
# ── UI ──
if __name__ == "__main__":
    custom_css = """
        .gradio-container {
            background-color: #FDFAF4 !important;
            font-family: 'Segoe UI', sans-serif !important;
        }
        #load-btn {
            background-color: #B8C9D9 !important;
            color: #2A3D4D !important;
            border: none !important;
            border-radius: 8px !important;
        }
        #load-btn:hover {
            background-color: #9ab3c7 !important;
        }
        .gr-input, .gr-textarea {
            background-color: white !important;
            border: 1px solid #B8C9D9 !important;
            border-radius: 8px !important;
            color: #2C2C2A !important;
        }
        .gr-chatbot .user {
            background-color: #5C1A2E !important;
            color: #F5E6C8 !important;
            border-radius: 14px 14px 4px 14px !important;
        }
        .gr-chatbot .bot {
            background-color: #F5E6C8 !important;
            color: #2C2C2A !important;
            border-radius: 14px 14px 14px 4px !important;
            border: 1px solid #B8C9D9 !important;
        }
    """

    with gr.Blocks() as demo:

        gr.HTML("""
        <div style="background-color: #FDFAF4; padding: 10px;">
            <p style="color: #FDFAF4;">.</p>
        </div>
        """)

        gr.HTML("""
            <div style="margin-top: 60px; background-color: #5C1A2E; padding: 20px 24px; border-radius: 12px; margin-bottom: 16px; display: flex; align-items: center; gap: 16px;">
                <div style="font-size: 48px;">🐾</div>
                <div>
                    <h1 style="color: #F5E6C8; margin: 0; font-size: 28px; font-weight: 600;">DataBuddy</h1>
                    <p style="color: #B8C9D9; margin: 0; font-size: 14px; letter-spacing: 0.08em;">Upload · Ask · Understand</p>
                </div>
            </div>
        """)

        agent_state = gr.State(None)

        # ── NEW: API key input ──
        api_key_input = gr.Textbox(
            placeholder="Enter your Gemini API key...",
            label="Gemini API Key",
            type="password"  # hides the key as user types
        )

        model_dropdown = gr.Dropdown(
        choices=[
            ("gemini-3.1-flash-lite  ", "gemini-3.1-flash-lite"),
            ("gemini-3-flash  ", "gemini-3-flash"),
            ("gemini-2.5-flash  ", "gemini-2.5-flash"),
            ("gemini-2.0-flash  ", "gemini-2.0-flash"),
        ],
        value="gemini-3.1-flash-lite",
        label="Select Model"
    )

        with gr.Row():
            dataset_input = gr.Textbox(
                placeholder="Enter Hugging Face dataset name  e.g. scikit-learn/iris",
                label="",
                scale=4
            )
            load_btn = gr.Button("Load Dataset 🐾", scale=1, elem_id="load-btn")

        status = gr.Textbox(
            label="",
            interactive=False,
            placeholder="Status will appear here...",
        )

        gr.HTML("""
            <p style="color: #5C1A2E; font-size: 13px; margin: 4px 0 8px;">
                💡 Try asking: <em>What are the missing values?</em> · <em>Show correlations</em> · <em>Perform a full EDA</em> · <em>Detect outliers</em>
            </p>
        """)

        chatbot = gr.Chatbot(height=450)
        msg_input = gr.Textbox(placeholder="Ask anything about your dataset...", label="")
        submit_btn = gr.Button("Ask 🐾")

        def respond(message, history, agent):
            text, images, tools = run_chat(message, history, agent)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": text})
            for img in images:
                history.append({"role": "assistant", "content": gr.Image(img)})
            return history, ""

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, agent_state],
            outputs=[chatbot, msg_input]
        )

        load_btn.click(
            fn=initialize_agent,
            inputs=[api_key_input, dataset_input, model_dropdown, gr.State("current")],  # ── NEW: passes key
            outputs=[agent_state, status]
        )
        gr.api(run_experiment, api_name="run_experiment")

    demo.launch(css=custom_css)