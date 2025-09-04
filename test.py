from langchain.agents import initialize_agent, Tool
from langchain.chat_models import ChatOpenAI
import pandas as pd

# ---------- Step 1: Load the CSV ----------
def load_csv(file_path: str):
    df = pd.read_csv(file_path)
    return df

# ---------- Step 2: Validate Data ----------
def validate_data(file_path: str):
    df = pd.read_csv(file_path)
    issues = {}

    # Check for missing values
    missing = df.isnull().sum()
    issues["missing_values"] = missing[missing > 0].to_dict()

    # Check for invalid values
    invalid_rows = []
    for idx, row in df.iterrows():
        if pd.isna(row.get("name")) or pd.isna(row.get("amount paid")) or pd.isna(row.get("date")):
            invalid_rows.append(idx)
        elif not isinstance(row["amount paid"], (int, float)):
            invalid_rows.append(idx)
    issues["invalid_rows"] = invalid_rows

    return issues

# ---------- Step 3: Extract Relevant Columns ----------
def create_new_csv(file_path: str, output_path: str):
    df = pd.read_csv(file_path)
    new_df = df[["name", "amount paid", "date"]]
    new_df.to_csv(output_path, index=False)
    return f"New CSV created at {output_path}"

# ---------- Step 4: LangChain Tools ----------
tools = [
    Tool(
        name="Load CSV",
        func=load_csv,
        description="Loads a CSV file into a dataframe"
    ),
    Tool(
        name="Validate Data",
        func=validate_data,
        description="Checks for missing or invalid data in the CSV"
    ),
    Tool(
        name="Create Clean CSV",
        func=lambda x: create_new_csv(x, "cleaned_output.csv"),
        description="Creates a new CSV with columns name, amount paid, date"
    )
]

# ---------- Step 5: Initialize Agent ----------
llm = ChatOpenAI(temperature=0, model="gpt-4")
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Example Usage
file_path = "data.csv"
print(agent.run(f"Validate the file {file_path} and then create a new CSV with only the required columns."))
