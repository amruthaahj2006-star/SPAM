import pandas as pd

def load_data(path: str) -> pd.DataFrame:
    """
    Load the spam CSV dataset.
    Encoding is set to latin-1 because the common SMS spam dataset uses this.
    """
    df = pd.read_csv(path, encoding="latin-1")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset:
    - Remove Unnamed columns
    - Rename v1 -> label, v2 -> message
    - Drop missing messages
    - Drop duplicate messages
    - Convert labels: ham -> 0, spam -> 1
    """
    # Remove Unnamed columns
    unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=unnamed_cols, errors="ignore")

    # Rename columns if present
    if "v1" in df.columns:
        df = df.rename(columns={"v1": "label"})
    if "v2" in df.columns:
        df = df.rename(columns={"v2": "message"})

    # Keep only label and message if others exist
    df = df[[c for c in ["label", "message"] if c in df.columns]]

    # Drop missing messages
    df = df.dropna(subset=["message"])

    # Strip whitespace from messages
    df["message"] = df["message"].astype(str).str.strip()

    # Drop duplicates
    df = df.drop_duplicates(subset=["message"])

    # Map labels
    df["label"] = df["label"].map({"ham": 0, "spam": 1})

    # Drop rows with unknown labels (if any)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df