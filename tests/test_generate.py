import pandas as pd
from src.data.generate import generate_synthetic_conversations


def test_generate_length():
    df = generate_synthetic_conversations(500, seed=1)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 500
