# Data tests: is houses.csv safe to train on?
import pandas as pd

data = pd.read_csv("houses.csv")

def test_columns_are_correct():
    assert list(data.columns) == ["rooms", "size_sqft", "price_lakh"]

def test_no_empty_cells():
    assert data.notna().all().all()

def test_all_values_are_positive():
    assert (data > 0).all().all()

def test_enough_houses():
    assert len(data) >= 20
