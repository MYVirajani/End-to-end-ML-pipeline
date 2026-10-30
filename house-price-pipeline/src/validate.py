import sys
import pandas as pd
import pandera as pa
import yaml

params = yaml.safe_load(open("params.yaml"))["data"]
df = pd.read_csv(params["raw_path"])

schema = pa.DataFrameSchema(
    {
        "Area": pa.Column(float, pa.Check.gt(0), coerce=True),
        "Bedrooms": pa.Column(int, pa.Check.in_range(0, 20), coerce=True),
        "Bathrooms": pa.Column(int, pa.Check.in_range(0, 20), coerce=True),
        "Floors": pa.Column(int, pa.Check.in_range(1, 10), coerce=True),
        "YearBuilt": pa.Column(int, pa.Check.in_range(1800, 2026), coerce=True),
        "Location": pa.Column(str),
        "Condition": pa.Column(str, pa.Check.isin(["Excellent", "Good", "Fair", "Poor"])),
        "Garage": pa.Column(str, pa.Check.isin(["Yes", "No"])),
        "Price": pa.Column(float, pa.Check.gt(0), coerce=True),
    },
    checks=pa.Check(lambda d: len(d) > 500, error="Too few rows"),
    strict=False,
)

try:
    schema.validate(df, lazy=True)
    print(f"Data OK: {df.shape[0]} rows, {df.shape[1]} columns")
except pa.errors.SchemaErrors as e:
    print(e.failure_cases)
    sys.exit(1)