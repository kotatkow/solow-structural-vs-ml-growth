import wbgapi as wb
import pandas as pd
from pathlib import Path

indicators = {
    "NY.GDP.PCAP.KD": "gdp_per_capita",
    "NE.GDI.TOTL.ZS": "investment_rate",
    "SP.POP.GROW": "population_growth",
    "SE.SEC.ENRR": "school_enrollment"
}

years = [1990, 2000, 2010, 2020]

df = wb.data.DataFrame(indicators, time=years).reset_index()

df = df.rename(columns={
    "economy": "country",
    "time": "year"
})

Path("data/raw").mkdir(parents=True, exist_ok=True)
df.to_csv("data/raw/wdi_solow_raw_extended.csv", index=False)

print("Saved to data/raw/wdi_solow_raw_extended.csv")
print(df.head())