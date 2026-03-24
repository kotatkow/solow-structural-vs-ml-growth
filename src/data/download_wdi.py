import wbgapi as wb
import pandas as pd
from pathlib import Path

# World Bank indicators
indicators = {
    "NY.GDP.PCAP.KD": "gdp_per_capita",      # GDP per capita, constant USD
    "NE.GDI.TOTL.ZS": "investment_rate",     # Gross capital formation (% of GDP)
    "SP.POP.GROW": "population_growth"       # Population growth (annual %)
}

years = [1990, 2000, 2010, 2020]

# Download as long-format dataframe
df = wb.data.DataFrame(indicators, time=years).reset_index()

# Clean columns
df = df.rename(columns={
    "economy": "country",
    "time": "year"
})

# Make sure output folder exists
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Save
df.to_csv("data/raw/wdi_solow_raw.csv", index=False)

print("Saved to data/raw/wdi_solow_raw.csv")
print(df.head())