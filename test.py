import time
import sys
import numpy as np
import pandas as pd
import random
import json
import difflib

gsheetid = "1HSIjDNlYd58u6IuLOwQkqvH4uRsxDNjYbLSRfXPDhfg"
sheet_name = "Data"
gsheet_url = "https://docs.google.com/spreadsheets/d/{}/gviz/tq?tqx=out:csv&sheet={}".format(gsheetid, sheet_name)
df = pd.read_csv(gsheet_url)
print(df)