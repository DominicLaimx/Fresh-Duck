import streamlit as st
import pandas as pd
import numpy as np



df = pd.read_csv("output_1.csv")

#page settings
st.set_page_config(layout="wide",page_title="MyApp")
st.header("MyAPP")

def highlight_rows(row):
    value = row.loc['Status']
    if value == 'Expiring':
        color = '#FFB3BA' # Red
    elif value == 'Fresh':
        color = '#BAFFC9' # Green
    else:
        color = '#D3D3D3' #Grey
    return ['background-color: {}'.format(color) for r in row]


st.table(df.sort_values(['Status'], ascending=True).
              reset_index(drop=True).style.apply(highlight_rows, axis=1))