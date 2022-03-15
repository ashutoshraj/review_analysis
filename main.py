import pandas as pd
import numpy as np
import shap
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
from pkg_summarize.summarize import extract_summary
from pkg_keyword.keyword import extract
from pkg_sentiment.sentiment import get_sentiment
from pkg_preprocess.preprocess import clean_data
import plotly.express as px

st.set_page_config(page_title="Analytics Dashboard", layout="wide")
st.markdown("## Analytics Dashboard")

col1, buffer, col2 = st.columns([3, 1, 3])

data = col2.expander('Data View')
config = col1.expander("Configuration")
detail_view = config.checkbox('Detailed View')
pre_load = config.checkbox('Preloaded Data')

df = pd.read_csv(r"Downloads\G2_ringcentral-video.csv")

data.dataframe(df)

df["review"] = df["Pros"] + df["Cons"]

if not detail_view:
    df["Rating"] = df["Rating"].astype(int)
    df["Rating"] = df["Rating"].replace(0, 1)

df_neg = df[df["Rating"] <= 3][["Rating", "review"]]
df_neu = df[(df["Rating"] > 3) & (df["Rating"] < 4)][["Rating", "review"]]
df_pos = df[df["Rating"] >= 4][["Rating", "review"]]

fig1 = px.pie(df, values="Rating", names="Rating")

fig1.update_layout(
 title="<b>Ringcentral Rating Distribution</b>")
col1.plotly_chart(fig1)

# col2.markdown("### Positive review summary")
pos_doc = clean_data(df_pos)
neg_doc = clean_data(df_neg)

pos_keyword = extract(pos_doc)
neg_keyword = extract(neg_doc)

neg_keyword_ = [(x, -100 * y) for x, y in neg_keyword]
pos_keyword_ = [(x, 100 * y) for x, y in pos_keyword]

key = pd.concat([pd.DataFrame(pos_keyword_), pd.DataFrame(neg_keyword_)])
key = key.reset_index(drop=True)

key["key_phrase"] = key[0]
key["score"] = key[1]
del key[1]
del key[0]

fig2 = go.Figure()

category_order = list(key["key_phrase"].unique())

fig2.add_trace(go.Bar(
 x=key["score"],
 y=key["key_phrase"],
 orientation='h',
 marker_color=key["score"]
))

fig2.update_layout(
 barmode='relative',
 title="<b>Ringcentral Key Features</b>"
)

col1.plotly_chart(fig2)

pos_doc_text = ". ".join(pos_doc)
neg_doc_text = ". ".join(neg_doc)

col2.subheader("Positive review summary")
# col2.caption(extract_summary(pos_doc_text))
col2.caption(
 "Able to conduct team-building meetings remotely some employees are more comfortable with Ringcentral as opposed "
 "to other providers/apps . I love that the 'dashboard is simple and easy to use It really helps some of the older "
 "people I work with who are a little unenthusiastic about ZOOM . I like that its all tied into one platform with "
 "ring central calling .")

col2.subheader("Negative review summary")
# col2.caption(extract_summary(neg_doc_text))
col2.caption("If you join from the browser, it 's almost unuseable. The lag was awful it had an echo and the camera "
 "stopped working. .There are often glitches in the program where it turns voices into sounding robotic "
 "does n't allow a screen share. Platform feels dated and clunky Admin portal is difficult to navigate "
 "Video conferencing sessions frequently lag glitch or do not work as expected.")