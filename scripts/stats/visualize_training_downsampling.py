import plotly.graph_objects as go
from plotly.subplots import make_subplots

import time

x = ["100", "75", "50", "25", "10"]

gsrl_conll_2009_f1 = [92.4, 91.9, 91.5, 90.5, 86.7]
dsrl_conll_2009_f1 = [92.5, 92.0, 91.5, 90.5, 87.9]
conll_2009_train = [100, 75, 50, 25, 10]

conll_2012_f1 = [87.3, 87.1, 86.4, 85.1, 81.2]
dsrl_conll_2012_f1 = [87.4, 87.1, 86.6, 85.7, 83.8]
conll_2012_train = [100, 75, 50, 25, 10]

# # garbage graph
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=[0, 1, 2, 3, 4], y=[0, 1, 4, 9, 16]))
fig1.write_image("scripts/stats/random.pdf")

time.sleep(2)


fig = go.Figure()
fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("CoNLL-2009", "CoNLL-2012"),
    shared_yaxes=True,
    specs=[[{"secondary_y": True}, {"secondary_y": True}]],
    horizontal_spacing=0.05,
)

fig.add_trace(
    go.Bar(
        x=x,
        y=conll_2012_train,
        marker_color="rgba(128, 128, 128, 0.25)",
        showlegend=False,
    ),
    row=1,
    col=2,
    secondary_y=True,
)

fig.add_trace(
    go.Bar(
        x=x,
        y=conll_2009_train,
        marker_color="rgba(128, 128, 128, 0.25)",
        showlegend=False,
    ),
    row=1,
    col=1,
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(
        name="GSRL",
        x=x,
        y=gsrl_conll_2009_f1,
        text=gsrl_conll_2009_f1,
        mode="lines+markers+text",
        textposition=["bottom center"] * 4 + ["bottom center"],
        line=dict(color="#601a4a", width=4),
        marker_size=14,
        # showlegend=False,
        marker={"symbol": "diamond"},
    ),
    row=1,
    col=1,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        name="Our System",
        x=x,
        y=dsrl_conll_2009_f1,
        text=dsrl_conll_2009_f1,
        mode="lines+markers+text",
        textposition=["top center"] * 4 + ["top center"],
        line=dict(color="#63acbe", width=4),
        marker_size=14,
        # showlegend=False,
    ),
    row=1,
    col=1,
    secondary_y=False,
)


fig.add_trace(
    go.Scatter(
        name="GSRL",
        x=x,
        y=conll_2012_f1,
        text=conll_2012_f1,
        mode="lines+markers+text",
        textposition="bottom center",
        line=dict(color="#601a4a", width=4),
        marker_size=14,
        showlegend=False,
        marker={"symbol": "diamond"},
    ),
    row=1,
    col=2,
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(
        name="Our system",
        x=x,
        y=dsrl_conll_2012_f1,
        text=dsrl_conll_2012_f1,
        mode="lines+markers+text",
        textposition="top center",
        line=dict(color="#63acbe", width=4),
        marker_size=14,
        showlegend=False,
    ),
    row=1,
    col=2,
    secondary_y=False,
)


fig.update_yaxes(
    title_text="F1 Score (%)",
    range=[80, 95],
    row=1,
    col=1,
    secondary_y=False,
    tickfont_size=20,
    titlefont_size=24,
)
fig.update_yaxes(row=1, col=1, secondary_y=True, visible=False, range=[0, 100])
fig.update_yaxes(row=1, col=2, range=[40, 100])
fig.update_yaxes(row=1, col=2, secondary_y=True, visible=False, range=[0, 100])
fig.update_xaxes(
    title_text="Train Data (%)", row=1, col=1, tickfont_size=20, titlefont_size=22
)
fig.update_xaxes(
    title_text="Train Data (%)", row=1, col=2, tickfont_size=20, titlefont_size=22
)


fig.update_traces(textfont_size=20)
fig.layout.annotations[0].update(x=0.29, y=0.93, font_size=24)
fig.layout.annotations[1].update(x=0.79, y=0.93, font_size=24)

fig.update_layout(
    autosize=False,
    width=700,
    height=600,
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="left",
        x=0.01,
        bgcolor="rgba(255, 255, 255, 0)",
        bordercolor="rgba(255, 255, 255, 0)",
        font={"size": 25},
    ),
)

fig.write_image("scripts/stats/training_downsampling.pdf")
