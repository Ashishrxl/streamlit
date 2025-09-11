import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO

st.title("Multi-Scenario Forecast Dashboard")

# --- Sample Data: Replace this with your actual forecasts ---
# Assume each forecast is a DataFrame with 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
# forecasts = {'Scenario A': df_a, 'Scenario B': df_b, ...}

# Sidebar to select scenarios
selected_scenarios = st.sidebar.multiselect(
    "Select Scenarios to Compare",
    options=list(forecasts.keys()),
    default=list(forecasts.keys())[:2]
)

if not selected_scenarios:
    st.warning("Please select at least one scenario.")
    st.stop()

# --- Multi-Scenario Dynamic Ranking Plot ---
def plot_multi_scenario_dynamic(forecasts, selected_scenarios):
    fig = go.Figure()
    
    # Combine all forecasts into a single DataFrame
    combined = pd.DataFrame({'ds': forecasts[selected_scenarios[0]]['ds']})
    for scenario in selected_scenarios:
        combined[scenario] = forecasts[scenario]['yhat']
    
    # Compute best/worst scenario per timestamp
    combined['Best'] = combined[selected_scenarios].idxmax(axis=1)
    combined['Worst'] = combined[selected_scenarios].idxmin(axis=1)

    # Plot each scenario with confidence intervals
    for scenario in selected_scenarios:
        df_s = forecasts[scenario]
        fig.add_trace(go.Scatter(
            x=df_s['ds'], y=df_s['yhat'],
            mode='lines', name=f"{scenario} Forecast"
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([df_s['ds'], df_s['ds'][::-1]]),
            y=pd.concat([df_s['yhat_upper'], df_s['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(0,100,80,0.1)',
            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False
        ))
    
    # Highlight best and worst dynamically
    fig.add_trace(go.Scatter(
        x=combined['ds'], y=[combined.loc[i, combined.loc[i,'Best']] for i in range(len(combined))],
        mode='markers', marker=dict(color='green', size=8, symbol='triangle-up'),
        name='Best Scenario'
    ))
    fig.add_trace(go.Scatter(
        x=combined['ds'], y=[combined.loc[i, combined.loc[i,'Worst']] for i in range(len(combined))],
        mode='markers', marker=dict(color='red', size=8, symbol='triangle-down'),
        name='Worst Scenario'
    ))

    fig.update_layout(
        title="Multi-Scenario Forecast with Dynamic Ranking",
        xaxis_title="Date",
        yaxis_title="Forecast Value",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig

fig = plot_multi_scenario_dynamic(forecasts, selected_scenarios)
st.plotly_chart(fig, use_container_width=True)

# --- Download Functionality ---
def convert_df_to_excel(forecasts, selected_scenarios):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for scenario in selected_scenarios:
            forecasts[scenario].to_excel(writer, sheet_name=scenario, index=False)
    processed_data = output.getvalue()
    return processed_data

excel_data = convert_df_to_excel(forecasts, selected_scenarios)
st.download_button(
    label="Download Forecasts as Excel",
    data=excel_data,
    file_name="multi_scenario_forecasts.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)