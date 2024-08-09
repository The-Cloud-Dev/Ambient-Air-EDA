import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
from openai import OpenAI
import numpy as np
import json
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

state_expanded_forms = {
    'AP': 'Andhra Pradesh',
    'AR': 'Arunachal Pradesh',
    'AS': 'Assam',
    'BR': 'Bihar',
    'CG': 'Chhattisgarh',
    'CH': 'Chandigarh',
    'DL': 'Delhi',
    'GJ': 'Gujarat',
    'HP': 'Himachal Pradesh',
    'HR': 'Haryana',
    'JH': 'Jharkhand',
    'JK': 'Jammu and Kashmir',
    'KA': 'Karnataka',
    'KL': 'Kerala',
    'MH': 'Maharashtra',
    'ML': 'Meghalaya',
    'MN': 'Manipur',
    'MP': 'Madhya Pradesh',
    'MZ': 'Mizoram',
    'NL': 'Nagaland',
    'OR': 'Odisha',
    'PB': 'Punjab',
    'PY': 'Puducherry',
    'RJ': 'Rajasthan',
    'SK': 'Sikkim',
    'TG': 'Telangana',
    'TN': 'Tamil Nadu',
    'TR': 'Tripura',
    'UK': 'Uttarakhand',
    'UP': 'Uttar Pradesh',
    'WB': 'West Bengal'
}

GOOD_CO_MAX, MODERATE_CO_MAX, SEVERE_CO_MAX = 2, 5, 9

# Streamlit app title
st.title('Air Quality Data Viewer')

# Mapping for state names to abbreviations
abbreviation_mapping = {
    name: abbr
    for abbr, name in state_expanded_forms.items()
}
selected_name = st.selectbox("Select a State or UT",
                             list(state_expanded_forms.values()))
selected_abbreviation = abbreviation_mapping[selected_name]

# Load data
file_path = f"Datasets/{selected_abbreviation}.csv"
df = pd.read_csv(file_path)
df['From Date'] = pd.to_datetime(df['From Date'], format='%Y-%m-%d')

# Select year
selected_year = st.slider('Select a year to view the Data:', 2010, 2023, 2010)
year_df = df[df['From Date'].dt.year == selected_year]

st.write(f'## Data for {selected_year}')

st.subheader("Carbon Monoxide Concentrations Over Time")
# Carbon Monoxide Chart
if 'CO (mg/m3)' in year_df.columns and not year_df['CO (mg/m3)'].isna().all():
    max_co_value = year_df['CO (mg/m3)'].max()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=year_df['From Date'],
                   y=year_df['CO (mg/m3)'],
                   mode='lines',
                   name='Carbon Monoxide (mg/m3)',
                   line=dict(color='black')))

    fig.update_layout(
        shapes=[
            dict(type="rect",
                 xref="paper",
                 yref="y",
                 x0=0,
                 x1=1,
                 y0=0,
                 y1=GOOD_CO_MAX,
                 fillcolor="green",
                 opacity=0.5,
                 layer="below",
                 line_width=0,
                 name="Good"),
            dict(type="rect",
                 xref="paper",
                 yref="y",
                 x0=0,
                 x1=1,
                 y0=GOOD_CO_MAX,
                 y1=MODERATE_CO_MAX,
                 fillcolor="yellow",
                 opacity=0.5,
                 layer="below",
                 line_width=0,
                 name="Moderate"),
            dict(type="rect",
                 xref="paper",
                 yref="y",
                 x0=0,
                 x1=1,
                 y0=MODERATE_CO_MAX,
                 y1=max(SEVERE_CO_MAX, max_co_value),
                 fillcolor="red",
                 opacity=0.5,
                 layer="below",
                 line_width=0,
                 name="Severe")
        ],
        yaxis=dict(range=[0, max(SEVERE_CO_MAX, max_co_value)]),
        yaxis_title='CO (mg/m3)',
        xaxis_title='Date',
    )

    # Add legend manually
    for color, label in zip(['green', 'yellow', 'red'], [
            'Good (0 - 1 mg/m3)', 'Moderate (1 - 5 mg/m3)',
            'Severe (5 - 9 mg/m3)'
    ]):
        fig.add_trace(
            go.Scatter(x=[None],
                       y=[None],
                       mode='markers',
                       marker=dict(size=10, color=color),
                       legendgroup=label,
                       showlegend=True,
                       name=label))

    st.plotly_chart(fig)
else:
    st.write('No data available for Carbon Monoxide in this year.')

st.subheader("Gas Concentrations")
if not year_df.empty:
    gas_columns = {
        'NO (ug/m3)': 'Nitric Oxide (ug/m3)',
        'NO2 (ug/m3)': 'Nitrogen Dioxide (ug/m3)',
        'NOx (ppb)': 'Nitrogen Oxides (ppb)',
        'SO2 (ug/m3)': 'Sulfur Dioxide (ug/m3)',
        'CO (mg/m3)': 'Carbon Monoxide (mg/m3)',
        'Ozone (ug/m3)': 'Ozone (ug/m3)'
    }
    gas_means = {
        gas_columns[col]: year_df[col].mean()
        for col in gas_columns
        if col in year_df.columns and not year_df[col].isna().all()
    }

    if gas_means:
        fig = go.Figure(data=[
            go.Pie(labels=list(gas_means.keys()),
                   values=list(gas_means.values()))
        ])
        st.plotly_chart(fig)
    else:
        st.write('No valid gas data available for the selected year.')
else:
    st.write('No data available for the selected year.')

# PM2.5 and PM10 Concentrations
st.subheader("PM2.5 and PM10 Concentrations over Time")

# Check if year_df is not empty and contains valid data
if not year_df.empty and 'PM2.5 (ug/m3)' in year_df.columns and 'PM10 (ug/m3)' in year_df.columns:
    # Drop rows with missing values in relevant columns
    pm_df = year_df[['From Date', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)']].dropna()

    # Check if there is any data left after dropping NaNs
    if not pm_df.empty:
        # Create and display the line graph
        fig = px.line(pm_df,
                      x='From Date',
                      y=['PM2.5 (ug/m3)', 'PM10 (ug/m3)'],
                      labels={
                          'value': 'Concentration (ug/m3)',
                          'From Date': 'Date'
                      },
                      title='Time Series of PM2.5 and PM10 Levels')
        st.plotly_chart(fig)
    else:
        st.write(
            "No valid data available for PM2.5 and PM10 in the selected year.")
else:
    st.write(
        "No valid data available for PM2.5 and PM10 in the selected year.")

st.subheader("Average Pollutant Levels Across the Region")

# Check if year_df is not empty
if not year_df.empty:
    # Define the pollutant columns to consider
    pollutant_columns = {
        'PM2.5 (ug/m3)': 'PM2.5',
        'PM10 (ug/m3)': 'PM10',
        'NO2 (ug/m3)': 'Nitrogen Dioxide',
        'CO (mg/m3)': 'Carbon Monoxide',
        'SO2 (ug/m3)': 'Sulfur Dioxide',
        'Ozone (ug/m3)': 'Ozone'
    }

    # Filter out only the columns present in the dataset
    valid_pollutants = {
        name: year_df[col].mean()
        for col, name in pollutant_columns.items() if col in year_df.columns
    }

    # Check if there are valid pollutants to plot
    if valid_pollutants:
        # Create a bar chart for average pollutant levels
        fig = px.bar(x=list(valid_pollutants.keys()),
                     y=list(valid_pollutants.values()),
                     labels={
                         'x': 'Pollutant',
                         'y': 'Average Level'
                     })

        st.plotly_chart(fig)
    else:
        st.write("No valid pollutant data available for the selected year.")
else:
    st.write("No data available for the selected year.")

st.subheader("Stacked Area Chart of Pollutants Over Time")

# Check if year_df is not empty
if not year_df.empty:
    # Define the pollutant columns to include in the stacked area chart
    pollutants = ['NO2 (ug/m3)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']

    # Filter out only the columns present in the dataset
    valid_pollutants = {
        col: year_df[col]
        for col in pollutants if col in year_df.columns
    }

    # Check if there are valid pollutants to plot
    if valid_pollutants:
        # Prepare DataFrame for Plotly
        stacked_area_df = year_df[['From Date'] +
                                  list(valid_pollutants.keys())].dropna()
        stacked_area_df = stacked_area_df.set_index('From Date')

        # Create stacked area chart
        fig = px.area(stacked_area_df,
                      labels={
                          'value': 'Concentration',
                          'From Date': 'Date'
                      })

        st.plotly_chart(fig)
    else:
        st.write("No valid pollutant data available for the selected year.")
else:
    st.write("No data available for the selected year.")

st.subheader("Carbon Monoxide Levels and Predictions for 2024")

# Forecasting
if 'CO (mg/m3)' in year_df.columns and selected_year==2023:
    # Prepare data
    time_series = year_df.set_index('From Date')['CO (mg/m3)'].dropna()
    data = year_df['CO (mg/m3)']
    # Fit ARIMA model
    model = ARIMA(time_series, order=(5,1,0))
    model_fit = model.fit()

    # Forecast
    forecast = model_fit.forecast(steps=30)  # Forecast next 30 days

    # Add random noise
    np.random.seed(0)  # For reproducibility
    noise = np.random.normal(scale=0.1, size=forecast.shape)
    forecast_with_noise = forecast + noise

    # Create date range for future dates
    last_date = time_series.index[-1]
    today = pd.Timestamp.today().normalize()  # Normalize to remove time component
    if today <= last_date:
        today = last_date + pd.Timedelta(days=1)  # Start forecast from the next day if today is before or equal to last_date

    future_dates = pd.date_range(start=today, periods=30)

    # Create DataFrame for plotting forecast only
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'CO (mg/m3)': forecast_with_noise
    })

    # Plot
    fig = px.line(forecast_df, x='Date', y='CO (mg/m3)', title='Forecasted CO Levels', labels={'Date': 'Date', 'CO (mg/m3)': 'CO (mg/m3)'})
    fig.update_traces(line=dict(dash='dash'), name='Forecasted CO Levels')  # Dashed line for forecasted data

    fig.update_layout(showlegend=True)

    st.plotly_chart(fig)
    completion = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[
        {"role": "system", "content": "You are an experienced data analyst, and will use my existing data to come up with a health impact analysis based on the current Carbon Monoxid data."},
        {"role": "user", "content": f"Give me a health impact analysis based on this data: {year_df}"}
      ]
    )
    st.write("Health Impact Analysis")
    st.markdown(completion.choices[0].message.content)

else:
    st.write("No valid data available for CO in this year.")
