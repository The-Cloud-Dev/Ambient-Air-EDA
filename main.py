import streamlit as st
import pandas as pd
import base64
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
import plotly.express as px
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
import mailtrap as mt
from mailtrap import Mail, Address, Attachment, Disposition, MailtrapClient
from openai import OpenAI
import requests
import numpy as np
import re
import json
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AQICN_API_KEY = os.getenv("AQICN_API_KEY")
MT_API_KEY = os.getenv("MT_API_KEY")
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

st.title('Air Quality Data Viewer')

abbreviation_mapping = {
    name: abbr
    for abbr, name in state_expanded_forms.items()
}
st.sidebar.title("Menu")
menu = st.sidebar.radio("Navigation", ["Graph", "AI Analysis", "Mailing"])
if menu =="Graph" or menu == "AI Analysis":
    selected_name = st.selectbox("Select a State or UT",
         list(state_expanded_forms.values()),index=6)
    selected_abbreviation = abbreviation_mapping[selected_name]
    file_path = f"Datasets/{selected_abbreviation}.csv"
    df = pd.read_csv(file_path)
    df['From Date'] = pd.to_datetime(df['From Date'], format='%Y-%m-%d')
    selected_year = st.slider('Select a year to view the Data:', 2010, 2023, 2023)
    year_df = df[df['From Date'].dt.year == selected_year]

if menu == "Graph":
    st.write(f'## Data for {selected_year}')
    st.sidebar.subheader("Graphs")
    graph_option = st.sidebar.radio("", ["Carbon Monoxide", "Gas Concentrations", "PM2.5 and PM10", "Average Pollutant Levels", "Stacked Area Chart"])

    if graph_option == "Carbon Monoxide":
        st.subheader("Carbon Monoxide Concentrations Over Time")
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
    elif graph_option == "Gas Concentrations":
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
        
        
    elif graph_option == "PM2.5 and PM10":
        st.subheader("PM2.5 and PM10 Concentrations over Time")
        if not year_df.empty and 'PM2.5 (ug/m3)' in year_df.columns and 'PM10 (ug/m3)' in year_df.columns:
            pm_df = year_df[['From Date', 'PM2.5 (ug/m3)', 'PM10 (ug/m3)']].dropna()
            if not pm_df.empty:
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
    elif graph_option == "Average Pollutant Levels":
        st.subheader("Average Pollutant Levels Across the Region")
        
        if not year_df.empty:
            pollutant_columns = {
                'PM2.5 (ug/m3)': 'PM2.5',
                'PM10 (ug/m3)': 'PM10',
                'NO2 (ug/m3)': 'Nitrogen Dioxide',
                'CO (mg/m3)': 'Carbon Monoxide',
                'SO2 (ug/m3)': 'Sulfur Dioxide',
                'Ozone (ug/m3)': 'Ozone'
            }
        
            valid_pollutants = {
                name: year_df[col].mean()
                for col, name in pollutant_columns.items() if col in year_df.columns
            }
        
            if valid_pollutants:
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
    elif graph_option == "Stacked Area Chart":
        st.subheader("Stacked Area Chart of Pollutants Over Time")
        
        if not year_df.empty:
            pollutants = ['NO2 (ug/m3)', 'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']
        
            valid_pollutants = {
                col: year_df[col]
                for col in pollutants if col in year_df.columns
            }
            if valid_pollutants:
                stacked_area_df = year_df[['From Date'] +
                                          list(valid_pollutants.keys())].dropna()
                stacked_area_df = stacked_area_df.set_index('From Date')
        
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
if menu =="AI Analysis":
    st.subheader("Carbon Monoxide Levels and Predictions for 2024")
    if 'CO (mg/m3)' in year_df.columns and selected_year==2023:
        time_series = year_df.set_index('From Date')['CO (mg/m3)'].dropna()
        data = year_df['CO (mg/m3)']
        
        model = ARIMA(time_series, order=(5,1,0))
        model_fit = model.fit()
    
        forecast = model_fit.forecast(steps=30)  
    
        np.random.seed(0)  
        noise = np.random.normal(scale=0.1, size=forecast.shape)
        forecast_with_noise = forecast + noise
        last_date = time_series.index[-1]
        today = pd.Timestamp.today().normalize() 
        if today <= last_date:
            today = last_date + pd.Timedelta(days=1) 
        future_dates = pd.date_range(start=today, periods=30)
    
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'CO (mg/m3)': forecast_with_noise
        })
        
        fig = px.line(forecast_df, x='Date', y='CO (mg/m3)', title='Forecasted CO Levels', labels={'Date': 'Date', 'CO (mg/m3)': 'CO (mg/m3)'})
        fig.update_traces(line=dict(dash='dash'), name='Forecasted CO Levels')
    
        fig.update_layout(showlegend=True)
    
        st.plotly_chart(fig)
        completion = client.chat.completions.create(
          model="gpt-4o-mini",
          messages=[
            {"role": "system", "content": "You are an experienced data analyst, and will use my existing data to come up with a health impact analysis based on the current Carbon Monoxid data."},
            {"role": "user", "content": f"I will be providing you with the data of pollutants in an Indian State over a single year. Your job is to find out, where these pollutants are coming from, what can be done to reduce them, and to generate a health impact analysis based on the data. The name of the state is {selected_name} This analysis will be targeted towards NGOs, and must include all the details mentioned above, Feel free to use the data here, but make sure you are NOT MENTIONING ANYTHING THAT IS NOT GIVEN IN THE DATAFRAME: {year_df}"}
          ]
        )
        st.subheader("Health Impact Analysis (AI Generated)")
        st.markdown(completion.choices[0].message.content)
    
    else:
        st.write("No valid data available for CO in this year.")

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(pattern, email):
        return True
    return False
if menu == "Mailing":
    st.subheader("Daily Air Quality Insights")
    
    email = st.text_input("Enter your email:", placeholder="john@doe.com")
    is_valid_email = validate_email(email)
    
    if email and not is_valid_email:
        st.error("Please enter a valid email address.")
    if st.button("Submit", disabled=not is_valid_email):
        st.success("Email validated! Ready to proceed.")
    
        def generate_aqi_graph(aqi_data):
            dates = [datetime.strptime(d['day'], '%Y-%m-%d').date() for d in aqi_data]
            values = [d['avg'] for d in aqi_data]
            plt.figure(figsize=(6, 4))
            plt.plot(dates, values, marker='o', color='blue')
            plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: datetime.fromordinal(int(x)).strftime('%d-%b')))
            plt.title('AQI Trend Over the Past Week')
            plt.xlabel('Date')
            plt.ylabel('AQI Level')
            plt.grid(False)
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
    
            return buffer.read()
            
        response = requests.get(f"https://api.waqi.info/feed/hyderabad/?token={AQICN_API_KEY}")
        data = response.json()
        aqi = data['data']['aqi']
        city = data['data']['city']['name']
        dominant_pollutant = data['data']['dominentpol']
        forecast = data['data']['forecast']['daily']['pm25'] 
        aqi_graph_image = generate_aqi_graph(forecast)
        email_content = f"""
        <!doctype html>
        <html>
          <head>
            <meta charset="UTF-8">
            <style>
              body {{
                font-family: Arial, sans-serif;
                color: #333;
              }}
              .header {{
                text-align: center;
                padding: 20px;
              }}
              .aqi-box {{
                font-size: 24px;
                font-weight: bold;
                background-color: {'#00e400' if aqi <= 50 else '#ff0000'};
                color: white;
                padding: 10px;
                text-align: center;
                margin: 20px 0;
              }}
              .section {{
                margin-bottom: 20px;
              }}
              .recommendation {{
                font-weight: bold;
                color: #ff0000;
              }}
            </style>
          </head>
          <body>
            <div class="header">
              <h1>Air Quality Update for {city}</h1>
              <p>Date: {datetime.now().strftime('%Y-%m-%d')}</p>
            </div>
            <div class="aqi-box">
              Current AQI: {aqi}
            </div>
            <div class="section">
              <h1>Dominant Pollutant: {dominant_pollutant.upper().replace('PM25', 'PM2.5')}</h1>
              <p style="font-size: 16px;">The current dominant pollutant is <strong>{dominant_pollutant.upper().replace('PM25', 'PM2.5')}</strong>, which is affecting the air quality today.</p>
            </div>
            <div class="section">
              <h1>AQI Trend</h1>
              <p style="font-size: 16px;">Below is the AQI trend over the past week:</p>
              <img src="cid:aqi_graph.png" alt="AQI Trend Graph" style="width: 100%;">
            </div>
            <div class="section">
              <h1>Health Impact & Recommendations</h1>
              <p style="font-size: 16px;" class="recommendation">For today, it's recommended to limit outdoor activities, especially for sensitive groups.</p>
            </div>
            <div class="section">
              <h1>Weather Influence</h1>
              <p style="font-size: 16px;" >Today's weather, with a temperature of {data['data']['iaqi']['t']['v']}°C, is likely to influence the air quality, potentially increasing pollution levels.</p>
            </div>
          </body>
        </html>
        """
        mail = Mail(
            sender=Address(email="ambientair@demomailtrap.com", name="Air Quality Monitor"),
            to=[Address(email=email)],
            subject=f"Air Quality Update for {city} on {datetime.now().strftime('%Y-%m-%d')}",
            html=email_content,
            attachments=[
                Attachment(
                    content=base64.b64encode(aqi_graph_image), 
                    filename="aqi_graph.png",
                    disposition=Disposition.INLINE,
                    mimetype="image/png",
                    content_id="aqi_graph.png"
                )
            ]
        )
    
        client = MailtrapClient(token=MT_API_KEY)
        client.send(mail)
