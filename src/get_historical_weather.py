import requests
from pydantic.v1 import BaseModel, Field
import datetime
import plotly.express as px
import pandas as pd
from langchain.tools import StructuredTool
import plotly.graph_objects as go

# Define the input schema
class OpenMeteoHistoryInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")
    start_date: str = Field(..., description="Start date of the historical weather period in format YYYY-MM-DD")
    end_date: str = Field(..., description="End date of the historical weather period in format YYYY-MM-DD")
        
class HistoricalTemperature:
    def __init__(self):
        self.fig = None
    def get_historical_temperature(self, latitude: float, longitude: float, start_date: str, end_date:str) -> dict:
        """Fetch historical temperatures for given coordinates across a given time period."""

        BASE_URL = "https://archive-api.open-meteo.com/v1/era5"

        # Parameters for the request
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'temperature_2m',
        }

        # Make the request
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            results = response.json()
        else:
            raise Exception(f"API Request failed with status code: {response.status_code}")

        current_utc_time = datetime.datetime.utcnow()
        time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in results['hourly']['time']]
        temperature_list = results['hourly']['temperature_2m']

        closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
        current_temperature = temperature_list[closest_time_index]
        df = pd.DataFrame({"time":results["hourly"]["time"], "temperature": results["hourly"]["temperature_2m"]})

        fig = go.Figure()
        for col in df.columns:
            if col == "time":
                continue
            fig.add_trace(go.Scatter(x=df.time, y=df[col], name=col))

        self.fig = fig
        max_temp = df.temperature.max()
        max_date = df.iloc[df["temperature"].idxmax()].time

        min_temp = df.temperature.min()
        min_date = df.iloc[df["temperature"].idxmax()].time
        summary_string = f"The maximum temperature in the given period is {max_temp} and occured on the {max_date}. "
        summary_string += f"The minimum temperature in the given period is {min_temp} and occured on the {min_date}."

        return summary_string