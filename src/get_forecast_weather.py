import requests
from pydantic.v1 import BaseModel, Field
import datetime
from langchain.tools import tool
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class OpenMeteoInput(BaseModel):
    latitude: float = Field(description="Latitude of the location to fetch weather data for")
    longitude: float = Field(description="Longitude of the location to fetch weather data for")

class ForecastWeather:
    def __init__(self) -> None:
        self.fig = None
    # @tool(args_schema=OpenMeteoInput)
    def get_forecast_weather(self, latitude: float, longitude: float) -> str:
        """Fetch forecast weather for given coordinates."""

        BASE_URL = "https://api.open-meteo.com/v1/forecast"

        # Parameters for the request
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'temperature_2m,relative_humidity_2m',
            'forecast_days': 7,
        }

        # Make the request
        response = requests.get(BASE_URL, params=params)
        
        if response.status_code == 200:
            results = response.json()
        else:
            raise Exception(f"API Request failed with status code: {response.status_code}")
        
        weather_forecast_df = pd.DataFrame(results["hourly"])
        fig = go.Figure()
        for col in weather_forecast_df.columns:
            if col == "time":
                continue
            fig.add_trace(go.Scatter(x=weather_forecast_df.time, y=weather_forecast_df[col], name=col))
        llm_return_string = "The user is being shown a chart of the forecast weather."
        llm_return_string += " It shows them the following features:\n"
        for col in weather_forecast_df.columns:
            if col == "time":
                continue
            # Column doesn't equal time
            weather_variable = " ".join(col.split("_")[:-1])
            weather_variable_max_value = weather_forecast_df[col].max()
            weather_variable_min_value = weather_forecast_df[col].min()
            llm_return_string += "In the next 7 days {} will have a high of {} and a low of {}.\n".format(
                weather_variable,
                weather_variable_max_value,
                weather_variable_min_value
            )
        self.fig = fig
        return llm_return_string
