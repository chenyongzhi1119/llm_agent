"""Weather query tool - using wttr.in free API"""

import httpx

from src.tools.base import Tool


class WeatherTool(Tool):
    name = "weather"
    description = "Get current weather for a city. Returns temp, condition, humidity, wind."
    parameters = {
        "city": {"description": "city name e.g. 'Beijing'", "required": True},
    }

    def execute(self, **kwargs) -> str:
        city = kwargs.get("city", "")
        if not city:
            return "Error: please provide a city name"

        try:
            url = f"https://wttr.in/{city}?format=j1"
            response = httpx.get(url, timeout=10, follow_redirects=True)
            response.raise_for_status()
            data = response.json().get("data", response.json())

            current = data.get("current_condition", [{}])[0]
            area = data.get("nearest_area", [{}])[0]

            city_name = area.get("areaName", [{}])[0].get("value", city) if area else city
            country = area.get("country", [{}])[0].get("value", "") if area else ""
            temp_c = current.get("temp_C", "N/A")
            feels_like = current.get("FeelsLikeC", "N/A")
            desc = current.get("weatherDesc", [{}])[0].get("value", "N/A")
            humidity = current.get("humidity", "N/A")
            wind_speed = current.get("windspeedKmph", "N/A")
            wind_dir = current.get("winddir16Point", "N/A")

            return (
                f"Weather for {city_name}, {country}:\n"
                f"  Condition: {desc}\n"
                f"  Temperature: {temp_c}C (feels like {feels_like}C)\n"
                f"  Humidity: {humidity}%\n"
                f"  Wind: {wind_speed} km/h {wind_dir}"
            )

        except Exception as e:
            return f"Weather query error: {e}"
