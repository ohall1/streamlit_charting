from langchain.tools import tool


@tool
def centigrade_to_fahrenheit(centigrade: float) -> str:
    """Convert temperature from degrees centigrade to fahrenheit.."""
    return f"{centigrade}°C is {centigrade*9/5+32}°F"
