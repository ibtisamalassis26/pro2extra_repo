from datetime import datetime, timedelta
from langchain_core.tools import tool

@tool
def get_current_datetime() -> str:
    """Returns the current date and time in YYYY-MM-DD format."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate_future_or_past_date(days: int) -> str:
    """Calculates the date a specific number of days in the future (positive number) or past (negative number) from today."""
    target_date = datetime.now() + timedelta(days=days)
    return target_date.strftime("%Y-%m-%d (%A)")

@tool
def calculate_days_until(target_date_str: str) -> str:
    """Calculates how many days remain until a target date formatted as YYYY-MM-DD."""
    try:
        target_date = datetime.strptime(target_date_str, "%Y-%m-%d").date()
        today = datetime.now().date()
        delta = (target_date - today).days
        return f"{delta} days remaining until {target_date_str}"
    except ValueError:
        return "Invalid date format. Please provide the target date in YYYY-MM-DD format."

@tool
def calculator(expression: str) -> str:
    """Evaluates mathematical expressions safely."""
    try:
        allowed_chars = "0123456789+-*/(). "
        if all(char in allowed_chars for char in expression):
            return str(eval(expression))
        else:
            return "Invalid mathematical expression."
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"