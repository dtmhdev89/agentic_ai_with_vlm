import inspect
from typing import Callable, Dict, Any
from langchain_core.tools import tool


class MathCalCulators:
    def __new__(cls):
        callable_methods: Dict[str, Callable[..., Any]] = {}

        for name, obj in inspect.getmembers(cls):
            if callable(obj) and not name.startswith("__"):
                callable_methods[name] = obj

        return callable_methods

    @staticmethod
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers"""

        return a + b

    @staticmethod
    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers"""

        return a * b

    @staticmethod
    @tool
    def divide(a: float, b: float) -> float | None:
        """Divide two numbers"""
        try:
            result = a / b
        except ZeroDivisionError as e:
            result = None
            print("Errors: ", str(e))

        return result
