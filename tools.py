from langchain_core.tools import tool
from datetime import datetime
import ast
import operator


@tool
def get_current_datetime() -> str:
    """Returns the current date and time formatted as YYYY-MM-DD HH:MM:SS.
    Use this tool whenever you need to know today's date or the current year, month, or day.
    """
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")


@tool
def calculator(expression: str) -> str:
    """Evaluates a mathematical expression and returns the result as a string.
    Examples of expression: '15 + 23', '100 / 4', '30 * 12'.
    Use this tool when you need to perform calculations or compute differences between numbers.
    """
    try:
        allowed_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.Num):  # Compatibility for older Python AST
                return node.n
            elif isinstance(node, ast.BinOp):
                op = allowed_operators[type(node.op)]
                return op(eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):
                op = {ast.USub: operator.neg, ast.UAdd: operator.pos}[type(node.op)]
                return op(eval_node(node.operand))
            else:
                raise TypeError(f"Unsupported syntax: {type(node)}")

        parsed_expr = ast.parse(expression, mode="eval").body
        result = eval_node(parsed_expr)
        return str(result)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"