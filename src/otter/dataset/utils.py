import re


def extract_code(response: str) -> str:
    """从 LLM response 中提取 Python 代码块。

    优先匹配 ```python ... ``` 包裹的代码块，
    如果没有匹配到则返回原始文本（去除首尾空白）。
    """
    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()
