"""
에이전트가 사용할 도구(Tools)와 추론 엔진(LLM)의 사양을 정의하고 바인딩
에이전트의 핵심 지능 아키텍처인 ReAct(Reason + Act)의 재료들만 모아둔 모듈
"""

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


load_dotenv()

@tool
def triple(value: int) -> int:
    """
    입력값을 3배로 만들어서 돌려주는 함수.

    Args:
        value (int): 3배로 만들 숫자.

    Returns:
        int: 3배가 된 숫자.
    """
    return value * 3


tools = [triple, TavilySearch(max_results=5)]


llm = ChatOpenAI(model="gpt-4o")
"""
# bind_tools(tools)

LLM에게 사용할 수 있는 도구들의 '설명서(Schema)'를 강제로 주입 (@tools로 등록된 함수의 이름/설명/Type힌트를 Json Scheme으로 변환하여 LLM에게 전달)
이 과정을 통해 LLM은 단순 텍스트 답변 대신, 상황에 맞춰 
도구의 이름과 필요한 파라미터(JSON)를 반환하는 '추론 엔진'으로 업그레이드 ex) tool_calls: [{ name: "get_wether", input: "location" }]
⭐️ 참고: 실제로 도구를 실행하는 것이 아니라, 실행할 '의사'를 결정하게 만드는 바인딩 단계
"""

llm_with_tools = llm.bind_tools(tools)    # 사용 가능한 "도구 설명서" 주입
