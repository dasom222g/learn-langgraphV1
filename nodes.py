"""
LangGraph의  실행 단위인 'Node'들만 모아둔 모듈
"""

from email import message
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from react import tools, llm_with_tools

load_dotenv()


SYSTEM_MESSAGE = """
너는 실시간 정보를 정확히 찾아내는 분석가야. 
정보를 한 번에 찾지 못하면 검색어를 구체적으로 바꿔서 다시 시도해. 
반드시 숫자를 찾아낸 뒤에야 triple 도구를 사용해
"""

# 첫번째 노드
# 모든 노드는 state를 받아오고, state를 반환
def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    에이전트의 '추론(Reasoning)' 노드 함수입니다.
    현재 장부(State)의 메시지 이력을 읽어 LLM에게 전달하고, 
    LLM의 판단(답변 또는 도구 호출)을 다시 장부에 추가합니다.
    """
    # print(state)

    messages = [
      {
        "role": "system",
        "content": SYSTEM_MESSAGE
      },
      *state["messages"]
    ]
    response = llm_with_tools.invoke(input=messages)

    return { "messages": [response]}


# 두번째 노드
# ToolNode: 실제 특정 함수를 호출해주는 Acting단계로 최신 state의 tool_calls속성에 값이 있을때 ToolNode로 가서 실행
tool_node = ToolNode(tools)
