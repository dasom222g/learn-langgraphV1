"""
전체 워크플로우(Graph)를 조립하고 실행하는 엔트리 포인트
"""

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from nodes import run_agent_reasoning, tool_node
from langgraph.graph import StateGraph, END, MessagesState

load_dotenv()

AGENT_REASON = "agent_reason"  # 추론하는 노드
ACT = "act"   # 행동하여 Tool Calling하는 노드

def should_continue(state: MessagesState) -> str:
    """
    [조건부 로직] State를 확인하여 다음 목적지 노드를 리턴하는 '교통 제어' 함수입니다.
    """
    if state["messages"][-1].tool_calls:
        # 최신 state에 tool 명세서 있는경우 (추론 결과 툴 사용 필요하다고 판단)
        return ACT
    # 함수 사용 필요없는 경우
    return END


def view_graph(app: CompiledStateGraph, flow: StateGraph) -> None:
    # 흐름도를 이미지로 저장
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")


def set_flow() -> CompiledStateGraph:
    # 노드를 연결하는 그래프 만들기
    
    # 그래프 초기화
    flow = StateGraph(MessagesState)

    # 각 노드 등록
    flow.add_node(
        AGENT_REASON,      # 노드 이름 지정
        run_agent_reasoning   # 실행될 노드 함수
    )  # 두뇌(LLM)등록

    flow.add_node(ACT, tool_node)   # 손발(Tool) 등록

    # 진입점 설정: 그래프가 시작될 때 가장 먼저 시작할 노드를 지정
    flow.set_entry_point(AGENT_REASON)

    # 조건부로 노드 잇기
    # AGENT_REASON 노드가 끝나면 'should_continue' 함수를 실행해 결과값에 따라 길을 갈라줌
    # should_continue 결과값이 "act"면 ACT노드 실행, "__end__"면 __end__실행
    flow.add_conditional_edges(
        AGENT_REASON,        # 출발지 노드
        should_continue,     # 판단 함수 (Router)
        { END: END, ACT: ACT }   # 목적지 지도 (Mapping)
    )

    flow.add_edge(ACT, AGENT_REASON)  # 무조건 ACT -> AGENT_REASON (실선)

    # 설계도를 실제 실행 가능한 '앱'으로 변환
    app = flow.compile()

    # 그래프 시각화
    view_graph(app, flow)

    return app;
    

if __name__ == "__main__":
    app = set_flow()
    response = app.invoke({"messages": [HumanMessage(content="실시간 서울의 기온은 몇 도야? 온도를 알려주고, 그 값에 3을 곱해줘.")]})
    
    print("🔥 결과: \n", response["messages"][-1].content)
