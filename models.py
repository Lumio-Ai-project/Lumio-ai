from typing import Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    query: str
    chat_history: Annotated[list, add_messages]
    query_rewrite: str
    context: Optional[list[str]]
    response: str