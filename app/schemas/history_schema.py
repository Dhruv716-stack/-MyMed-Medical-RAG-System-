from pydantic import BaseModel


class HistoryResponse(BaseModel):

    history: str