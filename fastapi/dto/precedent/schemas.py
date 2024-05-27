from typing import Optional
from pydantic import BaseModel

class Precedent(BaseModel):
    판례정보일련번호: Optional[int]
    사건명: Optional[str]
    사건번호: Optional[str]
    선고일자: Optional[int]
    선고: Optional[str]
    법원명: Optional[str]
    사건종류명: Optional[str]
    판결유형: Optional[str]
    판시사항: Optional[str]
    판결요지: Optional[str]
    참조조문: Optional[str]
    참조판례: Optional[str]
    전문: Optional[str]
    