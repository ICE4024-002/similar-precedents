from pydantic import BaseModel

class Precedent(BaseModel):
    판례정보일련번호: int
    사건명: str
    사건번호: str
    선고일자: int
    선고: str
    법원명: str
    사건종류명: str
    판결유형: str
    판시사항: str
    판결요지: str
    참조조문: str
    참조판례: str
    전문: str

