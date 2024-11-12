from pydantic import BaseModel

class StudentData(BaseModel):
    ge: str
    cst: str
    tnp: str
    twp: str
    iap: str
    arr: str
    ls: str
    as_: str  # renamed "as" to "as_" to avoid using a reserved keyword
    fmi: str
    fs: str
    fq: str
    mq: int
    fo: str
    mo: str
    nf: str
    sh: str
    ss: str
    me: str
    tt: str
    atd: str