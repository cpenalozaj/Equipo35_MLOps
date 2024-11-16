from pydantic import BaseModel
from typing import Literal

class StudentData(BaseModel):
    ge: Literal["F", "M"]
    cst: Literal["G", "OBC", "MOBC", "ST", "SC"]
    tnp: Literal["Good", "Vg", "Pass", "Best"]
    twp: Literal["Good", "Vg", "Pass", "Best"]
    iap: Literal["Vg", "Good", "Pass", "Best"]
    arr: Literal["Y", "N"]
    ls: Literal["V", "T"]
    as_: Literal["Paid", "Free"]   # renamed "as" to "as_" to avoid using a reserved keyword
    fmi: Literal["Medium", "Low", "Am", "High", "Vh"]
    fs: Literal["Average", "Small", "Large"]
    fq: Literal["Um", "12", "10", "Il", "Degree", "Pg"]
    mq: int
    fo: Literal["Farmer", "Service", "Business", "Others", "Retired"]
    mo: Literal["Housewife", "Service", "Business", "Retired", "Others"]    
    nf: Literal["Small", "Average", "Large"]
    sh: Literal["Poor", "Average", "Good"]
    ss: Literal["Govt", "Private"]
    me: Literal["Asm", "Eng", "Hin", "Ben"]
    tt: Literal["Small", "Average", "Large"]
    atd: Literal["Good", "Average", "Poor"]