from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(index=True)
    identity_number: str = Field(index=True, unique=True)
    
    templates: List["IrisTemplate"] = Relationship(back_populates="user")

class IrisTemplate(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    eye_side: str  # "L" or "R"
    iris_code_hex: str  # Store binary data as hex string
    mask_hex: str
    
    user: Optional[User] = Relationship(back_populates="templates")
