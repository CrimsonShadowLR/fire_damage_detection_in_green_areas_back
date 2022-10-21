from typing import Iterable

from pydantic import BaseModel


class FSBaseModel(BaseModel):
    @classmethod
    def from_orms(cls, orm_objects: Iterable):
        """
        Converts each of the ORM Django objects from the iterable into their respective
        Pydantic schema (FastAPI -> Pydantic)
        """
        return [cls.from_orm(orm_object) for orm_object in orm_objects]
