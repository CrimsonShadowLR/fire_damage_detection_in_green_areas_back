from typing import Iterable

from pydantic import BaseModel


class CustomBaseModel(BaseModel):
    @classmethod
    def from_orms(cls, orm_objects: Iterable):
        """
        Converts ORM (Object-Relational Mapping) objects into Pydantic schemas and
        returns them in a list.
        """
        return [cls.from_orm(orm_object) for orm_object in orm_objects]
