from fastapi import status


class FSException(Exception):
    status_code: int = status.HTTP_422_UNPROCESSABLE_ENTITY
    error_code: str
    error_message: str

    @property
    def detail(self):
        return {
            "error_code": self.error_code,
            "error_message": self.error_message,
            "status_code": self.status_code,
        }