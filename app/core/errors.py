class AppError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        details: dict | list | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class BadRequestError(AppError):
    def __init__(self, code: str, message: str, details: dict | list | None = None) -> None:
        super().__init__(status_code=400, code=code, message=message, details=details)


class NotFoundError(AppError):
    def __init__(self, code: str, message: str, details: dict | list | None = None) -> None:
        super().__init__(status_code=404, code=code, message=message, details=details)


class ConflictError(AppError):
    def __init__(self, code: str, message: str, details: dict | list | None = None) -> None:
        super().__init__(status_code=409, code=code, message=message, details=details)
