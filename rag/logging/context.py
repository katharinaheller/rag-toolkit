import structlog


def bind_context(**kwargs):
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context():
    structlog.contextvars.clear_contextvars()
