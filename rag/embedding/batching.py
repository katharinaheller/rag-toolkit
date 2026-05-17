from typing import Generator, Iterable, List, TypeVar

T = TypeVar("T")


def batch_iter(items: Iterable[T], batch_size: int) -> Generator[List[T], None, None]:
    """Yield fixed-size batches from any iterable without materializing it."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    batch: List[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
