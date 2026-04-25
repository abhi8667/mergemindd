from __future__ import annotations


class BroadcastLayer:
    def __init__(self) -> None:
        self._buffer: list[dict[str, float | int]] = []

    @property
    def buffer(self) -> list[dict[str, float | int]]:
        return list(self._buffer)

    def update(self, packets: list[dict[str, float | int]]) -> None:
        self._buffer = [dict(packet) for packet in packets]

    def receive_for(self, receiver_id: int) -> list[dict[str, float | int]]:
        return [dict(packet) for packet in self._buffer if int(packet["sender_id"]) != receiver_id]

    def clear(self) -> None:
        self._buffer = []
