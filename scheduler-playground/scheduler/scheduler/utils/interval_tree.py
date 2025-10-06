"""Lightweight interval tree utilities used by the playground scheduler.

The original implementation represented bookings as unsorted ``list`` objects
on ``ResourceManager`` which meant that every booking/unbooking required a
linear scan followed by potentially expensive list slicing.  The helper in this
module implements a tiny interval tree tailored for scheduling workloads.  It
stores intervals in a binary-search tree ordered by start timestamp and keeps
track of the maximum ``end`` value per node so overlap checks can skip entire
subtrees.

The implementation purposefully only exposes the operations required by the
rest of the codebase: ``insert``, ``remove`` and ``overlaps``.  Each method runs
in :math:`O(\log n)` time which keeps the inner scheduling loops fast even when
hundreds of bookings have already been made.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Tuple


Interval = Tuple[datetime, datetime]


@dataclass(slots=True)
class _Node:
    start: datetime
    end: datetime
    max_end: datetime
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

    def update(self) -> None:
        """Recompute the ``max_end`` cache after mutations."""

        candidate = self.end
        if self.left is not None and self.left.max_end > candidate:
            candidate = self.left.max_end
        if self.right is not None and self.right.max_end > candidate:
            candidate = self.right.max_end
        self.max_end = candidate


class IntervalTree:
    """Simple balanced-ish interval tree used by :class:`ResourceManager`.

    The tree performs local rotations when necessary which keeps it sufficiently
    balanced for the workloads we see in the playground.  It is intentionally
    lightweight – we do not need the full generality of interval trees from
    external libraries and keeping the implementation in-tree avoids an extra
    runtime dependency.
    """

    __slots__ = ("_root",)

    def __init__(self) -> None:
        self._root: Optional[_Node] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def insert(self, start: datetime, end: datetime) -> None:
        if end <= start:
            raise ValueError("Interval end must be greater than start")
        self._root = self._insert(self._root, start, end)

    def remove(self, start: datetime, end: datetime) -> None:
        if self._root is None:
            return
        self._root = self._remove(self._root, start, end)

    def overlaps(self, start: datetime, end: datetime) -> bool:
        if self._root is None or end <= start:
            return False
        node = self._root
        while node is not None:
            if start < node.end and end > node.start:
                return True
            if node.left is not None and node.left.max_end > start:
                node = node.left
            else:
                node = node.right
        return False

    def iter(self) -> Iterator[Interval]:
        """Yield all intervals in order. Primarily used for debugging/tests."""

        yield from self._iter_nodes(self._root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _insert(self, node: Optional[_Node], start: datetime, end: datetime) -> _Node:
        if node is None:
            return _Node(start=start, end=end, max_end=end)
        if start < node.start:
            node.left = self._insert(node.left, start, end)
        else:
            node.right = self._insert(node.right, start, end)
        node.update()
        return self._rebalance(node)

    def _remove(self, node: Optional[_Node], start: datetime, end: datetime) -> Optional[_Node]:
        if node is None:
            return None
        if start == node.start and end == node.end:
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left
            successor = self._min_node(node.right)
            node.start, node.end = successor.start, successor.end
            node.right = self._remove(node.right, successor.start, successor.end)
        elif start < node.start:
            node.left = self._remove(node.left, start, end)
        else:
            node.right = self._remove(node.right, start, end)
        node.update()
        return self._rebalance(node)

    def _min_node(self, node: _Node) -> _Node:
        while node.left is not None:
            node = node.left
        return node

    def _rebalance(self, node: _Node) -> _Node:
        """Perform naive AVL-style rebalancing to keep depth manageable."""

        balance = self._height(node.left) - self._height(node.right)
        if balance > 1:
            if self._height(node.left.left) < self._height(node.left.right):
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if balance < -1:
            if self._height(node.right.right) < self._height(node.right.left):
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def _height(self, node: Optional[_Node]) -> int:
        if node is None:
            return 0
        return 1 + max(self._height(node.left), self._height(node.right))

    def _rotate_left(self, node: _Node) -> _Node:
        assert node.right is not None
        new_root = node.right
        node.right = new_root.left
        new_root.left = node
        node.update()
        new_root.update()
        return new_root

    def _rotate_right(self, node: _Node) -> _Node:
        assert node.left is not None
        new_root = node.left
        node.left = new_root.right
        new_root.right = node
        node.update()
        new_root.update()
        return new_root

    def _iter_nodes(self, node: Optional[_Node]) -> Iterator[Interval]:
        if node is None:
            return
        yield from self._iter_nodes(node.left)
        yield (node.start, node.end)
        yield from self._iter_nodes(node.right)


def iter_intervals(tree: IntervalTree) -> Iterable[Interval]:
    """Convenience wrapper used by tests to list intervals."""

    return list(tree.iter())

