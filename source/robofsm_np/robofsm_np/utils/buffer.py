from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Tuple, Union

import numpy as np


# Type aliases (kept loose for Python 3.8 runtime hint evaluation).
IndexType = Union[np.ndarray, Tuple[Sequence[int], ...], Sequence[int]]
ValueType = Union[np.ndarray, float, int]


class HistoryBuffer:
    """History tensor buffer (numpy implementation)."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        zip_dims: Tuple[int, ...],
        n_history: int,
        device: Optional[Any] = None,
        dtype: Any = np.float32,
        value: ValueType = 0,
    ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int, ...]): Shape of data array.
            zip_dims (Tuple[int, ...]): Dimensions to zip together.
            n_history (int): Length of history.
            device (Any, optional): Kept for API compatibility with the torch
                version; numpy has no device concept so it is stored but unused.
            dtype (Any): Array data type.
            value (np.ndarray | float | int, optional): Initial value for buffer.
                It must be broadcastable to `buff`. Defaults to 0.
        """
        self.shape = tuple(shape)
        self.zip_dims = tuple(zip_dims)
        self.zip_shape = tuple(
            s if d not in self.zip_dims else 1 for d, s in enumerate(self.shape)
        )
        self.n_history = n_history
        self.device = device  # unused in numpy, kept for API parity
        self.dtype = dtype

        self._buff = np.zeros((n_history, *self.shape), dtype=dtype)

        # reset buffer
        self.ALL_INDICES = np.ones(self.zip_shape, dtype=np.bool_)
        self.reset(self.ALL_INDICES, value)

    @classmethod
    def init_like(
        cls,
        data: np.ndarray,
        zip_dims: Tuple[int, ...],
        n_history: int,
        device: Optional[Any] = None,
        dtype: Any = None,
        value: ValueType = 0,
    ) -> "HistoryBuffer":
        """Initialize the buffer with a reference array.

        Args:
            data (np.ndarray): Reference array. Shape is (n_env, n_dim).
            zip_dims (Tuple[int, ...]): Dimensions to zip together.
            n_history (int): Length of history.
            device (Any, optional): Unused in numpy. Defaults to None.
            dtype (Any, optional): Array data type. Defaults to None.
            value (np.ndarray | float | int, optional): Initial value for buffer.
                Must be broadcastable to `buff`. Defaults to 0.

        Returns:
            HistoryBuffer: Instance of initialized `HistoryBuffer`.
        """
        return cls(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_history=n_history,
            device=device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )

    def update(self, data: np.ndarray):
        """Update the buffer with new data.

        Args:
            data (np.ndarray): Data array.
        """
        # np.roll returns a fresh array, so there is no aliasing with self._buff.
        self._buff[...] = np.roll(self._buff, shift=1, axis=0)
        self._buff[0, ...] = data

    def reset(self, indices: IndexType, value: ValueType = 0):
        """Reset the buffer.

        Args:
            indices (np.ndarray | Tuple[Sequence[int], ...] | Sequence[int]):
                Indices to reset. Either a boolean mask array, a tuple of index
                arrays, or a single index array (treated as a 1-tuple).
            value (np.ndarray | float | int, optional): Reset value for buffer.
        """
        if isinstance(indices, np.ndarray) and indices.dtype != np.bool_:
            indices = (indices,)

        if isinstance(indices, np.ndarray):
            indices = indices.reshape(self.zip_shape)
            self._buff[...] = np.where(indices, value, self._buff)

        elif isinstance(indices, tuple):
            indices = tuple(
                i if d not in self.zip_dims else slice(None)
                for d, i in enumerate(indices)
            )
            self._buff[(slice(None), *indices)] = value

        else:
            raise ValueError(f"Not supported type for `indices`: {type(indices)}")

    @property
    def buff(self) -> np.ndarray:
        """Main buffer. Shape is (n_history, *shape)."""
        return self._buff


class SMABuffer:
    """Simple moving average buffer implemented on numpy arrays."""

    def __init__(
        self,
        shape: Tuple[int, ...],
        zip_dims: Tuple[int, ...],
        n_window: int,
        device: Optional[Any] = None,
        dtype: Any = np.float32,
        value: ValueType = 0,
    ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int, ...]): Shape of data array.
            zip_dims (Tuple[int, ...]): Dimensions to zip together.
            n_window (int): Size of window.
            device (Any, optional): Unused in numpy, kept for API parity.
            dtype (Any): Data type of data array.
            value (np.ndarray | float | int, optional): Initial value for `sma`.
                Must be broadcastable to `buff`. Defaults to 0.
        """
        self.shape = tuple(shape)
        self.zip_dims = tuple(zip_dims)
        self.zip_shape = tuple(
            s if d not in self.zip_dims else 1 for d, s in enumerate(self.shape)
        )
        self.n_window = n_window
        self.device = device  # unused in numpy, kept for API parity
        self.dtype = dtype

        self._buff = np.zeros((n_window, *self.shape), dtype=dtype)
        self._buff_len = np.zeros(self.zip_shape, dtype=np.int64)
        self._sma = np.zeros(self.shape, dtype=dtype)
        self.ptr = 0
        # index where new data must be pushed in (where old data is popped out)

        self.ALL_INDICES = np.ones(self.zip_shape, dtype=np.bool_)
        self.reset(self.ALL_INDICES, value)

    @classmethod
    def init_like(
        cls,
        data: np.ndarray,
        zip_dims: Tuple[int, ...],
        n_window: int,
        device: Optional[Any] = None,
        dtype: Any = None,
        value: ValueType = 0,
    ) -> "SMABuffer":
        """Initialize the buffer with a reference array.

        Args:
            data (np.ndarray): Reference array. Shape is (n_env, n_dim).
            zip_dims (Tuple[int, ...]): Dimensions to zip together.
            n_window (int): Size of window.
            device (Any, optional): Unused in numpy. Defaults to None.
            dtype (Any, optional): Array data type. Defaults to None.
            value (np.ndarray | float | int, optional): Initial value for `sma`.
                Must be broadcastable to `buff`. Defaults to 0.

        Returns:
            SMABuffer: Instance of initialized `SMABuffer`.
        """
        return cls(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_window=n_window,
            device=device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )

    def _update_sma(self):
        # _buff_len broadcasts (zip_shape) against the summed buffer (shape).
        self._sma[...] = self._buff.sum(axis=0) / self._buff_len.clip(min=1)

    def update(self, data: np.ndarray):
        """Update the buffer with new data.

        Args:
            data (np.ndarray): Data array.
        """
        self._buff[self.ptr, ...] = data
        self._buff_len += 1
        np.clip(self._buff_len, None, self.n_window, out=self._buff_len)
        self.ptr = (self.ptr + 1) % self.n_window

        self._update_sma()

    def reset(self, indices: IndexType, value: ValueType = 0):
        """Reset the buffer.

        Args:
            indices (np.ndarray | Tuple[Sequence[int], ...] | Sequence[int]):
                Indices to reset. Either a boolean mask array, a tuple of index
                arrays, or a single index array (treated as a 1-tuple).
            value (np.ndarray | float | int, optional): Reset value for buffer.
        """
        if isinstance(indices, np.ndarray) and indices.dtype != np.bool_:
            indices = (indices,)

        if isinstance(indices, np.ndarray):
            indices = indices.reshape(self.zip_shape)

            # mask has the same shape as _buff_len -> direct boolean fill
            self._buff_len[indices] = 0
            # mask (zip_shape) broadcasts against _buff (n_window, *shape)
            self._buff[...] = np.where(indices, 0.0, self._buff)

            # integer indexing returns a view, so writing to it edits _buff
            ptr_slot = self._buff[self.ptr, ...]
            ptr_slot[...] = np.where(indices, value, ptr_slot)

        elif isinstance(indices, tuple):
            indices = tuple(
                i if d not in self.zip_dims else slice(None)
                for d, i in enumerate(indices)
            )

            self._buff_len[indices] = 0
            self._buff[(slice(None), *indices)] = 0.0
            self._buff[self.ptr, ...][indices] = value

        else:
            raise ValueError(f"Not supported type for `indices`: {type(indices)}")

        self._update_sma()

    @property
    def sma(self) -> np.ndarray:
        """SMA value."""
        return self._sma
