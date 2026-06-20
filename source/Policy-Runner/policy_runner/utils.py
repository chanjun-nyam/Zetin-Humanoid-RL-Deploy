from __future__ import annotations

from collections.abc import Sequence
from typing import Tuple

import torch as th



class HistoryBuffer:
    """History tensor buffer.
    """


    def __init__(
            self,
            shape: Tuple[int],
            zip_dims: Tuple[int],
            n_history: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int]): Shape of data tensor.
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_history (int): Length of history.
            device (th.device): Tensor device.
            dtype (th.dtype): Tensor data type.
            value (th.Tensor, optional): Initial value for buffer. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.shape = shape
        self.zip_dims = zip_dims
        self.zip_shape = tuple([s if d not in zip_dims else 1 for d, s in enumerate(shape)])
        self.n_history = n_history
        self.device = device
        self.dtype = dtype

        self._buff = th.zeros(
            size=(n_history, *shape),
            device=device,
            dtype=dtype,
        )

        # reset buffer
        self.ALL_INDICES = th.ones(size=self.zip_shape, dtype=th.bool, device=device)
        self.reset(self.ALL_INDICES, value)


    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        zip_dims: Tuple[int],
        n_history: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> HistoryBuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_history (int): Length of history.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `buffer`. It must broadcastable to `buffer`. Defaults to 0.

        Returns:
            HistoryBuffer: Instance of initialized `HistoryBuffer`.
        """
        return HistoryBuffer(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_history=n_history,
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )


    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): Data tensor.
        """
        self._buff.copy_(self._buff.roll(shifts=1, dims=0))
        self._buff[0,...] = data


    def reset(self, indices: th.Tensor | Tuple[Sequence[int]] | Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            indices (th.Tensor | Tuple[Sequence[int]] | Sequence[int]): Indices to reset. It can be either boolen mask tensor or tuple of indices or just indices when length of tuple is 1.
            value (th.Tensor, optional): Reset value for buffer.
        """
        if isinstance(indices, th.Tensor) and indices.dtype != th.bool:
            indices = (indices,)

        if isinstance(indices, th.Tensor):
            indices = indices.view(*self.zip_shape)
            self._buff.copy_(th.where(indices, value, self._buff))

        elif isinstance(indices, tuple):
            indices = tuple([i if d not in self.zip_dims else slice(None) for d, i in enumerate(indices)])
            self._buff[:,*indices] = value

        else:
            raise ValueError(f'Not supported type for `indices`: {type(indices)}')


    @property
    def buff(self):
        """Main buffer. Shape is (n_history, *shape).
        """
        return self._buff



class SMABuffer:
    """Simple moving average buffer implemented on pytorch tensor.
    """


    def __init__(
            self,
            shape: Tuple[int],
            zip_dims: Tuple[int],
            n_window: int,
            device: th.device,
            dtype: th.dtype,
            value: th.Tensor = 0,
        ):
        """Initialize the buffer.

        Args:
            shape (Tuple[int]): Shape of data tensor.
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_window (int): Size of window.
            device (th.device): Device of data tensor.
            dtype (th.dtype): Data type of data tensor.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.
        """
        self.shape = shape
        self.zip_dims = zip_dims
        self.zip_shape = tuple([s if d not in zip_dims else 1 for d, s in enumerate(shape)])
        self.n_window = n_window
        self.device = device
        self.dtype = dtype

        self._buff = th.zeros(
            size=(n_window, *shape),
            device=device,
            dtype=dtype,
        )
        self._buff_len = th.zeros(
            size=self.zip_shape,
            device=device,
            dtype=th.int64,
        )
        self._sma = th.zeros(
            size=shape,
            device=device,
            dtype=dtype,
        )
        self.ptr = 0
        # index where new data must be push in (where old data must be pop out)

        self.ALL_INDICES = th.ones(size=self.zip_shape, dtype=th.bool, device=device)
        self.reset(self.ALL_INDICES, value)


    @classmethod
    def init_like(
        cls,
        data: th.Tensor,
        zip_dims: Tuple[int],
        n_window: int,
        device: th.device = None,
        dtype: th.dtype = None,
        value: th.Tensor = 0,
    ) -> SMABuffer:
        """Initialize the buffer with reference tensor.

        Args:
            data (th.Tensor): Reference tensor. Shape is (n_env, n_dim).
            zip_dims(Tuple[int]): Dimensions to zip together.
            n_window (int): Size of window.
            device (th.device, optional): Tensor device. Defaults to None.
            dtype (th.dtype, optional): Tensor data type. Defaults to None.
            value (th.Tensor, optional): Initial value for `sma`. It must broadcastable to `buffer`. Defaults to 0.

        Returns:
            SMABuffer: Instance of initialized `SMABuffer`.
        """
        return SMABuffer(
            shape=tuple(data.shape),
            zip_dims=zip_dims,
            n_window=n_window,
            device=data.device if device is None else device,
            dtype=data.dtype if dtype is None else dtype,
            value=value,
        )


    def _update_sma(self):
        self._sma.copy_(self._buff.sum(dim=0) / self._buff_len.clip(min=1))


    def update(self, data: th.Tensor):
        """Update the buffer with new data.

        Args:
            data (th.Tensor): Data tensor.
        """
        self._buff[self.ptr,...] = data
        self._buff_len.add_(1).clip_(max=self.n_window)
        self.ptr = (self.ptr + 1) % self.n_window

        self._update_sma()


    def reset(self, indices: th.Tensor | Tuple[Sequence[int]] | Sequence[int], value: th.Tensor = 0):
        """Reset the buffer.

        Args:
            indices (th.Tensor | Tuple[Sequence[int]] | Sequence[int]): Indices to reset. It can be either boolen mask tensor or tuple of indices or just indices when length of tuple is 1.
            value (th.Tensor, optional): Reset value for buffer.
        """
        if isinstance(indices, th.Tensor) and indices.dtype != th.bool:
            indices = (indices,)

        if isinstance(indices, th.Tensor):
            indices = indices.view(*self.zip_shape)

            self._buff_len.masked_fill_(indices, 0)
            self._buff.masked_fill_(indices, 0.0)
            ptr_slot = self._buff[self.ptr, ...]
            ptr_slot.copy_(th.where(indices, value, ptr_slot))

        elif isinstance(indices, tuple):
            indices = tuple([i if d not in self.zip_dims else slice(None) for d, i in enumerate(indices)])

            self._buff_len[indices] = 0
            self._buff[:,*indices] = 0.0
            self._buff[self.ptr,...][indices] = value

        else:
            raise ValueError(f'Not supported type for `indices`: {type(indices)}')

        self._update_sma()


    @property
    def sma(self):
        """SMA value.
        """
        return self._sma
