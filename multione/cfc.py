# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from sklearn.externals.array_api_compat.numpy import float32
import torch
from torch import nn
from typing import List, Optional, Union
import ncps
from cfc_cell import CfCCell #, WiredCfCCell
from ncps.torch.lstm import LSTMCell


class CfC(nn.Module):
    def __init__(
        self,
        input_size: int,    # number of input features (ls bands+modis ndvi+geom temp)
        timeless_input_size: int,    # number of features that are constant in time (dtm derivatrives, geom temp min/max)
        num_hidden_units: int,
        output_size: int,   # number of output features
        #return_sequences: bool = True,
        #batch_first: bool = True,  # Always
        #mixed_memory: bool = False, # True
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_layers: Optional[int | List[int]] = None,        
        backbone_dropout: Optional[float] = None,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.

        Examples::

             >>> from ncps.torch import CfC
             >>>
             >>> rnn = CfC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)

        :param input_size: Number of input features
        :param units: Number of hidden units
        :param proj_size: If not None, the output of the RNN will be projected to a tensor with dimension proj_size (i.e., an implict linear output layer)
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        """

        super(CfC, self).__init__()
        self.input_size = input_size
        self.hidden_size = num_hidden_units
        self.proj_size = output_size
        self.timeless_input_size = timeless_input_size
        #self.batch_first = batch_first
        #self.return_sequences = return_sequences

        

        #self.wired_false = True
        backbone_layers = [128] if backbone_layers is None else backbone_layers
        backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout    # type: ignore
        self.state_size = num_hidden_units
        #self.output_size = self.state_size  # ????
        self.rnn_cell = CfCCell(
            input_size,
            timeless_input_size,
            self.hidden_size,
            mode,
            activation,            
            backbone_layers,
            backbone_dropout,
        )
        #self.use_mixed = mixed_memory
        #if self.use_mixed:
        self.lstm = LSTMCell(input_size, self.state_size)

        # if output_size is None:
        #     self.fc = nn.Identity()
        # else:
        self.fc = nn.Linear(self.state_size, self.proj_size)

    def forward(self, input, input_timeless=None, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        #is_batched = input.dim() == 3  # always will be
        batch_dim = 0 #if self.batch_first else 1
        seq_dim = 1 #if self.batch_first else 0

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            # if self.use_mixed and isinstance(hx, torch.Tensor):
            #     raise RuntimeError(
            #         "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
            #     )
            h_state, c_state = hx if self.use_mixed else (hx, None)
            
            if h_state.dim() != 2:
                msg = (
                    "For batched 2-D input, hx and cx should "
                    f"also be 2-D but got ({h_state.dim()}-D) tensor"
                )
                raise RuntimeError(msg)            

        output_sequence = []
        for t in range(seq_len):
            inputs = input[:, t]
            ts = 1.0 if timespans is None else timespans[:, t] #.squeeze()

            #if self.use_mixed:
            h_state, c_state = self.lstm(inputs, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts)
            # if self.return_sequences:
            #     output_sequence.append(self.fc(h_out))

        # if self.return_sequences:
        #     stack_dim = 1 if self.batch_first else 0
        #     readout = torch.stack(output_sequence, dim=stack_dim)
        # else:
        readout = self.fc(h_out) #type: ignore
        hx = (h_state, c_state) if self.use_mixed else h_state

        return readout, hx