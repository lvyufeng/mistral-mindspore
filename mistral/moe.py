import dataclasses
from typing import List

import mindspore
from mindspore import ops, nn
from simple_parsing.helpers import Serializable


@dataclasses.dataclass
class MoeArgs(Serializable):
    num_experts: int
    num_experts_per_tok: int


class MoeLayer(nn.Cell):
    def __init__(self, experts: List[nn.Cell], gate: nn.Cell, moe_args: MoeArgs):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.CellList(experts)
        self.gate = gate
        self.args = moe_args

    def construct(self, inputs: mindspore.Tensor):
        gate_logits = self.gate(inputs)
        weights, selected_experts = ops.topk(gate_logits, self.args.num_experts_per_tok)
        weights = ops.softmax(weights, axis=1, dtype=mindspore.float32).to(inputs.dtype)
        results = ops.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = ops.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        return results
