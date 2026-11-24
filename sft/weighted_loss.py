"""Lower current reasoning loss without changing targets or materializing vocabulary logits."""

import os
from types import MethodType


def token_weights(labels, has_reasoning, end_thinking_id, reasoning_weight):
    weights = labels.ne(-100).float()
    for index, reasoning in enumerate(has_reasoning):
        if not reasoning:
            continue
        ends = labels[index].eq(end_thinking_id).nonzero().flatten()
        if not len(ends):
            raise ValueError("Supervised reasoning has no native closing delimiter")
        # Ending thinking remains at weight 1, along with calls and visible output.
        weights[index, :ends[0]] *= reasoning_weight
    return weights


def weighted_linear_loss(hidden, head, labels, weights, num_items=None):
    from unsloth_zoo.fused_losses.fused_linear import fused_linear_logprobs

    targets = labels[..., 1:]
    logprobs = fused_linear_logprobs(
        hidden[..., :-1, :], head.weight, targets, bias=head.bias,
        mask=targets.ne(-100), entropy_requires_grad=False,
    )
    count = targets.ne(-100).sum() if num_items is None else num_items
    return -(logprobs * weights[..., 1:]).sum() / count


def weighted_forward(self, *args, loss_weights=None, labels=None, **kwargs):
    if loss_weights is None:
        return self._unweighted_forward(*args, labels=labels, **kwargs)
    previous = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        outputs = self._unweighted_forward(*args, labels=labels, **kwargs)
    finally:
        if previous is None:
            os.environ.pop("UNSLOTH_RETURN_HIDDEN_STATES", None)
        else:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = previous
    hidden = outputs.logits
    outputs.loss = weighted_linear_loss(
        hidden, self.get_output_embeddings(), labels, loss_weights,
        kwargs.get("num_items_in_batch", kwargs.get("n_items")),
    )
    outputs.logits = hidden.new_empty(0)
    return outputs


def install_weighted_loss(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    base._unweighted_forward = base.forward
    base.forward = MethodType(weighted_forward, base)
