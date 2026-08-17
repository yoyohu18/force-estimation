"""Baseline model: a two-layer MLP over the flattened window.

    (B, history, 33)  --flatten-->  (B, history*33)
                      --Linear-->   (B, hidden)
                      --ReLU-->     (B, hidden)
                      --Linear-->   (B, 4)

Deliberately the smallest thing that can work. It exists to prove the pipeline
runs end to end and to give every later model something to beat, not to be
good. The reference numbers it has to clear, measured with k-nearest-neighbour
on the same task: MSE 0.0105 at history=20, and 0.01625 for predicting the
training mean and nothing else.

The output layer is plain linear — no sigmoid. foot_sensors does sit in
[0, 1.06], but a sigmoid saturates at both ends, cannot reach 1.06 at all, and
kills the gradient exactly where the interesting impact peaks are. An
unconstrained output can go negative, which is unphysical; `negative_fraction`
below reports how often it does, which is a cheap sanity check on training.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Flatten the window, one hidden layer, linear readout."""

    def __init__(self, history=20, n_features=33, hidden=64, n_out=4, dropout=0.0):
        super().__init__()  # must come first: it sets up the module registry
        self.flatten = nn.Flatten()  # (B, T, F) -> (B, T*F), leaves dim 0 alone
        self.fc1 = nn.Linear(history * n_features, hidden)
        self.act = nn.ReLU()
        # Dropout zeroes a random share of the hidden units each training step,
        # so no single unit can carry a feature on its own. nn.Module.eval()
        # turns it off, which is why validation must call model.eval().
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, n_out)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        return self.fc2(x)


def count_parameters(model):
    """Trainable parameters, and the per-layer breakdown."""
    per_layer = {n: p.numel() for n, p in model.named_parameters() if p.requires_grad}
    return sum(per_layer.values()), per_layer


def negative_fraction(pred):
    """Share of predicted forces below zero — those are physically impossible."""
    return float((pred < 0).float().mean())


if __name__ == "__main__":
    torch.manual_seed(0)

    model = MLP(history=20, hidden=64)
    print(model)

    total, per_layer = count_parameters(model)
    print(f"\ntrainable parameters: {total}")
    for name, n in per_layer.items():
        print(f"  {name:<12} {n:>7}   {100 * n / total:5.1f}%")

    x = torch.randn(256, 20, 33)
    y = model(x)
    print(f"\nforward: {tuple(x.shape)} -> {tuple(y.shape)}")
    print(f"  untrained output: mean {y.mean():+.4f}  std {y.std():.4f}")
    print(f"  (targets live around 0.2, so an untrained net is nowhere near)")

    # Without an activation, stacked Linear layers collapse into one Linear.
    a, b = nn.Linear(5, 7, bias=False), nn.Linear(7, 3, bias=False)
    z = torch.randn(4, 5)
    stacked = b(a(z))
    collapsed = z @ (b.weight @ a.weight).T
    print(f"\nno activation:  max|b(a(z)) - z @ (W2 W1)^T| = "
          f"{(stacked - collapsed).abs().max():.2e}")
    print("  two Linear layers really are one Linear layer -> ReLU is what buys depth")


class LSTMModel(nn.Module):
    """Read the window one frame at a time, predict from the final hidden state.

    Two things this buys over flattening the window into an MLP:

    Weight sharing. "How do I read one 33-dim frame" is learned once and
    applied 20 times, so the input weights are 33-wide instead of 660-wide.

    Order. A fixed permutation of the 20 frames can be absorbed into the MLP's
    first weight matrix (W1 P is just another W1), so the MLP cannot tell frame
    order at all. The recurrence can: h_t depends on h_{t-1}.

    The cell state updates as c_t = f_t * c_{t-1} + i_t * c~_t — additively, with
    no weight matrix multiplying c_{t-1}. That is what keeps the gradient alive
    across 20 steps where a plain RNN, which multiplies by W_h every step, would
    have it decay or explode geometrically.
    """

    def __init__(self, n_features=33, hidden=64, layers=1, n_out=4, dropout=0.0):
        super().__init__()
        # batch_first=True: (B, T, F). The default is (T, B, F), which silently
        # treats the batch as time and produces plausible-looking nonsense.
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        # nn.LSTM's own dropout only fires *between* stacked layers, so with
        # layers=1 it does nothing at all. This one acts on the readout.
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, n_out)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)  # out (B, T, hidden), h_n (layers, B, hidden)
        return self.fc(self.drop(out[:, -1]))  # last step has seen the whole window
