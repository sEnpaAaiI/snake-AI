import torch 
import torch.nn as nn

class SnakeModel(nn.Module):
    """
    the output is interpreted as follows
    right, left, up, down
    """

    def __init__(self,
                 input_units,
                 hidden_units,
                 output_units=4):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(input_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, output_units),
            nn.Softmax(dim=-1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.l(state)
        return x

# class SnakeModel:
#     def __init__(self,
#                  input_units,
#                  hidden_units,
#                  output_units):
#         self.l1 = np.random.randn(input_units, hidden_units)
#         self.l2 = np.random.randn(hidden_units, output_units)

#     def forward(self, state):
#         # 32 X 1, 32 X 10
#         x = state.T @ self.l1 # 1, 10
#         return x @ self.l2 # 1, 10 ... 10, 4 -> 1, 4
