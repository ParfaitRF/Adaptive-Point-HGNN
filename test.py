import geoopt.layers
import geoopt.layers.stereographic
import torch
import torch.nn as nn
import torch.optim as optim
import geoopt
from geoopt.manifolds.poincare import PoincareBall
from geoopt.layers import stereographic



class HyperbolicLinear(nn.Module):
    """A simple hyperbolic linear layer using the Poincare ball model."""
    def __init__(self, in_features, out_features, c=1.0):
        super().__init__()
        self.c = c
        self.manifold = PoincareBall(c=c)
        self.weight = geoopt.ManifoldParameter(
            torch.randn(out_features, in_features), manifold=self.manifold
        )
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        x = self.manifold.expmap0(x)  # Map to hyperbolic space
        output = self.manifold.mobius_matvec(self.weight, x)
        output = self.manifold.expmap0(output) + self.bias
        return output

class HyperbolicNN(nn.Module):
    """A basic hyperbolic neural network."""
    def __init__(self, input_dim, hidden_dim, output_dim, c=1.0):
        super().__init__()
        self.fc1 = HyperbolicLinear(input_dim, hidden_dim, c)
        self.fc2 = HyperbolicLinear(hidden_dim, output_dim, c)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

# Example usage
input_dim, hidden_dim, output_dim = 10, 20, 1
model = HyperbolicNN(input_dim, hidden_dim, output_dim, c=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Generate random data
x = torch.randn(32, input_dim)
y = torch.randn(32, output_dim)

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}: Loss = {loss.item()}")
