import torch
from torch import nn
from torchvision.transforms.functional import rgb_to_grayscale


class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel filters for edge detection
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x_kernel = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        sobel_y_kernel = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        )

        self.sobel_x.weight = nn.Parameter(
            sobel_x_kernel.view(1, 1, 3, 3), requires_grad=False
        )
        self.sobel_y.weight = nn.Parameter(
            sobel_y_kernel.view(1, 1, 3, 3), requires_grad=False
        )

    def forward(self, x):
        # Convert to grayscale if needed
        if x.shape[1] == 3:
            x = rgb_to_grayscale(x, num_output_channels=1)

        # Apply Sobel filters
        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)

        # Calculate gradient magnitude (edge detection result)
        edges = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)

        return edges


def total_variation_loss(image):
    vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
    horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    return vertical.abs()[..., :-1] + horizontal.abs()[..., :-1, :]
