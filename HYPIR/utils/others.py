import torch
from torch import nn


class EdgeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
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

    def forward(self, image):
        if image.shape[1] == 3:
            red, green, blue = image.unbind(dim=1)
            image = (
                0.2989 * red + 0.5870 * green + 0.1140 * blue
            ).unsqueeze(1)
        edge_x = self.sobel_x(image)
        edge_y = self.sobel_y(image)
        return torch.sqrt(edge_x.square() + edge_y.square() + 1e-6)


def total_variation_loss(image):
    vertical = image[:, :, 1:, :] - image[:, :, :-1, :]
    horizontal = image[:, :, :, 1:] - image[:, :, :, :-1]
    return vertical.abs()[..., :-1] + horizontal.abs()[..., :-1, :]
