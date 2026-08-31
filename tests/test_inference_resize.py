import sys
import types
import unittest
from unittest import mock

from HYPIR.utils.inference_resize import inference_resize_shape, resize_to_shape


class InferenceResizeShapeTest(unittest.TestCase):
    def test_none_keeps_original_shape(self):
        self.assertEqual(inference_resize_shape(480, 640, "none", 1024), (480, 640))

    def test_area_mode_scales_square_to_reference_area(self):
        self.assertEqual(inference_resize_shape(512, 512, "area", 1024), (1024, 1024))

    def test_area_mode_preserves_rectangular_aspect_ratio(self):
        self.assertEqual(inference_resize_shape(512, 1024, "area", 1024), (724, 1448))

    def test_area_mode_does_not_resize_larger_area(self):
        self.assertEqual(inference_resize_shape(512, 2048, "area", 1024), (512, 2048))

    def test_short_edge_mode_scales_wide_shape(self):
        self.assertEqual(inference_resize_shape(512, 1024, "short-edge", 1024), (1024, 2048))

    def test_short_edge_mode_scales_tall_shape(self):
        self.assertEqual(inference_resize_shape(1024, 512, "short-edge", 1024), (2048, 1024))

    def test_short_edge_mode_does_not_resize_large_shape(self):
        self.assertEqual(inference_resize_shape(1024, 2048, "short-edge", 1024), (1024, 2048))

    def test_invalid_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            inference_resize_shape(0, 640, "area", 1024)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            inference_resize_shape(480, 640, "square", 1024)

    def test_non_positive_reference_size_is_rejected_for_resize_modes(self):
        with self.assertRaises(ValueError):
            inference_resize_shape(480, 640, "area", 0)


class ResizeToShapeTest(unittest.TestCase):
    def test_bicubic_resize_enables_antialiasing_and_clamps_output(self):
        tensor = object()
        resized = mock.Mock()
        functional = types.ModuleType("torch.nn.functional")
        functional.interpolate = mock.Mock(return_value=resized)
        torch_nn = types.ModuleType("torch.nn")
        torch_nn.functional = functional
        torch = types.ModuleType("torch")
        torch.nn = torch_nn

        with mock.patch.dict(
            sys.modules,
            {"torch": torch, "torch.nn": torch_nn, "torch.nn.functional": functional},
        ):
            result = resize_to_shape(tensor, (512, 768))

        functional.interpolate.assert_called_once_with(
            tensor,
            size=(512, 768),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        resized.clamp.assert_called_once_with(0, 1)
        self.assertIs(result, resized.clamp.return_value)


if __name__ == "__main__":
    unittest.main()
