"""
Unit and Integration testing suite for Module 5 (Synthetic Dataset Generator).
Verifies:
1. Seed-based determinism (pixel-level equivalence).
2. Correctness of each degradation plugin.
3. Composure and execution of DegradationPipeline presets.
4. Dataset generator executions in both single and multi-threaded modes.
"""

import os
import shutil
import unittest
import json
from pathlib import Path
import numpy as np
import cv2

from dataset_manager.config import PipelineConfig
from synthetic_generator import (
    BaseDegradation,
    DegradationRegistry,
    DegradationPipeline,
    SyntheticDatasetGenerator,
    create_mock_pattern,
)


class TestDegradationsUnit(unittest.TestCase):
    """Unit tests for individual degradation operators."""

    def setUp(self):
        # Create a deterministic base clean image
        self.image = np.full((128, 128), 127, dtype=np.uint8)
        cv2.circle(self.image, (64, 64), 30, 200, -1)
        self.state1 = np.random.RandomState(42)
        self.state2 = np.random.RandomState(42)

    def test_determinism_gaussian_noise(self):
        """GaussianNoise must be strictly deterministic with identical RandomStates."""
        op1 = DegradationRegistry.get_operator("gaussian_noise", std_range=[0.05, 0.05])
        op2 = DegradationRegistry.get_operator("gaussian_noise", std_range=[0.05, 0.05])
        
        out1, meta1 = op1.apply(self.image, self.state1)
        out2, meta2 = op2.apply(self.image, self.state2)
        
        np.testing.assert_array_equal(out1, out2)
        self.assertEqual(meta1["std"], meta2["std"])

    def test_determinism_poisson_noise(self):
        """PoissonNoise must be strictly deterministic."""
        op1 = DegradationRegistry.get_operator("poisson_noise", scale_range=[20.0, 20.0])
        op2 = DegradationRegistry.get_operator("poisson_noise", scale_range=[20.0, 20.0])
        
        out1, meta1 = op1.apply(self.image, self.state1)
        out2, meta2 = op2.apply(self.image, self.state2)
        
        np.testing.assert_array_equal(out1, out2)

    def test_defocus_blur(self):
        """DefocusBlur circular convolution radius check."""
        op = DegradationRegistry.get_operator("defocus_blur", radius_range=[4, 4])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, self.image.shape)
        self.assertEqual(meta["radius"], 4)
        self.assertEqual(meta["kernel_size"], 9)

    def test_motion_blur(self):
        """MotionBlur directional convolution."""
        op = DegradationRegistry.get_operator("motion_blur", size_range=[9, 9], angle_range=[45.0, 45.0])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, self.image.shape)
        self.assertEqual(meta["kernel_size"], 9)
        self.assertAlmostEqual(meta["angle"], 45.0)

    def test_jpeg_compression(self):
        """JPEG compression artifacts."""
        op = DegradationRegistry.get_operator("jpeg_compression", quality_range=[50, 50])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, self.image.shape)
        self.assertEqual(meta["compression_quality"], 50)

    def test_downsampling_resizes(self):
        """Downsampling operator scale reduction."""
        op = DegradationRegistry.get_operator("downsampling", scale_range=[0.5, 0.5])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, (64, 64))
        self.assertAlmostEqual(meta["scale"], 0.5)

    def test_bicubic_degradation_restores_shape(self):
        """Bicubic degradation downsamples and returns to original shape."""
        op = DegradationRegistry.get_operator("bicubic_degradation", scale_range=[0.5, 0.5])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, self.image.shape)
        self.assertAlmostEqual(meta["scale"], 0.5)

    def test_scanning_charging_streaks(self):
        """ScanningChargingStreaks applies horizontal streaking lines."""
        op = DegradationRegistry.get_operator("scanning_charging_streaks", streaks_range=[2, 2], strength_range=[0.2, 0.2])
        out, meta = op.apply(self.image, self.state1)
        self.assertEqual(out.shape, self.image.shape)
        self.assertEqual(meta["streaks_count"], 2)


class TestDegradationPipeline(unittest.TestCase):
    """Pipeline and Presets test suite."""

    def test_preset_sem_loading(self):
        """Verify that presets load correct plugins."""
        pipe = DegradationPipeline(preset_name="sem")
        op_names = [op[0] for op in pipe.operators]
        self.assertIn("gaussian_blur", op_names)
        self.assertIn("scanning_charging_streaks", op_names)
        self.assertIn("sensor_noise", op_names)

    def test_pipeline_determinism(self):
        """Verify pipeline execution determinism."""
        img = np.full((128, 128), 128, dtype=np.uint8)
        pipe1 = DegradationPipeline(preset_name="sem")
        pipe2 = DegradationPipeline(preset_name="sem")
        
        state1 = np.random.RandomState(100)
        state2 = np.random.RandomState(100)

        out1, meta1 = pipe1.run(img, state1)
        out2, meta2 = pipe2.run(img, state2)

        np.testing.assert_array_equal(out1, out2)
        self.assertEqual(meta1["severity_level"], meta2["severity_level"])


class TestGeneratorIntegration(unittest.TestCase):
    """Integration tests running SyntheticDatasetGenerator."""

    def setUp(self):
        self.config_path = Path("./config.yaml")
        self.config = PipelineConfig.load_from_yaml(self.config_path)
        self.test_clean_dir = Path("./datasets/test_clean_sources")
        self.test_output_dir = Path("./datasets/test_Synthetic_Generated")
        
        # Override paths for testing isolation
        self.config.synthetic_generator.clean_source_dir = self.test_clean_dir
        self.config.synthetic_generator.output_dir = self.test_output_dir
        self.config.synthetic_generator.num_samples = 4
        self.config.synthetic_generator.preset = "sem"

    def tearDown(self):
        # Cleanup test directories
        if self.test_clean_dir.exists():
            shutil.rmtree(self.test_clean_dir)
        if self.test_output_dir.exists():
            shutil.rmtree(self.test_output_dir)

    def test_dataset_generation_single_threaded(self):
        """Run batch dataset generator in single-threaded mode."""
        self.config.synthetic_generator.multiprocessing = False
        generator = SyntheticDatasetGenerator(self.config)
        
        metadata = generator.generate()
        
        self.assertEqual(len(metadata), 4)
        report_path = self.test_output_dir / "degradation_report.json"
        self.assertTrue(report_path.exists())

        # Verify image assets generated
        for item in metadata:
            self.assertTrue((self.test_output_dir / item["noisy_image"]).exists())
            self.assertTrue((self.test_output_dir / item["gt_image"]).exists())
            # Verify sidecar json exists
            self.assertTrue((self.test_output_dir / f"synthetic_{item['sample_index']:05d}_meta.json").exists())

    def test_dataset_generation_multiprocessing(self):
        """Run dataset generator with multiprocessing Pool."""
        self.config.synthetic_generator.multiprocessing = True
        self.config.synthetic_generator.num_workers = 2
        generator = SyntheticDatasetGenerator(self.config)
        
        metadata = generator.generate()
        self.assertEqual(len(metadata), 4)
        self.assertTrue((self.test_output_dir / "degradation_report.json").exists())


if __name__ == "__main__":
    unittest.main()
