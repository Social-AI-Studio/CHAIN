"""
Multi-view rendering utilities for physics environments.

This module provides multi-view rendering capabilities to capture
the physics simulation from different camera angles and viewpoints.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Optional
import math
from chainbench.core import CameraConfig

# Conditional pybullet import
try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    # Mock pybullet for testing
    class MockPyBullet:
        ER_BULLET_HARDWARE_OPENGL = 0
        @staticmethod
        def computeViewMatrix(*args, **kwargs): return []
        @staticmethod  
        def computeProjectionMatrixFOV(*args, **kwargs): return []
        @staticmethod
        def getCameraImage(*args, **kwargs): return (512, 512, np.zeros((512, 512, 3), dtype=np.uint8), None, None)
    p = MockPyBullet()


class MultiViewRenderer:
    """Renderer for capturing multiple viewpoints of the physics simulation."""

    def __init__(self, width: int = 512, height: int = 512):
        self.width = width
        self.height = height

        self.camera_configs: Dict[str, CameraConfig] = {}
        self.set_camera_config(
            "front",
            position=[0, -1.5, 0.8],  # 正前方，y轴负方向看向原点
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=1.0,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "side",
            position=[1.5, 0, 0.8],   # 右侧，x轴正方向看向原点
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=1.0,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "top",
            position=[0, 0, 2.0],     # 顶部，z轴正方向俯视
            target=[0, 0, 0.3],
            up=[0, 1, 0],
            fov=60.0,
            aspect_ratio=1.0,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )
        self.set_camera_config(
            "perspective",
            position=[1.0, -1.0, 1.2],  # 斜上方视角
            target=[0, 0, 0.3],
            up=[0, 0, 1],
            fov=60.0,
            aspect_ratio=1.0,
            near_plane=0.1,
            far_plane=10.0,
            image_width=width,
            image_height=height
        )

    def set_camera_config(self, camera_name: str, position: List[float], 
                         target: List[float], up: List[float] = [0, 0, 1],
                         fov: float = 60.0, aspect_ratio: float = 1.0,
                         near_plane: float = 0.1, far_plane: float = 10.0,
                         image_width: Optional[int] = None, image_height: Optional[int] = None) -> None:
        """Set custom camera configuration using CameraConfig."""
        if image_width is None:
            image_width = self.width
        if image_height is None:
            image_height = self.height
        self.camera_configs[camera_name] = CameraConfig(
            position=tuple(position),
            target=tuple(target),
            up_vector=tuple(up),
            fov=fov,
            aspect_ratio=aspect_ratio,
            near_plane=near_plane,
            far_plane=far_plane,
            image_width=image_width,
            image_height=image_height
        )

    def render_single_view(self, camera_name: str) -> Image.Image:
        """Render from a single camera viewpoint."""
        if camera_name not in self.camera_configs:
            raise ValueError(f"Unknown camera: {camera_name}")

        config: CameraConfig = self.camera_configs[camera_name]

        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=list(config.position),
            cameraTargetPosition=list(config.target),
            cameraUpVector=list(config.up_vector)
        )

        # Compute projection matrix
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=config.fov,
            aspect=config.aspect_ratio,
            nearVal=config.near_plane,
            farVal=config.far_plane
        )

        # Capture image
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=config.image_width,
            height=config.image_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )

        # Convert to PIL Image
        rgb_array = np.array(rgb_img, dtype=np.uint8)
        rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
        rgb_array = rgb_array.reshape(height, width, 3)

        image = Image.fromarray(rgb_array)
        view = self._add_camera_label(image, camera_name)

        # Add label
        return view

    def render_multi_view(self, camera_order: List[str] = ["front", "side", "top", "perspective"]) -> Image.Image:
        """Render multiple views and combine into single image."""
        views = {}

        for camera_name in camera_order:
            views[camera_name] = self.render_single_view(camera_name)

        # Create combined image
        combined_width = self.width * 2
        combined_height = self.height * 2
        combined_image = Image.new("RGB", (combined_width, combined_height), color="white")

        # Paste views in 2x2 grid
        positions = [
            (0, 0),                              # top-left: front
            (self.width, 0),                     # top-right: side  
            (0, self.height),                    # bottom-left: top
            (self.width, self.height)            # bottom-right: perspective
        ]

        for i, camera_name in enumerate(camera_order):
            combined_image.paste(views[camera_name], positions[i])

        return combined_image
        
    def render_circular_views(self, center: List[float] = [0, 0, 0.3], 
                            radius: float = 1.5, height: float = 0.8, 
                            num_views: int = 8) -> Dict[str, Image.Image]:
        """Render views arranged in a circle around the scene."""
        views = {}

        for i in range(num_views):
            angle = 2 * math.pi * i / num_views

            # Calculate camera position
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            z = height

            camera_name = f"circular_{i:02d}"

            self.set_camera_config(camera_name, [x, y, z], center)

            # Render view
            views[camera_name] = self.render_single_view(camera_name)

        return views

    def _add_camera_label(self, image: Image.Image, label: str) -> Image.Image:
        """Add camera name label to image."""
        # Create a copy to avoid modifying original
        labeled_image = image.copy()
        draw = ImageDraw.Draw(labeled_image)

        # Try to load a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            try:
                font = ImageFont.load_default()
            except:
                font = None

        # Draw label background
        text_bbox = draw.textbbox((0, 0), label.upper(), font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 4
        bg_bbox = [
            5, 5,
            5 + text_width + 2 * padding,
            5 + text_height + 2 * padding
        ]

        draw.rectangle(bg_bbox, fill="black", outline="white")

        # Draw text
        draw.text(
            (5 + padding, 5 + padding),
            label.upper(),
            fill="white",
            font=font
        )

        return labeled_image
