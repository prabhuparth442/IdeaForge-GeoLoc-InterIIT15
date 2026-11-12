"""Camera numbers for the simulator. Same attribute names as header.data.Camera, so either can be used."""


class SimCamera:
    """Pinhole camera with OpenCV lens distortion. Pixel (0, 0) is the centre of the top-left pixel."""

    def __init__(self, fx, fy, cx, cy, width, height, k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0):
        assert fx > 0 and fy > 0, "focal lengths must be positive (pixels)"
        assert width > 0 and height > 0, "image size must be positive (pixels)"
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.width, self.height = int(width), int(height)
        self.k1, self.k2, self.k3, self.p1, self.p2 = k1, k2, k3, p1, p2

    def scaled(self, factor):
        """Same camera at a different resolution (factor 0.25 turns 1280 x 1024 into 320 x 256), for cheap tests."""
        assert factor > 0, "factor must be positive"
        width = int(round(self.width * factor))
        height = int(round(self.height * factor))
        return SimCamera(self.fx * factor, self.fy * factor, (self.cx + 0.5) * factor - 0.5, (self.cy + 0.5) * factor - 0.5,
                         width, height, self.k1, self.k2, self.k3, self.p1, self.p2)

    def to_dict(self):
        return dict(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy, width=self.width, height=self.height,
                    k1=self.k1, k2=self.k2, k3=self.k3, p1=self.p1, p2=self.p2)


def from_camera(camera):
    """Copy any object that has the camera attributes (header.data.Camera, NudtCamera) into a SimCamera."""
    return SimCamera(camera.fx, camera.fy, camera.cx, camera.cy, camera.width, camera.height,
                     camera.k1, camera.k2, camera.k3, camera.p1, camera.p2)
