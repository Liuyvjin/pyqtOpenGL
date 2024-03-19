from math import radians, tan
from enum import Enum, auto
from .transform3d import Matrix4x4, Quaternion, Vector3


class Camera:

    def __init__(
            self,
            position=Vector3(0., 0., 5),
            yaw=0,
            pitch=0,
            roll=0,
            fov=45,
    ):
        """View Corrdinate System
        default front vector: (0, 0, -1)
        default up Vector: (0, 1, 0)
        yaw: rotate around VCS y-axis
        pitch: rotate around VCS x-axis
        roll: rotate around VCS z-axis
        """
        self.pos = Vector3(position)  # 世界坐标系原点指向相机位置的向量, 在相机坐标下的坐标
        self.quat = Quaternion.fromEulerAngles(pitch, yaw, roll)
        self.fov = fov

    def get_view_matrix(self):
        return Matrix4x4.fromTranslation(-self.pos.x, -self.pos.y, -self.pos.z) * self.quat

    def set_view_matrix(self, view_matrix: Matrix4x4):
        self.quat = view_matrix.toQuaternion()
        self.pos = -view_matrix.toTranslation()

    def get_quat_pos(self):
        return self.quat.copy(), self.pos.copy()

    def set_quat_pos(self, quat=None, pos=None):
        if quat is not None:
            self.quat = quat
        if pos is not None:
            self.pos = pos

    def get_projection_matrix(self, width, height, fov=None):
        distance = max(self.pos.z, 1)
        if fov is None:
            fov = self.fov

        return Matrix4x4.create_projection(
            fov,
            width / height,
            0.001 * distance,
            100.0 * distance
        )

    def get_proj_view_matrix(self, width, height, fov=None):
        return self.get_projection_matrix(width, height, fov) * self.get_view_matrix()

    def get_view_pos(self):
        """计算相机在世界坐标系下的坐标"""
        return self.quat.inverse() * self.pos

    def orbit(self, yaw, pitch, roll=0, base=None):
        """Orbits the camera around the center position.
        *yaw* and *pitch* are given in degrees."""
        q = Quaternion.fromEulerAngles(pitch, yaw, roll)

        if base is None:
            base = self.quat

        self.quat = q * base

    def pan(self, dx, dy, dz=0.0, width=1000, base=None):
        """Pans the camera by the given amount of *dx*, *dy* and *dz*."""
        if base is None:
            base = self.pos

        scale = self.pos.z * 2. * tan(0.5 * radians(self.fov)) / width

        self.pos = base + Vector3([-dx * scale, -dy * scale, dz * scale])
        if self.pos.z < 0.1:
            self.pos.z = 0.1

    def set_params(self, position=None, pitch=None, yaw=None, roll=0, fov=None):
        if position is not None:
            self.pos = Vector3(position)
        if yaw is not None or pitch is not None:
            self.quat = Quaternion.fromEulerAngles(pitch, yaw, roll)
        if fov is not None:
            self.fov = fov

    def get_params(self):
        return self.pos, self.quat.toEulerAngles()
