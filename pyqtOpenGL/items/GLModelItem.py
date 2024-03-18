from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .MeshData import Mesh
from .light import LightMixin, light_fragment_shader
from .GLMeshItem import vertex_shader
from typing import List

__all__ = ['GLModelItem']


class GLModelItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        path,
        lights = list(),
        glOptions='translucent',
        parentItem=None,
    ):
        super().__init__(parentItem=parentItem)
        self._path = path
        self._directory = Path(path).parent
        self.setGLOptions(glOptions)
        # model
        self.meshes: List[Mesh] = Mesh.load_model(path)
        self._order = list(range(len(self.meshes)))
        # light
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, light_fragment_shader)

        for m in self.meshes:
            m.initializeGL()

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        self.setupLight(self.shader)

        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
            for i in self._order:
                self.meshes[i].paint(self.shader)

    def setMaterial(self, mesh_id, material):
        self.meshes[mesh_id].setMaterial(material)

    def getMaterial(self, mesh_id):
        return self.meshes[mesh_id]._material

    def setMaterialData(self, **kwargs):
        for mesh in self.meshes:
            mesh._material.set_data(**kwargs)

    def setPaintOrder(self, order: list):
        """设置绘制顺序, order为mesh的索引列表"""
        assert max(order) < len(self.meshes) and min(order) >= 0
        self._order = order
