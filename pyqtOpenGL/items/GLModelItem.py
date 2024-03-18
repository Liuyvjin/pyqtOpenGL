from pathlib import Path
from ..GLGraphicsItem import GLGraphicsItem, PickColorManager
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
            lights=None,
            glOptions='translucent',
            parentItem=None,
            selectable=False
    ):
        super().__init__(parentItem=parentItem, selectable=selectable)
        if lights is None:
            lights = list()
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
        self.pick_shader = Shader(_pick_vertex_shader, self.pick_fragment_shader)
        for m in self.meshes:
            m.initializeGL()

    def paint(self, model_matrix=Matrix4x4()):
        if not self.selected():
            self.setupGLState()
            self.setupLight(self.shader)

            with self.shader:
                self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
                self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
                self.shader.set_uniform("model", model_matrix.glData, "mat4")
                self.shader.set_uniform("ViewPos", self.view_pos(), "vec3")
                for i in self._order:
                    self.meshes[i].paint(self.shader)
        else:
            self.paint_selected(model_matrix)

    def paint_selected(self, model_matrix=Matrix4x4()):
        self.paint_pickMode(model_matrix)  # TODO: 暂时的实现

    def paint_pickMode(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        with self.pick_shader:
            self.pick_shader.set_uniform("view", self.proj_view_matrix().glData, "mat4")
            self.pick_shader.set_uniform("model", model_matrix.glData, "mat4")
            self.pick_shader.set_uniform("pickColor", self._pickColor, "float")
            for i in self._order:
                self.meshes[i].paint(self.pick_shader)

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


_pick_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 model;

void main() {
    TexCoords = aTexCoords;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    gl_Position = view * vec4(FragPos, 1.0);
}
"""