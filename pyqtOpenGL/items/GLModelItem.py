from pathlib import Path

from OpenGL.raw.GL.VERSION.GL_1_0 import glDepthFunc, glStencilFunc, GL_ALWAYS, glDisable, GL_DEPTH_TEST, glStencilOp, \
    GL_KEEP, GL_REPLACE, glStencilMask, glEnable, GL_STENCIL_TEST, GL_NOTEQUAL, glClear, GL_STENCIL_BUFFER_BIT

from ..GLGraphicsItem import GLGraphicsItem, PickColorManager
from ..transform3d import Matrix4x4, Quaternion, Vector3
from .shader import Shader
from .MeshData import Mesh
from .light import LightMixin, light_fragment_shader
from .GLMeshItem import mesh_vertex_shader
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
        self.shader = Shader(mesh_vertex_shader, light_fragment_shader)
        self.pick_shader = Shader(mesh_vertex_shader, self.pick_fragment_shader)
        self.selected_shader = Shader(mesh_vertex_shader, self.selected_fragment_shader)
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
                self.shader.set_uniform("ViewPos", self.view_pos(), "vec3")  # 计算光线夹角用
                for i in self._order:
                    self.meshes[i].paint(self.shader)
        else:
            self.paint_selected(model_matrix)

    def paint_selected(self, model_matrix=Matrix4x4()):

        # Step 1: Set stencil function to GL_ALWAYS and update stencil buffer to 1
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_ALWAYS, 1, 0xFF)
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE)
        glStencilMask(0xFF)

        self.setupGLState()
        self.setupLight(self.shader)

        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos", self.view_pos(), "vec3")
            for i in self._order:
                self.meshes[i].paint(self.shader)

        # 设置模板缓冲区为只读，关闭深度测试
        glStencilFunc(GL_NOTEQUAL, 1, 0xFF)
        glStencilMask(0x00)
        glDisable(GL_DEPTH_TEST)

        # 放大模型
        scaled_model_matrix = model_matrix.scale(1.01, 1.01, 1.01)  # Scale by a small factor

        # 绘制border
        with self.selected_shader:
            self.selected_shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.selected_shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.selected_shader.set_uniform("model", scaled_model_matrix.glData, "mat4")
            self.selected_shader.set_uniform("selectedColor", self._selectedColor, "vec4")
            for i in self._order:
                self.meshes[i].paint(self.selected_shader)

        # 恢复深度测试和模板缓冲区
        glStencilMask(0xFF)
        glClear(GL_STENCIL_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_STENCIL_TEST)

    def paint_pickMode(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        with self.pick_shader:
            self.pick_shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.pick_shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
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


# _pick_vertex_shader = """
# #version 330 core
# layout (location = 0) in vec3 aPos;
# layout (location = 1) in vec3 aNormal;
# layout (location = 2) in vec2 aTexCoords;
#
# out vec2 TexCoords;
# out vec3 FragPos;
# out vec3 Normal;
#
# uniform mat4 view;
# uniform mat4 model;
#
# void main() {
#     TexCoords = aTexCoords;
#     FragPos = vec3(model * vec4(aPos, 1.0));
#     Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
#     gl_Position = view * vec4(FragPos, 1.0);
# }
# """