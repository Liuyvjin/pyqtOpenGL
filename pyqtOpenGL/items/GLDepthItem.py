from pathlib import Path
from typing import List, Union

from .shader import Shader
from .MeshData import vertex_normal, Mesh
from ..transform3d import Matrix4x4, Vector3
from ..GLGraphicsItem import GLGraphicsItem
from .GLMeshItem import vertex_shader

__all__ = ['GLDepthItem']

class GLDepthItem(GLGraphicsItem):

    def __init__(
        self,
        vertexes = None,
        indices = None,
        path: Union[str, Path] = None,
        glOptions: Union[str, dict] = 'translucent_cull',
        parentItem: GLGraphicsItem = None,
    ):
        """将模型渲染为深度图

        :param vertexes: 顶点数据, 当给定 `path` 时无效, defaults to None
        :param indices: 索引数据, 当给定 `path` 时无效, defaults to None
        :param path: 模型文件路径, defaults to None
        :param glOptions: defaults to 'translucent_cull'
        :param parentItem: defaults to None
        """
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)

        # model
        if path is not None:
            self.meshes: List[Mesh] = Mesh.load_model(path)
        else:
            self.meshes = [Mesh(vertexes, indices)]
        self._order = list(range(len(self.meshes)))

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)

        for m in self.meshes:
            m.initializeGL()

        # 设置背景为黑色
        self.view().bg_color = (0., 0., 0., 1.)

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()

        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
            for i in self._order:
                self.meshes[i].paint(self.shader)


fragment_shader = """
#version 330 core

out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform vec3 ViewPos;
uniform mat4 view;

void main() {
    vec3 frag_pos_wrt_camera = vec3(view * vec4(FragPos, 1.0));

    //float distance = abs(frag_pos_wrt_camera.z);
    float distance = length(frag_pos_wrt_camera);

    // 0 -> 1.2, 20 -> 0
    FragColor = vec4(vec3(max((20 - distance), 0) / 20 * 1.2), 1);
}
"""