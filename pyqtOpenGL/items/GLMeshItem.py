import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import vertex_normal, Mesh
from .light import LightMixin, light_fragment_shader

__all__ = ['GLMeshItem', 'mesh_vertex_shader']


class GLMeshItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        vertexes = None,
        indices = None,
        normals = None,
        texcoords = None,
        lights = list(),
        material = None,
        calc_normals = True,
        mesh : Mesh = None,
        glOptions = 'opaque',
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        if mesh is not None:
            self._mesh = mesh
        else:
            self._mesh = Mesh(vertexes, indices, texcoords, normals,
                          material, None, gl.GL_STATIC_DRAW,
                          calc_normals)
        self.addLight(lights)

    def initializeGL(self):
        self.shader = Shader(mesh_vertex_shader, light_fragment_shader)
        self._mesh.initializeGL()

    def paint(self, model_matrix=Matrix4x4()):
        self.setupGLState()
        self.setupLight(self.shader)

        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
            self._mesh.paint(self.shader)
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)

    def setMaterial(self, material):
        self._mesh.setMaterial(material)

    def getMaterial(self):
        return self._mesh.getMaterial()

mesh_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoords;

out vec2 TexCoords;
out vec3 FragPos;
out vec3 Normal;

uniform mat4 view;
uniform mat4 proj;
uniform mat4 model;

void main() {
    TexCoords = aTexCoords;
    FragPos = vec3(model * vec4(aPos, 1.0));
    Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""
