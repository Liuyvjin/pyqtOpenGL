"""
Defines a 2D select box for mouse selection.
"""
from PyQt5.QtCore import QPoint

from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .BufferObject import VAO, VBO
import numpy as np
import OpenGL.GL as gl


__all__ = ['GLSelectBox']


class GLSelectBox(GLGraphicsItem):
    """
    选择框，用于指示鼠标框选的区域
    A select box for mouse selection.
    """

    def __init__(self, glOptions='ontop', parentItem=None):
        super().__init__(parentItem=None)
        self.setGLOptions(glOptions)
        self.__ortho_matrix = Matrix4x4()
        self.vertices = np.array([
            # 顶点坐标
            -1, -1,
            1, -1,
            1, 1,
            1, 1,
            -1, 1,
            -1, -1,
        ], dtype=np.float32).reshape(-1, 2)
        self.setVisible(False)
        self.__selectStart = QPoint()
        self.__selectEnd = QPoint()

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([self.vertices], [[2]], usage=gl.GL_DYNAMIC_DRAW)
        self.vbo.setAttrPointer([0], attr_id=[[0]])

    def setSelectStart(self, pos: QPoint):
        self.__selectStart = pos

    def setSelectEnd(self, pos: QPoint):
        self.__selectEnd = pos

    def size(self):
        return self.__selectEnd - self.__selectStart

    def start(self):
        return self.__selectStart

    def end(self):
        return self.__selectEnd

    def center(self):
        return (self.__selectStart + self.__selectEnd) / 2

    def updateGL(self):
        self.vertices = np.array([
            self.__selectStart.x(), self.__selectStart.y(),
            self.__selectEnd.x(), self.__selectStart.y(),
            self.__selectEnd.x(), self.__selectEnd.y(),
            self.__selectEnd.x(), self.__selectEnd.y(),
            self.__selectStart.x(), self.__selectEnd.y(),
            self.__selectStart.x(), self.__selectStart.y()
        ], dtype=np.float32).reshape(-1, 2)
        self.vbo.updateData([0], [self.vertices])

    def paint(self, _=None):
        self.setupGLState()
        self.__ortho_matrix.setToIdentity()
        self.__ortho_matrix.ortho(0, self.view().deviceWidth(), self.view().deviceHeight(), 0, -1, 1)
        self.shader.set_uniform("projection", self.__ortho_matrix.glData, "mat4")
        with self.shader:
            self.vao.bind()
            # paint faces
            self.shader.set_uniform("is_surface", True, "bool")
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)
            # paint border
            self.shader.set_uniform("is_surface", False, "bool")
            gl.glLineWidth(0.6)
            gl.glDrawArrays(gl.GL_LINE_LOOP, 0, 6)


vertex_shader = """
#version 330 core

uniform mat4 projection;

layout (location = 0) in vec2 iPos;

void main() {
    gl_Position = projection * vec4(iPos, 0.0, 1.0);
}
"""

fragment_shader = """
#version 330 core
uniform bool is_surface;
out vec4 FragColor;

vec4 surface_color = vec4(1.0, 1.0, 1.0, 0.1);
vec4 border_color = vec4(1.0, 1.0, 1.0, 0.9);

void main() {
    if (is_surface) FragColor = surface_color;
    else FragColor = border_color;
}
"""
