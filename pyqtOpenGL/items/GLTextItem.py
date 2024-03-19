import sys
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .shader import Shader
from .BufferObject import VAO, VBO
from .texture import Texture2D
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import OpenGL.GL as gl
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLTextItem']


class GLTextItem(GLGraphicsItem):
    """Draws points at a list of 3D positions."""

    def __init__(
            self,
            text: str = None,
            pos=None,
            font=None,  # "times.ttf", "msyh.ttc", "Deng.ttf"
            color=(255, 255, 255, 255),
            fontsize=40,
            fixed=True,  # 是否固定在视图上, if True, pos is in viewport, else in world
            glOptions='ontop',
            parentItem=None
    ):
        super().__init__(parentItem=parentItem)
        if pos is None:
            pos = [0, 0, -10]
        self.setGLOptions(glOptions)
        self.setDepthValue(100)
        self._fixed = fixed
        self._tex_update_flag = False
        self._wh_update_flag = False
        self._pixel_wh = [0, 0]
        self._text_w = 0
        self._text_h = 0
        self.vertices = np.array([
            # 顶点坐标             # texcoord
            -1, -1, 0, 0.0, 0.0,
            1, -1, 0, 1.0, 0.0,
            1, 1, 0, 1.0, 1.0,
            1, 1, 0, 1.0, 1.0,
            -1, 1, 0, 0.0, 1.0,
            -1, -1, 0, 0.0, 0.0,
        ], dtype=np.float32).reshape(-1, 5)

        self.setData(text, font, color, fontsize, pos)

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        self.vbo = VBO([self.vertices], [[3, 2]], usage=gl.GL_STATIC_DRAW)
        self.vbo.setAttrPointer([0], attr_id=[[0, 1]])
        self.tex = Texture2D(None, flip_y=True, wrap_s=gl.GL_CLAMP_TO_EDGE, wrap_t=gl.GL_CLAMP_TO_EDGE)

    def updateGL(self):
        if self._tex_update_flag:
            self.tex.updateTexture(self._image)
            self._tex_update_flag = False

        if self._wh_update_flag:
            w, h = self._text_w, self._text_h
            self.vertices[:, :3] = np.array([
                [0, 0, 0], [w, 0, 0],
                [w, h, 0], [w, h, 0],
                [0, h, 0], [0, 0, 0]
            ])

            self.vbo.updateData([0], [self.vertices])
            self._wh_update_flag = False

    def setData(self, text: str = None, font=None, color=None, fontsize=None, pos=None):
        if text is not None:
            self._text = text
            self._tex_update_flag = True
        if color is not None:
            self._color = np.array(color)
            if np.max(self._color) <= 1:
                self._color = (self._color * 255).astype(np.uint8)
            if self._color.shape[0] == 3:
                self._color = np.append(self._color, 255)
            self._tex_update_flag = True
        if font is not None:
            self._font = font
            self._tex_update_flag = True
        else:
            if sys.platform == "win32":
                self._font = "Deng.ttf"
            elif sys.platform in ("linux", "linux2"):
                self._font = "/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf"

        if fontsize is not None:
            self._fontsize = fontsize
            self._tex_update_flag = True

        if self._tex_update_flag:
            font = ImageFont.truetype(self._font, self._fontsize, encoding="unic")  # Deng.ttf, msyh.ttc
            self._pixel_wh = np.array(font.getbbox(text)[2:])
            image = Image.new("RGBA", tuple(self._pixel_wh), (0, 0, 0, 0))  # 背景透明
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), self._text, font=font, fill=tuple(self._color), encoding="utf-8")
            self._image = np.array(image, np.uint8)

        if pos is not None:
            self._pos = np.array(pos)

        self.update()

    def paint(self, model_matrix=Matrix4x4()):
        if self._text is None:
            return

        # calc text box size (w, h), mantain pixel size
        if self._fixed:
            w = self._pixel_wh[0] * 2 / self.view().deviceWidth()
            h = self._pixel_wh[1] * 2 / self.view().deviceHeight()
            pos = self._pos * 2 - 1  # map to [-1, 1]
            pos[2] = 0
        else:
            pixelsize = self.view().pixelSize(model_matrix * self._pos)
            w = self._pixel_wh[0] * pixelsize
            h = self._pixel_wh[1] * pixelsize
            pos = self._pos

        if w != self._text_w or h != self._text_h:
            self._wh_update_flag = True
            self._text_w = w
            self._text_h = h

        self.updateGL()
        self.setupGLState()

        self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
        self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
        self.shader.set_uniform("model", model_matrix.glData, "mat4")
        self.shader.set_uniform("is_fixed", self._fixed, "bool")
        self.shader.set_uniform("text_pos", pos, "vec3")
        self.tex.bind()
        self.shader.set_uniform("texture1", self.tex, "sample2D")

        with self.shader:
            self.vao.bind()
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 6)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform bool is_fixed;
uniform vec3 text_pos;

layout (location = 0) in vec3 iPos;
layout (location = 1) in vec2 iTexCoord;

out vec2 TexCoord;

void main() {
    if (is_fixed) {
        gl_Position = vec4(text_pos + iPos, 1.0);
    } else {
        //gl_Position = vec4(text_pos + iPos, 1.0);
        vec4 tpos = view * model * vec4(text_pos, 1.0);
        gl_Position = proj * vec4(tpos.xyz + iPos, 1.0);
    }
    TexCoord =iTexCoord;
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoord;
uniform sampler2D texture1;

void main() {
    FragColor = texture(texture1, TexCoord);
}
"""
