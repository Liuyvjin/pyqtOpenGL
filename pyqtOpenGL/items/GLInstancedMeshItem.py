import OpenGL.GL as gl
import numpy as np
from .shader import Shader
from .BufferObject import VBO, EBO, VAO
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4, Vector3
from .MeshData import cone, direction_matrixs, vertex_normal
from .light import LightMixin

__all__ = ['GLInstancedMeshItem']


class GLInstancedMeshItem(GLGraphicsItem, LightMixin):

    def __init__(
        self,
        pos = None,  # nx3
        vertexes = None,
        indices = None,
        normals = None,
        lights = list(),
        color = [1., 1., 1.],
        size = 1.,
        opacity = 1.,
        glOptions = 'opaque',
        calcNormals = True,
        parentItem = None
    ):
        super().__init__(parentItem=parentItem)
        self.setGLOptions(glOptions)
        self._vertices = vertexes
        self._indices = indices
        self._normals = normals
        self._pos = None
        self._calcNormals = calcNormals
        if self._calcNormals and self._normals is None and self._indices is not None:
            self._normals = vertex_normal(self._vertices, self._indices)

        self.setData(pos, color, size, opacity)
        self.addLight(lights)

    def setData(self, pos=None, color=None, size=None, opacity=None):
        if color is not None:
            self._color = np.array(color, dtype=np.float32)
            self._gl_update_flag = True

        if pos is not None:
            self._pos = np.array(pos, dtype=np.float32).reshape(-1, 3)
            self._gl_update_flag = True
            if self._color is not None and self._color.size!=self._pos.shape[0]*3 and self._color.size>=3:
                self._color = np.tile(self._color.ravel()[:3], (max(self._pos.shape[0], 1), 1))

        if opacity is not None:
            self._opacity = opacity

        if size is not None:
            self._size = size

        self.update()

    def initializeGL(self):
        self.shader = Shader(vertex_shader, fragment_shader)
        self.vao = VAO()
        # cone
        self.vbo_mesh = VBO(
            [self._vertices, self._normals],
            [3, 3], gl.GL_STATIC_DRAW,
        )
        if self._calcNormals:
            self.vbo_mesh.setAttrPointer([0, 1], divisor=0, attr_id=[0, 1])
        else:
            self.vbo_mesh.setAttrPointer([0], divisor=0, attr_id=[0])

        if self._indices is not None:
            self.ebo = EBO(self._indices)

        # pos
        self.vbo_pos = VBO([self._pos, self._color], [3,3], usage=gl.GL_DYNAMIC_DRAW)
        self.vbo_pos.setAttrPointer([0, 1], [2, 3], divisor=1)

    def updateGL(self):
        if not self._gl_update_flag:
            return

        self.vao.bind()
        self.vbo_pos.updateData([0, 1], [self._pos, self._color])
        self.vbo_pos.setAttrPointer([0, 1], [2, 3], divisor=1)
        self._gl_update_flag = False

    def paint(self, model_matrix=Matrix4x4()):
        if self._pos is None:
            return
        self.updateGL()
        self.setupGLState()
        self.setupLight(self.shader)

        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_LINE)
        self.vao.bind()
        with self.shader:
            self.shader.set_uniform("view", self.view_matrix().glData, "mat4")
            self.shader.set_uniform("proj", self.proj_matrix().glData, "mat4")
            self.shader.set_uniform("model", model_matrix.glData, "mat4")
            self.shader.set_uniform("ViewPos",self.view_pos(), "vec3")
            self.shader.set_uniform("size", self._size, "float")
            self.shader.set_uniform("calcNormal", self._calcNormals, "bool")
            self.shader.set_uniform("opacity", self._opacity, "float")

            if self._indices is not None:
                self.ebo.bind()
                gl.glDrawElementsInstanced(
                    gl.GL_TRIANGLES,
                    self._indices.size,
                    gl.GL_UNSIGNED_INT, None,
                    self._pos.shape[0],
                )
            else:
                gl.glDrawArraysInstanced(
                    gl.GL_TRIANGLES,
                    0,
                    self._vertices.size,
                    self._pos.shape[0],
                )
        # gl.glPolygonMode(gl.GL_FRONT_AND_BACK, gl.GL_FILL)


vertex_shader = """
#version 330 core

uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;
uniform float size;
uniform bool calcNormal;

layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 stPos;
layout (location = 3) in vec3 aColor;

out vec3 FragPos;
out vec3 Normal;
out vec3 oColor;

void main() {
    oColor = aColor;
    FragPos = vec3(model * vec4(stPos + aPos*size, 1.0));
    if (calcNormal){
        Normal = normalize(mat3(transpose(inverse(model))) * aNormal);
    } else {
        Normal = vec3(0, 0, 1);
    }
    gl_Position = proj * view * vec4(FragPos, 1.0);
}
"""


fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 oColor;
in vec3 FragPos;
in vec3 Normal;

uniform float opacity;
uniform vec3 ViewPos;

struct PointLight {
    vec3 position;

    float constant;
    float linear;
    float quadratic;

    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    bool directional;
};
#define MAX_POINT_LIGHTS 10
uniform PointLight pointLight[MAX_POINT_LIGHTS];
uniform int nr_point_lights;

float shininess = 32.0;

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewPos)
{
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 lightDir = vec3(0);
    float attenuation = 1.0;
    float distance = 0.0;
    if (light.directional)
        lightDir = normalize(light.position);
    else{
        lightDir = normalize(light.position - fragPos);
        distance = length(light.position - fragPos);
        attenuation = 1.0 / (light.constant + light.linear * distance +
                     light.quadratic * (distance * distance));
    }
    //vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    // 漫反射着色
    float diff = max(dot(normal, lightDir), 0.0);
    // 镜面光着色
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess);
    //float spec = pow(max(dot(normal, halfwayDir), 0.0), shininess);

    // 合并结果
    vec3 ambient  = light.ambient  * oColor * 0.4;
    vec3 diffuse  = light.diffuse  * diff * oColor * 0.6;
    vec3 specular = light.specular * spec * oColor * 0.2;

    return attenuation * (ambient + specular + diffuse);
}

void main() {
    vec3 result = vec3(0);
    for(int i = 0; i < nr_point_lights; i++)
        result += CalcPointLight(pointLight[i], Normal, FragPos, ViewPos);
    if(nr_point_lights == 0){
        result = oColor;
    }
    FragColor = vec4(result, opacity);
}
"""
