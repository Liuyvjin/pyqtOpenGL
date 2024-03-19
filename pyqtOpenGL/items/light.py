import numpy as np
import math
import OpenGL.GL as gl
from ctypes import c_float, sizeof, c_void_p, Structure
from .shader import Shader
from .BufferObject import VAO, VBO, EBO
from ..transform3d import Vector3, Matrix4x4
from ..GLGraphicsItem import GLGraphicsItem
from .MeshData import sphere


__all__ = ["PointLight", "LightMixin", "light_fragment_shader"]

class PointLight(GLGraphicsItem):

    def __init__(
        self,
        pos = Vector3(0.0, 0.0, 0.0),
        ambient = Vector3(0.2, 0.2, 0.2),
        diffuse = Vector3(0.8, 0.8, 0.8),
        specular = Vector3(1.0, 1.0, 1.0),
        constant = 1.0,
        linear = 0.01,
        quadratic = 0.001,
        visible = True,
        directional = False, # 平行光源选项
        glOptions = "opaque",
    ):
        super().__init__(parentItem=None)
        self.setGLOptions(glOptions)
        self.position = Vector3(pos)
        self.amibent = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.constant = constant
        self.linear = linear
        self.quadratic = quadratic
        self.__visible = visible
        self.directional = directional

    def set_uniform(self, shader: Shader, name: str):
        shader.set_uniform(name + ".position", self.position, "vec3")
        shader.set_uniform(name + ".ambient", self.amibent, "vec3")
        shader.set_uniform(name + ".diffuse", self.diffuse, "vec3")
        shader.set_uniform(name + ".specular", self.specular, "vec3")
        shader.set_uniform(name + ".constant", self.constant, "float")
        shader.set_uniform(name + ".linear", self.linear, "float")
        shader.set_uniform(name + ".quadratic", self.quadratic, "float")
        shader.set_uniform(name + ".directional", self.directional, "bool")

    def set_data(self, pos=None, ambient=None, diffuse=None, specular=None, visible=None):
        if pos is not None:
            self.position = Vector3(pos)
        if ambient is not None:
            self.amibent = Vector3(ambient)
        if diffuse is not None:
            self.diffuse = Vector3(diffuse)
        if specular is not None:
            self.specular = Vector3(specular)
        if visible is not None:
            self.__visible = visible

    def translate(self, dx, dy, dz):
        self.position += [dx, dy, dz]

    def rotate(self, x, y, z, angle):
        tr = Matrix4x4().fromAxisAndAngle(x, y, z, angle)
        self.position = tr * self.position

    def scale(self, x, y, z):
        self.position *= [x, y, z]

    @classmethod
    def initializeGL(cls):
        """在GLViewWidget创建后自动调用, 初始化光源的渲染数据"""
        cls._light_vert, cls._light_idx = sphere(0.1, 12, 12)
        cls._light_vao = VAO()
        cls._light_vbo = VBO([cls._light_vert], [3])
        cls._light_vbo.setAttrPointer([0], [0])
        cls._light_ebo = EBO(cls._light_idx)
        cls._light_shader = Shader(vertex_shader, fragment_shader)

    def paint(self, view_matrix: Matrix4x4):
        if not self.__visible:
            return

        self.setupGLState()
        with PointLight._light_shader:
            PointLight._light_shader.set_uniform("view", view_matrix.glData, "mat4")
            PointLight._light_shader.set_uniform("_lightPos", self.position, "vec3")
            PointLight._light_shader.set_uniform("_lightColor", self.diffuse, "vec3")
            PointLight._light_vao.bind()
            gl.glDrawElements(gl.GL_TRIANGLES, self._light_idx.size, gl.GL_UNSIGNED_INT, c_void_p(0))


class LightMixin():

    @property
    def light_count(self):
        return len(self.lights)

    def addLight(self, light: PointLight):
        self.lights = list()
        if isinstance(light, PointLight):
            self.lights.append(light)
        elif isinstance(light, list):
            self.lights.extend(light)

    def setupLight(self, shader: Shader):
        """设置光源 uniform 属性
        """
        for i, light in enumerate(self.lights):
            light.set_uniform(shader, "pointLight[" + str(i) + "]")
        shader.set_uniform("nr_point_lights", len(self.lights), "int")



vertex_shader = """
#version 330 core

uniform mat4 view;

layout (location = 0) in vec3 iPos;
uniform vec3 _lightPos;

void main() {
    gl_Position = view * vec4(iPos + _lightPos, 1.0);
}
"""

fragment_shader = """
#version 330 core
out vec4 FragColor;

uniform vec3 _lightColor;

void main() {
    FragColor = vec4(_lightColor, 1.0);
}
"""

light_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
in vec3 FragPos;
in vec3 Normal;

uniform vec3 ViewPos;

struct Material {
    float opacity;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
    bool use_texture;
    sampler2D tex_diffuse;
};
uniform Material material;

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

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewPos)
{
    vec3 viewDir = normalize(viewPos - fragPos);
    vec3 lightDir = vec3(0);
    float attenuation = 1.0;
    float distance = 0.0;
    if (light.directional)
        lightDir = normalize(light.position);
    else
        lightDir = normalize(light.position - fragPos);
        distance = length(light.position - fragPos);
        attenuation = 1.0 / (light.constant + light.linear * distance +
                     light.quadratic * (distance * distance));

    //vec3 halfwayDir = normalize(lightDir + viewDir);
    vec3 reflectDir = reflect(-lightDir, normal);
    // 漫反射着色
    float diff = max(dot(normal, lightDir), 0.0);
    // 镜面光着色
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), max(material.shininess, 1.0));

    // 合并结果
    vec3 ambient  = vec3(0);
    vec3 diffuse  = vec3(0);
    vec3 specular = vec3(0);
    if (material.use_texture) {
        ambient  = light.ambient  * vec3(texture(material.tex_diffuse, TexCoords));
        diffuse  = light.diffuse  * diff * vec3(texture(material.tex_diffuse, TexCoords));
        specular = light.specular * spec * vec3(texture(material.tex_diffuse, TexCoords));
    } else {
        ambient  = light.ambient  * material.ambient;
        diffuse  = light.diffuse  * diff * material.diffuse;
        specular = light.specular * spec * material.specular;
    }

    return attenuation * (ambient + specular + diffuse);
}

void main() {
    vec3 result = vec3(0);
    for(int i = 0; i < nr_point_lights; i++)
        result += CalcPointLight(pointLight[i], Normal, FragPos, ViewPos);
    FragColor = vec4(result, material.opacity);
    //float gamma = 2.2;
    //FragColor = vec4(pow(result, vec3(1.0 / gamma)), material.opacity);
}
"""
