import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtOpenGL import GLViewWidget
from pyqtOpenGL.items import *
from pyqtOpenGL.transform3d import *
import PIL.Image as Image
from time import time


class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)

        ver1, ind1, _, norm1 = sphere(2, 20, 20, calc_uv_norm=True)
        ver2, ind2 = cylinder(radius=[1.2, 1], cols=12, rows=8, length=2.4)
        img = Image.open("./pyqtOpenGL/items/resources/textures/box.png")
        img = np.array(img, dtype='f4')

        self.img = GLImageItem(img, width_height=(0.2, 0.2))
        self.ax = GLAxisItem(size=(8, 8, 8))
        self.ax_fixed = GLAxisItem(fix_to_corner=True)
        self.box = GLBoxTextureItem(size=(2, 2, 2)).translate(0, 1.1, 0)
        self.text = GLTextItem(text="Hello World", pos=(2, 6, -1), color=(1, 0.6, 1), fixed=False)
        self.arrow = GLArrowPlotItem(
            start_pos=ver1 + [5, -1, 0],
            end_pos=ver1 + norm1 + [5, -1, 0],
            color=[1, 1, 0]
        )

        # -- scatter and line
        pos = np.random.uniform(-2, 2, size=(15, 3)).astype('f4') * (2, 1, 2) + [0, -3, 0]
        color = np.random.uniform(0, 1, size=(15, 3)).astype('f4')
        self.scatter = GLScatterPlotItem(pos=pos, color=color, size=2)
        self.line = GLLinePlotItem(pos=pos, color=color, lineWidth=2)

        # -- lights
        self.light = PointLight(pos=[0, 5, 4], ambient=(0.8, 0.8, 0.8), diffuse=(0.8, 0.8, 0.8))
        self.light1 = PointLight(pos=[0, -5, 1], diffuse=(0, .8, 0))
        self.light2 = PointLight(pos=[-12, 3, 2], diffuse=(0.8, 0, 0))

        # -- grid
        self.grid = GLGridItem(
            size=(7, 7), lineWidth=1,
            lights=[self.light, self.light1, self.light2]
        )

        # -- model
        self.model = GLModelItem(
            "./pyqtOpenGL/items/resources/objects/cyborg/cyborg.obj",
            lights=[self.light, self.light1, self.light2],
            selectable=True
        ).translate(0, 2, 0)
        # self.model = GLModelItem(
        #     "./pyqtOpenGL/items/resources/objects/USS-BB-63/USS-BB-63.obj",
        #     lights=[self.light, self.light1, self.light2],
        #     selectable=True
        # )

        # -- mesh
        self.mesh1 = GLInstancedMeshItem(
            pos=[[5, -1, 0], [-3, 5, -5], [4, 6, -8]],
            lights=[self.light, self.light1, self.light2],
            indices=ind1,
            vertexes=ver1,
            normals=norm1,
            color=(0.7, 0.8, 0.8)
        )

        self.mesh2 = GLMeshItem(
            indices=ind2,
            vertexes=ver2,
            lights=[self.light, self.light1, self.light2],
            material=Material((0.4, 0.1, 0.1), diffuse=(0.6, 0.1, 0.3))
        ).rotate(-50, 1, 0.4, 0).translate(-6, 2, -2)

        # -- surface
        self.zmap = np.random.uniform(0, 1.5, (25, 25))
        self.texture = Texture2D(sin_texture(0))
        self.surf = GLSurfacePlotItem(
            zmap=self.zmap, x_size=6, lights=[self.light, self.light1, self.light2],
            material=Material((0.2, 0.2, 0.2), diffuse=(0.5, 0.5, 0.5), textures=[self.texture])
        ).rotate(-90, 1, 0, 0).translate(-6, -1, 0)

        # -- 3d grid
        z = np.random.uniform(-3, -2, (5, 6))
        y = np.arange(5) + 2
        x = np.arange(6) + 1
        X, Y = np.meshgrid(x, y, indexing='xy')
        grid = np.stack([X, Y, z], axis=2)
        color = np.random.random((5, 6, 3))
        self.grid3d = GL3DGridItem(grid=grid, fill=True, opacity=0.5, color=color)
        self.grid3d.setDepthValue(10)
        self.grid3d1 = GL3DGridItem(grid=grid, fill=False, color=(0, 0, 0))
        self.grid3d1.setDepthValue(-1)

        self.addItem(self.grid3d)
        self.addItem(self.grid3d1)
        self.addItem(self.text)
        self.addItem(self.img)
        self.addItem(self.ax)
        self.addItem(self.ax_fixed)
        self.addItem(self.grid)
        self.addItem(self.scatter)
        self.addItem(self.line)
        self.addItem(self.box)
        self.addItem(self.arrow)
        self.addItem(self.model)
        self.addItem(self.surf)
        self.addItem(self.mesh1)
        self.addItem(self.mesh2)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(20)

    def onTimeout(self):
        self.light.rotate(0, 1, 0.4, 1)
        self.light1.rotate(1, 1, 0, -2)
        self.light2.rotate(0.2, 1., 0., 1.5)
        self.texture.updateTexture(sin_texture(time()))
        self.update()


def sin_texture(t):
    delta = t % 100
    x = np.linspace(-10, 10, 50, dtype='f4')
    y = x.copy()
    X, Y = np.meshgrid(x, y, indexing='xy')
    Z = (np.sin(np.sqrt(X ** 2 + Y ** 2) * np.pi / 5 - delta * np.pi) + 1) / 5
    return np.stack([Z, Z, Z], axis=2)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())
