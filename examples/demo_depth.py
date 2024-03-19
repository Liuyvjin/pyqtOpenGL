import numpy as np
from PyQt5 import QtCore,  QtWidgets
from PyQt5.QtGui import QCloseEvent
from pyqtOpenGL.items import *
from pyqtOpenGL import GLViewWidget, tb, Matrix4x4


class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.resize(800, 600)
        self.camera.set_params((0,0,10), pitch=-65, yaw=0, roll=-90)
        self.addItem(GLAxisItem(fix_to_corner=True))

        # plane
        ver0, _, _ = plane(15, 15)
        self.addItem(GLDepthItem(vertexes=ver0, glOptions='translucent'))

        # sphere
        ver1, ind1 = sphere(radius=1, rows=22, cols=40)
        self.addItem(GLDepthItem(vertexes=ver1, indices=ind1).translate(3,-3,1))

        # clinder 1
        ver2, ind2 = cylinder(radius=2, length=0.5, rows=20, cols=40)
        self.addItem(GLDepthItem(vertexes=ver2, indices=ind2).translate(0,2.5,0))
        ver3, ind3 = cylinder(radius=[2, 0], length=4, rows=20, cols=40)
        self.addItem(GLDepthItem(vertexes=ver3, indices=ind3).translate(-3,-1,0))

        # model
        model = GLDepthItem(path="../pyqtOpenGL/items/resources/objects/cyborg/cyborg.obj")
        model.applyTransform(Matrix4x4.fromEulerAngles(0, 90, 90)).translate(0,2.5,0.5)
        self.addItem(model)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(25)

    def onTimeout(self):
        self.update()

    def closeEvent(self, a0: QCloseEvent) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())