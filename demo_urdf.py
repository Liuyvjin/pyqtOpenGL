import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtOpenGL import GLViewWidget
from pyqtOpenGL.items  import *
from pyqtOpenGL.transform3d import *
from pyqtOpenGL.items.GLURDFItem import GLURDFItem
from pyqtOpenGL import tb

class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.camera.set_params((0.09, 3.55, 23.83), 17.95, 0.13, 0.11)
        self.ax = GLAxisItem(size=(8, 8, 8), width=4)

        # -- lights
        self.light = PointLight(pos=[0, 15, 0],
                               ambient=(0.3, 0.3, 0.3),
                               diffuse=(0.7, 0.7, 0.7),
                               specular=(1, 1, 1),
                               visible=True,
                               directional=True)
        self.light2 = PointLight(pos=[5, -5, 14],
                               ambient=(0.3, 0.3, 0.3),
                               diffuse=(0.7, 0.7, 0.7),
                               specular=(1, 1, 1),
                               visible=True,
                               directional=True)
        # -- grid
        self.grid = GLGridItem(
            size=(50, 50), spacing=(2.5, 2.5), lineWidth=1,
            lights=[self.light]
        )

        self.model = GLURDFItem(
            "./pyqtOpenGL/items/resources/objects/panda/panda.urdf",
            lights=[self.light, self.light2],
        )

        self.model.scale(10, 10, 10)
        self.model.rotate(-90, 1, 0, 0)
        self.model.print_links()
        self.model.print_joints()

        #
        self.addItem(self.ax)
        self.addItem(self.grid)
        self.addItem(self.model)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(20)

        # tool_box
        j_value = self.model.get_joints()
        j_name = self.model.get_joints_name()
        j_limits = self.model.get_joints_limit()
        with tb.window("control", self, 10, size=(400, 300)):
            tb.add_drag_array(
                "joints",
                value = j_value,
                min_val = j_limits[:, 0],
                max_val = j_limits[:, 1],
                step=0.01, decimals=2, horizontal=False,
                format=[name+": %.2f" for name in j_name],
                callback=self.on_changed
            )

    def onTimeout(self):
        self.update()

    def on_changed(self, data):
        id, val = data
        self.model.set_joint(id, val)

    def closeEvent(self, a0) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())