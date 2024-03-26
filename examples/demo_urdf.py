import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtOpenGL.items import *
from pyqtOpenGL import GLViewWidget, tb, Matrix4x4

class GLView(GLViewWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.resize(1200, 900)
        self.camera.set_params((-0.01, 0.38, 3.94), -73, 0.0, -90)
        self.ax = GLAxisItem(fix_to_corner=True)

        # -- lights
        self.light = PointLight(pos=[0, 0, 15],
                               ambient=(0.3, 0.3, 0.3),
                               diffuse=(0.7, 0.7, 0.7),
                               specular=(1, 1, 1),
                               visible=True,
                               directional=True)
        self.light2 = PointLight(pos=[5, 14, -5],
                               ambient=(0.3, 0.3, 0.3),
                               diffuse=(0.7, 0.7, 0.7),
                               specular=(1, 1, 1),
                               visible=True,
                               directional=True)
        # -- grid
        self.grid = GLGridItem(
            size=(5, 5), spacing=(0.25, 0.25), lineWidth=1,
            lights=[self.light]
        ).rotate(90, 1, 0, 0)

        self.model = GLURDFItem(
            "./pyqtOpenGL/items/resources/objects/panda/panda_with_gelslim.urdf",
            lights=[self.light, self.light2]
        )
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
        j_value = self.model.get_joints_attr("value")
        j_name = self.model.get_joints_attr("name")
        j_limits = np.array(self.model.get_joints_attr("limit"))
        links_name = self.model.get_links_name()

        with tb.window("control", self, 10, size=(400, 300), pos=(-400, 0)):
            tb.add_drag_array(
                "joints",
                value = j_value,
                min_val = j_limits[:, 0], max_val = j_limits[:, 1],
                step=[0.01]*7 + [0.001, 0.001], decimals=[2]*7+[3,3],
                format=[name+": %.3f" for name in j_name],
                callback=self.on_changed, horizontal=False
            )
            tb.add_text("axis_visible")
            tb.add_checklist("axis_visible", items=links_name, callback=self.on_changed,
                             exclusive=False, horizontal=False, value=None)

    def onTimeout(self):
        self.update()

    def on_changed(self, data):
        tag, val = data
        label = self.sender().get_label()
        if label == "joints":
            self.model.set_joint(tag, val)
        elif label == "axis_visible":
            self.model.set_link(tag, axis_visiable=val)

    def closeEvent(self, a0) -> None:
        tb.clean()
        return super().closeEvent(a0)


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = GLView(None)
    win.show()
    sys.exit(app.exec_())