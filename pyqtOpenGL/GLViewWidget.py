import cv2
import numpy as np
from typing import List, Set
from OpenGL.GL import *  # noqa
from math import radians, cos, sin, tan, sqrt
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import QPoint

from pyqtOpenGL.items.GLSelectBox import GLSelectBox
from .camera import Camera
from .functions import mkColor
from .transform3d import Matrix4x4, Quaternion, Vector3
from .GLGraphicsItem import GLGraphicsItem, PickColorManager
from .items.light import PointLight


class GLViewWidget(QtWidgets.QOpenGLWidget):

    def __init__(
            self,
            cam_position=Vector3(0., 0., 10.),
            yaw=0.,
            pitch=0.,
            roll=0.,
            fov=45.,
            bg_color=(0.2, 0.3, 0.3, 1.),
            # bg_color = (0.95, 0.95, 0.95, 1.),
            parent=None,
    ):
        """
        Basic widget for displaying 3D data
          - Rotation/scale controls
        """
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        self.camera = Camera(cam_position, yaw, pitch, roll, fov)
        self.bg_color = bg_color
        self.items: List[GLGraphicsItem] = []
        self.lights: Set[PointLight] = set()

        # 选择框
        self.select_box = GLSelectBox()
        self.select_box.setView(self)
        self.selected_items = []  # 用于管理物体的选中状态

        self.last_pos = None
        self.press_pos = None

        # 设置多重采样抗锯齿
        format = QtGui.QSurfaceFormat()
        format.setSamples(4)
        self.setFormat(format)

    def get_proj_view_matrix(self):
        view = self.camera.get_view_matrix()
        proj = self.camera.get_projection_matrix(
            self.deviceWidth(),
            self.deviceHeight()
        )
        return proj * view

    def get_proj_matrix(self):
        return self.camera.get_projection_matrix(
            self.deviceWidth(),
            self.deviceHeight()
        )

    def get_view_matrix(self):
        return self.camera.get_view_matrix()

    def deviceWidth(self):
        dpr = self.devicePixelRatioF()
        return int(self.width() * dpr)

    def deviceHeight(self):
        dpr = self.devicePixelRatioF()
        return int(self.height() * dpr)

    def deviceRatio(self):
        return self.height() / self.width()

    def reset(self):
        self.camera.set_params(Vector3(0., 0., 10.), 0, 0, 0, 45)

    def addItem(self, item: GLGraphicsItem):
        self.items.append(item)
        item.setView(self)
        if hasattr(item, 'lights'):
            self.lights |= set(item.lights)
        self.items.sort(key=lambda a: a.depthValue())
        self.update()

    def addItems(self, items: List[GLGraphicsItem]):
        for item in items:
            self.addItem(item)

    def removeItem(self, item):
        """
        Remove the item from the scene.
        """
        self.items.remove(item)
        item._setView(None)
        self.update()

    def clear(self):
        """
        Remove all items from the scene.
        """
        for item in self.items:
            item._setView(None)
        self.items = []
        self.update()

    def setBackgroundColor(self, *args, **kwds):
        """
        Set the background color of the widget. Accepts the same arguments as
        :func:`~pyqtgraph.mkColor`.
        """
        self.bg_color = mkColor(*args, **kwds).getRgbF()
        self.update()

    def getViewport(self):
        return (0, 0, self.deviceWidth(), self.deviceHeight())

    def initializeGL(self):
        """initialize OpenGL state after creating the GL context."""
        PointLight.initializeGL()
        # 创建频幕大小的帧缓冲区，这样就不需要调整缓冲区大小
        screen = QtWidgets.QApplication.screenAt(QtGui.QCursor.pos())
        WIN_WID, WIN_HEI = screen.size().width(), screen.size().height()
        self._createFramebuffer(WIN_WID, WIN_HEI)
        self.select_box.initializeGL()
        glEnable(GL_MULTISAMPLE)
        self.addItem(self.select_box)

    def paintGL(self):
        """
        viewport specifies the arguments to glViewport. If None, then we use self.opts['viewport']
        region specifies the sub-region of self.opts['viewport'] that should be rendered.
        Note that we may use viewport != self.opts['viewport'] when exporting.
        """
        glClearColor(*self.bg_color)
        glDepthMask(GL_TRUE)
        glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        self.select_box.updateGL() if self.select_box.visible() else None
        self.drawItems(pickMode=False)

    # 在外部调用paintGL
    def paintGL_outside(self):
        """
        在外部调用paintGL
        """
        self.makeCurrent()
        self.paintGL()
        self.doneCurrent()
        self.parent().update() if self.parent() else self.update()

    def _createFramebuffer(self, width, height):
        """
        创建帧缓冲区, 用于拾取
        Create a framebuffer for picking
        """
        self.__framebuffer = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.__framebuffer)
        self.__texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.__texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, None)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.__texture, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def _resizeFramebuffer(self, width, height):
        glBindFramebuffer(GL_FRAMEBUFFER, self.__framebuffer)
        glBindTexture(GL_TEXTURE_2D, self.__texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED, GL_FLOAT, None)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def pickItems(self, x_, y_, w_, h_):
        ratio = 2  # 为了提高渲染和拾取速度，暂将渲染视口缩小4倍
        x_, y_, w_, h_ = self._normalizeRect(x_, y_, w_, h_, ratio)
        glBindFramebuffer(GL_FRAMEBUFFER, self.__framebuffer)
        glViewport(0, 0, self.deviceWidth() // ratio, self.deviceHeight() // ratio)
        glClearColor(0, 0, 0, 0)
        glDisable(GL_MULTISAMPLE)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # 设置拾取区域
        glScissor(int(x_), int(self.deviceHeight() // ratio - y_ - h_), int(w_), int(h_))
        glEnable(GL_SCISSOR_TEST)
        # 在这里设置额外的拾取参数，例如鼠标位置等
        self.drawItems(pickMode=True)
        glDisable(GL_SCISSOR_TEST)
        pixels = glReadPixels(x_, self.deviceHeight() // ratio - y_ - h_, w_, h_, GL_RED, GL_FLOAT)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glClearColor(*self.bg_color)
        glEnable(GL_MULTISAMPLE)
        glViewport(0, 0, self.deviceWidth(), self.deviceHeight())
        # 获取拾取到的物体
        pick_data = np.frombuffer(pixels, dtype=np.float32)
        # # 保存为图片
        # img_data = np.frombuffer(pixels, dtype=np.uint8).reshape(h_, w_,
        # img_data = np.flipud(img_data)
        # import PIL.Image as Image
        # img = Image.fromarray(img_data)
        # img.save('pick.png')
        # 去掉所有为0.0的数据
        pick_data = pick_data[pick_data != 0.0]
        # 获取选中的物体
        selected_items = []
        id_set = list()
        for id_ in pick_data:
            if id_ in id_set:
                continue
            item: GLGraphicsItem = PickColorManager().get(id_)
            if item:
                selected_items.append(item)
            id_set.append(id_)
        return selected_items

    def _normalizeRect(self, x_, y, w, h, ratio: int = 3):
        """
        防止拾取区域超出窗口范围
        Prevent the picking area from exceeding the window range
        :param x_: 拾取区域左下角的x坐标
        :param y: 拾取区域左下角的y坐标
        :param w: 拾取区域的宽度
        :param h: 拾取区域的高度
        :param ratio: 缩放比例
        :return:
        """
        if ratio > 5:
            ratio = 5
            print("[Warning] ratio should be less than 5.")
        # 防止拾取区域超出窗口范围
        if w < 0:
            x_, w = max(0, x_ + w), -w
        if h < 0:
            y, h = max(0, y + h), -h
        if x_ + w > self.deviceWidth():
            w = self.deviceWidth() - x_
        if y + h > self.deviceHeight():
            h = self.deviceHeight() - y
        x_, y, w, h = x_ // ratio, y // ratio, w // ratio, h // ratio

        # 防止拾取区域过小
        if w <= 6 // ratio:
            x_ = max(0, x_ - 1)
            w = 3 // ratio
        if h <= 6 // ratio:
            y = max(0, y - 1)
            h = 3 // ratio

        return x_, y, w, h

    def drawItems(self, pickMode=False):
        if pickMode:  # 拾取模式
            for it in self.items:
                try:
                    it.drawItemTree_pickMode()
                except Exception as e:
                    printExc()
                    print("Error while drawing item %s in pick mode." % str(it))
        else:
            for it in self.items:  # 正常模式
                try:
                    it.drawItemTree()
                except Exception as e:  # noqa
                    printExc()
                    print("Error while drawing item %s." % str(it))

            # draw lights
            for light in self.lights:
                light.paint(self.get_proj_view_matrix())

    def get_selected_item(self):
        """
        得到选中的物体，不改变物体的选中状态，不添加到self.selected_items中
        """
        self.makeCurrent()
        size = self.select_box.size()
        selected_items = self.pickItems(self.select_box.start().x(), self.select_box.start().y(), size.x(), size.y())
        self.paintGL()
        self.doneCurrent()
        self.parent().update() if self.parent() else self.update()
        return selected_items

    def pixelSize(self, pos=Vector3(0, 0, 0)):
        """
        depth: z-value in global coordinate system
        Return the approximate (y) size of a screen pixel at the location pos
        Pos may be a Vector or an (N,3) array of locations
        """
        pos = self.get_view_matrix() * pos  # convert to view coordinates
        fov = self.camera.fov
        return max(-pos[2], 0) * 2. * tan(0.5 * radians(fov)) / self.deviceHeight()

    def mousePressEvent(self, ev):
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.press_pos = lpos
        self.last_pos = lpos
        self.cam_press_quat, self.cam_press_pos = self.camera.get_quat_pos()
        if ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.select_box.setSelectStart(lpos)

    def mouseMoveEvent(self, ev):
        ctrl_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        shift_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)
        alt_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()

        cam_quat, cam_pos = self.camera.get_quat_pos()
        diff = lpos - self.last_pos
        self.last_pos = lpos

        if shift_down and not alt_down:  # 锁定水平或垂直转动
            cam_quat, cam_pos = self.cam_press_quat, self.cam_press_pos
            diff = lpos - self.press_pos
            if abs(diff.x()) > abs(diff.y()):
                diff.setY(0)
            else:
                diff.setX(0)

        if ctrl_down:
            diff *= 0.1

        if alt_down:
            roll = diff.x() / 5

        if ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            if alt_down:
                self.camera.orbit(0, 0, roll, base=cam_quat)
            else:
                self.camera.orbit(diff.x(), diff.y(), base=cam_quat)
        elif ev.buttons() == QtCore.Qt.MouseButton.MiddleButton:
            self.camera.pan(diff.x(), -diff.y(), 0, base=cam_pos)
        elif ev.buttons() == QtCore.Qt.MouseButton.LeftButton:
            self.select_box.setVisible(True)
            self.select_box.setSelectEnd(lpos)
        self.update()

    def mouseReleaseEvent(self, ev):
        ctl_down = (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)
        self.select_box.setSelectEnd(ev.position() if hasattr(ev, 'position') else ev.localPos())
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            self.select_box.setVisible(False)
            new_s_items = self.get_selected_item()  # 此函数仅用于获取选中的物体，不改变物体的选中状态
            if not new_s_items:  # 没有选中的物体
                if ctl_down:  # 如果按下ctrl键，不清空选中的物体
                    return
                # 如果不按下ctrl键，清空选中的物体
                for it in self.selected_items:
                    it.setSelected(False)
                self.selected_items.clear()
                return
            # 如果不按下ctrl键，取两集合的交集的补集（即从选中的物体中去掉已经选中的物体）
            if not ctl_down:
                for it in new_s_items:
                    if it in self.selected_items:
                        it.setSelected(False)
                        self.selected_items.remove(it)
                    else:
                        it.setSelected(True)
                        self.selected_items.append(it)
            # 如果按下ctrl键，取两集合相加
            else:
                for it in new_s_items:
                    if it not in self.selected_items:
                        it.setSelected(True)
                        self.selected_items.append(it)

    def wheelEvent(self, ev):
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        if (ev.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier):
            self.camera.fov *= 0.999 ** delta
        else:
            self.camera.pos.z = self.camera.pos.z * 0.999 ** delta
        self.update()

    def readQImage(self):
        """
        Read the current buffer pixels out as a QImage.
        """
        return self.grabFramebuffer()

    def readImage(self):
        """
        Read the current buffer pixels out as a cv2 Image.
        """
        qimage = self.grabFramebuffer()
        w, h = self.width(), self.height()
        bytes_ = qimage.bits().asstring(qimage.byteCount())
        img = np.frombuffer(bytes_, np.uint8).reshape((h, w, 4))
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def isCurrent(self):
        """
        Return True if this GLWidget's context is current.
        """
        return self.context() == QtGui.QOpenGLContext.currentContext()

    def keyPressEvent(self, a0) -> None:
        """按键处理"""
        if a0.text() == '1':
            pos, euler = self.camera.get_params()
            print(f"pos: ({pos.x:.2f}, {pos.y:.2f}, {pos.z:.2f})  "
                  f"euler: ({euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f})")
        elif a0.text() == '2':
            self.camera.set_params((0.00, 0.00, 886.87),
                                   pitch=-31.90, yaw=-0, roll=-90)
            # self.camera.set_params((1.72, -2.23, 27.53),pitch=-27.17, yaw=2.64, roll=-70.07)


import warnings
import traceback
import sys


def formatException(exctype, value, tb, skip=0):
    """Return a list of formatted exception strings.

    Similar to traceback.format_exception, but displays the entire stack trace
    rather than just the portion downstream of the point where the exception is
    caught. In particular, unhandled exceptions that occur during Qt signal
    handling do not usually show the portion of the stack that emitted the
    signal.
    """
    lines = traceback.format_exception(exctype, value, tb)
    lines = [lines[0]] + traceback.format_stack()[:-(skip + 1)] + ['  --- exception caught here ---\n'] + lines[1:]
    return lines


def getExc(indent=4, prefix='|  ', skip=1):
    lines = formatException(*sys.exc_info(), skip=skip)
    lines2 = []
    for l in lines:
        lines2.extend(l.strip('\n').split('\n'))
    lines3 = [" " * indent + prefix + l for l in lines2]
    return '\n'.join(lines3)


def printExc(msg='', indent=4, prefix='|'):
    """Print an error message followed by an indented exception backtrace
    (This function is intended to be called within except: blocks)"""
    exc = getExc(indent=0, prefix="", skip=2)
    # print(" "*indent + prefix + '='*30 + '>>')
    warnings.warn("\n".join([msg, exc]), RuntimeWarning, stacklevel=2)
    # print(" "*indent + prefix + '='*30 + '<<')


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    win = GLViewWidget(None)
    win.show()
    sys.exit(app.exec_())
