from pathlib import Path
from typing import Union, List, Dict, Tuple
import numpy as np
import OpenGL.GL as gl
from ..GLGraphicsItem import GLGraphicsItem
from ..transform3d import Matrix4x4
from .GLModelItem import GLModelItem
from .GLAxisItem import GLAxisItem
from pathlib import Path

import xml.etree.ElementTree as ET
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent

__all__ = ['GLURDFItem', 'GLLinkItem', 'Joint']

rad2deg = 180 / np.pi


def parse_origin(origin: ET.Element):
    rpy_str = origin.get('rpy', '0 0 0').split()
    rpy = [float(val) * rad2deg for val in rpy_str]  # degree
    xyz_str = origin.get('xyz', '0 0 0').split()
    xyz = [float(val) for val in xyz_str]
    return rpy, xyz


class GLLinkItem(GLGraphicsItem):

    def __init__(
            self,
            name: str = None,
            mesh_path: Union[str, Path] = None,
            lights: list = None,
            origin: Matrix4x4 = None,
            glOptions: str = "translucent",
            axis_visiable: bool = False,
            parentItem: GLGraphicsItem = None,
    ):
        super().__init__(parentItem=parentItem)
        self.name = name
        self.visual_model = None
        # axis
        self.axis = GLAxisItem(size=(0.12, 0.12, 0.12), tip_size=0.12)
        self.axis.setVisible(axis_visiable)
        self.addChildItem(self.axis)

        # visual model: optional
        if mesh_path is not None:
            self.visual_model = GLModelItem(
                path=mesh_path,
                lights=lights,
                glOptions=glOptions,
            )

            # visual origin transform, visual model 相对于 连体坐标系 的位姿
            if origin is not None:
                self.visual_model.setTransform(origin)

            self.addChildItem(self.visual_model)

    def childLinks(self) -> List['GLLinkItem']:
        return [item for item in self.childItems() if isinstance(item, GLLinkItem)]

    def set_data(
            self,
            axis_visiable: bool = None,
            visual_visiable: bool = None,
    ):
        if axis_visiable is not None:
            self.axis.setVisible(axis_visiable)
            if axis_visiable and self.visual_model is not None:
                self.visual_model.setMaterialData(opacity=0.5)
            elif self.visual_model is not None:
                self.visual_model.setMaterialData(opacity=1)

        if visual_visiable is not None and self.visual_model is not None:
            self.visual_model.setVisible(visual_visiable)

    def set_origin(self, origin: Matrix4x4):
        """visual model 相对于 连体坐标系 的位姿"""
        if self.visual_model is not None:
            self.visual_model.setTransform(origin)


@dataclass
class Joint:
    name: str
    parent: GLLinkItem
    child: GLLinkItem
    type: str  # revolute, prismatic. fixed
    axis: np.ndarray = None
    limit: np.ndarray = None
    origin: Matrix4x4 = None  # child axis relative to parent axis
    _value: float = 0

    def __post_init__(self):
        assert self.type in ['revolute', 'prismatic', 'fixed'], \
            "type must be revolute, prismatic or fixed"
        if self.origin is None:
            self.origin = Matrix4x4()
        if self.type == 'fixed':
            self.limit = np.array([0, 0])

        self.set_value(self._value)
        self.parent.addChildItem(self.child)  # 添加父子关系

    @property
    def value(self):
        return self._value

    def set_value(self, value):  # rad or m
        self._value = np.clip(value, self.limit[0], self.limit[1])
        tf = Matrix4x4()
        # 根据关节类型设置关节值
        if self.type == 'revolute':
            tf = tf.fromAxisAndAngle(self.axis[0], self.axis[1],
                                     self.axis[2], rad2deg * value)
        elif self.type == 'prismatic':
            t = self.axis * value
            tf = tf.moveto(t[0], t[1], t[2])
        self.child.setTransform(self.origin * tf)

    def set_origin(self, origin: Matrix4x4):
        self.origin = origin
        self.set_value(self._value)


class GLURDFItem(GLGraphicsItem):
    """ Displays a GelSlim model with a surface plot on top of it."""

    def __init__(
            self,
            urdf_path: Union[str, Path],
            lights: list,
            glOptions: str = "translucent",
            parentItem=None,
            axis_visiable=False,
            **kwargs  # 传递给 GLLinkItem.set_data
    ):
        super().__init__(parentItem=parentItem)

        # 解析xml文件
        self._lights = lights
        self._glOptions = glOptions
        self._urdf_path = Path(urdf_path)
        self._base_dir = self._urdf_path.parent
        self._urdf = ET.parse(urdf_path)
        self._links: Dict[str, GLLinkItem] = dict()
        self._joints: Dict[str, Joint] = dict()

        # 遍历每个link元素
        for link in self._urdf.findall('link'):
            name = link.get('name')
            self._links[name] = self._parse_link(name, link)
            self._links[name].set_data(axis_visiable=axis_visiable, **kwargs)

        # 遍历每个joint元素
        for joint in self._urdf.findall('joint'):
            name = joint.get('name')
            type = joint.get('type')
            origin = Matrix4x4.fromRpyXyz(*parse_origin(joint.find('origin')))
            parent = self._links[joint.find('parent').get('link')]
            child = self._links[joint.find('child').get('link')]
            axis = None
            limit = None

            if type in ['revolute', 'prismatic']:
                axis = np.array(joint.find('axis').get('xyz').split(), dtype=float)
                limit = np.array([joint.find('limit').get('lower'), joint.find('limit').get('upper')], dtype=float)

            self._joints[name] = Joint(name, parent, child, type, axis, limit, origin)

        # 设置 base_link
        self.base_link = None
        for link in self._links.values():
            if link.parentItem() is None:
                self.addChildItem(link)
                self.base_link = link
                break

        if self.base_link is None:
            raise ValueError("No base link found")

    def set_joint(self, name: Union[int, str], value):
        """ 设置活动关节的值, 若 name 为 int, 表示活动关节序号 """
        joint = self.get_joint(name)
        joint.set_value(value)

    def get_joint(self, name: Union[int, str]) -> Joint:
        """ 返回关节实例, 若 name 为 int, 表示活动关节序号 """
        if isinstance(name, int):
            return self.get_joints(movable=True)[name]
        else:
            return self._joints[name]

    def set_joints(self, values: Union[list, np.ndarray]):
        """ 设置所有活动关节 """
        for i, joint in enumerate(self.get_joints(movable=True)):
            joint.set_value(values[i])

    def get_joints(self, movable=True) -> List[Joint]:
        """ 默认返回所有活动关节 """
        if movable:
            return [joint for joint in self._joints.values() if joint.type != 'fixed']
        else:
            return list(self._joints.values())

    def get_joints_attr(self, attr: str, movable=True) -> List:
        """ 所有活动关节的属性, attr: name, value, axis, limit"""
        joints = self.get_joints(movable)
        return [getattr(joint, attr) for joint in joints]

    def get_links_name(self) -> List[str]:
        return list(self._links.keys())

    def set_link(self, name: Union[int, str], **kwargs):  # axis_visiable, visual_visiable
        if isinstance(name, int):
            name = list(self._links.keys())[name]
        self._links[name].set_data(**kwargs)

    def get_link(self, name: Union[int, str]) -> GLLinkItem:
        if isinstance(name, int):
            name = list(self._links.keys())[name]
        return self._links[name]

    def add_link(self, link: GLLinkItem,
                 parent_link: Union[int, str],
                 joint_name: str,
                 joint_type: str,
                 joint_axis: Tuple[float, float, float],
                 joint_limit: Tuple[float, float],
                 origin: Matrix4x4 = None):
        """添加一个link和joint到urdf模型中"""

        self._links[link.name] = link

        self._joints[joint_name] = Joint(
            joint_name,
            parent=self.get_link(parent_link),
            child=link,
            type=joint_type,
            axis=np.array(joint_axis),
            limit=np.array(joint_limit),
            origin=origin
        )

    def print_links(self):
        """dfs print"""
        prefix = ' '
        print(self._urdf_path)
        stack = [(self.base_link, 1)]  # node, 缩进级别

        while (stack):
            node, level = stack.pop()
            print(prefix * level + node.name)

            for child in reversed(node.childLinks()):
                stack.append((child, level + 1))
        print()

    def print_joints(self):
        for name, joint in self._joints.items():
            print(f"{name} | {joint.type} | val: {joint.value} | "
                  f"axis: {joint.axis} | limit: {joint.limit}")
        print()

    def _parse_link(self, name: str, link_elem: ET.Element) -> GLLinkItem:
        mesh_path = None
        origin_tf = Matrix4x4()

        # visual mesh path: optional
        visual = link_elem.find('visual')
        if visual is not None:
            mesh = visual.find('geometry').find('mesh')
            if mesh is not None:
                mesh_path = self._base_dir / mesh.get('filename')

            # visual mesh origin transform: optional
            origin = visual.find('origin')
            if origin is not None:
                origin_tf = Matrix4x4.fromRpyXyz(*parse_origin(origin))

        return GLLinkItem(
            name=name,
            mesh_path=mesh_path,
            lights=self._lights,
            origin=origin_tf,
            glOptions=self._glOptions,
        )
