from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.spatial.transform import Rotation
from skrobot.coordinates import Coordinates
from skrobot.model import Joint
from skrobot.model.robot_model import RobotModel
from collections import defaultdict, deque
from typing import List


def homogeneous(*, R: Optional[np.ndarray] = None, t: Optional[np.ndarray] = None):
    T = np.eye(4)
    if R is not None:
        assert R.shape == (3, 3)
        T[:3, :3] = R

    if t is not None:
        assert t.shape == (3,)
        T[:3, 3] = t

    return T


@dataclass(frozen=True)
class RevoluteJointMeta:
    name: str
    kind: Literal["revolute"]

    parent: str
    child: str

    axis: np.ndarray  # Axis along which the joint rotates
    child_transform: np.ndarray  # Transform from me to my child

    min_rot: float
    max_rot: float


JointMeta = RevoluteJointMeta


@dataclass(frozen=True)
class LinkMeta:
    name: str
    starting_T: Optional[np.ndarray] = field(default=None)


@dataclass(frozen=True)
class RobotMeta:
    joints: List[JointMeta]
    root_link: LinkMeta
    starting_position: Optional[np.ndarray] = field(default=None)


def load_meta(robot: RobotModel) -> RobotMeta:
    """Load a RobotModel into an Iterable of RevoluteJoints
    Assumes a single chain.
    """
    robot.angle_vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    joints: List[JointMeta] = []

    joint: Joint
    for joint in robot.joint_list:
        joint_type = joint.joint_type
        assert joint_type == "revolute"

        assert joint.parent_link and joint.child_link

        T_parent = joint.parent_link.worldcoords()
        T_child = joint.child_link.worldcoords()
        T_p2c = T_parent.inverse_transformation() * T_child

        # Check that the parent to child mapping is correctly preserved by our joint:
        assert np.allclose(T_parent.T() @ T_p2c.T(), joint.child_link.worldcoords().T())

        jm = RevoluteJointMeta(
            name=joint.name or "Unknown",
            kind=joint.joint_type,
            parent=joint.parent_link.name,
            child=joint.child_link.name,
            axis=joint.axis,  # type: ignore
            child_transform=T_p2c.T(),
            min_rot=joint.min_joint_angle,
            max_rot=joint.max_joint_angle,
        )

        joints.append(jm)

    assert robot.root_link
    root_link = LinkMeta(robot.root_link.name, robot.root_link.worldcoords().T())

    return RobotMeta(topological_sort_joints(joints, root_link), root_link)


def topological_sort_joints(
    joints: List[JointMeta], root_link: LinkMeta
) -> List[JointMeta]:
    """Return joints sorted so that a joint comes before any joint whose parent equals its child.

    Raises:
        ValueError: if there is a cycle or duplicate joint names.
    """
    # We can do this with Kahn's algorithmm, but this is only done once with
    # n< 100 values, so we can do it much more simply in O(n**2)

    # We depopulate this:
    jlookup: Dict[str, List[JointMeta]] = defaultdict(list)
    for j in joints:
        jlookup[j.parent].append(j)
    out: List[JointMeta] = []
    todo = [root_link.name]

    while len(todo):
        curr_link = todo.pop()
        # Terminal links have no children, so in the base case this will return nothing.
        if curr_joints := jlookup.pop(curr_link, None):
            out.extend(curr_joints)
            todo.extend((j.child for j in curr_joints))

    assert (
        len(jlookup) == 0
    ), f"Joints not present in tree: {[v.name for vv in jlookup.values() for v in vv]}"

    return out


def forward_kinematics(
    rbt: RobotMeta, joint_values: Dict[str, float]
) -> Dict[str, np.ndarray]:
    assert len(joint_values) == len(rbt.joints)

    T = {
        rbt.root_link.name: (
            np.eye(4) if rbt.root_link.starting_T is None else rbt.root_link.starting_T
        )
    }

    for meta in rbt.joints:
        assert meta.kind == "revolute"
        parent_T = T[meta.parent]
        assert meta.child not in T, "No more than one joint per child"

        angle = joint_values[meta.name]

        rotation = Rotation.from_rotvec(meta.axis * angle)
        joint_T = homogeneous(R=rotation.as_matrix())

        child_T = parent_T @ meta.child_transform @ joint_T

        T[meta.child] = child_T

    return T
