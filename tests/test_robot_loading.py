from pprint import pprint
import numpy as np
import skrobot
import skrobot.models.urdf
import logging

logger = logging.getLogger(__name__)

from ..base import load_meta, forward_kinematics

np.set_printoptions(suppress=True, precision=1)

# Setup for all tests:
robot = skrobot.models.urdf.RobotModelFromURDF(urdf_file="so101/so101_new_calib.urdf")


def test_topo():
    rbt = load_meta(robot)
    ref = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert [j.name for j in rbt.joints] == ref


def test_fk_1():
    rbt = load_meta(robot)
    test_angles = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    ]

    for i, angles in enumerate(test_angles):
        robot.angle_vector(angles)

        np.testing.assert_allclose(
            robot.angle_vector(), angles, err_msg=f"Test case {i} violates angle limit."
        )
        robot.forward_kinematics()
        angle_set = {j.name: v for j, v in zip(robot.joint_list, angles)}
        preds = forward_kinematics(rbt, angle_set)
        link: skrobot.model.Link
        # Test in sorted order, goes well together with simple use case.
        for link in robot.link_list:
            pred = preds.get(link.name, None)
            if pred is not None:
                assert pred.shape == (4, 4)
                ref = link.worldpos()
                np.testing.assert_array_almost_equal(
                    pred[:3, 3], ref, err_msg=f"Mismatch in {i} at {link.name}"
                )
