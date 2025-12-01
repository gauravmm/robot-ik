from pprint import pprint
import numpy as np
import skrobot
import skrobot.models.urdf
import logging

logger = logging.getLogger(__name__)

from base import forward_kinematics_vec, load_meta, forward_kinematics

np.set_printoptions(suppress=True, precision=1)

# Setup for all tests:
robot = skrobot.models.urdf.RobotModelFromURDF(urdf_file="so101/so101_new_calib.urdf")
rbt = load_meta(robot)


def test_fk_1():
    test_angles = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.1, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        [-0.1, 0.2, -0.3, 0.4, -0.5, 0.6],
        [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
    ]

    for meta in rbt.joints:
        print(meta.name)

    for i, angles in enumerate(test_angles):
        robot.angle_vector(angles)

        np.testing.assert_allclose(
            robot.angle_vector(), angles, err_msg=f"Test case {i} violates angle limit."
        )
        preds = forward_kinematics_vec(rbt, angles)
        link: skrobot.model.Link
        for link in robot.link_list:
            if pred := preds.get(link.name):
                ref = link.worldcoords().T()
                np.testing.assert_array_almost_equal(
                    pred, ref, err_msg=f"Mismatch in {i} at {link.name}"
                )
