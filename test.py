import numpy as np
from scipy.spatial.transform import Rotation
import utils

pos = np.array((0.052327, -0.560935, 0.224860), dtype=np.float32)
quat = np.array((0.737653, 0.005817, -0.003028, -0.675148), dtype=np.float32)
T_1 = utils.pos_quat_to_transform(pos, quat)

pos = np.array((0., 0., 0.))
quat = Rotation.from_euler("z", np.pi).as_quat()
T_2 = utils.pos_quat_to_transform(pos, quat)

T_3 = T_2 @ T_1

pos, quat = utils.transform_to_pos_quat(T_3)

print(list(pos))
print(list(quat))

print(Rotation.from_euler("xyz", (-1.6592, -0.0034, -3.1289)).as_quat())
print(Rotation.from_euler("xyz", (0.0, -0.0, -1.568)).as_quat())
