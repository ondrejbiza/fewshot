import pybullet as pb
import time
from sre_parse import fix_flags
from pybullet_planning.pybullet_tools.kuka_primitives import BodyPose, BodyConf, Command, get_grasp_gen, \
    get_ik_fn, get_free_motion_gen, get_holding_motion_gen
from pybullet_planning.pybullet_tools import utils as pu


pu.connect(use_gui=True)
pu.disable_real_time()
pu.draw_global_system()

pb.setPhysicsEngineParameter(enableFileCaching=0)

while True:

    mug = pu.load_model("../data/mugs/0.urdf", fixed_base=False)    
    print(pb.getBodyInfo(mug, 0))

    time.sleep(1)
    pu.remove_body(mug)
