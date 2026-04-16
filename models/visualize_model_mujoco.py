##
#
# Use Mujoco's Viewer to show the model.
#
##

import mujoco
from mujoco.viewer import launch
import numpy as np

# Load the model from XML
xml_file = "./models/paddle_ball.xml"
# xml_file = "./models/hopper.xml"
# xml_file = "./models/g1_23dof.xml"

# load and launch the model
model =  mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)

# set the print precision
np.set_printoptions(precision=4, suppress=True)

# print some info about the model
print("\n#####################  INFO  #####################")

# file name
print("Model file name:", xml_file)

# joints
joint_type_dict = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
print("\nNumber of joints:", model.njnt)
for i in range(model.njnt):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = model.jnt_type[i]
    lower, upper = model.jnt_range[i]
    print(f"    Joint {i} name:", name)
    print(f"    Joint {i} type:", joint_type_dict[jtype])
    if jtype in (2, 3):  # slide or hinge
        print(f"    Limits: [{lower:.4f}, {upper:.4f}]")
    else:
        print("    Limits: N/A")

# actuators
print("\nNumber of actuators:", model.nu)
for i in range(model.nu):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    atype = model.actuator_trntype[i]  # transmission type
    lower, upper = model.actuator_ctrlrange[i]
    gear = model.actuator_gear[i, 0]
    print(f"    Actuator {i} name:", name)
    print(f"    Actuator {i} transmission type:", atype)
    print(f"    Control limits: [{lower:.4f}, {upper:.4f}]")
    print(f"    Gear ratio: {gear}")

# bodies
print("\nNumber of bodies:", model.nbody)
total_mass = 0.0
for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        mass = model.body_mass[i]
        inertia = model.body_inertia[i]
        total_mass += mass
        print(f"    Body {i} name: {name}")
        print(f"    Body {i} mass: {mass:.4f}")
        print(f"    Body {i} inertia: {inertia}")

print(f"\n    Total mass: {total_mass:.4f}")

print("\n##################################################")

# launch the viewer
launch(model)
