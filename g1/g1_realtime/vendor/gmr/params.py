from pathlib import Path

HERE = Path(__file__).parent
ROBOT_XML_DICT = {"unitree_g1": HERE / "unitree_g1/g1_mocap_29dof.xml"}
IK_CONFIG_DICT = {"bvh_lafan1": {"unitree_g1": HERE / "bvh_lafan1_to_g1.json"}}
