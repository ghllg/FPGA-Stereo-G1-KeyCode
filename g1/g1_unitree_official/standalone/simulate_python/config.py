from pathlib import Path
ROBOT = 'g1'
ROBOT_SCENE = str(Path(__file__).resolve().parent / 'policy_scene.xml')
import os
DOMAIN_ID = int(os.environ.get('G1_DDS_DOMAIN','74'))
INTERFACE = None
USE_JOYSTICK = 0
JOYSTICK_TYPE = 'xbox'
JOYSTICK_DEVICE = 0
PRINT_SCENE_INFORMATION = False
ENABLE_ELASTIC_BAND = False
SIMULATE_DT = 0.001
VIEWER_DT = 0.02
