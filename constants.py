import string
"""
constants.py

This file contains constants used by the Environment and State classes.

COMP3702 2022 Assignment 1 Support Code

Last updated by njc 29/07/22
"""

# === Robot Orientations ===============================================================================================
# Possible orientations for the robot. '*' indicates the front side of the robot
#
#    UP          DOWN        UP_LEFT     UP_RIGHT     DOWN_LEFT   DOWN_RIGHT
#   ____         ____         ____         ____         ____         ____
#  /    \       /    \       /    \       /    \       /    \       /    \
# /  *   \     /  |   \     /  *   \     /   *  \     /   /  \     /  \   \
# \  |   /     \  *   /     \   \  /     \  /   /     \  *   /     \   *  /
#  \____/       \____/       \____/       \____/       \____/       \____/
#

ROBOT_UP = 'U.'
ROBOT_DOWN = 'D.'
ROBOT_UP_LEFT = 'UL'
ROBOT_UP_RIGHT = 'UR'
ROBOT_DOWN_LEFT = 'DL'
ROBOT_DOWN_RIGHT = 'DR'
ROBOT_ORIENTATIONS = [ROBOT_UP, ROBOT_DOWN, ROBOT_UP_LEFT, ROBOT_UP_RIGHT, ROBOT_DOWN_LEFT, ROBOT_DOWN_RIGHT]

# ===Robot Actions ====================================================================================================
FORWARD = 0
REVERSE = 1
SPIN_LEFT = 2
SPIN_RIGHT = 3
ROBOT_ACTIONS = [FORWARD, REVERSE, SPIN_LEFT, SPIN_RIGHT]
# total_cost = base_cost + push_cost if robot is pushing or pulling a widget, else total_cost = base_cost
ACTION_BASE_COST = {FORWARD: 1.0, REVERSE: 1.0, SPIN_LEFT: 0.1, SPIN_RIGHT: 0.11}
ACTION_PUSH_COST = {FORWARD: 0.8, REVERSE: 0.5, SPIN_LEFT: 0.0, SPIN_RIGHT: 0.0}

# === Widget Types =====================================================================================================
# Possible widget types. The type of an individual widget always stays the same.
WIDGET3 = '3'
WIDGET4 = '4'
WIDGET5 = '5'
WIDGET_TYPES = [WIDGET3, WIDGET4, WIDGET5]
WIDGET_ORIENTS = dict()
WIDGET_SYMBOLS = []
# Widget movement types.
TRANSLATE = 0
SPIN_CW = 1
SPIN_CCW = 2
WIDGET_MOVE_TYPES = [TRANSLATE, SPIN_CW, SPIN_CCW]


# === 3-Widget Orientations ============================================================================================
# Possible orientations for the 3-tile widget. 'X' indicates centre of mass.
#
# VERTICAL
#   ____
#  /    \          SLANT_RIGHT              SLANT_LEFT
# /      \                  ____        ____
# \      /                 /    \      /    \
#  \____/             ____/      \    /      \____
#  /    \            /    \      /    \      /    \
# /  \/  \      ____/  \/  \____/      \____/  \/  \____
# \  /\  /     /    \  /\  /                \  /\  /    \
#  \____/     /      \____/                  \____/      \
#  /    \     \      /                            \      /
# /      \     \____/                              \____/
# \      /
#  \____/
#
VERTICAL = 'V'
SLANT_LEFT = 'L'
SLANT_RIGHT = 'R'
WIDGET3_ORIENTATIONS = [VERTICAL, SLANT_LEFT, SLANT_RIGHT]
WIDGET_ORIENTS[WIDGET3] = WIDGET3_ORIENTATIONS
WIDGET_SYMBOLS += [WIDGET3 + ori for ori in WIDGET3_ORIENTATIONS]

# === 4-Widget Orientations ============================================================================================
# Possible orientations for the 4-tile widget. 'X' indicates centre of mass.
#
#            UP                   DOWN
#           ____            ____        ____
#          /    \          /    \      /    \
#         /      \        /      \____/      \
#         \      /        \      /    \      /
#          \____/          \____/  \/  \____/
#          /    \               \  /\  /
#     ____/  \/  \____           \____/
#    /    \  /\  /    \          /    \
#   /      \____/      \        /      \
#   \      /    \      /        \      /
#    \____/      \____/          \____/
#
UP = 'U'
DOWN = 'D'
WIDGET4_ORIENTATIONS = [UP, DOWN]
WIDGET_ORIENTS[WIDGET4] = WIDGET4_ORIENTATIONS
WIDGET_SYMBOLS += [WIDGET4 + ori for ori in WIDGET4_ORIENTATIONS]

# === 5-Widget Orientations ============================================================================================
# Possible orientations for the 5-tile widget. 'X' indicates centre of mass.
#
#                              SLANT_RIGHT             SLANT_LEFT
#                                 ____                    ____
#      HORIZONTAL                /    \                  /    \
#   ____        ____            /      \____        ____/      \
#  /    \      /    \           \      /    \      /    \      /
# /      \____/      \           \____/      \    /      \____/
# \      /    \      /           /    \      /    \      /    \
#  \____/  \/  \____/       ____/  \/  \____/      \____/  \/  \____
#  /    \  /\  /    \      /    \  /\  /                \  /\  /    \
# /      \____/      \    /      \____/                  \____/      \
# \      /    \      /    \      /    \                  /    \      /
#  \____/      \____/      \____/      \                /      \____/
#                               \      /                \      /
#                                \____/                  \____/
#
HORIZONTAL = 'H'
WIDGET5_ORIENTATIONS = [HORIZONTAL, SLANT_LEFT, SLANT_RIGHT]
WIDGET_ORIENTS[WIDGET5] = WIDGET5_ORIENTATIONS
WIDGET_SYMBOLS += [WIDGET5 + ori for ori in WIDGET5_ORIENTATIONS]

# === Other Symbols ====================================================================================================
FREE_SPACE = '  '
TARGET = 'TT'
OBSTACLE = 'XX'
HAZARD = '!!'
ENVIRONMENT_SYMBOLS = [FREE_SPACE, TARGET, OBSTACLE, HAZARD]
IGNORED_SYMBOLS = [2 * c for c in string.ascii_lowercase]   # double lowercase letter is valid but ignored

ALL_VALID_SYMBOLS = ROBOT_ORIENTATIONS + WIDGET_SYMBOLS + ENVIRONMENT_SYMBOLS + IGNORED_SYMBOLS

# === Render Parameters ================================================================================================
RENDER_CELL_TOP_WIDTH = 7
RENDER_CELL_DEPTH = 4
RENDER_CELL_SIDE_WIDTH = RENDER_CELL_DEPTH // 2

