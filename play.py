import sys

from constants import *
from environment import Environment

"""
play.py

Running this file launches an interactive environment simulation. Becoming familiar with the environment mechanics may
be helpful in designing your solution.

The script takes 1 argument, input_filename, which must be a valid testcase file (e.g. one of the provided files in the
testcases directory).

When prompted for an action, press W to move the robot forward, S to move the robot in reverse, A to turn the robot
left (counterclockwise) and D to turn the robot right (clockwise). Use Q to exit the simulation, and R to reset the
environment to the initial configuration.

COMP3702 2022 Assignment 1 Support Code

Last updated by njc 30/07/22
"""


def main(arglist):
    # === handle getchar for each OS ===================================================================================
    try:
        import msvcrt

        def windows_getchar():
            return msvcrt.getch().decode('utf-8')

        getchar = windows_getchar

    except ImportError:
        import tty
        import termios

        def unix_getchar():
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch

        getchar = unix_getchar

    # === run interactive simulation ===================================================================================
    if len(arglist) != 1:
        print("Running this file launches a playable interactive environment session.")
        print("Usage: play.py [input_filename]")
        return

    input_file = arglist[0]

    env = Environment(input_file)
    state = env.get_init_state()
    total_reward = 0

    # run simulation
    while True:
        env.render(state)
        print("Press 'W' to move the robot forward, 'S' to move the robot in reverse, 'A' to turn the robot left "
              "(counterclockwise) and 'D' to turn the robot right (clockwise). Use '[' to exit the simulation, and ']' "
              "to reset the environment to the initial configuration.")

        char = getchar()

        if char == '[':
            print('Exiting simulation.')
            return

        if char == ']':
            print('Resetting environment to initial configuration.')
            state = env.get_init_state()
            continue

        if char in ['w', 'W', 'a', 'A', 's', 'S', 'd', 'D']:
            if char == 'w' or char == 'W':
                action = FORWARD
            elif char == 's' or char == 'S':
                action = REVERSE
            elif char == 'a' or char == 'A':
                action = SPIN_LEFT
            else:   # char == 'd' or char == 'D'
                action = SPIN_RIGHT

            action_readable = {FORWARD: 'Forward', REVERSE: 'Reverse', SPIN_LEFT: 'Spin Left', SPIN_RIGHT: 'Spin Right'}
            print(f'\nSelected: {action_readable[action]}')

            reward, new_state = env.perform_action(state, action)
            print(f'Received a reward value of {reward}')

            total_reward += reward
            state = new_state

            if env.is_solved(state):
                env.render(state)
                print(f'Environment solved with a total reward of {round(total_reward, 1)}!')
                return


if __name__ == '__main__':
    main(sys.argv[1:])

