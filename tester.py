import sys
import time
import math
import random
import hashlib
import json
import platform
import traceback
import os

# automatic timeout handling will only be performed on Unix
if platform.system() != 'Windows':
    import signal

    WINDOWS = False
else:
    WINDOWS = True

from constants import *
from environment import Environment
from control.state import State as ControlState
from control.environment import Environment as ControlEnvironment


"""
Tester script. Multiprocessing Version.

Use this script to evaluate your solution. You may modify this file if desired. When submitting to GradeScope, an
unmodified version of this file will be used to evaluate your code.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 15/08/22
"""

THREADS = 1
DEBUG_MODE = False      # set this to True to disable time limit checking (useful when running in a debugger)

TC_PREFIX = 'testcases/ex'
TC_SUFFIX = '.txt'
FORCE_VALID = True
DISABLE_TIME_LIMITS = True

TIMEOUT = 35 * 60  # timeout after 35 minutes
VALIDATION_SET_SIZE = 20
VALIDATION_SET_LOOKAHEAD = 100
VISUALISE_TIME_PER_STEP = 0.7

ACTION_READABLE = {FORWARD: 'Forward', REVERSE: 'Reverse', SPIN_LEFT: 'Spin Left',
                   SPIN_RIGHT: 'Spin Right'}

POINTS_PER_TESTCASE = 10.0
MINIMUM_MARK_INCREMENT = 0.1

# === scoring params ===
COMPLETION_POINTS = 2.5
REWARD_POINTS = 2.5
ITERATIONS_POINTS = 2.5
TIMING_POINTS = 2.5

REWARD_SCALING = 1.3
ITERATIONS_SCALING = 1.3
TIMING_SCALING = 2.0


class TimeOutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeOutException


def stable_hash(x):
    return hashlib.md5(str(x).encode('utf-8')).hexdigest()


def state_stable_hash(s: ControlState):
    return stable_hash(str((s.robot_posit, s.robot_orient, s.widget_centres, s.widget_orients)))


def mcts_plan(solver, state):
    t0 = time.time()
    while time.time() - t0 < solver.environment.online_time_tgt:
        solver.mcts_simulate(state)
    return solver.mcts_select_action(state)


def print_usage():
    print("Usage: python tester.pyc [testcases] [-v (optional)]")
    print("    testcases = a comma separated list of numbers (e.g. '1,3,4')")
    print("    if -v is specified, the solver's trajectory will be visualised")


def compute_score(points, scaling, actual, target):
    return points * (1.0 - min(max(actual - target, 0) / (scaling * target), 1.0))


def round_to_increment(score):
    return math.ceil(score * (1 / MINIMUM_MARK_INCREMENT)) / (1 / MINIMUM_MARK_INCREMENT)


def update_logfile(filename, tc_idx, total_score, max_score, tests):
    total_score = math.ceil(total_score * (1 / MINIMUM_MARK_INCREMENT)) / (1 / MINIMUM_MARK_INCREMENT)
    msg0 = '\n\n=== Summary ============================================================'
    msg1 = f'Testcases: {tc_idx}'
    msg2 = f'Total Score: {round(total_score, 1)} (out of max possible score {max_score})'
    log_data = {"output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n', "tests": tests}
    with open(filename, 'w') as outfile:
        json.dump(log_data, outfile)


def run_test_mp(filename_i_vis):
    """
    Run test for a single search type, testcase index pair.
    :param filename_i_vis: (filename, testcase index, visualise)
    :return: thread_id, test_result, leaderboard_result (None if not applicable)
    """
    filename, i, vis = filename_i_vis
    msg0 = f'=== Testcase {i} ============================================================'

    try:
        from solution import Solver
    except ModuleNotFoundError:
        msg1 = "/!\\ There was an error importing your Solver module. Please ensure:\n" \
               "    * You have uploaded the individual files you've modified (e.g. solution.py, etc) and not the " \
               "entire project directory\n" \
               "    * You are not importing any packages which use a GUI (e.g. tkinter, turtle)"
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
        return test_result, None

    if not WINDOWS:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIMEOUT)

    t0 = time.time()
    env = Environment(filename)
    t_env = time.time() - t0
    if t_env > 0.1 and not DISABLE_TIME_LIMITS:
        msg1 = '/!\\ Your Environment __init__ method appears to be taking a long time to complete. Make sure any ' \
               'expensive computations (e.g. performing value iteration) are in your Solver class.'
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
        return test_result, None
    control_env = ControlEnvironment(filename)
    t0 = time.time()
    try:
        solver = Solver(env)
    except BaseException as e:
        msg1 = f'/!\\ Program crashed while Solver was being initialised on testcase {i}.'
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        if int(sys.version.split('.')[1]) <= 9:
            err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
        else:
            err = ''.join(traceback.format_exception(e))
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
        return test_result, None
    t_init = time.time() - t0
    if t_init > 0.001 and not DISABLE_TIME_LIMITS:
        msg1 = '/!\\ Your __init__ method appears to be taking a long time to complete. Make sure any expensive ' \
               'computations (e.g. performing value iteration) are in vi_iteration/pi_iteration/mcts_simulate ' \
               'instead of __init__.'
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
        return test_result, None

    # === plan offline =================================================================================================
    # construct validation set
    val_states = []
    for i in range(VALIDATION_SET_SIZE):
        temp_state = control_env.get_init_state()
        for j in range(VALIDATION_SET_LOOKAHEAD):
            random.seed(stable_hash((i, j)))
            temp_action = random.choice(ROBOT_ACTIONS)
            _, temp_state = control_env.perform_action(temp_state, temp_action)
        val_states.append(temp_state)

    t0 = time.time()
    try:
        if control_env.solve_type == 'vi':
            try:
                solver.vi_initialise()
            except BaseException as e:
                msg1 = f'/!\\ Program crashed in solver.vi_initialise() on testcase {i}.'
                msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                if int(sys.version.split('.')[1]) <= 9:
                    err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                else:
                    err = ''.join(traceback.format_exception(e))
                test_result = {"score": 0,
                               "max_score": POINTS_PER_TESTCASE,
                               "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                return test_result, None

            iterations = 0
            vs_values = {vs: 0.0 for vs in val_states}
            while not solver.vi_is_converged():
                # read values for states in validation set for this iteration
                try:
                    for vs in val_states:
                        vs_values[vs] = solver.vi_get_state_value(vs)
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.vi_get_state_values() on testcase {i}. Make sure ' \
                           f'your solver.vi_get_state_values() method returns 0 for states which have not had ' \
                           f'V(s) computed.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

                # run an iteration
                try:
                    solver.vi_iteration()
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.vi_iteration() on testcase {i}'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

                iterations += 1

            # check for convergence
            for vs in val_states:
                try:
                    if abs(vs_values[vs] - solver.vi_get_state_value(vs)) > (control_env.epsilon * 1.1):
                        msg1 = '/!\\ Your value iteration terminated before convergence is reached. Make sure ' \
                               'that your solver.vi_is_converged() method is working as intended.'
                        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                        test_result = {"score": 0,
                                       "max_score": POINTS_PER_TESTCASE,
                                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
                        return test_result, None
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.vi_get_state_values() on testcase {i}. Make sure ' \
                           f'your solver.vi_get_state_values() method returns 0 for states which have not had ' \
                           f'V(s) computed.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None
            # convergence check passed
        else:
            try:
                solver.pi_initialise()
            except BaseException as e:
                msg1 = f'/!\\ Program crashed in solver.pi_initialise() on testcase {i}.'
                msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                if int(sys.version.split('.')[1]) <= 9:
                    err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                else:
                    err = ''.join(traceback.format_exception(e))
                test_result = {"score": 0,
                               "max_score": POINTS_PER_TESTCASE,
                               "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                return test_result, None

            iterations = 0
            vs_policy = {vs: FORWARD for vs in val_states}
            while not solver.pi_is_converged():
                # read values for states in validation set for this iteration
                try:
                    for vs in val_states:
                        vs_policy[vs] = solver.pi_select_action(vs)
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.pi_select_action() on testcase {i}. Make sure ' \
                           f'your solver.pi_select_action() method returns FORWARD for states which have not had ' \
                           f'pi(s) computed.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

                # run an iteration
                try:
                    solver.pi_iteration()
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.pi_iteration() on testcase {i}'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

                iterations += 1

            # check for convergence
            for vs in val_states:
                try:
                    if vs_policy[vs] != solver.pi_select_action(vs):
                        msg1 = '/!\\ Your value iteration terminated before convergence is reached. Make sure ' \
                               'that your solver.pi_is_converged() method is working as intended.'
                        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                        test_result = {"score": 0,
                                       "max_score": POINTS_PER_TESTCASE,
                                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
                        return test_result, None
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.pi_select_action() on testcase {i}. Make sure ' \
                           f'your solver.pi_select_action() method returns FORWARD for states which have not had ' \
                           f'pi(s) computed.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None
            # convergence check passed
    except TimeOutException:
        msg1 = f'/!\\ Program exceeded the maximum allowed time ({TIMEOUT // 60} minutes) in ' \
               f'solver.{control_env.solve_type}_plan_offline() and was terminated.'
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
        return test_result, None
    except BaseException as e:
        msg1 = f'/!\\ Program crashed in solver.{control_env.solve_type}_plan_offline() on testcase {i}'
        msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
        if int(sys.version.split('.')[1]) <= 9:
            err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
        else:
            err = ''.join(traceback.format_exception(e))
        test_result = {"score": 0,
                       "max_score": POINTS_PER_TESTCASE,
                       "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
        return test_result, None
    t_offline = time.time() - t0

    # === simulate episode =============================================================================================
    t_online_max = 0
    total_reward = 0
    persistent_state = control_env.get_init_state()
    visit_count = {persistent_state: 1}
    if vis:
        control_env.render(persistent_state)
        time.sleep(VISUALISE_TIME_PER_STEP)

    while not control_env.is_solved(persistent_state) and total_reward > (control_env.reward_tgt * 10000):
        # query solver to select an action
        if not WINDOWS and not DEBUG_MODE:
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(control_env.online_time_tgt + 1))
        try:
            t0 = time.time()
            if control_env.solve_type == 'vi':
                try:
                    action = solver.vi_select_action(persistent_state)
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.vi_select_action() on testcase {i}.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

            else:   # control_env.solve_type == 'pi'
                try:
                    action = solver.pi_select_action(persistent_state)
                except BaseException as e:
                    msg1 = f'/!\\ Program crashed in solver.pi_select_action() on testcase {i}.'
                    msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
                    if int(sys.version.split('.')[1]) <= 9:
                        err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
                    else:
                        err = ''.join(traceback.format_exception(e))
                    test_result = {"score": 0,
                                   "max_score": POINTS_PER_TESTCASE,
                                   "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
                    return test_result, None

            t_online = time.time() - t0
            if not WINDOWS and not DEBUG_MODE:
                signal.alarm(0)
            if t_online > t_online_max:
                t_online_max = t_online
        except TimeOutException:
            m = control_env.solve_type + "_select_action()"
            msg1 = f'/!\\ Program exceeded the maximum allowed time ({TIMEOUT // 60} minutes) in ' \
                   f'solver.{m} and was terminated.'
            msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
            test_result = {"score": 0,
                           "max_score": POINTS_PER_TESTCASE,
                           "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
            return test_result, None
        except BaseException as e:
            m = control_env.solve_type + "_select_action()"
            msg1 = f'/!\\ Program crashed in solver.{m} on testcase {i}'
            msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
            if int(sys.version.split('.')[1]) <= 9:
                err = ''.join(traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__))
            else:
                err = ''.join(traceback.format_exception(e))
            test_result = {"score": 0,
                           "max_score": POINTS_PER_TESTCASE,
                           "output": msg0 + '\n' + msg1 + '\n' + err + '\n' + msg2 + '\n'}
            return test_result, None

        if action not in ROBOT_ACTIONS:
            m = control_env.solve_type + "_select_action()"
            msg1 = f'/!\\ Unrecognised action returned by {m}.'
            msg2 = f'\nTestcase total score: 0.0 / {POINTS_PER_TESTCASE}'
            test_result = {"score": 0,
                           "max_score": POINTS_PER_TESTCASE,
                           "output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n'}
            return test_result, None

        # simulate outcome of action
        seed = (str(control_env.episode_seed) + state_stable_hash(persistent_state) +
                stable_hash(visit_count[persistent_state]))
        reward, persistent_state = control_env.perform_action(persistent_state, action, seed=seed)

        # updated visited state count (for de-randomisation)
        visit_count[persistent_state] = visit_count.get(persistent_state, 0) + 1

        # update episode reward
        total_reward += reward

        if vis:
            print(f'\nSelected: {ACTION_READABLE[action]}')
            print(f'Received a reward value of {reward}')

            control_env.render(persistent_state)
            time.sleep(VISUALISE_TIME_PER_STEP)

    # assign scores based on iterations, time to solve, total reward
    completion_score = COMPLETION_POINTS if control_env.is_solved(persistent_state) else 0.0

    reward_score = compute_score(REWARD_POINTS, REWARD_SCALING,
                                 -total_reward, -control_env.reward_tgt)
    reward_score = round_to_increment(reward_score)

    timing_score = compute_score(TIMING_POINTS, TIMING_SCALING,
                                 t_offline, control_env.offline_time_tgt)
    timing_score = round_to_increment(timing_score)

    iterations_score = compute_score(ITERATIONS_POINTS, ITERATIONS_SCALING,
                                     iterations, control_env.iterations_tgt)
    iterations_score = round_to_increment(iterations_score)

    tc_total_score = completion_score + reward_score + timing_score + iterations_score

    msg1 = f'Agent successfully reached the goal ' \
           f'--> Score: {round(completion_score, 1)} / {COMPLETION_POINTS}'
    msg2 = f'Total Reward: {total_reward},    Target: {control_env.reward_tgt}  ' \
           f'--> Score: {round(reward_score, 1)} / {REWARD_POINTS}'
    msg3 = f'Time Elapsed: {t_offline},    Target: {control_env.offline_time_tgt}  ' \
           f'--> Score: {round(timing_score, 1)} / {TIMING_POINTS}'
    msg4 = f'Iterations Performed: {iterations},    Target: {control_env.iterations_tgt}  ' \
           f'--> Score: {round(iterations_score, 1)} / {ITERATIONS_POINTS}'
    msg5 = f'\nTestcase total score: {tc_total_score} / {POINTS_PER_TESTCASE}'
    test_result = {"score": tc_total_score,
                   "max_score": POINTS_PER_TESTCASE,
                   "output": (msg0 + '\n' + msg1 + '\n' + msg2 + '\n' + msg3 + '\n' + msg4 + '\n' +
                              msg5 + '\n')}

    return test_result, None


def main(arglist):
    # parse command line arguments
    if len(arglist) != 1 and len(arglist) != 2 and len(arglist) != 3:
        print("Run this script to test and evaluate the performance of your code.\n")
        print_usage()
        return

    try:
        tc_idx = [int(i) for i in arglist[0].split(',')]
    except ValueError:
        print("Invalid testcases list given.")
        print_usage()
        return

    visualise = False
    write_logfile = False
    results_filename = None
    i = 1
    while i < len(arglist):
        if arglist[i] == '-v':
            visualise = True
            i += 1
        elif arglist[i] == '-l':
            assert len(arglist) > i + 1, '/!\\ write_logfile is enabled but no filename is given'
            write_logfile = True
            results_filename = arglist[i + 1]
            i += 2
        else:
            print("Unrecognised command line argument given.")
            print_usage()
            return

    total_score = 0.0
    max_score = POINTS_PER_TESTCASE * len(tc_idx)
    tests = []
    leaderboard = []
    if THREADS == 1 or visualise:  # run sequentially if visualise is enabled
        # loop over all selected testcases
        for i in tc_idx:
            tc_filename = TC_PREFIX + str(i) + TC_SUFFIX

            test_result, leaderboard_result = run_test_mp((tc_filename, i, visualise))
            tests.append(test_result)
            if leaderboard_result is not None:
                leaderboard.append(leaderboard_result)
            total_score += test_result['score']
    else:  # run in parallel otherwise
        from multiprocessing import Pool
        inputs = [(TC_PREFIX + str(i) + TC_SUFFIX, i, False) for i in tc_idx]
        with Pool(THREADS) as p:
            results = p.map(run_test_mp, inputs)
        for t, lb in results:
            tests.append(t)
            # if lb is not None:
            #     leaderboard.append(lb)
            total_score += t['score']

    # print output for each test in order
    for t in tests:
        print(t['output'])

    # generate summary and write results file
    total_score = math.ceil(total_score * (1 / MINIMUM_MARK_INCREMENT)) / (1 / MINIMUM_MARK_INCREMENT)
    msg0 = '\n\n=== Summary ============================================================'
    msg1 = f'Testcases: {tc_idx}'
    msg2 = f'Total Score: {round(total_score, 1)} (out of max possible score {max_score})'
    log_data = {"output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n', "tests": tests}
    # log_data = {"output": msg0 + '\n' + msg1 + '\n' + msg2 + '\n', "tests": tests, "leaderboard": leaderboard}
    print(log_data['output'])
    if write_logfile:
        with open(results_filename, 'w') as outfile:
            json.dump(log_data, outfile)


if __name__ == '__main__':
    main(sys.argv[1:])
