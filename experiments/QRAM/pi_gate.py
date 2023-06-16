import numpy as np
import qutip as qt
import h5py
import sys
import os

sys.path.append(os.path.join('experiments/QRAM/qutip_sims'))

from PulseSequence import PulseSequence
from QSwitch import QSwitch
from QutipToPico import QutipToPico as Q2P


filename_pulse = 'pulse_1_2_drives_shorter_time.hdf5'
filename_pulse = os.path.join('experiments/QRAM/pulses', filename_pulse)

f = h5py.File(filename_pulse, "r")
pulses_IQ = np.array(f['pulse'])
tpulses = np.array(f['ts'])

# define the ideal gate

cutoffs = [3, 3, 3, 3]

U_q0 = qt.qeye(cutoffs[0])
U_q1_mat = np.eye(cutoffs[1], dtype=complex)
U_q1_mat[:2, :2] = np.array([[0, 1], [1, 0]])
U_q1 = qt.Qobj(U_q1_mat, dims=[[cutoffs[1]], [cutoffs[1]]])
U_q2 = qt.qeye(cutoffs[2])
U_q3 = qt.qeye(cutoffs[3])

U_goal = qt.tensor(U_q0, U_q1, U_q2, U_q3)

qtp = Q2P(pulses_IQ, tpulses, U_goal, None)

# function to accept pulse and times and return fidelity
def get_fidelities(
    pulse: np.ndarray, 
    times: np.ndarray,
    print_result: bool = False
):
    pulse = pulse.T
    experiment = Q2P(pulse, times, U_goal, None, cutoffs=cutoffs)
    result = experiment.get_fidelity()
    if print_result:
        print(f'Fidelity: {result}')
    return result

# if __name__ == '__main__':
#     # print(pulses_IQ.shape)
#     # print(tpulses.shape)
#     print(get_fidelities(pulses_IQ, tpulses))

