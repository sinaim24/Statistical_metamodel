import numpy as np
import scipy.io as sio
from scipy.special import erf
from matplotlib import pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial

# --------------------------------------------------------------------------
# Control Parameters : W = (W_1, W_2) = (POA,SPL) where,
# POA                : percentage of open area
# SPL                : Sound Pressure Level
#
# Quantity of Interest (QoI): (R(7,1),V(7,1))  where,
# R                  : Resistance
# V                  : Reactance
# Sampled frequency points  = [500 1000 1500 2000 2500 3000 3500];
# nq : 7
# nw : 2
# N : Number of realizations
# MatR_ww(nw,N)   : Control parameter - Random vector
# Rmean_ww(nw,1)  : mean of MatR_ww
# Rstd_ww(nw,1)   : standard deviation of MatR_ww
# MatR_qq(2*nq,N) : QoI - Random vector
# Rmean_qq(2*nq,1): mean of MatR_qq
# Rstd_qq(2*nq,1) : standard deviation of MatR_qq
# --------------------------------------------------------------------------

nw = 2
nq = 7
Rfreq = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500])  # Frequency sampling points
n = 1 + nw  # For Silverman bandwidth
nb_points = 499  # For creating the grid for conditional confidence intervals

# User defined points for conditioning
MatRww_0 = {'P1': [[0.03, 0.03, 0.045, 0.045], [130, 145, 133, 141]],
            'P2': [[0.075, 0.075, 0.1, 0.1], [133, 141, 130, 145]]}


def compute_conditional_expectation(nb_parallel, N, MatR_ww0tilde, MatR_wwtilde, cow, MatR_qqtilde) -> np.ndarray:
    """
    This Function calculates the conditional expectation w -> E{Q|W = w0} of a random vector Q for a given value of
    the control parameter W = w0.

    :param nb_parallel: Number of cores used for parallel computation
    :return: MatR_ConditionalExpectation(nq,N_0)
    """
    matRdiff_ww0 = np.tile(MatR_ww0tilde[:, nb_parallel], (N, 1)).T - MatR_wwtilde
    matRsum_ww0 = np.sum(matRdiff_ww0 ** 2, axis=0)
    matRsq_ww0 = cow * matRsum_ww0
    matRexp_ww0 = np.exp(-matRsq_ww0)
    matRnum_ww0 = MatR_qqtilde * matRexp_ww0
    Rnum = np.sum(matRnum_ww0, axis=1)
    den = np.sum(matRexp_ww0)
    MatR_ConditionalExpectation = Rnum / den
    return MatR_ConditionalExpectation


def compute_cond_var(nb_parallel, N, MatR_ww0tilde, MatR_wwtilde, cow, MatR_qqtilde) -> np.ndarray:
    """
    This Function calculates the conditional variance w0 -> sigma^2{Q|W = w0} of a random vector Q for a given value of
    the control parameter W = w0.
    :param: nb_parallel - Number of cores used for parallel computation
    :return: MatR_cond_var(nq,N_0)
    """
    matRdiff_ww0 = np.tile(MatR_ww0tilde[:, nb_parallel], (N, 1)).T - MatR_wwtilde
    matRsum_ww0 = np.sum(matRdiff_ww0 ** 2, axis=0)
    matRsq_ww0 = cow * matRsum_ww0
    matRexp_ww0 = np.exp(-matRsq_ww0)
    matRnum_ww0 = (MatR_qqtilde ** 2) * matRexp_ww0
    Rnum = np.sum(matRnum_ww0, axis=1)
    den = np.sum(matRexp_ww0)
    MatR_cond_var = Rnum / den
    return MatR_cond_var


def compute_conditional_confidence_interval(N_0, cow, coq, MatR_wwtilde, MatR_ww0tilde, nb_parallel, R_qq,
                                                   mean_qq, std_qq):
    """
    This function calculates the conditional confidence regions (upper and lower bounds) Q|W = w0 of a random vector Q
    for a given value of the control parameter W = w0.
    :param nb_parallel: The index for the parallel computation
    :param R_qq: Random vector Q for the given index kq
    :param mean_qq: Mean of Q for the given index kq
    :param std_qq: Standard deviation of Q for the given index kq
    :return: Tuple containing (upper bound, lower bound) of the confidence interval
    """

    def sub_routine_conditional_confidence_interval(R_ww0tilde):
        """
        Subroutine to calculate the upper and lower bounds for the conditional confidence interval
        """
        # Normalize the QoI random vector for this iteration
        R_qqtilde = (R_qq - mean_qq) / std_qq  # (1, N)

        # Set up the grid for the conditional CDF computation
        nbpoint_temp = nb_points
        maxq = np.max(R_qq)
        minq = np.min(R_qq)

        # Defining the step size
        stepq = (maxq - minq) / nbpoint_temp
        R_qstar = np.arange(minq, maxq + stepq, stepq)  # (1, nbpoint)

        # Normalize the QoI samples
        R_qstartilde = (R_qstar - mean_qq) / std_qq  # (1, nbpoint)

        # Efficient difference calculation
        MatRexpo_ww0 = MatR_wwtilde - R_ww0tilde[:, np.newaxis]  # (nw, N)
        MatRsum_ww0 = np.exp(-cow * np.sum(MatRexpo_ww0 ** 2, axis=0))  # (1, N)

        # Calculate the denominator of the conditional CDF
        den = np.sum(MatRsum_ww0)

        # Compute the conditional CDF
        MatRqqdiff_temp = R_qstartilde[:, np.newaxis] - R_qqtilde  # (nbpoint, N)
        MatRF = 0.5 + 0.5 * erf(coq * MatRqqdiff_temp)  # (nbpoint, N)

        # Calculate the numerator of the conditional CDF
        num_ib = np.sum(MatRF * MatRsum_ww0, axis=1)  # (nbpoint,)
        Rcdfcond = num_ib / den  # Conditional CDF for each point in `MatRqstar`

        # Determine the upper and lower bounds for the confidence interval
        # Initialize the upper and lower bounds to defaults
        ib_upper = len(R_qstartilde) - 1  # Default to the last index
        ib_lower = 0  # Default to the first index
        pc = 0.98 # Conditional confidence region
        nbpoint = len(R_qstartilde)
        for ib in range(nbpoint):
            if Rcdfcond[ib] >= pc:
                ib_upper = ib
                break

        for ib in range(nbpoint):
            if Rcdfcond[ib] >= (1 - pc):
                ib_lower = ib
                break

        # Check if ib_upper or ib_lower were not set properly
        if ib_upper is None or ib_lower is None:
            pc = 0.96  # If not set, use a lower value for pc
            # Find ib_upper again
            for ib in range(len(R_qstartilde)):
                if Rcdfcond[ib] >= pc:
                    ib_upper = ib
                    break
            # Find ib_lower again
            for ib in range(len(R_qstartilde)):
                if Rcdfcond[ib] >= (1 - pc):
                    ib_lower = ib
                    break

        q_lower = R_qstar[ib_lower]
        q_upper = R_qstar[ib_upper]
        return q_upper, q_lower

    # Initialize arrays to store the upper and lower bounds
    Rcdfcond_upper = np.zeros(N_0)
    Rcdfcond_lower = np.zeros(N_0)

    # Loop over the conditioning points to compute bounds
    for i in range(N_0):
        Rcdfcond_upper[i], Rcdfcond_lower[i] = sub_routine_conditional_confidence_interval(MatR_ww0tilde[:, i])

    return Rcdfcond_upper, Rcdfcond_lower


# Define the task for parallel computation
def parallel_compute_confidence_interval(kq, N_0, cow, coq, MatR_qq, Rmean_qq, Rstd_qq, MatR_wwtilde, MatR_ww0tilde):
    return compute_conditional_confidence_interval(N_0, cow, coq, MatR_wwtilde, MatR_ww0tilde, nb_parallel=kq, R_qq=MatR_qq[kq, :],
                                                   mean_qq=Rmean_qq[kq, 0], std_qq=Rstd_qq[kq, 0])


# Defining a function for plotting graphs
def plot_curves(MatR_xx, Nx, str_label, color_x, title_x, MatRww_0, x_limits, y_limits, saveName):
    for i in range(Nx):  # Plot for each conditioning point
        plt.figure()
        plt.plot(Rfreq, MatR_xx[:, i], color=color_x, label=None)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(str_label)
        plt.title(title_x+f'@ (POA, SPL) = ({MatRww_0[0][i]}, {MatRww_0[1][i]})')
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(saveName+f'_{MatRww_0[0][i]}_{MatRww_0[1][i]}.png')  # Save figure as .png
        plt.show()


# Defining a function for plotting graphs
def plot_confidence_interval(MatR_xx, MatRxx_L, MatRxx_U, Nx, str_label, color_x, title_x, MatRww_0, x_limits, y_limits, saveName):
    for i in range(Nx):  # Plot for each conditioning point
        plt.figure()
        plt.plot(Rfreq, MatR_xx[:, i], color=color_x)
        # Plot the confidence region (shaded area between upper and lower bounds)
        plt.fill_between(Rfreq, MatRxx_L[:, i], MatRxx_U[:, i], color=color_x, alpha=0.2, edgecolor=None, label=None)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(str_label)
        plt.title(title_x+f'@ (POA, SPL) = ({MatRww_0[0][i]}, {MatRww_0[1][i]})')
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(saveName+f'_{MatRww_0[0][i]}_{MatRww_0[1][i]}.png')  # Save figure as .png
        plt.show()


# Define the main function
def main():
    # iterating over the partitions
    for iterPart in range(2):
        # Loading data
        rawData = sio.loadmat('MatRxx_arP'+str(iterPart+1)+'.mat')
        data = np.array(rawData['MatRxxph_ar'])
        _, N = data.shape
        MatR_ww = data[:nw, :]  # First nw rows (for control parameters)
        MatR_qq = data[nw:, :]  # Remaining rows (for QoI)
        del rawData, data
        # MatR_rr = MatR_qq[:nq, :]
        # MatR_vv = MatR_qq[nq:, :]
        Rmean_qq = np.mean(MatR_qq, axis=1, keepdims=True)
        Rstd_qq = np.std(MatR_qq, axis=1, keepdims=True)
        Rmean_ww = np.mean(MatR_ww, axis=1, keepdims=True)
        Rstd_ww = np.std(MatR_ww, axis=1, keepdims=True)
        s = (4 / (N * (2 + n))) ** (1 / (n + 4))  # Silverman bandwidth
        cow = 1 / (2 * s * s)  # Multiplication factor in the formula for conditional expectation E{Q|W=w}
        coq = 1 / (np.sqrt(2) * s)  # Multiplication factor in the formula for conditional confidence interval
        MatRww_0_ = MatRww_0['P'+str(iterPart+1)]
        N_0 = len(MatRww_0_[0])
        # Normalizing
        MatR_qqtilde = (MatR_qq - Rmean_qq) / Rstd_qq
        MatR_wwtilde = (MatR_ww - Rmean_ww) / Rstd_ww
        MatR_ww0tilde = (MatRww_0_ - Rmean_ww) / Rstd_ww

        n_cores = cpu_count()
        with Pool(n_cores) as pool:
            call_compute_conditional_expectation = pool.map(partial(compute_conditional_expectation, N=N, MatR_ww0tilde=MatR_ww0tilde,
                                                                    MatR_wwtilde=MatR_wwtilde, cow=cow, MatR_qqtilde=MatR_qqtilde), range(N_0))
            call_compute_cond_var = pool.map(partial(compute_cond_var, N=N, MatR_ww0tilde=MatR_ww0tilde,
                                                                    MatR_wwtilde=MatR_wwtilde, cow=cow, MatR_qqtilde=MatR_qqtilde), range(N_0))
            call_compute_conditional_confidence_interval = pool.map(partial(parallel_compute_confidence_interval, N_0=N_0, cow=cow, coq=coq,
                                                                            MatR_wwtilde=MatR_wwtilde, MatR_ww0tilde=MatR_ww0tilde, MatR_qq=MatR_qq,
                                                                            Rmean_qq=Rmean_qq, Rstd_qq=Rstd_qq), range(2 * nq))

        # ---Conditional Expectation - Rescaling operation
        MatRmeancond  = np.array(call_compute_conditional_expectation).T
        MatRmeancondR = Rmean_qq[:nq] + MatRmeancond[:nq, :] * Rstd_qq[:nq]
        MatRmeancondV = Rmean_qq[nq:] + MatRmeancond[nq:, :] * Rstd_qq[nq:]

        # ---Conditional Standard Deviation - Rescaling operation
        MatRmeancond2  = np.array(call_compute_cond_var).T  # E{Q^2 | W = w0}
        MatRsigmacondR = Rstd_qq[:nq] * np.sqrt(MatRmeancond2[:nq, :] - (MatRmeancond[:nq, :] ** 2))
        MatRsigmacondV = Rstd_qq[nq:] * np.sqrt(MatRmeancond2[nq:, :] - (MatRmeancond[nq:, :] ** 2))

        # ---Conditional Confidence Intervals
        # Initialize matrices to store the results
        MatRcdfcond_upper = np.zeros((2 * nq, N_0))
        MatRcdfcond_lower = np.zeros((2 * nq, N_0))

        # Collect the results and store them in the upper/lower bound matrices
        for kq, (upper, lower) in enumerate(call_compute_conditional_confidence_interval):
            MatRcdfcond_upper[kq, :], MatRcdfcond_lower[kq, :] = np.array(upper).T, np.array(lower).T

        # Extract the final matrices for resistance (R) and reactance (V)
        MatRcond_RU = MatRcdfcond_upper[:nq, :]
        MatRcond_RL = MatRcdfcond_lower[:nq, :]
        MatRcond_VU = MatRcdfcond_upper[nq:, :]
        MatRcond_VL = MatRcdfcond_lower[nq:, :]

        # Define x-axis (frequency) and y-axis limits for the plots
        x_limits = (500, 3500)  # Frequency range
        y_limits_mean_R = (0, 2.5)  # Y-axis limits for mean
        y_limits_mean_V = (-11, 4)  # Y-axis limits for mean
        y_limits_std = (0, 1)  # Y-axis limits for std deviation

        # ---Plotting graphs for Conditional Standard Deviation
        plot_curves(MatRsigmacondR, N_0, 'Std Resistance', 'blue', 'Conditional Standard Deviation Resistance', MatRww_0_, x_limits, y_limits_std, 'Conditional_std_resist')
        plot_curves(MatRsigmacondV, N_0, 'Std Reactance', 'blue', 'Conditional Standard Deviation Reactance', MatRww_0_, x_limits, y_limits_std, 'Conditional_std_react')

        # ---Plotting graphs and confidence regions for Conditional Expectation
        plot_confidence_interval(MatRmeancondR, MatRcond_RL, MatRcond_RU, N_0, 'Resistance', 'red', 'Conditional Expectation and 98% CI',
                                 MatRww_0_, x_limits, y_limits_mean_R, 'Conditional_mean_CI_resist')
        plot_confidence_interval(MatRmeancondV, MatRcond_VL, MatRcond_VU, N_0, 'Reactance', 'red',
                                 'Conditional Expectation and 98% CI', MatRww_0_, x_limits, y_limits_mean_V, 'Conditional_mean_CI_react')


if __name__ == "__main__":
    main()