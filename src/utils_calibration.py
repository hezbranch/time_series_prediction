import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

from scipy.special import expit

def plot_binary_clf_calibration_curve_and_histograms(
        y_true=None,
        y_prob=None,
        bins=20,
        info_per_bin=None,
        B=0.03,
        fig_kws=dict(
            figsize=(1.5*2.75, 1.5*4),
            tight_layout=True),
        ):
    if info_per_bin is None:
        info_per_bin = calc_binary_clf_calibration_per_bin(
            y_true, y_prob, bins)

    fig_h = plt.figure(**fig_kws)
    ax_grid = gridspec.GridSpec(
        nrows=4, ncols=1,
        height_ratios=[1, 1, 6, 0.1],
        )
    ax_TP = fig_h.add_subplot(ax_grid[0,0])
    ax_TN = fig_h.add_subplot(ax_grid[1,0])
    ax_cal = fig_h.add_subplot(ax_grid[2,0])

    # Plot calibration curve
    # First, lay down idealized line from 0-1
    unit_grid = np.linspace(0, 1, 10)
    ax_cal.plot(
        unit_grid, unit_grid, 'k--', alpha=0.5)
    # Then, plot actual-vs-expected fractions on top
    mask = np.isfinite(info_per_bin['fracTP_per_bin'])
    ax_cal.plot(
        info_per_bin['xcenter_per_bin'][mask],
        info_per_bin['fracTP_per_bin'][mask],
        'ks-')
    ax_cal.set_ylabel('frac. true positive')
    ax_cal.set_xlabel('predicted proba.')
    ax_cal.set_title("Calibration")

    # Plot TP histogram
    ax_TP.bar(
        info_per_bin['xcenter_per_bin'],
        info_per_bin['countTP_per_bin'] / np.nansum(info_per_bin['countTP_per_bin']),
        width=0.8*info_per_bin['xwidth_per_bin'],
        color='b')
    ax_TP.set_title("Predicted Probas for True Positives")
    ax_TP.set_ylabel("")
    ax_TP.set_ylim([-B, 1.0+B])

    # Plot TN histogram
    ax_TN.bar(
        info_per_bin['xcenter_per_bin'],
        info_per_bin['countTN_per_bin'] / np.nansum(info_per_bin['countTN_per_bin']),
        width=0.9*info_per_bin['xwidth_per_bin'],
        color='r')
    ax_TN.set_title("Predicted Probas for True Negatives")
    ax_TN.set_ylabel("")
    ax_TN.set_ylim([-B, 1.0+B])

    for ax in [ax_cal, ax_TP, ax_TN]:
        ax.set_xlim([-B, 1.0 + B])
    ax_cal.set_ylim([-B, 1.0 + B])

def calc_binary_clf_calibration_per_bin(
        y_true, y_prob,
        bins=10):
    """ 
    """
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError("y_prob has values outside [0, 1]")

    bins = np.asarray(bins)
    if bins.ndim == 1 and bins.size > 1:
        bin_edges = bins
    else:
        bin_edges = np.linspace(0, 1, int(bins) + 1)
    if bin_edges[-1] == 1.0:
        bin_edges[-1] += 1e-8
    assert bin_edges.ndim == 1
    assert bin_edges.size > 2
    nbins = bin_edges.size - 1
    # Assign each predicted probability into one bin
    # from 0, 1, ... nbins
    binids = np.digitize(y_prob, bin_edges) - 1
    assert binids.max() <= nbins
    assert binids.min() >= 0

    count_per_bin = np.bincount(binids, minlength=nbins)
    countTP_per_bin = np.bincount(binids, minlength=nbins, weights=y_true == 1)
    countTN_per_bin = np.bincount(binids, minlength=nbins, weights=y_true == 0)

    # This divide will (and should) yield nan
    # if any bin has no content
    with np.errstate(divide='ignore', invalid='ignore'):
        fracTP_per_bin = countTP_per_bin / np.asarray(count_per_bin, dtype=np.float64)

    info_per_bin = dict(
        count_per_bin=count_per_bin,
        countTP_per_bin=countTP_per_bin,
        countTN_per_bin=countTN_per_bin,
        fracTP_per_bin=fracTP_per_bin,
        xcenter_per_bin=0.5 * (bin_edges[:-1] + bin_edges[1:]),
        xwidth_per_bin=(bin_edges[1:] - bin_edges[:-1]),
        bin_edges=bin_edges,
        )
    return info_per_bin


if __name__ == '__main__':
    prng = np.random.RandomState(0)
    thr_true = prng.rand(100000)
    u_true = 0.65 * prng.randn(100000)
    y_true = np.asarray(expit(u_true) >= thr_true, dtype=np.float32)
    y_prob = expit(u_true)

    bins = 20

    info_per_bin = calc_binary_clf_calibration_per_bin(
        y_true=y_true,
        y_prob=y_prob,
        bins=bins)
    bin_edges = info_per_bin['bin_edges']
    for bb in range(bin_edges.size - 1):
        print("bin [%.2f, %.2f]  count %5d  fracTP %.3f" % (
            bin_edges[bb],
            bin_edges[bb+1],
            info_per_bin['count_per_bin'][bb],
            info_per_bin['fracTP_per_bin'][bb],
            ))

    plot_binary_clf_calibration_curve_and_histograms(
        info_per_bin=info_per_bin)

    plt.show()