import matplotlib.pyplot as plt

# Subplots helper: hide axes, minimize space between, maximize window
def cleanSubplots(r, c, pad=0.05):
    f, ax = plt.subplots(r, c)
    if r == 1 or c == 1:
        for a in ax:
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
    else:
        for aRow in ax:
            for a in aRow:
                a.get_xaxis().set_visible(False)
                a.get_yaxis().set_visible(False)

    f.subplots_adjust(left=pad, right=1.0-pad, top=1.0-pad, bottom=pad, hspace=pad)
    plt.get_current_fig_manager().window.showMaximized()
    return ax
