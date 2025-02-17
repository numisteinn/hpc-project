import matplotlib.pyplot as plt


def prep_figure():
    global grid, ax1, ax2
    grid = plt.GridSpec(3, 1, wspace=0.0, hspace=0.3)
    ax1 = plt.subplot(grid[0:2, 0])
    ax2 = plt.subplot(grid[2, 0])


def plot_state(step, t_all, pos_save, KE_save, PE_save):
    plt.sca(ax1)
    plt.cla()
    xx = pos_save[:, 0, max(step - 50, 0) : step]
    yy = pos_save[:, 1, max(step - 50, 0) : step]
    plt.scatter(xx, yy, s=1, color=[0.7, 0.7, 1])
    plt.scatter(pos_save[:, 0, step], pos_save[:, 1, step], s=10, color="blue")
    ax1.set(xlim=(-2, 2), ylim=(-2, 2))
    ax1.set_aspect("equal", "box")
    ax1.set_xticks([-2, -1, 0, 1, 2])
    ax1.set_yticks([-2, -1, 0, 1, 2])

    plt.sca(ax2)
    plt.cla()
    plt.scatter(t_all, KE_save, color="red", s=1, label="KE")
    plt.scatter(t_all, PE_save, color="blue", s=1, label="PE")
    plt.scatter(t_all, KE_save + PE_save, color="black", s=1, label="Etot")
    ax2.set(xlim=(0, t_all[-1]), ylim=(-300, 300))
    ax2.set_aspect(0.007)

    plt.pause(0.001)


def plot_finalize(path):
    plt.sca(ax2)
    plt.xlabel("time")
    plt.ylabel("energy")
    ax2.legend(loc="upper right")
    plt.savefig(path, dpi=240)
