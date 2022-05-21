import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def stats_plotter(agents, points, total_wins, output_prefix=''):
    """Plot the points distribution for the agent, with mean and standard deviation."""
    num_evaluations = len(points[0])
    colors = ['#348ABD', '#8EBA42']
    for i in range(len(agents)):
        plt.figure(figsize=(10, 6))
        res = plt.hist(points[i], bins=15, edgecolor='black', color=colors[i],
            label=f"{agents[i].name} {i} points")
        plt.title(f"{agents[i].name} {i} won {total_wins[i]/num_evaluations:.2%}")
        plt.vlines(np.mean(points[i]),
            ymin=0,
            ymax=max(res[0])/10,
            label='Points mean',
            color='black',
            linewidth=3)
        plt.vlines([np.mean(points[i]) - np.std(points[i]),
            np.mean(points[i]) + np.std(points[i])],
            ymin=0,
            ymax=max(res[0])/10,
            label='Points mean $\pm$ std',
            color='#E24A33',
            linewidth=3)
        plt.xlim(0, 120)
        plt.legend()
        if output_prefix:
            # if an output path is specified, save the plot
            plt.savefig(f"{output_prefix}_{agents[i].name}")
        else:
            # else show it
            plt.show()
        plt.close()


def evaluate_summary(winners, points, agents, evaluation_dir):
    """Plot the win rate for each agent in a head-to-head match."""
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.bar([0, 1], np.asarray(winners)/sum(winners), 
        edgecolor='#348ABD', color=['coral', 'lightseagreen'])
    plt.ylim(0, 1)
    plt.xticks([0, 1], [ag.name for ag in agents])
    plt.ylabel("% of victories")
    plt.text(0.25, 0.1, f"STD points: {round(np.std(points[0]), 2)}", {"size" : 18},
        horizontalalignment='center', color='black',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.75, 0.1, f"STD points: {round(np.std(points[1]), 2)}", {"size" : 18},
        horizontalalignment='center', color='black',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.25, 0.2, f"MEAN points: {round(np.mean(points[0]), 2)}", {"size" : 18},
        horizontalalignment='center', color='black',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.75, 0.2, f"MEAN points: {round(np.mean(points[1]), 2)}", {"size" : 18},
        horizontalalignment='center', color='black',
        verticalalignment='center', transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.5))
    plt.title(evaluation_dir[evaluation_dir.find('/')+1:])
    plt.savefig(evaluation_dir)
    plt.close()


def training_summary(x, vict_hist, point_hist, labels, FLAGS, evaluation_dir):
    """Track the evolution of training over time 
    in terms of win percentage and average points obtained.
    """
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax[0].set_title(f"Summary of {FLAGS.num_epochs} epochs", {'size' : 21})

    y1 = np.asarray(vict_hist).T[0]/FLAGS.num_evaluations
    y2 = np.asarray(vict_hist).T[1]/FLAGS.num_evaluations
    ax[0].plot(x, y1, linestyle='--', label=labels[0], color='#8EBA42')
    ax[0].plot(x, y2, linestyle='--', label=labels[1], color='#E24A33')
    ax[0].set_ylabel('Win ratio', {'size' : 15})
    ax[0].set_ylim(0,1)
    ax[0].hlines(np.mean(y1), x[0], x[-1], alpha=0.2, color='#8EBA42')
    ax[0].hlines(np.mean(y2), x[0], x[-1], alpha=0.2, color='#E24A33')
    ax[0].legend()

    y1 = np.mean(np.asarray(point_hist)[:,0,:], 1)
    y2 = np.mean(np.asarray(point_hist)[:,1,:], 1)
    ax[1].plot(x, y1, linestyle='--', label=labels[0], color='#8EBA42')
    ax[1].plot(x, y2, linestyle='--', label=labels[1], color='#E24A33')
    ax[1].set_ylabel('Average points obtained', {'size' : 15})
    ax[1].set_xlabel('Epoch', {'size' : 15})
    ax[1].hlines(np.mean(y1), x[0], x[-1], alpha=0.2, color='#8EBA42')
    ax[1].hlines(np.mean(y2), x[0], x[-1], alpha=0.2, color='#E24A33')
    ax[1].legend()

    plt.savefig(evaluation_dir)
    plt.close()


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill=' '):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end='\r')