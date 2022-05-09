from matplotlib import pyplot as plt
import seaborn as sns
import pickle
import os


def plot_curve(root, training_stats_file):
    objects = []
    with (open(os.path.join(root, training_stats_file), "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
    objects[0].plot(x="epoch", y="loss", ax=ax)
    
    fig.suptitle('SWAV simplifed loss curve', fontsize=20)
    plt.savefig('loss_curve.png')


if __name__ == "__main__":
    training_stats_file = "stats0.pkl"

    plt.style.use('seaborn')
    sns.set(style='whitegrid')
    plot_curve('./experiments', training_stats_file)