import numpy as np
import matplotlib.pyplot as plt
import os.path
from pylab import figure, axes, pie, title, show, savefig

def get_info(dir_name, ep_increment=0.1, ep_range=5):
    real = ("test (real)", np.loadtxt(os.path.join(dir_name, "test (real).txt")))
    advs = [("test (adversarial %f)" % (i*ep_increment), np.loadtxt(os.path.join(dir_name, "test (adversarial %f).txt" % (i*ep_increment)))) for i in range(1,ep_range+1)]
    return [real]+advs

def make_plot(name, infos, write_file):
    plt.clf()
    fig = plt.figure()
    xs = range(len(infos[0][1]))
    for i in infos:
        plt.plot(xs, i[1], label=i[0])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(name)
    plt.legend()
    plt.show()
    #fig = plt.figure()
    savefig(write_file)

def make_plots(names_and_dirs, write_dir):
    for (n, d) in names_and_dirs:
        infos = get_info(d, ep_increment=0.1, ep_range=5)
        print(infos)
        write_file = os.path.join(write_dir, n+".png")
        make_plot(n, infos, write_file)

if __name__=='__main__':
    make_plots([("Single, adversarial training, fgsm 0.3", "single_epochs100_ep0.3"),
                ("Single, adversarial training, fgsm 0.5", "single_epochs100_ep0.5"),
                ("Single, adversarial training, fgsm 1.0", "single_epochs100_ep1"),
                ("Mix 10, pretrained, fgsm 0.3, reg 0.5", "mix10_pretrain1_epochs100_ep0.3_reg0.5")], "plots")
