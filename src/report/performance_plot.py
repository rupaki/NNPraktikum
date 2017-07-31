import matplotlib.pyplot as plt


class PerformancePlot(object):
    '''
    Class to plot the performances
    Very simple and badly-styled
    '''

    def __init__(self, name):
        '''
        Constructor
        '''
        self.name = name

    def draw_performance_epoch(self, ax, performances, epochs):
        ax.plot(range(epochs), performances, 'k',
                 range(epochs), performances, 'ro')
        ax.set_title(self.name)
        ax.set_ylim(ymax=1)
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Epoch")
        #plt.show()
