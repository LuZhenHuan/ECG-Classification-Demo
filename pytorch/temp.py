import numpy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

confusion = numpy.random.rand(8,8)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion)
fig.colorbar(cax)