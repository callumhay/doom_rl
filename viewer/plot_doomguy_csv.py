from timeit import repeat
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy import stats
import argparse


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Display updated (refreshed) stats for doomguy from the given csv file.")
  parser.add_argument('-csv', '--csv', help="The csv file to plot.", default="")
  parser.add_argument('-refresh', '--refresh', help="Time in milliseconds for each refresh", default=10000)
  parser.add_argument('-last_n', '--last_n', help="Show the last n elements of the data in each graph, -1 to show all", default=-1, type=int)
  parser.add_argument('-trendline', '--trendline', help="Show the trendlines in each plot", default=True, type=bool)
  args = parser.parse_args()
  if len(args.csv) == 0:
    parser.print_usage()
    exit()

  plt.style.use('dark_background')
  fig, subplotAxs = plt.subplots(2,3)

  labels_xy = [
    ["Learning Rate",  0, 0, "aqua"],
    ["Ep Reward",      0, 1, "green"],
    ["Total Reward",   0, 2, "lavender"],
    ["Loss",           1, 0, "plum"],
    ["Q-Value",        1, 1, "turquoise"],
    ["Epsilon",        1, 2, "gold"]
  ]
  plots = []
  for x in range(2):
    for y in range(3):
      idx = x*3+y
      ax = subplotAxs[x][y]
      ax.title.set_text(labels_xy[idx][0])
      plot, = ax.plot([],[], color=labels_xy[idx][3])
      plots.append(plot)

  last_trendline_plots = {}
  def animate(i=-1):
    data = pd.read_csv(args.csv, header=0, skipinitialspace=True)
    x = data['Episode']

    i = 0
    for (title, xIdx, yIdx, colour) in labels_xy:
      y = data[title]
      ax = subplotAxs[xIdx][yIdx]
      #ax.cla()
      plot = plots[i]
      if args.last_n != -1 and len(x) > args.last_n:
        lastX, lastY = x[-args.last_n:], y[-args.last_n:]
        plot.set_data(lastX, lastY)
        if i != -1 and args.trendline and len(x) == len(y):
          z = np.polyfit(lastX, lastY, 2)
          p = np.poly1d(z)
          if title in last_trendline_plots: ax.lines.remove(last_trendline_plots[title])
          trendline, = ax.plot(lastX, p(lastX), color="grey", linewidth=1, linestyle="--")
          last_trendline_plots[title] = trendline
      else:
        plot.set_data(x,y)
    
      ax.relim()
      ax.autoscale()

      i += 1
      
    fig.tight_layout()
    return plots

  ani = FuncAnimation(fig, animate, init_func=animate, interval=args.refresh)
  fig.tight_layout()
  plt.show()


  