from matplotlib import pyplot as plt

def show_image(image, title: str=None):
  fig = plt.figure(frameon=False)
  ax = fig.add_subplot(1, 1, 1)
  ax.axis('off')
  ax.title.set_text(title)
  ax.xaxis.set_visible(False)
  ax.yaxis.set_visible(False)
  ax.imshow(image)
  plt.grid(False)
  plt.show()