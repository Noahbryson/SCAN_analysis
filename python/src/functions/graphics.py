import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import cmcrameri.cm as cmc

def three_color_gradient(color1:tuple[float],color2:tuple[float],color3:tuple[float],steps:int=100):
      half_steps = steps // 2
      gradient1 = np.linspace(color1, color2, half_steps, endpoint=False)
      gradient2 = np.linspace(color2, color3, steps - half_steps)
      full_gradient = np.vstack((gradient1, gradient2))
      
      return full_gradient

def circular_gradientN(color1:tuple[float],color2:tuple[float],color3:tuple[float],resolution:float=1,shift=30):
      """
      circular_gradient generates a circular color gradient from three colors. more colors options coming in the future

      Args:
            colors (tuple): all colors to be included in the gradient
            color1 (tuple[float]): rgb color 1
            color2 (tuple[float]): rgb color 2
            color3 (tuple[float]): rgb color 3
            resolution (float, optional): How many degrees per color step to make. Defaults to 1.
            shift (int, optional): degrees from zero to shift the color wheel. Defaults to 30.

      Returns:
            np.ndarray: 360/resolution x 2 numpy array of the color map. 
      """
      N = int(360 / resolution)
      steps = N // 3
      gradient12 = np.linspace(color1, color2, steps, endpoint=False)
      gradient13 = np.linspace(color3, color1, steps, endpoint=False)
      gradient32 = np.linspace(color2, color3, steps, endpoint=False)
      colors = np.vstack([gradient12,gradient32,gradient13])
      shift = int(shift/resolution)
      colors = np.roll(colors,shift)
      return colors

def circular_gradient3(color1:tuple[float],color2:tuple[float],color3:tuple[float],resolution:float=1,shift=30):
      """
      circular_gradient generates a circular color gradient from three colors. more colors options coming in the future

      Args:
            color1 (tuple[float]): rgb color 1
            color2 (tuple[float]): rgb color 2
            color3 (tuple[float]): rgb color 3
            resolution (float, optional): How many degrees per color step to make. Defaults to 1.
            shift (int, optional): degrees from zero to shift the color wheel. Defaults to 30.

      Returns:
            np.ndarray: 360/resolution x 2 numpy array of the color map. 
      """
      N = int(360 * resolution)
      steps = N // 3
      gradient12 = np.linspace(color1, color2, steps, endpoint=False)
      gradient23 = np.linspace(color2, color3, steps, endpoint=False)
      gradient31 = np.linspace(color3, color1, steps, endpoint=False)
      colors = np.vstack([gradient12,gradient23,gradient31])
      shift = int(shift*resolution)
      # shift = int(180/resolution)
      color = np.roll(colors,shift,axis=0)
      return color


import numpy as np
from collections import defaultdict


def adjust_color_contrast(hex_code: str, adjustment: float):
      pass



def colorwheel_standalone(rgb_array)->None:
      plt.figure(figsize=(8, 8))
      ax = plt.subplot(121, projection='polar')

      # Plot the color wheel
      theta = np.linspace(0, 2*np.pi, len(rgb_array))
      ax.scatter(theta, np.ones(len(rgb_array)), c=rgb_array, s=200)

      # Customize plot
      ax.set_xticks([])
      ax.set_yticks([])
      ax.spines['polar'].set_visible(False)
      ax.set_title('360-degree Color Wheel')

      ax2 = plt.subplot(122)
      patches,texts = ax2.pie(np.ones(len(rgb_array)),colors=rgb_array,wedgeprops = {'linewidth': 0})
      for p in patches:
            p.set_edgecolor(p.get_facecolor())
      # ax2.spines['polar'].set_visible(False)

def colorwheel(rgb_array,ax: Axes):
      patches,texts = ax.pie(np.ones(len(rgb_array)),colors=rgb_array,wedgeprops = {'linewidth': 0})
      for p in patches:
            p.set_edgecolor(p.get_facecolor())
      ax.set_xticks([])
      ax.set_yticks([])
      return ax



def hex2RGB(ip: str):
      ip = ip.replace('#','')
      return tuple(int(ip[i:i+2],16)/255 for i in (0, 2, 4))

def targetColorSwatch(targetNames,targetColors,ax:Axes):
      N = len(targetNames)
      xy = np.ones(N)
      ax.bar(targetNames,xy,color=targetColors)
      # for i,j in zip(xy,targetNames):
      #       ax.text(i-.1,i,j)
      return ax


def circle_gradient_key(color_gradient,target_names,target_colors)->Figure:
      # colorwheel_standalone(color_gradient)
      fig = plt.figure()
      a1 = plt.subplot(121)
      a1 = colorwheel(color_gradient,ax=a1)
      a2 = plt.subplot(122)
      a2=targetColorSwatch(target_names,target_colors,ax=a2)
      return fig


def default_gradient(resolution:int,run_circular: bool=False):
      # yellow #FFFF00)
      # Cyan / Aqua	#00FFFF
      # Magenta / Fuchsia	#FF00FF
      
      # c1 = '#A23B96';c2 = '#5BB39C';c3 = '#FAB27A'
      c1 = '#5C60AA';c2 = '#7CC14E';c3 = '#EF575A'
      c1 = '#A23B96';c2 = '#5BB39C';c3 = '#FF9506'
      c1 = '#4363d8';c2 = '#aaffc3';c3 = '#ff00ff'
      c1 = '#00ffff';c2 = '#aaffc3';c3 = '#ff00ff'
      c1 = '#00ffff';c2 = '#FF9506';c3 = '#ff00ff'
      c1 = '#4363d8';c2 = '#3cb44b';c3 = '#e6194B'
      c1 = '#ff00ff';c2 = '#ffff00';c3 = '#00ffff'
      c3 = '#ff00ff';c2 = '#ffff00';c1 = '#00ffff'
      # c1 = '#de24bf';c2 = '#3ba253';c3 = '#f28b0c'
      # c1 = '#FFFF00';c2 = '#00FFFF';c3 = '#FF00FF'
      # c1 = '#FAB27A';c2 = '#00FFFF';c3 = '#FF00FF'
      cmap = cmc.romaO
      c = np.roll(cmap.colors,0)
      locs = [int(30*len(c)/360),int(150*len(c)/360),int(270*len(c)/360)]
      if run_circular:
            c= circular_gradient3(hex2RGB(c1),hex2RGB(c2),hex2RGB(c3),resolution=resolution,shift=30)
      tColors = [c[i,:] for i in locs]
      return c,tColors


def stockRGB(mode: str='hex')-> tuple:
      mode = mode.lower()
      c1 = '#4363d8';c2 = '#3cb44b';c3 = '#e6194B'
      match mode:
            case 'hex':
                  return c1,c2,c3
            case 'rgb':
                  return hex2RGB(c1),hex2RGB(c2),hex2RGB(c3)

if __name__ == '__main__':
      c1 = '#A23B96'
      c2 = '#5BB39C'
      c3 = '#FAB27A'
      
      tNames = ['Hand','Foot','Face']
      
      # c1 = '#ff00ff'; c2 = '#0fff05'; c3 = '#00ffff'
      # colors = circular_gradient3(hex2RGB(c1),hex2RGB(c2),hex2RGB(c3),resolution=5,shift=30)
      # tColors = [hex2RGB(c1),hex2RGB(c2),hex2RGB(c3)]
      # circle_gradient_key(colors,tNames,tColors)
      
      # c1 = '#ff00ff'; c2 = '#ffff00'; c3 = '#00ffff'
      # colors = circular_gradient3(hex2RGB(c1),hex2RGB(c2),hex2RGB(c3),resolution=5,shift=30)
      # tColors = [hex2RGB(c1),hex2RGB(c2),hex2RGB(c3)]
      # # colors = circular_gradient3([.1,.7,.41],hex2RGB(c1),[0,0,1],resolution=5,shift=30)
      # # tColors = [[.1,.7,.41],hex2RGB(c1),[0,0,1]]
      # circle_gradient_key(colors,tNames,tColors)
      # colors = default_gradient(100)
      # circ = default_gradient(1)
      c1 = '#4363d8';c2 = '#3cb44b';c3 = '#e6194B'
      colors = circular_gradient3(hex2RGB(c1),hex2RGB(c2),hex2RGB(c3),resolution=1,shift=30)
      tColors = [hex2RGB(c1),hex2RGB(c2),hex2RGB(c3)]
      # circle_gradient_key(colors,tNames,tColors)
      
      
      cmap = cmc.romaO
      c = np.roll(cmap.colors,-75)
      c = np.roll(cmap.colors,0)
      locs = [int(30*len(c)/360),int(150*len(c)/360),int(270*len(c)/360)]
      tColors = [c[i,:] for i in locs]
      circle_gradient_key(c,tNames,tColors)
      
      # cmap = cmc.bamO
      # c = np.roll(cmap.colors,-30)
      # locs = [int(30*len(c)/360),int(150*len(c)/360),int(270*len(c)/360)]
      # tColors = [c[i,:] for i in locs]
      # circle_gradient_key(c,tNames,tColors)
      
      # cmap = cmc.corkO
      # c = np.roll(cmap.colors,45)
      # locs = [int(30*len(c)/360),int(150*len(c)/360),int(270*len(c)/360)]
      # tColors = [c[i,:] for i in locs]
      # circle_gradient_key(c,tNames,tColors)
      
      fig=plt.figure()
      ax = fig.gca()
      colorwheel(c,ax) 
      
      fig=plt.figure()
      ax = fig.gca()
      colorwheel(default_gradient(1),ax)      
      plt.show()  