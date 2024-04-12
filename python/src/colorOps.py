import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

class colorOps():
    def __init__(self):
        self.rgb_s = 1.0
        self.cmyk_s = 100.0

    def rgb2cmyk(self,rgb:tuple or list): # type: ignore
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        if r==0 and g==0 and b == 0:
            return 0,0,0,self.cmyk_s
        c = 1 - r /self.rgb_s
        m = 1 - g /self.rgb_s
        y = 1 - b /self.rgb_s
        min_cmy = min([c,m,y])
        c = c - min_cmy
        m = m - min_cmy
        y = y - min_cmy
        k = min_cmy
        out = [self.cmyk_s* i for i in [c,m,y,k]]
        return tuple(out)

    def cmyk2rgb(self,cmyk: tuple or list): # type: ignore
        c = cmyk[0]
        m = cmyk[1]
        y = cmyk[2]
        k = cmyk[3]
        r = self.rgb_s * (1.0 - c / float(self.cmyk_s)) * (1.0 - k / float(self.cmyk_s))
        g = self.rgb_s * (1.0 - m / float(self.cmyk_s)) * (1.0 - k / float(self.cmyk_s))
        b = self.rgb_s * (1.0 - y / float(self.cmyk_s)) * (1.0 - k / float(self.cmyk_s))
        out = [i/self.rgb_s for i in [r,g,b]]
        return tuple(out)
    

if __name__ == "__main__":
    imPath = r"C:\Users\nbrys\Downloads\cmyk_blue_test.png"
    im = Image.open(imPath)
    print('running test colors')
    cmyk_col = (52,3.83,0,0)
    cw = colorOps()
    rgb_out = cw.cmyk2rgb(cmyk_col)
    print(rgb_out)
    cmyk_out = cw.rgb2cmyk(rgb_out)
    print(cmyk_out)

    fig, ax = plt.subplots(1,1)
    ax.imshow(im)
    # Create a Rectangle patch
    rect = patches.Rectangle((50, 100), 40, 30, linewidth=1, edgecolor=rgb_out, facecolor=rgb_out)

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()