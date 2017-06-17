from PIL import Image


def generate_image(size, r):
    img = Image.new('RGB', size, (255, 255, 255))
    xsz, ysz = size
    i_i = (51, 153, 255)
    i_o = (255, 204, 255)
    ii_i = (255, 102, 153)
    ii_o = (255, 255, 153)
    iii_i = (0, 255, 0)
    iii_o = (153, 204, 255)
    iv_o = (204, 255, 153)
    iv_i = (255, 204, 153)

    hx = xsz/2
    hy = ysz/2
    for y in range(ysz):
        for x in range(xsz):
            if x >= hx and y >= hy:
                #Quadrant IV
                h = int(0.75*xsz)
                k = int(0.75*ysz)
                if  (x - h)**2 + (y - k)**2 <= r**2:
                    img.putpixel((x, y), iv_i)
                else:
                    img.putpixel((x, y), iv_o)

            elif x >= hx and y < hy:
                #Quadrant I
                h = int(0.75*xsz)
                k = int(0.25*ysz)
                if (x - h) ** 2 + (y - k) ** 2 <= r ** 2:
                    img.putpixel((x, y), i_i)
                else:
                    img.putpixel((x, y), i_o)

            elif x < hx and y >= hy:
                #Quadrant III
                h = int(0.25*xsz)
                k = int(0.75*ysz)
                if (x - h) ** 2 + (y - k) ** 2 <= r ** 2:
                    img.putpixel((x, y), iii_i)
                else:
                    img.putpixel((x, y), iii_o)

            else:
                #Quadrant II
                h = int(0.25*xsz)
                k = int(0.25*ysz)
                if (x - h) ** 2 + (y - k) ** 2 <= r ** 2:
                    img.putpixel((x, y), ii_i)
                else:
                    img.putpixel((x, y), ii_o)
    return img
