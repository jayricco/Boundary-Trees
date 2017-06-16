import kivy
import numpy as np
import random
from BoundaryTree import BoundaryTree
from PIL import Image
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.core.image import Texture
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.config import Config
from kivy.clock import Clock
from array import array
kivy.require('1.10.0')


def generate_image(xsz, ysz, r):
    img = Image.new('RGB', (xsz, ysz), (255, 255, 255))
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


class GroundTruthWidget(Widget):

    def __init__(self, **kwargs):
        super(GroundTruthWidget, self).__init__(**kwargs)
        self.view_size=kwargs['view_size']
        self.img_size=kwargs['img_size']
        self.sample_event = None
        self.boundary_tree = None
        self.pointer = None
        self.gt_rect = None
        self.tt_rect = None
        self.gt_tex = None
        self.tt_tex = None
        self.bt_len_last = None
        self.iter_label = None

        self.mul = (self.view_size[0]/self.img_size[0], self.view_size[1]/self.img_size[1])

        self.lines = {}
        mgx, mgy = np.meshgrid(range(self.img_size[0]), range(self.img_size[1]))
        self.img_mesh = (mgx, mgy)
        self.sample_list = zip(mgx.flatten(), mgy.flatten())
        random.shuffle(self.sample_list)
        self.sl_index = 0
        self.gt_img = generate_image(self.img_size[0], self.img_size[1], 9)

        self.gt_tex = Texture.create(size=self.view_size, colorfmt='rgb')
        self.tt_tex = Texture.create(size=self.view_size, colorfmt='rgb')
        gt_buf = array('B')
        tt_buf = array('B')
        gt_resize = self.gt_img.resize(size=self.view_size, resample=Image.NEAREST)
        for t in gt_resize.getdata():
            gt_buf.extend([t[0], t[1], t[2]])
            tt_buf.extend([255, 255, 255])
        self.gt_tex.blit_buffer(gt_buf, colorfmt='rgb', bufferfmt='ubyte')
        self.tt_tex.blit_buffer(tt_buf, colorfmt='rgb', bufferfmt='ubyte')

        with self.canvas:
            self.gt_rect = Rectangle(texture=self.gt_tex, pos=(0, 0), size=self.view_size)
            self.tt_rect = Rectangle(texture=self.tt_tex, pos=(self.view_size[0], 0), size=self.view_size)
            self.iter_label = Label(text='Iteration: %d/%d' % (self.sl_index, len(self.sample_list)))
            self.iter_label.color = (0, 0, 0)
            self.iter_label.pos = (self.size[0]*0.05, self.size[1]*0.80)


    def sample_gt(self, dt):
        if self.pointer is not None:
            self.canvas.children.remove(self.pointer)
        if self.sl_index == len(self.sample_list):
            Clock.unschedule(self.event)
            return

        sample_pos = np.asarray(self.sample_list[self.sl_index])
        pixel_val = np.asarray(self.gt_img.getpixel(tuple(sample_pos)))

        if self.boundary_tree is None:
            self.boundary_tree = BoundaryTree(k=4, root_data=(sample_pos, pixel_val), img_mesh=self.img_mesh)
        else:
            self.boundary_tree.train(sample_pos, pixel_val)

        with self.canvas:
            Color(255, 0, 0)
            self.pointer = Ellipse(pos=np.multiply(sample_pos, self.mul), size=self.mul)
            self.iter_label.text = 'Iteration: %d/%d' % (self.sl_index, len(self.sample_list))

            if self.sl_index % 100 == 0 or self.sl_index == len(self.sample_list)-1:
                self.tt_rect.texture = self.boundary_tree.blit_to_texture(self.tt_tex, img_size=self.img_size, resample_method=Image.NEAREST)
                self.tt_rect.flag_update()

        if self.bt_len_last is None or len(self.boundary_tree.data) > self.bt_len_last:
            with self.canvas:
                Color(0, 0, 0, 0.32)
                for family in self.boundary_tree.relations.items():
                    (parent_x, parent_y) = np.multiply(self.boundary_tree.data[family[0]][0], self.mul)
                    for child in family[1]:
                        if self.lines.has_key((family[0], child)):
                            continue
                        (child_x, child_y) = np.multiply(self.boundary_tree.data[child][0], self.mul)
                        self.lines[(family[0], child)] = Line(points=[parent_x, parent_y, child_x, child_y], width=1.2)
            self.bt_len_last = len(self.boundary_tree.data)

        self.sl_index += 1

        self.canvas.ask_update()

view_size = (300, 300)
img_size = (100, 100)


class GroundTruthApp(App):
    def build(self):
        view = GroundTruthWidget(size=(view_size[0]*2, view_size[1]), img_size=img_size, view_size=view_size)
        evt = Clock.schedule_interval(view.sample_gt, 1.0/60.0)
        view.event = evt
        return view

if __name__ == "__main__":
    Config.set('graphics', 'width', view_size[0]*2)
    Config.set('graphics', 'height', view_size[1])
    GroundTruthApp().run()

