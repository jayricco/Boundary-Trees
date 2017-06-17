import random
from BoundaryTree import BoundaryTree
from util.boundary_tree_utils import *
from util import generate_image

import kivy
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.core.image import Texture
from kivy.graphics import Color, Rectangle, Ellipse, Line
from kivy.config import Config
from kivy.clock import Clock
kivy.require('1.10.0')

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
        self.max_iters = 10000
        self.epoch = 0.0
        self.epoch_thing = 0
        self.accuracy = 0.0
        self.average_query_time = 0.0
        self.average_depth = 0.0
        self.mul = (self.view_size[0]/self.img_size[0], self.view_size[1]/self.img_size[1])

        self.lines = {}
        mgx, mgy = np.meshgrid(range(self.img_size[0]), range(self.img_size[1]))
        self.img_mesh = np.asarray(zip(mgx.flatten(), mgy.flatten()))
        self.epoch_meter = {}
        for coord in zip(mgx.flatten(), mgy.flatten()):
            self.epoch_meter[tuple(coord)] = 0
        self.epoch_total = float(len(self.epoch_meter.keys()))
        self.iter = 0
        self.gt_img = generate_image(self.img_size, 15)

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
            self.iter_label = Label(text='', font_size='12sp')
            self.iter_label.color = (0, 0, 0)
            self.iter_label.pos = (self.size[0]*0.43, self.size[1]*0.81)

    def sample_gt(self, dt):

        if self.iter == self.max_iters:
            Clock.unschedule(self.event)

        sample_pos = np.asarray((random.randint(0, self.img_size[0]-1), random.randint(0, self.img_size[1]-1)))
        self.epoch_meter[tuple(sample_pos)] += 1
        self.epoch = len([v for v in self.epoch_meter.values() if v > self.epoch_thing]) / self.epoch_total
        if self.epoch >= self.epoch_thing+1:
            self.epoch_thing += 1

        pixel_val = np.asarray(self.gt_img.getpixel(tuple(sample_pos)))
        if self.boundary_tree is None:
            self.boundary_tree = BoundaryTree(k=6, root_data=(sample_pos, pixel_val))
        else:
            (qt, dep) = self.boundary_tree.train(sample_pos, pixel_val)
            self.average_query_time = (self.average_query_time + qt)/2.0
            self.average_depth = (self.average_depth + dep)/2.0

        with self.canvas:
            if self.pointer is not None:
                self.canvas.children.remove(self.pointer)
            Color(255, 0, 0)
            self.pointer = Ellipse(pos=np.multiply(sample_pos, self.mul), size=self.mul)
            self.iter_label.text = 'Iter: %d | Acc: %.2f | Epoch: %.2f | Avg Query Time: %.5fs | Avg Q.Depth: %.4f | Avg Par.Full.: %.2f' % (self.iter, self.accuracy, self.epoch, self.average_query_time, self.average_depth, boundary_tree_fullness(self.boundary_tree))

            if self.iter % 1000 == 0:
                self.accuracy = boundary_tree_accuracy(self.boundary_tree, self.gt_img, self.img_mesh)
                self.tt_rect.texture = boundary_tree_to_texture(self.boundary_tree, self.tt_tex, size=self.img_size, img_mesh=self.img_mesh)
                self.tt_rect.flag_update()

        if self.bt_len_last is None or len(self.boundary_tree.data) > self.bt_len_last:
            with self.canvas:
                Color(0, 0, 0, 0.32)
                for family in self.boundary_tree.relations.items():
                    (parent_x, parent_y) = np.add(np.multiply(self.boundary_tree.data[family[0]][0], self.mul), np.divide(self.mul, 2))
                    for child in family[1]:
                        if self.lines.has_key((family[0], child)):
                            continue
                        (child_x, child_y) = np.add(np.multiply(self.boundary_tree.data[child][0], self.mul), np.divide(self.mul, 2))
                        self.lines[(family[0], child)] = Line(points=[parent_x, parent_y, child_x, child_y], width=1.1)
            self.bt_len_last = len(self.boundary_tree.data)
        self.iter += 1

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

