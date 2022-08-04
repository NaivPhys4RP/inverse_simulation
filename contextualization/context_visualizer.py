#!/usr/bin/env python3

import cv2
import json
import math
import sys
import gi
from graph_tool.all import *
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gio, Gtk, Pango, Gdk
import cairo
import numpy as np
# This would typically be its own file
MENU_XML = """
<?xml version="1.0" encoding="UTF-8"?>
<interface>
  <menu id="app-menu">
    <section>
      <attribute name="label" translatable="yes">Change label</attribute>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 1</attribute>
        <attribute name="label" translatable="yes">String 1</attribute>
      </item>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 2</attribute>
        <attribute name="label" translatable="yes">String 2</attribute>
      </item>
      <item>
        <attribute name="action">win.change_label</attribute>
        <attribute name="target">String 3</attribute>
        <attribute name="label" translatable="yes">String 3</attribute>
      </item>
    </section>
    <section>
      <item>
        <attribute name="action">win.maximize</attribute>
        <attribute name="label" translatable="yes">Maximize</attribute>
      </item>
    </section>
    <section>
      <item>
        <attribute name="action">app.about</attribute>
        <attribute name="label" translatable="yes">_About</attribute>
      </item>
      <item>
        <attribute name="action">app.quit</attribute>
        <attribute name="label" translatable="yes">_Quit</attribute>
        <attribute name="accel">&lt;Primary&gt;q</attribute>
    </item>
    </section>
  </menu>
</interface>
"""

class ImaginationAreaFrame(Gtk.Frame):
    def __init__(self, css=None, border_width=0):
        super().__init__()
        self.set_border_width(border_width)
        self.set_size_request(100, 100)
        self.vexpand = True
        self.hexpand = True
        self.surface = None
        self.min_index=-1
        self.area = Gtk.DrawingArea()
        self.add(self.area)

        self.init_surface(self.area)
        self.context = cairo.Context(self.surface)
        self.area.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.area.add_events(Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.area.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        self.area.connect('button-press-event', self.on_press)
        self.area.connect('button-release-event', self.on_press)
        self.area.connect('motion-notify-event', self.on_press)
        self.area.connect("draw", self.on_draw)
        self.area.connect('configure-event', self.on_configure)

    def set_nearest_vertex(self, x,y,delta=30.0):
        pass


    def on_release(self):
        pass

    def on_press(self, widget, event):
        pass


    def init_surface(self, area):
        # Destroy previous buffer
        if self.surface is not None:
            self.surface.finish()
            self.surface = None

        # Create a new buffer
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.area.get_allocated_width(),self.area.get_allocated_width())

    def redraw(self):
        self.init_surface(self.area)
        context = cairo.Context(self.surface)
        context.scale(self.surface.get_width(), self.surface.get_height())
        self.do_drawing(context)
        self.surface.flush()

    def on_configure(self, area, event, data=None):
        self.redraw()
        return False

    def on_draw(self, area, context):
        if self.surface is not None:
            context.set_source_surface(self.surface, 0.0, 0.0)
            context.paint()
        else:
            print('Invalid surface')
        return False

    def draw_radial_gradient_rect(self, ctx):

        x0, y0 = 0.3, 0.3
        x1, y1 = 0.5, 0.5
        r0 = 0
        r1 = 1
        pattern = cairo.RadialGradient(x0, y0, r0, x1, y1, r1)
        pattern.add_color_stop_rgba(0, 1, 1, 0.5, 1)
        pattern.add_color_stop_rgba(1, 0.2, 0.4, 0.1, 1)
        ctx.rectangle(0, 0, 1, 1)
        ctx.set_source(pattern)
        ctx.fill()

    def do_drawing(self, ctx):
        self.draw_radial_gradient_rect(ctx)



####################################################################################################################################################################################


class OntologyAreaFrame(Gtk.Frame):
    def __init__(self, css=None, border_width=0):
        super().__init__()
        self.set_border_width(border_width)
        self.set_size_request(100, 100)
        self.vexpand = True
        self.hexpand = True
        self.surface = None
        self.min_index=-1
        self.area = Gtk.DrawingArea()
        self.add(self.area)

        self.init_surface(self.area)
        self.context = cairo.Context(self.surface)
        #self.build_graph()
        #graph_tool.draw.cairo_draw(self.g, self.pos, self.context, vertex_text=self.x, vertex_color=self.c)
        self.area.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.area.add_events(Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.area.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        self.area.connect('button-press-event', self.on_press)
        self.area.connect('button-release-event', self.on_press)
        self.area.connect('motion-notify-event', self.on_press)
        #self.area.connect("clicked", self.on_click)
        self.area.connect("draw", self.on_draw)
        self.area.connect('configure-event', self.on_configure)

    def set_nearest_vertex(self, x,y,delta=30.0):
        self.min_index=-1
        min_dist=+math.inf
        #print("Num vertices",self.g.num_vertices())
        for i in range(self.g.num_vertices()):
            x2=self.pos[i][0]*self.area.get_allocated_width()
            y2=self.pos[i][1]*self.area.get_allocated_width()
            print(x,y,x2,y2)
            d=np.sqrt(pow(x-x2,2)+pow(y-y2,2))
            if(d<min_dist) and (d<=delta):
                min_dist=d
                self.min_index=i


    def on_release(self):
        if self.min_index > -1:
            self.h[self.min_index] = False
            self.min_index=-1
            self.redraw()
            self.area.queue_draw()

    def on_press(self, widget, event):
        if event.type==Gdk.EventType.BUTTON_PRESS:
            print(event.x, event.y, event.type)
            self.set_nearest_vertex(event.x, event.y)
            if self.min_index>-1:
                self.h[self.min_index]=True
                self.mouseX=event.x
                self.mouseY=event.y
                self.redraw()
                self.area.queue_draw()
        else:
            if event.type==Gdk.EventType.BUTTON_RELEASE:
                self.on_release()
            else:
                if event.type==Gdk.EventType.MOTION_NOTIFY and self.min_index>-1:

                    deltaX=event.x-self.mouseX
                    deltaY = event.y - self.mouseY
                    self.mouseX = event.x
                    self.mouseY = event.y
                    self.pos[self.min_index][0]=min(max(self.pos[self.min_index][0]+deltaX/self.area.get_allocated_width(),0.),1.0)
                    self.pos[self.min_index][1] = min(max(self.pos[self.min_index][1] + deltaY / self.area.get_allocated_width(), 0.), 1.0)
                    self.redraw()
                    self.area.queue_draw()


                    print("SOME MOTION")

    def build_graph(self):

        self.g = Graph(directed=True)
        v1 = self.g.add_vertex()
        v12=self.g.add_vertex()
        v2 = self.g.add_vertex()
        v3 = self.g.add_vertex()
        v31 = self.g.add_vertex()
        v4 = self.g.add_vertex()
        v34 = self.g.add_vertex()

        e012 = self.g.add_edge(v1, v12)
        e112 = self.g.add_edge(v12, v2)
        e031 = self.g.add_edge(v3, v31)
        e131 = self.g.add_edge(v31, v1)
        e034 = self.g.add_edge(v3, v34)
        e134 = self.g.add_edge(v34, v4)

        self.x = self.g.new_vertex_property("string")
        self.x[v1] = "Milk"
        self.x[v2] = "MilkBottle"
        self.x[v3] = "MilkPreparation"
        self.x[v4] = "Task"
        self.x[v12] = "is_in"
        self.x[v31] = "involves"
        self.x[v34] = "is_a"

        self.y = self.g.new_edge_property("string")
        self.y[e012] = ""
        self.y[e112] = ""
        self.y[e031] = ""
        self.y[e131] = ""
        self.y[e034] = ""
        self.y[e134] = ""

        self.m = self.g.new_edge_property("string")
        self.m[e012]="none"
        self.m[e112] = "arrow"
        self.m[e031] = "none"
        self.m[e131] = "arrow"
        self.m[e034] = "none"
        self.m[e134] = "arrow"

        self.s = self.g.new_vertex_property("string")
        self.s[v2] = "circle"
        self.s[v1] = "circle"
        self.s[v3] = "circle"
        self.s[v4] = "circle"
        self.s[v12] = "square"
        self.s[v31] = "square"
        self.s[v34] = "square"

        self.c = self.g.new_vertex_property("vector<float>")
        self.c[v2] = np.array([1.,0., 0.0, 1.],dtype="float")
        self.c[v1] = np.array([0.7,0., 0.0, 1.],dtype="float")
        self.c[v3] = np.array([0.5,0., 0.0, 1.],dtype="float")
        self.c[v4] = np.array([0.3,0., 0.0, 1.],dtype="float")
        self.c[v12]=(self.c[v1].get_array()+self.c[v2].get_array())/2.
        self.c[v31] = (self.c[v3].get_array()+self.c[v1].get_array())/2.
        self.c[v34] = (self.c[v4].get_array()+self.c[v3].get_array())/2.
        self.pos=self.g.new_vertex_property("vector<float>")
        self.pos[v1] = np.array([0.3,0.3*0.4],dtype="float")
        self.pos[v2] = np.array([0.3,0.9*0.4],dtype="float")
        self.pos[v3] = np.array([0.7, 0.3*0.4], dtype="float")
        self.pos[v4] = np.array([0.9, 0.9*0.4], dtype="float")
        self.pos[v31] = (self.pos[v1].get_array()+self.pos[v3].get_array())/2.
        self.pos[v12] = (self.pos[v1].get_array()+self.pos[v2].get_array())/2.
        self.pos[v34] = (self.pos[v4].get_array()+self.pos[v3].get_array())/2.

        self.h=self.g.new_vertex_property("bool")
        self.h[v1]=False
        self.h[v2] = False
        self.h[v3] = False
        self.h[v4] = False
        self.h[v12] = False
        self.h[v31] = False
        self.h[v34] = False

        self.mouseX=0.0
        self.mouseY=0.0

    def init_surface(self, area):
        # Destroy previous buffer
        if self.surface is not None:
            self.surface.finish()
            self.surface = None

        # Create a new buffer
        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.area.get_allocated_width(),self.area.get_allocated_width())

    def redraw(self):
        print(self.g, self.c, self.x)
        self.init_surface(self.area)
        context = cairo.Context(self.surface)
        context.scale(self.surface.get_width(), self.surface.get_height())
        self.do_drawing(context)
        self.surface.flush()

    def on_configure(self, area, event, data=None):
        self.redraw()
        return False

    def on_draw(self, area, context):
        if self.surface is not None:
            context.set_source_surface(self.surface, 0.0, 0.0)
            context.paint()
        else:
            print('Invalid surface')
        return False

    def draw_radial_gradient_rect(self, ctx):

        x0, y0 = 0.3, 0.3
        x1, y1 = 0.5, 0.5
        r0 = 0
        r1 = 1
        pattern = cairo.RadialGradient(x0, y0, r0, x1, y1, r1)
        pattern.add_color_stop_rgba(0, 1, 1, 0.5, 1)
        pattern.add_color_stop_rgba(1, 0.2, 0.4, 0.1, 1)
        ctx.rectangle(0, 0, 1, 1)
        ctx.set_source_rgb(1, 1, 1)
        ctx.fill()

        graph_tool.draw.cairo_draw(self.g, self.pos, ctx, edge_dash_style=[.005, .005, 0],  edge_end_marker=self.m, edge_font_family="bahnschrift", vertex_font_family="bahnschrift", vertex_font_size=12, edge_font_size=20, edge_marker_size=18, vertex_fill_color=[.7, .8, .9, 0.9], vertex_color=self.c, edge_text=self.y, vertex_text=self.x, vertex_shape=self.s, vertex_size=80,vertex_halo_size=1.2,vertex_halo=self.h,vertex_pen_width=3.0, edge_pen_width=1)

    def do_drawing(self, ctx):
        self.draw_radial_gradient_rect(ctx)


class SearchDialog(Gtk.Dialog):
    def __init__(self, parent):
        super().__init__(title="Search", transient_for=parent, modal=True)
        self.add_buttons(
            Gtk.STOCK_FIND,
            Gtk.ResponseType.OK,
            Gtk.STOCK_CANCEL,
            Gtk.ResponseType.CANCEL,
        )

        box = self.get_content_area()

        label = Gtk.Label(label="Insert text you want to search for:")
        box.add(label)

        self.entry = Gtk.Entry()
        box.add(self.entry)

        self.show_all()

class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(1920, 1080)
        # This will be in the windows group and have the "win" prefix
        max_action = Gio.SimpleAction.new_stateful(
            "maximize", None, GLib.Variant.new_boolean(False)
        )
        max_action.connect("change-state", self.on_maximize_toggle)
        self.add_action(max_action)

        # Keep it in sync with the actual state
        self.connect(
            "notify::is-maximized",
            lambda obj, pspec: max_action.set_state(
                GLib.Variant.new_boolean(obj.props.is_maximized)
            ),
        )

        lbl_variant = GLib.Variant.new_string("String 1")
        lbl_action = Gio.SimpleAction.new_stateful(
            "change_label", lbl_variant.get_type(), lbl_variant
        )
        lbl_action.connect("change-state", self.on_change_label_state)
        self.add_action(lbl_action)


        self.label = Gtk.Label(label=lbl_variant.get_string(), margin=30)

        self.grid = Gtk.Grid()
        self.top_grid = Gtk.Grid(column_homogeneous=True, column_spacing=10, row_spacing=10)
        self.context_editor_frame=Gtk.Frame()
        self.ontology_frame = OntologyAreaFrame()
        self.imagination_frame = ImaginationAreaFrame()
        self.ce_label=Gtk.Label(label="", margin=10)
        self.ce_label.set_markup("<b>Context Editor - Abstract Context Description Language (ACDL)</b>")
        self.context_editor_frame.set_label_widget(self.ce_label)
        #self.grid.attach(self.label, 1, 20, 2, 1)

        self.o_label = Gtk.Label(label="", margin=10)
        self.o_label.set_markup("<b>Ontology - Context formalization</b>")
        self.ontology_frame.set_label_widget(self.o_label)

        self.i_label = Gtk.Label(label="", margin=10)
        self.i_label.set_markup("<b>Imagination - Mind eye's view of context</b>")
        self.imagination_frame.set_label_widget(self.i_label)

        self.lsepartor_label = Gtk.Label(label="", margin=1)
        self.rsepartor_label = Gtk.Label(label="", margin=1)
        self.bsepartor_label = Gtk.Label(label="", margin=1)

        self.context_editor_frame.add(self.grid)

        self.add(self.top_grid)
        self.top_grid.attach(self.lsepartor_label, 0, 0, 1, 1)
        self.top_grid.attach(self.rsepartor_label, 114, 0, 1, 1)
        self.top_grid.attach(self.bsepartor_label, 0, 2, 1, 1)
        self.top_grid.attach(self.context_editor_frame,1,0,40,2)
        self.top_grid.attach(self.ontology_frame, 41, 0, 72, 1)
        self.top_grid.attach(self.imagination_frame, 41, 1, 72, 1)
        self.create_textview()
        self.create_toolbar()
        self.create_buttons()
        self.grid.show_all()
        self.top_grid.show_all()
    def create_toolbar(self):
        toolbar = Gtk.Toolbar()
        self.grid.attach(toolbar, 0, 0, 3, 1)

        button_bold = Gtk.ToolButton()
        button_bold.set_icon_name("format-text-bold-symbolic")
        toolbar.insert(button_bold, 0)

        button_italic = Gtk.ToolButton()
        button_italic.set_icon_name("format-text-italic-symbolic")
        toolbar.insert(button_italic, 1)

        button_underline = Gtk.ToolButton()
        button_underline.set_icon_name("format-text-underline-symbolic")
        toolbar.insert(button_underline, 2)

        button_bold.connect("clicked", self.on_button_clicked, self.tag_bold)
        button_italic.connect("clicked", self.on_button_clicked, self.tag_italic)
        button_underline.connect("clicked", self.on_button_clicked, self.tag_underline)

        toolbar.insert(Gtk.SeparatorToolItem(), 3)

        radio_justifyleft = Gtk.RadioToolButton()
        radio_justifyleft.set_icon_name("format-justify-left-symbolic")
        toolbar.insert(radio_justifyleft, 4)

        radio_justifycenter = Gtk.RadioToolButton.new_from_widget(radio_justifyleft)
        radio_justifycenter.set_icon_name("format-justify-center-symbolic")
        toolbar.insert(radio_justifycenter, 5)

        radio_justifyright = Gtk.RadioToolButton.new_from_widget(radio_justifyleft)
        radio_justifyright.set_icon_name("format-justify-right-symbolic")
        toolbar.insert(radio_justifyright, 6)

        radio_justifyfill = Gtk.RadioToolButton.new_from_widget(radio_justifyleft)
        radio_justifyfill.set_icon_name("format-justify-fill-symbolic")
        toolbar.insert(radio_justifyfill, 7)

        radio_justifyleft.connect(
            "toggled", self.on_justify_toggled, Gtk.Justification.LEFT
        )
        radio_justifycenter.connect(
            "toggled", self.on_justify_toggled, Gtk.Justification.CENTER
        )
        radio_justifyright.connect(
            "toggled", self.on_justify_toggled, Gtk.Justification.RIGHT
        )
        radio_justifyfill.connect(
            "toggled", self.on_justify_toggled, Gtk.Justification.FILL
        )

        toolbar.insert(Gtk.SeparatorToolItem(), 8)

        button_clear = Gtk.ToolButton()
        button_clear.set_icon_name("edit-clear-symbolic")
        button_clear.connect("clicked", self.on_clear_clicked)
        toolbar.insert(button_clear, 9)

        toolbar.insert(Gtk.SeparatorToolItem(), 10)

        button_search = Gtk.ToolButton()
        button_search.set_icon_name("system-search-symbolic")
        button_search.connect("clicked", self.on_search_clicked)
        toolbar.insert(button_search, 11)

    def create_textview(self):
        scrolledwindow = Gtk.ScrolledWindow()
        #scrolledwindow.set_min_content_height(50)
        #scrolledwindow.set_min_content_width(50)
        #scrolledwindow.set_hexpand(True)
        scrolledwindow.set_vexpand(True)
        self.grid.attach(scrolledwindow, 0, 1, 4, 5)

        self.textview = Gtk.TextView()
        self.textbuffer = self.textview.get_buffer()
        self.context_template=json.loads('{"who": {"type": "robot", "name":"PR2"}, "where": {"type": "location", "name": "kitchen"}, "what": {"type":"action", "name":"preparing", "object":"Breakfast"}, "why":{}, "how":{}, "when":{}}')

        self.textbuffer.set_text(
            json.dumps(self.context_template, indent=28, sort_keys=True)
        )
        scrolledwindow.add(self.textview)

        self.tag_bold = self.textbuffer.create_tag("bold", weight=Pango.Weight.BOLD)
        self.tag_italic = self.textbuffer.create_tag("italic", style=Pango.Style.ITALIC)
        self.tag_underline = self.textbuffer.create_tag(
            "underline", underline=Pango.Underline.SINGLE
        )
        self.tag_found = self.textbuffer.create_tag("found", background="yellow")

    def create_buttons(self):

        self.proceed_button = Gtk.Button(label="Run")
        self.proceed_button.connect("clicked", self.do_clicked)
        self.grid.attach(self.proceed_button, 0, 10, 1, 1)

        self.reset_button = Gtk.Button(label="Reset")
        self.reset_button.connect("clicked", self.do_clicked)
        self.grid.attach(self.reset_button, 0, 11, 1, 1)

        check_editable = Gtk.CheckButton(label="Editable")
        check_editable.set_active(True)
        check_editable.connect("toggled", self.on_editable_toggled)
        self.grid.attach(check_editable, 1, 10, 1, 1)

        check_cursor = Gtk.CheckButton(label="Cursor Visible")
        check_cursor.set_active(True)
        check_editable.connect("toggled", self.on_cursor_toggled)
        self.grid.attach_next_to(
            check_cursor, check_editable, Gtk.PositionType.RIGHT, 1, 1
        )

        radio_wrapnone = Gtk.RadioButton.new_with_label_from_widget(None, "No Wrapping")
        self.grid.attach(radio_wrapnone, 1, 11, 1, 1)

        radio_wrapchar = Gtk.RadioButton.new_with_label_from_widget(
            radio_wrapnone, "Character Wrapping"
        )
        self.grid.attach_next_to(
            radio_wrapchar, radio_wrapnone, Gtk.PositionType.RIGHT, 1, 1
        )

        radio_wrapword = Gtk.RadioButton.new_with_label_from_widget(
            radio_wrapnone, "Word Wrapping"
        )
        self.grid.attach_next_to(
            radio_wrapword, radio_wrapchar, Gtk.PositionType.RIGHT, 1, 1
        )

        radio_wrapnone.connect("toggled", self.on_wrap_toggled, Gtk.WrapMode.NONE)
        radio_wrapchar.connect("toggled", self.on_wrap_toggled, Gtk.WrapMode.CHAR)
        radio_wrapword.connect("toggled", self.on_wrap_toggled, Gtk.WrapMode.WORD)

    def do_clicked(self, widget):
        if(widget==self.proceed_button):
            self.imagination_frame.redraw()
            self.imagination_frame.area.queue_draw()
            self.ontology_frame.build_graph()
            self.ontology_frame.redraw()
            self.ontology_frame.area.queue_draw()

        else:
            if(widget==self.reset_button):
                self.textbuffer.set_text(json.dumps(self.context_template, indent=28, sort_keys=True))
                self.ontology_frame.g.clear()
                self.ontology_frame.redraw()
                self.ontology_frame.area.queue_draw()


    def on_button_clicked(self, widget, tag):
        bounds = self.textbuffer.get_selection_bounds()
        if len(bounds) != 0:
            start, end = bounds
            self.textbuffer.apply_tag(tag, start, end)

    def on_clear_clicked(self, widget):
        start = self.textbuffer.get_start_iter()
        end = self.textbuffer.get_end_iter()
        self.textbuffer.remove_all_tags(start, end)

    def on_editable_toggled(self, widget):
        self.textview.set_editable(widget.get_active())

    def on_cursor_toggled(self, widget):
        self.textview.set_cursor_visible(widget.get_active())

    def on_wrap_toggled(self, widget, mode):
        self.textview.set_wrap_mode(mode)

    def on_justify_toggled(self, widget, justification):
        self.textview.set_justification(justification)

    def on_search_clicked(self, widget):
        dialog = SearchDialog(self)
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            cursor_mark = self.textbuffer.get_insert()
            start = self.textbuffer.get_iter_at_mark(cursor_mark)
            if start.get_offset() == self.textbuffer.get_char_count():
                start = self.textbuffer.get_start_iter()

            self.search_and_mark(dialog.entry.get_text(), start)

        dialog.destroy()

    def search_and_mark(self, text, start):
        end = self.textbuffer.get_end_iter()
        match = start.forward_search(text, 0, end)

        if match is not None:
            match_start, match_end = match
            self.textbuffer.apply_tag(self.tag_found, match_start, match_end)
            self.search_and_mark(text, match_end)

    def on_change_label_state(self, action, value):
        action.set_state(value)
        self.label.set_text(value.get_string())

    def on_maximize_toggle(self, action, value):
        action.set_state(value)
        if value.get_boolean():
            self.maximize()
        else:
            self.unmaximize()


class Application(Gtk.Application):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            application_id="org.example.myapp",
            flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE,
            **kwargs
        )
        self.window = None

        self.add_main_option(
            "test",
            ord("t"),
            GLib.OptionFlags.NONE,
            GLib.OptionArg.NONE,
            "Command line test",
            None,
        )

    def do_startup(self):
        Gtk.Application.do_startup(self)

        action = Gio.SimpleAction.new("about", None)
        action.connect("activate", self.on_about)
        self.add_action(action)

        action = Gio.SimpleAction.new("quit", None)
        action.connect("activate", self.on_quit)
        self.add_action(action)

        builder = Gtk.Builder.new_from_string(MENU_XML, -1)
        self.set_app_menu(builder.get_object("app-menu"))

    def do_activate(self):
        # We only allow a single window and raise any existing ones
        if not self.window:
            # Windows are associated with the application
            # when the last one is closed the application shuts down
            self.window = AppWindow(application=self, title="NaivPhys4RP - Visualizing context-specific imagination")

        self.window.present()

    def do_command_line(self, command_line):
        options = command_line.get_options_dict()
        # convert GVariantDict -> GVariant -> dict
        options = options.end().unpack()

        if "test" in options:
            # This is printed on the main instance
            print("Test argument recieved: %s" % options["test"])

        self.activate()
        return 0

    def on_about(self, action, param):
        about_dialog = Gtk.AboutDialog(transient_for=self.window, modal=True)
        about_dialog.present()

    def on_quit(self, action, param):
        self.quit()


if __name__ == "__main__":
    app = Application()
    app.run(sys.argv)


#graph_draw(g,edge_color="blue",vertex_fill_color=c,vertex_color=c, vertex_text=x, edge_text=y, edge_pen_width=3, vertex_font_size=19, edge_font_size=19, vertex_aspect=1. ,adjust_aspect=True, fit_view_ink=True, fit_view=True)