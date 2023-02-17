#!/usr/bin/env python3
import time
import lark
from threading import Thread
import PIL
import cv2
import json
import math
import itertools
import sys
import gi
import actionlib
import roslib
from difflib import SequenceMatcher
roslib.load_manifest('rosprolog')
roslib.load_manifest('naivphys4rp_msgs')
import rospy
from rosprolog_client import PrologException, Prolog
from naivphys4rp_msgs.msg import *
gi.require_version('WebKit2', '4.0') 
from gi.repository import WebKit2
from pyvis.network import Network
import networkx as nx
gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gio, Gtk, Pango, Gdk, GdkPixbuf
import cairo
import numpy as np
from graph_tool.all import *
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
        self.imagination=None

        self.init_surface(self.area)
        self.area.set_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.area.add_events(Gdk.EventMask.BUTTON_RELEASE_MASK)
        self.area.add_events(Gdk.EventMask.POINTER_MOTION_MASK)
        self.area.connect('button-press-event', self.on_press)
        self.area.connect('button-release-event', self.on_press)
        self.area.connect('motion-notify-event', self.on_press)
        self.area.connect("draw", self.on_draw)
        self.area.connect('configure-event', self.on_configure)

    def imagine(self):
        self.imagination=[]
        filename = "../../../resources/litImage.jpg"
        self.imagination.append(cv2.imread(filename))
        filename = "../../../resources/maskImage.jpg"
        self.imagination.append(cv2.imread(filename))
        filename = "../../../resources/depthImage.jpg"
        self.imagination.append(cv2.imread(filename))


    def forget(self):
        if self.imagination is not None:
            del self.imagination
            self.imagination=None

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

    def draw_image(self, ctx):
        if self.imagination is None:
            ctx.rectangle(0.0, 0, 1.0, 1.0)
            ctx.set_source_rgba(1.0,1.0,1.0,1.0)
            ctx.fill()
        else:
            x_space=5
            x_dim=self.imagination[0].shape[1] +x_space+self.imagination[1].shape[1]+ x_space+self.imagination[2].shape[1]
            y_dim=max(self.imagination[0].shape[0], self.imagination[1].shape[0], self.imagination[2].shape[0])
            z_dim=max(self.imagination[0].shape[2], self.imagination[1].shape[2], self.imagination[2].shape[2],4)
            img_arr = np.ones([y_dim, x_dim, z_dim], dtype="uint8")*255
            for l in range(3):
                img_arr[:self.imagination[0].shape[0] , :self.imagination[0].shape[1] , l] = self.imagination[0][:, :, l]
                img_arr[:self.imagination[1].shape[0] , self.imagination[0].shape[1]+x_space:self.imagination[0].shape[1]+x_space+self.imagination[1].shape[1], l] = self.imagination[1][:, :, l]
                img_arr[:self.imagination[2].shape[0], self.imagination[0].shape[1] +self.imagination[1].shape[1]+2*x_space:, l] = self.imagination[2][:, :, l]
            # im = PIL.Image.open(filename)
            img_arr=cv2.resize(img_arr, (0,0), fx=self.area.get_allocated_width()*1.0/(1.0*x_dim), fy=self.area.get_allocated_width()*1.0/(1.0*x_dim))
            # im.putalpha(256)  # create alpha channel
            # arr = np.array(im)
            height, width, channels = img_arr.shape
            self.surface = cairo.ImageSurface.create_for_data(img_arr, cairo.FORMAT_ARGB32, width, height)
            # surface = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_ARGB32, width, height)
            print("IMAGE SHAPE", height, width, channels)
            # Create a new buffer



    def do_drawing(self, ctx):
        self.draw_image(ctx)



####################################################################################################################################################################################


class OntologyAreaFrame(Gtk.Frame):
    def __init__(self, css=None, border_width=0):
        super().__init__()
        self.set_border_width(border_width)
        self.set_size_request(100, 100)
        self.vexpand = True
        self.hexpand = True
        self.surface = None
        self.view = WebKit2.WebView()
        self.settings = WebKit2.Settings()

        font_size = 14
        font_default = "VLGothic"
        font_serif = "VLGothic"
        font_sans_serif = "VLGothic"
        font_monospace = "VLGothic"

        self.settings.set_property("serif-font-family", font_serif)
        self.settings.set_property("sans-serif-font-family", font_sans_serif)
        self.settings.set_property("monospace-font-family", font_monospace)
        self.settings.set_property("default-font-family", font_default)
        self.settings.set_property("default-font-size", font_size)
        self.view.set_settings(self.settings)
        self.add(self.view)
        self.view.load_html("", "ontology.html")
	
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

    def destroy_graph(self):
        if self.g is not None:
            self.g.clear()
            self.g=None

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

    def draw_graph(self, ctx):

        if self.g is None:
            ctx.rectangle(0, 0, 1, 1)
            ctx.set_source_rgb(1, 1, 1)
            ctx.fill()
        else:
            graph_tool.draw.cairo_draw(self.g, self.pos, ctx, edge_dash_style=[.005, .005, 0],  edge_end_marker=self.m, edge_font_family="bahnschrift", vertex_font_family="bahnschrift", vertex_font_size=12, edge_font_size=20, edge_marker_size=18, vertex_fill_color=[.7, .8, .9, 0.9], vertex_color=self.c, edge_text=self.y, vertex_text=self.x, vertex_shape=self.s, vertex_size=80,vertex_halo_size=1.2,vertex_halo=self.h,vertex_pen_width=3.0, edge_pen_width=1)

    def do_drawing(self, ctx):
        self.draw_graph(ctx)


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

class Transformers():
    def __init__(self, state_verb):
        self.state_verb=state_verb

    def context(self, trees):
        interpretation = []
        for statements in trees.children:
            if statements is not None:
                if statements.__class__==lark.Tree:
                    if statements.data.value=='statement':
                        interpretation=interpretation+self.statement(statements)
        return interpretation

    def statement(self,trees):
        subject=self.decompose(trees.children[0])
        subject_noun=subject['noun'][0]
        object=self.decompose(trees.children[2])
        subject_adjectiv=subject['adjectiv']
        subject_terminal=self.getTerminal(subject_noun)
        object_noun = object['noun'][0]
        object_adjectiv = object['adjectiv']
        object_terminal=self.getTerminal(object_noun)
        interpretation=[[subject_terminal,self.getTerminal(trees.children[1]), object_terminal]]
        interpretation=interpretation+self.subject(subject_adjectiv,subject_terminal)+self.object(object_adjectiv,object_terminal)
        return interpretation

    def subject(self,adjectives, noun):
        interpretation=[]
        for adj in adjectives:
            interpretation.append([str(noun),self.state_verb,self.getTerminal(adj)])
        return interpretation

    def object(self,adjectives, noun):
        interpretation=[]
        for adj in adjectives:
            interpretation.append([str(noun),self.state_verb,self.getTerminal(adj)])
        return interpretation

    def decompose(self,trees):
        result={'noun':[],'adjectiv':[]}
        for res in trees.children:
            if  (res is not None) and (res.__class__==lark.Tree) and (res.data.value in ["noun","adjectiv"]):
                result[res.data.value].append(res)
        return result

    def getTerminal(self,trees):
        terminal=""
        if trees is None:
            pass
        else:
            if trees.__class__==lark.Tree:
                for elt in trees.children:
                    r=self.getTerminal(elt)
                    if len(terminal)>0 and len(r)>0:
                        terminal=terminal+" "+r
                    else:
                        terminal = terminal + r
            else:
                r=str(trees)
                if len(terminal) > 0 and len(r) > 0:
                    terminal = terminal + " " + r
                else:
                    terminal = terminal + r
        return terminal

class AppWindow(Gtk.ApplicationWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize(1920, 1080)
        #the graph
        self.nt = None
        #create action client for ontology
        self.path_name="/ontology/naivphys4rp.owl"
        self.package="belief_state"
        self.namespace = "http://www.semanticweb.org/franklin/ontologies/2022/7/naivphys4rp.owl#"  # default ontology's namespace
        self.concepts=[]
        self.relations=[]
        self.symbol_grounding_table = {}
        self.onto_load_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_load', KBLoadOntologyAction)
        print("Client ontology loading is started !")
        print("Waiting for action server ...")
        self.onto_load_client.wait_for_server()
        print("Loading ontology ...")
        self.grammarFile='grammar.json'
        self.grammar=json.dumps("{}")
        self.state_verb="is"
        self.grammar = """
        	context: (statement delimiter)*

        	statement: subject verb object
        	
        	object: [determinant] (adjectiv)* noun
        	
        	subject: [determinant] (adjectiv)* noun
        	
        	determinant: DET

        	verb: iverb | dverb

        	iverb: dverb preposition

        	noun: NOUN

        	delimiter: FULLSTOP | COMMA

        	dverb: DVERB

        	preposition: PREP
        	
        	adjectiv: (COLOR|SIZE|SHAPE|MATERIAL|TIGHTNESS)
            
            FULLSTOP: "."
            
            COMMA: ","
        
            DVERB: "has" | "is" | "participates" | "makes" | "cooks" | "prepares" | "pours" | "moves" | "transports" | "picks"  ["up"] | "places" | "sees" | "observes" | "looks" | "perceives"
        
            PREP: "in" | "on" | "over" | "under" | "of" | "at" | "to" | "behind" | "left to" | "right to" | "near to" | "far from" | "near" | "in front of"

        	NOUN: "box" | "milk" | "robot" | "bowl" | "spoon" | "bottle" | "table" | "fridge" | "mug" | "drawer" | "island" | "sink" | "muesli" | "cornflakes" | "kitchen"
        	
        	DET: "a" | "an" | "the"
        	
            COLOR: "red" | "orange" | "brown" | "yellow" | "green" | "blue" | "white" | "gray" | "black" | "violet" |"pink"
            
            SHAPE: "cubic" |"conical" | "cylindrical" | "filiform" | "flat" | "rectangular" | "circular" | "triangular"
            
            MATERIAL: "plastic" | "woody" | "glass" | "steel" | "carton" | "ceramic"
            
            TIGHTNESS: "solid" | "gaz" | "liquid" | "powder"
            
            SIZE: "large" | "big" | "medium" | "small" | "tiny"

        	%import common.ESCAPED_STRING

        	%import common.SIGNED_NUMBER

        	%import common.WS

        	%ignore WS
        """
        self.json_parser = lark.Lark(self.grammar, start="context",ambiguity='explicit')

        goal=KBLoadOntologyGoal()
        goal.package=self.package
        goal.namespace=self.namespace
        goal.relative_path=self.path_name
        self.onto_load_client.send_goal(goal)
        self.onto_load_client.wait_for_result()
        if(self.onto_load_client.get_result().status==True):
            print("Ontology loaded successfully!")
        else:
            print("Ontology failed to load!!!")

        self.onto_class_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_subclass', KBSubClassAction)
        print("Client ontology subclass is started !")
        print("Waiting for action server subclass...")
        self.onto_class_client.wait_for_server()
        print("Action server subclass loaded")

        self.onto_property_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_subproperty', KBSubPropertyAction)
        print("Client ontology subproperty is started !")
        print("Waiting for action server subproperty...")
        self.onto_property_client.wait_for_server()
        print("Action server subproperty loaded")

        self.onto_list_property_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_list_property', KBListPropertyAction)
        print("Client list property is started !")
        print("Waiting for action server list property...")
        self.onto_list_property_client.wait_for_server()
        print("Action server list property loaded")

        self.onto_list_class_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_list_class', KBListClassAction)
        print("Client list class is started !")
        print("Waiting for action server list class...")
        self.onto_list_class_client.wait_for_server()
        print("Action server list class loaded")

        self.onto_property_io_client = actionlib.SimpleActionClient('naivphys4rp_knowrob_property_io', KBPropertyIOAction)
        print("Client property io is started !")
        print("Waiting for action server property io...")
        self.onto_property_io_client.wait_for_server()
        print("Action server property io loaded")

        """
                ################## List of Class ########################################
                goal = KBListClassGoal()
                self.onto_list_class_client.send_goal(goal)
                self.onto_list_class_client.wait_for_result()
                results = self.onto_list_class_client.get_result()
                self.generic_class_prefix="_:Description"
                if (len(results.classes) > 0):
                    print("Knowrob query executed successfully!")
                    for cls in results.classes:
                        if len(cls.split(self.generic_class_prefix))<2:
                            self.concepts.append(cls)
                    print(len(self.concepts),self.concepts)
                else:
                    print("Knowrob query failed!!")

                ####################### List of Property ##############################################
                goal = KBListPropertyGoal()
                self.onto_list_property_client.send_goal(goal)
                self.onto_list_property_client.wait_for_result()
                results = self.onto_list_property_client.get_result()
                if (len(results.properties) > 0):
                    print("Knowrob query executed successfully!")
                    self.concepts = results.properties.copy()
                    print(len(results.properties),results.properties)
                else:
                    print("Knowrob query failed!!")
                """
        ################ List of Range ####################################
        goal = KBSubClassGoal()
        goal.is_sub = True
        goal.namespace = "http://www.w3.org/2002/07/owl#"
        goal.class_name = 'Thing'
        self.onto_class_client.send_goal(goal)
        self.onto_class_client.wait_for_result()
        results = self.onto_class_client.get_result()
        self.classes = {}
        self.generic_class_prefix = "_:Description"
        if (len(results.classes) > 0):
            print("Knowrob query executed successfully!")
            for cls in results.classes:
                if len(cls.split(self.generic_class_prefix)) < 2 and len(cls.split(self.namespace)) >= 2:
                    self.classes[cls.split(self.namespace).pop()]=cls
            print(len(self.classes), self.classes)
        else:
            print("Knowrob query failed!!")

        ######################### List of Subproperty ################################
        goal = KBSubPropertyGoal()
        goal.is_sub = True
        goal.namespace = "http://www.w3.org/2002/07/owl#"
        goal.property_name = 'topObjectProperty'
        self.onto_property_client.send_goal(goal)
        self.onto_property_client.wait_for_result()
        results = self.onto_property_client.get_result()
        self.object_properties = {}
        self.generic_class_prefix = "_:Description"
        if (len(results.propertyes) > 0):
            print("Knowrob query executed successfully!")
            for cls in results.propertyes:
                if len(cls.split(self.generic_class_prefix)) < 2 and len(cls.split(self.namespace)) >= 2:
                    self.object_properties[cls.split(self.namespace).pop()]=cls
            print(len(self.object_properties), self.object_properties)
        else:
            print("Knowrob query failed!!")
        self.object_properties[self.state_verb]="rdfs:SubClassOf"

        ################## List of Action as Object Property ########################################
        goal = KBSubClassGoal()
        goal.class_name = 'Action'
        goal.is_sub = True
        goal.namespace = self.namespace
        self.actions = {}
        self.onto_class_client.send_goal(goal)
        self.onto_class_client.wait_for_result()
        results = self.onto_class_client.get_result()
        self.generic_class_prefix = "_:Description"
        if (len(results.classes) > 0):
            print("Knowrob query executed successfully!")
            for cls in results.classes:
                if len(cls.split(self.generic_class_prefix)) < 2:
                    self.actions[cls.split(self.namespace).pop()] = cls
                    self.object_properties[cls.split(self.namespace).pop()] = cls
            print(len(self.actions), self.actions)
        else:
            print("Knowrob query failed!!")

        ######################### List of Subproperty ################################
        goal = KBSubPropertyGoal()
        goal.is_sub = True
        goal.namespace = "http://www.w3.org/2002/07/owl#"
        goal.property_name = 'topDataProperty'
        self.onto_property_client.send_goal(goal)
        self.onto_property_client.wait_for_result()
        results = self.onto_property_client.get_result()
        self.data_properties = {}
        self.generic_class_prefix = "_:Description"
        if (len(results.propertyes) > 0):
            print("Knowrob query executed successfully!")
            for cls in results.propertyes:
                if len(cls.split(self.generic_class_prefix)) < 2 and len(cls.split(self.namespace)) >= 2:
                    self.data_properties[cls.split(self.namespace).pop()]=cls
            print(len(self.data_properties), self.data_properties)
        else:
            print("Knowrob query failed!!")

        ######################### List of property range ################################
        self.property_io={}
        for prop in self.data_properties:
            goal = KBPropertyIOGoal()
            goal.type_range="extension"
            goal.namespace = self.namespace
            goal.property_name = prop.split(self.namespace).pop()
            self.onto_property_io_client.send_goal(goal)
            self.onto_property_io_client.wait_for_result()
            self.property_io[prop] = self.onto_property_io_client.get_result().erange

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
        #print("SIZE GRID", self.top_grid.get_size())
        self.top_grid.attach(self.lsepartor_label, 0, 0, 1, 1)
        self.top_grid.attach(self.rsepartor_label, 114, 0, 1, 1)
        self.top_grid.attach(self.bsepartor_label, 0, 49, 1, 1)
        self.top_grid.attach(self.context_editor_frame,1,0,40,49)
        self.top_grid.attach(self.ontology_frame, 41, 0, 72, 31)
        self.top_grid.attach(self.imagination_frame, 41, 31, 72, 18)
        self.create_textview()
        self.create_toolbar()
        self.create_buttons()
        self.grid.show_all()
        self.top_grid.show_all()

    def getBaseName(self,lists):
        return [r.split(self.namespace).pop() for r in lists]

    def closest(self,s,l):
        l=list(l)
        s=s.lower().replace('_','').replace(' ','')
        res=l[0]
        dist=+np.Inf
        for r in l:
            re = r.lower().replace('_','').replace(' ','')
            d=self.distance(s,re)
            if d<dist:
                dist=d
                res=r
        return res

    def distance(self, str1, str2):
        m=SequenceMatcher(None,str1,str2).find_longest_match(0,len(str1), 0,len(str2))
        return len(str1)+len(str2)-2*m.size

    def parse_context(self, text):
        try:
            parsing=self.json_parser.parse(text.lower())
            return (True, parsing,Transformers(self.state_verb).context(parsing))
        except Exception as e:
            print(e)
            return (False,str(e),[])
        """
        #reading grammar file
        with open(self.grammarFile, 'r') as infile:
            self.grammar = json.load(infile)
        infile.close()

        #writing grammar file
        with open(self.grammarFile, 'w') as infile:
            json.load(infile,self.grammar)
        infile.close()
        """


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

        self.dynamic_button = Gtk.CheckButton(label="Dynamic")
        self.dynamic_button.set_active(True)
        self.dynamic_button.connect("toggled", self.on_dynamic_toggled)
        self.grid.attach_next_to(
            self.dynamic_button,check_cursor, Gtk.PositionType.RIGHT, 1, 1
        )
        self.spinner = Gtk.Spinner()
        """
        self.imageFrame = Gtk.Frame()
        self.imageFrame.set_shadow_type(Gtk.ShadowType.IN)
        self.filename = "../../../resources/search.gif"
        self.pixbuf = GdkPixbuf.Pixbuf.new_from_file(self.filename)
        self.transparent = self.pixbuf.add_alpha(True, 0xff, 0xff, 0xff)
        self.image = Gtk.Image.new_from_pixbuf(self.pixbuf)
        self.imageFrame.add(self.image)
        """
        self.grid.attach(self.spinner, 3, 0, 1, 1)
        #self.spinner.start()

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

    def build_graph(self):
        #parse input context
        startIter, endIter = self.textbuffer.get_bounds()
        text = self.textbuffer.get_text(startIter, endIter, False)
        res=self.parse_context(text)
        if res[0]:
            title0 = "\n 0. Input Context Description as Informal Narrative \n\n\n"
            result0=text
            title1="\n\n\n 1. Syntactical Parsing of Context Description \n\n\n"
            result1=res[1].pretty()
            self.textbuffer.set_text(title0+result0+title1+result1)
            start_iter = self.textbuffer.get_start_iter()
            end_iter=self.textbuffer.get_end_iter()
            start_iter.forward_chars(0)
            end_iter.backward_chars(len(result0+title1+result1))
            self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter, end_iter)
            start_iter.forward_chars(len(title0+result0))
            end_iter.forward_chars(len(result0 + title1))
            self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter,end_iter)
        else:
            title0 = "\n 0. Input Context Description as Informal Narrative \n\n\n"
            result0 = text
            title1 = "\n\n\n 1. Syntactical Parsing of Context Description \n\n\n"
            result1 = "\n\n\n Error(s) found: \n\n"+str(res[1])
            self.textbuffer.set_text(title0+result0+title1+result1)
            start_iter = self.textbuffer.get_start_iter()
            end_iter=self.textbuffer.get_end_iter()
            start_iter.forward_chars(0)
            end_iter.backward_chars(len(result0+title1+result1))
            self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter, end_iter)
            start_iter.forward_chars(len(title0+result0))
            end_iter.forward_chars(len(result0 + title1))
            self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter,end_iter)
            self.spinner.stop()
            return
        if res[2]==[]:
            self.textbuffer.set_text(str(res[1]))
            self.spinner.stop()
            return
        graph = res[2]
        graph.sort()
        graph=list(graph for graph,_ in itertools.groupby(graph))

        # symbol grounding

        self.symbol_grounding_table = {}
        for rel in graph:
            if rel[0] not in self.symbol_grounding_table.keys():
                self.symbol_grounding_table[rel[0]]=self.classes[self.closest(rel[0],self.classes.keys())]

            if rel[1] != self.state_verb:
                self.symbol_grounding_table[rel[2]]=self.classes[self.closest(rel[2],self.classes.keys())]
                res=self.closest(rel[1], self.object_properties.keys())
                print("***************************************** ",res,rel[1],self.object_properties.keys())
                self.symbol_grounding_table[rel[1]] = self.object_properties[res]
            else:
                key=None
                rel2=rel[2][:1].upper()+rel[2][1:]
                print("++++++++++++++++++++++++++++ ",rel2)
                for k in self.data_properties.keys():
                    if rel2 in self.property_io[k]:
                        key=k
                        break
                if key is None:
                    self.symbol_grounding_table[rel[2]] = self.classes[self.closest(rel[2], self.classes.keys())]
                    self.symbol_grounding_table[rel[1]] = self.object_properties[self.closest(rel[1], self.object_properties.keys())]
                else:
                    self.symbol_grounding_table[rel[1]] = self.data_properties[key]
                    self.symbol_grounding_table[rel[2]] = "owl:oneOf(rdfs:range("+str(self.data_properties[key])+"))"
        #Build String
        title2= "\n 2. Grounding of Narrative Symbols in the Ontology \n\n\n"
        result2=""
        for elt in self.symbol_grounding_table.keys():
            result2=result2+elt+" ----> "+self.symbol_grounding_table[elt]+"\n\n"
        self.textbuffer.set_text(title0 + result0 + title1 + result1+ title2 + result2)
        start_iter = self.textbuffer.get_start_iter()
        end_iter = self.textbuffer.get_end_iter()
        start_iter.forward_chars(0)
        end_iter.backward_chars(len(result0 + title1 + result1+ title2 + result2))
        self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter, end_iter)
        start_iter.forward_chars(len(title0 + result0))
        end_iter.forward_chars(len(result0 + title1))
        self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter, end_iter)
        start_iter = self.textbuffer.get_start_iter()
        end_iter = self.textbuffer.get_end_iter()
        start_iter.forward_chars(len(title0+result0+title1+result1))
        end_iter.backward_chars(len(result2))
        self.textbuffer.apply_tag(self.textbuffer.get_tag_table().lookup('bold'), start_iter,end_iter)
        #creating empty graph
        self.nx_graph = nx.DiGraph()
        for rel in graph:
            self.nx_graph.add_edge(rel[0],rel[2],label=rel[1])
            time.sleep(1)
            self.nt = Network('100%', '100%', directed=True)
            self.nt.from_nx(self.nx_graph)
            self.nt.set_options("""
                                                    const options = {
                                                      "physics": {
                                                        "enabled": true,
                                                        "barnesHut": {
                                                          "gravitationalConstant": -4200,
                                                          "centralGravity": 0.95,
                                                          "springLength": 90,
                                                          "springConstant": 0.015,
                                                          "damping": 0.01
                                                        },
                                                        "maxVelocity": 143,
                                                        "minVelocity": 0.0001
                                                      }
                                                    }
                                                    """)
            if self.nt is not None:
                self.nt.options['physics']['enabled'] = self.dynamic_button.get_active()
            graphText = self.nt.generate_html()
            self.ontology_frame.view.load_html(graphText, "ontology.html")
        """
        self.nx_graph.nodes[1]['title'] = 'Number 1'
        self.nx_graph.nodes[1]['group'] = 1
        self.nx_graph.nodes[3]['title'] = 'I belong to a different group!'
        self.nx_graph.nodes[3]['group'] = 10
        self.nx_graph.add_node(20, size=20, title='couple', group=2)
        self.nx_graph.add_node(21, size=15, title='couple', group=2)
        self.nx_graph.add_edge(20, 21, weight=5, label="loves")
        self.nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)
        """

        self.imagination_frame.imagine()
        self.imagination_frame.redraw()
        self.imagination_frame.area.queue_draw()
        self.spinner.stop()

    def do_clicked(self, widget):
        if(widget==self.proceed_button):
            self.spinner.start()
            thread = Thread(target=self.build_graph, args=[])
            thread.start()

            #thread=Thread(target=self.spin,args=['stop'])
            #thread.start()
            #thread.join()

        else:
            if(widget==self.reset_button):
                self.textbuffer.set_text(json.dumps(self.context_template, indent=28, sort_keys=True))
                self.nt=None
                self.ontology_frame.view.load_html("", "ontology.html")
                self.imagination_frame.forget()
                self.imagination_frame.redraw()
                self.imagination_frame.area.queue_draw()
                self.dynamic_button.set_active(True)
                self.spinner.stop()


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

    def on_dynamic_toggled(self, widget):
            if self.nt is not None:
                self.nt.options['physics']['enabled'] = widget.get_active()
                graphText = self.nt.generate_html()
                self.ontology_frame.view.load_html(graphText, "ontology.html")


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

if __name__ == '__main__':
    rospy.init_node('naivphys4rp_imagination_node')
    app = Application()
    app.run(sys.argv)
    rospy.spin()
#graph_draw(g,edge_color="blue",vertex_fill_color=c,vertex_color=c, vertex_text=x, edge_text=y, edge_pen_width=3, vertex_font_size=19, edge_font_size=19, vertex_aspect=1. ,adjust_aspect=True, fit_view_ink=True, fit_view=True)
