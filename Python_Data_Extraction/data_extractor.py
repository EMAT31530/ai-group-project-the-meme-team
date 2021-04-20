import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import math

global default_colors
prop_cycle = plt.rcParams['axes.prop_cycle']
default_colors = ["red", "green", "blue", "purple", "orange", "black"]

def colorInterp(c1,c2,mix=0):
    '''Return the midpoint colour between c1 and c2, mix between [0, 1]'''
    c1=np.array(mpl.colors.to_rgb(c1))
    c2=np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)

def get_colors(c1, c2, n):
    '''Returns a list of n colors that interpolates c1 to c2'''
    gradient = []
    for x in range(n+1):
        gradient.append(colorInterp(c1,c2,x/n))
    return gradient

def parseLine(line):
    '''Parses a line of data from Unity agent output file'''
    variables_str = line.split(";")
    #This is the duttiest line of code I've ever written, but trust it works
    variables = [[float(j) for j in i.strip("(").strip(")").split(",")] for i in variables_str]
    return variables

def plot_circle(axes, pos, r, c, three_d=False):
    '''Plot a circle on axes at pos with radius r and color c. Set three_d=True if on 3d axis'''
    theta = np.linspace(0, math.pi * 2, 100)
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)
    if three_d:
        circle_z = np.ones(100)*pos[1]
        axes.plot(circle_x + pos[0], circle_y + pos[2], circle_z, zdir='z', color=c)
    else:
        axes.plot(circle_x + pos[0], circle_y + pos[2], color=c)


def graph(data, axes, c):
    '''Plot data on axes using color specified by c'''
    #### End position
    end = data[-1]
    #### Plot xyz position
    #Obtain episode xyz pos
    pos = np.array(data[0])
    pos = np.subtract(pos, end)     #Normalize to target
    pos = np.swapaxes(pos, 0, 1)    #Swap axis: [[x1, y1, z1], [x2, y2, z2]...] -> [[x1, x2 ...], [y1, y2 ...] ...]

    #Plot all position plots
    axes[0][0].plot(pos[0], pos[1], pos[2], zdir='y', color=c)
    axes[0][1].plot(pos[0], pos[1], color=c)
    axes[0][2].plot(pos[0], pos[2], color=c)
    axes[0][3].plot(pos[2], pos[1], color=c)

    #### Plot velocity magnitude
    vel = np.array(data[2])
    mag = np.linalg.norm(vel, axis=1)
    axes[1].plot(mag, color=c)

    #### Plot rotation dot upright
    rot = np.array(data[1]) * 100
    axes[2].plot(rot, color=c)

    #### Plot points at square limits to force equal axes on 3D and x-z plot
    limit = max([max(pos[0]), abs(min(pos[0])), max(pos[2]), abs(min(pos[2]))])
    limit_x = [limit, limit, -limit, -limit]
    limit_y = [limit, -limit, limit, -limit]

    axes[0][0].plot(limit_x, limit_y, [0,0,0,0], lw=0.0)
    axes[0][2].plot(limit_x, limit_y, lw=0.0)

def axes_setup(agent=None):
    #Position plot figure and formatting
    fig_pos = plt.figure()
    ax_pos1 = fig_pos.add_subplot(2, 2, 1, projection='3d')
    ax_pos1.set_xticks([])
    ax_pos1.set_yticks([])
    ax_pos1.set_zticks([])
    ax_pos2 = fig_pos.add_subplot(2, 2, 2)
    ax_pos2.set_xlabel("x")
    ax_pos2.set_ylabel("y")
    ax_pos2.set_xticks([])
    ax_pos2.set_yticks([])
    ax_pos3 = fig_pos.add_subplot(2, 2, 3)
    ax_pos3.set_xlabel("x")
    ax_pos3.set_ylabel("z")
    ax_pos3.set_aspect("equal")
    ax_pos3.set_xticks([])
    ax_pos3.set_yticks([])
    ax_pos4 = fig_pos.add_subplot(2, 2, 4, sharex=ax_pos2, sharey=ax_pos2)
    ax_pos4.set_xlabel("z")
    ax_pos4.set_ylabel("y")
    ax_pos4.set_xticks([])
    ax_pos4.set_yticks([])
    fig_pos.subplots_adjust(wspace=0.15, hspace=0.15)
    
    #Velocity plot figure and formatting
    fig_vel = plt.figure()
    ax_vel = plt.axes()
    ax_vel.set_xlabel("Time step")
    ax_vel.set_ylabel("Speed")
    
    #Rotation dot y-axis figure and formatting
    fig_rot = plt.figure()
    ax_rot = plt.axes()
    ax_rot.set_xlabel("Time step")
    ax_rot.set_ylabel("% vertical upward")
    ax_rot.set_ylim([-120, 120])

    if agent != None:
        fig_pos_title = 'Agent position (normalised to target) in 3D space - ' + agent + " (" + str(len(episodes)) + " episodes)"
        ax_vel_title = 'Agent Speed vs time step - ' + agent + " (" + str(len(episodes)) + " episodes)"
        ax_rot_title = 'Upright vs time step - ' + agent + " (" + str(len(episodes)) + " episodes)"
    else:
        fig_pos_title = 'Agent position (normalised to target) in 3D space - All'
        ax_vel_title = 'Agent Speed vs time step - All'
        ax_rot_title = 'Upright vs time step - All'
        
    fig_pos.suptitle(fig_pos_title, fontsize=10)
    ax_vel.set_title(ax_vel_title, fontsize=10)
    ax_rot.set_title(ax_rot_title, fontsize=10)

    
    #List of axes for this agent
    axes = [[ax_pos1, ax_pos2, ax_pos3, ax_pos4], ax_vel, ax_rot]
    return axes

    
    

#Get all agents to plot
agents = next(os.walk('Data'))[1]

#Axes that all agents plot on
global_axes = axes_setup()

for agent in agents:
    path = "Data\\" + agent         #Path to agent subfolder
    episodes = os.listdir(path)     #Get all episodes of agent

    #Generate a color gradient specific to this agent
    #Agent number indexes global colors list
    #Gradient created from that color to white
    agent_axes = axes_setup(agent=agent)
    episode_colors = get_colors(default_colors[agents.index(agent)], "white", len(episodes) + 1)

    #For each episode of this agent
    for episode in episodes:

        #Initialise lists to store data
        pos = []
        vel = []
        rot = []
        
        #Read data and format
        file = open(path + "\\" + episode, "r")
        data = file.readlines()
        for line in data:
            newPoint = parseLine(line.strip("\n"))

            pos.append(newPoint[0])
            vel.append(newPoint[1])
            rot.append(newPoint[2])
        end = newPoint[3]
        
        plotData = [pos, vel, rot, end]

        #Plot episode
        color = episode_colors.pop(0)
        graph(plotData, agent_axes, color)
        graph(plotData, global_axes, color)

    #Plot black circle at target position on 3d plot and x-z plot
    plot_circle(agent_axes[0][0], [0, 0, 0], 5, "black", three_d=True)
    plot_circle(agent_axes[0][2], [0, 0, 0], 5, "black")

plt.show()

            
