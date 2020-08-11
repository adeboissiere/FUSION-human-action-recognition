import numpy as np
import matplotlib.pyplot as plt
import time
import mpl_toolkits.mplot3d as plt3d

def animateJointCoordinates(joint_coordinates, connexion_tuples):
    '''
    joint_coordinates : shape(joints, 3, seq_len)
    
    
    '''
    fig = plt.figure()
    fig.set_size_inches(10,10)
    ax = plt.axes(projection='3d')
    
    plt.ion()
    fig.show()
    fig.canvas.draw()
    
    x = 0
    y = 2
    z = 1

    
    for t in range(joint_coordinates.shape[2]):
    # for t in range(1):
        ax.clear()
        
        # Camera coordinate system
        axis_length = 0.2
        
        ax.scatter([0], [0], [0], color="red")
        ax.scatter([axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length], marker="v", color="red")
        
        x_axis = plt3d.art3d.Line3D([0, axis_length], [0, 0], [0, 0])
        x_axis.set_color("red")
        y_axis = plt3d.art3d.Line3D([0, 0], [0, axis_length], [0, 0])
        y_axis.set_color("red")
        z_axis = plt3d.art3d.Line3D([0, 0], [0, 0], [0, axis_length])
        z_axis.set_color("red")
        ax.add_line(x_axis)
        ax.add_line(y_axis)
        ax.add_line(z_axis)
        
        # New coordinate system
        x_spine_mid = joint_coordinates[1, x, 0]
        y_spine_mid = joint_coordinates[1, y, 0]
        z_spine_mid = joint_coordinates[1, z, 0]
        
        ax.scatter(x_spine_mid, y_spine_mid, z_spine_mid, color="green")
        ax.scatter([x_spine_mid + axis_length, x_spine_mid, x_spine_mid], 
                   [y_spine_mid, y_spine_mid + axis_length, y_spine_mid], 
                   [z_spine_mid, z_spine_mid, z_spine_mid + axis_length], marker="v", color="green")
        
        x_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid + axis_length], [y_spine_mid, y_spine_mid], [z_spine_mid, z_spine_mid])
        x_axis.set_color("green")
        y_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid], [y_spine_mid, y_spine_mid + axis_length], [z_spine_mid, z_spine_mid])
        y_axis.set_color("green")
        z_axis = plt3d.art3d.Line3D([x_spine_mid, x_spine_mid], [y_spine_mid, y_spine_mid], [z_spine_mid, z_spine_mid + axis_length])
        z_axis.set_color("green")
        ax.add_line(x_axis)
        ax.add_line(y_axis)
        ax.add_line(z_axis)
        
        # Translation vector
        trans_vec = plt3d.art3d.Line3D([0, x_spine_mid], [0, y_spine_mid], [0, z_spine_mid], linestyle='--', color="black")
        ax.add_line(trans_vec)
        
        
        # Subject coordinates
        ax.set_xlim3d(min(np.amin(joint_coordinates[:, x, :]),-axis_length), max(np.amax(joint_coordinates[:, x, :]), axis_length))
        ax.set_ylim3d(min(np.amin(joint_coordinates[:, y, :]),-axis_length), max(np.amax(joint_coordinates[:, y, :]), axis_length))
        ax.set_zlim3d(min(np.amin(joint_coordinates[:, z, :]),-axis_length), max(np.amax(joint_coordinates[:, z, :]), axis_length))
        
        ax.scatter(joint_coordinates[:, x, t], joint_coordinates[:, y, t], joint_coordinates[:, z, t], color="blue")
        
        for i in range(connexion_tuples.shape[0]):
            j1 = connexion_tuples[i, 0]
            j2 = connexion_tuples[i, 1]
            
            
            joint_line = plt3d.art3d.Line3D([joint_coordinates[j1, x, t], joint_coordinates[j2, x, t]], 
                                            [joint_coordinates[j1, y, t], joint_coordinates[j2, y, t]], 
                                            [joint_coordinates[j1, z, t], joint_coordinates[j2, z, t]], linestyle=':')
            
            ax.add_line(joint_line)
        
        ax.view_init(10, 50)
        # ax.set_axis_off()
        fig.canvas.draw()
        
        plt.pause(.001)
        # plt.savefig(str(t) + '.png')
        # time.sleep(0.01)