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
        ax.clear()
        ax.set_xlim3d(np.amin(joint_coordinates[:, x, :]), np.amax(joint_coordinates[:, x, :]))
        ax.set_ylim3d(np.amin(joint_coordinates[:, y, :]), np.amax(joint_coordinates[:, y, :]))
        ax.set_zlim3d(np.amin(joint_coordinates[:, z, :]), np.amax(joint_coordinates[:, z, :]))
        
        ax.scatter(joint_coordinates[:, x, t], joint_coordinates[:, y, t], joint_coordinates[:, z, t])
        
        
        line = plt3d.art3d.Line3D([0.1, 0.1], [0.1, 0.5], [0.1, 0.1])
        
        head_neck = plt3d.art3d.Line3D([joint_coordinates[0, x, t], joint_coordinates[1, x, t]], 
                                      [joint_coordinates[0, y, t], joint_coordinates[1, y, t]], 
                                      [joint_coordinates[0, z, t], joint_coordinates[1, z, t]])
        
        # ax.add_line(head_neck)
        
        for i in range(connexion_tuples.shape[0]):
            j1 = connexion_tuples[i, 0]
            j2 = connexion_tuples[i, 1]
            
            
            joint_line = plt3d.art3d.Line3D([joint_coordinates[j1, x, t], joint_coordinates[j2, x, t]], 
                                            [joint_coordinates[j1, y, t], joint_coordinates[j2, y, t]], 
                                            [joint_coordinates[j1, z, t], joint_coordinates[j2, z, t]])
            
            ax.add_line(joint_line)
        
        ax.view_init(10, 10)
        
        fig.canvas.draw()
        plt.pause(.001)
        # time.sleep(0.01)