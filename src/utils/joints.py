# To associate an ID to a joint
from enum import IntEnum
import numpy as np


# See https://medium.com/@lisajamhoury/understanding-kinect-v2-joints-and-coordinate-system-4f4b90b9df16 for joints info
class Joints(IntEnum):
    SPINEBASE = 0
    SPINEMID = 1
    NECK = 2
    HEAD = 3
    
    SHOULDERLEFT = 4
    ELBOWLEFT = 5
    WRISTLEFT = 6
    HANDLEFT = 7

    SHOULDERRIGHT = 8
    ELBOWRIGHT = 9
    WRISTRIGHT = 10
    HANDRIGHT = 11

    HIPLEFT = 12
    KNEELEFT = 13
    ANKLELEFT = 14
    FOOTLEFT = 15

    HIPRIGHT = 16
    KNEERIGHT = 17
    ANKLERIGHT = 18
    FOOTRIGHT = 19

    SPINESHOULDER = 20

    HANDTIPLEFT = 21
    THUMBLEFT = 22

    HANDTIPRIGHT = 23
    THUMBRIGHT = 24


kinematic_chain = np.array([Joints.SPINEMID, Joints.SPINEBASE, Joints.HIPLEFT, Joints.KNEELEFT, Joints.ANKLELEFT, Joints.FOOTLEFT,
                            Joints.ANKLELEFT, Joints.KNEELEFT, Joints.HIPLEFT, Joints.SPINEBASE,
                            Joints.SPINEMID,  Joints.SPINEBASE, Joints.HIPRIGHT, Joints.KNEERIGHT, Joints.ANKLERIGHT, Joints.FOOTRIGHT,
                            Joints.ANKLERIGHT, Joints.KNEERIGHT, Joints.HIPRIGHT, Joints.SPINEBASE,
                            Joints.SPINEMID, Joints.SPINESHOULDER, Joints.NECK, Joints.HEAD,
                            Joints.NECK, Joints.SPINESHOULDER, Joints.SPINEMID,
                            Joints.SPINESHOULDER, Joints.SHOULDERLEFT, Joints.ELBOWLEFT, Joints.WRISTLEFT, Joints.HANDLEFT, Joints.HANDTIPLEFT,
                            Joints.HANDLEFT, Joints.WRISTLEFT, Joints.THUMBLEFT, Joints.WRISTLEFT, Joints.ELBOWLEFT, Joints.SHOULDERLEFT, Joints.SPINESHOULDER, Joints.SPINEMID,
                            Joints.SPINESHOULDER, Joints.SHOULDERRIGHT, Joints.ELBOWRIGHT, Joints.WRISTRIGHT, Joints.HANDRIGHT, Joints.HANDTIPRIGHT,
                            Joints.HANDRIGHT, Joints.WRISTRIGHT, Joints.THUMBRIGHT, Joints.WRISTRIGHT, Joints.ELBOWRIGHT, Joints.SHOULDERRIGHT, Joints.SPINESHOULDER, Joints.SPINEMID])

'''
kinematic_chain = np.array([Joints.SPINEMID, Joints.SPINEBASE, Joints.HIPLEFT, Joints.KNEELEFT, Joints.ANKLELEFT, Joints.FOOTLEFT,
                            Joints.ANKLELEFT, Joints.KNEELEFT, Joints.HIPLEFT,
                            Joints.SPINEMID,  Joints.SPINEBASE, Joints.HIPRIGHT, Joints.KNEERIGHT, Joints.ANKLERIGHT, Joints.FOOTRIGHT,
                            Joints.ANKLERIGHT, Joints.KNEERIGHT, Joints.HIPRIGHT, Joints.SPINEBASE,
                            Joints.SPINEMID, Joints.SPINESHOULDER, Joints.NECK, Joints.HEAD,
                            Joints.NECK, Joints.SPINESHOULDER, Joints.SPINEMID,
                            Joints.SPINESHOULDER, Joints.SHOULDERLEFT, Joints.ELBOWLEFT, Joints.WRISTLEFT, Joints.HANDLEFT, Joints.HANDTIPLEFT,
                            Joints.HANDLEFT, Joints.WRISTLEFT, Joints.ELBOWLEFT, Joints.SHOULDERLEFT, Joints.SPINESHOULDER, Joints.SPINEMID,
                            Joints.SPINESHOULDER, Joints.SHOULDERRIGHT, Joints.ELBOWRIGHT, Joints.WRISTRIGHT, Joints.HANDRIGHT, Joints.HANDTIPRIGHT,
                            Joints.HANDRIGHT, Joints.WRISTRIGHT, Joints.ELBOWRIGHT, Joints.SHOULDERRIGHT, Joints.SPINESHOULDER, Joints.SPINEMID])
'''

connexion_tuples = np.array([[Joints.SPINEBASE, Joints.SPINEMID],
                             [Joints.SPINEMID, Joints.SPINESHOULDER],
                             [Joints.SPINESHOULDER, Joints.NECK],
                             [Joints.NECK, Joints.HEAD],

                             [Joints.SPINESHOULDER, Joints.SHOULDERLEFT], # 4
                             [Joints.SHOULDERLEFT, Joints.ELBOWLEFT],
                             [Joints.ELBOWLEFT, Joints.WRISTLEFT],
                             [Joints.WRISTLEFT, Joints.HANDLEFT],
                             [Joints.HANDLEFT, Joints.HANDTIPLEFT],
                             [Joints.HANDLEFT, Joints.THUMBLEFT],

                             [Joints.SPINESHOULDER, Joints.SHOULDERRIGHT], # 10
                             [Joints.SHOULDERRIGHT, Joints.ELBOWRIGHT],
                             [Joints.ELBOWRIGHT, Joints.WRISTRIGHT],
                             [Joints.WRISTRIGHT, Joints.HANDRIGHT],
                             [Joints.HANDRIGHT, Joints.HANDTIPRIGHT],
                             [Joints.HANDRIGHT, Joints.THUMBRIGHT],

                             [Joints.SPINEBASE, Joints.HIPRIGHT], # 16
                             [Joints.HIPRIGHT, Joints.KNEERIGHT],
                             [Joints.KNEERIGHT, Joints.ANKLERIGHT],
                             [Joints.ANKLERIGHT, Joints.FOOTRIGHT],

                             [Joints.SPINEBASE, Joints.HIPLEFT], # 20
                             [Joints.HIPLEFT, Joints.KNEELEFT],
                             [Joints.KNEELEFT, Joints.ANKLELEFT],
                             [Joints.ANKLELEFT, Joints.FOOTLEFT]])  # shape (n_connexions, 2)