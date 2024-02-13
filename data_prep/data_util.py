import os, pdb, cv2,json
import numpy as np

def boxStr2p1p2(s, h, w):
    data = [round(float(c), 2) for c in s.split(",")]
    # range
    data[0] = max(0, min(w, data[0]))
    data[1] = max(0, min(h, data[1]))
    data[2] = max(0, min(w, data[2]))
    data[3] = max(0, min(h, data[3]))
    
    return ((data[0], data[1]), (data[2], data[3]))

def boxStr2Center(s, h, w):
    minp, maxp = boxStr2p1p2(s, h, w)
    return (round(float(minp[0]+maxp[0])/2, 2), round(float(minp[1]+maxp[1])/2, 2))

def boxStr2xywh(s, h, w):
    '''
    Return [x, y, w, h] for COCO box label.
    '''
    data = [int(float(c)) for c in s.split(",")]
    # range
    data[0] = max(0, min(w, data[0]))
    data[1] = max(0, min(h, data[1]))
    data[2] = max(0, min(w, data[2]))
    data[3] = max(0, min(h, data[3]))
    return [ data[0], data[1], data[2]-data[0], data[3]-data[1] ]
    
    
def boxStr2xyxy(s, h, w):
    '''
    Return [x, y, w, h] for COCO box label.
    '''
    # data = [int(float(c)) for c in s.split(",")]
    data = [int(float(c.strip()[1:-1])) if c.strip()[0]=="'" else int(float(c.strip())) for c in s.split(",") ]
    # range
    
    data[0] = max(0, min(w, data[0]))
    data[1] = max(0, min(h, data[1]))
    data[2] = max(0, min(w, data[2]))
    data[3] = max(0, min(h, data[3]))
    return [ data[0], data[1], data[2], data[3]]
    
    
def blendMask(I,mask,color, alpha):
    for c in range(3):
        Ic = I[:,:,c]
        Ic[mask] = ((Ic[mask].astype(np.float32)*alpha) + (float(color[c])*(1-alpha))).astype(np.uint8)
        I[:,:,c] = Ic


def parseSide(s):
    if   s == 'left_hand': return 0
    elif s == 'right_hand': return 1
    else:
        print(f'Weird hand side label is {s}, {type(s)}')
        pdb.set_trace()

def parseSideInverse(s):
    if   s == 0: return 'left'
    elif s == 1: return 'right'
    elif s == -1: return '*'
    else:
        print(f'Weird hand side label is {s}, {type(s)}')
        pdb.set_trace()
        
def parseState(s):
    if   s == 'no_contact': return 0
    elif s == 'self_contact': return 1
    elif s == 'other_person_contact': return 2
    elif s == 'object_contact': return 3
    elif s in ['inconclusive', 'None']: return None
    else:
        print(f'Weird hand state label is {s}')
        pdb.set_trace()
        
def parseStateInverse(s):
    if   s == 0: return 'notC'
    elif s == 1: return 'otherPC'
    elif s == 2: return 'selfC'
    elif s == 3: return 'objectC'
    elif s ==-1: return '*'
    else:
        print(f'Weird hand state label is {s}')
        pdb.set_trace()
        
        
        
def parseToolType(s):
    if s == 'None': return -1
    elif s == 'tool_,_touched': return 0
    elif s == 'tool_,_held': return 1
    elif s == 'tool_,_used': return 2
    elif s == 'container_,_touched': return 3
    elif s == 'container_,_held': return 4
    elif s == 'neither_,_touched': return 5
    elif s == 'neither_,_held': return 6
    else:
        print(f'Weird tool type label is {s}')
        pdb.set_trace()
        
def parseToolTypeInverse(s):
    if s == -1: return '*'
    elif s == 0: return 'tool,touched'
    elif s == 1: return 'tool,held'
    elif s == 2: return 'tool,used'
    elif s == 3: return 'container,touched'
    elif s == 4: return 'container,held'
    elif s == 5: return 'neither,touched'
    elif s == 6: return 'neither,held'
    else:
        print(f'Weird tool type label is {s}')
        pdb.set_trace()
        

def parseGraspType(s):
    '''Parse String to Int.
    '''
    if s == 'None': return -1
    elif s == 'NP-Palm': return 0
    elif s == 'NP-Fin': return 1
    elif s == 'Pow-Pris': return 2
    elif s == 'Pre-Pris': return 3
    elif s == 'Pow-Circ': return 4
    elif s == 'Pre-Circ': return 5
    elif s == 'Exten': return 6
    elif s == 'Later': return 7
    elif s in ['other', 'Other']: return 8
    else:
        print(f'Weird grasp type label is {s}')
        pdb.set_trace()
    

def parseGraspTypeInverse(s):
    '''Parse Int to String.
    '''
    if s == -1: return '*'
    elif s == 0: return 'NP-Palm'
    elif s == 1: return 'NP-Fin'
    elif s == 2: return 'Pow-Pris'
    elif s == 3: return 'Pre-Pris'
    elif s == 4: return 'Pow-Circ'
    elif s == 5: return 'Pre-Circ'
    elif s == 6: return 'Exten'
    elif s == 7: return 'Later'
    elif s == 8: return 'Other'
    else:
        print(f'Weird grasp type label is {s}')
        pdb.set_trace()
        
        
def parseMask(mask):
    area = np.sum(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    polygons = []
    for object in contours:
        coords = []
        for point in object:
            coords.append(round(point[0][0], 2))
            coords.append(round(point[0][1], 2))
        polygons.append(coords)
    # pdb.set_trace()
    return area, polygons


def parseOffset(handCenter, objectCenter):
    '''
    Calculate offset from hand to object bbox center.
    offset: [unit_vector[0], unit_vector[1], magnitude]
    '''
    scalar = 1000 # TODO: the scalar needs testing
    vec = np.array([objectCenter[0]-handCenter[0], objectCenter[1]-handCenter[1]]) / scalar
    norm = np.linalg.norm(vec)
    unit_vec = vec / norm
    offset = [unit_vec[0], unit_vec[1], norm]
    return offset    



class NpEncoder(json.JSONEncoder):
    """
    Numpy encoder for json.dump().
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
