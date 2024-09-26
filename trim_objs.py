import os
import numpy as np
import trimesh

def init():
    vertexNum = 35184

    ## remove some components of head, such as eyelash/teech/tongue...
    seleted_arr_mh = [[30162, 31783], [19272, 24574], [10227, 19042], [1129, 9920]]
    removed_indices = list(range(1, vertexNum+1))
    seleted_arr_mh.sort()
    for arr in seleted_arr_mh[::-1]:
        removed_indices = removed_indices[:arr[0]] + removed_indices[arr[1]+1:]

    index_map = {}
    shift_value = 0
    for i in range(vertexNum):
        if(i+1 in removed_indices[shift_value:]):
            shift_value += 1
        else:
            index_map[i+1] = i+1 - shift_value

    shift_value = 0
    haha = []
    for i in range(vertexNum):
        if(i+1 in removed_indices[shift_value:]):
            haha.append(i-shift_value)
            shift_value += 1

    return vertexNum, index_map, haha

def obj2lines(mh_mesh_fname, vertex_num):
    triangles, triangle_lines = [], []
    lines = open(mh_mesh_fname, 'r').readlines()
    lines = [line for line in lines if line.startswith('v ') or line.startswith('f ')]

    vertices = []
    if(len(triangles)==0):
        for i, line in enumerate(lines):
            line = line.strip()
            if(i<vertex_num):
                vertices.append(np.array(line.split()[1:]).astype(np.float32))
            else:
                triangles.append(np.array([part.split('/')[0] for part in line.split()[1:]]).astype(np.float32))
        for triangle in triangles:
            triangle_lines.append('f %d %d %d' % (triangle[0], triangle[1], triangle[2]))
    else:
        for i, line in enumerate(lines):
            line = line.strip()
            if(i<vertex_num):
                vertices.append(np.array(line.split()[1:]).astype(np.float32))
            else:
                break

    scale = 1
    lines = []
    for vertex in vertices:
        lines.append('v %f %f %f\n' % (vertex[0]*scale, vertex[1]*scale, vertex[2]*scale))

    lines.extend(triangle_lines)
    return lines

def reserveSkull(lines, vertexNum, index_map, haha, savePath):
    triangles = lines[vertexNum:]
    lines = lines[:vertexNum]

    for h in haha:
        lines.pop(h)

    outObjFile = open(savePath, 'w')

    tridx = np.load('data/tridx.npy').tolist()
    for idx in tridx:
        vector = triangles[idx].strip().split()[1:]
        vector = list(map(lambda x: int(x), vector))
        lines.append('f %d %d %d\n' %  (index_map[vector[0]], index_map[vector[1]], index_map[vector[2]]))

    outObjFile.writelines(lines)


vertexNum, index_map, haha = init()

root = r'D:\thomsonProblem\record_1000'
for file in os.listdir(root):
    if(not file.endswith('.obj')):
        continue
    lines = obj2lines(os.path.join(root, file), vertexNum)
    reserveSkull(lines, vertexNum, index_map, haha, os.path.join(root, '_' + file))

for file in os.listdir(root):
    if(not file.endswith('.obj')):
        continue
    
    mesh = trimesh.load(os.path.join(root, file), process=False, maintain_order=True)
    mesh.vertices[:, -1] -= 150
    mesh.export(os.path.join(root, file[1:]))