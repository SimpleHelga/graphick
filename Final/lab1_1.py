import math
import numpy as np
from PIL import Image,ImageOps

matrix= np.zeros((2000,2000,3),dtype=np.uint8)
z_buffer=np.full((2000,2000),np.inf)
matrix[0:2000,0:2000]= (144,203,194)

Tz=1
g=10000*Tz
def rotate(vertices, alpha, beta, gamma):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
    R = R_x @ R_y @ R_z

    transformed_vertices = [R @ vertex+[0,-0.03,Tz] for vertex in vertices]

    return transformed_vertices
def bercent(x,y,x0,y0,x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2))/((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def draw_triangle_gouraud_textured(matrix, z_buffer, x0, y0, z0, x1, y1, z1, x2, y2, z2,
                                   v0_idx, v1_idx, v2_idx, vertex_normals, light_dir,
                                   vt0, vt1, vt2, texture):
    # Проекция вершин
    u0 = g * x0 / z0 + matrix.shape[1] // 2
    v0 = g * y0 / z0 + matrix.shape[0] // 2
    u1 = g * x1 / z1 + matrix.shape[1] // 2
    v1 = g * y1 / z1 + matrix.shape[0] // 2
    u2 = g * x2 / z2 + matrix.shape[1] // 2
    v2 = g * y2 / z2 + matrix.shape[0] // 2

    # Ограничивающий прямоугольник
    xmin = max(0, math.floor(min(u0, u1, u2)))
    xmax = min(matrix.shape[1], math.ceil(max(u0, u1, u2)))
    ymin = max(0, math.floor(min(v0, v1, v2)))
    ymax = min(matrix.shape[0], math.ceil(max(v0, v1, v2)))

    # Вычисляем освещение для каждой вершины
    i0 = max(0, np.dot(vertex_normals[v0_idx], light_dir))
    i1 = max(0, np.dot(vertex_normals[v1_idx], light_dir))
    i2 = max(0, np.dot(vertex_normals[v2_idx], light_dir))

    for x in range(xmin, xmax):
        for y in range(ymin, ymax):
            l0, l1, l2 = bercent(x, y, u0, v0, u1, v1, u2, v2)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buffer[y, x]:
                    z_buffer[y, x] = z

                    # 1. Получаем цвет из текстуры
                    tex_u = l0 * vt0[0] + l1 * vt1[0] + l2 * vt2[0]
                    tex_v = l0 * vt0[1] + l1 * vt1[1] + l2 * vt2[1]
                    tex_x = int(tex_u * (texture.shape[1] - 1))
                    tex_y = int((1 - tex_v) * (texture.shape[0] - 1))  # Переворот по Y
                    tex_color = texture[tex_y, tex_x]

                    # 2. Вычисляем интенсивность освещения
                    intensity = l0 * i0 + l1 * i1 + l2 * i2

                    # 3. Комбинируем текстуру с освещением
                    combined_color = (
                        int(tex_color[0] * intensity),
                        int(tex_color[1] * intensity),
                        int(tex_color[2] * intensity)
                    )

                    matrix[y, x] = combined_color
def norm(x0,y0,z0,x1,y1,z1,x2,y2,z2):
    x=(y1-y2)*(z1-z0)-(y1-y0)*(z1-z2)
    y=((x1-x2)*(z1-z0)-(x1-x0)*(z1-z2))
    z=(x1-x2)*(y1-y0)-(x1-x0)*(y1-y2)
    return x,y,z
def ligth(x,y,z):
    cosl=z/(math.sqrt(x**2+y**2+z**2))
    return cosl
f= open("model_1.obj")
texture= np.array(Image.open("bunny-atlas.jpg"))
vec=[]
lis=[]
texture_coords=[]
texture_indices=[]
alpha,beta,gamma= np.radians(0),np.radians(180),np.radians(0)
x,y,z=0,0,0
for s in f:
    split = s.strip().split()
    if not split:
        continue
    if split[0] == "v":
        vec.append([float(x) for x in split[1:]])
    elif split[0] == "vt":
        texture_coords.append([float(x) for x in split[1:]])
    elif split[0] == "f":
        face_verts = []
        face_textures = []
        for part in split[1:]:
            indices = part.split('/')
            face_verts.append(int(indices[0]))
            if len(indices) > 1 and indices[1]:
                face_textures.append(int(indices[1]))
        lis.append(face_verts)
        if face_textures:
            texture_indices.append(face_textures)
light_dir= np.array([0,0,1])
transformed_vec=rotate(vec,alpha,beta, gamma)

# Вычисляем нормали вершин
vertex_normals = [np.zeros(3) for _ in range(len(vec))]
vertex_counts = [0 for _ in range(len(vec))]

for face in lis:
    v0, v1, v2 = face[0] - 1, face[1] - 1, face[2] - 1
    x0, y0, z0 = vec[v0]
    x1, y1, z1 = vec[v1]
    x2, y2, z2 = vec[v2]
    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    for v_idx in [v0, v1, v2]:
        vertex_normals[v_idx] += [xn, yn, zn]
        vertex_counts[v_idx] += 1

# Нормализуем
for i in range(len(vertex_normals)):
    if vertex_counts[i] > 0:
        vertex_normals[i] /= vertex_counts[i]
        length = np.linalg.norm(vertex_normals[i])
        if length > 0:
            vertex_normals[i] /= length

for i, face in enumerate(lis):
    v0_idx, v1_idx, v2_idx = face[0] - 1, face[1] - 1, face[2] - 1
    vt0_idx, vt1_idx, vt2_idx = texture_indices[i][0] - 1, texture_indices[i][1] - 1, texture_indices[i][2] - 1

    x0, y0, z0 = transformed_vec[v0_idx]
    x1, y1, z1 = transformed_vec[v1_idx]
    x2, y2, z2 = transformed_vec[v2_idx]

    # Проверка на лицевую грань
    xn, yn, zn = norm(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    if ligth(xn, yn, zn) < 0:
        draw_triangle_gouraud_textured(
            matrix, z_buffer,
            x0, y0, z0, x1, y1, z1, x2, y2, z2,
            v0_idx, v1_idx, v2_idx, vertex_normals, light_dir,
            texture_coords[vt0_idx],
            texture_coords[vt1_idx],
            texture_coords[vt2_idx],
            texture
        )
image=Image.fromarray(matrix,mode="RGB")
image=ImageOps.flip(image)
image.save('image.png')