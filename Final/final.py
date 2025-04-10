import math
import numpy as np
from PIL import Image

# Параметры рендера
IMAGE_WIDTH = 2000
IMAGE_HEIGHT = 2000
PROJECTION_SCALE = 10000
BACKGROUND_COLOR = (144, 203, 194)

# Инициализация буферов
matrix = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)
z_buffer = np.full((IMAGE_HEIGHT, IMAGE_WIDTH), np.inf)


class ModelTransform:
    def __init__(self):
        self.translation = [0.0, -0.03, 1.0]
        self.rotation_euler = [np.pi/2, 2*np.pi, 0]
        self.scale = 1.0
        self.quaternion = [0.0, 0.0, 0.0, 1.0]


def quaternion_rotation(vertices, q):
    qw, qx, qy, qz = q
    rotated = []
    for v in vertices:
        x, y, z = v
        # Кватернионное умножение
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        # Умножение на сопряженный
        xx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        yy = iy * qw + iw * -qy + iz * -qx - ix * -qz
        zz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        rotated.append([xx, yy, zz])
    return rotated


def transform_vertices(vertices, transform, use_quaternion=False):
    if use_quaternion:
        rotated = quaternion_rotation(vertices, transform.quaternion)
    else:
        alpha, beta, gamma = transform.rotation_euler
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
        rotated = [R @ vertex for vertex in vertices]

    scaled = [np.array(v) * transform.scale for v in rotated]
    translated = [v + np.array(transform.translation) for v in scaled]
    return translated


def barycentric(x, y, x0, y0, x1, y1, x2, y2):
    denom = (x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2)
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / denom
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / denom
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2


def compute_face_normal(v0, v1, v2):
    u = v1 - v0
    v = v2 - v0
    normal = np.cross(u, v)
    return normal / np.linalg.norm(normal)


def load_model(obj_path):
    vertices = []
    texture_coords = []
    faces = []
    texture_indices = []

    with open(obj_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == 'v':
                vertices.append([float(p) for p in parts[1:]])
            elif parts[0] == 'vt':
                texture_coords.append([float(p) for p in parts[1:]])
            elif parts[0] == 'f':
                face_verts = []
                face_textures = []
                for part in parts[1:]:
                    indices = part.split('/')
                    face_verts.append(int(indices[0]))
                    if len(indices) > 1 and indices[1]:
                        face_textures.append(int(indices[1]))
                # Разбиваем на треугольники для полигонов >3 вершин
                for i in range(1, len(face_verts) - 1):
                    faces.append([face_verts[0], face_verts[i], face_verts[i + 1]])
                    if face_textures:
                        texture_indices.append([face_textures[0], face_textures[i], face_textures[i + 1]])
    return vertices, faces, texture_coords, texture_indices


def compute_vertex_normals(vertices, faces):
    vertex_normals = np.zeros((len(vertices), 3))
    for face in faces:
        v0, v1, v2 = [i - 1 for i in face]
        normal = compute_face_normal(np.array(vertices[v0]),
                                     np.array(vertices[v1]),
                                     np.array(vertices[v2]))
        vertex_normals[v0] += normal
        vertex_normals[v1] += normal
        vertex_normals[v2] += normal
    # Нормализуем
    norms = np.linalg.norm(vertex_normals, axis=1)
    norms[norms == 0] = 1  # Чтобы избежать деления на ноль
    return vertex_normals / norms[:, np.newaxis]


def draw_triangle_gouraud_textured(matrix, z_buffer, transform,
                                   v0, v1, v2,
                                   vt0, vt1, vt2,
                                   vertex_normals, texture):
    # Преобразование вершин
    x0, y0, z0 = v0
    x1, y1, z1 = v1
    x2, y2, z2 = v2

    # Проекция на экран
    u0 = PROJECTION_SCALE * x0 / z0 + IMAGE_WIDTH // 2
    v0 = PROJECTION_SCALE * y0 / z0 + IMAGE_HEIGHT // 2
    u1 = PROJECTION_SCALE * x1 / z1 + IMAGE_WIDTH // 2
    v1 = PROJECTION_SCALE * y1 / z1 + IMAGE_HEIGHT // 2
    u2 = PROJECTION_SCALE * x2 / z2 + IMAGE_WIDTH // 2
    v2 = PROJECTION_SCALE * y2 / z2 + IMAGE_HEIGHT // 2

    # Ограничивающий прямоугольник
    xmin = max(0, math.floor(min(u0, u1, u2)))
    xmax = min(IMAGE_WIDTH - 1, math.ceil(max(u0, u1, u2)))
    ymin = max(0, math.floor(min(v0, v1, v2)))
    ymax = min(IMAGE_HEIGHT - 1, math.ceil(max(v0, v1, v2)))

    # Освещение вершин
    light_dir = np.array([0, 0, 1])
    i0 = max(0.1, np.dot(vertex_normals[0], light_dir))
    i1 = max(0.1, np.dot(vertex_normals[1], light_dir))
    i2 = max(0.1, np.dot(vertex_normals[2], light_dir))

    # Основной цикл растеризации
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            l0, l1, l2 = barycentric(x, y, u0, v0, u1, v1, u2, v2)
            if l0 >= 0 and l1 >= 0 and l2 >= 0:
                # Глубина
                z = l0 * z0 + l1 * z1 + l2 * z2
                if z < z_buffer[y, x]:
                    z_buffer[y, x] = z

                    # Текстурирование
                    tex_u = l0 * vt0[0] + l1 * vt1[0] + l2 * vt2[0]
                    tex_v = l0 * vt0[1] + l1 * vt1[1] + l2 * vt2[1]
                    tex_x = int(tex_u * (texture.width - 1))
                    tex_y = int((1 - tex_v) * (texture.height - 1))
                    tex_color = texture.getpixel((tex_x, tex_y))

                    # Освещение
                    intensity = l0 * i0 + l1 * i1 + l2 * i2
                    color = (
                        int(tex_color[0] * intensity),
                        int(tex_color[1] * intensity),
                        int(tex_color[2] * intensity)
                    )

                    matrix[y, x] = color


# Основной код выполнения
if __name__ == "__main__":
    # Загрузка модели и текстуры
    vertices, faces, texture_coords, texture_indices = load_model("cat.obj")
    texture = Image.open("Cat.jpg")

    # Инициализация преобразований
    transform = ModelTransform()
    transform.translation = [ 0, -0.03, 500]

    # Вычисление нормалей вершин
    vertex_normals = compute_vertex_normals(vertices, faces)

    # Преобразование вершин
    transformed_vertices = transform_vertices(vertices, transform)

    # Отрисовка всех полигонов
    for i, face in enumerate(faces):
        v0_idx, v1_idx, v2_idx = [idx - 1 for idx in face]
        v0 = transformed_vertices[v0_idx]
        v1 = transformed_vertices[v1_idx]
        v2 = transformed_vertices[v2_idx]

        vt0 = texture_coords[texture_indices[i][0] - 1] if texture_indices else [0, 0]
        vt1 = texture_coords[texture_indices[i][1] - 1] if texture_indices else [0, 0]
        vt2 = texture_coords[texture_indices[i][2] - 1] if texture_indices else [0, 0]

        draw_triangle_gouraud_textured(
            matrix, z_buffer, transform,
            v0, v1, v2,
            vt0, vt1, vt2,
            [vertex_normals[v0_idx], vertex_normals[v1_idx], vertex_normals[v2_idx]],
            texture
        )

    # Сохранение результата
    Image.fromarray(matrix). save("output.png")
    print("Рендеринг завершен. ")
