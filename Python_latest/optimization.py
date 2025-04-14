# optimization.py
import config
import functools
import math
import numpy as np

def weighted_lb_eigen_projection(mesh, num_eigenvectors=3, max_iter=1000, tol=1e-6):
    """
    使用加权拉普拉斯-贝尔特拉米特征分解生成平滑初始映射。

    参数:
        mesh - 输入的网格数据
        num_eigenvectors - 需要的非平凡特征向量数量，默认取3个
        max_iter - 最大迭代次数
        tol - 收敛阈值

    返回:
        smoothed_map - 平滑后的初始映射
    """
    L = construct_laplacian_matrix(mesh)  # 拉普拉斯矩阵
    M = compute_area_weight_matrix(mesh)  # 面积权重矩阵

    # 使用 numpy 的特征分解，计算最小的 num_eigenvectors 个特征向量
    # L 的特征值与 M 对角元素的比率给出特征向量
    eigvals, eigvecs = np.linalg.eigh(L)

    # 选择前 num_eigenvectors 个非平凡特征向量
    # 特征向量从第二个开始，因为第一个是常数向量
    smoothed_map = eigvecs[:, 1:num_eigenvectors + 1]

    # 将特征向量归一化到单位球面
    smoothed_map /= np.linalg.norm(smoothed_map, axis=1, keepdims=True)
    return smoothed_map


def compute_area_weight_matrix(mesh):
    """计算面积权重矩阵 M"""
    n = len(mesh.vertices)
    M = np.zeros((n, n))

    for vertex in mesh.vertices:
        area = compute_vertex_area(vertex)
        M[vertex.index, vertex.index] = area

    return M


def compute_vertex_area(vertex):
    """计算每个顶点的面积权重"""
    area = 0.0
    start_halfedge = vertex.halfedge
    current_halfedge = start_halfedge

    while True:
        if current_halfedge.facet is None:
            break
        area += compute_facet_area(current_halfedge.facet) / 3.0
        current_halfedge = current_halfedge.opposite.next
        if current_halfedge == start_halfedge or current_halfedge is None:
            break

    return area

def compute_facet_area(facet):
    """计算三角形面的面积"""
    he1 = facet.halfedge
    he2 = he1.next
    he3 = he2.next
    p1 = [he1.vertex.x, he1.vertex.y, he1.vertex.z]
    p2 = [he2.vertex.x, he2.vertex.y, he2.vertex.z]
    p3 = [he3.vertex.x, he3.vertex.y, he3.vertex.z]
    edge1 = create_vector(p1, p2)
    edge2 = create_vector(p1, p3)
    cross_prod = cross_product(edge1, edge2)
    return 0.5 * norm(cross_prod)

# Normalizing F_k in fast_algorithm
def fast_algorithm(F0, rho, delta, xi, epsilon, mesh, max_iter=1000):
    L = construct_laplacian_matrix(mesh)
    F_k = F0.copy()
    Q_k = 1
    C_k = E(F_k, L)
    k = 0

    while np.linalg.norm(compute_gradient(F_k, L)) > epsilon and k < max_iter:
        # 打印当前的谐波能量
        minimized_harmonic_energy = E(F_k, L)
        if k % 100 == 0:
            print(f"Iteration {k}, Harmonic Energy: {minimized_harmonic_energy}")
            print(f"F_k sample [0]: {F_k[0]}")

        # 计算步长 tau_k
        tau_k = choose_step_size(F_k, delta, rho, C_k, L)

        # 使用 tau_k 更新 F_k
        F_k_next = Y(F_k, tau_k, L)

        # 更新 Q 和 C
        Q_k_next = xi * Q_k + 1
        C_k_next = (xi * Q_k * C_k + E(F_k_next, L)) / Q_k_next

        # 打印调试信息
        print(f"Iteration {k}, Step Size: {tau_k}, New Energy: {C_k_next}")

        # 更新变量
        F_k = F_k_next
        Q_k = Q_k_next
        C_k = C_k_next
        k += 1

    return F_k

def orthogonal_constrained_optimization(mesh, F0, tolerance, max_iterations=1000, initial_step_size=0.1, beta=0.9):
    """
    基于球面约束的优化方法，最小化谐波能量，确保每一步的更新满足球面约束。
    """
    # 初始化
    X = F0.copy()
    L = construct_laplacian_matrix(mesh)
    step_size = initial_step_size  # 初始化步长

    prev_gradient = np.zeros_like(X)  # 用于存储前一轮的梯度

    for k in range(max_iterations):
        # 计算梯度 G
        G = np.dot(L, X)

        # 计算更新方向矩阵 A
        A = np.dot(G, X.T) - np.dot(X, G.T)

        # 动态调整步长：使用动量法加速
        tau = step_size
        step_decrease_factor = 0.9  # 动态步长减少因子

        # 使用曲线搜索公式计算下一步
        I = np.eye(X.shape[0])
        Y_tau = np.linalg.inv(I + (tau / 2) * A).dot(I - (tau / 2) * A).dot(X)

        # 计算新的能量
        energy_new = E(Y_tau, L)
        energy_old = E(X, L)

        # 检查能量是否减少
        if energy_new < energy_old:
            X = Y_tau  # 更新 X
            print(f"Iteration {k + 1}, Energy: {energy_new}")

            # 动态步长调整
            step_size = step_size * (1 + beta) if energy_new < energy_old else step_size * step_decrease_factor

        else:
            # 如果不满足条件，则减小步长
            step_size *= step_decrease_factor

        # 检查梯度范数是否满足收敛条件
        if np.linalg.norm(G) < tolerance:
            print(f"Converged after {k + 1} iterations.")
            break

    return X

# def orthogonal_constrained_optimization(mesh, F0, tolerance, energy_tolerance, max_iterations=1000,
#                                         learning_rate=0.001):
#     """
#     使用正交约束优化来最小化谐波能量，并确保结果映射在单位球面上。
#
#     参数:
#         mesh - 输入的网格数据
#         F0 - 初始映射
#         tolerance - 梯度收敛阈值
#         energy_tolerance - 能量收敛阈值
#         max_iterations - 最大迭代次数
#         learning_rate - 学习率（步长控制）
#
#     返回:
#         X - 优化后的映射结果
#     """
#     L = construct_laplacian_matrix(mesh)
#     X = F0.copy()
#
#     for i in range(max_iterations):
#         # 计算梯度
#         gradient = np.zeros_like(X)
#         for dim in range(3):
#             gradient[:, dim] = np.dot(L, X[:, dim])
#
#         # 更新 X，朝负梯度方向移动
#         X -= learning_rate * gradient
#
#         # 将 X 归一化到单位球面
#         norms = np.linalg.norm(X, axis=1, keepdims=True)
#         X /= norms
#
#         # 检查梯度范数是否满足收敛条件
#         gradient_norm = np.linalg.norm(gradient)
#         if gradient_norm < tolerance:
#             print(f"Converged after {i + 1} iterations.")
#             break
#
#     return X

def choose_step_size(F_k, delta, rho, C_k, L, max_h=10):
    h = 0
    tau_k = delta
    gradient = compute_gradient(F_k, L)
    gradient_norm_sq = np.sum(gradient ** 2)

    while h < max_h:
        # 使用当前步长 tau_k 更新 F_k
        F_k_next = Y(F_k, tau_k, L)
        new_energy = E(F_k_next, L)

        # 检查是否满足下降条件
        if new_energy <= C_k + rho * tau_k * gradient_norm_sq:
            return tau_k  # 满足条件，返回该步长

        # 如果不满足条件，减小步长
        tau_k *= delta
        h += 1

    return tau_k  # 返回最终找到的最小步长


def Y(F_k, tau_k, L):
    # 计算梯度
    gradient = compute_gradient(F_k, L)

    # 投影梯度到球面切平面
    projected_gradient = project_gradient_to_sphere(gradient, F_k)

    # 沿测地线方向更新 F_k
    F_k_next = F_k - tau_k * projected_gradient

    # 将 F_k_next 归一化到单位球面
    F_k_next /= np.linalg.norm(F_k_next, axis=1, keepdims=True)
    return F_k_next


# finished
def gauss_map(mesh):
    """Compute the Gauss map for initial parameterization by normalizing vertex normals."""
    # Calculate vertex normals first
    mesh.calculate_vertex_normals()

    # Normalize each vertex normal and update vertex coordinates
    for vertex in mesh.vertices:
        norm_length = np.linalg.norm(vertex.normal)
        if norm_length > 1e-6:
            # Normalize the normal precisely to the unit sphere
            normalized_normal = vertex.normal / norm_length
            vertex.normal = normalized_normal
            # Update vertex coordinates based on the normalized normal
            vertex.x, vertex.y, vertex.z = normalized_normal
        else:
            print(f"Warning: Vertex {vertex.index} has a very small normal vector length.")

        # Verify the normalization
        if not np.allclose(np.linalg.norm(vertex.normal), 1, atol=1e-6):
            print(f"Normalization issue at Vertex {vertex.index}: {vertex.normal}")

    print("Gauss map computed and all vertex normals are on the unit sphere.")

def compute_gradient(F, L):
    # 初始化梯度矩阵
    gradient = np.zeros_like(F)

    # 针对每个维度（x, y, z）分别计算梯度
    for i in range(3):
        grad_component = np.dot(L, F[:, i])
        grad_component = np.clip(grad_component, -1e3, 1e3)  # 限制范围以提高稳定性
        gradient[:, i] = grad_component

    # 将梯度投影到球面切平面上
    return project_gradient_to_sphere(gradient, F)

def project_gradient_to_sphere(grad, F):
    """
    计算投影梯度，即将梯度投影到球面切平面上。
    grad - 当前的梯度向量
    F - 当前点的坐标向量，表示在球面上的位置
    返回：投影到球面切平面的梯度
    """
    # 计算叉积，将梯度投影到球面切平面
    dot_product = np.sum(grad * F, axis=1, keepdims=True)
    projected_grad = grad - dot_product * F # Project gradient onto tangent plane
    return projected_grad

def compute_harmonic_energy(mesh):
    # 构建拉普拉斯矩阵 L，应该是 (2502, 2502) 大小
    L = construct_laplacian_matrix(mesh)
    # 构建初始映射 F0，大小为 (2502, 3)
    F0 = np.array([[vertex.x, vertex.y, vertex.z] for vertex in mesh.vertices])
    # 计算初始谐波能量 E(F0) = 0.5 * sum(F0[:, i].T @ L @ F0[:, i]) for each dimension i
    energy = 0.5 * sum(np.dot(F0[:, i].T, np.dot(L, F0[:, i])) for i in range(3))
    return energy


def E(F, L):
    energy = 0.5 * sum(np.dot(F[:, i], np.dot(L, F[:, i])) for i in range(3))
    # print("E: Current computed energy:", energy)
    return energy

def construct_laplacian_matrix(mesh):
    vertices = mesh.vertices
    facets = mesh.facets
    n = len(vertices)
    L = np.zeros((n, n))

    for face in facets:
        u, v, w = face.a, face.b, face.c

        # Compute cotangent weights without setting negative values to zero
        cot_alpha = cotangent_three_vertices(vertices[w], vertices[u], vertices[v])  # Angle at u
        cot_beta = cotangent_three_vertices(vertices[u], vertices[v], vertices[w])   # Angle at v
        cot_gamma = cotangent_three_vertices(vertices[v], vertices[w], vertices[u])  # Angle at w

        # Update off-diagonal entries
        L[u, v] -= cot_gamma
        L[v, u] -= cot_gamma
        L[v, w] -= cot_alpha
        L[w, v] -= cot_alpha
        L[w, u] -= cot_beta
        L[u, w] -= cot_beta

        # Update diagonal entries
        L[u, u] += cot_beta + cot_gamma
        L[v, v] += cot_gamma + cot_alpha
        L[w, w] += cot_alpha + cot_beta

    return L

def compute_laplacian(vertex):
    """
    Compute the Laplacian for a given vertex based on the positions of neighboring vertices.
    """
    laplacian = [0.0, 0.0, 0.0]  # Initialize to zero vector
    count = 0

    # Iterate over all neighbors (through half-edges)
    start_halfedge = vertex.halfedge
    current_halfedge = start_halfedge

    while True:
        if current_halfedge.opposite is None:
            print(f"Warning: Vertex {vertex.index} has a boundary edge (missing opposite). Skipping.")
            break  # or `continue` if you want to skip only this edge but continue

        # Get the neighbor vertex
        neighbor_vertex = current_halfedge.opposite.vertex
        if neighbor_vertex is not None:
            # Compute the difference and accumulate it
            diff = [
                neighbor_vertex.x - vertex.x,
                neighbor_vertex.y - vertex.y,
                neighbor_vertex.z - vertex.z
            ]
            laplacian = [laplacian[i] + diff[i] for i in range(3)]
            count += 1

        # Move to the next half-edge
        current_halfedge = current_halfedge.opposite.next

        # Stop if we've looped back to the start
        if current_halfedge == start_halfedge or current_halfedge is None:
            break

    # Normalize Laplacian if there are neighbors
    if count > 0:
        laplacian = [l / count for l in laplacian]
        norm = math.sqrt(sum([l ** 2 for l in laplacian]))
        if norm > 1e-6:  # Avoid division by zero
            laplacian = [l / norm for l in laplacian]

    return laplacian

def cotangent_three_vertices(v1, v2, v3):
    """
    Calculate the cotangent of the angle opposite to the edge between v1 and v2 in the triangle (v1, v2, v3).

    Parameters:
    - v1, v2, v3: Vertex instances representing the vertices of a triangle in counter-clockwise order.

    Returns:
    - cotangent_value: Cotangent of the angle at v3, opposite the edge v1-v2.
    """
    # Create vectors from v3 to v1 and from v3 to v2
    vector1 = create_vector(v2.get_vertex(), v1.get_vertex())  # Vector v1 -> v2
    vector2 = create_vector(v3.get_vertex(), v1.get_vertex())  # Vector v1 -> v3

    # Compute dot and cross products
    dot_product = dot(vector1, vector2)
    cross_product_norm = norm(cross_product(vector1, vector2))

    # # Handle cases where the cross product norm is too small to avoid division by zero
    # if cross_product_norm < 1e-6:
    #     print(f"Warning: Small cross product norm between vertices ({v1.index}, {v2.index}, {v3.index}), returning 0.")
    #     return 0.0

    cotangent_value = dot_product / cross_product_norm
    return cotangent_value

def update_direction(X, G):
    # 使用公式 A = GX^T - XG^T 计算更新方向
    A = np.dot(G, X.T) - np.dot(X, G.T)
    return A

def cotangent(he):
    # Compute cotangent of the angle opposite the half-edge
    u = he.vertex.get_vertex()
    v = he.next.vertex.get_vertex()
    w = he.next.next.vertex.get_vertex()
    vec_u = np.array(u) - np.array(w)
    vec_v = np.array(v) - np.array(w)
    cosine = np.dot(vec_u, vec_v)
    sine = np.linalg.norm(np.cross(vec_u, vec_v))
    cotangent = cosine / sine if sine != 0 else 0
    return cotangent


def allclose(v1, v2):
    """Compare if v1 and v2 are close

    v1, v2 - any numerical type or list/tuple of numerical types

    Return bool if vectors are close, up to some epsilon specified in config.py
    """
    v1 = make_iterable(v1)
    v2 = make_iterable(v2)

    elementwise_compare = map(
        (lambda x, y: abs(x - y) < config.EPSILON), v1, v2)
    return functools.reduce((lambda x, y: x and y), elementwise_compare)

def make_iterable(obj):
    """Check if obj is iterable, if not return an iterable with obj inside it.
    Otherwise just return obj.
    obj - any type
    Return an iterable
    """
    try:
        iter(obj)
    except:
        return [obj]
    else:
        return obj

def dot(v1, v2):
    """Dot product(inner product) of v1 and v2

    v1, v2 - python list

    Return v1 dot v2
    """
    elementwise_multiply = map((lambda x, y: x * y), v1, v2)
    return functools.reduce((lambda x, y: x + y), elementwise_multiply)

def norm(vec):
    """ Return the Euclidean norm of a 3d vector.
    vec - a 3d vector expressed as a list of 3 floats.
    """
    return math.sqrt(functools.reduce((lambda x, y: x + y * y), vec, 0.0))


def normalize(vec):
    """Normalize a vector

    vec - python list

    Return normalized vector
    """
    if norm(vec) < 1e-6:
        return [0 for i in range(len(vec))]
    return list(map(lambda x: x / norm(vec), vec))


def cross_product(v1, v2):
    """ Return the cross product of v1, v2.

    v1, v2 - 3d vector expressed as a list of 3 floats.
    """
    # 将 v1 和 v2 转换为列表
    v1 = list(v1)
    v2 = list(v2)
    x3 = v1[1] * v2[2] - v2[1] * v1[2]
    y3 = -(v1[0] * v2[2] - v2[0] * v1[2])
    z3 = v1[0] * v2[1] - v2[0] * v1[1]
    return [x3, y3, z3]

def create_vector(p1, p2):
    """Contruct a vector going from p1 to p2.

    p1, p2 - python list wth coordinates [x,y,z].

    Return a list [x,y,z] for the coordinates of vector
    """
    return list(map((lambda x,y: x-y), p2, p1))