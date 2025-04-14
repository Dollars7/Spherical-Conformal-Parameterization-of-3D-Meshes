import sys
import math
from optimization import allclose, make_iterable, dot, norm, normalize, cross_product, create_vector

class HalfedgeMesh:
    def __init__(self, filename=None, vertices=[], halfedges=[], facets=[]):
        self.vertices = vertices  # List of Vertex types
        self.halfedges = halfedges  # List of HalfEdge types
        self.facets = facets  # List of Facet types
        self.filename = filename  # File path for mesh data
        # dictionary of all the edges given indexes
        # Which is faster?
        self.edges = {} # Empty dictionary to store edge pairs and their halfedges
        if filename:
            self.vertices, self.halfedges, self.facets, self.edges = \
                    self.read_file(filename)

    def __eq__(self, other):
        return (isinstance(other, type(self)) and 
            (self.vertices, self.halfedges, self.facets) ==
            (other.vertices, other.halfedges, other.facets))

    def __hash__(self):
        return (hash(str(self.vertices)) ^ hash(str(self.halfedges)) ^ hash(str(self.facets)) ^ 
            hash((str(self.vertices), str(self.halfedges), str(self.facets))))

    # 文件类型来调用相应的解析器
    def read_file(self, filename):
        """Determine the type of file and use the appropriate parser."""
        try:
            # 根据文件扩展名判断使用文本还是二进制模式
            file_extension = filename.split('.')[-1].lower()
            if file_extension == "off":
                # 以文本模式打开 OFF 文件
                with open(filename, 'r') as file:
                    first_line = file.readline().strip()
                    if first_line.lower() != 'off' and first_line.lower() != 'stoff':  # 检查文件第一行是否是 "OFF"
                        raise ValueError(f"Invalid OFF file format: {filename}")
                    if first_line.startswith('OFF') or first_line.startswith('STOFF'):
                        return self.parse_off(file)  # 使用 parse_off 解析 OFF 文件
                    else:
                        raise ValueError(f"Unsupported OFF file format: {filename}")
            else:
                # 以文本模式打开 OBJ 文件
                with open(filename, 'r') as file:
                    first_line = file.readline().strip().lower()
                    while first_line.startswith('#') or not first_line:  # Skip comment lines and blank lines
                        first_line = file.readline().strip().lower()

                    file.seek(0)  # Reset to beginning of file for parsing

                    if first_line.startswith('v ') or file_extension == "obj":
                        return self.parse_obj(file)  # Use parse_obj for OBJ files
                    else:
                        raise ValueError(f"Unsupported file format: {filename}")

        except IOError as e:
            print(f"I/O error({e.errno}): {e.strerror}")
        except ValueError as e:
            print(f"Value error: {e}")

        # Return empty structures if there was an error
        return [], [], [], {}

    def read_off_vertices(self, file_object, number_vertices):
        """Read each line of the file_object and return a list of Vertex types.
        The list will be as [V1, V2, ..., Vn] for n vertices
        Return a list of vertices.
        """
        vertices = []
        # Read all the vertices in
        for index in range(number_vertices):
            line = file_object.readline().split()
            #print(line)
            try:
                # convert strings to floats
                line = list(map(float, line))
            except ValueError as e:
                raise ValueError("vertices " + str(e))

            vertices.append(Vertex(line[0], line[1], line[2], index))

        return vertices

    def parse_build_halfedge_off(self, file_object, number_facets, vertices):
        facets = []
        Edges = {}  # Dictionary to store halfedges based on vertex pairs (u, v)
        halfedge_count = 0

        # First pass: Create halfedges without setting opposites
        for index in range(number_facets):
            line = list(map(int, file_object.readline().split()))
            if line[0] != 3:
                raise ValueError(f"Facet {index} is not a triangle!")

            facet = Facet(line[1], line[2], line[3], index)
            facets.append(facet)

            # Define the edges of this facet
            all_facet_edges = list(zip(line[1:], line[2:] + [line[1]]))

            for i in range(3):
                u, v = all_facet_edges[i]

                if (u, v) not in Edges:
                    Edges[(u, v)] = Halfedge(vertex=vertices[u], facet=facet, index=halfedge_count)
                    halfedge_count += 1
                    # Set the half-edge for the vertex if it hasn't been set
                    if vertices[u].halfedge is None:
                        vertices[u].halfedge = Edges[(u, v)]
                # else:
                #     print(f"Duplicate half-edge detected between vertices {u} and {v}.")
                # Assign facet and vertex reference
                Edges[(u, v)].facet = facet
                vertices[v].halfedge = Edges[(u, v)]

            # Set the halfedge of the facet to one of its halfedges
            facet.halfedge = Edges[all_facet_edges[0]]

            # Set next and previous pointers for the halfedges within the facet
            for i in range(3):
                u, v = all_facet_edges[i]
                next_edge = all_facet_edges[(i + 1) % 3]
                prev_edge = all_facet_edges[(i - 1) % 3]
                Edges[(u, v)].next = Edges[next_edge]
                Edges[(u, v)].prev = Edges[prev_edge]

        # Second pass: Set opposites
        for (u, v), halfedge in Edges.items():
            if (v, u) in Edges:
                halfedge.opposite = Edges[(v, u)]
                Edges[(v, u)].opposite = halfedge
            # else:
            #     print(f"Warning: No opposite edge found for edge from vertex {u} to {v} "
            #           f"({vertices[u].x}, {vertices[u].y}, {vertices[u].z}) to "
            #           f"({vertices[v].x}, {vertices[v].y}, {vertices[v].z}). "
            #           f"Total facets: {len(facets)}, Total vertices: {len(vertices)}.")
        return facets, Edges

    def build_halfedge(self, vertices, facets):
        """Builds halfedge structure from vertices and facets."""
        edges = {}
        halfedges = []

        for facet in facets:
            facet_vertices = [facet.a, facet.b, facet.c]
            first_halfedge = None
            prev_halfedge = None

            for i in range(len(facet_vertices)):
                start = facet_vertices[i]
                end = facet_vertices[(i + 1) % len(facet_vertices)]

                # 如果边不存在，则创建新的半边
                if (start, end) not in edges:
                    he = Halfedge(vertex=vertices[end], facet=facet, index=len(halfedges))
                    halfedges.append(he)
                    edges[(start, end)] = he

                    # # 将 halfedge 添加到起点顶点的 halfedges 列表中
                    # vertices[start].halfedges.append(he)

                    # 确认half-edge环的连接
                    if first_halfedge is None:
                        first_halfedge = he

                    if prev_halfedge is not None:
                        prev_halfedge.next = he
                        he.prev = prev_halfedge

                    prev_halfedge = he

                # Link opposite halfedges
                if (end, start) in edges:
                    edges[(start, end)].opposite = edges[(end, start)]
                    edges[(end, start)].opposite = edges[(start, end)]

            # Close the loop of the facet's halfedges
            if first_halfedge and prev_halfedge:
                first_halfedge.prev = prev_halfedge
                prev_halfedge.next = first_halfedge

            facet.halfedge = first_halfedge
            # 调试输出: 确认half-edge环
            # print(f"Facet {facet.index}: first halfedge = {first_halfedge.index}, "
            #       f"vertices = {facet_vertices}")
        return vertices, halfedges, facets, edges

    def parse_off(self, file_object):
        """Parses OFF files

        Returns a HalfedgeMesh
        """
        facets, halfedges, vertices = [], [], []
        vertices_faces_edges_counts = list(map(int, file_object.readline().split()))
        # print(vertices_faces_edges_counts)
        number_vertices = vertices_faces_edges_counts[0]
        vertices = self.read_off_vertices(file_object, number_vertices)

        number_facets = vertices_faces_edges_counts[1]
        facets, Edges = self.parse_build_halfedge_off(file_object,
                                                      number_facets, vertices)
        i = 0
        for key, value in Edges.items():
            value.index = i
            halfedges.append(value)
            i += 1

        return vertices, halfedges, facets, Edges
    # Implement the parse_obj and parse_ply functions
    def parse_obj(self, file_object):
        """Parses OBJ files and builds the halfedge structure."""
        vertices = []
        facets = []

        # Parse the OBJ file line by line
        for line in file_object:
            line = line.strip()  # Remove any leading/trailing whitespaces
            if line.lower().startswith('v '):  # Handle vertices, allow for lowercase 'v'
                parts = line.split()
                vertices.append(Vertex(float(parts[1]), float(parts[2]), float(parts[3]), len(vertices)))
            elif line.lower().startswith('f '):  # Handle faces, allow for lowercase 'f'
                parts = line.split()
                # OBJ format indices start from 1, so subtract 1 for 0-based indexing
                facets.append(Facet(int(parts[1].split('/')[0]) - 1,
                                    int(parts[2].split('/')[0]) - 1,
                                    int(parts[3].split('/')[0]) - 1, len(facets)))

        # Build halfedge structure from vertices and facets
        return self.build_halfedge(vertices, facets)

    def get_halfedge(self, u, v):
        """Retrieve halfedge with starting vertex u and target vertex v
        u - starting vertex
        v - target vertex
        Returns a halfedge
        """
        return self.edges[(u, v)]

    def calculate_vertex_normals(self):
        # Step 1: Initialize vertex normals to zero for accumulation
        for vertex in self.vertices:
            vertex.normal = [0.0, 0.0, 0.0]

        # Step 2: Accumulate the normalized face normals to each vertex
        for facet in self.facets:
            he1 = facet.halfedge
            he2 = he1.next
            he3 = he2.next

            # Calculate the vectors of the two edges of the facet
            # Calculate the vectors of the two edges of the facet
            edge1 = create_vector([he1.vertex.x, he1.vertex.y, he1.vertex.z],
                                  [he2.vertex.x, he2.vertex.y, he2.vertex.z])
            edge2 = create_vector([he1.vertex.x, he1.vertex.y, he1.vertex.z],
                                  [he3.vertex.x, he3.vertex.y, he3.vertex.z])

            # Compute the face normal using the cross product
            face_normal = cross_product(edge1, edge2)
            face_normal = normalize(face_normal)  # Ensure the normal is a unit vector

            # Add this normalized face normal to each vertex in the facet
            for he in [he1, he2, he3]:
                vertex = he.vertex
                vertex.normal = [vertex.normal[i] + face_normal[i] for i in range(3)]

        # Step 3: Normalize accumulated normals for each vertex
        for vertex in self.vertices:
            norm_length = norm(vertex.normal)
            if norm_length > 1e-6:
                vertex.normal = [coord / norm_length for coord in vertex.normal]
            else:
                print(f"Warning: Vertex {vertex.index} has a very small accumulated normal length.")
                vertex.normal = [0.0, 0.0, 1.0]  # Default to a unit vector in z-direction if zero-length

        # # Debug: Print vertex normals and their magnitudes
        # for vertex in self.vertices:
        #     normal_length = norm(vertex.normal)
        #     print(f"Vertex {vertex.index} normal: {vertex.normal}, length: {normal_length}")

    # 在 HalfedgeMesh 类中定义 apply_spherical_constraint 函数
    def apply_spherical_constraint(self):
        for vertex in self.vertices:
            # 计算顶点位置的向量长度（即欧几里得范数）
            length = math.sqrt(vertex.x ** 2 + vertex.y ** 2 + vertex.z ** 2)

            # 防止除以零
            if length > 1e-6:
                # 将顶点位置归一化到单位球面上
                vertex.x /= length
                vertex.y /= length
                vertex.z /= length

    def debug_halfedge_structure(self):
        """Debugging output and consistency checks for the half-edge structure."""
        for he in self.halfedges:
            print(f"Half-edge {he.index}: vertex = {he.vertex.index} "
                  f"({he.vertex.x}, {he.vertex.y}, {he.vertex.z}), "
                  f"next = {he.next.index if he.next else 'None'}, "
                  f"prev = {he.prev.index if he.prev else 'None'}, "
                  f"opposite = {he.opposite.index if he.opposite else 'None'}, "
                  f"incident face = f{he.facet.index if he.facet else 'None'}")
            # Consistency checks
            if he.next and he.next.prev != he:
                print(f"Warning: Half-edge {he.index} next.prev inconsistency.")
            if he.prev and he.prev.next != he:
                print(f"Warning: Half-edge {he.index} prev.next inconsistency.")
            if he.opposite and he.opposite.opposite != he:
                print(f"Warning: Half-edge {he.index} opposite.opposite inconsistency.")

class Vertex:
    def __init__(self, x=0, y=0, z=0, index=None, halfedge=None):
        """Create a vertex with given index at given point.
        x        - x-coordinate of the point
        y        - y-coordinate of the point
        z        - z-coordinate of the point
        index    - integer index of this vertex
        halfedge - a halfedge that points to the vertex
        """
        self.x = x
        self.y = y
        self.z = z
        self.index = index
        self.halfedge = halfedge

    def __eq__(x, y):
        return x.__key() == y.__key() and type(x) == type(y)

    def __key(self):
        return (self.x, self.y, self.z, self.index)

    def __hash__(self):
        return hash(self.__key())

    def get_vertex(self):
        return [self.x, self.y, self.z]


class Facet:

    def __init__(self, a=-1, b=-1, c=-1, index=None, halfedge=None):
        """Create a facet with the given index with three vertices.

        a, b, c - indices for the vertices in the facet, counter clockwise.
        index - index of facet in the mesh
        halfedge - a Halfedge that belongs to the facet
        """
        self.a = a
        self.b = b
        self.c = c
        self.index = index
        # halfedge going ccw around this facet.
        self.halfedge = halfedge

    def __eq__(self, other):
        return self.a == other.a and self.b == other.b and self.c == other.c \
            and self.index == other.index and self.halfedge == other.halfedge

    def __hash__(self):
        return hash(self.halfedge) ^ hash(self.a) ^ hash(self.b) ^ \
            hash(self.c) ^ hash(self.index) ^ \
            hash((self.halfedges, self.a, self.b, self.c, self.index))

    def get_normal(self):
        """Calculate the normal of facet

        Return a python list that contains the normal
        """
        vertex_a = [self.halfedge.vertex.x, self.halfedge.vertex.y,
                    self.halfedge.vertex.z]

        vertex_b = [self.halfedge.next.vertex.x, self.halfedge.next.vertex.y,
                    self.halfedge.next.vertex.z]

        vertex_c = [self.halfedge.prev.vertex.x, self.halfedge.prev.vertex.y,
                    self.halfedge.prev.vertex.z]

        # create edge 1 with vector difference
        edge1 = [u - v for u, v in zip(vertex_b, vertex_a)]
        edge1 = normalize(edge1)
        # create edge 2 ...
        edge2 = [u - v for u, v in zip(vertex_c, vertex_b)]
        edge2 = normalize(edge2)

        # cross product
        normal = cross_product(edge1, edge2)

        normal = normalize(normal)

        return normal


class Halfedge:

    def __init__(self, next=None, opposite=None, prev=None, vertex=None,
                 facet=None, index=None):
        """Create a halfedge with given index.
        """
        self.opposite = opposite
        self.next = next
        self.prev = prev
        self.vertex = vertex
        self.facet = facet
        self.index = index

    def __eq__(self, other):
        return (self.vertex == other.vertex) and \
               (self.prev.vertex == other.prev.vertex) and \
               (self.index == other.index)

    def __hash__(self):
        return hash(self.opposite) ^ hash(self.next) ^ hash(self.prev) ^ \
                hash(self.vertex) ^ hash(self.facet) ^ hash(self.index) ^ \
                hash((self.opposite, self.next, self.prev, self.vertex,
                    self.facet, self.index))

    def get_angle_normal(self):
        """Calculate the angle between the normals that neighbor the edge.
        Return an angle in radians
        """
        a = self.facet.get_normal()
        b = self.opposite.facet.get_normal()

        dir = [self.vertex.x - self.prev.vertex.x,
               self.vertex.y - self.prev.vertex.y,
               self.vertex.z - self.prev.vertex.z]
        dir = normalize(dir)
        ab = dot(a, b)
        args = ab / (norm(a) * norm(b))

        if allclose(args, 1):
            args = 1
        elif allclose(args, -1):
            args = -1
        assert (args <= 1.0 and args >= -1.0)
        angle = math.acos(args)
        if not (angle % math.pi == 0):
            e = cross_product(a, b)
            e = normalize(e)
            vec = dir
            vec = normalize(vec)
            if (allclose(vec, e)):
                return angle
            else:
                return -angle
        else:
            return 0
