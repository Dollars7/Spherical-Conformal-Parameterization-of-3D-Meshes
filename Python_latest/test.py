''' read the file line by line to understanding how the half edge load
an off file and to save the half edge mesh as an obj file.

author:
    zhangsihao yang
'''
import os
import numpy as np
import math
import mesh
from mesh import HalfedgeMesh
import optimization
from optimization import compute_harmonic_energy

# Debug flag for controlled output
DEBUG = True

def debug_print(*args, **kwargs):
    """Print debug messages if DEBUG is True."""
    if DEBUG:
        print(*args, **kwargs)

# Save functions for different formats
def save_spherical_mapping_as_obj(mesh, F_k, file_name):
    """Save the spherical mapping as an OBJ file with vertices and faces."""
    with open(file_name, 'w') as open_file:
        for i in range(len(F_k)):
            open_file.write("v {} {} {}\n".format(F_k[i][0], F_k[i][1], F_k[i][2]))

        for facet in mesh.facets:
            if facet.a is not None and facet.b is not None and facet.c is not None:
                open_file.write("f {} {} {}\n".format(facet.a + 1, facet.b + 1, facet.c + 1))
    print(f"Spherical mapping saved as '{file_name}' with vertices and faces.")

def save_halfmesh_as_obj(mesh, file_name):
    """Save the half-edge mesh as an OBJ file."""
    with open(file_name, 'w') as open_file:
        for vertex in mesh.vertices:
            lv = vertex.get_vertex()
            open_file.write("v {} {} {} \n".format(lv[0], lv[1], lv[2]))

        for face in mesh.facets:
            if face.a is None or face.b is None or face.c is None:
                debug_print(f"Error: Facet has None values: {face}")
                continue
            open_file.write("f {} {} {}\n".format(face.a + 1, face.b + 1, face.c + 1))


def save_for_g3dogl(mesh, file_name):
    """Save the half-edge mesh in a G3dOGL compatible format."""
    with open(file_name, 'w') as file:
        for vertex in mesh.vertices:
            vx, vy, vz = vertex.get_vertex()
            nx, ny, nz = vertex.normal
            opos_x, opos_y, opos_z = vx * 0.7, vy * 0.7, vz * 0.7
            onx, ony, onz = nx, ny, nz
            file.write(f"Vertex {vertex.index + 1} {vx:.6f} {vy:.6f} {vz:.6f}\n")
            file.write(f"    {{normal=({nx:.6f}, {ny:.6f}, {nz:.6f})\n")
            file.write(f"     Opos=({opos_x:.6f}, {opos_y:.6f}, {opos_z:.6f})\n")
            file.write(f"     Onormal=({onx:.6f}, {ony:.6f}, {onz:.6f})}}\n\n")

# 4. Optional: Save normals to a file for visualizatioan
def save_normals_as_obj(mesh, file_name):
    """Save the normals as an OBJ file with vertices and faces."""
    with open(file_name, 'w') as open_file:
        # Write each normal as a vertex on the unit sphere
        for vertex in mesh.vertices:
            nx, ny, nz = vertex.normal  # Assuming normals are calculated
            open_file.write("v {} {} {}\n".format(nx, ny, nz))

        # Write face indices
        for facet in mesh.facets:
            if facet.a is not None and facet.b is not None and facet.c is not None:
                open_file.write("f {} {} {}\n".format(facet.a + 1, facet.b + 1, facet.c + 1))

    print(f"Normals with faces saved as '{file_name}'")

def save_gauss_map_as_obj(mesh, file_name):
    """Save Gauss map as an OBJ file with vertex normals projected to the unit sphere and faces."""
    with open(file_name, 'w') as open_file:
        for vertex in mesh.vertices:
            # 使用归一化后的法向量 (vertex.normal) 作为单位球投影的顶点
            nx, ny, nz = vertex.normal
            open_file.write("v {} {} {}\n".format(nx, ny, nz))

        for facet in mesh.facets:
            if facet.a is not None and facet.b is not None and facet.c is not None:
                open_file.write("f {} {} {}\n".format(facet.a + 1, facet.b + 1, facet.c + 1))
    print(f"Gauss map with faces saved as '{file_name}'")


print('--------------------Hello Project3--------------------')

# data_path = 'tests/data/bunny.off'
data_path = 'tests/data/brain_python.off'
# data_path = 'tests/data/brain.obj'
# data_path = 'tmp.obj'
print (data_path)
output_dir = 'output'
mesh = mesh.HalfedgeMesh(data_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

try:
    # Check if the mesh loaded successfully
    if len(mesh.vertices) == 0 or len(mesh.facets) == 0:
        print("Error: Mesh failed to load properly.")
    else:
        print('----------------------------------------')
        print(
            f"Loaded mesh with {len(mesh.vertices)} vertices, {len(mesh.facets)} facets, {len(mesh.halfedges)} halfedges")
        print('----------------------------------------')

        # 1. Calculate Vertex Normals
        mesh.calculate_vertex_normals()
        print("Vertex Normals Calculated")
        save_normals_as_obj(mesh, os.path.join(output_dir, 'normals.obj'))

        # 2. Initialize Gauss Map by projecting vertices onto the unit sphere
        optimization.gauss_map(mesh)
        save_gauss_map_as_obj(mesh, os.path.join(output_dir, 'gauss_map.obj'))

        # 3. Apply Spherical Constraint (if needed) to ensure all vertices are on the unit sphere
        # Validate spherical constraint 检查球面约束 - 使用循环检查每个顶点是否在单位球上。
        for vertex in mesh.vertices:
            length = math.sqrt(vertex.x ** 2 + vertex.y ** 2 + vertex.z ** 2)
            assert abs(length - 1.0) < 1e-6, f"Vertex {vertex.index} is not on the unit sphere"
        print("All vertices are on the unit sphere.")

        mesh.apply_spherical_constraint()
        # Save the constrained initial mesh
        save_halfmesh_as_obj(mesh, 'output/initial_constrained_mesh.obj')
        print("Initial constrained mesh saved as 'initial_constrained_mesh.obj'")

        # 4. Compute initial harmonic energy
        initial_energy = compute_harmonic_energy(mesh)
        print(f"Initial Harmonic Energy: {initial_energy}")

        # ---- Fast Algorithm Section ----

        # Preparing for the optimization
        F0 = np.array([[vertex.x, vertex.y, vertex.z] for vertex in mesh.vertices])
        F0 = F0 / np.max(np.abs(F0))  # Normalize F0
        debug_print("Normalized F0:", F0)
        debug_print("Shape of F0:", F0.shape)

        L = optimization.construct_laplacian_matrix(mesh)
        debug_print("Shape of L:", L.shape)

        # Set algorithm parameters
        rho = 0.5
        delta = 0.001  # Smaller initial step size
        epsilon = 1e-6  # Adjust convergence threshold
        xi = 0.9

        tolerance = 1e-7
        energy_tolerance = 1e-7
        max_iter = 5000
        learning_rate = 0.001
        max_smoothing_iterations = 3

        previous_energy = initial_energy
        F_k = np.array([[vertex.x, vertex.y, vertex.z] for vertex in mesh.vertices])

        # for iteration in range(max_smoothing_iterations):
        #     print(f"--- Iteration {iteration + 1} ---")
        #
        #     # 1. Apply Laplace-Beltrami Eigen Projection to get a smoother initial map
        #     F_k = optimization.weighted_lb_eigen_projection(mesh)
        #     try:
        #         F_k = optimization.orthogonal_constrained_optimization(mesh, F_k, tolerance,
        #                                                                max_iterations=max_iter,
        #                                                                initial_step_size=learning_rate)
        #         assert np.allclose(np.linalg.norm(F_k, axis=1), 1,
        #                            atol=1e-6), "Some vertices are not on the unit sphere."
        #
        #         current_energy = optimization.E(F_k, optimization.construct_laplacian_matrix(mesh))
        #         print(
        #             f"Orthogonal Constrained Optimization - Final Minimized Harmonic Energy after iteration {iteration + 1}: {current_energy}")
        #
        #         output_file_ortho = os.path.join(output_dir, f'spherical_mapping_{iteration + 1}.obj')
        #         save_spherical_mapping_as_obj(mesh, F_k, output_file_ortho)
        #
        #
        #         # # Check if the energy has converged
        #         # if abs(current_energy - previous_energy) < energy_tolerance:
        #         #     print("Converged to stable energy level. Stopping iterations.")
        #         #     break
        #
        #         previous_energy = current_energy
        #
        #     except Exception as e:
        #         print(f"An error occurred during Orthogonal Constrained Optimization in iteration {iteration + 1}: {e}")
        #         break
        #
        # # Run the optimization algorithm
        try:
            smoothed_map = optimization.weighted_lb_eigen_projection(mesh)
            optimized_F_fast = optimization.fast_algorithm(F_k, rho, delta, xi, epsilon, mesh, max_iter=6000)
            assert np.allclose(np.linalg.norm(optimized_F_fast, axis=1), 1,
                               atol=1e-6), "Some vertices are not on the unit sphere."

            final_harmonic_energy_fast = optimization.E(optimized_F_fast, L)
            print("Fast Algorithm - Final Minimized Harmonic Energy:", final_harmonic_energy_fast)

            output_file_fast = os.path.join(output_dir, 'spherical_mapping_fast.obj')
            save_spherical_mapping_as_obj(mesh, optimized_F_fast, output_file_fast)
            print(f"Fast Algorithm result saved as '{output_file_fast}'")

        except Exception as e:
            print(f"An error occurred while fast_algorithm processing the mesh: {e}")

            # ---- Run Orthogonal Constrained Optimization ----
        try:
            optimized_F_ortho = optimization.orthogonal_constrained_optimization(mesh, F0, tolerance,
                                                                                 max_iterations=max_iter,
                                                                                 initial_step_size=learning_rate)
            # Verify that all vertices remain on the unit sphere
            assert np.allclose(np.linalg.norm(optimized_F_ortho, axis=1), 1,
                               atol=1e-6), "Some vertices are not on the unit sphere."

            final_harmonic_energy_ortho = optimization.E(optimized_F_ortho, L)
            print("Orthogonal Constrained Optimization - Final Minimized Harmonic Energy:", final_harmonic_energy_ortho)

            output_file_ortho = os.path.join(output_dir, 'spherical_mapping_ortho.obj')
            save_spherical_mapping_as_obj(mesh, optimized_F_ortho, output_file_ortho)
            print(f"Orthogonal Constrained Optimization result saved as '{output_file_ortho}'")

        except Exception as e:
            print(f"Error in Orthogonal Constrained Optimization: {e}")

except Exception as e:
    print(f"An error occurred while loading or testing the mesh: {e}")
