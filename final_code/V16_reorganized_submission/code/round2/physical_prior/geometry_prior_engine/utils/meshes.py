import drjit as dr
import mitsuba as mi
import numpy as np


def remove_mesh_duplicate_vertices(mesh: mi.Mesh) -> None:

    vertices = dr.unravel(mi.Point3f, mesh.vertex_positions_buffer()).numpy()
    faces = dr.unravel(mi.Point3u, mesh.faces_buffer()).numpy()
    texcoords = dr.unravel(mi.Vector2f, mesh.vertex_texcoords_buffer()).numpy()

    unique_vertices, unique_indices, inverse_indices = np.unique(
        vertices,
        axis=1,
        return_inverse=True,
        return_index=True,
    )
    new_faces = inverse_indices[faces]

    params = mi.traverse(mesh)
    params["vertex_positions"] = dr.ravel(mi.Point3f(unique_vertices))
    params["faces"] = dr.ravel(mi.Point3u(new_faces))
    if mesh.has_vertex_texcoords():
        new_texcoords = texcoords[:, unique_indices]
        params["vertex_texcoords"] = dr.ravel(mi.Vector2f(new_texcoords))
    params.update()
    if mesh.has_vertex_normals():
        mesh.recompute_vertex_normals()
