import open3d as o3d
from tqdm import tqdm
import numpy as np
import numpy.linalg as la
from pathlib import Path
import time
import copy


def load_pointcloud_dat(file: Path):
    dtype = np.dtype(
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("pad_12_15", np.uint8, 4),
            ("normal_x", np.float32),
            ("normal_y", np.float32),
            ("normal_z", np.float32),
            ("pad_28_31", np.uint8, 4),
            ("intensity", np.float32),
            ("curvature", np.float32),
            ("pad_40_47", np.uint8, 8),
        ]
    )
    with open(file, "rb") as reader:
        bdata_raw = np.fromfile(reader, dtype=dtype, count=-1)

    # for s in ["x","y","z","normal_x","normal_y","normal_z"]:
    #     vneg = np.min(bdata_raw[s])
    #     vpos = np.max(bdata_raw[s])
    #     if (vneg < -10000) or (10000 < vpos):
    #         print("warn : bdata_raw[%s] has invalid range of %e ~ %e (@%s)"%(s,vneg,vpos,file.name))
    return bdata_raw


def evaluate_degeneracy(bdata_raw):
    J = np.vstack((bdata_raw["normal_x"], bdata_raw["normal_y"], bdata_raw["normal_z"])).T
    U, S, Vt = la.svd(J)
    # H = J.T @ J
    # eigval,eigvec = la.eig(H)

    degen_dir = Vt.T[:, 2]
    if degen_dir[0] < 0:
        degen_dir = -degen_dir

    degen_lamda_raw = S[2] ** 2

    degen_lamda_point_scaled = degen_lamda_raw / bdata_raw.shape[0]
    degen_lamda_maxeig_scaled = degen_lamda_raw / S[0] ** 2
    degen_lamda_trace_scaled = degen_lamda_raw / np.sum(S**2)

    # degen_norm =
    # degen_norm = 1 - degen_lamda_maxeig_scaled
    # degen_norm = 1 - 3 * degen_lamda_trace_scaled  # 3 DoF

    degen_norm = np.clip(1 / degen_lamda_maxeig_scaled, 0, 10)
    degen_vec = degen_norm * degen_dir
    return degen_vec


def o3d_pointcloud_from_XYZINormal(bdata_raw):
    points = np.vstack((bdata_raw["x"], bdata_raw["y"], bdata_raw["z"])).T
    normals = np.vstack((bdata_raw["normal_x"], bdata_raw["normal_y"], bdata_raw["normal_z"])).T
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def o3d_arrow_from_vector(dirvec, origin, scale=1):
    v = np.asarray(dirvec)
    length = np.linalg.norm(v)
    arrow = o3d.geometry.TriangleMesh.create_arrow(
        cylinder_radius=0.02,
        cone_radius=0.05,
        cylinder_height=0.9,
        cone_height=0.1,
    )

    # scaling
    scale_factor = length * scale
    arrow.scale(scale_factor, center=(0, 0, 0))  # 原点を中心にスケーリング

    # rotation
    default_dir = np.array([0.0, 0.0, 1.0])
    target_dir = v / length
    rotation_axis = np.cross(default_dir, target_dir)
    angle = np.arccos(np.dot(default_dir, target_dir))
    if np.abs(angle) < 1e-6:
        R = np.identity(3)
    elif np.abs(angle - np.pi) < 1e-6:
        R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    else:
        R = arrow.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
    arrow.rotate(R, center=(0, 0, 0))

    # transition
    scaled_v = target_dir * scale_factor
    target_center = origin + scaled_v / 2.0
    arrow.translate(target_center, relative=False)

    # etc
    arrow.paint_uniform_color([np.clip(1 * length, 0, 1), 0, 0])
    return arrow


def o3d_update_triangle_mesh(obj, newobj):
    obj.vertices = newobj.vertices
    obj.triangles = newobj.triangles


def o3d_update_pointcloud(obj, newobj):
    obj.points = newobj.points
    obj.normals = newobj.normals


def animate_pointclouds(pointcloud_dict, delay_sec=0.1):
    names, geometries = zip(*pointcloud_dict.items())
    num_geometries = len(geometries[0])
    num_data = len(names)
    state = {"idx": 0, "last_update_time": time.time(), "playing": True}

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="PointCloud Animation")

    render_option = vis.get_render_option()
    render_option.point_size = 10.0
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.point_show_normal = True
    render_option.point_color_option = o3d.visualization.PointColorOption.Normal

    k_geometry = lambda i: "geometry" + str(i)
    k_handler = lambda i: "update_handler" + str(i)

    for i, geom in enumerate(geometries[0]):
        state[k_geometry(i)] = copy.deepcopy(geom)
        state[k_handler(i)] = (
            o3d_update_pointcloud
            if isinstance(geom, o3d.cpu.pybind.geometry.PointCloud)
            else o3d_update_triangle_mesh if isinstance(geom, o3d.cpu.pybind.geometry.TriangleMesh) else None
        )
        vis.add_geometry(state[k_geometry(i)])

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    def draw(vis):
        for i in range(num_geometries):
            state[k_handler(i)](state[k_geometry(i)], geometries[state["idx"]][i])
            vis.update_geometry(state[k_geometry(i)])
        #      = state["update"]
        # state["pointcloud"].points = pointclouds[state["idx"]].points
        # state["pointcloud"].normals = pointclouds[state["idx"]].normals

    def toggle_play(vis):
        state["playing"] = not state["playing"]
        return False

    def next_frame(vis):
        state["playing"] = False
        state["idx"] = (state["idx"] + 1) % num_data
        draw(vis)
        return False

    def previous_frame(vis):
        state["playing"] = False
        state["idx"] = (state["idx"] - 1) % num_data
        draw(vis)
        return False

    def animation_callback(vis):
        if state["playing"]:
            current_time = time.time()
            if current_time - state["last_update_time"] > delay_sec:
                state["last_update_time"] = current_time
                state["idx"] = (state["idx"] + 1) % num_data
                draw(vis)
        return False

    vis.register_key_callback(ord(";"), next_frame)
    vis.register_key_callback(ord(","), previous_frame)
    vis.register_key_callback(ord("."), toggle_play)
    vis.register_animation_callback(animation_callback)
    vis.run()
    vis.destroy_window()


def main():
    path = Path("data/sano_tunnel")
    files = list(path.glob("*.dat"))

    visdata = {}
    for i, file in tqdm(enumerate(sorted(files)), total=len(files)):
        bdata_raw = load_pointcloud_dat(file)
        degen_vec = evaluate_degeneracy(bdata_raw)
        o3d_pc = o3d_pointcloud_from_XYZINormal(bdata_raw)
        o3d_vec = o3d_arrow_from_vector(degen_vec, np.array([0.0, 0.0, 0.0]), 1)
        visdata[file.name] = [o3d_pc, o3d_vec]
    animate_pointclouds(visdata)


if __name__ == "__main__":
    main()
