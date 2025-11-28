import numpy as np
import open3d as o3d


def create_scaled_arrow_from_vector(v: np.ndarray):
    """
    (3,)形状のNumPyベクトル v を、デフォルトの矢印形状をスケーリングして描画する。
    デフォルトの矢印はZ軸正方向に伸びる。

    Args:
        v (np.ndarray): 描画したいベクトル (例: np.array([1, 2, 3]))
        # デフォルトのcreate_arrowの引数 (全長9.0)
        default_shaft_radius (float): デフォルト矢柄の半径。
        default_cone_radius (float): デフォルト矢頭の半径。
        default_cylinder_height (float): デフォルト矢柄の高さ。
        default_cone_height (float): デフォルト矢頭の高さ。

    Returns:
        o3d.geometry.TriangleMesh: 矢印メッシュオブジェクト。
    """
    v = np.asarray(v)
    length = np.linalg.norm(v)

    if length < 1e-6:
        # ベクトルの長さがゼロに近い場合は描画をスキップ
        return o3d.geometry.TriangleMesh()

    # 1. デフォルトサイズの矢印を作成
    arrow = o3d.geometry.TriangleMesh.create_arrow()
    default_total_length = 9.0

    # 2. ターゲットの長さに合わせて全体をスケーリング
    scale_factor = length / default_total_length
    # arrow.scale() は中心を指定しないと原点を中心にスケーリングされます。
    # 矢印の根元 (Z=-cylinder_height) が原点になるように調整するため、
    # 回転後、最後にtranslateします。
    arrow.scale(scale_factor, center=(0, 0, 0))  # 原点を中心にスケーリング

    # 3. 回転行列 R の計算 (デフォルトのZ軸を v の方向にアライメント)
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

    # 4. 回転の適用
    # スケーリング済みの矢印を回転
    arrow.rotate(R, center=(0, 0, 0))

    # 5. 移動量の適用 (原点からベクトルの中心へ)
    # arrow.translate() は relative=True (デフォルト) だと現在の位置からの相対移動。
    # relative=False だと絶対位置への移動。
    # 原点からベクトルの半分まで移動すると矢印の中心がベクトルの中間点に来る。
    # 矢印の根元が原点になるようにしたい場合
    # 矢印の全長は 'length' になっているので、根元は Z軸上 -length/2 の位置。
    # これを (0,0,0) に合わせるため、length/2 だけ移動。
    # スケーリング後の矢印の幾何学的中心は原点 (0,0,0) にある。
    # これをベクトル v の方向かつ長さ v/2 だけ移動。

    # スケーリングによって矢印の根元が(0,0,0)からずれてしまっているので、
    # 矢印の幾何学的中心が原点にある状態で、ベクトルvの半分まで移動させれば、
    # 矢印の根元が原点に来る。
    arrow.translate(v / 2.0, relative=False)

    arrow.paint_uniform_color([0.8, 0.2, 0.2])

    return arrow


# --- 使用例 ---
if __name__ == "__main__":
    # 描画したいベクトル
    vector_v = np.array([1.5, 0.5, 1.0])

    # ベクトルから矢印メッシュを作成
    arrow_mesh = create_scaled_arrow_from_vector(vector_v)

    # 可視化
    origin = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    origin.translate([0, 0, 0])
    origin.paint_uniform_color([0.1, 0.1, 0.8])

    coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    o3d.visualization.draw_geometries([arrow_mesh, origin, coords])
