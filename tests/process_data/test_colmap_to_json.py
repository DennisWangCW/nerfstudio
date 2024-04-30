"""
Process images test
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from nerfstudio.process_data import colmap_utils
from nerfstudio.process_data import equirect_utils, process_data_utils
import json
import struct
from scipy.spatial.transform import Rotation as R
import collections
from typing import Any, Dict, Literal, Optional, Union
from plyfile import PlyData

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)
    
def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [  # type: ignore
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    images = {}
    qvecs, tvecs = [], []
    point_ids = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":   # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            qvecs.append(qvec)
            tvecs.append(tvec)
            point_ids.append(point3D_ids)
        print("point3d_ids shape: ", point3D_ids.shape)
        print("xys shape: ", xys.shape)
    return qvecs, tvecs, point_ids


def read_extrinsics_text(path):
    """
    Taken from https://github.com/colmap/colmap/blob/dev/scripts/python/read_write_model.py
    """
    images = {}
    qvecs, tvecs = [], []
    point_ids = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                qvecs.append(qvec)
                tvecs.append(tvec)
                point_ids.append(point3D_ids)
        print("point3d_ids shape (txt): ", point3D_ids.shape)
        print("xys shape (txt): ", xys.shape)
    return qvecs, tvecs, point_ids


def write_arrays_to_txt(filename, array_list):
    with open(filename, 'w') as f:
        for array in array_list:
            line = ' '.join(map(str, array))
            f.write(line + '\n')


def json_to_colmap(json_file: Path, camera_filename: Path, image_filename: Path, keep_original_world_coordinate: bool = False):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    #### preparing data for cameras.txt
    w = data["w"]
    h = data["h"]
    fl_x = data["fl_x"]
    fl_y = data["fl_y"]
    cx = data["cx"]
    cy = data["cy"]
    k1 = data["k1"]
    k2 = data["k2"]
    p1 = data["p1"]
    p2 = data["p2"]
    params = [fl_x, fl_y, cx, cy, k1, k2, p1, p2]
    camera_model = data["camera_model"]
    camera_id = 1

    cameras = {}
    cameras[camera_id] = Camera(
                id=camera_id, model=camera_model, width=w, height=h, params=np.array(params)
            )
    camera_path = Path("/workspace/write/cameras.txt")
    write_cameras_text(cameras, path=camera_path)

    # write cameras.txt
    print("Begin to write cameras.txt......")
    with open(camera_filename, 'w') as f:
        line1 = "# Camera list with one line of data per camera:"
        line2 = "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"
        line3 = "# Number of cameras: 1"
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")
        camera_info = []
        camera_info.append(str(camera_id))
        camera_info.append(camera_model)
        camera_info.append(str(w))
        camera_info.append(str(h))
        camera_info.append(str(fl_x))
        camera_info.append(str(fl_y))
        camera_info.append(str(cx))
        camera_info.append(str(cy))
        camera_info.append(str(k1))
        camera_info.append(str(k2))
        camera_info.append(str(p1))
        camera_info.append(str(p2))
        line = " ".join(camera_info)
        f.write(line + "\n")
    


    print("Done writing write cameras.txt......")
    #### preparing data for images.txt
    original_paths = []
    image_ids = []
    image_names = []
    frames = data["frames"]
    tvecs, qvecs = [], []
    for frame in frames:
        image_id = frame["colmap_im_id"]
        file_path = frame["file_path"]
        original_path = frame["original_path"]
        transform_matrix =  np.array(frame["transform_matrix"])
        c2w = transform_matrix
        if not keep_original_world_coordinate:
            c2w[2, :] *= -1
            c2w = c2w[np.array([0, 2, 1, 3]), :]
        c2w[0:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)
        translation = w2c[:3,3]
        rotation = w2c[:3,:3]
        tvec = translation
        qvec = R.from_matrix(rotation).as_quat()
        # qvec = rotmat2qvec(rotation)
        qvec = qvec[np.array([3, 0, 1, 2])]
        image_name = file_path
        qvecs.append(qvec)
        tvecs.append(tvec)
        image_ids.append(image_id)
        image_names.append(image_name)
        original_paths.append(original_path)

    # write images.txt
    qvecs = [list(qvec) for qvec in qvecs]
    tvecs = [list(tvec) for tvec in tvecs]
    # camera_ids = [camera_id for _ in image_ids]
    camera_ids = [camera_id for _ in image_ids]
    images = {}
    image_infos = []
    for im_id, qvec, tvec, cam_id, im_name in zip(image_ids, qvecs, tvecs, camera_ids, image_names):
        image_info = [im_id] + qvec + tvec + [cam_id] + [im_name]
        image_info = [str(im_info) for im_info in image_info]
        image_infos.append(image_info)

        images[im_id] = Image(
                id=im_id,
                qvec=np.array(qvec),
                tvec=np.array(tvec),
                camera_id=cam_id,
                name=im_name,
                xys=np.zeros([1,2]),
                point3D_ids=np.zeros([1]),
        )
    image_path = Path("/workspace/write/images.txt")
    write_images_text(images, path = image_path)

    print("Begin to write images.txt......")    
    with open(image_filename, 'w') as f:
        line1 = "# Image list with two lines of data per image:"
        line2 = "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME"
        line3 = "#   POINTS2D[] as (X, Y, POINT3D_ID)"
        line4 = "# Number of images: {}, mean observations per image: {}".format(len(image_infos), 0)
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")
        f.write(line4 + "\n")

        for image_info in image_infos:
            line = ' '.join(image_info)
            f.write(line + '\n')
    print("Done writing images.txt......")
    
    return qvecs, tvecs, original_paths


def read_points3D_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, num_bytes=43, format_char_sequence="QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            track_elems = read_next_bytes(fid, num_bytes=8 * track_length, format_char_sequence="ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
            )
    return points3D

def read_points3D_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                image_ids = np.array(tuple(map(int, elems[8::2])))
                point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs
                )
    return points3D

def create_ply_from_colmap(
    filename: str, recon_dir: Path, output_dir: Path, applied_transform: Union[torch.Tensor, None]
) -> None:
    """Writes a ply file from colmap.

    Args:
        filename: file name for .ply
        recon_dir: Directory to grab colmap points
        output_dir: Directory to output .ply
    """
    if (recon_dir / "points3D.bin").exists():
        colmap_points = read_points3D_binary(recon_dir / "points3D.bin")
    elif (recon_dir / "points3D.txt").exists():
        colmap_points = read_points3D_text(recon_dir / "points3D.txt")
    else:
        raise ValueError(f"Could not find points3D.txt or points3D.bin in {recon_dir}")

    # Load point Positions
    points3D = torch.from_numpy(np.array([p.xyz for p in colmap_points.values()], dtype=np.float32))
    if applied_transform is not None:
        assert applied_transform.shape == (3, 4)
        points3D = torch.einsum("ij,bj->bi", applied_transform[:3, :3], points3D) + applied_transform[:3, 3]

    # Load point colours
    points3D_rgb = torch.from_numpy(np.array([p.rgb for p in colmap_points.values()], dtype=np.uint8))

    # write ply
    with open(output_dir / filename, "w") as f:
        # Header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points3D)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uint8 red\n")
        f.write("property uint8 green\n")
        f.write("property uint8 blue\n")
        f.write("end_header\n")

        for coord, color in zip(points3D, points3D_rgb):
            x, y, z = coord
            r, g, b = color
            f.write(f"{x:8f} {y:8f} {z:8f} {r} {g} {b}\n")


def write_images_text(images, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    if len(images) == 0:
        mean_observations = 0
    else:
        mean_observations = sum((len(img.point3D_ids) for _, img in images.items())) / len(images)
    HEADER = (
        "# Image list with two lines of data per image:\n"
        + "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        + "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        + "# Number of images: {}, mean observations per image: {}\n".format(len(images), mean_observations)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, img in images.items():
            image_header = [img.id, *img.qvec, *img.tvec, img.camera_id, img.name]
            first_line = " ".join(map(str, image_header))
            fid.write(first_line + "\n")

            points_strings = []
            for xy, point3D_id in zip(img.xys, img.point3D_ids):
                points_strings.append(" ".join(map(str, [*xy, point3D_id])))
            fid.write(" ".join(points_strings) + "\n")


def write_images_binary(images, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(images), "Q")
        for _, img in images.items():
            write_next_bytes(fid, img.id, "i")
            write_next_bytes(fid, img.qvec.tolist(), "dddd")
            write_next_bytes(fid, img.tvec.tolist(), "ddd")
            write_next_bytes(fid, img.camera_id, "i")
            for char in img.name:
                write_next_bytes(fid, char.encode("utf-8"), "c")
            write_next_bytes(fid, b"\x00", "c")
            write_next_bytes(fid, len(img.point3D_ids), "Q")
            for xy, p3d_id in zip(img.xys, img.point3D_ids):
                write_next_bytes(fid, [*xy, p3d_id], "ddq")


def ply_to_colmap(sparse_ply: Path, point_filename: Path):
    plydata = PlyData.read(sparse_ply)
    vertex = plydata["vertex"]

    with open(point_filename, "w") as f:
        line1 = "# 3D point list with one line of data per point:"
        line2 = "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)"
        line3 = "# Number of points: {}, mean track length: {}".format(len(vertex), 0.0)
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        f.write(line3 + "\n")
        counter = 0

        points3D = {}
        for data in vertex:
            data = list(data)
            newdata = [counter] + data + [0.0] + [0, 0.0] 
            newdata = [str(dt) for dt in newdata]
            f.write(" ".join(newdata) + "\n")
            counter += 1
            error = 0.0

            points3D[counter] = Point3D(
                id=counter, xyz=data[:3], rgb=data[3:], error=error, image_ids=str(counter), point2D_idxs=np.zeros([1])
            )
        file_name = Path("/workspace/write/points3D.txt")
        write_points3D_text(points3D, path=file_name)


def write_points3D_text(points3D, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DText(const std::string& path)
        void Reconstruction::WritePoints3DText(const std::string& path)
    """
    if len(points3D) == 0:
        mean_track_length = 0
    else:
        mean_track_length = sum((len(pt.image_ids) for _, pt in points3D.items())) / len(points3D)
    HEADER = (
        "# 3D point list with one line of data per point:\n"
        + "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        + "# Number of points: {}, mean track length: {}\n".format(len(points3D), mean_track_length)
    )

    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, pt in points3D.items():
            point_header = [pt.id, *pt.xyz, *pt.rgb, pt.error]
            fid.write(" ".join(map(str, point_header)) + " ")
            track_strings = []
            for image_id, point2D in zip(pt.image_ids, pt.point2D_idxs):
                track_strings.append(" ".join(map(str, [image_id, point2D])))
            fid.write(" ".join(track_strings) + "\n")


def write_points3D_binary(points3D, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadPoints3DBinary(const std::string& path)
        void Reconstruction::WritePoints3DBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(points3D), "Q")
        for _, pt in points3D.items():
            write_next_bytes(fid, pt.id, "Q")
            write_next_bytes(fid, pt.xyz.tolist(), "ddd")
            write_next_bytes(fid, pt.rgb.tolist(), "BBB")
            write_next_bytes(fid, pt.error, "d")
            track_length = pt.image_ids.shape[0]
            write_next_bytes(fid, track_length, "Q")
            for image_id, point2D_id in zip(pt.image_ids, pt.point2D_idxs):
                write_next_bytes(fid, [image_id, point2D_id], "ii")


def write_next_bytes(fid, data, format_char_sequence, endian_character="<"):
    """pack and write to a binary file.
    :param fid:
    :param data: data to send, if multiple elements are sent at the same time,
    they should be encapsuled either in a list or a tuple
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    should be the same length as the data list or tuple
    :param endian_character: Any of {@, =, <, >, !}
    """
    if isinstance(data, (list, tuple)):
        bytes = struct.pack(endian_character + format_char_sequence, *data)
    else:
        bytes = struct.pack(endian_character + format_char_sequence, data)
    fid.write(bytes)


#  cameras[camera_id] = Camera(
#                 id=camera_id, model=model_name, width=width, height=height, params=np.array(params)
#             )
#         assert len(cameras) == num_cameras
#     return cameras


def write_cameras_text(cameras, path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    HEADER = (
        "# Camera list with one line of data per camera:\n"
        + "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
        + "# Number of cameras: {}\n".format(len(cameras))
    )
    with open(path, "w") as fid:
        fid.write(HEADER)
        for _, cam in cameras.items():
            to_write = [cam.id, cam.model, cam.width, cam.height, *cam.params]
            line = " ".join([str(elem) for elem in to_write])
            fid.write(line + "\n")


def write_cameras_binary(cameras, path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    with open(path_to_model_file, "wb") as fid:
        write_next_bytes(fid, len(cameras), "Q")
        for _, cam in cameras.items():
            model_id = CAMERA_MODEL_NAMES[cam.model].model_id
            camera_properties = [cam.id, model_id, cam.width, cam.height]
            write_next_bytes(fid, camera_properties, "iiQQ")
            for p in cam.params:
                write_next_bytes(fid, float(p), "d")
    return cameras


if __name__ == "__main__":
    absolute_colmap_model_path = Path("/workspace/trial_new/distorted/chunk_0/colmap")
    image_id_to_depth_path = None
    camera_mask_path = None
    image_rename_map = None
    image_reverse_rename_map = None

    # binary_path = Path("/workspace/trial_new/distorted/chunk_0/colmap/sparse/0/images.bin")
    binary_path = Path("/workspace/trial_new/colmap/sparse/0/images.bin")
    qvecs, tvecs, point_ids = read_extrinsics_binary(binary_path)

    # txt_path = Path("/workspace/txt/images.txt")
    # qvecs_txt, tvecs_txt, point_ids_txt = read_extrinsics_text(txt_path)

    # json_file = Path("/workspace/trial_new/distorted/chunk_0/transforms.json")
    json_file = Path("/workspace/trial_new/transforms.json")
    camera_filename = Path("/workspace/camera.txt")
    image_filename = Path("/workspace/images.txt")
    point_file_name = Path("/workspace/point3D.txt")
    nqvecs, ntvecs, original_paths = json_to_colmap(json_file=json_file, image_filename=image_filename, camera_filename=camera_filename)

    print("Original qvecs:\n", qvecs)
    print("New qvecs:\n", nqvecs)
    # print("Original tvecs:\n", tvecs)
    # print("New tvecs:\n", ntvecs)

    print("Point ids: ", point_ids)
    print("Length of point ids: ", len(point_ids))
    print("Length of qvecs ids: ", len(qvecs))
    print("Length of nqvecs ids: ", len(nqvecs))

    qvec_errors = 0
    counter = 0
    for qvec, nqvec, original_path in zip(qvecs, nqvecs, original_paths):
        error = np.linalg.norm(np.abs(qvec) - np.abs(nqvec)) 
        qvec_errors += error
        if error > 1e-10:
            counter += 1
            print("qvec = {}; nqvec = {}, original path = {}".format(qvec, nqvec, original_path))
    print("Qvec errors: ", qvec_errors)
    print("Dismatched vecs: ", counter)

    tvec_errors = 0
    for tvec, ntvec in zip(tvecs, ntvecs):
        error = np.linalg.norm(tvec - ntvec)
        tvec_errors += error
    print("Tvec errors: ", tvec_errors)

    # sparse_ply = Path("/workspace/trial_new/chunk_0/sparse_pc.ply")
    sparse_ply = Path("/workspace/trial_new/sparse_pc.ply")
    point_filename = Path("/workspace/points3D.txt")
    ply_to_colmap(sparse_ply=sparse_ply, point_filename=point_filename)
    