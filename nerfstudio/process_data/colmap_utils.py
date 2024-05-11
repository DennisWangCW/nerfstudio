# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tools supporting the execution of COLMAP and preparation of COLMAP-based datasets for nerfstudio training.
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union
import appdirs
import cv2
import numpy as np
import requests
import torch
from packaging.version import Version
import shutil
from rich.progress import track
from scipy.spatial.transform import Rotation as R
import collections
from plyfile import PlyData

# TODO(1480) use pycolmap instead of colmap_parsing_utils
# import pycolmap
from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3D_binary,
    read_points3D_text,
    write_cameras_binary,
    write_cameras_text,
    write_images_binary,
    write_images_text,
    write_points3D_binary,
    write_points3D_text,
)
from nerfstudio.process_data.process_data_utils import CameraModel
from nerfstudio.utils import colormaps
from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command
import subprocess

from nerfstudio.data.utils.colmap_parsing_utils import Camera, Image, Point3D
# Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
# BaseImage = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
# Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def get_available_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True)
        gpu_names = result.stdout.strip().split('\n')
        return len(gpu_names)
    except FileNotFoundError:
        print("nvidia-smi is not installed or Nvidia driver is not properly configured.")
        return 0
    
def get_colmap_version(colmap_cmd: str, default_version: str = "3.8") -> Version:
    """Returns the version of COLMAP.
    This code assumes that colmap returns a version string of the form
    "COLMAP 3.8 ..." which may not be true for all versions of COLMAP.

    Args:
        default_version: Default version to return if COLMAP version can't be determined.
    Returns:
        The version of COLMAP.
    """
    output = run_command(f"{colmap_cmd} -h", verbose=False)
    assert output is not None
    for line in output.split("\n"):
        if line.startswith("COLMAP"):
            version = line.split(" ")[1]
            version = Version(version)
            return version
    CONSOLE.print(f"[bold red]Could not find COLMAP version. Using default {default_version}")
    return Version(default_version)


def get_vocab_tree() -> Path:
    """Return path to vocab tree. Downloads vocab tree if it doesn't exist.

    Returns:
        The path to the vocab tree.
    """
    vocab_tree_filename = Path(appdirs.user_data_dir("nerfstudio")) / "vocab_tree.fbow"

    if not vocab_tree_filename.exists():
        r = requests.get("https://demuc.de/colmap/vocab_tree_flickr100K_words32K.bin", stream=True)
        vocab_tree_filename.parent.mkdir(parents=True, exist_ok=True)
        with open(vocab_tree_filename, "wb") as f:
            total_length = r.headers.get("content-length")
            assert total_length is not None
            for chunk in track(
                r.iter_content(chunk_size=1024),
                total=int(total_length) / 1024 + 1,
                description="Downloading vocab tree...",
            ):
                if chunk:
                    f.write(chunk)
                    f.flush()
    return vocab_tree_filename

def run_colmap_image_undistortion(
    image_dir: Path,
    colmap_cmd: str = "colmap",
    verbose: bool = False,
) -> None:
    img_undist_cmd = [f"{colmap_cmd} image_undistorter",
        f"--image_path {image_dir / 'input'}",
        f"--input_path {image_dir / 'distorted/sparse/0'}",
        f"--output_path {image_dir}",
        f"--output_type COLMAP"]
    img_undist_cmd = " ".join(img_undist_cmd)
    with status(msg="[bold yellow]Running COLMAP image undistortion...", spinner="moon", verbose=verbose):
        run_command(img_undist_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done undistorting COLMAP images.")
    
    sparse_path = image_dir.joinpath("sparse")
    sparse_path.joinpath("0").mkdir(exist_ok=True, parents=True)
    # Copy each file from the source directory to the destination directory
    for file in sparse_path.iterdir():
        if file.name == '0':
            continue
        source_file = sparse_path.joinpath(file.name)
        destination_file = sparse_path.joinpath("0").joinpath(file.name)
        source_file.replace(destination_file)
    colmap_dir = image_dir.joinpath("colmap").joinpath("sparse")
    shutil.copytree(sparse_path.absolute(), colmap_dir.absolute())
    shutil.rmtree(sparse_path.absolute())
    shutil.rmtree(image_dir.joinpath("input"))
    shutil.rmtree(image_dir.joinpath("distorted"))


def run_colmap(
    image_dir: Path,
    colmap_dir: Path,
    camera_model: CameraModel,
    camera_mask_path: Optional[Path] = None,
    gpu: bool = True,
    rank: int = 0,
    verbose: bool = False,
    matching_method: Literal["vocab_tree", "exhaustive", "sequential"] = "vocab_tree",
    refine_intrinsics: bool = True,
    colmap_cmd: str = "colmap",
) -> None:
    """Runs COLMAP on the images.

    Args:
        image_dir: Path to the directory containing the images.
        colmap_dir: Path to the output directory.
        camera_model: Camera model to use.
        camera_mask_path: Path to the camera mask.
        gpu: If True, use GPU.
        verbose: If True, logs the output of the command.
        matching_method: Matching method to use.
        refine_intrinsics: If True, refine intrinsics.
        colmap_cmd: Path to the COLMAP executable.
    """

    colmap_version = get_colmap_version(colmap_cmd)

    colmap_database_path = colmap_dir / "database.db"
    colmap_database_path.unlink(missing_ok=True)

    # Feature extraction
    feature_extractor_cmd = [
        f"{colmap_cmd} feature_extractor",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        "--ImageReader.single_camera 1",
        f"--ImageReader.camera_model {camera_model.value}",
        f"--SiftExtraction.use_gpu {int(gpu)}",
    ]
    if int(gpu):
        # feature_extractor_cmd.append(f"--SiftExtraction.gpu_index={','.join(map(str, range(get_available_gpus())))}")
        num_gpus = get_available_gpus()
        if num_gpus > 0:
            feature_extractor_cmd.append(f"--SiftExtraction.gpu_index={rank % num_gpus}")
    if camera_mask_path is not None:
        feature_extractor_cmd.append(f"--ImageReader.camera_mask_path {camera_mask_path}")
    feature_extractor_cmd = " ".join(feature_extractor_cmd)
    with status(msg="[bold yellow]Running COLMAP feature extractor...", spinner="moon", verbose=verbose):
        run_command(feature_extractor_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done extracting COLMAP features.")

    # Feature matching
    feature_matcher_cmd = [
        f"{colmap_cmd} {matching_method}_matcher",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--SiftMatching.use_gpu {int(gpu)}",
    ]
    if int(gpu):
        # feature_matcher_cmd.append(f"--SiftMatching.gpu_index={','.join(map(str, range(get_available_gpus())))}")
        num_gpus = get_available_gpus()
        if num_gpus > 0:
            feature_matcher_cmd.append(f"--SiftMatching.gpu_index={rank % num_gpus}")
    if matching_method == "vocab_tree":
        vocab_tree_filename = get_vocab_tree()
        feature_matcher_cmd.append(f'--VocabTreeMatching.vocab_tree_path "{vocab_tree_filename}"')
    feature_matcher_cmd = " ".join(feature_matcher_cmd)
    with status(msg="[bold yellow]Running COLMAP feature matcher...", spinner="runner", verbose=verbose):
        run_command(feature_matcher_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done matching COLMAP features.")

    # Bundle adjustment
    sparse_dir = colmap_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    mapper_cmd = [
        f"{colmap_cmd} mapper",
        f"--database_path {colmap_dir / 'database.db'}",
        f"--image_path {image_dir}",
        f"--output_path {sparse_dir}",
    ]
    if colmap_version >= Version("3.7"):
        mapper_cmd.append("--Mapper.ba_global_function_tolerance=1e-6")

    mapper_cmd = " ".join(mapper_cmd)

    with status(
        msg="[bold yellow]Running COLMAP bundle adjustment... (This may take a while)",
        spinner="circle",
        verbose=verbose,
    ):
        run_command(mapper_cmd, verbose=verbose)
    CONSOLE.log("[bold green]:tada: Done COLMAP bundle adjustment.")

    if refine_intrinsics:
        with status(msg="[bold yellow]Refine intrinsics...", spinner="dqpb", verbose=verbose):
            bundle_adjuster_cmd = [
                f"{colmap_cmd} bundle_adjuster",
                f"--input_path {sparse_dir}/0",
                f"--output_path {sparse_dir}/0",
                "--BundleAdjustment.refine_principal_point 1",
            ]
            run_command(" ".join(bundle_adjuster_cmd), verbose=verbose)
        CONSOLE.log("[bold green]:tada: Done refining intrinsics.")


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        # out["k1"] = 0.0
        # out["k2"] = 0.0
        # out["p1"] = 0.0
        # out["p2"] = 0.0
        camera_model = CameraModel.PINHOLE
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    camera_mask_path: Optional[Path] = None,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    image_rename_map: Optional[Dict[str, str]] = None,
    image_reverse_rename_map: Optional[Dict[str, str]] = None,
    ply_filename="sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        camera_model: Camera model used.
        camera_mask_path: Path to the camera mask.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        image_rename_map: Use these image names instead of the names embedded in the COLMAP db
        keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
                    Colmap optimized world often have y direction of the first camera pointing towards down direction,
                    while nerfstudio world set z direction to be up direction for viewer.
    Returns:
        The number of registered images.
    """
    print("checking paths: \n")
    print("camera_mask_path: ", camera_mask_path)
    print("image_id_to_depth_path: ", image_id_to_depth_path)
    print("keep_original_world_coordinate: ", keep_original_world_coordinate)
    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    frames = []
    for im_id, im_data in im_id_to_image.items():
        # NB: COLMAP uses Eigen / scalar-first quaternions
        # * https://colmap.github.io/format.html
        # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
        # the `rotation_matrix()` handles that format for us.

        # TODO(1480) BEGIN use pycolmap API
        # rotation = im_data.rotation_matrix()
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        original_name = name
        if image_rename_map is not None:
            name = image_rename_map[name]
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "original_path":  image_reverse_rename_map[original_name] if image_reverse_rename_map is not None else "",
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if camera_mask_path is not None:
            frame["mask_path"] = camera_mask_path.relative_to(camera_mask_path.parent.parent).as_posix()
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))
        frames.append(frame)

    if set(cam_id_to_camera.keys()) != {1}:
        raise RuntimeError("Only single camera shared for all images is supported.")
    out = parse_colmap_camera_params(cam_id_to_camera[1])
    out["frames"] = frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    # create ply from colmap
    assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
    create_ply_from_colmap(
        ply_filename,
        recon_dir,
        output_dir,
        torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
    )
    out["ply_file_path"] = ply_filename

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)


def json_to_colmap(json_file: Path, output_dir: Path, keep_original_world_coordinate: Optional[bool] = False) -> None:
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    #### preparing data for cameras.txt
    camera_model = data["camera_model"]
    assert camera_model == "PINHOLE" or camera_model == "OPENCV", "currently only support pinhole or opencv"
    params = []
    if camera_model == "PINHOLE" or camera_model == "OPENCV":
        w = data["w"]
        h = data["h"]
        fl_x = data["fl_x"]
        fl_y = data["fl_y"]
        cx = data["cx"]
        cy = data["cy"]
        params = params + [fl_x, fl_y, cx, cy]
    if camera_model == "OPENCV":
        k1 = data["k1"]
        k2 = data["k2"]
        p1 = data["p1"]
        p2 = data["p2"]
        params =params + [k1, k2, p1, p2]
    
    camera_id = 1

    cameras = {}
    cameras[camera_id] = Camera(
                id=camera_id, model=camera_model, width=w, height=h, params=np.array(params)
            )    
    camera_filename_txt = output_dir.joinpath("cameras.txt")
    camera_filename_bin = output_dir.joinpath("cameras.bin")
    write_cameras_text(cameras, path=camera_filename_txt)
    # write_cameras_binary(cameras, path_to_model_file=camera_filename_bin)
    CONSOLE.log("[bold green]:tada: Done writing the merged cameras intrinsics file!")

    #### preparing data for images.txt
    image_ids = []
    image_names = []
    frames = data["frames"]
    tvecs, qvecs = [], []
    for frame in frames:
        image_id = frame["colmap_im_id"]
        file_path = frame["file_path"]
        # original_path = frame["original_path"]
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
        # original_paths.append(original_path)

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
                point3D_ids=-np.ones([1]),
        )

    image_filename_txt = output_dir.joinpath("images.txt")
    image_filename_bin = output_dir.joinpath("images.bin")
    write_images_text(images, path = image_filename_txt)
    # write_images_binary(images, path_to_model_file=image_filename_bin)
    CONSOLE.log("[bold green]:tada: Done writing the merged images extrinsics file!")


def ply_to_colmap(sparse_ply: Path, output_dir: Path):
    plydata = PlyData.read(sparse_ply)
    vertex = plydata["vertex"]

    points3D = {}
    counter = 0
    for data in vertex:
        data = list(data)
        newdata = [counter] + data + [0.0] + [0, 0.0] 
        newdata = [str(dt) for dt in newdata]
        counter += 1
        error = 0.0
        points3D[counter] = Point3D(
            id=counter, xyz=data[:3], rgb=data[3:], error=error, image_ids=str(counter), point2D_idxs=np.zeros([1])
        )

    point3d_filename_txt = output_dir.joinpath("points3D.txt")
    point3d_filename_bin = output_dir.joinpath("points3D.bin")
    write_points3D_text(points3D, path=point3d_filename_txt)
    # write_points3D_binary(points3D, path_to_model_file=point3d_filename_bin)
    CONSOLE.log("[bold green]:tada: Done writing the merged point3D cloud file!")


def create_sfm_depth(
    recon_dir: Path,
    output_dir: Path,
    verbose: bool = True,
    depth_scale_to_integer_factor: float = 1000.0,
    min_depth: float = 0.001,
    max_depth: float = 10000,
    max_repoj_err: float = 2.5,
    min_n_visible: int = 2,
    include_depth_debug: bool = False,
    input_images_dir: Optional[Path] = None,
) -> Dict[int, Path]:
    """Converts COLMAP's points3d.bin to sparse depth map images encoded as
    16-bit "millimeter depth" PNGs.

    Notes:
     * This facility does NOT use COLMAP dense reconstruction; it creates depth
        maps from sparse SfM points here.
     * COLMAP does *not* reconstruct metric depth unless you give it calibrated
        (metric) intrinsics as input. Therefore, "depth" in this function has
        potentially ambiguous units.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        verbose: If True, logs progress of depth image creation.
        depth_scale_to_integer_factor: Use this parameter to tune the conversion of
          raw depth measurements to integer depth values.  This value should
          be equal to 1. / `depth_unit_scale_factor`, where
          `depth_unit_scale_factor` is the value you provide at training time.
          E.g. for millimeter depth, leave `depth_unit_scale_factor` at 1e-3
          and depth_scale_to_integer_factor at 1000.
        min_depth: Discard points closer than this to the camera.
        max_depth: Discard points farther than this from the camera.
        max_repoj_err: Discard points with reprojection error greater than this
          amount (in pixels).
        min_n_visible: Discard 3D points that have been triangulated with fewer
          than this many frames.
        include_depth_debug: Also include debug images showing depth overlaid
          upon RGB.
    Returns:
        Depth file paths indexed by COLMAP image id
    """

    # TODO(1480) use pycolmap
    # recon = pycolmap.Reconstruction(recon_dir)
    # ptid_to_info = recon.points3D
    # cam_id_to_camera = recon.cameras
    # im_id_to_image = recon.images
    ptid_to_info = read_points3D_binary(recon_dir / "points3D.bin")
    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")

    # Only support first camera
    CAMERA_ID = 1
    W = cam_id_to_camera[CAMERA_ID].width
    H = cam_id_to_camera[CAMERA_ID].height

    if verbose:
        iter_images = track(
            im_id_to_image.items(), total=len(im_id_to_image.items()), description="Creating depth maps ..."
        )
    else:
        iter_images = iter(im_id_to_image.items())

    image_id_to_depth_path = {}
    for im_id, im_data in iter_images:
        # TODO(1480) BEGIN delete when abandoning colmap_parsing_utils
        pids = [pid for pid in im_data.point3D_ids if pid != -1]
        xyz_world = np.array([ptid_to_info[pid].xyz for pid in pids])
        rotation = qvec2rotmat(im_data.qvec)
        z = (rotation @ xyz_world.T)[-1] + im_data.tvec[-1]
        errors = np.array([ptid_to_info[pid].error for pid in pids])
        n_visible = np.array([len(ptid_to_info[pid].image_ids) for pid in pids])
        uv = np.array([im_data.xys[i] for i in range(len(im_data.xys)) if im_data.point3D_ids[i] != -1])
        # TODO(1480) END delete when abandoning colmap_parsing_utils

        # TODO(1480) BEGIN use pycolmap API

        # # Get only keypoints that have corresponding triangulated 3D points
        # p2ds = im_data.get_valid_points2D()

        # xyz_world = np.array([ptid_to_info[p2d.point3D_id].xyz for p2d in p2ds])

        # # COLMAP OpenCV convention: z is always positive
        # z = (im_data.rotation_matrix() @ xyz_world.T)[-1] + im_data.tvec[-1]

        # # Mean reprojection error in image space
        # errors = np.array([ptid_to_info[p2d.point3D_id].error for p2d in p2ds])

        # # Number of frames in which each frame is visible
        # n_visible = np.array([ptid_to_info[p2d.point3D_id].track.length() for p2d in p2ds])

        # Note: these are *unrectified* pixel coordinates that should match the original input
        # no matter the camera model
        # uv = np.array([p2d.xy for p2d in p2ds])

        # TODO(1480) END use pycolmap API

        idx = np.where(
            (z >= min_depth)
            & (z <= max_depth)
            & (errors <= max_repoj_err)
            & (n_visible >= min_n_visible)
            & (uv[:, 0] >= 0)
            & (uv[:, 0] < W)
            & (uv[:, 1] >= 0)
            & (uv[:, 1] < H)
        )
        z = z[idx]
        uv = uv[idx]

        uu, vv = uv[:, 0].astype(int), uv[:, 1].astype(int)
        depth = np.zeros((H, W), dtype=np.float32)
        depth[vv, uu] = z

        # E.g. if `depth` is metric and in units of meters, and `depth_scale_to_integer_factor`
        # is 1000, then `depth_img` will be integer millimeters.
        depth_img = (depth_scale_to_integer_factor * depth).astype(np.uint16)

        out_name = str(im_data.name)
        depth_path = output_dir / out_name
        if depth_path.suffix == ".jpg":
            depth_path = depth_path.with_suffix(".png")
        cv2.imwrite(str(depth_path), depth_img)  # type: ignore

        image_id_to_depth_path[im_id] = depth_path

        if include_depth_debug:
            assert input_images_dir is not None, "Need explicit input_images_dir for debug images"
            assert input_images_dir.exists(), input_images_dir

            depth_flat = depth.flatten()[:, None]
            overlay = 255.0 * colormaps.apply_depth_colormap(torch.from_numpy(depth_flat)).numpy()
            overlay = overlay.reshape([H, W, 3])
            input_image_path = input_images_dir / im_data.name
            input_image = cv2.imread(str(input_image_path))  # type: ignore
            debug = 0.3 * input_image + 0.7 + overlay

            out_name = out_name + ".debug.jpg"
            output_path = output_dir / "debug_depth" / out_name
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), debug.astype(np.uint8))  # type: ignore

    return image_id_to_depth_path


def get_matching_summary(num_initial_frames: int, num_matched_frames: int) -> str:
    """Returns a summary of the matching results.

    Args:
        num_initial_frames: The number of initial frames.
        num_matched_frames: The number of matched frames.

    Returns:
        A summary of the matching results.
    """
    match_ratio = num_matched_frames / num_initial_frames
    if match_ratio == 1:
        return "[bold green]COLMAP found poses for all images, CONGRATS!"
    if match_ratio < 0.4:
        result = f"[bold red]COLMAP only found poses for {num_matched_frames / num_initial_frames * 100:.2f}%"
        result += " of the images. This is low.\nThis can be caused by a variety of reasons,"
        result += " such poor scene coverage, blurry images, or large exposure changes."
        return result
    if match_ratio < 0.8:
        result = f"[bold yellow]COLMAP only found poses for {num_matched_frames / num_initial_frames * 100:.2f}%"
        result += " of the images.\nThis isn't great, but may be ok."
        result += "\nMissing poses can be caused by a variety of reasons, such poor scene coverage, blurry images,"
        result += " or large exposure changes."
        return result
    return f"[bold green]COLMAP found poses for {num_matched_frames / num_initial_frames * 100:.2f}% of the images."


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
