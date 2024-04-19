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

"""Helper utils for processing data into the nerfstudio format."""

import math
import re
import shutil
import sys
import os
from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, OrderedDict, Tuple, Union
import random 
import subprocess
import multiprocessing
import json

import cv2
import imageio
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

try:
    import rawpy
except ImportError:
    import newrawpy as rawpy  # type: ignore

import numpy as np

from nerfstudio.utils.rich_utils import CONSOLE, status
from nerfstudio.utils.scripts import run_command
from scipy.stats import norm


POLYCAM_UPSCALING_TIMES = 2

"""Lowercase suffixes to treat as raw image."""
ALLOWED_RAW_EXTS = [".cr2"]
"""Suffix to use for converted images from raw."""
RAW_CONVERTED_SUFFIX = ".jpg"


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "equirectangular": CameraModel.EQUIRECTANGULAR,
    "pinhole": CameraModel.PINHOLE,
    "simple_pinhole": CameraModel.SIMPLE_PINHOLE,
}


def list_images(data: Path, recursive: bool = False) -> List[Path]:
    """Lists all supported images in a directory

    Args:
        data: Path to the directory of images.
        recursive: Whether to search check nested folders in `data`.
    Returns:
        Paths to images contained in the directory
    """
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"] + ALLOWED_RAW_EXTS
    glob_str = "**/[!.]*" if recursive else "[!.]*"
    image_paths = sorted([p for p in data.glob(glob_str) if p.suffix.lower() in allowed_exts])
    return image_paths


def get_image_filenames(directory: Path, max_num_images: int = -1) -> Tuple[List[Path], int]:
    """Returns a list of image filenames in a directory.

    Args:
        dir: Path to the directory.
        max_num_images: The maximum number of images to return. -1 means no limit.
    Returns:
        A tuple of A list of image filenames, number of original image paths.
    """
    image_paths = list_images(directory)
    num_orig_images = len(image_paths)

    if max_num_images != -1 and num_orig_images > max_num_images:
        idx = np.round(np.linspace(0, num_orig_images - 1, max_num_images)).astype(int)
    else:
        idx = np.arange(num_orig_images)

    image_filenames = list(np.array(image_paths)[idx])

    return image_filenames, num_orig_images


def get_num_frames_in_video(video: Path) -> int:
    """Returns the number of frames in a video.

    Args:
        video: Path to a video.

    Returns:
        The number of frames in a video.
    """
    cmd = f'ffprobe -v error -select_streams v:0 -count_packets \
            -show_entries stream=nb_read_packets -of csv=p=0 "{video}"'
    output = run_command(cmd)
    assert output is not None
    number_match = re.search(r"\d+", output)
    assert number_match is not None
    return int(number_match[0])


def convert_video_to_images(
    video_path: Path,
    image_dir: Path,
    num_frames_target: int,
    num_downscales: int,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    verbose: bool = False,
    image_prefix: str = "frame_",
    keep_image_dir: bool = False,
) -> Tuple[List[str], int]:
    """Converts a video into a sequence of images.

    Args:
        video_path: Path to the video.
        output_dir: Path to the output directory.
        num_frames_target: Number of frames to extract.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        crop_factor: Portion of the image to crop. Should be in [0,1] (top, bottom, left, right)
        verbose: If True, logs the output of the command.
        image_prefix: Prefix to use for the image filenames.
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        A tuple containing summary of the conversion and the number of extracted frames.
    """

    # If keep_image_dir is False, then remove the output image directory and its downscaled versions
    if not keep_image_dir:
        for i in range(num_downscales + 1):
            dir_to_remove = image_dir if i == 0 else f"{image_dir}_{2**i}"
            shutil.rmtree(dir_to_remove, ignore_errors=True)
    image_dir.mkdir(exist_ok=True, parents=True)

    for i in crop_factor:
        if i < 0 or i > 1:
            CONSOLE.print("[bold red]Error: Invalid crop factor. All crops must be in [0,1].")
            sys.exit(1)

    if video_path.is_dir():
        CONSOLE.print(f"[bold red]Error: Video path is a directory, not a path: {video_path}")
        sys.exit(1)
    if video_path.exists() is False:
        CONSOLE.print(f"[bold red]Error: Video does not exist: {video_path}")
        sys.exit(1)

    with status(msg="Converting video to images...", spinner="bouncingBall", verbose=verbose):
        num_frames = get_num_frames_in_video(video_path)
        if num_frames == 0:
            CONSOLE.print(f"[bold red]Error: Video has no frames: {video_path}")
            sys.exit(1)
        CONSOLE.print("Number of frames in video:", num_frames)

        ffmpeg_cmd = f'ffmpeg -i "{video_path}"'

        crop_cmd = ""
        if crop_factor != (0.0, 0.0, 0.0, 0.0):
            height = 1 - crop_factor[0] - crop_factor[1]
            width = 1 - crop_factor[2] - crop_factor[3]
            start_x = crop_factor[2]
            start_y = crop_factor[0]
            crop_cmd = f"crop=w=iw*{width}:h=ih*{height}:x=iw*{start_x}:y=ih*{start_y},"

        if num_frames_target == 0:
            spacing = 1
        else:
            spacing = num_frames // num_frames_target

        downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}[out{i}]" for i in range(num_downscales + 1)]
        downscale_dirs = [Path(str(image_dir) + (f"_{2**i}" if i > 0 else "")) for i in range(num_downscales + 1)]
        downscale_paths = [downscale_dirs[i] / f"{image_prefix}%05d.png" for i in range(num_downscales + 1)]

        for dir in downscale_dirs:
            dir.mkdir(parents=True, exist_ok=True)

        downscale_chain = (
            f"split={num_downscales + 1}"
            + "".join([f"[t{i}]" for i in range(num_downscales + 1)])
            + ";"
            + ";".join(downscale_chains)
        )

        ffmpeg_cmd += " -vsync vfr"

        if spacing > 1:
            CONSOLE.print("Number of frames to extract:", math.ceil(num_frames / spacing))
            select_cmd = f"thumbnail={spacing},setpts=N/TB,"
        else:
            CONSOLE.print("[bold red]Can't satisfy requested number of frames. Extracting all frames.")
            ffmpeg_cmd += " -pix_fmt bgr8"
            select_cmd = ""

        downscale_cmd = f' -filter_complex "{select_cmd}{crop_cmd}{downscale_chain}"' + "".join(
            [f' -map "[out{i}]" "{downscale_paths[i]}"' for i in range(num_downscales + 1)]
        )

        ffmpeg_cmd += downscale_cmd

        run_command(ffmpeg_cmd, verbose=verbose)

        num_final_frames = len(list(image_dir.glob("*.png")))
        summary_log = []
        summary_log.append(f"Starting with {num_frames} video frames")
        summary_log.append(f"We extracted {num_final_frames} images with prefix '{image_prefix}'")
        CONSOLE.log("[bold green]:tada: Done converting video to images.")

        return summary_log, num_final_frames
    

def copy_images_list(
    image_paths: List[Path],
    image_dir: Path,
    num_downscales: int,
    image_prefix: str = "frame_",
    crop_border_pixels: Optional[int] = None,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    verbose: bool = False,
    keep_image_dir: bool = False,
    upscale_factor: Optional[int] = None,
    nearest_neighbor: bool = False,
    same_dimensions: bool = True,
    rename: bool = True,
) -> List[Path]:
    """Copy all images in a list of Paths. Useful for filtering from a directory.
    Args:
        image_paths: List of Paths of images to copy to a new directory.
        image_dir: Path to the output directory.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        image_prefix: Prefix for the image filenames.
        crop_border_pixels: If not None, crops each edge by the specified number of pixels.
        crop_factor: Portion of the image to crop. Should be in [0,1] (top, bottom, left, right)
        verbose: If True, print extra logging.
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        A list of the copied image Paths.
    """

    # Remove original directory and its downscaled versions
    # only if we provide a proper image folder path and keep_image_dir is False
    if image_dir.is_dir() and len(image_paths) and not keep_image_dir:
        # check that output directory is not the same as input directory
        if image_dir != image_paths[0].parent:
            for i in range(num_downscales + 1):
                dir_to_remove = image_dir if i == 0 else f"{image_dir}_{2**i}"
                shutil.rmtree(dir_to_remove, ignore_errors=True)
    image_dir.mkdir(exist_ok=True, parents=True)

    copied_image_paths = []

    # Images should be 1-indexed for the rest of the pipeline.
    for idx, image_path in enumerate(image_paths):
        if verbose:
            CONSOLE.log(f"Copying image {idx + 1} of {len(image_paths)}...")
        if not rename:
            copied_image_path =  image_dir
        else:
            copied_image_path = image_dir / f"{image_prefix}{idx + 1:05d}{image_path.suffix}"
        try:
            # if CR2 raw, we want to read raw and write RAW_CONVERTED_SUFFIX, and change the file suffix for downstream processing
            if image_path.suffix.lower() in ALLOWED_RAW_EXTS:
                copied_image_path = image_dir / f"{image_prefix}{idx + 1:05d}{RAW_CONVERTED_SUFFIX}"
                with rawpy.imread(str(image_path)) as raw:
                    rgb = raw.postprocess()
                imageio.imsave(copied_image_path, rgb)
                image_paths[idx] = copied_image_path
            elif same_dimensions:
                # Fast path; just copy the file
                shutil.copy(image_path, copied_image_path)
            else:
                # Slow path; let ffmpeg perform autorotation (and clear metadata)
                ffmpeg_cmd = f"ffmpeg -y -i {image_path} -metadata:s:v:0 rotate=0 {copied_image_path}"
                if verbose:
                    CONSOLE.log(f"... {ffmpeg_cmd}")
                run_command(ffmpeg_cmd, verbose=verbose)
        except shutil.SameFileError:
            pass
        copied_image_paths.append(copied_image_path)

    nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
    downscale_chains = [f"[t{i}]scale=iw/{2**i}:ih/{2**i}{nn_flag}[out{i}]" for i in range(num_downscales + 1)]
    downscale_dirs = [Path(str(image_dir) + (f"_{2**i}" if i > 0 else "")) for i in range(num_downscales + 1)]

    for dir in downscale_dirs:
        dir.mkdir(parents=True, exist_ok=True)

    downscale_chain = (
        f"split={num_downscales + 1}"
        + "".join([f"[t{i}]" for i in range(num_downscales + 1)])
        + ";"
        + ";".join(downscale_chains)
    )

    num_frames = len(image_paths)
    # ffmpeg batch commands assume all images are the same dimensions.
    # When this is not the case (e.g. mixed portrait and landscape images), we need to do individually.
    # (Unfortunately, that is much slower.)
    for framenum in range(1, (1 if same_dimensions else num_frames) + 1):
        framename = f"{image_prefix}%05d" if same_dimensions else f"{image_prefix}{framenum:05d}"
        ffmpeg_cmd = f'ffmpeg -y -noautorotate -i "{image_dir / f"{framename}{copied_image_paths[0].suffix}"}" '

        crop_cmd = ""
        if crop_border_pixels is not None:
            crop_cmd = f"crop=iw-{crop_border_pixels*2}:ih-{crop_border_pixels*2}[cropped];[cropped]"
        elif crop_factor != (0.0, 0.0, 0.0, 0.0):
            height = 1 - crop_factor[0] - crop_factor[1]
            width = 1 - crop_factor[2] - crop_factor[3]
            start_x = crop_factor[2]
            start_y = crop_factor[0]
            crop_cmd = f"crop=w=iw*{width}:h=ih*{height}:x=iw*{start_x}:y=ih*{start_y}[cropped];[cropped]"

        select_cmd = "[0:v]"
        if upscale_factor is not None:
            select_cmd = f"[0:v]scale=iw*{upscale_factor}:ih*{upscale_factor}:flags=neighbor[upscaled];[upscaled]"

        downscale_cmd = f' -filter_complex "{select_cmd}{crop_cmd}{downscale_chain}"' + "".join(
            [
                f' -map "[out{i}]" -q:v 2 "{downscale_dirs[i] / f"{framename}{copied_image_paths[0].suffix}"}"'
                for i in range(num_downscales + 1)
            ]
        )

        ffmpeg_cmd += downscale_cmd
        if verbose:
            CONSOLE.log(f"... {ffmpeg_cmd}")
        run_command(ffmpeg_cmd, verbose=verbose)

    if num_frames == 0:
        CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
    else:
        CONSOLE.log(f"[bold green]:tada: Done copying images with prefix '{image_prefix}'.")

    return copied_image_paths


def copy_folder(
        source_dir: Path,
        target_dir: Path,
):
    shutil.copytree(source_dir.absolute(), target_dir.absolute()) 


def empty_folder(folder_path: Path):
    shutil.rmtree(folder_path.absolute())


def copy_images_list_new(
    image_paths: List[Path],
    image_dir: Path,
    num_downscales: int,
    keep_image_dir: bool = False,
) -> List[Path]:
    """Copy all images in a list of Paths. Useful for filtering from a directory.
    Args:
        image_paths: List of Paths of images to copy to a new directory.
        image_dir: Path to the output directory.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        A list of the copied image Paths.
    """

    # Remove original directory and its downscaled versions
    # only if we provide a proper image folder path and keep_image_dir is False
    if image_dir.is_dir() and len(image_paths) and not keep_image_dir:
        # check that output directory is not the same as input directory
        if image_dir != image_paths[0].parent:
            for i in range(num_downscales + 1):
                dir_to_remove = image_dir if i == 0 else f"{image_dir}_{2**i}"
                shutil.rmtree(dir_to_remove, ignore_errors=True)
    image_dir.mkdir(exist_ok=True, parents=True)

    copied_image_paths = []

    # Images should be 1-indexed for the rest of the pipeline.
    for idx, image_path in enumerate(image_paths):
        copied_image_path =  image_dir
        try:
            shutil.copy(image_path, copied_image_path)
        except shutil.SameFileError:
            pass
        copied_image_paths.append(copied_image_path)

    # CONSOLE.log("[bold red]:skull: No usable images in the data folder.")

    return copied_image_paths

def copy_and_upscale_polycam_depth_maps_list(
    polycam_depth_image_filenames: List[Path],
    depth_dir: Path,
    num_downscales: int,
    crop_border_pixels: Optional[int] = None,
    verbose: bool = False,
) -> List[Path]:
    """
    Copy depth maps to working location and upscale them to match the RGB images dimensions and finally crop them
    equally as RGB Images.
    Args:
        polycam_depth_image_filenames: List of Paths of images to copy to a new directory.
        depth_dir: Path to the output directory.
        crop_border_pixels: If not None, crops each edge by the specified number of pixels.
        verbose: If True, print extra logging.
    Returns:
        A list of the copied depth maps paths.
    """
    depth_dir.mkdir(parents=True, exist_ok=True)

    # copy and upscale them to new directory
    with status(
        msg="[bold yellow] Upscaling depth maps...",
        spinner="growVertical",
        verbose=verbose,
    ):
        upscale_factor = 2**POLYCAM_UPSCALING_TIMES
        assert upscale_factor > 1
        assert isinstance(upscale_factor, int)

        copied_depth_map_paths = copy_images_list(
            image_paths=polycam_depth_image_filenames,
            image_dir=depth_dir,
            num_downscales=num_downscales,
            crop_border_pixels=crop_border_pixels,
            verbose=verbose,
            upscale_factor=upscale_factor,
            nearest_neighbor=True,
        )

    CONSOLE.log("[bold green]:tada: Done upscaling depth maps.")
    return copied_depth_map_paths


def copy_images(
    data: Path,
    image_dir: Path,
    image_prefix: str = "frame_",
    verbose: bool = False,
    keep_image_dir: bool = False,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    num_downscales: int = 0,
    same_dimensions: bool = True,
) -> OrderedDict[Path, Path]:
    """Copy images from a directory to a new directory.

    Args:
        data: Path to the directory of images.
        image_dir: Path to the output directory.
        image_prefix: Prefix for the image filenames.
        verbose: If True, print extra logging.
        crop_factor: Portion of the image to crop. Should be in [0,1] (top, bottom, left, right)
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        The mapping from the original filenames to the new ones.
    """
    with status(msg="[bold yellow]Copying images...", spinner="bouncingBall", verbose=verbose):
        image_paths = list_images(data)

        if len(image_paths) == 0:
            CONSOLE.log("[bold red]:skull: No usable images in the data folder.")
            sys.exit(1)

        copied_images = copy_images_list(
            image_paths=image_paths,
            image_dir=image_dir,
            crop_factor=crop_factor,
            verbose=verbose,
            image_prefix=image_prefix,
            keep_image_dir=keep_image_dir,
            num_downscales=num_downscales,
            same_dimensions=same_dimensions,
        )
        return OrderedDict((original_path, new_path) for original_path, new_path in zip(image_paths, copied_images))


def downscale_images(
    image_dir: Path,
    num_downscales: int,
    folder_name: str = "images",
    nearest_neighbor: bool = False,
    verbose: bool = False,
) -> str:
    """(Now deprecated; much faster integrated into copy_images.)
    Downscales the images in the directory. Uses FFMPEG.

    Args:
        image_dir: Path to the directory containing the images.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        folder_name: Name of the output folder
        nearest_neighbor: Use nearest neighbor sampling (useful for depth images)
        verbose: If True, logs the output of the command.

    Returns:
        Summary of downscaling.
    """

    if num_downscales == 0:
        return "No downscaling performed."

    with status(
        msg="[bold yellow]Downscaling images...",
        spinner="growVertical",
        verbose=verbose,
    ):
        downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
        for downscale_factor in downscale_factors:
            assert downscale_factor > 1
            assert isinstance(downscale_factor, int)
            downscale_dir = image_dir.parent / f"{folder_name}_{downscale_factor}"
            downscale_dir.mkdir(parents=True, exist_ok=True)
            # Using %05d ffmpeg commands appears to be unreliable (skips images).
            for f in list_images(image_dir):
                filename = f.name
                nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
                ffmpeg_cmd = [
                    f'ffmpeg -y -noautorotate -i "{image_dir / filename}" ',
                    f"-q:v 2 -vf scale=iw/{downscale_factor}:ih/{downscale_factor}{nn_flag} ",
                    f'"{downscale_dir / filename}"',
                ]
                ffmpeg_cmd = " ".join(ffmpeg_cmd)
                run_command(ffmpeg_cmd, verbose=verbose)

    CONSOLE.log("[bold green]:tada: Done downscaling images.")
    downscale_text = [f"[bold blue]{2**(i+1)}x[/bold blue]" for i in range(num_downscales)]
    downscale_text = ", ".join(downscale_text[:-1]) + " and " + downscale_text[-1]
    return f"We downsampled the images by {downscale_text}"


def find_tool_feature_matcher_combination(
    sfm_tool: Literal["any", "colmap", "hloc"],
    feature_type: Literal[
        "any",
        "sift",
        "superpoint",
        "superpoint_aachen",
        "superpoint_max",
        "superpoint_inloc",
        "r2d2",
        "d2net-ss",
        "sosnet",
        "disk",
    ],
    matcher_type: Literal[
        "any",
        "NN",
        "superglue",
        "superglue-fast",
        "NN-superpoint",
        "NN-ratio",
        "NN-mutual",
        "adalam",
        "disk+lightglue",
        "superpoint+lightglue",
    ],
) -> Union[
    Tuple[None, None, None],
    Tuple[
        Literal["colmap", "hloc"],
        Literal[
            "sift",
            "superpoint_aachen",
            "superpoint_max",
            "superpoint_inloc",
            "r2d2",
            "d2net-ss",
            "sosnet",
            "disk",
        ],
        Literal[
            "NN",
            "superglue",
            "superglue-fast",
            "NN-superpoint",
            "NN-ratio",
            "NN-mutual",
            "adalam",
            "disk+lightglue",
            "superpoint+lightglue",
        ],
    ],
]:
    """Find a valid combination of sfm tool, feature type, and matcher type.
    Basically, replace the default parameters 'any' by usable value

    Args:
        sfm_tool: Sfm tool name (any, colmap, hloc)
        feature_type: Type of image features (any, sift, superpoint, ...)
        matcher_type: Type of matching algorithm (any, NN, superglue,...)

    Returns:
        Tuple of sfm tool, feature type, and matcher type.
        Returns (None,None,None) if no valid combination can be found
    """
    if sfm_tool == "any":
        if (feature_type in ("any", "sift")) and (matcher_type in ("any", "NN")):
            sfm_tool = "colmap"
        else:
            sfm_tool = "hloc"

    if sfm_tool == "colmap":
        if (feature_type not in ("any", "sift")) or (matcher_type not in ("any", "NN")):
            return (None, None, None)
        return ("colmap", "sift", "NN")
    if sfm_tool == "hloc":
        if feature_type in ("any", "superpoint"):
            feature_type = "superpoint_aachen"

        if matcher_type == "any":
            matcher_type = "superglue"
        elif matcher_type == "NN":
            matcher_type = "NN-mutual"

        return (sfm_tool, feature_type, matcher_type)
    return (None, None, None)


def generate_circle_mask(height: int, width: int, percent_radius) -> Optional[np.ndarray]:
    """generate a circle mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The mask or None if the radius is too large.
    """
    if percent_radius <= 0.0:
        CONSOLE.log("[bold red]:skull: The radius of the circle mask must be positive.")
        sys.exit(1)
    if percent_radius >= 1.0:
        return None
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(percent_radius * np.sqrt(width**2 + height**2) / 2.0)
    cv2.circle(mask, center, radius, 1, -1)
    return mask


def generate_crop_mask(height: int, width: int, crop_factor: Tuple[float, float, float, float]) -> Optional[np.ndarray]:
    """generate a crop mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        crop_factor: The percent of the image to crop in each direction [top, bottom, left, right].

    Returns:
        The mask or None if no cropping is performed.
    """
    if np.all(np.array(crop_factor) == 0.0):
        return None
    if np.any(np.array(crop_factor) < 0.0) or np.any(np.array(crop_factor) > 1.0):
        CONSOLE.log("[bold red]Invalid crop percentage, must be between 0 and 1.")
        sys.exit(1)
    top, bottom, left, right = crop_factor
    mask = np.zeros((height, width), dtype=np.uint8)
    top = int(top * height)
    bottom = int(bottom * height)
    left = int(left * width)
    right = int(right * width)
    mask[top : height - bottom, left : width - right] = 1.0
    return mask


def generate_mask(
    height: int,
    width: int,
    crop_factor: Tuple[float, float, float, float],
    percent_radius: float,
) -> Optional[np.ndarray]:
    """generate a mask of the given size.

    Args:
        height: The height of the mask.
        width: The width of the mask.
        crop_factor: The percent of the image to crop in each direction [top, bottom, left, right].
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The mask or None if no mask is needed.
    """
    crop_mask = generate_crop_mask(height, width, crop_factor)
    circle_mask = generate_circle_mask(height, width, percent_radius)
    if crop_mask is None:
        return circle_mask
    if circle_mask is None:
        return crop_mask
    return crop_mask * circle_mask


def save_mask(
    image_dir: Path,
    num_downscales: int,
    crop_factor: Tuple[float, float, float, float] = (0, 0, 0, 0),
    percent_radius: float = 1.0,
) -> Optional[Path]:
    """Save a mask for each image in the image directory.

    Args:
        image_dir: The directory containing the images.
        num_downscales: The number of downscaling levels.
        crop_factor: The percent of the image to crop in each direction [top, bottom, left, right].
        percent_radius: The radius of the circle as a percentage of the image diagonal size.

    Returns:
        The path to the mask file or None if no mask is needed.
    """
    image_path = next(image_dir.glob("frame_*"))
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]
    mask = generate_mask(height, width, crop_factor, percent_radius)
    if mask is None:
        return None
    mask *= 255
    mask_path = image_dir.parent / "masks"
    mask_path.mkdir(exist_ok=True)
    cv2.imwrite(str(mask_path / "mask.png"), mask)
    downscale_factors = [2**i for i in range(num_downscales + 1)[1:]]
    for downscale in downscale_factors:
        mask_path_i = image_dir.parent / f"masks_{downscale}"
        mask_path_i.mkdir(exist_ok=True)
        mask_path_i = mask_path_i / "mask.png"
        mask_i = cv2.resize(
            mask,
            (width // downscale, height // downscale),
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imwrite(str(mask_path_i), mask_i)
    CONSOLE.log(":tada: Generated and saved masks.")
    return mask_path / "mask.png"


def copy_image_to_subfolder(image_dir: Path, subfolder_dir: Path, image_index: list):
    images = list(sorted(image_dir.iterdir())) # , key=lambda x: int(x.name.split(".")[0])))
    image_paths = [images[image_index[i]] for i in range(len(image_index))]
    copy_images_list_new(image_paths=image_paths, image_dir=subfolder_dir, num_downscales=0, keep_image_dir=True)

def sample_index(chunk_length : int, num_sample_images_per_chunk : int, sample_strategy: str) -> List[int]:
    if sample_strategy == "random":
        index = random.sample(range(chunk_length), num_sample_images_per_chunk)
    elif sample_strategy == "uniform":
        assert num_sample_images_per_chunk <= chunk_length
        index = list(range(0, chunk_length, chunk_length // num_sample_images_per_chunk)) 
    else:
        exit(0)
    return index

def sample_images(sample_strategy: str, num_chunks: int, num_images_per_chunk: int, overlapped_fraction: float, image_dir: Path, output_dir: Path):
    images = list(sorted(image_dir.iterdir()))
    num_images = len(images)
    chunk_length = num_images // num_chunks
    non_overlapped_chunk_length = int(num_images // (num_chunks * (1 + overlapped_fraction)))
    overlapped_chunk_length = chunk_length - non_overlapped_chunk_length

    num_sample_images_per_non_overlapped_chunk = int(float(non_overlapped_chunk_length) / float(chunk_length) * num_images_per_chunk)
    num_sample_images_per_overlapped_chunk = num_images_per_chunk - num_sample_images_per_non_overlapped_chunk

    # firstly sample non-overlapped 
    non_overlapped_sample_index = []
    for i in range(num_chunks):
        index = sample_index(chunk_length=non_overlapped_chunk_length, num_sample_images_per_chunk=num_sample_images_per_non_overlapped_chunk, sample_strategy=sample_strategy)
        index = [inx + overlapped_chunk_length // 2 + i * chunk_length for inx in index]
        non_overlapped_sample_index.append(index)

    # secondly sample overlapped 
    overlapped_sample_index = []
    for i in range(num_chunks+1):
        index = sample_index(chunk_length=overlapped_chunk_length, num_sample_images_per_chunk=num_sample_images_per_overlapped_chunk, sample_strategy=sample_strategy)
        index = [inx - overlapped_chunk_length // 2 + i * chunk_length for inx in index]
        overlapped_sample_index.append(index)

    # lastly merge overlapped and non-overlapped 
    for i in range(num_chunks):
        non_overlapped_index = non_overlapped_sample_index[i]
        overlapped_index_left = overlapped_sample_index[i]
        overlapped_index_right = overlapped_sample_index[i+1]
        left_index = overlapped_index_left
        right_index = overlapped_index_right
        
        if i == 0:
            for j in range(len(overlapped_index_left)):
                if overlapped_index_left[j] >= 0:
                    break
            left_index = overlapped_index_left[j:]
        elif i == num_chunks - 1:
            for j in range(len(overlapped_index_right)):
                if overlapped_index_right[j] >= num_images:
                    break
            right_index = overlapped_index_right[:j] 
        
        index = left_index + non_overlapped_index + right_index
        chunk_folder = output_dir.joinpath("chunk_" + str(i)).joinpath("raw")
        copy_image_to_subfolder(image_dir=image_dir, subfolder_dir=chunk_folder, image_index=index)
        CONSOLE.log("[bold green]:tada: Done copying video/image chunks_{}.".format(i))

def resample_images():
    return 

def execute_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout,stderr

def colmap_multiprocessing(image_dir : Path, parallel: Optional[bool] = True):
    cmds = []
    chunk_paths = list(sorted(image_dir.iterdir()))
    for chunk_path in chunk_paths:
        cmd = "ns-process-data images --data {} --output-dir {}".format(chunk_path.joinpath("raw").absolute(), chunk_path)
        cmds.append(cmd)
    if parallel:
        CONSOLE.log("[bold green]:tada: Running colmap in parallel.")
        pool = multiprocessing.Pool(processes=len(cmds))
        results = pool.map(execute_cmd, cmds)
        pool.close()
        pool.join()
    else:
        CONSOLE.log("[bold green]:tada: Running colmap in tandem.")
        results = []
        for cmd in cmds:
            result = os.system(command=cmd)
            results.append((str(result).encode('utf-8'), str(result).encode('utf-8')))

    CONSOLE.log("[bold green]:tada: Colmap processing has done for all video/image chunks.")
    # for cmd, (stdout, stderr) in zip(cmds, results):
    #    CONSOLE.log("[bold green]:tada: Command {} executed with stdout: {} and stderr: {}.".format(stdout.decode(), stderr.decode())) 


def filter_points(data, threshold=0.9):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    ranges = norm.interval(threshold, loc=means, scale=stds)
    in_range = np.all((data >= ranges[0]) & (data <= ranges[1]), axis=1)
    indices_in_range = np.where(in_range)[0]
    indices_out_of_range = np.where(~in_range)[0]
    return indices_in_range, indices_out_of_range



def compute_coordinate_transform_matrix(chunk1: Path, chunk2: Path):
    with open(chunk1.joinpath("transforms.json").absolute(), 'r') as f:
        data1 = json.load(f)
        frames1 = data1["frames"]
        original_ids1 = dict(zip([frames1[i]["original_path"] for i in range(len(frames1))], [i for i in range(len(frames1))]))
        trans1 = [frames1[i]["transform_matrix"] for i in range(len(frames1))]

    with open(chunk2.joinpath("transforms.json").absolute(), 'r+') as f:
        data2 = json.load(f)
        frames2 = data2["frames"]
        original_ids2 = dict(zip([frames2[i]["original_path"] for i in range(len(frames2))], [i for i in range(len(frames2))]))
        trans2 = [frames2[i]["transform_matrix"] for i in range(len(frames2))]
    
        # find common ids:
        shared_ids = []
        for id1 in original_ids1.keys():
            if id1 in original_ids2.keys():
                shared_ids.append(id1)

        # compute coordinate average transformation matrix
        transitions, rotations = [], []
        positions1, positions2 = [], []
        untrusted_frames = []

        for id in shared_ids:
            mat1 = np.array(trans1[original_ids1[id]])
            mat2 = np.array(trans2[original_ids2[id]])
            pos1 = mat1[:3, 3]
            pos2 = mat2[:3, 3]
            positions1.append(pos1)
            positions2.append(pos2)
        positions1 = np.array(positions1)
        positions2 = np.array(positions2)
        distances1 = cdist(positions1, positions1).mean()
        distances2 = cdist(positions2, positions2).mean()
        scale = distances2 / distances1
        print("old distances1: ", distances1)
        print("old distances2: ", distances2)
        inrange, outrange = filter_points(positions1 - positions2, threshold=0.9)
        # recompute mutual distances
        positions1 = positions1[inrange]
        positions2 = positions2[inrange]
        distances1 = cdist(positions1, positions1).mean()
        distances2 = cdist(positions2, positions2).mean()
        scale = distances2 / distances1
        print("new distances1: ", distances1)
        print("new distances2: ", distances2)
        print("estimated scale is: ", scale)

        for i in range(len(shared_ids)):
            mat1 = np.array(trans1[original_ids1[shared_ids[i]]])
            mat2 = np.array(trans2[original_ids2[shared_ids[i]]])
            mat2[:3, 3] /= scale
            coor_trans = np.dot(mat1, np.linalg.inv(mat2))  # 2-->1
            transition = coor_trans[:3, 3]
            rotation = R.from_matrix(coor_trans[:3, :3])
            transitions.append(transition)
            rotations.append(rotation)

        transitions = [transitions[i] for i in inrange]
        rotations = [rotations[i] for i in inrange]

        for i in outrange:
            untrusted_frames.append(shared_ids[i])

        print("transitions:\n", transitions)
        print("rotations: \n", [ori.as_euler('xyz') for ori in rotations])
        average_transition = sum(transitions) / len(transitions)
        average_rotation_matrix = R.mean(R.concatenate(rotations)).as_matrix()
        average_transform_matrix = np.eye(4)
        average_transform_matrix[:3, :3] = average_rotation_matrix
        average_transform_matrix[:3, 3] = average_transition

        data2["scale"] = scale
        data2["coordination_transform_matrix"] = average_transform_matrix.tolist()
        data2["precedent_chunk"] = str(chunk1.absolute())
        f.seek(0)
        f.truncate()
        json.dump(data2, f, indent=4)
        print("untrusted frames:\n", untrusted_frames)
    return untrusted_frames


def recursive_compute_trans_matrix(chunk_path: Path):
    with open(chunk_path.joinpath("transforms.json"), 'r') as f:
        data = json.load(f)
        if "precedent_chunk" in data.keys() and "coordination_transform_matrix" in data.keys():
            transform_matrix = data["coordination_transform_matrix"]
            scale = data["scale"]
            precedent_chunk_path = Path(data["precedent_chunk"])
            pre_transform_matrix, pre_scale = recursive_compute_trans_matrix(precedent_chunk_path)
            transform_matrix =  np.dot(pre_transform_matrix, transform_matrix)
            scale =  pre_scale * scale
        else:
            scale = 1.0
            transform_matrix = np.eye(4)
        return transform_matrix, scale

def merge_frames(transform_info_file: Path, output_file: Path):
    with open(transform_info_file.absolute(),'r') as f:
        data = json.load(f)
        chunk_paths = data["chunk_path"]
        chunk_paths = [Path(chunk_path) for chunk_path in chunk_paths]
        transform_matrix = np.array(data["transform_matrix"])
    frames = []
    stamps = dict()
    for ck_path, trans_mat in zip(chunk_paths, transform_matrix):
        with open(ck_path.joinpath("transforms.json"),'r') as f:
            data = json.load(f)
            data_keys = list(data.keys())
            for k in data_keys:
                if k != "frames":
                    data.pop(k)
            new_frames = data["frames"]
            for i in range(len(new_frames)):
                global_id = new_frames[i]["original_path"]
                if global_id in stamps:
                    continue
                stamps[global_id] = True
                new_frames[i]["transform_matrix"] = np.dot(new_frames[i]["transform_matrix"], trans_mat).tolist()
                new_frames[i]["file_path"] = new_frames[i]["original_path"]
                new_frames[i].pop("original_path")
                frames.append(new_frames[i])
    with open(output_file.absolute(), 'r+') as f:
        data = json.load(f)
        data["frames"] = frames
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)

def merge_camera_intrinsics(chunk_paths: List[Path], output_file: Path):
    intrinsics_key = ['w', 'h', 'fl_x', 'fl_y', 'cx', 'cy', 'k1', 'k2', 'p1', 'p2']
    intrinsics_val = [0.0 for _ in intrinsics_key]
    intrinsics = dict(zip(intrinsics_key, intrinsics_val))
    for chunk_path in chunk_paths:
        with open(chunk_path.joinpath("transforms.json"), 'r') as f:
            data = json.load(f)
            for k in intrinsics_key:
                intrinsics[k] += data[k]
    for k in intrinsics.keys():
        intrinsics[k] = intrinsics[k] / len(chunk_paths)
    intrinsics['w'] = int(intrinsics['w'])
    intrinsics['h'] = int(intrinsics['h'])
    
    with open(output_file, 'w+') as f:
        data = dict()
        data.update(intrinsics)
        json.dump(data, f, indent=4)

def transform_point_cloud(points, transform_matrix, scale):
    """
    Apply transformation matrix to the points.
    """
    homogeneous_points = np.hstack((points / scale, np.ones((len(points), 1))))
    transformed_points = np.dot(homogeneous_points, transform_matrix.T)
    transformed_points = transformed_points[:, :3]
    return transformed_points


def create_output(positions, colors, filename):
    colors = colors.reshape(-1, 3) 
    positions = positions.reshape(-1,3)
    vertices = np.hstack([positions, colors])
    np.savetxt(filename, vertices, fmt='%f %f %f %d %d %d')  

    ply_header = \
    '''ply
    format ascii 1.0
    element vertex %(vert_num)d
    property float x
    property float y
    property float z
    property uint8 red
    property uint8 green
    property uint8 blue
    end_header
    '''
    with open(filename, 'r+') as f:
        old = f.read()
        f.seek(0)
        f.write(ply_header % dict(vert_num=len(vertices)))
        f.write(old)


def recursive_transform(chunk_path: Path, positions, colors):
    transforms_file = chunk_path.joinpath("transforms.json")
    with open(transforms_file, 'r') as f:
        data = json.load(f)
        if "precedent_chunk" in data.keys() and "coordination_transform_matrix" in data.keys():
            transform_matrix = data["coordination_transform_matrix"]
            scale = data["scale"]
            precedent_chunk_path = Path(data["precedent_chunk"])
            ply_data1 = PlyData.read(precedent_chunk_path.joinpath("sparse_pc.ply"))
            points1 = np.array([(vertex['x'], vertex['y'], vertex['z'], vertex['red'], vertex['green'], vertex['blue']) for vertex in ply_data1['vertex']])
            positions1 = points1[:, :3]
            colors1 = points1[:, 3:]
            positions = transform_point_cloud(positions, np.array(transform_matrix), scale)
            positions = np.concatenate([positions, positions1])
            colors = np.concatenate([colors, colors1])
            # create_output(positions=positions, colors=colors, filename=precedent_chunk_path.joinpath("from_{}_sparse_pc.ply".format(precedent_chunk_path.name)))
            return recursive_transform(precedent_chunk_path, positions=positions, colors=colors)
        else:
            return positions, colors


def merge_ply_files(transform_info_file: Path, output_file: Path):
    with open(transform_info_file, 'r') as f:
        data = json.load(f)
        chunk_paths = data["chunk_path"]
        chunk_paths = [Path(chunk_path) for chunk_path in chunk_paths]
    
    merged_positions, merged_colors = [], []
    for chunk_path in chunk_paths:
        ply_data = PlyData.read(chunk_path.joinpath("sparse_pc.ply"))
        points = np.array([(vertex['x'], vertex['y'], vertex['z'], vertex['red'], vertex['green'], vertex['blue']) for vertex in ply_data['vertex']])
        positions, colors = recursive_transform(chunk_path=chunk_path, positions=points[:, :3], colors=points[:, 3:])
        merged_positions.append(positions)
        merged_colors.append(colors)
    merged_positions = np.concatenate(merged_positions)
    merged_colors = np.concatenate(merged_colors)
    create_output(positions=merged_positions, colors=merged_colors, filename=output_file)


def merge_camera_model(chunk_paths: List[Path], output_file: Path):
    camera_model = ''
    for chunk_path in chunk_paths:
        with open(chunk_path.joinpath("transforms.json"), 'r') as f:
            data = json.load(f)
            if not camera_model:
                camera_model = data['camera_model']
            else:
                if data['camera_model'] != camera_model:
                    print("different camera model found!")
                    exit(0)

    with open(output_file, 'r+') as f:
        data = json.load(f)
        data['camera_model'] = camera_model
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)

def merge_ply_file_path(output_file: Path):
    with open(output_file.absolute(), 'r+') as f:
        data = json.load(f)
        data["ply_file_path"] = "sparse_pc.ply" 
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)    

def merge_images(transform_info_file: Path, image_dir: Path, output_dir: Path, output_file: Path):
    with open(transform_info_file, 'r') as f:
        data = json.load(f)
        untrusted_frames = set(data["untrusted_frames"])

    chunk_paths = list(sorted(image_dir.iterdir()))
    for chunk_path in chunk_paths:
        images = chunk_path.joinpath("raw").iterdir()
        images = set([im.name for im in images])
        untrusted = images.intersection(untrusted_frames)
        images.difference(untrusted)
        images = list(images)
        images_paths = [chunk_path.joinpath("raw").joinpath(im) for im in images]
        copy_images_list_new(image_paths=images_paths, image_dir=output_dir.joinpath("images"), num_downscales=0, keep_image_dir=True)
    
    with open(output_file, 'r+') as f:
        data = json.load(f)
        frames = data["frames"]
        images = [frame["file_path"] for frame in frames]
        indices = []
        for i in range(len(images)):
            if images[i] not in untrusted_frames:
                indices.append(i)
        frames = [frames[i] for i in indices]
        data["frames"] = frames
        f.seek(0)
        f.truncate()
        json.dump(data, f, indent=4)

        
def merge_chunks(image_dir: Path, output_dir: Path):
    chunk_paths = list(sorted(image_dir.iterdir()))
    output_dir.mkdir(exist_ok=True, parents=True)
    transform_matrices = []
    scales = []
    chunk_paths.sort(key=lambda x: x.absolute())
    untrusted_frames = []
    for i in range(len(chunk_paths)-1):
        untrusted_frames.extend(compute_coordinate_transform_matrix(chunk_paths[i], chunk_paths[i+1]))
    for chunk_path in chunk_paths:
        transform_matrix, scale = recursive_compute_trans_matrix(chunk_path=chunk_path)
        transform_matrices.append(transform_matrix.tolist())
        scales.append(scale)
    
    transform_info = dict()
    transform_info["transform_matrix"] = transform_matrices
    transform_info["scale"] = scales
    transform_info["chunk_path"] = [str(chunk_path.absolute()) for chunk_path in chunk_paths]
    transform_info["untrusted_frames"] = untrusted_frames
    transform_info_file = output_dir.joinpath("transform_merge_chunks.json")

    with open(transform_info_file, 'w+') as f:
        json.dump(transform_info, f, indent=4)
    
    new_transforms_json_file = output_dir.joinpath("transforms.json")
    new_ply_file = output_dir.joinpath("sparse_pc.ply")
    recursive_ply_file = output_dir.joinpath("recursive_sparse.ply")
    merge_camera_intrinsics(chunk_paths, output_file=new_transforms_json_file)
    merge_frames(transform_info_file=transform_info_file, output_file=new_transforms_json_file)
    merge_camera_model(chunk_paths, output_file=new_transforms_json_file)
    merge_ply_file_path(new_transforms_json_file)
    merge_ply_files(transform_info_file=transform_info_file, output_file=new_ply_file)
    merge_images(transform_info_file=transform_info_file, image_dir=image_dir, output_dir=output_dir, output_file=new_transforms_json_file)

def plot_trajectory(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract XYZ coordinates from poses
    xyz = np.array([[np.cos(p[1])*np.cos(p[0]), np.cos(p[1])*np.sin(p[0]), np.sin(p[1])] for p in poses[:, :3]])

    # Plot trajectory
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], marker='o', linestyle='-', color='b')

    # Plot orientation at each sample point
    for pose in poses:
        x, y, z = np.cos(pose[1])*np.cos(pose[0]), np.cos(pose[1])*np.sin(pose[0]), np.sin(pose[1])
        R = np.array([[np.cos(pose[2]), -np.sin(pose[2]), 0],
                      [np.sin(pose[2]), np.cos(pose[2]), 0],
                      [0, 0, 1]])
        width = 0.2
        height = 0.2
        rect = np.array([[-width/2, width/2, width/2, -width/2],
                         [-height/2, -height/2, height/2, height/2],
                         [0, 0, 0, 0]])
        rect_rotated = np.dot(R, rect)
        ax.plot_surface(x + rect_rotated[0, :].reshape((2, 2)), 
                        y + rect_rotated[1, :].reshape((2, 2)), 
                        z + rect_rotated[2, :].reshape((2, 2)),
                        color='red', alpha=0.5)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    ax.set_title('Interpolated Trajectory')
    plt.show()

def pose_interpolation(image_dir: Path, merged_dir: Path, output_dir: Path):
    merged_file = merged_dir.joinpath("transforms.json")
    output_file = output_dir.joinpath("transforms_interpolation.json")

    images = list(sorted(image_dir.iterdir()))
    images = [im.name for im in images]
    stamps = np.linspace(0, len(images) // 5, len(images))
    index = dict(zip(images, list(stamps)))

    with open(merged_file, 'r') as f:
        data = json.load(f)
    sampled_frames = data['frames']
    sampled_frames.sort(key=lambda x: x['file_path'])
    sampled_stamps = np.array([index[f['file_path']] for f in sampled_frames])

    # pop images that are out of the range of sampled stamps
    for i in range(len(images)):
        if images[i] == sampled_frames[0]['file_path']:
            break 
    for j in reversed(range(len(images))):
        if images[j] == sampled_frames[-1]['file_path']:
            break
    images = images[i:j+1]
    stamps = stamps[i:j+1]
    index = dict(zip(images, list(stamps)))

    
    positions = []
    rotations = []
    for frame in sampled_frames:
        transform_matrix = np.array(frame['transform_matrix'])
        position = transform_matrix[:3, 3]
        rotation = R.from_matrix(transform_matrix[:3, :3])
        positions.append(position)
        rotations.append(rotation)
    positions = np.array(positions)
    rotations = R.concatenate(rotations)

    # interpolate positions and orientations separately
    spline_position = CubicSpline(sampled_stamps, positions.T, axis=1)
    slerp = Slerp(sampled_stamps.tolist(), rotations)

    position_mse = np.mean(np.linalg.norm(spline_position(sampled_stamps).T - positions, axis=1)**2)
    orientation_mse = np.mean(np.linalg.norm(slerp(sampled_stamps.tolist()).as_euler('xyz') - rotations.as_euler('xyz'), axis=1)**2)
    CONSOLE.log("[bold green]:tada: Position interpolation error is {}.".format(position_mse))
    CONSOLE.log("[bold green]:tada: Orientation interpolation error is {}.".format(orientation_mse))

    # Compute interpolated transform matrix for unsampled images
    interpolated_positions = spline_position(stamps)
    interpolated_orientations = slerp(stamps.tolist())
    # interpolated_poses = np.concatenate((interpolated_positions.T, interpolated_orientations.T), axis=1)
    # # plot_trajectory(interpolated_poses)

    rotations = interpolated_orientations.as_matrix()
    positions = [np.array(position) for position in interpolated_positions.T.tolist()]

    transform_matrices = []
    for position, rotation in zip(positions, rotations):
        transform_matrix = np.eye(4)
        transform_matrix[:3, 3] = position  
        transform_matrix[:3, :3] = rotation  
        transform_matrices.append(transform_matrix.tolist())

    with open(output_file, 'w') as f:
        data = dict()
        data["file_path"] = [str(im) for im in images]
        data["transform_matrix"] =transform_matrices
        json.dump(data, f, indent=4)


def undistorting_images(image_dir: Path, output_dir: Path, many_chunks: bool=False):
    output_dir.mkdir(exist_ok=True, parents=True)
    empty_folder(output_dir)
    if many_chunks:
        chunk_paths = image_dir.iterdir()
        for chunk_path in chunk_paths:
            chunk_name = chunk_path.name
            chunk_dir = output_dir.joinpath(chunk_name)
            chunk_dir.mkdir(exist_ok=True, parents=True)
            input_dir = chunk_dir.joinpath("input")
            distorted_dir = chunk_dir.joinpath("distorted")
            input_dir.mkdir(exist_ok=True, parents=True)
            distorted_dir.mkdir(exist_ok=True, parents=True)
            copy_folder(chunk_path.joinpath("images"), input_dir)
            copy_folder(chunk_path.joinpath("colmap"), distorted_dir)
    else:
        input_dir = output_dir.joinpath("input")
        distorted_dir = output_dir.joinpath("distorted")
        input_dir.mkdir(exist_ok=True, parents=True)
        distorted_dir.mkdir(exist_ok=True, parents=True)
        copy_folder(image_dir.joinpath("images"), input_dir)
        copy_folder(image_dir.joinpath("colmap"), distorted_dir)


def grid_allocation(grid_size : float, image_dir: Path, merged_dir: Path, output_dir: Path, sampled: bool):
    if sampled:
        json_file = merged_dir.joinpath("transforms.json").absolute()
    else:
        json_file = merged_dir.joinpath("transforms_interpolation.json").absolute()
    with open(json_file, 'r') as f:
        data = json.load(f)
        if sampled:
            frames = [fm['file_path'] for fm in data["frames"]]
            transformation_matrices = [fm["transform_matrix"] for fm in data["frames"]]
        else:
            frames = data['file_path']
            transformation_matrices = data["transform_matrix"]
        # allocating grid ids according to their poses
        if "grid_id" in frames:
            CONSOLE.log("[bold red]:tada: Each images has already been allocated a grid ID. Re-allocating grid ID......")
        grid_ids = []
        
        for transform_matrix in transformation_matrices:
            pose_position = np.dot(transform_matrix, [0, 0, 0, 1])[:3]
            grid_x = int(pose_position[0] / grid_size)
            grid_y = int(pose_position[1] / grid_size)
            grid_ids.append("{}_{}".format(grid_x, grid_y))
        
    # copy images to their corresponding directories
    unique_grid_ids = list(set(grid_ids))
    if sampled:
        output_dirs = dict(zip(unique_grid_ids, [output_dir.joinpath("grid_{}".format(gid)).joinpath("sampled").absolute() for gid in unique_grid_ids]))
    else:
        output_dirs = dict(zip(unique_grid_ids, [output_dir.joinpath("grid_{}".format(gid)).joinpath("original").absolute() for gid in unique_grid_ids]))
    
    image_paths = dict(zip(unique_grid_ids, [[] for _ in unique_grid_ids]))
    for grid_id, image in zip(grid_ids, frames):
        image_paths[grid_id].append(image_dir.joinpath(image))
    
    for grid_id in unique_grid_ids:
        copy_images_list_new(image_paths=image_paths[grid_id], image_dir=output_dirs[grid_id], num_downscales=0, keep_image_dir=True)
    CONSOLE.log("[bold green]:tada: Successfully copied images to their corresponding blocks.")

        


    





