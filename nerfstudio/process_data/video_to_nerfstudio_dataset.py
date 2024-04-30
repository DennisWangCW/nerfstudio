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

"""Processes a video to a nerfstudio compatible dataset."""

import shutil
from dataclasses import dataclass
from typing import Literal

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class VideoToNerfstudioDataset(ColmapConverterToNerfstudioDataset):
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images and downscales them.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    """

    num_frames_target: int = 300
    """Target number of frames to use per video, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        summary_log = []
        summary_log_eval = []
        if not self.skip_video_processing:    
            # Convert video to images
            if self.camera_type == "equirectangular":
                # create temp images folder to store the equirect and perspective images
                temp_image_dir = self.output_dir / "temp_images"
                temp_image_dir.mkdir(parents=True, exist_ok=True)
                summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                    self.data,
                    image_dir=temp_image_dir,
                    num_frames_target=self.num_frames_target if not self.get_full_images else 0,
                    num_downscales=0,
                    crop_factor=(0.0, 0.0, 0.0, 0.0),
                    verbose=self.verbose,
                )
            else:
                # If we're not dealing with equirects we can downscale in one step.
                summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                    self.data,
                    image_dir=self.image_dir,
                    num_frames_target=self.num_frames_target if not self.get_full_images else 0,
                    num_downscales=self.num_downscales,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                    image_prefix="frame_train_" if self.eval_data is not None else "frame_",
                    keep_image_dir=False,
                )
                if self.eval_data is not None:
                    summary_log_eval, num_extracted_frames_eval = process_data_utils.convert_video_to_images(
                        self.eval_data,
                        image_dir=self.image_dir,
                        num_frames_target=self.num_frames_target if not self.get_full_images else 0,
                        num_downscales=self.num_downscales,
                        crop_factor=self.crop_factor,
                        verbose=self.verbose,
                        image_prefix="frame_eval_",
                        keep_image_dir=True,
                    )
                    summary_log += summary_log_eval
                    num_extracted_frames += num_extracted_frames_eval

            # Generate planar projections if equirectangular
            if self.camera_type == "equirectangular":
                if self.eval_data is not None:
                    raise ValueError("Cannot use eval_data with camera_type equirectangular.")

                perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                    self.output_dir / "temp_images", self.images_per_equirect
                )

                equirect_utils.generate_planar_projections_from_equirectangular(
                    self.output_dir / "temp_images",
                    perspective_image_size,
                    self.images_per_equirect,
                    crop_factor=self.crop_factor,
                )

                # copy the perspective images to the image directory
                process_data_utils.copy_images(
                    self.output_dir / "temp_images" / "planar_projections",
                    image_dir=self.output_dir / "images",
                    verbose=False,
                )

                # remove the temp_images folder
                shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

                self.camera_type = "perspective"

                # # Downscale images
                summary_log.append(
                    process_data_utils.downscale_images(self.image_dir, self.num_downscales, verbose=self.verbose)
                )

            # Create mask
            mask_path = process_data_utils.save_mask(
                image_dir=self.image_dir,
                num_downscales=self.num_downscales,
                crop_factor=(0.0, 0.0, 0.0, 0.0),
                percent_radius=self.percent_radius_crop,
            )
            if mask_path is not None:
                summary_log.append(f"Saved mask to {mask_path}")
        
        if self.get_full_images:
            if not self.skip_video_processing:
                process_data_utils.sample_images(sample_strategy=self.chunk_image_sample_strategy, 
                                                num_chunks=self.num_overlapped_chunks, 
                                                overlapped_fraction=self.overlapped_fraction,
                                                num_images_per_chunk=self.target_frames_per_chunk, 
                                                image_dir=self.image_dir, output_dir=self.sample_dir)
            CONSOLE.log("[bold green]:tada: Done generating video/image chunks!")

            if not self.skip_colmap:
                if self.undistorted:
                    process_data_utils.colmap_multiprocessing(image_dir=self.sample_dir, output_dir=self.undistorted_dir, parallel=self.parallel_colmap, undistorted=self.undistorted, skip_colmap=self.skip_colmap, skip_image_processing=self.skip_video_processing)
                    CONSOLE.log("[bold green]:tada: Done Colmap processing for each chunk! Images are undistorted version!")
                else:
                    process_data_utils.colmap_multiprocessing(image_dir=self.sample_dir, output_dir=self.distorted_dir, parallel=self.parallel_colmap, undistorted=self.undistorted, skip_colmap=self.skip_colmap, skip_image_processing=self.skip_video_processing)
                    CONSOLE.log("[bold green]:tada: Done Colmap processing for each chunk! Images are distorted version!")
            # if self.undistorted:
            #     # process_data_utils.undistorting_images(image_dir=self.sample_dir, output_dir=self.undistorted_dir, num_overlapped_chunks=self.num_overlapped_chunks)
            #     chunk_paths = self.undistorted_dir.iterdir()
            #     for chunk_path in chunk_paths:
            #         self._run_colmap_image_undistortion()
                # process_data_utils.colmap_undistortion_multiprocessing(image_dir=self.undistorted_dir)
                CONSOLE.log("[bold green]:tada: Done undistorting all images!")
            process_data_utils.merge_chunks(image_dir=self.distorted_dir if not self.undistorted else self.undistorted_dir, output_dir=self.merge_dir)
            CONSOLE.log("[bold green]:tada: Done merging all video/image chunks!")
            self._run_json_and_ply_to_colmap(self.merge_dir)

            # process_data_utils.pose_interpolation(image_dir=self.image_dir, merged_dir=self.merge_dir, output_dir=self.merge_dir)
            # process_data_utils.grid_allocation(grid_size=self.grid_size,image_dir=self.image_dir, merged_dir=self.merge_dir, output_dir=self.grid_dir, sampled=False)
        else:
            # Run Colmap
            if not self.skip_colmap:
                self._run_colmap(mask_path)
            if self.undistorted:
                process_data_utils.undistorting_images(output_dir=self.undistorted_dir)
                self._run_colmap_image_undistortion()

            # Export depth maps
            image_id_to_depth_path, log_tmp = self._export_depth()
            summary_log += log_tmp

            summary_log += self._save_transforms(num_extracted_frames, image_id_to_depth_path, mask_path)

            CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")

            for summary in summary_log:
                CONSOLE.log(summary)
