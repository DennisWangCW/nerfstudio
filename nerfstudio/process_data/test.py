
from process_data_utils import compute_coordinate_transform_matrix
from pathlib import Path


if __name__ == "__main__":
    path1 = Path("/workspace/tmp/samples/chunk_0")
    path2 = Path("/workspace/tmp/samples/chunk_1")
    compute_coordinate_transform_matrix(path1, path2)