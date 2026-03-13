# you may need to install openvdb via conda or from source, as it's not available on PyPI/uv.
import openvdb as vdb # assumes openvdb is installed
import numpy as np
import argparse

def load_vdb_to_numpy(path, grid_name='density'):
    grid = vdb.read(path, grid_name)
    # Get the dimensions of the active voxels
    bbox = grid.evalActiveVoxelBoundingBox()
    dim = np.array(bbox[1]) - np.array(bbox[0]) + 1
    
    # Create a dense array and fill it
    data = np.zeros(dim, dtype=np.float32)
    grid.copyToArray(data, ijk=bbox[0])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert OpenVDB file to NumPy array')
    parser.add_argument('vdb_path', type=str, help='Path to the input OpenVDB file')
    parser.add_argument('output_path', type=str, help='Path to save the output NumPy file')
    parser.add_argument('--grid_name', type=str, default='density', help='Name of the grid to extract from the VDB file')
    args = parser.parse_args()

    density_data = load_vdb_to_numpy(args.vdb_path, args.grid_name)
    density_data = np.transpose(density_data, (2, 1, 0))
    density_data = np.ascontiguousarray(density_data)
    np.save(args.output_path, density_data)
    print(f"Converted VDB grid '{args.grid_name}' to NumPy array with shape: {density_data.shape} and saved to {args.output_path}")