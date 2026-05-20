import os
from analysis import *

# Root directory containing subdirectories with PDB/DCD files
root_dir = os.getcwd()

# Loop through each subdirectory
for subdir in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir)
    if os.path.isdir(subdir_path):
        pdb_file = os.path.join(subdir_path, "system.pdb")
        dcd_file = os.path.join(subdir_path, "trajectory.dcd")

        # Only run if both files exist
        if os.path.exists(pdb_file) and os.path.exists(dcd_file):
            print(f"Running analysis for {subdir_path}..")

            # Save current directory
            old_cwd = os.getcwd()
            try:
                # Change working directory to the subdirectory
                os.chdir(subdir_path)

                # Run analysis in the subdirectory
                runAll(pdb=pdb_file, dcd=dcd_file)
            finally:
                # Go back to original directory
                os.chdir(old_cwd)

        else:
            print(f"Skipping {subdir_path}: missing PDB or DCD file.")
            
            
# import os
# import subprocess

# # Name of your analysis script
# ANALYSIS_SCRIPT = "analysis.py"

# def has_required_files(path):
#     """Check if folder contains system.pdb + trajectory.dcd."""
#     files = os.listdir(path)
#     return ("system.pdb" in files) and ("trajectory.dcd" in files)

# def find_simulation_dirs(root="."):
#     """Yield all subdirectories with required files."""
#     for current, dirs, files in os.walk(root):
#         if has_required_files(current):
#             yield current

# def run_analysis_in_dir(folder):
#     """Run 'python analysis.py' inside the folder."""
#     print(f"\n=== Running analysis in {folder} ===")

#     # Run analysis.py in the folder using a subprocess
#     subprocess.run(
#         ["python", ANALYSIS_SCRIPT],
#         cwd=folder,
#         check=True
#     )

#     print(f"=== Finished {folder} ===")

# if __name__ == "__main__":
#     root = "."  # or absolute path

#     sim_dirs = list(find_simulation_dirs(root))

#     if not sim_dirs:
#         print("No simulation directories found.")
#         exit()

#     print(f"Found {len(sim_dirs)} directories.")

#     for folder in sim_dirs:
#         run_analysis_in_dir(folder)

#     print("\nAll simulation folders complete.")
