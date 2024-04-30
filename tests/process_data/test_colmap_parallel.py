import multiprocessing 
import subprocess
from pathlib import Path
import os 

def execute_cmd(cmd):
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    return stdout,stderr

def colmap_multiprocessing(image_dir : Path, parallel: bool = True):
    cmds = []
    chunk_paths = list(sorted(image_dir.iterdir()))
    for chunk_path in chunk_paths:
        cmd = "ns-process-data images --data {} --output-dir {}".format(chunk_path.absolute(), "./tmp")
        cmds.append(cmd)
    if parallel:
        print("[bold green]:tada: Running colmap in parallel.")
        pool = multiprocessing.Pool(processes=len(cmds))
        results = pool.map(execute_cmd, cmds)
        pool.close()
        pool.join()
    else:
        print("[bold green]:tada: Running colmap in tandem.")
        results = []
        for cmd in cmds:
            result = os.system(command=cmd)
            results.append((str(result).encode('utf-8'), str(result).encode('utf-8')))

    print("[bold green]:tada: Colmap processing has done for all video/image chunks.")


if __name__ == "__main__":
    path = "/workspace/trial_new/samples/"
    colmap_multiprocessing(image_dir=Path(path), parallel=True)