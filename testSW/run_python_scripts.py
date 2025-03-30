import os
import subprocess

path1 = ["test_fib.py","test_fib_rand.py","test_ortho_vers.py","test_unif_vers.py",
    "test_riesz_rand.py","test_halton_smoothed2D.py","test_halton_smoothed3D.py","test_halton_rand3D.py",
    "test_halton_leluc.py","test_halton_rand_leluc.py","test_sobol_smoothed3D.py","test_sobol_rand3D.py",
    "test_sobol_leluc.py", "test_sobol_rand_leluc.py" "test_sphericalharmonics.py","test_sphericalharmonics_v2.py", 
    "concatenate_sphereharmonics_errors.py", "concatenate_sphereharmonics_timers.py"]


for file in path1:
    try:
        print(f"Running {file}...")
        subprocess.run(["python3", file], check=True)
        print(f"Finished {file}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error running {file}: {e}\n")

    

