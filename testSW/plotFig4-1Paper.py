import numpy as np
import matplotlib.pyplot as plt

dimensions = [2,3,5,10,20,50]

fib_errors = np.load("errors_fib_SWIncluded_in_3D.npy")


fib_rand_errors = np.load("errors_fib_SWIncluded_rand_in_3D.npy")




ortho_errors_list = [np.load("errors_ortho_SWIncluded_in_"+str(d)+"D.npy") for d in dimensions]

unif_errors_list = [np.load("errors_unif_SWIncluded_PointsStored_in_"+str(d)+"D.npy") for d  in dimensions]

riesz_errors_list = [np.load("errors_riesz_rand_in_"+str(d)+"D.npy") for d in dimensions]

halton_errors_list = [np.load("errors_haltonarea_smoothed_SWIncluded_in_2D.npy")] + [np.load("errors_haltonarea_smoothed_SWIncluded_in_3D.npy")] + [np.load("errors_halton_leluc_in_"+str(d)+"D.npy") for d in dimension[2:]]

halton_rand_errors_list = [np.load("errors_haltonarearand_SWIncluded_in_3D.npy")] + [np.load("errors_haltonrand_leluc_in_"+str(d)+"D.npy") for d in dimension[2:]]

sobol_errors_list = [np.load("errors_sobolarea_smoothed_SWIncluded_in_3D.npy")] + [np.load("errors_sobol_leluc_in_"+str(d)+"D.npy" for d in dimension[2:])]

sobol_rand_errors = [np.load("errors_sobolarearand_SWIncluded_in_3D.npy")] + [np.load("errors_sobolarea_smoothed_SWIncluded_in_3D.npy")] + [np.load("errors_sobolrand_leluc_in_"+str(d)+"D.npy" for d in dimension[2:])]

spherical_harmonics_errors_list = [np.load("errors_sphereharmonics_concatenated_in_"+str(d)+"D.npy") for d in [3,5,10,20]]

# # Example error arrays (replace these with your actual data)
# errors = {
#     2: [np.random.rand(10) for _ in range(3)],
#     3: [np.random.rand(10) for _ in range(4)],
#     5: [np.random.rand(10) for _ in range(5)],
#     10: [np.random.rand(10) for _ in range(2)],
# }

# # Create subplots
# num_dims = len(errors)
# fig, axes = plt.subplots(num_dims, 1, figsize=(6, 4 * num_dims))

# if num_dims == 1:
#     axes = [axes]  # Ensure axes is iterable when there's only one subplot

# # Plot errors for each dimension
# for ax, (dim, err_lists) in zip(axes, errors.items()):
#     for err in err_lists:
#         ax.plot(err, marker='o', linestyle='-', label=f'Error Set {err_lists.index(err) + 1}')
#     ax.set_title(f'Errors in Dimension {dim}')
#     ax.set_xlabel('Index')
#     ax.set_ylabel('Error')
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.show()

num_dims = 6

fig, axs = plt.subplots(3,2,figsize=(12, 6))

for i in range(3):
    for j in range(2):
        if i == 0 and j == 0:
            axs[i,j].set_title("2D")
            axs[i,j].plot(riesz_errors_list[0], label = "Riesz randomized")
            axs[i,j].plot(ortho_errors_list[0], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[0], label = "Uniform")
            axs[i,j].plot(halton_errors_list[0], label = "Halton")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")

        elif i == 0 and j == 1:
            axs[i,j].set_title("3D")
            axs[i,j].plot(riesz_errors_list[1], label = "Riesz")
            axs[i,j].plot(ortho_errors_list[1], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[1], label = "Uniform")
            axs[i,j].plot(halton_errors_list[1], label = "Halton")
            axs[i,j].plot(halton_rand_errors_list[0], label = "Halton Rand Area")
            axs[i,j].plot(sobol_errors_list[1], label = "Sobol")
            axs[i,j].plot(sobol_rand_errors_list[0], label = "Sobol Rand Area")
            axs[i,j].plot(fib_errors, label = "Fibonacci")
            axs[i,j].plot(fib_rand_errors, label = "Fibonacci randomized")
            axs[i,j].plot(spherical_harmonics_errors_list[0], label = "SHCV")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")
        elif i == 1 and j == 0:
            axs[i,j].set_title("5D")
            axs[i,j].plot(riesz_errors_list[2], label = "Riesz")
            axs[i,j].plot(ortho_errors_list[2], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[2], label = "Uniform")
            axs[i,j].plot(halton_errors_list[2], label = "Halton")
            axs[i,j].plot(halton_rand_errors_list[1], label = "Halton Rand Area")
            axs[i,j].plot(sobol_errors_list[2], label = "Sobol")
            axs[i,j].plot(sobol_rand_errors_list[1], label = "Sobol Rand Area")
            axs[i,j].plot(spherical_harmonics_errors_list[1], label = "SHCV")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")
        elif i == 1 and j == 1:
            axs[i,j].set_title("10D")
            axs[i,j].plot(riesz_errors_list[3], label = "Riesz")
            axs[i,j].plot(ortho_errors_list[3], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[3], label = "Uniform")
            axs[i,j].plot(halton_errors_list[3], label = "Halton")
            axs[i,j].plot(halton_rand_errors_list[2], label = "Halton Rand Area")
            axs[i,j].plot(sobol_errors_list[3], label = "Sobol")
            axs[i,j].plot(sobol_rand_errors_list[2], label = "Sobol Rand Area")
            axs[i,j].plot(spherical_harmonics_errors_list[2], label = "SHCV")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")
        elif i == 2 and j == 0:
            axs[i,j].set_title("20D")
            axs[i,j].plot(riesz_errors_list[4], label = "Riesz")
            axs[i,j].plot(ortho_errors_list[4], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[4], label = "Uniform")
            axs[i,j].plot(halton_errors_list[4], label = "Halton")
            axs[i,j].plot(halton_rand_errors_list[3], label = "Halton Rand Area")
            axs[i,j].plot(sobol_errors_list[4], label = "Sobol")
            axs[i,j].plot(sobol_rand_errors_list[3], label = "Sobol Rand Area")
            axs[i,j].plot(spherical_harmonics_errors_list[4], label = "SHCV")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")
        elif i == 2 and j == 1:
            axs[i,j].set_title("50D")
            axs[i,j].plot(riesz_errors_list[5], label = "Riesz")
            axs[i,j].plot(ortho_errors_list[5], label = "Orthonormal")
            axs[i,j].plot(unif_errors_list[5], label = "Uniform")
            axs[i,j].plot(halton_errors_list[5], label = "Halton")
            axs[i,j].plot(halton_rand_errors_list[4], label = "Halton Rand Area")
            axs[i,j].legend()
            axs[i,j].set_xlabel("N_sample")
            axs[i,j].set_ylabel("Error")

plt.tight_layout()
plt.show()

