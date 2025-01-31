import numpy as np

dim_list = [3,5,10,20]

for d in dim_list:
    x = np.load("timers_sphereharmonics_SWIncluded_PointsStored_in_"+str(d)+"D.npy")
    y = np.load("new_timers_sphereharmonics_SWIncluded_in_"+str(d)+"D.npy")

    z = []

    for rec1, rec2 in zip(x[:6],y[:6]):
        z.append(min(rec1,rec2))

    z = z + [rec for rec in x[6:15]]
    print(z)
    print(len(z))

    np.save("timers_sphereharmonics_concatenated_in_"+str(d)+"D.npy", np.array(z))