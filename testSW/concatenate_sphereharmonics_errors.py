import numpy as np

dim_list = [3,5,10,20]

for d in dim_list:
    x = np.load("errors_sphereharmonics_SWIncluded_PointsStored_in_"+str(d)+"D.npy")
    y = np.load("new_errors_sphereharmonics_SWIncluded_in_"+str(d)+"D.npy")
    # print(x)
    z = []
    # z = [min(rec1,rec2) for rec1,rec2 in zip(y[:6],x[:6])] + [rec for rec in x[6:15]]

    for rec1, rec2 in zip(x[:6],y[:6]):
        z.append(min(rec1,rec2))

    z = z + [rec for rec in x[6:15]]
    print(z)
    print(len(z))

    np.save("errors_sphereharmonics_concatenated_in_"+str(d)+"D.npy", np.array(z))


    print(np.array(z))

