import os

hyper_params = [
    ("dim_of_latent_factor", [25, 50, 75, 100, 125, 150]),
    ("l2_reg", [0.1, 0.01, 0.001, 0.0001, 0.00001]),
    ("type_of_interaction", ["PairwiseEuDist", "PairwiseLog", "PointwiseMLPCE"]),
    ("eudist_margin", [0.5, 1.0, 1.5, 2.0]),
    ("mlp_dim1", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim2", [-1, 50, 75, 100, 125, 150]),
    ("mlp_dim3", [-1, 50, 75, 100, 125, 150])
]

def calc_reward_given_descriptor(descriptor):
    descriptor = descriptor.split("_")
    descriptor = [int(d) for d in descriptor]
    assert len(descriptor) >= 3
    if descriptor[2] == 0:
        assert len(descriptor) >= 4
        f = open("ped_citeulike.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        code = code.replace("CHANGE_MARGIN_HERE", str(hyper_params[3][1][descriptor[3]]))
        outf = open("ped_citeulike_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python ped_citeulike_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        return (float(lines[-1].split()[-1]) - 0.92) * 10
        #
    elif descriptor[2] == 1:
        f = open("bpr_citeulike.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        outf = open("bpr_citeulike_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python bpr_citeulike_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        return (float(lines[-1].split()[-1]) - 0.92) * 10
        #
    elif descriptor[2] == 2:
        # construct MLP_DIMS first
        MLP_DIMS = []
        for i in range(len(descriptor) - 3):
            curr_layer = hyper_params[4][1][descriptor[3+i]]
            if curr_layer < 0:
                break
            else:
                MLP_DIMS.append(curr_layer)
        MLP_DIMS.append(1)
        MLP_DIMS = str(MLP_DIMS)
        # print(MLP_DIMS)
        #
        f = open("pmlp_citeulike.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        code = code.replace("CHANGE_MLP_DIMS_HERE", MLP_DIMS)
        outf = open("pmlp_citeulike_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python pmlp_citeulike_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        return (float(lines[-1].split()[-1]) - 0.92) * 10
        #
    else:
        print("wrong interaction type")
        return None
