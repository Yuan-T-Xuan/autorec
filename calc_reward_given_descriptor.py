import os
from hyper_params import hyper_params

def calc_reward_given_descriptor(descriptor):
    descriptor = descriptor.split("_")
    descriptor = [int(d) for d in descriptor]
    assert len(descriptor) >= 3
    if descriptor[2] == 0:
        f = open("ped_template.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        code = code.replace("CHANGE_MARGIN_HERE", str(hyper_params[3][1][descriptor[3]]))
        outf = open("ped_executor_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python ped_executor_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        print("***************")
        print(lines[-1])
        return float(lines[-1].split()[-1])
        #
    elif descriptor[2] == 1:
        f = open("bpr_template.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        outf = open("bpr_executor_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python bpr_executor_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        print("***************")
        print(lines[-1])
        return float(lines[-1].split()[-1])
        #
    elif descriptor[2] == 2:
        # construct MLP_DIMS first
        MLP_DIMS = []
        for i in range(len(descriptor) - 4):
            curr_layer = hyper_params[4][1][descriptor[4+i]]
            if curr_layer < 0:
                break
            else:
                MLP_DIMS.append(curr_layer)
        MLP_DIMS.append(1)
        MLP_DIMS = str(MLP_DIMS)
        #
        f = open("pmlp_template.py")
        code = f.read()
        f.close()
        code = code.replace("CHANGE_DIM_HERE", str(hyper_params[0][1][descriptor[0]]))
        code = code.replace("CHANGE_L2_REG_HERE", str(hyper_params[1][1][descriptor[1]]))
        code = code.replace("CHANGE_MLP_DIMS_HERE", MLP_DIMS)
        outf = open("pmlp_executor_temp.py", 'w')
        outf.write(code)
        outf.close()
        os.system("python pmlp_executor_temp.py > result_tmp.out")
        f = open("result_tmp.out")
        lines = f.readlines()
        f.close()
        print("***************")
        print(lines[-1])
        return float(lines[-1].split()[-1])
        #
    else:
        raise Exception('wrong type_of_interaction!!!')

