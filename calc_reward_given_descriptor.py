import os
from hyper_params import hyper_params

def calc_reward_given_descriptor(descriptor):
    descriptor = descriptor.split("_")
    descriptor = [int(d) for d in descriptor]
    if True:
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
