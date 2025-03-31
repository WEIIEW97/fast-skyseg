from InferStudio.InferOFL import InferOFL
from InferStudio import modelInsight
from InferStudio import debugger
from InferStudio.compiler.compiler import compiler, stage, chipType, target, option
import os

if __name__ == "__main__":
    model_dir = "onnx/lraspp_mobilenetv3.onnx"
    out_dir = "/home/william/extdisk/data/ACE20k/ACE20k_sky/nvp_profile/"
    g_boardIP = "192.168.100.133"

    opt = option()
    opt.LoadFromModelDir(model_dir)
    opt.SetOutputPath(out_dir)
    opt.SetChipType(chipType.N161SX)
    opt.SetTarget(target.cpu)

    compiler_obj = compiler(opt)
    ret, msg = compiler_obj.Convert2Compile("")

    board_res = InferOFL.runOnlyOnce(compiler_obj, 0, g_boardIP)