ncnn_bin_dir="/home/william/Codes/ncnn-20241226-ubuntu-2204/bin"
mbv3_onnx_path="onnx/mbv3_1ch_fp32_simp.onnx"

$ncnn_bin_dir/onnx2ncnn $mbv3_onnx_path \
    ncnn/mbv3_1ch_fp32.param \
    ncnn/mbv3_1ch_fp32.bin \

$ncnn_bin_dir/ncnnoptimize \
    ncnn/mbv3_1ch_fp32.param \
    ncnn/mbv3_1ch_fp32.bin \
    ncnn/mbv3_1ch_fp32_opt.param \
    ncnn/mbv3_1ch_fp32_opt.bin \
    0
