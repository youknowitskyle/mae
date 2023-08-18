from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

model_fp32 = './output_dir/model.onnx'
model_quant = './output_dir/model.quant.onnx'

model = onnx.load(model_fp32)

names = names=[node.name for node in model.graph.node]
prefix = ["MatMul", "Add", "Relu"]
linear_names = [v for v in names if v.split("_")[0] in prefix]


quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8, nodes_to_exclude=['/patch_embed/proj/Conv'], extra_options={"MatMulConstBOnly":True})