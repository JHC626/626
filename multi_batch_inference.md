# 关于在jetson中的模型多batch推理

Author：JHC


Date:2024年12月20日


## 1.将模型转为onnx格式
起初转不了动态 batch 一直认为是onnx转 tensorrt 或者是推理代码出了问题，后面多次排查才发现实际上是 onnx 模型没有实现动态 batch ，如果是使用例如YOLO这种模型，在 dynamic=True 这里千完不要忘记，起初认为是不是 onnx 版本出了问题，后来发现实际上就是这一个小参数的问题，导致 onnx 始终没有转为动态 batch ，起初走了很多弯路，找各种程序企图改已经是静态 batch 的 onnx 模型为动态，实际上都是“治标不治本”，可能输入输出最后是动态的了，但实际上中间的网络结构没有变，结果还是转不了。

以yolov8官方模型为例
```bash
from ultralytics import YOLO
import torch
# 加载预训练的 PyTorch 模型
pytorch_model_path = r'/JHC/yolov8m_planeship_hbb.pt'
model = YOLO(pytorch_model_path)
model.export(format='onnx', opset=11, dynamic=True, simplify=True)
```

如果是正常的自己构建的模型则需要用下面方式导出，注意如果 pt 文件是一个字典类型如 YOLO 这个pt，是不可以直接 torch.load 的，需要配合其他网络结构的文件来先构建好整个模型。
```bash
# 导出模型
torch.onnx.export(
    model,                        # 模型
    dummy_input,                  # 示例输入
    "1.onnx",       # 导出的 ONNX 文件名
    export_params=True,           # 是否导出参数
    opset_version=11,             # ONNX 操作集版本
      
    input_names=['input'],        # 输入名称
    output_names=['output'],      # 输出名称
    dynamic_axes=dynamic_axes    # 动态轴设置
)
```
## 2.将onnx转为tensorrt
使用 trtexec 工具即可轻松的完成 onnx 到 tensorrt 的转换
```bash
trtexec --onnx=/JHC/yolov8m_planeship_hbb_onnx_1.12.0.onnx --minShapes=images:1x1x1024x1024 --optShapes=images:8x1x1024x1024 --maxShapes=images:32x1x1024x1024 --saveEngine=/JHC/dynamic_yolov8m_planeship_hbb_onnx_1.12.0.plan
```
## 3.模型推理
可参考如下程序的格式进行推理 注：以下示例 tensorrt 版本为10.3.0，很多老版本的函数方法等都不可再用。 
```bash
def slide_batches_inference(image_array_batches:np.ndarray ,img_size)->np.ndarray:
    """推理一个小batch的窗口图像

    Args:
        image_array_batches (np.ndarray): [Batch, 1, img_size, img_size] (128,1,1024,1024)的图像
        model: 寒武纪的magic model
        img_size: 推理的图像尺寸

    Returns:
        preds (np.ndarray): 模型输出文件，如果是yolov8 obb的话，为 [Batch, 20, 21504], 然后再去做 NMS 后处理

    """
    preprocessed_image_array_batches = img_batches_preprocess(image_array_batches, normalization=True)

    inputs = preprocessed_image_array_batches # 存了当前 batch 要推理的数据 (batch_size, 1, 1024, 1024)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    with open("/JHC/dynamic_yolov8m_planeship_hbb_onnx_1.12.0.plan", "rb") as f:
        model_data = f.read()
    engine = runtime.deserialize_cuda_engine(model_data)

# 创建执行上下文
    context = engine.create_execution_context()
    
# 定义输入和输出大小
     # 动态批量大小
     # 输入绑定索引为 0

    input_size = [image_array_batches.shape[0],1,1024, 1024]
    output_size = [image_array_batches.shape[0], 19, 21504]
    
# 计算缓冲区大小
    input_buffer_size = int(np.prod(input_size) * np.float32().nbytes)
    output_buffer_size = int(np.prod(output_size) * np.float32().nbytes)
# 分配缓冲区
    input_buffer = cuda.mem_alloc(input_buffer_size)
    output_buffer = cuda.mem_alloc(output_buffer_size)
# 获取输入和输出张量的名称
    input_name = engine.get_tensor_name(0)  # 假设输入张量是第一个绑定
    output_name = engine.get_tensor_name(1)  # 假设输出张量是第二个绑定
    context.set_input_shape(input_name, input_size)
# 设置张量地址
    context.set_tensor_address(input_name, int(input_buffer))
    context.set_tensor_address(output_name, int(output_buffer))
# 创建 CUDA 流
    stream =cuda.Stream()
# 异步将输入数据从主机拷贝到设备（cudaMemcpyAsync）
    # input_data = np.random.rand(*input_size).astype(np.float32)
    # input_data = inputs[0].ravel()
    input_data = inputs
    cuda.memcpy_htod_async(input_buffer, input_data, stream)
    stream.synchronize()
# 执行推理
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
# 获取输出数据
    outputs = np.empty(output_size, dtype=np.float32)
    cuda.memcpy_dtoh_async(outputs, output_buffer, stream)
    stream.synchronize()
    preds = torch.from_numpy(outputs) 
    return preds
```