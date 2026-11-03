# FPGA 双目动捕与 G1 仿真：关键源码

双目摄像头、AMD FPGA 和 PC 协同处理人体动作，并将三维骨架映射到 Unitree G1 仿真。本仓库面向比赛评审和代码阅读，仅发布关键源码、必要结构配置和第三方许可证。

## 全流程

```text
双摄采集 → 帧配对与图像预处理 → 板端 CPU/FPGA 二维姿态推理
       → JPEG/骨架传输 → PC 二维精修 → 三维重建与缺失处理
       → 动作映射、参考轨迹与保护控制 → G1 / MuJoCo → 四画面 UI
```

协同路径中，板端运行 PP-TinyPose，为 PC 提供人体初定位；PC 端使用 RTMPose 精修二维关键点，再完成三维重建与 G1 对接。最终二维精修并非全部运行在 FPGA 上。

| 目录 | 主要内容 | 阅读入口 |
|---|---|---|
| `board_fpga/` | 双摄采集、20 ms 配对、推理调度、HLS、Linux 驱动与构建源码 | [双目入口](board_fpga/ultrascale_plus/u02/inference/host/pose_stereo_pipeline.cpp)、[HLS](board_fpga/ultrascale_plus/u02/inference/hls/stream_dw128_write_v2.cpp) |
| `board_hybrid_overlay/` | JPEG 与骨架联合传输所需的新增/替换源码；按同名相对路径覆盖板端基线源码后用于协同链 | [协同入口](board_hybrid_overlay/ultrascale_plus/u02/inference/host/pose_stereo_pipeline.cpp) |
| `pc/` | 四画面 UI、PC 精修、三维模型与推理、标定、BVH 导出 | [UI](pc/demo_station/app.py)、[二维精修](pc/demo_station/pose_refiner.py)、[三维重建](pc/pose3d_train/online_pair20.py) |
| `runtime_extensions/` | 实时接收、预测、事务缓存与质量保护的外部模块 | [事务流](runtime_extensions/transaction/transaction_stream.py)、[预测服务](runtime_extensions/prediction/prediction_server.py) |
| `g1/` | G1 重定向、物理跟踪、控制服务、场景与必要依赖源文件 | [重定向](g1/g1_realtime/tracker.py)、[物理跟踪](g1/g1_physical/physical.py) |

## 实时与文件演示入口

- 真实双摄链：`pc/demo_station/app.py` / `Start.cmd`。
- 独立 BVH 文件演示：`pc/demo_station/bvh_app.py` / `Start-BVH.cmd`。由文件生成二维投影、三维骨架和 G1 动作，属于文件回放，不是现场双摄识别。
- 原生 MuJoCo 窗口联动：`pc/demo_station/bvh_native_viewer.py`。窗口跟随主 UI 的姿态与播放位置；物理结果由 `bvh_worker.py` 调用 G1 运行环境生成。

## 依赖与使用范围

这是供阅读和集成的源码集合，不是解压即可运行的完整安装包。需要另行准备模型权重、G1 控制策略及机器人网格、相机标定、匹配的 FPGA 位流/内核驱动、Python 环境和编译工具链。部分原始路径已去除本机用户名，部署时仍需配置路径；`runtime_extensions/` 按功能重新归类，调用入口需要对应部署位置。

Python 依赖参考各模块的 `requirements*.txt`；主要包括 PySide6、NumPy、OpenCV、PyTorch、ONNX Runtime、MuJoCo。FPGA/板端构建还依赖 Vivado/Vitis HLS、ARM 交叉编译工具链、ncnn 和 JPEG 库。

本仓库不包含日志、测试结果、历史交接、录像、数据集、权重、编译产物、运行资格记录或本机账号配置。现场资格与模型身份检查保留在源码中，不以缺少这些文件为理由绕过检查。

第三方源代码及授权说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本仓库没有为整个项目额外声明统一的 MIT/Apache 等许可；各第三方子目录继续遵循其自身许可证。
