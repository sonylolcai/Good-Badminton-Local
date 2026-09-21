# Mac ChatGPT 任务书：羽毛球球拍关键点模型训练
> **用途**：把本文件和 `deploy/release/good-badminton-racket-pose-training-bundle-20260831.zip` 一起带到 Mac。将下方“给 Mac ChatGPT 的任务提示词”完整发送给 Mac 上的 ChatGPT/Codex，让它在你确认后编排远程 NVIDIA GPU 训练。
>
> **当前结论**：Mac 负责准备、校验、上传、监控和下载结果；正式训练必须使用一台具备 NVIDIA CUDA 的远程 GPU。不要在 Mac 或当前 Windows 本机上启动 100 epoch 正式训练，也不要安装本仓库的 `requirements.txt` 到 GPU 环境。

## 1. 已准备好的输入与事实

项目目录中已有以下可移动文件：

| 文件 | 作用 |
| --- | --- |
| `deploy/release/good-badminton-racket-pose-training-bundle-20260831.zip` | 自包含训练包：源码、YOLO Pose 数据集、基础权重和远程训练说明。|
| `deploy/REMOTE_GPU_RACKET_POSE_TRAINING.md` | 面向 GPU 服务器的简版操作说明。|
| `datasets/racket_pose_v1/` | 已生成的数据集目录；如已带走 ZIP，则不需要再单独复制。|

训练包的已知校验值：

```text
SHA-256: E99CD7904CAC2A315E3F7686D3B81E2DAAA2BE54FB03C8DFE1B52879F48C06C7
大小: 100.57 MiB
```

数据集事实，不可在训练中随意改变：

- 任务：单类别 `racket` 的 YOLO Pose。
- 5 个关键点顺序：`grip, throat, head_top, head_left, head_right`。
- `kpt_shape: [5, 3]`，左右拍头翻转映射：`flip_idx: [0, 1, 2, 4, 3]`。
- 训练/验证按完整源视频切分，不能把同一源视频的相邻帧同时放入训练和验证。
- 当前版本：94 张训练图、20 张验证图；210 / 35 个完整球拍实例；36 个不完整标注组已记录在 `skipped_groups.csv`。
- `racket-pose.yaml` 是可迁移的相对路径配置；解压后不要加回 Windows 的绝对路径。

## 2. 给 Mac ChatGPT 的任务提示词

复制以下内容到 Mac 上的 ChatGPT/Codex。方括号内容由使用者提供；不要让模型虚构服务器地址、账号、密码或训练结果。

```text
你正在执行 Good-Badminton 的“球拍 YOLO Pose 远程 GPU 训练”任务。

本机是 Mac，只负责校验训练包、通过 SSH/SCP 将训练包传到远程 NVIDIA GPU、监控训练并把产物取回。不要把 Mac MPS、CPU 或本机没有验证过的环境当成正式训练环境。

训练包路径：[/Users/你的用户名/Downloads/good-badminton-racket-pose-training-bundle-20260831.zip]
远程 GPU 信息：主机 [HOST]，用户 [USER]，远程工作目录 [REMOTE_DIR]。

先完成以下只读检查，并逐项报告证据：
1. 验证 ZIP 的 SHA-256 是否为 E99CD7904CAC2A315E3F7686D3B81E2DAAA2BE54FB03C8DFE1B52879F48C06C7；不一致则停止。
2. 解压到新的、非覆盖的目录，确认存在 datasets/racket_pose_v1/racket-pose.yaml、weights/yolo11n-pose.pt、deploy/REMOTE_GPU_RACKET_POSE_TRAINING.md。
3. 在远程服务器确认 nvidia-smi 可用，且 Python 中 torch.cuda.is_available() 为 True，并记录 torch、CUDA 和 ultralytics 版本。任一项失败则停止，不要训练。
4. 不得执行 pip install -r requirements.txt，因为该文件固定了 Windows CPU 版 PyTorch；保留远程服务器原有的 CUDA PyTorch。只有 ultralytics 缺失时，才说明兼容性风险并请求我确认安装方案。

在我确认 SSH/SCP 目标和安装方案后：
5. 上传 ZIP，保留上传后的 SHA-256 校验记录；远程解压到一个带日期的全新目录。
6. 在解压后的 Good-Badminton 目录设置 YOLO_CONFIG_DIR="$PWD/.yolo_config"，使用以下基线命令训练：
   yolo pose train model=weights/yolo11n-pose.pt data=datasets/racket_pose_v1/racket-pose.yaml epochs=100 imgsz=960 batch=-1 patience=30 device=0 workers=4 project=runs/racket_pose name=baseline_01
7. 训练期间保存并持续汇报命令、启动时间、GPU 型号、实际 batch、每 epoch 指标、报错和运行目录；不可把进程启动成功描述成训练完成。
8. 完成后下载 runs/racket_pose/baseline_01 的完整目录，至少包括 weights/best.pt、weights/last.pt、results.csv、args.yaml、曲线图和验证可视化结果。
9. 以表格交付：环境证据、数据集版本与切分、最终 best epoch、pose 指标、失败样本、已知限制、建议的下一轮实验。没有证据的数据必须标为“未验证”。

模型与数据边界：
- 不修改原始 JPG/JSON 标注；派生数据、训练输出和人工评审结论必须分开保存。
- 不因单次验证得分就宣称“可商用”或“可实时使用”。先检查整个保留视频、远景小球拍、遮挡、运动模糊、不同机位与双打画面。
- 不把远程访问密钥、密码、.env 文件、用户视频上传到 GitHub 或聊天记录。
```

## 3. Mac 端操作流程

### 3.1 从 Windows 拷贝并校验

通过 SMB 共享或移动硬盘，把训练 ZIP 拷贝到 Mac，例如 `~/Downloads/`。Mac 终端中执行：

```bash
shasum -a 256 ~/Downloads/good-badminton-racket-pose-training-bundle-20260831.zip
```

输出必须与本任务书的 SHA-256 完全相同。不相同就重新拷贝，不能继续解压或上传。

### 3.2 解压到独立目录

```bash
mkdir -p ~/Documents/good-badminton-training/baseline_01
ditto -x -k ~/Downloads/good-badminton-racket-pose-training-bundle-20260831.zip \
  ~/Documents/good-badminton-training/baseline_01
cd ~/Documents/good-badminton-training/baseline_01/Good-Badminton
ls datasets/racket_pose_v1/racket-pose.yaml weights/yolo11n-pose.pt
```

解压目录应当只用于本次基线。下一次实验使用新的目录或新的 `name`，不得覆盖 `baseline_01`。

### 3.3 准备远程 GPU 前的必要条件

在开始前，使用者需向 Mac ChatGPT 提供：

- GPU 服务器的主机名/IP、SSH 用户名、端口（若不是 22）和远程工作目录；
- 已经能登录的 SSH 密钥或人工登录方式；
- 是否允许在服务器安装缺失的 Ultralytics；
- 是否允许使用该服务器处理训练集中出现的比赛画面。

不要把密码、私钥内容或 `.env` 文件粘贴到 ChatGPT。若服务器没有 CUDA GPU，改换服务器，而不是退回 Mac CPU/MPS 进行正式训练。

### 3.4 上传与服务器预检

将下列 `[USER]`、`[HOST]`、`[REMOTE_DIR]` 替换为真实值后执行：

```bash
scp ~/Downloads/good-badminton-racket-pose-training-bundle-20260831.zip \
  [USER]@[HOST]:[REMOTE_DIR]/
ssh [USER]@[HOST]
```

服务器上先执行：

```bash
cd [REMOTE_DIR]
sha256sum good-badminton-racket-pose-training-bundle-20260831.zip
mkdir -p baseline_01
unzip good-badminton-racket-pose-training-bundle-20260831.zip -d baseline_01
cd baseline_01/Good-Badminton
nvidia-smi
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python3 -c "import ultralytics; print(ultralytics.__version__)"
```

要求：`torch.cuda.is_available()` 必须输出 `True`。若不是，停止并保存诊断输出。

### 3.5 启动训练与监控

服务器预检通过后，执行：

```bash
export YOLO_CONFIG_DIR="$PWD/.yolo_config"
mkdir -p "$YOLO_CONFIG_DIR"
yolo pose train \
  model=weights/yolo11n-pose.pt \
  data=datasets/racket_pose_v1/racket-pose.yaml \
  epochs=100 imgsz=960 batch=-1 patience=30 \
  device=0 workers=4 \
  project=runs/racket_pose name=baseline_01
```

训练过程中至少保留：

- `nvidia-smi` 的 GPU 型号和显存证据；
- 训练启动命令和时间；
- `runs/racket_pose/baseline_01/results.csv`；
- 实际批大小、停止 epoch、异常信息；
- `weights/best.pt` 与 `weights/last.pt`。

如果显存不足，优先按顺序尝试：减小 `imgsz`，再固定更小的 `batch`；每项变更必须新建实验名，例如 `baseline_01_imgsz768`，并记录原因。不要静默改动数据切分、关键点顺序或基础权重。

### 3.6 下载与验收

训练结束后，从 Mac 下载完整运行目录：

```bash
mkdir -p ~/Documents/good-badminton-training/results
scp -r [USER]@[HOST]:[REMOTE_DIR]/baseline_01/Good-Badminton/runs/racket_pose/baseline_01 \
  ~/Documents/good-badminton-training/results/
```

基线训练完成的最低验收条件：

1. 有 CUDA 可用、GPU 型号、Torch/CUDA/Ultralytics 版本的记录。
2. 有完整且未改动的 `racket_pose_v1` 数据集切分证据。
3. 有可复现训练命令、`args.yaml`、`results.csv`、`best.pt` 和 `last.pt`。
4. 有验证集定量结果与验证可视化；不能只报告“训练没有报错”。
5. 对远景小球拍、遮挡、运动模糊、不同比赛来源的失败样本单独复核。

本次产物只能称为“基线模型”。是否替换现有球拍检测能力，需要在相同固定机位视频上与原方案做召回、误检、最久中断和运行时对比后再决定。

## 4. 故障处理边界

| 情况 | 应做什么 | 不应做什么 |
| --- | --- | --- |
| ZIP 校验失败 | 重新从 Windows 复制 | 带着损坏包继续训练 |
| `nvidia-smi` 不存在或 CUDA 为 `False` | 停止，换 CUDA 服务器 | 在 Mac/CPU 上假装完成正式训练 |
| 服务器没有 Ultralytics | 先报告当前 Torch/CUDA，取得确认后安装兼容版本 | 直接安装 Windows `requirements.txt` |
| 显存不足 | 创建新实验，记录减小图像尺寸或 batch 的变更 | 覆盖原实验或改数据集切分 |
| 指标低或验证图明显错误 | 导出失败样本，检查标注质量、远景比例和场景覆盖 | 宣称模型已可商用 |
| SSH/SCP 无法连接 | 报告错误和目标信息缺失 | 编造远程路径、账号或上传已成功 |
