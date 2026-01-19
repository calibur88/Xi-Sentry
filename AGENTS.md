# Xi-哨兵：黎曼深渊结构监测系统

## 项目概述

**Xi-哨兵** 是全球首个针对黎曼猜想临界线（σ=0.5）的高精度结构健康监测系统。该项目将复平面的零点分布视为宇宙数学基础的应力集中点，通过构建"数学应力场拓扑"，将抽象的解析函数转化为可视化的结构工程问题。

### 核心特性

- **高精度深渊扫描阵列**：垂直/水平剖面高密度采样（2000点/轴）
- **数学应力场可视化**：将复变函数值映射为直观的深渊深度指标
- **零点定位引擎**：基于复数域牛顿-拉夫逊迭代的精密零点锁定
- **双语输出支持**：自动适配中英文字体，消除跨平台显示问题
- **智能JSON序列化**：MPMathEncoder类支持mpmath对象无缝序列化
- **数据结构化管理**：MathematicalPoint和AbyssScanResult数据类封装探测结果

### 技术栈

- **Python 3.14.2**：主编程语言（已验证版本）
- **mpmath 1.3.0**：高精度数学计算（默认100+ dps精度）
- **matplotlib 3.10.8**：数据可视化与绘图
- **numpy 2.4.1**：数值计算
- **pandas 2.3.3**：数据分析与处理

## 项目结构

```
Xi-Sentry/
├── Xi-Sentry.py              # 主程序入口（Refined V2.1 - 高精度巡检架构）
├── README.md                 # 项目说明文档
├── LICENSE.md                # 通用学术开放知识产权许可协议
├── AGENTS.md                 # 本文件（系统上下文说明）
├── Xi-哨兵代码重写.docx      # 代码重构文档
├── .venv/                    # Python虚拟环境（Python 3.14.2）
│   ├── Scripts/
│   │   ├── python.exe
│   │   ├── pip.exe
│   │   └── activate.bat
│   └── Lib/site-packages/
│       ├── mpmath-1.3.0.dist-info/
│       ├── matplotlib-3.10.8.dist-info/
│       ├── numpy-2.4.1.dist-info/
│       ├── pandas-2.3.3.dist-info/
│       └── ...
└── xi_sentry_output_v2_1/    # 输出目录（V2.1版本）
    ├── vertical_vert_14_1768815062.json
    ├── horizontal_horiz_50_1768815073.json
    ├── plot_vert_14_1768815062.png
    └── plot_horiz_50_1768815073.png
```

## 快速开始

### 环境准备

1. **Python版本**：确保已安装 Python 3.14.2（项目已在该版本测试）
2. **依赖安装**：项目使用虚拟环境，依赖包已预装在 `.venv/` 中
   ```bash
   # 激活虚拟环境
   .venv\Scripts\activate
   
   # 验证依赖（可选）
   pip list | findstr -i "mpmath matplotlib numpy pandas"
   ```

### 运行系统

```bash
# 在项目根目录执行
python Xi-Sentry.py
```

### 预期输出

程序将执行以下标准巡检协议：

1. **基准零点验证**：验证首个非平凡零点（t≈14.1347）的精确位置
2. **垂直剖面扫描**：固定t=14.1347，沿σ∈[0.1,0.9]进行2000点采样
3. **结构对称性验证**：检测临界线σ=0.5处的对称性偏差（精度可达10⁻⁴⁴）
4. **水平剖面扫描**：锁定σ=0.5，沿t∈[10,50]探测零点分布

每次运行将生成：
- JSON格式的扫描数据（存储在 `xi_sentry_output_v2_1/`）
- PNG格式的可视化图表（应力场与对称性偏差）

## 系统配置

所有关键参数可在 `Xi-Sentry.py` 的 `Config` 类中调整：

```python
class Config:
    # 核心参数调谐区
    OBSERVATION_RESOLUTION = 2000          # 单轴采样密度(建议: 500-3000)
    SIGMA_RANGE_VERTICAL = (0.1, 0.9)      # 垂直剖面实部区间
    T_RANGE_HORIZONTAL = (10, 50)          # 水平剖面虚部区间
    DEFAULT_PRECISION = 100                # dps (十进制精度位数)
    
    # 输出与可视化
    FIGURE_SIZE = (16, 10)                 # 图表尺寸
    DPI = 200                              # 渲染分辨率
    COLOR_MAP = "magma_r"                  # 颜色映射方案
    OUTPUT_DIR = Path("./xi_sentry_output_v2_1")
    
    # 判定阈值
    ZERO_THRESHOLD = 1e-20                 # 零点判定容差
    SYMMETRY_THRESHOLD = 1e-40             # 对称性判定容差
```

## 开发约定

### 代码风格

- **编码**：UTF-8 with BOM
- **注释**：中英双语注释，优先使用英文
- **命名**：Python驼峰命名法（如 `XiSentrySystem`）
- **文档**：模块级docstring + 函数级说明

### 模块架构

系统采用分层架构设计：

```
FontManager              # 字体兼容性引擎（跨平台CJK支持）
Config                   # 系统配置中心（全局参数调谐）
MPMathEncoder            # mpmath对象JSON序列化器
MathematicalPoint        # 深渊探测节点（数据封装）
AbyssScanResult          # 深渊扫描结果（日志管理）
RiemannEngine            # ξ(s)计算核心（高精度封装）
AbyssScanner             # 深渊扫描阵列（采样器）
Visualizer               # 可视化测绘核心（双图输出）
XiSentrySystem           # 主控系统（协议化巡检流程）
```

### 数据结构

- **MathematicalPoint**：探测节点数据类，包含σ、t坐标、函数值、应力强度、零点状态、对称性偏差
- **AbyssScanResult**：扫描结果数据类，封装完整的扫描日志、时间戳、统计数据和DataFrame转换方法

### 输出规范

- **数据格式**：JSON（支持mpmath对象序列化）
- **图像格式**：PNG（1600×1000像素，200 DPI）
- **文件命名**：`{scan_type}_{suffix}.{ext}`
- **编码**：UTF-8（JSON），UTF-8（图表元数据）

## 许可证

本项目采用 **通用学术开放知识产权许可协议 1.0**：

- **学术研究/教育/个人**：完全开放，仅需署名
- **商业用途**：需提前联系作者获取书面授权

### 署名要求

在论文、报告或衍生代码中至少包含以下一项：

- 作者：calibur88
- 项目名称：Xi-哨兵 / Xi-Sentry
- 项目链接：https://github.com/calibur88/Xi-Sentry
- ORCID：0009-0003-6134-3736
- 联系邮箱：jiuxin303@qq.com

详见 `LICENSE.md` 文件。

## 常见问题

### Q: 如何修改扫描精度？
A: 编辑 `Xi-Sentry.py` 中的 `Config.DEFAULT_PRECISION` 值，默认为100 dps。

### Q: 如何调整扫描范围？
A: 修改 `Config.SIGMA_RANGE_VERTICAL` 和 `Config.T_RANGE_HORIZONTAL`。

### Q: 如何在Windows上激活虚拟环境？
A: 运行 `.venv\Scripts\activate.bat` 或 `.venv\Scripts\Activate.ps1`。

### Q: 生成的图表如何查看？
A: 输出文件位于 `xi_sentry_output_v2_1/` 目录，使用图片查看器打开PNG文件。

### Q: 为什么使用MPMathEncoder？
A: mpmath对象无法直接序列化为JSON，MPMathEncoder类提供特殊编码支持，确保高精度数值的完整保存。

### Q: MathematicalPoint和AbyssScanResult有什么作用？
A: MathematicalPoint封装单个探测点的全部信息，AbyssScanResult管理完整的扫描日志，支持DataFrame转换便于数据分析。

## 联系方式

- **作者**：calibur88
- **项目仓库**：https://github.com/calibur88/Xi-Sentry
- **ORCID**：0009-0003-6134-3736
- **联系邮箱**：jiuxin303@qq.com

## 系统状态

🟢 **当前系统状态**：深渊结构健康，临界线零微裂缝，宇宙数学框架稳健性确认。