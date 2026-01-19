# Xi-哨兵 / Xi-Sentry

高精度黎曼ξ函数数值验证与可视化系统。通过数学应力场拓扑监测临界线结构完整性。

High-precision Riemann ξ-function numerical verification and visualization system. Monitors critical line structural integrity via mathematical stress field topology.

> **AI协作声明 / AI Collaboration Statement**  
> 本项目的研究过程与成果撰写均在人工智能的协作下完成。作者作为研究指导者与整合者，负责框架构建、方向把控与最终成果的整合与发布。  
> This project was completed with AI collaboration. The author served as research director and integrator, responsible for framework construction, direction control, and final integration and publication.

---

## 声明 / Disclaimer

"本系统提供的所有数值结果均为有限采样估算，不能作为黎曼猜想 
的数学证明。对于关键性研究结论，建议使用SageMath/MPMATH官方
实现进行交叉验证。"

"All numerical results provided by this system are finite sampling estimates and cannot constitute a mathematical proof of the Riemann Hypothesis. For critical research conclusions, cross-validation using official SageMath/MPMATH implementations is recommended."

---

## 核心功能 / Core Features

### 双模式扫描 / Dual-mode scanning
- 垂直剖面(σ扫描)、水平剖面(t扫描)

### 应力场可视化 / Stress field visualization
- 公式：-log|ξ(s)| 量化数学应力强度

### 对称性验证 / Symmetry verification
- 公式：|ξ(s)-ξ(1-s)| 验证函数方程

### 零点精确定位 / Zero-point precise localization
- 方法：复数域牛顿法，精度达 1e-12

---

## 技术规格 / Technical Specifications

| 参数 / Parameter              | 默认值 / Default       |
|-------------------------------|------------------------|
| 采样密度 / Sampling density   | 2000点/轴             |
| 计算精度 / Precision          | 100 dps               |
| 对称性阈值 / Symmetry threshold | 1e-40                 |
| 垂直扫描范围 / Vertical scan range | σ: 0.1-0.9           |
| 水平扫描范围 / Horizontal scan range | t: 10-50             |

---

## 快速开始 / Quick Start

```bash
git clone https://github.com/calibur88/Xi-Sentry
cd Xi-Sentry
pip install mpmath matplotlib pandas numpy
python xi_sentry.py
```

---

## 输出文件 / Output Files

- **JSON日志**：包含完整扫描数据和统计信息
- **PNG图表**：应力场与对称性偏差可视化
- **目录**：`./xi_sentry_output_v2_1/`

---

## 配置说明 / Configuration

核心参数在 `Config` 类中调整：
- `OBSERVATION_RESOLUTION`: 采样密度
- `DEFAULT_PRECISION`: 计算精度(dps)
- `SIGMA_RANGE_VERTICAL / T_RANGE_HORIZONTAL`: 扫描范围

---

## 学术应用 / Academic Applications

1. 黎曼猜想数值验证 / Riemann Hypothesis numerical verification
2. 复分析教学演示 / Complex analysis teaching demonstration
3. 高精度计算基准测试 / High-precision computation benchmarking
4. 可重复数值实验平台 / Reproducible numerical experiment platform

---

## 许可证 / License

- **学术研究**：署名即可（作者/项目名/链接任选）
- **商业用途**：需书面授权（jiuxin303@qq.com）

---

## 联系信息 / Contact Information

- **作者 / Author**: calibur88
- **GitHub**: [https://github.com/calibur88/Xi-Sentry](https://github.com/calibur88/Xi-Sentry)
- **ORCID**: 0009-0003-6134-3736
- **邮箱 / Email**: jiuxin303@qq.com

---

## 系统状态 / System Status

- **状态**：运行正常 / Operational
- **最后更新 / Last Updated**: 2024年1月 / January 2024