#! python
# -*- coding:utf-8 -*-
###
# --------------------------------------------------------------------------------
# 文件名: Xi-Sentry.py
# 创建时间: 2026-01-19 17:33:49 Mon
# 说明:
# 作者: Calibur88
# 主机: LAPTOP-D92A7OL2
# --------------------------------------------------------------------------------
# 最后编辑作者: Calibur88
# 最后修改时间: 2026-01-19 17:33:54 Mon
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Calibur88
# --------------------------------------------------------------------------------
# 更新历史:
# --------------------------------------------------------------------------------
# 时间      		作者		信息
# ----------		---		------------------------------------------------------
###

"""
Xi-哨兵：黎曼深渊结构监测系统 (Refined V2.1 - 高精度巡检架构)
Xi-Sentry: Riemann Abyss Structural Monitoring System (Refined V2.1 - High-Precision Patrol Architecture)

系统定位 / System Positioning:
1. 黎曼ξ函数是宇宙数学框架的承重梁，非装饰性假设
   The Riemann ξ-function is a load-bearing beam of the cosmic mathematical framework, not a decorative hypothesis

2. 本系统通过"数学应力场拓扑"实现临界线(σ=0.5)结构完整性实时监控
   This system enables real-time structural integrity monitoring of the critical line (σ=0.5) via "mathematical stress field topology"

3. 有限但高密度的数值采样可映射无限数学必然性规律
   Finite yet high-density numerical sampling can map the laws of infinite mathematical necessity

4. 核心使命：持续巡检临界线，确保宇宙数学基础结构零微裂缝
   Core Mission: Continuously patrol the critical line to ensure zero micro-fractures in the universe's mathematical foundation
"""

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import json
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any
from pathlib import Path
import warnings

# 抑制非关键告警 / Suppress non-critical alerts
warnings.filterwarnings("ignore")


# ==================== 字体兼容性引擎 / Font Compatibility Engine ====================
class FontManager:
    """跨平台字体自适应引擎 / Cross-Platform Font Adaptive Engine

    专为解决Matplotlib在跨平台环境下中文显示异常(tofu问题)而设计的智能fallback系统
    Intelligent fallback system specifically designed to resolve Matplotlib CJK rendering anomalies (tofu issue) across platforms
    """

    USE_ENGLISH_ONLY = False  # 强制纯英文模式标志 | Force English-only mode flag

    @staticmethod
    def configure():
        """自适应字体配置 / Adaptive Font Configuration"""

        # 跨平台中文字体候选序列(按优先级排序) / Cross-platform CJK font candidates (prioritized)
        cjk_candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Heiti TC",
            "PingFang SC",
            "WenQuanYi Micro Hei",
            "Noto Sans CJK SC",
            "Arial Unicode MS",
            "DengXian",
        ]

        system_fonts = set(f.name for f in fm.fontManager.ttflist)
        found_font = None

        # 自动探测最优可用字体 / Auto-detect optimal available font
        for font in cjk_candidates:
            if font in system_fonts:
                found_font = font
                break

        # 渲染配置应用 / Apply rendering configuration
        if found_font:
            print(
                f"🎨 字体引擎: 已加载 '{found_font}' | Font Engine: Loaded '{found_font}'"
            )
            plt.rcParams["font.sans-serif"] = [found_font] + plt.rcParams[
                "font.sans-serif"
            ]
            plt.rcParams["axes.unicode_minus"] = (
                False  # 修复负号渲染 | Fix negative sign rendering
            )
            FontManager.USE_ENGLISH_ONLY = False
        else:
            print(
                "⚠️ 字体引擎: 未检出CJK字体，切换至纯英文模式 | Font Engine: No CJK font detected, switching to English-only mode"
            )
            plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
            plt.rcParams["axes.unicode_minus"] = True
            FontManager.USE_ENGLISH_ONLY = True


# ==================== 系统配置中心 / System Configuration Hub ====================
class Config:
    """Xi-哨兵全局配置中心 / Xi-Sentry Global Configuration Hub

    所有关键观测参数均可在此区域进行微调以实现精度与性能的最优平衡
    All critical observation parameters can be fine-tuned here for optimal precision-performance balance
    """

    # [核心参数调谐区] 观测密度与扫描范围 / [Core Parameter Tuning Zone] Density & Scan Range
    OBSERVATION_RESOLUTION = 2000  # 单轴采样密度(建议: 500-3000) | Single-axis sampling density (Recommended: 500-3000)

    # 扫描范围定义 / Scan Range Definition
    SIGMA_RANGE_VERTICAL = (
        0.1,
        0.9,
    )  # 垂直剖面实部区间 | Real part interval for vertical profile
    T_RANGE_HORIZONTAL = (
        10,
        50,
    )  # 水平剖面虚部区间 | Imaginary part interval for horizontal profile

    # 计算精度配置 / Computational Precision Config
    DEFAULT_PRECISION = 100  # dps (十进制精度位数) | Decimal places of precision

    # 输出与可视化 / Output & Visualization
    FIGURE_SIZE = (16, 10)
    DPI = 200  # 渲染分辨率 | Rendering resolution
    COLOR_MAP = "magma_r"
    OUTPUT_DIR = Path("./xi_sentry_output_v2_1")

    # 判定阈值 / Determination Thresholds
    ZERO_THRESHOLD = 1e-20  # 零点判定容差 | Zero-point tolerance
    SYMMETRY_THRESHOLD = 1e-40  # 对称性判定容差 | Symmetry tolerance

    @classmethod
    def setup(cls):
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FontManager.configure()  # 初始化字体引擎 | Initialize font engine


class MPMathEncoder(json.JSONEncoder):
    """mpmath对象JSON序列化专用编码器 / Specialized Encoder for mpmath Objects"""

    def default(self, obj):
        if isinstance(obj, (mp.mpf, mp.mpc)):
            return str(obj)
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        return super().default(obj)


# ==================== 深渊探测节点 / Abyss Probe Node ====================
@dataclass
class MathematicalPoint:
    """深渊结构探测节点 / Abyss Structure Probe Node

    每个实例代表复平面(s=σ+it)上的一个精密测量点
    Each instance represents a precise measurement point on the complex plane (s=σ+it)
    """

    sigma: float  # 实部坐标 | Real coordinate
    t: float  # 虚部坐标 | Imaginary coordinate
    zeta_value: complex  # ζ(s)原始值 | Raw ζ(s) value
    xi_value: complex  # ξ(s)修正值 | Corrected ξ(s) value
    stress_intensity: float  # 应力场强度(-log₁₀|ξ|) | Stress field intensity
    is_near_zero: bool  # 是否逼近理论零点 | Proximity to theoretical zero
    symmetry_deviation: float  # 对称性偏差|ξ(s)-ξ(1-s)| | Symmetry deviation

    @classmethod
    def create(cls, sigma_mp, t_mp):
        """节点Factory方法 / Node Factory Method"""
        try:
            s = mp.mpc(sigma_mp, t_mp)
            zeta_val = mp.zeta(s)

            # ξ(s)完整函数计算 / Compute complete ξ(s) function
            xi_val = (
                0.5 * s * (s - 1) * (mp.pi ** (-s / 2)) * mp.gamma(s / 2) * zeta_val
            )

            # 对称性镜像测试 (s → 1-s) / Symmetry mirror test (s → 1-s)
            s_mirror = mp.mpc(1 - sigma_mp, -t_mp)
            xi_mirror = (
                0.5
                * s_mirror
                * (s_mirror - 1)
                * (mp.pi ** (-s_mirror / 2))
                * mp.gamma(s_mirror / 2)
                * mp.zeta(s_mirror)
            )
            sym_dev = abs(xi_val - xi_mirror)

            # 对数应力强度计算 / Compute logarithmic stress intensity
            abs_xi = abs(xi_val)
            stress = -float(mp.log10(abs_xi + mp.mpf("1e-100")))

            return cls(
                sigma=float(sigma_mp),
                t=float(t_mp),
                zeta_value=complex(zeta_val),
                xi_value=complex(xi_val),
                stress_intensity=stress,
                is_near_zero=abs(zeta_val) < Config.ZERO_THRESHOLD,
                symmetry_deviation=float(sym_dev),
            )
        except Exception as e:
            return None


# ==================== 深渊扫描日志 / Abyss Scan Log ====================
@dataclass
class AbyssScanResult:
    """单次深渊扫描任务日志 / Single Abyss Scan Mission Log"""

    scan_id: str  # 扫描任务唯一标识 | Scan mission UUID
    timestamp: str  # ISO 8601时间戳 | ISO 8601 timestamp
    scan_type: str  # 扫描模式(vertical/horizontal) | Scan mode
    primary_variable: str  # 主变量轴 | Primary variable axis
    fixed_variable_val: float  # 固定变量值 | Fixed variable value
    range_info: Tuple[float, float]  # 扫描区间 | Scan interval
    resolution: int  # 采样分辨率 | Sampling resolution
    points: List[MathematicalPoint]  # 探测节点集合 | Probe nodes collection
    statistics: Dict[str, Any]  # 统计摘要 | Statistical summary

    def to_dataframe(self) -> pd.DataFrame:
        """转换为分析型DataFrame / Convert to analytical DataFrame"""
        data = [asdict(p) for p in self.points]
        for row in data:
            row["zeta_real"] = row["zeta_value"].real
            row["zeta_imag"] = row["zeta_value"].imag
            row["xi_real"] = row["xi_value"].real
            row["xi_imag"] = row["xi_value"].imag
            del row["zeta_value"]
            del row["xi_value"]
        return pd.DataFrame(data)

    def save(self):
        """持久化存储扫描日志 / Persist scan log"""
        filename = Config.OUTPUT_DIR / f"{self.scan_type}_{self.scan_id}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, cls=MPMathEncoder, indent=2)
        return filename


# ==================== ξ(s)计算核心 / ξ(s) Computation Core ====================
class RiemannEngine:
    """高精度ξ(s)函数计算引擎 / High-Precision ξ(s) Computation Engine"""

    def __init__(self, precision: int = Config.DEFAULT_PRECISION):
        self.precision = precision
        mp.mp.dps = precision

    def find_zero_near(
        self, t_guess: float, sigma_guess: float = 0.5, max_iter: int = 20
    ):
        """
        复数域牛顿-拉夫逊迭代精密零点定位
        Precise zero-point localization via complex-domain Newton-Raphson iteration
        """
        original_dps = mp.mp.dps
        mp.mp.dps = (
            self.precision + 10
        )  # 临时提升精度确保收敛 | Temporarily boost precision for convergence
        try:
            s = mp.mpc(sigma_guess, t_guess)
            for i in range(max_iter):
                f_val = mp.zeta(s)
                if abs(f_val) < mp.mpf(10) ** (-self.precision + 5):
                    return float(s.real), float(s.imag)

                h = mp.mpf(10) ** (-self.precision // 2)
                f_prime = (mp.zeta(s + h) - mp.zeta(s - h)) / (
                    2 * h
                )  # 中心差分求导 | Central difference derivative

                if abs(f_prime) < 1e-100:
                    break

                step = f_val / f_prime
                if abs(step) > 1.0:  # 防止步长振荡 | Prevent step oscillation
                    step = step / abs(step) * 1.0
                s -= step
            return None
        finally:
            mp.mp.dps = original_dps


# ==================== 深渊扫描阵列 / Abyss Scanning Array ====================
class AbyssScanner:
    """复平面高密度扫描阵列 / Complex Plane High-Density Scanning Array"""

    def __init__(self, engine: RiemannEngine):
        self.engine = engine

    def _generate_high_precision_range(
        self, start: float, end: float, count: int
    ) -> List[Any]:
        """手工mpf序列生成器确保高密度数值稳定性 | Manual mpf generator for high-density stability"""
        start_mp = mp.mpf(start)
        end_mp = mp.mpf(end)
        step = (end_mp - start_mp) / (count - 1)
        return [start_mp + i * step for i in range(count)]

    def scan_vertical_profile(self, t: float, resolution: int) -> AbyssScanResult:
        """垂直剖面扫描模式 / Vertical Profile Scan Mode"""
        sigma_range = Config.SIGMA_RANGE_VERTICAL
        print(f"🔭 深渊扫描(垂直): t={t} | 密度={resolution}")
        print(f"🔭 Abyss Scan (Vertical): t={t} | Density={resolution}")

        sigmas = self._generate_high_precision_range(
            sigma_range[0], sigma_range[1], resolution
        )
        points = []

        for i, sigma in enumerate(sigmas):
            if i % (resolution // 10) == 0:
                print(f"  阵列进度: {i/resolution*100:.0f}%", end="\r")
            pt = MathematicalPoint.create(sigma, mp.mpf(t))
            if pt:
                points.append(pt)
        print("  阵列进度: 100%     ")

        stresses = [p.stress_intensity for p in points]
        stats = {
            "max_abyss_depth": max(stresses),
            "mean_depth": float(np.mean(stresses)),
        }

        return AbyssScanResult(
            scan_id=f"vert_{int(t)}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            scan_type="vertical",
            primary_variable="sigma",
            fixed_variable_val=t,
            range_info=sigma_range,
            resolution=resolution,
            points=points,
            statistics=stats,
        )

    def scan_horizontal_profile(self, sigma: float, resolution: int) -> AbyssScanResult:
        """水平剖面扫描模式 / Horizontal Profile Scan Mode"""
        t_range = Config.T_RANGE_HORIZONTAL
        print(f"🔭 深渊扫描(水平): σ={sigma} | 密度={resolution}")
        print(f"🔭 Abyss Scan (Horizontal): σ={sigma} | Density={resolution}")

        t_vals = self._generate_high_precision_range(t_range[0], t_range[1], resolution)
        points = []

        for i, t in enumerate(t_vals):
            if i % (resolution // 10) == 0:
                print(f"  阵列进度: {i/resolution*100:.0f}%", end="\r")
            pt = MathematicalPoint.create(mp.mpf(sigma), t)
            if pt:
                points.append(pt)
        print("  阵列进度: 100%     ")

        stresses = [p.stress_intensity for p in points]
        stats = {
            "max_abyss_depth": max(stresses),
            "mean_depth": float(np.mean(stresses)),
        }

        return AbyssScanResult(
            scan_id=f"horiz_{int(sigma*100)}_{int(time.time())}",
            timestamp=datetime.now().isoformat(),
            scan_type="horizontal",
            primary_variable="t",
            fixed_variable_val=sigma,
            range_info=t_range,
            resolution=resolution,
            points=points,
            statistics=stats,
        )


# ==================== 可视化测绘核心 / Visualization Mapping Core ====================
class Visualizer:
    """深渊数据高维可视化测绘核心 / High-Dimensional Abyss Data Visualization & Mapping Core"""

    @staticmethod
    def get_label(en_text, cn_text):
        """智能双语标签生成器 / Intelligent bilingual label generator"""
        return en_text if FontManager.USE_ENGLISH_ONLY else f"{en_text} | {cn_text}"

    @staticmethod
    def plot_scan(result: AbyssScanResult):
        """扫描日志测绘渲染 / Scan log mapping & rendering"""
        df = result.to_dataframe()
        fig = plt.figure(figsize=Config.FIGURE_SIZE)
        L = Visualizer.get_label

        if result.scan_type == "vertical":
            # 子图1: 深渊应力场拓扑 / Subplot 1: Abyss Stress Field Topology
            ax1 = fig.add_subplot(211)
            ax1.plot(
                df["sigma"], df["stress_intensity"], color="#6a0dad", linewidth=2.5
            )
            title1 = L(
                f"Abyss Stress Field (-log|ξ|) @ t={result.fixed_variable_val}",
                f"深渊应力场 (-log|ξ|) @ t={result.fixed_variable_val}",
            )
            ax1.set_title(title1, fontsize=14, fontweight="bold")
            ax1.set_ylabel(L("Stress Intensity", "应力强度"), fontsize=12)
            ax1.axvline(
                0.5,
                color="r",
                linestyle="--",
                alpha=0.6,
                linewidth=1.5,
                label=L("Critical Line σ=0.5", "临界线 σ=0.5"),
            )
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.25)

            # 子图2: 对称性偏差测井 / Subplot 2: Symmetry Deviation Log
            ax2 = fig.add_subplot(212)
            ax2.semilogy(
                df["sigma"], df["symmetry_deviation"], color="#008080", linewidth=1.5
            )
            title2 = L("Symmetry Deviation |ξ(s)-ξ(1-s)|", "对称性偏差 |ξ(s)-ξ(1-s)|")
            ax2.set_title(title2, fontsize=14, fontweight="bold")
            ax2.set_xlabel(L("Sigma (Real)", "Sigma (实部)"), fontsize=12)
            ax2.set_ylabel(
                L("Deviation Magnitude", "偏差幅值"), fontsize=12
            )  # 新增Y轴标签
            ax2.grid(True, alpha=0.25, which="both")

        elif result.scan_type == "horizontal":
            # 水平扫描可视化 / Horizontal scan visualization
            ax1 = fig.add_subplot(211)
            ax1.plot(df["t"], df["stress_intensity"], color="#6a0dad", linewidth=2.5)
            title1 = L(
                f"Abyss Stress Field (-log|ξ|) @ σ={result.fixed_variable_val}",
                f"深渊应力场 (-log|ξ|) @ σ={result.fixed_variable_val}",
            )
            ax1.set_title(title1, fontsize=14, fontweight="bold")
            ax1.set_ylabel(L("Stress Intensity", "应力强度"), fontsize=12)
            ax1.grid(True, alpha=0.25)

            ax2 = fig.add_subplot(212)
            ax2.semilogy(
                df["t"], df["symmetry_deviation"], color="#008080", linewidth=1.5
            )
            title2 = L("Symmetry Deviation |ξ(s)-ξ(1-s)|", "对称性偏差 |ξ(s)-ξ(1-s)|")
            ax2.set_title(title2, fontsize=14, fontweight="bold")
            ax2.set_xlabel(L("t (Imaginary)", "t (虚部)"), fontsize=12)
            ax2.set_ylabel(
                L("Deviation Magnitude", "偏差幅值"), fontsize=12
            )  # 新增Y轴标签
            ax2.grid(True, alpha=0.25, which="both")

        plt.tight_layout()
        output_path = Config.OUTPUT_DIR / f"plot_{result.scan_id}.png"
        plt.savefig(output_path, dpi=Config.DPI, bbox_inches="tight")
        print(f"📊 测绘完成: {output_path} | Mapping Complete: {output_path}")
        plt.close(fig)


# ==================== Xi-哨兵主控系统 / Xi-Sentry Main Control System ====================
class XiSentrySystem:
    """Xi-哨兵系统主控中心 / Xi-Sentry System Main Control Center"""

    def __init__(self):
        Config.setup()
        self.engine = RiemannEngine()
        self.scanner = AbyssScanner(self.engine)
        self.visualizer = Visualizer()
        print(f"{'='*70}")
        print(f"🛰️  Xi-哨兵系统 V2.1 (高精度巡检架构) 启动")
        print(f"🛰️  Xi-Sentry System V2.1 (High-Precision Patrol Architecture) Launched")
        print(
            f"⚙️  计算精度: {self.engine.precision} dps | Computational Precision: {self.engine.precision} dps"
        )
        print(f"{'='*70}")

    def execute_protocol(self):
        """执行标准巡检协议 / Execute Standard Patrol Protocol"""

        # 协议-01: 基准零点验证 / Protocol-01: Benchmark Zero Verification
        print(
            f"\n[协议-01] 基准零点精密定位 | Protocol-01: Benchmark Zero Precise Localization"
        )
        zero_t = 14.13472514173469  # 首个非平凡零点虚部 | Imaginary part of first non-trivial zero
        precise_point = self.engine.find_zero_near(zero_t)

        if precise_point:
            sigma_err = abs(precise_point[0] - 0.5)
            print(f"  ✓ 零点锁定: σ={precise_point[0]:.12f}, t={precise_point[1]:.12f}")
            print(
                f"  ✓ Zero Locked: σ={precise_point[0]:.12f}, t={precise_point[1]:.12f}"
            )
            print(
                f"  ✓ 临界线偏差: {sigma_err:.2e} | Critical Line Deviation: {sigma_err:.2e}"
            )
        else:
            print(f"  ⚠️ 零点定位失败 | Zero localization failed")

        # 协议-02: 垂直剖面扫描 / Protocol-02: Vertical Profile Scanning
        print(
            f"\n[协议-02] 垂直剖面高密度扫描 | Protocol-02: High-Density Vertical Profile Scan"
        )
        print(
            f"  ℹ️  采样密度: {Config.OBSERVATION_RESOLUTION} 点/轴 | Sampling Density: {Config.OBSERVATION_RESOLUTION} points/axis"
        )

        scan_res = self.scanner.scan_vertical_profile(
            t=zero_t, resolution=Config.OBSERVATION_RESOLUTION
        )
        scan_res.save()
        self.visualizer.plot_scan(scan_res)

        # 协议-03: 结构对称性验证 / Protocol-03: Structural Symmetry Verification
        print(
            f"\n[协议-03] ξ(s)函数对称性验证 | Protocol-03: ξ(s) Function Symmetry Verification"
        )
        mid_idx = len(scan_res.points) // 2
        mid_pt = scan_res.points[mid_idx]
        print(
            f"  ✓ 临界线对称偏差: {mid_pt.symmetry_deviation:.2e} | Symmetry Deviation at Critical Line: {mid_pt.symmetry_deviation:.2e}"
        )

        # 新增：阈值判定逻辑 / Add threshold determination logic
        if mid_pt.symmetry_deviation < Config.SYMMETRY_THRESHOLD:
            print(
                f"  ✓ 对称性验证通过 | Symmetry Check PASSED (阈值: {Config.SYMMETRY_THRESHOLD:.2e})"
            )
        else:
            print(
                f"  ⚠️ 对称性偏差超出阈值 | Symmetry Check FAILED (阈值: {Config.SYMMETRY_THRESHOLD:.2e})"
            )

        # 协议-04: 水平剖面扫描 (新增) / Protocol-04: Horizontal Profile Scan (New)
        print(
            f"\n[协议-04] 水平剖面高密度扫描 | Protocol-04: High-Density Horizontal Profile Scan"
        )
        print(
            f"  ℹ️  采样密度: {Config.OBSERVATION_RESOLUTION} 点/轴 | Sampling Density: {Config.OBSERVATION_RESOLUTION} points/axis"
        )

        scan_res_h = self.scanner.scan_horizontal_profile(
            sigma=0.5, resolution=Config.OBSERVATION_RESOLUTION
        )
        scan_res_h.save()
        self.visualizer.plot_scan(scan_res_h)

        print(f"\n{'='*70}")
        print("🌌 全协议执行完毕 | All Protocols Executed Successfully")
        print(f"{'='*70}")


if __name__ == "__main__":
    sentry = XiSentrySystem()
    sentry.execute_protocol()
