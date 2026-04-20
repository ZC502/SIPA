import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from core.sipa_yumi_engine import SIPAYuMiEngine, JointSample

def run_visualization():
    # 1. 加载数据
    with open("debug_payload_seq.json", "r") as f:
        payloads = json.load(f)

    engine = SIPAYuMiEngine()
    results = []

    # 2. 模拟运行引擎并记录数据
    for i, packet in enumerate(payloads):
        # 兼容不同层级的 JSON 结构
        res = packet.get("_embedded", {}).get("resources", [{}])[0]
        
        # 提取 7 轴数据
        try:
            joints = [
                float(res["rax_1"]), float(res["rax_2"]), float(res["rax_3"]), 
                float(res["rax_4"]), float(res["rax_5"]), float(res["rax_6"]), 
                float(res.get("eax_a", 0))
            ]
        except KeyError:
            continue
        
        # 核心修复点：使用 q= 而不是 joints=
        # 同时将 list 转换为 numpy array 以匹配引擎要求
        sample = JointSample(q=np.array(joints), timestamp=i * 0.2) 
        
        result = engine.update(sample)
        
        results.append({
            "time": i * 0.2,
            "assoc": result.associator_norm if result.associator_norm is not None else 0
        })

    df = pd.DataFrame(results)

    # 3. 绘图逻辑
    plt.figure(figsize=(10, 6))
    
    # 使用对数坐标轴，突出 11,000 的峰值
    plt.semilogy(df["time"], df["assoc"] + 1e-6, color='#d62728', linewidth=2, label='NARH Associator Score')
    
    plt.title("SIPA Diagnostic: IRB 14050 Arm-Angle Discontinuity", fontsize=12)
    plt.xlabel("Time (s)")
    plt.ylabel("Algebraic Residual (Log Scale)")
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # 标注峰值
    if not df.empty:
        peak_idx = df["assoc"].idxmax()
        peak_val = df.loc[peak_idx, "assoc"]
        peak_time = df.loc[peak_idx, "time"]
        if peak_val > 100:
            plt.annotate(f'CRITICAL: {peak_val:.1f}', 
                         xy=(peak_time, peak_val), 
                         xytext=(peak_time-1, peak_val*1.5),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.savefig("diagnostic_plot.png", dpi=300)
    print("✅ Success! 'diagnostic_plot.png' has been generated.")

if __name__ == "__main__":
    run_visualization()
