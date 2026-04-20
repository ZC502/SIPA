import json
import matplotlib.pyplot as plt
import pandas as pd
from core.sipa_yumi_engine import SIPAYuMiEngine, JointSample

def run_visualization():
    # 1. load data
    with open("debug_payload_seq.json", "r") as f:
        payloads = json.load(f)

    engine = SIPAYuMiEngine()
    results = []

    # 2. record data
    for i, packet in enumerate(payloads):
        res = packet["_embedded"]["resources"][0]
        # jita data
        joints = [
            res["rax_1"], res["rax_2"], res["rax_3"], 
            res["rax_4"], res["rax_5"], res["rax_6"], res["eax_a"]
        ]
        
        sample = JointSample(joints=joints, timestamp=i * 0.2) #  200ms 
        engine.update(sample)
        
        results.append({
            "frame": i,
            "time": i * 0.2,
            "assoc": engine.last_associator_val
        })

    df = pd.DataFrame(results)

    # 3. claw
    plt.figure(figsize=(12, 6))
    
    # logarithmic coordinate axis
    plt.semilogy(df["time"], df["assoc"].abs(), color='#d62728', linewidth=2, label='NARH Associator Score')
    
    # picture
    plt.title("SIPA Diagnostic: IRB 14050 Arm-Angle Discontinuity Detection", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Algebraic Residual (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    # annotation
    peak_idx = df["assoc"].idxmax()
    peak_val = df.loc[peak_idx, "assoc"]
    peak_time = df.loc[peak_idx, "time"]
    
    plt.annotate(f'CRITICAL: {peak_val:.1f}', 
                 xy=(peak_time, peak_val), 
                 xytext=(peak_time-1, peak_val*2),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1))

    plt.legend()
    
    # save picture
    output_file = "diagnostic_plot.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Visualization saved to {output_file}")

if __name__ == "__main__":
    run_visualization()
