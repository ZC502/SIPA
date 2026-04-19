import json

def generate_yumi_sequence():
    frames = []
    base_q = [10.0, -20.0, 30.0, 0.0, 45.0, 10.0, 5.0] # J1-J6, eax_a
    
    for i in range(20):
        # Simulate tiny smooth movements
        current_q = [q + i * 0.1 for q in base_q]
        
        # Create an "Arm-angle jump" at frame 10.
        if i == 10:
            current_q[6] += 5.0  # eax_a Jump of 5 degrees
            
        frame = {
            "_embedded": {
                "resources": [{
                    "rax_1": current_q[0], "rax_2": current_q[1], "rax_3": current_q[2],
                    "rax_4": current_q[3], "rax_5": current_q[4], "rax_6": current_q[5],
                    "eax_a": current_q[6]
                }]
            }
        }
        frames.append(frame)
    
    with open("debug_payload_seq.json", "w") as f:
        json.dump(frames, f, indent=2)
    print("Successfully generated debug_payload_seq.json with a jump at frame 10.")

if __name__ == "__main__":
    generate_yumi_sequence()
