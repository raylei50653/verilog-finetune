import subprocess
import os

MODEL_NAME = "verilog-llama3-q3"
OUTPUT_DIR = "benchmark_results"

GUIDED_PROMPT = """
You are an expert Verilog engineer. Write a module named 'fifo_controller_guided'.
It controls a synchronous FIFO with a depth of 16.

**Specifications:**
1. **Ports**: 
   - Input: clk, reset, wr_en, rd_en
   - Output: full, empty, wr_ptr [3:0], rd_ptr [3:0]
   - Internal Reg: count [4:0] (to track number of items)

**Logic Requirements (Implement inside always @(posedge clk)):**
1. **Reset**: Set all pointers and count to 0.
2. **Write Operation**: If (wr_en && !full), increment 'wr_ptr' by 1.
3. **Read Operation**: If (rd_en && !empty), increment 'rd_ptr' by 1.
4. **Counter Logic**:
   - If (wr_en && !full && !rd_en) -> count increases by 1.
   - If (rd_en && !empty && !wr_en) -> count decreases by 1.
   - If both read and write happen simultaneously -> count stays the same.
5. **Flags (Combinational or Registered)**: 
   - 'full' is high when count equals 16.
   - 'empty' is high when count equals 0.

**Constraint**: Do not use multiple always blocks for the same signal. Keep logic simple and correct.
"""

def run_guided_test():
    print(f"🧭 Testing GUIDED GENERATION on: {MODEL_NAME}")
    print("Prompt: FIFO Controller with explicit logic steps.")
    print("-" * 60)
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    try:
        cmd = ["ollama", "run", MODEL_NAME, GUIDED_PROMPT]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        filename = f"{OUTPUT_DIR}/Guided_FIFO.v"
        with open(filename, "w") as f:
            f.write(f"// Prompt: {GUIDED_PROMPT}\n\n")
            f.write(result.stdout)
        
        print(f"Done. Result saved to: {filename}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_guided_test()
