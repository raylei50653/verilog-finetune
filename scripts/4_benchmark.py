import subprocess
import re
import os
import json
import time

# Configuration
MODEL_NAME = "verilog-llama3"
OUTPUT_DIR = "benchmark_results"

# Benchmark Prompts (Levels 1-4)
BENCHMARKS = [
    # Level 1: Basic & Combinational
    {"id": "L1_01_Mux2to1", "prompt": "Write a 2-to-1 multiplexer in Verilog. Inputs: a, b, sel. Output: y. Use a continuous assignment (assign)."},
    {"id": "L1_02_Comparator", "prompt": "Write a module that compares two 4-bit numbers A and B. Output a single bit 'equal' which is high if they are the same."},
    {"id": "L1_03_PriorityEnc", "prompt": "Implement a 4-to-2 priority encoder in Verilog. If multiple bits are high, the highest index bit should take priority. Include a 'valid' output."},
    
    # Level 2: Sequential Logic
    {"id": "L2_01_BCDCounter", "prompt": "Design a 4-bit BCD (Binary Coded Decimal) counter. It should count from 0 to 9 and then wrap around to 0. Inputs: clk, reset (synchronous)."},
    {"id": "L2_02_EdgeDetect", "prompt": "Write a module that detects the rising edge of an input signal 'in'. Output a pulse 'edge_detected' for exactly one clock cycle."},
    {"id": "L2_03_UnivShiftReg", "prompt": "Create an 8-bit shift register with modes: 00: Hold, 01: Shift Right, 10: Shift Left, 11: Load Parallel Data. Use a 2-bit 'mode' input."},

    # Level 3: FSM & Arithmetic
    {"id": "L3_01_SeqDetect", "prompt": "Design a Moore-type FSM to detect the sequence '1101'. Output 'found' should be high when the sequence is detected. Use localparam for state definitions."},
    {"id": "L3_02_TrafficLight", "prompt": "Write a simplified traffic light controller for a crossroad. States: Green (10 cycles), Yellow (2 cycles), Red (10 cycles). Use a 'timer' counter internally."},
    {"id": "L3_03_SatAdder", "prompt": "Implement an 8-bit signed adder. If the result overflows, saturate it to the maximum or minimum possible value (clipping) instead of wrapping around."},

    # Level 4: Architecture
    {"id": "L4_01_DualPortRAM", "prompt": "Write a Verilog module for a dual-port RAM. Parameters: DATA_WIDTH (default 8), ADDR_WIDTH (default 4). Port A is write-only, Port B is read-only."},
    {"id": "L4_02_ClockDiv", "prompt": "Create a clock divider that takes a 50MHz input clock and produces a 1Hz output clock. Use parameters for the division ratio calculation."},
    {"id": "L4_03_UART_Tx", "prompt": "Design a module that takes an 8-bit byte and formats it into a UART frame: 1 start bit (low), 8 data bits, and 1 stop bit (high). Output the serial data bit by bit."}
]

def run_ollama(prompt):
    """Executes Ollama and returns the output."""
    try:
        cmd = ["ollama", "run", MODEL_NAME, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Ollama: {e}")
        return ""

def extract_verilog(text):
    """Extracts Verilog code block from Markdown."""
    # Pattern to find ```verilog ... ``` or just ``` ... ```
    pattern = r"```(?:verilog)?\s*(.*?)"""
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[0].strip() # Return the first code block
    
    # Fallback: if no markdown blocks, check for module/endmodule
    if "module" in text and "endmodule" in text:
        return text.strip()
        
    return ""

def basic_syntax_check(code):
    """Performs very basic checks since iverilog is missing."""
    issues = []
    if "module" not in code:
        issues.append("Missing 'module' keyword")
    if "endmodule" not in code:
        issues.append("Missing 'endmodule' keyword")
    if ";" not in code:
        issues.append("No semicolons found (likely malformed)")
    
    return "PASS" if not issues else f"FAIL: {', '.join(issues)}"

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"🚀 Starting Benchmark on model: {MODEL_NAME}")
    print(f"📂 Results will be saved to: {OUTPUT_DIR}/")
    print("-" * 60)
    
    results_summary = []

    for bench in BENCHMARKS:
        print(f"Running: {bench['id']} ... ", end="", flush=True)
        start_time = time.time()
        
        raw_output = run_ollama(bench['prompt'])
        verilog_code = extract_verilog(raw_output)
        
        elapsed = time.time() - start_time
        
        # Save raw output and code
        base_filename = f"{OUTPUT_DIR}/{bench['id']}"
        with open(f"{base_filename}.md", "w") as f:
            f.write(f"# Prompt\n{bench['prompt']}\n\n# Raw Output\n{raw_output}")
            
        status = "NO_CODE"
        if verilog_code:
            with open(f"{base_filename}.v", "w") as f:
                f.write(verilog_code)
            status = basic_syntax_check(verilog_code)
        
        print(f"[{status}] ({elapsed:.2f}s)")
        
        results_summary.append({
            "id": bench['id'],
            "status": status,
            "has_code": bool(verilog_code),
            "length": len(verilog_code) if verilog_code else 0
        })

    # Write summary
    print("-" * 60)
    print("📊 Benchmark Summary:")
    passed = sum(1 for r in results_summary if r['status'] == 'PASS')
    print(f"Total Tests: {len(BENCHMARKS)}")
    print(f"Structure Check Pass: {passed}/{len(BENCHMARKS)}")
    
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(results_summary, f, indent=2)

if __name__ == "__main__":
    main()
